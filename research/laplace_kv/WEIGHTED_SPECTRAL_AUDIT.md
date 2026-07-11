# Attention-weighted spectral correction audit

Status: rejected as a LaplaceKV candidate. A current-query oracle shows that
the attention objective contains useful low-rank structure, but the gain
disappears in the causal lifecycle. The oracle cannot encode a sealed tile
without retaining the original cache, and its quantized factors still have
large tail errors. No production or native-kernel work is justified by this
result.

## Construction

The experiment starts with a deterministic two-bit row-normalized residual
codec using an implicit Walsh-Hadamard transform. Let `K2` and `V2` be its
decoded tiles and let `EK=K-K2`, `EV=V-V2` be the quantization errors. It then
stores a low-rank correction to those errors. This differs from eOptShrinkQ:
the low-rank branch corrects the error left by quantization under an attention
metric instead of estimating a denoised low-rank KV signal under Frobenius
loss.

For the invalid current-query oracle, `Q` contains the seven simultaneous
Qwen2.5 query heads served by one KV head, and `A` contains their exact
attention rows. The corrections solve

```text
CK = argmin(rank(C) <= 7) ||(EK - C) Q^T||F
CV = argmin(rank(C) <= 5) ||A (EV - C)||F
```

by truncated SVD in the measured space followed by a pseudoinverse. Each
correction is factored, its left and right factors use a tile-local adaptive
four-bit Lloyd codebook, and its singular values use FP16. The ranks sum to
12, the largest total factor budget that fits the sealed D64 tile below
three bits after layout overhead.

The ordinary residual-SVD control uses the same two-bit base, ranks, factor
codec, record layout, and direct-factor attention equations. Only the
attention weighting is removed.

## Invalid oracle result

Input: `/tmp/laplace-qwen-256-all.kvt`, containing queries for every prefix
from 1 through 256. The table below selects the 336 queries at prefix 256 over
48 layer/KV-head identities. The sealed screen leaves tokens 0 through 127
exact and compresses tokens 128 through 255. The full-prefix screen compresses
both tiles.

### Sealed tile

| Method | Attention mean | P95 | Max | K-score mean |
|--------|---------------:|----:|----:|-------------:|
| K4/V2 Q8 | 24.003% | 50.615% | 77.044% | 2.580% |
| Residual SVD Q4 | 52.309% | 152.461% | 756.421% | 15.068% |
| Attention oracle Q4 | 18.950% | 65.178% | 289.326% | 4.262% |
| Attention oracle exact factors | 1.330% | 4.969% | 17.586% | 0.000% |

### Full prefix

| Method | Attention mean | P95 | Max | K-score mean |
|--------|---------------:|----:|----:|-------------:|
| K4/V2 Q8 | 41.865% | 87.971% | 137.450% | 4.755% |
| Residual SVD Q4 | 83.902% | 198.511% | 1,167.687% | 21.623% |
| Attention oracle Q4 | 25.032% | 81.749% | 235.220% | 5.477% |
| Attention oracle exact factors | 1.459% | 5.453% | 17.586% | 0.000% |

This is real headroom, but it is not a codec. Computing `CK` and `CV` requires
the exact quantization error, current queries, and exact attention weights.
Those queries do not exist when an old tile is sealed. Once the original tile
has been discarded, the encoder cannot recover `EK` or `EV`. Retaining FP16
KV until every future query arrives removes the memory saving, while
re-encoding every old tile on every decode step adds the same raw-cache
requirement plus quadratic work.

The exact-factor branch also shows that the four-bit factor representation is
the main oracle bottleneck. It is not a storage candidate: exact factors are
not counted and do not fit this record. The Q4 branch beats the K4/V2 mean but
has much worse maximum error on the sealed tile and remains far from evidence
for the model-level two-percent gate.

## Causal lifecycle test

The causal test compresses the first tile once. Its metric uses only the 896
queries and attention rows observed through prefix 128, then freezes the
correction. It evaluates all 43,008 queries at prefixes 129 through 256. This
is already more adaptation than the publication protocol permits because it
is request-, layer-, and head-specific.

| Method | Attention mean | P95 | Max | K-score mean |
|--------|---------------:|----:|----:|-------------:|
| K4/V2 Q8 | 23.154% | 79.422% | 238.679% | 4.530% |
| Causal weighted Q4 | 33.000% | 136.025% | 674.830% | 11.570% |
| Causal weighted exact factors | 26.321% | 136.196% | 600.113% | 10.643% |
| Residual SVD Q4 | 59.576% | 193.136% | 2,012.200% | 16.945% |

Even exact correction factors lose to K4/V2 after the query distribution
moves forward. Factor quantization is therefore not the cause of the causal
failure. The learned metric itself does not generalize. A leave-one-query-head
screen at prefix 256 reached the same decision, with worse tail errors than
the K4/V2 control.

## Complete storage and direct-factor feasibility

One addressable K/V tile-pair record contains:

- 4,096 bytes of two-bit K/V residual codes;
- 512 bytes of FP16 residual row norms;
- 1,152 bytes of four-bit correction-factor codes;
- 24 bytes of FP16 singular values;
- 64 bytes of adaptive factor codebooks;
- a four-byte pair header;
- padding from 5,852 to 5,888 bytes; and
- one four-byte offset.

The result is 5,892 bytes, or 2.876953 bits per scalar. The implicit
Walsh-Hadamard transform needs no matrix table. Counting the shared 16-byte
FP32 residual codebook gives 2.877116 bits per scalar over 48 tile pairs and
2.877035 over 96 tile pairs.

The mutable FP16 tail still breaks the whole-cache limit at some contexts:

| Context | Tail tokens | Effective bits/scalar |
|--------:|------------:|----------------------:|
| 4,095 | 127 | 3.284188 |
| 16,383 | 127 | 2.978743 |
| 65,535 | 127 | 2.902399 |

A direct-factor kernel is structurally possible. For a correction `U B`, key
scores add `U(Bq)` and value output adds `(w^T U)B`, so dense KV reconstruction
is unnecessary. A tile pair reads 5,892 bytes versus 32,768 bytes for FP16 and
6,788 bytes for the K4/V2 research record. It performs about 16,384 residual
and 2,304 factor multiply-accumulates per pair, plus two D64 Walsh-Hadamard
transforms per head and decode step. This is about 1.14 times the FP16
multiply-accumulate count at long context, with 5.56 times less cache traffic.
These are operation and byte counts, not a speed measurement. No native
kernel was built because the causal quality screen failed.

## Prior-art boundary

The broad space is occupied:

- [TurboQuant](https://arxiv.org/abs/2504.19874) supplies the rotated
  row-normalized low-bit residual idea.
- [eOptShrinkQ](https://arxiv.org/abs/2605.02905) combines a low-rank spectral
  branch with a TurboQuant residual and direct-factor attention.
- [OSCAR](https://arxiv.org/abs/2605.17757) uses offline attention-aware
  covariance to choose fixed rotations and clipping for INT2 KV.
- [STAR-KV](https://arxiv.org/abs/2606.08382) uses sensitivity-aware low-rank
  decomposition, adaptive rank selection, and mixed-precision quantization.
- [OjaKV](https://arxiv.org/abs/2509.21623) updates a context-dependent
  low-rank subspace during prefill and decode.
- [Quantization Dominates Rank Reduction](https://arxiv.org/abs/2604.11501)
  reports that low-rank removal is especially damaging under the softmax
  Fisher metric and that the problem grows with GQA.

Correcting quantization error under left and right attention operators is
mathematically different from eOptShrinkQ's Frobenius-optimal signal
shrinkage. That distinction is enough to run this ablation. It is not enough
for a novelty claim because covariance-aware quantization, sensitivity-aware
low rank, online subspace adaptation, and quantized low-rank residual hybrids
already exist.

## Completed follow-up: model-weight metrics

The oracle identified a narrower experiment without trace calibration. The
follow-up derived a full D by D query metric for each KV head from the matching
query-projection blocks. Under an isotropic normalized-hidden-state
assumption, the output-coordinate covariance of those `W_Q` blocks supplied
`H_K`. Matching `W_O` blocks supplied `H_V`, which approximated how value
error reaches the residual stream. It then solved

```text
min rank(CK)<=rK  trace((EK-CK) H_K (EK-CK)^T)
min rank(CV)<=rV  trace((EV-CV) H_V (EV-CV)^T)
```

with full matrix square roots, not diagonal trace statistics. This was
training-free and fixed by model weights. It was a distinct test from eOpt's
data-spectrum shrinker and OSCAR's offline activation-covariance rotation.
It was not a new method: weight-induced error metrics are a standard
optimization pattern, and STAR-KV already occupies sensitivity-aware low-rank
compression broadly. Novelty would require a measured Pareto improvement and
an ablation showing that the weight-only metric, correction-after-quantization,
and direct-factor Apple layout each matter.

The official Qwen2.5 0.5B Q4_K_M GGUF was obtained. The extractor dequantized
its actual Q5_0 query and output projections, validated their runtime layout,
and averaged the exact RoPE conjugation over the model's 32,768-position
context. The matching causal trace screen and 1,024-prediction lifecycle then
rejected the result. The exact-three-bit structured candidate changed
perplexity by +824.06%. See `WEIGHT_METRIC_EXTRACTION.md`,
`FROZEN_WEIGHT_MATH_AUDIT.md`, and `GENERALIZED_WEIGHTED_AUDIT.md`.

## Decision

Do not integrate this trace-conditioned correction. It proves that attention
weighting can expose a strong rank-12 oracle, but it fails causality,
generalization, tail quality, and the 4K fully counted storage invariant. The
completed model-weight follow-up also failed. No native kernel or production
integration is justified for either spectral branch.

Reproduce with:

The recorded environment used Python 3.9 and NumPy 2.0.2. The shared factor
quantizer fixes each SVD component sign by making the largest-absolute
left-factor coordinate non-negative before fitting its codebook.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  research/laplace_kv/prototype_weighted_spectral.py \
  /tmp/laplace-qwen-256-all.kvt
```
