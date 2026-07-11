# LaplaceKV remaining design-space audit

Date: 2026-07-12

This audit asks a narrow question: what mechanism still has a credible path to
all of the LaplaceKV gates at once?

- at most 3.0 encoded bits per K/V scalar including overhead;
- direct, fixed-work decode faster than FP16 and the former four-bit path on
  Apple Silicon at 16K and 64K;
- no more than 2% model-quality degradation;
- training-free operation without prompt, model, layer, or head calibration;
- one attention contract for MHA, GQA, MQA, and supported head dimensions;
- a contribution beyond existing rotation, normalization, low-rank, vector
  quantization, eviction, and entropy-coding methods.

No eligible new format was found. During this audit, the strongest remaining
fixed-width idea was tested with a more favorable activation-derived oracle
than its proposed weight-only implementation. It failed the trace screen. The
recommended eOptShrinkQ reproduction and direct-factor Apple benchmark were
then completed. The reproduction lost to K4/V2 on attention quality, and every
eligible direct-factor rank lost to K8/V6 speed. The frozen-model-weight screen
then failed a real lifecycle by +824.06% perplexity. The newer OCTOPUS and
HyperQuant public vector frontiers also have no reported point that passes the
three-bit storage, 2% quality, and decode-speed gates together. Ordinary
weighted SVD and these vector mechanisms are already occupied and are not
Laplace novelty claims. vLLM 0.25 adds a strong fused INT4 cache, but its
payload alone costs `4 + 32/D` bits per scalar. It also occupies broad KV
metadata-field co-packing by storing the zero point inside a float32 scale.
An exhaustive fixed-alphabet sweep and the final zero-metadata stateful screens
also failed. Their results close parameter retuning, stale scale state, a
symbol-clocked scalar gain, and the tested one-bit backward value forest.

## What the existing results establish

The fixed K4/V2 code body is the only local representation that has separately
shown both plausible model quality and enough native decode speed:

- K4/V2 with Q6 scale metadata stayed within the preliminary perplexity limit
  on Qwen2.5, Qwen3.5, and Phi-3. These runs are below the publication protocol
  length and are not quality passes.
- With decoded metadata already resident, the fixed Apple Silicon K4/V2
  kernel was 3.07x to 3.55x faster than FP16 and 1.19x to 1.29x faster than
  K8/V6 at 16K and 64K.
- The code body is exactly 3.0 bits per scalar. Any stored scale, zero point,
  index, table, tail, or alignment byte makes it fail storage.
- Hiding metadata in the code syndromes restored the exact rate but perturbed
  the codes, raised Qwen2.5 perplexity by 6.85%, and required a prepass that
  was slower than FP16.

The other broad mechanisms have direct local negative evidence:

| Mechanism | Best relevant local evidence | Consequence |
|-----------|------------------------------|-------------|
| Scalar precision allocation | A future-query three-bit oracle had 25.54% mean output error; global allocation had 35.10% | Moving ordinary scalar bits cannot close the gap |
| Fixed K3/K4 stripes | 19.96% preliminary perplexity increase | Deterministic row stripes are too crude |
| Joint K15/V3 mixed-radix | Exactly 3.0 bits but 6.38% Qwen2.5 perplexity increase | Better byte packing does not recover the missing V level |
| Split K12/V4 mixed-radix | Exactly 3.0 bits but 49.75% Qwen2.5 perplexity increase | Preserving V4 while reducing key states also fails lifecycle quality |
| Balanced stochastic V3 | Same exact K15/V3 record, but 84.07% mean error on seed 0 | Unbiased rounding increases realized attention variance |
| Same-token K-to-V residual | Prompt-fitted exact-K/R2 had 49.17% mean error | The residual is not compressible enough even under invalid calibration |
| Exhaustive scalar alphabets | The only two-trace winner, K7/V6 Q8/Q8, changed lifecycle perplexity by +107.86% | Retuning fixed K/V levels and metadata cannot close the gate |
| Previous-tile scale state | Zero-metadata K4/V2 measured 40.08% and 37.18% mean error | Stale normalization state does not track the next tile |
| Symbol-clocked scale | A current-RMS oracle still measured at least 50.36% mean error | Scalar decoder-synchronized gain is insufficient |
| Backward value forest | Exact-key R1 measured 27.70% and 24.33%; complete K4 was worse | The direct identity is exact, but the codec misses quality |
| Fixed product VQ | 2.877 bits, 40.86% mean output error | Small model-independent product tables are insufficient |
| Shared online value codebook | Exact-key VQ256 still had 30.10% mean output error | Prompt-trained value dictionaries do not meet the trace bound |
| Low-rank values | Rank-6 full-prefix oracle had 62.04% mean output error | Per-head value rank is not low enough at this rate |
| Query/output subspaces | Rank-32 correction had 47.88% mean output error | A causal subspace does not capture future attention sensitivity |
| Cross-layer prediction | Best tested predictor had 106.52% mean output error | Layer anchors do not predict the needed residual |
| Token summaries | Six representatives used 1.562 bits but had 123.71% mean output error | The trace requires nearly every token |
| Random-feature state | Rank 64 used an estimated 2.09 bits but had 133.03% mean output error | An untrained recurrent softmax approximation is not close |
| Query-aware eviction | Even an exact future-query 2.88-bit survivor oracle had 7.98% mean error | Selection alone cannot meet the rate |
| Entropy coding | Fully counted rANS reached 2.992 bits but was 35x to 40x slower than FP16 | Serial entropy state is incompatible with the CPU hot path |
| Pre-RoPE fixed VQ | Optimistic exactly-three-bit screen had 68.48% mean output error | Removing RoPE difficulty does not fix the codebook |
| Shared fixed quantizer | Exactly 3.0 bits but 101.63% mean error against K4/V2 Q8 | A metadata-free universal table does not recover the needed scales |
| Causal shared quantizer | Prompt-derived table still had 52.24% mean error against K4/V2 Q8 | Calibration is forbidden and did not rescue quality |
| Causal coordinate normalization | Strong activation-derived K4/V2 oracle had 35.12% mean error against exact on the sealed-half screen and 101.22% on the full prefix | Weight-derived implicit scales are unlikely to do better |
| Token replay regeneration | Token IDs use 0.0052 bit/scalar on Qwen2.5, but full replay costs about 11,552x normal decode work at 16K and 37,621x at 64K | It is an inactive-session policy, not a hot cache |

These failures do not prove that every possible three-bit codec fails. They do
rule out repeating the same mechanisms with small parameter changes.

## Information and kernel constraints

For one token and one head of width `D`, the cache contains `2D` scalars. A
three-bit average permits exactly `6D` encoded bits. The common fixed choices
are therefore:

| K body | V body | Mean body rate | Remaining rate |
|-------:|-------:|---------------:|---------------:|
| 4 | 2 | 3.000 | 0 |
| 3.75 | 2 | 2.875 | 0.125 |
| 3 | 2 | 2.500 | 0.500 |
| 3 | 3 | 3.000 | 0 |
| 4 | 1 | 2.500 | 0.500 |

The local quality results make the first row the only defensible fixed-body
starting point. This changes that branch of the problem from "compress K4/V2"
to "make K4/V2 require no cache-side metadata."

There is also no distribution-free quality guarantee at finite rate. A codec
with `6D` bits has `2^(6D)` cache records but the source space is continuous.
Two distinct K/V states must share a record, and an adversarial query or
downstream projection can magnify their difference. TurboQuant formalizes
near-optimal distortion under its source assumptions and reports neutrality at
3.5 bits and marginal degradation at 2.5 bits, not a guarantee for every
possible pretrained model. Its lower-bound analysis also means that another
data-oblivious Euclidean vector quantizer is unlikely to create a large
distortion improvement without using attention-specific side information.
[TurboQuant](https://arxiv.org/abs/2504.19874)

The Apple kernel adds a second constraint. The successful fixed K4/V2
benchmark has predictable packed reads and SIMD work. A candidate that needs a
tile prepass, pointer chasing, sparse reconstruction, entropy state, or a dense
basis reconstruction gives back the bandwidth win. Metadata may be processed
once when a token is inserted or folded into existing model operations, but it
cannot be scanned again for every query.

## Audited remaining mechanisms

### 1. Weight-folded implicit normalization

Status: rejected as the next candidate by an optimistic trace screen.

Dot-product attention has a larger exact change-of-basis freedom than the
orthogonal rotation tested so far. For invertible per-head matrices `A` and
`B`, using row-vector notation,

```text
K' = K A                 Q' = Q A^-T
V' = V B                 W_O' = B^-1 W_O
```

preserves both `Q K^T` and the post-attention output in exact arithmetic. The
key transform can be applied after RoPE. If `A` is diagonal with one shared
value per RoPE pair, it commutes with the pair rotation and can instead be
folded into the existing Q and K projection or QK-normalization weights. The
value transform can be folded into the V and output projections.

Choose the diagonal values deterministically from the projection weights, not
from activation samples. Under the usual isotropic input approximation, the
RMS of one projection output is the L2 norm of its effective weight column.
For each RoPE pair, normalize by the RMS of the two K-column norms. Normalize
each V coordinate by its V-column norm. Include the preceding RMSNorm gain in
the effective columns. The inverse factors go into Q and `W_O` as above.

This creates a model-weight-derived coordinate system in which one fixed,
analytic K4 codebook and one fixed, analytic V2 codebook can be used without a
per-token or per-tile scale. Tokens are quantized once on insertion and never
re-encoded when a tile fills. The cache is a fixed `6D`-bit record at every
position, including the mutable end, with no headers, indexes, or scale fields.
The hot attention traversal is the already-fast K4/V2 body and needs no
metadata prepass.

This is not established as novel by algebra alone. Equivalent scaling and
rotation are broad prior art in SmoothQuant and QuaRot, and InnerQ already
folds a prefill-derived per-channel K normalization into the query.
[SmoothQuant](https://arxiv.org/abs/2211.10438)
[QuaRot](https://arxiv.org/abs/2404.00456)
[InnerQ](https://arxiv.org/abs/2602.23200)

The narrower untested point had been the complete system:
projection-weight-derived, calibration-free pair scaling, fixed K4/V2 with no
cache metadata, single-write online storage, and direct Apple SIMD consumption.
Before touching model weights, `prototype_weight_folded.py` tested a stronger
normalizer. It used the actual first 64 activation vectors to derive one causal
scale per K RoPE pair and per V coordinate, then applied fixed analytic K4/V2
levels. These activation-derived scales have more information about the
request than projection-column norms, so they are an optimistic quality
control for the proposed weight-only normalization. They are not a formal
attention-error upper bound because they minimize coordinate scale mismatch,
not the downstream attention objective.

On the sealed-half D64 screen, K4/V2 Q8 measured 24.003% mean error against
exact attention. The causal coordinate candidate measured 35.115% against
exact and 40.828% against K4/V2 Q8. Applying the fixed Hadamard transform made
it 61.229% against exact. When every prefix token was encoded, the coordinate
candidate reached 101.222% mean error against exact while K4/V2 Q8 measured
41.865%. This is too far from the known K4/V2 point to justify a model-weight
rewrite or cached lifecycle implementation.

The result agrees with KVarN's finding that incorrect token scales drive error
accumulation during autoregressive decode. A constant weight-derived scale
cannot replace the actual token norm on this trace.
[KVarN](https://arxiv.org/abs/2606.03458)

Reproduce with:

```bash
python3 research/laplace_kv/prototype_weight_folded.py TRACE.kvt
python3 research/laplace_kv/prototype_weight_folded.py TRACE.kvt --start 0
```

### 2. RoPE-block bit allocation with amortized scales

Allocating precision by RoPE pair is much better motivated than alternating
token rows. Block-GTQ reports that label-free energy allocation reduces key
logit error in every one of its 367 layer comparisons and produces large
retrieval gains at low bit widths. This is direct evidence that the rejected
striped candidate spent its low bits along the wrong axis.
[Block-GTQ](https://arxiv.org/abs/2606.24033)

It is ranked second because it conflicts with several gates. The broad
mechanism is now published. A layer/head-specific allocation needs a mask or a
model-derived pattern, which either consumes metadata or complicates the SIMD
layout. Its reported packed K3/V3 path is 3.24x smaller than FP16, about 4.94
raw bits per scalar before any interpretation of auxiliary state, not a
demonstration of the Laplace three-bit gate. The local exactly-three-bit
pre-RoPE product screen also shows that V2 remains an independent bottleneck.

Use Block-GTQ as a comparison and as evidence for pairwise sensitivity, not as
the Laplace foundation.

### 3. Spectral shared component plus quantized residual

eOptShrinkQ reports about 2.2 bits per entry with stronger quality than
TurboQuant at 3.0 bits by removing a low-rank shared component and quantizing
the residual. It is the strongest published evidence that cross-token
structure can help below three bits.
[eOptShrinkQ](https://arxiv.org/abs/2605.02905)

This was the best unresolved experiment, but only as a reproduced baseline and
kernel-feasibility test. The broad spectral mechanism is occupied, and the
reported work does not establish a direct CPU hot-path latency win. Laplace's
exact-key value SVD, cross-layer predictors, and tile/full-prefix low-rank
screens all failed badly. Those screens discarded a low-rank approximation;
they did not reproduce eOptShrinkQ's different claim that optimal shrinkage
removes the shared component so the full-rank residual becomes easier to
quantize. The primary-source result therefore conflicts with, but is not
refuted by, the local tests.

A direct attention path can consume factors without reconstructing dense K/V.
For `K = U_K B_K + R_K`, logits are
`(U_K (B_K q)) + R_K q`. For `V = U_V B_V + R_V`, the value result is
`(w^T U_V) B_V + w^T R_V`. This changes the Apple question from dense
reconstruction to whether factor traffic, residual codes, rank metadata, and
the extra small products are cheaper than FP16 and fixed K4/V2. The complete
record still has to include every factor, residual scale, rank, offset, tail,
and alignment byte.

Recent matched-budget evidence also finds quantization consistently stronger
than rank reduction across MHA and GQA models, with damage caused by removing
directions that can change softmax routing.
[Quantization Dominates Rank Reduction](https://arxiv.org/abs/2604.11501)

Reproducing eOptShrinkQ does not make the codec a Laplace contribution. Novelty
would require a materially different weighted shrinkage objective, an
attention-direct factor layout, or another mechanism with a measured Pareto
gain that the original method does not provide.

The completed D64 reproduction measured 39.886% mean sealed-tile attention
error with Q4 factors, versus 38.646% for ordinary rank-matched SVD and 24.003%
for K4/V2 Q8. Its optimistic exact-factor bound was still 36.195%. The separate
native direct-factor sweep lost to K8/V6 at every eligible rank at 16K and 64K.
See `EOPTSHRINKQ_AUDIT.md` and `EOPT_DIRECT_KERNEL_AUDIT.md`.

### 4. Shared or amortized quantizers

Sharing one scale field or codebook across many tokens makes its overhead tend
toward zero, but never makes a K4/V2 body plus positive metadata at most three
bits. The body must still lose bits, or the scale must be derived rather than
stored. K3/V2 and K3/K4 local screens show the quality cost of the first option.

The literature also occupies the strongest forms:

- CommVQ uses EM-trained additive codebooks that commute with RoPE. Its learned
  encoder and codebook violate the no-training gate.
  [CommVQ](https://arxiv.org/abs/2506.18879)
- Lexico uses sparse coding over a roughly 4K-atom universal dictionary. It
  reports 90% to 95% retained task performance at 15% to 25% of full cache,
  which is outside the 2% gate, and sparse reconstruction is a poor match for
  the direct Apple loop.
  [Lexico](https://arxiv.org/abs/2412.08890)
- KVTC uses PCA, adaptive quantization, entropy coding, and initial
  calibration. Those are already excluded individually by the protocol or
  local native result.
  [KVTC](https://arxiv.org/abs/2511.01815)
- FibQuant replaces scalar product tables with a universal radial-angular
  vector code matched to the spherical source after normalization and
  rotation. It is fixed-rate, random-access, and calibration-free, so it is a
  stronger external baseline than Laplace's simple two-dimensional Gaussian
  product prototype. Its reported near-FP16 TinyLlama point is at 4x
  compression, or four bits per scalar. It does not establish the Laplace
  three-bit, cross-model, or Apple-speed gates.
  [FibQuant](https://arxiv.org/abs/2605.11478)
- OCTOPUS jointly quantizes rotated triplets through an octahedral directional
  map and norm quantizer, with a fused Triton decode. This occupies another
  high-quality fixed-rate vector geometry. Its data-oblivious construction is
  relevant as a baseline, but reproducing it is not a new Laplace mechanism.
  [OCTOPUS](https://arxiv.org/abs/2605.21226)
- OSCAR reports accurate, fast INT2 attention by deriving rotations and
  clipping thresholds from offline attention-aware covariance. Its offline
  model statistics violate the no-calibration gate even before an Apple port.
  [OSCAR](https://arxiv.org/abs/2605.17757)

KIVI's per-channel K and per-token V result and KVarN's token-scale result also
show why one quantizer shared indiscriminately across both axes is unlikely to
be universal.
[KIVI](https://arxiv.org/abs/2402.02750)

### 5. Same-token K-to-V prediction

K and V share one source activation, so a fixed `D x D` map can predict one
from the other. A K4 plus one-bit value-residual record with two FP16 scales is
2.75 bits per scalar at D64 and no more than three bits for every supported
head width. Direct attention can compute `(w^T K) A + w^T R` without dense V
reconstruction.

The optimistic causal screen fitted a full ridge map on the first 128 prompt
tokens, froze it, and evaluated the next 128. This already violates the
no-calibration gate. Exact K with a two-bit residual measured 49.173% mean
error on the official trace, and the encodable K4/R1 form measured 141.641%.
This does not prove that every frozen-weight map is worse, but it provides no
evidence that the residual can meet the rate. A model rewrite is not
justified. See `KV_PREDICTIVE_AUDIT.md`.

### 6. Other exact attention invariances

Several identities are useful but do not create enough independent storage:

- `K -> K + 1 c^T` leaves softmax weights unchanged. This saves at most one
  `D`-vector per whole cache if the common translation is not stored. The
  local affine candidate already used the missing key zero and still had
  144.40% mean output error.
- `V -> V + 1 b^T` shifts the head output by `b`. The vector must be retained
  or folded into a later operation, so it removes no asymptotic information.
- Pre-RoPE K storage avoids positional rotation damage but adds a
  position-dependent rotation to decode. KVQuant already introduced it, and
  the local fixed-rate screen failed.
  [KVQuant](https://arxiv.org/abs/2401.18079)
- A general value basis can be folded into `W_O`, and a general K basis can be
  paired with the inverse Q basis. Dense bases add `O(D^2)` per-token work or
  must be folded into model projections. This general identity is therefore
  useful only as part of rank 1 above.

### 7. Structured value bases

Palu decomposes projection weights and reconstructs full K/V from compressed
intermediates, reporting roughly 50% cache reduction before combinations with
quantization. Eigen Attention reports up to 40% cache reduction. Both are far
from the raw rate required here, and their reconstruction or projected-space
operators are designed around GPU matrix kernels.
[Palu](https://arxiv.org/abs/2407.21118)
[Eigen Attention](https://arxiv.org/abs/2408.05646)

Joint value bases across heads remain untested locally, but they cannot be a
universal cache format. Per-head attention weights differ, so a shared latent
must be accumulated separately for every query head. Storing the pre-projection
hidden state can halve MHA K/V storage in principle, but it increases each
head's scan dimension from `D` to the model width and is worse than ordinary
GQA/MQA storage. It fails the decode-speed constraint even before low-bit
error is considered.

### 8. Copy-on-write and workload deduplication

Exact prefix sharing should be implemented as a systems feature, but it is not
raw per-sequence compression. vLLM retains paged KV block management and
prefix sharing even though v0.25 removed its legacy PagedAttention kernel
implementation. SGLang's RadixAttention reuses common prefix trees. These
methods can make aggregate bytes per request very small when many requests
share a prefix, with exact quality.
[PagedAttention](https://arxiv.org/abs/2309.06180)
[SGLang](https://arxiv.org/abs/2312.07104)

For one unique sequence, the physical K/V record is unchanged. A zero-hit
workload receives no compression. Copy-on-write therefore cannot satisfy the
three-bit raw archive gate and cannot be reported as a LaplaceKV codec ratio.
It remains compatible with any eventual format.

Token-log regeneration is the other exact systems-level reduction. The local
audit counts only 0.0052 bit per K/V scalar for Qwen2.5 token IDs, but exact
prompt replay costs about 11,552 times normal cached-decode work at 16K and
37,621 times at 64K. A complete FP16 checkpoint constrained by the three-bit
budget still leaves about 10,126x and 34,827x the normal work. Regeneration is
reasonable for a suspended session and disqualifying for the hot decode path.

### 9. Recurrent and finite summaries

A finite recurrent state replaces exact softmax with another attention
operator unless its feature dimension grows with the represented keys.
Performer gives an unbiased random-feature approximation, but the local FP64
oracle exceeded 100% mean output error at the available state rate.
[Performer](https://arxiv.org/abs/2009.14794)

Infini-attention combines local attention with a compressive linear-attention
memory inside a model trained for that architecture. It is not a post-hoc
cache codec for an unchanged softmax model.
[Infini-attention](https://arxiv.org/abs/2404.07143)

The associative-recall analysis in Zoology gives a more fundamental warning:
for its multi-query recall setting, low-dimensional gated recurrence requires
memory growing with the number of key/value pairs. A universal constant-size
summary cannot retain arbitrary exact retrieval.
[Zoology](https://arxiv.org/abs/2312.04927)

The current KV-CAT paper makes the boundary explicit. It proves that models
for the same sequence function can learn either compressible or inherently
non-compressible KV representations, then improves compression by continued
training. This is evidence against a universal post-hoc recurrent fix under a
strict no-training rule.
[KV-CAT](https://arxiv.org/abs/2605.05971)

## Ranking

| Rank | Mechanism | Storage path | Apple decode path | Quality chance | Novelty room | Decision |
|-----:|-----------|--------------|-------------------|----------------|--------------|----------|
| 1 | Spectral shared component plus residual | Fully counted long-context tiles can fit | Every eligible direct-factor rank lost to K8/V6 | Local reproduction lost to K4/V2 | Generic codec and weighted objectives occupied | Rejected; retain as baseline |
| 2 | RoPE-pair allocation plus compact norm | Can fit with dimension-dependent layouts | Fixed but irregular | Moderate external evidence | Mechanism occupied | Baseline or fallback only |
| 3 | Weight-folded implicit normalization plus fixed K4/V2 | Exact 3.0 by construction | Reuses fast fixed body | Optimistic local screen failed | Narrow system-level room | Reject before model integration |
| 4 | Shared codebook or scale | Requires body reduction or derived scale | Table lookup can be fixed | Poor local evidence | Crowded | Do not prioritize |
| 5 | Same-token K-to-V residual | At most 3 bits with K4/R1 | K reread plus one D-squared product | Prompt-fitted oracle failed | Broad prediction is occupied | Reject before weight derivation |
| 6 | Structured value/weight basis | Accounting unresolved at 3 bits | Reconstruction or wider scan | Poor local evidence | Crowded | Reject as foundation |
| 7 | Exact prefix COW | Excellent only with sharing hits | Neutral or beneficial | Exact | Mature prior art | Separate feature, not codec |
| 8 | Recurrent summary | Asymptotically small | Fast if model accepts it | Failed locally, training-dependent | Mature architecture area | Reject for unchanged models |

## Completed decisive experiment

The exact eOptShrinkQ construction was reproduced on the existing Laplace
trace and paired with both a traffic model and a native direct-factor attention
benchmark. It was not integrated into the engine.

This was the right experiment because it was the only reviewed sub-three-bit
quality point that could plausibly meet the target. The local reproduction
implemented its decisive denoise-then-quantize residual and resolved the
contradiction before more speculative code was written.

### Frozen reproduction

1. Implement the paper's block construction, separable-noise estimation,
   eOptShrink singular-value rule, automatic BBP rank selection, and exact
   residual definition. Do not substitute ordinary truncated SVD.
2. Implement the paper's exact TurboQuant residual mode and bit allocation.
   A generic scalar K4/V2 residual is not an eOptShrinkQ reproduction.
3. Count left factors, right factors, singular values, residual codes, norms,
   scale fields, ranks, offsets, alignment, mutable state, and decoder tables.
   Report mean, p95, maximum, and context-length effective bits. Reject any
   configuration whose maximum complete record exceeds 3.0 bits per scalar.
4. On the D64 trace, report K and V reconstruction error separately, score
   error and ordering, attention-mass movement, output error, and the residual
   coordinate/norm statistics claimed by the paper. Compare against ordinary
   truncated SVD plus the same residual quantizer to isolate shrinkage.
5. Repeat on D96 and D256 traces before drawing a universality conclusion.
   The method's spiked-model assumptions must be measured, not presumed.

### Direct-factor Apple feasibility

For each fully counted passing rank, model attention without reconstructing
dense K/V:

```text
scores = U_K * (B_K * q) + residual_key_dot(q)
output = (weights^T * U_V) * B_V + residual_value_sum(weights)
```

Count bytes read and operations for the factors, residual, scales, and online
softmax at 16K and 64K for D64, D96, D256, and D512. Then write one native
microbenchmark only for configurations whose traffic model is lower than both
FP16 and fixed K4/V2. Compare on the same machine against FP16, K8/V6, the
former in-repo four-bit path, and the strongest reproducible public baseline.
If the former source cannot be restored, its speed gate remains unproven.

### Decision reached

- If the exact reproduction misses either quality or complete three-bit
  storage on Laplace traces, reject spectral decomposition under this goal.
- If it passes quality and storage but the direct-factor model cannot beat
  fixed K4/V2 traffic and work, retain it only as an archival format.
- If it passes all three screens, use it as the strongest baseline. The next
  Laplace candidate must then add a distinct attention-weighted shrinker,
  fixed direct-factor layout, or other measurable contribution. Reimplementing
  eOptShrinkQ alone cannot satisfy the novelty gate.

The reproduction failed the quality comparison and the direct-factor sweep
failed the K8/V6 speed gate. The attention-conditioned follow-up found an
invalid future-query oracle but failed its causal lifecycle. The completed
frozen `W_Q`/`W_O` screen also failed: its exact-three-bit K3/K4/V2 candidate
changed Qwen2.5 perplexity by +824.06% over 1,024 predictions.

## vLLM 0.25 production INT4 baseline

vLLM 0.25 introduced fused per-token-head asymmetric INT4 KV after one
deterministic randomized Hadamard transform. Each K or V vector stores `D/2`
packed code bytes plus one float32 scale. The four-bit zero point is inserted
into the low four bits of the scale representation, so the complete payload is
`4 + 32/D` bits per scalar before page overhead. That is 4.50 bits at D64,
4.25 at D128, and 4.125 at D256. No supported head width reaches the Laplace
three-bit gate.

The Triton attention kernel follows the page block table, extracts the scale
and zero point, unpacks codes, and performs online softmax and value
accumulation without reconstructing dense K/V. That raises the production
systems baseline and directly occupies broad metadata-field co-packing. It is
not a collision with the exact rejected Laplace syndrome channel, which
modified code indices to carry complete scale metadata.

The predecessor proposal reports one Qwen3.5 GSM8K comparison. INT4 is 0.2
percentage points below FP16 on strict match and 2.8 points below it on
flexible match. Its INT2 path loses 8.0 strict-match points and was dropped.
The final release still has no complete FP16/BF16 matrix across perplexity,
retrieval, and decode performance, and the Triton implementation is not an
Apple Silicon backend. The format therefore fails Laplace's storage gate by
construction and has insufficient public evidence for the other two gates. See
`VLLM_025_AUDIT.md` for the source-level audit.

vLLM's existing TurboQuant backend also misses the combined gate. Its smallest
tagged D128 record is K3/V3 at 3.1875 bits per scalar. Automatic native-KV
protection for the first and last two dense-model layers raises a 36-layer
model to 4.611 bits per scalar overall. The official integration benchmark
reported a Qwen3-4B GSM8K drop from 0.900 to 0.720 and only 65% of baseline
decode-heavy throughput for that preset.

## Public vector frontier update

Two newer data-free vector codecs test the remaining non-product geometry
more directly than another local scalar ablation.

[OCTOPUS](https://arxiv.org/abs/2605.21226) uses rotated triplets, octahedral
direction coordinates, and quantized triplet norms. Its Qwen2.5 results do not
contain a point that passes both Laplace gates:

| Nominal K=V rate | Effective rate | WikiText-2 PPL change | C4 PPL change |
|-----------------:|---------------:|----------------------:|--------------:|
| 2 bits | 2.50 bits/scalar | +34.7% | +41.5% |
| 3 bits | 3.50 bits/scalar | +7.2% | +5.9% |
| 4 bits | 4.50 bits/scalar | +2.7% | +1.5% |

The reported language recipe also keeps a 32-token native-precision window
and the first and last transformer blocks in FP16. The paper says the tested
rotation codecs diverged beyond perplexity 1,000 without that block
protection. Its H200 fused decode at the storage-eligible two-bit point was
8.9 times slower than BF16 attention. A local reproduction cannot turn this
published point into a quality, storage, or speed pass.

[HyperQuant](https://arxiv.org/abs/2606.23406) combines randomized Hadamard
tiles, E8 lattice quantization, bit stripping, Rice coding, and optional bias
correction. It improves the same Qwen2.5 protocol but still separates the
gates:

| Target rate | Complete rate from reported KV compression | WikiText-2 PPL change | C4 PPL change |
|------------:|--------------------------------------------:|----------------------:|--------------:|
| 2 bits | 2.50 bits/scalar | +7.4% | +8.1% |
| 3 bits | 3.48 bits/scalar | +1.8% | +1.4% |

These rows include the paper's 32-token residual window. The two-bit point
passes storage but fails quality. The three-bit target passes the preliminary
quality limit but its complete cache exceeds three bits per scalar. The paper
also reports no H100 throughput win because each attention step decodes the
variable-length Rice stream over the full context. That is the same hot-path
shape rejected by Laplace's native rANS result.

The current public frontier therefore does not provide a hidden eligible
codec to port. Cross-layer integer-code sharing is another occupied mechanism,
and its published XQuant configurations use layer cutoffs and optional
task-specific settings. It can be a baseline, not an original Laplace
foundation. Without a genuinely new mechanism, the combined no-training,
universal-quality, three-bit, and faster-than-K8/V6 objective has no remaining
evidence-backed candidate in the current design space.

The July 2026 scan does not change this conclusion. GSRQ's nominal 0.375 to
2-bit rates omit its learned calibrated residual-VQ codebooks. Its fused
low-rate kernel is fast, but the accelerated point loses substantial
perplexity and LongBench quality. MosaicKV has the strongest new packed sparse
systems result, but its measured 3x FP16 reduction is about 5.33 bits per
scalar and per-model quality loss reaches 3.55%. RaBitQCache is a retrieval
index that keeps the complete FP16 cache in host storage. Full accounting is
in `JULY_2026_FRONTIER_AUDIT.md`.

## Zero-byte side-channel screen

Jointly permuting the token rows of a sealed K/V tile is an exact attention
invariance. For any permutation matrix `P`,

```text
softmax(q (P K)^T) P V = softmax(q K^T) V
```

The row order can therefore carry metadata without adding cache bytes or
changing the stored K/V pairs. Its capacity is still limited. A 128-token
tile carries `floor(log2(128!)) = 716` bits and a 256-token tile carries 1,683
bits. The D64 K4/V2 Q6 record needs 3,648 metadata bits at 128 tokens and
5,952 at 256 tokens after the six FP16 field headers are counted. Row order
alone cannot preserve the known K4/V2 quality point.

A complete row-scale construction did fit the channel at 256 tokens. It used
three-bit K and V row-scale indices, fixed Gaussian K4/V2 reconstruction
levels, and 1,552 metadata bits. Its full-prefix trace error was 103.872% mean,
292.243% p95, and 573.176% maximum. Giving the same scalar codebooks exact row
scales measured 106.758% mean, so scale-field precision was not the blocker.
The unquantized row permutation identity matched to `7.421e-16` relative
error.

A second exact channel used redundant key codewords. Twelve reconstruction
levels leave four of the 16 K4 nibbles available as aliases. Choosing an alias
can carry one bit while decoding to the same K12 value. On the 48 D64,
256-token tiles, four fixed central levels supplied at least 6,277 carriers,
enough for the 4,992-bit Q5 metadata record. The resulting K12/V4 Q5 trace
measured 57.553% mean, 103.184% p95, and 234.042% maximum, versus 56.906%,
108.906%, and 261.296% for K16/V4 Q8 on the same full-prefix screen.

The alias channel removes metadata error but cannot restore the four missing
key levels. The stronger K12/V4 Q6/Q5 lifecycle already changed Qwen2.5
perplexity by 49.75%. A distribution-independent carrier guarantee also needs
an adaptive alias map and a bootstrap field; the fixed central map is only a
local count. No alias lifecycle or kernel was added.

A narrower adaptive screen retains 13 key levels. Codewords 13 through 15
duplicate fixed levels 5 through 7 and carry the Q6 metadata bits. At D64 and
256 tokens, 44 of the 48 captured tiles had the required 5,952 carriers. The
other four use the K12 fallback. This K13/K12-V4 Q6 reconstruction measured
58.677% mean, 111.588% p95, and 228.140% maximum full-prefix error, versus
56.906%, 108.906%, and 261.296% for K16/V4 Q8.

The hybrid uses ordinary four-bit key nibbles and two-bit value codes, so its
physical body is exactly three bits per scalar. Alias extraction maps each
duplicate nibble back to its base level and does not change the reconstruction.
It still has two unresolved gates. The decoder must extract almost six
kilobits before the Q6 scale fields are known, and fixed-position trace error
has repeatedly failed to predict autoregressive quality. The next valid work
would be a capture-only lifecycle followed by an alias-extraction Apple
benchmark. No production implementation is justified before both pass.

Redundant quantization-index channels are established information-hiding
prior art. See [reversible quantization index
modulation](https://arxiv.org/abs/2305.17879). A passing KV layout would still
need a distinct attention or systems contribution for the novelty gate.

Simple lossless metadata and code paths did not provide a fast escape. Over
the 48 D64 tiles, native LZ4 kept 3 tiles within 6,144 bytes and LZFSE kept 12.
A fixed canonical Huffman code plus Q6 metadata kept 36, but the worst tile was
7,072 bytes, or 3.453 bits per scalar. The already-measured rANS path remains
the only local lossless code-stream compression that fits every captured tile
with the tested compact metadata, and its native decode is 35x to 40x slower
than FP16.

These tests close row permutation alone, fixed K12 aliases, and ordinary
native compression as standalone fixes. The adaptive K13/K12 alias layout is
only a falsification candidate. Row permutation remains a useful general
storage channel, but a future candidate must use it without a tile sort in the
decode path and without lowering the effective key alphabet.
