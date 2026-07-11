# LaplaceKV experiment log

This file records results as they are produced. Failed experiments remain in
the log. The publication decision follows `PROTOCOL.md`.

## 2026-07-11: implementation checks

- Release build: passed without compiler warnings.
- CTest: 10 of 10 targets passed.
- Scalar LaplaceKV path: 3,005 checks passed.
- AddressSanitizer and UndefinedBehaviorSanitizer: `test_laplace_kv`,
  `test_kvcache`, and `test_arch` passed.

## 2026-07-11: isolated attention benchmark

Apple Silicon, release build, one KV head, head dimension 128:

| Context | Resident | Streaming | Streaming RAM | Archive | Output error |
|---------|----------|-----------|---------------|---------|--------------|
| 512 | 0.008 ms | 0.080 ms | 0.06 MiB | 0.11 MiB | 2.98% |
| 4,096 | 0.050 ms | 0.123 ms | 0.06 MiB | 0.91 MiB | 2.76% |
| 16,384 | 0.142 ms | 0.258 ms | 0.06 MiB | 3.62 MiB | 3.19% |
| 65,536 | 0.457 ms | 1.50 ms | 0.06 MiB | 14.50 MiB | 3.34% |

The former streamed K8/V8 implementation measured 2.01 ms and 16.50 MiB at
65,536 tokens in the same benchmark. Current K8/V6 was about 25% faster and
12% smaller in that run.

## 2026-07-11: long-context retrieval

Qwen2.5 0.5B Instruct Q4_K_M, 14,474-token prompt, greedy decode:

- Expected: `COBALT-CEDAR-4826`
- Recovered: `COBALT-CEDAR-4826`
- Prefill: 69.8 tok/s
- Decode: 27.7 tok/s

This is one successful retrieval case, not a completed retrieval matrix.

## 2026-07-11: cached next-token quality

Model: Qwen2.5 0.5B Instruct Q4_K_M. Corpus: the repository README before the
current findings section was added. FP16 and LaplaceKV were evaluated in one
process with alternating execution order through one-token cached decode.

| Format | Predictions | PPL change | Top-1 agreement | Mean KL | p95 KL | Max KL |
|--------|-------------|------------|-----------------|---------|--------|--------|
| K8/V6 | 1,718 | +0.0785% | 95.809% | 0.005182 | 0.013519 | 0.698916 |
| K8/V8 | 1,718 | -0.3756% | 95.867% | 0.005212 | 0.013360 | 0.678927 |

V8 did not materially improve agreement or divergence. Value precision is not
the dominant error source on this case.

The K8/V6 perplexity result is inside the current 2% limit. Top-1 agreement is
a diagnostic under revision 3. One model and one corpus do not establish the
universal-quality claim.

## 2026-07-11: preliminary score-path ablations

These 512-prediction runs are diagnostic and used different README revisions,
so they are not direct publication comparisons:

- Disabling Walsh-Hadamard rotation: +2.54% PPL, 93.36% top-1 agreement,
  mean KL 0.0277, max KL 3.998. Rotation is necessary at head dimension 64.
- Keeping K8 storage but using an FP32 query against the K8 codes: -0.59% PPL,
  97.66% top-1 agreement, mean KL 0.00396, max KL 0.158. Query quantization is
  a material error source and is the next controlled ablation.

Both must be rerun on an immutable corpus before an implementation decision.

## 2026-07-11: controlled cross-model corpus

Corpus: `research/GEMMA4_SPEC.md`, SHA-256
`dc6a59b36368dfec102260b4d3d1a3b21fcfd6fcfe42d81cfa9c3631cf73fca7`.
FP16 and K8/V6 resident modes ran side by side through one-token cached decode
with alternating execution order.

| Model | Predictions | Head dim | PPL change | Top-1 | Mean KL | p95 KL | Max KL | Laplace/FP16 time |
|-------|-------------|----------|------------|-------|---------|--------|--------|-------------------|
| Qwen2.5 0.5B | 1,024 | 64 | -0.692% | 97.070% | 0.003241 | 0.011161 | 0.242928 | 0.64x |
| Qwen3.5 0.8B | 512 | 256 | +0.278% | 98.633% | 0.000929 | 0.003207 | 0.039436 | 0.86x |
| Phi-3 Mini | 512 | 96 | +0.824% | 85.156% | 0.074218 | 0.350786 | 2.808352 | 1.37x |

All three models stay inside the current perplexity limit. Their top-1 results
remain useful diagnostics, and Phi-3 regresses latency badly. Its
non-power-of-two head dimension bypasses rotation, making that path the
clearest current performance and divergence blocker.

## 2026-07-12: initial 2.25-bit design-target search

An opt-in capture of the Qwen2.5 0.5B candidate path recorded the original K,
V, and query vectors at prediction 256. It produced 336 layer/KV-head/query
traces at head dimension 64. Normal inference is unchanged when capture is
disabled.

Reproduce the capture with a build configured using
`-DLAPLACE_KV_CAPTURE=ON`, then set `LAPLACE_KV_CAPTURE` to the output path and
`LAPLACE_KV_CAPTURE_AT` to the prediction position. Run
`prototype_delta.py` on the resulting file. Default builds compile the hook
out completely.

The first candidate stored quantized causal deltas instead of independent KV
vectors. Its attention formulation was algebraically direct: prefix sums of
query/delta-key products produced the logits, and suffix sums of the softmax
weights multiplied the value deltas. The direct result matched attention over
explicitly reconstructed deltas to a maximum relative error of `2.41e-5`.

The codec itself failed:

| Candidate | Estimated long-context bits/scalar | Mean output error | p95 | Max |
|-----------|------------------------------------|-------------------|-----|-----|
| DPCM K2/V2 | 2.25 | 115.84% | 322.11% | 1806.12% |
| DPCM K2/V2 plus analytical score-bias correction | 2.25 | 60.34% | 144.65% | 507.53% |
| DPCM K3/V3 | 3.25 | 29.57% | 72.88% | 409.19% |
| DPCM K2/ternary-V | 2.05 | 100.14% | 280.43% | 1737.98% |

Post-RoPE states were not locally predictable enough. The exact prefix/suffix
identity remains useful, but causal DPCM is rejected as the compression
foundation.

For scale, the same harness reproduced KVarN's public 16-iteration variance
normalization and asymmetric RTN recipe on 128-token tiles. This is a local
NumPy reference over Laplace traces, not an official KVarN end-to-end result:

| K/V bits | Mean output error | p95 | Max |
|----------|-------------------|-----|-----|
| 2/2 | 82.75% | 206.78% | 1383.82% |
| 3/2 | 47.02% | 103.95% | 283.21% |
| 4/2 | 41.60% | 87.94% | 136.42% |
| 4/4 | 11.60% | 23.29% | 44.89% |
| 8/8 | 0.68% | 1.32% | 3.27% |

The large maxima include traces whose exact attention-output norm is close to
zero, but the means show substantial approximation error in every low-bit row.
The current K8/V6 control measured 3.41% mean, 7.91% p95, and 52.39% maximum
on these same traces.

The first version of this table omitted KVarN's mandatory Hadamard rotation.
The values above are the corrected independent NumPy reproduction after the
official source audit. At head dimension 64 and a 128-token tile, K4/V2 uses
3.5625 effective bits per scalar and K2/V2 uses 2.5625 once all six FP16
scale/zero-point vectors are counted.

## 2026-07-12: representation search beyond scalar quantization

Revision 3 makes model-level quality the hard 2% criterion and restores 3.0
effective bits as the hard storage limit. Trace error remains a diagnostic.
The following tests use optimistic assumptions, so large failures still reject
a mechanism before an end-to-end implementation.

An oracle allocated every available upgrade bit with knowledge of the exact
current query and output. Metadata was counted as free:

| Candidate | Target bits | Mean output error | p95 | Max |
|-----------|-------------|-------------------|-----|-----|
| Independent K2/V2 | 3.0 | 144.03% | 377.73% | 2174.12% |
| Independent K3/V3 | 3.0 | 61.07% | 195.01% | 890.92% |
| Independent K4/V2 | 3.0 | 44.31% | 98.28% | 465.77% |
| Independent K8/V2 | 3.0 | 29.41% | 47.97% | 81.58% |
| Query-aware mixed-precision oracle | 3.0 | 25.54% | 99.53% | 438.73% |

Only 16.67% of traces passed 2% even for the query-aware oracle. Independent
scalar storage and selective refinement are rejected below 3 bits. A rank-32
causal query/output subspace correction still had 47.88% mean error. A shared
256-entry four-dimensional value codebook, equivalent to two value bits before
metadata, had 30.10% mean error with exact keys.

A second oracle allocated precision globally across all 48 layer/KV-head
caches instead of imposing the same rate on each. It knew every evaluation
query, paid no metadata, and solved the discrete three-bit allocation exactly.
The result was 35.10% mean, 59.52% p95, 97.66% maximum, and zero traces below
2%. Global adaptive scalar precision is also rejected.

Exact-key value-only bounds showed that the captured values are neither sparse
nor low rank at the needed rate. Two-bit top-component sparsity had 67.42% mean
error, a two-bit rank-4 factorization per 64-token tile had 55.65%, and an
optimistic 1.88-bit rank-6 factorization over the full 256-token prefix had
62.04%. Metadata and factor rounding were free in these tests.

Query-independent representative attention was then tested on 64-token tiles.
Every stored mean and moment was rounded to FP16 and 64-byte tile alignment was
included:

| Representatives | Format | Bits/scalar | Mean | p95 | Max |
|-----------------|--------|-------------|------|-----|-----|
| 6 of 64 | K/V centroids | 1.562 | 123.71% | 437.44% | 2225.87% |
| 12 of 64 | K/V centroids | 3.062 | 111.01% | 420.95% | 2295.63% |
| 48 of 64 | K/V centroids | 12.125 | 7.78% | 32.08% | 83.27% |
| 64 of 64 | K/V centroids | 16.125 | 0.09% | 0.25% | 2.36% |

Scalar key variance, diagonal K/V covariance, and second-order correction did
not materially improve the result. The trace needs almost every token. The
broad mechanism is also occupied by CentroidKV, SemantiCache, MomentKV,
Multipole Attention, and WildCat.

A fixed positive-random-feature attention state was tested as an FP64 oracle.
At rank 64 its estimated FP16 state cost was 2.09 bits/scalar at this 256-token
context, but mean output error was 133.03%. Increasing rank did not converge
usefully. This known Performer-style route is rejected without model training.

Cross-layer predictive storage used exact anchor layers and ternary residuals.
Scalar, coordinate-wise, and full linear predictors all failed. The best tested
24-layer coordinate result had 106.52% mean output error. AQUA-KV, xKV, XQuant,
and BaseMix-KV also occupy the broad anchor, transform, and residual method.

Finally, an impossible query-aware survivor oracle retained the exact FP16
entries with the largest current attention scores:

| Retained | Storage before metadata | Mean | p95 | Max | Mean attention mass |
|----------|-------------------------|------|-----|-----|---------------------|
| 5% | 0.80 bits | 24.48% | 73.03% | 163.87% | 80.86% |
| 10% | 1.60 bits | 13.83% | 46.39% | 98.46% | 88.65% |
| 18% | 2.88 bits | 7.98% | 27.38% | 90.14% | 93.28% |
| 50% | 8.00 bits | 1.75% | 6.71% | 42.45% | 98.71% |

Pure eviction is rejected on this trace even with future-query knowledge. An
even stronger hybrid kept the best 18% and summarized all removed entries with
full, query-specific second moments. It reached 1.66% mean error, but p95 was
6.45% and maximum error was 13.57%. Those full moments are not storable inside
the remaining 0.12-bit budget. This is an upper bound, not a candidate result.

The first new rounding mechanism applied zero-overhead sigma-delta error
feedback to variance-normalized value tiles. A 256-token tile reduces scale
overhead enough for K4/V1 to fit at 2.969 bits on head dimension 64. Shaping
improved its mean error from 155.49% to 124.84%, but remained unusable. K3/V1
improved from 177.58% to 143.99% at 2.469 bits. Shaping K4/V2 slightly worsened
mean error from 44.59% to 46.88%. Marginal preservation alone cannot recover
peaked, query-dependent value aggregation.

Reproduce the fixed-position tests with:

```bash
python3 research/laplace_kv/prototype_oracle.py TRACE.kvt
python3 research/laplace_kv/prototype_allocator.py TRACE.kvt
python3 research/laplace_kv/prototype_codebook.py TRACE.kvt
python3 research/laplace_kv/prototype_coreset.py TRACE.kvt
python3 research/laplace_kv/prototype_crosslayer.py TRACE.kvt
python3 research/laplace_kv/prototype_state.py TRACE.kvt
python3 research/laplace_kv/prototype_survivor.py TRACE.kvt
python3 research/laplace_kv/prototype_hybrid.py TRACE.kvt
python3 research/laplace_kv/prototype_noiseshape.py TRACE.kvt
python3 research/laplace_kv/prototype_value_structure.py TRACE.kvt
```

`prototype_subspace.py` needs an all-position trace so its first 64 queries are
strictly earlier than the position-256 evaluation. Build with
`-DLAPLACE_KV_CAPTURE=ON`, set `LAPLACE_KV_CAPTURE` to the output file, and omit
`LAPLACE_KV_CAPTURE_AT` to record all positions. Trace files contain model
activations and are not committed.

## 2026-07-12: official KVarN simulator and bounded-codec search

The capture build now contains an opt-in FP32-cache simulator for the released
dense KVarN equations. Its deterministic C++ fixture matches the independent
Python reference, including the FP16 input boundary, store transform, eight
normalization iterations, FP16 metadata, tile lifecycle, and inverse transform.
Normal builds compile this path out. Simulator JSON explicitly marks timing and
storage as invalid.

An earlier simulator revision left the unsealed sink and mutable tile in FP32
while comparing against an FP16 cache. The following rows are retained only as
historical rejection screens. They are not results of the corrected lifecycle:

| Model or candidate | Predictions | PPL change | Top-1 | Mean KL | Result |
|--------------------|------------:|-----------:|------:|--------:|--------|
| Qwen2.5 K4/V2 FP16 metadata | 1,024 | -0.695% | 92.090% | 0.08716 | pre-fix screen |
| Qwen3.5 K4/V2 FP16 metadata | 512 | +0.337% | 96.875% | 0.03783 | pre-fix screen |
| Qwen2.5 K2/V2 | 512 | +86.012% | 76.367% | 1.01059 | pre-fix reject |
| Qwen2.5 K3/V2 G256 | 1,024 | +11.190% | 88.184% | 0.2153 | pre-fix reject |

A single K3/V2 tile had looked acceptable. The 1,024-prediction run showed
that the error accumulates. This is why trace-only and one-boundary results no
longer decide candidate quality.

In the same pre-fix screen, linear Q5 metadata changed Qwen2.5 perplexity by
+6.632%. Polarity-specific FP8 metadata increased mean trace error from
23.739% to 30.486% and was rejected. Q8 linear metadata preserved the sealed
K4/V2 result:

| Model | Predictions | Head dim | PPL change | Top-1 | Mean KL |
|-------|------------:|---------:|-----------:|------:|--------:|
| Qwen2.5 | 1,024 | 64 | -2.883% | 91.797% | 0.08348 |
| Qwen3.5 | 512 | 256 | +0.367% | 96.875% | 0.03925 |
| Phi-3 | 512 | 96 | -1.544% | 79.492% | 0.22249 |

These use the corrected FP16 sink and mutable-tile lifecycle. They are
preliminary PPL screens, not protocol quality passes, because they are below
the required 2,048 predictions and do not include the retrieval matrix.

Phi-3 uses a new even-dimension transform. For D96 it composes H32 with a
three-block Householder transform. Its float32 roundtrip error is `1.326e-7`,
its FP16-boundary roundtrip error is `2.881e-4`, and it avoids padding D96 to
D128. This validates the simulator path, not the production cache.

Actual four-lane context-rANS roundtrips over 48 captured D64/G128 tiles gave
2.859375 mean, 2.973828 p95, and 2.992188 maximum bits per scalar. The count
includes Q8 metadata, eight rANS states, frequency tables, stream offsets, one
uint64 directory offset, and 16-byte alignment. Losslessly compressed FP16
metadata missed the hard tail limit with a 3.164062 maximum.

The native fused rANS experiment is a decisive rejection of this CPU decode
design. It is a synthetic, single-thread microbenchmark over deterministic
correlated codes, not a complete model benchmark:

| Context | D | rANS | FP16 | Slowdown |
|--------:|--:|-----:|-----:|---------:|
| 16K | 64 | 8.189 ms | 0.231 ms | 35.5x |
| 16K | 96 | 12.077 ms | 0.304 ms | 39.8x |
| 64K | 64 | 32.736 ms | 0.927 ms | 35.3x |
| 64K | 96 | 48.172 ms | 1.299 ms | 37.1x |

The C++ benchmark builds real indexed records, verifies four-lane roundtrips,
and consumes decoded K and V directly without materializing a dense tile. The
serial entropy state dominates the memory saving. This rANS route is rejected
for the Apple CPU decode hot path.

Full-cache accounting also exposed the mutable-tail problem. Immediate K3/V2
tail storage stayed below 3 bits but changed Qwen2.5 perplexity by +1,852.97%.
K4/V2 per-token tail storage still changed it by +515.67%. Both reconstructed
tails were consumed by the sealed encoder, with no hidden FP16 copy. These
results reject the first two strict-budget tail candidates.

Two fixed-width follow-ups also failed. A K3/V3 per-token tail changed
Qwen2.5 perplexity by +450.65% at 512 predictions. A sealed tile with
alternating K3/K4 rows, V2 values, and Q6 metadata fit its D64 tile budget at
2.972656 bits per scalar before the directory, but changed perplexity by
+19.96% at 1,024 predictions. Scalar precision redistribution does not recover
the K4/V2 quality point.

A rotating block-floating record used exactly 3 bits per scalar, including a
four-bit exponent per 32 coordinates. It changed Qwen2.5 perplexity by
+13,063.82% at 512 predictions and is rejected. Dropping the key zero through
softmax affine invariance and using coordinate-wise V2 fit at 2.9746 bits, but
its mean trace error was 144.40%.

The fixed product-vector search used model-independent two-dimensional
codebooks, log8 row scales, and a 2.875-bit payload. Fully counted rate on the
captured trace was 2.876628 bits per scalar. Its joint attention error was
40.855% mean, 89.830% p95, and 469.121% maximum. This is worse than the
23.739% K4/V2 reference mean and does not justify model integration. The broad
method also overlaps existing KV vector quantizers.

Reproduce the new storage and native results with:

```bash
python3 research/laplace_kv/prototype_kvarn_official.py TRACE.kvt
python3 research/laplace_kv/prototype_scale_field.py TRACE.kvt
python3 research/laplace_kv/prototype_lossless_metadata.py TRACE.kvt
python3 research/laplace_kv/prototype_tail_huffman.py TRACE.kvt
python3 research/laplace_kv/prototype_fixed_product.py TRACE.kvt
python3 research/laplace_kv/prototype_affine_invariant.py TRACE.kvt
python3 research/laplace_kv/prototype_even_transform.py
clang++ -O3 -std=c++20 -mcpu=native -o /tmp/bench-rans \
  research/laplace_kv/bench_context_rans_native.cpp
```

## 2026-07-12: fixed K4/V2 kernel and syndrome-metadata rejection

Linear Q6 metadata remained inside the preliminary perplexity limit without
syndrome embedding:

| Model | Predictions | Head dim | PPL change | Top-1 | Mean KL |
|-------|------------:|---------:|-----------:|------:|--------:|
| Qwen2.5 | 1,024 | 64 | -0.421% | 90.137% | 0.16472 |
| Qwen3.5 | 512 | 256 | +0.456% | 96.289% | 0.03845 |
| Phi-3 | 512 | 96 | -1.148% | 78.906% | 0.21679 |

These remain preliminary screens. They are below the 2,048-prediction
protocol minimum and Q6 metadata stored separately pushes fixed K4/V2 above
the three-bit limit.

A direct Apple Silicon K4/V2 attention kernel showed that the fixed code body
is fast when decoded metadata is already available. Across D64 and D96 at 16K
and 64K, it was 3.07x to 3.55x faster than FP16 and 1.19x to 1.29x faster than
the current K8/V6 path. The measured FP32 metadata brings the record to 3.875
to 4.125 bits per scalar. Modeled Q8 metadata still brings it to 3.227 to
3.293 bits per scalar.

The follow-up hid all 3,648 bits of D64/G128 Q6 metadata inside the LSB
syndromes of the K4/V2 indices. This produces a fixed 6,144-byte sealed tile,
exactly 3.0 bits per scalar with no external tile metadata. It is rejected for
three independent reasons:

- Qwen2.5 at 1,024 predictions changed perplexity by +6.847%, above the 2%
  quality limit. Top-1 agreement was 98.730% and mean KL was 0.07361.
- The optimized full metadata prepass reached only 0.836x FP16 and 0.331x
  K8/V6 speed at 16K, then 0.826x and 0.324x at 64K.
- Syndrome or parity embedding in quantized indices is established prior art.
  The KV-specific application was not found, but the coding mechanism is not
  a defensible foundational novelty claim.

The temporary lifecycle simulator was removed after rejection. The standalone
prototype, Apple Silicon benchmark, full accounting, and prior-art sources are
in `SYNDROME_EMBEDDING_AUDIT.md`, `FIXED_K4V2_BENCH.md`, and
`SYNDROME_METADATA_NOVELTY.md`.

The source-linked Cyberpunk technology review is in
`CYBERPUNK_TECH_LORE.md`. Its useful result is a systems direction rather than
a codec claim: separate archival and executable forms, promote independent
sealed tiles through a bounded buffer, share exact prefixes copy-on-write, and
keep SSD or token-replay regeneration as explicitly accounted capacity tiers.
It found no canonical or real quantum mechanism that changes classical
storage bounds.

## 2026-07-12: remaining fixed-width and regeneration screens

Five additional mechanisms were tested after the syndrome rejection.

The shared-quantizer experiment fully counted fixed decoder tables, causal
tables, anchors, code positions, and 64-byte alignment at no more than three
bits per scalar. The model-independent fixed candidate differed from the
K4/V2 Q8 reference by 101.631% mean attention-output error. The best
prompt-trained per-head table still measured 52.244% and independently
violates the no-calibration rule. Exact common-key and common-value gauge
identities matched to `2.30e-9`, but their quantized versions were worse.

Pre-RoPE fixed product coding improved mean error from 85.536% to 68.484%,
but remained unusable. With exact values, the pre-RoPE keys alone still caused
59.216% mean error; with exact keys, the two-bit values caused 39.082%.
Pre-RoPE key quantization is also established by KVQuant and RotateKV, so it
does not supply a new foundation.

An optimistic implicit-normalization screen used per-coordinate scales derived
from the first 64 activations. This gives the proposed weight-derived scaling
more information than projection-column norms would provide while treating all
scale storage as free. The sealed-half fixed K4/V2 candidate measured 35.115%
mean error against exact, compared with 24.003% for K4/V2 Q8. It differed from
that reference by 40.828%. On the full prefix it reached 101.222% mean error.
The fixed Hadamard transform made the result worse. Weight-folded scale removal
is rejected before model integration.

Joint mixed-radix packing solved the D64 byte budget exactly. K15 and V3 form
45-state pairs; two pairs fit in one fixed 11-bit word. The 5,632-byte code
stream plus 456-byte Q6 metadata and 56 alignment bytes makes a 6,144-byte
tile, exactly three bits per scalar. Its byte-exact roundtrip passed, but the
corrected Qwen2.5 lifecycle changed perplexity by +6.377%. The value alphabet
was the trace bottleneck, with 36.122% mean error under exact keys. This
candidate is rejected before native work.

The inverse K12/V4 allocation used separate fixed streams: five base-12 key
indices per 18-bit word, direct two-bit values, Q6 key metadata, Q5 value
metadata, and a 16-byte header. Its D64/G128 record is exactly 6,144 bytes, or
3 bits per scalar. It transferred across two trace prompts with 26.403% and
25.922% mean error, but a 1,024-prediction Qwen2.5 lifecycle changed
perplexity from 99.1984 to 148.5478, or +49.7482%. The temporary FP32 simulator
was removed.

Balanced dependent V3 rounding retained the exact K15/V3 byte layout and
bounded cumulative coordinate rounding error. On the first predeclared seed,
it worsened the causal mean from 53.317% deterministic to 84.066%, versus
41.710% for K4/V2 Q8. The all-seed acceptance rule failed immediately, so the
remaining seeds and lifecycle were not run.

Same-token K-to-V prediction was tested as a distinct residual route. A full
ridge map fitted on tokens 0 through 127 was frozen for tokens 128 through
255. This invalid prompt-calibrated oracle still measured 49.173% mean error
with exact keys and a two-bit value residual. Its actual K4/R1 form fits 2.75
bits per scalar at D64 including two FP16 scales, but measured 141.641% mean
error. No weight-derived map, lifecycle, or native kernel was built. Details
are in `MIXED_RADIX_AUDIT.md` and `KV_PREDICTIVE_AUDIT.md`.

Token-log regeneration reaches 0.005208 bit per K/V scalar only by replaying
the model. At 16K and 64K, optimistic full replay requires 23.342 and 231.907
TOP. A fully counted three-bit FP16 checkpoint schedule still needs 20.730 and
216.604 TOP. Estimated latency is hundreds to thousands of seconds per
generated token. One BF16 residual per token fits at 2.333 bits but reconstructs
one layer only; storing every layer costs 56 bits per scalar. Regeneration is
rejected for hot decode and retained only as a possible suspended-session
rebuild mechanism.

Reproducers and complete accounting are in `SHARED_QUANTIZER_AUDIT.md`,
`PREROPE_FIXED_AUDIT.md`, `DESIGN_SPACE_AUDIT.md`, `MIXED_RADIX_AUDIT.md`, and
`REGENERATION_CACHE_AUDIT.md`.

## 2026-07-12: eOptShrinkQ and direct-factor rejection

The public eOptShrinkQ construction was reproduced from the paper, its cited
eOptShrink estimator, and the cited author's MATLAB. The paper leaves several
details ambiguous and its PyTorch implementation is not public, so the local
result reports those choices explicitly and includes exact-factor and ordinary
rank-matched SVD controls.

On the D64 sealed-tile screen, eOptShrinkQ with Q4 factors measured 39.886%
mean attention error. Ordinary rank-matched SVD measured 38.646%, K4/V2 Q8
measured 24.003%, and the optimistic eOptShrinkQ exact-factor bound measured
36.195%. Full-prefix means were 57.732%, 58.054%, 41.865%, and 46.531%,
respectively. The published shrinker therefore does not improve the attention
objective on this trace.

The fully addressable record averaged 2.570 bits per scalar and had a 2.783
maximum across the measured tile pairs. A portable FP32 Haar table and TQ
decoder table add 0.083 bits per scalar over the trace. The FP16 mutable tail
raises effective storage at 4,095 tokens to 3.236 bits per scalar, so the
format is not universally below three bits.

A separate Apple Silicon benchmark consumed two-bit residuals and Q4 factors
directly without reconstructing dense K/V. At the largest rank inside each
sealed-tile budget, D64 was approximately FP16 speed while D96, D128, and D256
were slower. Every eligible rank was slower than K8/V6 at both 16K and 64K;
even rank 1 reached only 0.67x to 0.84x K8/V6 speed. The benchmark is synthetic
and does not claim model quality. Release `-Werror`, AddressSanitizer, and
UndefinedBehaviorSanitizer checks passed.

The standalone implementation and full qualification are in
`EOPTSHRINKQ_AUDIT.md`, `EOPT_DIRECT_KERNEL_AUDIT.md`,
`prototype_eoptshrinkq.py`, and `bench_eopt_direct_native.cpp`.

The follow-up changed the low-rank objective from Frobenius reconstruction to
correction of the two-bit residual under the measured query and attention
operators. When allowed to see the current queries and exact attention
weights, its Q4-factor record measured 18.950% sealed and 25.032% full-prefix
mean error, better than the 24.003% and 41.865% K4/V2 Q8 controls. The record
was fully counted at 2.877 bits per scalar. This is an invalid future-query
oracle, not a cache encoder.

The causal lifecycle result rejects the trace-conditioned version. A metric
fit only to queries through token 128 then frozen for tokens 129 through 256
measured 33.000% mean error with Q4 factors and 26.321% even with exact
factors, versus 23.154% for K4/V2 Q8. Its FP16 mutable tail also raises the
4,095-token cache to 3.284 bits per scalar. Generic model-weight-derived query
and output metrics have substantial prior-art overlap, so `W_Q`/`W_O`-weighted
SVD alone is not a Laplace novelty claim. Details are in
`WEIGHTED_SPECTRAL_AUDIT.md` and `WEIGHT_METRIC_PRIOR_ART.md`.

## 2026-07-12: exhaustive fixed-alphabet frontier

The D64/G128 sweep enumerated 2,879 complete layouts at or below 3 bits per
scalar. K7/V6 Q8/Q8 was the only nondominated point that beat the K16/V4 Q8
control on mean and p95 on both fixed traces. Its exact mixed-radix streams,
header, byte padding, and 64-byte alignment total 6,144 bytes, exactly 3 bits
per scalar.

The real lifecycle rejected it. Qwen2.5 perplexity changed from 99.1984007 to
206.191565 over 1,024 predictions, or +107.857751%. Top-1 agreement was
58.300781% and mean KL was 0.963934. No native kernel was built. See
`ALPHABET_FRONTIER_AUDIT.md`.

## 2026-07-12: causal direct-codec screens

Three later candidates failed the fixed traces:

- Exact previous-tile normalization state used for zero-metadata K4/V2
  measured 40.077% and 37.175% mean error, versus 23.650% and 24.003% for the
  stored-metadata controls.
- A symbol-clocked K4/V2 gain servo measured 51.606% and 51.105% mean error.
  An unavailable current-tile RMS oracle still measured 52.092% and 51.805%.
- A backward one-bit value forest has an exact reverse-subtree attention
  identity and a complete 2.75-bit D64 record. With exact keys it measured
  27.695% and 24.326% mean error, worse than the 23.650% and 24.003% controls.
  Its complete row-K4 path measured 43.840% and 38.909%.

All three were rejected before model integration or native-kernel work. See
`CAUSAL_DIRECT_AUDIT.md`.

## 2026-07-12 decision snapshot

No paper. Fixed K4/V2 has enough native speed, but no tested metadata design
keeps that speed and the complete record at or below three bits. The exact
three-bit syndrome, K15/V3, K12/V4, and balanced-rounding designs fail quality,
and syndrome metadata also fails speed. The exhaustive scalar frontier's
K7/V6 winner fails its lifecycle by +107.86%. Same-token K-to-V prediction,
stale scale state, a symbol-clocked gain servo, and a backward value forest
also fail. Shared quantizers, pre-RoPE coding, implicit normalization, strict
tails, recurrence, and regeneration are rejected. The next experiment
was an exact eOptShrinkQ reproduction with full storage accounting and direct
factor attention. It failed both the quality comparison and the K8/V6 speed
gate. The matching official Qwen2.5 GGUF was then used to derive full-context
frozen `W_Q` and `W_O` metrics. Weighted truncated SVD lost after its factors
were counted and quantized. A structured K3 plus 31 selected K4 key
coordinates with V2 fit exactly 3 bits per scalar, but changed perplexity from
99.1984 to 916.6573 over 1,024 real predictions, or +824.0646%. Its temporary
simulator was removed and no native kernel was built. The K8/V6 control still
exceeds the storage limit, Gemma 4 quality is missing, and no second-engine
adapter exists. The public vector frontier also separates the gates: OCTOPUS
reports +34.7%/+41.5% perplexity change at 2.50 effective bits per scalar,
while HyperQuant reports +7.4%/+8.1% at the same complete rate and needs 3.48
effective bits for its +1.8%/+1.4% point. The breakthrough and goal gates
remain open.

The newest public scan reaches the same decision. GSRQ omits learned
calibrated codebooks from its nominal sub-one-bit rate and loses substantial
quality at the accelerated point. MosaicKV reports strong packed sparse
attention but only a measured 3x FP16 memory reduction, about 5.33 bits per
scalar, with up to 3.55% per-model quality loss. RaBitQCache retains the full
FP16 cache in host memory. See `JULY_2026_FRONTIER_AUDIT.md`.

vLLM 0.25 adds a fused asymmetric INT4 per-token-head KV path with direct
paged attention. It stores the four-bit zero point inside the low four bits of
the float32 scale representation. This occupies broad KV metadata-field
co-packing and raises the production-kernel baseline, but its payload costs
`4 + 32/D` bits per scalar before page overhead. The release provides no
complete final-code FP16/BF16 quality and speed matrix. It therefore does not
pass the combined Laplace gates. See `VLLM_025_AUDIT.md`.

TurboQuant itself is unchanged between the vLLM 0.24 and 0.25 tags. Its
smallest tagged D128 slot is K3/V3 at 3.1875 bits per scalar. Automatic native
KV for four dense-model boundary layers raises the whole-model rate above 4.6
bits per scalar for 36 layers. The official integration benchmark reports
0.720 GSM8K versus 0.900 for its baseline and 65% of baseline decode-heavy
throughput at that setting.

## 2026-07-13: fail-closed 2,048-prediction controls

The publication screen was raised to 2,048 cached next-token predictions and
made fail-closed for simulated storage and timing. All runs below used the
same Qwen2.5 0.5B Instruct Q4_K_M file, the same technical prose corpus, and
the actual growing cached decode lifecycle.

| Codec | PPL change | Top-1 | Mean KL | Rate evidence | Screen status |
|-------|-----------:|------:|--------:|---------------|---------------|
| Laplace K8/V6 | -0.245% | 95.996% | 0.00495 | 7.501 bps measured logical | early screen only |
| Fixed K4/V2 | +4.774% | 84.131% | 0.07880 | storage invalid, FP32 simulator | reject |
| MLX-LM affine Q2/G64 | about +16,766% | 8.250% | 5.31648 | 2.5 bps theoretical payload | reject |
| MLX-VLM TurboQuant K2/V3 | +9,402.033% | 10.596% | 5.16834 | 2.75 bps theoretical before constants | reject |
| KIVI K2/V2 G32 R128 | +34.977% | 69.580% | 0.32369 | at least 3.40625 bps at T2048 D64 before headers | reject |

KIVI was reproduced from [the official repository](https://github.com/jy-yuan/KIVI)
at commit
`876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`. Keys use asymmetric two-bit
quantization per coordinate over 32-token groups. Values use asymmetric
two-bit quantization per token over 32-coordinate groups. The most recent 128
tokens remain FP16. The lifecycle ordering matches the released decode path:
completed 128-token key chunks are sealed after their last FP16 attention use,
and one value leaves the FP16 window after each later attention step.

KIVI's packed two-bit codes are not a complete two-bit cache. Each 32-scalar
group also stores an FP16 minimum and FP16 scale. This adds one bit per scalar,
making the quantized body exactly 3 bits per scalar before the FP16 window,
headers, or alignment. It cannot satisfy the complete 3-bit limit at any
finite context length.

The Qwen model SHA-256 was
`74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`.
The corpus SHA-256 was
`9fcba412b78ce9c39775e662496f68efe6431fc5532a043c4a6f06866d058743`.
The corrected capture binary SHA-256 was
`a5da4b1daa4ac650ccc0f3ab6480a25a3ea3a316da92fd80320dd4f74ecba11e`.

The source-matched MLX-LM simulator used commit
`15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`. The source-matched MLX-VLM
TurboQuant simulator used commit
`ff2c9a3d06237c6c4d093598cd6935a955bdd4d0`. Neither has an official
differential fixture yet, so neither is described as exact. Their FP32
lifecycle outputs invalidate timing and measured storage.

Reproduction uses a capture-enabled build and the fail-closed runner. Replace
the model path, choose `k4v2`, `mlxq2`, `turboquant2.5`, or `kivi2`, and use a
new output path for each run:

```bash
cmake -B build-capture -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLAPLACE_NATIVE=ON -DLAPLACE_KV_CAPTURE=ON
cmake --build build-capture
python3 tools/validate_laplace_kv.py \
  --laplace build-capture/laplace \
  --model qwen=/path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --corpus prose=research/laplace_kv/corpora/technical_prose_v1.txt \
  --baseline kivi2 --output /tmp/laplace-kivi2.jsonl
```

These tested scalar baselines fail this model and corpus at this operating
point. That is not a broader impossibility result. No production codec or
paper claim follows from these negative controls.

## 2026-07-13: zero-byte side-channel screen

Two exact storage channels were screened without a model run. Joint K/V token
row order can carry 716 bits in a 128-token tile or 1,683 bits in a 256-token
tile because attention is invariant to a joint row permutation. This is less
than the 3,648 or 5,952 bits needed by the corresponding D64 K4/V2 Q6 metadata.
A row-scale K4/V2 construction that did fit the 256-token channel measured
103.872% mean full-prefix trace error. Exact row scales did not improve it.

Redundant K4 nibbles can carry bits without changing their decoded value. A
K12/V4 Q5 construction had at least 6,277 fixed central carriers for its
4,992-bit metadata on every one of the 48 captured D64 tiles. It measured
57.553% mean, 103.184% p95, and 234.042% maximum trace error, close to the
K16/V4 Q8 full-prefix control. This does not reopen the family: the stronger
K12/V4 Q6/Q5 lifecycle had already changed perplexity by +49.75%.

An adaptive K13/K12 follow-up uses duplicate key nibbles to carry the 5,952-bit
Q6 record. K13 had enough fixed carriers on 44 of 48 tiles; four used K12. The
hybrid measured 58.677% mean, 111.588% p95, and 228.140% maximum full-prefix
error. Its packed body remains exactly three bits per scalar and its aliases
decode to the original K13 or K12 levels. It is not a candidate pass: lifecycle
quality, mutable-tail accounting, native extraction speed, and novelty remain
unverified.

Lossless native codecs did not close the gap. LZ4 kept 3 of 48 tiles under the
three-bit cap, LZFSE kept 12, and a fixed Huffman path kept 36. The Huffman
worst case was 3.453 bits per scalar. No production or capture-simulator code
was added. Full accounting and the decision are in `DESIGN_SPACE_AUDIT.md`.

## Current decision

No paper and no breakthrough claim. K8/V6 passes this one Qwen quality screen
but exceeds the storage limit. Every tested sub-three-bit quality simulator
fails the 2% screen, and none supplies valid native timing and complete storage
evidence. Zero-byte row-order and code-alias channels have not produced a
validated survivor. The required cross-model, retrieval, native Apple Silicon
speed, memory, I/O, portability, and novelty gates remain open.
