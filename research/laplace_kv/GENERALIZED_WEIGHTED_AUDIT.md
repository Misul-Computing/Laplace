# Frozen-weight generalized KV metric audit

Date: 2026-07-12

## Decision

Rejected as the next LaplaceKV foundation. The frozen `W_Q` and `W_O`
metrics are mathematically valid quadratic proxies, but weighted truncated SVD
lost to K4/V2 after its factors were quantized into the counted record. A
generalized eOptShrink rule is not justified by the available noise model.

One structured format is worth retaining as a negative control. K3 on every
key coordinate, K4 on the 31 most weight-sensitive key coordinates, and V2
fits one D64 K/V token in 383 bits. Its completed 128-token record is exactly
3 bits per scalar after its mask and alignment. It improved mean attention
error on the regenerated trace, but lost on the older prompt and had much
worse tail errors on both. A final 1,024-prediction model lifecycle check then
failed by a wide margin. No 2,048-prediction run, native kernel, production
integration, or model-specific mask artifact is justified.

## Inputs and coordinate contract

Model:

```text
/tmp/qwen2.5-0.5b-instruct-q4_k_m.gguf
491400032 bytes
SHA-256 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
```

Authoritative trace regenerated from that GGUF and `longprompt.txt`:

```text
/tmp/laplace-qwen-official-256-all.kvt
SHA-256 5dad73517d9afd590d61d797020e475cd3d53787c76c8bab6073f864f875df25
```

Independent older prompt trace:

```text
/tmp/laplace-qwen-256-all.kvt
SHA-256 ef59cf9c809541fceab50ac6345a9b63dce984a5d4525e5d01d351e7d4f04240
```

The model has 24 layers, 14 query heads, two KV heads, GQA group size seven,
and D64 heads. Every tested query and output projection is Q5_0 in the actual
Q4_K_M GGUF. Attention RMSNorm gains and query biases are F32. The prototype
dequantizes the same tensors that Laplace consumes.

The trace capture is inside `store_k_wh`, `store_v_wh`, and `attention_wh`.
It records K, V, Q, and attention output after the engine's Walsh-Hadamard
transform. The screen therefore conjugates ordinary post-RoPE metrics into
the WH domain and applies them directly to captured vectors. It does not
rotate the trace a second time.

## Frozen metrics

For query head `h`, let `Wq_h` be its D by H query projection block, `g` the
pre-attention RMSNorm gain, and `b_h` its query bias. The declared isotropic
normalized-input proxy is

```text
Cq_h = Wq_h diag(g^2) Wq_h^T + b_h b_h^T.
```

The key metric sums `Cq_h` over the seven query heads sharing one KV head,
averages its exact NeoX RoPE conjugates over absolute positions 0 through
4095, and enters the captured domain:

```text
M_K = H [mean_p R_p (sum_h Cq_h) R_p^T] H^T.
```

For the matching D-column block `Wo_h` of the output projection, the value
proxy is

```text
M_V = H [sum_h Wo_h^T Wo_h] H^T.
```

`M_V` omits cross-head terms involving unknown future attention weights. It
is a local downstream-output proxy, not exact layer loss. The reported
post-WO error uses the quadratic form from the GQA-aggregated `M_V`; the trace
does not identify each concurrently scheduled query head well enough to
reconstruct the exact concatenated layer output.

Both metrics are independently normalized to trace D. Pooling their
coordinate sensitivities is therefore an equal-total K/V proxy, not a
physical comparison of their unnormalized losses.

The PSD implementation symmetrizes each metric, eigendecomposes it, discards
eigenvalues below `1e-8` of the maximum, and uses the Moore-Penrose inverse on
the retained support. No Qwen metric needed truncation. Key condition numbers
had mean `1.342e4` and maximum `3.874e5`; value condition numbers had mean
`6.101` and maximum `28.98`.

## What is mathematically valid

For a fixed PSD metric `M`, the rank-constrained problem

```text
min rank(C)<=r ||(X-C) M^(1/2)||F
```

has the minimum-norm solution

```text
S = M^(1/2)
C = [X S]_r S^+
```

where `[.]_r` is truncated SVD and `S^+` is the pseudoinverse on the declared
support. The prototype uses the same per-tile ranks as Euclidean eOptShrinkQ,
then codes the residual with TQ2 and the low-rank factors with the same Q4
factor representation.

There is no defensible generalized spiked shrinker in this experiment. If
the original residual noise is i.i.d., multiplying by `M^(1/2)` makes it
column-correlated. Applying eOpt's ordinary bulk edge, D-transform, and
singular-vector overlaps to `X M^(1/2)` would violate its noise assumptions.
A valid extension needs a colored-noise spectral law and direction-dependent
overlap derivation. The prototype does not present the invalid rescaling
heuristic as a method.

## Causal first-tile result

The first 128-token tile is encoded once. Queries from positions 129 through
256 are evaluated with that frozen tile and an exact later tail. No prompt
queries, attention rows, or activation statistics select the metric, rank,
position prior, precision, or coordinate masks.

### Regenerated official trace

| Method | Attention mean | P95 | Max | K-score mean | Post-WO proxy mean |
|--------|---------------:|----:|----:|-------------:|-------------------:|
| K4/V2 Q8 | 31.914% | 95.484% | 198.544% | 4.566% | 37.987% |
| Euclidean rank-matched TSVD Q4 | 37.788% | 108.228% | 1,664.750% | 13.640% | 40.742% |
| Euclidean eOptShrinkQ Q4 | 36.920% | 106.851% | 1,652.026% | 13.559% | 39.723% |
| Weighted TSVD Q4 | 46.496% | 136.797% | 1,864.799% | 13.750% | 50.214% |
| Weighted TSVD, uncounted exact factors | 30.406% | 97.394% | 1,849.642% | 10.699% | 33.868% |
| Frozen metric basis Q4 | 35.467% | 121.391% | 800.796% | 14.963% | 39.166% |
| Q2/Q4 equal-total rank 46 | 36.688% | 126.650% | 1,013.083% | 16.922% | 41.461% |
| Q2/Q4 K46/V0 | 32.381% | 105.593% | 761.277% | 11.509% | 37.416% |
| K3 plus selected K4x31, V2 | 30.644% | 106.075% | 747.892% | 9.501% | 36.123% |

Exact weighted factors show some mean headroom, but their p95 is already
worse than K4/V2 and their maximum error is over nine times larger. They also
do not fit the counted Q4 record. Factor quantization removes the mean gain
and makes the counted method the worst row in the main comparison.

The K3/K4/V2 structured format improves the official mean by 1.270 percentage
points and the post-WO proxy mean by 1.864 points. Its p95 is 10.591 points
worse and its maximum is 3.77 times the K4/V2 maximum.

The main table uses the 4,096-position prior declared above. A companion run
used the model's complete 32,768-position context so its mask matched the
later `LWPSD1` lifecycle artifact:

| Method | Attention mean | P95 | Max |
|--------|---------------:|----:|----:|
| K4/V2 Q8 | 31.914% | 95.484% | 198.544% |
| K3 plus selected K4x31, V2 | 30.025% | 104.477% | 605.766% |

The wider prior slightly improves the structured candidate's mean and tails,
but its p95 and maximum remain worse than K4/V2.

### Older prompt trace

| Method | Attention mean | P95 | Max | K-score mean | Post-WO proxy mean |
|--------|---------------:|----:|----:|-------------:|-------------------:|
| K4/V2 Q8 | 23.154% | 79.422% | 238.679% | 4.530% | 26.937% |
| Weighted TSVD Q4 | 37.346% | 118.648% | 740.641% | 13.684% | 39.821% |
| Weighted TSVD, uncounted exact factors | 24.044% | 64.100% | 1,059.798% | 11.231% | 26.650% |
| Frozen metric basis Q4 | 30.376% | 113.224% | 565.919% | 15.677% | 33.191% |
| Q2/Q4 equal-total rank 46 | 32.656% | 124.587% | 798.216% | 16.799% | 36.480% |
| Q2/Q4 K46/V0 | 28.708% | 109.969% | 703.640% | 11.819% | 32.620% |
| K3 plus selected K4x31, V2 | 26.827% | 105.458% | 493.242% | 9.583% | 30.943% |

The structured candidate's mean gain does not transfer. It loses to K4/V2
by 3.673 percentage points and again has worse p95 and maximum error.

## Both tiles encoded at position 256

| Trace | Method | Attention mean | P95 | Max | K-score mean | Post-WO proxy mean |
|-------|--------|---------------:|----:|----:|-------------:|-------------------:|
| Regenerated | K4/V2 Q8 | 54.244% | 104.525% | 178.812% | 4.070% | 63.637% |
| Regenerated | K3 plus selected K4x31, V2 | 49.851% | 137.804% | 264.702% | 11.079% | 56.923% |
| Older prompt | K4/V2 Q8 | 41.865% | 87.971% | 137.450% | 4.755% | 45.719% |
| Older prompt | K3 plus selected K4x31, V2 | 42.401% | 93.091% | 343.914% | 12.796% | 46.501% |

The same pattern remains: one prompt has a better mean, the other loses, and
tail and key-score errors are consistently worse.

With the full 32,768-position prior, the regenerated final-prefix candidate
measured 50.470% mean, 115.727% p95, and 308.802% maximum error. The K4/V2
control measured 54.244%, 104.525%, and 178.812%. The mean again hides the
candidate's worse tail.

## Model lifecycle rejection

A temporary compile-gated FP32 simulator applied the exact structured rule to
every cached row during a real Qwen2.5 evaluation. It loaded the version-2
`LWPSD1` artifact built with the model's full 32,768-position RoPE prior,
converted each ordinary-domain key metric to the WH domain, selected its 31
largest diagonal coordinates, and used the same FP16 row norms and analytic
Gaussian Q2, Q3, and Q4 decoder constants as the prototype.

```text
predictions           1024
FP16 perplexity       99.1984007
candidate perplexity 916.657324
perplexity change    +824.064620%
top-1 agreement        21.093750%
mean KL                 2.8876975
p95 KL                  6.1500576
maximum KL             12.6939586
```

The simulator stored decoded rows in FP32, so its reported timing and memory
were intentionally invalid. The quality failure is independent of a packed
layout. It exceeds the 2% model-level limit by more than two orders of
magnitude. The planned 2,048-prediction run was stopped, and the simulator was
removed rather than retained as dead engine code.

Reproduction command used before removal:

```bash
LAPLACE_KV_BASELINE=k3k4w \
LAPLACE_KV_WEIGHT_METRICS=/tmp/qwen2.5-weight-metrics.lwm \
./build-capture-make/laplace \
  /tmp/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --eval-file research/REPORT-2026-07.md \
  --eval-limit 1024 --max-seq 1025 --laplace-resident
```

## Storage

The spectral methods used mean rank sum `5.271` on the regenerated trace.
Their mean aligned pair record is `2.552734` bits per scalar, with every
record below 3 bits. A practical weighted encoder retaining FP16 metric
eigenvectors and FP32 eigenvalues adds 16,896 bytes per layer/KV head. With
the mutable FP16 tail and shared dense rotation counted, effective weighted
storage is `3.232870` bits at context 4,095, `2.722737` at 16,383, and
`2.595233` at 65,535. The counted quality failure makes further layout work
irrelevant.

The fixed-basis Q4 record averages `2.479167` bits before its model-specific
FP16 basis state. It remains below 3 bits at the reported long contexts, but
its quality and extra basis transforms reject it.

The strongest structured row uses:

```text
K: 31*4 + 33*3 bits       223 bits
V: 64*2 bits              128 bits
two FP16 row norms         32 bits
total per token           383 bits
128-token body           6128 bytes
one 64-bit key mask         8 bytes
raw record               6136 bytes
64-byte aligned record   6144 bytes
effective rate          3.000 bits/scalar
```

The analytic Gaussian Q2, Q3, and Q4 levels require no prompt-trained
codebook. They can be fixed decoder constants. The mask is derived solely
from the frozen key metric.

An immediate partial tail can pack 383 bits per token without falling back to
FP16. One mask costs 64 bits per layer/KV head, so a partial-only cache exceeds
3 bits below 64 tokens unless the mask is recomputed rather than retained.
Completed records and immediate tails at 64 tokens or more can stay at or
below 3 bits. This is a storage construction, not an implemented lifecycle;
headers, addressability, and decoder constants still need a production layout
before making a whole-cache claim.

A conservative Q2/Q4 rank-46 record uses 380 bits per token. Its 6,080-byte
body leaves 64 bytes in the aligned record for two masks, a header, and fixed
padding while remaining exactly 6,144 bytes. It has cleaner layout slack but
worse quality than the K3/K4/V2 row.

## Prior-art and next-action boundary

Frozen-weight quadratic metrics, sensitivity-ranked mixed precision,
weighted low-rank approximation, low-rank KV factors, and direct compressed
attention are all occupied broad ideas. This experiment does not establish a
novel generalized shrinker. A narrow model-only WH coordinate rule could be a
Laplace implementation detail only after it wins a measured storage, quality,
and speed frontier. It does not do so here.

Do not build a native kernel or run the longer lifecycle for these formats.
The trace tails were poor and the actual 1,024-prediction quality result was
catastrophic. Continue searching for a method that controls key-score tails
rather than optimizing only a frozen quadratic mean.

## Reproduction

The recorded environment used Python 3.9, NumPy 2.0.2, and `gguf` 0.18.0.
The shared factor quantizer fixes each SVD component sign by making the
largest-absolute left-factor coordinate non-negative before fitting its
codebook.

```bash
PYTHONPYCACHEPREFIX=/tmp/laplace-pycache \
python3 research/laplace_kv/prototype_generalized_weighted.py \
  /tmp/laplace-qwen-official-256-all.kvt \
  /tmp/qwen2.5-0.5b-instruct-q4_k_m.gguf

PYTHONPYCACHEPREFIX=/tmp/laplace-pycache \
python3 research/laplace_kv/prototype_generalized_weighted.py \
  /tmp/laplace-qwen-official-256-all.kvt \
  /tmp/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --rope-horizon 32768

PYTHONPYCACHEPREFIX=/tmp/laplace-pycache \
python3 research/laplace_kv/prototype_generalized_weighted.py \
  /tmp/laplace-qwen-256-all.kvt \
  /tmp/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

The script contains direct self-checks for the closed-form RoPE covariance
average and full-rank weighted reconstruction.
