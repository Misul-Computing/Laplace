# Frozen-metric Walsh coordinate allocation

Status: storage passes at long context, quality fails. Do not promote this
candidate to production or a native kernel.

## Candidate

`prototype_wh_metric.py` tests a training-free mixed-precision scalar codec.
For a raw K or V row `x`, normalized Walsh-Hadamard matrix `H`, and stored
FP16 norm `nh`, the architectural encoder is

```text
nh = FP16(||x||2)
y  = (x / nh) H
```

The captured trace is already in the `y` domain, so the reproducer does not
apply `H` a second time. It divides by the stored norm at the encoder boundary
and multiplies by that norm once after decoding. Zero-norm rows use the fixed
all-zero path.

Each Walsh coordinate uses a fixed analytic Gaussian Lloyd-Max quantizer for
`N(0, 1/64)`. Most coordinates use Q2. A frozen mask selects at most 47
coordinates across K and V for Q4. The Q2 and Q4 levels are rounded to FP32.
There is no prompt calibration, trace-derived scale, trained codebook, random
draw, or per-tile statistic other than the two FP16 row norms.

For frozen raw-coordinate metric `M`, coordinate sensitivity is

```text
s = diag(H M H^T).
```

The fixed allocation controls select the largest `mK` key sensitivities and
the largest `mV` value sensitivities. The tested splits were K47/V0, K46/V0,
K32/V15, and K24/V23. The global selectors normalize each kind's 64
sensitivities to unit sum before selecting the largest 47 or 46 from their
union. This equal-trace rule was declared before evaluation because the raw K
and V proxy magnitudes are not directly comparable. Matching stripe controls
select the first `mK` and `mV` Walsh coordinates.

Selecting the largest diagonal entries is optimal only for independent,
equal-variance coordinate errors under the frozen quadratic proxy. Real
quantization residuals do not meet that assumption. The selector is therefore
a deterministic heuristic, not a new optimality result.

## Inputs

The trace was regenerated from the same official Qwen2.5 0.5B Instruct
Q4_K_M GGUF used to extract the metrics.

```text
GGUF SHA-256    74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
metric file     /tmp/qwen2.5-weight-metrics.lwm
metric SHA-256  70144b19ec6e7d84d147402f2644e9ec6674accc76709fa18a0991519b659b7d
trace file      /tmp/laplace-qwen-official-256-all.kvt
trace SHA-256   5dad73517d9afd590d61d797020e475cd3d53787c76c8bab6073f864f875df25
trace records   86,016 queries, all positions 1 through 256
topology        24 layers, 2 KV heads, D64
```

Run from the repository root:

```sh
python3 research/laplace_kv/prototype_wh_metric.py \
  /tmp/laplace-qwen-official-256-all.kvt \
  /tmp/qwen2.5-weight-metrics.lwm
```

## Storage

For one token and one layer/KV-head pair, the 47-coordinate body contains

```text
Q2 K/V codes                     256 bits
47 Q4 extra code bits             94 bits
two FP16 row norms                32 bits
total                            382 bits
rate                        2.984375 bits/scalar
```

Two 64-bit masks add 16 bytes per layer/KV head. One 128-row self-contained
pair is 6,112 bytes of body and norms plus 16 bytes of masks. Rounding the
6,128-byte record to 64 bytes gives 6,144 bytes, exactly 3 bits per scalar.

The immediate layout packs every token from token zero into one model-wide
bit pool. It stores the masks once per layer/KV head, one shared 80-byte FP32
Q2/Q4 level table, and aligns the complete pool once to 64 bytes. It needs no
mutable FP16 or FP32 tail. For this 48-layer/head topology, the fully counted
47-coordinate rate first reaches at most 3 bits at context 71:

| Context | Pool bytes | Effective bits/scalar |
|--------:|-----------:|----------------------:|
| 1 | 3,200 | 4.166667 |
| 64 | 147,584 | 3.002604 |
| 127 | 291,968 | 2.993438 |
| 4,096 | 9,388,928 | 2.984660 |
| 16,384 | 37,553,024 | 2.984446 |
| 65,536 | 150,209,408 | 2.984393 |

The 46-coordinate body is 380 bits per token and leaves more layout room. Its
fully counted rate first reaches at most 3 bits at context 36, measures
2.969035 bits at 4,096, and approaches 2.96875 bits. Neither layout meets the
3-bit ceiling at very short context. Any implementation that adds independent
tile alignment, headers, offsets, or a mutable uncompressed tail must count
them again and may lose the storage pass.

## Official-trace results

The causal screen encodes the structured candidates immediately from token
zero. The K4/V2 Q8 control retains its actual completed-128-token-tile
lifecycle, so its incomplete tail remains exact. Errors are attention-output
relative errors against the FP16 trace.

| Method | Mean | P95 | Maximum |
|--------|-----:|----:|--------:|
| K4/V2 Q8 | 16.193% | 78.368% | 198.544% |
| TQ2 | 89.587% | 244.370% | 1999.262% |
| metric K47/V0 | 53.579% | 123.398% | 887.615% |
| stripe K47/V0 | 59.987% | 154.051% | 860.958% |
| metric K46/V0 | 54.053% | 124.218% | 887.022% |
| stripe K46/V0 | 60.537% | 155.452% | 860.958% |
| metric K32/V15 | 59.161% | 149.491% | 1883.951% |
| stripe K32/V15 | 70.790% | 198.453% | 1820.994% |
| metric K24/V23 | 63.769% | 166.916% | 1478.691% |
| stripe K24/V23 | 74.911% | 207.556% | 2077.933% |
| metric global 47 | 60.361% | 153.704% | 1358.950% |
| metric global 46 | 60.579% | 154.527% | 1549.766% |

The final-prefix diagnostic evaluates all 336 layer/query-head records at
position 256 with both complete tiles encoded for every method:

| Method | Mean | P95 | Maximum |
|--------|-----:|----:|--------:|
| K4/V2 Q8 | 54.244% | 104.525% | 178.812% |
| TQ2 | 107.215% | 311.931% | 580.858% |
| metric K47/V0 | 57.772% | 153.091% | 411.099% |
| stripe K47/V0 | 65.028% | 172.508% | 511.003% |
| metric K46/V0 | 58.070% | 136.153% | 411.099% |
| stripe K46/V0 | 66.418% | 179.288% | 511.003% |
| metric K32/V15 | 67.959% | 164.145% | 868.954% |
| metric K24/V23 | 78.703% | 240.906% | 1002.988% |
| metric global 47 | 72.674% | 179.518% | 1021.859% |
| metric global 46 | 73.873% | 177.557% | 1064.177% |

The frozen metric consistently improves over matching fixed stripes, so it
contains useful coordinate sensitivity. It still does not reach K4/V2. The
best final-prefix mean is 6.50% worse than K4/V2, its P95 is 46.46% worse,
and its maximum is 2.30 times as large. Across the full causal lifecycle its
mean error is 3.31 times the K4/V2 control.

## Decision

Reject this format. It establishes that a zero-calibration frozen-weight
metric can rank useful Walsh coordinates and that an immediately packed
long-context record can fit below 3 bits including masks, levels, and global
alignment. It does not preserve attention quality, does not meet the short
context storage ceiling, and has no evidence for the model-level 2% quality
gate. The quality failure comes before native speed testing, cross-model
testing, or production integration.
