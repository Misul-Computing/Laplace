# Direct spectral-factor attention audit

Status: storage-feasible as a sealed-tile experiment, rejected as the next
LaplaceKV kernel because every measured rank is slower than K8/V6.

`bench_eopt_direct_native.cpp` tests the Apple CPU attention path suggested by
the factorized representation in
[eOptShrinkQ](https://arxiv.org/abs/2605.02905). It is a standalone synthetic
kernel experiment. It does not reproduce eOptShrink rank selection, prove
model quality, or change the production engine.

For each 128-token tile and each of K and V, the benchmark represents the
normalized matrix as

```text
X = row_norm * R2 + U4 * diag(singular_fp16) * B4
```

`R2` is a fixed signed four-level residual proxy packed at two bits. `U4` and
`B4` use 4-bit indices into stored FP16 codebooks. The synthetic generator uses
fixed linear codebook contents, while the record reserves the same bytes that
data-adaptive codebooks would require. The benchmark stores one FP16 norm per
residual row and one FP16 singular value per retained component.

Attention consumes this representation directly:

```text
scores = row_norm_K * (R_K * q + U_K * (diag(s_K) * (B_K * q)))
output = (weights * row_norm_V)^T * R_V
       + ((weights * row_norm_V)^T * U_V) * diag(s_V) * B_V
```

The hot path never reconstructs a dense K or V tile. Two-bit residual QK uses
signed `SDOT`; weighted V uses `USDOT`. Factor products decode 4-bit indices
while forming the small projections. Softmax and online tile merging are in
the timed path.

## Complete sealed-record count

The published eOptShrinkQ headline equation for one matrix is

```text
b + r * (n + d) * bs / (n * d)
```

where `b` is the residual width and `bs` is the factor width. The paper says
that FP16 row norms and FP16 shrunken singular values are left out of that
equation because they are shared across TQ comparisons or negligible. That is
a valid comparison convention, but it is not the complete physical size
required by the LaplaceKV gate. The paper also says that its SVD factors use
data-adaptive Lloyd-Max codebooks without assigning storage to those
codebooks. It does not count a rank field, section lengths, archive addressing,
or alignment. Its final limitations section acknowledges an FP16 buffer of up
to 127 recent tokens and the absence of a hardware-aware latency benchmark.

This benchmark uses the following optimistic but self-contained combined K/V
record. Every section is 64-byte aligned.

| Field | Bytes |
|---|---:|
| Header, including dimension, rank, lengths, and offsets | 64 |
| K and V residual codes | `2 * align64(128*d/4)` |
| K and V FP16 row norms | `2 * align64(128*2)` |
| K-U and V-U 4-bit codes | `2 * align64(128*r/2)` |
| K-B and V-B 4-bit codes | `2 * align64(r*d/2)` |
| Four 16-entry FP16 factor codebooks and two `r`-entry FP16 singular arrays | `align64(128 + 4*r)` |

The TQ residual codebook is fixed and shared, so it costs no tile bytes. The
four stored factor codebooks are a lower bound for data-adaptive factor
quantization: one each for K-U, K-B, V-U, and V-B. Per-component codebooks
would cost more. No external tile directory is assumed. Adaptive variable-rank
random access would need either a separate directory or fixed-size slots;
sequential traversal can read the record length from each header.

The standalone program keeps sections in separate C++ vectors for convenient
parametric testing. Their allocator bookkeeping is not archive data and is not
part of the modeled record. A production implementation would serialize the
listed sections contiguously. The benchmark therefore includes pointer chasing
between sections but does not read header or alignment padding in the hot path.

For a record size `S`, the fully counted rate is

```text
8*S / (2*128*d) bits per K/V scalar.
```

Only ranks at or below 3.0 bits were run:

| D | Largest rank | Tile bytes | Bits/scalar |
|---:|---:|---:|---:|
| 64 | 6 | 6,016 | 2.937500 |
| 96 | 10 | 9,216 | 3.000000 |
| 128 | 13 | 12,288 | 3.000000 |
| 256 | 19 | 24,512 | 2.992188 |

The sealed-record count does not hide the paper's FP16 mutable tail. If a
context contains `m = N mod 128` unsealed tokens, its effective rate is

```text
((N - m) * sealed_bits + m * 16) / N.
```

At the worst 127-token phase just below 16K, the largest ranks above become
3.039, 3.101, 3.101, and 3.093 bits per scalar for D64, D96, D128, and D256.
At the same phase just below 64K, they become 2.963, 3.025, 3.025, and 3.017.
Thus a sealed tile at or below three bits is not by itself a context-wide
three-bit format. A production candidate would need rank headroom or a
compressed online tail.

## Native results

Machine: Apple M5 Pro MacBook Pro, 24 GB, macOS 27.0. Compiler: Apple clang
21.0.0. The process was single threaded. Results are warm-cache medians of 21
trials. FP16 and K8/V6 controls use the same dense values materialized from the
largest-rank synthetic record for that dimension. Every eligible rank from 1
through the listed maximum was timed. The table shows both endpoints.

| Context | D | Rank | Bits | Direct | FP16 | K8/V6 | vs FP16 | vs K8/V6 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16K | 64 | 1 | 2.500000 | 0.1137 ms | 0.1915 ms | 0.0758 ms | 1.684x | 0.666x |
| 16K | 64 | 6 | 2.937500 | 0.1875 ms | 0.1915 ms | 0.0758 ms | 1.021x | 0.404x |
| 16K | 96 | 1 | 2.333333 | 0.1366 ms | 0.2573 ms | 0.0972 ms | 1.884x | 0.711x |
| 16K | 96 | 10 | 3.000000 | 0.2853 ms | 0.2573 ms | 0.0972 ms | 0.902x | 0.341x |
| 16K | 128 | 1 | 2.250000 | 0.1633 ms | 0.3096 ms | 0.1223 ms | 1.897x | 0.749x |
| 16K | 128 | 13 | 3.000000 | 0.3831 ms | 0.3096 ms | 0.1223 ms | 0.808x | 0.319x |
| 16K | 256 | 1 | 2.140625 | 0.2693 ms | 0.6716 ms | 0.2197 ms | 2.494x | 0.816x |
| 16K | 256 | 19 | 2.992188 | 0.7106 ms | 0.6716 ms | 0.2197 ms | 0.945x | 0.309x |
| 64K | 64 | 1 | 2.500000 | 0.4530 ms | 0.7860 ms | 0.3052 ms | 1.735x | 0.674x |
| 64K | 64 | 6 | 2.937500 | 0.7429 ms | 0.7860 ms | 0.3052 ms | 1.058x | 0.411x |
| 64K | 96 | 1 | 2.333333 | 0.5565 ms | 1.0226 ms | 0.3972 ms | 1.838x | 0.714x |
| 64K | 96 | 10 | 3.000000 | 1.1533 ms | 1.0226 ms | 0.3972 ms | 0.887x | 0.344x |
| 64K | 128 | 1 | 2.250000 | 0.6533 ms | 1.2513 ms | 0.4916 ms | 1.915x | 0.752x |
| 64K | 128 | 13 | 3.000000 | 1.5364 ms | 1.2513 ms | 0.4916 ms | 0.814x | 0.320x |
| 64K | 256 | 1 | 2.140625 | 1.0900 ms | 2.7264 ms | 0.9103 ms | 2.501x | 0.835x |
| 64K | 256 | 19 | 2.992188 | 2.8473 ms | 2.7264 ms | 0.9103 ms | 0.958x | 0.320x |

Lower ranks can beat FP16 because the two-bit residual traffic is small. No
rank in the complete sweep beats K8/V6. Even rank 1 reaches only 0.666x to
0.816x K8/V6 speed at 16K and 0.674x to 0.835x at 64K. At ranks close to the
three-bit limit, factor decode and the extra small products consume the FP16
bandwidth advantage as well.

The largest-rank direct outputs differed from their materialized FP16
references by 0.122% to 0.152% relative L2 at 16K and 0.114% to 0.126% at 64K.
This verifies the synthetic direct algebra and measures the benchmark's query
and weight quantization error. It says nothing about compression quality:
the FP16 reference was generated from the same synthetic quantized factors
and residual codes, rather than from real model KV.

## Reproduction

```sh
clang++ -O3 -std=c++20 -mcpu=native -Wall -Wextra -Werror -Isrc \
  -o /tmp/bench-eopt-direct \
  research/laplace_kv/bench_eopt_direct_native.cpp \
  src/laplace_kv.cpp src/ops.cpp -framework Accelerate
/tmp/bench-eopt-direct 16384 21
/tmp/bench-eopt-direct 65536 21
```

The benchmark also passed AddressSanitizer and UndefinedBehaviorSanitizer for
every eligible rank and all four dimensions at a 128-token context.

## Decision

This path is a useful optimistic feasibility bound, not a LaplaceKV candidate.
It confirms that factorized attention can avoid dense reconstruction and can
beat FP16 at low rank. It also shows that direct factor consumption does not
beat the current K8/V6 kernel on this Apple CPU at any fully counted rank.

The spectral method still requires the separate real-trace reproduction to
determine whether ranks and residual quality are useful. If that quality test
passes, the representation may remain interesting as an archive format or a
GPU-oriented baseline. It does not satisfy LaplaceKV's Apple decode-speed gate,
and reproducing the published spectral mechanism would not satisfy the novelty
gate by itself.
