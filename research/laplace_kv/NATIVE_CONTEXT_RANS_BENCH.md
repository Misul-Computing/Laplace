# Native two-context rANS microbenchmark

Status: rejected CPU decode candidate.

`bench_context_rans_native.cpp` implements the two-context K4/V2 candidate as
an actual variable-length archive. Each 128-token record contains four rANS
lanes for K, four lanes for V, two K context tables, two V context tables, Q8
metadata, stream lengths, and 16-byte record padding. A uint64 directory entry
addresses every record.

K uses the top bit of the prior token's same-coordinate K4 code as context. V
uses the top bit of the prior V2 code. The decoder consumes K directly into QK
and V directly into weighted value accumulation. It stores only 128 scores,
128 weights, metadata, the output, and one previous-code row. It never writes a
dense decoded K or V tile.

The decoder uses four independent rANS states and a branchless NEON lookup for
the 16-symbol K cumulative distribution. Every record is decoded back to its
original codes before timing. The fused entropy output is also checked against
the packed-code output.

## Build and run

```sh
clang++ -O3 -std=c++20 -mcpu=native -Wall -Wextra -Werror \
  -o /tmp/bench-rans \
  research/laplace_kv/bench_context_rans_native.cpp
/tmp/bench-rans 16384 7
/tmp/bench-rans 65536 5
```

Machine: Apple M5 Pro MacBook Pro, 24 GB, arm64. The process was single
threaded. Inputs were deterministic synthetic codes with an 80% persistent K
context bit and a 90% persistent V context bit. Timings are warm-cache medians.

## Results

| Context | D | Entropy bits/scalar | Entropy | Packed K4/V2 proxy | FP16 | Entropy slowdown |
|--------:|--:|--------------------:|--------:|---------------------:|-----:|-----------------:|
| 16K | 64 | 2.958069 | 8.189 ms | 0.661 ms | 0.231 ms | 35.5x |
| 16K | 96 | 2.870850 | 12.077 ms | 1.009 ms | 0.304 ms | 39.8x |
| 64K | 64 | 2.958435 | 32.736 ms | 2.734 ms | 0.927 ms | 35.3x |
| 64K | 96 | 2.871267 | 48.172 ms | 4.096 ms | 1.299 ms | 37.1x |

Entropy versus packed fusion relative error was exactly zero in every reported
run. Entropy versus FP16 relative error ranged from 5.95e-6 to 1.33e-5 and is
caused by the FP16 baseline's storage rounding.

The packed K4/V2 result is a local proxy using the same Q8 metadata. It is not
TurboQuant. The former TurboQuant sources are absent from the current tree and
cannot be called by this standalone benchmark.

## Decision

This route meets the sealed-tile storage target on these synthetic D64 and D96
inputs, but it is not close to the decode-speed target. Replacing scalar binary
search with branchless NEON reduced rANS time substantially and still left it
about 35x to 40x slower than FP16 at long context. The serial state update,
renormalization, and per-symbol model dependency dominate the saved memory
traffic.

These measurements reject this exact rANS codec for the single-query Apple CPU
hot path. GQA could amortize one code decode across several queries, but that
requires a separate grouped kernel and cannot be claimed from this result.
The benchmark is synthetic and does not measure model quality, prefill, SSD
streaming, or a complete inference token.
