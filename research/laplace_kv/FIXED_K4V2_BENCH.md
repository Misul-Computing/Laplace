# Fixed-width K4/V2 native benchmark

Status: direct code traversal passed; syndrome-carried metadata failed speed.

`bench_fixed_k4v2_native.cpp` measures a 128-token fixed-width K4/V2
attention path on Apple Silicon. K is token-major and packs two unsigned
4-bit codes per byte. V is coordinate-major and packs four unsigned 2-bit
codes per byte. The kernel unpacks codes with NEON table and zip operations,
then uses I8MM `usdot` for both QK and weighted V accumulation. A matrix
multiply instruction is not useful for the single-query shape because there
is no second query row to reuse.

The metadata equations match the KVarN-style dense baseline used elsewhere
in this directory:

```text
K[t,d] = (K4[t,d] * ka[d] + kb[d]) * kc[t]
V[t,d] = (V2[t,d] * va[t] + vb[t]) * vc[d]
```

All six metadata arrays are already decoded as FP32 in the benchmark. The
`fixed_full_ms` path includes per-tile query conditioning, query
quantization, QK, softmax, weight conditioning and quantization, V
accumulation, and online tile merging. The `fixed_kernel_ms` path moves the
query-conditioned K metadata out of the timed region and is an optimistic
upper bound. Both paths still read tile metadata during attention.

## Build and run

```sh
clang++ -O3 -std=c++20 -mcpu=native -Wall -Wextra -Werror -Isrc \
  -o /tmp/bench-fixed-k4v2 \
  research/laplace_kv/bench_fixed_k4v2_native.cpp \
  src/laplace_kv.cpp src/ops.cpp -framework Accelerate
/tmp/bench-fixed-k4v2 16384 21
/tmp/bench-fixed-k4v2 65536 21
```

Machine: Apple M5 Pro MacBook Pro, 24 GB, macOS 27.0. Compiler: Apple clang
21.0.0. The process was single threaded. Results are warm-cache medians of 21
trials. The input codes and metadata are deterministic synthetic uniform
samples. FP16 stores the same reconstructed K and V values after FP16
rounding. The K8/V6 comparison calls the current resident `LaplaceKV`
implementation on those same values.

## Results

| Context | D | Fixed full | FP16 | K8/V6 | vs FP16 | vs K8/V6 |
|--------:|--:|-----------:|-----:|------:|--------:|---------:|
| 16K | 64 | 0.0753 ms | 0.2308 ms | 0.0899 ms | 3.07x | 1.19x |
| 16K | 96 | 0.0795 ms | 0.2824 ms | 0.0976 ms | 3.55x | 1.23x |
| 64K | 64 | 0.2407 ms | 0.7869 ms | 0.3070 ms | 3.27x | 1.28x |
| 64K | 96 | 0.3088 ms | 1.0409 ms | 0.3979 ms | 3.37x | 1.29x |

The preprepared kernel upper bound took 0.0722, 0.0762, 0.2302, and 0.2928
ms in table order. The full path matched it exactly at the output. Its output
relative error against FP16 was 6.27e-5 to 1.17e-4. K8/V6 relative error
against FP16 was 0.00166 to 0.00204. These are synthetic kernel checks, not
model quality results.

## Hamming metadata extraction

The D64 syndrome candidate hides a 3,648-bit Q6 metadata payload in code LSBs.
Each Hamming(7,3) group recovers three bits by XORing the one-based positions
of its seven set carrier bits. There are 1,170 K groups over 8,190 K codes and
46 V groups over 322 V codes. The result is packed into exactly 456 bytes.

The final benchmark performs this extraction before every tile is consumed.
It scans the 4,096 packed K bytes into a carrier bitstream and the first 81
packed V bytes needed by the V groups, computes all 1,216 syndromes, and packs
all 3,648 output bits. Only after that prepass does it use the synthetic FP32
metadata and execute the full attention path. A known-vector self-check covers
both K and V syndrome positions. The extracted payload does not supply the
synthetic scales, as permitted for this extraction-cost test.

| Context | D | Syndrome full | FP16 | K8/V6 | vs FP16 | vs K8/V6 |
|--------:|--:|--------------:|-----:|------:|--------:|---------:|
| 16K | 64 | 0.2303 ms | 0.1925 ms | 0.0762 ms | 0.836x | 0.331x |
| 64K | 64 | 0.9450 ms | 0.7802 ms | 0.3064 ms | 0.826x | 0.324x |

An independent run of the original literal seven-carrier extractor measured
0.4036 ms against 0.2033 ms FP16 and 0.0792 ms K8/V6 at 16K, then 1.4962 ms
against 0.7857 ms and 0.3065 ms at 64K. The packed-bitstream implementation
cuts extraction cost substantially, but neither implementation passes the
speed gate. The optimized syndrome path is about 20% slower than FP16 and
about 3.0x slower than K8/V6.

## Storage accounting

K4/V2 codes alone are exactly 3 bits per scalar. They cannot be a fully
counted 3-bit format because metadata and addressing are additional. The FP32
metadata actually used by this benchmark produces 4.125 bits per scalar at
D64 and 3.875 at D96. Even the previously modeled Q8 serialization, including
six FP16 low/step pairs, produces 3.292969 and 3.226562 bits per scalar.

The preprepared upper-bound path also holds query-conditioned scratch for
every tile. It used 9,216 and 13,312 bytes at 16K, and 36,864 and 53,248 bytes
at 64K for D64 and D96. The full path needs only one 512-byte query-code
scratch buffer, but it still assumes decoded FP32 metadata is immediately
available.

## Limits

This benchmark proves that fixed code traversal is fast enough on this M5 Pro
when entropy decoding is removed. It also shows that a full same-tile Hamming
prepass removes that speed advantage. It does not provide a format that meets
the 3-bit storage gate without syndrome embedding. It excludes archive offsets,
alignment, mutable-tail storage, cache lifecycle work, multiple heads, model
layers, and full token decode. It does not test real KV traces, retrieval, or
perplexity. The measured K4/V2 quantization equations passed earlier model
screens only with FP16 metadata, so the modeled Q8 metadata size cannot be
paired with those quality results without a new end-to-end validation.

No production change follows from this result by itself. A viable candidate
still needs metadata at no net bit cost, or code payload below 3 bits with
enough room for fully counted metadata, while preserving this direct fixed
width kernel shape.
