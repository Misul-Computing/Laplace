# vLLM 0.25 KV cache audit

Date: 2026-07-12

## Decision

vLLM 0.25, published on 2026-07-11, adds a strong fused INT4 production
baseline, not a format that
passes the LaplaceKV storage target. Its packed K/V payload costs
`4 + 32/D` bits per scalar before page and allocator overhead. No supported
head width can reach three bits. The release also provides no complete
FP16/BF16 quality and decode-speed comparison for the merged implementation.

The implementation does close one broad novelty route. It stores a four-bit
zero point in the low four bits of a float32 scale's bit representation. A
claim that Laplace invented hiding KV quantization metadata inside another
stored field is therefore unavailable. Laplace's rejected code-index syndrome
construction is a different mechanism, but broad metadata co-packing is now
direct current KV-cache prior art.

## What PagedAttention removal means

The [v0.25.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.25.0)
says that PagedAttention was removed, then identifies the removed code as the
legacy attention implementation. [PR
#47361](https://github.com/vllm-project/vllm/pull/47361) deletes the old CUDA
v1/v2 attention kernels and bindings. It does not delete paged KV allocation,
block tables, or prefix sharing.

The current [KV cache
interface](https://docs.vllm.ai/en/latest/api/vllm/v1/kv_cache_interface/)
still defines page sizes and block-table capacity. The new INT4 source calls
its writer a paged-cache writer and its reader paged attention over the packed
cache. The release headline describes a kernel replacement, not abandonment
of paging.

The systems lesson for Laplace is to keep block allocation, residency policy,
and attention kernels separate. SSD movement is a cache-residency decision.
It is not part of the compression ratio. A packed codec should be consumed by
a replaceable fused kernel without reconstructing dense K/V.

## INT4 format

The official [INT4 per-token-head source
documentation](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/ops/int4_per_token_head/)
shows this sequence for each token and KV head, separately for K and V:

1. Convert the vector to float32.
2. Apply one deterministic randomized Walsh-Hadamard transform.
3. Compute an asymmetric min/max scale and zero point.
4. Quantize to codes 0 through 15 and pack two codes per byte.
5. Clear the low four bits of the float32 scale representation and insert the
   four-bit zero point there.
6. Write packed codes and the modified scale into the paged cache.

The quantizer is:

```text
scale = max((maximum - minimum) / 15, 1e-6)
zero  = clamp(round(-minimum / scale), 0, 15)
code  = clamp(round(value / scale + zero), 0, 15)
```

For one K or V vector of width `D`, the record is `D/2` code bytes plus one
four-byte scale. K and V together cost `D + 8` bytes for `2D` scalars:

```text
effective bits/scalar = 8(D + 8) / (2D) = 4 + 32/D
```

| Head width | Payload bits/scalar |
|-----------:|--------------------:|
| 32 | 5.0000 |
| 64 | 4.5000 |
| 128 | 4.2500 |
| 256 | 4.1250 |
| 512 | 4.0625 |

These rates exclude page padding, block tables, and allocator metadata. The
zero point adds no separate byte, but the float32 scale remains fully counted.

The transform path asserts a power-of-two final dimension, and the packed
writer requires an even dimension. The Triton backend declares head widths of
at least 32 and block sizes divisible by 16. D96, one of Laplace's required
paths, is therefore not supported by this INT4 transform path.

## Decode path

The query receives the forward transform before the packed attention kernel.
The Triton kernel follows the page block table, reads packed K/V and scales,
extracts the zero points, unpacks nibbles, corrects the K dot products and V
accumulation, and performs online softmax without writing a dense cache. The
result receives the inverse transform afterward.

This is the right systems shape for a low-bit cache: direct packed writes,
direct paged reads, and fused dequantization. It is also a useful future
second-engine adapter shape for LaplaceKV once a codec passes Laplace's own
gates.

## Published evidence

The merged [INT4 KV pull
request](https://github.com/vllm-project/vllm/pull/40835) reports one GSM8K
5-shot row, 0.912 flexible exact match and 0.910 strict exact match, each with
0.0273 uncertainty. Its public body does not give an FP16/BF16 baseline,
long-context perplexity or retrieval, or a final-code performance matrix.

The original [INT2/INT4 proposal](https://github.com/vllm-project/vllm/pull/39074)
does include one same-model FP16 row for Qwen3.5-35B-A3B on GSM8K:

| Format | Flexible exact match | Strict exact match |
|--------|---------------------:|-------------------:|
| FP16 | 0.940 | 0.912 |
| INT4 per token/head | 0.912 | 0.910 |
| INT2 per token/head | 0.864 | 0.832 |

INT4 is 0.2 percentage points below FP16 on strict match, but 2.8 percentage
points below it on flexible match. The reported standard errors overlap and
the evidence covers only one task and model. INT2 loses 8.0 percentage points
on strict match and did not ship in v0.25.

An earlier pull-request comment reports a 30.5% total-throughput gain for an
INT4 run on one AMD setup. The comment does not name the comparator and the
measured dedicated decode and prefill kernels were later removed before
merge. Those numbers cannot be attributed to the final v0.25 implementation.

## Apple relevance

The new codec is implemented in the Triton attention backend, with GPU paths
for CUDA, ROCm, and XPU. It is not the macOS CPU backend and cannot be used as
an Apple Silicon implementation. The release's Apple Silicon entry fixes an
OpenMP hang; it is unrelated to INT4 KV.

The reusable idea is architectural: transform while writing, keep packed
records in the block cache, and consume them directly in one attention
kernel. Laplace needs a NEON implementation and Apple measurements rather than
a port of the Triton code.

## TurboQuant in v0.25

TurboQuant is present in v0.25, but it is not a new release feature. The
tagged configuration file has the same Git blob in v0.24 and v0.25.

At D128, the [tagged slot-size
formulas](https://github.com/vllm-project/vllm/blob/v0.25.0/vllm/model_executor/layers/quantization/turboquant/config.py)
give these complete per-token, per-head code and metadata records:

| Preset | K/V bytes | Bits/scalar | Raw FP16 capacity |
|--------|----------:|------------:|------------------:|
| K8/V4 | 196 | 6.1250 | 2.61x |
| K4/V4 | 134 | 4.1875 | 3.82x |
| K3/V4 | 118 | 3.6875 | 4.34x |
| K3/V3 | 102 | 3.1875 | 5.02x |

The K record includes one FP16 norm for low-bit keys. The V record includes
one FP16 scale and one FP16 zero point. Slot alignment is counted. Even the
smallest tagged record remains above 3 bits per scalar.

For dense models, vLLM automatically leaves the first two and last two layers
at native KV precision. The source says this is required because aggressive
presets lost about 30 GSM8K points without boundary protection. Including
those four FP16 layers raises K3/V3 to 4.789 bits per scalar for 32 layers and
4.611 bits for 36 layers, or 3.34x and 3.47x whole-model FP16 capacity.

The official [TurboQuant integration pull
request](https://github.com/vllm-project/vllm/pull/38479) reports Qwen3-4B
GSM8K scores of 0.900 for the baseline, 0.860 for K8/V4, 0.840 for K4/V4,
0.780 for K3/V4, and 0.720 for K3/V3. Its decode-heavy throughput was 79%,
71%, 68%, and 65% of baseline, respectively. Only K8/V4 reached throughput
parity in one very-long-prefill case. These are not Apple measurements, but
they rule out treating vLLM's tagged TurboQuant modes as a caveat-free speed
and quality frontier.

## Effect on LaplaceKV

- Add vLLM 0.25 INT4 to the mandatory production baseline set.
- Retire broad metadata-hiding novelty wording.
- Keep the exact three-bit storage target. vLLM's payload exceeds it by
  construction.
- Preserve Laplace's architecture-neutral cache contract and make page
  allocation, RAM or SSD residency, and packed attention independent layers.
- Do not copy the format. Its useful contribution is evidence that a fused
  transformed packed-cache path can be engineered cleanly.

This audit does not change the current paper decision. vLLM raises the systems
and prior-art bars, but it does not supply a reported point that passes
Laplace's complete storage, quality, and speed gates together.
