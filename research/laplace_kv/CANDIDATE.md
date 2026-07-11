# LaplaceKV bounded candidate L3-A

Status: rejected. Its K3/V2 mutable tail changed Qwen2.5 perplexity by
+1,852.97%, and its synthetic single-thread native rANS microbenchmark was 35x
to 40x slower than FP16.

## Contract

The cache has no FP16 sink. Every KV scalar is represented by either a mutable
tail record or a sealed 128-token tile. The packed size includes code payloads,
metadata, frequency tables, stream headers, directory offsets, and alignment.

The transform factors an even head dimension as `D = m * p`, where `p` is its
largest power-of-two divisor. It applies a normalized Walsh-Hadamard transform
inside each `p` block and the self-inverse Householder transform
`I - 2/m 11^T` across blocks. Power-of-two dimensions retain the existing
Walsh-Hadamard result. Odd dimensions are unsupported by this candidate.

## Mutable tail

Each new K and V vector is transformed immediately. K uses symmetric midrise
3-bit quantization and V uses symmetric midrise 2-bit quantization. Each vector
has one FP16 scale. Codes are packed across the head with no per-token offset.

The rate is `2.5 + 16/D` bits per scalar. It is 2.75 at D64, 2.667 at D96,
and 2.5625 at D256. When a tail reaches 128 tokens, the sealed encoder consumes
the reconstructed tail. The encoder does not retain a hidden FP16 or FP32 copy.

## Sealed tile

K uses asymmetric 4-bit codes and V uses asymmetric 2-bit codes after eight
iterations of dual-axis variance normalization. The six scale and zero fields
use one linear 8-bit field each. Every field stores an FP16 low and step, for
24 header bytes total.

The primary code stream uses four interleaved rANS states. It has two models
for K, selected by the top bit of the previous token's K4 code at the same
dimension, and two models for V, selected by the top bit of the previous
token's V2 code. K is serialized token-major. V is serialized in the order
required by its direct decode microtile. The record includes normalized
frequency tables, internal stream offsets, and 16-byte final alignment. A
separate uint64 directory entry gives the tile's absolute offset.

If the primary record would exceed exactly `3 * 2 * D * 128 / 8` bytes after
adding its directory entry, the encoder stores fixed-width K3/V2 codes with the
same Q8 metadata. This fits below 3 bits per scalar at D64 and larger. At D32,
the body, metadata, and directory exactly consume the 3-bit budget before a
mode or framing bit, so the candidate also fails the full supported-dimension
contract. The fallback is lossy relative to K4/V2 and must be counted and
included in all quality results.

## Decode contract

The attention kernel decodes K symbols directly into QK accumulation. After
online softmax, it decodes V through a bounded microtile and accumulates the
weighted values. A full dense K or V tile is never materialized. GQA queries
sharing one KV head consume each decoded symbol once. MHA is the required
worst-case performance test.

## Frozen gates

- At most 3.0 encoded bits per scalar for every cache state, including the
  mutable tail and all packed overhead.
- Perplexity increase no greater than 2% for every registered model.
- Actual cached decode, not trace error, decides quality.
- D64, D96, and D256 must pass before the format can replace the control.
- The native D96 MHA kernel must beat FP16 and the former 4-bit path at 16K and
  64K before any speed claim.
- A literature audit and measured ablation must isolate a contribution beyond
  KVarN normalization, ordinary context coding, and ordinary interleaved rANS.

The current Python roundtrip and FP32 cache simulator do not establish native
storage, RAM, or timing results.
