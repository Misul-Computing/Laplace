# Addressable entropy tile and decode audit

Status: rejected after native benchmarking. This is not a production claim.

## Verdict

Lossless entropy coding of K4/V2 codes can fit below 3 bits per scalar on the
captured D64 tiles. It does not establish a hard 3-bit bound. The full cache
also exceeds 3 bits at important context lengths if it keeps a 128-token FP16
sink or an FP32 mutable tail.

A direct AArch64 decode-attention kernel is technically possible, but the
measured implementation was 35x to 40x slower than FP16 in the synthetic
single-thread benchmark. GQA could amortize one symbol decode across query
heads, but that unmeasured case does not rescue the current MHA result.

## Minimal random-access record

Each 128-token tile is independently decodable. K symbols are serialized in
token-major order for QK. V symbols are serialized in coordinate-major order
for weighted value accumulation. Reordering does not change the histogram.

| Field | Bytes |
|-------|------:|
| Four 32-bit rANS states per K stream | included in K payload |
| Four 32-bit rANS states per V stream | included in V payload |
| K frequencies, 15 uint16 values | 30 |
| V frequencies, 3 uint16 values | 6 |
| K and V payload lengths, uint16 | 4 |
| Absolute archive offset, uint64 | 8 |
| Six FP16 metadata low/step pairs | 24 |
| Q5 metadata codes | `ceil(3 * (D + 128) * 5 / 8)` |
| Record padding | 0 to 3 |

The final K and V frequencies are implied by the 4096 total. Two uint16
payload lengths cover the supported dimensions through D512. The absolute
offset works for archives larger than 4 GiB. Four-byte record alignment is
enough because rANS renormalizes from bytes and decoded codes enter registers.

For D64, fixed metadata and addressability cost 432 bytes before padding. The
3-bit tile budget is 6144 bytes. The four-lane roundtrip in
`prototype_entropy_addressable.py` measured:

| Quantity | Mean | p95 | Maximum |
|----------|-----:|----:|--------:|
| rANS payload bits/scalar | 2.676656 | 2.747876 | 2.761230 |
| Fully counted sealed tile | 2.888265 | 2.958984 | 2.972656 |

The maximum tile is 6088 bytes, leaving 56 bytes. This result covers 48 D64
tiles from `/tmp/laplace-qwen-256-all.kvt`. It does not cover D32, D96, D256,
D512, another model, a mutable tail, allocator overhead, or a sink.

Reproduce it with:

```sh
python3 research/laplace_kv/prototype_entropy_addressable.py \
  /tmp/laplace-qwen-256-all.kvt
```

## Why there is no lossless escape under a hard 3-bit cap

Fixed-width K4/V2 codes already consume exactly 3 bits per scalar. At D64,
adding the 432-byte metadata, tables, header, and offset makes a raw escape
6576 bytes, or 3.210938 bits per scalar before any extra alignment. A uniform
or nearly uniform tile can therefore exceed the cap.

Valid choices are limited:

1. Report the over-budget tile and give up the hard claim.
2. Enforce a per-head aggregate budget and spend compression savings from
   other tiles. This still needs a lower-precision fallback when the ledger is
   empty.
3. Requantize an over-budget tile, for example to a selective K3/V2 mode. This
   is lossy relative to K4/V2 and must pass the full quality matrix.

A raw K4/V2 escape is random-access safe but cannot satisfy the stated storage
gate. A raw K8/V6 or FP16 escape is further over budget.

## Sink and mutable-tail accounting

The sealed-tile rate is not the cache rate. With the measured mean rate of
2.888265, a 128-token FP16 sink and no partial tail produce:

| Context | Effective bits/scalar |
|--------:|----------------------:|
| 4K | 3.298007 |
| 16K | 2.990700 |
| 64K | 2.913874 |

Immediately before a 128-token FP32 tail seals, the corresponding rates are
4.200962 at 4095 tokens, 3.216379 at 16383, and 2.970290 at 65535. The research
simulator originally left its sink in FP32. The corrected simulator rounds the
sink and mutable baseline tile to FP16 immediately. The table remains a
storage-policy calculation, not a measurement of simulator allocation.

The hard storage target therefore needs an incremental tail representation and
a compressed sink. Sealed-tile entropy alone cannot pass it.

## Direct AArch64 consumption path

The smallest viable kernel operates on one KV head and all query heads in its
GQA group:

1. Unpack the six Q5 metadata fields into small FP32 scale arrays.
2. Transform each query into the stored domain and form the tile-specific key
   coefficient vector.
3. Quantize key coefficients to signed Q8 in 16-coordinate blocks.
4. Decode four interleaved K rANS states into one 16-byte code vector at a
   time. Feed the same vector to `USDOT` for every query head. Apply each block
   scale, the key zero term, and the per-token column scale to 128 scores.
5. Compute each query's local softmax and merge it with the running online
   softmax state.
6. Form nonnegative Q8 vectors from `weight * value_row_scale`.
7. Decode V in coordinate-major order. Feed each 16-code vector directly to
   `USDOT` for every query, add the value zero term, apply the coordinate
   scale, and accumulate the output.
8. Apply the inverse transform once after all tiles.

This uses only score, weight, metadata, and output scratch. It never writes a
dense decoded K or V tile. It is still a two-phase tile kernel because V
weights are unavailable until K scores and softmax are complete.

The Q8 coefficient step is an additional approximation. It needs a simulator
and end-to-end quality test before it can replace FP32 coefficient FMAs.

## Likely speed limits

- One rANS state is serial. Four interleaved states are required to expose
  independent work to the out-of-order core. Interleaving is a standard ANS
  technique, not a Laplace contribution.
- Every symbol needs a state update, symbol lookup, and conditional byte
  refill. The current Python prototype's 4096-entry decode table must not be
  rebuilt for every tile in the hot path.
- A production experiment should compare the current 12-bit frequency table
  with an 8-bit table. A 256-entry K decode table is cheap to build, cuts the
  stored frequency field from 36 to 18 bytes, and may trade a small amount of
  compression for much lower lookup cost.
- The tested four-context token predictor is compatible with four interleaved
  states. The context for symbol `i` is symbol `i - D`; because supported D is
  a multiple of 16, both symbols use the same state lane when the lane is
  `i % 4`.
- The same predictor is naturally token-major. K can enter QK directly. V
  should decode a 16-token by 16-coordinate microtile, transpose that 256-byte
  scratch block, and then use `USDOT` against 16 weights. Decoding a whole V
  tile would discard the main memory advantage.

- MHA with GQA ratio 1 is the worst case because entropy decode cannot be
  amortized across query heads. Phi-3 D96 is the required speed and portability
  test.
- The existing FP16 path already reuses each KV vector across its GQA group.
  The candidate must beat that path, not a per-query reread baseline.
- Variable records should be read sequentially during full attention. The
  offset index exists for random access and rollback, not for a gather-heavy
  normal path.
- The released KVarN transform is power-of-two only. The Laplace research
  simulator now uses a separately validated H32 plus Householder transform at
  D96 rather than padding to D128.

### Four-context exact recount

The four-context option was recounted with four states per K and V stream,
600 bytes of Q8 metadata, 144 bytes of context frequency tables, 8 bytes of
payload lengths, and a separate absolute uint64 archive offset:

| Record alignment | Mean | p95 | Maximum |
|-----------------:|-----:|----:|--------:|
| 4 bytes | 2.814982 | 2.977441 | 2.998047 |
| 16 bytes | 2.817871 | 2.980078 | 3.003906 |

The 16-byte form fails the storage gate. The 4-byte form passes these 48 D64
tiles with only 4 bytes left in the maximum tile. It has no practical escape
margin and is not a universal bound.

An ephemeral process-owned cache does not need a magic value, version, or CRC
in every tile. The cache object fixes the format. A mode bit can occupy the
high bit of a payload length, and the record size follows from both lengths and
the fixed fields. A serialized persistent format would need a global header,
which must be included in its cache-wide size. Any retained sealed-tile flag,
container header, allocator slack, sink, and mutable tail must still be
reported. The sink and tail already fail the full-cache limit above.

## Native result

The standalone C++ microbenchmark builds real indexed records, verifies every
four-lane roundtrip, and consumes K and V without a dense decoded tile. At 16K
and 64K, D64 and D96 were 35x to 40x slower than FP16. The exact synthetic
single-thread measurements and build command are in
`NATIVE_CONTEXT_RANS_BENCH.md`. This rejects the current CPU rANS design before
full cache integration.

Background on interleaved entropy states: Fabian Giesen, "Interleaved entropy
coders," https://arxiv.org/abs/1402.3392. Arm documents `SDOT` as four groups
of four 8-bit products accumulated into 32-bit lanes:
https://developer.arm.com/documentation/ddi0596/2021-12/SVE-Instructions/SDOT--vectors---Signed-integer-dot-product-.
