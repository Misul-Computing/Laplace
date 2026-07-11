# Shared-quantizer KV screen

Status: rejected. The model-independent fixed candidate family fits in exactly
3 bits per scalar and is fixed-width, but its best mean attention error is
101.631% against the K4/V2 Q8 reference. Prompt-derived codebooks are invalid
under the training-free protocol and also fail the trace screen.

## Candidates

`prototype_shared_quantizer.py` applies the existing FP16 Walsh-Hadamard
rotation and compares seven metadata-free or amortized layouts:

- `fixed` uses deterministic signed logarithmic K4 and V2 reconstruction
  levels. The levels are part of the decoder and consume no cache storage.
- `fixed+K-gauge` subtracts the first key from every key. This is exact because
  it adds the same scalar to every attention logit.
- `fixed+KV-gauge` also subtracts the first value and adds it to the attention
  output. The FP16 value anchor is fully counted.
- `causal-RMS` stores two FP16 scales estimated from the first 64 prompt tokens
  and uses fixed standard-normal reconstruction levels.
- `causal-shared` stores one 16-level K table and one 4-level V table, trained
  by deterministic one-dimensional Lloyd updates on the first 64 prompt
  tokens. The 20 FP16 levels cost 40 bytes per layer and KV head.
- The two causal gauge variants train the same tables after exact K or K/V
  centering.

The causal candidates are optimistic controls, not protocol candidates. They
calibrate each layer and head on the prompt, and their first 64 tokens require
a temporary uncompressed training buffer. Both properties violate the frozen
training-free and full-cache-state requirements.

All layouts keep fixed code positions. The first few K coordinates use 3-bit
indices where space is needed for a shared table or value anchor; the rest use
K4 and every value uses V2. A decoder can address every code directly without
an entropy stream, tile directory, or variable-length prepass.

## Exact storage

For `T` tokens and head dimension `D`, the mixed body uses
`T * (6D - R)` bits, where `R` is the number of 3-bit K coordinates. Tables,
anchors, and final 64-byte alignment are included below. The format has no
per-record header because bit allocation, table size, and tile size are fixed
by the cache contract.

At D64 and one 128-token tile:

| Layout | Body | Shared data | Aligned total | Bits/scalar |
|--------|-----:|------------:|--------------:|------------:|
| fixed | 6,144 B | 0 B | 6,144 B | 3.000000 |
| fixed + K gauge | 6,144 B | 0 B | 6,144 B | 3.000000 |
| fixed + K/V gauge | 6,016 B | 128 B FP16 V anchor | 6,144 B | 3.000000 |
| causal RMS | 6,128 B | 4 B scales | 6,144 B | 3.000000 |
| causal shared | 6,064 B | 40 B tables | 6,144 B | 3.000000 |
| causal shared + K/V gauge | 5,968 B | 40 B tables + 128 B anchor | 6,144 B | 3.000000 |

The fixed K/V gauge reduces `D/8` K coordinates to 3 bits. The table-only
layout reduces five. The table plus anchor layout reduces `D/8 + 3`. These
rules produce the same exact 3-bit result at D96 and other supported dimensions
that are multiples of 16. At 256 tokens, the shared objects amortize and the
measured D64 rates are 2.968750 to 3.000000 bits per scalar after alignment.

This accounting covers a sealed archive or completed prefix. It does not make
the prompt-trained variants valid during their first 64 mutable tokens.

## Gauge identity

The anchor is always token zero, so it is causal. For keys,

`softmax(q * (K - k0)) = softmax(q * K)`.

For values, normalized attention weights sum to one, so

`attention(K - k0, V - v0) + v0 = attention(K, V)`.

Across all 336 captured queries, the unquantized gauge implementation matched
ordinary attention to a maximum relative error of `2.30e-9`. The K anchor is
not needed by the decoder and costs zero bits. The V anchor is stored as D
FP16 values and is counted above.

An online encoder still needs the uncentered K anchor while it appends future
tokens. The packed archive can omit it, but a full persistent-RAM measurement
must count that encoder state unless it is regenerated. This caveat only makes
the rejected gauge candidates less favorable.

## D64 trace screen

Input: `/tmp/laplace-qwen-256.kvt`, containing 336 queries across 48 Qwen2.5
layer/head prefixes. The reference is the existing unembedded K4/V2 format
with Q8 KVarN metadata and eight variance-normalization iterations. Errors in
the table are relative to that reference, not a model-level quality score.

The sealed-tile screen leaves tokens 0 through 127 exact and encodes tokens
128 through 255, matching the prior K4/V2 research setup:

| Candidate | Mean | p95 | Max |
|-----------|-----:|----:|----:|
| fixed | 101.631% | 265.471% | 1,596.195% |
| fixed + K gauge | 189.663% | 589.234% | 4,544.759% |
| fixed + K/V gauge | 217.232% | 561.714% | 4,832.164% |
| causal RMS | 61.661% | 162.956% | 489.975% |
| causal shared | 52.244% | 128.263% | 341.706% |
| causal shared + K gauge | 58.979% | 139.427% | 471.743% |
| causal shared + K/V gauge | 61.084% | 134.153% | 469.504% |

The K4/V2 Q8 reference itself measured 24.003% mean, 50.615% p95, and 77.044%
maximum against exact FP16 attention in this screen. The best shared candidate
therefore adds a large error on top of an already lossy low-bit reference.

The full-prefix screen encodes all 256 tokens. K4/V2 Q8 measured 41.865% mean
error against exact attention. The deterministic candidate measured 234.946%
against K4/V2 Q8. The best prompt-trained table measured 83.967% against it.

Reproduce with:

```bash
python3 research/laplace_kv/prototype_shared_quantizer.py \
  /tmp/laplace-qwen-256.kvt
```

## Decision

Reject the mechanism before model integration or a native kernel:

1. The model-independent fixed format meets storage and addressability, but
   misses the K4/V2 Q8 attention point by more than 100% mean error.
2. Exact gauge centering does not rescue fixed companding. Once quantized, both
   gauge variants are worse, and paying for the V anchor forces additional K3
   coordinates.
3. The optimistic prompt-trained table is still 52.244% away from the
   reference on the smaller sealed-tile screen.
4. Per-head prompt calibration and first-tile training are forbidden by the
   protocol even if their quality were acceptable.
5. Shared scalar tables and prompt-derived scales are established quantization
   techniques, so this path does not supply a novelty contribution.

No production code changed. A native benchmark would only test the expected
benefit of fixed-width table lookup after the quality and training-free gates
have already failed.
