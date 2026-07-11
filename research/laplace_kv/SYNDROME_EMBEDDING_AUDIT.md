# Syndrome-embedded K4/V2 metadata

Status: rejected. The sealed-tile representation reaches exactly 3 bits per
scalar, but it fails both the model-quality and native decode-speed gates.

## Format and bit count

This experiment stores a D64/G128 K4/V2 tile in its fixed-width code indices:

- K codes: `64 * 128 * 4 = 32,768` bits.
- V codes: `64 * 128 * 2 = 16,384` bits.
- Total: 49,152 bits for 16,384 K/V scalars, exactly 3.0 bits per scalar.
- External metadata, offsets, headers, padding, and alignment: zero bits.

The 6,144-byte tile has a fixed stride and is already 64-byte aligned. The
head dimension, group size, and metadata precision are part of the cache
contract, so they do not need a per-tile tag.

This count applies to one sealed tile. The FP16 sink and mutable-tail lifecycle
used by the trace harness are outside this tile result and still need a full
cache-state accounting.

Each of the six KVarN scale fields is linearly quantized. Its FP16 low and step
are included in the payload, not treated as free. The complete payload sizes
are 3,072 bits for Q5, 3,648 bits for Q6, and 4,800 bits for Q8.

For each Hamming group, the parity of code index LSBs is multiplied by the
standard parity-check matrix. The encoder flips the one index named by the
syndrome difference. A parity flip changes a K4 or V2 index by one level. If
both directions exist, it chooses the neighbor nearest the source activation.
The decoder extracts the syndrome bits and reconstructs every metadata bit.
The script asserts this roundtrip on every tile before attention is evaluated.

"Reversible" here means exact metadata recovery. The original unmodified code
indices cannot also be recovered without extra information. That stronger
interpretation is impossible at the same fixed size by the pigeonhole
principle.

## Layouts

The K-only layout selects the largest ordinary Hamming group that has enough
K capacity. Q5 uses Hamming(7,3); Q6 and Q8 use Hamming(3,2).

The prioritized Q6 K-first layout uses 1,170 Hamming(7,3) K groups for 3,510
payload bits and 46 Hamming(7,3) V groups for the remaining 138 bits. Its
selected payload capacity is therefore exactly `3,510 + 138 = 3,648` bits.
It consumes 8,190 of 8,192 K carriers and 322 of 8,192 V carriers. Its
expected changes are 1,023.75 K indices and 40.25 V indices. Across the 48
captured tiles it measured 1,024.38 K and 39.81 V changes.

The joint layout concatenates K before V and uses the largest ordinary group
that fits. A repeated-column matrix variant was also tested. It consumes all
otherwise-unused carriers and chooses the lowest local-error carrier with the
required parity-check column. This selector is implicit and costs no bits.
It did not reduce attention error.

## D64 trace result

Input: `/tmp/laplace-qwen-256.kvt`, 336 queries over 48 layer/head tiles. The
first 128 tokens remain FP16 and the completed 128-token tile is encoded. All
errors are attention-output relative errors against unembedded K4/V2 with Q8
metadata. They are diagnostics, not model-level quality results.

| Metadata | Layout | Mean K flips | Mean V flips | Mean error | p95 | Max |
|----------|--------|-------------:|-------------:|-----------:|----:|----:|
| Q5 | unembedded | 0.00 | 0.00 | 8.809% | 25.166% | 126.076% |
| Q5 | K-only Hamming | 895.90 | 0.00 | 10.317% | 26.028% | 121.409% |
| Q5 | joint Hamming | 510.96 | 207.83 | 11.299% | 30.293% | 125.347% |
| Q6 | unembedded | 0.00 | 0.00 | 7.139% | 29.406% | 120.109% |
| Q6 | K-only Hamming | 1,361.35 | 0.00 | 11.360% | 30.978% | 134.884% |
| Q6 | K-first Hamming(7,3) | 1,024.38 | 39.81 | 11.565% | 33.626% | 236.956% |
| Q6 | joint Hamming | 510.73 | 343.04 | 10.511% | 31.348% | 132.609% |
| Q8 | K-only Hamming | 1,794.44 | 0.00 | 7.924% | 19.549% | 98.287% |
| Q8 | K-first Hamming(7,3) | 1,020.35 | 373.75 | 8.967% | 20.770% | 123.816% |
| Q8 | joint Hamming | 1,020.65 | 372.42 | 9.091% | 21.459% | 124.535% |

The lowest mean error was Q8 K-only at 7.924%. Among Q6 layouts, joint
Hamming was lowest at 10.511%. The prioritized Q6 K-first layout limited V
changes as designed, but its 11.565% mean and 236.956% maximum were worse than
the unembedded Q6 control. The repeated-column variants were worse in mean
error for every metadata precision. Local scalar reconstruction cost did not
predict attention sensitivity.

At the trace stage, this result justified one actual cached-decode lifecycle
test because trace error has not reliably predicted model perplexity. That
test is reported below and rejects the candidate. Fitting Q6 metadata is not
free: the embedding adds a material perturbation to the K4/V2 codes. Standard
Hamming matrix embedding is prior art and is not a novelty contribution by
itself.

Reproduce with:

```bash
python3 research/laplace_kv/prototype_syndrome_embedding.py \
  /tmp/laplace-qwen-256.kvt
```

## Corrected cached-decode lifecycle screen

The exact Q6 K-first layout was also run through the capture-only FP32 cache
simulator. The first 128 tokens remained FP16-equivalent, then each complete
128-token tile was quantized as K4/V2, its 3,648 metadata bits were embedded
into 1,170 K Hamming(7,3) groups followed by 46 V groups, and all later decode
steps read the reconstructed cached values. This matches the lifecycle used by
the other research baselines instead of quantizing a static trace after the
fact.

Qwen2.5 0.5B, 1,024 predictions, D64:

| Metric | FP16 | Q6 syndrome |
|--------|-----:|------------:|
| Perplexity | 1.747757 | 1.867432 |
| Perplexity delta | | +6.847329% |
| Top-1 agreement | | 98.730469% |
| Mean KL | | 0.0736093 |
| P95 KL | | 0.2294893 |
| Maximum KL | | 6.5261223 |

Decision: reject this layout. It exceeds the 2% perplexity-deviation gate by
more than three times. The high top-1 agreement does not override that failure.
The simulator deliberately uses FP32 backing storage, so its reported memory
and timing are not format measurements. No production path changed.

## Native speed screen

The optimized Apple Silicon extractor scans the packed K/V carriers and emits
all 456 metadata bytes before attention can use the tile. It reached 0.836x
FP16 speed at 16K and 0.826x at 64K, and only 0.331x and 0.324x the current
K8/V6 path. See [FIXED_K4V2_BENCH.md](FIXED_K4V2_BENCH.md) for the complete
method and timings. This independently fails both decode-speed gates.

The temporary capture-only lifecycle implementation was removed after the
candidate failed. It is intentionally not retained as dead engine code. The
standalone prototype above reproduces the exact tile layout and trace screen;
the measured lifecycle result remains here as rejection evidence.
