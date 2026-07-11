# Joint mixed-radix K/V tile

Status: rejected. Three exact D64/G128 variants were tested. K15/V3 changed
Qwen2.5 perplexity by +6.38%, K12/V4 changed it by +49.75%, and balanced
stochastic V3 failed its first causal seed before lifecycle work.

## Format

The candidate keeps KVarN's even-dimension transform, dual-axis variance
normalization, and Q6 scale fields, but changes the scalar alphabets to 15 key
levels and 3 value levels. One key code and the matching value code form a
45-state symbol. Two symbols have `45^2 = 2,025` states and fit in one fixed
11-bit word.

At D64/G128:

```text
K/V element pairs:        8,192
11-bit words:             4,096
code payload:             45,056 bits = 5,632 bytes
six Q6 metadata fields:   3,648 bits  =   456 bytes
raw record:                              6,088 bytes
64-byte aligned record:                  6,144 bytes
K/V scalars:                             16,384
effective rate:                          3.000000 bits/scalar
```

The prototype implements the actual little-endian 11-bit stream, decodes every
word, separates both radix-45 symbols, and asserts exact recovery of all K15
and V3 indices. There is no entropy state, directory, or same-tile metadata
prepass. The 56 alignment bytes are fully counted.

This exact layout is the D64 experiment. Smaller head dimensions need a lower
radix or larger tile to remain within three bits after metadata and alignment;
therefore the candidate had not yet passed the universal-dimension gate.

## Trace screen

Input: `/tmp/laplace-qwen-256.kvt`, 336 Qwen2.5 queries. The first 128 tokens
remain FP16-equivalent and the completed second tile is encoded.

| Candidate | Mean error | P95 | Maximum |
|-----------|-----------:|----:|--------:|
| K15/V3 Q6 | 38.906% | 79.499% | 241.329% |
| K15/V3 with exact V | 10.009% | 28.105% | 139.260% |
| Exact K with V3 Q6 | 36.122% | 73.706% | 133.870% |
| K16/V4 Q8 control | 24.006% | 50.578% | 77.371% |

The value alphabet is the main error source. Trace error is only a diagnostic,
so the candidate received one corrected cached-decode lifecycle screen.

## K12/V4 split-stream follow-up

The first result left one obvious inverse allocation untested. K12/V4 keeps
all four value levels and removes four key levels. A separate fixed stream
avoids the rounding loss of the original joint pair packing:

```text
K indices: 8,192
five base-12 K indices per 18-bit word
K words: 1,639 * 18 bits                  29,502 bits, 3,688 bytes
V4 indices: 8,192 * 2 bits                16,384 bits, 2,048 bytes
Q6 key metadata: (2D + G) * 6               1,536 bits,   192 bytes
Q5 value metadata: (2G + D) * 5             1,600 bits,   200 bytes
fixed header                                                16 bytes
complete D64/G128 record                                 6,144 bytes
effective rate                                    3.000000 bits/scalar
```

The standalone packer groups five K12 indices into each little-endian 18-bit
word, packs V4 directly, decodes both streams, and checks every index. The two
unused bits after byte packing are included. The record is already 64-byte
aligned.

On the regenerated official trace, K12/V4 Q6/Q5 measured 26.403% mean,
62.207% p95, and 147.634% maximum attention error. Its K16/V4 Q8 control
measured 23.652%, 52.953%, and 77.751%. On the older prompt, the candidate
measured 25.922%, 54.511%, and 149.515%, versus 24.006%, 50.578%, and 77.371%
for the control. The mean transferred better than K15/V3, but its tails were
still worse.

A temporary capture-only simulator then applied the K12/V4 rule at every
completed tile during 1,024 real Qwen2.5 predictions:

| Metric | FP16 | K12/V4 Q6/Q5 |
|--------|-----:|---------------:|
| Perplexity | 99.1984007 | 148.547842 |
| Perplexity change | | +49.748223% |
| Top-1 agreement | | 72.753906% |
| Mean KL | | 0.5782062 |
| P95 KL | | 2.9729926 |
| Maximum KL | | 9.9010655 |

The simulator stored decoded FP32 rows, so its reported timing and memory were
invalid. The quality result exceeds the two-percent limit by nearly 25 times.
The engine hook was removed.

The 6,144-byte construction also covers only a complete tile. A partial tile
cannot amortize its row and column metadata at three bits. A different mutable
format would still be required even if lifecycle quality had passed.

## Balanced stochastic V3 follow-up

The deterministic K15/V3 ablation showed that V3 caused most of the error.
A final seal-time experiment replaced nearest rounding with either independent
stochastic rounding or balanced dependent rounding. The balanced encoder uses
one counter-derived offset per value coordinate and rounds cumulative
fractional codes so every prefix error is below one code step. It changes no
stored field and keeps the exact 6,144-byte record.

The acceptance rule required all eight fixed seeds to improve deterministic
K15/V3 mean and p95 and reach the K4/V2 Q8 control. Seed 0 was therefore a
decisive early rejection:

| V3 rounding | Mean | P95 | Maximum |
|-------------|-----:|----:|--------:|
| Deterministic nearest | 53.317% | 134.587% | 218.032% |
| Independent stochastic | 83.069% | 189.860% | 273.074% |
| Balanced dependent | 84.066% | 204.350% | 293.031% |
| K16/V4 Q8 control | 41.710% | 105.167% | 179.157% |

This causal screen encodes tokens 0 through 127 once and evaluates queries at
the 256-token prefix with the later tail exact. Balanced rounding is unbiased
over random seeds, but attention consumes one realized cache. Its added
variance is much worse than the deterministic MSE choice. The remaining seven
seeds, lifecycle, and native work were not run because the predeclared
all-seed rule had already failed.

## Corrected lifecycle result

Qwen2.5 0.5B, 1,024 predictions, D64:

| Metric | FP16 | K15/V3 Q6 |
|--------|-----:|------------:|
| Perplexity | 1.74775694 | 1.85921366 |
| Perplexity delta | | +6.377129% |
| Top-1 agreement | | 98.046875% |
| Mean KL | | 0.0698234 |
| P95 KL | | 0.2925886 |
| Maximum KL | | 4.1335668 |

The simulator kept the same FP16 sink and mutable-tile lifecycle as the other
corrected research baselines. Its reported FP32 backing memory and timing are
invalid format measurements.

## Decision

Reject before native kernel work or cross-model promotion. The candidate fails
the 2% quality limit by more than three times. Joint mixed-radix packing solves
the byte-accounting problem without an entropy decoder, but dropping V from
four reconstruction levels to three loses too much attention output fidelity.
The K12/V4 follow-up preserves V4 and also has an exact fixed record, but its
model lifecycle fails by +49.75%. Neither allocation is a LaplaceKV foundation.
Balanced stochastic V3 also fails its first causal seed by a wide margin.

The temporary capture-only lifecycle hook was removed after this decision.
The standalone prototype preserves the byte-exact packing and trace evidence.

Reproduce the storage and trace screen with:

```bash
python3 research/laplace_kv/prototype_mixed_radix.py \
  /tmp/laplace-qwen-256.kvt

python3 research/laplace_kv/prototype_mixed_radix.py \
  /tmp/laplace-qwen-official-256-all.kvt \
  --key-levels 12 --value-levels 4 \
  --key-metadata-bits 6 --value-metadata-bits 5 --split-k12-v4

python3 research/laplace_kv/prototype_mixed_radix.py \
  /tmp/laplace-qwen-official-256-all.kvt \
  --sink 0 --value-rounding balanced --seed 0
```
