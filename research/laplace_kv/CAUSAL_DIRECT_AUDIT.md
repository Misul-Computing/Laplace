# Causal direct-codec audit

Date: 2026-07-12

## Decision

Three training-free candidates were screened after the fixed scalar-alphabet
frontier failed its model lifecycle. All are rejected before native-kernel or
model integration work:

- Reusing the previous tile's exact normalization state makes a zero-metadata
  K4/V2 record, but stale state loses badly on both traces.
- A decoder-synchronized gain servo also fits the K4/V2 code body in exactly
  3 bits per scalar. It fails even when initialized from the current tile's
  unavailable RMS.
- A backward value forest has an exact direct-attention identity and a complete
  D64 record of 2.75 bits per scalar. Its exact-key screen nearly matches the
  control on one trace, but fails the predeclared two-trace gate. Adding its
  fully counted K4 key path makes the result substantially worse.

The failures close the current stale-scale, symbol-clocked scale, and
one-bit backward value-forest branches. They do not establish a general lower
bound for all causal codecs.

## Previous-tile normalization state

The first 128-token tile remains exact. Its exact KVarN-style dual-axis state
is then reused to encode the next 128-token tile as K4/V2 without storing new
metadata. This is more favorable than a deployable predictor because it starts
from the exact preceding source tile rather than only previously decoded
symbols.

| Trace | Candidate | Mean | P95 | Maximum |
|-------|-----------|-----:|----:|--------:|
| Official | K4/V2 Q8 control | 23.650% | 52.951% | 77.804% |
| Official | Previous-state K4/V2 | 40.077% | 88.805% | 133.687% |
| Official | Previous-state K4, exact V | 30.253% | 85.266% | 131.312% |
| Official | Exact K, previous-state V2 | 26.847% | 60.652% | 77.590% |
| Older | K4/V2 Q8 control | 24.003% | 50.615% | 77.044% |
| Older | Previous-state K4/V2 | 37.175% | 73.700% | 196.231% |
| Older | Previous-state K4, exact V | 24.418% | 61.581% | 191.927% |
| Older | Exact K, previous-state V2 | 27.272% | 62.757% | 192.570% |

The state does not track tile-to-tile scale changes well enough. Key error is
the larger problem on the official trace, while the stale V2 maximum becomes
the larger warning on the older trace.

## Symbol-clocked gain servo

This codec stores only K4 and V2 codes. The decoder regenerates one gain state
from decoded code occupancy and changes it by at most one eighth-octave per
token. The analytical clipping recipes are fixed across both traces. No scale,
checkpoint, or lookup-table index is stored in the record.

The code body is exactly 3 bits per scalar:

```text
K4 body                         4DG bits
V2 body                         2DG bits
stored gain state                  0 bits
total                           6DG bits
rate                    6DG / 2DG = 3 bits/scalar
```

| Trace | Candidate | Mean | P95 | Maximum |
|-------|-----------|-----:|----:|--------:|
| Official | K4/V2 Q8 control | 23.650% | 52.951% | 77.804% |
| Official | Previous fixed | 53.791% | 170.237% | 340.567% |
| Official | Previous servo | 51.606% | 141.582% | 354.423% |
| Official | Previous servo plus feedback | 51.612% | 145.934% | 386.323% |
| Official | Current-RMS fixed oracle | 51.646% | 173.555% | 341.067% |
| Official | Current-RMS servo oracle | 52.092% | 154.923% | 355.050% |
| Older | K4/V2 Q8 control | 24.003% | 50.615% | 77.044% |
| Older | Previous fixed | 56.057% | 186.855% | 458.139% |
| Older | Previous servo | 51.105% | 124.132% | 601.422% |
| Older | Previous servo plus feedback | 52.917% | 136.629% | 542.068% |
| Older | Current-RMS fixed oracle | 50.359% | 132.744% | 333.806% |
| Older | Current-RMS servo oracle | 51.805% | 134.729% | 624.677% |

Even the unavailable current-tile RMS oracle is far behind the stored-metadata
control. The failure is not just one-token adaptation lag. A single scalar
gain is not enough state for these transformed K/V rows.

## Backward value forest

Each transformed value chooses zero or one of the previous 15 decoded values
as its parent. The record stores a one-bit signed residual per coordinate, one
FP8 residual scale, and a four-bit parent distance. A K4 vector stores one
FP16 scale with its zero point co-packed into the same field. The prototype
uses an exact zero point and does not perturb the low FP16 scale bits, so its
key path is optimistic.

For every value node `i`, let `p(i)` be its parent and let `R_i` be its stored
residual. A reverse scan accumulates each attention weight into its ancestors:

```text
sum_i weight_i * V_i
  = sum_j subtree_weight_j * R_j
```

This computes attention over the decoded forest without reconstructing dense
values. The direct result matched explicit decoded-cache attention to
`7.64e-6` maximum relative error on the official trace and `2.20e-4` on the
older trace.

At D64/G128, the complete record is:

```text
K4 and V1 bodies                    5DG bits
K scale/zero, V scale, parent       28G bits
16-byte header                       128 bits
raw record                         5,584 bytes
64-byte aligned record             5,632 bytes
effective rate                      2.75 bits/scalar
```

| Trace | Candidate | Mean | P95 | Maximum |
|-------|-----------|-----:|----:|--------:|
| Official | K4/V2 Q8 control | 23.650% | 52.951% | 77.804% |
| Official | Exact K, zero R1 | 39.664% | 74.667% | 102.813% |
| Official | Exact K, forest R1 | 27.695% | 62.163% | 74.109% |
| Official | Row K4, exact V | 33.440% | 102.745% | 619.543% |
| Official | Row K4, forest R1 | 43.840% | 103.392% | 495.069% |
| Older | K4/V2 Q8 control | 24.003% | 50.615% | 77.044% |
| Older | Exact K, zero R1 | 43.711% | 74.327% | 90.634% |
| Older | Exact K, forest R1 | 24.326% | 52.682% | 59.038% |
| Older | Row K4, exact V | 28.521% | 91.403% | 443.157% |
| Older | Row K4, forest R1 | 38.909% | 89.785% | 407.050% |

The exact-key forest is close on the older trace, but it is worse than the
control on mean and p95 on both traces. Its complete K4 path is much worse.
The candidate therefore fails before a lifecycle or native kernel.

## Prior art and scope

Decoder-synchronized adaptive quantization is old. Jayant's 1973 adaptive
quantizer changes step size from preceding codewords, and Ortega and
Vetterli's 1997 backward adaptation redesigns a quantizer from decoded data
without side information. Error feedback and noise shaping are also broad
prior art. AQUA-KV and sequential KV compression occupy predictive KV coding,
while SINQ and KVarN occupy self-normalized low-bit KV formats.

An attention-specific recurrence or direct data structure can still be new,
but it needs a measured Pareto improvement. The exact forest identity is a
useful algebraic result, not a paper contribution after its codec fails the
quality screen.

Primary references:

- [Jayant, 1973](https://onlinelibrary.wiley.com/doi/10.1002/j.1538-7305.1973.tb02008.x)
- [Ortega and Vetterli, 1997](https://infoscience.epfl.ch/server/api/core/bitstreams/a72d526a-94f6-4e07-9c9a-517975ae0338/content)
- [AQUA-KV](https://arxiv.org/abs/2501.19392)
- [Sequential KV compression](https://arxiv.org/abs/2604.15356)
- [SINQ](https://arxiv.org/html/2509.22944)
- [KVarN](https://arxiv.org/html/2606.03458)

## Reproduction

Run each script on both fixed traces:

```bash
python3 research/laplace_kv/prototype_backward_scale.py TRACE.kvt
python3 research/laplace_kv/prototype_gain_servo.py TRACE.kvt
python3 research/laplace_kv/prototype_value_forest.py TRACE.kvt
```
