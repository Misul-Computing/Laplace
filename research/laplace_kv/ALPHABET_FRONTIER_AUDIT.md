# Fixed scalar-alphabet frontier audit

Date: 2026-07-12

## Decision

Rejected as a LaplaceKV foundation. An exhaustive D64/G128 screen enumerated
every separate fixed K/V alphabet from K3 through K16 and V2 through V8, with
independent Q2 through Q8 metadata fields, an actual base-radix group pack no
wider than 64 bits, a 16-byte header, byte rounding, and 64-byte record
alignment. Of 2,879 feasible layouts, 63 were nondominated in alphabet and
metadata precision.

K7/V6 Q8/Q8 was the only frontier point that beat K16/V4 Q8 mean and p95 on
both trace prompts. Its exact record is 3 bits per scalar. A real 1,024-token
Qwen2.5 lifecycle then changed perplexity by +107.86%. The temporary simulator
was removed and no native kernel was built.

The fixed scalar-alphabet family is closed under the current three-bit,
training-free goal. Short fixed-position attention mean and p95 are not safe
selection metrics for autoregressive cache quality.

## Enumerated family

For every candidate, the sweep independently selects the most compact group
size whose packed word is at most 64 bits. It counts:

- separate key and value code streams;
- all partial final groups and byte padding;
- `2D + G` key normalization fields;
- `2G + D` value normalization fields;
- independently selected metadata precision;
- a fixed 16-byte header; and
- 64-byte record alignment.

The normalization and reconstruction are the same deterministic even-dimension
transform and dual-axis variance balancing used by the KVarN control. The
sweep changes only fixed alphabet sizes, metadata precision, and exact pack
grouping. It does not use entropy coding, prompt-selected codebooks, or
future-query information.

Dominance is defined over four integers: key levels, value levels, key
metadata bits, and value metadata bits. A layout is removed when another
three-bit layout is at least as large in all four and strictly larger in one.
This leaves 63 points from 2,879 feasible combinations.

## Trace frontier

The screen keeps tokens 0 through 127 exact, encodes tokens 128 through 255,
and evaluates the 336 queries at the 256-token prefix.

### Regenerated official trace

| Format | Mean | P95 | Maximum |
|--------|-----:|----:|--------:|
| K16/V4 Q8 control | 23.650% | 52.951% | 77.804% |
| K7/V7 Q6/Q4 | 20.097% | 54.332% | 119.229% |
| K7/V6 Q8/Q8 | 20.748% | 50.641% | 123.099% |
| K9/V5 Q7/Q6 | 21.732% | 53.294% | 110.795% |
| K9/V5 Q8/Q5 | 21.920% | 53.761% | 108.735% |
| K8/V6 Q7/Q4 | 21.974% | 57.152% | 116.715% |

### Older prompt trace

| Format | Mean | P95 | Maximum |
|--------|-----:|----:|--------:|
| K16/V4 Q8 control | 24.003% | 50.615% | 77.044% |
| K8/V6 Q7/Q4 | 20.568% | 42.157% | 229.205% |
| K9/V5 Q8/Q5 | 21.622% | 45.205% | 236.670% |
| K9/V5 Q7/Q6 | 21.687% | 45.707% | 299.547% |
| K7/V6 Q8/Q8 | 22.237% | 47.909% | 144.992% |

K7/V6 Q8/Q8 was the unique point in these tables that improved both mean and
p95 against the control on both prompts. Its maximum error was still 1.59x the
official control and 1.88x the older control. It was promoted because the
model lifecycle, not trace maximum alone, is the quality gate.

## Exact K7/V6 record

At D64/G128:

```text
key indices                              8,192
16 base-7 indices per 45-bit word
key stream                  512 * 45 = 23,040 bits = 2,880 bytes

value indices                            8,192
17 base-6 indices per 44-bit word
value stream                482 * 44 = 21,208 bits = 2,651 bytes

Q8 key metadata                 (2D + G) * 8 = 2,048 bits
Q8 value metadata               (2G + D) * 8 = 2,560 bits
fixed header                                      128 bits
metadata and header                              592 bytes

raw record                                     6,123 bytes
64-byte aligned record                         6,144 bytes
K/V scalars                                      16,384
effective rate                             3.000000 bits/scalar
```

The standalone verifier constructs every 45-bit and 44-bit word, decodes the
base-7 and base-6 digits, and compares every index. The last value group and
its byte padding are counted. No entropy state or directory is hidden.

As with the earlier tile formats, this is a complete-tile construction. It
does not solve the partial mutable-tail rate.

## Model lifecycle rejection

A temporary capture-only FP32 simulator applied the exact K7/V6 quantization
rule to every completed tile. Its memory and timing fields are invalid. Only
the quality comparison is used:

| Metric | FP16 | K7/V6 Q8/Q8 |
|--------|-----:|-------------:|
| Perplexity | 99.1984007 | 206.191565 |
| Perplexity change | | +107.857751% |
| Top-1 agreement | | 58.300781% |
| Mean KL | | 0.9639340 |
| P95 KL | | 3.9115109 |
| Maximum KL | | 9.4179179 |

The candidate fails the two-percent limit by more than 53 times. Its better
fixed-position mean and p95 did not predict autoregressive behavior. The
worse maximum error was a more useful warning, but even trace maximum is not a
validated standalone rejection threshold.

## Reproduction

Enumerate and rank the frontier:

```bash
PYTHONPYCACHEPREFIX=/tmp/laplace-pycache \
python3 research/laplace_kv/prototype_alphabet_sweep.py \
  /tmp/laplace-qwen-official-256-all.kvt
```

Verify the selected pack and rerun each prompt:

```bash
PYTHONPYCACHEPREFIX=/tmp/laplace-pycache \
python3 research/laplace_kv/prototype_alphabet_sweep.py \
  /tmp/laplace-qwen-official-256-all.kvt --candidate 7 6 8 8

PYTHONPYCACHEPREFIX=/tmp/laplace-pycache \
python3 research/laplace_kv/prototype_alphabet_sweep.py \
  /tmp/laplace-qwen-256-all.kvt --candidate 7 6 8 8
```
