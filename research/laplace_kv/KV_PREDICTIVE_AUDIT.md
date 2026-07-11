# Same-token K-to-V predictive residual audit

Date: 2026-07-12

## Decision

Rejected before engine integration. A causal full-matrix predictor fitted on
the first 128 tokens is an invalid prompt-calibrated oracle, yet its held-out
one-bit and two-bit value residuals both lost badly to K4/V2 Q8. Exact keys
with a two-bit residual still measured 49.17% mean attention error. The
fully encodable K4/R1 form measured 141.64%.

A frozen-weight map could generalize differently, but this stronger
request-specific fit provides no evidence that its residual will be
compressible. No model lifecycle, weight derivation, or native kernel is
justified.

## Candidate

K and V for the same token are projections of the same hidden activation. The
screen asks whether a fixed map `A` can predict V from K well enough that only
a one-bit residual must be stored:

```text
V = K A + R
```

For attention weights `w`, direct consumption does not require reconstructing
every V row:

```text
w^T V = (w^T K) A + w^T R
```

The candidate stores a transformed symmetric K4 row, a transformed sign-only
R1 row, and one FP16 scale for each. Its per-token rate is:

```text
K codes                 4D bits
residual signs          1D bits
two FP16 scales          32 bits
effective rate = (5D + 32) / (2D)
```

| Head width | Bits/scalar |
|-----------:|------------:|
| 32 | 3.000000 |
| 64 | 2.750000 |
| 96 | 2.666667 |
| 128 | 2.625000 |
| 256 | 2.562500 |
| 512 | 2.531250 |

This fixed per-token record has no mutable-tail exception. A production
decoder would reread K during value accumulation and perform one `D x D`
matrix product per output head. The matrix itself would also need to be
derived or counted. Those costs are irrelevant after the quality rejection.

## Optimistic causal screen

For each layer and KV head, the prototype fits ridge regression on tokens 0
through 127:

```text
A = (K^T K + lambda I)^-1 K^T V
```

It freezes `A`, encodes tokens 128 through 255, and evaluates only queries at
the 256-token prefix. This uses prompt activations and therefore violates the
training-free gate. It is an optimistic rejection bound, not a deployable
encoder. K and residual rows use the same deterministic even-dimension
transform and FP16 scale rounding as the earlier tail screens.

### Regenerated official trace

| Method | Mean | P95 | Maximum |
|--------|-----:|----:|--------:|
| K4/V2 Q8 | 23.650% | 52.951% | 77.804% |
| Exact K, R1 | 88.532% | 199.999% | 312.716% |
| Exact K, R2 | 49.173% | 121.897% | 235.983% |
| Row K4, exact V | 44.576% | 162.164% | 491.779% |
| Row K4, R1 | 141.641% | 422.708% | 773.722% |
| Row K4, R2 | 102.544% | 363.288% | 877.457% |

### Older prompt trace

| Method | Mean | P95 | Maximum |
|--------|-----:|----:|--------:|
| K4/V2 Q8 | 24.003% | 50.615% | 77.044% |
| Exact K, R1 | 96.287% | 211.466% | 369.051% |
| Exact K, R2 | 56.913% | 133.370% | 225.723% |
| Row K4, exact V | 33.284% | 106.824% | 584.384% |
| Row K4, R1 | 131.279% | 331.808% | 664.922% |
| Row K4, R2 | 93.674% | 263.517% | 1,132.295% |

The predictor does not leave a low-entropy value residual on held-out tokens.
The failure persists with exact keys, so key quantization is not the deciding
problem. R2 already exceeds the K4/V2 reference by more than two times on the
official mean. R1 is unusable.

## Boundary

Same-token K-to-V prediction is distinct from the previously rejected
cross-layer predictor. It uses the shared source activation rather than a
different layer as the anchor. The request-fitted full matrix fails. That does
not mathematically exclude every projection-weight map, but it removes the
evidence needed to justify a model rewrite or hot-path implementation.

Reproduce with:

```bash
PYTHONPYCACHEPREFIX=/tmp/laplace-pycache \
python3 research/laplace_kv/prototype_kv_predictive.py \
  /tmp/laplace-qwen-official-256-all.kvt

PYTHONPYCACHEPREFIX=/tmp/laplace-pycache \
python3 research/laplace_kv/prototype_kv_predictive.py \
  /tmp/laplace-qwen-256-all.kvt
```
