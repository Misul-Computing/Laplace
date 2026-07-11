# Pre-RoPE fixed-rate screen

Status: rejected as a LaplaceKV foundation.

This experiment asks whether storing keys before RoPE makes a model-independent,
fixed-rate product quantizer accurate enough at the complete three-bit budget.
The decoder would rotate reconstructed keys by their token position during
attention. Values use the same fixed rotated product codec without RoPE.

## Fully allocated rate

For head dimension `D`, each K/V vector stores one log8 RMS scale. Values use
two bits per scalar. Keys use seven-bit two-dimensional codes, with
`D / 2 - 16` pairs promoted to eight-bit codes and spread deterministically
across the vector. The cache payload is therefore:

```text
K codes: 7 * D/2 + (D/2 - 16) = 4D - 16 bits/vector
V codes: 2D bits/vector
scales:  16 bits per K/V vector pair
total:   6D bits for 2D scalars = 3.0 bits/scalar
```

The trace screen treats the fixed decoder tables as executable constants and
does not count their 816-byte reference representation. A production candidate
would need an analytic table or a slightly lower code rate to count that memory.
This optimistic omission makes a large quality failure decisive.

Both K and V first use the deterministic even-dimension orthogonal transform.
The two-dimensional codebooks are fixed Gaussian Lloyd-Max tables and do not
use model, layer, head, or prompt calibration.

## D64 trace result

Input: `/tmp/laplace-qwen-256.kvt`, 336 queries. Qwen2.5 uses full D64 RoPE
with base 1,000,000. Errors are attention-output relative errors against the
uncompressed trace.

| Candidate | Mean | P95 | Maximum |
|-----------|-----:|----:|--------:|
| Post-RoPE K/V | 85.536% | 224.499% | 1154.711% |
| Pre-RoPE K/V | 68.484% | 147.464% | 443.534% |
| Pre-RoPE K, exact V | 59.216% | 171.289% | 652.382% |
| Exact K, two-bit V | 39.082% | 75.421% | 85.368% |
| Pre-RoPE K centered on first 128 tokens | 60.676% | 133.403% | 627.014% |

Pre-RoPE storage improves the fixed codec, but the remaining error is far too
large to justify the additional position-dependent rotation work in the Apple
decode kernel. Exact-key results also show that the two-bit value codec is an
independent bottleneck. A common post-RoPE key anchor and exactly corrected
value anchor did not improve the result.

## Novelty

Pre-RoPE key quantization is established prior art. KVQuant explicitly
introduces it to avoid RoPE-induced quantization difficulty:
[KVQuant](https://arxiv.org/abs/2401.18079). RotateKV also uses pre-RoPE
grouped-head rotation:
[RotateKV](https://arxiv.org/abs/2501.16383). The fixed product quantizer also
falls inside the broad vector-quantization space occupied by FibQuant and
other rotation codecs.

The exact common-key and common-value translation identities remain useful
attention invariances, but this ablation does not create a quality or novelty
result.

Reproduce with:

```bash
python3 research/laplace_kv/prototype_prerope_fixed.py \
  /tmp/laplace-qwen-256.kvt
```
