# Frozen weight metric extraction

This is the input audit for the completed weight-only spectral screen. It does
not establish a LaplaceKV quality, speed, storage, or novelty claim. The
resulting rejection is in `GENERALIZED_WEIGHTED_AUDIT.md`.

## Source model

The input is the official Qwen2.5 0.5B Instruct Q4_K_M GGUF downloaded from
the Qwen model repository:

```text
/tmp/qwen2.5-0.5b-instruct-q4_k_m.gguf
491400032 bytes
SHA-256 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
```

The GGUF metadata and every layer agree on:

```text
architecture     qwen2
layers           24
hidden           896
query heads      14
KV heads         2
GQA group        7 contiguous query heads per KV head
head dimension   64
context          32768
RoPE dimension   64
RoPE base        1000000
```

The model's Q and O projection tensors are `Q5_0 [896,896]` in all 24
layers. They are not Q4_K tensors. Q4_K_M is the file-level quantization
recipe, and tensors whose 896-element rows do not fit 256-element K-quant
blocks use Q5_0. Attention RMSNorm gains and Q biases are F32. The extractor
uses `GGUFContext` and `Laplace::dequantize`, then checks one complete Q and O
projection against the scalar runtime matmul path. This preserves the mixed
geometry actually consumed by Laplace.

The shape interpretation follows the checks in `arch_llama.cpp`: GGUF
`dims[0]` is the input dimension and `dims[1]` is the output dimension. Q is
therefore decoded as `[14*64 output, 896 input]`. O is decoded as
`[896 output, 14*64 input]`. The engine maps query head `h` to KV head
`floor(h/7)`, so each metric aggregates the same seven contiguous heads.

## Metrics

For layer `l`, KV head `k`, its seven query heads `G(k)`, attention gain
vector `g`, Q block `Wq_h`, Q bias `b_h`, and O input-head block `Wo_h`, the
extractor computes

```text
Cq(l,k) = (1 / (7*D)) sum_h in G(k)
           [Wq_h diag(g^2) Wq_h^T + b_h b_h^T]

Hk_abs(l,k) = (1/P) sum_t=0..P-1 R_t Cq(l,k) R_t^T

Hv_diag(l,k) = (1/7) sum_h in G(k) Wo_h^T Wo_h
```

Here `D=64`, `P=32768`, and `R_t` is the exact NeoX-style rotation used by
`ops::rope_apply`. The `1/D` in `Cq` is the square of the engine's attention
score scale `1/sqrt(D)`. The elementwise attention RMSNorm gain and Q bias
outer product are included. The proxy assumes the scalar RMS-normalized
hidden vector is zero-mean with identity second moment before the learned
gain. This is a frozen-weight assumption, not an activation measurement.

`Hk_abs` is one explicit absolute-query-position average over the full GGUF
context. It is suitable as a constant first-screen metric for post-RoPE key
error. It is not the exact relative-position metric for a cached row at
position `s`. That metric remains row-dependent:

```text
Hk_rel(s) = R_s [E_delta R_delta Cq R_delta^T] R_s^T
```

The extractor does not silently treat these as equivalent.

`Hv_diag` uses every matching GQA O block but is an equal-head diagonal-block
proxy. A shared KV value error reaches O as a sum across query heads with
different attention weights. Its exact metric contains cross-head terms
`E[a_h a_j] Wo_h^T Wo_j`, and frozen weights alone do not define those
attention-weight moments. The causal trace and model lifecycle later rejected
the resulting proxy, as recorded in `GENERALIZED_WEIGHTED_AUDIT.md`.

Qwen2.5 has no per-head Q RMSNorm tensor. The extractor rejects a model that
does, since that nonlinear normalization is outside this linear proxy.

## Output and validation

Build and run:

```bash
xcrun clang++ -std=c++20 -O3 -Wall -Wextra -Werror -Isrc \
  research/laplace_kv/extract_weight_metrics.cpp \
  src/gguf.cpp src/mmap.cpp src/matmul.cpp \
  -framework Accelerate -framework IOKit -framework CoreFoundation \
  -o /tmp/extract_weight_metrics

/tmp/extract_weight_metrics \
  /tmp/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  /tmp/qwen2.5-weight-metrics.lwm
```

The little-endian version-2 output starts with a 48-byte `LWPSD1` header. It
then contains three float64 sections:

```text
metrics       [key,value][layer][kv_head][row][column]
eigenvalues   [key,value][layer][kv_head][component]
eigenvectors  [key,value][layer][kv_head][component][coordinate]
```

Eigenvalues are divided by the matrix trace and sorted from largest to
smallest. Each corresponding eigenvector is a row with unit L2 norm. Its sign
is fixed by requiring its largest-absolute coordinate to be positive. For a
post-RoPE key or value residual row `x`, coefficient `i` is therefore
`dot(eigenvectors[kind,l,h,i], x)`. Reconstruction uses the same row basis.
This convention directly supports fixed-basis per-token residual codecs,
including a test that allocates a fixed total coefficient count across K and
V. The expected file size is 6,340,656 bytes.

The extractor checks its closed-form RoPE average against direct matrix
conjugation, validates every tensor name and shape, rejects non-finite decoded
weights, compares dequantization with runtime matmul, checks symmetry,
requires every matrix to pass unjittered Cholesky, and reconstructs each
trace-normalized metric from its emitted eigenbasis within `1e-11`. An independent NumPy
`eigvalsh` check reads the binary layout and verifies every matrix is positive
definite.

Verified output:

```text
/tmp/qwen2.5-weight-metrics.lwm
6340656 bytes
SHA-256 70144b19ec6e7d84d147402f2644e9ec6674accc76709fa18a0991519b659b7d
maximum symmetry error       4.441e-16
minimum Cholesky pivot       2.456780e-04
independent minimum eigenvalue 9.171171e-05
key metric trace range       [1.537819e+00, 8.111728e+01]
value metric trace range     [8.208353e+00, 2.740067e+01]
eigenvalue sum range         [0.9999999999999996, 1.0000000000000004]
maximum basis orthogonality error 7.550e-15
maximum basis reconstruction error 2.220e-15
```

The trace file currently has no model hash, so pairing it with this GGUF is a
provenance assumption unless the trace is regenerated or its format is
extended.
