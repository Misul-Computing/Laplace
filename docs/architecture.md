# Architecture

## Scope of this branch

This branch contains a GGUF command-line runtime. Broader package formats and
automatic model detection are future work. They need source, correctness, and
performance gates before they become public support.

The architecture described here is a source map, not a promise that any model
package will run.

## Input and model construction

`src/gguf.cpp` opens a GGUF file and exposes its metadata and tensors.
`src/model.cpp` builds a `Model` from that data. `src/laplace_arch.cpp` selects
the current named architecture path from `general.architecture` metadata:

| Metadata value | Current path |
| --- | --- |
| `gemma4` | `Gemma4Arch` |
| `llama`, `qwen2`, `qwen3` | `LlamaArch` |
| `qwen3next`, `qwen35` | `Qwen3NextArch` |
| `phi3` | `Phi3Arch` |

The table does not make all layouts with a related name safe. Each named path
also uses fixed tensor-name mappings in its architecture source. If model
metadata or tensor geometry is incomplete, the loader can fail to construct a
model.

`src/safetensors.cpp` is a SafeTensors parser with unit tests. The current CLI
does not accept a SafeTensors or MLX package as its model argument. Support for
those package formats and automatic model detection is future work.

## Execution

The current path executes embeddings, layers, a final normalization, and an
output projection through `Model`. `src/matmul.cpp` selects available matrix
operations. `src/matmul_simd.cpp` provides Apple SIMD implementations. On
Apple platforms, `src/metal.mm` supplies the optional Metal implementation.

`LAPLACE_METAL=1` is consulted only by `matmul_lm_head` for the final output
projection. The existing model layers do not use that switch. The CLI does not
expose per-operation fallback or command-buffer status. A run with the switch
is not a model-wide Metal result.

The command-line generator uses `src/tokenizer.cpp`, `src/sampler.cpp`, and
`src/kvcache.cpp`. It can perform batched prompt prefill and token-at-a-time
decode. It can also use prompt lookup or lookahead speculative decoding when
the sampling mode permits it.

## State and storage experiments

The tree contains research paths for LaplaceKV, recurrence, and MoE residency.
They have separate acceptance conditions because memory reduction, numerical
agreement, and token throughput are different properties. In particular, a
format with a smaller stored payload can still reduce decode speed or change
model output.

Treat the following areas as experimental until a model-level gate qualifies
them for a stated package and configuration:

- LaplaceKV compressed and streaming modes.
- Model-file-backed expert residency in `src/laplace_moe.cpp`.
- Quantization conversion or derived-weight experiments.

## Future work

Future work can broaden package parsing and model detection. It must not make a
new model format public support without an explicit correctness and performance
record. This branch remains a GGUF runtime with limited, source-defined paths.
See [Support](support.md) for the current boundary.
