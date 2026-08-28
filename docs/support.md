# Support

## Support levels

Laplace uses three support levels.

| Level | Meaning |
| --- | --- |
| Implemented | The checked-in source contains a path for the stated input or operation. |
| Qualified | A named artifact and configuration passed the required correctness gate. |
| Released | The project documents the behavior as a public support commitment. |

Implemented does not mean qualified or released. A benchmark does not extend
support to another quantization, model revision, or hardware generation.
This support document lists no qualified model artifact or configuration.

## Package inputs

| Input | Current level | Notes |
| --- | --- | --- |
| One GGUF file | Implemented | The public CLI accepts a GGUF path. The loader checks only the contracts implemented by the selected path. |
| Sharded GGUF | Not released | Do not assume that the CLI accepts a GGUF shard set. |
| SafeTensors file or index | Parser only | Unit-tested parsing exists, but the public CLI does not load SafeTensors models. |
| MLX directory or package | Future work | Package parsing, model detection, and execution need implementation and qualification. |
| Remote model code, Python hooks, or globs | Unsupported | The runtime must not execute package-provided code to infer a model. |

## Model layouts

The source has paths for Gemma 4, Llama-style, Qwen3-Next hybrid, and Phi-3
layouts. The exact metadata mapping is in [Architecture](architecture.md).
This list describes source coverage. It does not certify every exported model
with a related marketing name.

The source also contains MoE-related code. It does not make every MoE model
compatible. Router dimensions, expert tensor layouts, quantization planes,
state layout, and output handling need an explicit model-level test.

## Quantization and numerical behavior

The source implements several GGUF quantization kernels. Kernel presence is
not format-wide support. A valid route must match the tensor role, logical
shape, physical layout, quantization type, and storage planes.

For an unsupported matmul format, the current runtime prints
`matmul: unsupported weight type`, zeroes the output, and continues. Treat that
run as failed. Do not use its generated text or timing as evidence.

Compare generated token IDs or logits against an independent reference before
you report numerical agreement. Check multi-token state behavior, not only the
first token. Record the error bound and the exact artifact.

## Metal and Apple Silicon

Metal has narrow public scope in this branch. `LAPLACE_METAL=1` is consulted
only for the final output projection. Model layers stay on the existing path.
The CLI does not report per-operation fallback or command-buffer status. Do not
label an environment-variable run as a model-wide GPU result.

Do not generalize a result from one Apple Silicon generation to another. Apple
GPU, memory, and storage capabilities vary by device configuration.

## Report an unsupported input

When an input fails, report the package format and full digest. Report relevant
metadata, tensor role when known, logical shape, physical layout, quantization
type, error text, and the exact command. Do not attach proprietary weights or
access tokens. Prefer a small synthetic fixture for a reproducer.
