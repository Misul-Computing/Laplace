# Laplace

Laplace is the Apple Inference Engine for M-series Macs.

Laplace is being built to load local language models and run them through an
Apple-native Metal pipeline. Its execution core keeps compatible graph work on
the GPU from token embedding to final logits. It reads supported quantized
weights directly and uses Apple unified memory for model data and runtime
state.

## Why Laplace

Laplace builds its runtime plan from the model's operations, tensor layouts,
state, and weight formats. It compiles those facts into one typed graph, then
selects Metal kernels from the active Mac's capabilities.

This design gives Laplace one execution system for compatible dense,
recurrent, and mixture-of-experts (MoE) models. Model names, file names, and
paths do not choose runtime kernels.

## Current alpha

The engine contains these implemented parts:

- Package validation for GGUF, MLX, and SafeTensors files. It checks file
  identity, tensor bounds, planes, layouts, aliases, and digests.
- A versioned model graph with explicit operators, data edges, layers, tensor
  roles, and persistent state.
- A capability planner that matches the full graph to Metal kernels and the
  physical weight formats in the package.
- A session-owned Metal execution core for embedding, attention, recurrent
  state, feed-forward work, expert routing, normalization, and output
  projection.
- Transactional token state. The session advances only after the GPU completes
  the token command.

Laplace also includes direct Metal paths for quantized matrix work, mapped
model weights, device-side routing, and compact result handling.

## How it works

```text
model package
  -> checked tensor index
  -> typed model graph
  -> capability and kernel plan
  -> session-owned Metal resources
  -> token execution
```

The loader describes what the model contains. The planner decides how the
active Apple GPU can execute it. The Metal session owns the command queue,
weight mappings, temporary buffers, and model state for the full run.

The public command-line tool currently provides GGUF metadata inspection. The
strict product loader and Metal runtime are connected through versioned schema,
manifest, tokenizer, planner, and session interfaces.

## Build

Use an Apple Silicon Mac with CMake and the macOS Metal toolchain.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLAPLACE_NATIVE=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Run

Inspect local GGUF metadata:

```bash
./build/laplace /absolute/path/to/model.gguf
```

See [Architecture](docs/architecture.md), [Support](docs/support.md), and
[Benchmarks](docs/benchmarks.md) for the engine design, current capability
surface, and benchmark format.

## License

Apache-2.0. See [LICENSE](LICENSE).
