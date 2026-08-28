# Laplace

Laplace is the Apple Inference Engine, made for Apple Silicon Macs. It is
being built to maximize efficiency, throughput, and performance across
M-series Macs.

Laplace turns a local model package into a semantic execution plan and runs
admitted work in a native Metal session. The project focuses on low-copy
loading, unified memory, capability-aware kernels, and complete token
execution.

## Current work

Laplace is being developed as a universal semantic loader and Apple-native
inference runtime.

- GGUF import reads metadata and tensor evidence, then derives a semantic
  model for planning.
- MLX and SafeTensors ingestion checks package files, shard maps, and tensor
  layouts before semantic execution.
- Dense token execution covers embedding, attention, feed-forward layers,
  normalization, and output projection on Metal when the complete plan is
  admitted.
- The semantic model includes routed and expert operators for
  mixture-of-experts (MoE) support. Current Metal planning rejects MoE
  packages until the full dataflow is qualified.
- Apple-native runtime work covers memory-mapped weights, unified-memory
  resource registration, core topology, power source, thermal state, and
  runtime metrics. Power policy and M-series tuning remain in development.

## Status

The public route is in active development. It runs a GGUF package only when
import and semantic planning succeed. Tensor formats and the active Metal
device must also satisfy the complete plan.

MLX and SafeTensors packages currently stop after physical ingestion. They
return `IMPORT_SEMANTICS_MISSING`. MoE packages return `KERNEL_UNAVAILABLE`
until their Metal dataflow is qualified. Unsupported package or device
conditions return a `CompatibilityReport`.

The current session owns FP32 global state. Streamed and compressed key-value
cache modes are outside the public route.

## Build

Build on a native Apple Silicon Mac with CMake and the macOS Metal toolchain.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLAPLACE_NATIVE=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Run

Inspect a package and create its Metal plan:

```bash
./build/laplace /absolute/path/to/model.gguf
```

Generate text with fixed sampling settings:

```bash
./build/laplace /absolute/path/to/model.gguf \
  -p "Hello, Laplace" -n 32 --greedy --seed 7 --max-seq 2048 --bench
```

The `--bench` option reports prefill and decode separately. It records the
Metal command-buffer count for each phase. The command does not download a
model.

See [Architecture](docs/architecture.md), [Support](docs/support.md), and
[Benchmarks](docs/benchmarks.md) for implementation details, supported input
states, and measurement requirements.

## License

Apache-2.0. See [LICENSE](LICENSE).
