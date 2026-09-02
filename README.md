# Laplace

Laplace is the Apple Inference Engine for Apple Silicon Macs.

It is built to maximize M-series inference throughput and efficiency through
an Apple-native Metal runtime, unified memory, a universal semantic model
compiler and loader, and automatic capability planning. The same architecture
is designed for dense, recurrent, and MoE models.

The current alpha turns compatible GGUF packages into typed physical facts, a
semantic graph, and a capability-checked execution plan. A session owns its
Metal resources and mutable state, then executes each validated token step as
a transaction. The V1 design applies the same package pipeline to GGUF and
MLX contracts.

## Why Laplace

Apple Silicon puts CPU and GPU work in one unified-memory system. Laplace keeps
the execution model close to that hardware: Metal handles tensor work through
compatible plans, runtime capability queries select those plans, and
session-owned state keeps token execution explicit. See Apple's [unified-memory guidance](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus)
and [Metal compute documentation](https://developer.apple.com/documentation/metal).

Universal means one semantic compiler and execution runtime that selects from
the graph, tensor physical contract, execution phase, and queried device
capability. Model names, paths, and hashes remain metadata.

## What exists today

The current tree contains the core of that architecture:

- Physical artifact validation records checked files, tensor planes, layouts,
  aliases, and digests before semantic loading.
- The semantic model represents tensors, values, operators, layers, state,
  constraints, capability requirements, and tokenizer/template digests in a
  versioned form.
- The capability planner matches complete execution plans against operators,
  physical formats, state contracts, and device capabilities.
- The canonical Metal session provides dense prefill and decode transactions
  with session-owned resources, checkpoint, commit, and rollback state.

Semantic routed operators and expert-axis tensor contracts are present
alongside dense and recurrent graph primitives.

Laplace is under active development and its interfaces may change before V1.

## Architecture

```text
local package
  -> physical artifact index
  -> versioned semantic graph and state contracts
  -> capability-aware execution plan
  -> session-owned Metal resources and state
  -> transactional prefill or decode
```

Every plan entry binds an operator to its tensor and state contract. The
session commits a token position only after the command transaction completes.
This gives the runtime one place to validate physical layout, capability
requirements, resource ownership, and mutable state before work is admitted.

Dense, recurrent, routed, and expert operators share the same semantic graph,
physical contracts, capability planner, and session state model.

## Build

Build on a native Apple Silicon Mac with CMake and the macOS Metal toolchain.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLAPLACE_NATIVE=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Run

Inspect a local package and create its native plan:

```bash
./build/laplace /absolute/path/to/model.gguf
```

Run a fixed native sample with phase-separated measurement:

```bash
./build/laplace /absolute/path/to/model.gguf \
  -p "Hello, Laplace" -n 32 --greedy --seed 7 --max-seq 2048 --bench
```

The `--bench` option reports prefill and decode separately.

Run `./build/laplace --help` for the full option list with defaults.

See [Architecture](docs/architecture.md), [Support](docs/support.md), and
[Benchmarks](docs/benchmarks.md) for the execution model, support levels, and
measurement contract.

## License

Apache-2.0. See [LICENSE](LICENSE).
