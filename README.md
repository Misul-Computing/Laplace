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

Talk to a model:

```bash
./build/laplace /absolute/path/to/model.gguf
```

That opens an interactive chat after the model and Metal programs load. Each message is framed with the model's own
chat template, the conversation history carries across turns, and Ctrl-D
exits. If the package's template is not one of the recognized shapes, the
session says so and continues with plain text framing.

Run one fixed response with phase-separated measurement:

```bash
./build/laplace /absolute/path/to/model.gguf \
  -p "Hello, Laplace" -n 32 --greedy --seed 7 --max-seq 2048 --bench
```

The `--bench` option reports prefill and decode separately. `--info` prints
package metadata, and `--raw-prompt` bypasses the chat template.

Run `./build/laplace --help` for the full option list with defaults.

## Models

The normal model route compiles declared tensor storage and semantic operations
into one shared Metal program. It currently executes embedding, projection,
normalization, activation, rotary position, and causal or sliding attention
operations, including tied weights and mixed stored precisions. Graphs needing
unsupported operations fail with a compatibility report; recurrent and routed
expert execution through this compiler is still in progress.

Chat templates are compiled from the package's own template text. The first
turn includes its opening context; later turns close the preceding response
and reuse the existing session without repeating the start token. The context
limit is fixed for a session; when it fills, start a new session.
Fetch a small test model and talk to it:

```bash
python3 scripts/download_model.py
./build/laplace models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

Inspect any GGUF without the engine:

```bash
python3 scripts/gguf_header.py /absolute/path/to/model.gguf
```

Packages fail closed with the reason when a contract is not expressible:
windowed sources without a declared layer pattern default to all-window
attention (exact within the window), and a source needing a non-unit
embedding scale or a non-SiLU activation declares it through the
`embedding_scale` and `feed_forward_activation` metadata keys. Sources whose
tokenizers were trained with BPE merge ranks (serialized in GGUF as pieces
and scores only) use longest-piece segmentation; merge ranks are not part of
the GGUF contract.

See [Architecture](docs/architecture.md), [Support](docs/support.md), and
[Benchmarks](docs/benchmarks.md) for the execution model, support levels, and
measurement contract.

## License

Apache-2.0. See [LICENSE](LICENSE).
