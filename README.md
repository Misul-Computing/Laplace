# Laplace

Laplace is an Apple Silicon inference engine for engineers who need a local
model path they can inspect. It maps a local package, derives an experimental
structural semantic model from checked evidence, selects one complete Metal
plan, and executes an admitted token in a session-owned Metal command buffer.

The important boundary is simple: a package either has an admitted complete
plan, or Laplace returns a `CompatibilityReport`. It does not select a
model-family implementation by name. It does not run model layers on the CPU
after a Metal error. It does not use an LM-head-only GPU handoff.

## Why Laplace exists

Conventional local runtimes commonly dispatch by model family and can cross
between CPU and GPU as they go. Laplace turns checked package evidence into
one Apple-native Metal token transaction over unified memory. There is no
layer-by-layer transfer and no hidden CPU continuation after a Metal failure.

New architectures can map to semantic operators instead of adding another
family executor. When Laplace cannot prove a complete contract, it refuses
deterministically instead of producing plausible but incorrect output. This
is an architectural boundary, not a published speed claim.

## Current contract

Laplace builds and runs on native macOS arm64 Apple Silicon. The public route
uses unified memory for mapped weights and session-owned Metal resources.
For every admitted token, Metal executes the embedding, layer operations,
final normalization, and output projection. A failed submission aborts the
transaction. The CLI reports the refusal code, phase, operator, tensor, and
detail instead of continuing on another backend.

The loader has separate stages:

- A GGUF file can become executable only after checked artifact mapping,
  semantic import, and canonical planning.
- A SafeTensors file and an MLX package are parsed into a checked physical
  artifact index. They currently have no accepted semantic certificate, so
  the CLI refuses execution with `IMPORT_SEMANTICS_MISSING`.
- Mixture-of-experts execution is disabled pending dataflow qualification.
  Laplace refuses it with `KERNEL_UNAVAILABLE`.
- The public session currently owns FP32 global state. Compressed, resident,
  and streamed LaplaceKV modes are not admitted by this route.

Generic GGUF resolution is experimental structural inference, not a proof of
arbitrary package semantics. Universal semantic loading is the architecture
and goal; current execution requires explicit resolver coverage plus
independent package and device qualification. This is intentionally not a
claim that every GGUF, SafeTensors, MLX package, quantization format, model
family, or Apple GPU is supported.

The current tokenizer and chat-template behavior is also experimental and
approximate. It is initialized from the retained validated package bytes, not
by reopening the path. Initialization failures refuse with
`TOKENIZER_RUNTIME_UNSUPPORTED`, but successful CLI generation is not a
tokenizer-conformance qualification.

The CLI submits dense prompts as one prefill span; it does not loop over tokens
to hide a missing dense batch contract. The checked native F16 witness covers
two initial tokens. For semantic models that declare recurrent convolution or
delta-matrix state, the CLI submits one token at a time through the same
session-owned Metal route until candidate-state chaining is admitted. This is
not a CPU fallback. Other prompt spans remain capability-gated and return a
report when they are not admitted.

## Build

Use an Apple Silicon Mac with CMake and the macOS Metal toolchain.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLAPLACE_NATIVE=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

`LAPLACE_NATIVE=ON` is the default. It creates a binary for the local Apple
Silicon host. Build and test on each target system before you make a device
claim.

## Run

Inspect the package and create its canonical Metal plan:

```bash
./build/laplace /absolute/path/to/model.gguf
```

Generate with deterministic sampling settings:

```bash
./build/laplace /absolute/path/to/model.gguf \
  -p "Hello, Laplace" -n 32 --greedy --seed 7 --max-seq 2048 --bench
```

`--bench` reports prefill and decode separately. It also reports how many
Metal command buffers the CLI submitted in each phase. It does not convert one
local measurement into a device or model family claim.

The supported CLI options are `-p`, `-n`, `-t`, `--top-k`, `--top-p`,
`--greedy`, `--seed`, `--max-seq`, `--no-chat`, `--raw-channels`, `--no-spec`,
and `--bench`. `--no-spec` is accepted for reproducible scripts. Speculative
generation is not implemented in the canonical route.

The legacy `--laplace-kv`, `--laplace-resident`, `--laplace-stream`,
`--laplace-kv-q4`, `--kv-fp16`, and `--kv-fp32` options fail closed. They do
not select a hidden CPU or cache fallback.

## Verify a native Metal transaction

The normal test suite uses synthetic packages and no model download. It covers
artifact checks, semantic import, compatibility reports, capability planning,
SafeTensors and MLX physical ingestion, and the public CLI refusal contract.

Run this additional native-device witness on an Apple Silicon desktop session:

```bash
cmake --build build --target test_canonical_metal
./build/test_canonical_metal --prefill-batch
```

That witness is separate because a sandbox can compile Metal code without
having access to a Metal device. See [Architecture](docs/architecture.md),
[Support](docs/support.md), and [Benchmarks](docs/benchmarks.md) for the
current boundary and measurement procedure.

## License

Apache-2.0. See [LICENSE](LICENSE).
