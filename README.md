# Laplace

## The Apple Silicon inference engine

Laplace is an Apple Silicon inference-engine project written in C++20. It uses
Apple-specific SIMD kernels and an opt-in Metal backend.

## Current public scope

Today, the command-line program accepts one GGUF file. It reads GGUF metadata,
loads supported tensor layouts, tokenizes a prompt, and generates tokens. The
source contains named Gemma 4, Llama-style, Qwen3-Next, and Phi-3 architecture
paths. The public scope does not yet include a universal model loader, an MLX
runtime, or general MoE serving.

Each model and quantization layout needs an explicit correctness test. See
[Support](docs/support.md) for the current compatibility boundary.

The current Metal switch has narrow scope. `LAPLACE_METAL=1` is consulted for
the final output projection. Model layers stay on the existing path. The CLI
does not report per-operation fallback or command-buffer status. Do not use the
switch as a model-wide GPU comparison.

The repository also contains research code for KV storage, streaming, and MoE
residency. Those components are not a blanket quality, memory, or speed claim.

For the detailed contract, see [Architecture](docs/architecture.md),
[Support](docs/support.md), and the [benchmark procedure](docs/benchmarks.md).

## Build Laplace

Build on an Apple Silicon Mac with a current CMake toolchain.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLAPLACE_NATIVE=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Omit `LAPLACE_NATIVE=ON` to avoid host-specific `-march=native`. That choice
does not make a binary portable across Apple Silicon generations. Test the
binary on each target device. CMake can warn when it cannot find OpenMP. That
warning does not make the build invalid.

## Run a GGUF model

Pass a GGUF file with no prompt to inspect its metadata.

```bash
./build/laplace model.gguf
```

Generate text with a prompt.

```bash
./build/laplace model.gguf -p "Hello, Laplace" -n 32 --kv-fp32
```

Use a deterministic, non-speculative command when you compare runs.

```bash
./build/laplace model.gguf -p "Hello, Laplace" -n 32 \
  --greedy --seed 7 --no-spec --max-seq 128 --kv-fp32 --bench
```

The current default is the experimental LaplaceKV mode. With that default, a
configured `--max-seq` above `4096` automatically selects file-backed
streaming. Use `--kv-fp32` for the basic reproducible commands above.

The command accepts these commonly used options:

| Option | Purpose |
| --- | --- |
| `-n N` | Set the maximum number of generated tokens. |
| `-t T` | Set the sampling temperature. |
| `--top-k K` | Set the top-k sampling limit. |
| `--top-p P` | Set the nucleus-sampling limit. |
| `--greedy` | Use greedy token selection. |
| `--seed S` | Set the sampling seed. |
| `--max-seq L` | Set the maximum sequence length. |
| `--no-spec` | Disable speculative decoding. |
| `--bench` | Print prefill and decode timing. |
| `--kv-fp16` | Select the FP16 KV cache. |
| `--kv-fp32` | Select the FP32 KV cache. |
| `--laplace-kv` | Select the LaplaceKV research cache mode. |
| `--laplace-stream` | Request streaming storage for the LaplaceKV research mode. |
| `--laplace-resident` | Request resident storage for the LaplaceKV research mode. |

Run `./build/laplace` without arguments to print basic usage.

## Report results responsibly

Report the model artifact, digest, quantization, and prompt token count.
Report the generated token count and `--max-seq`. Report sampling settings,
build revision, macOS version, hardware, power state, thermal state when
available, and Metal status. Keep prefill and decode results separate. A local
result does not establish a claim for another model, layout, or M-series device.

Follow the [benchmark procedure](docs/benchmarks.md) for the required record.

## Contribute to Laplace

Read [CONTRIBUTING.md](CONTRIBUTING.md) before you change code or documentation.
Use a focused branch for an independent change. Preserve unrelated changes.
Publish performance claims only with reproducible evidence.

## License

Apache-2.0. See [LICENSE](LICENSE).
