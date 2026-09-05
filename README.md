# Laplace

**Apple Inference Engine**

Laplace is built for fast, efficient local inference on Apple Silicon. Its
core idea is a universal model compiler: describe what a model computes and
how its tensors are stored, then compile that description into native Metal
execution on your Mac.

## Why Laplace

**One compiler, shared improvements.** Model operations and tensor storage are
represented separately. The runtime compiles those facts into shared execution
programs, so an improvement to execution can serve different compatible model
layouts and weight precisions. Universality means improving the common path
instead of accumulating a separate engine for every model.

**Apple-native execution and memory.** Metal performs tensor computation while
session-owned resources keep model data and working state close to the GPU.
Artifact mappings and intermediate storage are reused, reducing repeated setup
and allocation during generation.

**A conversation is a persistent session.** Model resources and conversation
state stay alive across turns. Token steps use transactional state updates,
with commit and rollback contracts that keep execution and history aligned.
The package's chat template frames messages, and replies stream to the terminal.

**Performance you can inspect.** Prefill and decode timing are reported
separately. The aim is maximum useful throughput and efficiency on M-series
hardware, with correctness and memory costs measured alongside speed.

```text
Model package
  -> declared operations + tensor storage
  -> validated shared execution program
  -> native Metal session
  -> streamed responses with persistent state
```

Compatible dense models run today, including mixed stored precisions and tied
weights. Dense, recurrent, and routed expert models are part of the wider
architecture; recurrent and expert execution in the shared compiler is still
being built. Laplace is an active alpha, and complete universality remains a
development goal. See [current support](docs/support.md) and the
[architecture guide](docs/architecture.md).

Chat, generate from a prompt or file, and inspect models from one command.
Once your model is downloaded, your prompts and conversation stay on your Mac.

## Get started

You need an Apple Silicon Mac, Apple's command-line developer tools, CMake
3.16 or newer, and Python 3. Metal execution uses the macOS system frameworks.

```bash
git clone https://github.com/Misul-Computing/Laplace.git
cd Laplace
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLAPLACE_NATIVE=ON
cmake --build build --target laplace --parallel 4
```

Use a GGUF you already have:

```bash
./build/laplace /path/to/model.gguf
```

Or download a small starter model, then launch it:

```bash
python3 scripts/download_model.py
./build/laplace models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

The download needs an internet connection and several hundred MB of disk
space. Once the model is on disk, inference runs offline.

After `Ready` appears, type a message and press Enter. Responses stream as
they are generated, and the session remembers earlier turns. Type `/help`
for commands and context usage, or `/exit` to leave. Ctrl-D also exits.

## Everyday use

Generate one response:

```bash
./build/laplace /path/to/model.gguf -p "Explain unified memory in three sentences."
```

Work from a text file and save the response:

```bash
./build/laplace /path/to/model.gguf --prompt-file prompt.txt > answer.txt
```

Inspect a model without starting generation:

```bash
./build/laplace /path/to/model.gguf --info
```

Set the response budget and conversation capacity:

```bash
./build/laplace /path/to/model.gguf -n 512 --max-seq 4096
```

`-n` limits each response, including any reasoning tokens the model emits.
`--max-seq` covers the whole conversation: prompts, replies, and template
tokens. The default is the package's context limit capped at 2048. Larger
contexts use more memory and must fit the package's declared limit. Start a
new session when the context fills.

For repeatable greedy generation with timing:

```bash
./build/laplace /path/to/model.gguf \
  -p "Explain why the sky is blue." -n 64 --greedy --bench
```

`--bench` reports prefill and decode separately. For sampling, use `-t`
(temperature), `--top-p`, `--top-k`, and `--seed`. Use `--raw-prompt` for text
completion without chat framing. Run `./build/laplace --help` for all options.

Responses go to standard output; status and timing go to standard error.

## Install the command

After building, install into your home directory without administrator access:

```bash
cmake --install build --prefix "$HOME/.local"
export PATH="$HOME/.local/bin:$PATH"
laplace /path/to/model.gguf
```

Add the `export` line to `~/.zshrc` to keep the command available in new terminals.

## Development

The quickstart builds the application only. To build and run the test suite:

```bash
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
```

The source universality policy check currently fails while transitional code
remains linked. See [known gaps](docs/support.md#known-gaps) before interpreting
test results. [Benchmark guidance](docs/benchmarks.md) describes how to record
performance and correctness together.

Source is in `src/`, checks in `tests/`, utilities in `scripts/`, and historical
experiments in [`research/`](research/README.md).

## License

[Apache-2.0](LICENSE).
