# Benchmarks

A Laplace benchmark records one model package, one Mac, one source revision,
and one command. This keeps speed results reproducible.

## Record the run

Record these values before execution:

1. Git revision and working-tree status.
2. Model file size, format, and full SHA-256 digest.
3. Mac model, unified-memory size, and macOS version.
4. Power source and thermal state when macOS reports them.
5. Prompt, token count, context limit, and sampling settings.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLAPLACE_NATIVE=ON
cmake --build build
```

## Run

After the product loader admits a package, use a fixed command:

```bash
./build/laplace /absolute/path/to/model.gguf \
  -p "fixed prompt" -n 128 --greedy --seed 7 \
  --max-seq 2048 --bench
```

Run the command at least five times under the same device conditions. Report
the median and full range.

## Report

Keep these measurements separate:

- model load and session construction
- prompt prefill
- raw decode tokens per second
- Metal command buffers per token
- GPU time and wall time
- mapped model bytes and session-owned bytes

Record the output token IDs or a checked logit comparison for the same run.
Also record the selected plan and any compatibility report.

Requested tensor bytes describe kernel addressing. They are not a hardware
memory-bandwidth measurement. Report measured bandwidth only when a supported
hardware counter provides it.
