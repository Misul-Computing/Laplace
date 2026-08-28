# Benchmarks

## Purpose

Use this procedure to make a local measurement reproducible. It does not make
a comparative or product claim by itself.

Measure prefill and decode separately. Prefill processes a prompt batch.
Decode normally processes one generated token at a time and often has different
memory behavior.

## Prepare the run

1. Build the exact revision in Release mode.
2. Record the revision and whether the source tree has local changes.
3. Record the model filename, full SHA-256 digest, package format,
   quantization, and file size.
4. Record the Mac model, M-series chip, unified-memory size, and macOS
   version. Record the battery or AC state. Record thermal state when the
   system exposes it.
5. Use a fixed prompt and record its token count after tokenization.
6. Use fixed sampling settings. Use `--greedy --seed 7 --no-spec` when you
   need a deterministic decode comparison.

## Run the CLI benchmark

Run this command after you replace the model path and prompt with the recorded
values:

```bash
./build/laplace /absolute/path/to/model.gguf -p "fixed prompt" -n 128 \
  --greedy --seed 7 --no-spec --max-seq 2048 --kv-fp32 --bench
```

The program writes prefill and decode timing to standard error. Keep the raw
output. The benchmark line reports tokens, milliseconds, and tokens per second
for each phase.

Do not use `LAPLACE_METAL=1` as an end-to-end GPU comparison. In this branch,
the switch is consulted only for the final output projection. The CLI does not
report operation routing or command-buffer status.

## Benchmark record

Include this information in a result report:

- Source: Git revision, source-tree state, compiler, and build options.
- Model: Format, full digest, size, quantization, and architecture metadata.
- Device: Mac model, chip, unified memory, and macOS version.
- Environment: AC or battery, battery percentage, and thermal state when
  available.
- Workload: Prompt text or digest, prompt token count, generated token count,
  and `--max-seq`.
- Decode settings: Temperature, top-k, top-p, seed, and speculative-decoding
  mode.
- Backend: Record the Metal request. State that the public CLI does not report command-buffer outcome or per-operation fallback.
- Timing: Warm-ups, samples, median, range, prefill, and decode.
- Correctness: Reference method, error bound, and token or top-1 comparison.

The displayed decode count includes the first token derived from the final
prefill logits. That token has no decode forward. Do not use a short run as
headline decode evidence.

An internal qualification harness can alternate control and candidate samples
in one process after warm-up. The public CLI does not provide that A/B harness.
Preserve raw logs. Discard and label a run if power, thermal state, model
mapping, or execution route changed during the comparison.

## Interpretation

Tokens per second is a workload result, not a hardware property. It changes
with the model, quantization layout, context length, prompt length, sample
count, cache state, backend route, and device condition. Do not compare a
single-token kernel time to an end-to-end generation rate. Do not combine
prefill and decode into one headline rate without reporting both components.
