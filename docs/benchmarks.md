# Benchmarks

A Laplace benchmark is an exact package, device, and command result. It is not
a hardware claim and it does not establish support for another package.

## Before the run

1. Build the recorded revision in Release mode and save `git status --short`.
2. Record the model path, full SHA-256, file size, package format, and tensor
   layouts that the plan admitted.
3. Record the Mac model, unified-memory size, macOS version, power source, and
   thermal condition when available.
4. Record the prompt or prompt digest and its token count.
5. Use fixed sampling. `--greedy --seed 7 --no-spec` is the default comparison
   command for an admitted package.

## Command

```bash
./build/laplace /absolute/path/to/model.gguf \
  -p "fixed prompt" -n 128 --greedy --seed 7 --no-spec \
  --max-seq 2048 --bench
```

Save standard output and standard error. The CLI prints prefill and decode
separately and records the Metal command-buffer count for each phase. A report
must include both phases. Do not combine them into one token-per-second value.

## Correctness and routing

Before you publish a performance result, run an independent token or logits
comparison for the exact package. Record the reference implementation, error
bound, token range, and state behavior. A fast result is not useful if a
token transaction failed, changed state incorrectly, or selected a fallback.

If the CLI reports `CompatibilityReport`, record the code, phase, operator,
tensor, and detail. Do not change the report into a benchmark result. A missing
Metal device in a sandbox is an environment limit, not evidence that an Apple
Silicon desktop will behave the same way.

Laplace publishes no benchmark figure in this repository revision. Add one
only with the complete record above and a repeatable command.
