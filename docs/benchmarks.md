# Benchmarks

A Laplace benchmark records one exact package, device, revision, and command.
Each result applies to that recorded configuration.

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

Before publishing a performance result, run an independent token or logits
comparison for the exact package. Record the reference implementation, error
bound, token range, and state behavior. A fast result is useful only when the
token transaction completed with the expected state and route.

Record the package digest, command, report fields, and device state with every
result. Laplace publishes benchmark figures only with this complete record and
a repeatable command.
