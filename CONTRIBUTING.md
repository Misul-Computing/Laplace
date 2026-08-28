# Contribute to Laplace

## Before you edit

Use a focused branch for an independent change. Inspect the status before and
after your change. Do not reset, delete, stage, or reformat files outside your
change.

Read the relevant source before you make a design claim. Treat prior benchmark
reports as hypotheses. Reproduce an observation, identify the affected
contract, and use primary documentation for platform behavior when it matters.

## Change requirements

Keep each change narrow. A runtime change needs:

1. A failing test or focused reproducer that shows the problem.
2. The smallest correction that preserves the stated contract.
3. A passing test that exercises the correction and a counterexample when the
   format or state boundary is ambiguous.
4. A source and behavior check for fallback, ownership, and numerical state.
5. A fresh build, relevant tests, and `git diff --check`.

The public route derives semantics from checked package evidence. Do not add a
model-family switch or use an artifact digest as a runtime selector. Do not
hide a CPU path in a GPU result or convert unsupported quantization data
without reporting it.

## Build and test

Use these commands from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLAPLACE_NATIVE=ON
cmake --build build
ctest --test-dir build --output-on-failure
git diff --check
```

Run the smallest relevant test while you develop. Run the full configured test
suite before you describe a change as complete. Build and run
`test_canonical_metal --prefill-batch` on a native Apple Silicon session when
the change affects the canonical Metal transaction. A skipped device test does
not qualify a Metal feature.

## Benchmark and model work

Follow the [benchmark procedure](docs/benchmarks.md). Keep correctness, memory,
and throughput as separate gates. Use an independent reference for numerical
claims. Preserve a failure log and a passing log. Never replace a failed
measurement with a later result without labeling the earlier result.

Use only model artifacts that you may access and redistribute. Do not upload
weights or expose access credentials. Publish an artifact digest only when the
artifact license and project policy allow it.

## Documentation

Write in direct, active sentences. Use sentence-case headings. State what the
source and evidence show, then state the limit. Keep commands and option names
literal. Do not make an M-series-wide, universal-model, quality, or throughput
claim from one local measurement.

## Git and publication

You can commit to your branch or fork. Obtain maintainer approval before you
push directly to a protected project branch. Obtain approval before you publish
a release or make a benchmark claim on behalf of the project. A pull request
must identify its file list, target branch, validation, and source-deletion
rationale. Do not include unrelated changes.
