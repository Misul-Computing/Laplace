# AGENTS.md

## Project contract

Laplace is a C++20 native macOS arm64 inference engine. The public executable
uses checked package ingestion, semantic model import, capability planning, and
a session-owned canonical Metal transaction. Do not add a model-family dispatch
or CPU execution continuation to the public route.

`CompatibilityReport` is the required failure surface. If a package, semantic
operator, state ABI, physical layout, or Metal capability is not admitted,
return a precise report. Do not silently convert the contract into a fallback.

SafeTensors and MLX loaders currently establish physical package facts only.
They must refuse execution until a semantic certificate or unique proof exists.
MoE execution is disabled pending independent dataflow qualification. Do not
restore it from a synthetic fixture or substitute its residual, payload, or
expert geometry.

## Build and test

Build only on an Apple Silicon Mac:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLAPLACE_NATIVE=ON
cmake --build build
ctest --test-dir build --output-on-failure
git diff --check
```

The configured suite uses no model download. It checks package ownership,
physical ingestion, semantic import, plan refusal, and the public CLI.

The native Metal witness is separate because a sandbox may lack a device:

```bash
cmake --build build --target test_canonical_metal
./build/test_canonical_metal --prefill-batch
```

Do not treat a skipped device test as a Metal qualification.

## Change rules

1. Inspect the active worktree, branch, diff, and source before editing.
2. Add a focused failing test before a behavior change.
3. Keep artifact validation, semantic interpretation, planning, and execution
   separate. Evidence at one stage does not prove the next stage.
4. For Metal changes, test state rollback and failure behavior as well as a
   successful submission. A complete token contains embedding, layer work,
   final normalization, and output projection.
5. Do not claim format-wide, model-wide, device-wide, quality, or performance
   support from one fixture or one local run.
6. Do not download models, add generated binaries, copy private artifacts, or
   record credentials in the repository.

## Publication

Use plain technical prose. State supported behavior and limits together. Keep
benchmark numbers out of public text unless the exact command, artifact,
device, environment, and correctness record are available.

Do not commit, push, create a pull request, publish a result, or add an AI
co-author without the user's explicit approval for that action.
