# Architecture

## Package to plan

The public executable has one route:

```text
local package
  -> ArtifactSet or MLX physical graph
  -> ArtifactIndex
  -> semantic model and LAPIR operators
  -> capability planner
  -> session-owned canonical Metal program
  -> one-token transaction or CompatibilityReport
```

`ArtifactSet` maps only checked regular files and retains their identity and
digest. `ArtifactIndex` records physical facts, tensor planes, aliases, and
diagnostics without treating raw tensor names as execution semantics. The
current generic GGUF importer is experimental structural inference from
normalized metadata and tensor evidence; it is not a proof of arbitrary
activation, normalization, position, residual, or tokenizer semantics.
Universal semantic loading is the architecture goal. Current execution needs
explicit resolver coverage and independent package/device qualification.

SafeTensors and MLX ingestion stop at the physical stage today. This is a
deliberate boundary. A parser must not invent an operator graph or execute
package-supplied code to make a model run.

## Token transaction

The canonical program owns mapped weight registration, Metal resources,
persistent state, command submission, and the token position. The planner
admits a package only when every required operation has one compatible Metal
implementation. A failed plan or command submission returns a
`CompatibilityReport`; it does not continue tensor execution on the CPU.

For an admitted dense token, the transaction contains:

1. Token embedding.
2. Attention projections, position transform, causal attention, and residual.
3. FFN normalization, projections, activation, combine, and residual.
4. Final normalization and LM output projection.

The session commits its position only after the command transaction completes.
An aborted transaction does not publish provisional state to the next token.

The code also has a recurrent lowering surface. It remains capability-gated
and has no published package qualification. MoE admission is disabled pending
an independent dataflow qualification. The runtime must not substitute a
different residual edge, payload formula, or expert geometry to make an MoE
package run.

## Resource rules

Laplace uses Apple unified memory for host-visible mapped artifacts and
session-owned Metal resources. The canonical program registers the exact
retained tensor ranges and checks their coverage. Derived load-time storage is
allowed only when its source and replacement ranges are explicit. It is not a
per-token CPU fallback.

The public session currently uses FP32 global state. Legacy compressed and
streaming KV modes are outside this route. The planner also treats unsupported
physical layouts, state formats, and capability gaps as refusals, not as
requests to reuse an old named architecture implementation.
