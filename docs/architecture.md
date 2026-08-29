# Architecture

The current alpha compiles compatible GGUF packages into a capability-checked
semantic program for Apple Silicon. The runtime keeps package facts, graph
meaning, physical tensor contracts, device capabilities, and mutable session
state as separate pieces of one route. The V1 design applies this architecture
to GGUF and MLX package contracts.

## Package to transaction

```text
compatible GGUF package
  -> physical artifact index
  -> versioned semantic graph and state contracts
  -> capability-aware execution plan
  -> session-owned Metal resources
  -> transactional prefill or decode
```

`ArtifactSet` retains checked regular files and their digests. `ArtifactIndex`
records tensor planes, aliases, layouts, and other physical facts. The
semantic model then carries tensors, values, operators, layers, state,
constraints, capability requirements, and tokenizer/template digests in a
versioned form.

The planner binds each planned operator to a kernel descriptor, tensor spans,
state bindings, and a phase. Runtime selection uses semantic graph facts,
physical format, execution phase, queried Apple capabilities, session
resources, and planner cost entries. Model names, paths, and hashes stay
outside kernel selection.

## Metal session

The canonical Metal program owns mapped weight registration, Metal resources,
command submission, and token position. Dense token execution covers
embedding, attention, feed-forward layers, normalization, and output
projection. Dense prefill and decode share the same session-owned state and
transaction boundary.

The session publishes a new position only after its command transaction
completes. Checkpoint, commit, and rollback are part of the runtime surface,
so mutable token effects have one explicit ownership model.

## Dense, MoE, and recurrent primitives

The semantic graph includes routed operators, expert axes, top-k selection,
routed linear work, activation, and weighted expert reduction. Expert-bank
physical contracts use the same typed graph and tensor contract system as
dense layers.

Recurrent convolution, delta-matrix, gated-attention, and state-update
operators are represented in the semantic model and planner.

## Resource model

Laplace uses Apple unified memory for host-visible mapped artifacts and
session-owned Metal resources. Exact retained tensor ranges, layouts, planes,
and state bindings are checked before execution. Capability queries drive the
plan, allowing the same semantic machinery to adapt across compatible Apple
Silicon devices.
