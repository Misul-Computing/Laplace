# Architecture

Laplace converts a model package into a complete Metal execution plan. The
same pipeline handles compatible dense, recurrent, and mixture-of-experts
(MoE) graphs.

## 1. Package index

The loader opens each package file once. It records file identity, size, and
digest. It also checks every tensor span, plane, layout, alias, and physical
weight format.

GGUF, MLX, and SafeTensors packages use this physical validation layer.

## 2. Model graph

The compiler converts package facts into a typed graph. The graph records:

- operators and their input and output edges
- tensor roles, shapes, layouts, planes, and weight formats
- layer boundaries and execution order
- persistent attention, recurrent, routing, and sampler state
- tokenizer and prompt contracts

Runtime kernel selection uses these typed facts. Model names, file paths, and
artifact hashes stay outside the execution policy.

## 3. Metal plan

The planner compares the full graph with the active device. Each plan entry
binds one graph operation to:

- a Metal kernel and execution phase
- exact tensor spans and physical formats
- state inputs and outputs
- device capabilities and resource limits

The planner accepts a package when every required operation has a compatible
entry.

## 4. Metal session

Each model run owns its Metal command queue, weight registrations, temporary
buffers, logits, and persistent state. The session uses Apple unified memory
for mapped model data and shared resources.

The runtime registers the exact source ranges required by the plan. Derived
weights use session-owned storage with explicit source and replacement ranges.

## 5. Token transaction

A token step starts from the last committed state. Laplace encodes the full
Metal command, submits it, and waits for completion. For an admitted state
contract, the session then publishes the new token position and state.

This boundary covers key-value state, recurrent matrices, router worklists,
sampler state, and token history.

## Execution graphs

Dense graphs contain attention and feed-forward operations. Recurrent graphs
add convolution and state-update operations. MoE graphs add router selection,
expert work, activation, and weighted reduction.

All three graph classes use the same package index, model graph, planner,
resource ownership, and token transaction.
