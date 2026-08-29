# Support

Laplace reports support per package contract and device capability. The table
summarizes present capabilities and development focus.

| Capability | Current level | What the tree provides |
| --- | --- | --- |
| Physical artifact indexing | Implemented in tree | Checked files, tensor planes, aliases, layouts, bounds, and digests. |
| Versioned semantic model | Implemented in tree | Typed tensors, values, operators, layers, state, constraints, capability requirements, and tokenizer/template digests. |
| Capability-aware planning | Implemented in tree | Complete plan entries with kernel, tensor, state, phase, and resource contracts. |
| Transactional session state | Implemented in tree | Session-owned state with checkpoint, commit, and rollback surfaces. |
| Dense canonical Metal token | Active alpha | Embedding, attention, feed-forward, normalization, and output projection in the canonical session path. |
| Dense prefill and decode | Active alpha | Shared session state and phase-specific Metal transactions. |
| GGUF package route | Active alpha | Physical artifact validation, metadata and tensor import, semantic graph construction, and native planning components. |
| MLX and SafeTensors ingestion | Active alpha | Physical package and shard validation are present in the ingestion surface. |
| MoE semantic primitives | Active alpha | Routed operators, expert-axis tensor contracts, top-k semantics, and weighted expert reduction. |
| Recurrent semantic primitives | Active alpha | Convolution, delta-matrix, gated-attention, and state-update operators in the semantic and planning surfaces. |
| Native M-series qualification | Active alpha | Capability queries, Metal resource checks, exact package/device records, and reproducible measurement gates. |

Support is determined by the complete typed package, physical format, semantic
contract, and active device capability.
