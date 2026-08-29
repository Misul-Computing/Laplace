# Support

Laplace records support at three levels:

- Implemented means the source and contract tests are present.
- Admitted means the product loader built one complete execution plan.
- Qualified means an exact package and Apple device passed a recorded
  correctness run.

## Current alpha

| Area | Implemented source | Product state |
| --- | --- | --- |
| GGUF | Metadata parsing, checked tensor indexing, closed-schema compiler interface, and typed package construction. | Metadata inspection is available. Product generation waits for an admitted closed schema. |
| MLX and SafeTensors | Checked package and shard loading, tensor indexing, and layout validation. | Physical package loading is implemented. |
| Dense graphs | Attention, feed-forward, normalization, state, planner, and Metal execution components. | No package is admitted in this public revision. |
| Recurrent graphs | Convolution, recurrent state, gated attention, planner, and Metal execution components. | No package is admitted in this public revision. |
| MoE graphs | Router, expert work, activation, down projection, weighted reduction, planner, and Metal components. | No package is admitted in this public revision. |
| Quantized weights | Format-aware planner entries and direct Metal matrix kernels for supported physical tuples. | Admission depends on the full tensor tuple and active device. |
| Token programs | Versioned byte-BPE and SentencePiece programs with bounded parsing, digest binding, and deterministic execution tests. | Package use depends on closed-schema admission. |
| Token state | State descriptors, session ownership, checkpoint, command completion, and commit interfaces. | Admission depends on the graph's complete state contract. |

## Package reports

The product loader returns a structured compatibility report when it cannot
build a complete plan. The report identifies the stage, operator, tensor, and
missing contract or capability.

## V1 direction

V1 focuses on:

- one package-to-graph compiler for GGUF and MLX
- full Metal execution for dense, recurrent, and MoE graphs
- direct support for the physical weight formats used by local model packages
- device-driven kernel selection across M-series Macs
- repeatable throughput and correctness qualification
