# Support

Laplace reports support per package contract, not per marketing family name.
`Implemented` means source exists. `Admitted` means the package passed
artifact checks, semantic import, and complete-plan selection on the current
runtime. `Qualified` requires a separate recorded correctness result for an
exact artifact and device.

| Input or feature | Current level | Public behavior |
| --- | --- | --- |
| One regular GGUF file | Admitted only by exact plan | The CLI maps it through `ArtifactSet`, imports semantic operators, and creates a full Metal plan or reports a refusal. |
| SafeTensors file | Physical ingestion | The parser and `ArtifactIndex` validate the physical tensors. Execution refuses because no semantic certificate is accepted. |
| Closed MLX package | Physical ingestion | The loader validates the declared shard graph without executing package code. Execution refuses because no semantic certificate is accepted. |
| Dense causal token | Admitted only by exact plan | Metal executes the embedding, attention, FFN, final normalization, and output projection in the session transaction. |
| Dense prompt prefill span | Admitted only by exact plan | The CLI submits one span. The native F16 witness covers two initial tokens; other lengths remain capability-gated. |
| Recurrent prompt prefill | Metal-only token sequence | Recurrent convolution or delta-matrix state is submitted one token at a time in the same session until candidate-state chaining is admitted. |
| Recurrent token | Candidate surface | It must satisfy exact semantic and pipeline checks. No package or device qualification is published. |
| MoE token | Unsupported | `KERNEL_UNAVAILABLE`. The former synthetic MoE slice is not qualified. |
| Compressed or streamed KV | Unsupported | The canonical route owns FP32 global state only. |

## Failure contract

The CLI prints a `CompatibilityReport` with a stable error code and phase.
Common examples are:

- `IMPORT_SEMANTICS_MISSING`: a physical SafeTensors or MLX package has no
  accepted semantic proof.
- `KERNEL_UNAVAILABLE`: the semantic model needs an operation that has no
  admitted complete Metal implementation.
- `CAPABILITY_MISSING`: the active Apple device cannot create the required
  Metal resource or pipeline.
- `FALLBACK_FORBIDDEN`: a complete plan would require a non-Metal execution
  continuation.

Do not treat a parse result, a model-family label, a single kernel result, or
a one-device run as support for another package or device. Report the package
digest, command, report fields, and device state when you file an issue.
