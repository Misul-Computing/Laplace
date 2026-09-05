# Architecture

Laplace separates a model's meaning from the way its tensors are stored.
The shared compiler lowers those descriptions into executable programs for
Apple Silicon. This lets execution work apply across compatible model layouts
and stored precisions without using model names as optimization switches.

## Package to execution

```text
Model package
  -> validated artifacts and declared tensor storage
  -> semantic operations and state contracts
  -> semantic program + physical decode programs
  -> verified program package
  -> session-owned Metal execution
```

`ArtifactSet` retains package files and owned blobs. `ArtifactIndex` records
tensor planes, layouts, aliases, bounds, and other physical facts. Semantic
model values and operators describe computation independently of those bytes.

`source_program_compiler.cpp` connects package ingestion to the shared route.
`semantic_program_compiler.cpp` lowers supported model operations to ProgramIR.
`codec_certificate_physical_program.cpp` lowers declared storage arithmetic
into physical decode programs. The resulting package binds programs, resources,
tokenizer contracts, and state schemas before execution is admitted.

Physical loads express how to read tensor values from their stored bytes.
They do not require expanding a whole model into another weight format.
Semantic programs use those values through the same execution machinery.

## Metal session

`program_metal.mm` compiles and executes verified programs through Metal.
Artifact mappings are shared across tensor resources, intermediates are reused,
and eligible final state outputs write into transactional candidate buffers.
Packed loads read bounded byte windows using the declared bit layout.

`RuntimeSession` owns execution state and token history. A successful token
step publishes its updated state; a failed step does not commit the candidate.
Rollback support is constrained by the state generations retained by the
execution route. It is not an unlimited conversation undo mechanism.

Dense prefill and decode share the session. The supported compiler operations
include embedding, projection, normalization, activation, rotary position,
and causal or sliding grouped-query attention, including tied weights.

## Tokenization and conversation

The token program carries vocabulary, normalization, encoding, decoding, stop
IDs, and prompt instructions. Chat framing is derived from a supported
structural subset of the package's template text. The first turn supplies
opening context; later turns close the prior response and append the next
message without starting a new session.

The CLI reuses this session for a conversation and streams decoded text.
Context capacity is fixed when the session is created.

## Extending coverage

The semantic model also represents recurrent and routed expert operations.
Their lowering into the shared execution compiler remains in progress.
GGUF and MLX/SafeTensors ingestion surfaces feed package contracts; ingestion
alone does not establish complete execution support.

The intended rule for new optimizations is structural: use operations, tensor
layouts, state requirements, execution phase, and device capabilities. The
linked source tree still contains transitional category-based paths, tracked
by the failing universality source-policy check. See [Support](support.md)
for current limits and [Benchmarks](benchmarks.md) for qualification requirements.
