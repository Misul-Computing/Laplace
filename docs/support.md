# Support

Laplace is an Apple Silicon inference engine in active alpha development.
Support depends on the package's declared operations, tensor storage,
tokenizer, and the active device's capabilities.

## What you can use today

- Local terminal chat with streamed replies and conversation history.
- One-shot generation from text or a prompt file, with configurable sampling.
- GGUF model inspection and phase-separated prefill/decode timing.
- Shared Metal execution for compatible dense models, including mixed stored
  precisions and tied embedding/output weights.
- Embedding, projection, bias, normalization, activation, rotary position,
  and causal or sliding grouped-query attention in the shared model compiler.

Chat framing is derived from the package's template. A structural subset of
Jinja templates is supported. When a template cannot be expressed, Laplace
reports the fallback; later turns use plain text framing. Context is fixed for
a session, and conversation state is not saved between launches.

The quickstart model is a convenient starting point, not a model-family
requirement. Other packages enter the same loader and execution path.

## Known gaps

- Recurrent and routed expert operations have semantic representations, but
  their execution is not yet lowered by the shared model compiler. Such
  packages fail with a compatibility report.
- MLX/SafeTensors artifact and shard ingestion exists. This does not establish
  end-to-end support for every package or architecture in those formats.
- The linked product still includes transitional category-based code.
  `test_universal_source_policy` remains a known failing test. The complete
  source tree has not met the universality policy yet.
- A local mixed-weight raw-generation check diverged from the saved llama.cpp
  reference after an initially matching prefix. The cause is unresolved;
  full numerical and generation parity is not claimed.
- Package and device coverage is still being qualified. A successful run on
  one model or Mac does not establish general compatibility or a speedup.

## If a model does not load

Check the path with `laplace /path/to/model.gguf --info`. A compatibility report
means the declared package contract cannot currently enter the requested
execution path. Keep its full text when reporting an issue, together with the
model source, command, Laplace commit, macOS version, and Mac chip.

For a prompt-length error, shorten the prompt or request a larger `--max-seq`
within the model's declared context limit. For a full chat context, exit and
start a new session. For truncated replies, increase `-n` and ensure enough
context remains.

## Package contract details

Artifact validation checks files, tensor planes, aliases, bounds, and digests.
The semantic model describes operators, state, constraints, and tokenizer
contracts. Metal sessions own their resources and expose checkpoint, commit,
and rollback state.

Windowed sources without a declared layer pattern default to all-window
attention. A source requiring a non-unit embedding scale or a non-SiLU
activation declares it through `embedding_scale` and
`feed_forward_activation` metadata. Tokenizers that supply pieces and scores
without BPE merge ranks use longest-piece segmentation in that route.

See [Architecture](architecture.md) for the execution contracts and
[Benchmarks](benchmarks.md) for measurement requirements.
