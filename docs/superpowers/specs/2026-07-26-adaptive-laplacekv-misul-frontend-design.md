# Adaptive LaplaceKV and Misul Frontend Design

## Goal

Make the dense K4/V2 formula the compact path inside the default universal
LaplaceKV policy without accepting its measured long-lifecycle quality loss.
Misul Terminal becomes the dedicated user-facing CLI for discovering,
configuring, and running Laplace models.

The implementation must remain architecture-independent. Codec decisions use
runtime tile data and measured reconstruction behavior. They must never inspect
model names, architecture names, layer numbers, or tensor-name conventions.

## Current evidence

The existing fixed K4/V2 representation stores:

```text
K[t,d] = (K4[t,d] * ka[d] + kb[d]) * kc[t]
V[t,d] = (V2[t,d] * va[t] + vb[t]) * vc[d]
```

Its 128-token record costs 3.5625 bits per K/V scalar at D64 and 3.375 bits at
D128 before one tile-state byte. A direct fixed-width Apple Silicon kernel
measured 1.19x to 1.29x faster than the current K8/V6 attention path in
isolated 16K and 64K tests. Preliminary model screens looked near-lossless,
but the registered 2,048-prediction Qwen2.5 lifecycle measured +4.774%
perplexity and 84.131% top-1 agreement.

The existing K8/V6 representation costs about 7.5 logical bits per scalar and
measured -0.245% perplexity change, 95.996% top-1 agreement, and 0.00495 mean
KL on the same registered lifecycle. It is the quality control, not the
desired final storage rate.

These results require a fail-closed adaptive policy. K4/V2 is preferred when
it meets the control's error envelope. A tile that does not meet that envelope
is stored as K8/V6.

## LaplaceKV architecture

### One public automatic mode

`KVCacheMode::LAPLACE` remains the default and becomes the adaptive mode.
`--laplace-kv` selects it explicitly. `--laplace-kv-q4`, `--kv-fp16`, and
`--kv-fp32` remain comparison and research controls.

The automatic mode operates on 128-token logical tiles:

- The mutable tile remains FP32.
- When the tile fills, the engine creates one K4/V2 candidate and two 64-token
  K8/V6 control tiles from the same FP32 source.
- It reconstructs both candidates and computes deterministic K and V error
  summaries against the FP32 source.
- It stores only the selected representation.

The selector compares both RMS and worst-vector relative L2 error separately
for keys and values. K4/V2 is accepted only when none of its four error
summaries exceeds the matching K8/V6 summary. Non-finite values, zero-norm
ambiguity, failed encoding, or unsupported dimensions select K8/V6. This is a
strict local safety rule rather than a model-specific heuristic.

The selector does not establish an end-to-end quality guarantee by itself.
The existing 2,048-prediction lifecycle and cross-model checks remain the
release gates. If adaptive mode does not match the registered K8/V6 quality
envelope, K8/V6 remains the effective default until the selector is tightened.

### Variable-size sealed storage

Each layer/head owns an append-only sealed payload and a fixed descriptor per
logical tile:

```text
format: K4_V2 or K8_V6
offset: byte offset in the resident arena or streaming archive
size: encoded payload bytes
```

K4/V2 stores one 128-token payload. K8/V6 stores two existing 64-token
payloads back-to-back. Descriptors are indexed by layer, KV head, and logical
tile, so attention does one predictable format branch per 128 tokens rather
than per scalar or per token.

Resident mode appends to per-head byte arenas. Streaming mode appends to the
existing unlinked cache file and retains only descriptors, the mutable FP32
tile, and bounded read scratch in RAM. Offsets, descriptors, alignment,
mutable storage, and archive traffic are included in reported memory totals.

No payload is re-encoded during attention. No dense tile is materialized in
the hot path. K4/V2 calls its direct fixed-width operations. K8/V6 calls the
existing packed operations twice.

### Observability

Laplace reports:

- selected K4/V2 and K8/V6 tile counts;
- the physical encoded bytes and effective bits per scalar;
- resident or streaming archive bytes;
- bytes read and written in streaming mode.

`--bench` prints these counters after generation. Tests can query them through
small read-only accessors. There is no tuning flag for the selector in the
first implementation.

## Misul Terminal product surface

Misul Terminal is the CLI frontend for Laplace. The `misul` command no longer
asks the user to choose a provider or authenticate a cloud account.

At startup it:

1. Resolves the Laplace binary from `MISUL_LAPLACE_BINARY`, the adjacent
   Laplace development build, or `laplace` on `PATH`.
2. Resolves a model from an explicit CLI path, `MISUL_LAPLACE_MODEL`, or GGUF
   files in the Laplace model directory.
3. Starts the existing terminal session using the Laplace provider.

The existing TypeScript provider remains the single process boundary. It
constructs Laplace arguments, sends the prompt through a temporary file, and
streams parsed output back to the terminal. Environment variable names are
standardized on the `MISUL_LAPLACE_*` prefix.

The first product surface retains only controls Laplace can honor:

- model path;
- maximum generated tokens;
- context length, defaulting to 262144;
- extra Laplace arguments for development and benchmarking.

The adaptive `--laplace-kv` policy is always supplied unless the extra
arguments contain an explicit KV comparison flag. Misul does not silently
select the lossy fixed-Q4 research mode.

Generic provider packages can remain in the monorepo as internal libraries,
but the default executable, startup flow, help text, and documentation present
Misul as the Laplace frontend.

## System prompt

Replace the adapted consumer constitution with a concise Laplace coding-agent
prompt. It must state that:

- the model is running inside Misul Terminal on Laplace;
- repository claims require current tool evidence;
- requested changes should be implemented minimally and verified;
- destructive unrequested actions require confirmation;
- completion claims require fresh tests or an explicit unverified status.

Remove all descriptions or instructions concerning subagents, delegation,
ACP agents, multi-provider operation, provider selection, model-agnostic
operation, consumer wellbeing policy, and unrelated chat-product behavior.

The existing stable block construction and content-addressed cache behavior
remain unchanged.

This prompt cleanup does not delete dormant extension or provider library
code. It removes those concepts from the Laplace product surface and from the
model's operating instructions.

## Failure behavior

- Missing Laplace binary: Misul exits with the attempted resolution paths and
  the relevant environment variable.
- Missing model: Misul lists the searched model locations and asks for an
  explicit model path.
- Laplace process failure: preserve stderr and exit status in the terminal
  error without retrying automatically.
- Unsupported KV dimension: Laplace selects K8/V6, then FP16 if the existing
  K8/V6 contract cannot represent it.
- Streaming I/O failure: initialization fails rather than silently switching
  to an uncounted resident allocation.

## Tests and acceptance

Laplace tests are written first and must cover:

- a low-error tile selecting K4/V2;
- a sensitive tile selecting K8/V6;
- non-finite candidate data failing closed;
- mixed-format attention matching direct per-format attention;
- resident and streaming descriptor round trips;
- exact encoded-byte accounting;
- no model or architecture input to the selector.

Laplace acceptance requires the release build, full CTest suite, registered
2,048-prediction quality screen, cross-model screen where local models are
available, and 16K/64K attention benchmarks. Performance and quality numbers
are reported as measurements, never inferred from unit tests.

Misul tests are written first and must cover:

- Laplace binary and model resolution precedence;
- 262144 default context;
- automatic `--laplace-kv` argument insertion and explicit override handling;
- absence of provider selection in default CLI help and startup;
- absence of subagent, delegation, ACP, and model-agnostic language from the
  default prompt;
- preservation of deterministic prompt block construction.

Misul acceptance requires its focused AI and terminal tests, TypeScript build,
and a local CLI launch against the built Laplace binary. A generation smoke
test is run only when a suitable local GGUF is available.

## Scope limits

This change does not add model-name fast paths, architecture-specific cache
classes, cloud-provider deletion, new dependencies, background model
downloads, or automatic quality claims. It does not promise 42 tok/s without a
matching measured full-model benchmark.
