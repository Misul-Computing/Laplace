# Exact regeneration cache audit

Status: rejected as the hot LaplaceKV representation. Token replay is useful
for evicting an inactive session, but it cannot satisfy the three-bit storage
gate and the long-context decode-speed gate at the same time. No production
code follows from this audit.

## Question

Can Laplace store token IDs plus a small set of exact checkpoints, regenerate
old K/V when needed, and still beat resident FP16 and K8/V6 attention at 16K
and 64K?

The answer is no for dense causal attention. Token IDs are an extremely compact
description of the cache, but decoding them back into K/V is prompt replay.
An exact checkpoint that avoids replay is the prefix K/V state itself, or
layer-local residual states whose fully counted size is larger than the K/V
cache on the GQA reference model.

This conclusion is specific to hot single-request decode. Regeneration remains
a reasonable policy for a suspended request whose K/V would otherwise occupy
memory while doing no work.

## Exactness contract

For a fixed model, token sequence, positions, and deterministic execution
schedule, replay can reconstruct the FP16 cache exactly: run the same forward
operations again and round K/V at the same boundary. A different prefill batch
shape or reduction order is not guaranteed to be bit-identical. A portable
archive would therefore also identify the model and numerical contract.

The model weights are shared deployment data and are not counted as cache
bytes. Token IDs are counted as `uint32_t`. Temporary K/V built while replaying
is counted as active memory and temporary storage.

The reference shape is Qwen2.5-0.5B:

- 24 layers
- 14 query heads
- 2 KV heads
- head dimension 64
- hidden dimension 896
- about 0.49 billion parameters

These dimensions come from the [official model card and
configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). There are

```text
S = 2 * layers * KV_heads * head_dim = 6,144
```

K/V scalars per token. The official context is 32,768 tokens. The 64K rows
below are format and operation-count extrapolations for the protocol's 64K
attention target, not a model-level Qwen result.

## Storage accounting

K8/V6 uses `7 + 32 / head_dim` bits per scalar for sealed 64-token tiles. It
is 7.5 bits at D64. A token log uses `32 / S = 0.005208` bit per K/V scalar.

| Representation | Bits/scalar | 16K stored | 64K stored | Effect |
|----------------|------------:|-----------:|-----------:|--------|
| Resident FP16 K/V | 16.000 | 192 MiB | 768 MiB | exact baseline |
| Resident K8/V6 | 7.500 | 90 MiB | 360 MiB | lossy control |
| Token IDs only | 0.005208 | 0.0625 MiB | 0.25 MiB | exact after canonical full replay |
| One BF16 residual/token | 2.333 | 28 MiB | 112 MiB | one BF16 layer directly; not exact for Laplace |
| One FP32 residual/token | 4.667 | 56 MiB | 224 MiB | one layer directly; fails rate gate |
| BF16 residual at every layer | 56.000 | 672 MiB | 2,688 MiB | direct BF16 projections; not exact for Laplace |
| FP32 residual at every layer | 112.000 | 1,344 MiB | 5,376 MiB | direct reconstruction of FP16 K/V |

The one-residual rows are important: a residual at layer `l` directly
reconstructs K/V at layer `l`, not every layer. This is developed below.

### Active RAM

For this Qwen shape with Laplace's fixed eight-row reserve, shared non-weight
state is 10.164 MiB at 16K and 24.977 MiB at 64K. That count includes FP32
model row buffers, QKV and FFN scratch, attention logits, RoPE tables, the
eight-row vocabulary-logit buffer, and prompt token IDs. It excludes model
weights, allocator metadata, thread stacks, the tokenizer, and operating-system
pages. Those exclusions do not change between cache modes.

| Mode | 16K persistent representation | 16K active total | 64K persistent representation | 64K active total |
|------|------------------------------:|-----------------:|------------------------------:|-----------------:|
| Resident FP16 | 192 MiB | about 202.16 MiB | 768 MiB | about 792.98 MiB |
| Resident K8/V6 | about 91.51 MiB including FP32 tail | about 101.68 MiB | about 361.55 MiB including FP32 tail | about 386.53 MiB |
| Token log, current token-wise replay | 0.0625 MiB | about 202.16 MiB at replay peak | 0.25 MiB | about 792.98 MiB at replay peak |
| Three-bit FP16 checkpoint replay | at most 36 MiB | about 202.16 MiB at replay peak | at most 144 MiB | about 792.98 MiB at replay peak |
| Every-layer FP32 residuals | 1,344 MiB | at least 1,354.16 MiB | 5,376 MiB | at least 5,400.98 MiB |

The current replay path reaches a full FP16 cache at the final replayed token,
even though only the token log or early checkpoint persists afterward. A new
layer-major replay implementation could avoid the full transient cache by
running exact tiled attention. It would need at least one full FP32 residual
sequence buffer, 56 MiB at 16K and 224 MiB at 64K, and normally separate input
and output buffers, 112 MiB and 448 MiB, plus per-layer QKV and attention
scratch. Its stored token archive would remain tiny, but its prompt replay work
would be unchanged.

## Full K/V checkpoints

Let `n` be the current context and `c` the position of one stored FP16 K/V
checkpoint. A conservative record that also retains every token ID costs

```text
bytes = 2 * S * c + 4 * n
```

The complete three-bit allowance is `3 * S * n / 8` bytes. Therefore the
checkpoint can cover no more than about 18.75% of the sequence. Exact integer
accounting gives:

| Context | Three-bit allowance | Latest FP16 checkpoint | Suffix replayed per decode |
|--------:|--------------------:|-----------------------:|---------------------------:|
| 16,384 | 36 MiB | token 3,066 | 13,318 tokens |
| 65,536 | 144 MiB | token 12,266 | 53,270 tokens |

Deleting old checkpoints or arranging them geometrically does not improve the
bound. Only the latest complete prefix state can restart the computation, and
multiple snapshots duplicate its prefix. A K8/V6 checkpoint could cover about
40% at D64, but it is not exact and still leaves 60% prompt replay.

The replay builds K/V for the suffix. With the current token-wise engine, peak
resident K/V returns to 192 MiB at 16K and 768 MiB at 64K before the final
logits are produced. Streaming that temporary cache to SSD can bound RAM, but
it allocates and traverses a full temporary cache for every generated token.
That is offload, not three-bit encoded storage.

## Replay work

Count a multiply and add as two operations. For active parameter count `P`,
query heads `Hq`, and head dimension `D`, the operation model for replaying
`n` tokens through the current `forward_batch` contract is

```text
full replay = 2 * P * n
            + 2 * layers * Hq * D * n * (n + 1)
```

The first term is model projection, MLP, and output-head work. The second term
is causal QK and weighted-V attention over all prefix queries. It excludes
normalization, RoPE, softmax, activation functions, sampling, and cache
construction.

Starting at a complete K/V checkpoint `c` changes the bound to

```text
suffix replay = 2 * P * (n - c)
              + 2 * layers * Hq * D
                * (n * (n + 1) - c * (c + 1))
```

The quadratic term barely falls because every replayed suffix query still
attends to the checkpointed prefix.

| Context | Normal cached decode | Token-only replay | Three-bit checkpoint replay |
|--------:|---------------------:|------------------:|----------------------------:|
| 16K | 2.389 GOP | 27.602 TOP | 24.193 TOP |
| 64K | 6.617 GOP | 248.946 TOP | 230.454 TOP |

The token-only path performs 11,552 times the logical work of normal cached
decode at 16K and 37,621 times at 64K. The best three-bit FP16 checkpoint still
performs 10,126 times and 34,827 times the cached work.

An optimized regeneration path could omit the LM head for intermediate prompt
tokens. Using the official 0.36-billion non-embedding parameter count lowers
full replay to 23.342 TOP at 16K and 231.907 TOP at 64K. The checkpoint path
falls to 20.730 TOP and 216.604 TOP. Those optimistic values remain 9,769 and
35,046 times normal cached decode for full replay, and 8,676 and 32,734 times
for checkpoint replay.

The repository's measured 14,474-token Qwen prefill ran at 69.8 tokens/s, or
207.4 seconds for the prompt. Calibrating the current-contract expression
above to that run gives an effective 111.9 GOP/s and these estimates:

| Context | Token-only replay latency | Three-bit checkpoint latency |
|--------:|--------------------------:|------------------------------:|
| 16K | 246.8 s | 216.3 s |
| 64K | 2,225.6 s | 2,060.2 s |

These are analytical extrapolations, not new latency measurements. They are
already generous because they omit non-matmul work. The same real run decoded
from its cache at 27.7 tokens/s, about 36 ms per token near 14.5K.

## Memory traffic and energy-relevant work

With GQA sharing each stored K/V vector once, the current token-wise path makes
1.500 TiB of logical FP16 cache traversals during full replay at 16K and 24.000
TiB at 64K. The three-bit checkpoint schedule still traverses 1.448 TiB and
23.160 TiB. A single normal FP16 decode traverses 192 MiB and 768 MiB
respectively. K8/V6 reduces the normal decode archive to 90 MiB and 360 MiB.
These are logical bytes touched, not a claim that every byte misses the CPU
cache or crosses DRAM exactly once.

Laplace currently processes prefill in batches of at most eight rows. If the
roughly 400 MiB Q4 model is read once per batch, token-only replay accounts for
about 800 GiB of weight reads at 16K and 3.125 TiB at 64K. The three-bit
checkpoint schedule accounts for about 650 GiB and 2.54 TiB. A fully optimized
large GEMM could reuse weights better than this implementation, but it cannot
remove the operation counts or quadratic causal attention.

No joule claim is made because package power was not measured. FLOPs and bytes
moved are the energy-relevant evidence. Both grow by orders of magnitude over
cached decode. SSD replay would add writes and reads on every token.

The target machine has 307 GB/s of advertised unified-memory bandwidth
according to [Apple's M5 Pro specification](https://support.apple.com/en-us/126318).
That makes local unified RAM a different case from PCIe offload. The 2024
[I/O-aware partial recomputation paper](https://arxiv.org/abs/2411.17089)
recomputes some K/V to hide CPU-to-GPU transfers; it does not show that prompt
replay beats a local packed cache in unified memory.

## Residual checkpoint audit

At one fixed layer, the identity is straightforward:

```text
K_l,p = RoPE(WK_l * Norm(h_l,p))
V_l,p =      WV_l * Norm(h_l,p)
```

The layer-local residual `h_l,p` is enough to reconstruct that layer's K/V.
This does not collapse the layer dimension:

- storing `h_0,p` permits direct layer-0 projection, but obtaining `h_l,p`
  requires running layers `0..l-1` over the causal prefix;
- storing `h_L,p` does not provide a general inverse for earlier residuals;
- avoiding replay requires storing `h_l,p` for every layer that will be
  projected directly.

This exposes an unresolved accounting gap in the March 2026
[KV-Direct paper](https://arxiv.org/abs/2603.19664). Its per-token equation
counts one `hidden_dim` residual shared across all layers, while its
reconstruction is indexed by layer. Cross-task patching at each layer proves
that the corresponding layer residual is sufficient at that layer. It does
not prove that one selected residual directly reconstructs all other layers.
The associated [KV-Direct repository](https://github.com/Kaleemullahqasim/KV-Direct)
currently says that code and experimental scripts are coming later, so the
claimed bounded implementation cannot be reproduced or inspected.

For Qwen2.5-0.5B, one BF16 residual stream happens to fit at 2.333 effective
bits per K/V scalar. It is less compact than token IDs and still forces nearly
full prompt replay. Storing every layer's residual costs 56 bits per scalar,
3.5 times the FP16 K/V cache. Laplace residuals are FP32, raising that to 112
bits per scalar.

Even if storage were ignored, projecting K and V from all layer residuals
costs

```text
4 * n * layers * hidden * KV_heads * head_dim
```

or 180.4 GOP at 16K and 721.6 GOP at 64K before attention. K8/V6 avoids these
model-weight projections and directly consumes packed codes.

## Layer schedules and selective regeneration

Suppose `J` layer-boundary residual streams are retained for every token. At
`b` bytes per residual element, their rate is

```text
8 * b * hidden * J / S bits per original K/V scalar.
```

On Qwen2.5, one BF16 boundary costs 2.333 bits and one FP32 boundary costs
4.667 bits. The three-bit gate therefore permits at most one BF16 boundary and
no FP32 boundary. A single boundary still requires replaying at least 23 of 24
historical layer transitions. Token embeddings regenerated from token IDs make
that one stored boundary unnecessary.

Selective head regeneration does not change this dependency. All K and V heads
at a layer derive from the same `h_l,p`; without that residual they require the
preceding causal computation. Dense softmax also assigns positive weight to
every finite logit, so an exact universal method cannot declare an old head or
token irrelevant from position alone.

Exact prefix sharing can reduce physical memory across requests with identical
prefixes and can improve speed. It is already part of
[PagedAttention](https://arxiv.org/abs/2309.06180), and it does not compress a
unique request. At least six requests would have to share the same FP16 prefix
before its physical bytes amortize below three bits per request.

## Prior art and decision

The underlying ideas are occupied:

- The original [Transformer](https://arxiv.org/abs/1706.03762) defines causal
  attention over prior positions.
- [FlashAttention](https://arxiv.org/abs/2205.14135) tiles exact attention to
  reduce working-memory traffic, but exact causal work remains quadratic for a
  full replay.
- [Sublinear-memory checkpointing](https://arxiv.org/abs/1604.06174) formalizes
  the general compute-for-memory trade.
- vLLM uses recomputation for cache-pressure preemption and explicitly warns
  that it harms end-to-end latency in its [optimization
  documentation](https://github.com/vllm-project/vllm/blob/main/docs/configuration/optimization.md).
- I/O-aware partial K/V recomputation and KV-Direct occupy the more specific
  inference mechanisms discussed above.

There is no universal online schedule that stays within three fully counted
bits, reconstructs dense-attention K/V exactly, and beats resident FP16 or
K8/V6 long-context decode. No standalone native kernel was added because every
candidate fails either storage before a kernel is relevant or requires full
prompt replay.

The retained systems use is narrow: when a session becomes inactive, discard
its K/V and keep the model identity plus token IDs. Rebuild once if the session
resumes. This minimizes idle-session storage at the cost of time to first
token. It must not be described as raw KV compression or as a hot decode path.
