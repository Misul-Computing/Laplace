# LaplaceKV design record

The preregistered publication criteria are in
[`laplace_kv/PROTOCOL.md`](laplace_kv/PROTOCOL.md).

LaplaceKV is an architecture-neutral KV storage and attention layer for
autoregressive inference. It has no model names, learned codebooks, calibration
data, or layer-specific thresholds. Laplace currently supplies the first
adapter. The storage and attention contract is independent of its model code.

K8/V6 is now the control, not the intended final format. A paper candidate
must fit within 3.0 effective bits per scalar including all overhead, target
2.25 bits or less, and pass the quality, latency, portability, and novelty
gates in the publication protocol.

## Current format

- A mutable 64-token K/V tile remains FP32.
- A completed tile is transformed with Walsh-Hadamard when the head dimension
  is a power of two. Other supported dimensions use the original domain.
- Keys are symmetric signed 8-bit, token-major, with one scale per token.
- Values are symmetric signed 6-bit, coordinate-major, with one scale per
  token. The high four and low two bits are stored separately and consumed
  directly by I8MM/NEON.
- Attention uses online softmax. It never evicts tokens or approximates the
  softmax support.
- Grouped-query heads share each archive read and each packed K/V traversal.

The implementation supports head dimensions from 32 through 512 in multiples
of 16. Power-of-two dimensions use the rotation. Non-power-of-two dimensions
use the same tile format without it. Unsupported dimensions fall back to FP16
in the Laplace executable rather than running an unsafe approximation.

## Storage policy

The packed format and storage location are separate. Resident mode keeps
sealed tiles in unified memory. Streaming mode puts them in an unlinked file,
disables the macOS file cache, and reads large sequential ranges. Only the FP32
tails and bounded per-active-head read buffers remain resident.

At head dimension 128, one 64-token tile is 14,848 bytes:

- K codes: 8,192 bytes
- V codes: 6,144 bytes
- K/V scales: 512 bytes

That is 14.50 MiB per layer and KV head at 65,536 tokens in the archive. A
24-layer, two-KV-head model uses 696 MiB of archive space. Its persistent KV
RAM is about 3 MiB for FP32 tails. With two concurrent 7.25 MiB read buffers,
peak KV working RAM is about 17.5 MiB. The former 4-bit implementation required
about 408 MiB at the same shape, so the streaming working set is about 23 times
smaller. Archive space is not counted as RAM.

## Measured results

Release build, Apple Silicon, one head, dimension 128:

| Context | Resident attention | Streaming attention | Streaming RAM | Archive |
|---------|--------------------|---------------------|---------------|---------|
| 512 | 0.008 ms | 0.080 ms | 0.06 MiB | 0.11 MiB |
| 4,096 | 0.050 ms | 0.123 ms | 0.06 MiB | 0.91 MiB |
| 16,384 | 0.142 ms | 0.258 ms | 0.06 MiB | 3.62 MiB |
| 65,536 | 0.456 ms | 1.462 ms | 0.06 MiB | 14.50 MiB |

At 65,536 tokens the prior streamed K8/V8 implementation took 2.008 ms and
used a 16.50 MiB archive. K8/V6 is 27% faster in this run and 12% smaller.
Synthetic output relative error was 3.34% at that context. Across 48 captured
layer/head traces, mean output error was 1.66%, p95 was 2.50%, and the maximum
was 3.74%.

A Qwen2.5 0.5B Q4 retrieval run recovered `COBALT-CEDAR-4826` exactly from a
14,474-token prompt. Prefill was 69.8 tok/s and eight-token decode was 27.7
tok/s. A 40-token resident run recovered `SILVER-ORCHID-7319` exactly and
decoded at 148.7 tok/s. These are correctness and regression measurements, not
claims about every model or machine.

## Rejected foundations

- TurboQuant-style coordinate quantization was removed. Its compact storage
  still requires dequantization during dense attention and its old 4-bit path
  was slower than direct packed traversal at long context.
- RaBitQ targets approximate nearest-neighbor distance estimation. Its
  probabilistic bounds do not certify a dense softmax output, and the metadata
  needed for the tested KV layout erased the memory advantage.
- Quest-style page selection can reduce reads by evaluating only selected
  pages. It changes dense attention support. Conservative bounds on the real
  traces selected enough pages that filesystem read granularity removed most
  of the traffic saving.
- Progressive K4 refinement had unacceptable worst cases on captured traces.
  V4 output error was about 7%. K4 plus a K4 residual reduced that to about
  0.51%, but conservative refinement still read about 94% of archive traffic.
- POSIX AIO double buffering was slower than synchronous large sequential
  reads on the target Mac.
- Independent scalar quantization, causal deltas, shared online codebooks, and
  query-subspace correction all failed optimistic trace tests below 3 bits.
- An exact future-query oracle that allocated three bits globally across all
  layer/head caches still had 35.10% mean output error. Value-only sparsity and
  low-rank oracles also failed with exact keys.
- Query-independent centroid and moment coresets required nearly every token
  on the captured trace. The broad mechanism also collides with CentroidKV,
  SemantiCache, MomentKV, Multipole Attention, and WildCat.
- A fixed random-feature attention state had more than 100% mean output error,
  even before its state was rounded to the proposed storage precision.
- Cross-layer ternary residuals failed with exact anchor layers. Full linear
  predictors trained on the first 64 tokens made the result worse. AQUA-KV,
  xKV, XQuant, and BaseMix-KV also occupy the broad predictive-coding method.
- A query-aware survivor oracle at the 2.88-bit FP16 retention limit still had
  7.98% mean, 27.38% p95, and 90.14% maximum attention-output error. Pure
  eviction cannot pass the trace diagnostic on this capture.
- Zero-overhead sigma-delta rounding improved K4/V1 at 2.969 bits, but its mean
  output error remained 124.84%. Preserving tile marginals does not preserve
  peaked attention.
- A fully counted two-context, four-lane rANS K4/V2 record reached 2.992 bits
  per scalar at the worst captured D64 tile. Its native fused decoder was 35x
  to 40x slower than FP16, so entropy coding is rejected for the Apple CPU hot
  path.
- Strict K3/V2, K4/V2, and K3/V3 mutable-tail formats failed cached model
  quality by hundreds to thousands of percent. Reconstructing those tails
  before tile sealing did not recover the lost information.
- Alternating fixed K3/K4 rows plus V2 fit below 3 bits but changed perplexity
  by 19.96%. A rotating block-floating three-bit record changed it by more than
  13,000%. Fixed scalar layouts are rejected at this rate.
- A fixed product vector quantizer reached 2.877 bits per scalar but had 40.86%
  mean attention error. The broad method also overlaps CommVQ, HQMQ,
  TurboQuant, and VQKV.
- Fully counted shared fixed or prompt-derived scalar tables failed the trace
  screen by 52% to 102% mean error against K4/V2 Q8. Prompt-derived tables also
  violate the no-calibration rule. Exact K/V gauge centering made the quantized
  result worse.
- Pre-RoPE fixed product coding improved over the same post-RoPE codec but still
  had 68.48% mean attention error. KVQuant and RotateKV already establish the
  pre-RoPE mechanism.
- An optimistic activation-normalized fixed K4/V2 control failed even when its
  per-coordinate scales were treated as free. This rejects the weaker
  projection-weight-derived implicit-scale proposal before model integration.
- A fixed joint K15/V3 mixed-radix tile reached exactly 3 bits per scalar,
  including Q6 metadata and alignment, but changed Qwen2.5 perplexity by
  +6.38%.
- A split K12/V4 tile kept four value levels and fit Q6 key plus Q5 value
  metadata in exactly 3 bits per scalar. It changed Qwen2.5 perplexity by
  +49.75%. Balanced stochastic V3 rounding made the original K15/V3 trace
  result substantially worse on the first predeclared seed.
- A same-token K-to-V predictive residual fit at 2.75 bits per scalar for D64,
  including FP16 row scales. An invalid prompt-fitted full-matrix oracle still
  measured 49.17% mean error with exact keys and a two-bit residual, so the
  training-free weight-derived route was not implemented.
- Token and checkpoint regeneration reduced stored bytes only by adding 20.7
  to 216.6 TOP of replay work per generated token at 16K and 64K. It is rejected
  for hot decode.
- A faithful public eOptShrinkQ reconstruction averaged 2.57 bits per scalar
  after record metadata, but its D64 mean attention error was 39.89% versus
  24.00% for K4/V2 Q8. Exact low-rank factors still measured 36.20%, so factor
  quantization was not the main failure. The paper's FP16 tail and decoder
  table also raise the fully counted 4,095-token cache to 3.24 bits per scalar.
- A standalone native direct-factor kernel avoided dense K/V reconstruction,
  but every fully counted rank lost to K8/V6 at both 16K and 64K. Ranks near
  the three-bit limit were only about FP16 speed at D64 and slower at D96,
  D128, and D256. Spectral residuals are rejected for the Apple hot path.
- An attention-conditioned residual correction reached 17.39% mean error at
  2.877 bits per scalar when it was allowed to see the current queries and
  exact attention weights. The causal version trained on the first tile then
  measured 33.00%, versus 23.15% for K4/V2 Q8, even though that per-request
  calibration already violates the protocol. The oracle identifies future
  attention sensitivity as useful information but is not an encoder.
- Generic `W_Q`/`W_O`-weighted SVD is not a clean novelty claim. OSCAR,
  Expected Attention, Palu, Eigen Attention, STAR-KV, OjaKV, and related work
  occupy attention metrics, weight-derived covariance, low-rank KV, factor
  folding, or direct compressed attention. A distinct spectral result would
  need a calibration-free RoPE/GQA-aware generalized shrinker, not just a
  different weighting matrix.
- A full-context frozen-weight mask selected 31 K4 key coordinates over a K3
  base and paired them with V2. The completed D64 record was exactly 3 bits per
  scalar, but Qwen2.5 perplexity changed by +824.06% over 1,024 predictions.
  The compile-gated simulator was removed and the longer lifecycle and native
  kernel were not run.
- An exhaustive fixed-alphabet sweep tested 2,879 complete three-bit layouts.
  K7/V6 Q8/Q8 was the only point that improved mean and p95 against K16/V4 Q8
  on both traces, but its 1,024-prediction Qwen2.5 lifecycle changed perplexity
  by +107.86%.
- Reusing exact previous-tile scale state and regenerating scale with a
  decoder-synchronized gain servo both failed the trace control. A 2.75-bit
  backward value forest had an exact direct-attention identity, but failed the
  two-trace gate even with exact keys. Its fully counted K4 path was worse.

The closest current baselines are
[vLLM 0.25 INT4 KV](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/ops/int4_per_token_head/),
[GSRQ](https://arxiv.org/abs/2607.01065),
[MosaicKV](https://arxiv.org/abs/2607.00760),
[TurboQuant](https://arxiv.org/abs/2504.19874),
[KVarN](https://arxiv.org/abs/2606.03458),
[eOptShrinkQ](https://arxiv.org/abs/2605.02905),
[OCTOPUS](https://arxiv.org/abs/2605.21226),
[FibQuant](https://arxiv.org/abs/2605.11478),
[OSCAR](https://arxiv.org/abs/2605.17757), and
[HyperQuant](https://arxiv.org/abs/2606.23406). KVarN reports 2.3 effective bits
for its paper configuration and now ships a stricter K4/V2 implementation.
vLLM's fused per-token-head INT4 path uses asymmetric quantization after a
deterministic randomized Hadamard transform and stores each four-bit zero point
inside the low four bits of a float32 scale. Its fully counted payload is
`4 + 32/D` bits per scalar, so it fails the three-bit storage gate at every
supported head width. It occupies broad metadata-field co-packing but not the
rejected Laplace code-index syndrome mechanism. The release has no complete
final-code FP16/BF16 quality and speed matrix. See `VLLM_025_AUDIT.md`.
GSRQ's sub-one-bit headline excludes its learned calibrated codebooks and its
accelerated low-rate point fails quality. MosaicKV has a strong packed sparse
kernel, but its measured 3x FP16 reduction is about 5.33 bits per scalar and
its per-model quality loss reaches 3.55%. See `JULY_2026_FRONTIER_AUDIT.md`.
eOptShrinkQ reports about 2.2 bits but does not provide hardware-aware latency
results. The local reconstruction and direct kernel do not reproduce a usable
quality or speed advantage. OCTOPUS fails the local quality limit at its
2.50-effective-bit point. HyperQuant also fails quality at 2.50 effective bits,
and its preliminary quality-passing point uses 3.48 effective bits plus a
variable-length decode that is not a throughput win on H100. These baselines
raise the novelty bar beyond merely beating TurboQuant's bit rate. Also
relevant are [Quantized Keys Steal
Attention](https://arxiv.org/abs/2605.26266),
[RaBitQ](https://arxiv.org/abs/2405.12497),
[Quest](https://arxiv.org/abs/2406.10774), and
[KIVI](https://arxiv.org/abs/2402.02750).

## Engine adapter contract

An engine adapter supplies layer index, KV-head index, position, head
dimension, K/V vectors, query vectors, causal end positions, and the attention
scale. It receives one output vector per query head. Storage mode, tile sealing,
quantization, grouped-query reuse, and online softmax stay inside LaplaceKV.

Laplace adapters cover Gemma 4, Llama and its Qwen aliases, Phi-3, and
Qwen3-Next. A second inference engine should only need to map its tensor layout
to this contract. The packed tile implementation has no dependency on GGUF or
Laplace model classes.
