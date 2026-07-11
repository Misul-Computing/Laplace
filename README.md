# Laplace

<h3 align="center">The Apple Silicon inference engine</h3>

## About

Laplace is a from-scratch LLM inference engine built exclusively for Apple
Silicon. It targets perf-per-watt, not raw throughput: the differentiator is
how many tokens you get per joule on a laptop battery, not how fast a
water-cooled server farm can go.

Laplace is fast and efficient with:

- NEON + DOTPROD + I8MM SIMD matmul kernels for common GGUF quant types, with
  scalar fallback for the remaining supported types
- Power-aware thread count and QoS policy, with automatic battery and AC modes
- LaplaceKV: K8/V6 sealed tiles, a mutable FP32 tail, grouped-query attention,
  and optional SSD-backed streaming for bounded KV memory
- LaplaceMoE SSD streaming: runs MoE models larger than RAM by streaming expert
  weights on demand with a bounded resident set
- Speculative decoding: prompt-lookup and Jacobi-iteration lookahead
- WH-domain DeltaNet state compression for Qwen3-Next hybrid models
- Metal GPU backend with function-constant-specialized dequant kernels

Laplace supports:

- Gemma 4 (MoE, hybrid attention, GeGLU)
- Llama 2/3, Mistral, Qwen2/Qwen2.5/Qwen3 dense
- Qwen3-Next (hybrid attention + Gated DeltaNet)
- Phi-3 / Phi-3-mini

## LaplaceKV at 65K context

These are isolated one-head attention and storage measurements, not full-model
decode results.

| Measurement | LaplaceKV | Previous path | Improvement |
|-------------|-----------|---------------|-------------|
| Active KV RAM, 24 layers and 2 KV heads | 17.5 MiB | 408 MiB resident 4-bit | **23.3x less active RAM** |
| Streamed attention, 1 head at dimension 128 | 1.50 ms | 2.01 ms streamed K8/V8 | **25% faster** |
| Archive, 1 layer and KV head | 14.50 MiB | 16.50 MiB K8/V8 | **12% smaller** |
| Raw packed size versus FP16 | 232 bytes/token | 512 bytes/token | **2.21x compression** |

The 23.3x result is the reduction in memory occupied while inference is
running. LaplaceKV achieves it by keeping the 64-token FP32 tails and bounded
read buffers in RAM while sealed K8/V6 tiles live in a temporary SSD archive.
The archive is 696 MiB for the complete 24-layer, two-KV-head example. Raw
K8/V6 encoding by itself is 2.21x smaller than FP16.

## Getting started

Build:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLAPLACE_NATIVE=ON
cmake --build build
ctest --test-dir build
```

Run:

```bash
./build/laplace model.gguf                      # dump metadata
./build/laplace model.gguf -p "Hello, Laplace"  # generate
./build/laplace model.gguf -p "prompt" --bench  # time prefill + decode
```

Key flags:

| Flag | Description |
|------|-------------|
| `-n N` | Max tokens to generate |
| `-t T` | Sampling temperature |
| `--top-k K` | Top-k sampling |
| `--top-p P` | Nucleus sampling |
| `--greedy` | Greedy decoding |
| `--seed S` | RNG seed |
| `--max-seq L` | Max context length |
| `--bench` | Time prefill + decode |
| `--laplace-kv` | LaplaceKV with automatic resident or streaming storage (default) |
| `--laplace-stream` | Keep sealed KV tiles in an SSD-backed archive |
| `--laplace-resident` | Keep sealed KV tiles in unified memory |
| `--kv-fp16` | Uncompressed FP16 comparison path |
| `--kv-fp32` | Uncompressed FP32 comparison path |
| `--eval-file PATH` | Compare LaplaceKV against FP16 on cached next-token prediction |
| `--eval-limit N` | Limit the number of evaluation predictions |
| `--draft N` | Speculative decoding max draft length |
| `--draft-mode {prompt,lookahead}` | Draft strategy |
| `--no-spec` | Disable speculative decoding |

Build options:

| Option | Description |
|--------|-------------|
| `-DLAPLACE_NATIVE=ON` | Use `-march=native` |
| `-DLAPLACE_LTO=ON` | Link-time optimization (~5% gain) |
| `-DLAPLACE_OPENMP=OFF` | Disable threading |
| `-DLAPLACE_KV_CAPTURE=ON` | Build the opt-in KV research trace hook |

## How it works

### Power-aware scheduling

A custom thread pool parks idle workers on a futex (zero CPU when inactive).
A background PowerMonitor polls power state, thermal level, and throughput
every 3 seconds and adjusts the active thread count and core mode
automatically: E-core mode on battery to minimize watts, performance mode on
AC to maximize throughput. Online probing tries +/-1 thread and measures
tok/s to find the actual optimum.

### KV-cache compression

LaplaceKV seals each completed 64-token tile as K8/V6. Keys use token-major
SDOT layout so logits retain K8 accuracy. Values use a coordinate-major packed
6-bit layout consumed directly by I8MM/NEON during the weighted sum. The
current tile remains FP32, avoiding quantization churn during decode.

Resident mode keeps sealed tiles in unified memory. Streaming mode places them
in an unlinked temporary file with the macOS file cache disabled, leaving only
the mutable tail and bounded read buffers in RAM. The default selects resident
storage for configured contexts up to 4096 tokens and streaming above that.
All supported architectures use the same grouped-query attention interface.
The design record and rejected alternatives are in
[`research/LAPLACE_KV.md`](research/LAPLACE_KV.md).

#### Current LaplaceKV findings

These are current measurements, not universal claims.

- At head dimension 128, a sealed 64-token tile occupies 14,848 bytes. This
  is 2.21x smaller than FP16 KV and 4.41x smaller than FP32 KV. It is not
  smaller than a true 4-bit archive.
- At 65,536 tokens, one-head streamed attention measured 1.50 ms versus
  2.01 ms for the previous streamed K8/V8 path, about 25% faster. Its archive
  was 14.50 MiB versus 16.50 MiB.
- For a 24-layer model with two KV heads at 65,536 tokens, streaming uses
  about 17.5 MiB of active KV RAM and 696 MiB of temporary archive space. The
  former resident 4-bit path used about 408 MiB of RAM at the same shape.
  This is a RAM working-set reduction from SSD-backed storage, not a 23x raw
  compression ratio.
- A 2,048-prediction cached evaluation on Qwen2.5 0.5B changed perplexity by
  -0.245% versus FP16, with 95.996% top-1 agreement and mean KL divergence of
  0.00495. The K8/V6 control passes this one quality screen but remains above
  the storage limit.
- On one fixed cross-model corpus, Qwen3.5 reached 98.63% top-1 agreement,
  while Phi-3 reached only 85.16% and was 1.37x slower than FP16. The current
  non-power-of-two head-dimension path is not publication-ready or universal.
- A 14,474-token retrieval prompt recovered `COBALT-CEDAR-4826` exactly.
- A compile-gated K4/V2 reference simulator with Q8 metadata stayed within
  the preliminary 2% perplexity limit on Qwen2.5, Qwen3.5, and Phi-3. Its
  sealed D64 entropy record measured 2.859 mean and 2.992 maximum bits per
  scalar on 48 captured tiles, with every stated byte counted.
- The longer 2,048-prediction Qwen2.5 lifecycle rejected fixed K4/V2:
  perplexity increased by 4.774%, top-1 agreement fell to 84.131%, and mean
  KL reached 0.0788. This supersedes the shorter quality impression.
- That entropy format is not a production candidate. In a synthetic,
  single-thread native microbenchmark, its fused rANS kernel was 35x to 40x
  slower than FP16 at 16K and 64K. A local packed K4/V2 proxy was also slower;
  it was not a production or fully optimized K4/V2 kernel.
- Strict three-bit mutable tails failed model quality. K3/V2, K4/V2, and
  K3/V3 per-token tails changed Qwen2.5 perplexity by +1,853%, +516%, and
  +451%, respectively. A fixed three-bit block-floating format changed it by
  +13,064%.
- A model-independent product vector quantizer fit at 2.877 bits per scalar,
  but its mean attention error was 40.86%.
- A fixed-width K4/V2 Apple Silicon kernel was 3.07x to 3.55x faster than
  FP16 and 1.19x to 1.29x faster than K8/V6 in the 16K and 64K synthetic
  attention benchmark. Its separately stored metadata raises the format above
  3 bits per scalar. Hiding complete Q6 metadata in code-index syndromes made
  the sealed tile exactly 3.0 bits per scalar, but changed Qwen2.5 perplexity
  by +6.85% and reached only 0.83x FP16 and 0.32x K8/V6 decode speed. That
  candidate is rejected.
- A fully counted eOptShrinkQ reconstruction averaged 2.57 bits per scalar at
  long context, but its D64 mean attention error was 39.89% versus 24.00% for
  K4/V2 Q8. Its native direct-factor kernel lost to K8/V6 at every tested rank
  and dimension. The FP16 tail and decoder table also raised its 4,095-token
  cache to 3.24 bits per scalar.
- An attention-conditioned 2.877-bit oracle beat K4/V2 mean error only when it
  could see the exact current queries and attention weights. The causal
  version measured 33.00% mean error versus 23.15% for K4/V2 Q8, even with
  request-specific calibration, and is rejected. This result narrows the open
  question to a fixed model-weight metric, not a deployable codec.
- A frozen-weight mixed-precision candidate used K3 for every key coordinate,
  K4 for the 31 most weight-sensitive key coordinates, and V2. Its completed
  D64 record fit exactly 3 bits per scalar, but a real 1,024-prediction
  Qwen2.5 lifecycle changed perplexity from 99.20 to 916.66, or +824.06%.
  The temporary simulator was removed and no native kernel was built.
- A split K12/V4 record preserved four value levels and fit Q6/Q5 metadata in
  exactly 3 bits per scalar, but its Qwen2.5 lifecycle changed perplexity by
  +49.75%. Balanced stochastic V3 rounding kept the exact K15/V3 record but
  worsened mean trace error from 53.32% to 84.07% on the first fixed seed.
- An optimistic prompt-fitted same-token K-to-V predictor still measured
  49.17% mean trace error with exact keys and a two-bit value residual. Its
  fully encodable 2.75-bit D64 K4/R1 form measured 141.64% and is rejected.
- An exhaustive fixed-alphabet sweep evaluated 2,879 complete three-bit
  layouts. Its only two-trace mean and p95 winner, K7/V6 Q8/Q8, changed
  Qwen2.5 perplexity by +107.86% over 1,024 predictions.
- Zero-metadata previous-tile scale state and a decoder-synchronized gain
  servo both lost badly to K4/V2 Q8. A 2.75-bit backward value forest had an
  exact direct-attention identity, but its exact-key form missed the two-trace
  gate and its complete K4 path was substantially worse.
- Two recent public vector codecs also fail Laplace's combined gates.
  [OCTOPUS](https://arxiv.org/abs/2605.21226) reports +34.7%/+41.5%
  perplexity change at 2.50 effective bits, and
  [HyperQuant](https://arxiv.org/abs/2606.23406) reports +7.4%/+8.1% at the
  same complete rate. HyperQuant reaches +1.8%/+1.4% at its 3-bit target only
  with a complete 3.48-bit cache, and its variable-length decode is not a
  throughput win on H100.
- [vLLM 0.25](https://github.com/vllm-project/vllm/releases/tag/v0.25.0) adds
  fused per-token-head INT4 KV with a zero point packed into the float32 scale.
  Its complete payload is `4 + 32/D` bits per scalar, or 4.25 bits at D128
  before page overhead. The merged Triton path is not an Apple Silicon backend
  and has no complete final-code speed matrix.
- The [vLLM TurboQuant integration](https://github.com/vllm-project/vllm/pull/38479)
  uses a 3.1875-bit D128 K3/V3 slot before native-precision boundary layers.
  Its reported Qwen3-4B GSM8K score was 0.720 versus 0.900 for the baseline,
  with 65% of baseline decode-heavy throughput at that setting.
- The source-matched MLX-VLM TurboQuant K2/V3 D64 simulator failed the same local
  2,048-prediction Qwen2.5 lifecycle: perplexity increased by 9,402%, top-1
  agreement was 10.596%, and mean KL was 5.168. The simulator used the
  official seeded rotation, spherical codebooks, and FP16 norms.
- The source-matched MLX-LM affine Q2/G64 simulator also failed: perplexity
  increased by about 16,766%, top-1 agreement was 8.25%, and mean KL was
  5.316. Its 2.5-bit rate is a theoretical packed payload, not measured native
  storage or speed.
- A [KIVI](https://github.com/jy-yuan/KIVI) K2/V2 G32 simulator with its
  128-token FP16 window changed
  Qwen2.5 perplexity by 34.977%, with 69.580% top-1 agreement and mean KL
  0.3237. Its FP16 min and scale fields make the quantized body 3 bits per
  scalar before the residual window, so the complete cache also misses the
  storage gate.
- The July public frontier also has no combined pass. GSRQ's nominal rate
  excludes learned codebooks and its accelerated low-rate point fails quality
  ([GSRQ](https://arxiv.org/abs/2607.01065)).
  [MosaicKV](https://arxiv.org/abs/2607.00760) reports a strong packed sparse
  kernel at a measured 3x FP16 memory reduction, equivalent to about 5.33 bits
  per scalar.

The cross-model publication protocol, including failure thresholds and honest
memory accounting, is in
[`research/laplace_kv/PROTOCOL.md`](research/laplace_kv/PROTOCOL.md).
The running measurements, including failed ablations, are in
[`research/laplace_kv/RESULTS.md`](research/laplace_kv/RESULTS.md).
The vLLM 0.25 format, evidence, and PagedAttention removal are audited in
[`research/laplace_kv/VLLM_025_AUDIT.md`](research/laplace_kv/VLLM_025_AUDIT.md).
The exhaustive scalar frontier and causal direct-codec screens are in
[`research/laplace_kv/ALPHABET_FRONTIER_AUDIT.md`](research/laplace_kv/ALPHABET_FRONTIER_AUDIT.md)
and
[`research/laplace_kv/CAUSAL_DIRECT_AUDIT.md`](research/laplace_kv/CAUSAL_DIRECT_AUDIT.md).
The newest public codec and systems results are audited in
[`research/laplace_kv/JULY_2026_FRONTIER_AUDIT.md`](research/laplace_kv/JULY_2026_FRONTIER_AUDIT.md).
The exact K12/V4, balanced V3, and same-token predictive screens are in
[`research/laplace_kv/MIXED_RADIX_AUDIT.md`](research/laplace_kv/MIXED_RADIX_AUDIT.md)
and
[`research/laplace_kv/KV_PREDICTIVE_AUDIT.md`](research/laplace_kv/KV_PREDICTIVE_AUDIT.md).
The source-linked Cyberpunk technology study and the systems ideas extracted
from it are in
[`research/laplace_kv/CYBERPUNK_TECH_LORE.md`](research/laplace_kv/CYBERPUNK_TECH_LORE.md).

The research target is at most 3 effective bits per K/V scalar including all
overhead, faster 16K and 64K decode than FP16 and the fixed K4/V2 reference
kernel, and no
more than 2% model-level quality degradation on every tested model. K8/V6 is a
control and does not meet the storage target. No tested sub-three-bit format
meets the quality and speed gates, so no paper or breakthrough claim is made.

### LaplaceMoE

When a MoE model exceeds unified memory, `plan_residency` auto-detects the
overflow, pins dense weights to RAM, marks expert tensors `MADV_RANDOM` to
suppress OS readahead on the full expert stack, and pages active experts from
the model file. A bounded cache retains recently used experts. No env var is
needed.

Result on a 24GB Mac with Gemma 4 26B Q4 (model is 16.8 GB, exceeds the 65%
RAM budget): 0.33 tok/s before LaplaceMoE residency, 8.24 tok/s after.

### Speculative decoding

Prompt-lookup (default): n-gram match in the context. No model call. Best on
input-grounded text.

Lookahead: Jacobi-iteration draft. N model calls per step, then the existing
batched verifier.

### WH-domain DeltaNet state

For Qwen3-Next's Gated DeltaNet layers, the recurrent state is rotated into
the Walsh-Hadamard basis before the state update. Toggle with
`LAPLACE_DELTA_STATE_WH=1`. Mathematically equivalent to the FP32 path up to
FP rounding. No published work compresses a linear-attention state this way.

## Performance

| Model | Quant | Hardware | Decode | Prefill |
|-------|-------|----------|--------|---------|
| Gemma 4 26B MoE | Q4 | Apple M5 Pro | 18 tok/s | 19 tok/s |
| Gemma 4 26B MoE | Q4 | 24GB Mac (streaming) | 8.24 tok/s | - |

Both are memory-bandwidth bound. The M5 Pro runs 8 threads (5P + 3E) to
avoid thermal throttling that kicks in within 3s at 15 threads.

## Roadmap

- Power-aware phase scheduling: prefill on all P-cores, decode on E-cores,
  attention on E-cores. Continuous allocation per layer or per token.
- Metal GPU rebuild: simdgroup_matrix accelerators, shared-memory-mode
  zero-copy for unified memory, fused dequant kernels.
- On-the-fly quantization: analyze tensor sensitivity at load time, quantize
  easier layers more aggressively.
- Automatic architecture adaptation: read GGUF metadata and auto-generate
  the forward pass.
- Streaming model loading: stream any model from disk with near-zero memory
  overhead beyond the current layer.

## Contributing

Pull requests welcome. See `AGENTS.md` for build instructions, architecture
overview, and coding conventions.

## License

Apache 2.0
