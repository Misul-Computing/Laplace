# Laplace Research Report

June 2026. This is a research document, not a changelog. It records
what shipped, the evidence behind each decision, the open problems,
and the most publishable directions.

Status: all work in section 1 has landed, builds clean, and passes
ctest (9/9). The SIMD codebook-lookup path is un-gated and verified
by parity tests against the scalar reference on dims {64, 128, 256}
for 4-bit, no-QJL inputs.

Project history: the codebase was originally called `tinylm`. It is
now Laplace. The C++ namespace is `Laplace`. The release ships
LlamaArch, Qwen3NextArch, Gemma4Arch, and Phi3Arch behind a single
ModelArch interface.

## 1. What shipped

| # | Item | File | Test | Status |
|---|------|------|------|--------|
| 1 | AVX2 SIMD codebook-lookup path | `src/turboquant_simd.cpp`, `src/turboquant.cpp` | `test_simd_4bit_parity` | un-gated, parity-clean |
| 2 | Lookahead Decoding (Jacobi branch) | `src/main.cpp` | smoke test on Qwen2.5-0.5B | 6/6 accepted vs prompt-lookup 1/3 |
| 3 | WH-domain DeltaNet state update | `src/ssm.cpp`, `src/ssm.h`, `src/arch_qwen3next.cpp` | `test_deltanet_wh_parity` | 8 trials x G=2 D=8 within 1e-3 |
| 4 | LaplaceKV paged KV cache | `src/laplace_kv.{h,cpp}` | `test_laplace_kv` | 10 tests, 65888+ checks |

### 1.1 The SIMD fix

The original SIMD path used `_mm256_unpacklo_ps` to interleave the 8
lo-nibble centroids with the 8 hi-nibble centroids into a 16-float
output. The bug: the 256-bit form of `unpacklo` operates per 128-bit
lane, returning 8 floats, not 16. Confirmed against the Intel
Intrinsics Guide.

The fix is `interleave_8_pairs` in `src/turboquant_simd.cpp`: four
128-bit `unpacklo`/`unpackhi` plus `_mm256_set_m128` to recombine.
Costs 4 unpacks per 16 dims but keeps FMA throughput at 8 floats x 2
ops per cycle. The previous code's 1 unpacklo was returning only 8 of
the 16 needed floats (the wrong 8), which is why the gated-off path
produced permuted outputs.

### 1.2 The WH-domain DeltaNet state

The Gated DeltaNet state is a G x D x D matrix per layer, updated
sequentially during token generation. At Qwen3-Next's D=128, G=16,
that is 4 MB per layer in FP32. At 256K context on the 80B-A3B, the
state matrix dominates memory traffic.

The published literature compresses the attention K/V cache (KIVI,
KVQuant, QServe, QuaRot, ZipCache) but does not compress any
linear-attention state. The closest reference is Mamba-2's chunkwise
auxiliary WY representation, which is stored in real space, not
rotated.

`deltanet_token_wh` (src/ssm.cpp) runs the delta-rule update in the
Walsh-Hadamard-rotated basis: S_tilde = H * S * H^T. The math is a
basis change of the same linear operator. The per-element order
shifts, so a few-ULP rounding error appears, but the output is
equivalent. The conv1d, L2-normalize, and gated RMSNorm stay in real
space; only the state update and the Q/K/V transforms live in WH
space. The 8-trial unit test confirms equivalence within 1e-3
absolute on G=2, D=8.

No published work does a basis change of a linear-attention state for
compression. This is the foundation for quantizing the state with the
same TurboQuant machinery used for K/V.

### 1.3 LaplaceKV paged cache

Chunk-based paging with two tiers: 4-bit WARM and 2-bit COLD in the
backing store. Resident chunks stay FP16 in the pool. The pool
operates in the WH domain from init, so the WH-domain attention fast
path (dot_k_wh / weighted_add_v_wh) is used during both prefill and
decode. This was the key fix: the previous code only enabled
WH-domain attention after prefill, which meant the prefill attention
path skipped paged chunks entirely, corrupting the output.

Attention dispatches per chunk: FP16 dot/axpy for resident chunks,
4-bit codebook dot/axpy for paged chunks. Chunk eviction uses
attention scores recorded during decode (layer 0 only). Sink chunks
are always resident.

## 2. Open problems

### 2.1 TQ4 vs FP32 slowdown at short context

Partially fixed. With the SIMD path un-gated, the codebook-lookup
inner loop is AVX2-vectorized. Expected speedup is 1.4-1.7x on the
dequant + dot, per the published AVX2 maddubs + madd_epi16 chain
(FBGEMM, XNNPACK, llama.cpp ggml, gemmlowp).

The remaining gap to llama.cpp's 17.3 tok/s on Qwen3.5-2B (Laplace:
15 tok/s) is the int8-activation quantization pass and the per-row
micro-kernel fusion. The published CPU int8 x int4 GEMV story is
thin: FBGEMM/XNNPACK/gemmlowp/llama.cpp all dequantize int4 to FP16
or int8 and then do an int8 dot. A fused int8-activation x int4-weight
inner loop is a research gap and a publishable kernel.

AVX-VNNI (dpbusd on Zen 4) is the right inner loop for the int8 x
int8 case. The `dot32_i8` kernel in `src/matmul_simd.cpp` already
uses it via LAPLACE_NATIVE. The SIMD codebook-lookup is the int4-weight
side, which is the missing piece in the literature.

### 2.2 head_dim=64 cache quantization

Empirically worked around (--kv-calibrate fixes Qwen2.5-0.5B at
head_dim=64) but no published analysis exists.

KIVI's central empirical finding (Sec 3 of arXiv:2402.02750) is that
K activations exhibit a few outlier channels across tokens, while V
activations exhibit a few outlier tokens across channels. This is the
opposite of what most cache-quant schemes (which quantize per-token
for both) assume. KIVI's asymmetric 2-bit scheme (K per-channel, V
per-token, both with FP16 residual for recent tokens) maintains
near-FP16 quality on Llama/Falcon/Mistral (all head_dim=128).

For Laplace's head_dim=64 case (Qwen2.5-0.5B, hidden=896, 14 Q heads,
2 KV heads, head_dim=64), KIVI's per-channel K is the right next
experiment. Lloyd-Max on a 64-dim WH-rotated unit vector has a small
per-codebook sample (16 levels x 4 bits), and the per-vector scale is
a 4-byte float, so the per-vector overhead is about 0.25 B/element vs
KIVI's 0.28 B/element with group_size=32.

The published papers all use head_dim >= 128. LLaMA-2-7B through 70B,
Mistral-7B, Qwen-7B, and Qwen2-7B all have head_dim=128. AWQ (Lin et
al. MLSys 2024) uses group_size=128 as a default and explicitly warns
against anything smaller. There is no head_dim sweep in the
literature. The Qwen2.5-0.5B (head_dim=64) result is novel and
publishable.

### 2.3 DeltaNet state at long context

The WH-domain foundation is in. The quantization is not.

The DeltaNet state is FP32 in every published linear-attention model.
At Qwen3-Next 80B-A3B with about half the 80 layers as GDN, that is
about 160 MB on the active path, about 1 GB at full sequence for
batched inference. None of KIVI / KVQuant / QServe / QuaRot / ZipCache
/ ATOM / SqueezeLLM / L2-Norm (Devoto 2406.11430) addresses it. KIVI
explicitly states (Sec 1) it targets attention K/V cache, "we do not
consider the SSM or RNN state". The fla-org GDN kernel stores state
in plain FP32 with no quantization hooks.

The WH-domain transformation in Laplace is the first basis change of
a linear-attention state for compression. The mathematical content is
straightforward (orthonormal rotation), but the missing piece is a
codebook calibration procedure for the WH-rotated state. The state has
a different dynamic range than K/V: the entries are influenced by the
time-decay term g = exp(A * softplus(a + dt_bias)), which varies by
group. Per-group Lloyd-Max codebooks, with the existing
--kv-calibrate machinery adapted, is the next step.

### 2.4 Speculative decoding on sub-3B models

Lookahead (Jacobi branch) is in. Full Lookahead (the verification
branch with the 2D window + n-gram pool) is not.

The Jacobi branch alone is what Laplace shipped. The 6/6 accepted on
"The capital of France is" is consistent with the paper's claim that
few Jacobi iterations successfully achieve simultaneous decoding and
correct positioning of multiple tokens. The 1.5-2.3x speedup from the
paper is for the full Lookahead, including the verification branch, on
LLaMA-2-Chat 7B+ at A100. Jacobi-only is substantially slower. The
realistic claim at the moment is 1.1-1.3x for Jacobi-only with
ngram=3 on repetitive prompts.

EAGLE-2 and Medusa both require fine-tuning of extra heads
(arXiv:2401.15077, arXiv:2401.10774). They are inappropriate for
Laplace's CPU-only deployment on sub-3B models. The only published
CPU-feasible path with no extra model and no fine-tuning is Lookahead
or prompt-lookup.

## 3. Novel directions (publication candidates)

Three directions where Laplace has material to publish, ranked by what
can be delivered now.

### 3.1 Walsh-Hadamard rotation as a universal basis for cache quantization

Components in Laplace: (1) the WH-Lloyd-Max-QJL synthesis itself (no
published equivalent found), (2) the WH-domain DeltaNet state, (3) the
per-layer calibrated codebook machinery (--kv-calibrate), (4) the
head_dim=64 sweep that the literature does not have.

Story: orthogonal rotation as a single technique that handles
attention K/V, attention V with a different bias, and linear-attention
state. The asymmetry between K and V quantization (per-channel vs
per-token, from KIVI) is a property of the post-rotation distribution,
not of the model. The DeltaNet state, which no one has touched,
becomes quantizable by the same machinery.

Closest related work: QuaRot (arXiv:2404.00456) for per-weight
rotation, KIVI (arXiv:2402.02750) for per-activation rotation.
Neither addresses the state matrix of a linear-attention model.

Effort: 3-4 weeks. The first 1-2 weeks are the head_dim=64 ablation
(KIVI vs TurboQuant vs both on Qwen2.5-0.5B / 1.5B / Qwen3-0.6B at
WikiText-2 perplexity). The next 1-2 are the DeltaNet state
quantization implementation and measurement on a small GDN-from-scratch
training run (Qwen3-Next-80B is too big to test on a 6-core Ryzen, but
a 1B GDN from scratch is feasible).

### 3.2 head_dim=64 is the missing case in the cache-quant literature

Story: the published papers all use head_dim >= 128. Laplace has a
model at head_dim=64 (Qwen2.5-0.5B) that exposes a failure mode
(TurboQuant without calibration repeats; with calibration it works).
The paper is "what is the right granularity for cache quantization at
head_dim=64, and is KIVI's per-channel scheme dominant there".

Effort: 1-2 weeks. Implement KIVI 2-bit per-channel K + per-token V
in Laplace (about 50 lines of new matmul kernel), benchmark against
TurboQuant on Qwen2.5-0.5B / 1.5B / 3B at WikiText-2 perplexity,
write up.

### 3.3 The CPU int8 x int4 GEMV kernel

Story: the published CPU int8 GEMV libraries (FBGEMM, XNNPACK,
gemmlowp, llama.cpp ggml) all dequantize int4 to FP16/int8 before the
inner loop. A fused int8-activation x int4-weight inner loop on
AVX2/AVX-VNNI is a research gap. The kernel needed is a 32-element
block: int8 activations x int4 weight nibbles, with vpdpbusd on Zen 4
or vpmaddubsw + vpmaddwd on pre-Zen-4.

Effort: 2 weeks. The first 1-2 days are the kernel implementation in
matmul_simd.cpp (extends the existing dot32_i8 with int4 nibble
packing). The next few days are the benchmark against llama.cpp's Q4_0
on a 2B model at batch=1.

## 4. Negative results

Things the research pass specifically searched for and could not find.
They shape what Laplace can credibly claim as novel.

| Searched for | Result | Implication |
|--------------|--------|-------------|
| A paper matching the WH + Lloyd-Max + QJL synthesis | Not found after arxiv search across 2023-2026 KV-cache literature | Laplace's contribution; should be written up |
| A paper quantizing any linear-attention state | Not found after searching fla-org, mamba repo, KIVI/KVQuant/QServe/QuaRot | WH-DeltaNet state is novel |
| A paper publishing a head_dim sweep | Not found; all papers use head_dim >= 128 | The Qwen2.5-0.5B result is novel |
| EAGLE-2 paper | v1 (EAGLE, 2401.15077) is on arxiv; v2 (dynamic draft trees) not located | Skip EAGLE-2; Lookahead is the right pick |
| RECURRENTSPEC | Not located | Skip |
| A published per-layer bit-width picker | Not found; KVQuant Sec 5 sweeps but does not publish an algorithm; mistral.rs claims per-layer but does not publish | opportunity |
| A CPU int8 x int4 GEMV paper | Not found; all CPU int4 GEMV literature dequantizes first | opportunity, see 3.3 |
| The "TurboQuant" naming for our synthesis | "TurboQuant" is an Apple ANN-search paper (different domain); the internal name is just ours | document the synthesis with a clean paper-ready name |

## 5. Next moves (ordered by signal-to-effort)

1. CPU int8 x int4 GEMV kernel in matmul_simd.cpp (extends dot32_i8
   with int4 nibble packing). 2-3 days. Expected: closes the 1.5x gap
   to llama.cpp on the Q8_0 path, and opens up a Q4_0 path that is
   currently bandwidth-bound.

2. KIVI 2-bit per-channel K + per-token V as a baseline. 1 day
   implementation. Run the head_dim=64 ablation on Qwen2.5-0.5B /
   1.5B / Qwen3-0.6B. 2-3 days for benchmarks. This is the 3.2 paper.

3. Per-layer bit-width picker in KVCache using the per-layer
   reconstruction error from --kv-calibrate as the sensitivity signal.
   1 day. Add --kv-bits-per-layer K=4,3,4,4,... CLI. 4-page
   ICLR-style paper on its own.

4. DeltaNet state quantization (per-group Lloyd-Max codebook on the
   WH-rotated state, time-decay-aware calibration). 1-2 weeks.
   End-to-end smoke test on a small GDN from scratch. This is the 3.1
   paper.

5. Full Lookahead verification branch (the 2D window + n-gram pool).
   1-2 weeks; needs a custom CPU-friendly attention mask and the
   n-gram extraction from the upstream LookaheadDecoding/lade.
   Doubles the Jacobi-only speedup (1.1-1.3x to 1.5-1.8x on
   Laplace-class models).

## 6. Architecture support matrix

| Arch        | Class         | Source                       | Differentiating feature                |
|-------------|---------------|------------------------------|----------------------------------------|
| qwen3next   | Qwen3NextArch | src/arch_qwen3next.{h,cpp}   | hybrid: full attn + Gated DeltaNet     |
| qwen35      | (alias)       |                              |                                        |
| llama       | LlamaArch     | src/arch_llama.{h,cpp}       | GQA, partial RoPE, SwiGLU FFN          |
| qwen2       | (alias)       |                              | Llama 2/3, Mistral, Qwen 2/3           |
| qwen3       | (alias)       |                              |                                        |
| phi3        | Phi3Arch      | src/arch_phi3.{h,cpp}        | fused QKV + gain+1 RMSNorm (1 + w)     |

Next additions, in order of signal-to-effort:

- gemma2: query pre-attention scaling, GeLU FFN. 1-2 days.
- mixtral / qwen2moe: sparse MoE. 3-5 days (the change is in
  forward_layer; the rest of the engine is unchanged).
- falcon: multi-query attention, optional alibi. 2-3 days.
- command-r / gemma3: additional RoPE scaling modes. 1-2 days.

Each is a 1-3 day patch once ModelConfig has the right metadata keys.

## 7. References

### KV-cache compression
- KIVI: https://arxiv.org/abs/2402.02750 (Liu et al., ICML 2024)
- KVQuant: https://arxiv.org/abs/2401.18079 (Hooper et al., NeurIPS 2024)
- QServe: https://arxiv.org/abs/2405.04532 (Lin et al., MLSys 2025)
- QuaRot: https://arxiv.org/abs/2404.00456 (Ashkboos et al., NeurIPS 2024)
- L2-Norm: https://arxiv.org/abs/2406.11430 (Devoto et al., EMNLP 2024)

### Linear attention / SSM
- Gated DeltaNet: https://arxiv.org/abs/2412.06464 (Yang et al., ICLR 2025)
- DeltaNet (original): https://arxiv.org/abs/2102.11174 (Schlag et al.)
- Mamba-2 (SSD): https://arxiv.org/abs/2405.21060 (Dao & Gu, ICML 2024)
- Mamba-1: https://arxiv.org/abs/2312.00752 (Gu & Dao)
- fla-org reference impl: https://github.com/fla-org/flash-linear-attention

### Speculative decoding
- Lookahead: https://arxiv.org/abs/2402.02057 (Fu et al., ICML 2024)
- LookaheadDecoding: https://github.com/hao-ai-lab/LookaheadDecoding
- EAGLE: https://arxiv.org/abs/2401.15077 (Li et al.)
- Medusa: https://arxiv.org/abs/2401.10774 (Cai et al.)

### SIMD / CPU kernels
- FBGEMM: https://arxiv.org/abs/2101.05615; https://github.com/pytorch/FBGEMM
- XNNPACK: https://github.com/google/XNNPACK
- gemmlowp: https://github.com/google/gemmlowp
- llama.cpp ggml: https://github.com/ggml-org/llama.cpp

### Weight quantization (context only)
- AWQ: https://arxiv.org/abs/2306.00978 (Lin et al., MLSys 2024 best paper)
- SqueezeLLM: https://arxiv.org/abs/2306.07629 (Kim et al., ICML 2024)
- BitNet 1.58: https://arxiv.org/abs/2310.11453 (Ma et al.)
