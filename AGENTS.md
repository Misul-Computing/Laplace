# AGENTS.md

## Global rules (apply to every project unless overridden below)

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Owner standing rules (dttdrv)

Apply to every project unless a project file overrides them.

### Commits and pull requests
- Never add an AI or assistant co-author. No `Co-Authored-By` trailer for Claude, and no "Generated with Claude Code" footer. Commits and PRs read as the user's own work.

### Writing style: no AI tells
Write like a person, not a model. Cut the usual giveaways from prose, code comments, commits, and docs:
- No em dashes or en dashes. Use a plain hyphen, a comma, or a new sentence.
- Avoid the stock vocabulary: "delve", "leverage" as a verb, "boasts", "robust", "seamless", "elevate", "intricate", "testament to", "in today's fast-paced world", "it's worth noting", "rest assured".
- No "It's not just X, it's Y" or "not only X but also Y" constructions.
- No filler openers ("Great question", "Certainly", "I hope this helps") and no self-referential hedging.
- No decorative emoji or unicode symbols in code, commits, or prose. Straight ASCII quotes, not curly.
- Prefer a plain sentence over a bullet list, and a short answer over a long one. No bold-label lists added for decoration.

### Building on this machine
- Default to the minimal solution. The `ponytail` skill is installed; follow its ladder (YAGNI, reuse what exists, stdlib, native platform, one line) before reaching for new code or dependencies.
- Keep installs minimal and optimized. Do not pull large toolchains unless a task actually needs them; confirm the footprint first.

---

# Project notes (Laplace-specific)

## Project

Laplace is a C++20 Apple Silicon LLM inference engine. Binary is `build/laplace`.
The C++ namespace is `Laplace`. CMake project name is `laplace`.

## Build

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLAPLACE_NATIVE=ON
cmake --build build
ctest --test-dir build
```

`--bench` flag times prefill + decode.

## KV cache modes

| Mode             | Flag                 | Storage       | Quality |
|------------------|----------------------|---------------|---------|
| LaplaceKV auto   | `--laplace-kv`       | K8/V6         | low loss |
| LaplaceKV stream | `--laplace-stream`   | bounded RAM   | low loss |
| LaplaceKV RAM    | `--laplace-resident` | K8/V6 in RAM  | low loss |
| FP16 baseline    | `--kv-fp16`          | 2 bytes/value | ~exact  |
| FP32 baseline    | `--kv-fp32`          | 4 bytes/value | exact   |

LaplaceKV is the default. It seals completed 64-token tiles as K8/V6
and keeps the mutable tile FP32. Automatic mode uses resident storage at
configured contexts up to 4096 tokens and SSD-backed streaming above that.
All architecture paths use the shared grouped-query attention contract.

## Architecture support

| Arch        | Class         | File                          |
|-------------|---------------|-------------------------------|
| gemma4      | Gemma4Arch    | `src/arch_gemma4.{h,cpp}`      |
| qwen3next   | Qwen3NextArch | `src/arch_qwen3next.{h,cpp}`  |
| llama       | LlamaArch     | `src/arch_llama.{h,cpp}`      |
| phi3        | Phi3Arch      | `src/arch_phi3.{h,cpp}`       |

Aliases: qwen2/qwen3 map to llama. qwen35 maps to qwen3next.

To add an architecture: implement ModelArch, add a clause to
`laplace_arch.cpp`, add the source to `CMakeLists.txt` LAPLACE_SOURCES,
add a factory test case to `tests/test_arch.cpp`. About 200-400 lines
per arch.

## Threading

Custom thread pool (`src/threadpool.h`) with dynamic power management.
All possible workers created at startup; `active_count_` controls how
many are active at runtime. Inactive workers park on a separate futex
and consume zero CPU. Active workers can be confined to E-cores
(`QOS_CLASS_BACKGROUND`) or run at default QoS, switched at runtime.

A background `PowerMonitor` (`src/power_monitor.h`) polls power state,
thermal level, and actual throughput every 3s. It adjusts thread count
and core mode automatically: E-core mode on battery (minimize watts),
performance mode on AC (maximize throughput). Online probing tries +/-1
thread and measures tok/s to find the actual optimum. See `CHANGES.md`
for full details.

Apple Silicon only. x86 dropped until after 1.0.

## SIMD

NEON+FMA matmul kernels. Q4_0 int8 GEMV uses SDOT (I8MM) on Apple Silicon.
LaplaceKV uses SDOT for K8 logits and I8MM/NEON for packed V6 accumulation.
RMSNorm, RoPE, and FP16 dot/axpy are vectorized via NEON.

## Debug env vars

LAPLACE_LEGACY_ATTN=1: scalar attention loop (A/B comparison).
LAPLACE_NOSIMD=1: force scalar matmul.
LAPLACE_PROF=1: phase timing for prefill/decode.
LAPLACE_DEBUG_THREADS=1: print topology, thread count, mode at startup.
LAPLACE_DEBUG_POWER=1: print power monitor decisions (probes, adjustments).
LAPLACE_ECORES=1/0: force E-core mode on/off (default: auto from battery).
LAPLACE_DELTA_STATE_WH=1: WH-domain DeltaNet state (Qwen3-Next).

## Known limitations

LaplaceKV supports head dimensions from 32 through 512 in multiples of 16.
Power-of-two dimensions use Walsh-Hadamard rotation; other supported
dimensions use the same format without rotation. Other dimensions fall back
to FP16. Streaming trades SSD traffic for a KV working set bounded by the
number of active KV heads rather than context length.

## Vision and roadmap

The goal is to make Laplace the most efficient LLM inference engine on
Apple Silicon. Everything below is novel work, not copies of existing
engines.

### Priorities

1. Extreme efficiency without a memory hit. Perf-per-watt is the
   differentiator, not raw throughput. Target: 3-5x better perf/W than
   llama.cpp on Apple Silicon. Paths: power-aware thread scheduling
   (E-cores on battery), park-based idle waiting instead of spin-wait,
   hybrid P+E core allocation per inference phase.

   Dynamic power management: the engine should decide on the fly how
   much power to use and where to spend it. Not just "use 8 cores or
   4 cores", but a continuous allocation: what percentage of each
   compute unit (P-cores, E-cores) to use for each phase of
   inference. Prefill is compute-heavy and can use all P-cores.
   Decode is memory-bound and can run on E-cores alone, or with a
   few P-cores for the compute-bound matmuls and E-cores for the
   memory-bound phases (attention, RMSNorm, sampling). The system
   monitors power draw, temperature, and battery level in real time,
   and adjusts the allocation per layer or per token. This is hard
   because it requires modeling the cost of each compute unit for
   each kernel, and because the OS scheduler and thermal framework
   have their own opinions. But nobody does this, and it is the path
   to 3-5x perf/W.

2. Extreme KV cache compression without a decode penalty. Current: LaplaceKV
   K8/V6 tiles with grouped-query reuse and resident or SSD-backed storage.
   Target: reduce value precision further only when real trace and retrieval
   tests preserve quality and total attention traffic falls.

3. Streaming quantization of models (long-term). The idea: stream a
   model from disk in its quantized form without loading the whole file
   into memory. LaplaceMoE is the first step (per-expert SSD streaming). The
   dream: stream any model, any size, with near-zero memory
   overhead beyond what is needed for the current layer.

   On-the-fly quantization: quantize easier parts of a model on the spot
   during loading, trading up to 1% quality degradation for big memory
   savings. The engine should analyze each tensor's sensitivity and
   decide the optimal quant level (Q8, Q6, Q4, Q2) per-tensor, without
   slowing down inference. A layer that barely affects output can drop
   to 2-bit; a critical layer stays Q8. The user sees a smaller memory
   footprint with no perceptible quality loss and no speed penalty.

4. Automatic architecture adaptation. Instead of hand-writing each arch
   (LlamaArch, Gemma4Arch, etc.), build a system that reads a model's
   GGUF metadata and tensor topology, then auto-generates the forward
   pass. The hand-written archs become reference implementations and
   fast paths for known models.

5. Extreme speeds, miles better than llama.cpp, with no sacrifices.
   No quality degradation, no memory bloat, no feature loss. Just faster.

### Competitive landscape

Study competitor source codes for ideas (not copying):
- vllm: GPU-first, good batching and PagedAttention design. Their KV
  cache paging concept inspired LaplaceKV.
- llama.cpp: the reference CPU engine. Good quantization kernels, but
  no power awareness, no adaptive arch support.
- ollama: wrapper around llama.cpp, not worth studying.

Everything we build should be novel. If a feature exists in another
engine, we should understand why they did it that way, then find a
better approach or skip it.

### Platform roadmap

Apple Silicon (M-series, A18 Pro). NEON + DOTPROD + I8MM + BF16.
Power-aware scheduling. No NPU offload: M-series Macs have the Apple
Neural Engine but it is not practical for LLM inference (Core ML model
size limits, ANE designed for small models).
