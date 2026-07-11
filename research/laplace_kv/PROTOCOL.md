# LaplaceKV publication protocol

Status: revision 4. The hard storage, quality, and performance limits match the
project goal. Logical simulations remain early screens rather than publication
evidence.

## Claim under test

LaplaceKV is a training-free KV-cache format and attention implementation that
offers a better quality, decode-time, and active-RAM tradeoff on Apple Silicon
than the available baselines, without model-specific calibration.

## Breakthrough gate

The final algorithm must satisfy all of these requirements. Resident-memory
offload does not substitute for raw compression.

- No more than 3.0 encoded bits per K/V scalar. For a complete cache record,
  `bps = 8B / sum(2 * T * H_l * D_l)`, where `B` is its minimum allocated byte
  extent and the sum covers every attention layer. `B` includes codes, scales,
  zero points, norms, model/request-specific codebooks and tables, indexes,
  directories, residual and full-precision windows, headers, alignment, page
  slack, and allocation rounding. Fixed decoder constants that can be
  regenerated from the algorithm are reported as resident decoder RAM but are
  not encoded bytes. The design target remains 2.25 bps or less.
- No more than 2.0% degradation from FP16 on every registered model-level
  quality metric. Attention-output error is a diagnostic and must be reported,
  but one near-zero output does not by itself reject an end-to-end result.
- Lower 16K and 64K attention latency than both FP16 and the same-machine fixed
  K4/V2 reference kernel in resident and streaming modes.
- Training-free operation with no model, layer, or head calibration.
- One format and decode contract for MHA, GQA, and MQA across every supported
  head dimension.
- A mechanism and ablation that are materially distinct from TurboQuant's
  rotation and scalar codebook, KVarN's Hadamard plus dual-axis variance
  normalization, and eOptShrinkQ's spectral denoising plus TurboQuant residual.
- A same-machine quality, encoded-size, and latency comparison against the
  strongest reproducible public method, not only the former in-repo 4-bit
  implementation.

The current K8/V6 format does not pass this gate. It is the experimental
control used to locate the quality and decode bottlenecks.

This protocol does not permit the following claims unless their separate gates
pass:

- Raw compression superior to lower-bit methods.
- Lossless or caveat-free inference.
- Support for every model or inference engine.
- Fastest KV-cache implementation on every platform.

## Control implementation

The current K8/V6 implementation is the control:

- 64-token tiles.
- FP32 mutable tail.
- Per-token symmetric K8 and V6 scales.
- Token-major K codes and coordinate-major packed V codes.
- Optional Walsh-Hadamard rotation for power-of-two head dimensions.
- Online softmax with no token eviction.
- Resident and SSD-backed streaming storage.

The next candidate must be specified before its full cross-model results are
opened. Every candidate change reruns the complete matrix against this control.

## Local model matrix

| Family | Model | Attention layout | Head dimension |
|--------|-------|------------------|----------------|
| Qwen2.5 | Qwen2.5 0.5B Instruct Q4_K_M | GQA, 14Q/2KV | 64 |
| Qwen3.5 | Qwen3.5 0.8B Base Q4_0 | hybrid, 8Q/2KV | 256 |
| Phi-3 | Phi-3 Mini 4K Instruct Q4 | MHA, 32Q/32KV | 96 |
| Gemma 4 | Apex Gemma 4 26B MoE Q4 | mixed GQA | 256/512 storage |

These cover power-of-two and non-power-of-two dimensions, MHA and GQA, dense
and MoE models, and a hybrid attention/state-space architecture.

## Baselines

Every comparison must use the same model file, prompt tokens, process priority,
thread policy, machine power state, and build flags.

- FP32 KV.
- FP16 KV.
- LaplaceKV resident.
- LaplaceKV streaming.
- Fixed in-repo K4/V2 reference kernel.
- Apple MLX-LM affine Q2/G64.
- MLX-VLM TurboQuant K2/V3.
- KIVI-2 G32 R128, including its FP16 residual window.

The current external-codec implementations are source-matched FP32 lifecycle
simulators for quality screening. Their storage and timing fields are invalid.
They do not count as direct speed or memory comparisons. A direct comparison
requires a differential fixture against the official implementation, a native
packed implementation, and same-machine timing with all transforms included.
The local Q2/G64 CPU lookup-table benchmark is a proxy, not the MLX Metal
kernel, and is labeled as such in its output.

External KIVI, TurboQuant, vLLM 0.25 INT4, CommVQ, MiniKV, and other published
numbers may be reported as context but not as direct comparisons. A direct
claim requires an implementation on the same machine or an upstream benchmark
using the same model, hardware, and quality test.

## Quality gate

For each local model:

1. Evaluate at least 2,048 next-token predictions, or the model's full context
   if smaller, through the actual cached decode path.
2. Report NLL, perplexity, top-1 agreement, and mean/p95/max KL divergence
   against FP16.
3. Run deterministic retrieval at 25%, 50%, 75%, and 95% prompt depth at 1K,
   4K, 16K, and 64K. Contexts beyond a model's declared limit are recorded as
   unsupported, not silently replaced.
4. Run at least three prompt classes: prose, source code, and repeated/adverse
   token patterns.

Pass thresholds:

- Perplexity increase no greater than 2.0% for every model.
- No benchmark score decreases by more than 2.0% relative to FP16 for any
  model. Absolute-score changes are also reported.
- Retrieval success 100% across the registered matrix.
- No NaN, infinity, repetition collapse, or deterministic crash.

Top-1 agreement, KL divergence, and mean/p95/max attention-output error remain
mandatory diagnostics. They are not substitutes for model-level quality and
do not have independent pass thresholds.

Failure on one model fails the universal-quality claim. Results are still
publishable as a scoped systems result if the scope is stated exactly.

## Performance gate

Measure prefill and single-token decode after warmup at 512, 4K, 16K, and 64K
where supported. Use interleaved mode order and report median, p5, and p95 over
at least 20 trials. Record thermal state and active thread count.

Pass thresholds for a superiority claim:

- LaplaceKV resident and streaming are both faster than FP16 and the fixed
  K4/V2 reference at 16K and 64K.
- No mode is described as faster outside the contexts where it wins.

## Memory and I/O gate

Report all of the following separately:

- Packed archive bytes.
- Persistent KV RAM.
- Peak KV scratch RAM.
- Whole-process peak resident set.
- Bytes read and written per generated token.
- Temporary disk allocation.
- SSD write volume for a complete long-context request.

Raw compression is archive bytes divided by FP16 KV bytes. RAM reduction from
streaming must not be described as raw compression.

Logical payload, minimum complete allocated record, persistent allocated RAM,
fixed decoder constants, and peak scratch are separate fields. A theoretical
payload or FP32 simulator never satisfies the storage gate.

## Frozen corpora

The registered local corpora are:

| Class | Path | SHA-256 |
|-------|------|---------|
| Prose | `corpora/technical_prose_v1.txt` | `9fcba412b78ce9c39775e662496f68efe6431fc5532a043c4a6f06866d058743` |
| Code | `corpora/code_v1.txt` | `de72137211593601bcec0dd8495c65c603f01d22c0d1f3c3c20d8bdd8a180ba2` |
| Adverse | `corpora/adverse_v1.txt` | `1fe25a2b9d3889c901013d4d68c66462ea21d03895696444af0cfe40edb923cc` |

Changing a corpus creates a new version and hash. It does not overwrite an
existing registered input.

## Portability gate

The current implementation supports head dimensions 32 through 512 in
multiples of 16. Other dimensions fall back to FP16.

The universal-engine claim requires:

- Passing the model matrix above.
- Passing at least one additional model family not used during development.
- An adapter in a second inference engine.
- Identical packed-tile results between both engines.

Until then, the permitted wording is "architecture-neutral interface with
Laplace adapters."

## Novelty gate

The literature audit must cover, at minimum, TurboQuant, KIVI, KVQuant,
KVarN, eOptShrinkQ, vLLM 0.25 INT4 KV, OCTOPUS, FibQuant, OSCAR,
GSRQ, MosaicKV, RaBitQCache, MixKVQ, TriAttention,
VQKV, HQMQ, BalanceKV, Quantized Keys Steal Attention,
sequential predictive KV coding, residual-stream recomputation, CommVQ,
MiniKV, Lexico, Palu, Quest, RaBitQ, CacheGen, KVTC, and current SSD-backed KV
systems. The audit must also address the worst-case limits in
`The risk of KV cache compression`. For each component, identify prior art and
isolate what is new in the complete system.

A paper requires at least one contribution that is both new and supported by
an ablation. A new combination of known pieces is sufficient only if the
combination creates a measured Pareto improvement unavailable from each piece
alone.

## Decision rule

Write a paper only if the quality, performance, memory, portability, and
novelty gates pass for the exact claims in its title and abstract. Otherwise,
publish a no-go report containing all measurements, failures, and the next
experiment required.

No individual benchmark or runner may emit an overall project `passed` field.
Perplexity and logical storage form an early screen only. Retrieval, native
latency, memory and I/O, portability, and novelty remain separate required
evidence. The final decision must cite every required artifact by hash.
