# Universal Routed Execution and Q4 KV Design

## Scope

Improve the feature-driven execution path without introducing model-family,
architecture-name, layer-number, or model-size branches. The immediate
measurement target is the local 26B-A4B model, but every new interface operates
on tensor shape, routed-expert count, cache geometry, and observed runtime
behavior.

The work has two independent deliverables:

1. remove avoidable routed-expert I/O and dispatch work from single-token
   decode;
2. restore the fixed-width K4/V2 cache candidate as an opt-in global-layer
   codec with complete storage and quality accounting.

K4/V2 does not become the automatic default unless it passes the existing
cross-model quality protocol. Forty-two tokens per second is a performance
gate, not a result that may be claimed before measurement.

## Evidence and root cause

The recovered K4/V2 kernel is a useful long-context attention candidate, but
the completed lifecycle result is not lossless. Its fixed code traversal beat
K8/V6 in isolated 16K and 64K attention. Its separately stored metadata raises
the complete record above three bits per scalar, and the longer Qwen2.5 test
increased perplexity by 4.774%.

The 26B profile at 2K context places attention below four percent of token
time. Routed-expert paging and expert GEMV dominate. The current paging path:

- calls `touch_expert` and then unconditionally reads the same expert;
- reads cache hits again;
- creates an outer group of temporary threads, each of which creates another
  temporary thread group;
- reads into disposable buffers before the mapped tensor is consumed.

This is the first performance target. KV work cannot substitute for fixing it.

## Routed expert acquisition

`LaplaceMoE` exposes one acquisition operation for a tensor and an immutable
list of routed expert identifiers. A synchronous wrapper is available for
callers that need the data immediately. The executor uses an asynchronous
ticket so down-expert misses can overlap gate/up compute. Completion returns a
residency result that distinguishes hits from misses.

The acquisition operation:

- validates identifiers from tensor geometry;
- updates the bounded expert cache once;
- skips all I/O for resident experts;
- services only misses;
- uses persistent workers rather than per-layer `std::thread` construction;
- never changes model values or routing decisions.

The executor requests gate/up experts after routing. Down experts are acquired
while gate/up GEMV runs. The operation is generic for every stacked expert
tensor whose expert axis is described by tensor shape.

## Route-conditioned exact kernels

The routed CPU path receives tensor descriptors, active expert identifiers,
activations, and routing weights. It does not receive a model or architecture
name.

Two kernels replace intermediate work:

- `fused_moe_gate_up_geglu` quantizes the shared activation once, evaluates
  paired gate/up rows for each routed expert, applies GeGLU, and writes only
  the expert hidden vector.
- `fused_moe_down_accumulate` evaluates down rows and applies routing and
  per-expert scale while accumulating directly into the final output.

Both kernels preserve the current quantized dot products. Their output must
match the existing decomposed path within the established floating-point
tolerance before integration. Unsupported quant formats use the existing
path.

Thread partitioning follows contiguous row ranges. Runtime thread count and
QoS remain the responsibility of the existing power-aware thread pool.

## Fixed-width Q4 KV

The restored cache format uses the measured fixed-width equations:

```text
K[t,d] = (K4[t,d] * ka[d] + kb[d]) * kc[t]
V[t,d] = (V2[t,d] * va[t] + vb[t]) * vc[d]
```

Codes use token-major K4 and coordinate-major V2 layouts. Metadata is stored
and counted. No syndrome embedding, hidden FP16 copy, entropy stream, or
unreported scratch is permitted.

The codec:

- supports the same head-dimension range as LaplaceKV;
- keeps a mutable FP32 tile until it seals;
- implements resident and streaming tile traversal;
- is selectable for global layers through an explicit research flag;
- leaves bounded sliding layers in FP16;
- reports encoded bytes, resident bytes, archive bytes, and scratch bytes.

It remains opt-in until the existing 2,048-prediction, retrieval, latency, and
memory gates pass. Failing a gate keeps K8/V6 as the automatic global format.

## Automatic adaptation

Topology synthesis describes what operations and tensors exist. Runtime
selection describes which verified kernel is fastest for those shapes.

Selection rules may use:

- tensor quant type and dimensions;
- routed expert count;
- cache head dimension, context, and storage mode;
- measured same-process kernel timings;
- power and thermal state.

Selection rules may not use architecture names, model filenames, layer
numbers, or hard-coded 26B geometry.

## Cleanup boundary

Code is deleted only when its replacement passes parity and performance
tests. The intended deletions are:

- nested temporary expert page-in threads;
- unconditional reads of resident experts;
- decomposed MoE intermediate buffers made unused by fused kernels;
- dead or misleading comments that describe the superseded path.

The optional Metal backend is outside this change. Its measured full-model
path is slower and numerically different, so it is not used to claim the
42-token target.

## Verification gates

- red-green tests for expert cache hits, miss-only I/O, and invalid IDs;
- differential tests for both fused MoE kernels across supported quant types;
- full CTest pass;
- deterministic FP32, FP16, K8/V6, and opt-in K4/V2 output comparison;
- interleaved sustained decode trials on the exact local 26B-A4B model;
- 42 tok/s measured on AC before declaring the performance target met;
- 256K allocation and generation check with complete memory accounting;
- no architecture or model-name condition in changed production code.
