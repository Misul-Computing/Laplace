# Adaptive Gemma 4 Text Inference Design

## Scope

Run the local `BugTraceAI-Apex-G4-26B-Q4.gguf` through a generic,
feature-driven execution path. Text inference is in scope. Vision and the
separate MTP assistant model are not.

## Topology synthesis

`Model` first asks the adaptive planner to inspect GGUF metadata and tensors.
The planner uses the metadata namespace only to read standardized fields. It
does not dispatch on `general.architecture`.

The planner produces immutable model and layer records. Each layer records
the operations and exact dimensions needed by the hot loop:

- attention head counts, head dimensions, sliding window, and RoPE parameters
- projected or shared K/V
- dense gated FFN
- top-k routed experts and their scales
- normalization, residual, layer scale, and output transforms

Tensor presence and shape select capabilities. Missing or contradictory
inputs fail before memory reservation with a precise diagnostic.

Gemma 4 uses this path and its dedicated architecture class is removed.
Existing architecture classes remain temporary fallbacks for models that the
adaptive planner cannot yet describe.

## Runtime

The planner runs once during model loading. Inference walks the immutable
layer plan, so the hot loop performs no metadata lookups and no architecture
name checks.

RoPE values are generated for the positions in the current batch instead of
precomputing full 256K tables.

## KV cache

KV storage is configured per layer:

- sliding layers use their real head count and dimension with a 1,024-token
  logical ring
- global layers use their real head count and dimension for the requested
  context
- LaplaceKV remains the default compression for long global caches
- FP16 and FP32 remain exact comparison paths

Logical positions are preserved when a sliding layer wraps. Attention reads
only the declared window. Reported storage includes every layer's actual
allocation.

## Correctness and performance gates

- topology tests use a synthetic GGUF whose architecture label is changed
  without changing its feature metadata or tensors
- invalid tensor and metadata combinations fail during planning
- sliding-cache tests cross the ring boundary and match an FP32 reference
- the full unit suite passes
- the local 26B-A4B model produces the expected deterministic response
- 256K reservation completes without allocating every layer at the largest
  global shape
- FP16 and LaplaceKV outputs are compared on a fixed prompt before any quality
  claim
- prefill and decode timings are reported as measurements, not projections

