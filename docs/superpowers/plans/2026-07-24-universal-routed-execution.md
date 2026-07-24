# Universal Routed Execution and Q4 KV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove universal routed-expert decode overhead and restore the measured fixed-width Q4 KV candidate without weakening quality gates.

**Architecture:** Add one shape-driven expert acquisition contract, then consume active experts through exact fused CPU kernels. Add K4/V2 as a fully counted opt-in codec for global layers while retaining K8/V6 automatic mode until quality evidence permits promotion.

**Tech Stack:** C++20, Apple Silicon NEON/DOTPROD/I8MM, GGUF tensors, existing thread pool, CMake and CTest.

## Global Constraints

- Do not dispatch on architecture name, model filename, layer number, or 26B-specific dimensions.
- Do not add dependencies.
- Do not claim 42 tok/s until sustained AC measurements reach it.
- Do not make K4/V2 automatic unless the existing quality protocol passes.
- Preserve FP16, FP32, and K8/V6 modes.
- Use tests before production changes.

---

### Task 1: Observable miss-only expert acquisition

**Files:**
- Modify: `src/laplace_moe.h`
- Modify: `src/laplace_moe.cpp`
- Create: `tests/test_laplace_moe.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `ExpertAcquireStats { requested, hits, misses, bytes_read }`
- Produces: `ExpertAcquireTicket LaplaceMoE::prefetch(const Tensor*, const int*, int)`
- Produces: `ExpertAcquireStats LaplaceMoE::wait(ExpertAcquireTicket)`
- Produces: synchronous `LaplaceMoE::acquire(const Tensor*, const int*, int)`
- Produces: a test-only read observer compiled only into `test_laplace_moe`

- [ ] **Step 1: Write a failing cache-hit test**

Create a temporary file-backed stacked tensor with four equal expert slices.
Acquire experts `{1, 3}` twice and assert that the second call increases hits
without increasing miss reads.

- [ ] **Step 2: Verify the test fails**

Run:

```bash
cmake --build build-make -j8 --target test_laplace_moe
./build-make/test_laplace_moe
```

Expected: build failure because `LaplaceMoE::acquire` is absent.

- [ ] **Step 3: Implement the minimum acquisition state**

Reuse the existing cache map. Make one locked critical section decide hits and
misses. Validate each ID against `tensor->dims[2]`. The first implementation
may complete the ticket inline. Read only misses and update stats after
successful reads.

- [ ] **Step 4: Verify focused tests**

Run the test executable and expect all hit, miss, invalid-ID, and byte-count
checks to pass.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt src/laplace_moe.h src/laplace_moe.cpp tests/test_laplace_moe.cpp
git commit -m "Add miss-only expert acquisition"
```

### Task 2: Persistent bounded I/O workers

**Files:**
- Modify: `src/laplace_moe.cpp`
- Modify: `tests/test_laplace_moe.cpp`

**Interfaces:**
- Consumes: `LaplaceMoE::prefetch`, `LaplaceMoE::wait`, and `LaplaceMoE::acquire`
- Produces: one process-wide bounded I/O worker group with synchronous acquire completion

- [ ] **Step 1: Add a failing worker-reuse test**

Expose a test-only worker creation counter. Perform acquisitions across
multiple tensors and assert the count is bounded by the configured I/O worker
count rather than acquisition count.

- [ ] **Step 2: Verify the test fails**

Run `./build-make/test_laplace_moe`. Expected: the counter grows for repeated
acquisition under the existing temporary-thread implementation.

- [ ] **Step 3: Replace nested thread construction**

Build workers once on first use. Queue one contiguous expert range per miss.
`prefetch` returns a shared completion ticket, `wait` blocks on its condition
variable, and `acquire` calls both. Keep the current `LAPLACE_IO_THREADS`
bound.

- [ ] **Step 4: Verify**

Run `./build-make/test_laplace_moe` and the full test suite.

- [ ] **Step 5: Commit**

```bash
git add src/laplace_moe.cpp tests/test_laplace_moe.cpp
git commit -m "Reuse bounded expert IO workers"
```

### Task 3: Route-conditioned fused MoE kernels

**Files:**
- Modify: `src/matmul.h`
- Modify: `src/matmul.cpp`
- Modify: `src/matmul_simd.cpp`
- Modify: `tests/test_matmul.cpp`

**Interfaces:**
- Produces: `fused_moe_gate_up_geglu(...)`
- Produces: `fused_moe_down_accumulate(...)`
- Falls back by returning `false` for unsupported quant types

- [ ] **Step 1: Add differential tests**

For Q4_0, Q4_K, Q6_K, and Q8_0 stacked tensors, compare:

```text
gate GEMV + up GEMV + GeGLU
```

against `fused_moe_gate_up_geglu`, and compare:

```text
per-expert down GEMV + weighted sum
```

against `fused_moe_down_accumulate`.

- [ ] **Step 2: Verify red**

Build and run `test_matmul`. Expected: failure because both APIs are absent.

- [ ] **Step 3: Implement paired row traversal**

Quantize each activation once. Partition `expert_count * output_rows`
contiguously across the existing thread pool. Reuse the existing quantized dot
functions. Apply GeGLU or routing accumulation immediately after the required
dots.

- [ ] **Step 4: Verify green**

Run SIMD and scalar test targets. Require existing relative-error tolerances
and finite outputs.

- [ ] **Step 5: Commit**

```bash
git add src/matmul.h src/matmul.cpp src/matmul_simd.cpp tests/test_matmul.cpp
git commit -m "Fuse routed expert decode kernels"
```

### Task 4: Adaptive executor integration

**Files:**
- Modify: `src/arch_adaptive.cpp`
- Modify: `src/model.cpp`
- Modify: `tests/test_arch.cpp`

**Interfaces:**
- Consumes: miss-only acquisition and fused MoE kernels
- Produces: existing decomposed fallback for unsupported tensor formats

- [ ] **Step 1: Add an execution-choice test**

Use a synthetic topology and tensors to assert that support is decided by
tensor shape and quant type, not metadata namespace.

- [ ] **Step 2: Verify red**

Run `./build-make/test_arch`. Expected: no fused capability decision exists.

- [ ] **Step 3: Integrate**

Acquire gate/up experts immediately after routing. Start down acquisition
before gate/up compute. Use fused APIs when supported, otherwise preserve the
existing path. Remove only intermediate storage proven unused.

- [ ] **Step 4: Verify**

Run `test_arch`, `test_matmul`, and a deterministic local-model parity prompt.

- [ ] **Step 5: Commit**

```bash
git add src/arch_adaptive.cpp src/model.cpp tests/test_arch.cpp
git commit -m "Use routed expert execution plan"
```

### Task 5: Fully counted fixed-width K4/V2 codec

**Files:**
- Create: `src/laplace_kv_q4.h`
- Create: `src/laplace_kv_q4.cpp`
- Modify: `src/kvcache.h`
- Modify: `src/kvcache.cpp`
- Modify: `src/main.cpp`
- Modify: `CMakeLists.txt`
- Create: `tests/test_laplace_kv_q4.cpp`

**Interfaces:**
- Produces: resident and streaming `LaplaceKVQ4`
- Produces: explicit `--laplace-kv-q4` research flag
- Produces: complete encoded, resident, archive, and scratch byte accounting

- [ ] **Step 1: Add fixed-equation and accounting tests**

Use deterministic K/V tiles at head dimensions 64, 96, 256, and 512. Compare
decoded attention with the independent scalar equations. Assert every code,
metadata, alignment, mutable-tail, archive, and scratch byte.

- [ ] **Step 2: Verify red**

Build the new test target. Expected: failure because `LaplaceKVQ4` is absent.

- [ ] **Step 3: Implement resident tiles**

Port the already measured token-major K4 and coordinate-major V2 traversal
from the standalone benchmark. Store all six metadata fields explicitly.
Keep the mutable tile FP32.

- [ ] **Step 4: Implement streaming tiles**

Reuse the existing unlinked-file and bounded-read conventions. Do not add an
entropy stream or metadata prepass.

- [ ] **Step 5: Integrate as opt-in global storage**

Map `--laplace-kv-q4` to Q4 only for unbounded/global layers. Keep sliding
layers FP16. Leave default `--laplace-kv` unchanged.

- [ ] **Step 6: Verify**

Run the codec tests, full CTest, and deterministic model comparison.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt src/laplace_kv_q4.h src/laplace_kv_q4.cpp src/kvcache.h src/kvcache.cpp src/main.cpp tests/test_laplace_kv_q4.cpp
git commit -m "Restore fixed-width Q4 KV research mode"
```

### Task 6: Performance, quality, and cleanup gates

**Files:**
- Modify only production files whose old path is proven unused
- Modify: `README.md`
- Modify: `research/laplace_kv/RESULTS.md`

**Interfaces:**
- Produces: measured results and honest automatic-mode decision

- [ ] **Step 1: Run fresh verification**

```bash
cmake --build build-make -j8
ctest --test-dir build-make --output-on-failure
```

- [ ] **Step 2: Run deterministic parity**

Compare FP32, FP16, K8/V6, and K4/V2 output tokens on the local model.

- [ ] **Step 3: Measure sustained decode**

Run interleaved modes for at least 128 generated tokens after warmup on AC.
Report median and range. Do not claim 42 tok/s unless the measured median
reaches it.

- [ ] **Step 4: Run long-context gates**

Run the registered 2,048-prediction quality evaluation, retrieval checks, and
256K allocation check. Keep K4/V2 opt-in on any gate failure.

- [ ] **Step 5: Delete superseded code**

Delete nested page-in functions and unused intermediate buffers only when
call-site search and tests prove they are unused.

- [ ] **Step 6: Commit**

```bash
git add README.md research/laplace_kv/RESULTS.md src tests
git commit -m "Document routed execution results"
```
