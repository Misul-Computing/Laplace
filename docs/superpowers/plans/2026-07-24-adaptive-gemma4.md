# Adaptive Gemma 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the local Gemma 4 26B-A4B text GGUF through a generic topology plan with correct, compact 256K KV storage.

**Architecture:** Build an immutable capability plan from GGUF metadata and tensor topology, then execute it through one adaptive model path. Configure KV storage per layer and compute RoPE only for active positions.

**Tech Stack:** C++20, CMake, GGUF, Apple Silicon NEON/Metal, existing LaplaceKV.

## Global Constraints

- Do not dispatch inference code on `general.architecture`.
- Do not add dependencies.
- Preserve existing architecture paths as temporary fallbacks.
- Preserve FP16 and FP32 comparison modes.
- Fail before inference when topology is incomplete or contradictory.

---

### Task 1: Generic topology synthesis

**Files:**
- Create: `src/topology.h`
- Create: `src/topology.cpp`
- Modify: `src/model.h`
- Modify: `src/model.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/test_arch.cpp`

**Interfaces:**
- Produces: `bool synthesize_topology(const GGUFContext&, TopologyPlan*, std::string*)`
- Produces: immutable `TopologyPlan::layers` records consumed by the adaptive executor

- [ ] Add a synthetic GGUF test that describes mixed sliding/global attention and MoE under a non-Gemma architecture label.
- [ ] Run `./build-make/test_arch` and confirm the new test fails because the planner is absent.
- [ ] Implement the smallest metadata-and-shape planner that accepts the fixture and rejects inconsistent dimensions.
- [ ] Run `./build-make/test_arch` and confirm it passes.

### Task 2: Adaptive execution path

**Files:**
- Create: `src/arch_adaptive.h`
- Create: `src/arch_adaptive.cpp`
- Modify: `src/laplace_arch.cpp`
- Modify: `src/model.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/test_arch.cpp`

**Interfaces:**
- Consumes: `TopologyPlan`
- Produces: `AdaptiveArch`, selected by planner capability rather than architecture name

- [ ] Add a factory/model-init test proving the adaptive path is selected for the renamed fixture.
- [ ] Run the focused test and confirm it fails on architecture-name dispatch.
- [ ] Move the Gemma 4 feature execution into `AdaptiveArch`, replace hard-coded metadata namespace and layer assumptions with the plan, and remove `Gemma4Arch`.
- [ ] Run `./build-make/test_arch` and confirm it passes.

### Task 3: Heterogeneous long-context KV and lazy RoPE

**Files:**
- Modify: `src/kvcache.h`
- Modify: `src/kvcache.cpp`
- Modify: `src/model.h`
- Modify: `src/model.cpp`
- Modify: `src/arch_adaptive.h`
- Modify: `src/arch_adaptive.cpp`
- Test: `tests/test_kvcache.cpp`
- Test: `tests/test_arch.cpp`

**Interfaces:**
- Produces: `KVLayerConfig { heads, head_dim, capacity, sliding_window, mode }`
- Produces: `KVCache::init(const std::vector<KVLayerConfig>&)`

- [ ] Add a ring-wrap test that stores absolute positions beyond a sliding layer's capacity and compares reads with the last-window FP32 reference.
- [ ] Run `./build-make/test_kvcache` and confirm the test fails.
- [ ] Add per-layer KV subcaches with logical-position mapping and exact storage accounting.
- [ ] Add a topology allocation test for five global and twenty-five sliding layers at 262,144 tokens.
- [ ] Replace full-context RoPE tables with batch-position scratch generation.
- [ ] Run `./build-make/test_kvcache ./build-make/test_arch` and confirm both pass.

### Task 4: Metal, output, and model validation

**Files:**
- Modify only files directly implicated by reproduced failures.
- Test: existing test targets plus the local GGUF.

**Interfaces:**
- Produces: deterministic text generation without shader compile errors or leaked terminal control tokens

- [ ] Reproduce the Metal compile error with the local GGUF and add the smallest relevant test.
- [ ] Fix the invalid Metal address-space casts and run `./build-make/test_metal`.
- [ ] Add terminal-token stopping or filtering using tokenizer-declared special tokens.
- [ ] Build with `cmake --build build-make -j8`.
- [ ] Run `ctest --test-dir build-make --output-on-failure`.
- [ ] Run the local model in FP32, FP16, and LaplaceKV modes on a fixed greedy prompt and compare tokens.
- [ ] Reserve 262,144 context, record resident/archive allocation, then run `--bench` at a practical prompt length.

