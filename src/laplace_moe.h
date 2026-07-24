// laplace_moe.h - Mac M-series SSD expert streaming for MoE models
//
// Active routed experts can be paged in from the model file and retained
// within a bounded resident set.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "tensor.h"

namespace Laplace {

struct ExpertAcquireStats {
    int requested = 0;
    int hits = 0;
    int misses = 0;
    int invalid = 0;
    size_t bytes_read = 0;
};

struct ExpertAcquireState;

class ExpertAcquireTicket {
public:
    ExpertAcquireTicket() = default;

private:
    explicit ExpertAcquireTicket(std::shared_ptr<ExpertAcquireState> state)
        : state_(std::move(state)) {}
    std::shared_ptr<ExpertAcquireState> state_;
    friend class LaplaceMoE;
};

class LaplaceMoE {
public:
    // Global mode: when true, dense weights are pinned and expert tensors
    // stream from SSD. Set by Model::plan_residency()
    // based on model size vs physical RAM.
    static bool streaming_enabled() { return streaming_enabled_; }
    static void set_streaming(bool v) { streaming_enabled_ = v; }

    static ExpertAcquireTicket prefetch(
        const Tensor* tensor, const int* expert_idx, int n);
    static ExpertAcquireStats wait(const ExpertAcquireTicket& ticket);
    static ExpertAcquireStats acquire(
        const Tensor* tensor, const int* expert_idx, int n);
    static int io_worker_count();

    static void set_file_fd(int fd);
    static void set_mmap_base(const uint8_t* base);

    // Size of one expert's slice in a 3D stacked tensor.
    static size_t per_expert_bytes(const Tensor* tensor);

    // Pointer to a single expert's slice in a 3D stacked tensor.
    static const uint8_t* expert_data(const Tensor* tensor, int expert_idx);

    static void touch_expert(const Tensor* tensor, int expert_idx);
    static void evict_cold(int k_tokens);
    static void set_cache_budget(size_t bytes);
    static void set_current_token(int n);

private:
    static bool streaming_enabled_;
};

} // namespace Laplace
