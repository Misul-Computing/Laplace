// laplace_moe.h - expert working-set cache + SSD staging
//
// Routed expert slices live in a bounded RAM LRU (default ~4 GB, scaled
// by quant block size). Misses are pread from the model file. GEMV uses
// pointers into those slots, not the 15 GB expert mmap.
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
    static bool streaming_enabled() { return streaming_enabled_; }
    static void set_streaming(bool v) { streaming_enabled_ = v; }

    static void hint(const Tensor* tensor, int expert_idx);
    static ExpertAcquireTicket prefetch(
        const Tensor* tensor, const int* expert_idx, int n);
    static ExpertAcquireStats wait(const ExpertAcquireTicket& ticket);
    static ExpertAcquireStats acquire(
        const Tensor* tensor, const int* expert_idx, int n);
    static int io_worker_count();

    // Resolve selected experts into RAM. If dest is non-null, pack a
    // contiguous view (dims[2] = n) for tests and scalar fallbacks.
    // If bases is non-null, bases[i] points at the cached slice (no extra
    // copy). GEMV should use bases when present.
    static ExpertAcquireStats load_experts(
        const Tensor& src, const int* ids, int n, Tensor* dest,
        const uint8_t** bases = nullptr);

    static void drop_mmap(const Tensor* tensor);

    static void set_file_fd(int fd);
    static void set_mmap_base(const uint8_t* base);

    static size_t per_expert_bytes(const Tensor* tensor);
    static const uint8_t* expert_data(const Tensor* tensor, int expert_idx);

    static void touch_expert(const Tensor* tensor, int expert_idx);
    static void evict_cold(int k_tokens);
    static void set_cache_budget(size_t bytes);
    static void set_current_token(int n);

private:
    static bool streaming_enabled_;
};

} // namespace Laplace
