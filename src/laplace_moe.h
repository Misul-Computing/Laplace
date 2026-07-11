// laplace_moe.h - Mac M-series SSD expert streaming for MoE models
//
// Active routed experts can be paged in from the model file and retained
// within a bounded resident set.
#pragma once

#include <cstddef>
#include <cstdint>

#include "tensor.h"

namespace Laplace {

class LaplaceMoE {
public:
    // Global mode: when true, dense weights are pinned and expert tensors
    // stream from SSD. Set by Model::plan_residency()
    // based on model size vs physical RAM.
    static bool streaming_enabled() { return streaming_enabled_; }
    static void set_streaming(bool v) { streaming_enabled_ = v; }

    static void pagein_expert_mt(const Tensor* tensor, int expert_idx);
    static void pagein_all_mt(const Tensor* tensor, const int* expert_idx, int n);

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
