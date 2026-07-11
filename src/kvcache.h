// kvcache.h - shared KV storage contract.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "laplace_kv.h"

namespace Laplace {

enum class KVCacheMode {
    FP32,
    FP16,
    LAPLACE,
};

class KVCache {
public:
    bool init(int n_layers, int n_kv_heads, int head_dim, int capacity,
              KVCacheMode mode = KVCacheMode::LAPLACE);
    void free();

    KVCacheMode mode() const { return mode_; }
    int capacity() const { return capacity_; }
    int n_layers() const { return n_layers_; }
    int n_kv_heads() const { return n_kv_heads_; }
    int head_dim() const { return head_dim_; }

    void load_k(int layer, int head, int pos, float* output) const;
    void load_v(int layer, int head, int pos, float* output) const;
    void store_k(int layer, int head, int pos, const float* input);
    void store_v(int layer, int head, int pos, const float* input);

    void load_k_wh(int layer, int head, int pos, float* output) const;
    void load_v_wh(int layer, int head, int pos, float* output) const;
    void store_k_wh(int layer, int head, int pos, const float* input);
    void store_v_wh(int layer, int head, int pos, const float* input);

    float dot_k_wh(int layer, int head, int pos, const float* query_wh) const;
    void weighted_add_v_wh(int layer, int head, int pos, float weight,
                           float* output_wh) const;
    void dot_k_all_wh(int layer, int head, int n_tokens,
                      const float* query_wh, float logit_scale,
                      float* scores, int first_token = 0) const;
    void weighted_add_v_all_wh(int layer, int head, int n_tokens,
                               const float* weights, float* output_wh,
                               int first_token = 0) const;
    void attention_all_wh(int layer, int head, int n_tokens,
                          const float* query_wh, float logit_scale,
                          float* output_wh, int first_token = 0) const;
    void attention_batch_all_wh(int layer, int head, int count,
                                const int* n_tokens,
                                const float* const* queries_wh,
                                float logit_scale, float* const* outputs_wh,
                                int first_token = 0) const;

    float* slot_k(int layer, int head, int pos);
    float* slot_v(int layer, int head, int pos);
    const float* slot_k(int layer, int head, int pos) const;
    const float* slot_v(int layer, int head, int pos) const;
    float* head_k(int layer, int head);
    float* head_v(int layer, int head);
    const float* head_k(int layer, int head) const;
    const float* head_v(int layer, int head) const;
    const uint16_t* head_k16(int layer, int head) const;
    const uint16_t* head_v16(int layer, int head) const;

    bool laplace_rotated() const {
        return laplace_ && laplace_->uses_rotation();
    }
    bool streaming() const { return laplace_ && laplace_->streaming(); }
    uint64_t stream_calls() const {
        return laplace_ ? laplace_->stream_calls() : 0;
    }
    uint64_t archive_read_bytes() const {
        return laplace_ ? laplace_->archive_read_bytes() : 0;
    }
    uint64_t archive_write_bytes() const {
        return laplace_ ? laplace_->archive_write_bytes() : 0;
    }
    void set_streaming(bool enabled) { streaming_ = enabled; }
#if defined(LAPLACE_KV_CAPTURE)
    bool set_research_bfp3();
    bool set_research_kivi_2();
    bool set_research_mlx_q2();
    bool set_research_turboquant_2_5();
    bool set_research_baseline(int key_bits, int value_bits,
                               int group = 128, int sink_tokens = 128,
                               int metadata_bits = 0,
                               int tail_key_bits = 0,
                               int tail_value_bits = 0);
#endif
    size_t encoded_bytes(int n_tokens) const {
        if (laplace_) return laplace_->encoded_bytes(n_tokens);
        int tokens = n_tokens < 0 ? 0
                   : n_tokens > capacity_ ? capacity_ : n_tokens;
        size_t element_bytes = mode_ == KVCacheMode::FP16
                             ? sizeof(uint16_t) : sizeof(float);
        return static_cast<size_t>(n_layers_) * n_kv_heads_ * tokens
             * head_dim_ * 2 * element_bytes;
    }
    size_t storage_bytes() const {
        size_t bytes = (k32_.capacity() + v32_.capacity()) * sizeof(float)
                     + (k16_.capacity() + v16_.capacity()) * sizeof(uint16_t);
        return bytes + (laplace_ ? laplace_->storage_bytes() : 0);
    }
    size_t archive_read_buffer_bytes() const {
        return laplace_ ? laplace_->archive_read_buffer_bytes() : 0;
    }
    size_t archive_bytes() const {
        return laplace_ ? laplace_->archive_bytes() : 0;
    }

private:
    size_t slot_index(int layer, int head, int pos) const {
        return ((static_cast<size_t>(layer) * n_kv_heads_ + head) * capacity_
                + pos) * head_dim_;
    }

    std::vector<float> k32_;
    std::vector<float> v32_;
    std::vector<uint16_t> k16_;
    std::vector<uint16_t> v16_;
    std::unique_ptr<LaplaceKV> laplace_;
    bool streaming_ = false;
#if defined(LAPLACE_KV_CAPTURE)
    int research_key_bits_ = 0;
    int research_value_bits_ = 0;
    int research_group_ = 0;
    int research_sink_tokens_ = 0;
    int research_metadata_bits_ = 0;
    int research_tail_key_bits_ = 0;
    int research_tail_value_bits_ = 0;
    bool research_bfp3_ = false;
    bool research_kivi_2_ = false;
    bool research_mlx_q2_ = false;
    bool research_turboquant_2_5_ = false;
#endif
    KVCacheMode mode_ = KVCacheMode::LAPLACE;
    int n_layers_ = 0;
    int n_kv_heads_ = 0;
    int head_dim_ = 0;
    int capacity_ = 0;
};

} // namespace Laplace
