// kvcache.h - shared KV storage contract.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "laplace_kv_adaptive.h"
#include "laplace_kv_q4.h"

namespace Laplace {

enum class KVCacheMode {
    FP32,
    FP16,
    LAPLACE,
    LAPLACE_Q4,
};

enum class KVStorageKind {
    FP32,
    FP16,
    Adaptive,
    FixedQ4,
};

struct KVLayerConfig {
    int n_kv_heads = 0;
    int head_dim = 0;
    int capacity = 0;
    int sliding_window = 0;
    KVCacheMode mode = KVCacheMode::LAPLACE;
};

class KVCache {
public:
    bool init(int n_layers, int n_kv_heads, int head_dim, int capacity,
              KVCacheMode mode = KVCacheMode::LAPLACE);
    bool init(const std::vector<KVLayerConfig>& layers);
    void free();

    KVCacheMode mode() const { return mode_; }
    KVCacheMode mode(int layer) const;
    KVStorageKind storage_kind() const;
    KVStorageKind storage_kind(int layer) const;
    int capacity() const { return capacity_; }
    int capacity(int layer) const;
    int n_layers() const { return n_layers_; }
    int n_kv_heads() const { return n_kv_heads_; }
    int n_kv_heads(int layer) const;
    int head_dim() const { return head_dim_; }
    int head_dim(int layer) const;
    bool ring(int layer) const;

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
        return (laplace_ && laplace_->uses_rotation()) ||
               (laplace_q4_ && laplace_q4_->uses_rotation());
    }
    bool laplace_rotated(int layer) const;
    bool streaming() const;
    uint64_t stream_calls() const;
    uint64_t archive_read_bytes() const;
    uint64_t archive_write_bytes() const;
    uint64_t q4_tiles() const;
    uint64_t k8_tiles() const;
    uint64_t fp16_tiles() const;
    LaplaceKVAdaptiveFormat tile_format(
        int layer, int head, int tile) const;
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
    size_t encoded_bytes(int n_tokens) const;
    size_t logical_scalars(int n_tokens) const;
    size_t storage_bytes() const;
    size_t archive_read_buffer_bytes() const;
    size_t archive_bytes() const;

private:
    size_t slot_index(int layer, int head, int pos) const {
        return ((static_cast<size_t>(layer) * n_kv_heads_ + head) * capacity_
                + pos) * head_dim_;
    }

    std::vector<float> k32_;
    std::vector<float> v32_;
    std::vector<uint16_t> k16_;
    std::vector<uint16_t> v16_;
    std::unique_ptr<LaplaceKVAdaptive> laplace_;
    std::unique_ptr<LaplaceKVQ4> laplace_q4_;
    std::vector<std::unique_ptr<KVCache>> layer_caches_;
    std::vector<KVLayerConfig> layer_configs_;
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

    KVCache* layer_cache(int layer);
    const KVCache* layer_cache(int layer) const;
    int physical_pos(int layer, int pos) const;
};

} // namespace Laplace
