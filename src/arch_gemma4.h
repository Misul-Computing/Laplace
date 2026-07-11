// arch_gemma4.h - Gemma 4 26B-A4B MoE architecture
#pragma once

#include "arch.h"

#include <vector>

namespace Laplace {

class Gemma4Arch : public ModelArch {
public:
    const char* name() const override { return "gemma4"; }
    bool load_config(const GGUFContext& gguf, ModelConfig* cfg) override;
    bool load_weights(const GGUFContext& gguf, const ModelConfig& cfg,
                      std::vector<LayerWeights>* layers,
                      std::vector<bool>* is_attention,
                      std::vector<int>* kv_layer_idx) override;
    void reserve(const ModelConfig& cfg, int max_seq, int max_batch,
                 ModelBuffers* buf) override;
    void forward_layer(int layer, const LayerWeights& W, const ModelConfig& cfg,
                       int M, int pos0, KVCache& kv, ModelBuffers* buf,
                       float* checkpoints) override;

private:
    struct LayerTypeInfo {
        bool is_global = false;
        int head_dim = 256;
        int n_kv_heads = 8;
        int n_q_heads = 16;
    };
    std::vector<LayerTypeInfo> layer_types_;

    // Dual RoPE tables (computed in reserve).
    std::vector<float> rope_cos_swa_;   // [max_seq, 128]
    std::vector<float> rope_sin_swa_;
    std::vector<float> rope_cos_full_;  // [max_seq, 256]
    std::vector<float> rope_sin_full_;
    int rope_pairs_swa_ = 0;
    int rope_pairs_full_ = 0;

    // Proportional RoPE frequency factors for global layers.
    std::vector<float> rope_freqs_full_;  // [256]
    float rope_freq_base_swa_ = 10000.0f; // sliding layer RoPE base

    // Temporary padded buffer for storing 256-dim K/V into 512-dm cache slots.
    std::vector<float> pad_buf_;

    void attention_batch(int layer, int M, int pos0, KVCache& kv,
                         const LayerWeights& W, const LayerTypeInfo& lti,
                         const ModelConfig& cfg, ModelBuffers* buf);
    void moe_ffn(const LayerWeights& W, const ModelConfig& cfg,
                 int M, const float* residual, ModelBuffers* buf);
};

} // namespace Laplace
