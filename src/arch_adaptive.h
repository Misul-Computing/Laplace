// arch_adaptive.h - feature-driven GGUF execution
#pragma once

#include "arch.h"
#include "topology.h"

#include <vector>

namespace Laplace {

class AdaptiveArch : public ModelArch {
public:
    explicit AdaptiveArch(TopologyPlan plan) : plan_(std::move(plan)) {}
    const char* name() const override { return "adaptive"; }
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
    bool needs_shared_rope_table() const override { return false; }

private:
    struct LayerTypeInfo {
        bool is_global = false;
        int head_dim = 0;
        int n_kv_heads = 0;
        int n_q_heads = 0;
        int sliding_window = 0;
        int rope_dim = 0;
        float rope_base = 0.0f;
    };
    TopologyPlan plan_;
    std::vector<LayerTypeInfo> layer_types_;

    // Proportional RoPE frequency factors for global layers.
    std::vector<float> rope_freqs_full_;  // [256]

    void attention_batch(int layer, int M, int pos0, KVCache& kv,
                         const LayerWeights& W, const LayerTypeInfo& lti,
                         const ModelConfig& cfg, ModelBuffers* buf);
    void moe_ffn(const LayerWeights& W, const ModelConfig& cfg,
                 int M, const float* residual, ModelBuffers* buf);
};

} // namespace Laplace
