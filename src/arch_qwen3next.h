// arch_qwen3next.h - Qwen3-Next / Qwen3.5 hybrid (full attention + DeltaNet)
#pragma once

#include <vector>

#include "arch.h"

namespace Laplace {

class Qwen3NextArch : public ModelArch {
public:
    const char* name() const override { return "qwen3next"; }
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
    void attention_batch(int layer, int M, int pos0, KVCache& kv,
                         const LayerWeights& W, const ModelConfig& cfg,
                         ModelBuffers* buf);
    void deltanet_batch(int layer, int M, const LayerWeights& W,
                        const ModelConfig& cfg, ModelBuffers* buf,
                        float* checkpoints);

    std::vector<int> kv_layer_idx_;  // global layer -> dense KV-cache layer index
};

} // namespace Laplace
