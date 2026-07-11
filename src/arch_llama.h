// arch_llama.h - standard Llama/Qwen2/Qwen3 transformer
#pragma once

#include "arch.h"

namespace Laplace {

class LlamaArch : public ModelArch {
public:
    const char* name() const override { return "llama"; }
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
    // Attention using WH-domain K/V storage. Avoids inverse WH per position;
    // instead WH-transforms Q once and inverse WHs the attention output once.
    void attention_batch_wh(int layer, int M, int pos0, KVCache& kv,
                            const LayerWeights& W, const ModelConfig& cfg,
                            ModelBuffers* buf);
};

} // namespace Laplace
