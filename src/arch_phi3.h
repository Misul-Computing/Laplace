// arch_phi3.h - Microsoft Phi3 / Phi-3-mini transformer
//
// Distinguishing features vs Llama:
//   - Fused QKV projection (single weight `attn_qkv.weight` covering
//     [Q | K | V], so the matmul is one call instead of three).
//   - Fused gate+up projection in the FFN (single weight
//     `ffn_gate_up.weight` covering [gate | up]).
//   - Phi3 RMSNorm uses the gain+1 trick: y = x / sqrt(mean(x^2) + eps)
//     * (1 + w) instead of the standard * w.  See ops::rmsnorm_phi3.
//   - Optional QKV bias (`attn_qkv.bias`) - not all Phi3 variants
//     emit it; loaded if present, ignored otherwise.
//   - Sliding-window attention (`attention.sliding_window`) - not
//     wired in this first pass; the full attention mask covers the
//     window.  The non-windowed path is identical to Llama for the
//     case where the context fits in a single window.
#pragma once

#include "arch.h"

namespace Laplace {

class Phi3Arch : public ModelArch {
public:
    const char* name() const override { return "phi3"; }
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
};

} // namespace Laplace
