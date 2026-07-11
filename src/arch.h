// arch.h - architecture abstraction layer for Laplace
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "gguf.h"
#include "kvcache.h"

namespace Laplace {

struct ModelConfig;
struct LayerWeights;

// Activation and persistent state scratch buffers sized for max_batch rows.
struct ModelBuffers {
    std::vector<float> x;          // hidden states [M, hidden]
    std::vector<float> xb;         // sub-block outputs [M, hidden]
    std::vector<float> x_norm;     // post-RMSNorm [M, hidden]

    // Attention projections.  Size is architecture-specific:
    //   - Llama/Qwen2/Qwen3: M * (Hq*Dh + 2*Hk*Dh)
    //   - Qwen3-Next:        M * (2*Hq*Dh + 2*Hk*Dh) to hold gated Q
    std::vector<float> qkv;

    std::vector<float> attn_out;      // [M, n_q_heads * head_dim]
    std::vector<float> attn_logits;   // [n_q_heads * max_seq]

    std::vector<float> ffn_gate;      // [M, intermediate]
    std::vector<float> ffn_up;        // [M, intermediate]
    std::vector<float> ffn_hidden;    // [M, intermediate]

    // DeltaNet scratch (Qwen3-Next only)
    std::vector<float> dnet_qkv;
    std::vector<float> dnet_gate;
    std::vector<float> dnet_b_proj;
    std::vector<float> dnet_a_proj;
    std::vector<float> dnet_o;
    std::vector<float> dnet_normed;

    // SSM persistent state (Qwen3-Next only)
    std::vector<float> ssm_conv_state;
    std::vector<float> ssm_recurrent;

    // MoE scratch (Gemma4)
    std::vector<float> moe_router_logits;  // [M, n_experts]
    std::vector<float> moe_expert_out;     // [M, hidden] accumulator

    // RoPE tables: [max_seq, rope_pairs]
    std::vector<float> rope_cos;
    std::vector<float> rope_sin;
    int rope_pairs = 0;
};

class ModelArch {
public:
    virtual ~ModelArch() = default;
    virtual const char* name() const = 0;

    // Fill cfg from GGUF metadata.  Does not load token/output head weights.
    virtual bool load_config(const GGUFContext& gguf, ModelConfig* cfg) = 0;

    // Discover per-layer tensors.  Sets is_attention/kv_layer_idx maps.
    virtual bool load_weights(const GGUFContext& gguf, const ModelConfig& cfg,
                              std::vector<LayerWeights>* layers,
                              std::vector<bool>* is_attention,
                              std::vector<int>* kv_layer_idx) = 0;

    // Size architecture-specific scratch buffers for (max_seq, max_batch).
    virtual void reserve(const ModelConfig& cfg, int max_seq, int max_batch,
                         ModelBuffers* buf) = 0;

    // Run one transformer block: buf->x is the residual stream in/out.
    virtual void forward_layer(int layer, const LayerWeights& W,
                               const ModelConfig& cfg, int M, int pos0,
                               KVCache& kv, ModelBuffers* buf,
                               float* checkpoints) = 0;
};

// Static factory: maps general.architecture strings to implementations.
std::unique_ptr<ModelArch> create_arch(const std::string& name);

} // namespace Laplace
