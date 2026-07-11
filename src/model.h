// model.h - small LLM inference engine with pluggable architectures
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "arch.h"
#include "gguf.h"
#include "kvcache.h"
#include "tensor.h"

namespace Laplace {

struct ModelConfig {
    int n_layers        = 0;
    int n_q_heads       = 0;    // full-attention layer dims
    int n_kv_heads      = 0;
    int head_dim        = 0;
    int hidden          = 0;
    int intermediate    = 0;
    int vocab           = 0;
    int max_seq_len     = 0;
    float rms_eps       = 1e-6f;
    float rope_freq_base = 1e7f;
    int rope_dim_count  = 64;
    int rope_sections[4] = {11, 11, 10, 0};
    bool tied_lm_head   = false;

    // DeltaNet (linear-attention) layer dims
    int ssm_group_count  = 16;
    int ssm_state_size   = 128;
    int ssm_inner_size   = 2048;
    int ssm_conv_kernel  = 4;
    int ssm_time_step_rank = 16;

    // Gemma4 MoE
    int n_experts        = 0;    // total routed experts (e.g. 128)
    int n_experts_used   = 0;    // top-k per token (e.g. 8)
    int expert_inter     = 0;    // per-expert intermediate size (e.g. 704)
    float logit_softcap  = 0.0f; // 0 = disabled; Gemma4 uses 30.0
    float embed_scale    = 1.0f; // multiply embeddings by this (Gemma: sqrt(hidden))
};

struct LayerWeights {
    // Always present
    const Tensor* attn_norm          = nullptr;  // input RMSNorm
    const Tensor* post_attn_norm     = nullptr;  // pre-FFN RMSNorm
    const Tensor* ffn_norm           = nullptr;
    const Tensor* ffn_gate           = nullptr;
    const Tensor* ffn_up             = nullptr;
    const Tensor* ffn_down           = nullptr;

    // DeltaNet fused in_proj_qkv [H, 3*inner]
    const Tensor* attn_qkv           = nullptr;

    // Full-attention only
    const Tensor* attn_q             = nullptr;  // [Hq*Dh, H] or gated [2*Hq*Dh, H]
    const Tensor* attn_k             = nullptr;  // [Hk*Dh, H]
    const Tensor* attn_v             = nullptr;  // [Hk*Dh, H]
    const Tensor* attn_q_bias        = nullptr;  // [Hq*Dh]
    const Tensor* attn_k_bias        = nullptr;  // [Hk*Dh]
    const Tensor* attn_v_bias        = nullptr;  // [Hk*Dh]
    const Tensor* attn_q_norm        = nullptr;  // [Dh]
    const Tensor* attn_k_norm        = nullptr;  // [Dh]
    const Tensor* attn_output        = nullptr;  // O projection [H, Hq*Dh]

    // SSM / DeltaNet only
    const Tensor* attn_gate          = nullptr;  // in_proj_z [inner,H]  (the gate)
    const Tensor* ssm_a              = nullptr;  // A = -exp(A_log), [G]
    const Tensor* ssm_conv1d         = nullptr;  // [kernel, conv_dim]
    const Tensor* ssm_dt_bias        = nullptr;  // [G]
    const Tensor* ssm_alpha          = nullptr;  // [H, G] -> a_t (decay input)
    const Tensor* ssm_beta           = nullptr;  // [H, G] -> b_t (write strength)
    const Tensor* ssm_norm           = nullptr;  // per-group RMSNorm [D]
    const Tensor* ssm_out            = nullptr;  // O projection [H, inner]

    // MoE (Gemma4)
    const Tensor* moe_gate_inp       = nullptr;  // router [hidden, n_experts]
    const Tensor* moe_gate_inp_scale = nullptr;  // router pre-scale [hidden]
    const Tensor* moe_gate_up_exps   = nullptr;  // fused gate+up [hidden, 2*expert_inter, n_experts]
    const Tensor* moe_down_exps      = nullptr;  // down [expert_inter, hidden, n_experts]
    const Tensor* moe_down_exps_scale= nullptr;  // per-expert down scale [n_experts]
    const Tensor* pre_ffw_norm_2     = nullptr;  // pre-MoE router norm [hidden]
    const Tensor* post_ffw_norm_1    = nullptr;  // post dense FFN norm [hidden]
    const Tensor* post_ffw_norm_2    = nullptr;  // post MoE norm [hidden]
    const Tensor* post_ffw_norm      = nullptr;  // post combined FFN norm [hidden]
    const Tensor* layer_output_scale = nullptr;  // scalar layer output scale [1]
};

class Model {
public:
    bool init(const GGUFContext& gguf);

    const ModelConfig& config() const { return cfg_; }
    const Tensor* output_norm()    const { return output_norm_; }
    const Tensor* output_weight()  const { return output_; }

    bool reserve(int max_seq_len, int max_batch = 8);
    int max_batch() const { return max_batch_; }

    // Forward one token at position `pos`. Writes next-token logits into
    // `logits` (cfg_.vocab floats).
    void forward(int token, int pos, KVCache& kv, float* logits);

    // Forward M tokens at positions pos0..pos0+M-1 in one batched pass
    // (weights stream once per layer). Writes logits for every position into
    // logits[M * vocab]. If `checkpoints` is non-null it must hold
    // M * ssm_state_floats() floats; after token m is absorbed, the full SSM
    // state (conv + recurrent, all layers) is copied to checkpoint m so a
    // speculative decoder can roll back to any accepted prefix.
    void forward_batch(const int* tokens, int M, int pos0, KVCache& kv,
                       float* logits, float* checkpoints = nullptr);

    // Floats per SSM-state checkpoint, and rollback to a saved checkpoint.
    size_t ssm_state_floats() const {
        return buffers_.ssm_conv_state.size() + buffers_.ssm_recurrent.size();
    }
    void restore_ssm_state(const float* checkpoint);

    bool is_attention_layer(int i) const { return is_attention_[i]; }
    int max_seq() const { return max_seq_; }

private:
    ModelConfig cfg_;
    std::unique_ptr<ModelArch> arch_;
    const Tensor* token_embd_  = nullptr;
    const Tensor* output_norm_ = nullptr;
    const Tensor* output_      = nullptr;  // LM head (may be tied to token_embd_)
    std::vector<LayerWeights> layers_;
    std::vector<bool> is_attention_;  // per layer: true = full attention, false = DeltaNet
    std::vector<int> kv_layer_idx_;   // global layer -> dense KV-cache layer index (-1 for SSM layers)

    int max_seq_ = 0;
    int max_batch_ = 1;
    bool streaming_experts_ = false;  // dense pinned, experts stream from SSD

    ModelBuffers buffers_;

    // Decide which weights stay resident and which stream, based on model
    // size vs physical RAM. Faults dense weights into RAM so the OS keeps
    // them resident; leaves expert tensors to stream on demand when the
    // model does not fit. Called at the end of init().
    void plan_residency();
    bool streaming_experts() const { return streaming_experts_; }
};

} // namespace Laplace
