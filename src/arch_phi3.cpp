// arch_phi3.cpp - Microsoft Phi3 / Phi-3-mini transformer
//
// Forward pass mirrors Llama's with three differences:
//   1. Fused QKV: one matmul `attn_qkv.weight` of shape
//      [hidden, (n_q + 2*n_kv) * head_dim].  We split the output
//      into Q, K, V slices in place.
//   2. Fused gate+up FFN: one matmul `ffn_gate_up.weight` of shape
//      [hidden, 2*intermediate].  Split into gate, up.
//   3. Phi3 RMSNorm (gain+1): ops::rmsnorm_phi3.
//
// Sliding-window attention is a no-op in this pass: when the
// context is within the window size, the full causal mask is
// equivalent.  Wiring the real sliding window is the next step
// if/when a Phi3-mini-instruct-4k-class model is tested.
#include "arch_phi3.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "matmul.h"
#include "model.h"
#include "ops.h"
#include "threadpool.h"
#include "trace.h"

namespace Laplace {

bool Phi3Arch::load_config(const GGUFContext& gguf, ModelConfig* cfg) {
    const auto& m = gguf.metadata();
    auto arch = meta_str(m, "general.architecture");
    if (!arch || *arch != "phi3") return false;
    const std::string A = *arch + ".";

    cfg->n_layers     = static_cast<int>(meta_int(m, (A + "block_count").c_str()));
    cfg->hidden       = static_cast<int>(meta_int(m, (A + "embedding_length").c_str()));
    cfg->intermediate = static_cast<int>(meta_int(m, (A + "feed_forward_length").c_str()));
    cfg->n_q_heads    = static_cast<int>(meta_int(m, (A + "attention.head_count").c_str()));
    cfg->n_kv_heads   = static_cast<int>(meta_int(m, (A + "attention.head_count_kv").c_str()));
    cfg->head_dim     = static_cast<int>(meta_int(
        m, (A + "attention.key_length").c_str(),
        cfg->n_q_heads > 0 ? cfg->hidden / cfg->n_q_heads : 0));
    cfg->max_seq_len  = static_cast<int>(meta_int(m, (A + "context_length").c_str(), 2048));
    cfg->rms_eps      = static_cast<float>(meta_float(m, (A + "attention.layer_norm_rms_epsilon").c_str(), 1e-5f));
    cfg->rope_freq_base = static_cast<float>(meta_float(m, (A + "rope.freq_base").c_str(), 10000.0f));
    cfg->rope_dim_count = static_cast<int>(meta_int(m, (A + "rope.dimension_count").c_str(), cfg->head_dim));
    if (auto* p = meta_as<MetaArrayI32>(m, (A + "rope.dimension_sections").c_str())) {
        for (int i = 0; i < 4 && i < (int)p->size(); i++) cfg->rope_sections[i] = (*p)[i];
    }
    cfg->tied_lm_head = false;  // Phi3 ships an explicit output weight.

    // Phi3's SSM group dims are unused but keep the struct sane.
    cfg->ssm_group_count = 0;
    cfg->ssm_state_size  = 0;
    cfg->ssm_inner_size  = 0;
    cfg->ssm_conv_kernel = 0;
    cfg->ssm_time_step_rank = 0;
    return true;
}

bool Phi3Arch::load_weights(const GGUFContext& gguf, const ModelConfig& cfg,
                             std::vector<LayerWeights>* layers,
                             std::vector<bool>* is_attention,
                             std::vector<int>* kv_layer_idx) {
    layers->assign(cfg.n_layers, {});
    is_attention->assign(cfg.n_layers, true);
    kv_layer_idx->assign(cfg.n_layers, -1);
    for (int i = 0; i < cfg.n_layers; i++) {
        const std::string p = "blk." + std::to_string(i) + ".";
        LayerWeights& L = (*layers)[i];
        L.attn_norm      = gguf.find_tensor((p + "attn_norm.weight").c_str());
        L.post_attn_norm = gguf.find_tensor((p + "ffn_norm.weight").c_str());
        L.attn_qkv       = gguf.find_tensor((p + "attn_qkv.weight").c_str());
        L.attn_output    = gguf.find_tensor((p + "attn_output.weight").c_str());
        L.ffn_gate       = gguf.find_tensor((p + "ffn_gate_up.weight").c_str());
        if (!L.ffn_gate) {
            L.ffn_gate = gguf.find_tensor((p + "ffn_up.weight").c_str());
        }
        L.ffn_down       = gguf.find_tensor((p + "ffn_down.weight").c_str());
        // Phi3 fuses gate+up into ffn_gate; use it directly as the
        // "gate" slot, and split in forward_layer.
        // Optional QKV bias (Phi3-mini emits it; older variants don't).
        L.attn_q_bias    = gguf.find_tensor((p + "attn_qkv.bias").c_str());
        if (!L.attn_norm || !L.post_attn_norm || !L.attn_qkv
            || !L.attn_output || !L.ffn_gate || !L.ffn_down) {
            fprintf(stderr, "phi3: missing required tensor in layer %d\n", i);
            return false;
        }
        (*kv_layer_idx)[i] = i;
    }
    return true;
}

void Phi3Arch::reserve(const ModelConfig& cfg, int max_seq, int max_batch,
                       ModelBuffers* buf) {
    const size_t M = static_cast<size_t>(max_batch);
    buf->x.assign(M * cfg.hidden, 0.0f);
    buf->xb.assign(M * cfg.hidden, 0.0f);
    buf->x_norm.assign(M * cfg.hidden, 0.0f);

    // Fused QKV: 1 * (n_q + 2 * n_kv) * head_dim
    const int qkv_dim = (cfg.n_q_heads + 2 * cfg.n_kv_heads) * cfg.head_dim;
    buf->qkv.assign(M * qkv_dim, 0.0f);
    buf->attn_out.assign(M * cfg.n_q_heads * cfg.head_dim, 0.0f);
    buf->attn_logits.assign(static_cast<size_t>(cfg.n_q_heads) * max_seq, 0.0f);

    // Fused gate+up: 2 * intermediate.  The fused matmul writes
    // [all_gates | all_ups] consecutively; we split in forward.
    buf->ffn_gate.assign(M * 2 * cfg.intermediate, 0.0f);
    buf->ffn_up.assign(M * cfg.intermediate, 0.0f);
    buf->ffn_hidden.assign(M * cfg.intermediate, 0.0f);

    // Phi3 has no SSM state; these stay empty.
    buf->ssm_conv_state.clear();
    buf->ssm_recurrent.clear();
}

void Phi3Arch::forward_layer(int layer, const LayerWeights& W, const ModelConfig& cfg,
                              int M, int pos0, KVCache& kv, ModelBuffers* buf,
                              float* /*checkpoints*/) {
    const int H    = cfg.hidden;
    const int Hq   = cfg.n_q_heads;
    const int Hk   = cfg.n_kv_heads;
    const int Dh   = cfg.head_dim;
    const int qkv_dim = (Hq + 2 * Hk) * Dh;
    const int qn   = Hq * Dh;
    const int kn   = Hk * Dh;
    const int gqa  = Hq / Hk;

    // 1. Pre-attention norm (Phi3 gain+1).
    for (int m = 0; m < M; m++) {
        ops::rmsnorm_phi3(buf->x.data() + static_cast<size_t>(m) * H,
                          reinterpret_cast<const float*>(W.attn_norm->data),
                          buf->x_norm.data() + static_cast<size_t>(m) * H,
                          H, cfg.rms_eps);
    }

    // 2. Fused QKV matmul + split.  Output layout in GGUF is
    //    [all Q rows | all K rows | all V rows] (head-major within
    //    each section); the matmul writes it contiguously.
    matmul_rows(buf->x_norm.data(), *W.attn_qkv, buf->qkv.data(), M, H, qkv_dim);

    // Optional QKV bias.  We add it in place if present.  Tiled
    // view: qkv[m * qkv_dim + h_q * Dh + d] for query head h_q,
    // qkv[m * qkv_dim + (Hq + h_k) * Dh + d] for k head, etc.
    if (W.attn_q_bias) {
        const float* bias = reinterpret_cast<const float*>(W.attn_q_bias->data);
        for (int m = 0; m < M; m++) {
            float* qkv_m = buf->qkv.data() + static_cast<size_t>(m) * qkv_dim;
            for (int h = 0; h < Hq + 2 * Hk; h++) {
                for (int d = 0; d < Dh; d++) {
                    qkv_m[h * Dh + d] += bias[h * Dh + d];
                }
            }
        }
    }

    // 3. Per-head QK-RMSNorm (Phi3-mini has it; older Phi3 doesn't).
    //    For this first pass we skip it; the model still works for
    //    the non-QK-norm Phi3 variants.  TODO when a QK-norm
    //    variant is downloaded.
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
    const int max_seq = static_cast<int>(buf->attn_logits.size()) / cfg.n_q_heads;
    const bool laplace_fast = kv.mode() == KVCacheMode::LAPLACE;
    const bool laplace_rotated = laplace_fast && kv.laplace_rotated();
    const bool fp32_fast = kv.mode() == KVCacheMode::FP32;

    for (int m = 0; m < M; m++) {
        const int pos = pos0 + m;
        float* qp = buf->qkv.data() + static_cast<size_t>(m) * qkv_dim;
        float* kp = qp + qn;
        float* vp = kp + kn;

        // 4. Partial RoPE on Q and K (the first rope_pairs dims).
        if (buf->rope_pairs > 0) {
            ops::rope_apply(qp, Hq, Dh, buf->rope_pairs,
                            buf->rope_cos.data() + static_cast<size_t>(pos) * buf->rope_pairs,
                            buf->rope_sin.data() + static_cast<size_t>(pos) * buf->rope_pairs);
            ops::rope_apply(kp, Hk, Dh, buf->rope_pairs,
                            buf->rope_cos.data() + static_cast<size_t>(pos) * buf->rope_pairs,
                            buf->rope_sin.data() + static_cast<size_t>(pos) * buf->rope_pairs);
        }
        if (M == 1) {
            trace("attn_q_roped", layer, qp, qn);
            trace("attn_k_roped", layer, kp, kn);
            trace("attn_v", layer, vp, kn);
        }

        // 5. Write K, V to the cache; causal attention over [0, pos].
        for (int h = 0; h < Hk; h++) {
            kv.store_k(layer, h, pos, kp + h * Dh);
            kv.store_v(layer, h, pos, vp + h * Dh);
        }

        if (laplace_rotated) {
            for (int h = 0; h < Hq; h++) walsh_hadamard(qp + h * Dh, Dh);
        }

        const int end = pos + 1;
        float* ao_m = buf->attn_out.data() + static_cast<size_t>(m) * Hq * Dh;
        for (int d = 0; d < Hq * Dh; d++) ao_m[d] = 0.0f;

        if (laplace_fast) {
            std::vector<int> ends(Hq, end);
            std::vector<const float*> queries(Hq);
            std::vector<float*> outputs(Hq);
            for (int h = 0; h < Hq; h++) {
                queries[h] = qp + h * Dh;
                outputs[h] = ao_m + h * Dh;
            }
            ThreadPool::get().parallel_for(Hk, [&](int kvh) {
                const int h0 = kvh * gqa;
                kv.attention_batch_all_wh(
                    layer, kvh, gqa, ends.data() + h0,
                    queries.data() + h0, scale, outputs.data() + h0);
                if (laplace_rotated) {
                    for (int hi = 0; hi < gqa; hi++) {
                        inverse_walsh_hadamard(ao_m + (h0 + hi) * Dh, Dh);
                    }
                }
            });
        } else {
            for (int kvh = 0; kvh < Hk; kvh++) {
                const int h0 = kvh * gqa;
                const float* Kbase = fp32_fast ? kv.head_k(layer, kvh) : nullptr;
                const float* Vbase = fp32_fast ? kv.head_v(layer, kvh) : nullptr;
                const uint16_t* Kbase16 = fp32_fast ? nullptr : kv.head_k16(layer, kvh);
                const uint16_t* Vbase16 = fp32_fast ? nullptr : kv.head_v16(layer, kvh);
                for (int t = 0; t < end; t++) {
                    for (int hi = 0; hi < gqa; hi++) {
                        int h = h0 + hi;
                        const float* qh = qp + h * Dh;
                        float* la = buf->attn_logits.data() + static_cast<size_t>(h) * max_seq;
                        if (fp32_fast) {
                            la[t] = ops::dot(qh, Kbase + static_cast<size_t>(t) * Dh, Dh) * scale;
                        } else {
                            la[t] = ops::dot_f16(qh, Kbase16 + static_cast<size_t>(t) * Dh, Dh) * scale;
                        }
                    }
                }
                for (int hi = 0; hi < gqa; hi++) {
                    int h = h0 + hi;
                    float* la = buf->attn_logits.data() + static_cast<size_t>(h) * max_seq;
                    float maxv = la[0];
                    for (int t = 1; t < end; t++) if (la[t] > maxv) maxv = la[t];
                    float sumv = 0.0f;
                    for (int t = 0; t < end; t++) {
                        la[t] = std::exp(la[t] - maxv);
                        sumv += la[t];
                    }
                    for (int t = 0; t < end; t++) la[t] /= sumv;
                }
                for (int t = 0; t < end; t++) {
                    for (int hi = 0; hi < gqa; hi++) {
                        int h = h0 + hi;
                        float w = buf->attn_logits[static_cast<size_t>(h) * max_seq + t];
                        if (w == 0.0f) continue;
                        if (fp32_fast) {
                            ops::axpy(ao_m + h * Dh, w, Vbase + static_cast<size_t>(t) * Dh, Dh);
                        } else {
                            ops::axpy_f16(ao_m + h * Dh, w, Vbase16 + static_cast<size_t>(t) * Dh, Dh);
                        }
                    }
                }
            }
        }
        if (M == 1) trace("attn_gated", layer, ao_m, Hq * Dh);
    }

    // 6. Output projection.
    matmul_rows(buf->attn_out.data(), *W.attn_output, buf->xb.data(),
                M, Hq * Dh, H);
    if (M == 1) trace("attn_residual", layer, buf->xb.data(), H);
    for (size_t j = 0; j < static_cast<size_t>(M) * H; j++) buf->x[j] += buf->xb[j];

    // 7. Pre-FFN norm (Phi3 gain+1).
    for (int m = 0; m < M; m++) {
        ops::rmsnorm_phi3(buf->x.data() + static_cast<size_t>(m) * H,
                          reinterpret_cast<const float*>(W.post_attn_norm->data),
                          buf->x_norm.data() + static_cast<size_t>(m) * H,
                          H, cfg.rms_eps);
    }

    // 8. Fused gate+up matmul.  Output is [all_gates | all_ups],
    //    both of length intermediate.  We split into ffn_gate and
    //    ffn_up via a memmove on the second half.  Equivalently
    //    we could do two matmuls; the fused matmul is faster
    //    (weights stream once) so we keep it and pay the copy.
    matmul_rows(buf->x_norm.data(), *W.ffn_gate, buf->ffn_gate.data(),
                M, H, 2 * cfg.intermediate);
    const size_t up_off = static_cast<size_t>(cfg.intermediate) * sizeof(float);
    for (int m = 0; m < M; m++) {
        std::memcpy(buf->ffn_up.data() + static_cast<size_t>(m) * cfg.intermediate,
                    buf->ffn_gate.data() + static_cast<size_t>(m) * 2 * cfg.intermediate + cfg.intermediate,
                    up_off);
    }
    ops::swiglu(buf->ffn_gate.data(), buf->ffn_up.data(), buf->ffn_hidden.data(),
                M * cfg.intermediate);

    // 9. FFN down projection + residual.
    matmul_rows(buf->ffn_hidden.data(), *W.ffn_down, buf->xb.data(),
                M, cfg.intermediate, H);
    if (M == 1) trace("ffn_out", layer, buf->xb.data(), H);
    for (size_t j = 0; j < static_cast<size_t>(M) * H; j++) buf->x[j] += buf->xb[j];
}

} // namespace Laplace
