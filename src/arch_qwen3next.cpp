// arch_qwen3next.cpp - Qwen3-Next / Qwen3.5 hybrid transformer
#include "arch_qwen3next.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <chrono>
#include <cstdlib>

#include "matmul.h"
#include "model.h"
#include "ops.h"
#include "ssm.h"
#include "threadpool.h"
#include "trace.h"

namespace {
// LAPLACE_PROF=1: split out the non-matmul hot spots for the Qwen3-Next path.
struct PhaseProf {
    bool on = std::getenv("LAPLACE_PROF") != nullptr;
    double dnet_token = 0.0, ckpt = 0.0, attn_core = 0.0, act = 0.0, norms = 0.0;
    ~PhaseProf() {
        if (!on) return;
        fprintf(stderr, "PROF dnet_token: %.3f s, checkpoints: %.3f s, attn_core: %.3f s, swiglu: %.3f s, norms+resid: %.3f s\n",
                dnet_token, ckpt, attn_core, act, norms);
    }
};
PhaseProf g_pprof;
struct PhaseTimer {
    double* acc;
    std::chrono::steady_clock::time_point t0;
    explicit PhaseTimer(double* a) : acc(a) {
        if (g_pprof.on) t0 = std::chrono::steady_clock::now();
    }
    ~PhaseTimer() {
        if (g_pprof.on) *acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};
}

namespace Laplace {

bool Qwen3NextArch::load_config(const GGUFContext& gguf, ModelConfig* cfg) {
    const auto& m = gguf.metadata();
    auto arch = meta_str(m, "general.architecture");
    if (!arch || (*arch != "qwen3next" && *arch != "qwen35")) return false;
    const std::string A = *arch + ".";

    cfg->n_layers     = static_cast<int>(meta_int(m, (A + "block_count").c_str()));
    cfg->hidden       = static_cast<int>(meta_int(m, (A + "embedding_length").c_str()));
    cfg->intermediate = static_cast<int>(meta_int(m, (A + "feed_forward_length").c_str()));
    cfg->n_q_heads    = static_cast<int>(meta_int(m, (A + "attention.head_count").c_str()));
    cfg->n_kv_heads   = static_cast<int>(meta_int(m, (A + "attention.head_count_kv").c_str()));
    cfg->head_dim     = static_cast<int>(meta_int(m, (A + "attention.key_length").c_str()));
    cfg->max_seq_len   = static_cast<int>(meta_int(m, (A + "context_length").c_str(), 2048));
    cfg->rms_eps       = static_cast<float>(meta_float(m, (A + "attention.layer_norm_rms_epsilon").c_str(), 1e-6));
    cfg->rope_freq_base = static_cast<float>(meta_float(m, (A + "rope.freq_base").c_str(), 1e7));
    cfg->rope_dim_count = static_cast<int>(meta_int(m, (A + "rope.dimension_count").c_str(), 64));
    if (auto* p = meta_as<MetaArrayI32>(m, (A + "rope.dimension_sections").c_str())) {
        for (int i = 0; i < 4 && i < (int)p->size(); i++) cfg->rope_sections[i] = (*p)[i];
    }

    cfg->ssm_group_count   = static_cast<int>(meta_int(m, (A + "ssm.group_count").c_str(), 16));
    cfg->ssm_state_size    = static_cast<int>(meta_int(m, (A + "ssm.state_size").c_str(), 128));
    cfg->ssm_inner_size    = static_cast<int>(meta_int(m, (A + "ssm.inner_size").c_str(), 2048));
    cfg->ssm_conv_kernel   = static_cast<int>(meta_int(m, (A + "ssm.conv_kernel").c_str(), 4));
    cfg->ssm_time_step_rank = static_cast<int>(meta_int(m, (A + "ssm.time_step_rank").c_str(), 16));
    return true;
}

bool Qwen3NextArch::load_weights(const GGUFContext& gguf, const ModelConfig& cfg,
                                 std::vector<LayerWeights>* layers,
                                 std::vector<bool>* is_attention,
                                 std::vector<int>* kv_layer_idx) {
    layers->assign(cfg.n_layers, {});
    is_attention->assign(cfg.n_layers, false);
    kv_layer_idx->assign(cfg.n_layers, -1);
    kv_layer_idx_.assign(cfg.n_layers, -1);
    int n_attn_layers = 0;
    for (int i = 0; i < cfg.n_layers; i++) {
        const std::string p = "blk." + std::to_string(i) + ".";
        LayerWeights& L = (*layers)[i];

        L.attn_norm      = gguf.find_tensor((p + "attn_norm.weight").c_str());
        L.post_attn_norm = gguf.find_tensor((p + "post_attention_norm.weight").c_str());
        L.ffn_norm       = gguf.find_tensor((p + "ffn_norm.weight").c_str());
        L.ffn_gate       = gguf.find_tensor((p + "ffn_gate.weight").c_str());
        L.ffn_up         = gguf.find_tensor((p + "ffn_up.weight").c_str());
        L.ffn_down       = gguf.find_tensor((p + "ffn_down.weight").c_str());
        L.attn_qkv       = gguf.find_tensor((p + "attn_qkv.weight").c_str());

        // SSM / DeltaNet tensors
        L.attn_gate    = gguf.find_tensor((p + "attn_gate.weight").c_str());
        L.ssm_a        = gguf.find_tensor((p + "ssm_a").c_str());
        L.ssm_conv1d   = gguf.find_tensor((p + "ssm_conv1d.weight").c_str());
        L.ssm_dt_bias  = gguf.find_tensor((p + "ssm_dt.bias").c_str());
        L.ssm_alpha    = gguf.find_tensor((p + "ssm_alpha.weight").c_str());
        L.ssm_beta     = gguf.find_tensor((p + "ssm_beta.weight").c_str());
        L.ssm_norm     = gguf.find_tensor((p + "ssm_norm.weight").c_str());
        L.ssm_out      = gguf.find_tensor((p + "ssm_out.weight").c_str());

        // Full-attention tensors (optional — some models have no attn layers)
        L.attn_q      = gguf.find_tensor((p + "attn_q.weight").c_str());
        L.attn_k      = gguf.find_tensor((p + "attn_k.weight").c_str());
        L.attn_v      = gguf.find_tensor((p + "attn_v.weight").c_str());
        L.attn_q_norm = gguf.find_tensor((p + "attn_q_norm.weight").c_str());
        L.attn_k_norm = gguf.find_tensor((p + "attn_k_norm.weight").c_str());
        L.attn_output = gguf.find_tensor((p + "attn_output.weight").c_str());

        // The FFN norm is called "post_attention_norm.weight" in this model,
        // not "ffn_norm.weight".  Use whichever is available.
        if (!L.ffn_norm) L.ffn_norm = L.post_attn_norm;

        if (L.attn_q && L.attn_k && L.attn_v &&
            L.attn_q_norm && L.attn_k_norm && L.attn_output) {
            (*is_attention)[i] = true;
            (*kv_layer_idx)[i] = n_attn_layers;
            kv_layer_idx_[i] = n_attn_layers++;
            const uint64_t qd  = static_cast<uint64_t>(cfg.n_q_heads) * cfg.head_dim;
            const uint64_t kvd = static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
            if ((L.attn_q->dims[1] != qd && L.attn_q->dims[1] != 2 * qd) ||
                L.attn_k->dims[1] != kvd || L.attn_v->dims[1] != kvd ||
                L.attn_output->dims[0] != qd) {
                fprintf(stderr, "qwen3next: layer %d attention shapes inconsistent "
                        "(q=%llu k=%llu v=%llu o_in=%llu, expected q=%llu or %llu, kv=%llu, o_in=%llu)\n",
                        i,
                        (unsigned long long)L.attn_q->dims[1],
                        (unsigned long long)L.attn_k->dims[1],
                        (unsigned long long)L.attn_v->dims[1],
                        (unsigned long long)L.attn_output->dims[0],
                        (unsigned long long)qd, (unsigned long long)(2 * qd),
                        (unsigned long long)kvd, (unsigned long long)qd);
                return false;
            }
        } else if (L.ssm_a && L.ssm_conv1d && L.ssm_alpha && L.ssm_beta && L.ssm_out) {
            (*is_attention)[i] = false;
        } else {
            fprintf(stderr, "qwen3next: layer %d has neither full-attn nor SSM weights\n", i);
            return false;
        }
    }
    return true;
}

void Qwen3NextArch::reserve(const ModelConfig& cfg, int max_seq, int max_batch,
                            ModelBuffers* buf) {
    const size_t M = static_cast<size_t>(max_batch);
    buf->x.assign(M * cfg.hidden, 0.0f);
    buf->xb.assign(M * cfg.hidden, 0.0f);
    buf->x_norm.assign(M * cfg.hidden, 0.0f);

    buf->qkv.assign(M * (2 * cfg.n_q_heads + 2 * cfg.n_kv_heads) * cfg.head_dim, 0.0f);
    buf->attn_out.assign(M * cfg.n_q_heads * cfg.head_dim, 0.0f);
    buf->attn_logits.assign(static_cast<size_t>(cfg.n_q_heads) * max_seq, 0.0f);

    buf->ffn_gate.assign(M * cfg.intermediate, 0.0f);
    buf->ffn_up.assign(M * cfg.intermediate, 0.0f);
    buf->ffn_hidden.assign(M * cfg.intermediate, 0.0f);

    // DeltaNet
    buf->dnet_qkv.assign(M * 3 * cfg.ssm_inner_size, 0.0f);
    buf->dnet_gate.assign(M * cfg.ssm_inner_size, 0.0f);
    buf->dnet_b_proj.assign(M * cfg.ssm_group_count, 0.0f);
    buf->dnet_a_proj.assign(M * cfg.ssm_group_count, 0.0f);
    buf->dnet_o.assign(static_cast<size_t>(cfg.ssm_group_count) * cfg.ssm_state_size, 0.0f);
    buf->dnet_normed.assign(M * cfg.ssm_inner_size, 0.0f);

    buf->ssm_conv_state.assign(static_cast<size_t>(cfg.n_layers) * 3 * cfg.ssm_inner_size * (cfg.ssm_conv_kernel - 1), 0.0f);
    buf->ssm_recurrent.assign(static_cast<size_t>(cfg.n_layers) * cfg.ssm_group_count * cfg.ssm_state_size * cfg.ssm_state_size, 0.0f);
}

void Qwen3NextArch::attention_batch(int layer, int M, int pos0, KVCache& kv,
                                    const LayerWeights& W, const ModelConfig& cfg,
                                    ModelBuffers* buf) {
    const int kvl = kv_layer_idx_[layer];
    const int Hq = cfg.n_q_heads;
    const int Hk = cfg.n_kv_heads;
    const int Dh = cfg.head_dim;
    const int H  = cfg.hidden;

    for (int m = 0; m < M; m++) {
        ops::rmsnorm(buf->x.data() + static_cast<size_t>(m) * H,
                     reinterpret_cast<const float*>(W.attn_norm->data),
                     buf->x_norm.data() + static_cast<size_t>(m) * H,
                     H, cfg.rms_eps);
    }

    // Projections, batched (weights stream once for all M tokens).
    // Qwen3-Next gated attention: attn_q outputs, per head,
    // [query (Dh) | output gate (Dh)], i.e. 2*Hq*Dh total.
    const bool gated = W.attn_q->dims[1] == static_cast<uint64_t>(2 * Hq) * Dh;
    const int q_stride = gated ? 2 * Dh : Dh;
    const int qn = Hq * q_stride;
    const int kn = Hk * Dh;
    float* qp_all = buf->qkv.data();                       // [M, qn]
    float* kp_all = qp_all + static_cast<size_t>(M) * qn;  // [M, kn]
    float* vp_all = kp_all + static_cast<size_t>(M) * kn;  // [M, kn]
    matmul_rows(buf->x_norm.data(), *W.attn_q, qp_all, M, H, qn);
    matmul_rows(buf->x_norm.data(), *W.attn_k, kp_all, M, H, kn);
    matmul_rows(buf->x_norm.data(), *W.attn_v, vp_all, M, H, kn);

    const float* qnw = reinterpret_cast<const float*>(W.attn_q_norm->data);
    const float* knw = reinterpret_cast<const float*>(W.attn_k_norm->data);
    const int gqa = Hq / Hk;
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
    const int max_seq = static_cast<int>(buf->attn_logits.size()) / cfg.n_q_heads;

    PhaseTimer pt_attn(&g_pprof.attn_core);
    for (int m = 0; m < M; m++) {
        const int pos = pos0 + m;
        float* qp = qp_all + static_cast<size_t>(m) * qn;
        float* kp = kp_all + static_cast<size_t>(m) * kn;
        float* vp = vp_all + static_cast<size_t>(m) * kn;

        // Per-head QK-RMSNorm + partial RoPE (Q rows are strided when gated).
        for (int h = 0; h < Hq; h++) {
            float* q = qp + h * q_stride;
            ops::rmsnorm_rows(q, qnw, q, 1, Dh, cfg.rms_eps);
            ops::rope_apply(q, 1, Dh, buf->rope_pairs,
                            buf->rope_cos.data() + static_cast<size_t>(pos) * buf->rope_pairs,
                            buf->rope_sin.data() + static_cast<size_t>(pos) * buf->rope_pairs);
        }
        ops::rmsnorm_rows(kp, knw, kp, Hk, Dh, cfg.rms_eps);
        ops::rope_apply(kp, Hk, Dh, buf->rope_pairs,
                        buf->rope_cos.data() + static_cast<size_t>(pos) * buf->rope_pairs,
                        buf->rope_sin.data() + static_cast<size_t>(pos) * buf->rope_pairs);
        if (M == 1) {
            trace("attn_q_roped", layer, qp, qn);
            trace("attn_k_roped", layer, kp, kn);
            trace("attn_v", layer, vp, kn);
        }

        // Write K, V to the cache; token m may attend to every earlier batch
        // token because their slots were written in previous iterations.
        for (int h = 0; h < Hk; h++) {
            kv.store_k(kvl, h, pos, kp + h * Dh);
            kv.store_v(kvl, h, pos, vp + h * Dh);
        }

        const bool laplace_fast = kv.mode() == KVCacheMode::LAPLACE;
        const bool laplace_rotated = laplace_fast && kv.laplace_rotated();
        if (laplace_rotated) {
            for (int h = 0; h < Hq; h++) walsh_hadamard(qp + h * q_stride, Dh);
        }

        // Causal attention over [0, pos], GQA-amortized: load each K/V once
        // per (kvh, t) and reuse across all gqa query heads in the group.
        const int end = pos + 1;
        float* ao_m = buf->attn_out.data() + static_cast<size_t>(m) * Hq * Dh;
        for (int d = 0; d < Hq * Dh; d++) ao_m[d] = 0.0f;

        if (laplace_fast) {
            std::vector<int> ends(Hq, end);
            std::vector<const float*> queries(Hq);
            std::vector<float*> outputs(Hq);
            for (int h = 0; h < Hq; h++) {
                queries[h] = qp + h * q_stride;
                outputs[h] = ao_m + h * Dh;
            }
            ThreadPool::get().parallel_for(Hk, [&](int kvh) {
                int h0 = kvh * gqa;
                kv.attention_batch_all_wh(
                    kvl, kvh, gqa, ends.data() + h0,
                    queries.data() + h0, scale, outputs.data() + h0);
                for (int hi = 0; hi < gqa; hi++) {
                    int h = h0 + hi;
                    float* ao = ao_m + h * Dh;
                    if (laplace_rotated) inverse_walsh_hadamard(ao, Dh);
                    if (gated) {
                        const float* g = qp + h * q_stride + Dh;
                        for (int d = 0; d < Dh; d++) {
                            ao[d] *= 1.0f / (1.0f + std::exp(-g[d]));
                        }
                    }
                }
            });
            if (M == 1) trace("attn_gated", layer, ao_m, Hq * Dh);
            continue;
        }

        const bool fp32_fast = (kv.mode() == KVCacheMode::FP32);

        for (int kvh = 0; kvh < Hk; kvh++) {
            const int h0 = kvh * gqa;
            const float* Kbase = fp32_fast ? kv.head_k(kvl, kvh) : nullptr;
            const float* Vbase = fp32_fast ? kv.head_v(kvl, kvh) : nullptr;
            const uint16_t* Kbase16 = fp32_fast ? nullptr : kv.head_k16(kvl, kvh);
            const uint16_t* Vbase16 = fp32_fast ? nullptr : kv.head_v16(kvl, kvh);
            for (int t = 0; t < end; t++) {
                for (int hi = 0; hi < gqa; hi++) {
                    int h = h0 + hi;
                    const float* qh = qp + h * q_stride;
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
                for (int t = 0; t < end; t++) { la[t] = std::exp(la[t] - maxv); sumv += la[t]; }
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

        // Per-head output gate (Qwen3-Next): ao *= sigmoid(gate).
        if (gated) {
            for (int h = 0; h < Hq; h++) {
                const float* g = qp + h * q_stride + Dh;
                float* ao = ao_m + h * Dh;
                for (int d = 0; d < Dh; d++) ao[d] *= 1.0f / (1.0f + std::exp(-g[d]));
            }
        }
        if (M == 1) trace("attn_gated", layer, ao_m, Hq * Dh);
    }

    // Output projection: [M, Hq*Dh] -> [M, hidden], batched.
    matmul_rows(buf->attn_out.data(), *W.attn_output, buf->xb.data(),
                M, Hq * Dh, static_cast<int>(W.attn_output->dims[1]));
}

void Qwen3NextArch::deltanet_batch(int layer, int M, const LayerWeights& W,
                                   const ModelConfig& cfg, ModelBuffers* buf,
                                   float* checkpoints) {
    const int G = cfg.ssm_group_count;
    const int D = cfg.ssm_state_size;
    const int inner = cfg.ssm_inner_size;
    const int conv_dim = 3 * inner;
    const int hist = cfg.ssm_conv_kernel - 1;
    const int H = cfg.hidden;

    for (int m = 0; m < M; m++) {
        ops::rmsnorm(buf->x.data() + static_cast<size_t>(m) * H,
                     reinterpret_cast<const float*>(W.attn_norm->data),
                     buf->x_norm.data() + static_cast<size_t>(m) * H,
                     H, cfg.rms_eps);
    }

    // All four input projections, batched across tokens.
    matmul_rows(buf->x_norm.data(), *W.attn_qkv,  buf->dnet_qkv.data(),    M, H, conv_dim);
    matmul_rows(buf->x_norm.data(), *W.attn_gate, buf->dnet_gate.data(),   M, H, inner);
    matmul_rows(buf->x_norm.data(), *W.ssm_beta,  buf->dnet_b_proj.data(), M, H, G);
    matmul_rows(buf->x_norm.data(), *W.ssm_alpha, buf->dnet_a_proj.data(), M, H, G);

    DeltaNetParams p;
    p.G = G;
    p.D = D;
    p.inner = inner;
    p.conv_kernel = cfg.ssm_conv_kernel;
    p.conv_dim = conv_dim;
    p.A        = reinterpret_cast<const float*>(W.ssm_a->data);
    p.dt_bias  = reinterpret_cast<const float*>(W.ssm_dt_bias->data);
    p.ssm_norm = reinterpret_cast<const float*>(W.ssm_norm->data);
    p.conv_w   = reinterpret_cast<const float*>(W.ssm_conv1d->data);

    const size_t conv_off = static_cast<size_t>(layer) * conv_dim * hist;
    const size_t rec_off  = static_cast<size_t>(layer) * G * D * D;
    const size_t conv_n   = static_cast<size_t>(conv_dim) * hist;
    const size_t rec_n    = static_cast<size_t>(G) * D * D;

    // Recurrent core runs per token (state is sequential); after each token,
    // optionally checkpoint this layer's state slice for speculative rollback.
    // Hoist the env-var check: the state buffer is shared between the real-
    // space and WH-domain paths, so a per-token flip would silently corrupt
    // the state basis.  Set once at process start.
    static const bool kUseWhState =
        std::getenv("LAPLACE_DELTA_STATE_WH") != nullptr;
    for (int m = 0; m < M; m++) {
        {
            PhaseTimer pt(&g_pprof.dnet_token);
            if (kUseWhState) {
                deltanet_token_wh(
                    p,
                    buf->ssm_conv_state.data() + conv_off,
                    buf->ssm_recurrent.data()  + rec_off,
                    buf->dnet_qkv.data()    + static_cast<size_t>(m) * conv_dim,
                    buf->dnet_gate.data()   + static_cast<size_t>(m) * inner,
                    buf->dnet_b_proj.data() + static_cast<size_t>(m) * G,
                    buf->dnet_a_proj.data() + static_cast<size_t>(m) * G,
                    buf->dnet_o.data(),
                    buf->dnet_normed.data() + static_cast<size_t>(m) * inner);
            } else {
                deltanet_token(
                    p,
                    buf->ssm_conv_state.data() + conv_off,
                    buf->ssm_recurrent.data()  + rec_off,
                    buf->dnet_qkv.data()    + static_cast<size_t>(m) * conv_dim,
                    buf->dnet_gate.data()   + static_cast<size_t>(m) * inner,
                    buf->dnet_b_proj.data() + static_cast<size_t>(m) * G,
                    buf->dnet_a_proj.data() + static_cast<size_t>(m) * G,
                    buf->dnet_o.data(),
                    buf->dnet_normed.data() + static_cast<size_t>(m) * inner);
            }
        }
        if (checkpoints) {
            PhaseTimer pt(&g_pprof.ckpt);
            float* cp = checkpoints + static_cast<size_t>(m) * (buf->ssm_conv_state.size() + buf->ssm_recurrent.size());
            std::memcpy(cp + conv_off, buf->ssm_conv_state.data() + conv_off,
                        conv_n * sizeof(float));
            std::memcpy(cp + buf->ssm_conv_state.size() + rec_off,
                        buf->ssm_recurrent.data() + rec_off, rec_n * sizeof(float));
        }
    }

    // Output projection [M, inner] -> [M, hidden], batched.
    matmul_rows(buf->dnet_normed.data(), *W.ssm_out, buf->xb.data(), M, inner, H);
}

void Qwen3NextArch::forward_layer(int layer, const LayerWeights& W, const ModelConfig& cfg,
                                  int M, int pos0, KVCache& kv, ModelBuffers* buf,
                                  float* checkpoints) {
    const int H = cfg.hidden;

    // Mixer sub-block (full attention or DeltaNet).
    if (W.attn_q) {
        attention_batch(layer, M, pos0, kv, W, cfg, buf);
    } else {
        deltanet_batch(layer, M, W, cfg, buf, checkpoints);
    }
    if (M == 1) trace("mixer_out", layer, buf->xb.data(), H);
    for (size_t j = 0; j < static_cast<size_t>(M) * H; j++) buf->x[j] += buf->xb[j];
    if (M == 1) trace("attn_residual", layer, buf->x.data(), H);

    // FFN sub-block, projections batched.
    {
        PhaseTimer pt(&g_pprof.norms);
        for (int m = 0; m < M; m++) {
            ops::rmsnorm(buf->x.data() + static_cast<size_t>(m) * H,
                         reinterpret_cast<const float*>(W.ffn_norm->data),
                         buf->x_norm.data() + static_cast<size_t>(m) * H,
                         H, cfg.rms_eps);
        }
    }
    matmul_rows(buf->x_norm.data(), *W.ffn_gate, buf->ffn_gate.data(), M, H, cfg.intermediate);
    matmul_rows(buf->x_norm.data(), *W.ffn_up,   buf->ffn_up.data(),   M, H, cfg.intermediate);
    {
        PhaseTimer pt(&g_pprof.act);
        ops::swiglu(buf->ffn_gate.data(), buf->ffn_up.data(), buf->ffn_hidden.data(), M * cfg.intermediate);
    }
    matmul_rows(buf->ffn_hidden.data(), *W.ffn_down, buf->xb.data(), M, cfg.intermediate, H);
    if (M == 1) trace("ffn_out", layer, buf->xb.data(), H);
    for (size_t j = 0; j < static_cast<size_t>(M) * H; j++) buf->x[j] += buf->xb[j];
}

} // namespace Laplace
