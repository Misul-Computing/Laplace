// arch_adaptive.cpp - feature-driven GGUF execution
#include "arch_adaptive.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>

#include "matmul.h"
#include "model.h"
#include "ops.h"
#include "token_graph_backend.h"
#include "laplace_moe.h"
#include "threadpool.h"

namespace Laplace {

namespace {
// RMSNorm without scale weight: y = x / sqrt(mean(x^2) + eps)
void rmsnorm_no_scale(const float* x, float* y, int n, float eps) {
    float ms = 0.0f;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    float inv = 1.0f / std::sqrt(ms / n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * inv;
}

// LAPLACE_PROF=1: per-kernel timing for the Adaptive path. Prints averages
// every 100 decode tokens (a new token is detected when forward_layer is
// called with layer==0).
struct KernelProf {
    bool on = std::getenv("LAPLACE_PROF") != nullptr;
    enum K {
        k_attn_norm, k_qkv_proj, k_qk_norm, k_rope, k_kv_store, k_attn_core, k_o_proj,
        k_post_attn_norm, k_ffn_norm, k_ffn_gate_up, k_geglu_dense, k_ffn_down,
        k_post_ffw_norm_1, k_moe_router, k_moe_pre_norm, k_moe_prefetch_gu,
        k_moe_gate_up, k_moe_prefetch_dn, k_moe_gate_wait, k_moe_geglu, k_moe_down,
        k_moe_combine, k_post_ffw_norm_2, k_combine_dense_moe, k_post_ffw_norm,
        k_residual_add, k_layer_out_scale, NK
    };
    double acc[NK] = {};
    long   cnt[NK] = {};
    long   tokens = 0;
    static const char* names[NK];
    void maybe_print() {
        if (!on) return;
        if (++tokens % 30 != 0) return;
        fprintf(stderr, "\n=== PROF adaptive per-kernel (last 30 decode tokens) ===\n");
        double total = 0.0;
        for (int i = 0; i < NK; i++) {
            if (cnt[i] == 0) continue;
            double avg_ms = 1e3 * acc[i] / cnt[i];
            double per_tok_ms = 1e3 * acc[i] / 30.0;
            fprintf(stderr, "  %-20s %7.3f ms/call  %8.3f ms/tok  (%ld calls)\n",
                    names[i], avg_ms, per_tok_ms, cnt[i]);
            total += per_tok_ms;
        }
        fprintf(stderr, "  %-20s                       %8.3f ms/tok\n", "TOTAL", total);
        for (int i = 0; i < NK; i++) { acc[i] = 0.0; cnt[i] = 0; }
    }
};
KernelProf g_kprof;
const char* KernelProf::names[KernelProf::NK] = {
    "attn_norm", "qkv_proj", "qk_norm", "rope", "kv_store", "attn_core", "o_proj",
    "post_attn_norm", "ffn_norm", "ffn_gate_up", "geglu_dense", "ffn_down",
    "post_ffw_norm_1", "moe_router", "moe_pre_norm", "moe_prefetch_gu",
    "moe_gate_up", "moe_prefetch_dn", "moe_gate_wait", "moe_geglu", "moe_down",
    "moe_combine", "post_ffw_norm_2", "combine_dense_moe", "post_ffw_norm",
    "residual_add", "layer_out_scale"
};
struct KTimer {
    int k;
    std::chrono::steady_clock::time_point t0;
    bool on;
    explicit KTimer(int k_) : k(k_) {
        on = g_kprof.on;
        if (on) t0 = std::chrono::steady_clock::now();
    }
    ~KTimer() {
        if (on) {
            g_kprof.acc[k] += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            g_kprof.cnt[k]++;
        }
    }
};
} // namespace

bool AdaptiveArch::load_config(const GGUFContext& gguf, ModelConfig* cfg) {
    (void)gguf;
    cfg->n_layers = static_cast<int>(plan_.layers.size());
    cfg->hidden = plan_.hidden;
    cfg->intermediate = plan_.intermediate;
    cfg->max_seq_len = plan_.max_seq_len;
    cfg->rms_eps = plan_.rms_eps;
    cfg->n_experts = plan_.n_experts;
    cfg->n_experts_used = plan_.n_experts_used;
    cfg->expert_inter = plan_.expert_intermediate;
    cfg->logit_softcap = plan_.logit_softcap;
    cfg->embed_scale = plan_.embed_scale;

    cfg->n_q_heads = 0;
    cfg->n_kv_heads = 0;
    cfg->head_dim = 0;
    layer_types_.assign(cfg->n_layers, {});
    for (int i = 0; i < cfg->n_layers; ++i) {
        const LayerTopology& source = plan_.layers[i];
        LayerTypeInfo& output = layer_types_[i];
        output.is_global = source.sliding_window == 0;
        output.n_q_heads = source.n_q_heads;
        output.n_kv_heads = source.n_kv_heads;
        output.head_dim = source.head_dim;
        output.sliding_window = source.sliding_window;
        output.intermediate = source.intermediate > 0
            ? source.intermediate : plan_.intermediate;
        output.swiglu = source.swiglu;
        output.rope_dim = source.rope_dim;
        output.rope_base = source.rope_base;
        cfg->n_q_heads = std::max(cfg->n_q_heads, source.n_q_heads);
        cfg->n_kv_heads = std::max(cfg->n_kv_heads, source.n_kv_heads);
        cfg->head_dim = std::max(cfg->head_dim, source.head_dim);
        if (source.sliding_window == 0) {
            cfg->rope_freq_base = source.rope_base;
            cfg->rope_dim_count = source.rope_dim;
        }
    }
    return true;
}

bool AdaptiveArch::load_weights(const GGUFContext& gguf, const ModelConfig& cfg,
                              std::vector<LayerWeights>* layers,
                              std::vector<bool>* is_attention,
                              std::vector<int>* kv_layer_idx) {
    layers->assign(cfg.n_layers, {});
    is_attention->assign(cfg.n_layers, true);
    kv_layer_idx->assign(cfg.n_layers, -1);
    kv_layer_idx_.assign(cfg.n_layers, -1);
    int last_kv_owner = -1;

    // Load the shared rope_freqs tensor for global layers (p-RoPE).
    const Tensor* rope_freqs = gguf.find_tensor("rope_freqs.weight");
    if (rope_freqs && rope_freqs->n_dims == 1) {
        rope_freqs_full_.resize(rope_freqs->dims[0]);
        const float* src = reinterpret_cast<const float*>(rope_freqs->data);
        std::copy(src, src + rope_freqs->dims[0], rope_freqs_full_.data());
    }

    for (int i = 0; i < cfg.n_layers; i++) {
        const std::string p = "blk." + std::to_string(i) + ".";
        LayerWeights& L = (*layers)[i];

        L.attn_norm      = gguf.find_tensor((p + "attn_norm.weight").c_str());
        L.post_attn_norm = gguf.find_tensor((p + "post_attention_norm.weight").c_str());
        L.attn_q         = gguf.find_tensor((p + "attn_q.weight").c_str());
        L.attn_k         = gguf.find_tensor((p + "attn_k.weight").c_str());
        L.attn_v         = gguf.find_tensor((p + "attn_v.weight").c_str());
        L.attn_q_norm    = gguf.find_tensor((p + "attn_q_norm.weight").c_str());
        L.attn_k_norm    = gguf.find_tensor((p + "attn_k_norm.weight").c_str());
        L.attn_output    = gguf.find_tensor((p + "attn_output.weight").c_str());

        L.ffn_norm       = gguf.find_tensor((p + "ffn_norm.weight").c_str());
        L.ffn_gate       = gguf.find_tensor((p + "ffn_gate.weight").c_str());
        L.ffn_up         = gguf.find_tensor((p + "ffn_up.weight").c_str());
        L.ffn_down       = gguf.find_tensor((p + "ffn_down.weight").c_str());

        // MoE tensors
        L.moe_gate_inp       = gguf.find_tensor((p + "ffn_gate_inp.weight").c_str());
        L.moe_gate_inp_scale = gguf.find_tensor((p + "ffn_gate_inp.scale").c_str());
        L.moe_gate_up_exps   = gguf.find_tensor((p + "ffn_gate_up_exps.weight").c_str());
        L.moe_down_exps      = gguf.find_tensor((p + "ffn_down_exps.weight").c_str());
        L.moe_down_exps_scale= gguf.find_tensor((p + "ffn_down_exps.scale").c_str());
        L.pre_ffw_norm_2     = gguf.find_tensor((p + "pre_ffw_norm_2.weight").c_str());
        L.post_ffw_norm_1    = gguf.find_tensor((p + "post_ffw_norm_1.weight").c_str());
        L.post_ffw_norm_2    = gguf.find_tensor((p + "post_ffw_norm_2.weight").c_str());
        L.post_ffw_norm      = gguf.find_tensor((p + "post_ffw_norm.weight").c_str());
        L.layer_output_scale = gguf.find_tensor((p + "layer_output_scale.weight").c_str());

        if (L.attn_k) {
            last_kv_owner = i;
            (*kv_layer_idx)[i] = i;
            kv_layer_idx_[i] = i;
            layer_types_[i].owns_kv = true;
        } else if (last_kv_owner >= 0) {
            const LayerWeights& src = (*layers)[last_kv_owner];
            L.attn_k = src.attn_k;
            L.attn_v = src.attn_v;
            L.attn_k_norm = src.attn_k_norm;
            (*kv_layer_idx)[i] = (*kv_layer_idx)[last_kv_owner];
            kv_layer_idx_[i] = kv_layer_idx_[last_kv_owner];
            layer_types_[i].owns_kv = false;
        } else {
            fprintf(stderr, "adaptive: layer %d has no K and no earlier KV owner\n", i);
            return false;
        }

        if (!L.attn_norm || !L.attn_q || !L.attn_k || !L.attn_output ||
            !L.ffn_norm || !L.ffn_gate || !L.ffn_up || !L.ffn_down) {
            fprintf(stderr, "adaptive: layer %d missing core tensors\n", i);
            return false;
        }
        if (L.moe_gate_inp && !L.moe_gate_up_exps) {
            fprintf(stderr, "adaptive: layer %d has router but no expert weights\n", i);
            return false;
        }
    }
    return true;
}

void AdaptiveArch::reserve(const ModelConfig& cfg, int max_seq, int max_batch,
                         ModelBuffers* buf) {
    const size_t M = static_cast<size_t>(max_batch);
    const int H = cfg.hidden;
    const int Dh_max = cfg.head_dim;  // 512
    const int Hq = cfg.n_q_heads;     // 16
    const int Hk_max = cfg.n_kv_heads; // 8

    buf->x.assign(M * H, 0.0f);
    buf->xb.assign(M * H, 0.0f);
    buf->x_norm.assign(M * H, 0.0f);

    // Q/K/V: max sizes across layer types
    // Global: Q=16*512=8192, K=2*512=1024, V=K (no V proj)
    // Sliding: Q=16*256=4096, K=8*256=2048, V=8*256=2048
    // Use max: Q=8192, K=8*512=4096, V=4096
    buf->qkv.assign(M * (Hq * Dh_max + 2 * Hk_max * Dh_max), 0.0f);
    buf->attn_out.assign(M * Hq * Dh_max, 0.0f);
    buf->attn_logits.assign(static_cast<size_t>(Hq) * max_seq, 0.0f);

    // FFN buffers: max(intermediate=2112, gate_up=1408)
    int ffn_buf = std::max(cfg.intermediate, cfg.expert_inter * 2);
    buf->ffn_gate.assign(M * ffn_buf, 0.0f);
    buf->ffn_up.assign(M * ffn_buf, 0.0f);
    buf->ffn_hidden.assign(M * ffn_buf, 0.0f);

    // MoE scratch
    if (cfg.n_experts > 0) {
        buf->moe_router_logits.assign(M * cfg.n_experts, 0.0f);
        buf->moe_expert_out.assign(M * H, 0.0f);
    }

    // Clear SSM/DeltaNet buffers (not used by Adaptive)
    buf->dnet_qkv.clear();
    buf->dnet_gate.clear();
    buf->dnet_b_proj.clear();
    buf->dnet_a_proj.clear();
    buf->dnet_o.clear();
    buf->dnet_normed.clear();
    buf->ssm_conv_state.clear();
    buf->ssm_recurrent.clear();

}

void AdaptiveArch::attention_batch(int layer, int M, int pos0, KVCache& kv,
                                 const LayerWeights& W, const LayerTypeInfo& lti,
                                 const ModelConfig& cfg, ModelBuffers* buf) {
    const int Hq = lti.n_q_heads;
    const int Hk = lti.n_kv_heads;
    const int Dh = lti.head_dim;
    const int cache =
        (layer >= 0 && layer < static_cast<int>(kv_layer_idx_.size()) &&
         kv_layer_idx_[layer] >= 0)
            ? kv_layer_idx_[layer] : layer;
    const int Dh_store = kv.head_dim(cache);
    const int gqa = Hq / Hk;
    // Gemma files RMSNorm Q/K so scale 1. Llama files do not; use 1/sqrt(d).
    const float scale = (W.attn_q_norm && W.attn_k_norm)
        ? 1.0f
        : 1.0f / std::sqrt(static_cast<float>(Dh));
    const int max_seq = static_cast<int>(buf->attn_logits.size()) / cfg.n_q_heads;
    const int swa_window = lti.sliding_window;
    const bool use_v_proj = (W.attn_v != nullptr);
    const int qn = Hq * Dh;
    const int kn = Hk * Dh;
    // Per-token QKV layout in buf->qkv: [Q | K | V], V omitted if no v_proj.
    // Always reserve V space: for global layers V is a copy of K projection
    // (normed separately, no RoPE), so it needs its own buffer.
    const int row_qkv = qn + 2 * kn;

    for (int m = 0; m < M; m++) {
        const int pos = pos0 + m;
        float* qkv_m = buf->qkv.data() + static_cast<size_t>(m) * row_qkv;
        float* qp = qkv_m;
        float* kp = qkv_m + qn;
        float* vp = kp + kn;
        const float* xn = buf->x_norm.data() + static_cast<size_t>(m) * cfg.hidden;

        // Q/K/V projections. For global layers (no v_proj), copy K to V
        // before any norming so V stays as the raw K projection.
        // Batch Q/K/V into one Metal command buffer (shared input xn).
        {
            KTimer _t(KernelProf::k_qkv_proj);
            MatmulBatchSpec qkv[3];
            int nqkv = 0;
            qkv[nqkv++] = {xn, W.attn_q, qp, cfg.hidden, qn};
            qkv[nqkv++] = {xn, W.attn_k, kp, cfg.hidden, kn};
            if (use_v_proj)
                qkv[nqkv++] = {xn, W.attn_v, vp, cfg.hidden, kn};
            if (!matmul_gpu_batch(qkv, nqkv))
                matmul_gemm_batch(qkv, nqkv);
            if (!use_v_proj)
                std::memcpy(vp, kp, static_cast<size_t>(kn) * sizeof(float));
        }

        // Q/K per-head RMSNorm when those tensors exist (Gemma). Llama files
        // have no q/k norm tensors.
        {
            KTimer _t(KernelProf::k_qk_norm);
            if (W.attn_q_norm && W.attn_k_norm) {
                const float* qnw = reinterpret_cast<const float*>(W.attn_q_norm->data);
                ops::rmsnorm_rows(qp, qnw, qp, Hq, Dh, cfg.rms_eps);
                const float* knw = reinterpret_cast<const float*>(W.attn_k_norm->data);
                ops::rmsnorm_rows(kp, knw, kp, Hk, Dh, cfg.rms_eps);
                for (int h = 0; h < Hk; h++)
                    rmsnorm_no_scale(vp + h * Dh, vp + h * Dh, Dh, cfg.rms_eps);
            }
        }

        // RoPE on Q and K only (not V)
        {
            KTimer _t(KernelProf::k_rope);
            const int pairs = lti.rope_dim / 2;
            float cosines[256];
            float sines[256];
            for (int pair = 0; pair < pairs; ++pair) {
                double inverse = std::pow(
                    static_cast<double>(lti.rope_base),
                    -2.0 * pair / lti.rope_dim);
                if (lti.is_global &&
                    pair < static_cast<int>(rope_freqs_full_.size())) {
                    inverse *= rope_freqs_full_[pair];
                }
                const double angle = pos * inverse;
                cosines[pair] = static_cast<float>(std::cos(angle));
                sines[pair] = static_cast<float>(std::sin(angle));
            }
            ops::rope_apply(qp, Hq, Dh, pairs, cosines, sines);
            ops::rope_apply(kp, Hk, Dh, pairs, cosines, sines);
        }

        // Store K/V only on the owning layer. Shared-KV layers reuse cache.
        {
            KTimer _t(KernelProf::k_kv_store);
            if (lti.owns_kv) {
                for (int h = 0; h < Hk; h++) {
                    kv.store_k(cache, h, pos, kp + h * Dh);
                    kv.store_v(cache, h, pos, vp + h * Dh);
                }
            }
        }

        // Attention. Decode (M=1) uses contiguous FP16/FP32 streaming with
        // GQA amortization (K/V loaded once per KV head) and parallelism across
        // KV heads via the thread pool. Prefill and non-contiguous cache modes
        // fall back to the per-token load_k/load_v loop.
        {
            KTimer _t(KernelProf::k_attn_core);
            int start = (swa_window > 0) ? std::max(0, pos - swa_window + 1) : 0;
            int end = pos + 1;
            float* ao = buf->attn_out.data() + static_cast<size_t>(m) * Hq * Dh;

            const KVStorageKind storage = kv.storage_kind(cache);
            const bool ring = kv.ring(cache);
            const bool fp32_fast =
                storage == KVStorageKind::FP32 && !ring;
            const bool laplace_fast =
                (storage == KVStorageKind::Adaptive ||
                 storage == KVStorageKind::FixedQ4) && !ring;

            if (laplace_fast) {
                const bool rotated = kv.laplace_rotated(cache);
                static thread_local std::vector<float> query_wh, output_wh;
                size_t scratch_size = static_cast<size_t>(Hq) * Dh_store;
                query_wh.assign(scratch_size, 0.0f);
                output_wh.assign(scratch_size, 0.0f);
                for (int h = 0; h < Hq; h++) {
                    float* transformed = query_wh.data()
                                       + static_cast<size_t>(h) * Dh_store;
                    std::memcpy(transformed, qp + h * Dh,
                                static_cast<size_t>(Dh) * sizeof(float));
                    if (rotated) walsh_hadamard(transformed, Dh_store);
                }

                float* query_data = query_wh.data();
                float* output_data = output_wh.data();
                std::vector<int> ends(Hq, end);
                std::vector<const float*> queries(Hq);
                std::vector<float*> outputs(Hq);
                for (int h = 0; h < Hq; h++) {
                    queries[h] = query_data
                               + static_cast<size_t>(h) * Dh_store;
                    outputs[h] = output_data
                               + static_cast<size_t>(h) * Dh_store;
                }
                ThreadPool::get().parallel_for(Hk, [&](int kvh) {
                    int h0 = kvh * gqa;
                    kv.attention_batch_all_wh(
                        cache, kvh, gqa, ends.data() + h0,
                        queries.data() + h0, scale, outputs.data() + h0,
                        start);
                    for (int hi = 0; hi < gqa; hi++) {
                        int h = h0 + hi;
                        float* transformed_output = output_data
                            + static_cast<size_t>(h) * Dh_store;
                        if (rotated) {
                            inverse_walsh_hadamard(transformed_output, Dh_store);
                        }
                        std::memcpy(ao + h * Dh, transformed_output,
                                    static_cast<size_t>(Dh) * sizeof(float));
                    }
                });
            } else if (ring) {
                for (int d = 0; d < Hq * Dh; d++) ao[d] = 0.0f;
                ThreadPool::get().parallel_for(Hk, [&](int kvh) {
                    const int h0 = kvh * gqa;
                    float key[512];
                    float value[512];
                    for (int t = start; t < end; ++t) {
                        kv.load_k(cache, kvh, t, key);
                        for (int hi = 0; hi < gqa; ++hi) {
                            const int h = h0 + hi;
                            const float* qh = qp + h * Dh;
                            float* logits = buf->attn_logits.data() +
                                static_cast<size_t>(h) * max_seq;
                            logits[t] = ops::dot(qh, key, Dh) * scale;
                        }
                    }
                    for (int hi = 0; hi < gqa; ++hi) {
                        const int h = h0 + hi;
                        float* logits = buf->attn_logits.data() +
                            static_cast<size_t>(h) * max_seq;
                        float maximum = -1e30f;
                        for (int t = start; t < end; ++t) {
                            maximum = std::max(maximum, logits[t]);
                        }
                        float sum = 0.0f;
                        for (int t = start; t < end; ++t) {
                            logits[t] = std::exp(logits[t] - maximum);
                            sum += logits[t];
                        }
                        const float inverse = 1.0f / sum;
                        for (int t = start; t < end; ++t) {
                            logits[t] *= inverse;
                        }
                    }
                    for (int t = start; t < end; ++t) {
                        kv.load_v(cache, kvh, t, value);
                        for (int hi = 0; hi < gqa; ++hi) {
                            const int h = h0 + hi;
                            const float weight =
                                buf->attn_logits[static_cast<size_t>(h) *
                                                max_seq + t];
                            ops::axpy(ao + h * Dh, weight, value, Dh);
                        }
                    }
                });
            } else {
                // Zero all head outputs before parallel accumulation.
                for (int d = 0; d < Hq * Dh; d++) ao[d] = 0.0f;

                // Each KV head is independent: load K/V once, reuse across the
                // gqa query heads sharing it.
                ThreadPool::get().parallel_for(Hk, [&](int kvh) {
                    const int h0 = kvh * gqa;
                    const float* Kbase = fp32_fast ? kv.head_k(cache, kvh) : nullptr;
                    const float* Vbase = fp32_fast ? kv.head_v(cache, kvh) : nullptr;
                    const uint16_t* Kbase16 = fp32_fast ? nullptr : kv.head_k16(cache, kvh);
                    const uint16_t* Vbase16 = fp32_fast ? nullptr : kv.head_v16(cache, kvh);

                    // QK^T: stream contiguous K, fused FP16 dot (no FP32 materialization).
                    for (int t = start; t < end; t++) {
                        for (int hi = 0; hi < gqa; hi++) {
                            const float* qh = qp + (h0 + hi) * Dh;
                            float* la = buf->attn_logits.data() +
                                        static_cast<size_t>(h0 + hi) * max_seq;
                            if (fp32_fast)
                                la[t] = ops::dot(qh, Kbase +
                                        static_cast<size_t>(t) * Dh_store, Dh) * scale;
                            else
                                la[t] = ops::dot_f16(qh, Kbase16 +
                                        static_cast<size_t>(t) * Dh_store, Dh) * scale;
                        }
                    }
                    // Softmax per query head.
                    for (int hi = 0; hi < gqa; hi++) {
                        int h = h0 + hi;
                        float* la = buf->attn_logits.data() +
                                    static_cast<size_t>(h) * max_seq;
                        float maxv = -1e30f;
                        for (int t = start; t < end; t++)
                            if (la[t] > maxv) maxv = la[t];
                        float sumv = 0.0f;
                        for (int t = start; t < end; t++) {
                            la[t] = std::exp(la[t] - maxv);
                            sumv += la[t];
                        }
                        float inv = 1.0f / sumv;
                        for (int t = start; t < end; t++) la[t] *= inv;
                    }
                    // V accumulation: stream contiguous V, fused FP16 axpy.
                    for (int t = start; t < end; t++) {
                        for (int hi = 0; hi < gqa; hi++) {
                            int h = h0 + hi;
                            float w = buf->attn_logits[static_cast<size_t>(h) * max_seq + t];
                            if (w == 0.0f) continue;
                            if (fp32_fast)
                                ops::axpy(ao + h * Dh, w, Vbase +
                                          static_cast<size_t>(t) * Dh_store, Dh);
                            else
                                ops::axpy_f16(ao + h * Dh, w, Vbase16 +
                                              static_cast<size_t>(t) * Dh_store, Dh);
                        }
                    }
                });
            }
        }
    }

    // Output projection: [M, Hq*Dh] -> [M, hidden]
    {
        KTimer _t(KernelProf::k_o_proj);
        if (M == 1) {
            MatmulBatchSpec oproj = {buf->attn_out.data(), W.attn_output, buf->xb.data(),
                                     Hq * Dh, cfg.hidden};
            if (!matmul_gpu_batch(&oproj, 1))
                matmul_gemm_batch(&oproj, 1);
        } else {
            matmul_rows(buf->attn_out.data(), *W.attn_output, buf->xb.data(),
                        M, Hq * Dh, cfg.hidden);
        }
    }
}

void AdaptiveArch::moe_ffn(const LayerWeights& W, const ModelConfig& cfg,
                         int M, const float* residual, ModelBuffers* buf) {
    const int H = cfg.hidden;
    const int n_exp = cfg.n_experts;
    const int top_k = cfg.n_experts_used;
    const int exp_inter = cfg.expert_inter;
    const float inv_sqrt_h = 1.0f / std::sqrt(static_cast<float>(H));

    // Reusable scratch buffers - avoid per-token heap allocations in the
    // expert loop (was 8 * 30 = 240 malloc/free per token).
    static thread_local std::vector<float> tmp_buf, moe_in_buf, hidden_bufs;
    if ((int)tmp_buf.size() < H) tmp_buf.resize(H);
    if ((int)moe_in_buf.size() < H) moe_in_buf.resize(H);
    if ((int)hidden_bufs.size() < top_k * exp_inter) hidden_bufs.resize(top_k * exp_inter);

    for (int m = 0; m < M; m++) {
        const float* x = residual + static_cast<size_t>(m) * H;
        float* out = buf->moe_expert_out.data() + static_cast<size_t>(m) * H;
        std::memset(out, 0, static_cast<size_t>(H) * sizeof(float));

        // Router: scaleless RMSNorm, then scale, then project
        float* tmp = tmp_buf.data();
        float* moe_in = moe_in_buf.data();
        int top_idx[16];
        float top_w[16];
        const float* down_scale;
        {
            KTimer _t(KernelProf::k_moe_router);
            rmsnorm_no_scale(x, tmp, H, cfg.rms_eps);
            const float* gate_scale = reinterpret_cast<const float*>(W.moe_gate_inp_scale->data);
            for (int i = 0; i < H; i++) tmp[i] *= gate_scale[i] * inv_sqrt_h;

            float* logits = buf->moe_router_logits.data() + static_cast<size_t>(m) * n_exp;
            MatmulBatchSpec router_spec = {tmp, W.moe_gate_inp, logits, H, n_exp};
            matmul_gemm_batch(&router_spec, 1);

            // Softmax
            float maxl = logits[0];
            for (int e = 1; e < n_exp; e++) if (logits[e] > maxl) maxl = logits[e];
            float suml = 0.0f;
            for (int e = 0; e < n_exp; e++) { logits[e] = std::exp(logits[e] - maxl); suml += logits[e]; }
            for (int e = 0; e < n_exp; e++) logits[e] /= suml;

            // Top-k selection
            for (int k = 0; k < top_k; k++) {
                int best = 0;
                for (int e = 1; e < n_exp; e++) if (logits[e] > logits[best]) best = e;
                top_idx[k] = best;
                top_w[k] = logits[best];
                logits[best] = -1e30f;  // remove from consideration
            }
            // Renormalize
            float wsum = 0.0f;
            for (int k = 0; k < top_k; k++) wsum += top_w[k];
            for (int k = 0; k < top_k; k++) top_w[k] /= wsum;

            down_scale = reinterpret_cast<const float*>(W.moe_down_exps_scale->data);
        }

        const uint8_t* up_bases[16] = {};
        const uint8_t* dn_bases[16] = {};
        int local[16];
        const Tensor* up_w = W.moe_gate_up_exps;
        const Tensor* dn_w = W.moe_down_exps;
        const int* gemv_idx = top_idx;
        const uint8_t* const* up_b = nullptr;
        const uint8_t* const* dn_b = nullptr;
        if (LaplaceMoE::streaming_enabled()) {
            int order[16];
            for (int k = 0; k < top_k; k++) order[k] = k;
            std::sort(order, order + top_k, [&](int a, int b) {
                return top_idx[a] < top_idx[b];
            });
            int sorted_idx[16];
            float sorted_w[16];
            for (int t = 0; t < top_k; t++) {
                sorted_idx[t] = top_idx[order[t]];
                sorted_w[t] = top_w[order[t]];
            }
            for (int k = 0; k < top_k; k++) {
                top_idx[k] = sorted_idx[k];
                top_w[k] = sorted_w[k];
                local[k] = k;
            }
            gemv_idx = local;
            up_b = up_bases;
            dn_b = dn_bases;
            LaplaceMoE::load_experts(*W.moe_gate_up_exps, top_idx, top_k,
                                     nullptr, up_bases);
            LaplaceMoE::load_experts(*W.moe_down_exps, top_idx, top_k,
                                     nullptr, dn_bases);
        }
        {
            KTimer _t(KernelProf::k_moe_pre_norm);
            const float* pn2w = reinterpret_cast<const float*>(W.pre_ffw_norm_2->data);
            ops::rmsnorm(x, pn2w, moe_in, H, cfg.rms_eps);
        }
        {
            KTimer _t(KernelProf::k_moe_gate_up);
            fused_moe_gate_up_geglu(
                moe_in, *up_w, hidden_bufs.data(),
                gemv_idx, top_k, H, exp_inter, up_b);
        }
        float route_weight[16];
        for (int k = 0; k < top_k; k++)
            route_weight[k] = top_w[k] * down_scale[top_idx[k]];
        {
            KTimer _t(KernelProf::k_moe_down);
            fused_moe_down_accumulate(
                hidden_bufs.data(), *dn_w, gemv_idx,
                route_weight, top_k, exp_inter, H, out, dn_b);
        }
    }
}

void AdaptiveArch::forward_layer(int layer, const LayerWeights& W, const ModelConfig& cfg,
                               int M, int pos0, KVCache& kv, ModelBuffers* buf,
                               float* checkpoints) {
    (void)checkpoints;
    const int H = cfg.hidden;
    const LayerTypeInfo& lti = layer_types_[layer];

    // Token boundary: trigger per-100-token report when layer wraps to 0.
    if (layer == 0) g_kprof.maybe_print();

    TokenGraphBackend& token_graph = active_token_graph_backend();
    if (M == 1 && token_graph.active()) {
        if (layer == 0 && pos0 > 0 && metal_tok_kv_needs_seed()) {
            static thread_local std::vector<float> kpack, vpack;
            std::vector<char> seen(cfg.n_layers, 0);
            for (int i = 0; i < cfg.n_layers; i++) {
                if (!layer_types_[i].owns_kv) continue;
                int cid = (i < (int)kv_layer_idx_.size() && kv_layer_idx_[i] >= 0)
                    ? kv_layer_idx_[i] : i;
                if (cid < 0 || cid >= cfg.n_layers || seen[cid]) continue;
                seen[cid] = 1;
                const int Hk = layer_types_[i].n_kv_heads;
                const int Dh = layer_types_[i].head_dim;
                const int kn = Hk * Dh;
                if ((int)kpack.size() < kn) {
                    kpack.resize(kn);
                    vpack.resize(kn);
                }
                for (int t = 0; t < pos0; t++) {
                    for (int h = 0; h < Hk; h++) {
                        kv.load_k(cid, h, t, kpack.data() + h * Dh);
                        kv.load_v(cid, h, t, vpack.data() + h * Dh);
                    }
                    metal_tok_import_kv(cid, t, kpack.data(), vpack.data(), kn);
                }
            }
        }
        MetalTokLayer L;
        L.attn_norm = W.attn_norm;
        L.attn_q = W.attn_q;
        L.attn_k = W.attn_k;
        L.attn_v = W.attn_v;
        L.attn_o = W.attn_output;
        L.q_norm = W.attn_q_norm;
        L.k_norm = W.attn_k_norm;
        L.post_attn_norm = W.post_attn_norm;
        L.ffn_norm = W.ffn_norm;
        L.ffn_gate = W.ffn_gate;
        L.ffn_up = W.ffn_up;
        L.ffn_down = W.ffn_down;
        L.moe_gate = W.moe_gate_inp;
        L.moe_gate_scale = W.moe_gate_inp_scale;
        L.moe_up = W.moe_gate_up_exps;
        L.moe_dn = W.moe_down_exps;
        L.moe_dn_scale = W.moe_down_exps_scale;
        L.pre_ffw_2 = W.pre_ffw_norm_2;
        L.post_ffw_1 = W.post_ffw_norm_1;
        L.post_ffw_2 = W.post_ffw_norm_2;
        L.post_ffw = W.post_ffw_norm;
        L.out_scale = W.layer_output_scale;
        L.rope_freqs = rope_freqs_full_.empty() ? nullptr : rope_freqs_full_.data();
        L.n_rope_freqs = (int)rope_freqs_full_.size();
        L.H = H;
        L.inter = lti.intermediate > 0 ? lti.intermediate : cfg.intermediate;
        L.exp_inter = cfg.expert_inter;
        L.n_experts = cfg.n_experts;
        L.n_used = cfg.n_experts_used;
        L.Hq = lti.n_q_heads;
        L.Hk = lti.n_kv_heads;
        L.Dh = lti.head_dim;
        L.rope_dim = lti.rope_dim;
        L.window = lti.sliding_window;
        L.cache_id = (layer >= 0 && layer < (int)kv_layer_idx_.size() &&
                      kv_layer_idx_[layer] >= 0)
            ? kv_layer_idx_[layer] : layer;
        L.rope_base = lti.rope_base;
        L.attention_scale = 1.0f / std::sqrt(static_cast<float>(L.Dh));
        L.rms_eps = cfg.rms_eps;
        L.swiglu = lti.swiglu;
        L.owns_kv = lti.owns_kv;
        L.is_global = lti.is_global;
        if (token_graph.layer(L)) {
            static const bool tok_timing =
                std::getenv("LAPLACE_TOK_TIMING") != nullptr;
            if (tok_timing) {
                double ms = 0;
                metal_tok_flush(&ms);
                fprintf(stderr, "[tok-timing] layer %d: %.3f ms\n", layer, ms);
            }
            return;
        }
        fprintf(stderr, "[metal] tok_layer %d failed\n", layer);
        token_graph.abort();
        return;
    }

    // Pre-attention RMSNorm
    {
        KTimer _t(KernelProf::k_attn_norm);
        for (int m = 0; m < M; m++) {
            ops::rmsnorm(buf->x.data() + static_cast<size_t>(m) * H,
                         reinterpret_cast<const float*>(W.attn_norm->data),
                         buf->x_norm.data() + static_cast<size_t>(m) * H,
                         H, cfg.rms_eps);
        }
    }

    // Attention sub-block
    attention_batch(layer, M, pos0, kv, W, lti, cfg, buf);

    // Post-attention norm on attention output when present, then residual.
    {
        KTimer _t(KernelProf::k_post_attn_norm);
        for (int m = 0; m < M; m++) {
            if (W.post_attn_norm) {
                ops::rmsnorm(buf->xb.data() + static_cast<size_t>(m) * H,
                             reinterpret_cast<const float*>(W.post_attn_norm->data),
                             buf->xb.data() + static_cast<size_t>(m) * H,
                             H, cfg.rms_eps);
            }
            for (int j = 0; j < H; j++)
                buf->x[static_cast<size_t>(m) * H + j] += buf->xb[static_cast<size_t>(m) * H + j];
        }
    }

    // FFN block
    const bool has_moe = (W.moe_gate_inp != nullptr);

    // Dense FFN: ffn_norm(attn_out) -> gate/up -> geglu -> down
    const int inter = lti.intermediate > 0 ? lti.intermediate : cfg.intermediate;
    {
        KTimer _t(KernelProf::k_ffn_norm);
        for (int m = 0; m < M; m++) {
            ops::rmsnorm(buf->x.data() + static_cast<size_t>(m) * H,
                         reinterpret_cast<const float*>(W.ffn_norm->data),
                         buf->x_norm.data() + static_cast<size_t>(m) * H,
                         H, cfg.rms_eps);
        }
    }

    bool gpu_ffn_moe = false;
    if (M == 1 && has_moe && W.ffn_gate && W.ffn_up && W.ffn_down &&
        W.moe_gate_up_exps && W.moe_down_exps && matmul_gpu_available()) {
        const int n_exp = cfg.n_experts;
        const int top_k = cfg.n_experts_used;
        const int exp_inter = cfg.expert_inter;
        const float inv_sqrt_h = 1.0f / std::sqrt(static_cast<float>(H));
        static thread_local std::vector<float> rtmp, moe_in_buf;
        if ((int)rtmp.size() < H) rtmp.resize(H);
        if ((int)moe_in_buf.size() < H) moe_in_buf.resize(H);
        float* tmp = rtmp.data();
        rmsnorm_no_scale(buf->x.data(), tmp, H, cfg.rms_eps);
        const float* gate_scale =
            reinterpret_cast<const float*>(W.moe_gate_inp_scale->data);
        for (int i = 0; i < H; i++) tmp[i] *= gate_scale[i] * inv_sqrt_h;
        float* logits = buf->moe_router_logits.data();
        MatmulBatchSpec router_spec = {tmp, W.moe_gate_inp, logits, H, n_exp};
        matmul_gemm_batch(&router_spec, 1);
        float maxl = logits[0];
        for (int e = 1; e < n_exp; e++) if (logits[e] > maxl) maxl = logits[e];
        float suml = 0.0f;
        for (int e = 0; e < n_exp; e++) {
            logits[e] = std::exp(logits[e] - maxl);
            suml += logits[e];
        }
        for (int e = 0; e < n_exp; e++) logits[e] /= suml;
        int top_idx[16];
        float top_w[16];
        for (int k = 0; k < top_k; k++) {
            int best = 0;
            for (int e = 1; e < n_exp; e++) if (logits[e] > logits[best]) best = e;
            top_idx[k] = best;
            top_w[k] = logits[best];
            logits[best] = -1e30f;
        }
        float wsum = 0.0f;
        for (int k = 0; k < top_k; k++) wsum += top_w[k];
        for (int k = 0; k < top_k; k++) top_w[k] /= wsum;
        const float* down_scale =
            reinterpret_cast<const float*>(W.moe_down_exps_scale->data);
        const uint8_t* up_bases[16] = {};
        const uint8_t* dn_bases[16] = {};
        if (LaplaceMoE::streaming_enabled()) {
            int order[16];
            for (int k = 0; k < top_k; k++) order[k] = k;
            std::sort(order, order + top_k, [&](int a, int b) {
                return top_idx[a] < top_idx[b];
            });
            int sorted_idx[16];
            float sorted_w[16];
            for (int t = 0; t < top_k; t++) {
                sorted_idx[t] = top_idx[order[t]];
                sorted_w[t] = top_w[order[t]];
            }
            for (int k = 0; k < top_k; k++) {
                top_idx[k] = sorted_idx[k];
                top_w[k] = sorted_w[k];
            }
            LaplaceMoE::load_experts(*W.moe_gate_up_exps, top_idx, top_k,
                                     nullptr, up_bases);
            LaplaceMoE::load_experts(*W.moe_down_exps, top_idx, top_k,
                                     nullptr, dn_bases);
        }
        ops::rmsnorm(buf->x.data(),
                     reinterpret_cast<const float*>(W.pre_ffw_norm_2->data),
                     moe_in_buf.data(), H, cfg.rms_eps);
        float route_w[16];
        for (int k = 0; k < top_k; k++)
            route_w[k] = top_w[k] * down_scale[top_idx[k]];
        (void)up_bases;
        (void)dn_bases;
        gpu_ffn_moe = matmul_decode_ffn_moe(
            buf->x_norm.data(), *W.ffn_gate, *W.ffn_up, *W.ffn_down,
            buf->xb.data(), H, inter, lti.swiglu,
            moe_in_buf.data(), W.moe_gate_up_exps, W.moe_down_exps,
            top_idx, top_k, exp_inter, route_w,
            buf->moe_expert_out.data());
    }

    if (!gpu_ffn_moe) {
    {
        KTimer _t(KernelProf::k_ffn_gate_up);
        if (M == 1) {
            MatmulBatchSpec ffn[2] = {
                {buf->x_norm.data(), W.ffn_gate, buf->ffn_gate.data(), H, inter},
                {buf->x_norm.data(), W.ffn_up,   buf->ffn_up.data(),   H, inter},
            };
            if (!matmul_gpu_batch(ffn, 2))
                matmul_gemm_batch(ffn, 2);
        } else {
            matmul_rows(buf->x_norm.data(), *W.ffn_gate, buf->ffn_gate.data(), M, H, inter);
            matmul_rows(buf->x_norm.data(), *W.ffn_up,   buf->ffn_up.data(),   M, H, inter);
        }
    }
    {
        KTimer _t(KernelProf::k_geglu_dense);
        if (lti.swiglu) {
            ops::swiglu(buf->ffn_gate.data(), buf->ffn_up.data(), buf->ffn_hidden.data(),
                        M * inter);
        } else {
            ops::geglu(buf->ffn_gate.data(), buf->ffn_up.data(), buf->ffn_hidden.data(),
                       M * inter);
        }
    }
    {
        KTimer _t(KernelProf::k_ffn_down);
        if (M == 1) {
            MatmulBatchSpec dproj = {buf->ffn_hidden.data(), W.ffn_down, buf->xb.data(),
                                     inter, H};
            if (!matmul_gpu_batch(&dproj, 1))
                matmul_gemm_batch(&dproj, 1);
        } else {
            matmul_rows(buf->ffn_hidden.data(), *W.ffn_down, buf->xb.data(), M, inter, H);
        }
    }
    if (has_moe)
        moe_ffn(W, cfg, M, buf->x.data(), buf);
    } // !gpu_ffn_moe

    if (has_moe) {
        {
            KTimer _t(KernelProf::k_post_ffw_norm_1);
            for (int m = 0; m < M; m++) {
                ops::rmsnorm(buf->xb.data() + static_cast<size_t>(m) * H,
                             reinterpret_cast<const float*>(W.post_ffw_norm_1->data),
                             buf->xb.data() + static_cast<size_t>(m) * H,
                             H, cfg.rms_eps);
            }
        }

        // Post-norm on MoE output
        {
            KTimer _t(KernelProf::k_post_ffw_norm_2);
            for (int m = 0; m < M; m++) {
                ops::rmsnorm(buf->moe_expert_out.data() + static_cast<size_t>(m) * H,
                             reinterpret_cast<const float*>(W.post_ffw_norm_2->data),
                             buf->moe_expert_out.data() + static_cast<size_t>(m) * H,
                             H, cfg.rms_eps);
            }
        }
        // Combine: dense + MoE
        {
            KTimer _t(KernelProf::k_combine_dense_moe);
            for (size_t j = 0; j < static_cast<size_t>(M) * H; j++)
                buf->xb[j] += buf->moe_expert_out[j];
        }
    }

    // Post-FFW norm on combined output
    if (W.post_ffw_norm) {
        KTimer _t(KernelProf::k_post_ffw_norm);
        for (int m = 0; m < M; m++) {
            ops::rmsnorm(buf->xb.data() + static_cast<size_t>(m) * H,
                         reinterpret_cast<const float*>(W.post_ffw_norm->data),
                         buf->xb.data() + static_cast<size_t>(m) * H,
                         H, cfg.rms_eps);
        }
    }

    {
        KTimer _t(KernelProf::k_residual_add);
        for (size_t j = 0; j < static_cast<size_t>(M) * H; j++)
            buf->x[j] += buf->xb[j];
    }
    if (W.layer_output_scale) {
        KTimer _t(KernelProf::k_layer_out_scale);
        float s = reinterpret_cast<const float*>(W.layer_output_scale->data)[0];
        for (size_t j = 0; j < static_cast<size_t>(M) * H; j++)
            buf->x[j] *= s;
    }

    // LaplaceMoE: when streaming is enabled (model does not fit in RAM),
    // the OS page cache manages expert residency. Expert tensors are left
    // at default madvise so the OS can do sequential readahead per expert.
    // LaplaceMoE page-in pulls the active experts from the model file
    // ones this token needs, and the MoE dispatch uses sequential
    // per-expert matmuls instead of the fused kernel to avoid 8-way
    // random SSD access. The OS LRU keeps the hot working set.
}

} // namespace Laplace
