// arch_llama.cpp - standard Llama/Qwen2/Qwen3 transformer
#include "arch_llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "matmul.h"
#include "model.h"
#include "ops.h"
#include "threadpool.h"

namespace Laplace {

namespace {
// Add bias vector to each row of a [rows, cols] row-major matrix.
void add_bias_rows(float* out, const float* bias, int rows, int cols) {
    if (!bias) return;
    for (int r = 0; r < rows; r++) {
        float* row = out + static_cast<size_t>(r) * cols;
        for (int c = 0; c < cols; c++) row[c] += bias[c];
    }
}

// A/B toggle: set LAPLACE_LEGACY_ATTN=1 to use the pre-optimization scalar
// per-query-head attention loop (load_k/load_v per (h,t), no GQA amortization,
// no SIMD). Lets us compare kernels in the same binary without rebuilding.
bool use_legacy_attn() {
    static const bool v = std::getenv("LAPLACE_LEGACY_ATTN") != nullptr;
    return v;
}
} // namespace

bool LlamaArch::load_config(const GGUFContext& gguf, ModelConfig* cfg) {
    const auto& m = gguf.metadata();
    auto arch = meta_str(m, "general.architecture");
    if (!arch || (*arch != "llama" && *arch != "qwen2" && *arch != "qwen3")) return false;
    const std::string A = *arch + ".";

    cfg->n_layers     = static_cast<int>(meta_int(m, (A + "block_count").c_str()));
    cfg->hidden       = static_cast<int>(meta_int(m, (A + "embedding_length").c_str()));
    cfg->intermediate = static_cast<int>(meta_int(m, (A + "feed_forward_length").c_str()));
    cfg->n_q_heads    = static_cast<int>(meta_int(m, (A + "attention.head_count").c_str()));
    cfg->n_kv_heads   = static_cast<int>(meta_int(m, (A + "attention.head_count_kv").c_str()));
    cfg->head_dim     = static_cast<int>(meta_int(m, (A + "attention.key_length").c_str(), cfg->hidden / cfg->n_q_heads));
    if (auto* vl = meta_as<MetaArrayI32>(m, (A + "attention.value_length").c_str())) {
        // Some models (e.g. Qwen3) store key_length and value_length separately.
        // We keep a single head_dim; verify they match if both are present.
        if (!vl->empty() && (*vl)[0] != cfg->head_dim) {
            fprintf(stderr, "llama: key_length (%d) != value_length (%d) not supported\n",
                    cfg->head_dim, (*vl)[0]);
            return false;
        }
    }
    cfg->max_seq_len   = static_cast<int>(meta_int(m, (A + "context_length").c_str(), 2048));
    cfg->rms_eps       = static_cast<float>(meta_float(m, (A + "attention.layer_norm_rms_epsilon").c_str(), 1e-6));
    cfg->rope_freq_base = static_cast<float>(meta_float(m, (A + "rope.freq_base").c_str(), 1e4));
    cfg->rope_dim_count = static_cast<int>(meta_int(m, (A + "rope.dimension_count").c_str(), cfg->head_dim));
    return true;
}

bool LlamaArch::load_weights(const GGUFContext& gguf, const ModelConfig& cfg,
                             std::vector<LayerWeights>* layers,
                             std::vector<bool>* is_attention,
                             std::vector<int>* kv_layer_idx) {
    layers->assign(cfg.n_layers, {});
    is_attention->assign(cfg.n_layers, true);
    kv_layer_idx->assign(cfg.n_layers, -1);

    const uint64_t H    = static_cast<uint64_t>(cfg.hidden);
    const uint64_t qd   = static_cast<uint64_t>(cfg.n_q_heads) * cfg.head_dim;
    const uint64_t kvd  = static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
    const uint64_t inter = static_cast<uint64_t>(cfg.intermediate);

    for (int i = 0; i < cfg.n_layers; i++) {
        const std::string p = "blk." + std::to_string(i) + ".";
        LayerWeights& L = (*layers)[i];

        L.attn_norm      = gguf.find_tensor((p + "attn_norm.weight").c_str());
        L.post_attn_norm = gguf.find_tensor((p + "post_attention_norm.weight").c_str());
        L.ffn_norm       = gguf.find_tensor((p + "ffn_norm.weight").c_str());
        L.ffn_gate       = gguf.find_tensor((p + "ffn_gate.weight").c_str());
        L.ffn_up         = gguf.find_tensor((p + "ffn_up.weight").c_str());
        L.ffn_down       = gguf.find_tensor((p + "ffn_down.weight").c_str());

        L.attn_q         = gguf.find_tensor((p + "attn_q.weight").c_str());
        L.attn_k         = gguf.find_tensor((p + "attn_k.weight").c_str());
        L.attn_v         = gguf.find_tensor((p + "attn_v.weight").c_str());
        L.attn_q_bias    = gguf.find_tensor((p + "attn_q.bias").c_str());
        L.attn_k_bias    = gguf.find_tensor((p + "attn_k.bias").c_str());
        L.attn_v_bias    = gguf.find_tensor((p + "attn_v.bias").c_str());
        L.attn_output    = gguf.find_tensor((p + "attn_output.weight").c_str());
        L.attn_qkv       = gguf.find_tensor((p + "attn_qkv.weight").c_str());

        // Optional Q/K RMSNorm (present in Qwen3, absent in Llama/Qwen2)
        L.attn_q_norm    = gguf.find_tensor((p + "attn_q_norm.weight").c_str());
        L.attn_k_norm    = gguf.find_tensor((p + "attn_k_norm.weight").c_str());

        if (!L.ffn_norm) L.ffn_norm = L.post_attn_norm;

        (*kv_layer_idx)[i] = i;

        // Validate shapes.
        if (!L.attn_norm || !L.ffn_norm || !L.ffn_gate || !L.ffn_up || !L.ffn_down) {
            fprintf(stderr, "llama: layer %d missing norms or FFN weights\n", i);
            return false;
        }
        if (!L.attn_output) {
            fprintf(stderr, "llama: layer %d missing attn_output\n", i);
            return false;
        }
        // GGUF dims are reversed: dims[0] is the innermost (input) dimension,
        // dims[1] is the outermost (output) dimension.
        if (L.attn_output->dims[0] != qd || L.attn_output->dims[1] != H) {
            fprintf(stderr, "llama: layer %d attn_output shape mismatch\n", i);
            return false;
        }
        if (L.attn_qkv) {
            // Fused QKV: [H, qd + 2*kvd]
            if (L.attn_qkv->dims[0] != H ||
                L.attn_qkv->dims[1] != qd + 2 * kvd) {
                fprintf(stderr, "llama: layer %d fused attn_qkv shape mismatch\n", i);
                return false;
            }
        } else if (L.attn_q && L.attn_k && L.attn_v) {
            if (L.attn_q->dims[0] != H || L.attn_q->dims[1] != qd ||
                L.attn_k->dims[0] != H || L.attn_k->dims[1] != kvd ||
                L.attn_v->dims[0] != H || L.attn_v->dims[1] != kvd) {
                fprintf(stderr, "llama: layer %d attention Q/K/V shape mismatch\n", i);
                return false;
            }
        } else {
            fprintf(stderr, "llama: layer %d missing attention Q/K/V weights\n", i);
            return false;
        }
        if (L.attn_q_norm && (L.attn_q_norm->dims[0] != static_cast<uint64_t>(cfg.head_dim))) {
            fprintf(stderr, "llama: layer %d q_norm size mismatch\n", i);
            return false;
        }
        if (L.attn_k_norm && (L.attn_k_norm->dims[0] != static_cast<uint64_t>(cfg.head_dim))) {
            fprintf(stderr, "llama: layer %d k_norm size mismatch\n", i);
            return false;
        }
        if (L.ffn_gate->dims[0] != H || L.ffn_gate->dims[1] != inter ||
            L.ffn_up->dims[0]   != H || L.ffn_up->dims[1]   != inter ||
            L.ffn_down->dims[0] != inter || L.ffn_down->dims[1] != H) {
            fprintf(stderr, "llama: layer %d FFN shape mismatch\n", i);
            return false;
        }
    }
    return true;
}

void LlamaArch::reserve(const ModelConfig& cfg, int max_seq, int max_batch,
                        ModelBuffers* buf) {
    const size_t M = static_cast<size_t>(max_batch);
    buf->x.assign(M * cfg.hidden, 0.0f);
    buf->xb.assign(M * cfg.hidden, 0.0f);
    buf->x_norm.assign(M * cfg.hidden, 0.0f);

    buf->qkv.assign(M * (cfg.n_q_heads + 2 * cfg.n_kv_heads) * cfg.head_dim, 0.0f);
    buf->attn_out.assign(M * cfg.n_q_heads * cfg.head_dim, 0.0f);
    buf->attn_logits.assign(static_cast<size_t>(cfg.n_q_heads) * max_seq, 0.0f);

    buf->ffn_gate.assign(M * cfg.intermediate, 0.0f);
    buf->ffn_up.assign(M * cfg.intermediate, 0.0f);
    buf->ffn_hidden.assign(M * cfg.intermediate, 0.0f);

    // No DeltaNet / SSM state for standard transformers.
    buf->dnet_qkv.clear();
    buf->dnet_gate.clear();
    buf->dnet_b_proj.clear();
    buf->dnet_a_proj.clear();
    buf->dnet_o.clear();
    buf->dnet_normed.clear();
    buf->ssm_conv_state.clear();
    buf->ssm_recurrent.clear();
}

void LlamaArch::attention_batch(int layer, int M, int pos0, KVCache& kv,
                                const LayerWeights& W, const ModelConfig& cfg,
                                ModelBuffers* buf) {
    const int Hq = cfg.n_q_heads;
    const int Hk = cfg.n_kv_heads;
    const int Dh = cfg.head_dim;
    const int H  = cfg.hidden;
    const int max_seq = static_cast<int>(buf->attn_logits.size()) / cfg.n_q_heads;

    const int qn = Hq * Dh;
    const int kn = Hk * Dh;

    float* qp_all = buf->qkv.data();                       // [M, qn]
    float* kp_all = qp_all + static_cast<size_t>(M) * qn;  // [M, kn]
    float* vp_all = kp_all + static_cast<size_t>(M) * kn;  // [M, kn]

    if (W.attn_qkv) {
        matmul_rows(buf->x_norm.data(), *W.attn_qkv, buf->qkv.data(), M, H, qn + 2 * kn);
        // Fused QKV bias, if present, is laid out as [q_bias | k_bias | v_bias].
        if (W.attn_q_bias || W.attn_k_bias || W.attn_v_bias) {
            add_bias_rows(qp_all, W.attn_q_bias ? reinterpret_cast<const float*>(W.attn_q_bias->data) : nullptr, M, qn);
            add_bias_rows(kp_all, W.attn_k_bias ? reinterpret_cast<const float*>(W.attn_k_bias->data) : nullptr, M, kn);
            add_bias_rows(vp_all, W.attn_v_bias ? reinterpret_cast<const float*>(W.attn_v_bias->data) : nullptr, M, kn);
        }
    } else {
        matmul_rows(buf->x_norm.data(), *W.attn_q, qp_all, M, H, qn);
        matmul_rows(buf->x_norm.data(), *W.attn_k, kp_all, M, H, kn);
        matmul_rows(buf->x_norm.data(), *W.attn_v, vp_all, M, H, kn);
        add_bias_rows(qp_all, W.attn_q_bias ? reinterpret_cast<const float*>(W.attn_q_bias->data) : nullptr, M, qn);
        add_bias_rows(kp_all, W.attn_k_bias ? reinterpret_cast<const float*>(W.attn_k_bias->data) : nullptr, M, kn);
        add_bias_rows(vp_all, W.attn_v_bias ? reinterpret_cast<const float*>(W.attn_v_bias->data) : nullptr, M, kn);
    }

    // Optional Q/K RMSNorm (Qwen3).
    if (W.attn_q_norm) {
        const float* qnw = reinterpret_cast<const float*>(W.attn_q_norm->data);
        ops::rmsnorm_rows(qp_all, qnw, qp_all, M * Hq, Dh, cfg.rms_eps);
    }
    if (W.attn_k_norm) {
        const float* knw = reinterpret_cast<const float*>(W.attn_k_norm->data);
        ops::rmsnorm_rows(kp_all, knw, kp_all, M * Hk, Dh, cfg.rms_eps);
    }

    // RoPE on Q and K.
    for (int m = 0; m < M; m++) {
        const int pos = pos0 + m;
        const float* cs = buf->rope_cos.data() + static_cast<size_t>(pos) * buf->rope_pairs;
        const float* sn = buf->rope_sin.data() + static_cast<size_t>(pos) * buf->rope_pairs;
        ops::rope_apply(qp_all + static_cast<size_t>(m) * qn, Hq, Dh, buf->rope_pairs, cs, sn);
        ops::rope_apply(kp_all + static_cast<size_t>(m) * kn, Hk, Dh, buf->rope_pairs, cs, sn);
    }

    // Store K, V and compute causal attention.
    const int gqa = Hq / Hk;
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));

    for (int m = 0; m < M; m++) {
        const int pos = pos0 + m;
        float* qp = qp_all + static_cast<size_t>(m) * qn;
        float* kp = kp_all + static_cast<size_t>(m) * kn;
        float* vp = vp_all + static_cast<size_t>(m) * kn;

        for (int h = 0; h < Hk; h++) {
            kv.store_k(layer, h, pos, kp + h * Dh);
            kv.store_v(layer, h, pos, vp + h * Dh);
        }

        // Causal attention. Default path is GQA-amortized + SIMD (iterate by
        // KV head, load each K/V once, reuse across the gqa group, contiguous
        // head_k/head_v in FP32). LAPLACE_LEGACY_ATTN=1 selects the original
        // scalar per-query-head loop for A/B comparison.
        const int end = pos + 1;
        float* ao_m = buf->attn_out.data() + static_cast<size_t>(m) * Hq * Dh;

        if (use_legacy_attn()) {
            std::vector<float> kt_buf(Dh), vt_buf(Dh);
            for (int h = 0; h < Hq; h++) {
                int kvh = h / gqa;
                const float* qh = qp + h * Dh;
                float* la = buf->attn_logits.data() + static_cast<size_t>(h) * max_seq;
                for (int t = 0; t < end; t++) {
                    kv.load_k(layer, kvh, t, kt_buf.data());
                    float s = 0.0f;
                    for (int d = 0; d < Dh; d++) s += qh[d] * kt_buf[d];
                    la[t] = s * scale;
                }
                float maxv = la[0];
                for (int t = 1; t < end; t++) if (la[t] > maxv) maxv = la[t];
                float sumv = 0.0f;
                for (int t = 0; t < end; t++) { la[t] = std::exp(la[t] - maxv); sumv += la[t]; }
                for (int t = 0; t < end; t++) la[t] /= sumv;
                float* ao = ao_m + h * Dh;
                for (int d = 0; d < Dh; d++) ao[d] = 0.0f;
                for (int t = 0; t < end; t++) {
                    kv.load_v(layer, kvh, t, vt_buf.data());
                    float w = la[t];
                    for (int d = 0; d < Dh; d++) ao[d] += w * vt_buf[d];
                }
            }
        } else {
            // FP32/FP16 fast path: contiguous head streaming.
            // FP32 uses head_k/head_v (float*), FP16 uses head_k16/head_v16
            // (uint16_t*) with fused NEON FP16 dot/axpy.
            const bool fp32_fast = (kv.mode() == KVCacheMode::FP32);
            for (int d = 0; d < Hq * Dh; d++) ao_m[d] = 0.0f;
            for (int kvh = 0; kvh < Hk; kvh++) {
                const int h0 = kvh * gqa;
                const float* Kbase = fp32_fast ? kv.head_k(layer, kvh) : nullptr;
                const float* Vbase = fp32_fast ? kv.head_v(layer, kvh) : nullptr;
                const uint16_t* Kbase16 = fp32_fast ? nullptr : kv.head_k16(layer, kvh);
                const uint16_t* Vbase16 = fp32_fast ? nullptr : kv.head_v16(layer, kvh);
                for (int t = 0; t < end; t++) {
                    for (int hi = 0; hi < gqa; hi++) {
                        const float* qh = qp + (h0 + hi) * Dh;
                        float* la = buf->attn_logits.data() + static_cast<size_t>(h0 + hi) * max_seq;
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
        }
    }

    // Output projection: [M, Hq*Dh] -> [M, hidden].
    matmul_rows(buf->attn_out.data(), *W.attn_output, buf->xb.data(),
                M, Hq * Dh, H);
}

void LlamaArch::attention_batch_wh(int layer, int M, int pos0, KVCache& kv,
                                    const LayerWeights& W, const ModelConfig& cfg,
                                    ModelBuffers* buf) {
    const int Hq = cfg.n_q_heads;
    const int Hk = cfg.n_kv_heads;
    const int Dh = cfg.head_dim;
    const int H  = cfg.hidden;
    const bool rotated = kv.laplace_rotated();
    const int qn = Hq * Dh;
    const int kn = Hk * Dh;

    float* qp_all = buf->qkv.data();                       // [M, qn]
    float* kp_all = qp_all + static_cast<size_t>(M) * qn;  // [M, kn]
    float* vp_all = kp_all + static_cast<size_t>(M) * kn;  // [M, kn]

    if (W.attn_qkv) {
        matmul_rows(buf->x_norm.data(), *W.attn_qkv, buf->qkv.data(), M, H, qn + 2 * kn);
        // Fused QKV bias, if present, is laid out as [q_bias | k_bias | v_bias].
        if (W.attn_q_bias || W.attn_k_bias || W.attn_v_bias) {
            add_bias_rows(qp_all, W.attn_q_bias ? reinterpret_cast<const float*>(W.attn_q_bias->data) : nullptr, M, qn);
            add_bias_rows(kp_all, W.attn_k_bias ? reinterpret_cast<const float*>(W.attn_k_bias->data) : nullptr, M, kn);
            add_bias_rows(vp_all, W.attn_v_bias ? reinterpret_cast<const float*>(W.attn_v_bias->data) : nullptr, M, kn);
        }
    } else {
        matmul_rows(buf->x_norm.data(), *W.attn_q, qp_all, M, H, qn);
        matmul_rows(buf->x_norm.data(), *W.attn_k, kp_all, M, H, kn);
        matmul_rows(buf->x_norm.data(), *W.attn_v, vp_all, M, H, kn);
        add_bias_rows(qp_all, W.attn_q_bias ? reinterpret_cast<const float*>(W.attn_q_bias->data) : nullptr, M, qn);
        add_bias_rows(kp_all, W.attn_k_bias ? reinterpret_cast<const float*>(W.attn_k_bias->data) : nullptr, M, kn);
        add_bias_rows(vp_all, W.attn_v_bias ? reinterpret_cast<const float*>(W.attn_v_bias->data) : nullptr, M, kn);
    }

    // Optional Q/K RMSNorm (Qwen3).
    if (W.attn_q_norm) {
        const float* qnw = reinterpret_cast<const float*>(W.attn_q_norm->data);
        ops::rmsnorm_rows(qp_all, qnw, qp_all, M * Hq, Dh, cfg.rms_eps);
    }
    if (W.attn_k_norm) {
        const float* knw = reinterpret_cast<const float*>(W.attn_k_norm->data);
        ops::rmsnorm_rows(kp_all, knw, kp_all, M * Hk, Dh, cfg.rms_eps);
    }

    // RoPE on Q and K, then enter LaplaceKV's storage domain.
    for (int m = 0; m < M; m++) {
        const int pos = pos0 + m;
        const float* cs = buf->rope_cos.data() + static_cast<size_t>(pos) * buf->rope_pairs;
        const float* sn = buf->rope_sin.data() + static_cast<size_t>(pos) * buf->rope_pairs;
        float* qp = qp_all + static_cast<size_t>(m) * qn;
        float* kp = kp_all + static_cast<size_t>(m) * kn;
        float* vp = vp_all + static_cast<size_t>(m) * kn;
        ops::rope_apply(qp, Hq, Dh, buf->rope_pairs, cs, sn);
        ops::rope_apply(kp, Hk, Dh, buf->rope_pairs, cs, sn);

        for (int h = 0; h < Hk; h++) {
            if (rotated) {
                walsh_hadamard(kp + h * Dh, Dh);
                walsh_hadamard(vp + h * Dh, Dh);
            }
            kv.store_k_wh(layer, h, pos, kp + h * Dh);
            kv.store_v_wh(layer, h, pos, vp + h * Dh);
        }
        if (rotated) {
            for (int h = 0; h < Hq; h++) walsh_hadamard(qp + h * Dh, Dh);
        }
    }

    // Causal attention entirely in WH domain.
    // Process query heads in GQA groups so each packed K/V vector is dequantized
    // only once per KV head position and reused by all query heads in the group.
    const int gqa = Hq / Hk;
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));

    if (kv.streaming()) {
        std::fill_n(buf->attn_out.data(),
                    static_cast<size_t>(M) * Hq * Dh, 0.0f);
        ThreadPool::get().parallel_for(Hk, [&](int kvh) {
            const int h0 = kvh * gqa;
            const int count = M * gqa;
            std::vector<int> ends(count);
            std::vector<const float*> queries(count);
            std::vector<float*> outputs(count);
            for (int m = 0; m < M; m++) {
                for (int hi = 0; hi < gqa; hi++) {
                    int index = m * gqa + hi;
                    int h = h0 + hi;
                    ends[index] = pos0 + m + 1;
                    queries[index] = qp_all + static_cast<size_t>(m) * qn
                                   + h * Dh;
                    outputs[index] = buf->attn_out.data()
                                   + static_cast<size_t>(m) * Hq * Dh
                                   + h * Dh;
                }
            }
            kv.attention_batch_all_wh(
                layer, kvh, count, ends.data(), queries.data(),
                scale, outputs.data());
        });
        if (rotated) {
            for (int m = 0; m < M; m++) {
                float* output = buf->attn_out.data()
                              + static_cast<size_t>(m) * Hq * Dh;
                for (int h = 0; h < Hq; h++) {
                    walsh_hadamard(output + h * Dh, Dh);
                }
            }
        }
        matmul_rows(buf->attn_out.data(), *W.attn_output, buf->xb.data(),
                    M, Hq * Dh, H);
        return;
    }

    for (int m = 0; m < M; m++) {
        const int pos = pos0 + m;
        float* qp = qp_all + static_cast<size_t>(m) * qn;
        const int end = pos + 1;
        float* ao_m = buf->attn_out.data() + static_cast<size_t>(m) * Hq * Dh;
        for (int d = 0; d < Hq * Dh; d++) ao_m[d] = 0.0f;

        ThreadPool::get().parallel_for(Hq, [&](int h) {
            kv.attention_all_wh(
                layer, h / gqa, end, qp + h * Dh,
                scale, ao_m + h * Dh);
        });
        if (rotated) {
            for (int h = 0; h < Hq; h++) {
                walsh_hadamard(ao_m + h * Dh, Dh);
            }
        }
    }

    // Output projection: [M, Hq*Dh] -> [M, hidden].
    matmul_rows(buf->attn_out.data(), *W.attn_output, buf->xb.data(),
                M, Hq * Dh, H);
}

void LlamaArch::forward_layer(int layer, const LayerWeights& W, const ModelConfig& cfg,
                              int M, int pos0, KVCache& kv, ModelBuffers* buf,
                              float* checkpoints) {
    (void)checkpoints;  // standard transformers have no SSM state to checkpoint
    const int H = cfg.hidden;

    // Pre-attention RMSNorm.
    for (int m = 0; m < M; m++) {
        ops::rmsnorm(buf->x.data() + static_cast<size_t>(m) * H,
                     reinterpret_cast<const float*>(W.attn_norm->data),
                     buf->x_norm.data() + static_cast<size_t>(m) * H,
                     H, cfg.rms_eps);
    }

    if (kv.mode() == KVCacheMode::LAPLACE) {
        attention_batch_wh(layer, M, pos0, kv, W, cfg, buf);
    } else {
        attention_batch(layer, M, pos0, kv, W, cfg, buf);
    }
    for (size_t j = 0; j < static_cast<size_t>(M) * H; j++) buf->x[j] += buf->xb[j];

    // Pre-FFN RMSNorm.
    for (int m = 0; m < M; m++) {
        ops::rmsnorm(buf->x.data() + static_cast<size_t>(m) * H,
                     reinterpret_cast<const float*>(W.ffn_norm->data),
                     buf->x_norm.data() + static_cast<size_t>(m) * H,
                     H, cfg.rms_eps);
    }

    // FFN sub-block.
    matmul_rows(buf->x_norm.data(), *W.ffn_gate, buf->ffn_gate.data(), M, H, cfg.intermediate);
    matmul_rows(buf->x_norm.data(), *W.ffn_up,   buf->ffn_up.data(),   M, H, cfg.intermediate);
    ops::swiglu(buf->ffn_gate.data(), buf->ffn_up.data(), buf->ffn_hidden.data(), M * cfg.intermediate);
    matmul_rows(buf->ffn_hidden.data(), *W.ffn_down, buf->xb.data(), M, cfg.intermediate, H);
    for (size_t j = 0; j < static_cast<size_t>(M) * H; j++) buf->x[j] += buf->xb[j];
}

} // namespace Laplace
