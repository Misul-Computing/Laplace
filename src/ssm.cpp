// ssm.cpp - Gated DeltaNet single-token recurrent core
#include "ssm.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "ops.h"
#include "trace.h"

namespace Laplace {

namespace {

inline bool is_power_of_two(int x) { return x > 0 && (x & (x - 1)) == 0; }

inline float silu_f(float x)   { return x / (1.0f + std::exp(-x)); }
inline float sigmoid_f(float x){ return 1.0f / (1.0f + std::exp(-x)); }
inline float softplus_f(float x){ return std::log1p(std::exp(x)); }

inline void l2norm_inplace(float* v, int n) {
    float sumsq = 0.0f;
    for (int i = 0; i < n; i++) sumsq += v[i] * v[i];
    float inv = 1.0f / std::sqrt(sumsq + 1e-6f);  // eps for numerical stability
    for (int i = 0; i < n; i++) v[i] *= inv;
}

// Causal depthwise conv1d, single step.
// GGUF stores the conv weight with dims = [K, C] (innermost dim is K), so
// channel c's K taps are contiguous at w_p[c*K + 0 .. c*K + K-1]. The last
// tap multiplies the newest sample.
// conv_state layout: [C * (K-1)], one row of (K-1) floats per channel.
void causal_conv1d_step(
    const float* in,        // [C]
    const float* wp,        // [C * K] = C rows of K taps
    float* conv_state,      // [C * (K-1)]
    float* out,             // [C]
    int C, int K
) {
    int hist = K - 1;
    // Single pass per channel: compute, update history, then write the
    // output. `out` may alias `in`, so in[c] must not be read after out[c]
    // is stored.
    for (int c = 0; c < C; c++) {
        float x_new = in[c];
        float* hist_row = conv_state + c * hist;
        const float* wc = wp + c * K;
        float acc = wc[K - 1] * x_new;
        for (int k = 0; k < hist; k++) acc += wc[k] * hist_row[k];
        for (int k = 0; k + 1 < hist; k++) hist_row[k] = hist_row[k + 1];
        hist_row[hist - 1] = x_new;
        out[c] = silu_f(acc);
    }
}

} // namespace

void deltanet_token(
    const DeltaNetParams& p,
    float* conv_state,
    float* recurrent,
    float* qkv_proj,
    const float* gate_proj,
    const float* b_proj,
    const float* a_proj,
    float* o_raw,
    float* o_normed
) {
    const int KVH  = p.G;            // k heads
    const int D    = p.D;
    const int Vh   = p.num_v_heads;  // v heads
    const int KD   = p.key_dim;
    if (D > 128 || Vh < 1 || KVH < 1 || Vh % KVH != 0) {
        fprintf(stderr, "deltanet: unsupported geometry G=%d Vh=%d D=%d\n",
                KVH, Vh, D);
        return;
    }
    const int r = Vh / KVH;          // v heads per k head
    constexpr float rms_eps = 1e-6f;
    (void)r;

    // 1. Causal depthwise conv1d on the QKV (followed by SiLU), in place.
    causal_conv1d_step(qkv_proj, p.conv_w, conv_state, qkv_proj,
                       p.conv_dim, p.conv_kernel);
    trace("dn_conv_silu", -1, qkv_proj, p.conv_dim);

    // 2. L2-normalize Q and K per k-head (in place). V is left as-is.
    // Q is scaled by 1/sqrt(D): near-zero outputs are eps-dominated in
    // the gated RMSNorm, so the reference's scaling must be reproduced.
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(D));
    for (int h = 0; h < KVH; h++) {
        float* q_h = qkv_proj + h * D;
        l2norm_inplace(q_h, D);
        for (int i = 0; i < D; i++) q_h[i] *= q_scale;
        l2norm_inplace(qkv_proj + KD + h * D, D);
    }
    const float* v_base = qkv_proj + 2 * KD;

    // 3+4. Per-v-head state update. Each v-head owns a state matrix; its
    // q and k come from the parent k-head (r v-heads share one k head).
    // Decay and beta are indexed by v-head.
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int hv = 0; hv < Vh; hv++) {
        // Tiled v-head order in the GGUF: head hv sits in k-group hv % KVH.
        const int kh = hv % KVH;
        const float* q_g = qkv_proj + kh * D;
        const float* k_g = qkv_proj + KD + kh * D;
        const float* v_g = v_base + hv * D;
        float* S = recurrent + hv * D * D;

        float A     = p.A[hv];
        float a_t   = softplus_f(a_proj[hv] + p.dt_bias[hv]);
        float g_d   = std::exp(A * a_t);
        float beta  = sigmoid_f(b_proj[hv]);

        float retrieved[128];
        for (int j = 0; j < D; j++) retrieved[j] = 0.0f;
        for (int i = 0; i < D; i++) {
            float* Si = S + i * D;
            float k_i = k_g[i];
            for (int j = 0; j < D; j++) {
                Si[j] *= g_d;
                retrieved[j] += Si[j] * k_i;
            }
        }

        float delta[128];
        for (int j = 0; j < D; j++) delta[j] = beta * (v_g[j] - retrieved[j]);

        float* o_g = o_raw + hv * D;
        for (int j = 0; j < D; j++) o_g[j] = 0.0f;
        for (int i = 0; i < D; i++) {
            float* Si = S + i * D;
            float k_i = k_g[i];
            float q_i = q_g[i];
            for (int j = 0; j < D; j++) {
                Si[j] += k_i * delta[j];
                o_g[j] += Si[j] * q_i;
            }
        }
    }

    // 5. Gated RMSNorm, Qwen3-Next order (NOT Mamba2): normalize first,
    // apply the per-head weight (length D), THEN multiply by silu(z).
    for (int hv = 0; hv < Vh; hv++) {
        const float* o_g = o_raw + hv * D;
        const float* z_g = p.gate_fused
            ? qkv_proj + 2 * KD + p.inner + hv * D
            : gate_proj + hv * D;
        float* on_g = o_normed + hv * D;
        float ms = 0.0f;
        for (int i = 0; i < D; i++) ms += o_g[i] * o_g[i];
        float inv = 1.0f / std::sqrt(ms / D + rms_eps);
        for (int i = 0; i < D; i++) {
            on_g[i] = o_g[i] * inv * p.ssm_norm[i] * silu_f(z_g[i]);
        }
    }
    trace("dn_o_raw", -1, o_raw, p.inner);
    trace("dn_o_normed", -1, o_normed, p.inner);
}

void deltanet_token_wh(
    const DeltaNetParams& p,
    float* conv_state,
    float* recurrent_wh,
    float* qkv_proj,
    const float* gate_proj,
    const float* b_proj,
    const float* a_proj,
    float* o_raw,
    float* o_normed
) {
    const int KVH  = p.G;
    const int D    = p.D;
    const int Vh   = p.num_v_heads;
    const int KD   = p.key_dim;
    if (D > 128 || Vh < 1 || KVH < 1 || Vh % KVH != 0) {
        fprintf(stderr, "deltanet_wh: unsupported geometry G=%d Vh=%d D=%d\n",
                KVH, Vh, D);
        return;
    }
    if (!is_power_of_two(D)) {
        fprintf(stderr, "deltanet_wh: state size %d must be a power of two (Walsh-Hadamard)\n", D);
        return;
    }
    const int r = Vh / KVH;
    constexpr float rms_eps = 1e-6f;
    (void)r;

    // 1. Causal depthwise conv1d on the QKV (followed by SiLU), in place.
    //    Applied in real space, before the WH rotation: the conv weights
    //    are not rotated.
    causal_conv1d_step(qkv_proj, p.conv_w, conv_state, qkv_proj,
                       p.conv_dim, p.conv_kernel);
    trace("dn_wh_conv_silu", -1, qkv_proj, p.conv_dim);

    // 2. L2-normalize Q and K per k-head, scale Q by 1/sqrt(D). Both are
    //    WH-invariant, so doing them in real space matches the reference.
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(D));
    for (int h = 0; h < KVH; h++) {
        float* q_h = qkv_proj + h * D;
        l2norm_inplace(q_h, D);
        for (int i = 0; i < D; i++) q_h[i] *= q_scale;
        l2norm_inplace(qkv_proj + KD + h * D, D);
    }
    float* v_base = qkv_proj + 2 * KD;

    // 3. WH-rotate q and k per k-head, v per v-head.
    for (int h = 0; h < KVH; h++) {
        walsh_hadamard(qkv_proj + h * D, D);
        walsh_hadamard(qkv_proj + KD + h * D, D);
    }
    for (int hv = 0; hv < Vh; hv++) {
        walsh_hadamard(v_base + hv * D, D);
    }

    // 4. State update in WH space, per v-head (q/k from the parent
    //    k-head). Identical math to the real-space path on rotated data.
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int hv = 0; hv < Vh; hv++) {
        // Tiled v-head order: see deltanet_token.
        const int kh = hv % KVH;
        const float* q_tilde = qkv_proj + kh * D;
        const float* k_tilde = qkv_proj + KD + kh * D;
        const float* v_tilde = v_base + hv * D;
        float* S = recurrent_wh + hv * D * D;

        float A     = p.A[hv];
        float a_t   = softplus_f(a_proj[hv] + p.dt_bias[hv]);
        float g_d   = std::exp(A * a_t);
        float beta  = sigmoid_f(b_proj[hv]);

        float retrieved_tilde[128];
        for (int j = 0; j < D; j++) retrieved_tilde[j] = 0.0f;
        for (int i = 0; i < D; i++) {
            float* Si = S + i * D;
            float k_i = k_tilde[i];
            for (int j = 0; j < D; j++) {
                Si[j] *= g_d;
                retrieved_tilde[j] += Si[j] * k_i;
            }
        }

        float delta_tilde[128];
        for (int j = 0; j < D; j++) delta_tilde[j] = beta * (v_tilde[j] - retrieved_tilde[j]);

        float* o_g = o_raw + hv * D;
        for (int j = 0; j < D; j++) o_g[j] = 0.0f;
        for (int i = 0; i < D; i++) {
            float* Si = S + i * D;
            float k_i = k_tilde[i];
            float q_i = q_tilde[i];
            for (int j = 0; j < D; j++) {
                Si[j] += k_i * delta_tilde[j];
                o_g[j] += Si[j] * q_i;
            }
        }
    }

    // 5. Inverse-WH the output per v-head (H is orthonormal and
    //    self-inverse).
    for (int hv = 0; hv < Vh; hv++) {
        walsh_hadamard(o_raw + hv * D, D);
    }

    // 6. Gated RMSNorm, Qwen3-Next order: normalize first, apply the
    //    per-head weight (length D), THEN multiply by silu(z).
    for (int hv = 0; hv < Vh; hv++) {
        const float* o_g = o_raw + hv * D;
        const float* z_g = p.gate_fused
            ? qkv_proj + 2 * KD + p.inner + hv * D
            : gate_proj + hv * D;
        float* on_g = o_normed + hv * D;
        float ms = 0.0f;
        for (int i = 0; i < D; i++) ms += o_g[i] * o_g[i];
        float inv = 1.0f / std::sqrt(ms / D + rms_eps);
        for (int i = 0; i < D; i++) {
            on_g[i] = o_g[i] * inv * p.ssm_norm[i] * silu_f(z_g[i]);
        }
    }
    trace("dn_wh_o_raw", -1, o_raw, p.inner);
    trace("dn_wh_o_normed", -1, o_normed, p.inner);
}

} // namespace Laplace
