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
    const int G     = p.G;
    const int D     = p.D;
    if (D > 128) {
        fprintf(stderr, "deltanet: state size %d exceeds the supported 128\n", D);
        return;
    }
    const int inner = p.inner;
    constexpr float rms_eps = 1e-6f;

    // 1. Causal depthwise conv1d on the QKV (followed by SiLU), in place.
    causal_conv1d_step(qkv_proj, p.conv_w, conv_state, qkv_proj,
                       p.conv_dim, p.conv_kernel);
    trace("dn_conv_silu", -1, qkv_proj, p.conv_dim);

    // 2. L2-normalize Q and K per group (in place). V is left as-is.
    // The fused projection is de-interleaved: [all Q | all K | all V],
    // group-major within each section.
    // Q is then scaled by 1/sqrt(D). This does NOT cancel in the gated
    // RMSNorm: near-zero outputs are eps-dominated there, so the norm is
    // not scale-invariant and the reference's scaling must be reproduced.
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(D));
    for (int g = 0; g < G; g++) {
        float* q_g = qkv_proj + 0 * inner + g * D;
        l2norm_inplace(q_g, D);
        for (int i = 0; i < D; i++) q_g[i] *= q_scale;
        l2norm_inplace(qkv_proj + 1 * inner + g * D, D);
    }

    // 3+4. Per-group state update. Two fused row-major passes over S (rows
    // are contiguous and the j-loops vectorize):
    //   pass 1: S_i *= g; retrieved += S_i * k[i]
    //   pass 2: S_i += k[i] * delta; o += S_i * q[i]
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int g = 0; g < G; g++) {
        const float* q_g = qkv_proj + 0 * inner + g * D;
        const float* k_g = qkv_proj + 1 * inner + g * D;
        const float* v_g = qkv_proj + 2 * inner + g * D;
        float* S = recurrent + g * D * D;

        float A     = p.A[g];
        float a_t   = softplus_f(a_proj[g] + p.dt_bias[g]);
        float g_d   = std::exp(A * a_t);
        float beta  = sigmoid_f(b_proj[g]);

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

        float* o_g = o_raw + g * D;
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
    // apply the per-group weight (length D, shared across groups), THEN
    // multiply by silu(z).
    for (int g = 0; g < G; g++) {
        const float* o_g  = o_raw     + g * D;
        const float* z_g  = gate_proj + g * D;
        float*       on_g = o_normed  + g * D;
        float ms = 0.0f;
        for (int i = 0; i < D; i++) ms += o_g[i] * o_g[i];
        float inv = 1.0f / std::sqrt(ms / D + rms_eps);
        for (int i = 0; i < D; i++) {
            on_g[i] = o_g[i] * inv * p.ssm_norm[i] * silu_f(z_g[i]);
        }
    }
    trace("dn_o_raw", -1, o_raw, G * D);
    trace("dn_o_normed", -1, o_normed, G * D);
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
    const int G     = p.G;
    const int D     = p.D;
    if (D > 128) {
        fprintf(stderr, "deltanet_wh: state size %d exceeds the supported 128\n", D);
        return;
    }
    if (!is_power_of_two(D)) {
        fprintf(stderr, "deltanet_wh: state size %d must be a power of two (Walsh-Hadamard)\n", D);
        return;
    }
    const int inner = p.inner;
    constexpr float rms_eps = 1e-6f;

    // 1. Causal depthwise conv1d on the QKV (followed by SiLU), in place.
    //    The conv1d is applied in real space, before the WH rotation; the
    //    conv weights are not rotated, so this is the right place to apply
    //    them.
    causal_conv1d_step(qkv_proj, p.conv_w, conv_state, qkv_proj,
                       p.conv_dim, p.conv_kernel);
    trace("dn_wh_conv_silu", -1, qkv_proj, p.conv_dim);

    // 2. L2-normalize Q and K per group, scale Q by 1/sqrt(D).  These
    //    operations are WH-invariant (L2 norm is rotation-invariant; the
    //    scalar scale commutes with WH), so doing them in real space gives
    //    the same result as doing them in WH space.
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(D));
    for (int g = 0; g < G; g++) {
        float* q_g = qkv_proj + 0 * inner + g * D;
        l2norm_inplace(q_g, D);
        for (int i = 0; i < D; i++) q_g[i] *= q_scale;
        l2norm_inplace(qkv_proj + 1 * inner + g * D, D);
    }

    // 3. WH-rotate q, k, v per group: q̃ = H·q, k̃ = H·k, ṽ = H·v.
    //    This is the basis change that makes the state update operate in the
    //    rotated basis.  Done in place over the qkv_proj slots.
    for (int g = 0; g < G; g++) {
        walsh_hadamard(qkv_proj + 0 * inner + g * D, D);
        walsh_hadamard(qkv_proj + 1 * inner + g * D, D);
        walsh_hadamard(qkv_proj + 2 * inner + g * D, D);
    }

    // 4. State update in WH space.  The structure mirrors the real-space
    //    update: scale-decay the state, compute the retrieved vector in WH
    //    space (S̃^T·k̃), form the delta in WH space, fold it back into the
    //    state via the outer product, and produce õ = S̃^T·q̃.
    //
    //    The only structural difference from the real-space path is that
    //    q, k, v are now the WH-domain versions; the math is identical.
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int g = 0; g < G; g++) {
        const float* q_tilde = qkv_proj + 0 * inner + g * D;
        const float* k_tilde = qkv_proj + 1 * inner + g * D;
        const float* v_tilde = qkv_proj + 2 * inner + g * D;
        float* S = recurrent_wh + g * D * D;

        float A     = p.A[g];
        float a_t   = softplus_f(a_proj[g] + p.dt_bias[g]);
        float g_d   = std::exp(A * a_t);
        float beta  = sigmoid_f(b_proj[g]);

        // Pass 1: S̃ *= g; retrieved_tilde += S̃[i,j] * k̃[i]
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

        // delta_tilde = beta * (ṽ - retrieved_tilde)
        float delta_tilde[128];
        for (int j = 0; j < D; j++) delta_tilde[j] = beta * (v_tilde[j] - retrieved_tilde[j]);

        // Pass 2: S̃ += k̃[i] * delta_tilde[j]; õ += S̃[i,j] * q̃[i]
        float* o_g = o_raw + g * D;
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

    // 5. Inverse-WH the output (õ in WH space -> o in real space).  H is
    //    orthonormal and self-inverse, so walsh_hadamard does the inverse.
    for (int g = 0; g < G; g++) {
        walsh_hadamard(o_raw + g * D, D);
    }

    // 6. Gated RMSNorm, Qwen3-Next order: normalize first, apply the
    //    per-group weight (length D, shared across groups), THEN multiply
    //    by silu(z).  Operates in real space, identical to the real-space
    //    path.
    for (int g = 0; g < G; g++) {
        const float* o_g  = o_raw     + g * D;
        const float* z_g  = gate_proj + g * D;
        float*       on_g = o_normed  + g * D;
        float ms = 0.0f;
        for (int i = 0; i < D; i++) ms += o_g[i] * o_g[i];
        float inv = 1.0f / std::sqrt(ms / D + rms_eps);
        for (int i = 0; i < D; i++) {
            on_g[i] = o_g[i] * inv * p.ssm_norm[i] * silu_f(z_g[i]);
        }
    }
    trace("dn_wh_o_raw", -1, o_raw, G * D);
    trace("dn_wh_o_normed", -1, o_normed, G * D);
}

} // namespace Laplace
