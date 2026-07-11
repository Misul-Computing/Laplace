// test_arch.cpp - minimal tests for architecture abstraction and shared ops
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "arch.h"
#include "ops.h"
#include "ssm.h"
#include "test_util.h"

using namespace Laplace;

namespace {

void test_rmsnorm() {
    const int n = 4;
    float x[n] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[n] = {1.0f, 1.0f, 1.0f, 1.0f};
    float y[n];
    ops::rmsnorm(x, w, y, n, 1e-6f);
    float ms = 0.0f;
    for (int i = 0; i < n; i++) ms += y[i] * y[i];
    CHECK_MSG(almost_equal(ms, float(n), 1e-5f, 1e-6f),
              "rmsnorm did not preserve unit norm: %f", std::sqrt(ms));
}

void test_rmsnorm_rows() {
    const int rows = 3, cols = 4;
    float x[rows * cols] = {
        1, 2, 3, 4,
        0, 0, 0, 0,
        -1, -2, -3, -4,
    };
    float w[cols] = {1, 1, 1, 1};
    float y[rows * cols];
    ops::rmsnorm_rows(x, w, y, rows, cols, 1e-6f);
    for (int r = 0; r < rows; r++) {
        float ms = 0.0f;
        for (int i = 0; i < cols; i++) ms += y[r * cols + i] * y[r * cols + i];
        if (r == 1) {
            // Zero input stays zero.
            CHECK_MSG(almost_equal(ms, 0.0f, 1e-5f, 1e-6f),
                      "rmsnorm_rows zero row ms=%f", ms);
        } else {
            CHECK_MSG(almost_equal(ms, float(cols), 1e-5f, 1e-6f),
                      "rmsnorm_rows row %d ms=%f", r, ms);
        }
    }
}

void test_rope_roundtrip() {
    // RoPE is a rotation; applying it twice with cos/sin and cos/-sin inverts.
    const int n_heads = 2;
    const int head_dim = 8;
    const int rope_pairs = head_dim / 2;
    std::vector<float> cos(rope_pairs), sin(rope_pairs);
    for (int p = 0; p < rope_pairs; p++) {
        double angle = (p + 1) * 0.1;
        cos[p] = static_cast<float>(std::cos(angle));
        sin[p] = static_cast<float>(std::sin(angle));
    }
    float orig[n_heads * head_dim];
    for (int i = 0; i < n_heads * head_dim; i++) orig[i] = static_cast<float>(i + 1);

    float x[n_heads * head_dim];
    std::memcpy(x, orig, sizeof(x));
    ops::rope_apply(x, n_heads, head_dim, rope_pairs, cos.data(), sin.data());
    for (int p = 0; p < rope_pairs; p++) sin[p] = -sin[p];
    ops::rope_apply(x, n_heads, head_dim, rope_pairs, cos.data(), sin.data());

    int bad = 0;
    for (int i = 0; i < n_heads * head_dim; i++) {
        if (!almost_equal(x[i], orig[i], 1e-5f, 1e-6f)) bad++;
    }
    CHECK_MSG(bad == 0, "rope round-trip mismatch %d/%d", bad, n_heads * head_dim);
}

void test_factory() {
    auto a1 = create_arch("qwen3next");
    CHECK(a1 != nullptr);
    if (a1) CHECK(std::string(a1->name()) == "qwen3next");

    auto a2 = create_arch("qwen35");
    CHECK(a2 != nullptr);
    if (a2) CHECK(std::string(a2->name()) == "qwen3next");

    auto a3 = create_arch("llama");
    CHECK(a3 != nullptr);
    if (a3) CHECK(std::string(a3->name()) == "llama");

    auto a4 = create_arch("qwen2");
    CHECK(a4 != nullptr);
    if (a4) CHECK(std::string(a4->name()) == "llama");

    auto a5 = create_arch("qwen3");
    CHECK(a5 != nullptr);
    if (a5) CHECK(std::string(a5->name()) == "llama");

    auto a6 = create_arch("phi3");
    CHECK(a6 != nullptr);
    if (a6) CHECK(std::string(a6->name()) == "phi3");

    auto a7 = create_arch("unknown");
    CHECK(a7 == nullptr);
}

// Phi3's RMSNorm uses the gain+1 trick: y = x / sqrt(mean(x^2) + eps)
// * (1 + w).  The standard rmsnorm doesn't match this; verify our
// rmsnorm_phi3 against a hand-rolled reference.
void test_rmsnorm_phi3() {
    const int n = 8;
    float x[n]    = { 0.5f, -1.0f,  2.0f,  0.0f,  3.0f, -2.5f,  0.1f,  4.0f };
    float w[n]    = { 0.0f,  0.1f, -0.2f,  0.0f,  0.3f, -0.4f,  0.0f,  0.5f };
    float y_phi3[n];
    ops::rmsnorm_phi3(x, w, y_phi3, n, 1e-5f);

    float ms = 0.0f;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    float inv = 1.0f / std::sqrt(ms / n + 1e-5f);
    int bad = 0;
    for (int i = 0; i < n; i++) {
        float ref = x[i] * inv * (1.0f + w[i]);
        if (!almost_equal(y_phi3[i], ref, 1e-5f, 1e-6f)) bad++;
    }
    CHECK_MSG(bad == 0, "rmsnorm_phi3 mismatch %d/%d", bad, n);
}

// WH-domain DeltaNet state parity: deltanet_token and deltanet_token_wh
// must produce identical outputs (modulo FP rounding) when the WH-domain
// state is initialized to H·S·H^T.  This is the equivalence proof that
// the WH-domain transform preserves the delta-rule update mathematically;
// the per-element order may shift, so we tolerate a small absolute
// difference (a few ULPs after the FMA chain).
void test_deltanet_wh_parity() {
    const int G = 2, D = 8, inner = 16;
    const int conv_kernel = 4, conv_dim = 3 * inner;
    const int hist = conv_kernel - 1;
    const int state_n = G * D * D;
    const int trials = 8;

    std::mt19937 rng(31415);
    std::normal_distribution<float> N(0.0f, 1.0f);

    DeltaNetParams p;
    p.G = G; p.D = D; p.inner = inner;
    p.conv_kernel = conv_kernel; p.conv_dim = conv_dim;

    // Initialize learnable per-group tensors.
    std::vector<float> A(G), dt_bias(G), ssm_norm(D), conv_w(conv_dim * conv_kernel);
    for (int i = 0; i < G; i++)  { A[i] = -0.5f * std::abs(N(rng)); dt_bias[i] = N(rng) * 0.1f; }
    for (int i = 0; i < D; i++)  { ssm_norm[i] = 1.0f + 0.1f * N(rng); }
    for (auto& w : conv_w)      { w = N(rng) * 0.1f; }
    p.A        = A.data();
    p.dt_bias  = dt_bias.data();
    p.ssm_norm = ssm_norm.data();
    p.conv_w   = conv_w.data();

    for (int t = 0; t < trials; t++) {
        std::vector<float> S(state_n);
        for (auto& v : S) v = N(rng) * 0.5f;

        std::vector<float> qkv(conv_dim);
        for (auto& v : qkv) v = N(rng) * 0.5f;
        std::vector<float> gate(inner);
        for (auto& v : gate) v = N(rng) * 0.5f;
        std::vector<float> b(G);
        for (auto& v : b) v = N(rng) * 0.5f;
        std::vector<float> a(G);
        for (auto& v : a) v = N(rng) * 0.5f;

        // Convert S to WH-domain: S_tilde = H · S · H^T.
        // Two-pass: first column-transform (S · H^T), then row-transform.
        std::vector<float> S_tilde(state_n, 0.0f);
        std::vector<float> S_temp(state_n, 0.0f);
        for (int g = 0; g < G; g++) {
            std::vector<float> col(D);
            // Pass 1: column transform.
            for (int b = 0; b < D; b++) {
                for (int i = 0; i < D; i++) col[i] = S[g * D * D + i * D + b];
                walsh_hadamard(col.data(), D);
                for (int i = 0; i < D; i++) S_temp[g * D * D + i * D + b] = col[i];
            }
            // Pass 2: row transform.
            for (int i = 0; i < D; i++) {
                std::memcpy(col.data(), S_temp.data() + g * D * D + i * D, D * sizeof(float));
                walsh_hadamard(col.data(), D);
                std::memcpy(S_tilde.data() + g * D * D + i * D, col.data(), D * sizeof(float));
            }
        }

        // Real-space path.
        std::vector<float> conv_state_fp32(conv_dim * hist, 0.0f);
        std::vector<float> S_fp32 = S;
        std::vector<float> qkv_fp32 = qkv;
        std::vector<float> o_raw_fp32(G * D, 0.0f);
        std::vector<float> o_normed_fp32(inner, 0.0f);
        deltanet_token(p,
                       conv_state_fp32.data(),
                       S_fp32.data(),
                       qkv_fp32.data(),
                       gate.data(),
                       b.data(),
                       a.data(),
                       o_raw_fp32.data(),
                       o_normed_fp32.data());

        // WH-domain path.  The conv_state is shared (real-space conv1d is
        // applied identically to both paths), so we use a fresh copy.
        std::vector<float> conv_state_wh(conv_dim * hist, 0.0f);
        std::vector<float> S_wh = S_tilde;
        std::vector<float> qkv_wh = qkv;
        std::vector<float> o_raw_wh(G * D, 0.0f);
        std::vector<float> o_normed_wh(inner, 0.0f);
        deltanet_token_wh(p,
                          conv_state_wh.data(),
                          S_wh.data(),
                          qkv_wh.data(),
                          gate.data(),
                          b.data(),
                          a.data(),
                          o_raw_wh.data(),
                          o_normed_wh.data());

        int bad_o = 0;
        double max_err_o = 0.0;
        for (int i = 0; i < inner; i++) {
            double d = std::fabs(o_normed_fp32[i] - o_normed_wh[i]);
            if (d > max_err_o) max_err_o = d;
            if (d > 1e-3) bad_o++;
        }
        CHECK_MSG(bad_o == 0,
                  "WH-domain DeltaNet o_normed parity trial=%d: %d/%d disagree, max_err=%.4e",
                  t, bad_o, inner, max_err_o);

        // The post-update WH-domain state, inverse-rotated, should equal
        // the post-update real-space state.
        std::vector<float> S_wh_real(state_n, 0.0f);
        for (int g = 0; g < G; g++) {
            std::vector<float> tmp(D * D);
            std::vector<float> tmp2(D * D);
            for (int i = 0; i < D; i++) {
                for (int j = 0; j < D; j++) tmp[i * D + j] = S_wh[g * D * D + i * D + j];
            }
            for (int b = 0; b < D; b++) {
                std::vector<float> col(D);
                for (int i = 0; i < D; i++) col[i] = tmp[i * D + b];
                walsh_hadamard(col.data(), D);
                for (int i = 0; i < D; i++) tmp2[i * D + b] = col[i];
            }
            for (int i = 0; i < D; i++) {
                std::vector<float> col(D);
                for (int j = 0; j < D; j++) col[j] = tmp2[i * D + j];
                walsh_hadamard(col.data(), D);
                for (int j = 0; j < D; j++) S_wh_real[g * D * D + i * D + j] = col[j];
            }
        }
        int bad_s = 0;
        double max_err_s = 0.0;
        for (int i = 0; i < state_n; i++) {
            double d = std::fabs(S_fp32[i] - S_wh_real[i]);
            if (d > max_err_s) max_err_s = d;
            if (d > 1e-3) bad_s++;
        }
        CHECK_MSG(bad_s == 0,
                  "WH-domain DeltaNet state parity trial=%d: %d/%d disagree, max_err=%.4e",
                  t, bad_s, state_n, max_err_s);
    }
}

} // namespace

int main() {
    test_rmsnorm();
    test_rmsnorm_rows();
    test_rmsnorm_phi3();
    test_rope_roundtrip();
    test_factory();
    test_deltanet_wh_parity();
    return test_summary("test_arch");
}
