// test_matmul - kernel correctness against an independent GGUF-layout reference.
//
// Contract under test:
//   GGUF tensors are row-major with dims[0] innermost. A weight W with
//   dims = [K, N] is stored as N contiguous rows of K elements, so
//   element (k, j) lives at j*K + k, and quantized blocks of row j are
//   contiguous at block index j*(K/QK) + b.
//
//   matmul_row(x, W, y, K, N) must compute y[j] = sum_k x[k] * W[k, j].

#include <array>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "fp16.h"
#include "column_grouped_affine_uint2_skip.h"
#include "kernels.h"
#include "matmul.h"
#include "ops.h"
#include "tensor.h"

#include "quant_ref.h"
#include "test_util.h"

using namespace Laplace;

namespace {

#include "../src/iq2_xxs_tables.inc"
#include "../src/iq1_s_tables.inc"

Tensor make_tensor(GGMLType type, int K, int N, const uint8_t* data) {
    Tensor t;
    t.name = "test";
    t.type = type;
    t.n_dims = 2;
    t.dims[0] = static_cast<uint64_t>(K);
    t.dims[1] = static_cast<uint64_t>(N);
    t.data = data;
    return t;
}

Tensor make_expert_tensor(GGMLType type, int K, int N, int experts,
                          const uint8_t* data) {
    Tensor t = make_tensor(type, K, N, data);
    t.n_dims = 3;
    t.dims[2] = static_cast<uint64_t>(experts);
    return t;
}

// Reference matmul on a fully dequantized row-major [N rows x K] weight.
void ref_matmul(const float* x, const float* w_rows, float* y, int K, int N) {
    for (int j = 0; j < N; j++) {
        double acc = 0.0;
        for (int k = 0; k < K; k++) acc += static_cast<double>(x[k]) * w_rows[static_cast<size_t>(j) * K + k];
        y[j] = static_cast<float>(acc);
    }
}

std::vector<float> random_x(int K, XorShift32& rng) {
    std::vector<float> x(K);
    for (auto& v : x) v = rng.next_float();
    return x;
}

uint16_t small_fp16(XorShift32& rng) {
    // d in roughly [0.005, 0.07]: keeps accumulations well-conditioned.
    float d = 0.005f + 0.065f * (rng.next_float() * 0.5f + 0.5f);
    return fp32_to_fp16(d);
}

void check_against_ref(const char* what, GGMLType type, int K, int N,
                       const uint8_t* wdata, const std::vector<float>& w_ref_rows) {
    XorShift32 rng(7u + static_cast<uint32_t>(type));
    std::vector<float> x = random_x(K, rng);
    std::vector<float> y(N, 0.0f), y_ref(N, 0.0f);

    Tensor t = make_tensor(type, K, N, wdata);
    matmul_row(x.data(), t, y.data(), K, N);

    // Quantized-weight matmuls MAY quantize activations to int8 blocks first
    // (the SIMD path does; the scalar fallback computes with exact float
    // activations). Accept a match against either reference:
    // the exact product, or the product with x replaced by its Q8 round-trip
    // (spec: per 32 elements, d = amax/127, q = round(x/d)).
    bool int8_path = type == GGMLType::Q8_0 || type == GGMLType::Q4_0 ||
                     type == GGMLType::Q4_K || type == GGMLType::Q6_K;
    ref_matmul(x.data(), w_ref_rows.data(), y_ref.data(), K, N);
    std::vector<float> y_ref_q = y_ref;
    if (int8_path) {
        std::vector<float> x_q = x;
        for (int b = 0; b < K / 32; b++) {
            float* xb = x_q.data() + b * 32;
            float amax = 0.0f;
            for (int l = 0; l < 32; l++) amax = std::fmax(amax, std::fabs(xb[l]));
            float d = amax / 127.0f;
            float id = d > 0.0f ? 1.0f / d : 0.0f;
            for (int l = 0; l < 32; l++) xb[l] = d * std::lroundf(xb[l] * id);
        }
        ref_matmul(x_q.data(), w_ref_rows.data(), y_ref_q.data(), K, N);
    }

    float rel = 1e-3f;
    float abs_tol = int8_path ? 1e-3f : 1e-4f;

    int bad = 0;
    for (int j = 0; j < N; j++) {
        if (!almost_equal(y[j], y_ref[j], rel, abs_tol) &&
            !almost_equal(y[j], y_ref_q[j], rel, abs_tol)) bad++;
    }
    CHECK_MSG(bad == 0, "%s: %d/%d outputs match neither reference, e.g. y[0]=%g exact=%g q8=%g",
              what, bad, N, y[0], y_ref[0], y_ref_q[0]);

    // dequantize() must reproduce the reference rows too.
    std::vector<float> deq(static_cast<size_t>(K) * N, 0.0f);
    dequantize(t, deq.data(), K * N);
    bad = 0;
    for (size_t i = 0; i < deq.size(); i++) {
        if (!almost_equal(deq[i], w_ref_rows[i], 1e-5f, 1e-6f)) bad++;
    }
    CHECK_MSG(bad == 0, "%s: dequantize %d/%zu mismatch, e.g. deq[0]=%g ref=%g",
              what, bad, deq.size(), deq[0], w_ref_rows[0]);
}

void test_f32(int K, int N) {
    XorShift32 rng(11);
    std::vector<float> w(static_cast<size_t>(K) * N);
    for (auto& v : w) v = rng.next_float();
    check_against_ref("f32", GGMLType::F32, K, N,
                      reinterpret_cast<const uint8_t*>(w.data()), w);
}

void test_f16(int K, int N) {
    XorShift32 rng(13);
    std::vector<uint16_t> w(static_cast<size_t>(K) * N);
    std::vector<float> w_ref(w.size());
    for (size_t i = 0; i < w.size(); i++) {
        w[i] = fp32_to_fp16(rng.next_float());
        w_ref[i] = fp16_to_fp32(w[i]);
    }
    check_against_ref("f16", GGMLType::F16, K, N,
                      reinterpret_cast<const uint8_t*>(w.data()), w_ref);
}

void test_bf16(int K, int N) {
    XorShift32 rng(15);
    std::vector<uint16_t> w(static_cast<size_t>(K) * N);
    std::vector<float> w_ref(w.size());
    for (size_t i = 0; i < w.size(); i++) {
        w[i] = fp32_to_bf16(rng.next_float());
        w_ref[i] = bf16_to_fp32(w[i]);
    }
    check_against_ref("bf16", GGMLType::BF16, K, N,
                      reinterpret_cast<const uint8_t*>(w.data()), w_ref);
}

void test_q4_0(int K, int N) {
    XorShift32 rng(17);
    int blocks_per_row = K / 32;
    std::vector<quant_ref::block_q4_0> w(static_cast<size_t>(blocks_per_row) * N);
    for (auto& b : w) {
        b.d = small_fp16(rng);
        for (auto& q : b.qs) q = rng.next_byte();
    }
    std::vector<float> w_ref(static_cast<size_t>(K) * N);
    for (int j = 0; j < N; j++)
        for (int b = 0; b < blocks_per_row; b++)
            quant_ref::dequant_q4_0(&w[static_cast<size_t>(j) * blocks_per_row + b],
                                   w_ref.data() + static_cast<size_t>(j) * K + b * 32);
    check_against_ref("q4_0", GGMLType::Q4_0, K, N,
                      reinterpret_cast<const uint8_t*>(w.data()), w_ref);
}

void test_q8_0(int K, int N) {
    XorShift32 rng(19);
    int blocks_per_row = K / 32;
    std::vector<quant_ref::block_q8_0> w(static_cast<size_t>(blocks_per_row) * N);
    for (auto& b : w) {
        b.d = small_fp16(rng);
        for (auto& q : b.qs) q = static_cast<int8_t>(rng.next_byte());
    }
    std::vector<float> w_ref(static_cast<size_t>(K) * N);
    for (int j = 0; j < N; j++)
        for (int b = 0; b < blocks_per_row; b++)
            quant_ref::dequant_q8_0(&w[static_cast<size_t>(j) * blocks_per_row + b],
                                   w_ref.data() + static_cast<size_t>(j) * K + b * 32);
    check_against_ref("q8_0", GGMLType::Q8_0, K, N,
                      reinterpret_cast<const uint8_t*>(w.data()), w_ref);
}

void test_q4_k(int K, int N) {
    XorShift32 rng(23);
    int blocks_per_row = K / 256;
    std::vector<quant_ref::block_q4_K> w(static_cast<size_t>(blocks_per_row) * N);
    for (auto& b : w) {
        b.d = small_fp16(rng);
        b.dmin = small_fp16(rng);
        for (auto& s : b.scales) s = rng.next_byte();
        for (auto& q : b.qs) q = rng.next_byte();
    }
    std::vector<float> w_ref(static_cast<size_t>(K) * N);
    for (int j = 0; j < N; j++)
        for (int b = 0; b < blocks_per_row; b++)
            quant_ref::dequant_q4_K(&w[static_cast<size_t>(j) * blocks_per_row + b],
                                   w_ref.data() + static_cast<size_t>(j) * K + b * 256);
    check_against_ref("q4_K", GGMLType::Q4_K, K, N,
                      reinterpret_cast<const uint8_t*>(w.data()), w_ref);
}

void test_q6_k(int K, int N) {
    XorShift32 rng(29);
    int blocks_per_row = K / 256;
    std::vector<quant_ref::block_q6_K> w(static_cast<size_t>(blocks_per_row) * N);
    for (auto& b : w) {
        for (auto& q : b.ql) q = rng.next_byte();
        for (auto& q : b.qh) q = rng.next_byte();
        for (auto& s : b.scales) s = static_cast<int8_t>(rng.next() % 17) - 8;
        b.d = small_fp16(rng);
    }
    std::vector<float> w_ref(static_cast<size_t>(K) * N);
    for (int j = 0; j < N; j++)
        for (int b = 0; b < blocks_per_row; b++)
            quant_ref::dequant_q6_K(&w[static_cast<size_t>(j) * blocks_per_row + b],
                                   w_ref.data() + static_cast<size_t>(j) * K + b * 256);
    check_against_ref("q6_K", GGMLType::Q6_K, K, N,
                      reinterpret_cast<const uint8_t*>(w.data()), w_ref);
}

// matmul_rows must produce, for every row m, what matmul_row gives for that
// row alone. The 4-row micro-kernels accumulate in a different order, so
// results may differ by float reassociation but nothing more.
void test_batched_rows() {
    const int M = 7, K = 512, N = 64;   // exercises the 4-row path + remainder
    for (GGMLType type : {GGMLType::F32, GGMLType::Q8_0, GGMLType::Q4_K, GGMLType::Q6_K}) {
        XorShift32 rng(41u + static_cast<uint32_t>(type));
        size_t rb = static_cast<size_t>(K) / elements_per_block(type) * bytes_per_block(type);
        std::vector<uint8_t> wdata(rb * N);
        for (auto& v : wdata) v = rng.next_byte();
        // For float weights, fill with valid floats instead of random bytes.
        if (type == GGMLType::F32) {
            float* p = reinterpret_cast<float*>(wdata.data());
            for (size_t i = 0; i < wdata.size() / 4; i++) p[i] = rng.next_float();
        }
        Tensor t = make_tensor(type, K, N, wdata.data());

        std::vector<float> x(static_cast<size_t>(M) * K);
        for (auto& v : x) v = rng.next_float();

        std::vector<float> y_batch(static_cast<size_t>(M) * N, 0.0f);
        std::vector<float> y_single(static_cast<size_t>(M) * N, 0.0f);
        matmul_rows(x.data(), t, y_batch.data(), M, K, N);
        for (int m = 0; m < M; m++) {
            matmul_row(x.data() + static_cast<size_t>(m) * K, t,
                       y_single.data() + static_cast<size_t>(m) * N, K, N);
        }
        int bad = 0;
        for (size_t i = 0; i < y_batch.size(); i++) {
            float a = y_batch[i], b = y_single[i];
            if (std::isnan(a) && std::isnan(b)) continue;  // random scales may be NaN
            if (!almost_equal(a, b, 1e-4f, 1e-4f)) bad++;
        }
        CHECK_MSG(bad == 0, "%s batched: %d/%zu values differ from single-row",
                  type_name(type), bad, y_batch.size());
    }
}

void test_column_grouped_u2_skip_is_cpu_rejected() {
    std::array<uint8_t, 256> storage{};
    Tensor t = make_tensor(GGMLType::COLUMN_GROUPED_AFFINE_U2_SKIP_256,
                           256, 1, storage.data());
    float x[256]{};
    float y[256];
    std::fill(std::begin(y), std::end(y), 17.0f);
    const int expert = 0;
    const float route = 1.0f;
    float hidden[256]{};
    float output[256];
    std::fill(std::begin(output), std::end(output), 23.0f);

    auto expect_cpu_rejection = [](const char* what, auto&& call) {
        bool rejected = false;
        try {
            call();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        CHECK_MSG(rejected, "%s must reject CPU execution", what);
    };

    expect_cpu_rejection("matmul_row", [&] {
        matmul_row(x, t, y, 256, 1);
    });
    expect_cpu_rejection("matmul_rows", [&] {
        matmul_rows(x, t, y, 1, 256, 1);
    });
    expect_cpu_rejection("matmul_lm_head", [&] {
        matmul_lm_head(x, t, y, 1, 256, 1);
    });
    MatmulBatchSpec spec{x, &t, y, 256, 1};
    expect_cpu_rejection("matmul_gemm_batch", [&] {
        matmul_gemm_batch(&spec, 1);
    });
    expect_cpu_rejection("fused_moe_gemm_idx", [&] {
        fused_moe_gemm_idx(x, t, y, &expert, 1, 256, 1);
    });
    expect_cpu_rejection("fused_moe_gemm_multi", [&] {
        fused_moe_gemm_multi(x, t, y, &expert, 1, 256, 1);
    });
    expect_cpu_rejection("fused_moe_gate_up_geglu", [&] {
        fused_moe_gate_up_geglu(x, t, hidden, &expert, 1, 256, 1);
    });
    expect_cpu_rejection("fused_moe_down_accumulate", [&] {
        fused_moe_down_accumulate(hidden, t, &expert, &route, 1, 256, 1,
                                  output);
    });
    expect_cpu_rejection("dequantize", [&] {
        dequantize(t, y, 256);
    });
    CHECK(std::all_of(std::begin(y), std::end(y), [](float value) {
        return value == 17.0f;
    }));
    CHECK(std::all_of(std::begin(output), std::end(output), [](float value) {
        return value == 23.0f;
    }));

    // The SIMD dispatch boundary must reject the type before deriving a
    // row stride from its deliberately variable-size format.
    if (kernels::gemm_fn gemm = kernels::get_simd_gemm())
        CHECK(!gemm(x, storage.data(), t.type, y, 1, 256, 1));
    if (kernels::moe_gemv_fn gemv = kernels::get_simd_moe_gemv())
        CHECK(!gemv(x, storage.data(), t.type, &expert, 1, y, 256, 1));
    if (kernels::moe_gemv_multi_fn gemv = kernels::get_simd_moe_gemv_multi())
        CHECK(!gemv(x, storage.data(), t.type, &expert, 1, y, 256, 1));
    if (kernels::moe_gate_up_fn gate_up = kernels::get_simd_moe_gate_up())
        CHECK(!gate_up(x, storage.data(), t.type, &expert, 1, hidden,
                       256, 1, nullptr));
    if (kernels::moe_down_fn down = kernels::get_simd_moe_down())
        CHECK(!down(hidden, storage.data(), t.type, &expert, &route, 1,
                    output, 256, 1, nullptr));
}

void test_routed_moe_fusion() {
    constexpr int experts = 5;
    constexpr int selected = 3;
    constexpr int input_dim = 64;
    constexpr int hidden_dim = 32;
    constexpr int output_dim = 48;
    const int expert_idx[selected] = {4, 1, 3};
    const float route_weight[selected] = {0.2f, -0.1f, 0.7f};

    XorShift32 rng(47);
    std::vector<float> x(input_dim);
    for (float& v : x) v = rng.next_float();

    std::vector<float> gate_up_w(static_cast<size_t>(experts) * 2 * hidden_dim * input_dim);
    for (float& v : gate_up_w) v = rng.next_float();
    Tensor gate_up = make_expert_tensor(
        GGMLType::F32, input_dim, 2 * hidden_dim, experts,
        reinterpret_cast<const uint8_t*>(gate_up_w.data()));

    std::vector<float> gate_up_ref(static_cast<size_t>(selected) * 2 * hidden_dim);
    std::vector<float> hidden_ref(static_cast<size_t>(selected) * hidden_dim);
    std::vector<float> hidden_fused(hidden_ref.size());
    fused_moe_gemm_idx(x.data(), gate_up, gate_up_ref.data(), expert_idx,
                       selected, input_dim, 2 * hidden_dim);
    for (int k = 0; k < selected; k++) {
        const float* gu = gate_up_ref.data() + static_cast<size_t>(k) * 2 * hidden_dim;
        ops::geglu(gu, gu + hidden_dim,
                   hidden_ref.data() + static_cast<size_t>(k) * hidden_dim,
                   hidden_dim);
    }
    CHECK(fused_moe_gate_up_geglu(x.data(), gate_up, hidden_fused.data(),
                                  expert_idx, selected, input_dim, hidden_dim));
    for (size_t i = 0; i < hidden_ref.size(); i++)
        CHECK(almost_equal(hidden_fused[i], hidden_ref[i], 1e-5f, 1e-5f));

    std::vector<float> down_w(static_cast<size_t>(experts) * output_dim * hidden_dim);
    for (float& v : down_w) v = rng.next_float();
    Tensor down = make_expert_tensor(
        GGMLType::F32, hidden_dim, output_dim, experts,
        reinterpret_cast<const uint8_t*>(down_w.data()));

    std::vector<float> expert_out(static_cast<size_t>(selected) * output_dim);
    std::vector<float> output_ref(output_dim, 0.0f);
    std::vector<float> output_fused(output_dim, 0.0f);
    fused_moe_gemm_multi(hidden_ref.data(), down, expert_out.data(),
                         expert_idx, selected, hidden_dim, output_dim);
    for (int k = 0; k < selected; k++)
        for (int j = 0; j < output_dim; j++)
            output_ref[j] += route_weight[k] *
                expert_out[static_cast<size_t>(k) * output_dim + j];

    CHECK(fused_moe_down_accumulate(hidden_ref.data(), down, expert_idx,
                                    route_weight, selected, hidden_dim,
                                    output_dim, output_fused.data()));
    for (int j = 0; j < output_dim; j++)
        CHECK(almost_equal(output_fused[j], output_ref[j], 1e-5f, 1e-5f));
}

std::vector<uint8_t> make_quant_expert_weights(GGMLType type, int K, int N,
                                                int experts, XorShift32& rng) {
    size_t blocks = static_cast<size_t>(experts) * N *
                    (K / elements_per_block(type));
    std::vector<uint8_t> data(blocks * bytes_per_block(type));
    if (type == GGMLType::Q4_0) {
        auto* w = reinterpret_cast<quant_ref::block_q4_0*>(data.data());
        for (size_t i = 0; i < blocks; i++) {
            w[i].d = small_fp16(rng);
            for (auto& q : w[i].qs) q = rng.next_byte();
        }
    } else if (type == GGMLType::Q8_0) {
        auto* w = reinterpret_cast<quant_ref::block_q8_0*>(data.data());
        for (size_t i = 0; i < blocks; i++) {
            w[i].d = small_fp16(rng);
            for (auto& q : w[i].qs) q = static_cast<int8_t>(rng.next_byte());
        }
    } else if (type == GGMLType::Q4_K) {
        auto* w = reinterpret_cast<quant_ref::block_q4_K*>(data.data());
        for (size_t i = 0; i < blocks; i++) {
            w[i].d = small_fp16(rng);
            w[i].dmin = small_fp16(rng);
            for (auto& s : w[i].scales) s = rng.next_byte();
            for (auto& q : w[i].qs) q = rng.next_byte();
        }
    } else if (type == GGMLType::Q6_K) {
        auto* w = reinterpret_cast<quant_ref::block_q6_K*>(data.data());
        for (size_t i = 0; i < blocks; i++) {
            for (auto& q : w[i].ql) q = rng.next_byte();
            for (auto& q : w[i].qh) q = rng.next_byte();
            for (auto& s : w[i].scales)
                s = static_cast<int8_t>(rng.next() % 17) - 8;
            w[i].d = small_fp16(rng);
        }
    }
    return data;
}

void test_routed_moe_quant_fusion(GGMLType type) {
    constexpr int experts = 4;
    constexpr int selected = 3;
    constexpr int input_dim = 512;
    constexpr int hidden_dim = 256;
    constexpr int output_dim = 256;
    const int expert_idx[selected] = {3, 0, 2};
    const float route_weight[selected] = {0.5f, 0.3f, 0.2f};

    XorShift32 rng(53u + static_cast<uint32_t>(type));
    std::vector<float> x(input_dim);
    for (float& v : x) v = rng.next_float();
    std::vector<uint8_t> gate_up_data =
        make_quant_expert_weights(type, input_dim, 2 * hidden_dim, experts, rng);
    Tensor gate_up = make_expert_tensor(type, input_dim, 2 * hidden_dim,
                                        experts, gate_up_data.data());

    std::vector<float> gate_up_ref(static_cast<size_t>(selected) * 2 * hidden_dim);
    std::vector<float> hidden_ref(static_cast<size_t>(selected) * hidden_dim);
    std::vector<float> hidden_fused(hidden_ref.size());
    fused_moe_gemm_idx(x.data(), gate_up, gate_up_ref.data(), expert_idx,
                       selected, input_dim, 2 * hidden_dim);
    for (int k = 0; k < selected; k++) {
        const float* gu = gate_up_ref.data() + static_cast<size_t>(k) * 2 * hidden_dim;
        ops::geglu(gu, gu + hidden_dim,
                   hidden_ref.data() + static_cast<size_t>(k) * hidden_dim,
                   hidden_dim);
    }
    CHECK(fused_moe_gate_up_geglu(x.data(), gate_up, hidden_fused.data(),
                                  expert_idx, selected, input_dim, hidden_dim));
    for (size_t i = 0; i < hidden_ref.size(); i++)
        CHECK(almost_equal(hidden_fused[i], hidden_ref[i], 1e-5f, 1e-5f));

    std::vector<uint8_t> down_data =
        make_quant_expert_weights(type, hidden_dim, output_dim, experts, rng);
    Tensor down = make_expert_tensor(type, hidden_dim, output_dim,
                                     experts, down_data.data());
    std::vector<float> expert_out(static_cast<size_t>(selected) * output_dim);
    std::vector<float> output_ref(output_dim, 0.0f);
    std::vector<float> output_fused(output_dim, 0.0f);
    fused_moe_gemm_multi(hidden_ref.data(), down, expert_out.data(), expert_idx,
                         selected, hidden_dim, output_dim);
    for (int k = 0; k < selected; k++)
        for (int j = 0; j < output_dim; j++)
            output_ref[j] += route_weight[k] *
                expert_out[static_cast<size_t>(k) * output_dim + j];
    CHECK(fused_moe_down_accumulate(hidden_ref.data(), down, expert_idx,
                                    route_weight, selected, hidden_dim,
                                    output_dim, output_fused.data()));
    for (int j = 0; j < output_dim; j++)
        CHECK(almost_equal(output_fused[j], output_ref[j], 1e-5f, 1e-5f));
}

void test_fp16_conversions() {
    CHECK(fp16_to_fp32(0x3C00) == 1.0f);
    CHECK(fp16_to_fp32(0xC000) == -2.0f);
    CHECK(fp16_to_fp32(0x0000) == 0.0f);
    CHECK(fp16_to_fp32(0x7BFF) == 65504.0f);     // max normal
    CHECK(fp16_to_fp32(0x0001) == 5.9604644775390625e-08f);  // smallest subnormal
    CHECK(fp32_to_fp16(1.0f) == 0x3C00);
    CHECK(fp32_to_fp16(-2.0f) == 0xC000);
    // Round-trip across a sweep of values.
    XorShift32 rng(31);
    for (int i = 0; i < 1000; i++) {
        uint16_t h = static_cast<uint16_t>(rng.next() & 0x7FFF);
        if ((h & 0x7C00) == 0x7C00) continue;  // skip inf/nan
        float f = fp16_to_fp32(h);
        CHECK(fp32_to_fp16(f) == h);
    }
}

void dequant_q2_k_independent(const uint8_t* bytes, float* values) {
    const uint8_t* scales = bytes;
    const uint8_t* qs = bytes + 16;
    float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(bytes + 80));
    float dmin = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(bytes + 82));
    int group = 0;
    for (int half = 0; half < 2; ++half) {
        for (int shift = 0; shift < 8; shift += 2) {
            for (int part = 0; part < 2; ++part) {
                uint8_t sm = scales[group++];
                float scale = d * (sm & 0x0f);
                float minimum = dmin * (sm >> 4);
                for (int lane = 0; lane < 16; ++lane) {
                    int index = half * 128 + (shift / 2) * 32 + part * 16 + lane;
                    values[index] = scale * ((qs[half * 32 + part * 16 + lane] >> shift) & 3) - minimum;
                }
            }
        }
    }
}

void test_derived_q2_k(GGMLType source_type) {
    constexpr int K = 512;
    constexpr int N = 2;
    XorShift32 rng(71u + static_cast<uint32_t>(source_type));
    std::vector<uint8_t> source = make_quant_expert_weights(source_type, K, N, 1, rng);

    DerivedQ2KStorage derived;
    DerivedStorageError error = DerivedStorageError::None;
    CHECK(derive_q2_k_from_gguf(source_type, source, K, N, &derived, &error));
    CHECK(error == DerivedStorageError::None);
    CHECK(derived.contract.version == 1);
    CHECK(derived.contract.source_format == source_type);
    CHECK(derived.contract.logical_k == K);
    CHECK(derived.contract.logical_n == N);
    CHECK(derived.contract.block_width == 256);
    CHECK(derived.contract.bytes_per_block == 84);
    CHECK(derived.contract.alignment == 128);
    CHECK(derived.bytes.size() == static_cast<size_t>(N) * (K / 256) * 84);
    const std::array<uint8_t, 32> zero_digest{};
    CHECK(derived.source_digest != zero_digest);
    CHECK(derived.storage_digest != zero_digest);
    CHECK(derived.source_digest != derived.storage_digest);

    Tensor source_tensor = make_tensor(source_type, K, N, source.data());
    std::vector<float> source_values(static_cast<size_t>(K) * N);
    std::vector<float> production_values(source_values.size());
    std::vector<float> independent_values(source_values.size());
    if (source_type == GGMLType::Q4_K) {
        const auto* blocks = reinterpret_cast<const quant_ref::block_q4_K*>(source.data());
        for (int block = 0; block < N * (K / 256); ++block)
            quant_ref::dequant_q4_K(blocks + block, source_values.data() + block * 256);
    } else {
        const auto* blocks = reinterpret_cast<const quant_ref::block_q6_K*>(source.data());
        for (int block = 0; block < N * (K / 256); ++block)
            quant_ref::dequant_q6_K(blocks + block, source_values.data() + block * 256);
    }
    CHECK(decode_derived_q2_k(derived, production_values));
    for (int block = 0; block < N * (K / 256); ++block)
        dequant_q2_k_independent(derived.bytes.data() + static_cast<size_t>(block) * 84,
                                independent_values.data() + static_cast<size_t>(block) * 256);

    double error_energy = 0.0;
    double zero_energy = 0.0;
    for (size_t i = 0; i < source_values.size(); ++i) {
        CHECK(production_values[i] == independent_values[i]);
        double delta = static_cast<double>(source_values[i]) - independent_values[i];
        error_energy += delta * delta;
        zero_energy += static_cast<double>(source_values[i]) * source_values[i];
    }
    CHECK(error_energy < zero_energy);

    DerivedQ2KStorage rejected;
    CHECK(!derive_q2_k_from_gguf(GGMLType::F16, source, K, N, &rejected, &error));
    CHECK(error == DerivedStorageError::SourceFormatUnsupported);
    CHECK(!derive_q2_k_from_gguf(source_type, source, K - 1, N, &rejected, &error));
    CHECK(error == DerivedStorageError::ShapeUnsupported);
    CHECK(!derive_q2_k_from_gguf(source_type,
                                std::span<const uint8_t>(source.data(), source.size() - 1),
                                K, N, &rejected, &error));
    CHECK(error == DerivedStorageError::SourceLengthMismatch);
}

void test_derived_q2_k_ties_to_even() {
    quant_ref::block_q4_K source{};
    source.d = fp32_to_fp16(1.0f);
    source.scales[0] = 1;
    source.scales[1] = 6;
    for (int lane = 0; lane < 32; ++lane)
        source.qs[lane] = (lane & 1) ? 0x11 : 0x00;

    DerivedQ2KStorage derived;
    DerivedStorageError error = DerivedStorageError::None;
    const auto* bytes = reinterpret_cast<const uint8_t*>(&source);
    CHECK(derive_q2_k_from_gguf(GGMLType::Q4_K,
                                std::span<const uint8_t>(bytes, sizeof(source)),
                                256, 1, &derived, &error));
    CHECK(error == DerivedStorageError::None);
    CHECK((derived.bytes[0] & 0x0f) == 2);
}

void test_iq2_xxs_physical_contract() {
    CHECK(static_cast<uint32_t>(GGMLType::IQ2_XXS) == 16);
    CHECK(elements_per_block(GGMLType::IQ2_XXS) == 256);
    CHECK(bytes_per_block(GGMLType::IQ2_XXS) == 66);
    CHECK(std::strcmp(type_name(GGMLType::IQ2_XXS), "IQ2_XXS") == 0);

    std::array<uint8_t, 66> block{};
    const uint16_t d = fp32_to_fp16(2.0f);
    std::memcpy(block.data(), &d, sizeof(d));

    // First 32-value group: grid indices 0, 1, 2, 3; sign indices
    // 0, 1, 2, 3; sub-block scale nibble 1. Remaining groups use zero
    // indices/signs and scale nibble 0.
    block[2] = 0;
    block[3] = 1;
    block[4] = 2;
    block[5] = 3;
    const uint32_t signs_and_scale =
        (1u << 7) | (2u << 14) | (3u << 21) | (1u << 28);
    std::memcpy(block.data() + 6, &signs_and_scale, sizeof(signs_and_scale));

    Tensor tensor = make_tensor(GGMLType::IQ2_XXS, 256, 1, block.data());
    CHECK(tensor.nbytes() == 66);
    std::array<float, 256> decoded{};
    dequantize(tensor, decoded.data(), static_cast<int>(decoded.size()));

    const std::array<float, 32> expected = {
        6, 6, 6, 6, 6, 6, 6, 6,
        -32.25f, 6, 6, 6, 6, 6, 6, -6,
        18.75f, -18.75f, 6, 6, 6, 6, 6, -6,
        -6, -32.25f, 6, 6, 6, 6, 6, 6,
    };
    for (size_t i = 0; i < expected.size(); ++i) CHECK(decoded[i] == expected[i]);
    for (size_t i = expected.size(); i < decoded.size(); ++i) CHECK(decoded[i] == 2.0f);
}

void test_iq1_s_physical_contract() {
    CHECK(static_cast<uint32_t>(GGMLType::IQ1_S) == 19);
    CHECK(elements_per_block(GGMLType::IQ1_S) == 256);
    CHECK(bytes_per_block(GGMLType::IQ1_S) == 50);
    CHECK(std::strcmp(type_name(GGMLType::IQ1_S), "IQ1_S") == 0);

    std::array<uint8_t, 50> block{};
    const uint16_t d = fp32_to_fp16(2.0f);
    std::memcpy(block.data(), &d, sizeof(d));
    block[3] = 1; // Pinned grid 1 is {1,-1,-1,-1,-1,-1,-1,-1}.
    const uint16_t first_scale = 1u << 12;
    std::memcpy(block.data() + 34, &first_scale, sizeof(first_scale));

    Tensor tensor = make_tensor(GGMLType::IQ1_S, 256, 1, block.data());
    CHECK(tensor.nbytes() == 50);
    std::array<float, 256> decoded{};
    dequantize(tensor, decoded.data(), static_cast<int>(decoded.size()));
    for (size_t lane = 0; lane != 8; ++lane) CHECK(decoded[lane] == -5.25f);
    CHECK(decoded[8] == 6.75f);
    for (size_t lane = 9; lane != 32; ++lane) CHECK(decoded[lane] == -5.25f);
    for (size_t lane = 32; lane != decoded.size(); ++lane) CHECK(decoded[lane] == -1.75f);
}

void dequant_iq2_xxs_independent(const uint8_t* bytes, float* values) {
    uint16_t dh;
    std::memcpy(&dh, bytes, sizeof(dh));
    const float d = fp16_to_fp32(dh);
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        uint32_t grid_indices;
        uint32_t signs_and_scale;
        std::memcpy(&grid_indices, bytes + 2 + 8 * ib32, sizeof(grid_indices));
        std::memcpy(&signs_and_scale, bytes + 6 + 8 * ib32, sizeof(signs_and_scale));
        const float db = d * (0.5f + static_cast<float>(signs_and_scale >> 28)) * 0.25f;
        for (int group = 0; group < 4; ++group) {
            const uint8_t grid_index = static_cast<uint8_t>(grid_indices >> (8 * group));
            const uint64_t grid = kIq2XxsGrid[grid_index];
            const uint8_t signs = kIq2XxsSigns[(signs_and_scale >> (7 * group)) & 127u];
            for (int lane = 0; lane < 8; ++lane) {
                const float magnitude = static_cast<float>((grid >> (8 * lane)) & 0xffu);
                values[ib32 * 32 + group * 8 + lane] =
                    db * magnitude * ((signs & (1u << lane)) ? -1.0f : 1.0f);
            }
        }
    }
}

void dequant_iq1_s_independent(const uint8_t* bytes, float* values) {
    uint16_t dh;
    std::memcpy(&dh, bytes, sizeof(dh));
    const float d = fp16_to_fp32(dh);
    const uint8_t* qs = bytes + 2;
    const uint16_t* qh = reinterpret_cast<const uint16_t*>(bytes + 34);
    for (int block32 = 0; block32 != 8; ++block32) {
        const float scale = d * static_cast<float>(2 * ((qh[block32] >> 12) & 7) + 1);
        const float delta = (qh[block32] & 0x8000) ? -0.125f : 0.125f;
        for (int group = 0; group != 4; ++group) {
            const uint16_t index = static_cast<uint16_t>(
                qs[4 * block32 + group] | (((qh[block32] >> (3 * group)) & 7) << 8));
            const uint64_t grid = kIq1SGrid[index];
            for (int lane = 0; lane != 8; ++lane) {
                const int8_t level = static_cast<int8_t>(grid >> (8 * lane));
                values[block32 * 32 + group * 8 + lane] =
                    scale * (static_cast<float>(level) + delta);
            }
        }
    }
}

void test_derived_iq1_s(GGMLType source_type) {
    constexpr int K = 512;
    constexpr int N = 2;
    XorShift32 rng(173u + static_cast<uint32_t>(source_type));
    std::vector<uint8_t> source = make_quant_expert_weights(source_type, K, N, 1, rng);
    std::vector<float> importance(K);
    for (int i = 0; i != K; ++i)
        importance[i] = i == 17 ? 0.0f : 0.25f + static_cast<float>((i % 19) + 1) / 19.0f;
    std::vector<uint8_t> derived(static_cast<size_t>(N) * (K / 256) * 50);

    DerivedIQ1SRecord record;
    DerivedStorageError error = DerivedStorageError::None;
    CHECK(derive_iq1_s_from_gguf(source_type, source, K, N, importance,
                                 derived, &record, &error));
    CHECK(error == DerivedStorageError::None);
    CHECK(record.contract.version == 1);
    CHECK(record.contract.source_format == source_type);
    CHECK(record.contract.logical_k == K);
    CHECK(record.contract.logical_n == N);
    CHECK(record.contract.block_width == 256);
    CHECK(record.contract.bytes_per_block == 50);
    CHECK(record.contract.alignment == 128);

    std::vector<float> source_values(static_cast<size_t>(K) * N);
    std::vector<float> production_values(source_values.size());
    std::vector<float> independent_values(source_values.size());
    dequantize(make_tensor(source_type, K, N, source.data()), source_values.data(),
               static_cast<int>(source_values.size()));
    dequantize(make_tensor(GGMLType::IQ1_S, K, N, derived.data()), production_values.data(),
               static_cast<int>(production_values.size()));
    for (int block = 0; block != N * (K / 256); ++block)
        dequant_iq1_s_independent(derived.data() + static_cast<size_t>(block) * 50,
                                  independent_values.data() + static_cast<size_t>(block) * 256);

    double error_energy = 0.0;
    double zero_energy = 0.0;
    for (size_t i = 0; i != source_values.size(); ++i) {
        CHECK(production_values[i] == independent_values[i]);
        const double delta = static_cast<double>(source_values[i]) - independent_values[i];
        error_energy += delta * delta;
        zero_energy += static_cast<double>(source_values[i]) * source_values[i];
    }
    CHECK(error_energy < zero_energy);

    std::vector<uint8_t> repeated(derived.size());
    DerivedIQ1SRecord repeated_record;
    CHECK(derive_iq1_s_from_gguf(source_type, source, K, N, importance,
                                 repeated, &repeated_record, &error));
    CHECK(repeated == derived);
    CHECK(repeated_record == record);

    std::vector<float> zero_importance(K, 0.0f);
    std::vector<uint8_t> zero_derived(derived.size(), 0xff);
    DerivedIQ1SRecord zero_record;
    CHECK(derive_iq1_s_from_gguf(source_type, source, K, N, zero_importance,
                                 zero_derived, &zero_record, &error));
    CHECK(std::all_of(zero_derived.begin(), zero_derived.end(),
                      [](uint8_t byte) { return byte == 0; }));

    CHECK(!derive_iq1_s_from_gguf(GGMLType::F16, source, K, N, importance,
                                  derived, &record, &error));
    CHECK(error == DerivedStorageError::SourceFormatUnsupported);
    CHECK(!derive_iq1_s_from_gguf(source_type, source, K - 1, N, importance,
                                  derived, &record, &error));
    CHECK(error == DerivedStorageError::ShapeUnsupported);
}

void test_iq1_s_upstream_neighbor_order() {
    constexpr uint16_t pattern = 0x0001;
    const std::array<float, 8> values = {
        0.125f, -0.875f, -0.875f, -0.875f,
        -0.875f, -0.875f, -0.875f, -0.875f,
    };
    const std::array<float, 8> weights = {1, 1, 1, 1, 1, 1, 1, 1};
    const std::array<float, 3> quant_values = {-0.875f, 0.125f, 1.125f};
    std::array<int8_t, 8> selected{};
    const uint16_t grid = iq1_s_upstream_neighbor_for_testing(
        pattern, values.data(), weights.data(), 1.0f,
        quant_values.data(), selected.data());
    CHECK(grid == 0);
    const std::array<int8_t, 8> expected = {0, 0, 0, 0, 0, 0, 0, 0};
    CHECK(selected == expected);
    CHECK(iq1_s_candidate_order_is_upstream_for_testing());
}

void test_derived_iq2_xxs(GGMLType source_type) {
    constexpr int K = 512;
    constexpr int N = 2;
    XorShift32 rng(113u + static_cast<uint32_t>(source_type));
    std::vector<uint8_t> source = make_quant_expert_weights(source_type, K, N, 1, rng);
    std::vector<float> importance(K);
    for (int i = 0; i < K; ++i) importance[i] = 0.25f + static_cast<float>((i % 17) + 1) / 17.0f;
    std::vector<uint8_t> derived(static_cast<size_t>(N) * (K / 256) * 66);

    DerivedIQ2XXSRecord record;
    DerivedStorageError error = DerivedStorageError::None;
    CHECK(derive_iq2_xxs_from_gguf(source_type, source, K, N, importance,
                                   derived, &record, &error));
    CHECK(error == DerivedStorageError::None);
    CHECK(record.contract.version == 1);
    CHECK(record.contract.source_format == source_type);
    CHECK(record.contract.logical_k == K);
    CHECK(record.contract.logical_n == N);
    CHECK(record.contract.block_width == 256);
    CHECK(record.contract.bytes_per_block == 66);
    CHECK(record.contract.alignment == 128);
    const std::array<uint8_t, 32> zero_digest{};
    CHECK(record.source_digest != zero_digest);
    CHECK(record.importance_digest != zero_digest);
    CHECK(record.storage_digest != zero_digest);

    std::vector<float> source_values(static_cast<size_t>(K) * N);
    std::vector<float> decoded(source_values.size());
    Tensor source_tensor = make_tensor(source_type, K, N, source.data());
    dequantize(source_tensor, source_values.data(), static_cast<int>(source_values.size()));
    for (int block = 0; block < N * (K / 256); ++block)
        dequant_iq2_xxs_independent(derived.data() + static_cast<size_t>(block) * 66,
                                    decoded.data() + static_cast<size_t>(block) * 256);

    double error_energy = 0.0;
    double zero_energy = 0.0;
    for (size_t i = 0; i < source_values.size(); ++i) {
        CHECK(std::isfinite(decoded[i]));
        const double delta = static_cast<double>(source_values[i]) - decoded[i];
        error_energy += delta * delta;
        zero_energy += static_cast<double>(source_values[i]) * source_values[i];
    }
    CHECK(error_energy < zero_energy);

    std::vector<uint8_t> repeated(derived.size());
    DerivedIQ2XXSRecord repeated_record;
    CHECK(derive_iq2_xxs_from_gguf(source_type, source, K, N, importance,
                                   repeated, &repeated_record, &error));
    CHECK(repeated == derived);
    CHECK(repeated_record == record);

    std::vector<float> zero_lane_importance = importance;
    for (size_t index = 0; index < zero_lane_importance.size(); index += 17)
        zero_lane_importance[index] = 0.0f;
    std::vector<uint8_t> zero_lane_first(derived.size()), zero_lane_second(derived.size());
    DerivedIQ2XXSRecord zero_lane_first_record, zero_lane_second_record;
    CHECK(derive_iq2_xxs_from_gguf(source_type, source, K, N, zero_lane_importance,
                                   zero_lane_first, &zero_lane_first_record, &error));
    CHECK(derive_iq2_xxs_from_gguf(source_type, source, K, N, zero_lane_importance,
                                   zero_lane_second, &zero_lane_second_record, &error));
    CHECK(zero_lane_first == zero_lane_second);
    CHECK(zero_lane_first_record == zero_lane_second_record);

    CHECK(!derive_iq2_xxs_from_gguf(GGMLType::F16, source, K, N, importance,
                                    derived, &record, &error));
    CHECK(error == DerivedStorageError::SourceFormatUnsupported);
    CHECK(!derive_iq2_xxs_from_gguf(source_type, source, K, N,
                                    std::span<const float>(importance.data(), K - 1),
                                    derived, &record, &error));
    CHECK(error == DerivedStorageError::ImportanceLengthMismatch);
    CHECK(!derive_iq2_xxs_from_gguf(source_type, source, K, N, importance,
                                    std::span<uint8_t>(derived.data(), derived.size() - 1),
                                    &record, &error));
    CHECK(error == DerivedStorageError::DestinationLengthMismatch);
}

void test_iq2_xxs_upstream_neighbor_distance() {
    constexpr uint16_t pattern = 0x025a;
    const std::array<float, 8> values = {
        0.841011972f, 3.89600805f, 0.154266322f, 1.64835645f,
        1.88284592f, 1.00601754f, 3.98389467f, 0.850422401f,
    };
    const std::array<float, 8> optimization_weights = {
        0.0116678907f, 15.7142842f, 0.125438932f, 0.0f,
        0.00480734332f, 16.3359027f, 0.900414651f, 0.00103265648f,
    };
    std::array<uint8_t, 8> selected{};
    CHECK(iq2_xxs_upstream_neighbor_for_testing(
              pattern, values.data(), optimization_weights.data(), 0.383549282f,
              selected.data()) == 85);
    const std::array<uint8_t, 8> expected = {2, 2, 1, 1, 1, 0, 1, 0};
    CHECK(selected == expected);
}

void test_affine_u2_block256_physical_contract() {
    constexpr uint64_t K = 256;
    constexpr uint64_t N = 2;
    DerivedAffineUInt2Storage storage;
    storage.contract.logical_k = K;
    storage.contract.logical_n = N;
    storage.packed_weights.resize(static_cast<size_t>(K * N / 4));
    storage.scales.resize(static_cast<size_t>(K * N / 256));
    storage.biases.resize(storage.scales.size());

    for (size_t index = 0; index != storage.packed_weights.size(); ++index)
        storage.packed_weights[index] = 0xe4; // LSB-first levels 0,1,2,3.
    for (size_t group = 0; group != storage.scales.size(); ++group) {
        storage.scales[group] = fp32_to_fp16(group & 1 ? 0.25f : 0.5f);
        storage.biases[group] = fp32_to_fp16(group & 1 ? 1.0f : -0.75f);
    }

    DerivedStorageError error = DerivedStorageError::None;
    CHECK(validate_affine_u2_block256(storage, &error));
    CHECK(error == DerivedStorageError::None);
    std::vector<float> decoded;
    CHECK(decode_affine_u2_block256(storage, decoded, &error));
    CHECK(decoded.size() == K * N);
    for (uint64_t row = 0; row != N; ++row) {
        for (uint64_t column = 0; column != K; ++column) {
            const uint64_t group = row * (K / 256) + column / 256;
            const float scale = group & 1 ? 0.25f : 0.5f;
            const float bias = group & 1 ? 1.0f : -0.75f;
            CHECK(decoded[row * K + column] == bias + scale * static_cast<float>(column & 3));
        }
    }

    storage.contract.logical_k = K - 1;
    CHECK(!validate_affine_u2_block256(storage, &error));
    CHECK(error == DerivedStorageError::ShapeUnsupported);
}

void test_derived_affine_u2_block256(GGMLType source_type) {
    constexpr uint64_t K = 512;
    constexpr uint64_t N = 2;
    constexpr size_t alignment = 128;
    XorShift32 rng(911u + static_cast<uint32_t>(source_type));
    std::vector<uint8_t> source = make_quant_expert_weights(
        source_type, static_cast<int>(K), static_cast<int>(N), 1, rng);
    std::vector<float> importance(K);
    for (size_t index = 0; index != importance.size(); ++index)
        importance[index] = index % 31 == 0
            ? 0.0f
            : 0.25f + static_cast<float>((index % 17) + 1) / 17.0f;
    const size_t blocks = static_cast<size_t>(N * (K / 256));
    const size_t packed_size = blocks * 64;
    const size_t plane_size = blocks * sizeof(uint16_t);
    std::vector<uint8_t> arena(packed_size + 2 * plane_size + 3 * alignment);
    const auto align = [](uint8_t* pointer) {
        const uintptr_t value = reinterpret_cast<uintptr_t>(pointer);
        return reinterpret_cast<uint8_t*>((value + 127u) & ~uintptr_t{127u});
    };
    uint8_t* packed = align(arena.data());
    uint8_t* scale_bytes = align(packed + packed_size);
    uint8_t* bias_bytes = align(scale_bytes + plane_size);
    std::span<uint8_t> packed_span(packed, packed_size);
    std::span<uint16_t> scale_span(reinterpret_cast<uint16_t*>(scale_bytes), blocks);
    std::span<uint16_t> bias_span(reinterpret_cast<uint16_t*>(bias_bytes), blocks);

    DerivedAffineUInt2Record record;
    DerivedStorageError error = DerivedStorageError::None;
    CHECK(derive_affine_u2_block256_from_gguf(
        source_type, source, K, N, importance, packed_span, scale_span, bias_span,
        &record, &error));
    CHECK(error == DerivedStorageError::None);
    CHECK(record.contract.logical_k == K);
    CHECK(record.contract.logical_n == N);
    CHECK(record.source_format == source_type);
    const std::array<uint8_t, 32> zero_digest{};
    CHECK(record.source_digest != zero_digest);
    CHECK(record.importance_digest != zero_digest);
    CHECK(record.storage_digest != zero_digest);

    std::vector<float> source_values(K * N);
    dequantize(make_tensor(source_type, static_cast<int>(K), static_cast<int>(N),
                           source.data()),
               source_values.data(), static_cast<int>(source_values.size()));
    std::vector<float> decoded(source_values.size());
    for (size_t index = 0; index != decoded.size(); ++index) {
        const size_t block = index / 256;
        const uint8_t bits = packed_span[index / 4];
        const uint8_t q = static_cast<uint8_t>((bits >> (2 * (index & 3))) & 3u);
        decoded[index] = fp16_to_fp32(scale_span[block]) * static_cast<float>(q) +
                         fp16_to_fp32(bias_span[block]);
        CHECK(std::isfinite(decoded[index]));
    }
    double weighted_error = 0.0;
    double weighted_zero = 0.0;
    for (size_t index = 0; index != decoded.size(); ++index) {
        const double weight = importance[index % K];
        const double delta = static_cast<double>(source_values[index]) - decoded[index];
        weighted_error += weight * delta * delta;
        weighted_zero += weight * static_cast<double>(source_values[index]) *
                         source_values[index];
    }
    CHECK(weighted_error < weighted_zero);

    std::vector<uint8_t> repeated_arena(arena.size());
    uint8_t* repeated_packed = align(repeated_arena.data());
    uint8_t* repeated_scale_bytes = align(repeated_packed + packed_size);
    uint8_t* repeated_bias_bytes = align(repeated_scale_bytes + plane_size);
    std::span<uint8_t> repeated_packed_span(repeated_packed, packed_size);
    std::span<uint16_t> repeated_scale_span(
        reinterpret_cast<uint16_t*>(repeated_scale_bytes), blocks);
    std::span<uint16_t> repeated_bias_span(
        reinterpret_cast<uint16_t*>(repeated_bias_bytes), blocks);
    DerivedAffineUInt2Record repeated_record;
    CHECK(derive_affine_u2_block256_from_gguf(
        source_type, source, K, N, importance, repeated_packed_span,
        repeated_scale_span, repeated_bias_span, &repeated_record, &error));
    CHECK(std::equal(packed_span.begin(), packed_span.end(), repeated_packed_span.begin()));
    CHECK(std::equal(scale_span.begin(), scale_span.end(), repeated_scale_span.begin()));
    CHECK(std::equal(bias_span.begin(), bias_span.end(), repeated_bias_span.begin()));
    CHECK(repeated_record == record);

    CHECK(!derive_affine_u2_block256_from_gguf(
        GGMLType::F16, source, K, N, importance, packed_span, scale_span,
        bias_span, &record, &error));
    CHECK(error == DerivedStorageError::SourceFormatUnsupported);
    CHECK(!derive_affine_u2_block256_from_gguf(
        source_type, source, K, N,
        std::span<const float>(importance.data(), importance.size() - 1),
        packed_span, scale_span, bias_span, &record, &error));
    CHECK(error == DerivedStorageError::ImportanceLengthMismatch);
    std::vector<float> zero_importance(K, 0.0f);
    CHECK(!derive_affine_u2_block256_from_gguf(
        source_type, source, K, N, zero_importance, packed_span, scale_span,
        bias_span, &record, &error));
    CHECK(error == DerivedStorageError::InvalidImportance);
    CHECK(!derive_affine_u2_block256_from_gguf(
        source_type, source, K, N, importance,
        std::span<uint8_t>(packed_span.data() + 1, packed_span.size()),
        scale_span, bias_span, &record, &error));
    CHECK(error == DerivedStorageError::DestinationAlignmentMismatch);
}

void test_derived_column_grouped_affine_u2_skip(GGMLType source_type) {
    constexpr uint64_t K = 256;
    constexpr uint64_t N = 256;
    constexpr size_t alignment = 128;
    XorShift32 rng(1201u + static_cast<uint32_t>(source_type));
    std::vector<uint8_t> source = make_quant_expert_weights(
        source_type, static_cast<int>(K), static_cast<int>(N), 1, rng);
    std::vector<float> importance(K);
    for (size_t column = 0; column != importance.size(); ++column)
        importance[column] = column % 29u == 0u
            ? 0.0f : 0.5f + static_cast<float>((column % 11u) + 1u) / 11.0f;

    ColumnGroupedAffineUInt2SkipV1Contract contract;
    ColumnGroupedAffineUInt2SkipV1Error contract_error{};
    CHECK(column_grouped_affine_uint2_skip_v1_make_contract(K, N, &contract,
                                                            &contract_error));
    std::vector<uint8_t> arena(static_cast<size_t>(contract.values_bytes +
                                                   contract.scale_bytes +
                                                   contract.bias_bytes) + 3u * alignment);
    const auto align = [](uint8_t* pointer) {
        const uintptr_t value = reinterpret_cast<uintptr_t>(pointer);
        return reinterpret_cast<uint8_t*>((value + 127u) & ~uintptr_t{127u});
    };
    uint8_t* values = align(arena.data());
    uint8_t* scale_bytes = align(values + contract.values_bytes);
    uint8_t* bias_bytes = align(scale_bytes + contract.scale_bytes);
    std::span<uint8_t> values_span(values, static_cast<size_t>(contract.values_bytes));
    std::span<uint16_t> scales(reinterpret_cast<uint16_t*>(scale_bytes),
                               static_cast<size_t>(contract.group_count));
    std::span<uint16_t> biases(reinterpret_cast<uint16_t*>(bias_bytes),
                               static_cast<size_t>(contract.group_count));

    ColumnGroupedAffineUInt2SkipV1Storage storage;
    DerivedStorageError error = DerivedStorageError::None;
    CHECK(derive_column_grouped_affine_u2_skip_v1_from_gguf(
        source_type, source, K, N, importance, values_span, scales, biases,
        &storage, &error));
    CHECK(error == DerivedStorageError::None);
    const std::array<uint8_t, 32> zero{};
    CHECK(storage.source_digest != zero);
    CHECK(storage.provenance_digest != zero);
    CHECK(storage.storage_digest != zero);
    CHECK(column_grouped_affine_uint2_skip_v1_validate(
        storage, storage.source_digest, storage.provenance_digest, &contract_error));

    std::vector<float> source_values(K * N);
    if (source_type == GGMLType::Q4_K) {
        const auto* blocks = reinterpret_cast<const quant_ref::block_q4_K*>(source.data());
        for (size_t row = 0; row != N; ++row)
            quant_ref::dequant_q4_K(blocks + row, source_values.data() + row * K);
    } else {
        const auto* blocks = reinterpret_cast<const quant_ref::block_q6_K*>(source.data());
        for (size_t row = 0; row != N; ++row)
            quant_ref::dequant_q6_K(blocks + row, source_values.data() + row * K);
    }
    std::vector<float> decoded(source_values.size());
    CHECK(column_grouped_affine_uint2_skip_v1_decode(
        storage, storage.source_digest, storage.provenance_digest, decoded,
        &contract_error));
    double error_energy = 0.0;
    double zero_energy = 0.0;
    for (size_t index = 0; index != decoded.size(); ++index) {
        CHECK(std::isfinite(decoded[index]));
        const double delta = static_cast<double>(source_values[index]) - decoded[index];
        error_energy += delta * delta;
        zero_energy += static_cast<double>(source_values[index]) * source_values[index];
    }
    CHECK(error_energy < zero_energy);

    std::vector<uint8_t> repeated_arena(arena.size());
    uint8_t* repeated_values = align(repeated_arena.data());
    uint8_t* repeated_scale_bytes = align(repeated_values + contract.values_bytes);
    uint8_t* repeated_bias_bytes = align(repeated_scale_bytes + contract.scale_bytes);
    std::span<uint8_t> repeated_values_span(
        repeated_values, static_cast<size_t>(contract.values_bytes));
    std::span<uint16_t> repeated_scales(
        reinterpret_cast<uint16_t*>(repeated_scale_bytes),
        static_cast<size_t>(contract.group_count));
    std::span<uint16_t> repeated_biases(
        reinterpret_cast<uint16_t*>(repeated_bias_bytes),
        static_cast<size_t>(contract.group_count));
    ColumnGroupedAffineUInt2SkipV1Storage repeated;
    CHECK(derive_column_grouped_affine_u2_skip_v1_from_gguf(
        source_type, source, K, N, importance, repeated_values_span,
        repeated_scales, repeated_biases, &repeated, &error));
    CHECK(std::equal(values_span.begin(), values_span.end(), repeated_values_span.begin()));
    CHECK(std::equal(scales.begin(), scales.end(), repeated_scales.begin()));
    CHECK(std::equal(biases.begin(), biases.end(), repeated_biases.begin()));
    CHECK(repeated.source_digest == storage.source_digest);
    CHECK(repeated.provenance_digest == storage.provenance_digest);
    CHECK(repeated.storage_digest == storage.storage_digest);

    CHECK(!derive_column_grouped_affine_u2_skip_v1_from_gguf(
        GGMLType::F16, source, K, N, importance, values_span, scales, biases,
        &storage, &error));
    CHECK(error == DerivedStorageError::SourceFormatUnsupported);
    CHECK(!derive_column_grouped_affine_u2_skip_v1_from_gguf(
        source_type, source, K, N,
        std::span<const float>(importance.data(), importance.size() - 1u),
        values_span, scales, biases, &storage, &error));
    CHECK(error == DerivedStorageError::ImportanceLengthMismatch);
    std::vector<float> zero_importance(K, 0.0f);
    CHECK(!derive_column_grouped_affine_u2_skip_v1_from_gguf(
        source_type, source, K, N, zero_importance, values_span, scales,
        biases, &storage, &error));
    CHECK(error == DerivedStorageError::InvalidImportance);
    CHECK(!derive_column_grouped_affine_u2_skip_v1_from_gguf(
        source_type, source, K, N, importance,
        std::span<uint8_t>(values_span.data() + 1u, values_span.size()), scales,
        biases, &storage, &error));
    CHECK(error == DerivedStorageError::DestinationAlignmentMismatch);
}

} // namespace

int main() {
    test_fp16_conversions();
    test_iq2_xxs_physical_contract();
    test_iq1_s_physical_contract();
    test_derived_iq2_xxs(GGMLType::Q4_K);
    test_derived_iq2_xxs(GGMLType::Q6_K);
    test_derived_iq1_s(GGMLType::Q4_K);
    test_derived_iq1_s(GGMLType::Q6_K);
    test_iq1_s_upstream_neighbor_order();
    test_iq2_xxs_upstream_neighbor_distance();
    test_affine_u2_block256_physical_contract();
    test_derived_affine_u2_block256(GGMLType::Q4_K);
    test_derived_affine_u2_block256(GGMLType::Q6_K);
    test_derived_column_grouped_affine_u2_skip(GGMLType::Q4_K);
    test_derived_column_grouped_affine_u2_skip(GGMLType::Q6_K);
    test_derived_q2_k(GGMLType::Q4_K);
    test_derived_q2_k(GGMLType::Q6_K);
    test_derived_q2_k_ties_to_even();
    test_column_grouped_u2_skip_is_cpu_rejected();
    test_batched_rows();
    test_routed_moe_fusion();
    for (GGMLType type : {GGMLType::Q4_0, GGMLType::Q8_0,
                          GGMLType::Q4_K, GGMLType::Q6_K})
        test_routed_moe_quant_fusion(type);

    // K must be a multiple of 256 for K-quants; exercise N=1, narrow N, and
    // N >= 128 (the threshold for the OpenMP-parallel row loop).
    for (int N : {1, 8, 33, 256}) {
        test_f32 (512, N);
        test_f16 (512, N);
        test_bf16(512, N);
        test_q4_0(512, N);
        test_q8_0(512, N);
        test_q4_k(512, N);
        test_q6_k(512, N);
    }
    // Model-sized K (26B hidden=2816, expert_inter=704*2=1408)
    for (int N : {1, 8, 256, 4096}) {
        test_q4_k(2816, N);
        test_q6_k(2816, N);
    }
    return test_summary("test_matmul");
}
