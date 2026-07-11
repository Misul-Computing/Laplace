// test_matmul - kernel correctness against an independent GGUF-layout reference.
//
// Contract under test:
//   GGUF tensors are row-major with dims[0] innermost. A weight W with
//   dims = [K, N] is stored as N contiguous rows of K elements, so
//   element (k, j) lives at j*K + k, and quantized blocks of row j are
//   contiguous at block index j*(K/QK) + b.
//
//   matmul_row(x, W, y, K, N) must compute y[j] = sum_k x[k] * W[k, j].

#include <cstring>
#include <vector>

#include "fp16.h"
#include "matmul.h"
#include "tensor.h"

#include "quant_ref.h"
#include "test_util.h"

using namespace Laplace;

namespace {

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
    for (GGMLType type : {GGMLType::F32, GGMLType::Q8_0, GGMLType::Q4_K}) {
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

} // namespace

int main() {
    test_fp16_conversions();
    test_batched_rows();

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
