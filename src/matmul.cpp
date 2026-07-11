#include "matmul.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "fp16.h"
#include "kernels.h"
#include "threadpool.h"

#if defined(__APPLE__)
namespace Laplace {
extern bool metal_gemv(const float* x, const Tensor& w, float* y, int K, int N);
extern bool metal_gemm(const float* x, const Tensor& w, float* y, int M, int K, int N);
extern void metal_register_weights(const void* base, size_t size);
}
static bool metal_enabled() {
    const char* e = std::getenv("LAPLACE_METAL");
    return e && e[0] == '1';
}
#else
namespace Laplace { bool metal_gemv(const float*, const Tensor&, float*, int, int) { return false; } bool metal_gemm(const float*, const Tensor&, float*, int, int, int) { return false; } void metal_register_weights(const void*, size_t) {} }
static bool metal_enabled() { return false; }
#endif

namespace Laplace {

using kernels::block_q4_0;
using kernels::block_q5_0;
using kernels::block_q8_0;
using kernels::block_q4_K;
using kernels::block_q6_K;
using kernels::block_q2_K;
using kernels::block_q3_K;
using kernels::block_q5_K;
using kernels::get_scale_min_k4;

namespace {

// ---------------- runtime kernel dispatch -----------------------------------
// The SIMD back-end (matmul_simd.cpp, compiled with ARMv8.x ISA flags)
// exports a whole-GEMM entry point. LAPLACE_NOSIMD=1 forces the portable
// scalar path below.

kernels::gemm_fn simd_gemm() {
    static const kernels::gemm_fn fn = [] {
        const char* off = std::getenv("LAPLACE_NOSIMD");
        if (off && off[0] == '1') return static_cast<kernels::gemm_fn>(nullptr);
        return kernels::get_simd_gemm();
    }();
    return fn;
}

// ---------------- scalar per-row dot products (portable reference) ----------
// GGUF tensors are row-major with dims[0] innermost: a weight with
// dims = [K, N] is N contiguous rows of K elements. Quantized rows are
// sequences of K/QK blocks. All kernels walk one output row at a time,
// which is both correct and cache-friendly (each row streams sequentially).

float dot_row_f32(const float* x, const uint8_t* row, int K) {
    const float* w = reinterpret_cast<const float*>(row);
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += x[k] * w[k];
    return acc;
}

float dot_row_f16(const float* x, const uint8_t* row, int K) {
    const uint16_t* w = reinterpret_cast<const uint16_t*>(row);
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += x[k] * fp16_to_fp32(w[k]);
    return acc;
}

float dot_row_bf16(const float* x, const uint8_t* row, int K) {
    const uint16_t* w = reinterpret_cast<const uint16_t*>(row);
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += x[k] * bf16_to_fp32(w[k]);
    return acc;
}

float dot_row_q4_0(const float* x, const uint8_t* row, int K) {
    const block_q4_0* w = reinterpret_cast<const block_q4_0*>(row);
    const int n_blocks = K / 32;
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q4_0& blk = w[b];
        const float* xb = x + b * 32;
        float d = fp16_to_fp32(blk.d);
        float sum = 0.0f;
        for (int l = 0; l < 16; l++) {
            sum += xb[l +  0] * static_cast<float>((blk.qs[l] & 0xF) - 8);
            sum += xb[l + 16] * static_cast<float>((blk.qs[l] >>  4) - 8);
        }
        acc += d * sum;
    }
    return acc;
}

float dot_row_q5_0(const float* x, const uint8_t* row, int K) {
    const block_q5_0* w = reinterpret_cast<const block_q5_0*>(row);
    const int n_blocks = K / 32;
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q5_0& blk = w[b];
        const float* xb = x + b * 32;
        float d = fp16_to_fp32(blk.d);
        uint32_t qh = blk.qh;
        float sum = 0.0f;
        for (int l = 0; l < 16; l++) {
            int lo0 = (blk.qs[l] & 0xF) | (((qh >> (l +  0)) & 1) << 4);
            int lo1 = (blk.qs[l] >> 4)   | (((qh >> (l + 16)) & 1) << 4);
            sum += xb[l +  0] * static_cast<float>(lo0 - 16);
            sum += xb[l + 16] * static_cast<float>(lo1 - 16);
        }
        acc += d * sum;
    }
    return acc;
}

float dot_row_q8_0(const float* x, const uint8_t* row, int K) {
    const block_q8_0* w = reinterpret_cast<const block_q8_0*>(row);
    const int n_blocks = K / 32;
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q8_0& blk = w[b];
        const float* xb = x + b * 32;
        float sum = 0.0f;
        for (int l = 0; l < 32; l++) sum += xb[l] * blk.qs[l];
        acc += fp16_to_fp32(blk.d) * sum;
    }
    return acc;
}

float dot_row_q4_k(const float* x, const uint8_t* row, int K) {
    const block_q4_K* w = reinterpret_cast<const block_q4_K*>(row);
    const int n_blocks = K / 256;
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q4_K& blk = w[b];
        const float* xb = x + b * 256;
        float d    = fp16_to_fp32(blk.d);
        float dmin = fp16_to_fp32(blk.dmin);
        const uint8_t* q = blk.qs;
        int is = 0;
        for (int jb = 0; jb < 256; jb += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, blk.scales, &sc, &m);
            float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, blk.scales, &sc, &m);
            float d2 = d * sc, m2 = dmin * m;
            float s1 = 0.0f, s2 = 0.0f, x1 = 0.0f, x2 = 0.0f;
            for (int l = 0; l < 32; l++) {
                s1 += xb[jb + l]      * static_cast<float>(q[l] & 0xF);
                x1 += xb[jb + l];
                s2 += xb[jb + 32 + l] * static_cast<float>(q[l] >>  4);
                x2 += xb[jb + 32 + l];
            }
            acc += d1 * s1 - m1 * x1 + d2 * s2 - m2 * x2;
            q += 32;
            is += 2;
        }
    }
    return acc;
}

float dot_row_q6_k(const float* x, const uint8_t* row, int K) {
    const block_q6_K* w = reinterpret_cast<const block_q6_K*>(row);
    const int n_blocks = K / 256;
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q6_K& blk = w[b];
        float d = fp16_to_fp32(blk.d);
        const uint8_t* ql = blk.ql;
        const uint8_t* qh = blk.qh;
        const int8_t*  sc = blk.scales;
        const float*   xb = x + b * 256;
        for (int n_off = 0; n_off < 256; n_off += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = static_cast<int>((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = static_cast<int>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = static_cast<int>((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = static_cast<int>((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                acc += xb[n_off + l +  0] * (d * sc[is + 0] * q1);
                acc += xb[n_off + l + 32] * (d * sc[is + 2] * q2);
                acc += xb[n_off + l + 64] * (d * sc[is + 4] * q3);
                acc += xb[n_off + l + 96] * (d * sc[is + 6] * q4);
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return acc;
}

float dot_row_q2_k(const float* x, const uint8_t* row, int K) {
    const block_q2_K* w = reinterpret_cast<const block_q2_K*>(row);
    const int n_blocks = K / 256;
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q2_K& blk = w[b];
        const float* xb = x + b * 256;
        float d    = fp16_to_fp32(blk.d);
        float dmin = fp16_to_fp32(blk.dmin);
        const uint8_t* q  = blk.qs;
        const uint8_t* sc = blk.scales;
        const float* xq = xb;
        int is = 0;
        float isum = 0.0f, summs = 0.0f;
        for (int k = 0; k < 2; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float s = sc[is] & 0xF, m = sc[is] >> 4;
                float suml = 0.0f, xsum = 0.0f;
                for (int l = 0; l < 16; l++) { suml += xq[l] * ((q[l] >> shift) & 3); xsum += xq[l]; }
                isum += s * suml; summs += m * xsum;
                xq += 16; is++;
                s = sc[is] & 0xF; m = sc[is] >> 4;
                suml = 0.0f; xsum = 0.0f;
                for (int l = 16; l < 32; l++) { suml += xq[l - 16] * ((q[l] >> shift) & 3); xsum += xq[l - 16]; }
                isum += s * suml; summs += m * xsum;
                xq += 16; is++;
                shift += 2;
            }
            q += 32;
        }
        acc += d * isum - dmin * summs;
    }
    return acc;
}

float dot_row_q3_k(const float* x, const uint8_t* row, int K) {
    const block_q3_K* w = reinterpret_cast<const block_q3_K*>(row);
    const int n_blocks = K / 256;
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q3_K& blk = w[b];
        const float* xb = x + b * 256;
        float d = fp16_to_fp32(blk.d);
        // Unpack 16 6-bit scales from the 12-byte packed scales array.
        uint32_t auxs[4];
        std::memcpy(auxs, blk.scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const uint8_t* scales = reinterpret_cast<const uint8_t*>(auxs);
        const uint8_t* q3 = blk.qs;
        const uint8_t* hm = blk.hmask;
        const float* xq = xb;
        int is = 0;
        uint32_t m = 1;
        for (int j = 0; j < 2; j++) {
            for (int s = 0; s < 4; s++) {
                int shift = s * 2;
                float sum0 = 0.0f, sum1 = 0.0f;
                for (int l = 0; l < 16; l++) {
                    int a = (q3[l] >> shift) & 3;
                    a -= (hm[l] & m) ? 0 : 4;
                    sum0 += xq[l] * a;
                }
                for (int l = 16; l < 32; l++) {
                    int a = (q3[l] >> shift) & 3;
                    a -= (hm[l] & m) ? 0 : 4;
                    sum1 += xq[l] * a;
                }
                acc += d * (scales[is] - 32) * sum0; is++;
                acc += d * (scales[is] - 32) * sum1; is++;
                xq += 32;
                m <<= 1;
            }
            q3 += 32;
        }
    }
    return acc;
}

float dot_row_q5_k(const float* x, const uint8_t* row, int K) {
    const block_q5_K* w = reinterpret_cast<const block_q5_K*>(row);
    const int n_blocks = K / 256;
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q5_K& blk = w[b];
        const float* xb = x + b * 256;
        float d    = fp16_to_fp32(blk.d);
        float dmin = fp16_to_fp32(blk.dmin);
        const uint8_t* q = blk.qs;
        int is = 0;
        for (int jb = 0; jb < 256; jb += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, blk.scales, &sc, &m);
            float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, blk.scales, &sc, &m);
            float d2 = d * sc, m2 = dmin * m;
            int bit = 2 * (jb / 64);
            float s1 = 0.0f, s2 = 0.0f, x1 = 0.0f, x2 = 0.0f;
            for (int l = 0; l < 32; l++) {
                int v1 = (q[l] & 0xF) + 16 * ((blk.qh[l] >> bit) & 1);
                int v2 = (q[l] >> 4) + 16 * ((blk.qh[l] >> (bit + 1)) & 1);
                s1 += xb[jb + l]      * v1; x1 += xb[jb + l];
                s2 += xb[jb + 32 + l] * v2; x2 += xb[jb + 32 + l];
            }
            acc += d1 * s1 - m1 * x1 + d2 * s2 - m2 * x2;
            q += 32; is += 2;
        }
    }
    return acc;
}

// Scalar GEMM. The dot kernel is a template parameter so the call is direct
// and inlines into the row loop.
using dot_f_fn = float (*)(const float*, const uint8_t*, int);

template <dot_f_fn DOT>
void gemm_scalar(const float* x, const uint8_t* data, float* y,
                 int M, int K, int N, size_t rb) {
    auto body = [&](int j) {
        const uint8_t* row = data + static_cast<size_t>(j) * rb;
        for (int m = 0; m < M; m++) {
            y[static_cast<size_t>(m) * N + j] = DOT(x + static_cast<size_t>(m) * K, row, K);
        }
    };
    if (M > 1 || N >= 128) {
        ThreadPool::get().parallel_for(N, body);
    } else {
        for (int j = 0; j < N; j++) body(j);
    }
}

bool gemm_fallback(const float* x, const uint8_t* w, GGMLType type,
                   float* y, int M, int K, int N) {
    const size_t rb = static_cast<size_t>(K) / elements_per_block(type) * bytes_per_block(type);
    switch (type) {
        case GGMLType::F32:  gemm_scalar<dot_row_f32 >(x, w, y, M, K, N, rb); return true;
        case GGMLType::F16:  gemm_scalar<dot_row_f16 >(x, w, y, M, K, N, rb); return true;
        case GGMLType::BF16: gemm_scalar<dot_row_bf16>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q4_0: gemm_scalar<dot_row_q4_0>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q5_0: gemm_scalar<dot_row_q5_0>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q8_0: gemm_scalar<dot_row_q8_0>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q4_K: gemm_scalar<dot_row_q4_k>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q6_K: gemm_scalar<dot_row_q6_k>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q2_K: gemm_scalar<dot_row_q2_k>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q3_K: gemm_scalar<dot_row_q3_k>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q5_K: gemm_scalar<dot_row_q5_k>(x, w, y, M, K, N, rb); return true;
        default: return false;
    }
}

// LAPLACE_PROF=1: accumulate wall time spent inside the matmuls and report at
// process exit (separates matmul cost from everything else).
struct MatmulProf {
    bool   on = std::getenv("LAPLACE_PROF") != nullptr;
    double seconds = 0.0;
    long   calls = 0;
    ~MatmulProf() {
        if (on) fprintf(stderr, "PROF matmul_row: %.3f s over %ld calls\n", seconds, calls);
    }
};
MatmulProf g_prof;

struct ProfScope {
    std::chrono::steady_clock::time_point t0;
    ProfScope() {
        if (g_prof.on) t0 = std::chrono::steady_clock::now();
    }
    ~ProfScope() {
        if (g_prof.on) {
            g_prof.seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            g_prof.calls++;
        }
    }
};

} // namespace

// MLX affine quantization dot product.
// w: packed uint32 weights (32/bits elements per word, LSB first)
// scales: per-group scale (fp16), biases: per-group bias (fp16)
static float dot_row_mlx(const float* x, const uint8_t* w,
                         const uint16_t* scales, const uint16_t* biases,
                         int K, int bits, int group_size) {
    int epw = 32 / bits;
    int mask = (1 << bits) - 1;
    int n_groups = K / group_size;
    const uint32_t* qw = reinterpret_cast<const uint32_t*>(w);
    float acc = 0;
    for (int g = 0; g < n_groups; g++) {
        float scale = fp16_to_fp32(scales[g]);
        float bias = fp16_to_fp32(biases[g]);
        for (int i = 0; i < group_size; i++) {
            int k = g * group_size + i;
            int word_idx = k / epw;
            int shift = (k % epw) * bits;
            float q = float((qw[word_idx] >> shift) & mask);
            acc += x[k] * (scale * q + bias);
        }
    }
    return acc;
}

void matmul_rows(const float* x, const Tensor& w, float* y, int M, int K, int N) {
    if (w.type == GGMLType::MLX_AFFINE) {
        int bits = w.mlx_bits;
        int gs = w.mlx_group_size;
        int epw = 32 / bits;
        size_t packed_K = (size_t)(K + epw - 1) / epw;
        size_t rb = packed_K * 4;
        for (int m = 0; m < M; m++) {
            for (int j = 0; j < N; j++) {
                const uint8_t* row = w.data + (size_t)j * rb;
                const uint16_t* sc = reinterpret_cast<const uint16_t*>(w.scales) + (size_t)j * (K / gs);
                const uint16_t* bi = reinterpret_cast<const uint16_t*>(w.biases) + (size_t)j * (K / gs);
                y[(size_t)m * N + j] = dot_row_mlx(x + (size_t)m * K, row, sc, bi, K, bits, gs);
            }
        }
        return;
    }
    ProfScope prof;
    if (kernels::gemm_fn gemm = simd_gemm()) {
        if (gemm(x, w.data, w.type, y, M, K, N)) return;
    }
    if (!gemm_fallback(x, w.data, w.type, y, M, K, N)) {
        fprintf(stderr, "matmul: unsupported weight type %s\n", type_name(w.type));
        std::memset(y, 0, sizeof(float) * static_cast<size_t>(M) * N);
    }
}

void matmul_row(const float* x, const Tensor& w, float* y, int K, int N) {
    matmul_rows(x, w, y, 1, K, N);
}

void matmul_lm_head(const float* x, const Tensor& w, float* y, int M, int K, int N) {
    if (metal_enabled()) {
        if (metal_gemm(x, w, y, M, K, N)) return;
    }
    matmul_rows(x, w, y, M, K, N);
}

void matmul_register_weights(const void* base, size_t size) {
    metal_register_weights(base, size);
}

bool matmul_gemm_batch(const MatmulBatchSpec* specs, int n) {
    for (int i = 0; i < n; i++)
        matmul_rows(specs[i].x, *specs[i].w, specs[i].y, 1, specs[i].K, specs[i].N);
    return true;
}

void fused_moe_gemm_idx(const float* x, const Tensor& w, float* y,
                        const int* expert_idx, int n_experts,
                        int K, int N) {
    // Fused MoE GEMV with indirect expert access. One parallel_for across
    // all selected experts' columns. Activation quantized once.
    ProfScope prof;
    if (kernels::moe_gemv_fn moe = [] {
        const char* off = std::getenv("LAPLACE_NOSIMD");
        if (off && off[0] == '1') return static_cast<kernels::moe_gemv_fn>(nullptr);
        return kernels::get_simd_moe_gemv();
    }()) {
        if (moe(x, w.data, w.type, expert_idx, n_experts, y, K, N)) return;
    }
    // Fallback: sequential matmul per expert.
    for (int k = 0; k < n_experts; k++) {
        int e = expert_idx[k];
        size_t per_expert = static_cast<size_t>(N) * (static_cast<size_t>(K) / elements_per_block(w.type) * bytes_per_block(w.type));
        Tensor view = w;
        view.data = w.data + static_cast<size_t>(e) * per_expert;
        matmul_row(x, view, y + static_cast<size_t>(k) * N, K, N);
    }
}

void fused_moe_gemm_multi(const float* x, const Tensor& w, float* y,
                          const int* expert_idx, int n_experts,
                          int K, int N) {
    // Fused MoE GEMV with per-expert activations. Each expert k has its
    // own input at x[k * K]. One parallel_for across all experts' columns.
    ProfScope prof;
    if (kernels::moe_gemv_multi_fn moe = [] {
        const char* off = std::getenv("LAPLACE_NOSIMD");
        if (off && off[0] == '1') return static_cast<kernels::moe_gemv_multi_fn>(nullptr);
        return kernels::get_simd_moe_gemv_multi();
    }()) {
        if (moe(x, w.data, w.type, expert_idx, n_experts, y, K, N)) return;
    }
    // Fallback: sequential matmul per expert.
    for (int k = 0; k < n_experts; k++) {
        int e = expert_idx[k];
        size_t per_expert = static_cast<size_t>(N) * (static_cast<size_t>(K) / elements_per_block(w.type) * bytes_per_block(w.type));
        Tensor view = w;
        view.data = w.data + static_cast<size_t>(e) * per_expert;
        matmul_row(x + static_cast<size_t>(k) * K, view, y + static_cast<size_t>(k) * N, K, N);
    }
}

// ---------------- Dequantize (embeddings, verification) ---------------------

void dequantize(const Tensor& w, float* dst, int n) {
    switch (w.type) {
        case GGMLType::F32: {
            const float* p = reinterpret_cast<const float*>(w.data);
            std::memcpy(dst, p, sizeof(float) * n);
            return;
        }
        case GGMLType::F16: {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(w.data);
            for (int i = 0; i < n; i++) dst[i] = fp16_to_fp32(p[i]);
            return;
        }
        case GGMLType::BF16: {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(w.data);
            for (int i = 0; i < n; i++) dst[i] = bf16_to_fp32(p[i]);
            return;
        }
        case GGMLType::Q4_0: {
            const block_q4_0* p = reinterpret_cast<const block_q4_0*>(w.data);
            for (int i = 0; i < n; i += 32) {
                float d = fp16_to_fp32(p[i/32].d);
                for (int l = 0; l < 16; l++) {
                    dst[i + l]      = d * ((p[i/32].qs[l] & 0xF) - 8);
                    dst[i + l + 16] = d * ((p[i/32].qs[l] >>  4) - 8);
                }
            }
            return;
        }
        case GGMLType::Q5_0: {
            const block_q5_0* p = reinterpret_cast<const block_q5_0*>(w.data);
            for (int i = 0; i < n; i += 32) {
                float d = fp16_to_fp32(p[i/32].d);
                uint32_t qh = p[i/32].qh;
                for (int l = 0; l < 16; l++) {
                    int lo0 = (p[i/32].qs[l] & 0xF) | (((qh >> (l +  0)) & 1) << 4);
                    int lo1 = (p[i/32].qs[l] >> 4)   | (((qh >> (l + 16)) & 1) << 4);
                    dst[i + l]      = d * (lo0 - 16);
                    dst[i + l + 16] = d * (lo1 - 16);
                }
            }
            return;
        }
        case GGMLType::Q8_0: {
            const block_q8_0* p = reinterpret_cast<const block_q8_0*>(w.data);
            for (int i = 0; i < n; i += 32) {
                float d = fp16_to_fp32(p[i/32].d);
                for (int l = 0; l < 32; l++) dst[i + l] = d * p[i/32].qs[l];
            }
            return;
        }
        case GGMLType::Q4_K: {
            const block_q4_K* p = reinterpret_cast<const block_q4_K*>(w.data);
            for (int i = 0; i < n; i += 256) {
                const block_q4_K& blk = p[i/256];
                float d = fp16_to_fp32(blk.d);
                float dmin = fp16_to_fp32(blk.dmin);
                const uint8_t* q = blk.qs;
                int is = 0;
                for (int jb = 0; jb < 256; jb += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is + 0, blk.scales, &sc, &m);
                    float d1 = d * sc, m1 = dmin * m;
                    get_scale_min_k4(is + 1, blk.scales, &sc, &m);
                    float d2 = d * sc, m2 = dmin * m;
                    for (int l = 0; l < 32; l++) dst[i + jb + l]      = d1 * (q[l] & 0xF) - m1;
                    for (int l = 0; l < 32; l++) dst[i + jb + 32 + l] = d2 * (q[l] >> 4) - m2;
                    q += 32; is += 2;
                }
            }
            return;
        }
        case GGMLType::Q6_K: {
            const block_q6_K* p = reinterpret_cast<const block_q6_K*>(w.data);
            for (int i = 0; i < n; i += 256) {
                const block_q6_K& blk = p[i/256];
                float d = fp16_to_fp32(blk.d);
                const uint8_t* ql = blk.ql;
                const uint8_t* qh = blk.qh;
                const int8_t*  sc = blk.scales;
                float* y = dst + i;
                for (int n_off = 0; n_off < 256; n_off += 128) {
                    for (int l = 0; l < 32; l++) {
                        int is = l / 16;
                        int q1 = static_cast<int>((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        int q2 = static_cast<int>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        int q3 = static_cast<int>((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        int q4 = static_cast<int>((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[l +  0] = d * sc[is + 0] * q1;
                        y[l + 32] = d * sc[is + 2] * q2;
                        y[l + 64] = d * sc[is + 4] * q3;
                        y[l + 96] = d * sc[is + 6] * q4;
                    }
                    y  += 128;
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
            }
            return;
        }
        case GGMLType::Q2_K: {
            const block_q2_K* p = reinterpret_cast<const block_q2_K*>(w.data);
            for (int i = 0; i < n; i += 256) {
                const block_q2_K& blk = p[i/256];
                float d = fp16_to_fp32(blk.d), dmin = fp16_to_fp32(blk.dmin);
                for (int j = 0; j < 16; j++) {
                    float sc = blk.scales[j] & 0xF, mn = blk.scales[j] >> 4;
                    int half = j/8, jj = j%8, group = jj/2, lo = jj%2;
                    for (int l = 0; l < 16; l++) {
                        int q = (blk.qs[half*32 + lo*16 + l] >> (group*2)) & 3;
                        dst[i + j*16 + l] = d * sc * q - dmin * mn;
                    }
                }
            }
            return;
        }
        case GGMLType::Q3_K: {
            const block_q3_K* p = reinterpret_cast<const block_q3_K*>(w.data);
            const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
            for (int i = 0; i < n; i += 256) {
                const block_q3_K& blk = p[i/256];
                float d = fp16_to_fp32(blk.d);
                uint32_t auxs[4];
                std::memcpy(auxs, blk.scales, 12);
                uint32_t tmp = auxs[2];
                auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
                const uint8_t* scales = reinterpret_cast<const uint8_t*>(auxs);
                for (int j = 0; j < 16; j++) {
                    float sc = static_cast<float>(scales[j]) - 32;
                    for (int l = 0; l < 16; l++) {
                        int e = j*16 + l;
                        int g = e/32, li = e%32;
                        int low2 = (blk.qs[(g/4)*32 + li] >> ((g%4)*2)) & 3;
                        int hbit = (blk.hmask[li] >> g) & 1;
                        int a = low2 - 4 + 4*hbit;
                        dst[i + e] = d * sc * a;
                    }
                }
            }
            return;
        }
        case GGMLType::Q5_K: {
            const block_q5_K* p = reinterpret_cast<const block_q5_K*>(w.data);
            for (int i = 0; i < n; i += 256) {
                const block_q5_K& blk = p[i/256];
                float d = fp16_to_fp32(blk.d), dmin = fp16_to_fp32(blk.dmin);
                const uint8_t* q = blk.qs;
                int is = 0;
                for (int jb = 0; jb < 256; jb += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is + 0, blk.scales, &sc, &m); float d1 = d*sc, m1 = dmin*m;
                    get_scale_min_k4(is + 1, blk.scales, &sc, &m); float d2 = d*sc, m2 = dmin*m;
                    int bit = 2 * (jb / 64);
                    for (int l = 0; l < 32; l++) {
                        int v1 = (q[l] & 0xF) + 16 * ((blk.qh[l] >> bit) & 1);
                        int v2 = (q[l] >> 4) + 16 * ((blk.qh[l] >> (bit + 1)) & 1);
                        dst[i + jb + l]      = d1 * v1 - m1;
                        dst[i + jb + 32 + l] = d2 * v2 - m2;
                    }
                    q += 32; is += 2;
                }
            }
            return;
        }
        default:
            fprintf(stderr, "dequantize: unsupported type %s\n", type_name(w.type));
            std::memset(dst, 0, sizeof(float) * n);
    }
}

} // namespace Laplace
