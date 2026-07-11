// ops.h - generic bandwidth-bound kernels shared across architectures
// Apple Silicon only. NEON + FMA for float, NEON FP16 for half precision.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace Laplace {

// Orthonormal Walsh-Hadamard transform used by LaplaceKV and recurrent state.
// The transform is its own inverse. `n` must be a power of two.
void walsh_hadamard(float* values, int n);
inline void inverse_walsh_hadamard(float* values, int n) {
    walsh_hadamard(values, n);
}

namespace ops {

// SIMD dot over `n` floats and weighted axpy. Used by the attention
// QK^T and PV inner loops.
inline float dot(const float* a, const float* b, int n) {
#if defined(__aarch64__)
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4)
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    float s = vaddvq_f32(acc);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
#endif
}

inline void axpy(float* y, float w, const float* x, int n) {
#if defined(__aarch64__)
    float32x4_t wv = vdupq_n_f32(w);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t yv = vld1q_f32(y + i);
        yv = vfmaq_f32(yv, wv, vld1q_f32(x + i));
        vst1q_f32(y + i, yv);
    }
    for (; i < n; i++) y[i] += w * x[i];
#else
    for (int i = 0; i < n; i++) y[i] += w * x[i];
#endif
}

// Fused FP16 dot: dot(float a, fp16 b). Converts b on the fly via NEON.
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline float dot_f16(const float* a, const uint16_t* b, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float16x4_t bv = vld1_f16(reinterpret_cast<const __fp16*>(b + i));
        float32x4_t bv32 = vcvt_f32_f16(bv);
        acc = vfmaq_f32(acc, vld1q_f32(a + i), bv32);
    }
    float s = vaddvq_f32(acc);
    for (; i < n; i++) {
        uint16_t h = b[i];
        uint32_t f = ((h & 0x8000) << 16) | (((h & 0x7c00) + 0x1c000) << 13) | ((h & 0x03ff) << 13);
        float fv;
        std::memcpy(&fv, &f, sizeof(fv));
        s += a[i] * fv;
    }
    return s;
}

inline void axpy_f16(float* y, float w, const uint16_t* x, int n) {
    float32x4_t wv = vdupq_n_f32(w);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float16x4_t xv16 = vld1_f16(reinterpret_cast<const __fp16*>(x + i));
        float32x4_t xv = vcvt_f32_f16(xv16);
        float32x4_t yv = vld1q_f32(y + i);
        yv = vfmaq_f32(yv, wv, xv);
        vst1q_f32(y + i, yv);
    }
    for (; i < n; i++) {
        uint16_t h = x[i];
        uint32_t f = ((h & 0x8000) << 16) | (((h & 0x7c00) + 0x1c000) << 13) | ((h & 0x03ff) << 13);
        float fv;
        std::memcpy(&fv, &f, sizeof(fv));
        y[i] += w * fv;
    }
}
#else
// Scalar fallback: no NEON FP16.
inline float dot_f16(const float* a, const uint16_t* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        uint16_t h = b[i];
        uint32_t f = ((h & 0x8000) << 16) | (((h & 0x7c00) + 0x1c000) << 13) | ((h & 0x03ff) << 13);
        float fv;
        std::memcpy(&fv, &f, sizeof(fv));
        s += a[i] * fv;
    }
    return s;
}
inline void axpy_f16(float* y, float w, const uint16_t* x, int n) {
    for (int i = 0; i < n; i++) {
        uint16_t h = x[i];
        uint32_t f = ((h & 0x8000) << 16) | (((h & 0x7c00) + 0x1c000) << 13) | ((h & 0x03ff) << 13);
        float fv;
        std::memcpy(&fv, &f, sizeof(fv));
        y[i] += w * fv;
    }
}
#endif

// RMSNorm: y[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]
void rmsnorm(const float* x, const float* w, float* y, int n, float eps);

// Phi3-style RMSNorm: same as rmsnorm but the gain is (1 + w[i])
// instead of w[i].  Phi3-mini uses this trick to keep the weight
// initialization near zero; not equivalent to a standard RMSNorm
// at the same w.
void rmsnorm_phi3(const float* x, const float* w, float* y, int n, float eps);

// Row-wise RMSNorm for a row-major [rows, cols] matrix.
void rmsnorm_rows(const float* x, const float* w, float* y, int rows, int cols, float eps);

// SwiGLU elementwise: out[i] = silu(gate[i]) * up[i]
void swiglu(const float* gate, const float* up, float* out, int n);

// GeGLU elementwise: out[i] = gelu_tanh(gate[i]) * up[i]
void geglu(const float* gate, const float* up, float* out, int n);

// Neox-style RoPE applied to the first 2*rope_pairs dims of each head.
// qk is [n_heads, head_dim]; cos_ptr/sin_ptr each contain rope_pairs entries
// for the target position.
void rope_apply(float* qk, int n_heads, int head_dim, int rope_pairs,
                const float* cos_ptr, const float* sin_ptr);

} // namespace ops
} // namespace Laplace
