// ops.cpp - generic bandwidth-bound kernels shared across architectures
// Apple Silicon only. NEON + FMA for float paths.
#include "ops.h"

#include <cmath>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace Laplace {

void walsh_hadamard(float* values, int n) {
    if (n <= 1) return;
    for (int width = 1; width < n; width *= 2) {
        for (int start = 0; start < n; start += 2 * width) {
            for (int index = start; index < start + width; index++) {
                float a = values[index];
                float b = values[index + width];
                values[index] = a + b;
                values[index + width] = a - b;
            }
        }
    }
    float scale = 1.0f / std::sqrt(static_cast<float>(n));
    for (int index = 0; index < n; index++) values[index] *= scale;
}

namespace ops {

namespace {
inline float silu_f(float x) { return x / (1.0f + std::exp(-x)); }
} // namespace

void rmsnorm(const float* x, const float* w, float* y, int n, float eps) {
#if defined(__aarch64__)
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        acc = vfmaq_f32(acc, v0, v0);
        acc = vfmaq_f32(acc, v1, v1);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vfmaq_f32(acc, v, v);
    }
    float ms = vaddvq_f32(acc);
    for (; i < n; i++) ms += x[i] * x[i];
    float inv = 1.0f / std::sqrt(ms / n + eps);
    float32x4_t invv = vdupq_n_f32(inv);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        float32x4_t w0 = vld1q_f32(w + i);
        float32x4_t w1 = vld1q_f32(w + i + 4);
        vst1q_f32(y + i, vmulq_f32(vmulq_f32(v0, invv), w0));
        vst1q_f32(y + i + 4, vmulq_f32(vmulq_f32(v1, invv), w1));
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t wv = vld1q_f32(w + i);
        vst1q_f32(y + i, vmulq_f32(vmulq_f32(v, invv), wv));
    }
    for (; i < n; i++) y[i] = x[i] * inv * w[i];
#else
    float ms = 0.0f;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    float inv = 1.0f / std::sqrt(ms / n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * inv * w[i];
#endif
}

void rmsnorm_rows(const float* x, const float* w, float* y, int rows, int cols, float eps) {
    for (int r = 0; r < rows; r++) {
        rmsnorm(x + static_cast<size_t>(r) * cols, w,
                y + static_cast<size_t>(r) * cols, cols, eps);
    }
}

void rmsnorm_phi3(const float* x, const float* w, float* y, int n, float eps) {
#if defined(__aarch64__)
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        acc = vfmaq_f32(acc, v0, v0);
        acc = vfmaq_f32(acc, v1, v1);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vfmaq_f32(acc, v, v);
    }
    float ms = vaddvq_f32(acc);
    for (; i < n; i++) ms += x[i] * x[i];
    float inv = 1.0f / std::sqrt(ms / n + eps);
    float32x4_t invv = vdupq_n_f32(inv);
    float32x4_t one = vdupq_n_f32(1.0f);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        float32x4_t w0 = vld1q_f32(w + i);
        float32x4_t w1 = vld1q_f32(w + i + 4);
        float32x4_t g0 = vaddq_f32(one, w0);
        float32x4_t g1 = vaddq_f32(one, w1);
        vst1q_f32(y + i, vmulq_f32(vmulq_f32(v0, invv), g0));
        vst1q_f32(y + i + 4, vmulq_f32(vmulq_f32(v1, invv), g1));
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t wv = vld1q_f32(w + i);
        float32x4_t g = vaddq_f32(one, wv);
        vst1q_f32(y + i, vmulq_f32(vmulq_f32(v, invv), g));
    }
    for (; i < n; i++) y[i] = x[i] * inv * (1.0f + w[i]);
#else
    float sumsq = 0.0f;
    for (int i = 0; i < n; i++) sumsq += x[i] * x[i];
    float inv = 1.0f / std::sqrt(sumsq / n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * inv * (1.0f + w[i]);
#endif
}

void swiglu(const float* gate, const float* up, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = silu_f(gate[i]) * up[i];
}

void geglu(const float* gate, const float* up, float* out, int n) {
    const float c = 0.7978845608028654f;  // sqrt(2/pi)
    const float c2 = 0.044715f;
#if defined(__aarch64__)
    float32x4_t cv = vdupq_n_f32(c);
    float32x4_t c2v = vdupq_n_f32(c2);
    float32x4_t half_v = vdupq_n_f32(0.5f);
    float32x4_t one_v = vdupq_n_f32(1.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t g = vld1q_f32(gate + i);
        float32x4_t u = vld1q_f32(up + i);
        float32x4_t g2 = vmulq_f32(g, g);
        float32x4_t g3 = vmulq_f32(g2, g);
        // tanh(c * (g + c2 * g3)) - use scalar tanh per element for accuracy
        float32x4_t arg = vfmaq_f32(vmulq_f32(cv, g), cv, vmulq_f32(c2v, g3));
        float args[4];
        vst1q_f32(args, arg);
        float32x4_t tanh_v = vdupq_n_f32(0.0f);
        tanh_v = vsetq_lane_f32(std::tanh(args[0]), tanh_v, 0);
        tanh_v = vsetq_lane_f32(std::tanh(args[1]), tanh_v, 1);
        tanh_v = vsetq_lane_f32(std::tanh(args[2]), tanh_v, 2);
        tanh_v = vsetq_lane_f32(std::tanh(args[3]), tanh_v, 3);
        float32x4_t gelu = vmulq_f32(half_v, vmulq_f32(g, vaddq_f32(one_v, tanh_v)));
        vst1q_f32(out + i, vmulq_f32(gelu, u));
    }
    for (; i < n; i++) {
        float g = gate[i];
        float g2 = g * g;
        float g3 = g2 * g;
        float gelu = 0.5f * g * (1.0f + std::tanh(c * (g + c2 * g3)));
        out[i] = gelu * up[i];
    }
#else
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float g2 = g * g;
        float g3 = g2 * g;
        float gelu = 0.5f * g * (1.0f + std::tanh(c * (g + 0.044715f * g3)));
        out[i] = gelu * up[i];
    }
#endif
}

void rope_apply(float* qk, int n_heads, int head_dim, int rope_pairs,
                const float* cos_ptr, const float* sin_ptr) {
    const int half = rope_pairs;
#if defined(__aarch64__)
    for (int h = 0; h < n_heads; h++) {
        float* v = qk + h * head_dim;
        int p = 0;
        for (; p + 4 <= half; p += 4) {
            float32x4_t c = vld1q_f32(cos_ptr + p);
            float32x4_t s = vld1q_f32(sin_ptr + p);
            float32x4_t a = vld1q_f32(v + p);
            float32x4_t b = vld1q_f32(v + p + half);
            // v[p]     = a*c - b*s
            // v[p+half] = a*s + b*c
            vst1q_f32(v + p,        vfmsq_f32(vmulq_f32(a, c), b, s));
            vst1q_f32(v + p + half, vfmaq_f32(vmulq_f32(a, s), b, c));
        }
        for (; p < half; p++) {
            float c = cos_ptr[p];
            float s = sin_ptr[p];
            float a = v[p];
            float b = v[p + half];
            v[p]        = a * c - b * s;
            v[p + half] = a * s + b * c;
        }
    }
#else
    for (int h = 0; h < n_heads; h++) {
        float* v = qk + h * head_dim;
        for (int p = 0; p < half; p++) {
            float c = cos_ptr[p];
            float s = sin_ptr[p];
            float a = v[p];
            float b = v[p + half];
            v[p]        = a * c - b * s;
            v[p + half] = a * s + b * c;
        }
    }
#endif
}

} // namespace ops
} // namespace Laplace
