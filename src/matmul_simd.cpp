// matmul_simd.cpp - ARM64 NEON (+ DOTPROD) kernel back-end.
//
// Apple Silicon only. Compiled with -march=armv8.5-a+dotprod+fp16+i8mm+bf16.
// NEON is 128-bit, so we process 16 elements per int8 dot. DOTPROD (vdotq_s32)
// fuses the int8 multiply-add chain into one instruction per 16 elements.
// FP16 vector arithmetic and I8MM are also available.

#include "kernels.h"

#include <arm_neon.h>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__aarch64__)

#include "threadpool.h"
#include "fp16.h"

namespace Laplace {
namespace kernels {

namespace {

// Quantized-activation block layout for the NEON back-end.
struct ActQ8 {
    float   d;
    float   sum;
    int16_t sum_lo;
    int16_t sum_hi;
    int8_t  qs[32];
};

// ---------------- helpers ---------------------------------------------------

inline float hsum_f32x4(float32x4_t v) {
    return vaddvq_f32(v);
}

// fp16 -> fp32 via the software conversion in fp16.h. Called once per block
// (32 or 256 elements), so the cost is negligible vs the dot product itself.
inline float fp16_scale(uint16_t h) {
    return fp16_to_fp32(h);
}

// 16 signed int8 dot 16 signed int8 -> 4 x int32 (each lane sums 4 products).
#if defined(__ARM_FEATURE_DOTPROD)
inline int32x4_t dot16_i8(int8x16_t a, int8x16_t b) {
    return vdotq_s32(vdupq_n_s32(0), a, b);
}
#else
inline int32x4_t dot16_i8(int8x16_t a, int8x16_t b) {
    int16x8_t a0 = vmovl_s8(vget_low_s8(a));
    int16x8_t a1 = vmovl_s8(vget_high_s8(a));
    int16x8_t b0 = vmovl_s8(vget_low_s8(b));
    int16x8_t b1 = vmovl_s8(vget_high_s8(b));
    int16x8_t p0 = vmulq_s16(a0, b0);
    int16x8_t p1 = vmulq_s16(a1, b1);
    int32x4_t r = vpaddlq_s16(p0);
    r = vpadalq_s16(r, p1);
    return r;
}
#endif

// 16 unsigned bytes dot 16 signed int8 -> 4 x int32 (each lane sums 4 products).
// I8MM's vusdotq_s32 handles the mixed-sign multiply natively. With DOTPROD
// only, we reinterpret the unsigned bytes as signed (valid while values
// stay in 0..127) and use vdotq_s32.
#if defined(__ARM_FEATURE_MATMUL_INT8)
inline int32x4_t dot16_u8i8(uint8x16_t a, int8x16_t b) {
    return vusdotq_s32(vdupq_n_s32(0), a, b);
}
#elif defined(__ARM_FEATURE_DOTPROD)
inline int32x4_t dot16_u8i8(uint8x16_t a, int8x16_t b) {
    return vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(a), b);
}
#else
inline int32x4_t dot16_u8i8(uint8x16_t a, int8x16_t b) {
    int16x8_t a0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a)));
    int16x8_t a1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a)));
    int16x8_t b0 = vmovl_s8(vget_low_s8(b));
    int16x8_t b1 = vmovl_s8(vget_high_s8(b));
    int16x8_t p0 = vmulq_s16(a0, b0);
    int16x8_t p1 = vmulq_s16(a1, b1);
    int32x4_t r = vpaddlq_s16(p0);
    r = vpadalq_s16(r, p1);
    return r;
}
#endif

// ---------------- float-activation dot products -----------------------------

float dot_row_f32_neon(const float* x, const uint8_t* row, int K) {
    const float* w = reinterpret_cast<const float*>(row);
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(x + k),      vld1q_f32(w + k));
        acc0 = vfmaq_f32(acc0, vld1q_f32(x + k + 4),  vld1q_f32(w + k + 4));
        acc1 = vfmaq_f32(acc1, vld1q_f32(x + k + 8),  vld1q_f32(w + k + 8));
        acc1 = vfmaq_f32(acc1, vld1q_f32(x + k + 12), vld1q_f32(w + k + 12));
    }
    float acc = hsum_f32x4(vaddq_f32(acc0, acc1));
    for (; k < K; k++) acc += x[k] * w[k];
    return acc;
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
float dot_row_f16_neon(const float* x, const uint8_t* row, int K) {
    const uint16_t* w = reinterpret_cast<const uint16_t*>(row);
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        float32x4_t w0 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(w + k)));
        float32x4_t w1 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(w + k + 4)));
        acc0 = vfmaq_f32(acc0, vld1q_f32(x + k),     w0);
        acc1 = vfmaq_f32(acc1, vld1q_f32(x + k + 4), w1);
    }
    float acc = hsum_f32x4(vaddq_f32(acc0, acc1));
    for (; k < K; k++) acc += x[k] * fp16_to_fp32(w[k]);
    return acc;
}
#endif

float dot_row_bf16_neon(const float* x, const uint8_t* row, int K) {
    const uint16_t* w = reinterpret_cast<const uint16_t*>(row);
    float32x4_t acc = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        // BF16 is the top 16 bits of FP32: widen u16 to u32 then shift left 16.
        uint32x4_t f = vshlq_n_u32(vmovl_u16(vld1_u16(w + k)), 16);
        acc = vfmaq_f32(acc, vld1q_f32(x + k), vreinterpretq_f32_u32(f));
    }
    float r = hsum_f32x4(acc);
    for (; k < K; k++) r += x[k] * bf16_to_fp32(w[k]);
    return r;
}

// ---------------- int8-activation dot products ------------------------------

float dot_row_q8_0_int_neon(const ActQ8* xa, const uint8_t* row, int K) {
    const block_q8_0* w = reinterpret_cast<const block_q8_0*>(row);
    const int n_blocks = K / 32;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (int b = 0; b < n_blocks; b++) {
        int8x16_t wv0 = vld1q_s8(w[b].qs);
        int8x16_t wv1 = vld1q_s8(w[b].qs + 16);
        int8x16_t xv0 = vld1q_s8(xa[b].qs);
        int8x16_t xv1 = vld1q_s8(xa[b].qs + 16);
        float32x4_t pf0 = vcvtq_f32_s32(dot16_i8(wv0, xv0));
        float32x4_t pf1 = vcvtq_f32_s32(dot16_i8(wv1, xv1));
        float scale = fp16_scale(w[b].d) * xa[b].d;
        acc0 = vfmaq_f32(acc0, vdupq_n_f32(scale), pf0);
        acc1 = vfmaq_f32(acc1, vdupq_n_f32(scale), pf1);
    }
    return hsum_f32x4(vaddq_f32(acc0, acc1));
}

float dot_row_q4_0_int_neon(const ActQ8* xa, const uint8_t* row, int K) {
    const block_q4_0* w = reinterpret_cast<const block_q4_0*>(row);
    const int n_blocks = K / 32;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    float32x4_t acc = vdupq_n_f32(0.0f);
    float acc_offset = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        uint8x16_t q  = vld1q_u8(w[b].qs);
        uint8x16_t lo = vandq_u8(q, m4);
        uint8x16_t hi = vandq_u8(vshrq_n_u8(q, 4), m4);
        int8x16_t xv0 = vld1q_s8(xa[b].qs);
        int8x16_t xv1 = vld1q_s8(xa[b].qs + 16);
        float32x4_t pf0 = vcvtq_f32_s32(dot16_u8i8(lo, xv0));
        float32x4_t pf1 = vcvtq_f32_s32(dot16_u8i8(hi, xv1));
        float scale = fp16_scale(w[b].d) * xa[b].d;
        acc = vfmaq_f32(acc, vdupq_n_f32(scale), pf0);
        acc = vfmaq_f32(acc, vdupq_n_f32(scale), pf1);
        acc_offset += fp16_scale(w[b].d) * 8.0f * xa[b].sum;
    }
    return hsum_f32x4(acc) - acc_offset;
}

float dot_row_q5_0_int_neon(const ActQ8* xa, const uint8_t* row, int K) {
    const block_q5_0* w = reinterpret_cast<const block_q5_0*>(row);
    const int n_blocks = K / 32;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int8x16_t  m16 = vdupq_n_s8(16);
    // Variable right-shift amounts [0..7] for extracting individual bits.
    // vshl_u8 with a negative shift performs a logical right shift.
    const int8x8_t   shr = {0, -1, -2, -3, -4, -5, -6, -7};
    const uint8x8_t  m1 = vdup_n_u8(1);
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int b = 0; b < n_blocks; b++) {
        // 4-bit nibbles from qs (same packing as Q4_0)
        uint8x16_t q  = vld1q_u8(w[b].qs);
        uint8x16_t lo = vandq_u8(q, m4);
        uint8x16_t hi = vandq_u8(vshrq_n_u8(q, 4), m4);

        // Extract the 5th bit from qh. qh byte 0 holds bits 0..7, byte 1
        // holds bits 8..15, etc. lo (elements 0..15) uses qh bits 0..15;
        // hi (elements 16..31) uses bits 16..31. Each bit is shifted to
        // position 4 and OR'd into the 4-bit nibble.
        uint8x8_t qh8 = vcreate_u8(static_cast<uint64_t>(w[b].qh));
        uint8x8_t h0_lo = vshl_n_u8(vand_u8(vshl_u8(vdup_lane_u8(qh8, 0), shr), m1), 4);
        uint8x8_t h0_hi = vshl_n_u8(vand_u8(vshl_u8(vdup_lane_u8(qh8, 1), shr), m1), 4);
        uint8x8_t h1_lo = vshl_n_u8(vand_u8(vshl_u8(vdup_lane_u8(qh8, 2), shr), m1), 4);
        uint8x8_t h1_hi = vshl_n_u8(vand_u8(vshl_u8(vdup_lane_u8(qh8, 3), shr), m1), 4);
        lo = vorrq_u8(lo, vcombine_u8(h0_lo, h0_hi));
        hi = vorrq_u8(hi, vcombine_u8(h1_lo, h1_hi));

        // Subtract 16 (Q5_0 zero point) and dot with int8 activations
        int8x16_t wv0 = vsubq_s8(vreinterpretq_s8_u8(lo), m16);
        int8x16_t wv1 = vsubq_s8(vreinterpretq_s8_u8(hi), m16);
        int8x16_t xv0 = vld1q_s8(xa[b].qs);
        int8x16_t xv1 = vld1q_s8(xa[b].qs + 16);
        float32x4_t pf0 = vcvtq_f32_s32(dot16_i8(wv0, xv0));
        float32x4_t pf1 = vcvtq_f32_s32(dot16_i8(wv1, xv1));
        float scale = fp16_scale(w[b].d) * xa[b].d;
        acc = vfmaq_f32(acc, vdupq_n_f32(scale), pf0);
        acc = vfmaq_f32(acc, vdupq_n_f32(scale), pf1);
    }
    return hsum_f32x4(acc);
}

float dot_row_q4_k_int_neon(const ActQ8* xa, const uint8_t* row, int K) {
    const block_q4_K* w = reinterpret_cast<const block_q4_K*>(row);
    const int n_blocks = K / 256;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    // Dual accumulators to break the FMA dependency chain and use both
    // FMA units on Apple Silicon P-cores.
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float acc_min = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q4_K& blk = w[b];
        const ActQ8* xab = xa + b * 8;   // 8 act blocks of 32 per superblock
        float d    = fp16_scale(blk.d);
        float dmin = fp16_scale(blk.dmin);
        const uint8_t* q = blk.qs;
        int is = 0;
        for (int jb = 0; jb < 4; jb++) {           // 4 chunks of 64 elements
            uint8_t sc, m;
            get_scale_min_k4(is + 0, blk.scales, &sc, &m);
            float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, blk.scales, &sc, &m);
            float d2 = d * sc, m2 = dmin * m;
            const ActQ8& a1 = xab[jb * 2 + 0];     // lo-nibble elements
            const ActQ8& a2 = xab[jb * 2 + 1];     // hi-nibble elements

            uint8x16_t q0 = vld1q_u8(q);
            uint8x16_t q1 = vld1q_u8(q + 16);
            uint8x16_t wlo0 = vandq_u8(q0, m4);
            uint8x16_t wlo1 = vandq_u8(q1, m4);
            uint8x16_t whi0 = vandq_u8(vshrq_n_u8(q0, 4), m4);
            uint8x16_t whi1 = vandq_u8(vshrq_n_u8(q1, 4), m4);

            int8x16_t x1_0 = vld1q_s8(a1.qs);
            int8x16_t x1_1 = vld1q_s8(a1.qs + 16);
            int8x16_t x2_0 = vld1q_s8(a2.qs);
            int8x16_t x2_1 = vld1q_s8(a2.qs + 16);

            // Nibbles are unsigned 0..15, activations are signed int8.
            // dot16_u8i8 handles the mixed-sign multiply.
            float32x4_t pf1_0 = vcvtq_f32_s32(dot16_u8i8(wlo0, x1_0));
            float32x4_t pf1_1 = vcvtq_f32_s32(dot16_u8i8(wlo1, x1_1));
            float32x4_t pf2_0 = vcvtq_f32_s32(dot16_u8i8(whi0, x2_0));
            float32x4_t pf2_1 = vcvtq_f32_s32(dot16_u8i8(whi1, x2_1));

            float scale1 = d1 * a1.d;
            float scale2 = d2 * a2.d;
            // Alternate accumulators to break dependency chain.
            acc0 = vfmaq_f32(acc0, vdupq_n_f32(scale1), pf1_0);
            acc1 = vfmaq_f32(acc1, vdupq_n_f32(scale1), pf1_1);
            acc0 = vfmaq_f32(acc0, vdupq_n_f32(scale2), pf2_0);
            acc1 = vfmaq_f32(acc1, vdupq_n_f32(scale2), pf2_1);
            acc_min += m1 * a1.sum + m2 * a2.sum;
            q += 32;
            is += 2;
        }
    }
    return hsum_f32x4(vaddq_f32(acc0, acc1)) - acc_min;
}

float dot_row_q6_k_int_neon(const ActQ8* xa, const uint8_t* row, int K) {
    const block_q6_K* w = reinterpret_cast<const block_q6_K*>(row);
    const int n_blocks = K / 256;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const uint8x16_t m2 = vdupq_n_u8(0x03);
    const int8x16_t  b32 = vdupq_n_s8(32);
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (int b = 0; b < n_blocks; b++) {
        const block_q6_K& blk = w[b];
        float d = fp16_scale(blk.d);
        const uint8_t* ql = blk.ql;
        const uint8_t* qh = blk.qh;
        const int8_t*  sc = blk.scales;
        const ActQ8*   xb = xa + b * 8;
        for (int half = 0; half < 2; half++) {
            for (int g = 0; g < 4; g++) {
                uint8x16_t lo0, lo1;
                if (g == 0 || g == 2) {
                    lo0 = vld1q_u8(ql);
                    lo1 = vld1q_u8(ql + 16);
                } else {
                    lo0 = vld1q_u8(ql + 32);
                    lo1 = vld1q_u8(ql + 48);
                }
                if (g >= 2) {
                    lo0 = vshrq_n_u8(lo0, 4);
                    lo1 = vshrq_n_u8(lo1, 4);
                }
                lo0 = vandq_u8(lo0, m4);
                lo1 = vandq_u8(lo1, m4);
                int8x16_t neg_shift = vdupq_n_s8(static_cast<int8_t>(-(2 * g)));
                uint8x16_t h0 = vandq_u8(vshlq_u8(vld1q_u8(qh),      neg_shift), m2);
                uint8x16_t h1 = vandq_u8(vshlq_u8(vld1q_u8(qh + 16), neg_shift), m2);
                int8x16_t w0 = vsubq_s8(
                    vreinterpretq_s8_u8(vorrq_u8(lo0, vshlq_n_u8(h0, 4))), b32);
                int8x16_t w1 = vsubq_s8(
                    vreinterpretq_s8_u8(vorrq_u8(lo1, vshlq_n_u8(h1, 4))), b32);

                const ActQ8& a = xb[half * 4 + g];
                int8x16_t xv0 = vld1q_s8(a.qs);
                int8x16_t xv1 = vld1q_s8(a.qs + 16);
                int32x4_t p0 = dot16_i8(w0, xv0);
                int32x4_t p1 = dot16_i8(w1, xv1);
                float32x4_t pf0 = vcvtq_f32_s32(p0);
                float32x4_t pf1 = vcvtq_f32_s32(p1);
                float scale0 = d * a.d * static_cast<float>(sc[2 * g]);
                float scale1 = d * a.d * static_cast<float>(sc[2 * g + 1]);
                acc0 = vfmaq_f32(acc0, vdupq_n_f32(scale0), pf0);
                acc1 = vfmaq_f32(acc1, vdupq_n_f32(scale1), pf1);
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return hsum_f32x4(vaddq_f32(acc0, acc1));
}

// ---------------- activation quantization (NEON) ----------------------------
// Round-to-nearest-even via vcvtnq_s32_f32, packed to int8 with the
// saturating-narrow sequence (NEON vqmovn_s32 path).

void quantize_act_q8_neon(const float* x, int K, ActQ8* out) {
    const int n_blocks = K / 32;
    for (int b = 0; b < n_blocks; b++) {
        const float* xb = x + b * 32;
        ActQ8& a = out[b];
        float32x4_t v0 = vld1q_f32(xb +  0);
        float32x4_t v1 = vld1q_f32(xb +  4);
        float32x4_t v2 = vld1q_f32(xb +  8);
        float32x4_t v3 = vld1q_f32(xb + 12);
        float32x4_t v4 = vld1q_f32(xb + 16);
        float32x4_t v5 = vld1q_f32(xb + 20);
        float32x4_t v6 = vld1q_f32(xb + 24);
        float32x4_t v7 = vld1q_f32(xb + 28);

        // amax = max(|x|) across 32 floats
        float32x4_t mx = vmaxq_f32(
            vmaxq_f32(vmaxq_f32(vabsq_f32(v0), vabsq_f32(v1)),
                      vmaxq_f32(vabsq_f32(v2), vabsq_f32(v3))),
            vmaxq_f32(vmaxq_f32(vabsq_f32(v4), vabsq_f32(v5)),
                      vmaxq_f32(vabsq_f32(v6), vabsq_f32(v7))));
        float amax = vmaxvq_f32(mx);

        float d  = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        a.d = d;
        float32x4_t vid = vdupq_n_f32(id);
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(v0, vid));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(v1, vid));
        int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(v2, vid));
        int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(v3, vid));
        int32x4_t i4 = vcvtnq_s32_f32(vmulq_f32(v4, vid));
        int32x4_t i5 = vcvtnq_s32_f32(vmulq_f32(v5, vid));
        int32x4_t i6 = vcvtnq_s32_f32(vmulq_f32(v6, vid));
        int32x4_t i7 = vcvtnq_s32_f32(vmulq_f32(v7, vid));

        // Block sum for K-quant min correction.
        int32x4_t s0 = vaddq_s32(vaddq_s32(i0, i1), vaddq_s32(i2, i3));
        int32x4_t s1 = vaddq_s32(vaddq_s32(i4, i5), vaddq_s32(i6, i7));
        a.sum_lo = static_cast<int16_t>(vaddvq_s32(s0));
        a.sum_hi = static_cast<int16_t>(vaddvq_s32(s1));
        a.sum = d * static_cast<float>(a.sum_lo + a.sum_hi);

        // Pack int32 -> int16 -> int8 (saturating, values in [-127, 127]).
        int16x4_t p0 = vqmovn_s32(i0);
        int16x4_t p1 = vqmovn_s32(i1);
        int16x4_t p2 = vqmovn_s32(i2);
        int16x4_t p3 = vqmovn_s32(i3);
        int16x4_t p4 = vqmovn_s32(i4);
        int16x4_t p5 = vqmovn_s32(i5);
        int16x4_t p6 = vqmovn_s32(i6);
        int16x4_t p7 = vqmovn_s32(i7);
        int16x8_t q0 = vcombine_s16(p0, p1);
        int16x8_t q1 = vcombine_s16(p2, p3);
        int16x8_t q2 = vcombine_s16(p4, p5);
        int16x8_t q3 = vcombine_s16(p6, p7);
        int8x8_t  r0 = vqmovn_s16(q0);
        int8x8_t  r1 = vqmovn_s16(q1);
        int8x8_t  r2 = vqmovn_s16(q2);
        int8x8_t  r3 = vqmovn_s16(q3);
        vst1q_s8(a.qs,      vcombine_s8(r0, r1));
        vst1q_s8(a.qs + 16, vcombine_s8(r2, r3));
    }
}

// ---------------- GEMM row loops --------------------------------------------
// Quantize all activation rows once,
// then parallelize across output columns (N). The 4-row micro-kernels are
// omitted (DOT4 = nullptr); the single-row path is correct, just not optimal
// for batched matmuls.

const ActQ8* quantize_all(const float* x, int M, int K) {
    static thread_local std::vector<ActQ8> scratch;
    const int bpr = K / 32;
    if ((int)scratch.size() < M * bpr) scratch.resize(static_cast<size_t>(M) * bpr);
    for (int m = 0; m < M; m++) {
        quantize_act_q8_neon(x + static_cast<size_t>(m) * K, K,
                             scratch.data() + static_cast<size_t>(m) * bpr);
    }
    return scratch.data();
}

using dot_int_fn = float (*)(const ActQ8*, const uint8_t*, int);
using dot_f_fn   = float (*)(const float*,    const uint8_t*, int);

template <dot_int_fn DOT>
void gemm_int(const float* x, const uint8_t* data, float* y,
              int M, int K, int N, size_t rb) {
    const ActQ8* xa = quantize_all(x, M, K);
    const int bpr = K / 32;
    constexpr int N_DST = 2;
    auto body = [&](int b) {
        const int j0 = b * N_DST;
        const int j_end = j0 + N_DST < N ? j0 + N_DST : N;
        for (int j = j0; j < j_end; j++) {
            const uint8_t* row = data + static_cast<size_t>(j) * rb;
            for (int m = 0; m < M; m++) {
                y[static_cast<size_t>(m) * N + j] =
                    DOT(xa + static_cast<size_t>(m) * bpr, row, K);
            }
        }
    };
    const int n_blocks = (N + N_DST - 1) / N_DST;
    if (M > 1 || N >= 128) {
        ThreadPool::get().parallel_for(n_blocks, body);
    } else {
        for (int b = 0; b < n_blocks; b++) body(b);
    }
}

template <dot_f_fn DOT>
void gemm_f(const float* x, const uint8_t* data, float* y,
            int M, int K, int N, size_t rb) {
    constexpr int N_DST = 2;
    auto body = [&](int b) {
        const int j0 = b * N_DST;
        const int j_end = j0 + N_DST < N ? j0 + N_DST : N;
        for (int j = j0; j < j_end; j++) {
            const uint8_t* row = data + static_cast<size_t>(j) * rb;
            for (int m = 0; m < M; m++) {
                y[static_cast<size_t>(m) * N + j] =
                    DOT(x + static_cast<size_t>(m) * K, row, K);
            }
        }
    };
    const int n_blocks = (N + N_DST - 1) / N_DST;
    if (M > 1 || N >= 128) {
        ThreadPool::get().parallel_for(n_blocks, body);
    } else {
        for (int b = 0; b < n_blocks; b++) body(b);
    }
}

bool gemm_simd(const float* x, const uint8_t* w, GGMLType type,
               float* y, int M, int K, int N) {
    const size_t rb = static_cast<size_t>(K) / elements_per_block(type) * bytes_per_block(type);
    switch (type) {
        case GGMLType::Q8_0: gemm_int<dot_row_q8_0_int_neon>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q4_0: gemm_int<dot_row_q4_0_int_neon>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q5_0: gemm_int<dot_row_q5_0_int_neon>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q4_K: gemm_int<dot_row_q4_k_int_neon>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q6_K: gemm_int<dot_row_q6_k_int_neon>(x, w, y, M, K, N, rb); return true;
        case GGMLType::F32:  gemm_f<dot_row_f32_neon>(x, w, y, M, K, N, rb); return true;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        case GGMLType::F16:  gemm_f<dot_row_f16_neon>(x, w, y, M, K, N, rb); return true;
#endif
        case GGMLType::BF16: gemm_f<dot_row_bf16_neon>(x, w, y, M, K, N, rb); return true;
        default: return false;
    }
}

} // namespace

gemm_fn get_simd_gemm() {
    return gemm_simd;
}

// Fused MoE GEMV with shared activation (gate_up case): all experts see
// the same input x. Quantizes x once, then one parallel_for.
template <dot_int_fn DOT>
bool moe_gemv_int(const float* x, const uint8_t* w, GGMLType type,
                  const int* expert_idx, int n_experts,
                  float* y, int K, int N) {
    const ActQ8* xa = quantize_all(x, 1, K);
    const size_t rb = static_cast<size_t>(K) / elements_per_block(type) * bytes_per_block(type);
    const size_t per_expert = static_cast<size_t>(N) * rb;
    auto body = [&](int idx) {
        int k = idx / N;
        int j = idx % N;
        const uint8_t* row = w + static_cast<size_t>(expert_idx[k]) * per_expert
                              + static_cast<size_t>(j) * rb;
        y[static_cast<size_t>(k) * N + j] = DOT(xa, row, K);
    };
    ThreadPool::get().parallel_for(n_experts * N, body);
    return true;
}

// Fused MoE GEMV with per-expert activations (down case): each expert k
// has its own input at x[k * K]. Quantizes all n_experts rows once,
// then one parallel_for with the correct activation per expert.
template <dot_int_fn DOT>
bool moe_gemv_multi_int(const float* x, const uint8_t* w, GGMLType type,
                        const int* expert_idx, int n_experts,
                        float* y, int K, int N) {
    const ActQ8* xa = quantize_all(x, n_experts, K);
    const int bpr = K / 32;
    const size_t rb = static_cast<size_t>(K) / elements_per_block(type) * bytes_per_block(type);
    const size_t per_expert = static_cast<size_t>(N) * rb;
    auto body = [&](int idx) {
        int k = idx / N;
        int j = idx % N;
        const uint8_t* row = w + static_cast<size_t>(expert_idx[k]) * per_expert
                              + static_cast<size_t>(j) * rb;
        y[static_cast<size_t>(k) * N + j] = DOT(xa + k * bpr, row, K);
    };
    ThreadPool::get().parallel_for(n_experts * N, body);
    return true;
}

template <dot_f_fn DOT>
bool moe_gemv_f(const float* x, const uint8_t* w, GGMLType type,
                const int* expert_idx, int n_experts,
                float* y, int K, int N) {
    const size_t rb = static_cast<size_t>(K) / elements_per_block(type) * bytes_per_block(type);
    const size_t per_expert = static_cast<size_t>(N) * rb;
    auto body = [&](int idx) {
        int k = idx / N;
        int j = idx % N;
        const uint8_t* row = w + static_cast<size_t>(expert_idx[k]) * per_expert
                              + static_cast<size_t>(j) * rb;
        y[static_cast<size_t>(k) * N + j] = DOT(x, row, K);
    };
    ThreadPool::get().parallel_for(n_experts * N, body);
    return true;
}

template <dot_f_fn DOT>
bool moe_gemv_multi_f(const float* x, const uint8_t* w, GGMLType type,
                      const int* expert_idx, int n_experts,
                      float* y, int K, int N) {
    const size_t rb = static_cast<size_t>(K) / elements_per_block(type) * bytes_per_block(type);
    const size_t per_expert = static_cast<size_t>(N) * rb;
    auto body = [&](int idx) {
        int k = idx / N;
        int j = idx % N;
        const uint8_t* row = w + static_cast<size_t>(expert_idx[k]) * per_expert
                              + static_cast<size_t>(j) * rb;
        y[static_cast<size_t>(k) * N + j] = DOT(x + k * K, row, K);
    };
    ThreadPool::get().parallel_for(n_experts * N, body);
    return true;
}

bool moe_gemv_simd(const float* x, const uint8_t* w, GGMLType type,
                   const int* expert_idx, int n_experts,
                   float* y, int K, int N) {
    switch (type) {
        case GGMLType::Q8_0: return moe_gemv_int<dot_row_q8_0_int_neon>(x, w, type, expert_idx, n_experts, y, K, N);
        case GGMLType::Q4_0: return moe_gemv_int<dot_row_q4_0_int_neon>(x, w, type, expert_idx, n_experts, y, K, N);
        case GGMLType::Q5_0: return moe_gemv_int<dot_row_q5_0_int_neon>(x, w, type, expert_idx, n_experts, y, K, N);
        case GGMLType::Q4_K: return moe_gemv_int<dot_row_q4_k_int_neon>(x, w, type, expert_idx, n_experts, y, K, N);
        case GGMLType::Q6_K: return moe_gemv_int<dot_row_q6_k_int_neon>(x, w, type, expert_idx, n_experts, y, K, N);
        case GGMLType::F32:  return moe_gemv_f<dot_row_f32_neon>(x, w, type, expert_idx, n_experts, y, K, N);
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        case GGMLType::F16:  return moe_gemv_f<dot_row_f16_neon>(x, w, type, expert_idx, n_experts, y, K, N);
#endif
        case GGMLType::BF16: return moe_gemv_f<dot_row_bf16_neon>(x, w, type, expert_idx, n_experts, y, K, N);
        default: return false;
    }
}

bool moe_gemv_multi_simd(const float* x, const uint8_t* w, GGMLType type,
                         const int* expert_idx, int n_experts,
                         float* y, int K, int N) {
    switch (type) {
        case GGMLType::Q8_0: return moe_gemv_multi_int<dot_row_q8_0_int_neon>(x, w, type, expert_idx, n_experts, y, K, N);
        case GGMLType::Q4_0: return moe_gemv_multi_int<dot_row_q4_0_int_neon>(x, w, type, expert_idx, n_experts, y, K, N);
        case GGMLType::Q5_0: return moe_gemv_multi_int<dot_row_q5_0_int_neon>(x, w, type, expert_idx, n_experts, y, K, N);
        case GGMLType::Q4_K: return moe_gemv_multi_int<dot_row_q4_k_int_neon>(x, w, type, expert_idx, n_experts, y, K, N);
        case GGMLType::Q6_K: return moe_gemv_multi_int<dot_row_q6_k_int_neon>(x, w, type, expert_idx, n_experts, y, K, N);
        case GGMLType::F32:  return moe_gemv_multi_f<dot_row_f32_neon>(x, w, type, expert_idx, n_experts, y, K, N);
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        case GGMLType::F16:  return moe_gemv_multi_f<dot_row_f16_neon>(x, w, type, expert_idx, n_experts, y, K, N);
#endif
        case GGMLType::BF16: return moe_gemv_multi_f<dot_row_bf16_neon>(x, w, type, expert_idx, n_experts, y, K, N);
        default: return false;
    }
}

moe_gemv_fn get_simd_moe_gemv() {
    return moe_gemv_simd;
}

moe_gemv_multi_fn get_simd_moe_gemv_multi() {
    return moe_gemv_multi_simd;
}

} // namespace kernels
} // namespace Laplace

#else  // non-AArch64, or built without SIMD support

namespace Laplace {
namespace kernels {

gemm_fn get_simd_gemm() { return nullptr; }
moe_gemv_fn get_simd_moe_gemv() { return nullptr; }
moe_gemv_multi_fn get_simd_moe_gemv_multi() { return nullptr; }

} // namespace kernels
} // namespace Laplace

#endif
