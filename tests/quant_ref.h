// quant_ref.h - independent reference for GGUF quantization block layouts.
//
// These structs and scalar dequantizers are transcribed from the GGUF
// quantization spec, NOT from src/matmul.cpp, so the tests catch layout
// mistakes in the engine.
#pragma once

#include <cstdint>
#include <cstring>

#include "fp16.h"

namespace quant_ref {

// ---- block layouts (exact GGUF memory layout) ------------------------------

struct block_q4_0 {            // 18 B, 32 elements
    uint16_t d;
    uint8_t  qs[16];
};
static_assert(sizeof(block_q4_0) == 18, "q4_0 layout");

struct block_q8_0 {            // 34 B, 32 elements
    uint16_t d;
    int8_t   qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "q8_0 layout");

struct block_q4_K {            // 144 B, 256 elements
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(block_q4_K) == 144, "q4_K layout");

struct block_q6_K {            // 210 B, 256 elements.  NOTE: d is LAST.
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
};
static_assert(sizeof(block_q6_K) == 210, "q6_K layout");

// ---- scalar reference dequantizers ------------------------------------------

inline void dequant_q4_0(const block_q4_0* b, float* y) {
    float d = Laplace::fp16_to_fp32(b->d);
    for (int l = 0; l < 16; l++) {
        y[l +  0] = d * (static_cast<int>(b->qs[l] & 0xF) - 8);
        y[l + 16] = d * (static_cast<int>(b->qs[l] >>  4) - 8);
    }
}

inline void dequant_q8_0(const block_q8_0* b, float* y) {
    float d = Laplace::fp16_to_fp32(b->d);
    for (int l = 0; l < 32; l++) y[l] = d * b->qs[l];
}

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

inline void dequant_q4_K(const block_q4_K* b, float* y) {
    float d    = Laplace::fp16_to_fp32(b->d);
    float dmin = Laplace::fp16_to_fp32(b->dmin);
    const uint8_t* q = b->qs;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, b->scales, &sc, &m);
        float d1 = d * sc, m1 = dmin * m;
        get_scale_min_k4(is + 1, b->scales, &sc, &m);
        float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; l++) y[j + l]      = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) y[j + 32 + l] = d2 * (q[l] >>  4) - m2;
        q += 32;
        is += 2;
    }
}

inline void dequant_q6_K(const block_q6_K* b, float* y) {
    float d = Laplace::fp16_to_fp32(b->d);
    const uint8_t* ql = b->ql;
    const uint8_t* qh = b->qh;
    const int8_t*  sc = b->scales;
    for (int n = 0; n < 256; n += 128) {
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

} // namespace quant_ref
