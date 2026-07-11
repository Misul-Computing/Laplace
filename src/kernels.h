// kernels.h - internal interface between the portable matmul front-end and
// the optional SIMD kernel back-end.
//
// matmul.cpp owns the scalar reference implementation and the dispatch; the
// SIMD back-end (matmul_simd.cpp, compiled with ARMv8.x ISA flags) exports
// whole GEMM entry points through get_simd_kernels(). The dispatch boundary
// is one indirect call per matmul - the row loops, activation quantization
// and dot kernels live together inside the back-end so they fully inline.
// LAPLACE_NOSIMD=1 forces the portable scalar path.
#pragma once

#include <cstdint>

#include "tensor.h"

namespace Laplace {
namespace kernels {

// ---- GGUF quantization block layouts (exact memory layout) -----------------

struct block_q4_0 {            // 18 B, 32 elements
    uint16_t d;
    uint8_t  qs[16];
};
static_assert(sizeof(block_q4_0) == 18, "q4_0 layout");

struct block_q5_0 {            // 22 B, 32 elements
    uint16_t d;
    uint32_t qh;               // 32 high bits, one per element
    uint8_t  qs[16];           // 4-bit low nibbles (same packing as q4_0)
} __attribute__((packed));
static_assert(sizeof(block_q5_0) == 22, "q5_0 layout");

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

struct block_q6_K {            // 210 B, 256 elements. NOTE: d is the LAST field.
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
};
static_assert(sizeof(block_q6_K) == 210, "q6_K layout");

struct block_q2_K {            // 84 B, 256 elements. NOTE: d/dmin are the LAST fields.
    uint8_t  scales[16];
    uint8_t  qs[64];
    uint16_t d;
    uint16_t dmin;
};
static_assert(sizeof(block_q2_K) == 84, "q2_K layout");

struct block_q3_K {            // 110 B, 256 elements. NOTE: d is the LAST field.
    uint8_t  hmask[32];
    uint8_t  qs[64];
    uint8_t  scales[12];
    uint16_t d;
};
static_assert(sizeof(block_q3_K) == 110, "q3_K layout");

struct block_q5_K {            // 176 B, 256 elements
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[32];
    uint8_t  qs[128];
};
static_assert(sizeof(block_q5_K) == 176, "q5_K layout");

// Extract scale and min from the K-quant packed scales array.
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// ---- back-end interface -----------------------------------------------------

// Computes y[M, N] = X[M, K] @ W^T where W is N contiguous rows of K
// elements in `type` format. Internally parallel. Returns false if
// the back-end does not handle `type` (caller falls back to scalar).
using gemm_fn = bool (*)(const float* x, const uint8_t* w, GGMLType type,
                         float* y, int M, int K, int N);

// Fused MoE GEMV with shared activation: for each expert k, column j:
//   y[k * N + j] = dot(x, w + expert_idx[k] * per_expert + j * rb, K)
// Quantizes x once, then one parallel_for across n_experts * N.
using moe_gemv_fn = bool (*)(const float* x, const uint8_t* w, GGMLType type,
                             const int* expert_idx, int n_experts,
                             float* y, int K, int N);

// Fused MoE GEMV with per-expert activations: for each expert k, column j:
//   y[k * N + j] = dot(x + k * K, w + expert_idx[k] * per_expert + j * rb, K)
// Quantizes all n_experts rows of x, then one parallel_for.
using moe_gemv_multi_fn = bool (*)(const float* x, const uint8_t* w, GGMLType type,
                                   const int* expert_idx, int n_experts,
                                   float* y, int K, int N);

// Returns the best GEMM entry point this build provides for the RUNNING
// cpu, or nullptr when the CPU (or architecture) lacks the required
// features.
gemm_fn get_simd_gemm();
moe_gemv_fn get_simd_moe_gemv();
moe_gemv_multi_fn get_simd_moe_gemv_multi();

} // namespace kernels
} // namespace Laplace
