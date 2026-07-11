// matmul.h - GEMM dispatch over quantization formats
#pragma once

#include <cstddef>
#include <cstdint>

#include "tensor.h"

namespace Laplace {

// y[M,N] += x[M,K] @ w[K,N]  (row-major, FP32 activations and accumulators)
// w must already be cast to the right layout in the GGUF file; this routine
// just dispatches to the kernel for the weight's quantization type.
//
// We support the [1, K] x [K, N] -> [1, N] "decode" case fast (one token
// against a single row of activations), and the general [M, K] x [K, N] case
// for prefill.
void matmul_row(const float* x, const Tensor& w, float* y, int K, int N);
void matmul_rows(const float* x, const Tensor& w, float* y, int M, int K, int N);

// LM head: tries GPU first (GPU is 2.5x faster for the large vocab projection),
// falls back to CPU. Used only for the final logits projection.
void matmul_lm_head(const float* x, const Tensor& w, float* y, int M, int K, int N);

// Register mmap'd weight region for zero-copy GPU access.
void matmul_register_weights(const void* base, size_t size);

// Fused MoE GEMV with indirect expert access. expert_idx[k] selects which
// expert from the stacked weight tensor to use. y[k * N + j] = output of
// expert expert_idx[k], column j. One parallel_for for all experts.
void fused_moe_gemm_idx(const float* x, const Tensor& w, float* y,
                        const int* expert_idx, int n_experts,
                        int K, int N);

// Fused MoE GEMV with per-expert activations. x[k * K + i] is expert k's
// input. y[k * N + j] = output of expert expert_idx[k], column j.
// One parallel_for for all experts.
void fused_moe_gemm_multi(const float* x, const Tensor& w, float* y,
                          const int* expert_idx, int n_experts,
                          int K, int N);

// Standalone dequantize (for testing and verification).
// dst must hold exactly `n` floats.
void dequantize(const Tensor& w, float* dst, int n);

// Batch GEMV: dispatch multiple M=1 matmuls in one Metal command buffer.
// Falls back to sequential dispatch when Metal is unavailable.
struct MatmulBatchSpec {
    const float* x;
    const Tensor* w;
    float* y;
    int K, N;
};
bool matmul_gemm_batch(const MatmulBatchSpec* specs, int n);

} // namespace Laplace
