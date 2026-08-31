// ssm.h - Gated DeltaNet (linear attention) single-token state step
//
// Implements the recurrent form from "Gated DeltaNet: Improving Mamba2 with
// Delta Rule" (Qwen3-Next's linear-attention layer).
//
// For each layer we maintain two state buffers:
//   - conv_state   [conv_dim * (kernel-1)]  for the depthwise causal conv1d
//   - recurrent    [G * D * D]              the per-group state matrix
//
// The input projections (QKV, z gate, a/b) and the output projection are done
// by the caller, batched across tokens; deltanet_token() is the per-token
// recurrent core:
//   1. Causal depthwise conv1d over the projected QKV (kernel=4) + SiLU
//   2. Per-group: L2-normalize Q and K; scale Q by 1/sqrt(D)
//   3. decay g = exp(A * softplus(a + dt_bias)), beta = sigmoid(b)
//      (A = ssm_a as stored in GGUF, already -exp(A_log))
//   4. State update per group:
//        S = g * S;  S += outer(k, beta * (v - S^T k));  o = S^T q
//   5. Gated RMSNorm (Qwen3-Next order): norm(o) * w * silu(z)
#pragma once

#include "tensor.h"

namespace Laplace {

struct DeltaNetParams {
    int G;            // num k heads (q/k groups; v heads are num_v_heads)
    int D;            // head dim, shared by k and v heads
    int inner;        // value_dim = num_v_heads * D
    int num_v_heads;  // value heads: state, decay, and beta are per v-head
    int key_dim;      // G * D, width of the q and k sections
    int conv_kernel;  // 4
    int conv_dim;     // in_proj width: 2*key_dim + inner (+inner if z fused)
    int gate_fused;   // 1 when z is the tail of qkv_proj, 0 with attn_gate

    // Per-head learned params. A and dt_bias are length num_v_heads;
    // ssm_norm is length D (per v-head RMSNorm); conv_w is depthwise.
    const float* A;         // pre-transformed -exp(A_log), as stored in GGUF
    const float* dt_bias;
    const float* ssm_norm;  // length D
    const float* conv_w;    // [conv_dim, kernel] F32 depthwise (kernel innermost)
};

// Single-token recurrent step. Mutates conv_state and recurrent in place.
// qkv_proj [conv_dim] is the token's projected QKV ([Q | K | V | z?],
// section-major, head-major within a section); it is clobbered (conv +
// SiLU run in place). gate_proj [inner] is the projected z gate when
// gate_fused is 0. b_proj/a_proj [num_v_heads] are the projected
// write-strength/decay inputs. o_raw [inner] is scratch; the gated-normed
// output is written to o_normed [inner].
void deltanet_token(
    const DeltaNetParams& p,
    float* conv_state,      // [conv_dim * (kernel-1)]
    float* recurrent,       // [G * D * D]
    float* qkv_proj,        // [conv_dim] in/out
    const float* gate_proj, // [inner]
    const float* b_proj,    // [G]
    const float* a_proj,    // [G]
    float* o_raw,           // [G * D] scratch
    float* o_normed         // [inner] out
);

// WH-domain variant of deltanet_token.  Operates on the state in the
// Walsh-Hadamard-rotated basis: at entry, recurrent_wh is S̃ = H·S·H^T
// (the same shape, G*D*D); at exit, it holds the updated S̃.  q, k, v
// enter in real space; we rotate them to WH space in place.  The output
// o_normed is in real space (the WH-domain output is inverse-rotated in
// o_raw before the gated RMSNorm).
//
// Mathematically equivalent to deltanet_token up to FP rounding: H is
// orthonormal, so any H·X·H^T-then-update sequence is a basis change
// of the same operator.  In practice the per-element order differs, so
// results match within a few ULPs.
void deltanet_token_wh(
    const DeltaNetParams& p,
    float* conv_state,      // [conv_dim * (kernel-1)]
    float* recurrent_wh,    // [G * D * D]  in/out (WH-domain)
    float* qkv_proj,        // [conv_dim] in/out
    const float* gate_proj, // [inner]
    const float* b_proj,    // [G]
    const float* a_proj,    // [G]
    float* o_raw,           // [G * D] scratch
    float* o_normed         // [inner] out
);

} // namespace Laplace
