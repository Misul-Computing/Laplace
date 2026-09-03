static const char* src_prefill_f16 = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void prefill_f16_rows(
    device const half* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& M [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    const uint total = uint(M) * uint(N);
    if (index >= total || K <= 0 || N <= 0 || M != 2) return;
    const uint row = index / uint(N);
    const uint column = index - row * uint(N);
    float sum = 0.0f;
    const device half* weight_row = weights + uint64_t(column) * uint(K);
    const device float* input_row = input + uint64_t(row) * uint(K);
    for (int k = 0; k < K; ++k) sum += float(weight_row[k]) * input_row[k];
    output[index] = sum;
}
)METAL"
// Tiled batched prefill projection. One threadgroup (a single simdgroup of
// 32 lanes) computes a 32-row x 32-column output tile: activations for one
// k-block are staged cooperatively in threadgroup memory, so every weight
// element is fetched once per tile instead of once per output row, and each
// lane keeps its weight slice in registers as 128-bit vectors. Requires
// K % 8 == 0 (self-guarded and admitted host-side); M and N need no
// alignment. Edge tiles are zero-padded, and the accumulation order over k
// matches the scalar kernel, so results are bit-identical for the same M.
R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void prefill_f16_tile(
    device const half* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& M [[buffer(5)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (K <= 0 || N <= 0 || M <= 0) return;
    // 128-bit vector casts below require this; admission also enforces it.
    if (uint(K) % 8u != 0u) return;
    const uint row_start = tg_id.y * 32u;
    const uint col_start = tg_id.x * 32u;
    // Whole-tile guard: uniform across all lanes so barrier stays collective.
    if (row_start >= uint(M) || col_start >= uint(N)) return;
    const uint column = col_start + lane;
    const bool column_valid = column < uint(N);
    threadgroup float stage[32][32];
    float acc[32];
    #pragma unroll
    for (uint r = 0; r < 32u; ++r) acc[r] = 0.0f;

    const uint k_blocks = (uint(K) + 31u) / 32u;
    const uint64_t weight_row =
        column_valid ? uint64_t(column) * uint64_t(uint(K)) : 0;

    for (uint kb = 0; kb < k_blocks; ++kb) {
        const uint k_base = kb * 32u;
        // Cooperative activation stage: each lane owns stage row `lane`,
        // reading a contiguous 128-byte run of the input row.
        const uint64_t source_row = uint64_t(row_start + lane) * uint64_t(uint(K));
        const bool row_valid = row_start + lane < uint(M);
        if (row_valid) {
            const device const float4* src =
                reinterpret_cast<device const float4*>(input + source_row + k_base);
            threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(stage[lane]);
            #pragma unroll
            for (uint v = 0; v < 8u; ++v) {
                if (k_base + v * 4u + 4u <= uint(K)) {
                    dst[v] = src[v];
                } else {
                    #pragma unroll
                    for (uint e = 0; e < 4u; ++e)
                        stage[lane][v * 4u + e] =
                            (k_base + v * 4u + e < uint(K))
                                ? input[source_row + k_base + v * 4u + e] : 0.0f;
                }
            }
        } else {
            #pragma unroll
            for (uint t = 0; t < 32u; ++t) stage[lane][t] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Weight slice for this lane's column, one k-block in registers.
        half4 w[8];
        if (column_valid) {
            const device const half4* src = reinterpret_cast<device const half4*>(
                weights + weight_row + k_base);
            #pragma unroll
            for (uint v = 0; v < 8u; ++v) {
                if (k_base + v * 4u + 4u <= uint(K)) {
                    w[v] = src[v];
                } else {
                    #pragma unroll
                    for (uint e = 0; e < 4u; ++e)
                        w[v][e] = (k_base + v * 4u + e < uint(K))
                            ? weights[weight_row + k_base + v * 4u + e] : half(0.0h);
                }
            }
        } else {
            #pragma unroll
            for (uint v = 0; v < 8u; ++v) w[v] = half4(0.0h);
        }

        #pragma unroll
        for (uint r = 0; r < 32u; ++r) {
            threadgroup const float4* a = reinterpret_cast<threadgroup const float4*>(stage[r]);
            #pragma unroll
            for (uint v = 0; v < 8u; ++v) {
                const float4 av = a[v];
                const half4 wv = w[v];
                acc[r] = fma(av.x, float(wv.x), acc[r]);
                acc[r] = fma(av.y, float(wv.y), acc[r]);
                acc[r] = fma(av.z, float(wv.z), acc[r]);
                acc[r] = fma(av.w, float(wv.w), acc[r]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!column_valid) return;
    #pragma unroll
    for (uint r = 0; r < 32u; ++r) {
        const uint row = row_start + r;
        if (row < uint(M)) output[uint64_t(row) * uint64_t(uint(N)) + column] = acc[r];
    }
}
)METAL";

// Generic V1 sampler. It consumes device-resident F32 logits and writes one
// compact result. Non-finite input is rejected so a successful result is
// always a finite logit. The host descriptor validates the policy fields.
static const char* src_sampler = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct sampler_result {
    uint token_id;
    float logit;
    uint status;
    uint reserved;
};

kernel void sampler_greedy_f32(
    device const float* logits [[buffer(0)]],
    device sampler_result* result [[buffer(1)]],
    constant uint& vocabulary [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup float best_values[256];
    threadgroup uint best_ids[256];
    threadgroup uint nonfinite[256];

    bool found = false;
    float best_value = -INFINITY;
    uint best_id = 0u;
    uint has_nonfinite = 0u;
    for (ulong index = tid; index < ulong(vocabulary); index += ulong(threads)) {
        const float value = logits[index];
        if (!isfinite(value)) {
            has_nonfinite = 1u;
            continue;
        }
        if (!found || value > best_value ||
            (value == best_value && uint(index) < best_id)) {
            found = true;
            best_value = value;
            best_id = uint(index);
        }
    }
    best_values[tid] = found ? best_value : -INFINITY;
    best_ids[tid] = found ? best_id : 0xffffffffu;
    nonfinite[tid] = has_nonfinite;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0u) return;
    for (uint lane = 0u; lane < threads; ++lane)
        if (nonfinite[lane] != 0u) {
            result->token_id = 0u;
            result->logit = 0.0f;
            result->status = 2u;
            result->reserved = 0u;
            return;
        }

    bool reduced = false;
    float reduced_value = -INFINITY;
    uint reduced_id = 0u;
    for (uint lane = 0u; lane < threads; ++lane) {
        const uint id = best_ids[lane];
        const float value = best_values[lane];
        if (id == 0xffffffffu) continue;
        if (!reduced || value > reduced_value ||
            (value == reduced_value && id < reduced_id)) {
            reduced = true;
            reduced_value = value;
            reduced_id = id;
        }
    }
    if (!reduced) {
        result->token_id = 0u;
        result->logit = 0.0f;
        result->status = 2u;
        result->reserved = 0u;
        return;
    }
    result->token_id = reduced_id;
    result->logit = reduced_value;
    result->status = 0u;
    result->reserved = 0u;
}
)METAL";
