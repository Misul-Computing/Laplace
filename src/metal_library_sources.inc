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
