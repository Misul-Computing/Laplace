#include "tensor.h"
#include "matmul.h"
#include "token_graph_backend.h"
#include "kernels.h"
#include "fp16.h"
#include "column_grouped_q4.h"
#include "semantic_model.h"
#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
#include "column_grouped_affine_lowbit.h"
#include "column_grouped_affine_uint2_skip.h"
#endif
#include <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>
#include <chrono>
#include <unistd.h>
#include <cmath>
#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace Laplace {

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_q;
static id<MTLBuffer> g_xbuf;
static id<MTLBuffer> g_ybuf;
static std::mutex g_xy_mtx;
static std::once_flag g_init;
static std::mutex g_pipe_mtx;
static std::unordered_map<int, id<MTLComputePipelineState>> g_pipes;
static std::unordered_map<int, id<MTLComputePipelineState>> g_gemm_pipes;
static std::unordered_map<int, id<MTLComputePipelineState>> g_sparse_row_pipes;
static std::unordered_map<int, id<MTLComputePipelineState>> g_sparse_column_pipes;
static id<MTLLibrary> g_lib;
static id<MTLLibrary> g_gemm_lib;
static id<MTLLibrary> g_prefill_f16_lib;
static id<MTLLibrary> g_m4_lib;
static id<MTLLibrary> g_sampler_lib;
#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
static id<MTLLibrary> g_router_lib;
#endif
static id<MTLComputePipelineState> g_m4_pipe;
static id<MTLComputePipelineState> g_prefill_f16_pipe;
static id<MTLComputePipelineState> g_sampler_pipe;
static std::once_flag g_sampler_init;
#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
static id<MTLComputePipelineState> g_router_topk_pipe;
#endif
static id<MTLComputePipelineState> g_q4k_pipe;
static id<MTLComputePipelineState> g_q2k_pipe;
static id<MTLComputePipelineState> g_iq2_xxs_pipe;
static id<MTLComputePipelineState> g_iq1_s_pipe;
static id<MTLBuffer> g_iq1_s_grid;
static id<MTLComputePipelineState> g_affine_u2_256_pipe;
#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
static id<MTLLibrary> g_mpp_int2_lib;
static id<MTLComputePipelineState> g_mpp_int2_m1_pipe;
static id<MTLComputePipelineState> g_q2k_two_row_pipe;
static id<MTLComputePipelineState> g_q2k_streamed_pipe;
static bool g_test_q2k_two_row_pipeline = false;
static std::once_flag g_column_grouped_q4_init;
static id<MTLLibrary> g_column_grouped_q4_lib;
static id<MTLComputePipelineState> g_column_grouped_q4_dense_pipe;
static id<MTLComputePipelineState> g_column_grouped_q4_selector_pipe;
static id<MTLComputePipelineState> g_column_grouped_q4_sparse_pipe;
static std::once_flag g_column_grouped_affine_lowbit_init;
static id<MTLLibrary> g_column_grouped_affine_lowbit_lib;
static id<MTLComputePipelineState> g_column_grouped_affine_lowbit_pipe;
static id<MTLComputePipelineState> g_column_grouped_affine_lowbit_q2_pipe;
static id<MTLComputePipelineState> g_column_grouped_affine_lowbit_q3_pipe;
static id<MTLComputePipelineState> g_column_grouped_affine_lowbit_q4_pipe;
static std::once_flag g_column_grouped_affine_uint2_skip_init;
static id<MTLLibrary> g_column_grouped_affine_uint2_skip_lib;
static id<MTLComputePipelineState> g_column_grouped_affine_uint2_skip_selector_pipe;
static id<MTLComputePipelineState> g_column_grouped_affine_uint2_skip_dense_pipe;
static id<MTLComputePipelineState> g_column_grouped_affine_uint2_skip_sparse_pipe;
static id<MTLComputePipelineState> g_column_grouped_affine_uint2_skip_reduce_pipe;
#endif
static id<MTLComputePipelineState> g_q4k_id_pipe;
static id<MTLComputePipelineState> g_q8_pipe;
#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
static std::unordered_map<int, id<MTLComputePipelineState>> g_moe_down_reduce_pipes;
#endif
static bool g_m4 = false;
static std::once_flag g_gemm_init;
static std::once_flag g_prefill_f16_init;
static std::mutex g_dispatch_metrics_mutex;
static MetalDispatchMetrics g_dispatch_metrics;

static void record_completed_command(id<MTLCommandBuffer> command,
                                     std::chrono::steady_clock::time_point wait_start,
                                     std::chrono::steady_clock::time_point wait_end) {
    std::lock_guard<std::mutex> lock(g_dispatch_metrics_mutex);
    ++g_dispatch_metrics.command_buffers;
    g_dispatch_metrics.cpu_wait_ms += std::chrono::duration<double, std::milli>(wait_end - wait_start).count();
    const CFTimeInterval gpu_start = [command GPUStartTime];
    const CFTimeInterval gpu_end = [command GPUEndTime];
    if (gpu_end >= gpu_start) g_dispatch_metrics.gpu_time_ms += 1000.0 * (gpu_end - gpu_start);
}

static const char* command_buffer_error_name(NSInteger code) {
    switch (code) {
        case MTLCommandBufferErrorInternal: return "Internal";
        case MTLCommandBufferErrorTimeout: return "Timeout";
        case MTLCommandBufferErrorPageFault: return "PageFault";
        case MTLCommandBufferErrorAccessRevoked: return "AccessRevoked";
        case MTLCommandBufferErrorNotPermitted: return "NotPermitted";
        case MTLCommandBufferErrorOutOfMemory: return "OutOfMemory";
        case MTLCommandBufferErrorInvalidResource: return "InvalidResource";
        case MTLCommandBufferErrorMemoryless: return "Memoryless";
        case MTLCommandBufferErrorStackOverflow: return "StackOverflow";
        default: return "Unknown";
    }
}

static std::string command_buffer_failure_detail(id<MTLCommandBuffer> command) {
    std::ostringstream out;
    out << "Metal command buffer status=" << static_cast<unsigned long>(command.status);
    NSError* error = command.error;
    if (!error) return out.str();
    const char* domain = error.domain.UTF8String;
    const char* description = error.localizedDescription.UTF8String;
    out << " domain=" << (domain ? domain : "")
        << " code=" << static_cast<long long>(error.code)
        << "(" << command_buffer_error_name(error.code) << ")"
        << " description=" << (description ? description : "");
    NSArray* encoders = [error.userInfo objectForKey:MTLCommandBufferEncoderInfoErrorKey];
    if (encoders.count != 0) {
        out << " encoders=[";
        for (NSUInteger index = 0; index != encoders.count; ++index) {
            id<MTLCommandBufferEncoderInfo> encoder = [encoders objectAtIndex:index];
            if (index != 0) out << ',';
            const char* label = encoder.label.UTF8String;
            out << "{label=" << (label ? label : "")
                << ",state=" << static_cast<long long>(encoder.errorState) << '}';
        }
        out << ']';
    }
    return out.str();
}

void metal_dispatch_metrics_reset() {
    std::lock_guard<std::mutex> lock(g_dispatch_metrics_mutex);
    g_dispatch_metrics = {};
}

MetalDispatchMetrics metal_dispatch_metrics() {
    std::lock_guard<std::mutex> lock(g_dispatch_metrics_mutex);
    return g_dispatch_metrics;
}

#include "matmul2d.inc"
#include "gemv_legacy.inc"
#include "iq1_s_tables.inc"

// Zero-copy mmap registration: one or more large Metal buffers covering the
// mmap'd GGUF file. Tensors within the mmap are resolved to (buffer, offset)
// at dispatch time, avoiding the copy in get_weight_buf.
struct MmapBuf {
    id<MTLBuffer> buf;
    const uint8_t* base;
    size_t size;
    const uint8_t* registration_base;
};

struct MetalWeightContext {
    mutable std::mutex mutex;
    std::vector<MmapBuf> mmap_bufs;
    std::unordered_map<const void*, id<MTLBuffer>> copied_bufs;
    bool require_registered_weights = false;
    uint64_t implicit_copy_count = 0;

    ~MetalWeightContext() {
        for (const MmapBuf& buffer : mmap_bufs) [buffer.buf release];
        for (const auto& [_, buffer] : copied_bufs) [buffer release];
    }

    uint64_t byte_count() const {
        uint64_t total = 0;
        for (const MmapBuf& buffer : mmap_bufs) total += buffer.size;
        for (const auto& [_, buffer] : copied_bufs) total += [buffer length];
        return total;
    }
};

static MetalWeightContext g_legacy_weight_context;
static thread_local MetalWeightContext* g_active_weight_context = &g_legacy_weight_context;


// GEMM source: compiled lazily on first metal_gemm(M>1) call.
// Kept separate because metal_simdgroup_matrix increases compile time
// significantly, and we don't want to block init() for a kernel that
// may never be used (decode-only sessions never hit the GEMM path).
static const char* src_gemm = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#define QK 32
#define QK_K 256

enum : int { T_F32=0, T_F16=1, T_BF16=30, T_Q4_0=2, T_Q4_1=3, T_Q5_0=6, T_Q5_1=7, T_Q8_0=8, T_Q2_K=10, T_Q3_K=11, T_Q4_K=12, T_Q5_K=13, T_Q6_K=14 };

struct q4_K_blk { half d; half dmin; uchar scales[12]; uchar qs[128]; };
struct q8_0_blk { half d; int8_t qs[32]; };
struct q6_K_blk { uchar ql[128]; uchar qh[64]; int8_t scales[16]; half d; };

static inline float dequant_f32(device const uchar* row, int k) { return ((device const float*)row)[k]; }
static inline float dequant_f16(device const uchar* row, int k) { return float(((device const half*)row)[k]); }
static inline float dequant_bf16(device const uchar* row, int k) { ushort v = ((device const ushort*)row)[k]; return as_type<float>((uint32_t)v << 16); }

static inline float deq_q4_0(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const struct { half d; uchar qs[16]; }* blk = (decltype(blk))row;
    float d = float(blk[b].d); uint8_t q = blk[b].qs[l%16];
    return d * float((l < 16 ? (q & 0xF) : (q >> 4)) - 8);
}
static inline float deq_q4_1(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const struct { half d; half m; uchar qs[16]; }* blk = (decltype(blk))row;
    float d = float(blk[b].d), m = float(blk[b].m); uint8_t q = blk[b].qs[l%16];
    return d * float(l < 16 ? (q & 0xF) : (q >> 4)) + m;
}
static inline float deq_q5_0(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const struct { half d; uchar qh[4]; uchar qs[16]; }* blk = (decltype(blk))row;
    float d = float(blk[b].d); uint qh = (uint)blk[b].qh[0]|((uint)blk[b].qh[1]<<8)|((uint)blk[b].qh[2]<<16)|((uint)blk[b].qh[3]<<24);
    uint8_t q = blk[b].qs[l%16]; int val = (l < 16 ? (q & 0xF) : (q >> 4)) | (((qh >> l) & 1) << 4);
    return d * float(val - 16);
}
static inline float deq_q5_1(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const struct { half d; half m; uchar qh[4]; uchar qs[16]; }* blk = (decltype(blk))row;
    float d = float(blk[b].d), m = float(blk[b].m); uint qh = (uint)blk[b].qh[0]|((uint)blk[b].qh[1]<<8)|((uint)blk[b].qh[2]<<16)|((uint)blk[b].qh[3]<<24);
    uint8_t q = blk[b].qs[l%16]; int val = (l < 16 ? (q & 0xF) : (q >> 4)) | (((qh >> l) & 1) << 4);
    return d * float(val) + m;
}
static inline float deq_q8_0(device const uchar* row, int k) {
    int b = k/QK, l = k%QK;
    device const uchar* block = row + b * 34;
    int q = int(block[2 + l]);
    return float(*(device const half*)block) *
           float(q < 128 ? q : q - 256);
}
static inline float deq_q2_K(device const uchar* row, int k) {
    int b = k/QK_K, l = k%QK_K; device const struct { uchar scales[16]; uchar qs[64]; half d; }* blk = (decltype(blk))row;
    uchar sc = blk[b].scales[l/16]; uchar q = blk[b].qs[l/4];
    int shift = (l % 4) * 2; int qv = (q >> shift) & 3;
    return float(blk[b].d) * float(qv - 2) * float(sc);
}
static inline float deq_q3_K(device const uchar* row, int k) {
    int b = k/QK_K, l = k%QK_K; device const struct { half d; uchar scales[12]; uchar qs[64]; }* blk = (decltype(blk))row;
    int is = l/16; uchar sc = blk[b].scales[is] & 0x3F; int sign = (blk[b].scales[is] >> 6) & 1;
    uchar q = blk[b].qs[l/4]; int shift = (l % 4) * 2; int qv = (q >> shift) & 3;
    return float(blk[b].d) * float(qv - (sign ? 4 : 0)) * float(sc);
}
static inline float deq_q4_K(device const uchar* row, int k) {
    int b = k/QK_K, l = k%QK_K; device const q4_K_blk* blk = (device const q4_K_blk*)row;
    uchar sc = blk[b].scales[l/16]; uchar q = blk[b].qs[l/2 + (l>=128?64:0)];
    int shift = (l % 32 < 16) ? 0 : 4; int qv = (q >> shift) & 0xF;
    return float(blk[b].d) * float(qv - 8) * float(sc);
}
static inline float deq_q5_K(device const uchar* row, int k) {
    int b = k/QK_K, l = k%QK_K; device const struct { half d; half dmin; uchar scales[12]; uchar qh[32]; uchar qs[128]; }* blk = (decltype(blk))row;
    uchar sc = blk[b].scales[l/16]; uchar qh_v = blk[b].qh[l/8]; uchar q = blk[b].qs[l/2 + (l>=128?64:0)];
    int shift = (l % 32 < 16) ? 0 : 4; int qv = (q >> shift) & 0xF; int qh_bit = (qh_v >> (l%8)) & 1;
    return float(blk[b].d) * float(qv + qh_bit*16 - 8) * float(sc);
}
static inline float deq_q6_K(device const uchar* row, int k) {
    int b = k/QK_K, l = k%QK_K; device const q6_K_blk* blk = (device const q6_K_blk*)row;
    int is = l/16; int8_t sc = blk[b].scales[is]; uchar ql = blk[b].ql[l/2 + (l>=128?64:0)];
    uchar qh = blk[b].qh[l/4]; int shift = (l % 32 < 16) ? 0 : 4;
    int qv = ((ql >> shift) & 0xF) | (((qh >> (l%4)) & 1) << 4);
    return float(blk[b].d) * float(qv - 32) * float(sc);
}

constant int QUANT_TYPE [[function_constant(0)]];

static inline float dequant(device const uchar* row, int k) {
    switch (QUANT_TYPE) {
        case T_F32: return dequant_f32(row, k);
        case T_F16: return dequant_f16(row, k);
        case T_BF16: return dequant_bf16(row, k);
        case T_Q4_0: return deq_q4_0(row, k);
        case T_Q4_1: return deq_q4_1(row, k);
        case T_Q5_0: return deq_q5_0(row, k);
        case T_Q5_1: return deq_q5_1(row, k);
        case T_Q8_0: return deq_q8_0(row, k);
        case T_Q2_K: return deq_q2_K(row, k);
        case T_Q3_K: return deq_q3_K(row, k);
        case T_Q4_K: return deq_q4_K(row, k);
        case T_Q5_K: return deq_q5_K(row, k);
        case T_Q6_K: return deq_q6_K(row, k);
        default: return 0;
    }
}

// GEMM kernel for M > 1 (prefill). Uses simdgroup_matrix 8x8 tiles.
// Threadgroup: 128 threads = 4 SIMD groups in 2x2, covering 32x32 output.
// Each SIMD group handles 16x16 output via 2x2 simdgroup_matrix tiles.
// Weights are dequantized to threadgroup shared memory, then loaded into
// simdgroup_matrix tiles for hardware-accelerated multiply-accumulate.
kernel void gemm_simdgroup(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint64_t& rb [[buffer(5)]],
    constant int& M [[buffer(6)]],
    constant int& n_tg_x [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]
) {
    const int BM = 32, BN = 32, BK = 32;
    int row_start = (int)(tgid / (uint)n_tg_x) * BM;
    int col_start = (int)(tgid % (uint)n_tg_x) * BN;
    if (row_start >= M || col_start >= N) return;
    int sg_m = (int)(sgid / 2);
    int sg_n = (int)(sgid % 2);

    threadgroup float Xs[BM][BK];
    threadgroup float Ws[BN][BK];
    threadgroup float Ys[BM][BN];

    simdgroup_matrix<float, 8, 8> Ymat[2][2];

    for (int ko = 0; ko < K; ko += BK) {
        // Load X tile [BM, BK] from device (float, already contiguous)
        for (int idx = (int)tid; idx < BM * BK; idx += 128) {
            int r = idx / BK, c = idx % BK;
            int mr = row_start + r, ki = ko + c;
            Xs[r][c] = (mr < M && ki < K) ? x[(uint64_t)mr * K + ki] : 0;
        }
        // Dequant W tile [BN, BK] to threadgroup shared memory
        for (int idx = (int)tid; idx < BN * BK; idx += 128) {
            int r = idx / BK, c = idx % BK;
            int ni = col_start + r, ki = ko + c;
            device const uchar* wrow = W + (uint64_t)ni * rb;
            Ws[r][c] = (ni < N && ki < K) ? dequant(wrow, ki) : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply-accumulate in 8x8 simdgroup_matrix tiles
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<float, 8, 8> Xa[2], Wa[2];
            simdgroup_load(Xa[0], (threadgroup float*)&Xs[sg_m*16][kk], BK, ulong2(0,0));
            simdgroup_load(Xa[1], (threadgroup float*)&Xs[sg_m*16 + 8][kk], BK, ulong2(0,0));
            // W is stored as [N, K]; we need W^T[K, N] for the multiply,
            // so load with transpose=true.
            simdgroup_load(Wa[0], (threadgroup float*)&Ws[sg_n*16][kk], BK, ulong2(0,0), true);
            simdgroup_load(Wa[1], (threadgroup float*)&Ws[sg_n*16 + 8][kk], BK, ulong2(0,0), true);
            simdgroup_multiply_accumulate(Ymat[0][0], Xa[0], Wa[0], Ymat[0][0]);
            simdgroup_multiply_accumulate(Ymat[0][1], Xa[0], Wa[1], Ymat[0][1]);
            simdgroup_multiply_accumulate(Ymat[1][0], Xa[1], Wa[0], Ymat[1][0]);
            simdgroup_multiply_accumulate(Ymat[1][1], Xa[1], Wa[1], Ymat[1][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results through threadgroup buffer for bounds-safe writes
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            simdgroup_store(Ymat[i][j], (threadgroup float*)&Ys[sg_m*16 + i*8][sg_n*16 + j*8], BN, ulong2(0,0));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int idx = (int)tid; idx < BM * BN; idx += 128) {
        int r = idx / BN, c = idx % BN;
        int mr = row_start + r, nc = col_start + c;
        if (mr < M && nc < N) y[(uint64_t)mr * N + nc] = Ys[r][c];
    }
}
)METAL";

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

#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
static const char* src_router_topk = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void router_topk_f32(
    device const float* logits [[buffer(0)]],
    device uint* ids [[buffer(1)]],
    device float* weights [[buffer(2)]],
    constant uint& expert_count [[buffer(3)]],
    constant uint& selected_count [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup float values[512];
    threadgroup uint selected[16];
    for (uint expert = tid; expert < expert_count; expert += threads)
        values[expert] = logits[expert];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0) return;

    for (uint slot = 0; slot < selected_count; ++slot) {
        uint best = 0;
        float best_value = -INFINITY;
        for (uint expert = 0; expert < expert_count; ++expert) {
            bool already_selected = false;
            for (uint earlier = 0; earlier < slot; ++earlier)
                already_selected = already_selected || selected[earlier] == expert;
            if (already_selected) continue;
            const float value = values[expert];
            if (value > best_value || (value == best_value && expert < best)) {
                best_value = value;
                best = expert;
            }
        }
        selected[slot] = best;
        ids[slot] = best;
    }

    float maximum = -INFINITY;
    for (uint expert = 0; expert < expert_count; ++expert)
        maximum = max(maximum, values[expert]);
    float full_normalizer = 0.0f;
    for (uint expert = 0; expert < expert_count; ++expert)
        full_normalizer += exp(values[expert] - maximum);
    float selected_normalizer = 0.0f;
    for (uint slot = 0; slot < selected_count; ++slot) {
        const float probability = exp(values[selected[slot]] - maximum) / full_normalizer;
        weights[slot] = probability;
        selected_normalizer += probability;
    }
    const float inverse = 1.0f / selected_normalizer;
    for (uint slot = 0; slot < selected_count; ++slot)
        weights[slot] *= inverse;
}
)METAL";
#endif

static const char* mpp_header_dir() {
    const char* e = std::getenv("LAPLACE_MPP_DIR");
    if (e && e[0]) return e;
    return "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/"
           "Frameworks/MetalPerformancePrimitives.framework/Headers";
}

static bool parse_include(const std::string& line, std::string& inc, int& kind) {
    auto p = line.find("#include");
    if (p == std::string::npos) return false;
    for (size_t i = 0; i < p; i++)
        if (line[i] != ' ' && line[i] != '\t') return false;
    auto lt = line.find('<', p);
    auto q = line.find('"', p);
    if (lt != std::string::npos && (q == std::string::npos || lt < q)) {
        auto gt = line.find('>', lt);
        if (gt == std::string::npos) return false;
        inc = line.substr(lt + 1, gt - lt - 1);
        const char* pfx = "MetalPerformancePrimitives/";
        if (inc.rfind(pfx, 0) == 0) {
            inc = inc.substr(std::strlen(pfx));
            kind = 1;
        } else kind = 0;
        return true;
    }
    if (q != std::string::npos) {
        auto q2 = line.find('"', q + 1);
        if (q2 == std::string::npos) return false;
        inc = line.substr(q + 1, q2 - q - 1);
        kind = 2;
        return true;
    }
    return false;
}

static std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void flatten_metal(const std::string& src, const std::string& curdir,
                          const std::string& hdr, std::string& out,
                          std::unordered_set<std::string>& seen) {
    std::istringstream in(src);
    std::string line;
    while (std::getline(in, line)) {
        std::string inc;
        int kind = 0;
        if (!parse_include(line, inc, kind) || kind == 0) {
            out += line;
            out += '\n';
            continue;
        }
        std::string path;
        if (kind == 1) path = hdr + "/" + inc;
        else {
            path = curdir + "/" + inc;
            if (read_text(path).empty()) path = hdr + "/" + inc;
            if (read_text(path).empty()) path = hdr + "/__impl/" + inc;
        }
        if (!seen.insert(path).second) continue;
        std::string body = read_text(path);
        if (body.empty()) { out += line; out += '\n'; continue; }
        auto slash = path.rfind('/');
        flatten_metal(body, slash == std::string::npos ? curdir : path.substr(0, slash),
                      hdr, out, seen);
    }
}

// Test-only LaplaceColumnGroupedQ4V1 kernels. This is a separately compiled
// experimental storage layout, not a GGML or SpQt compatibility path.
#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
static void init();
static const char* src_column_grouped_q4 = R"METAL(
#include <metal_stdlib>
using namespace metal;

static inline float column_grouped_q4_value(device const uchar* bytes, uint logical_n,
                                             uint column, uint row) {
    const uint output_blocks = logical_n >> 8u;
    const ulong offset = (ulong(column) * output_blocks + (row >> 8u)) * 136ul;
    device const uchar* block = bytes + offset;
    const uchar packed = block[(row & 255u) >> 1u];
    const uint q = (uint(packed) >> (4u * (row & 1u))) & 15u;
    const uint scale_bits = uint(block[128]) | (uint(block[129]) << 8u) |
                            (uint(block[130]) << 16u) | (uint(block[131]) << 24u);
    const uint bias_bits = uint(block[132]) | (uint(block[133]) << 8u) |
                           (uint(block[134]) << 16u) | (uint(block[135]) << 24u);
    return as_type<float>(scale_bits) * float(q) + as_type<float>(bias_bits);
}

kernel void column_grouped_q4_select_nonzero(
    device const float* input [[buffer(0)]],
    device atomic_uint* selected_count [[buffer(1)]],
    device uint* selected_indices [[buffer(2)]],
    constant uint& logical_k [[buffer(3)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= logical_k || input[column] == 0.0f) return;
    const uint slot = atomic_fetch_add_explicit(selected_count, 1u, memory_order_relaxed);
    selected_indices[slot] = column;
}

kernel void column_grouped_q4_dense_gemv(
    device const uchar* bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& logical_k [[buffer(3)]],
    constant uint& logical_n [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= logical_n) return;
    float sum = 0.0f;
    for (uint column = 0; column != logical_k; ++column)
        sum += input[column] * column_grouped_q4_value(bytes, logical_n, column, row);
    output[row] = sum;
}

kernel void column_grouped_q4_sparse_gemv(
    device const uchar* bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const atomic_uint* selected_count [[buffer(3)]],
    device const uint* selected_indices [[buffer(4)]],
    constant uint& logical_k [[buffer(5)]],
    constant uint& logical_n [[buffer(6)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= logical_n) return;
    const uint count = atomic_load_explicit(selected_count, memory_order_relaxed);
    float sum = 0.0f;
    for (uint index = 0; index != count; ++index) {
        const uint column = selected_indices[index];
        if (column < logical_k)
            sum += input[column] * column_grouped_q4_value(bytes, logical_n, column, row);
    }
    output[row] = sum;
}
)METAL";

static bool column_grouped_q4_pipelines_ready() {
    init();
    std::call_once(g_column_grouped_q4_init, [] {
        if (!g_dev) return;
        NSError* error = nil;
        g_column_grouped_q4_lib = [g_dev newLibraryWithSource:
            [NSString stringWithUTF8String:src_column_grouped_q4] options:nil error:&error];
        if (!g_column_grouped_q4_lib) {
            std::fprintf(stderr, "[metal] ColumnGroupedQ4V1 library: %s\n",
                         error ? error.localizedDescription.UTF8String : "unknown");
            return;
        }
        auto make = [&](const char* name) {
            id<MTLFunction> function = [g_column_grouped_q4_lib newFunctionWithName:
                [NSString stringWithUTF8String:name]];
            if (!function) return static_cast<id<MTLComputePipelineState>>(nil);
            NSError* pipeline_error = nil;
            id<MTLComputePipelineState> pipeline =
                [g_dev newComputePipelineStateWithFunction:function error:&pipeline_error];
            [function release];
            if (!pipeline)
                std::fprintf(stderr, "[metal] ColumnGroupedQ4V1 %s: %s\n", name,
                             pipeline_error ? pipeline_error.localizedDescription.UTF8String : "unknown");
            return pipeline;
        };
        g_column_grouped_q4_selector_pipe = make("column_grouped_q4_select_nonzero");
        g_column_grouped_q4_dense_pipe = make("column_grouped_q4_dense_gemv");
        g_column_grouped_q4_sparse_pipe = make("column_grouped_q4_sparse_gemv");
    });
    return g_dev && g_q && g_column_grouped_q4_dense_pipe &&
           g_column_grouped_q4_selector_pipe && g_column_grouped_q4_sparse_pipe;
}
#endif

#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
// Test-only direct execution of ColumnGroupedAffineLowBitV1. The three planes
// stay distinct so this seam cannot accidentally consume a legacy affine-u2
// or column-q4 representation.
static const char* src_column_grouped_affine_lowbit_v1 = R"METAL(
#include <metal_stdlib>
using namespace metal;

static inline uint affine_lowbit_code(device const uchar* values, uint lane,
                                      uint bits, uint packed_bytes) {
    const uint bit = lane * bits;
    const uint byte = bit >> 3u;
    const uint shift = bit & 7u;
    uint word = uint(values[byte]);
    if (byte + 1u < packed_bytes) word |= uint(values[byte + 1u]) << 8u;
    return (word >> shift) & ((1u << bits) - 1u);
}

kernel void column_grouped_affine_lowbit_v1_gemv(
    device const uchar* values [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const half* biases [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& logical_k [[buffer(5)]],
    constant uint& logical_n [[buffer(6)]],
    constant uint& bits [[buffer(7)]],
    constant uint& packed_bytes [[buffer(8)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= logical_n) return;
    const uint output_groups = logical_n >> 8u;
    const uint output_group = row >> 8u;
    const uint lane = row & 255u;
    float sum = 0.0f;
    for (uint column = 0; column != logical_k; ++column) {
        const ulong group = ulong(column) * ulong(output_groups) + ulong(output_group);
        device const uchar* packed = values + group * ulong(packed_bytes);
        const uint code = affine_lowbit_code(packed, lane, bits, packed_bytes);
        sum += input[column] * (float(scales[group]) * float(code) + float(biases[group]));
    }
    output[row] = sum;
}

static inline uint affine_lowbit_code_q3(device const uchar* values, uint lane) {
    const uint bit = lane * 3u;
    const uint byte = bit >> 3u;
    const uint shift = bit & 7u;
    uint word = uint(values[byte]);
    if (shift > 5u) word |= uint(values[byte + 1u]) << 8u;
    return (word >> shift) & 7u;
}

// A/B replacement experiment. Each entry point fixes its byte geometry so the
// hot loop has no runtime bit-width branch and each thread owns a packed row group.
kernel void column_grouped_affine_lowbit_v1_q2(
    device const uchar* values [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const half* biases [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& logical_k [[buffer(5)]],
    constant uint& logical_n [[buffer(6)]],
    uint global_id [[thread_position_in_grid]]) {
    constexpr uint rows_per_thread = 4u;
    const uint total_threads = logical_n / rows_per_thread;
    if (global_id >= total_threads) return;
    const uint output_groups = logical_n >> 8u;
    const uint output_group = global_id >> 6u;
    const uint byte_lane = global_id & 63u;
    const uint row = (output_group << 8u) + (byte_lane << 2u);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    uint column = 0u;
    for (; column + 3u < logical_k; column += 4u) {
        const ulong group0 = ulong(column) * ulong(output_groups) + ulong(output_group);
        const ulong group1 = group0 + ulong(output_groups);
        const ulong group2 = group1 + ulong(output_groups);
        const ulong group3 = group2 + ulong(output_groups);
        device const uchar* packed0 = values + group0 * 64ul;
        device const uchar* packed1 = values + group1 * 64ul;
        device const uchar* packed2 = values + group2 * 64ul;
        device const uchar* packed3 = values + group3 * 64ul;
        const uint byte0 = uint(packed0[byte_lane]);
        const uint byte1 = uint(packed1[byte_lane]);
        const uint byte2 = uint(packed2[byte_lane]);
        const uint byte3 = uint(packed3[byte_lane]);
        const float input0 = input[column];
        const float input1 = input[column + 1u];
        const float input2 = input[column + 2u];
        const float input3 = input[column + 3u];
        const float scale0 = float(scales[group0]);
        const float scale1 = float(scales[group1]);
        const float scale2 = float(scales[group2]);
        const float scale3 = float(scales[group3]);
        const float bias0 = float(biases[group0]);
        const float bias1 = float(biases[group1]);
        const float bias2 = float(biases[group2]);
        const float bias3 = float(biases[group3]);
        sum0 += input0 * (scale0 * float(byte0 & 3u) + bias0);
        sum1 += input0 * (scale0 * float((byte0 >> 2u) & 3u) + bias0);
        sum2 += input0 * (scale0 * float((byte0 >> 4u) & 3u) + bias0);
        sum3 += input0 * (scale0 * float(byte0 >> 6u) + bias0);
        sum0 += input1 * (scale1 * float(byte1 & 3u) + bias1);
        sum1 += input1 * (scale1 * float((byte1 >> 2u) & 3u) + bias1);
        sum2 += input1 * (scale1 * float((byte1 >> 4u) & 3u) + bias1);
        sum3 += input1 * (scale1 * float(byte1 >> 6u) + bias1);
        sum0 += input2 * (scale2 * float(byte2 & 3u) + bias2);
        sum1 += input2 * (scale2 * float((byte2 >> 2u) & 3u) + bias2);
        sum2 += input2 * (scale2 * float((byte2 >> 4u) & 3u) + bias2);
        sum3 += input2 * (scale2 * float(byte2 >> 6u) + bias2);
        sum0 += input3 * (scale3 * float(byte3 & 3u) + bias3);
        sum1 += input3 * (scale3 * float((byte3 >> 2u) & 3u) + bias3);
        sum2 += input3 * (scale3 * float((byte3 >> 4u) & 3u) + bias3);
        sum3 += input3 * (scale3 * float(byte3 >> 6u) + bias3);
    }
    for (; column < logical_k; ++column) {
        const ulong group = ulong(column) * ulong(output_groups) + ulong(output_group);
        const uint packed = uint(values[group * 64ul + ulong(byte_lane)]);
        const float value = input[column];
        const float scale = float(scales[group]);
        const float bias = float(biases[group]);
        sum0 += value * (scale * float(packed & 3u) + bias);
        sum1 += value * (scale * float((packed >> 2u) & 3u) + bias);
        sum2 += value * (scale * float((packed >> 4u) & 3u) + bias);
        sum3 += value * (scale * float(packed >> 6u) + bias);
    }
    output[row] = sum0;
    output[row + 1u] = sum1;
    output[row + 2u] = sum2;
    output[row + 3u] = sum3;
}

kernel void column_grouped_affine_lowbit_v1_q3(
    device const uchar* values [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const half* biases [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& logical_k [[buffer(5)]],
    constant uint& logical_n [[buffer(6)]],
    uint global_id [[thread_position_in_grid]]) {
    constexpr uint rows_per_thread = 2u;
    const uint total_threads = logical_n / rows_per_thread;
    if (global_id >= total_threads) return;
    const uint output_groups = logical_n >> 8u;
    const uint output_group = global_id >> 7u;
    const uint byte_lane = global_id & 127u;
    const uint lane = byte_lane << 1u;
    const uint row = (output_group << 8u) + lane;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    uint column = 0u;
    for (; column + 3u < logical_k; column += 4u) {
        const ulong group0 = ulong(column) * ulong(output_groups) + ulong(output_group);
        const ulong group1 = group0 + ulong(output_groups);
        const ulong group2 = group1 + ulong(output_groups);
        const ulong group3 = group2 + ulong(output_groups);
        device const uchar* packed0 = values + group0 * 96ul;
        device const uchar* packed1 = values + group1 * 96ul;
        device const uchar* packed2 = values + group2 * 96ul;
        device const uchar* packed3 = values + group3 * 96ul;
        const uint code00 = affine_lowbit_code_q3(packed0, lane);
        const uint code01 = affine_lowbit_code_q3(packed0, lane + 1u);
        const uint code10 = affine_lowbit_code_q3(packed1, lane);
        const uint code11 = affine_lowbit_code_q3(packed1, lane + 1u);
        const uint code20 = affine_lowbit_code_q3(packed2, lane);
        const uint code21 = affine_lowbit_code_q3(packed2, lane + 1u);
        const uint code30 = affine_lowbit_code_q3(packed3, lane);
        const uint code31 = affine_lowbit_code_q3(packed3, lane + 1u);
        const float input0 = input[column];
        const float input1 = input[column + 1u];
        const float input2 = input[column + 2u];
        const float input3 = input[column + 3u];
        const float scale0 = float(scales[group0]);
        const float scale1 = float(scales[group1]);
        const float scale2 = float(scales[group2]);
        const float scale3 = float(scales[group3]);
        const float bias0 = float(biases[group0]);
        const float bias1 = float(biases[group1]);
        const float bias2 = float(biases[group2]);
        const float bias3 = float(biases[group3]);
        sum0 += input0 * (scale0 * float(code00) + bias0);
        sum1 += input0 * (scale0 * float(code01) + bias0);
        sum0 += input1 * (scale1 * float(code10) + bias1);
        sum1 += input1 * (scale1 * float(code11) + bias1);
        sum0 += input2 * (scale2 * float(code20) + bias2);
        sum1 += input2 * (scale2 * float(code21) + bias2);
        sum0 += input3 * (scale3 * float(code30) + bias3);
        sum1 += input3 * (scale3 * float(code31) + bias3);
    }
    for (; column < logical_k; ++column) {
        const ulong group = ulong(column) * ulong(output_groups) + ulong(output_group);
        device const uchar* packed = values + group * 96ul;
        const float value = input[column];
        const float scale = float(scales[group]);
        const float bias = float(biases[group]);
        sum0 += value * (scale * float(affine_lowbit_code_q3(packed, lane)) + bias);
        sum1 += value * (scale * float(affine_lowbit_code_q3(packed, lane + 1u)) + bias);
    }
    output[row] = sum0;
    output[row + 1u] = sum1;
}

kernel void column_grouped_affine_lowbit_v1_q4(
    device const uchar* values [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const half* biases [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& logical_k [[buffer(5)]],
    constant uint& logical_n [[buffer(6)]],
    uint global_id [[thread_position_in_grid]]) {
    constexpr uint rows_per_thread = 2u;
    const uint total_threads = logical_n / rows_per_thread;
    if (global_id >= total_threads) return;
    const uint output_groups = logical_n >> 8u;
    const uint output_group = global_id >> 7u;
    const uint byte_lane = global_id & 127u;
    const uint row = (output_group << 8u) + (byte_lane << 1u);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    uint column = 0u;
    for (; column + 3u < logical_k; column += 4u) {
        const ulong group0 = ulong(column) * ulong(output_groups) + ulong(output_group);
        const ulong group1 = group0 + ulong(output_groups);
        const ulong group2 = group1 + ulong(output_groups);
        const ulong group3 = group2 + ulong(output_groups);
        device const uchar* packed0 = values + group0 * 128ul;
        device const uchar* packed1 = values + group1 * 128ul;
        device const uchar* packed2 = values + group2 * 128ul;
        device const uchar* packed3 = values + group3 * 128ul;
        const uint byte0 = uint(packed0[byte_lane]);
        const uint byte1 = uint(packed1[byte_lane]);
        const uint byte2 = uint(packed2[byte_lane]);
        const uint byte3 = uint(packed3[byte_lane]);
        const float input0 = input[column];
        const float input1 = input[column + 1u];
        const float input2 = input[column + 2u];
        const float input3 = input[column + 3u];
        const float scale0 = float(scales[group0]);
        const float scale1 = float(scales[group1]);
        const float scale2 = float(scales[group2]);
        const float scale3 = float(scales[group3]);
        const float bias0 = float(biases[group0]);
        const float bias1 = float(biases[group1]);
        const float bias2 = float(biases[group2]);
        const float bias3 = float(biases[group3]);
        sum0 += input0 * (scale0 * float(byte0 & 15u) + bias0);
        sum1 += input0 * (scale0 * float(byte0 >> 4u) + bias0);
        sum0 += input1 * (scale1 * float(byte1 & 15u) + bias1);
        sum1 += input1 * (scale1 * float(byte1 >> 4u) + bias1);
        sum0 += input2 * (scale2 * float(byte2 & 15u) + bias2);
        sum1 += input2 * (scale2 * float(byte2 >> 4u) + bias2);
        sum0 += input3 * (scale3 * float(byte3 & 15u) + bias3);
        sum1 += input3 * (scale3 * float(byte3 >> 4u) + bias3);
    }
    for (; column < logical_k; ++column) {
        const ulong group = ulong(column) * ulong(output_groups) + ulong(output_group);
        const uint packed = uint(values[group * 128ul + ulong(byte_lane)]);
        const float value = input[column];
        const float scale = float(scales[group]);
        const float bias = float(biases[group]);
        sum0 += value * (scale * float(packed & 15u) + bias);
        sum1 += value * (scale * float(packed >> 4u) + bias);
    }
    output[row] = sum0;
    output[row + 1u] = sum1;
}

)METAL";

static bool column_grouped_affine_lowbit_v1_pipeline_ready() {
    init();
    std::call_once(g_column_grouped_affine_lowbit_init, [] {
        if (!g_dev) return;
        NSError* error = nil;
        g_column_grouped_affine_lowbit_lib = [g_dev newLibraryWithSource:
            [NSString stringWithUTF8String:src_column_grouped_affine_lowbit_v1]
            options:nil error:&error];
        if (!g_column_grouped_affine_lowbit_lib) {
            std::fprintf(stderr, "[metal] ColumnGroupedAffineLowBitV1 library: %s\n",
                         error ? error.localizedDescription.UTF8String : "unknown");
            return;
        }
        id<MTLFunction> function = [g_column_grouped_affine_lowbit_lib
            newFunctionWithName:@"column_grouped_affine_lowbit_v1_gemv"];
        if (!function) return;
        g_column_grouped_affine_lowbit_pipe =
            [g_dev newComputePipelineStateWithFunction:function error:&error];
        [function release];
        if (!g_column_grouped_affine_lowbit_pipe)
            std::fprintf(stderr, "[metal] ColumnGroupedAffineLowBitV1 pipeline: %s\n",
                         error ? error.localizedDescription.UTF8String : "unknown");

        const auto make_optimized = [&](const char* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> optimized_function =
                [g_column_grouped_affine_lowbit_lib newFunctionWithName:
                    [NSString stringWithUTF8String:name]];
            if (!optimized_function) return nil;
            NSError* optimized_error = nil;
            id<MTLComputePipelineState> optimized_pipeline =
                [g_dev newComputePipelineStateWithFunction:optimized_function
                                                      error:&optimized_error];
            [optimized_function release];
            if (!optimized_pipeline)
                std::fprintf(stderr, "[metal] ColumnGroupedAffineLowBitV1 %s: %s\n", name,
                             optimized_error ? optimized_error.localizedDescription.UTF8String : "unknown");
            return optimized_pipeline;
        };
        g_column_grouped_affine_lowbit_q2_pipe =
            make_optimized("column_grouped_affine_lowbit_v1_q2");
        g_column_grouped_affine_lowbit_q3_pipe =
            make_optimized("column_grouped_affine_lowbit_v1_q3");
        g_column_grouped_affine_lowbit_q4_pipe =
            make_optimized("column_grouped_affine_lowbit_v1_q4");
    });
    return g_dev && g_q && g_column_grouped_affine_lowbit_pipe;
}

static id<MTLComputePipelineState> column_grouped_affine_lowbit_v1_optimized_pipeline(
    uint8_t bits) {
    switch (bits) {
        case 2u: return g_column_grouped_affine_lowbit_q2_pipe;
        case 3u: return g_column_grouped_affine_lowbit_q3_pipe;
        case 4u: return g_column_grouped_affine_lowbit_q4_pipe;
        default: return nil;
    }
}

// Test-only direct consumer for ColumnGroupedAffineUInt2SkipV1. One
// threadgroup owns one 256-row output block. Each of its 64 threads reads one
// packed byte and accumulates four output rows. The sparse path consumes only
// a device-built active-column list.
static const char* src_column_grouped_affine_uint2_skip_v1 = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void column_grouped_affine_uint2_skip_v1_select(
    device const float* input [[buffer(0)]],
    device atomic_uint* active_count [[buffer(1)]],
    device uint* active_columns [[buffer(2)]],
    constant uint& logical_k [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup uint offsets[256];
    const uint begin = uint((ulong(logical_k) * ulong(thread_index)) / 256ul);
    const uint end = uint((ulong(logical_k) * ulong(thread_index + 1u)) / 256ul);
    uint local_count = 0u;
    for (uint column = begin; column != end; ++column) {
        const float value = input[column];
        if (isfinite(value) && value != 0.0f) ++local_count;
    }
    offsets[thread_index] = local_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < 256u; stride <<= 1u) {
        const uint index = (thread_index + 1u) * (stride << 1u) - 1u;
        if (index < 256u) offsets[index] += offsets[index - stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index == 0u) offsets[255] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        const uint index = (thread_index + 1u) * (stride << 1u) - 1u;
        if (index < 256u) {
            const uint left = offsets[index - stride];
            offsets[index - stride] = offsets[index];
            offsets[index] += left;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint output_index = offsets[thread_index];
    for (uint column = begin; column != end; ++column) {
        const float value = input[column];
        if (isfinite(value) && value != 0.0f) active_columns[output_index++] = column;
    }
    if (thread_index == 255u)
        atomic_store_explicit(active_count, offsets[255] + local_count,
                              memory_order_relaxed);
}

kernel void column_grouped_affine_uint2_skip_v1_dense(
    device const uchar* values [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const half* biases [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& logical_k [[buffer(5)]],
    constant uint& logical_n [[buffer(6)]],
    uint byte_lane [[thread_index_in_threadgroup]],
    uint output_block [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
    if (byte_lane >= 64u || output_block >= (logical_n >> 8u)) return;
    const ulong group_base = ulong(output_block) * ulong(logical_k);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (uint column = 0u; column != logical_k; ++column) {
        const ulong group = group_base + ulong(column);
        const uint packed = uint(values[group * 64ul + ulong(byte_lane)]);
        float input_value = 0.0f;
        float scale = 0.0f;
        float bias = 0.0f;
        if (simd_lane == 0u) {
            input_value = input[column];
            scale = float(scales[group]);
            bias = float(biases[group]);
        }
        input_value = simd_broadcast_first(input_value);
        scale = simd_broadcast_first(scale);
        bias = simd_broadcast_first(bias);
        sum0 += input_value * (scale * float(packed & 3u) + bias);
        sum1 += input_value * (scale * float((packed >> 2u) & 3u) + bias);
        sum2 += input_value * (scale * float((packed >> 4u) & 3u) + bias);
        sum3 += input_value * (scale * float(packed >> 6u) + bias);
    }
    const uint row = (output_block << 8u) + (byte_lane << 2u);
    output[row] = sum0;
    output[row + 1u] = sum1;
    output[row + 2u] = sum2;
    output[row + 3u] = sum3;
}

static inline void column_grouped_affine_uint2_skip_v1_accumulate(
    device const uchar* values,
    device const uint* metadata,
    device const float* input,
    device const uint* active_columns,
    uint index,
    uint logical_k,
    uint group_base,
    uint simd_lane,
    thread float* sums0,
    thread float* sums1,
    thread float* bias_sum) {
    const uint column = active_columns[index];
    if (column >= logical_k) return;
    const uint group = group_base + column;
    device const ushort* packed_groups =
        reinterpret_cast<device const ushort*>(values + ulong(group) * 64ul);
    const uint packed = uint(packed_groups[simd_lane]);
    const uint packed0 = packed & 255u;
    const uint packed1 = packed >> 8u;
    const float input_value = input[column];
    const half2 scale_bias = as_type<half2>(metadata[group]);
    const float scaled_input = input_value * float(scale_bias.x);
    const float biased_input = input_value * float(scale_bias.y);
    *bias_sum += biased_input;
    sums0[0] = fma(scaled_input, float(packed0 & 3u), sums0[0]);
    sums0[1] = fma(scaled_input, float((packed0 >> 2u) & 3u), sums0[1]);
    sums0[2] = fma(scaled_input, float((packed0 >> 4u) & 3u), sums0[2]);
    sums0[3] = fma(scaled_input, float(packed0 >> 6u), sums0[3]);
    sums1[0] = fma(scaled_input, float(packed1 & 3u), sums1[0]);
    sums1[1] = fma(scaled_input, float((packed1 >> 2u) & 3u), sums1[1]);
    sums1[2] = fma(scaled_input, float((packed1 >> 4u) & 3u), sums1[2]);
    sums1[3] = fma(scaled_input, float(packed1 >> 6u), sums1[3]);
}

kernel void column_grouped_affine_uint2_skip_v1_sparse(
    device const uchar* values [[buffer(0)]],
    device const uint* metadata [[buffer(1)]],
    device const float* input [[buffer(3)]],
    device float* partial_output [[buffer(4)]],
    device const atomic_uint* active_count [[buffer(5)]],
    device const uint* active_columns [[buffer(6)]],
    constant uint& logical_k [[buffer(7)]],
    constant uint& logical_n [[buffer(8)]],
    constant uint& split_count [[buffer(9)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[512];
    const uint output_block = threadgroup_index / split_count;
    const uint split = threadgroup_index - output_block * split_count;
    if (thread_index >= 64u || output_block >= (logical_n >> 8u)) return;
    uint count = 0u;
    if (simd_lane == 0u)
        count = min(atomic_load_explicit(active_count, memory_order_relaxed), logical_k);
    count = simd_broadcast_first(count);
    const uint simd_count = split_count * 2u;
    const uint simd_global = split * 2u + simd_index;
    const uint begin = (count * simd_global) / simd_count;
    const uint end = (count * (simd_global + 1u)) / simd_count;
    const uint group_base = output_block * logical_k;
    float sums0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sums1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float bias_sum = 0.0f;
    uint index = begin;
    for (; index + 3u < end; index += 4u) {
        const uint4 columns = uint4(active_columns[index], active_columns[index + 1u],
                                    active_columns[index + 2u], active_columns[index + 3u]);
        if (any(columns >= uint4(logical_k))) {
            for (uint tail = 0u; tail != 4u; ++tail)
                column_grouped_affine_uint2_skip_v1_accumulate(
                    values, metadata, input, active_columns, index + tail, logical_k,
                    group_base, simd_lane, sums0, sums1, &bias_sum);
            continue;
        }
        const uint4 groups = uint4(group_base) + columns;
        const float4 input_values = float4(input[columns.x], input[columns.y],
                                           input[columns.z], input[columns.w]);
        const half2 sb0 = as_type<half2>(metadata[groups.x]);
        const half2 sb1 = as_type<half2>(metadata[groups.y]);
        const half2 sb2 = as_type<half2>(metadata[groups.z]);
        const half2 sb3 = as_type<half2>(metadata[groups.w]);
        const float4 scaled_inputs = input_values *
            float4(float(sb0.x), float(sb1.x), float(sb2.x), float(sb3.x));
        const float4 biased_inputs = input_values *
            float4(float(sb0.y), float(sb1.y), float(sb2.y), float(sb3.y));
        bias_sum += biased_inputs.x + biased_inputs.y + biased_inputs.z + biased_inputs.w;
        const uint4 packed = uint4(
            uint(reinterpret_cast<device const ushort*>(
                values + ulong(groups.x) * 64ul)[simd_lane]),
            uint(reinterpret_cast<device const ushort*>(
                values + ulong(groups.y) * 64ul)[simd_lane]),
            uint(reinterpret_cast<device const ushort*>(
                values + ulong(groups.z) * 64ul)[simd_lane]),
            uint(reinterpret_cast<device const ushort*>(
                values + ulong(groups.w) * 64ul)[simd_lane]));
        uint4 codes = packed;
        for (uint part = 0u; part != 4u; ++part) {
            sums0[part] += dot(scaled_inputs, float4(codes & uint4(3u)));
            codes >>= uint4(2u);
        }
        for (uint part = 0u; part != 4u; ++part) {
            sums1[part] += dot(scaled_inputs, float4(codes & uint4(3u)));
            codes >>= uint4(2u);
        }
    }
    for (; index < end; ++index)
        column_grouped_affine_uint2_skip_v1_accumulate(
            values, metadata, input, active_columns, index, logical_k, group_base,
            simd_lane, sums0, sums1, &bias_sum);
    const uint local_base = simd_index * 256u + simd_lane * 8u;
    for (uint part = 0u; part != 4u; ++part) {
        partials[local_base + part] = sums0[part] + bias_sum;
        partials[local_base + 4u + part] = sums1[part] + bias_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row_in_block = thread_index * 4u;
    const ulong output_base = (ulong(output_block) * ulong(split_count) + ulong(split)) * 256ul;
    for (uint part = 0u; part != 4u; ++part)
        partial_output[output_base + ulong(row_in_block + part)] =
            partials[row_in_block + part] + partials[256u + row_in_block + part];
}

kernel void column_grouped_affine_uint2_skip_v1_reduce(
    device const float* partial_output [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& logical_n [[buffer(2)]],
    constant uint& split_count [[buffer(3)]],
    uint row_in_block [[thread_index_in_threadgroup]],
    uint output_block [[threadgroup_position_in_grid]]) {
    if (row_in_block >= 256u || output_block >= (logical_n >> 8u)) return;
    float sum = 0.0f;
    const ulong base = ulong(output_block) * ulong(split_count) * 256ul +
                       ulong(row_in_block);
    for (uint split = 0u; split != split_count; ++split)
        sum += partial_output[base + ulong(split) * 256ul];
    output[(output_block << 8u) + row_in_block] = sum;
}
)METAL";

static bool column_grouped_affine_uint2_skip_v1_pipelines_ready() {
    init();
    std::call_once(g_column_grouped_affine_uint2_skip_init, [] {
        if (!g_dev) return;
        NSError* error = nil;
        g_column_grouped_affine_uint2_skip_lib = [g_dev newLibraryWithSource:
            [NSString stringWithUTF8String:src_column_grouped_affine_uint2_skip_v1]
            options:nil error:&error];
        if (!g_column_grouped_affine_uint2_skip_lib) {
            std::fprintf(stderr, "[metal] ColumnGroupedAffineUInt2SkipV1 library: %s\n",
                         error ? error.localizedDescription.UTF8String : "unknown");
            return;
        }
        const auto make = [&](const char* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> function = [g_column_grouped_affine_uint2_skip_lib
                newFunctionWithName:[NSString stringWithUTF8String:name]];
            if (!function) return nil;
            NSError* pipeline_error = nil;
            id<MTLComputePipelineState> pipeline =
                [g_dev newComputePipelineStateWithFunction:function error:&pipeline_error];
            [function release];
            if (!pipeline)
                std::fprintf(stderr, "[metal] ColumnGroupedAffineUInt2SkipV1 %s: %s\n",
                             name, pipeline_error
                                       ? pipeline_error.localizedDescription.UTF8String
                                       : "unknown");
            return pipeline;
        };
        g_column_grouped_affine_uint2_skip_selector_pipe =
            make("column_grouped_affine_uint2_skip_v1_select");
        g_column_grouped_affine_uint2_skip_dense_pipe =
            make("column_grouped_affine_uint2_skip_v1_dense");
        g_column_grouped_affine_uint2_skip_sparse_pipe =
            make("column_grouped_affine_uint2_skip_v1_sparse");
        g_column_grouped_affine_uint2_skip_reduce_pipe =
            make("column_grouped_affine_uint2_skip_v1_reduce");
    });
    return g_dev && g_q && g_column_grouped_affine_uint2_skip_lib &&
           g_column_grouped_affine_uint2_skip_selector_pipe &&
           g_column_grouped_affine_uint2_skip_dense_pipe &&
           g_column_grouped_affine_uint2_skip_sparse_pipe &&
           g_column_grouped_affine_uint2_skip_reduce_pipe;
}

static bool affine_v1_overlap(const void* left, size_t left_bytes,
                              const void* right, size_t right_bytes);

static bool column_grouped_affine_uint2_skip_v1_contract_valid(
    const ColumnGroupedAffineUInt2SkipV1Contract& contract) {
    if (contract.version != 1u || contract.reserved != 0u || contract.logical_k == 0u ||
        contract.logical_n == 0u || contract.logical_k > UINT32_MAX / 64u ||
        contract.logical_n > UINT32_MAX || contract.group_count > UINT32_MAX ||
        contract.group_elements != 256u ||
        contract.packed_bytes_per_group != 64u || contract.scale_bytes_per_group != 2u ||
        contract.bias_bytes_per_group != 2u || contract.plane_alignment != 128u ||
        (contract.logical_n % 256u) != 0u)
        return false;
    const uint64_t output_blocks = contract.logical_n / 256u;
    if (contract.logical_k > std::numeric_limits<uint64_t>::max() / output_blocks)
        return false;
    const uint64_t groups = output_blocks * contract.logical_k;
    if (groups > std::numeric_limits<uint64_t>::max() / 64u)
        return false;
    return contract.group_count == groups && contract.values_bytes == groups * 64u &&
           contract.scale_bytes == groups * 2u && contract.bias_bytes == groups * 2u &&
           contract.values_bytes <= std::numeric_limits<size_t>::max() &&
           contract.scale_bytes <= std::numeric_limits<size_t>::max() &&
           contract.bias_bytes <= std::numeric_limits<size_t>::max();
}

static bool column_grouped_affine_uint2_skip_v1_planes_valid(
    const ColumnGroupedAffineUInt2SkipV1Contract& contract,
    const ColumnGroupedAffineUInt2SkipV1Planes& planes) {
    if (!planes.values || !planes.scales || !planes.biases ||
        planes.values_bytes != static_cast<size_t>(contract.values_bytes) ||
        planes.scale_count != static_cast<size_t>(contract.group_count) ||
        planes.bias_count != static_cast<size_t>(contract.group_count) ||
        (reinterpret_cast<uintptr_t>(planes.values) % contract.plane_alignment) != 0u ||
        (reinterpret_cast<uintptr_t>(planes.scales) % contract.plane_alignment) != 0u ||
        (reinterpret_cast<uintptr_t>(planes.biases) % contract.plane_alignment) != 0u)
        return false;
    if (planes.scale_count > std::numeric_limits<size_t>::max() / sizeof(uint16_t))
        return false;
    const size_t metadata_bytes = planes.scale_count * sizeof(uint16_t);
    return !affine_v1_overlap(planes.values, planes.values_bytes, planes.scales,
                              metadata_bytes) &&
           !affine_v1_overlap(planes.values, planes.values_bytes, planes.biases,
                              metadata_bytes) &&
           !affine_v1_overlap(planes.scales, metadata_bytes, planes.biases,
                              metadata_bytes);
}

static bool affine_v1_contract_valid(const ColumnGroupedAffineLowBitV1Contract& contract) {
    if (contract.version != 1u || contract.reserved != 0u || contract.logical_k == 0u ||
        contract.logical_n == 0u || (contract.bits != 2u && contract.bits != 3u && contract.bits != 4u) ||
        contract.group_elements != 256u || contract.logical_n % contract.group_elements != 0u ||
        contract.scale_bytes_per_group != sizeof(uint16_t) ||
        contract.bias_bytes_per_group != sizeof(uint16_t) || contract.plane_alignment != 128u)
        return false;
    const uint32_t expected_packed_bytes = (256u * contract.bits + 7u) / 8u;
    const uint64_t expected_group_count =
        static_cast<uint64_t>(contract.logical_k) * (contract.logical_n / 256u);
    if (contract.packed_bytes != expected_packed_bytes ||
        contract.group_count != expected_group_count)
        return false;
    return expected_group_count <= std::numeric_limits<size_t>::max() / expected_packed_bytes &&
           contract.values_bytes == expected_group_count * expected_packed_bytes;
}

static bool affine_v1_range_end(const void* pointer, size_t bytes, uintptr_t* end) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(pointer);
    if (bytes > std::numeric_limits<uintptr_t>::max() - start) return false;
    *end = start + bytes;
    return true;
}

static bool affine_v1_overlap(const void* left, size_t left_bytes,
                              const void* right, size_t right_bytes) {
    uintptr_t left_end = 0;
    uintptr_t right_end = 0;
    if (!affine_v1_range_end(left, left_bytes, &left_end) ||
        !affine_v1_range_end(right, right_bytes, &right_end)) return true;
    const uintptr_t left_start = reinterpret_cast<uintptr_t>(left);
    const uintptr_t right_start = reinterpret_cast<uintptr_t>(right);
    return left_start < right_end && right_start < left_end;
}

static bool affine_v1_planes_valid(const ColumnGroupedAffineLowBitV1Contract& contract,
                                   const ColumnGroupedAffineLowBitV1Planes& planes) {
    if (!planes.values || !planes.scales || !planes.biases ||
        planes.values_bytes != static_cast<size_t>(contract.values_bytes) ||
        planes.scale_count != static_cast<size_t>(contract.group_count) ||
        planes.bias_count != static_cast<size_t>(contract.group_count)) return false;
    if ((reinterpret_cast<uintptr_t>(planes.values) % contract.plane_alignment) != 0u ||
        (reinterpret_cast<uintptr_t>(planes.scales) % contract.plane_alignment) != 0u ||
        (reinterpret_cast<uintptr_t>(planes.biases) % contract.plane_alignment) != 0u)
        return false;
    if (planes.scale_count > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) return false;
    const size_t scale_bytes = planes.scale_count * sizeof(uint16_t);
    const size_t bias_bytes = planes.bias_count * sizeof(uint16_t);
    if (affine_v1_overlap(planes.values, planes.values_bytes, planes.scales, scale_bytes) ||
        affine_v1_overlap(planes.values, planes.values_bytes, planes.biases, bias_bytes) ||
        affine_v1_overlap(planes.scales, scale_bytes, planes.biases, bias_bytes)) return false;
    for (size_t index = 0; index != planes.scale_count; ++index) {
        const float scale = fp16_to_fp32(planes.scales[index]);
        const float bias = fp16_to_fp32(planes.biases[index]);
        if (!std::isfinite(scale) || scale < 0.0f || !std::isfinite(bias)) return false;
    }
    return true;
}
#endif

static void init() {
    std::call_once(g_init, []{
        setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", 0);
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) return;
        g_q = [g_dev newCommandQueue];
        g_xbuf = [g_dev newBufferWithLength:2*1024*1024 options:MTLResourceStorageModeShared];
        g_ybuf = [g_dev newBufferWithLength:32*1024*1024 options:MTLResourceStorageModeShared];
        NSError* err = nil;
        MTLCompileOptions* opt = [MTLCompileOptions new];
        if (@available(macOS 26.0, *))
            opt.languageVersion = MTLLanguageVersion4_0;
        g_lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:src_gemv]
                                    options:opt error:&err];
        if (!g_lib) {
            err = nil;
            g_lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:src_gemv]
                                        options:nil error:&err];
        }
        if (!g_lib) { fprintf(stderr, "[metal] %s\n", err ? [[err localizedDescription] UTF8String] : "?"); g_dev = nil; return; }
        bool m4fam = false;
        if (@available(macOS 26.0, *))
            m4fam = [g_dev supportsFamily:(MTLGPUFamily)5002];
        if (m4fam) {
            std::string flat;
            std::unordered_set<std::string> seen;
            const char* hdr = mpp_header_dir();
            flatten_metal(src_matmul2d, hdr, hdr, flat, seen);
            MTLCompileOptions* opt4 = [MTLCompileOptions new];
            opt4.languageVersion = MTLLanguageVersion4_0;
            opt4.maxTotalThreadsPerThreadgroup = 128;
            NSError* err4 = nil;
            g_m4_lib = [g_dev newLibraryWithSource:
                [NSString stringWithUTF8String:flat.c_str()]
                options:opt4 error:&err4];
            if (!g_m4_lib) {
                fprintf(stderr, "[metal] matmul2d compile failed: %s\n",
                        err4 ? [[err4 localizedDescription] UTF8String] : "?");
            } else {
                g_m4 = true;
                fprintf(stderr, "[metal] %s metal4=1 matmul2d=1\n",
                        [[g_dev name] UTF8String]);
            }
        }
        if (!g_m4)
            fprintf(stderr, "[metal] %s metal4=0\n", [[g_dev name] UTF8String]);
#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
        if (@available(macOS 27.0, *)) {
            std::string flat;
            std::unordered_set<std::string> seen;
            const char* hdr = mpp_header_dir();
            flatten_metal(src_mpp_int2, hdr, hdr, flat, seen);
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion4_1;
            options.maxTotalThreadsPerThreadgroup = 64;
            NSError* error = nil;
            g_mpp_int2_lib = [g_dev newLibraryWithSource:
                [NSString stringWithUTF8String:flat.c_str()]
                options:options error:&error];
            if (!g_mpp_int2_lib)
                fprintf(stderr, "[metal] MPP Int2 compile failed: %s\n",
                        error ? error.localizedDescription.UTF8String : "unknown");
        }
#endif
    });
}

// Lazily compile the GEMM library (simdgroup_matrix). Called on first
// metal_gemm(M>1) call, not during init(), to avoid blocking decode-only
// sessions with a slow compile.
static void init_gemm() {
    std::call_once(g_gemm_init, []{
        if (!g_dev) return;
        NSError* err = nil;
        g_gemm_lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:src_gemm] options:nil error:&err];
        if (!g_gemm_lib) { fprintf(stderr, "[metal] gemm lib: %s\n", err ? [[err localizedDescription] UTF8String] : "?"); }
    });
}

static id<MTLComputePipelineState> get_sampler_pipe() {
    std::call_once(g_sampler_init, [] {
        if (!g_dev) return;
        NSError* error = nil;
        g_sampler_lib = [g_dev newLibraryWithSource:
            [NSString stringWithUTF8String:src_sampler] options:nil error:&error];
        if (!g_sampler_lib) {
            std::fprintf(stderr, "[metal] sampler library: %s\n",
                         error ? error.localizedDescription.UTF8String : "unknown");
            return;
        }
        id<MTLFunction> function = [g_sampler_lib newFunctionWithName:@"sampler_greedy_f32"];
        if (!function) {
            std::fprintf(stderr, "[metal] sampler function missing\n");
            return;
        }
        g_sampler_pipe = [g_dev newComputePipelineStateWithFunction:function error:&error];
        [function release];
        if (!g_sampler_pipe)
            std::fprintf(stderr, "[metal] sampler pipeline: %s\n",
                         error ? error.localizedDescription.UTF8String : "unknown");
    });
    return g_sampler_pipe;
}

static id<MTLComputePipelineState> get_prefill_f16_pipe() {
    std::call_once(g_prefill_f16_init, []{
        if (!g_dev) return;
        NSError* error = nil;
        g_prefill_f16_lib = [g_dev newLibraryWithSource:
            [NSString stringWithUTF8String:src_prefill_f16] options:nil error:&error];
        if (!g_prefill_f16_lib) {
            fprintf(stderr, "[metal] prefill_f16 library: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "?");
            return;
        }
        id<MTLFunction> function = [g_prefill_f16_lib newFunctionWithName:@"prefill_f16_rows"];
        g_prefill_f16_pipe = [g_dev newComputePipelineStateWithFunction:function error:&error];
        if (!g_prefill_f16_pipe) {
            fprintf(stderr, "[metal] prefill_f16 pipeline: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "?");
        }
    });
    return g_prefill_f16_pipe;
}

// Create a specialized pipeline for a given quant type using function constants.
// The Metal compiler bakes in QUANT_TYPE, eliminating the dequant switch and
// dead-code-eliminating unused kernel branches. Called on first encounter of
// each type, then cached in g_pipes.
static id<MTLComputePipelineState> get_pipe(int type) {
    {
        std::lock_guard<std::mutex> lk(g_pipe_mtx);
        auto it = g_pipes.find(type);
        if (it != g_pipes.end()) return it->second;
    }
    if (!g_lib) return nil;
    MTLFunctionConstantValues* vals = [MTLFunctionConstantValues new];
    int t = type;
    [vals setConstantValue:&t type:MTLDataTypeInt atIndex:0];
    NSError* err = nil;
    id<MTLFunction> f = [g_lib newFunctionWithName:@"gemv" constantValues:vals error:&err];
    if (!f) { fprintf(stderr, "[metal] function for type %d: %s\n", type, err ? [[err localizedDescription] UTF8String] : "?"); return nil; }
    id<MTLComputePipelineState> pipe = [g_dev newComputePipelineStateWithFunction:f error:&err];
    if (!pipe) { fprintf(stderr, "[metal] pipeline for type %d: %s\n", type, err ? [[err localizedDescription] UTF8String] : "?"); return nil; }
    std::lock_guard<std::mutex> lk(g_pipe_mtx);
    g_pipes[type] = pipe;
    return pipe;
}

static id<MTLComputePipelineState> get_sparse_pipe(
    const char* name, int type, bool affine,
    std::unordered_map<int, id<MTLComputePipelineState>>& cache) {
    const int key = type | (affine ? 0x100 : 0);
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;
    if (!g_lib || (type != static_cast<int>(GGMLType::Q4_K) &&
                   type != static_cast<int>(GGMLType::Q6_K))) return nil;
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    [values setConstantValue:&type type:MTLDataTypeInt atIndex:0];
    [values setConstantValue:&affine type:MTLDataTypeBool atIndex:1];
    NSError* error = nil;
    id<MTLFunction> function = [g_lib newFunctionWithName:[NSString stringWithUTF8String:name]
                                           constantValues:values error:&error];
    if (!function) return nil;
    id<MTLComputePipelineState> pipeline = [g_dev newComputePipelineStateWithFunction:function error:&error];
    if (pipeline) cache[key] = pipeline;
    return pipeline;
}

static id<MTLComputePipelineState> get_q4k_pipe() {
    if (g_q4k_pipe) return g_q4k_pipe;
    if (!g_lib) return nil;
    NSError* err = nil;
    id<MTLFunction> f = [g_lib newFunctionWithName:@"gemv_q4k"];
    if (!f) {
        fprintf(stderr, "[metal] gemv_q4k: missing kernel\n");
        return nil;
    }
    g_q4k_pipe = [g_dev newComputePipelineStateWithFunction:f error:&err];
    if (!g_q4k_pipe)
        fprintf(stderr, "[metal] gemv_q4k pipe: %s\n",
                err ? [[err localizedDescription] UTF8String] : "?");
    return g_q4k_pipe;
}

static void mark_fused_q4k() {
    static std::once_flag once;
    std::call_once(once, []{ fprintf(stderr, "[metal] fused_q4k=1\n"); });
}

static id<MTLBuffer> get_weight_buf(const void* ptr, size_t len, size_t& offset);
static id<MTLBuffer> find_mmap_buf_locked(const void* ptr, size_t len, size_t& offset);
static bool strict_tensor_data_span(const Tensor& tensor, uint64_t expected);

static bool bind_q4k(id<MTLComputeCommandEncoder> enc, const Tensor& w,
                     int K, int N, id<MTLBuffer> xbuf, size_t xoff,
                     id<MTLBuffer> ybuf, size_t yoff) {
    if (w.type != GGMLType::Q4_K || w.n_dims != 2 || (K % 256) != 0 || N < 1 ||
        w.dims[0] != static_cast<uint64_t>(K) || w.dims[1] != static_cast<uint64_t>(N)) return false;
    const uint64_t row_bytes = (static_cast<uint64_t>(K) / 256u) * bytes_per_block(w.type);
    if (static_cast<uint64_t>(N) > UINT64_MAX / row_bytes ||
        !strict_tensor_data_span(w, static_cast<uint64_t>(N) * row_bytes)) return false;
    id<MTLComputePipelineState> pipe = get_q4k_pipe();
    if (!pipe || !xbuf || !ybuf) return false;
    uint64_t rb = row_bytes;
    size_t w_off = 0;
    id<MTLBuffer> wb = get_weight_buf(w.data, (size_t)N * rb, w_off);
    if (!wb) return false;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:wb offset:w_off atIndex:0];
    [enc setBuffer:xbuf offset:xoff atIndex:1];
    [enc setBuffer:ybuf offset:yoff atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc setBytes:&N length:4 atIndex:4];
    [enc setBytes:&rb length:8 atIndex:5];
    int M = 1;
    [enc setBytes:&M length:4 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)N + 3) / 4, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    mark_fused_q4k();
    return true;
}

static id<MTLComputePipelineState> get_q2k_pipe() {
    if (g_q2k_pipe) return g_q2k_pipe;
    if (!g_lib) return nil;
    NSError* err = nil;
    id<MTLFunction> f = [g_lib newFunctionWithName:@"gemv_q2k"];
    if (!f) return nil;
    g_q2k_pipe = [g_dev newComputePipelineStateWithFunction:f error:&err];
    if (!g_q2k_pipe)
        fprintf(stderr, "[metal] gemv_q2k pipe: %s\n",
                err ? [[err localizedDescription] UTF8String] : "?");
    return g_q2k_pipe;
}

#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
static bool bind_q2k_two_row(id<MTLComputeCommandEncoder>, const Tensor&,
                             int, int, id<MTLBuffer>, size_t,
                             id<MTLBuffer>, size_t);
static bool bind_q2k_streamed(id<MTLComputeCommandEncoder>, const Tensor&,
                              int, int, id<MTLBuffer>, size_t,
                              id<MTLBuffer>, size_t);
#endif

static bool bind_q2k(id<MTLComputeCommandEncoder> enc, const Tensor& w,
                     int K, int N, id<MTLBuffer> xbuf, size_t xoff,
                     id<MTLBuffer> ybuf, size_t yoff) {
#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
    if (g_test_q2k_two_row_pipeline)
        return bind_q2k_two_row(enc, w, K, N, xbuf, xoff, ybuf, yoff);
#endif
    if (w.type != GGMLType::Q2_K || w.n_dims != 2 || (K % 256) != 0 || N < 1 ||
        w.dims[0] != static_cast<uint64_t>(K) || w.dims[1] != static_cast<uint64_t>(N)) return false;
    const uint64_t row_bytes = (static_cast<uint64_t>(K) / 256u) * bytes_per_block(w.type);
    if (static_cast<uint64_t>(N) > UINT64_MAX / row_bytes ||
        !strict_tensor_data_span(w, static_cast<uint64_t>(N) * row_bytes)) return false;
    id<MTLComputePipelineState> pipe = get_q2k_pipe();
    if (!pipe || pipe.threadExecutionWidth != 32 ||
        pipe.maxTotalThreadsPerThreadgroup < 64 || !xbuf || !ybuf) return false;
    uint64_t rb = row_bytes;
    size_t w_off = 0;
    id<MTLBuffer> wb = get_weight_buf(w.data, (size_t)N * (size_t)rb, w_off);
    if (!wb) return false;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:wb offset:w_off atIndex:0];
    [enc setBuffer:xbuf offset:xoff atIndex:1];
    [enc setBuffer:ybuf offset:yoff atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc setBytes:&N length:4 atIndex:4];
    [enc setBytes:&rb length:8 atIndex:5];
    int M = 1;
    [enc setBytes:&M length:4 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)N + 7) / 8, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    return true;
}

static id<MTLComputePipelineState> get_iq2_xxs_pipe() {
    if (g_iq2_xxs_pipe) return g_iq2_xxs_pipe;
    if (!g_lib) return nil;
    NSError* error = nil;
    id<MTLFunction> function = [g_lib newFunctionWithName:@"gemv_iq2_xxs"];
    if (!function) return nil;
    g_iq2_xxs_pipe = [g_dev newComputePipelineStateWithFunction:function error:&error];
    if (!g_iq2_xxs_pipe)
        fprintf(stderr, "[metal] gemv_iq2_xxs pipe: %s\n",
                error ? [[error localizedDescription] UTF8String] : "?");
    return g_iq2_xxs_pipe;
}

static bool bind_iq2_xxs(id<MTLComputeCommandEncoder> encoder, const Tensor& weight,
                         int K, int N, id<MTLBuffer> input, size_t input_offset,
                         id<MTLBuffer> output, size_t output_offset) {
    if (weight.type != GGMLType::IQ2_XXS || weight.n_dims != 2 || (K % 256) != 0 || N < 1 ||
        weight.dims[0] != static_cast<uint64_t>(K) || weight.dims[1] != static_cast<uint64_t>(N)) return false;
    const uint64_t row_bytes = (static_cast<uint64_t>(K) / 256u) * 66u;
    if (static_cast<uint64_t>(N) > UINT64_MAX / row_bytes ||
        !strict_tensor_data_span(weight, static_cast<uint64_t>(N) * row_bytes)) return false;
    id<MTLComputePipelineState> pipeline = get_iq2_xxs_pipe();
    if (!pipeline || pipeline.threadExecutionWidth != 32 ||
        pipeline.maxTotalThreadsPerThreadgroup < 64 || !input || !output) return false;
    size_t weight_offset = 0;
    id<MTLBuffer> weight_buffer =
        get_weight_buf(weight.data, static_cast<size_t>(N) * row_bytes, weight_offset);
    if (!weight_buffer) return false;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:weight_buffer offset:weight_offset atIndex:0];
    [encoder setBuffer:input offset:input_offset atIndex:1];
    [encoder setBuffer:output offset:output_offset atIndex:2];
    [encoder setBytes:&K length:sizeof(K) atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&row_bytes length:sizeof(row_bytes) atIndex:5];
    const int M = 1;
    [encoder setBytes:&M length:sizeof(M) atIndex:6];
    [encoder setThreadgroupMemoryLength:256 * sizeof(uint64_t) + 128 atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake((static_cast<NSUInteger>(N) + 7) / 8, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    return true;
}

static id<MTLComputePipelineState> get_iq1_s_pipe() {
    if (g_iq1_s_pipe) return g_iq1_s_pipe;
    if (!g_lib) return nil;
    NSError* error = nil;
    id<MTLFunction> function = [g_lib newFunctionWithName:@"gemv_iq1_s"];
    if (!function) return nil;
    g_iq1_s_pipe = [g_dev newComputePipelineStateWithFunction:function error:&error];
    if (!g_iq1_s_pipe)
        fprintf(stderr, "[metal] gemv_iq1_s pipe: %s\n",
                error ? [[error localizedDescription] UTF8String] : "?");
    return g_iq1_s_pipe;
}

static id<MTLBuffer> get_iq1_s_grid() {
    if (g_iq1_s_grid) return g_iq1_s_grid;
    if (!g_dev) return nil;
    std::array<uint32_t, 2048> packed{};
    for (size_t grid_index = 0; grid_index != packed.size(); ++grid_index) {
        const uint64_t levels = kIq1SGrid[grid_index];
        for (unsigned lane = 0; lane != 4; ++lane) {
            const int8_t first = static_cast<int8_t>(levels >> (8 * lane));
            const int8_t second = static_cast<int8_t>(levels >> (8 * (lane + 4)));
            const uint8_t byte = static_cast<uint8_t>(first + 1) |
                static_cast<uint8_t>((second + 1) << 4);
            packed[grid_index] |= static_cast<uint32_t>(byte) << (8 * lane);
        }
    }
    g_iq1_s_grid = [g_dev newBufferWithBytes:packed.data()
                                      length:sizeof(packed)
                                     options:MTLResourceStorageModeShared];
    return g_iq1_s_grid;
}

static bool bind_iq1_s(id<MTLComputeCommandEncoder> encoder, const Tensor& weight,
                       int K, int N, id<MTLBuffer> input, size_t input_offset,
                       id<MTLBuffer> output, size_t output_offset) {
    if (weight.type != GGMLType::IQ1_S || weight.n_dims != 2 || (K % 256) != 0 || N < 1 ||
        weight.dims[0] != static_cast<uint64_t>(K) || weight.dims[1] != static_cast<uint64_t>(N)) return false;
    const uint64_t row_bytes = (static_cast<uint64_t>(K) / 256u) * 50u;
    if (static_cast<uint64_t>(N) > UINT64_MAX / row_bytes ||
        !strict_tensor_data_span(weight, static_cast<uint64_t>(N) * row_bytes)) return false;
    id<MTLComputePipelineState> pipeline = get_iq1_s_pipe();
    id<MTLBuffer> grid = get_iq1_s_grid();
    if (!pipeline || pipeline.threadExecutionWidth != 32 ||
        pipeline.maxTotalThreadsPerThreadgroup < 64 || !grid || !input || !output) return false;
    size_t weight_offset = 0;
    id<MTLBuffer> weight_buffer =
        get_weight_buf(weight.data, static_cast<size_t>(N) * row_bytes, weight_offset);
    if (!weight_buffer) return false;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:weight_buffer offset:weight_offset atIndex:0];
    [encoder setBuffer:input offset:input_offset atIndex:1];
    [encoder setBuffer:output offset:output_offset atIndex:2];
    [encoder setBytes:&K length:sizeof(K) atIndex:3];
    [encoder setBytes:&N length:sizeof(N) atIndex:4];
    [encoder setBytes:&row_bytes length:sizeof(row_bytes) atIndex:5];
    const int M = 1;
    [encoder setBytes:&M length:sizeof(M) atIndex:6];
    [encoder setBuffer:grid offset:0 atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake((static_cast<NSUInteger>(N) + 7) / 8, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    return true;
}

#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
static id<MTLComputePipelineState> get_q2k_two_row_pipe() {
    if (g_q2k_two_row_pipe) return g_q2k_two_row_pipe;
    if (!g_lib) return nil;
    NSError* error = nil;
    id<MTLFunction> function = [g_lib newFunctionWithName:@"gemv_q2k_two_row"];
    if (!function) return nil;
    g_q2k_two_row_pipe = [g_dev newComputePipelineStateWithFunction:function error:&error];
    if (!g_q2k_two_row_pipe)
        fprintf(stderr, "[metal] gemv_q2k_two_row pipe: %s\n",
                error ? [[error localizedDescription] UTF8String] : "?");
    return g_q2k_two_row_pipe;
}

static bool bind_q2k_two_row(id<MTLComputeCommandEncoder> encoder, const Tensor& weight,
                             int K, int N, id<MTLBuffer> input, size_t input_offset,
                             id<MTLBuffer> output, size_t output_offset) {
    if (weight.type != GGMLType::Q2_K || weight.n_dims != 2 || (K % 256) != 0 || N < 1 ||
        weight.dims[0] != static_cast<uint64_t>(K) ||
        weight.dims[1] != static_cast<uint64_t>(N)) return false;
    id<MTLComputePipelineState> pipeline = get_q2k_two_row_pipe();
    if (!pipeline || pipeline.threadExecutionWidth != 32 ||
        pipeline.maxTotalThreadsPerThreadgroup < 64 || !input || !output) return false;
    const uint64_t row_bytes = ((uint64_t)K / 256) * bytes_per_block(weight.type);
    if (static_cast<uint64_t>(N) > UINT64_MAX / row_bytes ||
        !strict_tensor_data_span(weight, static_cast<uint64_t>(N) * row_bytes)) return false;
    size_t weight_offset = 0;
    id<MTLBuffer> weight_buffer = get_weight_buf(
        weight.data, (size_t)N * (size_t)row_bytes, weight_offset);
    if (!weight_buffer) return false;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:weight_buffer offset:weight_offset atIndex:0];
    [encoder setBuffer:input offset:input_offset atIndex:1];
    [encoder setBuffer:output offset:output_offset atIndex:2];
    [encoder setBytes:&K length:4 atIndex:3];
    [encoder setBytes:&N length:4 atIndex:4];
    [encoder setBytes:&row_bytes length:8 atIndex:5];
    int M = 1;
    [encoder setBytes:&M length:4 atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(((NSUInteger)N + 3) / 4, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    return true;
}

static id<MTLComputePipelineState> get_q2k_streamed_pipe() {
    if (g_q2k_streamed_pipe) return g_q2k_streamed_pipe;
    if (!g_lib) return nil;
    NSError* error = nil;
    id<MTLFunction> function = [g_lib newFunctionWithName:@"gemv_q2k_streamed"];
    if (!function) return nil;
    g_q2k_streamed_pipe = [g_dev newComputePipelineStateWithFunction:function error:&error];
    if (!g_q2k_streamed_pipe)
        fprintf(stderr, "[metal] gemv_q2k_streamed pipe: %s\n",
                error ? [[error localizedDescription] UTF8String] : "?");
    return g_q2k_streamed_pipe;
}

static bool bind_q2k_streamed(id<MTLComputeCommandEncoder> encoder, const Tensor& weight,
                              int K, int N, id<MTLBuffer> input, size_t input_offset,
                              id<MTLBuffer> output, size_t output_offset) {
    if (weight.type != GGMLType::Q2_K || weight.n_dims != 2 || (K % 256) != 0 || N < 1 ||
        weight.dims[0] != static_cast<uint64_t>(K) ||
        weight.dims[1] != static_cast<uint64_t>(N)) return false;
    id<MTLComputePipelineState> pipeline = get_q2k_streamed_pipe();
    if (!pipeline || pipeline.threadExecutionWidth != 32 ||
        pipeline.maxTotalThreadsPerThreadgroup < 64 || !input || !output) return false;
    const uint64_t row_bytes = ((uint64_t)K / 256) * bytes_per_block(weight.type);
    if (static_cast<uint64_t>(N) > UINT64_MAX / row_bytes ||
        !strict_tensor_data_span(weight, static_cast<uint64_t>(N) * row_bytes)) return false;
    size_t weight_offset = 0;
    id<MTLBuffer> weight_buffer = get_weight_buf(
        weight.data, (size_t)N * (size_t)row_bytes, weight_offset);
    if (!weight_buffer) return false;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:weight_buffer offset:weight_offset atIndex:0];
    [encoder setBuffer:input offset:input_offset atIndex:1];
    [encoder setBuffer:output offset:output_offset atIndex:2];
    [encoder setBytes:&K length:4 atIndex:3];
    [encoder setBytes:&N length:4 atIndex:4];
    [encoder setBytes:&row_bytes length:8 atIndex:5];
    int M = 1;
    [encoder setBytes:&M length:4 atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(((NSUInteger)N + 7) / 8, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    return true;
}
#endif

static id<MTLComputePipelineState> get_q4k_id_pipe() {
    if (g_q4k_id_pipe) return g_q4k_id_pipe;
    if (!g_lib) return nil;
    NSError* err = nil;
    id<MTLFunction> f = [g_lib newFunctionWithName:@"gemv_q4k_id"];
    if (!f) {
        fprintf(stderr, "[metal] gemv_q4k_id: missing kernel\n");
        return nil;
    }
    g_q4k_id_pipe = [g_dev newComputePipelineStateWithFunction:f error:&err];
    if (!g_q4k_id_pipe)
        fprintf(stderr, "[metal] gemv_q4k_id pipe: %s\n",
                err ? [[err localizedDescription] UTF8String] : "?");
    return g_q4k_id_pipe;
}

static void mark_fused_q4k_id() {
    static std::once_flag once;
    std::call_once(once, []{ fprintf(stderr, "[metal] fused_q4k_id=1\n"); });
}

static bool bind_q4k_id(id<MTLComputeCommandEncoder> enc, const Tensor& w,
                        int K, int N, const int* ids, int n_used,
                        id<MTLBuffer> xbuf, size_t xoff,
                        id<MTLBuffer> ybuf, size_t yoff) {
    if (w.type != GGMLType::Q4_K || w.n_dims < 3 || (K % 256) != 0 || N < 1 || n_used < 1 ||
        w.dims[0] != static_cast<uint64_t>(K) || w.dims[1] != static_cast<uint64_t>(N))
        return false;
    if (!ids || w.n_dims < 3) return false;
    id<MTLComputePipelineState> pipe = get_q4k_id_pipe();
    if (!pipe || !xbuf || !ybuf) return false;
    const int n_total = (int)w.dims[2];
    if (n_total <= 0) return false;
    uint64_t rb = ((uint64_t)K / 256) * bytes_per_block(w.type);
    if (rb == 0 || static_cast<uint64_t>(N) > UINT64_MAX / rb) return false;
    uint64_t expert_stride = static_cast<uint64_t>(N) * rb;
    if (static_cast<uint64_t>(n_total) > UINT64_MAX / expert_stride) return false;
    uint64_t span_u64 = static_cast<uint64_t>(n_total) * expert_stride;
    if (span_u64 > SIZE_MAX) return false;
    size_t w_off = 0;
    size_t span = static_cast<size_t>(span_u64);
    if (span == 0) span = 1;
    if (!strict_tensor_data_span(w, span)) return false;
    id<MTLBuffer> wb = get_weight_buf(w.data, span, w_off);
    if (!wb) return false;
    int x_mul = 0;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:wb offset:w_off atIndex:0];
    [enc setBuffer:xbuf offset:xoff atIndex:1];
    [enc setBuffer:ybuf offset:yoff atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc setBytes:&N length:4 atIndex:4];
    [enc setBytes:&rb length:8 atIndex:5];
    [enc setBytes:&x_mul length:4 atIndex:6];
    static id<MTLBuffer> idscratch;
    static size_t idscratch_n = 0;
    if (!idscratch || idscratch_n < (size_t)n_used) {
        idscratch = [g_dev newBufferWithLength:(size_t)n_used * 4
                                       options:MTLResourceStorageModeShared];
        idscratch_n = (size_t)n_used;
    }
    if (!idscratch) return false;
    memcpy([idscratch contents], ids, (size_t)n_used * 4);
    [enc setBuffer:idscratch offset:0 atIndex:7];
    [enc setBytes:&expert_stride length:8 atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)N + 3) / 4, 1, (NSUInteger)n_used)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    mark_fused_q4k_id();
    return true;
}

static bool bind_q4k_id_dev(id<MTLComputeCommandEncoder> enc, const Tensor& w,
                            int K, int N, id<MTLBuffer> ids_buf, size_t ids_off,
                            int n_used, id<MTLBuffer> xbuf, size_t xoff,
                            id<MTLBuffer> ybuf, size_t yoff, int x_mul) {
    if (w.type != GGMLType::Q4_K || w.n_dims < 3 || (K % 256) != 0 || N < 1 || n_used < 1 ||
        w.dims[0] != static_cast<uint64_t>(K) || w.dims[1] != static_cast<uint64_t>(N))
        return false;
    if (!ids_buf || w.n_dims < 3) return false;
    id<MTLComputePipelineState> pipe = get_q4k_id_pipe();
    if (!pipe || !xbuf || !ybuf) return false;
    const int n_total = (int)w.dims[2];
    if (n_total <= 0) return false;
    uint64_t rb = ((uint64_t)K / 256) * bytes_per_block(w.type);
    if (rb == 0 || static_cast<uint64_t>(N) > UINT64_MAX / rb) return false;
    uint64_t expert_stride = static_cast<uint64_t>(N) * rb;
    if (static_cast<uint64_t>(n_total) > UINT64_MAX / expert_stride) return false;
    uint64_t span_u64 = static_cast<uint64_t>(n_total) * expert_stride;
    if (span_u64 > SIZE_MAX) return false;
    size_t w_off = 0;
    size_t span = static_cast<size_t>(span_u64);
    if (span == 0) span = 1;
    if (!strict_tensor_data_span(w, span)) return false;
    id<MTLBuffer> wb = get_weight_buf(w.data, span, w_off);
    if (!wb) return false;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:wb offset:w_off atIndex:0];
    [enc setBuffer:xbuf offset:xoff atIndex:1];
    [enc setBuffer:ybuf offset:yoff atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc setBytes:&N length:4 atIndex:4];
    [enc setBytes:&rb length:8 atIndex:5];
    [enc setBytes:&x_mul length:4 atIndex:6];
    [enc setBuffer:ids_buf offset:ids_off atIndex:7];
    [enc setBytes:&expert_stride length:8 atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)N + 3) / 4, 1, (NSUInteger)n_used)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    mark_fused_q4k_id();
    return true;
}

static id<MTLComputePipelineState> g_q8_id_pipe;
static id<MTLComputePipelineState> get_q8_id_pipe() {
    if (g_q8_id_pipe) return g_q8_id_pipe;
    if (!g_lib) return nil;
    NSError* err = nil;
    id<MTLFunction> f = [g_lib newFunctionWithName:@"gemv_q8_id"];
    if (!f) return nil;
    g_q8_id_pipe = [g_dev newComputePipelineStateWithFunction:f error:&err];
    return g_q8_id_pipe;
}

static bool bind_q8_id_dev(id<MTLComputeCommandEncoder> enc, const Tensor& w,
                           int K, int N, id<MTLBuffer> ids_buf, size_t ids_off,
                           int n_used, id<MTLBuffer> xbuf, size_t xoff,
                           id<MTLBuffer> ybuf, size_t yoff, int x_mul) {
    if (w.type != GGMLType::Q8_0 || w.n_dims < 3 || (K % 32) != 0 || N < 1 || n_used < 1 ||
        w.dims[0] != static_cast<uint64_t>(K) || w.dims[1] != static_cast<uint64_t>(N))
        return false;
    if (!ids_buf || w.n_dims < 3) return false;
    id<MTLComputePipelineState> pipe = get_q8_id_pipe();
    if (!pipe || !xbuf || !ybuf) return false;
    const int n_total = (int)w.dims[2];
    if (n_total <= 0) return false;
    uint64_t rb = ((uint64_t)K / 32) * bytes_per_block(w.type);
    if (rb == 0 || static_cast<uint64_t>(N) > UINT64_MAX / rb) return false;
    uint64_t expert_stride = static_cast<uint64_t>(N) * rb;
    if (static_cast<uint64_t>(n_total) > UINT64_MAX / expert_stride) return false;
    uint64_t span_u64 = static_cast<uint64_t>(n_total) * expert_stride;
    if (span_u64 > SIZE_MAX) return false;
    size_t w_off = 0;
    size_t span = static_cast<size_t>(span_u64);
    if (span == 0) span = 1;
    if (!strict_tensor_data_span(w, span)) return false;
    id<MTLBuffer> wb = get_weight_buf(w.data, span, w_off);
    if (!wb) return false;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:wb offset:w_off atIndex:0];
    [enc setBuffer:xbuf offset:xoff atIndex:1];
    [enc setBuffer:ybuf offset:yoff atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc setBytes:&N length:4 atIndex:4];
    [enc setBytes:&rb length:8 atIndex:5];
    [enc setBytes:&x_mul length:4 atIndex:6];
    [enc setBuffer:ids_buf offset:ids_off atIndex:7];
    [enc setBytes:&expert_stride length:8 atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)N + 3) / 4, 1, (NSUInteger)n_used)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    return true;
}

static std::unordered_map<int, id<MTLComputePipelineState>> g_id_pipes;
static id<MTLComputePipelineState> get_id_pipe(int type) {
    auto it = g_id_pipes.find(type);
    if (it != g_id_pipes.end()) return it->second;
    if (!g_lib) return nil;
    MTLFunctionConstantValues* vals = [MTLFunctionConstantValues new];
    [vals setConstantValue:&type type:MTLDataTypeInt atIndex:0];
    NSError* err = nil;
    id<MTLFunction> f = [g_lib newFunctionWithName:@"gemv_id" constantValues:vals error:&err];
    if (!f) return nil;
    id<MTLComputePipelineState> pipe = [g_dev newComputePipelineStateWithFunction:f error:&err];
    if (pipe) g_id_pipes[type] = pipe;
    return pipe;
}

static bool bind_gemv_id(id<MTLComputeCommandEncoder> enc, const Tensor& w,
                         int K, int N, id<MTLBuffer> ids_buf, size_t ids_off,
                         int n_used, id<MTLBuffer> xbuf, size_t xoff,
                         id<MTLBuffer> ybuf, size_t yoff, int x_mul) {
    if (N < 1 || n_used < 1 || !ids_buf || w.n_dims < 3 ||
        w.dims[0] != static_cast<uint64_t>(K) || w.dims[1] != static_cast<uint64_t>(N)) return false;
    id<MTLComputePipelineState> pipe = get_id_pipe((int)w.type);
    if (!pipe || !xbuf || !ybuf) return false;
    const int n_total = (int)w.dims[2];
    if (n_total <= 0) return false;
    uint64_t rb = ((uint64_t)K + elements_per_block(w.type) - 1)
                  / elements_per_block(w.type) * bytes_per_block(w.type);
    if (rb == 0 || static_cast<uint64_t>(N) > UINT64_MAX / rb) return false;
    uint64_t expert_stride = static_cast<uint64_t>(N) * rb;
    if (static_cast<uint64_t>(n_total) > UINT64_MAX / expert_stride) return false;
    uint64_t span_u64 = static_cast<uint64_t>(n_total) * expert_stride;
    if (span_u64 > SIZE_MAX) return false;
    size_t w_off = 0;
    size_t span = static_cast<size_t>(span_u64);
    if (span == 0) span = 1;
    if (!strict_tensor_data_span(w, span)) return false;
    id<MTLBuffer> wb = get_weight_buf(w.data, span, w_off);
    if (!wb) return false;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:wb offset:w_off atIndex:0];
    [enc setBuffer:xbuf offset:xoff atIndex:1];
    [enc setBuffer:ybuf offset:yoff atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc setBytes:&N length:4 atIndex:4];
    [enc setBytes:&rb length:8 atIndex:5];
    [enc setBytes:&x_mul length:4 atIndex:6];
    [enc setBuffer:ids_buf offset:ids_off atIndex:7];
    [enc setBytes:&expert_stride length:8 atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N, 1, (NSUInteger)n_used)
       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return true;
}

#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
static id<MTLComputePipelineState> get_moe_down_reduce_pipe(int type) {
    auto it = g_moe_down_reduce_pipes.find(type);
    if (it != g_moe_down_reduce_pipes.end()) return it->second;
    if (!g_lib || (type != static_cast<int>(GGMLType::Q5_0) &&
                   type != static_cast<int>(GGMLType::Q8_0))) return nil;
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    [values setConstantValue:&type type:MTLDataTypeInt atIndex:0];
    NSError* error = nil;
    id<MTLFunction> function = [g_lib newFunctionWithName:@"moe_down_reduce"
                                           constantValues:values error:&error];
    if (!function) return nil;
    id<MTLComputePipelineState> pipeline =
        [g_dev newComputePipelineStateWithFunction:function error:&error];
    [function release];
    [values release];
    if (pipeline) {
        g_moe_down_reduce_pipes[type] = pipeline;
        fprintf(stderr, "[metal] moe_down_reduce=%s\n", type_name(static_cast<GGMLType>(type)));
    } else {
        fprintf(stderr, "[metal] moe_down_reduce %s pipe: %s\n",
                type_name(static_cast<GGMLType>(type)),
                error ? [[error localizedDescription] UTF8String] : "?");
    }
    return pipeline;
}

bool metal_test_moe_down_reduce(
    const MetalMoeDownReduceSpec& spec, const Tensor& down, size_t source_bytes,
    const float* input, size_t input_values, const uint32_t* expert_ids,
    const float* route_weights, const float* expert_scales,
    size_t expert_scale_values, float* output, size_t output_values,
    double* gpu_ms, MetalMoeDownReducePipelineCaps* capabilities) {
    if (gpu_ms) *gpu_ms = 0.0;
    if (capabilities) *capabilities = {};
    init();
    if (!g_dev || !g_q || !g_lib || !input || !expert_ids || !route_weights || !output ||
        !gpu_ms || !capabilities || !down.data || down.n_dims != 3 ||
        (down.type != GGMLType::Q5_0 && down.type != GGMLType::Q8_0) ||
        spec.input_width == 0 || spec.output_width == 0 || spec.expert_count == 0 ||
        spec.expert_count > 1024 || spec.selected_count == 0 || spec.selected_count > 16 ||
        spec.selected_count > spec.expert_count ||
        spec.input_width > static_cast<uint32_t>(INT_MAX) ||
        spec.output_width > static_cast<uint32_t>(INT_MAX))
        return false;

    const uint64_t K = spec.input_width;
    const uint64_t N = spec.output_width;
    const uint64_t E = spec.expert_count;
    const uint64_t R = spec.selected_count;
    if (down.dims[0] != K || down.dims[1] != N || down.dims[2] != E ||
        K % static_cast<uint64_t>(elements_per_block(down.type)) != 0)
        return false;
    const uint64_t row_bytes = (K / elements_per_block(down.type)) *
                               bytes_per_block(down.type);
    if (row_bytes == 0 || N > std::numeric_limits<uint64_t>::max() / row_bytes)
        return false;
    const uint64_t expert_stride = N * row_bytes;
    if (expert_stride == 0 || E > std::numeric_limits<uint64_t>::max() / expert_stride)
        return false;
    const uint64_t expected_source_bytes = E * expert_stride;
    if (expected_source_bytes == 0 ||
        expected_source_bytes > std::numeric_limits<size_t>::max() ||
        source_bytes != static_cast<size_t>(expected_source_bytes))
        return false;
    if (R > std::numeric_limits<size_t>::max() / static_cast<size_t>(K) ||
        input_values != static_cast<size_t>(R * K) ||
        output_values != static_cast<size_t>(N))
        return false;
    if ((expert_scales == nullptr) != (expert_scale_values == 0) ||
        (expert_scales && expert_scale_values != static_cast<size_t>(E)))
        return false;
    for (uint32_t slot = 0; slot != spec.selected_count; ++slot) {
        if (expert_ids[slot] >= spec.expert_count || !std::isfinite(route_weights[slot]))
            return false;
    }
    if (expert_scales) {
        for (uint32_t expert = 0; expert != spec.expert_count; ++expert)
            if (!std::isfinite(expert_scales[expert])) return false;
    }

    id<MTLComputePipelineState> pipeline =
        get_moe_down_reduce_pipe(static_cast<int>(down.type));
    if (!pipeline || pipeline.threadExecutionWidth == 0 ||
        pipeline.maxTotalThreadsPerThreadgroup < pipeline.threadExecutionWidth)
        return false;
    capabilities->thread_execution_width =
        static_cast<uint32_t>(pipeline.threadExecutionWidth);
    capabilities->max_total_threads_per_threadgroup =
        static_cast<uint32_t>(pipeline.maxTotalThreadsPerThreadgroup);

    const size_t input_bytes = input_values * sizeof(float);
    const size_t output_bytes = output_values * sizeof(float);
    const size_t worklist_bytes = static_cast<size_t>(R) * sizeof(uint32_t);
    const size_t route_bytes = static_cast<size_t>(R) * sizeof(float);
    const bool has_scales = expert_scales != nullptr;
    const size_t scale_bytes = has_scales ? static_cast<size_t>(E) * sizeof(float) : sizeof(float);
    const float unit_scale = 1.0f;
    @autoreleasepool {
        id<MTLBuffer> input_buffer =
            [g_dev newBufferWithBytes:input length:input_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> ids_buffer =
            [g_dev newBufferWithBytes:expert_ids length:worklist_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> route_buffer =
            [g_dev newBufferWithBytes:route_weights length:route_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> scale_buffer = has_scales
            ? [g_dev newBufferWithBytes:expert_scales length:scale_bytes options:MTLResourceStorageModeShared]
            : [g_dev newBufferWithBytes:&unit_scale length:scale_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer =
            [g_dev newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
        size_t weight_offset = 0;
        id<MTLBuffer> weight_buffer = get_weight_buf(down.data, source_bytes, weight_offset);
        id<MTLCommandBuffer> command =
            (input_buffer && ids_buffer && route_buffer && scale_buffer && output_buffer && weight_buffer)
                ? [g_q commandBuffer] : nil;
        id<MTLComputeCommandEncoder> encoder = command ? [command computeCommandEncoder] : nil;
        if (!encoder) {
            [input_buffer release];
            [ids_buffer release];
            [route_buffer release];
            [scale_buffer release];
            [output_buffer release];
            return false;
        }
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:weight_buffer offset:weight_offset atIndex:0];
        [encoder setBuffer:input_buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer offset:0 atIndex:2];
        const int K_arg = static_cast<int>(K);
        const int N_arg = static_cast<int>(N);
        const int selected_arg = static_cast<int>(R);
        [encoder setBytes:&K_arg length:sizeof(K_arg) atIndex:3];
        [encoder setBytes:&N_arg length:sizeof(N_arg) atIndex:4];
        [encoder setBytes:&row_bytes length:sizeof(row_bytes) atIndex:5];
        [encoder setBytes:&expert_stride length:sizeof(expert_stride) atIndex:6];
        [encoder setBuffer:ids_buffer offset:0 atIndex:7];
        [encoder setBuffer:route_buffer offset:0 atIndex:8];
        [encoder setBuffer:scale_buffer offset:0 atIndex:9];
        [encoder setBytes:&selected_arg length:sizeof(selected_arg) atIndex:10];
        [encoder setBytes:&has_scales length:sizeof(has_scales) atIndex:11];
        const NSUInteger threads = static_cast<NSUInteger>(pipeline.threadExecutionWidth);
        [encoder dispatchThreadgroups:MTLSizeMake((static_cast<NSUInteger>(N) + threads - 1) / threads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        const auto wait_start = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "[metal] generic MoE down/reduce command: %s\n",
                         command_buffer_failure_detail(command).c_str());
            [input_buffer release];
            [ids_buffer release];
            [route_buffer release];
            [scale_buffer release];
            [output_buffer release];
            return false;
        }
        record_completed_command(command, wait_start, std::chrono::steady_clock::now());
        const CFTimeInterval gpu_start = [command GPUStartTime];
        const CFTimeInterval gpu_end = [command GPUEndTime];
        if (gpu_end >= gpu_start) *gpu_ms = 1000.0 * (gpu_end - gpu_start);
        std::memcpy(output, [output_buffer contents], output_bytes);
        [input_buffer release];
        [ids_buffer release];
        [route_buffer release];
        [scale_buffer release];
        [output_buffer release];
        return true;
    }
}
#endif

static id<MTLComputePipelineState> get_q8_pipe() {
    if (g_q8_pipe) return g_q8_pipe;
    if (!g_lib) return nil;
    NSError* err = nil;
    id<MTLFunction> f = [g_lib newFunctionWithName:@"gemv_q8"];
    if (!f) {
        fprintf(stderr, "[metal] gemv_q8: missing kernel\n");
        return nil;
    }
    g_q8_pipe = [g_dev newComputePipelineStateWithFunction:f error:&err];
    if (!g_q8_pipe)
        fprintf(stderr, "[metal] gemv_q8 pipe: %s\n",
                err ? [[err localizedDescription] UTF8String] : "?");
    return g_q8_pipe;
}

static void mark_fused_q8() {
    static std::once_flag once;
    std::call_once(once, []{ fprintf(stderr, "[metal] fused_q8=1\n"); });
}

static bool bind_q8(id<MTLComputeCommandEncoder> enc, const Tensor& w,
                    int K, int N, id<MTLBuffer> xbuf, size_t xoff,
                    id<MTLBuffer> ybuf, size_t yoff) {
    if (w.type != GGMLType::Q8_0 || w.n_dims != 2 || (K % 32) != 0 || N < 1 ||
        w.dims[0] != static_cast<uint64_t>(K) || w.dims[1] != static_cast<uint64_t>(N)) return false;
    const uint64_t row_bytes = (static_cast<uint64_t>(K) / 32u) * bytes_per_block(w.type);
    if (static_cast<uint64_t>(N) > UINT64_MAX / row_bytes ||
        !strict_tensor_data_span(w, static_cast<uint64_t>(N) * row_bytes)) return false;
    id<MTLComputePipelineState> pipe = get_q8_pipe();
    if (!pipe || !xbuf || !ybuf) return false;
    uint64_t rb = row_bytes;
    size_t w_off = 0;
    id<MTLBuffer> wb = get_weight_buf(w.data, (size_t)N * rb, w_off);
    if (!wb) return false;
    int M = 1;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:wb offset:w_off atIndex:0];
    [enc setBuffer:xbuf offset:xoff atIndex:1];
    [enc setBuffer:ybuf offset:yoff atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc setBytes:&N length:4 atIndex:4];
    [enc setBytes:&rb length:8 atIndex:5];
    [enc setBytes:&M length:4 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)N + 3) / 4, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    mark_fused_q8();
    return true;
}

// Pipeline for the gemm_simdgroup kernel. Uses the same QUANT_TYPE function
// constant to specialize the dequant function used inside the kernel.
static id<MTLComputePipelineState> get_gemm_pipe(int type) {
    {
        std::lock_guard<std::mutex> lk(g_pipe_mtx);
        auto it = g_gemm_pipes.find(type);
        if (it != g_gemm_pipes.end()) return it->second;
    }
    init_gemm();
    if (!g_gemm_lib) return nil;
    MTLFunctionConstantValues* vals = [MTLFunctionConstantValues new];
    int t = type;
    [vals setConstantValue:&t type:MTLDataTypeInt atIndex:0];
    NSError* err = nil;
    id<MTLFunction> f = [g_gemm_lib newFunctionWithName:@"gemm_simdgroup" constantValues:vals error:&err];
    if (!f) { fprintf(stderr, "[metal] gemm function for type %d: %s\n", type, err ? [[err localizedDescription] UTF8String] : "?"); return nil; }
    id<MTLComputePipelineState> pipe = [g_dev newComputePipelineStateWithFunction:f error:&err];
    if (!pipe) { fprintf(stderr, "[metal] gemm pipeline for type %d: %s\n", type, err ? [[err localizedDescription] UTF8String] : "?"); return nil; }
    std::lock_guard<std::mutex> lk(g_pipe_mtx);
    g_gemm_pipes[type] = pipe;
    return pipe;
}

bool metal_available() { init(); return g_dev != nil; }
bool metal_device_present() { return MTLCreateSystemDefaultDevice() != nil; }

static bool metal_sampler_descriptor_valid(const MetalSamplerDescriptor& descriptor) {
    return descriptor.version == 1 &&
           descriptor.mode == MetalSamplerMode::Greedy &&
           descriptor.tie_policy == MetalSamplerTiePolicy::FirstIndex &&
           descriptor.nonfinite_policy == MetalSamplerNonFinitePolicy::Reject &&
           descriptor.reserved == 0 && descriptor.temperature == 1.0f &&
           descriptor.top_k == 0 && descriptor.top_p == 1.0f &&
           descriptor.rng_seed == 0 && descriptor.rng_counter == 0;
}

bool metal_sampler_greedy(const MetalSamplerDescriptor& descriptor,
                          const MetalSamplerDeviceLogits& logits,
                          MetalSamplerResult* result) {
    if (!result || !metal_sampler_descriptor_valid(descriptor) || !logits.buffer ||
        logits.vocabulary == 0 ||
        logits.byte_offset % alignof(float) != 0) return false;
    const size_t bytes = static_cast<size_t>(logits.vocabulary) * sizeof(float);
    if (bytes / sizeof(float) != logits.vocabulary) return false;
    init();
    id<MTLBuffer> source = (id<MTLBuffer>)(const_cast<void*>(logits.buffer));
    id<MTLComputePipelineState> pipeline = get_sampler_pipe();
    if (!g_dev || !g_q || !source || !pipeline ||
        logits.byte_offset > [source length] ||
        bytes > [source length] - logits.byte_offset) return false;
    const NSUInteger threads = std::min<NSUInteger>(256u,
        pipeline.maxTotalThreadsPerThreadgroup);
    if (threads == 0) return false;

    @autoreleasepool {
    id<MTLBuffer> result_buffer = [g_dev newBufferWithLength:sizeof(MetalSamplerResult)
                                                       options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> encoder = command ? [command computeCommandEncoder] : nil;
    if (!result_buffer || !command || !encoder) {
        [result_buffer release];
        return false;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:source offset:logits.byte_offset atIndex:0];
    [encoder setBuffer:result_buffer offset:0 atIndex:1];
    const uint32_t vocabulary = logits.vocabulary;
    [encoder setBytes:&vocabulary length:sizeof(vocabulary) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    const auto wait_start = std::chrono::steady_clock::now();
    [command commit];
    [command waitUntilCompleted];
    const auto wait_end = std::chrono::steady_clock::now();
    const bool completed = command.status == MTLCommandBufferStatusCompleted;
    if (completed) record_completed_command(command, wait_start, wait_end);
    bool ok = false;
    if (completed) {
        std::memcpy(result, [result_buffer contents], sizeof(MetalSamplerResult));
        ok = result->status == MetalSamplerResultStatus::Success;
    }
    if (!completed)
        std::fprintf(stderr, "[metal] sampler command: %s\n",
                     command_buffer_failure_detail(command).c_str());
    [result_buffer release];
    return ok;
    }
}

#if defined(LAPLACE_TESTING)
bool metal_test_sampler_greedy(const MetalSamplerDescriptor& descriptor,
                               const float* logits, uint32_t vocabulary,
                               MetalSamplerResult* result,
                               uint32_t* command_buffers) {
    if (command_buffers) *command_buffers = 0;
    if (!logits || vocabulary == 0 || !result || !command_buffers) return false;
    init();
    const size_t bytes = static_cast<size_t>(vocabulary) * sizeof(float);
    if (!g_dev || bytes / sizeof(float) != vocabulary) return false;
    id<MTLBuffer> input = [g_dev newBufferWithBytes:logits length:bytes
                                             options:MTLResourceStorageModeShared];
    if (!input) return false;
    const MetalDispatchMetrics before = metal_dispatch_metrics();
    const MetalSamplerDeviceLogits device_logits{input, 0, vocabulary};
    const bool ok = metal_sampler_greedy(descriptor, device_logits, result);
    const MetalDispatchMetrics after = metal_dispatch_metrics();
    *command_buffers = static_cast<uint32_t>(after.command_buffers - before.command_buffers);
    [input release];
    return ok;
}
#endif

#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
uint32_t metal_test_thermal_state() {
    if (@available(macOS 10.10.3, *)) {
        return static_cast<uint32_t>([NSProcessInfo processInfo].thermalState);
    }
    return UINT32_MAX;
}
#endif

// Create zero-copy Metal buffers covering the mmap'd GGUF file. The pointer
// is page-aligned (mmap guarantees this). The length is rounded up to page
// size. If the mmap exceeds maxBufferLength, it is split into multiple
// buffers. Tensors are later resolved to (buffer, offset) at dispatch time.
static bool metal_register_mmap_impl(const void* base, size_t size,
                                     size_t forced_max_buffer,
                                     uint32_t fail_after_chunk) {
    if (!g_dev || !base || size == 0) return false;
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return false;
    const size_t page = static_cast<size_t>(page_size);
    const size_t remainder = size % page;
    if (remainder != 0 && size > SIZE_MAX - (page - remainder)) return false;
    const size_t aligned = size + (remainder == 0 ? 0 : page - remainder);
    size_t max_buf = static_cast<size_t>(g_dev.maxBufferLength);
    if (max_buf == 0) return false;
    if (forced_max_buffer != 0 && forced_max_buffer < max_buf) max_buf = forced_max_buffer;
    if (max_buf < page) return false;
    max_buf -= max_buf % page;
    if (max_buf == 0) return false;
    const uint8_t* basep = (const uint8_t*)base;
    // Overlap chunks so a 3-D expert stack near a split still lives in one
    // MTLBuffer. Id kernels address experts by stride from tensor start.
    size_t overlap = 1024ull * 1024ull * 1024ull;
    if (overlap > (size_t)max_buf / 4) overlap = (size_t)max_buf / 4;
    overlap -= overlap % page;
    size_t off = 0;
    int n_bufs = 0;
    bool complete = true;
    MetalWeightContext& context = *g_active_weight_context;
    std::vector<MmapBuf> staged;
    try {
        while (off < aligned) {
            size_t chunk = aligned - off;
            if (chunk > max_buf) chunk = max_buf;
            chunk -= chunk % page;
            if (chunk == 0) chunk = page;
            if (fail_after_chunk != UINT32_MAX &&
                static_cast<uint32_t>(n_bufs) == fail_after_chunk) {
                complete = false;
                fprintf(stderr, "[metal] injected mmap buffer failure at chunk %d\n", n_bufs);
                break;
            }
            const uint8_t* ptr = basep + off;
            id<MTLBuffer> buf = [g_dev newBufferWithBytesNoCopy:(void*)ptr
                                                        length:chunk
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
            if (!buf) {
                fprintf(stderr, "[metal] mmap buffer failed at %p size %zu\n", ptr, chunk);
                complete = false;
                break;
            }
            try {
                staged.push_back({buf, ptr, chunk, basep});
            } catch (...) {
                [buf release];
                complete = false;
                break;
            }
            n_bufs++;
            const size_t remaining = aligned - off;
            if (chunk >= remaining) {
                off = aligned;
                break;
            }
            const size_t step = chunk > overlap ? chunk - overlap : chunk;
            if (step == 0 || off > aligned - step) {
                complete = false;
                break;
            }
            off += step;
        }
    } catch (...) {
        complete = false;
    }

    if (!complete || off != aligned || staged.empty()) {
        for (const MmapBuf& buffer : staged) [buffer.buf release];
        fprintf(stderr, "[metal] model %.1f GB, registration rolled back after %d buffer(s)\n",
                size / 1e9, n_bufs);
        return false;
    }

    try {
        std::lock_guard<std::mutex> lk(context.mutex);
        context.mmap_bufs.reserve(context.mmap_bufs.size() + staged.size());
        context.mmap_bufs.insert(context.mmap_bufs.end(), staged.begin(), staged.end());
    } catch (...) {
        for (const MmapBuf& buffer : staged) [buffer.buf release];
        return false;
    }
    staged.clear();
    fprintf(stderr, "[metal] model %.1f GB, %d zero-copy buffer(s)\n", size/1e9, n_bufs);
    return true;
}

static bool metal_register_mmap(const void* base, size_t size) {
    return metal_register_mmap_impl(base, size, 0, UINT32_MAX);
}

#if defined(LAPLACE_TESTING)
static bool metal_register_mmap_for_testing(const void* base, size_t size,
                                            size_t chunk_limit,
                                            uint32_t fail_after_chunk) {
    return metal_register_mmap_impl(base, size, chunk_limit, fail_after_chunk);
}
#endif

void metal_register_weights(const void* base, size_t size) {
    init();
    (void)metal_register_mmap(base, size);
}

void metal_unregister_weights(const void* base) {
    if (!base) return;
    MetalWeightContext& context = *g_active_weight_context;
    std::lock_guard<std::mutex> lk(context.mutex);
    auto first = std::remove_if(context.mmap_bufs.begin(), context.mmap_bufs.end(), [&](const MmapBuf& buffer) {
        if (buffer.registration_base != static_cast<const uint8_t*>(base)) return false;
        [buffer.buf release];
        return true;
    });
    context.mmap_bufs.erase(first, context.mmap_bufs.end());
}

// Look up a tensor pointer in the registered mmap buffers. Returns the Metal
// buffer and sets offset to the byte offset within that buffer. If the tensor
// spans two buffers (at a split boundary), returns nil so the caller falls
// back to the copy path.
static id<MTLBuffer> find_mmap_buf_locked(const void* ptr, size_t len, size_t& offset) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    for (const auto& mb : g_active_weight_context->mmap_bufs) {
        const uintptr_t base = reinterpret_cast<uintptr_t>(mb.base);
        if (address < base) continue;
        const uintptr_t delta = address - base;
        if (delta <= mb.size && len <= mb.size - static_cast<size_t>(delta)) {
            offset = static_cast<size_t>(delta);
            return mb.buf;
        }
    }
    return nil;
}

static size_t mmap_buf_coverage_count_locked(const void* ptr, size_t len) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    size_t count = 0;
    for (const MmapBuf& buffer : g_active_weight_context->mmap_bufs) {
        const uintptr_t base = reinterpret_cast<uintptr_t>(buffer.base);
        if (address < base) continue;
        const uintptr_t delta = address - base;
        if (delta <= buffer.size && len <= buffer.size - static_cast<size_t>(delta)) ++count;
    }
    return count;
}

static id<MTLBuffer> get_weight_buf(const void* ptr, size_t len, size_t& offset) {
    offset = 0;
    {
        MetalWeightContext& context = *g_active_weight_context;
        std::lock_guard<std::mutex> lk(context.mutex);
        const char* nommap = std::getenv("LAPLACE_NO_MMAP_GPU");
        if (!(nommap && nommap[0] == '1')) {
            id<MTLBuffer> mb = find_mmap_buf_locked(ptr, len, offset);
            if (mb) return mb;
        }
        size_t dummy = 0;
        const bool mmap_addr = find_mmap_buf_locked(ptr, 1, dummy) != nil;
        if (context.require_registered_weights) return nil;
        if (mmap_addr) {
            auto it = context.copied_bufs.find(ptr);
            if (it != context.copied_bufs.end()) {
                offset = 0;
                return it->second;
            }
        }
        madvise((void*)ptr, len, MADV_WILLNEED);
        id<MTLBuffer> buf = [g_dev newBufferWithBytes:ptr length:len
                                             options:MTLResourceStorageModeShared];
        if (buf) ++context.implicit_copy_count;
        if (buf && mmap_addr) {
            context.copied_bufs[ptr] = buf;
            static int ncopy = 0;
            if (ncopy++ < 4)
                fprintf(stderr, "[metal] weight copy %.1f MB (mmap split)\n", len / 1e6);
        }
        if (buf) madvise((void*)ptr, len, MADV_DONTNEED);
        return buf;
    }
}

// A canonical session may register a larger artifact range for zero-copy
// access.  The Tensor view still owns the authority for the exact bytes a
// binder may read.  Legacy diagnostic callers have no span metadata, so they
// retain the historical non-strict behavior.
static bool strict_tensor_data_span(const Tensor& tensor, uint64_t expected) {
    if (!g_active_weight_context->require_registered_weights) return true;
    return tensor.data != nullptr && expected != 0 && expected <= SIZE_MAX &&
           tensor.data_bytes == expected;
}

struct NativePack {
    id<MTLBuffer> qs = nil;
    id<MTLBuffer> scale = nil;
    id<MTLBuffer> amin = nil;
};
static std::unordered_map<const void*, NativePack> g_pack;
static std::mutex g_pack_mtx;

static NativePack* get_or_pack(const Tensor& w, int K, int N) {
    if (!w.data || K <= 0 || N <= 0 || (K % 32) != 0) return nullptr;
    if (w.type != GGMLType::Q6_K && w.type != GGMLType::Q8_0 &&
        w.type != GGMLType::Q4_K) return nullptr;
    std::lock_guard<std::mutex> lk(g_pack_mtx);
    auto it = g_pack.find(w.data);
    if (it != g_pack.end()) return &it->second;
    const int ng = K / 16;
    const size_t qbytes = (size_t)N * (size_t)K;
    const size_t sbytes = (size_t)N * (size_t)ng * 2;
    id<MTLBuffer> qb = [g_dev newBufferWithLength:qbytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> sb = [g_dev newBufferWithLength:sbytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> mb = [g_dev newBufferWithLength:sbytes options:MTLResourceStorageModeShared];
    if (!qb || !sb || !mb) return nullptr;
    auto* q = (int8_t*)[qb contents];
    auto* sc = (uint16_t*)[sb contents];
    auto* am = (uint16_t*)[mb contents];
    std::memset(am, 0, sbytes);
    if (w.type == GGMLType::Q8_0) {
        const int nb = K / 32, rb = nb * 34;
        for (int n = 0; n < N; n++) {
            const auto* row = (const kernels::block_q8_0*)(w.data + (size_t)n * rb);
            for (int b = 0; b < nb; b++) {
                std::memcpy(q + (size_t)n * K + b * 32, row[b].qs, 32);
                sc[n * ng + b * 2] = row[b].d;
                sc[n * ng + b * 2 + 1] = row[b].d;
            }
        }
    } else if (w.type == GGMLType::Q4_K) {
        const int nsb = K / 256;
        const int rb = nsb * (int)sizeof(kernels::block_q4_K);
        for (int n = 0; n < N; n++) {
            const auto* row = (const kernels::block_q4_K*)(w.data + (size_t)n * rb);
            int8_t* qn = q + (size_t)n * K;
            for (int sbk = 0; sbk < nsb; sbk++) {
                const auto& blk = row[sbk];
                float d = fp16_to_fp32(blk.d), dmin = fp16_to_fp32(blk.dmin);
                for (int j = 0; j < 8; j++) {
                    uint8_t sv, mv;
                    kernels::get_scale_min_k4(j, blk.scales, &sv, &mv);
                    int gi = n * ng + sbk * 16 + j * 2;
                    sc[gi] = sc[gi + 1] = fp32_to_fp16(d * (float)sv);
                    am[gi] = am[gi + 1] = fp32_to_fp16(dmin * (float)mv);
                }
                const uint8_t* qq = blk.qs;
                int8_t* qd = qn + sbk * 256;
                for (int jb = 0; jb < 256; jb += 64) {
                    for (int l = 0; l < 32; l++) {
                        qd[jb + l] = (int8_t)(qq[l] & 0xF);
                        qd[jb + 32 + l] = (int8_t)(qq[l] >> 4);
                    }
                    qq += 32;
                }
            }
        }
    } else {
        const int nsb = K / 256;
        const int rb = nsb * (int)sizeof(kernels::block_q6_K);
        for (int n = 0; n < N; n++) {
            const auto* row = (const kernels::block_q6_K*)(w.data + (size_t)n * rb);
            int8_t* qn = q + (size_t)n * K;
            for (int sbk = 0; sbk < nsb; sbk++) {
                const auto& blk = row[sbk];
                float d = fp16_to_fp32(blk.d);
                for (int g = 0; g < 16; g++)
                    sc[n * ng + sbk * 16 + g] = fp32_to_fp16(d * (float)blk.scales[g]);
                const uint8_t* ql = blk.ql;
                const uint8_t* qh = blk.qh;
                int8_t* qd = qn + sbk * 256;
                for (int n_off = 0; n_off < 256; n_off += 128) {
                    for (int l = 0; l < 32; l++) {
                        qd[n_off + l] = (int8_t)((int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                        qd[n_off + l + 32] = (int8_t)((int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                        qd[n_off + l + 64] = (int8_t)((int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                        qd[n_off + l + 96] = (int8_t)((int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                    }
                    ql += 64;
                    qh += 32;
                }
            }
        }
    }
    NativePack p{qb, sb, mb};
    auto [ins, ok] = g_pack.emplace(w.data, p);
    (void)ok;
    static int pack_logs = 0;
    if (pack_logs++ < 2)
        fprintf(stderr, "[metal] packed type=%d N=%d K=%d (%.1f MB)\n",
                (int)w.type, N, K, (qbytes + 2 * sbytes) / 1e6);
    return &ins->second;
}

void metal_pack_tensor(const Tensor& w, int K, int N) {
    init();
    if (!g_dev) return;
    get_or_pack(w, K, N);
}

static void mark_tensorops() {
    static std::once_flag once;
    std::call_once(once, []{ fprintf(stderr, "[metal] tensorops_decode=1\n"); });
}

static bool bind_rows(id<MTLComputeCommandEncoder> enc, const Tensor& w,
                      int K, int N, id<MTLBuffer> xbuf, size_t xoff,
                      id<MTLBuffer> ybuf, size_t yoff) {
    if (!g_m4 || !g_m4_lib) return false;
    // LM head (N=262144) TensorOps was 15.3/5.0 tok/s. FFN/MoE tiles are
    // 44-88 groups. Keep the huge GEMV on CPU.
    if (N < 512 || N >= 65536) return false;
    if (const char* no = std::getenv("LAPLACE_NO_TENSOROPS"); no && no[0] == '1')
        return false;
    if (w.type != GGMLType::Q8_0) return false;
    if (K <= 0 || (K % 32) != 0) return false;
    const uint64_t source_row_bytes = (static_cast<uint64_t>(K) / 32u) * bytes_per_block(w.type);
    if (source_row_bytes == 0 || static_cast<uint64_t>(N) > UINT64_MAX / source_row_bytes ||
        !strict_tensor_data_span(w, static_cast<uint64_t>(N) * source_row_bytes)) return false;
    if (!g_m4_pipe) {
        NSError* err = nil;
        id<MTLFunction> f = [g_m4_lib newFunctionWithName:@"gemv_m4_rows"];
        if (!f) return false;
        g_m4_pipe = [g_dev newComputePipelineStateWithFunction:f error:&err];
        if (!g_m4_pipe) return false;
    }
    NativePack* p = get_or_pack(w, K, N);
    if (!p) return false;
    int M = 1;
    uint64_t z = 0;
    [enc setComputePipelineState:g_m4_pipe];
    [enc setBuffer:p->qs offset:0 atIndex:0];
    [enc setBuffer:xbuf offset:xoff atIndex:1];
    [enc setBuffer:ybuf offset:yoff atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc setBytes:&N length:4 atIndex:4];
    [enc setBytes:&z length:8 atIndex:5];
    [enc setBytes:&M length:4 atIndex:6];
    [enc setBuffer:p->scale offset:0 atIndex:7];
    [enc setBuffer:p->amin offset:0 atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake((N + 31) / 32, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    mark_tensorops();
    return true;
}

static bool bind_fused(id<MTLComputeCommandEncoder> enc, const Tensor& w,
                       int K, int N, id<MTLBuffer> xbuf, size_t xoff,
                       id<MTLBuffer> ybuf, size_t yoff) {
    if (w.type == GGMLType::Q2_K) return bind_q2k(enc, w, K, N, xbuf, xoff, ybuf, yoff);
    if (w.type == GGMLType::IQ2_XXS) return bind_iq2_xxs(enc, w, K, N, xbuf, xoff, ybuf, yoff);
    if (w.type == GGMLType::IQ1_S) return bind_iq1_s(enc, w, K, N, xbuf, xoff, ybuf, yoff);
    if (w.type == GGMLType::Q4_K) return bind_q4k(enc, w, K, N, xbuf, xoff, ybuf, yoff);
    if (w.type == GGMLType::Q8_0) return bind_q8(enc, w, K, N, xbuf, xoff, ybuf, yoff);
    id<MTLComputePipelineState> pipe = get_pipe((int)w.type);
    if (!pipe || !xbuf || !ybuf) return false;
    if (N < 1) return false;
    if (w.type != GGMLType::F32 && w.type != GGMLType::F16 &&
        w.type != GGMLType::Q2_K && N < 512)
        return false;
    uint64_t rb = ((uint64_t)K + elements_per_block(w.type) - 1)
                  / elements_per_block(w.type) * bytes_per_block(w.type);
    if (static_cast<uint64_t>(N) > UINT64_MAX / rb ||
        !strict_tensor_data_span(w, static_cast<uint64_t>(N) * rb)) return false;
    size_t w_off = 0;
    id<MTLBuffer> wb = get_weight_buf(w.data, (size_t)N * rb, w_off);
    if (!wb) return false;
    int M = 1;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:wb offset:w_off atIndex:0];
    [enc setBuffer:xbuf offset:xoff atIndex:1];
    [enc setBuffer:ybuf offset:yoff atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc setBytes:&N length:4 atIndex:4];
    [enc setBytes:&rb length:8 atIndex:5];
    [enc setBytes:&M length:4 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N * (NSUInteger)M, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return true;
}

static bool bind_q6k(id<MTLComputeCommandEncoder> enc, const Tensor& w,
                     int K, int N, id<MTLBuffer> xbuf, size_t xoff,
                     id<MTLBuffer> ybuf, size_t yoff);

static bool bind_any(id<MTLComputeCommandEncoder> enc, const Tensor& w,
                     int K, int N, id<MTLBuffer> xbuf, size_t xoff,
                     id<MTLBuffer> ybuf, size_t yoff) {
    // Q8_0 TensorOps first: fused Q8 downs on this CB were 12.4 tok/s
    // vs 21.5 with pack. Q4_K has no TensorOps path and uses bind_fused.
    if (bind_rows(enc, w, K, N, xbuf, xoff, ybuf, yoff)) return true;
    return bind_fused(enc, w, K, N, xbuf, xoff, ybuf, yoff);
}

static bool supported(GGMLType t) {
    switch (t) {
        case GGMLType::F32: case GGMLType::F16: case GGMLType::BF16:
        case GGMLType::Q4_0: case GGMLType::Q4_1: case GGMLType::Q5_0:
        case GGMLType::Q5_1: case GGMLType::Q8_0: case GGMLType::Q4_K:
        case GGMLType::Q6_K: case GGMLType::Q2_K: case GGMLType::Q3_K:
        case GGMLType::Q5_K: case GGMLType::IQ2_XXS: case GGMLType::IQ1_S: return true;
        default: return false;
    }
}

bool metal_gemm(const float* x, const Tensor& w, float* y, int M, int K, int N) {
    init();
    if (!g_dev || !g_lib || N <= 0 || K <= 0 || M <= 0) return false;
    if (!supported(w.type)) return false;
    if (N >= 65536 && w.type != GGMLType::Q6_K) return false;
    int type = (int)w.type;
    uint64_t rb = ((uint64_t)K + elements_per_block(w.type) - 1) / elements_per_block(w.type) * bytes_per_block(w.type);

    if (M > 1) {
        return false;
        // GEMM path: simdgroup_matrix kernel for prefill
        id<MTLComputePipelineState> pipe = get_gemm_pipe(type);
        if (!pipe) return false;
        @autoreleasepool {
            std::lock_guard<std::mutex> lk(g_xy_mtx);
            size_t wbytes = (size_t)N * rb;
            size_t w_off = 0;
            id<MTLBuffer> wb = get_weight_buf(w.data, wbytes, w_off);
            if (!wb) {
                wb = [g_dev newBufferWithBytes:w.data length:wbytes options:MTLResourceStorageModeShared];
                w_off = 0;
            }
            size_t xbytes = (size_t)M * K * 4, ybytes = (size_t)M * N * 4;
            id<MTLBuffer> xb = [g_dev newBufferWithBytes:x length:xbytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> yb = [g_dev newBufferWithLength:ybytes options:MTLResourceStorageModeShared];
            if (!wb || !xb || !yb) return false;
            int n_tg_x = (N + 31) / 32;
            int n_tg_y = (M + 31) / 32;
            id<MTLCommandBuffer> cmd = [g_q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setBuffer:wb offset:w_off atIndex:0];
            [enc setBuffer:xb offset:0 atIndex:1];
            [enc setBuffer:yb offset:0 atIndex:2];
            [enc setBytes:&K length:4 atIndex:3];
            [enc setBytes:&N length:4 atIndex:4];
            [enc setBytes:&rb length:8 atIndex:5];
            [enc setBytes:&M length:4 atIndex:6];
            [enc setBytes:&n_tg_x length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(n_tg_x * n_tg_y, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            const auto wait_start = std::chrono::steady_clock::now();
            [cmd commit]; [cmd waitUntilCompleted];
            record_completed_command(cmd, wait_start, std::chrono::steady_clock::now());
            memcpy(y, [yb contents], ybytes);
        }
        return true;
    }

    // GEMV path: M == 1. TensorOps on the LM head (N>=65536).
    @autoreleasepool {
        std::lock_guard<std::mutex> lk(g_xy_mtx);
        size_t xbytes = (size_t)M * K * 4, ybytes = (size_t)M * N * 4;
        id<MTLBuffer> xb = nil, yb = nil;
        if (g_xbuf && xbytes <= (size_t)g_xbuf.length) {
            memcpy([g_xbuf contents], x, xbytes);
            xb = g_xbuf;
        } else {
            xb = [g_dev newBufferWithBytes:x length:xbytes options:MTLResourceStorageModeShared];
        }
        yb = [g_dev newBufferWithLength:ybytes options:MTLResourceStorageModeShared];
        if (!xb || !yb) return false;
        id<MTLCommandBuffer> cmd = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        bool native = false;
        if (w.type == GGMLType::IQ1_S)
            native = bind_iq1_s(enc, w, K, N, xb, 0, yb, 0);
        else if (w.type == GGMLType::IQ2_XXS)
            native = bind_iq2_xxs(enc, w, K, N, xb, 0, yb, 0);
        else if (w.type == GGMLType::Q4_K)
            native = bind_q4k(enc, w, K, N, xb, 0, yb, 0);
        else if (w.type == GGMLType::Q8_0)
            native = bind_q8(enc, w, K, N, xb, 0, yb, 0);
        if (!native)
            native = bind_rows(enc, w, K, N, xb, 0, yb, 0);
        if (!native) {
            id<MTLComputePipelineState> pipe = get_pipe(type);
            if (!pipe) return false;
            size_t wbytes = (size_t)N * rb;
            size_t w_off = 0;
            id<MTLBuffer> wb = get_weight_buf(w.data, wbytes, w_off);
            if (!wb) {
                wb = [g_dev newBufferWithBytes:w.data length:wbytes options:MTLResourceStorageModeShared];
                w_off = 0;
            }
            if (!wb) return false;
            [enc setComputePipelineState:pipe];
            [enc setBuffer:wb offset:w_off atIndex:0];
            [enc setBuffer:xb offset:0 atIndex:1];
            [enc setBuffer:yb offset:0 atIndex:2];
            [enc setBytes:&K length:4 atIndex:3];
            [enc setBytes:&N length:4 atIndex:4];
            [enc setBytes:&rb length:8 atIndex:5];
            [enc setBytes:&M length:4 atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N * (NSUInteger)M, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        [enc endEncoding];
        const auto wait_start = std::chrono::steady_clock::now();
        [cmd commit]; [cmd waitUntilCompleted];
        record_completed_command(cmd, wait_start, std::chrono::steady_clock::now());
        memcpy(y, [yb contents], ybytes);
    }
    return true;
}

bool metal_gemv(const float* x, const Tensor& w, float* y, int K, int N) {
    return metal_gemm(x, w, y, 1, K, N);
}

// One CB, `reps` GEMVs, one wait. Used to time the kernel without per-call
// submit tax. Writes only the last y.
bool metal_gemv_repeat(const float* x, const Tensor& w, float* y,
                       int K, int N, int reps) {
    init();
    if (!g_dev || !g_lib || N <= 0 || K <= 0 || reps <= 0) return false;
    if (w.type != GGMLType::IQ1_S && w.type != GGMLType::IQ2_XXS &&
        w.type != GGMLType::Q2_K && w.type != GGMLType::Q4_K &&
        w.type != GGMLType::Q6_K && w.type != GGMLType::Q8_0) return false;
    @autoreleasepool {
        std::lock_guard<std::mutex> lk(g_xy_mtx);
        size_t xbytes = (size_t)K * 4, ybytes = (size_t)N * 4;
        memcpy([g_xbuf contents], x, xbytes);
        id<MTLBuffer> yb = [g_dev newBufferWithLength:ybytes
                                              options:MTLResourceStorageModeShared];
        if (!yb) return false;
        id<MTLCommandBuffer> cmd = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        for (int i = 0; i < reps; i++) {
            bool ok = false;
            if (w.type == GGMLType::IQ1_S)
                ok = bind_iq1_s(enc, w, K, N, g_xbuf, 0, yb, 0);
            else if (w.type == GGMLType::IQ2_XXS)
                ok = bind_iq2_xxs(enc, w, K, N, g_xbuf, 0, yb, 0);
            else if (w.type == GGMLType::Q4_K)
                ok = bind_q4k(enc, w, K, N, g_xbuf, 0, yb, 0);
            else if (w.type == GGMLType::Q6_K)
                ok = bind_q6k(enc, w, K, N, g_xbuf, 0, yb, 0);
            else if (w.type == GGMLType::Q8_0)
                ok = bind_q8(enc, w, K, N, g_xbuf, 0, yb, 0);
            else
                ok = bind_fused(enc, w, K, N, g_xbuf, 0, yb, 0);
            if (!ok) {
                [enc endEncoding];
                return false;
            }
        }
        [enc endEncoding];
        const auto wait_start = std::chrono::steady_clock::now();
        [cmd commit];
        [cmd waitUntilCompleted];
        record_completed_command(cmd, wait_start, std::chrono::steady_clock::now());
        memcpy(y, [yb contents], ybytes);
    }
    return true;
}

#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
static bool metal_test_q2k_candidate_ab(const float* x, const Tensor& w,
                                        float* baseline, float* candidate,
                                        int K, int N, int reps, bool streamed,
                                        double* baseline_gpu_ms, double* candidate_gpu_ms) {
    init();
    if (!g_dev || !g_lib || !x || !baseline || !candidate ||
        !baseline_gpu_ms || !candidate_gpu_ms || w.type != GGMLType::Q2_K ||
        K <= 0 || N <= 0 || reps <= 0 || (K % 256) != 0) return false;
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(g_xy_mtx);
        id<MTLBuffer> input = [g_dev newBufferWithBytes:x length:(size_t)K * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        if (!input) return false;
        const auto run = [&](bool candidate_arm, float* output, double* gpu_ms) {
            id<MTLBuffer> result = [g_dev newBufferWithLength:(size_t)N * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
            if (!result) return false;
            id<MTLCommandBuffer> command = [g_q commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!command || !encoder) return false;
            for (int repetition = 0; repetition != reps; ++repetition) {
                const bool bound = !candidate_arm
                    ? bind_q2k(encoder, w, K, N, input, 0, result, 0)
                    : streamed
                        ? bind_q2k_streamed(encoder, w, K, N, input, 0, result, 0)
                        : bind_q2k_two_row(encoder, w, K, N, input, 0, result, 0);
                if (!bound) {
                    [encoder endEncoding];
                    return false;
                }
            }
            [encoder endEncoding];
            const auto wait_start = std::chrono::steady_clock::now();
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) return false;
            record_completed_command(command, wait_start, std::chrono::steady_clock::now());
            const CFTimeInterval start = command.GPUStartTime;
            const CFTimeInterval end = command.GPUEndTime;
            if (!(end >= start)) return false;
            *gpu_ms = 1000.0 * (end - start);
            memcpy(output, result.contents, (size_t)N * sizeof(float));
            return true;
        };
        return run(false, baseline, baseline_gpu_ms) &&
               run(true, candidate, candidate_gpu_ms);
    }
}

bool metal_test_q2k_two_row_ab(const float* x, const Tensor& w,
                               float* baseline, float* candidate,
                               int K, int N, int reps,
                               double* baseline_gpu_ms, double* candidate_gpu_ms) {
    return metal_test_q2k_candidate_ab(x, w, baseline, candidate, K, N, reps, false,
                                       baseline_gpu_ms, candidate_gpu_ms);
}

bool metal_test_q2k_streamed_ab(const float* x, const Tensor& w,
                                float* baseline, float* candidate,
                                int K, int N, int reps,
                                double* baseline_gpu_ms, double* candidate_gpu_ms) {
    return metal_test_q2k_candidate_ab(x, w, baseline, candidate, K, N, reps, true,
                                       baseline_gpu_ms, candidate_gpu_ms);
}

void metal_test_select_q2k_two_row_pipeline(bool enabled) {
    g_test_q2k_two_row_pipeline = enabled;
}

bool metal_test_q2k_pipeline_widths(uint32_t* baseline, uint32_t* candidate) {
    init();
    if (!baseline || !candidate) return false;
    id<MTLComputePipelineState> baseline_pipeline = get_q2k_pipe();
    id<MTLComputePipelineState> candidate_pipeline = get_q2k_two_row_pipe();
    if (!baseline_pipeline || !candidate_pipeline) return false;
    *baseline = static_cast<uint32_t>(baseline_pipeline.threadExecutionWidth);
    *candidate = static_cast<uint32_t>(candidate_pipeline.threadExecutionWidth);
    return true;
}

bool metal_test_q2k_streamed_pipeline_widths(uint32_t* baseline, uint32_t* candidate) {
    init();
    if (!baseline || !candidate) return false;
    id<MTLComputePipelineState> baseline_pipeline = get_q2k_pipe();
    id<MTLComputePipelineState> candidate_pipeline = get_q2k_streamed_pipe();
    if (!baseline_pipeline || !candidate_pipeline) return false;
    *baseline = static_cast<uint32_t>(baseline_pipeline.threadExecutionWidth);
    *candidate = static_cast<uint32_t>(candidate_pipeline.threadExecutionWidth);
    return true;
}

bool metal_test_q2k_pipeline_limits(uint32_t* baseline_width, uint32_t* baseline_max,
                                    uint32_t* two_row_width, uint32_t* two_row_max,
                                    uint32_t* streamed_width, uint32_t* streamed_max) {
    init();
    if (!baseline_width || !baseline_max || !two_row_width || !two_row_max ||
        !streamed_width || !streamed_max) return false;
    id<MTLComputePipelineState> baseline = get_q2k_pipe();
    id<MTLComputePipelineState> two_row = get_q2k_two_row_pipe();
    id<MTLComputePipelineState> streamed = get_q2k_streamed_pipe();
    if (!baseline || !two_row || !streamed) return false;
    *baseline_width = static_cast<uint32_t>(baseline.threadExecutionWidth);
    *baseline_max = static_cast<uint32_t>(baseline.maxTotalThreadsPerThreadgroup);
    *two_row_width = static_cast<uint32_t>(two_row.threadExecutionWidth);
    *two_row_max = static_cast<uint32_t>(two_row.maxTotalThreadsPerThreadgroup);
    *streamed_width = static_cast<uint32_t>(streamed.threadExecutionWidth);
    *streamed_max = static_cast<uint32_t>(streamed.maxTotalThreadsPerThreadgroup);
    return true;
}

bool metal_test_iq2_xxs_pipeline_limits(uint32_t* width, uint32_t* maximum_threads) {
    init();
    if (!width || !maximum_threads) return false;
    id<MTLComputePipelineState> pipeline = get_iq2_xxs_pipe();
    if (!pipeline) return false;
    *width = static_cast<uint32_t>(pipeline.threadExecutionWidth);
    *maximum_threads = static_cast<uint32_t>(pipeline.maxTotalThreadsPerThreadgroup);
    return true;
}

bool metal_test_iq1_s_pipeline_limits(uint32_t* width, uint32_t* maximum_threads) {
    init();
    if (!width || !maximum_threads) return false;
    id<MTLComputePipelineState> pipeline = get_iq1_s_pipe();
    if (!pipeline) return false;
    *width = static_cast<uint32_t>(pipeline.threadExecutionWidth);
    *maximum_threads = static_cast<uint32_t>(pipeline.maxTotalThreadsPerThreadgroup);
    return true;
}

#endif

static id<MTLComputePipelineState> get_affine_u2_256_pipe() {
    if (g_affine_u2_256_pipe) return g_affine_u2_256_pipe;
    if (!g_lib) return nil;
    NSError* error = nil;
    id<MTLFunction> function = [g_lib newFunctionWithName:@"gemv_affine_u2_256"];
    if (!function) {
        std::fprintf(stderr, "[metal] affine UInt2/256 function unavailable\n");
        return nil;
    }
    g_affine_u2_256_pipe =
        [g_dev newComputePipelineStateWithFunction:function error:&error];
    [function release];
    if (!g_affine_u2_256_pipe) {
        std::fprintf(stderr, "[metal] affine UInt2/256 pipeline: %s\n",
                     error ? error.localizedDescription.UTF8String : "unknown");
        return nil;
    }
    return g_affine_u2_256_pipe;
}

#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)

bool metal_test_affine_u2_block256(const float* input,
                                   const uint8_t* packed_weights,
                                   const uint16_t* scales,
                                   const uint16_t* biases,
                                   float* output, int K, int N,
                                   double* gpu_ms, uint32_t samples,
                                   uint32_t dispatches_per_command,
                                   bool distinct_weights,
                                   uint64_t* requested_bytes,
                                   uint32_t* width, uint32_t* maximum_threads) {
    init();
    if (!g_dev || !g_q || !g_lib || !input || !packed_weights || !scales ||
        !biases || !output || !gpu_ms || samples == 0 || dispatches_per_command == 0 ||
        !requested_bytes || !width ||
        !maximum_threads || K <= 0 || N <= 0 || K % 512 != 0 || N % 8 != 0)
        return false;
    id<MTLComputePipelineState> pipeline = get_affine_u2_256_pipe();
    if (!pipeline || pipeline.threadExecutionWidth != 32 ||
        pipeline.maxTotalThreadsPerThreadgroup < 64)
        return false;
    const size_t packed_length = static_cast<size_t>(N) * K / 4;
    const size_t plane_values = static_cast<size_t>(N) * (K / 256);
    const size_t plane_length = plane_values * sizeof(uint16_t);
    @autoreleasepool {
        id<MTLBuffer> input_buffer = [g_dev newBufferWithBytes:input
                                                        length:static_cast<size_t>(K) * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        const uint32_t copies = distinct_weights ? dispatches_per_command : 1;
        if (packed_length > std::numeric_limits<size_t>::max() / copies ||
            plane_length > std::numeric_limits<size_t>::max() / copies)
            return false;
        id<MTLBuffer> packed_buffer = [g_dev newBufferWithLength:packed_length * copies
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> scale_buffer = [g_dev newBufferWithLength:plane_length * copies
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> bias_buffer = [g_dev newBufferWithLength:plane_length * copies
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer = [g_dev newBufferWithLength:static_cast<size_t>(N) * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        auto release_buffers = [&] {
            [input_buffer release];
            [packed_buffer release];
            [scale_buffer release];
            [bias_buffer release];
            [output_buffer release];
        };
        if (!input_buffer || !packed_buffer || !scale_buffer || !bias_buffer ||
            !output_buffer) {
            release_buffers();
            return false;
        }
        for (uint32_t dispatch = 0; dispatch != copies; ++dispatch) {
            std::memcpy(static_cast<uint8_t*>(packed_buffer.contents) + dispatch * packed_length,
                        packed_weights, packed_length);
            std::memcpy(static_cast<uint8_t*>(scale_buffer.contents) + dispatch * plane_length,
                        scales, plane_length);
            std::memcpy(static_cast<uint8_t*>(bias_buffer.contents) + dispatch * plane_length,
                        biases, plane_length);
        }
        for (uint32_t sample = 0; sample != samples; ++sample) {
            id<MTLCommandBuffer> command = [g_q commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!command || !encoder) {
                release_buffers();
                return false;
            }
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input_buffer offset:0 atIndex:3];
            [encoder setBuffer:output_buffer offset:0 atIndex:4];
            [encoder setBytes:&K length:sizeof(K) atIndex:5];
            [encoder setBytes:&N length:sizeof(N) atIndex:6];
            for (uint32_t dispatch = 0; dispatch != dispatches_per_command; ++dispatch) {
                const uint32_t copy = distinct_weights ? dispatch : 0;
                [encoder setBuffer:packed_buffer offset:copy * packed_length atIndex:0];
                [encoder setBuffer:scale_buffer offset:copy * plane_length atIndex:1];
                [encoder setBuffer:bias_buffer offset:copy * plane_length atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(N / 8), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }
            [encoder endEncoding];
            const auto wait_start = std::chrono::steady_clock::now();
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) {
                std::fprintf(stderr, "[metal] affine UInt2/256 command: %s\n",
                             command_buffer_failure_detail(command).c_str());
                release_buffers();
                return false;
            }
            record_completed_command(command, wait_start, std::chrono::steady_clock::now());
            const CFTimeInterval start = command.GPUStartTime;
            const CFTimeInterval end = command.GPUEndTime;
            if (!(end >= start)) {
                release_buffers();
                return false;
            }
            gpu_ms[sample] = 1000.0 * (end - start) / dispatches_per_command;
        }
        *requested_bytes = packed_length + 2 * plane_length;
        *width = static_cast<uint32_t>(pipeline.threadExecutionWidth);
        *maximum_threads =
            static_cast<uint32_t>(pipeline.maxTotalThreadsPerThreadgroup);
        std::memcpy(output, output_buffer.contents, static_cast<size_t>(N) * sizeof(float));
        release_buffers();
        return true;
    }
}

bool metal_test_mpp_int2_m1(const float* input, const uint8_t* packed_weights,
                            const uint8_t* e8m0_scales, float* output,
                            int K, int N, double* gpu_ms,
                            uint64_t* data_bytes, uint64_t* scale_bytes,
                            uint32_t* width, uint32_t* maximum_threads) {
    init();
    if (!g_dev || !g_q || !g_mpp_int2_lib || !input || !packed_weights ||
        !e8m0_scales || !output || !gpu_ms || !data_bytes || !scale_bytes ||
        !width || !maximum_threads || K != 5120 || N != 17408)
        return false;
    if (!g_mpp_int2_m1_pipe) {
        NSError* error = nil;
        id<MTLFunction> function = [g_mpp_int2_lib newFunctionWithName:@"gemv_mpp_int2_m1"];
        if (!function) return false;
        g_mpp_int2_m1_pipe = [g_dev newComputePipelineStateWithFunction:function
                                                                  error:&error];
        if (!g_mpp_int2_m1_pipe) {
            std::fprintf(stderr, "[metal] MPP Int2 pipeline: %s\n",
                         error ? error.localizedDescription.UTF8String : "unknown");
            return false;
        }
    }
    const size_t packed_length = static_cast<size_t>(N) * K / 4;
    const size_t scale_length = static_cast<size_t>(N) * (K / 32);
    constexpr int padded_rows = 8;
    std::vector<uint16_t> half_input(static_cast<size_t>(K) * padded_rows);
    for (int index = 0; index != K; ++index)
        half_input[static_cast<size_t>(index)] = fp32_to_fp16(input[index]);
    @autoreleasepool {
        id<MTLBuffer> input_buffer = [g_dev newBufferWithBytes:half_input.data()
                                                          length:half_input.size() * sizeof(uint16_t)
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> weight_buffer = [g_dev newBufferWithBytes:packed_weights
                                                           length:packed_length
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> scale_buffer = [g_dev newBufferWithBytes:e8m0_scales
                                                          length:scale_length
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer = [g_dev newBufferWithLength:static_cast<size_t>(N) * padded_rows * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
        if (!input_buffer || !weight_buffer || !scale_buffer || !output_buffer)
            return false;
        id<MTLCommandBuffer> command = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return false;
        [encoder setComputePipelineState:g_mpp_int2_m1_pipe];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:scale_buffer offset:0 atIndex:2];
        [encoder setBuffer:output_buffer offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake((static_cast<size_t>(N) + 15) / 16, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [encoder endEncoding];
        const auto wait_start = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "[metal] MPP Int2 command: %s\n",
                         command_buffer_failure_detail(command).c_str());
            return false;
        }
        record_completed_command(command, wait_start, std::chrono::steady_clock::now());
        const CFTimeInterval start = command.GPUStartTime;
        const CFTimeInterval end = command.GPUEndTime;
        if (!(end >= start)) return false;
        *gpu_ms = 1000.0 * (end - start);
        *data_bytes = packed_length;
        *scale_bytes = scale_length;
        *width = static_cast<uint32_t>(g_mpp_int2_m1_pipe.threadExecutionWidth);
        *maximum_threads = static_cast<uint32_t>(g_mpp_int2_m1_pipe.maxTotalThreadsPerThreadgroup);
        std::memcpy(output, output_buffer.contents, static_cast<size_t>(N) * sizeof(float));
        return true;
    }
}
#endif

static size_t align256(size_t n) { return (n + 255) & ~size_t(255); }

static bool grow_shared(id<MTLBuffer>& buf, size_t need) {
    if (buf && buf.length >= need) return true;
    id<MTLBuffer> next = [g_dev newBufferWithLength:need
                                           options:MTLResourceStorageModeShared];
    if (!next) return false;
    buf = next;
    return true;
}

static id<MTLComputePipelineState> g_glu_pipe;
static id<MTLComputePipelineState> g_glu_experts_pipe;
static id<MTLComputePipelineState> get_glu_pipe() {
    if (g_glu_pipe) return g_glu_pipe;
    if (!g_lib) return nil;
    NSError* err = nil;
    id<MTLFunction> f = [g_lib newFunctionWithName:@"act_glu"];
    if (!f) return nil;
    g_glu_pipe = [g_dev newComputePipelineStateWithFunction:f error:&err];
    return g_glu_pipe;
}

static id<MTLComputePipelineState> get_glu_experts_pipe() {
    if (g_glu_experts_pipe) return g_glu_experts_pipe;
    if (!g_lib) return nil;
    NSError* err = nil;
    id<MTLFunction> f = [g_lib newFunctionWithName:@"act_glu_experts"];
    if (!f) return nil;
    g_glu_experts_pipe = [g_dev newComputePipelineStateWithFunction:f error:&err];
    return g_glu_experts_pipe;
}

static bool enqueue_glu(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> buf,
                        size_t gate_off, size_t up_off, size_t hid_off,
                        int n, int swiglu) {
    id<MTLComputePipelineState> pipe = get_glu_pipe();
    if (!pipe || !enc || !buf || n <= 0) return false;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:buf offset:gate_off atIndex:0];
    [enc setBuffer:buf offset:up_off atIndex:1];
    [enc setBuffer:buf offset:hid_off atIndex:2];
    [enc setBytes:&n length:4 atIndex:3];
    [enc setBytes:&swiglu length:4 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((n + 63) / 64, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    return true;
}

static bool enqueue_moe_glu(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> buf,
                            size_t gate_up_off, size_t out_off,
                            int selected, int intermediate, int swiglu) {
    id<MTLComputePipelineState> pipe = get_glu_experts_pipe();
    if (!pipe || !enc || !buf || selected <= 0 || intermediate <= 0 ||
        pipe.threadExecutionWidth == 0 ||
        pipe.maxTotalThreadsPerThreadgroup < pipe.threadExecutionWidth)
        return false;
    const NSUInteger threads = std::min<NSUInteger>(
        64, pipe.maxTotalThreadsPerThreadgroup -
                (pipe.maxTotalThreadsPerThreadgroup % pipe.threadExecutionWidth));
    if (threads == 0) return false;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:buf offset:gate_up_off atIndex:0];
    [enc setBuffer:buf offset:out_off atIndex:1];
    [enc setBytes:&intermediate length:4 atIndex:2];
    [enc setBytes:&selected length:4 atIndex:3];
    [enc setBytes:&swiglu length:4 atIndex:4];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)intermediate, (NSUInteger)selected, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    return true;
}

bool metal_act_glu(const float* gate, const float* up, float* out, int n, bool swiglu) {
    init();
    if (!g_dev || !gate || !up || !out || n <= 0 || !get_glu_pipe()) return false;
    std::lock_guard<std::mutex> lk(g_xy_mtx);
    size_t nb = (size_t)n * 4;
    size_t g_off = 0, u_off = align256(nb), o_off = u_off + align256(nb);
    if (!grow_shared(g_ybuf, o_off + nb)) return false;
    uint8_t* base = (uint8_t*)[g_ybuf contents];
    std::memcpy(base + g_off, gate, nb);
    std::memcpy(base + u_off, up, nb);
    id<MTLCommandBuffer> cmd = [g_q commandBuffer];
    if (!cmd) return false;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    int sw = swiglu ? 1 : 0;
    if (!enqueue_glu(enc, g_ybuf, g_off, u_off, o_off, n, sw)) {
        [enc endEncoding];
        return false;
    }
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status == MTLCommandBufferStatusError) return false;
    std::memcpy(out, (const uint8_t*)[g_ybuf contents] + o_off, nb);
    return true;
}

static id<MTLCommandBuffer> g_bcmd;
static float* g_bdest[8];
static size_t g_boff[8];
static int g_bn = 0;
static int g_bN[8];

bool metal_gemv_begin(const MatmulBatchSpec* specs, int n) {
    init();
    if (!g_dev || !g_lib || n <= 0 || n > 8) return false;
    for (int i = 0; i < n; i++) {
        if (!specs[i].x || !specs[i].w || !specs[i].y) return false;
        if (specs[i].K < 32 || specs[i].N < 512 || specs[i].N >= 65536)
            return false;
    }
    std::lock_guard<std::mutex> lk(g_xy_mtx);
    size_t xcur = 0, ycur = 0;
    size_t xoff[8], yoff[8];
    for (int i = 0; i < n; i++) {
        xoff[i] = xcur;
        xcur += align256((size_t)specs[i].K * 4);
        yoff[i] = ycur;
        ycur += align256((size_t)specs[i].N * 4);
    }
    if (!grow_shared(g_xbuf, xcur) || !grow_shared(g_ybuf, ycur))
        return false;
    uint8_t* xb = (uint8_t*)[g_xbuf contents];
    for (int i = 0; i < n; i++)
        std::memcpy(xb + xoff[i], specs[i].x, (size_t)specs[i].K * 4);
    id<MTLCommandBuffer> cmd = [g_q commandBuffer];
    if (!cmd) return false;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    for (int i = 0; i < n; i++) {
        if (!bind_any(enc, *specs[i].w, specs[i].K, specs[i].N,
                      g_xbuf, xoff[i], g_ybuf, yoff[i])) {
            [enc endEncoding];
            return false;
        }
    }
    [enc endEncoding];
    [cmd commit];
    g_bcmd = cmd;
    g_bn = n;
    for (int i = 0; i < n; i++) {
        g_bdest[i] = specs[i].y;
        g_boff[i] = yoff[i];
        g_bN[i] = specs[i].N;
    }
    return true;
}

void metal_gemv_end() {
    if (!g_bcmd) return;
    [g_bcmd waitUntilCompleted];
    const uint8_t* yb = (const uint8_t*)[g_ybuf contents];
    for (int i = 0; i < g_bn; i++)
        std::memcpy(g_bdest[i], yb + g_boff[i], (size_t)g_bN[i] * 4);
    g_bcmd = nil;
    g_bn = 0;
}

bool metal_decode_ffn_moe(
    const float* x_norm, const Tensor& ffn_gate, const Tensor& ffn_up,
    const Tensor& ffn_down, float* xb, int H, int inter, bool swiglu,
    const float* moe_in, const Tensor* moe_up_stack, const Tensor* moe_dn_stack,
    const int* expert_ids, int n_exp, int exp_inter, const float* route_w,
    float* moe_out) {
    init();
    if (!g_dev || !x_norm || !xb || H < 512 || inter < 512) return false;
    if (n_exp < 0 || n_exp > 16) return false;
    const int gu_n = exp_inter * 2;
    if (n_exp > 0 && gu_n < 512) return false;
    size_t x_norm_off = 0;
    size_t moe_in_off = align256((size_t)H * 4);
    size_t xtotal = moe_in_off + (n_exp > 0 ? align256((size_t)H * 4) : 0);
    size_t y_gate = 0;
    size_t y_up = align256((size_t)inter * 4);
    size_t y_hid = y_up + align256((size_t)inter * 4);
    size_t y_down = y_hid + align256((size_t)inter * 4);
    size_t y_cur = y_down + align256((size_t)H * 4);
    size_t moe_gu[16], moe_h[16], moe_d[16];
    if (n_exp > 0) {
        if (!moe_in || !moe_up_stack || !moe_dn_stack || !expert_ids ||
            !moe_out || !route_w)
            return false;
        if (!moe_up_stack->data || !moe_dn_stack->data) return false;
    }
    for (int k = 0; k < n_exp; k++) {
        moe_gu[k] = y_cur + (size_t)k * (size_t)gu_n * 4;
        moe_h[k] = y_cur + (size_t)n_exp * (size_t)gu_n * 4
                   + (size_t)k * (size_t)exp_inter * 4;
        moe_d[k] = y_cur + (size_t)n_exp * (size_t)gu_n * 4
                   + (size_t)n_exp * (size_t)exp_inter * 4
                   + (size_t)k * (size_t)H * 4;
    }
    if (n_exp > 0)
        y_cur += (size_t)n_exp * ((size_t)gu_n + (size_t)exp_inter + (size_t)H) * 4;

    std::lock_guard<std::mutex> lk(g_xy_mtx);
    if (!grow_shared(g_xbuf, xtotal) || !grow_shared(g_ybuf, y_cur))
        return false;
    uint8_t* xbase = (uint8_t*)[g_xbuf contents];
    std::memcpy(xbase + x_norm_off, x_norm, (size_t)H * 4);
    if (n_exp > 0)
        std::memcpy(xbase + moe_in_off, moe_in, (size_t)H * 4);

    if (!get_glu_pipe()) return false;
    // One CB. Ending a compute encoder is the GPU barrier; do not
    // waitUntilCompleted between GEMV and GeGLU.
    id<MTLCommandBuffer> cmd = [g_q commandBuffer];
    if (!cmd) return false;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    if (!bind_any(enc, ffn_gate, H, inter, g_xbuf, x_norm_off, g_ybuf, y_gate) ||
        !bind_any(enc, ffn_up, H, inter, g_xbuf, x_norm_off, g_ybuf, y_up)) {
        [enc endEncoding]; return false;
    }
    if (n_exp > 0) {
        bool id_ok = false;
        if (moe_up_stack->type == GGMLType::Q4_K)
            id_ok = bind_q4k_id(enc, *moe_up_stack, H, gu_n, expert_ids, n_exp,
                                g_xbuf, moe_in_off, g_ybuf, moe_gu[0]);
        if (!id_ok) {
            Tensor upv;
            size_t up_per = (size_t)moe_up_stack->nbytes() / moe_up_stack->dims[2];
            for (int k = 0; k < n_exp; k++) {
                upv = *moe_up_stack;
                upv.data = moe_up_stack->data + (size_t)expert_ids[k] * up_per;
                if (!bind_any(enc, upv, H, gu_n, g_xbuf, moe_in_off,
                              g_ybuf, moe_gu[k])) {
                    [enc endEncoding]; return false;
                }
            }
        }
    }
    [enc endEncoding];
    enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    int sw = swiglu ? 1 : 0;
    if (!enqueue_glu(enc, g_ybuf, y_gate, y_up, y_hid, inter, sw)) {
        [enc endEncoding]; return false;
    }
    for (int k = 0; k < n_exp; k++) {
        if (!enqueue_glu(enc, g_ybuf, moe_gu[k],
                         moe_gu[k] + (size_t)exp_inter * 4,
                         moe_h[k], exp_inter, sw)) {
            [enc endEncoding]; return false;
        }
    }
    [enc endEncoding];
    enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    if (!bind_any(enc, ffn_down, inter, H, g_ybuf, y_hid, g_ybuf, y_down)) {
        [enc endEncoding]; return false;
    }
    if (n_exp > 0) {
        Tensor dnv;
        size_t dn_per = (size_t)moe_dn_stack->nbytes() / moe_dn_stack->dims[2];
        for (int k = 0; k < n_exp; k++) {
            dnv = *moe_dn_stack;
            dnv.data = moe_dn_stack->data + (size_t)expert_ids[k] * dn_per;
            if (!bind_any(enc, dnv, exp_inter, H, g_ybuf, moe_h[k],
                          g_ybuf, moe_d[k])) {
                [enc endEncoding]; return false;
            }
        }
    }
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status == MTLCommandBufferStatusError) return false;

    const uint8_t* ybase = (const uint8_t*)[g_ybuf contents];
    std::memcpy(xb, ybase + y_down, (size_t)H * 4);
    if (n_exp > 0) {
        std::memset(moe_out, 0, (size_t)H * sizeof(float));
        for (int k = 0; k < n_exp; k++) {
            const float* src = (const float*)(ybase + moe_d[k]);
            float rw = route_w[k];
            for (int j = 0; j < H; j++)
                moe_out[j] += rw * src[j];
        }
    }
    return true;
}

static id<MTLComputePipelineState> g_q6k_pipe;
static id<MTLComputePipelineState> get_q6k_pipe() {
    if (g_q6k_pipe) return g_q6k_pipe;
    if (!g_lib) return nil;
    NSError* err = nil;
    id<MTLFunction> f = [g_lib newFunctionWithName:@"gemv_q6k"];
    if (!f) {
        fprintf(stderr, "[metal] gemv_q6k: missing kernel\n");
        return nil;
    }
    g_q6k_pipe = [g_dev newComputePipelineStateWithFunction:f error:&err];
    if (!g_q6k_pipe)
        fprintf(stderr, "[metal] gemv_q6k pipe: %s\n",
                err ? [[err localizedDescription] UTF8String] : "?");
    return g_q6k_pipe;
}

static bool bind_q6k(id<MTLComputeCommandEncoder> enc, const Tensor& w,
                     int K, int N, id<MTLBuffer> xbuf, size_t xoff,
                     id<MTLBuffer> ybuf, size_t yoff) {
    if (w.type != GGMLType::Q6_K || w.n_dims != 2 || (K % 256) != 0 || N < 1 ||
        w.dims[0] != static_cast<uint64_t>(K) || w.dims[1] != static_cast<uint64_t>(N)) return false;
    const uint64_t row_bytes = (static_cast<uint64_t>(K) / 256u) * bytes_per_block(w.type);
    if (static_cast<uint64_t>(N) > UINT64_MAX / row_bytes ||
        !strict_tensor_data_span(w, static_cast<uint64_t>(N) * row_bytes)) return false;
    id<MTLComputePipelineState> pipe = get_q6k_pipe();
    if (!pipe || !xbuf || !ybuf) return false;
    uint64_t rb = row_bytes;
    size_t w_off = 0;
    id<MTLBuffer> wb = get_weight_buf(w.data, static_cast<size_t>(N) * static_cast<size_t>(rb), w_off);
    if (!wb) return false;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:wb offset:w_off atIndex:0];
    [enc setBuffer:xbuf offset:xoff atIndex:1];
    [enc setBuffer:ybuf offset:yoff atIndex:2];
    [enc setBytes:&K length:4 atIndex:3];
    [enc setBytes:&N length:4 atIndex:4];
    [enc setBytes:&rb length:8 atIndex:5];
    int M = 1;
    [enc setBytes:&M length:4 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)N + 3) / 4, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    return true;
}

static std::unordered_map<std::string, id<MTLComputePipelineState>> g_named;
static id<MTLComputePipelineState> named_pipe(const char* name) {
    auto it = g_named.find(name);
    if (it != g_named.end()) return it->second;
    if (!g_lib) return nil;
    NSError* err = nil;
    id<MTLFunction> f = [g_lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!f) return nil;
    id<MTLComputePipelineState> p = [g_dev newComputePipelineStateWithFunction:f error:&err];
    if (p) g_named[name] = p;
    return p;
}

#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
static id<MTLComputePipelineState> get_router_topk_pipe() {
    if (g_router_topk_pipe) return g_router_topk_pipe;
    if (!g_dev) return nil;
    NSError* error = nil;
    if (!g_router_lib) {
        g_router_lib = [g_dev newLibraryWithSource:
            [NSString stringWithUTF8String:src_router_topk] options:nil error:&error];
        if (!g_router_lib) {
            std::fprintf(stderr, "[metal] router top-k library: %s\n",
                         error ? error.localizedDescription.UTF8String : "unknown");
            return nil;
        }
    }
    id<MTLFunction> function = [g_router_lib newFunctionWithName:@"router_topk_f32"];
    if (!function) return nil;
    g_router_topk_pipe = [g_dev newComputePipelineStateWithFunction:function error:&error];
    [function release];
    if (!g_router_topk_pipe)
        std::fprintf(stderr, "[metal] router top-k pipeline: %s\n",
                     error ? error.localizedDescription.UTF8String : "unknown");
    return g_router_topk_pipe;
}
#endif

static void enc_1d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                   int n, int tpt = 64) {
    [enc setComputePipelineState:p];
    [enc dispatchThreadgroups:MTLSizeMake((n + tpt - 1) / tpt, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
}

#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
bool metal_test_router_top_k(const MetalRouterTopKSpec& spec, const float* logits,
                             uint32_t* ids, float* weights,
                             MetalRouterPipelineCaps* capabilities) {
    init();
    if (!g_dev || !g_q || !logits || !ids || !weights || !capabilities ||
        spec.expert_count == 0 || spec.expert_count > 512 ||
        spec.selected_count == 0 || spec.selected_count > 16 ||
        spec.selected_count > spec.expert_count ||
        spec.score_domain != MetalRouterScoreDomain::Logits ||
        spec.normalization != MetalRouterNormalization::SelectThenNormalizeSoftmax ||
        spec.tie_policy != MetalRouterTiePolicy::LowestExpertId)
        return false;
    for (uint32_t expert = 0; expert != spec.expert_count; ++expert)
        if (!std::isfinite(logits[expert])) return false;

    id<MTLComputePipelineState> pipeline = get_router_topk_pipe();
    if (!pipeline || pipeline.threadExecutionWidth == 0 ||
        pipeline.maxTotalThreadsPerThreadgroup < 256)
        return false;
    capabilities->thread_execution_width = static_cast<uint32_t>(pipeline.threadExecutionWidth);
    capabilities->max_total_threads_per_threadgroup =
        static_cast<uint32_t>(pipeline.maxTotalThreadsPerThreadgroup);

    @autoreleasepool {
        id<MTLBuffer> logits_buffer =
            [g_dev newBufferWithBytes:logits
                                length:static_cast<size_t>(spec.expert_count) * sizeof(float)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> ids_buffer =
            [g_dev newBufferWithLength:static_cast<size_t>(spec.selected_count) * sizeof(int32_t)
                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> weights_buffer =
            [g_dev newBufferWithLength:static_cast<size_t>(spec.selected_count) * sizeof(float)
                                options:MTLResourceStorageModeShared];
        if (!logits_buffer || !ids_buffer || !weights_buffer) {
            [logits_buffer release];
            [ids_buffer release];
            [weights_buffer release];
            return false;
        }
        id<MTLCommandBuffer> command = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> encoder = command ? [command computeCommandEncoder] : nil;
        if (!encoder) {
            [logits_buffer release];
            [ids_buffer release];
            [weights_buffer release];
            return false;
        }
        const int expert_count = static_cast<int>(spec.expert_count);
        const int selected_count = static_cast<int>(spec.selected_count);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:logits_buffer offset:0 atIndex:0];
        [encoder setBuffer:ids_buffer offset:0 atIndex:1];
        [encoder setBuffer:weights_buffer offset:0 atIndex:2];
        [encoder setBytes:&expert_count length:sizeof(expert_count) atIndex:3];
        [encoder setBytes:&selected_count length:sizeof(selected_count) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        const auto wait_start = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "[metal] router top-k command: %s\n",
                         command_buffer_failure_detail(command).c_str());
            [logits_buffer release];
            [ids_buffer release];
            [weights_buffer release];
            return false;
        }
        record_completed_command(command, wait_start, std::chrono::steady_clock::now());
        std::memcpy(ids, [ids_buffer contents],
                    static_cast<size_t>(spec.selected_count) * sizeof(uint32_t));
        std::memcpy(weights, [weights_buffer contents],
                    static_cast<size_t>(spec.selected_count) * sizeof(float));
        [logits_buffer release];
        [ids_buffer release];
        [weights_buffer release];
        return true;
    }
}

bool metal_test_moe_batched_activation(
    const float* gate_up, size_t gate_up_values,
    const uint32_t* expert_ids, uint32_t selected_count,
    uint32_t intermediate, bool swiglu, MetalMoeActivationTestPath path,
    float* output, size_t output_values,
    MetalMoeActivationTestMetrics* metrics) {
    if (metrics) *metrics = {};
    init();
    if (!g_dev || !g_q || !gate_up || !expert_ids || !output || !metrics ||
        selected_count == 0 || selected_count > 16 || intermediate == 0 ||
        intermediate > static_cast<uint32_t>(INT_MAX) ||
        (path != MetalMoeActivationTestPath::PerExpertLoop &&
         path != MetalMoeActivationTestPath::Batched2D))
        return false;
    if (selected_count > SIZE_MAX / intermediate) return false;
    const size_t selected_values = static_cast<size_t>(selected_count) * intermediate;
    if (selected_values > SIZE_MAX / 2u || selected_values > SIZE_MAX / sizeof(float))
        return false;
    const size_t expected_gate_up_values = selected_values * 2u;
    if (expected_gate_up_values > SIZE_MAX / sizeof(float) ||
        gate_up_values != expected_gate_up_values || output_values != selected_values)
        return false;
    const size_t gate_up_bytes = expected_gate_up_values * sizeof(float);
    const size_t output_bytes = output_values * sizeof(float);
    const size_t output_offset = align256(gate_up_bytes);
    if (output_offset > SIZE_MAX - output_bytes) return false;

    id<MTLComputePipelineState> pipeline =
        path == MetalMoeActivationTestPath::Batched2D
            ? get_glu_experts_pipe() : get_glu_pipe();
    if (!pipeline || pipeline.threadExecutionWidth == 0 ||
        pipeline.maxTotalThreadsPerThreadgroup < pipeline.threadExecutionWidth)
        return false;

    @autoreleasepool {
        id<MTLBuffer> workspace =
            [g_dev newBufferWithLength:output_offset + output_bytes
                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> ids =
            [g_dev newBufferWithBytes:expert_ids
                                length:static_cast<size_t>(selected_count) * sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
        if (!workspace || !ids) {
            [workspace release];
            [ids release];
            return false;
        }
        std::memcpy([workspace contents], gate_up, gate_up_bytes);
        std::memset(static_cast<uint8_t*>([workspace contents]) + output_offset,
                    0, output_bytes);

        id<MTLCommandBuffer> command = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> encoder = command ? [command computeCommandEncoder] : nil;
        if (!encoder) {
            [workspace release];
            [ids release];
            return false;
        }
        const int selected = static_cast<int>(selected_count);
        const int width = static_cast<int>(intermediate);
        const int sw = swiglu ? 1 : 0;
        bool encoded = true;
        uint32_t activation_dispatches = 0;
        if (path == MetalMoeActivationTestPath::Batched2D) {
            encoded = enqueue_moe_glu(encoder, workspace, 0, output_offset,
                                      selected, width, sw);
            activation_dispatches = encoded ? 1u : 0u;
        } else {
            for (uint32_t slot = 0; encoded && slot != selected_count; ++slot) {
                const size_t gate_offset = static_cast<size_t>(slot) *
                                           2u * intermediate * sizeof(float);
                const size_t slot_output = output_offset +
                                           static_cast<size_t>(slot) *
                                           intermediate * sizeof(float);
                encoded = enqueue_glu(encoder, workspace, gate_offset,
                                      gate_offset + static_cast<size_t>(intermediate) * sizeof(float),
                                      slot_output, width, sw);
                if (encoded) ++activation_dispatches;
            }
        }
        [encoder endEncoding];
        if (!encoded) {
            [workspace release];
            [ids release];
            return false;
        }

        const auto wait_start = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        const auto wait_end = std::chrono::steady_clock::now();
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "[metal] MoE activation command: %s\n",
                         command_buffer_failure_detail(command).c_str());
            [workspace release];
            [ids release];
            return false;
        }
        record_completed_command(command, wait_start, wait_end);
        const CFTimeInterval gpu_start = [command GPUStartTime];
        const CFTimeInterval gpu_end = [command GPUEndTime];
        metrics->gpu_ms = gpu_end >= gpu_start ? 1000.0 * (gpu_end - gpu_start) : 0.0;
        metrics->command_buffers = 1;
        metrics->activation_dispatches = activation_dispatches;
        metrics->expert_ids_unchanged =
            std::memcmp([ids contents], expert_ids,
                        static_cast<size_t>(selected_count) * sizeof(uint32_t)) == 0;
        metrics->gate_up_unchanged =
            std::memcmp([workspace contents], gate_up, gate_up_bytes) == 0;
        std::memcpy(output,
                    static_cast<const uint8_t*>([workspace contents]) + output_offset,
                    output_bytes);
        const bool unchanged = metrics->expert_ids_unchanged && metrics->gate_up_unchanged;
        [workspace release];
        [ids release];
        return unchanged;
    }
}

// Test-only generic gathered-expert slice. The tensor is a stacked Q4_K
// [hidden, 2 * intermediate, expert] bank. Each selected expert produces one
// gate/up row pair in a single command buffer, followed by the exact
// GELU-tanh activation. The source span is supplied explicitly because a
// Tensor view does not own its backing length; this keeps the contract
// fail-closed at the boundary instead of relying on an unchecked pointer.
bool metal_test_moe_q4k_gate_up_gelu(const float* input, const Tensor& gate_up,
                                     size_t source_bytes,
                                     const uint32_t* expert_ids, uint32_t selected_count,
                                     float* output, int hidden, int intermediate,
                                     double* gpu_ms, uint32_t* pipeline_width,
                                     uint32_t* maximum_threads) {
    if (gpu_ms) *gpu_ms = 0.0;
    if (pipeline_width) *pipeline_width = 0;
    if (maximum_threads) *maximum_threads = 0;
    init();
    if (!g_dev || !g_q || !input || !expert_ids || !output || !gpu_ms ||
        !pipeline_width || !maximum_threads || !gate_up.data ||
        gate_up.type != GGMLType::Q4_K || gate_up.n_dims != 3 ||
        hidden < 256 || (hidden % 256) != 0 || intermediate <= 0 ||
        intermediate > INT_MAX / 2 || selected_count == 0 || selected_count > 16)
        return false;

    const uint64_t K = static_cast<uint32_t>(hidden);
    const uint64_t I = static_cast<uint32_t>(intermediate);
    const uint64_t N = I * 2u;
    const uint64_t E = gate_up.dims[2];
    if (E == 0 || E > 512 || gate_up.dims[0] != K || gate_up.dims[1] != N)
        return false;

    const uint64_t blocks_per_row = K / 256u;
    if (blocks_per_row > std::numeric_limits<uint64_t>::max() /
                            static_cast<uint64_t>(bytes_per_block(GGMLType::Q4_K)))
        return false;
    const uint64_t row_bytes = blocks_per_row * bytes_per_block(GGMLType::Q4_K);
    if (N > std::numeric_limits<uint64_t>::max() / row_bytes)
        return false;
    const uint64_t expert_stride = N * row_bytes;
    if (E > std::numeric_limits<uint64_t>::max() / expert_stride)
        return false;
    const uint64_t expected_span = E * expert_stride;
    if (expected_span == 0 || expected_span > std::numeric_limits<size_t>::max() ||
        static_cast<size_t>(expected_span) != source_bytes)
        return false;
    for (uint32_t slot = 0; slot != selected_count; ++slot)
        if (expert_ids[slot] >= E) return false;

    id<MTLComputePipelineState> q4k = get_q4k_id_pipe();
    id<MTLComputePipelineState> glu = get_glu_experts_pipe();
    if (!q4k || !glu || q4k.threadExecutionWidth == 0 || glu.threadExecutionWidth == 0 ||
        q4k.maxTotalThreadsPerThreadgroup < 64 ||
        glu.maxTotalThreadsPerThreadgroup < 1)
        return false;
    *pipeline_width = static_cast<uint32_t>(q4k.threadExecutionWidth);
    *maximum_threads = static_cast<uint32_t>(q4k.maxTotalThreadsPerThreadgroup);

    const size_t input_bytes = static_cast<size_t>(K) * sizeof(float);
    const size_t gate_up_values = static_cast<size_t>(selected_count) *
                                  static_cast<size_t>(N);
    const size_t hidden_values = static_cast<size_t>(selected_count) *
                                 static_cast<size_t>(I);
    if (gate_up_values > std::numeric_limits<size_t>::max() / sizeof(float) ||
        hidden_values > std::numeric_limits<size_t>::max() / sizeof(float) ||
        gate_up_values > std::numeric_limits<size_t>::max() - hidden_values)
        return false;
    const size_t gate_up_bytes = gate_up_values * sizeof(float);
    const size_t hidden_offset = gate_up_bytes;
    const size_t hidden_bytes = hidden_values * sizeof(float);
    const size_t output_bytes = hidden_offset + hidden_bytes;

    @autoreleasepool {
        id<MTLBuffer> input_buffer =
            [g_dev newBufferWithBytes:input length:input_bytes
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> ids_buffer =
            [g_dev newBufferWithBytes:expert_ids
                                length:static_cast<size_t>(selected_count) * sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer =
            [g_dev newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
        size_t weight_offset = 0;
        id<MTLBuffer> weight_buffer =
            get_weight_buf(gate_up.data, source_bytes, weight_offset);
        if (!input_buffer || !ids_buffer || !output_buffer || !weight_buffer) {
            [input_buffer release];
            [ids_buffer release];
            [output_buffer release];
            return false;
        }

        id<MTLCommandBuffer> command = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            command ? [command computeCommandEncoder] : nil;
        if (!encoder) {
            [input_buffer release];
            [ids_buffer release];
            [output_buffer release];
            return false;
        }
        const bool projection_bound = bind_q4k_id_dev(
            encoder, gate_up, hidden, static_cast<int>(N), ids_buffer, 0,
            static_cast<int>(selected_count), input_buffer, 0, output_buffer, 0, 0);
        const bool activation_bound = projection_bound && enqueue_moe_glu(
            encoder, output_buffer, 0, hidden_offset, static_cast<int>(selected_count),
            static_cast<int>(I), false);
        [encoder endEncoding];
        if (!activation_bound) {
            [input_buffer release];
            [ids_buffer release];
            [output_buffer release];
            return false;
        }

        const auto wait_start = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "[metal] generic MoE gate/up command: %s\n",
                         command_buffer_failure_detail(command).c_str());
            [input_buffer release];
            [ids_buffer release];
            [output_buffer release];
            return false;
        }
        record_completed_command(command, wait_start, std::chrono::steady_clock::now());
        const CFTimeInterval gpu_start = [command GPUStartTime];
        const CFTimeInterval gpu_end = [command GPUEndTime];
        if (gpu_end >= gpu_start) *gpu_ms = 1000.0 * (gpu_end - gpu_start);
        std::memcpy(output, static_cast<const uint8_t*>([output_buffer contents]) +
                              hidden_offset, hidden_bytes);
        [input_buffer release];
        [ids_buffer release];
        [output_buffer release];
        return true;
    }
}

bool metal_test_column_grouped_q4_v1(const ColumnGroupedQ4V1Storage& storage,
                                     const float* input, float* output, bool sparse,
                                     uint32_t* selected_columns, double* gpu_ms,
                                     uint32_t* thread_execution_width,
                                     uint32_t* max_threads_per_threadgroup) {
    ColumnGroupedQ4Error error = ColumnGroupedQ4Error::None;
    if (!input || !output || !selected_columns || !gpu_ms || !thread_execution_width ||
        !max_threads_per_threadgroup ||
        !validate_column_grouped_q4_v1(storage, storage.source_digest, &error) ||
        storage.contract.logical_k > static_cast<uint32_t>(INT_MAX) ||
        storage.contract.logical_n > static_cast<uint32_t>(INT_MAX)) return false;
    if (!column_grouped_q4_pipelines_ready()) return false;

    const uint32_t logical_k = storage.contract.logical_k;
    const uint32_t logical_n = storage.contract.logical_n;
    const size_t input_bytes = static_cast<size_t>(logical_k) * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(logical_n) * sizeof(float);
    id<MTLComputePipelineState> gemv = sparse ? g_column_grouped_q4_sparse_pipe
                                               : g_column_grouped_q4_dense_pipe;
    if (!gemv || gemv.threadExecutionWidth == 0 ||
        gemv.maxTotalThreadsPerThreadgroup < gemv.threadExecutionWidth) return false;

    @autoreleasepool {
        id<MTLBuffer> weights = [g_dev newBufferWithBytes:storage.bytes.data()
                                                    length:storage.bytes.size()
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> input_buffer = [g_dev newBufferWithBytes:input length:input_bytes
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer = [g_dev newBufferWithLength:output_bytes
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> selected_count = nil;
        id<MTLBuffer> selected_indices = nil;
        if (sparse) {
            selected_count = [g_dev newBufferWithLength:sizeof(uint32_t)
                                                 options:MTLResourceStorageModeShared];
            selected_indices = [g_dev newBufferWithLength:static_cast<size_t>(logical_k) * sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
        }
        auto release_buffers = [&] {
            [weights release];
            [input_buffer release];
            [output_buffer release];
            [selected_count release];
            [selected_indices release];
        };
        if (!weights || !input_buffer || !output_buffer ||
            (sparse && (!selected_count || !selected_indices))) {
            release_buffers();
            return false;
        }
        if (sparse) std::memset([selected_count contents], 0, sizeof(uint32_t));

        id<MTLCommandBuffer> command = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) {
            [encoder endEncoding];
            release_buffers();
            return false;
        }
        if (sparse) {
            id<MTLComputePipelineState> selector = g_column_grouped_q4_selector_pipe;
            const NSUInteger selector_threads = selector.threadExecutionWidth;
            if (!selector || selector_threads == 0 ||
                selector.maxTotalThreadsPerThreadgroup < selector_threads) {
                [encoder endEncoding];
                release_buffers();
                return false;
            }
            [encoder setComputePipelineState:selector];
            [encoder setBuffer:input_buffer offset:0 atIndex:0];
            [encoder setBuffer:selected_count offset:0 atIndex:1];
            [encoder setBuffer:selected_indices offset:0 atIndex:2];
            [encoder setBytes:&logical_k length:sizeof(logical_k) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake((logical_k + selector_threads - 1) / selector_threads, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(selector_threads, 1, 1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        const NSUInteger gemv_threads = gemv.threadExecutionWidth;
        [encoder setComputePipelineState:gemv];
        [encoder setBuffer:weights offset:0 atIndex:0];
        [encoder setBuffer:input_buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer offset:0 atIndex:2];
        if (sparse) {
            [encoder setBuffer:selected_count offset:0 atIndex:3];
            [encoder setBuffer:selected_indices offset:0 atIndex:4];
            [encoder setBytes:&logical_k length:sizeof(logical_k) atIndex:5];
            [encoder setBytes:&logical_n length:sizeof(logical_n) atIndex:6];
        } else {
            [encoder setBytes:&logical_k length:sizeof(logical_k) atIndex:3];
            [encoder setBytes:&logical_n length:sizeof(logical_n) atIndex:4];
        }
        [encoder dispatchThreadgroups:MTLSizeMake((logical_n + gemv_threads - 1) / gemv_threads, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(gemv_threads, 1, 1)];
        [encoder endEncoding];

        const auto wait_start = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        const auto wait_end = std::chrono::steady_clock::now();
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "[metal] ColumnGroupedQ4V1 command: %s\n",
                         command_buffer_failure_detail(command).c_str());
            release_buffers();
            return false;
        }
        record_completed_command(command, wait_start, wait_end);
        std::memcpy(output, [output_buffer contents], output_bytes);
        *selected_columns = sparse ? *static_cast<const uint32_t*>([selected_count contents]) : logical_k;
        const CFTimeInterval gpu_start = command.GPUStartTime;
        const CFTimeInterval gpu_end = command.GPUEndTime;
        *gpu_ms = gpu_end >= gpu_start ? 1000.0 * (gpu_end - gpu_start) : -1.0;
        *thread_execution_width = static_cast<uint32_t>(gemv.threadExecutionWidth);
        *max_threads_per_threadgroup = static_cast<uint32_t>(gemv.maxTotalThreadsPerThreadgroup);
        release_buffers();
        return true;
    }
}
#endif

struct TokWS {
    id<MTLBuffer> ws = nil;
    id<MTLBuffer> kcache = nil;
    id<MTLBuffer> vcache = nil;
    id<MTLCommandBuffer> cmd = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    int H = 0, inter = 0, exp_inter = 0, n_used = 0, n_experts = 0;
    int Hq = 0, Hk = 0, Dh = 0, query_capacity = 0, kv_stride = 0;
    int max_seq = 0, n_layers = 0, pos = 0;
    uint64_t kv_width_sum = 0;
    uint64_t kv_elements = 0;
    uint32_t batch_rows = 1;
    size_t x = 0, xn = 0, xb = 0, moe = 0, q = 0, k = 0, v = 0, ao = 0;
    size_t fg = 0, fu = 0, fh = 0, mgu = 0, mh = 0, md = 0;
    size_t route = 0, ids = 0, rw = 0, tmp = 0, ones = 0;
    size_t bytes = 0;
    bool live = false;
    bool async = false;
};

enum class TokProfileSegment : uint8_t { Start, Qkv, Attention, Ffn, Final };
enum class TokProfileMode : uint8_t { None, CounterSamples };

struct TokProfile {
    TokProfileMode mode = TokProfileMode::None;
    id<MTLCounterSampleBuffer> samples = nil;
    uint64_t timestamp_frequency = 0;
    size_t capacity = 0;
    std::vector<TokProfileSegment> segments;
    std::array<double, 4> milliseconds{};
    double cpu_wait_ms = 0.0;

    void reset() {
        if (samples) [samples release];
        mode = TokProfileMode::None;
        samples = nil;
        timestamp_frequency = 0;
        capacity = 0;
        segments.clear();
        milliseconds = {};
        cpu_wait_ms = 0.0;
    }
};

struct SparseFfnProxyResource {
    id<MTLBuffer> coefficients = nil;
    uint32_t input_blocks = 0;
    uint32_t output_blocks = 0;
    uint32_t selected_blocks = 0;
};

struct ActivationImportanceResource {
    id<MTLBuffer> values = nil;
    uint32_t width = 0;
};

struct ColumnGroupedAffineU2SkipResource {
    const uint8_t* values = nullptr;
    const uint8_t* scales = nullptr;
    const uint8_t* biases = nullptr;
    size_t values_bytes = 0;
    size_t scale_bytes = 0;
    size_t bias_bytes = 0;
    uint32_t logical_k = 0;
    uint32_t logical_n = 0;
    id<MTLBuffer> metadata = nil;
};

struct MetalBufferOwner {
    id<MTLBuffer> value = nil;

    ~MetalBufferOwner() { [value release]; }

    MetalBufferOwner() = default;
    MetalBufferOwner(const MetalBufferOwner&) = delete;
    MetalBufferOwner& operator=(const MetalBufferOwner&) = delete;

    id<MTLBuffer> get() const { return value; }

    id<MTLBuffer> take() {
        id<MTLBuffer> result = value;
        value = nil;
        return result;
    }
};

struct ColumnGroupedAffineU2SkipRegistrationCandidate {
    MetalBufferOwner metadata;
    MetalBufferOwner active_columns;
    MetalBufferOwner partial;
    MetalBufferOwner active_count;
    MetalBufferOwner numerical_error;
    MetalBufferOwner selected_bytes;
    uint32_t new_max_k = 0;
    uint32_t new_max_n = 0;
};

struct MetalTokContext {
    MetalWeightContext* weights = &g_legacy_weight_context;
    id<MTLCommandQueue> queue = nil;
    TokWS tok;
    id<MTLBuffer> kcache = nil;
    id<MTLBuffer> vcache = nil;
    size_t kv_bytes = 0;
    int kv_seeded_to = -1;
    id<MTLCommandBuffer> last = nil;
    id<MTLBuffer> logits = nil;
    id<MTLBuffer> sampler_result = nil;
    id<MTLBuffer> sparse_block_ids = nil;
    id<MTLBuffer> sparse_scores = nil;
#if defined(LAPLACE_METAL_TESTING)
    id<MTLBuffer> sparse_oracle_starts = nil;
    id<MTLBuffer> sparse_oracle_outputs = nil;
    id<MTLBuffer> sparse_oracle_prefix = nil;
    id<MTLBuffer> sparse_oracle_pairs = nil;
    id<MTLBuffer> sparse_oracle_pair_scores = nil;
    std::vector<id<MTLBuffer>> sparse_oracle_runs;
    size_t sparse_oracle_output_bytes = 0;
    uint32_t sparse_oracle_pair_count = 0;
    uint32_t sparse_oracle_slots = 0;
#endif
    uint32_t sparse_block_count = 0;
    bool sparse_affine = false;
    std::vector<SparseFfnProxyResource> sparse_proxies;
    std::vector<ActivationImportanceResource> importance;
    std::vector<ColumnGroupedAffineU2SkipResource> column_grouped_affine_u2_skip;
    id<MTLBuffer> column_grouped_affine_u2_active_count = nil;
    id<MTLBuffer> column_grouped_affine_u2_active_columns = nil;
    id<MTLBuffer> column_grouped_affine_u2_partial = nil;
    id<MTLBuffer> column_grouped_affine_u2_numerical_error = nil;
    id<MTLBuffer> column_grouped_affine_u2_selected_bytes = nil;
    uint32_t column_grouped_affine_u2_max_k = 0;
    uint32_t column_grouped_affine_u2_max_n = 0;
    size_t logits_bytes = 0;
    MetalTokMetrics metrics;
    TokProfile profile;
    bool single_command_buffer = false;
    bool detailed_errors = false;
    std::string failure_detail;
#if defined(LAPLACE_METAL_TESTING)
    bool fail_after_completed_submission = false;
#endif

    ~MetalTokContext() {
        if (tok.enc) [tok.enc endEncoding];
        [queue release];
        [tok.ws release];
        [kcache release];
        [vcache release];
        [logits release];
        [sampler_result release];
        [sparse_block_ids release];
        [sparse_scores release];
#if defined(LAPLACE_METAL_TESTING)
        [sparse_oracle_starts release];
        [sparse_oracle_outputs release];
        [sparse_oracle_prefix release];
        [sparse_oracle_pairs release];
        [sparse_oracle_pair_scores release];
        for (id<MTLBuffer> run : sparse_oracle_runs) [run release];
#endif
        for (const SparseFfnProxyResource& proxy : sparse_proxies) [proxy.coefficients release];
        for (const ActivationImportanceResource& resource : importance) [resource.values release];
        for (const ColumnGroupedAffineU2SkipResource& resource : column_grouped_affine_u2_skip)
            [resource.metadata release];
        [column_grouped_affine_u2_active_count release];
        [column_grouped_affine_u2_active_columns release];
        [column_grouped_affine_u2_partial release];
        [column_grouped_affine_u2_numerical_error release];
        [column_grouped_affine_u2_selected_bytes release];
        profile.reset();
    }
};

static MetalTokContext g_legacy_tok_context;
static thread_local MetalTokContext* g_active_tok_context = &g_legacy_tok_context;

class MetalTokScope {
public:
    explicit MetalTokScope(MetalTokContext& context)
        : previous_tok_(g_active_tok_context), previous_weights_(g_active_weight_context) {
        g_active_tok_context = &context;
        g_active_weight_context = context.weights;
    }

    ~MetalTokScope() {
        g_active_weight_context = previous_weights_;
        g_active_tok_context = previous_tok_;
    }

private:
    MetalTokContext* previous_tok_;
    MetalWeightContext* previous_weights_;
};

static MetalTokContext& active_tok_context() { return *g_active_tok_context; }

#define g_tok (active_tok_context().tok)
#define g_kcache (active_tok_context().kcache)
#define g_vcache (active_tok_context().vcache)
#define g_kv_bytes (active_tok_context().kv_bytes)
#define g_kv_seeded_to (active_tok_context().kv_seeded_to)
#define g_tok_last (active_tok_context().last)
#define g_logits_buf (active_tok_context().logits)
#define g_logits_bytes (active_tok_context().logits_bytes)

static void tok_split() {
    if (g_tok.enc) {
        [g_tok.enc endEncoding];
        g_tok.enc = nil;
    }
}

static bool tok_enc();

static void tok_profile_add(TokProfileSegment segment, double milliseconds) {
    if (segment == TokProfileSegment::Start) return;
    MetalTokContext& context = active_tok_context();
    context.profile.milliseconds[static_cast<size_t>(segment) - 1] += milliseconds;
}

static bool tok_profile_sample_count(uint32_t token_count, uint32_t layer_count,
                                     size_t& sample_count) {
    if (token_count == 0 || layer_count == 0) return false;
    constexpr size_t kMaximum = std::numeric_limits<size_t>::max();
    const size_t layers = static_cast<size_t>(layer_count);
    if (layers > (kMaximum - 1) / 3) return false;
    const size_t per_token = 3 * layers + 1;
    if (static_cast<size_t>(token_count) > (kMaximum - 1) / per_token) return false;
    sample_count = static_cast<size_t>(token_count) * per_token + 1;
    return true;
}

#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
bool metal_tok_profile_sample_count_for_testing(uint32_t token_count, uint32_t layer_count,
                                                uint64_t* sample_count) {
    size_t count = 0;
    if (!sample_count || !tok_profile_sample_count(token_count, layer_count, count) ||
        count > std::numeric_limits<uint64_t>::max()) return false;
    *sample_count = static_cast<uint64_t>(count);
    return true;
}
#endif

static void tok_profile_prepare(int n_layers, uint32_t token_count) {
    MetalTokContext& context = active_tok_context();
    context.profile.reset();
    const char* enabled = std::getenv("LAPLACE_CANONICAL_METAL_PROFILE");
    if (!enabled || enabled[0] != '1') return;
    size_t capacity = 0;
    if (n_layers < 1 ||
        !tok_profile_sample_count(token_count, static_cast<uint32_t>(n_layers), capacity)) {
        fprintf(stderr, "[metal] canonical profile disabled: sample count overflow\n");
        return;
    }
    if (!g_dev || ![g_dev supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]) {
        fprintf(stderr, "[metal] canonical profile disabled: dispatch-boundary timestamps unavailable\n");
        return;
    }
    id<MTLCounterSet> timestamp_set = nil;
    for (id<MTLCounterSet> set in g_dev.counterSets) {
        if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
            timestamp_set = set;
            break;
        }
    }
    if (!timestamp_set) {
        fprintf(stderr, "[metal] canonical profile disabled: timestamp counter set unavailable\n");
        return;
    }
    uint64_t frequency = 0;
    if (@available(macOS 26.0, *)) frequency = [g_dev queryTimestampFrequency];
    if (frequency == 0) {
        fprintf(stderr, "[metal] canonical profile disabled: timestamp frequency unavailable\n");
        return;
    }
    MTLCounterSampleBufferDescriptor* descriptor = [MTLCounterSampleBufferDescriptor new];
    descriptor.counterSet = timestamp_set;
    descriptor.label = @"laplace-canonical-token-profile";
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.sampleCount = capacity;
    NSError* error = nil;
    id<MTLCounterSampleBuffer> samples = [g_dev newCounterSampleBufferWithDescriptor:descriptor error:&error];
    [descriptor release];
    if (!samples) {
        fprintf(stderr, "[metal] canonical profile disabled: sample allocation failed\n");
        return;
    }
    context.profile.mode = TokProfileMode::CounterSamples;
    context.profile.samples = samples;
    context.profile.timestamp_frequency = frequency;
    context.profile.capacity = capacity;
}

static bool tok_profile_mark(TokProfileSegment segment) {
    MetalTokContext& context = active_tok_context();
    TokProfile& profile = context.profile;
    if (profile.mode == TokProfileMode::None) return true;
    if (profile.mode == TokProfileMode::CounterSamples) {
        if (!g_tok.enc || profile.segments.size() >= profile.capacity) {
            profile.mode = TokProfileMode::None;
            return true;
        }
        [g_tok.enc sampleCountersInBuffer:profile.samples atSampleIndex:profile.segments.size() withBarrier:YES];
        profile.segments.push_back(segment);
        return true;
    }
    return true;
}

static void tok_profile_finish() {
    MetalTokContext& context = active_tok_context();
    TokProfile& profile = context.profile;
    MetalTokMetrics& metrics = context.metrics;
    if (profile.mode == TokProfileMode::CounterSamples && profile.segments.size() > 1) {
        NSData* data = [profile.samples resolveCounterRange:NSMakeRange(0, profile.segments.size())];
        if (data.length == profile.segments.size() * sizeof(MTLCounterResultTimestamp)) {
            const auto* timestamps = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
            bool valid = true;
            for (size_t index = 1; index < profile.segments.size(); ++index) {
                const uint64_t before = timestamps[index - 1].timestamp;
                const uint64_t after = timestamps[index].timestamp;
                if (before == MTLCounterErrorValue || after == MTLCounterErrorValue || after < before) {
                    valid = false;
                    break;
                }
                tok_profile_add(profile.segments[index],
                                1000.0 * static_cast<double>(after - before) /
                                    static_cast<double>(profile.timestamp_frequency));
            }
            metrics.profiled = valid;
            metrics.counter_samples = valid;
            metrics.counter_sample_count = valid ? static_cast<uint64_t>(profile.segments.size()) : 0;
        }
    }
    metrics.qkv_gpu_ms = profile.milliseconds[0];
    metrics.attention_gpu_ms = profile.milliseconds[1];
    metrics.ffn_gpu_ms = profile.milliseconds[2];
    metrics.final_gpu_ms = profile.milliseconds[3];
}

static bool tok_enc() {
    id<MTLCommandQueue> queue = active_tok_context().queue
        ? active_tok_context().queue : g_q;
    if (!queue) return false;
    if (!g_tok.cmd) {
        if (active_tok_context().detailed_errors) {
            MTLCommandBufferDescriptor* descriptor = [MTLCommandBufferDescriptor new];
            descriptor.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
            g_tok.cmd = [queue commandBufferWithDescriptor:descriptor];
            [descriptor release];
        } else {
            g_tok.cmd = [queue commandBuffer];
        }
        if (!g_tok.cmd) return false;
    }
    if (!g_tok.enc) {
        g_tok.enc = [g_tok.cmd computeCommandEncoder];
        if (!g_tok.enc) return false;
        if (active_tok_context().detailed_errors)
            g_tok.enc.label = @"Laplace canonical full-token compute";
    }
    return true;
}

namespace {
bool encode_activation_importance(id<MTLComputeCommandEncoder>, id<MTLBuffer>, size_t,
                                  const ActivationImportanceResource&);
}

static bool tok_accumulate_importance(uint32_t slot, size_t input_offset, uint32_t width) {
    MetalTokContext& context = active_tok_context();
    return slot == UINT32_MAX ||
           (slot < context.importance.size() && context.importance[slot].width == width &&
            tok_enc() && encode_activation_importance(g_tok.enc, g_tok.ws, input_offset,
                                                       context.importance[slot]));
}

// Commit finished work so the GPU runs layer L while the CPU encodes
// L+1. The queue executes command buffers in commit order, so results
// are identical to one buffer; only the encode/exec overlap changes.
static void tok_maybe_commit() {
    if (active_tok_context().single_command_buffer || !g_tok.async || !g_tok.cmd) return;
    tok_split();
    [g_tok.cmd commit];
    g_tok.cmd = nil;
}

static void tok_barrier() {
    // Apple GPUs execute dispatches within one compute encoder in encode
    // order, so explicit barriers only drain the pipeline. Keep available
    // behind LAPLACE_TOK_BARRIERS=1 for A/B testing.
    static const bool on = [] {
        const char* e = std::getenv("LAPLACE_TOK_BARRIERS");
        return e && e[0] == '1';
    }();
    if (on && g_tok.enc)
        [g_tok.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

// Diagnostic: commit and wait so wall time of the encoded-so-far segment
// can be printed under LAPLACE_TOK_SEG=1.
static void tok_seg(const char* name) {
    static const bool on = [] {
        const char* e = std::getenv("LAPLACE_TOK_SEG");
        return e && e[0] == '1';
    }();
    if (active_tok_context().single_command_buffer || !on || !g_tok.cmd) return;
    tok_split();
    auto t0 = std::chrono::steady_clock::now();
    [g_tok.cmd commit];
    [g_tok.cmd waitUntilCompleted];
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    g_tok.cmd = nil;
    tok_enc();
    fprintf(stderr, "[tok-seg] %s %.3f ms\n", name, ms);
}

static bool tok_begin(int H, int inter, int exp_inter, int n_used, int n_experts,
                      int Hq, int Hk, int Dh, int max_seq, int n_layers, int pos,
                      uint32_t profile_token_count, bool continue_command_buffer,
                      uint32_t batch_rows = 1, int query_capacity = 0,
                      int kv_stride = 0, uint64_t kv_width_sum = 0) {
    init();
    if (!g_dev || H < 1 || max_seq < 1 || n_layers < 1) return false;
    if (n_used < 0 || n_experts < 0 || n_experts > 512) return false;
    const bool has_attention = Hq != 0 || Hk != 0 || Dh != 0;
    if ((has_attention && (Hq < 1 || Hk < 1 || Dh < 1)) ||
        (!has_attention && (Hq != 0 || Hk != 0 || Dh != 0)) ||
        (batch_rows != 1 && batch_rows != 2)) return false;
    if ((query_capacity == 0) != (kv_stride == 0) || query_capacity < 0 || kv_stride < 0 ||
        (has_attention && (Hq > INT_MAX / Dh || Hk > INT_MAX / Dh))) return false;
    const int layer_query_width = has_attention ? Hq * Dh : 0;
    const int layer_kv_width = has_attention ? Hk * Dh : 0;
    if (query_capacity == 0) query_capacity = layer_query_width;
    if (kv_stride == 0) kv_stride = layer_kv_width;
    if ((has_attention && (query_capacity < layer_query_width || kv_stride < layer_kv_width ||
                           (kv_width_sum != 0 && kv_width_sum < static_cast<uint64_t>(layer_kv_width)))) ||
        (!has_attention && (query_capacity != 0 || kv_stride != 0 || kv_width_sum != 0))) return false;
    id<MTLBuffer> keep_ws = g_tok.ws;
    size_t keep_bytes = g_tok.bytes;
    id<MTLCommandBuffer> keep_cmd = nil;
    if (continue_command_buffer) {
        tok_split();
        keep_cmd = g_tok.cmd;
        if (!keep_cmd) return false;
    }
    if (!continue_command_buffer) active_tok_context().failure_detail.clear();
    g_tok = TokWS{};
    g_tok.ws = keep_ws;
    g_tok.bytes = keep_bytes;
    g_tok.cmd = keep_cmd;
    g_tok.H = H;
    g_tok.inter = inter;
    g_tok.exp_inter = exp_inter;
    g_tok.n_used = n_used;
    g_tok.n_experts = n_experts;
    g_tok.Hq = Hq;
    g_tok.Hk = Hk;
    g_tok.Dh = Dh;
    g_tok.query_capacity = query_capacity;
    g_tok.kv_stride = kv_stride;
    g_tok.max_seq = max_seq;
    g_tok.n_layers = n_layers;
    g_tok.pos = pos;
    g_tok.kv_width_sum = kv_width_sum;
    g_tok.batch_rows = batch_rows;
    if (n_used != 0 && (n_used > INT_MAX / 2 ||
                          exp_inter > INT_MAX / (2 * n_used) ||
                          n_used > INT_MAX / H)) return false;
    int qn = query_capacity;
    int kn = kv_stride;
    int gu = n_used * 2 * exp_inter;
    size_t o = 0;
    bool workspace_valid = true;
    auto take = [&](int n, bool rows = true) {
        if (n < 1) {
            workspace_valid = false;
            return size_t{0};
        }
        const size_t count = static_cast<size_t>(n);
        const size_t multiplier = rows ? static_cast<size_t>(batch_rows) : 1;
        if (count > SIZE_MAX / multiplier / sizeof(float)) {
            workspace_valid = false;
            return size_t{0};
        }
        const size_t bytes = count * multiplier * sizeof(float);
        if (bytes > SIZE_MAX - 255 || o > SIZE_MAX - align256(bytes)) {
            workspace_valid = false;
            return size_t{0};
        }
        size_t at = o;
        o += align256(bytes);
        return at;
    };
    g_tok.x = take(H);
    g_tok.xn = take(H);
    g_tok.xb = take(H);
    g_tok.moe = take(H);
    g_tok.q = take(std::max(qn, 1));
    g_tok.k = take(std::max(kn, 1));
    g_tok.v = take(std::max(kn, 1));
    g_tok.ao = take(std::max(qn, 1));
    g_tok.fg = take(inter);
    g_tok.fu = take(inter);
    g_tok.fh = take(inter);
    g_tok.mgu = take(std::max(gu, 1));
    g_tok.mh = take(std::max(n_used * exp_inter, 1));
    g_tok.md = take(std::max(n_used * H, 1));
    g_tok.route = take(std::max(n_experts, 1));
    g_tok.ids = take(std::max(n_used, 1));
    g_tok.rw = take(std::max(n_used, 1));
    g_tok.tmp = take(H);
    g_tok.ones = take(H, false);
    g_tok.bytes = o;
    if (!workspace_valid || o == 0) {
        g_tok = TokWS{};
        return false;
    }
    if (!g_tok.ws || keep_bytes < o) {
        g_tok.ws = [g_dev newBufferWithLength:g_tok.bytes
                                     options:MTLResourceStorageModeShared];
    }
    if (has_attention) {
        const size_t sequence = static_cast<size_t>(max_seq);
        const uint64_t widths = kv_width_sum != 0
            ? kv_width_sum
            : static_cast<uint64_t>(n_layers) * static_cast<uint64_t>(kv_stride);
        if (widths == 0 || widths > SIZE_MAX || sequence > SIZE_MAX / static_cast<size_t>(widths) ||
            sequence * static_cast<size_t>(widths) > SIZE_MAX / sizeof(float)) {
            g_tok = TokWS{};
            return false;
        }
        g_tok.kv_elements = sequence * static_cast<size_t>(widths);
        const size_t kvb = static_cast<size_t>(g_tok.kv_elements) * sizeof(float);
        if (kvb > UINT64_MAX / 2u) {
            g_tok = TokWS{};
            return false;
        }
        if (!g_kcache || g_kv_bytes != kvb) {
            [g_kcache release];
            [g_vcache release];
            g_kcache = [g_dev newBufferWithLength:kvb options:MTLResourceStorageModeShared];
            g_vcache = [g_dev newBufferWithLength:kvb options:MTLResourceStorageModeShared];
            g_kv_bytes = kvb;
            g_kv_seeded_to = -1;
        }
        g_tok.kcache = g_kcache;
        g_tok.vcache = g_vcache;
    } else {
        [g_kcache release];
        [g_vcache release];
        g_kcache = nil;
        g_vcache = nil;
        g_kv_bytes = 0;
        g_kv_seeded_to = -1;
    }
    if (!g_tok.ws || (has_attention && (!g_tok.kcache || !g_tok.vcache))) {
        g_tok = TokWS{};
        return false;
    }
    if (has_attention && pos == 0) {
        const size_t kvb = g_kv_bytes;
        memset([g_kcache contents], 0, kvb);
        memset([g_vcache contents], 0, kvb);
        g_kv_seeded_to = 0;
    }
    float* ones = (float*)((uint8_t*)[g_tok.ws contents] + g_tok.ones);
    for (int i = 0; i < H; i++) ones[i] = 1.0f;
    if (!active_tok_context().single_command_buffer) {
        if (const char* e = std::getenv("LAPLACE_TOK_ASYNC")) g_tok.async = e[0] == '1';
    }
    g_tok.live = true;
    if (!continue_command_buffer) {
        active_tok_context().metrics = {};
        if (active_tok_context().column_grouped_affine_u2_numerical_error)
            *static_cast<uint32_t*>(
                active_tok_context().column_grouped_affine_u2_numerical_error.contents) = 0u;
        if (active_tok_context().column_grouped_affine_u2_selected_bytes)
            *static_cast<uint64_t*>(
                active_tok_context().column_grouped_affine_u2_selected_bytes.contents) = 0u;
        tok_profile_prepare(n_layers, profile_token_count);
    }
    static std::once_flag once;
    std::call_once(once, []{ fprintf(stderr, "[metal] token_cb=1\n"); });
    return true;
}

bool metal_tok_begin(int H, int inter, int exp_inter, int n_used, int n_experts,
                     int Hq, int Hk, int Dh, int max_seq, int n_layers, int pos) {
    return tok_begin(H, inter, exp_inter, n_used, n_experts, Hq, Hk, Dh, max_seq, n_layers, pos, 1, false);
}

bool metal_tok_active() { return g_tok.live; }

// Diagnostic: commit and wait on the token CB so callers can measure
// per-layer GPU wall time under LAPLACE_TOK_TIMING=1.
bool metal_tok_flush(double* ms_out) {
    if (!g_tok.live || !g_tok.cmd) { if (ms_out) *ms_out = 0; return true; }
    tok_split();
    auto t0 = std::chrono::steady_clock::now();
    [g_tok.cmd commit];
    [g_tok.cmd waitUntilCompleted];
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    bool ok = g_tok.cmd.status != MTLCommandBufferStatusError;
    g_tok.cmd = nil;
    g_tok.enc = nil;
    if (ms_out) *ms_out = ms;
    return ok;
}

void metal_tok_upload_x(const float* x, int H) {
    if (!g_tok.live || !x) return;
    memcpy((uint8_t*)[g_tok.ws contents] + g_tok.x, x, (size_t)H * 4);
}

static bool tok_upload_embedding_to(const Tensor& embedding, uint32_t token, int H, int vocab,
                                    float scale, size_t destination, bool profile_start) {
    if (!g_tok.live || !embedding.data || H < 1 || vocab < 1 ||
        !std::isfinite(scale) || scale <= 0.0f || token >= static_cast<uint32_t>(vocab)) return false;
    if (embedding.type != GGMLType::F16 && embedding.type != GGMLType::Q4_K && embedding.type != GGMLType::Q6_K) return false;
    if (embedding.type != GGMLType::F16 && H % 256 != 0) return false;
    if (embedding.n_dims != 2 || embedding.dims[0] != static_cast<uint64_t>(H) ||
        embedding.dims[1] != static_cast<uint64_t>(vocab)) return false;
    const size_t rows = embedding.type == GGMLType::F16 ? static_cast<size_t>(H) : static_cast<size_t>(H / 256);
    const size_t row_bytes = embedding.type == GGMLType::F16 ? sizeof(uint16_t) :
        embedding.type == GGMLType::Q4_K ? sizeof(kernels::block_q4_K) : sizeof(kernels::block_q6_K);
    if (rows > std::numeric_limits<size_t>::max() / static_cast<size_t>(vocab) / row_bytes) return false;
    const size_t table_bytes = rows * static_cast<size_t>(vocab) * row_bytes;
    if (!strict_tensor_data_span(embedding, table_bytes)) return false;
    size_t offset = 0;
    id<MTLBuffer> table = get_weight_buf(embedding.data, table_bytes, offset);
    const char* pipeline_name = embedding.type == GGMLType::F16 ? "embedding_f16" :
        embedding.type == GGMLType::Q4_K ? "embedding_q4k" : "embedding_q6k";
    auto pipeline = named_pipe(pipeline_name);
    if (!table || !pipeline || !tok_enc()) return false;
    [g_tok.enc setComputePipelineState:pipeline];
    [g_tok.enc setBuffer:table offset:offset atIndex:0];
    [g_tok.enc setBuffer:g_tok.ws offset:destination atIndex:1];
    [g_tok.enc setBytes:&token length:4 atIndex:2];
    [g_tok.enc setBytes:&H length:4 atIndex:3];
    [g_tok.enc setBytes:&scale length:4 atIndex:4];
    enc_1d(g_tok.enc, pipeline, H);
    return !profile_start || tok_profile_mark(TokProfileSegment::Start);
}

bool metal_tok_upload_embedding(const Tensor& embedding, uint32_t token, int H, int vocab) {
    return tok_upload_embedding_to(embedding, token, H, vocab, 1.0f, g_tok.x, true);
}

bool metal_tok_kv_needs_seed() { return g_tok.live && g_kv_seeded_to < 0; }

void metal_tok_import_kv(int cache_id, int t, const float* k, const float* v, int kn) {
    if (!g_tok.live || !g_tok.kcache || !k || !v) return;
    if (g_tok.kv_width_sum != 0) return;
    if (cache_id < 0 || cache_id >= g_tok.n_layers) return;
    if (t < 0 || t >= g_tok.max_seq) return;
    int stride = g_tok.kv_stride;
    if (kn > stride) kn = stride;
    size_t off = ((size_t)cache_id * (size_t)g_tok.max_seq + (size_t)t) * (size_t)stride;
    memcpy((float*)[g_tok.kcache contents] + off, k, (size_t)kn * 4);
    memcpy((float*)[g_tok.vcache contents] + off, v, (size_t)kn * 4);
    if (t + 1 > g_kv_seeded_to) g_kv_seeded_to = t + 1;
}

static bool bind_affine_u2_256(id<MTLComputeCommandEncoder> enc, const Tensor& w,
                               int K, int N, id<MTLBuffer> xbuf, size_t xoff,
                               id<MTLBuffer> ybuf, size_t yoff) {
    if (w.type != GGMLType::GROUPED_AFFINE_U2_256 || w.mlx_bits != 2 ||
        w.mlx_group_size != 256 || w.n_dims != 2 || K <= 0 || N <= 0 ||
        (K % 512) != 0 || (N % 8) != 0 ||
        w.dims[0] != static_cast<uint64_t>(K) || w.dims[1] != static_cast<uint64_t>(N) ||
        !w.data || !w.scales || !w.biases || !xbuf || !ybuf || !enc) return false;
    const uint64_t values = static_cast<uint64_t>(K) * static_cast<uint64_t>(N) / 4;
    const uint64_t planes = static_cast<uint64_t>(N) * (static_cast<uint64_t>(K) / 256);
    if (planes == 0 || values > SIZE_MAX || planes > SIZE_MAX / sizeof(uint16_t)) return false;
    const size_t values_bytes = static_cast<size_t>(values);
    const size_t plane_bytes = static_cast<size_t>(planes) * sizeof(uint16_t);
    if (!strict_tensor_data_span(w, values_bytes) ||
        (g_active_weight_context->require_registered_weights &&
         (w.scale_bytes != plane_bytes || w.bias_bytes != plane_bytes))) return false;
    size_t values_offset = 0;
    size_t scales_offset = 0;
    size_t biases_offset = 0;
    id<MTLBuffer> values_buffer = get_weight_buf(w.data, values_bytes, values_offset);
    id<MTLBuffer> scales_buffer = get_weight_buf(w.scales, plane_bytes, scales_offset);
    id<MTLBuffer> biases_buffer = get_weight_buf(w.biases, plane_bytes, biases_offset);
    // Canonical sessions require all three source ranges to be registered;
    // nil here is a hard binding failure, never an implicit upload/copy.
    id<MTLComputePipelineState> pipeline = get_affine_u2_256_pipe();
    if (!values_buffer || !scales_buffer || !biases_buffer || !pipeline ||
        pipeline.threadExecutionWidth != 32 || pipeline.maxTotalThreadsPerThreadgroup < 64)
        return false;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:values_buffer offset:values_offset atIndex:0];
    [enc setBuffer:scales_buffer offset:scales_offset atIndex:1];
    [enc setBuffer:biases_buffer offset:biases_offset atIndex:2];
    [enc setBuffer:xbuf offset:xoff atIndex:3];
    [enc setBuffer:ybuf offset:yoff atIndex:4];
    [enc setBytes:&K length:sizeof(K) atIndex:5];
    [enc setBytes:&N length:sizeof(N) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(N / 8), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    return true;
}

static bool bind_column_grouped_affine_u2_skip_256(
    id<MTLComputeCommandEncoder> encoder, const Tensor& tensor,
    int K, int N, id<MTLBuffer> input, size_t input_offset,
    id<MTLBuffer> output, size_t output_offset) {
    MetalTokContext& context = active_tok_context();
    if (!encoder || !input || !output ||
        tensor.type != GGMLType::COLUMN_GROUPED_AFFINE_U2_SKIP_256 ||
        tensor.n_dims != 2 || K <= 0 || N <= 0 || (N % 256) != 0 ||
        tensor.dims[0] != static_cast<uint64_t>(K) ||
        tensor.dims[1] != static_cast<uint64_t>(N) ||
        tensor.mlx_bits != 2 || tensor.mlx_group_size != 256 ||
        tensor.data_bytes > SIZE_MAX || tensor.scale_bytes > SIZE_MAX ||
        tensor.bias_bytes > SIZE_MAX ||
        !context.column_grouped_affine_u2_active_count ||
        !context.column_grouped_affine_u2_active_columns ||
        !context.column_grouped_affine_u2_partial ||
        !context.column_grouped_affine_u2_numerical_error ||
        !context.column_grouped_affine_u2_selected_bytes ||
        static_cast<uint32_t>(K) > context.column_grouped_affine_u2_max_k ||
        static_cast<uint32_t>(N) > context.column_grouped_affine_u2_max_n)
        return false;
    const ColumnGroupedAffineU2SkipResource* resource = nullptr;
    for (const ColumnGroupedAffineU2SkipResource& candidate :
         context.column_grouped_affine_u2_skip) {
        if (candidate.values == tensor.data && candidate.scales == tensor.scales &&
            candidate.biases == tensor.biases && candidate.logical_k == static_cast<uint32_t>(K) &&
            candidate.logical_n == static_cast<uint32_t>(N) &&
            candidate.values_bytes == static_cast<size_t>(tensor.data_bytes) &&
            candidate.scale_bytes == static_cast<size_t>(tensor.scale_bytes) &&
            candidate.bias_bytes == static_cast<size_t>(tensor.bias_bytes)) {
            resource = &candidate;
            break;
        }
    }
    if (!resource || !resource->metadata) return false;
    const uint64_t output_blocks_64 = static_cast<uint64_t>(N / 256);
    if (output_blocks_64 > UINT32_MAX ||
        static_cast<uint64_t>(K) > UINT64_MAX / output_blocks_64 ||
        static_cast<uint64_t>(K) * output_blocks_64 > SIZE_MAX / 64u)
        return false;
    const uint32_t logical_k = static_cast<uint32_t>(K);
    const uint32_t logical_n = static_cast<uint32_t>(N);
    const uint32_t output_blocks = static_cast<uint32_t>(output_blocks_64);
    constexpr uint32_t split_count = 16u;
    const size_t values_bytes = static_cast<size_t>(
        static_cast<uint64_t>(K) * output_blocks_64 * 64u);
    size_t values_offset = 0;
    id<MTLBuffer> values = get_weight_buf(tensor.data, values_bytes, values_offset);
    id<MTLComputePipelineState> selector =
        named_pipe("column_grouped_affine_u2_skip_256_select");
    id<MTLComputePipelineState> partial =
        named_pipe("column_grouped_affine_u2_skip_256_partial");
    id<MTLComputePipelineState> reduce =
        named_pipe("column_grouped_affine_u2_skip_256_reduce");
    if (!values || !selector || !partial || !reduce ||
        resource->metadata.length < static_cast<size_t>(
            static_cast<uint64_t>(K) * output_blocks_64 * sizeof(uint32_t)) ||
        context.column_grouped_affine_u2_active_columns.length <
            static_cast<size_t>(K) * sizeof(uint32_t) ||
        context.column_grouped_affine_u2_partial.length <
            static_cast<size_t>(N) * split_count * sizeof(float))
        return false;

    [encoder setComputePipelineState:selector];
    [encoder setBuffer:input offset:input_offset atIndex:0];
    [encoder setBuffer:context.column_grouped_affine_u2_active_count offset:0 atIndex:1];
    [encoder setBuffer:context.column_grouped_affine_u2_active_columns offset:0 atIndex:2];
    [encoder setBuffer:context.column_grouped_affine_u2_numerical_error offset:0 atIndex:3];
    [encoder setBuffer:context.column_grouped_affine_u2_selected_bytes offset:0 atIndex:4];
    [encoder setBytes:&logical_k length:sizeof(logical_k) atIndex:5];
    [encoder setBytes:&output_blocks length:sizeof(output_blocks) atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:partial];
    [encoder setBuffer:values offset:values_offset atIndex:0];
    [encoder setBuffer:resource->metadata offset:0 atIndex:1];
    [encoder setBuffer:input offset:input_offset atIndex:2];
    [encoder setBuffer:context.column_grouped_affine_u2_partial offset:0 atIndex:3];
    [encoder setBuffer:context.column_grouped_affine_u2_active_count offset:0 atIndex:4];
    [encoder setBuffer:context.column_grouped_affine_u2_active_columns offset:0 atIndex:5];
    [encoder setBytes:&logical_k length:sizeof(logical_k) atIndex:6];
    [encoder setBytes:&logical_n length:sizeof(logical_n) atIndex:7];
    [encoder setBytes:&split_count length:sizeof(split_count) atIndex:8];
    [encoder dispatchThreadgroups:MTLSizeMake(
        static_cast<NSUInteger>(output_blocks) * split_count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:reduce];
    [encoder setBuffer:context.column_grouped_affine_u2_partial offset:0 atIndex:0];
    [encoder setBuffer:output offset:output_offset atIndex:1];
    [encoder setBytes:&logical_n length:sizeof(logical_n) atIndex:2];
    [encoder setBytes:&split_count length:sizeof(split_count) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(output_blocks, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return true;
}

static void tok_record_projection_bytes(const Tensor& w, int K, int N) {
    if (w.type == GGMLType::GROUPED_AFFINE_U2_256 && w.mlx_bits == 2 &&
        w.mlx_group_size == 256 &&
        K > 0 && N > 0 && (K % 256) == 0) {
        const uint64_t values = static_cast<uint64_t>(K) * static_cast<uint64_t>(N) / 4;
        const uint64_t planes = static_cast<uint64_t>(N) * (static_cast<uint64_t>(K) / 256);
        if (planes <= (UINT64_MAX - values) / 4)
            active_tok_context().metrics.requested_projection_source_bytes += values + planes * 4;
        return;
    }
    if (w.type == GGMLType::COLUMN_GROUPED_AFFINE_U2_SKIP_256) return;
    const int block_elements = elements_per_block(w.type);
    const size_t block_bytes = bytes_per_block(w.type);
    if (K > 0 && N > 0 && block_elements > 0 && K % block_elements == 0 && block_bytes != 0) {
        const uint64_t row_bytes = static_cast<uint64_t>(K / block_elements) * block_bytes;
        active_tok_context().metrics.requested_projection_source_bytes += static_cast<uint64_t>(N) * row_bytes;
    }
}

static void tok_record_projection_dispatch(GGMLType type) {
    ++active_tok_context().metrics.projection_dispatches;
    if (type == GGMLType::Q4_K) ++active_tok_context().metrics.q4k_projection_dispatches;
    if (type == GGMLType::Q6_K) ++active_tok_context().metrics.q6k_projection_dispatches;
    if (type == GGMLType::GROUPED_AFFINE_U2_256)
        ++active_tok_context().metrics.grouped_affine_u2_projection_dispatches;
    if (type == GGMLType::COLUMN_GROUPED_AFFINE_U2_SKIP_256)
        ++active_tok_context().metrics.column_grouped_affine_u2_skip_projection_dispatches;
}

static bool tok_bind(const Tensor& w, size_t xoff, size_t yoff, int K, int N) {
    if (!tok_enc()) return false;
    bool bound = false;
    if (w.type == GGMLType::GROUPED_AFFINE_U2_256)
        bound = bind_affine_u2_256(g_tok.enc, w, K, N, g_tok.ws, xoff, g_tok.ws, yoff);
    else if (w.type == GGMLType::COLUMN_GROUPED_AFFINE_U2_SKIP_256)
        bound = bind_column_grouped_affine_u2_skip_256(
            g_tok.enc, w, K, N, g_tok.ws, xoff, g_tok.ws, yoff);
    else if (w.type == GGMLType::Q4_K)
        bound = bind_q4k(g_tok.enc, w, K, N, g_tok.ws, xoff, g_tok.ws, yoff);
    else if (w.type == GGMLType::Q8_0)
        bound = bind_q8(g_tok.enc, w, K, N, g_tok.ws, xoff, g_tok.ws, yoff);
    else if (w.type == GGMLType::Q6_K)
        bound = bind_q6k(g_tok.enc, w, K, N, g_tok.ws, xoff, g_tok.ws, yoff);
    else
        bound = bind_fused(g_tok.enc, w, K, N, g_tok.ws, xoff, g_tok.ws, yoff)
             || bind_any(g_tok.enc, w, K, N, g_tok.ws, xoff, g_tok.ws, yoff);
    if (!bound) return false;
    tok_record_projection_bytes(w, K, N);
    tok_record_projection_dispatch(w.type);
    return true;
}

static bool tok_bind_f16_rows(const Tensor& w, size_t xoff, size_t yoff, int M, int K, int N) {
    if (M != 2 || w.type != GGMLType::F16 || w.n_dims != 2 || K <= 0 || N <= 0 ||
        w.dims[0] != static_cast<uint64_t>(K) || w.dims[1] != static_cast<uint64_t>(N) ||
        !tok_enc()) return false;
    if (static_cast<uint64_t>(K) > SIZE_MAX / static_cast<uint64_t>(N) / sizeof(uint16_t))
        return false;
    size_t weight_offset = 0;
    const size_t weight_bytes = static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(uint16_t);
    if (!strict_tensor_data_span(w, weight_bytes)) return false;
    id<MTLBuffer> weights = get_weight_buf(w.data, weight_bytes, weight_offset);
    id<MTLComputePipelineState> pipeline = get_prefill_f16_pipe();
    // The active canonical batch context requires registered weight spans.
    // A nil buffer here is a hard error, never an upload/copy fallback.
    if (!weights || !pipeline) return false;
    [g_tok.enc setComputePipelineState:pipeline];
    [g_tok.enc setBuffer:weights offset:weight_offset atIndex:0];
    [g_tok.enc setBuffer:g_tok.ws offset:xoff atIndex:1];
    [g_tok.enc setBuffer:g_tok.ws offset:yoff atIndex:2];
    [g_tok.enc setBytes:&K length:sizeof(K) atIndex:3];
    [g_tok.enc setBytes:&N length:sizeof(N) atIndex:4];
    [g_tok.enc setBytes:&M length:sizeof(M) atIndex:5];
    const uint64_t outputs = static_cast<uint64_t>(M) * static_cast<uint64_t>(N);
    if (outputs > static_cast<uint64_t>(INT_MAX)) return false;
    enc_1d(g_tok.enc, pipeline, static_cast<int>(outputs), 64);
    tok_record_projection_bytes(w, K, N);
    ++active_tok_context().metrics.projection_dispatches;
    ++active_tok_context().metrics.batched_projection_dispatches;
    return true;
}

static bool tok_select_sparse_window(id<MTLBuffer> scores, uint32_t total_blocks,
                                     uint32_t selected_blocks) {
    MetalTokContext& context = active_tok_context();
    if (!scores || !tok_enc() || !context.sparse_affine ||
        !context.sparse_block_ids || selected_blocks == 0 ||
        selected_blocks != context.sparse_block_count || selected_blocks > total_blocks)
        return false;
    id<MTLComputePipelineState> pipeline = named_pipe("sparse_select_contiguous_window");
    if (!pipeline || pipeline.threadExecutionWidth != 32 ||
        pipeline.maxTotalThreadsPerThreadgroup < 32) return false;
    [g_tok.enc setComputePipelineState:pipeline];
    [g_tok.enc setBuffer:scores offset:0 atIndex:0];
    [g_tok.enc setBuffer:context.sparse_block_ids offset:0 atIndex:1];
    [g_tok.enc setBytes:&total_blocks length:sizeof(total_blocks) atIndex:2];
    [g_tok.enc setBytes:&selected_blocks length:sizeof(selected_blocks) atIndex:3];
    [g_tok.enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [g_tok.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return true;
}

static bool tok_proxy_sparse_scores(size_t input_offset, id<MTLBuffer> coefficients,
                                    id<MTLBuffer> scores, uint32_t input_blocks,
                                    uint32_t output_blocks) {
    if (!coefficients || !scores || !tok_enc() || input_blocks == 0 || output_blocks == 0 ||
        input_blocks > 4096 || output_blocks > 4096 ||
        static_cast<size_t>(output_blocks) > SIZE_MAX / input_blocks ||
        static_cast<size_t>(input_blocks) * output_blocks > SIZE_MAX / sizeof(float) ||
        coefficients.length < static_cast<size_t>(input_blocks) * output_blocks * sizeof(float) ||
        scores.length < static_cast<size_t>(output_blocks) * sizeof(float)) return false;
    id<MTLComputePipelineState> pipeline = named_pipe("sparse_proxy_block_scores");
    if (!pipeline || pipeline.threadExecutionWidth != 32 ||
        pipeline.maxTotalThreadsPerThreadgroup < 32) return false;
    [g_tok.enc setComputePipelineState:pipeline];
    [g_tok.enc setBuffer:g_tok.ws offset:input_offset atIndex:0];
    [g_tok.enc setBuffer:coefficients offset:0 atIndex:1];
    [g_tok.enc setBuffer:scores offset:0 atIndex:2];
    [g_tok.enc setBytes:&input_blocks length:sizeof(input_blocks) atIndex:3];
    [g_tok.enc setBytes:&output_blocks length:sizeof(output_blocks) atIndex:4];
    [g_tok.enc dispatchThreadgroups:MTLSizeMake(output_blocks, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [g_tok.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return true;
}

static bool tok_bind_sparse_rows(const Tensor& w, size_t xoff, size_t yoff,
                                 int K, int packed_N, uint32_t block_offset = 0);
#if defined(LAPLACE_METAL_TESTING)
static bool tok_bind_sparse_columns_to(const Tensor& w, id<MTLBuffer> input, size_t input_offset,
                                       id<MTLBuffer> output, size_t output_offset,
                                       id<MTLBuffer> block_ids, uint32_t block_count,
                                       int full_K, int N);
#endif

static bool tok_select_sparse_proxy(size_t input_offset, uint32_t slot) {
    MetalTokContext& context = active_tok_context();
    if (slot >= context.sparse_proxies.size()) return false;
    const SparseFfnProxyResource& proxy = context.sparse_proxies[slot];
    if (!proxy.coefficients || !context.sparse_scores || !context.sparse_block_ids ||
        context.sparse_block_ids.length < 2 * sizeof(uint32_t)) return false;
    context.sparse_block_count = proxy.selected_blocks;
    context.sparse_affine = true;
    return tok_proxy_sparse_scores(input_offset, proxy.coefficients, context.sparse_scores,
                                   proxy.input_blocks, proxy.output_blocks) &&
           tok_select_sparse_window(context.sparse_scores, proxy.output_blocks,
                                    proxy.selected_blocks);
}

#if defined(LAPLACE_METAL_TESTING)
static bool tok_record_sparse_window(uint32_t slot) {
    MetalTokContext& context = active_tok_context();
    if (!context.sparse_block_ids || !context.sparse_oracle_starts ||
        slot >= context.sparse_oracle_slots || !tok_enc()) return false;
    id<MTLComputePipelineState> pipeline = named_pipe("sparse_record_window");
    if (!pipeline || pipeline.threadExecutionWidth == 0 ||
        pipeline.maxTotalThreadsPerThreadgroup < pipeline.threadExecutionWidth) return false;
    const uint32_t index = 3u * slot;
    [g_tok.enc setComputePipelineState:pipeline];
    [g_tok.enc setBuffer:context.sparse_block_ids offset:0 atIndex:0];
    [g_tok.enc setBuffer:context.sparse_oracle_starts offset:0 atIndex:1];
    [g_tok.enc setBytes:&index length:sizeof(index) atIndex:2];
    [g_tok.enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
    return true;
}

static bool tok_record_sparse_pair(uint32_t slot, uint32_t run_blocks) {
    MetalTokContext& context = active_tok_context();
    if (!context.sparse_block_ids || !context.sparse_oracle_starts || run_blocks == 0 ||
        slot >= context.sparse_oracle_slots || !tok_enc()) return false;
    id<MTLComputePipelineState> pipeline = named_pipe("sparse_record_pair");
    if (!pipeline || pipeline.threadExecutionWidth == 0 ||
        pipeline.maxTotalThreadsPerThreadgroup < pipeline.threadExecutionWidth) return false;
    const uint32_t index = 3u * slot + 1u;
    [g_tok.enc setComputePipelineState:pipeline];
    [g_tok.enc setBuffer:context.sparse_block_ids offset:0 atIndex:0];
    [g_tok.enc setBuffer:context.sparse_oracle_starts offset:0 atIndex:1];
    [g_tok.enc setBytes:&index length:sizeof(index) atIndex:2];
    [g_tok.enc setBytes:&run_blocks length:sizeof(run_blocks) atIndex:3];
    [g_tok.enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
    return true;
}

static bool tok_select_sparse_dense_oracle(const Tensor& gate, const Tensor& up,
                                           const Tensor& down,
                                           size_t input_offset, uint32_t slot,
                                           int hidden, int full_intermediate,
                                           int packed_intermediate) {
    MetalTokContext& context = active_tok_context();
    if (slot >= context.sparse_proxies.size() || full_intermediate <= 0 ||
        full_intermediate % 256 != 0 || packed_intermediate <= 0 ||
        packed_intermediate > full_intermediate || g_tok.inter < full_intermediate)
        return false;
    const SparseFfnProxyResource& proxy = context.sparse_proxies[slot];
    if (proxy.output_blocks != static_cast<uint32_t>(full_intermediate / 256) ||
        proxy.selected_blocks != static_cast<uint32_t>(packed_intermediate / 256) ||
        !tok_select_sparse_proxy(input_offset, slot) ||
        !tok_record_sparse_window(slot) ||
        !tok_bind(gate, input_offset, g_tok.fg, hidden, full_intermediate) ||
        !tok_bind(up, input_offset, g_tok.fu, hidden, full_intermediate) ||
        !tok_enc() ||
        !enqueue_glu(g_tok.enc, g_tok.ws, g_tok.fg, g_tok.fu, g_tok.fh,
                     full_intermediate, 1)) return false;
    if (proxy.selected_blocks != 34 || proxy.output_blocks != 68) return false;
    const uint32_t run_blocks = proxy.selected_blocks / 2;
    if (run_blocks == 0 || !context.sparse_scores ||
        static_cast<size_t>(proxy.output_blocks) > SIZE_MAX / static_cast<size_t>(hidden) / sizeof(float) ||
        static_cast<size_t>(proxy.output_blocks + 1) > SIZE_MAX / static_cast<size_t>(hidden) / sizeof(float))
        return false;
    const size_t contribution_bytes = static_cast<size_t>(proxy.output_blocks) * hidden * sizeof(float);
    const size_t prefix_bytes = static_cast<size_t>(proxy.output_blocks + 1) * hidden * sizeof(float);
    if (!context.sparse_oracle_outputs || context.sparse_oracle_output_bytes < contribution_bytes) {
        id<MTLBuffer> contributions = [g_dev newBufferWithLength:contribution_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> prefix = [g_dev newBufferWithLength:prefix_bytes
                                                   options:MTLResourceStorageModeShared];
        if (!contributions || !prefix) {
            [contributions release];
            [prefix release];
            return false;
        }
        [context.sparse_oracle_outputs release];
        [context.sparse_oracle_prefix release];
        context.sparse_oracle_outputs = contributions;
        context.sparse_oracle_prefix = prefix;
        context.sparse_oracle_output_bytes = contribution_bytes;
    }
    if (!context.sparse_oracle_prefix || context.sparse_oracle_prefix.length < prefix_bytes) return false;
    if (!context.sparse_block_ids ||
        context.sparse_block_ids.length < static_cast<size_t>(proxy.selected_blocks) * sizeof(uint32_t)) {
        id<MTLBuffer> ids = [g_dev newBufferWithLength:static_cast<size_t>(proxy.selected_blocks) * sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];
        if (!ids) return false;
        [context.sparse_block_ids release];
        context.sparse_block_ids = ids;
    }
    if (!context.sparse_oracle_pairs) {
        std::vector<std::array<uint32_t, 2>> pairs;
        for (uint32_t first = 0; first + run_blocks <= proxy.output_blocks; ++first) {
            for (uint32_t second = first + run_blocks;
                 second + run_blocks <= proxy.output_blocks; ++second) {
                pairs.push_back({first, second});
            }
        }
        if (pairs.empty() || pairs.size() > UINT32_MAX) return false;
        id<MTLBuffer> pair_buffer = [g_dev newBufferWithBytes:pairs.data()
                                                       length:pairs.size() * sizeof(pairs[0])
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> score_buffer = [g_dev newBufferWithLength:pairs.size() * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        if (!pair_buffer || !score_buffer) {
            [pair_buffer release];
            [score_buffer release];
            return false;
        }
        context.sparse_oracle_pairs = pair_buffer;
        context.sparse_oracle_pair_scores = score_buffer;
        context.sparse_oracle_pair_count = static_cast<uint32_t>(pairs.size());
    }
    if (!context.sparse_oracle_pair_scores || context.sparse_oracle_pair_count == 0) return false;
    if (context.sparse_oracle_runs.size() != proxy.output_blocks) {
        for (id<MTLBuffer> run : context.sparse_oracle_runs) [run release];
        context.sparse_oracle_runs.clear();
        context.sparse_oracle_runs.reserve(proxy.output_blocks);
        for (uint32_t block = 0; block != proxy.output_blocks; ++block) {
            const uint32_t run_data[2] = {block, 1};
            id<MTLBuffer> run = [g_dev newBufferWithBytes:run_data length:sizeof(run_data)
                                               options:MTLResourceStorageModeShared];
            if (!run) return false;
            context.sparse_oracle_runs.push_back(run);
        }
    }
    for (uint32_t block = 0; block != proxy.output_blocks; ++block) {
        if (!tok_bind_sparse_columns_to(
                down, g_tok.ws, g_tok.fh + static_cast<size_t>(block) * 256 * sizeof(float),
                context.sparse_oracle_outputs, static_cast<size_t>(block) * hidden * sizeof(float),
                context.sparse_oracle_runs[block], 1, full_intermediate, hidden)) return false;
    }
    [g_tok.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    id<MTLComputePipelineState> prefix_pipeline = named_pipe("sparse_prefix_block_contributions");
    id<MTLComputePipelineState> score_pipeline = named_pipe("sparse_downstream_pair_scores");
    id<MTLComputePipelineState> select_pipeline = named_pipe("sparse_select_best_pair");
    if (!prefix_pipeline || prefix_pipeline.threadExecutionWidth == 0 ||
        prefix_pipeline.maxTotalThreadsPerThreadgroup < 64 ||
        !score_pipeline || score_pipeline.threadExecutionWidth != 32 ||
        score_pipeline.maxTotalThreadsPerThreadgroup < 64 ||
        !select_pipeline || select_pipeline.threadExecutionWidth != 32 ||
        select_pipeline.maxTotalThreadsPerThreadgroup < 32) return false;
    const uint32_t hidden_u32 = static_cast<uint32_t>(hidden);
    [g_tok.enc setComputePipelineState:prefix_pipeline];
    [g_tok.enc setBuffer:context.sparse_oracle_outputs offset:0 atIndex:0];
    [g_tok.enc setBuffer:context.sparse_oracle_prefix offset:0 atIndex:1];
    [g_tok.enc setBytes:&proxy.output_blocks length:sizeof(proxy.output_blocks) atIndex:2];
    [g_tok.enc setBytes:&hidden_u32 length:sizeof(hidden_u32) atIndex:3];
    [g_tok.enc dispatchThreadgroups:MTLSizeMake((hidden_u32 + 63) / 64, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [g_tok.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [g_tok.enc setComputePipelineState:score_pipeline];
    [g_tok.enc setBuffer:context.sparse_oracle_prefix offset:0 atIndex:0];
    [g_tok.enc setBuffer:context.sparse_oracle_pairs offset:0 atIndex:1];
    [g_tok.enc setBuffer:context.sparse_oracle_pair_scores offset:0 atIndex:2];
    [g_tok.enc setBytes:&proxy.output_blocks length:sizeof(proxy.output_blocks) atIndex:3];
    [g_tok.enc setBytes:&run_blocks length:sizeof(run_blocks) atIndex:4];
    [g_tok.enc setBytes:&hidden_u32 length:sizeof(hidden_u32) atIndex:5];
    [g_tok.enc dispatchThreadgroups:MTLSizeMake(context.sparse_oracle_pair_count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [g_tok.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [g_tok.enc setComputePipelineState:select_pipeline];
    [g_tok.enc setBuffer:context.sparse_oracle_pair_scores offset:0 atIndex:0];
    [g_tok.enc setBuffer:context.sparse_oracle_pairs offset:0 atIndex:1];
    [g_tok.enc setBuffer:context.sparse_block_ids offset:0 atIndex:2];
    [g_tok.enc setBytes:&context.sparse_oracle_pair_count length:sizeof(context.sparse_oracle_pair_count) atIndex:3];
    [g_tok.enc setBytes:&run_blocks length:sizeof(run_blocks) atIndex:4];
    [g_tok.enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [g_tok.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    context.sparse_affine = false;
    return tok_record_sparse_pair(slot, run_blocks) &&
           tok_bind_sparse_rows(gate, input_offset, g_tok.fg, hidden, packed_intermediate) &&
           tok_bind_sparse_rows(up, input_offset, g_tok.fu, hidden, packed_intermediate);
}
#endif

static bool tok_bind_sparse_rows(const Tensor& w, size_t xoff, size_t yoff,
                                 int K, int packed_N, uint32_t block_offset) {
    MetalTokContext& context = active_tok_context();
    if (packed_N <= 0 || packed_N % 256 != 0) return false;
    const uint32_t selected_blocks = static_cast<uint32_t>(packed_N / 256);
    if (!tok_enc() || !context.sparse_block_ids || selected_blocks == 0 ||
        static_cast<uint64_t>(block_offset) + selected_blocks >
            context.sparse_block_ids.length / sizeof(uint32_t) ||
        selected_blocks > static_cast<uint32_t>(INT_MAX / 256) ||
        w.n_dims != 2 || w.dims[0] != static_cast<uint64_t>(K) ||
        w.dims[1] > INT_MAX || (w.type != GGMLType::Q4_K && w.type != GGMLType::Q6_K)) return false;
    const int full_N = static_cast<int>(w.dims[1]);
    const size_t bpb = bytes_per_block(w.type);
    if (K <= 0 || K % 256 != 0 || bpb == 0) return false;
    const uint64_t rb = static_cast<uint64_t>(K / 256) * bpb;
    if (static_cast<uint64_t>(full_N) > SIZE_MAX / rb ||
        !strict_tensor_data_span(w, static_cast<uint64_t>(full_N) * rb)) return false;
    size_t weight_offset = 0;
    id<MTLBuffer> weights = get_weight_buf(w.data, static_cast<size_t>(full_N * rb), weight_offset);
    id<MTLComputePipelineState> pipeline = get_sparse_pipe(
        "gemv_sparse_rows", static_cast<int>(w.type), context.sparse_affine,
        g_sparse_row_pipes);
    if (!weights || !pipeline || pipeline.threadExecutionWidth != 32 ||
        pipeline.maxTotalThreadsPerThreadgroup < 64) return false;
    [g_tok.enc setComputePipelineState:pipeline];
    [g_tok.enc setBuffer:weights offset:weight_offset atIndex:0];
    [g_tok.enc setBuffer:g_tok.ws offset:xoff atIndex:1];
    [g_tok.enc setBuffer:g_tok.ws offset:yoff atIndex:2];
    [g_tok.enc setBytes:&K length:sizeof(K) atIndex:3];
    [g_tok.enc setBytes:&packed_N length:sizeof(packed_N) atIndex:4];
    [g_tok.enc setBytes:&rb length:sizeof(rb) atIndex:5];
    [g_tok.enc setBuffer:context.sparse_block_ids
                  offset:static_cast<size_t>(block_offset) * sizeof(uint32_t) atIndex:6];
    [g_tok.enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((packed_N + 3) / 4), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    tok_record_projection_bytes(w, K, packed_N);
    tok_record_projection_dispatch(w.type);
    return true;
}

#if defined(LAPLACE_METAL_TESTING)
static bool tok_bind_sparse_columns_to(const Tensor& w, id<MTLBuffer> input, size_t input_offset,
                                       id<MTLBuffer> output, size_t output_offset,
                                       id<MTLBuffer> block_ids, uint32_t block_count,
                                       int full_K, int N) {
    if (!tok_enc() || !input || !output || !block_ids || block_count == 0 ||
        block_count > static_cast<uint32_t>(INT_MAX / 256) || w.n_dims != 2 ||
        w.dims[0] != static_cast<uint64_t>(full_K) || w.dims[1] != static_cast<uint64_t>(N) ||
        (w.type != GGMLType::Q4_K && w.type != GGMLType::Q6_K)) return false;
    const int packed_K = static_cast<int>(block_count * 256u);
    const size_t bpb = bytes_per_block(w.type);
    if (full_K <= 0 || full_K % 256 != 0 || N <= 0 || bpb == 0) return false;
    const uint64_t rb = static_cast<uint64_t>(full_K / 256) * bpb;
    if (static_cast<uint64_t>(N) > SIZE_MAX / rb ||
        !strict_tensor_data_span(w, static_cast<uint64_t>(N) * rb)) return false;
    size_t weight_offset = 0;
    id<MTLBuffer> weights = get_weight_buf(w.data, static_cast<size_t>(N * rb), weight_offset);
    id<MTLComputePipelineState> pipeline = get_sparse_pipe(
        "gemv_sparse_columns", static_cast<int>(w.type), true, g_sparse_column_pipes);
    if (!weights || !pipeline || pipeline.threadExecutionWidth != 32 ||
        pipeline.maxTotalThreadsPerThreadgroup < 64) return false;
    [g_tok.enc setComputePipelineState:pipeline];
    [g_tok.enc setBuffer:weights offset:weight_offset atIndex:0];
    [g_tok.enc setBuffer:input offset:input_offset atIndex:1];
    [g_tok.enc setBuffer:output offset:output_offset atIndex:2];
    [g_tok.enc setBytes:&full_K length:sizeof(full_K) atIndex:3];
    [g_tok.enc setBytes:&packed_K length:sizeof(packed_K) atIndex:4];
    [g_tok.enc setBytes:&N length:sizeof(N) atIndex:5];
    [g_tok.enc setBytes:&rb length:sizeof(rb) atIndex:6];
    [g_tok.enc setBuffer:block_ids offset:0 atIndex:7];
    [g_tok.enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((N + 3) / 4), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    return true;
}
#endif

static bool tok_bind_sparse_columns(const Tensor& w, size_t xoff, size_t yoff,
                                    int full_K, int packed_K, int N,
                                    uint32_t block_offset = 0) {
    MetalTokContext& context = active_tok_context();
    if (packed_K <= 0 || packed_K % 256 != 0) return false;
    const uint32_t selected_blocks = static_cast<uint32_t>(packed_K / 256);
    if (!tok_enc() || !context.sparse_block_ids || selected_blocks == 0 ||
        static_cast<uint64_t>(block_offset) + selected_blocks >
            context.sparse_block_ids.length / sizeof(uint32_t) ||
        selected_blocks > static_cast<uint32_t>(INT_MAX / 256) ||
        w.n_dims != 2 || w.dims[0] != static_cast<uint64_t>(full_K) ||
        w.dims[1] != static_cast<uint64_t>(N) ||
        (w.type != GGMLType::Q4_K && w.type != GGMLType::Q6_K)) return false;
    const size_t bpb = bytes_per_block(w.type);
    if (full_K <= 0 || full_K % 256 != 0 || packed_K <= 0 || N <= 0 || bpb == 0) return false;
    const uint64_t rb = static_cast<uint64_t>(full_K / 256) * bpb;
    if (static_cast<uint64_t>(N) > SIZE_MAX / rb ||
        !strict_tensor_data_span(w, static_cast<uint64_t>(N) * rb)) return false;
    size_t weight_offset = 0;
    id<MTLBuffer> weights = get_weight_buf(w.data, static_cast<size_t>(N * rb), weight_offset);
    id<MTLComputePipelineState> pipeline = get_sparse_pipe(
        "gemv_sparse_columns", static_cast<int>(w.type), context.sparse_affine,
        g_sparse_column_pipes);
    if (!weights || !pipeline || pipeline.threadExecutionWidth != 32 ||
        pipeline.maxTotalThreadsPerThreadgroup < 64) return false;
    [g_tok.enc setComputePipelineState:pipeline];
    [g_tok.enc setBuffer:weights offset:weight_offset atIndex:0];
    [g_tok.enc setBuffer:g_tok.ws offset:xoff atIndex:1];
    [g_tok.enc setBuffer:g_tok.ws offset:yoff atIndex:2];
    [g_tok.enc setBytes:&full_K length:sizeof(full_K) atIndex:3];
    [g_tok.enc setBytes:&packed_K length:sizeof(packed_K) atIndex:4];
    [g_tok.enc setBytes:&N length:sizeof(N) atIndex:5];
    [g_tok.enc setBytes:&rb length:sizeof(rb) atIndex:6];
    [g_tok.enc setBuffer:context.sparse_block_ids
                  offset:static_cast<size_t>(block_offset) * sizeof(uint32_t) atIndex:7];
    [g_tok.enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((N + 3) / 4), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    const uint64_t physical = static_cast<uint64_t>(N) * selected_blocks * bpb;
    context.metrics.requested_projection_source_bytes += physical;
    tok_record_projection_dispatch(w.type);
    return true;
}

static bool tok_rms(const Tensor& w, size_t src, size_t dst, int n, float eps) {
    if (!tok_enc() || !w.data || n < 1 || !std::isfinite(eps) || eps < 0.0f) return false;
    size_t woff = 0;
    id<MTLBuffer> wb = get_weight_buf(w.data, (size_t)n * 4, woff);
    if (!wb) return false;
    auto p = named_pipe("rmsnorm_f32");
    if (!p) return false;
    [g_tok.enc setComputePipelineState:p];
    [g_tok.enc setBuffer:g_tok.ws offset:src atIndex:0];
    [g_tok.enc setBuffer:wb offset:woff atIndex:1];
    [g_tok.enc setBuffer:g_tok.ws offset:dst atIndex:2];
    [g_tok.enc setBytes:&n length:4 atIndex:3];
    [g_tok.enc setBytes:&eps length:4 atIndex:4];
    [g_tok.enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return true;
}

static bool tok_add(size_t a, size_t b, size_t o, int n) {
    if (!tok_enc()) return false;
    auto p = named_pipe("vec_add");
    if (!p) return false;
    [g_tok.enc setComputePipelineState:p];
    [g_tok.enc setBuffer:g_tok.ws offset:a atIndex:0];
    [g_tok.enc setBuffer:g_tok.ws offset:b atIndex:1];
    [g_tok.enc setBuffer:g_tok.ws offset:o atIndex:2];
    [g_tok.enc setBytes:&n length:4 atIndex:3];
    enc_1d(g_tok.enc, p, n);
    return true;
}

static bool tok_add_bias(const Tensor& bias, size_t value, int n) {
    if (!tok_enc() || !bias.data || bias.type != GGMLType::F32) return false;
    size_t offset = 0;
    id<MTLBuffer> buffer = get_weight_buf(bias.data, static_cast<size_t>(n) * sizeof(float), offset);
    auto pipeline = named_pipe("vec_add");
    if (!buffer || !pipeline) return false;
    [g_tok.enc setComputePipelineState:pipeline];
    [g_tok.enc setBuffer:g_tok.ws offset:value atIndex:0];
    [g_tok.enc setBuffer:buffer offset:offset atIndex:1];
    [g_tok.enc setBuffer:g_tok.ws offset:value atIndex:2];
    [g_tok.enc setBytes:&n length:4 atIndex:3];
    enc_1d(g_tok.enc, pipeline, n);
    return true;
}

static bool tok_attention_cache_binding(const MetalTokLayer& layer, int width,
                                        uint64_t& base, int& stride) {
    if (!g_tok.kcache || !g_tok.vcache || width < 1 || g_tok.max_seq < 1) return false;
    if (layer.cache_width_offset != UINT64_MAX) {
        if (g_tok.kv_width_sum == 0 || layer.cache_width_offset > g_tok.kv_width_sum ||
            static_cast<uint64_t>(width) > g_tok.kv_width_sum - layer.cache_width_offset ||
            layer.cache_width_offset > UINT64_MAX / static_cast<uint64_t>(g_tok.max_seq)) {
            return false;
        }
        base = layer.cache_width_offset * static_cast<uint64_t>(g_tok.max_seq);
        stride = width;
    } else {
        if (layer.cache_id < 0 || layer.cache_id >= g_tok.n_layers || g_tok.kv_stride < width)
            return false;
        const uint64_t row_width = static_cast<uint64_t>(g_tok.max_seq) *
                                   static_cast<uint64_t>(g_tok.kv_stride);
        if (static_cast<uint64_t>(layer.cache_id) > UINT64_MAX / row_width) return false;
        base = static_cast<uint64_t>(layer.cache_id) * row_width;
        stride = g_tok.kv_stride;
    }
    const uint64_t tail = static_cast<uint64_t>(g_tok.max_seq - 1) *
                          static_cast<uint64_t>(stride);
    if (base > UINT64_MAX - tail) return false;
    const uint64_t last = base + tail;
    return last <= g_tok.kv_elements &&
           static_cast<uint64_t>(width) <= g_tok.kv_elements - last;
}

bool metal_tok_layer(const MetalTokLayer& L) {
    if (!g_tok.live) return false;
    const int H = L.H, Hq = L.Hq, Hk = L.Hk, Dh = L.Dh;
    if (Hq < 1 || Hk < 1 || Dh < 1 || Hq > INT_MAX / Dh || Hk > INT_MAX / Dh)
        return false;
    const int qn = Hq * Dh, kn = Hk * Dh;
    uint64_t cache_base = 0;
    int cache_stride = 0;
#define TOKFAIL(s) do { fprintf(stderr, "[metal] tok fail %s Hq=%d Hk=%d Dh=%d inter=%d\n", \
    s, Hq, Hk, Dh, L.inter); return false; } while (0)
    const float moe_router_normalization_scale =
        std::bit_cast<float>(L.moe_router_normalization_scale_bits);
    const int rope_frequency_dimension = L.rope_frequency_dimension == 0
        ? L.rope_dim : L.rope_frequency_dimension;
    if (L.n_used > 0 &&
        (L.key_state_alias || !std::isfinite(moe_router_normalization_scale) ||
         moe_router_normalization_scale <= 0.0f)) {
        TOKFAIL("moe_semantic_contract");
    }
    if (!L.attn_norm || !L.attn_q || !L.attn_k || !L.attn_o ||
        qn > g_tok.query_capacity || kn > g_tok.kv_stride ||
        !tok_attention_cache_binding(L, kn, cache_base, cache_stride) ||
        !std::isfinite(L.rms_eps) || L.rms_eps < 0.0f ||
        !std::isfinite(L.rope_base) || L.rope_base <= 0.0f ||
        !std::isfinite(L.attention_scale) || L.attention_scale <= 0.0f ||
        L.rope_dim < 1 || L.rope_dim > Dh || (L.rope_dim % 2) != 0 ||
        rope_frequency_dimension < L.rope_dim || (rope_frequency_dimension % 2) != 0 ||
        ((L.q_norm == nullptr) != (L.k_norm == nullptr)) ||
        ((L.q_norm != nullptr) && (!std::isfinite(L.q_norm_eps) || !std::isfinite(L.k_norm_eps) ||
                                   L.q_norm_eps < 0.0f || L.k_norm_eps < 0.0f)) ||
        (L.query_gate_split && (qn > INT_MAX / 2 || 2 * qn > g_tok.inter))) TOKFAIL("missing attn");
    if (!tok_rms(*L.attn_norm, g_tok.x, g_tok.xn, H, L.rms_eps)) TOKFAIL("attn_norm");
    tok_barrier();
    if (!tok_bind(*L.attn_q, g_tok.xn, L.query_gate_split ? g_tok.fg : g_tok.q, H,
                  L.query_gate_split ? 2 * qn : qn)) TOKFAIL("attn_q");
    if (!tok_bind(*L.attn_k, g_tok.xn, g_tok.k, H, kn)) TOKFAIL("attn_k");
    if (!L.key_state_alias && !L.attn_v) TOKFAIL("attn_v_missing");
    if (L.key_state_alias && L.attn_v) TOKFAIL("attn_v_alias_binding");
    if (L.key_state_alias) {
        if (!tok_enc()) TOKFAIL("enc_copy");
        auto cp = named_pipe("vec_copy");
        if (!cp) TOKFAIL("vec_copy");
        [g_tok.enc setComputePipelineState:cp];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.k atIndex:0];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.v atIndex:1];
        [g_tok.enc setBytes:&kn length:4 atIndex:2];
        enc_1d(g_tok.enc, cp, kn);
    } else {
        if (!tok_bind(*L.attn_v, g_tok.xn, g_tok.v, H, kn)) TOKFAIL("attn_v");
    }
    if (L.attn_q_bias && !tok_add_bias(*L.attn_q_bias, L.query_gate_split ? g_tok.fg : g_tok.q,
                                        L.query_gate_split ? 2 * qn : qn)) TOKFAIL("attn_q_bias");
    if (L.attn_k_bias && !tok_add_bias(*L.attn_k_bias, g_tok.k, kn)) TOKFAIL("attn_k_bias");
    if (L.attn_v_bias && !tok_add_bias(*L.attn_v_bias, g_tok.v, kn)) TOKFAIL("attn_v_bias");
    if (!tok_profile_mark(TokProfileSegment::Qkv)) TOKFAIL("qkv_profile");
    tok_barrier();
    if (L.query_gate_split) {
        auto split = named_pipe("axis_split_f32");
        if (!split || !tok_enc()) TOKFAIL("axis_split");
        [g_tok.enc setComputePipelineState:split];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.fg atIndex:0];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.q atIndex:1];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.fu atIndex:2];
        [g_tok.enc setBytes:&Hq length:sizeof(Hq) atIndex:3];
        [g_tok.enc setBytes:&Dh length:sizeof(Dh) atIndex:4];
        [g_tok.enc setBytes:&Dh length:sizeof(Dh) atIndex:5];
        enc_1d(g_tok.enc, split, 2 * qn);
        tok_barrier();
    }
    if (!tok_enc()) TOKFAIL("enc_qknorm");
    if (L.q_norm && L.k_norm) {
        auto pr = named_pipe("rmsnorm_rows_f32");
        if (!pr) TOKFAIL("rmsnorm_rows");
        size_t qwo = 0, kwo = 0;
        id<MTLBuffer> qw = get_weight_buf(L.q_norm->data, (size_t)Dh * 4, qwo);
        id<MTLBuffer> kw = get_weight_buf(L.k_norm->data, (size_t)Dh * 4, kwo);
        if (!qw || !kw) TOKFAIL("qknorm_w");
        [g_tok.enc setComputePipelineState:pr];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.q atIndex:0];
        [g_tok.enc setBuffer:qw offset:qwo atIndex:1];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.q atIndex:2];
        [g_tok.enc setBytes:&Dh length:4 atIndex:3];
        [g_tok.enc setBytes:&Hq length:4 atIndex:4];
        [g_tok.enc setBytes:&L.q_norm_eps length:4 atIndex:5];
        [g_tok.enc dispatchThreadgroups:MTLSizeMake(Hq, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.k atIndex:0];
        [g_tok.enc setBuffer:kw offset:kwo atIndex:1];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.k atIndex:2];
        [g_tok.enc setBytes:&Hk length:4 atIndex:4];
        [g_tok.enc setBytes:&L.k_norm_eps length:4 atIndex:5];
        [g_tok.enc dispatchThreadgroups:MTLSizeMake(Hk, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    tok_barrier();
    if (!tok_enc()) TOKFAIL("enc_rope");
    auto rp = named_pipe(L.rope_multi_section ? "rope_multisection_f32" :
                         L.rope_interleaved ? "rope_interleaved_f32" : "rope_f32");
    if (!rp) TOKFAIL("rope");
    int pairs = L.rope_dim / 2;
    const int frequency_dimension = rope_frequency_dimension;
    if (pairs < 1 || frequency_dimension < L.rope_dim || (frequency_dimension % 2) != 0) {
        TOKFAIL("rope_frequency_dimension");
    }
    int use_pf = (L.is_global && L.rope_freqs && L.n_rope_freqs >= pairs) ? 1 : 0;
    id<MTLBuffer> pfb = g_tok.ws;
    size_t pfo = g_tok.tmp;
    if (use_pf) {
        size_t dummy = 0;
        pfb = get_weight_buf(L.rope_freqs, (size_t)pairs * 4, dummy);
        pfo = dummy;
        if (!pfb) TOKFAIL("rope_pf");
    }
    [g_tok.enc setComputePipelineState:rp];
    [g_tok.enc setBuffer:g_tok.ws offset:g_tok.q atIndex:0];
    [g_tok.enc setBytes:&Hq length:4 atIndex:1];
    [g_tok.enc setBytes:&Dh length:4 atIndex:2];
    [g_tok.enc setBytes:&pairs length:4 atIndex:3];
    [g_tok.enc setBytes:&L.rope_base length:4 atIndex:4];
    const std::array<int, 4> rope_positions = {g_tok.pos, g_tok.pos, g_tok.pos, g_tok.pos};
    if (L.rope_multi_section) {
        [g_tok.enc setBytes:rope_positions.data() length:sizeof(rope_positions) atIndex:5];
        [g_tok.enc setBytes:L.rope_sections length:sizeof(L.rope_sections) atIndex:6];
        [g_tok.enc setBytes:&use_pf length:4 atIndex:7];
        [g_tok.enc setBuffer:pfb offset:pfo atIndex:8];
        [g_tok.enc setBytes:&frequency_dimension length:4 atIndex:9];
    } else {
        [g_tok.enc setBytes:&g_tok.pos length:4 atIndex:5];
        [g_tok.enc setBytes:&use_pf length:4 atIndex:6];
        [g_tok.enc setBuffer:pfb offset:pfo atIndex:7];
        [g_tok.enc setBytes:&frequency_dimension length:4 atIndex:8];
    }
    enc_1d(g_tok.enc, rp, Hq * pairs);
    [g_tok.enc setBuffer:g_tok.ws offset:g_tok.k atIndex:0];
    [g_tok.enc setBytes:&Hk length:4 atIndex:1];
    enc_1d(g_tok.enc, rp, Hk * pairs);
    tok_barrier();
    if (!tok_enc()) TOKFAIL("enc_kv");
    if (L.owns_kv) {
        auto kw = named_pipe("kv_write");
        if (!kw) TOKFAIL("kv_write");
        [g_tok.enc setComputePipelineState:kw];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.k atIndex:0];
        [g_tok.enc setBuffer:g_tok.kcache offset:0 atIndex:1];
        [g_tok.enc setBytes:&g_tok.pos length:4 atIndex:2];
        [g_tok.enc setBytes:&g_tok.max_seq length:4 atIndex:3];
        [g_tok.enc setBytes:&cache_base length:sizeof(cache_base) atIndex:4];
        [g_tok.enc setBytes:&kn length:4 atIndex:5];
        [g_tok.enc setBytes:&cache_stride length:4 atIndex:6];
        enc_1d(g_tok.enc, kw, kn);
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.v atIndex:0];
        [g_tok.enc setBuffer:g_tok.vcache offset:0 atIndex:1];
        enc_1d(g_tok.enc, kw, kn);
    }
    tok_barrier();
    if (!tok_enc()) TOKFAIL("enc_attn");
    if (!std::isfinite(L.attention_scale) || L.attention_scale <= 0.0f) TOKFAIL("attention_scale");
    const float scale = L.attention_scale;
    auto ap = named_pipe("attn_decode");
    if (!ap) TOKFAIL("attn");
    [g_tok.enc setComputePipelineState:ap];
    [g_tok.enc setBuffer:g_tok.ws offset:g_tok.q atIndex:0];
    [g_tok.enc setBuffer:g_tok.kcache offset:0 atIndex:1];
    [g_tok.enc setBuffer:g_tok.vcache offset:0 atIndex:2];
    [g_tok.enc setBuffer:g_tok.ws offset:g_tok.ao atIndex:3];
    [g_tok.enc setBytes:&Hq length:4 atIndex:4];
    [g_tok.enc setBytes:&Hk length:4 atIndex:5];
    [g_tok.enc setBytes:&Dh length:4 atIndex:6];
    [g_tok.enc setBytes:&g_tok.pos length:4 atIndex:7];
    [g_tok.enc setBytes:&L.window length:4 atIndex:8];
    [g_tok.enc setBytes:&g_tok.max_seq length:4 atIndex:9];
    [g_tok.enc setBytes:&cache_base length:sizeof(cache_base) atIndex:10];
    [g_tok.enc setBytes:&scale length:4 atIndex:11];
    [g_tok.enc setBytes:&cache_stride length:4 atIndex:12];
    [g_tok.enc dispatchThreadgroups:MTLSizeMake((NSUInteger)Hq, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    tok_barrier();
    if (L.query_gate_split) {
        auto gated = named_pipe("gated_attention_f32");
        if (!gated || !tok_enc()) TOKFAIL("gated_attention");
        [g_tok.enc setComputePipelineState:gated];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.ao atIndex:0];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.fu atIndex:1];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.ao atIndex:2];
        [g_tok.enc setBytes:&qn length:sizeof(qn) atIndex:3];
        enc_1d(g_tok.enc, gated, qn);
        tok_barrier();
    }
    if (!tok_bind(*L.attn_o, g_tok.ao, g_tok.xb, qn, H)) TOKFAIL("attn_o");
    tok_barrier();
    if (L.post_attn_norm &&
        !tok_rms(*L.post_attn_norm, g_tok.xb, g_tok.xb, H, L.rms_eps))
        TOKFAIL("post_attn_norm");
    tok_barrier();
    if (!tok_add(g_tok.x, g_tok.xb, g_tok.x, H)) TOKFAIL("post_attn_add");
    if (!tok_profile_mark(TokProfileSegment::Attention)) TOKFAIL("attention_profile");
    tok_seg("attn");
    tok_barrier();
    if (!tok_rms(*L.ffn_norm, g_tok.x, g_tok.xn, H, L.rms_eps)) TOKFAIL("ffn_norm");
    if (!tok_accumulate_importance(L.ffn_input_importance_slot, g_tok.xn,
                                   static_cast<uint32_t>(H))) TOKFAIL("ffn_input_importance");
    tok_barrier();
    if (L.sparse_ffn) {
#if defined(LAPLACE_METAL_TESTING)
        if (L.sparse_ffn_dense_oracle) {
            if (L.sparse_ffn_proxy_slot == UINT32_MAX ||
                !tok_select_sparse_dense_oracle(*L.ffn_gate, *L.ffn_up, *L.ffn_down, g_tok.xn,
                                                L.sparse_ffn_proxy_slot, H,
                                                L.sparse_ffn_full_intermediate, L.inter))
                TOKFAIL("sparse_ffn_dense_oracle");
        } else
#endif
        {
            if ((L.sparse_ffn_proxy_slot != UINT32_MAX &&
                 !tok_select_sparse_proxy(g_tok.xn, L.sparse_ffn_proxy_slot)) ||
                L.sparse_ffn_full_intermediate < L.inter ||
                !tok_bind_sparse_rows(*L.ffn_gate, g_tok.xn, g_tok.fg, H, L.inter,
                                      L.sparse_ffn_block_offset))
                TOKFAIL("sparse_ffn_gate");
            if (!tok_bind_sparse_rows(*L.ffn_up, g_tok.xn, g_tok.fu, H, L.inter,
                                      L.sparse_ffn_block_offset))
                TOKFAIL("sparse_ffn_up");
        }
    } else {
        if (!tok_bind(*L.ffn_gate, g_tok.xn, g_tok.fg, H, L.inter)) TOKFAIL("ffn_gate");
        if (!tok_bind(*L.ffn_up, g_tok.xn, g_tok.fu, H, L.inter)) TOKFAIL("ffn_up");
    }
    const int n_used = L.n_used;
    const int gu_n = L.exp_inter * 2;
    if (n_used > 0 && L.moe_gate && L.moe_up && L.moe_dn) {
        if (!L.pre_ffw_2 || !L.moe_gate_scale || !L.moe_reduce_left_to_right)
            TOKFAIL("moe_semantic_bindings");
        auto pns = named_pipe("rmsnorm_noscale");
        if (!pns || !tok_enc()) TOKFAIL("moe_noscale");
        [g_tok.enc setComputePipelineState:pns];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.x atIndex:0];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.tmp atIndex:1];
        [g_tok.enc setBytes:&H length:4 atIndex:2];
        [g_tok.enc setBytes:&L.rms_eps length:4 atIndex:3];
        [g_tok.enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        auto pm = named_pipe("vec_mul");
        size_t swo = 0;
        id<MTLBuffer> sw = get_weight_buf(L.moe_gate_scale->data, (size_t)H * 4, swo);
        if (!pm || !sw) TOKFAIL("moe_scale_w");
        [g_tok.enc setComputePipelineState:pm];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.tmp atIndex:0];
        [g_tok.enc setBuffer:sw offset:swo atIndex:1];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.tmp atIndex:2];
        [g_tok.enc setBytes:&H length:4 atIndex:3];
        enc_1d(g_tok.enc, pm, H);
        auto ps = named_pipe("vec_scale");
        if (!ps) TOKFAIL("vec_scale");
        [g_tok.enc setComputePipelineState:ps];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.tmp atIndex:0];
        [g_tok.enc setBytes:&moe_router_normalization_scale length:4 atIndex:1];
        [g_tok.enc setBytes:&H length:4 atIndex:2];
        enc_1d(g_tok.enc, ps, H);
        tok_barrier();
        if (!tok_bind(*L.moe_gate, g_tok.tmp, g_tok.route, H, L.n_experts))
            TOKFAIL("moe_gate");
        tok_barrier();
        if (!tok_enc()) TOKFAIL("enc_topk");
        auto tk = named_pipe("router_topk");
        if (!tk) TOKFAIL("topk");
        [g_tok.enc setComputePipelineState:tk];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.route atIndex:0];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.ids atIndex:1];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.rw atIndex:2];
        [g_tok.enc setBytes:&L.n_experts length:4 atIndex:3];
        [g_tok.enc setBytes:&n_used length:4 atIndex:4];
        [g_tok.enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (L.moe_dn_scale && L.moe_dn_scale->data) {
            auto ads = named_pipe("apply_down_scale");
            size_t dso = 0;
            id<MTLBuffer> ds = get_weight_buf(L.moe_dn_scale->data,
                                              (size_t)L.n_experts * 4, dso);
            if (!ads || !ds) TOKFAIL("down_scale");
            [g_tok.enc setComputePipelineState:ads];
            [g_tok.enc setBuffer:g_tok.ws offset:g_tok.ids atIndex:0];
            [g_tok.enc setBuffer:g_tok.ws offset:g_tok.rw atIndex:1];
            [g_tok.enc setBuffer:ds offset:dso atIndex:2];
            [g_tok.enc setBytes:&n_used length:4 atIndex:3];
            [g_tok.enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
               threadsPerThreadgroup:MTLSizeMake((NSUInteger)std::max(n_used, 1), 1, 1)];
        }
        tok_barrier();
        if (!tok_rms(*L.pre_ffw_2, g_tok.x, g_tok.tmp, H, L.rms_eps)) TOKFAIL("pre_ffw_2");
        tok_barrier();
        if (!tok_enc()) TOKFAIL("enc_moe_up");
        if (!bind_q4k_id_dev(g_tok.enc, *L.moe_up, H, gu_n, g_tok.ws, g_tok.ids,
                             n_used, g_tok.ws, g_tok.tmp, g_tok.ws, g_tok.mgu, 0))
            TOKFAIL("moe_up_id");
    }
    tok_split();
    if (!tok_enc()) TOKFAIL("enc_glu");
    int sw = L.swiglu ? 1 : 0;
    const int moe_sw = L.moe_gelu_tanh ? 0 : sw;
    if (!enqueue_glu(g_tok.enc, g_tok.ws, g_tok.fg, g_tok.fu, g_tok.fh, L.inter, sw))
        TOKFAIL("glu");
    if (!tok_accumulate_importance(L.ffn_down_importance_slot, g_tok.fh,
                                   static_cast<uint32_t>(L.inter)))
        TOKFAIL("ffn_down_importance");
    tok_seg("dense_ffn");
    if (n_used > 0 &&
        !enqueue_moe_glu(g_tok.enc, g_tok.ws, g_tok.mgu, g_tok.mh,
                         n_used, L.exp_inter, moe_sw))
        TOKFAIL("glu_exp");
    tok_barrier();
    if (L.sparse_ffn) {
        if (!tok_bind_sparse_columns(*L.ffn_down, g_tok.fh, g_tok.xb,
                                     L.sparse_ffn_full_intermediate, L.inter, H,
                                     L.sparse_ffn_block_offset))
            TOKFAIL("sparse_ffn_down");
    } else if (!tok_bind(*L.ffn_down, g_tok.fh, g_tok.xb, L.inter, H)) TOKFAIL("ffn_down");
    if (n_used > 0) {
        if (!L.moe_dn) TOKFAIL("moe_dn");
        if (L.moe_dn->type == GGMLType::Q4_K) {
            if (!bind_q4k_id_dev(g_tok.enc, *L.moe_dn, L.exp_inter, H, g_tok.ws,
                                 g_tok.ids, n_used, g_tok.ws, g_tok.mh,
                                 g_tok.ws, g_tok.md, 1))
                TOKFAIL("moe_dn_q4");
        } else if (L.moe_dn->type == GGMLType::Q8_0) {
            if (!bind_q8_id_dev(g_tok.enc, *L.moe_dn, L.exp_inter, H, g_tok.ws,
                                g_tok.ids, n_used, g_tok.ws, g_tok.mh,
                                g_tok.ws, g_tok.md, 1))
                TOKFAIL("moe_dn_q8");
        } else if (!bind_gemv_id(g_tok.enc, *L.moe_dn, L.exp_inter, H, g_tok.ws,
                                 g_tok.ids, n_used, g_tok.ws, g_tok.mh,
                                 g_tok.ws, g_tok.md, 1)) {
            TOKFAIL("moe_dn_id");
        }
    }
    if (n_used > 0) {
        tok_barrier();
        if (!tok_enc()) TOKFAIL("enc_combine");
        auto mc = named_pipe("moe_combine");
        if (!mc) TOKFAIL("combine");
        [g_tok.enc setComputePipelineState:mc];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.md atIndex:0];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.moe atIndex:1];
        [g_tok.enc setBytes:&H length:4 atIndex:2];
        [g_tok.enc setBytes:&n_used length:4 atIndex:3];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.rw atIndex:4];
        enc_1d(g_tok.enc, mc, H);
        tok_barrier();
        if (L.post_ffw_1 && !tok_rms(*L.post_ffw_1, g_tok.xb, g_tok.xb, H, L.rms_eps))
            TOKFAIL("post_ffw_1");
        if (L.post_ffw_2 && !tok_rms(*L.post_ffw_2, g_tok.moe, g_tok.moe, H, L.rms_eps))
            TOKFAIL("post_ffw_2");
        tok_barrier();
        if (!tok_add(g_tok.xb, g_tok.moe, g_tok.xb, H)) TOKFAIL("moe_add");
    }
    tok_seg("moe");
    tok_barrier();
    if (L.post_ffw && !tok_rms(*L.post_ffw, g_tok.xb, g_tok.xb, H, L.rms_eps))
        TOKFAIL("post_ffw");
    tok_barrier();
    if (!tok_add(g_tok.x, g_tok.xb, g_tok.x, H)) TOKFAIL("res_add");
    tok_barrier();
    if (L.out_scale && L.out_scale->data) {
        float s = reinterpret_cast<const float*>(L.out_scale->data)[0];
        auto ps = named_pipe("vec_scale");
        if (!ps || !tok_enc()) return false;
        [g_tok.enc setComputePipelineState:ps];
        [g_tok.enc setBuffer:g_tok.ws offset:g_tok.x atIndex:0];
        [g_tok.enc setBytes:&s length:4 atIndex:1];
        [g_tok.enc setBytes:&H length:4 atIndex:2];
        enc_1d(g_tok.enc, ps, H);
    }
    if (!tok_profile_mark(TokProfileSegment::Ffn)) TOKFAIL("ffn_profile");
#undef TOKFAIL
    tok_maybe_commit();
    return true;
}

static size_t tok_batch_row(size_t base, int width, uint32_t row) {
    return base + static_cast<size_t>(row) * static_cast<size_t>(width) * sizeof(float);
}

static bool tok_prefill_f16_layer_supported(const MetalTokLayer& L) {
    const auto f16 = [](const Tensor* tensor) {
        return tensor && tensor->data && tensor->type == GGMLType::F16 && tensor->n_dims == 2;
    };
    const auto f32 = [](const Tensor* tensor) {
        return tensor && tensor->data && tensor->type == GGMLType::F32;
    };
    const int frequency_dimension = L.rope_frequency_dimension == 0
        ? L.rope_dim : L.rope_frequency_dimension;
    return g_tok.live && g_tok.batch_rows == 2 && L.H == g_tok.H &&
           L.Hq > 0 && L.Hk > 0 && L.Dh > 0 && L.Hq <= INT_MAX / L.Dh &&
           L.Hk <= INT_MAX / L.Dh && L.Hq * L.Dh <= g_tok.query_capacity &&
           L.Hk * L.Dh <= g_tok.kv_stride && L.inter == g_tok.inter &&
           L.owns_kv && L.n_used == 0 && L.n_experts == 0 && L.exp_inter == 0 &&
           L.swiglu && !L.query_gate_split && !L.rope_interleaved && !L.rope_multi_section &&
           !L.rope_freqs && !L.attn_q_bias && !L.attn_k_bias && !L.attn_v_bias &&
           !L.q_norm && !L.k_norm && !L.post_attn_norm && !L.sparse_ffn &&
           !L.moe_gate && !L.moe_gate_scale && !L.moe_up && !L.moe_dn && !L.moe_dn_scale &&
           !L.pre_ffw_2 && !L.post_ffw_1 && !L.post_ffw_2 && !L.post_ffw && !L.out_scale &&
           L.rope_dim > 0 && L.rope_dim <= L.Dh && (L.rope_dim % 2) == 0 &&
           frequency_dimension >= L.rope_dim && (frequency_dimension % 2) == 0 &&
           std::isfinite(L.rope_base) && L.rope_base > 0.0f &&
           std::isfinite(L.attention_scale) && L.attention_scale > 0.0f && std::isfinite(L.rms_eps) &&
           L.rms_eps >= 0.0f && f32(L.attn_norm) && f32(L.ffn_norm) && f16(L.attn_q) &&
           f16(L.attn_k) && f16(L.attn_v) && f16(L.attn_o) && f16(L.ffn_gate) &&
           f16(L.ffn_up) && f16(L.ffn_down);
}

static bool tok_prefill_f16_rope_and_kv(const MetalTokLayer& L, uint32_t row, int position) {
    if (!tok_enc()) return false;
    const int qn = L.Hq * L.Dh;
    const int kn = L.Hk * L.Dh;
    uint64_t cache_base = 0;
    int cache_stride = 0;
    if (!tok_attention_cache_binding(L, kn, cache_base, cache_stride)) return false;
    const int pairs = L.rope_dim / 2;
    const int frequency_dimension = L.rope_frequency_dimension == 0
        ? L.rope_dim : L.rope_frequency_dimension;
    const size_t q = tok_batch_row(g_tok.q, qn, row);
    const size_t k = tok_batch_row(g_tok.k, kn, row);
    const size_t v = tok_batch_row(g_tok.v, kn, row);
    id<MTLComputePipelineState> rope = named_pipe("rope_f32");
    id<MTLComputePipelineState> write = named_pipe("kv_write");
    if (!rope || !write) return false;
    const int no_position_factors = 0;
    [g_tok.enc setComputePipelineState:rope];
    [g_tok.enc setBuffer:g_tok.ws offset:q atIndex:0];
    [g_tok.enc setBytes:&L.Hq length:sizeof(L.Hq) atIndex:1];
    [g_tok.enc setBytes:&L.Dh length:sizeof(L.Dh) atIndex:2];
    [g_tok.enc setBytes:&pairs length:sizeof(pairs) atIndex:3];
    [g_tok.enc setBytes:&L.rope_base length:sizeof(L.rope_base) atIndex:4];
    [g_tok.enc setBytes:&position length:sizeof(position) atIndex:5];
    [g_tok.enc setBytes:&no_position_factors length:sizeof(no_position_factors) atIndex:6];
    [g_tok.enc setBuffer:g_tok.ws offset:g_tok.tmp atIndex:7];
    [g_tok.enc setBytes:&frequency_dimension length:sizeof(frequency_dimension) atIndex:8];
    enc_1d(g_tok.enc, rope, L.Hq * pairs);
    [g_tok.enc setBuffer:g_tok.ws offset:k atIndex:0];
    [g_tok.enc setBytes:&L.Hk length:sizeof(L.Hk) atIndex:1];
    enc_1d(g_tok.enc, rope, L.Hk * pairs);
    tok_barrier();
    [g_tok.enc setComputePipelineState:write];
    [g_tok.enc setBuffer:g_tok.ws offset:k atIndex:0];
    [g_tok.enc setBuffer:g_tok.kcache offset:0 atIndex:1];
    [g_tok.enc setBytes:&position length:sizeof(position) atIndex:2];
    [g_tok.enc setBytes:&g_tok.max_seq length:sizeof(g_tok.max_seq) atIndex:3];
    [g_tok.enc setBytes:&cache_base length:sizeof(cache_base) atIndex:4];
    [g_tok.enc setBytes:&kn length:sizeof(kn) atIndex:5];
    [g_tok.enc setBytes:&cache_stride length:sizeof(cache_stride) atIndex:6];
    enc_1d(g_tok.enc, write, kn);
    [g_tok.enc setBuffer:g_tok.ws offset:v atIndex:0];
    [g_tok.enc setBuffer:g_tok.vcache offset:0 atIndex:1];
    enc_1d(g_tok.enc, write, kn);
    return true;
}

static bool tok_prefill_f16_attention(const MetalTokLayer& L, uint32_t row, int position) {
    if (!tok_enc()) return false;
    const int qn = L.Hq * L.Dh;
    const int kn = L.Hk * L.Dh;
    uint64_t cache_base = 0;
    int cache_stride = 0;
    if (!tok_attention_cache_binding(L, kn, cache_base, cache_stride)) return false;
    const size_t q = tok_batch_row(g_tok.q, qn, row);
    const size_t ao = tok_batch_row(g_tok.ao, qn, row);
    id<MTLComputePipelineState> attention = named_pipe("attn_decode");
    if (!attention) return false;
    if (!std::isfinite(L.attention_scale) || L.attention_scale <= 0.0f) return false;
    const float scale = L.attention_scale;
    [g_tok.enc setComputePipelineState:attention];
    [g_tok.enc setBuffer:g_tok.ws offset:q atIndex:0];
    [g_tok.enc setBuffer:g_tok.kcache offset:0 atIndex:1];
    [g_tok.enc setBuffer:g_tok.vcache offset:0 atIndex:2];
    [g_tok.enc setBuffer:g_tok.ws offset:ao atIndex:3];
    [g_tok.enc setBytes:&L.Hq length:sizeof(L.Hq) atIndex:4];
    [g_tok.enc setBytes:&L.Hk length:sizeof(L.Hk) atIndex:5];
    [g_tok.enc setBytes:&L.Dh length:sizeof(L.Dh) atIndex:6];
    [g_tok.enc setBytes:&position length:sizeof(position) atIndex:7];
    [g_tok.enc setBytes:&L.window length:sizeof(L.window) atIndex:8];
    [g_tok.enc setBytes:&g_tok.max_seq length:sizeof(g_tok.max_seq) atIndex:9];
    [g_tok.enc setBytes:&cache_base length:sizeof(cache_base) atIndex:10];
    [g_tok.enc setBytes:&scale length:sizeof(scale) atIndex:11];
    [g_tok.enc setBytes:&cache_stride length:sizeof(cache_stride) atIndex:12];
    [g_tok.enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(L.Hq), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return true;
}

static bool metal_tok_dense_prefill_batch_layer(const MetalTokLayer& L) {
    if (!tok_prefill_f16_layer_supported(L)) return false;
    const int H = L.H;
    const int qn = L.Hq * L.Dh;
    const int kn = L.Hk * L.Dh;
    const uint32_t rows = 2;
    for (uint32_t row = 0; row != rows; ++row)
        if (!tok_rms(*L.attn_norm, tok_batch_row(g_tok.x, H, row),
                     tok_batch_row(g_tok.xn, H, row), H, L.rms_eps)) return false;
    tok_barrier();
    if (!tok_bind_f16_rows(*L.attn_q, g_tok.xn, g_tok.q, rows, H, qn) ||
        !tok_bind_f16_rows(*L.attn_k, g_tok.xn, g_tok.k, rows, H, kn) ||
        !tok_bind_f16_rows(*L.attn_v, g_tok.xn, g_tok.v, rows, H, kn) ||
        !tok_profile_mark(TokProfileSegment::Qkv)) return false;
    tok_barrier();
    for (uint32_t row = 0; row != rows; ++row)
        if (!tok_prefill_f16_rope_and_kv(L, row, g_tok.pos + static_cast<int>(row))) return false;
    tok_barrier();
    for (uint32_t row = 0; row != rows; ++row)
        if (!tok_prefill_f16_attention(L, row, g_tok.pos + static_cast<int>(row))) return false;
    tok_barrier();
    if (!tok_bind_f16_rows(*L.attn_o, g_tok.ao, g_tok.xb, rows, qn, H)) return false;
    tok_barrier();
    for (uint32_t row = 0; row != rows; ++row) {
        if (!tok_add(tok_batch_row(g_tok.x, H, row), tok_batch_row(g_tok.xb, H, row),
                     tok_batch_row(g_tok.x, H, row), H) ||
            !tok_rms(*L.ffn_norm, tok_batch_row(g_tok.x, H, row),
                     tok_batch_row(g_tok.xn, H, row), H, L.rms_eps)) return false;
    }
    if (!tok_profile_mark(TokProfileSegment::Attention)) return false;
    tok_barrier();
    if (!tok_bind_f16_rows(*L.ffn_gate, g_tok.xn, g_tok.fg, rows, H, L.inter) ||
        !tok_bind_f16_rows(*L.ffn_up, g_tok.xn, g_tok.fu, rows, H, L.inter)) return false;
    tok_barrier();
    for (uint32_t row = 0; row != rows; ++row)
        if (!tok_enc() || !enqueue_glu(g_tok.enc, g_tok.ws,
                                       tok_batch_row(g_tok.fg, L.inter, row),
                                       tok_batch_row(g_tok.fu, L.inter, row),
                                       tok_batch_row(g_tok.fh, L.inter, row), L.inter, 1)) return false;
    tok_barrier();
    if (!tok_bind_f16_rows(*L.ffn_down, g_tok.fh, g_tok.xb, rows, L.inter, H)) return false;
    tok_barrier();
    for (uint32_t row = 0; row != rows; ++row)
        if (!tok_add(tok_batch_row(g_tok.x, H, row), tok_batch_row(g_tok.xb, H, row),
                     tok_batch_row(g_tok.x, H, row), H)) return false;
    return tok_profile_mark(TokProfileSegment::Ffn);
}

bool metal_tok_end(float* x, int H) {
    if (!g_tok.live) return false;
    tok_split();
    if (g_tok.cmd) {
        [g_tok.cmd commit];
        g_tok_last = g_tok.cmd;
        g_tok.cmd = nil;
    }
    if (g_tok_last) {
        const auto wait_start = std::chrono::steady_clock::now();
        [g_tok_last waitUntilCompleted];
        record_completed_command(g_tok_last, wait_start, std::chrono::steady_clock::now());
        MTLCommandBufferStatus st = g_tok_last.status;
        g_tok_last = nil;
        if (st == MTLCommandBufferStatusError) {
            g_tok = TokWS{};
            return false;
        }
    }
    if (x) memcpy(x, (uint8_t*)[g_tok.ws contents] + g_tok.x, (size_t)H * 4);
    g_tok.live = false;
    g_tok.cmd = nil;
    g_tok.enc = nil;
    return true;
}

void metal_tok_abort() {
    if (g_tok.enc) [g_tok.enc endEncoding];
    // Do not commit a partially encoded command buffer. The CPU KV cache was
    // never updated on this path, and forcing a seed on the next begin keeps
    // GPU cache bytes from the abandoned token unreachable.
    g_tok.enc = nil;
    g_tok.cmd = nil;
    g_tok.live = false;
    g_kv_seeded_to = -1;
}

bool metal_tok_lm(const Tensor& w, float* logits, int H, int vocab) {
    return metal_tok_final(Tensor{}, w, logits, H, vocab, 0.f);
}

static bool tok_finish_column_grouped_affine_u2_skip() {
    MetalTokContext& context = active_tok_context();
    if (context.column_grouped_affine_u2_selected_bytes) {
        auto* selected = static_cast<uint64_t*>(
            context.column_grouped_affine_u2_selected_bytes.contents);
        if (UINT64_MAX - context.metrics.requested_projection_source_bytes < *selected) {
            context.failure_detail = "column-grouped UInt2 byte accounting overflow";
            return false;
        }
        context.metrics.requested_projection_source_bytes += *selected;
        *selected = 0u;
    }
    if (context.column_grouped_affine_u2_numerical_error &&
        *static_cast<const uint32_t*>(
            context.column_grouped_affine_u2_numerical_error.contents) != 0u) {
        context.failure_detail = "column-grouped UInt2 selector rejected a non-finite activation";
        return false;
    }
    return true;
}

static bool metal_tok_final_impl(const Tensor& norm, const Tensor& lm, float* logits,
                                 const MetalSamplerDescriptor* sampler,
                                 MetalSamplerResult* sampled,
                                 int H, int vocab, float eps) {
    const bool copy_logits = logits != nullptr;
    const bool sample_logits = sampler != nullptr && sampled != nullptr;
    if (!g_tok.live || copy_logits == sample_logits || vocab < 1 || !lm.data ||
        (sample_logits && !metal_sampler_descriptor_valid(*sampler))) return false;
    if (norm.data && !tok_rms(norm, g_tok.x, g_tok.xn, H, eps)) return false;
    tok_barrier();
    if (!tok_enc()) return false;
    size_t need = (size_t)vocab * 4;
    if (!g_logits_buf || g_logits_bytes < need) {
        g_logits_buf = [g_dev newBufferWithLength:need
                                          options:MTLResourceStorageModeShared];
        g_logits_bytes = need;
    }
    if (!g_logits_buf) return false;
    if (!bind_q6k(g_tok.enc, lm, H, vocab, g_tok.ws, g_tok.xn, g_logits_buf, 0) &&
        !bind_any(g_tok.enc, lm, H, vocab, g_tok.ws, g_tok.xn, g_logits_buf, 0)) {
        fprintf(stderr, "[metal] tok_final: LM bind failed type=%d N=%d\n",
                (int)lm.type, vocab);
        return false;
    }
    tok_record_projection_bytes(lm, H, vocab);
    tok_record_projection_dispatch(lm.type);
    if (sample_logits) {
        id<MTLComputePipelineState> pipeline = get_sampler_pipe();
        if (!pipeline) return false;
        if (!active_tok_context().sampler_result) {
            active_tok_context().sampler_result =
                [g_dev newBufferWithLength:sizeof(MetalSamplerResult)
                                   options:MTLResourceStorageModeShared];
        }
        const NSUInteger threads = std::min<NSUInteger>(
            256u, pipeline.maxTotalThreadsPerThreadgroup);
        if (!active_tok_context().sampler_result || threads == 0) return false;
        tok_barrier();
        [g_tok.enc setComputePipelineState:pipeline];
        [g_tok.enc setBuffer:g_logits_buf offset:0 atIndex:0];
        [g_tok.enc setBuffer:active_tok_context().sampler_result offset:0 atIndex:1];
        const uint32_t vocabulary = static_cast<uint32_t>(vocab);
        [g_tok.enc setBytes:&vocabulary length:sizeof(vocabulary) atIndex:2];
        [g_tok.enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }
    if (!tok_profile_mark(TokProfileSegment::Final)) return false;
    tok_split();
    if (!g_tok.cmd) return false;
    const auto wait_start = std::chrono::steady_clock::now();
    [g_tok.cmd commit];
    [g_tok.cmd waitUntilCompleted];
    const auto wait_end = std::chrono::steady_clock::now();
    if (g_tok.cmd.status != MTLCommandBufferStatusCompleted) {
        active_tok_context().failure_detail = command_buffer_failure_detail(g_tok.cmd);
        return false;
    }
    record_completed_command(g_tok.cmd, wait_start, wait_end);
    if (!tok_finish_column_grouped_affine_u2_skip()) return false;
    MetalTokMetrics& metrics = active_tok_context().metrics;
    metrics.cpu_wait_ms = std::chrono::duration<double, std::milli>(wait_end - wait_start).count();
    const CFTimeInterval gpu_start = [g_tok.cmd GPUStartTime];
    const CFTimeInterval gpu_end = [g_tok.cmd GPUEndTime];
    metrics.gpu_time_ms = gpu_end >= gpu_start ? 1000.0 * (gpu_end - gpu_start) : 0.0;
    uint64_t proxy_bytes = active_tok_context().sparse_scores ?
        [active_tok_context().sparse_scores length] : 0;
#if defined(LAPLACE_METAL_TESTING)
    if (active_tok_context().sparse_oracle_starts)
        proxy_bytes += [active_tok_context().sparse_oracle_starts length];
    if (active_tok_context().sparse_oracle_outputs)
        proxy_bytes += [active_tok_context().sparse_oracle_outputs length];
    if (active_tok_context().sparse_oracle_prefix)
        proxy_bytes += [active_tok_context().sparse_oracle_prefix length];
    if (active_tok_context().sparse_oracle_pairs)
        proxy_bytes += [active_tok_context().sparse_oracle_pairs length];
    if (active_tok_context().sparse_oracle_pair_scores)
        proxy_bytes += [active_tok_context().sparse_oracle_pair_scores length];
    for (id<MTLBuffer> run : active_tok_context().sparse_oracle_runs)
        proxy_bytes += [run length];
#endif
    for (const SparseFfnProxyResource& proxy : active_tok_context().sparse_proxies)
        if (proxy.coefficients) proxy_bytes += [proxy.coefficients length];
    for (const ActivationImportanceResource& resource : active_tok_context().importance)
        if (resource.values) proxy_bytes += [resource.values length];
    metrics.peak_session_bytes = active_tok_context().weights->byte_count() + g_tok.bytes +
                                 2 * g_kv_bytes + g_logits_bytes +
                                 (active_tok_context().sampler_result
                                      ? [active_tok_context().sampler_result length] : 0) +
                                 (active_tok_context().sparse_block_ids ?
                                      [active_tok_context().sparse_block_ids length] : 0) + proxy_bytes;
    metrics.kv_cache_bytes = 2 * static_cast<uint64_t>(g_kv_bytes);
    tok_profile_finish();
    if (copy_logits) {
        memcpy(logits, [g_logits_buf contents], need);
    } else {
        const MetalSamplerResult result =
            *static_cast<const MetalSamplerResult*>(
                [active_tok_context().sampler_result contents]);
        if (result.status != MetalSamplerResultStatus::Success ||
            result.token_id >= static_cast<uint32_t>(vocab) ||
            !std::isfinite(result.logit) || result.reserved != 0) {
            active_tok_context().failure_detail =
                result.status == MetalSamplerResultStatus::NonFiniteLogit
                    ? "canonical Metal sampler rejected a non-finite logit"
                    : "canonical Metal sampler returned an invalid result";
            return false;
        }
        *sampled = result;
    }
    g_tok.live = false;
    g_tok.cmd = nil;
    g_tok.enc = nil;
    static std::once_flag once;
    std::call_once(once, []{ fprintf(stderr, "[metal] fused_lm=1\n"); });
    return true;
}

bool metal_tok_final(const Tensor& norm, const Tensor& lm, float* logits,
                     int H, int vocab, float eps) {
    return metal_tok_final_impl(norm, lm, logits, nullptr, nullptr, H, vocab, eps);
}

struct RecurrentTokenState {
    id<MTLBuffer> history_current = nil;
    id<MTLBuffer> history_candidate = nil;
    id<MTLBuffer> state_current = nil;
    id<MTLBuffer> state_candidate = nil;
    id<MTLBuffer> qkv = nil;
    id<MTLBuffer> conv_weight = nil;
    id<MTLBuffer> gate = nil;
    id<MTLBuffer> beta = nil;
    id<MTLBuffer> alpha = nil;
    id<MTLBuffer> dt_bias = nil;
    id<MTLBuffer> decay = nil;
    id<MTLBuffer> norm = nil;
    id<MTLBuffer> output = nil;
    size_t history_bytes = 0;
    size_t state_bytes = 0;
    size_t qkv_bytes = 0;
    size_t output_bytes = 0;
    int qk_heads = 0;
    int value_heads = 0;
    int head_dimension = 0;
    int kernel = 0;
    bool seeded = false;
    bool pending = false;
#if defined(LAPLACE_TESTING)
    bool fail_after_completed_submission = false;
#endif

    ~RecurrentTokenState() {
        for (id<MTLBuffer> buffer : {history_current, history_candidate, state_current, state_candidate,
                                     qkv, conv_weight, gate, beta, alpha, dt_bias, decay, norm, output}) {
            [buffer release];
        }
    }
};

class MetalTokSession {
public:
    MetalTokSession() {
        token_.weights = &weights_;
        token_.single_command_buffer = true;
        token_.queue = [g_dev newCommandQueue];
    }

    ~MetalTokSession() = default;

private:
    friend std::shared_ptr<MetalTokSession> metal_tok_session_create();
    friend void metal_tok_session_enable_error_diagnostics(MetalTokSession&, bool);
    friend const char* metal_tok_session_last_failure(const MetalTokSession&);
    friend void metal_tok_session_require_registered_weights(MetalTokSession&, bool);
    friend uint32_t metal_tok_session_weight_span_coverage(const MetalTokSession&, const void*, size_t);
    friend MetalResourceSnapshot metal_tok_session_resource_snapshot(const MetalTokSession&);
    friend bool metal_tok_session_dense_ready(MetalTokSession&);
    friend bool metal_tok_session_affine_u2_256_ready(MetalTokSession&);
    friend bool metal_tok_session_column_grouped_affine_u2_skip_256_ready(MetalTokSession&);
    friend bool metal_tok_session_register_column_grouped_affine_u2_skip_256(
        MetalTokSession&, const Tensor&);
    friend MetalTokMoeCapabilities metal_tok_session_moe_capabilities(MetalTokSession&);
    friend bool metal_tok_session_moe_ready(MetalTokSession&);
    friend bool metal_tok_session_recurrent_ready(MetalTokSession&);
    friend bool metal_tok_session_register_weights(MetalTokSession&, const void*, size_t);
#if defined(LAPLACE_TESTING)
    friend bool metal_tok_session_register_weights_for_testing(MetalTokSession&, const void*, size_t,
                                                               size_t, uint32_t);
#endif
    friend void metal_tok_session_unregister_weights(MetalTokSession&, const void*);
    friend bool metal_tok_session_set_sparse_ffn_runs(MetalTokSession&, const MetalSparseBlockRun*, uint32_t);
    friend bool metal_tok_session_set_sparse_ffn_layer_ids(MetalTokSession&, const uint32_t*,
                                                            const uint32_t*, const uint32_t*,
                                                            uint32_t);
    friend bool metal_tok_session_set_sparse_ffn_proxy(MetalTokSession&, uint32_t, const float*,
                                                       uint32_t, uint32_t, uint32_t);
    friend bool metal_tok_session_set_importance_slots(MetalTokSession&, const uint32_t*, uint32_t);
    friend bool metal_tok_session_read_importance(const MetalTokSession&, uint32_t, float*, uint32_t,
                                                  uint32_t*);
#if defined(LAPLACE_METAL_TESTING)
    friend bool metal_tok_session_enable_sparse_ffn_dense_oracle_for_testing(MetalTokSession&, uint32_t);
    friend bool metal_tok_session_sparse_ffn_windows_for_testing(const MetalTokSession&, uint32_t*, uint32_t);
#endif
    friend bool metal_tok_session_begin(MetalTokSession&, int, int, int, int, int, int, int, int, int, int, int, uint32_t);
    friend bool metal_tok_session_begin_with_attention_capacity(
        MetalTokSession&, int, int, int, int, int, MetalTokAttentionCapacity,
        int, int, int, uint32_t);
    friend bool metal_tok_session_begin_prefill_batch(MetalTokSession&, int, int, int, int, int,
                                                      int, int, int, int, int, int, uint32_t);
    friend bool metal_tok_session_begin_prefill_batch_with_attention_capacity(
        MetalTokSession&, int, int, int, int, int, MetalTokAttentionCapacity,
        int, int, int, uint32_t);
    friend bool metal_tok_session_begin_continuing(MetalTokSession&, int, int, int, int, int, int, int, int, int, int, int);
    friend bool metal_tok_session_begin_continuing_with_attention_capacity(
        MetalTokSession&, int, int, int, int, int, MetalTokAttentionCapacity,
        int, int, int);
    friend bool metal_tok_session_upload_embedding(MetalTokSession&, const Tensor&, uint32_t, int, int, float);
    friend bool metal_tok_session_upload_embeddings_batch(MetalTokSession&, const Tensor&, const uint32_t*,
                                                          uint32_t, int, int, float);
    friend bool metal_tok_session_upload_x(MetalTokSession&, const float*, int);
    friend bool metal_tok_session_layer(MetalTokSession&, const MetalTokLayer&);
    friend bool metal_tok_session_dense_prefill_batch_layer(MetalTokSession&, const MetalTokLayer&, uint32_t);
    friend bool metal_tok_session_select_prefill_batch_row(MetalTokSession&, uint32_t);
    friend bool metal_tok_session_recurrent_layer(MetalTokSession&, const MetalTokRecurrentLayer&);
    friend bool metal_tok_session_recurrent_commit(MetalTokSession&);
    friend bool metal_tok_session_commit_token(MetalTokSession&);
    friend bool metal_tok_session_seal_token(MetalTokSession&);
    friend bool metal_tok_session_final(MetalTokSession&, const Tensor&, const Tensor&, float*, int, int, float);
    friend bool metal_tok_session_final_sampled(MetalTokSession&, const Tensor&, const Tensor&,
                                                 const MetalSamplerDescriptor&, MetalSamplerResult*,
                                                 int, int, float);
    friend MetalTokMetrics metal_tok_session_metrics(const MetalTokSession&);
    friend void metal_tok_session_abort(MetalTokSession&);
    friend bool metal_tok_session_recurrent_seed(MetalTokSession&, const float*, const float*, int, int, int, int);
    friend bool metal_tok_session_recurrent_step(MetalTokSession&, const float*, const float*, const float*, const float*,
                                                 const float*, const float*, const float*, const float*, float*,
                                                 int, int, int, int, float, float);
    friend bool metal_tok_session_recurrent_snapshot(const MetalTokSession&, float*, float*, int, int, int, int);
    friend bool metal_tok_session_recurrent_snapshot_slot(const MetalTokSession&, uint32_t, float*, float*, int, int, int, int);
#if defined(LAPLACE_METAL_TESTING)
    friend void metal_tok_session_fail_after_completed_submission_for_testing(MetalTokSession&);
#endif
#if defined(LAPLACE_TESTING)
    friend void metal_tok_session_recurrent_fail_after_completed_submission_for_testing(MetalTokSession&);
    friend uintptr_t metal_tok_session_queue_identity_for_testing(const MetalTokSession&);
    friend bool metal_tok_session_download_x_for_testing(const MetalTokSession&, float*, int);
    friend bool metal_test_sparse_ffn_original_spans(const float*, const Tensor&, const Tensor&,
                                                     const Tensor&, const MetalSparseBlockRun*,
                                                     uint32_t, float*, int, int);
    friend bool metal_test_select_contiguous_window(const float*, uint32_t, uint32_t,
                                                    uint32_t*, uint32_t*, double*);
    friend bool metal_test_sparse_ffn_selected_window(const float*, const Tensor&, const Tensor&,
                                                      const Tensor&, const float*, uint32_t, uint32_t,
                                                      float*, uint32_t*, int, int, double*);
    friend bool metal_test_sparse_ffn_proxy_window(const float*, const Tensor&, const Tensor&,
                                                   const Tensor&, const float*, uint32_t, uint32_t,
                                                   uint32_t, float*, uint32_t*, int, int, double*);
    friend bool metal_test_activation_importance_accumulator(const float*, const float*, uint32_t,
                                                             float*, uint32_t*);
#endif
#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
    friend bool metal_tok_session_probe_ffn_for_testing(MetalTokSession&, const float*,
                                                        const Tensor&, const Tensor&, const Tensor&,
                                                        const MetalSparseBlockRun*, uint32_t,
                                                        float*, int, int, double*);
#endif

    RecurrentTokenState* recurrent_slot(uint32_t slot) {
        if (slot >= kSemanticModelMaximumLayers) return nullptr;
        if (recurrent_.size() <= slot) recurrent_.resize(static_cast<size_t>(slot) + 1);
        if (!recurrent_[slot]) recurrent_[slot] = std::make_unique<RecurrentTokenState>();
        return recurrent_[slot].get();
    }

    const RecurrentTokenState* recurrent_slot(uint32_t slot) const {
        return slot < recurrent_.size() ? recurrent_[slot].get() : nullptr;
    }

    void mark_recurrent_pending(uint32_t slot) {
        if (std::find(recurrent_pending_.begin(), recurrent_pending_.end(), slot) == recurrent_pending_.end())
            recurrent_pending_.push_back(slot);
    }

    void publish_recurrent() {
        for (uint32_t slot : recurrent_pending_) {
            RecurrentTokenState* recurrent = recurrent_slot(slot);
            if (!recurrent) continue;
            std::swap(recurrent->history_current, recurrent->history_candidate);
            std::swap(recurrent->state_current, recurrent->state_candidate);
            recurrent->pending = false;
        }
        recurrent_pending_.clear();
    }

    void discard_recurrent() {
        for (uint32_t slot : recurrent_pending_) {
            if (RecurrentTokenState* recurrent = recurrent_slot(slot)) recurrent->pending = false;
        }
        recurrent_pending_.clear();
    }

    MetalWeightContext weights_;
    MetalTokContext token_;
    std::vector<std::unique_ptr<RecurrentTokenState>> recurrent_;
    std::vector<uint32_t> recurrent_pending_;
};

std::shared_ptr<MetalTokSession> metal_tok_session_create() {
    init();
    if (!g_dev || !g_lib) return {};
    auto session = std::make_shared<MetalTokSession>();
    return session->token_.queue ? std::move(session) : std::shared_ptr<MetalTokSession>{};
}

#if defined(LAPLACE_TESTING)
uintptr_t metal_tok_session_queue_identity_for_testing(const MetalTokSession& session) {
    return reinterpret_cast<uintptr_t>(session.token_.queue);
}
#endif

void metal_tok_session_enable_error_diagnostics(MetalTokSession& session, bool enabled) {
    session.token_.detailed_errors = enabled;
}

const char* metal_tok_session_last_failure(const MetalTokSession& session) {
    return session.token_.failure_detail.empty() ? nullptr : session.token_.failure_detail.c_str();
}

void metal_tok_session_require_registered_weights(MetalTokSession& session, bool required) {
    session.weights_.require_registered_weights = required;
}

uint32_t metal_tok_session_weight_span_coverage(const MetalTokSession& session,
                                                const void* base, size_t size) {
    MetalTokScope scope(const_cast<MetalTokContext&>(session.token_));
    std::lock_guard<std::mutex> lock(session.weights_.mutex);
    if (!base || size == 0) return 0;
    const size_t count = mmap_buf_coverage_count_locked(base, size);
    return count > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(count);
}

MetalResourceSnapshot metal_tok_session_resource_snapshot(const MetalTokSession& session) {
    MetalTokScope scope(const_cast<MetalTokContext&>(session.token_));
    std::lock_guard<std::mutex> lock(session.weights_.mutex);
    MetalResourceSnapshot snapshot;
    snapshot.current_allocated_size = g_dev.currentAllocatedSize;
    snapshot.recommended_max_working_set_size = g_dev.recommendedMaxWorkingSetSize;
    snapshot.registered_weight_bytes = session.weights_.byte_count();
    snapshot.implicit_weight_copies = session.weights_.implicit_copy_count;
    for (const ColumnGroupedAffineU2SkipResource& resource :
         session.token_.column_grouped_affine_u2_skip)
        if (resource.metadata) snapshot.session_owned_metadata_bytes += resource.metadata.length;
    for (id<MTLBuffer> buffer : {session.token_.column_grouped_affine_u2_active_count,
                                 session.token_.column_grouped_affine_u2_active_columns,
                                 session.token_.column_grouped_affine_u2_partial,
                                 session.token_.column_grouped_affine_u2_numerical_error,
                                 session.token_.column_grouped_affine_u2_selected_bytes})
        if (buffer) snapshot.transient_workspace_bytes += buffer.length;
    return snapshot;
}

bool metal_tok_session_dense_ready(MetalTokSession& session) {
    MetalTokScope scope(session.token_);
    init();
    auto suitable = [](id<MTLComputePipelineState> pipeline) {
        return pipeline && pipeline.threadExecutionWidth != 0 && pipeline.maxTotalThreadsPerThreadgroup >= 64;
    };
    const bool sparse_proxy_ready = session.token_.sparse_proxies.empty() ||
        (named_pipe("sparse_proxy_block_scores") &&
         named_pipe("sparse_proxy_block_scores").threadExecutionWidth == 32 &&
         named_pipe("sparse_proxy_block_scores").maxTotalThreadsPerThreadgroup >= 32 &&
         named_pipe("sparse_select_contiguous_window") &&
         named_pipe("sparse_select_contiguous_window").threadExecutionWidth == 32 &&
         named_pipe("sparse_select_contiguous_window").maxTotalThreadsPerThreadgroup >= 32);
#if defined(LAPLACE_METAL_TESTING)
    const bool sparse_oracle_ready = !session.token_.sparse_oracle_starts ||
        (named_pipe("sparse_prefix_block_contributions") &&
         named_pipe("sparse_prefix_block_contributions").threadExecutionWidth != 0 &&
         named_pipe("sparse_prefix_block_contributions").maxTotalThreadsPerThreadgroup >= 64 &&
         named_pipe("sparse_downstream_pair_scores") &&
         named_pipe("sparse_downstream_pair_scores").threadExecutionWidth == 32 &&
         named_pipe("sparse_downstream_pair_scores").maxTotalThreadsPerThreadgroup >= 64 &&
         named_pipe("sparse_select_best_pair") &&
         named_pipe("sparse_select_best_pair").threadExecutionWidth == 32 &&
         named_pipe("sparse_select_best_pair").maxTotalThreadsPerThreadgroup >= 32 &&
         named_pipe("sparse_record_window") &&
         named_pipe("sparse_record_window").threadExecutionWidth != 0 &&
         named_pipe("sparse_record_window").maxTotalThreadsPerThreadgroup >=
             named_pipe("sparse_record_window").threadExecutionWidth &&
         named_pipe("sparse_record_pair") &&
         named_pipe("sparse_record_pair").threadExecutionWidth != 0 &&
         named_pipe("sparse_record_pair").maxTotalThreadsPerThreadgroup >=
             named_pipe("sparse_record_pair").threadExecutionWidth);
#else
    const bool sparse_oracle_ready = true;
#endif
    return g_dev && session.token_.queue && g_lib && sparse_proxy_ready && sparse_oracle_ready &&
           suitable(get_iq2_xxs_pipe()) &&
           suitable(get_pipe(static_cast<int>(GGMLType::F16))) && suitable(get_glu_pipe()) &&
           suitable(named_pipe("embedding_f16")) && suitable(named_pipe("embedding_q4k")) &&
           suitable(named_pipe("embedding_q6k")) &&
           suitable(named_pipe("rmsnorm_f32")) &&
           suitable(named_pipe("vec_add")) && suitable(named_pipe("rope_f32")) &&
           suitable(named_pipe("rope_interleaved_f32")) &&
           suitable(named_pipe("rope_multisection_f32")) &&
           suitable(named_pipe("axis_split_f32")) && suitable(named_pipe("gated_attention_f32")) &&
           suitable(named_pipe("kv_write")) && suitable(named_pipe("attn_decode"));
}

bool metal_tok_session_affine_u2_256_ready(MetalTokSession& session) {
    MetalTokScope scope(session.token_);
    init();
    const id<MTLComputePipelineState> pipeline = get_affine_u2_256_pipe();
    return g_dev && session.token_.queue && g_lib && pipeline &&
           pipeline.threadExecutionWidth == 32 &&
           pipeline.maxTotalThreadsPerThreadgroup >= 64;
}

bool metal_tok_session_column_grouped_affine_u2_skip_256_ready(
    MetalTokSession& session) {
    MetalTokScope scope(session.token_);
    init();
    const id<MTLComputePipelineState> selector =
        named_pipe("column_grouped_affine_u2_skip_256_select");
    const id<MTLComputePipelineState> partial =
        named_pipe("column_grouped_affine_u2_skip_256_partial");
    const id<MTLComputePipelineState> reduce =
        named_pipe("column_grouped_affine_u2_skip_256_reduce");
    return g_dev && session.token_.queue && g_lib && selector && partial && reduce &&
           selector.threadExecutionWidth != 0 &&
           selector.maxTotalThreadsPerThreadgroup >= 256 &&
           partial.threadExecutionWidth == 32 &&
           partial.maxTotalThreadsPerThreadgroup >= 64 &&
           reduce.threadExecutionWidth != 0 &&
           reduce.maxTotalThreadsPerThreadgroup >= 256;
}

bool metal_tok_session_register_column_grouped_affine_u2_skip_256(
    MetalTokSession& session, const Tensor& tensor) {
    MetalTokScope scope(session.token_);
    session.token_.failure_detail.clear();
    if (tensor.type != GGMLType::COLUMN_GROUPED_AFFINE_U2_SKIP_256 ||
        tensor.n_dims != 2 || !tensor.data || !tensor.scales || !tensor.biases ||
        tensor.mlx_bits != 2 || tensor.mlx_group_size != 256 ||
        tensor.dims[0] == 0 || tensor.dims[0] > UINT32_MAX ||
        tensor.dims[1] == 0 || tensor.dims[1] > UINT32_MAX ||
        tensor.data_bytes > SIZE_MAX || tensor.scale_bytes > SIZE_MAX ||
        tensor.bias_bytes > SIZE_MAX ||
        (reinterpret_cast<uintptr_t>(tensor.data) & 127u) != 0u ||
        (reinterpret_cast<uintptr_t>(tensor.scales) & 127u) != 0u ||
        (reinterpret_cast<uintptr_t>(tensor.biases) & 127u) != 0u)
        return false;
    const uint32_t logical_k = static_cast<uint32_t>(tensor.dims[0]);
    const uint32_t logical_n = static_cast<uint32_t>(tensor.dims[1]);
    ColumnGroupedAffineUInt2SkipV1Contract contract;
    ColumnGroupedAffineUInt2SkipV1Error contract_error{};
    if (!column_grouped_affine_uint2_skip_v1_make_contract(
            logical_k, logical_n, &contract, &contract_error))
        return false;
    if (contract.group_count > UINT32_MAX) {
        session.token_.failure_detail =
            "column-grouped UInt2 shader group index exceeds uint32";
        return false;
    }
    if (contract.group_count > SIZE_MAX / sizeof(uint32_t) ||
        static_cast<uint64_t>(logical_n) * 16u > SIZE_MAX / sizeof(float))
        return false;
    const uint64_t output_blocks = static_cast<uint64_t>(logical_n / 256u);
    const uint64_t max_shader_group_index =
        (output_blocks - 1u) * static_cast<uint64_t>(logical_k) +
        (static_cast<uint64_t>(logical_k) - 1u);
    if (contract.group_count > UINT32_MAX || max_shader_group_index > UINT32_MAX) {
        session.token_.failure_detail =
            "column-grouped UInt2 shader group index exceeds uint32";
        return false;
    }
    if (tensor.data_bytes != contract.values_bytes ||
        tensor.scale_bytes != contract.scale_bytes ||
        tensor.bias_bytes != contract.bias_bytes)
        return false;
    for (const ColumnGroupedAffineU2SkipResource& resource :
         session.token_.column_grouped_affine_u2_skip) {
        if (resource.values != tensor.data) continue;
        return resource.scales == tensor.scales && resource.biases == tensor.biases &&
               resource.values_bytes == static_cast<size_t>(tensor.data_bytes) &&
               resource.scale_bytes == static_cast<size_t>(tensor.scale_bytes) &&
               resource.bias_bytes == static_cast<size_t>(tensor.bias_bytes) &&
               resource.logical_k == logical_k && resource.logical_n == logical_n;
    }

    try {
        auto& resources = session.token_.column_grouped_affine_u2_skip;
        if (resources.size() == resources.max_size()) {
            session.token_.failure_detail =
                "column-grouped UInt2 registration allocation failed";
            return false;
        }
        resources.reserve(resources.size() + 1u);
#if defined(LAPLACE_TESTING)
        if (const char* forced = std::getenv("LAPLACE_TEST_FORCE_U2_REGISTRATION_ALLOC_FAIL");
            forced && std::strcmp(forced, "1") == 0)
            throw std::bad_alloc();
#endif

        ColumnGroupedAffineU2SkipRegistrationCandidate candidate;
        const size_t metadata_bytes =
            static_cast<size_t>(contract.group_count) * sizeof(uint32_t);
        candidate.metadata.value = [g_dev newBufferWithLength:metadata_bytes
                                                         options:MTLResourceStorageModeShared];
        candidate.new_max_k = std::max(session.token_.column_grouped_affine_u2_max_k,
                                       logical_k);
        candidate.new_max_n = std::max(session.token_.column_grouped_affine_u2_max_n,
                                       logical_n);
        if (candidate.new_max_k != session.token_.column_grouped_affine_u2_max_k)
            candidate.active_columns.value = [g_dev newBufferWithLength:
                static_cast<size_t>(candidate.new_max_k) * sizeof(uint32_t)
                                                                  options:MTLResourceStorageModePrivate];
        if (candidate.new_max_n != session.token_.column_grouped_affine_u2_max_n)
            candidate.partial.value = [g_dev newBufferWithLength:
                static_cast<size_t>(candidate.new_max_n) * 16u * sizeof(float)
                                                           options:MTLResourceStorageModePrivate];
        if (!session.token_.column_grouped_affine_u2_active_count)
            candidate.active_count.value = [g_dev newBufferWithLength:sizeof(uint32_t)
                                                               options:MTLResourceStorageModeShared];
        if (!session.token_.column_grouped_affine_u2_numerical_error)
            candidate.numerical_error.value = [g_dev newBufferWithLength:sizeof(uint32_t)
                                                                    options:MTLResourceStorageModeShared];
        if (!session.token_.column_grouped_affine_u2_selected_bytes)
            candidate.selected_bytes.value = [g_dev newBufferWithLength:sizeof(uint64_t)
                                                                  options:MTLResourceStorageModeShared];
        const bool allocation_failed = !candidate.metadata.get() ||
            (candidate.new_max_k != session.token_.column_grouped_affine_u2_max_k &&
             !candidate.active_columns.get()) ||
            (candidate.new_max_n != session.token_.column_grouped_affine_u2_max_n &&
             !candidate.partial.get()) ||
            (!session.token_.column_grouped_affine_u2_active_count &&
             !candidate.active_count.get()) ||
            (!session.token_.column_grouped_affine_u2_numerical_error &&
             !candidate.numerical_error.get()) ||
            (!session.token_.column_grouped_affine_u2_selected_bytes &&
             !candidate.selected_bytes.get());
        if (allocation_failed) {
            session.token_.failure_detail =
                "column-grouped UInt2 registration allocation failed";
            return false;
        }

        auto* combined = static_cast<uint32_t*>(candidate.metadata.get().contents);
        for (size_t group = 0; group != static_cast<size_t>(contract.group_count); ++group) {
            uint16_t scale_bits = 0;
            uint16_t bias_bits = 0;
            std::memcpy(&scale_bits, tensor.scales + group * sizeof(uint16_t), sizeof(scale_bits));
            std::memcpy(&bias_bits, tensor.biases + group * sizeof(uint16_t), sizeof(bias_bits));
            if (!std::isfinite(fp16_to_fp32(scale_bits)) ||
                !std::isfinite(fp16_to_fp32(bias_bits))) {
                session.token_.failure_detail =
                    "column-grouped UInt2 registration has non-finite metadata";
                return false;
            }
            combined[group] = static_cast<uint32_t>(scale_bits) |
                              (static_cast<uint32_t>(bias_bits) << 16u);
        }

        ColumnGroupedAffineU2SkipResource resource{
            tensor.data, tensor.scales, tensor.biases,
            static_cast<size_t>(tensor.data_bytes), static_cast<size_t>(tensor.scale_bytes),
            static_cast<size_t>(tensor.bias_bytes), logical_k, logical_n,
            candidate.metadata.get()};
        resources.push_back(resource);

        if (candidate.active_columns.get()) {
            [session.token_.column_grouped_affine_u2_active_columns release];
            session.token_.column_grouped_affine_u2_active_columns = candidate.active_columns.take();
            session.token_.column_grouped_affine_u2_max_k = candidate.new_max_k;
        }
        if (candidate.partial.get()) {
            [session.token_.column_grouped_affine_u2_partial release];
            session.token_.column_grouped_affine_u2_partial = candidate.partial.take();
            session.token_.column_grouped_affine_u2_max_n = candidate.new_max_n;
        }
        if (candidate.active_count.get())
            session.token_.column_grouped_affine_u2_active_count = candidate.active_count.take();
        if (candidate.numerical_error.get())
            session.token_.column_grouped_affine_u2_numerical_error = candidate.numerical_error.take();
        if (candidate.selected_bytes.get())
            session.token_.column_grouped_affine_u2_selected_bytes = candidate.selected_bytes.take();
        candidate.metadata.take();
        return true;
    } catch (const std::bad_alloc&) {
        session.token_.failure_detail = "column-grouped UInt2 registration allocation failed";
        return false;
    } catch (const std::length_error&) {
        session.token_.failure_detail = "column-grouped UInt2 registration allocation failed";
        return false;
    }
}

MetalTokMoeCapabilities metal_tok_session_moe_capabilities(MetalTokSession& session) {
    MetalTokScope scope(session.token_);
    init();
    auto suitable = [](id<MTLComputePipelineState> pipeline, NSUInteger threads) {
        return pipeline && pipeline.threadExecutionWidth != 0 &&
               pipeline.maxTotalThreadsPerThreadgroup >= threads;
    };
    MetalTokMoeCapabilities capabilities;
    if (!g_dev || !session.token_.queue || !g_lib) return capabilities;
    capabilities.router_topk =
        suitable(named_pipe("rmsnorm_noscale"), 256) &&
        suitable(named_pipe("vec_mul"), 64) && suitable(named_pipe("vec_scale"), 64) &&
        suitable(named_pipe("router_topk"), 256);
    capabilities.gate_up_q4_k = suitable(get_q4k_id_pipe(), 64) &&
                                suitable(get_glu_experts_pipe(), 32);
    capabilities.down_q5_0 =
        suitable(get_id_pipe(static_cast<int>(GGMLType::Q5_0)), 32);
    capabilities.down_q8_0 = suitable(get_q8_id_pipe(), 64);
    capabilities.reduce = suitable(named_pipe("apply_down_scale"), 16) &&
                          suitable(named_pipe("moe_combine"), 32);
    return capabilities;
}

bool metal_tok_session_moe_ready(MetalTokSession& session) {
    const MetalTokMoeCapabilities capabilities =
        metal_tok_session_moe_capabilities(session);
    return capabilities.router_topk && capabilities.gate_up_q4_k &&
           (capabilities.down_q5_0 || capabilities.down_q8_0) &&
           capabilities.reduce;
}

bool metal_tok_session_recurrent_ready(MetalTokSession& session) {
    MetalTokScope scope(session.token_);
    init();
    auto suitable = [](id<MTLComputePipelineState> pipeline) {
        return pipeline && pipeline.threadExecutionWidth != 0 && pipeline.maxTotalThreadsPerThreadgroup >= 32;
    };
    id<MTLComputePipelineState> q2 = get_q2k_pipe();
    return g_dev && session.token_.queue && g_lib && q2 && q2.threadExecutionWidth == 32 &&
           q2.maxTotalThreadsPerThreadgroup >= 64 &&
           suitable(get_iq2_xxs_pipe()) &&
           suitable(get_pipe(static_cast<int>(GGMLType::Q4_K))) &&
           suitable(get_pipe(static_cast<int>(GGMLType::Q6_K))) && suitable(get_glu_pipe()) &&
           suitable(named_pipe("rmsnorm_f32")) && suitable(named_pipe("vec_add")) &&
           suitable(named_pipe("dnet_conv_silu")) && suitable(named_pipe("dnet_l2_qk")) &&
           suitable(named_pipe("dnet_update"));
}

bool metal_test_axis_split_gated_attention(const float* fused, int rows, int first_width,
                                           int second_width, const float* context,
                                           float* query, float* output) {
    init();
    if (!g_dev || !g_q || !g_lib || !fused || !context || !query || !output || rows < 1 ||
        first_width < 1 || second_width < 1 || first_width > INT_MAX - second_width) return false;
    const int width = first_width + second_width;
    if (rows > INT_MAX / width || rows > INT_MAX / first_width) return false;
    const int fused_count = rows * width;
    const int output_count = rows * first_width;
    id<MTLBuffer> fused_buffer = [g_dev newBufferWithBytes:fused length:(size_t)fused_count * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> context_buffer = [g_dev newBufferWithBytes:context length:(size_t)output_count * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> query_buffer = [g_dev newBufferWithLength:(size_t)output_count * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> gate_buffer = [g_dev newBufferWithLength:(size_t)output_count * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [g_dev newBufferWithLength:(size_t)output_count * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
    auto release = [&] {
        [fused_buffer release];
        [context_buffer release];
        [query_buffer release];
        [gate_buffer release];
        [output_buffer release];
    };
    id<MTLComputePipelineState> split = named_pipe("axis_split_f32");
    id<MTLComputePipelineState> gated = named_pipe("gated_attention_f32");
    if (!fused_buffer || !context_buffer || !query_buffer || !gate_buffer || !output_buffer || !split || !gated) {
        release();
        return false;
    }
    id<MTLCommandBuffer> command = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder) {
        [encoder endEncoding];
        release();
        return false;
    }
    [encoder setComputePipelineState:split];
    [encoder setBuffer:fused_buffer offset:0 atIndex:0];
    [encoder setBuffer:query_buffer offset:0 atIndex:1];
    [encoder setBuffer:gate_buffer offset:0 atIndex:2];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
    [encoder setBytes:&first_width length:sizeof(first_width) atIndex:4];
    [encoder setBytes:&second_width length:sizeof(second_width) atIndex:5];
    enc_1d(encoder, split, fused_count);
    [encoder setComputePipelineState:gated];
    [encoder setBuffer:context_buffer offset:0 atIndex:0];
    [encoder setBuffer:gate_buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBytes:&output_count length:sizeof(output_count) atIndex:3];
    enc_1d(encoder, gated, output_count);
    [encoder endEncoding];
    const auto wait_start = std::chrono::steady_clock::now();
    [command commit];
    [command waitUntilCompleted];
    const auto wait_end = std::chrono::steady_clock::now();
    const bool completed = command.status != MTLCommandBufferStatusError;
    if (completed) {
        record_completed_command(command, wait_start, wait_end);
        std::memcpy(query, [query_buffer contents], (size_t)output_count * sizeof(float));
        std::memcpy(output, [output_buffer contents], (size_t)output_count * sizeof(float));
    }
    release();
    return completed;
}

bool metal_test_q6k_embedding(const Tensor& embedding, uint32_t token, float* output,
                              int width, int vocabulary) {
    init();
    if (!g_dev || !g_q || !g_lib || !embedding.data || !output || embedding.type != GGMLType::Q6_K ||
        embedding.n_dims != 2 || width < 1 || vocabulary < 1 || width % 256 != 0 ||
        token >= static_cast<uint32_t>(vocabulary) || embedding.dims[0] != static_cast<uint64_t>(width) ||
        embedding.dims[1] != static_cast<uint64_t>(vocabulary)) return false;
    const size_t blocks = static_cast<size_t>(width / 256) * static_cast<size_t>(vocabulary);
    if (blocks > std::numeric_limits<size_t>::max() / sizeof(kernels::block_q6_K)) return false;
    id<MTLBuffer> table = [g_dev newBufferWithBytes:embedding.data length:blocks * sizeof(kernels::block_q6_K)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> result = [g_dev newBufferWithLength:static_cast<size_t>(width) * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLComputePipelineState> pipeline = named_pipe("embedding_q6k");
    if (!table || !result || !pipeline) {
        [table release];
        [result release];
        return false;
    }
    id<MTLCommandBuffer> command = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder) {
        [encoder endEncoding];
        [table release];
        [result release];
        return false;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:table offset:0 atIndex:0];
    [encoder setBuffer:result offset:0 atIndex:1];
    [encoder setBytes:&token length:sizeof(token) atIndex:2];
    [encoder setBytes:&width length:sizeof(width) atIndex:3];
    enc_1d(encoder, pipeline, width);
    [encoder endEncoding];
    const auto wait_start = std::chrono::steady_clock::now();
    [command commit];
    [command waitUntilCompleted];
    const auto wait_end = std::chrono::steady_clock::now();
    const bool completed = command.status != MTLCommandBufferStatusError;
    if (completed) {
        record_completed_command(command, wait_start, wait_end);
        std::memcpy(output, [result contents], static_cast<size_t>(width) * sizeof(float));
    }
    [table release];
    [result release];
    return completed;
}

bool metal_test_q4k_embedding(const Tensor& embedding, uint32_t token, float* output,
                              int width, int vocabulary) {
    init();
    if (!g_dev || !g_q || !g_lib || !embedding.data || !output || embedding.type != GGMLType::Q4_K ||
        embedding.n_dims != 2 || width < 1 || vocabulary < 1 || width % 256 != 0 ||
        token >= static_cast<uint32_t>(vocabulary) || embedding.dims[0] != static_cast<uint64_t>(width) ||
        embedding.dims[1] != static_cast<uint64_t>(vocabulary)) return false;
    const size_t blocks = static_cast<size_t>(width / 256) * static_cast<size_t>(vocabulary);
    if (blocks > std::numeric_limits<size_t>::max() / sizeof(kernels::block_q4_K)) return false;
    id<MTLBuffer> table = [g_dev newBufferWithBytes:embedding.data length:blocks * sizeof(kernels::block_q4_K)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> result = [g_dev newBufferWithLength:static_cast<size_t>(width) * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLComputePipelineState> pipeline = named_pipe("embedding_q4k");
    if (!table || !result || !pipeline) {
        [table release];
        [result release];
        return false;
    }
    id<MTLCommandBuffer> command = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder) {
        [encoder endEncoding];
        [table release];
        [result release];
        return false;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:table offset:0 atIndex:0];
    [encoder setBuffer:result offset:0 atIndex:1];
    [encoder setBytes:&token length:sizeof(token) atIndex:2];
    [encoder setBytes:&width length:sizeof(width) atIndex:3];
    enc_1d(encoder, pipeline, width);
    [encoder endEncoding];
    const auto wait_start = std::chrono::steady_clock::now();
    [command commit];
    [command waitUntilCompleted];
    const auto wait_end = std::chrono::steady_clock::now();
    const bool completed = command.status != MTLCommandBufferStatusError;
    if (completed) {
        record_completed_command(command, wait_start, wait_end);
        std::memcpy(output, [result contents], static_cast<size_t>(width) * sizeof(float));
    }
    [table release];
    [result release];
    return completed;
}

#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
bool metal_test_column_grouped_affine_lowbit_v1(
    const ColumnGroupedAffineLowBitV1Contract& contract,
    const ColumnGroupedAffineLowBitV1Planes& planes,
    const float* input, float* output, double* gpu_ms,
    uint32_t* thread_execution_width, uint32_t* max_threads_per_threadgroup) {
    if (!affine_v1_contract_valid(contract) || !affine_v1_planes_valid(contract, planes) ||
        !input || !output || !gpu_ms || !thread_execution_width || !max_threads_per_threadgroup)
        return false;
    if (!column_grouped_affine_lowbit_v1_pipeline_ready()) return false;
    id<MTLBuffer> values = [g_dev newBufferWithBytes:planes.values
                                                length:planes.values_bytes
                                              options:MTLResourceStorageModeShared];
    const size_t plane_bytes = planes.scale_count * sizeof(uint16_t);
    id<MTLBuffer> scales = [g_dev newBufferWithBytes:planes.scales
                                                length:plane_bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> biases = [g_dev newBufferWithBytes:planes.biases
                                                length:plane_bytes
                                              options:MTLResourceStorageModeShared];
    const size_t input_bytes = static_cast<size_t>(contract.logical_k) * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(contract.logical_n) * sizeof(float);
    id<MTLBuffer> input_buffer = [g_dev newBufferWithBytes:input length:input_bytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [g_dev newBufferWithLength:output_bytes
                                                      options:MTLResourceStorageModeShared];
    auto release = [&] {
        [values release];
        [scales release];
        [biases release];
        [input_buffer release];
        [output_buffer release];
    };
    if (!values || !scales || !biases || !input_buffer || !output_buffer) {
        release();
        return false;
    }
    id<MTLCommandBuffer> command = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> encoder = command ? [command computeCommandEncoder] : nil;
    if (!command || !encoder) {
        if (encoder) [encoder endEncoding];
        release();
        return false;
    }
    const uint32_t logical_k = contract.logical_k;
    const uint32_t logical_n = contract.logical_n;
    const uint32_t bits = contract.bits;
    const uint32_t packed_bytes = contract.packed_bytes;
    const NSUInteger width = g_column_grouped_affine_lowbit_pipe.threadExecutionWidth;
    const NSUInteger maximum_threads =
        g_column_grouped_affine_lowbit_pipe.maxTotalThreadsPerThreadgroup;
    if (width == 0 || maximum_threads < width) {
        [encoder endEncoding];
        release();
        return false;
    }
    [encoder setComputePipelineState:g_column_grouped_affine_lowbit_pipe];
    [encoder setBuffer:values offset:0 atIndex:0];
    [encoder setBuffer:scales offset:0 atIndex:1];
    [encoder setBuffer:biases offset:0 atIndex:2];
    [encoder setBuffer:input_buffer offset:0 atIndex:3];
    [encoder setBuffer:output_buffer offset:0 atIndex:4];
    [encoder setBytes:&logical_k length:sizeof(logical_k) atIndex:5];
    [encoder setBytes:&logical_n length:sizeof(logical_n) atIndex:6];
    [encoder setBytes:&bits length:sizeof(bits) atIndex:7];
    [encoder setBytes:&packed_bytes length:sizeof(packed_bytes) atIndex:8];
    const NSUInteger groups =
        (static_cast<NSUInteger>(contract.logical_n) + width - 1u) / width;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    const auto wait_start = std::chrono::steady_clock::now();
    [command commit];
    [command waitUntilCompleted];
    const auto wait_end = std::chrono::steady_clock::now();
    const bool completed = command.status == MTLCommandBufferStatusCompleted;
    if (completed) {
        record_completed_command(command, wait_start, wait_end);
        const CFTimeInterval gpu_start = [command GPUStartTime];
        const CFTimeInterval gpu_end = [command GPUEndTime];
        *gpu_ms = gpu_end >= gpu_start ? 1000.0 * (gpu_end - gpu_start) :
                  std::chrono::duration<double, std::milli>(wait_end - wait_start).count();
        std::memcpy(output, [output_buffer contents], output_bytes);
        *thread_execution_width = static_cast<uint32_t>(width);
        *max_threads_per_threadgroup = static_cast<uint32_t>(maximum_threads);
    }
    release();
    return completed;
}

bool metal_test_column_grouped_affine_lowbit_v1_benchmark(
    const ColumnGroupedAffineLowBitV1Contract& contract,
    const ColumnGroupedAffineLowBitV1Planes& planes,
    const float* input, float* output, uint32_t warmups, uint32_t samples,
    double* sample_gpu_ms, uint64_t* requested_bytes,
    uint32_t* thread_execution_width, uint32_t* max_threads_per_threadgroup) {
    if (!affine_v1_contract_valid(contract) || !affine_v1_planes_valid(contract, planes) ||
        !input || !output || !sample_gpu_ms || !requested_bytes ||
        !thread_execution_width || !max_threads_per_threadgroup || warmups < 5u || samples < 5u)
        return false;
    if (contract.group_count > std::numeric_limits<uint64_t>::max() /
                                  (sizeof(uint16_t) * 2u) ||
        contract.values_bytes > std::numeric_limits<uint64_t>::max() -
                                    contract.group_count * (sizeof(uint16_t) * 2u) ||
        static_cast<uint64_t>(contract.logical_k) >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / sizeof(float) ||
        static_cast<uint64_t>(contract.logical_n) >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / sizeof(float))
        return false;
    if (!column_grouped_affine_lowbit_v1_pipeline_ready()) return false;

    const size_t plane_bytes = planes.scale_count * sizeof(uint16_t);
    const size_t input_bytes = static_cast<size_t>(contract.logical_k) * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(contract.logical_n) * sizeof(float);
    id<MTLBuffer> values = [g_dev newBufferWithBytes:planes.values
                                                length:planes.values_bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> scales = [g_dev newBufferWithBytes:planes.scales
                                                length:plane_bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> biases = [g_dev newBufferWithBytes:planes.biases
                                                length:plane_bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> input_buffer = [g_dev newBufferWithBytes:input length:input_bytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [g_dev newBufferWithLength:output_bytes
                                                      options:MTLResourceStorageModeShared];
    auto release = [&] {
        [values release];
        [scales release];
        [biases release];
        [input_buffer release];
        [output_buffer release];
    };
    if (!values || !scales || !biases || !input_buffer || !output_buffer) {
        release();
        return false;
    }

    const uint32_t logical_k = contract.logical_k;
    const uint32_t logical_n = contract.logical_n;
    const uint32_t bits = contract.bits;
    const uint32_t packed_bytes = contract.packed_bytes;
    const NSUInteger width = g_column_grouped_affine_lowbit_pipe.threadExecutionWidth;
    const NSUInteger maximum_threads =
        g_column_grouped_affine_lowbit_pipe.maxTotalThreadsPerThreadgroup;
    if (width == 0 || maximum_threads < width) {
        release();
        return false;
    }
    const NSUInteger groups =
        (static_cast<NSUInteger>(contract.logical_n) + width - 1u) / width;
    if (groups == 0) {
        release();
        return false;
    }

    const auto run_sample = [&](double* elapsed_ms) -> bool {
        id<MTLCommandBuffer> command = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> encoder = command ? [command computeCommandEncoder] : nil;
        if (!command || !encoder) {
            if (encoder) [encoder endEncoding];
            return false;
        }
        [encoder setComputePipelineState:g_column_grouped_affine_lowbit_pipe];
        [encoder setBuffer:values offset:0 atIndex:0];
        [encoder setBuffer:scales offset:0 atIndex:1];
        [encoder setBuffer:biases offset:0 atIndex:2];
        [encoder setBuffer:input_buffer offset:0 atIndex:3];
        [encoder setBuffer:output_buffer offset:0 atIndex:4];
        [encoder setBytes:&logical_k length:sizeof(logical_k) atIndex:5];
        [encoder setBytes:&logical_n length:sizeof(logical_n) atIndex:6];
        [encoder setBytes:&bits length:sizeof(bits) atIndex:7];
        [encoder setBytes:&packed_bytes length:sizeof(packed_bytes) atIndex:8];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        const auto wait_start = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        const auto wait_end = std::chrono::steady_clock::now();
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "[metal] ColumnGroupedAffineLowBitV1 benchmark command: %s\n",
                         command_buffer_failure_detail(command).c_str());
            return false;
        }
        record_completed_command(command, wait_start, wait_end);
        const CFTimeInterval gpu_start = [command GPUStartTime];
        const CFTimeInterval gpu_end = [command GPUEndTime];
        const double host_ms =
            std::chrono::duration<double, std::milli>(wait_end - wait_start).count();
        *elapsed_ms = gpu_end >= gpu_start ? 1000.0 * (gpu_end - gpu_start) : host_ms;
        return std::isfinite(*elapsed_ms) && *elapsed_ms > 0.0;
    };

    for (uint32_t warmup = 0; warmup != warmups; ++warmup) {
        double ignored_ms = 0.0;
        if (!run_sample(&ignored_ms)) {
            release();
            return false;
        }
    }
    metal_dispatch_metrics_reset();
    for (uint32_t sample = 0; sample != samples; ++sample) {
        if (!run_sample(&sample_gpu_ms[sample])) {
            release();
            return false;
        }
    }
    std::memcpy(output, [output_buffer contents], output_bytes);
    *requested_bytes = contract.values_bytes +
                       contract.group_count * (sizeof(uint16_t) * 2u);
    *thread_execution_width = static_cast<uint32_t>(width);
    *max_threads_per_threadgroup = static_cast<uint32_t>(maximum_threads);
    release();
    return true;
}

static bool metal_test_column_grouped_affine_lowbit_v1_optimized_run(
    const ColumnGroupedAffineLowBitV1Contract& contract,
    const ColumnGroupedAffineLowBitV1Planes& planes,
    const float* input, float* output, uint32_t warmups, uint32_t samples,
    double* sample_gpu_ms, uint64_t* requested_bytes,
    uint32_t* thread_execution_width, uint32_t* max_threads_per_threadgroup) {
    if (!affine_v1_contract_valid(contract) || !affine_v1_planes_valid(contract, planes) ||
        !input || !output || !sample_gpu_ms || !requested_bytes ||
        !thread_execution_width || !max_threads_per_threadgroup || samples == 0u)
        return false;
    if (contract.group_count > std::numeric_limits<uint64_t>::max() /
                                  (sizeof(uint16_t) * 2u) ||
        contract.values_bytes > std::numeric_limits<uint64_t>::max() -
                                    contract.group_count * (sizeof(uint16_t) * 2u) ||
        static_cast<uint64_t>(contract.logical_k) >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / sizeof(float) ||
        static_cast<uint64_t>(contract.logical_n) >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / sizeof(float))
        return false;
    if (!column_grouped_affine_lowbit_v1_pipeline_ready()) return false;
    id<MTLComputePipelineState> pipeline =
        column_grouped_affine_lowbit_v1_optimized_pipeline(contract.bits);
    if (!pipeline) return false;

    const size_t plane_bytes = planes.scale_count * sizeof(uint16_t);
    const size_t input_bytes = static_cast<size_t>(contract.logical_k) * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(contract.logical_n) * sizeof(float);
    id<MTLBuffer> values = [g_dev newBufferWithBytes:planes.values
                                                length:planes.values_bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> scales = [g_dev newBufferWithBytes:planes.scales
                                                length:plane_bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> biases = [g_dev newBufferWithBytes:planes.biases
                                                length:plane_bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> input_buffer = [g_dev newBufferWithBytes:input length:input_bytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [g_dev newBufferWithLength:output_bytes
                                                      options:MTLResourceStorageModeShared];
    auto release = [&] {
        [values release];
        [scales release];
        [biases release];
        [input_buffer release];
        [output_buffer release];
    };
    if (!values || !scales || !biases || !input_buffer || !output_buffer) {
        release();
        return false;
    }

    const NSUInteger width = pipeline.threadExecutionWidth;
    const NSUInteger maximum_threads = pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger threads_per_threadgroup = width;
    const uint32_t rows_per_thread = contract.bits == 2u ? 4u : 2u;
    if (width == 0 || maximum_threads < threads_per_threadgroup) {
        release();
        return false;
    }
    const NSUInteger total_threads =
        (static_cast<NSUInteger>(contract.logical_n) + rows_per_thread - 1u) /
        rows_per_thread;
    const NSUInteger groups = (total_threads + width - 1u) / width;
    const uint32_t logical_k = contract.logical_k;
    const uint32_t logical_n = contract.logical_n;

    const auto run_sample = [&](double* elapsed_ms) -> bool {
        id<MTLCommandBuffer> command = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> encoder = command ? [command computeCommandEncoder] : nil;
        if (!command || !encoder) {
            if (encoder) [encoder endEncoding];
            return false;
        }
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:values offset:0 atIndex:0];
        [encoder setBuffer:scales offset:0 atIndex:1];
        [encoder setBuffer:biases offset:0 atIndex:2];
        [encoder setBuffer:input_buffer offset:0 atIndex:3];
        [encoder setBuffer:output_buffer offset:0 atIndex:4];
        [encoder setBytes:&logical_k length:sizeof(logical_k) atIndex:5];
        [encoder setBytes:&logical_n length:sizeof(logical_n) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];
        [encoder endEncoding];
        const auto wait_start = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        const auto wait_end = std::chrono::steady_clock::now();
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "[metal] ColumnGroupedAffineLowBitV1 optimized command: %s\n",
                         command_buffer_failure_detail(command).c_str());
            return false;
        }
        record_completed_command(command, wait_start, wait_end);
        const CFTimeInterval gpu_start = [command GPUStartTime];
        const CFTimeInterval gpu_end = [command GPUEndTime];
        const double host_ms =
            std::chrono::duration<double, std::milli>(wait_end - wait_start).count();
        *elapsed_ms = gpu_end >= gpu_start ? 1000.0 * (gpu_end - gpu_start) : host_ms;
        return std::isfinite(*elapsed_ms) && *elapsed_ms > 0.0;
    };

    for (uint32_t warmup = 0; warmup != warmups; ++warmup) {
        double ignored_ms = 0.0;
        if (!run_sample(&ignored_ms)) {
            release();
            return false;
        }
    }
    if (warmups != 0u) metal_dispatch_metrics_reset();
    for (uint32_t sample = 0; sample != samples; ++sample) {
        if (!run_sample(&sample_gpu_ms[sample])) {
            release();
            return false;
        }
    }
    std::memcpy(output, [output_buffer contents], output_bytes);
    *requested_bytes = contract.values_bytes +
                       contract.group_count * (sizeof(uint16_t) * 2u);
    *thread_execution_width = static_cast<uint32_t>(width);
    *max_threads_per_threadgroup = static_cast<uint32_t>(maximum_threads);
    release();
    return true;
}

bool metal_test_column_grouped_affine_lowbit_v1_optimized(
    const ColumnGroupedAffineLowBitV1Contract& contract,
    const ColumnGroupedAffineLowBitV1Planes& planes,
    const float* input, float* output, double* gpu_ms,
    uint32_t* thread_execution_width, uint32_t* max_threads_per_threadgroup) {
    if (!gpu_ms) return false;
    double sample_gpu_ms = 0.0;
    uint64_t requested_bytes = 0;
    const bool executed = metal_test_column_grouped_affine_lowbit_v1_optimized_run(
        contract, planes, input, output, 0u, 1u, &sample_gpu_ms, &requested_bytes,
        thread_execution_width, max_threads_per_threadgroup);
    if (executed) *gpu_ms = sample_gpu_ms;
    return executed;
}

bool metal_test_column_grouped_affine_lowbit_v1_optimized_benchmark(
    const ColumnGroupedAffineLowBitV1Contract& contract,
    const ColumnGroupedAffineLowBitV1Planes& planes,
    const float* input, float* output, uint32_t warmups, uint32_t samples,
    double* sample_gpu_ms, uint64_t* requested_bytes,
    uint32_t* thread_execution_width, uint32_t* max_threads_per_threadgroup) {
    if (warmups < 5u || samples < 5u) return false;
    return metal_test_column_grouped_affine_lowbit_v1_optimized_run(
        contract, planes, input, output, warmups, samples, sample_gpu_ms,
        requested_bytes, thread_execution_width, max_threads_per_threadgroup);
}

static bool metal_test_column_grouped_affine_uint2_skip_v1_run(
    const ColumnGroupedAffineUInt2SkipV1Contract& contract,
    const ColumnGroupedAffineUInt2SkipV1Planes& planes,
    const float* input, float* output, bool sparse,
    uint32_t warmups,
    uint32_t samples, double* sample_gpu_ms, uint64_t* requested_bytes,
    uint32_t* thread_execution_width, uint32_t* max_threads_per_threadgroup) {
    if (!column_grouped_affine_uint2_skip_v1_contract_valid(contract) ||
        !column_grouped_affine_uint2_skip_v1_planes_valid(contract, planes) ||
        !input || !output || samples == 0u || !sample_gpu_ms || !requested_bytes ||
        !thread_execution_width || !max_threads_per_threadgroup ||
        contract.logical_k > std::numeric_limits<size_t>::max() / sizeof(float) ||
        contract.logical_n > std::numeric_limits<size_t>::max() / sizeof(float))
        return false;
    if (!column_grouped_affine_uint2_skip_v1_pipelines_ready()) return false;
    id<MTLComputePipelineState> gemv = sparse
        ? g_column_grouped_affine_uint2_skip_sparse_pipe
        : g_column_grouped_affine_uint2_skip_dense_pipe;
    const NSUInteger width = gemv.threadExecutionWidth;
    const NSUInteger maximum_threads = gemv.maxTotalThreadsPerThreadgroup;
    const NSUInteger selector_width =
        g_column_grouped_affine_uint2_skip_selector_pipe.threadExecutionWidth;
    const NSUInteger selector_maximum =
        g_column_grouped_affine_uint2_skip_selector_pipe.maxTotalThreadsPerThreadgroup;
    if (!gemv || width == 0u || maximum_threads < 64u || (64u % width) != 0u ||
        (sparse && (width != 32u || selector_width == 0u ||
                    selector_maximum < 256u ||
                    g_column_grouped_affine_uint2_skip_reduce_pipe.threadExecutionWidth == 0u ||
                    g_column_grouped_affine_uint2_skip_reduce_pipe.maxTotalThreadsPerThreadgroup < 256u)))
        return false;

    const uint32_t logical_k = static_cast<uint32_t>(contract.logical_k);
    const uint32_t logical_n = static_cast<uint32_t>(contract.logical_n);
    uint32_t split_count = 16u;
    if (const char* value = std::getenv("LAPLACE_TEST_UINT2_SKIP_SPLIT")) {
        const unsigned long parsed = std::strtoul(value, nullptr, 10);
        if (parsed >= 1u && parsed <= 32u)
            split_count = static_cast<uint32_t>(parsed);
    }
    const size_t output_blocks = logical_n / 256u;
    if (sparse && output_blocks > std::numeric_limits<size_t>::max() /
                                      (split_count * 256u * sizeof(float)))
        return false;
    const size_t partial_bytes = sparse
        ? output_blocks * split_count * 256u * sizeof(float)
        : 0u;
    const size_t metadata_bytes = planes.scale_count * sizeof(uint16_t);
    const size_t input_bytes = static_cast<size_t>(logical_k) * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(logical_n) * sizeof(float);
    id<MTLBuffer> values = [g_dev newBufferWithBytes:planes.values
                                                length:planes.values_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> scales = [g_dev newBufferWithBytes:planes.scales
                                                length:metadata_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> biases = [g_dev newBufferWithBytes:planes.biases
                                                length:metadata_bytes
                                               options:MTLResourceStorageModeShared];
    std::vector<uint32_t> sparse_metadata;
    id<MTLBuffer> sparse_metadata_buffer = nil;
    if (sparse) {
        sparse_metadata.resize(planes.scale_count);
        for (size_t index = 0; index != sparse_metadata.size(); ++index)
            sparse_metadata[index] = uint32_t(planes.scales[index]) |
                                     (uint32_t(planes.biases[index]) << 16u);
        sparse_metadata_buffer = [g_dev newBufferWithBytes:sparse_metadata.data()
                                                    length:sparse_metadata.size() * sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> input_buffer = [g_dev newBufferWithBytes:input length:input_bytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [g_dev newBufferWithLength:output_bytes
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> active_count = nil;
    id<MTLBuffer> active_columns = nil;
    id<MTLBuffer> partial_output = nil;
    if (sparse) {
        active_count = [g_dev newBufferWithLength:sizeof(uint32_t)
                                          options:MTLResourceStorageModePrivate];
        active_columns = [g_dev newBufferWithLength:static_cast<size_t>(logical_k) *
                                                        sizeof(uint32_t)
                                            options:MTLResourceStorageModePrivate];
    }
    if (sparse) {
        partial_output = [g_dev newBufferWithLength:partial_bytes
                                            options:MTLResourceStorageModePrivate];
    }
    const auto release = [&] {
        [values release];
        [scales release];
        [biases release];
        [input_buffer release];
        [output_buffer release];
        [active_count release];
        [active_columns release];
        [partial_output release];
        [sparse_metadata_buffer release];
    };
    if (!values || !scales || !biases || !input_buffer || !output_buffer ||
        (sparse && (!active_count || !active_columns || !partial_output ||
                    !sparse_metadata_buffer))) {
        release();
        return false;
    }

    const NSUInteger selector_groups = sparse ? 1u : 0u;
    const auto profile_stage = [&](uint32_t stage, double* elapsed_ms) -> bool {
        id<MTLCommandBuffer> command = [g_q commandBuffer];
        if (!command) return false;
        if (stage == 0u) {
            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            if (!blit) return false;
            [blit fillBuffer:active_count range:NSMakeRange(0, sizeof(uint32_t)) value:0];
            [blit endEncoding];
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) return false;
        if (stage == 0u) {
            [encoder setComputePipelineState:g_column_grouped_affine_uint2_skip_selector_pipe];
            [encoder setBuffer:input_buffer offset:0 atIndex:0];
            [encoder setBuffer:active_count offset:0 atIndex:1];
            [encoder setBuffer:active_columns offset:0 atIndex:2];
            [encoder setBytes:&logical_k length:sizeof(logical_k) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else if (stage == 1u) {
            [encoder setComputePipelineState:g_column_grouped_affine_uint2_skip_sparse_pipe];
            [encoder setBuffer:values offset:0 atIndex:0];
            [encoder setBuffer:sparse_metadata_buffer offset:0 atIndex:1];
            [encoder setBuffer:input_buffer offset:0 atIndex:3];
            [encoder setBuffer:partial_output offset:0 atIndex:4];
            [encoder setBuffer:active_count offset:0 atIndex:5];
            [encoder setBuffer:active_columns offset:0 atIndex:6];
            [encoder setBytes:&logical_k length:sizeof(logical_k) atIndex:7];
            [encoder setBytes:&logical_n length:sizeof(logical_n) atIndex:8];
            [encoder setBytes:&split_count length:sizeof(split_count) atIndex:9];
            [encoder dispatchThreadgroups:MTLSizeMake(output_blocks * split_count, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        } else {
            [encoder setComputePipelineState:g_column_grouped_affine_uint2_skip_reduce_pipe];
            [encoder setBuffer:partial_output offset:0 atIndex:0];
            [encoder setBuffer:output_buffer offset:0 atIndex:1];
            [encoder setBytes:&logical_n length:sizeof(logical_n) atIndex:2];
            [encoder setBytes:&split_count length:sizeof(split_count) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(output_blocks, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [encoder endEncoding];
        const auto wait_start = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        const auto wait_end = std::chrono::steady_clock::now();
        if (command.status != MTLCommandBufferStatusCompleted) return false;
        const CFTimeInterval gpu_start = command.GPUStartTime;
        const CFTimeInterval gpu_end = command.GPUEndTime;
        const double host_ms =
            std::chrono::duration<double, std::milli>(wait_end - wait_start).count();
        *elapsed_ms = gpu_end >= gpu_start ? 1000.0 * (gpu_end - gpu_start) : host_ms;
        return std::isfinite(*elapsed_ms) && *elapsed_ms > 0.0;
    };
    if (sparse && std::getenv("LAPLACE_TEST_UINT2_SKIP_PROFILE_STAGES")) {
        constexpr uint32_t profile_warmups = 10u;
        constexpr uint32_t profile_samples = 50u;
        for (uint32_t stage = 0u; stage != 3u; ++stage) {
            std::array<double, profile_samples> timings{};
            for (uint32_t sample = 0u; sample != profile_warmups + profile_samples; ++sample) {
                double elapsed_ms = 0.0;
                if (!profile_stage(stage, &elapsed_ms)) {
                    release();
                    return false;
                }
                if (sample >= profile_warmups)
                    timings[sample - profile_warmups] = elapsed_ms;
            }
            std::sort(timings.begin(), timings.end());
            static const char* names[] = {"selector", "partial", "reduce"};
            std::fprintf(stderr,
                         "column-grouped-affine-uint2-skip-stage stage=%s K=%u N=%u "
                         "split=%u median_gpu_ms=%.6f range=%.6f..%.6f\n",
                         names[stage], logical_k, logical_n, split_count,
                         timings[profile_samples / 2u], timings.front(), timings.back());
        }
    }
    const auto run_sample = [&](double* elapsed_ms) -> bool {
        id<MTLCommandBuffer> command = [g_q commandBuffer];
        if (!command) return false;
        if (sparse) {
            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            if (!blit) return false;
            [blit fillBuffer:active_count range:NSMakeRange(0, sizeof(uint32_t)) value:0];
            [blit endEncoding];
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) return false;
        if (sparse) {
            [encoder setComputePipelineState:g_column_grouped_affine_uint2_skip_selector_pipe];
            [encoder setBuffer:input_buffer offset:0 atIndex:0];
            [encoder setBuffer:active_count offset:0 atIndex:1];
            [encoder setBuffer:active_columns offset:0 atIndex:2];
            [encoder setBytes:&logical_k length:sizeof(logical_k) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(selector_groups, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        [encoder setComputePipelineState:gemv];
        [encoder setBuffer:values offset:0 atIndex:0];
        [encoder setBuffer:(sparse ? sparse_metadata_buffer : scales)
                   offset:0 atIndex:1];
        if (!sparse) [encoder setBuffer:biases offset:0 atIndex:2];
        [encoder setBuffer:input_buffer offset:0 atIndex:3];
        if (sparse) {
            [encoder setBuffer:partial_output offset:0 atIndex:4];
            [encoder setBuffer:active_count offset:0 atIndex:5];
            [encoder setBuffer:active_columns offset:0 atIndex:6];
            [encoder setBytes:&logical_k length:sizeof(logical_k) atIndex:7];
            [encoder setBytes:&logical_n length:sizeof(logical_n) atIndex:8];
            [encoder setBytes:&split_count length:sizeof(split_count) atIndex:9];
        } else {
            [encoder setBuffer:output_buffer offset:0 atIndex:4];
            [encoder setBytes:&logical_k length:sizeof(logical_k) atIndex:5];
            [encoder setBytes:&logical_n length:sizeof(logical_n) atIndex:6];
        }
        const NSUInteger gemv_groups = sparse
            ? output_blocks * split_count : output_blocks;
        const NSUInteger gemv_threads = 64u;
        [encoder dispatchThreadgroups:MTLSizeMake(gemv_groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(gemv_threads, 1, 1)];
        if (sparse) {
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [encoder setComputePipelineState:g_column_grouped_affine_uint2_skip_reduce_pipe];
            [encoder setBuffer:partial_output offset:0 atIndex:0];
            [encoder setBuffer:output_buffer offset:0 atIndex:1];
            [encoder setBytes:&logical_n length:sizeof(logical_n) atIndex:2];
            [encoder setBytes:&split_count length:sizeof(split_count) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(output_blocks, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [encoder endEncoding];
        const auto wait_start = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        const auto wait_end = std::chrono::steady_clock::now();
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "[metal] ColumnGroupedAffineUInt2SkipV1 command: %s\n",
                         command_buffer_failure_detail(command).c_str());
            return false;
        }
        record_completed_command(command, wait_start, wait_end);
        const CFTimeInterval gpu_start = command.GPUStartTime;
        const CFTimeInterval gpu_end = command.GPUEndTime;
        const double host_ms =
            std::chrono::duration<double, std::milli>(wait_end - wait_start).count();
        *elapsed_ms = gpu_end >= gpu_start ? 1000.0 * (gpu_end - gpu_start) : host_ms;
        return std::isfinite(*elapsed_ms) && *elapsed_ms > 0.0;
    };

    for (uint32_t warmup = 0; warmup != warmups; ++warmup) {
        double ignored_ms = 0.0;
        if (!run_sample(&ignored_ms)) {
            release();
            return false;
        }
    }
    if (warmups != 0u) metal_dispatch_metrics_reset();
    for (uint32_t sample = 0; sample != samples; ++sample) {
        if (!run_sample(&sample_gpu_ms[sample])) {
            release();
            return false;
        }
    }
    std::memcpy(output, [output_buffer contents], output_bytes);
    uint64_t active_columns_count = logical_k;
    if (sparse) {
        active_columns_count = 0;
        for (uint32_t column = 0; column != logical_k; ++column)
            if (std::isfinite(input[column]) && input[column] != 0.0f)
                ++active_columns_count;
    }
    *requested_bytes = active_columns_count * static_cast<uint64_t>(output_blocks) *
                       (64u + sizeof(uint16_t) + sizeof(uint16_t));
    *thread_execution_width = static_cast<uint32_t>(width);
    *max_threads_per_threadgroup = static_cast<uint32_t>(maximum_threads);
    release();
    return true;
}

bool metal_test_column_grouped_affine_uint2_skip_v1(
    const ColumnGroupedAffineUInt2SkipV1Contract& contract,
    const ColumnGroupedAffineUInt2SkipV1Planes& planes,
    const float* input, float* output, bool sparse, double* gpu_ms,
    uint32_t* thread_execution_width, uint32_t* max_threads_per_threadgroup) {
    if (!gpu_ms) return false;
    uint64_t requested_bytes = 0;
    return metal_test_column_grouped_affine_uint2_skip_v1_run(
        contract, planes, input, output, sparse, 0u, 1u, gpu_ms, &requested_bytes,
        thread_execution_width, max_threads_per_threadgroup);
}

bool metal_test_column_grouped_affine_uint2_skip_v1_benchmark(
    const ColumnGroupedAffineUInt2SkipV1Contract& contract,
    const ColumnGroupedAffineUInt2SkipV1Planes& planes,
    const float* input, float* output, bool sparse, uint32_t warmups,
    uint32_t samples, double* sample_gpu_ms, uint64_t* requested_bytes,
    uint32_t* thread_execution_width, uint32_t* max_threads_per_threadgroup) {
    return metal_test_column_grouped_affine_uint2_skip_v1_run(
        contract, planes, input, output, sparse, warmups, samples, sample_gpu_ms,
        requested_bytes, thread_execution_width, max_threads_per_threadgroup);
}
#endif

bool metal_test_rope_interleaved(float* query, float* key, int query_heads, int key_heads,
                                 int head_dimension, int rotary_dimension, int frequency_dimension,
                                 float rope_base, int position) {
    init();
    if (!g_dev || !g_q || !g_lib || !query || !key || query_heads < 1 || key_heads < 1 ||
        head_dimension < 2 || rotary_dimension < 2 || rotary_dimension > head_dimension ||
        rotary_dimension % 2 != 0 || frequency_dimension < rotary_dimension ||
        frequency_dimension % 2 != 0 || !std::isfinite(rope_base) || rope_base <= 0) return false;
    const int pairs = rotary_dimension / 2;
    if (query_heads > INT_MAX / head_dimension || key_heads > INT_MAX / head_dimension ||
        query_heads > INT_MAX / pairs || key_heads > INT_MAX / pairs) return false;
    const int query_count = query_heads * head_dimension;
    const int key_count = key_heads * head_dimension;
    id<MTLBuffer> query_buffer = [g_dev newBufferWithBytes:query length:(size_t)query_count * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> key_buffer = [g_dev newBufferWithBytes:key length:(size_t)key_count * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
    auto release = [&] {
        [query_buffer release];
        [key_buffer release];
    };
    id<MTLComputePipelineState> rope = named_pipe("rope_interleaved_f32");
    if (!query_buffer || !key_buffer || !rope) {
        release();
        return false;
    }
    id<MTLCommandBuffer> command = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder) {
        [encoder endEncoding];
        release();
        return false;
    }
    const int no_frequency_scale = 0;
    [encoder setComputePipelineState:rope];
    [encoder setBuffer:query_buffer offset:0 atIndex:0];
    [encoder setBytes:&query_heads length:sizeof(query_heads) atIndex:1];
    [encoder setBytes:&head_dimension length:sizeof(head_dimension) atIndex:2];
    [encoder setBytes:&pairs length:sizeof(pairs) atIndex:3];
    [encoder setBytes:&rope_base length:sizeof(rope_base) atIndex:4];
    [encoder setBytes:&position length:sizeof(position) atIndex:5];
    [encoder setBytes:&no_frequency_scale length:sizeof(no_frequency_scale) atIndex:6];
    [encoder setBuffer:query_buffer offset:0 atIndex:7];
    [encoder setBytes:&frequency_dimension length:sizeof(frequency_dimension) atIndex:8];
    enc_1d(encoder, rope, query_heads * pairs);
    [encoder setBuffer:key_buffer offset:0 atIndex:0];
    [encoder setBytes:&key_heads length:sizeof(key_heads) atIndex:1];
    [encoder setBuffer:key_buffer offset:0 atIndex:7];
    [encoder setBytes:&frequency_dimension length:sizeof(frequency_dimension) atIndex:8];
    enc_1d(encoder, rope, key_heads * pairs);
    [encoder endEncoding];
    const auto wait_start = std::chrono::steady_clock::now();
    [command commit];
    [command waitUntilCompleted];
    const auto wait_end = std::chrono::steady_clock::now();
    const bool completed = command.status != MTLCommandBufferStatusError;
    if (completed) {
        record_completed_command(command, wait_start, wait_end);
        std::memcpy(query, [query_buffer contents], (size_t)query_count * sizeof(float));
        std::memcpy(key, [key_buffer contents], (size_t)key_count * sizeof(float));
    }
    release();
    return completed;
}

bool metal_test_rope_multisection(float* query, float* key, int query_heads, int key_heads,
                                  int head_dimension, int rotary_dimension, int frequency_dimension,
                                  float rope_base, const int positions[4],
                                  const uint32_t sections[4]) {
    init();
    if (!g_dev || !g_q || !g_lib || !query || !key || !positions || !sections || query_heads < 1 || key_heads < 1 ||
        head_dimension < 2 || rotary_dimension < 2 || rotary_dimension > head_dimension ||
        rotary_dimension % 2 != 0 || frequency_dimension < rotary_dimension ||
        frequency_dimension % 2 != 0 || !std::isfinite(rope_base) || rope_base <= 0) return false;
    const int pairs = rotary_dimension / 2;
    uint64_t section_total = 0;
    for (uint32_t index = 0; index != 4; ++index) section_total += sections[index];
    if (section_total != static_cast<uint32_t>(pairs) || query_heads > INT_MAX / head_dimension ||
        key_heads > INT_MAX / head_dimension || query_heads > INT_MAX / pairs || key_heads > INT_MAX / pairs) return false;
    const int query_count = query_heads * head_dimension;
    const int key_count = key_heads * head_dimension;
    id<MTLBuffer> query_buffer = [g_dev newBufferWithBytes:query length:(size_t)query_count * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> key_buffer = [g_dev newBufferWithBytes:key length:(size_t)key_count * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
    auto release = [&] { [query_buffer release]; [key_buffer release]; };
    id<MTLComputePipelineState> rope = named_pipe("rope_multisection_f32");
    if (!query_buffer || !key_buffer || !rope) { release(); return false; }
    id<MTLCommandBuffer> command = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder) { [encoder endEncoding]; release(); return false; }
    const int no_frequency_scale = 0;
    [encoder setComputePipelineState:rope];
    [encoder setBuffer:query_buffer offset:0 atIndex:0];
    [encoder setBytes:&query_heads length:sizeof(query_heads) atIndex:1];
    [encoder setBytes:&head_dimension length:sizeof(head_dimension) atIndex:2];
    [encoder setBytes:&pairs length:sizeof(pairs) atIndex:3];
    [encoder setBytes:&rope_base length:sizeof(rope_base) atIndex:4];
    [encoder setBytes:positions length:4 * sizeof(*positions) atIndex:5];
    [encoder setBytes:sections length:4 * sizeof(*sections) atIndex:6];
    [encoder setBytes:&no_frequency_scale length:sizeof(no_frequency_scale) atIndex:7];
    [encoder setBuffer:query_buffer offset:0 atIndex:8];
    [encoder setBytes:&frequency_dimension length:sizeof(frequency_dimension) atIndex:9];
    enc_1d(encoder, rope, query_heads * pairs);
    [encoder setBuffer:key_buffer offset:0 atIndex:0];
    [encoder setBytes:&key_heads length:sizeof(key_heads) atIndex:1];
    [encoder setBuffer:key_buffer offset:0 atIndex:8];
    [encoder setBytes:&frequency_dimension length:sizeof(frequency_dimension) atIndex:9];
    enc_1d(encoder, rope, key_heads * pairs);
    [encoder endEncoding];
    const auto wait_start = std::chrono::steady_clock::now();
    [command commit];
    [command waitUntilCompleted];
    const auto wait_end = std::chrono::steady_clock::now();
    const bool completed = command.status != MTLCommandBufferStatusError;
    if (completed) {
        record_completed_command(command, wait_start, wait_end);
        std::memcpy(query, [query_buffer contents], (size_t)query_count * sizeof(float));
        std::memcpy(key, [key_buffer contents], (size_t)key_count * sizeof(float));
    }
    release();
    return completed;
}

bool metal_tok_session_register_weights(MetalTokSession& session, const void* base, size_t size) {
    MetalTokScope scope(session.token_);
    init();
    return metal_register_mmap(base, size);
}

#if defined(LAPLACE_TESTING)
bool metal_tok_session_register_weights_for_testing(MetalTokSession& session,
                                                    const void* base, size_t size,
                                                    size_t chunk_limit,
                                                    uint32_t fail_after_chunk) {
    MetalTokScope scope(session.token_);
    init();
    return metal_register_mmap_for_testing(base, size, chunk_limit, fail_after_chunk);
}
#endif

void metal_tok_session_unregister_weights(MetalTokSession& session, const void* base) {
    MetalTokScope scope(session.token_);
    metal_unregister_weights(base);
}

bool metal_tok_session_set_sparse_ffn_runs(MetalTokSession& session,
                                           const MetalSparseBlockRun* runs,
                                           uint32_t count) {
    MetalTokScope scope(session.token_);
    if (count == 0) {
        [session.token_.sparse_block_ids release];
        session.token_.sparse_block_ids = nil;
        session.token_.sparse_block_count = 0;
        session.token_.sparse_affine = false;
        return true;
    }
    if (!runs || count > 1024) return false;
    uint64_t end = 0, blocks = 0;
    for (uint32_t index = 0; index < count; ++index) {
        if (runs[index].count == 0 || runs[index].first < end ||
            static_cast<uint64_t>(runs[index].first) + runs[index].count > UINT32_MAX)
            return false;
        end = static_cast<uint64_t>(runs[index].first) + runs[index].count;
        blocks += runs[index].count;
        if (blocks > static_cast<uint64_t>(INT_MAX) / 256) return false;
    }
    std::vector<uint32_t> block_ids;
    block_ids.reserve(static_cast<size_t>(blocks));
    for (uint32_t index = 0; index < count; ++index)
        for (uint32_t offset = 0; offset < runs[index].count; ++offset)
            block_ids.push_back(runs[index].first + offset);
    id<MTLBuffer> table = [g_dev newBufferWithBytes:block_ids.data()
                                             length:block_ids.size() * sizeof(block_ids[0])
                                            options:MTLResourceStorageModeShared];
    if (!table) return false;
    [session.token_.sparse_block_ids release];
    session.token_.sparse_block_ids = table;
    session.token_.sparse_block_count = static_cast<uint32_t>(blocks);
    session.token_.sparse_affine = count == 1;
    return true;
}

bool metal_tok_session_set_sparse_ffn_layer_ids(MetalTokSession& session,
                                                 const uint32_t* ids,
                                                 const uint32_t* offsets,
                                                 const uint32_t* counts,
                                                 uint32_t layer_count) {
    MetalTokScope scope(session.token_);
    if (!ids || !offsets || !counts || layer_count == 0 || layer_count > 1024)
        return false;
    uint64_t total = 0;
    for (uint32_t layer = 0; layer != layer_count; ++layer) {
        if (counts[layer] == 0 || offsets[layer] != total ||
            total > UINT32_MAX - counts[layer]) return false;
        const uint32_t* row = ids + offsets[layer];
        for (uint32_t index = 1; index != counts[layer]; ++index)
            if (row[index] <= row[index - 1]) return false;
        total += counts[layer];
    }
    if (total == 0 || total > SIZE_MAX / sizeof(uint32_t)) return false;
    const size_t count = static_cast<size_t>(total);
    id<MTLBuffer> table = [g_dev newBufferWithBytes:ids
                                             length:count * sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];
    if (!table) return false;
    [session.token_.sparse_block_ids release];
    session.token_.sparse_block_ids = table;
    session.token_.sparse_block_count = 0;
    session.token_.sparse_affine = false;
    return true;
}

bool metal_tok_session_set_sparse_ffn_proxy(MetalTokSession& session, uint32_t slot,
                                            const float* coefficients, uint32_t input_blocks,
                                            uint32_t output_blocks, uint32_t selected_blocks) {
    MetalTokScope scope(session.token_);
    if (!coefficients || slot >= 256 || input_blocks == 0 || output_blocks == 0 ||
        input_blocks > 4096 || output_blocks > 4096 || selected_blocks == 0 ||
        selected_blocks > output_blocks ||
        static_cast<size_t>(output_blocks) > SIZE_MAX / input_blocks ||
        static_cast<size_t>(input_blocks) * output_blocks > SIZE_MAX / sizeof(float)) return false;
    const size_t coefficient_count = static_cast<size_t>(input_blocks) * output_blocks;
    bool any_positive = false;
    for (size_t index = 0; index != coefficient_count; ++index) {
        if (!std::isfinite(coefficients[index]) || coefficients[index] < 0.0f) return false;
        any_positive |= coefficients[index] > 0.0f;
    }
    if (!any_positive) return false;
    id<MTLBuffer> coefficient_buffer = [g_dev newBufferWithBytes:coefficients
                                                          length:coefficient_count * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    if (!coefficient_buffer) return false;
    if (session.token_.sparse_proxies.size() <= slot)
        session.token_.sparse_proxies.resize(static_cast<size_t>(slot) + 1);
    SparseFfnProxyResource& proxy = session.token_.sparse_proxies[slot];
    [proxy.coefficients release];
    proxy = {coefficient_buffer, input_blocks, output_blocks, selected_blocks};
    const size_t score_bytes = static_cast<size_t>(output_blocks) * sizeof(float);
    if (!session.token_.sparse_scores || session.token_.sparse_scores.length < score_bytes) {
        id<MTLBuffer> scores = [g_dev newBufferWithLength:score_bytes
                                                  options:MTLResourceStorageModeShared];
        if (!scores) return false;
        [session.token_.sparse_scores release];
        session.token_.sparse_scores = scores;
    }
    if (!session.token_.sparse_block_ids || session.token_.sparse_block_ids.length < 2 * sizeof(uint32_t)) {
        const uint32_t placeholder[2] = {0, selected_blocks};
        id<MTLBuffer> table = [g_dev newBufferWithBytes:placeholder length:sizeof(placeholder)
                                             options:MTLResourceStorageModeShared];
        if (!table) return false;
        [session.token_.sparse_block_ids release];
        session.token_.sparse_block_ids = table;
    }
    return true;
}

bool metal_tok_session_set_importance_slots(MetalTokSession& session, const uint32_t* widths,
                                            uint32_t slot_count) {
    MetalTokScope scope(session.token_);
    if (g_tok.live || !widths || slot_count == 0 || slot_count > 256 ||
        !named_pipe("activation_importance_accumulate")) return false;
    std::vector<ActivationImportanceResource> candidate(slot_count);
    for (uint32_t slot = 0; slot != slot_count; ++slot) {
        if (widths[slot] == 0 || widths[slot] > (1u << 24) ||
            static_cast<size_t>(widths[slot]) > (SIZE_MAX / sizeof(float)) - 1) {
            for (const ActivationImportanceResource& resource : candidate)
                [resource.values release];
            return false;
        }
        const size_t bytes = (static_cast<size_t>(widths[slot]) + 1) * sizeof(float);
        candidate[slot].values = [g_dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        candidate[slot].width = widths[slot];
        if (!candidate[slot].values) {
            for (const ActivationImportanceResource& resource : candidate)
                [resource.values release];
            return false;
        }
        std::memset([candidate[slot].values contents], 0, bytes);
    }
    for (const ActivationImportanceResource& resource : session.token_.importance)
        [resource.values release];
    session.token_.importance = std::move(candidate);
    return true;
}

bool metal_tok_session_read_importance(const MetalTokSession& session, uint32_t slot,
                                       float* sum_squares, uint32_t width,
                                       uint32_t* sample_count) {
    MetalTokScope scope(const_cast<MetalTokContext&>(session.token_));
    if (!sum_squares || !sample_count || g_tok.live || slot >= session.token_.importance.size())
        return false;
    const ActivationImportanceResource& resource = session.token_.importance[slot];
    if (!resource.values || resource.width != width) return false;
    const uint8_t* bytes = static_cast<const uint8_t*>([resource.values contents]);
    uint32_t count = 0;
    std::memcpy(&count, bytes + static_cast<size_t>(width) * sizeof(float), sizeof(count));
    if (count == 0) return false;
    const float* values = reinterpret_cast<const float*>(bytes);
    for (uint32_t index = 0; index != width; ++index)
        if (!std::isfinite(values[index]) || values[index] < 0.0f) return false;
    std::memcpy(sum_squares, values, static_cast<size_t>(width) * sizeof(float));
    *sample_count = count;
    return true;
}

namespace {

bool encode_activation_importance(id<MTLComputeCommandEncoder> encoder,
                                  id<MTLBuffer> input, size_t input_offset,
                                  const ActivationImportanceResource& resource) {
    id<MTLComputePipelineState> pipeline = named_pipe("activation_importance_accumulate");
    if (!encoder || !input || !resource.values || resource.width == 0 || !pipeline ||
        pipeline.threadExecutionWidth == 0 || pipeline.maxTotalThreadsPerThreadgroup < 64) return false;
    const size_t count_offset = static_cast<size_t>(resource.width) * sizeof(float);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:input_offset atIndex:0];
    [encoder setBuffer:resource.values offset:0 atIndex:1];
    [encoder setBuffer:resource.values offset:count_offset atIndex:2];
    [encoder setBytes:&resource.width length:sizeof(resource.width) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake((resource.width + 63) / 64, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    return true;
}

} // namespace

#if defined(LAPLACE_TESTING)
bool metal_test_activation_importance_accumulator(const float* first, const float* second,
                                                  uint32_t width, float* sums,
                                                  uint32_t* sample_count) {
    if (!first || !second || !sums || !sample_count || width == 0) return false;
    std::shared_ptr<MetalTokSession> session = metal_tok_session_create();
    if (!session || !metal_tok_session_set_importance_slots(*session, &width, 1)) return false;
    id<MTLBuffer> first_buffer = [g_dev newBufferWithBytes:first
                                                   length:static_cast<size_t>(width) * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> second_buffer = [g_dev newBufferWithBytes:second
                                                    length:static_cast<size_t>(width) * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    const bool encoded = first_buffer && second_buffer && command && encoder &&
        encode_activation_importance(encoder, first_buffer, 0, session->token_.importance[0]);
    if (encoded) [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    const bool complete_encoding = encoded &&
        encode_activation_importance(encoder, second_buffer, 0, session->token_.importance[0]);
    if (encoder) [encoder endEncoding];
    bool completed = false;
    if (complete_encoding) {
        [command commit];
        [command waitUntilCompleted];
        completed = command.status == MTLCommandBufferStatusCompleted;
    }
    [first_buffer release];
    [second_buffer release];
    return completed && metal_tok_session_read_importance(*session, 0, sums, width, sample_count);
}
#endif

#if defined(LAPLACE_METAL_TESTING)
bool metal_tok_session_enable_sparse_ffn_dense_oracle_for_testing(MetalTokSession& session,
                                                                  uint32_t slots) {
    MetalTokScope scope(session.token_);
    if (slots == 0 || slots > 256 || session.token_.sparse_proxies.size() < slots ||
        static_cast<size_t>(slots) > SIZE_MAX / (3 * sizeof(uint32_t))) return false;
    const uint32_t selected_blocks = session.token_.sparse_proxies.front().selected_blocks;
    if (selected_blocks == 0) return false;
    for (uint32_t slot = 0; slot != slots; ++slot)
        if (session.token_.sparse_proxies[slot].selected_blocks != selected_blocks) return false;
    id<MTLBuffer> starts = [g_dev newBufferWithLength:static_cast<size_t>(slots) * 3 * sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> ids = [g_dev newBufferWithLength:static_cast<size_t>(selected_blocks) * sizeof(uint32_t)
                                          options:MTLResourceStorageModeShared];
    if (!starts || !ids) {
        [starts release];
        [ids release];
        return false;
    }
    std::memset([starts contents], 0, static_cast<size_t>(slots) * 3 * sizeof(uint32_t));
    [session.token_.sparse_oracle_starts release];
    [session.token_.sparse_block_ids release];
    session.token_.sparse_oracle_starts = starts;
    session.token_.sparse_block_ids = ids;
    session.token_.sparse_oracle_slots = slots;
    return true;
}

bool metal_tok_session_sparse_ffn_windows_for_testing(const MetalTokSession& session,
                                                       uint32_t* starts, uint32_t slots) {
    if (!starts || slots == 0 || slots != session.token_.sparse_oracle_slots ||
        !session.token_.sparse_oracle_starts ||
        session.token_.sparse_oracle_starts.length < static_cast<size_t>(slots) * 3 * sizeof(uint32_t))
        return false;
    std::memcpy(starts, [session.token_.sparse_oracle_starts contents],
                static_cast<size_t>(slots) * 3 * sizeof(uint32_t));
    return true;
}
#endif

#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
bool metal_tok_session_probe_ffn_for_testing(MetalTokSession& session, const float* input,
                                              const Tensor& gate, const Tensor& up,
                                              const Tensor& down,
                                              const MetalSparseBlockRun* runs,
                                              uint32_t run_count, float* output,
                                              int hidden, int full_intermediate,
                                              double* gpu_ms) {
    if (!input || !output || !gpu_ms || hidden <= 0 || full_intermediate <= 0 ||
        (run_count != 0 && !runs)) return false;
    uint64_t selected_blocks = 0;
    for (uint32_t index = 0; index < run_count; ++index) selected_blocks += runs[index].count;
    if (run_count == 0) selected_blocks = static_cast<uint64_t>(full_intermediate) / 256;
    if (selected_blocks == 0 || selected_blocks > static_cast<uint64_t>(INT_MAX) / 256) return false;
    const int packed_intermediate = static_cast<int>(selected_blocks * 256);
    if (!metal_tok_session_set_sparse_ffn_runs(session, runs, run_count) ||
        !metal_tok_session_begin(session, hidden, packed_intermediate, 0, 0, 0,
                                 1, 1, 32, 1, 1, 0) ||
        !metal_tok_session_upload_x(session, input, hidden)) return false;
    MetalTokScope scope(session.token_);
    const bool sparse = run_count != 0;
    const bool gate_ok = sparse
        ? tok_bind_sparse_rows(gate, g_tok.x, g_tok.fg, hidden, packed_intermediate)
        : tok_bind(gate, g_tok.x, g_tok.fg, hidden, full_intermediate);
    const bool up_ok = sparse
        ? tok_bind_sparse_rows(up, g_tok.x, g_tok.fu, hidden, packed_intermediate)
        : tok_bind(up, g_tok.x, g_tok.fu, hidden, full_intermediate);
    if (!gate_ok || !up_ok || !tok_enc() ||
        !enqueue_glu(g_tok.enc, g_tok.ws, g_tok.fg, g_tok.fu, g_tok.fh,
                     packed_intermediate, 1)) {
        metal_tok_abort();
        return false;
    }
    const bool down_ok = sparse
        ? tok_bind_sparse_columns(down, g_tok.fh, g_tok.xb, full_intermediate,
                                  packed_intermediate, hidden)
        : tok_bind(down, g_tok.fh, g_tok.xb, full_intermediate, hidden);
    if (!down_ok || !g_tok.enc || !g_tok.cmd) {
        metal_tok_abort();
        return false;
    }
    [g_tok.enc endEncoding];
    g_tok.enc = nil;
    const auto wait_start = std::chrono::steady_clock::now();
    [g_tok.cmd commit];
    [g_tok.cmd waitUntilCompleted];
    const auto wait_end = std::chrono::steady_clock::now();
    record_completed_command(g_tok.cmd, wait_start, wait_end);
    const bool completed = g_tok.cmd.status == MTLCommandBufferStatusCompleted &&
                           tok_finish_column_grouped_affine_u2_skip();
    const CFTimeInterval start = [g_tok.cmd GPUStartTime];
    const CFTimeInterval end = [g_tok.cmd GPUEndTime];
    *gpu_ms = completed && end >= start ? 1000.0 * (end - start) : 0.0;
    if (completed)
        std::memcpy(output, static_cast<const uint8_t*>([g_tok.ws contents]) + g_tok.xb,
                    static_cast<size_t>(hidden) * sizeof(float));
    g_tok.live = false;
    g_tok.cmd = nil;
    return completed && *gpu_ms > 0.0;
}
#endif

#if defined(LAPLACE_TESTING)
bool metal_test_select_contiguous_window(const float* scores, uint32_t total_blocks,
                                         uint32_t selected_blocks, uint32_t* selected_first,
                                         uint32_t* selected_count, double* gpu_ms) {
    if (!scores || !selected_first || !selected_count || !gpu_ms || total_blocks == 0 ||
        selected_blocks == 0 || selected_blocks > total_blocks || total_blocks > 4096)
        return false;
    std::shared_ptr<MetalTokSession> session = metal_tok_session_create();
    const MetalSparseBlockRun placeholder{0, selected_blocks};
    if (!session || !metal_tok_session_set_sparse_ffn_runs(*session, &placeholder, 1) ||
        !metal_tok_session_begin(*session, 1, static_cast<int>(selected_blocks * 256u),
                                 0, 0, 0, 1, 1, 32, 1, 1, 0)) return false;
    MetalTokScope scope(session->token_);
    id<MTLBuffer> score_buffer = [g_dev newBufferWithBytes:scores
                                                    length:static_cast<size_t>(total_blocks) * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    if (!score_buffer || !tok_select_sparse_window(score_buffer, total_blocks, selected_blocks) ||
        !g_tok.enc || !g_tok.cmd) {
        [score_buffer release];
        metal_tok_abort();
        return false;
    }
    [g_tok.enc endEncoding];
    g_tok.enc = nil;
    const auto wait_start = std::chrono::steady_clock::now();
    [g_tok.cmd commit];
    [g_tok.cmd waitUntilCompleted];
    const auto wait_end = std::chrono::steady_clock::now();
    record_completed_command(g_tok.cmd, wait_start, wait_end);
    const bool completed = g_tok.cmd.status != MTLCommandBufferStatusError;
    const CFTimeInterval start = [g_tok.cmd GPUStartTime];
    const CFTimeInterval end = [g_tok.cmd GPUEndTime];
    *gpu_ms = completed && end >= start ? 1000.0 * (end - start) : 0.0;
    if (completed) {
        const uint32_t* run = static_cast<const uint32_t*>([session->token_.sparse_block_ids contents]);
        *selected_first = run[0];
        *selected_count = run[1];
    }
    [score_buffer release];
    g_tok.live = false;
    g_tok.cmd = nil;
    return completed && *gpu_ms > 0.0;
}

bool metal_test_sparse_ffn_selected_window(const float* input, const Tensor& gate,
                                           const Tensor& up, const Tensor& down,
                                           const float* scores, uint32_t total_blocks,
                                           uint32_t selected_blocks, float* output,
                                           uint32_t* selected_first, int hidden,
                                           int full_intermediate, double* gpu_ms) {
    if (!input || !scores || !output || !selected_first || !gpu_ms || hidden <= 0 ||
        full_intermediate <= 0 || total_blocks != static_cast<uint32_t>(full_intermediate / 256) ||
        selected_blocks == 0 || selected_blocks > total_blocks) return false;
    std::shared_ptr<MetalTokSession> session = metal_tok_session_create();
    const MetalSparseBlockRun placeholder{0, selected_blocks};
    const int packed_intermediate = static_cast<int>(selected_blocks * 256u);
    if (!session || !metal_tok_session_set_sparse_ffn_runs(*session, &placeholder, 1) ||
        !metal_tok_session_begin(*session, hidden, packed_intermediate, 0, 0, 0,
                                 1, 1, 32, 1, 1, 0) ||
        !metal_tok_session_upload_x(*session, input, hidden)) return false;
    MetalTokScope scope(session->token_);
    id<MTLBuffer> score_buffer = [g_dev newBufferWithBytes:scores
                                                    length:static_cast<size_t>(total_blocks) * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    if (!score_buffer || !tok_select_sparse_window(score_buffer, total_blocks, selected_blocks) ||
        !tok_bind_sparse_rows(gate, g_tok.x, g_tok.fg, hidden, packed_intermediate) ||
        !tok_bind_sparse_rows(up, g_tok.x, g_tok.fu, hidden, packed_intermediate) || !tok_enc() ||
        !enqueue_glu(g_tok.enc, g_tok.ws, g_tok.fg, g_tok.fu, g_tok.fh,
                     packed_intermediate, 1) ||
        !tok_bind_sparse_columns(down, g_tok.fh, g_tok.xb, full_intermediate,
                                 packed_intermediate, hidden) || !g_tok.enc || !g_tok.cmd) {
        [score_buffer release];
        metal_tok_abort();
        return false;
    }
    [g_tok.enc endEncoding];
    g_tok.enc = nil;
    const auto wait_start = std::chrono::steady_clock::now();
    [g_tok.cmd commit];
    [g_tok.cmd waitUntilCompleted];
    const auto wait_end = std::chrono::steady_clock::now();
    record_completed_command(g_tok.cmd, wait_start, wait_end);
    const bool completed = g_tok.cmd.status != MTLCommandBufferStatusError;
    const CFTimeInterval start = [g_tok.cmd GPUStartTime];
    const CFTimeInterval end = [g_tok.cmd GPUEndTime];
    *gpu_ms = completed && end >= start ? 1000.0 * (end - start) : 0.0;
    if (completed) {
        std::memcpy(output, static_cast<const uint8_t*>([g_tok.ws contents]) + g_tok.xb,
                    static_cast<size_t>(hidden) * sizeof(float));
        const uint32_t* run = static_cast<const uint32_t*>([session->token_.sparse_block_ids contents]);
        *selected_first = run[0];
    }
    [score_buffer release];
    g_tok.live = false;
    g_tok.cmd = nil;
    return completed && *gpu_ms > 0.0;
}

bool metal_test_sparse_ffn_proxy_window(const float* input, const Tensor& gate,
                                        const Tensor& up, const Tensor& down,
                                        const float* proxy_coefficients,
                                        uint32_t input_blocks, uint32_t output_blocks,
                                        uint32_t selected_blocks, float* output,
                                        uint32_t* selected_first, int hidden,
                                        int full_intermediate, double* gpu_ms) {
    if (!input || !proxy_coefficients || !output || !selected_first || !gpu_ms ||
        hidden <= 0 || full_intermediate <= 0 || input_blocks == 0 || output_blocks == 0 ||
        input_blocks != static_cast<uint32_t>(hidden / 256) ||
        output_blocks != static_cast<uint32_t>(full_intermediate / 256) ||
        hidden % 256 != 0 || full_intermediate % 256 != 0 ||
        selected_blocks == 0 || selected_blocks > output_blocks ||
        static_cast<size_t>(output_blocks) > SIZE_MAX / input_blocks ||
        static_cast<size_t>(input_blocks) * output_blocks > SIZE_MAX / sizeof(float)) return false;
    std::shared_ptr<MetalTokSession> session = metal_tok_session_create();
    const MetalSparseBlockRun placeholder{0, selected_blocks};
    const int packed_intermediate = static_cast<int>(selected_blocks * 256u);
    if (!session || !metal_tok_session_set_sparse_ffn_runs(*session, &placeholder, 1) ||
        !metal_tok_session_set_sparse_ffn_proxy(*session, 0, proxy_coefficients,
                                                input_blocks, output_blocks, selected_blocks) ||
        !metal_tok_session_begin(*session, hidden, packed_intermediate, 0, 0, 0,
                                 1, 1, 32, 1, 1, 0) ||
        !metal_tok_session_upload_x(*session, input, hidden)) return false;
    MetalTokScope scope(session->token_);
    if (!tok_select_sparse_proxy(g_tok.x, 0) ||
        !tok_bind_sparse_rows(gate, g_tok.x, g_tok.fg, hidden, packed_intermediate) ||
        !tok_bind_sparse_rows(up, g_tok.x, g_tok.fu, hidden, packed_intermediate) || !tok_enc() ||
        !enqueue_glu(g_tok.enc, g_tok.ws, g_tok.fg, g_tok.fu, g_tok.fh,
                     packed_intermediate, 1) ||
        !tok_bind_sparse_columns(down, g_tok.fh, g_tok.xb, full_intermediate,
                                 packed_intermediate, hidden) || !g_tok.enc || !g_tok.cmd) {
        metal_tok_abort();
        return false;
    }
    [g_tok.enc endEncoding];
    g_tok.enc = nil;
    const auto wait_start = std::chrono::steady_clock::now();
    [g_tok.cmd commit];
    [g_tok.cmd waitUntilCompleted];
    const auto wait_end = std::chrono::steady_clock::now();
    record_completed_command(g_tok.cmd, wait_start, wait_end);
    const bool completed = g_tok.cmd.status != MTLCommandBufferStatusError;
    const CFTimeInterval start = [g_tok.cmd GPUStartTime];
    const CFTimeInterval end = [g_tok.cmd GPUEndTime];
    *gpu_ms = completed && end >= start ? 1000.0 * (end - start) : 0.0;
    if (completed) {
        std::memcpy(output, static_cast<const uint8_t*>([g_tok.ws contents]) + g_tok.xb,
                    static_cast<size_t>(hidden) * sizeof(float));
        const uint32_t* run = static_cast<const uint32_t*>([session->token_.sparse_block_ids contents]);
        *selected_first = run[0];
    }
    g_tok.live = false;
    g_tok.cmd = nil;
    return completed && *gpu_ms > 0.0;
}

bool metal_test_sparse_ffn_original_spans(const float* input, const Tensor& gate,
                                          const Tensor& up, const Tensor& down,
                                          const MetalSparseBlockRun* runs, uint32_t run_count,
                                          float* output, int hidden, int full_intermediate) {
    if (!input || !output || hidden <= 0 || full_intermediate <= 0 || !runs || run_count == 0)
        return false;
    std::shared_ptr<MetalTokSession> session = metal_tok_session_create();
    double ignored_gpu_ms = 0.0;
    return session && metal_tok_session_probe_ffn_for_testing(
        *session, input, gate, up, down, runs, run_count, output,
        hidden, full_intermediate, &ignored_gpu_ms);
}
#endif

bool metal_tok_session_begin(MetalTokSession& session, int H, int inter, int exp_inter, int n_used, int n_experts,
                             int Hq, int Hk, int Dh, int max_seq, int n_layers, int pos,
                             uint32_t profile_token_count) {
    MetalTokScope scope(session.token_);
    return tok_begin(H, inter, exp_inter, n_used, n_experts, Hq, Hk, Dh, max_seq, n_layers, pos,
                     profile_token_count, false);
}

bool metal_tok_session_begin_with_attention_capacity(
    MetalTokSession& session, int H, int inter, int exp_inter, int n_used, int n_experts,
    MetalTokAttentionCapacity capacity, int max_seq, int n_layers, int pos,
    uint32_t profile_token_count) {
    MetalTokScope scope(session.token_);
    if (capacity.query_width < 1 || capacity.key_value_width < 1) return false;
    return tok_begin(H, inter, exp_inter, n_used, n_experts, 1, 1, 1,
                     max_seq, n_layers, pos, profile_token_count, false, 1,
                     capacity.query_width, capacity.key_value_width,
                     capacity.key_value_width_sum);
}

bool metal_tok_session_begin_prefill_batch(MetalTokSession& session, int H, int inter,
                                           int exp_inter, int n_used, int n_experts,
                                           int Hq, int Hk, int Dh, int max_seq, int n_layers,
                                           int pos, uint32_t rows) {
    MetalTokScope scope(session.token_);
    if (rows != 2) return false;
    return tok_begin(H, inter, exp_inter, n_used, n_experts, Hq, Hk, Dh, max_seq, n_layers, pos,
                     rows, false, rows);
}

bool metal_tok_session_begin_prefill_batch_with_attention_capacity(
    MetalTokSession& session, int H, int inter, int exp_inter, int n_used, int n_experts,
    MetalTokAttentionCapacity capacity, int max_seq, int n_layers, int pos, uint32_t rows) {
    MetalTokScope scope(session.token_);
    if (rows != 2 || capacity.query_width < 1 || capacity.key_value_width < 1) return false;
    return tok_begin(H, inter, exp_inter, n_used, n_experts, 1, 1, 1,
                     max_seq, n_layers, pos, rows, false, rows,
                     capacity.query_width, capacity.key_value_width,
                     capacity.key_value_width_sum);
}

bool metal_tok_session_begin_continuing(MetalTokSession& session, int H, int inter, int exp_inter,
                                        int n_used, int n_experts, int Hq, int Hk, int Dh,
                                        int max_seq, int n_layers, int pos) {
    MetalTokScope scope(session.token_);
    return tok_begin(H, inter, exp_inter, n_used, n_experts, Hq, Hk, Dh, max_seq, n_layers, pos, 1, true);
}

bool metal_tok_session_begin_continuing_with_attention_capacity(
    MetalTokSession& session, int H, int inter, int exp_inter, int n_used, int n_experts,
    MetalTokAttentionCapacity capacity, int max_seq, int n_layers, int pos) {
    MetalTokScope scope(session.token_);
    if (capacity.query_width < 1 || capacity.key_value_width < 1) return false;
    return tok_begin(H, inter, exp_inter, n_used, n_experts, 1, 1, 1,
                     max_seq, n_layers, pos, 1, true, 1,
                     capacity.query_width, capacity.key_value_width,
                     capacity.key_value_width_sum);
}

bool metal_tok_session_upload_embedding(MetalTokSession& session, const Tensor& embedding, uint32_t token,
                                         int H, int vocab, float scale) {
    MetalTokScope scope(session.token_);
    return tok_upload_embedding_to(embedding, token, H, vocab, scale, g_tok.x, true);
}

bool metal_tok_session_upload_embeddings_batch(MetalTokSession& session, const Tensor& embedding,
                                               const uint32_t* tokens, uint32_t rows,
                                               int H, int vocab, float scale) {
    MetalTokScope scope(session.token_);
    if (!tokens || !g_tok.live || rows != 2 || g_tok.batch_rows != rows || H != g_tok.H) return false;
    for (uint32_t row = 0; row != rows; ++row) {
        if (!tok_upload_embedding_to(embedding, tokens[row], H, vocab, scale,
                                     tok_batch_row(g_tok.x, H, row), row == 0)) return false;
    }
    return true;
}

bool metal_tok_session_upload_x(MetalTokSession& session, const float* x, int H) {
    MetalTokScope scope(session.token_);
    if (!g_tok.live || !x || H != g_tok.H) return false;
    metal_tok_upload_x(x, H);
    return true;
}

bool metal_tok_session_layer(MetalTokSession& session, const MetalTokLayer& layer) {
    MetalTokScope scope(session.token_);
    return metal_tok_layer(layer);
}

bool metal_tok_session_dense_prefill_batch_layer(MetalTokSession& session,
                                                  const MetalTokLayer& layer, uint32_t rows) {
    MetalTokScope scope(session.token_);
    return rows == 2 && g_tok.batch_rows == rows && metal_tok_dense_prefill_batch_layer(layer);
}

bool metal_tok_session_select_prefill_batch_row(MetalTokSession& session, uint32_t row) {
    MetalTokScope scope(session.token_);
    if (!g_tok.live || g_tok.batch_rows != 2 || row >= g_tok.batch_rows) return false;
    g_tok.x = tok_batch_row(g_tok.x, g_tok.H, row);
    g_tok.xn = tok_batch_row(g_tok.xn, g_tok.H, row);
    return true;
}

bool metal_tok_session_seal_token(MetalTokSession& session) {
    MetalTokScope scope(session.token_);
    if (!g_tok.live || !session.recurrent_pending_.empty()) return false;
    tok_split();
    g_tok.live = false;
    return g_tok.cmd != nil;
}

bool metal_tok_session_final(MetalTokSession& session, const Tensor& norm, const Tensor& lm, float* logits,
                             int H, int vocab, float eps) {
    MetalTokScope scope(session.token_);
    if (!metal_tok_final(norm, lm, logits, H, vocab, eps)) {
        session.discard_recurrent();
        return false;
    }
#if defined(LAPLACE_METAL_TESTING)
    if (session.token_.fail_after_completed_submission) {
        session.token_.fail_after_completed_submission = false;
        session.token_.failure_detail = "injected post-completion failure";
        session.discard_recurrent();
        return false;
    }
#endif
    session.publish_recurrent();
    return true;
}

bool metal_tok_session_final_sampled(MetalTokSession& session, const Tensor& norm,
                                     const Tensor& lm,
                                     const MetalSamplerDescriptor& descriptor,
                                     MetalSamplerResult* result,
                                     int H, int vocab, float eps) {
    MetalTokScope scope(session.token_);
    if (!metal_tok_final_impl(norm, lm, nullptr, &descriptor, result,
                              H, vocab, eps)) {
        session.discard_recurrent();
        return false;
    }
#if defined(LAPLACE_METAL_TESTING)
    if (session.token_.fail_after_completed_submission) {
        session.token_.fail_after_completed_submission = false;
        session.token_.failure_detail = "injected post-completion failure";
        session.discard_recurrent();
        return false;
    }
#endif
    session.publish_recurrent();
    return true;
}

MetalTokMetrics metal_tok_session_metrics(const MetalTokSession& session) {
    return session.token_.metrics;
}

void metal_tok_session_abort(MetalTokSession& session) {
    MetalTokScope scope(session.token_);
    session.discard_recurrent();
    metal_tok_abort();
}

namespace {

struct RecurrentSizes {
    size_t channels = 0;
    size_t history = 0;
    size_t state = 0;
    size_t output = 0;
};

bool recurrent_sizes(int qk_heads, int value_heads, int head_dimension, int kernel, RecurrentSizes& sizes) {
    if (qk_heads < 1 || value_heads < 1 || value_heads % qk_heads != 0 ||
        qk_heads > INT_MAX / 2 || head_dimension < 1 || head_dimension > 128 ||
        head_dimension % 32 != 0 || kernel < 2) return false;
    const uint64_t channels = static_cast<uint64_t>(head_dimension) *
                              (2ull * static_cast<uint64_t>(qk_heads) + static_cast<uint64_t>(value_heads));
    const uint64_t history = channels * static_cast<uint64_t>(kernel - 1);
    const uint64_t state = static_cast<uint64_t>(value_heads) * head_dimension * head_dimension;
    const uint64_t output = static_cast<uint64_t>(value_heads) * head_dimension;
    if (channels > INT_MAX || channels > SIZE_MAX / sizeof(float) || history > SIZE_MAX / sizeof(float) ||
        state > SIZE_MAX / sizeof(float) || output > SIZE_MAX / sizeof(float)) return false;
    sizes = {static_cast<size_t>(channels), static_cast<size_t>(history), static_cast<size_t>(state),
             static_cast<size_t>(output)};
    return true;
}

bool recurrent_buffer(id<MTLBuffer>& buffer, size_t bytes) {
    if (buffer && [buffer length] == bytes) return true;
    [buffer release];
    buffer = [g_dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    return buffer != nil;
}

bool recurrent_matches(const RecurrentTokenState& state, int qk_heads, int value_heads,
                       int head_dimension, int kernel) {
    return state.seeded && state.qk_heads == qk_heads && state.value_heads == value_heads &&
           state.head_dimension == head_dimension && state.kernel == kernel;
}

bool recurrent_step_buffers(RecurrentTokenState& state, const RecurrentSizes& sizes, int kernel) {
    return recurrent_buffer(state.qkv, sizes.channels * sizeof(float)) &&
           recurrent_buffer(state.conv_weight, sizes.channels * static_cast<size_t>(kernel) * sizeof(float)) &&
           recurrent_buffer(state.gate, sizes.output * sizeof(float)) &&
           recurrent_buffer(state.beta, static_cast<size_t>(state.value_heads) * sizeof(float)) &&
           recurrent_buffer(state.alpha, static_cast<size_t>(state.value_heads) * sizeof(float)) &&
           recurrent_buffer(state.dt_bias, static_cast<size_t>(state.value_heads) * sizeof(float)) &&
           recurrent_buffer(state.decay, static_cast<size_t>(state.value_heads) * sizeof(float)) &&
           recurrent_buffer(state.norm, static_cast<size_t>(state.head_dimension) * sizeof(float)) &&
           recurrent_buffer(state.output, sizes.output * sizeof(float));
}

bool recurrent_prepare_empty(RecurrentTokenState& state, const RecurrentSizes& sizes,
                             int qk_heads, int value_heads, int head_dimension, int kernel) {
    if (state.seeded) return recurrent_matches(state, qk_heads, value_heads, head_dimension, kernel) && !state.pending;
    if (!recurrent_buffer(state.history_current, sizes.history * sizeof(float)) ||
        !recurrent_buffer(state.history_candidate, sizes.history * sizeof(float)) ||
        !recurrent_buffer(state.state_current, sizes.state * sizeof(float)) ||
        !recurrent_buffer(state.state_candidate, sizes.state * sizeof(float))) return false;
    std::memset([state.history_current contents], 0, sizes.history * sizeof(float));
    std::memset([state.history_candidate contents], 0, sizes.history * sizeof(float));
    std::memset([state.state_current contents], 0, sizes.state * sizeof(float));
    std::memset([state.state_candidate contents], 0, sizes.state * sizeof(float));
    state.history_bytes = sizes.history * sizeof(float);
    state.state_bytes = sizes.state * sizeof(float);
    state.qkv_bytes = sizes.channels * sizeof(float);
    state.output_bytes = sizes.output * sizeof(float);
    state.qk_heads = qk_heads;
    state.value_heads = value_heads;
    state.head_dimension = head_dimension;
    state.kernel = kernel;
    state.seeded = true;
    return true;
}

bool recurrent_matrix(const Tensor* tensor, int K, int N) {
    return tensor && tensor->data && tensor->n_dims == 2 && tensor->dims[0] == static_cast<uint64_t>(K) &&
           tensor->dims[1] == static_cast<uint64_t>(N) &&
           (tensor->type == GGMLType::IQ2_XXS || tensor->type == GGMLType::Q2_K ||
            tensor->type == GGMLType::Q4_K ||
            tensor->type == GGMLType::Q6_K ||
            tensor->type == GGMLType::COLUMN_GROUPED_AFFINE_U2_SKIP_256) &&
           K % 256 == 0;
}

bool recurrent_vector(const Tensor* tensor, int n) {
    return tensor && tensor->data && tensor->type == GGMLType::F32 && tensor->n_dims == 1 &&
           tensor->dims[0] == static_cast<uint64_t>(n);
}

const char* recurrent_layer_preflight(const MetalTokRecurrentLayer& layer, int H, int inter, int Dh) {
    RecurrentSizes sizes;
    if (layer.H != H) return "hidden width differs from token session";
    if (!std::isfinite(layer.l2_epsilon) || layer.l2_epsilon < 0.0f ||
        !std::isfinite(layer.rms_epsilon) || layer.rms_epsilon < 0.0f) return "recurrent epsilon is invalid";
    if (!recurrent_sizes(layer.qk_heads, layer.value_heads, layer.head_dimension, layer.kernel, sizes))
        return "recurrent head geometry is unsupported";
    const int channels = static_cast<int>(sizes.channels);
    const int output_width = static_cast<int>(sizes.output);
    if (channels > inter || output_width > inter || layer.ffn_intermediate <= 0 ||
        layer.ffn_intermediate > inter || layer.value_heads > Dh) return "recurrent workspace geometry is unsupported";
    if (!layer.input_norm) return "recurrent input norm is absent";
    if (!layer.input_norm->data) return "recurrent input norm has no artifact span";
    if (layer.input_norm->type != GGMLType::F32) return "recurrent input norm storage is not FP32";
    if (layer.input_norm->n_dims != 1) return "recurrent input norm rank is not one";
    if (layer.input_norm->dims[0] != static_cast<uint64_t>(layer.H)) return "recurrent input norm width differs from hidden width";
    if (!recurrent_matrix(layer.qkv, layer.H, channels)) return "recurrent QKV matrix physical contract is unsupported";
    if (!recurrent_matrix(layer.gate, layer.H, output_width)) return "recurrent gate matrix physical contract is unsupported";
    if (!recurrent_matrix(layer.beta, layer.H, layer.value_heads)) return "recurrent beta matrix physical contract is unsupported";
    if (!recurrent_matrix(layer.alpha, layer.H, layer.value_heads)) return "recurrent alpha matrix physical contract is unsupported";
    if (!recurrent_matrix(layer.output, output_width, layer.H)) return "recurrent output matrix physical contract is unsupported";
    if (!recurrent_vector(layer.ffn_norm, layer.H)) return "recurrent FFN norm is not an exact FP32 vector";
    const int source_intermediate = layer.sparse_ffn ? layer.sparse_ffn_full_intermediate
                                                     : layer.ffn_intermediate;
    if (source_intermediate < layer.ffn_intermediate || source_intermediate % 256 != 0)
        return "recurrent sparse FFN geometry is unsupported";
    if (!recurrent_matrix(layer.ffn_gate, layer.H, source_intermediate)) return "recurrent FFN gate matrix physical contract is unsupported";
    if (!recurrent_matrix(layer.ffn_up, layer.H, source_intermediate)) return "recurrent FFN up matrix physical contract is unsupported";
    if (!recurrent_matrix(layer.ffn_down, source_intermediate, layer.H)) return "recurrent FFN down matrix physical contract is unsupported";
    if (!layer.conv || !layer.conv->data || layer.conv->type != GGMLType::F32 || layer.conv->n_dims != 2 ||
        layer.conv->dims[0] != static_cast<uint64_t>(layer.kernel) ||
        layer.conv->dims[1] != static_cast<uint64_t>(channels)) return "recurrent convolution physical contract is unsupported";
    if (!recurrent_vector(layer.dt_bias, layer.value_heads)) return "recurrent time-step bias is not an exact FP32 vector";
    if (!recurrent_vector(layer.decay, layer.value_heads)) return "recurrent decay is not an exact FP32 vector";
    if (!recurrent_vector(layer.norm, layer.head_dimension)) return "recurrent norm is not an exact FP32 vector";
    return nullptr;
}

} // namespace

#if defined(LAPLACE_METAL_TESTING)
const char* metal_tok_recurrent_layer_preflight_for_testing(const MetalTokRecurrentLayer& layer,
                                                            int H, int inter, int Dh) {
    return recurrent_layer_preflight(layer, H, inter, Dh);
}
#endif

bool metal_tok_session_recurrent_layer(MetalTokSession& session, const MetalTokRecurrentLayer& layer) {
    MetalTokScope scope(session.token_);
    init();
    RecurrentSizes sizes;
    if (!g_dev || !session.token_.queue || !g_lib || !g_tok.live) {
        session.token_.failure_detail = "recurrent Metal session resources are unavailable";
        return false;
    }
    if (const char* reason = recurrent_layer_preflight(layer, g_tok.H, g_tok.inter, g_tok.H)) {
        session.token_.failure_detail = reason;
        return false;
    }
    if (!recurrent_sizes(layer.qk_heads, layer.value_heads, layer.head_dimension, layer.kernel, sizes)) {
        session.token_.failure_detail = "recurrent state geometry is unsupported";
        return false;
    }
    const int channels = static_cast<int>(sizes.channels);
    const int output_width = static_cast<int>(sizes.output);
    RecurrentTokenState* recurrent = session.recurrent_slot(layer.state_slot);
    if (!recurrent || !recurrent_prepare_empty(*recurrent, sizes, layer.qk_heads, layer.value_heads,
                                               layer.head_dimension, layer.kernel)) return false;
    if (!tok_rms(*layer.input_norm, g_tok.x, g_tok.xn, layer.H, layer.rms_epsilon) ||
        !tok_bind(*layer.qkv, g_tok.xn, g_tok.fg, layer.H, channels) ||
        !tok_bind(*layer.gate, g_tok.xn, g_tok.fu, layer.H, output_width) ||
        !tok_bind(*layer.beta, g_tok.xn, g_tok.tmp, layer.H, layer.value_heads) ||
        !tok_bind(*layer.alpha, g_tok.xn, g_tok.ones, layer.H, layer.value_heads) ||
        !tok_profile_mark(TokProfileSegment::Qkv) || !tok_enc()) return false;
    size_t conv_offset = 0, dt_offset = 0, decay_offset = 0, norm_offset = 0;
    id<MTLBuffer> conv = get_weight_buf(layer.conv->data, sizes.channels * static_cast<size_t>(layer.kernel) * sizeof(float), conv_offset);
    id<MTLBuffer> dt = get_weight_buf(layer.dt_bias->data, static_cast<size_t>(layer.value_heads) * sizeof(float), dt_offset);
    id<MTLBuffer> decay = get_weight_buf(layer.decay->data, static_cast<size_t>(layer.value_heads) * sizeof(float), decay_offset);
    id<MTLBuffer> norm = get_weight_buf(layer.norm->data, static_cast<size_t>(layer.head_dimension) * sizeof(float), norm_offset);
    id<MTLComputePipelineState> conv_pipe = named_pipe("dnet_conv_silu");
    id<MTLComputePipelineState> l2_pipe = named_pipe("dnet_l2_qk");
    id<MTLComputePipelineState> update_pipe = named_pipe("dnet_update");
    if (!conv || !dt || !decay || !norm || !conv_pipe || !l2_pipe || !update_pipe) return false;
    [g_tok.enc setComputePipelineState:conv_pipe];
    [g_tok.enc setBuffer:g_tok.ws offset:g_tok.fg atIndex:0];
    [g_tok.enc setBuffer:conv offset:conv_offset atIndex:1];
    [g_tok.enc setBuffer:recurrent->history_current offset:0 atIndex:2];
    [g_tok.enc setBuffer:g_tok.ws offset:g_tok.fg atIndex:3];
    [g_tok.enc setBuffer:recurrent->history_candidate offset:0 atIndex:4];
    [g_tok.enc setBytes:&channels length:sizeof(channels) atIndex:5];
    [g_tok.enc setBytes:&layer.kernel length:sizeof(layer.kernel) atIndex:6];
    [g_tok.enc dispatchThreadgroups:MTLSizeMake((channels + 63) / 64, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [g_tok.enc setComputePipelineState:l2_pipe];
    [g_tok.enc setBuffer:g_tok.ws offset:g_tok.fg atIndex:0];
    [g_tok.enc setBytes:&layer.qk_heads length:sizeof(layer.qk_heads) atIndex:1];
    [g_tok.enc setBytes:&layer.head_dimension length:sizeof(layer.head_dimension) atIndex:2];
    [g_tok.enc setBytes:&layer.l2_epsilon length:sizeof(layer.l2_epsilon) atIndex:3];
    [g_tok.enc dispatchThreadgroups:MTLSizeMake(2 * layer.qk_heads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [g_tok.enc setComputePipelineState:update_pipe];
    [g_tok.enc setBuffer:g_tok.ws offset:g_tok.fg atIndex:0];
    [g_tok.enc setBuffer:g_tok.ws offset:g_tok.fu atIndex:1];
    [g_tok.enc setBuffer:g_tok.ws offset:g_tok.tmp atIndex:2];
    [g_tok.enc setBuffer:g_tok.ws offset:g_tok.ones atIndex:3];
    [g_tok.enc setBuffer:dt offset:dt_offset atIndex:4];
    [g_tok.enc setBuffer:decay offset:decay_offset atIndex:5];
    [g_tok.enc setBuffer:norm offset:norm_offset atIndex:6];
    [g_tok.enc setBuffer:recurrent->state_current offset:0 atIndex:7];
    [g_tok.enc setBuffer:recurrent->state_candidate offset:0 atIndex:8];
    [g_tok.enc setBuffer:g_tok.ws offset:g_tok.fh atIndex:9];
    [g_tok.enc setBytes:&layer.qk_heads length:sizeof(layer.qk_heads) atIndex:10];
    [g_tok.enc setBytes:&layer.value_heads length:sizeof(layer.value_heads) atIndex:11];
    [g_tok.enc setBytes:&layer.head_dimension length:sizeof(layer.head_dimension) atIndex:12];
    [g_tok.enc setBytes:&layer.rms_epsilon length:sizeof(layer.rms_epsilon) atIndex:13];
    [g_tok.enc dispatchThreadgroups:MTLSizeMake(layer.value_heads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    if (!tok_bind(*layer.output, g_tok.fh, g_tok.xb, output_width, layer.H) ||
        !tok_add(g_tok.x, g_tok.xb, g_tok.x, layer.H) ||
        !tok_profile_mark(TokProfileSegment::Attention) ||
        !tok_rms(*layer.ffn_norm, g_tok.x, g_tok.xn, layer.H, layer.rms_epsilon)) return false;
    if (!tok_accumulate_importance(layer.ffn_input_importance_slot, g_tok.xn,
                                   static_cast<uint32_t>(layer.H))) return false;
    if (layer.sparse_ffn) {
#if defined(LAPLACE_METAL_TESTING)
        const bool selected = layer.sparse_ffn_dense_oracle
            ? layer.sparse_ffn_proxy_slot != UINT32_MAX &&
              tok_select_sparse_dense_oracle(*layer.ffn_gate, *layer.ffn_up, *layer.ffn_down, g_tok.xn,
                                             layer.sparse_ffn_proxy_slot, layer.H,
                                             layer.sparse_ffn_full_intermediate,
                                             layer.ffn_intermediate)
            : ((layer.sparse_ffn_proxy_slot == UINT32_MAX ||
                tok_select_sparse_proxy(g_tok.xn, layer.sparse_ffn_proxy_slot)) &&
               layer.sparse_ffn_full_intermediate >= layer.ffn_intermediate &&
               tok_bind_sparse_rows(*layer.ffn_gate, g_tok.xn, g_tok.fg,
                                    layer.H, layer.ffn_intermediate,
                                    layer.sparse_ffn_block_offset) &&
               tok_bind_sparse_rows(*layer.ffn_up, g_tok.xn, g_tok.fu,
                                    layer.H, layer.ffn_intermediate,
                                    layer.sparse_ffn_block_offset));
#else
        const bool selected = (layer.sparse_ffn_proxy_slot == UINT32_MAX ||
                               tok_select_sparse_proxy(g_tok.xn, layer.sparse_ffn_proxy_slot)) &&
                              layer.sparse_ffn_full_intermediate >= layer.ffn_intermediate &&
                              tok_bind_sparse_rows(*layer.ffn_gate, g_tok.xn, g_tok.fg,
                                                   layer.H, layer.ffn_intermediate,
                                                   layer.sparse_ffn_block_offset) &&
                              tok_bind_sparse_rows(*layer.ffn_up, g_tok.xn, g_tok.fu,
                                                   layer.H, layer.ffn_intermediate,
                                                   layer.sparse_ffn_block_offset);
#endif
        if (!selected ||
            !enqueue_glu(g_tok.enc, g_tok.ws, g_tok.fg, g_tok.fu, g_tok.fh,
                         layer.ffn_intermediate, 1) ||
            !tok_accumulate_importance(layer.ffn_down_importance_slot, g_tok.fh,
                                       static_cast<uint32_t>(layer.ffn_intermediate)) ||
            !tok_bind_sparse_columns(*layer.ffn_down, g_tok.fh, g_tok.xb,
                                     layer.sparse_ffn_full_intermediate,
                                     layer.ffn_intermediate, layer.H,
                                     layer.sparse_ffn_block_offset)) return false;
    } else if (!tok_bind(*layer.ffn_gate, g_tok.xn, g_tok.fg, layer.H, layer.ffn_intermediate) ||
               !tok_bind(*layer.ffn_up, g_tok.xn, g_tok.fu, layer.H, layer.ffn_intermediate) ||
               !enqueue_glu(g_tok.enc, g_tok.ws, g_tok.fg, g_tok.fu, g_tok.fh,
                            layer.ffn_intermediate, 1) ||
               !tok_accumulate_importance(layer.ffn_down_importance_slot, g_tok.fh,
                                          static_cast<uint32_t>(layer.ffn_intermediate)) ||
               !tok_bind(*layer.ffn_down, g_tok.fh, g_tok.xb,
                         layer.ffn_intermediate, layer.H)) return false;
    if (!tok_add(g_tok.x, g_tok.xb, g_tok.x, layer.H) ||
        !tok_profile_mark(TokProfileSegment::Ffn)) return false;
    recurrent->pending = true;
    session.mark_recurrent_pending(layer.state_slot);
    return true;
}

bool metal_tok_session_recurrent_commit(MetalTokSession& session) {
    if (!session.token_.tok.live || session.recurrent_pending_.empty()) return false;
    return metal_tok_session_commit_token(session);
}

bool metal_tok_session_commit_token(MetalTokSession& session) {
    MetalTokScope scope(session.token_);
    if (!g_tok.live) return false;
    tok_split();
    if (!g_tok.cmd) {
        session.discard_recurrent();
        g_tok.live = false;
        return false;
    }
    const auto wait_start = std::chrono::steady_clock::now();
    [g_tok.cmd commit];
    [g_tok.cmd waitUntilCompleted];
    const auto wait_end = std::chrono::steady_clock::now();
    record_completed_command(g_tok.cmd, wait_start, wait_end);
    const bool completed = g_tok.cmd.status == MTLCommandBufferStatusCompleted;
    if (!completed) session.token_.failure_detail = command_buffer_failure_detail(g_tok.cmd);
    MetalTokMetrics& metrics = session.token_.metrics;
    metrics.cpu_wait_ms = std::chrono::duration<double, std::milli>(wait_end - wait_start).count();
    const CFTimeInterval gpu_start = [g_tok.cmd GPUStartTime];
    const CFTimeInterval gpu_end = [g_tok.cmd GPUEndTime];
    metrics.gpu_time_ms = gpu_end >= gpu_start ? 1000.0 * (gpu_end - gpu_start) : 0.0;
    uint64_t importance_bytes = 0;
    for (const ActivationImportanceResource& resource : session.token_.importance)
        if (resource.values) importance_bytes += [resource.values length];
    metrics.peak_session_bytes = session.weights_.byte_count() + g_tok.bytes +
                                 2 * g_kv_bytes + importance_bytes;
    metrics.kv_cache_bytes = 2 * static_cast<uint64_t>(g_kv_bytes);
    g_tok.live = false;
    g_tok.cmd = nil;
    g_tok.enc = nil;
    if (!completed) {
        session.discard_recurrent();
        return false;
    }
    if (!tok_finish_column_grouped_affine_u2_skip()) {
        session.discard_recurrent();
        return false;
    }
#if defined(LAPLACE_TESTING)
    bool injected_failure = false;
    for (const std::unique_ptr<RecurrentTokenState>& recurrent : session.recurrent_) {
        if (!recurrent || !recurrent->fail_after_completed_submission) continue;
        recurrent->fail_after_completed_submission = false;
        injected_failure |= recurrent->pending;
    }
    if (injected_failure) {
        session.discard_recurrent();
        return false;
    }
#endif
    session.publish_recurrent();
    return true;
}

bool metal_tok_session_recurrent_seed(MetalTokSession& session, const float* history, const float* state,
                                      int qk_heads, int value_heads, int head_dimension, int kernel) {
    MetalTokScope scope(session.token_);
    init();
    RecurrentSizes sizes;
    if (!g_dev || !history || !state || !recurrent_sizes(qk_heads, value_heads, head_dimension, kernel, sizes)) return false;
    RecurrentTokenState* recurrent = session.recurrent_slot(0);
    if (!recurrent) return false;
    if (!recurrent_prepare_empty(*recurrent, sizes, qk_heads, value_heads, head_dimension, kernel)) return false;
    std::memcpy([recurrent->history_current contents], history, sizes.history * sizeof(float));
    std::memcpy([recurrent->state_current contents], state, sizes.state * sizeof(float));
    return true;
}

bool metal_tok_session_recurrent_step(MetalTokSession& session, const float* qkv, const float* conv_weight,
                                      const float* gate, const float* beta, const float* alpha,
                                      const float* dt_bias, const float* decay, const float* norm,
                                      float* output, int qk_heads, int value_heads, int head_dimension,
                                      int kernel, float l2_epsilon, float rms_epsilon) {
    MetalTokScope scope(session.token_);
    init();
    RecurrentSizes sizes;
    if (!g_dev || !session.token_.queue || !g_lib || !qkv || !conv_weight || !gate || !beta || !alpha || !dt_bias || !decay ||
        !norm || !output || !std::isfinite(l2_epsilon) || l2_epsilon < 0.0f ||
        !std::isfinite(rms_epsilon) || rms_epsilon < 0.0f ||
        !recurrent_sizes(qk_heads, value_heads, head_dimension, kernel, sizes)) return false;
    RecurrentTokenState* recurrent = session.recurrent_slot(0);
    if (!recurrent || !recurrent_matches(*recurrent, qk_heads, value_heads, head_dimension, kernel) ||
        recurrent->pending || !recurrent_step_buffers(*recurrent, sizes, kernel)) return false;
    id<MTLComputePipelineState> conv_pipe = named_pipe("dnet_conv_silu");
    id<MTLComputePipelineState> l2_pipe = named_pipe("dnet_l2_qk");
    id<MTLComputePipelineState> update_pipe = named_pipe("dnet_update");
    if (!conv_pipe || !l2_pipe || !update_pipe) return false;
    std::memcpy([recurrent->qkv contents], qkv, sizes.channels * sizeof(float));
    std::memcpy([recurrent->conv_weight contents], conv_weight, sizes.channels * static_cast<size_t>(kernel) * sizeof(float));
    std::memcpy([recurrent->gate contents], gate, sizes.output * sizeof(float));
    std::memcpy([recurrent->beta contents], beta, static_cast<size_t>(value_heads) * sizeof(float));
    std::memcpy([recurrent->alpha contents], alpha, static_cast<size_t>(value_heads) * sizeof(float));
    std::memcpy([recurrent->dt_bias contents], dt_bias, static_cast<size_t>(value_heads) * sizeof(float));
    std::memcpy([recurrent->decay contents], decay, static_cast<size_t>(value_heads) * sizeof(float));
    std::memcpy([recurrent->norm contents], norm, static_cast<size_t>(head_dimension) * sizeof(float));
    id<MTLCommandBuffer> command = [session.token_.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder) return false;
    const int channels = static_cast<int>(sizes.channels);
    [encoder setComputePipelineState:conv_pipe];
    [encoder setBuffer:recurrent->qkv offset:0 atIndex:0];
    [encoder setBuffer:recurrent->conv_weight offset:0 atIndex:1];
    [encoder setBuffer:recurrent->history_current offset:0 atIndex:2];
    [encoder setBuffer:recurrent->qkv offset:0 atIndex:3];
    [encoder setBuffer:recurrent->history_candidate offset:0 atIndex:4];
    [encoder setBytes:&channels length:sizeof(channels) atIndex:5];
    [encoder setBytes:&kernel length:sizeof(kernel) atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake((channels + 63) / 64, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder setComputePipelineState:l2_pipe];
    [encoder setBuffer:recurrent->qkv offset:0 atIndex:0];
    [encoder setBytes:&qk_heads length:sizeof(qk_heads) atIndex:1];
    [encoder setBytes:&head_dimension length:sizeof(head_dimension) atIndex:2];
    [encoder setBytes:&l2_epsilon length:sizeof(l2_epsilon) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(2 * qk_heads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder setComputePipelineState:update_pipe];
    [encoder setBuffer:recurrent->qkv offset:0 atIndex:0];
    [encoder setBuffer:recurrent->gate offset:0 atIndex:1];
    [encoder setBuffer:recurrent->beta offset:0 atIndex:2];
    [encoder setBuffer:recurrent->alpha offset:0 atIndex:3];
    [encoder setBuffer:recurrent->dt_bias offset:0 atIndex:4];
    [encoder setBuffer:recurrent->decay offset:0 atIndex:5];
    [encoder setBuffer:recurrent->norm offset:0 atIndex:6];
    [encoder setBuffer:recurrent->state_current offset:0 atIndex:7];
    [encoder setBuffer:recurrent->state_candidate offset:0 atIndex:8];
    [encoder setBuffer:recurrent->output offset:0 atIndex:9];
    [encoder setBytes:&qk_heads length:sizeof(qk_heads) atIndex:10];
    [encoder setBytes:&value_heads length:sizeof(value_heads) atIndex:11];
    [encoder setBytes:&head_dimension length:sizeof(head_dimension) atIndex:12];
    [encoder setBytes:&rms_epsilon length:sizeof(rms_epsilon) atIndex:13];
    [encoder dispatchThreadgroups:MTLSizeMake(value_heads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    const auto wait_start = std::chrono::steady_clock::now();
    [command commit];
    [command waitUntilCompleted];
    const auto wait_end = std::chrono::steady_clock::now();
    record_completed_command(command, wait_start, wait_end);
    if (command.status == MTLCommandBufferStatusError) return false;
#if defined(LAPLACE_TESTING)
    if (recurrent->fail_after_completed_submission) {
        recurrent->fail_after_completed_submission = false;
        return false;
    }
#endif
    std::swap(recurrent->history_current, recurrent->history_candidate);
    std::swap(recurrent->state_current, recurrent->state_candidate);
    std::memcpy(output, [recurrent->output contents], sizes.output * sizeof(float));
    return true;
}

bool metal_tok_session_recurrent_snapshot(const MetalTokSession& session, float* history, float* state,
                                          int qk_heads, int value_heads, int head_dimension, int kernel) {
    return metal_tok_session_recurrent_snapshot_slot(session, 0, history, state,
                                                      qk_heads, value_heads, head_dimension, kernel);
}

bool metal_tok_session_recurrent_snapshot_slot(const MetalTokSession& session, uint32_t state_slot,
                                               float* history, float* state, int qk_heads, int value_heads,
                                               int head_dimension, int kernel) {
    RecurrentSizes sizes;
    const RecurrentTokenState* recurrent = session.recurrent_slot(state_slot);
    if (!history || !state || !recurrent_sizes(qk_heads, value_heads, head_dimension, kernel, sizes) ||
        !recurrent || !recurrent_matches(*recurrent, qk_heads, value_heads, head_dimension, kernel) ||
        recurrent->pending) return false;
    std::memcpy(history, [recurrent->history_current contents], sizes.history * sizeof(float));
    std::memcpy(state, [recurrent->state_current contents], sizes.state * sizeof(float));
    return true;
}

#if defined(LAPLACE_TESTING)
void metal_tok_session_recurrent_fail_after_completed_submission_for_testing(MetalTokSession& session) {
    for (const std::unique_ptr<RecurrentTokenState>& recurrent : session.recurrent_) {
        if (recurrent) recurrent->fail_after_completed_submission = true;
    }
}

bool metal_tok_session_download_x_for_testing(const MetalTokSession& session, float* x, int H) {
    MetalTokScope scope(const_cast<MetalTokContext&>(session.token_));
    if (!x || H != g_tok.H || !g_tok.ws) return false;
    std::memcpy(x, static_cast<const uint8_t*>([g_tok.ws contents]) + g_tok.x,
                static_cast<size_t>(H) * sizeof(float));
    return true;
}
#endif

#if defined(LAPLACE_METAL_TESTING)
void metal_tok_session_fail_after_completed_submission_for_testing(MetalTokSession& session) {
    session.token_.fail_after_completed_submission = true;
}
#endif

bool metal_test_attn(const float* Q, const float* Kc, const float* Vc,
                     float* out, int Hq, int Hk, int Dh, int pos,
                     int window, int max_seq, float scale, int kn_stride) {
    init();
    if (!g_dev || !g_lib || !Q || !Kc || !Vc || !out) return false;
    if (Hq < 1 || Hk < 1 || Dh < 1 || pos < 0 || max_seq < 1) return false;
    if (pos >= max_seq) return false;
    const int kn = Hk * Dh;
    if (kn_stride < kn) kn_stride = kn;
    const int qn = Hq * Dh;
    const size_t qbytes = (size_t)qn * 4;
    const size_t kbytes = (size_t)max_seq * (size_t)kn_stride * 4;
    id<MTLBuffer> qb = [g_dev newBufferWithBytes:Q length:qbytes
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> kb = [g_dev newBufferWithBytes:Kc length:kbytes
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> vb = [g_dev newBufferWithBytes:Vc length:kbytes
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> ob = [g_dev newBufferWithLength:qbytes
                                         options:MTLResourceStorageModeShared];
    auto p = named_pipe("attn_decode");
    if (!qb || !kb || !vb || !ob || !p) return false;
    id<MTLCommandBuffer> cmd = [g_q commandBuffer];
    if (!cmd) return false;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    uint64_t cache_base = 0;
    [enc setComputePipelineState:p];
    [enc setBuffer:qb offset:0 atIndex:0];
    [enc setBuffer:kb offset:0 atIndex:1];
    [enc setBuffer:vb offset:0 atIndex:2];
    [enc setBuffer:ob offset:0 atIndex:3];
    [enc setBytes:&Hq length:4 atIndex:4];
    [enc setBytes:&Hk length:4 atIndex:5];
    [enc setBytes:&Dh length:4 atIndex:6];
    [enc setBytes:&pos length:4 atIndex:7];
    [enc setBytes:&window length:4 atIndex:8];
    [enc setBytes:&max_seq length:4 atIndex:9];
    [enc setBytes:&cache_base length:sizeof(cache_base) atIndex:10];
    [enc setBytes:&scale length:4 atIndex:11];
    [enc setBytes:&kn_stride length:4 atIndex:12];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)Hq, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status == MTLCommandBufferStatusError) return false;
    memcpy(out, [ob contents], qbytes);
    return true;
}

bool metal_test_gated_delta_net(const float* qkv_input, const float* conv_weight,
                                const float* conv_history_input, const float* gate,
                                const float* beta_projection, const float* alpha_projection,
                                const float* dt_bias, const float* decay, const float* norm_weight,
                                const float* state_input, float* conv_history_output,
                                float* state_output, float* output,
                                int qk_heads, int value_heads, int head_dimension, int kernel,
                                float l2_epsilon, float rms_epsilon) {
    init();
    if (!g_dev || !g_q || !g_lib || !qkv_input || !conv_weight || !conv_history_input || !gate ||
        !beta_projection || !alpha_projection || !dt_bias || !decay || !norm_weight || !state_input ||
        !conv_history_output || !state_output || !output || qk_heads < 1 || value_heads < 1 ||
        value_heads % qk_heads != 0 || head_dimension < 1 || head_dimension > 128 ||
        head_dimension % 32 != 0 || kernel < 2 || !std::isfinite(l2_epsilon) || l2_epsilon < 0.0f ||
        !std::isfinite(rms_epsilon) || rms_epsilon < 0.0f) return false;
    const uint64_t channel_count = static_cast<uint64_t>(head_dimension) *
                                   (2ull * static_cast<uint64_t>(qk_heads) + value_heads);
    const uint64_t history_count = channel_count * static_cast<uint64_t>(kernel - 1);
    const uint64_t state_count = static_cast<uint64_t>(value_heads) * head_dimension * head_dimension;
    const uint64_t output_count = static_cast<uint64_t>(value_heads) * head_dimension;
    if (channel_count > INT_MAX || history_count > SIZE_MAX / sizeof(float) ||
        state_count > SIZE_MAX / sizeof(float) || output_count > SIZE_MAX / sizeof(float)) return false;
    const size_t qkv_bytes = static_cast<size_t>(channel_count) * sizeof(float);
    const size_t history_bytes = static_cast<size_t>(history_count) * sizeof(float);
    const size_t state_bytes = static_cast<size_t>(state_count) * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(output_count) * sizeof(float);
    @autoreleasepool {
        id<MTLBuffer> qkv = [g_dev newBufferWithBytes:qkv_input length:qkv_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> conv_weight_buffer = [g_dev newBufferWithBytes:conv_weight length:qkv_bytes * kernel
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> history_in = [g_dev newBufferWithBytes:conv_history_input length:history_bytes
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> history_out = [g_dev newBufferWithLength:history_bytes
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> gate_buffer = [g_dev newBufferWithBytes:gate length:output_bytes
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> beta_buffer = [g_dev newBufferWithBytes:beta_projection length:(size_t)value_heads * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> alpha_buffer = [g_dev newBufferWithBytes:alpha_projection length:(size_t)value_heads * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> dt_buffer = [g_dev newBufferWithBytes:dt_bias length:(size_t)value_heads * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> decay_buffer = [g_dev newBufferWithBytes:decay length:(size_t)value_heads * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> norm_buffer = [g_dev newBufferWithBytes:norm_weight length:(size_t)head_dimension * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> state_in = [g_dev newBufferWithBytes:state_input length:state_bytes
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> state_out = [g_dev newBufferWithLength:state_bytes
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> result = [g_dev newBufferWithLength:output_bytes
                                                  options:MTLResourceStorageModeShared];
        id<MTLComputePipelineState> conv_pipe = named_pipe("dnet_conv_silu");
        id<MTLComputePipelineState> l2_pipe = named_pipe("dnet_l2_qk");
        id<MTLComputePipelineState> update_pipe = named_pipe("dnet_update");
        if (!qkv || !conv_weight_buffer || !history_in || !history_out || !gate_buffer || !beta_buffer ||
            !alpha_buffer || !dt_buffer || !decay_buffer || !norm_buffer || !state_in || !state_out || !result ||
            !conv_pipe || !l2_pipe || !update_pipe) return false;
        id<MTLCommandBuffer> command = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return false;
        const int channels = static_cast<int>(channel_count);
        [encoder setComputePipelineState:conv_pipe];
        [encoder setBuffer:qkv offset:0 atIndex:0];
        [encoder setBuffer:conv_weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:history_in offset:0 atIndex:2];
        [encoder setBuffer:qkv offset:0 atIndex:3];
        [encoder setBuffer:history_out offset:0 atIndex:4];
        [encoder setBytes:&channels length:sizeof(channels) atIndex:5];
        [encoder setBytes:&kernel length:sizeof(kernel) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake((channels + 63) / 64, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [encoder setComputePipelineState:l2_pipe];
        [encoder setBuffer:qkv offset:0 atIndex:0];
        [encoder setBytes:&qk_heads length:sizeof(qk_heads) atIndex:1];
        [encoder setBytes:&head_dimension length:sizeof(head_dimension) atIndex:2];
        [encoder setBytes:&l2_epsilon length:sizeof(l2_epsilon) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(2 * qk_heads, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder setComputePipelineState:update_pipe];
        [encoder setBuffer:qkv offset:0 atIndex:0];
        [encoder setBuffer:gate_buffer offset:0 atIndex:1];
        [encoder setBuffer:beta_buffer offset:0 atIndex:2];
        [encoder setBuffer:alpha_buffer offset:0 atIndex:3];
        [encoder setBuffer:dt_buffer offset:0 atIndex:4];
        [encoder setBuffer:decay_buffer offset:0 atIndex:5];
        [encoder setBuffer:norm_buffer offset:0 atIndex:6];
        [encoder setBuffer:state_in offset:0 atIndex:7];
        [encoder setBuffer:state_out offset:0 atIndex:8];
        [encoder setBuffer:result offset:0 atIndex:9];
        [encoder setBytes:&qk_heads length:sizeof(qk_heads) atIndex:10];
        [encoder setBytes:&value_heads length:sizeof(value_heads) atIndex:11];
        [encoder setBytes:&head_dimension length:sizeof(head_dimension) atIndex:12];
        [encoder setBytes:&rms_epsilon length:sizeof(rms_epsilon) atIndex:13];
        [encoder dispatchThreadgroups:MTLSizeMake(value_heads, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];
        const auto wait_start = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        record_completed_command(command, wait_start, std::chrono::steady_clock::now());
        if (command.status == MTLCommandBufferStatusError) return false;
        std::memcpy(conv_history_output, [history_out contents], history_bytes);
        std::memcpy(state_output, [state_out contents], state_bytes);
        std::memcpy(output, [result contents], output_bytes);
        return true;
    }
}

namespace {
class MetalTokenGraphBackend final : public TokenGraphBackend {
public:
    bool available() const override { return matmul_gpu_available(); }
    bool begin(int H, int inter, int exp_inter, int n_used, int n_experts,
               int Hq, int Hk, int Dh, int max_seq, int n_layers,
               int pos) override {
        return metal_tok_begin(H, inter, exp_inter, n_used, n_experts,
                               Hq, Hk, Dh, max_seq, n_layers, pos);
    }
    bool active() const override { return metal_tok_active(); }
    void upload_x(const float* x, int H) override { metal_tok_upload_x(x, H); }
    bool layer(const MetalTokLayer& layer) override { return metal_tok_layer(layer); }
    bool end(float* x, int H) override { return metal_tok_end(x, H); }
    void abort() override { metal_tok_abort(); }
};
}

TokenGraphBackend& metal_token_graph_backend() {
    static MetalTokenGraphBackend backend;
    return backend;
}

} // namespace Laplace
