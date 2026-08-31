// Native Metal 4 TensorOps GEMV with a real M tile: 32 output rows of W
// are the matmul M dimension. x is padded to N=16 (one live column).
// W is a device int8 plane [N, K] (row-major n*K+k) plus per-16 half scales
// and optional mins. No dequant loop over GGUF blocks.
static const char* src_matmul2d = R"METAL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void gemv_m4_rows(
    device int8_t* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint64_t& unused [[buffer(5)]],
    constant int& Mpad [[buffer(6)]],
    device const half* scale [[buffer(7)]],
    device const half* amin [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    (void)unused; (void)Mpad;
    const int n0 = int(tgid) * 32;
    const int n_threads = 128;
    const int ng = K / 16;
    threadgroup half Bt[16 * 16];
    threadgroup float Ct[16 * 32];
    threadgroup float Yt[32];

    constexpr auto desc = matmul2d_descriptor(
        32, 16, 16, false, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;

    auto tB = tensor<threadgroup half, extents<int32_t, 16, 16>, tensor_inline>(
        Bt, extents<int32_t, 16, 16>());
    auto tC = tensor<threadgroup float, extents<int32_t, 16, 32>, tensor_inline>(
        Ct, extents<int32_t, 16, 32>());
    auto AW = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        W, dextents<int32_t, 2>(K, N));

    for (int i = int(tid); i < 32; i += n_threads) Yt[i] = 0.f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k0 = 0, g = 0; k0 < K; k0 += 16, g++) {
        for (int i = int(tid); i < 16 * 16; i += n_threads) {
            int n = i & 15;
            int k = i >> 4;
            Bt[n + k * 16] = (n == 0) ? half(x[k0 + k]) : half(0.f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto tA = AW.slice(k0, n0);
        op.run(tA, tB, tC);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32u) {
            int n = n0 + int(tid);
            if (n < N) {
                float sx = 0.f;
                for (int k = 0; k < 16; k++) sx += x[k0 + k];
                Yt[tid] += Ct[tid * 16] * float(scale[n * ng + g])
                         - sx * float(amin[n * ng + g]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < 32u) {
        int n = n0 + int(tid);
        if (n < N) y[n] = Yt[tid];
    }
}

)METAL";

// macOS 27-only source is compiled separately with Metal language 4.1 so the
// existing Metal 4 library remains usable on systems without Int2 symbols.
static const char* src_mpp_int2 = R"METAL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal;
using namespace mpp::tensor_ops;

using mpp_int2_scales = tensor_blockwise<tensor_plane_scales,
                                         device metal_fp8_ue8m0_format,
                                         32, 1>;
using mpp_int2_weights = tensor<device int2b_format,
                                dextents<int, 2>, tensor_inline,
                                mpp_int2_scales>;

kernel void gemv_mpp_int2_m1(
    device half* x [[buffer(0)]],
    device uchar* weights [[buffer(1)]],
    device uchar* scales [[buffer(2)]],
    device float* y [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]])
{
    constexpr int K = 5120;
    constexpr int N = 17408;
    constexpr int M = 8;
    constexpr int NT = 16;
    const int n0 = int(tgid) * NT;
    auto input = tensor<device half, dextents<int, 2>, tensor_inline>(
        x, dextents<int, 2>(K, M), array<int, 2>({1, K}));
    auto weights_full = mpp_int2_weights(
        weights, dextents<int, 2>(K, N), array<int, 2>({1, K}),
        mpp_int2_scales(scales));
    auto output_full = tensor<device float, dextents<int, 2>, tensor_inline>(
        y, dextents<int, 2>(N, M), array<int, 2>({1, N}));
    constexpr auto desc = matmul2d_descriptor(
        M, NT, K, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<2>> op;
    auto weight = weights_full.slice(0, n0);
    auto output = output_full.slice(n0, 0);
    op.run(input, weight, output);
}
)METAL";
