#include "../src/tensor.h"
#include "../src/kernels.h"
#include "../src/matmul.h"
#include "../src/fp16.h"
#include "../src/gguf.h"
#include "../src/kvcache.h"
#include "../src/model.h"
#include "../src/semantic_model.h"
#include "../src/token_graph_backend.h"
#include "../src/sparse_ffn_bank.h"
#include "../src/column_grouped_q4.h"
#include "../src/column_grouped_affine_lowbit.h"
#include "../src/column_grouped_affine_uint2_skip.h"
#include "gguf_writer.h"
#include "test_util.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>
#include <cmath>
#include <sys/mman.h>
#include <unistd.h>

using namespace Laplace;
using namespace Laplace::kernels;
namespace {
#include "../src/iq1_s_tables.inc"
}
namespace Laplace {
extern bool metal_available();
extern bool metal_device_present();
extern bool metal_gemv(const float* x, const Tensor& w, float* y, int K, int N);
extern bool metal_gemv_repeat(const float* x, const Tensor& w, float* y,
                              int K, int N, int reps);
extern bool metal_test_q2k_two_row_ab(const float* x, const Tensor& w,
                                      float* baseline, float* candidate,
                                      int K, int N, int reps,
                                      double* baseline_gpu_ms, double* candidate_gpu_ms);
extern bool metal_test_q2k_streamed_ab(const float* x, const Tensor& w,
                                       float* baseline, float* candidate,
                                       int K, int N, int reps,
                                       double* baseline_gpu_ms, double* candidate_gpu_ms);
extern bool metal_test_q2k_pipeline_limits(uint32_t* baseline_width, uint32_t* baseline_max,
                                           uint32_t* two_row_width, uint32_t* two_row_max,
                                           uint32_t* streamed_width, uint32_t* streamed_max);
extern bool metal_test_iq2_xxs_pipeline_limits(uint32_t* width, uint32_t* maximum_threads);
extern bool metal_test_iq1_s_pipeline_limits(uint32_t* width, uint32_t* maximum_threads);
extern bool metal_test_mpp_int2_m1(const float* input, const uint8_t* packed_weights,
                                   const uint8_t* e8m0_scales, float* output,
                                   int K, int N, double* gpu_ms,
                                   uint64_t* data_bytes, uint64_t* scale_bytes,
                                   uint32_t* width, uint32_t* maximum_threads);
extern bool metal_test_affine_u2_block256(const float* input,
                                          const uint8_t* packed_weights,
                                          const uint16_t* scales,
                                          const uint16_t* biases,
                                          float* output, int K, int N,
                                          double* gpu_ms, uint32_t samples,
                                          uint32_t dispatches_per_command,
                                          bool distinct_weights,
                                          uint64_t* requested_bytes,
                                          uint32_t* width, uint32_t* maximum_threads);
extern bool metal_test_affine_u2_block256_variant(
    const float* input, const uint8_t* packed_weights,
    const uint16_t* scales, const uint16_t* biases, float* output,
    int K, int N, uint32_t rows_per_simd, uint32_t simdgroups,
    bool masked_qdot, bool packed_word_load, uint32_t values_per_lane,
    double* gpu_ms, uint32_t samples, uint32_t dispatches_per_command,
    bool distinct_weights, uint64_t* requested_bytes,
    uint32_t* width, uint32_t* maximum_threads);
extern bool metal_test_affine_u2_block256_paired(
    const float* input, const uint8_t* packed_weights,
    const uint16_t* scales, const uint16_t* biases,
    float* control_output, float* candidate_output, int K, int N,
    double* control_gpu_ms, double* candidate_gpu_ms,
    uint32_t warmup_pairs, uint32_t measured_pairs,
    uint32_t dispatches_per_command, bool distinct_weights,
    uint64_t* requested_bytes, uint32_t* command_buffers);
extern bool metal_test_activation_importance_accumulator(const float* first, const float* second,
                                                         uint32_t width, float* sums,
                                                         uint32_t* sample_count);
extern bool metal_act_glu(const float* gate, const float* up, float* out, int n, bool swiglu);
extern bool metal_test_gated_delta_net(const float* qkv_input, const float* conv_weight,
                                       const float* conv_history_input, const float* gate,
                                       const float* beta_projection, const float* alpha_projection,
                                       const float* dt_bias, const float* decay, const float* norm_weight,
                                       const float* state_input, float* conv_history_output,
                                       float* state_output, float* output,
                                       int qk_heads, int value_heads, int head_dimension, int kernel,
                                       float l2_epsilon, float rms_epsilon);
extern bool metal_test_axis_split_gated_attention(const float* fused, int rows, int first_width,
                                                  int second_width, const float* context,
                                                  float* query, float* output);
extern bool metal_test_rope_interleaved(float* query, float* key, int query_heads, int key_heads,
                                        int head_dimension, int rotary_dimension, int frequency_dimension,
                                        float rope_base, int position);
extern bool metal_test_rope_multisection(float* query, float* key, int query_heads, int key_heads,
                                         int head_dimension, int rotary_dimension, int frequency_dimension,
                                         float rope_base, const int positions[4],
                                         const uint32_t sections[4]);
extern bool metal_test_q6k_embedding(const Tensor& embedding, uint32_t token, float* output,
                                     int width, int vocabulary);
extern bool metal_test_q4k_embedding(const Tensor& embedding, uint32_t token, float* output,
                                     int width, int vocabulary);
// Test-only physical-tuple sweep. The production binder remains unchanged;
// this API selects only the quant format, shape, and queried launch tuple.
extern bool metal_test_quantized_gemv_variant(
    const Tensor& weight, const float* input, float* output,
    uint32_t rows_per_simd, uint32_t simdgroups, uint32_t dispatches_per_command,
    bool llama_q6_variant, bool q6_reference,
    uint32_t warmups,
    uint32_t samples, double* sample_gpu_ms, uint64_t* requested_bytes,
    uint32_t* command_buffers, uint32_t* thread_execution_width,
    uint32_t* max_threads_per_threadgroup);
extern bool metal_test_rmsnorm_q8_0(
    const Tensor& weight, const float* input, float* output, int n, float eps,
    double* gpu_ms, uint64_t* requested_bytes, uint32_t* command_buffers,
    uint32_t* thread_execution_width, uint32_t* max_threads_per_threadgroup);
extern bool metal_test_sparse_ffn_original_spans(const float* input, const Tensor& gate,
                                                 const Tensor& up, const Tensor& down,
                                                 const MetalSparseBlockRun* runs, uint32_t run_count,
                                                 float* output, int hidden, int full_intermediate);
extern bool metal_test_select_contiguous_window(const float* scores, uint32_t total_blocks,
                                                uint32_t selected_blocks, uint32_t* selected_first,
                                                uint32_t* selected_count, double* gpu_ms);
extern bool metal_test_sparse_ffn_selected_window(const float* input, const Tensor& gate,
                                                  const Tensor& up, const Tensor& down,
                                                  const float* scores, uint32_t total_blocks,
                                                  uint32_t selected_blocks, float* output,
                                                  uint32_t* selected_first, int hidden,
                                                  int full_intermediate, double* gpu_ms);
extern bool metal_test_sparse_ffn_proxy_window(const float* input, const Tensor& gate,
                                               const Tensor& up, const Tensor& down,
                                               const float* proxy_coefficients,
                                               uint32_t input_blocks, uint32_t output_blocks,
                                               uint32_t selected_blocks, float* output,
                                               uint32_t* selected_first, int hidden,
                                               int full_intermediate, double* gpu_ms);
extern bool metal_test_moe_q4k_gate_up_gelu(const float* input, const Tensor& gate_up,
                                            size_t source_bytes,
                                            const uint32_t* expert_ids, uint32_t selected_count,
                                            float* output, int hidden, int intermediate,
                                            double* gpu_ms, uint32_t* pipeline_width,
                                            uint32_t* maximum_threads);
// Test-only seam for the Laplace-owned derived format. It is intentionally
// absent from matmul.h: production routing must not see this experiment.
extern bool metal_test_column_grouped_q4_v1(const ColumnGroupedQ4V1Storage& storage,
                                            const float* input, float* output, bool sparse,
                                            uint32_t* selected_columns, double* gpu_ms,
    uint32_t* thread_execution_width,
    uint32_t* max_threads_per_threadgroup);
extern bool metal_test_q6k_alias_paired(
    const Tensor& weight, const float* input, float* reference_output,
    float* candidate_output, uint32_t warmups, uint32_t samples,
    double* reference_gpu_ms, double* candidate_gpu_ms,
    double* reference_wall_ms, double* candidate_wall_ms,
    uint64_t* requested_bytes, uint32_t* command_buffers,
    uint32_t* thread_execution_width, uint32_t* max_threads_per_threadgroup);
extern bool metal_test_column_grouped_affine_lowbit_v1(
    const ColumnGroupedAffineLowBitV1Contract& contract,
    const ColumnGroupedAffineLowBitV1Planes& planes,
    const float* input, float* output, double* gpu_ms,
    uint32_t* thread_execution_width, uint32_t* max_threads_per_threadgroup);
extern bool metal_test_column_grouped_affine_lowbit_v1_benchmark(
    const ColumnGroupedAffineLowBitV1Contract& contract,
    const ColumnGroupedAffineLowBitV1Planes& planes,
    const float* input, float* output, uint32_t warmups, uint32_t samples,
    double* sample_gpu_ms, uint64_t* requested_bytes,
    uint32_t* thread_execution_width, uint32_t* max_threads_per_threadgroup);
extern bool metal_test_column_grouped_affine_lowbit_v1_optimized(
    const ColumnGroupedAffineLowBitV1Contract& contract,
    const ColumnGroupedAffineLowBitV1Planes& planes,
    const float* input, float* output, double* gpu_ms,
    uint32_t* thread_execution_width, uint32_t* max_threads_per_threadgroup);
extern bool metal_test_column_grouped_affine_lowbit_v1_optimized_benchmark(
    const ColumnGroupedAffineLowBitV1Contract& contract,
    const ColumnGroupedAffineLowBitV1Planes& planes,
    const float* input, float* output, uint32_t warmups, uint32_t samples,
    double* sample_gpu_ms, uint64_t* requested_bytes,
    uint32_t* thread_execution_width, uint32_t* max_threads_per_threadgroup);
extern bool metal_test_column_grouped_affine_uint2_skip_v1(
    const ColumnGroupedAffineUInt2SkipV1Contract& contract,
    const ColumnGroupedAffineUInt2SkipV1Planes& planes,
    const float* input, float* output, bool sparse, double* gpu_ms,
    uint32_t* thread_execution_width, uint32_t* max_threads_per_threadgroup);
extern bool metal_test_column_grouped_affine_uint2_skip_v1_benchmark(
    const ColumnGroupedAffineUInt2SkipV1Contract& contract,
    const ColumnGroupedAffineUInt2SkipV1Planes& planes,
    const float* input, float* output, bool sparse, uint32_t warmups,
    uint32_t samples, double* sample_gpu_ms, uint64_t* requested_bytes,
    uint32_t* thread_execution_width, uint32_t* max_threads_per_threadgroup);
}

static void rng_fill(float* v, int n, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1103515245 + 12345;
        v[i] = static_cast<float>(
            static_cast<int>((s >> 16) % 200) - 100) / 10.0f;
    }
}

static uint16_t f2h(float f) { return fp32_to_fp16(f); }

static size_t trellis_tile_index(uint32_t row, uint32_t col) {
    const uint32_t group = col < 8u ? col : col - 8u;
    const uint32_t pair = group * 4u + ((row & 7u) >> 1u);
    const uint32_t slot = (row >= 8u ? 2u : 0u) + (row & 1u) +
                          (col >= 8u ? 4u : 0u);
    return static_cast<size_t>(pair * 8u + slot);
}

static TrellisPhysicalDescriptor trellis_descriptor_for_bits(uint8_t bits,
                                                             uint32_t K, uint32_t N) {
    TrellisPhysicalDescriptor descriptor{};
    CHECK(trellis_select_descriptor(TrellisTileLayout::TensorCore16x16, 1, bits,
                                    K, N, TrellisCodebook::Default, {32, 1024},
                                    &descriptor));
    return descriptor;
}

static void test_trellis_decode_dot_small() {
    constexpr uint32_t K = 32, N = 16;
    for (uint8_t bits : {uint8_t(1), uint8_t(2), uint8_t(7)}) {
        const TrellisPhysicalDescriptor descriptor = trellis_descriptor_for_bits(bits, K, N);
        const size_t tile_bytes = trellis_packed_bytes_per_tile(descriptor.bits);
        std::vector<uint8_t> packed(descriptor.packed_plane_bytes);
        std::vector<float> input(K), output(N), reference(N, 0.0f);
        rng_fill(input.data(), K, 91u + bits);
        std::array<uint16_t, TrellisTileValueCount> encoded{};
        std::array<uint16_t, TrellisTileValueCount> states{};
        for (uint32_t tile = 0; tile != K / 16u; ++tile) {
            for (size_t i = 0; i != encoded.size(); ++i)
                encoded[i] = static_cast<uint16_t>(
                    (i * 73u + (i / 7u) * 11u + tile * 29u) & ((1u << bits) - 1u));
            CHECK(trellis_pack_tile(encoded,
                                    std::span<uint8_t>(packed.data() + tile * tile_bytes,
                                                       tile_bytes), descriptor.bits));
            CHECK(trellis_unpack_tile(
                std::span<const uint8_t>(packed.data() + tile * tile_bytes, tile_bytes),
                states, descriptor.bits));
            for (uint32_t col = 0; col != N; ++col)
                for (uint32_t row = 0; row != 16; ++row) {
                    const uint16_t cw = states[trellis_tile_index(row, col)];
                    reference[col] += input[tile * 16u + row] *
                        trellis_decode_codeword(cw, TrellisCodebook::Default);
                }
        }
        std::vector<double> samples(3);
        uint64_t requested_bytes = 0;
        uint32_t command_buffers = 0, implicit_copies = 99;
        CHECK(metal_test_trellis_decode_dot(
            descriptor, packed.data(), input.data(), output.data(), TrellisCodebook::Default,
            1, static_cast<uint32_t>(samples.size()), samples.data(), &requested_bytes,
            &command_buffers, &implicit_copies));
        CHECK(requested_bytes == descriptor.packed_plane_bytes);
        CHECK(command_buffers == 4);
        CHECK(implicit_copies == 0);
        for (uint32_t col = 0; col != N; ++col)
            CHECK_MSG(std::isfinite(output[col]) &&
                          almost_equal(output[col], reference[col], 3.0e-2f, 3.0e-2f),
                      "trellis dot k=%u col=%u actual=%g expected=%g", bits, col,
                      output[col], reference[col]);
        std::fprintf(stderr, "PASS trellis decode-dot K=%u N=%u k=%u command_buffers=%u "
                             "implicit_copies=%u requested_bytes=%llu\n",
                     K, N, bits, command_buffers, implicit_copies,
                     static_cast<unsigned long long>(requested_bytes));
            }
}

static void test_trellis_decode_dot_real_shape() {
    constexpr uint32_t K = 5120, N = 17408;
    for (uint8_t bits : {uint8_t(1), uint8_t(2)}) {
        const TrellisPhysicalDescriptor descriptor = trellis_descriptor_for_bits(bits, K, N);
        const size_t tile_bytes = trellis_packed_bytes_per_tile(descriptor.bits);
        std::vector<uint8_t> packed(descriptor.packed_plane_bytes);
        std::array<uint16_t, TrellisTileValueCount> encoded{};
        for (size_t i = 0; i != encoded.size(); ++i)
            encoded[i] = static_cast<uint16_t>(
                (i * 73u + (i / 7u) * 11u + 3u) & ((1u << bits) - 1u));
        for (uint32_t tile = 0; tile != descriptor.tile_count; ++tile)
            CHECK(trellis_pack_tile(encoded,
                                    std::span<uint8_t>(packed.data() + tile * tile_bytes,
                                                       tile_bytes), descriptor.bits));
        std::vector<float> input(K), output(N);
        rng_fill(input.data(), K, 123u + bits);
        constexpr uint32_t warmups = 2, sample_count = 5;
        std::array<double, sample_count> samples{};
        uint64_t requested_bytes = 0;
        uint32_t command_buffers = 0, implicit_copies = 99;
        CHECK(metal_test_trellis_decode_dot(
            descriptor, packed.data(), input.data(), output.data(), TrellisCodebook::Default,
            warmups, sample_count, samples.data(), &requested_bytes,
            &command_buffers, &implicit_copies));
        std::vector<double> sorted(samples.begin(), samples.end());
        std::sort(sorted.begin(), sorted.end());
        const double median = sorted[sample_count / 2u];
        const double gbps = static_cast<double>(requested_bytes) / 1.0e9 /
                            (median / 1000.0);
        CHECK(command_buffers == warmups + sample_count);
        CHECK(implicit_copies == 0);
        for (float value : output) CHECK(std::isfinite(value));
        std::fprintf(stderr,
                     "PASS trellis decode-dot benchmark K=%u N=%u k=%u packed_bytes=%llu "
                     "median_gpu_ms=%.6f range_gpu_ms=%.6f..%.6f requested_GBps=%.6f "
                     "width=%u max_threads=%u command_buffers=%u implicit_copies=%u\n",
                     K, N, bits, static_cast<unsigned long long>(requested_bytes),
                     median, sorted.front(), sorted.back(), gbps,
                     32u, 1024u,
                     command_buffers, implicit_copies);
    }
}

static void test_tensorops_descriptor_probe() {
    constexpr uint64_t K = 512;
    constexpr uint64_t N = 16;
    constexpr uint64_t data_plane_bytes = K * N / 4;
    constexpr uint64_t scale_plane_bytes = (K / 32) * N;
    constexpr size_t allocation_bytes = 4096;
    void* data = mmap(nullptr, allocation_bytes, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANON, -1, 0);
    void* scales = mmap(nullptr, allocation_bytes, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(data != MAP_FAILED);
    CHECK(scales != MAP_FAILED);
    if (data == MAP_FAILED || scales == MAP_FAILED) {
        if (data != MAP_FAILED) munmap(data, allocation_bytes);
        if (scales != MAP_FAILED) munmap(scales, allocation_bytes);
        return;
    }
    std::memset(data, 0, allocation_bytes);
    std::memset(scales, 0, allocation_bytes);

    MetalTensorOpsProbeDescriptor descriptor;
    descriptor.dimensions = {K, N};
    descriptor.strides = {1, K};
    descriptor.data_plane = {data_plane_bytes, 0};
    descriptor.scale_plane = {scale_plane_bytes, 0};
    descriptor.block_factors = {32, 1};
    descriptor.attachments = {data, allocation_bytes, scales, allocation_bytes};

    auto rejects = [&](const MetalTensorOpsProbeDescriptor& invalid) {
        const MetalTensorOpsProbeResult result = metal_test_tensorops_descriptor_probe(invalid);
        CHECK(result.status == MetalTensorOpsProbeStatus::Rejected);
        CHECK(result.attached_planes == 0);
        CHECK(result.command_buffers == 0);
        CHECK(result.implicit_copies == 0);
    };

    auto invalid = descriptor;
    invalid.version = 2;
    rejects(invalid);
    invalid = descriptor;
    invalid.rank = 1;
    rejects(invalid);
    invalid = descriptor;
    invalid.storage = static_cast<MetalTensorOpsProbeStorage>(99);
    rejects(invalid);
    invalid = descriptor;
    invalid.dimensions[0] = K - 1;
    rejects(invalid);
    invalid = descriptor;
    invalid.strides[0] = 2;
    rejects(invalid);
    invalid = descriptor;
    invalid.strides[1] = K / 2;
    rejects(invalid);
    invalid = descriptor;
    invalid.data_plane.byte_offset = 1;
    rejects(invalid);
    invalid = descriptor;
    invalid.scale_plane.byte_length = 0;
    rejects(invalid);
    invalid = descriptor;
    invalid.block_factors[0] = 16;
    rejects(invalid);
    invalid = descriptor;
    invalid.attachments.data = nullptr;
    rejects(invalid);
    invalid = descriptor;
    invalid.attachments.scale_bytes = scale_plane_bytes - 1;
    rejects(invalid);
    std::fprintf(stderr, "tensorops invalid descriptors: OK\n");

    auto incompatible = descriptor;
    incompatible.dimensions[0] = K * 2;
    incompatible.strides[1] = K * 2;
    incompatible.data_plane.byte_length = data_plane_bytes * 2;
    incompatible.scale_plane.byte_length = scale_plane_bytes * 2;
    const MetalTensorOpsProbeResult skipped = metal_test_tensorops_descriptor_probe(incompatible);
    CHECK(skipped.status == MetalTensorOpsProbeStatus::SkippedSdkUnavailable);
    CHECK(skipped.attached_planes == 0);
    CHECK(skipped.command_buffers == 0);
    CHECK(skipped.implicit_copies == 0);
    std::fprintf(stderr, "tensorops incompatible descriptor: SKIP SDK_UNAVAILABLE\n");

    auto accepts_or_skips = [&](MetalTensorOpsProbeStorage storage) {
        auto valid = descriptor;
        valid.storage = storage;
        const MetalTensorOpsProbeResult result = metal_test_tensorops_descriptor_probe(valid);
        CHECK(result.implicit_copies == 0);
        if (result.status == MetalTensorOpsProbeStatus::Ran) {
            CHECK(result.attached_planes == 2);
            CHECK(result.command_buffers == 1);
            std::fprintf(stderr, "tensorops probe: RAN storage=%u\n",
                         static_cast<unsigned>(storage));
        } else if (result.status == MetalTensorOpsProbeStatus::SkippedSdkUnavailable) {
            CHECK(result.attached_planes == 0);
            CHECK(result.command_buffers == 0);
            std::fprintf(stderr, "tensorops probe: SKIP SDK_UNAVAILABLE storage=%u\n",
                         static_cast<unsigned>(storage));
        } else {
            CHECK_MSG(false, "tensorops probe failed");
        }
    };
    accepts_or_skips(MetalTensorOpsProbeStorage::UInt2);
    accepts_or_skips(MetalTensorOpsProbeStorage::Int2);

    munmap(data, allocation_bytes);
    munmap(scales, allocation_bytes);
}

struct SamplerOracleResult {
    bool valid = false;
    uint32_t token_id = 0;
    float logit = 0.0f;
};

static SamplerOracleResult sampler_cpu_oracle(const float* logits, uint32_t vocabulary) {
    SamplerOracleResult result;
    if (!logits || vocabulary == 0) return result;
    for (uint32_t token = 0; token != vocabulary; ++token) {
        if (!std::isfinite(logits[token])) return {};
        if (!result.valid || logits[token] > result.logit ||
            (logits[token] == result.logit && token < result.token_id)) {
            result.valid = true;
            result.token_id = token;
            result.logit = logits[token];
        }
    }
    return result;
}

static void test_sampler() {
    CHECK(sizeof(MetalSamplerResult) == 16);
    MetalSamplerDescriptor descriptor;
    CHECK(descriptor.version == 1);
    CHECK(descriptor.mode == MetalSamplerMode::Greedy);
    CHECK(descriptor.tie_policy == MetalSamplerTiePolicy::FirstIndex);
    CHECK(descriptor.nonfinite_policy == MetalSamplerNonFinitePolicy::Reject);
    CHECK(descriptor.temperature == 1.0f);
    CHECK(descriptor.top_k == 0);
    CHECK(descriptor.top_p == 1.0f);
    CHECK(descriptor.rng_seed == 0);
    CHECK(descriptor.rng_counter == 0);

    auto run = [&](const std::vector<float>& logits, bool expect_command) {
        const SamplerOracleResult expected = sampler_cpu_oracle(logits.data(),
                                                                static_cast<uint32_t>(logits.size()));
        const std::vector<float> before = logits;
        MetalSamplerResult actual;
        actual.token_id = 0xdecafbadU;
        actual.logit = -123.0f;
        actual.status = MetalSamplerResultStatus::InvalidInput;
        actual.reserved = 0x55aaU;
        uint32_t command_buffers = 0;
        const bool ok = metal_test_sampler_greedy(
            descriptor, logits.data(), static_cast<uint32_t>(logits.size()), &actual,
            &command_buffers);
        CHECK_MSG(ok == expected.valid,
                  "sampler n=%zu ok=%d expected=%d cb=%u status=%u token=%u logit=%g expected_logit=%g",
                  logits.size(), ok, expected.valid, command_buffers,
                  static_cast<unsigned>(actual.status), actual.token_id,
                  actual.logit, expected.logit);
        CHECK(command_buffers == (expect_command ? 1U : 0U));
        CHECK(std::memcmp(logits.data(), before.data(), logits.size() * sizeof(float)) == 0);
        if (expected.valid) {
            CHECK_MSG(actual.status == MetalSamplerResultStatus::Success,
                      "sampler n=%zu status=%u token=%u logit=%g",
                      logits.size(), static_cast<unsigned>(actual.status),
                      actual.token_id, actual.logit);
            CHECK_MSG(actual.token_id == expected.token_id,
                      "sampler n=%zu token=%u expected=%u status=%u logit=%g",
                      logits.size(), actual.token_id, expected.token_id,
                      static_cast<unsigned>(actual.status), actual.logit);
            CHECK_MSG(actual.logit == expected.logit,
                      "sampler n=%zu logit=%g expected=%g token=%u status=%u",
                      logits.size(), actual.logit, expected.logit,
                      actual.token_id, static_cast<unsigned>(actual.status));
            CHECK(actual.reserved == 0);
        } else {
            CHECK(actual.status == MetalSamplerResultStatus::NonFiniteLogit);
            CHECK(actual.reserved == 0);
        }
    };

    run({3.0f}, true);
    run({-4.0f, 7.0f, 7.0f, 6.0f}, true);
    std::vector<float> boundary(256, -9.0f);
    boundary[255] = 11.0f;
    run(boundary, true);
    boundary.push_back(11.0f);
    run(boundary, true);
    run({1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f}, true);
    run({1.0f, std::numeric_limits<float>::infinity(), 3.0f}, true);
    run({1.0f, -std::numeric_limits<float>::infinity(), 3.0f}, true);

    MetalSamplerResult invalid;
    invalid.token_id = 17;
    invalid.logit = 19.0f;
    invalid.status = MetalSamplerResultStatus::InvalidInput;
    invalid.reserved = 23;
    uint32_t command_buffers = 99;
    CHECK(!metal_test_sampler_greedy(descriptor, nullptr, 0, &invalid, &command_buffers));
    CHECK(command_buffers == 0);
    CHECK(invalid.token_id == 17);
    CHECK(invalid.logit == 19.0f);
    CHECK(invalid.status == MetalSamplerResultStatus::InvalidInput);
    CHECK(invalid.reserved == 23);

    MetalSamplerDescriptor future = descriptor;
    future.version = 2;
    const float future_logits[] = {1.0f};
    command_buffers = 99;
    CHECK(!metal_test_sampler_greedy(future, future_logits, 1,
                                     &invalid, &command_buffers));
    CHECK(command_buffers == 0);
    CHECK(invalid.token_id == 17);
    CHECK(invalid.logit == 19.0f);
    CHECK(invalid.status == MetalSamplerResultStatus::InvalidInput);
    CHECK(invalid.reserved == 23);
}

static void test_weight_registration_is_transactional() {
    const long page_size = sysconf(_SC_PAGESIZE);
    CHECK(page_size > 0);
    if (page_size <= 0) return;
    const size_t page = static_cast<size_t>(page_size);
    const size_t bytes = page * 48u;
    void* first_mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANON, -1, 0);
    void* failed_mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(first_mapping != MAP_FAILED);
    CHECK(failed_mapping != MAP_FAILED);
    if (first_mapping == MAP_FAILED || failed_mapping == MAP_FAILED) {
        if (first_mapping != MAP_FAILED) munmap(first_mapping, bytes);
        if (failed_mapping != MAP_FAILED) munmap(failed_mapping, bytes);
        return;
    }

    auto session = metal_tok_session_create();
    CHECK(session != nullptr);
    if (!session) {
        munmap(first_mapping, bytes);
        munmap(failed_mapping, bytes);
        return;
    }
    const size_t chunk = page * 16u;
    CHECK(metal_tok_session_register_weights_for_testing(
        *session, first_mapping, bytes, chunk, UINT32_MAX));
    const MetalResourceSnapshot registered = metal_tok_session_resource_snapshot(*session);
    // 48 pages split into 16-page chunks with 4 pages of overlap yields four
    // published buffers and 60 registered pages.
    CHECK(registered.registered_weight_bytes == page * 60u);
    CHECK(metal_tok_session_weight_span_coverage(*session, first_mapping, page) > 0);

    const MetalResourceSnapshot before_failure = registered;
    CHECK(!metal_tok_session_register_weights_for_testing(
        *session, failed_mapping, bytes, chunk, 1));
    const MetalResourceSnapshot after_failure = metal_tok_session_resource_snapshot(*session);
    CHECK(after_failure.registered_weight_bytes == before_failure.registered_weight_bytes);
    CHECK(after_failure.implicit_weight_copies == before_failure.implicit_weight_copies);
    CHECK(metal_tok_session_weight_span_coverage(*session, first_mapping, page) > 0);
    CHECK(metal_tok_session_weight_span_coverage(*session, failed_mapping, page) == 0);

    metal_tok_session_unregister_weights(*session, first_mapping);
    const MetalResourceSnapshot after_unregister = metal_tok_session_resource_snapshot(*session);
    CHECK(after_unregister.registered_weight_bytes == 0);
    for (size_t offset = 0; offset < bytes; offset += page * 12u) {
        const auto* address = static_cast<const uint8_t*>(first_mapping) + offset;
        CHECK(metal_tok_session_weight_span_coverage(*session, address, page) == 0);
    }
    CHECK(metal_tok_session_weight_span_coverage(*session, failed_mapping, page) == 0);
    munmap(first_mapping, bytes);
    munmap(failed_mapping, bytes);
}

static void test_router_top_k_device() {
    constexpr uint32_t E = 128;
    constexpr uint32_t R = 8;
    const MetalRouterTopKSpec spec{
        E, R, MetalRouterScoreDomain::Logits,
        MetalRouterNormalization::SelectThenNormalizeSoftmax,
        MetalRouterTiePolicy::LowestExpertId};
    std::vector<float> scores(E, 0.0f);
    std::vector<uint32_t> actual_ids(R, UINT32_MAX);
    std::vector<float> actual_weights(R, -1.0f);
    uint32_t width = 0, max_threads = 0;

    metal_dispatch_metrics_reset();
    MetalRouterPipelineCaps capabilities;
    CHECK(metal_test_router_top_k(spec, scores.data(), actual_ids.data(),
                                  actual_weights.data(), &capabilities));
    width = capabilities.thread_execution_width;
    max_threads = capabilities.max_total_threads_per_threadgroup;
    CHECK(actual_ids == std::vector<uint32_t>({0, 1, 2, 3, 4, 5, 6, 7}));
    for (float weight : actual_weights) CHECK(almost_equal(weight, 0.125f, 1e-6f, 1e-7f));
    CHECK(width != 0);
    CHECK(max_threads >= width);
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    RouterTopKPayload payload;
    payload.expert_count = E;
    payload.selected_count = R;
    auto oracle = semantic_router_top_k(scores, payload);
    CHECK(std::holds_alternative<SemanticRouterResult>(oracle));
    if (const auto* expected = std::get_if<SemanticRouterResult>(&oracle)) {
        CHECK(actual_ids == expected->ids);
        for (uint32_t slot = 0; slot != R; ++slot)
            CHECK(almost_equal(actual_weights[slot], expected->weights[slot], 1e-6f, 1e-7f));
    }

    scores.assign(E, -100.0f);
    scores[3] = std::numeric_limits<float>::max();
    scores[5] = std::numeric_limits<float>::max();
    scores[7] = std::numeric_limits<float>::max();
    scores[11] = 0.0f;
    scores[13] = -1.0f;
    scores[17] = -std::numeric_limits<float>::max();
    std::fill(actual_ids.begin(), actual_ids.end(), UINT32_MAX);
    std::fill(actual_weights.begin(), actual_weights.end(), -1.0f);
    metal_dispatch_metrics_reset();
    CHECK(metal_test_router_top_k(spec, scores.data(), actual_ids.data(),
                                  actual_weights.data(), &capabilities));
    CHECK(actual_ids[0] == 3 && actual_ids[1] == 5 && actual_ids[2] == 7);
    CHECK(actual_ids[3] == 11 && actual_ids[4] == 13);
    CHECK(actual_ids[5] == 0 && actual_ids[6] == 1 && actual_ids[7] == 2);
    CHECK(almost_equal(actual_weights[0], 1.0f / 3.0f, 1e-6f, 1e-7f));
    CHECK(almost_equal(actual_weights[1], 1.0f / 3.0f, 1e-6f, 1e-7f));
    CHECK(almost_equal(actual_weights[2], 1.0f / 3.0f, 1e-6f, 1e-7f));
    oracle = semantic_router_top_k(scores, payload);
    CHECK(std::holds_alternative<SemanticRouterResult>(oracle));
    if (const auto* expected = std::get_if<SemanticRouterResult>(&oracle)) {
        CHECK(actual_ids == expected->ids);
        for (uint32_t slot = 0; slot != R; ++slot)
            CHECK(almost_equal(actual_weights[slot], expected->weights[slot], 1e-6f, 1e-7f));
    }
    CHECK(metal_dispatch_metrics().command_buffers == 1);

    std::array<float, E> nonfinite = [] {
        std::array<float, E> value{};
        value.fill(0.0f);
        value[4] = std::numeric_limits<float>::quiet_NaN();
        return value;
    }();
    std::fill(actual_ids.begin(), actual_ids.end(), UINT32_MAX);
    std::fill(actual_weights.begin(), actual_weights.end(), -1.0f);
    CHECK(!metal_test_router_top_k(spec, nonfinite.data(), actual_ids.data(),
                                   actual_weights.data(), &capabilities));
    CHECK(actual_ids[0] == UINT32_MAX && actual_weights[0] == -1.0f);
    nonfinite[4] = std::numeric_limits<float>::infinity();
    CHECK(!metal_test_router_top_k(spec, nonfinite.data(), actual_ids.data(),
                                   actual_weights.data(), &capabilities));
    CHECK(!metal_test_router_top_k({0, R}, scores.data(), actual_ids.data(),
                                   actual_weights.data(), &capabilities));
    CHECK(!metal_test_router_top_k({E, E + 1}, scores.data(), actual_ids.data(),
                                   actual_weights.data(), &capabilities));
    CHECK(!metal_test_router_top_k({E, R, static_cast<MetalRouterScoreDomain>(2)},
                                   scores.data(), actual_ids.data(), actual_weights.data(),
                                   &capabilities));
    CHECK(!metal_test_router_top_k({E, R, MetalRouterScoreDomain::Logits,
                                    static_cast<MetalRouterNormalization>(2)},
                                   scores.data(), actual_ids.data(), actual_weights.data(),
                                   &capabilities));
    CHECK(!metal_test_router_top_k({E, R, MetalRouterScoreDomain::Logits,
                                    MetalRouterNormalization::SelectThenNormalizeSoftmax,
                                    static_cast<MetalRouterTiePolicy>(2)},
                                   scores.data(), actual_ids.data(), actual_weights.data(),
                                   &capabilities));
    std::fprintf(stderr, "router top-k: OK command_buffers=1 cpu_route=0 "
                         "invalid=pass nonfinite=pass capability=pass\n");
}

namespace {

gguf_writer::TensorDecl model_f32_tensor(std::string name,
                                         std::vector<uint64_t> dims,
                                         float value = 0.0f) {
    size_t count = 1;
    for (uint64_t dim : dims) count *= static_cast<size_t>(dim);
    gguf_writer::TensorDecl tensor;
    tensor.name = std::move(name);
    tensor.dims = std::move(dims);
    tensor.type = 0;
    tensor.data.resize(count * sizeof(float));
    for (size_t i = 0; i < count; ++i)
        std::memcpy(tensor.data.data() + i * sizeof(float), &value, sizeof(value));
    return tensor;
}

void write_named_cpu_model(const char* path) {
    gguf_writer::Writer writer;
    writer.kv_str("general.architecture", "llama");
    writer.kv_u32("llama.block_count", 1);
    writer.kv_u32("llama.embedding_length", 4);
    writer.kv_u32("llama.feed_forward_length", 8);
    writer.kv_u32("llama.context_length", 16);
    writer.kv_u32("llama.attention.head_count", 1);
    writer.kv_u32("llama.attention.head_count_kv", 1);
    writer.kv_u32("llama.attention.key_length", 4);
    writer.kv_u32("llama.expert_count", 2);
    writer.kv_u32("llama.expert_used_count", 1);
    writer.kv_u32("llama.expert_feed_forward_length", 4);

    auto token_embd = model_f32_tensor("token_embd.weight", {4, 3});
    const float token_zero[] = {1.f, 0.f, 0.f, 0.f};
    std::memcpy(token_embd.data.data(), token_zero, sizeof(token_zero));
    writer.add_tensor(std::move(token_embd));
    writer.add_tensor(model_f32_tensor("output_norm.weight", {4}, 1.f));
    auto output = model_f32_tensor("output.weight", {4, 3});
    const float output_rows[] = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
    };
    std::memcpy(output.data.data(), output_rows, sizeof(output_rows));
    writer.add_tensor(std::move(output));
    writer.add_tensor(model_f32_tensor("rope_freqs.weight", {2}));
    writer.add_tensor(model_f32_tensor("blk.0.attn_norm.weight", {4}, 1.f));
    writer.add_tensor(model_f32_tensor("blk.0.post_attention_norm.weight", {4}, 1.f));
    writer.add_tensor(model_f32_tensor("blk.0.attn_q.weight", {4, 4}));
    writer.add_tensor(model_f32_tensor("blk.0.attn_k.weight", {4, 4}));
    writer.add_tensor(model_f32_tensor("blk.0.attn_v.weight", {4, 4}));
    writer.add_tensor(model_f32_tensor("blk.0.attn_output.weight", {4, 4}));
    writer.add_tensor(model_f32_tensor("blk.0.ffn_gate.weight", {4, 8}));
    writer.add_tensor(model_f32_tensor("blk.0.ffn_up.weight", {4, 8}));
    writer.add_tensor(model_f32_tensor("blk.0.ffn_down.weight", {8, 4}));
    // This forces adaptive loading to reject the package. The named CPU
    // Llama executor then remains the actual layer caller under test.
    writer.add_tensor(model_f32_tensor("blk.0.ffn_gate_inp.weight", {4, 2}));
    CHECK(writer.write_file(path));
}

void write_adaptive_model(const char* path, int layers = 1) {
    gguf_writer::Writer writer;
    writer.kv_str("general.architecture", "llama");
    writer.kv_u32("llama.block_count", static_cast<uint32_t>(layers));
    writer.kv_u32("llama.embedding_length", 4);
    writer.kv_u32("llama.feed_forward_length", 8);
    writer.kv_u32("llama.context_length", 16);
    writer.kv_u32("llama.attention.head_count", 1);
    writer.kv_u32("llama.attention.head_count_kv", 1);
    writer.kv_u32("llama.attention.key_length", 4);
    writer.add_tensor(model_f32_tensor("token_embd.weight", {4, 3}));
    writer.add_tensor(model_f32_tensor("output_norm.weight", {4}, 1.f));
    writer.add_tensor(model_f32_tensor("output.weight", {4, 3}));
    writer.add_tensor(model_f32_tensor("rope_freqs.weight", {2}));
    for (int layer = 0; layer != layers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        writer.add_tensor(model_f32_tensor(prefix + "attn_norm.weight", {4}, 1.f));
        writer.add_tensor(model_f32_tensor(prefix + "post_attention_norm.weight", {4}, 1.f));
        writer.add_tensor(model_f32_tensor(prefix + "ffn_norm.weight", {4}, 1.f));
        writer.add_tensor(model_f32_tensor(prefix + "attn_q.weight", {4, 4}));
        writer.add_tensor(model_f32_tensor(prefix + "attn_k.weight", {4, 4}));
        writer.add_tensor(model_f32_tensor(prefix + "attn_v.weight", {4, 4}));
        writer.add_tensor(model_f32_tensor(prefix + "attn_output.weight", {4, 4}));
        writer.add_tensor(model_f32_tensor(prefix + "ffn_gate.weight", {4, 8}));
        writer.add_tensor(model_f32_tensor(prefix + "ffn_up.weight", {4, 8}));
        writer.add_tensor(model_f32_tensor(prefix + "ffn_down.weight", {8, 4}));
    }
    CHECK(writer.write_file(path));
}

class FakeTokenGraphBackend final : public TokenGraphBackend {
public:
    bool available() const override { return available_; }
    bool begin(int, int, int, int, int, int, int, int, int, int, int) override {
        ++begin_calls;
        active_ = true;
        return true;
    }
    bool active() const override { return active_; }
    void upload_x(const float*, int) override {}
    bool layer(const MetalTokLayer&) override {
        ++layer_calls;
        return fail_on_layer == 0 || layer_calls != fail_on_layer;
    }
    bool end(float* x, int H) override {
        ++end_calls;
        for (int i = 0; i < H; ++i) x[i] = -1.f;
        active_ = false;
        return true;
    }
    void abort() override {
        ++abort_calls;
        active_ = false;
    }

    bool available_ = true;
    int begin_calls = 0;
    int layer_calls = 0;
    int end_calls = 0;
    int abort_calls = 0;
    int fail_on_layer = 0;

private:
    bool active_ = false;
};

void run_named_cpu_model(FakeTokenGraphBackend* backend, float* logits) {
    constexpr const char* path = "test_metal_named_cpu.gguf";
    write_named_cpu_model(path);
    GGUFContext gguf;
    CHECK(gguf.open(path));
    Model model;
    CHECK(model.init(gguf));
    CHECK(model.arch() != nullptr);
    if (model.arch()) CHECK(std::string(model.arch()->name()) == "llama");
    CHECK(model.reserve(16));

    KVCache kv;
    CHECK(kv.init(1, 1, 4, 16, KVCacheMode::FP16));
    model.set_token_graph_backend_for_test(backend);
    model.enable_metal_token(true);
    model.forward(0, 0, kv, logits);

    std::remove(path);
}

void test_named_cpu_output_is_not_overwritten_by_token_graph() {
    FakeTokenGraphBackend backend;
    float logits[3] = {};
    run_named_cpu_model(&backend, logits);

    CHECK(backend.begin_calls == 0);
    CHECK(backend.end_calls == 0);
    CHECK(almost_equal(logits[0], 2.f));
    CHECK(almost_equal(logits[1], 0.f));
    CHECK(almost_equal(logits[2], 0.f));
}

void test_unavailable_executor_cannot_begin_token_graph() {
    FakeTokenGraphBackend backend;
    backend.available_ = false;
    float logits[3] = {};
    run_named_cpu_model(&backend, logits);
    CHECK(!backend.available());
    CHECK(backend.begin_calls == 0);
    CHECK(backend.end_calls == 0);
    CHECK(almost_equal(logits[0], 2.f));
}

void test_supported_executor_submits_a_complete_fake_graph() {
    constexpr const char* path = "test_metal_adaptive.gguf";
    write_adaptive_model(path);
    GGUFContext gguf;
    CHECK(gguf.open(path));
    Model model;
    CHECK(model.init(gguf));
    CHECK(model.arch() != nullptr);
    if (model.arch()) CHECK(std::string(model.arch()->name()) == "adaptive");
    CHECK(model.reserve(16));
    KVCache kv;
    CHECK(kv.init(1, 1, 4, 16, KVCacheMode::FP16));
    FakeTokenGraphBackend backend;
    model.set_token_graph_backend_for_test(&backend);
    model.enable_metal_token(true);
    float logits[3] = {};
    model.forward(0, 0, kv, logits);
    CHECK(backend.begin_calls == 1);
    CHECK(backend.layer_calls == 1);
    CHECK(backend.end_calls == 1);
    std::remove(path);
}

void test_late_token_graph_failure_aborts_without_cpu_continuation() {
    constexpr const char* path = "test_metal_adaptive_failure.gguf";
    write_adaptive_model(path, 2);
    GGUFContext gguf;
    CHECK(gguf.open(path));
    Model model;
    CHECK(model.init(gguf));
    CHECK(model.reserve(16));
    KVCache kv;
    CHECK(kv.init(2, 1, 4, 16, KVCacheMode::FP16));
    FakeTokenGraphBackend backend;
    backend.fail_on_layer = 2;
    model.set_token_graph_backend_for_test(&backend);
    model.enable_metal_token(true);
    float logits[3] = {7.f, 7.f, 7.f};
    CHECK(!model.forward(0, 0, kv, logits));
    CHECK(backend.begin_calls == 1);
    CHECK(backend.layer_calls == 2);
    CHECK(backend.abort_calls == 1);
    CHECK(backend.end_calls == 0);
    CHECK(almost_equal(logits[0], 7.f));

    backend.fail_on_layer = 0;
    CHECK(model.forward(0, 1, kv, logits));
    CHECK(backend.begin_calls == 2);
    CHECK(backend.end_calls == 1);
    std::remove(path);
}

float test_sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }
float test_softplus(float value) {
    return value > 0.0f ? value + std::log1p(std::exp(-value)) : std::log1p(std::exp(value));
}

void gated_delta_reference(const std::vector<float>& qkv_input, const std::vector<float>& conv_weight,
                           const std::vector<float>& conv_history_input, const std::vector<float>& gate,
                           const std::vector<float>& beta_projection, const std::vector<float>& alpha_projection,
                           const std::vector<float>& dt_bias, const std::vector<float>& decay,
                           const std::vector<float>& norm_weight, const std::vector<float>& state_input,
                           std::vector<float>& conv_history_output, std::vector<float>& state_output,
                           std::vector<float>& output, int qk_heads, int value_heads, int head_dimension,
                           int kernel, float l2_epsilon, float rms_epsilon) {
    const int channels = head_dimension * (2 * qk_heads + value_heads);
    const int history = kernel - 1;
    std::vector<float> qkv(channels);
    conv_history_output = conv_history_input;
    for (int channel = 0; channel < channels; ++channel) {
        const float* weight = conv_weight.data() + static_cast<size_t>(channel) * kernel;
        const float* history_row = conv_history_input.data() + static_cast<size_t>(channel) * history;
        float sum = weight[kernel - 1] * qkv_input[channel];
        for (int tap = 0; tap < history; ++tap) sum += history_row[tap] * weight[tap];
        qkv[channel] = sum * test_sigmoid(sum);
        for (int tap = 0; tap + 1 < history; ++tap)
            conv_history_output[static_cast<size_t>(channel) * history + tap] = history_row[tap + 1];
        conv_history_output[static_cast<size_t>(channel) * history + history - 1] = qkv_input[channel];
    }
    for (int qk_head = 0; qk_head < qk_heads; ++qk_head) {
        for (int part = 0; part != 2; ++part) {
            float sum = 0.0f;
            float* vector = qkv.data() + static_cast<size_t>(part * qk_heads + qk_head) * head_dimension;
            for (int column = 0; column < head_dimension; ++column) sum += vector[column] * vector[column];
            const float scale = 1.0f / std::sqrt(sum + l2_epsilon);
            for (int column = 0; column < head_dimension; ++column) vector[column] *= scale;
        }
        float* query = qkv.data() + static_cast<size_t>(qk_head) * head_dimension;
        const float query_scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));
        for (int column = 0; column < head_dimension; ++column) query[column] *= query_scale;
    }
    state_output.resize(state_input.size());
    output.resize(static_cast<size_t>(value_heads) * head_dimension);
    const float* key_base = qkv.data() + static_cast<size_t>(qk_heads) * head_dimension;
    const float* value_base = key_base + static_cast<size_t>(qk_heads) * head_dimension;
    for (int value_head = 0; value_head < value_heads; ++value_head) {
        const int qk_head = value_head % qk_heads;
        const float* query = qkv.data() + static_cast<size_t>(qk_head) * head_dimension;
        const float* key = key_base + static_cast<size_t>(qk_head) * head_dimension;
        const float* value = value_base + static_cast<size_t>(value_head) * head_dimension;
        const float scale = std::exp(decay[value_head] * test_softplus(alpha_projection[value_head] + dt_bias[value_head]));
        const float beta = test_sigmoid(beta_projection[value_head]);
        std::vector<float> raw(head_dimension);
        std::vector<float> delta(head_dimension);
        for (int output_column = 0; output_column < head_dimension; ++output_column) {
            float retrieved = 0.0f;
            for (int key_row = 0; key_row < head_dimension; ++key_row) {
                const size_t index = (static_cast<size_t>(value_head) * head_dimension + key_row) * head_dimension + output_column;
                retrieved += state_input[index] * scale * key[key_row];
            }
            delta[output_column] = beta * (value[output_column] - retrieved);
        }
        for (int key_row = 0; key_row < head_dimension; ++key_row) {
            const size_t base = (static_cast<size_t>(value_head) * head_dimension + key_row) * head_dimension;
            for (int output_column = 0; output_column < head_dimension; ++output_column) {
                state_output[base + output_column] = state_input[base + output_column] * scale +
                                                     key[key_row] * delta[output_column];
                raw[output_column] += state_output[base + output_column] * query[key_row];
            }
        }
        float ms = 0.0f;
        for (float value : raw) ms += value * value;
        const float inv = 1.0f / std::sqrt(ms / head_dimension + rms_epsilon);
        for (int column = 0; column < head_dimension; ++column) {
            const size_t index = static_cast<size_t>(value_head) * head_dimension + column;
            output[index] = raw[column] * inv * norm_weight[column] *
                            (gate[index] * test_sigmoid(gate[index]));
        }
    }
}

void test_gated_delta_net_ping_pong_matches_independent_equation() {
    constexpr int qk_heads = 2;
    constexpr int value_heads = 4;
    constexpr int head_dimension = 32;
    constexpr int kernel = 3;
    constexpr float l2_epsilon = 1.0e-6f;
    constexpr float rms_epsilon = 1.0e-6f;
    const int channels = head_dimension * (2 * qk_heads + value_heads);
    const int history_size = channels * (kernel - 1);
    const int state_size = value_heads * head_dimension * head_dimension;
    std::vector<float> qkv(channels), conv_weight(static_cast<size_t>(channels) * kernel),
                       conv_history(history_size), gate(value_heads * head_dimension), beta(value_heads),
                       alpha(value_heads), dt(value_heads), decay(value_heads), norm(head_dimension), state(state_size);
    rng_fill(qkv.data(), channels, 401);
    rng_fill(conv_weight.data(), static_cast<int>(conv_weight.size()), 402);
    rng_fill(conv_history.data(), history_size, 403);
    rng_fill(gate.data(), static_cast<int>(gate.size()), 404);
    rng_fill(beta.data(), value_heads, 405);
    rng_fill(alpha.data(), value_heads, 406);
    rng_fill(dt.data(), value_heads, 407);
    rng_fill(decay.data(), value_heads, 408);
    rng_fill(norm.data(), head_dimension, 409);
    rng_fill(state.data(), state_size, 410);
    for (float& value : qkv) value *= 0.05f;
    for (float& value : conv_weight) value *= 0.05f;
    for (float& value : conv_history) value *= 0.05f;
    for (float& value : gate) value *= 0.05f;
    for (float& value : beta) value *= 0.05f;
    for (float& value : alpha) value *= 0.05f;
    for (float& value : dt) value *= 0.05f;
    for (float& value : decay) value = -std::fabs(value * 0.05f);
    for (float& value : norm) value = 1.0f + value * 0.005f;
    for (float& value : state) value *= 0.01f;

    std::vector<float> reference_history, reference_state, reference_output;
    gated_delta_reference(qkv, conv_weight, conv_history, gate, beta, alpha, dt, decay, norm, state,
                          reference_history, reference_state, reference_output, qk_heads, value_heads,
                          head_dimension, kernel, l2_epsilon, rms_epsilon);
    const std::vector<float> input_history = conv_history;
    const std::vector<float> input_state = state;
    std::vector<float> gpu_history(history_size), gpu_state(state_size), gpu_output(value_heads * head_dimension);
    CHECK(metal_test_gated_delta_net(qkv.data(), conv_weight.data(), conv_history.data(), gate.data(),
                                     beta.data(), alpha.data(), dt.data(), decay.data(), norm.data(), state.data(),
                                     gpu_history.data(), gpu_state.data(), gpu_output.data(), qk_heads, value_heads,
                                     head_dimension, kernel, l2_epsilon, rms_epsilon));
    CHECK(conv_history == input_history);
    CHECK(state == input_state);
    float max_output_error = 0.0f;
    float max_state_error = 0.0f;
    for (size_t index = 0; index < gpu_output.size(); ++index)
        max_output_error = std::max(max_output_error, std::fabs(gpu_output[index] - reference_output[index]));
    for (size_t index = 0; index < gpu_state.size(); ++index)
        max_state_error = std::max(max_state_error, std::fabs(gpu_state[index] - reference_state[index]));
    CHECK(max_output_error < 2.0e-4f);
    CHECK(max_state_error < 2.0e-4f);
    CHECK(gpu_history.size() == reference_history.size());
    for (size_t index = 0; index < gpu_history.size(); ++index)
        CHECK(almost_equal(gpu_history[index], reference_history[index], 2.0e-4f, 2.0e-4f));

    std::vector<float> high_alpha = alpha;
    std::vector<float> high_decay = decay;
    high_alpha[0] = 90.0f;
    high_decay[0] = -0.0001f;
    std::vector<float> high_history, high_state, high_output;
    gated_delta_reference(qkv, conv_weight, conv_history, gate, beta, high_alpha, dt, high_decay, norm, state,
                          high_history, high_state, high_output, qk_heads, value_heads, head_dimension, kernel,
                          l2_epsilon, rms_epsilon);
    CHECK(metal_test_gated_delta_net(qkv.data(), conv_weight.data(), conv_history.data(), gate.data(),
                                     beta.data(), high_alpha.data(), dt.data(), high_decay.data(), norm.data(), state.data(),
                                     gpu_history.data(), gpu_state.data(), gpu_output.data(), qk_heads, value_heads,
                                     head_dimension, kernel, l2_epsilon, rms_epsilon));
    float high_state_error = 0.0f;
    for (size_t index = 0; index < gpu_state.size(); ++index)
        high_state_error = std::max(high_state_error, std::fabs(gpu_state[index] - high_state[index]));
    CHECK(high_state_error < 2.0e-4f);

    auto session = metal_tok_session_create();
    CHECK(session != nullptr);
    if (session) {
        CHECK(metal_tok_session_recurrent_seed(*session, conv_history.data(), state.data(), qk_heads,
                                               value_heads, head_dimension, kernel));
        std::vector<float> session_output(value_heads * head_dimension);
        CHECK(metal_tok_session_recurrent_step(*session, qkv.data(), conv_weight.data(), gate.data(),
                                               beta.data(), alpha.data(), dt.data(), decay.data(), norm.data(),
                                               session_output.data(), qk_heads, value_heads, head_dimension, kernel,
                                               l2_epsilon, rms_epsilon));
        std::vector<float> session_history(history_size), session_state(state_size);
        CHECK(metal_tok_session_recurrent_snapshot(*session, session_history.data(), session_state.data(),
                                                   qk_heads, value_heads, head_dimension, kernel));
        for (size_t index = 0; index < session_output.size(); ++index)
            CHECK(almost_equal(session_output[index], reference_output[index], 2.0e-4f, 2.0e-4f));
        for (size_t index = 0; index < session_history.size(); ++index)
            CHECK(almost_equal(session_history[index], reference_history[index], 2.0e-4f, 2.0e-4f));
        for (size_t index = 0; index < session_state.size(); ++index)
            CHECK(almost_equal(session_state[index], reference_state[index], 2.0e-4f, 2.0e-4f));

        std::vector<float> qkv_next = qkv;
        qkv_next[0] += 0.03125f;
        std::vector<float> next_history, next_state, next_output;
        gated_delta_reference(qkv_next, conv_weight, reference_history, gate, beta, alpha, dt, decay, norm,
                              reference_state, next_history, next_state, next_output, qk_heads, value_heads,
                              head_dimension, kernel, l2_epsilon, rms_epsilon);
        metal_tok_session_recurrent_fail_after_completed_submission_for_testing(*session);
        CHECK(!metal_tok_session_recurrent_step(*session, qkv_next.data(), conv_weight.data(), gate.data(),
                                                beta.data(), alpha.data(), dt.data(), decay.data(), norm.data(),
                                                session_output.data(), qk_heads, value_heads, head_dimension, kernel,
                                                l2_epsilon, rms_epsilon));
        CHECK(metal_tok_session_recurrent_snapshot(*session, session_history.data(), session_state.data(),
                                                   qk_heads, value_heads, head_dimension, kernel));
        for (size_t index = 0; index < session_history.size(); ++index)
            CHECK(almost_equal(session_history[index], reference_history[index], 2.0e-4f, 2.0e-4f));
        for (size_t index = 0; index < session_state.size(); ++index)
            CHECK(almost_equal(session_state[index], reference_state[index], 2.0e-4f, 2.0e-4f));

        CHECK(metal_tok_session_recurrent_step(*session, qkv_next.data(), conv_weight.data(), gate.data(),
                                               beta.data(), alpha.data(), dt.data(), decay.data(), norm.data(),
                                               session_output.data(), qk_heads, value_heads, head_dimension, kernel,
                                               l2_epsilon, rms_epsilon));
        CHECK(metal_tok_session_recurrent_snapshot(*session, session_history.data(), session_state.data(),
                                                   qk_heads, value_heads, head_dimension, kernel));
        for (size_t index = 0; index < session_output.size(); ++index)
            CHECK(almost_equal(session_output[index], next_output[index], 2.0e-4f, 2.0e-4f));
        for (size_t index = 0; index < session_history.size(); ++index)
            CHECK(almost_equal(session_history[index], next_history[index], 2.0e-4f, 2.0e-4f));
        for (size_t index = 0; index < session_state.size(); ++index)
            CHECK(almost_equal(session_state[index], next_state[index], 2.0e-4f, 2.0e-4f));
    }
    fprintf(stderr, "PASS DeltaNet ping-pong: output %.7g state %.7g\n", max_output_error, max_state_error);
}

void test_axis_split_gated_attention_matches_independent_equation() {
    const std::vector<float> fused = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> context = {2, -2, 1, -1};
    std::vector<float> query(4, 0.0f);
    std::vector<float> output(4, 0.0f);
    CHECK(metal_test_axis_split_gated_attention(fused.data(), 2, 2, 2, context.data(), query.data(), output.data()));
    CHECK(query == std::vector<float>({1, 2, 5, 6}));
    const std::vector<float> gate = {3, 4, 7, 8};
    for (size_t index = 0; index != output.size(); ++index) {
        const float expected = context[index] / (1.0f + std::exp(-gate[index]));
        CHECK(almost_equal(output[index], expected, 2.0e-6f, 2.0e-6f));
    }
    std::fprintf(stderr, "PASS AxisSplit gated attention: 4 values\n");
}

void test_interleaved_rope_matches_independent_equation() {
    constexpr int head_dimension = 4;
    constexpr int rotary_dimension = 4;
    constexpr float rope_base = 4.0f;
    constexpr int position = 1;
    std::vector<float> query = {1, 2, 3, 4};
    std::vector<float> key = {-1, 1, 2, -2};
    const auto rotate = [](const std::vector<float>& input) {
        std::vector<float> output = input;
        for (int pair = 0; pair != rotary_dimension / 2; ++pair) {
            const float angle = static_cast<float>(position) * std::pow(rope_base,
                -2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dimension));
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const int first = pair * 2;
            output[first] = input[first] * cosine - input[first + 1] * sine;
            output[first + 1] = input[first] * sine + input[first + 1] * cosine;
        }
        return output;
    };
    const std::vector<float> expected_query = rotate(query);
    const std::vector<float> expected_key = rotate(key);
    CHECK(metal_test_rope_interleaved(query.data(), key.data(), 1, 1, head_dimension,
                                      rotary_dimension, rotary_dimension, rope_base, position));
    for (int index = 0; index != head_dimension; ++index) {
        CHECK(std::fabs(query[index] - expected_query[index]) < 2.0e-6f);
        CHECK(std::fabs(key[index] - expected_key[index]) < 2.0e-6f);
    }
    std::fprintf(stderr, "PASS interleaved RoPE: %d values\n", head_dimension * 2);
}

void test_partial_interleaved_rope_uses_full_frequency_dimension() {
    constexpr int head_dimension = 8;
    constexpr int rotary_dimension = 4;
    constexpr int frequency_dimension = 8;
    constexpr float rope_base = 16.0f;
    constexpr int position = 2;
    std::vector<float> query = {1, 2, 3, 4, 50, 60, 70, 80};
    std::vector<float> key = {-1, 1, -2, 2, -50, -60, -70, -80};
    const auto rotate = [](const std::vector<float>& input) {
        std::vector<float> output = input;
        for (int pair = 0; pair != rotary_dimension / 2; ++pair) {
            const float angle = static_cast<float>(position) * std::pow(
                rope_base, -2.0f * static_cast<float>(pair) / static_cast<float>(frequency_dimension));
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const int first = pair * 2;
            output[first] = input[first] * cosine - input[first + 1] * sine;
            output[first + 1] = input[first] * sine + input[first + 1] * cosine;
        }
        return output;
    };
    const std::vector<float> expected_query = rotate(query);
    const std::vector<float> expected_key = rotate(key);
    CHECK(metal_test_rope_interleaved(query.data(), key.data(), 1, 1, head_dimension,
                                      rotary_dimension, frequency_dimension, rope_base, position));
    for (int index = 0; index != head_dimension; ++index) {
        CHECK(std::fabs(query[index] - expected_query[index]) < 2.0e-6f);
        CHECK(std::fabs(key[index] - expected_key[index]) < 2.0e-6f);
    }
    std::fprintf(stderr, "PASS partial interleaved RoPE frequency dimension: %d values\n",
                 head_dimension * 2);
}

void test_multisection_half_split_rope_matches_independent_equation() {
    constexpr int head_dimension = 8;
    constexpr int rotary_dimension = 8;
    constexpr float rope_base = 16.0f;
    const int positions[4] = {1, 2, 3, 4};
    const uint32_t sections[4] = {1, 1, 1, 1};
    std::vector<float> query = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> key = {-1, 1, -2, 2, -3, 3, -4, 4};
    const auto rotate = [&](const std::vector<float>& input) {
        std::vector<float> output = input;
        for (int pair = 0; pair != rotary_dimension / 2; ++pair) {
            const float angle = static_cast<float>(positions[pair]) * std::pow(rope_base,
                -2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dimension));
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            output[pair] = input[pair] * cosine - input[pair + rotary_dimension / 2] * sine;
            output[pair + rotary_dimension / 2] = input[pair] * sine + input[pair + rotary_dimension / 2] * cosine;
        }
        return output;
    };
    const std::vector<float> expected_query = rotate(query);
    const std::vector<float> expected_key = rotate(key);
    CHECK(metal_test_rope_multisection(query.data(), key.data(), 1, 1, head_dimension,
                                       rotary_dimension, rotary_dimension, rope_base, positions, sections));
    for (int index = 0; index != head_dimension; ++index) {
        CHECK(std::fabs(query[index] - expected_query[index]) < 2.0e-6f);
        CHECK(std::fabs(key[index] - expected_key[index]) < 2.0e-6f);
    }
    std::fprintf(stderr, "PASS multi-section RoPE: %d values\n", head_dimension * 2);
}

Tensor f32_matrix(std::vector<float>& values, int width, int rows) {
    Tensor tensor;
    tensor.type = GGMLType::F32;
    tensor.n_dims = 2;
    tensor.dims[0] = width;
    tensor.dims[1] = rows;
    tensor.data = reinterpret_cast<const uint8_t*>(values.data());
    return tensor;
}

Tensor f32_vector(std::vector<float>& values) {
    Tensor tensor;
    tensor.type = GGMLType::F32;
    tensor.n_dims = 1;
    tensor.dims[0] = values.size();
    tensor.data = reinterpret_cast<const uint8_t*>(values.data());
    return tensor;
}

void test_fused_query_gate_token_layer_stays_on_one_metal_transaction() {
    constexpr int hidden = 32;
    constexpr int intermediate = 128;
    std::vector<float> norm(hidden, 1.0f), query(hidden * 64), key(hidden * 32), value(hidden * 32),
                       output(hidden * 32), gate(hidden * intermediate), up(hidden * intermediate),
                       down(intermediate * hidden), logits(1);
    Tensor norm_tensor = f32_vector(norm);
    Tensor query_tensor = f32_matrix(query, hidden, 64);
    Tensor key_tensor = f32_matrix(key, hidden, 32);
    Tensor value_tensor = f32_matrix(value, hidden, 32);
    Tensor output_tensor = f32_matrix(output, 32, hidden);
    Tensor gate_tensor = f32_matrix(gate, hidden, intermediate);
    Tensor up_tensor = f32_matrix(up, hidden, intermediate);
    Tensor down_tensor = f32_matrix(down, intermediate, hidden);
    MetalTokLayer layer;
    layer.attn_norm = &norm_tensor;
    layer.attn_q = &query_tensor;
    layer.attn_k = &key_tensor;
    layer.attn_v = &value_tensor;
    layer.attn_o = &output_tensor;
    layer.ffn_norm = &norm_tensor;
    layer.ffn_gate = &gate_tensor;
    layer.ffn_up = &up_tensor;
    layer.ffn_down = &down_tensor;
    layer.H = hidden;
    layer.inter = intermediate;
    layer.Hq = 1;
    layer.Hk = 1;
    layer.Dh = 32;
    layer.rope_dim = 32;
    layer.rope_base = 10000.0f;
    layer.attention_scale = 1.0f / std::sqrt(32.0f);
    layer.rms_eps = 1.0e-6f;
    layer.swiglu = true;
    layer.owns_kv = true;
    layer.is_global = true;
    layer.query_gate_split = true;
    layer.rope_interleaved = true;
    auto session = metal_tok_session_create();
    CHECK(session != nullptr);
    if (!session) return;
    std::vector<float> input(hidden, 0.25f);
    CHECK(metal_tok_session_begin(*session, hidden, intermediate, 0, 0, 0, 1, 1, 32, 4, 1, 0));
    CHECK(metal_tok_session_upload_x(*session, input.data(), hidden));
    CHECK(metal_tok_session_layer(*session, layer));
    CHECK(metal_tok_session_final(*session, norm_tensor, output_tensor, logits.data(), hidden, 1, 1.0e-6f));
    CHECK(std::isfinite(logits[0]));
    const MetalTokMetrics metrics = metal_tok_session_metrics(*session);
    CHECK(metrics.projection_dispatches == 8);
    CHECK(metrics.requested_projection_source_bytes == 69760);
    std::fprintf(stderr, "PASS fused query/gate token layer\n");
}

void test_token_session_accepts_mixed_attention_geometry_with_explicit_capacity() {
    constexpr int hidden = 32;
    constexpr int intermediate = 32;
    std::vector<float> norm(hidden, 1.0f), zero_q(hidden * hidden), zero_kv(hidden * hidden),
                       zero_kv_small(hidden * 16), zero_out(hidden * hidden),
                       zero_ffn(hidden * intermediate), final_norm(hidden, 1.0f), lm(hidden),
                       input(hidden, 0.25f);
    Tensor norm_tensor = f32_vector(norm);
    Tensor q_tensor = f32_matrix(zero_q, hidden, hidden);
    Tensor kv_tensor = f32_matrix(zero_kv, hidden, hidden);
    Tensor kv_small_tensor = f32_matrix(zero_kv_small, hidden, 16);
    Tensor out_tensor = f32_matrix(zero_out, hidden, hidden);
    Tensor ffn_tensor = f32_matrix(zero_ffn, hidden, intermediate);
    Tensor down_tensor = f32_matrix(zero_ffn, intermediate, hidden);
    Tensor final_norm_tensor = f32_vector(final_norm);
    Tensor lm_tensor = f32_matrix(lm, hidden, 1);

    auto layer = [&](int query_heads, int kv_heads, int head_dimension, int cache_id,
                     const Tensor& kv) {
        MetalTokLayer result;
        result.attn_norm = &norm_tensor;
        result.attn_q = &q_tensor;
        result.attn_k = &kv;
        result.attn_v = &kv;
        result.attn_o = &out_tensor;
        result.ffn_norm = &norm_tensor;
        result.ffn_gate = &ffn_tensor;
        result.ffn_up = &ffn_tensor;
        result.ffn_down = &down_tensor;
        result.H = hidden;
        result.inter = intermediate;
        result.Hq = query_heads;
        result.Hk = kv_heads;
        result.Dh = head_dimension;
        result.rope_dim = head_dimension;
        result.rope_frequency_dimension = head_dimension;
        result.rope_base = 10000.0f;
        result.attention_scale = 1.0f;
        result.rms_eps = 1.0e-6f;
        result.swiglu = true;
        result.owns_kv = true;
        result.cache_id = cache_id;
        return result;
    };
    MetalTokLayer wide = layer(1, 1, 32, 0, kv_tensor);
    MetalTokLayer narrow = layer(2, 1, 16, 1, kv_small_tensor);
    wide.cache_width_offset = 0;
    narrow.cache_width_offset = 32;

    auto session = metal_tok_session_create();
    CHECK(session != nullptr);
    if (!session) return;
    metal_dispatch_metrics_reset();
    const MetalTokAttentionCapacity capacity{32, 32, 48};
    CHECK(metal_tok_session_begin_with_attention_capacity(
        *session, hidden, intermediate, 0, 0, 0, capacity, 4, 2, 0));
    CHECK(metal_tok_session_upload_x(*session, input.data(), hidden));
    CHECK(metal_tok_session_layer(*session, wide));
    CHECK(metal_tok_session_layer(*session, narrow));
    float logit = 0.0f;
    CHECK(metal_tok_session_final(*session, final_norm_tensor, lm_tensor,
                                  &logit, hidden, 1, 1.0e-6f));
    CHECK(std::isfinite(logit));
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    CHECK(metal_tok_session_metrics(*session).kv_cache_bytes == 1536);
    std::vector<float> actual(hidden);
    CHECK(metal_tok_session_download_x_for_testing(*session, actual.data(), hidden));
    for (int index = 0; index != hidden; ++index)
        CHECK(std::fabs(actual[index] - input[index]) < 2.0e-6f);
    std::fprintf(stderr, "PASS mixed attention geometry uses declared capacity\n");
}

void test_qk_norm_does_not_normalize_value_projection() {
    constexpr int hidden = 32;
    constexpr int intermediate = 32;
    std::vector<float> attention_norm(hidden), qk_norm(hidden, 1.0f), input(hidden),
                       query(hidden * hidden), key(hidden * hidden), value(hidden * hidden),
                       output(hidden * hidden), gate(hidden * intermediate), up(hidden * intermediate),
                       down(intermediate * hidden), final_norm(hidden, 1.0f), lm(hidden);
    for (int index = 0; index != hidden; ++index) {
        attention_norm[index] = 0.5f + 0.125f * static_cast<float>(index % 5);
        input[index] = 0.25f + 0.03125f * static_cast<float>(index);
        query[index * hidden + index] = 1.0f;
        key[index * hidden + index] = 1.0f;
        value[index * hidden + index] = 1.0f;
        output[index * hidden + index] = 1.0f;
    }
    Tensor attention_norm_tensor = f32_vector(attention_norm);
    Tensor qk_norm_tensor = f32_vector(qk_norm);
    Tensor query_tensor = f32_matrix(query, hidden, hidden);
    Tensor key_tensor = f32_matrix(key, hidden, hidden);
    Tensor value_tensor = f32_matrix(value, hidden, hidden);
    Tensor output_tensor = f32_matrix(output, hidden, hidden);
    Tensor gate_tensor = f32_matrix(gate, hidden, intermediate);
    Tensor up_tensor = f32_matrix(up, hidden, intermediate);
    Tensor down_tensor = f32_matrix(down, intermediate, hidden);
    Tensor final_norm_tensor = f32_vector(final_norm);
    Tensor lm_tensor = f32_matrix(lm, hidden, 1);
    MetalTokLayer layer;
    layer.attn_norm = &attention_norm_tensor;
    layer.attn_q = &query_tensor;
    layer.attn_k = &key_tensor;
    layer.attn_v = &value_tensor;
    layer.attn_o = &output_tensor;
    layer.q_norm = &qk_norm_tensor;
    layer.k_norm = &qk_norm_tensor;
    layer.ffn_norm = &attention_norm_tensor;
    layer.ffn_gate = &gate_tensor;
    layer.ffn_up = &up_tensor;
    layer.ffn_down = &down_tensor;
    layer.H = hidden;
    layer.inter = intermediate;
    layer.Hq = 1;
    layer.Hk = 1;
    layer.Dh = hidden;
    layer.rope_dim = hidden;
    layer.rope_base = 10000.0f;
    layer.attention_scale = 1.0f / std::sqrt(static_cast<float>(hidden));
    layer.rms_eps = 1.0e-6f;
    layer.q_norm_eps = 0.125f;
    layer.k_norm_eps = 0.25f;
    layer.swiglu = true;
    layer.owns_kv = true;
    layer.is_global = true;
    auto session = metal_tok_session_create();
    CHECK(session != nullptr);
    if (!session) return;
    CHECK(metal_tok_session_begin(*session, hidden, intermediate, 0, 0, 0, 1, 1, hidden, 4, 1, 0));
    CHECK(metal_tok_session_upload_x(*session, input.data(), hidden));
    CHECK(metal_tok_session_layer(*session, layer));
    float logit = 0.0f;
    CHECK(metal_tok_session_final(*session, final_norm_tensor, lm_tensor, &logit, hidden, 1, 1.0e-6f));
    std::vector<float> actual(hidden);
    CHECK(metal_tok_session_download_x_for_testing(*session, actual.data(), hidden));
    float sum = 0.0f;
    for (float element : input) sum += element * element;
    const float inverse_rms = 1.0f / std::sqrt(sum / static_cast<float>(hidden) + layer.rms_eps);
    for (int index = 0; index != hidden; ++index) {
        const float projected_value = input[index] * inverse_rms * attention_norm[index];
        CHECK(std::fabs(actual[index] - (input[index] + projected_value)) < 3.0e-5f);
    }
    std::fprintf(stderr, "PASS Q/K norm keeps V projection exact\n");
}

void test_token_layer_uses_declared_attention_scale() {
    constexpr int hidden = 32;
    constexpr int intermediate = 32;
    std::vector<float> norm(hidden, 1.0f), identity(hidden * hidden), zero(hidden * hidden),
                       final_norm(hidden, 1.0f), lm(hidden), first(hidden), second(hidden);
    for (int index = 0; index != hidden; ++index) identity[index * hidden + index] = 1.0f;
    first[0] = std::sqrt(static_cast<float>(hidden));
    second[1] = std::sqrt(static_cast<float>(hidden));
    Tensor norm_tensor = f32_vector(norm);
    Tensor identity_tensor = f32_matrix(identity, hidden, hidden);
    Tensor zero_tensor = f32_matrix(zero, hidden, hidden);
    Tensor final_norm_tensor = f32_vector(final_norm);
    Tensor lm_tensor = f32_matrix(lm, hidden, 1);

    auto run = [&](float attention_scale) {
        MetalTokLayer layer;
        layer.attn_norm = &norm_tensor;
        layer.attn_q = &identity_tensor;
        layer.attn_k = &identity_tensor;
        layer.attn_v = &identity_tensor;
        layer.attn_o = &identity_tensor;
        layer.ffn_norm = &norm_tensor;
        layer.ffn_gate = &zero_tensor;
        layer.ffn_up = &zero_tensor;
        layer.ffn_down = &zero_tensor;
        layer.H = hidden;
        layer.inter = intermediate;
        layer.Hq = 1;
        layer.Hk = 1;
        layer.Dh = hidden;
        layer.rope_dim = hidden;
        layer.rope_base = 10000.0f;
        layer.attention_scale = attention_scale;
        layer.rms_eps = 0.0f;
        layer.swiglu = true;
        layer.owns_kv = true;
        layer.is_global = true;
        auto session = metal_tok_session_create();
        CHECK(session != nullptr);
        std::vector<float> actual(hidden);
        if (!session) return actual;
        float logit = 0.0f;
        for (int position = 0; position != 2; ++position) {
            const float* input = position == 0 ? first.data() : second.data();
            CHECK(metal_tok_session_begin(*session, hidden, intermediate, 0, 0, 0,
                                          1, 1, hidden, 4, 1, position));
            CHECK(metal_tok_session_upload_x(*session, input, hidden));
            CHECK(metal_tok_session_layer(*session, layer));
            CHECK(metal_tok_session_final(*session, final_norm_tensor, lm_tensor,
                                          &logit, hidden, 1, 0.0f));
        }
        CHECK(metal_tok_session_download_x_for_testing(*session, actual.data(), hidden));
        return actual;
    };

    for (float scale : {0.01f, 0.2f}) {
        const std::vector<float> actual = run(scale);
        const float prior_probability = 1.0f / (1.0f + std::exp(32.0f * scale));
        const float root = std::sqrt(32.0f);
        CHECK(std::fabs(actual[0] - prior_probability * root) < 3.0e-4f);
        CHECK(std::fabs(actual[1] - (2.0f - prior_probability) * root) < 3.0e-4f);
    }
    std::fprintf(stderr, "PASS token layer uses declared attention scale\n");
}

} // namespace

// ponytail: quantize using the actual block layouts from kernels.h.
// One function per block family, parameterized by nothing. Universal.
static void pack_q4_0(const float* w, int K, int N, uint8_t* out) {
    int nb = K/32; auto blk = (block_q4_0*)out;
    for (int j = 0; j < N; j++) for (int b = 0; b < nb; b++) {
        const float* x = w + j*K + b*32; float amax = 0;
        for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax/8; blk[(uint64_t)j*nb+b].d = f2h(d);
        float id = d > 0 ? 1/d : 0;
        for (int i = 0; i < 16; i++) {
            int q0 = roundf(x[i]*id+8), q1 = roundf(x[i+16]*id+8);
            blk[(uint64_t)j*nb+b].qs[i] = (q0&0xF)|((q1&0xF)<<4);
        }
    }
}
static void pack_f16(const float* w, int K, int N, uint8_t* out) {
    auto* packed = reinterpret_cast<uint16_t*>(out);
    for (int index = 0; index != K * N; ++index) packed[index] = f2h(w[index]);
}
static void pack_q5_0(const float* w, int K, int N, uint8_t* out) {
    int nb = K/32; auto blk = (block_q5_0*)out;
    for (int j = 0; j < N; j++) for (int b = 0; b < nb; b++) {
        const float* x = w + j*K + b*32; float amax = 0;
        for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax/16; blk[(uint64_t)j*nb+b].d = f2h(d); blk[(uint64_t)j*nb+b].qh = 0;
        float id = d > 0 ? 1/d : 0; uint32_t qh = 0;
        for (int i = 0; i < 16; i++) {
            int q0 = roundf(x[i]*id+16), q1 = roundf(x[i+16]*id+16);
            blk[(uint64_t)j*nb+b].qs[i] = (q0&0xF)|((q1&0xF)<<4);
            qh |= ((q0>>4)&1)<<i; qh |= ((q1>>4)&1)<<(i+16);
        }
        blk[(uint64_t)j*nb+b].qh = qh;
    }
}
static void pack_q8_0(const float* w, int K, int N, uint8_t* out) {
    int nb = K/32; auto blk = (block_q8_0*)out;
    for (int j = 0; j < N; j++) for (int b = 0; b < nb; b++) {
        const float* x = w + j*K + b*32; float amax = 0;
        for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax/127; blk[(uint64_t)j*nb+b].d = f2h(d);
        float id = d > 0 ? 1/d : 0;
        for (int i = 0; i < 32; i++) blk[(uint64_t)j*nb+b].qs[i] = (int8_t)fmaxf(-127, fminf(127, roundf(x[i]*id)));
    }
}
static void pack_q4_K(const float* w, int K, int N, uint8_t* out) {
    int nb = K/256; auto blk = (block_q4_K*)out;
    unsigned rs = 999;
    for (int j = 0; j < N; j++) for (int b = 0; b < nb; b++) {
        const float* x = w + j*K + b*256; float amax = 0;
        for (int i = 0; i < 256; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax/15; auto& bb = blk[(uint64_t)j*nb+b]; bb.d = f2h(d); bb.dmin = f2h(amax/240);
        for (int s = 0; s < 12; s++) { rs = rs*1103515245+12345; bb.scales[s] = (uint8_t)(rs >> 16); }
        float id = d > 0 ? 1/d : 0;
        for (int jb = 0; jb < 4; jb++) {
            for (int l = 0; l < 32; l++) {
                int q0 = (int)roundf(x[jb * 64 + l] * id);
                int q1 = (int)roundf(x[jb * 64 + 32 + l] * id);
                bb.qs[jb * 32 + l] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
            }
        }
    }
}
static void pack_q6_K(const float* w, int K, int N, uint8_t* out) {
    int nb = K/256; auto blk = (block_q6_K*)out;
    unsigned rs = 777;
    for (int j = 0; j < N; j++) for (int b = 0; b < nb; b++) {
        const float* x = w + j*K + b*256; float amax = 0;
        for (int i = 0; i < 256; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax/31; auto& bb = blk[(uint64_t)j*nb+b]; bb.d = f2h(d);
        memset(bb.ql, 0, 128); memset(bb.qh, 0, 64);
        for (int s = 0; s < 16; s++) { rs = rs*1103515245+12345; bb.scales[s] = (int8_t)((rs >> 16) % 64 - 32); }
        float id = d > 0 ? 1/d : 0;
        for (int n = 0; n < 256; n++) {
            int val = fmaxf(0, fminf(63, roundf(x[n]*id+32)));
            int lo = val & 0xF, hi = (val >> 4) & 3;
            int half = n/128, l = n%128;
            if (l < 64) { bb.ql[half*64 + l] = (bb.ql[half*64 + l] & 0xF0) | lo; }
            else { bb.ql[half*64 + l - 64] = (bb.ql[half*64 + l - 64] & 0x0F) | (lo << 4); }
            int qh_byte = half*32 + l%32;
            bb.qh[qh_byte] |= hi << ((l/16)*2);
        }
    }
}
static void pack_q2_K(const float* w, int K, int N, uint8_t* out) {
    int nb = K/256; auto blk = (block_q2_K*)out;
    unsigned rs = 222;
    for (int j = 0; j < N; j++) for (int b = 0; b < nb; b++) {
        const float* x = w + j*K + b*256; float amax = 0;
        for (int i = 0; i < 256; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax/3; auto& bb = blk[(uint64_t)j*nb+b]; bb.d = f2h(d); bb.dmin = f2h(amax/15);
        for (int s = 0; s < 16; s++) { rs = rs*1103515245+12345; bb.scales[s] = (uint8_t)(rs >> 16); }
        float id = d > 0 ? 1/d : 0;
        memset(bb.qs, 0, 64);
        for (int e = 0; e < 256; e++) {
            int q = fmaxf(0, fminf(3, roundf(x[e]*id)));
            int sub = e/16, l = e%16;
            int half = sub/8, jj = sub%8, group = jj/2, lo = jj%2;
            bb.qs[half*32 + lo*16 + l] |= (uint8_t)(q << (group*2));
        }
    }
}

static void pack_synthetic_iq2_xxs(int K, int N, uint8_t* output) {
    const int blocks_per_row = K / 256;
    for (int row = 0; row < N; ++row) {
        for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
            uint8_t* block = output +
                (static_cast<size_t>(row) * blocks_per_row + block_index) * 66;
            std::memset(block, 0, 66);
            const uint16_t d = f2h(0.03125f * (1 + ((row + block_index) & 3)));
            std::memcpy(block, &d, sizeof(d));
            for (int group32 = 0; group32 < 8; ++group32) {
                uint8_t* packed = block + 2 + group32 * 8;
                for (int group8 = 0; group8 < 4; ++group8)
                    packed[group8] = static_cast<uint8_t>((row + block_index + group32 + group8) & 3);
                uint32_t signs_and_scale =
                    static_cast<uint32_t>((row * 13 + block_index * 11 + group32 * 7 + 0) & 127) |
                    (static_cast<uint32_t>((row * 13 + block_index * 11 + group32 * 7 + 5) & 127) << 7) |
                    (static_cast<uint32_t>((row * 13 + block_index * 11 + group32 * 7 + 9) & 127) << 14) |
                    (static_cast<uint32_t>((row * 13 + block_index * 11 + group32 * 7 + 17) & 127) << 21) |
                    (static_cast<uint32_t>((row + block_index + group32) & 15) << 28);
                std::memcpy(packed + 4, &signs_and_scale, sizeof(signs_and_scale));
            }
        }
    }
}

static std::array<uint8_t, 8> reference_iq2_xxs_grid(uint8_t index) {
    switch (index) {
        case 0: return {8, 8, 8, 8, 8, 8, 8, 8};
        case 1: return {43, 8, 8, 8, 8, 8, 8, 8};
        case 2: return {25, 25, 8, 8, 8, 8, 8, 8};
        case 3: return {8, 43, 8, 8, 8, 8, 8, 8};
        default: std::abort();
    }
}

static void reference_iq2_xxs_matmul(const std::vector<float>& input,
                                     const Tensor& weight,
                                     std::vector<float>& output, int K, int N) {
    const int blocks_per_row = K / 256;
    output.assign(N, 0.0f);
    for (int row = 0; row < N; ++row) {
        double sum = 0.0;
        for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
            const uint8_t* block = weight.data +
                (static_cast<size_t>(row) * blocks_per_row + block_index) * 66;
            uint16_t d_bits = 0;
            std::memcpy(&d_bits, block, sizeof(d_bits));
            const float d = fp16_to_fp32(d_bits);
            for (int group32 = 0; group32 < 8; ++group32) {
                const uint8_t* packed = block + 2 + group32 * 8;
                uint32_t signs_and_scale = 0;
                std::memcpy(&signs_and_scale, packed + 4, sizeof(signs_and_scale));
                const float scale = d * (0.5f + static_cast<float>(signs_and_scale >> 28)) * 0.25f;
                for (int group8 = 0; group8 < 4; ++group8) {
                    const auto grid = reference_iq2_xxs_grid(packed[group8]);
                    const uint8_t sign_index =
                        static_cast<uint8_t>((signs_and_scale >> (7 * group8)) & 127u);
                    const uint8_t signs = static_cast<uint8_t>(
                        sign_index | ((__builtin_popcount(sign_index) & 1) << 7));
                    for (int lane = 0; lane < 8; ++lane) {
                        const int k = block_index * 256 + group32 * 32 + group8 * 8 + lane;
                        const float value = scale * grid[lane] *
                            ((signs & (1u << lane)) ? -1.0f : 1.0f);
                        sum += static_cast<double>(input[k]) * value;
                    }
                }
            }
        }
        output[row] = static_cast<float>(sum);
    }
}

static void test_iq2_xxs_gemv_matches_independent_decoder() {
    constexpr int K = 1024;
    constexpr int N = 9;
    std::vector<float> input(K);
    rng_fill(input.data(), K, 2026);
    for (float& value : input) value *= 0.01f;
    std::vector<uint8_t> bytes(static_cast<size_t>(N) * (K / 256) * 66);
    pack_synthetic_iq2_xxs(K, N, bytes.data());
    Tensor weight;
    weight.type = GGMLType::IQ2_XXS;
    weight.n_dims = 2;
    weight.dims[0] = K;
    weight.dims[1] = N;
    weight.data = bytes.data();
    std::vector<float> expected;
    std::vector<float> actual(N);
    reference_iq2_xxs_matmul(input, weight, expected, K, N);
    CHECK(metal_gemv(input.data(), weight, actual.data(), K, N));
    for (int row = 0; row < N; ++row)
        CHECK(almost_equal(actual[row], expected[row], 1.0e-5f, 1.0e-5f));
}

static void test_iq2_xxs_real_shape_parity_and_throughput() {
    constexpr int K = 5120;
    constexpr int N = 17408;
    constexpr int repetitions = 5;
    std::vector<float> input(K);
    rng_fill(input.data(), K, 2031);
    for (float& value : input) value *= 0.01f;
    const size_t weight_bytes = static_cast<size_t>(N) * (K / 256) * 66;
    std::vector<uint8_t> bytes(weight_bytes);
    pack_synthetic_iq2_xxs(K, N, bytes.data());
    Tensor weight;
    weight.type = GGMLType::IQ2_XXS;
    weight.n_dims = 2;
    weight.dims[0] = K;
    weight.dims[1] = N;
    weight.data = bytes.data();
    std::vector<float> expected;
    std::vector<float> actual(N);
    reference_iq2_xxs_matmul(input, weight, expected, K, N);

    const size_t q4_bytes = static_cast<size_t>(N) * (K / 256) * sizeof(block_q4_K);
    const size_t q6_bytes = static_cast<size_t>(N) * (K / 256) * sizeof(block_q6_K);
    std::vector<uint8_t> q4_storage(q4_bytes);
    std::vector<uint8_t> q6_storage(q6_bytes);
    auto* q4_blocks = reinterpret_cast<block_q4_K*>(q4_storage.data());
    auto* q6_blocks = reinterpret_cast<block_q6_K*>(q6_storage.data());
    const size_t block_count = static_cast<size_t>(N) * (K / 256);
    for (size_t block = 0; block < block_count; ++block) {
        q4_blocks[block].d = f2h(0.01f);
        q4_blocks[block].dmin = f2h(0.001f);
        for (int i = 0; i < 12; ++i) q4_blocks[block].scales[i] = static_cast<uint8_t>(1 + ((block + i) % 63));
        for (int i = 0; i < 128; ++i) q4_blocks[block].qs[i] = static_cast<uint8_t>(block + i * 17);
        q6_blocks[block].d = f2h(0.001f);
        for (int i = 0; i < 16; ++i) q6_blocks[block].scales[i] = static_cast<int8_t>(1 + ((block + i) % 31));
        for (int i = 0; i < 128; ++i) q6_blocks[block].ql[i] = static_cast<uint8_t>(block + i * 11);
        for (int i = 0; i < 64; ++i) q6_blocks[block].qh[i] = static_cast<uint8_t>(block + i * 7);
    }
    Tensor q4_weight = weight;
    q4_weight.type = GGMLType::Q4_K;
    q4_weight.data = q4_storage.data();
    Tensor q6_weight = weight;
    q6_weight.type = GGMLType::Q6_K;
    q6_weight.data = q6_storage.data();
    std::vector<float> control_output(N);

    CHECK(metal_gemv_repeat(input.data(), q4_weight, control_output.data(), K, N, 1));
    CHECK(metal_gemv_repeat(input.data(), weight, actual.data(), K, N, 1));
    CHECK(metal_gemv_repeat(input.data(), q6_weight, control_output.data(), K, N, 1));
    const auto timed = [&](const Tensor& candidate, std::vector<float>& output) {
        metal_dispatch_metrics_reset();
        CHECK(metal_gemv_repeat(input.data(), candidate, output.data(), K, N, 1));
        const MetalDispatchMetrics metrics = metal_dispatch_metrics();
        CHECK(metrics.command_buffers == 1);
        CHECK(metrics.gpu_time_ms > 0.0);
        return metrics.gpu_time_ms;
    };
    std::vector<double> q4_times, iq2_times, q6_times;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        if ((repetition & 1) == 0) {
            q4_times.push_back(timed(q4_weight, control_output));
            iq2_times.push_back(timed(weight, actual));
            q6_times.push_back(timed(q6_weight, control_output));
        } else {
            q6_times.push_back(timed(q6_weight, control_output));
            iq2_times.push_back(timed(weight, actual));
            q4_times.push_back(timed(q4_weight, control_output));
        }
    }
    float max_error = 0.0f;
    float max_value = 0.0f;
    for (int row = 0; row < N; ++row) {
        max_error = std::max(max_error, std::fabs(actual[row] - expected[row]));
        max_value = std::max(max_value, std::fabs(expected[row]));
    }
    const float relative_error = max_value > 0.0f ? max_error / max_value : max_error;
    CHECK(relative_error < 1.0e-5f);
    const auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };
    const double gpu_ms_per_dispatch = median(iq2_times);
    const double requested_gb_per_second =
        static_cast<double>(weight_bytes) / 1.0e9 / (gpu_ms_per_dispatch / 1000.0);
    std::fprintf(stderr,
                 "IQ2_XXS real shape: K=%d N=%d bytes=%zu gpu_ms=%.6f requested_GB_s=%.3f rel=%.9g cb=1\n",
                 K, N, weight_bytes, gpu_ms_per_dispatch, requested_gb_per_second,
                 relative_error);
    std::fprintf(stderr,
                 "shape controls: Q4_K bytes=%zu median=%.6f range=%.6f..%.6f; "
                 "IQ2_XXS median=%.6f range=%.6f..%.6f; "
                 "Q6_K bytes=%zu median=%.6f range=%.6f..%.6f\n",
                 q4_bytes, median(q4_times), *std::min_element(q4_times.begin(), q4_times.end()),
                 *std::max_element(q4_times.begin(), q4_times.end()), gpu_ms_per_dispatch,
                 *std::min_element(iq2_times.begin(), iq2_times.end()),
                 *std::max_element(iq2_times.begin(), iq2_times.end()), q6_bytes, median(q6_times),
                 *std::min_element(q6_times.begin(), q6_times.end()),
                 *std::max_element(q6_times.begin(), q6_times.end()));
    uint32_t pipeline_width = 0;
    uint32_t maximum_threads = 0;
    CHECK(metal_test_iq2_xxs_pipeline_limits(&pipeline_width, &maximum_threads));
    CHECK(pipeline_width == 32);
    CHECK(maximum_threads >= 64);
    std::fprintf(stderr, "IQ2_XXS pipeline: width=%u max_threads=%u\n",
                 pipeline_width, maximum_threads);
}

static void test_iq2_xxs_registered_ffn_shapes_submit() {
    constexpr int hidden = 5120;
    constexpr int intermediate = 17408;
    const size_t tensor_bytes = static_cast<size_t>(intermediate) * (hidden / 256) * 66;
    const long page_value = sysconf(_SC_PAGESIZE);
    CHECK(page_value > 0);
    if (page_value <= 0) return;
    const size_t logical_bytes = 3 * tensor_bytes;
    const size_t page = static_cast<size_t>(page_value);
    const size_t mapped_bytes = (logical_bytes + page - 1) / page * page;
    void* mapped = mmap(nullptr, mapped_bytes, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(mapped != MAP_FAILED);
    if (mapped == MAP_FAILED) return;
    auto* storage = static_cast<uint8_t*>(mapped);
    pack_synthetic_iq2_xxs(hidden, intermediate, storage);
    std::memcpy(storage + tensor_bytes, storage, tensor_bytes);
    std::memcpy(storage + 2 * tensor_bytes, storage, tensor_bytes);

    Tensor gate;
    gate.type = GGMLType::IQ2_XXS;
    gate.n_dims = 2;
    gate.dims[0] = hidden;
    gate.dims[1] = intermediate;
    gate.data = storage;
    Tensor up = gate;
    up.data = storage + tensor_bytes;
    Tensor down = gate;
    down.dims[0] = intermediate;
    down.dims[1] = hidden;
    down.data = storage + 2 * tensor_bytes;

    std::vector<float> input(hidden), output(hidden);
    rng_fill(input.data(), hidden, 2041);
    for (float& value : input) value *= 0.01f;
    auto session = metal_tok_session_create();
    CHECK(session != nullptr);
    CHECK(session && metal_tok_session_register_weights(*session, mapped, mapped_bytes));
    double gpu_ms = 0.0;
    metal_dispatch_metrics_reset();
    CHECK(session && metal_tok_session_probe_ffn_for_testing(
        *session, input.data(), gate, up, down, nullptr, 0, output.data(),
        hidden, intermediate, &gpu_ms));
    const MetalDispatchMetrics metrics = metal_dispatch_metrics();
    CHECK(metrics.command_buffers == 1);
    CHECK(gpu_ms > 0.0);
    for (float value : output) CHECK(std::isfinite(value));
    std::fprintf(stderr,
                 "IQ2_XXS registered FFN shapes: gate/up K=%d N=%d down K=%d N=%d bytes=%zu cb=%llu gpu_ms=%.6f\n",
                 hidden, intermediate, intermediate, hidden, mapped_bytes,
                 static_cast<unsigned long long>(metrics.command_buffers), gpu_ms);
    if (session) metal_tok_session_unregister_weights(*session, mapped);
    munmap(mapped, mapped_bytes);
}

static void pack_synthetic_iq1_s(int K, int N, uint8_t* output) {
    const int blocks_per_row = K / 256;
    auto* blocks = reinterpret_cast<block_iq1_s*>(output);
    for (int row = 0; row != N; ++row) {
        for (int block_index = 0; block_index != blocks_per_row; ++block_index) {
            block_iq1_s& block = blocks[static_cast<size_t>(row) * blocks_per_row + block_index];
            block = {};
            block.d = f2h(0.015625f * (1 + ((row + block_index) & 3)));
            for (int group32 = 0; group32 != 8; ++group32) {
                uint16_t high = static_cast<uint16_t>(
                    ((row + block_index + group32) & 7) << 12);
                if ((row + block_index + group32) & 1) high |= 0x8000;
                for (int group8 = 0; group8 != 4; ++group8) {
                    const uint16_t grid = static_cast<uint16_t>(
                        (row * 17 + block_index * 29 + group32 * 43 + group8 * 509) & 2047);
                    block.qs[4 * group32 + group8] = static_cast<uint8_t>(grid);
                    high |= static_cast<uint16_t>((grid >> 8) << (3 * group8));
                }
                block.qh[group32] = high;
            }
        }
    }
}

static void reference_iq1_s_matmul(const std::vector<float>& input, const Tensor& weight,
                                   std::vector<float>& output, int K, int N) {
    const int blocks_per_row = K / 256;
    const auto* blocks = reinterpret_cast<const block_iq1_s*>(weight.data);
    output.assign(N, 0.0f);
    for (int row = 0; row != N; ++row) {
        double sum = 0.0;
        for (int block_index = 0; block_index != blocks_per_row; ++block_index) {
            const block_iq1_s& block = blocks[static_cast<size_t>(row) * blocks_per_row + block_index];
            const float d = fp16_to_fp32(block.d);
            for (int group32 = 0; group32 != 8; ++group32) {
                const uint16_t high = block.qh[group32];
                const float scale = d * static_cast<float>(2 * ((high >> 12) & 7) + 1);
                const float delta = (high & 0x8000) ? -0.125f : 0.125f;
                for (int group8 = 0; group8 != 4; ++group8) {
                    const uint16_t grid_index = static_cast<uint16_t>(
                        block.qs[4 * group32 + group8] |
                        (((high >> (3 * group8)) & 7) << 8));
                    const uint64_t grid = kIq1SGrid[grid_index];
                    for (int lane = 0; lane != 8; ++lane) {
                        const int8_t level = static_cast<int8_t>(grid >> (8 * lane));
                        const int k = block_index * 256 + group32 * 32 + group8 * 8 + lane;
                        sum += static_cast<double>(input[k]) *
                               scale * (static_cast<float>(level) + delta);
                    }
                }
            }
        }
        output[row] = static_cast<float>(sum);
    }
}

static void test_iq1_s_gemv_matches_independent_decoder() {
    constexpr int K = 512;
    constexpr int N = 9;
    std::vector<float> input(K);
    rng_fill(input.data(), K, 2041);
    for (float& value : input) value *= 0.01f;
    std::vector<uint8_t> bytes(static_cast<size_t>(N) * (K / 256) * 50);
    pack_synthetic_iq1_s(K, N, bytes.data());
    Tensor weight;
    weight.type = GGMLType::IQ1_S;
    weight.n_dims = 2;
    weight.dims[0] = K;
    weight.dims[1] = N;
    weight.data = bytes.data();
    std::vector<float> expected;
    std::vector<float> actual(N);
    reference_iq1_s_matmul(input, weight, expected, K, N);
    CHECK(metal_gemv(input.data(), weight, actual.data(), K, N));
    for (int row = 0; row != N; ++row)
        CHECK(almost_equal(actual[row], expected[row], 1.0e-5f, 1.0e-5f));
}

static void test_iq1_s_real_shape_parity_and_throughput() {
    constexpr int K = 5120;
    constexpr int N = 17408;
    constexpr int repetitions = 5;
    std::vector<float> input(K);
    rng_fill(input.data(), K, 2043);
    for (float& value : input) value *= 0.01f;
    const size_t weight_bytes = static_cast<size_t>(N) * (K / 256) * 50;
    std::vector<uint8_t> bytes(weight_bytes);
    pack_synthetic_iq1_s(K, N, bytes.data());
    Tensor weight;
    weight.type = GGMLType::IQ1_S;
    weight.n_dims = 2;
    weight.dims[0] = K;
    weight.dims[1] = N;
    weight.data = bytes.data();
    std::vector<float> expected;
    std::vector<float> actual(N);
    reference_iq1_s_matmul(input, weight, expected, K, N);

    CHECK(metal_gemv_repeat(input.data(), weight, actual.data(), K, N, 1));
    std::vector<double> times;
    for (int repetition = 0; repetition != repetitions; ++repetition) {
        metal_dispatch_metrics_reset();
        CHECK(metal_gemv_repeat(input.data(), weight, actual.data(), K, N, 1));
        const MetalDispatchMetrics metrics = metal_dispatch_metrics();
        CHECK(metrics.command_buffers == 1);
        CHECK(metrics.gpu_time_ms > 0.0);
        times.push_back(metrics.gpu_time_ms);
    }
    float max_error = 0.0f;
    float max_value = 0.0f;
    for (int row = 0; row != N; ++row) {
        max_error = std::max(max_error, std::fabs(actual[row] - expected[row]));
        max_value = std::max(max_value, std::fabs(expected[row]));
    }
    const float relative_error = max_value > 0.0f ? max_error / max_value : max_error;
    CHECK(relative_error < 1.0e-5f);
    std::sort(times.begin(), times.end());
    const double median = times[times.size() / 2];
    const double requested_gb_per_second =
        static_cast<double>(weight_bytes) / 1.0e9 / (median / 1000.0);
    uint32_t width = 0;
    uint32_t maximum_threads = 0;
    CHECK(metal_test_iq1_s_pipeline_limits(&width, &maximum_threads));
    CHECK(width == 32);
    CHECK(maximum_threads >= 64);
    std::fprintf(stderr,
                 "IQ1_S real shape: K=%d N=%d bytes=%zu gpu_ms=%.6f range=%.6f..%.6f "
                 "requested_GB_s=%.3f rel=%.9g cb=1 width=%u max_threads=%u\n",
                 K, N, weight_bytes, median, times.front(), times.back(),
                 requested_gb_per_second, relative_error, width, maximum_threads);
}

static void test_mpp_int2_m1_real_shape_admission() {
    constexpr int K = 5120;
    constexpr int N = 17408;
    constexpr int block = 32;
    std::vector<float> input(K);
    for (int k = 0; k != K; ++k)
        input[k] = static_cast<float>((k % 17) - 8) / 32.0f;
    const size_t data_size = static_cast<size_t>(N) * K / 4;
    const size_t scale_size = static_cast<size_t>(N) * (K / block);
    // Four signed Int2 values of +1 per byte. E8M0 exponent 127 is scale 1.
    std::vector<uint8_t> packed(data_size, 0x55);
    std::vector<uint8_t> scales(scale_size, 127);
    std::vector<float> output(N);
    double gpu_ms = 0.0;
    uint64_t data_bytes = 0;
    uint64_t scale_bytes = 0;
    uint32_t width = 0;
    uint32_t maximum_threads = 0;
    std::vector<double> times;
    for (int sample = 0; sample != 6; ++sample) {
        const bool executed = metal_test_mpp_int2_m1(
            input.data(), packed.data(), scales.data(), output.data(), K, N,
            &gpu_ms, &data_bytes, &scale_bytes, &width, &maximum_threads);
        CHECK(executed);
        if (!executed) return;
        if (sample != 0) times.push_back(gpu_ms);
    }
    std::sort(times.begin(), times.end());
    gpu_ms = times[times.size() / 2];
    float expected = 0.0f;
    for (float value : input) expected += fp16_to_fp32(fp32_to_fp16(value));
    for (float value : output) CHECK(almost_equal(value, expected, 1.0e-5f, 1.0e-4f));
    CHECK(data_bytes == data_size);
    CHECK(scale_bytes == scale_size);
    CHECK(gpu_ms > 0.0);
    CHECK(width > 0);
    CHECK(maximum_threads >= width);
    std::fprintf(stderr,
                 "MPP Int2 M=1 admission: K=%d N=%d data=%llu scales=%llu "
                 "gpu_ms_median=%.6f range=%.6f..%.6f width=%u max_threads=%u\n",
                 K, N, static_cast<unsigned long long>(data_bytes),
                 static_cast<unsigned long long>(scale_bytes), gpu_ms,
                 times.front(), times.back(), width, maximum_threads);
}

static void test_affine_u2_block256_real_shape() {
    constexpr int K = 5120;
    constexpr int N = 17408;
    constexpr int block = 256;
    constexpr int blocks_per_row = K / block;
    std::vector<float> input(K);
    for (int k = 0; k != K; ++k)
        input[k] = static_cast<float>((k % 31) - 15) / 64.0f;
    std::vector<uint8_t> packed(static_cast<size_t>(N) * K / 4, 0xe4);
    std::vector<uint16_t> scales(static_cast<size_t>(N) * blocks_per_row);
    std::vector<uint16_t> biases(scales.size());
    for (size_t group = 0; group != scales.size(); ++group) {
        scales[group] = fp32_to_fp16(static_cast<float>(1 + group % 7) / 64.0f);
        biases[group] = fp32_to_fp16(static_cast<float>(static_cast<int>(group % 5) - 2) / 128.0f);
    }
    std::array<double, blocks_per_row> sum_x{};
    std::array<double, blocks_per_row> sum_qx{};
    for (int k = 0; k != K; ++k) {
        sum_x[k / block] += input[k];
        sum_qx[k / block] += input[k] * static_cast<double>(k & 3);
    }
    std::vector<float> expected(N);
    for (int row = 0; row != N; ++row) {
        double sum = 0.0;
        for (int ib = 0; ib != blocks_per_row; ++ib) {
            const size_t group = static_cast<size_t>(row) * blocks_per_row + ib;
            sum += fp16_to_fp32(scales[group]) * sum_qx[ib] +
                   fp16_to_fp32(biases[group]) * sum_x[ib];
        }
        expected[row] = static_cast<float>(sum);
    }

    std::vector<float> actual(N);
    std::vector<double> times;
    uint64_t requested_bytes = 0;
    uint32_t width = 0;
    uint32_t maximum_threads = 0;
    std::array<double, 15> samples{};
    const bool executed = metal_test_affine_u2_block256(
        input.data(), packed.data(), scales.data(), biases.data(), actual.data(),
        K, N, samples.data(), static_cast<uint32_t>(samples.size()),
        4, true,
        &requested_bytes, &width, &maximum_threads);
    CHECK(executed);
    if (!executed) return;
    times.assign(samples.end() - 5, samples.end());
    float max_delta = 0.0f;
    float max_reference = 0.0f;
    for (int row = 0; row != N; ++row) {
        CHECK(std::isfinite(actual[row]));
        max_delta = std::max(max_delta, std::fabs(actual[row] - expected[row]));
        max_reference = std::max(max_reference, std::fabs(expected[row]));
    }
    CHECK(max_delta <= 1.0e-4f + 2.0e-5f * max_reference);
    std::sort(times.begin(), times.end());
    const double median = times[times.size() / 2];
    const double requested_gb_per_second =
        static_cast<double>(requested_bytes) / 1.0e9 / (median / 1000.0);
    std::array<double, 7> reused_samples{};
    CHECK(metal_test_affine_u2_block256(
        input.data(), packed.data(), scales.data(), biases.data(), actual.data(),
        K, N, reused_samples.data(), static_cast<uint32_t>(reused_samples.size()),
        20, false, &requested_bytes, &width, &maximum_threads));
    std::vector<double> reused_times(reused_samples.end() - 5, reused_samples.end());
    std::sort(reused_times.begin(), reused_times.end());
    const double reused_median = reused_times[reused_times.size() / 2];
    CHECK(requested_bytes == static_cast<uint64_t>(N) * (64 + 2 + 2) * blocks_per_row);
    CHECK(width == 32);
    CHECK(maximum_threads >= 64);
    std::fprintf(stderr,
                 "Affine UInt2/256 real shape: K=%d N=%d bytes=%llu gpu_ms=%.6f "
                 "range=%.6f..%.6f requested_GB_s=%.3f max_delta=%.9g "
                 "cb=1 distinct_dispatches=4 "
                 "same_weight_20_median=%.6f same_weight_range=%.6f..%.6f "
                 "width=%u max_threads=%u\n",
                 K, N, static_cast<unsigned long long>(requested_bytes), median,
                 times.front(), times.back(), requested_gb_per_second, max_delta,
                 reused_median, reused_times.front(), reused_times.back(),
                 width, maximum_threads);
}

static void test_affine_u2_variant_contract() {
    constexpr int K = 512;
    constexpr int N = 64;
    constexpr int block = 256;
    std::vector<float> input(K);
    for (int k = 0; k != K; ++k)
        input[k] = static_cast<float>((k % 23) - 11) / 32.0f;
    std::vector<uint8_t> packed(static_cast<size_t>(N) * K / 4);
    for (size_t index = 0; index != packed.size(); ++index)
        packed[index] = static_cast<uint8_t>((index * 73u + 19u) & 0xffu);
    std::vector<uint16_t> scales(static_cast<size_t>(N) * (K / block));
    std::vector<uint16_t> biases(scales.size());
    for (size_t index = 0; index != scales.size(); ++index) {
        scales[index] = fp32_to_fp16(static_cast<float>(1 + index % 5) / 32.0f);
        biases[index] = fp32_to_fp16(static_cast<float>(static_cast<int>(index % 3) - 1) / 64.0f);
    }
    std::vector<float> expected(N, 0.0f);
    for (int row = 0; row != N; ++row) {
        double sum = 0.0;
        for (int k = 0; k != K; ++k) {
            const size_t plane = static_cast<size_t>(row) * (K / block) + k / block;
            const uint8_t byte = packed[static_cast<size_t>(row) * (K / 4) + k / 4];
            const float q = static_cast<float>((byte >> (2 * (k & 3))) & 3u);
            sum += input[k] *
                   (fp16_to_fp32(scales[plane]) * q +
                    fp16_to_fp32(biases[plane]));
        }
        expected[row] = static_cast<float>(sum);
    }
    for (const uint32_t rows_per_simd : {1u, 2u, 4u, 8u}) {
        for (const uint32_t simdgroups : {1u, 2u, 4u}) {
            for (const bool masked_qdot : {false, true}) {
                for (const uint32_t values_per_lane : {8u, 16u}) {
                    std::vector<float> actual(N);
                    double gpu_ms = 0.0;
                    uint64_t requested_bytes = 0;
                    uint32_t width = 0;
                    uint32_t maximum_threads = 0;
                    CHECK(metal_test_affine_u2_block256_variant(
                        input.data(), packed.data(), scales.data(), biases.data(), actual.data(),
                        K, N, rows_per_simd, simdgroups, masked_qdot, true,
                        values_per_lane, &gpu_ms, 1, 1, false,
                        &requested_bytes, &width, &maximum_threads));
                    CHECK(requested_bytes ==
                          static_cast<uint64_t>(N) * (64 + 2 + 2) * (K / block));
                    CHECK(width == 32);
                    CHECK(maximum_threads >= 32 * simdgroups);
                    for (int row = 0; row != N; ++row) {
                        CHECK_MSG(
                            almost_equal(actual[row], expected[row], 1.0e-4f, 2.0e-5f),
                            "affine U2 variant rows=%u simdgroups=%u masked=%d "
                            "values_per_lane=%u row=%d actual=%g expected=%g",
                            rows_per_simd, simdgroups, masked_qdot, values_per_lane, row,
                            actual[row], expected[row]);
                    }
                }
            }
        }
    }
    CHECK(!metal_test_affine_u2_block256_variant(
        input.data(), packed.data(), scales.data(), biases.data(), expected.data(),
        K, N, 3, 2, false, false, 16, nullptr, 0, 1, false,
        nullptr, nullptr, nullptr));
    CHECK(!metal_test_affine_u2_block256_variant(
        input.data(), packed.data(), scales.data(), biases.data(), expected.data(),
        K, N, 4, 1, true, true, 32, nullptr, 0, 1, false,
        nullptr, nullptr, nullptr));
}

static void test_affine_u2_variant_finite_edge_domain() {
    constexpr int K = 512;
    constexpr int N = 8;
    constexpr int block = 256;
    const std::array<float, 16> edge_values = {
        0.0f,
        -0.0f,
        std::numeric_limits<float>::denorm_min(),
        -std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(),
        -std::numeric_limits<float>::min(),
        1.0e-30f,
        -1.0e-30f,
        1.0e-10f,
        -1.0e-10f,
        1.0f,
        -1.0f,
        std::numeric_limits<float>::max() / 65536.0f,
        -std::numeric_limits<float>::max() / 65536.0f,
        65504.0f,
        -65504.0f,
    };
    std::vector<uint8_t> packed(static_cast<size_t>(N) * K / 4);
    for (size_t index = 0; index != packed.size(); ++index)
        packed[index] = static_cast<uint8_t>((index * 101u + 37u) & 0xffu);
    std::vector<uint16_t> scales(static_cast<size_t>(N) * (K / block));
    std::vector<uint16_t> biases(scales.size());
    for (size_t index = 0; index != scales.size(); ++index) {
        scales[index] = fp32_to_fp16(static_cast<float>(1 + index % 3) / 1024.0f);
        biases[index] = fp32_to_fp16(
            static_cast<float>(static_cast<int>(index % 3) - 1) / 2048.0f);
    }

    std::vector<float> input(K);
    std::vector<float> expected(N);
    std::vector<float> control(N);
    std::vector<float> candidate(N);
    for (float edge : edge_values) {
        std::fill(input.begin(), input.end(), edge);
        for (int row = 0; row != N; ++row) {
            double sum = 0.0;
            for (int k = 0; k != K; ++k) {
                const size_t plane = static_cast<size_t>(row) * (K / block) + k / block;
                const uint8_t byte = packed[static_cast<size_t>(row) * (K / 4) + k / 4];
                const float q = static_cast<float>((byte >> (2 * (k & 3))) & 3u);
                sum += static_cast<double>(edge) *
                       (fp16_to_fp32(scales[plane]) * q +
                        fp16_to_fp32(biases[plane]));
            }
            expected[row] = static_cast<float>(sum);
        }
        double control_ms = 0.0;
        double candidate_ms = 0.0;
        uint64_t requested_bytes = 0;
        uint32_t width = 0;
        uint32_t maximum_threads = 0;
        CHECK(metal_test_affine_u2_block256_variant(
            input.data(), packed.data(), scales.data(), biases.data(), control.data(),
            K, N, 4, 2, false, false, 16, &control_ms, 1, 1, false,
            &requested_bytes, &width, &maximum_threads));
        CHECK(metal_test_affine_u2_block256_variant(
            input.data(), packed.data(), scales.data(), biases.data(), candidate.data(),
            K, N, 4, 1, true, false, 16, &candidate_ms, 1, 1, false,
            &requested_bytes, &width, &maximum_threads));
        CHECK(std::memcmp(control.data(), candidate.data(),
                          static_cast<size_t>(N) * sizeof(float)) == 0);
        for (int row = 0; row != N; ++row) {
            CHECK(std::isfinite(control[row]));
            CHECK(almost_equal(control[row], expected[row], 1.0e-4f, 5.0e-5f));
        }
    }
}

static void test_affine_u2_variant_matrix_real_shape() {
    constexpr int K = 5120;
    constexpr int N = 17408;
    constexpr int block = 256;
    constexpr uint32_t warmups = 8;
    constexpr uint32_t measured = 30;
    std::vector<float> input(K);
    for (int k = 0; k != K; ++k)
        input[k] = static_cast<float>((k % 31) - 15) / 64.0f;
    std::vector<uint8_t> packed(static_cast<size_t>(N) * K / 4, 0xe4);
    std::vector<uint16_t> scales(static_cast<size_t>(N) * (K / block));
    std::vector<uint16_t> biases(scales.size());
    for (size_t index = 0; index != scales.size(); ++index) {
        scales[index] = fp32_to_fp16(static_cast<float>(1 + index % 7) / 64.0f);
        biases[index] = fp32_to_fp16(
            static_cast<float>(static_cast<int>(index % 5) - 2) / 128.0f);
    }
    std::array<double, K / block> sum_x{};
    std::array<double, K / block> sum_qx{};
    for (int k = 0; k != K; ++k) {
        sum_x[k / block] += input[k];
        sum_qx[k / block] += input[k] * static_cast<double>(k & 3);
    }
    std::vector<float> expected(N);
    for (int row = 0; row != N; ++row) {
        double sum = 0.0;
        for (int ib = 0; ib != K / block; ++ib) {
            const size_t index = static_cast<size_t>(row) * (K / block) + ib;
            sum += fp16_to_fp32(scales[index]) * sum_qx[ib] +
                   fp16_to_fp32(biases[index]) * sum_x[ib];
        }
        expected[row] = static_cast<float>(sum);
    }

    double best_gb_s = 0.0;
    uint32_t best_rows = 0;
    uint32_t best_simdgroups = 0;
    bool best_masked_qdot = false;
    bool best_packed_word_load = false;
    uint32_t best_values_per_lane = 16;
    for (const uint32_t rows_per_simd : {1u, 2u, 4u, 8u}) {
        for (const uint32_t simdgroups : {1u, 2u, 4u}) {
            std::vector<float> actual(N);
            std::array<double, warmups + measured> samples{};
            uint64_t requested_bytes = 0;
            uint32_t width = 0;
            uint32_t maximum_threads = 0;
            const bool executed = metal_test_affine_u2_block256_variant(
                input.data(), packed.data(), scales.data(), biases.data(), actual.data(),
                K, N, rows_per_simd, simdgroups, false, false, 16,
                samples.data(), samples.size(),
                4, true, &requested_bytes, &width, &maximum_threads);
            CHECK(executed);
            if (!executed) continue;
            float max_delta = 0.0f;
            float max_reference = 0.0f;
            for (int row = 0; row != N; ++row) {
                CHECK(std::isfinite(actual[row]));
                max_delta = std::max(max_delta, std::fabs(actual[row] - expected[row]));
                max_reference = std::max(max_reference, std::fabs(expected[row]));
            }
            CHECK(max_delta <= 1.0e-4f + 2.0e-5f * max_reference);
            std::vector<double> times(samples.begin() + warmups, samples.end());
            std::sort(times.begin(), times.end());
            const double median = times[times.size() / 2];
            std::vector<double> deviations;
            deviations.reserve(times.size());
            for (const double sample : times)
                deviations.push_back(std::fabs(sample - median));
            std::sort(deviations.begin(), deviations.end());
            const double mad = deviations[deviations.size() / 2];
            const double p95 = times[(times.size() * 95 - 1) / 100];
            const double gb_s = static_cast<double>(requested_bytes) / 1.0e6 / median;
            if (gb_s > best_gb_s) {
                best_gb_s = gb_s;
                best_rows = rows_per_simd;
                best_simdgroups = simdgroups;
                best_masked_qdot = false;
                best_packed_word_load = false;
                best_values_per_lane = 16;
            }
            std::fprintf(stderr,
                         "Affine U2 variant K=%d N=%d rows=%u simdgroups=%u "
                         "bytes=%llu median_ms=%.6f mad_ms=%.6f p95_ms=%.6f "
                         "range=%.6f..%.6f requested_GB_s=%.3f max_delta=%.9g "
                         "width=%u max_threads=%u\n",
                         K, N, rows_per_simd, simdgroups,
                         static_cast<unsigned long long>(requested_bytes), median, mad,
                         p95, times.front(), times.back(), gb_s, max_delta, width,
                         maximum_threads);
        }
    }
    {
        constexpr uint32_t rows_per_simd = 4;
        constexpr uint32_t simdgroups = 1;
        for (const bool packed_word_load : {false, true}) {
            for (const uint32_t values_per_lane : {8u, 16u}) {
                std::vector<float> actual(N);
                std::array<double, warmups + measured> samples{};
                uint64_t requested_bytes = 0;
                uint32_t width = 0;
                uint32_t maximum_threads = 0;
                const bool executed = metal_test_affine_u2_block256_variant(
                    input.data(), packed.data(), scales.data(), biases.data(), actual.data(),
                    K, N, rows_per_simd, simdgroups, true, packed_word_load,
                    values_per_lane, samples.data(), samples.size(),
                    4, true, &requested_bytes, &width, &maximum_threads);
                CHECK(executed);
                if (!executed) continue;
                float max_delta = 0.0f;
                float max_reference = 0.0f;
                for (int row = 0; row != N; ++row) {
                    CHECK(std::isfinite(actual[row]));
                    max_delta = std::max(max_delta, std::fabs(actual[row] - expected[row]));
                    max_reference = std::max(max_reference, std::fabs(expected[row]));
                }
                CHECK(max_delta <= 1.0e-4f + 2.0e-5f * max_reference);
                std::vector<double> times(samples.begin() + warmups, samples.end());
                std::sort(times.begin(), times.end());
                const double median = times[times.size() / 2];
                const double gb_s = static_cast<double>(requested_bytes) / 1.0e6 / median;
                if (gb_s > best_gb_s) {
                    best_gb_s = gb_s;
                    best_rows = rows_per_simd;
                    best_simdgroups = simdgroups;
                    best_masked_qdot = true;
                    best_packed_word_load = packed_word_load;
                    best_values_per_lane = values_per_lane;
                }
                std::fprintf(stderr,
                             "Affine U2 masked-qdot K=%d N=%d rows=%u simdgroups=%u "
                             "word=%d values_per_lane=%u bytes=%llu median_ms=%.6f "
                             "range=%.6f..%.6f requested_GB_s=%.3f max_delta=%.9g "
                             "width=%u max_threads=%u\n",
                             K, N, rows_per_simd, simdgroups, packed_word_load,
                             values_per_lane,
                             static_cast<unsigned long long>(requested_bytes), median,
                             times.front(), times.back(), gb_s, max_delta, width,
                             maximum_threads);
            }
        }
    }
    std::fprintf(stderr,
                 "Affine U2 variant best rows=%u simdgroups=%u masked=%d word=%d values_per_lane=%u requested_GB_s=%.3f "
                 "gate_GB_s=225.000\n",
                 best_rows, best_simdgroups, best_masked_qdot,
                 best_packed_word_load, best_values_per_lane, best_gb_s);
    CHECK_MSG(best_gb_s >= 225.0,
              "affine U2 variant family missed 225 GB/s: best rows=%u simdgroups=%u masked=%d word=%d values_per_lane=%u GB/s=%.3f",
              best_rows, best_simdgroups, best_masked_qdot,
              best_packed_word_load, best_values_per_lane, best_gb_s);
}

static void test_affine_u2_masked_qdot_paired_real_shape() {
    constexpr int K = 5120;
    constexpr int N = 17408;
    constexpr int block = 256;
    constexpr uint32_t warmup_pairs = 4;
    constexpr uint32_t measured_pairs = 20;
    constexpr uint32_t dispatches_per_command = 4;
    std::vector<float> input(K);
    for (int k = 0; k != K; ++k)
        input[k] = static_cast<float>((k % 31) - 15) / 64.0f;
    const size_t packed_region = static_cast<size_t>(N) * K / 4;
    const size_t plane_region = static_cast<size_t>(N) * (K / block);
    std::vector<uint8_t> packed(packed_region * dispatches_per_command);
    std::vector<uint16_t> scales(plane_region * dispatches_per_command);
    std::vector<uint16_t> biases(plane_region * dispatches_per_command);
    std::array<std::array<uint8_t, K / block>, dispatches_per_command> block_bits{};
    std::array<std::array<double, K / block>, dispatches_per_command> sum_qx{};
    std::array<double, K / block> sum_x{};
    for (int k = 0; k != K; ++k) sum_x[k / block] += input[k];
    for (uint32_t copy = 0; copy != dispatches_per_command; ++copy) {
        for (int ib = 0; ib != K / block; ++ib) {
            const uint8_t bits = static_cast<uint8_t>(
                (0x13u + copy * 67u + static_cast<uint32_t>(ib) * 29u) & 0xffu);
            block_bits[copy][ib] = bits;
            for (int offset = 0; offset != block; ++offset) {
                const float q = static_cast<float>((bits >> (2 * (offset & 3))) & 3u);
                sum_qx[copy][ib] += input[ib * block + offset] * q;
            }
        }
        for (int row = 0; row != N; ++row) {
            for (int ib = 0; ib != K / block; ++ib) {
                std::fill_n(
                    packed.begin() + static_cast<ptrdiff_t>(
                        copy * packed_region + static_cast<size_t>(row) * (K / 4) +
                        static_cast<size_t>(ib) * (block / 4)),
                    block / 4, block_bits[copy][ib]);
            }
        }
        for (size_t index = 0; index != plane_region; ++index) {
            scales[copy * plane_region + index] = fp32_to_fp16(
                static_cast<float>(1 + (index + copy * 3) % 7) / 64.0f);
            biases[copy * plane_region + index] = fp32_to_fp16(
                static_cast<float>(static_cast<int>((index + copy * 2) % 5) - 2) /
                128.0f);
        }
    }
    std::vector<float> expected(static_cast<size_t>(N) * dispatches_per_command);
    for (uint32_t copy = 0; copy != dispatches_per_command; ++copy) {
        for (int row = 0; row != N; ++row) {
            double sum = 0.0;
            for (int ib = 0; ib != K / block; ++ib) {
                const size_t plane = copy * plane_region +
                                     static_cast<size_t>(row) * (K / block) + ib;
                sum += fp16_to_fp32(scales[plane]) * sum_qx[copy][ib] +
                       fp16_to_fp32(biases[plane]) * sum_x[ib];
            }
            expected[static_cast<size_t>(copy) * N + row] = static_cast<float>(sum);
        }
    }
    std::vector<float> control(static_cast<size_t>(N) * dispatches_per_command,
                               std::numeric_limits<float>::quiet_NaN());
    std::vector<float> candidate(control.size(),
                                 std::numeric_limits<float>::quiet_NaN());
    std::array<double, measured_pairs> control_ms{}, candidate_ms{};
    uint64_t requested_bytes = 0;
    uint32_t command_buffers = 0;
    CHECK(metal_test_affine_u2_block256_paired(
        input.data(), packed.data(), scales.data(), biases.data(),
        control.data(), candidate.data(), K, N,
        control_ms.data(), candidate_ms.data(), warmup_pairs, measured_pairs,
        dispatches_per_command, true, &requested_bytes, &command_buffers));
    CHECK(requested_bytes == static_cast<uint64_t>(N) * (64 + 2 + 2) * (K / block));
    CHECK(command_buffers == 2 * (warmup_pairs + measured_pairs));
    float max_delta = 0.0f;
    float max_reference_delta = 0.0f;
    float max_reference = 0.0f;
    for (uint32_t copy = 0; copy != dispatches_per_command; ++copy) {
        for (int row = 0; row != N; ++row) {
            const size_t index = static_cast<size_t>(copy) * N + row;
            CHECK(std::isfinite(control[index]));
            CHECK(std::isfinite(candidate[index]));
            max_delta = std::max(max_delta,
                                 std::fabs(control[index] - candidate[index]));
            max_reference_delta = std::max(
                max_reference_delta, std::fabs(control[index] - expected[index]));
            max_reference = std::max(max_reference, std::fabs(expected[index]));
        }
    }
    CHECK(max_delta == 0.0f);
    CHECK(max_reference_delta <= 1.0e-4f + 2.0e-5f * max_reference);
    uint32_t candidate_wins = 0;
    std::array<double, measured_pairs> candidate_ratios{};
    for (uint32_t pair = 0; pair != measured_pairs; ++pair) {
        candidate_wins += candidate_ms[pair] < control_ms[pair];
        candidate_ratios[pair] = candidate_ms[pair] / control_ms[pair];
    }
    std::sort(control_ms.begin(), control_ms.end());
    std::sort(candidate_ms.begin(), candidate_ms.end());
    std::sort(candidate_ratios.begin(), candidate_ratios.end());
    const double control_median = control_ms[measured_pairs / 2];
    const double candidate_median = candidate_ms[measured_pairs / 2];
    const double candidate_ratio_median = candidate_ratios[measured_pairs / 2];
    const double control_gb_s = static_cast<double>(requested_bytes) / 1.0e6 / control_median;
    const double candidate_gb_s = static_cast<double>(requested_bytes) / 1.0e6 / candidate_median;
    std::fprintf(stderr,
                 "Affine U2 paired control_ms=%.6f candidate_ms=%.6f "
                 "control_GB_s=%.3f candidate_GB_s=%.3f ratio=%.4f wins=%u/%u "
                 "control_range=%.6f..%.6f candidate_range=%.6f..%.6f "
                 "max_delta=%.9g max_reference_delta=%.9g command_buffers=%u\n",
                 control_median, candidate_median, control_gb_s, candidate_gb_s,
                 candidate_ratio_median, candidate_wins, measured_pairs,
                 control_ms.front(), control_ms.back(),
                 candidate_ms.front(), candidate_ms.back(), max_delta,
                 max_reference_delta, command_buffers);
    CHECK_MSG(candidate_ratio_median <= 0.95 && candidate_wins >= 15,
              "masked affine U2 candidate failed paired gate: control_ms=%.6f "
              "candidate_ms=%.6f ratio=%.4f wins=%u/%u",
              control_median, candidate_median, candidate_ratio_median,
              candidate_wins, measured_pairs);
}


static void pack_q3_K(const float* w, int K, int N, uint8_t* out) {
    int nb = K/256; auto blk = (block_q3_K*)out;
    unsigned rs = 333;
    for (int j = 0; j < N; j++) for (int b = 0; b < nb; b++) {
        const float* x = w + j*K + b*256; float amax = 0;
        for (int i = 0; i < 256; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax/4; auto& bb = blk[(uint64_t)j*nb+b]; bb.d = f2h(d);
        for (int s = 0; s < 12; s++) { rs = rs*1103515245+12345; bb.scales[s] = (uint8_t)(rs >> 16); }
        memset(bb.qs, 0, 64); memset(bb.hmask, 0, 32);
        float id = d > 0 ? 1/d : 0;
        for (int e = 0; e < 256; e++) {
            int q = (int)fmaxf(-4, fminf(3, roundf(x[e]*id)));
            int low2, hbit;
            if (q >= 0) { low2 = q; hbit = 1; } else { low2 = q + 4; hbit = 0; }
            int g = e/32, l = e%32;
            bb.qs[(g/4)*32 + l] |= (uint8_t)(low2 << ((g%4)*2));
            if (hbit) bb.hmask[l] |= (uint8_t)(1 << g);
        }
    }
}
static void pack_q5_K(const float* w, int K, int N, uint8_t* out) {
    int nb = K/256; auto blk = (block_q5_K*)out;
    unsigned rs = 444;
    for (int j = 0; j < N; j++) for (int b = 0; b < nb; b++) {
        const float* x = w + j*K + b*256; float amax = 0;
        for (int i = 0; i < 256; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax/31; auto& bb = blk[(uint64_t)j*nb+b]; bb.d = f2h(d); bb.dmin = f2h(amax/480);
        for (int s = 0; s < 12; s++) { rs = rs*1103515245+12345; bb.scales[s] = (uint8_t)(rs >> 16); }
        memset(bb.qs, 0, 128); memset(bb.qh, 0, 32);
        float id = d > 0 ? 1/d : 0;
        for (int e = 0; e < 256; e++) {
            int q = (int)fmaxf(0, fminf(31, roundf(x[e]*id)));
            int lo = q & 0xF, hi = (q >> 4) & 1;
            int jb = e/64, within = e%64;
            if (within < 32) {
                bb.qs[jb*32 + within] |= (uint8_t)lo;
                if (hi) bb.qh[within] |= (uint8_t)(1 << (2*jb));
            } else {
                int l = within - 32;
                bb.qs[jb*32 + l] |= (uint8_t)(lo << 4);
                if (hi) bb.qh[l] |= (uint8_t)(1 << (2*jb+1));
            }
        }
    }
}

static void reference_q4_scale_min(int index, const uint8_t scales[12], uint8_t& scale, uint8_t& minimum) {
    if (index < 4) {
        scale = scales[index] & 63;
        minimum = scales[index + 4] & 63;
    } else {
        scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
        minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
    }
}

// Independent FP32 GGML-K decoder. It intentionally shares no production
// matmul or Metal helper with the descriptor under test.
static void reference_quantized_matmul(const std::vector<float>& x, const Tensor& weight,
                                       std::vector<float>& output, int K, int N) {
    output.assign(N, 0.0f);
    if (weight.type == GGMLType::Q4_K) {
        const auto* rows = reinterpret_cast<const block_q4_K*>(weight.data);
        for (int row = 0; row < N; ++row) {
            float sum = 0.0f;
            for (int block = 0; block < K / 256; ++block) {
                const block_q4_K& source = rows[static_cast<size_t>(row) * (K / 256) + block];
                const float d = fp16_to_fp32(source.d);
                const float dmin = fp16_to_fp32(source.dmin);
                for (int chunk = 0; chunk != 4; ++chunk) {
                    uint8_t scale0 = 0, min0 = 0, scale1 = 0, min1 = 0;
                    reference_q4_scale_min(2 * chunk, source.scales, scale0, min0);
                    reference_q4_scale_min(2 * chunk + 1, source.scales, scale1, min1);
                    const uint8_t* quant = source.qs + chunk * 32;
                    const int base = block * 256 + chunk * 64;
                    for (int lane = 0; lane != 32; ++lane) {
                        sum += x[base + lane] * (d * scale0 * static_cast<float>(quant[lane] & 0x0f) - dmin * min0);
                        sum += x[base + 32 + lane] * (d * scale1 * static_cast<float>(quant[lane] >> 4) - dmin * min1);
                    }
                }
            }
            output[row] = sum;
        }
        return;
    }
    if (weight.type == GGMLType::Q5_0) {
        const auto* rows = reinterpret_cast<const block_q5_0*>(weight.data);
        for (int row = 0; row != N; ++row) {
            float sum = 0.0f;
            for (int block = 0; block != K / 32; ++block) {
                const block_q5_0& source = rows[static_cast<size_t>(row) * (K / 32) + block];
                const float d = fp16_to_fp32(source.d);
                for (int lane = 0; lane != 32; ++lane) {
                    const uint8_t packed = source.qs[lane % 16];
                    const int low = lane < 16 ? (packed & 0x0f) : (packed >> 4);
                    const int quant = low | (((source.qh >> lane) & 1u) << 4);
                    sum += x[block * 32 + lane] * d * static_cast<float>(quant - 16);
                }
            }
            output[row] = sum;
        }
        return;
    }
    if (weight.type == GGMLType::Q8_0) {
        const auto* rows = reinterpret_cast<const block_q8_0*>(weight.data);
        for (int row = 0; row != N; ++row) {
            float sum = 0.0f;
            for (int block = 0; block != K / 32; ++block) {
                const block_q8_0& source = rows[static_cast<size_t>(row) * (K / 32) + block];
                const float d = fp16_to_fp32(source.d);
                for (int lane = 0; lane != 32; ++lane)
                    sum += x[block * 32 + lane] * d * static_cast<float>(source.qs[lane]);
            }
            output[row] = sum;
        }
        return;
    }
    const auto* rows = reinterpret_cast<const block_q6_K*>(weight.data);
    for (int row = 0; row < N; ++row) {
        float sum = 0.0f;
        for (int block = 0; block < K / 256; ++block) {
            const block_q6_K& source = rows[static_cast<size_t>(row) * (K / 256) + block];
            const float d = fp16_to_fp32(source.d);
            for (int half = 0; half != 2; ++half) {
                const uint8_t* ql = source.ql + half * 64;
                const uint8_t* qh = source.qh + half * 32;
                const int8_t* scales = source.scales + half * 8;
                const int base = block * 256 + half * 128;
                for (int lane = 0; lane != 32; ++lane) {
                    const int scale_group = lane / 16;
                    const int q0 = static_cast<int>((ql[lane] & 0x0f) | (((qh[lane] >> 0) & 3) << 4)) - 32;
                    const int q1 = static_cast<int>((ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 3) << 4)) - 32;
                    const int q2 = static_cast<int>((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
                    const int q3 = static_cast<int>((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32;
                    sum += x[base + lane] * d * scales[scale_group] * q0;
                    sum += x[base + 32 + lane] * d * scales[scale_group + 2] * q1;
                    sum += x[base + 64 + lane] * d * scales[scale_group + 4] * q2;
                    sum += x[base + 96 + lane] * d * scales[scale_group + 6] * q3;
                }
            }
        }
        output[row] = sum;
    }
}

static void test_quantized_repeat_matches_independent_decoder() {
    constexpr int K = 256;
    constexpr int N = 7;
    std::vector<float> input(K), logical(static_cast<size_t>(K) * N);
    rng_fill(input.data(), K, 806);
    rng_fill(logical.data(), static_cast<int>(logical.size()), 807);
    for (float& value : logical) value *= 0.01f;

    struct Case {
        GGMLType type;
        void (*pack)(const float*, int, int, uint8_t*);
    };
    const Case cases[] = {
        {GGMLType::Q4_K, pack_q4_K},
        {GGMLType::Q6_K, pack_q6_K},
    };
    for (const Case& test : cases) {
        std::vector<uint8_t> bytes(static_cast<size_t>(N) * bytes_per_block(test.type));
        test.pack(logical.data(), K, N, bytes.data());
        Tensor weight;
        weight.type = test.type;
        weight.n_dims = 2;
        weight.dims[0] = K;
        weight.dims[1] = N;
        weight.data = bytes.data();
        CHECK(weight.nbytes() == bytes.size());

        std::vector<float> expected, actual(N);
        reference_quantized_matmul(input, weight, expected, K, N);
        CHECK(metal_gemv_repeat(input.data(), weight, actual.data(), K, N, 3));
        for (int row = 0; row != N; ++row)
            CHECK(almost_equal(actual[row], expected[row], 1.0e-2f, 1.0e-3f));
    }
}

static void test_quantized_m1_variant_sweep() {
    constexpr int K = 5120;
    constexpr int N = 17408;
    constexpr uint32_t warmups = 1;
    constexpr uint32_t measured_pairs = 5;
    struct Launch {
        uint32_t rows_per_simd;
        uint32_t simdgroups;
    };
    constexpr Launch baseline{2, 2};
    constexpr Launch candidates[] = {
        {1, 1}, {1, 2}, {2, 1}, {2, 4}, {4, 1}, {4, 2}, {4, 4},
    };

    std::vector<float> input(K);
    rng_fill(input.data(), K, 1801u);
    for (float& value : input) value *= 0.01f;

    auto make_weight = [&](GGMLType type) {
        const size_t bytes = static_cast<size_t>(N) * (K / 256) *
                             bytes_per_block(type);
        std::vector<uint8_t> packed(bytes, 0);
        const size_t block_count = static_cast<size_t>(N) * (K / 256);
        if (type == GGMLType::Q4_K) {
            auto* blocks = reinterpret_cast<block_q4_K*>(packed.data());
            for (size_t block = 0; block != block_count; ++block) {
                blocks[block].d = f2h(0.01f);
                blocks[block].dmin = f2h(0.001f);
                for (size_t i = 0; i != sizeof(blocks[block].scales); ++i)
                    blocks[block].scales[i] =
                        static_cast<uint8_t>((block * 13u + i * 7u + 3u) & 63u);
                for (size_t i = 0; i != sizeof(blocks[block].qs); ++i)
                    blocks[block].qs[i] =
                        static_cast<uint8_t>(block * 29u + i * 11u + 5u);
            }
        } else {
            auto* blocks = reinterpret_cast<block_q6_K*>(packed.data());
            for (size_t block = 0; block != block_count; ++block) {
                blocks[block].d = f2h(0.005f);
                for (size_t i = 0; i != sizeof(blocks[block].scales); ++i)
                    blocks[block].scales[i] = static_cast<int8_t>(
                        static_cast<int>((block * 5u + i * 3u) % 31u) - 15);
                for (size_t i = 0; i != sizeof(blocks[block].ql); ++i)
                    blocks[block].ql[i] =
                        static_cast<uint8_t>(block * 17u + i * 7u + 1u);
                for (size_t i = 0; i != sizeof(blocks[block].qh); ++i)
                    blocks[block].qh[i] =
                        static_cast<uint8_t>(block * 19u + i * 5u + 9u);
            }
        }
        Tensor weight;
        weight.type = type;
        weight.n_dims = 2;
        weight.dims[0] = K;
        weight.dims[1] = N;
        weight.data = packed.data();
        weight.data_bytes = packed.size();
        return std::pair<Tensor, std::vector<uint8_t>>(weight, std::move(packed));
    };

    auto median = [](const std::array<double, measured_pairs>& values) {
        std::array<double, measured_pairs> ordered = values;
        std::sort(ordered.begin(), ordered.end());
        return ordered[measured_pairs / 2u];
    };
    auto run = [&](const Tensor& weight, const Launch launch, uint32_t pair_index,
                   uint32_t warmup_count, double* elapsed, std::vector<float>& output) {
        if (launch.rows_per_simd == baseline.rows_per_simd &&
            launch.simdgroups == baseline.simdgroups) {
            metal_dispatch_metrics_reset();
            for (uint32_t warmup = 0; warmup != warmup_count; ++warmup) {
                if (!metal_gemv_repeat(input.data(), weight, output.data(), K, N, 1))
                    return false;
            }
            metal_dispatch_metrics_reset();
            const auto start = std::chrono::steady_clock::now();
            if (!metal_gemv_repeat(input.data(), weight, output.data(), K, N, 1))
                return false;
            const auto finish = std::chrono::steady_clock::now();
            const MetalDispatchMetrics metrics = metal_dispatch_metrics();
            const double wall_ms = std::chrono::duration<double, std::milli>(
                finish - start).count();
            *elapsed = metrics.gpu_time_ms > 0.0 ? metrics.gpu_time_ms : wall_ms;
            if (pair_index == UINT32_MAX) {
                std::fprintf(stderr,
                             "quantized-m1 production warm type=%s rows=2 groups=2 gpu_ms=%.6f wall_ms=%.6f\n",
                             type_name(weight.type), metrics.gpu_time_ms, wall_ms);
            }
            return metrics.command_buffers == 1u;
        }
        uint64_t requested_bytes = 0;
        uint32_t command_buffers = 0;
        uint32_t width = 0;
        uint32_t maximum_threads = 0;
        CHECK(metal_test_quantized_gemv_variant(
            weight, input.data(), output.data(), launch.rows_per_simd,
            launch.simdgroups, 1u, false, false, warmup_count, 1u, elapsed, &requested_bytes,
            &command_buffers, &width, &maximum_threads));
        CHECK(requested_bytes == weight.data_bytes);
        CHECK(command_buffers == warmup_count + 1u);
        CHECK(width == 32u);
        CHECK(maximum_threads >= 32u * launch.simdgroups);
        if (pair_index == UINT32_MAX) {
            std::fprintf(stderr,
                         "quantized-m1 warm type=%s rows=%u groups=%u gpu_ms=%.6f\n",
                         type_name(weight.type), launch.rows_per_simd,
                         launch.simdgroups, *elapsed);
        }
        return true;
    };

    const GGMLType types[] = {GGMLType::Q4_K, GGMLType::Q6_K};
    for (GGMLType type : types) {
        auto fixture = make_weight(type);
        const Tensor& weight = fixture.first;
        std::vector<float> expected;
        reference_quantized_matmul(input, weight, expected, K, N);
        for (const Launch candidate : candidates) {
            std::array<double, measured_pairs> baseline_ms{};
            std::array<double, measured_pairs> candidate_ms{};
            std::vector<float> baseline_output(N);
            std::vector<float> candidate_output(N);
            double ignored_ms = 0.0;
            CHECK(run(weight, baseline, UINT32_MAX, warmups, &ignored_ms, baseline_output));
            CHECK(run(weight, candidate, UINT32_MAX, warmups, &ignored_ms, candidate_output));
            for (int row = 0; row != N; ++row) {
                CHECK_MSG(std::isfinite(baseline_output[row]) &&
                              almost_equal(baseline_output[row], expected[row],
                                           1.0e-2f, 1.0e-3f),
                          "quantized m1 baseline type=%s row=%d actual=%g expected=%g",
                          type_name(type), row, baseline_output[row], expected[row]);
                CHECK_MSG(std::isfinite(candidate_output[row]) &&
                              almost_equal(candidate_output[row], expected[row],
                                           1.0e-2f, 1.0e-3f),
                          "quantized m1 candidate type=%s rows=%u groups=%u row=%d actual=%g expected=%g",
                          type_name(type), candidate.rows_per_simd, candidate.simdgroups,
                          row, candidate_output[row], expected[row]);
            }
            for (uint32_t pair = 0; pair != measured_pairs; ++pair) {
                const bool candidate_first = (pair & 1u) != 0;
                if (candidate_first) {
                    CHECK(run(weight, candidate, pair, 0u, &candidate_ms[pair], candidate_output));
                    CHECK(run(weight, baseline, pair, 0u, &baseline_ms[pair], baseline_output));
                } else {
                    CHECK(run(weight, baseline, pair, 0u, &baseline_ms[pair], baseline_output));
                    CHECK(run(weight, candidate, pair, 0u, &candidate_ms[pair], candidate_output));
                }
                for (int row = 0; row != N; ++row) {
                    CHECK(almost_equal(baseline_output[row], expected[row],
                                       1.0e-2f, 1.0e-3f));
                    CHECK(almost_equal(candidate_output[row], expected[row],
                                       1.0e-2f, 1.0e-3f));
                }
            }
            const double baseline_median = median(baseline_ms);
            const double candidate_median = median(candidate_ms);
            uint32_t wins = 0;
            for (uint32_t pair = 0; pair != measured_pairs; ++pair)
                wins += candidate_ms[pair] < baseline_ms[pair] ? 1u : 0u;
            std::fprintf(stderr,
                         "quantized-m1 sweep type=%s K=%d N=%d baseline=2x2 "
                         "candidate=%ux%u wins=%u/%u baseline_gpu_ms=%.6f "
                         "candidate_gpu_ms=%.6f result=%s\n",
                         type_name(type), K, N, candidate.rows_per_simd,
                         candidate.simdgroups, wins, measured_pairs,
                         baseline_median, candidate_median,
                         wins >= 4u && candidate_median < baseline_median
                             ? "repeatable-win" : "rejected");
        }
    }
}

static void test_quantized_dispatch_overhead() {
    constexpr int K = 256;
    constexpr int N = 32;
    constexpr uint32_t warmups = 5;
    constexpr uint32_t samples = 5;
    constexpr uint32_t dispatch_counts[] = {1u, 497u};
    std::vector<float> input(K), logical(static_cast<size_t>(K) * N);
    rng_fill(input.data(), K, 8041u);
    rng_fill(logical.data(), static_cast<int>(logical.size()), 8042u);
    for (float& value : input) value *= 0.01f;
    for (float& value : logical) value *= 0.01f;

    for (const GGMLType type : {GGMLType::Q4_K, GGMLType::Q6_K}) {
        const size_t weight_bytes = static_cast<size_t>(N) * bytes_per_block(type);
        std::vector<uint8_t> packed(weight_bytes);
        if (type == GGMLType::Q4_K) pack_q4_K(logical.data(), K, N, packed.data());
        else pack_q6_K(logical.data(), K, N, packed.data());
        Tensor weight;
        weight.type = type;
        weight.n_dims = 2;
        weight.dims[0] = K;
        weight.dims[1] = N;
        weight.data = packed.data();
        weight.data_bytes = packed.size();
        std::vector<float> expected;
        reference_quantized_matmul(input, weight, expected, K, N);

        std::array<double, samples> one_ms{}, many_ms{};
        uint64_t requested_bytes = 0;
        uint32_t command_buffers = 0, width = 0, maximum_threads = 0;
        std::vector<float> one_output(N), many_output(N);
        CHECK(metal_test_quantized_gemv_variant(
            weight, input.data(), one_output.data(), 2u, 2u, dispatch_counts[0],
            false, false, warmups, samples, one_ms.data(), &requested_bytes, &command_buffers,
            &width, &maximum_threads));
        CHECK(command_buffers == warmups + samples);
        CHECK(metal_test_quantized_gemv_variant(
            weight, input.data(), many_output.data(), 2u, 2u, dispatch_counts[1],
            false, false, warmups, samples, many_ms.data(), &requested_bytes, &command_buffers,
            &width, &maximum_threads));
        CHECK(command_buffers == warmups + samples);
        for (int row = 0; row != N; ++row) {
            CHECK(almost_equal(one_output[row], expected[row], 1.0e-2f, 1.0e-3f));
            CHECK(almost_equal(many_output[row], expected[row], 1.0e-2f, 1.0e-3f));
        }
        auto median = [](std::array<double, samples> values) {
            std::sort(values.begin(), values.end());
            return values[values.size() / 2u];
        };
        const double one_median = median(one_ms);
        const double many_median = median(many_ms);
        CHECK(one_median > 0.0 && many_median > 0.0);
        std::fprintf(stderr,
                     "quantized dispatch overhead type=%s K=%d N=%d one_dispatch_gpu_ms=%.6f "
                     "many_dispatch_gpu_ms=%.6f dispatches_per_command=%u "
                     "incremental_gpu_us=%.6f requested_bytes=%llu command_buffers=%u "
                     "one_cb_per_sample=1 exact=1 width=%u max_threads=%u\n",
                     type_name(type), K, N, one_median, many_median,
                     dispatch_counts[1], 1000.0 * (many_median - one_median) /
                         static_cast<double>(dispatch_counts[1] - dispatch_counts[0]),
                     static_cast<unsigned long long>(requested_bytes), command_buffers,
                     width, maximum_threads);
    }
}

static void test_quantized_q6k_llama_variant() {
    constexpr int K = 5120;
    constexpr int N = 17408;
    constexpr uint32_t warmups = 10;
    constexpr uint32_t measured_pairs = 20;
    std::vector<float> input(K);
    rng_fill(input.data(), K, 1811u);
    for (float& value : input) value *= 0.01f;

    CHECK(K % 256 == 0);
    const size_t blocks_per_row = static_cast<size_t>(K / 256);
    CHECK(static_cast<size_t>(N) <= std::numeric_limits<size_t>::max() / blocks_per_row);
    const size_t block_count = static_cast<size_t>(N) * blocks_per_row;
    CHECK(block_count <= std::numeric_limits<size_t>::max() /
          bytes_per_block(GGMLType::Q6_K));
    const size_t weight_bytes = block_count * bytes_per_block(GGMLType::Q6_K);
    std::vector<uint8_t> packed(weight_bytes, 0);
    auto* blocks = reinterpret_cast<block_q6_K*>(packed.data());
    for (size_t block = 0; block != block_count; ++block) {
        blocks[block].d = f2h(0.005f);
        for (size_t i = 0; i != sizeof(blocks[block].scales); ++i)
            blocks[block].scales[i] = static_cast<int8_t>(
                static_cast<int>((block * 5u + i * 3u) % 31u) - 15);
        for (size_t i = 0; i != sizeof(blocks[block].ql); ++i)
            blocks[block].ql[i] = static_cast<uint8_t>(block * 17u + i * 7u + 1u);
        for (size_t i = 0; i != sizeof(blocks[block].qh); ++i)
            blocks[block].qh[i] = static_cast<uint8_t>(block * 19u + i * 5u + 9u);
    }
    Tensor weight;
    weight.type = GGMLType::Q6_K;
    weight.n_dims = 2;
    weight.dims[0] = K;
    weight.dims[1] = N;
    weight.data = packed.data();
    weight.data_bytes = packed.size();
    std::vector<float> expected;
    reference_quantized_matmul(input, weight, expected, K, N);

    auto check_output = [&](const std::vector<float>& actual) {
        CHECK(actual.size() == expected.size());
        for (int row = 0; row != N; ++row)
            CHECK(std::isfinite(actual[row]) &&
                  almost_equal(actual[row], expected[row], 1.0e-2f, 1.0e-3f));
    };

    std::array<double, measured_pairs> reference_gpu_ms{}, candidate_gpu_ms{};
    std::array<double, measured_pairs> reference_wall_ms{}, candidate_wall_ms{};
    std::vector<float> candidate_output(N), baseline_output(N);
    uint64_t requested_bytes = 0;
    uint32_t command_buffers = 0, width = 0, maximum_threads = 0;
    CHECK(metal_test_q6k_alias_paired(
        weight, input.data(), baseline_output.data(), candidate_output.data(),
        warmups, measured_pairs, reference_gpu_ms.data(), candidate_gpu_ms.data(),
        reference_wall_ms.data(), candidate_wall_ms.data(), &requested_bytes,
        &command_buffers, &width, &maximum_threads));
    CHECK(requested_bytes == weight.data_bytes);
    CHECK(command_buffers == 2u * warmups + 4u * measured_pairs);
    CHECK(width == 32u);
    CHECK(maximum_threads >= 64u);
    check_output(baseline_output);
    check_output(candidate_output);
    auto median = [](std::array<double, measured_pairs> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2u];
    };
    auto range = [](const std::array<double, measured_pairs>& values) {
        const auto bounds = std::minmax_element(values.begin(), values.end());
        return std::pair<double, double>(*bounds.first, *bounds.second);
    };
    const double reference_gpu_median = median(reference_gpu_ms);
    const double candidate_gpu_median = median(candidate_gpu_ms);
    const double reference_wall_median = median(reference_wall_ms);
    const double candidate_wall_median = median(candidate_wall_ms);
    const auto reference_gpu_range = range(reference_gpu_ms);
    const auto candidate_gpu_range = range(candidate_gpu_ms);
    const auto reference_wall_range = range(reference_wall_ms);
    const auto candidate_wall_range = range(candidate_wall_ms);
    uint32_t wins = 0;
    for (uint32_t sample = 0; sample != measured_pairs; ++sample)
        wins += candidate_gpu_ms[sample] < reference_gpu_ms[sample] ? 1u : 0u;
    std::fprintf(stderr,
                 "quantized-q6k alias paired K=%d N=%d reference=gemv_q6k "
                 "candidate=gemv_q6k_llama_variant warmups=%u samples=%u "
                 "order=ABBA/BAAB wins=%u/%u reference_gpu_ms=%.6f "
                 "candidate_gpu_ms=%.6f reference_gpu_range_ms=%.6f..%.6f "
                 "candidate_gpu_range_ms=%.6f..%.6f reference_wall_ms=%.6f "
                 "candidate_wall_ms=%.6f reference_wall_range_ms=%.6f..%.6f "
                 "candidate_wall_range_ms=%.6f..%.6f command_buffers=%u "
                 "requested_bytes=%llu exact=1 result=%s\n",
                 K, N, warmups, measured_pairs, wins, measured_pairs,
                 reference_gpu_median, candidate_gpu_median,
                 reference_gpu_range.first, reference_gpu_range.second,
                 candidate_gpu_range.first, candidate_gpu_range.second,
                 reference_wall_median, candidate_wall_median,
                 reference_wall_range.first, reference_wall_range.second,
                 candidate_wall_range.first, candidate_wall_range.second,
                 command_buffers, static_cast<unsigned long long>(requested_bytes),
                 wins >= 16u && candidate_gpu_median < reference_gpu_median
                     ? "repeatable-win" : "rejected");
    return;

#if 0
    std::array<double, measured_pairs> baseline_ms{}, candidate_ms{};
    auto run_baseline = [&](double* elapsed) {
        uint64_t requested_bytes = 0;
        uint32_t command_buffers = 0, width = 0, maximum_threads = 0;
        CHECK(metal_test_quantized_gemv_variant(
            weight, input.data(), baseline_output.data(), 2u, 2u,
            dispatches_per_command, false, true,
            0u, 1u, elapsed, &requested_bytes, &command_buffers, &width,
            &maximum_threads));
        CHECK(requested_bytes == weight.data_bytes);
        CHECK(command_buffers == 1u);
        CHECK(width == 32u);
        CHECK(maximum_threads >= 64u);
        return std::isfinite(*elapsed) && *elapsed > 0.0;
    };
    auto run_candidate = [&](double* elapsed) {
        uint64_t requested_bytes = 0;
        uint32_t command_buffers = 0, width = 0, maximum_threads = 0;
        CHECK(metal_test_quantized_gemv_variant(
            weight, input.data(), candidate_output.data(), 2u, 2u,
            dispatches_per_command, true, false,
            0u, 1u, elapsed, &requested_bytes, &command_buffers, &width,
            &maximum_threads));
        CHECK(requested_bytes == weight.data_bytes);
        CHECK(command_buffers == 1u);
        CHECK(width == 32u);
        CHECK(maximum_threads >= 64u);
        return std::isfinite(*elapsed) && *elapsed > 0.0;
    };

    double ignored = 0.0;
    uint64_t requested_bytes = 0;
    uint32_t command_buffers = 0, width = 0, maximum_threads = 0;
    CHECK(metal_test_quantized_gemv_variant(
        weight, input.data(), candidate_output.data(), 2u, 2u,
        dispatches_per_command, true, false,
        warmups, 1u, &ignored, &requested_bytes, &command_buffers, &width,
        &maximum_threads));
    CHECK(requested_bytes == weight.data_bytes);
    CHECK(command_buffers == warmups + 1u);
    CHECK(width == 32u);
    CHECK(maximum_threads >= 64u);
    check_output(candidate_output);
    CHECK(run_baseline(&ignored));
    check_output(baseline_output);
    for (uint32_t pair = 0; pair != measured_pairs; ++pair) {
        const bool candidate_first = (pair & 1u) != 0;
        if (candidate_first) {
            CHECK(run_candidate(&candidate_ms[pair]));
            CHECK(run_baseline(&baseline_ms[pair]));
        } else {
            CHECK(run_baseline(&baseline_ms[pair]));
            CHECK(run_candidate(&candidate_ms[pair]));
        }
        check_output(candidate_output);
        check_output(baseline_output);
    }
    auto median = [](std::array<double, measured_pairs> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2u];
    };
    const double baseline_median = median(baseline_ms);
    const double candidate_median = median(candidate_ms);
    uint32_t wins = 0;
    for (uint32_t pair = 0; pair != measured_pairs; ++pair)
        wins += candidate_ms[pair] < baseline_ms[pair] ? 1u : 0u;
    std::fprintf(stderr,
                 "quantized-q6k-upstream sweep type=Q6_K K=%d N=%d baseline=2x2 "
                 "candidate=upstream-decomposition wins=%u/%u baseline_gpu_ms=%.6f "
                 "candidate_gpu_ms=%.6f result=%s\n",
                 K, N, wins, measured_pairs, baseline_median, candidate_median,
                 wins >= 4u && candidate_median < baseline_median
                     ? "repeatable-win" : "rejected");
#endif
}

static void test_rmsnorm_q8_0_weight() {
    constexpr int n = 512;
    constexpr float eps = 1.0e-5f;
    const size_t block_count = static_cast<size_t>(n / 32);
    const size_t weight_bytes = block_count * sizeof(block_q8_0);
    std::vector<block_q8_0> packed(block_count);
    for (size_t block = 0; block != block_count; ++block) {
        packed[block].d = f2h(0.003f + 0.0001f * static_cast<float>(block % 7));
        for (int lane = 0; lane != 32; ++lane)
            packed[block].qs[lane] = static_cast<int8_t>(
                static_cast<int>((block * 13u + static_cast<size_t>(lane) * 7u) % 255u) - 127);
    }
    const std::vector<uint8_t> packed_before(
        reinterpret_cast<const uint8_t*>(packed.data()),
        reinterpret_cast<const uint8_t*>(packed.data()) + weight_bytes);
    Tensor weight;
    weight.type = GGMLType::Q8_0;
    weight.n_dims = 1;
    weight.dims[0] = n;
    weight.data = reinterpret_cast<const uint8_t*>(packed.data());
    weight.data_bytes = weight_bytes;
    std::vector<float> input(n), expected(n), actual(n, 0.0f);
    rng_fill(input.data(), n, 9341u);
    for (float& value : input) value *= 0.02f;
    float sum = 0.0f;
    for (float value : input) sum += value * value;
    const float inv = 1.0f / std::sqrt(sum / static_cast<float>(n) + eps);
    for (int i = 0; i != n; ++i) {
        const block_q8_0& block = packed[static_cast<size_t>(i / 32)];
        expected[i] = input[i] * inv * fp16_to_fp32(block.d) *
                      static_cast<float>(block.qs[i % 32]);
    }
    double gpu_ms = 0.0;
    uint64_t requested_bytes = 0;
    uint32_t command_buffers = 0, width = 0, maximum_threads = 0;
    CHECK(metal_test_rmsnorm_q8_0(weight, input.data(), actual.data(), n, eps,
                                  &gpu_ms, &requested_bytes, &command_buffers,
                                  &width, &maximum_threads));
    CHECK(requested_bytes == weight_bytes);
    CHECK(command_buffers == 1u);
    CHECK(width == 32u);
    CHECK(maximum_threads >= 256u);
    CHECK(std::isfinite(gpu_ms) && gpu_ms > 0.0);
    const double valid_gpu_ms = gpu_ms;
    CHECK(std::memcmp(packed.data(), packed_before.data(), weight_bytes) == 0);
    for (int i = 0; i != n; ++i)
        CHECK(almost_equal(actual[i], expected[i], 2.0e-5f, 2.0e-5f));

    Tensor short_span = weight;
    short_span.data_bytes = weight_bytes - 1;
    CHECK(!metal_test_rmsnorm_q8_0(short_span, input.data(), actual.data(), n, eps,
                                   &gpu_ms, &requested_bytes, &command_buffers,
                                   &width, &maximum_threads));
    Tensor wrong_shape = weight;
    wrong_shape.dims[0] = n - 1;
    CHECK(!metal_test_rmsnorm_q8_0(wrong_shape, input.data(), actual.data(), n, eps,
                                   &gpu_ms, &requested_bytes, &command_buffers,
                                   &width, &maximum_threads));
    Tensor wrong_type = weight;
    wrong_type.type = GGMLType::F16;
    CHECK(!metal_test_rmsnorm_q8_0(wrong_type, input.data(), actual.data(), n, eps,
                                   &gpu_ms, &requested_bytes, &command_buffers,
                                   &width, &maximum_threads));
    CHECK(!metal_test_rmsnorm_q8_0(weight, input.data(), actual.data(), n + 1, eps,
                                   &gpu_ms, &requested_bytes, &command_buffers,
                                   &width, &maximum_threads));
    CHECK(!metal_test_rmsnorm_q8_0(weight, input.data(), actual.data(), n, -eps,
                                   &gpu_ms, &requested_bytes, &command_buffers,
                                   &width, &maximum_threads));
    CHECK(!metal_test_rmsnorm_q8_0(weight, input.data(), actual.data(), n,
                                   std::numeric_limits<float>::quiet_NaN(),
                                   &gpu_ms, &requested_bytes, &command_buffers,
                                   &width, &maximum_threads));
    std::fprintf(stderr,
                 "rmsnorm-q8_0 weight: N=%d bytes=%zu gpu_ms=%.6f one_cb=1 "
                 "parity=1 immutable=1 malformed_span=1 malformed_shape=1\n",
                 n, weight_bytes, valid_gpu_ms);
}

static void test_moe_q4k_gate_up_gelu() {
    constexpr int K = 512;
    constexpr int I = 256;
    constexpr int E = 4;
    constexpr int N = 2 * I;
    constexpr uint32_t R = 3;
    const size_t expert_stride = static_cast<size_t>(N) * (K / 256) * sizeof(block_q4_K);
    std::vector<float> input(K), logical(static_cast<size_t>(E) * N * K);
    rng_fill(input.data(), K, 7101);
    for (float& value : input) value *= 0.03f;
    for (int expert = 0; expert < E; ++expert) {
        for (int row = 0; row < N; ++row) {
            for (int k = 0; k < K; ++k) {
                logical[(static_cast<size_t>(expert) * N + row) * K + k] =
                    static_cast<float>(((expert + 1) * 17 + row * 7 + k * 3) % 101 - 50) / 37.0f;
            }
        }
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(E) * expert_stride);
    pack_q4_K(logical.data(), K, E * N, bytes.data());
    Tensor gate_up;
    gate_up.type = GGMLType::Q4_K;
    gate_up.n_dims = 3;
    gate_up.dims[0] = K;
    gate_up.dims[1] = N;
    gate_up.dims[2] = E;
    gate_up.data = bytes.data();
    const std::array<uint32_t, R> ids = {3, 1, 3};
    std::vector<float> expected(static_cast<size_t>(R) * I), actual(expected.size());
    for (uint32_t slot = 0; slot < R; ++slot) {
        Tensor expert_weight = gate_up;
        expert_weight.n_dims = 2;
        expert_weight.dims[2] = 0;
        expert_weight.data = bytes.data() + static_cast<size_t>(ids[slot]) * expert_stride;
        std::vector<float> gate_up_values;
        reference_quantized_matmul(input, expert_weight, gate_up_values, K, N);
        for (int i = 0; i < I; ++i) {
            const float gate = gate_up_values[i];
            const float cube = gate * gate * gate;
            const float activated = 0.5f * gate *
                (1.0f + std::tanh(0.7978845608028654f *
                                  (gate + 0.044715f * cube)));
            expected[static_cast<size_t>(slot) * I + i] = activated * gate_up_values[I + i];
        }
    }

    uint32_t pipeline_width = 0, maximum_threads = 0;
    double gpu_ms = 0.0;
    metal_dispatch_metrics_reset();
    CHECK(metal_test_moe_q4k_gate_up_gelu(input.data(), gate_up, bytes.size(), ids.data(), R,
                                          actual.data(), K, I, &gpu_ms,
                                          &pipeline_width, &maximum_threads));
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    CHECK(pipeline_width != 0);
    CHECK(maximum_threads >= pipeline_width);
    CHECK(gpu_ms > 0.0);
    for (size_t i = 0; i != actual.size(); ++i)
        CHECK(almost_equal(actual[i], expected[i], 2.0e-4f, 2.0e-4f));

    auto rejects_without_submission = [&](const Tensor& malformed,
                                          const uint32_t* malformed_ids,
                                          uint32_t count) {
        metal_dispatch_metrics_reset();
        CHECK(!metal_test_moe_q4k_gate_up_gelu(input.data(), malformed, bytes.size(),
                                               malformed_ids, count,
                                               actual.data(), K, I, &gpu_ms,
                                               &pipeline_width, &maximum_threads));
        CHECK(metal_dispatch_metrics().command_buffers == 0);
    };
    Tensor wrong_input = gate_up;
    wrong_input.dims[0] = K - 256;
    rejects_without_submission(wrong_input, ids.data(), R);
    Tensor wrong_output = gate_up;
    wrong_output.dims[1] = I;
    rejects_without_submission(wrong_output, ids.data(), R);
    Tensor wrong_expert_count = gate_up;
    wrong_expert_count.dims[2] = E + 1;
    rejects_without_submission(wrong_expert_count, ids.data(), R);
    const std::array<uint32_t, R> out_of_range = {3, E, 3};
    rejects_without_submission(gate_up, out_of_range.data(), R);
    rejects_without_submission(gate_up, ids.data(), 0);
    std::fprintf(stderr, "PASS generic MoE Q4_K gate/up GELU-tanh: experts=%d selected=%u one_cb=1\n",
                 E, R);
}

static float moe_activation_oracle(float gate, float up, bool swiglu) {
    if (swiglu) return (gate / (1.0f + std::exp(-gate))) * up;
    return 0.5f * gate *
           (1.0f + std::tanh(0.7978845608028654f *
                             (gate + 0.044715f * gate * gate * gate))) * up;
}

static void test_moe_batched_activation() {
    struct Case {
        uint32_t selected;
        uint32_t intermediate;
        bool swiglu;
    };
    constexpr std::array<Case, 4> cases = {{
        {1, 17, false},
        {2, 64, true},
        {8, 257, false},
        {16, 2816, true},
    }};

    for (const Case& test : cases) {
        const size_t gate_up_values = static_cast<size_t>(test.selected) *
                                      2u * test.intermediate;
        const size_t output_values = static_cast<size_t>(test.selected) *
                                     test.intermediate;
        std::vector<float> gate_up(gate_up_values), expected(output_values), actual(output_values);
        std::vector<uint32_t> ids(test.selected);
        for (uint32_t slot = 0; slot != test.selected; ++slot) {
            ids[slot] = (slot * 7u + 3u) % 23u;
            const size_t base = static_cast<size_t>(slot) * 2u * test.intermediate;
            const size_t output_base = static_cast<size_t>(slot) * test.intermediate;
            for (uint32_t channel = 0; channel != test.intermediate; ++channel) {
                const float gate = static_cast<float>(
                    static_cast<int>((slot * 17u + channel * 13u) % 97u) - 48) / 11.0f;
                const float up = static_cast<float>(
                    static_cast<int>((slot * 19u + channel * 5u) % 89u) - 44) / 13.0f;
                gate_up[base + channel] = gate;
                gate_up[base + test.intermediate + channel] = up;
                expected[output_base + channel] = moe_activation_oracle(gate, up, test.swiglu);
            }
        }
        const std::vector<float> gate_up_before = gate_up;
        const std::vector<uint32_t> ids_before = ids;
        MetalMoeActivationTestMetrics metrics;
        metal_dispatch_metrics_reset();
        CHECK(metal_test_moe_batched_activation(
            gate_up.data(), gate_up.size(), ids.data(), test.selected,
            test.intermediate, test.swiglu, MetalMoeActivationTestPath::Batched2D,
            actual.data(), actual.size(), &metrics));
        CHECK(metrics.command_buffers == 1);
        CHECK(metrics.activation_dispatches == 1);
        CHECK(metrics.expert_ids_unchanged);
        CHECK(metrics.gate_up_unchanged);
        CHECK(metal_dispatch_metrics().command_buffers == 1);
        CHECK(gate_up == gate_up_before);
        CHECK(ids == ids_before);
        for (size_t index = 0; index != actual.size(); ++index)
            CHECK_MSG(almost_equal(actual[index], expected[index], 2.0e-5f, 2.0e-5f),
                      "top_k=%u width=%u index=%zu actual=%g expected=%g",
                      test.selected, test.intermediate, index, actual[index], expected[index]);
        std::fprintf(stderr,
                     "moe batched activation case top_k=%u width=%u swiglu=%u "
                     "activation_dispatches=1 command_buffers=1\n",
                     test.selected, test.intermediate, test.swiglu ? 1u : 0u);
    }

    float one_gate_up[2] = {1.0f, 2.0f};
    float one_output = -7.0f;
    const uint32_t one_id = 0;
    auto rejects_without_submission = [&](const float* gate_up, size_t gate_up_values,
                                          const uint32_t* ids, uint32_t selected,
                                          uint32_t intermediate, size_t output_values,
                                          MetalMoeActivationTestPath path) {
        MetalMoeActivationTestMetrics metrics;
        metal_dispatch_metrics_reset();
        CHECK(!metal_test_moe_batched_activation(
            gate_up, gate_up_values, ids, selected, intermediate, false, path,
            &one_output, output_values, &metrics));
        CHECK(metrics.command_buffers == 0);
        CHECK(metrics.activation_dispatches == 0);
        CHECK(metal_dispatch_metrics().command_buffers == 0);
    };
    rejects_without_submission(nullptr, 2, &one_id, 1, 1, 1,
                               MetalMoeActivationTestPath::Batched2D);
    rejects_without_submission(one_gate_up, 2, nullptr, 1, 1, 1,
                               MetalMoeActivationTestPath::Batched2D);
    rejects_without_submission(one_gate_up, 2, &one_id, 0, 1, 1,
                               MetalMoeActivationTestPath::Batched2D);
    rejects_without_submission(one_gate_up, 2, &one_id, 17, 1, 1,
                               MetalMoeActivationTestPath::Batched2D);
    rejects_without_submission(one_gate_up, 2, &one_id, 1, 0, 1,
                               MetalMoeActivationTestPath::Batched2D);
    rejects_without_submission(one_gate_up, 1, &one_id, 1, 1, 1,
                               MetalMoeActivationTestPath::Batched2D);
    rejects_without_submission(one_gate_up, 2, &one_id, 1, 1, 0,
                               MetalMoeActivationTestPath::Batched2D);
    rejects_without_submission(one_gate_up, 2, &one_id, 1, 1, 1,
                               static_cast<MetalMoeActivationTestPath>(99));
    std::fprintf(stderr, "moe batched activation invalid bounds: OK\n");

    constexpr uint32_t selected = 16;
    constexpr uint32_t intermediate = 2816;
    const size_t gate_up_values = static_cast<size_t>(selected) * 2u * intermediate;
    const size_t output_values = static_cast<size_t>(selected) * intermediate;
    std::vector<float> gate_up(gate_up_values), baseline(output_values), candidate(output_values);
    std::array<uint32_t, selected> ids{};
    for (uint32_t slot = 0; slot != selected; ++slot) {
        ids[slot] = (slot * 11u + 5u) % 23u;
        const size_t base = static_cast<size_t>(slot) * 2u * intermediate;
        for (uint32_t channel = 0; channel != intermediate; ++channel) {
            gate_up[base + channel] = static_cast<float>(
                static_cast<int>((slot * 29u + channel * 7u) % 101u) - 50) / 17.0f;
            gate_up[base + intermediate + channel] = static_cast<float>(
                static_cast<int>((slot * 31u + channel * 3u) % 103u) - 51) / 19.0f;
        }
    }
    auto run = [&](MetalMoeActivationTestPath path, float* output) {
        MetalMoeActivationTestMetrics metrics;
        CHECK(metal_test_moe_batched_activation(
            gate_up.data(), gate_up.size(), ids.data(), selected, intermediate, false,
            path, output, output_values, &metrics));
        CHECK(metrics.command_buffers == 1);
        CHECK(metrics.activation_dispatches ==
              (path == MetalMoeActivationTestPath::Batched2D ? 1u : selected));
        return metrics.gpu_ms;
    };
    (void)run(MetalMoeActivationTestPath::PerExpertLoop, baseline.data());
    (void)run(MetalMoeActivationTestPath::Batched2D, candidate.data());
    constexpr size_t samples = 20;
    std::array<double, samples> baseline_ms{}, candidate_ms{};
    uint32_t candidate_wins = 0;
    for (size_t sample = 0; sample != samples; ++sample) {
        if ((sample & 1u) == 0u) {
            baseline_ms[sample] = run(MetalMoeActivationTestPath::PerExpertLoop, baseline.data());
            candidate_ms[sample] = run(MetalMoeActivationTestPath::Batched2D, candidate.data());
        } else {
            candidate_ms[sample] = run(MetalMoeActivationTestPath::Batched2D, candidate.data());
            baseline_ms[sample] = run(MetalMoeActivationTestPath::PerExpertLoop, baseline.data());
        }
        CHECK(std::memcmp(baseline.data(), candidate.data(), output_values * sizeof(float)) == 0);
        if (candidate_ms[sample] < baseline_ms[sample]) ++candidate_wins;
    }
    std::sort(baseline_ms.begin(), baseline_ms.end());
    std::sort(candidate_ms.begin(), candidate_ms.end());
    const double baseline_median = baseline_ms[samples / 2];
    const double candidate_median = candidate_ms[samples / 2];
    CHECK_MSG(candidate_median <= baseline_median,
              "baseline_median_ms=%.6f candidate_median_ms=%.6f candidate_wins=%u/%zu",
              baseline_median, candidate_median, candidate_wins, samples);
    std::fprintf(stderr,
                 "moe batched activation: OK baseline_median_ms=%.6f "
                 "candidate_median_ms=%.6f candidate_wins=%u/%zu max_delta=0\n",
                 baseline_median, candidate_median, candidate_wins, samples);
}

static void test_moe_down_reduce() {
    constexpr int K = 704;
    constexpr int N = 2816;
    constexpr int E = 5;
    constexpr uint32_t R = 4;
    const std::array<uint32_t, R> ids = {3, 1, 3, 0};
    const std::array<float, R> route = {0.25f, -0.5f, 0.75f, 0.5f};
    const std::array<float, E> scales = {2.0f, -0.5f, 1.0f, 1.25f, 0.0f};

    auto run_format = [&](GGMLType type, void (*pack)(const float*, int, int, uint8_t*),
                          const char* name) {
        const size_t expert_stride = static_cast<size_t>(N) * (K / 32) * bytes_per_block(type);
        std::vector<float> logical(static_cast<size_t>(N) * K);
        std::vector<uint8_t> bytes(static_cast<size_t>(E) * expert_stride);
        for (int expert = 0; expert < E; ++expert) {
            for (size_t index = 0; index != logical.size(); ++index)
                logical[index] = static_cast<float>(static_cast<int>(((expert + 1) * 19 + index * 7) % 101) - 50) / 37.0f;
            pack(logical.data(), K, N, bytes.data() + static_cast<size_t>(expert) * expert_stride);
        }

        Tensor down;
        down.type = type;
        down.n_dims = 3;
        down.dims[0] = K;
        down.dims[1] = N;
        down.dims[2] = E;
        down.data = bytes.data();
        CHECK(down.nbytes() == bytes.size());

        std::vector<float> input(static_cast<size_t>(R) * K);
        rng_fill(input.data(), static_cast<int>(input.size()), 8207 + static_cast<unsigned>(type));
        for (float& value : input) value *= 0.02f;
        std::vector<float> expected(N, 0.0f), actual(N, -999.0f);
        for (uint32_t slot = 0; slot != R; ++slot) {
            Tensor expert_weight = down;
            expert_weight.n_dims = 2;
            expert_weight.dims[2] = 0;
            expert_weight.data = bytes.data() + static_cast<size_t>(ids[slot]) * expert_stride;
            std::vector<float> contribution;
            const std::vector<float> slot_input(
                input.begin() + static_cast<size_t>(slot) * K,
                input.begin() + static_cast<size_t>(slot + 1) * K);
            reference_quantized_matmul(slot_input, expert_weight, contribution, K, N);
            for (int j = 0; j != N; ++j)
                expected[j] += route[slot] * scales[ids[slot]] * contribution[j];
        }

        const MetalMoeDownReduceSpec spec{K, N, E, R};
        MetalMoeDownReducePipelineCaps capabilities;
        double gpu_ms = 0.0;
        metal_dispatch_metrics_reset();
        CHECK(metal_test_moe_down_reduce(spec, down, bytes.size(), input.data(), input.size(),
                                          ids.data(), route.data(), scales.data(), scales.size(),
                                          actual.data(), actual.size(), &gpu_ms, &capabilities));
        CHECK(metal_dispatch_metrics().command_buffers == 1);
        CHECK(capabilities.thread_execution_width != 0);
        CHECK(capabilities.max_total_threads_per_threadgroup >=
              capabilities.thread_execution_width);
        CHECK(gpu_ms > 0.0);
        for (int j = 0; j != N; ++j)
            CHECK_MSG(almost_equal(actual[j], expected[j], 3.0e-4f, 3.0e-4f),
                      "%s j=%d actual=%g expected=%g", name, j, actual[j], expected[j]);

        const std::array<uint32_t, R> reordered_ids = {0, 3, 1, 3};
        const std::array<uint32_t, R> source_slot = {3, 0, 1, 2};
        const std::array<float, R> reordered_route = {route[3], route[0], route[1], route[2]};
        std::vector<float> reordered_input(static_cast<size_t>(R) * K);
        for (uint32_t slot = 0; slot != R; ++slot)
            std::copy_n(input.data() + static_cast<size_t>(source_slot[slot]) * K, K,
                        reordered_input.data() + static_cast<size_t>(slot) * K);
        std::fill(actual.begin(), actual.end(), -999.0f);
        metal_dispatch_metrics_reset();
        CHECK(metal_test_moe_down_reduce(spec, down, bytes.size(), reordered_input.data(),
                                          reordered_input.size(), reordered_ids.data(),
                                          reordered_route.data(), scales.data(), scales.size(),
                                          actual.data(), actual.size(), &gpu_ms, &capabilities));
        CHECK(metal_dispatch_metrics().command_buffers == 1);
        for (int j = 0; j != N; ++j)
            CHECK_MSG(almost_equal(actual[j], expected[j], 5.0e-4f, 5.0e-4f),
                      "%s reordered j=%d actual=%g expected=%g", name, j, actual[j], expected[j]);

        std::fill(expected.begin(), expected.end(), 0.0f);
        for (uint32_t slot = 0; slot != R; ++slot) {
            Tensor expert_weight = down;
            expert_weight.n_dims = 2;
            expert_weight.dims[2] = 0;
            expert_weight.data = bytes.data() + static_cast<size_t>(ids[slot]) * expert_stride;
            std::vector<float> contribution;
            const std::vector<float> slot_input(
                input.begin() + static_cast<size_t>(slot) * K,
                input.begin() + static_cast<size_t>(slot + 1) * K);
            reference_quantized_matmul(slot_input, expert_weight, contribution, K, N);
            for (int j = 0; j != N; ++j) expected[j] += route[slot] * contribution[j];
        }
        std::fill(actual.begin(), actual.end(), -999.0f);
        metal_dispatch_metrics_reset();
        CHECK(metal_test_moe_down_reduce(spec, down, bytes.size(), input.data(), input.size(),
                                          ids.data(), route.data(), nullptr, 0, actual.data(),
                                          actual.size(), &gpu_ms, &capabilities));
        CHECK(metal_dispatch_metrics().command_buffers == 1);
        for (int j = 0; j != N; ++j)
            CHECK_MSG(almost_equal(actual[j], expected[j], 3.0e-4f, 3.0e-4f),
                      "%s unscaled j=%d actual=%g expected=%g", name, j, actual[j], expected[j]);

        auto rejects_without_submission = [&](const MetalMoeDownReduceSpec& malformed_spec,
                                              const Tensor& malformed_weight,
                                              size_t malformed_source_bytes,
                                              const float* malformed_input,
                                              size_t malformed_input_values,
                                              const uint32_t* malformed_ids,
                                              const float* malformed_route,
                                              const float* malformed_scales,
                                              size_t malformed_scale_values,
                                              size_t malformed_output_values) {
            metal_dispatch_metrics_reset();
            CHECK(!metal_test_moe_down_reduce(
                malformed_spec, malformed_weight, malformed_source_bytes, malformed_input,
                malformed_input_values, malformed_ids, malformed_route, malformed_scales,
                malformed_scale_values, actual.data(), malformed_output_values, &gpu_ms,
                &capabilities));
            CHECK(metal_dispatch_metrics().command_buffers == 0);
        };
        Tensor wrong_type = down;
        wrong_type.type = GGMLType::Q4_K;
        rejects_without_submission(spec, wrong_type, bytes.size(), input.data(), input.size(),
                                   ids.data(), route.data(), scales.data(), scales.size(), actual.size());
        Tensor wrong_shape = down;
        wrong_shape.dims[0] = K - 32;
        rejects_without_submission(spec, wrong_shape, bytes.size(), input.data(), input.size(),
                                   ids.data(), route.data(), scales.data(), scales.size(), actual.size());
        rejects_without_submission(spec, down, bytes.size() - 1, input.data(), input.size(),
                                   ids.data(), route.data(), scales.data(), scales.size(), actual.size());
        rejects_without_submission(spec, down, bytes.size(), input.data(), input.size() - 1,
                                   ids.data(), route.data(), scales.data(), scales.size(), actual.size());
        rejects_without_submission(spec, down, bytes.size(), input.data(), input.size(),
                                   ids.data(), route.data(), scales.data(), scales.size(), actual.size() - 1);
        std::array<uint32_t, R> bad_ids = ids;
        bad_ids[1] = E;
        rejects_without_submission(spec, down, bytes.size(), input.data(), input.size(),
                                   bad_ids.data(), route.data(), scales.data(), scales.size(), actual.size());
        std::array<float, R> bad_route = route;
        bad_route[0] = std::numeric_limits<float>::quiet_NaN();
        rejects_without_submission(spec, down, bytes.size(), input.data(), input.size(),
                                   ids.data(), bad_route.data(), scales.data(), scales.size(), actual.size());
        std::array<float, E> bad_scales = scales;
        bad_scales[0] = std::numeric_limits<float>::infinity();
        rejects_without_submission(spec, down, bytes.size(), input.data(), input.size(),
                                   ids.data(), route.data(), bad_scales.data(), bad_scales.size(), actual.size());
        const MetalMoeDownReduceSpec zero_selected{K, N, E, 0};
        rejects_without_submission(zero_selected, down, bytes.size(), input.data(), input.size(),
                                   ids.data(), route.data(), scales.data(), scales.size(), actual.size());
        const MetalMoeDownReduceSpec wrong_experts{K, N, E + 1u, R};
        rejects_without_submission(wrong_experts, down, bytes.size(), input.data(), input.size(),
                                   ids.data(), route.data(), scales.data(), scales.size(), actual.size());
        const MetalMoeDownReduceSpec overflow{UINT32_MAX, N, E, R};
        rejects_without_submission(overflow, down, bytes.size(), input.data(), input.size(),
                                   ids.data(), route.data(), scales.data(), scales.size(), actual.size());
        std::fprintf(stderr, "PASS generic MoE %s gathered down + weighted reduce: experts=%d selected=%u one_cb=1\n",
                     name, E, R);
    };

    run_format(GGMLType::Q5_0, pack_q5_0, "Q5_0");
    run_format(GGMLType::Q8_0, pack_q8_0, "Q8_0");
}

static void test_derived_q2k_runs_on_metal() {
    constexpr int K = 256;
    constexpr int N = 32;
    std::vector<float> input(K), logical(static_cast<size_t>(K) * N);
    rng_fill(input.data(), K, 1901);
    rng_fill(logical.data(), static_cast<int>(logical.size()), 1902);
    for (float& value : logical) value *= 0.01f;

    std::vector<uint8_t> source(static_cast<size_t>(N) * sizeof(block_q4_K));
    pack_q4_K(logical.data(), K, N, source.data());
    DerivedQ2KStorage derived;
    DerivedStorageError error = DerivedStorageError::None;
    CHECK(derive_q2_k_from_gguf(GGMLType::Q4_K, source, K, N, &derived, &error));
    CHECK(error == DerivedStorageError::None);

    Tensor weight;
    weight.type = GGMLType::Q2_K;
    weight.n_dims = 2;
    weight.dims[0] = K;
    weight.dims[1] = N;
    weight.data = derived.bytes.data();
    std::vector<float> expected(N), actual(N);
    matmul_rows(input.data(), weight, expected.data(), 1, K, N);
    CHECK(metal_gemv_repeat(input.data(), weight, actual.data(), K, N, 3));
    for (int row = 0; row != N; ++row)
        CHECK(almost_equal(actual[row], expected[row], 1.0e-3f, 1.0e-4f));
}

static void test_activation_importance_accumulates_on_device() {
    constexpr uint32_t width = 7;
    const std::array<float, width> first = {1.0f, -2.0f, 0.5f, 0.0f, 3.0f, -4.0f, 2.5f};
    const std::array<float, width> second = {-3.0f, 1.0f, 1.5f, -2.0f, 0.5f, 2.0f, -1.5f};
    std::array<float, width> sums{};
    uint32_t sample_count = 0;
    CHECK(metal_test_activation_importance_accumulator(
        first.data(), second.data(), width, sums.data(), &sample_count));
    CHECK(sample_count == 2);
    for (uint32_t index = 0; index != width; ++index) {
        const float expected = first[index] * first[index] + second[index] * second[index];
        CHECK(almost_equal(sums[index], expected, 1.0e-6f, 1.0e-6f));
    }
    auto nonfinite = second;
    nonfinite[3] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!metal_test_activation_importance_accumulator(
        first.data(), nonfinite.data(), width, sums.data(), &sample_count));

    std::shared_ptr<MetalTokSession> empty = metal_tok_session_create();
    CHECK(empty != nullptr);
    CHECK(metal_tok_session_set_importance_slots(*empty, &width, 1));
    CHECK(!metal_tok_session_read_importance(*empty, 0, sums.data(), width, &sample_count));
}

static void test_q2k_two_row_pipeline_matches_current() {
    constexpr int K = 256;
    constexpr int N = 32;
    std::vector<float> input(K), logical(static_cast<size_t>(K) * N);
    rng_fill(input.data(), K, 1921);
    rng_fill(logical.data(), static_cast<int>(logical.size()), 1922);
    for (float& value : logical) value *= 0.01f;

    std::vector<uint8_t> source(static_cast<size_t>(N) * sizeof(block_q4_K));
    pack_q4_K(logical.data(), K, N, source.data());
    DerivedQ2KStorage derived;
    DerivedStorageError error = DerivedStorageError::None;
    CHECK(derive_q2_k_from_gguf(GGMLType::Q4_K, source, K, N, &derived, &error));

    Tensor weight;
    weight.type = GGMLType::Q2_K;
    weight.n_dims = 2;
    weight.dims[0] = K;
    weight.dims[1] = N;
    weight.data = derived.bytes.data();
    std::vector<float> expected(N), baseline(N), candidate(N);
    matmul_rows(input.data(), weight, expected.data(), 1, K, N);
    double baseline_gpu_ms = 0.0;
    double candidate_gpu_ms = 0.0;
    CHECK(metal_test_q2k_two_row_ab(input.data(), weight,
                                    baseline.data(), candidate.data(), K, N, 3,
                                    &baseline_gpu_ms, &candidate_gpu_ms));
    for (int row = 0; row != N; ++row) {
        CHECK(almost_equal(baseline[row], expected[row], 1.0e-3f, 1.0e-4f));
        CHECK(almost_equal(candidate[row], expected[row], 1.0e-3f, 1.0e-4f));
    }
}

static void test_q2k_streamed_pipeline_matches_current() {
    constexpr int K = 256;
    constexpr int N = 32;
    std::vector<float> input(K), logical(static_cast<size_t>(K) * N);
    rng_fill(input.data(), K, 1931);
    rng_fill(logical.data(), static_cast<int>(logical.size()), 1932);
    for (float& value : logical) value *= 0.01f;

    std::vector<uint8_t> source(static_cast<size_t>(N) * sizeof(block_q4_K));
    pack_q4_K(logical.data(), K, N, source.data());
    DerivedQ2KStorage derived;
    DerivedStorageError error = DerivedStorageError::None;
    CHECK(derive_q2_k_from_gguf(GGMLType::Q4_K, source, K, N, &derived, &error));

    Tensor weight;
    weight.type = GGMLType::Q2_K;
    weight.n_dims = 2;
    weight.dims[0] = K;
    weight.dims[1] = N;
    weight.data = derived.bytes.data();
    std::vector<float> expected(N), baseline(N), candidate(N);
    matmul_rows(input.data(), weight, expected.data(), 1, K, N);
    double baseline_gpu_ms = 0.0;
    double candidate_gpu_ms = 0.0;
    CHECK(metal_test_q2k_streamed_ab(input.data(), weight,
                                     baseline.data(), candidate.data(), K, N, 3,
                                     &baseline_gpu_ms, &candidate_gpu_ms));
    uint32_t baseline_width = 0, baseline_max = 0;
    uint32_t two_row_width = 0, two_row_max = 0;
    uint32_t streamed_width = 0, streamed_max = 0;
    CHECK(metal_test_q2k_pipeline_limits(&baseline_width, &baseline_max,
                                         &two_row_width, &two_row_max,
                                         &streamed_width, &streamed_max));
    CHECK(baseline_width == 32 && two_row_width == 32 && streamed_width == 32);
    CHECK(baseline_max >= 64 && two_row_max >= 64 && streamed_max >= 64);
    std::fprintf(stderr,
                 "Q2 pipeline limits: baseline=%u/%u two_row=%u/%u streamed=%u/%u\n",
                 baseline_width, baseline_max, two_row_width, two_row_max,
                 streamed_width, streamed_max);
    for (int row = 0; row != N; ++row) {
        CHECK(almost_equal(baseline[row], expected[row], 1.0e-3f, 1.0e-4f));
        CHECK(almost_equal(candidate[row], expected[row], 1.0e-3f, 1.0e-4f));
    }
}

static void test_sparse_ffn_bank_runs_on_metal() {
    constexpr int H = 512;
    constexpr int I = 1024;
    std::vector<float> gate_values(static_cast<size_t>(H) * I);
    std::vector<float> up_values(static_cast<size_t>(H) * I);
    std::vector<float> down_values(static_cast<size_t>(I) * H);
    rng_fill(gate_values.data(), static_cast<int>(gate_values.size()), 1911);
    rng_fill(up_values.data(), static_cast<int>(up_values.size()), 1912);
    rng_fill(down_values.data(), static_cast<int>(down_values.size()), 1913);
    for (float& value : gate_values) value *= 0.01f;
    for (float& value : up_values) value *= 0.01f;
    for (float& value : down_values) value *= 0.01f;
    std::vector<uint8_t> gate_bytes(static_cast<size_t>(I) * (H / 256) * sizeof(block_q4_K));
    std::vector<uint8_t> up_bytes(static_cast<size_t>(I) * (H / 256) * sizeof(block_q6_K));
    std::vector<uint8_t> down_bytes(static_cast<size_t>(H) * (I / 256) * sizeof(block_q4_K));
    pack_q4_K(gate_values.data(), H, I, gate_bytes.data());
    pack_q6_K(up_values.data(), H, I, up_bytes.data());
    pack_q4_K(down_values.data(), I, H, down_bytes.data());
    Tensor gate;
    gate.type = GGMLType::Q4_K; gate.n_dims = 2; gate.dims[0] = H; gate.dims[1] = I; gate.data = gate_bytes.data();
    Tensor up;
    up.type = GGMLType::Q6_K; up.n_dims = 2; up.dims[0] = H; up.dims[1] = I; up.data = up_bytes.data();
    Tensor down;
    down.type = GGMLType::Q4_K; down.n_dims = 2; down.dims[0] = I; down.dims[1] = H; down.data = down_bytes.data();
    const SparseBlockRun runs[] = {{0, 1}, {2, 1}};
    SparseFfnBank bank;
    SparseBankError error = SparseBankError::None;
    CHECK(pack_sparse_ffn_bank(gate, up, down, runs, bank, error));
    CHECK(bank.physical_bytes * 2 == bank.dense_source_bytes);

    const Tensor packed_gate = bank.gate.tensor();
    const Tensor packed_up = bank.up.tensor();
    const Tensor packed_down = bank.down.tensor();
    std::vector<float> x(H), gate_cpu(512), up_cpu(512), gate_gpu(512), up_gpu(512);
    std::vector<float> hidden_cpu(512), hidden_gpu(512), expected(H), actual(H);
    rng_fill(x.data(), H, 1914);
    matmul_rows(x.data(), packed_gate, gate_cpu.data(), 1, H, 512);
    matmul_rows(x.data(), packed_up, up_cpu.data(), 1, H, 512);
    CHECK(metal_gemv(x.data(), packed_gate, gate_gpu.data(), H, 512));
    CHECK(metal_gemv(x.data(), packed_up, up_gpu.data(), H, 512));
    for (int i = 0; i != 512; ++i) {
        hidden_cpu[i] = gate_cpu[i] / (1.0f + std::exp(-gate_cpu[i])) * up_cpu[i];
        hidden_gpu[i] = gate_gpu[i] / (1.0f + std::exp(-gate_gpu[i])) * up_gpu[i];
    }
    matmul_rows(hidden_cpu.data(), packed_down, expected.data(), 1, 512, H);
    CHECK(metal_gemv(hidden_gpu.data(), packed_down, actual.data(), 512, H));
    float maximum_error = 0.0f;
    float maximum_value = 0.0f;
    for (int i = 0; i != H; ++i) {
        CHECK(std::isfinite(actual[i]));
        maximum_error = std::max(maximum_error, std::fabs(actual[i] - expected[i]));
        maximum_value = std::max(maximum_value, std::fabs(expected[i]));
    }
    const float relative_error = maximum_value > 0.0f ? maximum_error / maximum_value : maximum_error;
    std::fprintf(stderr, "PASS sparse FFN bank: physical %.1f%% rel err %.6f\n",
                 100.0 * static_cast<double>(bank.physical_bytes) / static_cast<double>(bank.dense_source_bytes),
                 relative_error);
    CHECK(relative_error <= 2.0e-2f);

    std::vector<float> gate_full(I), up_full(I), full_hidden(I, 0.0f);
    std::vector<float> original_expected(H), original_actual(H);
    matmul_rows(x.data(), gate, gate_full.data(), 1, H, I);
    matmul_rows(x.data(), up, up_full.data(), 1, H, I);
    for (const SparseBlockRun& run : runs) {
        for (uint32_t block = run.first; block != run.first + run.count; ++block) {
            for (uint32_t offset = 0; offset != 256; ++offset) {
                const size_t channel = static_cast<size_t>(block) * 256 + offset;
                full_hidden[channel] = gate_full[channel] /
                    (1.0f + std::exp(-gate_full[channel])) * up_full[channel];
            }
        }
    }
    matmul_rows(full_hidden.data(), down, original_expected.data(), 1, I, H);
    const MetalSparseBlockRun metal_runs[] = {{0, 1}, {2, 1}};
    metal_dispatch_metrics_reset();
    CHECK(metal_test_sparse_ffn_original_spans(x.data(), gate, up, down, metal_runs,
                                               2, original_actual.data(), H, I));
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    float original_maximum_error = 0.0f;
    float original_maximum_value = 0.0f;
    for (int index = 0; index != H; ++index) {
        CHECK(std::isfinite(original_actual[index]));
        original_maximum_error = std::max(
            original_maximum_error, std::fabs(original_actual[index] - original_expected[index]));
        original_maximum_value = std::max(original_maximum_value, std::fabs(original_expected[index]));
    }
    const float original_relative_error = original_maximum_value > 0.0f
        ? original_maximum_error / original_maximum_value : original_maximum_error;
    std::fprintf(stderr, "PASS original-span sparse FFN: ids=[0,2] mixed=Q4_K/Q6_K rel err %.6f\n",
                 original_relative_error);
    CHECK(original_relative_error <= 2.0e-2f);

    std::array<float, 68> selector_scores{};
    std::array<double, 17> selector_gpu{};
    for (int trial = 0; trial != 17; ++trial) {
        for (uint32_t block = 0; block != selector_scores.size(); ++block)
            selector_scores[block] = static_cast<float>((block * 17u + trial * 13u) % 19u) * 0.001f;
        for (uint32_t block = 17; block != 51; ++block) selector_scores[block] += 1.0f;
        std::array<double, 69> prefix{};
        for (uint32_t block = 0; block != selector_scores.size(); ++block)
            prefix[block + 1] = prefix[block] + selector_scores[block];
        uint32_t expected_first = 0;
        double expected_energy = -1.0;
        for (uint32_t first = 0; first + 34 <= selector_scores.size(); ++first) {
            const double energy = prefix[first + 34] - prefix[first];
            if (energy > expected_energy) {
                expected_energy = energy;
                expected_first = first;
            }
        }
        CHECK(expected_first == 17);
        uint32_t selected_first = UINT32_MAX;
        uint32_t selected_count = 0;
        CHECK(metal_test_select_contiguous_window(selector_scores.data(), selector_scores.size(), 34,
                                                   &selected_first, &selected_count,
                                                   &selector_gpu[trial]));
        CHECK(selected_first == expected_first);
        CHECK(selected_count == 34);
    }
    std::sort(selector_gpu.begin(), selector_gpu.end());
    std::fprintf(stderr,
                 "PASS device contiguous selector: ids=17/17 Kout=34 median_gpu_ms=%.6f range_ms=%.6f..%.6f\n",
                 selector_gpu[selector_gpu.size() / 2], selector_gpu.front(), selector_gpu.back());

    const MetalSparseBlockRun all_run[] = {{0, I / 256}};
    std::vector<float> all_affine(H), all_selected(H);
    CHECK(metal_test_sparse_ffn_original_spans(x.data(), gate, up, down, all_run, 1,
                                               all_affine.data(), H, I));
    std::array<float, I / 256> all_scores{};
    for (uint32_t block = 0; block != all_scores.size(); ++block)
        all_scores[block] = static_cast<float>(block + 1);
    uint32_t all_first = UINT32_MAX;
    double selected_gpu_ms = 0.0;
    metal_dispatch_metrics_reset();
    CHECK(metal_test_sparse_ffn_selected_window(
        x.data(), gate, up, down, all_scores.data(), all_scores.size(), all_scores.size(),
        all_selected.data(), &all_first, H, I, &selected_gpu_ms));
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    CHECK(all_first == 0);
    CHECK(all_selected == all_affine);
    std::fprintf(stderr,
                 "PASS device-selected all-block FFN: first=%u count=%zu one_cb=1 gpu_ms=%.6f exact=1\n",
                 all_first, all_scores.size(), selected_gpu_ms);

    constexpr uint32_t input_blocks = H / 256;
    constexpr uint32_t output_blocks = I / 256;
    std::array<float, input_blocks> input_energy{};
    for (uint32_t block = 0; block != input_blocks; ++block)
        for (uint32_t offset = 0; offset != 256; ++offset) {
            const float value = x[block * 256 + offset];
            input_energy[block] += value * value;
        }
    std::array<float, input_blocks * output_blocks> proxy{};
    for (uint32_t output_block = 0; output_block != output_blocks; ++output_block)
        for (uint32_t input_block = 0; input_block != input_blocks; ++input_block)
            proxy[output_block * input_blocks + input_block] =
                0.001f * static_cast<float>(1 + input_block) +
                (output_block == 1 || output_block == 2 ? 1.0f : 0.01f);
    std::array<double, output_blocks + 1> proxy_prefix{};
    for (uint32_t output_block = 0; output_block != output_blocks; ++output_block) {
        double score = 0.0;
        for (uint32_t input_block = 0; input_block != input_blocks; ++input_block)
            score += static_cast<double>(input_energy[input_block]) *
                     proxy[output_block * input_blocks + input_block];
        proxy_prefix[output_block + 1] = proxy_prefix[output_block] + score;
    }
    uint32_t proxy_expected_first = 0;
    double proxy_best = -1.0;
    for (uint32_t first = 0; first + 2 <= output_blocks; ++first) {
        const double score = proxy_prefix[first + 2] - proxy_prefix[first];
        if (score > proxy_best) {
            proxy_best = score;
            proxy_expected_first = first;
        }
    }
    CHECK(proxy_expected_first == 1);
    const MetalSparseBlockRun proxy_run[] = {{proxy_expected_first, 2}};
    std::vector<float> proxy_affine(H), proxy_selected(H);
    CHECK(metal_test_sparse_ffn_original_spans(x.data(), gate, up, down, proxy_run, 1,
                                               proxy_affine.data(), H, I));
    uint32_t proxy_first = UINT32_MAX;
    double proxy_gpu_ms = 0.0;
    metal_dispatch_metrics_reset();
    CHECK(metal_test_sparse_ffn_proxy_window(
        x.data(), gate, up, down, proxy.data(), input_blocks, output_blocks, 2,
        proxy_selected.data(), &proxy_first, H, I, &proxy_gpu_ms));
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    CHECK(proxy_first == proxy_expected_first);
    CHECK(proxy_selected == proxy_affine);
    std::fprintf(stderr,
                 "PASS block-energy-surrogate-selected FFN: first=%u count=2 one_cb=1 gpu_ms=%.6f exact_retained=1\n",
                 proxy_first, proxy_gpu_ms);
}

static std::vector<float> reference_q6k_embedding_row(const Tensor& embedding, uint32_t token,
                                                       int width, int vocabulary) {
    std::vector<float> output(width);
    if (embedding.type != GGMLType::Q6_K || embedding.n_dims != 2 || embedding.dims[0] != static_cast<uint64_t>(width) ||
        embedding.dims[1] != static_cast<uint64_t>(vocabulary) || width % 256 != 0 || token >= static_cast<uint32_t>(vocabulary)) {
        return {};
    }
    const auto* rows = reinterpret_cast<const block_q6_K*>(embedding.data);
    const int blocks = width / 256;
    for (int block = 0; block != blocks; ++block) {
        const block_q6_K& source = rows[static_cast<size_t>(token) * blocks + block];
        const float d = fp16_to_fp32(source.d);
        for (int half = 0; half != 2; ++half) {
            const uint8_t* ql = source.ql + half * 64;
            const uint8_t* qh = source.qh + half * 32;
            const int8_t* scales = source.scales + half * 8;
            const int base = block * 256 + half * 128;
            for (int lane = 0; lane != 32; ++lane) {
                const int scale_group = lane / 16;
                output[base + lane] = d * scales[scale_group] *
                    (static_cast<int>((ql[lane] & 0x0f) | (((qh[lane] >> 0) & 3) << 4)) - 32);
                output[base + 32 + lane] = d * scales[scale_group + 2] *
                    (static_cast<int>((ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 3) << 4)) - 32);
                output[base + 64 + lane] = d * scales[scale_group + 4] *
                    (static_cast<int>((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32);
                output[base + 96 + lane] = d * scales[scale_group + 6] *
                    (static_cast<int>((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32);
            }
        }
    }
    return output;
}

// Independent FP32 GGML-K decoder. This intentionally does not call the
// production Q4_K GEMV decoder or any Metal helper.
static std::vector<float> reference_q4k_embedding_row(const Tensor& embedding, uint32_t token,
                                                       int width, int vocabulary) {
    std::vector<float> output(width);
    if (embedding.type != GGMLType::Q4_K || embedding.n_dims != 2 ||
        embedding.dims[0] != static_cast<uint64_t>(width) ||
        embedding.dims[1] != static_cast<uint64_t>(vocabulary) || width % 256 != 0 ||
        token >= static_cast<uint32_t>(vocabulary)) {
        return {};
    }
    const auto* rows = reinterpret_cast<const block_q4_K*>(embedding.data);
    const int blocks = width / 256;
    for (int block = 0; block != blocks; ++block) {
        const block_q4_K& source = rows[static_cast<size_t>(token) * blocks + block];
        const float d = fp16_to_fp32(source.d);
        const float dmin = fp16_to_fp32(source.dmin);
        for (int chunk = 0; chunk != 4; ++chunk) {
            uint8_t scale_lo = 0, minimum_lo = 0, scale_hi = 0, minimum_hi = 0;
            reference_q4_scale_min(2 * chunk, source.scales, scale_lo, minimum_lo);
            reference_q4_scale_min(2 * chunk + 1, source.scales, scale_hi, minimum_hi);
            const uint8_t* quant = source.qs + chunk * 32;
            const int base = block * 256 + chunk * 64;
            for (int lane = 0; lane != 32; ++lane) {
                output[base + lane] = d * scale_lo * static_cast<float>(quant[lane] & 0x0f) -
                                      dmin * minimum_lo;
                output[base + 32 + lane] = d * scale_hi * static_cast<float>(quant[lane] >> 4) -
                                           dmin * minimum_hi;
            }
        }
    }
    return output;
}

static void test_q6k_embedding_matches_independent_row_decoder() {
    constexpr int width = 256;
    constexpr int vocabulary = 3;
    constexpr uint32_t token = 2;
    std::vector<float> logical(static_cast<size_t>(width) * vocabulary);
    rng_fill(logical.data(), static_cast<int>(logical.size()), 799);
    for (float& value : logical) value *= 0.05f;
    std::vector<uint8_t> packed(static_cast<size_t>(vocabulary) * sizeof(block_q6_K));
    pack_q6_K(logical.data(), width, vocabulary, packed.data());
    Tensor embedding;
    embedding.type = GGMLType::Q6_K;
    embedding.n_dims = 2;
    embedding.dims[0] = width;
    embedding.dims[1] = vocabulary;
    embedding.data = packed.data();
    const std::vector<float> expected = reference_q6k_embedding_row(embedding, token, width, vocabulary);
    std::vector<float> actual(width);
    CHECK(expected.size() == actual.size());
    CHECK(metal_test_q6k_embedding(embedding, token, actual.data(), width, vocabulary));
    for (size_t index = 0; index != actual.size(); ++index)
        CHECK(almost_equal(actual[index], expected[index], 2.0e-6f, 2.0e-6f));
}

static void test_q4k_embedding_matches_independent_row_decoder() {
    constexpr int width = 256;
    constexpr int vocabulary = 3;
    constexpr uint32_t token = 1;
    std::vector<float> logical(static_cast<size_t>(width) * vocabulary);
    rng_fill(logical.data(), static_cast<int>(logical.size()), 800);
    for (float& value : logical) value *= 0.05f;
    std::vector<uint8_t> packed(static_cast<size_t>(vocabulary) * sizeof(block_q4_K));
    pack_q4_K(logical.data(), width, vocabulary, packed.data());
    Tensor embedding;
    embedding.type = GGMLType::Q4_K;
    embedding.n_dims = 2;
    embedding.dims[0] = width;
    embedding.dims[1] = vocabulary;
    embedding.data = packed.data();
    const std::vector<float> expected = reference_q4k_embedding_row(embedding, token, width, vocabulary);
    std::vector<float> actual(width);
    CHECK(expected.size() == actual.size());
    CHECK(metal_test_q4k_embedding(embedding, token, actual.data(), width, vocabulary));
    for (size_t index = 0; index != actual.size(); ++index)
        CHECK(almost_equal(actual[index], expected[index], 2.0e-6f, 2.0e-6f));
}

// The recurrent descriptor must consume the same Q4_K/Q6_K planes used by a
// semantic model while staying inside the session token transaction. This is
// deliberately not a named-architecture fixture.
static void test_session_recurrent_layer_consumes_quantized_planes() {
    constexpr int H = 256;
    constexpr int qk_heads = 1;
    constexpr int value_heads = 2;
    constexpr int head_dimension = 128;
    constexpr int kernel = 3;
    constexpr int channels = head_dimension * (2 * qk_heads + value_heads);
    constexpr int output_width = value_heads * head_dimension;
    constexpr int ffn_intermediate = 256;
    constexpr float l2_epsilon = 1.0e-6f;
    constexpr float rms_epsilon = 1.0e-6f;

    auto fill_small = [](std::vector<float>& values, unsigned seed) {
        rng_fill(values.data(), static_cast<int>(values.size()), seed);
        for (float& value : values) value *= 0.01f;
    };
    auto f32_tensor = [](const std::vector<float>& values, uint64_t dim0, uint64_t dim1 = 0) {
        Tensor tensor;
        tensor.type = GGMLType::F32;
        tensor.n_dims = dim1 ? 2 : 1;
        tensor.dims[0] = dim0;
        tensor.dims[1] = dim1;
        tensor.data = reinterpret_cast<const uint8_t*>(values.data());
        return tensor;
    };
    struct PackedTensor {
        std::vector<uint8_t> bytes;
        Tensor tensor;
    };
    auto packed_tensor = [&](GGMLType type, const std::vector<float>& values, int K, int N,
                             void (*pack)(const float*, int, int, uint8_t*)) {
        PackedTensor result;
        result.bytes.resize(static_cast<size_t>(N) * static_cast<size_t>(K / 256) * bytes_per_block(type));
        pack(values.data(), K, N, result.bytes.data());
        result.tensor.type = type;
        result.tensor.n_dims = 2;
        result.tensor.dims[0] = K;
        result.tensor.dims[1] = N;
        result.tensor.data = result.bytes.data();
        return result;
    };
    auto rms = [](const std::vector<float>& input, const std::vector<float>& weight, float epsilon) {
        float sum = 0.0f;
        for (float value : input) sum += value * value;
        const float scale = 1.0f / std::sqrt(sum / static_cast<float>(input.size()) + epsilon);
        std::vector<float> output(input.size());
        for (size_t index = 0; index != input.size(); ++index) output[index] = input[index] * scale * weight[index];
        return output;
    };

    std::vector<float> input(H), input_norm(H), qkv_weight(static_cast<size_t>(H) * channels),
                       gate_weight(static_cast<size_t>(H) * output_width), beta_weight(static_cast<size_t>(H) * value_heads),
                       alpha_weight(static_cast<size_t>(H) * value_heads), output_weight(static_cast<size_t>(output_width) * H),
                       conv_weight(static_cast<size_t>(channels) * kernel), dt_bias(value_heads), decay(value_heads),
                       recurrent_norm(head_dimension), history(static_cast<size_t>(channels) * (kernel - 1), 0.0f),
                       state(static_cast<size_t>(value_heads) * head_dimension * head_dimension, 0.0f),
                       ffn_norm(H), ffn_gate_weight(static_cast<size_t>(H) * ffn_intermediate),
                       ffn_up_weight(static_cast<size_t>(H) * ffn_intermediate),
                       ffn_down_weight(static_cast<size_t>(ffn_intermediate) * H);
    fill_small(input, 711);
    fill_small(input_norm, 712);
    fill_small(qkv_weight, 713);
    fill_small(gate_weight, 714);
    fill_small(beta_weight, 715);
    fill_small(alpha_weight, 716);
    fill_small(output_weight, 717);
    fill_small(conv_weight, 718);
    fill_small(dt_bias, 719);
    fill_small(decay, 720);
    fill_small(recurrent_norm, 721);
    fill_small(ffn_norm, 722);
    fill_small(ffn_gate_weight, 723);
    fill_small(ffn_up_weight, 724);
    fill_small(ffn_down_weight, 725);
    for (float& value : decay) value = -std::fabs(value);
    for (float& value : recurrent_norm) value += 1.0f;

    const Tensor input_norm_tensor = f32_tensor(input_norm, H);
    const Tensor conv_tensor = f32_tensor(conv_weight, kernel, channels);
    const Tensor dt_tensor = f32_tensor(dt_bias, value_heads);
    const Tensor decay_tensor = f32_tensor(decay, value_heads);
    const Tensor recurrent_norm_tensor = f32_tensor(recurrent_norm, head_dimension);
    const Tensor ffn_norm_tensor = f32_tensor(ffn_norm, H);
    PackedTensor qkv_tensor = packed_tensor(GGMLType::Q6_K, qkv_weight, H, channels, pack_q6_K);
    PackedTensor gate_tensor = packed_tensor(GGMLType::Q4_K, gate_weight, H, output_width, pack_q4_K);
    PackedTensor beta_tensor = packed_tensor(GGMLType::Q4_K, beta_weight, H, value_heads, pack_q4_K);
    PackedTensor alpha_tensor = packed_tensor(GGMLType::Q4_K, alpha_weight, H, value_heads, pack_q4_K);
    PackedTensor output_tensor = packed_tensor(GGMLType::Q4_K, output_weight, output_width, H, pack_q4_K);
    PackedTensor ffn_gate_tensor = packed_tensor(GGMLType::Q4_K, ffn_gate_weight, H, ffn_intermediate, pack_q4_K);
    PackedTensor ffn_up_tensor = packed_tensor(GGMLType::Q4_K, ffn_up_weight, H, ffn_intermediate, pack_q4_K);
    PackedTensor ffn_down_tensor = packed_tensor(GGMLType::Q4_K, ffn_down_weight, ffn_intermediate, H, pack_q4_K);

    const std::vector<float> normalized = rms(input, input_norm, rms_epsilon);
    std::vector<float> qkv(channels), gate(output_width), beta(value_heads), alpha(value_heads), projected(output_width);
    reference_quantized_matmul(normalized, qkv_tensor.tensor, qkv, H, channels);
    reference_quantized_matmul(normalized, gate_tensor.tensor, gate, H, output_width);
    reference_quantized_matmul(normalized, beta_tensor.tensor, beta, H, value_heads);
    reference_quantized_matmul(normalized, alpha_tensor.tensor, alpha, H, value_heads);
    std::vector<float> expected_history, expected_state, recurrent_output;
    gated_delta_reference(qkv, conv_weight, history, gate, beta, alpha, dt_bias, decay, recurrent_norm, state,
                          expected_history, expected_state, recurrent_output, qk_heads, value_heads,
                          head_dimension, kernel, l2_epsilon, rms_epsilon);
    reference_quantized_matmul(recurrent_output, output_tensor.tensor, projected, output_width, H);
    std::vector<float> recurrent_residual(H);
    for (int index = 0; index != H; ++index) recurrent_residual[index] = input[index] + projected[index];
    const std::vector<float> ffn_normalized = rms(recurrent_residual, ffn_norm, rms_epsilon);
    std::vector<float> ffn_gate(ffn_intermediate), ffn_up(ffn_intermediate), ffn_hidden(ffn_intermediate), ffn_down(H);
    reference_quantized_matmul(ffn_normalized, ffn_gate_tensor.tensor, ffn_gate, H, ffn_intermediate);
    reference_quantized_matmul(ffn_normalized, ffn_up_tensor.tensor, ffn_up, H, ffn_intermediate);
    for (int index = 0; index != ffn_intermediate; ++index) {
        const float gate_value = ffn_gate[index];
        const float sigmoid = gate_value >= 0.0f ? 1.0f / (1.0f + std::exp(-gate_value))
                                                 : std::exp(gate_value) / (1.0f + std::exp(gate_value));
        ffn_hidden[index] = gate_value * sigmoid * ffn_up[index];
    }
    reference_quantized_matmul(ffn_hidden, ffn_down_tensor.tensor, ffn_down, ffn_intermediate, H);
    std::vector<float> expected_x(H);
    for (int index = 0; index != H; ++index) expected_x[index] = recurrent_residual[index] + ffn_down[index];

    auto session = metal_tok_session_create();
    CHECK(session != nullptr);
    if (!session) return;
    CHECK(metal_tok_session_recurrent_ready(*session));
    CHECK(metal_tok_session_begin(*session, H, channels, 0, 0, 0, 1, 1, head_dimension, 4, 1, 0));
    CHECK(metal_tok_session_upload_x(*session, input.data(), H));
    MetalTokRecurrentLayer layer;
    layer.input_norm = &input_norm_tensor;
    layer.qkv = &qkv_tensor.tensor;
    layer.gate = &gate_tensor.tensor;
    layer.beta = &beta_tensor.tensor;
    layer.alpha = &alpha_tensor.tensor;
    layer.conv = &conv_tensor;
    layer.dt_bias = &dt_tensor;
    layer.decay = &decay_tensor;
    layer.norm = &recurrent_norm_tensor;
    layer.output = &output_tensor.tensor;
    layer.ffn_norm = &ffn_norm_tensor;
    layer.ffn_gate = &ffn_gate_tensor.tensor;
    layer.ffn_up = &ffn_up_tensor.tensor;
    layer.ffn_down = &ffn_down_tensor.tensor;
    layer.state_slot = kSemanticModelMaximumLayers - 1;
    layer.H = H;
    layer.qk_heads = qk_heads;
    layer.value_heads = value_heads;
    layer.head_dimension = head_dimension;
    layer.kernel = kernel;
    layer.ffn_intermediate = ffn_intermediate;
    layer.l2_epsilon = l2_epsilon;
    layer.rms_epsilon = rms_epsilon;
    CHECK(metal_tok_session_recurrent_layer(*session, layer));
    CHECK(metal_tok_session_recurrent_commit(*session));
    const MetalTokMetrics projection_metrics = metal_tok_session_metrics(*session);
    CHECK(projection_metrics.projection_dispatches == 8);
    CHECK(projection_metrics.q4k_projection_dispatches == 7);
    CHECK(projection_metrics.q6k_projection_dispatches == 1);
    CHECK(projection_metrics.requested_projection_source_bytes == 292416);

    std::vector<float> actual_history(history.size()), actual_state(state.size()), actual_x(H);
    CHECK(metal_tok_session_recurrent_snapshot_slot(*session, layer.state_slot, actual_history.data(), actual_state.data(),
                                                    qk_heads, value_heads, head_dimension, kernel));
    CHECK(metal_tok_session_download_x_for_testing(*session, actual_x.data(), H));
    auto maximum_error = [](const std::vector<float>& actual, const std::vector<float>& expected) {
        float error = 0.0f;
        for (size_t index = 0; index != actual.size(); ++index) error = std::max(error, std::fabs(actual[index] - expected[index]));
        return error;
    };
    const float history_error = maximum_error(actual_history, expected_history);
    const float state_error = maximum_error(actual_state, expected_state);
    const float x_error = maximum_error(actual_x, expected_x);
    fprintf(stderr, "[metal] recurrent quantized binding error: history %.7g state %.7g x %.7g\n",
            history_error, state_error, x_error);
    CHECK(history_error < 2.0e-4f);
    CHECK(state_error < 2.0e-4f);
    CHECK(x_error < 1.0e-3f);

    auto rejected_session = metal_tok_session_create();
    CHECK(rejected_session != nullptr);
    if (!rejected_session) return;
    CHECK(metal_tok_session_begin(*rejected_session, H, channels, 0, 0, 0, 1, 1, head_dimension, 4, 1, 0));
    CHECK(metal_tok_session_upload_x(*rejected_session, input.data(), H));
    CHECK(metal_tok_session_recurrent_layer(*rejected_session, layer));
    metal_tok_session_recurrent_fail_after_completed_submission_for_testing(*rejected_session);
    CHECK(!metal_tok_session_recurrent_commit(*rejected_session));
    std::fill(actual_history.begin(), actual_history.end(), 1.0f);
    std::fill(actual_state.begin(), actual_state.end(), 1.0f);
    CHECK(metal_tok_session_recurrent_snapshot_slot(*rejected_session, layer.state_slot, actual_history.data(), actual_state.data(),
                                                    qk_heads, value_heads, head_dimension, kernel));
    CHECK(maximum_error(actual_history, history) == 0.0f);
    CHECK(maximum_error(actual_state, state) == 0.0f);

    auto out_of_range_session = metal_tok_session_create();
    CHECK(out_of_range_session != nullptr);
    if (out_of_range_session) {
        CHECK(metal_tok_session_begin(*out_of_range_session, H, channels, 0, 0, 0,
                                      1, 1, head_dimension, 4, 1, 0));
        CHECK(metal_tok_session_upload_x(*out_of_range_session, input.data(), H));
        layer.state_slot = kSemanticModelMaximumLayers;
        CHECK(!metal_tok_session_recurrent_layer(*out_of_range_session, layer));
    }
}

void test_column_grouped_q4_v1_metal() {
    constexpr uint32_t K = 8;
    constexpr uint32_t N = 512;
    std::vector<float> source(static_cast<size_t>(K) * N);
    for (uint32_t n = 0; n != N; ++n) {
        for (uint32_t k = 0; k != K; ++k) {
            source[static_cast<size_t>(n) * K + k] =
                static_cast<float>(static_cast<int>((n * 37u + k * 13u) % 71u) - 35) / 9.0f;
        }
    }

    ColumnGroupedQ4V1Storage storage;
    ColumnGroupedQ4Error error = ColumnGroupedQ4Error::None;
    CHECK(column_grouped_q4_v1_from_f32(source, K, N, &storage, &error));
    CHECK(error == ColumnGroupedQ4Error::None);

    const std::array<float, K> sparse_input = {0.0f, -0.5f, 0.0f, 1.0f,
                                                 0.0f, -1.5f, 0.0f, 2.0f};
    const std::array<uint32_t, 4> active_columns = {1u, 3u, 5u, 7u};
    const std::array<uint32_t, K> all_columns = {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u};
    const std::array<float, K> all_active_input = {0.25f, -0.5f, 0.75f, -1.0f,
                                                    1.25f, -1.5f, 1.75f, -2.0f};

    auto check_case = [&](const std::array<float, K>& input, std::span<const uint32_t> selected_columns,
                          const char* label, double* elapsed_ms) {
        std::vector<float> expected(N), actual(N);
        const bool sparse = !selected_columns.empty();
        if (sparse) {
            CHECK(column_grouped_q4_v1_selected_gemv(storage, input, selected_columns, expected, &error));
        } else {
            CHECK(column_grouped_q4_v1_gemv(storage, input, expected, &error));
        }
        CHECK(error == ColumnGroupedQ4Error::None);
        uint32_t selected = 0;
        uint32_t thread_execution_width = 0;
        uint32_t max_threads_per_threadgroup = 0;
        metal_dispatch_metrics_reset();
        CHECK(metal_test_column_grouped_q4_v1(storage, input.data(), actual.data(), sparse,
                                              &selected, elapsed_ms, &thread_execution_width,
                                              &max_threads_per_threadgroup));
        CHECK(selected == (sparse ? selected_columns.size() : K));
        CHECK(thread_execution_width != 0);
        CHECK(max_threads_per_threadgroup >= thread_execution_width);
        CHECK(metal_dispatch_metrics().command_buffers == 1);
        for (uint32_t n = 0; n != N; ++n) {
            CHECK_MSG(almost_equal(actual[n], expected[n], 2.0e-4f, 2.0e-4f),
                      "%s n=%u actual=%g expected=%g", label, n, actual[n], expected[n]);
        }
    };

    double dense_ms = -1.0;
    double sparse_ms = -1.0;
    check_case(sparse_input, {}, "dense", &dense_ms);
    check_case(sparse_input, active_columns, "zero-sparse", &sparse_ms);
    check_case(all_active_input, {}, "all-active-dense", &dense_ms);
    check_case(all_active_input, all_columns, "all-active-sparse", &sparse_ms);
    // The all-active path is a deliberately retained counterexample: running
    // the selector must preserve dense output, but no speedup is asserted.
    fprintf(stderr, "[metal] column-grouped-q4 all-active dense=%.6fms sparse=%.6fms (counterexample)\n",
            dense_ms, sparse_ms);
}

void test_column_grouped_q4_v1_model_shape_timing() {
    // This is only the common decode matrix shape, not a loaded model. The
    // synthetic F32 source keeps the derived-format test artifact-free.
    constexpr uint32_t K = 5120;
    constexpr uint32_t N = 17408;
    std::vector<float> source(static_cast<size_t>(K) * N);
    for (uint32_t n = 0; n != N; ++n) {
        for (uint32_t k = 0; k != K; ++k) {
            source[static_cast<size_t>(n) * K + k] =
                static_cast<float>(static_cast<int>((n * 37u + k * 13u) % 71u) - 35) / 9.0f;
        }
    }
    std::vector<float> input(K);
    std::vector<uint32_t> selected;
    selected.reserve(K / 2u);
    for (uint32_t k = 0; k != K; ++k) {
        if ((k & 1u) != 0u) {
            input[k] = 0.0f;
        } else {
            input[k] = static_cast<float>((k % 8u) + 1u) * 0.125f;
            selected.push_back(k);
        }
    }
    CHECK(selected.size() == K / 2u);

    ColumnGroupedQ4V1Storage storage;
    ColumnGroupedQ4Error error = ColumnGroupedQ4Error::None;
    CHECK(column_grouped_q4_v1_from_f32(source, K, N, &storage, &error));
    std::vector<float> expected_dense(N), expected_sparse(N), actual_dense(N), actual_sparse(N);
    CHECK(column_grouped_q4_v1_gemv(storage, input, expected_dense, &error));
    CHECK(column_grouped_q4_v1_selected_gemv(storage, input, selected, expected_sparse, &error));
    CHECK(error == ColumnGroupedQ4Error::None);

    auto run = [&](bool sparse, std::vector<float>& actual, uint32_t expected_selected,
                   double* elapsed_ms) {
        uint32_t selected_columns = 0;
        uint32_t thread_execution_width = 0;
        uint32_t max_threads_per_threadgroup = 0;
        metal_dispatch_metrics_reset();
        CHECK(metal_test_column_grouped_q4_v1(storage, input.data(), actual.data(), sparse,
                                              &selected_columns, elapsed_ms, &thread_execution_width,
                                              &max_threads_per_threadgroup));
        CHECK(selected_columns == expected_selected);
        CHECK(thread_execution_width != 0);
        CHECK(max_threads_per_threadgroup >= thread_execution_width);
        CHECK(metal_dispatch_metrics().command_buffers == 1);
        const std::vector<float>& expected = sparse ? expected_sparse : expected_dense;
        for (uint32_t n = 0; n != N; ++n) {
            CHECK_MSG(almost_equal(actual[n], expected[n], 4.0e-4f, 4.0e-4f),
                      "model-shape %s n=%u actual=%g expected=%g",
                      sparse ? "sparse" : "dense", n, actual[n], expected[n]);
        }
    };

    double dense_ms = -1.0;
    double sparse_ms = -1.0;
    run(false, actual_dense, K, &dense_ms);
    run(true, actual_sparse, K / 2u, &sparse_ms);
    fprintf(stderr,
            "[metal] column-grouped-q4 synthetic K=%u N=%u dense=%.6fms sparse=%.6fms "
            "(one-sample diagnostic, not a performance claim)\n",
            K, N, dense_ms, sparse_ms);
}

template <typename T>
struct AffineV1AlignedPlane {
    T* data = nullptr;
    size_t count = 0;

    explicit AffineV1AlignedPlane(size_t count_in) : count(count_in) {
        const size_t bytes = std::max<size_t>(128u, ((count * sizeof(T) + 127u) / 128u) * 128u);
        data = static_cast<T*>(std::aligned_alloc(128u, bytes));
        CHECK(data != nullptr);
        if (data) std::memset(data, 0, bytes);
    }

    ~AffineV1AlignedPlane() { std::free(data); }
    AffineV1AlignedPlane(const AffineV1AlignedPlane&) = delete;
    AffineV1AlignedPlane& operator=(const AffineV1AlignedPlane&) = delete;
};

static ColumnGroupedAffineLowBitV1Contract affine_v1_contract(uint8_t bits,
                                                              uint32_t logical_k,
                                                              uint32_t logical_n) {
    ColumnGroupedAffineLowBitV1Contract contract;
    contract.bits = bits;
    contract.logical_k = logical_k;
    contract.logical_n = logical_n;
    contract.group_elements = 256u;
    contract.packed_bytes = (256u * bits + 7u) / 8u;
    contract.scale_bytes_per_group = sizeof(uint16_t);
    contract.bias_bytes_per_group = sizeof(uint16_t);
    contract.plane_alignment = 128u;
    contract.group_count = static_cast<uint64_t>(logical_k) * (logical_n / 256u);
    contract.values_bytes = contract.group_count * contract.packed_bytes;
    return contract;
}

static uint32_t affine_v1_code(uint8_t bits, uint32_t column, uint32_t output_group,
                               uint32_t lane) {
    const uint32_t maximum = (1u << bits) - 1u;
    return (lane * 37u + column * 11u + output_group * 19u + 3u) & maximum;
}

static void test_column_grouped_affine_lowbit_v1_metal() {
    constexpr uint32_t K = 515;
    constexpr uint32_t N = 512;
    const uint32_t output_groups = N / 256u;
    std::vector<float> input(K);
    for (uint32_t column = 0; column != K; ++column)
        input[column] = static_cast<float>(static_cast<int>(column % 23u) - 11) / 17.0f;

    for (uint8_t bits : {uint8_t{2}, uint8_t{3}, uint8_t{4}}) {
        const auto contract = affine_v1_contract(bits, K, N);
        AffineV1AlignedPlane<uint8_t> values(static_cast<size_t>(contract.values_bytes));
        AffineV1AlignedPlane<uint16_t> scales(static_cast<size_t>(contract.group_count));
        AffineV1AlignedPlane<uint16_t> biases(static_cast<size_t>(contract.group_count));
        ColumnGroupedAffineLowBitV1Planes planes{
            values.data, values.count, scales.data, scales.count, biases.data, biases.count};

        for (uint32_t column = 0; column != K; ++column) {
            for (uint32_t output_group = 0; output_group != output_groups; ++output_group) {
                const uint64_t group = static_cast<uint64_t>(column) * output_groups + output_group;
                const float scale = 0.03125f + static_cast<float>((column + output_group) % 7u) / 256.0f;
                const float bias = -0.75f + static_cast<float>((column * 3u + output_group) % 11u) / 32.0f;
                scales.data[group] = fp32_to_fp16(scale);
                biases.data[group] = fp32_to_fp16(bias);
                uint8_t* packed = values.data + group * contract.packed_bytes;
                for (uint32_t lane = 0; lane != 256u; ++lane) {
                    const uint32_t code = affine_v1_code(bits, column, output_group, lane);
                    const uint32_t bit = lane * bits;
                    for (uint8_t part = 0; part != bits; ++part)
                        packed[(bit + part) / 8u] |= static_cast<uint8_t>(
                            ((code >> part) & 1u) << ((bit + part) % 8u));
                }
            }
        }

        std::vector<float> expected(N), actual(N, -1.0f);
        for (uint32_t row = 0; row != N; ++row) {
            const uint32_t output_group = row / 256u;
            const uint32_t lane = row & 255u;
            float sum = 0.0f;
            for (uint32_t column = 0; column != K; ++column) {
                const uint64_t group = static_cast<uint64_t>(column) * output_groups + output_group;
                const float scale = fp16_to_fp32(scales.data[group]);
                const float bias = fp16_to_fp32(biases.data[group]);
                sum += input[column] *
                       (scale * static_cast<float>(affine_v1_code(bits, column, output_group, lane)) + bias);
            }
            expected[row] = sum;
        }

        double gpu_ms = -1.0;
        uint32_t width = 0;
        uint32_t maximum_threads = 0;
        metal_dispatch_metrics_reset();
        CHECK(metal_test_column_grouped_affine_lowbit_v1(
            contract, planes, input.data(), actual.data(), &gpu_ms, &width, &maximum_threads));
        CHECK(gpu_ms > 0.0);
        CHECK(width > 0);
        CHECK(maximum_threads >= width);
        CHECK(metal_dispatch_metrics().command_buffers == 1);
        for (uint32_t row = 0; row != N; ++row)
        CHECK_MSG(almost_equal(actual[row], expected[row], 4.0e-4f, 4.0e-4f),
                      "bits=%u row=%u actual=%g expected=%g", bits, row, actual[row], expected[row]);

        std::vector<float> optimized(N, -1.0f);
        double optimized_gpu_ms = -1.0;
        uint32_t optimized_width = 0;
        uint32_t optimized_maximum_threads = 0;
        metal_dispatch_metrics_reset();
        CHECK(metal_test_column_grouped_affine_lowbit_v1_optimized(
            contract, planes, input.data(), optimized.data(), &optimized_gpu_ms,
            &optimized_width, &optimized_maximum_threads));
        CHECK(optimized_gpu_ms > 0.0);
        CHECK(optimized_width > 0);
        CHECK(optimized_maximum_threads >= optimized_width);
        CHECK(metal_dispatch_metrics().command_buffers == 1);
        for (uint32_t row = 0; row != N; ++row)
            CHECK_MSG(almost_equal(optimized[row], expected[row], 4.0e-4f, 4.0e-4f),
                      "optimized bits=%u row=%u actual=%g expected=%g",
                      bits, row, optimized[row], expected[row]);
        std::fprintf(stderr, "column-grouped-affine bits=%u gpu_ms=%.6f width=%u max_threads=%u cb=1\n",
                     bits, gpu_ms, width, maximum_threads);
        std::fprintf(stderr,
                     "column-grouped-affine-optimized bits=%u gpu_ms=%.6f width=%u "
                     "max_threads=%u cb=1 parity=full\n",
                     bits, optimized_gpu_ms, optimized_width, optimized_maximum_threads);

        auto malformed = contract;
        malformed.bits = 1;
        CHECK(!metal_test_column_grouped_affine_lowbit_v1(
            malformed, planes, input.data(), actual.data(), &gpu_ms, &width, &maximum_threads));
        malformed = contract;
        malformed.packed_bytes += 1u;
        CHECK(!metal_test_column_grouped_affine_lowbit_v1(
            malformed, planes, input.data(), actual.data(), &gpu_ms, &width, &maximum_threads));
        malformed = contract;
        malformed.version = 2u;
        CHECK(!metal_test_column_grouped_affine_lowbit_v1(
            malformed, planes, input.data(), actual.data(), &gpu_ms, &width, &maximum_threads));
        malformed = contract;
        malformed.group_elements = 128u;
        CHECK(!metal_test_column_grouped_affine_lowbit_v1(
            malformed, planes, input.data(), actual.data(), &gpu_ms, &width, &maximum_threads));
        malformed = contract;
        malformed.scale_bytes_per_group = 4u;
        CHECK(!metal_test_column_grouped_affine_lowbit_v1(
            malformed, planes, input.data(), actual.data(), &gpu_ms, &width, &maximum_threads));
        malformed = contract;
        malformed.plane_alignment = 64u;
        CHECK(!metal_test_column_grouped_affine_lowbit_v1(
            malformed, planes, input.data(), actual.data(), &gpu_ms, &width, &maximum_threads));
        malformed = contract;
        malformed.group_count -= 1u;
        CHECK(!metal_test_column_grouped_affine_lowbit_v1(
            malformed, planes, input.data(), actual.data(), &gpu_ms, &width, &maximum_threads));
        auto malformed_planes = planes;
        malformed_planes.values_bytes -= 1u;
        CHECK(!metal_test_column_grouped_affine_lowbit_v1(
            contract, malformed_planes, input.data(), actual.data(), &gpu_ms, &width, &maximum_threads));
        malformed = contract;
        malformed.bits = 1;
        CHECK(!metal_test_column_grouped_affine_lowbit_v1_optimized(
            malformed, planes, input.data(), optimized.data(), &optimized_gpu_ms,
            &optimized_width, &optimized_maximum_threads));
        malformed_planes = planes;
        malformed_planes.scale_count -= 1u;
        CHECK(!metal_test_column_grouped_affine_lowbit_v1(
            contract, malformed_planes, input.data(), actual.data(), &gpu_ms, &width, &maximum_threads));
        malformed_planes = planes;
        malformed_planes.scales = nullptr;
        CHECK(!metal_test_column_grouped_affine_lowbit_v1(
            contract, malformed_planes, input.data(), actual.data(), &gpu_ms, &width, &maximum_threads));
        malformed_planes = planes;
        malformed_planes.values = values.data + 1;
        CHECK(!metal_test_column_grouped_affine_lowbit_v1(
            contract, malformed_planes, input.data(), actual.data(), &gpu_ms, &width, &maximum_threads));
        malformed_planes = planes;
        malformed_planes.biases = reinterpret_cast<uint16_t*>(values.data);
        CHECK(!metal_test_column_grouped_affine_lowbit_v1(
            contract, malformed_planes, input.data(), actual.data(), &gpu_ms, &width, &maximum_threads));
        CHECK(metal_dispatch_metrics().command_buffers == 1);
    }
    std::fprintf(stderr, "column-grouped-affine-lowbit K=%u N=%u q2/q3/q4 parity cb=1\n", K, N);
}

static void test_column_grouped_affine_lowbit_v1_real_shape() {
    constexpr uint32_t K = 5120;
    constexpr uint32_t N = 17408;
    constexpr uint32_t warmups = 5;
    constexpr uint32_t sample_count = 7;
    constexpr uint32_t output_groups = N / 256u;
    constexpr std::array<uint32_t, 8> parity_rows = {
        0u, 1u, 255u, 256u, 257u, 1023u, 16384u, N - 1u};

    std::vector<float> input(K);
    for (uint32_t column = 0; column != K; ++column)
        input[column] = static_cast<float>(static_cast<int>(column % 31u) - 15) / 64.0f;

    for (uint8_t bits : {uint8_t{2}, uint8_t{3}, uint8_t{4}}) {
        const auto contract = affine_v1_contract(bits, K, N);
        AffineV1AlignedPlane<uint8_t> values(static_cast<size_t>(contract.values_bytes));
        AffineV1AlignedPlane<uint16_t> scales(static_cast<size_t>(contract.group_count));
        AffineV1AlignedPlane<uint16_t> biases(static_cast<size_t>(contract.group_count));
        ColumnGroupedAffineLowBitV1Planes planes{
            values.data, values.count, scales.data, scales.count, biases.data, biases.count};

        for (uint32_t column = 0; column != K; ++column) {
            for (uint32_t output_group = 0; output_group != output_groups; ++output_group) {
                const uint64_t group = static_cast<uint64_t>(column) * output_groups + output_group;
                const float scale = 0.03125f + static_cast<float>((column + output_group) % 7u) / 256.0f;
                const float bias = -0.75f + static_cast<float>((column * 3u + output_group) % 11u) / 32.0f;
                scales.data[group] = fp32_to_fp16(scale);
                biases.data[group] = fp32_to_fp16(bias);
                uint8_t* packed = values.data + group * contract.packed_bytes;
                for (uint32_t lane = 0; lane != 256u; ++lane) {
                    const uint32_t code = affine_v1_code(bits, column, output_group, lane);
                    const uint32_t bit = lane * bits;
                    const uint32_t byte = bit >> 3u;
                    const uint32_t shift = bit & 7u;
                    packed[byte] |= static_cast<uint8_t>(code << shift);
                    if (shift + bits > 8u)
                        packed[byte + 1u] |= static_cast<uint8_t>(code >> (8u - shift));
                }
            }
        }

        std::vector<float> actual(N, -1.0f);
        std::array<double, sample_count> samples{};
        uint64_t requested_bytes = 0;
        uint32_t width = 0;
        uint32_t maximum_threads = 0;
        metal_dispatch_metrics_reset();
        const bool executed = metal_test_column_grouped_affine_lowbit_v1_benchmark(
            contract, planes, input.data(), actual.data(), warmups, sample_count,
            samples.data(), &requested_bytes, &width, &maximum_threads);
        CHECK(executed);
        if (!executed) return;
        CHECK(width > 0);
        CHECK(maximum_threads >= width);
        CHECK(metal_dispatch_metrics().command_buffers == sample_count);
        CHECK(requested_bytes == contract.values_bytes + contract.group_count * 4u);
        for (double sample : samples) CHECK(std::isfinite(sample) && sample > 0.0);
        for (float value : actual) CHECK(std::isfinite(value));

        for (uint32_t row : parity_rows) {
            const uint32_t output_group = row / 256u;
            const uint32_t lane = row & 255u;
            float expected = 0.0f;
            for (uint32_t column = 0; column != K; ++column) {
                const uint64_t group = static_cast<uint64_t>(column) * output_groups + output_group;
                const float scale = fp16_to_fp32(scales.data[group]);
                const float bias = fp16_to_fp32(biases.data[group]);
                expected += input[column] *
                            (scale * static_cast<float>(affine_v1_code(bits, column, output_group, lane)) + bias);
            }
            CHECK_MSG(almost_equal(actual[row], expected, 5.0e-3f, 5.0e-3f),
                      "real-shape bits=%u row=%u actual=%g expected=%g",
                      bits, row, actual[row], expected);
        }

        std::array<double, sample_count> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const double median = sorted[sample_count / 2u];
        const double requested_gb_per_second =
            static_cast<double>(requested_bytes) / 1.0e9 / (median / 1000.0);
        const bool meets_gate = requested_gb_per_second >= 200.0;
        std::fprintf(stderr,
                     "column-grouped-affine-real bits=%u K=%u N=%u warmups=%u samples=%u "
                     "median_gpu_ms=%.6f range=%.6f..%.6f requested_bytes=%llu "
                     "requested_GB_s=%.3f width=%u max_threads=%u cb=%llu finite=1 "
                     "parity=sampled gate_200GB_s=%s\n",
                     bits, K, N, warmups, sample_count, median, sorted.front(), sorted.back(),
                     static_cast<unsigned long long>(requested_bytes), requested_gb_per_second,
                     width, maximum_threads,
                     static_cast<unsigned long long>(metal_dispatch_metrics().command_buffers),
                     meets_gate ? "met" : "not-met");

        std::vector<float> optimized(N, -1.0f);
        std::array<double, sample_count> optimized_samples{};
        uint64_t optimized_requested_bytes = 0;
        uint32_t optimized_width = 0;
        uint32_t optimized_maximum_threads = 0;
        metal_dispatch_metrics_reset();
        const bool optimized_executed = metal_test_column_grouped_affine_lowbit_v1_optimized_benchmark(
            contract, planes, input.data(), optimized.data(), warmups, sample_count,
            optimized_samples.data(), &optimized_requested_bytes, &optimized_width,
            &optimized_maximum_threads);
        CHECK(optimized_executed);
        if (!optimized_executed) return;
        CHECK(optimized_width > 0);
        CHECK(optimized_maximum_threads >= optimized_width);
        CHECK(metal_dispatch_metrics().command_buffers == sample_count);
        CHECK(optimized_requested_bytes == contract.values_bytes + contract.group_count * 4u);
        for (double sample : optimized_samples) CHECK(std::isfinite(sample) && sample > 0.0);
        for (float value : optimized) CHECK(std::isfinite(value));

        for (uint32_t row : parity_rows) {
            const uint32_t output_group = row / 256u;
            const uint32_t lane = row & 255u;
            float expected = 0.0f;
            for (uint32_t column = 0; column != K; ++column) {
                const uint64_t group = static_cast<uint64_t>(column) * output_groups + output_group;
                const float scale = fp16_to_fp32(scales.data[group]);
                const float bias = fp16_to_fp32(biases.data[group]);
                expected += input[column] *
                            (scale * static_cast<float>(affine_v1_code(bits, column, output_group, lane)) + bias);
            }
            CHECK_MSG(almost_equal(optimized[row], expected, 5.0e-3f, 5.0e-3f),
                      "optimized real-shape bits=%u row=%u actual=%g expected=%g",
                      bits, row, optimized[row], expected);
        }

        std::array<double, sample_count> optimized_sorted = optimized_samples;
        std::sort(optimized_sorted.begin(), optimized_sorted.end());
        const double optimized_median = optimized_sorted[sample_count / 2u];
        const double optimized_requested_gb_per_second =
            static_cast<double>(optimized_requested_bytes) / 1.0e9 /
            (optimized_median / 1000.0);
        const bool optimized_meets_gate = optimized_requested_gb_per_second >= 200.0;
        std::fprintf(stderr,
                     "column-grouped-affine-optimized-real bits=%u K=%u N=%u warmups=%u samples=%u "
                     "median_gpu_ms=%.6f range=%.6f..%.6f requested_bytes=%llu "
                     "requested_GB_s=%.3f width=%u max_threads=%u cb=%llu finite=1 "
                     "parity=sampled gate_200GB_s=%s\n",
                     bits, K, N, warmups, sample_count, optimized_median,
                     optimized_sorted.front(), optimized_sorted.back(),
                     static_cast<unsigned long long>(optimized_requested_bytes),
                     optimized_requested_gb_per_second, optimized_width,
                     optimized_maximum_threads,
                     static_cast<unsigned long long>(metal_dispatch_metrics().command_buffers),
                     optimized_meets_gate ? "met" : "not-met");

    }
}

static ColumnGroupedAffineUInt2SkipV1Contract uint2_skip_test_contract(uint64_t logical_k,
                                                                        uint64_t logical_n) {
    ColumnGroupedAffineUInt2SkipV1Contract contract;
    contract.logical_k = logical_k;
    contract.logical_n = logical_n;
    contract.group_count = (logical_n / 256u) * logical_k;
    contract.values_bytes = contract.group_count * 64u;
    contract.scale_bytes = contract.group_count * sizeof(uint16_t);
    contract.bias_bytes = contract.group_count * sizeof(uint16_t);
    return contract;
}

static uint32_t uint2_skip_test_code(uint32_t output_block, uint32_t column,
                                     uint32_t lane) {
    return (output_block * 3u + column * 7u + lane * 13u + 1u) & 3u;
}

static void fill_uint2_skip_test_planes_raw(
    const ColumnGroupedAffineUInt2SkipV1Contract& contract,
    uint8_t* values, uint16_t* scales, uint16_t* biases,
    const std::vector<uint8_t>& active_columns, bool poison_inactive) {
    const uint32_t logical_k = static_cast<uint32_t>(contract.logical_k);
    const uint32_t output_blocks = static_cast<uint32_t>(contract.logical_n / 256u);
    for (uint32_t output_block = 0; output_block != output_blocks; ++output_block) {
        for (uint32_t column = 0; column != logical_k; ++column) {
            const uint64_t group = static_cast<uint64_t>(output_block) * logical_k + column;
            uint8_t* packed = values + group * 64u;
            for (uint32_t byte = 0; byte != 64u; ++byte) {
                uint8_t word = 0;
                for (uint32_t part = 0; part != 4u; ++part)
                    word |= static_cast<uint8_t>(uint2_skip_test_code(
                        output_block, column, byte * 4u + part) << (part * 2u));
                packed[byte] = word;
            }
            const bool active = active_columns.empty() || active_columns[column] != 0;
            if (poison_inactive && !active) {
                scales[group] = fp32_to_fp16(std::numeric_limits<float>::quiet_NaN());
                biases[group] = fp32_to_fp16(std::numeric_limits<float>::quiet_NaN());
            } else {
                scales[group] = fp32_to_fp16(
                    0.03125f + static_cast<float>((column + output_block) % 9u) / 64.0f);
                biases[group] = fp32_to_fp16(
                    -0.75f + static_cast<float>((column * 3u + output_block) % 13u) / 32.0f);
            }
        }
    }
}

static void fill_uint2_skip_test_planes(
    const ColumnGroupedAffineUInt2SkipV1Contract& contract,
    AffineV1AlignedPlane<uint8_t>& values,
    AffineV1AlignedPlane<uint16_t>& scales,
    AffineV1AlignedPlane<uint16_t>& biases,
    const std::vector<uint8_t>& active_columns,
    bool poison_inactive) {
    fill_uint2_skip_test_planes_raw(contract, values.data, scales.data, biases.data,
                                    active_columns, poison_inactive);
}

static void uint2_skip_test_expected_raw(
    const ColumnGroupedAffineUInt2SkipV1Contract& contract,
    const uint8_t* values, const uint16_t* scales, const uint16_t* biases,
    const float* input, bool sparse, float* output) {
    const uint32_t logical_k = static_cast<uint32_t>(contract.logical_k);
    const uint32_t logical_n = static_cast<uint32_t>(contract.logical_n);
    for (uint32_t row = 0; row != logical_n; ++row) {
        const uint32_t output_block = row / 256u;
        const uint32_t lane = row & 255u;
        double sum = 0.0;
        for (uint32_t column = 0; column != logical_k; ++column) {
            if (sparse && (!std::isfinite(input[column]) || input[column] == 0.0f)) continue;
            const uint64_t group = static_cast<uint64_t>(output_block) * logical_k + column;
            const uint8_t packed = values[group * 64u + lane / 4u];
            const uint32_t code = (packed >> ((lane & 3u) * 2u)) & 3u;
            const float scale = fp16_to_fp32(scales[group]);
            const float bias = fp16_to_fp32(biases[group]);
            sum += static_cast<double>(input[column]) *
                   (static_cast<double>(scale) * static_cast<double>(code) +
                    static_cast<double>(bias));
        }
        output[row] = static_cast<float>(sum);
    }
}

static void uint2_skip_test_expected(
    const ColumnGroupedAffineUInt2SkipV1Contract& contract,
    const AffineV1AlignedPlane<uint8_t>& values,
    const AffineV1AlignedPlane<uint16_t>& scales,
    const AffineV1AlignedPlane<uint16_t>& biases,
    const float* input, bool sparse, float* output) {
    uint2_skip_test_expected_raw(contract, values.data, scales.data, biases.data,
                                 input, sparse, output);
}

static void test_column_grouped_affine_uint2_skip_v1_metal() {
    constexpr uint32_t K = 517;
    constexpr uint32_t N = 512;
    const auto contract = uint2_skip_test_contract(K, N);
    AffineV1AlignedPlane<uint8_t> values(static_cast<size_t>(contract.values_bytes));
    AffineV1AlignedPlane<uint16_t> scales(static_cast<size_t>(contract.group_count));
    AffineV1AlignedPlane<uint16_t> biases(static_cast<size_t>(contract.group_count));
    ColumnGroupedAffineUInt2SkipV1Planes planes{
        values.data, values.count, scales.data, scales.count, biases.data, biases.count};
    std::vector<uint8_t> all_active(K, 1u);
    fill_uint2_skip_test_planes(contract, values, scales, biases, all_active, false);

    std::vector<float> input(K);
    for (uint32_t column = 0; column != K; ++column)
        input[column] = static_cast<float>(static_cast<int>(column % 19u) - 9) / 13.0f;
    std::vector<float> expected(N), dense(N, -1.0f), sparse(N, -1.0f);
    uint2_skip_test_expected(contract, values, scales, biases, input.data(), false, expected.data());

    auto run = [&](bool sparse_mode, std::vector<float>& actual, const char* label) {
        double gpu_ms = -1.0;
        uint32_t width = 0;
        uint32_t maximum_threads = 0;
        metal_dispatch_metrics_reset();
        CHECK(metal_test_column_grouped_affine_uint2_skip_v1(
            contract, planes, input.data(), actual.data(), sparse_mode, &gpu_ms,
            &width, &maximum_threads));
        CHECK(gpu_ms > 0.0);
        CHECK(width != 0);
        CHECK(maximum_threads >= width);
        CHECK(metal_dispatch_metrics().command_buffers == 1);
        for (uint32_t row = 0; row != N; ++row)
            CHECK_MSG(almost_equal(actual[row], expected[row], 8.0e-4f, 8.0e-4f),
                      "%s row=%u actual=%g expected=%g", label, row, actual[row], expected[row]);
    };
    run(false, dense, "dense");
    run(true, sparse, "all-active sparse");
    for (uint32_t row = 0; row != N; ++row)
        CHECK_MSG(almost_equal(sparse[row], dense[row], 8.0e-4f, 8.0e-4f),
                  "all-active parity row=%u dense=%g sparse=%g", row, dense[row], sparse[row]);

    std::vector<float> none_input(K, 0.0f), none_expected(N), none_actual(N, -1.0f);
    std::vector<uint8_t> no_active(K, 0u);
    fill_uint2_skip_test_planes(contract, values, scales, biases, no_active, true);
    uint2_skip_test_expected(contract, values, scales, biases, none_input.data(), true,
                             none_expected.data());
    auto run_none = [&](bool sparse_mode, std::vector<float>& actual, const char* label) {
        double gpu_ms = -1.0;
        uint32_t width = 0;
        uint32_t maximum_threads = 0;
        metal_dispatch_metrics_reset();
        CHECK(metal_test_column_grouped_affine_uint2_skip_v1(
            contract, planes, none_input.data(), actual.data(), sparse_mode, &gpu_ms,
            &width, &maximum_threads));
        CHECK(gpu_ms > 0.0);
        CHECK(metal_dispatch_metrics().command_buffers == 1);
        for (uint32_t row = 0; row != N; ++row)
            CHECK_MSG(std::isfinite(actual[row]) && actual[row] == 0.0f,
                      "%s poisoned row=%u actual=%g", label, row, actual[row]);
    };
    run_none(true, none_actual, "none sparse");

    std::vector<float> alternating_input(K);
    std::vector<uint8_t> alternating_active(K, 0u);
    for (uint32_t column = 0; column != K; ++column) {
        if (column == 3u) alternating_input[column] = std::numeric_limits<float>::quiet_NaN();
        else if (column == 5u) alternating_input[column] = std::numeric_limits<float>::infinity();
        else if ((column & 1u) != 0u) {
            alternating_input[column] = static_cast<float>((column % 17u) + 1u) / 11.0f;
            alternating_active[column] = 1u;
        } else {
            alternating_input[column] = 0.0f;
        }
    }
    fill_uint2_skip_test_planes(contract, values, scales, biases, alternating_active, true);
    std::vector<float> alternating_expected(N), alternating_actual(N, -1.0f);
    uint2_skip_test_expected(contract, values, scales, biases, alternating_input.data(), true,
                             alternating_expected.data());
    metal_dispatch_metrics_reset();
    double alternating_ms = -1.0;
    uint32_t width = 0, maximum_threads = 0;
    CHECK(metal_test_column_grouped_affine_uint2_skip_v1(
        contract, planes, alternating_input.data(), alternating_actual.data(), true,
        &alternating_ms, &width, &maximum_threads));
    CHECK(alternating_ms > 0.0);
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    for (uint32_t row = 0; row != N; ++row)
        CHECK_MSG(std::isfinite(alternating_actual[row]) &&
                      almost_equal(alternating_actual[row], alternating_expected[row],
                                   8.0e-4f, 8.0e-4f),
                  "alternating row=%u actual=%g expected=%g", row,
                  alternating_actual[row], alternating_expected[row]);

    // Re-run after a different selector population. The count reset in each CB
    // must make stale entries from the first run unobservable.
    fill_uint2_skip_test_planes(contract, values, scales, biases, all_active, false);
    std::vector<float> stale_first(N, -1.0f), stale_second(N, -1.0f);
    metal_dispatch_metrics_reset();
    CHECK(metal_test_column_grouped_affine_uint2_skip_v1(
        contract, planes, input.data(), stale_first.data(), true, &alternating_ms,
        &width, &maximum_threads));
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    fill_uint2_skip_test_planes(contract, values, scales, biases, alternating_active, true);
    metal_dispatch_metrics_reset();
    CHECK(metal_test_column_grouped_affine_uint2_skip_v1(
        contract, planes, alternating_input.data(), stale_second.data(), true, &alternating_ms,
        &width, &maximum_threads));
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    for (uint32_t row = 0; row != N; ++row)
        CHECK_MSG(almost_equal(stale_second[row], alternating_expected[row], 8.0e-4f, 8.0e-4f),
                  "stale-list row=%u actual=%g expected=%g", row, stale_second[row], alternating_expected[row]);

    auto malformed = contract;
    malformed.logical_n = 511u;
    metal_dispatch_metrics_reset();
    CHECK(!metal_test_column_grouped_affine_uint2_skip_v1(
        malformed, planes, input.data(), dense.data(), false, &alternating_ms,
        &width, &maximum_threads));
    CHECK(metal_dispatch_metrics().command_buffers == 0);
    malformed = contract;
    malformed.group_count = std::numeric_limits<uint64_t>::max();
    CHECK(!metal_test_column_grouped_affine_uint2_skip_v1(
        malformed, planes, input.data(), dense.data(), false, &alternating_ms,
        &width, &maximum_threads));
    auto malformed_planes = planes;
    malformed_planes.values_bytes -= 1u;
    CHECK(!metal_test_column_grouped_affine_uint2_skip_v1(
        contract, malformed_planes, input.data(), dense.data(), false, &alternating_ms,
        &width, &maximum_threads));
    malformed_planes = planes;
    malformed_planes.scales = nullptr;
    CHECK(!metal_test_column_grouped_affine_uint2_skip_v1(
        contract, malformed_planes, input.data(), dense.data(), false, &alternating_ms,
        &width, &maximum_threads));
    malformed_planes = planes;
    malformed_planes.values = values.data + 1u;
    CHECK(!metal_test_column_grouped_affine_uint2_skip_v1(
        contract, malformed_planes, input.data(), dense.data(), false, &alternating_ms,
        &width, &maximum_threads));
    CHECK(!metal_test_column_grouped_affine_uint2_skip_v1(
        contract, planes, input.data(), nullptr, false, &alternating_ms,
        &width, &maximum_threads));
    std::fprintf(stderr, "column-grouped-affine-uint2-skip K=%u N=%u dense/all-active/none/alternating "
                         "poison/stale-list/guards cb=1\n", K, N);
}

static void test_column_grouped_affine_uint2_skip_v1_real_shape() {
    constexpr uint32_t K = 5120;
    constexpr uint32_t N = 17408;
    constexpr uint32_t warmups = 1;
    constexpr uint32_t samples = 5;
    const auto contract = uint2_skip_test_contract(K, N);
    AffineV1AlignedPlane<uint8_t> values(static_cast<size_t>(contract.values_bytes));
    AffineV1AlignedPlane<uint16_t> scales(static_cast<size_t>(contract.group_count));
    AffineV1AlignedPlane<uint16_t> biases(static_cast<size_t>(contract.group_count));
    ColumnGroupedAffineUInt2SkipV1Planes planes{
        values.data, values.count, scales.data, scales.count, biases.data, biases.count};
    std::vector<uint8_t> all_active(K, 1u);
    fill_uint2_skip_test_planes(contract, values, scales, biases, all_active, false);
    std::vector<float> all_input(K);
    for (uint32_t column = 0; column != K; ++column) {
        const float magnitude = static_cast<float>((column % 31u) + 1u) / 64.0f;
        all_input[column] = (column & 1u) != 0u ? -magnitude : magnitude;
    }
    std::vector<float> sparse_input = all_input;
    for (uint32_t column = 0; column != K; ++column)
        if (column % 20u >= 11u) sparse_input[column] = 0.0f;
    std::vector<float> dense_all(N, -1.0f), sparse_all(N, -1.0f);
    std::vector<float> dense_55(N, -1.0f), sparse_55(N, -1.0f);

    auto benchmark = [&](const float* input, bool sparse, std::vector<float>& output,
                         const char* label, std::array<double, samples>& timings) {
        uint64_t requested_bytes = 0;
        uint32_t width = 0, maximum_threads = 0;
        metal_dispatch_metrics_reset();
        CHECK(metal_test_column_grouped_affine_uint2_skip_v1_benchmark(
            contract, planes, input, output.data(), sparse, warmups, samples,
            timings.data(), &requested_bytes, &width, &maximum_threads));
        CHECK(width != 0);
        CHECK(maximum_threads >= width);
        uint64_t active_columns = K;
        if (sparse) {
            active_columns = 0;
            for (uint32_t column = 0; column != K; ++column)
                if (std::isfinite(input[column]) && input[column] != 0.0f)
                    ++active_columns;
        }
        const uint64_t expected_bytes = active_columns * (N / 256u) * (64u + 2u + 2u);
        CHECK(requested_bytes == expected_bytes);
        CHECK(metal_dispatch_metrics().command_buffers == samples);
        for (double sample : timings) CHECK(std::isfinite(sample) && sample > 0.0);
        for (float value : output) CHECK(std::isfinite(value));
        auto sorted = timings;
        std::sort(sorted.begin(), sorted.end());
        const double median = sorted[samples / 2u];
        std::fprintf(stderr,
                     "column-grouped-affine-uint2-skip-real mode=%s K=%u N=%u warmups=%u samples=%u "
                     "median_gpu_ms=%.6f range=%.6f..%.6f requested_bytes=%llu "
                     "width=%u max_threads=%u command_buffers=%u encoded_bytes_only=1\n",
                     label, K, N, warmups, samples, median, sorted.front(), sorted.back(),
                     static_cast<unsigned long long>(requested_bytes), width, maximum_threads,
                     samples);
    };
    std::array<double, samples> dense_all_timings{}, sparse_all_timings{};
    std::array<double, samples> dense_55_timings{}, sparse_55_timings{};
    benchmark(all_input.data(), false, dense_all, "dense-all-active", dense_all_timings);
    benchmark(all_input.data(), true, sparse_all, "sparse-all-active", sparse_all_timings);
    for (uint32_t row = 0; row != N; ++row)
        CHECK_MSG(almost_equal(dense_all[row], sparse_all[row], 2.0e-3f, 2.0e-3f),
                  "real all-active row=%u dense=%g sparse=%g", row, dense_all[row], sparse_all[row]);
    benchmark(sparse_input.data(), false, dense_55, "dense-55-active", dense_55_timings);
    benchmark(sparse_input.data(), true, sparse_55, "sparse-55-active", sparse_55_timings);
    for (uint32_t row = 0; row != N; ++row)
        CHECK_MSG(almost_equal(dense_55[row], sparse_55[row], 2.0e-3f, 2.0e-3f),
                  "real 55-active row=%u dense=%g sparse=%g", row, dense_55[row], sparse_55[row]);

    std::fprintf(stderr,
                 "column-grouped-affine-uint2-skip-real K=%u N=%u paired_measurements=4 "
                 "warmup=1 samples=5 requested_bytes=%llu no_dram_claim=1\n",
                 K, N, static_cast<unsigned long long>(contract.values_bytes +
                                                        contract.scale_bytes + contract.bias_bytes));
}

struct ColumnGroupedU2MappedTensor {
    void* mapping = MAP_FAILED;
    size_t mapped_bytes = 0;
    Tensor tensor;
    ColumnGroupedAffineUInt2SkipV1Contract contract;
    ColumnGroupedAffineUInt2SkipV1Planes planes;

    ColumnGroupedU2MappedTensor(uint32_t K, uint32_t N) {
        ColumnGroupedAffineUInt2SkipV1Error error{};
        CHECK(column_grouped_affine_uint2_skip_v1_make_contract(K, N, &contract, &error));
        const long page_value = sysconf(_SC_PAGESIZE);
        CHECK(page_value > 0);
        if (page_value <= 0) return;
        const size_t page = static_cast<size_t>(page_value);
        const auto align128 = [](size_t value) { return (value + 127u) & ~size_t{127u}; };
        const size_t scales_offset = align128(static_cast<size_t>(contract.values_bytes));
        const size_t biases_offset = align128(scales_offset + static_cast<size_t>(contract.scale_bytes));
        const size_t logical_bytes = biases_offset + static_cast<size_t>(contract.bias_bytes);
        mapped_bytes = (logical_bytes + page - 1u) / page * page;
        mapping = mmap(nullptr, mapped_bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANON, -1, 0);
        CHECK(mapping != MAP_FAILED);
        if (mapping == MAP_FAILED) return;
        auto* bytes = static_cast<uint8_t*>(mapping);
        planes.values = bytes;
        planes.values_bytes = static_cast<size_t>(contract.values_bytes);
        planes.scales = reinterpret_cast<uint16_t*>(bytes + scales_offset);
        planes.scale_count = static_cast<size_t>(contract.group_count);
        planes.biases = reinterpret_cast<uint16_t*>(bytes + biases_offset);
        planes.bias_count = static_cast<size_t>(contract.group_count);
        tensor.type = GGMLType::COLUMN_GROUPED_AFFINE_U2_SKIP_256;
        tensor.n_dims = 2;
        tensor.dims[0] = K;
        tensor.dims[1] = N;
        tensor.data = planes.values;
        tensor.scales = reinterpret_cast<const uint8_t*>(planes.scales);
        tensor.biases = reinterpret_cast<const uint8_t*>(planes.biases);
        tensor.data_bytes = contract.values_bytes;
        tensor.scale_bytes = contract.scale_bytes;
        tensor.bias_bytes = contract.bias_bytes;
        tensor.mlx_bits = 2;
        tensor.mlx_group_size = 256;
    }

    ~ColumnGroupedU2MappedTensor() {
        if (mapping != MAP_FAILED) munmap(mapping, mapped_bytes);
    }
};

static void test_column_grouped_affine_uint2_skip_v1_session() {
    constexpr uint32_t H = 512;
    constexpr uint32_t I = 512;
    ColumnGroupedU2MappedTensor gate(H, I), up(H, I), down(I, H);
    ColumnGroupedU2MappedTensor allocation_probe(H, I);
    if (gate.mapping == MAP_FAILED || up.mapping == MAP_FAILED || down.mapping == MAP_FAILED)
        return;
    CHECK(allocation_probe.mapping != MAP_FAILED);
    std::vector<uint8_t> all_active(H, 1u);
    for (ColumnGroupedU2MappedTensor* item : {&gate, &up, &down})
        fill_uint2_skip_test_planes_raw(
            item->contract, const_cast<uint8_t*>(item->planes.values),
            const_cast<uint16_t*>(item->planes.scales),
            const_cast<uint16_t*>(item->planes.biases), all_active, false);

    std::vector<float> input(H);
    for (uint32_t column = 0; column != H; ++column) {
        const float magnitude = static_cast<float>((column % 19u) + 1u) / 127.0f;
        input[column] = column % 20u < 11u
            ? ((column & 1u) ? -magnitude : magnitude) : 0.0f;
    }
    std::vector<float> gate_reference(I), up_reference(I), hidden_reference(I), reference(H);
    uint2_skip_test_expected_raw(gate.contract, gate.planes.values, gate.planes.scales,
                                 gate.planes.biases, input.data(), true,
                                 gate_reference.data());
    uint2_skip_test_expected_raw(up.contract, up.planes.values, up.planes.scales,
                                 up.planes.biases, input.data(), true,
                                 up_reference.data());
    for (uint32_t row = 0; row != I; ++row) {
        const float value = gate_reference[row];
        hidden_reference[row] = value / (1.0f + std::exp(-value)) * up_reference[row];
    }
    uint2_skip_test_expected_raw(down.contract, down.planes.values, down.planes.scales,
                                 down.planes.biases, hidden_reference.data(), true,
                                 reference.data());

    auto session = metal_tok_session_create();
    CHECK(session != nullptr);
    CHECK(session && metal_tok_session_column_grouped_affine_u2_skip_256_ready(*session));
    for (ColumnGroupedU2MappedTensor* item : {&gate, &up, &down}) {
        CHECK(session && metal_tok_session_register_weights(
                             *session, item->mapping, item->mapped_bytes));
        CHECK(session && metal_tok_session_register_column_grouped_affine_u2_skip_256(
            *session, item->tensor));
    }
    if (!session) return;

    // The shader carries group indices in uint32. A valid uint32 dimension
    // pair can still produce more groups than the shader can address; reject
    // it before attempting any Metal allocation or session mutation.
    const MetalResourceSnapshot before_oversized =
        metal_tok_session_resource_snapshot(*session);
    Tensor oversized = gate.tensor;
    oversized.dims[0] = UINT32_MAX;
    oversized.dims[1] = 512u;
    ColumnGroupedAffineUInt2SkipV1Contract oversized_contract;
    ColumnGroupedAffineUInt2SkipV1Error oversized_error{};
    CHECK(column_grouped_affine_uint2_skip_v1_make_contract(
        oversized.dims[0], oversized.dims[1], &oversized_contract, &oversized_error));
    oversized.data_bytes = oversized_contract.values_bytes;
    oversized.scale_bytes = oversized_contract.scale_bytes;
    oversized.bias_bytes = oversized_contract.bias_bytes;
    setenv("LAPLACE_TEST_FORCE_U2_REGISTRATION_ALLOC_FAIL", "1", 1);
    CHECK(!metal_tok_session_register_column_grouped_affine_u2_skip_256(*session, oversized));
    unsetenv("LAPLACE_TEST_FORCE_U2_REGISTRATION_ALLOC_FAIL");
    const MetalResourceSnapshot after_oversized =
        metal_tok_session_resource_snapshot(*session);
    CHECK(after_oversized.session_owned_metadata_bytes ==
          before_oversized.session_owned_metadata_bytes);
    const char* oversized_failure = metal_tok_session_last_failure(*session);
    CHECK(oversized_failure != nullptr);
    CHECK(oversized_failure &&
          std::strstr(oversized_failure, "shader group index") != nullptr);

    if (allocation_probe.mapping != MAP_FAILED) {
        fill_uint2_skip_test_planes_raw(
            allocation_probe.contract, const_cast<uint8_t*>(allocation_probe.planes.values),
            const_cast<uint16_t*>(allocation_probe.planes.scales),
            const_cast<uint16_t*>(allocation_probe.planes.biases), all_active, false);
        auto allocation_session = metal_tok_session_create();
        CHECK(allocation_session != nullptr);
        if (!allocation_session) return;
        const MetalResourceSnapshot before_allocation_failure =
            metal_tok_session_resource_snapshot(*allocation_session);
        setenv("LAPLACE_TEST_FORCE_U2_REGISTRATION_ALLOC_FAIL", "1", 1);
        CHECK(!metal_tok_session_register_column_grouped_affine_u2_skip_256(
            *allocation_session, allocation_probe.tensor));
        unsetenv("LAPLACE_TEST_FORCE_U2_REGISTRATION_ALLOC_FAIL");
        const MetalResourceSnapshot after_allocation_failure =
            metal_tok_session_resource_snapshot(*allocation_session);
        CHECK(after_allocation_failure.session_owned_metadata_bytes ==
              before_allocation_failure.session_owned_metadata_bytes);
        const char* allocation_failure = metal_tok_session_last_failure(*allocation_session);
        CHECK(allocation_failure != nullptr);
        CHECK(allocation_failure &&
              std::strstr(allocation_failure, "registration allocation failed") != nullptr);
        CHECK(metal_tok_session_register_column_grouped_affine_u2_skip_256(
            *allocation_session, allocation_probe.tensor));
    }

    // The registration API must reject incomplete caller-owned spans. A
    // pointer alone cannot prove that the final byte is readable.
    Tensor short_values = gate.tensor;
    short_values.data_bytes -= 1;
    CHECK(!metal_tok_session_register_column_grouped_affine_u2_skip_256(
        *session, short_values));
    Tensor short_scales = gate.tensor;
    short_scales.scale_bytes -= 1;
    CHECK(!metal_tok_session_register_column_grouped_affine_u2_skip_256(
        *session, short_scales));
    Tensor short_biases = gate.tensor;
    short_biases.bias_bytes -= 1;
    CHECK(!metal_tok_session_register_column_grouped_affine_u2_skip_256(
        *session, short_biases));
    metal_tok_session_require_registered_weights(*session, true);
    std::vector<float> actual(H, std::numeric_limits<float>::quiet_NaN());
    double gpu_ms = 0.0;
    metal_dispatch_metrics_reset();
    CHECK(metal_tok_session_probe_ffn_for_testing(
        *session, input.data(), gate.tensor, up.tensor, down.tensor, nullptr, 0,
        actual.data(), H, I, &gpu_ms));
    CHECK(gpu_ms > 0.0);
    const MetalDispatchMetrics dispatch = metal_dispatch_metrics();
    CHECK(dispatch.command_buffers == 1);
    const MetalResourceSnapshot resources = metal_tok_session_resource_snapshot(*session);
    CHECK(resources.implicit_weight_copies == 0);
    CHECK(resources.session_owned_metadata_bytes ==
          3u * gate.contract.group_count * sizeof(uint32_t));
    CHECK(resources.transient_workspace_bytes >=
          static_cast<uint64_t>(H) * sizeof(uint32_t) +
              static_cast<uint64_t>(H) * 16u * sizeof(float));
    const MetalTokMetrics token_metrics = metal_tok_session_metrics(*session);
    CHECK(token_metrics.projection_dispatches == 3);
    CHECK(token_metrics.column_grouped_affine_u2_skip_projection_dispatches == 3);
    uint64_t input_active = 0;
    uint64_t hidden_active = 0;
    for (float value : input) input_active += value != 0.0f;
    for (float value : hidden_reference) hidden_active += value != 0.0f;
    const uint64_t expected_selected_bytes =
        2u * input_active * (I / 256u) * 68u +
        hidden_active * (H / 256u) * 68u;
    CHECK(token_metrics.requested_projection_source_bytes == expected_selected_bytes);
    for (uint32_t row = 0; row != H; ++row)
        CHECK_MSG(std::isfinite(actual[row]) &&
                      almost_equal(actual[row], reference[row], 3.0e-3f, 3.0e-3f),
                  "session column-grouped U2 row=%u actual=%g expected=%g", row,
                  actual[row], reference[row]);

    std::vector<float> nonfinite = input;
    nonfinite[17] = std::numeric_limits<float>::quiet_NaN();
    metal_dispatch_metrics_reset();
    CHECK(!metal_tok_session_probe_ffn_for_testing(
        *session, nonfinite.data(), gate.tensor, up.tensor, down.tensor, nullptr, 0,
        actual.data(), H, I, &gpu_ms));
    const char* failure = metal_tok_session_last_failure(*session);
    CHECK(failure != nullptr);
    CHECK(failure && std::strstr(failure, "non-finite activation") != nullptr);
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    metal_dispatch_metrics_reset();
    CHECK(metal_tok_session_probe_ffn_for_testing(
        *session, input.data(), gate.tensor, up.tensor, down.tensor, nullptr, 0,
        actual.data(), H, I, &gpu_ms));
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    for (uint32_t row = 0; row != H; ++row)
        CHECK(almost_equal(actual[row], reference[row], 3.0e-3f, 3.0e-3f));
    std::fprintf(stderr,
                 "column-grouped-affine-uint2-skip-session H=%u I=%u cb=1 "
                 "implicit_copies=0 metadata=%llu selected_bytes=%llu "
                 "nonfinite_reject=1 retry=1 gpu_ms=%.6f\n",
                 H, I,
                 static_cast<unsigned long long>(resources.session_owned_metadata_bytes),
                 static_cast<unsigned long long>(expected_selected_bytes), gpu_ms);
}

int main(int argc, char** argv) {
    // Keep the bounded physical primitive commands independent from the
    // broad session preflight below; this makes their evidence reproducible.
    if (argc == 2 && (std::strcmp(argv[1], "--trellis") == 0 ||
                      std::strcmp(argv[1], "--trellis-real-shape") == 0)) {
        if (!metal_device_present()) {
            std::fprintf(stderr, "SKIP: no Metal device\n");
            return 0;
        }
        setenv("LAPLACE_MATMUL2D", "1", 1);
        if (!metal_available()) {
            std::fprintf(stderr, "FAIL: Metal device present but kernels did not compile\n");
            return 1;
        }
        if (std::strcmp(argv[1], "--trellis") == 0) {
            test_trellis_decode_dot_small();
            return test_summary("test_metal_trellis");
        }
        test_trellis_decode_dot_real_shape();
        return test_summary("test_metal_trellis_real_shape");
    }
    uint64_t profile_samples = 0;
    CHECK(metal_tok_profile_sample_count_for_testing(1, 64, &profile_samples));
    CHECK(profile_samples == 194);
    CHECK(metal_tok_profile_sample_count_for_testing(2, 64, &profile_samples));
    CHECK(profile_samples == 387);
    CHECK(!metal_tok_profile_sample_count_for_testing(0, 64, &profile_samples));
    CHECK(!metal_tok_profile_sample_count_for_testing(UINT32_MAX, UINT32_MAX, &profile_samples));
    test_named_cpu_output_is_not_overwritten_by_token_graph();
    test_unavailable_executor_cannot_begin_token_graph();
    test_supported_executor_submits_a_complete_fake_graph();
    test_late_token_graph_failure_aborts_without_cpu_continuation();
    if (argc == 2 && std::strcmp(argv[1], "--tensorops-probe") == 0) {
        test_tensorops_descriptor_probe();
        return test_summary("test_metal_tensorops_probe");
    }
    if (!metal_device_present()) {
        fprintf(stderr, "SKIP: no Metal device\n");
        return test_summary("test_metal");
    }
    setenv("LAPLACE_MATMUL2D", "1", 1);
    if (!metal_available()) {
        fprintf(stderr, "FAIL: Metal device present but kernels did not compile\n");
        return 1;
    }
    if (argc == 2 && std::strcmp(argv[1], "--quantized-m1-variant-sweep") == 0) {
        test_quantized_m1_variant_sweep();
        return test_summary("test_metal_quantized_m1_variant_sweep");
    }
    if (argc == 2 && std::strcmp(argv[1], "--quantized-dispatch-overhead") == 0) {
        test_quantized_dispatch_overhead();
        return test_summary("test_metal_quantized_dispatch_overhead");
    }
    if (argc == 2 && std::strcmp(argv[1], "--quantized-q6k-upstream-variant") == 0) {
        test_quantized_q6k_llama_variant();
        return test_summary("test_metal_quantized_q6k_upstream_variant");
    }
    if (argc == 2 && std::strcmp(argv[1], "--rmsnorm-q8-0-weight") == 0) {
        test_rmsnorm_q8_0_weight();
        return test_summary("test_metal_rmsnorm_q8_0_weight");
    }
    if (argc == 2 && std::strcmp(argv[1], "--sampler") == 0) {
        test_sampler();
        return test_summary("test_metal_sampler");
    }
    test_weight_registration_is_transactional();
    {
        const auto first = metal_tok_session_create();
        const auto second = metal_tok_session_create();
        CHECK(first != nullptr);
        CHECK(second != nullptr);
        if (first && second) {
            const uintptr_t first_queue = metal_tok_session_queue_identity_for_testing(*first);
            const uintptr_t second_queue = metal_tok_session_queue_identity_for_testing(*second);
            CHECK(first_queue != 0);
            CHECK(second_queue != 0);
            CHECK(first_queue != second_queue);
        }
    }
    if (argc == 2 && std::strcmp(argv[1], "--session-queue") == 0) {
        return test_summary("test_metal_session_queue");
    }
    if (argc == 2 && std::strcmp(argv[1], "--recurrent-slot-boundary") == 0) {
        metal_dispatch_metrics_reset();
        test_session_recurrent_layer_consumes_quantized_planes();
        return test_summary("test_metal_recurrent_slot_boundary");
    }
    if (argc == 2 && std::strcmp(argv[1], "--router-topk") == 0) {
        test_router_top_k_device();
        return test_summary("test_metal_router_topk");
    }
    if (argc == 2 && std::strcmp(argv[1], "--mixed-attention-geometry") == 0) {
        test_token_session_accepts_mixed_attention_geometry_with_explicit_capacity();
        return test_summary("test_metal_mixed_attention_geometry");
    }
    if (argc == 2 && std::strcmp(argv[1], "--moe-gate-up") == 0) {
        test_moe_q4k_gate_up_gelu();
        return test_summary("test_metal_moe_gate_up");
    }
    if (argc == 2 && std::strcmp(argv[1], "--moe-activation") == 0) {
        test_moe_batched_activation();
        return test_summary("test_metal_moe_activation");
    }
    if (argc == 2 && std::strcmp(argv[1], "--moe-down-reduce") == 0) {
        test_moe_down_reduce();
        return test_summary("test_metal_moe_down_reduce");
    }
    if (argc == 2 && std::strcmp(argv[1], "--iq2-xxs") == 0) {
        test_iq2_xxs_gemv_matches_independent_decoder();
        return test_summary("test_metal_iq2_xxs");
    }
    if (argc == 2 && std::strcmp(argv[1], "--iq2-xxs-real-shape") == 0) {
        test_iq2_xxs_real_shape_parity_and_throughput();
        return test_summary("test_metal_iq2_xxs_real_shape");
    }
    if (argc == 2 && std::strcmp(argv[1], "--iq2-xxs-ffn-shapes") == 0) {
        test_iq2_xxs_registered_ffn_shapes_submit();
        return test_summary("test_metal_iq2_xxs_ffn_shapes");
    }
    if (argc == 2 && std::strcmp(argv[1], "--iq1-s") == 0) {
        test_iq1_s_gemv_matches_independent_decoder();
        return test_summary("test_metal_iq1_s");
    }
    if (argc == 2 && std::strcmp(argv[1], "--iq1-s-real-shape") == 0) {
        test_iq1_s_real_shape_parity_and_throughput();
        return test_summary("test_metal_iq1_s_real_shape");
    }
    if (argc == 2 && std::strcmp(argv[1], "--mpp-int2-m1") == 0) {
        test_mpp_int2_m1_real_shape_admission();
        return test_summary("test_metal_mpp_int2_m1");
    }
    if (argc == 2 && std::strcmp(argv[1], "--affine-u2-256-real-shape") == 0) {
        test_affine_u2_block256_real_shape();
        return test_summary("test_metal_affine_u2_256_real_shape");
    }
    if (argc == 2 && std::strcmp(argv[1], "--affine-u2-256-variants") == 0) {
        test_affine_u2_variant_contract();
        test_affine_u2_variant_finite_edge_domain();
        return test_summary("test_metal_affine_u2_256_variants");
    }
    if (argc == 2 &&
        std::strcmp(argv[1], "--affine-u2-256-variant-matrix-real-shape") == 0) {
        test_affine_u2_variant_matrix_real_shape();
        return test_summary("test_metal_affine_u2_256_variant_matrix_real_shape");
    }
    if (argc == 2 &&
        std::strcmp(argv[1], "--affine-u2-256-masked-paired-real-shape") == 0) {
        test_affine_u2_masked_qdot_paired_real_shape();
        return test_summary("test_metal_affine_u2_256_masked_paired_real_shape");
    }
    if (argc == 2 && std::strcmp(argv[1], "--activation-importance") == 0) {
        test_activation_importance_accumulates_on_device();
        return test_summary("test_metal_activation_importance");
    }
    if (argc == 2 && std::strcmp(argv[1], "--column-grouped-q4") == 0) {
        test_column_grouped_q4_v1_metal();
        return test_summary("test_metal_column_grouped_q4");
    }
    if (argc == 2 && std::strcmp(argv[1], "--column-grouped-q4-real-shape") == 0) {
        test_column_grouped_q4_v1_model_shape_timing();
        return test_summary("test_metal_column_grouped_q4_real_shape");
    }
    if (argc == 2 && std::strcmp(argv[1], "--column-grouped-affine-lowbit") == 0) {
        test_column_grouped_affine_lowbit_v1_metal();
        return test_summary("test_metal_column_grouped_affine_lowbit");
    }
    if (argc == 2 && std::strcmp(argv[1], "--column-grouped-affine-lowbit-real-shape") == 0) {
        test_column_grouped_affine_lowbit_v1_real_shape();
        return test_summary("test_metal_column_grouped_affine_lowbit_real_shape");
    }
    if (argc == 2 && std::strcmp(argv[1], "--column-grouped-affine-uint2-skip") == 0) {
        test_column_grouped_affine_uint2_skip_v1_metal();
        return test_summary("test_metal_column_grouped_affine_uint2_skip");
    }
    if (argc == 2 && std::strcmp(argv[1], "--column-grouped-affine-uint2-skip-real-shape") == 0) {
        test_column_grouped_affine_uint2_skip_v1_real_shape();
        return test_summary("test_metal_column_grouped_affine_uint2_skip_real_shape");
    }
    if (argc == 2 && std::strcmp(argv[1], "--column-grouped-affine-uint2-skip-session") == 0) {
        test_column_grouped_affine_uint2_skip_v1_session();
        return test_summary("test_metal_column_grouped_affine_uint2_skip_session");
    }
    metal_dispatch_metrics_reset();
    test_q6k_embedding_matches_independent_row_decoder();
    test_q4k_embedding_matches_independent_row_decoder();
    test_quantized_repeat_matches_independent_decoder();
    test_gated_delta_net_ping_pong_matches_independent_equation();
    test_session_recurrent_layer_consumes_quantized_planes();
    // Four direct primitive dispatches, three raw state transactions, and two
    // active-token recurrent transactions. Rejected state must not publish.
    CHECK(metal_dispatch_metrics().command_buffers == 11);
    metal_dispatch_metrics_reset();
    fprintf(stderr, "[metal] device available\n");
    test_moe_batched_activation();
    test_moe_down_reduce();
    test_derived_q2k_runs_on_metal();
    test_iq2_xxs_gemv_matches_independent_decoder();
    test_q2k_two_row_pipeline_matches_current();
    test_q2k_streamed_pipeline_matches_current();
    test_sparse_ffn_bank_runs_on_metal();
    test_axis_split_gated_attention_matches_independent_equation();
    test_interleaved_rope_matches_independent_equation();
    test_partial_interleaved_rope_uses_full_frequency_dimension();
    test_multisection_half_split_rope_matches_independent_equation();
    test_fused_query_gate_token_layer_stays_on_one_metal_transaction();
    test_qk_norm_does_not_normalize_value_projection();
    test_token_layer_uses_declared_attention_scale();

    const int K = 2816, N = 512;
    std::vector<float> x(K), w(K*N), y_gpu(N), y_ref(N);
    rng_fill(x.data(), K, 42);
    rng_fill(w.data(), K*N, 123);

    struct TC { GGMLType type; const char* name; void (*pack)(const float*, int, int, uint8_t*); bool is_float; };
    TC tests[] = {
        {GGMLType::F32, "F32", nullptr, true},
        {GGMLType::F16, "F16", pack_f16, false},
        {GGMLType::Q8_0, "Q8_0", pack_q8_0, false},
        {GGMLType::Q5_0, "Q5_0", pack_q5_0, false},
        {GGMLType::Q2_K, "Q2_K", pack_q2_K, false},
        {GGMLType::Q4_K, "Q4_K", pack_q4_K, false},
        {GGMLType::Q6_K, "Q6_K", pack_q6_K, false},
    };

    int pass = 0, fail = 0;
    for (auto& tc : tests) {
        size_t bpb = bytes_per_block(tc.type), epb = elements_per_block(tc.type);
        size_t nblocks = ((size_t)K*N + epb - 1) / epb;
        std::vector<uint8_t> storage(nblocks * bpb, 0);
        Tensor wt; wt.type = tc.type; wt.n_dims = 2; wt.dims[0] = K; wt.dims[1] = N; wt.data = storage.data();
        if (tc.is_float) memcpy(storage.data(), w.data(), (size_t)K*N*4);
        else tc.pack(w.data(), K, N, storage.data());

        setenv("LAPLACE_NOSIMD", "1", 1); setenv("LAPLACE_NOMETAL", "1", 1);
        matmul_rows(x.data(), wt, y_ref.data(), 1, K, N);
        unsetenv("LAPLACE_NOSIMD"); unsetenv("LAPLACE_NOMETAL");

        if (!metal_gemv(x.data(), wt, y_gpu.data(), K, N)) {
            fprintf(stderr, "FAIL %s: not supported\n", tc.name); fail++; continue;
        }

        float max_err = 0, max_val = 0;
        int bad = -1;
        for (int j = 0; j < N; j++) {
            float e = fabsf(y_ref[j]-y_gpu[j]);
            if (e > max_err) { max_err = e; bad = j; }
            max_val = fmaxf(max_val, fabsf(y_ref[j]));
        }
        float rel = max_val > 0 ? max_err/max_val : 0;
        if (!std::isfinite(rel) || rel > 0.01f) {
            fprintf(stderr, "FAIL %s: rel err %.6f (err=%.2f val=%.1f) first=(%.2f,%.2f) badj=%d (%.2f,%.2f)\n",
                    tc.name, rel, max_err, max_val, y_ref[0], y_gpu[0], bad,
                    bad >= 0 ? y_ref[bad] : 0, bad >= 0 ? y_gpu[bad] : 0);
            fail++;
        }
        else { fprintf(stderr, "PASS %s: rel err %.6f\n", tc.name, rel); pass++; }
    }

    {
        const int n = 2112;
        std::vector<float> gate(n), up(n), cpu(n), gpu(n);
        rng_fill(gate.data(), n, 7);
        rng_fill(up.data(), n, 9);
        const float c = 0.7978845608028654f, c2 = 0.044715f;
        auto cpu_glu = [&](float g, float u) {
            if (g > 8.0f) return g * u;
            if (g < -8.0f) return 0.0f;
            float g3 = g * g * g;
            return 0.5f * g * (1.0f + tanhf(c * (g + c2 * g3))) * u;
        };
        for (int i = 0; i < n; i++) cpu[i] = cpu_glu(gate[i], up[i]);
        gate[0] = 40.0f; up[0] = 2.0f; cpu[0] = cpu_glu(40.0f, 2.0f);
        gate[1] = -40.0f; up[1] = 2.0f; cpu[1] = cpu_glu(-40.0f, 2.0f);
        if (!metal_act_glu(gate.data(), up.data(), gpu.data(), n, false)) {
            fprintf(stderr, "FAIL GeGLU: not supported\n");
            fail++;
        } else {
            float max_err = 0, max_val = 0;
            for (int i = 0; i < n; i++) {
                max_err = fmaxf(max_err, fabsf(cpu[i] - gpu[i]));
                max_val = fmaxf(max_val, fabsf(cpu[i]));
            }
            float rel = max_val > 0 ? max_err / max_val : 0;
            if (!std::isfinite(rel) || rel > 0.01f) {
                fprintf(stderr, "FAIL GeGLU: rel err %.6f\n", rel);
                fail++;
            } else {
                fprintf(stderr, "PASS GeGLU: rel err %.6f\n", rel);
                pass++;
            }
        }
    }

    {
        // Shipped attn_decode vs CPU softmax. GQA + SWA. Dims are not a
        // named model: any Hq, Hk, Dh, T that fit the kernel contract.
        const int Hq = 4, Hk = 2, Dh = 32, max_seq = 16, pos = 7, window = 4;
        const float scale = 1.0f / sqrtf((float)Dh);
        const int kn = Hk * Dh, qn = Hq * Dh, T = pos + 1;
        std::vector<float> Q(qn), Kc((size_t)max_seq * kn), Vc((size_t)max_seq * kn);
        std::vector<float> gpu(qn), cpu(qn, 0);
        rng_fill(Q.data(), qn, 11);
        rng_fill(Kc.data(), max_seq * kn, 22);
        rng_fill(Vc.data(), max_seq * kn, 33);
        int start = window > 0 ? std::max(0, pos - window + 1) : 0;
        for (int h = 0; h < Hq; h++) {
            int kvh = h * Hk / Hq;
            float mx = -1e30f;
            std::vector<float> sc(T, 0);
            for (int t = start; t <= pos; t++) {
                const float* q = Q.data() + h * Dh;
                const float* k = Kc.data() + (size_t)t * kn + kvh * Dh;
                float d = 0;
                for (int i = 0; i < Dh; i++) d += q[i] * k[i];
                sc[t] = d * scale;
                mx = fmaxf(mx, sc[t]);
            }
            float sum = 0;
            for (int t = start; t <= pos; t++) {
                sc[t] = expf(sc[t] - mx);
                sum += sc[t];
            }
            float inv = sum > 0 ? 1.0f / sum : 0;
            float* o = cpu.data() + h * Dh;
            for (int t = start; t <= pos; t++) {
                const float* v = Vc.data() + (size_t)t * kn + kvh * Dh;
                float w = sc[t] * inv;
                for (int i = 0; i < Dh; i++) o[i] += w * v[i];
            }
        }
        if (!metal_test_attn(Q.data(), Kc.data(), Vc.data(), gpu.data(),
                             Hq, Hk, Dh, pos, window, max_seq, scale, 0)) {
            fprintf(stderr, "FAIL attn: not supported\n");
            fail++;
        } else {
            float max_err = 0, max_val = 0;
            for (int i = 0; i < qn; i++) {
                max_err = fmaxf(max_err, fabsf(cpu[i] - gpu[i]));
                max_val = fmaxf(max_val, fabsf(cpu[i]));
            }
            float rel = max_val > 0 ? max_err / max_val : 0;
            if (!std::isfinite(rel) || rel > 0.01f) {
                fprintf(stderr, "FAIL attn: rel err %.6f err=%.4f val=%.3f\n",
                        rel, max_err, max_val);
                fail++;
            } else {
                fprintf(stderr, "PASS attn: rel err %.6f\n", rel);
                pass++;
            }
        }
    }

    {
        // Geometry used on the first decode token of this GGUF: GQA, SWA,
        // Dh from topology, T from prompt length. Not a named model.
        const int Hq = 16, Hk = 8, Dh = 256, max_seq = 32, pos = 17, window = 1024;
        const float scale = 1.0f;
        const int kn = Hk * Dh, qn = Hq * Dh;
        std::vector<float> Q(qn), Kc((size_t)max_seq * kn), Vc((size_t)max_seq * kn);
        std::vector<float> gpu(qn), cpu(qn, 0);
        rng_fill(Q.data(), qn, 11);
        rng_fill(Kc.data(), max_seq * kn, 22);
        rng_fill(Vc.data(), max_seq * kn, 33);
        int start = window > 0 ? std::max(0, pos - window + 1) : 0;
        for (int h = 0; h < Hq; h++) {
            int kvh = h * Hk / Hq;
            float mx = -1e30f;
            std::vector<float> sc(pos + 1, 0);
            for (int t = start; t <= pos; t++) {
                const float* q = Q.data() + h * Dh;
                const float* k = Kc.data() + (size_t)t * kn + kvh * Dh;
                float d = 0;
                for (int i = 0; i < Dh; i++) d += q[i] * k[i];
                sc[t] = d * scale;
                mx = fmaxf(mx, sc[t]);
            }
            float sum = 0;
            for (int t = start; t <= pos; t++) {
                sc[t] = expf(sc[t] - mx);
                sum += sc[t];
            }
            float inv = sum > 0 ? 1.0f / sum : 0;
            float* o = cpu.data() + h * Dh;
            for (int t = start; t <= pos; t++) {
                const float* v = Vc.data() + (size_t)t * kn + kvh * Dh;
                float w = sc[t] * inv;
                for (int i = 0; i < Dh; i++) o[i] += w * v[i];
            }
        }
        if (!metal_test_attn(Q.data(), Kc.data(), Vc.data(), gpu.data(),
                             Hq, Hk, Dh, pos, window, max_seq, scale, 0)) {
            fprintf(stderr, "FAIL attn_swa: not supported\n");
            fail++;
        } else {
            float max_err = 0, max_val = 0;
            int nnan = 0;
            for (int i = 0; i < qn; i++) {
                if (!std::isfinite(gpu[i])) nnan++;
                max_err = fmaxf(max_err, fabsf(cpu[i] - gpu[i]));
                max_val = fmaxf(max_val, fabsf(cpu[i]));
            }
            float rel = max_val > 0 ? max_err / max_val : 0;
            if (nnan || !std::isfinite(rel) || rel > 0.01f) {
                fprintf(stderr, "FAIL attn_swa: rel err %.6f nan=%d err=%.4f val=%.3f gpu0=%.4g cpu0=%.4g\n",
                        rel, nnan, max_err, max_val, gpu[0], cpu[0]);
                fail++;
            } else {
                fprintf(stderr, "PASS attn_swa: rel err %.6f\n", rel);
                pass++;
            }
        }
    }

    {
        // Workspace KV is sized by max heads * max dim, not this layer's kn.
        const int Hq = 16, Hk = 8, Dh = 256, max_seq = 32, pos = 17, window = 1024;
        const int stride = 8 * 512;
        const float scale = 1.0f;
        const int qn = Hq * Dh;
        std::vector<float> Q(qn), Kc((size_t)max_seq * stride, 0), Vc((size_t)max_seq * stride, 0);
        std::vector<float> gpu(qn), cpu(qn, 0);
        rng_fill(Q.data(), qn, 11);
        for (int t = 0; t <= pos; t++) {
            rng_fill(Kc.data() + (size_t)t * stride, Hk * Dh, 22 + t);
            rng_fill(Vc.data() + (size_t)t * stride, Hk * Dh, 33 + t);
        }
        int start = window > 0 ? std::max(0, pos - window + 1) : 0;
        for (int h = 0; h < Hq; h++) {
            int kvh = h * Hk / Hq;
            float mx = -1e30f;
            std::vector<float> sc(pos + 1, 0);
            for (int t = start; t <= pos; t++) {
                const float* q = Q.data() + h * Dh;
                const float* k = Kc.data() + (size_t)t * stride + kvh * Dh;
                float d = 0;
                for (int i = 0; i < Dh; i++) d += q[i] * k[i];
                sc[t] = d * scale;
                mx = fmaxf(mx, sc[t]);
            }
            float sum = 0;
            for (int t = start; t <= pos; t++) {
                sc[t] = expf(sc[t] - mx);
                sum += sc[t];
            }
            float inv = sum > 0 ? 1.0f / sum : 0;
            float* o = cpu.data() + h * Dh;
            for (int t = start; t <= pos; t++) {
                const float* v = Vc.data() + (size_t)t * stride + kvh * Dh;
                float w = sc[t] * inv;
                for (int i = 0; i < Dh; i++) o[i] += w * v[i];
            }
        }
        if (!metal_test_attn(Q.data(), Kc.data(), Vc.data(), gpu.data(),
                             Hq, Hk, Dh, pos, window, max_seq, scale, stride)) {
            fprintf(stderr, "FAIL attn_stride: not supported\n");
            fail++;
        } else {
            float max_err = 0, max_val = 0;
            int nnan = 0;
            for (int i = 0; i < qn; i++) {
                if (!std::isfinite(gpu[i])) nnan++;
                max_err = fmaxf(max_err, fabsf(cpu[i] - gpu[i]));
                max_val = fmaxf(max_val, fabsf(cpu[i]));
            }
            float rel = max_val > 0 ? max_err / max_val : 0;
            if (nnan || !std::isfinite(rel) || rel > 0.01f) {
                fprintf(stderr, "FAIL attn_stride: rel %.6f nan=%d gpu0=%.4g cpu0=%.4g\n",
                        rel, nnan, gpu[0], cpu[0]);
                fail++;
            } else {
                fprintf(stderr, "PASS attn_stride: rel err %.6f\n", rel);
                pass++;
            }
        }
    }

    const MetalDispatchMetrics metrics = metal_dispatch_metrics();
    CHECK(metrics.command_buffers > 0);
    CHECK(metrics.cpu_wait_ms > 0.0);
    CHECK(metrics.gpu_time_ms > 0.0);
    fprintf(stderr, "%d passed, %d failed\n", pass, fail);
    return (fail || g_failures) ? 1 : 0;
}
