#include "matmul.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "fp16.h"
#include "kernels.h"
#include "token_graph_backend.h"
#include "threadpool.h"

#if defined(__APPLE__)
namespace Laplace {
extern bool metal_gemv(const float* x, const Tensor& w, float* y, int K, int N);
extern bool metal_gemm(const float* x, const Tensor& w, float* y, int M, int K, int N);
extern void metal_register_weights(const void* base, size_t size);
extern void metal_pack_tensor(const Tensor& w, int K, int N);
extern bool metal_available();
extern bool metal_gemv_begin(const MatmulBatchSpec* specs, int n);
extern void metal_gemv_end();
}
static bool metal_enabled() {
    const char* e = std::getenv("LAPLACE_METAL");
    return e && e[0] == '1';
}
#else
namespace Laplace {
bool metal_gemv(const float*, const Tensor&, float*, int, int) { return false; }
bool metal_gemm(const float*, const Tensor&, float*, int, int, int) { return false; }
void metal_dispatch_metrics_reset() {}
MetalDispatchMetrics metal_dispatch_metrics() { return {}; }
void metal_register_weights(const void*, size_t) {}
void metal_pack_tensor(const Tensor&, int, int) {}
bool metal_tok_begin(int, int, int, int, int, int, int, int, int, int, int) { return false; }
bool metal_tok_active() { return false; }
void metal_tok_upload_x(const float*, int) {}
bool metal_tok_upload_embedding(const Tensor&, uint32_t, int, int) { return false; }
void metal_tok_import_kv(int, int, const float*, const float*, int) {}
bool metal_tok_kv_needs_seed() { return false; }
bool metal_tok_layer(const MetalTokLayer&) { return false; }
bool metal_tok_flush(double* ms_out) { if (ms_out) *ms_out = 0; return true; }
bool metal_tok_end(float*, int) { return false; }
void metal_tok_abort() {}
bool metal_tok_lm(const Tensor&, float*, int, int) { return false; }
bool metal_tok_final(const Tensor&, const Tensor&, float*, int, int, float) {
    return false;
}
void metal_unregister_weights(const void*) {}
bool metal_test_attn(const float*, const float*, const float*, float*,
                     int, int, int, int, int, int, float, int) { return false; }
bool metal_test_q4k_embedding(const Tensor&, uint32_t, float*, int, int) { return false; }
bool metal_test_q6k_embedding(const Tensor&, uint32_t, float*, int, int) { return false; }

namespace {
class NullTokenGraphBackend final : public TokenGraphBackend {
public:
    bool available() const override { return false; }
    bool begin(int, int, int, int, int, int, int, int, int, int, int) override {
        return false;
    }
    bool active() const override { return false; }
    void upload_x(const float*, int) override {}
    bool layer(const MetalTokLayer&) override { return false; }
    bool end(float*, int) override { return false; }
    void abort() override {}
};
}

TokenGraphBackend& metal_token_graph_backend() {
    static NullTokenGraphBackend backend;
    return backend;
}
}
static bool metal_enabled() { return false; }
#endif

namespace Laplace {

using kernels::block_q4_0;
using kernels::block_q5_0;
using kernels::block_q8_0;
using kernels::block_q4_K;
using kernels::block_q6_K;
using kernels::block_q2_K;
using kernels::block_q3_K;
using kernels::block_q5_K;
using kernels::block_iq2_xxs;
using kernels::block_iq1_s;
using kernels::get_scale_min_k4;

namespace {

#include "iq2_xxs_tables.inc"
#include "iq1_s_tables.inc"

std::array<uint8_t, 32> digest_bytes(std::span<const uint8_t> bytes) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    constexpr size_t chunk_size = 1024 * 1024;
    for (size_t offset = 0; offset < bytes.size(); offset += chunk_size) {
        size_t chunk = std::min(chunk_size, bytes.size() - offset);
        CC_SHA256_Update(&context, bytes.data() + offset, static_cast<CC_LONG>(chunk));
    }
    std::array<uint8_t, 32> digest{};
    CC_SHA256_Final(digest.data(), &context);
    return digest;
}

void digest_u16(CC_SHA256_CTX& context, uint16_t value) {
    const uint8_t bytes[2] = {static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8)};
    CC_SHA256_Update(&context, bytes, sizeof(bytes));
}

void digest_u32(CC_SHA256_CTX& context, uint32_t value) {
    const uint8_t bytes[4] = {
        static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8),
        static_cast<uint8_t>(value >> 16), static_cast<uint8_t>(value >> 24),
    };
    CC_SHA256_Update(&context, bytes, sizeof(bytes));
}

void digest_u64(CC_SHA256_CTX& context, uint64_t value) {
    const uint8_t bytes[8] = {
        static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8),
        static_cast<uint8_t>(value >> 16), static_cast<uint8_t>(value >> 24),
        static_cast<uint8_t>(value >> 32), static_cast<uint8_t>(value >> 40),
        static_cast<uint8_t>(value >> 48), static_cast<uint8_t>(value >> 56),
    };
    CC_SHA256_Update(&context, bytes, sizeof(bytes));
}

std::array<uint8_t, 32> derived_storage_digest(const DerivedQ2KStorage& storage) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    static constexpr uint8_t domain[] = {'L','A','P','Q','2','K','0','1'};
    CC_SHA256_Update(&context, domain, sizeof(domain));
    digest_u16(context, storage.contract.version);
    digest_u32(context, static_cast<uint32_t>(storage.contract.source_format));
    digest_u64(context, storage.contract.logical_k);
    digest_u64(context, storage.contract.logical_n);
    digest_u32(context, storage.contract.block_width);
    digest_u32(context, storage.contract.bytes_per_block);
    digest_u32(context, storage.contract.alignment);
    CC_SHA256_Update(&context, storage.source_digest.data(), storage.source_digest.size());
    constexpr size_t chunk_size = 1024 * 1024;
    for (size_t offset = 0; offset < storage.bytes.size(); offset += chunk_size) {
        size_t chunk = std::min(chunk_size, storage.bytes.size() - offset);
        CC_SHA256_Update(&context, storage.bytes.data() + offset, static_cast<CC_LONG>(chunk));
    }
    std::array<uint8_t, 32> digest{};
    CC_SHA256_Final(digest.data(), &context);
    return digest;
}

std::array<uint8_t, 32> derived_iq2_xxs_digest(const DerivedIQ2XXSRecord& record,
                                               std::span<const uint8_t> bytes) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    static constexpr uint8_t domain[] = {'L','A','P','I','Q','2','X','1'};
    CC_SHA256_Update(&context, domain, sizeof(domain));
    digest_u16(context, record.contract.version);
    digest_u32(context, static_cast<uint32_t>(record.contract.source_format));
    digest_u64(context, record.contract.logical_k);
    digest_u64(context, record.contract.logical_n);
    digest_u32(context, record.contract.block_width);
    digest_u32(context, record.contract.bytes_per_block);
    digest_u32(context, record.contract.alignment);
    CC_SHA256_Update(&context, record.source_digest.data(), record.source_digest.size());
    CC_SHA256_Update(&context, record.importance_digest.data(), record.importance_digest.size());
    constexpr size_t chunk_size = 1024 * 1024;
    for (size_t offset = 0; offset < bytes.size(); offset += chunk_size) {
        const size_t chunk = std::min(chunk_size, bytes.size() - offset);
        CC_SHA256_Update(&context, bytes.data() + offset, static_cast<CC_LONG>(chunk));
    }
    std::array<uint8_t, 32> digest{};
    CC_SHA256_Final(digest.data(), &context);
    return digest;
}

std::array<uint8_t, 32> derived_iq1_s_digest(const DerivedIQ1SRecord& record,
                                             std::span<const uint8_t> bytes) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    static constexpr uint8_t domain[] = {'L','A','P','I','Q','1','S','1'};
    CC_SHA256_Update(&context, domain, sizeof(domain));
    digest_u16(context, record.contract.version);
    digest_u32(context, static_cast<uint32_t>(record.contract.source_format));
    digest_u64(context, record.contract.logical_k);
    digest_u64(context, record.contract.logical_n);
    digest_u32(context, record.contract.block_width);
    digest_u32(context, record.contract.bytes_per_block);
    digest_u32(context, record.contract.alignment);
    CC_SHA256_Update(&context, record.source_digest.data(), record.source_digest.size());
    CC_SHA256_Update(&context, record.importance_digest.data(), record.importance_digest.size());
    constexpr size_t chunk_size = 1024 * 1024;
    for (size_t offset = 0; offset < bytes.size(); offset += chunk_size) {
        const size_t chunk = std::min(chunk_size, bytes.size() - offset);
        CC_SHA256_Update(&context, bytes.data() + offset, static_cast<CC_LONG>(chunk));
    }
    std::array<uint8_t, 32> digest{};
    CC_SHA256_Final(digest.data(), &context);
    return digest;
}

std::array<uint8_t, 32> derived_affine_u2_digest(
    const DerivedAffineUInt2Record& record,
    std::span<const uint8_t> packed_weights,
    std::span<const uint16_t> scales,
    std::span<const uint16_t> biases) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    static constexpr uint8_t domain[] = {'L','A','P','A','U','2','0','1'};
    CC_SHA256_Update(&context, domain, sizeof(domain));
    digest_u16(context, record.contract.version);
    digest_u32(context, static_cast<uint32_t>(record.source_format));
    digest_u64(context, record.contract.logical_k);
    digest_u64(context, record.contract.logical_n);
    digest_u32(context, record.contract.block_width);
    digest_u32(context, record.contract.packed_bytes_per_block);
    digest_u32(context, record.contract.scale_bytes_per_block);
    digest_u32(context, record.contract.bias_bytes_per_block);
    digest_u32(context, record.contract.plane_alignment);
    CC_SHA256_Update(&context, record.source_digest.data(), record.source_digest.size());
    CC_SHA256_Update(&context, record.importance_digest.data(),
                     record.importance_digest.size());
    constexpr size_t chunk_size = 1024 * 1024;
    for (size_t offset = 0; offset < packed_weights.size(); offset += chunk_size) {
        const size_t chunk = std::min(chunk_size, packed_weights.size() - offset);
        CC_SHA256_Update(&context, packed_weights.data() + offset,
                         static_cast<CC_LONG>(chunk));
    }
    for (uint16_t value : scales) digest_u16(context, value);
    for (uint16_t value : biases) digest_u16(context, value);
    std::array<uint8_t, 32> digest{};
    CC_SHA256_Final(digest.data(), &context);
    return digest;
}

int nearest_nonnegative(float value) {
    // Match GGML's quantizer exactly: round finite values to nearest, ties to even.
    // Every caller supplies a small non-negative quantization coordinate.
    float biased = value + 12582912.0f;
    int bits;
    std::memcpy(&bits, &biased, sizeof(bits));
    return (bits & 0x007fffff) - 0x00400000;
}

bool quantize_affine_u2_block(const float* values, const float* weights,
                              uint8_t* packed, uint16_t* scale_bits,
                              uint16_t* bias_bits) {
    double weight_sum = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    bool initialized = false;
    for (int lane = 0; lane != 256; ++lane) {
        if (!std::isfinite(values[lane]) || !std::isfinite(weights[lane]) ||
            weights[lane] < 0.0f)
            return false;
        if (weights[lane] == 0.0f) continue;
        weight_sum += weights[lane];
        if (!initialized) {
            minimum = maximum = values[lane];
            initialized = true;
        } else {
            minimum = std::min(minimum, static_cast<double>(values[lane]));
            maximum = std::max(maximum, static_cast<double>(values[lane]));
        }
    }
    if (!initialized || !std::isfinite(weight_sum) || weight_sum <= 0.0)
        return false;

    double scale = (maximum - minimum) / 3.0;
    double bias = minimum;
    std::array<uint8_t, 256> levels{};
    for (int iteration = 0; iteration != 8; ++iteration) {
        double sum_q = 0.0;
        double sum_q2 = 0.0;
        double sum_x = 0.0;
        double sum_qx = 0.0;
        for (int lane = 0; lane != 256; ++lane) {
            const int level = scale > 0.0
                ? std::clamp(nearest_nonnegative(static_cast<float>(
                      (static_cast<double>(values[lane]) - bias) / scale)), 0, 3)
                : 0;
            levels[lane] = static_cast<uint8_t>(level);
            const double weight = weights[lane];
            sum_q += weight * level;
            sum_q2 += weight * level * level;
            sum_x += weight * values[lane];
            sum_qx += weight * level * values[lane];
        }
        const double determinant = weight_sum * sum_q2 - sum_q * sum_q;
        if (determinant > 0.0) {
            const double next_scale =
                (weight_sum * sum_qx - sum_q * sum_x) / determinant;
            const double next_bias = (sum_x - next_scale * sum_q) / weight_sum;
            if (!std::isfinite(next_scale) || !std::isfinite(next_bias) ||
                next_scale < 0.0)
                return false;
            scale = next_scale;
            bias = next_bias;
        } else {
            scale = 0.0;
            bias = sum_x / weight_sum;
        }
    }

    const uint16_t stored_scale = fp32_to_fp16(static_cast<float>(scale));
    const uint16_t stored_bias = fp32_to_fp16(static_cast<float>(bias));
    const float decoded_scale = fp16_to_fp32(stored_scale);
    const float decoded_bias = fp16_to_fp32(stored_bias);
    if (!std::isfinite(decoded_scale) || !std::isfinite(decoded_bias) ||
        decoded_scale < 0.0f)
        return false;
    std::fill(packed, packed + 64, uint8_t{0});
    for (int lane = 0; lane != 256; ++lane) {
        const int level = decoded_scale > 0.0f
            ? std::clamp(nearest_nonnegative(
                  (values[lane] - decoded_bias) / decoded_scale), 0, 3)
            : 0;
        packed[lane / 4] |= static_cast<uint8_t>(level << (2 * (lane & 3)));
    }
    *scale_bits = stored_scale;
    *bias_bits = stored_bias;
    return true;
}

float make_q2_group(const float* values, uint8_t* levels, float* minimum,
                    uint8_t* scratch) {
    float min_value = values[0];
    float max_value = values[0];
    float sum_weight = std::fabs(values[0]);
    float sum_value = sum_weight * values[0];
    for (int i = 1; i < 16; ++i) {
        min_value = std::min(min_value, values[i]);
        max_value = std::max(max_value, values[i]);
        float weight = std::fabs(values[i]);
        sum_weight += weight;
        sum_value += weight * values[i];
    }
    if (min_value > 0.0f) min_value = 0.0f;
    if (max_value == min_value) {
        std::fill(levels, levels + 16, uint8_t{0});
        *minimum = -min_value;
        return 0.0f;
    }

    float inverse_scale = 3.0f / (max_value - min_value);
    float scale = 1.0f / inverse_scale;
    float best_error = 0.0f;
    for (int i = 0; i < 16; ++i) {
        int level = std::clamp(nearest_nonnegative(inverse_scale * (values[i] - min_value)), 0, 3);
        levels[i] = static_cast<uint8_t>(level);
        float difference = scale * level + min_value - values[i];
        best_error += std::fabs(values[i]) * std::fabs(difference);
    }

    for (int step = 0; step <= 15; ++step) {
        inverse_scale = (-0.5f + 0.1f * step + 3.0f) / (max_value - min_value);
        float sum_level = 0.0f;
        float sum_level2 = 0.0f;
        float sum_value_level = 0.0f;
        for (int i = 0; i < 16; ++i) {
            int level = std::clamp(nearest_nonnegative(inverse_scale * (values[i] - min_value)), 0, 3);
            scratch[i] = static_cast<uint8_t>(level);
            float weight = std::fabs(values[i]);
            sum_level += weight * level;
            sum_level2 += weight * level * level;
            sum_value_level += weight * level * values[i];
        }
        float determinant = sum_weight * sum_level2 - sum_level * sum_level;
        if (determinant <= 0.0f) continue;
        float candidate_scale = (sum_weight * sum_value_level - sum_value * sum_level) / determinant;
        float candidate_min = (sum_level2 * sum_value - sum_level * sum_value_level) / determinant;
        if (candidate_min > 0.0f) {
            candidate_min = 0.0f;
            candidate_scale = sum_level2 > 0.0f ? sum_value_level / sum_level2 : 0.0f;
        }
        float error = 0.0f;
        for (int i = 0; i < 16; ++i) {
            float difference = candidate_scale * scratch[i] + candidate_min - values[i];
            error += std::fabs(values[i]) * std::fabs(difference);
        }
        if (error < best_error) {
            std::copy(scratch, scratch + 16, levels);
            best_error = error;
            scale = candidate_scale;
            min_value = candidate_min;
        }
    }
    *minimum = -min_value;
    return scale;
}

bool quantize_q2_k_block(const float* values, kernels::block_q2_K& output) {
    uint8_t levels[256]{};
    uint8_t scratch[16]{};
    float minima[16]{};
    float scales[16]{};
    float max_scale = 0.0f;
    float max_minimum = 0.0f;
    for (int group = 0; group < 16; ++group) {
        for (int lane = 0; lane < 16; ++lane)
            if (!std::isfinite(values[group * 16 + lane])) return false;
        scales[group] = make_q2_group(values + group * 16, levels + group * 16,
                                      &minima[group], scratch);
        max_scale = std::max(max_scale, scales[group]);
        max_minimum = std::max(max_minimum, minima[group]);
    }

    output = {};
    if (max_scale > 0.0f) {
        float inverse = 15.0f / max_scale;
        for (int group = 0; group < 16; ++group)
            output.scales[group] = static_cast<uint8_t>(std::clamp(nearest_nonnegative(inverse * scales[group]), 0, 15));
        output.d = fp32_to_fp16(max_scale / 15.0f);
    }
    if (max_minimum > 0.0f) {
        float inverse = 15.0f / max_minimum;
        for (int group = 0; group < 16; ++group)
            output.scales[group] |= static_cast<uint8_t>(std::clamp(nearest_nonnegative(inverse * minima[group]), 0, 15) << 4);
        output.dmin = fp32_to_fp16(max_minimum / 15.0f);
    }

    float d = fp16_to_fp32(output.d);
    float dmin = fp16_to_fp32(output.dmin);
    for (int group = 0; group < 16; ++group) {
        float group_scale = d * (output.scales[group] & 0x0f);
        if (group_scale == 0.0f) continue;
        float group_minimum = dmin * (output.scales[group] >> 4);
        for (int lane = 0; lane < 16; ++lane) {
            int level = nearest_nonnegative((values[group * 16 + lane] + group_minimum) / group_scale);
            levels[group * 16 + lane] = static_cast<uint8_t>(std::clamp(level, 0, 3));
        }
    }
    for (int half = 0; half < 2; ++half) {
        for (int lane = 0; lane < 32; ++lane) {
            int base = half * 128 + lane;
            output.qs[half * 32 + lane] = levels[base] |
                (levels[base + 32] << 2) | (levels[base + 64] << 4) |
                (levels[base + 96] << 6);
        }
    }
    return true;
}

struct IQ2XXSCodebook {
    struct CandidateRange {
        uint32_t offset = 0;
        uint16_t count = 0;
    };
    std::array<std::array<uint8_t, 8>, 256> levels{};
    std::array<int16_t, 256> sign_index{};
    std::array<CandidateRange, 65536> ranges{};
    std::vector<uint16_t> candidates;

    IQ2XXSCodebook() {
        sign_index.fill(-1);
        for (int index = 0; index < 128; ++index)
            sign_index[kIq2XxsSigns[index]] = static_cast<int16_t>(index);
        for (int grid = 0; grid < 256; ++grid) {
            for (int lane = 0; lane < 8; ++lane) {
                const uint8_t encoded = static_cast<uint8_t>(kIq2XxsGrid[grid] >> (8 * lane));
                levels[grid][lane] = encoded == 8 ? 0 : encoded == 25 ? 1 : 2;
            }
        }
        for (uint32_t pattern = 0; pattern < ranges.size(); ++pattern) {
            std::array<uint8_t, 8> target{};
            bool valid = true;
            for (int lane = 0; lane < 8; ++lane) {
                target[lane] = static_cast<uint8_t>((pattern >> (2 * lane)) & 3u);
                valid &= target[lane] < 3;
            }
            if (!valid) continue;
            std::array<uint16_t, 256> distance{};
            uint16_t first = std::numeric_limits<uint16_t>::max();
            uint16_t second = std::numeric_limits<uint16_t>::max();
            for (int grid = 0; grid < 256; ++grid) {
                int d2 = 0;
                for (int lane = 0; lane < 8; ++lane) {
                    int delta = static_cast<int>(levels[grid][lane]) - target[lane];
                    d2 += delta * delta;
                }
                distance[grid] = static_cast<uint16_t>(d2);
                if (d2 < first) {
                    second = first;
                    first = static_cast<uint16_t>(d2);
                } else if (d2 > first && d2 < second) {
                    second = static_cast<uint16_t>(d2);
                }
            }
            CandidateRange& range = ranges[pattern];
            range.offset = static_cast<uint32_t>(candidates.size());
            for (int grid = 0; grid < 256; ++grid) {
                if (distance[grid] == first || distance[grid] == second)
                    candidates.push_back(static_cast<uint16_t>(grid));
            }
            range.count = static_cast<uint16_t>(candidates.size() - range.offset);
        }
    }
};

const IQ2XXSCodebook& iq2_xxs_codebook() {
    static const IQ2XXSCodebook codebook;
    return codebook;
}

float make_iq2_positive_quants(const float* values, const float* weights, uint8_t* levels) {
    float maximum = 0.0f;
    for (int i = 0; i < 32; ++i) maximum = std::max(maximum, values[i]);
    if (maximum < 1e-15f) {
        std::fill(levels, levels + 32, uint8_t{0});
        return 0.0f;
    }
    float inverse_scale = 4.0f / maximum;
    float best_error = std::numeric_limits<float>::max();
    for (int step = -4; step <= 4; ++step) {
        const float candidate_inverse = (4.0f + 0.1f * step) / maximum;
        const float candidate_scale = 1.0f / candidate_inverse;
        float error = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const int level = std::min(4, nearest_nonnegative(candidate_inverse * values[i]));
            const float delta = values[i] - candidate_scale * level;
            error += weights[i] * delta * delta;
        }
        if (error < best_error) {
            best_error = error;
            inverse_scale = candidate_inverse;
        }
    }
    float sum_value_level = 0.0f;
    float sum_level2 = 0.0f;
    for (int i = 0; i < 32; ++i) {
        const int level = std::min(4, nearest_nonnegative(inverse_scale * values[i]));
        levels[i] = static_cast<uint8_t>(level);
        sum_value_level += weights[i] * values[i] * level;
        sum_level2 += weights[i] * level * level;
    }
    for (int iteration = 0; iteration < 5; ++iteration) {
        int changed = 0;
        for (int i = 0; i < 32; ++i) {
            const float slx = sum_value_level - weights[i] * values[i] * levels[i];
            const float sl2 = sum_level2 - weights[i] * levels[i] * levels[i];
            if (slx <= 0.0f || sl2 <= 0.0f) continue;
            const int candidate = std::min(4, nearest_nonnegative(values[i] * sl2 / slx));
            if (candidate == levels[i]) continue;
            const float new_slx = slx + weights[i] * values[i] * candidate;
            const float new_sl2 = sl2 + weights[i] * candidate * candidate;
            if (new_slx * new_slx * sum_level2 > sum_value_level * sum_value_level * new_sl2) {
                levels[i] = static_cast<uint8_t>(candidate);
                sum_value_level = new_slx;
                sum_level2 = new_sl2;
                ++changed;
            }
        }
        if (!changed) break;
    }
    return sum_level2 > 0.0f ? sum_value_level / sum_level2 : 0.0f;
}

uint16_t iq2_pattern(const uint8_t* levels) {
    uint16_t pattern = 0;
    for (int lane = 0; lane < 8; ++lane) pattern |= static_cast<uint16_t>(levels[lane] << (2 * lane));
    return pattern;
}

uint8_t choose_iq2_grid(uint16_t pattern, const float* values, const float* weights,
                        float scale, uint8_t* selected_levels) {
    const IQ2XXSCodebook& codebook = iq2_xxs_codebook();
    const auto range = codebook.ranges[pattern];
    float best_error = std::numeric_limits<float>::max();
    uint16_t best_grid = 0;
    for (uint16_t candidate = 0; candidate < range.count; ++candidate) {
        const uint16_t grid = codebook.candidates[range.offset + candidate];
        float error = 0.0f;
        for (int lane = 0; lane < 8; ++lane) {
            const float q = static_cast<float>(2 * codebook.levels[grid][lane] + 1);
            const float delta = scale * q - values[lane];
            error += weights[lane] * delta * delta;
        }
        if (error < best_error) {
            best_error = error;
            best_grid = grid;
        }
    }
    std::copy(codebook.levels[best_grid].begin(), codebook.levels[best_grid].end(),
              selected_levels);
    return static_cast<uint8_t>(best_grid);
}

bool quantize_iq2_xxs_block(const float* values, const float* importance,
                            kernels::block_iq2_xxs& output) {
    output = {};
    std::array<float, 8> scales{};
    std::array<uint32_t, 16> packed{};
    float sigma2 = 0.0f;
    for (int i = 0; i < 256; ++i) {
        if (!std::isfinite(values[i])) return false;
        sigma2 += values[i] * values[i];
    }
    sigma2 /= 256.0f;
    float maximum_scale = 0.0f;
    const IQ2XXSCodebook& codebook = iq2_xxs_codebook();
    for (int group32 = 0; group32 < 8; ++group32) {
        const float* source = values + group32 * 32;
        std::array<float, 32> magnitudes{};
        std::array<float, 32> weights{};
        std::array<float, 32> neighbor_weights{};
        std::array<uint8_t, 32> levels{};
        std::array<uint8_t, 4> signs{};
        for (int lane = 0; lane < 32; ++lane)
            weights[lane] = importance[group32 * 32 + lane] *
                            std::sqrt(sigma2 + source[lane] * source[lane]);
        for (int lane = 0; lane < 32; ++lane)
            neighbor_weights[lane] = std::sqrt(weights[lane]);
        for (int subgroup = 0; subgroup < 4; ++subgroup) {
            int negative_count = 0;
            uint8_t sign_pattern = 0;
            for (int lane = 0; lane < 8; ++lane) {
                const int index = subgroup * 8 + lane;
                magnitudes[index] = std::fabs(source[index]);
                if (source[index] < 0.0f) {
                    sign_pattern |= static_cast<uint8_t>(1u << lane);
                    ++negative_count;
                }
            }
            if (negative_count & 1) {
                int least = 0;
                float least_cost = weights[subgroup * 8] * source[subgroup * 8] * source[subgroup * 8];
                for (int lane = 1; lane < 8; ++lane) {
                    const int index = subgroup * 8 + lane;
                    const float cost = weights[index] * source[index] * source[index];
                    if (cost < least_cost) {
                        least = lane;
                        least_cost = cost;
                    }
                }
                magnitudes[subgroup * 8 + least] = -magnitudes[subgroup * 8 + least];
                sign_pattern ^= static_cast<uint8_t>(1u << least);
            }
            const int16_t sign_index = codebook.sign_index[sign_pattern];
            if (sign_index < 0) return false;
            signs[subgroup] = static_cast<uint8_t>(sign_index);
        }

        float scale = make_iq2_positive_quants(magnitudes.data(), weights.data(), levels.data());
        const float effective_maximum = scale * 3.0f;
        if (effective_maximum <= 0.0f) continue;
        float best_score = 0.0f;
        std::array<uint8_t, 32> best_levels{};
        for (int step = -6; step <= 6; ++step) {
            const float inverse = (5.0f + 0.1f * step) / effective_maximum;
            const float candidate_scale = 1.0f / inverse;
            std::array<uint8_t, 32> candidate_levels{};
            for (int subgroup = 0; subgroup < 4; ++subgroup) {
                std::array<uint8_t, 8> approximate{};
                for (int lane = 0; lane < 8; ++lane) {
                    const float coordinate = 0.5f * (inverse * magnitudes[subgroup * 8 + lane] - 1.0f);
                    approximate[lane] = static_cast<uint8_t>(std::clamp(nearest_nonnegative(coordinate), 0, 2));
                }
                choose_iq2_grid(iq2_pattern(approximate.data()), magnitudes.data() + subgroup * 8,
                                neighbor_weights.data() + subgroup * 8, candidate_scale,
                                candidate_levels.data() + subgroup * 8);
            }
            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int lane = 0; lane < 32; ++lane) {
                const float q = static_cast<float>(2 * candidate_levels[lane] + 1);
                sumqx += weights[lane] * magnitudes[lane] * q;
                sumq2 += weights[lane] * q * q;
            }
            if (sumq2 > 0.0f && sumqx * sumqx > best_score * sumq2) {
                scale = sumqx / sumq2;
                best_score = scale * sumqx;
                best_levels = candidate_levels;
            }
        }

        std::array<uint8_t, 4> grids{};
        if (scale > 0.0f) {
            const float inverse = 1.0f / scale;
            for (int subgroup = 0; subgroup < 4; ++subgroup) {
                std::array<uint8_t, 8> approximate{};
                for (int lane = 0; lane < 8; ++lane) {
                    const float coordinate = 0.5f * (inverse * magnitudes[subgroup * 8 + lane] - 1.0f);
                    approximate[lane] = static_cast<uint8_t>(std::clamp(nearest_nonnegative(coordinate), 0, 2));
                }
                grids[subgroup] = choose_iq2_grid(
                    iq2_pattern(approximate.data()), magnitudes.data() + subgroup * 8,
                    neighbor_weights.data() + subgroup * 8, scale,
                    best_levels.data() + subgroup * 8);
            }
            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int lane = 0; lane < 32; ++lane) {
                const float q = static_cast<float>(2 * best_levels[lane] + 1);
                sumqx += weights[lane] * magnitudes[lane] * q;
                sumq2 += weights[lane] * q * q;
            }
            if (sumq2 > 0.0f) scale = sumqx / sumq2;
        }
        if (scale < 0.0f) {
            scale = -scale;
            for (uint8_t& sign : signs) {
                const uint8_t flipped = static_cast<uint8_t>(~kIq2XxsSigns[sign]);
                sign = static_cast<uint8_t>(codebook.sign_index[flipped]);
            }
        }
        uint32_t grid_word = 0;
        uint32_t sign_word = 0;
        for (int subgroup = 0; subgroup < 4; ++subgroup) {
            grid_word |= static_cast<uint32_t>(grids[subgroup]) << (8 * subgroup);
            sign_word |= static_cast<uint32_t>(signs[subgroup]) << (7 * subgroup);
        }
        packed[2 * group32] = grid_word;
        packed[2 * group32 + 1] = sign_word;
        scales[group32] = scale;
        maximum_scale = std::max(maximum_scale, scale);
    }
    if (maximum_scale == 0.0f) return true;
    const float d = maximum_scale / 31.0f;
    output.d = fp32_to_fp16(d);
    for (int group32 = 0; group32 < 8; ++group32) {
        const int level = std::clamp(nearest_nonnegative(0.5f * (scales[group32] / d - 1.0f)), 0, 15);
        packed[2 * group32 + 1] |= static_cast<uint32_t>(level) << 28;
    }
    std::memcpy(output.qs, packed.data(), sizeof(output.qs));
    return true;
}

struct IQ1SCodebook {
    struct CandidateRange {
        uint32_t offset = 0;
        uint16_t count = 0;
    };
    static constexpr size_t pattern_count = 43691;
    std::array<std::array<uint8_t, 8>, 2048> levels{};
    std::array<int16_t, pattern_count> exact{};
    std::array<CandidateRange, pattern_count> ranges{};
    std::vector<uint16_t> candidates;

    IQ1SCodebook() {
        exact.fill(-1);
        for (uint16_t grid = 0; grid != levels.size(); ++grid) {
            uint16_t pattern = 0;
            for (int lane = 0; lane != 8; ++lane) {
                const int8_t value = static_cast<int8_t>(kIq1SGrid[grid] >> (8 * lane));
                const uint8_t level = static_cast<uint8_t>(value + 1);
                levels[grid][lane] = level;
                pattern |= static_cast<uint16_t>(level << (2 * lane));
            }
            exact[pattern] = static_cast<int16_t>(grid);
        }
        for (uint32_t pattern = 0; pattern != pattern_count; ++pattern) {
            std::array<uint8_t, 8> target{};
            bool valid = true;
            for (int lane = 0; lane != 8; ++lane) {
                target[lane] = static_cast<uint8_t>((pattern >> (2 * lane)) & 3u);
                valid &= target[lane] < 3;
            }
            if (!valid || exact[pattern] >= 0) continue;
            std::array<uint16_t, 3> best = {
                std::numeric_limits<uint16_t>::max(),
                std::numeric_limits<uint16_t>::max(),
                std::numeric_limits<uint16_t>::max()};
            for (const auto& grid : levels) {
                uint16_t distance = 0;
                for (int lane = 0; lane != 8; ++lane) {
                    const int delta = static_cast<int>(grid[lane]) - target[lane];
                    distance = static_cast<uint16_t>(distance + 4 * delta * delta);
                }
                for (size_t slot = 0; slot != best.size(); ++slot) {
                    if (distance == best[slot]) break;
                    if (distance < best[slot]) {
                        for (size_t move = best.size() - 1; move != slot; --move)
                            best[move] = best[move - 1];
                        best[slot] = distance;
                        break;
                    }
                }
            }
            CandidateRange& range = ranges[pattern];
            range.offset = static_cast<uint32_t>(candidates.size());
            for (const uint16_t wanted_distance : best) {
                for (uint16_t grid = 0; grid != levels.size(); ++grid) {
                    uint16_t distance = 0;
                    for (int lane = 0; lane != 8; ++lane) {
                        const int delta = static_cast<int>(levels[grid][lane]) - target[lane];
                        distance = static_cast<uint16_t>(distance + 4 * delta * delta);
                    }
                    if (distance == wanted_distance) candidates.push_back(grid);
                }
            }
            range.count = static_cast<uint16_t>(candidates.size() - range.offset);
        }
    }
};

const IQ1SCodebook& iq1_s_codebook() {
    static const IQ1SCodebook codebook;
    return codebook;
}

uint16_t iq1_pattern(const int8_t* levels) {
    uint16_t pattern = 0;
    for (int lane = 0; lane != 8; ++lane)
        pattern |= static_cast<uint16_t>(levels[lane]) << (2 * lane);
    return pattern;
}

uint16_t choose_iq1_grid(uint16_t pattern, const float* values, const float* weights,
                         float scale, const float* quant_values, int8_t* selected_levels) {
    const IQ1SCodebook& codebook = iq1_s_codebook();
    const int16_t exact = codebook.exact[pattern];
    if (exact >= 0) {
        std::copy(codebook.levels[exact].begin(), codebook.levels[exact].end(),
                  selected_levels);
        return static_cast<uint16_t>(exact);
    }
    const auto range = codebook.ranges[pattern];
    float best_error = std::numeric_limits<float>::max();
    uint16_t best_grid = 0;
    for (uint16_t candidate = 0; candidate != range.count; ++candidate) {
        const uint16_t grid = codebook.candidates[range.offset + candidate];
        float error = 0.0f;
        for (int lane = 0; lane != 8; ++lane) {
            const float delta = scale * quant_values[codebook.levels[grid][lane]] - values[lane];
            error += weights[lane] * delta * delta;
        }
        if (error < best_error) {
            best_error = error;
            best_grid = grid;
        }
    }
    std::copy(codebook.levels[best_grid].begin(), codebook.levels[best_grid].end(),
              selected_levels);
    return best_grid;
}

struct IQ1SortPair {
    float value;
    int index;
};

int compare_iq1_pairs(const void* left, const void* right) {
    const float a = static_cast<const IQ1SortPair*>(left)->value;
    const float b = static_cast<const IQ1SortPair*>(right)->value;
    return a < b ? -1 : a > b ? 1 : 0;
}

bool quantize_iq1_s_block(const float* values, const float* importance,
                          kernels::block_iq1_s& output) {
    output = {};
    constexpr float positive_values[3] = {-0.875f, 0.125f, 1.125f};
    constexpr float negative_values[3] = {-1.125f, -0.125f, 0.875f};
    std::array<float, 8> scales{};
    std::array<int8_t, 8> shifts{};
    float sum_squares = 0.0f;
    for (int lane = 0; lane != 256; ++lane) {
        if (!std::isfinite(values[lane])) return false;
        sum_squares += values[lane] * values[lane];
    }
    const float sigma2 = 2.0f * sum_squares / 256.0f;
    float maximum_scale = 0.0f;
    const IQ1SCodebook& codebook = iq1_s_codebook();
    for (int block32 = 0; block32 != 8; ++block32) {
        const float* source = values + block32 * 32;
        const float* source_importance = importance + block32 * 32;
        std::array<float, 32> weights{};
        std::array<IQ1SortPair, 32> sorted{};
        float maximum = 0.0f;
        for (int lane = 0; lane != 32; ++lane) {
            weights[lane] = source_importance[lane] *
                            std::sqrt(sigma2 + source[lane] * source[lane]);
            maximum = std::max(maximum, std::fabs(source[lane]));
            sorted[lane] = {source[lane], lane};
        }
        if (maximum < 1e-12f) {
            shifts[block32] = 1;
            continue;
        }
        std::qsort(sorted.data(), sorted.size(), sizeof(sorted[0]), compare_iq1_pairs);
        std::array<float, 33> cumulative_value{};
        std::array<float, 33> cumulative_weight{};
        for (int lane = 0; lane != 32; ++lane) {
            const int index = sorted[lane].index;
            cumulative_value[lane + 1] = cumulative_value[lane] + weights[index] * source[index];
            cumulative_weight[lane + 1] = cumulative_weight[lane] + weights[index];
        }
        float best_score = -std::numeric_limits<float>::max();
        float scale = maximum;
        int boundary1 = -1;
        int boundary2 = -1;
        int shift = 0;
        for (int first = 0; first <= 32; ++first) {
            for (int second = first; second <= 32; ++second) {
                for (int sign = 0; sign != 2; ++sign) {
                    const float* q = sign == 0 ? positive_values : negative_values;
                    const float sumqx =
                        cumulative_value[first] * q[0] +
                        (cumulative_value[second] - cumulative_value[first]) * q[1] +
                        (cumulative_value[32] - cumulative_value[second]) * q[2];
                    const float sumq2 =
                        cumulative_weight[first] * q[0] * q[0] +
                        (cumulative_weight[second] - cumulative_weight[first]) * q[1] * q[1] +
                        (cumulative_weight[32] - cumulative_weight[second]) * q[2] * q[2];
                    if (sumq2 > 0.0f && sumqx * sumqx > best_score * sumq2) {
                        scale = sumqx / sumq2;
                        best_score = scale * sumqx;
                        boundary1 = first;
                        boundary2 = second;
                        shift = sign == 0 ? 1 : -1;
                    }
                }
            }
        }
        std::array<int8_t, 32> levels{};
        if (boundary1 < 0 || boundary2 < 0 || shift == 0) {
            levels.fill(1);
            shifts[block32] = 1;
            continue;
        }
        for (int lane = 0; lane != boundary1; ++lane) levels[sorted[lane].index] = 0;
        for (int lane = boundary1; lane != boundary2; ++lane) levels[sorted[lane].index] = 1;
        for (int lane = boundary2; lane != 32; ++lane) levels[sorted[lane].index] = 2;
        if (scale < 0.0f) {
            for (int8_t& level : levels) level = static_cast<int8_t>(2 - level);
            scale = -scale;
            shift = -shift;
        }
        const float* quant_values = shift == 1 ? positive_values : negative_values;
        std::array<uint16_t, 4> grids{};
        bool off_grid = false;
        for (int group = 0; group != 4; ++group) {
            const uint16_t pattern = iq1_pattern(levels.data() + group * 8);
            off_grid |= codebook.exact[pattern] < 0;
            grids[group] = choose_iq1_grid(pattern, source + group * 8,
                                           weights.data() + group * 8, scale,
                                           quant_values, levels.data() + group * 8);
        }
        if (off_grid) {
            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int lane = 0; lane != 32; ++lane) {
                const float q = quant_values[levels[lane]];
                sumqx += weights[lane] * q * source[lane];
                sumq2 += weights[lane] * q * q;
            }
            if (sumqx > 0.0f && sumq2 > 0.0f) scale = sumqx / sumq2;
        }
        uint16_t high = 0;
        for (int group = 0; group != 4; ++group) {
            output.qs[4 * block32 + group] = static_cast<uint8_t>(grids[group]);
            high |= static_cast<uint16_t>((grids[group] >> 8) << (3 * group));
        }
        output.qh[block32] = high;
        scales[block32] = scale;
        shifts[block32] = static_cast<int8_t>(shift);
        maximum_scale = std::max(maximum_scale, scale);
    }
    if (maximum_scale == 0.0f) return true;
    const float d = maximum_scale / 15.0f;
    output.d = fp32_to_fp16(d * 1.125f);
    const float inverse = 1.0f / d;
    for (int block32 = 0; block32 != 8; ++block32) {
        int level = nearest_nonnegative(0.5f * (inverse * scales[block32] - 1.0f));
        level = std::clamp(level, 0, 7);
        if (shifts[block32] == -1) level |= 8;
        output.qh[block32] |= static_cast<uint16_t>(level << 12);
    }
    return true;
}

// ---------------- runtime kernel dispatch -----------------------------------
// The SIMD back-end (matmul_simd.cpp, compiled with ARMv8.x ISA flags)
// exports a whole-GEMM entry point. LAPLACE_NOSIMD=1 forces the portable
// scalar path below.

kernels::gemm_fn simd_gemm() {
    static const kernels::gemm_fn fn = [] {
        const char* off = std::getenv("LAPLACE_NOSIMD");
        if (off && off[0] == '1') return static_cast<kernels::gemm_fn>(nullptr);
        return kernels::get_simd_gemm();
    }();
    return fn;
}

// ---------------- scalar per-row dot products (portable reference) ----------
// GGUF tensors are row-major with dims[0] innermost: a weight with
// dims = [K, N] is N contiguous rows of K elements. Quantized rows are
// sequences of K/QK blocks. All kernels walk one output row at a time,
// which is both correct and cache-friendly (each row streams sequentially).

float dot_row_f32(const float* x, const uint8_t* row, int K) {
    const float* w = reinterpret_cast<const float*>(row);
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += x[k] * w[k];
    return acc;
}

float dot_row_f16(const float* x, const uint8_t* row, int K) {
    const uint16_t* w = reinterpret_cast<const uint16_t*>(row);
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += x[k] * fp16_to_fp32(w[k]);
    return acc;
}

float dot_row_bf16(const float* x, const uint8_t* row, int K) {
    const uint16_t* w = reinterpret_cast<const uint16_t*>(row);
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += x[k] * bf16_to_fp32(w[k]);
    return acc;
}

float dot_row_q4_0(const float* x, const uint8_t* row, int K) {
    const block_q4_0* w = reinterpret_cast<const block_q4_0*>(row);
    const int n_blocks = K / 32;
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q4_0& blk = w[b];
        const float* xb = x + b * 32;
        float d = fp16_to_fp32(blk.d);
        float sum = 0.0f;
        for (int l = 0; l < 16; l++) {
            sum += xb[l +  0] * static_cast<float>((blk.qs[l] & 0xF) - 8);
            sum += xb[l + 16] * static_cast<float>((blk.qs[l] >>  4) - 8);
        }
        acc += d * sum;
    }
    return acc;
}

float dot_row_q5_0(const float* x, const uint8_t* row, int K) {
    const block_q5_0* w = reinterpret_cast<const block_q5_0*>(row);
    const int n_blocks = K / 32;
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q5_0& blk = w[b];
        const float* xb = x + b * 32;
        float d = fp16_to_fp32(blk.d);
        uint32_t qh = blk.qh;
        float sum = 0.0f;
        for (int l = 0; l < 16; l++) {
            int lo0 = (blk.qs[l] & 0xF) | (((qh >> (l +  0)) & 1) << 4);
            int lo1 = (blk.qs[l] >> 4)   | (((qh >> (l + 16)) & 1) << 4);
            sum += xb[l +  0] * static_cast<float>(lo0 - 16);
            sum += xb[l + 16] * static_cast<float>(lo1 - 16);
        }
        acc += d * sum;
    }
    return acc;
}

float dot_row_q8_0(const float* x, const uint8_t* row, int K) {
    const block_q8_0* w = reinterpret_cast<const block_q8_0*>(row);
    const int n_blocks = K / 32;
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q8_0& blk = w[b];
        const float* xb = x + b * 32;
        float sum = 0.0f;
        for (int l = 0; l < 32; l++) sum += xb[l] * blk.qs[l];
        acc += fp16_to_fp32(blk.d) * sum;
    }
    return acc;
}

float dot_row_q4_k(const float* x, const uint8_t* row, int K) {
    const block_q4_K* w = reinterpret_cast<const block_q4_K*>(row);
    const int n_blocks = K / 256;
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q4_K& blk = w[b];
        const float* xb = x + b * 256;
        float d    = fp16_to_fp32(blk.d);
        float dmin = fp16_to_fp32(blk.dmin);
        const uint8_t* q = blk.qs;
        int is = 0;
        for (int jb = 0; jb < 256; jb += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, blk.scales, &sc, &m);
            float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, blk.scales, &sc, &m);
            float d2 = d * sc, m2 = dmin * m;
            float s1 = 0.0f, s2 = 0.0f, x1 = 0.0f, x2 = 0.0f;
            for (int l = 0; l < 32; l++) {
                s1 += xb[jb + l]      * static_cast<float>(q[l] & 0xF);
                x1 += xb[jb + l];
                s2 += xb[jb + 32 + l] * static_cast<float>(q[l] >>  4);
                x2 += xb[jb + 32 + l];
            }
            acc += d1 * s1 - m1 * x1 + d2 * s2 - m2 * x2;
            q += 32;
            is += 2;
        }
    }
    return acc;
}

float dot_row_q6_k(const float* x, const uint8_t* row, int K) {
    const block_q6_K* w = reinterpret_cast<const block_q6_K*>(row);
    const int n_blocks = K / 256;
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q6_K& blk = w[b];
        float d = fp16_to_fp32(blk.d);
        const uint8_t* ql = blk.ql;
        const uint8_t* qh = blk.qh;
        const int8_t*  sc = blk.scales;
        const float*   xb = x + b * 256;
        for (int n_off = 0; n_off < 256; n_off += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = static_cast<int>((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = static_cast<int>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = static_cast<int>((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = static_cast<int>((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                acc += xb[n_off + l +  0] * (d * sc[is + 0] * q1);
                acc += xb[n_off + l + 32] * (d * sc[is + 2] * q2);
                acc += xb[n_off + l + 64] * (d * sc[is + 4] * q3);
                acc += xb[n_off + l + 96] * (d * sc[is + 6] * q4);
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return acc;
}

float dot_row_q2_k(const float* x, const uint8_t* row, int K) {
    const block_q2_K* w = reinterpret_cast<const block_q2_K*>(row);
    const int n_blocks = K / 256;
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q2_K& blk = w[b];
        const float* xb = x + b * 256;
        float d    = fp16_to_fp32(blk.d);
        float dmin = fp16_to_fp32(blk.dmin);
        const uint8_t* q  = blk.qs;
        const uint8_t* sc = blk.scales;
        const float* xq = xb;
        int is = 0;
        float isum = 0.0f, summs = 0.0f;
        for (int k = 0; k < 2; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float s = sc[is] & 0xF, m = sc[is] >> 4;
                float suml = 0.0f, xsum = 0.0f;
                for (int l = 0; l < 16; l++) { suml += xq[l] * ((q[l] >> shift) & 3); xsum += xq[l]; }
                isum += s * suml; summs += m * xsum;
                xq += 16; is++;
                s = sc[is] & 0xF; m = sc[is] >> 4;
                suml = 0.0f; xsum = 0.0f;
                for (int l = 16; l < 32; l++) { suml += xq[l - 16] * ((q[l] >> shift) & 3); xsum += xq[l - 16]; }
                isum += s * suml; summs += m * xsum;
                xq += 16; is++;
                shift += 2;
            }
            q += 32;
        }
        acc += d * isum - dmin * summs;
    }
    return acc;
}

float dot_row_q3_k(const float* x, const uint8_t* row, int K) {
    const block_q3_K* w = reinterpret_cast<const block_q3_K*>(row);
    const int n_blocks = K / 256;
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q3_K& blk = w[b];
        const float* xb = x + b * 256;
        float d = fp16_to_fp32(blk.d);
        // Unpack 16 6-bit scales from the 12-byte packed scales array.
        uint32_t auxs[4];
        std::memcpy(auxs, blk.scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const uint8_t* scales = reinterpret_cast<const uint8_t*>(auxs);
        const uint8_t* q3 = blk.qs;
        const uint8_t* hm = blk.hmask;
        const float* xq = xb;
        int is = 0;
        uint32_t m = 1;
        for (int j = 0; j < 2; j++) {
            for (int s = 0; s < 4; s++) {
                int shift = s * 2;
                float sum0 = 0.0f, sum1 = 0.0f;
                for (int l = 0; l < 16; l++) {
                    int a = (q3[l] >> shift) & 3;
                    a -= (hm[l] & m) ? 0 : 4;
                    sum0 += xq[l] * a;
                }
                for (int l = 16; l < 32; l++) {
                    int a = (q3[l] >> shift) & 3;
                    a -= (hm[l] & m) ? 0 : 4;
                    sum1 += xq[l] * a;
                }
                acc += d * (scales[is] - 32) * sum0; is++;
                acc += d * (scales[is] - 32) * sum1; is++;
                xq += 32;
                m <<= 1;
            }
            q3 += 32;
        }
    }
    return acc;
}

float dot_row_q5_k(const float* x, const uint8_t* row, int K) {
    const block_q5_K* w = reinterpret_cast<const block_q5_K*>(row);
    const int n_blocks = K / 256;
    float acc = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q5_K& blk = w[b];
        const float* xb = x + b * 256;
        float d    = fp16_to_fp32(blk.d);
        float dmin = fp16_to_fp32(blk.dmin);
        const uint8_t* q = blk.qs;
        int is = 0;
        for (int jb = 0; jb < 256; jb += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, blk.scales, &sc, &m);
            float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, blk.scales, &sc, &m);
            float d2 = d * sc, m2 = dmin * m;
            int bit = 2 * (jb / 64);
            float s1 = 0.0f, s2 = 0.0f, x1 = 0.0f, x2 = 0.0f;
            for (int l = 0; l < 32; l++) {
                int v1 = (q[l] & 0xF) + 16 * ((blk.qh[l] >> bit) & 1);
                int v2 = (q[l] >> 4) + 16 * ((blk.qh[l] >> (bit + 1)) & 1);
                s1 += xb[jb + l]      * v1; x1 += xb[jb + l];
                s2 += xb[jb + 32 + l] * v2; x2 += xb[jb + 32 + l];
            }
            acc += d1 * s1 - m1 * x1 + d2 * s2 - m2 * x2;
            q += 32; is += 2;
        }
    }
    return acc;
}

// Scalar GEMM. The dot kernel is a template parameter so the call is direct
// and inlines into the row loop.
using dot_f_fn = float (*)(const float*, const uint8_t*, int);

template <dot_f_fn DOT>
void gemm_scalar(const float* x, const uint8_t* data, float* y,
                 int M, int K, int N, size_t rb) {
    auto body = [&](int j) {
        const uint8_t* row = data + static_cast<size_t>(j) * rb;
        for (int m = 0; m < M; m++) {
            y[static_cast<size_t>(m) * N + j] = DOT(x + static_cast<size_t>(m) * K, row, K);
        }
    };
    if (M > 1 || N >= 128) {
        ThreadPool::get().parallel_for(N, body);
    } else {
        for (int j = 0; j < N; j++) body(j);
    }
}

bool gemm_fallback(const float* x, const uint8_t* w, GGMLType type,
                   float* y, int M, int K, int N) {
    const size_t rb = static_cast<size_t>(K) / elements_per_block(type) * bytes_per_block(type);
    switch (type) {
        case GGMLType::F32:  gemm_scalar<dot_row_f32 >(x, w, y, M, K, N, rb); return true;
        case GGMLType::F16:  gemm_scalar<dot_row_f16 >(x, w, y, M, K, N, rb); return true;
        case GGMLType::BF16: gemm_scalar<dot_row_bf16>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q4_0: gemm_scalar<dot_row_q4_0>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q5_0: gemm_scalar<dot_row_q5_0>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q8_0: gemm_scalar<dot_row_q8_0>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q4_K: gemm_scalar<dot_row_q4_k>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q6_K: gemm_scalar<dot_row_q6_k>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q2_K: gemm_scalar<dot_row_q2_k>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q3_K: gemm_scalar<dot_row_q3_k>(x, w, y, M, K, N, rb); return true;
        case GGMLType::Q5_K: gemm_scalar<dot_row_q5_k>(x, w, y, M, K, N, rb); return true;
        default: return false;
    }
}

// LAPLACE_PROF=1: accumulate wall time spent inside the matmuls and report at
// process exit (separates matmul cost from everything else).
struct MatmulProf {
    bool   on = std::getenv("LAPLACE_PROF") != nullptr;
    double seconds = 0.0;
    long   calls = 0;
    ~MatmulProf() {
        if (on) fprintf(stderr, "PROF matmul_row: %.3f s over %ld calls\n", seconds, calls);
    }
};
MatmulProf g_prof;

struct ProfScope {
    std::chrono::steady_clock::time_point t0;
    ProfScope() {
        if (g_prof.on) t0 = std::chrono::steady_clock::now();
    }
    ~ProfScope() {
        if (g_prof.on) {
            g_prof.seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            g_prof.calls++;
        }
    }
};

} // namespace

#if defined(LAPLACE_TESTING)
uint8_t iq2_xxs_upstream_neighbor_for_testing(uint16_t pattern, const float* values,
                                              const float* optimization_weights, float scale,
                                              uint8_t* selected_levels) {
    std::array<float, 8> neighbor_weights{};
    for (size_t lane = 0; lane != neighbor_weights.size(); ++lane)
        neighbor_weights[lane] = std::sqrt(optimization_weights[lane]);
    return choose_iq2_grid(pattern, values, neighbor_weights.data(), scale, selected_levels);
}

uint16_t iq1_s_upstream_neighbor_for_testing(uint16_t pattern, const float* values,
                                             const float* weights, float scale,
                                             const float* quant_values,
                                             int8_t* selected_levels) {
    return choose_iq1_grid(pattern, values, weights, scale, quant_values,
                           selected_levels);
}

bool iq1_s_candidate_order_is_upstream_for_testing() {
    const IQ1SCodebook& codebook = iq1_s_codebook();
    for (uint32_t pattern = 0; pattern != codebook.ranges.size(); ++pattern) {
        const auto range = codebook.ranges[pattern];
        if (range.count == 0) continue;
        std::array<uint8_t, 8> target{};
        for (int lane = 0; lane != 8; ++lane)
            target[lane] = static_cast<uint8_t>((pattern >> (2 * lane)) & 3u);
        uint16_t previous_distance = 0;
        uint16_t previous_grid = 0;
        bool first = true;
        for (uint16_t candidate = 0; candidate != range.count; ++candidate) {
            const uint16_t grid = codebook.candidates[range.offset + candidate];
            uint16_t distance = 0;
            for (int lane = 0; lane != 8; ++lane) {
                const int delta = static_cast<int>(codebook.levels[grid][lane]) -
                                  static_cast<int>(target[lane]);
                distance = static_cast<uint16_t>(distance + 4 * delta * delta);
            }
            if (!first && (distance < previous_distance ||
                           (distance == previous_distance && grid <= previous_grid)))
                return false;
            first = false;
            previous_distance = distance;
            previous_grid = grid;
        }
    }
    return true;
}
#endif

bool validate_affine_u2_block256(const DerivedAffineUInt2Storage& storage,
                                 DerivedStorageError* error) {
    auto reject = [error](DerivedStorageError value) {
        if (error) *error = value;
        return false;
    };
    if (error) *error = DerivedStorageError::None;
    const DerivedAffineUInt2Contract expected{
        1, storage.contract.logical_k, storage.contract.logical_n,
        256, 64, 2, 2, 128,
    };
    if (storage.contract != expected)
        return reject(DerivedStorageError::ContractMismatch);
    if (storage.contract.logical_k == 0 || storage.contract.logical_n == 0 ||
        storage.contract.logical_k % 256 != 0)
        return reject(DerivedStorageError::ShapeUnsupported);
    const uint64_t blocks_per_row = storage.contract.logical_k / 256;
    if (storage.contract.logical_n >
        std::numeric_limits<uint64_t>::max() / blocks_per_row)
        return reject(DerivedStorageError::ShapeUnsupported);
    const uint64_t blocks = storage.contract.logical_n * blocks_per_row;
    if (blocks > std::numeric_limits<size_t>::max() / 64 ||
        storage.packed_weights.size() != static_cast<size_t>(blocks * 64) ||
        storage.scales.size() != blocks || storage.biases.size() != blocks)
        return reject(DerivedStorageError::DestinationLengthMismatch);
    for (uint64_t block = 0; block != blocks; ++block) {
        if (!std::isfinite(fp16_to_fp32(storage.scales[block])) ||
            !std::isfinite(fp16_to_fp32(storage.biases[block])))
            return reject(DerivedStorageError::NonFiniteSource);
    }
    return true;
}

bool decode_affine_u2_block256(const DerivedAffineUInt2Storage& storage,
                               std::vector<float>& output,
                               DerivedStorageError* error) {
    if (!validate_affine_u2_block256(storage, error)) return false;
    if (storage.contract.logical_n >
        std::numeric_limits<size_t>::max() / storage.contract.logical_k) {
        if (error) *error = DerivedStorageError::ShapeUnsupported;
        return false;
    }
    const size_t values = static_cast<size_t>(
        storage.contract.logical_n * storage.contract.logical_k);
    output.resize(values);
    for (size_t index = 0; index != values; ++index) {
        const size_t block = index / 256;
        const uint8_t packed = storage.packed_weights[index / 4];
        const uint8_t q = static_cast<uint8_t>((packed >> (2 * (index & 3))) & 3u);
        output[index] = fp16_to_fp32(storage.scales[block]) * static_cast<float>(q) +
                        fp16_to_fp32(storage.biases[block]);
    }
    return true;
}

bool derive_affine_u2_block256_from_gguf(
    GGMLType source_format, std::span<const uint8_t> source,
    uint64_t logical_k, uint64_t logical_n, std::span<const float> importance,
    std::span<uint8_t> packed_weights, std::span<uint16_t> scales,
    std::span<uint16_t> biases, DerivedAffineUInt2Record* record,
    DerivedStorageError* error) {
    auto reject = [error](DerivedStorageError value) {
        if (error) *error = value;
        return false;
    };
    if (error) *error = DerivedStorageError::None;
    if (!record) return reject(DerivedStorageError::ShapeUnsupported);
    uint64_t source_block_bytes = 0;
    if (source_format == GGMLType::Q4_K)
        source_block_bytes = sizeof(kernels::block_q4_K);
    else if (source_format == GGMLType::Q6_K)
        source_block_bytes = sizeof(kernels::block_q6_K);
    else
        return reject(DerivedStorageError::SourceFormatUnsupported);
    if (logical_k == 0 || logical_n == 0 || logical_k % 256 != 0)
        return reject(DerivedStorageError::ShapeUnsupported);
    if (importance.size() != logical_k)
        return reject(DerivedStorageError::ImportanceLengthMismatch);
    for (float value : importance)
        if (!std::isfinite(value) || value < 0.0f)
            return reject(DerivedStorageError::InvalidImportance);
    const uint64_t blocks_per_row = logical_k / 256;
    for (uint64_t block = 0; block != blocks_per_row; ++block) {
        bool any_positive = false;
        for (uint64_t lane = 0; lane != 256; ++lane)
            any_positive = any_positive || importance[block * 256 + lane] > 0.0f;
        if (!any_positive) return reject(DerivedStorageError::InvalidImportance);
    }
    if (logical_n > std::numeric_limits<uint64_t>::max() / blocks_per_row)
        return reject(DerivedStorageError::ShapeUnsupported);
    const uint64_t block_count = logical_n * blocks_per_row;
    if (block_count > std::numeric_limits<size_t>::max() / source_block_bytes ||
        source.size() != static_cast<size_t>(block_count * source_block_bytes))
        return reject(DerivedStorageError::SourceLengthMismatch);
    if (block_count > std::numeric_limits<size_t>::max() / 64 ||
        packed_weights.size() != static_cast<size_t>(block_count * 64) ||
        scales.size() != block_count || biases.size() != block_count)
        return reject(DerivedStorageError::DestinationLengthMismatch);
    constexpr uintptr_t alignment = 128;
    if ((reinterpret_cast<uintptr_t>(packed_weights.data()) & (alignment - 1)) != 0 ||
        (reinterpret_cast<uintptr_t>(scales.data()) & (alignment - 1)) != 0 ||
        (reinterpret_cast<uintptr_t>(biases.data()) & (alignment - 1)) != 0)
        return reject(DerivedStorageError::DestinationAlignmentMismatch);

    DerivedAffineUInt2Record derived;
    derived.contract.logical_k = logical_k;
    derived.contract.logical_n = logical_n;
    derived.source_format = source_format;
    derived.source_digest = digest_bytes(source);
    derived.importance_digest = digest_bytes(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(importance.data()), importance.size_bytes()));

    std::atomic<bool> failed{false};
    for (uint64_t base = 0; base < block_count; ) {
        const int count = static_cast<int>(std::min<uint64_t>(
            block_count - base, static_cast<uint64_t>(INT_MAX)));
        ThreadPool::get().parallel_for(count, [&](int index) {
            if (failed.load(std::memory_order_relaxed)) return;
            const uint64_t block = base + static_cast<uint64_t>(index);
            std::array<float, 256> values{};
            Tensor source_block;
            source_block.type = source_format;
            source_block.n_dims = 1;
            source_block.dims[0] = 256;
            source_block.data = source.data() +
                static_cast<size_t>(block * source_block_bytes);
            dequantize(source_block, values.data(), 256);
            const size_t importance_offset =
                static_cast<size_t>(block % blocks_per_row) * 256;
            if (!quantize_affine_u2_block(
                    values.data(), importance.data() + importance_offset,
                    packed_weights.data() + static_cast<size_t>(block) * 64,
                    scales.data() + block, biases.data() + block))
                failed.store(true, std::memory_order_relaxed);
        });
        if (failed.load(std::memory_order_relaxed))
            return reject(DerivedStorageError::NonFiniteSource);
        base += static_cast<uint64_t>(count);
    }
    derived.storage_digest = derived_affine_u2_digest(
        derived, packed_weights, scales, biases);
    *record = derived;
    return true;
}

bool derive_q2_k_from_gguf(GGMLType source_format, std::span<const uint8_t> source,
                           uint64_t logical_k, uint64_t logical_n,
                           DerivedQ2KStorage* output, DerivedStorageError* error) {
    auto reject = [error](DerivedStorageError value) {
        if (error) *error = value;
        return false;
    };
    if (error) *error = DerivedStorageError::None;
    if (!output) return reject(DerivedStorageError::ShapeUnsupported);

    uint64_t source_block_bytes = 0;
    if (source_format == GGMLType::Q4_K) source_block_bytes = sizeof(kernels::block_q4_K);
    else if (source_format == GGMLType::Q6_K) source_block_bytes = sizeof(kernels::block_q6_K);
    else return reject(DerivedStorageError::SourceFormatUnsupported);

    if (logical_k == 0 || logical_n == 0 || logical_k % 256 != 0)
        return reject(DerivedStorageError::ShapeUnsupported);
    uint64_t blocks_per_row = logical_k / 256;
    if (logical_n > std::numeric_limits<uint64_t>::max() / blocks_per_row)
        return reject(DerivedStorageError::ShapeUnsupported);
    uint64_t block_count = logical_n * blocks_per_row;
    if (block_count > std::numeric_limits<size_t>::max() / source_block_bytes ||
        source.size() != static_cast<size_t>(block_count * source_block_bytes))
        return reject(DerivedStorageError::SourceLengthMismatch);
    if (block_count > std::numeric_limits<size_t>::max() / sizeof(kernels::block_q2_K))
        return reject(DerivedStorageError::ShapeUnsupported);

    DerivedQ2KStorage derived;
    derived.contract.source_format = source_format;
    derived.contract.logical_k = logical_k;
    derived.contract.logical_n = logical_n;
    derived.source_digest = digest_bytes(source);
    derived.bytes.resize(static_cast<size_t>(block_count) * sizeof(kernels::block_q2_K));

    std::array<float, 256> values{};
    for (uint64_t block = 0; block < block_count; ++block) {
        Tensor source_block;
        source_block.type = source_format;
        source_block.n_dims = 1;
        source_block.dims[0] = 256;
        source_block.data = source.data() + static_cast<size_t>(block * source_block_bytes);
        dequantize(source_block, values.data(), 256);
        kernels::block_q2_K packed{};
        if (!quantize_q2_k_block(values.data(), packed))
            return reject(DerivedStorageError::NonFiniteSource);
        std::memcpy(derived.bytes.data() + static_cast<size_t>(block) * sizeof(packed),
                    &packed, sizeof(packed));
    }
    derived.storage_digest = derived_storage_digest(derived);
    *output = std::move(derived);
    return true;
}

bool decode_derived_q2_k(const DerivedQ2KStorage& storage, std::vector<float>& output) {
    const DerivedQ2KContract expected{
        1, storage.contract.source_format, storage.contract.logical_k,
        storage.contract.logical_n, 256, 84, 128,
    };
    if (storage.contract != expected ||
        (storage.contract.source_format != GGMLType::Q4_K &&
         storage.contract.source_format != GGMLType::Q6_K) ||
        storage.contract.logical_k == 0 || storage.contract.logical_n == 0 ||
        storage.contract.logical_k % 256 != 0)
        return false;
    uint64_t blocks_per_row = storage.contract.logical_k / 256;
    if (storage.contract.logical_n > std::numeric_limits<uint64_t>::max() / blocks_per_row)
        return false;
    uint64_t block_count = storage.contract.logical_n * blocks_per_row;
    if (block_count > std::numeric_limits<size_t>::max() / sizeof(kernels::block_q2_K) ||
        storage.bytes.size() != static_cast<size_t>(block_count) * sizeof(kernels::block_q2_K) ||
        derived_storage_digest(storage) != storage.storage_digest)
        return false;
    if (storage.contract.logical_n > std::numeric_limits<size_t>::max() / storage.contract.logical_k)
        return false;
    output.resize(static_cast<size_t>(storage.contract.logical_n * storage.contract.logical_k));
    for (uint64_t block = 0; block < block_count; ++block) {
        Tensor q2_block;
        q2_block.type = GGMLType::Q2_K;
        q2_block.n_dims = 1;
        q2_block.dims[0] = 256;
        q2_block.data = storage.bytes.data() + static_cast<size_t>(block) * sizeof(kernels::block_q2_K);
        dequantize(q2_block, output.data() + static_cast<size_t>(block) * 256, 256);
    }
    return true;
}

bool derive_iq2_xxs_from_gguf(GGMLType source_format, std::span<const uint8_t> source,
                              uint64_t logical_k, uint64_t logical_n,
                              std::span<const float> importance,
                              std::span<uint8_t> destination,
                              DerivedIQ2XXSRecord* record, DerivedStorageError* error) {
    auto reject = [error](DerivedStorageError value) {
        if (error) *error = value;
        return false;
    };
    if (error) *error = DerivedStorageError::None;
    if (!record) return reject(DerivedStorageError::ShapeUnsupported);

    uint64_t source_block_bytes = 0;
    if (source_format == GGMLType::Q4_K) source_block_bytes = sizeof(kernels::block_q4_K);
    else if (source_format == GGMLType::Q6_K) source_block_bytes = sizeof(kernels::block_q6_K);
    else return reject(DerivedStorageError::SourceFormatUnsupported);
    if (logical_k == 0 || logical_n == 0 || logical_k % 256 != 0)
        return reject(DerivedStorageError::ShapeUnsupported);
    if (importance.size() != logical_k)
        return reject(DerivedStorageError::ImportanceLengthMismatch);
    for (float weight : importance) {
        if (!std::isfinite(weight) || weight < 0.0f)
            return reject(DerivedStorageError::InvalidImportance);
    }

    const uint64_t blocks_per_row = logical_k / 256;
    if (logical_n > std::numeric_limits<uint64_t>::max() / blocks_per_row)
        return reject(DerivedStorageError::ShapeUnsupported);
    const uint64_t block_count = logical_n * blocks_per_row;
    if (block_count > std::numeric_limits<size_t>::max() / source_block_bytes ||
        source.size() != static_cast<size_t>(block_count * source_block_bytes))
        return reject(DerivedStorageError::SourceLengthMismatch);
    if (block_count > std::numeric_limits<size_t>::max() / sizeof(kernels::block_iq2_xxs) ||
        destination.size() != static_cast<size_t>(block_count) * sizeof(kernels::block_iq2_xxs))
        return reject(DerivedStorageError::DestinationLengthMismatch);

    DerivedIQ2XXSRecord derived;
    derived.contract.source_format = source_format;
    derived.contract.logical_k = logical_k;
    derived.contract.logical_n = logical_n;
    derived.source_digest = digest_bytes(source);
    const auto* importance_bytes = reinterpret_cast<const uint8_t*>(importance.data());
    derived.importance_digest = digest_bytes(std::span<const uint8_t>(
        importance_bytes, importance.size_bytes()));

    (void)iq2_xxs_codebook();
    std::atomic<bool> failed{false};
    for (uint64_t base = 0; base < block_count; ) {
        const int count = static_cast<int>(std::min<uint64_t>(
            block_count - base, static_cast<uint64_t>(INT_MAX)));
        ThreadPool::get().parallel_for(count, [&](int index) {
            if (failed.load(std::memory_order_relaxed)) return;
            const uint64_t block = base + static_cast<uint64_t>(index);
            std::array<float, 256> values{};
            Tensor source_block;
            source_block.type = source_format;
            source_block.n_dims = 1;
            source_block.dims[0] = 256;
            source_block.data = source.data() + static_cast<size_t>(block * source_block_bytes);
            dequantize(source_block, values.data(), 256);
            kernels::block_iq2_xxs packed{};
            const size_t column_offset = static_cast<size_t>(block % blocks_per_row) * 256;
            if (!quantize_iq2_xxs_block(values.data(), importance.data() + column_offset, packed)) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            std::memcpy(destination.data() + static_cast<size_t>(block) * sizeof(packed),
                        &packed, sizeof(packed));
        });
        if (failed.load(std::memory_order_relaxed))
            return reject(DerivedStorageError::NonFiniteSource);
        base += static_cast<uint64_t>(count);
    }
    derived.storage_digest = derived_iq2_xxs_digest(derived, destination);
    *record = derived;
    return true;
}

bool derive_iq1_s_from_gguf(GGMLType source_format, std::span<const uint8_t> source,
                            uint64_t logical_k, uint64_t logical_n,
                            std::span<const float> importance,
                            std::span<uint8_t> destination,
                            DerivedIQ1SRecord* record, DerivedStorageError* error) {
    auto reject = [error](DerivedStorageError value) {
        if (error) *error = value;
        return false;
    };
    if (error) *error = DerivedStorageError::None;
    if (!record) return reject(DerivedStorageError::ShapeUnsupported);

    uint64_t source_block_bytes = 0;
    if (source_format == GGMLType::Q4_K) source_block_bytes = sizeof(kernels::block_q4_K);
    else if (source_format == GGMLType::Q6_K) source_block_bytes = sizeof(kernels::block_q6_K);
    else return reject(DerivedStorageError::SourceFormatUnsupported);
    if (logical_k == 0 || logical_n == 0 || logical_k % 256 != 0)
        return reject(DerivedStorageError::ShapeUnsupported);
    if (importance.size() != logical_k)
        return reject(DerivedStorageError::ImportanceLengthMismatch);
    for (float weight : importance) {
        if (!std::isfinite(weight) || weight < 0.0f)
            return reject(DerivedStorageError::InvalidImportance);
    }

    const uint64_t blocks_per_row = logical_k / 256;
    if (logical_n > std::numeric_limits<uint64_t>::max() / blocks_per_row)
        return reject(DerivedStorageError::ShapeUnsupported);
    const uint64_t block_count = logical_n * blocks_per_row;
    if (block_count > std::numeric_limits<size_t>::max() / source_block_bytes ||
        source.size() != static_cast<size_t>(block_count * source_block_bytes))
        return reject(DerivedStorageError::SourceLengthMismatch);
    if (block_count > std::numeric_limits<size_t>::max() / sizeof(kernels::block_iq1_s) ||
        destination.size() != static_cast<size_t>(block_count) * sizeof(kernels::block_iq1_s))
        return reject(DerivedStorageError::DestinationLengthMismatch);

    DerivedIQ1SRecord derived;
    derived.contract.source_format = source_format;
    derived.contract.logical_k = logical_k;
    derived.contract.logical_n = logical_n;
    derived.source_digest = digest_bytes(source);
    const auto* importance_bytes = reinterpret_cast<const uint8_t*>(importance.data());
    derived.importance_digest = digest_bytes(std::span<const uint8_t>(
        importance_bytes, importance.size_bytes()));

    (void)iq1_s_codebook();
    std::atomic<bool> failed{false};
    for (uint64_t base = 0; base < block_count; ) {
        const int count = static_cast<int>(std::min<uint64_t>(
            block_count - base, static_cast<uint64_t>(INT_MAX)));
        ThreadPool::get().parallel_for(count, [&](int index) {
            if (failed.load(std::memory_order_relaxed)) return;
            const uint64_t block = base + static_cast<uint64_t>(index);
            std::array<float, 256> values{};
            Tensor source_block;
            source_block.type = source_format;
            source_block.n_dims = 1;
            source_block.dims[0] = 256;
            source_block.data = source.data() + static_cast<size_t>(block * source_block_bytes);
            dequantize(source_block, values.data(), 256);
            kernels::block_iq1_s packed{};
            const size_t column_offset = static_cast<size_t>(block % blocks_per_row) * 256;
            if (!quantize_iq1_s_block(values.data(), importance.data() + column_offset, packed)) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            std::memcpy(destination.data() + static_cast<size_t>(block) * sizeof(packed),
                        &packed, sizeof(packed));
        });
        if (failed.load(std::memory_order_relaxed))
            return reject(DerivedStorageError::NonFiniteSource);
        base += static_cast<uint64_t>(count);
    }
    derived.storage_digest = derived_iq1_s_digest(derived, destination);
    *record = derived;
    return true;
}

// MLX affine quantization dot product.
// w: packed uint32 weights (32/bits elements per word, LSB first)
// scales: per-group scale (fp16), biases: per-group bias (fp16)
static float dot_row_mlx(const float* x, const uint8_t* w,
                         const uint16_t* scales, const uint16_t* biases,
                         int K, int bits, int group_size) {
    int epw = 32 / bits;
    int mask = (1 << bits) - 1;
    int n_groups = K / group_size;
    const uint32_t* qw = reinterpret_cast<const uint32_t*>(w);
    float acc = 0;
    for (int g = 0; g < n_groups; g++) {
        float scale = fp16_to_fp32(scales[g]);
        float bias = fp16_to_fp32(biases[g]);
        for (int i = 0; i < group_size; i++) {
            int k = g * group_size + i;
            int word_idx = k / epw;
            int shift = (k % epw) * bits;
            float q = float((qw[word_idx] >> shift) & mask);
            acc += x[k] * (scale * q + bias);
        }
    }
    return acc;
}

void matmul_rows(const float* x, const Tensor& w, float* y, int M, int K, int N) {
    if (w.type == GGMLType::MLX_AFFINE) {
        int bits = w.mlx_bits;
        int gs = w.mlx_group_size;
        int epw = 32 / bits;
        size_t packed_K = (size_t)(K + epw - 1) / epw;
        size_t rb = packed_K * 4;
        for (int m = 0; m < M; m++) {
            for (int j = 0; j < N; j++) {
                const uint8_t* row = w.data + (size_t)j * rb;
                const uint16_t* sc = reinterpret_cast<const uint16_t*>(w.scales) + (size_t)j * (K / gs);
                const uint16_t* bi = reinterpret_cast<const uint16_t*>(w.biases) + (size_t)j * (K / gs);
                y[(size_t)m * N + j] = dot_row_mlx(x + (size_t)m * K, row, sc, bi, K, bits, gs);
            }
        }
        return;
    }
    ProfScope prof;
    if (kernels::gemm_fn gemm = simd_gemm()) {
        if (gemm(x, w.data, w.type, y, M, K, N)) return;
    }
    if (!gemm_fallback(x, w.data, w.type, y, M, K, N)) {
        fprintf(stderr, "matmul: unsupported weight type %s\n", type_name(w.type));
        std::memset(y, 0, sizeof(float) * static_cast<size_t>(M) * N);
    }
}

void matmul_row(const float* x, const Tensor& w, float* y, int K, int N) {
    matmul_rows(x, w, y, 1, K, N);
}

void matmul_lm_head(const float* x, const Tensor& w, float* y, int M, int K, int N) {
    static const bool lm_cpu = std::getenv("LAPLACE_LM_CPU") != nullptr;
    if (metal_enabled() && !lm_cpu) {
        if (metal_gemm(x, w, y, M, K, N)) return;
    }
    matmul_rows(x, w, y, M, K, N);
}

void matmul_use_gcd(bool on) {
    kernels::set_gcd_gemm(on);
}

bool matmul_gpu_available() {
    return metal_enabled() && metal_available();
}

bool matmul_gpu_batch(const MatmulBatchSpec* specs, int n) {
    if (!metal_enabled() || n <= 0) return false;
    if (!metal_gemv_begin(specs, n)) return false;
    metal_gemv_end();
    return true;
}

// ---------------- Dequantize (embeddings, verification) ---------------------

void dequantize(const Tensor& w, float* dst, int n) {
    switch (w.type) {
        case GGMLType::F32: {
            const float* p = reinterpret_cast<const float*>(w.data);
            std::memcpy(dst, p, sizeof(float) * n);
            return;
        }
        case GGMLType::F16: {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(w.data);
            for (int i = 0; i < n; i++) dst[i] = fp16_to_fp32(p[i]);
            return;
        }
        case GGMLType::BF16: {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(w.data);
            for (int i = 0; i < n; i++) dst[i] = bf16_to_fp32(p[i]);
            return;
        }
        case GGMLType::Q4_0: {
            const block_q4_0* p = reinterpret_cast<const block_q4_0*>(w.data);
            for (int i = 0; i < n; i += 32) {
                float d = fp16_to_fp32(p[i/32].d);
                for (int l = 0; l < 16; l++) {
                    dst[i + l]      = d * ((p[i/32].qs[l] & 0xF) - 8);
                    dst[i + l + 16] = d * ((p[i/32].qs[l] >>  4) - 8);
                }
            }
            return;
        }
        case GGMLType::Q5_0: {
            const block_q5_0* p = reinterpret_cast<const block_q5_0*>(w.data);
            for (int i = 0; i < n; i += 32) {
                float d = fp16_to_fp32(p[i/32].d);
                uint32_t qh = p[i/32].qh;
                for (int l = 0; l < 16; l++) {
                    int lo0 = (p[i/32].qs[l] & 0xF) | (((qh >> (l +  0)) & 1) << 4);
                    int lo1 = (p[i/32].qs[l] >> 4)   | (((qh >> (l + 16)) & 1) << 4);
                    dst[i + l]      = d * (lo0 - 16);
                    dst[i + l + 16] = d * (lo1 - 16);
                }
            }
            return;
        }
        case GGMLType::Q8_0: {
            const block_q8_0* p = reinterpret_cast<const block_q8_0*>(w.data);
            for (int i = 0; i < n; i += 32) {
                float d = fp16_to_fp32(p[i/32].d);
                for (int l = 0; l < 32; l++) dst[i + l] = d * p[i/32].qs[l];
            }
            return;
        }
        case GGMLType::Q4_K: {
            const block_q4_K* p = reinterpret_cast<const block_q4_K*>(w.data);
            for (int i = 0; i < n; i += 256) {
                const block_q4_K& blk = p[i/256];
                float d = fp16_to_fp32(blk.d);
                float dmin = fp16_to_fp32(blk.dmin);
                const uint8_t* q = blk.qs;
                int is = 0;
                for (int jb = 0; jb < 256; jb += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is + 0, blk.scales, &sc, &m);
                    float d1 = d * sc, m1 = dmin * m;
                    get_scale_min_k4(is + 1, blk.scales, &sc, &m);
                    float d2 = d * sc, m2 = dmin * m;
                    for (int l = 0; l < 32; l++) dst[i + jb + l]      = d1 * (q[l] & 0xF) - m1;
                    for (int l = 0; l < 32; l++) dst[i + jb + 32 + l] = d2 * (q[l] >> 4) - m2;
                    q += 32; is += 2;
                }
            }
            return;
        }
        case GGMLType::Q6_K: {
            const block_q6_K* p = reinterpret_cast<const block_q6_K*>(w.data);
            for (int i = 0; i < n; i += 256) {
                const block_q6_K& blk = p[i/256];
                float d = fp16_to_fp32(blk.d);
                const uint8_t* ql = blk.ql;
                const uint8_t* qh = blk.qh;
                const int8_t*  sc = blk.scales;
                float* y = dst + i;
                for (int n_off = 0; n_off < 256; n_off += 128) {
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
            return;
        }
        case GGMLType::Q2_K: {
            const block_q2_K* p = reinterpret_cast<const block_q2_K*>(w.data);
            for (int i = 0; i < n; i += 256) {
                const block_q2_K& blk = p[i/256];
                float d = fp16_to_fp32(blk.d), dmin = fp16_to_fp32(blk.dmin);
                for (int j = 0; j < 16; j++) {
                    float sc = blk.scales[j] & 0xF, mn = blk.scales[j] >> 4;
                    int half = j/8, jj = j%8, group = jj/2, lo = jj%2;
                    for (int l = 0; l < 16; l++) {
                        int q = (blk.qs[half*32 + lo*16 + l] >> (group*2)) & 3;
                        dst[i + j*16 + l] = d * sc * q - dmin * mn;
                    }
                }
            }
            return;
        }
        case GGMLType::IQ2_XXS: {
            for (int i = 0; i < n; i += 256) {
                block_iq2_xxs blk;
                std::memcpy(&blk, w.data + static_cast<size_t>(i / 256) * sizeof(blk),
                            sizeof(blk));
                const float d = fp16_to_fp32(blk.d);
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    uint32_t aux32[2];
                    std::memcpy(aux32, blk.qs + 4 * ib32, sizeof(aux32));
                    const uint8_t* grid_indices =
                        reinterpret_cast<const uint8_t*>(&aux32[0]);
                    const float db = d * (0.5f + static_cast<float>(aux32[1] >> 28)) * 0.25f;
                    for (int group = 0; group < 4; ++group) {
                        const uint64_t grid = kIq2XxsGrid[grid_indices[group]];
                        const uint8_t signs =
                            kIq2XxsSigns[(aux32[1] >> (7 * group)) & 127u];
                        for (int lane = 0; lane < 8; ++lane) {
                            const float magnitude =
                                static_cast<float>((grid >> (8 * lane)) & 0xffu);
                            dst[i + ib32 * 32 + group * 8 + lane] =
                                db * magnitude * ((signs & (1u << lane)) ? -1.0f : 1.0f);
                        }
                    }
                }
            }
            return;
        }
        case GGMLType::IQ1_S: {
            constexpr float delta_magnitude = 0.125f;
            for (int i = 0; i < n; i += 256) {
                block_iq1_s blk;
                std::memcpy(&blk, w.data + static_cast<size_t>(i / 256) * sizeof(blk),
                            sizeof(blk));
                const float d = fp16_to_fp32(blk.d);
                for (int ib32 = 0; ib32 != 8; ++ib32) {
                    const uint16_t high = blk.qh[ib32];
                    const float scale = d * static_cast<float>(2 * ((high >> 12) & 7) + 1);
                    const float delta = (high & 0x8000) ? -delta_magnitude : delta_magnitude;
                    for (int group = 0; group != 4; ++group) {
                        const uint16_t grid_index = static_cast<uint16_t>(
                            blk.qs[4 * ib32 + group] | (((high >> (3 * group)) & 7) << 8));
                        const uint64_t grid = kIq1SGrid[grid_index];
                        for (int lane = 0; lane != 8; ++lane) {
                            const int8_t level = static_cast<int8_t>(grid >> (8 * lane));
                            dst[i + ib32 * 32 + group * 8 + lane] =
                                scale * (static_cast<float>(level) + delta);
                        }
                    }
                }
            }
            return;
        }
        case GGMLType::Q3_K: {
            const block_q3_K* p = reinterpret_cast<const block_q3_K*>(w.data);
            const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
            for (int i = 0; i < n; i += 256) {
                const block_q3_K& blk = p[i/256];
                float d = fp16_to_fp32(blk.d);
                uint32_t auxs[4];
                std::memcpy(auxs, blk.scales, 12);
                uint32_t tmp = auxs[2];
                auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
                const uint8_t* scales = reinterpret_cast<const uint8_t*>(auxs);
                for (int j = 0; j < 16; j++) {
                    float sc = static_cast<float>(scales[j]) - 32;
                    for (int l = 0; l < 16; l++) {
                        int e = j*16 + l;
                        int g = e/32, li = e%32;
                        int low2 = (blk.qs[(g/4)*32 + li] >> ((g%4)*2)) & 3;
                        int hbit = (blk.hmask[li] >> g) & 1;
                        int a = low2 - 4 + 4*hbit;
                        dst[i + e] = d * sc * a;
                    }
                }
            }
            return;
        }
        case GGMLType::Q5_K: {
            const block_q5_K* p = reinterpret_cast<const block_q5_K*>(w.data);
            for (int i = 0; i < n; i += 256) {
                const block_q5_K& blk = p[i/256];
                float d = fp16_to_fp32(blk.d), dmin = fp16_to_fp32(blk.dmin);
                const uint8_t* q = blk.qs;
                int is = 0;
                for (int jb = 0; jb < 256; jb += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is + 0, blk.scales, &sc, &m); float d1 = d*sc, m1 = dmin*m;
                    get_scale_min_k4(is + 1, blk.scales, &sc, &m); float d2 = d*sc, m2 = dmin*m;
                    int bit = 2 * (jb / 64);
                    for (int l = 0; l < 32; l++) {
                        int v1 = (q[l] & 0xF) + 16 * ((blk.qh[l] >> bit) & 1);
                        int v2 = (q[l] >> 4) + 16 * ((blk.qh[l] >> (bit + 1)) & 1);
                        dst[i + jb + l]      = d1 * v1 - m1;
                        dst[i + jb + 32 + l] = d2 * v2 - m2;
                    }
                    q += 32; is += 2;
                }
            }
            return;
        }
        default:
            fprintf(stderr, "dequantize: unsupported type %s\n", type_name(w.type));
            std::memset(dst, 0, sizeof(float) * n);
    }
}

} // namespace Laplace
