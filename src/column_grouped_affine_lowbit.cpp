#include "column_grouped_affine_lowbit.h"

#include "fp16.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace Laplace {
namespace {

constexpr uint32_t kGroupElements = 256;
constexpr uint32_t kPlaneAlignment = 128;
constexpr size_t kQ4KBlockBytes = 144;
constexpr size_t kQ6KBlockBytes = 210;

void set_error(ColumnGroupedAffineLowBitError* error, ColumnGroupedAffineLowBitError value) {
    if (error) *error = value;
}

void sha_bytes(CC_SHA256_CTX& context, const uint8_t* bytes, size_t size) {
    constexpr size_t kChunk = 1u << 20;
    while (size != 0) {
        const size_t count = std::min(size, kChunk);
        CC_SHA256_Update(&context, bytes, static_cast<CC_LONG>(count));
        bytes += count;
        size -= count;
    }
}

void sha_u8(CC_SHA256_CTX& context, uint8_t value) {
    sha_bytes(context, &value, sizeof(value));
}

void sha_u16(CC_SHA256_CTX& context, uint16_t value) {
    const uint8_t bytes[2] = {static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8u)};
    sha_bytes(context, bytes, sizeof(bytes));
}

void sha_u32(CC_SHA256_CTX& context, uint32_t value) {
    const uint8_t bytes[4] = {
        static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8u),
        static_cast<uint8_t>(value >> 16u), static_cast<uint8_t>(value >> 24u),
    };
    sha_bytes(context, bytes, sizeof(bytes));
}

void sha_u64(CC_SHA256_CTX& context, uint64_t value) {
    const uint8_t bytes[8] = {
        static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8u),
        static_cast<uint8_t>(value >> 16u), static_cast<uint8_t>(value >> 24u),
        static_cast<uint8_t>(value >> 32u), static_cast<uint8_t>(value >> 40u),
        static_cast<uint8_t>(value >> 48u), static_cast<uint8_t>(value >> 56u),
    };
    sha_bytes(context, bytes, sizeof(bytes));
}

void sha_f32(CC_SHA256_CTX& context, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    sha_u32(context, bits);
}

std::array<uint8_t, 32> finish_sha(CC_SHA256_CTX& context) {
    std::array<uint8_t, 32> digest{};
    CC_SHA256_Final(digest.data(), &context);
    return digest;
}

uint16_t load_le_u16(const uint8_t* bytes) {
    return static_cast<uint16_t>(static_cast<uint16_t>(bytes[0]) |
                                 (static_cast<uint16_t>(bytes[1]) << 8u));
}

void get_scale_min_k4(int index, const uint8_t* packed, uint8_t* scale, uint8_t* minimum) {
    if (index < 4) {
        *scale = packed[index] & 63u;
        *minimum = packed[index + 4] & 63u;
    } else {
        *scale = static_cast<uint8_t>((packed[index + 4] & 0x0fu) |
                                      static_cast<uint8_t>((packed[index - 4] >> 6u) << 4u));
        *minimum = static_cast<uint8_t>((packed[index + 4] >> 4u) |
                                        static_cast<uint8_t>((packed[index] >> 6u) << 4u));
    }
}

bool valid_source_format(ColumnGroupedAffineLowBitSourceFormat format) {
    return format == ColumnGroupedAffineLowBitSourceFormat::Q4_K ||
           format == ColumnGroupedAffineLowBitSourceFormat::Q6_K;
}

size_t source_block_bytes(ColumnGroupedAffineLowBitSourceFormat format) {
    return format == ColumnGroupedAffineLowBitSourceFormat::Q4_K ? kQ4KBlockBytes :
           format == ColumnGroupedAffineLowBitSourceFormat::Q6_K ? kQ6KBlockBytes : 0;
}

bool checked_product(uint64_t left, uint64_t right, uint64_t* result) {
    if (right != 0 && left > std::numeric_limits<uint64_t>::max() / right) return false;
    *result = left * right;
    return true;
}

bool make_contract(uint8_t bits, uint32_t logical_k, uint32_t logical_n,
                   ColumnGroupedAffineLowBitV1Contract* contract) {
    if (!contract || (bits != 2 && bits != 3 && bits != 4) || logical_k == 0 || logical_n == 0 ||
        logical_n % kGroupElements != 0) return false;

    const uint32_t packed_bytes = (kGroupElements * bits + 7u) / 8u;
    const uint64_t output_groups = logical_n / kGroupElements;
    uint64_t group_count = 0;
    uint64_t values_bytes = 0;
    if (!checked_product(logical_k, output_groups, &group_count) ||
        !checked_product(group_count, packed_bytes, &values_bytes) ||
        group_count > std::numeric_limits<size_t>::max() ||
        values_bytes > std::numeric_limits<size_t>::max()) return false;

    ColumnGroupedAffineLowBitV1Contract candidate;
    candidate.bits = bits;
    candidate.logical_k = logical_k;
    candidate.logical_n = logical_n;
    candidate.packed_bytes = packed_bytes;
    candidate.scale_bytes_per_group = sizeof(uint16_t);
    candidate.bias_bytes_per_group = sizeof(uint16_t);
    candidate.group_count = group_count;
    candidate.values_bytes = values_bytes;
    candidate.plane_alignment = kPlaneAlignment;
    *contract = candidate;
    return true;
}

bool valid_contract(const ColumnGroupedAffineLowBitV1Contract& contract) {
    ColumnGroupedAffineLowBitV1Contract expected;
    return make_contract(contract.bits, contract.logical_k, contract.logical_n, &expected) &&
           contract == expected;
}

bool checked_plane_bytes(size_t count, size_t* bytes) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) return false;
    *bytes = count * sizeof(uint16_t);
    return true;
}

bool aligned(const void* pointer, uint32_t alignment) {
    return pointer != nullptr && (reinterpret_cast<uintptr_t>(pointer) % alignment) == 0;
}

bool range_end(const void* pointer, size_t bytes, uintptr_t* end) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(pointer);
    if (bytes > std::numeric_limits<uintptr_t>::max() - start) return false;
    *end = start + bytes;
    return true;
}

bool ranges_overlap(const void* left, size_t left_bytes, const void* right, size_t right_bytes) {
    uintptr_t left_end = 0;
    uintptr_t right_end = 0;
    if (!range_end(left, left_bytes, &left_end) || !range_end(right, right_bytes, &right_end)) return true;
    const uintptr_t left_start = reinterpret_cast<uintptr_t>(left);
    const uintptr_t right_start = reinterpret_cast<uintptr_t>(right);
    return left_start < right_end && right_start < left_end;
}

bool validate_plane_layout(const ColumnGroupedAffineLowBitV1Contract& contract,
                           const ColumnGroupedAffineLowBitV1Planes& planes,
                           ColumnGroupedAffineLowBitError* error) {
    if (!planes.values || !planes.scales || !planes.biases) {
        set_error(error, ColumnGroupedAffineLowBitError::PlanePointerNull);
        return false;
    }
    if (planes.values_bytes != static_cast<size_t>(contract.values_bytes) ||
        planes.scale_count != static_cast<size_t>(contract.group_count) ||
        planes.bias_count != static_cast<size_t>(contract.group_count)) {
        set_error(error, ColumnGroupedAffineLowBitError::PlaneLengthMismatch);
        return false;
    }
    if (!aligned(planes.values, contract.plane_alignment) ||
        !aligned(planes.scales, contract.plane_alignment) ||
        !aligned(planes.biases, contract.plane_alignment)) {
        set_error(error, ColumnGroupedAffineLowBitError::PlaneAlignmentMismatch);
        return false;
    }

    size_t scale_bytes = 0;
    size_t bias_bytes = 0;
    if (!checked_plane_bytes(planes.scale_count, &scale_bytes) ||
        !checked_plane_bytes(planes.bias_count, &bias_bytes) ||
        ranges_overlap(planes.values, planes.values_bytes, planes.scales, scale_bytes) ||
        ranges_overlap(planes.values, planes.values_bytes, planes.biases, bias_bytes) ||
        ranges_overlap(planes.scales, scale_bytes, planes.biases, bias_bytes)) {
        set_error(error, ColumnGroupedAffineLowBitError::PlaneOverlap);
        return false;
    }
    return true;
}

bool validate_output_input_disjoint(const ColumnGroupedAffineLowBitV1Contract& contract,
                                    const ColumnGroupedAffineLowBitV1Planes& planes,
                                    std::span<const uint8_t> source_blocks,
                                    std::span<const float> importance,
                                    ColumnGroupedAffineLowBitError* error) {
    size_t scale_bytes = 0;
    size_t bias_bytes = 0;
    if (!checked_plane_bytes(planes.scale_count, &scale_bytes) ||
        !checked_plane_bytes(planes.bias_count, &bias_bytes)) {
        set_error(error, ColumnGroupedAffineLowBitError::InvalidContract);
        return false;
    }
    const bool values_overlap =
        ranges_overlap(planes.values, static_cast<size_t>(contract.values_bytes),
                       source_blocks.data(), source_blocks.size()) ||
        ranges_overlap(planes.values, static_cast<size_t>(contract.values_bytes),
                       importance.data(), importance.size_bytes());
    const bool scales_overlap =
        ranges_overlap(planes.scales, scale_bytes, source_blocks.data(), source_blocks.size()) ||
        ranges_overlap(planes.scales, scale_bytes, importance.data(), importance.size_bytes());
    const bool biases_overlap =
        ranges_overlap(planes.biases, bias_bytes, source_blocks.data(), source_blocks.size()) ||
        ranges_overlap(planes.biases, bias_bytes, importance.data(), importance.size_bytes());
    if (values_overlap || scales_overlap || biases_overlap) {
        set_error(error, ColumnGroupedAffineLowBitError::SourcePlaneOverlap);
        return false;
    }
    return true;
}

bool validate_plane_data(const ColumnGroupedAffineLowBitV1Contract& contract,
                         const ColumnGroupedAffineLowBitV1Planes& planes,
                         ColumnGroupedAffineLowBitError* error) {
    if (!validate_plane_layout(contract, planes, error)) return false;
    for (size_t i = 0; i < planes.scale_count; ++i) {
        const float scale = fp16_to_fp32(planes.scales[i]);
        const float bias = fp16_to_fp32(planes.biases[i]);
        if (!std::isfinite(scale) || scale < 0.0f || !std::isfinite(bias)) {
            set_error(error, ColumnGroupedAffineLowBitError::ConversionFailed);
            return false;
        }
    }
    return true;
}

std::array<uint8_t, 32> derived_digest(const ColumnGroupedAffineLowBitV1Storage& storage) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    static constexpr uint8_t tag[] = {'L', 'A', 'P', 'A', 'F', '1', 'D'};
    sha_bytes(context, tag, sizeof(tag));
    const auto& c = storage.contract;
    sha_u16(context, c.version);
    sha_u8(context, c.bits);
    sha_u8(context, c.reserved);
    sha_u32(context, c.logical_k);
    sha_u32(context, c.logical_n);
    sha_u32(context, c.group_elements);
    sha_u32(context, c.packed_bytes);
    sha_u32(context, c.scale_bytes_per_group);
    sha_u32(context, c.bias_bytes_per_group);
    sha_u32(context, c.plane_alignment);
    sha_u64(context, c.group_count);
    sha_u64(context, c.values_bytes);
    sha_bytes(context, storage.source_digest.data(), storage.source_digest.size());
    sha_bytes(context, storage.planes.values, storage.planes.values_bytes);
    for (size_t i = 0; i < storage.planes.scale_count; ++i) sha_u16(context, storage.planes.scales[i]);
    for (size_t i = 0; i < storage.planes.bias_count; ++i) sha_u16(context, storage.planes.biases[i]);
    return finish_sha(context);
}

bool expected_source_size(ColumnGroupedAffineLowBitSourceFormat format, uint32_t logical_k,
                          uint32_t logical_n, size_t* bytes) {
    const size_t block_bytes = source_block_bytes(format);
    if (!valid_source_format(format) || logical_k == 0 || logical_n == 0 || logical_k % 256u != 0 ||
        logical_n % kGroupElements != 0) return false;
    uint64_t row_bytes = 0;
    uint64_t total = 0;
    if (!checked_product(logical_k / 256u, block_bytes, &row_bytes) ||
        !checked_product(logical_n, row_bytes, &total) || total > std::numeric_limits<size_t>::max())
        return false;
    *bytes = static_cast<size_t>(total);
    return true;
}

bool valid_source_inputs(ColumnGroupedAffineLowBitSourceFormat format,
                         std::span<const uint8_t> source_blocks,
                         std::span<const float> importance,
                         uint32_t logical_k, uint32_t logical_n,
                         ColumnGroupedAffineLowBitError* error) {
    size_t source_bytes = 0;
    if (!expected_source_size(format, logical_k, logical_n, &source_bytes)) {
        set_error(error, valid_source_format(format) ?
                           ColumnGroupedAffineLowBitError::SourceShapeUnsupported :
                           ColumnGroupedAffineLowBitError::InvalidSourceFormat);
        return false;
    }
    if (!source_blocks.data() || source_blocks.size() != source_bytes) {
        set_error(error, ColumnGroupedAffineLowBitError::SourceLengthMismatch);
        return false;
    }
    const uint64_t value_count = static_cast<uint64_t>(logical_k) * logical_n;
    if (value_count > std::numeric_limits<size_t>::max() || !importance.data() ||
        importance.size() != static_cast<size_t>(value_count)) {
        set_error(error, ColumnGroupedAffineLowBitError::ImportanceLengthMismatch);
        return false;
    }
    for (float weight : importance) {
        if (!std::isfinite(weight) || weight < 0.0f) {
            set_error(error, ColumnGroupedAffineLowBitError::ImportanceInvalid);
            return false;
        }
    }
    const uint32_t output_groups = logical_n / kGroupElements;
    for (uint32_t column = 0; column < logical_k; ++column) {
        for (uint32_t output_group = 0; output_group < output_groups; ++output_group) {
            bool any_positive = false;
            for (uint32_t lane = 0; lane < kGroupElements; ++lane) {
                const size_t row = static_cast<size_t>(output_group) * kGroupElements + lane;
                any_positive = any_positive || importance[row * logical_k + column] > 0.0f;
            }
            if (!any_positive) {
                set_error(error, ColumnGroupedAffineLowBitError::ImportanceInvalid);
                return false;
            }
        }
    }
    return true;
}

void decode_q4_k_block(const uint8_t* block, float* output) {
    const float d = fp16_to_fp32(load_le_u16(block));
    const float dmin = fp16_to_fp32(load_le_u16(block + 2));
    const uint8_t* q = block + 16;
    for (int part = 0; part < 4; ++part) {
        uint8_t scale0 = 0;
        uint8_t minimum0 = 0;
        uint8_t scale1 = 0;
        uint8_t minimum1 = 0;
        get_scale_min_k4(part * 2, block + 4, &scale0, &minimum0);
        get_scale_min_k4(part * 2 + 1, block + 4, &scale1, &minimum1);
        const float d0 = d * scale0;
        const float m0 = dmin * minimum0;
        const float d1 = d * scale1;
        const float m1 = dmin * minimum1;
        const size_t offset = static_cast<size_t>(part) * 64;
        const uint8_t* packed = q + static_cast<size_t>(part) * 32;
        for (size_t i = 0; i < 32; ++i) output[offset + i] = d0 * (packed[i] & 0x0fu) - m0;
        for (size_t i = 0; i < 32; ++i) output[offset + 32 + i] = d1 * (packed[i] >> 4u) - m1;
    }
}

void decode_q6_k_block(const uint8_t* block, float* output) {
    const uint8_t* ql = block;
    const uint8_t* qh = block + 128;
    const uint8_t* scales = block + 192;
    const float d = fp16_to_fp32(load_le_u16(block + 208));
    for (int half = 0; half < 2; ++half) {
        for (int lane = 0; lane < 32; ++lane) {
            const int scale_index = lane / 16;
            const int q1 = static_cast<int>((ql[lane] & 0x0fu) | ((qh[lane] & 3u) << 4u)) - 32;
            const int q2 = static_cast<int>((ql[lane + 32] & 0x0fu) |
                                            (((qh[lane] >> 2u) & 3u) << 4u)) - 32;
            const int q3 = static_cast<int>((ql[lane] >> 4u) |
                                            (((qh[lane] >> 4u) & 3u) << 4u)) - 32;
            const int q4 = static_cast<int>((ql[lane + 32] >> 4u) |
                                            (((qh[lane] >> 6u) & 3u) << 4u)) - 32;
            const int scale_offset = half * 8;
            output[half * 128 + lane] = d * static_cast<float>(static_cast<int8_t>(
                scales[scale_offset + scale_index])) * static_cast<float>(q1);
            output[half * 128 + lane + 32] =
                d * static_cast<float>(static_cast<int8_t>(scales[scale_offset + scale_index + 2])) *
                static_cast<float>(q2);
            output[half * 128 + lane + 64] =
                d * static_cast<float>(static_cast<int8_t>(scales[scale_offset + scale_index + 4])) *
                static_cast<float>(q3);
            output[half * 128 + lane + 96] =
                d * static_cast<float>(static_cast<int8_t>(scales[scale_offset + scale_index + 6])) *
                static_cast<float>(q4);
        }
        ql += 64;
        qh += 32;
    }
}

bool decode_source_rows(ColumnGroupedAffineLowBitSourceFormat format,
                        std::span<const uint8_t> source_blocks,
                        uint32_t logical_k, uint32_t output_group,
                        std::span<float> values,
                        ColumnGroupedAffineLowBitError* error) {
    const size_t block_bytes = source_block_bytes(format);
    const size_t blocks_per_row = logical_k / 256u;
    std::array<float, 256> block_values{};
    for (uint32_t lane = 0; lane < kGroupElements; ++lane) {
        const uint32_t row = output_group * kGroupElements + lane;
        for (size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            const size_t offset =
                (static_cast<size_t>(row) * blocks_per_row + block_index) * block_bytes;
            if (format == ColumnGroupedAffineLowBitSourceFormat::Q4_K)
                decode_q4_k_block(source_blocks.data() + offset, block_values.data());
            else
                decode_q6_k_block(source_blocks.data() + offset, block_values.data());
            for (size_t block_lane = 0; block_lane < block_values.size(); ++block_lane) {
                if (!std::isfinite(block_values[block_lane])) {
                    set_error(error, ColumnGroupedAffineLowBitError::NonFiniteSource);
                    return false;
                }
                values[static_cast<size_t>(lane) * logical_k + block_index * 256u + block_lane] =
                    block_values[block_lane];
            }
        }
    }
    return true;
}

uint32_t rounded_code(double value, uint32_t maximum) {
    if (!std::isfinite(value) || value <= 0.0) return 0;
    if (value >= static_cast<double>(maximum)) return maximum;
    const double lower = std::floor(value);
    const double fraction = value - lower;
    uint32_t result = static_cast<uint32_t>(lower);
    if (fraction > 0.5 || (fraction == 0.5 && (result & 1u) != 0)) ++result;
    return std::min(result, maximum);
}

struct GroupFit {
    float scale = 0.0f;
    float bias = 0.0f;
    std::array<uint8_t, kGroupElements> codes{};
};

void weighted_fit(const std::array<float, kGroupElements>& values,
                  const std::array<float, kGroupElements>& importance,
                  GroupFit* fit) {
    long double sum_weight = 0.0L;
    long double sum_q = 0.0L;
    long double sum_x = 0.0L;
    long double sum_qq = 0.0L;
    long double sum_qx = 0.0L;
    for (size_t i = 0; i < kGroupElements; ++i) {
        const long double weight = importance[i];
        const long double q = fit->codes[i];
        const long double x = values[i];
        sum_weight += weight;
        sum_q += weight * q;
        sum_x += weight * x;
        sum_qq += weight * q * q;
        sum_qx += weight * q * x;
    }
    const long double denominator = sum_weight * sum_qq - sum_q * sum_q;
    const long double numerator = sum_weight * sum_qx - sum_q * sum_x;
    if (!(denominator > 0.0L) || !(numerator > 0.0L)) return;
    const long double scale = numerator / denominator;
    const long double bias = (sum_x - scale * sum_q) / sum_weight;
    if (!std::isfinite(static_cast<double>(scale)) ||
        !std::isfinite(static_cast<double>(bias)) || scale < 0.0L) return;
    fit->scale = static_cast<float>(scale);
    fit->bias = static_cast<float>(bias);
}

bool quantize_group(const std::array<float, kGroupElements>& values,
                    const std::array<float, kGroupElements>& importance,
                    uint8_t bits, GroupFit* fit) {
    const uint32_t maximum = (1u << bits) - 1u;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum_value = -std::numeric_limits<float>::infinity();
    double sum_weight = 0.0;
    for (size_t i = 0; i < kGroupElements; ++i) {
        if (importance[i] == 0.0f) continue;
        minimum = std::min(minimum, values[i]);
        maximum_value = std::max(maximum_value, values[i]);
        sum_weight += importance[i];
    }
    if (!std::isfinite(minimum) || !std::isfinite(maximum_value) || !(sum_weight > 0.0)) return false;

    fit->bias = minimum;
    fit->scale = (maximum_value - minimum) / static_cast<float>(maximum);
    if (fit->scale == 0.0f) {
        fit->bias = minimum;
        fit->codes.fill(0);
        return std::isfinite(fit->bias);
    }

    for (size_t i = 0; i < kGroupElements; ++i)
        fit->codes[i] = static_cast<uint8_t>(rounded_code(
            (static_cast<double>(values[i]) - fit->bias) / fit->scale, maximum));
    weighted_fit(values, importance, fit);
    for (size_t i = 0; i < kGroupElements; ++i)
        fit->codes[i] = static_cast<uint8_t>(rounded_code(
            (static_cast<double>(values[i]) - fit->bias) / fit->scale, maximum));
    return std::isfinite(fit->scale) && fit->scale >= 0.0f && std::isfinite(fit->bias);
}

void pack_codes(uint8_t* output, uint8_t bits, const std::array<uint8_t, kGroupElements>& codes) {
    const size_t packed_bytes = (kGroupElements * bits) / 8u;
    std::memset(output, 0, packed_bytes);
    for (size_t i = 0; i < kGroupElements; ++i) {
        const size_t bit = i * bits;
        for (uint8_t part = 0; part < bits; ++part)
            output[(bit + part) / 8u] |=
                static_cast<uint8_t>(((codes[i] >> part) & 1u) << ((bit + part) % 8u));
    }
}

std::array<uint8_t, 32> source_digest_impl(ColumnGroupedAffineLowBitSourceFormat format,
                                           std::span<const uint8_t> source_blocks,
                                           std::span<const float> importance,
                                           uint32_t logical_k, uint32_t logical_n) {
    if ((!source_blocks.data() && !source_blocks.empty()) ||
        (!importance.data() && !importance.empty())) return {};
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    static constexpr uint8_t tag[] = {'L', 'A', 'P', 'A', 'F', '1', 'S'};
    sha_bytes(context, tag, sizeof(tag));
    sha_u8(context, static_cast<uint8_t>(format));
    sha_u32(context, logical_k);
    sha_u32(context, logical_n);
    sha_u64(context, source_blocks.size());
    sha_u64(context, importance.size());
    sha_bytes(context, source_blocks.data(), source_blocks.size());
    for (float value : importance) sha_f32(context, value);
    return finish_sha(context);
}

uint32_t unpack_code(const uint8_t* values, uint8_t bits, uint32_t index) {
    const uint32_t bit = index * bits;
    uint32_t code = 0;
    for (uint8_t part = 0; part < bits; ++part)
        code |= static_cast<uint32_t>((values[(bit + part) / 8u] >> ((bit + part) % 8u)) & 1u) << part;
    return code;
}

}  // namespace

bool column_grouped_affine_lowbit_v1_make_contract(
    uint8_t bits,
    uint32_t logical_k,
    uint32_t logical_n,
    ColumnGroupedAffineLowBitV1Contract* contract,
    ColumnGroupedAffineLowBitError* error) {
    if (!contract) {
        set_error(error, ColumnGroupedAffineLowBitError::InvalidOutput);
        return false;
    }
    if (bits != 2 && bits != 3 && bits != 4) {
        set_error(error, ColumnGroupedAffineLowBitError::InvalidBits);
        return false;
    }
    if (!make_contract(bits, logical_k, logical_n, contract)) {
        set_error(error, ColumnGroupedAffineLowBitError::InvalidContract);
        return false;
    }
    set_error(error, ColumnGroupedAffineLowBitError::None);
    return true;
}

std::array<uint8_t, 32> column_grouped_affine_lowbit_v1_source_digest(
    ColumnGroupedAffineLowBitSourceFormat format,
    std::span<const uint8_t> source_blocks,
    std::span<const float> importance,
    uint32_t logical_k,
    uint32_t logical_n) {
    return source_digest_impl(format, source_blocks, importance, logical_k, logical_n);
}

bool column_grouped_affine_lowbit_v1_convert(
    ColumnGroupedAffineLowBitSourceFormat format,
    std::span<const uint8_t> source_blocks,
    std::span<const float> importance,
    uint32_t logical_k,
    uint32_t logical_n,
    uint8_t bits,
    ColumnGroupedAffineLowBitV1Planes output_planes,
    ColumnGroupedAffineLowBitV1Storage* output,
    ColumnGroupedAffineLowBitError* error) {
    if (!output) {
        set_error(error, ColumnGroupedAffineLowBitError::InvalidOutput);
        return false;
    }
    ColumnGroupedAffineLowBitV1Contract contract;
    if (!make_contract(bits, logical_k, logical_n, &contract)) {
        set_error(error, bits == 2 || bits == 3 || bits == 4 ?
                           ColumnGroupedAffineLowBitError::InvalidContract :
                           ColumnGroupedAffineLowBitError::InvalidBits);
        return false;
    }
    if (!validate_plane_layout(contract, output_planes, error) ||
        !valid_source_inputs(format, source_blocks, importance, logical_k, logical_n, error)) return false;

    const auto provenance_digest =
        source_digest_impl(format, source_blocks, importance, logical_k, logical_n);
    if (!validate_output_input_disjoint(contract, output_planes, source_blocks, importance, error))
        return false;

    const uint64_t tile_count64 = static_cast<uint64_t>(logical_k) * kGroupElements;
    if (tile_count64 > std::numeric_limits<size_t>::max()) {
        set_error(error, ColumnGroupedAffineLowBitError::SourceShapeUnsupported);
        return false;
    }
    const size_t tile_count = static_cast<size_t>(tile_count64);
    if (tile_count64 > std::numeric_limits<size_t>::max() / sizeof(float)) {
        set_error(error, ColumnGroupedAffineLowBitError::SourceShapeUnsupported);
        return false;
    }
    std::vector<float> source_tile;
    try {
        source_tile.resize(tile_count);
    } catch (const std::bad_alloc&) {
        set_error(error, ColumnGroupedAffineLowBitError::AllocationFailed);
        return false;
    } catch (const std::length_error&) {
        set_error(error, ColumnGroupedAffineLowBitError::AllocationFailed);
        return false;
    }

    const uint32_t output_groups = logical_n / kGroupElements;
    std::memset(output_planes.values, 0, output_planes.values_bytes);
    for (uint32_t output_group = 0; output_group < output_groups; ++output_group) {
        if (!decode_source_rows(format, source_blocks, logical_k, output_group,
                                std::span<float>(source_tile.data(), source_tile.size()), error))
            return false;
        for (uint32_t column = 0; column < logical_k; ++column) {
            std::array<float, kGroupElements> values{};
            std::array<float, kGroupElements> group_importance{};
            for (uint32_t lane = 0; lane < kGroupElements; ++lane) {
                const size_t row = static_cast<size_t>(output_group) * kGroupElements + lane;
                values[lane] = source_tile[static_cast<size_t>(lane) * logical_k + column];
                group_importance[lane] = importance[row * logical_k + column];
            }
            GroupFit fit;
            if (!quantize_group(values, group_importance, bits, &fit)) {
                set_error(error, ColumnGroupedAffineLowBitError::ImportanceInvalid);
                return false;
            }
            const uint64_t group = static_cast<uint64_t>(column) * output_groups + output_group;
            pack_codes(output_planes.values + group * contract.packed_bytes, bits, fit.codes);
            output_planes.scales[group] = fp32_to_fp16(fit.scale);
            output_planes.biases[group] = fp32_to_fp16(fit.bias);
            if (!std::isfinite(fp16_to_fp32(output_planes.scales[group])) ||
                !std::isfinite(fp16_to_fp32(output_planes.biases[group]))) {
                set_error(error, ColumnGroupedAffineLowBitError::ConversionFailed);
                return false;
            }
        }
    }

    ColumnGroupedAffineLowBitV1Storage candidate;
    candidate.contract = contract;
    candidate.planes = output_planes;
    candidate.source_digest = provenance_digest;
    if (!validate_plane_data(candidate.contract, candidate.planes, error)) return false;
    candidate.derived_digest = derived_digest(candidate);
    *output = candidate;
    set_error(error, ColumnGroupedAffineLowBitError::None);
    return true;
}

bool validate_column_grouped_affine_lowbit_v1(
    const ColumnGroupedAffineLowBitV1Storage& storage,
    const std::array<uint8_t, 32>& expected_source_digest,
    ColumnGroupedAffineLowBitError* error) {
    if (!valid_contract(storage.contract)) {
        set_error(error, ColumnGroupedAffineLowBitError::InvalidContract);
        return false;
    }
    if (!validate_plane_data(storage.contract, storage.planes, error)) return false;
    if (storage.source_digest != expected_source_digest) {
        set_error(error, ColumnGroupedAffineLowBitError::SourceDigestMismatch);
        return false;
    }
    if (storage.derived_digest != derived_digest(storage)) {
        set_error(error, ColumnGroupedAffineLowBitError::DerivedDigestMismatch);
        return false;
    }
    set_error(error, ColumnGroupedAffineLowBitError::None);
    return true;
}

bool column_grouped_affine_lowbit_v1_decode(
    const ColumnGroupedAffineLowBitV1Storage& storage,
    std::span<float> output,
    ColumnGroupedAffineLowBitError* error) {
    const uint64_t value_count = static_cast<uint64_t>(storage.contract.logical_k) *
                                 storage.contract.logical_n;
    if (value_count > std::numeric_limits<size_t>::max() ||
        output.size() != static_cast<size_t>(value_count)) {
        set_error(error, ColumnGroupedAffineLowBitError::OutputLengthMismatch);
        return false;
    }
    if (!output.data()) {
        set_error(error, ColumnGroupedAffineLowBitError::OutputPointerNull);
        return false;
    }
    if (!validate_column_grouped_affine_lowbit_v1(storage, storage.source_digest, error)) return false;

    const uint32_t output_groups = storage.contract.logical_n / storage.contract.group_elements;
    for (uint32_t row = 0; row < storage.contract.logical_n; ++row) {
        for (uint32_t column = 0; column < storage.contract.logical_k; ++column) {
            const uint64_t group = static_cast<uint64_t>(column) * output_groups +
                                   row / storage.contract.group_elements;
            const uint8_t* packed = storage.planes.values + group * storage.contract.packed_bytes;
            output[static_cast<size_t>(row) * storage.contract.logical_k + column] =
                fp16_to_fp32(storage.planes.scales[group]) *
                    static_cast<float>(unpack_code(packed, storage.contract.bits,
                                                    row % storage.contract.group_elements)) +
                fp16_to_fp32(storage.planes.biases[group]);
        }
    }
    set_error(error, ColumnGroupedAffineLowBitError::None);
    return true;
}

}  // namespace Laplace
