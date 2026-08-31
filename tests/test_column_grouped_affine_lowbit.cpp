#include "column_grouped_affine_lowbit.h"
#include "fp16.h"
#include "quant_ref.h"
#include "test_util.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

using namespace Laplace;

namespace {

template <typename T>
struct AlignedPlane {
    T* data = nullptr;
    size_t count = 0;

    explicit AlignedPlane(size_t count_in) : count(count_in) {
        const size_t bytes = ((count * sizeof(T) + 127u) / 128u) * 128u;
        data = static_cast<T*>(std::aligned_alloc(128u, bytes == 0 ? 128u : bytes));
        CHECK(data != nullptr);
        if (data) std::memset(data, 0, bytes == 0 ? 64u : bytes);
    }

    ~AlignedPlane() { std::free(data); }
    AlignedPlane(const AlignedPlane&) = delete;
    AlignedPlane& operator=(const AlignedPlane&) = delete;
};

ColumnGroupedAffineLowBitV1Planes planes_for(AlignedPlane<uint8_t>& values,
                                             AlignedPlane<uint16_t>& scales,
                                             AlignedPlane<uint16_t>& biases) {
    return {values.data, values.count, scales.data, scales.count, biases.data, biases.count};
}

std::vector<uint8_t> make_q4_source(uint32_t logical_k, uint32_t logical_n) {
    const size_t blocks_per_row = logical_k / 256u;
    std::vector<uint8_t> bytes(static_cast<size_t>(logical_n) * blocks_per_row * 144u);
    for (uint32_t row = 0; row < logical_n; ++row) {
        for (size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            quant_ref::block_q4_K block{};
            block.d = fp32_to_fp16(0.025f + 0.001f * static_cast<float>(row % 7u));
            block.dmin = fp32_to_fp16(0.003f + 0.0005f * static_cast<float>(block_index));
            for (size_t i = 0; i < sizeof(block.scales); ++i)
                block.scales[i] = static_cast<uint8_t>(1u + ((row + i + block_index) % 17u));
            for (size_t i = 0; i < sizeof(block.qs); ++i)
                block.qs[i] = static_cast<uint8_t>(i * 29u + row * 7u + block_index * 11u);
            const size_t offset = (static_cast<size_t>(row) * blocks_per_row + block_index) * 144u;
            std::memcpy(bytes.data() + offset, &block, sizeof(block));
        }
    }
    return bytes;
}

std::vector<uint8_t> make_q6_source(uint32_t logical_k, uint32_t logical_n) {
    const size_t blocks_per_row = logical_k / 256u;
    std::vector<uint8_t> bytes(static_cast<size_t>(logical_n) * blocks_per_row * 210u);
    for (uint32_t row = 0; row < logical_n; ++row) {
        for (size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            quant_ref::block_q6_K block{};
            block.d = fp32_to_fp16(0.018f + 0.001f * static_cast<float>(row % 5u));
            for (size_t i = 0; i < sizeof(block.scales); ++i)
                block.scales[i] = static_cast<int8_t>((static_cast<int>(i + row + block_index) % 11) - 5);
            for (size_t i = 0; i < sizeof(block.ql); ++i)
                block.ql[i] = static_cast<uint8_t>(i * 17u + row * 5u + block_index * 3u);
            for (size_t i = 0; i < sizeof(block.qh); ++i)
                block.qh[i] = static_cast<uint8_t>(i * 13u + row * 3u + block_index * 7u);
            const size_t offset = (static_cast<size_t>(row) * blocks_per_row + block_index) * 210u;
            std::memcpy(bytes.data() + offset, &block, sizeof(block));
        }
    }
    return bytes;
}

std::vector<float> make_importance(uint32_t logical_k, uint32_t logical_n) {
    std::vector<float> importance(static_cast<size_t>(logical_k) * logical_n);
    for (uint32_t row = 0; row < logical_n; ++row) {
        for (uint32_t column = 0; column < logical_k; ++column) {
            importance[static_cast<size_t>(row) * logical_k + column] =
                0.25f + static_cast<float>((row * 19u + column * 7u) % 23u) / 11.0f;
        }
    }
    return importance;
}

std::vector<uint8_t> make_binary_q4_source(uint32_t logical_k, uint32_t logical_n) {
    const size_t blocks_per_row = logical_k / 256u;
    std::vector<uint8_t> bytes(static_cast<size_t>(logical_n) * blocks_per_row * 144u);
    for (uint32_t row = 0; row < logical_n; ++row) {
        for (size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            quant_ref::block_q4_K block{};
            block.d = fp32_to_fp16(1.0f);
            block.dmin = fp32_to_fp16(0.0f);
            for (size_t i = 0; i < 4; ++i) block.scales[i] = 1u;
            for (size_t i = 8; i < sizeof(block.scales); ++i) block.scales[i] = 1u;
            std::fill(std::begin(block.qs), std::end(block.qs),
                      static_cast<uint8_t>(row & 1u ? 0x11u : 0x00u));
            const size_t offset = (static_cast<size_t>(row) * blocks_per_row + block_index) * 144u;
            std::memcpy(bytes.data() + offset, &block, sizeof(block));
        }
    }
    return bytes;
}

double squared_error(std::span<const float> left, std::span<const float> right) {
    double result = 0.0;
    for (size_t i = 0; i < left.size(); ++i) {
        const double difference = static_cast<double>(left[i]) - right[i];
        result += difference * difference;
    }
    return result;
}

double squared_energy(std::span<const float> values) {
    double result = 0.0;
    for (float value : values) result += static_cast<double>(value) * value;
    return result;
}

uint32_t unpack_code(const ColumnGroupedAffineLowBitV1Storage& storage,
                     uint32_t column, uint32_t row) {
    const auto& c = storage.contract;
    const uint32_t output_groups = c.logical_n / c.group_elements;
    const uint64_t group = static_cast<uint64_t>(column) * output_groups + row / c.group_elements;
    const uint64_t bit = static_cast<uint64_t>(row % c.group_elements) * c.bits;
    const uint8_t* values = storage.planes.values + group * c.packed_bytes;
    uint32_t code = 0;
    for (uint8_t part = 0; part < c.bits; ++part)
        code |= static_cast<uint32_t>((values[(bit + part) / 8u] >> ((bit + part) % 8u)) & 1u) << part;
    return code;
}

void independent_decode(const ColumnGroupedAffineLowBitV1Storage& storage,
                         std::span<float> output) {
    const auto& c = storage.contract;
    const uint32_t output_groups = c.logical_n / c.group_elements;
    for (uint32_t row = 0; row < c.logical_n; ++row) {
        for (uint32_t column = 0; column < c.logical_k; ++column) {
            const uint64_t group = static_cast<uint64_t>(column) * output_groups + row / c.group_elements;
            output[static_cast<size_t>(row) * c.logical_k + column] =
                fp16_to_fp32(storage.planes.scales[group]) *
                    static_cast<float>(unpack_code(storage, column, row)) +
                fp16_to_fp32(storage.planes.biases[group]);
        }
    }
}

float maximum_error(std::span<const float> left, std::span<const float> right) {
    float result = 0.0f;
    for (size_t i = 0; i < left.size(); ++i)
        result = std::max(result, std::abs(left[i] - right[i]));
    return result;
}

std::vector<float> independent_source_values(std::span<const uint8_t> source,
                                             ColumnGroupedAffineLowBitSourceFormat format,
                                             uint32_t logical_k, uint32_t logical_n) {
    const size_t block_bytes = format == ColumnGroupedAffineLowBitSourceFormat::Q4_K ? 144u : 210u;
    const size_t blocks_per_row = logical_k / 256u;
    std::vector<float> values(static_cast<size_t>(logical_k) * logical_n);
    for (uint32_t row = 0; row < logical_n; ++row) {
        for (size_t block = 0; block < blocks_per_row; ++block) {
            const size_t offset = (static_cast<size_t>(row) * blocks_per_row + block) * block_bytes;
            if (format == ColumnGroupedAffineLowBitSourceFormat::Q4_K) {
                quant_ref::block_q4_K decoded{};
                std::memcpy(&decoded, source.data() + offset, sizeof(decoded));
                quant_ref::dequant_q4_K(&decoded, values.data() + static_cast<size_t>(row) * logical_k + block * 256u);
            } else {
                quant_ref::block_q6_K decoded{};
                std::memcpy(&decoded, source.data() + offset, sizeof(decoded));
                quant_ref::dequant_q6_K(&decoded, values.data() + static_cast<size_t>(row) * logical_k + block * 256u);
            }
        }
    }
    return values;
}

struct AllocatedStorage {
    AlignedPlane<uint8_t> values;
    AlignedPlane<uint16_t> scales;
    AlignedPlane<uint16_t> biases;
    ColumnGroupedAffineLowBitV1Planes planes;

    explicit AllocatedStorage(uint8_t bits, uint32_t logical_k, uint32_t logical_n)
        : values(static_cast<size_t>(logical_k) * (logical_n / 256u) * ((256u * bits) / 8u)),
          scales(static_cast<size_t>(logical_k) * (logical_n / 256u)),
          biases(static_cast<size_t>(logical_k) * (logical_n / 256u)),
          planes(planes_for(values, scales, biases)) {}
};

bool convert(ColumnGroupedAffineLowBitSourceFormat format, std::span<const uint8_t> source,
             std::span<const float> importance, uint8_t bits, AllocatedStorage& allocated,
             ColumnGroupedAffineLowBitV1Storage& storage, ColumnGroupedAffineLowBitError& error,
             uint32_t logical_k, uint32_t logical_n) {
    return column_grouped_affine_lowbit_v1_convert(
        format, source, importance, logical_k, logical_n, bits, allocated.planes, &storage, &error);
}

void test_exact_packed_goldens() {
    constexpr uint32_t K = 256;
    constexpr uint32_t N = 256;
    const auto source = make_binary_q4_source(K, N);
    const std::vector<float> importance(static_cast<size_t>(K) * N, 1.0f);
    ColumnGroupedAffineLowBitError error = ColumnGroupedAffineLowBitError::None;
    for (uint8_t bits : {uint8_t{2}, uint8_t{3}, uint8_t{4}}) {
        AllocatedStorage allocated(bits, K, N);
        ColumnGroupedAffineLowBitV1Storage storage;
        CHECK(convert(ColumnGroupedAffineLowBitSourceFormat::Q4_K, source, importance, bits,
                      allocated, storage, error, K, N));

        const size_t packed_bytes = (static_cast<size_t>(256u) * bits) / 8u;
        std::array<uint8_t, 128> golden_bytes{};
        if (bits == 2) {
            std::fill(golden_bytes.begin(), golden_bytes.begin() + packed_bytes, 0xccu);
        } else if (bits == 3) {
            constexpr std::array<uint8_t, 3> pattern = {0x38u, 0x8eu, 0xe3u};
            for (size_t i = 0; i < packed_bytes; ++i) golden_bytes[i] = pattern[i % pattern.size()];
        } else {
            std::fill(golden_bytes.begin(), golden_bytes.begin() + packed_bytes, 0xf0u);
        }
        bool packed_exact = true;
        for (uint64_t group = 0; group < storage.contract.group_count; ++group) {
            packed_exact = packed_exact && std::equal(
                golden_bytes.begin(), golden_bytes.begin() + packed_bytes,
                storage.planes.values + group * packed_bytes);
        }
        CHECK(packed_exact);

        const uint16_t expected_scale = fp32_to_fp16(1.0f / static_cast<float>((1u << bits) - 1u));
        bool affine_exact = true;
        for (uint64_t group = 0; group < storage.contract.group_count; ++group) {
            affine_exact = affine_exact && storage.planes.scales[group] == expected_scale &&
                           storage.planes.biases[group] == fp32_to_fp16(0.0f);
        }
        CHECK(affine_exact);

        std::vector<float> decoded(static_cast<size_t>(K) * N);
        CHECK(column_grouped_affine_lowbit_v1_decode(storage, decoded, &error));
        const float quantized_one = fp16_to_fp32(expected_scale) * static_cast<float>((1u << bits) - 1u);
        bool decode_exact = true;
        for (uint32_t row = 0; row < N; ++row) {
            for (uint32_t column = 0; column < K; ++column) {
                const float expected = (row & 1u) != 0 ? quantized_one : 0.0f;
                decode_exact = decode_exact &&
                               decoded[static_cast<size_t>(row) * K + column] == expected;
            }
        }
        CHECK(decode_exact);
    }
    printf("golden_bytes q2=64 q3=96 q4=128\n");
}

void test_multiple_output_groups() {
    constexpr uint32_t K = 512;
    constexpr uint32_t N = 512;
    const auto source = make_q4_source(K, N);
    const auto importance = make_importance(K, N);
    AllocatedStorage allocated(3, K, N);
    ColumnGroupedAffineLowBitV1Storage storage;
    ColumnGroupedAffineLowBitError error = ColumnGroupedAffineLowBitError::None;
    CHECK(convert(ColumnGroupedAffineLowBitSourceFormat::Q4_K, source, importance, 3,
                  allocated, storage, error, K, N));
    CHECK(storage.contract.group_count == 1024u);
    CHECK(storage.planes.values_bytes == 98304u);
    std::vector<float> actual(static_cast<size_t>(K) * N);
    std::vector<float> expected(actual.size());
    CHECK(column_grouped_affine_lowbit_v1_decode(storage, actual, &error));
    independent_decode(storage, expected);
    CHECK(maximum_error(actual, expected) == 0.0f);
    printf("multi_group_rows=512\n");
}

void test_q3_contract_and_source_converters() {
    constexpr uint32_t K = 256;
    constexpr uint32_t N = 256;
    const auto importance = make_importance(K, N);
    const auto q4_source = make_q4_source(K, N);
    const auto q6_source = make_q6_source(K, N);
    const auto q4_reference = independent_source_values(
        q4_source, ColumnGroupedAffineLowBitSourceFormat::Q4_K, K, N);
    const auto q6_reference = independent_source_values(
        q6_source, ColumnGroupedAffineLowBitSourceFormat::Q6_K, K, N);
    ColumnGroupedAffineLowBitError error = ColumnGroupedAffineLowBitError::None;

    AllocatedStorage q3_planes(3, K, N);
    ColumnGroupedAffineLowBitV1Storage q3;
    CHECK(convert(ColumnGroupedAffineLowBitSourceFormat::Q4_K, q4_source, importance, 3,
                  q3_planes, q3, error, K, N));
    CHECK(error == ColumnGroupedAffineLowBitError::None);
    CHECK(q3.contract.version == 1u);
    CHECK(q3.contract.bits == 3u);
    CHECK(q3.contract.group_elements == 256u);
    ColumnGroupedAffineLowBitV1Contract declared;
    CHECK(column_grouped_affine_lowbit_v1_make_contract(3, K, N, &declared, &error));
    CHECK(declared == q3.contract);
    CHECK(!column_grouped_affine_lowbit_v1_make_contract(5, K, N, &declared, &error));
    CHECK(error == ColumnGroupedAffineLowBitError::InvalidBits);
    CHECK(q3.planes.values_bytes == 24576u);
    CHECK(q3.planes.scale_count == 256u);
    CHECK(q3.planes.bias_count == 256u);
    printf("q3_bytes values=%zu scales=%zu biases=%zu\n", q3.planes.values_bytes,
           q3.planes.scale_count * sizeof(uint16_t), q3.planes.bias_count * sizeof(uint16_t));
    CHECK(validate_column_grouped_affine_lowbit_v1(q3, q3.source_digest, &error));
    CHECK(error == ColumnGroupedAffineLowBitError::None);

    std::vector<float> actual(static_cast<size_t>(K) * N);
    std::vector<float> expected(actual.size());
    CHECK(column_grouped_affine_lowbit_v1_decode(q3, actual, &error));
    independent_decode(q3, expected);
    CHECK_MSG(maximum_error(actual, expected) == 0.0f, "q3 decode error=%g",
              maximum_error(actual, expected));
    printf("q3_source_max_error q4=%g\n", maximum_error(actual, q4_reference));
    CHECK(std::isfinite(maximum_error(actual, q4_reference)));
    const double q4_energy = squared_energy(q4_reference);
    const double q4_error = squared_error(actual, q4_reference);
    CHECK(std::isfinite(q4_energy) && std::isfinite(q4_error) && q4_energy > 0.0);
    CHECK(q4_error < q4_energy);
    printf("q3_source_sse q4=%g energy=%g\n", q4_error, q4_energy);

    AllocatedStorage q3_repeat_planes(3, K, N);
    ColumnGroupedAffineLowBitV1Storage q3_repeat;
    CHECK(convert(ColumnGroupedAffineLowBitSourceFormat::Q4_K, q4_source, importance, 3,
                  q3_repeat_planes, q3_repeat, error, K, N));
    CHECK(std::memcmp(q3.planes.values, q3_repeat.planes.values, q3.planes.values_bytes) == 0);
    CHECK(std::memcmp(q3.planes.scales, q3_repeat.planes.scales,
                      q3.planes.scale_count * sizeof(uint16_t)) == 0);
    CHECK(std::memcmp(q3.planes.biases, q3_repeat.planes.biases,
                      q3.planes.bias_count * sizeof(uint16_t)) == 0);
    CHECK(q3.source_digest == q3_repeat.source_digest);
    CHECK(q3.derived_digest == q3_repeat.derived_digest);

    std::vector<float> uniform_importance(importance.size(), 1.0f);
    AllocatedStorage q3_uniform_planes(3, K, N);
    ColumnGroupedAffineLowBitV1Storage q3_uniform;
    CHECK(convert(ColumnGroupedAffineLowBitSourceFormat::Q4_K, q4_source, uniform_importance, 3,
                  q3_uniform_planes, q3_uniform, error, K, N));
    CHECK(std::memcmp(q3.planes.values, q3_uniform.planes.values, q3.planes.values_bytes) != 0 ||
          std::memcmp(q3.planes.scales, q3_uniform.planes.scales,
                      q3.planes.scale_count * sizeof(uint16_t)) != 0 ||
          std::memcmp(q3.planes.biases, q3_uniform.planes.biases,
                      q3.planes.bias_count * sizeof(uint16_t)) != 0);

    AllocatedStorage q6_planes(3, K, N);
    ColumnGroupedAffineLowBitV1Storage q6;
    CHECK(convert(ColumnGroupedAffineLowBitSourceFormat::Q6_K, q6_source, importance, 3,
                  q6_planes, q6, error, K, N));
    CHECK(error == ColumnGroupedAffineLowBitError::None);
    CHECK(q6.source_digest != q3.source_digest);
    CHECK(column_grouped_affine_lowbit_v1_decode(q6, actual, &error));
    independent_decode(q6, expected);
    CHECK_MSG(maximum_error(actual, expected) == 0.0f, "q6 decode error=%g",
              maximum_error(actual, expected));
    printf("q3_source_max_error q6=%g\n", maximum_error(actual, q6_reference));
    CHECK(std::isfinite(maximum_error(actual, q6_reference)));
    const double q6_energy = squared_energy(q6_reference);
    const double q6_error = squared_error(actual, q6_reference);
    CHECK(std::isfinite(q6_energy) && std::isfinite(q6_error) && q6_energy > 0.0);
    CHECK(q6_error < q6_energy);
    printf("q3_source_sse q6=%g energy=%g\n", q6_error, q6_energy);

    std::vector<float> invalid_importance = importance;
    for (uint32_t row = 0; row < N; ++row) invalid_importance[static_cast<size_t>(row) * K] = 0.0f;
    CHECK(!convert(ColumnGroupedAffineLowBitSourceFormat::Q4_K, q4_source, invalid_importance, 3,
                   q3_planes, q3, error, K, N));
    CHECK(error == ColumnGroupedAffineLowBitError::ImportanceInvalid);
    CHECK(!convert(ColumnGroupedAffineLowBitSourceFormat::Q4_K,
                   std::span<const uint8_t>(q4_source.data(), q4_source.size() - 1u), importance,
                   3, q3_planes, q3, error, K, N));
    CHECK(error == ColumnGroupedAffineLowBitError::SourceLengthMismatch);

    for (uint8_t bits : {uint8_t{2}, uint8_t{4}}) {
        AllocatedStorage planes(bits, K, N);
        ColumnGroupedAffineLowBitV1Storage storage;
        CHECK(convert(ColumnGroupedAffineLowBitSourceFormat::Q4_K, q4_source, importance, bits,
                      planes, storage, error, K, N));
        CHECK(storage.contract.bits == bits);
        CHECK(validate_column_grouped_affine_lowbit_v1(storage, storage.source_digest, &error));
    }

    AllocatedStorage malformed_planes(3, K, N);
    ColumnGroupedAffineLowBitV1Storage malformed = q3;
    malformed.planes = malformed_planes.planes;
    malformed.planes.values_bytes -= 1u;
    CHECK(!validate_column_grouped_affine_lowbit_v1(malformed, q3.source_digest, &error));
    CHECK(error == ColumnGroupedAffineLowBitError::PlaneLengthMismatch);

    malformed = q3;
    malformed.contract.bits = 2;
    CHECK(!validate_column_grouped_affine_lowbit_v1(malformed, q3.source_digest, &error));
    CHECK(error == ColumnGroupedAffineLowBitError::InvalidContract);

    malformed = q3;
    malformed.planes.values = q3.planes.values + 1;
    CHECK(!validate_column_grouped_affine_lowbit_v1(malformed, q3.source_digest, &error));
    CHECK(error == ColumnGroupedAffineLowBitError::PlaneAlignmentMismatch);

    AlignedPlane<uint16_t> malformed_scales(q3.planes.scale_count);
    AlignedPlane<uint16_t> malformed_biases(q3.planes.bias_count);
    std::memcpy(malformed_scales.data, q3.planes.scales,
                q3.planes.scale_count * sizeof(uint16_t));
    std::memcpy(malformed_biases.data, q3.planes.biases,
                q3.planes.bias_count * sizeof(uint16_t));
    malformed = q3;
    malformed.planes.scales = malformed_scales.data;
    malformed.planes.biases = malformed_biases.data;
    malformed.planes.scales[0] ^= 1u;
    CHECK(!validate_column_grouped_affine_lowbit_v1(malformed, q3.source_digest, &error));
    CHECK(error == ColumnGroupedAffineLowBitError::DerivedDigestMismatch);

    auto wrong_digest = q3.source_digest;
    wrong_digest[0] ^= 0x80u;
    CHECK(!validate_column_grouped_affine_lowbit_v1(q3, wrong_digest, &error));
    CHECK(error == ColumnGroupedAffineLowBitError::SourceDigestMismatch);

    CHECK(!column_grouped_affine_lowbit_v1_decode(q3, std::span<float>(actual.data(), actual.size() - 1u),
                                                   &error));
    CHECK(error == ColumnGroupedAffineLowBitError::OutputLengthMismatch);
    CHECK(!column_grouped_affine_lowbit_v1_convert(
        ColumnGroupedAffineLowBitSourceFormat::Q4_K, q4_source, importance, K, N, 1,
        q3_planes.planes, &q3, &error));
    CHECK(error == ColumnGroupedAffineLowBitError::InvalidBits);

    const auto expected_digest = column_grouped_affine_lowbit_v1_source_digest(
        ColumnGroupedAffineLowBitSourceFormat::Q4_K, q4_source, importance, K, N);
    CHECK(q3.source_digest == expected_digest);
    printf("q3_checks=%d\n", g_checks);
}

void test_input_output_overlap_rejection() {
    constexpr uint32_t K = 256;
    constexpr uint32_t N = 256;
    const auto source = make_q4_source(K, N);
    const auto importance = make_importance(K, N);
    AlignedPlane<uint8_t> aligned_source(source.size());
    std::memcpy(aligned_source.data, source.data(), source.size());
    const auto source_before = std::vector<uint8_t>(aligned_source.data,
                                                     aligned_source.data + source.size());
    ColumnGroupedAffineLowBitError error = ColumnGroupedAffineLowBitError::None;

    AllocatedStorage values_overlap(3, K, N);
    values_overlap.planes.values = aligned_source.data;
    ColumnGroupedAffineLowBitV1Storage storage;
    CHECK(!convert(ColumnGroupedAffineLowBitSourceFormat::Q4_K,
                   std::span<const uint8_t>(aligned_source.data, source.size()), importance, 3,
                   values_overlap, storage, error, K, N));
    CHECK(error == ColumnGroupedAffineLowBitError::SourcePlaneOverlap);
    CHECK(std::memcmp(aligned_source.data, source_before.data(), source_before.size()) == 0);

    AlignedPlane<float> aligned_importance(importance.size());
    std::memcpy(aligned_importance.data, importance.data(), importance.size() * sizeof(float));
    const auto importance_before = std::vector<float>(aligned_importance.data,
                                                      aligned_importance.data + importance.size());
    AllocatedStorage scales_overlap(3, K, N);
    scales_overlap.planes.scales = reinterpret_cast<uint16_t*>(aligned_importance.data);
    CHECK(!convert(ColumnGroupedAffineLowBitSourceFormat::Q4_K, source,
                   std::span<const float>(aligned_importance.data, importance.size()), 3,
                   scales_overlap, storage, error, K, N));
    CHECK(error == ColumnGroupedAffineLowBitError::SourcePlaneOverlap);
    CHECK(std::memcmp(aligned_importance.data, importance_before.data(),
                      importance_before.size() * sizeof(float)) == 0);

    AllocatedStorage biases_overlap(3, K, N);
    biases_overlap.planes.biases = reinterpret_cast<uint16_t*>(aligned_source.data);
    CHECK(!convert(ColumnGroupedAffineLowBitSourceFormat::Q4_K,
                   std::span<const uint8_t>(aligned_source.data, source.size()), importance, 3,
                   biases_overlap, storage, error, K, N));
    CHECK(error == ColumnGroupedAffineLowBitError::SourcePlaneOverlap);
    CHECK(std::memcmp(aligned_source.data, source_before.data(), source_before.size()) == 0);
}

}  // namespace

int main() {
    test_exact_packed_goldens();
    test_multiple_output_groups();
    test_q3_contract_and_source_converters();
    test_input_output_overlap_rejection();
    return test_summary("test_column_grouped_affine_lowbit");
}
