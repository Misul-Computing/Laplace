#include "column_grouped_affine_uint2_skip.h"
#include "fp16.h"
#include "test_util.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

using namespace Laplace;

namespace {

template <typename T>
struct AlignedPlane {
    T* data = nullptr;
    size_t count = 0;
    size_t bytes = 0;

    explicit AlignedPlane(size_t count_in) : count(count_in) {
        bytes = ((count * sizeof(T) + 127u) / 128u) * 128u;
        if (bytes == 0) bytes = 128;
        data = static_cast<T*>(std::aligned_alloc(128u, bytes));
        CHECK(data != nullptr);
        if (data) std::memset(data, 0, bytes);
    }

    ~AlignedPlane() { std::free(data); }
    AlignedPlane(const AlignedPlane&) = delete;
    AlignedPlane& operator=(const AlignedPlane&) = delete;
};

ColumnGroupedAffineUInt2SkipV1Planes planes_for(AlignedPlane<uint8_t>& values,
                                                AlignedPlane<uint16_t>& scales,
                                                AlignedPlane<uint16_t>& biases,
                                                size_t values_bytes) {
    return {values.data, values_bytes, scales.data, scales.count, biases.data, biases.count};
}

struct AllocatedStorage {
    AlignedPlane<uint8_t> values;
    AlignedPlane<uint16_t> scales;
    AlignedPlane<uint16_t> biases;
    ColumnGroupedAffineUInt2SkipV1Storage storage;

    AllocatedStorage(uint64_t logical_k, uint64_t logical_n)
        : values(static_cast<size_t>((logical_n / 256u) * logical_k * 64u)),
          scales(static_cast<size_t>((logical_n / 256u) * logical_k)),
          biases(static_cast<size_t>((logical_n / 256u) * logical_k)) {
        storage.planes = planes_for(
            values, scales, biases,
            static_cast<size_t>((logical_n / 256u) * logical_k * 64u));
    }
};

void fill_storage(AllocatedStorage& allocated, uint64_t logical_k, uint64_t logical_n) {
    ColumnGroupedAffineUInt2SkipV1Error error = ColumnGroupedAffineUInt2SkipV1Error::None;
    CHECK(column_grouped_affine_uint2_skip_v1_make_contract(
        logical_k, logical_n, &allocated.storage.contract, &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::None);

    for (uint64_t row_group = 0; row_group < logical_n / 256u; ++row_group) {
        for (uint64_t column = 0; column < logical_k; ++column) {
            const uint64_t group = row_group * logical_k + column;
            allocated.scales.data[group] = fp32_to_fp16(0.25f + 0.01f * static_cast<float>(group));
            allocated.biases.data[group] = fp32_to_fp16(-0.5f + 0.02f * static_cast<float>(group));
            uint8_t* packed = allocated.values.data + group * 64u;
            for (uint32_t lane = 0; lane < 256u; ++lane) {
                const uint8_t code = static_cast<uint8_t>((lane + 2u * group) & 3u);
                packed[lane / 4u] |= static_cast<uint8_t>(code << (2u * (lane & 3u)));
            }
        }
    }

    for (size_t i = 0; i < allocated.storage.source_digest.size(); ++i)
        allocated.storage.source_digest[i] = static_cast<uint8_t>(i + 1u);
    for (size_t i = 0; i < allocated.storage.provenance_digest.size(); ++i)
        allocated.storage.provenance_digest[i] = static_cast<uint8_t>(0xa0u + i);
    allocated.storage.storage_digest =
        column_grouped_affine_uint2_skip_v1_storage_digest(allocated.storage);
}

void test_contract_and_layout() {
    ColumnGroupedAffineUInt2SkipV1Contract contract;
    ColumnGroupedAffineUInt2SkipV1Error error = ColumnGroupedAffineUInt2SkipV1Error::None;
    CHECK(column_grouped_affine_uint2_skip_v1_make_contract(512, 512, &contract, &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::None);
    CHECK(contract.version == 1u);
    CHECK(contract.logical_k == 512u && contract.logical_n == 512u);
    CHECK(contract.group_elements == 256u);
    CHECK(contract.packed_bytes_per_group == 64u);
    CHECK(contract.group_count == 1024u);
    CHECK(contract.values_bytes == 65536u);
    CHECK(contract.scale_bytes == 2048u && contract.bias_bytes == 2048u);

    CHECK(column_grouped_affine_uint2_skip_v1_make_contract(5120, 17408, &contract, &error));
    CHECK(contract.group_count == 348160u);
    CHECK(contract.values_bytes == 22282240u);
    CHECK(contract.scale_bytes == 696320u && contract.bias_bytes == 696320u);

    CHECK(!column_grouped_affine_uint2_skip_v1_make_contract(512, 511, &contract, &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::ShapeUnsupported);
    CHECK(!column_grouped_affine_uint2_skip_v1_make_contract(
        UINT64_MAX, 256, &contract, &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::Overflow);
    CHECK(!column_grouped_affine_uint2_skip_v1_make_contract(
        2, UINT64_C(0xffffffffffffff00), &contract, &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::Overflow);
}

void test_decode_and_digests() {
    constexpr uint64_t K = 512;
    constexpr uint64_t N = 512;
    AllocatedStorage allocated(K, N);
    fill_storage(allocated, K, N);
    ColumnGroupedAffineUInt2SkipV1Error error = ColumnGroupedAffineUInt2SkipV1Error::None;
    CHECK(column_grouped_affine_uint2_skip_v1_validate(
        allocated.storage, allocated.storage.source_digest,
        allocated.storage.provenance_digest, &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::None);

    std::vector<float> decoded(static_cast<size_t>(K * N));
    CHECK(column_grouped_affine_uint2_skip_v1_decode(
        allocated.storage, allocated.storage.source_digest,
        allocated.storage.provenance_digest, decoded, &error));
    for (uint64_t row : std::array<uint64_t, 6>{0, 1, 255, 256, 257, 511}) {
        for (uint64_t column : std::array<uint64_t, 5>{0, 1, 255, 256, 511}) {
            const uint64_t group = (row / 256u) * K + column;
            const float scale = fp16_to_fp32(allocated.scales.data[group]);
            const float bias = fp16_to_fp32(allocated.biases.data[group]);
            const uint8_t code = static_cast<uint8_t>((row + 2u * group) & 3u);
            CHECK(decoded[static_cast<size_t>(row * K + column)] ==
                  scale * static_cast<float>(code) + bias);
        }
    }
    CHECK(column_grouped_affine_uint2_skip_v1_storage_digest(allocated.storage) ==
          allocated.storage.storage_digest);

    const auto source_digest = allocated.storage.source_digest;
    allocated.storage.source_digest[0] ^= 1u;
    CHECK(!column_grouped_affine_uint2_skip_v1_validate(
        allocated.storage, source_digest, allocated.storage.provenance_digest, &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::SourceDigestMismatch);

    allocated.storage.source_digest = source_digest;
    auto provenance_tampered = allocated.storage;
    const auto expected_provenance = provenance_tampered.provenance_digest;
    provenance_tampered.provenance_digest[0] ^= 1u;
    provenance_tampered.storage_digest =
        column_grouped_affine_uint2_skip_v1_storage_digest(provenance_tampered);
    CHECK(!column_grouped_affine_uint2_skip_v1_decode(
        provenance_tampered, source_digest, expected_provenance, decoded, &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::ProvenanceDigestMismatch);
}

void test_integrity_rejections() {
    constexpr uint64_t K = 256;
    constexpr uint64_t N = 256;
    AllocatedStorage allocated(K, N);
    fill_storage(allocated, K, N);
    ColumnGroupedAffineUInt2SkipV1Error error = ColumnGroupedAffineUInt2SkipV1Error::None;
    const auto source = allocated.storage.source_digest;
    const auto provenance = allocated.storage.provenance_digest;

    allocated.values.data[0] ^= 1u;
    CHECK(!column_grouped_affine_uint2_skip_v1_validate(allocated.storage, source, provenance, &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::StorageDigestMismatch);
    allocated.values.data[0] ^= 1u;
    allocated.storage.storage_digest =
        column_grouped_affine_uint2_skip_v1_storage_digest(allocated.storage);

    CHECK(!column_grouped_affine_uint2_skip_v1_validate(allocated.storage, source,
                                                        std::array<uint8_t, 32>{}, &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::DigestMissing);

    auto malformed = allocated.storage;
    malformed.planes.values = nullptr;
    CHECK(!column_grouped_affine_uint2_skip_v1_validate(malformed, source, provenance, &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::PlanePointerNull);

    malformed = allocated.storage;
    malformed.planes.values = allocated.storage.planes.values + 1;
    CHECK(!column_grouped_affine_uint2_skip_v1_validate(malformed, source, provenance, &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::PlaneAlignmentMismatch);

    malformed = allocated.storage;
    --malformed.planes.values_bytes;
    CHECK(!column_grouped_affine_uint2_skip_v1_validate(malformed, source, provenance, &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::PlaneLengthMismatch);

    malformed = allocated.storage;
    malformed.planes.biases = reinterpret_cast<const uint16_t*>(allocated.storage.planes.values);
    CHECK(!column_grouped_affine_uint2_skip_v1_validate(malformed, source, provenance, &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::PlaneOverlap);

    std::vector<float> too_short(static_cast<size_t>(K * N - 1u));
    CHECK(!column_grouped_affine_uint2_skip_v1_decode(
        allocated.storage, source, provenance, too_short, &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::OutputLengthMismatch);

    allocated.scales.data[0] = fp32_to_fp16(std::numeric_limits<float>::infinity());
    allocated.storage.storage_digest =
        column_grouped_affine_uint2_skip_v1_storage_digest(allocated.storage);
    CHECK(!column_grouped_affine_uint2_skip_v1_validate(allocated.storage, source, provenance, &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::NonFinitePlane);

    allocated.scales.data[0] = fp32_to_fp16(1.0f);
    allocated.biases.data[0] = fp32_to_fp16(std::numeric_limits<float>::quiet_NaN());
    allocated.storage.storage_digest =
        column_grouped_affine_uint2_skip_v1_storage_digest(allocated.storage);
    CHECK(!column_grouped_affine_uint2_skip_v1_validate(allocated.storage, source, provenance, &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::NonFinitePlane);
}

void test_signed_scale_is_valid() {
    constexpr uint64_t K = 256;
    constexpr uint64_t N = 256;
    AllocatedStorage allocated(K, N);
    fill_storage(allocated, K, N);
    allocated.scales.data[0] = fp32_to_fp16(-0.5f);
    allocated.biases.data[0] = fp32_to_fp16(0.25f);
    allocated.storage.storage_digest =
        column_grouped_affine_uint2_skip_v1_storage_digest(allocated.storage);

    ColumnGroupedAffineUInt2SkipV1Error error = ColumnGroupedAffineUInt2SkipV1Error::None;
    CHECK(column_grouped_affine_uint2_skip_v1_validate(
        allocated.storage, allocated.storage.source_digest,
        allocated.storage.provenance_digest, &error));
    std::vector<float> decoded(static_cast<size_t>(K * N));
    CHECK(column_grouped_affine_uint2_skip_v1_decode(
        allocated.storage, allocated.storage.source_digest,
        allocated.storage.provenance_digest, decoded, &error));
    const uint8_t code = allocated.values.data[0] & 3u;
    CHECK(decoded[0] == -0.5f * static_cast<float>(code) + 0.25f);
}

void test_literal_wire_vector_and_golden_digest() {
    AllocatedStorage allocated(1, 256);
    ColumnGroupedAffineUInt2SkipV1Error error = ColumnGroupedAffineUInt2SkipV1Error::None;
    CHECK(column_grouped_affine_uint2_skip_v1_make_contract(
        1, 256, &allocated.storage.contract, &error));
    std::memset(allocated.values.data, 0xe4, 64);
    auto* scale_bytes = reinterpret_cast<uint8_t*>(allocated.scales.data);
    auto* bias_bytes = reinterpret_cast<uint8_t*>(allocated.biases.data);
    scale_bytes[0] = 0x00;
    scale_bytes[1] = 0x3c;
    bias_bytes[0] = 0x00;
    bias_bytes[1] = 0xbc;
    for (size_t index = 0; index != 32; ++index) {
        allocated.storage.source_digest[index] = static_cast<uint8_t>(index + 1u);
        allocated.storage.provenance_digest[index] = static_cast<uint8_t>(0xa0u + index);
    }
    allocated.storage.storage_digest =
        column_grouped_affine_uint2_skip_v1_storage_digest(allocated.storage);
    constexpr std::array<uint8_t, 32> expected_digest = {
        0x6c, 0x3f, 0x2e, 0x4f, 0x9c, 0x7a, 0xc0, 0x71,
        0xef, 0x86, 0xdf, 0x3b, 0xb4, 0x8d, 0x43, 0xd1,
        0xc1, 0x9a, 0x02, 0x89, 0xb6, 0x44, 0xaf, 0x8b,
        0xba, 0x8a, 0xe7, 0x6d, 0x04, 0x0e, 0x23, 0xc1,
    };
    CHECK(allocated.storage.storage_digest == expected_digest);

    std::vector<float> decoded(256);
    CHECK(column_grouped_affine_uint2_skip_v1_decode(
        allocated.storage, allocated.storage.source_digest,
        allocated.storage.provenance_digest, decoded, &error));
    for (size_t row = 0; row != decoded.size(); ++row) {
        CHECK(decoded[row] == static_cast<float>(row & 3u) - 1.0f);
    }
}

void test_output_must_not_alias_storage() {
    constexpr size_t kOutputValues = 256;
    AlignedPlane<float> arena(kOutputValues + 96);
    auto* bytes = reinterpret_cast<uint8_t*>(arena.data);

    ColumnGroupedAffineUInt2SkipV1Storage storage;
    ColumnGroupedAffineUInt2SkipV1Error error = ColumnGroupedAffineUInt2SkipV1Error::None;
    CHECK(column_grouped_affine_uint2_skip_v1_make_contract(1, 256, &storage.contract, &error));
    storage.planes = {
        bytes, 64, reinterpret_cast<const uint16_t*>(bytes + 128), 1,
        reinterpret_cast<const uint16_t*>(bytes + 256), 1};
    std::memset(bytes, 0, 64);
    bytes[128] = 0x00;
    bytes[129] = 0x3c;
    bytes[256] = 0x00;
    bytes[257] = 0x00;
    for (size_t index = 0; index != 32; ++index) {
        storage.source_digest[index] = static_cast<uint8_t>(index + 1u);
        storage.provenance_digest[index] = static_cast<uint8_t>(0x40u + index);
    }
    storage.storage_digest = column_grouped_affine_uint2_skip_v1_storage_digest(storage);
    CHECK(!column_grouped_affine_uint2_skip_v1_decode(
        storage, storage.source_digest, storage.provenance_digest,
        std::span<float>(arena.data, kOutputValues), &error));
    CHECK(error == ColumnGroupedAffineUInt2SkipV1Error::OutputOverlap);
}

}  // namespace

int main() {
    test_contract_and_layout();
    test_decode_and_digests();
    test_integrity_rejections();
    test_signed_scale_is_valid();
    test_literal_wire_vector_and_golden_digest();
    test_output_must_not_alias_storage();
    return test_summary("test_column_grouped_affine_uint2_skip");
}
