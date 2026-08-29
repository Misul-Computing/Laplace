#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace Laplace {

inline constexpr uint32_t kColumnGroupedAffineUInt2SkipV1GroupElements = 256;
inline constexpr uint32_t kColumnGroupedAffineUInt2SkipV1PackedBytesPerGroup = 64;
inline constexpr uint32_t kColumnGroupedAffineUInt2SkipV1PlaneAlignment = 128;

enum class ColumnGroupedAffineUInt2SkipV1Error : uint8_t {
    None = 0,
    NullOutput,
    ShapeUnsupported,
    Overflow,
    ContractMismatch,
    PlanePointerNull,
    PlaneLengthMismatch,
    PlaneAlignmentMismatch,
    PlaneOverlap,
    NonFinitePlane,
    DigestMissing,
    SourceDigestMismatch,
    ProvenanceDigestMismatch,
    StorageDigestMismatch,
    OutputPointerNull,
    OutputLengthMismatch,
    OutputOverlap,
};

// Versioned physical layout for row-major logical weights W[N,K]. Group g
// contains one 256-value output-row group for one input column:
//
//     g = (row / 256) * K + column
//
// Values are LSB-first UInt2 codes. Each group has 64 value bytes, one FP16
// scale, and one FP16 bias. The three planes are separate and independently
// 128-byte aligned. No source format or model identity is part of this
// physical contract.
struct ColumnGroupedAffineUInt2SkipV1Contract {
    uint16_t version = 1;
    uint16_t reserved = 0;
    uint32_t group_elements = kColumnGroupedAffineUInt2SkipV1GroupElements;
    uint32_t packed_bytes_per_group = kColumnGroupedAffineUInt2SkipV1PackedBytesPerGroup;
    uint32_t scale_bytes_per_group = 2;
    uint32_t bias_bytes_per_group = 2;
    uint32_t plane_alignment = kColumnGroupedAffineUInt2SkipV1PlaneAlignment;
    uint64_t logical_k = 0;
    uint64_t logical_n = 0;
    uint64_t group_count = 0;
    uint64_t values_bytes = 0;
    uint64_t scale_bytes = 0;
    uint64_t bias_bytes = 0;

    friend bool operator==(const ColumnGroupedAffineUInt2SkipV1Contract&,
                           const ColumnGroupedAffineUInt2SkipV1Contract&) = default;
};

struct ColumnGroupedAffineUInt2SkipV1Planes {
    const uint8_t* values = nullptr;
    size_t values_bytes = 0;
    const uint16_t* scales = nullptr;
    size_t scale_count = 0;
    const uint16_t* biases = nullptr;
    size_t bias_count = 0;
};

struct ColumnGroupedAffineUInt2SkipV1Storage {
    ColumnGroupedAffineUInt2SkipV1Contract contract;
    ColumnGroupedAffineUInt2SkipV1Planes planes;
    std::array<uint8_t, 32> source_digest{};
    std::array<uint8_t, 32> provenance_digest{};
    std::array<uint8_t, 32> storage_digest{};
};

bool column_grouped_affine_uint2_skip_v1_make_contract(
    uint64_t logical_k, uint64_t logical_n,
    ColumnGroupedAffineUInt2SkipV1Contract* contract,
    ColumnGroupedAffineUInt2SkipV1Error* error);

// Scale and bias planes contain little-endian IEEE binary16 bytes. Computes
// the canonical digest over the versioned contract, source and provenance
// digests, and the exact bytes in all three planes. It returns a zero digest
// for a malformed storage view.
std::array<uint8_t, 32> column_grouped_affine_uint2_skip_v1_storage_digest(
    const ColumnGroupedAffineUInt2SkipV1Storage& storage);

bool column_grouped_affine_uint2_skip_v1_validate(
    const ColumnGroupedAffineUInt2SkipV1Storage& storage,
    const std::array<uint8_t, 32>& expected_source_digest,
    const std::array<uint8_t, 32>& expected_provenance_digest,
    ColumnGroupedAffineUInt2SkipV1Error* error);

// Independent scalar decoder. The output is row-major [logical_n][logical_k]
// and is validated against all recorded bytes and digests before decoding.
bool column_grouped_affine_uint2_skip_v1_decode(
    const ColumnGroupedAffineUInt2SkipV1Storage& storage,
    const std::array<uint8_t, 32>& expected_source_digest,
    const std::array<uint8_t, 32>& expected_provenance_digest,
    std::span<float> output,
    ColumnGroupedAffineUInt2SkipV1Error* error);

}  // namespace Laplace
