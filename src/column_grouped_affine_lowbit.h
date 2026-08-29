#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace Laplace {

enum class ColumnGroupedAffineLowBitSourceFormat : uint8_t {
    Q4_K = 1,
    Q6_K = 2,
};

enum class ColumnGroupedAffineLowBitError : uint8_t {
    None,
    InvalidOutput,
    InvalidBits,
    InvalidSourceFormat,
    InvalidContract,
    SourceLengthMismatch,
    SourceShapeUnsupported,
    ImportanceLengthMismatch,
    ImportanceInvalid,
    NonFiniteSource,
    ConversionFailed,
    PlanePointerNull,
    PlaneLengthMismatch,
    PlaneAlignmentMismatch,
    PlaneOverlap,
    SourcePlaneOverlap,
    AllocationFailed,
    SourceDigestMismatch,
    DerivedDigestMismatch,
    OutputLengthMismatch,
    OutputPointerNull,
};

// Physical layout for a matrix with row-major logical values [logical_n][logical_k].
// The packed values are grouped by input column, then by 256-output groups. The
// scale and bias arrays use the same group order and are caller-owned FP16 planes.
struct ColumnGroupedAffineLowBitV1Contract {
    uint16_t version = 1;
    uint8_t bits = 3;
    uint8_t reserved = 0;
    uint32_t logical_k = 0;
    uint32_t logical_n = 0;
    uint32_t group_elements = 256;
    uint32_t packed_bytes = 96;
    uint32_t scale_bytes_per_group = 2;
    uint32_t bias_bytes_per_group = 2;
    uint32_t plane_alignment = 128;
    uint64_t group_count = 0;
    uint64_t values_bytes = 0;

    friend bool operator==(const ColumnGroupedAffineLowBitV1Contract&,
                           const ColumnGroupedAffineLowBitV1Contract&) = default;
};

// The converter writes these planes but never allocates or frees them. The
// caller must keep all three aligned allocations alive for the storage view.
struct ColumnGroupedAffineLowBitV1Planes {
    uint8_t* values = nullptr;
    size_t values_bytes = 0;
    uint16_t* scales = nullptr;
    size_t scale_count = 0;
    uint16_t* biases = nullptr;
    size_t bias_count = 0;
};

struct ColumnGroupedAffineLowBitV1Storage {
    ColumnGroupedAffineLowBitV1Contract contract;
    ColumnGroupedAffineLowBitV1Planes planes;
    std::array<uint8_t, 32> source_digest{};
    std::array<uint8_t, 32> derived_digest{};
};

bool column_grouped_affine_lowbit_v1_make_contract(
    uint8_t bits,
    uint32_t logical_k,
    uint32_t logical_n,
    ColumnGroupedAffineLowBitV1Contract* contract,
    ColumnGroupedAffineLowBitError* error);

std::array<uint8_t, 32> column_grouped_affine_lowbit_v1_source_digest(
    ColumnGroupedAffineLowBitSourceFormat format,
    std::span<const uint8_t> source_blocks,
    std::span<const float> importance,
    uint32_t logical_k,
    uint32_t logical_n);

bool column_grouped_affine_lowbit_v1_convert(
    ColumnGroupedAffineLowBitSourceFormat format,
    std::span<const uint8_t> source_blocks,
    std::span<const float> importance,
    uint32_t logical_k,
    uint32_t logical_n,
    uint8_t bits,
    ColumnGroupedAffineLowBitV1Planes output_planes,
    ColumnGroupedAffineLowBitV1Storage* output,
    ColumnGroupedAffineLowBitError* error);

bool validate_column_grouped_affine_lowbit_v1(
    const ColumnGroupedAffineLowBitV1Storage& storage,
    const std::array<uint8_t, 32>& expected_source_digest,
    ColumnGroupedAffineLowBitError* error);

// Decodes to row-major logical values [logical_n][logical_k]. This is a
// standalone physical decoder and does not call the source converter.
bool column_grouped_affine_lowbit_v1_decode(
    const ColumnGroupedAffineLowBitV1Storage& storage,
    std::span<float> output,
    ColumnGroupedAffineLowBitError* error);

}  // namespace Laplace
