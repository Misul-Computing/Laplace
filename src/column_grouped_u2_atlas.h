#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <variant>
#include <vector>

#include "column_grouped_affine_uint2_skip.h"
#include "tensor.h"

namespace Laplace {

struct ColumnGroupedU2AtlasSource {
    uint32_t binding_id = 0;
    GGMLType source_format = GGMLType::F32;
    std::span<const uint8_t> source;
    uint64_t logical_k = 0;
    uint64_t logical_n = 0;
    std::span<const float> importance;
};

enum class ColumnGroupedU2AtlasError : uint8_t {
    None = 0,
    Empty,
    DuplicateBinding,
    SourceFormatUnsupported,
    ShapeUnsupported,
    SourceLengthMismatch,
    ImportanceInvalid,
    Overflow,
    AllocationFailed,
    ConversionFailed,
    IntegrityMismatch,
    SealFailed,
};

struct ColumnGroupedU2AtlasEntry {
    uint32_t binding_id = 0;
    GGMLType source_format = GGMLType::F32;
    uint64_t values_offset = 0;
    uint64_t scales_offset = 0;
    uint64_t biases_offset = 0;
    ColumnGroupedAffineUInt2SkipV1Storage storage;
};

class ColumnGroupedU2AtlasBuilder;

class ColumnGroupedU2Atlas {
public:
    const std::vector<ColumnGroupedU2AtlasEntry>& entries() const { return entries_; }
    const uint8_t* data() const { return mapping_.get(); }
    size_t logical_bytes() const { return logical_bytes_; }
    size_t mapped_bytes() const { return mapped_bytes_; }
    uint64_t source_bytes() const { return source_bytes_; }

private:
    friend class ColumnGroupedU2AtlasBuilder;

    ColumnGroupedU2Atlas() = default;

    std::shared_ptr<uint8_t> mapping_;
    std::vector<ColumnGroupedU2AtlasEntry> entries_;
    size_t logical_bytes_ = 0;
    size_t mapped_bytes_ = 0;
    uint64_t source_bytes_ = 0;
};

using ColumnGroupedU2AtlasResult =
    std::variant<ColumnGroupedU2Atlas, ColumnGroupedU2AtlasError>;

// Builds all selected tensors into one page-aligned anonymous mapping. The
// source views are needed only during construction. A successful atlas is
// sealed read-only before it is returned.
ColumnGroupedU2AtlasResult build_column_grouped_u2_atlas(
    std::span<const ColumnGroupedU2AtlasSource> sources);

}  // namespace Laplace
