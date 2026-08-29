#include "column_grouped_u2_atlas.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <unordered_set>

#include <sys/mman.h>
#include <unistd.h>

#include "matmul.h"

namespace Laplace {
namespace {

constexpr uint64_t kPlaneAlignment =
    kColumnGroupedAffineUInt2SkipV1PlaneAlignment;

bool checked_add(uint64_t left, uint64_t right, uint64_t& result) {
    if (right > std::numeric_limits<uint64_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool align_up(uint64_t value, uint64_t alignment, uint64_t& result) {
    uint64_t with_padding = 0;
    if (alignment == 0 || !checked_add(value, alignment - 1u, with_padding)) return false;
    result = with_padding / alignment * alignment;
    return true;
}

struct PendingEntry {
    const ColumnGroupedU2AtlasSource* source = nullptr;
    ColumnGroupedU2AtlasEntry entry;
};

ColumnGroupedU2AtlasError preflight(
    std::span<const ColumnGroupedU2AtlasSource> sources,
    std::vector<PendingEntry>& pending, uint64_t& logical_bytes,
    uint64_t& source_bytes) {
    if (sources.empty()) return ColumnGroupedU2AtlasError::Empty;

    std::unordered_set<uint32_t> binding_ids;
    try {
        binding_ids.reserve(sources.size());
        pending.reserve(sources.size());
    } catch (const std::bad_alloc&) {
        return ColumnGroupedU2AtlasError::AllocationFailed;
    }

    for (const ColumnGroupedU2AtlasSource& source : sources) {
        if (!binding_ids.insert(source.binding_id).second)
            return ColumnGroupedU2AtlasError::DuplicateBinding;
        if (source.source_format != GGMLType::Q4_K &&
            source.source_format != GGMLType::Q6_K)
            return ColumnGroupedU2AtlasError::SourceFormatUnsupported;
        if (source.logical_k == 0 || source.logical_n == 0 ||
            source.logical_k % 256u != 0 || source.logical_n % 256u != 0 ||
            source.importance.size() != source.logical_k)
            return ColumnGroupedU2AtlasError::ShapeUnsupported;

        bool any_positive = false;
        for (float value : source.importance) {
            if (!std::isfinite(value) || value < 0.0f)
                return ColumnGroupedU2AtlasError::ImportanceInvalid;
            any_positive = any_positive || value > 0.0f;
        }
        if (!any_positive) return ColumnGroupedU2AtlasError::ImportanceInvalid;

        const uint64_t blocks_per_row = source.logical_k / 256u;
        const uint64_t block_bytes = bytes_per_block(source.source_format);
        if (source.logical_n > std::numeric_limits<uint64_t>::max() / blocks_per_row)
            return ColumnGroupedU2AtlasError::Overflow;
        const uint64_t blocks = source.logical_n * blocks_per_row;
        if (block_bytes == 0 || blocks > std::numeric_limits<uint64_t>::max() / block_bytes)
            return ColumnGroupedU2AtlasError::Overflow;
        const uint64_t expected_source_bytes = blocks * block_bytes;
        if (expected_source_bytes > std::numeric_limits<size_t>::max() ||
            source.source.size() != static_cast<size_t>(expected_source_bytes))
            return ColumnGroupedU2AtlasError::SourceLengthMismatch;

        ColumnGroupedAffineUInt2SkipV1Contract contract;
        ColumnGroupedAffineUInt2SkipV1Error contract_error{};
        if (!column_grouped_affine_uint2_skip_v1_make_contract(
                source.logical_k, source.logical_n, &contract, &contract_error))
            return ColumnGroupedU2AtlasError::ShapeUnsupported;

        PendingEntry candidate;
        candidate.source = &source;
        candidate.entry.binding_id = source.binding_id;
        candidate.entry.source_format = source.source_format;
        candidate.entry.storage.contract = contract;
        if (!align_up(logical_bytes, kPlaneAlignment,
                      candidate.entry.values_offset) ||
            !checked_add(candidate.entry.values_offset, contract.values_bytes,
                         logical_bytes) ||
            !align_up(logical_bytes, kPlaneAlignment,
                      candidate.entry.scales_offset) ||
            !checked_add(candidate.entry.scales_offset, contract.scale_bytes,
                         logical_bytes) ||
            !align_up(logical_bytes, kPlaneAlignment,
                      candidate.entry.biases_offset) ||
            !checked_add(candidate.entry.biases_offset, contract.bias_bytes,
                         logical_bytes) ||
            !checked_add(source_bytes, expected_source_bytes, source_bytes))
            return ColumnGroupedU2AtlasError::Overflow;
        pending.push_back(std::move(candidate));
    }
    return ColumnGroupedU2AtlasError::None;
}

}  // namespace

class ColumnGroupedU2AtlasBuilder {
public:
    static ColumnGroupedU2Atlas make(
        std::shared_ptr<uint8_t> mapping,
        std::vector<ColumnGroupedU2AtlasEntry> entries,
        size_t logical_bytes, size_t mapped_bytes, uint64_t source_bytes) {
        ColumnGroupedU2Atlas atlas;
        atlas.mapping_ = std::move(mapping);
        atlas.entries_ = std::move(entries);
        atlas.logical_bytes_ = logical_bytes;
        atlas.mapped_bytes_ = mapped_bytes;
        atlas.source_bytes_ = source_bytes;
        return atlas;
    }
};

ColumnGroupedU2AtlasResult build_column_grouped_u2_atlas(
    std::span<const ColumnGroupedU2AtlasSource> sources) {
    std::vector<PendingEntry> pending;
    uint64_t logical_bytes = 0;
    uint64_t source_bytes = 0;
    const ColumnGroupedU2AtlasError preflight_error =
        preflight(sources, pending, logical_bytes, source_bytes);
    if (preflight_error != ColumnGroupedU2AtlasError::None) return preflight_error;

    const long page_value = ::sysconf(_SC_PAGESIZE);
    if (page_value <= 0 || logical_bytes > std::numeric_limits<size_t>::max())
        return ColumnGroupedU2AtlasError::Overflow;
    const uint64_t page = static_cast<uint64_t>(page_value);
    uint64_t mapped_bytes_u64 = 0;
    if (!align_up(logical_bytes, page, mapped_bytes_u64) ||
        mapped_bytes_u64 > std::numeric_limits<size_t>::max())
        return ColumnGroupedU2AtlasError::Overflow;
    const size_t mapped_bytes = static_cast<size_t>(mapped_bytes_u64);
    void* raw = ::mmap(nullptr, mapped_bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANON, -1, 0);
    if (raw == MAP_FAILED) return ColumnGroupedU2AtlasError::AllocationFailed;
    std::shared_ptr<uint8_t> mapping(
        static_cast<uint8_t*>(raw),
        [mapped_bytes](uint8_t* pointer) { (void)::munmap(pointer, mapped_bytes); });

    std::vector<ColumnGroupedU2AtlasEntry> entries;
    try {
        entries.reserve(pending.size());
    } catch (const std::bad_alloc&) {
        return ColumnGroupedU2AtlasError::AllocationFailed;
    }
    for (PendingEntry& pending_entry : pending) {
        ColumnGroupedU2AtlasEntry entry = pending_entry.entry;
        const auto& contract = entry.storage.contract;
        std::span<uint8_t> values(
            mapping.get() + entry.values_offset,
            static_cast<size_t>(contract.values_bytes));
        std::span<uint16_t> scales(
            reinterpret_cast<uint16_t*>(mapping.get() + entry.scales_offset),
            static_cast<size_t>(contract.group_count));
        std::span<uint16_t> biases(
            reinterpret_cast<uint16_t*>(mapping.get() + entry.biases_offset),
            static_cast<size_t>(contract.group_count));
        DerivedStorageError conversion_error = DerivedStorageError::None;
        if (!derive_column_grouped_affine_u2_skip_v1_from_gguf(
                pending_entry.source->source_format,
                pending_entry.source->source,
                pending_entry.source->logical_k,
                pending_entry.source->logical_n,
                pending_entry.source->importance,
                values, scales, biases, &entry.storage, &conversion_error))
            return ColumnGroupedU2AtlasError::ConversionFailed;
        ColumnGroupedAffineUInt2SkipV1Error integrity_error{};
        if (!column_grouped_affine_uint2_skip_v1_validate(
                entry.storage, entry.storage.source_digest,
                entry.storage.provenance_digest, &integrity_error))
            return ColumnGroupedU2AtlasError::IntegrityMismatch;
        entries.push_back(std::move(entry));
    }
    if (::mprotect(mapping.get(), mapped_bytes, PROT_READ) != 0)
        return ColumnGroupedU2AtlasError::SealFailed;

    return ColumnGroupedU2AtlasBuilder::make(
        std::move(mapping), std::move(entries),
        static_cast<size_t>(logical_bytes), mapped_bytes, source_bytes);
}

}  // namespace Laplace
