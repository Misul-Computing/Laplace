#include "metal_library_source_catalog.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

#include "gemv_legacy.inc"
#include "metal_library_sources.inc"

namespace Laplace {

namespace {

MetalPipelineCompileContract source_compile_contract(
    MetalPipelineLanguageVersion language_version) noexcept {
    MetalPipelineCompileContract contract;
    contract.version = 1;
    contract.language_version = language_version;
    contract.fast_math_enabled = true;
    contract.reserved = 0;
    return contract;
}

MetalPipelineDigest digest_source(std::span<const uint8_t> bytes) noexcept {
    MetalPipelineDigest digest{};
    if (bytes.size() > static_cast<size_t>(std::numeric_limits<CC_LONG>::max()))
        return digest;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
    return digest;
}

std::span<const uint8_t> source_bytes(const char* source) noexcept {
    const size_t size = std::strlen(source);
    return {reinterpret_cast<const uint8_t*>(source), size};
}

MetalLibrarySourceRecord make_record(StructuralMetalLibraryId id,
                                     const char* source,
                                     MetalPipelineLanguageVersion language_version) noexcept {
    const std::span<const uint8_t> bytes = source_bytes(source);
    return {id, bytes, digest_source(bytes), source_compile_contract(language_version)};
}

const std::array<MetalLibrarySourceRecord,
                 MetalLibrarySourceCatalog::kLibraryCount>&
canonical_records() noexcept {
    static const std::array<MetalLibrarySourceRecord,
                            MetalLibrarySourceCatalog::kLibraryCount>
        records = {
            make_record(StructuralMetalLibraryId::Core, src_gemv,
                        MetalPipelineLanguageVersion::V2_3),
            make_record(StructuralMetalLibraryId::Prefill, src_prefill_f16,
                        MetalPipelineLanguageVersion::V2_0),
            make_record(StructuralMetalLibraryId::Sampler, src_sampler,
                        MetalPipelineLanguageVersion::V2_0),
        };
    return records;
}

bool valid_id(StructuralMetalLibraryId id) noexcept {
    return id == StructuralMetalLibraryId::Core ||
           id == StructuralMetalLibraryId::Prefill ||
           id == StructuralMetalLibraryId::Sampler;
}

bool same_bytes(std::span<const uint8_t> left,
                std::span<const uint8_t> right) noexcept {
    return left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin());
}

} // namespace

MetalLibrarySourceCatalogValidation validate_metal_library_source_catalog(
    std::span<const MetalLibrarySourceRecord> records) noexcept {
    if (records.size() != MetalLibrarySourceCatalog::kLibraryCount)
        return {MetalLibrarySourceCatalogError::WrongCount, records.size()};

    for (size_t index = 0; index != records.size(); ++index) {
        const auto& record = records[index];
        if (!valid_id(record.id))
            return {MetalLibrarySourceCatalogError::UnknownId, index};
        for (size_t prior = 0; prior != index; ++prior) {
            if (records[prior].id == record.id)
                return {MetalLibrarySourceCatalogError::DuplicateId, index};
            if (records[prior].source_digest == record.source_digest)
                return {MetalLibrarySourceCatalogError::DuplicateDigest, index};
        }
        if (record.bytes.empty() || record.bytes.data() == nullptr)
            return {MetalLibrarySourceCatalogError::EmptySource, index};
        if (record.bytes.back() == 0 ||
            std::find(record.bytes.begin(), record.bytes.end(), 0) !=
                record.bytes.end())
            return {MetalLibrarySourceCatalogError::EmbeddedNul, index};
        if (digest_source(record.bytes) != record.source_digest)
            return {MetalLibrarySourceCatalogError::DigestMismatch, index};
        if (record.compile_contract.version != 1 ||
            record.compile_contract.reserved != 0 ||
            static_cast<uint8_t>(record.compile_contract.language_version) <
                static_cast<uint8_t>(MetalPipelineLanguageVersion::V2_0) ||
            static_cast<uint8_t>(record.compile_contract.language_version) >
                static_cast<uint8_t>(MetalPipelineLanguageVersion::V4_1))
            return {MetalLibrarySourceCatalogError::InvalidCompileContract, index};
    }

    const auto& expected = canonical_records();
    for (size_t index = 0; index != records.size(); ++index) {
        size_t expected_index = expected.size();
        for (size_t candidate = 0; candidate != expected.size(); ++candidate)
            if (expected[candidate].id == records[index].id) {
                expected_index = candidate;
                break;
            }
        if (expected_index == expected.size())
            return {MetalLibrarySourceCatalogError::UnknownId, index};
        const auto& wanted = expected[expected_index];
        if (!same_bytes(records[index].bytes, wanted.bytes))
            return {MetalLibrarySourceCatalogError::SourceChanged, index};
    }
    return {};
}

MetalLibrarySourceCatalog::MetalLibrarySourceCatalog(
    std::array<MetalLibrarySourceRecord, kLibraryCount> records) noexcept
    : records_(records) {
    const auto validation = validate_metal_library_source_catalog(records_);
    valid_ = validation.ok();
    if (!valid_) return;
    std::sort(records_.begin(), records_.end(), [](const auto& left, const auto& right) {
        return static_cast<uint8_t>(left.id) < static_cast<uint8_t>(right.id);
    });
    for (size_t index = 0; index != records_.size(); ++index) {
        compiler_identities_[index] = {
            records_[index].id, records_[index].source_digest};
        transaction_sources_[index] = {
            records_[index].bytes, records_[index].source_digest,
            records_[index].compile_contract};
    }

    std::array<uint8_t, 256> identity_bytes{};
    size_t offset = 0;
    constexpr char domain[] = "laplace.metal-library-catalog.v1";
    const auto append = [&identity_bytes, &offset](uint8_t value) noexcept {
        if (offset < identity_bytes.size()) identity_bytes[offset++] = value;
    };
    for (size_t index = 0; index + 1 < sizeof(domain); ++index)
        append(static_cast<uint8_t>(domain[index]));
    for (const auto& record : records_) {
        append(static_cast<uint8_t>(record.id));
        for (const uint8_t value : record.source_digest) append(value);
        append(record.compile_contract.version);
        append(static_cast<uint8_t>(record.compile_contract.language_version));
        append(record.compile_contract.fast_math_enabled ? 1u : 0u);
        append(record.compile_contract.reserved);
    }
    CC_SHA256(identity_bytes.data(), static_cast<CC_LONG>(offset),
              catalog_digest_.data());
}

const MetalLibrarySourceRecord* MetalLibrarySourceCatalog::find(
    StructuralMetalLibraryId id) const noexcept {
    for (const auto& record : records_)
        if (record.id == id) return &record;
    return nullptr;
}

const MetalLibrarySourceCatalog& product_metal_library_source_catalog() noexcept {
    static const MetalLibrarySourceCatalog catalog(canonical_records());
    return catalog;
}

#if defined(LAPLACE_TESTING)
MetalLibrarySourceCatalogTestResult make_metal_library_source_catalog_for_testing(
    std::span<const MetalLibrarySourceRecord> records) {
    const auto validation = validate_metal_library_source_catalog(records);
    if (!validation.ok()) return validation;
    std::array<MetalLibrarySourceRecord, MetalLibrarySourceCatalog::kLibraryCount>
        normalized{};
    std::copy(records.begin(), records.end(), normalized.begin());
    std::sort(normalized.begin(), normalized.end(), [](const auto& left, const auto& right) {
        return static_cast<uint8_t>(left.id) < static_cast<uint8_t>(right.id);
    });
    return MetalLibrarySourceCatalog(normalized);
}
#endif

} // namespace Laplace
