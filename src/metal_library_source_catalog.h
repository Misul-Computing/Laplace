#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>

#include "metal_pipeline_transaction.h"
#include "structural_metal_compiler.h"

namespace Laplace {

struct MetalLibrarySourceRecord {
    StructuralMetalLibraryId id = StructuralMetalLibraryId::Core;
    std::span<const uint8_t> bytes;
    MetalPipelineDigest source_digest{};
    MetalPipelineCompileContract compile_contract{};
};

enum class MetalLibrarySourceCatalogError : uint8_t {
    Valid = 0,
    WrongCount = 1,
    UnknownId = 2,
    DuplicateId = 3,
    DuplicateDigest = 4,
    EmptySource = 5,
    EmbeddedNul = 6,
    DigestMismatch = 7,
    SourceChanged = 8,
    InvalidCompileContract = 9,
};

struct MetalLibrarySourceCatalogValidation {
    MetalLibrarySourceCatalogError error = MetalLibrarySourceCatalogError::Valid;
    size_t index = 0;

    bool ok() const noexcept {
        return error == MetalLibrarySourceCatalogError::Valid;
    }
};

class MetalLibrarySourceCatalog {
public:
    static constexpr size_t kLibraryCount = 3;

    std::span<const MetalLibrarySourceRecord> records() const noexcept {
        return records_;
    }
    std::span<const StructuralMetalLibraryIdentity> compiler_identities() const noexcept {
        return compiler_identities_;
    }
    std::span<const MetalPipelineLibrarySource> transaction_sources() const noexcept {
        return transaction_sources_;
    }
    const MetalPipelineDigest& catalog_digest() const noexcept {
        return catalog_digest_;
    }
    const MetalLibrarySourceRecord* find(StructuralMetalLibraryId id) const noexcept;
    bool valid() const noexcept { return valid_; }

private:
    explicit MetalLibrarySourceCatalog(
        std::array<MetalLibrarySourceRecord, kLibraryCount> records) noexcept;

    std::array<MetalLibrarySourceRecord, kLibraryCount> records_{};
    std::array<StructuralMetalLibraryIdentity, kLibraryCount>
        compiler_identities_{};
    std::array<MetalPipelineLibrarySource, kLibraryCount>
        transaction_sources_{};
    MetalPipelineDigest catalog_digest_{};
    bool valid_ = false;

    friend const MetalLibrarySourceCatalog&
    product_metal_library_source_catalog() noexcept;
#if defined(LAPLACE_TESTING)
    friend std::variant<MetalLibrarySourceCatalog,
                        MetalLibrarySourceCatalogValidation>
    make_metal_library_source_catalog_for_testing(
        std::span<const MetalLibrarySourceRecord> records);
#endif
};

const MetalLibrarySourceCatalog& product_metal_library_source_catalog() noexcept;

MetalLibrarySourceCatalogValidation validate_metal_library_source_catalog(
    std::span<const MetalLibrarySourceRecord> records) noexcept;

#if defined(LAPLACE_TESTING)
using MetalLibrarySourceCatalogTestResult =
    std::variant<MetalLibrarySourceCatalog, MetalLibrarySourceCatalogValidation>;

MetalLibrarySourceCatalogTestResult make_metal_library_source_catalog_for_testing(
    std::span<const MetalLibrarySourceRecord> records);
#endif

} // namespace Laplace
