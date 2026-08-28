#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "semantic_model.h"

namespace Laplace {

// Adapter-neutral scalar storage vocabulary. It intentionally includes types
// that the semantic runtime may still reject later; parsing a physical package
// and admitting an executable kernel are separate decisions.
enum class ArtifactScalarType : uint16_t {
    F32 = 1,
    F16 = 2,
    BF16 = 3,
    I64 = 4,
    U64 = 5,
    I32 = 6,
    U32 = 7,
    I16 = 8,
    U16 = 9,
    I8 = 10,
    U8 = 11,
    Bool = 12,
    Packed = 13,
};

enum class ArtifactFactAuthority : uint8_t {
    Declared = 1,
    Structural = 2,
    Derived = 3,
};

struct ArtifactF32Bits {
    uint32_t value = 0;
    friend bool operator==(ArtifactF32Bits, ArtifactF32Bits) = default;
};

using ArtifactFactValue = std::variant<uint64_t, int64_t, bool, ArtifactF32Bits,
                                       std::vector<uint64_t>, Sha256Digest>;

struct ArtifactSourceSpan {
    ArtifactId artifact_id{};
    uint64_t offset = 0;
    uint64_t length = 0;
    friend bool operator==(const ArtifactSourceSpan&, const ArtifactSourceSpan&) = default;
};

struct ArtifactFact {
    CanonicalFactKey key{};
    ArtifactFactValue value = uint64_t{0};
    ArtifactSourceSpan source;
    ArtifactFactAuthority authority = ArtifactFactAuthority::Declared;
    friend bool operator==(const ArtifactFact&, const ArtifactFact&) = default;
};

struct ArtifactTensorRoleEvidence {
    TensorRole role = TensorRole::TokenEmbedding;
    CanonicalFactKey evidence_key{};
    ArtifactFactAuthority authority = ArtifactFactAuthority::Structural;
    friend bool operator==(const ArtifactTensorRoleEvidence&, const ArtifactTensorRoleEvidence&) = default;
};

struct ArtifactTensorPlane {
    PlaneKind kind = PlaneKind::Values;
    ArtifactScalarType storage_type = ArtifactScalarType::F32;
    ArtifactSourceSpan source;
    uint64_t logical_elements = 0;
    uint32_t bytes_per_block = 0;
    uint32_t elements_per_block = 0;
    uint32_t alignment = 0;
    friend bool operator==(const ArtifactTensorPlane&, const ArtifactTensorPlane&) = default;
};

struct ArtifactTensorRecord {
    uint32_t id = UINT32_MAX;
    ArtifactScalarType logical_type = ArtifactScalarType::F32;
    std::vector<uint64_t> logical_dimensions;
    PhysicalLayout layout;
    Quantization quantization;
    // Optional candidates/proofs for the later semantic resolver. A physical
    // package index is valid without any claimed tensor role.
    std::vector<ArtifactTensorRoleEvidence> role_evidence;
    std::vector<ArtifactTensorPlane> planes;
    friend bool operator==(const ArtifactTensorRecord&, const ArtifactTensorRecord&) = default;
};

enum class ArtifactAliasKind : uint8_t {
    ExactSharedSpan = 1,
    TiedOutput = 2,
};

enum class ArtifactAliasDirection : uint8_t {
    Bidirectional = 1,
    SourceToTarget = 2,
};

struct ArtifactAlias {
    ArtifactAliasKind kind = ArtifactAliasKind::ExactSharedSpan;
    ArtifactAliasDirection direction = ArtifactAliasDirection::Bidirectional;
    uint32_t source_tensor_id = UINT32_MAX;
    uint32_t target_tensor_id = UINT32_MAX;
    TensorRole semantic_role = TensorRole::TokenEmbedding;
    friend bool operator==(const ArtifactAlias&, const ArtifactAlias&) = default;
};

// Raw spellings are retained only for diagnostics. They are deliberately not
// part of canonical bytes or the index digest and are never planner input.
struct ArtifactDiagnostic {
    ArtifactId artifact_id{};
    CanonicalFactKey fact_key{};
    uint32_t tensor_id = UINT32_MAX;
    std::string metadata_spelling;
    std::string tensor_spelling;
};

struct ArtifactIndexInput {
    std::vector<PackageView> artifacts;
    std::vector<ArtifactFact> metadata_facts;
    std::vector<ArtifactFact> package_facts;
    std::vector<ArtifactTensorRecord> tensors;
    std::vector<ArtifactAlias> aliases;
    std::vector<ArtifactDiagnostic> diagnostics;
};

constexpr uint32_t artifact_plane_mask(PlaneKind kind) {
    const uint32_t value = static_cast<uint32_t>(kind);
    return value >= 1 && value <= 32 ? uint32_t{1} << (value - 1) : 0;
}

class ArtifactIndex {
public:
    static std::variant<ArtifactIndex, CompatibilityReport> build(ArtifactIndexInput input);

    std::span<const PackageView> artifacts() const noexcept { return artifacts_; }
    std::span<const ArtifactFact> metadata_facts() const noexcept { return metadata_facts_; }
    std::span<const ArtifactFact> package_facts() const noexcept { return package_facts_; }
    std::span<const ArtifactTensorRecord> tensors() const noexcept { return tensors_; }
    std::span<const ArtifactAlias> aliases() const noexcept { return aliases_; }
    std::span<const ArtifactDiagnostic> diagnostics() const noexcept { return diagnostics_; }
    const std::vector<uint8_t>& canonical_bytes() const noexcept { return canonical_bytes_; }
    const Sha256Digest& digest() const noexcept { return digest_; }

private:
    std::vector<PackageView> artifacts_;
    std::vector<ArtifactFact> metadata_facts_;
    std::vector<ArtifactFact> package_facts_;
    std::vector<ArtifactTensorRecord> tensors_;
    std::vector<ArtifactAlias> aliases_;
    std::vector<ArtifactDiagnostic> diagnostics_;
    std::vector<uint8_t> canonical_bytes_;
    Sha256Digest digest_{};
};

} // namespace Laplace
