#pragma once

#include <array>
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
    None = 0,
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

enum class ArtifactPhysicalEncoding : uint16_t {
    Unknown = 0,
    F32 = 1,
    F16 = 2,
    Q4_K = 3,
    Q5_0 = 4,
    Q6_K = 5,
    Q8_0 = 6,
    Q4_0 = 7,
    // Source-neutral MLX-compatible affine tuple: little-endian uint32 words,
    // 2 LSB-first bits/value, 256 values/group, FP16 scale and FP16 bias.
    GroupedAffineU2_256 = 8,
    // Output-block-major UInt2 skip tuple. Its group order is
    // (output_row / 256) * K + input_column and is a separate ABI from the
    // legacy GroupedAffineU2_256 input-block-major contract.
    ColumnGroupedAffineU2Skip256 = 9,
};

// Exact GGUF wire contract for one tensor values plane. The auxiliary fields
// describe metadata stored inside the packed block, not separate planes.
struct ArtifactPhysicalFormat {
    uint16_t version = 0;
    ArtifactPhysicalEncoding encoding = ArtifactPhysicalEncoding::Unknown;
    ArtifactScalarType value_type = ArtifactScalarType::None;
    ArtifactScalarType scale_type = ArtifactScalarType::None;
    ArtifactScalarType zero_type = ArtifactScalarType::None;
    ArtifactScalarType subscale_type = ArtifactScalarType::None;
    ArtifactScalarType subzero_type = ArtifactScalarType::None;
    uint32_t block_elements = 0;
    uint32_t block_bytes = 0;
    uint32_t scale_bytes = 0;
    uint32_t zero_bytes = 0;
    uint32_t subscale_bytes = 0;
    uint32_t subzero_bytes = 0;
    ArtifactScalarType bias_type = ArtifactScalarType::None;
    uint32_t bias_bytes = 0;
    friend bool operator==(const ArtifactPhysicalFormat&, const ArtifactPhysicalFormat&) = default;
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

using ArtifactF32Vector = std::vector<ArtifactF32Bits>;
using ArtifactI64Vector = std::vector<int64_t>;

using ArtifactFactValue = std::variant<uint64_t, int64_t, bool, ArtifactF32Bits,
                                       std::vector<uint64_t>, ArtifactF32Vector,
                                       ArtifactI64Vector, Sha256Digest>;

enum class ArtifactFactState : uint8_t {
    Present = 1,
    Missing = 2,
    WrongType = 3,
    Malformed = 4,
    Ambiguous = 5,
};

struct ArtifactCoordinate {
    uint32_t root = 0;
    uint32_t layer = UINT32_MAX;
    uint32_t slot = UINT32_MAX;
    uint32_t instance = UINT32_MAX;
    uint32_t expert = UINT32_MAX;
    uint8_t bank_axis = UINT8_MAX;
    uint32_t bank_extent = 0;
    uint64_t bank_stride = 0;
    friend bool operator==(const ArtifactCoordinate&, const ArtifactCoordinate&) = default;
};

struct ArtifactPhysicalAxisContract {
    uint8_t source_rank = 0;
    std::array<uint8_t, 8> source_axis_order =
        {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    uint8_t block_axis = UINT8_MAX;
    uint32_t block_elements = 0;
    uint32_t bytes_per_block = 0;
    uint64_t row_stride_bytes = 0;
    uint32_t plane_order = 0;
    friend bool operator==(const ArtifactPhysicalAxisContract&, const ArtifactPhysicalAxisContract&) = default;
};

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
    ArtifactFactState state = ArtifactFactState::Present;
    ArtifactCoordinate scope;
    friend bool operator==(const ArtifactFact&, const ArtifactFact&) = default;
};

struct ArtifactTensorRoleEvidence {
    TensorRole role = TensorRole::TokenEmbedding;
    CanonicalFactKey evidence_key{};
    ArtifactFactAuthority authority = ArtifactFactAuthority::Structural;
    ArtifactCoordinate scope;
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
    ArtifactCoordinate coordinate;
    ArtifactPhysicalAxisContract axis;
    ArtifactPhysicalFormat format;
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
    CanonicalFactKey proof_key{};
    ArtifactFactAuthority proof_authority = ArtifactFactAuthority::Declared;
    ArtifactCoordinate proof_scope;
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
    const Sha256Digest& normalized_digest() const noexcept { return normalized_digest_; }
    const Sha256Digest& provenance_digest() const noexcept { return provenance_digest_; }

private:
    std::vector<PackageView> artifacts_;
    std::vector<ArtifactFact> metadata_facts_;
    std::vector<ArtifactFact> package_facts_;
    std::vector<ArtifactTensorRecord> tensors_;
    std::vector<ArtifactAlias> aliases_;
    std::vector<ArtifactDiagnostic> diagnostics_;
    std::vector<uint8_t> canonical_bytes_;
    Sha256Digest digest_{};
    Sha256Digest normalized_digest_{};
    Sha256Digest provenance_digest_{};
};

} // namespace Laplace
