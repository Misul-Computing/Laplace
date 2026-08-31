#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include "artifact_index.h"

namespace Laplace {

inline constexpr uint16_t kNormalizedSourceEvidenceVersion = 1;
inline constexpr size_t kNormalizedSourceEvidenceMaximumCandidates = 4096;
inline constexpr size_t kNormalizedSourceEvidenceMaximumDimensions = 8;
inline constexpr size_t kNormalizedSourceEvidenceMaximumPlanes = 8;

enum class NormalizedEvidenceStrength : uint8_t {
    Declared = 1,
    Structural = 2,
    Derived = 3,
};

// This record contains typed facts only. It has no source spelling, source
// format, model identity, artifact identity, or runtime policy.
struct NormalizedPlaneEvidence {
    PlaneKind kind = PlaneKind::Values;
    ArtifactScalarType storage_type = ArtifactScalarType::None;
    uint64_t logical_elements = 0;
    uint32_t bytes_per_block = 0;
    uint32_t elements_per_block = 0;
    uint32_t alignment = 0;
    friend bool operator==(const NormalizedPlaneEvidence&, const NormalizedPlaneEvidence&) = default;
};

struct NormalizedTensorEvidence {
    TensorRole role = TensorRole::TokenEmbedding;
    ArtifactCoordinate coordinate;
    NormalizedEvidenceStrength strength = NormalizedEvidenceStrength::Structural;
    ArtifactScalarType logical_type = ArtifactScalarType::None;
    std::vector<uint64_t> dimensions;
    PhysicalLayout layout;
    ArtifactPhysicalFormat format;
    Quantization quantization;
    std::vector<NormalizedPlaneEvidence> planes;
    friend bool operator==(const NormalizedTensorEvidence&, const NormalizedTensorEvidence&) = default;
};

// Binding coordinates are retained for the later compiler, but are kept out
// of canonical semantic bytes so equal typed evidence from two containers has
// equal identity even when their file offsets differ.
struct NormalizedTensorBinding {
    uint32_t ordinal = UINT32_MAX;
    uint32_t source_tensor_id = UINT32_MAX;
    std::vector<ArtifactSourceSpan> planes;
    friend bool operator==(const NormalizedTensorBinding&, const NormalizedTensorBinding&) = default;
};

struct NormalizedEvidenceCandidate {
    NormalizedTensorEvidence evidence;
    NormalizedTensorBinding binding;
};

class NormalizedSourceEvidence {
public:
    uint16_t version() const noexcept { return version_; }
    std::span<const NormalizedTensorEvidence> tensors() const noexcept { return tensors_; }
    std::span<const NormalizedTensorBinding> bindings() const noexcept { return bindings_; }
    std::span<const uint8_t> canonical_bytes() const noexcept { return canonical_bytes_; }
    const Sha256Digest& canonical_digest() const noexcept { return canonical_digest_; }

private:
    friend std::variant<NormalizedSourceEvidence, CompatibilityReport>
    normalize_source_evidence(std::span<const NormalizedEvidenceCandidate> candidates);

    uint16_t version_ = kNormalizedSourceEvidenceVersion;
    std::vector<NormalizedTensorEvidence> tensors_;
    std::vector<NormalizedTensorBinding> bindings_;
    std::vector<uint8_t> canonical_bytes_;
    Sha256Digest canonical_digest_{};
};

using NormalizedSourceEvidenceResult =
    std::variant<NormalizedSourceEvidence, CompatibilityReport>;

NormalizedSourceEvidenceResult
normalize_source_evidence(std::span<const NormalizedEvidenceCandidate> candidates);

} // namespace Laplace
