#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "artifact_set.h"
#include "artifact_index.h"
#include "codec_program.h"
#include "compatibility_report.h"
#include "physical_codec.h"
#include "semantic_model.h"

namespace Laplace {

class RuntimePackage;
struct CodecBindingBuilder;

// A copied, package-bound plane descriptor. It has no pointer into the
// package and retains the artifact digest and size needed by later consumers
// to reject replay against a different package.
class ResolvedCodecPlane {
public:
    PlaneKind kind() const noexcept { return kind_; }
    ArtifactScalarType storage_type() const noexcept { return storage_type_; }
    ScalarType semantic_storage_type() const noexcept { return semantic_storage_type_; }
    ArtifactId artifact_id() const noexcept { return artifact_id_; }
    uint64_t offset() const noexcept { return offset_; }
    uint64_t length() const noexcept { return length_; }
    uint32_t alignment() const noexcept { return alignment_; }
    uint32_t flags() const noexcept { return flags_; }
    uint64_t logical_elements() const noexcept { return logical_elements_; }
    uint32_t bytes_per_block() const noexcept { return bytes_per_block_; }
    uint32_t elements_per_block() const noexcept { return elements_per_block_; }
    const Sha256Digest& artifact_digest() const noexcept { return artifact_digest_; }
    uint64_t artifact_size() const noexcept { return artifact_size_; }

private:
    ResolvedCodecPlane(PlaneKind kind, ArtifactScalarType storage_type,
                       ScalarType semantic_storage_type, ArtifactId artifact_id,
                       uint64_t offset, uint64_t length, uint32_t alignment,
                       uint32_t flags, uint64_t logical_elements,
                       uint32_t bytes_per_block, uint32_t elements_per_block,
                       Sha256Digest artifact_digest, uint64_t artifact_size)
        : kind_(kind), storage_type_(storage_type), semantic_storage_type_(semantic_storage_type),
          artifact_id_(artifact_id),
          offset_(offset), length_(length), alignment_(alignment), flags_(flags),
          logical_elements_(logical_elements), bytes_per_block_(bytes_per_block),
          elements_per_block_(elements_per_block), artifact_digest_(artifact_digest),
          artifact_size_(artifact_size) {}

    friend struct CodecBindingBuilder;
    PlaneKind kind_;
    ArtifactScalarType storage_type_;
    ScalarType semantic_storage_type_;
    ArtifactId artifact_id_;
    uint64_t offset_;
    uint64_t length_;
    uint32_t alignment_;
    uint32_t flags_;
    uint64_t logical_elements_;
    uint32_t bytes_per_block_;
    uint32_t elements_per_block_;
    Sha256Digest artifact_digest_;
    uint64_t artifact_size_;
};

class ResolvedCodecTensor {
public:
    uint32_t occurrence_index() const noexcept { return occurrence_index_; }
    uint32_t tensor_slot() const noexcept { return tensor_slot_; }
    uint32_t tensor_id() const noexcept { return tensor_id_; }
    const PhysicalCodecIdentity& physical_identity() const noexcept { return physical_identity_; }
    const CodecProgramIdentity& program_identity() const noexcept { return program_identity_; }
    std::span<const Dimension> dimensions() const noexcept { return dimensions_; }
    const std::array<uint64_t, 8>& strides() const noexcept { return strides_; }
    std::span<const ResolvedCodecPlane> planes() const noexcept { return planes_; }

private:
    ResolvedCodecTensor(uint32_t occurrence_index, uint32_t tensor_slot, uint32_t tensor_id,
                        PhysicalCodecIdentity physical_identity,
                        CodecProgramIdentity program_identity,
                        std::vector<Dimension> dimensions,
                        std::array<uint64_t, 8> strides,
                        std::vector<ResolvedCodecPlane> planes)
        : occurrence_index_(occurrence_index), tensor_slot_(tensor_slot), tensor_id_(tensor_id),
          physical_identity_(std::move(physical_identity)),
          program_identity_(program_identity), dimensions_(std::move(dimensions)),
          strides_(strides), planes_(std::move(planes)) {}

    friend struct CodecBindingBuilder;
    uint32_t occurrence_index_;
    uint32_t tensor_slot_;
    uint32_t tensor_id_;
    PhysicalCodecIdentity physical_identity_;
    CodecProgramIdentity program_identity_;
    std::vector<Dimension> dimensions_;
    std::array<uint64_t, 8> strides_{};
    std::vector<ResolvedCodecPlane> planes_;
};

class ResolvedCodecOperator {
public:
    uint32_t operator_id() const noexcept { return operator_id_; }
    std::span<const ResolvedCodecTensor> tensors() const noexcept { return tensors_; }

private:
    ResolvedCodecOperator(uint32_t operator_id, std::vector<ResolvedCodecTensor> tensors)
        : operator_id_(operator_id), tensors_(std::move(tensors)) {}

    friend struct CodecBindingBuilder;
    uint32_t operator_id_;
    std::vector<ResolvedCodecTensor> tensors_;
};

// Immutable-by-API bindings. Construction is internal to preflight; all
// public accessors expose const views, and every result is all-or-nothing.
class ResolvedCodecBindings {
public:
    const Sha256Digest& package_fingerprint() const noexcept { return package_fingerprint_; }
    std::span<const ResolvedCodecOperator> operators() const noexcept { return operators_; }

    // Verifies both the immutable package fingerprint and each copied source
    // artifact witness before a later consumer reuses these bindings.
    bool matches_package(const RuntimePackage& package) const noexcept;

private:
    ResolvedCodecBindings(Sha256Digest package_fingerprint,
                          std::vector<ResolvedCodecOperator> operators)
        : package_fingerprint_(package_fingerprint), operators_(std::move(operators)) {}

    friend struct CodecBindingBuilder;

    Sha256Digest package_fingerprint_;
    std::vector<ResolvedCodecOperator> operators_;
};

// Pure package admission. It performs no Metal queries, allocation of device
// resources, or decoder execution.
static_assert(!std::is_aggregate_v<ResolvedCodecPlane>);
static_assert(!std::is_aggregate_v<ResolvedCodecTensor>);
static_assert(!std::is_aggregate_v<ResolvedCodecOperator>);
static_assert(!std::is_aggregate_v<ResolvedCodecBindings>);
static_assert(!std::is_default_constructible_v<ResolvedCodecPlane>);
static_assert(!std::is_default_constructible_v<ResolvedCodecTensor>);
static_assert(!std::is_default_constructible_v<ResolvedCodecOperator>);
static_assert(!std::is_default_constructible_v<ResolvedCodecBindings>);

using CodecBindingPreflightResult = std::variant<ResolvedCodecBindings, CompatibilityReport>;

// Product admission resolves and validates the canonical certificates carried
// by the package. It does not require an application-global format registry.
CodecBindingPreflightResult preflight_codec_bindings(const RuntimePackage& package);

} // namespace Laplace
