#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include "codec_binding.h"
#include "semantic_dispatch_program.h"

namespace Laplace {

struct SessionRequest;
class RuntimePackage;

inline constexpr uint16_t kBoundDispatchRequirementsVersionV2 = 2;

// These records contain the exact immutable inputs used to bind one semantic
// dispatch step. They deliberately separate semantic descriptors from source
// span provenance. No record stores a path, name, model family, or pointer.
struct BoundDispatchTensor {
    uint32_t occurrence_index = kSemanticDispatchUnresolved;
    uint32_t tensor_id = kSemanticDispatchUnresolved;
    uint32_t tensor_slot = kSemanticDispatchUnresolved;
    CodecProgramIdentity codec_program_identity{};
    PhysicalCodecIdentity physical_identity{};
    Sha256Digest semantic_tensor_digest{};
    Sha256Digest source_span_digest{};

    friend bool operator==(const BoundDispatchTensor&,
                           const BoundDispatchTensor&) = default;
};

struct BoundDispatchValue {
    uint32_t value_id = kSemanticDispatchUnresolved;
    Sha256Digest descriptor_digest{};

    friend bool operator==(const BoundDispatchValue&,
                           const BoundDispatchValue&) = default;
};

struct BoundDispatchState {
    uint32_t state_id = kSemanticDispatchUnresolved;
    SemanticDispatchStateAccess access = SemanticDispatchStateAccess::Read;
    StateUpdateKind update_kind = StateUpdateKind::AppendKey;
    Sha256Digest descriptor_digest{};

    friend bool operator==(const BoundDispatchState&,
                           const BoundDispatchState&) = default;
};

struct BoundDispatchOutputBinding {
    uint32_t logits_value_id = kSemanticDispatchUnresolved;
    uint32_t selected_row = kSemanticDispatchUnresolved;
    uint32_t vocabulary_size = 0;
    uint32_t terminal_operator_id = kSemanticDispatchUnresolved;
    Sha256Digest descriptor_digest{};

    friend bool operator==(const BoundDispatchOutputBinding&,
                           const BoundDispatchOutputBinding&) = default;
};

struct BoundDispatchWorkspace {
    uint32_t workspace_id = 0;
    Sha256Digest descriptor_digest{};

    friend bool operator==(const BoundDispatchWorkspace&,
                           const BoundDispatchWorkspace&) = default;
};

// A step-aligned binding record. It contains no source names, locations, or
// device facts. The vectors are immutable through this API.
class BoundDispatchStep {
public:
    uint16_t version() const noexcept { return version_; }
    uint32_t ordinal() const noexcept { return ordinal_; }
    const SemanticDispatchRequirement& requirement() const noexcept { return requirement_; }
    const Sha256Digest& bound_digest() const noexcept { return bound_digest_; }
    const Sha256Digest& binding_digest() const noexcept { return bound_digest_; }
    std::span<const BoundDispatchTensor> tensors() const noexcept { return tensors_; }
    std::span<const BoundDispatchValue> input_values() const noexcept {
        return input_values_;
    }
    std::span<const BoundDispatchValue> output_values() const noexcept {
        return output_values_;
    }
    std::span<const BoundDispatchState> states() const noexcept { return states_; }
    std::span<const SemanticDispatchSessionEffect> session_effects() const noexcept {
        return session_effects_;
    }
    const std::optional<SemanticDispatchSamplerBinding>& sampler_binding() const noexcept {
        return sampler_binding_;
    }
    const BoundDispatchWorkspace& workspace() const noexcept { return workspace_; }

    // Compatibility accessors for the current structural compiler. Product
    // code must consume tensors() and validate the v2 record directly.
    std::vector<uint32_t> codec_occurrence_indices() const {
        std::vector<uint32_t> result;
        result.reserve(tensors_.size());
        for (const BoundDispatchTensor& tensor : tensors_)
            result.push_back(tensor.occurrence_index);
        return result;
    }
    std::vector<CodecProgramIdentity> codec_program_identities() const {
        std::vector<CodecProgramIdentity> result;
        result.reserve(tensors_.size());
        for (const BoundDispatchTensor& tensor : tensors_)
            result.push_back(tensor.codec_program_identity);
        return result;
    }
    std::vector<PhysicalCodecIdentity> physical_identities() const {
        std::vector<PhysicalCodecIdentity> result;
        result.reserve(tensors_.size());
        for (const BoundDispatchTensor& tensor : tensors_)
            result.push_back(tensor.physical_identity);
        return result;
    }

private:
    BoundDispatchStep(uint32_t ordinal, SemanticDispatchRequirement requirement,
                      Sha256Digest bound_digest,
                      std::vector<BoundDispatchTensor> tensors,
                      std::vector<BoundDispatchValue> input_values,
                      std::vector<BoundDispatchValue> output_values,
                      std::vector<BoundDispatchState> states,
                      std::vector<SemanticDispatchSessionEffect> session_effects,
                      std::optional<SemanticDispatchSamplerBinding> sampler_binding,
                      BoundDispatchWorkspace workspace)
        : version_(kBoundDispatchRequirementsVersionV2), ordinal_(ordinal),
          requirement_(std::move(requirement)),
          bound_digest_(bound_digest),
          tensors_(std::move(tensors)), input_values_(std::move(input_values)),
          output_values_(std::move(output_values)), states_(std::move(states)),
          session_effects_(std::move(session_effects)),
          sampler_binding_(std::move(sampler_binding)),
          workspace_(std::move(workspace)) {}

    friend class BoundDispatchRequirementsBuilder;
    uint16_t version_ = kBoundDispatchRequirementsVersionV2;
    uint32_t ordinal_ = 0;
    SemanticDispatchRequirement requirement_;
    Sha256Digest bound_digest_{};
    std::vector<BoundDispatchTensor> tensors_;
    std::vector<BoundDispatchValue> input_values_;
    std::vector<BoundDispatchValue> output_values_;
    std::vector<BoundDispatchState> states_;
    std::vector<SemanticDispatchSessionEffect> session_effects_;
    std::optional<SemanticDispatchSamplerBinding> sampler_binding_;
    BoundDispatchWorkspace workspace_;
};

class BoundDispatchProgram {
public:
    uint16_t version() const noexcept { return version_; }
    const Sha256Digest& program_digest() const noexcept { return program_digest_; }
    const Sha256Digest& bound_digest() const noexcept { return bound_digest_; }
    const Sha256Digest& binding_digest() const noexcept { return bound_digest_; }
    const SemanticDispatchRequest& request() const noexcept { return request_; }
    std::span<const BoundDispatchStep> steps() const noexcept { return steps_; }
    const std::optional<BoundDispatchOutputBinding>& output_binding() const noexcept {
        return output_binding_;
    }

private:
    BoundDispatchProgram(Sha256Digest program_digest, SemanticDispatchRequest request,
                         std::vector<BoundDispatchStep> steps,
                         std::optional<BoundDispatchOutputBinding> output_binding,
                         Sha256Digest bound_digest)
        : version_(kBoundDispatchRequirementsVersionV2),
          program_digest_(program_digest), request_(request),
          steps_(std::move(steps)), output_binding_(std::move(output_binding)),
          bound_digest_(bound_digest) {}

    friend class BoundDispatchRequirementsBuilder;
    uint16_t version_ = kBoundDispatchRequirementsVersionV2;
    Sha256Digest program_digest_{};
    SemanticDispatchRequest request_;
    std::vector<BoundDispatchStep> steps_;
    std::optional<BoundDispatchOutputBinding> output_binding_;
    Sha256Digest bound_digest_{};
};

class BoundDispatchRequirements {
public:
    uint16_t version() const noexcept { return version_; }
    const Sha256Digest& package_fingerprint() const noexcept { return package_fingerprint_; }
    const Sha256Digest& bound_digest() const noexcept { return bound_digest_; }
    const Sha256Digest& binding_digest() const noexcept { return bound_digest_; }
    std::span<const BoundDispatchProgram> programs() const noexcept { return programs_; }

private:
    BoundDispatchRequirements(Sha256Digest package_fingerprint,
                              std::vector<BoundDispatchProgram> programs,
                              Sha256Digest bound_digest)
        : version_(kBoundDispatchRequirementsVersionV2),
          package_fingerprint_(package_fingerprint), programs_(std::move(programs)),
          bound_digest_(bound_digest) {}

    friend class BoundDispatchRequirementsBuilder;
    uint16_t version_ = kBoundDispatchRequirementsVersionV2;
    Sha256Digest package_fingerprint_{};
    std::vector<BoundDispatchProgram> programs_;
    Sha256Digest bound_digest_{};
};

using BoundDispatchRequirementsResult =
    std::variant<BoundDispatchRequirements, CompatibilityReport>;

// Revalidates immutable records at an ownership boundary. A null result is
// success; a report rejects v1, zero-digest, or tampered records.
std::optional<CompatibilityReport> validate_bound_dispatch_step(
    const BoundDispatchStep& step);
std::optional<CompatibilityReport> validate_bound_dispatch_program(
    const BoundDispatchProgram& program);
std::optional<CompatibilityReport> validate_bound_dispatch_requirements(
    const BoundDispatchRequirements& requirements);

// Binds canonical semantic dispatch to the exact package codec occurrences.
// The result is all-or-nothing and has one record for every program step.
BoundDispatchRequirementsResult bind_dispatch_requirements(
    const RuntimePackage& package, const ResolvedCodecBindings& bindings,
    const SessionRequest& session_request,
    std::span<const SemanticDispatchProgram> programs);

} // namespace Laplace
