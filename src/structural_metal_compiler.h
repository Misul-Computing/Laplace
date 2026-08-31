#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "bound_dispatch_requirements.h"
#include "codec_certificate.h"
#include "metal_codec_capability.h"
#include "metal_pipeline_transaction.h"
#include "physical_codec.h"
#include "semantic_dispatch_program.h"
#include "semantic_model.h"

namespace Laplace {

// These IDs are owned by this compiler. They are not source-format or model
// selectors. The transaction later associates each recipe with source bytes.
enum class StructuralMetalLibraryId : uint8_t {
    Core = 1,
    Prefill = 2,
    Sampler = 3,
};

struct StructuralMetalLibraryIdentity {
    StructuralMetalLibraryId id = StructuralMetalLibraryId::Core;
    MetalPipelineDigest source_digest{};

    friend bool operator==(const StructuralMetalLibraryIdentity&,
                           const StructuralMetalLibraryIdentity&) = default;
};

struct StructuralMetalLibraryIdentitySet {
    std::vector<StructuralMetalLibraryIdentity> identities;

    const StructuralMetalLibraryIdentity* find(
        StructuralMetalLibraryId id) const noexcept;
};

enum class StructuralMetalPrimitive : uint8_t {
    EmbeddingF16 = 1,
    EmbeddingQ4 = 2,
    EmbeddingQ6 = 3,
    RmsNormF32 = 4,
    RmsNormNoScale = 5,
    GemvF16 = 6,
    GemvQ4 = 7,
    GemvQ6 = 8,
    GemvAffineU2 = 9,
    PrefillF16Rows = 10,
    VecAdd = 11,
    SwiGlu = 12,
    RopeHalfSplit = 13,
    RopeInterleaved = 14,
    RopeMultiSection = 15,
    KvWrite = 16,
    Attention = 17,
    GatedAttention = 18,
    AxisSplit = 19,
    DnetConvSilu = 20,
    DnetL2 = 21,
    DnetUpdate = 22,
    RouterTopK = 23,
    GemvQ4Expert = 24,
    GatedActivationExperts = 25,
    ExpertReduce = 26,
    ApplyDownScale = 27,
    VecScale = 28,
    SamplerGreedy = 29,
    ColumnGroupedSelect = 30,
    ColumnGroupedPartial = 31,
    ColumnGroupedReduce = 32,
    GemvF32 = 33,
    VecMul = 34,
    RmsNormRowsF32 = 35,
};

enum class StructuralMetalCoverageKind : uint8_t {
    Standalone = 1,
    FusedOwner = 2,
    FusedCovered = 3,
};

struct StructuralMetalSemanticCoverage {
    uint32_t operator_id = kSemanticDispatchUnresolved;
    StructuralMetalCoverageKind kind = StructuralMetalCoverageKind::Standalone;
    uint32_t owner_step_ordinal = kSemanticDispatchUnresolved;
    uint32_t primitive_first = 0;
    uint32_t primitive_count = 0;

    friend bool operator==(const StructuralMetalSemanticCoverage&,
                           const StructuralMetalSemanticCoverage&) = default;
};

struct StructuralMetalPrimitiveInvocation {
    StructuralMetalPrimitive primitive = StructuralMetalPrimitive::VecAdd;
    uint32_t order = 0;
    uint32_t recipe_index = kSemanticDispatchUnresolved;

    friend bool operator==(const StructuralMetalPrimitiveInvocation&,
                           const StructuralMetalPrimitiveInvocation&) = default;
};

enum class StructuralMetalBundleGroupKind : uint8_t {
    Graph = 1,
    Layer = 2,
    Final = 3,
    Sampler = 4,
};

// Structural execution shape derived from the typed semantic operator DAG.
// It is an executor contract, not a model/family or source-format selector.
enum class StructuralMetalExecutionShape : uint8_t {
    GraphEmbedding = 1,
    DenseAttention = 2,
    RecurrentDelta = 3,
    MoeRoutedFeedForward = 4,
    FinalOutput = 5,
    GreedySampler = 6,
};

// An execution group is the bridge to the current composite executor. Its
// step range remains tied to the authoritative semantic program, while its
// ordered primitive list describes every PSO invocation in that group.
class StructuralMetalBundleGroup {
public:
    uint32_t ordinal() const noexcept { return ordinal_; }
    StructuralMetalBundleGroupKind kind() const noexcept { return kind_; }
    StructuralMetalExecutionShape shape() const noexcept { return shape_; }
    uint32_t first_step() const noexcept { return first_step_; }
    uint32_t step_count() const noexcept { return step_count_; }
    uint32_t owner_step_ordinal() const noexcept { return owner_step_ordinal_; }
    ExecutionPhase phase() const noexcept { return phase_; }
    uint32_t batch_rows() const noexcept { return batch_rows_; }
    std::span<const uint32_t> covered_step_ordinals() const noexcept {
        return covered_step_ordinals_;
    }
    std::span<const uint32_t> covered_operator_ids() const noexcept {
        return covered_operator_ids_;
    }
    std::span<const StructuralMetalPrimitiveInvocation> primitives() const noexcept {
        return primitives_;
    }
    std::span<const StructuralMetalSemanticCoverage> coverage() const noexcept {
        return coverage_;
    }
    std::span<const SemanticDispatchStateEffect> state_effects() const noexcept {
        return state_effects_;
    }

    StructuralMetalBundleGroup(
        uint32_t ordinal, StructuralMetalBundleGroupKind kind,
        StructuralMetalExecutionShape shape,
        uint32_t first_step, uint32_t step_count, uint32_t owner_step_ordinal,
        ExecutionPhase phase, uint32_t batch_rows,
        std::vector<uint32_t> covered_step_ordinals,
        std::vector<uint32_t> covered_operator_ids,
        std::vector<StructuralMetalPrimitiveInvocation> primitives,
        std::vector<StructuralMetalSemanticCoverage> coverage,
        std::vector<SemanticDispatchStateEffect> state_effects)
        : ordinal_(ordinal), kind_(kind), shape_(shape), first_step_(first_step),
          step_count_(step_count), owner_step_ordinal_(owner_step_ordinal),
          phase_(phase), batch_rows_(batch_rows),
          covered_step_ordinals_(std::move(covered_step_ordinals)),
          covered_operator_ids_(std::move(covered_operator_ids)),
          primitives_(std::move(primitives)), coverage_(std::move(coverage)),
          state_effects_(std::move(state_effects)) {}

private:
    uint32_t ordinal_ = 0;
    StructuralMetalBundleGroupKind kind_ = StructuralMetalBundleGroupKind::Graph;
    StructuralMetalExecutionShape shape_ = StructuralMetalExecutionShape::GraphEmbedding;
    uint32_t first_step_ = 0;
    uint32_t step_count_ = 0;
    uint32_t owner_step_ordinal_ = kSemanticDispatchUnresolved;
    ExecutionPhase phase_ = ExecutionPhase::Decode;
    uint32_t batch_rows_ = 1;
    std::vector<uint32_t> covered_step_ordinals_;
    std::vector<uint32_t> covered_operator_ids_;
    std::vector<StructuralMetalPrimitiveInvocation> primitives_;
    std::vector<StructuralMetalSemanticCoverage> coverage_;
    std::vector<SemanticDispatchStateEffect> state_effects_;
};

class StructuralMetalBundleStep {
public:
    uint32_t ordinal() const noexcept { return ordinal_; }
    StructuralMetalPrimitiveInvocation primitive(size_t index) const noexcept;
    std::span<const StructuralMetalPrimitiveInvocation> primitives() const noexcept {
        return primitives_;
    }
    std::span<const StructuralMetalSemanticCoverage> coverage() const noexcept {
        return coverage_;
    }
    const SemanticDispatchRequirement& requirement() const noexcept {
        return requirement_;
    }
    const Sha256Digest& bound_digest() const noexcept { return bound_digest_; }
    std::span<const uint32_t> covered_operator_ids() const noexcept {
        return covered_operator_ids_;
    }
    std::span<const uint32_t> input_values() const noexcept { return input_values_; }
    std::span<const uint32_t> output_values() const noexcept { return output_values_; }
    std::span<const uint32_t> tensor_ids() const noexcept { return tensor_ids_; }
    std::span<const SemanticDispatchStateEffect> state_effects() const noexcept {
        return state_effects_;
    }
    ExecutionPhase phase() const noexcept { return phase_; }
    uint32_t batch_rows() const noexcept { return batch_rows_; }
    OperatorKind operation() const noexcept { return operation_; }
    bool fused() const noexcept { return fused_; }

    StructuralMetalBundleStep(
        uint32_t ordinal, SemanticDispatchRequirement requirement,
        Sha256Digest bound_digest, std::vector<uint32_t> covered_operator_ids,
        std::vector<uint32_t> input_values, std::vector<uint32_t> output_values,
        std::vector<uint32_t> tensor_ids,
        std::vector<SemanticDispatchStateEffect> state_effects,
        std::vector<StructuralMetalPrimitiveInvocation> primitives,
        std::vector<StructuralMetalSemanticCoverage> coverage,
        ExecutionPhase phase, uint32_t batch_rows, OperatorKind operation,
        bool fused)
        : ordinal_(ordinal), requirement_(std::move(requirement)),
          bound_digest_(bound_digest),
          covered_operator_ids_(std::move(covered_operator_ids)),
          input_values_(std::move(input_values)),
          output_values_(std::move(output_values)), tensor_ids_(std::move(tensor_ids)),
          state_effects_(std::move(state_effects)), primitives_(std::move(primitives)),
          coverage_(std::move(coverage)), phase_(phase), batch_rows_(batch_rows),
          operation_(operation), fused_(fused) {}

private:
    uint32_t ordinal_ = 0;
    SemanticDispatchRequirement requirement_;
    Sha256Digest bound_digest_{};
    std::vector<uint32_t> covered_operator_ids_;
    std::vector<uint32_t> input_values_;
    std::vector<uint32_t> output_values_;
    std::vector<uint32_t> tensor_ids_;
    std::vector<SemanticDispatchStateEffect> state_effects_;
    std::vector<StructuralMetalPrimitiveInvocation> primitives_;
    std::vector<StructuralMetalSemanticCoverage> coverage_;
    ExecutionPhase phase_ = ExecutionPhase::Decode;
    uint32_t batch_rows_ = 1;
    OperatorKind operation_ = OperatorKind::EmbeddingLookup;
    bool fused_ = false;

    friend class StructuralMetalCompiler;
};

class StructuralMetalProgramBundle {
public:
    StructuralMetalProgramBundle() = default;
    ExecutionPhase phase() const noexcept { return phase_; }
    uint32_t batch_rows() const noexcept { return batch_rows_; }
    const Sha256Digest& program_digest() const noexcept { return program_digest_; }
    std::span<const StructuralMetalBundleStep> steps() const noexcept { return steps_; }
    std::span<const StructuralMetalBundleGroup> groups() const noexcept {
        return groups_;
    }

    StructuralMetalProgramBundle(ExecutionPhase phase, uint32_t batch_rows,
                                 Sha256Digest program_digest,
                                 std::vector<StructuralMetalBundleStep> steps,
                                 std::vector<StructuralMetalBundleGroup> groups)
        : phase_(phase), batch_rows_(batch_rows), program_digest_(program_digest),
          steps_(std::move(steps)), groups_(std::move(groups)) {}

private:
    ExecutionPhase phase_ = ExecutionPhase::Decode;
    uint32_t batch_rows_ = 1;
    Sha256Digest program_digest_{};
    std::vector<StructuralMetalBundleStep> steps_;
    std::vector<StructuralMetalBundleGroup> groups_;

    friend class StructuralMetalCompiler;
};

class StructuralMetalCompilation {
public:
    std::span<const StructuralMetalProgramBundle> programs() const noexcept {
        return programs_;
    }
    std::span<const MetalPipelineRecipe> recipes() const noexcept { return recipes_; }
    const MetalPipelineRecipe* recipe(uint32_t index) const noexcept;
    const MetalPipelineDigest& compilation_digest() const noexcept {
        return compilation_digest_;
    }

private:
    StructuralMetalCompilation(std::vector<StructuralMetalProgramBundle> programs,
                               std::vector<MetalPipelineRecipe> recipes,
                               MetalPipelineDigest compilation_digest)
        : programs_(std::move(programs)), recipes_(std::move(recipes)),
          compilation_digest_(compilation_digest) {}

    std::vector<StructuralMetalProgramBundle> programs_;
    std::vector<MetalPipelineRecipe> recipes_;
    MetalPipelineDigest compilation_digest_{};

    friend class StructuralMetalCompiler;
};

struct StructuralMetalCompilerInput {
    const BoundDispatchRequirements* bound_requirements = nullptr;
    std::span<const SemanticDispatchProgram> programs;
    const SemanticModel* semantic_model = nullptr;
    std::span<const PhysicalCodecSpec> certificate_specs;
    // Portable recipe lookup only. Device facts are validated by the later
    // same-device capability transaction, never synthesized here.
    const MetalCodecCapabilityRegistry* codec_capabilities = nullptr;
    StructuralMetalLibraryIdentitySet libraries;
};

using StructuralMetalCompilerResult =
    std::variant<StructuralMetalCompilation, CompatibilityReport>;
using StructuralMetalCompileResult = StructuralMetalCompilerResult;

StructuralMetalCompilerResult compile_structural_metal(
    const StructuralMetalCompilerInput& input);

StructuralMetalCompilerResult compile_structural_metal(
    const BoundDispatchRequirements& bound_requirements,
    std::span<const SemanticDispatchProgram> programs,
    const SemanticModel& semantic_model,
    std::span<const PhysicalCodecSpec> certificate_specs,
    const StructuralMetalLibraryIdentitySet& libraries,
    const MetalCodecCapabilityRegistry* codec_capabilities = nullptr);

#if defined(LAPLACE_TESTING)
MetalPipelineRecipe structural_metal_primitive_recipe_for_testing(
    StructuralMetalPrimitive primitive);
#endif

} // namespace Laplace
