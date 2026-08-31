#pragma once

#include <cstdint>
#include <optional>
#include <vector>
#include <variant>

#include "compatibility_report.h"
#include "semantic_model.h"

namespace Laplace {

inline constexpr uint32_t kSemanticDispatchUnresolved = UINT32_MAX;
inline constexpr OperatorKind kSemanticDispatchNoOperator = static_cast<OperatorKind>(0);

enum class SemanticDispatchStepKind : uint8_t {
    Operator = 1,
    GreedySampler = 2,
};

enum class SemanticDispatchStateAccess : uint8_t {
    Read = 1,
    Write = 2,
    ReadWrite = 3,
};

enum class SemanticDispatchSessionEffectKind : uint8_t {
    PositionCandidateAdvance = 1,
    TokenHistoryCandidateAppend = 2,
    CandidateOutputWrite = 3,
};

struct SemanticDispatchSessionEffect {
    SemanticDispatchSessionEffectKind kind =
        SemanticDispatchSessionEffectKind::PositionCandidateAdvance;
    friend bool operator==(const SemanticDispatchSessionEffect&,
                           const SemanticDispatchSessionEffect&) = default;
};

struct SemanticDispatchSamplerBinding {
    uint32_t logits_value_id = kSemanticDispatchUnresolved;
    uint32_t selected_row = kSemanticDispatchUnresolved;
    uint32_t vocabulary_size = 0;
    uint32_t candidate_output_id = kSemanticDispatchUnresolved;
    friend bool operator==(const SemanticDispatchSamplerBinding&,
                           const SemanticDispatchSamplerBinding&) = default;
};

// Exact terminal row that represents this program's logits. This is present
// for language-model output programs whether or not device sampling follows.
struct SemanticDispatchOutputBinding {
    uint32_t logits_value_id = kSemanticDispatchUnresolved;
    uint32_t selected_row = kSemanticDispatchUnresolved;
    uint32_t vocabulary_size = 0;
    uint32_t terminal_operator_id = kSemanticDispatchUnresolved;
    friend bool operator==(const SemanticDispatchOutputBinding&,
                           const SemanticDispatchOutputBinding&) = default;
};

struct SemanticDispatchStateEffect {
    uint32_t state_id = kSemanticDispatchUnresolved;
    SemanticDispatchStateAccess access = SemanticDispatchStateAccess::Read;
    StateUpdateKind update_kind = StateUpdateKind::AppendKey;
    friend bool operator==(const SemanticDispatchStateEffect&,
                           const SemanticDispatchStateEffect&) = default;
};

// This immutable identity contains only normalized semantic contracts. The
// execution plan owns later Metal pipeline slots and certificate bindings.
struct SemanticDispatchRequirement {
    uint16_t version = 1;
    SemanticDispatchStepKind step_kind = SemanticDispatchStepKind::Operator;
    OperatorKind operation = OperatorKind::EmbeddingLookup;
    uint16_t semantic_version = 0;
    ExecutionPhase phase = ExecutionPhase::Decode;
    uint32_t batch_rows = 1;
    NumericalClass numerical_class = NumericalClass::ExactFp32;
    Sha256Digest identity{};
    friend bool operator==(const SemanticDispatchRequirement&,
                           const SemanticDispatchRequirement&) = default;
};

struct SemanticDispatchStep {
    uint32_t ordinal = 0;
    SemanticDispatchStepKind kind = SemanticDispatchStepKind::Operator;
    uint32_t operator_id = kSemanticDispatchUnresolved;
    OperatorKind operation = OperatorKind::EmbeddingLookup;
    std::vector<uint32_t> covered_operator_ids;
    ExecutionPhase phase = ExecutionPhase::Decode;
    NumericalClass numerical_class = NumericalClass::ExactFp32;
    std::vector<uint32_t> input_values;
    std::vector<uint32_t> output_values;
    std::vector<uint32_t> tensor_ids;
    std::vector<SemanticDispatchStateEffect> state_effects;
    std::vector<SemanticDispatchSessionEffect> session_effects;
    std::optional<SemanticDispatchSamplerBinding> sampler_binding;
    // Every step owns one private workspace identity in this first slice.
    // This is an alias contract, not a Metal allocation.
    uint32_t workspace_id = 0;
    SemanticDispatchRequirement requirement;
    uint32_t requirement_index = kSemanticDispatchUnresolved;
    friend bool operator==(const SemanticDispatchStep&, const SemanticDispatchStep&) = default;
};

struct SemanticDispatchLayerView {
    uint32_t layer_index = 0;
    uint32_t first_step = kSemanticDispatchUnresolved;
    uint32_t step_count = 0;
    uint32_t flags = 0;
    friend bool operator==(const SemanticDispatchLayerView&, const SemanticDispatchLayerView&) = default;
};

struct SemanticDispatchRequest {
    ExecutionPhase phase = ExecutionPhase::Decode;
    uint32_t batch_rows = 1;
    NumericalClass numerical_class = NumericalClass::ExactFp32;
    bool include_speculative = false;
    bool include_greedy_sampler = false;
    friend bool operator==(const SemanticDispatchRequest&, const SemanticDispatchRequest&) = default;
};

struct SemanticDispatchProgram {
    Sha256Digest model_digest{};
    Sha256Digest program_digest{};
    SemanticDispatchRequest request;
    std::vector<SemanticDispatchStep> steps;
    // Exact requirements may be shared by multiple steps. Steps remain
    // distinct and preserve graph order even when requirements are equal.
    std::vector<SemanticDispatchRequirement> requirements;
    std::vector<SemanticDispatchLayerView> layer_views;
    std::vector<uint32_t> terminal_operator_ids;
    std::optional<SemanticDispatchOutputBinding> output_binding;
    friend bool operator==(const SemanticDispatchProgram&, const SemanticDispatchProgram&) = default;
};

using SemanticDispatchProgramResult = std::variant<SemanticDispatchProgram, CompatibilityReport>;

SemanticDispatchProgramResult build_semantic_dispatch_program(
    const SemanticModel& model, const SemanticDispatchRequest& request);

// Revalidates a previously built program against the current semantic model
// and request. The optional report receives the first deterministic failure.
bool validate_semantic_dispatch_program(
    const SemanticModel& model, const SemanticDispatchRequest& request,
    const SemanticDispatchProgram& program, CompatibilityReport* failure = nullptr);

} // namespace Laplace
