#include <algorithm>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "semantic_dispatch_program.h"
#include "test_util.h"

using namespace Laplace;

namespace {

SemanticValue value(uint32_t id, uint32_t width, ScalarType type = ScalarType::F32) {
    return {id, type, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, width}}, 0};
}

SemanticTensor vector_tensor(uint32_t id, TensorRole role, uint32_t width, uint64_t offset) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, width}};
    tensor.layout.rank = 1;
    tensor.layout.axis_order = {0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides[0] = 1;
    tensor.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, offset,
                      static_cast<uint64_t>(width) * sizeof(float), 64, 0}};
    return tensor;
}

SemanticTensor matrix_tensor(uint32_t id, TensorRole role, uint32_t input,
                             uint32_t output, uint64_t offset) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, input}, {DimensionKind::Constant, output}};
    tensor.layout.rank = 2;
    tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides[0] = output;
    tensor.layout.strides[1] = 1;
    tensor.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, offset,
                      static_cast<uint64_t>(input) * output * sizeof(float), 64, 0}};
    return tensor;
}

SemanticModel dense_model() {
    constexpr uint32_t width = 4;
    constexpr uint32_t vocabulary = 3;
    SemanticModel model;
    model.maximum_context = 32768;
    model.vocabulary_size = vocabulary;
    model.bos_id = 0;
    model.eos_id = 2;
    model.stop_ids = {2};
    model.tensors = {
        matrix_tensor(0, TensorRole::TokenEmbedding, vocabulary, width, 0),
        vector_tensor(1, TensorRole::AttentionNormWeight, width, 256),
        matrix_tensor(2, TensorRole::QueryWeight, width, width, 512),
        matrix_tensor(3, TensorRole::KeyWeight, width, width, 768),
        matrix_tensor(4, TensorRole::ValueWeight, width, width, 1024),
        matrix_tensor(5, TensorRole::AttentionOutputWeight, width, width, 1280),
        vector_tensor(6, TensorRole::FfnNormWeight, width, 1536),
        matrix_tensor(7, TensorRole::FfnGateWeight, width, width, 1792),
        matrix_tensor(8, TensorRole::FfnUpWeight, width, width, 2048),
        matrix_tensor(9, TensorRole::FfnDownWeight, width, width, 2304),
        vector_tensor(10, TensorRole::FinalNormWeight, width, 2560),
        matrix_tensor(11, TensorRole::OutputWeight, width, vocabulary, 2816),
    };
    for (uint32_t id = 0; id != 18; ++id) model.values.push_back(value(id, id == 17 ? vocabulary : width));
    model.values.push_back({18, ScalarType::U32, {{DimensionKind::Symbol, 1}}, 0});
    model.input_values_first = 18;
    model.input_values_count = 1;
    model.output_values_first = 17;
    model.output_values_count = 1;
    const auto add = [&](OperatorKind kind, std::vector<uint32_t> inputs,
                         std::vector<uint32_t> outputs, std::vector<uint32_t> tensors,
                         std::vector<uint32_t> states, OperatorPayload payload) {
        SemanticOperator op;
        op.id = static_cast<uint32_t>(model.operators.size());
        op.kind = kind;
        op.semantic_version = 1;
        op.inputs = std::move(inputs);
        op.outputs = std::move(outputs);
        op.tensors = std::move(tensors);
        op.states = std::move(states);
        op.payload = std::move(payload);
        model.operators.push_back(std::move(op));
    };
    constexpr uint32_t epsilon = 0x358637bdu;
    add(OperatorKind::EmbeddingLookup, {18}, {0}, {0}, {},
        EmbeddingLookupPayload{0x3f800000u, vocabulary, width, 0});
    add(OperatorKind::RmsNorm, {0}, {1}, {1}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {1}, {2}, {2}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {3}, {3}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {4}, {4}, {}, LinearPayload{});
    add(OperatorKind::Rope, {2, 3}, {5, 6}, {}, {},
        RopePayload{RopePairing::HalfSplit, true, width, 0x49742400u, 0x3f800000u});
    add(OperatorKind::CausalAttention, {5, 6, 4}, {7}, {}, {0, 1},
        CausalAttentionPayload{1, 1, width, 0x3e800000u, AttentionMask::Causal,
                                CachePolicy::Global});
    add(OperatorKind::Linear, {7}, {8}, {5}, {}, LinearPayload{});
    add(OperatorKind::Add, {0, 8}, {9}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {9}, {10}, {6}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {10}, {11}, {7}, {}, LinearPayload{});
    add(OperatorKind::Linear, {10}, {12}, {8}, {}, LinearPayload{});
    add(OperatorKind::SwiGlu, {11, 12}, {13}, {}, {}, SwiGluPayload{});
    add(OperatorKind::Linear, {13}, {14}, {9}, {}, LinearPayload{});
    add(OperatorKind::Add, {9, 14}, {15}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {15}, {16}, {10}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {16}, {17}, {11}, {}, LinearPayload{});
    model.layers = {{0, 1, 14, 0}};
    StateFormat key_format;
    key_format.encoded_domain = TransformDomain::RopeApplied;
    key_format.alignment = 64;
    SemanticState key_state;
    key_state.id = 0;
    key_state.kind = StateKind::KeyCache;
    key_state.update_kind = StateUpdateKind::AppendKey;
    key_state.dimensions = {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1},
                            {DimensionKind::Constant, width}};
    key_state.formats = {key_format};
    SemanticState value_state = key_state;
    value_state.id = 1;
    value_state.kind = StateKind::ValueCache;
    value_state.update_kind = StateUpdateKind::AppendValue;
    value_state.formats[0].encoded_domain = TransformDomain::Untransformed;
    model.states = {key_state, value_state};
    return model;
}

SemanticModel recurrent_conv_model() {
    SemanticModel model;
    model.schema_major = 3;
    model.opset_major = 3;
    model.maximum_context = 4;
    model.vocabulary_size = 2;
    model.bos_id = 0;
    model.eos_id = 1;
    model.stop_ids = {1};
    model.values = {
        value(0, 6), value(1, 2), value(2, 2), value(3, 2),
    };
    model.input_values_first = 0;
    model.input_values_count = 1;
    model.output_values_first = 1;
    model.output_values_count = 3;
    SemanticTensor weight = matrix_tensor(0, TensorRole::RecurrentConvWeight, 6, 2, 0);
    model.tensors = {weight};
    SemanticOperator op;
    op.id = 0;
    op.kind = OperatorKind::DepthwiseConvSilu;
    op.semantic_version = 3;
    op.inputs = {0};
    op.outputs = {1, 2, 3};
    op.tensors = {0};
    op.states = {0};
    op.payload = DepthwiseConvSiluPayload{1, 1, 2, 2};
    model.operators = {op};
    model.layers = {{0, 0, 1, 0}};
    StateFormat format;
    format.kind = StateFormatKind::RecurrentContiguous;
    format.logical_type = ScalarType::F32;
    format.encoded_type = ScalarType::F32;
    format.codec = CodecKind::Fp32;
    format.cache_policy = CachePolicy::Recurrent;
    format.layout_policy = LayoutPolicy::ChannelMajorHistory;
    format.alignment = 64;
    SemanticState state;
    state.id = 0;
    state.kind = StateKind::RecurrentConvHistory;
    state.semantic_version = 3;
    state.update_kind = StateUpdateKind::ShiftHistory;
    state.position_policy = PositionPolicy::ReplaceAtCursor;
    state.dimensions = {{DimensionKind::Constant, 6}, {DimensionKind::Constant, 1}};
    state.formats = {format};
    model.states = {state};
    return model;
}

bool rejected(const SemanticModel& model, SemanticDispatchProgram program) {
    CompatibilityReport report;
    return !validate_semantic_dispatch_program(model, SemanticDispatchRequest{}, program, &report) &&
           report.code == CompatibilityError::IR_REFERENCE_INVALID;
}

void test_dense_program() {
    const SemanticModel model = dense_model();
    const SemanticDispatchRequest request;
    const auto result = build_semantic_dispatch_program(model, request);
    CHECK(std::holds_alternative<SemanticDispatchProgram>(result));
    if (!std::holds_alternative<SemanticDispatchProgram>(result)) return;
    const auto& program = std::get<SemanticDispatchProgram>(result);
    CHECK(program.steps.size() == 17);
    CHECK(program.requirements.size() < program.steps.size());
    CHECK(program.steps.front().operator_id == 0);
    CHECK(program.steps.back().operator_id == 16);
    CHECK(program.steps.front().covered_operator_ids == std::vector<uint32_t>{0});
    CHECK(program.steps[6].state_effects.size() == 2);
    CHECK(program.steps[6].state_effects[0].state_id == 0);
    CHECK(program.steps[6].state_effects[0].update_kind == StateUpdateKind::AppendKey);
    CHECK(program.steps[6].requirement.batch_rows == 1);
    CHECK(program.layer_views.size() == 1);
    CHECK(program.layer_views[0].first_step == 1);
    CHECK(program.layer_views[0].step_count == 14);
    const std::vector<uint32_t> terminals = {6, 16};
    CHECK(program.terminal_operator_ids == terminals);
    CHECK(program.output_binding.has_value());
    if (program.output_binding) {
        CHECK(program.output_binding->logits_value_id == 17);
        CHECK(program.output_binding->selected_row == 0);
        CHECK(program.output_binding->vocabulary_size == 3);
        CHECK(program.output_binding->terminal_operator_id == 16);
    }
    CHECK(validate_semantic_dispatch_program(model, request, program));
    const auto repeated = build_semantic_dispatch_program(model, request);
    CHECK(std::holds_alternative<SemanticDispatchProgram>(repeated));
    if (std::holds_alternative<SemanticDispatchProgram>(repeated)) {
        CHECK(std::get<SemanticDispatchProgram>(repeated).program_digest == program.program_digest);
        CHECK(std::get<SemanticDispatchProgram>(repeated).model_digest == program.model_digest);
    }
}

void test_program_mutations() {
    const SemanticModel model = dense_model();
    const auto result = build_semantic_dispatch_program(model, {});
    CHECK(std::holds_alternative<SemanticDispatchProgram>(result));
    if (!std::holds_alternative<SemanticDispatchProgram>(result)) return;
    const auto original = std::get<SemanticDispatchProgram>(result);
    auto mutated = original;
    mutated.steps.erase(mutated.steps.begin() + 4);
    CHECK(rejected(model, std::move(mutated)));
    mutated = original;
    mutated.steps[4].covered_operator_ids.push_back(3);
    CHECK(rejected(model, std::move(mutated)));
    mutated = original;
    mutated.steps[4].input_values.push_back(99);
    CHECK(rejected(model, std::move(mutated)));
    mutated = original;
    std::swap(mutated.steps[15], mutated.steps[16]);
    CHECK(rejected(model, std::move(mutated)));
    mutated = original;
    std::swap(mutated.terminal_operator_ids[0], mutated.terminal_operator_ids[1]);
    CHECK(rejected(model, std::move(mutated)));
    mutated = original;
    mutated.steps[6].state_effects[0].update_kind = StateUpdateKind::AppendValue;
    CHECK(rejected(model, std::move(mutated)));
    mutated = original;
    mutated.steps[5].workspace_id = mutated.steps[4].workspace_id;
    CHECK(rejected(model, std::move(mutated)));
    mutated = original;
    mutated.steps[4].ordinal = mutated.steps[3].ordinal;
    CHECK(rejected(model, std::move(mutated)));
    mutated = original;
    mutated.requirements[0].identity = {};
    CHECK(rejected(model, std::move(mutated)));
    mutated = original;
    CHECK(mutated.output_binding.has_value());
    if (mutated.output_binding) {
        mutated.output_binding->selected_row = 1;
        CHECK(rejected(model, std::move(mutated)));
    }
}

void test_speculation_and_sampler() {
    SemanticModel model = dense_model();
    model.schema_major = 6;
    model.opset_major = 6;
    for (SemanticOperator& op : model.operators) op.semantic_version = 6;
    for (SemanticState& state : model.states) state.semantic_version = 6;
    model.values.push_back(value(19, 4));
    SemanticOperator speculative;
    speculative.id = static_cast<uint32_t>(model.operators.size());
    speculative.kind = OperatorKind::EmbeddingLookup;
    speculative.semantic_version = 6;
    speculative.inputs = {18};
    speculative.outputs = {19};
    speculative.tensors = {0};
    speculative.payload = EmbeddingLookupPayload{0x3f800000u, 3, 4, 0};
    model.operators.push_back(speculative);
    model.layers.push_back({1, 17, 1, kSemanticLayerFlagSpeculative});
    auto result = build_semantic_dispatch_program(model, {});
    CHECK(std::holds_alternative<SemanticDispatchProgram>(result));
    if (std::holds_alternative<SemanticDispatchProgram>(result)) {
        const auto& program = std::get<SemanticDispatchProgram>(result);
        CHECK(program.steps.size() == 17);
        CHECK(program.layer_views.size() == 2);
        CHECK(program.layer_views[1].first_step == kSemanticDispatchUnresolved);
        CHECK(program.layer_views[1].step_count == 0);
    }
    SemanticDispatchRequest request;
    request.include_speculative = true;
    result = build_semantic_dispatch_program(model, request);
    CHECK(std::holds_alternative<CompatibilityReport>(result));
    request.include_speculative = false;
    request.include_greedy_sampler = true;
    result = build_semantic_dispatch_program(model, request);
    CHECK(std::holds_alternative<SemanticDispatchProgram>(result));
    if (!std::holds_alternative<SemanticDispatchProgram>(result)) return;
    const auto& program = std::get<SemanticDispatchProgram>(result);
    CHECK(program.steps.size() == 18);
    CHECK(program.steps.back().kind == SemanticDispatchStepKind::GreedySampler);
    CHECK(program.steps.back().operation == kSemanticDispatchNoOperator);
    CHECK(program.steps.back().phase == ExecutionPhase::Output);
    CHECK(program.steps.back().input_values == std::vector<uint32_t>{17});
    CHECK(program.steps.back().covered_operator_ids.empty());
    CHECK(program.steps.back().sampler_binding.has_value());
    if (program.steps.back().sampler_binding) {
        CHECK(program.steps.back().sampler_binding->logits_value_id == 17);
        CHECK(program.steps.back().sampler_binding->selected_row == 0);
        CHECK(program.steps.back().sampler_binding->vocabulary_size == 3);
        CHECK(program.output_binding.has_value());
        if (program.output_binding) {
            CHECK(program.steps.back().sampler_binding->logits_value_id ==
                  program.output_binding->logits_value_id);
            CHECK(program.steps.back().sampler_binding->selected_row ==
                  program.output_binding->selected_row);
            CHECK(program.steps.back().sampler_binding->vocabulary_size ==
                  program.output_binding->vocabulary_size);
        }
    }
    CHECK(program.steps.back().session_effects.size() == 3);
    CHECK(validate_semantic_dispatch_program(model, request, program));

    request.batch_rows = 2;
    result = build_semantic_dispatch_program(model, request);
    CHECK(std::holds_alternative<SemanticDispatchProgram>(result));
    if (const auto* batched = std::get_if<SemanticDispatchProgram>(&result)) {
        CHECK(batched->output_binding.has_value());
        CHECK(batched->steps.back().sampler_binding.has_value());
        if (batched->output_binding && batched->steps.back().sampler_binding) {
            CHECK(batched->output_binding->selected_row == 1);
            CHECK(batched->steps.back().sampler_binding->selected_row == 1);
        }
    }
}

void test_invalid_graphs() {
    SemanticModel model = dense_model();
    auto result = build_semantic_dispatch_program(model, {});
    CHECK(std::holds_alternative<SemanticDispatchProgram>(result));
    if (!std::holds_alternative<SemanticDispatchProgram>(result)) return;
    model.operators.back().inputs = {17};
    result = build_semantic_dispatch_program(model, {});
    CHECK(std::holds_alternative<CompatibilityReport>(result));
    model = dense_model();
    std::swap(model.states[0], model.states[1]);
    model.states[0].id = 0;
    model.states[1].id = 1;
    result = build_semantic_dispatch_program(model, {});
    CHECK(std::holds_alternative<CompatibilityReport>(result));

    model = dense_model();
    model.values.push_back(value(19, 4));
    SemanticOperator orphan;
    orphan.id = static_cast<uint32_t>(model.operators.size());
    orphan.kind = OperatorKind::RmsNorm;
    orphan.semantic_version = 1;
    orphan.inputs = {15};
    orphan.outputs = {19};
    orphan.tensors = {10};
    orphan.payload = RmsNormPayload{0x358637bdu, -1, 1};
    model.operators.push_back(orphan);
    result = build_semantic_dispatch_program(model, {});
    CHECK(std::holds_alternative<CompatibilityReport>(result));
}

void test_recurrent_state_effect() {
    const SemanticModel model = recurrent_conv_model();
    const auto result = build_semantic_dispatch_program(model, {});
    CHECK(std::holds_alternative<SemanticDispatchProgram>(result));
    if (const auto* program = std::get_if<SemanticDispatchProgram>(&result)) {
        CHECK(program->steps.size() == 1);
        CHECK(program->steps[0].state_effects.size() == 1);
        if (program->steps[0].state_effects.size() == 1) {
            CHECK(program->steps[0].state_effects[0].update_kind ==
                  StateUpdateKind::ShiftHistory);
        }
    }
}

} // namespace

int main() {
    test_dense_program();
    test_program_mutations();
    test_speculation_and_sampler();
    test_invalid_graphs();
    test_recurrent_state_effect();
    return test_summary("test_semantic_dispatch_program");
}
