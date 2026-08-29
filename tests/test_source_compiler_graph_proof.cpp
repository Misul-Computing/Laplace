#include <algorithm>
#include <string_view>
#include <utility>

#include "source_compiler_graph_proof.h"
#include "test_util.h"

using namespace Laplace;

namespace {

SemanticTensor tensor(uint32_t id) {
    SemanticTensor result;
    result.id = id;
    result.role = id == 0 ? TensorRole::TokenEmbedding
                          : (id == 1 ? TensorRole::AttentionNormWeight
                                     : TensorRole::OutputWeight);
    return result;
}

SemanticModel compact_decoder() {
    SemanticModel model;
    model.input_values_first = 0;
    model.input_values_count = 1;
    model.output_values_first = 4;
    model.output_values_count = 1;

    for (uint32_t id = 0; id != 5; ++id) {
        model.values.push_back({id, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0});
    }
    for (uint32_t id = 0; id != 3; ++id) model.tensors.push_back(tensor(id));
    model.states.push_back({0});
    model.states.push_back({1});
    model.operators = {
        {0, OperatorKind::EmbeddingLookup, 1, {0}, {1}, {0}, {},
         EmbeddingLookupPayload{0x3f800000u, 1, 1, 0}},
        {1, OperatorKind::RmsNorm, 1, {1}, {2}, {1}, {}, RmsNormPayload{}},
        {2, OperatorKind::CausalAttention, 1, {2, 1, 0}, {3}, {}, {0, 1},
         CausalAttentionPayload{1, 1, 1, 0x3f800000u}},
        {3, OperatorKind::Linear, 1, {3}, {4}, {2}, {}, LinearPayload{}},
    };
    // Embedding and output projection are graph-level operators. Only the
    // repeated decoder body belongs to the semantic layer.
    model.layers.push_back({0, 1, 2, 0});
    return model;
}

SemanticModel layered_add_chain(uint32_t layer_count) {
    SemanticModel model;
    model.input_values_first = 0;
    model.input_values_count = 1;
    model.output_values_first = layer_count;
    model.output_values_count = 1;
    model.values.reserve(static_cast<size_t>(layer_count) + 1);
    model.operators.reserve(layer_count);
    model.layers.reserve(layer_count);
    for (uint32_t id = 0; id <= layer_count; ++id) {
        model.values.push_back({id, ScalarType::F32,
                                {{DimensionKind::Constant, 1}}, 0});
    }
    for (uint32_t id = 0; id != layer_count; ++id) {
        model.operators.push_back(
            {id, OperatorKind::Add, 1, {id, 0}, {id + 1}, {}, {}, AddPayload{}});
        model.layers.push_back({id, id, 1, 0});
    }
    return model;
}

void expect_refusal(const SemanticModel& model, CompatibilityError code,
                    std::string_view detail = {}) {
    const auto result = prove_source_candidate_graph(model);
    CHECK(std::holds_alternative<CompatibilityReport>(result));
    if (const auto* report = std::get_if<CompatibilityReport>(&result)) {
        CHECK(report->code == code);
        if (!detail.empty()) CHECK(report->detail.find(detail) != std::string::npos);
    }
}

void test_accepts_compact_decoder_graph() {
    const auto result = prove_source_candidate_graph(compact_decoder());
    CHECK(std::holds_alternative<SourceCompilerGraphProof>(result));
    if (const auto* proof = std::get_if<SourceCompilerGraphProof>(&result)) {
        CHECK(proof->topological_operator_ids == std::vector<uint32_t>({0, 1, 2, 3}));
        CHECK(proof->input_value_count == 1);
        CHECK(proof->output_value_count == 1);
        CHECK(proof->value_count == 5);
        CHECK(proof->operator_count == 4);
        CHECK(proof->layer_count == 1);
        CHECK(source_compiler_graph_proof_matches(compact_decoder(), *proof));
        auto changed = compact_decoder();
        changed.output_values_first = 3;
        CHECK(!source_compiler_graph_proof_matches(changed, *proof));
    }
}

void test_rejects_stale_tensor_binding() {
    const auto result = prove_source_candidate_graph(compact_decoder());
    CHECK(std::holds_alternative<SourceCompilerGraphProof>(result));
    if (const auto* proof = std::get_if<SourceCompilerGraphProof>(&result)) {
        auto changed_reference = compact_decoder();
        changed_reference.operators[0].tensors = {1};
        changed_reference.operators[1].tensors = {0};
        CHECK(!source_compiler_graph_proof_matches(changed_reference, *proof));

        auto changed_record = compact_decoder();
        changed_record.tensors[0].role = TensorRole::AttentionNormWeight;
        CHECK(!source_compiler_graph_proof_matches(changed_record, *proof));
    }
}

void test_rejects_stale_execution_tensor_contract() {
    const auto result = prove_source_candidate_graph(compact_decoder());
    CHECK(std::holds_alternative<SourceCompilerGraphProof>(result));
    if (const auto* proof = std::get_if<SourceCompilerGraphProof>(&result)) {
        auto changed_layout = compact_decoder();
        changed_layout.tensors[0].layout.rank = 1;
        changed_layout.tensors[0].layout.strides[0] = 1;
        CHECK(!source_compiler_graph_proof_matches(changed_layout, *proof));

        auto changed_quantization = compact_decoder();
        changed_quantization.tensors[0].quantization.flags = 1;
        CHECK(!source_compiler_graph_proof_matches(changed_quantization, *proof));

        auto changed_planes = compact_decoder();
        changed_planes.tensors[0].planes.push_back(
            {PlaneKind::Values, ScalarType::F16, ArtifactId{0}, 0, 2, 2, 0});
        CHECK(!source_compiler_graph_proof_matches(changed_planes, *proof));

        auto changed_expert_stride = compact_decoder();
        changed_expert_stride.tensors[0].expert_axis.per_expert_byte_stride = 256;
        CHECK(!source_compiler_graph_proof_matches(changed_expert_stride, *proof));

        auto changed_vocabulary = compact_decoder();
        changed_vocabulary.vocabulary_size = 2;
        CHECK(!source_compiler_graph_proof_matches(changed_vocabulary, *proof));
    }
}

void test_rejects_invalid_versioned_payload_before_proof() {
    auto model = compact_decoder();
    auto& attention = std::get<CausalAttentionPayload>(model.operators[2].payload);
    attention.window = AttentionWindowKind::Sliding;
    attention.window_tokens = 1;
    expect_refusal(model, CompatibilityError::IR_REFERENCE_INVALID,
                   "operator semantic contract");
}

void test_enforces_shared_layer_ceiling() {
    CHECK(std::holds_alternative<SourceCompilerGraphProof>(
        prove_source_candidate_graph(layered_add_chain(kSemanticModelMaximumLayers))));
    expect_refusal(layered_add_chain(kSemanticModelMaximumLayers + 1),
                   CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                   "layer count");
}

void test_rejects_stale_operator_kind_or_version() {
    const auto result = prove_source_candidate_graph(compact_decoder());
    CHECK(std::holds_alternative<SourceCompilerGraphProof>(result));
    if (const auto* proof = std::get_if<SourceCompilerGraphProof>(&result)) {
        auto changed_kind = compact_decoder();
        changed_kind.operators[3].kind = OperatorKind::RmsNorm;
        changed_kind.operators[3].payload = RmsNormPayload{};
        CHECK(!source_compiler_graph_proof_matches(changed_kind, *proof));

        auto changed_version = compact_decoder();
        changed_version.operators[3].semantic_version = 2;
        CHECK(!source_compiler_graph_proof_matches(changed_version, *proof));
    }
}

void test_rejects_stale_layer_range() {
    const auto result = prove_source_candidate_graph(compact_decoder());
    CHECK(std::holds_alternative<SourceCompilerGraphProof>(result));
    if (const auto* proof = std::get_if<SourceCompilerGraphProof>(&result)) {
        auto changed = compact_decoder();
        changed.layers[0].first_operator = 2;
        CHECK(!source_compiler_graph_proof_matches(changed, *proof));
    }
}

void test_rejects_stale_graph_io_range() {
    const auto result = prove_source_candidate_graph(compact_decoder());
    CHECK(std::holds_alternative<SourceCompilerGraphProof>(result));
    if (const auto* proof = std::get_if<SourceCompilerGraphProof>(&result)) {
        auto changed = compact_decoder();
        changed.output_values_first = 3;
        changed.operators[2].outputs = {4};
        changed.operators[3].inputs = {4};
        changed.operators[3].outputs = {3};
        CHECK(std::holds_alternative<SourceCompilerGraphProof>(
            prove_source_candidate_graph(changed)));
        CHECK(!source_compiler_graph_proof_matches(changed, *proof));
    }
}

void test_rejects_stale_state_binding() {
    const auto result = prove_source_candidate_graph(compact_decoder());
    CHECK(std::holds_alternative<SourceCompilerGraphProof>(result));
    if (const auto* proof = std::get_if<SourceCompilerGraphProof>(&result)) {
        auto changed = compact_decoder();
        changed.states[0].kind = StateKind::ValueCache;
        CHECK(!source_compiler_graph_proof_matches(changed, *proof));
    }
}

void test_rejects_invalid_signature_before_reference_accounting() {
    auto model = compact_decoder();
    model.operators[0].tensors.clear();
    expect_refusal(model, CompatibilityError::IR_REFERENCE_INVALID, "operator signature");
}

void test_rejects_operator_tensor_role_mismatch() {
    auto model = compact_decoder();
    model.tensors[0].role = TensorRole::AttentionNormWeight;
    expect_refusal(model, CompatibilityError::IR_REFERENCE_INVALID, "operator tensor role mismatch");
}

void test_rejects_duplicate_producer() {
    auto model = compact_decoder();
    model.operators[3].outputs = {1};
    expect_refusal(model, CompatibilityError::IR_REFERENCE_INVALID, "duplicate producer");
}

void test_rejects_self_edge() {
    auto model = compact_decoder();
    model.operators[3].inputs = {4};
    expect_refusal(model, CompatibilityError::IR_REFERENCE_INVALID, "self-edge");
}

void test_rejects_cycle() {
    auto model = compact_decoder();
    model.operators[1].inputs = {2};
    model.operators[1].outputs = {3};
    model.operators[2].inputs = {3, 1, 0};
    model.operators[2].outputs = {2};
    expect_refusal(model, CompatibilityError::IR_REFERENCE_INVALID, "cycle");
}

void test_rejects_dependency_that_reverses_serialized_program() {
    auto model = compact_decoder();
    model.operators[1].inputs = {3};
    model.operators[1].outputs = {2};
    model.operators[2].inputs = {1, 1, 0};
    model.operators[2].outputs = {3};
    model.operators[3].inputs = {2};
    expect_refusal(model, CompatibilityError::IR_REFERENCE_INVALID, "serialized operator program");
}

void test_rejects_cross_layer_dependency() {
    auto model = compact_decoder();
    model.values.push_back({5, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0});
    model.output_values_first = 5;
    model.operators[2].inputs = {2, 4, 0};
    model.operators[3].inputs = {0};
    model.operators.push_back(
        {4, OperatorKind::Add, 1, {3, 4}, {5}, {}, {}, AddPayload{}});
    model.layers = {{0, 1, 2, 0}, {1, 3, 2, 0}};
    expect_refusal(model, CompatibilityError::IR_REFERENCE_INVALID,
                   "serialized operator program");
}

void test_rejects_state_shared_without_access_effect_contract() {
    auto model = compact_decoder();
    model.values.push_back({5, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0});
    model.operators[2].states = {0, 0};
    model.operators[2].inputs = {1, 0, 0};
    model.operators[2].outputs = {3};
    model.operators[3] = {3, OperatorKind::Add, 1, {2, 3}, {5}, {}, {}, AddPayload{}};
    model.operators.push_back(
        {4, OperatorKind::Linear, 1, {5}, {4}, {3}, {}, LinearPayload{}});
    model.layers.front().operator_count = 3;
    expect_refusal(model, CompatibilityError::IR_STATE_INVALID, "access/effect");
}

void test_rejects_disconnected_zero_output_operator() {
    auto model = compact_decoder();
    model.operators.push_back(
        {4, OperatorKind::Add, 1, {0}, {}, {}, {}, AddPayload{}});
    expect_refusal(model, CompatibilityError::IR_REFERENCE_INVALID,
                   "operator signature");
}

void test_rejects_unresolved_input() {
    auto model = compact_decoder();
    model.values.push_back({5, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0});
    model.operators[3].inputs = {5};
    expect_refusal(model, CompatibilityError::IR_REFERENCE_INVALID, "unresolved input");
}

void test_rejects_unreachable_logits() {
    auto model = compact_decoder();
    model.operators[3].inputs.clear();
    expect_refusal(model, CompatibilityError::IR_REFERENCE_INVALID, "operator signature");
}

void test_rejects_dead_produced_value() {
    auto model = compact_decoder();
    model.values.push_back({5, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0});
    model.operators[2].outputs = {3, 5};
    expect_refusal(model, CompatibilityError::IR_REFERENCE_INVALID, "operator signature");
}

void test_rejects_invalid_layer_coverage() {
    auto model = compact_decoder();
    model.layers.push_back({1, 2, 1, 0});
    expect_refusal(model, CompatibilityError::IMPORT_SCHEMA_INCOMPLETE, "layer coverage");
}

void test_accepts_graph_level_operators_between_layers() {
    auto model = compact_decoder();
    model.layers = {{0, 0, 1, 0}, {1, 2, 1, 0}};
    const auto result = prove_source_candidate_graph(model);
    CHECK(std::holds_alternative<SourceCompilerGraphProof>(result));
}

void test_rejects_unconsumed_tensor() {
    auto model = compact_decoder();
    model.tensors.push_back(tensor(3));
    expect_refusal(model, CompatibilityError::IMPORT_TENSOR_UNMAPPED, "unconsumed tensor");
}

void test_accepts_an_unreferenced_inactive_program_tensor() {
    auto model = compact_decoder();
    SemanticTensor inactive = tensor(3);
    inactive.flags = kSemanticTensorFlagInactiveProgram;
    model.tensors.push_back(std::move(inactive));
    CHECK(std::holds_alternative<SourceCompilerGraphProof>(
        prove_source_candidate_graph(model)));
}

void test_rejects_a_referenced_inactive_program_tensor() {
    auto model = compact_decoder();
    model.tensors[2].flags = kSemanticTensorFlagInactiveProgram;
    expect_refusal(model, CompatibilityError::IR_REFERENCE_INVALID,
                   "inactive-program tensor is referenced");
}

void test_rejects_unconsumed_state() {
    auto model = compact_decoder();
    model.states.push_back({2});
    expect_refusal(model, CompatibilityError::IR_STATE_INVALID, "unconsumed state");
}

} // namespace

int main() {
    test_accepts_compact_decoder_graph();
    test_rejects_stale_tensor_binding();
    test_rejects_stale_execution_tensor_contract();
    test_rejects_invalid_versioned_payload_before_proof();
    test_enforces_shared_layer_ceiling();
    test_rejects_stale_operator_kind_or_version();
    test_rejects_stale_layer_range();
    test_rejects_stale_graph_io_range();
    test_rejects_stale_state_binding();
    test_rejects_invalid_signature_before_reference_accounting();
    test_rejects_operator_tensor_role_mismatch();
    test_rejects_duplicate_producer();
    test_rejects_self_edge();
    test_rejects_cycle();
    test_rejects_dependency_that_reverses_serialized_program();
    test_rejects_cross_layer_dependency();
    test_rejects_state_shared_without_access_effect_contract();
    test_rejects_disconnected_zero_output_operator();
    test_rejects_unresolved_input();
    test_rejects_unreachable_logits();
    test_rejects_dead_produced_value();
    test_rejects_invalid_layer_coverage();
    test_accepts_graph_level_operators_between_layers();
    test_rejects_unconsumed_tensor();
    test_accepts_an_unreferenced_inactive_program_tensor();
    test_rejects_a_referenced_inactive_program_tensor();
    test_rejects_unconsumed_state();
    return test_summary("test_source_compiler_graph_proof");
}
