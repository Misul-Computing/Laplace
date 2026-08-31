#include "closed_v1_source_schema.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace Laplace;

namespace {

void check(bool condition, const char* expression, int line) {
    if (!condition) {
        std::cerr << "check failed at line " << line << ": " << expression << '\n';
        std::exit(1);
    }
}

#define CHECK(condition) check((condition), #condition, __LINE__)

constexpr CanonicalFactKey kWidthFact{0x43563101u};

SourceTensorEvidence source_tensor(uint32_t id, const char* name, uint64_t offset) {
    SourceTensorEvidence tensor;
    tensor.physical_tensor_id = id;
    tensor.source_spelling = name;
    tensor.dimensions = {1};
    tensor.span = {3, offset, 4};
    tensor.physical_type_code = 1;
    tensor.physical_block_axis = 0;
    tensor.physical_block_elements = 1;
    tensor.physical_block_bytes = 4;
    return tensor;
}

SourceNamePattern exact(const char* spelling) {
    SourceNamePattern pattern;
    pattern.kind = SourceNamePatternKind::Exact;
    pattern.literal = spelling;
    return pattern;
}

PhysicalLayout vector_layout() {
    PhysicalLayout layout;
    layout.rank = 1;
    layout.axis_order[0] = 0;
    layout.strides[0] = 1;
    return layout;
}

ClosedV1TensorTemplate tensor_template(uint32_t id, uint32_t source_slot_id,
                                       const char* name) {
    ClosedV1TensorTemplate tensor;
    tensor.id = id;
    tensor.source_slot_id = source_slot_id;
    tensor.role = id == 0 ? TensorRole::TokenEmbedding
                          : (id == 1 ? TensorRole::AttentionNormWeight : TensorRole::OutputWeight);
    tensor.dimensions = {{DimensionKind::Constant, 1}};
    tensor.source_dimensions = tensor.dimensions;
    tensor.source_names = {exact(name)};
    tensor.layout = vector_layout();
    tensor.plane_storage_type = ScalarType::F32;
    tensor.physical_type_code = 1;
    tensor.physical_block_axis = 0;
    tensor.physical_block_elements = 1;
    tensor.physical_block_bytes = 4;
    tensor.plane_alignment = 4;
    return tensor;
}

ClosedV1Schema schema() {
    ClosedV1Schema result;
    result.schema_id = 17;
    result.source_format = SourceFormat::Gguf;
    ClosedV1GraphTemplate& graph = result.graph;
    graph.maximum_context = 32768;
    graph.tokenizer.vocabulary_size = 1;
    graph.bos_id = 0;
    graph.eos_id = 0;
    graph.input_values_first = 0;
    graph.input_values_count = 1;
    graph.output_values_first = 4;
    graph.output_values_count = 1;
    graph.tensors = {
        tensor_template(0, 10, "embed.weight"),
        tensor_template(1, 11, "norm.weight"),
        tensor_template(2, 12, "output.weight"),
    };
    graph.values = {
        {0, ScalarType::U32, {{DimensionKind::Constant, 1}}, 0, {}},
        {1, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0, {}},
        {2, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0, {}},
        {3, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0, {}},
        {4, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0, {}},
    };
    graph.operators = {
        {0, OperatorKind::EmbeddingLookup, 1, {0}, {1}, {0}, {},
         EmbeddingLookupPayload{0x3f800000u, 1, 1, 0}, {}},
        {1, OperatorKind::RmsNorm, 1, {1}, {2}, {1}, {},
         RmsNormPayload{0x3a83126fu, -1, 1}, {}},
        {2, OperatorKind::CausalAttention, 1, {2, 1, 0}, {3}, {}, {0, 1},
         CausalAttentionPayload{1, 1, 1, 0x3f800000u}, {}},
        {3, OperatorKind::Linear, 1, {3}, {4}, {2}, {}, LinearPayload{}, {}},
    };
    graph.layers = {{0, 1, 2, 0, {}}};
    StateFormat key_format;
    key_format.encoded_domain = TransformDomain::RopeApplied;
    key_format.alignment = 64;
    StateFormat value_format;
    value_format.alignment = 64;
    graph.states = {
        {0, StateKind::KeyCache, 1, StateUpdateKind::AppendKey, PositionPolicy::AppendOnly,
         {{DimensionKind::Constant, 1}}, {key_format}, 0, {}},
        {1, StateKind::ValueCache, 1, StateUpdateKind::AppendValue, PositionPolicy::AppendOnly,
         {{DimensionKind::Constant, 1}}, {value_format}, 0, {}},
    };
    return result;
}

ClosedV1SourcePackage package() {
    ClosedV1SourcePackage result;
    result.source.format = SourceFormat::Gguf;
    result.source.artifacts = {{3, 12}};
    result.source.tensors = {
        source_tensor(20, "embed.weight", 0),
        source_tensor(21, "norm.weight", 4),
        source_tensor(22, "output.weight", 8),
    };
    return result;
}

void test_compiles_complete_dense_graph();
void test_schema_selection_and_facts();
void test_rejects_bad_templates_and_bindings();
void test_rejects_logical_physical_mismatch();
void test_rejects_operator_tensor_role_mismatch();
void test_catalog_evaluates_record_controls();
void test_expansion_limit_covers_nested_and_encoded_work();

} // namespace

int main() {
    test_compiles_complete_dense_graph();
    test_schema_selection_and_facts();
    test_rejects_bad_templates_and_bindings();
    test_rejects_logical_physical_mismatch();
    test_rejects_operator_tensor_role_mismatch();
    test_catalog_evaluates_record_controls();
    test_expansion_limit_covers_nested_and_encoded_work();
    return 0;
}

namespace {

void test_compiles_complete_dense_graph() {
    const ClosedV1SourceSchemaResult result =
        compile_closed_v1_source_schema(schema(), package());
    CHECK(result.success());
    CHECK(result.compiled.has_value());
    if (!result.compiled) return;
    const SemanticModel& model = result.compiled->model;
    CHECK(model.schema_major == 1 && model.opset_major == 1);
    CHECK(model.tensors.size() == 3 && model.values.size() == 5);
    CHECK(model.operators.size() == 4 && model.layers.size() == 1);
    CHECK(model.states.size() == 2);
    CHECK(model.tensors[0].planes.size() == 1);
    CHECK(model.tensors[0].planes[0].artifact_id.value == 0);
    CHECK(model.tensors[0].planes[0].offset == 0);
    CHECK(model.tensors[2].planes[0].offset == 8);
    CHECK(source_compiler_graph_proof_matches(model, result.compiled->graph_proof));

    ClosedV1Schema tied = schema();
    tied.graph.aliases.push_back({10, 12, SourceAliasKind::TiedOutput, 77});
    ClosedV1SourcePackage tied_package = package();
    tied_package.source.tensors[2].span = tied_package.source.tensors[0].span;
    tied_package.source.alias_proofs.push_back({77, 10, 12, SourceAliasKind::TiedOutput});
    const ClosedV1SourceSchemaResult tied_result =
        compile_closed_v1_source_schema(tied, tied_package);
    CHECK(tied_result.success());
    if (tied_result.compiled)
        CHECK(tied_result.compiled->model.tensors[0].planes[0].offset ==
              tied_result.compiled->model.tensors[2].planes[0].offset);
}

void test_schema_selection_and_facts() {
    ClosedV1Schema fact_schema = schema();
    fact_schema.predicates.push_back({kWidthFact, uint64_t{1}});
    fact_schema.graph.symbols.push_back({1, kWidthFact, 0, 1, 4});
    fact_schema.graph.values[1].dimensions[0] = {DimensionKind::Symbol, 1};

    ClosedV1SourcePackage fact_package = package();
    fact_package.facts.push_back({kWidthFact, uint64_t{1}});
    const ClosedV1SourceSchemaResult accepted =
        compile_closed_v1_source_schema(fact_schema, fact_package);
    CHECK(accepted.success());
    if (accepted.compiled) {
        CHECK(accepted.compiled->model.values[1].dimensions[0].kind == DimensionKind::Constant);
        CHECK(accepted.compiled->model.values[1].dimensions[0].constant_or_symbol == 1);
    }

    ClosedV1SourcePackage missing = package();
    CHECK(compile_closed_v1_source_schema(fact_schema, missing).error ==
          ClosedV1SchemaError::MissingFact);
    missing.facts.push_back({kWidthFact, true});
    CHECK(compile_closed_v1_source_schema(fact_schema, missing).error ==
          ClosedV1SchemaError::WrongFact);

    ClosedV1Schema wrong_value = fact_schema;
    wrong_value.predicates[0].value = uint64_t{2};
    CHECK(compile_closed_v1_source_schemas(std::array{wrong_value}, fact_package).error ==
          ClosedV1SchemaError::NoMatchingSchema);

    const std::array<ClosedV1Schema, 2> ambiguous = {fact_schema, fact_schema};
    CHECK(compile_closed_v1_source_schemas(ambiguous, fact_package).error ==
          ClosedV1SchemaError::AmbiguousSchema);
}

void test_rejects_bad_templates_and_bindings() {
    ClosedV1Schema repeated = schema();
    repeated.graph.repeat = {2, 1};
    CHECK(compile_closed_v1_source_schema(repeated, package()).error ==
          ClosedV1SchemaError::RepeatBoundExceeded);

    ClosedV1Schema duplicate = schema();
    duplicate.graph.values.push_back(duplicate.graph.values[1]);
    CHECK(compile_closed_v1_source_schema(duplicate, package()).error ==
          ClosedV1SchemaError::DuplicateId);

    ClosedV1Schema cycle = schema();
    cycle.graph.operators[1].inputs = {3};
    cycle.graph.operators[2].inputs = {2, 1, 0};
    CHECK(compile_closed_v1_source_schema(cycle, package()).error ==
          ClosedV1SchemaError::GraphCycle);

    ClosedV1SourcePackage missing = package();
    missing.source.tensors.erase(missing.source.tensors.begin() + 1);
    CHECK(compile_closed_v1_source_schema(schema(), missing).error ==
          ClosedV1SchemaError::MissingTensorBinding);

    ClosedV1SourcePackage extra = package();
    extra.source.artifacts[0].byte_length = 16;
    extra.source.tensors.push_back(source_tensor(23, "unused.weight", 12));
    CHECK(compile_closed_v1_source_schema(schema(), extra).error ==
          ClosedV1SchemaError::UnmatchedSourceTensor);
}

void test_rejects_logical_physical_mismatch() {
    ClosedV1Schema mismatched = schema();
    mismatched.graph.tensors[0].dimensions = {{DimensionKind::Constant, 2}};
    CHECK(compile_closed_v1_source_schema(mismatched, package()).error ==
          ClosedV1SchemaError::PhysicalContractMismatch);

    ClosedV1Schema wrong_source_dimensions = schema();
    wrong_source_dimensions.graph.tensors[0].source_dimensions =
        {{DimensionKind::Constant, 2}};
    CHECK(compile_closed_v1_source_schema(wrong_source_dimensions, package()).error ==
          ClosedV1SchemaError::PhysicalContractMismatch);

    ClosedV1Schema wrong_bytes = schema();
    wrong_bytes.graph.tensors[0].physical_block_bytes = 8;
    CHECK(compile_closed_v1_source_schema(wrong_bytes, package()).error ==
          ClosedV1SchemaError::PhysicalContractMismatch);
}

void test_rejects_operator_tensor_role_mismatch() {
    ClosedV1Schema mismatched = schema();
    mismatched.graph.tensors[0].role = TensorRole::OutputWeight;
    CHECK(compile_closed_v1_source_schema(mismatched, package()).error ==
          ClosedV1SchemaError::InvalidReference);
}

void test_catalog_evaluates_record_controls() {
    const auto check_record_control = [](ClosedV1Schema controlled) {
        const std::array<ClosedV1Schema, 2> candidates = {controlled, schema()};
        const ClosedV1SourceSchemaResult result =
            compile_closed_v1_source_schemas(candidates, package());
        CHECK(result.success());
        CHECK(result.schema_id == schema().schema_id);
    };
    ClosedV1Schema tensor = schema();
    tensor.graph.tensors[0].control.has_variant = true;
    tensor.graph.tensors[0].control.variant = {kWidthFact, uint64_t{2}};
    check_record_control(tensor);

    ClosedV1Schema value = schema();
    value.graph.values[0].control.has_variant = true;
    value.graph.values[0].control.variant = {kWidthFact, uint64_t{2}};
    check_record_control(value);

    ClosedV1Schema operator_schema = schema();
    operator_schema.graph.operators[0].control.has_variant = true;
    operator_schema.graph.operators[0].control.variant = {kWidthFact, uint64_t{2}};
    check_record_control(operator_schema);

    ClosedV1Schema layer = schema();
    layer.graph.layers[0].control.has_variant = true;
    layer.graph.layers[0].control.variant = {kWidthFact, uint64_t{2}};
    check_record_control(layer);

    ClosedV1Schema state = schema();
    state.graph.states[0].control.has_variant = true;
    state.graph.states[0].control.variant = {kWidthFact, uint64_t{2}};
    check_record_control(state);
}

void test_expansion_limit_covers_nested_and_encoded_work() {
    ClosedV1Schema nested = schema();
    nested.maximum_expansion_steps = 16;
    nested.graph.operators[0].inputs.resize(64, 0);
    CHECK(compile_closed_v1_source_schema(nested, package()).error ==
          ClosedV1SchemaError::RepeatBoundExceeded);

    ClosedV1Schema encoded = schema();
    encoded.maximum_expansion_steps = 128;
    CHECK(compile_closed_v1_source_schema(encoded, package()).error ==
          ClosedV1SchemaError::RepeatBoundExceeded);
}

} // namespace
