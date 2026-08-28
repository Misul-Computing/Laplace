#include <cstring>
#include <map>
#include <memory>
#include <vector>

#include "compat_rule.h"
#include "test_util.h"

using namespace Laplace;

namespace {

SemanticModel one_tensor_model() {
    SemanticModel model;
    model.maximum_context = 32768;
    model.vocabulary_size = 3;
    model.bos_id = 1;
    model.eos_id = 2;
    SemanticTensor tensor;
    tensor.id = 0;
    tensor.role = TensorRole::TokenEmbedding;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, 2}, {DimensionKind::Constant, 2}};
    tensor.layout.kind = PhysicalLayoutKind::ContiguousRowMajor;
    tensor.layout.packing = PackingKind::None;
    tensor.layout.rank = 2;
    tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides[0] = 2;
    tensor.layout.strides[1] = 1;
    tensor.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, 0, 16, 4, 0}};
    model.tensors.push_back(tensor);
    return model;
}

CompatibilityRule one_tensor_rule() {
    CompatibilityRule rule;
    rule.rule_id = "dense-test-v1";
    rule.rule_revision = 7;
    rule.package_format = PackageFormat::Gguf;
    rule.qualification_state = RuleQualificationState::Draft;
    rule.metadata = {{MetadataPredicateKind::ExactU64, "layers", 1, {}}};
    rule.tensors = {{0, TensorPatternKind::AnchoredDecimalCapture, "blk.{d}.weight", 0,
                     TensorRole::TokenEmbedding, ScalarType::F32, PhysicalLayoutKind::ContiguousRowMajor,
                     QuantizationKind::None, {{DimensionKind::Constant, 2}, {DimensionKind::Constant, 2}},
                     1}};
    rule.constraints = {{ConstraintKind::ExactValue, ConstraintOperandKind::Value, ConstraintOperandKind::Constant,
                         0, 0, 0, 0, 2, 1}};
    rule.capabilities = {{Capability::ScalarFp32, 1, 0}};
    rule.fallbacks = {{FallbackKind::ExactCpu, ExecutionPhase::Decode, NumericalClass::ExactFp32, 0}};
    rule.semantic_template = one_tensor_model();
    return rule;
}

PackageEvidence one_tensor_package() {
    PackageEvidence package;
    package.metadata.emplace("layers", uint64_t{1});
    package.tensors.push_back({"blk.0.weight", {2, 2}, ScalarType::F32,
                               PhysicalLayoutKind::ContiguousRowMajor, QuantizationKind::None,
                               0, 0, ArtifactId{0}, 0, 16, 4});
    return package;
}

void test_canonical_rule_bytes_and_match() {
    CompatibilityRule rule = one_tensor_rule();
    auto encoded = encode_compatibility_rule(rule);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
    if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) return;
    const auto& bytes = std::get<std::vector<uint8_t>>(encoded);
    CHECK(bytes.size() >= 128);
    CHECK(std::memcmp(bytes.data(), "LAPRUL10", 8) == 0);
    auto decoded = decode_compatibility_rule(bytes);
    CHECK(std::holds_alternative<CompatibilityRule>(decoded));
    if (auto* decoded_rule = std::get_if<CompatibilityRule>(&decoded)) {
        CHECK(decoded_rule->rule_id == rule.rule_id);
        CHECK(decoded_rule->metadata.size() == 1);
        CHECK(decoded_rule->tensors.size() == 1);
        CHECK(decoded_rule->constraints == rule.constraints);
        CHECK(decoded_rule->capabilities == rule.capabilities);
        CHECK(decoded_rule->fallbacks == rule.fallbacks);
        CHECK(decoded_rule->semantic_template.tensors.size() == 1);
        CHECK(rule_fingerprint(*decoded_rule) == rule_fingerprint(rule));
    }

    auto result = evaluate_rules({rule}, one_tensor_package());
    CHECK(std::holds_alternative<SemanticModel>(result));
    if (auto* model = std::get_if<SemanticModel>(&result)) {
        CHECK(model->tensors.size() == 1);
        CHECK(model->tensors[0].planes[0].offset == 0);
        CHECK(model->tensors[0].planes[0].length == 16);
    }
}

void test_fixed_evaluator_fail_closed() {
    CompatibilityRule rule = one_tensor_rule();
    auto missing = one_tensor_package();
    missing.metadata["layers"] = uint64_t{2};
    auto no_match = evaluate_rules({rule}, missing);
    CHECK(std::holds_alternative<CompatibilityReport>(no_match));
    if (auto* report = std::get_if<CompatibilityReport>(&no_match)) {
        CHECK(report->code == CompatibilityError::IMPORT_SEMANTICS_MISSING);
    }

    auto unexplained = one_tensor_package();
    unexplained.tensors.push_back({"unknown.weight", {2}, ScalarType::F32,
                                   PhysicalLayoutKind::ContiguousRowMajor, QuantizationKind::None,
                                   0, 0, ArtifactId{0}, 16, 8, 4});
    auto unmapped = evaluate_rules({rule}, unexplained);
    CHECK(std::holds_alternative<CompatibilityReport>(unmapped));
    if (auto* report = std::get_if<CompatibilityReport>(&unmapped)) {
        CHECK(report->code == CompatibilityError::IMPORT_TENSOR_UNMAPPED);
    }

    auto conflict = evaluate_rules({rule, rule}, one_tensor_package());
    CHECK(std::holds_alternative<CompatibilityReport>(conflict));
    if (auto* report = std::get_if<CompatibilityReport>(&conflict)) {
        CHECK(report->code == CompatibilityError::IMPORT_RULE_CONFLICT);
    }
}

void test_rule_separates_logical_and_physical_types() {
    CompatibilityRule rule = one_tensor_rule();
    rule.tensors[0].storage_type = ScalarType::F16;
    rule.semantic_template.tensors[0].planes[0].storage_type = ScalarType::F16;
    auto package = one_tensor_package();
    package.tensors[0].storage_type = ScalarType::F16;
    package.tensors[0].length = 8;
    auto result = evaluate_rules({rule}, package);
    CHECK(std::holds_alternative<SemanticModel>(result));
    if (auto* model = std::get_if<SemanticModel>(&result)) {
        CHECK(model->tensors[0].logical_type == ScalarType::F32);
        CHECK(model->tensors[0].planes[0].storage_type == ScalarType::F16);
    }
}

void test_rule_declares_transpose_and_tied_alias() {
    CompatibilityRule rule = one_tensor_rule();
    rule.tensors[0].dimensions = {{DimensionKind::Constant, 2}, {DimensionKind::Constant, 3}};
    rule.tensors[0].transform = TensorTransformKind::LogicalTranspose;
    rule.semantic_template.tensors[0].dimensions = {{DimensionKind::Constant, 3}, {DimensionKind::Constant, 2}};
    SemanticTensor alias = rule.semantic_template.tensors[0];
    alias.id = 1;
    alias.role = TensorRole::OutputWeight;
    alias.planes[0].flags = 1;
    rule.semantic_template.tensors.push_back(alias);
    rule.tensors[0].alias_template_ids = {1};

    auto package = one_tensor_package();
    package.tensors[0].dimensions = {2, 3};
    package.tensors[0].length = 24;
    auto result = evaluate_rules({rule}, package);
    CHECK(std::holds_alternative<SemanticModel>(result));
    if (auto* model = std::get_if<SemanticModel>(&result)) {
        CHECK(model->tensors[0].dimensions[0].constant_or_symbol == 3);
        CHECK(model->tensors[0].dimensions[1].constant_or_symbol == 2);
        CHECK(model->tensors[1].planes[0].offset == model->tensors[0].planes[0].offset);
        CHECK(model->tensors[1].planes[0].flags == 1);
    }
}

void test_rule_expands_bounded_layer_capture() {
    CompatibilityRule rule = one_tensor_rule();
    rule.tensors[0].repetition_count = 2;
    rule.tensors[0].template_stride = 1;
    SemanticTensor second = rule.semantic_template.tensors[0];
    second.id = 1;
    rule.semantic_template.tensors.push_back(second);
    auto package = one_tensor_package();
    package.tensors.push_back({"blk.1.weight", {2, 2}, ScalarType::F32,
                               PhysicalLayoutKind::ContiguousRowMajor, QuantizationKind::None,
                               0, 0, ArtifactId{0}, 16, 16, 4});
    auto result = evaluate_rules({rule}, package);
    CHECK(std::holds_alternative<SemanticModel>(result));
    if (auto* model = std::get_if<SemanticModel>(&result)) {
        CHECK(model->tensors.size() == 2);
        CHECK(model->tensors[1].planes[0].offset == 16);
    }
}

void test_rule_accepts_exact_root_name() {
    CompatibilityRule rule = one_tensor_rule();
    rule.tensors[0].kind = TensorPatternKind::ExactName;
    rule.tensors[0].pattern = "token_embd.weight";
    auto package = one_tensor_package();
    package.tensors[0].name = "token_embd.weight";
    auto result = evaluate_rules({rule}, package);
    CHECK(std::holds_alternative<SemanticModel>(result));
}

void test_rule_binds_template_semantics() {
    CompatibilityRule rule = one_tensor_rule();
    rule.semantic_template_digest = semantic_model_digest(rule.semantic_template);
    rule.semantic_template.vocabulary_size = 4;
    auto result = evaluate_rules({rule}, one_tensor_package());
    CHECK(std::holds_alternative<CompatibilityReport>(result));
    if (auto* report = std::get_if<CompatibilityReport>(&result)) {
        CHECK(report->code == CompatibilityError::IMPORT_SEMANTICS_MISSING);
    }
}

void test_rule_encodes_recurrent_tensor_role() {
    CompatibilityRule rule = one_tensor_rule();
    rule.rule_id = "recurrent-role-v1";
    rule.semantic_template.schema_major = 3;
    rule.semantic_template.opset_major = 3;
    rule.semantic_template.tensors[0].role = TensorRole::RecurrentQkvWeight;
    rule.tensors[0].role = TensorRole::RecurrentQkvWeight;

    auto encoded = encode_compatibility_rule(rule);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
    if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) return;
    auto decoded = decode_compatibility_rule(std::get<std::vector<uint8_t>>(encoded));
    CHECK(std::holds_alternative<CompatibilityRule>(decoded));
    if (const auto* round_trip = std::get_if<CompatibilityRule>(&decoded)) {
        CHECK(round_trip->tensors[0].role == TensorRole::RecurrentQkvWeight);
        CHECK(round_trip->semantic_template.tensors[0].role == TensorRole::RecurrentQkvWeight);
    }
    auto matched = evaluate_rules({rule}, one_tensor_package());
    CHECK(std::holds_alternative<SemanticModel>(matched));
}

} // namespace

int main() {
    test_canonical_rule_bytes_and_match();
    test_fixed_evaluator_fail_closed();
    test_rule_separates_logical_and_physical_types();
    test_rule_declares_transpose_and_tied_alias();
    test_rule_expands_bounded_layer_capture();
    test_rule_accepts_exact_root_name();
    test_rule_binds_template_semantics();
    test_rule_encodes_recurrent_tensor_role();
    return test_summary("test_compat_rule");
}
