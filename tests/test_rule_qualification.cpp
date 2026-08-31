#include <array>
#include <cstring>

#include "compat_rule.h"
#include "test_util.h"

using namespace Laplace;

namespace {

struct RuleTensorCoverage {
    uint32_t template_id;
    const char* pattern;
    TensorRole role;
    std::array<uint64_t, 2> dimensions;
    uint8_t rank;
    ScalarType storage_type;
    TensorTransformKind transform;
    uint32_t repetition_count;
    uint32_t template_stride;
    std::array<uint32_t, 1> aliases;
    uint32_t alias_count;
};

#include "fixtures/qwen2.5-0.5b-f16-coverage.inc"

Sha256Digest first_dense_digest() {
    return {{
        0x73, 0xf9, 0x32, 0xc0, 0xa3, 0xa3, 0xda, 0xf1,
        0x12, 0x25, 0x45, 0x47, 0x39, 0x73, 0xb4, 0x18,
        0x9e, 0xf0, 0x81, 0x1c, 0xd6, 0xf2, 0xf1, 0x4d,
        0x73, 0xaf, 0xe9, 0xfa, 0x1e, 0xdd, 0x8e, 0xfd,
    }};
}

void test_exact_rule_coverage() {
    const auto& rules = bundled_compatibility_rules();
    CHECK(rules.size() == 1);
    if (rules.size() != 1) return;
    const CompatibilityRule& rule = rules.front();
    CHECK(rule.schema_major == 1 && rule.schema_minor == 0 && rule.evaluator_major == 1 && rule.evaluator_minor == 0);
    CHECK(rule.rule_id == "dense-f16-151936-v1" && rule.rule_revision == 1);
    CHECK(rule.package_format == PackageFormat::Gguf);
    CHECK(rule.qualification_state == RuleQualificationState::CanonicalRuntimePassed);
    CHECK(rule.metadata.size() == 12);
    CHECK(rule.tensors.size() == kTensorCoverage.size());
    CHECK(rule.semantic_template.maximum_context == 32768);
    CHECK(rule.semantic_template.vocabulary_size == 151936);
    CHECK(rule.semantic_template.tensors.size() == 291);
    CHECK(rule.semantic_template.layers.size() == 24);
    CHECK(rule.semantic_template.states.size() == 48);
    CHECK(rule.semantic_template.operators.size() == 339);
    CHECK(rule.capabilities == rule.semantic_template.capabilities);
    CHECK(rule.fallbacks == rule.semantic_template.fallbacks);
    CHECK(rule.capabilities == std::vector<CapabilityRequirement>({{Capability::ScalarFp32, 1, 0}}));
    CHECK(rule.fallbacks == std::vector<SemanticFallback>({{FallbackKind::ExactCpu, ExecutionPhase::Decode, NumericalClass::ExactFp32, 0}}));
    CHECK(rule_fingerprint(rule).hex() == "da0ecff3a0e97b3f61338fa301416da841e4401b853f5785dfe56c33baca4263");
    CHECK(rule.semantic_template_digest == semantic_model_digest(rule.semantic_template));

    bool found_artifact_digest = false;
    for (const MetadataPredicate& predicate : rule.metadata) {
        CHECK(predicate.kind == MetadataPredicateKind::ExactU64 || predicate.kind == MetadataPredicateKind::Digest);
        if (predicate.key == "artifact.content_sha256") {
            CHECK(predicate.kind == MetadataPredicateKind::Digest);
            CHECK(predicate.digest == first_dense_digest());
            found_artifact_digest = true;
        }
    }
    CHECK(found_artifact_digest);

    for (size_t index = 0; index != kTensorCoverage.size() && index != rule.tensors.size(); ++index) {
        const RuleTensorCoverage& expected = kTensorCoverage[index];
        const TensorPattern& actual = rule.tensors[index];
        CHECK(actual.template_id == expected.template_id);
        CHECK(actual.pattern == expected.pattern);
        CHECK(actual.role == expected.role && actual.logical_type == ScalarType::F32);
        CHECK(actual.layout == PhysicalLayoutKind::ContiguousRowMajor && actual.quantization == QuantizationKind::None);
        CHECK(actual.storage_type == expected.storage_type && actual.transform == expected.transform);
        CHECK(actual.repetition_count == expected.repetition_count && actual.template_stride == expected.template_stride);
        CHECK(actual.dimensions.size() == expected.rank);
        for (size_t dimension = 0; dimension != actual.dimensions.size() && dimension != expected.rank; ++dimension) {
            CHECK((actual.dimensions[dimension] == Dimension{DimensionKind::Constant, expected.dimensions[dimension]}));
        }
        CHECK(actual.alias_template_ids.size() == expected.alias_count);
        for (size_t alias = 0; alias != actual.alias_template_ids.size() && alias != expected.alias_count; ++alias) {
            CHECK(actual.alias_template_ids[alias] == expected.aliases[alias]);
        }
        for (uint32_t repeat = 0; repeat != actual.repetition_count; ++repeat) {
            const uint32_t tensor_id = actual.template_id + repeat * actual.template_stride;
            CHECK(tensor_id < rule.semantic_template.tensors.size());
            if (tensor_id >= rule.semantic_template.tensors.size()) continue;
            const SemanticTensor& tensor = rule.semantic_template.tensors[tensor_id];
            CHECK(tensor.role == actual.role && tensor.logical_type == actual.logical_type);
            CHECK(tensor.planes.size() == 1 && tensor.planes[0].storage_type == actual.storage_type);
        }
    }

    for (size_t index = 0; index != rule.semantic_template.states.size(); index += 2) {
        const SemanticState& key = rule.semantic_template.states[index];
        const SemanticState& value = rule.semantic_template.states[index + 1];
        CHECK(key.kind == StateKind::KeyCache && value.kind == StateKind::ValueCache);
        CHECK(key.formats.size() == 1 && value.formats.size() == 1);
        if (key.formats.size() == 1 && value.formats.size() == 1) {
            CHECK(key.formats[0].encoded_domain == TransformDomain::RopeApplied);
            CHECK(value.formats[0].encoded_domain == TransformDomain::Untransformed);
            CHECK(key.formats[0].codec == CodecKind::Fp32 && value.formats[0].codec == CodecKind::Fp32);
        }
    }
}

} // namespace

int main() {
    test_exact_rule_coverage();
    return test_summary("test_rule_qualification");
}
