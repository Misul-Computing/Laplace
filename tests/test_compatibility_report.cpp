#include "compatibility_report.h"
#include "test_util.h"

using namespace Laplace;

namespace {

void expect_stage(CompatibilityError code, CompatibilityStage expected) {
    const CompatibilityReport report = compatibility_report(code);
    CHECK(report.stage == expected);
    CHECK(compatibility_stage(code) == expected);
}

void test_refusal_taxonomy() {
    expect_stage(CompatibilityError::PACKAGE_BAD_MAGIC, CompatibilityStage::Package);
    expect_stage(CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS, CompatibilityStage::Semantic);
    expect_stage(CompatibilityError::IMPORT_SCHEMA_NOT_FOUND, CompatibilityStage::Semantic);
    expect_stage(CompatibilityError::IMPORT_SCHEMA_AMBIGUOUS, CompatibilityStage::Semantic);
    expect_stage(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE, CompatibilityStage::Semantic);
    expect_stage(CompatibilityError::IMPORT_SCHEMA_LIMIT, CompatibilityStage::Semantic);
    expect_stage(CompatibilityError::IMPORT_MANIFEST_INVALID, CompatibilityStage::Semantic);
    expect_stage(CompatibilityError::IMPORT_MANIFEST_DOWNGRADE, CompatibilityStage::Semantic);
    expect_stage(CompatibilityError::IMPORT_CLOSURE_INCOMPLETE, CompatibilityStage::Semantic);
    expect_stage(CompatibilityError::IMPORT_ALIAS_AMBIGUOUS, CompatibilityStage::Semantic);
    expect_stage(CompatibilityError::IMPORT_INTERACTION_UNSUPPORTED, CompatibilityStage::Tokenizer);
    expect_stage(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED, CompatibilityStage::PhysicalFormat);
    expect_stage(CompatibilityError::CAPABILITY_MISSING, CompatibilityStage::Capability);
    expect_stage(CompatibilityError::STATE_ABI_MISMATCH, CompatibilityStage::StateRollback);
    expect_stage(CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED, CompatibilityStage::Tokenizer);
    expect_stage(CompatibilityError::RULE_QUALIFICATION_REQUIRED, CompatibilityStage::Benchmark);
    expect_stage(CompatibilityError::PACKAGE_AUTHORITY_REQUIRED, CompatibilityStage::Session);
    expect_stage(CompatibilityError::AUTHORITY_INVALID, CompatibilityStage::Session);
    CHECK(compatibility_phase(CompatibilityError::PACKAGE_AUTHORITY_REQUIRED) ==
          CompatibilityPhase::Session);
    CHECK(compatibility_message(CompatibilityError::PACKAGE_AUTHORITY_REQUIRED) ==
          "package execution authority is required");
    CHECK(compatibility_message(CompatibilityError::IMPORT_SCHEMA_AMBIGUOUS) ==
          "source schema match is ambiguous");
    CHECK(compatibility_message(CompatibilityError::AUTHORITY_INVALID) ==
          "package execution authority is invalid");
}

void test_diagnostic_location_does_not_change_refusal() {
    CompatibilityReport report = compatibility_report(CompatibilityError::KERNEL_UNAVAILABLE, "missing kernel");
    report.artifact_id = ArtifactId{4};
    report.artifact_index = 4;
    report.layer = 3;
    report.operator_id = 2;
    report.tensor_id = 1;
    report.state_id = 0;
    report.source_offset = 512;
    report.source_length = 144;
    report.metadata_key = "general.architecture";
    report.rule_id = "fixture-v1";
    report.expected = "qualified Metal kernel";
    report.observed = "none";

    CHECK(report.source_offset == 512);
    CHECK(report.source_length == 144);
    CHECK(report.artifact_id == ArtifactId{4});
    CHECK(compatibility_stage(report.code) == CompatibilityStage::Capability);

    report.source_offset = UINT64_MAX;
    report.source_length = 0;
    report.metadata_key = "different";
    report.rule_id = "different";
    report.expected = "different";
    report.observed = "different";
    report.detail = "different";
    CHECK(compatibility_stage(report.code) == CompatibilityStage::Capability);
}

void test_phase_fact_and_message_are_stable() {
    CompatibilityReport import = compatibility_report(CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS,
                                                       "two canonical fact sets match");
    import.artifact_id = ArtifactId{7};
    import.source_offset = 96;
    import.source_length = 24;
    import.tensor_id = 3;
    import.fact_key = CanonicalFactKey{11};

    CHECK(import.code == CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS);
    CHECK(import.stage == CompatibilityStage::Semantic);
    CHECK(import.phase == CompatibilityPhase::Import);
    CHECK(import.message == "import semantics are ambiguous");
    CHECK(import.detail == "two canonical fact sets match");
    CHECK(import.artifact_id == ArtifactId{7});
    CHECK(import.source_offset == 96 && import.source_length == 24);
    CHECK(import.tensor_id == 3);
    CHECK(import.fact_key == CanonicalFactKey{11});
    CHECK(import.operator_id == UINT32_MAX && import.state_id == UINT32_MAX);

    const CompatibilityReport same = compatibility_report(CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS,
                                                           "two canonical fact sets match");
    CHECK(same.code == import.code && same.stage == import.stage && same.phase == import.phase);
    CHECK(same.message == import.message && same.detail == import.detail);

    const CompatibilityReport plan = compatibility_report(CompatibilityError::KERNEL_UNAVAILABLE);
    CHECK(plan.stage == CompatibilityStage::Capability);
    CHECK(plan.phase == CompatibilityPhase::Plan);
    CHECK(plan.message == "required kernel is unavailable");
    CHECK(plan.fact_key == CanonicalFactKey{} && plan.metadata_key.empty());
}

} // namespace

int main() {
    test_refusal_taxonomy();
    test_diagnostic_location_does_not_change_refusal();
    test_phase_fact_and_message_are_stable();
    return test_summary("test_compatibility_report");
}
