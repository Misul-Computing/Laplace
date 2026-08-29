#include "source_schema.h"

#include <array>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <span>
#include <vector>

namespace {

using namespace Laplace;

void check(bool condition, const char* expression, int line) {
    if (!condition) {
        std::cerr << "check failed at line " << line << ": " << expression << '\n';
        std::exit(1);
    }
}

#define CHECK(condition) check((condition), #condition, __LINE__)

SourceNamePattern exact(const char* spelling) {
    SourceNamePattern pattern;
    pattern.kind = SourceNamePatternKind::Exact;
    pattern.literal = spelling;
    return pattern;
}

SourceTensorEvidence tensor(uint32_t id, const char* spelling, PhysicalSpan span) {
    SourceTensorEvidence value;
    value.physical_tensor_id = id;
    value.source_spelling = spelling;
    value.dimensions = {2, 3};
    value.span = span;
    value.content_digest.fill(0x5a);
    return value;
}

SourcePackageEvidence package_with(std::initializer_list<SourceTensorEvidence> tensors) {
    SourcePackageEvidence package;
    package.format = SourceFormat::Gguf;
    package.artifacts.push_back({3, 4096});
    package.alias_proofs.push_back({77, 1, 2, SourceAliasKind::TiedOutput});
    package.tensors.assign(tensors.begin(), tensors.end());
    return package;
}

SourceSchema tied_schema() {
    SourceSchema schema;
    schema.source_format = SourceFormat::Gguf;
    schema.slots = {
        {1, 10, {2, 3}, true, false},
        {2, kSourceSchemaNoId, {2, 3}, true, true},
    };
    schema.selectors = {
        {10, 1, {}, {exact("embed.weight")}, {2, 3}, 1},
    };
    schema.aliases = {
        {1, 2, SourceAliasKind::TiedOutput, 77},
    };
    return schema;
}

SourceSchema separate_schema() {
    SourceSchema schema;
    schema.source_format = SourceFormat::Gguf;
    schema.slots = {
        {1, 10, {2, 3}, true, false},
        {2, 11, {2, 3}, true, true},
    };
    schema.selectors = {
        {10, 1, {}, {exact("embed.weight")}, {2, 3}, 1},
        {11, 2, {}, {exact("lm_head.weight")}, {2, 3}, 1},
    };
    return schema;
}

void test_version_and_data_only_result() {
    const SourceSchemaVersion expected_version{1, 0};
    CHECK(SourceSchema::version() == expected_version);
    CHECK(SourceSchemaEvaluator::version() == expected_version);

    const SourcePackageEvidence package = package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
    });
    const SourceSchemaResult result = compile_source_schema(tied_schema(), package);
    CHECK(result.success());
    CHECK(result.compiled.bindings.size() == 2);
    CHECK(result.compiled.bindings[0].physical_tensor_id == 17);
    CHECK(result.compiled.bindings[1].physical_tensor_id == 17);
    CHECK(result.compiled.bindings[0].span == result.compiled.bindings[1].span);
    CHECK(result.compiled.aliases.size() == 1);
    CHECK(result.compiled.aliases[0].kind == SourceAliasKind::TiedOutput);
}

void test_alias_is_required_and_equal_bytes_do_not_tie() {
    SourceSchema no_alias = tied_schema();
    no_alias.aliases.clear();
    const SourceSchemaResult missing = compile_source_schema(
        no_alias, package_with({tensor(17, "embed.weight", {3, 64, 24})}));
    CHECK(missing.error == SourceSchemaError::MissingOutput);

    SourceSchema separate = separate_schema();
    SourcePackageEvidence equal_bytes = package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
        tensor(18, "lm_head.weight", {3, 128, 24}),
    });
    equal_bytes.tensors[1].content_digest = equal_bytes.tensors[0].content_digest;
    const SourceSchemaResult independent = compile_source_schema(separate, equal_bytes);
    CHECK(independent.success());
    CHECK(independent.compiled.bindings[0].physical_tensor_id !=
          independent.compiled.bindings[1].physical_tensor_id);

    SourcePackageEvidence shared_span = equal_bytes;
    shared_span.tensors[1].span = shared_span.tensors[0].span;
    const SourceSchemaResult inferred = compile_source_schema(separate, shared_span);
    CHECK(inferred.error == SourceSchemaError::AliasRequired);

    SourceSchema incomplete_alias = tied_schema();
    incomplete_alias.aliases[0].proof_key = kSourceSchemaNoId;
    CHECK(compile_source_schema(incomplete_alias, package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
    })).error == SourceSchemaError::AliasInvalid);
}

void test_declared_alias_requires_the_same_span() {
    SourceSchema schema = separate_schema();
    schema.aliases.push_back({1, 2, SourceAliasKind::TiedOutput, 77});
    SourcePackageEvidence shared = package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
        tensor(18, "lm_head.weight", {3, 128, 24}),
    });
    shared.tensors[1].span = shared.tensors[0].span;
    const SourceSchemaResult accepted = compile_source_schema(schema, shared);
    CHECK(accepted.success());
    CHECK(accepted.compiled.bindings[0].physical_tensor_id == 17);
    CHECK(accepted.compiled.bindings[1].physical_tensor_id == 18);

    const SourceSchemaResult mismatch = compile_source_schema(schema, package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
        tensor(18, "lm_head.weight", {3, 128, 24}),
    }));
    CHECK(mismatch.error == SourceSchemaError::AliasMismatch);
}

void test_exact_cardinality_and_extra_tensors() {
    SourceSchema schema = tied_schema();
    schema.selectors[0].exact_cardinality = 2;
    const SourceSchemaResult duplicate = compile_source_schema(schema, package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
        tensor(18, "embed.weight", {3, 128, 24}),
    }));
    CHECK(duplicate.error == SourceSchemaError::InvalidSelector);

    const SourceSchemaResult extra = compile_source_schema(tied_schema(), package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
        tensor(18, "unmapped.weight", {3, 128, 24}),
    }));
    CHECK(extra.error == SourceSchemaError::UnmatchedTensor);
}

void test_bounded_decimal_selector() {
    SourceSchema schema;
    schema.slots = {
        {4, 40, {2, 3}, true, false},
        {5, 40, {2, 3}, true, true},
    };
    SourceNamePattern pattern;
    pattern.kind = SourceNamePatternKind::AnchoredDecimal;
    pattern.prefix = "blk.";
    pattern.suffix = ".weight";
    schema.selectors = {{40, kSourceSchemaNoId, {4, 5}, {pattern}, {2, 3}, 2}};
    const SourceSchemaResult accepted = compile_source_schema(schema, package_with({
        tensor(32, "blk.0.weight", {3, 64, 24}),
        tensor(31, "blk.1.weight", {3, 128, 24}),
    }));
    CHECK(accepted.success());
    CHECK(accepted.compiled.bindings[0].physical_tensor_id == 32);
    CHECK(accepted.compiled.bindings[1].physical_tensor_id == 31);

    const SourceSchemaResult leading_zero = compile_source_schema(schema, package_with({
        tensor(31, "blk.00.weight", {3, 64, 24}),
        tensor(32, "blk.1.weight", {3, 128, 24}),
    }));
    CHECK(leading_zero.error == SourceSchemaError::SelectorCardinalityMismatch);

    const SourceSchemaResult out_of_domain = compile_source_schema(schema, package_with({
        tensor(31, "blk.1.weight", {3, 64, 24}),
        tensor(32, "blk.2.weight", {3, 128, 24}),
    }));
    CHECK(out_of_domain.error == SourceSchemaError::SelectorNotMatched);
}

void test_schema_dimensions_and_outputs_are_proved() {
    SourceSchema wrong_dimensions = separate_schema();
    wrong_dimensions.slots[0].dimensions = {9, 9};
    CHECK(compile_source_schema(wrong_dimensions, package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
        tensor(18, "lm_head.weight", {3, 128, 24}),
    })).error == SourceSchemaError::InvalidSelector);

    SourcePackageEvidence wrong_tensor_dimensions = package_with({
        tensor(17, "embed.weight", {3, 64, 324}),
    });
    wrong_tensor_dimensions.tensors[0].dimensions = {9, 9};
    CHECK(compile_source_schema(tied_schema(), wrong_tensor_dimensions).error ==
          SourceSchemaError::SelectorNotMatched);

    SourceSchema no_output = tied_schema();
    no_output.slots[1].output = false;
    CHECK(compile_source_schema(no_output, package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
    })).error == SourceSchemaError::MissingOutput);
}

void test_proof_coordinates_do_not_change_graph_identity() {
    SourcePackageEvidence package = package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
    });
    SourceSchema left = tied_schema();
    SourceSchema right = left;
    right.aliases[0].proof_key = 78;
    const SourceSchemaResult a = compile_source_schema(left, package);
    package.alias_proofs[0].proof_key = 78;
    const SourceSchemaResult b = compile_source_schema(right, package);
    CHECK(a.success());
    CHECK(b.success());
    CHECK(a.compiled.graph_bytes == b.compiled.graph_bytes);

    SourceSchema unproved = tied_schema();
    unproved.aliases[0].proof_key = 999;
    CHECK(compile_source_schema(unproved, package).error == SourceSchemaError::AliasInvalid);
}

void test_registry_limits_are_terminal() {
    SourceSchema limited = separate_schema();
    limited.maximum_expansion_steps = 1;
    const std::array<SourceSchema, 2> schemas = {limited, separate_schema()};
    const SourceSchemaResult refused = evaluate_source_schemas(schemas, package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
        tensor(18, "lm_head.weight", {3, 128, 24}),
    }));
    CHECK(refused.error == SourceSchemaError::SchemaLimitExceeded);

    std::array<SourceSchema, 65> too_many{};
    too_many.fill(tied_schema());
    CHECK(evaluate_source_schemas(too_many, package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
    })).error == SourceSchemaError::SchemaLimitExceeded);
}

void test_artifact_closure_rejects_bad_spans() {
    const SourceSchema schema = tied_schema();
    SourcePackageEvidence missing = package_with({tensor(17, "embed.weight", {4, 64, 24})});
    CHECK(compile_source_schema(schema, missing).error == SourceSchemaError::InvalidPhysicalSpan);

    SourcePackageEvidence zero = package_with({tensor(17, "embed.weight", {3, 64, 0})});
    CHECK(compile_source_schema(schema, zero).error == SourceSchemaError::InvalidPhysicalSpan);

    SourcePackageEvidence unknown = package_with({tensor(17, "embed.weight", {99, 64, 24})});
    CHECK(compile_source_schema(schema, unknown).error == SourceSchemaError::InvalidPhysicalSpan);

    SourcePackageEvidence outside = package_with({tensor(17, "embed.weight", {3, 4090, 24})});
    CHECK(compile_source_schema(schema, outside).error == SourceSchemaError::InvalidPhysicalSpan);

    SourcePackageEvidence overflow = package_with({tensor(17, "embed.weight",
                                                             {3, UINT64_MAX - 3, 24})});
    CHECK(compile_source_schema(schema, overflow).error == SourceSchemaError::InvalidPhysicalSpan);
}

void test_block_quantized_physical_spans() {
    SourceSchema schema;
    schema.source_format = SourceFormat::Gguf;
    schema.slots = {{1, 10, {256, 2}, true, true}};
    schema.selectors = {{10, 1, {}, {exact("quant.weight")}, {256, 2}, 1}};

    SourceTensorEvidence quant = tensor(17, "quant.weight", {3, 64, 288});
    quant.dimensions = {256, 2};
    quant.physical_type_code = 12;
    quant.physical_block_elements = 256;
    quant.physical_block_bytes = 144;
    const SourceSchemaResult accepted = compile_source_schema(schema, package_with({quant}));
    CHECK(accepted.success());
    CHECK(accepted.compiled.bindings[0].physical_type_code == 12);
    CHECK(accepted.compiled.bindings[0].physical_block_axis == 0);
    CHECK(accepted.compiled.bindings[0].physical_block_elements == 256);
    CHECK(accepted.compiled.bindings[0].physical_block_bytes == 144);

    SourceTensorEvidence other_type = quant;
    other_type.physical_type_code = 13;
    const SourceSchemaResult differently_typed =
        compile_source_schema(schema, package_with({other_type}));
    CHECK(differently_typed.success());
    CHECK(differently_typed.compiled.binding_bytes != accepted.compiled.binding_bytes);

    SourceTensorEvidence wrong_length = quant;
    wrong_length.span.length = 289;
    CHECK(compile_source_schema(schema, package_with({wrong_length})).error ==
          SourceSchemaError::InvalidPhysicalSpan);

    SourceSchema misaligned_schema = schema;
    misaligned_schema.slots[0].dimensions = {384, 2};
    misaligned_schema.selectors[0].source_dimensions = {384, 2};
    SourceTensorEvidence misaligned = quant;
    misaligned.dimensions = {384, 2};
    misaligned.span.length = 432;
    CHECK(compile_source_schema(misaligned_schema, package_with({misaligned})).error ==
          SourceSchemaError::InvalidPhysicalSpan);
}

void test_alias_direction_and_cycles_are_rejected() {
    SourceSchema not_output = tied_schema();
    not_output.slots[1].output = false;
    SourcePackageEvidence package = package_with({tensor(17, "embed.weight", {3, 64, 24})});
    CHECK(compile_source_schema(not_output, package).error == SourceSchemaError::MissingOutput);

    SourceSchema optional_output = tied_schema();
    optional_output.slots[1].required = false;
    CHECK(compile_source_schema(optional_output, package).error == SourceSchemaError::MissingOutput);

    SourceSchema cycle = separate_schema();
    cycle.slots[0].output = true;
    cycle.slots[1].output = false;
    cycle.aliases = {
        {1, 2, SourceAliasKind::ExactSharedSpan, kSourceSchemaNoId},
        {2, 1, SourceAliasKind::ExactSharedSpan, kSourceSchemaNoId},
    };
    SourcePackageEvidence shared = package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
        tensor(18, "lm_head.weight", {3, 64, 24}),
    });
    shared.tensors[1].span = shared.tensors[0].span;
    CHECK(compile_source_schema(cycle, shared).error == SourceSchemaError::AliasInvalid);
}

void test_deterministic_work_budget() {
    SourceSchema schema = separate_schema();
    schema.maximum_expansion_steps = 3;
    CHECK(compile_source_schema(schema, package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
        tensor(18, "lm_head.weight", {3, 128, 24}),
    })).error == SourceSchemaError::SchemaLimitExceeded);
    CHECK(compile_source_schema(schema, package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
        tensor(18, "lm_head.weight", {3, 128, 24}),
    })).error == SourceSchemaError::SchemaLimitExceeded);
}

void test_renames_are_schema_data_and_two_matches_are_ambiguous() {
    SourceSchema renamed = tied_schema();
    renamed.selectors[0].names = {exact("renamed.embedding")};
    const SourcePackageEvidence renamed_package = package_with({
        tensor(17, "renamed.embedding", {3, 64, 24}),
    });
    const SourceSchemaResult accepted = compile_source_schema(renamed, renamed_package);
    CHECK(accepted.success());
    const SourceSchemaResult original = compile_source_schema(tied_schema(), package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
    }));
    CHECK(original.success());
    CHECK(original.compiled.graph_bytes == accepted.compiled.graph_bytes);

    const SourceSchemaResult old_spelling = compile_source_schema(renamed,
        package_with({tensor(17, "embed.weight", {3, 64, 24})}));
    CHECK(old_spelling.error == SourceSchemaError::SelectorCardinalityMismatch);

    SourceSchema alternate = tied_schema();
    alternate.selectors[0].names.push_back(exact("renamed.embedding"));
    const std::array<SourceSchema, 2> schemas = {renamed, alternate};
    const SourceSchemaResult ambiguous = evaluate_source_schemas(schemas, renamed_package);
    CHECK(ambiguous.error == SourceSchemaError::AmbiguousSchemas);
    CHECK(ambiguous.matching_schema_count == 2);
    CHECK(ambiguous.compiled.bindings.empty());
}

void test_version_and_format_refusals() {
    SourceSchema bad_version = tied_schema();
    bad_version.schema_major = 2;
    CHECK(compile_source_schema(bad_version, package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
    })).error == SourceSchemaError::SchemaVersionUnsupported);

    SourceSchema bad_format = tied_schema();
    bad_format.source_format = SourceFormat::Mlx;
    CHECK(compile_source_schema(bad_format, package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
    })).error == SourceSchemaError::SourceFormatMismatch);

    const std::array<SourceSchema, 1> none = {bad_format};
    CHECK(evaluate_source_schemas(none, package_with({
        tensor(17, "embed.weight", {3, 64, 24}),
    })).error == SourceSchemaError::NoMatchingSchema);
}

} // namespace

int main() {
    test_version_and_data_only_result();
    test_alias_is_required_and_equal_bytes_do_not_tie();
    test_declared_alias_requires_the_same_span();
    test_exact_cardinality_and_extra_tensors();
    test_bounded_decimal_selector();
    test_schema_dimensions_and_outputs_are_proved();
    test_proof_coordinates_do_not_change_graph_identity();
    test_registry_limits_are_terminal();
    test_artifact_closure_rejects_bad_spans();
    test_block_quantized_physical_spans();
    test_alias_direction_and_cycles_are_rejected();
    test_deterministic_work_budget();
    test_renames_are_schema_data_and_two_matches_are_ambiguous();
    test_version_and_format_refusals();
    std::cout << "all source schema checks passed\n";
    return 0;
}
