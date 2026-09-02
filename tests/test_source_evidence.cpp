// test_source_evidence - coverage for summarize_source_program, the only
// production module with no test executable today. It checks the summary
// counts and that every digest field is exactly the corresponding
// package-component digest, so drift between the summary and the package
// cannot go unnoticed.

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "artifact_index.h"
#include "artifact_set.h"
#include "physical_program.h"
#include "program_ir.h"
#include "program_package.h"
#include "program_state.h"
#include "source_evidence.h"
#include "test_util.h"
#include "token_program.h"

using namespace Laplace;

namespace {

ValueType scalar_type(ElementType type) { return {type, {}}; }

ValueType tensor_type(ElementType type, uint64_t extent) {
    return {type, {{DimensionExpression::Constant, extent, {}}}};
}

Program semantic_program() {
    Function token;
    token.id = 10;
    token.entry_region_id = 11;
    token.regions = {{11, {{100, scalar_type(ElementType::U32)}}, {}, {100}}};
    token.result_types = {scalar_type(ElementType::U32)};

    Function scores;
    scores.id = 20;
    scores.entry_region_id = 21;
    scores.regions = {{21, {{200, tensor_type(ElementType::F32, 10)}}, {}, {200}}};
    scores.result_types = {tensor_type(ElementType::F32, 10)};

    const ValueType state_type{
        ElementType::F16,
        {{DimensionExpression::Parameter, 5, {}}}};
    Instruction read;
    read.id = 31;
    read.primitive = {Primitive::StateRead, 1, 0};
    read.attributes = StateAttributes{30};
    read.outputs = {{300, state_type}};
    Instruction write;
    write.id = 32;
    write.primitive = {Primitive::StateWrite, 1, 0};
    write.inputs = {300};
    write.effect_predecessors = {31};
    write.attributes = StateAttributes{30};
    Function state;
    state.id = 33;
    state.entry_region_id = 34;
    state.regions = {{34, {}, {read, write}, {}}};

    Program program;
    program.dimension_parameters = {{5, 2, 3}};
    program.state_references = {{30, state_type, UINT32_MAX, true}};
    program.functions = {token, scores, state};
    program.exports = {
        {10, 0, scalar_type(ElementType::U32)},
        {20, 0, tensor_type(ElementType::F32, 10)},
    };
    return program;
}

StateSchema state_schema() {
    StateSchema schema;
    schema.dimension_bindings = {{5, 2}};
    schema.slots = {{30, ElementType::F16, {2}, 7}};
    return schema;
}

TokenProgramDefinition token_definition() {
    TokenProgramDefinition definition;
    for (unsigned value = 0; value != 256; ++value)
        definition.byte_map[value] = static_cast<uint8_t>(value);
    definition.unknown_token_id = 0;
    definition.vocabulary = {
        {"<unk>", static_cast<uint16_t>(VocabFlags::Special), 0},
        {"a", 0, 0}, {"b", 0, 0}, {"ab", 0, 0}, {" ", 0, 0},
        {"A", 0, 0},
        {"<bos>", static_cast<uint16_t>(VocabFlags::Special), 0},
        {"<eos>", static_cast<uint16_t>(VocabFlags::Special), 0},
        {"x", 0, 0},
        {"<x>", static_cast<uint16_t>(VocabFlags::Special), 0},
    };
    definition.merges = {{1, 2, 3, 0}};
    definition.prompt = {
        {PromptOpcode::EmitUserText, {}},
        {PromptOpcode::EmitGenerationPrompt, "!"},
        {PromptOpcode::End, {}},
    };
    definition.prompt_max_bytes = 128;
    definition.normalizer.kind = NormalizerKind::None;
    definition.pretokenizer.kind = PretokenizerKind::ByteLevel;
    definition.postprocessor.kind = PostprocessorKind::None;
    definition.decoder.kind = DecoderKind::ByteLevel;
    return definition;
}

PhysicalProgram physical_program() {
    PhysicalProgram program;
    program.logical_rank = 1;
    program.planes.push_back({PhysicalPlaneStorage::External, 1, 0, 0});
    program.policies.push_back({});
    program.instructions = {
        {PhysicalOpcode::Coordinate, PhysicalValueType::Index,
         {kNoPhysicalValue, kNoPhysicalValue, kNoPhysicalValue},
         kNoPhysicalPlane, kNoPhysicalPolicy, 0, 0, PhysicalBitOrder::Lsb0Little},
        {PhysicalOpcode::ConstIndex, PhysicalValueType::Index,
         {kNoPhysicalValue, kNoPhysicalValue, kNoPhysicalValue},
         kNoPhysicalPlane, kNoPhysicalPolicy, 32, 0, PhysicalBitOrder::Lsb0Little},
        {PhysicalOpcode::IndexMultiply, PhysicalValueType::Index,
         {0, 1, kNoPhysicalValue}, kNoPhysicalPlane, kNoPhysicalPolicy, 0, 0,
         PhysicalBitOrder::Lsb0Little},
        {PhysicalOpcode::LoadBits, PhysicalValueType::U32,
         {2, kNoPhysicalValue, kNoPhysicalValue}, 0, kNoPhysicalPolicy, 0, 32,
         PhysicalBitOrder::Lsb0Little},
        {PhysicalOpcode::BitsToF32, PhysicalValueType::F32,
         {3, kNoPhysicalValue, kNoPhysicalValue}, kNoPhysicalPlane, 0, 0, 0,
         PhysicalBitOrder::Lsb0Little},
    };
    program.result = 4;
    return program;
}

std::variant<VerifiedProgramPackage, CompatibilityReport> build_package() {
    auto token_wire = serialize_token_program(token_definition());
    CHECK(std::holds_alternative<std::vector<uint8_t>>(token_wire));
    if (!std::holds_alternative<std::vector<uint8_t>>(token_wire)) std::abort();

    std::vector<uint8_t> weights;
    for (uint32_t index = 0; index != 10; ++index) {
        const uint32_t bits = std::bit_cast<uint32_t>(static_cast<float>(index));
        for (unsigned shift = 0; shift != 32; shift += 8)
            weights.push_back(static_cast<uint8_t>(bits >> shift));
    }
    auto weight_view = ArtifactSet::make_owned_blob(
        ArtifactId{7}, ArtifactRole::Primary, weights);
    auto token_view = ArtifactSet::make_owned_blob(
        ArtifactId{2}, ArtifactRole::Shard,
        std::get<std::vector<uint8_t>>(token_wire));
    CHECK(std::holds_alternative<PackageView>(weight_view));
    CHECK(std::holds_alternative<PackageView>(token_view));
    if (!std::holds_alternative<PackageView>(weight_view) ||
        !std::holds_alternative<PackageView>(token_view))
        std::abort();

    const Sha256Digest token_digest = std::get<PackageView>(token_view).digest();
    ArtifactIndexInput input;
    input.artifacts.push_back(std::get<PackageView>(std::move(weight_view)));
    input.artifacts.push_back(std::get<PackageView>(std::move(token_view)));
    auto built_index = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<ArtifactIndex>(built_index));
    if (!std::holds_alternative<ArtifactIndex>(built_index)) std::abort();

    auto verified_program = verify_and_canonicalize_program(semantic_program());
    CHECK(std::holds_alternative<VerifiedProgram>(verified_program));
    if (!std::holds_alternative<VerifiedProgram>(verified_program)) std::abort();
    auto verified_state = verify_state_schema(state_schema(),
                                              std::get<VerifiedProgram>(verified_program));
    CHECK(std::holds_alternative<VerifiedStateSchema>(verified_state));
    if (!std::holds_alternative<VerifiedStateSchema>(verified_state)) std::abort();

    const PhysicalProgram physical = physical_program();
    auto physical_wire = encode_physical_program(physical);
    auto physical_digest = physical_program_digest(physical);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(physical_wire));
    CHECK(std::holds_alternative<PhysicalProgramDigest>(physical_digest));
    if (!std::holds_alternative<std::vector<uint8_t>>(physical_wire) ||
        !std::holds_alternative<PhysicalProgramDigest>(physical_digest))
        std::abort();

    PhysicalProgramRecord record{
        std::get<PhysicalProgramDigest>(physical_digest),
        std::get<std::vector<uint8_t>>(std::move(physical_wire)),
        {ElementType::F32, {10}}};
    PhysicalResourceBinding resource;
    resource.resource_id = 1;
    resource.program_digest = record.digest;
    resource.semantic_function_id = 20;
    resource.semantic_value_id = 200;
    resource.planes = {{0, ArtifactId{7}, 0, weights.size()}};

    const std::vector<uint8_t> token_wire_bytes =
        std::get<std::vector<uint8_t>>(std::move(token_wire));
    std::vector<TokenEndpointBinding> token_bindings = {
        {TokenEndpointKind::InputToken, 10, 100},
        {TokenEndpointKind::OutputScores, 20, 0},
    };
    std::vector<PhysicalProgramRecord> records = {std::move(record)};
    std::vector<PhysicalResourceBinding> resources = {std::move(resource)};
    return build_program_package(
        std::get<ArtifactIndex>(std::move(built_index)),
        std::get<VerifiedProgram>(std::move(verified_program)),
        std::get<VerifiedStateSchema>(std::move(verified_state)),
        {ArtifactId{2}, 0, token_wire_bytes.size(), token_digest},
        token_bindings, records, resources);
}

int32_t states_count(const SourceProgramSummary& summary) {
    return summary.primitive_occurrences[static_cast<size_t>(Primitive::StateRead)] +
           summary.primitive_occurrences[static_cast<size_t>(Primitive::StateWrite)];
}

void test_summary_counts() {
    auto result = build_package();
    CHECK(std::holds_alternative<VerifiedProgramPackage>(result));
    if (!std::holds_alternative<VerifiedProgramPackage>(result)) return;
    const VerifiedProgramPackage& package = std::get<VerifiedProgramPackage>(result);
    const SourceProgramSummaryResult summary_result = summarize_source_program(package);
    CHECK(std::holds_alternative<SourceProgramSummary>(summary_result));
    if (!std::holds_alternative<SourceProgramSummary>(summary_result)) return;
    const SourceProgramSummary& summary = std::get<SourceProgramSummary>(summary_result);

    CHECK(summary.function_count == 3);
    CHECK(summary.export_count == 2);
    CHECK(summary.state_reference_count == 1);
    CHECK(summary.physical_program_count == 1);
    CHECK(summary.physical_resource_count == 1);
    CHECK(states_count(summary) == 2);  // one StateRead, one StateWrite
    uint64_t total_instructions = 0;
    for (uint64_t occurrence : summary.primitive_occurrences)
        total_instructions += occurrence;
    CHECK(total_instructions == 2);
}

void test_summary_digests_match_package() {
    auto result = build_package();
    CHECK(std::holds_alternative<VerifiedProgramPackage>(result));
    if (!std::holds_alternative<VerifiedProgramPackage>(result)) return;
    const VerifiedProgramPackage& package = std::get<VerifiedProgramPackage>(result);
    const SourceProgramSummaryResult summary_result = summarize_source_program(package);
    CHECK(std::holds_alternative<SourceProgramSummary>(summary_result));
    if (!std::holds_alternative<SourceProgramSummary>(summary_result)) return;
    const SourceProgramSummary& summary = std::get<SourceProgramSummary>(summary_result);

    CHECK(summary.package_digest == package.digest());
    CHECK(summary.semantic_program_digest.bytes ==
          program_digest(package.semantic_program()).bytes);
    CHECK(summary.state_schema_digest.bytes ==
          state_schema_digest(package.state_schema()).bytes);
    CHECK(summary.physical_package_digest == package.physical_package().digest());
    CHECK(summary.token_vocabulary_digest == package.token_program().vocabulary_digest());
    CHECK(summary.token_prompt_digest == package.token_program().prompt_digest());

    // Summaries are deterministic: a second pass must reproduce every field.
    const SourceProgramSummaryResult again = summarize_source_program(package);
    CHECK(std::holds_alternative<SourceProgramSummary>(again));
    if (!std::holds_alternative<SourceProgramSummary>(again)) return;
    CHECK(std::get<SourceProgramSummary>(again) == summary);
}

void test_program_package_wire_encodes() {
    // Smoke check that a summarized package goes through the wire encoder.
    auto result = build_package();
    if (!std::holds_alternative<VerifiedProgramPackage>(result)) return;
    const VerifiedProgramPackage& package = std::get<VerifiedProgramPackage>(result);
    auto encoded = encode_program_package(package);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
}

} // namespace

int main() {
    test_summary_counts();
    test_summary_digests_match_package();
    test_program_package_wire_encodes();
    return test_summary("test_source_evidence");
}
