#include "program_package.h"
#include "container_schema_program.h"
#include "test_util.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <variant>
#include <vector>

using namespace Laplace;

namespace {

ValueType scalar(ElementType type) { return {type, {}}; }

ValueType tensor(ElementType type, uint64_t extent) {
    return {type, {{DimensionExpression::Constant, extent, {}}}};
}

Program semantic_program() {
    Function token;
    token.id = 10;
    token.entry_region_id = 11;
    token.regions = {{11, {{100, scalar(ElementType::U32)}}, {}, {100}}};
    token.result_types = {scalar(ElementType::U32)};

    Function scores;
    scores.id = 20;
    scores.entry_region_id = 21;
    scores.regions = {{21, {{200, tensor(ElementType::F32, 10)}}, {}, {200}}};
    scores.result_types = {tensor(ElementType::F32, 10)};

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
        {10, 0, scalar(ElementType::U32)},
        {20, 0, tensor(ElementType::F32, 10)},
    };
    return program;
}

Program renamed_semantic_program() {
    Program program = semantic_program();
    program.dimension_parameters[0].id = 55;
    program.state_references[0].id = 3000;
    program.state_references[0].type.dimensions[0].value = 55;

    Function& token = program.functions[0];
    token.id = 1000;
    token.entry_region_id = 1100;
    token.regions[0].id = 1100;
    token.regions[0].arguments[0].id = 10000;
    token.regions[0].yields[0] = 10000;

    Function& scores = program.functions[1];
    scores.id = 2000;
    scores.entry_region_id = 2100;
    scores.regions[0].id = 2100;
    scores.regions[0].arguments[0].id = 20000;
    scores.regions[0].yields[0] = 20000;

    Function& state = program.functions[2];
    state.id = 3300;
    state.entry_region_id = 3400;
    state.regions[0].id = 3400;
    Instruction& read = state.regions[0].instructions[0];
    read.id = 3100;
    std::get<StateAttributes>(read.attributes).state_id = 3000;
    read.outputs[0].id = 30000;
    read.outputs[0].type.dimensions[0].value = 55;
    Instruction& write = state.regions[0].instructions[1];
    write.id = 3200;
    write.inputs[0] = 30000;
    write.effect_predecessors[0] = 3100;
    std::get<StateAttributes>(write.attributes).state_id = 3000;

    program.exports[0].function_id = 1000;
    program.exports[1].function_id = 2000;
    std::reverse(program.functions.begin(), program.functions.end());
    std::reverse(program.exports.begin(), program.exports.end());
    return program;
}

StateSchema state_schema(uint64_t extent = 2) {
    StateSchema schema;
    schema.dimension_bindings = {{5, extent}};
    schema.slots = {{30, ElementType::F16, {extent}, 7}};
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
         kNoPhysicalPlane, kNoPhysicalPolicy, 0, 0,
         PhysicalBitOrder::Lsb0Little},
        {PhysicalOpcode::ConstIndex, PhysicalValueType::Index,
         {kNoPhysicalValue, kNoPhysicalValue, kNoPhysicalValue},
         kNoPhysicalPlane, kNoPhysicalPolicy, 32, 0,
         PhysicalBitOrder::Lsb0Little},
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

struct Fixture {
    ArtifactIndex index;
    VerifiedProgram program;
    VerifiedStateSchema state;
    TokenProgramSource token_source;
    std::vector<TokenEndpointBinding> token_bindings;
    std::vector<PhysicalProgramRecord> records;
    std::vector<PhysicalResourceBinding> resources;
};

Fixture fixture(uint32_t weight_artifact = 7, uint32_t token_artifact = 2) {
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
        ArtifactId{weight_artifact}, ArtifactRole::Primary, weights);
    auto token_view = ArtifactSet::make_owned_blob(
        ArtifactId{token_artifact}, ArtifactRole::Shard,
        std::get<std::vector<uint8_t>>(token_wire));
    CHECK(std::holds_alternative<PackageView>(weight_view));
    CHECK(std::holds_alternative<PackageView>(token_view));
    if (!std::holds_alternative<PackageView>(weight_view) ||
        !std::holds_alternative<PackageView>(token_view))
        std::abort();
    const Sha256Digest token_digest =
        std::get<PackageView>(token_view).digest();
    ArtifactIndexInput input;
    input.artifacts.push_back(std::get<PackageView>(std::move(weight_view)));
    input.artifacts.push_back(std::get<PackageView>(std::move(token_view)));
    auto built_index = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<ArtifactIndex>(built_index));
    if (!std::holds_alternative<ArtifactIndex>(built_index)) std::abort();

    auto verified_program = verify_and_canonicalize_program(semantic_program());
    CHECK(std::holds_alternative<VerifiedProgram>(verified_program));
    if (!std::holds_alternative<VerifiedProgram>(verified_program)) std::abort();
    VerifiedProgram program =
        std::get<VerifiedProgram>(std::move(verified_program));
    auto verified_state = verify_state_schema(state_schema(), program);
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
    resource.planes = {{0, ArtifactId{weight_artifact}, 0, weights.size()}};
    return {
        std::get<ArtifactIndex>(std::move(built_index)),
        std::move(program),
        std::get<VerifiedStateSchema>(std::move(verified_state)),
        {ArtifactId{token_artifact}, 0,
         std::get<std::vector<uint8_t>>(token_wire).size(), token_digest},
        {{TokenEndpointKind::InputToken, 10, 100},
         {TokenEndpointKind::OutputScores, 20, 0}},
        {std::move(record)},
        {std::move(resource)},
    };
}

Fixture renamed_fixture() {
    Fixture result = fixture(70, 20);
    auto program = verify_and_canonicalize_program(renamed_semantic_program());
    CHECK(std::holds_alternative<VerifiedProgram>(program));
    if (!std::holds_alternative<VerifiedProgram>(program)) std::abort();
    result.program = std::get<VerifiedProgram>(std::move(program));
    StateSchema schema;
    schema.dimension_bindings = {{55, 2}};
    schema.slots = {{3000, ElementType::F16, {2}, 91}};
    auto state = verify_state_schema(std::move(schema), result.program);
    CHECK(std::holds_alternative<VerifiedStateSchema>(state));
    if (!std::holds_alternative<VerifiedStateSchema>(state)) std::abort();
    result.state = std::get<VerifiedStateSchema>(std::move(state));
    result.token_bindings = {
        {TokenEndpointKind::InputToken, 1000, 10000},
        {TokenEndpointKind::OutputScores, 2000, 0},
    };
    result.resources[0].semantic_function_id = 2000;
    result.resources[0].semantic_value_id = 20000;
    return result;
}

ProgramPackageResult build(Fixture source) {
    return build_program_package(
        std::move(source.index), std::move(source.program),
        std::move(source.state), source.token_source, source.token_bindings,
        source.records, source.resources);
}

ContainerSchemaProgram package_container_schema(uint32_t section_id = 23) {
    ContainerSchemaProgram program;
    program.register_count = 3;
    program.predicate_count = 1;
    ContainerSchemaInstruction match;
    match.opcode = ContainerSchemaOpcode::MatchBytes;
    match.literal = {'P', 'K', 'G', 'X'};
    ContainerSchemaInstruction length;
    length.opcode = ContainerSchemaOpcode::ReadU32Le;
    length.destination = 0;
    ContainerSchemaInstruction cursor;
    cursor.opcode = ContainerSchemaOpcode::CaptureCursor;
    cursor.destination = 1;
    ContainerSchemaInstruction emit;
    emit.opcode = ContainerSchemaOpcode::EmitRange;
    emit.input_a = 1;
    emit.input_b = 0;
    emit.section_id = section_id;
    ContainerSchemaInstruction advance;
    advance.opcode = ContainerSchemaOpcode::Advance;
    advance.input_a = 0;
    ContainerSchemaInstruction end;
    end.opcode = ContainerSchemaOpcode::RequireCursorEnd;
    program.instructions = {
        std::move(match), length, cursor, emit, advance, end};
    return program;
}

std::vector<uint8_t> package_container(std::span<const uint8_t> package) {
    std::vector<uint8_t> result = {'P', 'K', 'G', 'X'};
    const uint32_t size = static_cast<uint32_t>(package.size());
    for (unsigned shift = 0; shift != 32; shift += 8)
        result.push_back(static_cast<uint8_t>(size >> shift));
    result.insert(result.end(), package.begin(), package.end());
    return result;
}

ContainerSchemaProgram duplicate_package_schema() {
    ContainerSchemaProgram program;
    program.register_count = 3;
    program.predicate_count = 1;
    ContainerSchemaInstruction match;
    match.opcode = ContainerSchemaOpcode::MatchBytes;
    match.literal = {'D', 'U', 'P', 'X'};
    ContainerSchemaInstruction length;
    length.opcode = ContainerSchemaOpcode::ReadU32Le;
    length.destination = 0;
    ContainerSchemaInstruction first_cursor;
    first_cursor.opcode = ContainerSchemaOpcode::CaptureCursor;
    first_cursor.destination = 1;
    ContainerSchemaInstruction first_emit;
    first_emit.opcode = ContainerSchemaOpcode::EmitRange;
    first_emit.input_a = 1;
    first_emit.input_b = 0;
    first_emit.section_id = 23;
    ContainerSchemaInstruction first_advance;
    first_advance.opcode = ContainerSchemaOpcode::Advance;
    first_advance.input_a = 0;
    ContainerSchemaInstruction second_cursor;
    second_cursor.opcode = ContainerSchemaOpcode::CaptureCursor;
    second_cursor.destination = 2;
    ContainerSchemaInstruction second_emit = first_emit;
    second_emit.input_a = 2;
    ContainerSchemaInstruction second_advance = first_advance;
    ContainerSchemaInstruction end;
    end.opcode = ContainerSchemaOpcode::RequireCursorEnd;
    program.instructions = {
        std::move(match), length, first_cursor, first_emit, first_advance,
        second_cursor, second_emit, second_advance, end};
    return program;
}

std::vector<uint8_t> duplicate_package_container(
    std::span<const uint8_t> package) {
    std::vector<uint8_t> result = {'D', 'U', 'P', 'X'};
    const uint32_t size = static_cast<uint32_t>(package.size());
    for (unsigned shift = 0; shift != 32; shift += 8)
        result.push_back(static_cast<uint8_t>(size >> shift));
    result.insert(result.end(), package.begin(), package.end());
    result.insert(result.end(), package.begin(), package.end());
    return result;
}

ContainerSchemaProgram product_container_schema() {
    ContainerSchemaProgram program;
    program.register_count = 6;
    program.predicate_count = 1;
    ContainerSchemaInstruction match;
    match.opcode = ContainerSchemaOpcode::MatchBytes;
    match.literal = {'P', 'R', 'D', 'X'};
    ContainerSchemaInstruction weight_length;
    weight_length.opcode = ContainerSchemaOpcode::ReadU64Le;
    weight_length.destination = 0;
    ContainerSchemaInstruction token_length = weight_length;
    token_length.destination = 1;
    ContainerSchemaInstruction package_length = weight_length;
    package_length.destination = 2;
    const auto emit = [](uint32_t cursor_register, uint32_t length_register,
                         uint32_t section_id) {
        std::array<ContainerSchemaInstruction, 3> result;
        result[0].opcode = ContainerSchemaOpcode::CaptureCursor;
        result[0].destination = cursor_register;
        result[1].opcode = ContainerSchemaOpcode::EmitRange;
        result[1].input_a = cursor_register;
        result[1].input_b = length_register;
        result[1].section_id = section_id;
        result[2].opcode = ContainerSchemaOpcode::Advance;
        result[2].input_a = length_register;
        return result;
    };
    const auto weight = emit(3, 0, 41);
    const auto token = emit(4, 1, 42);
    const auto package = emit(5, 2, 23);
    ContainerSchemaInstruction end;
    end.opcode = ContainerSchemaOpcode::RequireCursorEnd;
    program.instructions = {
        match, weight_length, token_length, package_length,
        weight[0], weight[1], weight[2], token[0], token[1], token[2],
        package[0], package[1], package[2], end};
    return program;
}

std::vector<uint8_t> product_container(const ArtifactIndex& index,
                                       std::span<const uint8_t> package) {
    const auto find = [&](ArtifactId id) -> const PackageView* {
        for (const PackageView& artifact : index.artifacts())
            if (artifact.artifact_id() == id) return &artifact;
        return nullptr;
    };
    const PackageView* weights = find(ArtifactId{7});
    const PackageView* token = find(ArtifactId{2});
    CHECK(weights != nullptr);
    CHECK(token != nullptr);
    if (weights == nullptr || token == nullptr) return {};
    std::vector<uint8_t> result = {'P', 'R', 'D', 'X'};
    const auto append_u64 = [&](uint64_t value) {
        for (unsigned shift = 0; shift != 64; shift += 8)
            result.push_back(static_cast<uint8_t>(value >> shift));
    };
    append_u64(weights->bytes().size());
    append_u64(token->bytes().size());
    append_u64(package.size());
    result.insert(result.end(), weights->bytes().begin(), weights->bytes().end());
    result.insert(result.end(), token->bytes().begin(), token->bytes().end());
    result.insert(result.end(), package.begin(), package.end());
    return result;
}

void test_container_loader_builds_artifact_index() {
    Fixture source = fixture();
    auto package = build(source);
    CHECK(std::holds_alternative<VerifiedProgramPackage>(package));
    if (!std::holds_alternative<VerifiedProgramPackage>(package)) return;
    auto wire = encode_program_package(
        std::get<VerifiedProgramPackage>(package));
    CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
    if (!std::holds_alternative<std::vector<uint8_t>>(wire)) return;
    const auto container = product_container(
        source.index, std::get<std::vector<uint8_t>>(wire));
    auto container_view = ArtifactSet::make_owned_blob(
        ArtifactId{99}, ArtifactRole::Primary, container);
    CHECK(std::holds_alternative<PackageView>(container_view));
    if (!std::holds_alternative<PackageView>(container_view)) return;
    const std::array<ContainerSchemaProgram, 1> schemas = {
        product_container_schema()};
    const std::array<ContainerArtifactSection, 2> sections = {{
        {41, ArtifactId{7}, ArtifactRole::Primary},
        {42, ArtifactId{2}, ArtifactRole::Shard},
    }};
    auto loaded = load_container_program_package(
        std::get<PackageView>(container_view), schemas, sections, 23);
    CHECK(std::holds_alternative<VerifiedProgramPackage>(loaded));
    if (const auto* verified = std::get_if<VerifiedProgramPackage>(&loaded))
        CHECK(verified->digest() ==
              std::get<VerifiedProgramPackage>(package).digest());

    auto missing = sections;
    missing[1].section_id = 44;
    CHECK(std::holds_alternative<CompatibilityReport>(
        load_container_program_package(std::get<PackageView>(container_view),
                                       schemas, missing, 23)));
    auto duplicate = sections;
    duplicate[1].section_id = duplicate[0].section_id;
    CHECK(std::holds_alternative<CompatibilityReport>(
        load_container_program_package(std::get<PackageView>(container_view),
                                       schemas, duplicate, 23)));
    auto reused_id = sections;
    reused_id[1].artifact_id = reused_id[0].artifact_id;
    CHECK(std::holds_alternative<CompatibilityReport>(
        load_container_program_package(std::get<PackageView>(container_view),
                                       schemas, reused_id, 23)));

    ContainerSchemaProgram overlap = product_container_schema();
    overlap.register_count = 7;
    ContainerSchemaInstruction zero;
    zero.opcode = ContainerSchemaOpcode::SetConstant;
    zero.destination = 6;
    overlap.instructions.insert(overlap.instructions.begin() + 4, zero);
    overlap.instructions[7].input_a = 6;
    ContainerSchemaInstruction catch_up;
    catch_up.opcode = ContainerSchemaOpcode::Advance;
    catch_up.input_a = 0;
    overlap.instructions.insert(overlap.instructions.end() - 1, catch_up);
    const std::array<ContainerSchemaProgram, 1> overlap_schemas = {overlap};
    CHECK(std::holds_alternative<CompatibilityReport>(
        load_container_program_package(std::get<PackageView>(container_view),
                                       overlap_schemas, sections, 23)));

    auto oversized = container;
    std::fill(oversized.begin() + 4, oversized.begin() + 12, 0xff);
    auto oversized_view = ArtifactSet::make_owned_blob(
        ArtifactId{99}, ArtifactRole::Primary, oversized);
    CHECK(std::holds_alternative<PackageView>(oversized_view));
    if (std::holds_alternative<PackageView>(oversized_view)) {
        CHECK(std::holds_alternative<CompatibilityReport>(
            load_container_program_package(std::get<PackageView>(oversized_view),
                                           schemas, sections, 23)));
    }
}

void test_container_handoff_uses_verified_package_decoder() {
    auto source_package = build(fixture());
    CHECK(std::holds_alternative<VerifiedProgramPackage>(source_package));
    if (!std::holds_alternative<VerifiedProgramPackage>(source_package)) return;
    const auto encoded = encode_program_package(
        std::get<VerifiedProgramPackage>(source_package));
    CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
    if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) return;
    const auto container = package_container(
        std::get<std::vector<uint8_t>>(encoded));
    const std::array<ContainerSchemaProgram, 1> schemas = {
        package_container_schema()};

    Fixture destination = fixture();
    const auto loaded = decode_container_program_package(
        std::move(destination.index), schemas, container, 23);
    CHECK(std::holds_alternative<VerifiedProgramPackage>(loaded));
    if (const auto* verified = std::get_if<VerifiedProgramPackage>(&loaded))
        CHECK(verified->digest() ==
              std::get<VerifiedProgramPackage>(source_package).digest());

    const std::array<ContainerSchemaProgram, 1> missing = {
        package_container_schema(24)};
    Fixture missing_destination = fixture();
    CHECK(std::holds_alternative<CompatibilityReport>(
        decode_container_program_package(std::move(missing_destination.index),
                                         missing, container, 23)));

    auto corrupt = container;
    corrupt[8] ^= 1;
    Fixture corrupt_destination = fixture();
    CHECK(std::holds_alternative<CompatibilityReport>(
        decode_container_program_package(std::move(corrupt_destination.index),
                                         schemas, corrupt, 23)));

    const std::array<ContainerSchemaProgram, 1> duplicate_schema = {
        duplicate_package_schema()};
    const auto duplicate_container = duplicate_package_container(
        std::get<std::vector<uint8_t>>(encoded));
    Fixture duplicate_destination = fixture();
    const auto duplicate_result = decode_container_program_package(
        std::move(duplicate_destination.index), duplicate_schema,
        duplicate_container, 23);
    CHECK(std::holds_alternative<CompatibilityReport>(duplicate_result));
    if (const auto* failure =
            std::get_if<CompatibilityReport>(&duplicate_result))
        CHECK(failure->code == CompatibilityError::IMPORT_SCHEMA_AMBIGUOUS);
}

bool write_bytes(const char* path, std::span<const uint8_t> bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    return output.good();
}

std::vector<uint8_t> read_bytes(const char* path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

int emit_external_package(const char* path) {
    auto package = build(fixture());
    CHECK(std::holds_alternative<VerifiedProgramPackage>(package));
    if (!std::holds_alternative<VerifiedProgramPackage>(package))
        return test_summary("test_program_package_emit");
    const auto wire = encode_program_package(
        std::get<VerifiedProgramPackage>(package));
    CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
    if (const auto* bytes = std::get_if<std::vector<uint8_t>>(&wire))
        CHECK(write_bytes(path, *bytes));
    return test_summary("test_program_package_emit");
}

int emit_external_product_source(const char* package_path,
                                 const char* weights_path,
                                 const char* token_path) {
    Fixture source = fixture();
    auto package = build(source);
    CHECK(std::holds_alternative<VerifiedProgramPackage>(package));
    if (!std::holds_alternative<VerifiedProgramPackage>(package))
        return test_summary("test_program_product_emit");
    const auto wire = encode_program_package(
        std::get<VerifiedProgramPackage>(package));
    CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
    if (const auto* bytes = std::get_if<std::vector<uint8_t>>(&wire))
        CHECK(write_bytes(package_path, *bytes));
    const PackageView* weights = nullptr;
    const PackageView* token = nullptr;
    for (const PackageView& artifact : source.index.artifacts()) {
        if (artifact.artifact_id() == ArtifactId{7}) weights = &artifact;
        if (artifact.artifact_id() == ArtifactId{2}) token = &artifact;
    }
    CHECK(weights != nullptr);
    CHECK(token != nullptr);
    if (weights != nullptr) CHECK(write_bytes(weights_path, weights->bytes()));
    if (token != nullptr) CHECK(write_bytes(token_path, token->bytes()));
    return test_summary("test_program_product_emit");
}

int load_external_container(const char* schema_path, const char* container_path) {
    const auto schema_wire = read_bytes(schema_path);
    const auto container = read_bytes(container_path);
    const auto decoded_schemas = decode_container_schema_set(schema_wire);
    CHECK(std::holds_alternative<std::vector<ContainerSchemaProgram>>(
        decoded_schemas));
    if (!std::holds_alternative<std::vector<ContainerSchemaProgram>>(
            decoded_schemas))
        return test_summary("test_unseen_container_schema");
    Fixture destination = fixture();
    const auto package = decode_container_program_package(
        std::move(destination.index),
        std::get<std::vector<ContainerSchemaProgram>>(decoded_schemas),
        container, 23);
    CHECK(std::holds_alternative<VerifiedProgramPackage>(package));
    if (const auto* verified = std::get_if<VerifiedProgramPackage>(&package))
        CHECK(verified->complete());
    return test_summary("test_unseen_container_schema");
}

int load_external_product_container(const char* schema_path,
                                    const char* container_path) {
    const auto schema_wire = read_bytes(schema_path);
    const auto decoded_schemas = decode_container_schema_set(schema_wire);
    CHECK(std::holds_alternative<std::vector<ContainerSchemaProgram>>(
        decoded_schemas));
    auto loaded = ArtifactSet::load_single_file(container_path);
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    if (!std::holds_alternative<std::vector<ContainerSchemaProgram>>(
            decoded_schemas) ||
        !std::holds_alternative<ArtifactSet>(loaded))
        return test_summary("test_unseen_container_product");
    auto container = std::get<ArtifactSet>(std::move(loaded)).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(container));
    if (!std::holds_alternative<PackageView>(container))
        return test_summary("test_unseen_container_product");
    const std::array<ContainerArtifactSection, 2> sections = {{
        {41, ArtifactId{7}, ArtifactRole::Primary},
        {42, ArtifactId{2}, ArtifactRole::Shard},
    }};
    const auto package = load_container_program_package(
        std::get<PackageView>(container),
        std::get<std::vector<ContainerSchemaProgram>>(decoded_schemas),
        sections, 23);
    CHECK(std::holds_alternative<VerifiedProgramPackage>(package));
    if (const auto* verified = std::get_if<VerifiedProgramPackage>(&package))
        CHECK(verified->complete());
    return test_summary("test_unseen_container_product");
}

void test_complete_package_roundtrip_and_identity() {
    auto package = build(fixture());
    CHECK(std::holds_alternative<VerifiedProgramPackage>(package));
    if (!std::holds_alternative<VerifiedProgramPackage>(package)) return;
    const auto& verified = std::get<VerifiedProgramPackage>(package);
    CHECK(verified.complete());
    CHECK(verified.digest() != Sha256Digest{});
    CHECK(verified.token_program().definition().vocabulary.size() == 10);

    const auto wire = encode_program_package(verified);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
    if (!std::holds_alternative<std::vector<uint8_t>>(wire)) return;
    Fixture destination = fixture();
    const auto decoded = decode_program_package(
        std::move(destination.index),
        std::get<std::vector<uint8_t>>(wire));
    CHECK(std::holds_alternative<VerifiedProgramPackage>(decoded));
    if (const auto* roundtrip = std::get_if<VerifiedProgramPackage>(&decoded))
        CHECK(roundtrip->digest() == verified.digest());

    auto trailing = std::get<std::vector<uint8_t>>(wire);
    trailing.push_back(0);
    Fixture trailing_source = fixture();
    CHECK(std::holds_alternative<CompatibilityReport>(
        decode_program_package(std::move(trailing_source.index), trailing)));
    auto bad_magic = std::get<std::vector<uint8_t>>(wire);
    bad_magic.front() ^= 1;
    Fixture magic_source = fixture();
    CHECK(std::holds_alternative<CompatibilityReport>(
        decode_program_package(std::move(magic_source.index), bad_magic)));
    auto bad_digest = std::get<std::vector<uint8_t>>(wire);
    bad_digest.back() ^= 1;
    Fixture digest_source = fixture();
    CHECK(std::holds_alternative<CompatibilityReport>(
        decode_program_package(std::move(digest_source.index), bad_digest)));
}

void test_token_binding_and_provenance_counterexamples() {
    Fixture missing = fixture();
    missing.token_bindings.pop_back();
    CHECK(std::holds_alternative<CompatibilityReport>(build(std::move(missing))));

    Fixture duplicate = fixture();
    duplicate.token_bindings[1] = duplicate.token_bindings[0];
    CHECK(std::holds_alternative<CompatibilityReport>(build(std::move(duplicate))));

    Fixture stale = fixture();
    stale.token_source.digest.bytes[0] ^= 1;
    CHECK(std::holds_alternative<CompatibilityReport>(build(std::move(stale))));

    Fixture reordered = fixture();
    std::reverse(reordered.token_bindings.begin(), reordered.token_bindings.end());
    auto reordered_package = build(std::move(reordered));
    auto baseline_package = build(fixture());
    CHECK(std::holds_alternative<VerifiedProgramPackage>(reordered_package));
    CHECK(std::holds_alternative<VerifiedProgramPackage>(baseline_package));
    if (std::holds_alternative<VerifiedProgramPackage>(reordered_package) &&
        std::holds_alternative<VerifiedProgramPackage>(baseline_package)) {
        CHECK(std::get<VerifiedProgramPackage>(reordered_package).digest() ==
              std::get<VerifiedProgramPackage>(baseline_package).digest());
    }

    auto renamed_package = build(fixture(70, 20));
    CHECK(std::holds_alternative<VerifiedProgramPackage>(renamed_package));
    if (std::holds_alternative<VerifiedProgramPackage>(renamed_package) &&
        std::holds_alternative<VerifiedProgramPackage>(baseline_package)) {
        CHECK(std::get<VerifiedProgramPackage>(renamed_package).digest() ==
              std::get<VerifiedProgramPackage>(baseline_package).digest());
    }

    auto semantic_renamed = build(renamed_fixture());
    CHECK(std::holds_alternative<VerifiedProgramPackage>(semantic_renamed));
    if (std::holds_alternative<VerifiedProgramPackage>(semantic_renamed) &&
        std::holds_alternative<VerifiedProgramPackage>(baseline_package)) {
        CHECK(std::get<VerifiedProgramPackage>(semantic_renamed).digest() ==
              std::get<VerifiedProgramPackage>(baseline_package).digest());
    }

    Fixture changed_state = fixture();
    auto other_state = verify_state_schema(state_schema(3), changed_state.program);
    CHECK(std::holds_alternative<VerifiedStateSchema>(other_state));
    if (std::holds_alternative<VerifiedStateSchema>(other_state)) {
        changed_state.state =
            std::get<VerifiedStateSchema>(std::move(other_state));
        auto changed_package = build(std::move(changed_state));
        CHECK(std::holds_alternative<VerifiedProgramPackage>(changed_package));
        if (std::holds_alternative<VerifiedProgramPackage>(changed_package) &&
            std::holds_alternative<VerifiedProgramPackage>(baseline_package)) {
            CHECK(std::get<VerifiedProgramPackage>(changed_package).digest() !=
                  std::get<VerifiedProgramPackage>(baseline_package).digest());
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--emit-program-package")
        return emit_external_package(argv[2]);
    if (argc == 5 && std::string(argv[1]) == "--emit-program-product")
        return emit_external_product_source(argv[2], argv[3], argv[4]);
    if (argc == 4 && std::string(argv[1]) == "--load-container-schema")
        return load_external_container(argv[2], argv[3]);
    if (argc == 4 && std::string(argv[1]) == "--load-container-product-schema")
        return load_external_product_container(argv[2], argv[3]);
    test_complete_package_roundtrip_and_identity();
    test_token_binding_and_provenance_counterexamples();
    test_container_handoff_uses_verified_package_decoder();
    test_container_loader_builds_artifact_index();
    return test_summary("test_program_package");
}
