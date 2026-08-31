#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <CommonCrypto/CommonDigest.h>

#include "program_ir.h"
#include "program_state.h"
#include "test_util.h"

using namespace Laplace;

static_assert(!std::is_default_constructible_v<ExecutionResult>);
static_assert(!std::is_copy_constructible_v<ExecutionResult>);
static_assert(!std::is_copy_assignable_v<ExecutionResult>);
static_assert(std::is_move_constructible_v<ExecutionResult>);
static_assert(!std::is_constructible_v<ProgramStateExecutionAuthority>);

namespace {

DimensionExpr dimension(uint64_t value) {
    return {DimensionExpression::Constant, value, {}};
}

ValueType scalar(ElementType type = ElementType::F32) {
    return {type, {}};
}

ValueType tensor(std::initializer_list<uint64_t> dimensions,
                 ElementType type = ElementType::F32) {
    ValueType result;
    result.element_type = type;
    for (uint64_t value : dimensions) result.dimensions.push_back(dimension(value));
    return result;
}

TypedValue value(uint32_t id, ValueType type) {
    return {id, std::move(type)};
}

Instruction state_access(uint32_t id, Primitive primitive, uint32_t state_id,
                         ValueType type, uint32_t value_id,
                         std::vector<uint32_t> predecessors = {}) {
    Instruction item;
    item.id = id;
    item.primitive = {primitive, 1, 0};
    item.effect_predecessors = std::move(predecessors);
    item.attributes = StateAttributes{state_id};
    if (primitive == Primitive::StateRead) {
        item.outputs = {value(value_id, std::move(type))};
    } else {
        item.inputs = {value_id};
    }
    return item;
}

Program two_slot_program(bool renamed = false) {
    const uint32_t scalar_state = renamed ? 700 : 70;
    const uint32_t tensor_state = renamed ? 800 : 80;
    const uint32_t read_scalar = renamed ? 1010 : 10;
    const uint32_t write_scalar = renamed ? 1020 : 20;
    const uint32_t read_tensor = renamed ? 1030 : 30;
    const uint32_t write_tensor = renamed ? 1040 : 40;
    const uint32_t scalar_value = renamed ? 1100 : 100;
    const uint32_t tensor_value = renamed ? 1200 : 200;

    Region region;
    region.id = renamed ? 900 : 90;
    region.instructions.push_back(state_access(
        read_scalar, Primitive::StateRead, scalar_state, scalar(), scalar_value));
    region.instructions.push_back(state_access(
        write_scalar, Primitive::StateWrite, scalar_state, scalar(), scalar_value,
        {read_scalar}));
    region.instructions.push_back(state_access(
        read_tensor, Primitive::StateRead, tensor_state, tensor({2}), tensor_value,
        {write_scalar}));
    region.instructions.push_back(state_access(
        write_tensor, Primitive::StateWrite, tensor_state, tensor({2}), tensor_value,
        {read_tensor}));

    Function function;
    function.id = renamed ? 600 : 60;
    function.entry_region_id = region.id;
    function.regions.push_back(std::move(region));

    Program program;
    program.state_references.push_back({scalar_state, scalar(), 11, true});
    program.state_references.push_back({tensor_state, tensor({2}), 12, true});
    if (renamed) std::swap(program.state_references[0], program.state_references[1]);
    program.functions.push_back(std::move(function));
    return program;
}

Program changed_program() {
    Program program = two_slot_program();
    Region region;
    region.id = 500;
    Instruction constant;
    constant.id = 501;
    constant.primitive = {Primitive::Constant, 1, 0};
    constant.outputs = {value(502, scalar())};
    constant.attributes = ConstantAttributes{7};
    region.instructions.push_back(std::move(constant));
    region.yields = {502};
    Function function;
    function.id = 503;
    function.entry_region_id = region.id;
    function.regions.push_back(std::move(region));
    function.result_types = {scalar()};
    program.functions.push_back(std::move(function));
    program.exports.push_back({503, 0, scalar()});
    return program;
}

Program stateless_program() {
    Region region;
    region.id = 1;
    Instruction constant;
    constant.id = 2;
    constant.primitive = {Primitive::Constant, 1, 0};
    constant.outputs = {value(3, scalar(ElementType::U32))};
    constant.attributes = ConstantAttributes{5};
    region.instructions.push_back(std::move(constant));
    region.yields = {3};
    Function function;
    function.id = 4;
    function.entry_region_id = region.id;
    function.regions.push_back(std::move(region));
    function.result_types = {scalar(ElementType::U32)};
    Program program;
    program.functions.push_back(std::move(function));
    program.exports.push_back({4, 0, scalar(ElementType::U32)});
    return program;
}

Program single_state_program(ElementType type, bool writable) {
    Region region;
    region.id = 10;
    region.instructions.push_back(
        state_access(20, Primitive::StateRead, 70, scalar(type), 30));
    Function function;
    function.id = 40;
    function.entry_region_id = region.id;
    if (writable) {
        region.instructions.push_back(state_access(
            50, Primitive::StateWrite, 70, scalar(type), 30, {20}));
    } else {
        region.yields = {30};
        function.result_types = {scalar(type)};
    }
    function.regions.push_back(std::move(region));
    Program program;
    program.state_references.push_back(
        {70, scalar(type), UINT32_MAX, writable});
    program.functions.push_back(std::move(function));
    if (!writable) program.exports.push_back({40, 0, scalar(type)});
    return program;
}

Program aliased_program() {
    Region region;
    region.id = 10;
    region.instructions.push_back(
        state_access(20, Primitive::StateRead, 70, scalar(), 30));
    region.instructions.push_back(state_access(
        40, Primitive::StateWrite, 70, scalar(), 30, {20}));
    region.instructions.push_back(
        state_access(50, Primitive::StateRead, 80, scalar(), 60, {40}));
    region.instructions.push_back(state_access(
        70, Primitive::StateWrite, 80, scalar(), 60, {50}));
    Function function;
    function.id = 90;
    function.entry_region_id = region.id;
    function.regions.push_back(std::move(region));
    Program program;
    program.state_references.push_back({70, scalar(), 11, true});
    program.state_references.push_back({80, scalar(), 11, true});
    program.functions.push_back(std::move(function));
    return program;
}

Program parameter_state_program(bool renamed = false, uint64_t lower = 2) {
    const uint32_t parameter_id = renamed ? 55 : 5;
    const uint32_t state_id = renamed ? 700 : 70;
    ValueType type;
    type.element_type = ElementType::F16;
    type.dimensions.push_back(
        {DimensionExpression::Parameter, parameter_id, {}});
    Region region;
    region.id = renamed ? 100 : 10;
    region.instructions.push_back(
        state_access(renamed ? 200 : 20, Primitive::StateRead, state_id, type,
                     renamed ? 300 : 30));
    region.instructions.push_back(state_access(
        renamed ? 400 : 40, Primitive::StateWrite, state_id, type,
        renamed ? 300 : 30, {renamed ? 200u : 20u}));
    Function function;
    function.id = renamed ? 500 : 50;
    function.entry_region_id = region.id;
    function.regions.push_back(std::move(region));
    Program program;
    program.dimension_parameters.push_back({parameter_id, lower, 4});
    program.state_references.push_back({state_id, type, UINT32_MAX, true});
    program.functions.push_back(std::move(function));
    return program;
}

Program zero_extent_state_program() {
    ValueType type = tensor({0});
    Region region;
    region.id = 10;
    region.instructions.push_back(
        state_access(20, Primitive::StateRead, 70, type, 30));
    region.instructions.push_back(state_access(
        40, Primitive::StateWrite, 70, type, 30, {20}));
    Function function;
    function.id = 50;
    function.entry_region_id = region.id;
    function.regions.push_back(std::move(region));
    Program program;
    program.state_references.push_back({70, type, UINT32_MAX, true});
    program.functions.push_back(std::move(function));
    return program;
}

VerifiedProgram verified_program(Program program) {
    auto result = verify_and_canonicalize_program(std::move(program));
    CHECK(std::holds_alternative<VerifiedProgram>(result));
    if (!std::holds_alternative<VerifiedProgram>(result)) std::abort();
    return std::get<VerifiedProgram>(std::move(result));
}

StateSchema two_slot_schema(bool renamed = false) {
    StateSchema schema;
    if (renamed) {
        schema.slots.push_back({800, ElementType::F32, {2}, 902});
        schema.slots.push_back({700, ElementType::F32, {}, 901});
    } else {
        schema.slots.push_back({70, ElementType::F32, {}, 101});
        schema.slots.push_back({80, ElementType::F32, {2}, 202});
    }
    return schema;
}

VerifiedStateSchema verified_schema(StateSchema schema,
                                    const VerifiedProgram& program) {
    auto result = verify_state_schema(std::move(schema), program);
    CHECK(std::holds_alternative<VerifiedStateSchema>(result));
    if (!std::holds_alternative<VerifiedStateSchema>(result)) std::abort();
    return std::get<VerifiedStateSchema>(std::move(result));
}

std::vector<uint8_t> bytes32(uint32_t value) {
    std::vector<uint8_t> bytes(sizeof(value));
    std::memcpy(bytes.data(), &value, sizeof(value));
    return bytes;
}

std::vector<uint8_t> bytes64(uint64_t value) {
    std::vector<uint8_t> bytes(sizeof(value));
    std::memcpy(bytes.data(), &value, sizeof(value));
    return bytes;
}

uint32_t read32(std::span<const uint8_t> bytes) {
    uint32_t value = 0;
    if (bytes.size() == sizeof(value)) std::memcpy(&value, bytes.data(), sizeof(value));
    return value;
}

uint64_t read64(std::span<const uint8_t> bytes) {
    uint64_t value = 0;
    if (bytes.size() == sizeof(value)) std::memcpy(&value, bytes.data(), sizeof(value));
    return value;
}

uint64_t wire_u64(const std::vector<uint8_t>& bytes, size_t offset) {
    uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        value |= static_cast<uint64_t>(bytes[offset + shift / 8]) << shift;
    }
    return value;
}

void wire_put_u64(std::vector<uint8_t>& bytes, size_t offset, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        bytes[offset + shift / 8] = static_cast<uint8_t>(value >> shift);
    }
}

void reseal(std::vector<uint8_t>& bytes) {
    std::array<uint8_t, 32> digest{};
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size() - digest.size()),
              digest.data());
    std::copy(digest.begin(), digest.end(), bytes.end() - digest.size());
}

StateRoot root_with(const VerifiedStateSchema& schema, uint32_t scalar_value,
                    uint64_t tensor_value, bool renamed = false) {
    std::vector<StateCellValue> cells;
    cells.push_back({renamed ? 800u : 80u, bytes64(tensor_value)});
    cells.push_back({renamed ? 700u : 70u, bytes32(scalar_value)});
    auto result = make_state_root(schema, std::move(cells));
    CHECK(std::holds_alternative<StateRoot>(result));
    if (!std::holds_alternative<StateRoot>(result)) std::abort();
    return std::get<StateRoot>(std::move(result));
}

ProgramStateMutationResult resolve_for_testing(
    StateRoot& root, CandidateGeneration&& candidate,
    ExecutionDisposition disposition) {
    ExecutionResult result =
        program_state_execution_result_for_testing(candidate, disposition);
    return root.resolve_candidate(std::move(candidate), std::move(result));
}

void test_two_unrelated_cells_publish_atomically() {
    const VerifiedProgram program = verified_program(two_slot_program());
    const VerifiedStateSchema schema = verified_schema(two_slot_schema(), program);
    StateRoot root = root_with(schema, 3, 5);
    CHECK(root.valid() && !root.poisoned() && root.generation() == 0);

    auto opened = root.begin_candidate();
    CHECK(std::holds_alternative<CandidateGeneration>(opened));
    if (!std::holds_alternative<CandidateGeneration>(opened)) return;
    CandidateGeneration candidate =
        std::get<CandidateGeneration>(std::move(opened));
    CHECK(read32(candidate.read_slot(70)) == 3);
    CHECK(read64(candidate.read_slot(80)) == 5);
    CHECK(std::holds_alternative<std::monostate>(
        candidate.write_slot(70, bytes32(7))));
    CHECK(candidate.read_slot(70).data() != root.current_slot(70).data());
    CHECK(std::holds_alternative<std::monostate>(
        candidate.write_slot(80, bytes64(11))));
    CHECK(read32(root.current_slot(70)) == 3);
    CHECK(read64(root.current_slot(80)) == 5);
    CHECK(std::holds_alternative<std::monostate>(
        resolve_for_testing(root, std::move(candidate),
                            ExecutionDisposition::Completed)));
    CHECK(root.generation() == 1 && !root.candidate_active());
    CHECK(read32(root.current_slot(70)) == 7);
    CHECK(read64(root.current_slot(80)) == 11);
}

void test_read_views_own_payload_lifetime() {
    const VerifiedProgram program = verified_program(two_slot_program());
    const VerifiedStateSchema schema = verified_schema(two_slot_schema(), program);
    StateRoot root = root_with(schema, 3, 5);
    const StateReadView held_root = root.current_slot(70);
    CHECK(held_root.valid() && read32(held_root) == 3);

    auto opened = root.begin_candidate();
    CHECK(std::holds_alternative<CandidateGeneration>(opened));
    if (!std::holds_alternative<CandidateGeneration>(opened)) return;
    CandidateGeneration candidate =
        std::get<CandidateGeneration>(std::move(opened));
    CHECK(std::holds_alternative<std::monostate>(
        candidate.write_slot(70, bytes32(7))));
    const StateReadView held_candidate = candidate.read_slot(70);
    CHECK(held_candidate.valid() && read32(held_candidate) == 7);
    CHECK(std::holds_alternative<std::monostate>(
        candidate.write_slot(70, bytes32(9))));
    CHECK(read32(held_candidate) == 7);
    CHECK(read32(candidate.read_slot(70)) == 9);
    CHECK(std::holds_alternative<std::monostate>(
        resolve_for_testing(root, std::move(candidate),
                            ExecutionDisposition::Completed)));
    CHECK(read32(held_root) == 3);
    CHECK(read32(root.current_slot(70)) == 9);
}

void test_execution_disposition_matrix_and_abandonment() {
    const VerifiedProgram program = verified_program(two_slot_program());
    const VerifiedStateSchema schema = verified_schema(two_slot_schema(), program);

    {
        StateRoot root = root_with(schema, 11, 13);
        auto opened = root.begin_candidate();
        CHECK(std::holds_alternative<CandidateGeneration>(opened));
        if (!std::holds_alternative<CandidateGeneration>(opened)) return;
        CandidateGeneration candidate =
            std::get<CandidateGeneration>(std::move(opened));
        CHECK(std::holds_alternative<std::monostate>(
            candidate.write_slot(70, bytes32(17))));
        CHECK(std::holds_alternative<std::monostate>(resolve_for_testing(
            root, std::move(candidate),
            ExecutionDisposition::RejectedBeforeSubmission)));
        CHECK(!root.poisoned() && root.generation() == 0);
        CHECK(read32(root.current_slot(70)) == 11);
    }
    {
        StateRoot root = root_with(schema, 19, 23);
        auto opened = root.begin_candidate();
        CHECK(std::holds_alternative<CandidateGeneration>(opened));
        if (!std::holds_alternative<CandidateGeneration>(opened)) return;
        CandidateGeneration candidate =
            std::get<CandidateGeneration>(std::move(opened));
        CHECK(std::holds_alternative<std::monostate>(
            candidate.write_slot(70, bytes32(29))));
        CHECK(std::holds_alternative<std::monostate>(resolve_for_testing(
            root, std::move(candidate), ExecutionDisposition::Completed)));
        CHECK(!root.poisoned() && root.generation() == 1);
        CHECK(read32(root.current_slot(70)) == 29);
    }
    for (ExecutionDisposition disposition :
         {ExecutionDisposition::Failed, ExecutionDisposition::Indeterminate}) {
        StateRoot root = root_with(schema, 31, 37);
        auto opened = root.begin_candidate();
        CHECK(std::holds_alternative<CandidateGeneration>(opened));
        if (!std::holds_alternative<CandidateGeneration>(opened)) return;
        CandidateGeneration candidate =
            std::get<CandidateGeneration>(std::move(opened));
        CHECK(std::holds_alternative<std::monostate>(resolve_for_testing(
            root, std::move(candidate), disposition)));
        CHECK(root.poisoned() && !root.candidate_active());
        CHECK(program_state_has_retained_candidate_for_testing(root));
    }
    {
        StateRoot root = root_with(schema, 41, 43);
        {
            auto opened = root.begin_candidate();
            CHECK(std::holds_alternative<CandidateGeneration>(opened));
            if (!std::holds_alternative<CandidateGeneration>(opened)) return;
            CandidateGeneration candidate =
                std::get<CandidateGeneration>(std::move(opened));
            CHECK(std::holds_alternative<std::monostate>(
                candidate.write_slot(70, bytes32(47))));
        }
        CHECK(root.poisoned() && !root.candidate_active());
        CHECK(program_state_has_retained_candidate_for_testing(root));
        CHECK(!root.current_slot(70).valid());
        CHECK(std::holds_alternative<CompatibilityReport>(root.begin_candidate()));
        CHECK(std::holds_alternative<CompatibilityReport>(root.save()));
    }
}

void test_receipt_binds_effect_boundary() {
    const VerifiedProgram program = verified_program(two_slot_program());
    const VerifiedStateSchema schema = verified_schema(two_slot_schema(), program);
    StateRoot root = root_with(schema, 53, 59);
    auto opened = root.begin_candidate();
    CHECK(std::holds_alternative<CandidateGeneration>(opened));
    if (!std::holds_alternative<CandidateGeneration>(opened)) return;
    CandidateGeneration candidate =
        std::get<CandidateGeneration>(std::move(opened));
    ExecutionResult stale = program_state_execution_result_for_testing(
        candidate, ExecutionDisposition::Completed);
    CHECK(std::holds_alternative<std::monostate>(
        candidate.write_slot(70, bytes32(61))));
    CHECK(std::holds_alternative<CompatibilityReport>(
        root.resolve_candidate(std::move(candidate), std::move(stale))));
    CHECK(candidate.valid() && stale.valid());
    ExecutionResult invalid = program_state_execution_result_for_testing(
        candidate, static_cast<ExecutionDisposition>(255));
    CHECK(std::holds_alternative<CompatibilityReport>(
        root.resolve_candidate(std::move(candidate), std::move(invalid))));
    CHECK(candidate.valid() && invalid.valid());
    ExecutionResult current = program_state_execution_result_for_testing(
        candidate, ExecutionDisposition::Completed);
    CHECK(std::holds_alternative<std::monostate>(
        root.resolve_candidate(std::move(candidate), std::move(current))));
    CHECK(read32(root.current_slot(70)) == 61);
}

void test_schema_shape_alias_and_identity_reject() {
    const VerifiedProgram program = verified_program(two_slot_program());

    StateSchema wrong_shape = two_slot_schema();
    wrong_shape.slots[1].extents = {3};
    CHECK(std::holds_alternative<CompatibilityReport>(
        verify_state_schema(std::move(wrong_shape), program)));

    StateSchema wrong_type = two_slot_schema();
    wrong_type.slots[0].element_type = ElementType::U32;
    CHECK(std::holds_alternative<CompatibilityReport>(
        verify_state_schema(std::move(wrong_type), program)));

    StateSchema wrong_alias = two_slot_schema();
    wrong_alias.slots[1].alias_cell = wrong_alias.slots[0].alias_cell;
    CHECK(std::holds_alternative<CompatibilityReport>(
        verify_state_schema(std::move(wrong_alias), program)));

    StateSchema repeated = two_slot_schema();
    repeated.slots[1].state_reference_id = 70;
    CHECK(std::holds_alternative<CompatibilityReport>(
        verify_state_schema(std::move(repeated), program)));

    StateSchema missing = two_slot_schema();
    missing.slots.pop_back();
    CHECK(std::holds_alternative<CompatibilityReport>(
        verify_state_schema(std::move(missing), program)));
}

void test_all_element_types_and_immutable_write_reject() {
    const std::vector<std::pair<ElementType, size_t>> types = {
        {ElementType::I1, 1},  {ElementType::I32, 4},
        {ElementType::U32, 4}, {ElementType::U64, 8},
        {ElementType::F16, 2}, {ElementType::F32, 4},
    };
    for (const auto& [type, width] : types) {
        const VerifiedProgram program =
            verified_program(single_state_program(type, false));
        StateSchema schema;
        schema.slots.push_back({70, type, {}, 1});
        const VerifiedStateSchema verified =
            verified_schema(std::move(schema), program);
        CHECK(verified.slots().size() == 1);
        CHECK(verified.slots()[0].byte_count == width);
        CHECK(!verified.slots()[0].writable);
        std::vector<uint8_t> bits(width, 0xa5);
        auto made = make_state_root(verified, {{70, bits}});
        CHECK(std::holds_alternative<StateRoot>(made));
        if (!std::holds_alternative<StateRoot>(made)) continue;
        StateRoot root = std::get<StateRoot>(std::move(made));
        auto opened = root.begin_candidate();
        CHECK(std::holds_alternative<CandidateGeneration>(opened));
        if (!std::holds_alternative<CandidateGeneration>(opened)) continue;
        CandidateGeneration candidate =
            std::get<CandidateGeneration>(std::move(opened));
        CHECK(std::holds_alternative<CompatibilityReport>(
            candidate.write_slot(70, bits)));
        CHECK(std::holds_alternative<std::monostate>(
            resolve_for_testing(root, std::move(candidate),
                                ExecutionDisposition::RejectedBeforeSubmission)));
        CHECK(std::vector<uint8_t>(root.current_slot(70).begin(),
                                   root.current_slot(70).end()) == bits);
    }
}

void test_stateless_root_roundtrip() {
    const VerifiedProgram program = verified_program(stateless_program());
    const VerifiedStateSchema schema = verified_schema({}, program);
    CHECK(schema.slots().empty() && schema.cells().empty());
    auto made = make_state_root(schema, {});
    CHECK(std::holds_alternative<StateRoot>(made));
    if (!std::holds_alternative<StateRoot>(made)) return;
    StateRoot root = std::get<StateRoot>(std::move(made));
    auto opened = root.begin_candidate();
    CHECK(std::holds_alternative<CandidateGeneration>(opened));
    if (!std::holds_alternative<CandidateGeneration>(opened)) return;
    CandidateGeneration candidate =
        std::get<CandidateGeneration>(std::move(opened));
    CHECK(std::holds_alternative<std::monostate>(
        resolve_for_testing(root, std::move(candidate),
                            ExecutionDisposition::Completed)));
    auto saved = root.save();
    CHECK(std::holds_alternative<std::vector<uint8_t>>(saved));
    if (!std::holds_alternative<std::vector<uint8_t>>(saved)) return;
    auto restored = restore_state_root(
        schema, std::get<std::vector<uint8_t>>(saved));
    CHECK(std::holds_alternative<StateRoot>(restored));
    if (std::holds_alternative<StateRoot>(restored)) {
        CHECK(std::get<StateRoot>(restored).generation() == 1);
    }
}

void test_zero_extent_state_roundtrip() {
    const VerifiedProgram program = verified_program(zero_extent_state_program());
    StateSchema source;
    source.slots.push_back({70, ElementType::F32, {0}, 1});
    const VerifiedStateSchema schema = verified_schema(std::move(source), program);
    CHECK(schema.slots().size() == 1 && schema.cells().size() == 1);
    CHECK(schema.slots()[0].byte_count == 0);

    auto made = make_state_root(schema, {{70, {}}});
    CHECK(std::holds_alternative<StateRoot>(made));
    if (!std::holds_alternative<StateRoot>(made)) return;
    StateRoot root = std::get<StateRoot>(std::move(made));
    const StateReadView initial = root.current_slot(70);
    CHECK(initial.valid() && initial.empty());
    auto opened = root.begin_candidate();
    CHECK(std::holds_alternative<CandidateGeneration>(opened));
    if (!std::holds_alternative<CandidateGeneration>(opened)) return;
    CandidateGeneration candidate =
        std::get<CandidateGeneration>(std::move(opened));
    CHECK(candidate.read_slot(70).valid());
    CHECK(std::holds_alternative<std::monostate>(
        candidate.write_slot(70, {})));
    CHECK(std::holds_alternative<std::monostate>(resolve_for_testing(
        root, std::move(candidate), ExecutionDisposition::Completed)));
    CHECK(root.current_slot(70).valid() && root.current_slot(70).empty());

    auto saved = root.save();
    CHECK(std::holds_alternative<std::vector<uint8_t>>(saved));
    if (!std::holds_alternative<std::vector<uint8_t>>(saved)) return;
    auto restored = restore_state_root(
        schema, std::get<std::vector<uint8_t>>(saved));
    CHECK(std::holds_alternative<StateRoot>(restored));
    if (std::holds_alternative<StateRoot>(restored)) {
        const StateReadView view =
            std::get<StateRoot>(restored).current_slot(70);
        CHECK(view.valid() && view.empty());
    }

    const VerifiedProgram parameter_program =
        verified_program(parameter_state_program(false, 0));
    StateSchema parameter_source;
    parameter_source.dimension_bindings.push_back({5, 0});
    parameter_source.slots.push_back({70, ElementType::F16, {0}, 1});
    auto parameter_schema =
        verify_state_schema(std::move(parameter_source), parameter_program);
    CHECK(std::holds_alternative<VerifiedStateSchema>(parameter_schema));
    if (std::holds_alternative<VerifiedStateSchema>(parameter_schema)) {
        CHECK(std::get<VerifiedStateSchema>(parameter_schema)
                  .slots()[0]
                  .byte_count == 0);
    }
}

void test_alias_cells_share_one_candidate_payload() {
    const VerifiedProgram program = verified_program(aliased_program());
    StateSchema schema;
    schema.slots.push_back({80, ElementType::F32, {}, 500});
    schema.slots.push_back({70, ElementType::F32, {}, 500});
    const VerifiedStateSchema verified =
        verified_schema(std::move(schema), program);
    CHECK(verified.slots().size() == 2);
    CHECK(verified.cells().size() == 1);
    auto made = make_state_root(verified, {{70, bytes32(3)}});
    CHECK(std::holds_alternative<StateRoot>(made));
    if (!std::holds_alternative<StateRoot>(made)) return;
    StateRoot root = std::get<StateRoot>(std::move(made));
    auto opened = root.begin_candidate();
    CHECK(std::holds_alternative<CandidateGeneration>(opened));
    if (!std::holds_alternative<CandidateGeneration>(opened)) return;
    CandidateGeneration candidate =
        std::get<CandidateGeneration>(std::move(opened));
    CHECK(std::holds_alternative<std::monostate>(
        candidate.write_slot(80, bytes32(9))));
    CHECK(read32(candidate.read_slot(70)) == 9);
    CHECK(std::holds_alternative<std::monostate>(
        resolve_for_testing(root, std::move(candidate),
                            ExecutionDisposition::Completed)));
    CHECK(read32(root.current_slot(70)) == 9);
    CHECK(read32(root.current_slot(80)) == 9);

    StateSchema split;
    split.slots.push_back({70, ElementType::F32, {}, 1});
    split.slots.push_back({80, ElementType::F32, {}, 2});
    CHECK(std::holds_alternative<CompatibilityReport>(
        verify_state_schema(std::move(split), program)));
}

void test_symbolic_shape_binding_is_exact() {
    const VerifiedProgram program = verified_program(parameter_state_program());
    StateSchema schema;
    schema.dimension_bindings.push_back({5, 3});
    schema.slots.push_back({70, ElementType::F16, {3}, 1});
    const VerifiedStateSchema verified =
        verified_schema(schema, program);
    CHECK(verified.slots()[0].byte_count == 6);

    StateSchema bad_extent = schema;
    bad_extent.slots[0].extents = {4};
    CHECK(std::holds_alternative<CompatibilityReport>(
        verify_state_schema(std::move(bad_extent), program)));
    StateSchema bad_binding = schema;
    bad_binding.dimension_bindings[0].value = 5;
    CHECK(std::holds_alternative<CompatibilityReport>(
        verify_state_schema(std::move(bad_binding), program)));
    StateSchema missing_binding = schema;
    missing_binding.dimension_bindings.clear();
    CHECK(std::holds_alternative<CompatibilityReport>(
        verify_state_schema(std::move(missing_binding), program)));

    const VerifiedProgram renamed_program =
        verified_program(parameter_state_program(true));
    StateSchema renamed_schema;
    renamed_schema.dimension_bindings.push_back({55, 3});
    renamed_schema.slots.push_back({700, ElementType::F16, {3}, 99});
    const VerifiedStateSchema renamed =
        verified_schema(std::move(renamed_schema), renamed_program);
    CHECK(program_digest(program) == program_digest(renamed_program));
    CHECK(state_schema_digest(verified) == state_schema_digest(renamed));
}

void test_discard_is_byte_exact_after_each_effect_boundary() {
    const VerifiedProgram program = verified_program(two_slot_program());
    const VerifiedStateSchema schema = verified_schema(two_slot_schema(), program);
    StateRoot root = root_with(schema, 13, 17);
    const auto before = root.save();
    CHECK(std::holds_alternative<std::vector<uint8_t>>(before));
    if (!std::holds_alternative<std::vector<uint8_t>>(before)) return;

    for (uint32_t boundary = 1; boundary <= 2; ++boundary) {
        auto opened = root.begin_candidate();
        CHECK(std::holds_alternative<CandidateGeneration>(opened));
        if (!std::holds_alternative<CandidateGeneration>(opened)) return;
        CandidateGeneration candidate =
            std::get<CandidateGeneration>(std::move(opened));
        CHECK(std::holds_alternative<std::monostate>(
            candidate.write_slot(70, bytes32(19))));
        if (boundary == 2) {
            CHECK(std::holds_alternative<std::monostate>(
                candidate.write_slot(80, bytes64(23))));
        }
        CHECK(candidate.effect_boundaries() == boundary);
        CHECK(std::holds_alternative<std::monostate>(
            resolve_for_testing(root, std::move(candidate),
                                ExecutionDisposition::RejectedBeforeSubmission)));
        const auto after = root.save();
        CHECK(std::holds_alternative<std::vector<uint8_t>>(after));
        if (std::holds_alternative<std::vector<uint8_t>>(after)) {
            CHECK(std::get<std::vector<uint8_t>>(after) ==
                  std::get<std::vector<uint8_t>>(before));
        }
    }
}

void test_execution_failure_poisons_root() {
    const VerifiedProgram program = verified_program(two_slot_program());
    const VerifiedStateSchema schema = verified_schema(two_slot_schema(), program);
    StateRoot root = root_with(schema, 29, 31);
    auto opened = root.begin_candidate();
    CHECK(std::holds_alternative<CandidateGeneration>(opened));
    if (!std::holds_alternative<CandidateGeneration>(opened)) return;
    CandidateGeneration candidate =
        std::get<CandidateGeneration>(std::move(opened));
    CHECK(std::holds_alternative<std::monostate>(
        candidate.write_slot(70, bytes32(37))));
    CHECK(std::holds_alternative<std::monostate>(
        resolve_for_testing(root, std::move(candidate),
                            ExecutionDisposition::Failed)));
    CHECK(root.poisoned() && !root.candidate_active());
    CHECK(root.current_slot(70).empty());
    CHECK(std::holds_alternative<CompatibilityReport>(root.begin_candidate()));
    CHECK(std::holds_alternative<CompatibilityReport>(root.save()));
}

void test_save_restore_binds_program_and_every_cell() {
    const VerifiedProgram program = verified_program(two_slot_program());
    const VerifiedStateSchema schema = verified_schema(two_slot_schema(), program);
    StateRoot root = root_with(schema, 41, 43);
    auto opened = root.begin_candidate();
    CHECK(std::holds_alternative<CandidateGeneration>(opened));
    if (!std::holds_alternative<CandidateGeneration>(opened)) return;
    CandidateGeneration candidate =
        std::get<CandidateGeneration>(std::move(opened));
    CHECK(std::holds_alternative<std::monostate>(
        candidate.write_slot(70, bytes32(0x80000000u))));
    CHECK(std::holds_alternative<std::monostate>(
        resolve_for_testing(root, std::move(candidate),
                            ExecutionDisposition::Completed)));

    auto saved = root.save();
    CHECK(std::holds_alternative<std::vector<uint8_t>>(saved));
    if (!std::holds_alternative<std::vector<uint8_t>>(saved)) return;
    auto restored = restore_state_root(
        schema, std::get<std::vector<uint8_t>>(saved));
    CHECK(std::holds_alternative<StateRoot>(restored));
    if (std::holds_alternative<StateRoot>(restored)) {
        const StateRoot& result = std::get<StateRoot>(restored);
        CHECK(result.generation() == 1);
        CHECK(read32(result.current_slot(70)) == 0x80000000u);
        CHECK(read64(result.current_slot(80)) == 43);
    }

    const VerifiedProgram other_program = verified_program(changed_program());
    const VerifiedStateSchema other_schema =
        verified_schema(two_slot_schema(), other_program);
    CHECK(state_program_digest(schema) != state_program_digest(other_schema));
    CHECK(std::holds_alternative<CompatibilityReport>(restore_state_root(
        other_schema, std::get<std::vector<uint8_t>>(saved))));

    std::vector<uint8_t> corrupt = std::get<std::vector<uint8_t>>(saved);
    corrupt[48] ^= 1;
    CHECK(std::holds_alternative<CompatibilityReport>(
        restore_state_root(schema, corrupt)));
}

void test_wire_regions_reject_without_mutating_live_root() {
    const VerifiedProgram program = verified_program(two_slot_program());
    const VerifiedStateSchema schema = verified_schema(two_slot_schema(), program);
    StateRoot root = root_with(schema, 101, 103);
    auto saved = root.save();
    CHECK(std::holds_alternative<std::vector<uint8_t>>(saved));
    if (!std::holds_alternative<std::vector<uint8_t>>(saved)) return;
    const std::vector<uint8_t> baseline =
        std::get<std::vector<uint8_t>>(saved);
    const size_t cell_table = 128 + 2 * 288;
    const uint64_t first_payload = wire_u64(baseline, cell_table + 16);
    const uint64_t second_payload =
        wire_u64(baseline, cell_table + 64 + 16);
    const std::vector<size_t> offsets = {
        112,
        128 + 8,
        128 + 12,
        cell_table + 4,
        static_cast<size_t>(first_payload),
        static_cast<size_t>(first_payload + 4),
        static_cast<size_t>(second_payload),
    };
    for (size_t offset : offsets) {
        std::vector<uint8_t> corrupt = baseline;
        corrupt[offset] ^= 1;
        reseal(corrupt);
        CHECK_MSG(std::holds_alternative<CompatibilityReport>(
                      restore_state_root(schema, corrupt)),
                  "offset=%zu", offset);
        auto current = root.save();
        CHECK(std::holds_alternative<std::vector<uint8_t>>(current));
        if (std::holds_alternative<std::vector<uint8_t>>(current)) {
            CHECK(std::get<std::vector<uint8_t>>(current) == baseline);
        }
    }
    std::vector<uint8_t> bad_file_digest = baseline;
    bad_file_digest.back() ^= 1;
    CHECK(std::holds_alternative<CompatibilityReport>(
        restore_state_root(schema, bad_file_digest)));

    std::vector<uint8_t> aligned_gap = baseline;
    aligned_gap.insert(
        aligned_gap.begin() + static_cast<ptrdiff_t>(first_payload), 64, 0);
    wire_put_u64(aligned_gap, cell_table + 16, first_payload + 64);
    wire_put_u64(aligned_gap, cell_table + 64 + 16, second_payload + 64);
    wire_put_u64(aligned_gap, 16, aligned_gap.size());
    reseal(aligned_gap);
    CHECK(std::holds_alternative<CompatibilityReport>(
        restore_state_root(schema, aligned_gap)));

    std::vector<uint8_t> trailing = baseline;
    trailing.insert(trailing.end() - 32, 0);
    wire_put_u64(trailing, 16, trailing.size());
    reseal(trailing);
    CHECK(std::holds_alternative<CompatibilityReport>(
        restore_state_root(schema, trailing)));
}

void test_next_generation_reads_published_overlay() {
    const VerifiedProgram program = verified_program(two_slot_program());
    const VerifiedStateSchema schema = verified_schema(two_slot_schema(), program);
    StateRoot root = root_with(schema, 53, 59);

    auto first_open = root.begin_candidate();
    CHECK(std::holds_alternative<CandidateGeneration>(first_open));
    if (!std::holds_alternative<CandidateGeneration>(first_open)) return;
    CandidateGeneration first =
        std::get<CandidateGeneration>(std::move(first_open));
    CHECK(std::holds_alternative<std::monostate>(
        first.write_slot(70, bytes32(61))));
    CHECK(read32(first.read_slot(70)) == 61);
    CHECK(std::holds_alternative<std::monostate>(
        resolve_for_testing(root, std::move(first),
                            ExecutionDisposition::Completed)));

    auto second_open = root.begin_candidate();
    CHECK(std::holds_alternative<CandidateGeneration>(second_open));
    if (!std::holds_alternative<CandidateGeneration>(second_open)) return;
    CandidateGeneration second =
        std::get<CandidateGeneration>(std::move(second_open));
    CHECK(second.base_generation() == 1);
    CHECK(read32(second.read_slot(70)) == 61);
    CHECK(std::holds_alternative<std::monostate>(
        second.write_slot(70, bytes32(67))));
    CHECK(std::holds_alternative<std::monostate>(
        resolve_for_testing(root, std::move(second),
                            ExecutionDisposition::Completed)));
    CHECK(root.generation() == 2);
    CHECK(read32(root.current_slot(70)) == 67);
}

void test_one_active_foreign_and_second_resolution_reject() {
    const VerifiedProgram program = verified_program(two_slot_program());
    const VerifiedStateSchema schema = verified_schema(two_slot_schema(), program);
    StateRoot left = root_with(schema, 71, 73);
    StateRoot right = root_with(schema, 71, 73);
    auto left_open = left.begin_candidate();
    auto right_open = right.begin_candidate();
    CHECK(std::holds_alternative<CandidateGeneration>(left_open));
    CHECK(std::holds_alternative<CandidateGeneration>(right_open));
    CHECK(std::holds_alternative<CompatibilityReport>(left.begin_candidate()));
    CHECK(std::holds_alternative<CompatibilityReport>(left.save()));
    if (!std::holds_alternative<CandidateGeneration>(left_open) ||
        !std::holds_alternative<CandidateGeneration>(right_open)) {
        return;
    }
    CandidateGeneration left_candidate =
        std::get<CandidateGeneration>(std::move(left_open));
    CandidateGeneration right_candidate =
        std::get<CandidateGeneration>(std::move(right_open));
    ExecutionResult right_result = program_state_execution_result_for_testing(
        right_candidate, ExecutionDisposition::RejectedBeforeSubmission);
    CHECK(std::holds_alternative<CompatibilityReport>(
        left.resolve_candidate(std::move(right_candidate),
                               std::move(right_result))));
    CHECK(std::holds_alternative<std::monostate>(
        right.resolve_candidate(std::move(right_candidate),
                                std::move(right_result))));
    ExecutionResult left_result = program_state_execution_result_for_testing(
        left_candidate, ExecutionDisposition::RejectedBeforeSubmission);
    CHECK(std::holds_alternative<std::monostate>(
        left.resolve_candidate(std::move(left_candidate),
                               std::move(left_result))));
    CHECK(std::holds_alternative<CompatibilityReport>(
        left.resolve_candidate(std::move(left_candidate),
                               std::move(left_result))));
}

void test_move_invalidates_source_handles() {
    const VerifiedProgram program = verified_program(two_slot_program());
    const VerifiedStateSchema schema = verified_schema(two_slot_schema(), program);
    StateRoot source = root_with(schema, 107, 109);
    StateRoot moved(std::move(source));
    CHECK(!source.valid());
    CHECK(source.current_slot(70).empty());
    auto opened = moved.begin_candidate();
    CHECK(std::holds_alternative<CandidateGeneration>(opened));
    if (!std::holds_alternative<CandidateGeneration>(opened)) return;
    CandidateGeneration first =
        std::get<CandidateGeneration>(std::move(opened));
    CandidateGeneration second(std::move(first));
    CHECK(!first.valid());
    CHECK(first.read_slot(70).empty());
    CHECK(second.valid());
    CHECK(std::holds_alternative<std::monostate>(
        resolve_for_testing(moved, std::move(second),
                            ExecutionDisposition::RejectedBeforeSubmission)));

    StateRoot active_source = root_with(schema, 113, 127);
    auto active_open = active_source.begin_candidate();
    CHECK(std::holds_alternative<CandidateGeneration>(active_open));
    if (!std::holds_alternative<CandidateGeneration>(active_open)) return;
    CandidateGeneration active_candidate =
        std::get<CandidateGeneration>(std::move(active_open));
    CHECK(std::holds_alternative<std::monostate>(
        active_candidate.write_slot(70, bytes32(131))));
    ExecutionResult active_result = program_state_execution_result_for_testing(
        active_candidate, ExecutionDisposition::Completed);
    StateRoot active_moved(std::move(active_source));
    CandidateGeneration handle_moved(std::move(active_candidate));
    CHECK(!active_source.valid() && !active_candidate.valid());
    CHECK(std::holds_alternative<std::monostate>(
        active_moved.resolve_candidate(std::move(handle_moved),
                                       std::move(active_result))));
    CHECK(read32(active_moved.current_slot(70)) == 131);
}

void test_nonallocating_limits_and_digest_chunks() {
    const uint64_t thirty_two_gib = uint64_t{32} << 30;
    const uint64_t sixty_four_gib = uint64_t{64} << 30;
    uint64_t wire_bytes = 0;
    CHECK(program_state_wire_size_for_testing(
        std::array<uint32_t, 1>{0},
        std::array<uint64_t, 1>{thirty_two_gib}, &wire_bytes));
    CHECK(wire_bytes < sixty_four_gib);
    CHECK(!program_state_wire_size_for_testing(
        std::array<uint32_t, 1>{0},
        std::array<uint64_t, 1>{thirty_two_gib + 1}, &wire_bytes));

    const uint64_t exact_second = thirty_two_gib - 864;
    CHECK(program_state_wire_size_for_testing(
        std::array<uint32_t, 2>{0, 0},
        std::array<uint64_t, 2>{thirty_two_gib, exact_second},
        &wire_bytes));
    CHECK(wire_bytes == sixty_four_gib);
    CHECK(!program_state_wire_size_for_testing(
        std::array<uint32_t, 2>{0, 0},
        std::array<uint64_t, 2>{thirty_two_gib, exact_second + 1},
        &wire_bytes));
    CHECK(!program_state_wire_size_for_testing(
        std::array<uint32_t, 1>{33},
        std::array<uint64_t, 1>{1}, &wire_bytes));
    std::vector<uint32_t> too_many_ranks(4097, 0);
    std::vector<uint64_t> too_many_cells(4097, 1);
    CHECK(!program_state_wire_size_for_testing(
        too_many_ranks, too_many_cells, &wire_bytes));

    const uint64_t four_gib = uint64_t{1} << 32;
    const auto chunks =
        program_state_digest_chunks_for_testing(four_gib + 1);
    CHECK(!chunks.empty());
    CHECK(chunks.back().offset + chunks.back().length == four_gib + 1);
    CHECK(chunks.back().length <= 1024 * 1024);
}

void test_identity_ignores_local_ids_alias_labels_and_order() {
    const VerifiedProgram left_program = verified_program(two_slot_program(false));
    const VerifiedProgram right_program = verified_program(two_slot_program(true));
    CHECK(program_digest(left_program) == program_digest(right_program));
    const VerifiedStateSchema left_schema =
        verified_schema(two_slot_schema(false), left_program);
    const VerifiedStateSchema right_schema =
        verified_schema(two_slot_schema(true), right_program);
    CHECK(state_schema_digest(left_schema) == state_schema_digest(right_schema));

    StateRoot left = root_with(left_schema, 79, 83, false);
    StateRoot right = root_with(right_schema, 79, 83, true);
    auto left_saved = left.save();
    auto right_saved = right.save();
    CHECK(std::holds_alternative<std::vector<uint8_t>>(left_saved));
    CHECK(std::holds_alternative<std::vector<uint8_t>>(right_saved));
    if (std::holds_alternative<std::vector<uint8_t>>(left_saved) &&
        std::holds_alternative<std::vector<uint8_t>>(right_saved)) {
        CHECK(std::get<std::vector<uint8_t>>(left_saved) ==
              std::get<std::vector<uint8_t>>(right_saved));
        auto cross = restore_state_root(
            right_schema, std::get<std::vector<uint8_t>>(left_saved));
        CHECK(std::holds_alternative<StateRoot>(cross));
        if (std::holds_alternative<StateRoot>(cross)) {
            const StateRoot& restored = std::get<StateRoot>(cross);
            CHECK(read32(restored.current_slot(700)) == 79);
            CHECK(read64(restored.current_slot(800)) == 83);
        }
    }
}

void test_state_schema_structural_wire_roundtrip() {
    const VerifiedProgram program = verified_program(parameter_state_program());
    StateSchema source;
    source.dimension_bindings.push_back({5, 3});
    source.slots.push_back({70, ElementType::F16, {3}, 91});
    const VerifiedStateSchema schema = verified_schema(std::move(source), program);
    const auto wire = encode_state_schema_wire(schema);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
    if (!std::holds_alternative<std::vector<uint8_t>>(wire)) return;
    const auto& bytes = std::get<std::vector<uint8_t>>(wire);
    const auto decoded = decode_state_schema_wire(bytes, program);
    CHECK(std::holds_alternative<VerifiedStateSchema>(decoded));
    if (const auto* roundtrip = std::get_if<VerifiedStateSchema>(&decoded)) {
        CHECK(state_program_digest(*roundtrip) == program_digest(program));
        CHECK(state_schema_digest(*roundtrip) == state_schema_digest(schema));
    }

    auto trailing = bytes;
    trailing.push_back(0);
    CHECK(std::holds_alternative<CompatibilityReport>(
        decode_state_schema_wire(trailing, program)));
    auto bad_magic = bytes;
    bad_magic.front() ^= 1;
    CHECK(std::holds_alternative<CompatibilityReport>(
        decode_state_schema_wire(bad_magic, program)));
    auto bad_version = bytes;
    bad_version[8] = 2;
    CHECK(std::holds_alternative<CompatibilityReport>(
        decode_state_schema_wire(bad_version, program)));
    auto bad_alias = bytes;
    std::fill(bad_alias.end() - 4, bad_alias.end(), UINT8_MAX);
    CHECK(std::holds_alternative<CompatibilityReport>(
        decode_state_schema_wire(bad_alias, program)));

    const VerifiedProgram wrong_program = verified_program(two_slot_program());
    CHECK(std::holds_alternative<CompatibilityReport>(
        decode_state_schema_wire(bytes, wrong_program)));
}

} // namespace

int main() {
    test_two_unrelated_cells_publish_atomically();
    test_read_views_own_payload_lifetime();
    test_execution_disposition_matrix_and_abandonment();
    test_receipt_binds_effect_boundary();
    test_schema_shape_alias_and_identity_reject();
    test_all_element_types_and_immutable_write_reject();
    test_stateless_root_roundtrip();
    test_zero_extent_state_roundtrip();
    test_alias_cells_share_one_candidate_payload();
    test_symbolic_shape_binding_is_exact();
    test_discard_is_byte_exact_after_each_effect_boundary();
    test_execution_failure_poisons_root();
    test_save_restore_binds_program_and_every_cell();
    test_wire_regions_reject_without_mutating_live_root();
    test_next_generation_reads_published_overlay();
    test_one_active_foreign_and_second_resolution_reject();
    test_move_invalidates_source_handles();
    test_nonallocating_limits_and_digest_chunks();
    test_identity_ignores_local_ids_alias_labels_and_order();
    test_state_schema_structural_wire_roundtrip();
    return test_summary("test_program_state");
}
