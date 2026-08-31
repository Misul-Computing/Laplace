#include "program_reference.h"
#include "test_util.h"

#include <array>
#include <bit>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

using namespace Laplace;

namespace {

constexpr uint32_t kPhysicalFixtureValue = 0xf0000000u;

Instruction constant(uint32_t id, uint32_t value_id, float value) {
    Instruction instruction;
    instruction.id = id;
    instruction.primitive = {Primitive::Constant, 1, 0};
    instruction.outputs = {{value_id, {ElementType::F32, {}}}};
    instruction.attributes = ConstantAttributes{std::bit_cast<uint32_t>(value)};
    return instruction;
}

ValueType tensor_type() {
    return {ElementType::F32,
            {{DimensionExpression::Constant, 2, {}},
             {DimensionExpression::Constant, 2, {}}}};
}

Program tensor_program() {
    Region region;
    region.id = 1;
    Instruction add;
    add.id = 1;
    add.primitive = {Primitive::Add, 1, 0};
    add.inputs = {10, 11};
    add.outputs = {{12, tensor_type()}};
    region.instructions = {add};
    region.arguments = {{10, tensor_type()}, {11, tensor_type()}};
    region.yields = {12};
    Function function;
    function.id = 9;
    function.entry_region_id = 1;
    function.regions = {region};
    function.result_types = {tensor_type()};
    Program result;
    result.functions = {function};
    result.exports = {{9, 0, tensor_type()}};
    return result;
}

Program program(bool with_state) {
    Region region;
    region.id = 1;
    region.instructions = {constant(1, 10, 2.0f), constant(2, 11, 3.0f)};
    Instruction add;
    add.id = 3;
    add.primitive = {Primitive::Add, 1, 0};
    add.inputs = {10, 11};
    add.outputs = {{12, {ElementType::F32, {}}}};
    region.instructions.push_back(add);
    if (with_state) {
        Instruction write;
        write.id = 4;
        write.primitive = {Primitive::StateWrite, 1, 0};
        write.inputs = {12};
        write.attributes = StateAttributes{70};
        region.instructions.push_back(write);
        Instruction read;
        read.id = 5;
        read.primitive = {Primitive::StateRead, 1, 0};
        read.outputs = {{13, {ElementType::F32, {}}}};
        read.attributes = StateAttributes{70};
        read.effect_predecessors = {4};
        region.instructions.push_back(read);
        region.yields = {13};
    } else {
        region.yields = {12};
    }
    Function function;
    function.id = 9;
    function.entry_region_id = 1;
    function.regions = {region};
    function.result_types = {{ElementType::F32, {}}};
    Program result;
    result.functions = {function};
    result.exports = {{9, 0, {ElementType::F32, {}}}};
    if (with_state) result.state_references = {{70, {ElementType::F32, {}}, UINT32_MAX, true}};
    return result;
}

Program u64_loop_program() {
    const ValueType u64{ElementType::U64, {}};
    Region body;
    body.id = 20;
    body.arguments = {{21, u64}, {22, u64}};
    body.instructions = {Instruction{23, {Primitive::Constant, 1, 0}, {},
                                     {{24, u64}}, {}, {}, ConstantAttributes{1}},
                         Instruction{25, {Primitive::Add, 1, 0}, {22, 24},
                                     {{26, u64}}, {}, {}, NoAttributes{}}};
    body.yields = {26};
    Region root;
    root.id = 10;
    root.instructions = {
        Instruction{11, {Primitive::Constant, 1, 0}, {}, {{12, u64}}, {}, {},
                    ConstantAttributes{0}},
        Instruction{13, {Primitive::BoundedLoop, 1, 0}, {12}, {{14, u64}}, {20},
                    {}, LoopAttributes{0, 4, 1}}};
    root.yields = {14};
    Function function;
    function.id = 1;
    function.entry_region_id = 10;
    function.regions = {body, root};
    function.result_types = {u64};
    Program result;
    result.functions = {function};
    result.exports = {{1, 0, u64}};
    return result;
}

std::optional<VerifiedPhysicalProgramPackage> package_for(const Program& source) {
    PhysicalProgram physical;
    physical.planes.push_back({PhysicalPlaneStorage::External, 1, 0, 0});
    physical.instructions.push_back({PhysicalOpcode::ConstIndex, PhysicalValueType::Index,
                                     {kNoPhysicalValue, kNoPhysicalValue, kNoPhysicalValue},
                                     kNoPhysicalPlane, kNoPhysicalPolicy, 0, 0,
                                     PhysicalBitOrder::Lsb0Little});
    physical.instructions.push_back({PhysicalOpcode::LoadBits, PhysicalValueType::U32,
                                     {0, kNoPhysicalValue, kNoPhysicalValue}, 0,
                                     kNoPhysicalPolicy, 0, 8,
                                     PhysicalBitOrder::Lsb0Little});
    physical.result = 1;
    const auto wire = encode_physical_program(physical);
    const auto digest = physical_program_digest(physical);
    Program linked_source = source;
    CHECK(!linked_source.functions.empty());
    if (linked_source.functions.empty()) return std::nullopt;
    Function& linked_function = linked_source.functions.front();
    auto entry = std::find_if(
        linked_function.regions.begin(), linked_function.regions.end(),
        [&](const Region& region) { return region.id == linked_function.entry_region_id; });
    CHECK(entry != linked_function.regions.end());
    if (entry == linked_function.regions.end()) return std::nullopt;
    const auto existing_physical = std::find_if(
        entry->arguments.begin(), entry->arguments.end(),
        [](const TypedValue& value) { return value.id == kPhysicalFixtureValue; });
    if (existing_physical == entry->arguments.end())
        entry->arguments.push_back({kPhysicalFixtureValue, {ElementType::U32, {}}});
    else {
        const ValueType expected_physical{ElementType::U32, {}};
        CHECK(existing_physical->type == expected_physical);
    }
    const uint32_t linked_function_id = linked_function.id;
    const auto semantic = verify_and_canonicalize_program(std::move(linked_source));
    CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
    CHECK(std::holds_alternative<PhysicalProgramDigest>(digest));
    CHECK(std::holds_alternative<VerifiedProgram>(semantic));
    if (const auto* report = std::get_if<CompatibilityReport>(&semantic))
        CHECK_MSG(false, "reference semantic fixture rejected: code=%u detail=%s",
                  static_cast<unsigned>(report->code), report->detail.c_str());
    if (!std::holds_alternative<std::vector<uint8_t>>(wire) ||
        !std::holds_alternative<PhysicalProgramDigest>(digest) ||
        !std::holds_alternative<VerifiedProgram>(semantic)) return std::nullopt;
    auto blob = ArtifactSet::make_owned_blob(ArtifactId{1}, ArtifactRole::Primary,
                                             std::array<uint8_t, 1>{0x5a});
    CHECK(std::holds_alternative<PackageView>(blob));
    if (!std::holds_alternative<PackageView>(blob)) return std::nullopt;
    ArtifactIndexInput index_input;
    index_input.artifacts.push_back(std::get<PackageView>(std::move(blob)));
    auto index = ArtifactIndex::build(std::move(index_input));
    CHECK(std::holds_alternative<ArtifactIndex>(index));
    if (!std::holds_alternative<ArtifactIndex>(index)) return std::nullopt;
    PhysicalProgramRecord record{std::get<PhysicalProgramDigest>(digest),
                                 std::get<std::vector<uint8_t>>(wire),
                                 {ElementType::U32, {}}};
    PhysicalResourceBinding binding;
    binding.resource_id = 0;
    binding.program_digest = record.digest;
    binding.semantic_function_id = linked_function_id;
    binding.semantic_value_id = kPhysicalFixtureValue;
    binding.planes = {{0, ArtifactId{1}, 0, 1}};
    const auto loaded = load_physical_program_package(
        std::get<ArtifactIndex>(std::move(index)),
        std::get<VerifiedProgram>(std::move(semantic)),
        std::span<const PhysicalProgramRecord>(&record, 1),
        std::span<const PhysicalResourceBinding>(&binding, 1));
    CHECK(std::holds_alternative<VerifiedPhysicalProgramPackage>(loaded));
    if (!std::holds_alternative<VerifiedPhysicalProgramPackage>(loaded)) return std::nullopt;
    return std::get<VerifiedPhysicalProgramPackage>(std::move(loaded));
}

Program physical_input_program() {
    const ValueType u32{ElementType::U32, {}};
    Region region;
    region.id = 1;
    region.arguments = {{kPhysicalFixtureValue, u32}};
    region.yields = {kPhysicalFixtureValue};
    Function function;
    function.id = 9;
    function.entry_region_id = region.id;
    function.regions = {region};
    function.result_types = {u32};
    Program result;
    result.functions = {function};
    result.exports = {{function.id, 0, u32}};
    return result;
}

void test_bound_physical_resource_feeds_semantic_input() {
    const auto package = package_for(physical_input_program());
    CHECK(package.has_value());
    if (!package) return;
    ReferenceState state;
    const auto result = execute_reference(*package, state, {});
    CHECK(std::holds_alternative<ReferenceResult>(result));
    if (const auto* output = std::get_if<ReferenceResult>(&result)) {
        CHECK(output->exports.size() == 1);
        const ReferenceValue expected{ElementType::U32, {}, {0x5a}};
        CHECK(output->exports[0] == expected);
    }
}

void test_reference_execution_and_transaction() {
    const auto package = package_for(program(false));
    CHECK(package.has_value());
    if (!package) return;
    const auto decoded = decode_reference_resource(*package, 0, {});
    CHECK(std::holds_alternative<ScalarValue>(decoded));
    if (const auto* value = std::get_if<ScalarValue>(&decoded)) {
        CHECK(value->type == ElementType::U32);
        CHECK(value->bits == 0x5a);
    }
    const auto missing_resource = decode_reference_resource(*package, 99, {});
    CHECK(std::holds_alternative<CompatibilityReport>(missing_resource));
    ReferenceState state;
    const auto result = execute_reference(*package, state, {});
    CHECK(std::holds_alternative<ReferenceResult>(result));
    if (const auto* output = std::get_if<ReferenceResult>(&result)) {
        CHECK(output->exports.size() == 1);
        const ReferenceValue expected{ElementType::F32, {},
                                      {std::bit_cast<uint32_t>(5.0f)}};
        CHECK(output->exports[0] == expected);
        CHECK(output->generation == 1);
    }

    const auto state_package = package_for(program(true));
    CHECK(state_package.has_value());
    if (!state_package) return;
    ReferenceState stateful;
    const auto first = execute_reference(*state_package, stateful, {});
    CHECK(std::holds_alternative<ReferenceResult>(first));
    CHECK(stateful.slots.size() == 1);
    CHECK(stateful.slots[0].first == 70);
    CHECK(stateful.slots[0].second.bits[0] == std::bit_cast<uint32_t>(5.0f));
    const std::array<ReferenceInput, 1> invalid_inputs = {
        ReferenceInput{UINT32_MAX, ReferenceValue{}}};
    const auto bad = execute_reference(*state_package, stateful, invalid_inputs);
    CHECK(std::holds_alternative<CompatibilityReport>(bad));
    CHECK(stateful.slots[0].second.bits[0] == std::bit_cast<uint32_t>(5.0f));

    const auto tensor_package = package_for(tensor_program());
    CHECK(tensor_package.has_value());
    if (!tensor_package) return;
    const ReferenceValue left{ElementType::F32, {2, 2},
                              {std::bit_cast<uint32_t>(1.0f),
                               std::bit_cast<uint32_t>(2.0f),
                               std::bit_cast<uint32_t>(3.0f),
                               std::bit_cast<uint32_t>(4.0f)}};
    const ReferenceValue right{ElementType::F32, {2, 2},
                               {std::bit_cast<uint32_t>(5.0f),
                                std::bit_cast<uint32_t>(6.0f),
                                std::bit_cast<uint32_t>(7.0f),
                                std::bit_cast<uint32_t>(8.0f)}};
    const std::array<ReferenceInput, 2> tensor_inputs = {
        ReferenceInput{10, left}, ReferenceInput{11, right}};
    ReferenceState tensor_state;
    const auto tensor_result = execute_reference(*tensor_package, tensor_state,
                                                 tensor_inputs);
    CHECK(std::holds_alternative<ReferenceResult>(tensor_result));
    if (const auto* output = std::get_if<ReferenceResult>(&tensor_result)) {
        CHECK(output->exports.size() == 1);
        CHECK(output->exports[0].extents == std::vector<uint64_t>({2, 2}));
        CHECK(std::bit_cast<float>(static_cast<uint32_t>(output->exports[0].bits[3])) == 12.0f);
    }
    ReferenceState bad_tensor_state;
    const std::array<ReferenceInput, 2> wrong_shape_inputs = {
        ReferenceInput{10, ReferenceValue{ElementType::F32, {2},
                                          {std::bit_cast<uint32_t>(1.0f),
                                           std::bit_cast<uint32_t>(2.0f)}}},
        ReferenceInput{11, right}};
    CHECK(std::holds_alternative<CompatibilityReport>(
        execute_reference(*tensor_package, bad_tensor_state, wrong_shape_inputs)));
    const std::array<ReferenceInput, 3> unknown_inputs = {
        tensor_inputs[0], tensor_inputs[1],
        ReferenceInput{99, ReferenceValue{ElementType::F32, {},
                                          {std::bit_cast<uint32_t>(1.0f)}}}};
    CHECK(std::holds_alternative<CompatibilityReport>(
        execute_reference(*tensor_package, bad_tensor_state, unknown_inputs)));

    const auto loop_package = package_for(u64_loop_program());
    CHECK(loop_package.has_value());
    if (loop_package) {
        ReferenceState loop_state;
        const auto loop_result = execute_reference(*loop_package, loop_state, {});
        CHECK(std::holds_alternative<ReferenceResult>(loop_result));
        if (const auto* output = std::get_if<ReferenceResult>(&loop_result)) {
            CHECK(output->exports.size() == 1);
            CHECK(output->exports[0].type == ElementType::U64);
            CHECK(output->exports[0].bits[0] == 4);
        }
    }
}

} // namespace

int main() {
    test_bound_physical_resource_feeds_semantic_input();
    test_reference_execution_and_transaction();
    return test_summary("test_program_reference");
}
