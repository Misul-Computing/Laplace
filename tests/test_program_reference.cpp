#include "program_reference.h"
#include "test_util.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cmath>
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

Program algebra_program(Primitive primitive, bool unary) {
    const ValueType scalar{ElementType::F32, {}};
    Region region;
    region.id = 1;
    region.arguments = {{10, scalar}};
    std::vector<uint32_t> inputs = {10};
    if (!unary) {
        region.arguments.push_back({11, scalar});
        inputs.push_back(11);
    }
    region.instructions = {
        Instruction{20, {primitive, 1, 0}, std::move(inputs), {{30, scalar}},
                    {}, {}, NoAttributes{}}};
    region.yields = {30};
    Function function{9, 1, {std::move(region)}, {scalar}};
    Program result;
    result.functions = {std::move(function)};
    result.exports = {{9, 0, scalar}};
    return result;
}

std::variant<ReferenceResult, CompatibilityReport> run_algebra(
    Primitive primitive, float left, std::optional<float> right = std::nullopt) {
    auto verified = verify_and_canonicalize_program(
        algebra_program(primitive, !right.has_value()));
    if (const auto* report = std::get_if<CompatibilityReport>(&verified))
        return *report;
    const std::array<ReferenceInput, 2> inputs = {
        ReferenceInput{10, {ElementType::F32, {},
                            {std::bit_cast<uint32_t>(left)}}},
        ReferenceInput{11, {ElementType::F32, {},
                            {std::bit_cast<uint32_t>(right.value_or(0.0f))}}}};
    ReferenceState state;
    return execute_reference_program(
        std::get<VerifiedProgram>(verified), state,
        std::span<const ReferenceInput>(inputs).first(right ? 2 : 1));
}

void check_algebra(Primitive primitive, float left, std::optional<float> right,
                   float expected) {
    const auto result = run_algebra(primitive, left, right);
    CHECK(std::holds_alternative<ReferenceResult>(result));
    if (const auto* value = std::get_if<ReferenceResult>(&result)) {
        CHECK(value->exports.size() == 1);
        if (!value->exports.empty()) {
            const float observed = std::bit_cast<float>(
                static_cast<uint32_t>(value->exports.front().bits.front()));
            CHECK(std::abs(observed - expected) <= 2.0e-6f);
        }
    }
}

void test_generic_f32_algebra_reference() {
    const auto power = run_algebra(Primitive::Pow, 10000.0f, -0.375f);
    CHECK(std::holds_alternative<ReferenceResult>(power));
    if (const auto* value = std::get_if<ReferenceResult>(&power))
        CHECK(value->exports.front().bits == std::vector<uint64_t>{
            std::bit_cast<uint32_t>(std::pow(10000.0f, -0.375f))});
    CHECK(std::holds_alternative<CompatibilityReport>(
        run_algebra(Primitive::Pow, -2.0f, 0.5f)));
    CHECK(std::holds_alternative<CompatibilityReport>(
        run_algebra(Primitive::Pow, 2.0f, 1000.0f)));
    check_algebra(Primitive::Subtract, 5.0f, 2.0f, 3.0f);
    check_algebra(Primitive::Divide, 5.0f, 2.0f, 2.5f);
    check_algebra(Primitive::Maximum, -3.0f, 2.0f, 2.0f);
    check_algebra(Primitive::Negate, 1.25f, std::nullopt, -1.25f);
    check_algebra(Primitive::Exp, 1.0f, std::nullopt, std::exp(1.0f));
    check_algebra(Primitive::Log, 2.0f, std::nullopt, std::log(2.0f));
    check_algebra(Primitive::Rsqrt, 4.0f, std::nullopt, 0.5f);
    check_algebra(Primitive::Sin, 0.5f, std::nullopt, std::sin(0.5f));
    check_algebra(Primitive::Cos, 0.5f, std::nullopt, std::cos(0.5f));

    CHECK(std::holds_alternative<CompatibilityReport>(
        run_algebra(Primitive::Divide, 1.0f, 0.0f)));
    CHECK(std::holds_alternative<CompatibilityReport>(
        run_algebra(Primitive::Exp, 1000.0f)));
    CHECK(std::holds_alternative<CompatibilityReport>(
        run_algebra(Primitive::Log, -2.0f)));
    CHECK(std::holds_alternative<CompatibilityReport>(
        run_algebra(Primitive::Rsqrt, -1.0f)));
}

TensorIndexExpr iterator_index(uint32_t index) {
    return {TensorIndexExpression::Iterator, static_cast<int64_t>(index), {}};
}

TensorIndexMap index_map(std::initializer_list<uint32_t> iterators) {
    TensorIndexMap result;
    for (uint32_t iterator : iterators)
        result.results.push_back(iterator_index(iterator));
    return result;
}

Program dynamic_gather_program() {
    const auto dimension = [](uint64_t value) {
        return DimensionExpr{DimensionExpression::Constant, value, {}};
    };
    const ValueType table{ElementType::F32, {dimension(4), dimension(3)}};
    const ValueType index{ElementType::U32, {}};
    const ValueType output{ElementType::F32, {dimension(3)}};
    const ValueType scalar{ElementType::F32, {}};
    Region body{20, {{21, scalar}, {22, index}, {23, scalar}},
                {Instruction{24, {Primitive::Add, 1, 0}, {21, 23},
                             {{25, scalar}}, {}, {}, NoAttributes{}}},
                {25}};
    TensorIndexMap table_map;
    table_map.results = {
        TensorIndexExpr{TensorIndexExpression::SourceScalar, 1, {}},
        iterator_index(0)};
    StructuredTensorAttributes attributes;
    attributes.source_count = 2;
    attributes.iteration_dimensions = {dimension(3)};
    attributes.iterator_kinds = {TensorIteratorKind::Parallel};
    attributes.indexing_maps = {
        std::move(table_map), TensorIndexMap{}, index_map({0})};
    Region root;
    root.id = 1;
    root.arguments = {{10, table}, {11, index}};
    root.instructions = {
        Instruction{12, {Primitive::Constant, 1, 0}, {}, {{13, output}},
                    {}, {}, ConstantAttributes{0}},
        Instruction{14, {Primitive::StructuredTensor, 1, 0}, {10, 11, 13},
                    {{15, output}}, {20}, {}, std::move(attributes)}};
    root.yields = {15};
    Function function{9, 1, {std::move(body), std::move(root)}, {output}};
    Program program;
    program.minor = 1;
    program.functions = {std::move(function)};
    program.exports = {{9, 0, output}};
    return program;
}

void test_data_dependent_tensor_index_reference() {
    auto verified = verify_and_canonicalize_program(dynamic_gather_program());
    CHECK(std::holds_alternative<VerifiedProgram>(verified));
    if (!std::holds_alternative<VerifiedProgram>(verified)) return;
    const auto bits = [](float value) {
        return static_cast<uint64_t>(std::bit_cast<uint32_t>(value));
    };
    ReferenceValue table{ElementType::F32, {4, 3},
                         {bits(1), bits(2), bits(3), bits(4), bits(5), bits(6),
                          bits(7), bits(8), bits(9), bits(10), bits(11), bits(12)}};
    const std::array<ReferenceInput, 2> inputs = {
        ReferenceInput{10, std::move(table)},
        ReferenceInput{11, {ElementType::U32, {}, {2}}}};
    ReferenceState state;
    const auto result = execute_reference_program(
        std::get<VerifiedProgram>(verified), state, inputs);
    CHECK(std::holds_alternative<ReferenceResult>(result));
    if (const auto* output = std::get_if<ReferenceResult>(&result)) {
        CHECK(output->exports.size() == 1);
        if (!output->exports.empty())
            CHECK(output->exports.front().bits ==
                  std::vector<uint64_t>({bits(7), bits(8), bits(9)}));
    }

    auto invalid = inputs;
    invalid[1].value.bits.front() = 4;
    ReferenceState invalid_state;
    CHECK(std::holds_alternative<CompatibilityReport>(
        execute_reference_program(std::get<VerifiedProgram>(verified),
                                  invalid_state, invalid)));
}

Program structured_matmul_program() {
    const auto dimension = [](uint64_t value) {
        return DimensionExpr{DimensionExpression::Constant, value, {}};
    };
    const ValueType left{ElementType::F32, {dimension(2), dimension(3)}};
    const ValueType right{ElementType::F32, {dimension(3), dimension(2)}};
    const ValueType output{ElementType::F32, {dimension(2), dimension(2)}};
    const ValueType scalar{ElementType::F32, {}};

    Region body;
    body.id = 20;
    body.arguments = {{21, scalar}, {22, scalar}, {23, scalar}};
    body.instructions = {
        Instruction{24, {Primitive::Multiply, 1, 0}, {21, 22}, {{25, scalar}},
                    {}, {}, NoAttributes{}},
        Instruction{26, {Primitive::Add, 1, 0}, {23, 25}, {{27, scalar}},
                    {}, {}, NoAttributes{}},
    };
    body.yields = {27};

    StructuredTensorAttributes attributes;
    attributes.source_count = 2;
    attributes.iteration_dimensions = {dimension(2), dimension(2), dimension(3)};
    attributes.iterator_kinds = {
        TensorIteratorKind::Parallel,
        TensorIteratorKind::Parallel,
        TensorIteratorKind::Reduction,
    };
    attributes.indexing_maps = {
        index_map({0, 2}), index_map({2, 1}), index_map({0, 1})};

    Instruction zero{10, {Primitive::Constant, 1, 0}, {}, {{12, output}},
                     {}, {}, ConstantAttributes{0}};
    Instruction contraction{30, {Primitive::StructuredTensor, 1, 0},
                            {10, 11, 12}, {{13, output}}, {20}, {},
                            std::move(attributes)};
    Region root;
    root.id = 1;
    root.arguments = {{10, left}, {11, right}};
    root.instructions = {std::move(zero), std::move(contraction)};
    root.yields = {13};
    Function function;
    function.id = 9;
    function.entry_region_id = 1;
    function.regions = {std::move(body), std::move(root)};
    function.result_types = {output};
    Program program;
    program.minor = 1;
    program.functions = {std::move(function)};
    program.exports = {{9, 0, output}};
    return program;
}

Program structured_transpose_broadcast_program() {
    const auto dimension = [](uint64_t value) {
        return DimensionExpr{DimensionExpression::Constant, value, {}};
    };
    const ValueType matrix{ElementType::F32, {dimension(3), dimension(2)}};
    const ValueType bias{ElementType::F32, {dimension(3)}};
    const ValueType output{ElementType::F32, {dimension(2), dimension(3)}};
    const ValueType scalar{ElementType::F32, {}};
    Region body{20, {{21, scalar}, {22, scalar}, {23, scalar}},
                {Instruction{24, {Primitive::Add, 1, 0}, {21, 22},
                             {{25, scalar}}, {}, {}, NoAttributes{}}},
                {25}};
    StructuredTensorAttributes attributes;
    attributes.source_count = 2;
    attributes.iteration_dimensions = {dimension(2), dimension(3)};
    attributes.iterator_kinds = {
        TensorIteratorKind::Parallel, TensorIteratorKind::Parallel};
    attributes.indexing_maps = {
        index_map({1, 0}), index_map({1}), index_map({0, 1})};
    Region root;
    root.id = 1;
    root.arguments = {{10, matrix}, {11, bias}};
    root.instructions = {
        Instruction{12, {Primitive::Constant, 1, 0}, {}, {{13, output}},
                    {}, {}, ConstantAttributes{0}},
        Instruction{14, {Primitive::StructuredTensor, 1, 0}, {10, 11, 13},
                    {{15, output}}, {20}, {}, std::move(attributes)}};
    root.yields = {15};
    Function function{9, 1, {std::move(body), std::move(root)}, {output}};
    Program program;
    program.minor = 1;
    program.functions = {std::move(function)};
    program.exports = {{9, 0, output}};
    return program;
}

Program structured_reduction_program() {
    const auto dimension = [](uint64_t value) {
        return DimensionExpr{DimensionExpression::Constant, value, {}};
    };
    const ValueType matrix{ElementType::F32, {dimension(2), dimension(3)}};
    const ValueType output{ElementType::F32, {dimension(2)}};
    const ValueType scalar{ElementType::F32, {}};
    Region body{20, {{21, scalar}, {22, scalar}},
                {Instruction{23, {Primitive::Add, 1, 0}, {21, 22},
                             {{24, scalar}}, {}, {}, NoAttributes{}}},
                {24}};
    StructuredTensorAttributes attributes;
    attributes.source_count = 1;
    attributes.iteration_dimensions = {dimension(2), dimension(3)};
    attributes.iterator_kinds = {
        TensorIteratorKind::Parallel, TensorIteratorKind::Reduction};
    attributes.indexing_maps = {index_map({0, 1}), index_map({0})};
    Region root;
    root.id = 1;
    root.arguments = {{10, matrix}};
    root.instructions = {
        Instruction{11, {Primitive::Constant, 1, 0}, {}, {{12, output}},
                    {}, {}, ConstantAttributes{0}},
        Instruction{13, {Primitive::StructuredTensor, 1, 0}, {10, 12},
                    {{14, output}}, {20}, {}, std::move(attributes)}};
    root.yields = {14};
    Function function{9, 1, {std::move(body), std::move(root)}, {output}};
    Program program;
    program.minor = 1;
    program.functions = {std::move(function)};
    program.exports = {{9, 0, output}};
    return program;
}

Program oversized_structured_reduction_program() {
    const auto dimension = [](uint64_t value) {
        return DimensionExpr{DimensionExpression::Constant, value, {}};
    };
    const ValueType scalar{ElementType::F32, {}};
    Region body{20, {{21, scalar}, {22, scalar}},
                {Instruction{23, {Primitive::Add, 1, 0}, {21, 22},
                             {{24, scalar}}, {}, {}, NoAttributes{}}},
                {24}};
    StructuredTensorAttributes attributes;
    attributes.source_count = 1;
    attributes.iteration_dimensions = {dimension((uint64_t{1} << 24) + 1)};
    attributes.iterator_kinds = {TensorIteratorKind::Reduction};
    attributes.indexing_maps = {TensorIndexMap{}, TensorIndexMap{}};
    Region root;
    root.id = 1;
    root.arguments = {{10, scalar}};
    root.instructions = {
        Instruction{11, {Primitive::Constant, 1, 0}, {}, {{12, scalar}},
                    {}, {}, ConstantAttributes{0}},
        Instruction{13, {Primitive::StructuredTensor, 1, 0}, {10, 12},
                    {{14, scalar}}, {20}, {}, std::move(attributes)}};
    root.yields = {14};
    Function function{9, 1, {std::move(body), std::move(root)}, {scalar}};
    Program program;
    program.minor = 1;
    program.functions = {std::move(function)};
    program.exports = {{9, 0, scalar}};
    return program;
}

std::optional<VerifiedPhysicalProgramPackage> package_for(const Program& source);

void test_structured_tensor_reference() {
    const auto package = package_for(structured_matmul_program());
    CHECK(package.has_value());
    if (!package) return;
    const auto f32 = [](float value) {
        return static_cast<uint64_t>(std::bit_cast<uint32_t>(value));
    };
    const ReferenceValue left{ElementType::F32, {2, 3},
                              {f32(1), f32(2), f32(3),
                               f32(4), f32(5), f32(6)}};
    const ReferenceValue right{ElementType::F32, {3, 2},
                               {f32(1), f32(2), f32(3),
                                f32(4), f32(5), f32(6)}};
    const std::array<ReferenceInput, 2> inputs = {
        ReferenceInput{10, left}, ReferenceInput{11, right}};
    ReferenceState state;
    const auto result = execute_reference(*package, state, inputs);
    CHECK(std::holds_alternative<ReferenceResult>(result));
    if (const auto* output = std::get_if<ReferenceResult>(&result)) {
        CHECK(output->exports.size() == 1);
        CHECK(output->exports[0].extents == std::vector<uint64_t>({2, 2}));
        const std::vector<uint64_t> expected = {
            f32(22), f32(28), f32(49), f32(64)};
        CHECK(output->exports[0].bits == expected);
    }

    const auto transpose_package =
        package_for(structured_transpose_broadcast_program());
    CHECK(transpose_package.has_value());
    if (transpose_package) {
        const ReferenceValue matrix{ElementType::F32, {3, 2},
                                    {f32(1), f32(2), f32(3),
                                     f32(4), f32(5), f32(6)}};
        const ReferenceValue bias{ElementType::F32, {3},
                                  {f32(10), f32(20), f32(30)}};
        const std::array<ReferenceInput, 2> transpose_inputs = {
            ReferenceInput{10, matrix}, ReferenceInput{11, bias}};
        ReferenceState transpose_state;
        const auto transpose_result = execute_reference(
            *transpose_package, transpose_state, transpose_inputs);
        CHECK(std::holds_alternative<ReferenceResult>(transpose_result));
        if (const auto* values =
                std::get_if<ReferenceResult>(&transpose_result)) {
            CHECK(values->exports[0].bits ==
                  std::vector<uint64_t>({f32(11), f32(23), f32(35),
                                         f32(12), f32(24), f32(36)}));
        }
    }

    const auto reduction_package = package_for(structured_reduction_program());
    CHECK(reduction_package.has_value());
    if (reduction_package) {
        const ReferenceValue matrix{ElementType::F32, {2, 3},
                                    {f32(1), f32(2), f32(3),
                                     f32(4), f32(5), f32(6)}};
        const std::array<ReferenceInput, 1> reduction_inputs = {
            ReferenceInput{10, matrix}};
        ReferenceState reduction_state;
        const auto reduction_result = execute_reference(
            *reduction_package, reduction_state, reduction_inputs);
        CHECK(std::holds_alternative<ReferenceResult>(reduction_result));
        if (const auto* values =
                std::get_if<ReferenceResult>(&reduction_result)) {
            CHECK(values->exports[0].bits ==
                  std::vector<uint64_t>({f32(6), f32(15)}));
        }
    }

    const auto oversized_package =
        package_for(oversized_structured_reduction_program());
    CHECK(oversized_package.has_value());
    if (oversized_package) {
        const ReferenceValue scalar_input{ElementType::F32, {}, {f32(1)}};
        const std::array<ReferenceInput, 1> oversized_inputs = {
            ReferenceInput{10, scalar_input}};
        ReferenceState oversized_state;
        const auto oversized_result = execute_reference(
            *oversized_package, oversized_state, oversized_inputs);
        CHECK(std::holds_alternative<CompatibilityReport>(oversized_result));
        if (const auto* report =
                std::get_if<CompatibilityReport>(&oversized_result))
            CHECK(report->code == CompatibilityError::RUNTIME_NUMERICAL_FAILURE);
        CHECK(oversized_state.generation == 0);
    }
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

void test_coordinates_and_require_reference() {
    Program program = structured_reduction_program();
    const ValueType f32{ElementType::F32, {}}, u32{ElementType::U32, {}},
                    u64{ElementType::U64, {}}, i1{ElementType::I1, {}};
    Region& body = program.functions.front().regions.front();
    body.instructions = {
        Instruction{40, {Primitive::TensorCoordinate, 1, 0}, {}, {{41, u32}}, {}, {}, CoordinateAttributes{0}},
        Instruction{42, {Primitive::TensorCoordinate, 1, 0}, {}, {{43, u32}}, {}, {}, CoordinateAttributes{1}},
        Instruction{44, {Primitive::Convert, 1, 0}, {41}, {{45, f32}}, {}, {}, NoAttributes{}},
        Instruction{46, {Primitive::Convert, 1, 0}, {43}, {{47, f32}}, {}, {}, NoAttributes{}},
        constant(48, 49, 10.0f),
        Instruction{50, {Primitive::Multiply, 1, 0}, {45, 49}, {{51, f32}}, {}, {}, NoAttributes{}},
        Instruction{52, {Primitive::Add, 1, 0}, {51, 47}, {{53, f32}}, {}, {}, NoAttributes{}},
        Instruction{54, {Primitive::Add, 1, 0}, {53, 22}, {{55, f32}}, {}, {}, NoAttributes{}},
        Instruction{56, {Primitive::TensorCoordinate, 1, 0}, {}, {{57, u64}}, {}, {}, CoordinateAttributes{1}},
        Instruction{58, {Primitive::Constant, 1, 0}, {}, {{59, u64}}, {}, {}, ConstantAttributes{3}},
        Instruction{60, {Primitive::Less, 1, 0}, {57, 59}, {{61, i1}}, {}, {}, NoAttributes{}},
        Instruction{62, {Primitive::Require, 1, 0}, {61}, {{63, i1}}, {}, {}, NoAttributes{}},
        Instruction{64, {Primitive::Select, 1, 0}, {63, 55, 55}, {{65, f32}}, {}, {}, NoAttributes{}}};
    body.yields = {65};
    const auto run = [&] {
        const auto verified = verify_and_canonicalize_program(program);
        if (const auto* error = std::get_if<CompatibilityReport>(&verified))
            CHECK_MSG(false, "coordinate program: %s", error->detail.c_str());
        CHECK(std::holds_alternative<VerifiedProgram>(verified));
        if (const auto* error = std::get_if<CompatibilityReport>(&verified))
            return std::variant<ReferenceResult, CompatibilityReport>(*error);
        const std::array<ReferenceInput, 1> inputs = {
            ReferenceInput{10, {ElementType::F32, {2, 3}, std::vector<uint64_t>(6, 0)}}};
        ReferenceState state;
        return execute_reference_program(std::get<VerifiedProgram>(verified), state, inputs);
    };
    const auto result = run();
    CHECK(std::holds_alternative<ReferenceResult>(result));
    if (const auto* value = std::get_if<ReferenceResult>(&result))
        CHECK((value->exports.front().bits == std::vector<uint64_t>{
            std::bit_cast<uint32_t>(3.0f), std::bit_cast<uint32_t>(33.0f)}));
    body.instructions[9].attributes = ConstantAttributes{2};
    CHECK(std::holds_alternative<CompatibilityReport>(run()));
    Program require = algebra_program(Primitive::Require, true);
    auto& function = require.functions.front();
    function.result_types = {i1};
    function.regions.front().arguments.front().type = i1;
    function.regions.front().instructions.front().outputs.front().type = i1;
    require.exports.front().type = i1;
    const auto verified = verify_and_canonicalize_program(require);
    CHECK(std::holds_alternative<VerifiedProgram>(verified));
    if (const auto* proof = std::get_if<VerifiedProgram>(&verified)) {
        for (uint64_t predicate : {0u, 1u, 2u}) {
            const std::array<ReferenceInput, 1> inputs = {
                ReferenceInput{10, {ElementType::I1, {}, {predicate}}}};
            ReferenceState state;
            const auto actual = execute_reference_program(*proof, state, inputs);
            CHECK(std::holds_alternative<ReferenceResult>(actual) == (predicate == 1));
        }
    }
}

void test_arithmetic_policy_reference() {
    const auto run = [](Program program, float input) {
        const auto verified = verify_and_canonicalize_program(std::move(program));
        CHECK(std::holds_alternative<VerifiedProgram>(verified));
        if (const auto* error = std::get_if<CompatibilityReport>(&verified))
            return std::variant<ReferenceResult, CompatibilityReport>(*error);
        const std::array<ReferenceInput, 1> inputs = {
            ReferenceInput{10, {ElementType::F32, {}, {std::bit_cast<uint32_t>(input)}}}};
        ReferenceState state;
        return execute_reference_program(std::get<VerifiedProgram>(verified), state, inputs);
    };
    Program exponential = algebra_program(Primitive::Exp, true);
    for (unsigned flags = 0; flags < 4; ++flags) {
        exponential.functions.front().regions.front().instructions.front().attributes =
            ArithmeticAttributes{bool(flags & 1), bool(flags & 2)};
        const auto overflow = run(exponential, 100.0f);
        CHECK(std::holds_alternative<ReferenceResult>(overflow) == bool(flags & 2));
        const auto nonfinite = run(exponential, -INFINITY);
        CHECK(std::holds_alternative<ReferenceResult>(nonfinite) == bool(flags & 1));
    }
    Program silu = algebra_program(Primitive::Exp, true);
    const ValueType f32{ElementType::F32, {}};
    Region& root = silu.functions.front().regions.front();
    root.instructions = {
        Instruction{20, {Primitive::RequireFinite, 1, 0}, {10}, {{21, f32}}, {}, {}, NoAttributes{}},
        Instruction{22, {Primitive::Negate, 1, 0}, {21}, {{23, f32}}, {}, {}, NoAttributes{}},
        Instruction{24, {Primitive::Exp, 1, 0}, {23}, {{25, f32}}, {}, {}, ArithmeticAttributes{false, true}},
        constant(26, 27, 1.0f),
        Instruction{28, {Primitive::Add, 1, 0}, {27, 25}, {{29, f32}}, {}, {}, ArithmeticAttributes{true, true}},
        Instruction{31, {Primitive::Divide, 1, 0}, {21, 29}, {{32, f32}}, {}, {}, ArithmeticAttributes{true, true}},
        Instruction{33, {Primitive::RequireFinite, 1, 0}, {32}, {{30, f32}}, {}, {}, NoAttributes{}}};
    const auto result = run(silu, -100.0f);
    CHECK(std::holds_alternative<ReferenceResult>(result));
    if (const auto* value = std::get_if<ReferenceResult>(&result))
        CHECK(value->exports.front().bits == std::vector<uint64_t>{0x80000000});
    CHECK(std::holds_alternative<CompatibilityReport>(run(silu, INFINITY)));
    root.instructions[2].attributes = NoAttributes{};
    CHECK(std::holds_alternative<CompatibilityReport>(run(silu, -100.0f)));
    const Program guard = algebra_program(Primitive::RequireFinite, true);
    const auto negative_zero = run(guard, -0.0f);
    CHECK(std::holds_alternative<ReferenceResult>(negative_zero));
    if (const auto* value = std::get_if<ReferenceResult>(&negative_zero))
        CHECK(value->exports.front().bits == std::vector<uint64_t>{0x80000000});
    CHECK(std::holds_alternative<CompatibilityReport>(run(guard, INFINITY)));
    CHECK(std::holds_alternative<CompatibilityReport>(run(guard, NAN)));
}

void test_source_element_reference() {
    Program program = dynamic_gather_program();
    Region& root = program.functions.front().regions.back();
    root.arguments[1].type.dimensions = {
        {DimensionExpression::Constant, 1, {}}, {DimensionExpression::Constant, 3, {}}};
    auto& maps = std::get<StructuredTensorAttributes>(root.instructions.back().attributes).indexing_maps;
    const TensorIndexExpr zero{TensorIndexExpression::Constant, 0, {}};
    const TensorIndexExpr load{TensorIndexExpression::SourceElement, 1,
                              {zero, iterator_index(0)}};
    maps[1].results = {zero, iterator_index(0)};
    maps[0].results[0] = load;
    const auto run = [&](std::vector<uint64_t> indices) {
        const auto verified = verify_and_canonicalize_program(program);
        CHECK(std::holds_alternative<VerifiedProgram>(verified));
        if (const auto* error = std::get_if<CompatibilityReport>(&verified))
            return std::variant<ReferenceResult, CompatibilityReport>(*error);
        std::vector<uint64_t> data;
        for (size_t row = 0; row < 4; ++row)
            for (size_t column = 0; column < 3; ++column)
                data.push_back(std::bit_cast<uint32_t>(static_cast<float>(row * 10 + column)));
        const std::array<ReferenceInput, 2> inputs = {
            ReferenceInput{10, {ElementType::F32, {4, 3}, std::move(data)}},
            ReferenceInput{11, {ElementType::U32, {1, 3}, std::move(indices)}}};
        ReferenceState state;
        return execute_reference_program(std::get<VerifiedProgram>(verified), state, inputs);
    };
    const auto check = [&](std::vector<uint64_t> indices, std::vector<float> expected) {
        const auto actual = run(std::move(indices));
        CHECK(std::holds_alternative<ReferenceResult>(actual));
        if (const auto* result = std::get_if<ReferenceResult>(&actual)) {
            std::vector<uint64_t> bits;
            for (float value : expected) bits.push_back(std::bit_cast<uint32_t>(value));
            CHECK(result->exports.front().bits == bits);
        }
    };
    check({2, 0, 1}, {20, 1, 12});
    maps[0].results[0].operands[1] = load;
    check({2, 0, 1}, {10, 21, 2});
    CHECK(std::holds_alternative<CompatibilityReport>(run({3, 0, 1})));
    maps[0].bounds = TensorBoundsMode::Zero;
    CHECK(std::holds_alternative<CompatibilityReport>(run({3, 0, 1})));
    maps[0].results[0] = load;
    check({4, 0, 1}, {0, 1, 12});
    CHECK(std::holds_alternative<CompatibilityReport>(run({uint64_t{UINT32_MAX} + 1, 0, 1})));
    // An earlier zero-padded axis must not hide a bad later source lookup.
    maps[0].results = {{TensorIndexExpression::Constant, -1, {}}, load};
    maps[0].results[1].operands[1] = {TensorIndexExpression::Constant, 3, {}};
    CHECK(std::holds_alternative<CompatibilityReport>(run({2, 0, 1})));
    maps[0].results = {load, iterator_index(0)};
    maps[0].results[0].operands[1] = {
        TensorIndexExpression::Multiply, 0,
        {load, {TensorIndexExpression::Constant, INT64_MAX, {}}}};
    CHECK(std::holds_alternative<CompatibilityReport>(run({2, 0, 1})));
}

void test_typed_structured_body_reference() {
    Program program = dynamic_gather_program();
    Region& body = program.functions.front().regions.front();
    const ValueType f32{ElementType::F32, {}}, i1{ElementType::I1, {}};
    body.instructions = {
        Instruction{40, {Primitive::Convert, 1, 0}, {22}, {{41, f32}}, {}, {}, NoAttributes{}},
        Instruction{42, {Primitive::Less, 1, 0}, {21, 23}, {{43, i1}}, {}, {}, NoAttributes{}},
        Instruction{44, {Primitive::Select, 1, 0}, {43, 23, 21}, {{45, f32}}, {}, {}, NoAttributes{}},
        Instruction{46, {Primitive::Sqrt, 1, 0}, {45}, {{47, f32}}, {}, {}, NoAttributes{}},
        Instruction{48, {Primitive::Tanh, 1, 0}, {47}, {{49, f32}}, {}, {}, NoAttributes{}},
        Instruction{50, {Primitive::Equal, 1, 0}, {47, 47}, {{51, i1}}, {}, {}, NoAttributes{}},
        Instruction{52, {Primitive::Select, 1, 0}, {51, 49, 41}, {{53, f32}}, {}, {}, NoAttributes{}}};
    body.yields = {53};
    const auto verified = verify_and_canonicalize_program(program);
    CHECK(std::holds_alternative<VerifiedProgram>(verified));
    if (!std::holds_alternative<VerifiedProgram>(verified)) return;
    std::vector<uint64_t> table(12, std::bit_cast<uint32_t>(-1.0f));
    table[0] = std::bit_cast<uint32_t>(1.0f);
    table[1] = std::bit_cast<uint32_t>(4.0f);
    const std::array<ReferenceInput, 2> inputs = {
        ReferenceInput{10, {ElementType::F32, {4, 3}, std::move(table)}},
        ReferenceInput{11, {ElementType::U32, {}, {0}}}};
    ReferenceState state;
    const auto result = execute_reference_program(std::get<VerifiedProgram>(verified), state, inputs);
    CHECK(std::holds_alternative<ReferenceResult>(result));
    if (const auto* output = std::get_if<ReferenceResult>(&result))
        CHECK((output->exports.front().bits == std::vector<uint64_t>{
            std::bit_cast<uint32_t>(std::tanh(1.0f)),
            std::bit_cast<uint32_t>(std::tanh(2.0f)), 0}));
}

void test_typed_scalar_reference() {
    const auto run = [](Primitive code, std::vector<ReferenceValue> values,
                        ElementType output_type) -> std::variant<ReferenceResult, CompatibilityReport> {
        Region root;
        root.id = 1;
        std::vector<uint32_t> operands;
        std::vector<ReferenceInput> inputs;
        for (size_t i = 0; i < values.size(); ++i) {
            const uint32_t id = static_cast<uint32_t>(10 + i);
            root.arguments.push_back({id, {values[i].type, {}}});
            operands.push_back(id);
            inputs.push_back({id, std::move(values[i])});
        }
        const ValueType output{output_type, {}};
        root.instructions = {Instruction{20, {code, 1, 0}, std::move(operands),
            {{30, output}}, {}, {}, NoAttributes{}}};
        root.yields = {30};
        Program program;
        program.functions = {Function{9, 1, {std::move(root)}, {output}}};
        program.exports = {{9, 0, output}};
        const auto verified = verify_and_canonicalize_program(program);
        CHECK(std::holds_alternative<VerifiedProgram>(verified));
        if (const auto* error = std::get_if<CompatibilityReport>(&verified)) return *error;
        ReferenceState state;
        return execute_reference_program(std::get<VerifiedProgram>(verified), state, inputs);
    };
    const auto check = [&](Primitive code, std::vector<ReferenceValue> inputs,
                           ElementType type, uint64_t bits) {
        const auto result = run(code, std::move(inputs), type);
        CHECK(std::holds_alternative<ReferenceResult>(result));
        if (const auto* value = std::get_if<ReferenceResult>(&result))
            CHECK(value->exports.front().bits == std::vector<uint64_t>{bits});
    };
    check(Primitive::Less, {{ElementType::I32, {}, {UINT32_MAX}},
                           {ElementType::I32, {}, {0}}}, ElementType::I1, 1);
    check(Primitive::Less, {{ElementType::U64, {}, {UINT64_MAX}},
                           {ElementType::U64, {}, {0}}}, ElementType::I1, 0);
    check(Primitive::Equal, {{ElementType::F32, {}, {0x80000000}},
                            {ElementType::F32, {}, {0}}}, ElementType::I1, 1);
    for (uint64_t condition : {0u, 1u})
        check(Primitive::Select, {{ElementType::I1, {}, {condition}},
              {ElementType::U64, {}, {UINT64_MAX}}, {ElementType::U64, {}, {42}}},
              ElementType::U64, condition ? UINT64_MAX : 42);
    for (const auto& pair : std::vector<std::pair<uint32_t, uint32_t>>{
             {0, 0}, {1, 0x3f800000}, {16777217, 0x4b800000},
             {16777219, 0x4b800002}, {UINT32_MAX, 0x4f800000}})
        check(Primitive::Convert, {{ElementType::U32, {}, {pair.first}}},
              ElementType::F32, pair.second);
    check(Primitive::Sqrt, {{ElementType::F32, {}, {std::bit_cast<uint32_t>(2.0f)}}},
          ElementType::F32, std::bit_cast<uint32_t>(std::sqrt(2.0f)));
    check(Primitive::Tanh, {{ElementType::F32, {}, {std::bit_cast<uint32_t>(-1.25f)}}},
          ElementType::F32, std::bit_cast<uint32_t>(std::tanh(-1.25f)));
    CHECK(std::holds_alternative<CompatibilityReport>(run(Primitive::Sqrt,
        {{ElementType::F32, {}, {std::bit_cast<uint32_t>(-1.0f)}}}, ElementType::F32)));
    CHECK(std::holds_alternative<CompatibilityReport>(run(Primitive::Less,
        {{ElementType::F32, {}, {0x7fc00000}}, {ElementType::F32, {}, {0}}}, ElementType::I1)));
    CHECK(std::holds_alternative<CompatibilityReport>(run(Primitive::Select,
        {{ElementType::I1, {}, {2}}, {ElementType::U32, {}, {1}},
         {ElementType::U32, {}, {0}}}, ElementType::U32)));
}

} // namespace

int main() {
    test_coordinates_and_require_reference();
    test_arithmetic_policy_reference();
    test_source_element_reference();
    test_typed_structured_body_reference();
    test_typed_scalar_reference();
    test_generic_f32_algebra_reference();
    test_data_dependent_tensor_index_reference();
    test_bound_physical_resource_feeds_semantic_input();
    test_reference_execution_and_transaction();
    test_structured_tensor_reference();
    return test_summary("test_program_reference");
}
