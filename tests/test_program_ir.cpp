#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "program_ir.h"
#include "test_util.h"

using namespace Laplace;

namespace {

template <class T>
concept HasOpaqueTarget = requires(T value) { value.opaque_target; };

static_assert(!std::is_constructible_v<PrimitiveAttributes, std::string>);
static_assert(!std::is_constructible_v<PrimitiveAttributes, std::string_view>);
static_assert(!std::is_constructible_v<PrimitiveAttributes, const char*>);
static_assert(!std::is_constructible_v<PrimitiveAttributes, std::vector<uint8_t>>);
static_assert(!std::is_constructible_v<PrimitiveAttributes, void (*)()>);
static_assert(!HasOpaqueTarget<Instruction>);

DimensionExpr constant_dimension(uint64_t value) {
    return {DimensionExpression::Constant, value, {}};
}

DimensionExpr parameter_dimension(uint32_t id) {
    return {DimensionExpression::Parameter, id, {}};
}

ValueType scalar(ElementType type = ElementType::F32) {
    return {type, {}};
}

ValueType tensor(std::initializer_list<uint64_t> dimensions,
                 ElementType type = ElementType::F32) {
    ValueType result;
    result.element_type = type;
    for (uint64_t dimension : dimensions) {
        result.dimensions.push_back(constant_dimension(dimension));
    }
    return result;
}

TypedValue value(uint32_t id, ValueType type = scalar()) {
    return {id, std::move(type)};
}

Instruction instruction(uint32_t id, Primitive primitive,
                        std::vector<uint32_t> inputs,
                        std::vector<TypedValue> outputs,
                        PrimitiveAttributes attributes = NoAttributes{}) {
    Instruction result;
    result.id = id;
    result.primitive = {primitive, 1, 0};
    result.inputs = std::move(inputs);
    result.outputs = std::move(outputs);
    result.attributes = std::move(attributes);
    return result;
}

Program one_function_program(uint32_t function_id, uint32_t region_id,
                             std::vector<TypedValue> arguments,
                             std::vector<Instruction> instructions,
                             std::vector<uint32_t> yields,
                             std::vector<ValueType> result_types) {
    Region region;
    region.id = region_id;
    region.arguments = std::move(arguments);
    region.instructions = std::move(instructions);
    region.yields = std::move(yields);

    Function function;
    function.id = function_id;
    function.entry_region_id = region_id;
    function.regions.push_back(std::move(region));
    function.result_types = std::move(result_types);

    Program program;
    program.functions.push_back(std::move(function));
    for (uint32_t index = 0; index < program.functions.front().result_types.size(); ++index) {
        program.exports.push_back(
            {function_id, index, program.functions.front().result_types[index]});
    }
    return program;
}

const CompatibilityReport* rejection(const ProgramVerificationResult& result) {
    return std::get_if<CompatibilityReport>(&result);
}

bool rejected_with(const Program& program, CompatibilityError code) {
    const auto result = verify_and_canonicalize_program(program);
    const CompatibilityReport* report = rejection(result);
    return report != nullptr && report->code == code;
}

Program minimal_scalar_program() {
    std::vector<Instruction> instructions;
    instructions.push_back(instruction(
        10, Primitive::Constant, {}, {value(100)}, ConstantAttributes{2}));
    instructions.push_back(instruction(
        20, Primitive::Constant, {}, {value(200)}, ConstantAttributes{3}));
    instructions.push_back(instruction(
        30, Primitive::Add, {100, 200}, {value(300)}));
    return one_function_program(7, 9, {}, std::move(instructions), {300}, {scalar()});
}

Program algebra_program(Primitive primitive, bool unary,
                        ElementType type = ElementType::F32) {
    const ValueType value_type = scalar(type);
    std::vector<TypedValue> arguments = {value(10, value_type)};
    std::vector<uint32_t> inputs = {10};
    if (!unary) {
        arguments.push_back(value(11, value_type));
        inputs.push_back(11);
    }
    return one_function_program(
        7, 9, std::move(arguments),
        {instruction(20, primitive, std::move(inputs),
                     {value(30, value_type)})},
        {30}, {value_type});
}

void test_generic_f32_algebra_contract() {
    constexpr std::array binary = {
        Primitive::Subtract, Primitive::Divide, Primitive::Maximum, Primitive::Pow};
    constexpr std::array unary = {
        Primitive::Negate, Primitive::Exp, Primitive::Log,
        Primitive::Rsqrt, Primitive::Sin, Primitive::Cos};

    std::vector<ProgramDigest> digests;
    for (Primitive primitive : binary) {
        auto verified = verify_and_canonicalize_program(
            algebra_program(primitive, false));
        CHECK(std::holds_alternative<VerifiedProgram>(verified));
        if (const auto* value = std::get_if<VerifiedProgram>(&verified)) {
            digests.push_back(program_digest(*value));
            const auto wire = encode_program_wire(*value);
            CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
            if (const auto* bytes = std::get_if<std::vector<uint8_t>>(&wire))
                CHECK(std::holds_alternative<VerifiedProgram>(
                    decode_program_wire(*bytes)));
        }
        CHECK(rejected_with(algebra_program(primitive, false,
                                            ElementType::U32),
                            CompatibilityError::IR_SHAPE_MISMATCH));
    }
    for (Primitive primitive : unary) {
        auto verified = verify_and_canonicalize_program(
            algebra_program(primitive, true));
        CHECK(std::holds_alternative<VerifiedProgram>(verified));
        if (const auto* value = std::get_if<VerifiedProgram>(&verified)) {
            digests.push_back(program_digest(*value));
            const auto wire = encode_program_wire(*value);
            CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
            if (const auto* bytes = std::get_if<std::vector<uint8_t>>(&wire))
                CHECK(std::holds_alternative<VerifiedProgram>(
                    decode_program_wire(*bytes)));
        }
        CHECK(rejected_with(algebra_program(primitive, true,
                                            ElementType::U32),
                            CompatibilityError::IR_SHAPE_MISMATCH));
    }
    std::sort(digests.begin(), digests.end(),
              [](const ProgramDigest& left, const ProgramDigest& right) {
                  return left.bytes < right.bytes;
              });
    CHECK(std::adjacent_find(digests.begin(), digests.end()) == digests.end());

    Program wrong_arity = algebra_program(Primitive::Exp, true);
    auto& item = wrong_arity.functions.front().regions.front().instructions.front();
    item.inputs.push_back(10);
    CHECK(rejected_with(wrong_arity,
                        CompatibilityError::IR_CONSTRAINT_FAILED));

    Program attributed = algebra_program(Primitive::Maximum, false);
    attributed.functions.front().regions.front().instructions.front().attributes =
        ConstantAttributes{0};
    CHECK(rejected_with(attributed,
                        CompatibilityError::IR_CONSTRAINT_FAILED));
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

TensorIndexExpr source_scalar_index(uint32_t source) {
    return {TensorIndexExpression::SourceScalar,
            static_cast<int64_t>(source), {}};
}

Program dynamic_gather_program(ElementType index_type = ElementType::U32) {
    const ValueType table = tensor({4, 3});
    const ValueType index = scalar(index_type);
    const ValueType output = tensor({3});
    Region body;
    body.id = 20;
    body.arguments = {value(21), value(22, index), value(23)};
    body.instructions = {
        instruction(24, Primitive::Add, {21, 23}, {value(25)})};
    body.yields = {25};

    StructuredTensorAttributes attributes;
    attributes.source_count = 2;
    attributes.iteration_dimensions = {constant_dimension(3)};
    attributes.iterator_kinds = {TensorIteratorKind::Parallel};
    TensorIndexMap table_map;
    table_map.results = {source_scalar_index(1), iterator_index(0)};
    attributes.indexing_maps = {
        std::move(table_map), TensorIndexMap{}, index_map({0})};

    Instruction zero = instruction(
        10, Primitive::Constant, {}, {value(12, output)},
        ConstantAttributes{0});
    Instruction gather = instruction(
        30, Primitive::StructuredTensor, {10, 11, 12},
        {value(13, output)}, std::move(attributes));
    gather.regions = {20};
    Region root;
    root.id = 1;
    root.arguments = {value(10, table), value(11, index)};
    root.instructions = {std::move(zero), std::move(gather)};
    root.yields = {13};
    Function function{9, 1, {std::move(body), std::move(root)}, {output}};
    Program program;
    program.minor = 1;
    program.functions = {std::move(function)};
    program.exports = {{9, 0, output}};
    return program;
}

void test_data_dependent_tensor_index() {
    auto verified = verify_and_canonicalize_program(dynamic_gather_program());
    CHECK(std::holds_alternative<VerifiedProgram>(verified));
    if (const auto* value = std::get_if<VerifiedProgram>(&verified)) {
        const auto wire = encode_program_wire(*value);
        CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
        if (const auto* bytes = std::get_if<std::vector<uint8_t>>(&wire)) {
            const auto decoded = decode_program_wire(*bytes);
            CHECK(std::holds_alternative<VerifiedProgram>(decoded));
            if (const auto* roundtrip = std::get_if<VerifiedProgram>(&decoded))
                CHECK(program_digest(*roundtrip) == program_digest(*value));
        }
    }

    CHECK(rejected_with(dynamic_gather_program(ElementType::F32),
                        CompatibilityError::IR_SHAPE_MISMATCH));

    Program missing_source = dynamic_gather_program();
    auto& missing_map = std::get<StructuredTensorAttributes>(
        missing_source.functions.front().regions.back().instructions.back()
            .attributes).indexing_maps.front();
    missing_map.results.front().value = 2;
    CHECK(rejected_with(missing_source,
                        CompatibilityError::IR_REFERENCE_INVALID));

    Program malformed = dynamic_gather_program();
    auto& malformed_index = std::get<StructuredTensorAttributes>(
        malformed.functions.front().regions.back().instructions.back()
            .attributes).indexing_maps.front().results.front();
    malformed_index.operands.push_back(iterator_index(0));
    CHECK(rejected_with(malformed,
                        CompatibilityError::IR_CONSTRAINT_FAILED));
}

Program structured_matmul_program(uint32_t rename = 0) {
    const ValueType left = tensor({2, 3});
    const ValueType right = tensor({3, 2});
    const ValueType output = tensor({2, 2});

    Region body;
    body.id = 20 + rename;
    body.arguments = {value(21 + rename), value(22 + rename), value(23 + rename)};
    body.instructions = {
        instruction(24 + rename, Primitive::Multiply,
                    {21 + rename, 22 + rename}, {value(25 + rename)}),
        instruction(26 + rename, Primitive::Add,
                    {23 + rename, 25 + rename}, {value(27 + rename)}),
    };
    body.yields = {27 + rename};

    StructuredTensorAttributes attributes;
    attributes.source_count = 2;
    attributes.iteration_dimensions = {
        constant_dimension(2), constant_dimension(2), constant_dimension(3)};
    attributes.iterator_kinds = {
        TensorIteratorKind::Parallel,
        TensorIteratorKind::Parallel,
        TensorIteratorKind::Reduction,
    };
    attributes.indexing_maps = {
        index_map({0, 2}), index_map({2, 1}), index_map({0, 1})};

    Instruction zero = instruction(
        10 + rename, Primitive::Constant, {}, {value(12 + rename, output)},
        ConstantAttributes{0});
    Instruction contraction = instruction(
        30 + rename, Primitive::StructuredTensor,
        {10 + rename, 11 + rename, 12 + rename}, {value(13 + rename, output)},
        std::move(attributes));
    contraction.regions = {body.id};

    Region root;
    root.id = 1 + rename;
    root.arguments = {value(10 + rename, left), value(11 + rename, right)};
    root.instructions = {std::move(zero), std::move(contraction)};
    root.yields = {13 + rename};

    Function function;
    function.id = 9 + rename;
    function.entry_region_id = root.id;
    function.regions = {std::move(body), std::move(root)};
    function.result_types = {output};

    Program program;
    program.minor = 1;
    program.functions = {std::move(function)};
    program.exports = {{9 + rename, 0, output}};
    return program;
}

void test_structured_tensor_contract() {
    const auto verified = verify_and_canonicalize_program(structured_matmul_program());
    CHECK(std::holds_alternative<VerifiedProgram>(verified));
    if (!std::holds_alternative<VerifiedProgram>(verified)) return;

    const ProgramDigest expected =
        program_digest(std::get<VerifiedProgram>(verified));
    const auto renamed = verify_and_canonicalize_program(
        structured_matmul_program(1000));
    CHECK(std::holds_alternative<VerifiedProgram>(renamed));
    if (const auto* value = std::get_if<VerifiedProgram>(&renamed))
        CHECK(program_digest(*value) == expected);
    const auto wire = encode_program_wire(std::get<VerifiedProgram>(verified));
    CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
    if (!std::holds_alternative<std::vector<uint8_t>>(wire)) return;
    const auto roundtrip =
        decode_program_wire(std::get<std::vector<uint8_t>>(wire));
    CHECK(std::holds_alternative<VerifiedProgram>(roundtrip));
    if (const auto* decoded = std::get_if<VerifiedProgram>(&roundtrip))
        CHECK(program_digest(*decoded) == expected);

    Program old_version = structured_matmul_program();
    old_version.minor = 0;
    Region& old_root = old_version.functions.front().regions.back();
    old_root.arguments.push_back(value(12, tensor({2, 2})));
    old_root.instructions.erase(old_root.instructions.begin());
    CHECK(rejected_with(old_version, CompatibilityError::IR_VERSION_UNSUPPORTED));

    Program reduction_destination = structured_matmul_program();
    auto& reduction_attributes = std::get<StructuredTensorAttributes>(
        reduction_destination.functions.front().regions.back()
            .instructions.back().attributes);
    reduction_attributes.indexing_maps.back() = index_map({0, 2});
    CHECK(rejected_with(reduction_destination,
                        CompatibilityError::IR_SHAPE_MISMATCH));

    Program missing_parallel = structured_matmul_program();
    auto& missing_attributes = std::get<StructuredTensorAttributes>(
        missing_parallel.functions.front().regions.back()
            .instructions.back().attributes);
    missing_attributes.indexing_maps.back() = index_map({0, 0});
    CHECK(rejected_with(missing_parallel,
                        CompatibilityError::IR_SHAPE_MISMATCH));

    Program invalid_body = structured_matmul_program();
    invalid_body.functions.front().regions.front().arguments.front().type =
        tensor({1});
    CHECK(rejected_with(invalid_body,
                        CompatibilityError::IR_SHAPE_MISMATCH));

    Program missing_map = structured_matmul_program();
    std::get<StructuredTensorAttributes>(
        missing_map.functions.front().regions.back().instructions.back().attributes)
        .indexing_maps.pop_back();
    CHECK(rejected_with(missing_map,
                        CompatibilityError::IR_CONSTRAINT_FAILED));

    Program wrong_map_rank = structured_matmul_program();
    std::get<StructuredTensorAttributes>(
        wrong_map_rank.functions.front().regions.back()
            .instructions.back().attributes)
        .indexing_maps.front().results.pop_back();
    CHECK(rejected_with(wrong_map_rank,
                        CompatibilityError::IR_SHAPE_MISMATCH));

    Program invalid_iterator_kind = structured_matmul_program();
    std::get<StructuredTensorAttributes>(
        invalid_iterator_kind.functions.front().regions.back()
            .instructions.back().attributes)
        .iterator_kinds.front() = static_cast<TensorIteratorKind>(255);
    CHECK(rejected_with(invalid_iterator_kind,
                        CompatibilityError::IR_CONSTRAINT_FAILED));

    Program invalid_bounds_mode = structured_matmul_program();
    std::get<StructuredTensorAttributes>(
        invalid_bounds_mode.functions.front().regions.back()
            .instructions.back().attributes)
        .indexing_maps.front().bounds = static_cast<TensorBoundsMode>(255);
    CHECK(rejected_with(invalid_bounds_mode,
                        CompatibilityError::IR_CONSTRAINT_FAILED));

    Program zero_destination = structured_matmul_program();
    std::get<StructuredTensorAttributes>(
        zero_destination.functions.front().regions.back()
            .instructions.back().attributes)
        .indexing_maps.back().bounds = TensorBoundsMode::Zero;
    CHECK(rejected_with(zero_destination,
                        CompatibilityError::IR_SHAPE_MISMATCH));

    Program zero_divisor = structured_matmul_program();
    auto& divisor_attributes = std::get<StructuredTensorAttributes>(
        zero_divisor.functions.front().regions.back().instructions.back().attributes);
    divisor_attributes.indexing_maps.front().results.front() = {
        TensorIndexExpression::FloorDivide, 0,
        {iterator_index(0),
         TensorIndexExpr{TensorIndexExpression::Constant, 0, {}}}};
    CHECK(rejected_with(zero_divisor,
                        CompatibilityError::IR_CONSTRAINT_FAILED));

    Program invalid_iterator = structured_matmul_program();
    std::get<StructuredTensorAttributes>(
        invalid_iterator.functions.front().regions.back()
            .instructions.back().attributes)
        .indexing_maps.front().results.front() = iterator_index(99);
    CHECK(rejected_with(invalid_iterator,
                        CompatibilityError::IR_CONSTRAINT_FAILED));

    Program invalid_arity = structured_matmul_program();
    std::get<StructuredTensorAttributes>(
        invalid_arity.functions.front().regions.back()
            .instructions.back().attributes)
        .indexing_maps.front().results.front() = {
            TensorIndexExpression::Add, 0, {iterator_index(0)}};
    CHECK(rejected_with(invalid_arity,
                        CompatibilityError::IR_CONSTRAINT_FAILED));

    Program index_overflow = structured_matmul_program();
    std::get<StructuredTensorAttributes>(
        index_overflow.functions.front().regions.back()
            .instructions.back().attributes)
        .indexing_maps.front().results.front() = {
            TensorIndexExpression::Add, 0,
            {TensorIndexExpr{TensorIndexExpression::Constant, INT64_MAX, {}},
             TensorIndexExpr{TensorIndexExpression::Constant, 1, {}}}};
    CHECK(rejected_with(index_overflow,
                        CompatibilityError::IR_CONSTRAINT_FAILED));

    Program excessive_index_depth = structured_matmul_program();
    TensorIndexExpr deep = iterator_index(0);
    for (uint32_t depth = 0; depth < 34; ++depth) {
        deep = {TensorIndexExpression::Add, 0,
                {std::move(deep),
                 TensorIndexExpr{TensorIndexExpression::Constant, 0, {}}}};
    }
    std::get<StructuredTensorAttributes>(
        excessive_index_depth.functions.front().regions.back()
            .instructions.back().attributes)
        .indexing_maps.front().results.front() = std::move(deep);
    CHECK(rejected_with(excessive_index_depth,
                        CompatibilityError::IR_CONSTRAINT_FAILED));

    Program effectful_body = structured_matmul_program();
    effectful_body.functions.front().regions.front()
        .instructions.back().effect_predecessors = {24};
    CHECK(rejection(verify_and_canonicalize_program(effectful_body)) != nullptr);

    Program changed_bounds = structured_matmul_program();
    std::get<StructuredTensorAttributes>(
        changed_bounds.functions.front().regions.back().instructions.back().attributes)
        .indexing_maps.front().bounds = TensorBoundsMode::Zero;
    const auto changed = verify_and_canonicalize_program(std::move(changed_bounds));
    CHECK(std::holds_alternative<VerifiedProgram>(changed));
    if (const auto* value = std::get_if<VerifiedProgram>(&changed))
        CHECK(program_digest(*value) != expected);

    auto truncated = std::get<std::vector<uint8_t>>(wire);
    truncated.pop_back();
    CHECK(std::holds_alternative<CompatibilityReport>(
        decode_program_wire(truncated)));

    auto malformed_attributes = std::get<std::vector<uint8_t>>(wire);
    const std::array<uint8_t, 11> structured_marker = {
        30, 0, 0, 0,
        static_cast<uint8_t>(Primitive::StructuredTensor), 0,
        1, 0, 0, 0, 4};
    const auto marker = std::search(malformed_attributes.begin(),
                                    malformed_attributes.end(),
                                    structured_marker.begin(),
                                    structured_marker.end());
    CHECK(marker != malformed_attributes.end());
    if (marker != malformed_attributes.end()) {
        malformed_attributes[static_cast<size_t>(
            std::distance(malformed_attributes.begin(), marker)) + 10] = 255;
        CHECK(std::holds_alternative<CompatibilityReport>(
            decode_program_wire(malformed_attributes)));
    }
}

void test_minimal_programs() {
    const auto scalar_result = verify_and_canonicalize_program(minimal_scalar_program());
    CHECK(std::holds_alternative<VerifiedProgram>(scalar_result));

    const ValueType matrix = tensor({3, 5});
    Program tensor_program = one_function_program(
        17, 23, {value(40, matrix), value(80, matrix)},
        {instruction(99, Primitive::Add, {40, 80}, {value(120, matrix)})},
        {120}, {matrix});
    const auto tensor_result = verify_and_canonicalize_program(std::move(tensor_program));
    CHECK(std::holds_alternative<VerifiedProgram>(tensor_result));
}

void test_unseen_repeated_graph_with_multiple_outputs() {
    const ValueType matrix = tensor({2, 7});
    std::vector<Instruction> instructions;
    instructions.push_back(instruction(
        1, Primitive::Add, {10, 20}, {value(30, matrix)}));
    instructions.push_back(instruction(
        2, Primitive::Multiply, {30, 20}, {value(40, matrix)}));
    instructions.push_back(instruction(
        3, Primitive::Add, {40, 30}, {value(50, matrix)}));
    instructions.push_back(instruction(
        4, Primitive::Multiply, {50, 40}, {value(60, matrix)}));
    Program program = one_function_program(
        3, 5, {value(10, matrix), value(20, matrix)}, std::move(instructions),
        {50, 60}, {matrix, matrix});
    const auto result = verify_and_canonicalize_program(std::move(program));
    CHECK(std::holds_alternative<VerifiedProgram>(result));
}

void test_export_serialization_order_is_not_identity() {
    const ValueType matrix = tensor({2, 3});
    Program first = one_function_program(
        3, 5, {value(10, matrix), value(20, matrix)},
        {instruction(1, Primitive::Add, {10, 20}, {value(30, matrix)}),
         instruction(2, Primitive::Multiply, {10, 20}, {value(40, matrix)})},
        {30, 40}, {matrix, matrix});
    Program second = first;
    std::reverse(second.exports.begin(), second.exports.end());
    const auto left = verify_and_canonicalize_program(std::move(first));
    const auto right = verify_and_canonicalize_program(std::move(second));
    CHECK(std::holds_alternative<VerifiedProgram>(left));
    CHECK(std::holds_alternative<VerifiedProgram>(right));
    if (std::holds_alternative<VerifiedProgram>(left) &&
        std::holds_alternative<VerifiedProgram>(right)) {
        CHECK(program_digest(std::get<VerifiedProgram>(left)) ==
              program_digest(std::get<VerifiedProgram>(right)));
    }
}

Program bounded_state_program() {
    Region body;
    body.id = 90;
    body.arguments = {value(900, scalar(ElementType::U64)), value(901)};
    body.instructions.push_back(instruction(
        910, Primitive::Constant, {}, {value(902)}, ConstantAttributes{1}));
    body.instructions.push_back(instruction(
        920, Primitive::Add, {901, 902}, {value(903)}));
    body.yields = {903};

    Instruction read = instruction(
        10, Primitive::StateRead, {}, {value(100)}, StateAttributes{70});
    Instruction loop = instruction(
        20, Primitive::BoundedLoop, {100}, {value(200)},
        LoopAttributes{0, 4, 1});
    loop.regions = {90};
    Instruction write = instruction(
        30, Primitive::StateWrite, {200}, {}, StateAttributes{70});
    write.effect_predecessors = {10};

    Region root;
    root.id = 80;
    root.instructions = {std::move(read), std::move(loop), std::move(write)};
    root.yields = {200};

    Function function;
    function.id = 60;
    function.entry_region_id = 80;
    function.regions = {std::move(body), std::move(root)};
    function.result_types = {scalar()};

    Program program;
    program.state_references.push_back({70, scalar(), 11, true});
    program.functions.push_back(std::move(function));
    program.exports.push_back({60, 0, scalar()});
    return program;
}

void test_bounded_loop_and_state_edge() {
    const auto result = verify_and_canonicalize_program(bounded_state_program());
    CHECK(std::holds_alternative<VerifiedProgram>(result));
}

Program permuted_graph(bool reverse) {
    std::vector<Instruction> left;
    left.push_back(instruction(
        reverse ? 701 : 101, Primitive::Constant, {},
        {value(reverse ? 710 : 110)}, ConstantAttributes{2}));
    left.push_back(instruction(
        reverse ? 702 : 102, Primitive::Constant, {},
        {value(reverse ? 720 : 120)}, ConstantAttributes{3}));
    left.push_back(instruction(
        reverse ? 703 : 103, Primitive::Add,
        {reverse ? 710u : 110u, reverse ? 720u : 120u},
        {value(reverse ? 730 : 130)}));

    std::vector<Instruction> right;
    right.push_back(instruction(
        reverse ? 801 : 201, Primitive::Constant, {},
        {value(reverse ? 810 : 210)}, ConstantAttributes{4}));
    right.push_back(instruction(
        reverse ? 802 : 202, Primitive::Constant, {},
        {value(reverse ? 820 : 220)}, ConstantAttributes{5}));
    right.push_back(instruction(
        reverse ? 803 : 203, Primitive::Add,
        {reverse ? 810u : 210u, reverse ? 820u : 220u},
        {value(reverse ? 830 : 230)}));

    std::vector<Instruction> instructions;
    if (reverse) {
        instructions.insert(instructions.end(), right.begin(), right.end());
        instructions.insert(instructions.end(), left.begin(), left.end());
    } else {
        instructions.insert(instructions.end(), left.begin(), left.end());
        instructions.insert(instructions.end(), right.begin(), right.end());
    }
    instructions.push_back(instruction(
        reverse ? 900 : 300, Primitive::Multiply,
        {reverse ? 730u : 130u, reverse ? 830u : 230u},
        {value(reverse ? 910 : 310)}));
    return one_function_program(reverse ? 51 : 1, reverse ? 61 : 2, {},
                                std::move(instructions),
                                {reverse ? 910u : 310u}, {scalar()});
}

void test_canonical_digest_ignores_local_ids_and_independent_order() {
    const auto first = verify_and_canonicalize_program(permuted_graph(false));
    const auto second = verify_and_canonicalize_program(permuted_graph(true));
    CHECK(std::holds_alternative<VerifiedProgram>(first));
    CHECK(std::holds_alternative<VerifiedProgram>(second));
    if (const auto* a = std::get_if<VerifiedProgram>(&first)) {
        if (const auto* b = std::get_if<VerifiedProgram>(&second)) {
            CHECK(program_digest(*a) == program_digest(*b));
        }
    }
}

Program parameter_order_program(bool reverse_declarations) {
    ValueType shaped = scalar();
    shaped.dimensions = {parameter_dimension(11), parameter_dimension(22)};
    Program program = one_function_program(
        1, 2, {value(10, shaped), value(20, shaped)},
        {instruction(3, Primitive::Add, {10, 20}, {value(30, shaped)})},
        {30}, {shaped});
    if (reverse_declarations) {
        program.dimension_parameters = {{22, 3, 9}, {11, 1, 5}};
    } else {
        program.dimension_parameters = {{11, 1, 5}, {22, 3, 9}};
    }
    return program;
}

Program state_order_program(bool reverse_declarations) {
    Region region;
    region.id = 2;
    region.arguments = {value(10), value(20)};
    region.instructions = {
        instruction(30, Primitive::StateWrite, {10}, {}, StateAttributes{100}),
        instruction(40, Primitive::StateWrite, {20}, {}, StateAttributes{200}),
    };
    region.instructions.back().effect_predecessors = {30};
    region.yields = {10};

    Function function;
    function.id = 1;
    function.entry_region_id = 2;
    function.regions = {std::move(region)};
    function.result_types = {scalar()};

    Program program;
    if (reverse_declarations) {
        program.state_references = {
            {200, scalar(), UINT32_MAX, true},
            {100, scalar(), UINT32_MAX, true},
        };
    } else {
        program.state_references = {
            {100, scalar(), UINT32_MAX, true},
            {200, scalar(), UINT32_MAX, true},
        };
    }
    program.functions = {std::move(function)};
    program.exports = {{1, 0, scalar()}};
    return program;
}

void test_declaration_storage_order_is_not_semantic() {
    const auto dimensions_a =
        verify_and_canonicalize_program(parameter_order_program(false));
    const auto dimensions_b =
        verify_and_canonicalize_program(parameter_order_program(true));
    CHECK(std::holds_alternative<VerifiedProgram>(dimensions_a));
    CHECK(std::holds_alternative<VerifiedProgram>(dimensions_b));
    if (const auto* a = std::get_if<VerifiedProgram>(&dimensions_a)) {
        if (const auto* b = std::get_if<VerifiedProgram>(&dimensions_b)) {
            CHECK(program_digest(*a) == program_digest(*b));
        }
    }

    const auto states_a = verify_and_canonicalize_program(state_order_program(false));
    const auto states_b = verify_and_canonicalize_program(state_order_program(true));
    CHECK(std::holds_alternative<VerifiedProgram>(states_a));
    CHECK(std::holds_alternative<VerifiedProgram>(states_b));
    if (const auto* a = std::get_if<VerifiedProgram>(&states_a)) {
        if (const auto* b = std::get_if<VerifiedProgram>(&states_b)) {
            CHECK(program_digest(*a) == program_digest(*b));
        }
    }
}

void test_state_relation_is_semantic() {
    Program distinct = state_order_program(false);
    distinct.state_references[0].alias_group = 5;
    distinct.state_references[1].alias_group = 6;

    Program shared = state_order_program(false);
    shared.state_references[0].alias_group = 5;
    shared.state_references[1].alias_group = 5;

    const auto distinct_result = verify_and_canonicalize_program(std::move(distinct));
    const auto shared_result = verify_and_canonicalize_program(std::move(shared));
    CHECK(std::holds_alternative<VerifiedProgram>(distinct_result));
    CHECK(std::holds_alternative<VerifiedProgram>(shared_result));
    if (const auto* a = std::get_if<VerifiedProgram>(&distinct_result)) {
        if (const auto* b = std::get_if<VerifiedProgram>(&shared_result)) {
            CHECK(program_digest(*a) != program_digest(*b));
        }
    }

}

Program effect_predecessor_order_program(bool reverse) {
    Region region;
    region.id = 2;
    region.arguments = {value(1), value(2), value(3)};
    region.instructions = {
        instruction(10, Primitive::StateWrite, {1}, {}, StateAttributes{100}),
        instruction(20, Primitive::StateWrite, {2}, {}, StateAttributes{200}),
        instruction(30, Primitive::StateWrite, {3}, {}, StateAttributes{300}),
    };
    region.instructions.back().effect_predecessors =
        reverse ? std::vector<uint32_t>{20, 10}
                : std::vector<uint32_t>{10, 20};
    region.yields = {1};

    Function function;
    function.id = 1;
    function.entry_region_id = 2;
    function.regions = {std::move(region)};
    function.result_types = {scalar()};

    Program program;
    program.state_references = {
        {100, scalar(), UINT32_MAX, true},
        {200, scalar(), UINT32_MAX, true},
        {300, scalar(), UINT32_MAX, true},
    };
    program.functions = {std::move(function)};
    program.exports = {{1, 0, scalar()}};
    return program;
}

Program loop_effect_placement_program(bool inside_loop) {
    Region body;
    body.id = 90;
    body.arguments = {value(900, scalar(ElementType::U64)), value(901)};
    if (inside_loop) {
        body.instructions = {
            instruction(910, Primitive::Constant, {}, {value(902)},
                        ConstantAttributes{5}),
            instruction(920, Primitive::StateWrite, {902}, {},
                        StateAttributes{70}),
        };
    }
    body.yields = {901};

    Instruction loop = instruction(20, Primitive::BoundedLoop, {100}, {value(200)},
                                   LoopAttributes{0, 4, 1});
    loop.regions = {90};
    Region root;
    root.id = 80;
    root.arguments = {value(100)};
    root.instructions = {std::move(loop)};
    if (!inside_loop) {
        root.instructions.push_back(instruction(
            910, Primitive::Constant, {}, {value(902)}, ConstantAttributes{5}));
        root.instructions.push_back(instruction(
            920, Primitive::StateWrite, {902}, {}, StateAttributes{70}));
    }
    root.yields = {200};

    Function function;
    function.id = 60;
    function.entry_region_id = 80;
    function.regions = {std::move(body), std::move(root)};
    function.result_types = {scalar()};

    Program program;
    program.state_references = {{70, scalar(), UINT32_MAX, true}};
    program.functions = {std::move(function)};
    program.exports = {{60, 0, scalar()}};
    return program;
}

void test_effect_sets_and_control_containment() {
    const auto effects_a =
        verify_and_canonicalize_program(effect_predecessor_order_program(false));
    const auto effects_b =
        verify_and_canonicalize_program(effect_predecessor_order_program(true));
    CHECK(std::holds_alternative<VerifiedProgram>(effects_a));
    CHECK(std::holds_alternative<VerifiedProgram>(effects_b));
    if (const auto* a = std::get_if<VerifiedProgram>(&effects_a)) {
        if (const auto* b = std::get_if<VerifiedProgram>(&effects_b)) {
            CHECK(program_digest(*a) == program_digest(*b));
        }
    }

    Program tagged_aliases = effect_predecessor_order_program(false);
    tagged_aliases.state_references[1].alias_group = 0x80000000u;
    CHECK(std::holds_alternative<VerifiedProgram>(
        verify_and_canonicalize_program(std::move(tagged_aliases))));

    const auto inside =
        verify_and_canonicalize_program(loop_effect_placement_program(true));
    const auto outside =
        verify_and_canonicalize_program(loop_effect_placement_program(false));
    CHECK(std::holds_alternative<VerifiedProgram>(inside));
    CHECK(std::holds_alternative<VerifiedProgram>(outside));
    if (const auto* a = std::get_if<VerifiedProgram>(&inside)) {
        if (const auto* b = std::get_if<VerifiedProgram>(&outside)) {
            CHECK(program_digest(*a) != program_digest(*b));
        }
    }

    Program ambiguous = effect_predecessor_order_program(false);
    ambiguous.functions.front().regions.front().instructions[1].inputs = {1};
    CHECK(rejected_with(ambiguous, CompatibilityError::IR_STATE_INVALID));

    Program explicitly_chained = effect_predecessor_order_program(false);
    auto& effects = explicitly_chained.functions.front().regions.front().instructions;
    effects[1].inputs = {1};
    effects[1].effect_predecessors = {10};
    effects[2].effect_predecessors = {20};
    CHECK(std::holds_alternative<VerifiedProgram>(
        verify_and_canonicalize_program(std::move(explicitly_chained))));
}

Program constant_program(ElementType type, uint64_t bits) {
    const ValueType result_type = scalar(type);
    return one_function_program(
        1, 2, {},
        {instruction(3, Primitive::Constant, {}, {value(4, result_type)},
                     ConstantAttributes{bits})},
        {4}, {result_type});
}

Program dependency_chain_program(uint32_t instruction_count) {
    Region region;
    region.id = 1;
    region.arguments = {value(1, scalar(ElementType::U64))};
    region.instructions.reserve(instruction_count);
    uint32_t previous = 1;
    for (uint32_t index = 0; index < instruction_count; ++index) {
        const uint32_t output_id = index + 2;
        region.instructions.push_back(instruction(
            10000 + index, Primitive::Add, {previous, 1},
            {value(output_id, scalar(ElementType::U64))}));
        previous = output_id;
    }
    region.yields = {previous};

    Function function;
    function.id = 1;
    function.entry_region_id = 1;
    function.regions = {std::move(region)};
    function.result_types = {scalar(ElementType::U64)};

    Program program;
    program.functions = {std::move(function)};
    program.exports = {{1, 0, scalar(ElementType::U64)}};
    return program;
}

Program nested_loop_program(uint32_t instruction_count) {
    Function function;
    function.id = 1;
    function.entry_region_id = 1;
    function.regions.resize(instruction_count + 1);
    for (uint32_t depth = 0; depth <= instruction_count; ++depth) {
        Region& region = function.regions[depth];
        region.id = depth + 1;
        if (depth == 0) {
            region.arguments = {value(1, scalar(ElementType::U64))};
        } else {
            region.arguments = {
                value(100000 + depth * 2, scalar(ElementType::U64)),
                value(100001 + depth * 2, scalar(ElementType::U64)),
            };
        }
        const uint32_t carried = depth == 0 ? 1 : 100001 + depth * 2;
        if (depth == instruction_count) {
            region.yields = {carried};
            continue;
        }
        Instruction loop = instruction(
            10000 + depth, Primitive::BoundedLoop, {carried},
            {value(200000 + depth, scalar(ElementType::U64))},
            LoopAttributes{0, 1, 1});
        loop.regions = {depth + 2};
        region.instructions = {std::move(loop)};
        region.yields = {200000 + depth};
    }
    function.result_types = {scalar(ElementType::U64)};

    Program program;
    program.functions = {std::move(function)};
    program.exports = {{1, 0, scalar(ElementType::U64)}};
    return program;
}

Program state_effect_chain_program(uint32_t instruction_count) {
    Region region;
    region.id = 1;
    region.arguments = {value(1, scalar(ElementType::U64))};
    region.instructions.reserve(instruction_count);
    for (uint32_t index = 0; index < instruction_count; ++index) {
        Instruction write = instruction(10000 + index, Primitive::StateWrite,
                                        {1}, {}, StateAttributes{9});
        if (index != 0) write.effect_predecessors = {9999 + index};
        region.instructions.push_back(std::move(write));
    }
    region.yields = {1};

    Function function;
    function.id = 1;
    function.entry_region_id = 1;
    function.regions = {std::move(region)};
    function.result_types = {scalar(ElementType::U64)};

    Program program;
    program.state_references = {
        {9, scalar(ElementType::U64), 7, true},
    };
    program.functions = {std::move(function)};
    program.exports = {{1, 0, scalar(ElementType::U64)}};
    return program;
}

Program independent_state_reads(uint32_t count) {
    Program program;
    const auto type = scalar(ElementType::U64);
    program.state_references = {{9, type, UINT32_MAX, false}};
    Region region;
    region.id = 1;
    for (uint32_t i = 0; i < count; ++i) {
        region.instructions.push_back(instruction(i, Primitive::StateRead, {},
            {value(i, type)}, StateAttributes{9}));
        region.yields.push_back(i);
    }
    program.functions = {{1, 1, {std::move(region)},
                          std::vector<ValueType>(count, type)}};
    program.exports = {{1, 0, type}};
    return program;
}

void test_constant_storage_widths() {
    for (const auto [type, bits] : {
             std::pair{ElementType::I1, uint64_t{1}},
             std::pair{ElementType::F16, uint64_t{UINT16_MAX}},
             std::pair{ElementType::I32, uint64_t{UINT32_MAX}},
             std::pair{ElementType::U32, uint64_t{UINT32_MAX}},
             std::pair{ElementType::F32, uint64_t{UINT32_MAX}},
             std::pair{ElementType::U64, UINT64_MAX},
         }) {
        CHECK(std::holds_alternative<VerifiedProgram>(
            verify_and_canonicalize_program(constant_program(type, bits))));
    }
    for (const auto [type, bits] : {
             std::pair{ElementType::I1, uint64_t{2}},
             std::pair{ElementType::F16, uint64_t{1} << 16},
             std::pair{ElementType::I32, uint64_t{1} << 32},
             std::pair{ElementType::U32, uint64_t{1} << 32},
             std::pair{ElementType::F32, uint64_t{1} << 32},
         }) {
        CHECK(rejected_with(constant_program(type, bits),
                            CompatibilityError::IR_CONSTRAINT_FAILED));
    }
}

void test_loop_overflow_model() {
    Program wrapping_increment = bounded_state_program();
    auto& rejected = std::get<LoopAttributes>(
        wrapping_increment.functions.front().regions[1].instructions[1].attributes);
    rejected.lower = UINT64_MAX - 1;
    rejected.upper = UINT64_MAX;
    rejected.step = 2;
    CHECK(rejected_with(wrapping_increment, CompatibilityError::IR_CONSTRAINT_FAILED));

    Program exact_final_increment = bounded_state_program();
    auto& accepted = std::get<LoopAttributes>(
        exact_final_increment.functions.front().regions[1].instructions[1].attributes);
    accepted.lower = UINT64_MAX - 2;
    accepted.upper = UINT64_MAX;
    accepted.step = 2;
    CHECK(std::holds_alternative<VerifiedProgram>(
        verify_and_canonicalize_program(std::move(exact_final_increment))));
}

void test_dependency_depth_is_bounded() {
    CHECK(std::holds_alternative<VerifiedProgram>(
        verify_and_canonicalize_program(independent_state_reads(4096))));
    CHECK(rejected_with(independent_state_reads(4097),
                        CompatibilityError::IR_CONSTRAINT_FAILED));
    auto split_reads = independent_state_reads(2048);
    auto extra_reads = independent_state_reads(2049);
    extra_reads.functions.front().id = 2;
    extra_reads.exports.front().function_id = 2;
    split_reads.functions.push_back(std::move(extra_reads.functions.front()));
    split_reads.exports.push_back(extra_reads.exports.front());
    CHECK(rejected_with(split_reads, CompatibilityError::IR_CONSTRAINT_FAILED));

    CHECK(std::holds_alternative<VerifiedProgram>(
        verify_and_canonicalize_program(dependency_chain_program(4096))));
    CHECK(std::holds_alternative<VerifiedProgram>(
        verify_and_canonicalize_program(nested_loop_program(4096))));
    CHECK(std::holds_alternative<VerifiedProgram>(
        verify_and_canonicalize_program(state_effect_chain_program(4096))));
    CHECK(rejected_with(dependency_chain_program(4097),
                        CompatibilityError::IR_CONSTRAINT_FAILED));
    CHECK(rejected_with(nested_loop_program(4097),
                        CompatibilityError::IR_CONSTRAINT_FAILED));
    CHECK(rejected_with(state_effect_chain_program(4097),
                        CompatibilityError::IR_CONSTRAINT_FAILED));

    Program split_limit = dependency_chain_program(2048);
    Program split_second = dependency_chain_program(2048);
    split_second.functions.front().id = 2;
    split_second.exports.front().function_id = 2;
    split_limit.functions.push_back(std::move(split_second.functions.front()));
    split_limit.exports.push_back(std::move(split_second.exports.front()));
    CHECK(std::holds_alternative<VerifiedProgram>(
        verify_and_canonicalize_program(std::move(split_limit))));

    Program split_over = dependency_chain_program(2048);
    Program split_over_second = dependency_chain_program(2049);
    split_over_second.functions.front().id = 2;
    split_over_second.exports.front().function_id = 2;
    split_over.functions.push_back(std::move(split_over_second.functions.front()));
    split_over.exports.push_back(std::move(split_over_second.exports.front()));
    CHECK(std::holds_alternative<VerifiedProgram>(
        verify_and_canonicalize_program(std::move(split_over))));

    Program wide = dependency_chain_program(1024);
    for (uint32_t i = 1; i < 16; ++i) {
        auto branch = dependency_chain_program(1024);
        branch.functions.front().id = i + 1;
        branch.exports.front().function_id = i + 1;
        wide.functions.push_back(std::move(branch.functions.front()));
        wide.exports.push_back(std::move(branch.exports.front()));
    }
    const auto verified = verify_and_canonicalize_program(std::move(wide));
    CHECK(std::holds_alternative<VerifiedProgram>(verified));
}

uint64_t evaluate_scalar(const Program& program) {
    const Region& region = program.functions.front().regions.front();
    std::unordered_map<uint32_t, uint64_t> values;
    for (const Instruction& item : region.instructions) {
        switch (item.primitive.code) {
        case Primitive::Constant:
            values[item.outputs.front().id] =
                std::get<ConstantAttributes>(item.attributes).bits;
            break;
        case Primitive::Add:
            values[item.outputs.front().id] = values[item.inputs[0]] + values[item.inputs[1]];
            break;
        case Primitive::Multiply:
            values[item.outputs.front().id] = values[item.inputs[0]] * values[item.inputs[1]];
            break;
        default:
            break;
        }
    }
    return values.at(region.yields.front());
}

Program rewired_graph(bool use_second_constant) {
    Program program = minimal_scalar_program();
    Function& function = program.functions.front();
    Region& region = function.regions.front();
    Instruction& multiply = region.instructions.back();
    multiply.primitive.code = Primitive::Multiply;
    multiply.inputs = {100, use_second_constant ? 200u : 100u};
    region.yields = {300, 200};
    function.result_types = {scalar(), scalar()};
    program.exports.push_back({function.id, 1, scalar()});
    return program;
}

void test_edges_change_digest_and_result() {
    const Program first_program = rewired_graph(false);
    const Program second_program = rewired_graph(true);
    CHECK(evaluate_scalar(first_program) == 4);
    CHECK(evaluate_scalar(second_program) == 6);
    const auto first = verify_and_canonicalize_program(first_program);
    const auto second = verify_and_canonicalize_program(second_program);
    CHECK(std::holds_alternative<VerifiedProgram>(first));
    CHECK(std::holds_alternative<VerifiedProgram>(second));
    if (const auto* a = std::get_if<VerifiedProgram>(&first)) {
        if (const auto* b = std::get_if<VerifiedProgram>(&second)) {
            CHECK(program_digest(*a) != program_digest(*b));
        }
    }
}

Program sharing_program(bool shared) {
    std::vector<Instruction> instructions;
    instructions.push_back(instruction(
        1, Primitive::Constant, {}, {value(10)}, ConstantAttributes{2}));
    if (!shared) {
        instructions.push_back(instruction(
            2, Primitive::Constant, {}, {value(20)}, ConstantAttributes{2}));
    }
    instructions.push_back(instruction(
        3, Primitive::Add, {10, shared ? 10u : 20u}, {value(30)}));
    return one_function_program(1, 2, {}, std::move(instructions), {30}, {scalar()});
}

Program partial_export_program(bool shared_second_result) {
    Region region;
    region.id = 2;
    region.arguments = {value(10), value(20)};
    region.yields = {10, shared_second_result ? 10u : 20u};

    Function function;
    function.id = 1;
    function.entry_region_id = 2;
    function.regions = {std::move(region)};
    function.result_types = {scalar(), scalar()};

    Program program;
    program.functions = {std::move(function)};
    program.exports = {{1, 0, scalar()}};
    return program;
}

void test_sharing_is_part_of_canonical_identity() {
    const auto shared = verify_and_canonicalize_program(sharing_program(true));
    const auto duplicated = verify_and_canonicalize_program(sharing_program(false));
    CHECK(std::holds_alternative<VerifiedProgram>(shared));
    CHECK(std::holds_alternative<VerifiedProgram>(duplicated));
    if (const auto* a = std::get_if<VerifiedProgram>(&shared)) {
        if (const auto* b = std::get_if<VerifiedProgram>(&duplicated)) {
            CHECK(program_digest(*a) != program_digest(*b));
        }
    }

    const auto distinct_result =
        verify_and_canonicalize_program(partial_export_program(false));
    const auto shared_result =
        verify_and_canonicalize_program(partial_export_program(true));
    CHECK(std::holds_alternative<VerifiedProgram>(distinct_result));
    CHECK(std::holds_alternative<VerifiedProgram>(shared_result));
    if (const auto* a = std::get_if<VerifiedProgram>(&distinct_result)) {
        if (const auto* b = std::get_if<VerifiedProgram>(&shared_result)) {
            CHECK(program_digest(*a) != program_digest(*b));
        }
    }
}

void test_rejections() {
    Program use_before_definition = minimal_scalar_program();
    auto& use_before_instructions =
        use_before_definition.functions.front().regions.front().instructions;
    std::swap(use_before_instructions[1], use_before_instructions[2]);
    CHECK(rejected_with(use_before_definition, CompatibilityError::IR_REFERENCE_INVALID));

    Program invalid_dominance = bounded_state_program();
    Region& invalid_root = invalid_dominance.functions.front().regions[1];
    invalid_root.instructions.push_back(
        instruction(40, Primitive::Add, {903, 200}, {value(300)}));
    invalid_root.yields = {300};
    CHECK(rejected_with(invalid_dominance, CompatibilityError::IR_REFERENCE_INVALID));

    Program implicit_capture = bounded_state_program();
    Region& capture_body = implicit_capture.functions.front().regions.front();
    capture_body.instructions.back().inputs.front() = 100;
    CHECK(rejected_with(implicit_capture, CompatibilityError::IR_REFERENCE_INVALID));

    Program unresolved_dimension = minimal_scalar_program();
    ValueType unresolved = tensor({1});
    unresolved.dimensions = {parameter_dimension(77)};
    unresolved_dimension.functions.front().regions.front().instructions.front().outputs.front().type =
        unresolved;
    CHECK(rejected_with(unresolved_dimension, CompatibilityError::IR_SHAPE_MISMATCH));

    Region alias_region;
    alias_region.id = 4;
    alias_region.arguments = {value(1)};
    alias_region.instructions = {
        instruction(10, Primitive::StateWrite, {1}, {}, StateAttributes{10}),
        instruction(20, Primitive::StateWrite, {1}, {}, StateAttributes{20}),
    };
    Function alias_function;
    alias_function.id = 3;
    alias_function.entry_region_id = 4;
    alias_function.regions = {std::move(alias_region)};
    Program unsafe_alias;
    unsafe_alias.state_references = {
        {10, scalar(), 5, true},
        {20, scalar(), 5, true},
    };
    unsafe_alias.functions.push_back(std::move(alias_function));
    CHECK(rejected_with(unsafe_alias, CompatibilityError::IR_STATE_INVALID));

    Program unordered_effects = state_order_program(false);
    unordered_effects.functions.front().regions.front().instructions.back()
        .effect_predecessors.clear();
    CHECK(rejected_with(unordered_effects, CompatibilityError::IR_STATE_INVALID));

    Program unbounded = bounded_state_program();
    auto& loop_attributes = std::get<LoopAttributes>(
        unbounded.functions.front().regions[1].instructions[1].attributes);
    loop_attributes.upper = UINT64_MAX;
    CHECK(rejected_with(unbounded, CompatibilityError::IR_CONSTRAINT_FAILED));

    Program effect_cycle = bounded_state_program();
    Region& cycle_root = effect_cycle.functions.front().regions[1];
    cycle_root.instructions[0].effect_predecessors = {30};
    CHECK(rejected_with(effect_cycle, CompatibilityError::IR_STATE_INVALID));

    Program unknown_version = minimal_scalar_program();
    unknown_version.functions.front().regions.front().instructions.back().primitive.major = 2;
    CHECK(rejected_with(unknown_version, CompatibilityError::IR_VERSION_UNSUPPORTED));

    Program unknown_primitive = minimal_scalar_program();
    unknown_primitive.functions.front().regions.front().instructions.back().primitive.code =
        static_cast<Primitive>(UINT16_MAX);
    CHECK(rejected_with(unknown_primitive, CompatibilityError::IR_VERSION_UNSUPPORTED));

    Program invalid_element = minimal_scalar_program();
    invalid_element.functions.front().regions.front().instructions.front().outputs.front()
        .type.element_type = static_cast<ElementType>(UINT8_MAX);
    CHECK(rejected_with(invalid_element, CompatibilityError::IR_SHAPE_MISMATCH));

    Program dead_instruction = minimal_scalar_program();
    dead_instruction.functions.front().regions.front().instructions.push_back(instruction(
        40, Primitive::Constant, {}, {value(400)}, ConstantAttributes{9}));
    CHECK(rejected_with(dead_instruction, CompatibilityError::IR_REFERENCE_INVALID));

    Program dead_function = minimal_scalar_program();
    Function unused;
    unused.id = 50;
    unused.entry_region_id = 60;
    unused.regions.push_back({60, {}, {}, {}});
    dead_function.functions.push_back(std::move(unused));
    CHECK(rejected_with(dead_function, CompatibilityError::IR_REFERENCE_INVALID));

    Program unused_dimension = minimal_scalar_program();
    unused_dimension.dimension_parameters = {{50, 1, 4}};
    CHECK(rejected_with(unused_dimension, CompatibilityError::IR_SHAPE_MISMATCH));

    Program unused_state = minimal_scalar_program();
    unused_state.state_references = {{50, scalar(), UINT32_MAX, true}};
    CHECK(rejected_with(unused_state, CompatibilityError::IR_STATE_INVALID));
}

void test_structural_wire_roundtrip() {
    auto verified = verify_and_canonicalize_program(bounded_state_program());
    CHECK(std::holds_alternative<VerifiedProgram>(verified));
    if (!std::holds_alternative<VerifiedProgram>(verified)) return;
    const ProgramDigest expected =
        program_digest(std::get<VerifiedProgram>(verified));
    const auto wire = encode_program_wire(std::get<VerifiedProgram>(verified));
    CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
    if (!std::holds_alternative<std::vector<uint8_t>>(wire)) return;
    const auto& bytes = std::get<std::vector<uint8_t>>(wire);
    const auto decoded = decode_program_wire(bytes);
    CHECK(std::holds_alternative<VerifiedProgram>(decoded));
    if (const auto* roundtrip = std::get_if<VerifiedProgram>(&decoded))
        CHECK(program_digest(*roundtrip) == expected);

    auto trailing = bytes;
    trailing.push_back(0);
    CHECK(std::holds_alternative<CompatibilityReport>(
        decode_program_wire(trailing)));
    auto bad_magic = bytes;
    bad_magic.front() ^= 1;
    CHECK(std::holds_alternative<CompatibilityReport>(
        decode_program_wire(bad_magic)));
    auto bad_version = bytes;
    bad_version[8] = 2;
    CHECK(std::holds_alternative<CompatibilityReport>(
        decode_program_wire(bad_version)));

    Program reordered = minimal_scalar_program();
    std::swap(reordered.functions.front().regions.front().instructions[0],
              reordered.functions.front().regions.front().instructions[1]);
    auto reordered_verified = verify_and_canonicalize_program(std::move(reordered));
    CHECK(std::holds_alternative<VerifiedProgram>(reordered_verified));
    if (std::holds_alternative<VerifiedProgram>(reordered_verified)) {
        const auto reordered_wire = encode_program_wire(
            std::get<VerifiedProgram>(reordered_verified));
        CHECK(std::holds_alternative<std::vector<uint8_t>>(reordered_wire));
        if (std::holds_alternative<std::vector<uint8_t>>(reordered_wire)) {
            const auto reordered_decoded = decode_program_wire(
                std::get<std::vector<uint8_t>>(reordered_wire));
            CHECK(std::holds_alternative<VerifiedProgram>(reordered_decoded));
            if (const auto* roundtrip =
                    std::get_if<VerifiedProgram>(&reordered_decoded))
                CHECK(program_digest(*roundtrip) ==
                      program_digest(std::get<VerifiedProgram>(
                          reordered_verified)));
        }
    }
}

void test_coordinate_and_require_contract() {
    Program program = dynamic_gather_program();
    Region& body = program.functions.front().regions.front();
    body.instructions = {
        instruction(40, Primitive::TensorCoordinate, {}, {value(41, scalar(ElementType::U32))}, CoordinateAttributes{0}),
        instruction(42, Primitive::Convert, {41}, {value(43)})};
    body.yields = {43};
    const auto checked = verify_and_canonicalize_program(program);
    CHECK(std::holds_alternative<VerifiedProgram>(checked));
    if (const auto* verified = std::get_if<VerifiedProgram>(&checked)) {
        const auto wire = encode_program_wire(*verified);
        CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
        if (const auto* bytes = std::get_if<std::vector<uint8_t>>(&wire)) {
            const auto decoded = decode_program_wire(*bytes);
            CHECK(std::holds_alternative<VerifiedProgram>(decoded));
            if (const auto* restored = std::get_if<VerifiedProgram>(&decoded))
                CHECK(program_digest(*restored) == program_digest(*verified));
        }
    }
    Program large = program;
    auto& geometry = std::get<StructuredTensorAttributes>(
        large.functions.front().regions.back().instructions.back().attributes);
    geometry.iteration_dimensions.push_back(constant_dimension(uint64_t{1} << 32));
    geometry.iterator_kinds.push_back(TensorIteratorKind::Reduction);
    large.functions.front().regions.front().instructions.front().attributes = CoordinateAttributes{1};
    CHECK(std::holds_alternative<VerifiedProgram>(verify_and_canonicalize_program(large)));
    geometry.iteration_dimensions.back().value += 1;
    CHECK(rejected_with(large, CompatibilityError::IR_SHAPE_MISMATCH));
    geometry.iteration_dimensions.clear();
    geometry.iterator_kinds.clear();
    CHECK(rejected_with(large, CompatibilityError::IR_CONSTRAINT_FAILED));
    body.instructions.front().attributes = CoordinateAttributes{1};
    CHECK(rejected_with(program, CompatibilityError::IR_CONSTRAINT_FAILED));
    body.instructions.front().attributes = CoordinateAttributes{0};
    body.instructions.front().inputs = {22};
    CHECK(rejected_with(program, CompatibilityError::IR_CONSTRAINT_FAILED));
    body.instructions.front().inputs.clear();
    body.instructions.front().outputs.front().type = scalar();
    CHECK(rejected_with(program, CompatibilityError::IR_SHAPE_MISMATCH));
    Program root = one_function_program(0, 0, {},
        {instruction(1, Primitive::TensorCoordinate, {}, {value(2, scalar(ElementType::U64))}, CoordinateAttributes{0})},
        {2}, {scalar(ElementType::U64)});
    CHECK(rejected_with(root, CompatibilityError::IR_CONSTRAINT_FAILED));
    Program loop = bounded_state_program();
    loop.functions.front().regions.front().instructions.insert(
        loop.functions.front().regions.front().instructions.begin(),
        instruction(500, Primitive::TensorCoordinate, {}, {value(501, scalar(ElementType::U64))}, CoordinateAttributes{0}));
    CHECK(rejected_with(loop, CompatibilityError::IR_CONSTRAINT_FAILED));
    const auto i1 = scalar(ElementType::I1);
    Program require = one_function_program(0, 0, {value(1, i1)},
        {instruction(2, Primitive::Require, {1}, {value(3, i1)})}, {3}, {i1});
    CHECK(std::holds_alternative<VerifiedProgram>(verify_and_canonicalize_program(require)));
    require.functions.front().regions.front().instructions.front().attributes = ArithmeticAttributes{};
    CHECK(rejected_with(require, CompatibilityError::IR_CONSTRAINT_FAILED));
}

void test_arithmetic_policy_contract() {
    Program program = one_function_program(0, 0, {value(1)},
        {instruction(2, Primitive::Exp, {1}, {value(3)}, ArithmeticAttributes{false, true})},
        {3}, {scalar()});
    ProgramDigest previous{};
    for (unsigned flags = 0; flags < 4; ++flags) {
        program.functions.front().regions.front().instructions.front().attributes =
            ArithmeticAttributes{bool(flags & 1), bool(flags & 2)};
        const auto checked = verify_and_canonicalize_program(program);
        CHECK(std::holds_alternative<VerifiedProgram>(checked));
        if (const auto* verified = std::get_if<VerifiedProgram>(&checked)) {
            CHECK(program_digest(*verified) != previous);
            previous = program_digest(*verified);
            const auto wire = encode_program_wire(*verified);
            CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
            if (const auto* bytes = std::get_if<std::vector<uint8_t>>(&wire)) {
                const auto decoded = decode_program_wire(*bytes);
                CHECK(std::holds_alternative<VerifiedProgram>(decoded));
                if (const auto* restored = std::get_if<VerifiedProgram>(&decoded))
                    CHECK(program_digest(*restored) == program_digest(*verified));
            }
        }
    }
    auto& op = program.functions.front().regions.front().instructions.front();
    op.primitive.code = Primitive::RequireFinite;
    CHECK(rejected_with(program, CompatibilityError::IR_CONSTRAINT_FAILED));
    op.attributes = NoAttributes{};
    CHECK(std::holds_alternative<VerifiedProgram>(verify_and_canonicalize_program(program)));
    op.primitive.code = Primitive::Convert;
    op.attributes = ArithmeticAttributes{true, true};
    CHECK(rejected_with(program, CompatibilityError::IR_CONSTRAINT_FAILED));
    op.primitive.code = Primitive::Add;
    op.inputs = {1, 1};
    op.outputs.front().type = scalar(ElementType::U32);
    program.functions.front().regions.front().arguments.front().type = scalar(ElementType::U32);
    program.functions.front().result_types = {scalar(ElementType::U32)};
    program.exports.front().type = scalar(ElementType::U32);
    CHECK(rejected_with(program, CompatibilityError::IR_SHAPE_MISMATCH));
}

void test_source_element_index_contract() {
    Program program = dynamic_gather_program();
    Region& root = program.functions.front().regions.back();
    root.arguments[1].type = tensor({1, 3}, ElementType::U32);
    auto& maps = std::get<StructuredTensorAttributes>(root.instructions.back().attributes).indexing_maps;
    const TensorIndexExpr zero{TensorIndexExpression::Constant, 0, {}};
    maps[1].results = {zero, iterator_index(0)};
    const TensorIndexExpr load{TensorIndexExpression::SourceElement, 1,
                              {zero, iterator_index(0)}};
    maps[0].results[0] = load;
    const auto checked = verify_and_canonicalize_program(program);
    CHECK(std::holds_alternative<VerifiedProgram>(checked));
    if (const auto* verified = std::get_if<VerifiedProgram>(&checked)) {
        const auto wire = encode_program_wire(*verified);
        CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
        if (const auto* bytes = std::get_if<std::vector<uint8_t>>(&wire)) {
            const auto decoded = decode_program_wire(*bytes);
            CHECK(std::holds_alternative<VerifiedProgram>(decoded));
            if (const auto* restored = std::get_if<VerifiedProgram>(&decoded))
                CHECK(program_digest(*restored) == program_digest(*verified));
        }
    }
    maps[0].results[0].operands[1] = load;
    CHECK(std::holds_alternative<VerifiedProgram>(verify_and_canonicalize_program(program)));
    maps[0].results[0].operands.clear();
    CHECK(rejected_with(program, CompatibilityError::IR_SHAPE_MISMATCH));
    maps[0].results[0] = load;
    maps[0].results[0].value = 2;
    CHECK(rejected_with(program, CompatibilityError::IR_REFERENCE_INVALID));
    maps[0].results[0] = load;
    root.arguments[1].type.element_type = ElementType::F32;
    program.functions.front().regions.front().arguments[1].type.element_type = ElementType::F32;
    CHECK(rejected_with(program, CompatibilityError::IR_SHAPE_MISMATCH));
    root.arguments[1].type.element_type = ElementType::U32;
    program.functions.front().regions.front().arguments[1].type.element_type = ElementType::U32;
    maps[0].results[0].operands[1] = {
        TensorIndexExpression::Add, 0,
        {{TensorIndexExpression::Constant, INT64_MAX, {}},
         {TensorIndexExpression::Constant, 1, {}}}};
    CHECK(rejected_with(program, CompatibilityError::IR_CONSTRAINT_FAILED));
    TensorIndexExpr deep = load;
    for (size_t depth = 0; depth < 34; ++depth)
        deep = {TensorIndexExpression::SourceElement, 1, {zero, std::move(deep)}};
    maps[0].results[0] = std::move(deep);
    CHECK(rejected_with(program, CompatibilityError::IR_CONSTRAINT_FAILED));
}

void test_typed_scalar_primitives() {
    const ValueType f32 = scalar(), u32 = scalar(ElementType::U32);
    const ValueType i1 = scalar(ElementType::I1);
    const auto make = [&](Primitive code, std::vector<TypedValue> args,
                          std::vector<uint32_t> inputs, ValueType output) {
        return one_function_program(0, 0, std::move(args),
            {instruction(10, code, std::move(inputs), {value(11, output)})},
            {11}, {output});
    };
    const std::vector<Program> valid = {
        make(Primitive::Less, {value(1, u32), value(2, u32)}, {1, 2}, i1),
        make(Primitive::Equal, {value(1), value(2)}, {1, 2}, i1),
        make(Primitive::Select, {value(1, i1), value(2), value(3)}, {1, 2, 3}, f32),
        make(Primitive::Convert, {value(1, u32)}, {1}, f32),
        make(Primitive::Sqrt, {value(1)}, {1}, f32),
        make(Primitive::Tanh, {value(1)}, {1}, f32)};
    for (const Program& program : valid) {
        const auto checked = verify_and_canonicalize_program(program);
        CHECK(std::holds_alternative<VerifiedProgram>(checked));
        if (const auto* verified = std::get_if<VerifiedProgram>(&checked)) {
            const auto wire = encode_program_wire(*verified);
            CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
            if (const auto* bytes = std::get_if<std::vector<uint8_t>>(&wire)) {
                const auto decoded = decode_program_wire(*bytes);
                CHECK(std::holds_alternative<VerifiedProgram>(decoded));
                if (const auto* restored = std::get_if<VerifiedProgram>(&decoded))
                    CHECK(program_digest(*restored) == program_digest(*verified));
            }
        }
    }
    for (Program program : valid) {
        program.functions.front().regions.front().instructions.front().inputs.clear();
        CHECK(std::holds_alternative<CompatibilityReport>(verify_and_canonicalize_program(program)));
    }
    CHECK(rejected_with(make(Primitive::Less, {value(1), value(2, u32)}, {1, 2}, i1), CompatibilityError::IR_SHAPE_MISMATCH));
    CHECK(rejected_with(make(Primitive::Select, {value(1, u32), value(2), value(3)}, {1, 2, 3}, f32), CompatibilityError::IR_SHAPE_MISMATCH));
    CHECK(rejected_with(make(Primitive::Convert, {value(1)}, {1}, u32), CompatibilityError::IR_SHAPE_MISMATCH));
}

} // namespace

int main() {
    test_coordinate_and_require_contract();
    test_arithmetic_policy_contract();
    test_source_element_index_contract();
    test_typed_scalar_primitives();
    test_minimal_programs();
    test_generic_f32_algebra_contract();
    test_data_dependent_tensor_index();
    test_structured_tensor_contract();
    test_unseen_repeated_graph_with_multiple_outputs();
    test_export_serialization_order_is_not_identity();
    test_bounded_loop_and_state_edge();
    test_canonical_digest_ignores_local_ids_and_independent_order();
    test_declaration_storage_order_is_not_semantic();
    test_state_relation_is_semantic();
    test_effect_sets_and_control_containment();
    test_constant_storage_widths();
    test_loop_overflow_model();
    test_dependency_depth_is_bounded();
    test_edges_change_digest_and_result();
    test_sharing_is_part_of_canonical_identity();
    test_rejections();
    test_structural_wire_roundtrip();
    return test_summary("test_program_ir");
}
