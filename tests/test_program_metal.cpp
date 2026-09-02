#include "program_metal.h"
#include "program_reference.h"
#include "test_util.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cmath>
#include <fcntl.h>
#include <memory>
#include <optional>
#include <string>
#include <unistd.h>
#include <vector>

using namespace Laplace;

namespace {

DimensionExpr dimension(uint64_t value) {
    return {DimensionExpression::Constant, value, {}};
}

ValueType tensor(std::initializer_list<uint64_t> dimensions) {
    ValueType type{ElementType::F32, {}};
    for (uint64_t value : dimensions)
        type.dimensions.push_back(dimension(value));
    return type;
}

TensorIndexExpr iterator(uint32_t index) {
    return {TensorIndexExpression::Iterator, static_cast<int64_t>(index), {}};
}

TensorIndexMap map(std::initializer_list<uint32_t> iterators) {
    TensorIndexMap result;
    for (uint32_t index : iterators) result.results.push_back(iterator(index));
    return result;
}

MetalProgramValue f32_value(std::initializer_list<uint64_t> extents,
                            std::initializer_list<float> values);
ReferenceValue reference_value(const MetalProgramValue& value);

Program dynamic_gather_program() {
    const ValueType table{ElementType::F32, {dimension(4), dimension(3)}};
    const ValueType index{ElementType::U32, {}};
    const ValueType output = tensor({3});
    const ValueType scalar{ElementType::F32, {}};
    Region body{20, {{21, scalar}, {22, index}, {23, scalar}},
                {Instruction{24, {Primitive::Add, 1, 0}, {21, 23},
                             {{25, scalar}}, {}, {}, NoAttributes{}}},
                {25}};
    TensorIndexMap table_map;
    table_map.results = {
        TensorIndexExpr{TensorIndexExpression::SourceScalar, 1, {}},
        iterator(0)};
    StructuredTensorAttributes attributes;
    attributes.source_count = 2;
    attributes.iteration_dimensions = {dimension(3)};
    attributes.iterator_kinds = {TensorIteratorKind::Parallel};
    attributes.indexing_maps = {
        std::move(table_map), TensorIndexMap{}, map({0})};
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

void test_data_dependent_tensor_index_metal() {
    auto verified = verify_and_canonicalize_program(dynamic_gather_program());
    CHECK(std::holds_alternative<VerifiedProgram>(verified));
    if (!std::holds_alternative<VerifiedProgram>(verified)) return;
    auto compiled = compile_metal_program(std::get<VerifiedProgram>(verified));
    const auto* compile_error = std::get_if<CompatibilityReport>(&compiled);
    CHECK_MSG(std::holds_alternative<MetalProgramExecutable>(compiled),
              "dynamic gather compile code=%u detail=%s",
              compile_error ? static_cast<unsigned>(compile_error->code) : 0,
              compile_error ? compile_error->detail.c_str() : "none");
    if (!std::holds_alternative<MetalProgramExecutable>(compiled)) return;
    const std::array<MetalProgramInput, 2> inputs = {
        MetalProgramInput{10, f32_value({4, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})},
        MetalProgramInput{11, {ElementType::U32, {}, {2}}}};
    auto executable =
        std::get<MetalProgramExecutable>(std::move(compiled));
    const auto result = executable.execute(inputs);
    CHECK(std::holds_alternative<MetalProgramResult>(result));
    if (const auto* output = std::get_if<MetalProgramResult>(&result)) {
        CHECK(output->exports ==
              std::vector<MetalProgramValue>{f32_value({3}, {7, 8, 9})});
        CHECK(output->audit.command_buffers == 1);
        CHECK(output->audit.implicit_weight_copies == 0);
    }
    auto invalid = inputs;
    invalid[1].value.bits.front() = 4;
    const auto rejected = executable.execute(invalid);
    const auto* report = std::get_if<CompatibilityReport>(&rejected);
    CHECK(report != nullptr);
    if (report)
        CHECK(report->code == CompatibilityError::RUNTIME_INPUT_INVALID);
}

Program dependent_tensor_stages_program() {
    const ValueType vector = tensor({4});
    const ValueType scalar{ElementType::F32, {}};
    Region double_body;
    double_body.id = 20;
    double_body.arguments = {{21, scalar}, {22, scalar}};
    double_body.instructions = {
        Instruction{23, {Primitive::Constant, 1, 0}, {}, {{24, scalar}}, {}, {},
                    ConstantAttributes{std::bit_cast<uint32_t>(2.0f)}},
        Instruction{25, {Primitive::Multiply, 1, 0}, {21, 24}, {{26, scalar}},
                    {}, {}, NoAttributes{}},
        Instruction{27, {Primitive::Add, 1, 0}, {22, 26}, {{28, scalar}},
                    {}, {}, NoAttributes{}}};
    double_body.yields = {28};

    Region increment_body;
    increment_body.id = 30;
    increment_body.arguments = {{31, scalar}, {32, scalar}};
    increment_body.instructions = {
        Instruction{33, {Primitive::Constant, 1, 0}, {}, {{34, scalar}}, {}, {},
                    ConstantAttributes{std::bit_cast<uint32_t>(1.0f)}},
        Instruction{35, {Primitive::Add, 1, 0}, {31, 34}, {{36, scalar}},
                    {}, {}, NoAttributes{}},
        Instruction{37, {Primitive::Add, 1, 0}, {32, 36}, {{38, scalar}},
                    {}, {}, NoAttributes{}}};
    increment_body.yields = {38};

    const auto attributes = [] {
        StructuredTensorAttributes value;
        value.source_count = 1;
        value.iteration_dimensions = {dimension(4)};
        value.iterator_kinds = {TensorIteratorKind::Parallel};
        value.indexing_maps = {map({0}), map({0})};
        return value;
    };
    Region root;
    root.id = 1;
    root.arguments = {{10, vector}};
    root.instructions = {
        Instruction{11, {Primitive::Constant, 1, 0}, {}, {{12, vector}}, {}, {},
                    ConstantAttributes{0}},
        Instruction{13, {Primitive::StructuredTensor, 1, 0}, {10, 12},
                    {{14, vector}}, {20}, {}, attributes()},
        Instruction{15, {Primitive::Constant, 1, 0}, {}, {{16, vector}}, {}, {},
                    ConstantAttributes{0}},
        Instruction{17, {Primitive::StructuredTensor, 1, 0}, {14, 16},
                    {{18, vector}}, {30}, {}, attributes()}};
    root.yields = {18};
    Program program;
    program.minor = 1;
    program.functions = {{9, 1,
                          {std::move(double_body), std::move(increment_body),
                           std::move(root)},
                          {vector}}};
    program.exports = {{9, 0, vector}};
    return program;
}

void test_dependent_tensor_stages_one_command_buffer() {
    auto verified =
        verify_and_canonicalize_program(dependent_tensor_stages_program());
    CHECK(std::holds_alternative<VerifiedProgram>(verified));
    if (!std::holds_alternative<VerifiedProgram>(verified)) return;
    const VerifiedProgram& program = std::get<VerifiedProgram>(verified);
    auto compiled = compile_metal_program(program);
    const auto* report = std::get_if<CompatibilityReport>(&compiled);
    CHECK_MSG(std::holds_alternative<MetalProgramExecutable>(compiled),
              "dependent stages compile code=%u detail=%s",
              report ? static_cast<unsigned>(report->code) : 0,
              report ? report->detail.c_str() : "none");
    if (!std::holds_alternative<MetalProgramExecutable>(compiled)) return;
    const std::array<MetalProgramInput, 1> inputs = {
        MetalProgramInput{10, f32_value({4}, {1, 2, 3, 4})}};
    const auto metal =
        std::get<MetalProgramExecutable>(std::move(compiled)).execute(inputs);
    CHECK(std::holds_alternative<MetalProgramResult>(metal));
    ReferenceState state;
    const std::array<ReferenceInput, 1> reference_inputs = {
        ReferenceInput{10, reference_value(inputs.front().value)}};
    const auto reference = execute_reference_program(program, state,
                                                     reference_inputs);
    CHECK(std::holds_alternative<ReferenceResult>(reference));
    if (const auto* result = std::get_if<MetalProgramResult>(&metal)) {
        CHECK(result->exports ==
              std::vector<MetalProgramValue>{f32_value({4}, {3, 5, 7, 9})});
        if (const auto* expected = std::get_if<ReferenceResult>(&reference))
            CHECK(reference_value(result->exports.front()) ==
                  expected->exports.front());
        CHECK(result->audit.command_buffers == 1);
        CHECK(result->audit.implicit_weight_copies == 0);
    }
}

Program transactional_state_stages_program() {
    const ValueType vector = tensor({1});
    const ValueType table{ElementType::F32, {dimension(1), dimension(1)}};
    const ValueType index{ElementType::U32, {}};
    const ValueType scalar{ElementType::F32, {}};

    Region increment_body;
    increment_body.id = 20;
    increment_body.arguments = {{21, scalar}, {22, scalar}};
    increment_body.instructions = {
        Instruction{23, {Primitive::Constant, 1, 0}, {}, {{24, scalar}}, {}, {},
                    ConstantAttributes{std::bit_cast<uint32_t>(1.0f)}},
        Instruction{25, {Primitive::Add, 1, 0}, {21, 24}, {{26, scalar}},
                    {}, {}, NoAttributes{}},
        Instruction{27, {Primitive::Add, 1, 0}, {22, 26}, {{28, scalar}},
                    {}, {}, NoAttributes{}}};
    increment_body.yields = {28};

    Region output_body;
    output_body.id = 30;
    output_body.arguments = {
        {31, scalar}, {32, scalar}, {33, index}, {34, scalar}};
    output_body.instructions = {
        Instruction{35, {Primitive::Add, 1, 0}, {31, 32}, {{36, scalar}},
                    {}, {}, NoAttributes{}},
        Instruction{37, {Primitive::Add, 1, 0}, {34, 36}, {{38, scalar}},
                    {}, {}, NoAttributes{}}};
    output_body.yields = {38};

    StructuredTensorAttributes increment;
    increment.source_count = 1;
    increment.iteration_dimensions = {dimension(1)};
    increment.iterator_kinds = {TensorIteratorKind::Parallel};
    increment.indexing_maps = {map({0}), map({0})};
    StructuredTensorAttributes output;
    output.source_count = 3;
    output.iteration_dimensions = {dimension(1)};
    output.iterator_kinds = {TensorIteratorKind::Parallel};
    TensorIndexMap table_map;
    table_map.results = {
        TensorIndexExpr{TensorIndexExpression::SourceScalar, 2, {}},
        iterator(0)};
    output.indexing_maps = {
        map({0}), std::move(table_map), TensorIndexMap{}, map({0})};

    Region root;
    root.id = 1;
    root.arguments = {{9, index}, {10, table}};
    root.instructions = {
        Instruction{11, {Primitive::StateRead, 1, 0}, {}, {{12, vector}}, {}, {},
                    StateAttributes{70}},
        Instruction{13, {Primitive::Constant, 1, 0}, {}, {{14, vector}}, {}, {},
                    ConstantAttributes{0}},
        Instruction{15, {Primitive::StructuredTensor, 1, 0}, {12, 14},
                    {{16, vector}}, {20}, {}, std::move(increment)},
        Instruction{17, {Primitive::StateWrite, 1, 0}, {16}, {}, {}, {11},
                    StateAttributes{70}},
        Instruction{18, {Primitive::Constant, 1, 0}, {}, {{19, vector}}, {}, {},
                    ConstantAttributes{0}},
        Instruction{20, {Primitive::StructuredTensor, 1, 0}, {16, 10, 9, 19},
                    {{40, vector}}, {30}, {}, std::move(output)}};
    root.yields = {40};
    Program program;
    program.minor = 1;
    program.state_references = {{70, vector, UINT32_MAX, true}};
    program.functions = {{8, 1,
                          {std::move(increment_body), std::move(output_body),
                           std::move(root)},
                          {vector}}};
    program.exports = {{8, 0, vector}};
    return program;
}

void test_transactional_state_publication() {
    auto verified =
        verify_and_canonicalize_program(transactional_state_stages_program());
    CHECK(std::holds_alternative<VerifiedProgram>(verified));
    if (!std::holds_alternative<VerifiedProgram>(verified)) return;
    auto compiled = compile_metal_program(std::get<VerifiedProgram>(verified));
    const auto* report = std::get_if<CompatibilityReport>(&compiled);
    CHECK_MSG(std::holds_alternative<MetalProgramExecutable>(compiled),
              "transactional stages compile code=%u detail=%s",
              report ? static_cast<unsigned>(report->code) : 0,
              report ? report->detail.c_str() : "none");
    if (!std::holds_alternative<MetalProgramExecutable>(compiled)) return;
    auto executable =
        std::get<MetalProgramExecutable>(std::move(compiled));
    std::array<MetalProgramInput, 2> inputs = {
        MetalProgramInput{9, {ElementType::U32, {}, {0}}},
        MetalProgramInput{10, f32_value({1, 1}, {0})}};
    const auto first = executable.execute(inputs);
    CHECK(std::holds_alternative<MetalProgramResult>(first));
    if (const auto* result = std::get_if<MetalProgramResult>(&first)) {
        CHECK(result->exports ==
              std::vector<MetalProgramValue>{f32_value({1}, {1})});
        CHECK(result->audit.command_buffers == 1);
        CHECK(result->audit.state_generation == 1);
    }
    inputs[0].value.bits.front() = 1;
    CHECK(std::holds_alternative<CompatibilityReport>(
        executable.execute(inputs)));
    inputs[0].value.bits.front() = 0;
    const auto retry = executable.execute(inputs);
    CHECK(std::holds_alternative<MetalProgramResult>(retry));
    if (const auto* result = std::get_if<MetalProgramResult>(&retry)) {
        CHECK(result->exports ==
              std::vector<MetalProgramValue>{f32_value({1}, {2})});
        CHECK(result->audit.command_buffers == 1);
        CHECK(result->audit.state_generation == 2);
    }
}

Program contraction_program(uint32_t rename = 0) {
    const ValueType left = tensor({2, 3});
    const ValueType right = tensor({3, 2});
    const ValueType output = tensor({2, 2});
    const ValueType scalar{ElementType::F32, {}};

    Region body;
    body.id = 20 + rename;
    body.arguments = {{21 + rename, scalar}, {22 + rename, scalar},
                      {23 + rename, scalar}};
    body.instructions = {
        Instruction{24 + rename, {Primitive::Multiply, 1, 0},
                    {21 + rename, 22 + rename}, {{25 + rename, scalar}},
                    {}, {}, NoAttributes{}},
        Instruction{26 + rename, {Primitive::Add, 1, 0},
                    {23 + rename, 25 + rename}, {{27 + rename, scalar}},
                    {}, {}, NoAttributes{}}};
    body.yields = {27 + rename};

    StructuredTensorAttributes attributes;
    attributes.source_count = 2;
    attributes.iteration_dimensions = {dimension(2), dimension(2), dimension(3)};
    attributes.iterator_kinds = {TensorIteratorKind::Parallel,
                                 TensorIteratorKind::Parallel,
                                 TensorIteratorKind::Reduction};
    attributes.indexing_maps = {map({0, 2}), map({2, 1}), map({0, 1})};

    Region root;
    root.id = 1 + rename;
    root.arguments = {{10 + rename, left}, {11 + rename, right}};
    root.instructions = {
        Instruction{12 + rename, {Primitive::Constant, 1, 0}, {},
                    {{13 + rename, output}}, {}, {}, ConstantAttributes{0}},
        Instruction{14 + rename, {Primitive::StructuredTensor, 1, 0},
                    {10 + rename, 11 + rename, 13 + rename},
                    {{15 + rename, output}}, {20 + rename}, {},
                    std::move(attributes)}};
    root.yields = {15 + rename};

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

Program transpose_broadcast_program() {
    const ValueType matrix = tensor({3, 2});
    const ValueType bias = tensor({3});
    const ValueType output = tensor({2, 3});
    const ValueType scalar{ElementType::F32, {}};
    Region body{20, {{21, scalar}, {22, scalar}, {23, scalar}},
                {Instruction{24, {Primitive::Add, 1, 0}, {21, 22},
                             {{25, scalar}}, {}, {}, NoAttributes{}}},
                {25}};
    StructuredTensorAttributes attributes;
    attributes.source_count = 2;
    attributes.iteration_dimensions = {dimension(2), dimension(3)};
    attributes.iterator_kinds = {TensorIteratorKind::Parallel,
                                 TensorIteratorKind::Parallel};
    attributes.indexing_maps = {map({1, 0}), map({1}), map({0, 1})};
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

Program reduction_program() {
    const ValueType matrix = tensor({2, 3});
    const ValueType output = tensor({2});
    const ValueType scalar{ElementType::F32, {}};
    Region body{20, {{21, scalar}, {22, scalar}},
                {Instruction{23, {Primitive::Add, 1, 0}, {21, 22},
                             {{24, scalar}}, {}, {}, NoAttributes{}}},
                {24}};
    StructuredTensorAttributes attributes;
    attributes.source_count = 1;
    attributes.iteration_dimensions = {dimension(2), dimension(3)};
    attributes.iterator_kinds = {TensorIteratorKind::Parallel,
                                 TensorIteratorKind::Reduction};
    attributes.indexing_maps = {map({0, 1}), map({0})};
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

Program zero_padded_program() {
    const ValueType input = tensor({2});
    const ValueType output = tensor({4});
    const ValueType scalar{ElementType::F32, {}};
    Region body{20, {{21, scalar}, {22, scalar}},
                {Instruction{23, {Primitive::Add, 1, 0}, {21, 22},
                             {{24, scalar}}, {}, {}, NoAttributes{}}},
                {24}};
    TensorIndexMap padded;
    padded.bounds = TensorBoundsMode::Zero;
    padded.results = {{
        TensorIndexExpression::Add, 0,
        {iterator(0),
         TensorIndexExpr{TensorIndexExpression::Constant, -1, {}}}}};
    StructuredTensorAttributes attributes;
    attributes.source_count = 1;
    attributes.iteration_dimensions = {dimension(4)};
    attributes.iterator_kinds = {TensorIteratorKind::Parallel};
    attributes.indexing_maps = {std::move(padded), map({0})};
    Region root;
    root.id = 1;
    root.arguments = {{10, input}};
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

Program scalar_program() {
    const ValueType scalar{ElementType::F32, {}};
    Region root;
    root.id = 1;
    root.instructions = {
        Instruction{10, {Primitive::Constant, 1, 0}, {}, {{11, scalar}},
                    {}, {}, ConstantAttributes{std::bit_cast<uint32_t>(2.0f)}}};
    root.yields = {11};
    Function function{9, 1, {std::move(root)}, {scalar}};
    Program program;
    program.functions = {std::move(function)};
    program.exports = {{9, 0, scalar}};
    return program;
}

Program algebra_program(Primitive primitive, bool unary) {
    const ValueType input = tensor({4});
    const ValueType scalar{ElementType::F32, {}};
    Region body;
    body.id = 20;
    body.arguments = {{21, scalar}};
    std::vector<uint32_t> scalar_inputs = {21};
    if (!unary) {
        body.arguments.push_back({22, scalar});
        scalar_inputs.push_back(22);
    }
    body.arguments.push_back({23, scalar});
    body.instructions = {
        Instruction{24, {primitive, 1, 0}, std::move(scalar_inputs),
                    {{25, scalar}}, {}, {}, NoAttributes{}}};
    body.yields = {25};

    StructuredTensorAttributes attributes;
    attributes.source_count = unary ? 1 : 2;
    attributes.iteration_dimensions = {dimension(4)};
    attributes.iterator_kinds = {TensorIteratorKind::Parallel};
    attributes.indexing_maps = {map({0})};
    if (!unary) attributes.indexing_maps.push_back(map({0}));
    attributes.indexing_maps.push_back(map({0}));

    Region root;
    root.id = 1;
    root.arguments = {{10, input}};
    std::vector<uint32_t> tensor_inputs = {10};
    if (!unary) {
        root.arguments.push_back({11, input});
        tensor_inputs.push_back(11);
    }
    root.instructions = {
        Instruction{12, {Primitive::Constant, 1, 0}, {}, {{13, input}},
                    {}, {}, ConstantAttributes{0}}};
    tensor_inputs.push_back(13);
    root.instructions.push_back(
        Instruction{14, {Primitive::StructuredTensor, 1, 0},
                    std::move(tensor_inputs), {{15, input}}, {20}, {},
                    std::move(attributes)});
    root.yields = {15};
    Function function{9, 1, {std::move(body), std::move(root)}, {input}};
    Program result;
    result.minor = 1;
    result.functions = {std::move(function)};
    result.exports = {{9, 0, input}};
    return result;
}

void check_algebra_metal(Primitive primitive,
                         const MetalProgramValue& left,
                         std::optional<MetalProgramValue> right = std::nullopt) {
    auto verified = verify_and_canonicalize_program(
        algebra_program(primitive, !right.has_value()));
    CHECK(std::holds_alternative<VerifiedProgram>(verified));
    if (!std::holds_alternative<VerifiedProgram>(verified)) return;
    auto compiled = compile_metal_program(std::get<VerifiedProgram>(verified));
    const auto* compile_error = std::get_if<CompatibilityReport>(&compiled);
    CHECK_MSG(std::holds_alternative<MetalProgramExecutable>(compiled),
              "algebra primitive=%u compile code=%u detail=%s",
              static_cast<unsigned>(primitive),
              compile_error ? static_cast<unsigned>(compile_error->code) : 0,
              compile_error ? compile_error->detail.c_str() : "none");
    if (!std::holds_alternative<MetalProgramExecutable>(compiled)) return;
    std::vector<MetalProgramInput> inputs = {{10, left}};
    if (right) inputs.push_back({11, *right});
    const auto metal = std::get<MetalProgramExecutable>(std::move(compiled))
                           .execute(inputs);
    CHECK(std::holds_alternative<MetalProgramResult>(metal));
    ReferenceState state;
    std::vector<ReferenceInput> reference_inputs;
    for (const auto& input : inputs)
        reference_inputs.push_back({input.value_id, reference_value(input.value)});
    const auto reference = execute_reference_program(
        std::get<VerifiedProgram>(verified), state, reference_inputs);
    CHECK(std::holds_alternative<ReferenceResult>(reference));
    const auto* observed = std::get_if<MetalProgramResult>(&metal);
    const auto* expected = std::get_if<ReferenceResult>(&reference);
    if (!observed || !expected) return;
    CHECK(observed->audit.command_buffers == 1);
    CHECK(observed->audit.implicit_weight_copies == 0);
    CHECK(observed->exports.size() == 1);
    CHECK(expected->exports.size() == 1);
    if (observed->exports.empty() || expected->exports.empty()) return;
    CHECK(observed->exports.front().bits.size() ==
          expected->exports.front().bits.size());
    for (size_t index = 0;
         index < observed->exports.front().bits.size() &&
         index < expected->exports.front().bits.size(); ++index) {
        const float actual = std::bit_cast<float>(static_cast<uint32_t>(
            observed->exports.front().bits[index]));
        const float wanted = std::bit_cast<float>(static_cast<uint32_t>(
            expected->exports.front().bits[index]));
        CHECK(std::isfinite(actual));
        CHECK(std::abs(actual - wanted) <=
              3.0e-6f * std::max(1.0f, std::abs(wanted)));
    }
}

void test_generic_f32_algebra_metal() {
    const auto left = f32_value({4}, {-3.0f, -0.5f, 0.5f, 4.0f});
    const auto right = f32_value({4}, {2.0f, 0.25f, 2.0f, 8.0f});
    for (Primitive primitive : {Primitive::Subtract, Primitive::Divide,
                                Primitive::Maximum})
        check_algebra_metal(primitive, left, right);
    for (Primitive primitive : {Primitive::Negate, Primitive::Exp,
                                Primitive::Log, Primitive::Rsqrt,
                                Primitive::Sin, Primitive::Cos}) {
        MetalProgramValue input = left;
        if (primitive == Primitive::Log || primitive == Primitive::Rsqrt)
            input = f32_value({4}, {0.25f, 0.5f, 1.0f, 4.0f});
        check_algebra_metal(primitive, input);
    }
}

MetalProgramValue f32_value(std::initializer_list<uint64_t> extents,
                            std::initializer_list<float> values) {
    MetalProgramValue result;
    result.type = ElementType::F32;
    result.extents.assign(extents);
    for (float value : values)
        result.bits.push_back(std::bit_cast<uint32_t>(value));
    return result;
}

ReferenceValue reference_value(const MetalProgramValue& value) {
    return {value.type, value.extents, value.bits};
}

PhysicalInstruction physical_instruction(
    PhysicalOpcode opcode, PhysicalValueType result,
    std::initializer_list<uint32_t> operands = {}, uint64_t immediate = 0,
    uint16_t plane = kNoPhysicalPlane, uint16_t policy = kNoPhysicalPolicy,
    uint8_t bit_width = 0);

std::optional<VerifiedPhysicalProgramPackage>
bound_dependent_stages_package() {
    PhysicalProgram decoder;
    decoder.logical_rank = 1;
    decoder.planes = {{PhysicalPlaneStorage::External, 1, 0, 0}};
    decoder.policies.push_back({});
    decoder.instructions = {
        physical_instruction(PhysicalOpcode::Coordinate,
                             PhysicalValueType::Index),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 32),
        physical_instruction(PhysicalOpcode::IndexMultiply,
                             PhysicalValueType::Index, {0, 1}),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {2}, 0, 0,
                             kNoPhysicalPolicy, 32),
        physical_instruction(PhysicalOpcode::BitsToF32,
                             PhysicalValueType::F32, {3}, 0,
                             kNoPhysicalPlane, 0)};
    decoder.result = 4;
    auto canonical = canonicalize_physical_program(std::move(decoder));
    CHECK(std::holds_alternative<PhysicalProgram>(canonical));
    if (!std::holds_alternative<PhysicalProgram>(canonical))
        return std::nullopt;
    const PhysicalProgram& physical = std::get<PhysicalProgram>(canonical);
    auto wire = encode_physical_program(physical);
    auto digest = physical_program_digest(physical);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
    CHECK(std::holds_alternative<PhysicalProgramDigest>(digest));
    if (!std::holds_alternative<std::vector<uint8_t>>(wire) ||
        !std::holds_alternative<PhysicalProgramDigest>(digest))
        return std::nullopt;

    char path[] = "/private/tmp/laplace-program-metal-stages-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd < 0) return std::nullopt;
    std::vector<uint8_t> bytes;
    for (float value : {1, 2, 3, 4}) {
        const uint32_t bits = std::bit_cast<uint32_t>(value);
        for (unsigned shift = 0; shift != 32; shift += 8)
            bytes.push_back(static_cast<uint8_t>(bits >> shift));
    }
    CHECK(write(fd, bytes.data(), bytes.size()) ==
          static_cast<ssize_t>(bytes.size()));
    CHECK(close(fd) == 0);
    auto artifacts = ArtifactSet::load_single_file(path);
    unlink(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts))
        return std::nullopt;
    auto view = std::get<ArtifactSet>(std::move(artifacts)).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return std::nullopt;
    ArtifactIndexInput input;
    input.artifacts.push_back(std::get<PackageView>(std::move(view)));
    auto index = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<ArtifactIndex>(index));
    if (!std::holds_alternative<ArtifactIndex>(index)) return std::nullopt;

    auto semantic =
        verify_and_canonicalize_program(dependent_tensor_stages_program());
    CHECK(std::holds_alternative<VerifiedProgram>(semantic));
    if (!std::holds_alternative<VerifiedProgram>(semantic))
        return std::nullopt;
    PhysicalProgramRecord record{
        std::get<PhysicalProgramDigest>(digest),
        std::get<std::vector<uint8_t>>(std::move(wire)),
        {ElementType::F32, {4}}};
    PhysicalResourceBinding resource;
    resource.resource_id = 0;
    resource.program_digest = record.digest;
    resource.semantic_function_id = 9;
    resource.semantic_value_id = 10;
    resource.planes = {{0, ArtifactId{0}, 0, bytes.size()}};
    auto package = load_physical_program_package(
        std::get<ArtifactIndex>(std::move(index)),
        std::get<VerifiedProgram>(std::move(semantic)),
        std::span<const PhysicalProgramRecord>(&record, 1),
        std::span<const PhysicalResourceBinding>(&resource, 1));
    CHECK(std::holds_alternative<VerifiedPhysicalProgramPackage>(package));
    if (!std::holds_alternative<VerifiedPhysicalProgramPackage>(package))
        return std::nullopt;
    return std::get<VerifiedPhysicalProgramPackage>(std::move(package));
}

void test_bound_dependent_stages_one_command_buffer() {
    auto package = bound_dependent_stages_package();
    CHECK(package.has_value());
    if (!package) return;
    auto compiled = compile_metal_program(*package);
    const auto* compile_error = std::get_if<CompatibilityReport>(&compiled);
    CHECK_MSG(std::holds_alternative<MetalProgramExecutable>(compiled),
              "bound stages compile code=%u detail=%s",
              compile_error ? static_cast<unsigned>(compile_error->code) : 0,
              compile_error ? compile_error->detail.c_str() : "none");
    if (!std::holds_alternative<MetalProgramExecutable>(compiled)) return;
    const auto executed =
        std::get<MetalProgramExecutable>(std::move(compiled))
            .execute(std::span<const MetalProgramInput>{});
    CHECK(std::holds_alternative<MetalProgramResult>(executed));
    if (const auto* result = std::get_if<MetalProgramResult>(&executed)) {
        CHECK(result->exports ==
              std::vector<MetalProgramValue>{f32_value({4}, {3, 5, 7, 9})});
        CHECK(result->audit.command_buffers == 1);
        CHECK(result->audit.implicit_weight_copies == 0);
        CHECK(result->audit.explicit_upload_bytes == 0);
        CHECK(result->audit.zero_copy_plane_bytes == 16);
        CHECK(result->audit.persistent_plane_bytes == 16);
    }
}

void check_matches_reference(Program source,
                             std::span<const MetalProgramInput> inputs) {
    const auto verified = verify_and_canonicalize_program(std::move(source));
    CHECK(std::holds_alternative<VerifiedProgram>(verified));
    if (!std::holds_alternative<VerifiedProgram>(verified)) return;
    const VerifiedProgram& program = std::get<VerifiedProgram>(verified);
    auto compiled = compile_metal_program(program);
    CHECK(std::holds_alternative<MetalProgramExecutable>(compiled));
    if (!std::holds_alternative<MetalProgramExecutable>(compiled)) return;
    const auto metal = std::get<MetalProgramExecutable>(std::move(compiled))
                           .execute(inputs);
    CHECK(std::holds_alternative<MetalProgramResult>(metal));

    std::vector<ReferenceInput> reference_inputs;
    for (const MetalProgramInput& input : inputs)
        reference_inputs.push_back(
            {input.value_id, reference_value(input.value)});
    ReferenceState state;
    const auto reference = execute_reference_program(
        program, state, reference_inputs);
    CHECK(std::holds_alternative<ReferenceResult>(reference));
    if (const auto* metal_result = std::get_if<MetalProgramResult>(&metal)) {
        const auto* reference_result = std::get_if<ReferenceResult>(&reference);
        CHECK(reference_result != nullptr);
        if (reference_result) {
            CHECK(metal_result->exports.size() ==
                  reference_result->exports.size());
            if (metal_result->exports.size() ==
                reference_result->exports.size()) {
                for (size_t index = 0; index < metal_result->exports.size(); ++index)
                    CHECK(reference_value(metal_result->exports[index]) ==
                          reference_result->exports[index]);
            }
        }
        CHECK(metal_result->audit.command_buffers == 1);
        CHECK(metal_result->audit.implicit_weight_copies == 0);
    }
}

void test_contraction() {
    const auto verified = verify_and_canonicalize_program(contraction_program());
    CHECK(std::holds_alternative<VerifiedProgram>(verified));
    if (!std::holds_alternative<VerifiedProgram>(verified)) return;
    const VerifiedProgram& program = std::get<VerifiedProgram>(verified);
    auto compiled = compile_metal_program(program);
    CHECK(std::holds_alternative<MetalProgramExecutable>(compiled));
    if (!std::holds_alternative<MetalProgramExecutable>(compiled)) return;
    MetalProgramExecutable executable =
        std::get<MetalProgramExecutable>(std::move(compiled));
    CHECK(executable.program_digest() == program_digest(program));

    const std::array<MetalProgramInput, 2> inputs = {
        MetalProgramInput{10, f32_value({2, 3}, {1, 2, 3, 4, 5, 6})},
        MetalProgramInput{11, f32_value({3, 2}, {1, 2, 3, 4, 5, 6})}};
    const auto executed = executable.execute(inputs);
    CHECK(std::holds_alternative<MetalProgramResult>(executed));
    if (const auto* result = std::get_if<MetalProgramResult>(&executed)) {
        CHECK(result->exports.size() == 1);
        CHECK(result->exports[0] ==
              f32_value({2, 2}, {22, 28, 49, 64}));
        CHECK(result->audit.program_digest == program_digest(program));
        CHECK(result->audit.lowering_digest == executable.lowering_digest());
        CHECK(result->audit.command_buffers == 1);
        CHECK(result->audit.implicit_weight_copies == 0);
        CHECK(result->audit.explicit_upload_bytes == 48);
        CHECK(result->audit.explicit_download_bytes == 16);
        CHECK(result->audit.thread_execution_width != 0);
        CHECK(result->audit.max_total_threads_per_threadgroup >=
              result->audit.thread_execution_width);
    }

    std::vector<ReferenceInput> reference_inputs;
    for (const MetalProgramInput& input : inputs)
        reference_inputs.push_back(
            {input.value_id, reference_value(input.value)});
    ReferenceState reference_state;
    const auto reference = execute_reference_program(
        program, reference_state, reference_inputs);
    CHECK(std::holds_alternative<ReferenceResult>(reference));
    if (const auto* metal = std::get_if<MetalProgramResult>(&executed)) {
        const auto* expected = std::get_if<ReferenceResult>(&reference);
        CHECK(expected != nullptr);
        if (expected && !expected->exports.empty())
            CHECK(reference_value(metal->exports.front()) ==
                  expected->exports.front());
    }

    auto renamed = verify_and_canonicalize_program(contraction_program(1000));
    CHECK(std::holds_alternative<VerifiedProgram>(renamed));
    if (const auto* value = std::get_if<VerifiedProgram>(&renamed)) {
        CHECK(program_digest(*value) == program_digest(program));
        auto renamed_compile = compile_metal_program(*value);
        CHECK(std::holds_alternative<MetalProgramExecutable>(renamed_compile));
        if (const auto* renamed_executable =
                std::get_if<MetalProgramExecutable>(&renamed_compile))
            CHECK(renamed_executable->lowering_digest() ==
                  executable.lowering_digest());
    }

    std::array<MetalProgramInput, 2> malformed = inputs;
    malformed[0].value.extents = {6};
    const auto rejected = executable.execute(malformed);
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    if (const auto* report = std::get_if<CompatibilityReport>(&rejected))
        CHECK(report->code == CompatibilityError::RUNTIME_INPUT_INVALID);

    malformed = inputs;
    malformed[0].value.bits.front() = UINT64_MAX;
    CHECK(std::holds_alternative<CompatibilityReport>(
        executable.execute(malformed)));
    malformed = inputs;
    malformed[1].value_id = malformed[0].value_id;
    CHECK(std::holds_alternative<CompatibilityReport>(
        executable.execute(malformed)));
    CHECK(std::holds_alternative<CompatibilityReport>(
        executable.execute(std::span<const MetalProgramInput>(inputs).first(1))));
}

std::optional<VerifiedPhysicalProgramPackage> bound_contraction_package() {
    PhysicalProgram decoder;
    decoder.logical_rank = 2;
    decoder.planes = {{PhysicalPlaneStorage::External, 1, 0, 0}};
    decoder.policies.push_back({});
    decoder.instructions = {
        physical_instruction(PhysicalOpcode::Coordinate,
                             PhysicalValueType::Index, {}, 0),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 2),
        physical_instruction(PhysicalOpcode::IndexMultiply,
                             PhysicalValueType::Index, {0, 1}),
        physical_instruction(PhysicalOpcode::Coordinate,
                             PhysicalValueType::Index, {}, 1),
        physical_instruction(PhysicalOpcode::IndexAdd,
                             PhysicalValueType::Index, {2, 3}),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 32),
        physical_instruction(PhysicalOpcode::IndexMultiply,
                             PhysicalValueType::Index, {4, 5}),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {6}, 0, 0,
                             kNoPhysicalPolicy, 32),
        physical_instruction(PhysicalOpcode::BitsToF32,
                             PhysicalValueType::F32, {7}, 0,
                             kNoPhysicalPlane, 0),
    };
    decoder.result = 8;
    const auto canonical = canonicalize_physical_program(std::move(decoder));
    const auto* canonical_error =
        std::get_if<CompatibilityReport>(&canonical);
    CHECK_MSG(std::holds_alternative<PhysicalProgram>(canonical),
              "bound decoder canonical code=%u detail=%s",
              canonical_error ? static_cast<unsigned>(canonical_error->code) : 0,
              canonical_error ? canonical_error->detail.c_str() : "none");
    if (!std::holds_alternative<PhysicalProgram>(canonical)) return std::nullopt;
    const PhysicalProgram& physical = std::get<PhysicalProgram>(canonical);
    const auto wire = encode_physical_program(physical);
    const auto digest = physical_program_digest(physical);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
    CHECK(std::holds_alternative<PhysicalProgramDigest>(digest));
    if (!std::holds_alternative<std::vector<uint8_t>>(wire) ||
        !std::holds_alternative<PhysicalProgramDigest>(digest))
        return std::nullopt;

    char path[] = "/private/tmp/laplace-program-metal-bound-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd < 0) return std::nullopt;
    std::vector<uint8_t> bytes;
    for (float value : {1, 2, 3, 4, 5, 6}) {
        const uint32_t bits = std::bit_cast<uint32_t>(value);
        for (unsigned shift = 0; shift != 32; shift += 8)
            bytes.push_back(static_cast<uint8_t>(bits >> shift));
    }
    CHECK(write(fd, bytes.data(), bytes.size()) ==
          static_cast<ssize_t>(bytes.size()));
    CHECK(close(fd) == 0);
    auto artifacts = ArtifactSet::load_single_file(path);
    unlink(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return std::nullopt;
    auto view = std::get<ArtifactSet>(std::move(artifacts)).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return std::nullopt;
    ArtifactIndexInput index_input;
    index_input.artifacts.push_back(std::get<PackageView>(std::move(view)));
    auto index = ArtifactIndex::build(std::move(index_input));
    CHECK(std::holds_alternative<ArtifactIndex>(index));
    if (!std::holds_alternative<ArtifactIndex>(index)) return std::nullopt;

    auto semantic = verify_and_canonicalize_program(contraction_program());
    CHECK(std::holds_alternative<VerifiedProgram>(semantic));
    if (!std::holds_alternative<VerifiedProgram>(semantic)) return std::nullopt;
    PhysicalProgramRecord record{
        std::get<PhysicalProgramDigest>(digest),
        std::get<std::vector<uint8_t>>(wire), {ElementType::F32, {3, 2}}};
    PhysicalResourceBinding resource;
    resource.resource_id = 0;
    resource.program_digest = record.digest;
    resource.semantic_function_id = 9;
    resource.semantic_value_id = 11;
    resource.planes = {{0, ArtifactId{0}, 0, bytes.size()}};
    auto package = load_physical_program_package(
        std::get<ArtifactIndex>(std::move(index)),
        std::get<VerifiedProgram>(std::move(semantic)),
        std::span<const PhysicalProgramRecord>(&record, 1),
        std::span<const PhysicalResourceBinding>(&resource, 1));
    CHECK(std::holds_alternative<VerifiedPhysicalProgramPackage>(package));
    if (!std::holds_alternative<VerifiedPhysicalProgramPackage>(package))
        return std::nullopt;
    return std::get<VerifiedPhysicalProgramPackage>(std::move(package));
}

void test_bound_physical_contraction() {
    auto package = bound_contraction_package();
    CHECK(package.has_value());
    if (!package) return;
    auto compiled = compile_metal_program(*package);
    const auto* compile_error = std::get_if<CompatibilityReport>(&compiled);
    CHECK_MSG(std::holds_alternative<MetalProgramExecutable>(compiled),
              "bound contraction compile code=%u detail=%s",
              compile_error ? static_cast<unsigned>(compile_error->code) : 0,
              compile_error ? compile_error->detail.c_str() : "none");
    if (!std::holds_alternative<MetalProgramExecutable>(compiled)) return;
    const std::array<MetalProgramInput, 1> inputs = {
        MetalProgramInput{10, f32_value({2, 3}, {1, 2, 3, 4, 5, 6})}};
    const auto executed =
        std::get<MetalProgramExecutable>(std::move(compiled)).execute(inputs);
    CHECK(std::holds_alternative<MetalProgramResult>(executed));
    if (const auto* result = std::get_if<MetalProgramResult>(&executed)) {
        CHECK(result->exports ==
              std::vector<MetalProgramValue>{
                  f32_value({2, 2}, {22, 28, 49, 64})});
        CHECK(result->audit.command_buffers == 1);
        CHECK(result->audit.implicit_weight_copies == 0);
        CHECK(result->audit.explicit_upload_bytes == 24);
        CHECK(result->audit.zero_copy_plane_bytes == 24);
        CHECK(result->audit.persistent_plane_bytes == 24);
        CHECK(result->audit.explicit_download_bytes == 16);
    }
    ReferenceState state;
    const std::array<ReferenceInput, 1> reference_inputs = {
        ReferenceInput{10, reference_value(inputs[0].value)}};
    const auto reference = execute_reference(*package, state, reference_inputs);
    CHECK(std::holds_alternative<ReferenceResult>(reference));
    if (const auto* metal = std::get_if<MetalProgramResult>(&executed)) {
        const auto* expected = std::get_if<ReferenceResult>(&reference);
        CHECK(expected != nullptr);
        if (expected) CHECK(reference_value(metal->exports[0]) == expected->exports[0]);
    }

    auto rejected_compile = compile_metal_program(*package);
    CHECK(std::holds_alternative<MetalProgramExecutable>(rejected_compile));
    if (std::holds_alternative<MetalProgramExecutable>(rejected_compile)) {
        const std::array<MetalProgramInput, 2> extra_inputs = {
            inputs[0],
            MetalProgramInput{11, f32_value({3, 2}, {1, 2, 3, 4, 5, 6})}};
        CHECK(std::holds_alternative<CompatibilityReport>(
            std::get<MetalProgramExecutable>(std::move(rejected_compile))
                .execute(extra_inputs)));
    }
}

void test_transpose_and_reduction() {
    const std::array<MetalProgramInput, 2> transpose_inputs = {
        MetalProgramInput{10, f32_value({3, 2}, {1, 2, 3, 4, 5, 6})},
        MetalProgramInput{11, f32_value({3}, {10, 20, 30})}};
    check_matches_reference(transpose_broadcast_program(), transpose_inputs);

    const std::array<MetalProgramInput, 1> reduction_inputs = {
        MetalProgramInput{10, f32_value({2, 3}, {1, 2, 3, 4, 5, 6})}};
    check_matches_reference(reduction_program(), reduction_inputs);

    const std::array<MetalProgramInput, 1> padded_inputs = {
        MetalProgramInput{10, f32_value({2}, {7, 9})}};
    check_matches_reference(zero_padded_program(), padded_inputs);

    const auto scalar = verify_and_canonicalize_program(scalar_program());
    CHECK(std::holds_alternative<VerifiedProgram>(scalar));
    if (const auto* program = std::get_if<VerifiedProgram>(&scalar)) {
        const auto rejected = compile_metal_program(*program);
        CHECK(std::holds_alternative<CompatibilityReport>(rejected));
        if (const auto* report = std::get_if<CompatibilityReport>(&rejected))
            CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
    }
}

PhysicalInstruction physical_instruction(
    PhysicalOpcode opcode, PhysicalValueType result,
    std::initializer_list<uint32_t> operands, uint64_t immediate,
    uint16_t plane, uint16_t policy, uint8_t bit_width) {
    PhysicalInstruction instruction;
    instruction.opcode = opcode;
    instruction.result_type = result;
    size_t index = 0;
    for (uint32_t operand : operands) instruction.operands[index++] = operand;
    instruction.immediate = immediate;
    instruction.plane = plane;
    instruction.policy = policy;
    instruction.bit_width = bit_width;
    return instruction;
}

void test_generated_physical_decoder() {
    PhysicalProgram source;
    source.logical_rank = 1;
    source.planes = {
        {PhysicalPlaneStorage::External, 1, 0, 0},
        {PhysicalPlaneStorage::External, 1, 0, 0},
    };
    source.policies.push_back({});
    source.instructions = {
        physical_instruction(PhysicalOpcode::Coordinate,
                             PhysicalValueType::Index, {}, 0),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 3),
        physical_instruction(PhysicalOpcode::IndexMultiply,
                             PhysicalValueType::Index, {0, 1}),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {2}, 0, 0,
                             kNoPhysicalPolicy, 3),
        physical_instruction(PhysicalOpcode::ConstU32,
                             PhysicalValueType::U32, {}, 3),
        physical_instruction(PhysicalOpcode::U32And,
                             PhysicalValueType::U32, {3, 4}),
        physical_instruction(PhysicalOpcode::IndexFromU32,
                             PhysicalValueType::Index, {5}),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 32),
        physical_instruction(PhysicalOpcode::IndexMultiply,
                             PhysicalValueType::Index, {6, 7}),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {8}, 0, 1,
                             kNoPhysicalPolicy, 32),
        physical_instruction(PhysicalOpcode::BitsToF32,
                             PhysicalValueType::F32, {9}, 0,
                             kNoPhysicalPlane, 0),
    };
    source.result = 10;
    const auto packed =
        std::make_shared<const std::vector<uint8_t>>(
            std::initializer_list<uint8_t>{0x88, 0x16});
    std::vector<uint8_t> table;
    for (float value : {1.0f, -2.0f, 3.5f, 8.0f}) {
        const uint32_t bits = std::bit_cast<uint32_t>(value);
        for (unsigned shift = 0; shift != 32; shift += 8)
            table.push_back(static_cast<uint8_t>(bits >> shift));
    }
    const auto codebook =
        std::make_shared<const std::vector<uint8_t>>(std::move(table));
    const std::array<PhysicalPlaneBinding, 2> bindings = {
        PhysicalPlaneBinding{0, packed, 0, packed->size()},
        PhysicalPlaneBinding{1, codebook, 0, codebook->size()},
    };
    const PhysicalProgramResult canonical =
        canonicalize_physical_program(std::move(source));
    CHECK(std::holds_alternative<PhysicalProgram>(canonical));
    if (!std::holds_alternative<PhysicalProgram>(canonical)) return;
    const auto verified = verify_physical_program(
        std::get<PhysicalProgram>(canonical), bindings,
        {ElementType::F32, {5}});
    const auto* verify_error = std::get_if<CompatibilityReport>(&verified);
    CHECK_MSG(std::holds_alternative<VerifiedPhysicalProgram>(verified),
              "physical verify code=%u detail=%s",
              verify_error ? static_cast<unsigned>(verify_error->code) : 0,
              verify_error ? verify_error->detail.c_str() : "none");
    if (!std::holds_alternative<VerifiedPhysicalProgram>(verified)) return;
    const VerifiedPhysicalProgram& physical =
        std::get<VerifiedPhysicalProgram>(verified);
    auto compiled = compile_metal_physical_program(physical);
    CHECK(std::holds_alternative<MetalPhysicalProgramExecutable>(compiled));
    if (!std::holds_alternative<MetalPhysicalProgramExecutable>(compiled)) return;
    MetalPhysicalProgramExecutable executable =
        std::get<MetalPhysicalProgramExecutable>(std::move(compiled));
    const auto executed = executable.execute();
    CHECK(std::holds_alternative<MetalPhysicalProgramResult>(executed));
    if (const auto* result =
            std::get_if<MetalPhysicalProgramResult>(&executed)) {
        CHECK(result->value.type == ElementType::F32);
        CHECK(result->value.extents == std::vector<uint64_t>{5});
        CHECK(result->value.bits.size() == 5);
        for (uint64_t coordinate = 0; coordinate < 5; ++coordinate) {
            const auto expected = interpret_physical_value(
                physical, std::span<const uint64_t>(&coordinate, 1));
            CHECK(std::holds_alternative<ScalarValue>(expected));
            if (const auto* scalar = std::get_if<ScalarValue>(&expected))
                CHECK(result->value.bits[coordinate] == scalar->bits);
        }
        CHECK(result->audit.physical_program_digest == physical.digest());
        CHECK(result->audit.lowering_digest == executable.lowering_digest());
        CHECK(result->audit.explicit_upload_bytes == 18);
        CHECK(result->audit.explicit_download_bytes == 20);
        CHECK(result->audit.command_buffers == 1);
        CHECK(result->audit.implicit_weight_copies == 0);
    }
}

std::optional<VerifiedPhysicalProgram> verified_physical(
    PhysicalProgram source, std::span<const PhysicalPlaneBinding> bindings,
    LogicalTensorType logical) {
    const PhysicalProgramResult canonical =
        canonicalize_physical_program(std::move(source));
    CHECK(std::holds_alternative<PhysicalProgram>(canonical));
    if (!std::holds_alternative<PhysicalProgram>(canonical)) return std::nullopt;
    auto verified = verify_physical_program(
        std::get<PhysicalProgram>(canonical), bindings, logical);
    const auto* error = std::get_if<CompatibilityReport>(&verified);
    CHECK_MSG(std::holds_alternative<VerifiedPhysicalProgram>(verified),
              "physical verify code=%u detail=%s",
              error ? static_cast<unsigned>(error->code) : 0,
              error ? error->detail.c_str() : "none");
    if (!std::holds_alternative<VerifiedPhysicalProgram>(verified))
        return std::nullopt;
    return std::get<VerifiedPhysicalProgram>(std::move(verified));
}

void check_physical_metal(const VerifiedPhysicalProgram& physical) {
    auto compiled = compile_metal_physical_program(physical);
    const auto* compile_error = std::get_if<CompatibilityReport>(&compiled);
    CHECK_MSG(std::holds_alternative<MetalPhysicalProgramExecutable>(compiled),
              "physical compile code=%u detail=%s",
              compile_error ? static_cast<unsigned>(compile_error->code) : 0,
              compile_error ? compile_error->detail.c_str() : "none");
    if (!std::holds_alternative<MetalPhysicalProgramExecutable>(compiled)) return;
    const auto executed =
        std::get<MetalPhysicalProgramExecutable>(std::move(compiled)).execute();
    const auto* execute_error = std::get_if<CompatibilityReport>(&executed);
    CHECK_MSG(std::holds_alternative<MetalPhysicalProgramResult>(executed),
              "physical execute code=%u detail=%s",
              execute_error ? static_cast<unsigned>(execute_error->code) : 0,
              execute_error ? execute_error->detail.c_str() : "none");
    const auto* result = std::get_if<MetalPhysicalProgramResult>(&executed);
    if (!result) return;
    CHECK(result->value.type == physical.logical_type().element_type);
    CHECK(result->value.extents == physical.logical_type().extents);
    size_t count = 1;
    for (uint64_t extent : physical.logical_type().extents)
        count *= static_cast<size_t>(extent);
    CHECK(result->value.bits.size() == count);
    std::vector<uint64_t> coordinate(physical.logical_type().extents.size());
    for (size_t linear = 0; linear < count; ++linear) {
        size_t remainder = linear;
        for (size_t axis = coordinate.size(); axis != 0; --axis) {
            const size_t extent = static_cast<size_t>(
                physical.logical_type().extents[axis - 1]);
            coordinate[axis - 1] = remainder % extent;
            remainder /= extent;
        }
        const auto expected = interpret_physical_value(physical, coordinate);
        CHECK(std::holds_alternative<ScalarValue>(expected));
        if (const auto* scalar = std::get_if<ScalarValue>(&expected))
            CHECK(result->value.bits[linear] == scalar->bits);
    }
    CHECK(result->audit.physical_program_digest == physical.digest());
    CHECK(result->audit.command_buffers == 1);
    CHECK(result->audit.implicit_weight_copies == 0);
}

void test_generated_physical_bit_orders_and_inline_plane() {
    PhysicalProgram msb;
    msb.logical_rank = 1;
    msb.planes = {{PhysicalPlaneStorage::External, 1, 0, 0}};
    msb.instructions = {
        physical_instruction(PhysicalOpcode::Coordinate,
                             PhysicalValueType::Index, {}, 0),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 4),
        physical_instruction(PhysicalOpcode::IndexMultiply,
                             PhysicalValueType::Index, {0, 1}),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {2}, 0, 0,
                             kNoPhysicalPolicy, 4),
    };
    msb.instructions.back().bit_order = PhysicalBitOrder::Msb0Big;
    msb.result = 3;
    const auto packed = std::make_shared<const std::vector<uint8_t>>(
        std::initializer_list<uint8_t>{0x12, 0x34});
    const PhysicalPlaneBinding msb_binding{0, packed, 0, packed->size()};
    if (const auto verified = verified_physical(
            msb, std::span<const PhysicalPlaneBinding>(&msb_binding, 1),
            {ElementType::U32, {4}}))
        check_physical_metal(*verified);

    PhysicalProgram scaled;
    scaled.logical_rank = 1;
    scaled.planes = {
        {PhysicalPlaneStorage::External, 1, 0, 0},
        {PhysicalPlaneStorage::Inline, 1, 0, 2},
    };
    scaled.inline_bytes = {0x00, 0x38};
    scaled.policies.push_back({});
    scaled.instructions = {
        physical_instruction(PhysicalOpcode::Coordinate,
                             PhysicalValueType::Index, {}, 0),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 8),
        physical_instruction(PhysicalOpcode::IndexMultiply,
                             PhysicalValueType::Index, {0, 1}),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {2}, 0, 0,
                             kNoPhysicalPolicy, 8),
        physical_instruction(PhysicalOpcode::SignExtend,
                             PhysicalValueType::I32, {3}, 0,
                             kNoPhysicalPlane, kNoPhysicalPolicy, 8),
        physical_instruction(PhysicalOpcode::I32ToF32,
                             PhysicalValueType::F32, {4}, 0,
                             kNoPhysicalPlane, 0),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 0),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {6}, 0, 1,
                             kNoPhysicalPolicy, 16),
        physical_instruction(PhysicalOpcode::F16ToF32,
                             PhysicalValueType::F32, {7}, 0,
                             kNoPhysicalPlane, 0),
        physical_instruction(PhysicalOpcode::F32Multiply,
                             PhysicalValueType::F32, {5, 8}, 0,
                             kNoPhysicalPlane, 0),
    };
    scaled.result = 9;
    const auto signed_values = std::make_shared<const std::vector<uint8_t>>(
        std::initializer_list<uint8_t>{0xfe, 0x04, 0xf8, 0x10});
    const PhysicalPlaneBinding signed_binding{
        0, signed_values, 0, signed_values->size()};
    if (const auto verified = verified_physical(
            scaled,
            std::span<const PhysicalPlaneBinding>(&signed_binding, 1),
            {ElementType::F32, {4}}))
        check_physical_metal(*verified);
}

void test_generated_physical_numeric_policies() {
    PhysicalProgram conversion;
    conversion.logical_rank = 1;
    conversion.planes = {{PhysicalPlaneStorage::External, 1, 0, 0}};
    PhysicalNumericPolicy integer_policy;
    integer_policy.integer_overflow = PhysicalIntegerOverflow::Saturate;
    integer_policy.nan = PhysicalNanPolicy::Reject;
    integer_policy.infinity = PhysicalInfinityPolicy::Reject;
    conversion.policies = {{}, integer_policy};
    conversion.instructions = {
        physical_instruction(PhysicalOpcode::Coordinate,
                             PhysicalValueType::Index, {}, 0),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 32),
        physical_instruction(PhysicalOpcode::IndexMultiply,
                             PhysicalValueType::Index, {0, 1}),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {2}, 0, 0,
                             kNoPhysicalPolicy, 32),
        physical_instruction(PhysicalOpcode::BitsToF32,
                             PhysicalValueType::F32, {3}, 0,
                             kNoPhysicalPlane, 0),
        physical_instruction(PhysicalOpcode::F32ToU32,
                             PhysicalValueType::U32, {4}, 0,
                             kNoPhysicalPlane, 1),
    };
    conversion.result = 5;
    std::vector<uint8_t> source;
    for (float value : {2.5f, 3.5f, -3.5f, 4294967296.0f}) {
        const uint32_t bits = std::bit_cast<uint32_t>(value);
        for (unsigned shift = 0; shift != 32; shift += 8)
            source.push_back(static_cast<uint8_t>(bits >> shift));
    }
    const auto values =
        std::make_shared<const std::vector<uint8_t>>(std::move(source));
    const PhysicalPlaneBinding binding{0, values, 0, values->size()};
    if (const auto verified = verified_physical(
            conversion, std::span<const PhysicalPlaneBinding>(&binding, 1),
            {ElementType::U32, {4}}))
        check_physical_metal(*verified);

    PhysicalProgram rejected;
    rejected.planes = {{PhysicalPlaneStorage::External, 1, 0, 0}};
    PhysicalNumericPolicy reject_nan;
    reject_nan.nan = PhysicalNanPolicy::Reject;
    rejected.policies = {reject_nan};
    rejected.instructions = {
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 0),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {0}, 0, 0,
                             kNoPhysicalPolicy, 32),
        physical_instruction(PhysicalOpcode::BitsToF32,
                             PhysicalValueType::F32, {1}, 0,
                             kNoPhysicalPlane, 0),
    };
    rejected.result = 2;
    const auto nan = std::make_shared<const std::vector<uint8_t>>(
        std::initializer_list<uint8_t>{0x01, 0x00, 0xc0, 0x7f});
    const PhysicalPlaneBinding nan_binding{0, nan, 0, nan->size()};
    const auto verified = verified_physical(
        rejected, std::span<const PhysicalPlaneBinding>(&nan_binding, 1),
        {ElementType::F32, {}});
    if (verified) {
        auto compiled = compile_metal_physical_program(*verified);
        CHECK(std::holds_alternative<MetalPhysicalProgramExecutable>(compiled));
        if (std::holds_alternative<MetalPhysicalProgramExecutable>(compiled)) {
            const auto result =
                std::get<MetalPhysicalProgramExecutable>(std::move(compiled)).execute();
            CHECK(std::holds_alternative<CompatibilityReport>(result));
            if (const auto* report = std::get_if<CompatibilityReport>(&result))
                CHECK(report->code ==
                      CompatibilityError::RUNTIME_NUMERICAL_FAILURE);
        }
    }
}

void test_generated_physical_index_and_float_algebra() {
    PhysicalProgram indexed;
    indexed.logical_rank = 1;
    indexed.planes = {{PhysicalPlaneStorage::External, 1, 0, 0}};
    indexed.instructions = {
        physical_instruction(PhysicalOpcode::Coordinate,
                             PhysicalValueType::Index, {}, 0),
        physical_instruction(PhysicalOpcode::IndexDivideConstant,
                             PhysicalValueType::Index, {0}, 2),
        physical_instruction(PhysicalOpcode::IndexRemainderConstant,
                             PhysicalValueType::Index, {0}, 2),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 2),
        physical_instruction(PhysicalOpcode::IndexMultiply,
                             PhysicalValueType::Index, {1, 3}),
        physical_instruction(PhysicalOpcode::IndexAdd,
                             PhysicalValueType::Index, {4, 2}),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 0),
        physical_instruction(PhysicalOpcode::IndexSubtract,
                             PhysicalValueType::Index, {5, 6}),
        physical_instruction(PhysicalOpcode::IndexCeilDivideConstant,
                             PhysicalValueType::Index, {7}, 1),
        physical_instruction(PhysicalOpcode::Extent,
                             PhysicalValueType::Index, {}, 0),
        physical_instruction(PhysicalOpcode::IndexLess,
                             PhysicalValueType::Predicate, {8, 9}),
        physical_instruction(PhysicalOpcode::Select,
                             PhysicalValueType::Index, {10, 8, 6}),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 8),
        physical_instruction(PhysicalOpcode::IndexMultiply,
                             PhysicalValueType::Index, {11, 12}),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {13}, 0, 0,
                             kNoPhysicalPolicy, 8),
    };
    indexed.result = 14;
    const auto bytes = std::make_shared<const std::vector<uint8_t>>(
        std::initializer_list<uint8_t>{10, 11, 12, 13, 14, 15});
    const PhysicalPlaneBinding binding{0, bytes, 0, bytes->size()};
    if (const auto verified = verified_physical(
            indexed, std::span<const PhysicalPlaneBinding>(&binding, 1),
            {ElementType::U32, {6}}))
        check_physical_metal(*verified);

    PhysicalProgram algebra;
    algebra.planes = {{PhysicalPlaneStorage::External, 1, 0, 0}};
    PhysicalNumericPolicy fused;
    fused.contraction = PhysicalContractionPolicy::Fused;
    algebra.policies = {{}, fused};
    for (uint64_t bit_address : {uint64_t{0}, uint64_t{32}, uint64_t{64},
                                 uint64_t{96}, uint64_t{128}}) {
        const uint32_t address = algebra.instructions.size();
        algebra.instructions.push_back(physical_instruction(
            PhysicalOpcode::ConstIndex, PhysicalValueType::Index, {},
            bit_address));
        algebra.instructions.push_back(physical_instruction(
            PhysicalOpcode::LoadBits, PhysicalValueType::U32, {address}, 0,
            0, kNoPhysicalPolicy, 32));
        algebra.instructions.push_back(physical_instruction(
            PhysicalOpcode::BitsToF32, PhysicalValueType::F32,
            {address + 1}, 0, kNoPhysicalPlane, 0));
    }
    algebra.instructions.push_back(physical_instruction(
        PhysicalOpcode::F32Add, PhysicalValueType::F32, {2, 5}, 0,
        kNoPhysicalPlane, 0));
    algebra.instructions.push_back(physical_instruction(
        PhysicalOpcode::F32Subtract, PhysicalValueType::F32, {15, 8}, 0,
        kNoPhysicalPlane, 0));
    algebra.instructions.push_back(physical_instruction(
        PhysicalOpcode::F32Multiply, PhysicalValueType::F32, {16, 5}, 0,
        kNoPhysicalPlane, 0));
    algebra.instructions.push_back(physical_instruction(
        PhysicalOpcode::F32Fma, PhysicalValueType::F32, {17, 5, 8}, 0,
        kNoPhysicalPlane, 1));
    algebra.instructions.push_back(physical_instruction(
        PhysicalOpcode::F32Negate, PhysicalValueType::F32, {18}, 0,
        kNoPhysicalPlane, 0));
    algebra.instructions.push_back(physical_instruction(
        PhysicalOpcode::F32Clamp, PhysicalValueType::F32, {19, 11, 14}, 0,
        kNoPhysicalPlane, 0));
    algebra.result = 20;
    std::vector<uint8_t> algebra_bytes;
    for (float value : {1.5f, 2.0f, -1.0f, -10.0f, 10.0f}) {
        const uint32_t bits = std::bit_cast<uint32_t>(value);
        for (unsigned shift = 0; shift != 32; shift += 8)
            algebra_bytes.push_back(static_cast<uint8_t>(bits >> shift));
    }
    const auto algebra_owner =
        std::make_shared<const std::vector<uint8_t>>(std::move(algebra_bytes));
    const PhysicalPlaneBinding algebra_binding{
        0, algebra_owner, 0, algebra_owner->size()};
    if (const auto verified = verified_physical(
            algebra,
            std::span<const PhysicalPlaneBinding>(&algebra_binding, 1),
            {ElementType::F32, {}}))
        check_physical_metal(*verified);
}

void test_generated_physical_bitwise_and_narrowing() {
    PhysicalProgram bitwise;
    bitwise.planes = {{PhysicalPlaneStorage::Inline, 1, 0, 1}};
    bitwise.inline_bytes = {0x0f};
    bitwise.instructions = {
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 0),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {0}, 0, 0,
                             kNoPhysicalPolicy, 8),
        physical_instruction(PhysicalOpcode::ConstU32,
                             PhysicalValueType::U32, {}, 0xf0),
        physical_instruction(PhysicalOpcode::U32Or,
                             PhysicalValueType::U32, {1, 2}),
        physical_instruction(PhysicalOpcode::ConstU32,
                             PhysicalValueType::U32, {}, 0x33),
        physical_instruction(PhysicalOpcode::U32Xor,
                             PhysicalValueType::U32, {3, 4}),
        physical_instruction(PhysicalOpcode::ConstU32,
                             PhysicalValueType::U32, {}, 0x0f),
        physical_instruction(PhysicalOpcode::U32And,
                             PhysicalValueType::U32, {5, 6}),
        physical_instruction(PhysicalOpcode::U32ShiftLeftConstant,
                             PhysicalValueType::U32, {7}, 4),
        physical_instruction(PhysicalOpcode::U32ShiftRightConstant,
                             PhysicalValueType::U32, {8}, 2),
    };
    bitwise.result = 9;
    if (const auto verified = verified_physical(
            bitwise, {}, {ElementType::U32, {}}))
        check_physical_metal(*verified);

    PhysicalProgram narrowing;
    narrowing.planes = {{PhysicalPlaneStorage::Inline, 1, 0, 2}};
    narrowing.inline_bytes = {0xc0, 0x3f};
    PhysicalNumericPolicy integer_policy;
    integer_policy.nan = PhysicalNanPolicy::Reject;
    integer_policy.infinity = PhysicalInfinityPolicy::Reject;
    narrowing.policies = {{}, integer_policy};
    narrowing.instructions = {
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 0),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {0}, 0, 0,
                             kNoPhysicalPolicy, 16),
        physical_instruction(PhysicalOpcode::Bf16ToF32,
                             PhysicalValueType::F32, {1}, 0,
                             kNoPhysicalPlane, 0),
        physical_instruction(PhysicalOpcode::ConstU32,
                             PhysicalValueType::U32, {}, 2),
        physical_instruction(PhysicalOpcode::U32ToF32,
                             PhysicalValueType::F32, {3}, 0,
                             kNoPhysicalPlane, 0),
        physical_instruction(PhysicalOpcode::F32Multiply,
                             PhysicalValueType::F32, {2, 4}, 0,
                             kNoPhysicalPlane, 0),
        physical_instruction(PhysicalOpcode::ConstI32,
                             PhysicalValueType::I32, {}, UINT32_MAX),
        physical_instruction(PhysicalOpcode::I32ToF32,
                             PhysicalValueType::F32, {6}, 0,
                             kNoPhysicalPlane, 0),
        physical_instruction(PhysicalOpcode::F32Add,
                             PhysicalValueType::F32, {5, 7}, 0,
                             kNoPhysicalPlane, 0),
        physical_instruction(PhysicalOpcode::ConstF32Bits,
                             PhysicalValueType::F32, {},
                             std::bit_cast<uint32_t>(1.0f)),
        physical_instruction(PhysicalOpcode::F32Add,
                             PhysicalValueType::F32, {8, 9}, 0,
                             kNoPhysicalPlane, 0),
        physical_instruction(PhysicalOpcode::F32ToI32,
                             PhysicalValueType::I32, {10}, 0,
                             kNoPhysicalPlane, 1),
    };
    narrowing.result = 11;
    if (const auto verified = verified_physical(
            narrowing, {}, {ElementType::I32, {}}))
        check_physical_metal(*verified);

    PhysicalProgram selected;
    selected.planes = {{PhysicalPlaneStorage::Inline, 1, 0, 1}};
    selected.inline_bytes = {0};
    selected.instructions = {
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 0),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {0}, 0, 0,
                             kNoPhysicalPolicy, 1),
        physical_instruction(PhysicalOpcode::IndexFromU32,
                             PhysicalValueType::Index, {1}),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 1),
        physical_instruction(PhysicalOpcode::IndexLess,
                             PhysicalValueType::Predicate, {2, 3}),
        physical_instruction(PhysicalOpcode::ConstF32Bits,
                             PhysicalValueType::F32, {}, 0x7fa12345u),
        physical_instruction(PhysicalOpcode::ConstF32Bits,
                             PhysicalValueType::F32, {},
                             std::bit_cast<uint32_t>(1.0f)),
        physical_instruction(PhysicalOpcode::Select,
                             PhysicalValueType::F32, {4, 5, 6}),
    };
    selected.result = 7;
    if (const auto verified = verified_physical(
            selected, {}, {ElementType::F32, {}}))
        check_physical_metal(*verified);
}

void test_generated_physical_funnel_shift() {
    PhysicalProgram program;
    program.logical_rank = 1;
    program.planes = {{PhysicalPlaneStorage::External, 4, 0, 0}};
    program.instructions = {
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {0}, 0, 0,
                             kNoPhysicalPolicy, 32),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 32),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {2}, 0, 0,
                             kNoPhysicalPolicy, 32),
        physical_instruction(PhysicalOpcode::Coordinate,
                             PhysicalValueType::Index),
        physical_instruction(PhysicalOpcode::U32FunnelShiftRight,
                             PhysicalValueType::U32, {1, 3, 4}),
    };
    program.result = 5;
    std::vector<uint8_t> storage;
    for (uint32_t value : {0x89abcdefu, 0x01234567u}) {
        for (unsigned shift = 0; shift != 32; shift += 8)
            storage.push_back(static_cast<uint8_t>(value >> shift));
    }
    const auto owner =
        std::make_shared<const std::vector<uint8_t>>(std::move(storage));
    const PhysicalPlaneBinding binding{0, owner, 0, owner->size()};
    if (const auto verified = verified_physical(
            program, std::span<const PhysicalPlaneBinding>(&binding, 1),
            {ElementType::U32, {64}}))
        check_physical_metal(*verified);
}

void test_generated_physical_circular_window() {
    PhysicalProgram program;
    program.logical_rank = 1;
    program.planes = {{PhysicalPlaneStorage::External, 4, 0, 0}};
    program.instructions = {
        physical_instruction(PhysicalOpcode::Coordinate,
                             PhysicalValueType::Index),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 3),
        physical_instruction(PhysicalOpcode::IndexMultiply,
                             PhysicalValueType::Index, {0, 1}),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 755),
        physical_instruction(PhysicalOpcode::IndexAdd,
                             PhysicalValueType::Index, {2, 3}),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 16),
        physical_instruction(PhysicalOpcode::IndexAdd,
                             PhysicalValueType::Index, {4, 5}),
        physical_instruction(PhysicalOpcode::IndexDivideConstant,
                             PhysicalValueType::Index, {4}, 32),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 1),
        physical_instruction(PhysicalOpcode::IndexSubtract,
                             PhysicalValueType::Index, {6, 8}),
        physical_instruction(PhysicalOpcode::IndexDivideConstant,
                             PhysicalValueType::Index, {9}, 32),
        physical_instruction(PhysicalOpcode::IndexRemainderConstant,
                             PhysicalValueType::Index, {7}, 24),
        physical_instruction(PhysicalOpcode::IndexRemainderConstant,
                             PhysicalValueType::Index, {10}, 24),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 32),
        physical_instruction(PhysicalOpcode::IndexMultiply,
                             PhysicalValueType::Index, {11, 13}),
        physical_instruction(PhysicalOpcode::IndexMultiply,
                             PhysicalValueType::Index, {12, 13}),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {14}, 0, 0,
                             kNoPhysicalPolicy, 32),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {15}, 0, 0,
                             kNoPhysicalPolicy, 32),
        physical_instruction(PhysicalOpcode::IndexRemainderConstant,
                             PhysicalValueType::Index, {6}, 32),
        physical_instruction(PhysicalOpcode::IndexSubtract,
                             PhysicalValueType::Index, {13, 18}),
        physical_instruction(PhysicalOpcode::IndexRemainderConstant,
                             PhysicalValueType::Index, {19}, 32),
        physical_instruction(PhysicalOpcode::U32FunnelShiftRight,
                             PhysicalValueType::U32, {17, 16, 20}),
        physical_instruction(PhysicalOpcode::ConstU32,
                             PhysicalValueType::U32, {}, 0xffff),
        physical_instruction(PhysicalOpcode::U32And,
                             PhysicalValueType::U32, {21, 22}),
    };
    program.result = 23;
    std::vector<uint8_t> storage;
    for (uint32_t index = 0; index < 24; ++index) {
        const uint32_t word = 0x9e3779b9u * (index + 1u) ^ 0xa5a5a5a5u;
        for (unsigned shift = 0; shift != 32; shift += 8)
            storage.push_back(static_cast<uint8_t>(word >> shift));
    }
    const auto owner =
        std::make_shared<const std::vector<uint8_t>>(std::move(storage));
    const PhysicalPlaneBinding binding{0, owner, 0, owner->size()};
    if (const auto verified = verified_physical(
            program, std::span<const PhysicalPlaneBinding>(&binding, 1),
            {ElementType::U32, {256}}))
        check_physical_metal(*verified);
}

void test_generated_physical_procedural_codebook() {
    PhysicalProgram program;
    program.logical_rank = 1;
    program.planes = {{PhysicalPlaneStorage::External, 2, 0, 0}};
    PhysicalNumericPolicy fused;
    fused.contraction = PhysicalContractionPolicy::Fused;
    program.policies = {{}, fused};
    program.instructions = {
        physical_instruction(PhysicalOpcode::Coordinate,
                             PhysicalValueType::Index),
        physical_instruction(PhysicalOpcode::ConstIndex,
                             PhysicalValueType::Index, {}, 16),
        physical_instruction(PhysicalOpcode::IndexMultiply,
                             PhysicalValueType::Index, {0, 1}),
        physical_instruction(PhysicalOpcode::LoadBits,
                             PhysicalValueType::U32, {2}, 0, 0,
                             kNoPhysicalPolicy, 16),
        physical_instruction(PhysicalOpcode::ConstU32,
                             PhysicalValueType::U32, {}, 0x83dcd12d),
        physical_instruction(PhysicalOpcode::U32Multiply,
                             PhysicalValueType::U32, {3, 4}),
        physical_instruction(PhysicalOpcode::ConstU32,
                             PhysicalValueType::U32, {}, 0xff),
        physical_instruction(PhysicalOpcode::U32And,
                             PhysicalValueType::U32, {5, 6}),
        physical_instruction(PhysicalOpcode::U32ShiftRightConstant,
                             PhysicalValueType::U32, {5}, 8),
        physical_instruction(PhysicalOpcode::U32And,
                             PhysicalValueType::U32, {8, 6}),
        physical_instruction(PhysicalOpcode::U32ShiftRightConstant,
                             PhysicalValueType::U32, {5}, 16),
        physical_instruction(PhysicalOpcode::U32And,
                             PhysicalValueType::U32, {10, 6}),
        physical_instruction(PhysicalOpcode::U32ShiftRightConstant,
                             PhysicalValueType::U32, {5}, 24),
        physical_instruction(PhysicalOpcode::U32And,
                             PhysicalValueType::U32, {12, 6}),
        physical_instruction(PhysicalOpcode::U32Add,
                             PhysicalValueType::U32, {7, 9}),
        physical_instruction(PhysicalOpcode::U32Add,
                             PhysicalValueType::U32, {11, 13}),
        physical_instruction(PhysicalOpcode::U32Add,
                             PhysicalValueType::U32, {14, 15}),
        physical_instruction(PhysicalOpcode::ConstU32,
                             PhysicalValueType::U32, {}, 0x6400),
        physical_instruction(PhysicalOpcode::U32Add,
                             PhysicalValueType::U32, {16, 17}),
        physical_instruction(PhysicalOpcode::U32ToF32,
                             PhysicalValueType::F32, {18}, 0,
                             kNoPhysicalPlane, 0),
        physical_instruction(PhysicalOpcode::F32RoundToF16,
                             PhysicalValueType::F32, {19}, 0,
                             kNoPhysicalPlane, 0),
        physical_instruction(PhysicalOpcode::ConstU32,
                             PhysicalValueType::U32, {}, 0x1eee),
        physical_instruction(PhysicalOpcode::F16ToF32,
                             PhysicalValueType::F32, {21}, 0,
                             kNoPhysicalPlane, 0),
        physical_instruction(PhysicalOpcode::ConstU32,
                             PhysicalValueType::U32, {}, 0xc931),
        physical_instruction(PhysicalOpcode::F16ToF32,
                             PhysicalValueType::F32, {23}, 0,
                             kNoPhysicalPlane, 0),
        physical_instruction(PhysicalOpcode::F32Fma,
                             PhysicalValueType::F32, {20, 22, 24}, 0,
                             kNoPhysicalPlane, 1),
        physical_instruction(PhysicalOpcode::F32RoundToF16,
                             PhysicalValueType::F32, {25}, 0,
                             kNoPhysicalPlane, 0),
    };
    program.result = 26;
    const std::array<uint16_t, 7> indices = {
        0, 1, 2, 17, 0x1234, 0x8000, 0xffff};
    std::vector<uint8_t> storage;
    for (uint16_t value : indices) {
        storage.push_back(static_cast<uint8_t>(value));
        storage.push_back(static_cast<uint8_t>(value >> 8));
    }
    const auto owner =
        std::make_shared<const std::vector<uint8_t>>(std::move(storage));
    const PhysicalPlaneBinding binding{0, owner, 0, owner->size()};
    if (const auto verified = verified_physical(
            program, std::span<const PhysicalPlaneBinding>(&binding, 1),
            {ElementType::F32, {indices.size()}}))
        check_physical_metal(*verified);
}

} // namespace

int main() {
    test_generic_f32_algebra_metal();
    test_data_dependent_tensor_index_metal();
    test_dependent_tensor_stages_one_command_buffer();
    test_bound_dependent_stages_one_command_buffer();
    test_transactional_state_publication();
    test_contraction();
    test_bound_physical_contraction();
    test_transpose_and_reduction();
    test_generated_physical_decoder();
    test_generated_physical_bit_orders_and_inline_plane();
    test_generated_physical_numeric_policies();
    test_generated_physical_index_and_float_algebra();
    test_generated_physical_bitwise_and_narrowing();
    test_generated_physical_funnel_shift();
    test_generated_physical_circular_window();
    test_generated_physical_procedural_codebook();
    return test_summary("test_program_metal");
}
