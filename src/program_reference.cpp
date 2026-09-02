#include "program_reference.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace Laplace {
namespace {

constexpr size_t kMaximumReferenceTensorIterations = size_t{1} << 24;

CompatibilityReport reference_error(CompatibilityError code, const char* detail) {
    return compatibility_report(code, detail);
}

bool scalar_type_supported(ElementType type) noexcept {
    return type == ElementType::U32 || type == ElementType::I32 ||
           type == ElementType::U64 || type == ElementType::F32;
}

bool shape_for(const ValueType& type, std::vector<uint64_t>* extents) {
    if (!extents || !scalar_type_supported(type.element_type)) return false;
    extents->clear();
    for (const DimensionExpr& dimension : type.dimensions) {
        if (dimension.expression != DimensionExpression::Constant ||
            dimension.value == 0)
            return false;
        extents->push_back(dimension.value);
    }
    return true;
}

bool element_count(std::span<const uint64_t> extents, size_t* count) {
    if (!count) return false;
    size_t result = 1;
    for (uint64_t extent : extents) {
        if (extent == 0 || extent > std::numeric_limits<size_t>::max() / result)
            return false;
        result *= static_cast<size_t>(extent);
    }
    *count = result;
    return true;
}

ReferenceValue scalar_value(ElementType type, uint64_t bits) {
    ReferenceValue result;
    result.type = type;
    result.bits = {bits};
    return result;
}

bool value_matches_type(const ReferenceValue& value, const ValueType& type) {
    std::vector<uint64_t> extents;
    if (!shape_for(type, &extents) || value.type != type.element_type ||
        value.extents != extents)
        return false;
    size_t count = 0;
    return element_count(extents, &count) && value.bits.size() == count;
}

struct Evaluation {
    std::unordered_map<uint32_t, ReferenceValue> values;
    std::unordered_map<uint32_t, ReferenceValue> candidate_state;
};

bool output_type(const Instruction& instruction, uint32_t* id, ValueType* type) {
    if (!id || !type || instruction.outputs.size() != 1) return false;
    *id = instruction.outputs.front().id;
    *type = instruction.outputs.front().type;
    return *id != UINT32_MAX;
}

bool operand(const Evaluation& evaluation, uint32_t id, ReferenceValue* value) {
    if (!value) return false;
    const auto it = evaluation.values.find(id);
    if (it == evaluation.values.end()) return false;
    *value = it->second;
    return true;
}

bool binary_elementwise(const Evaluation& evaluation, const Instruction& instruction,
                        const ValueType& type, Primitive primitive,
                        ReferenceValue* result) {
    if (!result || instruction.inputs.size() != 2) return false;
    ReferenceValue left, right;
    if (!operand(evaluation, instruction.inputs[0], &left) ||
        !operand(evaluation, instruction.inputs[1], &right) ||
        !value_matches_type(left, type) || !value_matches_type(right, type))
        return false;
    result->type = type.element_type;
    result->extents = left.extents;
    result->bits.resize(left.bits.size());
    for (size_t index = 0; index < left.bits.size(); ++index) {
        if (type.element_type == ElementType::F32) {
            const float a = std::bit_cast<float>(static_cast<uint32_t>(left.bits[index]));
            const float b = std::bit_cast<float>(static_cast<uint32_t>(right.bits[index]));
            if (!std::isfinite(a) || !std::isfinite(b)) return false;
            float value = 0.0f;
            switch (primitive) {
            case Primitive::Add: value = a + b; break;
            case Primitive::Multiply: value = a * b; break;
            case Primitive::Subtract: value = a - b; break;
            case Primitive::Divide: value = a / b; break;
            case Primitive::Maximum: value = std::max(a, b); break;
            default: return false;
            }
            if (!std::isfinite(value)) return false;
            result->bits[index] = std::bit_cast<uint32_t>(value);
        } else if (type.element_type == ElementType::U32) {
            const uint64_t a = static_cast<uint32_t>(left.bits[index]);
            const uint64_t b = static_cast<uint32_t>(right.bits[index]);
            const bool multiply = primitive == Primitive::Multiply;
            if (!multiply && primitive != Primitive::Add) return false;
            const uint64_t value = multiply ? a * b : a + b;
            if (value > UINT32_MAX) return false;
            result->bits[index] = value;
        } else if (type.element_type == ElementType::I32) {
            const int64_t a = static_cast<int32_t>(left.bits[index]);
            const int64_t b = static_cast<int32_t>(right.bits[index]);
            const bool multiply = primitive == Primitive::Multiply;
            if (!multiply && primitive != Primitive::Add) return false;
            const int64_t value = multiply ? a * b : a + b;
            if (value < INT32_MIN || value > INT32_MAX) return false;
            result->bits[index] = static_cast<uint32_t>(static_cast<int32_t>(value));
        } else if (type.element_type == ElementType::U64) {
            const uint64_t a = left.bits[index];
            const uint64_t b = right.bits[index];
            const bool multiply = primitive == Primitive::Multiply;
            if (!multiply && primitive != Primitive::Add) return false;
            if ((!multiply && a > UINT64_MAX - b) ||
                (multiply && a != 0 && b > UINT64_MAX / a))
                return false;
            result->bits[index] = multiply ? a * b : a + b;
        } else {
            return false;
        }
    }
    return true;
}

bool unary_elementwise(const Evaluation& evaluation,
                       const Instruction& instruction, const ValueType& type,
                       Primitive primitive, ReferenceValue* result) {
    if (!result || instruction.inputs.size() != 1 ||
        type.element_type != ElementType::F32)
        return false;
    ReferenceValue input;
    if (!operand(evaluation, instruction.inputs.front(), &input) ||
        !value_matches_type(input, type))
        return false;
    result->type = ElementType::F32;
    result->extents = input.extents;
    result->bits.resize(input.bits.size());
    for (size_t index = 0; index < input.bits.size(); ++index) {
        const float value = std::bit_cast<float>(
            static_cast<uint32_t>(input.bits[index]));
        if (!std::isfinite(value)) return false;
        float computed = 0.0f;
        switch (primitive) {
        case Primitive::Negate: computed = -value; break;
        case Primitive::Exp: computed = std::exp(value); break;
        case Primitive::Log: computed = std::log(value); break;
        case Primitive::Rsqrt: computed = 1.0f / std::sqrt(value); break;
        case Primitive::Sin: computed = std::sin(value); break;
        case Primitive::Cos: computed = std::cos(value); break;
        default: return false;
        }
        if (!std::isfinite(computed)) return false;
        result->bits[index] = std::bit_cast<uint32_t>(computed);
    }
    return true;
}

bool eval_region(const Function& function, uint32_t region_id,
                 Evaluation* evaluation, std::vector<uint32_t>* yields,
                 uint32_t depth);

int64_t floor_divide_index(int64_t value, int64_t divisor) {
    const int64_t quotient = value / divisor;
    const int64_t remainder = value % divisor;
    return quotient - static_cast<int64_t>(remainder < 0);
}

bool evaluate_tensor_index(const TensorIndexExpr& expression,
                           std::span<const uint64_t> iterators,
                           std::span<const ReferenceValue> sources,
                           int64_t* result, uint32_t depth = 0) {
    if (!result || depth > 32) return false;
    switch (expression.expression) {
    case TensorIndexExpression::Constant:
        if (!expression.operands.empty()) return false;
        *result = expression.value;
        return true;
    case TensorIndexExpression::Iterator:
        if (!expression.operands.empty() || expression.value < 0 ||
            static_cast<uint64_t>(expression.value) >= iterators.size() ||
            iterators[static_cast<size_t>(expression.value)] >
                static_cast<uint64_t>(INT64_MAX))
            return false;
        *result = static_cast<int64_t>(
            iterators[static_cast<size_t>(expression.value)]);
        return true;
    case TensorIndexExpression::SourceScalar:
        if (!expression.operands.empty() || expression.value < 0 ||
            static_cast<uint64_t>(expression.value) >= sources.size())
            return false;
        {
            const ReferenceValue& source =
                sources[static_cast<size_t>(expression.value)];
            if (source.type != ElementType::U32 || !source.extents.empty() ||
                source.bits.size() != 1 || source.bits.front() > UINT32_MAX)
                return false;
            *result = static_cast<int64_t>(
                static_cast<uint32_t>(source.bits.front()));
        }
        return true;
    case TensorIndexExpression::Add:
    case TensorIndexExpression::Multiply:
    case TensorIndexExpression::FloorDivide:
    case TensorIndexExpression::Remainder:
        break;
    default:
        return false;
    }
    if (expression.value != 0 || expression.operands.size() != 2) return false;
    int64_t left = 0;
    int64_t right = 0;
    if (!evaluate_tensor_index(expression.operands[0], iterators, sources, &left,
                               depth + 1) ||
        !evaluate_tensor_index(expression.operands[1], iterators, sources, &right,
                               depth + 1))
        return false;
    if (expression.expression == TensorIndexExpression::Add ||
        expression.expression == TensorIndexExpression::Multiply) {
        const __int128 value = expression.expression == TensorIndexExpression::Add
                                   ? static_cast<__int128>(left) + right
                                   : static_cast<__int128>(left) * right;
        if (value < INT64_MIN || value > INT64_MAX) return false;
        *result = static_cast<int64_t>(value);
        return true;
    }
    if (right <= 0) return false;
    const int64_t quotient = floor_divide_index(left, right);
    *result = expression.expression == TensorIndexExpression::FloorDivide
                  ? quotient
                  : left - quotient * right;
    return true;
}

bool tensor_offset(const ReferenceValue& value, const TensorIndexMap& map,
                   std::span<const uint64_t> iterators,
                   std::span<const ReferenceValue> sources,
                   size_t* offset, bool* zero) {
    if (!offset || !zero || map.results.size() != value.extents.size()) return false;
    *offset = 0;
    *zero = false;
    for (size_t axis = 0; axis < map.results.size(); ++axis) {
        int64_t coordinate = 0;
        if (!evaluate_tensor_index(map.results[axis], iterators, sources,
                                   &coordinate))
            return false;
        if (coordinate < 0 || static_cast<uint64_t>(coordinate) >= value.extents[axis]) {
            if (map.bounds != TensorBoundsMode::Zero) return false;
            *zero = true;
            return true;
        }
        if (*offset > (std::numeric_limits<size_t>::max() -
                       static_cast<size_t>(coordinate)) /
                          static_cast<size_t>(value.extents[axis]))
            return false;
        *offset = *offset * static_cast<size_t>(value.extents[axis]) +
                  static_cast<size_t>(coordinate);
    }
    return *offset < value.bits.size();
}

bool eval_structured_tensor(const Function& function,
                            const Instruction& instruction,
                            Evaluation* evaluation, uint32_t depth) {
    const auto* attributes =
        std::get_if<StructuredTensorAttributes>(&instruction.attributes);
    if (!attributes || instruction.regions.size() != 1 ||
        instruction.inputs.size() !=
            attributes->source_count + instruction.outputs.size() ||
        attributes->indexing_maps.size() != instruction.inputs.size())
        return false;
    const Region* body = nullptr;
    for (const Region& candidate : function.regions)
        if (candidate.id == instruction.regions.front()) body = &candidate;
    if (!body || body->arguments.size() != instruction.inputs.size() ||
        body->yields.size() != instruction.outputs.size())
        return false;

    std::vector<uint64_t> iteration_extents;
    iteration_extents.reserve(attributes->iteration_dimensions.size());
    for (const DimensionExpr& dimension : attributes->iteration_dimensions) {
        if (dimension.expression != DimensionExpression::Constant ||
            dimension.value == 0)
            return false;
        iteration_extents.push_back(dimension.value);
    }
    size_t iteration_count = 0;
    if (!element_count(iteration_extents, &iteration_count) ||
        iteration_count > kMaximumReferenceTensorIterations)
        return false;

    std::vector<ReferenceValue> sources(attributes->source_count);
    for (size_t source = 0; source < sources.size(); ++source) {
        if (!operand(*evaluation, instruction.inputs[source], &sources[source]))
            return false;
    }
    std::vector<ReferenceValue> destinations(instruction.outputs.size());
    for (size_t result = 0; result < destinations.size(); ++result) {
        if (!operand(*evaluation,
                     instruction.inputs[attributes->source_count + result],
                     &destinations[result]) ||
            !value_matches_type(destinations[result],
                                instruction.outputs[result].type))
            return false;
    }

    std::vector<uint64_t> iterators(iteration_extents.size(), 0);
    for (size_t flat = 0; flat < iteration_count; ++flat) {
        size_t remainder = flat;
        for (size_t axis = iteration_extents.size(); axis != 0; --axis) {
            iterators[axis - 1] = remainder % iteration_extents[axis - 1];
            remainder /= static_cast<size_t>(iteration_extents[axis - 1]);
        }
        for (size_t source = 0; source < sources.size(); ++source) {
            size_t offset = 0;
            bool zero = false;
            if (!tensor_offset(sources[source], attributes->indexing_maps[source],
                               iterators, sources, &offset, &zero))
                return false;
            evaluation->values[body->arguments[source].id] =
                scalar_value(sources[source].type,
                             zero ? 0 : sources[source].bits[offset]);
        }
        std::vector<size_t> destination_offsets(destinations.size(), 0);
        for (size_t result = 0; result < destinations.size(); ++result) {
            const size_t input = attributes->source_count + result;
            bool zero = false;
            if (!tensor_offset(destinations[result],
                               attributes->indexing_maps[input], iterators,
                               sources,
                               &destination_offsets[result], &zero) || zero)
                return false;
            evaluation->values[body->arguments[input].id] =
                scalar_value(destinations[result].type,
                             destinations[result].bits[destination_offsets[result]]);
        }
        std::vector<uint32_t> body_yields;
        if (!eval_region(function, body->id, evaluation, &body_yields, depth + 1) ||
            body_yields.size() != destinations.size())
            return false;
        for (size_t result = 0; result < destinations.size(); ++result) {
            ReferenceValue yielded;
            if (!operand(*evaluation, body_yields[result], &yielded) ||
                yielded.type != destinations[result].type ||
                !yielded.extents.empty() || yielded.bits.size() != 1)
                return false;
            destinations[result].bits[destination_offsets[result]] = yielded.bits[0];
        }
    }
    for (size_t result = 0; result < destinations.size(); ++result)
        evaluation->values[instruction.outputs[result].id] =
            std::move(destinations[result]);
    return true;
}

bool eval_region(const Function& function, uint32_t region_id,
                 Evaluation* evaluation, std::vector<uint32_t>* yields,
                 uint32_t depth) {
    if (!evaluation || !yields || depth > 64) return false;
    const Region* region = nullptr;
    for (const Region& candidate : function.regions)
        if (candidate.id == region_id) region = &candidate;
    if (!region) return false;
    for (const Instruction& instruction : region->instructions) {
        uint32_t id = UINT32_MAX;
        ValueType type;
        switch (instruction.primitive.code) {
        case Primitive::Constant: {
            if (!output_type(instruction, &id, &type)) return false;
            const auto* attributes = std::get_if<ConstantAttributes>(&instruction.attributes);
            std::vector<uint64_t> extents;
            size_t count = 0;
            if (!attributes || !shape_for(type, &extents) ||
                !element_count(extents, &count))
                return false;
            ReferenceValue value;
            value.type = type.element_type;
            value.extents = std::move(extents);
            value.bits.assign(count, attributes->bits);
            evaluation->values[id] = std::move(value);
            break;
        }
        case Primitive::Add:
        case Primitive::Multiply:
        case Primitive::Subtract:
        case Primitive::Divide:
        case Primitive::Maximum: {
            if (!output_type(instruction, &id, &type)) return false;
            std::vector<uint64_t> extents;
            if (!shape_for(type, &extents)) return false;
            ReferenceValue value;
            if (!binary_elementwise(*evaluation, instruction, type,
                                    instruction.primitive.code,
                                    &value)) return false;
            evaluation->values[id] = std::move(value);
            break;
        }
        case Primitive::Negate:
        case Primitive::Exp:
        case Primitive::Log:
        case Primitive::Rsqrt:
        case Primitive::Sin:
        case Primitive::Cos: {
            if (!output_type(instruction, &id, &type)) return false;
            ReferenceValue value;
            if (!unary_elementwise(*evaluation, instruction, type,
                                   instruction.primitive.code, &value))
                return false;
            evaluation->values[id] = std::move(value);
            break;
        }
        case Primitive::StateRead: {
            if (!output_type(instruction, &id, &type)) return false;
            const auto* attributes = std::get_if<StateAttributes>(&instruction.attributes);
            if (!attributes) return false;
            const auto it = evaluation->candidate_state.find(attributes->state_id);
            if (it == evaluation->candidate_state.end() || !value_matches_type(it->second, type))
                return false;
            evaluation->values[id] = it->second;
            break;
        }
        case Primitive::StateWrite: {
            if (!instruction.outputs.empty() || instruction.inputs.size() != 1) return false;
            const auto* attributes = std::get_if<StateAttributes>(&instruction.attributes);
            ReferenceValue value;
            if (!attributes || !operand(*evaluation, instruction.inputs.front(), &value)) return false;
            evaluation->candidate_state[attributes->state_id] = std::move(value);
            break;
        }
        case Primitive::BoundedLoop: {
            if (!output_type(instruction, &id, &type) || instruction.regions.size() != 1 ||
                instruction.inputs.size() != 1) return false;
            const auto* attributes = std::get_if<LoopAttributes>(&instruction.attributes);
            if (!attributes || attributes->step == 0 || attributes->upper < attributes->lower) return false;
            ReferenceValue carried;
            if (!operand(*evaluation, instruction.inputs.front(), &carried) ||
                !value_matches_type(carried, type)) return false;
            const Region* body = nullptr;
            for (const Region& candidate : function.regions)
                if (candidate.id == instruction.regions.front()) body = &candidate;
            if (!body || body->arguments.size() != 2 || body->yields.size() != 1) return false;
            for (uint64_t iteration = attributes->lower; iteration < attributes->upper;) {
                evaluation->values[body->arguments[0].id] = scalar_value(ElementType::U64, iteration);
                evaluation->values[body->arguments[1].id] = carried;
                std::vector<uint32_t> body_yields;
                if (!eval_region(function, body->id, evaluation, &body_yields, depth + 1) ||
                    body_yields.size() != 1 || !operand(*evaluation, body_yields.front(), &carried)) return false;
                if (attributes->upper - iteration < attributes->step) break;
                iteration += attributes->step;
            }
            if (!value_matches_type(carried, type)) return false;
            evaluation->values[id] = std::move(carried);
            break;
        }
        case Primitive::StructuredTensor:
            if (!eval_structured_tensor(function, instruction, evaluation, depth))
                return false;
            break;
        }
    }
    *yields = region->yields;
    return true;
}

} // namespace

ReferencePhysicalResult decode_reference_resource(
    const VerifiedPhysicalProgramPackage& package, uint32_t resource_id,
    std::span<const uint64_t> logical_coordinate) {
    const auto resolved = package.resolve_resource(resource_id);
    if (const auto* report = std::get_if<CompatibilityReport>(&resolved))
        return *report;
    const auto resources = package.resources();
    const auto resource = std::find_if(
        resources.begin(), resources.end(),
        [resource_id](const PhysicalResourceBinding& binding) {
            return binding.resource_id == resource_id;
        });
    if (resource == resources.end())
        return reference_error(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                               "reference physical resource is unavailable");
    const auto programs = package.programs();
    const auto logical_types = package.program_logical_types();
    size_t program_index = 0;
    for (; program_index < programs.size(); ++program_index) {
        const auto digest = physical_program_digest(programs[program_index]);
        if (std::holds_alternative<PhysicalProgramDigest>(digest) &&
            std::get<PhysicalProgramDigest>(digest) == resource->program_digest)
            break;
    }
    if (program_index == programs.size() || program_index >= logical_types.size())
        return reference_error(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                               "reference physical program is unavailable");
    const auto& views = std::get<std::vector<PhysicalResourcePlaneView>>(resolved);
    std::vector<PhysicalPlaneBinding> bindings;
    bindings.reserve(views.size());
    for (const auto& view : views) {
        auto bytes = std::make_shared<std::vector<uint8_t>>(
            view.bytes.begin(), view.bytes.end());
        bindings.push_back({view.plane,
                            std::shared_ptr<const std::vector<uint8_t>>(std::move(bytes)),
                            0, view.bytes.size()});
    }
    const auto verified = verify_physical_program(
        programs[program_index], bindings, logical_types[program_index]);
    if (const auto* report = std::get_if<CompatibilityReport>(&verified))
        return *report;
    const auto interpreted = interpret_physical_value(
        std::get<VerifiedPhysicalProgram>(verified), logical_coordinate);
    if (const auto* value = std::get_if<ScalarValue>(&interpreted)) return *value;
    switch (std::get<PhysicalInterpretError>(interpreted)) {
    case PhysicalInterpretError::CoordinateInvalid:
        return reference_error(CompatibilityError::RUNTIME_INPUT_INVALID,
                               "reference physical coordinate is invalid");
    case PhysicalInterpretError::AddressOutOfBounds:
        return reference_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                               "reference physical address is out of bounds");
    case PhysicalInterpretError::ArithmeticOverflow:
    case PhysicalInterpretError::NumericalPolicyRejected:
    case PhysicalInterpretError::InternalInvariant:
        return reference_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE,
                               "reference physical interpretation failed");
    }
    return reference_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE,
                           "reference physical interpretation failed");
}

namespace {

ReferenceExecutionResult execute_reference_impl(
    const VerifiedProgram& semantic,
    const VerifiedPhysicalProgramPackage* package,
    ReferenceState& state,
    std::span<const ReferenceInput> inputs) {
    const Program& program = program_definition(semantic);
    if (program.functions.size() != 1 || program.exports.empty())
        return reference_error(CompatibilityError::IR_REFERENCE_INVALID,
                               "reference executor requires one exported function");
    const Function& function = program.functions.front();
    if (function.entry_region_id == UINT32_MAX)
        return reference_error(CompatibilityError::IR_REFERENCE_INVALID,
                               "reference function has no entry region");

    Evaluation evaluation;
    std::unordered_set<uint32_t> produced_values;
    std::unordered_map<uint32_t, ValueType> external_types;
    const Region* entry_region = nullptr;
    for (const Region& region : function.regions)
        if (region.id == function.entry_region_id) entry_region = &region;
    if (!entry_region)
        return reference_error(CompatibilityError::IR_REFERENCE_INVALID,
                               "reference entry region is unavailable");
    for (const TypedValue& argument : entry_region->arguments) {
        if (argument.id == UINT32_MAX ||
            !external_types.emplace(argument.id, argument.type).second)
            return reference_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "reference program has duplicate external values");
    }
    for (const Region& region : function.regions)
        for (const Instruction& instruction : region.instructions)
            for (const TypedValue& output : instruction.outputs)
                if (output.id == UINT32_MAX || !produced_values.insert(output.id).second)
                    return reference_error(CompatibilityError::IR_REFERENCE_INVALID,
                                           "reference program has duplicate output values");

    const std::span<const PhysicalResourceBinding> resources =
        package ? package->resources() : std::span<const PhysicalResourceBinding>{};
    for (const PhysicalResourceBinding& resource : resources) {
        if (resource.semantic_function_id != function.id)
            return reference_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "reference resource belongs to another function");
        const auto expected = external_types.find(resource.semantic_value_id);
        if (expected == external_types.end() ||
            evaluation.values.contains(resource.semantic_value_id))
            return reference_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "reference resource entry binding is invalid");
        ReferenceValue value;
        value.type = expected->second.element_type;
        if (!shape_for(expected->second, &value.extents))
            return reference_error(CompatibilityError::IR_SHAPE_MISMATCH,
                                   "reference resource type is unsupported");
        size_t count = 0;
        if (!element_count(value.extents, &count))
            return reference_error(CompatibilityError::IR_SHAPE_MISMATCH,
                                   "reference resource shape is invalid");
        value.bits.reserve(count);
        std::vector<uint64_t> coordinate(value.extents.size(), 0);
        for (size_t flat = 0; flat < count; ++flat) {
            size_t remainder = flat;
            for (size_t axis = value.extents.size(); axis != 0; --axis) {
                const uint64_t extent = value.extents[axis - 1];
                coordinate[axis - 1] = remainder % extent;
                remainder /= static_cast<size_t>(extent);
            }
            const auto decoded = decode_reference_resource(
                *package, resource.resource_id, coordinate);
            const auto* scalar = std::get_if<ScalarValue>(&decoded);
            if (!scalar || scalar->type != value.type)
                return reference_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE,
                                       "reference resource decode failed");
            value.bits.push_back(scalar->bits);
        }
        evaluation.values.emplace(resource.semantic_value_id, std::move(value));
    }

    for (const auto& reference : program.state_references) {
        std::vector<uint64_t> extents;
        if (reference.id == UINT32_MAX || !shape_for(reference.type, &extents))
            return reference_error(CompatibilityError::IR_STATE_INVALID,
                                   "reference state type is unsupported");
        size_t count = 0;
        if (!element_count(extents, &count)) return reference_error(
            CompatibilityError::IR_STATE_INVALID, "reference state shape is invalid");
        ReferenceValue zero;
        zero.type = reference.type.element_type;
        zero.extents = std::move(extents);
        zero.bits.assign(count, 0);
        evaluation.candidate_state[reference.id] = std::move(zero);
    }
    std::unordered_set<uint32_t> supplied_state;
    for (const auto& slot : state.slots) {
        const auto it = evaluation.candidate_state.find(slot.first);
        if (it == evaluation.candidate_state.end() || !supplied_state.insert(slot.first).second)
            return reference_error(CompatibilityError::STATE_ABI_MISMATCH,
                                   "reference state contains an unknown or duplicate slot");
        if (slot.second.type != it->second.type || slot.second.extents != it->second.extents ||
            slot.second.bits.size() != it->second.bits.size())
            return reference_error(CompatibilityError::STATE_ABI_MISMATCH,
                                   "reference state slot shape or type is invalid");
        it->second = slot.second;
    }
    std::unordered_set<uint32_t> supplied_inputs;
    for (const auto& input : inputs) {
        if (input.value_id == UINT32_MAX || !supplied_inputs.insert(input.value_id).second ||
            produced_values.contains(input.value_id) ||
            evaluation.values.contains(input.value_id))
            return reference_error(CompatibilityError::RUNTIME_INPUT_INVALID,
                                   "reference input IDs must be unique and external");
        const auto expected = external_types.find(input.value_id);
        if (expected == external_types.end() ||
            !value_matches_type(input.value, expected->second))
            return reference_error(CompatibilityError::RUNTIME_INPUT_INVALID,
                                   "reference input type or shape is invalid");
        evaluation.values.emplace(input.value_id, input.value);
    }
    for (const auto& [value_id, type] : external_types) {
        (void)type;
        if (!evaluation.values.contains(value_id))
            return reference_error(CompatibilityError::RUNTIME_INPUT_INVALID,
                                   "reference execution is missing an external input");
    }
    std::vector<uint32_t> yields;
    if (!eval_region(function, function.entry_region_id, &evaluation, &yields, 0))
        return reference_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE,
                               "reference program execution failed");

    ReferenceResult result;
    for (const ProgramExport& export_record : program.exports) {
        if (export_record.function_id != function.id || export_record.result_index >= yields.size())
            return reference_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "reference export does not match the executed function");
        ReferenceValue value;
        if (!operand(evaluation, yields[export_record.result_index], &value))
            return reference_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "reference export value is unavailable");
        if (!value_matches_type(value, export_record.type))
            return reference_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE,
                                   "reference export type or shape is invalid");
        result.exports.push_back(std::move(value));
    }
    state.slots.clear();
    state.slots.reserve(program.state_references.size());
    for (const auto& reference : program.state_references)
        state.slots.emplace_back(reference.id, evaluation.candidate_state[reference.id]);
    state.generation += 1;
    result.generation = state.generation;
    return result;
}

} // namespace

ReferenceExecutionResult execute_reference_program(
    const VerifiedProgram& program,
    ReferenceState& state,
    std::span<const ReferenceInput> inputs) {
    return execute_reference_impl(program, nullptr, state, inputs);
}

ReferenceExecutionResult execute_reference(
    const VerifiedPhysicalProgramPackage& package,
    ReferenceState& state,
    std::span<const ReferenceInput> inputs) {
    const auto& semantic = package.semantic_program();
    if (!semantic)
        return reference_error(
            CompatibilityError::KERNEL_UNAVAILABLE,
            "reference executor requires a verified semantic program");
    return execute_reference_impl(*semantic, &package, state, inputs);
}

} // namespace Laplace
