#include "program_metal.h"

#include "compat_rule.h"
#include "physical_program_package.h"
#include "program_package.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Laplace {
namespace {

CompatibilityReport metal_error(CompatibilityError code, std::string detail) {
    return compatibility_report(code, std::move(detail));
}

// These are language scalar ABIs, independent of physical weight encodings.
struct ScalarAbi { const char* name; const char* bits_name; size_t bytes; };
ScalarAbi scalar_abi(ElementType type) {
    switch (type) {
    case ElementType::I1: return {"uchar", "uchar", 1};
    case ElementType::I32: return {"int", "uint", 4};
    case ElementType::U32: return {"uint", "uint", 4};
    case ElementType::U64: return {"ulong", "ulong", 8};
    case ElementType::F16: return {"half", "ushort", 2};
    case ElementType::F32: return {"float", "uint", 4};
    }
    return {nullptr, nullptr, 0};
}
std::string scalar_literal(ElementType type, uint64_t bits) {
    const auto abi = scalar_abi(type);
    return "as_type<" + std::string(abi.name) + ">(" + abi.bits_name +
           "(" + std::to_string(bits) + "ul))";
}
std::vector<uint8_t> pack_value(const MetalProgramValue& value) {
    const size_t width = scalar_abi(value.type).bytes;
    std::vector<uint8_t> bytes(value.bits.size() * width);
    for (size_t i = 0; i < value.bits.size(); ++i)
        std::memcpy(bytes.data() + i * width, &value.bits[i], width);
    return bytes;
}
MetalProgramValue unpack_value(ElementType type, const std::vector<uint64_t>& extents,
                               size_t count, const void* data) {
    MetalProgramValue result{type, extents, std::vector<uint64_t>(count, 0)};
    const size_t width = scalar_abi(type).bytes;
    for (size_t i = 0; i < count; ++i)
        std::memcpy(&result.bits[i], static_cast<const uint8_t*>(data) + i * width, width);
    return result;
}

bool constant_shape(const ValueType& type, std::vector<uint64_t>* extents,
                    size_t* count) {
    if (!extents || !count || !scalar_abi(type.element_type).bytes)
        return false;
    extents->clear();
    size_t product = 1;
    for (const DimensionExpr& dimension : type.dimensions) {
        if (dimension.expression != DimensionExpression::Constant ||
            dimension.value == 0 ||
            dimension.value > std::numeric_limits<size_t>::max() / product)
            return false;
        extents->push_back(dimension.value);
        product *= static_cast<size_t>(dimension.value);
    }
    if (product > SIZE_MAX / scalar_abi(type.element_type).bytes) return false;
    *count = product;
    return true;
}

const Region* find_region(const Function& function, uint32_t id) {
    const auto found = std::find_if(
        function.regions.begin(), function.regions.end(),
        [id](const Region& region) { return region.id == id; });
    return found == function.regions.end() ? nullptr : &*found;
}

std::array<uint8_t, 32> source_digest(std::string_view source) {
    std::array<uint8_t, 32> result{};
    CC_SHA256(source.data(), static_cast<CC_LONG>(source.size()), result.data());
    return result;
}

bool emit_index(const TensorIndexExpr& expression, std::string* result,
                std::span<const std::string> source_scalars,
                std::span<const std::string> source_elements, size_t depth = 0) {
    if (!result || depth > 32) return false;
    switch (expression.expression) {
    case TensorIndexExpression::Constant:
        if (!expression.operands.empty()) return false;
        *result = std::to_string(expression.value) + "l";
        return true;
    case TensorIndexExpression::Iterator:
        if (!expression.operands.empty() || expression.value < 0 ||
            expression.value > UINT32_MAX)
            return false;
        *result = "i" + std::to_string(expression.value);
        return true;
    case TensorIndexExpression::SourceScalar:
        if (!expression.operands.empty() || expression.value < 0 ||
            static_cast<uint64_t>(expression.value) >= source_scalars.size() ||
            source_scalars[static_cast<size_t>(expression.value)].empty())
            return false;
        *result = source_scalars[static_cast<size_t>(expression.value)];
        return true;
    case TensorIndexExpression::SourceElement: {
        if (expression.value < 0 || static_cast<uint64_t>(expression.value) >= source_elements.size() ||
            source_elements[expression.value].empty()) return false;
        *result = source_elements[expression.value];
        for (const auto& operand : expression.operands) {
            std::string coordinate;
            if (!emit_index(operand, &coordinate, source_scalars, source_elements, depth + 1)) return false;
            *result += ", " + coordinate;
        }
        *result += ")";
        return true;
    }
    case TensorIndexExpression::Add:
    case TensorIndexExpression::Multiply:
    case TensorIndexExpression::FloorDivide:
    case TensorIndexExpression::Remainder:
        break;
    default:
        return false;
    }
    if (expression.operands.size() != 2) return false;
    std::string left;
    std::string right;
    if (!emit_index(expression.operands[0], &left, source_scalars, source_elements, depth + 1) ||
        !emit_index(expression.operands[1], &right, source_scalars, source_elements, depth + 1))
        return false;
    if (expression.expression == TensorIndexExpression::Add) {
        *result = "laplace_index_add(" + left + ", " + right + ", error)";
    } else if (expression.expression == TensorIndexExpression::Multiply) {
        *result = "laplace_index_mul(" + left + ", " + right + ", error)";
    } else if (expression.expression == TensorIndexExpression::FloorDivide) {
        *result = "laplace_floor_div(" + left + ", " + right + ")";
    } else {
        *result = "laplace_floor_mod(" + left + ", " + right + ")";
    }
    return true;
}

struct InputSpec {
    uint32_t value_id = UINT32_MAX;
    ElementType type = ElementType::F32;
    std::vector<uint64_t> extents;
    size_t element_count = 0;
};

struct Lowering {
    struct PhysicalPlane {
        uint32_t resource_id = UINT32_MAX;
        uint16_t plane = kNoPhysicalPlane;
    };
    std::string source;
    std::vector<InputSpec> inputs;
    std::vector<PhysicalPlane> physical_planes;
    ElementType output_type = ElementType::F32;
    std::vector<uint64_t> output_extents;
    size_t output_count = 0;
    uint32_t output_value_id = UINT32_MAX;
    uint32_t direct_state_index = UINT32_MAX;
};

struct MultiLowering {
    struct State {
        uint32_t state_id = UINT32_MAX;
        ElementType type = ElementType::F32;
        std::vector<uint64_t> extents;
        size_t element_count = 0;
        std::vector<uint32_t> read_value_ids;
        uint32_t write_value_id = UINT32_MAX;
    };
    std::vector<Lowering> stages;
    std::vector<InputSpec> inputs;
    std::vector<State> states;
    std::vector<uint32_t> exports;
    ElementType output_type = ElementType::F32;
    std::vector<uint64_t> output_extents;
    size_t output_count = 0;
};

struct PhysicalLowering {
    std::string source;
    size_t plane_count = 0;
    LogicalTensorType logical;
    size_t output_count = 0;
};

struct BoundPhysicalSource {
    const PhysicalResourceBinding* resource = nullptr;
    const PhysicalProgram* program = nullptr;
    const LogicalTensorType* logical = nullptr;
    size_t plane_base = 0;
};

struct MetalPhysicalPlaneSource {
    const uint8_t* bytes = nullptr;
    size_t length = 0;
    const uint8_t* mapping_base = nullptr;
    size_t mapping_length = 0;
    uint64_t mapping_offset = 0;
};

const char* physical_type_name(PhysicalValueType type) {
    switch (type) {
        case PhysicalValueType::Predicate: return "bool";
        case PhysicalValueType::Index: return "ulong";
        case PhysicalValueType::U32: return "uint";
        case PhysicalValueType::I32: return "int";
        case PhysicalValueType::F32: return "float";
    }
    return nullptr;
}

std::string physical_operand(std::string_view prefix, uint32_t value) {
    return std::string(prefix) + std::to_string(value);
}

std::string apply_policy(std::string expression,
                         const PhysicalNumericPolicy& policy,
                         std::string_view error_name) {
    return "as_type<float>(laplace_apply_policy(as_type<uint>(" + expression +
           "), " + std::to_string(static_cast<uint8_t>(policy.nan)) + "u, " +
           std::to_string(static_cast<uint8_t>(policy.infinity)) + "u, " +
           std::to_string(static_cast<uint8_t>(policy.subnormal)) + "u, " +
           std::to_string(static_cast<uint8_t>(policy.signed_zero)) +
           "u, " + std::string(error_name) + "))";
}

struct PhysicalSourceContext {
    std::string value_prefix = "v";
    std::string coordinate_prefix = "c";
    std::vector<std::string> plane_names;
    std::vector<size_t> plane_offset_indices;
    std::string offsets_name = "plane_offsets";
    std::string error_name = "error";
    std::string indent = "    ";
};

std::string physical_instruction_source(
    const PhysicalProgram& program, size_t index,
    const PhysicalSourceContext& context) {
    const PhysicalInstruction& instruction = program.instructions[index];
    const std::string name = physical_operand(
        context.value_prefix, static_cast<uint32_t>(index));
    const auto operand = [&](size_t slot) {
        return physical_operand(context.value_prefix,
                                instruction.operands[slot]);
    };
    const char* type = physical_type_name(instruction.result_type);
    if (!type) return {};
    std::string expression;
    switch (instruction.opcode) {
        case PhysicalOpcode::ConstIndex:
            expression = std::to_string(instruction.immediate) + "ul";
            break;
        case PhysicalOpcode::ConstU32:
            expression = std::to_string(static_cast<uint32_t>(instruction.immediate)) + "u";
            break;
        case PhysicalOpcode::ConstI32:
            expression = "as_type<int>(uint(" +
                         std::to_string(static_cast<uint32_t>(instruction.immediate)) + "u))";
            break;
        case PhysicalOpcode::ConstF32Bits:
            expression = "as_type<float>(uint(" +
                         std::to_string(static_cast<uint32_t>(instruction.immediate)) + "u))";
            break;
        case PhysicalOpcode::Coordinate:
            expression = context.coordinate_prefix +
                         std::to_string(instruction.immediate);
            break;
        case PhysicalOpcode::Extent:
            return {};
        case PhysicalOpcode::IndexAdd:
            expression = operand(0) + " + " + operand(1);
            break;
        case PhysicalOpcode::IndexSubtract:
            expression = operand(0) + " - " + operand(1);
            break;
        case PhysicalOpcode::IndexMultiply:
            expression = operand(0) + " * " + operand(1);
            break;
        case PhysicalOpcode::IndexDivideConstant:
            expression = operand(0) + " / " +
                         std::to_string(instruction.immediate) + "ul";
            break;
        case PhysicalOpcode::IndexRemainderConstant:
            expression = operand(0) + " % " +
                         std::to_string(instruction.immediate) + "ul";
            break;
        case PhysicalOpcode::IndexCeilDivideConstant:
            expression = operand(0) + " / " +
                         std::to_string(instruction.immediate) + "ul + (" +
                         operand(0) + " % " +
                         std::to_string(instruction.immediate) + "ul != 0ul)";
            break;
        case PhysicalOpcode::IndexFromU32:
            expression = "ulong(" + operand(0) + ")";
            break;
        case PhysicalOpcode::IndexLess:
            expression = operand(0) + " < " + operand(1);
            break;
        case PhysicalOpcode::Select:
            if (instruction.result_type == PhysicalValueType::F32)
                expression = "as_type<float>(" + operand(0) +
                             " ? as_type<uint>(" + operand(1) +
                             ") : as_type<uint>(" + operand(2) + "))";
            else
                expression = operand(0) + " ? " + operand(1) + " : " + operand(2);
            break;
        case PhysicalOpcode::LoadBits:
            if (instruction.plane >= context.plane_names.size() ||
                instruction.plane >= context.plane_offset_indices.size())
                return {};
            expression = std::string(instruction.bit_order == PhysicalBitOrder::Lsb0Little
                                         ? "laplace_load_lsb("
                                         : "laplace_load_msb(") +
                         context.plane_names[instruction.plane] + ", " +
                         context.offsets_name + "[" +
                         std::to_string(
                             context.plane_offset_indices[instruction.plane]) +
                         "u] * 8ul + " + operand(0) + ", " +
                         std::to_string(instruction.bit_width) + "u)";
            break;
        case PhysicalOpcode::U32And:
            expression = operand(0) + " & " + operand(1);
            break;
        case PhysicalOpcode::U32Or:
            expression = operand(0) + " | " + operand(1);
            break;
        case PhysicalOpcode::U32Xor:
            expression = operand(0) + " ^ " + operand(1);
            break;
        case PhysicalOpcode::U32Add:
            expression = operand(0) + " + " + operand(1);
            break;
        case PhysicalOpcode::U32Multiply:
            expression = operand(0) + " * " + operand(1);
            break;
        case PhysicalOpcode::U32ShiftLeftConstant:
            expression = operand(0) + " << " + std::to_string(instruction.immediate) + "u";
            break;
        case PhysicalOpcode::U32ShiftRightConstant:
            expression = operand(0) + " >> " + std::to_string(instruction.immediate) + "u";
            break;
        case PhysicalOpcode::U32FunnelShiftRight:
            expression = "laplace_funnel_shift_right(" + operand(0) + ", " +
                         operand(1) + ", " + operand(2) + ")";
            break;
        case PhysicalOpcode::SignExtend:
            if (instruction.bit_width == 32) {
                expression = "as_type<int>(" + operand(0) + ")";
            } else {
                expression = "int(" + operand(0) + " << " +
                             std::to_string(32 - instruction.bit_width) +
                             "u) >> " + std::to_string(32 - instruction.bit_width) + "u";
            }
            break;
        case PhysicalOpcode::F16ToF32:
            expression = program.policies[instruction.policy].nan ==
                                 PhysicalNanPolicy::PreserveIeee
                ? "as_type<float>(laplace_half_to_float_bits(ushort(" + operand(0) + ")))"
                : "float(as_type<half>(ushort(" + operand(0) + ")))";
            break;
        case PhysicalOpcode::Bf16ToF32:
            expression = "as_type<float>(uint(" + operand(0) + ") << 16u)";
            break;
        case PhysicalOpcode::BitsToF32:
            expression = "as_type<float>(uint(" + operand(0) + "))";
            break;
        case PhysicalOpcode::U32ToF32:
            expression = "float(" + operand(0) + ")";
            break;
        case PhysicalOpcode::I32ToF32:
            expression = "float(" + operand(0) + ")";
            break;
        case PhysicalOpcode::F32ToU32:
        case PhysicalOpcode::F32ToI32: {
            const PhysicalNumericPolicy& policy = program.policies[instruction.policy];
            expression = "laplace_float_to_integer(" + operand(0) + ", " +
                         std::to_string(instruction.opcode == PhysicalOpcode::F32ToI32) +
                         "u, " + std::to_string(static_cast<uint8_t>(policy.rounding)) +
                         "u, " + std::to_string(static_cast<uint8_t>(policy.integer_overflow)) +
                         "u, " + context.error_name + ")";
            if (instruction.opcode == PhysicalOpcode::F32ToI32)
                expression = "as_type<int>(" + expression + ")";
            break;
        }
        case PhysicalOpcode::F32Add:
            expression = operand(0) + " + " + operand(1);
            break;
        case PhysicalOpcode::F32Subtract:
            expression = operand(0) + " - " + operand(1);
            break;
        case PhysicalOpcode::F32Multiply:
            expression = operand(0) + " * " + operand(1);
            break;
        case PhysicalOpcode::F32Fma:
            expression = "fma(" + operand(0) + ", " + operand(1) + ", " + operand(2) + ")";
            break;
        case PhysicalOpcode::F32Negate:
            expression = "as_type<float>(as_type<uint>(" + operand(0) + ") ^ 0x80000000u)";
            break;
        case PhysicalOpcode::F32RoundToF16:
            expression = "float(half(" + operand(0) + "))";
            break;
        case PhysicalOpcode::F32Clamp:
            expression = "laplace_clamp(" + operand(0) + ", " + operand(1) +
                         ", " + operand(2) + ", " + context.error_name + ")";
            break;
    }
    if (expression.empty()) return {};
    if (instruction.policy != kNoPhysicalPolicy &&
        instruction.result_type == PhysicalValueType::F32)
        expression = apply_policy(std::move(expression),
                                  program.policies[instruction.policy],
                                  context.error_name);
    return context.indent + std::string(type) + " " + name + " = " +
           expression + ";\n";
}

constexpr std::string_view kPhysicalMetalPrelude = R"METAL(
inline uint laplace_half_to_float_bits(ushort value) {
    uint sign = uint(value & 0x8000u) << 16u;
    uint exponent = (uint(value) >> 10u) & 31u;
    uint fraction = uint(value) & 1023u;
    if (exponent == 0u) {
        if (fraction == 0u) return sign;
        uint shift = 0u;
        while ((fraction & 1024u) == 0u) {
            fraction <<= 1u;
            ++shift;
        }
        return sign | ((113u - shift) << 23u) | ((fraction & 1023u) << 13u);
    }
    if (exponent == 31u) return sign | 0x7f800000u | (fraction << 13u);
    return sign | ((exponent + 112u) << 23u) | (fraction << 13u);
}
inline uint laplace_load_lsb(device const uchar* bytes, ulong address, uint width) {
    if (width == 0u || width > 32u) return 0u;
    const uint bit_offset = uint(address & 7ul);
    const ulong first_byte = address >> 3ul;
    const uint byte_count = (bit_offset + width + 7u) >> 3u;
    ulong window = 0ul;
    for (uint index = 0u; index < byte_count; ++index)
        window |= ulong(bytes[first_byte + ulong(index)]) << (index * 8u);
    const uint mask = width == 32u ? 0xffffffffu : (1u << width) - 1u;
    return uint((window >> bit_offset) & ulong(mask));
}
inline uint laplace_load_msb(device const uchar* bytes, ulong address, uint width) {
    if (width == 0u || width > 32u) return 0u;
    const uint bit_offset = uint(address & 7ul);
    const ulong first_byte = address >> 3ul;
    const uint byte_count = (bit_offset + width + 7u) >> 3u;
    ulong window = 0ul;
    for (uint index = 0u; index < byte_count; ++index)
        window = (window << 8u) | ulong(bytes[first_byte + ulong(index)]);
    const uint shift = byte_count * 8u - bit_offset - width;
    const uint mask = width == 32u ? 0xffffffffu : (1u << width) - 1u;
    return uint((window >> shift) & ulong(mask));
}
inline uint laplace_funnel_shift_right(uint low, uint high, ulong shift) {
    if (shift == 0ul) return low;
    if (shift < 32ul)
        return (low >> uint(shift)) | (high << uint(32ul - shift));
    return high >> uint(shift - 32ul);
}
inline uint laplace_apply_policy(uint bits, uint nan_policy, uint infinity_policy,
                                  uint subnormal_policy, uint zero_policy,
                                  device atomic_uint* error) {
    uint sign = bits & 0x80000000u;
    uint exponent = bits & 0x7f800000u;
    uint fraction = bits & 0x007fffffu;
    if (exponent == 0x7f800000u && fraction != 0u) {
        if (nan_policy == 2u) atomic_store_explicit(error, 1u, memory_order_relaxed);
        if (nan_policy != 3u) bits = 0x7fc00000u;
    } else if (exponent == 0x7f800000u) {
        if (infinity_policy == 3u) atomic_store_explicit(error, 1u, memory_order_relaxed);
        else if (infinity_policy == 2u) bits = sign | 0x7f7fffffu;
    } else if (exponent == 0u && fraction != 0u && subnormal_policy == 2u) {
        bits = sign;
    }
    if ((bits & 0x7fffffffu) == 0u && zero_policy == 2u) bits = 0u;
    return bits;
}
inline uint laplace_float_to_integer(float value, uint signed_result,
                                     uint rounding, uint overflow,
                                     device atomic_uint* error) {
    if (!isfinite(value)) {
        atomic_store_explicit(error, 1u, memory_order_relaxed);
        return 0u;
    }
    float rounded = rounding == 1u ? rint(value) :
                    rounding == 2u ? trunc(value) :
                    rounding == 3u ? ceil(value) : floor(value);
    float minimum = signed_result != 0u ? -2147483648.0f : 0.0f;
    float maximum = signed_result != 0u ? 2147483648.0f : 4294967296.0f;
    if (rounded < minimum || rounded >= maximum) {
        if (overflow == 1u) atomic_store_explicit(error, 1u, memory_order_relaxed);
        if (rounded < minimum) return signed_result != 0u ? 0x80000000u : 0u;
        return signed_result != 0u ? 0x7fffffffu : 0xffffffffu;
    }
    return signed_result != 0u ? as_type<uint>(int(rounded)) : uint(rounded);
}
inline float laplace_clamp(float value, float lower, float upper,
                           device atomic_uint* error) {
    if (isnan(value) || isnan(lower) || isnan(upper)) return as_type<float>(0x7fc00000u);
    if (lower > upper) {
        atomic_store_explicit(error, 1u, memory_order_relaxed);
        return value;
    }
    return max(lower, min(value, upper));
}
)METAL";

std::variant<PhysicalLowering, CompatibilityReport> lower_physical_program(
    const PhysicalProgram& program, const LogicalTensorType& logical) {
    if (logical.extents.size() != program.logical_rank ||
        (logical.element_type != ElementType::U32 &&
         logical.element_type != ElementType::I32 &&
         logical.element_type != ElementType::F32))
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                           "Metal physical result type is unsupported");
    PhysicalLowering lowering;
    lowering.logical = logical;
    lowering.output_count = 1;
    for (uint64_t extent : logical.extents) {
        if (extent == 0 || extent > SIZE_MAX / lowering.output_count)
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "Metal physical output shape is unsupported");
        lowering.output_count *= static_cast<size_t>(extent);
    }
    if (lowering.output_count == 0 || lowering.output_count > UINT32_MAX)
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                           "Metal physical output count is unsupported");
    lowering.plane_count = program.planes.size();

    std::ostringstream source;
    source << "#include <metal_stdlib>\nusing namespace metal;\n"
           << kPhysicalMetalPrelude;
    source << "kernel void laplace_physical_v1(";
    for (size_t plane = 0; plane < lowering.plane_count; ++plane) {
        if (plane != 0) source << ", ";
        source << "device const uchar* plane" << plane << " [[buffer(" << plane << ")]]";
    }
    if (lowering.plane_count != 0) source << ", ";
    source << "constant ulong* plane_offsets [[buffer(" << lowering.plane_count
           << ")]], device uint* output [[buffer(" << lowering.plane_count + 1
           << ")]], device atomic_uint* error [[buffer("
           << lowering.plane_count + 2
           << ")]], uint gid [[thread_position_in_grid]]) {\n"
           << "    if (gid >= " << lowering.output_count << "u) return;\n"
           << "    ulong remainder = ulong(gid);\n";
    for (size_t axis = logical.extents.size(); axis != 0; --axis) {
        source << "    ulong c" << axis - 1 << " = remainder % "
               << logical.extents[axis - 1] << "ul; remainder /= "
               << logical.extents[axis - 1] << "ul;\n";
    }
    for (size_t index = 0; index < program.instructions.size(); ++index) {
        if (program.instructions[index].opcode == PhysicalOpcode::Extent) {
            const uint64_t axis = program.instructions[index].immediate;
            source << "    ulong v" << index << " = "
                   << logical.extents[axis] << "ul;\n";
            continue;
        }
        PhysicalSourceContext context;
        context.plane_names.reserve(lowering.plane_count);
        context.plane_offset_indices.reserve(lowering.plane_count);
        for (size_t plane = 0; plane < lowering.plane_count; ++plane) {
            context.plane_names.push_back("plane" + std::to_string(plane));
            context.plane_offset_indices.push_back(plane);
        }
        const std::string instruction =
            physical_instruction_source(program, index, context);
        if (instruction.empty())
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "Metal physical opcode lowering is unavailable");
        source << instruction;
    }
    const PhysicalInstruction& result = program.instructions[program.result];
    const std::string result_name = physical_operand("v", program.result);
    if (result.result_type == PhysicalValueType::F32)
        source << "    output[gid] = as_type<uint>(" << result_name << ");\n";
    else if (result.result_type == PhysicalValueType::I32)
        source << "    output[gid] = as_type<uint>(" << result_name << ");\n";
    else
        source << "    output[gid] = uint(" << result_name << ");\n";
    source << "}\n";
    lowering.source = source.str();
    return lowering;
}

std::optional<std::string> scalar_body_source(
    const Region& body, uint32_t source_count, std::string* yielded) {
    if (!yielded || body.arguments.size() != source_count + 1 || body.yields.size() != 1)
        return std::nullopt;
    std::unordered_map<uint32_t, std::string> values;
    std::unordered_map<uint32_t, ElementType> types;
    for (uint32_t i = 0; i <= source_count; ++i) {
        values.emplace(body.arguments[i].id, i == source_count ? "acc" : "s" + std::to_string(i));
        types.emplace(body.arguments[i].id, body.arguments[i].type.element_type);
    }
    std::ostringstream source;
    size_t temporary = 0;
    for (const Instruction& instruction : body.instructions) {
        if (instruction.outputs.size() != 1 || !instruction.outputs.front().type.dimensions.empty())
            return std::nullopt;
        const ElementType type = instruction.outputs.front().type.element_type;
        const auto abi = scalar_abi(type);
        if (!abi.name) return std::nullopt;
        const std::string name = "t" + std::to_string(temporary++);
        std::vector<std::string> args;
        for (uint32_t id : instruction.inputs) {
            const auto value = values.find(id);
            if (value == values.end()) return std::nullopt;
            args.push_back(value->second);
        }
        const Primitive code = instruction.primitive.code;
        const auto* arithmetic = std::get_if<ArithmeticAttributes>(&instruction.attributes);
        const bool allow_nonfinite_operands = arithmetic && arithmetic->allow_nonfinite_operands;
        const bool allow_nonfinite_result = arithmetic && arithmetic->allow_nonfinite_result;
        std::string expression;
        if (code == Primitive::Constant) {
            const auto* constant = std::get_if<ConstantAttributes>(&instruction.attributes);
            if (!constant || !args.empty()) return std::nullopt;
            expression = scalar_literal(type, constant->bits);
        } else if (code == Primitive::TensorCoordinate) {
            const auto* coordinate = std::get_if<CoordinateAttributes>(&instruction.attributes);
            if (!coordinate || !args.empty()) return std::nullopt;
            expression = std::string(abi.name) + "(i" + std::to_string(coordinate->axis) + ")";
        } else if (code == Primitive::Require) {
            if (args.size() != 1) return std::nullopt;
            source << "        if (" << args[0] << " != 1u) { atomic_store_explicit(error, 1u, memory_order_relaxed); return; }\n";
            expression = args[0];
        } else if (code == Primitive::Select) {
            if (args.size() != 3) return std::nullopt;
            expression = "(" + args[0] + " != 0 ? " + args[1] + " : " + args[2] + ")";
        } else if (code == Primitive::Convert) {
            if (args.size() != 1) return std::nullopt;
            // Metal integer-to-float conversion uses round-to-nearest-even.
            expression = "float(" + args[0] + ")";
        } else {
            for (uint32_t id : instruction.inputs) {
                if (!allow_nonfinite_operands && types.at(id) == ElementType::F32)
                    source << "        if (!isfinite(" << values.at(id)
                           << ")) { atomic_store_explicit(error, 1u, memory_order_relaxed); return; }\n";
            }
            if (code == Primitive::Less || code == Primitive::Equal) {
                if (args.size() != 2) return std::nullopt;
                expression = "uchar(" + args[0] + (code == Primitive::Less ? " < " : " == ") + args[1] + ")";
            } else if (code == Primitive::Add || code == Primitive::Multiply ||
                       code == Primitive::Subtract || code == Primitive::Divide || code == Primitive::Maximum || code == Primitive::Pow) {
                if (args.size() != 2) return std::nullopt;
                const char* operation = code == Primitive::Add ? " + " : code == Primitive::Multiply ? " * " :
                                        code == Primitive::Subtract ? " - " : " / ";
                if (type == ElementType::F32) {
                    expression = code == Primitive::Maximum ? "(" + args[0] + " < " + args[1] + " ? " + args[1] + " : " + args[0] + ")" :
                        code == Primitive::Pow ? "pow(" + args[0] + ", " + args[1] + ")" :
                        args[0] + operation + args[1];
                } else if (type == ElementType::U32 || type == ElementType::I32) {
                    const char* wide = type == ElementType::I32 ? "long" : "ulong";
                    source << "        " << wide << " wide" << name << " = " << wide << "(" << args[0]
                           << ")" << operation << wide << "(" << args[1] << ");\n";
                    source << "        if (wide" << name
                           << (type == ElementType::I32 ? " < -2147483648l || wide" + name + " > 2147483647l" : " > 4294967295ul")
                           << ") { atomic_store_explicit(error, 1u, memory_order_relaxed); return; }\n";
                    expression = std::string(abi.name) + "(wide" + name + ")";
                } else if (type == ElementType::U64) {
                    source << "        if (";
                    if (code == Primitive::Add) source << args[0] << " > 18446744073709551615ul - " << args[1];
                    else source << args[0] << " != 0ul && " << args[1] << " > 18446744073709551615ul / " << args[0];
                    source << ") { atomic_store_explicit(error, 1u, memory_order_relaxed); return; }\n";
                    expression = args[0] + operation + args[1];
                } else return std::nullopt;
            } else {
                if (args.size() != 1) return std::nullopt;
                if (code == Primitive::RequireFinite) expression = args[0];
                else if (code == Primitive::Negate) expression = "-" + args[0];
                else {
                    const char* function = code == Primitive::Exp ? "exp" : code == Primitive::Log ? "log" :
                        code == Primitive::Rsqrt ? "rsqrt" : code == Primitive::Sin ? "sin" :
                        code == Primitive::Cos ? "cos" : code == Primitive::Sqrt ? "sqrt" :
                        code == Primitive::Tanh ? "tanh" : nullptr;
                    if (!function) return std::nullopt;
                    expression = std::string(function) + "(" + args[0] + ")";
                }
            }
        }
        source << "        " << abi.name << " " << name << " = " << expression << ";\n";
        if (!allow_nonfinite_result && type == ElementType::F32 && code != Primitive::Constant && code != Primitive::Select)
            source << "        if (!isfinite(" << name
                   << ")) { atomic_store_explicit(error, 1u, memory_order_relaxed); return; }\n";
        values.emplace(instruction.outputs.front().id, name);
        types.emplace(instruction.outputs.front().id, type);
    }
    const auto result = values.find(body.yields.front());
    if (result == values.end()) return std::nullopt;
    *yielded = result->second;
    return source.str();
}

std::variant<Lowering, CompatibilityReport> lower_program(
    const VerifiedProgram& verified,
    const VerifiedPhysicalProgramPackage* physical_package = nullptr) {
    const Program& program = program_definition(verified);
    if (program.functions.size() != 1 || program.exports.size() != 1 ||
        !program.state_references.empty() ||
        !program.dimension_parameters.empty()) {
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                           "Metal structured lowering requires one stateless exported function");
    }
    const Function& function = program.functions.front();
    const Region* root = find_region(function, function.entry_region_id);
    if (!root || root->yields.size() != 1 ||
        program.exports.front().function_id != function.id ||
        program.exports.front().result_index != 0)
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                           "Metal structured function boundary is unsupported");

    std::vector<BoundPhysicalSource> bound_sources;
    std::unordered_map<uint32_t, size_t> bound_value_indices;
    if (physical_package) {
        if (!physical_package->semantic_program() ||
            program_digest(*physical_package->semantic_program()) !=
                program_digest(verified))
            return metal_error(CompatibilityError::AUTHORITY_INVALID,
                               "Metal physical package semantic identity is invalid");
        for (const PhysicalResourceBinding& resource :
             physical_package->resources()) {
            if (resource.semantic_function_id != function.id ||
                resource.semantic_value_id == UINT32_MAX ||
                bound_value_indices.contains(resource.semantic_value_id))
                return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "Metal physical resource linkage is invalid");
            size_t program_index = 0;
            for (; program_index < physical_package->programs().size();
                 ++program_index) {
                const auto digest = physical_program_digest(
                    physical_package->programs()[program_index]);
                if (std::holds_alternative<PhysicalProgramDigest>(digest) &&
                    std::get<PhysicalProgramDigest>(digest) ==
                        resource.program_digest)
                    break;
            }
            if (program_index >= physical_package->programs().size() ||
                program_index >=
                    physical_package->program_logical_types().size())
                return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "Metal physical resource program is unavailable");
            bound_value_indices.emplace(resource.semantic_value_id,
                                        bound_sources.size());
            bound_sources.push_back(
                {&resource, &physical_package->programs()[program_index],
                 &physical_package->program_logical_types()[program_index], 0});
        }
    }

    const Instruction* structured = nullptr;
    std::unordered_map<uint32_t, const Instruction*> definitions;
    for (const Instruction& instruction : root->instructions) {
        for (const TypedValue& output : instruction.outputs)
            definitions.emplace(output.id, &instruction);
        if (instruction.primitive.code == Primitive::StructuredTensor) {
            if (structured)
                return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                                   "Metal structured lowering accepts one root tensor program");
            structured = &instruction;
        } else if (instruction.primitive.code != Primitive::Constant) {
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "Metal root instruction is not structurally lowerable");
        }
    }
    if (!structured || structured->outputs.size() != 1 ||
        root->yields.front() != structured->outputs.front().id ||
        structured->inputs.empty())
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                           "Metal structured root result is unsupported");
    const auto* attributes =
        std::get_if<StructuredTensorAttributes>(&structured->attributes);
    if (!attributes || attributes->source_count == 0 ||
        structured->inputs.size() != attributes->source_count + 1 ||
        attributes->indexing_maps.size() != structured->inputs.size() ||
        structured->regions.size() != 1)
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                           "Metal structured attributes are unsupported");
    const Region* body = find_region(function, structured->regions.front());
    if (!body)
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                           "Metal structured scalar body is unavailable");

    Lowering lowering;
    lowering.output_value_id = structured->outputs.front().id;
    lowering.output_type = structured->outputs.front().type.element_type;
    if (!constant_shape(structured->outputs.front().type,
                        &lowering.output_extents, &lowering.output_count) ||
        lowering.output_count == 0 || lowering.output_count > UINT32_MAX)
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                           "Metal structured output shape is unsupported");

    std::unordered_map<uint32_t, size_t> argument_indices;
    for (size_t index = 0; index < root->arguments.size(); ++index) {
        if (bound_value_indices.contains(root->arguments[index].id)) continue;
        InputSpec spec;
        spec.value_id = root->arguments[index].id;
        spec.type = root->arguments[index].type.element_type;
        if (!constant_shape(root->arguments[index].type, &spec.extents,
                            &spec.element_count))
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "Metal structured input type is unsupported");
        argument_indices.emplace(spec.value_id, lowering.inputs.size());
        lowering.inputs.push_back(std::move(spec));
    }

    for (BoundPhysicalSource& bound : bound_sources) {
        bound.plane_base = lowering.physical_planes.size();
        for (uint16_t plane = 0; plane < bound.program->planes.size(); ++plane)
            lowering.physical_planes.push_back(
                {bound.resource->resource_id, plane});
    }

    std::vector<uint64_t> iteration_extents;
    iteration_extents.reserve(attributes->iteration_dimensions.size());
    for (const DimensionExpr& dimension : attributes->iteration_dimensions) {
        if (dimension.expression != DimensionExpression::Constant ||
            dimension.value == 0 || dimension.value > INT64_MAX)
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "Metal structured iteration shape is unsupported");
        iteration_extents.push_back(dimension.value);
    }

    const TensorIndexMap& destination_map = attributes->indexing_maps.back();
    std::vector<int32_t> iterator_to_output(iteration_extents.size(), -1);
    for (size_t axis = 0; axis < destination_map.results.size(); ++axis) {
        const TensorIndexExpr& expression = destination_map.results[axis];
        if (expression.expression != TensorIndexExpression::Iterator ||
            expression.value < 0 ||
            static_cast<uint64_t>(expression.value) >= iteration_extents.size())
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "Metal destination map is unsupported");
        iterator_to_output[static_cast<size_t>(expression.value)] =
            static_cast<int32_t>(axis);
    }

    struct SourceBinding {
        bool physical = false;
        size_t index = 0;
        const Instruction* constant = nullptr;
    };
    std::vector<std::vector<uint64_t>> constant_shapes;
    std::vector<SourceBinding> source_bindings;
    for (uint32_t source = 0; source < attributes->source_count; ++source) {
        const auto definition = definitions.find(structured->inputs[source]);
        if (definition != definitions.end() && definition->second->primitive.code == Primitive::Constant) {
            std::vector<uint64_t> extents;
            size_t count = 0;
            if (!constant_shape(definition->second->outputs.front().type, &extents, &count))
                return metal_error(CompatibilityError::IR_SHAPE_MISMATCH, "Metal constant source shape is invalid");
            source_bindings.push_back({false, constant_shapes.size(), definition->second});
            constant_shapes.push_back(std::move(extents));
            continue;
        }

        const auto argument = argument_indices.find(structured->inputs[source]);
        if (argument != argument_indices.end()) {
            source_bindings.push_back({false, argument->second});
            continue;
        }
        const auto bound = bound_value_indices.find(structured->inputs[source]);
        if (bound == bound_value_indices.end())
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "Metal source is not an external tensor binding");
        source_bindings.push_back({true, bound->second});
    }

    std::vector<std::string> source_scalar_expressions(attributes->source_count);
    std::vector<std::string> source_element_expressions(attributes->source_count);
    std::ostringstream indexed_loaders;
    for (uint32_t source = 0; source < attributes->source_count; ++source) {
        const SourceBinding& binding = source_bindings[source];
        const ElementType type = binding.constant ? binding.constant->outputs.front().type.element_type :
            binding.physical ? bound_sources[binding.index].logical->element_type : lowering.inputs[binding.index].type;
        if (type != ElementType::U32) continue;
        const auto& extents = binding.constant ? constant_shapes[binding.index] :
            binding.physical ? bound_sources[binding.index].logical->extents : lowering.inputs[binding.index].extents;
        const std::string name = "laplace_index_source" + std::to_string(source);
        std::string call = name + "(";
        indexed_loaders << "inline long " << name << "(";
        PhysicalSourceContext context;
        context.value_prefix = "iv";
        context.coordinate_prefix = "ic";
        if (binding.physical) {
            const auto& bound = bound_sources[binding.index];
            for (size_t plane = 0; plane < bound.program->planes.size(); ++plane) {
                indexed_loaders << "device const uchar* p" << plane << ", ";
                call += "physical" + std::to_string(bound.plane_base + plane) + ", ";
                context.plane_names.push_back("p" + std::to_string(plane));
                context.plane_offset_indices.push_back(bound.plane_base + plane);
            }
            indexed_loaders << "constant ulong* plane_offsets, ";
            call += "physical_offsets, ";
        } else if (!binding.constant) {
            indexed_loaders << "device const uint* values, ";
            call += "input" + std::to_string(binding.index) + ", ";
        }
        indexed_loaders << "device atomic_uint* error";
        call += "error";
        source_element_expressions[source] = call;
        if (extents.empty()) source_scalar_expressions[source] = call + ")";
        for (size_t axis = 0; axis < extents.size(); ++axis)
            indexed_loaders << ", long c" << axis;
        indexed_loaders << ") {\n";
        for (size_t axis = 0; axis < extents.size(); ++axis) {
            indexed_loaders << "    if (c" << axis << " < 0l || ulong(c" << axis << ") >= "
                << extents[axis] << "ul) { atomic_store_explicit(error, 2u, memory_order_relaxed); return 0l; }\n";
        }
        if (binding.physical) {
            const auto& bound = bound_sources[binding.index];
            for (size_t axis = 0; axis < extents.size(); ++axis)
                indexed_loaders << "    ulong ic" << axis << " = ulong(c" << axis << ");\n";
            for (size_t i = 0; i < bound.program->instructions.size(); ++i) {
                const auto& instruction = bound.program->instructions[i];
                if (instruction.opcode == PhysicalOpcode::Extent)
                    indexed_loaders << "    ulong iv" << i << " = " << extents[instruction.immediate] << "ul;\n";
                else indexed_loaders << physical_instruction_source(*bound.program, i, context);
            }
            indexed_loaders << "    return long(iv" << bound.program->result << ");\n}\n";
        } else if (binding.constant) {
            indexed_loaders << "    return " << std::get<ConstantAttributes>(binding.constant->attributes).bits << "l;\n}\n";
        } else {
            indexed_loaders << "    ulong offset = 0ul;\n";
            for (size_t axis = 0; axis < extents.size(); ++axis)
                indexed_loaders << "    offset = offset * " << extents[axis] << "ul + ulong(c" << axis << ");\n";
            indexed_loaders << "    return long(values[offset]);\n}\n";
        }
    }

    bool initial_is_constant = false;
    uint64_t initial_bits = 0;
    size_t initial_buffer = 0;
    const uint32_t initial_id = structured->inputs.back();
    if (const auto argument = argument_indices.find(initial_id);
        argument != argument_indices.end()) {
        initial_buffer = argument->second;
    } else {
        const auto definition = definitions.find(initial_id);
        if (definition == definitions.end() ||
            definition->second->primitive.code != Primitive::Constant) {
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "Metal destination initializer is unsupported");
        }
        const auto* constant =
            std::get_if<ConstantAttributes>(&definition->second->attributes);
        if (!constant ||
            definition->second->outputs.front().type !=
                structured->outputs.front().type)
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "Metal destination initializer is invalid");
        initial_is_constant = true;
        initial_bits = constant->bits;
    }

    std::string yielded;
    const auto body_source = scalar_body_source(
        *body, attributes->source_count, &yielded);
    if (!body_source)
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                           "Metal scalar body is unsupported");

    std::ostringstream source;
    source << "#include <metal_stdlib>\nusing namespace metal;\n";
    if (!lowering.physical_planes.empty()) source << kPhysicalMetalPrelude;
    source << R"METAL(
inline long laplace_index_add(long a, long b, device atomic_uint* error) {
    if ((b > 0l && a > 9223372036854775807l - b) ||
        (b < 0l && a < (-9223372036854775807l - 1l) - b)) {
        atomic_store_explicit(error, 2u, memory_order_relaxed); return 0l;
    }
    return a + b;
}
inline long laplace_index_mul(long a, long b, device atomic_uint* error) {
    const long lo = -9223372036854775807l - 1l;
    const long hi = 9223372036854775807l;
    if ((a > 0l && ((b > 0l && a > hi / b) || (b < 0l && b < lo / a))) ||
        (a < 0l && ((b > 0l && a < lo / b) || (b < 0l && a < hi / b)))) {
        atomic_store_explicit(error, 2u, memory_order_relaxed); return 0l;
    }
    return a * b;
}
inline long laplace_floor_div(long a, long b) {
    long q = a / b; long r = a % b; return q - long(r < 0l);
}
inline long laplace_floor_mod(long a, long b) {
    long r = a % b; return r < 0l ? r + b : r;
}
)METAL";
    source << indexed_loaders.str() << "kernel void laplace_structured_v1(";
    bool has_argument = false;
    for (size_t input = 0; input < lowering.inputs.size(); ++input) {
        if (has_argument) source << ", ";
        source << "device const "
               << scalar_abi(lowering.inputs[input].type).name
               << "* input" << input
               << " [[buffer(" << input << ")]]";
        has_argument = true;
    }
    const size_t physical_base = lowering.inputs.size();
    for (size_t plane = 0; plane < lowering.physical_planes.size(); ++plane) {
        if (has_argument) source << ", ";
        source << "device const uchar* physical" << plane << " [[buffer("
               << physical_base + plane << ")]]";
        has_argument = true;
    }
    size_t output_buffer = lowering.inputs.size();
    if (!lowering.physical_planes.empty()) {
        const size_t offsets_buffer =
            physical_base + lowering.physical_planes.size();
        source << ", constant ulong* physical_offsets [[buffer("
               << offsets_buffer << ")]]";
        output_buffer = offsets_buffer + 1;
    }
    const size_t error_buffer = output_buffer + 1;
    if (has_argument) source << ", ";
    source << "device " << scalar_abi(lowering.output_type).name
           << "* output [[buffer(" << output_buffer << ")]]";
    source << ", device atomic_uint* error [[buffer(" << error_buffer
           << ")]]";
    source << ", uint gid [[thread_position_in_grid]]) {\n"
           << "    if (gid >= " << lowering.output_count << "u) return;\n"
           << "    ulong out_remainder = ulong(gid);\n";
    for (size_t axis = lowering.output_extents.size(); axis != 0; --axis) {
        source << "    long o" << (axis - 1) << " = long(out_remainder % "
               << lowering.output_extents[axis - 1]
               << "ul); out_remainder /= " << lowering.output_extents[axis - 1]
               << "ul;\n";
    }
    for (size_t iterator = 0; iterator < iteration_extents.size(); ++iterator) {
        if (attributes->iterator_kinds[iterator] == TensorIteratorKind::Parallel) {
            const int32_t axis = iterator_to_output[iterator];
            if (axis < 0)
                return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                                   "Metal parallel iterator is not mapped to output");
            source << "    long i" << iterator << " = o" << axis << ";\n";
        }
    }
    source << "    " << scalar_abi(lowering.output_type).name << " acc = ";
    if (initial_is_constant) {
        source << scalar_literal(lowering.output_type, initial_bits) << ";\n";
    } else {
        source << "input" << initial_buffer << "[gid];\n";
    }
    size_t indent = 1;
    for (size_t iterator = 0; iterator < iteration_extents.size(); ++iterator) {
        if (attributes->iterator_kinds[iterator] != TensorIteratorKind::Reduction)
            continue;
        source << std::string(indent * 4, ' ') << "for (long i" << iterator
               << " = 0; i" << iterator << " < " << iteration_extents[iterator]
               << "l; ++i" << iterator << ") {\n";
        ++indent;
    }
    for (uint32_t source_index = 0; source_index < attributes->source_count;
         ++source_index) {
        const TensorIndexMap& index_map = attributes->indexing_maps[source_index];
        const SourceBinding& binding = source_bindings[source_index];
        const std::vector<uint64_t>& extents = binding.constant ? constant_shapes[binding.index] :
            binding.physical ? bound_sources[binding.index].logical->extents : lowering.inputs[binding.index].extents;
        if (index_map.results.size() != extents.size())
            return metal_error(CompatibilityError::IR_SHAPE_MISMATCH,
                               "Metal source index rank is invalid");
        std::vector<std::string> coordinates;
        coordinates.reserve(index_map.results.size());
        for (const TensorIndexExpr& expression : index_map.results) {
            std::string coordinate;
            if (!emit_index(expression, &coordinate,
                            source_scalar_expressions, source_element_expressions))
                return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                                   "Metal source index expression is unsupported");
            coordinates.push_back(std::move(coordinate));
        }
        const std::string padding(indent * 4, ' ');
        const ElementType source_type = body->arguments[source_index].type.element_type;
        if (!scalar_abi(source_type).bytes)
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "Metal scalar source type is unsupported");
        source << padding
               << scalar_abi(source_type).name
               << " s" << source_index << " = "
               << scalar_literal(source_type, 0) << ";\n";
        std::string condition = "true";
        for (size_t axis = 0; axis < coordinates.size(); ++axis) {
            source << padding << "long s" << source_index << "c" << axis
                   << " = " << coordinates[axis] << ";\n";
            condition += " && s" + std::to_string(source_index) + "c" +
                         std::to_string(axis) + " >= 0 && ulong(s" +
                         std::to_string(source_index) + "c" +
                         std::to_string(axis) + ") < " +
                         std::to_string(extents[axis]) + "ul";
        }
        source << padding << "if (" << condition << ") {\n";
        if (binding.constant) {
            source << padding << "    s" << source_index << " = " << scalar_literal(source_type,
                std::get<ConstantAttributes>(binding.constant->attributes).bits) << ";\n";
        } else if (!binding.physical) {
            source << padding << "    ulong offset = 0ul;\n";
            for (size_t axis = 0; axis < coordinates.size(); ++axis)
                source << padding << "    offset = offset * " << extents[axis]
                       << "ul + ulong(s" << source_index << "c" << axis
                       << ");\n";
            source << padding << "    s" << source_index << " = input"
                   << binding.index << "[offset];\n";
        } else {
            const BoundPhysicalSource& bound = bound_sources[binding.index];
            PhysicalSourceContext context;
            context.value_prefix = "r" + std::to_string(binding.index) + "v";
            context.coordinate_prefix =
                "r" + std::to_string(binding.index) + "c";
            context.offsets_name = "physical_offsets";
            context.error_name = "error";
            context.indent = padding + "    ";
            for (size_t axis = 0; axis < extents.size(); ++axis)
                source << context.indent << "ulong " << context.coordinate_prefix
                       << axis << " = ulong(s" << source_index << "c" << axis
                       << ");\n";
            for (size_t plane = 0; plane < bound.program->planes.size(); ++plane) {
                const size_t flat = bound.plane_base + plane;
                context.plane_names.push_back(
                    "physical" + std::to_string(flat));
                context.plane_offset_indices.push_back(flat);
            }
            for (size_t instruction = 0;
                 instruction < bound.program->instructions.size(); ++instruction) {
                if (bound.program->instructions[instruction].opcode ==
                    PhysicalOpcode::Extent) {
                    const uint64_t axis =
                        bound.program->instructions[instruction].immediate;
                    source << context.indent << "ulong "
                           << physical_operand(
                                  context.value_prefix,
                                  static_cast<uint32_t>(instruction))
                           << " = " << extents[axis] << "ul;\n";
                    continue;
                }
                const std::string instruction_source =
                    physical_instruction_source(*bound.program, instruction,
                                                context);
                if (instruction_source.empty())
                    return metal_error(
                        CompatibilityError::KERNEL_UNAVAILABLE,
                        "Metal bound physical opcode lowering is unavailable");
                source << instruction_source;
            }
            source << context.indent << "s" << source_index << " = "
                   << physical_operand(context.value_prefix,
                                       bound.program->result)
                   << ";\n";
        }
        source << padding << "}";
        if (index_map.bounds == TensorBoundsMode::Reject)
            source << " else { atomic_store_explicit(error, 2u, "
                      "memory_order_relaxed); return; }";
        source << "\n";
    }
    std::string body_text = *body_source;
    size_t position = 0;
    while ((position = body_text.find("        ", position)) != std::string::npos) {
        body_text.replace(position, 8, std::string(indent * 4, ' '));
        position += indent * 4;
    }
    source << body_text
           << std::string(indent * 4, ' ') << "acc = " << yielded << ";\n";
    while (indent > 1) {
        --indent;
        source << std::string(indent * 4, ' ') << "}\n";
    }
    source << "    output[gid] = acc;\n}\n";
    lowering.source = source.str();
    return lowering;
}

std::variant<MultiLowering, CompatibilityReport> lower_multi_program(
    const VerifiedProgram& verified,
    const VerifiedPhysicalProgramPackage* physical_package = nullptr) {
    const Program& program = program_definition(verified);
    if (physical_package &&
        (!physical_package->semantic_program() ||
         program_digest(*physical_package->semantic_program()) !=
             program_digest(verified)))
        return metal_error(CompatibilityError::AUTHORITY_INVALID,
                           "Metal staged physical authority is invalid");
    if (program.functions.size() != 1 || program.exports.size() != 1 ||
        !program.dimension_parameters.empty()) {
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                           "Metal staged lowering requires one exported function");
    }
    const Function& function = program.functions.front();
    const Region* root = find_region(function, function.entry_region_id);
    if (!root || root->yields.size() != 1 ||
        program.exports.front().function_id != function.id ||
        program.exports.front().result_index != 0)
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                           "Metal staged function boundary is unsupported");

    std::unordered_map<uint32_t, ValueType> types;
    std::unordered_map<uint32_t, const Instruction*> definitions;
    std::unordered_set<uint32_t> external_values;
    for (const TypedValue& argument : root->arguments) {
        types.emplace(argument.id, argument.type);
        external_values.insert(argument.id);
    }
    size_t stage_count = 0;
    for (const Instruction& instruction : root->instructions) {
        if (instruction.primitive.code != Primitive::Constant &&
            instruction.primitive.code != Primitive::StructuredTensor &&
            instruction.primitive.code != Primitive::StateRead &&
            instruction.primitive.code != Primitive::StateWrite)
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "Metal staged root instruction is unsupported");
        if (instruction.primitive.code == Primitive::StructuredTensor)
            ++stage_count;
        for (const TypedValue& output : instruction.outputs) {
            types.emplace(output.id, output.type);
            definitions.emplace(output.id, &instruction);
        }
    }
    if (stage_count < 2)
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                           "Metal staged lowering requires multiple tensor stages");

    MultiLowering result;
    std::unordered_set<uint32_t> physical_inputs;
    if (physical_package) {
        for (const PhysicalResourceBinding& resource :
             physical_package->resources()) {
            if (resource.semantic_function_id == function.id)
                physical_inputs.insert(resource.semantic_value_id);
        }
    }
    result.inputs.reserve(root->arguments.size());
    for (const TypedValue& argument : root->arguments) {
        if (physical_inputs.contains(argument.id)) continue;
        InputSpec spec;
        spec.value_id = argument.id;
        spec.type = argument.type.element_type;
        if (!constant_shape(argument.type, &spec.extents, &spec.element_count))
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "Metal staged input type is unsupported");
        result.inputs.push_back(std::move(spec));
    }

    std::unordered_map<uint32_t, size_t> state_indices;
    result.states.reserve(program.state_references.size());
    for (const StateReference& reference : program.state_references) {
        MultiLowering::State state;
        state.state_id = reference.id;
        state.type = reference.type.element_type;
        if (!constant_shape(reference.type, &state.extents,
                            &state.element_count))
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "Metal staged state type is unsupported");
        state_indices.emplace(state.state_id, result.states.size());
        result.states.push_back(std::move(state));
    }

    std::unordered_set<uint32_t> available = external_values;
    for (const Instruction& instruction : root->instructions) {
        if (instruction.primitive.code == Primitive::StateRead) {
            const auto* attributes =
                std::get_if<StateAttributes>(&instruction.attributes);
            const auto state = attributes
                ? state_indices.find(attributes->state_id)
                : state_indices.end();
            if (state == state_indices.end() || instruction.outputs.size() != 1 ||
                !instruction.inputs.empty() ||
                result.states[state->second].write_value_id != UINT32_MAX)
                return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                                   "Metal staged state read is unsupported");
            const uint32_t value = instruction.outputs.front().id;
            result.states[state->second].read_value_ids.push_back(value);
            available.insert(value);
            continue;
        }
        if (instruction.primitive.code == Primitive::StateWrite) {
            const auto* attributes =
                std::get_if<StateAttributes>(&instruction.attributes);
            const auto state = attributes
                ? state_indices.find(attributes->state_id)
                : state_indices.end();
            if (state == state_indices.end() || instruction.inputs.size() != 1 ||
                !instruction.outputs.empty() ||
                !available.contains(instruction.inputs.front()) ||
                result.states[state->second].write_value_id != UINT32_MAX)
                return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                                   "Metal staged state write is unsupported");
            result.states[state->second].write_value_id =
                instruction.inputs.front();
            continue;
        }
        if (instruction.primitive.code != Primitive::StructuredTensor)
            continue;
        if (instruction.outputs.size() != 1 || instruction.regions.size() != 1)
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "Metal staged tensor result is unsupported");
        const Region* body = find_region(function, instruction.regions.front());
        if (!body)
            return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                               "Metal staged tensor body is unavailable");

        Region stage_root;
        stage_root.id = root->id;
        std::vector<uint32_t> original_arguments;
        std::unordered_set<uint32_t> seen_arguments;
        for (uint32_t input : instruction.inputs) {
            const auto definition = definitions.find(input);
            if (definition != definitions.end() &&
                definition->second->primitive.code == Primitive::Constant) {
                stage_root.instructions.push_back(*definition->second);
                continue;
            }
            if (!available.contains(input))
                return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "Metal staged input is not available");
            if (!seen_arguments.insert(input).second) continue;
            const auto type = types.find(input);
            if (type == types.end())
                return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "Metal staged input type is unavailable");
            stage_root.arguments.push_back({input, type->second});
            original_arguments.push_back(input);
        }
        stage_root.instructions.push_back(instruction);
        stage_root.yields = {instruction.outputs.front().id};
        Function stage_function;
        stage_function.id = function.id;
        stage_function.entry_region_id = stage_root.id;
        stage_function.regions = {*body, std::move(stage_root)};
        stage_function.result_types = {instruction.outputs.front().type};
        Program stage_program;
        stage_program.major = program.major;
        stage_program.minor = program.minor;
        stage_program.functions = {std::move(stage_function)};
        stage_program.exports = {
            {function.id, 0, instruction.outputs.front().type}};
        auto verified_stage =
            verify_and_canonicalize_program(std::move(stage_program));
        if (const auto* report =
                std::get_if<CompatibilityReport>(&verified_stage))
            return *report;
        VerifiedProgram stage_verified =
            std::get<VerifiedProgram>(std::move(verified_stage));
        const Program& stage_definition = program_definition(stage_verified);
        const Region* canonical_root = find_region(
            stage_definition.functions.front(),
            stage_definition.functions.front().entry_region_id);
        if (!canonical_root ||
            canonical_root->arguments.size() != original_arguments.size())
            return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                               "Metal staged canonical arguments are invalid");
        std::unordered_map<uint32_t, uint32_t> canonical_to_original;
        for (size_t index = 0; index < original_arguments.size(); ++index)
            canonical_to_original.emplace(canonical_root->arguments[index].id,
                                          original_arguments[index]);

        std::optional<VerifiedPhysicalProgramPackage> stage_package;
        if (physical_package) {
            std::vector<PhysicalResourceBinding> stage_resources;
            for (const PhysicalResourceBinding& resource :
                 physical_package->resources()) {
                if (resource.semantic_function_id != function.id) continue;
                const auto argument = std::find(
                    original_arguments.begin(), original_arguments.end(),
                    resource.semantic_value_id);
                if (argument == original_arguments.end()) continue;
                PhysicalResourceBinding rebound = resource;
                rebound.semantic_function_id =
                    stage_definition.functions.front().id;
                rebound.semantic_value_id =
                    canonical_root->arguments[static_cast<size_t>(
                        argument - original_arguments.begin())].id;
                stage_resources.push_back(std::move(rebound));
            }
            if (!stage_resources.empty()) {
                std::sort(stage_resources.begin(), stage_resources.end(),
                          [](const auto& left, const auto& right) {
                              return left.resource_id < right.resource_id;
                          });
                std::vector<PhysicalProgramRecord> records;
                const auto programs = physical_package->programs();
                const auto logical_types =
                    physical_package->program_logical_types();
                if (programs.size() != logical_types.size())
                    return metal_error(
                        CompatibilityError::IMPORT_MANIFEST_INVALID,
                        "Metal staged physical program metadata is invalid");
                for (size_t index = 0; index < programs.size(); ++index) {
                    const auto digest = physical_program_digest(programs[index]);
                    if (!std::holds_alternative<PhysicalProgramDigest>(digest))
                        return std::get<CompatibilityReport>(digest);
                    const PhysicalProgramDigest identity =
                        std::get<PhysicalProgramDigest>(digest);
                    const bool referenced = std::any_of(
                        stage_resources.begin(), stage_resources.end(),
                        [&](const auto& resource) {
                            return resource.program_digest == identity;
                        });
                    if (!referenced) continue;
                    const auto wire = encode_physical_program(programs[index]);
                    if (!std::holds_alternative<std::vector<uint8_t>>(wire))
                        return std::get<CompatibilityReport>(wire);
                    records.push_back({
                        identity,
                        std::get<std::vector<uint8_t>>(wire),
                        logical_types[index]});
                }
                auto loaded = load_physical_program_package(
                    physical_package->physical_index(),
                    std::move(stage_verified), records, stage_resources);
                if (const auto* report =
                        std::get_if<CompatibilityReport>(&loaded))
                    return *report;
                stage_package = std::get<VerifiedPhysicalProgramPackage>(
                    std::move(loaded));
            }
        }

        auto lowered = stage_package
            ? lower_program(*stage_package->semantic_program(),
                            &*stage_package)
            : lower_program(stage_verified);
        if (const auto* report = std::get_if<CompatibilityReport>(&lowered))
            return *report;
        Lowering stage = std::get<Lowering>(std::move(lowered));
        for (InputSpec& input : stage.inputs) {
            const auto original = canonical_to_original.find(input.value_id);
            if (original == canonical_to_original.end())
                return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "Metal staged input remapping is invalid");
            input.value_id = original->second;
        }
        stage.output_value_id = instruction.outputs.front().id;
        available.insert(stage.output_value_id);
        result.stages.push_back(std::move(stage));
    }
    const uint32_t exported = root->yields.front();
    const auto output = std::find_if(result.stages.begin(), result.stages.end(),
        [&](const Lowering& stage) { return stage.output_value_id == exported; });
    if (!available.contains(exported) || output == result.stages.end())
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                           "Metal staged export is unsupported");
    result.exports = {exported};
    result.output_type = output->output_type;
    result.output_extents = output->output_extents;
    result.output_count = output->output_count;
    // A stage may own its destination when exactly one StateWrite consumes
    // its value. Later stages already wait on the existing stage barrier.
    for (Lowering& stage : result.stages) {
        size_t matching_states = 0;
        for (size_t index = 0; index < result.states.size(); ++index) {
            const MultiLowering::State& state = result.states[index];
            if (state.write_value_id != stage.output_value_id ||
                state.type != stage.output_type ||
                state.extents != stage.output_extents ||
                state.element_count != stage.output_count)
                continue;
            ++matching_states;
            stage.direct_state_index = static_cast<uint32_t>(index);
        }
        if (matching_states != 1)
            stage.direct_state_index = UINT32_MAX;
    }
    return result;
}

bool value_matches(const MetalProgramValue& value, const InputSpec& spec) {
    if (value.type != spec.type || value.extents != spec.extents ||
        value.bits.size() != spec.element_count)
        return false;
    return std::all_of(value.bits.begin(), value.bits.end(),
                       [&](uint64_t bits) {
                           const size_t width = scalar_abi(value.type).bytes;
                           return value.type == ElementType::I1 ? bits <= 1 :
                               width == 8 || bits < (uint64_t{1} << (width * 8));
                       });
}

} // namespace

struct MetalProgramExecutable::Impl {
    struct Plane {
        id<MTLBuffer> buffer = nil;
        uint32_t resource_id = UINT32_MAX;
        uint32_t plane_index = UINT32_MAX;
        uint64_t byte_offset = 0;
        uint64_t length = 0;
        bool zero_copy = false;
    };
    struct Stage {
        id<MTLComputePipelineState> pipeline = nil;
        std::vector<InputSpec> inputs;
        ElementType output_type = ElementType::F32;
    std::vector<uint64_t> output_extents;
        size_t output_count = 0;
        uint32_t output_value_id = UINT32_MAX;
        uint32_t direct_state_index = UINT32_MAX;
        std::vector<size_t> physical_plane_indices;
        id<MTLBuffer> plane_offsets = nil;
        id<MTLBuffer> output = nil;
    };
    struct State {
        uint32_t state_id = UINT32_MAX;
        ElementType type = ElementType::F32;
        std::vector<uint64_t> extents;
        size_t element_count = 0;
        std::vector<uint32_t> read_value_ids;
        uint32_t write_value_id = UINT32_MAX;
        bool direct_output = false;
        id<MTLBuffer> current = nil;
        id<MTLBuffer> candidate = nil;
        id<MTLBuffer> scratch = nil;
    };
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLBuffer> plane_offsets = nil;
    ProgramDigest digest{};
    std::array<uint8_t, 32> lowering{};
    std::vector<InputSpec> inputs;
    std::vector<Stage> stages;
    std::vector<State> states;
    std::vector<uint32_t> exports;
    uint64_t state_generation = 0;
    std::mutex execution_mutex;
    id<MTLBuffer> output = nil;
    id<MTLBuffer> numerical_error = nil;
    uint64_t intermediate_buffer_bytes = 0;
    std::vector<Plane> planes;
    std::unordered_map<uint32_t, id<MTLBuffer>> mapped_artifacts;
    std::shared_ptr<const VerifiedPhysicalProgramPackage> package_owner;
    uint64_t persistent_upload_bytes = 0;
    uint64_t zero_copy_plane_bytes = 0;
    uint64_t persistent_plane_bytes = 0;
    ElementType output_type = ElementType::F32;
    std::vector<uint64_t> output_extents;
    size_t output_count = 0;

    id<MTLBuffer> mapped_buffer(const PackageView& artifact, size_t length) {
        const auto found = mapped_artifacts.find(artifact.artifact_id().value);
        if (found != mapped_artifacts.end()) return [found->second retain];
        id<MTLBuffer> buffer = [device
            newBufferWithBytesNoCopy:const_cast<uint8_t*>(artifact.bytes().data())
                             length:length
                            options:MTLResourceStorageModeShared
                        deallocator:nil];
        if (buffer) mapped_artifacts.emplace(artifact.artifact_id().value, buffer);
        return [buffer retain];
    }

    bool allocate_workspace() {
        numerical_error = [device newBufferWithLength:sizeof(uint32_t)
                                              options:MTLResourceStorageModeShared];
        if (!numerical_error) return false;
        if (stages.empty()) {
            output = [device newBufferWithLength:output_count * scalar_abi(output_type).bytes
                                         options:MTLResourceStorageModeShared];
            if (!output) return false;
            intermediate_buffer_bytes = output.length;
        } else {
            for (Stage& stage : stages) {
                if (stage.direct_state_index != UINT32_MAX) continue;
                stage.output = [device newBufferWithLength:stage.output_count * scalar_abi(stage.output_type).bytes
                                                   options:MTLResourceStorageModeShared];
                if (!stage.output) return false;
                intermediate_buffer_bytes += stage.output.length;
            }
        }
        return true;
    }

    ~Impl() {
        [output release];
        [numerical_error release];
        for (const Stage& stage : stages) {
            [stage.pipeline release];
            [stage.plane_offsets release];
            [stage.output release];
        }
        for (const State& state : states) {
            [state.current release];
            [state.candidate release];
            [state.scratch release];
        }
        for (const Plane& plane : planes) [plane.buffer release];
        for (const auto& [id, buffer] : mapped_artifacts) [buffer release];
        [plane_offsets release];
        [pipeline release];
        [queue release];
        [device release];
    }
};

MetalProgramExecutable::MetalProgramExecutable(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MetalProgramExecutable::~MetalProgramExecutable() = default;
MetalProgramExecutable::MetalProgramExecutable(MetalProgramExecutable&&) noexcept = default;
MetalProgramExecutable& MetalProgramExecutable::operator=(
    MetalProgramExecutable&&) noexcept = default;

ProgramDigest MetalProgramExecutable::program_digest() const noexcept {
    return impl_ ? impl_->digest : ProgramDigest{};
}

std::array<uint8_t, 32> MetalProgramExecutable::lowering_digest() const noexcept {
    return impl_ ? impl_->lowering : std::array<uint8_t, 32>{};
}

MetalProgramExecutionResult MetalProgramExecutable::execute(
    std::span<const MetalProgramInput> inputs) const {
    if (!impl_ || inputs.size() != impl_->inputs.size())
        return metal_error(CompatibilityError::RUNTIME_INPUT_INVALID,
                           "Metal program input count is invalid");
    std::lock_guard lock(impl_->execution_mutex);
    @autoreleasepool {
        std::vector<const MetalProgramInput*> ordered(impl_->inputs.size(), nullptr);
        std::unordered_set<uint32_t> seen;
        for (const MetalProgramInput& input : inputs) {
            if (!seen.insert(input.value_id).second)
                return metal_error(CompatibilityError::RUNTIME_INPUT_INVALID,
                                   "Metal program input identity is duplicated");
            const auto expected = std::find_if(
                impl_->inputs.begin(), impl_->inputs.end(),
                [&](const InputSpec& spec) { return spec.value_id == input.value_id; });
            if (expected == impl_->inputs.end() ||
                !value_matches(input.value, *expected))
                return metal_error(CompatibilityError::RUNTIME_INPUT_INVALID,
                                   "Metal program input type or shape is invalid");
            ordered[static_cast<size_t>(expected - impl_->inputs.begin())] = &input;
        }
        if (std::find(ordered.begin(), ordered.end(), nullptr) != ordered.end())
            return metal_error(CompatibilityError::RUNTIME_INPUT_INVALID,
                               "Metal program input is missing");

        std::vector<id<MTLBuffer>> buffers;
        buffers.reserve(ordered.size());
        uint64_t upload_bytes = impl_->persistent_upload_bytes;
        for (const MetalProgramInput* input : ordered) {
            const auto bits = pack_value(input->value);
            const NSUInteger bytes = bits.size();
            id<MTLBuffer> buffer = [impl_->device
                newBufferWithBytes:bits.data()
                           length:bytes
                          options:MTLResourceStorageModeShared];
            if (!buffer) {
                for (id<MTLBuffer> value : buffers) [value release];
                return metal_error(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                                   "Metal program input buffer allocation failed");
            }
            buffers.push_back(buffer);
            upload_bytes += bytes;
        }
        if (!impl_->stages.empty()) {
            const auto release_inputs = [&] {
                for (id<MTLBuffer> value : buffers) [value release];
            };
            id<MTLBuffer> error_buffer = impl_->numerical_error;
            *static_cast<uint32_t*>(error_buffer.contents) = 0;
            id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            if (!command || !encoder) {
                if (encoder) [encoder endEncoding];
                release_inputs();
                return metal_error(
                    CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                    "Metal staged command allocation failed");
            }
            std::unordered_map<uint32_t, id<MTLBuffer>> value_buffers;
            for (size_t index = 0; index < impl_->inputs.size(); ++index)
                value_buffers.emplace(impl_->inputs[index].value_id,
                                      buffers[index]);
            for (const Impl::State& state : impl_->states) {
                for (uint32_t value_id : state.read_value_ids)
                    value_buffers.emplace(value_id, state.current);
            }
            uint32_t minimum_width = UINT32_MAX;
            uint32_t minimum_maximum = UINT32_MAX;
            bool encode_failed = false;
            for (size_t stage_index = 0;
                 stage_index < impl_->stages.size(); ++stage_index) {
                const Impl::Stage& stage = impl_->stages[stage_index];
                id<MTLBuffer> stage_output = stage.output;
                if (stage.direct_state_index != UINT32_MAX) {
                    if (stage.direct_state_index >= impl_->states.size()) {
                        encode_failed = true;
                        break;
                    }
                    stage_output =
                        impl_->states[stage.direct_state_index].candidate;
                }
                [encoder setComputePipelineState:stage.pipeline];
                for (NSUInteger index = 0; index < stage.inputs.size(); ++index) {
                    const auto value =
                        value_buffers.find(stage.inputs[index].value_id);
                    if (value == value_buffers.end()) {
                        encode_failed = true;
                        break;
                    }
                    [encoder setBuffer:value->second offset:0 atIndex:index];
                }
                if (encode_failed) break;
                NSUInteger next_buffer = stage.inputs.size();
                for (size_t plane_index : stage.physical_plane_indices) {
                    if (plane_index >= impl_->planes.size()) {
                        encode_failed = true;
                        break;
                    }
                    [encoder setBuffer:impl_->planes[plane_index].buffer
                                offset:0
                               atIndex:next_buffer++];
                }
                if (encode_failed) break;
                if (!stage.physical_plane_indices.empty())
                    [encoder setBuffer:stage.plane_offsets
                                offset:0
                               atIndex:next_buffer++];
                [encoder setBuffer:stage_output offset:0
                           atIndex:next_buffer++];
                [encoder setBuffer:error_buffer offset:0
                           atIndex:next_buffer++];
                const NSUInteger width = stage.pipeline.threadExecutionWidth;
                const NSUInteger maximum =
                    stage.pipeline.maxTotalThreadsPerThreadgroup;
                const NSUInteger threads = std::min<NSUInteger>(maximum, 256);
                if (width == 0 || maximum < width || threads < width) {
                    encode_failed = true;
                    break;
                }
                minimum_width = std::min<uint32_t>(
                    minimum_width, static_cast<uint32_t>(width));
                minimum_maximum = std::min<uint32_t>(
                    minimum_maximum, static_cast<uint32_t>(maximum));
                [encoder dispatchThreads:MTLSizeMake(stage.output_count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
                value_buffers.emplace(stage.output_value_id, stage_output);
                if (stage_index + 1 != impl_->stages.size())
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [encoder endEncoding];
            if (encode_failed) {
                release_inputs();
                return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                                   "Metal staged binding is unavailable");
            }
            bool has_state_write = false;
            for (const Impl::State& state : impl_->states)
                has_state_write = has_state_write ||
                    state.write_value_id != UINT32_MAX;
            if (has_state_write) {
                id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                if (!blit) {
                    release_inputs();
                    return metal_error(
                        CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                        "Metal staged state copy encoder is unavailable");
                }
                for (size_t index = 0; index < impl_->states.size(); ++index) {
                    const Impl::State& state = impl_->states[index];
                    if (state.write_value_id == UINT32_MAX) continue;
                    if (state.direct_output)
                        continue;
                    const auto value = value_buffers.find(state.write_value_id);
                    if (value == value_buffers.end()) {
                        [blit endEncoding];
                        release_inputs();
                        return metal_error(
                            CompatibilityError::IR_REFERENCE_INVALID,
                            "Metal staged state value is unavailable");
                    }
                    [blit copyFromBuffer:value->second
                            sourceOffset:0
                                toBuffer:state.candidate
                       destinationOffset:0
                                    size:state.element_count *
                                         scalar_abi(state.type).bytes];
                }
                [blit endEncoding];
            }
            const auto cpu_wait_start = std::chrono::steady_clock::now();
            [command commit];
            [command waitUntilCompleted];
            const auto cpu_wait_end = std::chrono::steady_clock::now();
            if (command.status != MTLCommandBufferStatusCompleted ||
                command.error) {
                std::string detail = "Metal staged command failed";
                if (command.error.localizedDescription)
                    detail += ": " + std::string(
                        [command.error.localizedDescription UTF8String]);
                release_inputs();
                return metal_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE,
                                   std::move(detail));
            }
            const uint32_t error_value =
                *static_cast<const uint32_t*>(error_buffer.contents);
            if (error_value != 0) {
                release_inputs();
                return metal_error(
                    error_value == 2
                        ? CompatibilityError::RUNTIME_INPUT_INVALID
                        : CompatibilityError::RUNTIME_NUMERICAL_FAILURE,
                    error_value == 2
                        ? "Metal staged tensor index is out of bounds"
                        : "Metal staged numerical policy rejected a value");
            }
            if (has_state_write) {
                for (Impl::State& state : impl_->states) {
                    if (state.write_value_id != UINT32_MAX)
                        std::swap(state.current, state.candidate);
                }
                ++impl_->state_generation;
            }
            if (impl_->exports.size() != 1) {
                release_inputs();
                return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                                   "Metal staged export is unavailable");
            }
            const auto exported = value_buffers.find(impl_->exports.front());
            if (exported == value_buffers.end()) {
                release_inputs();
                return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "Metal staged export binding is unavailable");
            }
            MetalProgramResult result;
            MetalProgramValue value = unpack_value(impl_->output_type, impl_->output_extents,
                                                       impl_->output_count, exported->second.contents);
            result.exports.push_back(std::move(value));
            result.audit.program_digest = impl_->digest;
            result.audit.lowering_digest = impl_->lowering;
            result.audit.explicit_upload_bytes = upload_bytes;
            result.audit.zero_copy_plane_bytes = impl_->zero_copy_plane_bytes;
            result.audit.persistent_plane_bytes =
                impl_->persistent_plane_bytes;
            result.audit.explicit_download_bytes =
                impl_->output_count * scalar_abi(impl_->output_type).bytes;
            result.audit.command_buffers = 1;
            result.audit.implicit_weight_copies = 0;
            result.audit.mapped_artifact_buffer_count = impl_->mapped_artifacts.size();
            result.audit.thread_execution_width = minimum_width;
            result.audit.max_total_threads_per_threadgroup = minimum_maximum;
            result.audit.state_generation = impl_->state_generation;
            result.audit.intermediate_buffer_bytes = impl_->intermediate_buffer_bytes;
            result.audit.cpu_wait_ms = std::chrono::duration<double, std::milli>(
                cpu_wait_end - cpu_wait_start).count();
            if (command.GPUEndTime >= command.GPUStartTime)
                result.audit.gpu_time_ms =
                    (command.GPUEndTime - command.GPUStartTime) * 1000.0;
            release_inputs();
            return result;
        }
        const NSUInteger output_bytes = impl_->output_count * scalar_abi(impl_->output_type).bytes;
        id<MTLBuffer> output = impl_->output;
        id<MTLBuffer> numerical_error = impl_->numerical_error;
        *static_cast<uint32_t*>(numerical_error.contents) = 0;

        id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) {
            for (id<MTLBuffer> value : buffers) [value release];
            return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                               "Metal program command allocation failed");
        }
        [encoder setComputePipelineState:impl_->pipeline];
        for (NSUInteger index = 0; index < buffers.size(); ++index)
            [encoder setBuffer:buffers[index] offset:0 atIndex:index];
        NSUInteger next_buffer = buffers.size();
        for (const Impl::Plane& plane : impl_->planes)
            [encoder setBuffer:plane.buffer offset:0 atIndex:next_buffer++];
        if (!impl_->planes.empty())
            [encoder setBuffer:impl_->plane_offsets offset:0
                       atIndex:next_buffer++];
        [encoder setBuffer:output offset:0 atIndex:next_buffer++];
        [encoder setBuffer:numerical_error offset:0 atIndex:next_buffer++];
        const NSUInteger width = impl_->pipeline.threadExecutionWidth;
        const NSUInteger maximum = impl_->pipeline.maxTotalThreadsPerThreadgroup;
        const NSUInteger threads = std::min<NSUInteger>(maximum, 256);
        if (width == 0 || maximum < width || threads < width) {
            [encoder endEncoding];
            for (id<MTLBuffer> value : buffers) [value release];
            return metal_error(CompatibilityError::CAPABILITY_MISSING,
                               "Metal program pipeline limits are invalid");
        }
        [encoder dispatchThreads:MTLSizeMake(impl_->output_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        const auto cpu_wait_start = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        const auto cpu_wait_end = std::chrono::steady_clock::now();
        if (command.status != MTLCommandBufferStatusCompleted || command.error) {
            std::string detail = "Metal program command failed";
            if (command.error.localizedDescription)
                detail += ": " + std::string(
                    [command.error.localizedDescription UTF8String]);
            for (id<MTLBuffer> value : buffers) [value release];
            return metal_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE,
                               std::move(detail));
        }
        const uint32_t error_value =
            *static_cast<const uint32_t*>(numerical_error.contents);
        if (error_value != 0) {
            for (id<MTLBuffer> value : buffers) [value release];
            if (error_value == 2)
                return metal_error(CompatibilityError::RUNTIME_INPUT_INVALID,
                                   "Metal tensor index is out of bounds");
            return metal_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE,
                               "Metal bound physical numerical policy rejected a value");
        }

        MetalProgramResult result;
        MetalProgramValue value = unpack_value(impl_->output_type, impl_->output_extents,
                                                       impl_->output_count, output.contents);
        result.exports.push_back(std::move(value));
        result.audit.program_digest = impl_->digest;
        result.audit.lowering_digest = impl_->lowering;
        result.audit.explicit_upload_bytes = upload_bytes;
        result.audit.zero_copy_plane_bytes = impl_->zero_copy_plane_bytes;
        result.audit.persistent_plane_bytes = impl_->persistent_plane_bytes;
        result.audit.explicit_download_bytes = output_bytes;
        result.audit.intermediate_buffer_bytes = impl_->intermediate_buffer_bytes;
        result.audit.command_buffers = 1;
        result.audit.implicit_weight_copies = 0;
        result.audit.mapped_artifact_buffer_count = impl_->mapped_artifacts.size();
        result.audit.thread_execution_width = static_cast<uint32_t>(width);
        result.audit.max_total_threads_per_threadgroup =
            static_cast<uint32_t>(maximum);
        result.audit.cpu_wait_ms = std::chrono::duration<double, std::milli>(
            cpu_wait_end - cpu_wait_start).count();
        if (command.GPUEndTime >= command.GPUStartTime)
            result.audit.gpu_time_ms =
                (command.GPUEndTime - command.GPUStartTime) * 1000.0;
        for (id<MTLBuffer> buffer : buffers) [buffer release];
        return result;
    }
}

MetalProgramExecutionResult MetalProgramExecutable::execute_sequence(
    std::span<const MetalProgramInputStep> steps) const {
    if (!impl_ || steps.empty() || impl_->stages.empty())
        return metal_error(CompatibilityError::RUNTIME_INPUT_INVALID,
                           "Metal program sequence is unsupported");
    std::lock_guard lock(impl_->execution_mutex);
    @autoreleasepool {
        std::vector<std::vector<id<MTLBuffer>>> input_buffers;
        input_buffers.reserve(steps.size());
        uint64_t upload_bytes = impl_->persistent_upload_bytes;
        const auto release_buffers = [&] {
            for (const auto& step : input_buffers)
                for (id<MTLBuffer> buffer : step) [buffer release];
        };
        for (const MetalProgramInputStep& step : steps) {
            if (step.inputs.size() != impl_->inputs.size()) {
                release_buffers();
                return metal_error(
                    CompatibilityError::RUNTIME_INPUT_INVALID,
                    "Metal program sequence input count is invalid");
            }
            std::vector<const MetalProgramInput*> ordered(
                impl_->inputs.size(), nullptr);
            std::unordered_set<uint32_t> seen;
            for (const MetalProgramInput& input : step.inputs) {
                if (!seen.insert(input.value_id).second) {
                    release_buffers();
                    return metal_error(
                        CompatibilityError::RUNTIME_INPUT_INVALID,
                        "Metal program sequence input identity is duplicated");
                }
                const auto expected = std::find_if(
                    impl_->inputs.begin(), impl_->inputs.end(),
                    [&](const InputSpec& spec) {
                        return spec.value_id == input.value_id;
                    });
                if (expected == impl_->inputs.end() ||
                    !value_matches(input.value, *expected)) {
                    release_buffers();
                    return metal_error(
                        CompatibilityError::RUNTIME_INPUT_INVALID,
                        "Metal program sequence input type or shape is invalid");
                }
                ordered[static_cast<size_t>(
                    expected - impl_->inputs.begin())] = &input;
            }
            if (std::find(ordered.begin(), ordered.end(), nullptr) !=
                ordered.end()) {
                release_buffers();
                return metal_error(
                    CompatibilityError::RUNTIME_INPUT_INVALID,
                    "Metal program sequence input is missing");
            }
            std::vector<id<MTLBuffer>> buffers;
            buffers.reserve(ordered.size());
            for (const MetalProgramInput* input : ordered) {
                const auto bits = pack_value(input->value);
                const NSUInteger bytes = bits.size();
                id<MTLBuffer> buffer = [impl_->device
                    newBufferWithBytes:bits.data()
                               length:bytes
                              options:MTLResourceStorageModeShared];
                if (!buffer) {
                    for (id<MTLBuffer> value : buffers) [value release];
                    release_buffers();
                    return metal_error(
                        CompatibilityError::PLAN_MEMORY_EXCEEDED,
                        "Metal program sequence input allocation failed");
                }
                buffers.push_back(buffer);
                upload_bytes += bytes;
            }
            input_buffers.push_back(std::move(buffers));
        }

        id<MTLBuffer> error_buffer = impl_->numerical_error;
        *static_cast<uint32_t*>(error_buffer.contents) = 0;
        id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
        if (!command) {
            release_buffers();
            return metal_error(
                CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                "Metal program sequence command allocation failed");
        }

        std::vector<id<MTLBuffer>> state_sources;
        state_sources.reserve(impl_->states.size());
        for (const Impl::State& state : impl_->states)
            state_sources.push_back(state.current);
        id<MTLBuffer> final_output = nil;
        uint32_t minimum_width = UINT32_MAX;
        uint32_t minimum_maximum = UINT32_MAX;
        bool any_state_write = false;
        for (const Impl::State& state : impl_->states)
            any_state_write = any_state_write ||
                state.write_value_id != UINT32_MAX;

        for (size_t step_index = 0; step_index < steps.size(); ++step_index) {
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            if (!encoder) {
                release_buffers();
                return metal_error(
                    CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                    "Metal program sequence encoder is unavailable");
            }
            std::unordered_map<uint32_t, id<MTLBuffer>> value_buffers;
            for (size_t index = 0; index < impl_->inputs.size(); ++index)
                value_buffers.emplace(impl_->inputs[index].value_id,
                                      input_buffers[step_index][index]);
            for (size_t index = 0; index < impl_->states.size(); ++index)
                for (uint32_t value_id : impl_->states[index].read_value_ids)
                    value_buffers.emplace(value_id, state_sources[index]);
            bool encode_failed = false;
            for (size_t stage_index = 0;
                 stage_index < impl_->stages.size(); ++stage_index) {
                const Impl::Stage& stage = impl_->stages[stage_index];
                id<MTLBuffer> stage_output = stage.output;
                if (stage.direct_state_index != UINT32_MAX) {
                    if (stage.direct_state_index >= impl_->states.size()) {
                        encode_failed = true;
                        break;
                    }
                    const Impl::State& state =
                        impl_->states[stage.direct_state_index];
                    stage_output = step_index % 2 == 0
                        ? state.candidate : state.scratch;
                }
                [encoder setComputePipelineState:stage.pipeline];
                for (NSUInteger index = 0; index < stage.inputs.size(); ++index) {
                    const auto value =
                        value_buffers.find(stage.inputs[index].value_id);
                    if (value == value_buffers.end()) {
                        encode_failed = true;
                        break;
                    }
                    [encoder setBuffer:value->second offset:0 atIndex:index];
                }
                if (encode_failed) break;
                NSUInteger next_buffer = stage.inputs.size();
                for (size_t plane_index : stage.physical_plane_indices) {
                    if (plane_index >= impl_->planes.size()) {
                        encode_failed = true;
                        break;
                    }
                    [encoder setBuffer:impl_->planes[plane_index].buffer
                                offset:0 atIndex:next_buffer++];
                }
                if (encode_failed) break;
                if (!stage.physical_plane_indices.empty())
                    [encoder setBuffer:stage.plane_offsets offset:0
                               atIndex:next_buffer++];
                [encoder setBuffer:stage_output offset:0 atIndex:next_buffer++];
                [encoder setBuffer:error_buffer offset:0 atIndex:next_buffer++];
                const NSUInteger width = stage.pipeline.threadExecutionWidth;
                const NSUInteger maximum =
                    stage.pipeline.maxTotalThreadsPerThreadgroup;
                const NSUInteger threads = std::min<NSUInteger>(maximum, 256);
                if (width == 0 || maximum < width || threads < width) {
                    encode_failed = true;
                    break;
                }
                minimum_width = std::min<uint32_t>(
                    minimum_width, static_cast<uint32_t>(width));
                minimum_maximum = std::min<uint32_t>(
                    minimum_maximum, static_cast<uint32_t>(maximum));
                [encoder dispatchThreads:MTLSizeMake(stage.output_count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
                value_buffers.emplace(stage.output_value_id, stage_output);
                if (stage_index + 1 != impl_->stages.size())
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [encoder endEncoding];
            if (encode_failed) {
                release_buffers();
                return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                                   "Metal program sequence binding is unavailable");
            }
            if (impl_->exports.size() != 1 ||
                !value_buffers.contains(impl_->exports.front())) {
                release_buffers();
                return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "Metal program sequence export is unavailable");
            }
            if (step_index + 1 == steps.size())
                final_output = value_buffers[impl_->exports.front()];

            if (any_state_write) {
                id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                if (!blit) {
                    release_buffers();
                    return metal_error(
                        CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                        "Metal program sequence state encoder is unavailable");
                }
                for (size_t index = 0; index < impl_->states.size(); ++index) {
                    Impl::State& state = impl_->states[index];
                    if (state.write_value_id == UINT32_MAX) continue;
                    const auto value = value_buffers.find(state.write_value_id);
                    if (value == value_buffers.end()) {
                        [blit endEncoding];
                        release_buffers();
                        return metal_error(
                            CompatibilityError::IR_REFERENCE_INVALID,
                            "Metal program sequence state value is unavailable");
                    }
                    id<MTLBuffer> target =
                        step_index % 2 == 0 ? state.candidate : state.scratch;
                    if (state.direct_output) {
                        state_sources[index] = target;
                        continue;
                    }
                    [blit copyFromBuffer:value->second sourceOffset:0
                                toBuffer:target destinationOffset:0
                                    size:state.element_count * scalar_abi(state.type).bytes];
                    state_sources[index] = target;
                }
                [blit endEncoding];
            }
        }

        const auto cpu_wait_start = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        const auto cpu_wait_end = std::chrono::steady_clock::now();
        if (command.status != MTLCommandBufferStatusCompleted || command.error) {
            std::string detail = "Metal program sequence command failed";
            if (command.error.localizedDescription)
                detail += ": " + std::string(
                    [command.error.localizedDescription UTF8String]);
            release_buffers();
            return metal_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE,
                               std::move(detail));
        }
        const uint32_t error_value =
            *static_cast<const uint32_t*>(error_buffer.contents);
        if (error_value != 0) {
            release_buffers();
            return metal_error(
                error_value == 2
                    ? CompatibilityError::RUNTIME_INPUT_INVALID
                    : CompatibilityError::RUNTIME_NUMERICAL_FAILURE,
                error_value == 2
                    ? "Metal program sequence tensor index is out of bounds"
                    : "Metal program sequence numerical policy rejected a value");
        }
        if (any_state_write) {
            for (Impl::State& state : impl_->states) {
                if (state.write_value_id == UINT32_MAX) continue;
                if (steps.size() % 2 == 1)
                    std::swap(state.current, state.candidate);
                else
                    std::swap(state.current, state.scratch);
            }
            impl_->state_generation += steps.size();
        }
        MetalProgramResult result;
        MetalProgramValue value = unpack_value(impl_->output_type, impl_->output_extents,
                                                       impl_->output_count, final_output.contents);
        result.exports.push_back(std::move(value));
        result.audit.program_digest = impl_->digest;
        result.audit.lowering_digest = impl_->lowering;
        result.audit.explicit_upload_bytes = upload_bytes;
        result.audit.zero_copy_plane_bytes = impl_->zero_copy_plane_bytes;
        result.audit.persistent_plane_bytes = impl_->persistent_plane_bytes;
        result.audit.explicit_download_bytes =
            impl_->output_count * scalar_abi(impl_->output_type).bytes;
        result.audit.command_buffers = 1;
        result.audit.implicit_weight_copies = 0;
        result.audit.mapped_artifact_buffer_count = impl_->mapped_artifacts.size();
        result.audit.thread_execution_width = minimum_width;
        result.audit.max_total_threads_per_threadgroup = minimum_maximum;
        result.audit.state_generation = impl_->state_generation;
        result.audit.intermediate_buffer_bytes = impl_->intermediate_buffer_bytes;
        result.audit.cpu_wait_ms = std::chrono::duration<double, std::milli>(
            cpu_wait_end - cpu_wait_start).count();
        if (command.GPUEndTime >= command.GPUStartTime)
            result.audit.gpu_time_ms =
                (command.GPUEndTime - command.GPUStartTime) * 1000.0;
        release_buffers();
        return result;
    }
}

class MetalProgramCompiler {
public:
static MetalProgramCompileResult compile(
    Lowering lowering, ProgramDigest digest,
    std::shared_ptr<const VerifiedPhysicalProgramPackage> package_owner = {}) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
            return metal_error(CompatibilityError::CAPABILITY_MISSING,
                               "Metal program requires an Apple GPU");
        NSString* source = [[NSString alloc]
            initWithBytes:lowering.source.data()
                    length:lowering.source.size()
                  encoding:NSUTF8StringEncoding];
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions =
            MTLMathFloatingPointFunctionsPrecise;
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                     options:options
                                                       error:&error];
        [options release];
        [source release];
        if (!library) {
            std::string detail = "generated Metal program failed to compile";
            if (error.localizedDescription)
                detail += ": " + std::string([error.localizedDescription UTF8String]);
            [device release];
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               std::move(detail));
        }
        id<MTLFunction> function = [library
            newFunctionWithName:@"laplace_structured_v1"];
        if (!function) {
            [library release];
            [device release];
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "generated Metal function is unavailable");
        }
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        [function release];
        [library release];
        if (!pipeline) {
            std::string detail = "generated Metal pipeline failed to build";
            if (error.localizedDescription)
                detail += ": " + std::string([error.localizedDescription UTF8String]);
            [device release];
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               std::move(detail));
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            [pipeline release];
            [device release];
            return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                               "Metal program command queue is unavailable");
        }
        auto impl = std::make_unique<MetalProgramExecutable::Impl>();
        impl->device = device;
        impl->queue = queue;
        impl->pipeline = pipeline;
        impl->digest = digest;
        impl->lowering = source_digest(lowering.source);
        impl->inputs = std::move(lowering.inputs);
        impl->output_extents = std::move(lowering.output_extents);
        impl->output_count = lowering.output_count;
        impl->output_type = lowering.output_type;
        impl->package_owner = std::move(package_owner);

        if (!lowering.physical_planes.empty()) {
            if (!impl->package_owner)
                return metal_error(CompatibilityError::AUTHORITY_INVALID,
                                   "Metal bound physical package owner is unavailable");
            const long raw_page = sysconf(_SC_PAGESIZE);
            if (raw_page <= 0)
                return metal_error(CompatibilityError::CAPABILITY_MISSING,
                                   "host page size is unavailable");
            const size_t page = static_cast<size_t>(raw_page);
            std::vector<uint64_t> offsets(
                lowering.physical_planes.size(), 0);
            size_t offset_index = 0;
            impl->planes.reserve(lowering.physical_planes.size());
            for (const Lowering::PhysicalPlane& requested :
                 lowering.physical_planes) {
                const auto resource = std::lower_bound(
                    impl->package_owner->resources().begin(),
                    impl->package_owner->resources().end(),
                    requested.resource_id,
                    [](const PhysicalResourceBinding& binding, uint32_t id) {
                        return binding.resource_id < id;
                    });
                if (resource == impl->package_owner->resources().end() ||
                    resource->resource_id != requested.resource_id)
                    return metal_error(
                        CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                        "Metal bound physical resource is unavailable");
                const PhysicalProgram* program =
                    impl->package_owner->find_program(resource->program_digest);
                if (!program || requested.plane >= program->planes.size())
                    return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                       "Metal bound physical plane is invalid");
                const ::Laplace::PhysicalPlane& declaration =
                    program->planes[requested.plane];
                MetalProgramExecutable::Impl::Plane plane;
                if (declaration.storage == PhysicalPlaneStorage::Inline) {
                    plane.length = declaration.byte_length;
                    if (declaration.inline_offset > program->inline_bytes.size() ||
                        declaration.byte_length >
                            program->inline_bytes.size() -
                                declaration.inline_offset)
                        return metal_error(
                            CompatibilityError::PACKAGE_BOUNDS_INVALID,
                            "Metal bound inline plane is out of bounds");
                    const uint8_t* bytes = program->inline_bytes.data() +
                                           declaration.inline_offset;
                    plane.buffer = [device
                        newBufferWithBytes:bytes
                                   length:declaration.byte_length
                                  options:MTLResourceStorageModeShared];
                    impl->persistent_upload_bytes += declaration.byte_length;
                } else {
                    const auto bound = std::find_if(
                        resource->planes.begin(), resource->planes.end(),
                        [&](const PhysicalPlaneSource& candidate) {
                            return candidate.plane == requested.plane;
                        });
                    if (bound == resource->planes.end())
                        return metal_error(
                            CompatibilityError::IR_REFERENCE_INVALID,
                            "Metal bound external plane is invalid");
                    plane.length = bound->length;
                    const auto artifact = std::find_if(
                        impl->package_owner->physical_index().artifacts().begin(),
                        impl->package_owner->physical_index().artifacts().end(),
                        [&](const PackageView& candidate) {
                            return candidate.artifact_id() == bound->artifact_id;
                        });
                    if (artifact ==
                            impl->package_owner->physical_index().artifacts().end() ||
                        bound->offset > artifact->bytes().size() ||
                        bound->length >
                            artifact->bytes().size() - bound->offset)
                        return metal_error(
                            CompatibilityError::PACKAGE_BOUNDS_INVALID,
                            "Metal bound mapped plane is out of bounds");
                    const uintptr_t base = reinterpret_cast<uintptr_t>(
                        artifact->bytes().data());
                    if (base % page != 0)
                        return metal_error(
                            CompatibilityError::IR_CONSTRAINT_FAILED,
                            "Metal bound artifact mapping is not page aligned");
                    const size_t remainder = artifact->bytes().size() % page;
                    if (remainder != 0 &&
                        artifact->bytes().size() >
                            SIZE_MAX - (page - remainder))
                        return metal_error(
                            CompatibilityError::PACKAGE_BOUNDS_INVALID,
                            "Metal bound artifact mapping length overflows");
                    const size_t mapped_length = artifact->bytes().size() +
                        (remainder == 0 ? 0 : page - remainder);
                    plane.buffer = impl->mapped_buffer(*artifact, mapped_length);
                    plane.byte_offset = bound->offset;
                    plane.zero_copy = true;
                    impl->zero_copy_plane_bytes += bound->length;
                }
                if (!plane.buffer)
                    return metal_error(
                        CompatibilityError::PLAN_MEMORY_EXCEEDED,
                        "Metal bound physical plane allocation failed");
                offsets[offset_index++] = plane.byte_offset;
                impl->persistent_plane_bytes += plane.length;
                impl->planes.push_back(plane);
            }
            impl->plane_offsets = [device
                newBufferWithBytes:offsets.data()
                           length:offsets.size() * sizeof(uint64_t)
                          options:MTLResourceStorageModeShared];
            if (!impl->plane_offsets)
                return metal_error(
                    CompatibilityError::PLAN_MEMORY_EXCEEDED,
                    "Metal bound physical offset allocation failed");
        }
        if (!impl->allocate_workspace())
            return metal_error(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                               "Metal program workspace allocation failed");
        return MetalProgramExecutable(std::move(impl));
    }
}

static MetalProgramCompileResult compile(MultiLowering lowering,
                                         ProgramDigest digest,
    std::shared_ptr<const VerifiedPhysicalProgramPackage> package_owner = {}) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
            return metal_error(CompatibilityError::CAPABILITY_MISSING,
                               "Metal staged program requires an Apple GPU");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            [device release];
            return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                               "Metal staged command queue is unavailable");
        }
        auto impl = std::make_unique<MetalProgramExecutable::Impl>();
        impl->device = device;
        impl->queue = queue;
        impl->digest = digest;
        impl->package_owner = std::move(package_owner);
        impl->inputs = std::move(lowering.inputs);
        impl->exports = std::move(lowering.exports);
        impl->output_extents = std::move(lowering.output_extents);
        impl->output_count = lowering.output_count;
        impl->output_type = lowering.output_type;
        impl->stages.reserve(lowering.stages.size());
        const bool has_physical_planes = std::any_of(
            lowering.stages.begin(), lowering.stages.end(),
            [](const Lowering& stage) {
                return !stage.physical_planes.empty();
            });
        size_t page = 0;
        if (has_physical_planes) {
            if (!impl->package_owner)
                return metal_error(
                    CompatibilityError::AUTHORITY_INVALID,
                    "Metal staged physical package owner is unavailable");
            const long raw_page = sysconf(_SC_PAGESIZE);
            if (raw_page <= 0)
                return metal_error(CompatibilityError::CAPABILITY_MISSING,
                                   "host page size is unavailable");
            page = static_cast<size_t>(raw_page);
        }
        const auto bind_plane = [&](const Lowering::PhysicalPlane& requested)
            -> std::variant<size_t, CompatibilityReport> {
            const auto existing = std::find_if(
                impl->planes.begin(), impl->planes.end(),
                [&](const MetalProgramExecutable::Impl::Plane& plane) {
                    return plane.resource_id == requested.resource_id &&
                           plane.plane_index == requested.plane;
                });
            if (existing != impl->planes.end())
                return static_cast<size_t>(existing - impl->planes.begin());
            const auto resource = std::lower_bound(
                impl->package_owner->resources().begin(),
                impl->package_owner->resources().end(), requested.resource_id,
                [](const PhysicalResourceBinding& binding, uint32_t id) {
                    return binding.resource_id < id;
                });
            if (resource == impl->package_owner->resources().end() ||
                resource->resource_id != requested.resource_id)
                return metal_error(
                    CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                    "Metal staged physical resource is unavailable");
            const PhysicalProgram* program =
                impl->package_owner->find_program(resource->program_digest);
            if (!program || requested.plane >= program->planes.size())
                return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "Metal staged physical plane is invalid");
            const ::Laplace::PhysicalPlane& declaration =
                program->planes[requested.plane];
            MetalProgramExecutable::Impl::Plane plane_binding;
            plane_binding.resource_id = requested.resource_id;
            plane_binding.plane_index = requested.plane;
            if (declaration.storage == PhysicalPlaneStorage::Inline) {
                plane_binding.length = declaration.byte_length;
                if (declaration.inline_offset > program->inline_bytes.size() ||
                    declaration.byte_length >
                        program->inline_bytes.size() - declaration.inline_offset)
                    return metal_error(
                        CompatibilityError::PACKAGE_BOUNDS_INVALID,
                        "Metal staged inline plane is out of bounds");
                plane_binding.buffer = [device
                    newBufferWithBytes:program->inline_bytes.data() +
                                       declaration.inline_offset
                               length:declaration.byte_length
                              options:MTLResourceStorageModeShared];
                impl->persistent_upload_bytes += declaration.byte_length;
            } else {
                const auto bound = std::find_if(
                    resource->planes.begin(), resource->planes.end(),
                    [&](const PhysicalPlaneSource& candidate) {
                        return candidate.plane == requested.plane;
                    });
                if (bound == resource->planes.end())
                    return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                       "Metal staged external plane is invalid");
                plane_binding.length = bound->length;
                const auto artifact = std::find_if(
                    impl->package_owner->physical_index().artifacts().begin(),
                    impl->package_owner->physical_index().artifacts().end(),
                    [&](const PackageView& candidate) {
                        return candidate.artifact_id() == bound->artifact_id;
                    });
                if (artifact ==
                        impl->package_owner->physical_index().artifacts().end() ||
                    bound->offset > artifact->bytes().size() ||
                    bound->length > artifact->bytes().size() - bound->offset)
                    return metal_error(
                        CompatibilityError::PACKAGE_BOUNDS_INVALID,
                        "Metal staged mapped plane is out of bounds");
                const uintptr_t base = reinterpret_cast<uintptr_t>(
                    artifact->bytes().data());
                if (base % page != 0)
                    return metal_error(
                        CompatibilityError::IR_CONSTRAINT_FAILED,
                        "Metal staged artifact mapping is not page aligned");
                const size_t remainder = artifact->bytes().size() % page;
                if (remainder != 0 && artifact->bytes().size() >
                        SIZE_MAX - (page - remainder))
                    return metal_error(
                        CompatibilityError::PACKAGE_BOUNDS_INVALID,
                        "Metal staged artifact mapping length overflows");
                const size_t mapped_length = artifact->bytes().size() +
                    (remainder == 0 ? 0 : page - remainder);
                plane_binding.buffer = impl->mapped_buffer(*artifact, mapped_length);
                plane_binding.byte_offset = bound->offset;
                plane_binding.zero_copy = true;
                impl->zero_copy_plane_bytes += bound->length;
            }
            if (!plane_binding.buffer)
                return metal_error(
                    CompatibilityError::PLAN_MEMORY_EXCEEDED,
                    "Metal staged physical plane allocation failed");
            impl->persistent_plane_bytes += plane_binding.length;
            impl->planes.push_back(plane_binding);
            return impl->planes.size() - 1;
        };
        std::string aggregate_source;
        for (size_t index = 0; index < lowering.stages.size(); ++index) {
            Lowering& stage = lowering.stages[index];
            aggregate_source.append("stage:");
            aggregate_source.append(std::to_string(index));
            aggregate_source.push_back('\n');
            aggregate_source.append(stage.source);
            NSString* source = [[NSString alloc]
                initWithBytes:stage.source.data()
                        length:stage.source.size()
                      encoding:NSUTF8StringEncoding];
            NSError* error = nil;
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions =
                MTLMathFloatingPointFunctionsPrecise;
            id<MTLLibrary> library = [device newLibraryWithSource:source
                                                         options:options
                                                           error:&error];
            [options release];
            [source release];
            if (!library) {
                std::string detail = "generated Metal stage failed to compile";
                if (error.localizedDescription)
                    detail += ": " + std::string(
                        [error.localizedDescription UTF8String]);
                return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                                   std::move(detail));
            }
            id<MTLFunction> function = [library
                newFunctionWithName:@"laplace_structured_v1"];
            if (!function) {
                [library release];
                return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                                   "generated Metal stage function is unavailable");
            }
            id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:function
                                                       error:&error];
            [function release];
            [library release];
            if (!pipeline) {
                std::string detail = "generated Metal stage pipeline failed";
                if (error.localizedDescription)
                    detail += ": " + std::string(
                        [error.localizedDescription UTF8String]);
                return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                                   std::move(detail));
            }
            MetalProgramExecutable::Impl::Stage compiled;
            compiled.pipeline = pipeline;
            compiled.inputs = std::move(stage.inputs);
            compiled.output_extents = std::move(stage.output_extents);
            compiled.output_count = stage.output_count;
            compiled.output_type = stage.output_type;
            compiled.output_value_id = stage.output_value_id;
            compiled.direct_state_index = stage.direct_state_index;
            std::vector<uint64_t> offsets(stage.physical_planes.size(), 0);
            size_t offset_index = 0;
            compiled.physical_plane_indices.reserve(
                stage.physical_planes.size());
            for (const Lowering::PhysicalPlane& plane :
                 stage.physical_planes) {
                auto bound = bind_plane(plane);
                if (const auto* report =
                        std::get_if<CompatibilityReport>(&bound))
                    return *report;
                const size_t plane_index = std::get<size_t>(bound);
                compiled.physical_plane_indices.push_back(plane_index);
                offsets[offset_index++] =
                    impl->planes[plane_index].byte_offset;
            }
            if (!offsets.empty()) {
                compiled.plane_offsets = [device
                    newBufferWithBytes:offsets.data()
                               length:offsets.size() * sizeof(uint64_t)
                              options:MTLResourceStorageModeShared];
                if (!compiled.plane_offsets)
                    return metal_error(
                        CompatibilityError::PLAN_MEMORY_EXCEEDED,
                        "Metal staged physical offset allocation failed");
            }
            impl->stages.push_back(std::move(compiled));
        }
        impl->states.reserve(lowering.states.size());
        for (MultiLowering::State& state : lowering.states) {
            MetalProgramExecutable::Impl::State compiled;
            compiled.state_id = state.state_id;
            compiled.type = state.type;
            compiled.extents = std::move(state.extents);
            compiled.element_count = state.element_count;
            compiled.read_value_ids = std::move(state.read_value_ids);
            compiled.write_value_id = state.write_value_id;
            compiled.direct_output = std::any_of(
                impl->stages.begin(), impl->stages.end(),
                [&](const MetalProgramExecutable::Impl::Stage& stage) {
                    return stage.direct_state_index == impl->states.size();
                });
            const NSUInteger bytes =
                compiled.element_count * scalar_abi(compiled.type).bytes;
            compiled.current = [device
                newBufferWithLength:bytes
                           options:MTLResourceStorageModeShared];
            compiled.candidate = [device
                newBufferWithLength:bytes
                           options:MTLResourceStorageModeShared];
            compiled.scratch = [device
                newBufferWithLength:bytes
                           options:MTLResourceStorageModeShared];
            if (!compiled.current || !compiled.candidate ||
                !compiled.scratch) {
                [compiled.current release];
                [compiled.candidate release];
                [compiled.scratch release];
                return metal_error(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                                   "Metal staged state allocation failed");
            }
            std::memset(compiled.current.contents, 0, bytes);
            std::memset(compiled.candidate.contents, 0, bytes);
            std::memset(compiled.scratch.contents, 0, bytes);
            impl->states.push_back(std::move(compiled));
        }
        impl->lowering = source_digest(aggregate_source);
        if (!impl->allocate_workspace())
            return metal_error(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                               "Metal program workspace allocation failed");
        return MetalProgramExecutable(std::move(impl));
    }
}
};

MetalProgramCompileResult compile_metal_program(
    const VerifiedProgram& program) {
    const Program& definition = program_definition(program);
    if (definition.functions.size() == 1) {
        const Function& function = definition.functions.front();
        if (const Region* root = find_region(function, function.entry_region_id)) {
            const size_t stages = static_cast<size_t>(std::count_if(
                root->instructions.begin(), root->instructions.end(),
                [](const Instruction& instruction) {
                    return instruction.primitive.code ==
                           Primitive::StructuredTensor;
                }));
            if (stages > 1) {
                auto lowered = lower_multi_program(program);
                if (const auto* report =
                        std::get_if<CompatibilityReport>(&lowered))
                    return *report;
                return MetalProgramCompiler::compile(
                    std::get<MultiLowering>(std::move(lowered)),
                    program_digest(program));
            }
        }
    }
    const auto lowered = lower_program(program);
    if (const auto* report = std::get_if<CompatibilityReport>(&lowered))
        return *report;
    return MetalProgramCompiler::compile(
        std::get<Lowering>(lowered), program_digest(program));
}

MetalProgramCompileResult compile_metal_program(
    const VerifiedPhysicalProgramPackage& package) {
    if (!package.semantic_program())
        return metal_error(CompatibilityError::AUTHORITY_INVALID,
                           "Metal program requires verified semantic authority");
    auto owner = std::make_shared<const VerifiedPhysicalProgramPackage>(package);
    const Program& definition =
        program_definition(*owner->semantic_program());
    if (definition.functions.size() == 1) {
        const Function& function = definition.functions.front();
        if (const Region* root =
                find_region(function, function.entry_region_id)) {
            const size_t stages = static_cast<size_t>(std::count_if(
                root->instructions.begin(), root->instructions.end(),
                [](const Instruction& instruction) {
                    return instruction.primitive.code ==
                           Primitive::StructuredTensor;
                }));
            if (stages > 1) {
                auto lowered = lower_multi_program(
                    *owner->semantic_program(), owner.get());
                if (const auto* report =
                        std::get_if<CompatibilityReport>(&lowered))
                    return *report;
                return MetalProgramCompiler::compile(
                    std::get<MultiLowering>(std::move(lowered)),
                    program_digest(*owner->semantic_program()),
                    std::move(owner));
            }
        }
    }
    const auto lowered = lower_program(*owner->semantic_program(), owner.get());
    if (const auto* report = std::get_if<CompatibilityReport>(&lowered))
        return *report;
    return MetalProgramCompiler::compile(
        std::get<Lowering>(lowered),
        program_digest(*owner->semantic_program()), std::move(owner));
}

MetalProgramCompileResult compile_metal_program(
    const VerifiedProgramPackage& package) {
    if (!package.complete() ||
        program_digest(package.semantic_program()) !=
            state_program_digest(package.state_schema()))
        return metal_error(CompatibilityError::AUTHORITY_INVALID,
                           "Metal program package authority is invalid");
    return compile_metal_program(package.physical_package());
}

struct MetalPhysicalProgramExecutable::Impl {
    struct Plane {
        id<MTLBuffer> buffer = nil;
        uint64_t byte_offset = 0;
        uint64_t length = 0;
        bool zero_copy = false;
    };
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
    PhysicalProgramDigest digest{};
    std::array<uint8_t, 32> lowering{};
    LogicalTensorType logical;
    std::vector<Plane> planes;
    std::shared_ptr<const VerifiedPhysicalProgramPackage> package_owner;
    uint64_t explicit_upload_bytes = 0;
    uint64_t zero_copy_plane_bytes = 0;
    uint64_t persistent_plane_bytes = 0;
    size_t output_count = 0;

    ~Impl() {
        for (const Plane& plane : planes) [plane.buffer release];
        [pipeline release];
        [queue release];
        [device release];
    }
};

MetalPhysicalProgramExecutable::MetalPhysicalProgramExecutable(
    std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MetalPhysicalProgramExecutable::~MetalPhysicalProgramExecutable() = default;
MetalPhysicalProgramExecutable::MetalPhysicalProgramExecutable(
    MetalPhysicalProgramExecutable&&) noexcept = default;
MetalPhysicalProgramExecutable& MetalPhysicalProgramExecutable::operator=(
    MetalPhysicalProgramExecutable&&) noexcept = default;

PhysicalProgramDigest
MetalPhysicalProgramExecutable::physical_program_digest() const noexcept {
    return impl_ ? impl_->digest : PhysicalProgramDigest{};
}

std::array<uint8_t, 32>
MetalPhysicalProgramExecutable::lowering_digest() const noexcept {
    return impl_ ? impl_->lowering : std::array<uint8_t, 32>{};
}

MetalPhysicalProgramExecutionResult
MetalPhysicalProgramExecutable::execute() const {
    if (!impl_)
        return metal_error(CompatibilityError::RUNTIME_INPUT_INVALID,
                           "Metal physical program is unavailable");
    @autoreleasepool {
        std::vector<uint64_t> plane_offsets;
        plane_offsets.reserve(impl_->planes.size());
        for (const Impl::Plane& plane : impl_->planes)
            plane_offsets.push_back(plane.byte_offset);
        id<MTLBuffer> offsets = [impl_->device
            newBufferWithBytes:plane_offsets.data()
                       length:plane_offsets.size() * sizeof(uint64_t)
                      options:MTLResourceStorageModeShared];
        const NSUInteger output_bytes = impl_->output_count * sizeof(uint32_t);
        id<MTLBuffer> output = [impl_->device
            newBufferWithLength:output_bytes
                       options:MTLResourceStorageModeShared];
        uint32_t no_error = 0;
        id<MTLBuffer> error = [impl_->device
            newBufferWithBytes:&no_error
                       length:sizeof(no_error)
                      options:MTLResourceStorageModeShared];
        if (!offsets || !output || !error) {
            [offsets release];
            [output release];
            [error release];
            return metal_error(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                               "Metal physical output allocation failed");
        }

        id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) {
            [encoder endEncoding];
            [offsets release];
            [output release];
            [error release];
            return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                               "Metal physical command allocation failed");
        }
        [encoder setComputePipelineState:impl_->pipeline];
        for (NSUInteger index = 0; index < impl_->planes.size(); ++index)
            [encoder setBuffer:impl_->planes[index].buffer offset:0 atIndex:index];
        [encoder setBuffer:offsets offset:0 atIndex:impl_->planes.size()];
        [encoder setBuffer:output offset:0 atIndex:impl_->planes.size() + 1];
        [encoder setBuffer:error offset:0 atIndex:impl_->planes.size() + 2];
        const NSUInteger width = impl_->pipeline.threadExecutionWidth;
        const NSUInteger maximum = impl_->pipeline.maxTotalThreadsPerThreadgroup;
        const NSUInteger threads = std::min<NSUInteger>(maximum, 256);
        if (width == 0 || maximum < width || threads < width) {
            [encoder endEncoding];
            [offsets release];
            [output release];
            [error release];
            return metal_error(CompatibilityError::CAPABILITY_MISSING,
                               "Metal physical pipeline limits are invalid");
        }
        [encoder dispatchThreads:MTLSizeMake(impl_->output_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted || command.error) {
            std::string detail = "Metal physical command failed";
            if (command.error.localizedDescription)
                detail += ": " + std::string(
                    [command.error.localizedDescription UTF8String]);
            [offsets release];
            [output release];
            [error release];
            return metal_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE,
                               std::move(detail));
        }
        if (*static_cast<const uint32_t*>(error.contents) != 0) {
            [offsets release];
            [output release];
            [error release];
            return metal_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE,
                               "Metal physical numerical policy rejected a value");
        }

        MetalPhysicalProgramResult result;
        result.value.type = impl_->logical.element_type;
        result.value.extents = impl_->logical.extents;
        const auto* words = static_cast<const uint32_t*>(output.contents);
        result.value.bits.assign(words, words + impl_->output_count);
        result.audit.physical_program_digest = impl_->digest;
        result.audit.lowering_digest = impl_->lowering;
        result.audit.explicit_upload_bytes = impl_->explicit_upload_bytes;
        result.audit.zero_copy_plane_bytes = impl_->zero_copy_plane_bytes;
        result.audit.persistent_plane_bytes = impl_->persistent_plane_bytes;
        result.audit.explicit_download_bytes = output_bytes;
        result.audit.command_buffers = 1;
        result.audit.implicit_weight_copies = 0;
        result.audit.thread_execution_width = static_cast<uint32_t>(width);
        result.audit.max_total_threads_per_threadgroup =
            static_cast<uint32_t>(maximum);
        [offsets release];
        [output release];
        [error release];
        return result;
    }
}

class MetalPhysicalCompiler {
public:
static MetalPhysicalProgramCompileResult compile(
    PhysicalLowering lowering, PhysicalProgramDigest digest,
    std::span<const MetalPhysicalPlaneSource> sources,
    std::shared_ptr<const VerifiedPhysicalProgramPackage> package_owner = {}) {
    if (sources.size() != lowering.plane_count)
        return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                           "Metal physical plane count does not match the program");
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
            return metal_error(CompatibilityError::CAPABILITY_MISSING,
                               "Metal physical program requires an Apple GPU");
        NSString* source = [[NSString alloc]
            initWithBytes:lowering.source.data()
                    length:lowering.source.size()
                  encoding:NSUTF8StringEncoding];
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                     options:options
                                                       error:&error];
        [options release];
        [source release];
        if (!library) {
            std::string detail = "generated Metal physical program failed to compile";
            if (error.localizedDescription)
                detail += ": " + std::string(
                    [error.localizedDescription UTF8String]);
            [device release];
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               std::move(detail));
        }
        id<MTLFunction> function =
            [library newFunctionWithName:@"laplace_physical_v1"];
        if (!function) {
            [library release];
            [device release];
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "generated Metal physical function is unavailable");
        }
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        [function release];
        [library release];
        if (!pipeline) {
            std::string detail = "generated Metal physical pipeline failed to build";
            if (error.localizedDescription)
                detail += ": " + std::string(
                    [error.localizedDescription UTF8String]);
            [device release];
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               std::move(detail));
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            [pipeline release];
            [device release];
            return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                               "Metal physical command queue is unavailable");
        }
        auto impl = std::make_unique<MetalPhysicalProgramExecutable::Impl>();
        impl->device = device;
        impl->queue = queue;
        impl->pipeline = pipeline;
        impl->digest = digest;
        impl->lowering = source_digest(lowering.source);
        impl->logical = std::move(lowering.logical);
        impl->output_count = lowering.output_count;
        impl->package_owner = std::move(package_owner);
        impl->planes.reserve(sources.size());
        for (const MetalPhysicalPlaneSource& source : sources) {
            if (!source.bytes || source.length == 0) {
                return metal_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                   "Metal physical plane source is unavailable");
            }
            MetalPhysicalProgramExecutable::Impl::Plane plane;
            plane.length = source.length;
            if (source.mapping_base) {
                if (source.mapping_length == 0 ||
                    source.mapping_offset > source.mapping_length ||
                    source.length > source.mapping_length - source.mapping_offset) {
                    return metal_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                       "Metal physical mapped plane is out of bounds");
                }
                plane.buffer = [device
                    newBufferWithBytesNoCopy:const_cast<uint8_t*>(source.mapping_base)
                                     length:source.mapping_length
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
                plane.byte_offset = source.mapping_offset;
                plane.zero_copy = true;
                impl->zero_copy_plane_bytes += source.length;
            } else {
                plane.buffer = [device newBufferWithBytes:source.bytes
                                                   length:source.length
                                                  options:MTLResourceStorageModeShared];
                impl->explicit_upload_bytes += source.length;
            }
            if (!plane.buffer)
                return metal_error(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                                   "Metal physical persistent plane binding failed");
            impl->persistent_plane_bytes += source.length;
            impl->planes.push_back(plane);
        }
        return MetalPhysicalProgramExecutable(std::move(impl));
    }
}
};

MetalPhysicalProgramCompileResult compile_metal_physical_program(
    const VerifiedPhysicalProgram& program) {
    const auto lowered = lower_physical_program(program.program(),
                                                program.logical_type());
    if (const auto* report = std::get_if<CompatibilityReport>(&lowered))
        return *report;
    std::vector<MetalPhysicalPlaneSource> sources;
    sources.reserve(program.program().planes.size());
    for (uint16_t plane = 0; plane < program.program().planes.size(); ++plane) {
        const std::span<const uint8_t> bytes =
            physical_program_plane_bytes(program, plane);
        if (bytes.empty())
            return metal_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                               "Metal physical plane bytes are unavailable");
        sources.push_back({bytes.data(), bytes.size(), nullptr, 0, 0});
    }
    return MetalPhysicalCompiler::compile(
        std::get<PhysicalLowering>(lowered), program.digest(), sources);
}

MetalPhysicalProgramCompileResult compile_metal_physical_resource(
    const RuntimePackage& runtime, uint32_t resource_id) {
    const auto package = runtime.physical_program_package();
    if (!package)
        return metal_error(CompatibilityError::AUTHORITY_INVALID,
                           "runtime package has no physical program authority");
    const auto authority =
        validate_physical_program_package_for_runtime(runtime, *package);
    if (const auto* report = std::get_if<CompatibilityReport>(&authority))
        return *report;
    const auto resource = std::lower_bound(
        package->resources().begin(), package->resources().end(), resource_id,
        [](const PhysicalResourceBinding& binding, uint32_t id) {
            return binding.resource_id < id;
        });
    if (resource == package->resources().end() ||
        resource->resource_id != resource_id)
        return metal_error(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                           "physical resource occurrence is unavailable");
    size_t program_index = 0;
    for (; program_index < package->programs().size(); ++program_index) {
        const auto digest = physical_program_digest(
            package->programs()[program_index]);
        if (std::holds_alternative<PhysicalProgramDigest>(digest) &&
            std::get<PhysicalProgramDigest>(digest) == resource->program_digest)
            break;
    }
    if (program_index >= package->programs().size() ||
        program_index >= package->program_logical_types().size())
        return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                           "physical resource program is unavailable");
    const PhysicalProgram& program = package->programs()[program_index];
    const LogicalTensorType& logical =
        package->program_logical_types()[program_index];
    const auto lowered = lower_physical_program(program, logical);
    if (const auto* report = std::get_if<CompatibilityReport>(&lowered))
        return *report;

    const long raw_page = sysconf(_SC_PAGESIZE);
    if (raw_page <= 0)
        return metal_error(CompatibilityError::CAPABILITY_MISSING,
                           "host page size is unavailable");
    const size_t page = static_cast<size_t>(raw_page);
    std::vector<MetalPhysicalPlaneSource> sources(program.planes.size());
    for (uint16_t plane = 0; plane < program.planes.size(); ++plane) {
        const PhysicalPlane& declaration = program.planes[plane];
        if (declaration.storage == PhysicalPlaneStorage::Inline) {
            if (declaration.inline_offset > program.inline_bytes.size() ||
                declaration.byte_length >
                    program.inline_bytes.size() - declaration.inline_offset)
                return metal_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                   "physical inline plane is out of bounds");
            const uint8_t* bytes = program.inline_bytes.data() +
                                   declaration.inline_offset;
            sources[plane] = {bytes,
                              static_cast<size_t>(declaration.byte_length),
                              nullptr, 0, 0};
            continue;
        }
        const auto bound = std::find_if(
            resource->planes.begin(), resource->planes.end(),
            [plane](const ::Laplace::PhysicalPlaneSource& candidate) {
                return candidate.plane == plane;
            });
        if (bound == resource->planes.end())
            return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                               "physical external plane is unbound");
        const auto artifact = std::find_if(
            package->physical_index().artifacts().begin(),
            package->physical_index().artifacts().end(),
            [&](const PackageView& candidate) {
                return candidate.artifact_id() == bound->artifact_id;
            });
        if (artifact == package->physical_index().artifacts().end() ||
            bound->offset > artifact->bytes().size() ||
            bound->length > artifact->bytes().size() - bound->offset)
            return metal_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                               "physical mapped plane is out of bounds");
        const uintptr_t base =
            reinterpret_cast<uintptr_t>(artifact->bytes().data());
        if (base % page != 0)
            return metal_error(CompatibilityError::IR_CONSTRAINT_FAILED,
                               "physical artifact mapping is not page aligned");
        const size_t remainder = artifact->bytes().size() % page;
        if (remainder != 0 &&
            artifact->bytes().size() > SIZE_MAX - (page - remainder))
            return metal_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                               "physical artifact mapping length overflows");
        const size_t mapped_length = artifact->bytes().size() +
            (remainder == 0 ? 0 : page - remainder);
        sources[plane] = {
            artifact->bytes().data() + static_cast<size_t>(bound->offset),
            static_cast<size_t>(bound->length), artifact->bytes().data(),
            mapped_length, bound->offset};
    }
    return MetalPhysicalCompiler::compile(
        std::get<PhysicalLowering>(lowered), resource->program_digest,
        sources, package);
}

} // namespace Laplace
