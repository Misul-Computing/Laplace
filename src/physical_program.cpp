#include "physical_program.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <utility>

namespace Laplace {
namespace {

// A certificate has at most 64 maps, 128 nodes, rank 8 and 8 planes.
// Its generic lowering uses <=48 instructions/map and <=64 shared helpers:
// 64*48 + 128 + 64 = 3264 nodes. The longest lowered load path is <=48
// nodes, preceded by <=20 coordinate/address nodes and followed by <=31
// source operations. Its 4096 access records occupy 32768 inline bytes.
// 4096 instruction records (36 bytes each), that table, and declarations
// fit in 256 KiB. These rounded language limits also bound verifier work;
// constant interning remains at most O(nodes^2), all other passes O(nodes).
constexpr size_t kMaximumPhysicalNodes = 4096;
constexpr size_t kMaximumPhysicalPlanes = 32;
constexpr size_t kMaximumPhysicalPolicies = 32;
constexpr size_t kMaximumPhysicalDepth = 128;
constexpr size_t kMaximumPhysicalWireBytes = 256 * 1024;

CompatibilityReport error_report(CompatibilityError code,
                                  const char* detail) {
    return compatibility_report(code, detail);
}

void append_u8(std::vector<uint8_t>& bytes, uint8_t value) {
    bytes.push_back(value);
}

void append_u16(std::vector<uint8_t>& bytes, uint16_t value) {
    append_u8(bytes, static_cast<uint8_t>(value));
    append_u8(bytes, static_cast<uint8_t>(value >> 8u));
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8)
        append_u8(bytes, static_cast<uint8_t>(value >> shift));
}

void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8)
        append_u8(bytes, static_cast<uint8_t>(value >> shift));
}

struct PhysicalOpcodeShape {
    uint8_t operands = 0;
    bool plane = false;
    bool policy = false;
};

bool physical_opcode_shape(PhysicalOpcode opcode,
                           PhysicalOpcodeShape* out) noexcept {
    if (!out) return false;
    switch (opcode) {
        case PhysicalOpcode::ConstIndex:
        case PhysicalOpcode::ConstU32:
        case PhysicalOpcode::ConstI32:
        case PhysicalOpcode::ConstF32Bits:
        case PhysicalOpcode::Coordinate:
        case PhysicalOpcode::Extent:
            *out = {};
            return true;
        case PhysicalOpcode::IndexAdd:
        case PhysicalOpcode::IndexSubtract:
        case PhysicalOpcode::IndexMultiply:
        case PhysicalOpcode::IndexLess:
        case PhysicalOpcode::U32And:
        case PhysicalOpcode::U32Or:
        case PhysicalOpcode::U32Xor:
        case PhysicalOpcode::U32Add:
        case PhysicalOpcode::U32Multiply:
            *out = {2, false, false};
            return true;
        case PhysicalOpcode::IndexDivideConstant:
        case PhysicalOpcode::IndexRemainderConstant:
        case PhysicalOpcode::IndexCeilDivideConstant:
        case PhysicalOpcode::IndexFromU32:
        case PhysicalOpcode::U32ShiftLeftConstant:
        case PhysicalOpcode::U32ShiftRightConstant:
        case PhysicalOpcode::SignExtend:
            *out = {1, false, false};
            return true;
        case PhysicalOpcode::Select:
        case PhysicalOpcode::U32FunnelShiftRight:
            *out = {3, false, false};
            return true;
        case PhysicalOpcode::LoadBits:
            *out = {1, true, false};
            return true;
        case PhysicalOpcode::F16ToF32:
        case PhysicalOpcode::Bf16ToF32:
        case PhysicalOpcode::BitsToF32:
        case PhysicalOpcode::U32ToF32:
        case PhysicalOpcode::I32ToF32:
        case PhysicalOpcode::F32ToU32:
        case PhysicalOpcode::F32ToI32:
        case PhysicalOpcode::F32Negate:
        case PhysicalOpcode::F32RoundToF16:
            *out = {1, false, true};
            return true;
        case PhysicalOpcode::F32Add:
        case PhysicalOpcode::F32Subtract:
        case PhysicalOpcode::F32Multiply:
            *out = {2, false, true};
            return true;
        case PhysicalOpcode::F32Fma:
        case PhysicalOpcode::F32Clamp:
            *out = {3, false, true};
            return true;
    }
    return false;
}

bool valid_physical_value_type(PhysicalValueType type) noexcept {
    switch (type) {
        case PhysicalValueType::Predicate:
        case PhysicalValueType::Index:
        case PhysicalValueType::U32:
        case PhysicalValueType::I32:
        case PhysicalValueType::F32:
            return true;
    }
    return false;
}

bool valid_numeric_policy(const PhysicalNumericPolicy& policy) noexcept {
    const bool rounding =
        policy.rounding == PhysicalRoundingMode::NearestEven ||
        policy.rounding == PhysicalRoundingMode::TowardZero ||
        policy.rounding == PhysicalRoundingMode::TowardPositive ||
        policy.rounding == PhysicalRoundingMode::TowardNegative;
    const bool overflow =
        policy.integer_overflow == PhysicalIntegerOverflow::Reject ||
        policy.integer_overflow == PhysicalIntegerOverflow::Saturate;
    const bool nan = policy.nan == PhysicalNanPolicy::CanonicalQuiet ||
                     policy.nan == PhysicalNanPolicy::Reject ||
                     policy.nan == PhysicalNanPolicy::PreserveIeee;
    const bool infinity =
        policy.infinity == PhysicalInfinityPolicy::Preserve ||
        policy.infinity == PhysicalInfinityPolicy::SaturateFinite ||
        policy.infinity == PhysicalInfinityPolicy::Reject;
    const bool subnormal =
        policy.subnormal == PhysicalSubnormalPolicy::Preserve ||
        policy.subnormal == PhysicalSubnormalPolicy::FlushToSignedZero;
    const bool zero =
        policy.signed_zero == PhysicalSignedZeroPolicy::Preserve ||
        policy.signed_zero == PhysicalSignedZeroPolicy::Positive;
    const bool contraction =
        policy.contraction == PhysicalContractionPolicy::Separate ||
        policy.contraction == PhysicalContractionPolicy::Fused;
    return rounding && overflow && nan && infinity && subnormal && zero &&
           contraction;
}

bool canonical_float_policy(const PhysicalNumericPolicy& policy,
                            PhysicalContractionPolicy contraction) noexcept {
    return policy.rounding == PhysicalRoundingMode::NearestEven &&
           policy.integer_overflow == PhysicalIntegerOverflow::Reject &&
           policy.contraction == contraction;
}

bool canonical_integer_to_float_policy(
    const PhysicalNumericPolicy& policy) noexcept {
    return policy == PhysicalNumericPolicy{};
}

bool canonical_float_to_integer_policy(
    const PhysicalNumericPolicy& policy) noexcept {
    return policy.nan == PhysicalNanPolicy::Reject &&
           policy.infinity == PhysicalInfinityPolicy::Reject &&
           policy.subnormal == PhysicalSubnormalPolicy::Preserve &&
           policy.signed_zero == PhysicalSignedZeroPolicy::Preserve &&
           policy.contraction == PhysicalContractionPolicy::Separate;
}

bool instruction_syntax_valid(const PhysicalProgram& program, size_t index,
                              const PhysicalOpcodeShape& shape) noexcept {
    const PhysicalInstruction& instruction = program.instructions[index];
    const auto operand_type = [&](size_t operand) {
        return program.instructions[instruction.operands[operand]].result_type;
    };
    const auto unary = [&](PhysicalValueType input,
                           PhysicalValueType result) {
        return shape.operands == 1 && operand_type(0) == input &&
               instruction.result_type == result;
    };
    const auto binary = [&](PhysicalValueType input,
                            PhysicalValueType result) {
        return shape.operands == 2 && operand_type(0) == input &&
               operand_type(1) == input && instruction.result_type == result;
    };
    const auto ternary = [&](PhysicalValueType input,
                             PhysicalValueType result) {
        return shape.operands == 3 && operand_type(0) == input &&
               operand_type(1) == input && operand_type(2) == input &&
               instruction.result_type == result;
    };
    const bool default_bits = instruction.bit_width == 0 &&
                              instruction.bit_order ==
                                  PhysicalBitOrder::Lsb0Little;
    const bool no_immediate = instruction.immediate == 0;

    switch (instruction.opcode) {
        case PhysicalOpcode::ConstIndex:
            return instruction.result_type == PhysicalValueType::Index &&
                   default_bits;
        case PhysicalOpcode::ConstU32:
            return instruction.result_type == PhysicalValueType::U32 &&
                   instruction.immediate <= UINT32_MAX && default_bits;
        case PhysicalOpcode::ConstI32:
            return instruction.result_type == PhysicalValueType::I32 &&
                   instruction.immediate <= UINT32_MAX && default_bits;
        case PhysicalOpcode::ConstF32Bits:
            return instruction.result_type == PhysicalValueType::F32 &&
                   instruction.immediate <= UINT32_MAX && default_bits;
        case PhysicalOpcode::Coordinate:
        case PhysicalOpcode::Extent:
            return instruction.result_type == PhysicalValueType::Index &&
                   instruction.immediate < program.logical_rank && default_bits;
        case PhysicalOpcode::IndexAdd:
        case PhysicalOpcode::IndexSubtract:
        case PhysicalOpcode::IndexMultiply:
            return binary(PhysicalValueType::Index, PhysicalValueType::Index) &&
                   no_immediate && default_bits;
        case PhysicalOpcode::IndexDivideConstant:
        case PhysicalOpcode::IndexRemainderConstant:
        case PhysicalOpcode::IndexCeilDivideConstant:
            return unary(PhysicalValueType::Index, PhysicalValueType::Index) &&
                   instruction.immediate != 0 && default_bits;
        case PhysicalOpcode::IndexFromU32:
            return unary(PhysicalValueType::U32, PhysicalValueType::Index) &&
                   no_immediate && default_bits;
        case PhysicalOpcode::IndexLess:
            return binary(PhysicalValueType::Index,
                          PhysicalValueType::Predicate) &&
                   no_immediate && default_bits;
        case PhysicalOpcode::Select:
            return shape.operands == 3 &&
                   operand_type(0) == PhysicalValueType::Predicate &&
                   operand_type(1) == operand_type(2) &&
                   instruction.result_type == operand_type(1) && no_immediate &&
                   default_bits;
        case PhysicalOpcode::LoadBits:
            return unary(PhysicalValueType::Index, PhysicalValueType::U32) &&
                   instruction.bit_width >= 1 && instruction.bit_width <= 32 &&
                   (instruction.bit_order == PhysicalBitOrder::Lsb0Little ||
                    instruction.bit_order == PhysicalBitOrder::Msb0Big) &&
                   no_immediate;
        case PhysicalOpcode::U32And:
        case PhysicalOpcode::U32Or:
        case PhysicalOpcode::U32Xor:
        case PhysicalOpcode::U32Add:
        case PhysicalOpcode::U32Multiply:
            return binary(PhysicalValueType::U32, PhysicalValueType::U32) &&
                   no_immediate && default_bits;
        case PhysicalOpcode::U32ShiftLeftConstant:
        case PhysicalOpcode::U32ShiftRightConstant:
            return unary(PhysicalValueType::U32, PhysicalValueType::U32) &&
                   instruction.immediate < 32 && default_bits;
        case PhysicalOpcode::U32FunnelShiftRight:
            return shape.operands == 3 &&
                   operand_type(0) == PhysicalValueType::U32 &&
                   operand_type(1) == PhysicalValueType::U32 &&
                   operand_type(2) == PhysicalValueType::Index &&
                   instruction.result_type == PhysicalValueType::U32 &&
                   no_immediate && default_bits;
        case PhysicalOpcode::SignExtend:
            return unary(PhysicalValueType::U32, PhysicalValueType::I32) &&
                   instruction.bit_width >= 1 && instruction.bit_width <= 32 &&
                   no_immediate &&
                   instruction.bit_order == PhysicalBitOrder::Lsb0Little;
        case PhysicalOpcode::F16ToF32:
        case PhysicalOpcode::Bf16ToF32:
        case PhysicalOpcode::BitsToF32:
            return unary(PhysicalValueType::U32, PhysicalValueType::F32) &&
                   no_immediate && default_bits &&
                   canonical_float_policy(
                       program.policies[instruction.policy],
                       PhysicalContractionPolicy::Separate);
        case PhysicalOpcode::U32ToF32:
            return unary(PhysicalValueType::U32, PhysicalValueType::F32) &&
                   no_immediate && default_bits &&
                   canonical_integer_to_float_policy(
                       program.policies[instruction.policy]);
        case PhysicalOpcode::I32ToF32:
            return unary(PhysicalValueType::I32, PhysicalValueType::F32) &&
                   no_immediate && default_bits &&
                   canonical_integer_to_float_policy(
                       program.policies[instruction.policy]);
        case PhysicalOpcode::F32ToU32:
            return unary(PhysicalValueType::F32, PhysicalValueType::U32) &&
                   no_immediate && default_bits &&
                   canonical_float_to_integer_policy(
                       program.policies[instruction.policy]);
        case PhysicalOpcode::F32ToI32:
            return unary(PhysicalValueType::F32, PhysicalValueType::I32) &&
                   no_immediate && default_bits &&
                   canonical_float_to_integer_policy(
                       program.policies[instruction.policy]);
        case PhysicalOpcode::F32Add:
        case PhysicalOpcode::F32Subtract:
        case PhysicalOpcode::F32Multiply:
            return binary(PhysicalValueType::F32, PhysicalValueType::F32) &&
                   no_immediate && default_bits &&
                   canonical_float_policy(
                       program.policies[instruction.policy],
                       PhysicalContractionPolicy::Separate);
        case PhysicalOpcode::F32Fma:
            return ternary(PhysicalValueType::F32, PhysicalValueType::F32) &&
                   no_immediate && default_bits &&
                   canonical_float_policy(
                       program.policies[instruction.policy],
                       PhysicalContractionPolicy::Fused);
        case PhysicalOpcode::F32Negate:
            return unary(PhysicalValueType::F32, PhysicalValueType::F32) &&
                   no_immediate && default_bits &&
                   canonical_float_policy(
                       program.policies[instruction.policy],
                       PhysicalContractionPolicy::Separate);
        case PhysicalOpcode::F32RoundToF16:
            return unary(PhysicalValueType::F32, PhysicalValueType::F32) &&
                   no_immediate && default_bits &&
                   program.policies[instruction.policy] ==
                       PhysicalNumericPolicy{};
        case PhysicalOpcode::F32Clamp:
            return ternary(PhysicalValueType::F32, PhysicalValueType::F32) &&
                   no_immediate && default_bits &&
                   canonical_float_policy(
                       program.policies[instruction.policy],
                       PhysicalContractionPolicy::Separate);
    }
    return false;
}

bool power_of_two(uint32_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

struct CanonicalPhysicalProgram {
    PhysicalProgram program;
    std::vector<uint16_t> source_planes;
};

using CanonicalPhysicalResult =
    std::variant<CanonicalPhysicalProgram, CompatibilityReport>;

CanonicalPhysicalResult canonicalize_physical(PhysicalProgram source) {
    if (source.abi_version != 1)
        return error_report(CompatibilityError::IR_VERSION_UNSUPPORTED,
                            "physical program ABI is unsupported");
    if (source.logical_rank > 8 || source.planes.empty() ||
        source.planes.size() > kMaximumPhysicalPlanes ||
        source.policies.size() > kMaximumPhysicalPolicies ||
        source.instructions.empty() ||
        source.instructions.size() > kMaximumPhysicalNodes ||
        source.inline_bytes.size() > kMaximumPhysicalWireBytes ||
        source.result >= source.instructions.size())
        return error_report(CompatibilityError::IR_CONSTRAINT_FAILED,
                            "physical program limits are invalid");

    for (const PhysicalPlane& plane : source.planes) {
        if ((plane.storage != PhysicalPlaneStorage::External &&
             plane.storage != PhysicalPlaneStorage::Inline) ||
            !power_of_two(plane.alignment) || plane.alignment > 4096 ||
            (plane.storage == PhysicalPlaneStorage::Inline &&
             (plane.byte_length == 0 || plane.alignment != 1)))
            return error_report(CompatibilityError::IR_CONSTRAINT_FAILED,
                                "physical plane declaration is invalid");
        if (plane.storage == PhysicalPlaneStorage::External) {
            if (plane.inline_offset != 0 || plane.byte_length != 0)
                return error_report(CompatibilityError::IR_CONSTRAINT_FAILED,
                                    "external plane has inline storage");
        } else {
            if (plane.inline_offset > source.inline_bytes.size() ||
                plane.byte_length >
                    source.inline_bytes.size() - plane.inline_offset)
                return error_report(CompatibilityError::IR_REFERENCE_INVALID,
                                    "inline plane is out of range");
        }
    }
    for (const PhysicalNumericPolicy& policy : source.policies) {
        if (!valid_numeric_policy(policy))
            return error_report(CompatibilityError::IR_CONSTRAINT_FAILED,
                                "physical numeric policy is invalid");
    }

    std::vector<uint8_t> depth(source.instructions.size(), 0);
    for (size_t index = 0; index < source.instructions.size(); ++index) {
        const PhysicalInstruction& instruction = source.instructions[index];
        PhysicalOpcodeShape shape;
        if (!physical_opcode_shape(instruction.opcode, &shape) ||
            !valid_physical_value_type(instruction.result_type))
            return error_report(CompatibilityError::IR_VERSION_UNSUPPORTED,
                                "physical opcode or value type is unsupported");
        uint8_t instruction_depth = 1;
        for (size_t operand = 0; operand < instruction.operands.size();
             ++operand) {
            const uint32_t value = instruction.operands[operand];
            if (operand < shape.operands) {
                if (value >= index)
                    return error_report(CompatibilityError::IR_REFERENCE_INVALID,
                                        "physical SSA operand is out of order");
                instruction_depth = std::max<uint8_t>(
                    instruction_depth,
                    static_cast<uint8_t>(depth[value] + 1));
            } else if (value != kNoPhysicalValue) {
                return error_report(CompatibilityError::IR_REFERENCE_INVALID,
                                    "physical instruction has an extra operand");
            }
        }
        if (shape.plane) {
            if (instruction.plane >= source.planes.size())
                return error_report(CompatibilityError::IR_REFERENCE_INVALID,
                                    "physical plane reference is invalid");
        } else if (instruction.plane != kNoPhysicalPlane) {
            return error_report(CompatibilityError::IR_REFERENCE_INVALID,
                                "physical instruction has an extra plane");
        }
        if (shape.policy) {
            if (instruction.policy >= source.policies.size())
                return error_report(CompatibilityError::IR_REFERENCE_INVALID,
                                    "physical policy reference is invalid");
        } else if (instruction.policy != kNoPhysicalPolicy) {
            return error_report(CompatibilityError::IR_REFERENCE_INVALID,
                                "physical instruction has an extra policy");
        }
        if (!instruction_syntax_valid(source, index, shape))
            return error_report(CompatibilityError::IR_CONSTRAINT_FAILED,
                                "physical instruction signature is invalid");
        if (instruction_depth > kMaximumPhysicalDepth)
            return error_report(CompatibilityError::IR_CONSTRAINT_FAILED,
                                "physical expression depth exceeds V1 limit");
        depth[index] = instruction_depth;
    }

    std::vector<uint8_t> visited(source.instructions.size(), 0);
    std::vector<uint32_t> order;
    order.reserve(source.instructions.size());
    const std::function<void(uint32_t)> visit = [&](uint32_t index) {
        if (visited[index]) return;
        visited[index] = 1;
        PhysicalOpcodeShape shape;
        physical_opcode_shape(source.instructions[index].opcode, &shape);
        for (uint8_t operand = 0; operand < shape.operands; ++operand)
            visit(source.instructions[index].operands[operand]);
        order.push_back(index);
    };
    visit(source.result);
    if (order.size() != source.instructions.size())
        return error_report(CompatibilityError::IR_REFERENCE_INVALID,
                            "physical program contains unreachable instructions");
    std::vector<uint8_t> used_planes(source.planes.size(), 0);
    std::vector<uint8_t> used_policies(source.policies.size(), 0);
    for (const PhysicalInstruction& instruction : source.instructions) {
        PhysicalOpcodeShape shape;
        physical_opcode_shape(instruction.opcode, &shape);
        if (shape.plane) used_planes[instruction.plane] = 1;
        if (shape.policy) used_policies[instruction.policy] = 1;
    }
    if (std::find(used_planes.begin(), used_planes.end(), 0) !=
            used_planes.end() ||
        std::find(used_policies.begin(), used_policies.end(), 0) !=
            used_policies.end())
        return error_report(CompatibilityError::IR_REFERENCE_INVALID,
                            "physical program contains unreachable records");

    std::vector<uint32_t> value_remap(source.instructions.size(),
                                      kNoPhysicalValue);
    std::vector<uint16_t> plane_remap(source.planes.size(), kNoPhysicalPlane);
    std::vector<uint16_t> policy_remap(source.policies.size(),
                                       kNoPhysicalPolicy);
    CanonicalPhysicalProgram canonical;
    canonical.program.abi_version = source.abi_version;
    canonical.program.logical_rank = source.logical_rank;
    canonical.program.instructions.reserve(order.size());

    const auto map_plane = [&](uint16_t old_plane,
                               CanonicalPhysicalProgram* output) -> uint16_t {
        uint16_t& mapped = plane_remap[old_plane];
        if (mapped != kNoPhysicalPlane) return mapped;
        mapped = static_cast<uint16_t>(output->program.planes.size());
        PhysicalPlane plane = source.planes[old_plane];
        if (plane.storage == PhysicalPlaneStorage::Inline) {
            const uint64_t old_offset = plane.inline_offset;
            plane.inline_offset = output->program.inline_bytes.size();
            output->program.inline_bytes.insert(
                output->program.inline_bytes.end(),
                source.inline_bytes.begin() + old_offset,
                source.inline_bytes.begin() + old_offset + plane.byte_length);
        }
        output->program.planes.push_back(plane);
        output->source_planes.push_back(old_plane);
        return mapped;
    };
    const auto map_policy = [&](uint16_t old_policy,
                                CanonicalPhysicalProgram* output) -> uint16_t {
        uint16_t& mapped = policy_remap[old_policy];
        if (mapped != kNoPhysicalPolicy) return mapped;
        const auto existing = std::find(output->program.policies.begin(),
                                        output->program.policies.end(),
                                        source.policies[old_policy]);
        if (existing != output->program.policies.end()) {
            mapped = static_cast<uint16_t>(
                existing - output->program.policies.begin());
            return mapped;
        }
        mapped = static_cast<uint16_t>(output->program.policies.size());
        output->program.policies.push_back(source.policies[old_policy]);
        return mapped;
    };

    for (uint32_t old_index : order) {
        PhysicalInstruction instruction = source.instructions[old_index];
        PhysicalOpcodeShape shape;
        physical_opcode_shape(instruction.opcode, &shape);
        for (uint8_t operand = 0; operand < shape.operands; ++operand)
            instruction.operands[operand] =
                value_remap[instruction.operands[operand]];
        const bool constant =
            instruction.opcode == PhysicalOpcode::ConstIndex ||
            instruction.opcode == PhysicalOpcode::ConstU32 ||
            instruction.opcode == PhysicalOpcode::ConstI32 ||
            instruction.opcode == PhysicalOpcode::ConstF32Bits;
        if (constant) {
            const auto existing = std::find(
                canonical.program.instructions.begin(),
                canonical.program.instructions.end(), instruction);
            if (existing != canonical.program.instructions.end()) {
                value_remap[old_index] = static_cast<uint32_t>(
                    existing - canonical.program.instructions.begin());
                continue;
            }
        }
        if (shape.plane)
            instruction.plane = map_plane(instruction.plane, &canonical);
        if (shape.policy)
            instruction.policy = map_policy(instruction.policy, &canonical);
        value_remap[old_index] =
            static_cast<uint32_t>(canonical.program.instructions.size());
        canonical.program.instructions.push_back(instruction);
    }
    canonical.program.result = value_remap[source.result];
    return canonical;
}

void append_section(std::vector<uint8_t>& wire, uint16_t tag,
                    const std::vector<uint8_t>& body) {
    append_u16(wire, tag);
    append_u32(wire, static_cast<uint32_t>(body.size()));
    wire.insert(wire.end(), body.begin(), body.end());
}

PhysicalProgramWireResult encode_canonical_physical(
    const PhysicalProgram& program) {
    std::vector<uint8_t> metadata;
    append_u16(metadata, program.abi_version);
    append_u8(metadata, program.logical_rank);
    append_u8(metadata, 0);
    append_u32(metadata, program.result);

    std::vector<uint8_t> planes;
    append_u32(planes, static_cast<uint32_t>(program.planes.size()));
    for (const PhysicalPlane& plane : program.planes) {
        append_u8(planes, static_cast<uint8_t>(plane.storage));
        append_u8(planes, 0);
        append_u16(planes, 0);
        append_u32(planes, plane.alignment);
        append_u64(planes, plane.inline_offset);
        append_u64(planes, plane.byte_length);
    }

    std::vector<uint8_t> policies;
    append_u32(policies, static_cast<uint32_t>(program.policies.size()));
    for (const PhysicalNumericPolicy& policy : program.policies) {
        append_u8(policies, static_cast<uint8_t>(policy.rounding));
        append_u8(policies, static_cast<uint8_t>(policy.integer_overflow));
        append_u8(policies, static_cast<uint8_t>(policy.nan));
        append_u8(policies, static_cast<uint8_t>(policy.infinity));
        append_u8(policies, static_cast<uint8_t>(policy.subnormal));
        append_u8(policies, static_cast<uint8_t>(policy.signed_zero));
        append_u8(policies, static_cast<uint8_t>(policy.contraction));
        append_u8(policies, 0);
    }

    std::vector<uint8_t> instructions;
    append_u32(instructions,
               static_cast<uint32_t>(program.instructions.size()));
    for (const PhysicalInstruction& instruction : program.instructions) {
        append_u16(instructions, static_cast<uint16_t>(instruction.opcode));
        append_u8(instructions,
                  static_cast<uint8_t>(instruction.result_type));
        append_u8(instructions, static_cast<uint8_t>(instruction.bit_order));
        for (uint32_t operand : instruction.operands)
            append_u32(instructions, operand);
        append_u16(instructions, instruction.plane);
        append_u16(instructions, instruction.policy);
        append_u64(instructions, instruction.immediate);
        append_u8(instructions, instruction.bit_width);
        for (int index = 0; index < 7; ++index) append_u8(instructions, 0);
    }

    std::vector<uint8_t> wire;
    const std::array<uint8_t, 8> magic =
        {'L', 'A', 'P', 'P', 'H', 'Y', '0', '1'};
    wire.insert(wire.end(), magic.begin(), magic.end());
    append_u16(wire, 1);
    append_u16(wire, 5);
    append_u32(wire, 0);
    append_section(wire, 1, metadata);
    append_section(wire, 2, planes);
    append_section(wire, 3, policies);
    append_section(wire, 4, instructions);
    append_section(wire, 5, program.inline_bytes);
    if (wire.size() > kMaximumPhysicalWireBytes)
        return error_report(CompatibilityError::IR_CONSTRAINT_FAILED,
                            "physical program wire exceeds V1 limit");
    const uint32_t total = static_cast<uint32_t>(wire.size());
    for (unsigned byte = 0; byte != 4; ++byte)
        wire[12 + byte] = static_cast<uint8_t>(total >> (byte * 8));
    return wire;
}

class PhysicalWireReader {
  public:
    explicit PhysicalWireReader(std::span<const uint8_t> bytes) : bytes_(bytes) {}

    bool u8(uint8_t* value) {
        if (!value || position_ >= bytes_.size()) return false;
        *value = bytes_[position_++];
        return true;
    }
    bool u16(uint16_t* value) {
        uint8_t a = 0, b = 0;
        if (!u8(&a) || !u8(&b)) return false;
        *value = static_cast<uint16_t>(a | (static_cast<uint16_t>(b) << 8));
        return true;
    }
    bool u32(uint32_t* value) {
        if (!value || remaining() < 4) return false;
        uint32_t result = 0;
        for (unsigned byte = 0; byte != 4; ++byte)
            result |= static_cast<uint32_t>(bytes_[position_++]) << (byte * 8);
        *value = result;
        return true;
    }
    bool u64(uint64_t* value) {
        if (!value || remaining() < 8) return false;
        uint64_t result = 0;
        for (unsigned byte = 0; byte != 8; ++byte)
            result |= static_cast<uint64_t>(bytes_[position_++]) << (byte * 8);
        *value = result;
        return true;
    }
    bool bytes(size_t count, std::span<const uint8_t>* out) {
        if (!out || count > remaining()) return false;
        *out = bytes_.subspan(position_, count);
        position_ += count;
        return true;
    }
    size_t remaining() const noexcept { return bytes_.size() - position_; }
    size_t position() const noexcept { return position_; }

  private:
    std::span<const uint8_t> bytes_;
    size_t position_ = 0;
};

} // namespace

PhysicalProgramResult canonicalize_physical_program(PhysicalProgram program) {
    try {
        CanonicalPhysicalResult result = canonicalize_physical(std::move(program));
        if (const auto* report = std::get_if<CompatibilityReport>(&result))
            return *report;
        return std::move(std::get<CanonicalPhysicalProgram>(result).program);
    } catch (const std::bad_alloc&) {
        return error_report(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                            "physical program canonicalization exceeded memory limits");
    }
}

PhysicalProgramWireResult encode_physical_program(
    const PhysicalProgram& program) {
    try {
        CanonicalPhysicalResult canonical = canonicalize_physical(program);
        if (const auto* report = std::get_if<CompatibilityReport>(&canonical))
            return *report;
        return encode_canonical_physical(
            std::get<CanonicalPhysicalProgram>(canonical).program);
    } catch (const std::bad_alloc&) {
        return error_report(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                            "physical program encoding exceeded memory limits");
    }
}

PhysicalProgramResult parse_physical_program(std::span<const uint8_t> wire) {
    try {
        if (wire.size() < 16 || wire.size() > kMaximumPhysicalWireBytes)
            return error_report(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                "physical program wire size is invalid");
        const std::array<uint8_t, 8> magic =
            {'L', 'A', 'P', 'P', 'H', 'Y', '0', '1'};
        if (!std::equal(magic.begin(), magic.end(), wire.begin()))
            return error_report(CompatibilityError::PACKAGE_BAD_MAGIC,
                                "physical program magic is invalid");
        PhysicalWireReader reader(wire.subspan(8));
        uint16_t wire_version = 0;
        uint16_t section_count = 0;
        uint32_t total_size = 0;
        if (!reader.u16(&wire_version) || !reader.u16(&section_count) ||
            !reader.u32(&total_size) || wire_version != 1 ||
            section_count != 5 || total_size != wire.size())
            return error_report(CompatibilityError::PACKAGE_VERSION_UNSUPPORTED,
                                "physical program header is invalid");

        std::array<std::span<const uint8_t>, 5> sections{};
        for (uint16_t expected = 1; expected <= 5; ++expected) {
            uint16_t tag = 0;
            uint32_t length = 0;
            if (!reader.u16(&tag) || !reader.u32(&length) || tag != expected ||
                !reader.bytes(length, &sections[expected - 1]))
                return error_report(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                    "physical program section is invalid");
        }
        if (reader.remaining() != 0)
            return error_report(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                "physical program has trailing bytes");

        PhysicalProgram program;
        {
            PhysicalWireReader section(sections[0]);
            uint8_t reserved = 0;
            if (!section.u16(&program.abi_version) ||
                !section.u8(&program.logical_rank) || !section.u8(&reserved) ||
                !section.u32(&program.result) || reserved != 0 ||
                section.remaining() != 0)
                return error_report(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                    "physical metadata section is invalid");
        }
        {
            PhysicalWireReader section(sections[1]);
            uint32_t count = 0;
            if (!section.u32(&count) || count == 0 ||
                count > kMaximumPhysicalPlanes)
                return error_report(CompatibilityError::IR_CONSTRAINT_FAILED,
                                    "physical plane count is invalid");
            program.planes.reserve(count);
            for (uint32_t index = 0; index < count; ++index) {
                uint8_t storage = 0, reserved8 = 0;
                uint16_t reserved16 = 0;
                PhysicalPlane plane;
                if (!section.u8(&storage) || !section.u8(&reserved8) ||
                    !section.u16(&reserved16) ||
                    !section.u32(&plane.alignment) ||
                    !section.u64(&plane.inline_offset) ||
                    !section.u64(&plane.byte_length) || reserved8 != 0 ||
                    reserved16 != 0)
                    return error_report(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                        "physical plane record is invalid");
                plane.storage = static_cast<PhysicalPlaneStorage>(storage);
                program.planes.push_back(plane);
            }
            if (section.remaining() != 0)
                return error_report(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                    "physical plane section has trailing bytes");
        }
        {
            PhysicalWireReader section(sections[2]);
            uint32_t count = 0;
            if (!section.u32(&count) || count > kMaximumPhysicalPolicies)
                return error_report(CompatibilityError::IR_CONSTRAINT_FAILED,
                                    "physical policy count is invalid");
            program.policies.reserve(count);
            for (uint32_t index = 0; index < count; ++index) {
                uint8_t rounding = 0, overflow = 0, nan = 0, infinity = 0,
                        subnormal = 0, zero = 0, contraction = 0, reserved = 0;
                if (!section.u8(&rounding) || !section.u8(&overflow) ||
                    !section.u8(&nan) || !section.u8(&infinity) ||
                    !section.u8(&subnormal) || !section.u8(&zero) ||
                    !section.u8(&contraction) || !section.u8(&reserved) ||
                    reserved != 0)
                    return error_report(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                        "physical policy record is invalid");
                program.policies.push_back(
                    {static_cast<PhysicalRoundingMode>(rounding),
                     static_cast<PhysicalIntegerOverflow>(overflow),
                     static_cast<PhysicalNanPolicy>(nan),
                     static_cast<PhysicalInfinityPolicy>(infinity),
                     static_cast<PhysicalSubnormalPolicy>(subnormal),
                     static_cast<PhysicalSignedZeroPolicy>(zero),
                     static_cast<PhysicalContractionPolicy>(contraction)});
            }
            if (section.remaining() != 0)
                return error_report(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                    "physical policy section has trailing bytes");
        }
        {
            PhysicalWireReader section(sections[3]);
            uint32_t count = 0;
            if (!section.u32(&count) || count == 0 ||
                count > kMaximumPhysicalNodes)
                return error_report(CompatibilityError::IR_CONSTRAINT_FAILED,
                                    "physical instruction count is invalid");
            program.instructions.reserve(count);
            for (uint32_t index = 0; index < count; ++index) {
                uint16_t opcode = 0;
                uint8_t type = 0, bit_order = 0;
                PhysicalInstruction instruction;
                if (!section.u16(&opcode) || !section.u8(&type) ||
                    !section.u8(&bit_order))
                    return error_report(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                        "physical instruction record is invalid");
                for (uint32_t& operand : instruction.operands) {
                    if (!section.u32(&operand))
                        return error_report(
                            CompatibilityError::PACKAGE_BOUNDS_INVALID,
                            "physical instruction operand is truncated");
                }
                if (!section.u16(&instruction.plane) ||
                    !section.u16(&instruction.policy) ||
                    !section.u64(&instruction.immediate) ||
                    !section.u8(&instruction.bit_width))
                    return error_report(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                        "physical instruction record is truncated");
                for (int reserved_index = 0; reserved_index < 7;
                     ++reserved_index) {
                    uint8_t reserved = 0;
                    if (!section.u8(&reserved) || reserved != 0)
                        return error_report(
                            CompatibilityError::PACKAGE_BOUNDS_INVALID,
                            "physical instruction reserved byte is invalid");
                }
                instruction.opcode = static_cast<PhysicalOpcode>(opcode);
                instruction.result_type =
                    static_cast<PhysicalValueType>(type);
                instruction.bit_order =
                    static_cast<PhysicalBitOrder>(bit_order);
                program.instructions.push_back(instruction);
            }
            if (section.remaining() != 0)
                return error_report(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                    "physical instruction section has trailing bytes");
        }
        program.inline_bytes.assign(sections[4].begin(), sections[4].end());

        PhysicalProgramResult canonical =
            canonicalize_physical_program(std::move(program));
        if (const auto* report = std::get_if<CompatibilityReport>(&canonical))
            return *report;
        PhysicalProgramWireResult reencoded = encode_canonical_physical(
            std::get<PhysicalProgram>(canonical));
        const auto* bytes = std::get_if<std::vector<uint8_t>>(&reencoded);
        if (!bytes || bytes->size() != wire.size() ||
            !std::equal(bytes->begin(), bytes->end(), wire.begin()))
            return error_report(CompatibilityError::IR_CONSTRAINT_FAILED,
                                "physical program wire is not canonical");
        return canonical;
    } catch (const std::bad_alloc&) {
        return error_report(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                            "physical program parsing exceeded memory limits");
    }
}

PhysicalProgramDigestResult physical_program_digest(
    const PhysicalProgram& program) {
    try {
        PhysicalProgramWireResult encoded = encode_physical_program(program);
        if (const auto* report = std::get_if<CompatibilityReport>(&encoded))
            return *report;
        const auto& wire = std::get<std::vector<uint8_t>>(encoded);
        const char domain[] = "laplace-physical-program-v1";
        std::vector<uint8_t> preimage;
        preimage.reserve(sizeof(domain) + 8 + wire.size());
        preimage.insert(preimage.end(), domain, domain + sizeof(domain));
        append_u64(preimage, wire.size());
        preimage.insert(preimage.end(), wire.begin(), wire.end());
        PhysicalProgramDigest digest{};
        CC_SHA256(preimage.data(), static_cast<CC_LONG>(preimage.size()),
                  digest.data());
        return digest;
    } catch (const std::bad_alloc&) {
        return error_report(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                            "physical digest exceeded memory limits");
    }
}

} // namespace Laplace
