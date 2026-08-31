#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include "compatibility_report.h"
#include "program_ir.h"

namespace Laplace {

using PhysicalProgramDigest = std::array<uint8_t, 32>;

inline constexpr uint32_t kNoPhysicalValue = UINT32_MAX;
inline constexpr uint16_t kNoPhysicalPlane = UINT16_MAX;
inline constexpr uint16_t kNoPhysicalPolicy = UINT16_MAX;

enum class PhysicalValueType : uint8_t {
    Predicate = 1,
    Index = 2,
    U32 = 3,
    I32 = 4,
    F32 = 5,
};

enum class PhysicalPlaneStorage : uint8_t {
    External = 1,
    Inline = 2,
};

enum class PhysicalBitOrder : uint8_t {
    Lsb0Little = 1,
    Msb0Big = 2,
};

enum class PhysicalRoundingMode : uint8_t {
    NearestEven = 1,
    TowardZero = 2,
    TowardPositive = 3,
    TowardNegative = 4,
};

enum class PhysicalIntegerOverflow : uint8_t {
    Reject = 1,
    Saturate = 2,
};

enum class PhysicalNanPolicy : uint8_t {
    CanonicalQuiet = 1,
    Reject = 2,
};

enum class PhysicalInfinityPolicy : uint8_t {
    Preserve = 1,
    SaturateFinite = 2,
    Reject = 3,
};

enum class PhysicalSubnormalPolicy : uint8_t {
    Preserve = 1,
    FlushToSignedZero = 2,
};

enum class PhysicalSignedZeroPolicy : uint8_t {
    Preserve = 1,
    Positive = 2,
};

enum class PhysicalContractionPolicy : uint8_t {
    Separate = 1,
    Fused = 2,
};

// Every floating-point or narrowing operation names one immutable policy.
// No host compiler or Metal default is part of the physical-program contract.
struct PhysicalNumericPolicy {
    PhysicalRoundingMode rounding = PhysicalRoundingMode::NearestEven;
    PhysicalIntegerOverflow integer_overflow =
        PhysicalIntegerOverflow::Reject;
    PhysicalNanPolicy nan = PhysicalNanPolicy::CanonicalQuiet;
    PhysicalInfinityPolicy infinity = PhysicalInfinityPolicy::Preserve;
    PhysicalSubnormalPolicy subnormal = PhysicalSubnormalPolicy::Preserve;
    PhysicalSignedZeroPolicy signed_zero =
        PhysicalSignedZeroPolicy::Preserve;
    PhysicalContractionPolicy contraction =
        PhysicalContractionPolicy::Separate;
    friend bool operator==(const PhysicalNumericPolicy&,
                           const PhysicalNumericPolicy&) = default;
};

// Plane numbers are local, anonymous program operands. Artifact identity and
// source-format taxonomy are bound outside this language.
struct PhysicalPlane {
    PhysicalPlaneStorage storage = PhysicalPlaneStorage::External;
    uint32_t alignment = 1;
    uint64_t inline_offset = 0;
    uint64_t byte_length = 0;
    friend bool operator==(const PhysicalPlane&, const PhysicalPlane&) = default;
};

enum class PhysicalOpcode : uint16_t {
    ConstIndex = 1,
    ConstU32 = 2,
    ConstI32 = 3,
    ConstF32Bits = 4,
    Coordinate = 5,
    Extent = 6,

    IndexAdd = 7,
    IndexSubtract = 8,
    IndexMultiply = 9,
    IndexDivideConstant = 10,
    IndexRemainderConstant = 11,
    IndexCeilDivideConstant = 12,
    IndexFromU32 = 13,
    IndexLess = 14,
    Select = 15,

    LoadBits = 16,
    U32And = 17,
    U32Or = 18,
    U32Xor = 19,
    U32ShiftLeftConstant = 20,
    U32ShiftRightConstant = 21,
    SignExtend = 22,

    F16ToF32 = 23,
    Bf16ToF32 = 24,
    BitsToF32 = 25,
    U32ToF32 = 26,
    I32ToF32 = 27,
    F32ToU32 = 28,
    F32ToI32 = 29,
    F32Add = 30,
    F32Subtract = 31,
    F32Multiply = 32,
    F32Fma = 33,
    F32Negate = 34,
    F32Clamp = 35,
};

struct PhysicalInstruction {
    PhysicalOpcode opcode = PhysicalOpcode::ConstIndex;
    PhysicalValueType result_type = PhysicalValueType::Index;
    std::array<uint32_t, 3> operands = {
        kNoPhysicalValue, kNoPhysicalValue, kNoPhysicalValue};
    uint16_t plane = kNoPhysicalPlane;
    uint16_t policy = kNoPhysicalPolicy;
    uint64_t immediate = 0;
    uint8_t bit_width = 0;
    PhysicalBitOrder bit_order = PhysicalBitOrder::Lsb0Little;
    friend bool operator==(const PhysicalInstruction&,
                           const PhysicalInstruction&) = default;
};

struct LogicalTensorType {
    ElementType element_type = ElementType::F32;
    std::vector<uint64_t> extents;
    friend bool operator==(const LogicalTensorType&,
                           const LogicalTensorType&) = default;
};

// This is the only Task-3 physical-program representation. Its complete
// meaning is the typed program, anonymous planes, and explicit policies below.
struct PhysicalProgram {
    uint16_t abi_version = 1;
    uint8_t logical_rank = 0;
    std::vector<PhysicalPlane> planes;
    std::vector<PhysicalNumericPolicy> policies;
    std::vector<PhysicalInstruction> instructions;
    uint32_t result = kNoPhysicalValue;
    std::vector<uint8_t> inline_bytes;
    friend bool operator==(const PhysicalProgram&, const PhysicalProgram&) =
        default;
};

using PhysicalProgramResult =
    std::variant<PhysicalProgram, CompatibilityReport>;
using PhysicalProgramWireResult =
    std::variant<std::vector<uint8_t>, CompatibilityReport>;
using PhysicalProgramDigestResult =
    std::variant<PhysicalProgramDigest, CompatibilityReport>;

PhysicalProgramResult canonicalize_physical_program(PhysicalProgram program);
PhysicalProgramWireResult encode_physical_program(const PhysicalProgram& program);
PhysicalProgramResult parse_physical_program(std::span<const uint8_t> wire);
PhysicalProgramDigestResult physical_program_digest(
    const PhysicalProgram& program);

} // namespace Laplace
