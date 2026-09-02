#include "physical_program_interpreter.h"

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

namespace Laplace {
namespace {

static_assert(sizeof(float) == sizeof(uint32_t));
static_assert(std::numeric_limits<float>::is_iec559);

CompatibilityReport physical_error(CompatibilityError code,
                                   const char* detail) {
    return compatibility_report(code, detail);
}

struct IntegerRange {
    uint64_t lower = 0;
    uint64_t upper = 0;
    bool known = false;
};

bool add_u64(uint64_t left, uint64_t right, uint64_t* out) noexcept {
    if (!out || right > std::numeric_limits<uint64_t>::max() - left)
        return false;
    *out = left + right;
    return true;
}

bool multiply_u64(uint64_t left, uint64_t right, uint64_t* out) noexcept {
    if (!out || (left != 0 &&
                 right > std::numeric_limits<uint64_t>::max() / left))
        return false;
    *out = left * right;
    return true;
}

ElementType logical_element_type(PhysicalValueType type) noexcept {
    switch (type) {
        case PhysicalValueType::Predicate: return ElementType::I1;
        case PhysicalValueType::Index: return ElementType::U64;
        case PhysicalValueType::U32: return ElementType::U32;
        case PhysicalValueType::I32: return ElementType::I32;
        case PhysicalValueType::F32: return ElementType::F32;
    }
    return static_cast<ElementType>(0);
}

bool exact_range(const IntegerRange& range, uint64_t* value) noexcept {
    if (!range.known || range.lower != range.upper || !value) return false;
    *value = range.lower;
    return true;
}

bool quotient_times_divisor(const PhysicalProgram& program, uint32_t node,
                            uint32_t* source, uint64_t* divisor) noexcept {
    if (node >= program.instructions.size()) return false;
    const PhysicalInstruction& multiply = program.instructions[node];
    if (multiply.opcode != PhysicalOpcode::IndexMultiply) return false;
    for (int order = 0; order < 2; ++order) {
        const uint32_t quotient_id = multiply.operands[order];
        const uint32_t constant_id = multiply.operands[1 - order];
        const PhysicalInstruction& quotient = program.instructions[quotient_id];
        const PhysicalInstruction& constant = program.instructions[constant_id];
        if (quotient.opcode == PhysicalOpcode::IndexDivideConstant &&
            constant.opcode == PhysicalOpcode::ConstIndex &&
            quotient.immediate == constant.immediate &&
            quotient.immediate != 0) {
            *source = quotient.operands[0];
            *divisor = quotient.immediate;
            return true;
        }
    }
    return false;
}

bool quotient_remainder_identity(const PhysicalProgram& program,
                                 const PhysicalInstruction& add,
                                 uint32_t* source) noexcept {
    for (int order = 0; order < 2; ++order) {
        uint32_t quotient_source = UINT32_MAX;
        uint64_t divisor = 0;
        if (!quotient_times_divisor(program, add.operands[order],
                                    &quotient_source, &divisor))
            continue;
        const PhysicalInstruction& remainder =
            program.instructions[add.operands[1 - order]];
        if (remainder.opcode == PhysicalOpcode::IndexRemainderConstant &&
            remainder.operands[0] == quotient_source &&
            remainder.immediate == divisor) {
            *source = quotient_source;
            return true;
        }
    }
    return false;
}

bool range_for_program(const PhysicalProgram& program,
                       const LogicalTensorType& logical,
                       const std::vector<uint64_t>& plane_bytes,
                       std::vector<IntegerRange>* ranges,
                       CompatibilityReport* failure) {
    ranges->assign(program.instructions.size(), {});
    for (size_t index = 0; index < program.instructions.size(); ++index) {
        const PhysicalInstruction& instruction = program.instructions[index];
        IntegerRange range;
        const auto input = [&](size_t operand) -> const IntegerRange& {
            return (*ranges)[instruction.operands[operand]];
        };
        uint64_t upper = 0;
        switch (instruction.opcode) {
            case PhysicalOpcode::ConstIndex:
                range = {instruction.immediate, instruction.immediate, true};
                break;
            case PhysicalOpcode::ConstU32:
                range = {static_cast<uint32_t>(instruction.immediate),
                         static_cast<uint32_t>(instruction.immediate), true};
                break;
            case PhysicalOpcode::ConstI32:
            case PhysicalOpcode::ConstF32Bits:
                break;
            case PhysicalOpcode::Coordinate:
                range = {0, logical.extents[instruction.immediate] - 1, true};
                break;
            case PhysicalOpcode::Extent:
                range = {logical.extents[instruction.immediate],
                         logical.extents[instruction.immediate], true};
                break;
            case PhysicalOpcode::IndexAdd: {
                uint32_t identity_source = UINT32_MAX;
                if (quotient_remainder_identity(program, instruction,
                                                &identity_source)) {
                    range = (*ranges)[identity_source];
                    break;
                }
                if (!input(0).known || !input(1).known ||
                    !add_u64(input(0).upper, input(1).upper, &upper))
                    goto unproved_index;
                range = {input(0).lower + input(1).lower, upper, true};
                break;
            }
            case PhysicalOpcode::IndexSubtract:
                if (!input(0).known || !input(1).known ||
                    input(0).lower < input(1).upper)
                    goto unproved_index;
                range = {input(0).lower - input(1).upper,
                         input(0).upper - input(1).lower, true};
                break;
            case PhysicalOpcode::IndexMultiply:
                if (!input(0).known || !input(1).known ||
                    !multiply_u64(input(0).upper, input(1).upper, &upper))
                    goto unproved_index;
                range = {input(0).lower * input(1).lower, upper, true};
                break;
            case PhysicalOpcode::IndexDivideConstant:
                if (!input(0).known) goto unproved_index;
                range = {input(0).lower / instruction.immediate,
                         input(0).upper / instruction.immediate, true};
                break;
            case PhysicalOpcode::IndexRemainderConstant:
                if (!input(0).known) goto unproved_index;
                range = {0,
                         std::min<uint64_t>(input(0).upper,
                                            instruction.immediate - 1),
                         true};
                break;
            case PhysicalOpcode::IndexCeilDivideConstant:
                if (!input(0).known) goto unproved_index;
                range = {
                    input(0).lower / instruction.immediate +
                        (input(0).lower % instruction.immediate != 0),
                    input(0).upper / instruction.immediate +
                        (input(0).upper % instruction.immediate != 0),
                    true};
                break;
            case PhysicalOpcode::IndexFromU32:
                if (!input(0).known) goto unproved_index;
                range = input(0);
                break;
            case PhysicalOpcode::IndexLess:
                range = {0, 1, true};
                break;
            case PhysicalOpcode::Select:
                if ((instruction.result_type == PhysicalValueType::Index ||
                     instruction.result_type == PhysicalValueType::U32) &&
                    input(1).known && input(2).known)
                    range = {std::min(input(1).lower, input(2).lower),
                             std::max(input(1).upper, input(2).upper), true};
                break;
            case PhysicalOpcode::LoadBits: {
                if (!input(0).known) goto unproved_address;
                uint64_t plane_bits = 0;
                uint64_t end = 0;
                if (instruction.plane >= plane_bytes.size() ||
                    !multiply_u64(plane_bytes[instruction.plane], 8,
                                  &plane_bits) ||
                    !add_u64(input(0).upper, instruction.bit_width, &end) ||
                    end > plane_bits)
                    goto unproved_address;
                range = {0,
                         instruction.bit_width == 32
                             ? UINT32_MAX
                             : ((uint64_t{1} << instruction.bit_width) - 1),
                         true};
                break;
            }
            case PhysicalOpcode::U32And: {
                uint64_t constant = 0;
                if (exact_range(input(0), &constant))
                    range = {0, static_cast<uint32_t>(constant), true};
                else if (exact_range(input(1), &constant))
                    range = {0, static_cast<uint32_t>(constant), true};
                else
                    range = {0, UINT32_MAX, true};
                break;
            }
            case PhysicalOpcode::U32Or:
            case PhysicalOpcode::U32Xor:
            case PhysicalOpcode::U32Add:
            case PhysicalOpcode::U32Multiply:
                range = {0, UINT32_MAX, true};
                break;
            case PhysicalOpcode::U32ShiftLeftConstant:
                if (!input(0).known ||
                    input(0).upper >
                        (UINT32_MAX >> instruction.immediate))
                    range = {0, UINT32_MAX, true};
                else
                    range = {input(0).lower << instruction.immediate,
                             input(0).upper << instruction.immediate, true};
                break;
            case PhysicalOpcode::U32ShiftRightConstant:
                if (!input(0).known)
                    range = {0, UINT32_MAX >> instruction.immediate, true};
                else
                    range = {input(0).lower >> instruction.immediate,
                             input(0).upper >> instruction.immediate, true};
                break;
            case PhysicalOpcode::U32FunnelShiftRight:
                if (!input(2).known || input(2).upper >= 64)
                    goto unproved_index;
                range = {0, UINT32_MAX, true};
                break;
            case PhysicalOpcode::F32ToU32:
                range = {0, UINT32_MAX, true};
                break;
            case PhysicalOpcode::SignExtend:
            case PhysicalOpcode::F16ToF32:
            case PhysicalOpcode::Bf16ToF32:
            case PhysicalOpcode::BitsToF32:
            case PhysicalOpcode::U32ToF32:
            case PhysicalOpcode::I32ToF32:
            case PhysicalOpcode::F32ToI32:
            case PhysicalOpcode::F32Add:
            case PhysicalOpcode::F32Subtract:
            case PhysicalOpcode::F32Multiply:
            case PhysicalOpcode::F32Fma:
            case PhysicalOpcode::F32Negate:
            case PhysicalOpcode::F32Clamp:
            case PhysicalOpcode::F32RoundToF16:
                break;
        }
        (*ranges)[index] = range;
        continue;

    unproved_index:
        *failure = physical_error(CompatibilityError::IR_CONSTRAINT_FAILED,
                                  "physical index arithmetic is not statically bounded");
        return false;
    unproved_address:
        *failure = physical_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                  "physical bit load is not statically in bounds");
        return false;
    }
    return true;
}

struct RuntimeValue {
    PhysicalValueType type = PhysicalValueType::Index;
    uint64_t bits = 0;
};

class ScopedNearestRounding {
  public:
    ScopedNearestRounding() noexcept {
        if (std::fegetenv(&previous_) != 0) return;
        saved_ = true;
        if (std::fesetenv(FE_DFL_ENV) != 0 ||
            std::fesetround(FE_TONEAREST) != 0)
            return;
        valid_ = std::fegetround() == FE_TONEAREST;
    }

    ~ScopedNearestRounding() {
        if (saved_) (void)std::fesetenv(&previous_);
    }

    bool valid() const noexcept { return valid_; }

    ScopedNearestRounding(const ScopedNearestRounding&) = delete;
    ScopedNearestRounding& operator=(const ScopedNearestRounding&) = delete;

  private:
    fenv_t previous_{};
    bool saved_ = false;
    bool valid_ = false;
};

uint32_t half_to_float_bits(uint16_t half) noexcept {
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
    const uint32_t exponent = (half >> 10) & 0x1fu;
    uint32_t fraction = half & 0x03ffu;
    if (exponent == 0) {
        if (fraction == 0) return sign;
        int shift = 0;
        while ((fraction & 0x0400u) == 0) {
            fraction <<= 1;
            ++shift;
        }
        fraction &= 0x03ffu;
        return sign | static_cast<uint32_t>(113 - shift) << 23 |
               fraction << 13;
    }
    if (exponent == 0x1fu)
        return sign | 0x7f800000u | fraction << 13;
    return sign | (exponent + 112u) << 23 | fraction << 13;
}

uint16_t float_to_half_nearest_even(uint32_t bits) noexcept {
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t fraction = bits & 0x007fffffu;
    if (exponent == 0xffu)
        return static_cast<uint16_t>(sign | 0x7c00u |
                                     (fraction == 0 ? 0u : 0x0200u));

    const int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31)
        return static_cast<uint16_t>(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        const uint32_t significand = fraction | 0x00800000u;
        const unsigned shift = static_cast<unsigned>(14 - half_exponent);
        uint32_t rounded = significand >> shift;
        const uint32_t remainder =
            significand & ((uint32_t{1} << shift) - 1u);
        const uint32_t halfway = uint32_t{1} << (shift - 1u);
        if (remainder > halfway ||
            (remainder == halfway && (rounded & 1u)))
            ++rounded;
        return static_cast<uint16_t>(sign | rounded);
    }

    uint32_t rounded = static_cast<uint32_t>(sign) |
                       (static_cast<uint32_t>(half_exponent) << 10) |
                       (fraction >> 13);
    const uint32_t remainder = fraction & 0x1fffu;
    if (remainder > 0x1000u ||
        (remainder == 0x1000u && (rounded & 1u)))
        ++rounded;
    return static_cast<uint16_t>(rounded);
}

uint32_t unsigned_to_float_bits(uint32_t value) noexcept {
    if (value == 0) return 0;
    const unsigned leading = std::countl_zero(value);
    unsigned exponent = 31u - leading;
    uint32_t significand = 0;
    if (exponent <= 23) {
        significand = value << (23 - exponent);
    } else {
        const unsigned shift = exponent - 23;
        significand = value >> shift;
        const uint32_t remainder_mask = (uint32_t{1} << shift) - 1;
        const uint32_t remainder = value & remainder_mask;
        const uint32_t halfway = uint32_t{1} << (shift - 1);
        if (remainder > halfway ||
            (remainder == halfway && (significand & 1u))) {
            ++significand;
            if (significand == (uint32_t{1} << 24)) {
                significand >>= 1;
                ++exponent;
            }
        }
    }
    return ((exponent + 127u) << 23) | (significand & 0x007fffffu);
}

uint32_t signed_to_float_bits(int32_t value) noexcept {
    if (value == 0) return 0;
    const bool negative = value < 0;
    const uint32_t magnitude = negative
                                   ? static_cast<uint32_t>(
                                         -static_cast<int64_t>(value))
                                   : static_cast<uint32_t>(value);
    return unsigned_to_float_bits(magnitude) |
           (negative ? 0x80000000u : 0u);
}

bool apply_float_policy(uint32_t bits, const PhysicalNumericPolicy& policy,
                        uint32_t* out) noexcept {
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t exponent = bits & 0x7f800000u;
    const uint32_t fraction = bits & 0x007fffffu;
    if (exponent == 0x7f800000u && fraction != 0) {
        if (policy.nan == PhysicalNanPolicy::Reject) return false;
        bits = 0x7fc00000u;
    } else if (exponent == 0x7f800000u) {
        if (policy.infinity == PhysicalInfinityPolicy::Reject) return false;
        if (policy.infinity == PhysicalInfinityPolicy::SaturateFinite)
            bits = sign | 0x7f7fffffu;
    } else if (exponent == 0 && fraction != 0 &&
               policy.subnormal ==
                   PhysicalSubnormalPolicy::FlushToSignedZero) {
        bits = sign;
    }
    if ((bits & 0x7fffffffu) == 0 &&
        policy.signed_zero == PhysicalSignedZeroPolicy::Positive)
        bits = 0;
    *out = bits;
    return true;
}

float float_from_bits(uint32_t bits) noexcept {
    return std::bit_cast<float>(bits);
}

uint32_t bits_from_float(float value) noexcept {
    return std::bit_cast<uint32_t>(value);
}

float separate_add(float left, float right) noexcept {
    volatile float a = left;
    volatile float b = right;
    volatile float result = a + b;
    return result;
}

float separate_subtract(float left, float right) noexcept {
    volatile float a = left;
    volatile float b = right;
    volatile float result = a - b;
    return result;
}

float separate_multiply(float left, float right) noexcept {
    volatile float a = left;
    volatile float b = right;
    volatile float result = a * b;
    return result;
}

double round_double(double value, PhysicalRoundingMode mode) noexcept {
    switch (mode) {
        case PhysicalRoundingMode::TowardZero: return std::trunc(value);
        case PhysicalRoundingMode::TowardPositive: return std::ceil(value);
        case PhysicalRoundingMode::TowardNegative: return std::floor(value);
        case PhysicalRoundingMode::NearestEven: {
            const double lower = std::floor(value);
            const double fraction = value - lower;
            if (fraction < 0.5) return lower;
            if (fraction > 0.5) return lower + 1.0;
            const double half = lower * 0.5;
            return half == std::floor(half) ? lower : lower + 1.0;
        }
    }
    return value;
}

bool float_to_integer(uint32_t bits, bool signed_result,
                      const PhysicalNumericPolicy& policy,
                      uint32_t* out) noexcept {
    const float value = float_from_bits(bits);
    if (std::isnan(value) || std::isinf(value)) return false;
    const double rounded = round_double(static_cast<double>(value),
                                        policy.rounding);
    const double minimum = signed_result
                               ? static_cast<double>(INT32_MIN)
                               : 0.0;
    const double maximum = signed_result
                               ? static_cast<double>(INT32_MAX)
                               : static_cast<double>(UINT32_MAX);
    if (!std::isfinite(rounded) || rounded < minimum || rounded > maximum) {
        if (policy.integer_overflow == PhysicalIntegerOverflow::Reject)
            return false;
        if (rounded < minimum)
            *out = signed_result ? static_cast<uint32_t>(INT32_MIN) : 0u;
        else
            *out = signed_result ? static_cast<uint32_t>(INT32_MAX)
                                 : UINT32_MAX;
        return true;
    }
    if (signed_result)
        *out = static_cast<uint32_t>(static_cast<int32_t>(rounded));
    else
        *out = static_cast<uint32_t>(rounded);
    return true;
}

bool load_bits(std::span<const uint8_t> bytes, uint64_t bit_address,
               uint8_t width, PhysicalBitOrder order, uint32_t* out) noexcept {
    uint64_t end = 0;
    uint64_t total_bits = 0;
    if (!out || width == 0 || width > 32 ||
        !multiply_u64(bytes.size(), 8, &total_bits) ||
        !add_u64(bit_address, width, &end) || end > total_bits)
        return false;
    uint32_t value = 0;
    if (order == PhysicalBitOrder::Lsb0Little) {
        for (uint8_t bit = 0; bit < width; ++bit) {
            const uint64_t address = bit_address + bit;
            value |= static_cast<uint32_t>(
                         (bytes[address / 8] >> (address % 8)) & 1u)
                     << bit;
        }
    } else if (order == PhysicalBitOrder::Msb0Big) {
        for (uint8_t bit = 0; bit < width; ++bit) {
            const uint64_t address = bit_address + bit;
            value = (value << 1u) |
                    ((bytes[address / 8] >> (7u - address % 8)) & 1u);
        }
    } else {
        return false;
    }
    *out = value;
    return true;
}

} // namespace

VerifiedPhysicalProgramResult verify_physical_program(
    PhysicalProgram program, std::span<const PhysicalPlaneBinding> bindings,
    const LogicalTensorType& logical_type) {
    try {
        const PhysicalProgram original = program;
        PhysicalProgramResult canonical_result =
            canonicalize_physical_program(std::move(program));
        if (const auto* report =
                std::get_if<CompatibilityReport>(&canonical_result))
            return *report;
        PhysicalProgram canonical =
            std::move(std::get<PhysicalProgram>(canonical_result));
        if (canonical != original)
            return physical_error(CompatibilityError::IR_CONSTRAINT_FAILED,
                                  "physical program must be canonical before binding");
        if (logical_type.extents.size() != canonical.logical_rank ||
            std::find(logical_type.extents.begin(), logical_type.extents.end(),
                      uint64_t{0}) != logical_type.extents.end())
            return physical_error(CompatibilityError::IR_SHAPE_MISMATCH,
                                  "physical logical tensor shape is invalid");
        const PhysicalValueType result_type =
            canonical.instructions[canonical.result].result_type;
        if ((result_type != PhysicalValueType::U32 &&
             result_type != PhysicalValueType::I32 &&
             result_type != PhysicalValueType::F32) ||
            (logical_type.element_type != ElementType::U32 &&
             logical_type.element_type != ElementType::I32 &&
             logical_type.element_type != ElementType::F32) ||
            logical_element_type(result_type) != logical_type.element_type)
            return physical_error(CompatibilityError::IR_SHAPE_MISMATCH,
                                  "physical result type does not match logical type");

        std::vector<PhysicalPlaneBinding> ordered(canonical.planes.size());
        std::vector<uint8_t> seen(canonical.planes.size(), 0);
        for (const PhysicalPlaneBinding& binding : bindings) {
            if (binding.plane >= canonical.planes.size() ||
                seen[binding.plane] || !binding.bytes || binding.length == 0 ||
                binding.offset > binding.bytes->size() ||
                binding.length > binding.bytes->size() - binding.offset)
                return physical_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                      "physical external plane binding is invalid");
            const PhysicalPlane& declaration = canonical.planes[binding.plane];
            if (declaration.storage != PhysicalPlaneStorage::External)
                return physical_error(CompatibilityError::IR_REFERENCE_INVALID,
                                      "inline plane has an external binding");
            const uintptr_t address = reinterpret_cast<uintptr_t>(
                binding.bytes->data() + binding.offset);
            if (address % declaration.alignment != 0)
                return physical_error(CompatibilityError::IR_CONSTRAINT_FAILED,
                                      "physical plane binding alignment is invalid");
            const size_t first = static_cast<size_t>(binding.offset);
            const size_t last = first + static_cast<size_t>(binding.length);
            auto snapshot = std::make_shared<std::vector<uint8_t>>(
                binding.bytes->begin() + first,
                binding.bytes->begin() + last);
            seen[binding.plane] = 1;
            ordered[binding.plane] = {
                binding.plane,
                std::shared_ptr<const std::vector<uint8_t>>(std::move(snapshot)),
                0,
                binding.length,
            };
        }
        std::vector<uint64_t> plane_bytes(canonical.planes.size(), 0);
        for (size_t plane = 0; plane < canonical.planes.size(); ++plane) {
            if (canonical.planes[plane].storage ==
                PhysicalPlaneStorage::External) {
                if (!seen[plane])
                    return physical_error(CompatibilityError::IR_REFERENCE_INVALID,
                                          "physical external plane is unbound");
                plane_bytes[plane] = ordered[plane].length;
            } else {
                if (seen[plane])
                    return physical_error(CompatibilityError::IR_REFERENCE_INVALID,
                                          "physical inline plane binding is duplicated");
                plane_bytes[plane] = canonical.planes[plane].byte_length;
            }
        }

        std::vector<IntegerRange> ranges;
        CompatibilityReport range_failure;
        if (!range_for_program(canonical, logical_type, plane_bytes, &ranges,
                               &range_failure))
            return range_failure;

        PhysicalProgramDigestResult digest_result =
            physical_program_digest(canonical);
        if (const auto* report =
                std::get_if<CompatibilityReport>(&digest_result))
            return *report;
        VerifiedPhysicalProgram verified;
        verified.program_ = std::move(canonical);
        verified.logical_ = logical_type;
        verified.digest_ = std::get<PhysicalProgramDigest>(digest_result);
        verified.bindings_ = std::move(ordered);
        return verified;
    } catch (const std::bad_alloc&) {
        return physical_error(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                              "physical verification exceeded memory limits");
    }
}

PhysicalInterpretResult interpret_physical_value(
    const VerifiedPhysicalProgram& verified,
    std::span<const uint64_t> logical_coordinate) {
    if (logical_coordinate.size() != verified.logical_.extents.size())
        return PhysicalInterpretError::CoordinateInvalid;
    for (size_t axis = 0; axis < logical_coordinate.size(); ++axis) {
        if (logical_coordinate[axis] >= verified.logical_.extents[axis])
            return PhysicalInterpretError::CoordinateInvalid;
    }
    ScopedNearestRounding rounding;
    if (!rounding.valid()) return PhysicalInterpretError::InternalInvariant;
    const PhysicalProgram& program = verified.program_;
    std::vector<RuntimeValue> values;
    try {
        values.reserve(program.instructions.size());
    } catch (const std::bad_alloc&) {
        return PhysicalInterpretError::InternalInvariant;
    }
    const auto operand = [&](const PhysicalInstruction& instruction,
                             size_t index) -> const RuntimeValue& {
        return values[instruction.operands[index]];
    };
    const auto plane_bytes = [&](uint16_t plane) -> std::span<const uint8_t> {
        const PhysicalPlane& declaration = program.planes[plane];
        if (declaration.storage == PhysicalPlaneStorage::Inline)
            return std::span<const uint8_t>(program.inline_bytes)
                .subspan(declaration.inline_offset, declaration.byte_length);
        const PhysicalPlaneBinding& binding = verified.bindings_[plane];
        return std::span<const uint8_t>(*binding.bytes)
            .subspan(binding.offset, binding.length);
    };

    for (const PhysicalInstruction& instruction : program.instructions) {
        RuntimeValue result{instruction.result_type, 0};
        uint64_t index_value = 0;
        uint32_t u32 = 0;
        const PhysicalNumericPolicy* policy =
            instruction.policy == kNoPhysicalPolicy
                ? nullptr
                : &program.policies[instruction.policy];
        switch (instruction.opcode) {
            case PhysicalOpcode::ConstIndex:
                result.bits = instruction.immediate;
                break;
            case PhysicalOpcode::ConstU32:
            case PhysicalOpcode::ConstI32:
            case PhysicalOpcode::ConstF32Bits:
                result.bits = static_cast<uint32_t>(instruction.immediate);
                break;
            case PhysicalOpcode::Coordinate:
                result.bits = logical_coordinate[instruction.immediate];
                break;
            case PhysicalOpcode::Extent:
                result.bits = verified.logical_.extents[instruction.immediate];
                break;
            case PhysicalOpcode::IndexAdd:
                if (!add_u64(operand(instruction, 0).bits,
                             operand(instruction, 1).bits, &result.bits))
                    return PhysicalInterpretError::ArithmeticOverflow;
                break;
            case PhysicalOpcode::IndexSubtract:
                if (operand(instruction, 0).bits < operand(instruction, 1).bits)
                    return PhysicalInterpretError::ArithmeticOverflow;
                result.bits = operand(instruction, 0).bits -
                              operand(instruction, 1).bits;
                break;
            case PhysicalOpcode::IndexMultiply:
                if (!multiply_u64(operand(instruction, 0).bits,
                                  operand(instruction, 1).bits, &result.bits))
                    return PhysicalInterpretError::ArithmeticOverflow;
                break;
            case PhysicalOpcode::IndexDivideConstant:
                result.bits = operand(instruction, 0).bits /
                              instruction.immediate;
                break;
            case PhysicalOpcode::IndexRemainderConstant:
                result.bits = operand(instruction, 0).bits %
                              instruction.immediate;
                break;
            case PhysicalOpcode::IndexCeilDivideConstant:
                index_value = operand(instruction, 0).bits;
                result.bits = index_value / instruction.immediate +
                              (index_value % instruction.immediate != 0);
                break;
            case PhysicalOpcode::IndexFromU32:
                result.bits = static_cast<uint32_t>(operand(instruction, 0).bits);
                break;
            case PhysicalOpcode::IndexLess:
                result.bits = operand(instruction, 0).bits <
                              operand(instruction, 1).bits;
                break;
            case PhysicalOpcode::Select:
                result.bits = operand(instruction, 0).bits
                                  ? operand(instruction, 1).bits
                                  : operand(instruction, 2).bits;
                break;
            case PhysicalOpcode::LoadBits:
                if (!load_bits(plane_bytes(instruction.plane),
                               operand(instruction, 0).bits,
                               instruction.bit_width, instruction.bit_order,
                               &u32))
                    return PhysicalInterpretError::AddressOutOfBounds;
                result.bits = u32;
                break;
            case PhysicalOpcode::U32And:
                result.bits = static_cast<uint32_t>(operand(instruction, 0).bits) &
                              static_cast<uint32_t>(operand(instruction, 1).bits);
                break;
            case PhysicalOpcode::U32Or:
                result.bits = static_cast<uint32_t>(operand(instruction, 0).bits) |
                              static_cast<uint32_t>(operand(instruction, 1).bits);
                break;
            case PhysicalOpcode::U32Xor:
                result.bits = static_cast<uint32_t>(operand(instruction, 0).bits) ^
                              static_cast<uint32_t>(operand(instruction, 1).bits);
                break;
            case PhysicalOpcode::U32Add:
                result.bits = static_cast<uint32_t>(
                    static_cast<uint32_t>(operand(instruction, 0).bits) +
                    static_cast<uint32_t>(operand(instruction, 1).bits));
                break;
            case PhysicalOpcode::U32Multiply:
                result.bits = static_cast<uint32_t>(
                    static_cast<uint32_t>(operand(instruction, 0).bits) *
                    static_cast<uint32_t>(operand(instruction, 1).bits));
                break;
            case PhysicalOpcode::U32ShiftLeftConstant:
                result.bits = static_cast<uint32_t>(operand(instruction, 0).bits)
                              << instruction.immediate;
                break;
            case PhysicalOpcode::U32ShiftRightConstant:
                result.bits = static_cast<uint32_t>(operand(instruction, 0).bits)
                              >> instruction.immediate;
                break;
            case PhysicalOpcode::U32FunnelShiftRight: {
                const uint64_t low = static_cast<uint32_t>(
                    operand(instruction, 0).bits);
                const uint64_t high = static_cast<uint32_t>(
                    operand(instruction, 1).bits);
                const uint64_t shift = operand(instruction, 2).bits;
                if (shift >= 64)
                    return PhysicalInterpretError::ArithmeticOverflow;
                result.bits = static_cast<uint32_t>(
                    ((high << 32) | low) >> shift);
                break;
            }
            case PhysicalOpcode::SignExtend: {
                const uint32_t width = instruction.bit_width;
                const uint32_t mask = width == 32
                                          ? UINT32_MAX
                                          : ((uint32_t{1} << width) - 1);
                uint32_t bits =
                    static_cast<uint32_t>(operand(instruction, 0).bits) & mask;
                const uint32_t sign = uint32_t{1} << (width - 1);
                if (bits & sign) bits |= ~mask;
                result.bits = bits;
                break;
            }
            case PhysicalOpcode::F16ToF32:
                u32 = half_to_float_bits(
                    static_cast<uint16_t>(operand(instruction, 0).bits));
                if (!apply_float_policy(u32, *policy, &u32))
                    return PhysicalInterpretError::NumericalPolicyRejected;
                result.bits = u32;
                break;
            case PhysicalOpcode::Bf16ToF32:
                u32 = static_cast<uint32_t>(
                          static_cast<uint16_t>(operand(instruction, 0).bits))
                      << 16;
                if (!apply_float_policy(u32, *policy, &u32))
                    return PhysicalInterpretError::NumericalPolicyRejected;
                result.bits = u32;
                break;
            case PhysicalOpcode::BitsToF32:
                u32 = static_cast<uint32_t>(operand(instruction, 0).bits);
                if (!apply_float_policy(u32, *policy, &u32))
                    return PhysicalInterpretError::NumericalPolicyRejected;
                result.bits = u32;
                break;
            case PhysicalOpcode::U32ToF32:
                u32 = unsigned_to_float_bits(
                    static_cast<uint32_t>(operand(instruction, 0).bits));
                if (!apply_float_policy(u32, *policy, &u32))
                    return PhysicalInterpretError::NumericalPolicyRejected;
                result.bits = u32;
                break;
            case PhysicalOpcode::I32ToF32:
                u32 = signed_to_float_bits(static_cast<int32_t>(
                    static_cast<uint32_t>(operand(instruction, 0).bits)));
                if (!apply_float_policy(u32, *policy, &u32))
                    return PhysicalInterpretError::NumericalPolicyRejected;
                result.bits = u32;
                break;
            case PhysicalOpcode::F32ToU32:
                if (!float_to_integer(
                        static_cast<uint32_t>(operand(instruction, 0).bits),
                        false, *policy, &u32))
                    return PhysicalInterpretError::NumericalPolicyRejected;
                result.bits = u32;
                break;
            case PhysicalOpcode::F32ToI32:
                if (!float_to_integer(
                        static_cast<uint32_t>(operand(instruction, 0).bits),
                        true, *policy, &u32))
                    return PhysicalInterpretError::NumericalPolicyRejected;
                result.bits = u32;
                break;
            case PhysicalOpcode::F32Add:
                u32 = bits_from_float(separate_add(
                    float_from_bits(static_cast<uint32_t>(
                        operand(instruction, 0).bits)),
                    float_from_bits(static_cast<uint32_t>(
                        operand(instruction, 1).bits))));
                if (!apply_float_policy(u32, *policy, &u32))
                    return PhysicalInterpretError::NumericalPolicyRejected;
                result.bits = u32;
                break;
            case PhysicalOpcode::F32Subtract:
                u32 = bits_from_float(separate_subtract(
                    float_from_bits(static_cast<uint32_t>(
                        operand(instruction, 0).bits)),
                    float_from_bits(static_cast<uint32_t>(
                        operand(instruction, 1).bits))));
                if (!apply_float_policy(u32, *policy, &u32))
                    return PhysicalInterpretError::NumericalPolicyRejected;
                result.bits = u32;
                break;
            case PhysicalOpcode::F32Multiply:
                u32 = bits_from_float(separate_multiply(
                    float_from_bits(static_cast<uint32_t>(
                        operand(instruction, 0).bits)),
                    float_from_bits(static_cast<uint32_t>(
                        operand(instruction, 1).bits))));
                if (!apply_float_policy(u32, *policy, &u32))
                    return PhysicalInterpretError::NumericalPolicyRejected;
                result.bits = u32;
                break;
            case PhysicalOpcode::F32Fma:
                u32 = bits_from_float(std::fma(
                    float_from_bits(static_cast<uint32_t>(
                        operand(instruction, 0).bits)),
                    float_from_bits(static_cast<uint32_t>(
                        operand(instruction, 1).bits)),
                    float_from_bits(static_cast<uint32_t>(
                        operand(instruction, 2).bits))));
                if (!apply_float_policy(u32, *policy, &u32))
                    return PhysicalInterpretError::NumericalPolicyRejected;
                result.bits = u32;
                break;
            case PhysicalOpcode::F32Negate:
                u32 = static_cast<uint32_t>(operand(instruction, 0).bits) ^
                      0x80000000u;
                if (!apply_float_policy(u32, *policy, &u32))
                    return PhysicalInterpretError::NumericalPolicyRejected;
                result.bits = u32;
                break;
            case PhysicalOpcode::F32RoundToF16:
                u32 = half_to_float_bits(float_to_half_nearest_even(
                    static_cast<uint32_t>(operand(instruction, 0).bits)));
                if (!apply_float_policy(u32, *policy, &u32))
                    return PhysicalInterpretError::NumericalPolicyRejected;
                result.bits = u32;
                break;
            case PhysicalOpcode::F32Clamp: {
                const float value = float_from_bits(static_cast<uint32_t>(
                    operand(instruction, 0).bits));
                const float lower = float_from_bits(static_cast<uint32_t>(
                    operand(instruction, 1).bits));
                const float upper = float_from_bits(static_cast<uint32_t>(
                    operand(instruction, 2).bits));
                float clamped = value;
                if (std::isnan(value) || std::isnan(lower) ||
                    std::isnan(upper)) {
                    clamped = std::numeric_limits<float>::quiet_NaN();
                } else {
                    if (lower > upper)
                        return PhysicalInterpretError::NumericalPolicyRejected;
                    clamped = std::max(lower, std::min(value, upper));
                }
                u32 = bits_from_float(clamped);
                if (!apply_float_policy(u32, *policy, &u32))
                    return PhysicalInterpretError::NumericalPolicyRejected;
                result.bits = u32;
                break;
            }
        }
        values.push_back(result);
    }

    const RuntimeValue& result = values[program.result];
    return ScalarValue{logical_element_type(result.type), result.bits};
}

std::span<const uint8_t> physical_program_plane_bytes(
    const VerifiedPhysicalProgram& verified, uint16_t plane) noexcept {
    if (plane >= verified.program_.planes.size()) return {};
    const PhysicalPlane& declaration = verified.program_.planes[plane];
    if (declaration.storage == PhysicalPlaneStorage::Inline) {
        if (declaration.inline_offset > verified.program_.inline_bytes.size() ||
            declaration.byte_length >
                verified.program_.inline_bytes.size() - declaration.inline_offset)
            return {};
        return std::span<const uint8_t>(verified.program_.inline_bytes)
            .subspan(static_cast<size_t>(declaration.inline_offset),
                     static_cast<size_t>(declaration.byte_length));
    }
    if (plane >= verified.bindings_.size()) return {};
    const PhysicalPlaneBinding& binding = verified.bindings_[plane];
    if (!binding.bytes || binding.offset > binding.bytes->size() ||
        binding.length > binding.bytes->size() - binding.offset)
        return {};
    return std::span<const uint8_t>(*binding.bytes)
        .subspan(static_cast<size_t>(binding.offset),
                 static_cast<size_t>(binding.length));
}

} // namespace Laplace
