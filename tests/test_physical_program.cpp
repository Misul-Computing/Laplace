#include "physical_program_interpreter.h"
#include "test_util.h"

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

using namespace Laplace;

static_assert(!std::is_default_constructible_v<VerifiedPhysicalProgram>);

namespace {

PhysicalPlane external_plane(uint32_t alignment = 1) {
    return {PhysicalPlaneStorage::External, alignment, 0, 0};
}

PhysicalInstruction instruction(
    PhysicalOpcode opcode, PhysicalValueType result,
    std::initializer_list<uint32_t> operands = {}, uint64_t immediate = 0,
    uint16_t plane = kNoPhysicalPlane, uint16_t policy = kNoPhysicalPolicy,
    uint8_t bit_width = 0,
    PhysicalBitOrder bit_order = PhysicalBitOrder::Lsb0Little) {
    PhysicalInstruction value;
    value.opcode = opcode;
    value.result_type = result;
    size_t index = 0;
    for (uint32_t operand : operands) value.operands[index++] = operand;
    value.immediate = immediate;
    value.plane = plane;
    value.policy = policy;
    value.bit_width = bit_width;
    value.bit_order = bit_order;
    return value;
}

std::shared_ptr<const std::vector<uint8_t>> bytes(
    std::initializer_list<uint8_t> values) {
    return std::make_shared<const std::vector<uint8_t>>(values);
}

std::shared_ptr<const std::vector<uint8_t>> u32_bytes(
    std::initializer_list<uint32_t> values) {
    std::vector<uint8_t> result;
    result.reserve(values.size() * 4);
    for (uint32_t value : values) {
        for (unsigned shift = 0; shift != 32; shift += 8)
            result.push_back(static_cast<uint8_t>(value >> shift));
    }
    return std::make_shared<const std::vector<uint8_t>>(std::move(result));
}

PhysicalProgram raw_load_program(
    uint8_t width, uint64_t address = 0,
    PhysicalBitOrder order = PhysicalBitOrder::Lsb0Little) {
    PhysicalProgram program;
    program.planes.push_back(external_plane());
    program.instructions.push_back(instruction(
        PhysicalOpcode::ConstIndex, PhysicalValueType::Index, {}, address));
    program.instructions.push_back(
        instruction(PhysicalOpcode::LoadBits, PhysicalValueType::U32, {0}, 0,
                    0, kNoPhysicalPolicy, width, order));
    program.result = 1;
    return program;
}

PhysicalNumericPolicy preserve_policy() { return {}; }

PhysicalProgram load_f32_program(uint16_t policy = 0) {
    PhysicalProgram program = raw_load_program(32);
    program.policies.push_back(preserve_policy());
    program.instructions.push_back(instruction(
        PhysicalOpcode::BitsToF32, PhysicalValueType::F32, {1}, 0,
        kNoPhysicalPlane, policy));
    program.result = 2;
    return program;
}

VerifiedPhysicalProgramResult verify_one(
    PhysicalProgram program,
    const std::shared_ptr<const std::vector<uint8_t>>& owner,
    const LogicalTensorType& logical, uint64_t offset = 0,
    uint64_t length = UINT64_MAX) {
    if (length == UINT64_MAX) length = owner->size() - offset;
    const PhysicalProgramResult canonical =
        canonicalize_physical_program(std::move(program));
    if (const auto* report = std::get_if<CompatibilityReport>(&canonical))
        return *report;
    const PhysicalPlaneBinding binding{0, owner, offset, length};
    return verify_physical_program(
        std::get<PhysicalProgram>(canonical),
        std::span<const PhysicalPlaneBinding>(&binding, 1),
        logical);
}

ScalarValue require_value(const VerifiedPhysicalProgramResult& verified,
                          std::span<const uint64_t> coordinate = {}) {
    if (!std::holds_alternative<VerifiedPhysicalProgram>(verified)) {
        const auto& report = std::get<CompatibilityReport>(verified);
        CHECK_MSG(false, "verification failed: code=%u detail=%s",
                  static_cast<unsigned>(report.code), report.detail.c_str());
        return {};
    }
    CHECK(true);
    const PhysicalInterpretResult interpreted = interpret_physical_value(
        std::get<VerifiedPhysicalProgram>(verified), coordinate);
    CHECK(std::holds_alternative<ScalarValue>(interpreted));
    if (!std::holds_alternative<ScalarValue>(interpreted)) return {};
    return std::get<ScalarValue>(interpreted);
}

void test_raw_unsigned_plane_and_wire() {
    const auto owner = bytes({0b11110101u});
    const auto verified = verify_one(raw_load_program(5), owner,
                                     {ElementType::U32, {}});
    const ScalarValue value = require_value(verified);
    CHECK(value.type == ElementType::U32);
    CHECK(value.bits == 21u);

    const auto encoded = encode_physical_program(raw_load_program(5));
    CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
    if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) return;
    const auto parsed = parse_physical_program(
        std::get<std::vector<uint8_t>>(encoded));
    CHECK(std::holds_alternative<PhysicalProgram>(parsed));
    if (std::holds_alternative<PhysicalProgram>(parsed))
        CHECK(std::get<PhysicalProgram>(parsed) == raw_load_program(5));
}

void test_cross_byte_sign_extension() {
    PhysicalProgram program = raw_load_program(5, 6);
    program.instructions.push_back(instruction(
        PhysicalOpcode::SignExtend, PhysicalValueType::I32, {1}, 0,
        kNoPhysicalPlane, kNoPhysicalPolicy, 5));
    program.result = 2;
    const ScalarValue value = require_value(
        verify_one(program, bytes({0x40, 0x07}), {ElementType::I32, {}}));
    CHECK(value.type == ElementType::I32);
    CHECK(static_cast<int32_t>(value.bits) == -3);

    PhysicalProgram msb = raw_load_program(5, 3, PhysicalBitOrder::Msb0Big);
    const ScalarValue msb_value = require_value(
        verify_one(msb, bytes({0b00010110u}), {ElementType::U32, {}}));
    CHECK(msb_value.bits == 0b10110u);
}

PhysicalProgram dynamic_codebook_program() {
    PhysicalProgram program;
    program.planes = {external_plane(), external_plane()};
    program.policies.push_back(preserve_policy());
    program.instructions = {
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index),
        instruction(PhysicalOpcode::LoadBits, PhysicalValueType::U32, {0}, 0,
                    0, kNoPhysicalPolicy, 2),
        instruction(PhysicalOpcode::IndexFromU32, PhysicalValueType::Index,
                    {1}),
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index, {},
                    32),
        instruction(PhysicalOpcode::IndexMultiply, PhysicalValueType::Index,
                    {2, 3}),
        instruction(PhysicalOpcode::LoadBits, PhysicalValueType::U32, {4}, 0,
                    1, kNoPhysicalPolicy, 32),
        instruction(PhysicalOpcode::BitsToF32, PhysicalValueType::F32, {5}, 0,
                    kNoPhysicalPlane, 0),
    };
    program.result = 6;
    return program;
}

void test_dynamic_codebook_plane() {
    const PhysicalProgram program = dynamic_codebook_program();
    const auto index = bytes({2});
    const auto table = u32_bytes(
        {std::bit_cast<uint32_t>(1.0f), std::bit_cast<uint32_t>(2.0f),
         std::bit_cast<uint32_t>(-3.5f), std::bit_cast<uint32_t>(4.0f)});
    const std::array<PhysicalPlaneBinding, 2> bindings = {
        PhysicalPlaneBinding{0, index, 0, index->size()},
        PhysicalPlaneBinding{1, table, 0, table->size()},
    };
    const auto verified = verify_physical_program(
        program, bindings, {ElementType::F32, {}});
    CHECK(require_value(verified).bits == std::bit_cast<uint32_t>(-3.5f));
}

PhysicalProgram funnel_shift_program() {
    PhysicalProgram program;
    program.logical_rank = 1;
    program.planes.push_back(external_plane());
    program.instructions = {
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index),
        instruction(PhysicalOpcode::LoadBits, PhysicalValueType::U32, {0}, 0,
                    0, kNoPhysicalPolicy, 32),
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index, {},
                    32),
        instruction(PhysicalOpcode::LoadBits, PhysicalValueType::U32, {2}, 0,
                    0, kNoPhysicalPolicy, 32),
        instruction(PhysicalOpcode::Coordinate, PhysicalValueType::Index),
        instruction(PhysicalOpcode::U32FunnelShiftRight,
                    PhysicalValueType::U32, {1, 3, 4}),
    };
    program.result = 5;
    return program;
}

void test_bounded_funnel_shift() {
    const uint32_t low = 0x89abcdefu;
    const uint32_t high = 0x01234567u;
    const auto owner = u32_bytes({low, high});
    const PhysicalProgram program = funnel_shift_program();
    const auto verified = verify_one(program, owner, {ElementType::U32, {64}});
    CHECK(std::holds_alternative<VerifiedPhysicalProgram>(verified));
    if (std::holds_alternative<VerifiedPhysicalProgram>(verified)) {
        const auto& value = std::get<VerifiedPhysicalProgram>(verified);
        const uint64_t joined = (uint64_t{high} << 32) | low;
        for (uint64_t shift : {uint64_t{0}, uint64_t{1}, uint64_t{31},
                               uint64_t{32}, uint64_t{47}, uint64_t{63}}) {
            const ScalarValue result = require_value(value, {&shift, 1});
            CHECK(result.bits == static_cast<uint32_t>(joined >> shift));
        }
    }

    const auto wire = encode_physical_program(program);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
    if (const auto* encoded = std::get_if<std::vector<uint8_t>>(&wire)) {
        const auto decoded = parse_physical_program(*encoded);
        CHECK(std::holds_alternative<PhysicalProgram>(decoded));
        if (const auto* parsed = std::get_if<PhysicalProgram>(&decoded))
            CHECK(*parsed == program);
    }

    CHECK(std::holds_alternative<CompatibilityReport>(
        verify_one(program, owner, {ElementType::U32, {65}})));
    PhysicalProgram wrong_shift = program;
    wrong_shift.instructions[4] = instruction(
        PhysicalOpcode::ConstU32, PhysicalValueType::U32, {}, 7);
    CHECK(std::holds_alternative<CompatibilityReport>(
        canonicalize_physical_program(std::move(wrong_shift))));
}

PhysicalProgram circular_window_program() {
    PhysicalProgram program;
    program.logical_rank = 1;
    program.planes.push_back(external_plane(4));
    program.instructions = {
        instruction(PhysicalOpcode::Coordinate, PhysicalValueType::Index),
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index, {}, 3),
        instruction(PhysicalOpcode::IndexMultiply, PhysicalValueType::Index,
                    {0, 1}),
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index, {},
                    755),
        instruction(PhysicalOpcode::IndexAdd, PhysicalValueType::Index, {2, 3}),
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index, {}, 16),
        instruction(PhysicalOpcode::IndexAdd, PhysicalValueType::Index, {4, 5}),
        instruction(PhysicalOpcode::IndexDivideConstant,
                    PhysicalValueType::Index, {4}, 32),
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index, {}, 1),
        instruction(PhysicalOpcode::IndexSubtract, PhysicalValueType::Index,
                    {6, 8}),
        instruction(PhysicalOpcode::IndexDivideConstant,
                    PhysicalValueType::Index, {9}, 32),
        instruction(PhysicalOpcode::IndexRemainderConstant,
                    PhysicalValueType::Index, {7}, 24),
        instruction(PhysicalOpcode::IndexRemainderConstant,
                    PhysicalValueType::Index, {10}, 24),
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index, {}, 32),
        instruction(PhysicalOpcode::IndexMultiply, PhysicalValueType::Index,
                    {11, 13}),
        instruction(PhysicalOpcode::IndexMultiply, PhysicalValueType::Index,
                    {12, 13}),
        instruction(PhysicalOpcode::LoadBits, PhysicalValueType::U32, {14}, 0,
                    0, kNoPhysicalPolicy, 32),
        instruction(PhysicalOpcode::LoadBits, PhysicalValueType::U32, {15}, 0,
                    0, kNoPhysicalPolicy, 32),
        instruction(PhysicalOpcode::IndexRemainderConstant,
                    PhysicalValueType::Index, {6}, 32),
        instruction(PhysicalOpcode::IndexSubtract, PhysicalValueType::Index,
                    {13, 18}),
        instruction(PhysicalOpcode::IndexRemainderConstant,
                    PhysicalValueType::Index, {19}, 32),
        instruction(PhysicalOpcode::U32FunnelShiftRight,
                    PhysicalValueType::U32, {17, 16, 20}),
        instruction(PhysicalOpcode::ConstU32, PhysicalValueType::U32, {},
                    0xffff),
        instruction(PhysicalOpcode::U32And, PhysicalValueType::U32, {21, 22}),
    };
    program.result = 23;
    return program;
}

void test_circular_packed_window() {
    std::vector<uint32_t> words(24);
    for (uint32_t index = 0; index < words.size(); ++index)
        words[index] = 0x9e3779b9u * (index + 1u) ^ 0xa5a5a5a5u;
    std::vector<uint8_t> storage;
    for (uint32_t word : words) {
        for (unsigned shift = 0; shift != 32; shift += 8)
            storage.push_back(static_cast<uint8_t>(word >> shift));
    }
    const auto owner =
        std::make_shared<const std::vector<uint8_t>>(std::move(storage));
    const auto verified = verify_one(circular_window_program(), owner,
                                     {ElementType::U32, {256}});
    CHECK(std::holds_alternative<VerifiedPhysicalProgram>(verified));
    if (!std::holds_alternative<VerifiedPhysicalProgram>(verified)) return;
    const auto& program = std::get<VerifiedPhysicalProgram>(verified);
    for (uint64_t coordinate = 0; coordinate < 256; ++coordinate) {
        const uint64_t begin = coordinate * 3 + 755;
        const uint64_t end = begin + 16;
        const uint64_t first = begin / 32;
        const uint64_t last = (end - 1) / 32;
        const uint64_t shift = ((last + 1) * 32 - end) % 32;
        const uint64_t joined =
            (uint64_t{words[first % words.size()]} << 32) |
            words[last % words.size()];
        const uint32_t expected = static_cast<uint32_t>(joined >> shift) &
                                  0xffffu;
        CHECK(require_value(program, {&coordinate, 1}).bits == expected);
    }
}

PhysicalProgram procedural_codebook_program() {
    PhysicalProgram program;
    program.logical_rank = 1;
    program.planes.push_back(external_plane(2));
    PhysicalNumericPolicy fused = preserve_policy();
    fused.contraction = PhysicalContractionPolicy::Fused;
    program.policies = {preserve_policy(), fused};
    program.instructions = {
        instruction(PhysicalOpcode::Coordinate, PhysicalValueType::Index),
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index, {}, 16),
        instruction(PhysicalOpcode::IndexMultiply, PhysicalValueType::Index,
                    {0, 1}),
        instruction(PhysicalOpcode::LoadBits, PhysicalValueType::U32, {2}, 0,
                    0, kNoPhysicalPolicy, 16),
        instruction(PhysicalOpcode::ConstU32, PhysicalValueType::U32, {},
                    0x83dcd12d),
        instruction(PhysicalOpcode::U32Multiply, PhysicalValueType::U32,
                    {3, 4}),
        instruction(PhysicalOpcode::ConstU32, PhysicalValueType::U32, {}, 0xff),
        instruction(PhysicalOpcode::U32And, PhysicalValueType::U32, {5, 6}),
        instruction(PhysicalOpcode::U32ShiftRightConstant,
                    PhysicalValueType::U32, {5}, 8),
        instruction(PhysicalOpcode::U32And, PhysicalValueType::U32, {8, 6}),
        instruction(PhysicalOpcode::U32ShiftRightConstant,
                    PhysicalValueType::U32, {5}, 16),
        instruction(PhysicalOpcode::U32And, PhysicalValueType::U32, {10, 6}),
        instruction(PhysicalOpcode::U32ShiftRightConstant,
                    PhysicalValueType::U32, {5}, 24),
        instruction(PhysicalOpcode::U32And, PhysicalValueType::U32, {12, 6}),
        instruction(PhysicalOpcode::U32Add, PhysicalValueType::U32, {7, 9}),
        instruction(PhysicalOpcode::U32Add, PhysicalValueType::U32, {11, 13}),
        instruction(PhysicalOpcode::U32Add, PhysicalValueType::U32, {14, 15}),
        instruction(PhysicalOpcode::ConstU32, PhysicalValueType::U32, {}, 0x6400),
        instruction(PhysicalOpcode::U32Add, PhysicalValueType::U32, {16, 17}),
        instruction(PhysicalOpcode::U32ToF32, PhysicalValueType::F32, {18}, 0,
                    kNoPhysicalPlane, 0),
        instruction(PhysicalOpcode::F32RoundToF16,
                    PhysicalValueType::F32, {19}, 0, kNoPhysicalPlane, 0),
        instruction(PhysicalOpcode::ConstU32, PhysicalValueType::U32, {}, 0x1eee),
        instruction(PhysicalOpcode::F16ToF32, PhysicalValueType::F32, {21}, 0,
                    kNoPhysicalPlane, 0),
        instruction(PhysicalOpcode::ConstU32, PhysicalValueType::U32, {}, 0xc931),
        instruction(PhysicalOpcode::F16ToF32, PhysicalValueType::F32, {23}, 0,
                    kNoPhysicalPlane, 0),
        instruction(PhysicalOpcode::F32Fma, PhysicalValueType::F32,
                    {20, 22, 24}, 0, kNoPhysicalPlane, 1),
        instruction(PhysicalOpcode::F32RoundToF16,
                    PhysicalValueType::F32, {25}, 0, kNoPhysicalPlane, 0),
    };
    program.result = 26;
    return program;
}

void test_procedural_codebook_arithmetic() {
    const std::array<uint16_t, 7> indices = {
        0, 1, 2, 17, 0x1234, 0x8000, 0xffff};
    std::vector<uint8_t> storage;
    for (uint16_t value : indices) {
        storage.push_back(static_cast<uint8_t>(value));
        storage.push_back(static_cast<uint8_t>(value >> 8));
    }
    const auto owner =
        std::make_shared<const std::vector<uint8_t>>(std::move(storage));
    const auto verified = verify_one(procedural_codebook_program(), owner,
                                     {ElementType::F32, {indices.size()}});
    CHECK(std::holds_alternative<VerifiedPhysicalProgram>(verified));
    if (!std::holds_alternative<VerifiedPhysicalProgram>(verified)) return;
    const auto& program = std::get<VerifiedPhysicalProgram>(verified);
    const float inverse = static_cast<float>(std::bit_cast<_Float16>(uint16_t{0x1eee}));
    const float bias = static_cast<float>(std::bit_cast<_Float16>(uint16_t{0xc931}));
    for (uint64_t coordinate = 0; coordinate < indices.size(); ++coordinate) {
        const uint32_t product =
            static_cast<uint32_t>(indices[coordinate]) * 0x83dcd12du;
        const uint32_t sum = 0x6400u + (product & 0xffu) +
                             ((product >> 8) & 0xffu) +
                             ((product >> 16) & 0xffu) +
                             ((product >> 24) & 0xffu);
        const float rounded_sum = static_cast<float>(static_cast<_Float16>(sum));
        const _Float16 expected_half =
            static_cast<_Float16>(std::fma(rounded_sum, inverse, bias));
        const uint32_t expected = std::bit_cast<uint32_t>(
            static_cast<float>(expected_half));
        CHECK(require_value(program, {&coordinate, 1}).bits == expected);
    }

    PhysicalProgram invalid = procedural_codebook_program();
    invalid.policies[0].rounding = PhysicalRoundingMode::TowardZero;
    CHECK(std::holds_alternative<CompatibilityReport>(
        canonicalize_physical_program(std::move(invalid))));
}

void test_anonymous_plane_permutation() {
    const PhysicalProgram first = dynamic_codebook_program();
    PhysicalProgram permuted = first;
    permuted.instructions[1].plane = 1;
    permuted.instructions[5].plane = 0;
    const auto first_digest = physical_program_digest(first);
    const auto second_digest = physical_program_digest(permuted);
    CHECK(std::holds_alternative<PhysicalProgramDigest>(first_digest));
    CHECK(std::holds_alternative<PhysicalProgramDigest>(second_digest));
    if (std::holds_alternative<PhysicalProgramDigest>(first_digest) &&
        std::holds_alternative<PhysicalProgramDigest>(second_digest))
        CHECK(std::get<PhysicalProgramDigest>(first_digest) ==
              std::get<PhysicalProgramDigest>(second_digest));

    const PhysicalProgramResult canonical =
        canonicalize_physical_program(permuted);
    CHECK(std::holds_alternative<PhysicalProgram>(canonical));
    if (!std::holds_alternative<PhysicalProgram>(canonical)) return;
    const auto index = bytes({2});
    const auto table = u32_bytes(
        {std::bit_cast<uint32_t>(1.0f), std::bit_cast<uint32_t>(2.0f),
         std::bit_cast<uint32_t>(-3.5f), std::bit_cast<uint32_t>(4.0f)});
    const std::array<PhysicalPlaneBinding, 2> canonical_bindings = {
        PhysicalPlaneBinding{0, index, 0, index->size()},
        PhysicalPlaneBinding{1, table, 0, table->size()},
    };
    const auto verified = verify_physical_program(
        std::get<PhysicalProgram>(canonical), canonical_bindings,
        {ElementType::F32, {}});
    CHECK(require_value(verified).bits == std::bit_cast<uint32_t>(-3.5f));
}

void test_axis_permutation_and_unpadded_tail() {
    PhysicalProgram program;
    program.logical_rank = 2;
    program.planes.push_back(external_plane());
    program.instructions = {
        instruction(PhysicalOpcode::Coordinate, PhysicalValueType::Index, {}, 1),
        instruction(PhysicalOpcode::IndexDivideConstant,
                    PhysicalValueType::Index, {0}, 3),
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index, {}, 3),
        instruction(PhysicalOpcode::IndexMultiply, PhysicalValueType::Index,
                    {1, 2}),
        instruction(PhysicalOpcode::IndexRemainderConstant,
                    PhysicalValueType::Index, {0}, 3),
        instruction(PhysicalOpcode::IndexAdd, PhysicalValueType::Index, {3, 4}),
        instruction(PhysicalOpcode::Coordinate, PhysicalValueType::Index, {}, 0),
        instruction(PhysicalOpcode::Extent, PhysicalValueType::Index, {}, 0),
        instruction(PhysicalOpcode::IndexMultiply, PhysicalValueType::Index,
                    {5, 7}),
        instruction(PhysicalOpcode::IndexAdd, PhysicalValueType::Index, {8, 6}),
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index, {}, 8),
        instruction(PhysicalOpcode::IndexMultiply, PhysicalValueType::Index,
                    {9, 10}),
        instruction(PhysicalOpcode::LoadBits, PhysicalValueType::U32, {11}, 0,
                    0, kNoPhysicalPolicy, 8),
    };
    program.result = 12;
    const auto owner = bytes({10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
    const auto verified = verify_one(program, owner, {ElementType::U32, {2, 5}});
    const std::array<uint64_t, 2> coordinate = {1, 4};
    CHECK(require_value(verified, coordinate).bits == 19);
    if (std::holds_alternative<VerifiedPhysicalProgram>(verified)) {
        const auto outside = interpret_physical_value(
            std::get<VerifiedPhysicalProgram>(verified),
            std::array<uint64_t, 2>{1, 5});
        CHECK(std::holds_alternative<PhysicalInterpretError>(outside));
        if (std::holds_alternative<PhysicalInterpretError>(outside))
            CHECK(std::get<PhysicalInterpretError>(outside) ==
                  PhysicalInterpretError::CoordinateInvalid);
    }
}

void test_inline_plane_and_narrow_conversions() {
    PhysicalProgram program;
    program.planes.push_back({PhysicalPlaneStorage::Inline, 1, 0, 2});
    program.inline_bytes = {0x00, 0x3c};
    program.policies.push_back(preserve_policy());
    program.instructions = {
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index),
        instruction(PhysicalOpcode::LoadBits, PhysicalValueType::U32, {0}, 0,
                    0, kNoPhysicalPolicy, 16),
        instruction(PhysicalOpcode::F16ToF32, PhysicalValueType::F32, {1}, 0,
                    kNoPhysicalPlane, 0),
    };
    program.result = 2;
    const auto verified = verify_physical_program(
        program, {}, {ElementType::F32, {}});
    CHECK(require_value(verified).bits == std::bit_cast<uint32_t>(1.0f));

    // Narrow conversions explicitly consume the low 16 bits.
    PhysicalProgram high_bits = program;
    high_bits.planes[0].byte_length = 4;
    high_bits.inline_bytes = {0x00, 0x3c, 0x01, 0x00};
    high_bits.instructions[1].bit_width = 32;
    const auto high_verified = verify_physical_program(
        high_bits, {}, {ElementType::F32, {}});
    CHECK(require_value(high_verified).bits == std::bit_cast<uint32_t>(1.0f));
}

void test_numeric_policy_bits() {
    CHECK(require_value(verify_one(load_f32_program(),
                                   u32_bytes({0x7fa12345u}),
                                   {ElementType::F32, {}}))
              .bits == 0x7fc00000u);
    CHECK(require_value(verify_one(load_f32_program(),
                                   u32_bytes({0xffc54321u}),
                                   {ElementType::F32, {}}))
              .bits == 0x7fc00000u);
    CHECK(require_value(verify_one(load_f32_program(),
                                   u32_bytes({0x80000000u}),
                                   {ElementType::F32, {}}))
              .bits == 0x80000000u);

    PhysicalProgram positive_zero = load_f32_program();
    positive_zero.policies[0].signed_zero =
        PhysicalSignedZeroPolicy::Positive;
    CHECK(require_value(verify_one(positive_zero, u32_bytes({0x80000000u}),
                                   {ElementType::F32, {}}))
              .bits == 0u);

    PhysicalProgram flush = load_f32_program();
    flush.policies[0].subnormal =
        PhysicalSubnormalPolicy::FlushToSignedZero;
    CHECK(require_value(verify_one(flush, u32_bytes({0x80000001u}),
                                   {ElementType::F32, {}}))
              .bits == 0x80000000u);

    PhysicalProgram multiply;
    multiply.planes.push_back(external_plane());
    PhysicalNumericPolicy saturating = preserve_policy();
    saturating.infinity = PhysicalInfinityPolicy::SaturateFinite;
    multiply.policies.push_back(saturating);
    multiply.instructions = {
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index),
        instruction(PhysicalOpcode::LoadBits, PhysicalValueType::U32, {0}, 0,
                    0, kNoPhysicalPolicy, 32),
        instruction(PhysicalOpcode::BitsToF32, PhysicalValueType::F32, {1}, 0,
                    kNoPhysicalPlane, 0),
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index, {},
                    32),
        instruction(PhysicalOpcode::LoadBits, PhysicalValueType::U32, {3}, 0,
                    0, kNoPhysicalPolicy, 32),
        instruction(PhysicalOpcode::BitsToF32, PhysicalValueType::F32, {4}, 0,
                    kNoPhysicalPlane, 0),
        instruction(PhysicalOpcode::F32Multiply, PhysicalValueType::F32,
                    {2, 5}, 0, kNoPhysicalPlane, 0),
    };
    multiply.result = 6;
    const auto multiplied = require_value(verify_one(
        multiply, u32_bytes({0x7f7fffffu, std::bit_cast<uint32_t>(2.0f)}),
        {ElementType::F32, {}}));
    CHECK(multiplied.bits == 0x7f7fffffu);
}

PhysicalProgram binary_float_program(PhysicalOpcode opcode,
                                     PhysicalNumericPolicy policy) {
    PhysicalProgram program;
    program.planes.push_back(external_plane());
    program.policies.push_back(policy);
    program.instructions = {
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index),
        instruction(PhysicalOpcode::LoadBits, PhysicalValueType::U32, {0}, 0,
                    0, kNoPhysicalPolicy, 32),
        instruction(PhysicalOpcode::BitsToF32, PhysicalValueType::F32, {1}, 0,
                    kNoPhysicalPlane, 0),
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index, {},
                    32),
        instruction(PhysicalOpcode::LoadBits, PhysicalValueType::U32, {3}, 0,
                    0, kNoPhysicalPolicy, 32),
        instruction(PhysicalOpcode::BitsToF32, PhysicalValueType::F32, {4}, 0,
                    kNoPhysicalPlane, 0),
        instruction(opcode, PhysicalValueType::F32, {2, 5}, 0,
                    kNoPhysicalPlane, 0),
    };
    program.result = 6;
    return program;
}

void test_preserve_ieee_load_policy() {
    for (uint32_t bits : {0u, 0x80000000u, 1u, 0x807fffffu, 0x3f800000u,
                          0x7f7fffffu, 0x7f800000u, 0xff800000u,
                          0x7f800001u, 0xff800001u, 0x7fc12345u, 0xffffffffu}) {
        auto program = load_f32_program();
        program.policies[0].nan = PhysicalNanPolicy::PreserveIeee;
        const auto verified = verify_one(program, u32_bytes({bits}), {ElementType::F32, {}});
        CHECK(require_value(verified).bits == bits);
        if (!std::holds_alternative<VerifiedPhysicalProgram>(verified)) continue;
        const auto wire = encode_physical_program(std::get<VerifiedPhysicalProgram>(verified).program());
        CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
        if (const auto* bytes = std::get_if<std::vector<uint8_t>>(&wire))
            CHECK(std::holds_alternative<PhysicalProgram>(parse_physical_program(*bytes)));

        program.policies[0].subnormal = PhysicalSubnormalPolicy::FlushToSignedZero;
        program.policies[0].signed_zero = PhysicalSignedZeroPolicy::Positive;
        program.policies[0].infinity = PhysicalInfinityPolicy::SaturateFinite;
        uint32_t expected = bits;
        if ((bits & 0x7fffffffu) < 0x00800000u) expected = 0;
        if ((bits & 0x7fffffffu) == 0x7f800000u)
            expected = (bits & 0x80000000u) | 0x7f7fffffu;
        CHECK(require_value(verify_one(program, u32_bytes({bits}), {ElementType::F32, {}})).bits == expected);
    }
}

void test_numeric_arithmetic_edges() {
    const auto zero_add = verify_one(
        binary_float_program(PhysicalOpcode::F32Add, preserve_policy()),
        u32_bytes({0x80000000u, 0x00000000u}),
        {ElementType::F32, {}});
    CHECK(require_value(zero_add).bits == 0x00000000u);

    PhysicalProgram clamp;
    clamp.planes.push_back(external_plane());
    clamp.policies.push_back(preserve_policy());
    for (uint64_t address : {uint64_t{0}, uint64_t{32}, uint64_t{64}}) {
        const uint32_t address_id = clamp.instructions.size();
        clamp.instructions.push_back(instruction(
            PhysicalOpcode::ConstIndex, PhysicalValueType::Index, {}, address));
        clamp.instructions.push_back(instruction(
            PhysicalOpcode::LoadBits, PhysicalValueType::U32, {address_id}, 0,
            0, kNoPhysicalPolicy, 32));
        clamp.instructions.push_back(instruction(
            PhysicalOpcode::BitsToF32, PhysicalValueType::F32,
            {address_id + 1}, 0, kNoPhysicalPlane, 0));
    }
    clamp.instructions.push_back(instruction(
        PhysicalOpcode::F32Clamp, PhysicalValueType::F32, {2, 5, 8}, 0,
        kNoPhysicalPlane, 0));
    clamp.result = 9;
    const auto clamped = verify_one(
        clamp,
        u32_bytes({0x7fa12345u, std::bit_cast<uint32_t>(-1.0f),
                   std::bit_cast<uint32_t>(1.0f)}),
        {ElementType::F32, {}});
    CHECK(require_value(clamped).bits == 0x7fc00000u);

    PhysicalProgram invalid =
        binary_float_program(PhysicalOpcode::F32Add, preserve_policy());
    invalid.policies[0].integer_overflow =
        PhysicalIntegerOverflow::Saturate;
    CHECK(std::holds_alternative<CompatibilityReport>(
        canonicalize_physical_program(invalid)));
}

PhysicalProgram float_to_u32_program(PhysicalNumericPolicy policy) {
    PhysicalProgram program = load_f32_program();
    program.policies.push_back(policy);
    program.instructions.push_back(instruction(
        PhysicalOpcode::F32ToU32, PhysicalValueType::U32, {2}, 0,
        kNoPhysicalPlane, 1));
    program.result = 3;
    return program;
}

PhysicalProgram float_to_i32_program(PhysicalNumericPolicy policy) {
    PhysicalProgram program = load_f32_program();
    program.policies.push_back(policy);
    program.instructions.push_back(instruction(
        PhysicalOpcode::F32ToI32, PhysicalValueType::I32, {2}, 0,
        kNoPhysicalPlane, 1));
    program.result = 3;
    return program;
}

PhysicalProgram integer_to_float_program(bool signed_input) {
    PhysicalProgram program = raw_load_program(32);
    program.policies.push_back(preserve_policy());
    uint32_t input = 1;
    if (signed_input) {
        program.instructions.push_back(instruction(
            PhysicalOpcode::SignExtend, PhysicalValueType::I32, {1}, 0,
            kNoPhysicalPlane, kNoPhysicalPolicy, 32));
        input = 2;
    }
    program.instructions.push_back(instruction(
        signed_input ? PhysicalOpcode::I32ToF32
                     : PhysicalOpcode::U32ToF32,
        PhysicalValueType::F32, {input}, 0, kNoPhysicalPlane, 0));
    program.result = program.instructions.size() - 1;
    return program;
}

void test_rounding_saturation_and_rejection() {
    PhysicalNumericPolicy conversion = preserve_policy();
    conversion.nan = PhysicalNanPolicy::Reject;
    conversion.infinity = PhysicalInfinityPolicy::Reject;
    conversion.integer_overflow = PhysicalIntegerOverflow::Saturate;
    CHECK(require_value(verify_one(
                            float_to_u32_program(conversion),
                            u32_bytes({std::bit_cast<uint32_t>(2.5f)}),
                            {ElementType::U32, {}}))
              .bits == 2u);
    CHECK(require_value(verify_one(
                            float_to_u32_program(conversion),
                            u32_bytes({std::bit_cast<uint32_t>(3.5f)}),
                            {ElementType::U32, {}}))
              .bits == 4u);
    CHECK(require_value(verify_one(
                            float_to_u32_program(conversion),
                            u32_bytes({std::bit_cast<uint32_t>(-1.0f)}),
                            {ElementType::U32, {}}))
              .bits == 0u);

    const auto infinity = verify_one(float_to_u32_program(conversion),
                                     u32_bytes({0x7f800000u}),
                                     {ElementType::U32, {}});
    CHECK(std::holds_alternative<VerifiedPhysicalProgram>(infinity));
    if (std::holds_alternative<VerifiedPhysicalProgram>(infinity)) {
        const auto result = interpret_physical_value(
            std::get<VerifiedPhysicalProgram>(infinity), {});
        CHECK(std::holds_alternative<PhysicalInterpretError>(result));
        if (std::holds_alternative<PhysicalInterpretError>(result))
            CHECK(std::get<PhysicalInterpretError>(result) ==
                  PhysicalInterpretError::NumericalPolicyRejected);
    }
}

void test_conversion_contract() {
    CHECK(require_value(verify_one(integer_to_float_program(false),
                                   u32_bytes({16777217u}),
                                   {ElementType::F32, {}}))
              .bits == std::bit_cast<uint32_t>(16777216.0f));
    CHECK(require_value(verify_one(integer_to_float_program(true),
                                   u32_bytes({0x80000000u}),
                                   {ElementType::F32, {}}))
              .bits == std::bit_cast<uint32_t>(-2147483648.0f));

    PhysicalProgram bf16 = raw_load_program(16);
    bf16.policies.push_back(preserve_policy());
    bf16.instructions.push_back(instruction(
        PhysicalOpcode::Bf16ToF32, PhysicalValueType::F32, {1}, 0,
        kNoPhysicalPlane, 0));
    bf16.result = 2;
    CHECK(require_value(verify_one(bf16, bytes({0x80, 0xbf}),
                                   {ElementType::F32, {}}))
              .bits == std::bit_cast<uint32_t>(-1.0f));

    PhysicalNumericPolicy conversion = preserve_policy();
    conversion.nan = PhysicalNanPolicy::Reject;
    conversion.infinity = PhysicalInfinityPolicy::Reject;
    conversion.rounding = PhysicalRoundingMode::TowardZero;
    CHECK(static_cast<int32_t>(
              require_value(verify_one(
                                float_to_i32_program(conversion),
                                u32_bytes({std::bit_cast<uint32_t>(-2.9f)}),
                                {ElementType::I32, {}}))
                  .bits) == -2);
    conversion.rounding = PhysicalRoundingMode::TowardNegative;
    CHECK(static_cast<int32_t>(
              require_value(verify_one(
                                float_to_i32_program(conversion),
                                u32_bytes({std::bit_cast<uint32_t>(-2.1f)}),
                                {ElementType::I32, {}}))
                  .bits) == -3);
    conversion.rounding = PhysicalRoundingMode::NearestEven;
    CHECK(static_cast<int32_t>(
              require_value(verify_one(
                                float_to_i32_program(conversion),
                                u32_bytes({std::bit_cast<uint32_t>(-2.5f)}),
                                {ElementType::I32, {}}))
                  .bits) == -2);

    PhysicalProgram irrelevant = integer_to_float_program(false);
    irrelevant.policies[0].nan = PhysicalNanPolicy::Reject;
    CHECK(std::holds_alternative<CompatibilityReport>(
        canonicalize_physical_program(irrelevant)));
}

void test_explicit_rounding_environment() {
    const auto verified = verify_one(
        binary_float_program(PhysicalOpcode::F32Add, preserve_policy()),
        u32_bytes({std::bit_cast<uint32_t>(1.0f),
                   std::bit_cast<uint32_t>(0x1p-24f)}),
        {ElementType::F32, {}});
    CHECK(std::holds_alternative<VerifiedPhysicalProgram>(verified));
    if (!std::holds_alternative<VerifiedPhysicalProgram>(verified)) return;

    const int previous = std::fegetround();
    CHECK(previous != -1);
    CHECK(std::fesetround(FE_UPWARD) == 0);
    const PhysicalInterpretResult result = interpret_physical_value(
        std::get<VerifiedPhysicalProgram>(verified), {});
    CHECK(std::fegetround() == FE_UPWARD);
    CHECK(std::fesetround(previous) == 0);
    CHECK(std::holds_alternative<ScalarValue>(result));
    if (std::holds_alternative<ScalarValue>(result))
        CHECK(std::get<ScalarValue>(result).bits ==
              std::bit_cast<uint32_t>(1.0f));

#ifdef FE_DFL_DISABLE_DENORMS_ENV
    const auto subnormal_product = verify_one(
        binary_float_program(PhysicalOpcode::F32Multiply, preserve_policy()),
        u32_bytes({0x00800000u, 0x3f000000u}), {ElementType::F32, {}});
    CHECK(std::holds_alternative<VerifiedPhysicalProgram>(subnormal_product));
    fenv_t saved{};
    CHECK(std::fegetenv(&saved) == 0);
    CHECK(std::fesetenv(FE_DFL_DISABLE_DENORMS_ENV) == 0);
    PhysicalInterpretResult subnormal_result = PhysicalInterpretError::InternalInvariant;
    if (std::holds_alternative<VerifiedPhysicalProgram>(subnormal_product))
        subnormal_result = interpret_physical_value(
            std::get<VerifiedPhysicalProgram>(subnormal_product), {});
    CHECK(std::fesetenv(&saved) == 0);
    CHECK(std::holds_alternative<ScalarValue>(subnormal_result));
    if (std::holds_alternative<ScalarValue>(subnormal_result))
        CHECK(std::get<ScalarValue>(subnormal_result).bits == 0x00400000u);
#endif
}

void test_binding_and_address_rejection() {
    const auto four = bytes({1, 2, 3, 4});
    CHECK(std::holds_alternative<VerifiedPhysicalProgram>(
        verify_one(raw_load_program(32), four, {ElementType::U32, {}})));
    CHECK(std::holds_alternative<CompatibilityReport>(
        verify_one(raw_load_program(32), four, {ElementType::U32, {}}, 0, 3)));
    CHECK(std::holds_alternative<CompatibilityReport>(verify_physical_program(
        raw_load_program(8), {}, {ElementType::U32, {}})));

    const PhysicalPlaneBinding duplicate[] = {
        {0, four, 0, 4}, {0, four, 0, 4}};
    CHECK(std::holds_alternative<CompatibilityReport>(verify_physical_program(
        raw_load_program(8), duplicate, {ElementType::U32, {}})));

    PhysicalProgram aligned = raw_load_program(8);
    aligned.planes[0].alignment = 2;
    CHECK(std::holds_alternative<CompatibilityReport>(
        verify_one(aligned, four, {ElementType::U32, {}}, 1, 1)));

    PhysicalProgram inline_aligned;
    inline_aligned.planes.push_back(
        {PhysicalPlaneStorage::Inline, 2, 0, 1});
    inline_aligned.inline_bytes = {0x5a};
    inline_aligned.instructions = {
        instruction(PhysicalOpcode::ConstIndex, PhysicalValueType::Index),
        instruction(PhysicalOpcode::LoadBits, PhysicalValueType::U32, {0}, 0,
                    0, kNoPhysicalPolicy, 8),
    };
    inline_aligned.result = 1;
    CHECK(std::holds_alternative<CompatibilityReport>(
        canonicalize_physical_program(inline_aligned)));
    CHECK(std::holds_alternative<CompatibilityReport>(
        verify_one(raw_load_program(1, UINT64_MAX), four,
                   {ElementType::U32, {}})));
    CHECK(std::holds_alternative<VerifiedPhysicalProgram>(
        verify_one(raw_load_program(1, 31), four, {ElementType::U32, {}})));
    CHECK(std::holds_alternative<CompatibilityReport>(
        verify_one(raw_load_program(1, 32), four, {ElementType::U32, {}})));

    PhysicalProgram ranked = raw_load_program(8);
    ranked.logical_rank = 1;
    CHECK(std::holds_alternative<CompatibilityReport>(
        verify_one(ranked, four, {ElementType::U32, {0}})));
    CHECK(std::holds_alternative<CompatibilityReport>(
        verify_one(raw_load_program(8), four, {ElementType::F16, {}})));

    PhysicalProgram divide_zero = raw_load_program(8);
    divide_zero.instructions.insert(
        divide_zero.instructions.begin() + 1,
        instruction(PhysicalOpcode::IndexDivideConstant,
                    PhysicalValueType::Index, {0}, 0));
    divide_zero.instructions[2].operands[0] = 1;
    divide_zero.result = 2;
    CHECK(std::holds_alternative<CompatibilityReport>(
        canonicalize_physical_program(divide_zero)));

    PhysicalProgram wrong_type = raw_load_program(8);
    wrong_type.instructions[1].operands[0] = 1;
    CHECK(std::holds_alternative<CompatibilityReport>(
        canonicalize_physical_program(wrong_type)));

    PhysicalProgram zero_width = raw_load_program(8);
    zero_width.instructions[1].bit_width = 0;
    CHECK(std::holds_alternative<CompatibilityReport>(
        canonicalize_physical_program(zero_width)));

    PhysicalProgram shift = raw_load_program(8);
    shift.instructions.push_back(instruction(
        PhysicalOpcode::U32ShiftLeftConstant, PhysicalValueType::U32, {1}, 32));
    shift.result = 2;
    CHECK(std::holds_alternative<CompatibilityReport>(
        canonicalize_physical_program(shift)));
}

PhysicalProgram duplicate_constant_program(bool duplicate, uint64_t second) {
    PhysicalProgram program;
    program.planes.push_back(external_plane());
    program.instructions.push_back(instruction(
        PhysicalOpcode::ConstIndex, PhysicalValueType::Index, {}, 0));
    if (duplicate)
        program.instructions.push_back(instruction(
            PhysicalOpcode::ConstIndex, PhysicalValueType::Index, {}, second));
    const uint32_t right = duplicate ? 1 : 0;
    program.instructions.push_back(instruction(
        PhysicalOpcode::IndexAdd, PhysicalValueType::Index, {0, right}));
    program.instructions.push_back(
        instruction(PhysicalOpcode::LoadBits, PhysicalValueType::U32,
                    {duplicate ? 2u : 1u}, 0, 0, kNoPhysicalPolicy, 1));
    program.result = duplicate ? 3 : 2;
    return program;
}

void test_canonical_identity() {
    const auto one = physical_program_digest(
        duplicate_constant_program(false, 0));
    const auto two = physical_program_digest(
        duplicate_constant_program(true, 0));
    const auto changed = physical_program_digest(
        duplicate_constant_program(true, 1));
    CHECK(std::holds_alternative<PhysicalProgramDigest>(one));
    CHECK(std::holds_alternative<PhysicalProgramDigest>(two));
    CHECK(std::holds_alternative<PhysicalProgramDigest>(changed));
    if (std::holds_alternative<PhysicalProgramDigest>(one) &&
        std::holds_alternative<PhysicalProgramDigest>(two) &&
        std::holds_alternative<PhysicalProgramDigest>(changed)) {
        CHECK(std::get<PhysicalProgramDigest>(one) ==
              std::get<PhysicalProgramDigest>(two));
        CHECK(std::get<PhysicalProgramDigest>(one) !=
              std::get<PhysicalProgramDigest>(changed));
    }

    PhysicalProgram unreachable = raw_load_program(8);
    unreachable.instructions.push_back(instruction(
        PhysicalOpcode::ConstU32, PhysicalValueType::U32, {}, 9));
    CHECK(std::holds_alternative<CompatibilityReport>(
        encode_physical_program(unreachable)));

    PhysicalProgram one_policy =
        binary_float_program(PhysicalOpcode::F32Add, preserve_policy());
    PhysicalProgram two_policies = one_policy;
    two_policies.policies.push_back(preserve_policy());
    two_policies.instructions[5].policy = 1;
    const auto one_policy_digest = physical_program_digest(one_policy);
    const auto two_policy_digest = physical_program_digest(two_policies);
    CHECK(std::holds_alternative<PhysicalProgramDigest>(one_policy_digest));
    CHECK(std::holds_alternative<PhysicalProgramDigest>(two_policy_digest));
    if (std::holds_alternative<PhysicalProgramDigest>(one_policy_digest) &&
        std::holds_alternative<PhysicalProgramDigest>(two_policy_digest))
        CHECK(std::get<PhysicalProgramDigest>(one_policy_digest) ==
              std::get<PhysicalProgramDigest>(two_policy_digest));
}

uint16_t read_u16(const std::vector<uint8_t>& wire, size_t offset) {
    return static_cast<uint16_t>(wire[offset] | wire[offset + 1] << 8);
}

uint32_t read_u32(const std::vector<uint8_t>& wire, size_t offset) {
    uint32_t value = 0;
    for (unsigned byte = 0; byte != 4; ++byte)
        value |= static_cast<uint32_t>(wire[offset + byte]) << (byte * 8);
    return value;
}

size_t section_body(const std::vector<uint8_t>& wire, uint16_t wanted) {
    size_t offset = 16;
    while (offset + 6 <= wire.size()) {
        const uint16_t tag = read_u16(wire, offset);
        const uint32_t length = read_u32(wire, offset + 2);
        if (tag == wanted) return offset + 6;
        offset += 6 + length;
    }
    return wire.size();
}

void test_wire_rejection() {
    const auto encoded = encode_physical_program(raw_load_program(5));
    CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
    if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) return;
    const auto wire = std::get<std::vector<uint8_t>>(encoded);

    auto bad_magic = wire;
    bad_magic[0] ^= 1;
    CHECK(std::holds_alternative<CompatibilityReport>(
        parse_physical_program(bad_magic)));
    auto bad_total = wire;
    bad_total[12] ^= 1;
    CHECK(std::holds_alternative<CompatibilityReport>(
        parse_physical_program(bad_total)));
    auto bad_reserved = wire;
    bad_reserved[section_body(wire, 1) + 3] = 1;
    CHECK(std::holds_alternative<CompatibilityReport>(
        parse_physical_program(bad_reserved)));
    auto unknown_opcode = wire;
    const size_t instruction_body = section_body(wire, 4);
    unknown_opcode[instruction_body + 4] = 0xff;
    unknown_opcode[instruction_body + 5] = 0xff;
    CHECK(std::holds_alternative<CompatibilityReport>(
        parse_physical_program(unknown_opcode)));
    auto trailing = wire;
    trailing.push_back(0);
    CHECK(std::holds_alternative<CompatibilityReport>(
        parse_physical_program(trailing)));
}

void test_owner_lifetime() {
    VerifiedPhysicalProgramResult verified = [&] {
        auto local_owner = bytes({0x5a});
        return verify_one(raw_load_program(8), local_owner,
                          {ElementType::U32, {}});
    }();
    CHECK(require_value(verified).bits == 0x5a);
}

void test_verified_binding_snapshot() {
    auto mutable_owner = std::make_shared<std::vector<uint8_t>>(
        std::initializer_list<uint8_t>{0x5a});
    std::shared_ptr<const std::vector<uint8_t>> binding_owner = mutable_owner;
    const auto verified = verify_one(raw_load_program(8), binding_owner,
                                     {ElementType::U32, {}});
    CHECK(std::holds_alternative<VerifiedPhysicalProgram>(verified));
    mutable_owner->clear();
    mutable_owner->shrink_to_fit();
    CHECK(require_value(verified).bits == 0x5a);
}

class FixtureReader {
  public:
    explicit FixtureReader(std::span<const uint8_t> bytes) : bytes_(bytes) {}

    bool u8(uint8_t* out) {
        if (!out || position_ >= bytes_.size()) return false;
        *out = bytes_[position_++];
        return true;
    }
    bool u16(uint16_t* out) {
        uint8_t a = 0, b = 0;
        if (!u8(&a) || !u8(&b)) return false;
        *out = static_cast<uint16_t>(a | static_cast<uint16_t>(b) << 8);
        return true;
    }
    bool u32(uint32_t* out) {
        if (!out || remaining() < 4) return false;
        uint32_t value = 0;
        for (unsigned byte = 0; byte != 4; ++byte)
            value |= static_cast<uint32_t>(bytes_[position_++]) << (byte * 8);
        *out = value;
        return true;
    }
    bool u64(uint64_t* out) {
        if (!out || remaining() < 8) return false;
        uint64_t value = 0;
        for (unsigned byte = 0; byte != 8; ++byte)
            value |= static_cast<uint64_t>(bytes_[position_++]) << (byte * 8);
        *out = value;
        return true;
    }
    bool take(size_t count, std::span<const uint8_t>* out) {
        if (!out || count > remaining()) return false;
        *out = bytes_.subspan(position_, count);
        position_ += count;
        return true;
    }
    size_t remaining() const { return bytes_.size() - position_; }

  private:
    std::span<const uint8_t> bytes_;
    size_t position_ = 0;
};

int run_unseen_fixture(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    CHECK(stream.good());
    if (!stream) return test_summary("test_physical_program_unseen");
    const std::vector<uint8_t> fixture{
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>()};
    FixtureReader reader(fixture);
    std::span<const uint8_t> magic;
    CHECK(reader.take(8, &magic));
    const std::array<uint8_t, 8> expected_magic =
        {'L', 'A', 'P', 'F', 'X', '0', '0', '1'};
    CHECK(magic.size() == expected_magic.size() &&
          std::equal(magic.begin(), magic.end(), expected_magic.begin()));
    uint16_t version = 0;
    uint8_t element = 0, rank = 0;
    CHECK(reader.u16(&version) && version == 2);
    CHECK(reader.u8(&element));
    CHECK(reader.u8(&rank) && rank <= 8);
    LogicalTensorType logical;
    logical.element_type = static_cast<ElementType>(element);
    std::vector<uint64_t> coordinate(rank, 0);
    logical.extents.resize(rank);
    for (uint64_t& extent : logical.extents) CHECK(reader.u64(&extent));
    for (uint64_t& value : coordinate) CHECK(reader.u64(&value));
    uint32_t wire_size = 0, plane_count = 0;
    CHECK(reader.u32(&wire_size));
    std::span<const uint8_t> wire;
    CHECK(reader.take(wire_size, &wire));
    const PhysicalProgramResult parsed = parse_physical_program(wire);
    CHECK(std::holds_alternative<PhysicalProgram>(parsed));
    if (!std::holds_alternative<PhysicalProgram>(parsed))
        return test_summary("test_physical_program_unseen");
    CHECK(reader.u32(&plane_count) && plane_count <= 32);
    std::vector<std::shared_ptr<const std::vector<uint8_t>>> owners;
    std::vector<PhysicalPlaneBinding> bindings;
    owners.reserve(plane_count);
    bindings.reserve(plane_count);
    for (uint32_t index = 0; index < plane_count; ++index) {
        uint32_t plane = UINT32_MAX;
        uint64_t length = 0;
        std::span<const uint8_t> payload;
        CHECK(reader.u32(&plane));
        CHECK(reader.u64(&length));
        CHECK(reader.take(length, &payload));
        owners.push_back(std::make_shared<const std::vector<uint8_t>>(
            payload.begin(), payload.end()));
        bindings.push_back({plane, owners.back(), 0, length});
    }
    CHECK(reader.remaining() == 0);
    const auto verified = verify_physical_program(
        std::get<PhysicalProgram>(parsed), bindings, logical);
    const ScalarValue value = require_value(verified, coordinate);
    CHECK(value.type == logical.element_type);
    std::printf("physical_value=0x%016llx\n",
                static_cast<unsigned long long>(value.bits));
    return test_summary("test_physical_program_unseen");
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--fixture")
        return run_unseen_fixture(argv[2]);
    test_raw_unsigned_plane_and_wire();
    test_cross_byte_sign_extension();
    test_dynamic_codebook_plane();
    test_bounded_funnel_shift();
    test_circular_packed_window();
    test_procedural_codebook_arithmetic();
    test_anonymous_plane_permutation();
    test_axis_permutation_and_unpadded_tail();
    test_inline_plane_and_narrow_conversions();
    test_numeric_policy_bits();
    test_preserve_ieee_load_policy();
    test_numeric_arithmetic_edges();
    test_rounding_saturation_and_rejection();
    test_conversion_contract();
    test_explicit_rounding_environment();
    test_binding_and_address_rejection();
    test_canonical_identity();
    test_wire_rejection();
    test_owner_lifetime();
    test_verified_binding_snapshot();
    return test_summary("test_physical_program");
}
