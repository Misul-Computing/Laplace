#include <array>
#include <bit>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <variant>
#include <vector>

#include "codec_program.h"
#include "physical_codec.h"
#include "test_util.h"

using namespace Laplace;

namespace {

constexpr uint64_t kReferenceDecodeLimit = 1u << 24;

uint32_t oracle_f32_bits(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h >> 15u) & 1u;
    int32_t exponent = static_cast<int32_t>((static_cast<uint32_t>(h) >> 10u) & 0x1fu);
    uint32_t mantissa = static_cast<uint32_t>(h) & 0x3ffu;
    uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign << 31u;
        } else {
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1u;
                --exponent;
            }
            ++exponent;
            result = (sign << 31u) | (static_cast<uint32_t>(exponent + 112) << 23u) |
                     ((mantissa & 0x3ffu) << 13u);
        }
    } else if (exponent == 31) {
        result = (sign << 31u) | (0xffu << 23u) | (mantissa << 13u);
    } else {
        result = (sign << 31u) | ((static_cast<uint32_t>(exponent + 112)) << 23u) |
                 (mantissa << 13u);
    }
    return result;
}

float oracle_f16(uint16_t h) {
    return std::bit_cast<float>(oracle_f32_bits(h));
}

float oracle_mul(float left, float right) {
    volatile float result = left * right;
    return result;
}

float oracle_sub(float left, float right) {
    volatile float result = left - right;
    return result;
}

void oracle_scale_min(int index, const uint8_t* q, uint8_t* scale, uint8_t* minimum) {
    if (index < 4) {
        *scale = q[index] & 63u;
        *minimum = q[index + 4] & 63u;
    } else {
        *scale = static_cast<uint8_t>((q[index + 4] & 0xfu) |
                                      ((q[index - 4] >> 6u) << 4u));
        *minimum = static_cast<uint8_t>((q[index + 4] >> 4u) |
                                        ((q[index] >> 6u) << 4u));
    }
}

std::vector<float> oracle_q4(std::span<const uint8_t> bytes) {
    std::vector<float> result(256);
    const float d = oracle_f16(static_cast<uint16_t>(bytes[0]) |
                               (static_cast<uint16_t>(bytes[1]) << 8u));
    const float dmin = oracle_f16(static_cast<uint16_t>(bytes[2]) |
                                  (static_cast<uint16_t>(bytes[3]) << 8u));
    const uint8_t* q = bytes.data() + 16;
    int scale_index = 0;
    for (size_t base = 0; base < result.size(); base += 64) {
        uint8_t scale = 0;
        uint8_t minimum = 0;
        oracle_scale_min(scale_index + 0, bytes.data() + 4, &scale, &minimum);
        const float d1 = oracle_mul(d, static_cast<float>(scale));
        const float m1 = oracle_mul(dmin, static_cast<float>(minimum));
        oracle_scale_min(scale_index + 1, bytes.data() + 4, &scale, &minimum);
        const float d2 = oracle_mul(d, static_cast<float>(scale));
        const float m2 = oracle_mul(dmin, static_cast<float>(minimum));
        for (size_t i = 0; i < 32; ++i) {
            result[base + i] = oracle_sub(oracle_mul(d1, static_cast<float>(q[i] & 0xfu)), m1);
            result[base + 32 + i] = oracle_sub(oracle_mul(d2, static_cast<float>(q[i] >> 4u)), m2);
        }
        q += 32;
        scale_index += 2;
    }
    return result;
}

CodecPlaneBinding plane(std::span<const uint8_t> storage, uint64_t offset,
                        uint64_t length, uint64_t stride) {
    return CodecPlaneBinding{storage, offset, length, stride};
}

void check_exact(const std::vector<float>& actual, const std::vector<float>& expected) {
    CHECK(actual.size() == expected.size());
    if (actual.size() != expected.size()) return;
    for (size_t i = 0; i < actual.size(); ++i) {
        CHECK(std::bit_cast<uint32_t>(actual[i]) == std::bit_cast<uint32_t>(expected[i]));
    }
}

void test_contract_bytes_and_digest() {
    const CodecProgramRegistry first = make_application_codec_registry();
    const CodecProgramRegistry second = make_application_codec_registry();
    CHECK(first.programs().size() == 2);
    CHECK(first.programs().size() == second.programs().size());
    for (size_t i = 0; i < first.programs().size(); ++i) {
        CHECK(first.programs()[i].canonical_bytes().size() ==
              second.programs()[i].canonical_bytes().size());
        CHECK(first.programs()[i].canonical_bytes().size() == 0 ||
              std::memcmp(first.programs()[i].canonical_bytes().data(),
                          second.programs()[i].canonical_bytes().data(),
                          first.programs()[i].canonical_bytes().size()) == 0);
        CHECK(first.programs()[i].identity().contract_digest ==
              second.programs()[i].identity().contract_digest);
        CHECK(codec_program_digest(first.programs()[i].canonical_bytes()) ==
              first.programs()[i].identity().contract_digest);
    }
    // These are stable wire witnesses, not artifact or model identifiers.
    const std::array<uint8_t, 32> expected_first = {
        0x98, 0x9b, 0xce, 0x5e, 0x44, 0xd0, 0x52, 0x65,
        0x8e, 0xf5, 0x7e, 0x5b, 0x7e, 0xef, 0xc1, 0x19,
        0x1e, 0x08, 0xf0, 0xc3, 0xc3, 0x3e, 0x83, 0xf6,
        0x2a, 0xae, 0xcb, 0x6e, 0x97, 0x4b, 0xb9, 0x20};
    const std::array<uint8_t, 32> expected_second = {
        0x06, 0xb1, 0x30, 0xb8, 0xe5, 0x24, 0x34, 0x05,
        0xe6, 0xc1, 0x82, 0x5a, 0xa0, 0xea, 0x75, 0x9e,
        0x53, 0x06, 0x94, 0x8c, 0x0a, 0x85, 0xac, 0xdd,
        0xa1, 0x71, 0x26, 0x16, 0x1e, 0xfb, 0x83, 0x2b};
    CHECK(first.programs()[0].identity().contract_digest == expected_first);
    CHECK(first.programs()[1].identity().contract_digest == expected_second);
}

void test_resolution_and_digest_fail_closed() {
    const CodecProgramRegistry registry = make_application_codec_registry();
    const CodecProgramIdentity known = registry.programs()[0].identity();
    CHECK(registry.resolve(known) != nullptr);

    CodecProgramIdentity unknown = known;
    unknown.contract_digest[0] ^= 0x80u;
    CHECK(registry.resolve(unknown) == nullptr);
    CodecProgramDeclaration declaration{unknown, 1, {}};
    const auto result = registry.decode(declaration, kReferenceDecodeLimit);
    CHECK(std::holds_alternative<CodecProgramError>(result));
    if (const auto* error = std::get_if<CodecProgramError>(&result))
        CHECK(*error == CodecProgramError::UnknownIdentity);

    CodecProgramIdentity tampered = known;
    tampered.abi_version = static_cast<uint16_t>(known.abi_version + 1u);
    CHECK(registry.resolve(tampered) == nullptr);
}

void test_plane_validation() {
    const CodecProgramRegistry registry = make_application_codec_registry();
    const CodecProgramIdentity f16 = registry.programs()[0].identity();
    const std::array<uint8_t, 4> bytes = {0x00, 0x3c, 0x00, 0xc0};

    auto expect_error = [&](CodecProgramDeclaration declaration, CodecProgramError expected) {
        const auto result = registry.decode(declaration, kReferenceDecodeLimit);
        CHECK(std::holds_alternative<CodecProgramError>(result));
        if (const auto* error = std::get_if<CodecProgramError>(&result)) CHECK(*error == expected);
    };
    expect_error({f16, 2, {plane(bytes, 1, 3, 2)}}, CodecProgramError::AccessOutOfBounds);
    expect_error({f16, 2, {plane(bytes, 2, 3, 2)}}, CodecProgramError::InvalidLength);
    expect_error({f16, 2, {plane(bytes, 0, 4, 1)}}, CodecProgramError::InvalidStride);
    expect_error({f16, 2, {plane(bytes, 0, 4, std::numeric_limits<uint64_t>::max())}},
                 CodecProgramError::ArithmeticOverflow);
    expect_error({f16, 2, {plane(bytes, std::numeric_limits<uint64_t>::max(), 2, 2)}},
                 CodecProgramError::InvalidOffset);
    expect_error({f16, std::numeric_limits<uint64_t>::max(), {plane(bytes, 0, 4, 2)}},
                 CodecProgramError::InvalidElementCount);
    expect_error({f16, 1, {}}, CodecProgramError::InvalidPlaneCount);
    expect_error({f16, 1, {plane(bytes, 0, 2, 2), plane(bytes, 2, 2, 2)}},
                 CodecProgramError::InvalidPlaneCount);

    const std::span<const uint8_t> empty;
    expect_error({f16, 1, {plane(empty, 0, 0, 2)}}, CodecProgramError::AccessOutOfBounds);

    const CodecProgramDeclaration overflow{
        registry.programs()[1].identity(), std::numeric_limits<uint64_t>::max() - 255,
        {plane(empty, 0, 0, std::numeric_limits<uint64_t>::max())}};
    CHECK(registry.programs()[1].validate(overflow) == CodecProgramError::ArithmeticOverflow);
}

void test_successful_offset_padding_and_cap_policy() {
    const CodecProgramRegistry registry = make_application_codec_registry();
    const CodecProgramIdentity f16 = registry.programs()[0].identity();
    const std::array<uint8_t, 8> bytes = {
        0xa5, 0x00, 0x3c, 0xa5, 0x00, 0xc0, 0xa5, 0x00};
    const CodecProgramDeclaration declaration{
        f16, 2, {plane(bytes, 1, 5, 3)}};
    CHECK(registry.programs()[0].validate(declaration) == CodecProgramError::None);
    const auto result = registry.decode(declaration, 2);
    CHECK(std::holds_alternative<std::vector<float>>(result));
    if (const auto* values = std::get_if<std::vector<float>>(&result)) {
        CHECK(values->size() == 2);
        if (values->size() == 2) {
            CHECK(std::bit_cast<uint32_t>((*values)[0]) == 0x3f800000u);
            CHECK(std::bit_cast<uint32_t>((*values)[1]) == 0xc0000000u);
        }
    }
    const auto capped = registry.decode(declaration, 1);
    CHECK(std::holds_alternative<CodecProgramError>(capped));
    if (const auto* error = std::get_if<CodecProgramError>(&capped))
        CHECK(*error == CodecProgramError::InvalidElementCount);
}

void test_canonical_program_is_executable_authority() {
    CodecProgramRegistry registry = make_application_codec_registry();
    CodecProgram& f16 = const_cast<CodecProgram&>(registry.programs()[0]);
    uint8_t* canonical = const_cast<uint8_t*>(f16.canonical_bytes().data());
    canonical[12] = 2; // Change only the canonical algorithm field.
    const std::array<uint8_t, 2> bytes = {0x00, 0x3c};
    const CodecProgramDeclaration declaration{
        f16.identity(), 1, {plane(bytes, 0, bytes.size(), 2)}};
    CHECK(f16.validate(declaration) == CodecProgramError::InvalidContract);
    CHECK(registry.resolve(f16.identity()) == nullptr);
}

void test_validate_rejects_structurally_valid_digest_drift() {
    CodecProgramRegistry registry = make_application_codec_registry();
    CodecProgram& q4 = const_cast<CodecProgram&>(registry.programs()[1]);
    uint8_t* canonical = const_cast<uint8_t*>(q4.canonical_bytes().data());
    canonical[384] = 14; // First Q4 nibble mask: still parseable, but not the identity's bytes.
    const std::array<uint8_t, 144> bytes{};
    const CodecProgramDeclaration declaration{
        q4.identity(), 256, {plane(bytes, 0, bytes.size(), bytes.size())}};
    CHECK(q4.validate(declaration) == CodecProgramError::ContractDigestMismatch);
}

void test_physical_canonical_schema_is_bounded() {
    CodecProgramRegistry registry = make_application_codec_registry();
    CodecProgram& f16 = const_cast<CodecProgram&>(registry.programs()[0]);
    uint8_t* canonical = const_cast<uint8_t*>(f16.canonical_bytes().data());
    canonical[96] = 1; // F16 physical block bytes must remain zero.
    const std::array<uint8_t, 2> bytes = {0x00, 0x3c};
    const CodecProgramDeclaration declaration{
        f16.identity(), 1, {plane(bytes, 0, bytes.size(), 2)}};
    CHECK(f16.validate(declaration) == CodecProgramError::InvalidContract);
    CHECK(registry.resolve(f16.identity()) == nullptr);

    registry = make_application_codec_registry();
    CodecProgram& bounded = const_cast<CodecProgram&>(registry.programs()[0]);
    canonical = const_cast<uint8_t*>(bounded.canonical_bytes().data());
    canonical[136] = 7; // More than the six semantic plane kinds.
    CHECK(bounded.validate(declaration) == CodecProgramError::InvalidContract);
    CHECK(registry.resolve(bounded.identity()) == nullptr);
}

void test_malformed_canonical_is_rejected() {
    CodecProgramRegistry registry = make_application_codec_registry();
    CodecProgram& f16 = const_cast<CodecProgram&>(registry.programs()[0]);
    uint8_t* canonical = const_cast<uint8_t*>(f16.canonical_bytes().data());
    canonical[0] ^= 0xffu; // Invalid magic; the bounded parser must fail closed.
    const std::array<uint8_t, 2> bytes = {0x00, 0x3c};
    const CodecProgramDeclaration declaration{
        f16.identity(), 1, {plane(bytes, 0, bytes.size(), 2)}};
    CHECK(f16.validate(declaration) == CodecProgramError::InvalidContract);
    CHECK(registry.resolve(f16.identity()) == nullptr);
}

PhysicalCodecIdentity f16_physical_identity(const CodecProgram& program) {
    PhysicalCodecIdentity identity;
    identity.arithmetic_version = 1;
    identity.arithmetic_digest = program.identity().contract_digest;
    identity.layout.kind = PhysicalLayoutKind::ContiguousRowMajor;
    identity.layout.version = 1;
    identity.layout.packing = PackingKind::None;
    identity.layout.rank = 1;
    identity.layout.block_rank = 0;
    identity.layout.axis_order = {0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    identity.quantization.kind = QuantizationKind::None;
    identity.quantization.version = 1;
    identity.quantization.accumulation_type = ScalarType::F32;
    identity.planes = {{PlaneKind::Values, ScalarType::F16, 1, 2, 0}};
    return identity;
}

PhysicalCodecIdentity q4_physical_identity(const CodecProgram& program) {
    PhysicalCodecIdentity identity;
    identity.arithmetic_version = 1;
    identity.arithmetic_digest = program.identity().contract_digest;
    identity.layout.kind = PhysicalLayoutKind::GgufBlocked;
    identity.layout.version = 1;
    identity.layout.packing = PackingKind::Gguf;
    identity.layout.rank = 2;
    identity.layout.block_rank = 1;
    identity.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    identity.layout.block_elements = 256;
    identity.layout.block_bytes = 144;
    identity.quantization.kind = QuantizationKind::BlockedAffine;
    identity.quantization.version = 1;
    identity.quantization.accumulation_type = ScalarType::F32;
    identity.quantization.scale_type = ScalarType::F16;
    identity.quantization.zero_type = ScalarType::F16;
    identity.quantization.block_elements = 256;
    identity.quantization.block_bytes = 144;
    identity.quantization.group_size = 256;
    identity.quantization.required_plane_mask = 1;
    identity.planes = {{PlaneKind::Values, ScalarType::U8, 256, 144, 0}};
    return identity;
}

void test_physical_identity_binding() {
    const CodecProgramRegistry registry = make_application_codec_registry();
    const CodecProgram& f16 = registry.programs()[0];
    const CodecProgram& q4 = registry.programs()[1];
    CHECK(f16.matches_physical_identity(f16_physical_identity(f16)));
    CHECK(q4.matches_physical_identity(q4_physical_identity(q4)));

    PhysicalCodecIdentity different_arithmetic = f16_physical_identity(f16);
    different_arithmetic.arithmetic_digest[0] ^= 1u;
    CHECK(!f16.matches_physical_identity(different_arithmetic));

    PhysicalCodecIdentity different_layout = q4_physical_identity(q4);
    different_layout.layout.block_bytes = 143;
    CHECK(!q4.matches_physical_identity(different_layout));
    different_layout = q4_physical_identity(q4);
    different_layout.layout.packing = PackingKind::None;
    CHECK(!q4.matches_physical_identity(different_layout));
    different_layout = q4_physical_identity(q4);
    different_layout.planes[0].bytes_per_block = 143;
    CHECK(!q4.matches_physical_identity(different_layout));
}

void test_rounding_environment_is_fail_closed() {
    const CodecProgramRegistry registry = make_application_codec_registry();
    const std::array<uint8_t, 2> bytes = {0x00, 0x3c};
    const CodecProgramDeclaration declaration{
        registry.programs()[0].identity(), 1, {plane(bytes, 0, bytes.size(), 2)}};
    const int previous = std::fegetround();
    CHECK(previous != -1);
    if (previous == -1) return;
    const int changed = std::fesetround(FE_DOWNWARD);
    CHECK(changed == 0);
    if (changed == 0) {
        CHECK(registry.programs()[0].validate(declaration) == CodecProgramError::None);
        const auto result = registry.decode(declaration, 1);
        CHECK(std::holds_alternative<CodecProgramError>(result));
        if (const auto* error = std::get_if<CodecProgramError>(&result))
            CHECK(*error == CodecProgramError::RoundingModeUnavailable);
    }
    CHECK(std::fesetround(previous) == 0);
}

void test_validation_binds_identity() {
    const CodecProgramRegistry registry = make_application_codec_registry();
    const CodecProgram& f16 = registry.programs()[0];
    const CodecProgramIdentity q4_identity = registry.programs()[1].identity();
    const std::array<uint8_t, 512> bytes{};
    const CodecProgramDeclaration cross_program{
        q4_identity, 256, {plane(bytes, 0, bytes.size(), 2)}};
    CHECK(f16.validate(cross_program) == CodecProgramError::UnknownIdentity);

    CodecProgramIdentity unknown = f16.identity();
    unknown.contract_digest[0] ^= 0x80u;
    const CodecProgramDeclaration unknown_program{
        unknown, 2, {plane(bytes, 0, 4, 2)}};
    CHECK(f16.validate(unknown_program) == CodecProgramError::UnknownIdentity);
}

void test_real_size_nonallocating_validation() {
    const CodecProgramRegistry registry = make_application_codec_registry();
    const CodecProgram& q4 = registry.programs()[1];
    constexpr uint64_t elements = 5120ull * 17408ull;
    constexpr uint64_t blocks = elements / 256ull;
    constexpr uint64_t bytes_required = blocks * 144ull;
    std::vector<uint8_t> storage(static_cast<size_t>(bytes_required));
    const CodecProgramDeclaration declaration{
        q4.identity(), elements,
        {plane(storage, 0, bytes_required, 144)}};
    CHECK(q4.validate(declaration) == CodecProgramError::None);
}

void test_f16_exact_oracle() {
    const CodecProgramRegistry registry = make_application_codec_registry();
    std::array<uint8_t, 12> bytes = {
        0x00, 0x3c, 0x00, 0xc0, 0x01, 0x00,
        0x00, 0x7c, 0x00, 0xfc, 0x00, 0x7e};
    CodecProgramDeclaration declaration{
        registry.programs()[0].identity(), 6,
        {plane(bytes, 0, bytes.size(), 2)}};
    const auto result = registry.decode(declaration, kReferenceDecodeLimit);
    CHECK(std::holds_alternative<std::vector<float>>(result));
    if (const auto* values = std::get_if<std::vector<float>>(&result)) {
        std::vector<float> expected;
        for (size_t i = 0; i < bytes.size(); i += 2) {
            expected.push_back(std::bit_cast<float>(oracle_f32_bits(
                static_cast<uint16_t>(bytes[i]) |
                (static_cast<uint16_t>(bytes[i + 1]) << 8u))));
        }
        check_exact(*values, expected);
    }
}

void test_f16_all_bit_patterns() {
    const CodecProgramRegistry registry = make_application_codec_registry();
    std::vector<uint8_t> bytes(65536u * 2u);
    for (uint32_t value = 0; value != 65536u; ++value) {
        bytes[value * 2u] = static_cast<uint8_t>(value);
        bytes[value * 2u + 1u] = static_cast<uint8_t>(value >> 8u);
    }
    const CodecProgramDeclaration declaration{
        registry.programs()[0].identity(), 65536,
        {plane(bytes, 0, bytes.size(), 2)}};
    const auto result = registry.decode(declaration, 65536);
    CHECK(std::holds_alternative<std::vector<float>>(result));
    if (const auto* values = std::get_if<std::vector<float>>(&result)) {
        CHECK(values->size() == 65536);
        bool exact = values->size() == 65536;
        for (uint32_t value = 0; exact && value != 65536u; ++value) {
            const uint16_t h = static_cast<uint16_t>(value);
            exact = std::bit_cast<uint32_t>((*values)[value]) == oracle_f32_bits(h);
        }
        CHECK(exact);
    }
}

void test_q4_exact_oracle() {
    const CodecProgramRegistry registry = make_application_codec_registry();
    std::array<uint8_t, 144> bytes{};
    bytes[0] = 0x55; bytes[1] = 0x35; // non-integral binary16 scale
    bytes[2] = 0x66; bytes[3] = 0x2e; // non-integral binary16 minimum scale
    for (size_t i = 0; i < 12; ++i) bytes[4 + i] = static_cast<uint8_t>(i + 1);
    for (size_t i = 0; i < 128; ++i) bytes[16 + i] = static_cast<uint8_t>(i * 37u + 0x13u);
    CodecProgramDeclaration declaration{
        registry.programs()[1].identity(), 256,
        {plane(bytes, 0, bytes.size(), bytes.size())}};
    const auto result = registry.decode(declaration, kReferenceDecodeLimit);
    CHECK(std::holds_alternative<std::vector<float>>(result));
    if (const auto* values = std::get_if<std::vector<float>>(&result)) {
        check_exact(*values, oracle_q4(bytes));
    }
}

void fill_q4_block(std::span<uint8_t> bytes, uint16_t d, uint16_t dmin,
                   std::span<const uint8_t, 12> scales, uint32_t multiplier,
                   uint8_t addend) {
    bytes[0] = static_cast<uint8_t>(d);
    bytes[1] = static_cast<uint8_t>(d >> 8u);
    bytes[2] = static_cast<uint8_t>(dmin);
    bytes[3] = static_cast<uint8_t>(dmin >> 8u);
    std::memcpy(bytes.data() + 4, scales.data(), scales.size());
    for (uint32_t i = 0; i != 128; ++i)
        bytes[16 + i] = static_cast<uint8_t>(i * multiplier + addend);
}

void test_q4_high_bits_and_padded_stride() {
    const CodecProgramRegistry registry = make_application_codec_registry();
    std::array<uint8_t, 320> bytes{};
    constexpr std::array<uint8_t, 12> scales0 = {
        0xc1, 0x82, 0x43, 0xf4, 0xd5, 0xa6, 0x67, 0xe8,
        0xb9, 0x9a, 0x5b, 0x7c};
    constexpr std::array<uint8_t, 12> scales1 = {
        0xc2, 0x83, 0x44, 0xf5, 0xd6, 0xa7, 0x68, 0xe9,
        0xba, 0x9b, 0x5c, 0x7d};
    fill_q4_block(std::span<uint8_t>(bytes).subspan(0, 144), 0x3555, 0x2e66,
                  scales0, 37, 0x13);
    fill_q4_block(std::span<uint8_t>(bytes).subspan(160, 144), 0x3c00, 0x3800,
                  scales1, 53, 0xa7);

    // Boundary values for every 64-value output group, both nibble halves,
    // independently pinned from ggml-org/ggml at 36da57138425487184aa1da2eee2cde155909c6f.
    constexpr std::array<uint32_t, 32> expected_boundary_bits = {
        0xbf8cc400u, 0x40243a00u, 0xc0487c00u, 0x3fc43800u,
        0xbf665800u, 0x41218f80u, 0x433a9f00u, 0x41f54600u,
        0x424c59a0u, 0x438204acu, 0x4283c490u, 0x4323dc28u,
        0x41c726c0u, 0x42f7bd50u, 0x439336ccu, 0x42e4f1b0u,
        0x40400000u, 0xc0e00000u, 0x41280000u, 0xc1840000u,
        0x41000000u, 0xc1400000u, 0x433f8000u, 0x440ca000u,
        0x43bc4000u, 0x42ad0000u, 0x44116000u, 0x43428000u,
        0x43398000u, 0x42360000u, 0x43e64000u, 0x445de000u};
    const CodecProgramDeclaration declaration{
        registry.programs()[1].identity(), 512,
        {plane(bytes, 0, 304, 160)}};
    const auto result = registry.decode(declaration, 512);
    CHECK(std::holds_alternative<std::vector<float>>(result));
    if (const auto* values = std::get_if<std::vector<float>>(&result)) {
        CHECK(values->size() == 512);
        size_t expected_index = 0;
        for (size_t block = 0; block != 2; ++block) {
            for (size_t chunk = 0; chunk != 4; ++chunk) {
                const size_t output = block * 256 + chunk * 64;
                for (size_t index : {output, output + 31, output + 32, output + 63}) {
                    CHECK(std::bit_cast<uint32_t>((*values)[index]) ==
                          expected_boundary_bits[expected_index++]);
                }
            }
        }
        const auto first_expected = oracle_q4(
            std::span<const uint8_t>(bytes).subspan(0, 144));
        const auto second_expected = oracle_q4(
            std::span<const uint8_t>(bytes).subspan(160, 144));
        bool exact = true;
        for (size_t i = 0; i != 256; ++i) {
            exact = exact && std::bit_cast<uint32_t>((*values)[i]) ==
                    std::bit_cast<uint32_t>(first_expected[i]);
            exact = exact && std::bit_cast<uint32_t>((*values)[256 + i]) ==
                    std::bit_cast<uint32_t>(second_expected[i]);
        }
        CHECK(exact);
    }
}

} // namespace

int main() {
    test_contract_bytes_and_digest();
    test_resolution_and_digest_fail_closed();
    test_plane_validation();
    test_successful_offset_padding_and_cap_policy();
    test_validation_binds_identity();
    test_canonical_program_is_executable_authority();
    test_validate_rejects_structurally_valid_digest_drift();
    test_malformed_canonical_is_rejected();
    test_physical_canonical_schema_is_bounded();
    test_physical_identity_binding();
    test_rounding_environment_is_fail_closed();
    test_real_size_nonallocating_validation();
    test_f16_exact_oracle();
    test_f16_all_bit_patterns();
    test_q4_exact_oracle();
    test_q4_high_bits_and_padded_stride();
    return test_summary("codec program");
}
