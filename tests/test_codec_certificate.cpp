#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <variant>
#include <vector>

#include "codec_certificate.h"
#include "physical_codec.h"
#include "test_util.h"

using namespace Laplace;

namespace {

CodecCertificatePlaneBinding plane(std::span<const uint8_t> bytes,
                                   uint64_t offset, uint64_t length,
                                   uint64_t stride) {
    return {bytes, offset, length, stride};
}

CodecCertificate parsed(const std::vector<uint8_t>& bytes) {
    const auto result = parse_codec_certificate(bytes);
    CHECK_MSG(std::holds_alternative<CodecCertificate>(result),
              "certificate parse error=%u bytes=%zu",
              std::holds_alternative<CodecCertificateError>(result)
                  ? static_cast<unsigned>(std::get<CodecCertificateError>(result))
                  : 0u,
              bytes.size());
    return std::holds_alternative<CodecCertificate>(result)
        ? std::get<CodecCertificate>(result) : CodecCertificate{};
}

float f16(const uint8_t* bytes) {
    const uint16_t value = static_cast<uint16_t>(bytes[0]) |
                           (static_cast<uint16_t>(bytes[1]) << 8u);
    const uint32_t sign = static_cast<uint32_t>(value >> 15u) & 1u;
    int32_t exponent = static_cast<int32_t>((value >> 10u) & 31u);
    uint32_t mantissa = value & 1023u;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign << 31u;
        } else {
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1u;
                --exponent;
            }
            ++exponent;
            bits = (sign << 31u) |
                   (static_cast<uint32_t>(exponent + 112) << 23u) |
                   ((mantissa & 1023u) << 13u);
        }
    } else if (exponent == 31) {
        bits = (sign << 31u) | 0x7f800000u | (mantissa << 13u);
    } else {
        bits = (sign << 31u) |
               (static_cast<uint32_t>(exponent + 112) << 23u) |
               (mantissa << 13u);
    }
    return std::bit_cast<float>(bits);
}

std::vector<float> decoded(const CodecCertificate& certificate,
                           const CodecCertificateBinding& binding,
                           uint64_t maximum) {
    const auto result = certificate.decode(binding, maximum);
    CHECK(std::holds_alternative<std::vector<float>>(result));
    return std::get<std::vector<float>>(result);
}

void test_raw_vectors() {
    const auto f16_certificate = parsed(make_raw_f16_codec_certificate());
    const auto f32_certificate = parsed(make_raw_f32_codec_certificate());
    CHECK(f16_certificate.summary().unit_bytes == 2);
    CHECK(f32_certificate.summary().unit_bytes == 4);
    CHECK(f16_certificate.summary().source_scalar ==
          CodecCertificateScalar::Binary16);
    CHECK(f32_certificate.summary().source_scalar ==
          CodecCertificateScalar::Binary32);
    CHECK(f16_certificate.summary().rank_independent);
    CHECK(f32_certificate.summary().rank_independent);

    const std::array<uint8_t, 6> f16_storage = {
        0x00, 0x80, 0x00, 0x7c, 0x00, 0x7e};
    const auto f16_values = decoded(
        f16_certificate,
        {3, {plane(f16_storage, 0, f16_storage.size(), 2)}}, 3);
    CHECK(f16_values.size() == 3);
    if (f16_values.size() == 3) {
        CHECK(std::bit_cast<uint32_t>(f16_values[0]) == 0x80000000u);
        CHECK(std::bit_cast<uint32_t>(f16_values[1]) == 0x7f800000u);
        CHECK(std::bit_cast<uint32_t>(f16_values[2]) == 0x7fc00000u);
    }

    const std::array<uint8_t, 16> f32_storage = {
        0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x7f,
        0x00, 0x00, 0x00, 0x80, 0x45, 0x23, 0xc1, 0x7f};
    const auto f32_values = decoded(
        f32_certificate,
        {4, {plane(f32_storage, 0, f32_storage.size(), 4)}}, 4);
    CHECK(f32_values.size() == 4);
    if (f32_values.size() == 4) {
        CHECK(std::bit_cast<uint32_t>(f32_values[0]) == 0x3f800000u);
        CHECK(std::bit_cast<uint32_t>(f32_values[1]) == 0x7f800000u);
        CHECK(std::bit_cast<uint32_t>(f32_values[2]) == 0x80000000u);
        CHECK(std::bit_cast<uint32_t>(f32_values[3]) == 0x7fc12345u);
    }
}

void test_structural_view() {
    const auto q4 = parsed(make_q4_k_codec_certificate());
    const auto& nodes = q4.node_summaries();
    const auto& maps = q4.access_map_summaries();
    const auto& accesses = q4.access_summaries();
    CHECK(nodes.size() == q4.summary().node_count);
    CHECK(maps.size() == 7);
    CHECK(accesses.size() == 1282);
    CHECK(nodes[0].operation == CodecCertificateNodeOperation::LoadScalar);
    CHECK(nodes[2].operation == CodecCertificateNodeOperation::LoadBits);
    CHECK(nodes[12].operation == CodecCertificateNodeOperation::Add);
    CHECK(nodes.back().operation == CodecCertificateNodeOperation::Add);
    CHECK(maps[2].plane == 0 && maps[2].count == 256);
    CHECK(accesses[maps[2].first].width_bits == 4);
    CHECK(accesses[maps[2].first].encoding ==
          CodecCertificateAccessEncoding::Unsigned8);

    const auto grouped = parsed(make_grouped_affine_u2_codec_certificate());
    CHECK(grouped.node_summaries().size() == grouped.summary().node_count);
    CHECK(grouped.access_map_summaries().size() == 3);
    CHECK(grouped.access_summaries().size() == 768);
    CHECK(grouped.node_summaries()[0].operation ==
          CodecCertificateNodeOperation::LoadBits);
    CHECK(grouped.access_summaries()[0].encoding ==
          CodecCertificateAccessEncoding::Unsigned32);
}

std::vector<float> q4_reference(const std::array<uint8_t, 144>& block) {
    const float d = f16(block.data());
    const float dmin = f16(block.data() + 2);
    std::vector<float> output(256);
    for (uint32_t group = 0; group < 8; ++group) {
        uint32_t scale = 0;
        uint32_t minimum = 0;
        if (group < 4) {
            scale = block[4 + group] & 63u;
            minimum = block[8 + group] & 63u;
        } else {
            scale = (block[8 + group] & 15u) |
                    ((block[4 + group - 4] >> 6u) << 4u);
            minimum = ((block[8 + group] >> 4u) & 15u) |
                      ((block[8 + group] >> 6u) << 4u);
        }
        const uint32_t q_base = 16 + (group / 2) * 32;
        const uint32_t output_base = (group / 2) * 64 + (group % 2) * 32;
        const uint32_t shift = (group % 2) * 4;
        const float scaled = d * static_cast<float>(scale);
        const float min_scaled = dmin * static_cast<float>(minimum);
        for (uint32_t i = 0; i < 32; ++i) {
            const uint32_t q = (block[q_base + i] >> shift) & 15u;
            output[output_base + i] =
                scaled * static_cast<float>(q) - min_scaled;
        }
    }
    return output;
}

void test_q4_vector() {
    const auto certificate = parsed(make_q4_k_codec_certificate());
    CHECK(certificate.summary().unit_elements == 256);
    CHECK(certificate.summary().unit_bytes == 144);
    CHECK(certificate.summary().plane_count == 1);
    std::array<uint8_t, 144> block{};
    block[0] = 0x55;
    block[1] = 0x35;
    block[2] = 0x66;
    block[3] = 0x2e;
    for (size_t i = 0; i < 12; ++i) block[4 + i] = i + 1;
    for (size_t i = 0; i < 128; ++i)
        block[16 + i] = static_cast<uint8_t>(i * 37u + 0x13u);
    const auto values = decoded(
        certificate, {1, {plane(block, 0, block.size(), block.size())}}, 256);
    const auto expected = q4_reference(block);
    CHECK(values.size() == expected.size());
    if (values.size() == expected.size())
        for (size_t i = 0; i < values.size(); ++i) {
            CHECK(std::fabs(values[i] - expected[i]) < 1e-5f);
        }
}

std::vector<float> q6_reference(const std::array<uint8_t, 210>& block) {
    const float d = f16(block.data());
    std::vector<float> output(256);
    for (uint32_t i = 0; i < 256; ++i) {
        const uint32_t segment = i / 128;
        const uint32_t within = i % 128;
        const uint32_t quarter = within / 32;
        const uint32_t lane = within % 32;
        const uint32_t ql_offset =
            2 + segment * 64 + (quarter / 2) * 32 + lane;
        const uint32_t qh_offset = 2 + 128 + segment * 32 + lane;
        const uint32_t ql =
            (block[ql_offset] >> ((quarter % 2) * 4u)) & 15u;
        const uint32_t qh =
            ((block[qh_offset] >> (quarter * 2u)) & 3u) << 4u;
        const int32_t scale = static_cast<int8_t>(block[194 + i / 16]);
        output[i] = d * static_cast<float>(static_cast<int32_t>(ql + qh) - 32) *
                    static_cast<float>(scale);
    }
    return output;
}

void test_q6_vector() {
    const auto certificate = parsed(make_q6_k_codec_certificate());
    CHECK(certificate.summary().unit_elements == 256);
    CHECK(certificate.summary().unit_bytes == 210);
    CHECK(certificate.plane_summaries()[0].storage_scalar ==
          CodecCertificateStorageScalar::Unsigned8);
    std::array<uint8_t, 210> block{};
    block[0] = 0x42;
    block[1] = 0x34;
    for (size_t i = 2; i < block.size(); ++i)
        block[i] = static_cast<uint8_t>(i * 29u + 7u);
    const auto values = decoded(
        certificate, {1, {plane(block, 0, block.size(), block.size())}}, 256);
    const auto expected = q6_reference(block);
    CHECK(values.size() == expected.size());
    if (values.size() == expected.size())
        for (size_t i = 0; i < values.size(); ++i)
            CHECK(std::fabs(values[i] - expected[i]) < 1e-5f);
}

void test_q8_vector() {
    const auto certificate = parsed(make_q8_0_codec_certificate());
    CHECK(certificate.summary().unit_elements == 32);
    CHECK(certificate.summary().unit_bytes == 34);
    std::array<uint8_t, 34> block{};
    block[0] = 0x00;
    block[1] = 0x3c; // binary16 1.0
    for (uint32_t index = 0; index != 32; ++index)
        block[2 + index] = static_cast<uint8_t>(static_cast<int8_t>(index) - 16);
    const auto values = decoded(
        certificate, {1, {plane(block, 0, block.size(), block.size())}}, 32);
    CHECK(values.size() == 32);
    if (values.size() == 32)
        for (uint32_t index = 0; index != 32; ++index)
            CHECK(values[index] == static_cast<float>(static_cast<int8_t>(index) - 16));
}

void test_grouped_affine_vector() {
    const auto certificate = parsed(make_grouped_affine_u2_codec_certificate());
    CHECK(certificate.summary().plane_count == 3);
    CHECK(certificate.plane_summaries()[0].storage_scalar ==
          CodecCertificateStorageScalar::Unsigned32);
    std::array<uint8_t, 64> values{};
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<uint8_t>(i * 13u + 5u);
    const std::array<uint8_t, 2> scale = {0x00, 0x3c};
    const std::array<uint8_t, 2> bias = {0x00, 0x38};
    const auto result = decoded(
        certificate,
        {1, {plane(values, 0, values.size(), values.size()),
             plane(scale, 0, scale.size(), scale.size()),
             plane(bias, 0, bias.size(), bias.size())}},
        256);
    CHECK(result.size() == 256);
    if (result.size() == 256) {
        for (uint32_t i = 0; i < 256; ++i) {
            const uint32_t word = static_cast<uint32_t>(values[(i / 16) * 4]) |
                                  (static_cast<uint32_t>(
                                      values[(i / 16) * 4 + 1]) << 8u) |
                                  (static_cast<uint32_t>(
                                      values[(i / 16) * 4 + 2]) << 16u) |
                                  (static_cast<uint32_t>(
                                      values[(i / 16) * 4 + 3]) << 24u);
            const float expected =
                f16(scale.data()) * static_cast<float>((word >> ((i % 16) * 2)) & 3u) +
                f16(bias.data());
            CHECK(std::fabs(result[i] - expected) < 1e-6f);
        }
    }
}

PhysicalCodecIdentity identity_for(const CodecCertificate& certificate,
                                   bool grouped = false) {
    PhysicalCodecIdentity identity;
    identity.arithmetic_version = certificate.identity().abi_version;
    identity.arithmetic_digest = certificate.identity().digest;
    identity.layout.kind = static_cast<PhysicalLayoutKind>(
        certificate.summary().physical_layout_kind);
    identity.layout.packing = static_cast<PackingKind>(
        certificate.summary().physical_layout_packing);
    identity.layout.block_rank =
        certificate.summary().physical_layout_block_rank;
    identity.layout.block_elements =
        certificate.summary().physical_layout_block_elements;
    identity.layout.block_bytes = certificate.summary().physical_layout_block_bytes;
    identity.quantization.kind = static_cast<QuantizationKind>(
        certificate.summary().physical_quantization_kind);
    identity.quantization.block_elements =
        certificate.summary().physical_quantization_block_elements;
    identity.quantization.block_bytes =
        certificate.summary().physical_quantization_block_bytes;
    identity.quantization.group_size =
        certificate.summary().physical_quantization_group_size;
    identity.quantization.required_plane_mask =
        certificate.summary().physical_quantization_required_plane_mask;
    if (identity.layout.kind == PhysicalLayoutKind::GgufBlocked) {
        identity.quantization.accumulation_type = ScalarType::F32;
        identity.quantization.scale_type = ScalarType::F16;
        identity.quantization.zero_type =
            identity.layout.block_bytes == 144 ? ScalarType::F16
                                               : static_cast<ScalarType>(0);
        identity.quantization.bias_type = static_cast<ScalarType>(0);
    }
    for (const auto& plane_summary : certificate.plane_summaries()) {
        const PlaneKind kind = plane_summary.role == CodecCertificatePlaneRole::Values
            ? PlaneKind::Values
            : plane_summary.role == CodecCertificatePlaneRole::Scales
                ? PlaneKind::Scales : PlaneKind::Biases;
        const ScalarType storage =
            plane_summary.storage_scalar == CodecCertificateStorageScalar::Binary16
                ? ScalarType::F16
                : plane_summary.storage_scalar == CodecCertificateStorageScalar::Binary32
                    ? ScalarType::F32
                    : plane_summary.storage_scalar == CodecCertificateStorageScalar::Unsigned32
                        ? ScalarType::U32 : ScalarType::U8;
        identity.planes.push_back({kind, storage,
                                   plane_summary.elements_per_unit,
                                   plane_summary.bytes_per_unit, 0});
    }
    (void)grouped;
    return identity;
}

void test_physical_identity() {
    const auto f16 = parsed(make_raw_f16_codec_certificate());
    CHECK(f16.matches_physical_identity(identity_for(f16)));
    const auto f32 = parsed(make_raw_f32_codec_certificate());
    CHECK(f32.matches_physical_identity(identity_for(f32)));
    const auto q4 = parsed(make_q4_k_codec_certificate());
    auto q4_identity = identity_for(q4);
    CHECK(q4.matches_physical_identity(q4_identity));
    q4_identity.layout.block_bytes = 143;
    CHECK(!q4.matches_physical_identity(q4_identity));
    const auto q6 = parsed(make_q6_k_codec_certificate());
    CHECK(q6.matches_physical_identity(identity_for(q6)));
    const auto q8 = parsed(make_q8_0_codec_certificate());
    auto q8_identity = identity_for(q8);
    q8_identity.quantization.scale_type = ScalarType::F16;
    CHECK(q8.matches_physical_identity(q8_identity));
    q8_identity.quantization.scale_type = ScalarType::F32;
    CHECK(!q8.matches_physical_identity(q8_identity));
    q8_identity = identity_for(q8);
    q8_identity.quantization.group_size = 64;
    CHECK(!q8.matches_physical_identity(q8_identity));
    const auto grouped = parsed(make_grouped_affine_u2_codec_certificate());
    CHECK(grouped.matches_physical_identity(identity_for(grouped, true)));
}

void test_fail_closed() {
    auto malformed = make_raw_f16_codec_certificate();
    malformed[10] = 3;
    CHECK(std::holds_alternative<CodecCertificateError>(
        parse_codec_certificate(malformed)));
    auto malformed_q8 = make_q8_0_codec_certificate();
    // The first plane is a required values plane; corrupt its role before
    // parsing so the Q8 certificate cannot be accepted as another layout.
    malformed_q8[68] = 0;
    CHECK(std::holds_alternative<CodecCertificateError>(
        parse_codec_certificate(malformed_q8)));
    malformed = make_raw_f16_codec_certificate();
    malformed[68] = 0;
    CHECK(std::holds_alternative<CodecCertificateError>(
        parse_codec_certificate(malformed)));
    malformed = make_raw_f16_codec_certificate();
    malformed[95] = 0xff;
    CHECK(std::holds_alternative<CodecCertificateError>(
        parse_codec_certificate(malformed)));
    malformed = make_raw_f16_codec_certificate();
    malformed[111] = 3;
    CHECK(std::holds_alternative<CodecCertificateError>(
        parse_codec_certificate(malformed)));
    const auto certificate = parsed(make_raw_f32_codec_certificate());
    const std::array<uint8_t, 4> storage = {0, 0, 0x80, 0x3f};
    CHECK(certificate.validate(
              {2, {plane(storage, 0, storage.size(), 4)}}) ==
          CodecCertificateError::AccessOutOfBounds);
    CHECK(certificate.validate(
              {1, {plane(storage, 0, storage.size(), 3)}}) ==
          CodecCertificateError::InvalidStride);
    CHECK(certificate.validate(
              {2, {plane(storage, 0, storage.size(), 4)}}) ==
          CodecCertificateError::AccessOutOfBounds);
    const auto decoded_result = certificate.decode(
        {2, {plane(storage, 0, storage.size(), 4)}}, 1);
    CHECK(std::holds_alternative<CodecCertificateError>(decoded_result));
    const std::vector<uint8_t> oversized(64u * 1024u + 1u, 0);
    CHECK(std::holds_alternative<CodecCertificateError>(
        parse_codec_certificate(oversized)));

    // The certificate graph is typed. The declared depth, terminal type,
    // load operation, and every arithmetic operand must agree with that graph.
    auto wrong_depth = make_raw_f32_codec_certificate();
    wrong_depth[31] = 2;
    CHECK(std::holds_alternative<CodecCertificateError>(
        parse_codec_certificate(wrong_depth)));

    auto unsigned_terminal = make_raw_f32_codec_certificate();
    unsigned_terminal[69] =
        static_cast<uint8_t>(CodecCertificateScalar::PackedUnsigned);
    unsigned_terminal[70] =
        static_cast<uint8_t>(CodecCertificateStorageScalar::Unsigned32);
    unsigned_terminal[71] = 1;
    unsigned_terminal[95] =
        static_cast<uint8_t>(CodecCertificateNodeOperation::LoadBits);
    unsigned_terminal[96] =
        static_cast<uint8_t>(CodecCertificateNodeValueType::Unsigned);
    unsigned_terminal[141] =
        static_cast<uint8_t>(CodecCertificateAccessEncoding::Unsigned32);
    CHECK(std::holds_alternative<CodecCertificateError>(
        parse_codec_certificate(unsigned_terminal)));

    auto scalar_from_integer = make_raw_f32_codec_certificate();
    scalar_from_integer[69] =
        static_cast<uint8_t>(CodecCertificateScalar::PackedUnsigned);
    scalar_from_integer[70] =
        static_cast<uint8_t>(CodecCertificateStorageScalar::Unsigned32);
    scalar_from_integer[71] = 1;
    scalar_from_integer[141] =
        static_cast<uint8_t>(CodecCertificateAccessEncoding::Unsigned32);
    CHECK(std::holds_alternative<CodecCertificateError>(
        parse_codec_certificate(scalar_from_integer)));

    auto integer_negation = make_q4_k_codec_certificate();
    constexpr size_t q4_first_node = 95;
    constexpr size_t q4_node_bytes = 16;
    integer_negation[q4_first_node + 8 * q4_node_bytes] =
        static_cast<uint8_t>(CodecCertificateNodeOperation::Neg);
    CHECK(std::holds_alternative<CodecCertificateError>(
        parse_codec_certificate(integer_negation)));

    // The current wire ABI has no ScalarType representation for these plane
    // storage values.  A record that uses the otherwise valid access encoding
    // must therefore fail closed rather than parse into an unrepresentable
    // identity.
    auto unsigned16_access = make_grouped_affine_u2_codec_certificate();
    constexpr size_t grouped_values_storage = 70;
    constexpr size_t grouped_first_record = 293;
    constexpr size_t grouped_first_record_encoding = 299;
    constexpr size_t grouped_record_bytes = 12;
    unsigned16_access[grouped_first_record_encoding] =
        static_cast<uint8_t>(CodecCertificateAccessEncoding::Unsigned16);
    CHECK(std::holds_alternative<CodecCertificateError>(
        parse_codec_certificate(unsigned16_access)));

    auto signed8_storage = make_grouped_affine_u2_codec_certificate();
    signed8_storage[grouped_values_storage] =
        static_cast<uint8_t>(CodecCertificateStorageScalar::Signed8);
    for (size_t record = 0; record < 256; ++record) {
        signed8_storage[grouped_first_record +
                        record * grouped_record_bytes + 4] = 0;
        signed8_storage[grouped_first_record +
                        record * grouped_record_bytes + 5] = 8;
        signed8_storage[grouped_first_record_encoding +
                        record * grouped_record_bytes] =
            static_cast<uint8_t>(CodecCertificateAccessEncoding::Signed8);
    }
    constexpr size_t grouped_nodes = 149;
    signed8_storage[grouped_nodes + 1] =
        static_cast<uint8_t>(CodecCertificateNodeValueType::Signed);
    CHECK(std::holds_alternative<CodecCertificateError>(
        parse_codec_certificate(signed8_storage)));

    auto unsigned16_storage = make_grouped_affine_u2_codec_certificate();
    unsigned16_storage[grouped_values_storage] =
        static_cast<uint8_t>(CodecCertificateStorageScalar::Unsigned16);
    for (size_t record = 0; record < 256; ++record) {
        unsigned16_storage[grouped_first_record +
                           record * grouped_record_bytes + 4] =
            static_cast<uint8_t>((record % 8) * 2);
        unsigned16_storage[grouped_first_record_encoding +
                           record * grouped_record_bytes] =
            static_cast<uint8_t>(CodecCertificateAccessEncoding::Unsigned16);
    }
    CHECK(std::holds_alternative<CodecCertificateError>(
        parse_codec_certificate(unsigned16_storage)));
}

} // namespace

int main() {
    test_raw_vectors();
    test_structural_view();
    test_q4_vector();
    test_q6_vector();
    test_q8_vector();
    test_grouped_affine_vector();
    test_physical_identity();
    test_fail_closed();
    return test_summary("test_codec_certificate");
}
