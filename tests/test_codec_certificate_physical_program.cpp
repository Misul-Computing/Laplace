#include <algorithm>
#include <bit>
#include <chrono>
#include <functional>
#include <numeric>

#include "codec_certificate_physical_program.h"
#include "physical_program_interpreter.h"
#include "test_util.h"
#ifdef LAPLACE_CODEC_TEST_METAL
#include "program_metal.h"
#endif

using namespace Laplace;
namespace {
void put(std::vector<uint8_t>& bytes, uint64_t value, unsigned count) {
    for (unsigned i = 0; i < count; ++i) bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
}
void set(std::vector<uint8_t>& bytes, size_t offset, uint64_t value, unsigned count) {
    for (unsigned i = 0; i < count; ++i) bytes[offset + i] = static_cast<uint8_t>(value >> (8 * i));
}

// Test-only grammar builder. Its physical tuple is deliberately unchanged from
// a raw declaration while the access grammar varies independently.
std::vector<uint8_t> fixture(uint32_t elements, uint32_t unit_bytes,
    CodecCertificateStorageScalar storage,
    const std::vector<std::vector<CodecCertificateAccessSummary>>& maps,
    CodecCertificateNodeValueType type, bool deep = false, bool constant_root = false) {
    using N = CodecCertificateNodeSummary;
    using O = CodecCertificateNodeOperation;
    std::vector<N> nodes;
    std::vector<uint16_t> layer;
    for (uint16_t m = 0; m < maps.size(); ++m) {
        layer.push_back(static_cast<uint16_t>(nodes.size()));
        nodes.push_back({type == CodecCertificateNodeValueType::Float ? O::LoadScalar : O::LoadBits,
            type, 0, 0, 0xffff, 0xffff, 0xffff, m});
        if (type != CodecCertificateNodeValueType::Float) {
            const auto load = layer.back();
            layer.back() = static_cast<uint16_t>(nodes.size());
            nodes.push_back({O::CastFloat, CodecCertificateNodeValueType::Float, 0xff, 0,
                             load});
        }
    }
    while (layer.size() > 1) {
        std::vector<uint16_t> next;
        for (size_t i = 0; i < layer.size(); i += 2) {
            if (i + 1 == layer.size()) { next.push_back(layer[i]); continue; }
            next.push_back(static_cast<uint16_t>(nodes.size()));
            nodes.push_back({O::Add, CodecCertificateNodeValueType::Float, 0xff, 0,
                             layer[i], layer[i + 1]});
        }
        layer = std::move(next);
    }
    std::vector<uint8_t> depth;
    for (const auto& node : nodes) {
        uint8_t d = 1;
        for (auto arg : {node.argument0, node.argument1, node.argument2})
            if (arg != 0xffff) d = std::max<uint8_t>(d, depth[arg] + 1);
        depth.push_back(d);
    }
    if (deep || (maps.size() == 64 && type == CodecCertificateNodeValueType::Float)) {
        const auto target = deep ? 32 : depth.back() + 1;
        while (depth.back() < target) {
            auto last = static_cast<uint16_t>(nodes.size() - 1);
            nodes.push_back({O::Neg, CodecCertificateNodeValueType::Float, 0xff, 0, last});
            depth.push_back(depth.back() + 1);
        }
    }
    if (constant_root) {
        nodes.push_back({O::Constant, CodecCertificateNodeValueType::Float, 0xff,
                         0, 0xffff, 0xffff, 0xffff, 0});
        nodes.push_back({O::Neg, CodecCertificateNodeValueType::Float, 0xff,
                         0, static_cast<uint16_t>(nodes.size() - 1)});
        depth.push_back(2);
    }
    auto bytes = make_raw_f32_codec_certificate();
    bytes.resize(68);
    set(bytes, 18, elements, 4); set(bytes, 22, unit_bytes, 4);
    set(bytes, 30, 1, 1); set(bytes, 31, depth.back(), 1);
    set(bytes, 32, nodes.size(), 2); set(bytes, 34, constant_root ? 1 : 0, 1);
    put(bytes, 1, 1); put(bytes, 3, 1); put(bytes, static_cast<uint8_t>(storage), 1);
    put(bytes, 1, 1); put(bytes, 1, 1);
    put(bytes, storage == CodecCertificateStorageScalar::Unsigned32 ? 32 : 8, 2);
    put(bytes, elements, 4); put(bytes, unit_bytes, 4); put(bytes, 1, 4);
    put(bytes, 0, 4); put(bytes, unit_bytes, 4);
    for (const auto& n : nodes) {
        put(bytes, static_cast<uint8_t>(n.operation), 1);
        put(bytes, static_cast<uint8_t>(n.value_type), 1);
        put(bytes, n.plane, 1); put(bytes, 0, 1);
        put(bytes, n.argument0, 2); put(bytes, n.argument1, 2); put(bytes, n.argument2, 2);
        put(bytes, n.immediate, 4); put(bytes, 0, 2);
    }
    if (constant_root) put(bytes, 0x80000000u, 4);
    put(bytes, 2, 4); put(bytes, maps.size(), 2); put(bytes, 0, 2);
    size_t total = 0;
    for (const auto& m : maps) total += m.size();
    put(bytes, total, 4);
    size_t first = 0;
    for (const auto& m : maps) {
        put(bytes, 0, 4); put(bytes, first, 4); put(bytes, m.size(), 4);
        first += m.size();
    }
    for (const auto& m : maps) for (const auto& a : m) {
        put(bytes, a.byte_offset, 4); put(bytes, a.bit_offset, 1);
        put(bytes, a.width_bits, 1); put(bytes, static_cast<uint8_t>(a.encoding), 1);
        put(bytes, a.flags, 1); put(bytes, a.value_shift, 1); put(bytes, 0, 3);
    }
    return bytes;
}

void parity(const std::vector<uint8_t>& wire, uint64_t units = 2,
            std::vector<uint64_t> extents = {}, bool nan_data = false,
            std::vector<uint64_t> element_strides = {}) {
    const auto parsed = parse_codec_certificate(wire);
    CHECK_MSG(std::holds_alternative<CodecCertificate>(parsed), "source parse error %u",
        std::holds_alternative<CodecCertificateError>(parsed) ? unsigned(std::get<CodecCertificateError>(parsed)) : 0);
    if (!std::holds_alternative<CodecCertificate>(parsed)) return;
    const auto& cert = std::get<CodecCertificate>(parsed);
    const uint64_t storage_count = units * cert.summary().unit_elements;
    if (extents.empty()) extents = {units, cert.summary().unit_elements};
    const uint64_t count = std::accumulate(extents.begin(), extents.end(), uint64_t{1}, std::multiplies<>{});
    if (element_strides.empty()) {
        element_strides.resize(extents.size());
        uint64_t stride = 1;
        for (size_t axis = extents.size(); axis-- > 0;) {
            element_strides[axis] = stride;
            stride *= extents[axis];
        }
    }
    std::vector<uint64_t> source_indexes;
    for (uint64_t i = 0; i < count; ++i) {
        uint64_t rest = i, mapped = 0;
        for (size_t axis = extents.size(); axis-- > 0;) {
            mapped += (rest % extents[axis]) * element_strides[axis];
            rest /= extents[axis];
        }
        CHECK(mapped < storage_count);
        source_indexes.push_back(mapped);
    }
    const LogicalTensorType logical{ElementType::F32, extents};
    std::vector<uint64_t> strides;
    std::vector<std::shared_ptr<const std::vector<uint8_t>>> owners;
    CodecCertificateBinding source{units, {}};
    for (const auto& plane : cert.plane_summaries()) {
        auto stride = plane.stride + plane.alignment * 3;
        strides.push_back(stride);
        auto owner = std::make_shared<std::vector<uint8_t>>((units - 1) * stride + plane.bytes_per_unit);
        XorShift32 random(0x5234);
        for (auto& byte : *owner) byte = nan_data ? 0xff : static_cast<uint8_t>(random.next());
        owners.push_back(owner);
        source.planes.push_back({*owner, 0, owner->size(), stride});
    }
    auto expected = cert.decode(source, storage_count);
    CHECK(std::holds_alternative<std::vector<float>>(expected));
    if (!std::holds_alternative<std::vector<float>>(expected)) return;
    const auto start = std::chrono::steady_clock::now();
    auto translation = translate_codec_certificate(cert, logical, element_strides, strides);
    CHECK_MSG(std::holds_alternative<TranslatedCodecCertificate>(translation), "%s",
        std::holds_alternative<CompatibilityReport>(translation) ? std::get<CompatibilityReport>(translation).detail.c_str() : "");
    if (!std::holds_alternative<TranslatedCodecCertificate>(translation)) return;
    const auto& translated = std::get<TranslatedCodecCertificate>(translation);
    CHECK(translated.required_units == *std::max_element(source_indexes.begin(), source_indexes.end()) / cert.summary().unit_elements + 1);
    CHECK(translated.program.instructions.size() <= 3264);
    CHECK(translated.program.inline_bytes.size() <= 32768);
    std::vector<unsigned> depths;
    for (const auto& instruction : translated.program.instructions) {
        unsigned depth = 1;
        for (auto operand : instruction.operands)
            if (operand != kNoPhysicalValue) depth = std::max(depth, depths[operand] + 1);
        depths.push_back(depth);
    }
    CHECK(*std::max_element(depths.begin(), depths.end()) <= 99);
    std::vector<PhysicalPlaneBinding> bindings;
    for (uint16_t p = 0; p < translated.source_planes.size(); ++p) {
        auto old = translated.source_planes[p];
        if (old != kNoPhysicalPlane) bindings.push_back({p, owners[old], 0, owners[old]->size()});
    }
    auto checked = verify_physical_program(translated.program, bindings, logical);
    CHECK_MSG(std::holds_alternative<VerifiedPhysicalProgram>(checked), "%s",
        std::holds_alternative<CompatibilityReport>(checked) ? std::get<CompatibilityReport>(checked).detail.c_str() : "");
    if (!std::holds_alternative<VerifiedPhysicalProgram>(checked)) return;
    const auto& verified = std::get<VerifiedPhysicalProgram>(checked);
    auto encoded = encode_physical_program(translated.program);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
    if (std::holds_alternative<std::vector<uint8_t>>(encoded)) {
        auto reparsed = parse_physical_program(std::get<std::vector<uint8_t>>(encoded));
        CHECK(std::holds_alternative<PhysicalProgram>(reparsed));
        if (std::holds_alternative<PhysicalProgram>(reparsed))
            CHECK(std::get<PhysicalProgram>(reparsed) == translated.program);
    }
    if (cert.access_summaries().size() == 4096) {
        CHECK(std::get<std::vector<uint8_t>>(encoded).size() > 64 * 1024);
        printf("maximum grammar: %zu nodes, %zu inline bytes, compile+verify %.3f ms\n",
            translated.program.instructions.size(), translated.program.inline_bytes.size(),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        auto too_many = translated.program;
        too_many.instructions.resize(4097);
        CHECK(std::holds_alternative<CompatibilityReport>(canonicalize_physical_program(too_many)));
        auto too_deep = translated.program;
        uint16_t policy = kNoPhysicalPolicy;
        for (uint16_t p = 0; p < too_deep.policies.size(); ++p)
            if (too_deep.policies[p].contraction == PhysicalContractionPolicy::Separate)
                policy = p;
        for (unsigned d = depths.back(); d < 129; ++d) {
            PhysicalInstruction n;
            n.opcode = PhysicalOpcode::F32Negate;
            n.result_type = PhysicalValueType::F32;
            n.operands[0] = too_deep.result;
            n.policy = policy;
            too_deep.result = too_deep.instructions.size();
            too_deep.instructions.push_back(n);
        }
        CHECK(std::holds_alternative<CompatibilityReport>(canonicalize_physical_program(too_deep)));
    }
    for (uint64_t index = 0; index < count; ++index) {
        auto remaining = index;
        std::vector<uint64_t> coordinate(extents.size());
        for (size_t axis = extents.size(); axis-- > 0;) {
            coordinate[axis] = remaining % extents[axis];
            remaining /= extents[axis];
        }
        auto actual = interpret_physical_value(verified, coordinate);
        CHECK(std::holds_alternative<ScalarValue>(actual));
        if (!std::holds_alternative<ScalarValue>(actual)) continue;
        auto want = std::bit_cast<uint32_t>(std::get<std::vector<float>>(expected)[source_indexes[index]]);
        CHECK_MSG(std::get<ScalarValue>(actual).bits == want, "element %llu actual %llx expected %x",
            (unsigned long long)index, (unsigned long long)std::get<ScalarValue>(actual).bits, want);
    }
#ifdef LAPLACE_CODEC_TEST_METAL
    auto compiled = compile_metal_physical_program(verified);
    CHECK_MSG(std::holds_alternative<MetalPhysicalProgramExecutable>(compiled), "%s",
        std::holds_alternative<CompatibilityReport>(compiled) ? std::get<CompatibilityReport>(compiled).detail.c_str() : "");
    if (std::holds_alternative<MetalPhysicalProgramExecutable>(compiled)) {
        auto execution = std::get<MetalPhysicalProgramExecutable>(compiled).execute();
        CHECK(std::holds_alternative<MetalPhysicalProgramResult>(execution));
        if (std::holds_alternative<MetalPhysicalProgramResult>(execution)) {
            const auto& bits = std::get<MetalPhysicalProgramResult>(execution).value.bits;
            CHECK(bits.size() == count);
            for (size_t i = 0; i < bits.size(); ++i) {
                auto wanted = std::bit_cast<uint32_t>(std::get<std::vector<float>>(expected)[source_indexes[i]]);
                const bool arithmetic = std::any_of(cert.node_summaries().begin(),
                    cert.node_summaries().end(), [](const auto& n) {
                        return n.operation >= CodecCertificateNodeOperation::Add &&
                               n.operation <= CodecCertificateNodeOperation::Neg;
                    });
                // Arithmetic NaN payload selection is device-dependent. Direct
                // loads, finite results, signed zeros and infinities stay exact.
                const bool arithmetic_nan = arithmetic &&
                    std::isnan(std::bit_cast<float>(wanted)) &&
                    std::isnan(std::bit_cast<float>(static_cast<uint32_t>(bits[i])));
                CHECK_MSG(bits[i] == wanted || arithmetic_nan, "GPU element %zu actual %llx expected %x", i,
                          (unsigned long long)bits[i], wanted);
            }
        }
    }
#endif
    if (!bindings.empty()) {
        auto invalid = bindings;
        invalid.front().length = 0;
        CHECK(std::holds_alternative<CompatibilityReport>(verify_physical_program(translated.program, invalid, logical)));
    }
    auto bad_strides = strides;
    bad_strides[0] = UINT64_MAX;
    CHECK(std::holds_alternative<CompatibilityReport>(translate_codec_certificate(cert, logical, element_strides, bad_strides)));
    CHECK(std::holds_alternative<CompatibilityReport>(translate_codec_certificate(cert, {ElementType::F32, {UINT64_MAX, 2}}, std::array<uint64_t, 2>{1, 1}, strides)));
}
} // namespace

int main() {
    const auto raw = parse_codec_certificate(make_raw_f32_codec_certificate());
    CHECK(std::holds_alternative<CodecCertificate>(raw));
    if (const auto* certificate = std::get_if<CodecCertificate>(&raw)) {
        const std::array<uint64_t, 1> element_stride{1}, plane_stride{4};
        const auto first = translate_codec_certificate(*certificate,
            {ElementType::F32, {2}}, element_stride, plane_stride);
        const auto second = translate_codec_certificate(*certificate,
            {ElementType::F32, {3}}, element_stride, plane_stride);
        CHECK(std::holds_alternative<TranslatedCodecCertificate>(first));
        CHECK(std::holds_alternative<TranslatedCodecCertificate>(second));
        if (std::holds_alternative<TranslatedCodecCertificate>(first) &&
            std::holds_alternative<TranslatedCodecCertificate>(second)) {
            const auto left = physical_program_digest(
                std::get<TranslatedCodecCertificate>(first).program);
            const auto right = physical_program_digest(
                std::get<TranslatedCodecCertificate>(second).program);
            CHECK(std::holds_alternative<PhysicalProgramDigest>(left));
            CHECK(std::holds_alternative<PhysicalProgramDigest>(right));
            if (std::holds_alternative<PhysicalProgramDigest>(left) &&
                std::holds_alternative<PhysicalProgramDigest>(right))
                CHECK(std::get<PhysicalProgramDigest>(left) !=
                      std::get<PhysicalProgramDigest>(right));
        }
    }
    for (const auto& wire : {make_raw_f16_codec_certificate(), make_raw_f32_codec_certificate(),
        make_q4_k_codec_certificate(), make_q5_0_codec_certificate(), make_q6_k_codec_certificate(),
        make_q8_0_codec_certificate(), make_grouped_affine_u2_codec_certificate()}) parity(wire);
    parity(make_raw_f16_codec_certificate(), 4, {}, true);
    parity(make_raw_f32_codec_certificate(), 4, {}, true);
    using E = CodecCertificateAccessEncoding;
    using T = CodecCertificateNodeValueType;
    using S = CodecCertificateStorageScalar;
    // Full-width field starts at bit 31; narrower fields reach the last bit.
    parity(fixture(4, 12, S::Unsigned32, {{{0, 31, 32, E::Unsigned32},
        {8, 7, 25, E::Unsigned32}, {11, 7, 1, E::Unsigned32, 0, 31},
        {UINT32_MAX, 0, 0, E::Unsigned8, 1}}}, T::Unsigned));
    std::vector<CodecCertificateAccessSummary> signed_records;
    for (uint8_t width = 1; width <= 8; ++width)
        for (uint8_t shift = 0; shift <= 32 - width; ++shift)
            signed_records.push_back({uint32_t(width % 3), 0, width, E::Signed8, 0, shift});
    signed_records.push_back({UINT32_MAX, 0, 0, E::Unsigned8, 1});
    parity(fixture(signed_records.size(), 3, S::Unsigned8, {signed_records}, T::Signed));
    parity(fixture(4, 9, S::Unsigned8, {{{0, 0, 16, E::Binary16}, {5, 0, 32, E::Binary32},
        {7, 0, 16, E::Binary16}, {UINT32_MAX, 0, 0, E::Unsigned8, 1}}}, T::Float));
    parity(fixture(1, 1, S::Unsigned8, {{{0, 0, 8, E::Signed8}}}, T::Signed, true));
    std::vector<std::vector<CodecCertificateAccessSummary>> maximum(64);
    for (uint32_t m = 0; m < 64; ++m)
        for (uint32_t j = 0; j < 64; ++j)
            maximum[m].push_back({uint32_t((m + j) % 7), 0, 16, E::Binary16});
    parity(fixture(64, 9, S::Unsigned8, maximum, T::Float), 2, {2, 2, 2, 2, 2, 2, 2, 1});
    const auto unused = parse_codec_certificate(fixture(1, 1, S::Unsigned8,
        {{{0, 0, 8, E::Signed8}}}, T::Signed, false, true));
    CHECK(std::holds_alternative<CodecCertificateError>(unused));
    if (std::holds_alternative<CodecCertificateError>(unused))
        CHECK(std::get<CodecCertificateError>(unused) == CodecCertificateError::InvalidTopology);
    std::vector<CodecCertificateAccessSummary> fields;
    for (uint8_t width = 1; width <= 32; ++width)
        for (uint8_t bit = 0; bit < 32; ++bit)
            fields.push_back({0, bit, width, E::Unsigned32, 0, uint8_t(32 - width)});
    parity(fixture(fields.size(), 8, S::Unsigned32, {fields}, T::Unsigned));
    parity(fixture(65536, 1048576, S::Unsigned32,
        {{{1048572, 0, 32, E::Unsigned32}}}, T::Unsigned), 1);
    CHECK(std::holds_alternative<CompatibilityReport>(translate_codec_certificate(
        CodecCertificate{}, {ElementType::F32, {1}}, {}, {})));
    const auto mapped = fixture(4, 12, S::Unsigned32,
        {{{0, 0, 32, E::Unsigned32}, {4, 0, 32, E::Unsigned32},
          {8, 0, 32, E::Unsigned32}, {8, 0, 8, E::Unsigned32}}}, T::Unsigned);
    parity(mapped, 2, {2, 4}, false, {1, 2});
    parity(mapped, 3, {2, 4}, false, {8, 1});
    parity(mapped, 1, {3}, false, {1});
    return test_summary("codec_certificate_physical_program");
}
