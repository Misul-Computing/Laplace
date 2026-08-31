#include <array>
#include <bit>
#include <cstdint>
#include <variant>
#include <vector>

#include "codec_certificate.h"
#include "metal_codec_capability.h"
#include "normalized_codec_program.h"
#include "test_util.h"

using namespace Laplace;

namespace {

CodecCertificate parse(const std::vector<uint8_t>& bytes) {
    const auto result = parse_codec_certificate(bytes);
    CHECK_MSG(std::holds_alternative<CodecCertificate>(result),
              "parse error=%u size=%zu",
              std::holds_alternative<CodecCertificateError>(result)
                  ? static_cast<unsigned>(std::get<CodecCertificateError>(result))
                  : 0u, bytes.size());
    return std::holds_alternative<CodecCertificate>(result)
        ? std::get<CodecCertificate>(result) : CodecCertificate{};
}

void store_u32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8)
        bytes[offset + shift / 8] = static_cast<uint8_t>(value >> shift);
}

// The helper changes only the order of the scale and bias access maps in the
// grouped fixture. Their node map references are changed with them, so the
// decoded program is identical while the serialized identity is different.
std::vector<uint8_t> reorder_grouped_maps(const std::vector<uint8_t>& input) {
    constexpr size_t kPlaneOffset = 68;
    constexpr size_t kPlaneBytes = 27;
    constexpr size_t kNodeOffset = kPlaneOffset + 3 * kPlaneBytes;
    constexpr size_t kNodeBytes = 16;
    constexpr size_t kMapHeaderOffset = kNodeOffset + 6 * kNodeBytes + 12;
    constexpr size_t kMapBytes = 12;
    constexpr size_t kRecordOffset = kMapHeaderOffset + 3 * kMapBytes;
    constexpr size_t kRecordBytes = 12;
    constexpr size_t kRecordsPerMap = 256;
    std::vector<uint8_t> output = input;
    const auto map1 = std::vector<uint8_t>(
        input.begin() + kMapHeaderOffset + kMapBytes,
        input.begin() + kMapHeaderOffset + 2 * kMapBytes);
    const auto map2 = std::vector<uint8_t>(
        input.begin() + kMapHeaderOffset + 2 * kMapBytes,
        input.begin() + kMapHeaderOffset + 3 * kMapBytes);
    std::copy(map2.begin(), map2.end(),
              output.begin() + kMapHeaderOffset + kMapBytes);
    std::copy(map1.begin(), map1.end(),
              output.begin() + kMapHeaderOffset + 2 * kMapBytes);
    store_u32(output, kMapHeaderOffset + kMapBytes + 4, 256);
    store_u32(output, kMapHeaderOffset + 2 * kMapBytes + 4, 512);

    const size_t group_bytes = kRecordsPerMap * kRecordBytes;
    const auto records1 = std::vector<uint8_t>(
        input.begin() + kRecordOffset + group_bytes,
        input.begin() + kRecordOffset + 2 * group_bytes);
    const auto records2 = std::vector<uint8_t>(
        input.begin() + kRecordOffset + 2 * group_bytes,
        input.begin() + kRecordOffset + 3 * group_bytes);
    std::copy(records2.begin(), records2.end(),
              output.begin() + kRecordOffset + group_bytes);
    std::copy(records1.begin(), records1.end(),
              output.begin() + kRecordOffset + 2 * group_bytes);

    // Node 1 loads plane 1 and node 2 loads plane 2. Their map indices swap.
    output[kNodeOffset + kNodeBytes + 2] = 1;
    store_u32(output, kNodeOffset + kNodeBytes + 10, 2);
    output[kNodeOffset + 2 * kNodeBytes + 2] = 2;
    store_u32(output, kNodeOffset + 2 * kNodeBytes + 10, 1);
    return output;
}

std::vector<uint8_t> change_grouped_access_arithmetic(
    const std::vector<uint8_t>& input) {
    // Header (68), three 27-byte planes, six 16-byte nodes, and the 12-byte
    // extension/map header precede the first 12-byte access record.
    constexpr size_t kRecordOffset = 68 + 3 * 27 + 6 * 16 + 12 + 3 * 12;
    std::vector<uint8_t> output = input;
    store_u32(output, kRecordOffset, 1);
    return output;
}

MetalCodecPhysicalTuple q4_physical() {
    MetalCodecPhysicalTuple physical;
    physical.layout.kind = PhysicalLayoutKind::GgufBlocked;
    physical.layout.version = 1;
    physical.layout.packing = PackingKind::Gguf;
    physical.layout.rank = 2;
    physical.layout.block_rank = 1;
    physical.layout.axis_order[0] = 0;
    physical.layout.axis_order[1] = 1;
    physical.layout.block_elements = 256;
    physical.layout.block_bytes = 144;
    physical.quantization.kind = QuantizationKind::BlockedAffine;
    physical.quantization.version = 1;
    physical.quantization.accumulation_type = ScalarType::F32;
    physical.quantization.block_elements = 256;
    physical.quantization.block_bytes = 144;
    physical.quantization.group_size = 256;
    physical.quantization.required_plane_mask = 1;
    physical.planes.push_back(
        {PlaneKind::Values, ScalarType::U8, 256, 144, 0});
    return physical;
}

MetalCodecPhysicalTuple grouped_physical() {
    MetalCodecPhysicalTuple physical;
    physical.layout.kind = PhysicalLayoutKind::GroupedAffine;
    physical.layout.version = 1;
    physical.layout.packing = PackingKind::LsbBitPacked;
    physical.layout.rank = 2;
    physical.layout.block_rank = 1;
    physical.layout.axis_order[0] = 1;
    physical.layout.axis_order[1] = 0;
    physical.layout.block_elements = 256;
    physical.layout.block_bytes = 64;
    physical.quantization.kind = QuantizationKind::BlockedAffine;
    physical.quantization.version = 1;
    physical.quantization.accumulation_type = ScalarType::F32;
    physical.quantization.scale_type = ScalarType::F16;
    physical.quantization.bias_type = ScalarType::F16;
    physical.quantization.block_elements = 256;
    physical.quantization.block_bytes = 64;
    physical.quantization.group_size = 256;
    physical.quantization.required_plane_mask = 7;
    physical.planes = {
        {PlaneKind::Values, ScalarType::U32, 256, 64, 0},
        {PlaneKind::Scales, ScalarType::F16, 256, 2, 0},
        {PlaneKind::Biases, ScalarType::F16, 256, 2, 0},
    };
    return physical;
}

MetalCodecRequirement grouped_requirement(
    const NormalizedCodecProgram& program) {
    MetalCodecRequirement requirement;
    requirement.operation_abi = 3;
    requirement.phase = 2;
    requirement.numerical_class = 1;
    requirement.shape_class = {5120, 17408, 1, 0};
    requirement.semantic_signature = program.semantic_signature;
    requirement.physical = grouped_physical();
    return requirement;
}

MetalCodecRequirement q4_requirement(const NormalizedCodecProgram& program) {
    MetalCodecRequirement requirement;
    requirement.operation_abi = 3;
    requirement.phase = 2;
    requirement.numerical_class = 1;
    requirement.shape_class = {5120, 17408, 1, 0};
    requirement.semantic_signature = program.semantic_signature;
    requirement.physical = q4_physical();
    return requirement;
}

MetalCodecDeviceFacts device() {
    return {0x3u, 1024, 32, 4, 1ull << 30};
}

MetalCodecCapability capability(const MetalCodecRequirement& requirement) {
    return {requirement, {0x1u, 32, 32, 4}, {17, 3}};
}

void test_normalization_and_provenance() {
    const auto original_bytes = make_grouped_affine_u2_codec_certificate();
    const auto reordered_bytes = reorder_grouped_maps(original_bytes);
    const auto changed_access_bytes =
        change_grouped_access_arithmetic(original_bytes);
    const CodecCertificate original = parse(original_bytes);
    const CodecCertificate reordered = parse(reordered_bytes);
    const CodecCertificate changed_access = parse(changed_access_bytes);
    CHECK(original.identity().digest != reordered.identity().digest);
    CHECK(original.identity().digest != changed_access.identity().digest);

    const auto first = normalize_codec_program(original);
    const auto second = normalize_codec_program(reordered);
    const auto changed = normalize_codec_program(changed_access);
    CHECK(std::holds_alternative<NormalizedCodecProgram>(first));
    CHECK(std::holds_alternative<NormalizedCodecProgram>(second));
    CHECK(std::holds_alternative<NormalizedCodecProgram>(changed));
    if (std::holds_alternative<NormalizedCodecProgram>(first) &&
        std::holds_alternative<NormalizedCodecProgram>(second)) {
        const auto& left = std::get<NormalizedCodecProgram>(first);
        const auto& right = std::get<NormalizedCodecProgram>(second);
        CHECK(left.semantic_signature == right.semantic_signature);
        CHECK(left.constant_words == right.constant_words);
        bool same_maps = left.access_maps.size() == right.access_maps.size();
        if (same_maps)
            for (size_t index = 0; index < left.access_maps.size(); ++index)
                same_maps = same_maps &&
                    left.access_maps[index].plane == right.access_maps[index].plane &&
                    left.access_maps[index].first == right.access_maps[index].first &&
                    left.access_maps[index].count == right.access_maps[index].count;
        bool same_accesses = left.accesses.size() == right.accesses.size();
        if (same_accesses)
            for (size_t index = 0; index < left.accesses.size(); ++index)
                same_accesses = same_accesses &&
                    left.accesses[index].byte_offset == right.accesses[index].byte_offset &&
                    left.accesses[index].bit_offset == right.accesses[index].bit_offset &&
                    left.accesses[index].width_bits == right.accesses[index].width_bits &&
                    left.accesses[index].encoding == right.accesses[index].encoding &&
                    left.accesses[index].flags == right.accesses[index].flags &&
                    left.accesses[index].value_shift == right.accesses[index].value_shift;
        CHECK(same_maps);
        CHECK(same_accesses);
        if (std::holds_alternative<NormalizedCodecProgram>(changed)) {
            CHECK(left.semantic_signature !=
                  std::get<NormalizedCodecProgram>(changed).semantic_signature);
            const MetalCodecRequirement original_requirement =
                grouped_requirement(left);
            MetalCodecRequirement changed_requirement =
                grouped_requirement(std::get<NormalizedCodecProgram>(changed));
            MetalCodecCapabilityRegistry registry;
            CompatibilityReport error;
            CHECK(registry.add({original_requirement, {1u, 32u, 32u, 4u},
                                {17, 7}}, &error));
            const auto changed_result =
                registry.resolve_portable(changed_requirement);
            CHECK(std::holds_alternative<CompatibilityReport>(changed_result));
            if (std::holds_alternative<CompatibilityReport>(changed_result))
                CHECK(std::get<CompatibilityReport>(changed_result).code ==
                      CompatibilityError::CAPABILITY_MISSING);
        }
    }
    const auto first_provenance = normalized_codec_provenance(original);
    const auto second_provenance = normalized_codec_provenance(reordered);
    CHECK(first_provenance.source_identity != second_provenance.source_identity);

    const CodecCertificate q4 = parse(make_q4_k_codec_certificate());
    const auto normalized_q4 = normalize_codec_program(q4);
    CHECK(std::holds_alternative<NormalizedCodecProgram>(normalized_q4));
    if (std::holds_alternative<NormalizedCodecProgram>(normalized_q4)) {
        const auto& program = std::get<NormalizedCodecProgram>(normalized_q4);
        CHECK(program.nodes.size() + 1 == q4.node_summaries().size());
        CHECK(program.valid());
    }
    const CodecCertificate q6 = parse(make_q6_k_codec_certificate());
    const auto normalized_q6 = normalize_codec_program(q6);
    CHECK(std::holds_alternative<NormalizedCodecProgram>(normalized_q6));
    CHECK(q6.constant_words().size() == 1);
    if (q6.constant_words().size() == 1)
        CHECK(q6.constant_words()[0] == std::bit_cast<uint32_t>(-32.0f));
}

void test_capability_matching() {
    const CodecCertificate q4 = parse(make_q4_k_codec_certificate());
    const auto normalized = normalize_codec_program(q4);
    CHECK(std::holds_alternative<NormalizedCodecProgram>(normalized));
    if (!std::holds_alternative<NormalizedCodecProgram>(normalized)) return;
    const auto& program = std::get<NormalizedCodecProgram>(normalized);
    const MetalCodecRequirement requirement = q4_requirement(program);
    const MetalCodecCapability entry = capability(requirement);

    MetalCodecCapabilityRegistry registry;
    const auto portable_empty = registry.resolve_portable(requirement);
    CHECK(std::holds_alternative<CompatibilityReport>(portable_empty));
    const auto empty_result = registry.resolve({requirement, device()});
    CHECK(std::holds_alternative<CompatibilityReport>(empty_result));
    if (std::holds_alternative<CompatibilityReport>(empty_result))
        CHECK(std::get<CompatibilityReport>(empty_result).code ==
              CompatibilityError::CAPABILITY_MISSING);
    CompatibilityReport error;
    CHECK(registry.add(entry, &error));
    CHECK(registry.size() == 1);
    const auto resolved = registry.resolve({requirement, device()});
    CHECK(std::holds_alternative<MetalCodecResolution>(resolved));
    if (std::holds_alternative<MetalCodecResolution>(resolved)) {
        const auto& value = std::get<MetalCodecResolution>(resolved);
        CHECK(value.lowering == entry.lowering);
        CHECK(value.capability_digest == metal_codec_capability_digest(entry));
    }

    MetalCodecRequirement unknown = requirement;
    unknown.semantic_signature[0] ^= 1;
    const auto portable_unknown = registry.resolve_portable(unknown);
    CHECK(std::holds_alternative<CompatibilityReport>(portable_unknown));
    if (std::holds_alternative<CompatibilityReport>(portable_unknown))
        CHECK(std::get<CompatibilityReport>(portable_unknown).code ==
              CompatibilityError::CAPABILITY_MISSING);
    const auto unknown_result = registry.resolve({unknown, device()});
    CHECK(std::holds_alternative<CompatibilityReport>(unknown_result));
    if (std::holds_alternative<CompatibilityReport>(unknown_result))
        CHECK(std::get<CompatibilityReport>(unknown_result).code ==
              CompatibilityError::CAPABILITY_MISSING);

    MetalCodecRequirement unsupported_operation = requirement;
    unsupported_operation.operation_abi = 99;
    const auto unsupported = registry.resolve({unsupported_operation, device()});
    CHECK(std::holds_alternative<CompatibilityReport>(unsupported));
    if (std::holds_alternative<CompatibilityReport>(unsupported))
        CHECK(std::get<CompatibilityReport>(unsupported).code ==
              CompatibilityError::CAPABILITY_MISSING);

    CHECK(!registry.add(entry, &error));
    CHECK(error.code == CompatibilityError::KERNEL_AMBIGUOUS);

    MetalCodecRequirement bad_physical = requirement;
    bad_physical.physical.layout.block_bytes = 145;
    const auto portable_bad_physical = registry.resolve_portable(bad_physical);
    CHECK(std::holds_alternative<CompatibilityReport>(portable_bad_physical));
    if (std::holds_alternative<CompatibilityReport>(portable_bad_physical))
        CHECK(std::get<CompatibilityReport>(portable_bad_physical).code ==
              CompatibilityError::CAPABILITY_MISSING);
    CHECK(std::holds_alternative<CompatibilityReport>(
        registry.resolve({bad_physical, device()})));
    MetalCodecRequirement bad_numerical = requirement;
    bad_numerical.numerical_class = 2;
    CHECK(std::holds_alternative<CompatibilityReport>(
        registry.resolve({bad_numerical, device()})));

    MetalCodecDeviceFacts weak_device = device();
    weak_device.feature_bits = 0;
    CHECK(std::holds_alternative<CompatibilityReport>(
        registry.resolve({requirement, weak_device})));
    weak_device = device();
    weak_device.max_threads_per_threadgroup = 16;
    CHECK(std::holds_alternative<CompatibilityReport>(
        registry.resolve({requirement, weak_device})));

    MetalCodecCapability second = entry;
    second.device.minimum_language_version = 5;
    CHECK(registry.add(second, &error));
    MetalCodecDeviceFacts ambiguous_device = device();
    ambiguous_device.language_version = 5;
    const auto ambiguous = registry.resolve({requirement, ambiguous_device});
    CHECK(std::holds_alternative<CompatibilityReport>(ambiguous));
    if (std::holds_alternative<CompatibilityReport>(ambiguous))
        CHECK(std::get<CompatibilityReport>(ambiguous).code ==
              CompatibilityError::KERNEL_AMBIGUOUS);

    MetalCodecCapability invalid = entry;
    invalid.lowering.identity = 0;
    CHECK(!registry.add(invalid, &error));
    MetalCodecQuery invalid_query{requirement, {}};
    CHECK(std::holds_alternative<CompatibilityReport>(
        registry.resolve(invalid_query)));
}

} // namespace

int main() {
    test_normalization_and_provenance();
    test_capability_matching();
    return test_summary("test_metal_codec_capability");
}
