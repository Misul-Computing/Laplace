#include "metal_codec_capability.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

namespace Laplace {
namespace {

constexpr size_t kMaximumPlanes = 32;

CompatibilityReport report(CompatibilityError code, const char* detail) {
    return compatibility_report(code, detail);
}

bool zero_digest(const NormalizedCodecDigest& digest) noexcept {
    return std::all_of(digest.begin(), digest.end(),
                       [](uint8_t value) { return value == 0; });
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
void append_digest(std::vector<uint8_t>& bytes,
                   const NormalizedCodecDigest& digest) {
    bytes.insert(bytes.end(), digest.begin(), digest.end());
}

void append_physical(std::vector<uint8_t>& bytes,
                     const MetalCodecPhysicalTuple& physical) {
    append_u16(bytes, static_cast<uint16_t>(physical.layout.kind));
    append_u16(bytes, physical.layout.version);
    append_u16(bytes, static_cast<uint16_t>(physical.layout.packing));
    append_u8(bytes, physical.layout.rank);
    append_u8(bytes, physical.layout.block_rank);
    for (uint8_t axis : physical.layout.axis_order) append_u8(bytes, axis);
    append_u32(bytes, physical.layout.block_elements);
    append_u32(bytes, physical.layout.block_bytes);
    append_u32(bytes, physical.layout.flags);

    append_u16(bytes, static_cast<uint16_t>(physical.quantization.kind));
    append_u16(bytes, physical.quantization.version);
    append_u16(bytes,
                static_cast<uint16_t>(physical.quantization.accumulation_type));
    append_u16(bytes, static_cast<uint16_t>(physical.quantization.scale_type));
    append_u16(bytes, static_cast<uint16_t>(physical.quantization.zero_type));
    append_u16(bytes, static_cast<uint16_t>(physical.quantization.bias_type));
    append_u32(bytes, physical.quantization.block_elements);
    append_u32(bytes, physical.quantization.block_bytes);
    append_u32(bytes, physical.quantization.group_size);
    append_u32(bytes, physical.quantization.required_plane_mask);
    append_u32(bytes, physical.quantization.flags);

    append_u32(bytes, static_cast<uint32_t>(physical.planes.size()));
    for (const PhysicalPlaneSchema& plane : physical.planes) {
        append_u16(bytes, static_cast<uint16_t>(plane.kind));
        append_u16(bytes, static_cast<uint16_t>(plane.storage_type));
        append_u32(bytes, plane.logical_elements_covered);
        append_u32(bytes, plane.bytes_per_block);
        append_u32(bytes, plane.flags);
    }
}

bool valid_physical(const MetalCodecPhysicalTuple& physical) noexcept {
    return static_cast<uint16_t>(physical.layout.kind) != 0 &&
           physical.layout.version != 0 &&
           static_cast<uint16_t>(physical.layout.packing) <= 2 &&
           physical.layout.rank != 0 && physical.layout.rank <= 8 &&
           physical.layout.block_rank <= 8 &&
           static_cast<uint16_t>(physical.quantization.kind) <= 2 &&
           physical.quantization.version != 0 && !physical.planes.empty() &&
           physical.planes.size() <= kMaximumPlanes;
}

bool valid_requirement(const MetalCodecRequirement& requirement) noexcept {
    return requirement.operation_abi != 0 && requirement.phase != 0 &&
           requirement.numerical_class != 0 && requirement.shape_class[0] != 0 &&
           !zero_digest(requirement.semantic_signature) &&
           valid_physical(requirement.physical);
}

bool valid_device_requirements(
    const MetalCodecDeviceRequirements& device) noexcept {
    return device.minimum_max_threads_per_threadgroup != 0 &&
           device.minimum_thread_execution_width != 0 &&
           device.minimum_language_version != 0;
}

bool valid_device(const MetalCodecDeviceFacts& device) noexcept {
    return device.max_threads_per_threadgroup != 0 &&
           device.thread_execution_width != 0 && device.language_version != 0;
}

bool supports(const MetalCodecDeviceFacts& facts,
              const MetalCodecDeviceRequirements& requirements) noexcept {
    return (facts.feature_bits & requirements.required_feature_bits) ==
               requirements.required_feature_bits &&
           facts.max_threads_per_threadgroup >=
               requirements.minimum_max_threads_per_threadgroup &&
           facts.thread_execution_width >=
               requirements.minimum_thread_execution_width &&
           facts.language_version >= requirements.minimum_language_version;
}

void append_requirement(std::vector<uint8_t>& bytes,
                        const MetalCodecRequirement& requirement) {
    append_u32(bytes, requirement.operation_abi);
    append_u16(bytes, requirement.phase);
    append_u16(bytes, requirement.numerical_class);
    for (uint32_t value : requirement.shape_class) append_u32(bytes, value);
    append_digest(bytes, requirement.semantic_signature);
    append_physical(bytes, requirement.physical);
}

} // namespace

MetalCodecCapabilityDigest metal_codec_capability_digest(
    const MetalCodecCapability& capability) {
    std::vector<uint8_t> bytes;
    bytes.reserve(256 + capability.requirement.physical.planes.size() * 16);
    const std::array<uint8_t, 8> domain =
        {'L', 'A', 'P', 'C', 'A', 'P', 1, 0};
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    append_requirement(bytes, capability.requirement);
    append_u32(bytes, capability.device.required_feature_bits);
    append_u32(bytes, capability.device.minimum_max_threads_per_threadgroup);
    append_u32(bytes, capability.device.minimum_thread_execution_width);
    append_u16(bytes, capability.device.minimum_language_version);
    append_u16(bytes, 0);
    append_u64(bytes, capability.lowering.identity);
    append_u32(bytes, capability.lowering.strategy);
    append_u32(bytes, 0);
    MetalCodecCapabilityDigest digest{};
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
    return digest;
}

bool MetalCodecCapabilityRegistry::add(
    const MetalCodecCapability& capability, CompatibilityReport* error) {
    const auto fail = [&](CompatibilityError code, const char* detail) {
        if (error) *error = report(code, detail);
        return false;
    };
    if (!valid_requirement(capability.requirement))
        return fail(CompatibilityError::IR_CONSTRAINT_FAILED,
                    "capability requirement is invalid");
    if (!valid_device_requirements(capability.device))
        return fail(CompatibilityError::CAPABILITY_MISSING,
                    "capability device requirements are invalid");
    if (capability.lowering.identity == 0 || capability.lowering.strategy == 0)
        return fail(CompatibilityError::CAPABILITY_MISSING,
                    "capability lowering is empty");
    if (std::any_of(capabilities_.begin(), capabilities_.end(),
                    [&](const MetalCodecCapability& existing) {
                        return existing.requirement == capability.requirement &&
                               existing.device == capability.device;
                    }))
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "capability requirement is duplicated");
    try {
        capabilities_.push_back(capability);
        return true;
    } catch (const std::bad_alloc&) {
        return fail(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                    "capability registry allocation failed");
    }
}

MetalCodecResolutionResult MetalCodecCapabilityRegistry::resolve(
    const MetalCodecQuery& query) const {
    if (!valid_requirement(query.requirement) || !valid_device(query.device))
        return report(CompatibilityError::CAPABILITY_MISSING,
                      "capability query is invalid");
    const MetalCodecCapability* match = nullptr;
    for (const MetalCodecCapability& capability : capabilities_) {
        if (!(capability.requirement == query.requirement) ||
            !supports(query.device, capability.device))
            continue;
        if (match != nullptr)
            return report(CompatibilityError::KERNEL_AMBIGUOUS,
                          "capability query has multiple matches");
        match = &capability;
    }
    if (match == nullptr)
        return report(CompatibilityError::CAPABILITY_MISSING,
                      "capability query has no match");
    return MetalCodecResolution{metal_codec_capability_digest(*match),
                                match->lowering};
}

MetalCodecResolutionResult MetalCodecCapabilityRegistry::resolve_portable(
    const MetalCodecRequirement& requirement) const {
    if (!valid_requirement(requirement))
        return report(CompatibilityError::CAPABILITY_MISSING,
                      "portable capability requirement is invalid");
    const MetalCodecCapability* match = nullptr;
    for (const MetalCodecCapability& capability : capabilities_) {
        if (!(capability.requirement == requirement)) continue;
        if (match != nullptr)
            return report(CompatibilityError::KERNEL_AMBIGUOUS,
                          "portable capability requirement has multiple matches");
        match = &capability;
    }
    if (match == nullptr)
        return report(CompatibilityError::CAPABILITY_MISSING,
                      "portable capability requirement has no match");
    return MetalCodecResolution{metal_codec_capability_digest(*match),
                                match->lowering};
}

} // namespace Laplace
