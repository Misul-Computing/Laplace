#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

#include "normalized_codec_program.h"
#include "physical_codec.h"

namespace Laplace {

using MetalCodecCapabilityDigest = std::array<uint8_t, 32>;

// The physical tuple is the complete tensor byte contract used for matching.
// The normalized decoder signature also contains its access semantics, so a
// capability must satisfy both identities.
struct MetalCodecPhysicalTuple {
    PhysicalLayoutSchema layout{};
    PhysicalQuantizationSchema quantization{};
    std::vector<PhysicalPlaneSchema> planes;

    friend bool operator==(const MetalCodecPhysicalTuple&,
                           const MetalCodecPhysicalTuple&) = default;
};

struct MetalCodecDeviceFacts {
    uint32_t feature_bits = 0;
    uint32_t max_threads_per_threadgroup = 0;
    uint32_t thread_execution_width = 0;
    uint16_t language_version = 0;
    uint64_t max_working_set_bytes = 0;

    friend bool operator==(const MetalCodecDeviceFacts&,
                           const MetalCodecDeviceFacts&) = default;
};

struct MetalCodecDeviceRequirements {
    uint32_t required_feature_bits = 0;
    uint32_t minimum_max_threads_per_threadgroup = 0;
    uint32_t minimum_thread_execution_width = 0;
    uint16_t minimum_language_version = 0;

    friend bool operator==(const MetalCodecDeviceRequirements&,
                           const MetalCodecDeviceRequirements&) = default;
};

// This key is entirely semantic and physical. The operation value is an
// application ABI token, not a source or architecture selector.
struct MetalCodecRequirement {
    uint32_t operation_abi = 0;
    uint16_t phase = 0;
    uint16_t numerical_class = 0;
    std::array<uint32_t, 4> shape_class{};
    NormalizedCodecDigest semantic_signature{};
    MetalCodecPhysicalTuple physical;

    friend bool operator==(const MetalCodecRequirement&,
                           const MetalCodecRequirement&) = default;
};

// The lowering fields are opaque application-owned values. Packages can
// request a capability, but cannot supply an executable handle.
struct MetalCodecLowering {
    uint64_t identity = 0;
    uint32_t strategy = 0;

    friend bool operator==(const MetalCodecLowering&,
                           const MetalCodecLowering&) = default;
};

// These strategies are application-owned semantic lowering contracts. They
// name a known decoder implementation without naming a source format or
// model. The structural compiler maps only the strategies it implements in
// this slice; unknown strategies fail closed.
enum class MetalCodecLoweringStrategy : uint32_t {
    StructuralEmbeddingF16 = 1,
    StructuralEmbeddingQ4 = 2,
    StructuralEmbeddingQ6 = 3,
    StructuralGemvF16 = 4,
    StructuralGemvQ4 = 5,
    StructuralGemvQ6 = 6,
    StructuralGemvAffineU2 = 7,
    StructuralGemvColumnGroupedU2 = 8,
    StructuralGemvF32 = 9,
    StructuralTensorF32 = 10,
};

struct MetalCodecCapability {
    MetalCodecRequirement requirement;
    MetalCodecDeviceRequirements device;
    MetalCodecLowering lowering;
};

struct MetalCodecQuery {
    MetalCodecRequirement requirement;
    MetalCodecDeviceFacts device;
};

struct MetalCodecResolution {
    MetalCodecCapabilityDigest capability_digest{};
    MetalCodecLowering lowering;
};

using MetalCodecResolutionResult =
    std::variant<MetalCodecResolution, CompatibilityReport>;

class MetalCodecCapabilityRegistry {
public:
    bool add(const MetalCodecCapability& capability,
             CompatibilityReport* error = nullptr);
    size_t size() const noexcept { return capabilities_.size(); }

    // Resolves a portable recipe without inventing or assuming device facts.
    // A requirement with multiple application-owned recipes is ambiguous and
    // must be resolved again only after a real same-device query.
    MetalCodecResolutionResult resolve_portable(
        const MetalCodecRequirement& requirement) const;
    MetalCodecResolutionResult resolve(const MetalCodecQuery& query) const;

private:
    std::vector<MetalCodecCapability> capabilities_;
};

MetalCodecCapabilityDigest metal_codec_capability_digest(
    const MetalCodecCapability& capability);

} // namespace Laplace
