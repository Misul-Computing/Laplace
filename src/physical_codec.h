#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "semantic_model.h"

namespace Laplace {

using PhysicalIdentityDigest = std::array<uint8_t, 32>;
inline constexpr size_t kPhysicalCodecRegistryMaximumCodecs = 1024;
inline constexpr size_t kPhysicalCodecRegistryMaximumTensors = 65536;
inline constexpr size_t kPhysicalCodecRegistryMaximumCertificateBytes = 64 * 1024;
inline constexpr size_t kPhysicalCodecRegistryMaximumCertificateTableBytes = 8 * 1024 * 1024;

// Immutable codec identity. Tensor extents, strides, plane spans, artifact
// locations, and device facts belong to the tensor binding or kernel query.
struct PhysicalLayoutSchema {
    PhysicalLayoutKind kind = PhysicalLayoutKind::ContiguousRowMajor;
    uint16_t version = 1;
    PackingKind packing = PackingKind::None;
    uint8_t rank = 0;
    uint8_t block_rank = 0;
    std::array<uint8_t, 8> axis_order = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    uint32_t block_elements = 0;
    uint32_t block_bytes = 0;
    uint32_t flags = 0;
    friend bool operator==(const PhysicalLayoutSchema&, const PhysicalLayoutSchema&) = default;
};

struct PhysicalQuantizationSchema {
    QuantizationKind kind = QuantizationKind::None;
    uint16_t version = 1;
    ScalarType accumulation_type = ScalarType::F32;
    ScalarType scale_type = static_cast<ScalarType>(0);
    ScalarType zero_type = static_cast<ScalarType>(0);
    ScalarType bias_type = static_cast<ScalarType>(0);
    uint32_t block_elements = 0;
    uint32_t block_bytes = 0;
    uint32_t group_size = 0;
    uint32_t required_plane_mask = 0;
    uint32_t flags = 0;
    friend bool operator==(const PhysicalQuantizationSchema&, const PhysicalQuantizationSchema&) = default;
};

struct PhysicalPlaneSchema {
    PlaneKind kind = PlaneKind::Values;
    ScalarType storage_type = ScalarType::F32;
    uint32_t logical_elements_covered = 0;
    uint32_t bytes_per_block = 0;
    uint32_t flags = 0;
    friend bool operator==(const PhysicalPlaneSchema&, const PhysicalPlaneSchema&) = default;
};

struct PhysicalCodecIdentity {
    uint16_t identity_version = 1;
    uint16_t arithmetic_version = 0;
    PhysicalIdentityDigest arithmetic_digest{};
    PhysicalIdentityDigest codebook_digest{};
    PhysicalLayoutSchema layout;
    PhysicalQuantizationSchema quantization;
    std::vector<PhysicalPlaneSchema> planes;
    friend bool operator==(const PhysicalCodecIdentity&, const PhysicalCodecIdentity&) = default;
};

struct PhysicalCodecSpec {
    PhysicalCodecIdentity identity;
    // Canonical bounded decode authority. Empty bytes retain a diagnostic
    // reference only. Product authority requires bytes whose digest is the
    // identity arithmetic digest.
    std::vector<uint8_t> certificate_bytes;
    friend bool operator==(const PhysicalCodecSpec&, const PhysicalCodecSpec&) = default;
};

struct PhysicalTensorCodecDeclaration {
    uint32_t tensor_id = 0;
    PhysicalCodecIdentity identity;
    friend bool operator==(const PhysicalTensorCodecDeclaration&,
                           const PhysicalTensorCodecDeclaration&) = default;
};

struct PhysicalCodecRegistry {
    std::vector<PhysicalCodecSpec> codecs;
    std::vector<PhysicalTensorCodecDeclaration> tensors;
    friend bool operator==(const PhysicalCodecRegistry&, const PhysicalCodecRegistry&) = default;
};

using PhysicalCodecRegistryEncodeResult = std::variant<std::vector<uint8_t>, CompatibilityReport>;
using PhysicalCodecRegistryDecodeResult = std::variant<PhysicalCodecRegistry, CompatibilityReport>;

bool valid_physical_codec_identity(const PhysicalCodecIdentity& identity);
bool validate_physical_codec_registry(const PhysicalCodecRegistry& registry,
                                      bool require_nonempty = false);
bool physical_codec_registry_is_canonical(const PhysicalCodecRegistry& registry);
bool physical_codec_registry_covers_model(const PhysicalCodecRegistry& registry,
                                          const SemanticModel& model);
bool physical_codec_registry_matches_model(const PhysicalCodecRegistry& registry,
                                           const SemanticModel& model);
bool physical_codec_identity_less(const PhysicalCodecIdentity& left,
                                  const PhysicalCodecIdentity& right);

std::optional<PhysicalCodecIdentity> physical_codec_identity(
    const SemanticTensor& tensor, uint16_t arithmetic_version,
    const PhysicalIdentityDigest& arithmetic_digest,
    const PhysicalIdentityDigest& codebook_digest = {});

PhysicalCodecRegistryEncodeResult encode_physical_codec_registry(
    const PhysicalCodecRegistry& registry);
PhysicalCodecRegistryDecodeResult decode_physical_codec_registry(
    std::span<const uint8_t> bytes);

// Returns zero when the registry is explicitly codec-empty. For a non-empty
// registry this digest covers codec arithmetic identities and every semantic
// tensor declaration and binding, including dimensions, locations, and planes.
Sha256Digest physical_codec_registry_digest(const PhysicalCodecRegistry& registry,
                                            const SemanticModel& model);

} // namespace Laplace
