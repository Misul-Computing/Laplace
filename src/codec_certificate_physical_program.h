#pragma once

#include "codec_certificate.h"
#include "physical_program.h"

namespace Laplace {

struct TranslatedCodecCertificate {
    PhysicalProgram program;
    uint64_t required_units = 0;
    // Indexed by canonical physical plane. Inline planes use kNoPhysicalPlane.
    std::vector<uint16_t> source_planes;
};

using CodecCertificatePhysicalResult =
    std::variant<TranslatedCodecCertificate, CompatibilityReport>;

// Compile the certificate grammar, independent of its physical identity.
// The caller must retain source binding validation (including unit composition)
// and verify the result against the same logical type and external byte spans.
// Logical strides are measured in decoded certificate elements, with origin 0.
// required_units includes padding and partial final units touched by that map.
CodecCertificatePhysicalResult translate_codec_certificate(
    const CodecCertificate& certificate, const LogicalTensorType& logical,
    std::span<const uint64_t> logical_element_strides,
    std::span<const uint64_t> plane_byte_strides);

} // namespace Laplace
