#pragma once

#include <array>
#include <cstdint>
#include <variant>
#include <vector>

#include "codec_certificate.h"
#include "compatibility_report.h"

namespace Laplace {

// Qualification adapter for the old category-bearing certificate grammar.
// Product physical programs use physical_program.h. This adapter remains only
// while existing certificate fixtures are translated.
using NormalizedCodecDigest = std::array<uint8_t, 32>;

enum class NormalizedCodecRounding : uint8_t {
    ExactNearestEven = 1,
};

struct NormalizedCodecProgram {
    uint16_t abi_version = 0;
    uint16_t certificate_version = 0;
    CodecCertificateSummary summary{};
    std::vector<CodecCertificatePlaneSummary> planes;
    std::vector<CodecCertificateNodeSummary> nodes;
    std::vector<CodecCertificateAccessMapSummary> access_maps;
    std::vector<CodecCertificateAccessSummary> accesses;
    std::vector<uint32_t> constant_words;
    NormalizedCodecRounding rounding =
        NormalizedCodecRounding::ExactNearestEven;
    NormalizedCodecDigest semantic_signature{};

    bool valid() const noexcept {
        return semantic_signature != NormalizedCodecDigest{};
    }
};

struct NormalizedCodecProvenance {
    uint16_t abi_version = 0;
    CodecCertificateDigest source_identity{};
};

using NormalizedCodecProgramResult =
    std::variant<NormalizedCodecProgram, CompatibilityReport>;

NormalizedCodecProgramResult normalize_codec_program(
    const CodecCertificate& certificate,
    NormalizedCodecRounding rounding =
        NormalizedCodecRounding::ExactNearestEven);

NormalizedCodecProvenance normalized_codec_provenance(
    const CodecCertificate& certificate) noexcept;

} // namespace Laplace
