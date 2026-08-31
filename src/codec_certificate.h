#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace Laplace {

struct PhysicalCodecIdentity;
class CodecCertificateFactory;

using CodecCertificateDigest = std::array<uint8_t, 32>;

enum class CodecCertificateScalar : uint8_t {
    Binary16 = 1,
    Binary32 = 2,
    PackedUnsigned = 3,
};

// Storage describes the byte container. Scalar describes the value produced
// by an access expression. Keeping these domains separate permits one source
// plane to contain binary16 parameters, signed scales, and packed values.
enum class CodecCertificateStorageScalar : uint8_t {
    Binary16 = 1,
    Binary32 = 2,
    Unsigned8 = 3,
    Unsigned16 = 4,
    Unsigned32 = 5,
    Signed8 = 6,
};

enum class CodecCertificatePlaneRole : uint8_t {
    Values = 1,
    Scales = 2,
    Biases = 3,
    Indexes = 4,
};

enum class CodecCertificateSpecialValuePolicy : uint8_t {
    PreserveIeee = 1,
};

// These views expose the normalized certificate graph. They are structural
// metadata for a compiler; they are not format or family selectors.
enum class CodecCertificateNodeOperation : uint8_t {
    LoadScalar = 1,
    LoadBits = 2,
    CastFloat = 3,
    Add = 4,
    Sub = 5,
    Mul = 6,
    Fma = 7,
    Neg = 8,
    Constant = 9,
};

enum class CodecCertificateNodeValueType : uint8_t {
    Unsigned = 1,
    Float = 2,
    Signed = 3,
};

enum class CodecCertificateAccessEncoding : uint8_t {
    Binary16 = 1,
    Binary32 = 2,
    Unsigned8 = 3,
    Unsigned16 = 4,
    Unsigned32 = 5,
    Signed8 = 6,
};

struct CodecCertificateNodeSummary {
    CodecCertificateNodeOperation operation =
        CodecCertificateNodeOperation::LoadScalar;
    CodecCertificateNodeValueType value_type =
        CodecCertificateNodeValueType::Unsigned;
    uint8_t plane = 0xff;
    uint8_t flags = 0;
    uint16_t argument0 = 0xffff;
    uint16_t argument1 = 0xffff;
    uint16_t argument2 = 0xffff;
    uint32_t immediate = 0;
};

struct CodecCertificateAccessMapSummary {
    uint8_t plane = 0;
    uint32_t first = 0;
    uint32_t count = 0;
};

struct CodecCertificateAccessSummary {
    uint32_t byte_offset = 0;
    uint8_t bit_offset = 0;
    uint8_t width_bits = 0;
    CodecCertificateAccessEncoding encoding =
        CodecCertificateAccessEncoding::Unsigned8;
    uint8_t flags = 0;
    uint8_t value_shift = 0;
};

enum class CodecCertificateError : uint8_t {
    None = 0,
    TooLarge = 1,
    InvalidMagic = 2,
    UnsupportedVersion = 3,
    NonCanonical = 4,
    ResourceLimit = 5,
    InvalidTopology = 6,
    InvalidPlane = 7,
    InvalidExpression = 8,
    InvalidSpecialValuePolicy = 9,
    InvalidPlaneBinding = 10,
    InvalidUnitCount = 11,
    InvalidOffset = 12,
    InvalidLength = 13,
    InvalidStride = 14,
    AccessOutOfBounds = 15,
    ArithmeticOverflow = 16,
    DecodeLimit = 17,
    UnsupportedEncoding = 18,
    InvalidAccessMap = 19,
    InvalidNodeType = 20,
};

struct CodecCertificateIdentity {
    uint16_t abi_version = 0;
    CodecCertificateDigest digest{};
    friend bool operator==(const CodecCertificateIdentity&,
                           const CodecCertificateIdentity&) = default;
};

struct CodecCertificateSummary {
    uint16_t abi_version = 0;
    uint16_t certificate_version = 0;
    CodecCertificateScalar source_scalar = CodecCertificateScalar::Binary16;
    CodecCertificateScalar output_scalar = CodecCertificateScalar::Binary32;
    CodecCertificateSpecialValuePolicy special_value_policy =
        CodecCertificateSpecialValuePolicy::PreserveIeee;
    uint32_t unit_elements = 0;
    uint32_t unit_bytes = 0;
    uint32_t maximum_units = 0;
    uint32_t plane_count = 0;
    uint32_t node_count = 0;
    uint32_t constant_count = 0;
    uint32_t expression_depth = 0;
    bool rank_independent = false;
    // Complete physical tuple required by the certificate. These fields are
    // wire values so this header does not duplicate the semantic-model ABI.
    uint8_t physical_layout_kind = 0;
    uint8_t physical_layout_packing = 0;
    uint8_t physical_layout_block_rank = 0;
    uint32_t physical_layout_block_elements = 0;
    uint32_t physical_layout_block_bytes = 0;
    uint8_t physical_quantization_kind = 0;
    uint32_t physical_quantization_block_elements = 0;
    uint32_t physical_quantization_block_bytes = 0;
    uint32_t physical_quantization_group_size = 0;
    uint32_t physical_quantization_required_plane_mask = 0;
};

struct CodecCertificatePlaneSummary {
    CodecCertificatePlaneRole role = CodecCertificatePlaneRole::Values;
    CodecCertificateScalar scalar = CodecCertificateScalar::Binary16;
    CodecCertificateStorageScalar storage_scalar =
        CodecCertificateStorageScalar::Binary16;
    uint8_t bit_order = 0;
    uint8_t byte_order = 0;
    uint16_t width_bits = 0;
    uint32_t elements_per_unit = 0;
    uint32_t bytes_per_unit = 0;
    uint32_t alignment = 0;
    uint32_t base = 0;
    uint32_t stride = 0;
};

struct CodecCertificatePlaneBinding {
    std::span<const uint8_t> storage{};
    uint64_t offset = 0;
    uint64_t length = 0;
    uint64_t stride = 0;
};

struct CodecCertificateBinding {
    uint64_t unit_count = 0;
    std::vector<CodecCertificatePlaneBinding> planes;
};

using CodecCertificateDecodeResult =
    std::variant<std::vector<float>, CodecCertificateError>;

class CodecCertificate {
public:
    CodecCertificate() = default;

    const CodecCertificateIdentity& identity() const noexcept { return identity_; }
    std::span<const uint8_t> canonical_bytes() const noexcept { return canonical_bytes_; }
    const CodecCertificateSummary& summary() const noexcept { return summary_; }
    const std::vector<CodecCertificatePlaneSummary>& plane_summaries() const noexcept {
        return plane_summaries_;
    }
    const std::vector<CodecCertificateNodeSummary>& node_summaries() const noexcept {
        return node_summaries_;
    }
    const std::vector<uint32_t>& constant_words() const noexcept {
        return constant_words_;
    }
    const std::vector<CodecCertificateAccessMapSummary>&
    access_map_summaries() const noexcept {
        return access_map_summaries_;
    }
    const std::vector<CodecCertificateAccessSummary>&
    access_summaries() const noexcept {
        return access_summaries_;
    }

    CodecCertificateError validate(const CodecCertificateBinding&) const noexcept;
    bool matches_physical_identity(const PhysicalCodecIdentity&) const noexcept;
    CodecCertificateDecodeResult decode(const CodecCertificateBinding&,
                                        uint64_t max_elements) const;

private:
    CodecCertificate(CodecCertificateIdentity identity,
                     CodecCertificateSummary summary,
                     std::vector<uint8_t> canonical_bytes,
                     std::vector<CodecCertificatePlaneSummary> plane_summaries,
                     std::vector<CodecCertificateNodeSummary> node_summaries,
                     std::vector<uint32_t> constant_words,
                     std::vector<CodecCertificateAccessMapSummary>
                         access_map_summaries,
                     std::vector<CodecCertificateAccessSummary> access_summaries)
        : identity_(identity), summary_(summary),
          canonical_bytes_(std::move(canonical_bytes)),
          plane_summaries_(std::move(plane_summaries)),
          node_summaries_(std::move(node_summaries)),
          constant_words_(std::move(constant_words)),
          access_map_summaries_(std::move(access_map_summaries)),
          access_summaries_(std::move(access_summaries)) {}

    CodecCertificateIdentity identity_;
    CodecCertificateSummary summary_;
    std::vector<uint8_t> canonical_bytes_;
    std::vector<CodecCertificatePlaneSummary> plane_summaries_;
    std::vector<CodecCertificateNodeSummary> node_summaries_;
    std::vector<uint32_t> constant_words_;
    std::vector<CodecCertificateAccessMapSummary> access_map_summaries_;
    std::vector<CodecCertificateAccessSummary> access_summaries_;

    friend class CodecCertificateFactory;
};

using CodecCertificateParseResult =
    std::variant<CodecCertificate, CodecCertificateError>;

CodecCertificateParseResult parse_codec_certificate(std::span<const uint8_t> bytes);

CodecCertificateDigest codec_certificate_digest(std::span<const uint8_t> bytes);

std::vector<uint8_t> make_raw_f16_codec_certificate();
std::vector<uint8_t> make_raw_f32_codec_certificate();
std::vector<uint8_t> make_q4_k_codec_certificate();
std::vector<uint8_t> make_q5_0_codec_certificate();
std::vector<uint8_t> make_q6_k_codec_certificate();
std::vector<uint8_t> make_q8_0_codec_certificate();
std::vector<uint8_t> make_grouped_affine_u2_codec_certificate();

} // namespace Laplace
