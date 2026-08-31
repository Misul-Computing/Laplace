#include "physical_codec.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

namespace Laplace {

namespace {

bool digest_is_zero(const PhysicalIdentityDigest& digest) {
    return std::all_of(digest.begin(), digest.end(), [](uint8_t byte) { return byte == 0; });
}

bool valid_scalar_type(ScalarType type) {
    switch (type) {
    case ScalarType::F32:
    case ScalarType::F16:
    case ScalarType::U32:
    case ScalarType::I32:
    case ScalarType::U8:
        return true;
    }
    return false;
}

uint32_t scalar_byte_width(ScalarType type) {
    switch (type) {
    case ScalarType::F32:
    case ScalarType::U32:
    case ScalarType::I32:
        return 4;
    case ScalarType::F16:
        return 2;
    case ScalarType::U8:
        return 1;
    }
    return 0;
}

bool valid_plane_kind(PlaneKind kind) {
    switch (kind) {
    case PlaneKind::Values:
    case PlaneKind::Scales:
    case PlaneKind::Biases:
    case PlaneKind::Zeros:
    case PlaneKind::Indexes:
    case PlaneKind::LayoutMetadata:
        return true;
    }
    return false;
}

bool valid_layout_kind(PhysicalLayoutKind kind) {
    switch (kind) {
    case PhysicalLayoutKind::ContiguousRowMajor:
    case PhysicalLayoutKind::GgufBlocked:
    case PhysicalLayoutKind::GroupedAffine:
    case PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip:
        return true;
    }
    return false;
}

bool valid_packing_kind(PackingKind kind) {
    switch (kind) {
    case PackingKind::None:
    case PackingKind::Gguf:
    case PackingKind::LsbBitPacked:
        return true;
    }
    return false;
}

bool valid_quantization_kind(QuantizationKind kind) {
    switch (kind) {
    case QuantizationKind::None:
    case QuantizationKind::BlockedAffine:
    case QuantizationKind::Codebook:
        return true;
    }
    return false;
}

bool power_of_two(uint32_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

uint32_t plane_mask(PlaneKind kind) {
    return 1u << (static_cast<uint16_t>(kind) - 1);
}

bool checked_multiply(uint64_t left, uint64_t right, uint64_t& result) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) return false;
    result = left * right;
    return true;
}

bool valid_tensor_plane_spans(const SemanticTensor& tensor) {
    for (const TensorPlane& plane : tensor.planes) {
        if (!power_of_two(plane.alignment) || plane.length == 0 ||
            plane.offset > UINT64_MAX - plane.length ||
            plane.offset % plane.alignment != 0) return false;
    }
    for (size_t left = 0; left != tensor.planes.size(); ++left) {
        const uint64_t left_end = tensor.planes[left].offset + tensor.planes[left].length;
        for (size_t right = left + 1; right != tensor.planes.size(); ++right) {
            if (tensor.planes[left].artifact_id != tensor.planes[right].artifact_id) continue;
            const uint64_t right_end = tensor.planes[right].offset + tensor.planes[right].length;
            if (tensor.planes[left].offset < right_end &&
                tensor.planes[right].offset < left_end) return false;
        }
    }
    return true;
}

template<class T>
int compare_value(const T& left, const T& right) {
    if (left < right) return -1;
    if (right < left) return 1;
    return 0;
}

template<class T>
int compare_enum(T left, T right) {
    return compare_value(static_cast<uint64_t>(left), static_cast<uint64_t>(right));
}

int compare_plane_schema(const PhysicalPlaneSchema& left, const PhysicalPlaneSchema& right) {
    if (const int value = compare_enum(left.kind, right.kind)) return value;
    if (const int value = compare_enum(left.storage_type, right.storage_type)) return value;
    if (const int value = compare_value(left.logical_elements_covered,
                                        right.logical_elements_covered)) return value;
    if (const int value = compare_value(left.bytes_per_block, right.bytes_per_block)) return value;
    return compare_value(left.flags, right.flags);
}

int compare_codec_identity(const PhysicalCodecIdentity& left,
                           const PhysicalCodecIdentity& right) {
    if (const int value = compare_value(left.identity_version, right.identity_version)) return value;
    if (const int value = compare_value(left.arithmetic_version, right.arithmetic_version)) return value;
    if (const int value = compare_value(left.arithmetic_digest, right.arithmetic_digest)) return value;
    if (const int value = compare_value(left.codebook_digest, right.codebook_digest)) return value;
    if (const int value = compare_enum(left.layout.kind, right.layout.kind)) return value;
    if (const int value = compare_value(left.layout.version, right.layout.version)) return value;
    if (const int value = compare_enum(left.layout.packing, right.layout.packing)) return value;
    if (const int value = compare_value(left.layout.rank, right.layout.rank)) return value;
    if (const int value = compare_value(left.layout.block_rank, right.layout.block_rank)) return value;
    if (const int value = compare_value(left.layout.axis_order, right.layout.axis_order)) return value;
    if (const int value = compare_value(left.layout.block_elements, right.layout.block_elements)) return value;
    if (const int value = compare_value(left.layout.block_bytes, right.layout.block_bytes)) return value;
    if (const int value = compare_value(left.layout.flags, right.layout.flags)) return value;
    if (const int value = compare_enum(left.quantization.kind, right.quantization.kind)) return value;
    if (const int value = compare_value(left.quantization.version, right.quantization.version)) return value;
    if (const int value = compare_enum(left.quantization.accumulation_type,
                                       right.quantization.accumulation_type)) return value;
    if (const int value = compare_enum(left.quantization.scale_type,
                                       right.quantization.scale_type)) return value;
    if (const int value = compare_enum(left.quantization.zero_type,
                                       right.quantization.zero_type)) return value;
    if (const int value = compare_enum(left.quantization.bias_type,
                                       right.quantization.bias_type)) return value;
    if (const int value = compare_value(left.quantization.block_elements,
                                        right.quantization.block_elements)) return value;
    if (const int value = compare_value(left.quantization.block_bytes,
                                        right.quantization.block_bytes)) return value;
    if (const int value = compare_value(left.quantization.group_size,
                                        right.quantization.group_size)) return value;
    if (const int value = compare_value(left.quantization.required_plane_mask,
                                        right.quantization.required_plane_mask)) return value;
    if (const int value = compare_value(left.quantization.flags, right.quantization.flags)) return value;
    const size_t common = std::min(left.planes.size(), right.planes.size());
    for (size_t index = 0; index != common; ++index) {
        if (const int value = compare_plane_schema(left.planes[index], right.planes[index])) return value;
    }
    return compare_value(left.planes.size(), right.planes.size());
}

void append_u8(std::vector<uint8_t>& bytes, uint8_t value) { bytes.push_back(value); }

void append_u16(std::vector<uint8_t>& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8));
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8) bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8) bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void append_digest(std::vector<uint8_t>& bytes, const PhysicalIdentityDigest& digest) {
    bytes.insert(bytes.end(), digest.begin(), digest.end());
}

void append_identity(std::vector<uint8_t>& bytes, const PhysicalCodecIdentity& identity) {
    append_u16(bytes, identity.identity_version);
    append_u16(bytes, identity.arithmetic_version);
    append_digest(bytes, identity.arithmetic_digest);
    append_digest(bytes, identity.codebook_digest);
    append_u16(bytes, static_cast<uint16_t>(identity.layout.kind));
    append_u16(bytes, identity.layout.version);
    append_u16(bytes, static_cast<uint16_t>(identity.layout.packing));
    append_u8(bytes, identity.layout.rank);
    append_u8(bytes, identity.layout.block_rank);
    for (uint8_t axis : identity.layout.axis_order) append_u8(bytes, axis);
    append_u32(bytes, identity.layout.block_elements);
    append_u32(bytes, identity.layout.block_bytes);
    append_u32(bytes, identity.layout.flags);
    append_u16(bytes, static_cast<uint16_t>(identity.quantization.kind));
    append_u16(bytes, identity.quantization.version);
    append_u16(bytes, static_cast<uint16_t>(identity.quantization.accumulation_type));
    append_u16(bytes, static_cast<uint16_t>(identity.quantization.scale_type));
    append_u16(bytes, static_cast<uint16_t>(identity.quantization.zero_type));
    append_u16(bytes, static_cast<uint16_t>(identity.quantization.bias_type));
    append_u32(bytes, identity.quantization.block_elements);
    append_u32(bytes, identity.quantization.block_bytes);
    append_u32(bytes, identity.quantization.group_size);
    append_u32(bytes, identity.quantization.required_plane_mask);
    append_u32(bytes, identity.quantization.flags);
    append_u32(bytes, static_cast<uint32_t>(identity.planes.size()));
    for (const PhysicalPlaneSchema& plane : identity.planes) {
        append_u16(bytes, static_cast<uint16_t>(plane.kind));
        append_u16(bytes, static_cast<uint16_t>(plane.storage_type));
        append_u32(bytes, plane.logical_elements_covered);
        append_u32(bytes, plane.bytes_per_block);
        append_u32(bytes, plane.flags);
    }
}

class Reader {
public:
    explicit Reader(std::span<const uint8_t> bytes) : bytes_(bytes) {}

    size_t remaining() const noexcept { return bytes_.size() - offset_; }

    bool take(size_t count, std::span<const uint8_t>& result) {
        if (count > remaining()) return false;
        result = bytes_.subspan(offset_, count);
        offset_ += count;
        return true;
    }

    bool u8(uint8_t& result) {
        if (remaining() == 0) return false;
        result = bytes_[offset_++];
        return true;
    }

    bool u16(uint16_t& result) {
        std::span<const uint8_t> value;
        if (!take(2, value)) return false;
        result = static_cast<uint16_t>(value[0]) | (static_cast<uint16_t>(value[1]) << 8);
        return true;
    }

    bool u32(uint32_t& result) {
        std::span<const uint8_t> value;
        if (!take(4, value)) return false;
        result = static_cast<uint32_t>(value[0]) |
                 (static_cast<uint32_t>(value[1]) << 8) |
                 (static_cast<uint32_t>(value[2]) << 16) |
                 (static_cast<uint32_t>(value[3]) << 24);
        return true;
    }

private:
    std::span<const uint8_t> bytes_;
    size_t offset_ = 0;
};

bool read_identity(Reader& reader, PhysicalCodecIdentity& identity) {
    uint16_t layout_kind = 0, packing = 0, quantization_kind = 0;
    uint16_t accumulation = 0, scale = 0, zero = 0, bias = 0;
    uint32_t plane_count = 0;
    if (!reader.u16(identity.identity_version) || !reader.u16(identity.arithmetic_version)) return false;
    std::span<const uint8_t> digest;
    if (!reader.take(identity.arithmetic_digest.size(), digest)) return false;
    std::copy(digest.begin(), digest.end(), identity.arithmetic_digest.begin());
    if (!reader.take(identity.codebook_digest.size(), digest)) return false;
    std::copy(digest.begin(), digest.end(), identity.codebook_digest.begin());
    if (!reader.u16(layout_kind) || !reader.u16(identity.layout.version) ||
        !reader.u16(packing) || !reader.u8(identity.layout.rank) ||
        !reader.u8(identity.layout.block_rank)) return false;
    identity.layout.kind = static_cast<PhysicalLayoutKind>(layout_kind);
    identity.layout.packing = static_cast<PackingKind>(packing);
    for (uint8_t& axis : identity.layout.axis_order) if (!reader.u8(axis)) return false;
    if (!reader.u32(identity.layout.block_elements) || !reader.u32(identity.layout.block_bytes) ||
        !reader.u32(identity.layout.flags) || !reader.u16(quantization_kind) ||
        !reader.u16(identity.quantization.version) || !reader.u16(accumulation) ||
        !reader.u16(scale) || !reader.u16(zero) || !reader.u16(bias) ||
        !reader.u32(identity.quantization.block_elements) ||
        !reader.u32(identity.quantization.block_bytes) ||
        !reader.u32(identity.quantization.group_size) ||
        !reader.u32(identity.quantization.required_plane_mask) ||
        !reader.u32(identity.quantization.flags) || !reader.u32(plane_count) ||
        plane_count > 6) return false;
    identity.quantization.kind = static_cast<QuantizationKind>(quantization_kind);
    identity.quantization.accumulation_type = static_cast<ScalarType>(accumulation);
    identity.quantization.scale_type = static_cast<ScalarType>(scale);
    identity.quantization.zero_type = static_cast<ScalarType>(zero);
    identity.quantization.bias_type = static_cast<ScalarType>(bias);
    identity.planes.clear();
    identity.planes.reserve(plane_count);
    for (uint32_t index = 0; index != plane_count; ++index) {
        uint16_t kind = 0, storage = 0;
        PhysicalPlaneSchema plane;
        if (!reader.u16(kind) || !reader.u16(storage) ||
            !reader.u32(plane.logical_elements_covered) ||
            !reader.u32(plane.bytes_per_block) || !reader.u32(plane.flags)) return false;
        plane.kind = static_cast<PlaneKind>(kind);
        plane.storage_type = static_cast<ScalarType>(storage);
        identity.planes.push_back(plane);
    }
    return valid_physical_codec_identity(identity);
}

CompatibilityReport codec_error(std::string detail) {
    CompatibilityReport report = package_report(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                                                 std::move(detail));
    report.stage = CompatibilityStage::Semantic;
    report.phase = CompatibilityPhase::Semantic;
    return report;
}

void append_dimension(std::vector<uint8_t>& bytes, const Dimension& dimension) {
    append_u16(bytes, static_cast<uint16_t>(dimension.kind));
    append_u64(bytes, dimension.constant_or_symbol);
}

void append_tensor_binding(std::vector<uint8_t>& bytes, const SemanticTensor& tensor) {
    append_u32(bytes, tensor.id);
    append_u16(bytes, static_cast<uint16_t>(tensor.role));
    append_u16(bytes, static_cast<uint16_t>(tensor.logical_type));
    append_u32(bytes, static_cast<uint32_t>(tensor.dimensions.size()));
    for (const Dimension& dimension : tensor.dimensions) append_dimension(bytes, dimension);
    append_u16(bytes, static_cast<uint16_t>(tensor.layout.kind));
    append_u16(bytes, tensor.layout.version);
    append_u16(bytes, static_cast<uint16_t>(tensor.layout.packing));
    append_u8(bytes, tensor.layout.rank);
    append_u8(bytes, tensor.layout.block_rank);
    for (uint8_t axis : tensor.layout.axis_order) append_u8(bytes, axis);
    for (uint64_t stride : tensor.layout.strides) append_u64(bytes, stride);
    append_u32(bytes, tensor.layout.block_elements);
    append_u32(bytes, tensor.layout.block_bytes);
    append_u32(bytes, tensor.layout.flags);
    append_u16(bytes, static_cast<uint16_t>(tensor.quantization.kind));
    append_u16(bytes, tensor.quantization.version);
    append_u16(bytes, static_cast<uint16_t>(tensor.quantization.accumulation_type));
    append_u16(bytes, static_cast<uint16_t>(tensor.quantization.scale_type));
    append_u16(bytes, static_cast<uint16_t>(tensor.quantization.zero_type));
    append_u16(bytes, static_cast<uint16_t>(tensor.quantization.bias_type));
    append_u32(bytes, tensor.quantization.block_elements);
    append_u32(bytes, tensor.quantization.block_bytes);
    append_u32(bytes, tensor.quantization.group_size);
    append_u32(bytes, tensor.quantization.required_plane_mask);
    append_u32(bytes, tensor.quantization.flags);
    append_u8(bytes, static_cast<uint8_t>(tensor.expert_axis.kind));
    append_u8(bytes, tensor.expert_axis.expert_axis);
    append_u8(bytes, tensor.expert_axis.member_axis);
    append_u8(bytes, tensor.expert_axis.input_axis);
    append_u8(bytes, tensor.expert_axis.output_axis);
    append_u32(bytes, tensor.expert_axis.expert_count);
    append_u64(bytes, tensor.expert_axis.per_expert_byte_stride);
    append_u32(bytes, tensor.expert_axis.flags);
    append_u16(bytes, tensor.flags);
    append_u32(bytes, static_cast<uint32_t>(tensor.planes.size()));
    for (const TensorPlane& plane : tensor.planes) {
        append_u16(bytes, static_cast<uint16_t>(plane.kind));
        append_u16(bytes, static_cast<uint16_t>(plane.storage_type));
        append_u32(bytes, plane.artifact_id.value);
        append_u64(bytes, plane.offset);
        append_u64(bytes, plane.length);
        append_u32(bytes, plane.alignment);
        append_u32(bytes, plane.flags);
    }
}

Sha256Digest digest_bytes(std::span<const uint8_t> bytes) {
    Sha256Digest digest;
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    for (size_t offset = 0; offset != bytes.size();) {
        const size_t chunk = std::min<size_t>(1024 * 1024, bytes.size() - offset);
        CC_SHA256_Update(&context, bytes.data() + offset, static_cast<CC_LONG>(chunk));
        offset += chunk;
    }
    CC_SHA256_Final(digest.bytes.data(), &context);
    return digest;
}

} // namespace

bool valid_physical_codec_identity(const PhysicalCodecIdentity& identity) {
    if (identity.identity_version != 1 || identity.arithmetic_version == 0 ||
        digest_is_zero(identity.arithmetic_digest) || !valid_layout_kind(identity.layout.kind) ||
        !valid_packing_kind(identity.layout.packing) ||
        !valid_quantization_kind(identity.quantization.kind) || identity.layout.version != 1 ||
        identity.layout.flags != 0 || identity.quantization.version != 1 ||
        identity.quantization.flags != 0 || identity.layout.rank == 0 ||
        identity.layout.rank > identity.layout.axis_order.size() ||
        identity.layout.block_rank > identity.layout.rank) return false;
    if ((identity.quantization.kind == QuantizationKind::Codebook) ==
        digest_is_zero(identity.codebook_digest)) return false;

    std::array<bool, 8> seen_axis{};
    for (uint8_t index = 0; index != identity.layout.rank; ++index) {
        const uint8_t axis = identity.layout.axis_order[index];
        if (axis >= identity.layout.rank || seen_axis[axis]) return false;
        seen_axis[axis] = true;
    }
    for (uint8_t index = identity.layout.rank; index != identity.layout.axis_order.size(); ++index) {
        if (identity.layout.axis_order[index] != 0xff) return false;
    }

    const auto valid_optional_scalar = [](ScalarType type) {
        return static_cast<uint16_t>(type) == 0 || valid_scalar_type(type);
    };
    if (!valid_scalar_type(identity.quantization.accumulation_type) ||
        !valid_optional_scalar(identity.quantization.scale_type) ||
        !valid_optional_scalar(identity.quantization.zero_type) ||
        !valid_optional_scalar(identity.quantization.bias_type)) return false;

    switch (identity.layout.kind) {
    case PhysicalLayoutKind::ContiguousRowMajor:
        if (identity.layout.packing != PackingKind::None || identity.layout.block_rank != 0 ||
            identity.layout.block_elements != 0 || identity.layout.block_bytes != 0) return false;
        break;
    case PhysicalLayoutKind::GgufBlocked:
        if (identity.layout.packing != PackingKind::Gguf || identity.layout.block_rank == 0 ||
            identity.layout.block_elements == 0 || identity.layout.block_bytes == 0) return false;
        break;
    case PhysicalLayoutKind::GroupedAffine:
        if (identity.layout.packing != PackingKind::LsbBitPacked || identity.layout.block_rank != 1 ||
            identity.layout.block_elements == 0 || identity.layout.block_bytes == 0) return false;
        break;
    case PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip:
        if (identity.layout.packing != PackingKind::LsbBitPacked || identity.layout.block_rank != 1 ||
            identity.layout.block_elements != 256 || identity.layout.block_bytes != 64) return false;
        break;
    }

    constexpr uint32_t all_planes = (1u << 6) - 1;
    if ((identity.quantization.required_plane_mask & ~all_planes) != 0) return false;
    if (identity.quantization.kind == QuantizationKind::None) {
        if (identity.layout.kind != PhysicalLayoutKind::ContiguousRowMajor ||
            identity.quantization.scale_type != static_cast<ScalarType>(0) ||
            identity.quantization.zero_type != static_cast<ScalarType>(0) ||
            identity.quantization.bias_type != static_cast<ScalarType>(0) ||
            identity.quantization.block_elements != 0 || identity.quantization.block_bytes != 0 ||
            identity.quantization.group_size != 0 || identity.quantization.required_plane_mask != 0) return false;
    } else if (identity.layout.kind == PhysicalLayoutKind::ContiguousRowMajor ||
               identity.layout.block_rank == 0 || identity.layout.block_elements == 0 ||
               identity.layout.block_bytes == 0 || identity.quantization.block_elements == 0 ||
               identity.quantization.block_bytes == 0 || identity.quantization.group_size == 0 ||
               identity.layout.block_elements != identity.quantization.block_elements ||
               identity.layout.block_bytes != identity.quantization.block_bytes ||
               identity.quantization.required_plane_mask == 0) return false;
    if (identity.quantization.kind == QuantizationKind::BlockedAffine &&
        identity.layout.kind != PhysicalLayoutKind::GgufBlocked &&
        identity.layout.kind != PhysicalLayoutKind::GroupedAffine &&
        identity.layout.kind != PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip) return false;
    if (identity.quantization.kind == QuantizationKind::Codebook &&
        identity.layout.kind != PhysicalLayoutKind::GgufBlocked) return false;

    uint32_t seen_planes = 0;
    for (size_t index = 0; index != identity.planes.size(); ++index) {
        const PhysicalPlaneSchema& plane = identity.planes[index];
        if (!valid_plane_kind(plane.kind) || !valid_scalar_type(plane.storage_type) ||
            plane.logical_elements_covered == 0 || plane.bytes_per_block == 0 ||
            (plane.flags & ~1u) != 0 || (index != 0 &&
             static_cast<uint16_t>(identity.planes[index - 1].kind) >= static_cast<uint16_t>(plane.kind))) return false;
        const uint32_t mask = plane_mask(plane.kind);
        if ((seen_planes & mask) != 0) return false;
        seen_planes |= mask;
    }
    if (identity.planes.empty()) return false;
    if (identity.quantization.kind == QuantizationKind::None) {
        return identity.planes.size() == 1 && identity.planes[0].kind == PlaneKind::Values &&
               identity.planes[0].logical_elements_covered == 1 &&
               identity.planes[0].bytes_per_block ==
                   scalar_byte_width(identity.planes[0].storage_type);
    }
    return seen_planes == identity.quantization.required_plane_mask;
}

bool physical_codec_identity_less(const PhysicalCodecIdentity& left,
                                  const PhysicalCodecIdentity& right) {
    return compare_codec_identity(left, right) < 0;
}

bool validate_physical_codec_registry(const PhysicalCodecRegistry& registry,
                                      bool require_nonempty) {
    if (registry.codecs.empty() || registry.tensors.empty()) {
        return !require_nonempty && registry.codecs.empty() && registry.tensors.empty();
    }
    if (registry.codecs.size() > kPhysicalCodecRegistryMaximumCodecs ||
        registry.tensors.size() > kPhysicalCodecRegistryMaximumTensors) return false;
    size_t certificate_bytes = 0;
    for (const PhysicalCodecSpec& codec : registry.codecs) {
        if (!valid_physical_codec_identity(codec.identity) ||
            codec.certificate_bytes.size() > kPhysicalCodecRegistryMaximumCertificateBytes ||
            codec.certificate_bytes.size() >
                kPhysicalCodecRegistryMaximumCertificateTableBytes - certificate_bytes) {
            return false;
        }
        certificate_bytes += codec.certificate_bytes.size();
        if (!codec.certificate_bytes.empty() &&
            digest_bytes(codec.certificate_bytes).bytes !=
            codec.identity.arithmetic_digest) {
            return false;
        }
    }
    for (const PhysicalTensorCodecDeclaration& declaration : registry.tensors) {
        if (declaration.tensor_id == UINT32_MAX ||
            !valid_physical_codec_identity(declaration.identity)) return false;
    }

    std::vector<const PhysicalCodecSpec*> codecs;
    codecs.reserve(registry.codecs.size());
    for (const PhysicalCodecSpec& codec : registry.codecs) codecs.push_back(&codec);
    std::sort(codecs.begin(), codecs.end(), [](const auto* left, const auto* right) {
        return physical_codec_identity_less(left->identity, right->identity);
    });
    for (size_t index = 1; index < codecs.size(); ++index) {
        if (codecs[index - 1]->identity == codecs[index]->identity) return false;
    }

    std::vector<const PhysicalTensorCodecDeclaration*> tensors;
    tensors.reserve(registry.tensors.size());
    for (const auto& declaration : registry.tensors) tensors.push_back(&declaration);
    std::sort(tensors.begin(), tensors.end(), [](const auto* left, const auto* right) {
        return left->tensor_id < right->tensor_id;
    });
    for (size_t index = 1; index < tensors.size(); ++index) {
        if (tensors[index - 1]->tensor_id == tensors[index]->tensor_id) return false;
    }

    std::vector<bool> used(codecs.size(), false);
    for (const auto* declaration : tensors) {
        const auto found = std::lower_bound(
            codecs.begin(), codecs.end(), declaration->identity,
            [](const auto* candidate, const PhysicalCodecIdentity& identity) {
                return physical_codec_identity_less(candidate->identity, identity);
            });
        if (found == codecs.end() || (*found)->identity != declaration->identity) return false;
        used[static_cast<size_t>(found - codecs.begin())] = true;
    }
    return std::find(used.begin(), used.end(), false) == used.end();
}

bool physical_codec_registry_is_canonical(const PhysicalCodecRegistry& registry) {
    if (!validate_physical_codec_registry(registry)) return false;
    for (size_t index = 1; index < registry.codecs.size(); ++index) {
        if (!physical_codec_identity_less(registry.codecs[index - 1].identity,
                                          registry.codecs[index].identity)) return false;
    }
    for (size_t index = 1; index < registry.tensors.size(); ++index) {
        if (registry.tensors[index - 1].tensor_id >= registry.tensors[index].tensor_id) return false;
    }
    return true;
}

bool physical_codec_registry_covers_model(const PhysicalCodecRegistry& registry,
                                          const SemanticModel& model) {
    if (!validate_physical_codec_registry(registry)) return false;
    std::vector<std::pair<uint32_t, uint32_t>> referenced;
    for (const SemanticOperator& op : model.operators) {
        for (uint32_t tensor_index : op.tensors) {
            if (tensor_index >= model.tensors.size()) return false;
            referenced.emplace_back(model.tensors[tensor_index].id, tensor_index);
        }
    }
    std::sort(referenced.begin(), referenced.end());
    for (size_t index = 1; index < referenced.size(); ++index) {
        if (referenced[index - 1].first == referenced[index].first &&
            referenced[index - 1].second != referenced[index].second) return false;
    }
    referenced.erase(std::unique(referenced.begin(), referenced.end()), referenced.end());
    std::vector<uint32_t> referenced_ids;
    referenced_ids.reserve(referenced.size());
    for (const auto& item : referenced) referenced_ids.push_back(item.first);
    std::vector<uint32_t> declared;
    declared.reserve(registry.tensors.size());
    for (const auto& declaration : registry.tensors) declared.push_back(declaration.tensor_id);
    std::sort(declared.begin(), declared.end());
    return referenced_ids == declared;
}

bool physical_codec_registry_matches_model(const PhysicalCodecRegistry& registry,
                                           const SemanticModel& model) {
    if (!validate_physical_codec_registry(registry, true) ||
        !physical_codec_registry_is_canonical(registry) ||
        registry.tensors.size() != model.tensors.size()) return false;
    std::vector<const SemanticTensor*> tensors;
    tensors.reserve(model.tensors.size());
    for (const SemanticTensor& tensor : model.tensors) tensors.push_back(&tensor);
    std::sort(tensors.begin(), tensors.end(), [](const auto* left, const auto* right) {
        return left->id < right->id;
    });
    for (size_t index = 1; index < tensors.size(); ++index) {
        if (tensors[index - 1]->id == tensors[index]->id) return false;
    }
    for (size_t index = 0; index != tensors.size(); ++index) {
        if (registry.tensors[index].tensor_id != tensors[index]->id) return false;
        const auto identity = physical_codec_identity(
            *tensors[index], registry.tensors[index].identity.arithmetic_version,
            registry.tensors[index].identity.arithmetic_digest,
            registry.tensors[index].identity.codebook_digest);
        if (!identity || *identity != registry.tensors[index].identity) return false;
    }
    return true;
}

std::optional<PhysicalCodecIdentity> physical_codec_identity(
    const SemanticTensor& tensor, uint16_t arithmetic_version,
    const PhysicalIdentityDigest& arithmetic_digest,
    const PhysicalIdentityDigest& codebook_digest) {
    PhysicalCodecIdentity identity;
    if (!valid_tensor_plane_spans(tensor)) return std::nullopt;
    identity.arithmetic_version = arithmetic_version;
    identity.arithmetic_digest = arithmetic_digest;
    identity.codebook_digest = codebook_digest;
    identity.layout = {tensor.layout.kind, tensor.layout.version, tensor.layout.packing,
                       tensor.layout.rank, tensor.layout.block_rank,
                       tensor.layout.axis_order, tensor.layout.block_elements,
                       tensor.layout.block_bytes, tensor.layout.flags};
    identity.quantization = {
        tensor.quantization.kind, tensor.quantization.version,
        tensor.quantization.accumulation_type, tensor.quantization.scale_type,
        tensor.quantization.zero_type, tensor.quantization.bias_type,
        tensor.quantization.block_elements, tensor.quantization.block_bytes,
        tensor.quantization.group_size, tensor.quantization.required_plane_mask,
        tensor.quantization.flags};
    uint64_t elements = 1;
    for (const Dimension& dimension : tensor.dimensions) {
        if (dimension.kind != DimensionKind::Constant || dimension.constant_or_symbol == 0 ||
            !checked_multiply(elements, dimension.constant_or_symbol, elements)) return std::nullopt;
    }
    const uint32_t elements_per_block = tensor.quantization.kind == QuantizationKind::None
        ? 1 : tensor.quantization.block_elements;
    if (elements_per_block == 0 || elements % elements_per_block != 0) return std::nullopt;
    const uint64_t block_count = elements / elements_per_block;
    identity.planes.reserve(tensor.planes.size());
    for (const TensorPlane& plane : tensor.planes) {
        if (block_count == 0 || plane.length == 0 || plane.length % block_count != 0 ||
            plane.length / block_count > UINT32_MAX) return std::nullopt;
        identity.planes.push_back({plane.kind, plane.storage_type, elements_per_block,
                                   static_cast<uint32_t>(plane.length / block_count), plane.flags});
    }
    std::sort(identity.planes.begin(), identity.planes.end(), [](const auto& left, const auto& right) {
        return static_cast<uint16_t>(left.kind) < static_cast<uint16_t>(right.kind);
    });
    return valid_physical_codec_identity(identity)
        ? std::optional<PhysicalCodecIdentity>(std::move(identity)) : std::nullopt;
}

PhysicalCodecRegistryEncodeResult encode_physical_codec_registry(
    const PhysicalCodecRegistry& registry) {
    if (!validate_physical_codec_registry(registry) || !physical_codec_registry_is_canonical(registry)) {
        return codec_error("physical codec registry is invalid or not canonically ordered");
    }
    std::vector<uint8_t> bytes;
    append_u32(bytes, static_cast<uint32_t>(registry.codecs.size()));
    for (const auto& codec : registry.codecs) {
        append_identity(bytes, codec.identity);
        append_u32(bytes, static_cast<uint32_t>(codec.certificate_bytes.size()));
        bytes.insert(bytes.end(), codec.certificate_bytes.begin(), codec.certificate_bytes.end());
    }
    append_u32(bytes, static_cast<uint32_t>(registry.tensors.size()));
    for (const auto& declaration : registry.tensors) {
        append_u32(bytes, declaration.tensor_id);
        append_identity(bytes, declaration.identity);
    }
    return bytes;
}

PhysicalCodecRegistryDecodeResult decode_physical_codec_registry(
    std::span<const uint8_t> bytes) {
    Reader reader(bytes);
    PhysicalCodecRegistry registry;
    uint32_t codec_count = 0, tensor_count = 0;
    if (!reader.u32(codec_count) || codec_count > kPhysicalCodecRegistryMaximumCodecs) {
        return codec_error("physical codec registry codec count is invalid");
    }
    registry.codecs.reserve(codec_count);
    size_t certificate_bytes_total = 0;
    for (uint32_t index = 0; index != codec_count; ++index) {
        PhysicalCodecSpec codec;
        uint32_t certificate_length = 0;
        std::span<const uint8_t> certificate_bytes;
        if (!read_identity(reader, codec.identity) ||
            !reader.u32(certificate_length) ||
            certificate_length > kPhysicalCodecRegistryMaximumCertificateBytes ||
            certificate_length >
                kPhysicalCodecRegistryMaximumCertificateTableBytes - certificate_bytes_total ||
            !reader.take(certificate_length, certificate_bytes)) {
            return codec_error("physical codec registry codec certificate is invalid or truncated");
        }
        certificate_bytes_total += certificate_length;
        codec.certificate_bytes.assign(certificate_bytes.begin(), certificate_bytes.end());
        registry.codecs.push_back(std::move(codec));
    }
    if (!reader.u32(tensor_count) || tensor_count > kPhysicalCodecRegistryMaximumTensors) {
        return codec_error("physical codec registry tensor count is invalid");
    }
    registry.tensors.reserve(tensor_count);
    for (uint32_t index = 0; index != tensor_count; ++index) {
        PhysicalTensorCodecDeclaration declaration;
        if (!reader.u32(declaration.tensor_id) || !read_identity(reader, declaration.identity)) {
            return codec_error("physical codec registry tensor declaration is invalid or truncated");
        }
        registry.tensors.push_back(std::move(declaration));
    }
    if (reader.remaining() != 0 || !validate_physical_codec_registry(registry) ||
        !physical_codec_registry_is_canonical(registry)) {
        return codec_error("physical codec registry is malformed, duplicate, unused, or reordered");
    }
    return registry;
}

Sha256Digest physical_codec_registry_digest(const PhysicalCodecRegistry& registry,
                                            const SemanticModel& model) {
    Sha256Digest empty;
    if (registry.codecs.empty() && registry.tensors.empty()) return empty;
    if (!physical_codec_registry_matches_model(registry, model)) return empty;
    const auto encoded = encode_physical_codec_registry(registry);
    if (std::holds_alternative<CompatibilityReport>(encoded)) return empty;
    std::vector<uint8_t> bytes;
    constexpr char domain[] = "laplace-physical-codec-registry-v2";
    bytes.insert(bytes.end(), domain, domain + sizeof(domain));
    const auto& registry_bytes = std::get<std::vector<uint8_t>>(encoded);
    bytes.insert(bytes.end(), registry_bytes.begin(), registry_bytes.end());
    std::vector<const SemanticTensor*> tensors;
    tensors.reserve(model.tensors.size());
    for (const SemanticTensor& tensor : model.tensors) tensors.push_back(&tensor);
    std::sort(tensors.begin(), tensors.end(), [](const auto* left, const auto* right) {
        return left->id < right->id;
    });
    append_u32(bytes, static_cast<uint32_t>(tensors.size()));
    for (const SemanticTensor* tensor : tensors) append_tensor_binding(bytes, *tensor);
    return digest_bytes(bytes);
}

} // namespace Laplace
