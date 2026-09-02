#include "artifact_index.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace Laplace {

namespace {

CompatibilityReport index_failure(CompatibilityError code, std::string detail,
                                  ArtifactId artifact = {}, uint32_t tensor = UINT32_MAX,
                                  CanonicalFactKey fact = {}) {
    CompatibilityReport report = compatibility_report(code, std::move(detail));
    report.artifact_id = artifact;
    report.artifact_index = artifact.value;
    report.tensor_id = tensor;
    report.fact_key = fact;
    return report;
}

bool valid_role(ArtifactRole role) {
    return role == ArtifactRole::Primary || role == ArtifactRole::Shard || role == ArtifactRole::Sidecar;
}

bool valid_authority(ArtifactFactAuthority authority) {
    return authority == ArtifactFactAuthority::Declared || authority == ArtifactFactAuthority::Structural ||
           authority == ArtifactFactAuthority::Derived;
}

bool valid_fact_state(ArtifactFactState state) {
    const uint8_t value = static_cast<uint8_t>(state);
    return value >= static_cast<uint8_t>(ArtifactFactState::Present) &&
           value <= static_cast<uint8_t>(ArtifactFactState::Ambiguous);
}

bool valid_scalar(ArtifactScalarType type) {
    const uint16_t value = static_cast<uint16_t>(type);
    return value >= static_cast<uint16_t>(ArtifactScalarType::F32) &&
           value <= static_cast<uint16_t>(ArtifactScalarType::Packed);
}

bool valid_plane(PlaneKind kind) {
    const uint16_t value = static_cast<uint16_t>(kind);
    return value >= static_cast<uint16_t>(PlaneKind::Values) &&
           value <= static_cast<uint16_t>(PlaneKind::LayoutMetadata);
}

bool valid_quantization(QuantizationKind kind) {
    return kind == QuantizationKind::None || kind == QuantizationKind::BlockedAffine ||
           kind == QuantizationKind::Codebook;
}

bool valid_layout(PhysicalLayoutKind kind) {
    return kind == PhysicalLayoutKind::ContiguousRowMajor ||
           kind == PhysicalLayoutKind::GgufBlocked ||
           kind == PhysicalLayoutKind::GroupedAffine ||
           kind == PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip;
}

bool valid_packing(PackingKind kind) {
    return kind == PackingKind::None || kind == PackingKind::Gguf ||
           kind == PackingKind::LsbBitPacked;
}

bool checked_multiply(uint64_t left, uint64_t right, uint64_t& result) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) return false;
    result = left * right;
    return true;
}

const PackageView* artifact_by_id(std::span<const PackageView> artifacts, ArtifactId id) {
    const auto it = std::lower_bound(artifacts.begin(), artifacts.end(), id,
                                     [](const PackageView& artifact, ArtifactId wanted) {
                                         return artifact.artifact_id().value < wanted.value;
                                     });
    return it != artifacts.end() && it->artifact_id() == id ? &*it : nullptr;
}

const ArtifactTensorRecord* tensor_by_id(std::span<const ArtifactTensorRecord> tensors, uint32_t id) {
    const auto it = std::lower_bound(tensors.begin(), tensors.end(), id,
                                     [](const ArtifactTensorRecord& tensor, uint32_t wanted) {
                                         return tensor.id < wanted;
                                     });
    return it != tensors.end() && it->id == id ? &*it : nullptr;
}

bool span_in_artifact(const ArtifactSourceSpan& span, std::span<const PackageView> artifacts) {
    const PackageView* artifact = artifact_by_id(artifacts, span.artifact_id);
    if (!artifact) return false;
    const uint64_t size = artifact->bytes().size();
    return span.offset <= size && span.length <= size - span.offset;
}

bool row_major_layout_is_exact(const ArtifactTensorRecord& tensor) {
    const size_t rank = tensor.logical_dimensions.size();
    if (rank == 0) return true;
    uint64_t stride = 1;
    for (size_t reverse = rank; reverse != 0; --reverse) {
        const size_t axis = reverse - 1;
        if (tensor.layout.strides[axis] != stride) return false;
        if (!checked_multiply(stride, tensor.logical_dimensions[axis], stride)) return false;
    }
    return true;
}

bool gguf_layout_is_exact(const ArtifactTensorRecord& tensor) {
    const size_t rank = tensor.logical_dimensions.size();
    uint64_t stride = 1;
    for (size_t axis = 0; axis != rank; ++axis) {
        if (tensor.layout.strides[axis] != stride) return false;
        if (!checked_multiply(stride, tensor.logical_dimensions[axis], stride)) return false;
    }
    return true;
}

bool grouped_affine_layout_is_exact(const ArtifactTensorRecord& tensor) {
    return tensor.logical_dimensions.size() == 2 && tensor.layout.rank == 2 &&
           tensor.layout.block_rank == 1 && tensor.layout.axis_order[0] == 1 &&
           tensor.layout.axis_order[1] == 0 && tensor.layout.strides[0] == 1 &&
           tensor.layout.strides[1] == tensor.logical_dimensions[1];
}

std::optional<ArtifactPhysicalFormat> expected_physical_format(ArtifactPhysicalEncoding encoding) {
    ArtifactPhysicalFormat format;
    format.version = 1;
    format.encoding = encoding;
    switch (encoding) {
    case ArtifactPhysicalEncoding::F32:
        format.value_type = ArtifactScalarType::F32;
        format.block_elements = 1;
        format.block_bytes = 4;
        return format;
    case ArtifactPhysicalEncoding::F16:
        format.value_type = ArtifactScalarType::F16;
        format.block_elements = 1;
        format.block_bytes = 2;
        return format;
    case ArtifactPhysicalEncoding::Q4_K:
        format.value_type = ArtifactScalarType::Packed;
        format.scale_type = ArtifactScalarType::F16;
        format.zero_type = ArtifactScalarType::F16;
        format.subscale_type = ArtifactScalarType::Packed;
        format.block_elements = 256;
        format.block_bytes = 144;
        format.scale_bytes = 2;
        format.zero_bytes = 2;
        format.subscale_bytes = 12;
        return format;
    case ArtifactPhysicalEncoding::Q5_0:
        format.value_type = ArtifactScalarType::Packed;
        format.scale_type = ArtifactScalarType::F16;
        format.block_elements = 32;
        format.block_bytes = 22;
        format.scale_bytes = 2;
        return format;
    case ArtifactPhysicalEncoding::Q4_0:
        format.value_type = ArtifactScalarType::Packed;
        format.scale_type = ArtifactScalarType::F16;
        format.block_elements = 32;
        format.block_bytes = 18;
        format.scale_bytes = 2;
        return format;
    case ArtifactPhysicalEncoding::Q6_K:
        format.value_type = ArtifactScalarType::Packed;
        format.scale_type = ArtifactScalarType::F16;
        format.subscale_type = ArtifactScalarType::I8;
        format.block_elements = 256;
        format.block_bytes = 210;
        format.scale_bytes = 2;
        format.subscale_bytes = 16;
        return format;
    case ArtifactPhysicalEncoding::Q8_0:
        format.value_type = ArtifactScalarType::Packed;
        format.scale_type = ArtifactScalarType::F16;
        format.block_elements = 32;
        format.block_bytes = 34;
        format.scale_bytes = 2;
        return format;
    case ArtifactPhysicalEncoding::GroupedAffineU2_256:
        format.version = 2;
        format.value_type = ArtifactScalarType::U32;
        format.scale_type = ArtifactScalarType::F16;
        format.bias_type = ArtifactScalarType::F16;
        format.block_elements = 256;
        format.block_bytes = 64;
        format.scale_bytes = 2;
        format.bias_bytes = 2;
        return format;
    case ArtifactPhysicalEncoding::ColumnGroupedAffineU2Skip256:
        format.value_type = ArtifactScalarType::U8;
        format.scale_type = ArtifactScalarType::F16;
        format.bias_type = ArtifactScalarType::F16;
        format.block_elements = 256;
        format.block_bytes = 64;
        format.scale_bytes = 2;
        format.bias_bytes = 2;
        return format;
    default:
        return std::nullopt;
    }
}

bool coordinate_is_set(const ArtifactCoordinate& coordinate) {
    return coordinate.root != 0 || coordinate.layer != UINT32_MAX ||
           coordinate.slot != UINT32_MAX || coordinate.instance != UINT32_MAX ||
           coordinate.expert != UINT32_MAX || coordinate.bank_axis != UINT8_MAX ||
           coordinate.bank_extent != 0 || coordinate.bank_stride != 0;
}

bool valid_coordinate(const ArtifactCoordinate& coordinate) {
    if (coordinate.root == UINT32_MAX) return false;
    if (coordinate.bank_axis == UINT8_MAX) {
        return coordinate.bank_extent == 0 && coordinate.bank_stride == 0;
    }
    return coordinate.bank_extent != 0;
}

bool coordinate_less(const ArtifactCoordinate& left, const ArtifactCoordinate& right) {
    return std::tie(left.root, left.layer, left.slot, left.instance, left.expert,
                    left.bank_axis, left.bank_extent, left.bank_stride) <
           std::tie(right.root, right.layer, right.slot, right.instance, right.expert,
                    right.bank_axis, right.bank_extent, right.bank_stride);
}

void append_u8(std::vector<uint8_t>& bytes, uint8_t value) {
    bytes.push_back(value);
}

void append_u16(std::vector<uint8_t>& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8));
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (uint32_t shift = 0; shift != 32; shift += 8) bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (uint32_t shift = 0; shift != 64; shift += 8) bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void append_digest(std::vector<uint8_t>& bytes, const Sha256Digest& digest) {
    bytes.insert(bytes.end(), digest.bytes.begin(), digest.bytes.end());
}

void append_source(std::vector<uint8_t>& bytes, const ArtifactSourceSpan& source) {
    append_u32(bytes, source.artifact_id.value);
    append_u64(bytes, source.offset);
    append_u64(bytes, source.length);
}

void append_plane_normalized(std::vector<uint8_t>& bytes, const ArtifactTensorPlane& plane) {
    append_u16(bytes, static_cast<uint16_t>(plane.kind));
    append_u16(bytes, static_cast<uint16_t>(plane.storage_type));
    append_u64(bytes, plane.logical_elements);
    append_u32(bytes, plane.bytes_per_block);
    append_u32(bytes, plane.elements_per_block);
    append_u32(bytes, plane.alignment);
}

void append_coordinate(std::vector<uint8_t>& bytes, const ArtifactCoordinate& coordinate) {
    append_u32(bytes, coordinate.root);
    append_u32(bytes, coordinate.layer);
    append_u32(bytes, coordinate.slot);
    append_u32(bytes, coordinate.instance);
    append_u32(bytes, coordinate.expert);
    append_u8(bytes, coordinate.bank_axis);
    append_u32(bytes, coordinate.bank_extent);
    append_u64(bytes, coordinate.bank_stride);
}

void append_axis(std::vector<uint8_t>& bytes, const ArtifactPhysicalAxisContract& axis) {
    append_u8(bytes, axis.source_rank);
    for (uint8_t value : axis.source_axis_order) append_u8(bytes, value);
    append_u8(bytes, axis.block_axis);
    append_u32(bytes, axis.block_elements);
    append_u32(bytes, axis.bytes_per_block);
    append_u64(bytes, axis.row_stride_bytes);
    append_u32(bytes, axis.plane_order);
}

void append_format(std::vector<uint8_t>& bytes, const ArtifactPhysicalFormat& format) {
    append_u16(bytes, format.version);
    append_u16(bytes, static_cast<uint16_t>(format.encoding));
    append_u16(bytes, static_cast<uint16_t>(format.value_type));
    append_u16(bytes, static_cast<uint16_t>(format.scale_type));
    append_u16(bytes, static_cast<uint16_t>(format.zero_type));
    append_u16(bytes, static_cast<uint16_t>(format.subscale_type));
    append_u16(bytes, static_cast<uint16_t>(format.subzero_type));
    append_u32(bytes, format.block_elements);
    append_u32(bytes, format.block_bytes);
    append_u32(bytes, format.scale_bytes);
    append_u32(bytes, format.zero_bytes);
    append_u32(bytes, format.subscale_bytes);
    append_u32(bytes, format.subzero_bytes);
    if (format.version >= 2 ||
        format.encoding == ArtifactPhysicalEncoding::ColumnGroupedAffineU2Skip256) {
        append_u16(bytes, static_cast<uint16_t>(format.bias_type));
        append_u32(bytes, format.bias_bytes);
    }
}

void append_fact(std::vector<uint8_t>& bytes, const ArtifactFact& fact, bool include_source) {
    append_u32(bytes, fact.key.value);
    append_u8(bytes, static_cast<uint8_t>(fact.authority));
    append_u8(bytes, static_cast<uint8_t>(fact.state));
    append_coordinate(bytes, fact.scope);
    if (include_source) append_source(bytes, fact.source);
    append_u8(bytes, static_cast<uint8_t>(fact.value.index() + 1));
    std::visit([&](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, uint64_t>) {
            append_u64(bytes, value);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            append_u64(bytes, static_cast<uint64_t>(value));
        } else if constexpr (std::is_same_v<T, bool>) {
            append_u8(bytes, value ? 1 : 0);
        } else if constexpr (std::is_same_v<T, ArtifactF32Bits>) {
            append_u32(bytes, value.value);
        } else if constexpr (std::is_same_v<T, std::vector<uint64_t>>) {
            append_u32(bytes, static_cast<uint32_t>(value.size()));
            for (uint64_t item : value) append_u64(bytes, item);
        } else if constexpr (std::is_same_v<T, ArtifactF32Vector>) {
            append_u32(bytes, static_cast<uint32_t>(value.size()));
            for (const ArtifactF32Bits item : value) append_u32(bytes, item.value);
        } else if constexpr (std::is_same_v<T, ArtifactI64Vector>) {
            append_u32(bytes, static_cast<uint32_t>(value.size()));
            for (int64_t item : value) append_u64(bytes, static_cast<uint64_t>(item));
        } else if constexpr (std::is_same_v<T, Sha256Digest>) {
            append_digest(bytes, value);
        }
    }, fact.value);
}

void append_tensor_normalized(std::vector<uint8_t>& bytes, const ArtifactTensorRecord& tensor) {
    append_coordinate(bytes, tensor.coordinate);
    append_axis(bytes, tensor.axis);
    append_format(bytes, tensor.format);
    append_u16(bytes, static_cast<uint16_t>(tensor.logical_type));
    append_u32(bytes, static_cast<uint32_t>(tensor.logical_dimensions.size()));
    for (uint64_t dimension : tensor.logical_dimensions) append_u64(bytes, dimension);
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
    append_u32(bytes, static_cast<uint32_t>(tensor.role_evidence.size()));
    for (const ArtifactTensorRoleEvidence& evidence : tensor.role_evidence) {
        append_u16(bytes, static_cast<uint16_t>(evidence.role));
        append_u32(bytes, evidence.evidence_key.value);
        append_u8(bytes, static_cast<uint8_t>(evidence.authority));
        append_coordinate(bytes, evidence.scope);
    }
    append_u32(bytes, static_cast<uint32_t>(tensor.planes.size()));
    for (const ArtifactTensorPlane& plane : tensor.planes) append_plane_normalized(bytes, plane);
}

Sha256Digest digest_bytes(std::span<const uint8_t> bytes) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    constexpr size_t chunk_size = 1024 * 1024;
    for (size_t offset = 0; offset < bytes.size(); offset += chunk_size) {
        const size_t count = std::min(chunk_size, bytes.size() - offset);
        CC_SHA256_Update(&context, bytes.data() + offset, static_cast<CC_LONG>(count));
    }
    Sha256Digest digest;
    CC_SHA256_Final(digest.bytes.data(), &context);
    return digest;
}

bool alias_connects(std::span<const ArtifactAlias> aliases, uint32_t left, uint32_t right) {
    return std::any_of(aliases.begin(), aliases.end(), [&](const ArtifactAlias& alias) {
        return (alias.source_tensor_id == left && alias.target_tensor_id == right) ||
               (alias.source_tensor_id == right && alias.target_tensor_id == left);
    });
}

bool identical_physical_contract(const ArtifactTensorRecord& left, const ArtifactTensorRecord& right) {
    if (left.logical_type != right.logical_type || left.logical_dimensions != right.logical_dimensions ||
        left.axis != right.axis || left.format != right.format || left.layout != right.layout ||
        left.quantization != right.quantization || left.planes.size() != right.planes.size()) return false;
    for (size_t index = 0; index != left.planes.size(); ++index) {
        if (!(left.planes[index] == right.planes[index])) return false;
    }
    return true;
}

struct SpanUse {
    ArtifactId artifact_id;
    uint64_t begin;
    uint64_t end;
    uint32_t tensor_id;
    PlaneKind plane;
};

} // namespace

std::variant<ArtifactIndex, CompatibilityReport> ArtifactIndex::build(ArtifactIndexInput input) {
    if (input.artifacts.empty() || input.artifacts.size() > UINT32_MAX ||
        input.metadata_facts.size() > UINT32_MAX || input.package_facts.size() > UINT32_MAX ||
        input.tensors.size() > UINT32_MAX || input.aliases.size() > UINT32_MAX) {
        return index_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                             "artifact index exceeds its bounded canonical representation");
    }

    std::sort(input.artifacts.begin(), input.artifacts.end(), [](const PackageView& left, const PackageView& right) {
        return left.artifact_id().value < right.artifact_id().value;
    });
    uint32_t primary_count = 0;
    for (size_t index = 0; index != input.artifacts.size(); ++index) {
        const PackageView& artifact = input.artifacts[index];
        if (artifact.artifact_id().value == UINT32_MAX || !valid_role(artifact.role()) || artifact.bytes().empty()) {
            return index_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                                 "artifact index contains an invalid immutable package member",
                                 artifact.artifact_id());
        }
        primary_count += artifact.role() == ArtifactRole::Primary;
        if (index != 0 && input.artifacts[index - 1].artifact_id() == artifact.artifact_id()) {
            return index_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                                 "artifact index contains a duplicate artifact ID", artifact.artifact_id());
        }
    }
    if (primary_count != 1) {
        return index_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                             "artifact index must retain exactly one primary artifact");
    }

    auto fact_less = [](const ArtifactFact& left, const ArtifactFact& right) {
        return left.key.value < right.key.value;
    };
    std::sort(input.metadata_facts.begin(), input.metadata_facts.end(), fact_less);
    std::sort(input.package_facts.begin(), input.package_facts.end(), fact_less);
    std::vector<uint32_t> fact_keys;
    fact_keys.reserve(input.metadata_facts.size() + input.package_facts.size());
    auto validate_facts = [&](const std::vector<ArtifactFact>& facts) -> std::optional<CompatibilityReport> {
        for (const ArtifactFact& fact : facts) {
            if (fact.key.value == UINT32_MAX || !valid_authority(fact.authority) ||
                !valid_fact_state(fact.state)) {
                return index_failure(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                     "artifact fact lacks a canonical key, authority, or state",
                                     fact.source.artifact_id, UINT32_MAX, fact.key);
            }
            if (fact.state == ArtifactFactState::Present && !span_in_artifact(fact.source, input.artifacts)) {
                CompatibilityReport report = index_failure(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                                           "artifact fact provenance is outside its immutable source",
                                                           fact.source.artifact_id, UINT32_MAX, fact.key);
                report.source_offset = fact.source.offset;
                report.source_length = fact.source.length;
                return report;
            }
            if (std::find(fact_keys.begin(), fact_keys.end(), fact.key.value) != fact_keys.end()) {
                return index_failure(CompatibilityError::IMPORT_RULE_CONFLICT,
                                     "artifact index contains a duplicate canonical fact key",
                                     fact.source.artifact_id, UINT32_MAX, fact.key);
            }
            if (const auto* list = std::get_if<std::vector<uint64_t>>(&fact.value);
                list && list->size() > UINT32_MAX) {
                return index_failure(CompatibilityError::RULE_LIMIT_EXCEEDED,
                                     "artifact fact list exceeds the canonical representation",
                                     fact.source.artifact_id, UINT32_MAX, fact.key);
            }
            if (const auto* list = std::get_if<ArtifactF32Vector>(&fact.value);
                list && list->size() > UINT32_MAX) {
                return index_failure(CompatibilityError::RULE_LIMIT_EXCEEDED,
                                     "artifact fact list exceeds the canonical representation",
                                     fact.source.artifact_id, UINT32_MAX, fact.key);
            }
            if (const auto* list = std::get_if<ArtifactI64Vector>(&fact.value);
                list && list->size() > UINT32_MAX) {
                return index_failure(CompatibilityError::RULE_LIMIT_EXCEEDED,
                                     "artifact fact list exceeds the canonical representation",
                                     fact.source.artifact_id, UINT32_MAX, fact.key);
            }
            fact_keys.push_back(fact.key.value);
        }
        return std::nullopt;
    };
    if (auto error = validate_facts(input.metadata_facts)) return std::move(*error);
    if (auto error = validate_facts(input.package_facts)) return std::move(*error);

    std::sort(input.tensors.begin(), input.tensors.end(), [](const ArtifactTensorRecord& left,
                                                             const ArtifactTensorRecord& right) {
        return left.id < right.id;
    });
    std::vector<SpanUse> spans;
    for (size_t tensor_index = 0; tensor_index != input.tensors.size(); ++tensor_index) {
        ArtifactTensorRecord& tensor = input.tensors[tensor_index];
        if (tensor.id == UINT32_MAX || (tensor_index != 0 && input.tensors[tensor_index - 1].id == tensor.id)) {
            return index_failure(CompatibilityError::IMPORT_TENSOR_DUPLICATE,
                                 "artifact index contains a duplicate or invalid tensor ID", {}, tensor.id);
        }
        if (!valid_coordinate(tensor.coordinate)) {
            return index_failure(CompatibilityError::IR_LAYOUT_MISMATCH,
                                 "tensor structural coordinate is invalid", {}, tensor.id);
        }
        if (tensor.coordinate.bank_axis != UINT8_MAX &&
            (tensor.coordinate.bank_axis >= tensor.logical_dimensions.size() ||
             tensor.coordinate.bank_extent == 0 || tensor.coordinate.bank_stride == 0 ||
             tensor.coordinate.bank_extent != tensor.logical_dimensions[tensor.coordinate.bank_axis])) {
            return index_failure(CompatibilityError::IR_LAYOUT_MISMATCH,
                                 "tensor structural bank coordinate does not match its shape", {}, tensor.id);
        }
        if (tensor.axis.source_rank != 0 && tensor.axis.source_rank != tensor.logical_dimensions.size()) {
            return index_failure(CompatibilityError::IR_LAYOUT_MISMATCH,
                                 "tensor physical axis rank does not match logical rank", {}, tensor.id);
        }
        if (tensor.axis.block_axis != UINT8_MAX &&
            (tensor.axis.block_axis >= tensor.logical_dimensions.size() ||
             tensor.axis.block_elements == 0 || tensor.axis.bytes_per_block == 0)) {
            return index_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                 "tensor physical block axis is invalid", {}, tensor.id);
        }
        if (tensor.layout.block_rank > tensor.layout.rank ||
            (tensor.layout.block_rank == 0 &&
             (tensor.layout.block_elements != 0 || tensor.layout.block_bytes != 0)) ||
            (tensor.layout.block_rank != 0 &&
             (tensor.layout.block_elements == 0 || tensor.layout.block_bytes == 0))) {
            return index_failure(CompatibilityError::IR_LAYOUT_MISMATCH,
                                 "tensor layout block contract is incomplete", {}, tensor.id);
        }
        if (tensor.axis.source_rank != 0) {
            std::array<bool, 8> source_axes{};
            for (size_t axis = 0; axis != tensor.axis.source_axis_order.size(); ++axis) {
                const uint8_t source_axis = tensor.axis.source_axis_order[axis];
                if (axis < tensor.axis.source_rank) {
                    if (source_axis >= tensor.axis.source_rank || source_axes[source_axis]) {
                        return index_failure(CompatibilityError::IR_LAYOUT_MISMATCH,
                                             "tensor source axis order is not a permutation", {}, tensor.id);
                    }
                    source_axes[source_axis] = true;
                } else if (source_axis != UINT8_MAX) {
                    return index_failure(CompatibilityError::IR_LAYOUT_MISMATCH,
                                         "tensor source axis order has data beyond its rank", {}, tensor.id);
                }
            }
        }
        if (!valid_scalar(tensor.logical_type) || tensor.logical_dimensions.size() > tensor.layout.axis_order.size() ||
            tensor.layout.rank != tensor.logical_dimensions.size()) {
            return index_failure(CompatibilityError::IR_SHAPE_MISMATCH,
                                 "tensor logical rank or scalar type is invalid", {}, tensor.id);
        }
        uint64_t element_count = 1;
        for (uint64_t dimension : tensor.logical_dimensions) {
            if (dimension == 0) {
                return index_failure(CompatibilityError::IR_SHAPE_MISMATCH,
                                     "tensor logical dimensions must be nonzero", {}, tensor.id);
            }
            if (!checked_multiply(element_count, dimension, element_count)) {
                return index_failure(CompatibilityError::IR_SHAPE_MISMATCH,
                                     "tensor logical element count overflows", {}, tensor.id);
            }
        }

        if (!valid_layout(tensor.layout.kind) || !valid_packing(tensor.layout.packing)) {
            return index_failure(CompatibilityError::IR_LAYOUT_MISMATCH,
                                 "tensor physical layout kind is invalid", {}, tensor.id);
        }
        std::array<bool, 8> seen_axes{};
        for (size_t axis = 0; axis != tensor.layout.axis_order.size(); ++axis) {
            const uint8_t physical_axis = tensor.layout.axis_order[axis];
            if (axis < tensor.layout.rank) {
                if (physical_axis >= tensor.layout.rank || seen_axes[physical_axis]) {
                    return index_failure(CompatibilityError::IR_LAYOUT_MISMATCH,
                                         "tensor physical axis order is not a permutation", {}, tensor.id);
                }
                seen_axes[physical_axis] = true;
            } else if (physical_axis != 0xff) {
                return index_failure(CompatibilityError::IR_LAYOUT_MISMATCH,
                                     "tensor physical axis order has data beyond its rank", {}, tensor.id);
            }
        }
        bool identity_axes = true;
        for (uint8_t axis = 0; axis != tensor.layout.rank; ++axis) {
            identity_axes = identity_axes && tensor.layout.axis_order[axis] == axis;
        }
        // A contiguous physical tensor may declare either axis zero or the
        // last axis as its unit-stride axis. The explicit strides carry that
        // fact; the scalar encoding does not identify a source container.
        const bool grouped_affine_layout =
            tensor.layout.kind == PhysicalLayoutKind::GroupedAffine ||
            tensor.layout.kind == PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip;
        const bool exact_strides = grouped_affine_layout
            ? grouped_affine_layout_is_exact(tensor)
            : (row_major_layout_is_exact(tensor) || gguf_layout_is_exact(tensor));
        const bool exact_axes = grouped_affine_layout
            ? tensor.layout.axis_order[0] == 1 && tensor.layout.axis_order[1] == 0
            : identity_axes;
        if (!exact_axes || !exact_strides) {
            return index_failure(CompatibilityError::IR_LAYOUT_MISMATCH,
                                 "contiguous tensor axes or strides are not canonical row-major", {}, tensor.id);
        }
        if ((tensor.layout.kind == PhysicalLayoutKind::ContiguousRowMajor &&
             tensor.layout.packing != PackingKind::None) ||
            (tensor.layout.kind == PhysicalLayoutKind::GgufBlocked &&
             tensor.layout.packing != PackingKind::Gguf) ||
            (grouped_affine_layout &&
             tensor.layout.packing != PackingKind::LsbBitPacked)) {
            return index_failure(CompatibilityError::IR_LAYOUT_MISMATCH,
                                 "tensor layout and packing contracts disagree", {}, tensor.id);
        }

        if (!valid_quantization(tensor.quantization.kind) ||
            (tensor.quantization.kind != QuantizationKind::None &&
             (tensor.quantization.block_elements == 0 || tensor.quantization.block_bytes == 0))) {
            return index_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                 "tensor quantization block contract is incomplete", {}, tensor.id);
        }
        if (tensor.quantization.kind == QuantizationKind::None &&
            (tensor.quantization.block_elements != 0 || tensor.quantization.block_bytes != 0 ||
             tensor.quantization.group_size != 0)) {
            return index_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                 "unquantized tensor carries a block quantization contract", {}, tensor.id);
        }
        if (tensor.axis.block_axis == UINT8_MAX &&
            (tensor.axis.block_elements != 0 || tensor.axis.bytes_per_block != 0)) {
            return index_failure(CompatibilityError::IR_LAYOUT_MISMATCH,
                                 "tensor axis carries block data without a block axis", {}, tensor.id);
        }
        if (tensor.axis.block_axis != UINT8_MAX) {
            const uint64_t dimension = tensor.logical_dimensions[tensor.axis.block_axis];
            if (dimension % tensor.axis.block_elements != 0 ||
                tensor.layout.block_rank == 0 || tensor.layout.block_elements != tensor.axis.block_elements ||
                tensor.layout.block_bytes != tensor.axis.bytes_per_block ||
                tensor.quantization.kind == QuantizationKind::None ||
                tensor.quantization.block_elements != tensor.axis.block_elements ||
                tensor.quantization.block_bytes != tensor.axis.bytes_per_block) {
                return index_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                     "tensor block size disagrees across axis, layout, and quantization contracts",
                                     {}, tensor.id);
            }
        } else if (tensor.layout.block_rank != 0 || tensor.layout.block_elements != 0 ||
                   tensor.layout.block_bytes != 0) {
            return index_failure(CompatibilityError::IR_LAYOUT_MISMATCH,
                                 "tensor layout carries block data without a block axis", {}, tensor.id);
        }
        if (tensor.format.encoding != ArtifactPhysicalEncoding::Unknown) {
            const auto expected = expected_physical_format(tensor.format.encoding);
            const auto values_plane = std::find_if(
                tensor.planes.begin(), tensor.planes.end(), [](const ArtifactTensorPlane& plane) {
                    return plane.kind == PlaneKind::Values;
                });
            if (!expected || tensor.format.version != expected->version || tensor.format != *expected ||
                values_plane == tensor.planes.end() ||
                values_plane->storage_type != expected->value_type) {
                return index_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                     "tensor physical format descriptor is not an exact supported contract", {}, tensor.id);
            }
            const bool gguf_quantized = tensor.format.encoding == ArtifactPhysicalEncoding::Q4_K ||
                                        tensor.format.encoding == ArtifactPhysicalEncoding::Q5_0 ||
                                        tensor.format.encoding == ArtifactPhysicalEncoding::Q6_K ||
                                        tensor.format.encoding == ArtifactPhysicalEncoding::Q8_0 ||
                                        tensor.format.encoding == ArtifactPhysicalEncoding::Q4_0;
            const bool grouped_affine =
                tensor.format.encoding == ArtifactPhysicalEncoding::GroupedAffineU2_256;
            const bool column_grouped_affine =
                tensor.format.encoding == ArtifactPhysicalEncoding::ColumnGroupedAffineU2Skip256;
            const bool quantized = gguf_quantized || grouped_affine || column_grouped_affine;
            if (gguf_quantized && tensor.axis.block_axis != 0) {
                return index_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                     "GGUF quantized tensor blocks must use physical axis 0", {}, tensor.id);
            }
            if (grouped_affine &&
                (tensor.layout.kind != PhysicalLayoutKind::GroupedAffine ||
                 tensor.layout.packing != PackingKind::LsbBitPacked ||
                 tensor.axis.block_axis != 1 || tensor.quantization.bias_type != ScalarType::F16 ||
                 tensor.quantization.required_plane_mask != 7)) {
                return index_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                     "grouped affine tensor physical contracts disagree", {}, tensor.id);
            }
            if (grouped_affine) {
                const auto plane_is = [&](PlaneKind kind, ArtifactScalarType type) {
                    return std::count_if(tensor.planes.begin(), tensor.planes.end(),
                                         [&](const ArtifactTensorPlane& plane) {
                                             return plane.kind == kind && plane.storage_type == type;
                                         }) == 1;
                };
                if (tensor.planes.size() != 3 ||
                    !plane_is(PlaneKind::Values, ArtifactScalarType::U32) ||
                    !plane_is(PlaneKind::Scales, ArtifactScalarType::F16) ||
                    !plane_is(PlaneKind::Biases, ArtifactScalarType::F16)) {
                    return index_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                         "grouped affine tensor plane set is not exact", {}, tensor.id);
                }
            }
            if (column_grouped_affine &&
                (tensor.layout.kind != PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip ||
                 tensor.layout.packing != PackingKind::LsbBitPacked ||
                 tensor.layout.block_elements != 256 || tensor.layout.block_bytes != 64 ||
                 tensor.axis.block_axis != 0 || tensor.quantization.bias_type != ScalarType::F16 ||
                 tensor.quantization.required_plane_mask != 7)) {
                return index_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                     "column-grouped affine tensor physical contracts disagree", {}, tensor.id);
            }
            if (column_grouped_affine) {
                const auto plane_is = [&](PlaneKind kind, ArtifactScalarType type) {
                    return std::count_if(tensor.planes.begin(), tensor.planes.end(),
                                         [&](const ArtifactTensorPlane& plane) {
                                             return plane.kind == kind && plane.storage_type == type;
                                         }) == 1;
                };
                if (tensor.planes.size() != 3 ||
                    !plane_is(PlaneKind::Values, ArtifactScalarType::U8) ||
                    !plane_is(PlaneKind::Scales, ArtifactScalarType::F16) ||
                    !plane_is(PlaneKind::Biases, ArtifactScalarType::F16)) {
                    return index_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                         "column-grouped affine tensor plane set is not exact", {}, tensor.id);
                }
            }
            uint64_t expected_row_stride = tensor.format.block_bytes;
            if (column_grouped_affine) {
                if (tensor.logical_dimensions.size() != 2 ||
                    !checked_multiply(tensor.logical_dimensions[1],
                                      tensor.format.block_bytes,
                                      expected_row_stride)) {
                    return index_failure(CompatibilityError::IR_SHAPE_MISMATCH,
                                         "column-grouped affine row stride overflows",
                                         {}, tensor.id);
                }
            } else if (!tensor.logical_dimensions.empty()) {
                size_t physical_axis = tensor.axis.block_axis;
                if (!quantized) {
                    physical_axis = row_major_layout_is_exact(tensor)
                        ? tensor.logical_dimensions.size() - 1
                        : 0;
                }
                const uint64_t block_dimension =
                    tensor.logical_dimensions[physical_axis];
                const uint64_t units = block_dimension / tensor.format.block_elements;
                if (!checked_multiply(units, tensor.format.block_bytes, expected_row_stride)) {
                    return index_failure(CompatibilityError::IR_SHAPE_MISMATCH,
                                         "tensor physical row stride overflows", {}, tensor.id);
                }
            }
            if (quantized != (tensor.quantization.kind != QuantizationKind::None) ||
                tensor.axis.block_elements != (quantized ? expected->block_elements : 0) ||
                tensor.axis.bytes_per_block != (quantized ? expected->block_bytes : 0) ||
                tensor.layout.block_elements != (quantized ? expected->block_elements : 0) ||
                tensor.layout.block_bytes != (quantized ? expected->block_bytes : 0) ||
                tensor.quantization.block_elements != (quantized ? expected->block_elements : 0) ||
                tensor.quantization.block_bytes != (quantized ? expected->block_bytes : 0) ||
                static_cast<uint16_t>(tensor.quantization.scale_type) !=
                    static_cast<uint16_t>(expected->scale_type) ||
                static_cast<uint16_t>(tensor.quantization.zero_type) !=
                    static_cast<uint16_t>(expected->zero_type) ||
                static_cast<uint16_t>(tensor.quantization.bias_type) !=
                    static_cast<uint16_t>(expected->bias_type) ||
                tensor.axis.row_stride_bytes != expected_row_stride) {
                return index_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                     "tensor physical format fields disagree across its contracts", {}, tensor.id);
            }
        }
        std::sort(tensor.role_evidence.begin(), tensor.role_evidence.end(),
                  [](const ArtifactTensorRoleEvidence& left, const ArtifactTensorRoleEvidence& right) {
                      return std::tie(left.role, left.evidence_key.value, left.authority,
                                      left.scope.root, left.scope.layer, left.scope.slot,
                                      left.scope.instance, left.scope.expert, left.scope.bank_axis,
                                      left.scope.bank_extent, left.scope.bank_stride) <
                             std::tie(right.role, right.evidence_key.value, right.authority,
                                      right.scope.root, right.scope.layer, right.scope.slot,
                                      right.scope.instance, right.scope.expert, right.scope.bank_axis,
                                      right.scope.bank_extent, right.scope.bank_stride);
                  });
        for (size_t role_index = 0; role_index != tensor.role_evidence.size(); ++role_index) {
            const ArtifactTensorRoleEvidence& evidence = tensor.role_evidence[role_index];
            if (!valid_authority(evidence.authority) || evidence.evidence_key.value == UINT32_MAX ||
                !valid_coordinate(evidence.scope) ||
                (role_index != 0 && tensor.role_evidence[role_index - 1] == evidence)) {
                return index_failure(CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS,
                                     "tensor role evidence is invalid or duplicated", {}, tensor.id,
                                     evidence.evidence_key);
            }
        }

        std::sort(tensor.planes.begin(), tensor.planes.end(), [](const ArtifactTensorPlane& left,
                                                                 const ArtifactTensorPlane& right) {
            return std::tie(left.kind, left.source.artifact_id.value, left.source.offset, left.source.length) <
                   std::tie(right.kind, right.source.artifact_id.value, right.source.offset, right.source.length);
        });
        if (tensor.planes.empty()) {
            return index_failure(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                                 "tensor has no physical planes", {}, tensor.id);
        }
        uint32_t actual_plane_mask = 0;
        for (size_t plane_index = 0; plane_index != tensor.planes.size(); ++plane_index) {
            const ArtifactTensorPlane& plane = tensor.planes[plane_index];
            if (!valid_plane(plane.kind) || !valid_scalar(plane.storage_type) ||
                plane.elements_per_block == 0 || plane.bytes_per_block == 0 ||
                plane.alignment == 0) {
                return index_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                     "tensor plane contract is incomplete", plane.source.artifact_id, tensor.id);
            }
            if (plane_index != 0 && tensor.planes[plane_index - 1].kind == plane.kind) {
                return index_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                     "tensor has duplicate physical plane kinds", plane.source.artifact_id, tensor.id);
            }
            if (!span_in_artifact(plane.source, input.artifacts) || plane.source.offset % plane.alignment != 0) {
                CompatibilityReport report = index_failure(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                                                           "tensor plane is outside or misaligned in its immutable source",
                                                           plane.source.artifact_id, tensor.id);
                report.source_offset = plane.source.offset;
                report.source_length = plane.source.length;
                return report;
            }
            if (plane.kind == PlaneKind::Values && plane.logical_elements != element_count) {
                return index_failure(CompatibilityError::IR_SHAPE_MISMATCH,
                                     "tensor value plane does not cover its logical element count",
                                     plane.source.artifact_id, tensor.id);
            }
            const uint64_t blocks = plane.logical_elements / plane.elements_per_block +
                                    (plane.logical_elements % plane.elements_per_block != 0);
            uint64_t expected_bytes = 0;
            if (!checked_multiply(blocks, plane.bytes_per_block, expected_bytes) ||
                expected_bytes != plane.source.length) {
                return index_failure(CompatibilityError::IR_SHAPE_MISMATCH,
                                     "tensor plane byte length does not match its block contract",
                                     plane.source.artifact_id, tensor.id);
            }
            actual_plane_mask |= artifact_plane_mask(plane.kind);
            if (plane.source.length != 0) {
                spans.push_back({plane.source.artifact_id, plane.source.offset,
                                 plane.source.offset + plane.source.length, tensor.id, plane.kind});
            }
            if (plane.kind == PlaneKind::Values && tensor.coordinate.bank_axis != UINT8_MAX) {
                uint64_t bank_bytes = 0;
                if (!checked_multiply(tensor.coordinate.bank_stride, tensor.coordinate.bank_extent, bank_bytes) ||
                    bank_bytes != plane.source.length) {
                    return index_failure(CompatibilityError::IR_LAYOUT_MISMATCH,
                                         "tensor bank coordinate stride does not cover its values plane",
                                         plane.source.artifact_id, tensor.id);
                }
            }
        }
        if ((actual_plane_mask & tensor.quantization.required_plane_mask) !=
            tensor.quantization.required_plane_mask) {
            return index_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                 "tensor is missing a required physical plane", {}, tensor.id);
        }
        if (tensor.quantization.kind == QuantizationKind::None &&
            actual_plane_mask != artifact_plane_mask(PlaneKind::Values)) {
            return index_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                 "unquantized tensor must contain exactly one values plane", {}, tensor.id);
        }
    }

    std::vector<const ArtifactTensorRecord*> structural;
    structural.reserve(input.tensors.size());
    for (const ArtifactTensorRecord& tensor : input.tensors) {
        if (coordinate_is_set(tensor.coordinate)) structural.push_back(&tensor);
    }
    std::sort(structural.begin(), structural.end(), [](const auto* left, const auto* right) {
        return coordinate_less(left->coordinate, right->coordinate);
    });
    for (size_t index = 1; index < structural.size(); ++index) {
        if (structural[index - 1]->coordinate == structural[index]->coordinate) {
            return index_failure(CompatibilityError::IMPORT_TENSOR_DUPLICATE,
                                 "artifact index contains duplicate structural tensor coordinates",
                                 {}, structural[index]->id);
        }
    }

    std::sort(input.aliases.begin(), input.aliases.end(), [](const ArtifactAlias& left, const ArtifactAlias& right) {
        return std::tie(left.source_tensor_id, left.target_tensor_id, left.kind, left.direction,
                        left.semantic_role, left.proof_key.value, left.proof_authority,
                        left.proof_scope.root, left.proof_scope.layer, left.proof_scope.slot,
                        left.proof_scope.instance, left.proof_scope.expert, left.proof_scope.bank_axis,
                        left.proof_scope.bank_extent, left.proof_scope.bank_stride) <
               std::tie(right.source_tensor_id, right.target_tensor_id, right.kind, right.direction,
                        right.semantic_role, right.proof_key.value, right.proof_authority,
                        right.proof_scope.root, right.proof_scope.layer, right.proof_scope.slot,
                        right.proof_scope.instance, right.proof_scope.expert, right.proof_scope.bank_axis,
                        right.proof_scope.bank_extent, right.proof_scope.bank_stride);
    });
    for (size_t alias_index = 0; alias_index != input.aliases.size(); ++alias_index) {
        const ArtifactAlias& alias = input.aliases[alias_index];
        if ((alias.kind != ArtifactAliasKind::ExactSharedSpan && alias.kind != ArtifactAliasKind::TiedOutput) ||
            (alias.direction != ArtifactAliasDirection::Bidirectional &&
             alias.direction != ArtifactAliasDirection::SourceToTarget) ||
            !valid_authority(alias.proof_authority) || !valid_coordinate(alias.proof_scope) ||
            alias.source_tensor_id == alias.target_tensor_id ||
            (alias_index != 0 && input.aliases[alias_index - 1] == alias)) {
            return index_failure(CompatibilityError::IR_REFERENCE_INVALID,
                                 "artifact alias declaration is invalid or duplicated", {}, alias.source_tensor_id);
        }
        const ArtifactTensorRecord* source = tensor_by_id(input.tensors, alias.source_tensor_id);
        const ArtifactTensorRecord* target = tensor_by_id(input.tensors, alias.target_tensor_id);
        if (!source || !target || !identical_physical_contract(*source, *target)) {
            return index_failure(CompatibilityError::IR_REFERENCE_INVALID,
                                 "artifact alias does not name tensors with identical source planes",
                                 {}, alias.source_tensor_id);
        }
        if (std::none_of(target->role_evidence.begin(), target->role_evidence.end(), [&](const auto& evidence) {
                return evidence.role == alias.semantic_role;
            })) {
            return index_failure(CompatibilityError::IR_REFERENCE_INVALID,
                                 "artifact alias semantic role is not supported by target evidence",
                                 {}, alias.target_tensor_id);
        }
    }

    std::sort(spans.begin(), spans.end(), [](const SpanUse& left, const SpanUse& right) {
        return std::tie(left.artifact_id.value, left.begin, left.end, left.tensor_id, left.plane) <
               std::tie(right.artifact_id.value, right.begin, right.end, right.tensor_id, right.plane);
    });
    for (size_t left_index = 0; left_index != spans.size(); ++left_index) {
        const SpanUse& left = spans[left_index];
        for (size_t right_index = left_index + 1; right_index != spans.size(); ++right_index) {
            const SpanUse& right = spans[right_index];
            if (right.artifact_id != left.artifact_id || right.begin >= left.end) break;
            const bool identical = left.begin == right.begin && left.end == right.end;
            if (!identical || left.tensor_id == right.tensor_id ||
                !alias_connects(input.aliases, left.tensor_id, right.tensor_id)) {
                CompatibilityReport report = index_failure(CompatibilityError::IMPORT_TENSOR_DUPLICATE,
                                                           "tensor source planes overlap without an exact alias",
                                                           left.artifact_id, right.tensor_id);
                report.source_offset = right.begin;
                report.source_length = right.end - right.begin;
                return report;
            }
        }
    }

    ArtifactIndex index;
    index.artifacts_ = std::move(input.artifacts);
    index.metadata_facts_ = std::move(input.metadata_facts);
    index.package_facts_ = std::move(input.package_facts);
    index.tensors_ = std::move(input.tensors);
    index.aliases_ = std::move(input.aliases);
    index.diagnostics_ = std::move(input.diagnostics);

    std::vector<uint8_t>& bytes = index.canonical_bytes_;
    static constexpr std::array<uint8_t, 8> magic = {'L', 'A', 'P', 'I', 'D', 'X', '0', '1'};
    bytes.insert(bytes.end(), magic.begin(), magic.end());
    append_u16(bytes, 1);
    append_u16(bytes, 0);
    append_u32(bytes, static_cast<uint32_t>(index.artifacts_.size()));
    append_u32(bytes, static_cast<uint32_t>(index.metadata_facts_.size()));
    append_u32(bytes, static_cast<uint32_t>(index.package_facts_.size()));
    append_u32(bytes, static_cast<uint32_t>(index.tensors_.size()));
    append_u32(bytes, static_cast<uint32_t>(index.aliases_.size()));

    for (const PackageView& artifact : index.artifacts_) {
        append_u8(bytes, static_cast<uint8_t>(artifact.role()));
    }
    for (const ArtifactFact& fact : index.metadata_facts_) append_fact(bytes, fact, false);
    for (const ArtifactFact& fact : index.package_facts_) append_fact(bytes, fact, false);
    std::vector<const ArtifactTensorRecord*> normalized_tensors;
    normalized_tensors.reserve(index.tensors_.size());
    for (const ArtifactTensorRecord& tensor : index.tensors_) normalized_tensors.push_back(&tensor);
    std::sort(normalized_tensors.begin(), normalized_tensors.end(), [](const auto* left, const auto* right) {
        std::vector<uint8_t> lhs;
        std::vector<uint8_t> rhs;
        append_tensor_normalized(lhs, *left);
        append_tensor_normalized(rhs, *right);
        return lhs < rhs;
    });
    for (const ArtifactTensorRecord* tensor : normalized_tensors) append_tensor_normalized(bytes, *tensor);
    for (const ArtifactAlias& alias : index.aliases_) {
        append_u8(bytes, static_cast<uint8_t>(alias.kind));
        append_u8(bytes, static_cast<uint8_t>(alias.direction));
        append_u32(bytes, alias.source_tensor_id);
        append_u32(bytes, alias.target_tensor_id);
        append_u16(bytes, static_cast<uint16_t>(alias.semantic_role));
        append_u32(bytes, alias.proof_key.value);
        append_u8(bytes, static_cast<uint8_t>(alias.proof_authority));
        append_coordinate(bytes, alias.proof_scope);
    }
    index.normalized_digest_ = digest_bytes(bytes);
    index.digest_ = index.normalized_digest_;

    std::vector<uint8_t> provenance;
    static constexpr std::array<uint8_t, 8> provenance_magic = {'L', 'A', 'P', 'R', 'O', 'V', '0', '1'};
    provenance.insert(provenance.end(), provenance_magic.begin(), provenance_magic.end());
    append_u32(provenance, static_cast<uint32_t>(index.artifacts_.size()));
    for (const PackageView& artifact : index.artifacts_) {
        append_u32(provenance, artifact.artifact_id().value);
        append_u8(provenance, static_cast<uint8_t>(artifact.role()));
        append_u64(provenance, artifact.bytes().size());
        append_digest(provenance, artifact.digest());
    }
    provenance.insert(provenance.end(), bytes.begin(), bytes.end());
    for (const ArtifactTensorRecord& tensor : index.tensors_) {
        append_u32(provenance, tensor.id);
        append_u32(provenance, static_cast<uint32_t>(tensor.planes.size()));
        for (const ArtifactTensorPlane& plane : tensor.planes) append_source(provenance, plane.source);
    }
    for (const ArtifactFact& fact : index.metadata_facts_) append_source(provenance, fact.source);
    for (const ArtifactFact& fact : index.package_facts_) append_source(provenance, fact.source);
    index.provenance_digest_ = digest_bytes(provenance);
    return index;
}

} // namespace Laplace
