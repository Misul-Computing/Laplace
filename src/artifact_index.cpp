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
    return kind == PhysicalLayoutKind::ContiguousRowMajor || kind == PhysicalLayoutKind::GgufBlocked;
}

bool valid_packing(PackingKind kind) {
    return kind == PackingKind::None || kind == PackingKind::Gguf;
}

bool power_of_two(uint32_t value) {
    return value != 0 && (value & (value - 1)) == 0;
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

void append_fact(std::vector<uint8_t>& bytes, const ArtifactFact& fact) {
    append_u32(bytes, fact.key.value);
    append_u8(bytes, static_cast<uint8_t>(fact.authority));
    append_source(bytes, fact.source);
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
        } else if constexpr (std::is_same_v<T, Sha256Digest>) {
            append_digest(bytes, value);
        }
    }, fact.value);
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

bool identical_plane_sets(const ArtifactTensorRecord& left, const ArtifactTensorRecord& right) {
    if (left.logical_dimensions != right.logical_dimensions || left.planes.size() != right.planes.size()) return false;
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
            if (fact.key.value == UINT32_MAX || !valid_authority(fact.authority)) {
                return index_failure(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                     "artifact fact lacks a canonical key or authority",
                                     fact.source.artifact_id, UINT32_MAX, fact.key);
            }
            if (!span_in_artifact(fact.source, input.artifacts)) {
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
        if (!valid_scalar(tensor.logical_type) || tensor.logical_dimensions.size() > tensor.layout.axis_order.size() ||
            tensor.layout.rank != tensor.logical_dimensions.size()) {
            return index_failure(CompatibilityError::IR_SHAPE_MISMATCH,
                                 "tensor logical rank or scalar type is invalid", {}, tensor.id);
        }
        uint64_t element_count = 1;
        for (uint64_t dimension : tensor.logical_dimensions) {
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
        if (tensor.layout.kind == PhysicalLayoutKind::ContiguousRowMajor &&
            (!identity_axes || !row_major_layout_is_exact(tensor))) {
            return index_failure(CompatibilityError::IR_LAYOUT_MISMATCH,
                                 "contiguous tensor axes or strides are not canonical row-major", {}, tensor.id);
        }

        if (!valid_quantization(tensor.quantization.kind) ||
            (tensor.quantization.kind != QuantizationKind::None &&
             (tensor.quantization.block_elements == 0 || tensor.quantization.block_bytes == 0))) {
            return index_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                 "tensor quantization block contract is incomplete", {}, tensor.id);
        }
        std::sort(tensor.role_evidence.begin(), tensor.role_evidence.end(),
                  [](const ArtifactTensorRoleEvidence& left, const ArtifactTensorRoleEvidence& right) {
                      return std::tie(left.role, left.evidence_key.value, left.authority) <
                             std::tie(right.role, right.evidence_key.value, right.authority);
                  });
        for (size_t role_index = 0; role_index != tensor.role_evidence.size(); ++role_index) {
            const ArtifactTensorRoleEvidence& evidence = tensor.role_evidence[role_index];
            if (!valid_authority(evidence.authority) || evidence.evidence_key.value == UINT32_MAX ||
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
                !power_of_two(plane.alignment)) {
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

    std::sort(input.aliases.begin(), input.aliases.end(), [](const ArtifactAlias& left, const ArtifactAlias& right) {
        return std::tie(left.source_tensor_id, left.target_tensor_id, left.kind, left.direction, left.semantic_role) <
               std::tie(right.source_tensor_id, right.target_tensor_id, right.kind, right.direction, right.semantic_role);
    });
    for (size_t alias_index = 0; alias_index != input.aliases.size(); ++alias_index) {
        const ArtifactAlias& alias = input.aliases[alias_index];
        if ((alias.kind != ArtifactAliasKind::ExactSharedSpan && alias.kind != ArtifactAliasKind::TiedOutput) ||
            (alias.direction != ArtifactAliasDirection::Bidirectional &&
             alias.direction != ArtifactAliasDirection::SourceToTarget) ||
            alias.source_tensor_id == alias.target_tensor_id ||
            (alias_index != 0 && input.aliases[alias_index - 1] == alias)) {
            return index_failure(CompatibilityError::IR_REFERENCE_INVALID,
                                 "artifact alias declaration is invalid or duplicated", {}, alias.source_tensor_id);
        }
        const ArtifactTensorRecord* source = tensor_by_id(input.tensors, alias.source_tensor_id);
        const ArtifactTensorRecord* target = tensor_by_id(input.tensors, alias.target_tensor_id);
        if (!source || !target || !identical_plane_sets(*source, *target)) {
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
        append_u32(bytes, artifact.artifact_id().value);
        append_u8(bytes, static_cast<uint8_t>(artifact.role()));
        append_u64(bytes, artifact.bytes().size());
        append_digest(bytes, artifact.digest());
    }
    for (const ArtifactFact& fact : index.metadata_facts_) append_fact(bytes, fact);
    for (const ArtifactFact& fact : index.package_facts_) append_fact(bytes, fact);
    for (const ArtifactTensorRecord& tensor : index.tensors_) {
        append_u32(bytes, tensor.id);
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
        }
        append_u32(bytes, static_cast<uint32_t>(tensor.planes.size()));
        for (const ArtifactTensorPlane& plane : tensor.planes) {
            append_u16(bytes, static_cast<uint16_t>(plane.kind));
            append_u16(bytes, static_cast<uint16_t>(plane.storage_type));
            append_source(bytes, plane.source);
            append_u64(bytes, plane.logical_elements);
            append_u32(bytes, plane.bytes_per_block);
            append_u32(bytes, plane.elements_per_block);
            append_u32(bytes, plane.alignment);
        }
    }
    for (const ArtifactAlias& alias : index.aliases_) {
        append_u8(bytes, static_cast<uint8_t>(alias.kind));
        append_u8(bytes, static_cast<uint8_t>(alias.direction));
        append_u32(bytes, alias.source_tensor_id);
        append_u32(bytes, alias.target_tensor_id);
        append_u16(bytes, static_cast<uint16_t>(alias.semantic_role));
    }
    index.digest_ = digest_bytes(bytes);
    return index;
}

} // namespace Laplace
