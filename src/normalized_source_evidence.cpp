#include "normalized_source_evidence.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace Laplace {
namespace {

CompatibilityReport failure(CompatibilityError code, const char* detail) {
    return compatibility_report(code, detail);
}

void append_u8(std::vector<uint8_t>& bytes, uint8_t value) {
    bytes.push_back(value);
}

void append_u16(std::vector<uint8_t>& bytes, uint16_t value) {
    append_u8(bytes, static_cast<uint8_t>(value));
    append_u8(bytes, static_cast<uint8_t>(value >> 8));
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8) {
        append_u8(bytes, static_cast<uint8_t>(value >> shift));
    }
}

void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8) {
        append_u8(bytes, static_cast<uint8_t>(value >> shift));
    }
}

bool append_count(std::vector<uint8_t>& bytes, size_t value) {
    if (value > UINT32_MAX) return false;
    append_u32(bytes, static_cast<uint32_t>(value));
    return true;
}

void append_coordinate(std::vector<uint8_t>& bytes,
                       const ArtifactCoordinate& coordinate) {
    append_u32(bytes, coordinate.root);
    append_u32(bytes, coordinate.layer);
    append_u32(bytes, coordinate.slot);
    append_u32(bytes, coordinate.instance);
    append_u32(bytes, coordinate.expert);
    append_u8(bytes, coordinate.bank_axis);
    append_u32(bytes, coordinate.bank_extent);
    append_u64(bytes, coordinate.bank_stride);
}

void append_layout(std::vector<uint8_t>& bytes, const PhysicalLayout& layout) {
    append_u16(bytes, static_cast<uint16_t>(layout.kind));
    append_u16(bytes, layout.version);
    append_u16(bytes, static_cast<uint16_t>(layout.packing));
    append_u8(bytes, layout.rank);
    append_u8(bytes, layout.block_rank);
    for (uint8_t axis : layout.axis_order) append_u8(bytes, axis);
    for (uint64_t stride : layout.strides) append_u64(bytes, stride);
    append_u32(bytes, layout.block_elements);
    append_u32(bytes, layout.block_bytes);
    append_u32(bytes, layout.flags);
}

void append_format(std::vector<uint8_t>& bytes,
                   const ArtifactPhysicalFormat& format) {
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
    append_u16(bytes, static_cast<uint16_t>(format.bias_type));
    append_u32(bytes, format.bias_bytes);
}

void append_quantization(std::vector<uint8_t>& bytes,
                         const Quantization& quantization) {
    append_u16(bytes, static_cast<uint16_t>(quantization.kind));
    append_u16(bytes, quantization.version);
    append_u16(bytes, static_cast<uint16_t>(quantization.accumulation_type));
    append_u16(bytes, static_cast<uint16_t>(quantization.scale_type));
    append_u16(bytes, static_cast<uint16_t>(quantization.zero_type));
    append_u16(bytes, static_cast<uint16_t>(quantization.bias_type));
    append_u32(bytes, quantization.block_elements);
    append_u32(bytes, quantization.block_bytes);
    append_u32(bytes, quantization.group_size);
    append_u32(bytes, quantization.required_plane_mask);
    append_u32(bytes, quantization.flags);
}

bool append_tensor(std::vector<uint8_t>& bytes,
                   const NormalizedTensorEvidence& tensor) {
    append_u16(bytes, static_cast<uint16_t>(tensor.role));
    append_coordinate(bytes, tensor.coordinate);
    append_u8(bytes, static_cast<uint8_t>(tensor.strength));
    append_u16(bytes, static_cast<uint16_t>(tensor.logical_type));
    if (!append_count(bytes, tensor.dimensions.size())) return false;
    for (uint64_t dimension : tensor.dimensions) append_u64(bytes, dimension);
    append_layout(bytes, tensor.layout);
    append_format(bytes, tensor.format);
    append_quantization(bytes, tensor.quantization);
    if (!append_count(bytes, tensor.planes.size())) return false;
    for (const NormalizedPlaneEvidence& plane : tensor.planes) {
        append_u16(bytes, static_cast<uint16_t>(plane.kind));
        append_u16(bytes, static_cast<uint16_t>(plane.storage_type));
        append_u64(bytes, plane.logical_elements);
        append_u32(bytes, plane.bytes_per_block);
        append_u32(bytes, plane.elements_per_block);
        append_u32(bytes, plane.alignment);
    }
    return true;
}

bool valid_coordinate(const ArtifactCoordinate& coordinate) {
    if (coordinate.root == UINT32_MAX) return false;
    if (coordinate.bank_axis == UINT8_MAX) {
        return coordinate.bank_extent == 0 && coordinate.bank_stride == 0;
    }
    return coordinate.bank_extent != 0 && coordinate.bank_stride != 0;
}

bool valid_tensor(const NormalizedEvidenceCandidate& candidate) {
    const NormalizedTensorEvidence& tensor = candidate.evidence;
    const NormalizedTensorBinding& binding = candidate.binding;
    if (binding.source_tensor_id == UINT32_MAX ||
        tensor.role == static_cast<TensorRole>(0) ||
        static_cast<uint16_t>(tensor.role) > static_cast<uint16_t>(TensorRole::OutputScaleWeight) ||
        !valid_coordinate(tensor.coordinate) ||
        tensor.logical_type == ArtifactScalarType::None ||
        tensor.dimensions.empty() || tensor.dimensions.size() > kNormalizedSourceEvidenceMaximumDimensions ||
        tensor.layout.rank != tensor.dimensions.size() ||
        tensor.format.version == 0 ||
        tensor.format.encoding == ArtifactPhysicalEncoding::Unknown ||
        tensor.format.block_elements == 0 || tensor.format.block_bytes == 0 ||
        tensor.planes.empty() || tensor.planes.size() > kNormalizedSourceEvidenceMaximumPlanes ||
        tensor.planes.size() != binding.planes.size()) {
        return false;
    }
    for (uint64_t dimension : tensor.dimensions) {
        if (dimension == 0) return false;
    }
    for (const NormalizedPlaneEvidence& plane : tensor.planes) {
        if (plane.kind == static_cast<PlaneKind>(0) ||
            plane.storage_type == ArtifactScalarType::None ||
            plane.logical_elements == 0 || plane.alignment == 0 ||
            (plane.alignment & (plane.alignment - 1)) != 0) {
            return false;
        }
    }
    for (size_t index = 0; index != tensor.planes.size(); ++index) {
        for (size_t previous = 0; previous != index; ++previous) {
            if (tensor.planes[previous].kind == tensor.planes[index].kind) return false;
        }
        const ArtifactSourceSpan& source = binding.planes[index];
        if (source.artifact_id.value == UINT32_MAX || source.length == 0) return false;
    }
    return true;
}

bool same_role_coordinate(const NormalizedTensorEvidence& left,
                          const NormalizedTensorEvidence& right) {
    return left.role == right.role && left.coordinate == right.coordinate;
}

bool same_binding(const NormalizedTensorBinding& left,
                  const NormalizedTensorBinding& right) {
    return left.source_tensor_id == right.source_tensor_id &&
           left.planes == right.planes;
}

bool less_source_spans(const std::vector<ArtifactSourceSpan>& left,
                       const std::vector<ArtifactSourceSpan>& right) {
    const size_t common = std::min(left.size(), right.size());
    for (size_t index = 0; index != common; ++index) {
        const ArtifactSourceSpan& l = left[index];
        const ArtifactSourceSpan& r = right[index];
        if (l.artifact_id.value != r.artifact_id.value) {
            return l.artifact_id.value < r.artifact_id.value;
        }
        if (l.offset != r.offset) return l.offset < r.offset;
        if (l.length != r.length) return l.length < r.length;
    }
    return left.size() < right.size();
}

bool less_candidate(const NormalizedEvidenceCandidate& left,
                    const NormalizedEvidenceCandidate& right) {
    std::vector<uint8_t> left_bytes;
    std::vector<uint8_t> right_bytes;
    append_tensor(left_bytes, left.evidence);
    append_tensor(right_bytes, right.evidence);
    if (left_bytes != right_bytes) return left_bytes < right_bytes;
    if (left.binding.source_tensor_id != right.binding.source_tensor_id) {
        return left.binding.source_tensor_id < right.binding.source_tensor_id;
    }
    return less_source_spans(left.binding.planes, right.binding.planes);
}

} // namespace

NormalizedSourceEvidenceResult
normalize_source_evidence(std::span<const NormalizedEvidenceCandidate> candidates) {
    if (candidates.empty()) {
        return failure(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                       "normalized source evidence contains no candidates");
    }
    if (candidates.size() > kNormalizedSourceEvidenceMaximumCandidates) {
        return failure(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                       "normalized source evidence exceeds its bounded candidate limit");
    }

    // Check vector bounds before copying caller-owned candidates. This keeps
    // malformed input from turning the normalization boundary into an
    // unbounded allocation path.
    for (const NormalizedEvidenceCandidate& candidate : candidates) {
        if (candidate.evidence.dimensions.size() > kNormalizedSourceEvidenceMaximumDimensions ||
            candidate.evidence.planes.size() > kNormalizedSourceEvidenceMaximumPlanes ||
            candidate.binding.planes.size() > kNormalizedSourceEvidenceMaximumPlanes) {
            return failure(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                           "normalized source evidence contains an oversized bounded field");
        }
    }

    std::vector<NormalizedEvidenceCandidate> ordered(candidates.begin(), candidates.end());
    for (const NormalizedEvidenceCandidate& candidate : ordered) {
        if (!valid_tensor(candidate)) {
            return failure(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                           "normalized source evidence has an invalid typed candidate");
        }
    }

    for (size_t index = 0; index != ordered.size(); ++index) {
        for (size_t previous = 0; previous != index; ++previous) {
            const auto& left = ordered[previous];
            const auto& right = ordered[index];
            if (left.binding.source_tensor_id == right.binding.source_tensor_id) {
                if (left.evidence == right.evidence && same_binding(left.binding, right.binding)) {
                    return failure(CompatibilityError::IMPORT_TENSOR_DUPLICATE,
                                   "normalized source evidence repeats a source tensor");
                }
                return failure(CompatibilityError::IMPORT_RULE_CONFLICT,
                               "normalized source evidence conflicts for one source tensor");
            }
            if (same_role_coordinate(left.evidence, right.evidence)) {
                if (left.evidence == right.evidence) {
                    return failure(CompatibilityError::IMPORT_TENSOR_DUPLICATE,
                                   "normalized source evidence has duplicate role coordinates");
                }
                return failure(CompatibilityError::IMPORT_RULE_CONFLICT,
                               "normalized source evidence has conflicting physical facts");
            }
        }
    }

    std::sort(ordered.begin(), ordered.end(), less_candidate);
    NormalizedSourceEvidence result;
    result.tensors_.reserve(ordered.size());
    result.bindings_.reserve(ordered.size());
    for (size_t index = 0; index != ordered.size(); ++index) {
        result.tensors_.push_back(ordered[index].evidence);
        NormalizedTensorBinding binding = ordered[index].binding;
        binding.ordinal = static_cast<uint32_t>(index);
        result.bindings_.push_back(std::move(binding));
    }

    constexpr std::string_view domain = "laplace-normalized-source-evidence-v1\0";
    result.canonical_bytes_.reserve(64 + ordered.size() * 256);
    result.canonical_bytes_.insert(result.canonical_bytes_.end(), domain.begin(), domain.end());
    append_u16(result.canonical_bytes_, result.version_);
    append_u32(result.canonical_bytes_, static_cast<uint32_t>(result.tensors_.size()));
    for (const NormalizedTensorEvidence& tensor : result.tensors_) {
        if (!append_tensor(result.canonical_bytes_, tensor)) {
            return failure(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                           "normalized source evidence canonical bytes overflow");
        }
    }

    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    size_t offset = 0;
    while (offset != result.canonical_bytes_.size()) {
        const size_t remaining = result.canonical_bytes_.size() - offset;
        const size_t chunk = std::min<size_t>(remaining, std::numeric_limits<CC_LONG>::max());
        CC_SHA256_Update(&context, result.canonical_bytes_.data() + offset,
                         static_cast<CC_LONG>(chunk));
        offset += chunk;
    }
    CC_SHA256_Final(result.canonical_digest_.bytes.data(), &context);
    return result;
}

} // namespace Laplace
