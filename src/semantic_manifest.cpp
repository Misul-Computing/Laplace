#include "semantic_manifest.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <unordered_set>

namespace Laplace {

struct SemanticManifest::Data {
    ArtifactIndex physical;
    SemanticModel model;
    TokenContract contract;
    std::optional<SourceCompilerGraphProof> graph_proof;
    std::vector<uint8_t> bytes;
    std::vector<uint8_t> semantic_bytes;
    std::optional<PackageView> carrier;
    ArtifactId artifact_id{};
    uint64_t artifact_size = 0;
    Sha256Digest artifact_digest{};
    Sha256Digest semantic_graph_digest{};
    Sha256Digest physical_binding_set_digest{};
    Sha256Digest interaction_contract_digest{};
    Sha256Digest package_fingerprint{};
    Sha256Digest body_digest{};
    Sha256Digest record_digest{};
};

namespace {

constexpr std::array<uint8_t, 8> kMagic = {'L', 'A', 'P', 'M', 'A', 'N', '0', '1'};
constexpr uint16_t kMajor = 1;
constexpr uint16_t kMinor = 0;
constexpr uint32_t kHeaderBytes = 64;
constexpr uint32_t kEnvelopeDigestBytes = 32;
// Body version 4 additionally carries one canonical, typed token authority
// descriptor after the transport payload. A complete LAPIR payload remains
// transport data, not graph identity.
constexpr uint32_t kBodyVersion = 4;
constexpr uint32_t kMaxBodyBytes = 16u * 1024u * 1024u + 64u * 1024u;
constexpr uint32_t kMaxStops = 1048576;
constexpr uint32_t kMaxArtifacts = 4096;

CompatibilityReport manifest_error(CompatibilityError code, std::string detail) {
    CompatibilityReport report = compatibility_report(code, std::move(detail));
    report.stage = CompatibilityStage::Semantic;
    report.phase = CompatibilityPhase::Semantic;
    return report;
}

CompatibilityReport carrier_error(CompatibilityError code, std::string detail) {
    return package_report(code, std::move(detail));
}

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

void append_digest(std::vector<uint8_t>& bytes, const Sha256Digest& digest) {
    bytes.insert(bytes.end(), digest.bytes.begin(), digest.bytes.end());
}

Sha256Digest digest_bytes(std::span<const uint8_t> bytes) {
    Sha256Digest digest;
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    for (size_t offset = 0; offset < bytes.size();) {
        const size_t chunk = std::min<size_t>(1024 * 1024, bytes.size() - offset);
        CC_SHA256_Update(&context, bytes.data() + offset, static_cast<CC_LONG>(chunk));
        offset += chunk;
    }
    CC_SHA256_Final(digest.bytes.data(), &context);
    return digest;
}

bool checked_multiply(uint64_t left, uint64_t right, uint64_t& result) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) return false;
    result = left * right;
    return true;
}

bool fits_size(uint64_t value) {
    return value <= std::numeric_limits<size_t>::max();
}

bool spans_overlap(std::span<const uint8_t> left, std::span<const uint8_t> right) {
    if (left.empty() || right.empty()) return false;
    const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left.data());
    const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right.data());
    if (left_begin > std::numeric_limits<uintptr_t>::max() - left.size() ||
        right_begin > std::numeric_limits<uintptr_t>::max() - right.size()) {
        return true;
    }
    const uintptr_t left_end = left_begin + left.size();
    const uintptr_t right_end = right_begin + right.size();
    return left_begin < right_end && right_begin < left_end;
}

class Reader {
public:
    explicit Reader(std::span<const uint8_t> bytes) : bytes_(bytes) {}

    size_t position() const noexcept { return offset_; }
    size_t remaining() const noexcept { return bytes_.size() - offset_; }
    std::span<const uint8_t> bytes() const noexcept { return bytes_; }

    bool take(size_t count, std::span<const uint8_t>& result) {
        if (count > remaining()) return false;
        result = bytes_.subspan(offset_, count);
        offset_ += count;
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

    bool u64(uint64_t& result) {
        std::span<const uint8_t> value;
        if (!take(8, value)) return false;
        result = 0;
        for (unsigned shift = 0; shift != 64; shift += 8) {
            result |= static_cast<uint64_t>(value[shift / 8]) << shift;
        }
        return true;
    }

private:
    std::span<const uint8_t> bytes_;
    size_t offset_ = 0;
};

bool valid_token_contract(const TokenContract& contract) {
    return contract.validate().ok();
}

bool token_fields_match_model(const TokenContract& contract, const SemanticModel& model) {
    return contract.vocabulary_size == model.vocabulary_size &&
           contract.bos_id == model.bos_id && contract.eos_id == model.eos_id &&
           contract.stop_ids == model.stop_ids &&
           contract.authoritative_tokenizer_digest.bytes == model.tokenizer_digest &&
           contract.authoritative_template_digest.bytes == model.template_digest;
}

const PackageView* package_by_id(const ArtifactIndex& physical, ArtifactId id) {
    for (const PackageView& artifact : physical.artifacts()) {
        if (artifact.artifact_id() == id) return &artifact;
    }
    return nullptr;
}

bool token_data_reference_matches(const ArtifactIndex& physical,
                                  const TokenArtifactReference& reference) {
    if (reference.artifact_id.value == UINT32_MAX || reference.length == 0) return false;
    const PackageView* artifact = package_by_id(physical, reference.artifact_id);
    if (!artifact || artifact->role() == ArtifactRole::Sidecar ||
        reference.offset > artifact->bytes().size() ||
        reference.length > artifact->bytes().size() - reference.offset) {
        return false;
    }
    return digest_bytes(artifact->bytes().subspan(static_cast<size_t>(reference.offset),
                                                  static_cast<size_t>(reference.length))) == reference.digest;
}

bool token_contract_package_bound(const ArtifactIndex& physical, const TokenContract& contract) {
    return contract.tokenizer_algorithm == TokenizerAlgorithm::TokenIdsOnly ||
           token_data_reference_matches(physical, contract.tokenizer_data);
}

bool supported_format(const ArtifactTensorRecord& tensor) {
    const auto& format = tensor.format;
    if (format.version != 1 && format.version != 2) return false;
    switch (format.encoding) {
    case ArtifactPhysicalEncoding::F32:
        return format == ArtifactPhysicalFormat{1, ArtifactPhysicalEncoding::F32,
                                                 ArtifactScalarType::F32, ArtifactScalarType::None,
                                                 ArtifactScalarType::None, ArtifactScalarType::None,
                                                 ArtifactScalarType::None, 1, 4, 0, 0, 0, 0};
    case ArtifactPhysicalEncoding::F16:
        return format == ArtifactPhysicalFormat{1, ArtifactPhysicalEncoding::F16,
                                                 ArtifactScalarType::F16, ArtifactScalarType::None,
                                                 ArtifactScalarType::None, ArtifactScalarType::None,
                                                 ArtifactScalarType::None, 1, 2, 0, 0, 0, 0};
    case ArtifactPhysicalEncoding::Q4_K:
        return format == ArtifactPhysicalFormat{1, ArtifactPhysicalEncoding::Q4_K,
                                                 ArtifactScalarType::Packed, ArtifactScalarType::F16,
                                                 ArtifactScalarType::F16, ArtifactScalarType::Packed,
                                                 ArtifactScalarType::None, 256, 144, 2, 2, 12, 0};
    case ArtifactPhysicalEncoding::Q5_0:
        return format == ArtifactPhysicalFormat{1, ArtifactPhysicalEncoding::Q5_0,
                                                 ArtifactScalarType::Packed, ArtifactScalarType::F16,
                                                 ArtifactScalarType::None, ArtifactScalarType::None,
                                                 ArtifactScalarType::None, 32, 22, 2, 0, 0, 0};
    case ArtifactPhysicalEncoding::Q4_0:
        return format == ArtifactPhysicalFormat{1, ArtifactPhysicalEncoding::Q4_0,
                                                 ArtifactScalarType::Packed, ArtifactScalarType::F16,
                                                 ArtifactScalarType::None, ArtifactScalarType::None,
                                                 ArtifactScalarType::None, 32, 18, 2, 0, 0, 0};
    case ArtifactPhysicalEncoding::Q6_K:
        return format == ArtifactPhysicalFormat{1, ArtifactPhysicalEncoding::Q6_K,
                                                 ArtifactScalarType::Packed, ArtifactScalarType::F16,
                                                 ArtifactScalarType::None, ArtifactScalarType::I8,
                                                 ArtifactScalarType::None, 256, 210, 2, 0, 16, 0};
    case ArtifactPhysicalEncoding::Q8_0:
        return format == ArtifactPhysicalFormat{1, ArtifactPhysicalEncoding::Q8_0,
                                                 ArtifactScalarType::Packed, ArtifactScalarType::F16,
                                                 ArtifactScalarType::None, ArtifactScalarType::None,
                                                 ArtifactScalarType::None, 32, 34, 2, 0, 0, 0};
    case ArtifactPhysicalEncoding::GroupedAffineU2_256:
        return format == ArtifactPhysicalFormat{2, ArtifactPhysicalEncoding::GroupedAffineU2_256,
                                                 ArtifactScalarType::U32, ArtifactScalarType::F16,
                                                 ArtifactScalarType::None, ArtifactScalarType::None,
                                                 ArtifactScalarType::None, 256, 64, 2, 0, 0, 0,
                                                 ArtifactScalarType::F16, 2};
    case ArtifactPhysicalEncoding::ColumnGroupedAffineU2Skip256:
        return format == ArtifactPhysicalFormat{
            1, ArtifactPhysicalEncoding::ColumnGroupedAffineU2Skip256,
            ArtifactScalarType::U8, ArtifactScalarType::F16,
            ArtifactScalarType::None, ArtifactScalarType::None,
            ArtifactScalarType::None, 256, 64, 2, 0, 0, 0,
            ArtifactScalarType::F16, 2};
    default:
        return false;
    }
}

bool semantic_storage_matches(ScalarType semantic, ArtifactScalarType physical,
                               ArtifactPhysicalEncoding encoding) {
    if ((encoding == ArtifactPhysicalEncoding::Q4_0 || encoding == ArtifactPhysicalEncoding::Q4_K ||
         encoding == ArtifactPhysicalEncoding::Q5_0 ||
         encoding == ArtifactPhysicalEncoding::Q6_K || encoding == ArtifactPhysicalEncoding::Q8_0) &&
        semantic == ScalarType::U8) {
        return physical == ArtifactScalarType::Packed;
    }
    if (encoding == ArtifactPhysicalEncoding::GroupedAffineU2_256 &&
        semantic == ScalarType::U32) {
        return physical == ArtifactScalarType::U32;
    }
    if (encoding == ArtifactPhysicalEncoding::ColumnGroupedAffineU2Skip256 &&
        semantic == ScalarType::U8) {
        return physical == ArtifactScalarType::U8;
    }
    switch (semantic) {
    case ScalarType::F32: return physical == ArtifactScalarType::F32;
    case ScalarType::F16: return physical == ArtifactScalarType::F16;
    case ScalarType::U8: return physical == ArtifactScalarType::U8;
    default: return false;
    }
}

bool physical_axis_contract_matches(const SemanticTensor& semantic,
                                    const ArtifactTensorRecord& physical) {
    const size_t rank = semantic.dimensions.size();

    if (rank == 0 ||
        rank != physical.logical_dimensions.size() ||
        rank > semantic.layout.axis_order.size() ||
        semantic.layout.rank != rank ||
        physical.layout.rank != rank ||
        semantic.layout.version != physical.layout.version ||
        semantic.layout.kind != physical.layout.kind ||
        semantic.layout.packing != physical.layout.packing ||
        semantic.layout.block_rank != physical.layout.block_rank ||
        semantic.layout.strides != physical.layout.strides ||
        semantic.layout.block_elements != physical.layout.block_elements ||
        semantic.layout.block_bytes != physical.layout.block_bytes ||
        semantic.layout.flags != physical.layout.flags) {
        return false;
    }

    std::array<bool, 8> semantic_axes{};
    std::array<bool, 8> physical_axes{};
    for (size_t physical_position = 0; physical_position != rank;
         ++physical_position) {
        const uint8_t semantic_axis =
            semantic.layout.axis_order[physical_position];
        const uint8_t physical_axis =
            physical.layout.axis_order[physical_position];
        if (semantic_axis >= rank || physical_axis >= rank ||
            semantic_axes[semantic_axis] || physical_axes[physical_axis] ||
            semantic.dimensions[semantic_axis].kind != DimensionKind::Constant ||
            semantic.dimensions[semantic_axis].constant_or_symbol == 0 ||
            semantic.dimensions[semantic_axis].constant_or_symbol !=
                physical.logical_dimensions[physical_axis]) {
            return false;
        }
        semantic_axes[semantic_axis] = true;
        physical_axes[physical_axis] = true;
    }
    for (size_t axis = 0; axis != rank; ++axis) {
        if (!semantic_axes[axis] || !physical_axes[axis]) return false;
    }

    // A transposed semantic view is safe only when the artifact declares the
    // source axes that its physical strides address. Without that declaration,
    // require the semantic and physical axis maps to be identical.
    if (physical.axis.source_rank == 0) {
        return semantic.layout.axis_order == physical.layout.axis_order;
    }
    if (physical.axis.source_rank != rank) return false;
    for (size_t physical_position = 0; physical_position != rank;
         ++physical_position) {
        if (physical.axis.source_axis_order[physical_position] !=
            physical.layout.axis_order[physical_position]) {
            return false;
        }
    }
    return true;
}

bool logical_type_matches(ScalarType semantic, ArtifactScalarType physical) {
    switch (semantic) {
    case ScalarType::F32: return physical == ArtifactScalarType::F32;
    case ScalarType::F16: return physical == ArtifactScalarType::F16;
    case ScalarType::U8: return physical == ArtifactScalarType::U8 || physical == ArtifactScalarType::Packed;
    case ScalarType::U32: return physical == ArtifactScalarType::U32;
    case ScalarType::I32: return physical == ArtifactScalarType::I32;
    default: return false;
    }
}

struct SpanKey {
    uint32_t artifact = UINT32_MAX;
    uint64_t offset = 0;
    uint64_t length = 0;
    friend bool operator==(const SpanKey& left, const SpanKey& right) {
        return left.artifact == right.artifact && left.offset == right.offset && left.length == right.length;
    }
};

SpanKey span_key(const ArtifactSourceSpan& span) {
    return {span.artifact_id.value, span.offset, span.length};
}

SpanKey span_key(const TensorPlane& plane) {
    return {plane.artifact_id.value, plane.offset, plane.length};
}

SpanKey span_key(const ArtifactTensorPlane& plane) {
    return span_key(plane.source);
}

const ArtifactTensorRecord* physical_tensor_by_id(std::span<const ArtifactTensorRecord> tensors, uint32_t id) {
    const auto found = std::lower_bound(tensors.begin(), tensors.end(), id,
                                        [](const ArtifactTensorRecord& tensor, uint32_t wanted) {
                                            return tensor.id < wanted;
                                        });
    return found == tensors.end() || found->id != id ? nullptr : &*found;
}

const SemanticTensor* semantic_tensor_by_id(const SemanticModel& model, uint32_t id) {
    return id < model.tensors.size() && model.tensors[id].id == id ? &model.tensors[id] : nullptr;
}

bool semantic_planes_are_canonical(const SemanticModel& model) {
    for (const SemanticTensor& tensor : model.tensors) {
        for (size_t index = 1; index != tensor.planes.size(); ++index) {
            if (static_cast<uint16_t>(tensor.planes[index - 1].kind) >=
                static_cast<uint16_t>(tensor.planes[index].kind)) {
                return false;
            }
        }
    }
    return true;
}

const ArtifactTensorPlane* physical_plane_by_kind(const ArtifactTensorRecord& tensor, PlaneKind kind) {
    const auto found = std::find_if(tensor.planes.begin(), tensor.planes.end(), [&](const auto& plane) {
        return plane.kind == kind;
    });
    return found == tensor.planes.end() ? nullptr : &*found;
}

const TensorPlane* semantic_plane_by_kind(const SemanticTensor& tensor, PlaneKind kind) {
    const auto found = std::find_if(tensor.planes.begin(), tensor.planes.end(), [&](const auto& plane) {
        return plane.kind == kind;
    });
    return found == tensor.planes.end() ? nullptr : &*found;
}

bool dimensions_element_count(const SemanticTensor& tensor, uint64_t& count) {
    count = 1;
    for (const Dimension& dimension : tensor.dimensions) {
        if (dimension.kind != DimensionKind::Constant ||
            !checked_multiply(count, dimension.constant_or_symbol, count)) return false;
    }
    return true;
}

bool tensor_binding_matches(const SemanticTensor& semantic, const ArtifactTensorRecord& physical) {
    if (!supported_format(physical) || !physical_axis_contract_matches(semantic, physical) ||
        !logical_type_matches(semantic.logical_type, physical.logical_type) ||
        semantic.quantization != physical.quantization ||
        semantic.planes.size() != physical.planes.size() ||
        (semantic.expert_axis.kind == ExpertAxisKind::None) !=
            (physical.coordinate.bank_axis == UINT8_MAX)) {
        return false;
    }
    if (semantic.expert_axis.kind == ExpertAxisKind::ExpertBank &&
        (physical.coordinate.bank_axis != semantic.expert_axis.expert_axis ||
         physical.coordinate.bank_extent != semantic.expert_axis.expert_count ||
         physical.coordinate.bank_stride != semantic.expert_axis.per_expert_byte_stride)) {
        return false;
    }
    uint64_t element_count = 0;
    if (!dimensions_element_count(semantic, element_count)) return false;
    for (const TensorPlane& source : semantic.planes) {
        if ((source.flags & ~1u) != 0) return false;
        const ArtifactTensorPlane* target = physical_plane_by_kind(physical, source.kind);
        if (!target || span_key(source) != span_key(*target) || source.alignment != target->alignment ||
            source.length == 0 || !semantic_storage_matches(source.storage_type, target->storage_type,
                                                             physical.format.encoding)) {
            return false;
        }
        if (source.kind == PlaneKind::Values &&
            (target->logical_elements != element_count || target->source.length != source.length)) return false;
        if (target->bytes_per_block == 0 || target->elements_per_block == 0) return false;
    }
    const ArtifactTensorPlane* values = physical_plane_by_kind(physical, PlaneKind::Values);
    return values != nullptr && semantic_plane_by_kind(semantic, PlaneKind::Values) != nullptr;
}

bool aliases_match_semantic_roles(const ArtifactIndex& physical, const SemanticModel& model) {
    for (const ArtifactAlias& alias : physical.aliases()) {
        const ArtifactTensorRecord* source = physical_tensor_by_id(physical.tensors(), alias.source_tensor_id);
        const ArtifactTensorRecord* target = physical_tensor_by_id(physical.tensors(), alias.target_tensor_id);
        const SemanticTensor* semantic_source = semantic_tensor_by_id(model, alias.source_tensor_id);
        const SemanticTensor* semantic_target = semantic_tensor_by_id(model, alias.target_tensor_id);
        if (!source || !target || !semantic_source || !semantic_target ||
            semantic_target->role != alias.semantic_role) {
            return false;
        }
        // A directional alias is intentionally checked in its declared
        // source-to-target orientation. Do not admit the symmetric relation
        // merely because the physical spans are interchangeable.
        if (alias.direction != ArtifactAliasDirection::Bidirectional &&
            alias.direction != ArtifactAliasDirection::SourceToTarget) {
            return false;
        }
    }
    return true;
}

bool physical_bindings_match(const ArtifactIndex& physical, const SemanticModel& model) {
    if (physical.artifacts().empty() || physical.tensors().size() != model.tensors.size() ||
        !semantic_planes_are_canonical(model) || !aliases_match_semantic_roles(physical, model)) return false;

    // ArtifactIndex has already validated shared spans against its aliases.
    // Tensor IDs, rather than physical spans, are the binding identity here.
    std::unordered_set<uint32_t> used;
    used.reserve(model.tensors.size());
    for (const SemanticTensor& tensor : model.tensors) {
        const ArtifactTensorRecord* found = physical_tensor_by_id(physical.tensors(), tensor.id);
        if (!found || !used.insert(tensor.id).second || !tensor_binding_matches(tensor, *found)) return false;
    }
    return used.size() == physical.tensors().size();
}

SemanticModel payload_model(const SemanticModel& model) {
    SemanticModel payload = model;
    // LAPIR001 predates package closures and only permits the local artifact
    // coordinate in its tensor planes.  Keep the actual closure coordinate in
    // the manifest's binding set and normalize only the transport payload.
    for (SemanticTensor& tensor : payload.tensors) {
        for (TensorPlane& plane : tensor.planes) plane.artifact_id = ArtifactId{0};
    }
    return payload;
}

bool bind_model_to_index(const ArtifactIndex& physical, SemanticModel& model) {
    for (SemanticTensor& semantic : model.tensors) {
        const ArtifactTensorRecord* found = physical_tensor_by_id(physical.tensors(), semantic.id);
        if (!found) return false;
        for (TensorPlane& source : semantic.planes) {
            const ArtifactTensorPlane* target = physical_plane_by_kind(*found, source.kind);
            if (!target) return false;
            source.artifact_id = target->source.artifact_id;
            source.offset = target->source.offset;
            source.length = target->source.length;
        }
    }
    return true;
}

bool transport_bindings_match(const ArtifactIndex& physical, const SemanticModel& model) {
    for (const SemanticTensor& semantic : model.tensors) {
        const ArtifactTensorRecord* found = physical_tensor_by_id(physical.tensors(), semantic.id);
        if (!found) return false;
        for (const TensorPlane& source : semantic.planes) {
            const ArtifactTensorPlane* target = physical_plane_by_kind(*found, source.kind);
            if (!target || source.artifact_id.value != 0 || source.offset != target->source.offset ||
                source.length != target->source.length || source.alignment != target->alignment ||
                (source.flags & ~1u) != 0) {
                return false;
            }
        }
    }
    return true;
}

bool model_contract_supported(const SemanticModel& model) {
    if (model.schema_minor != 0 || model.opset_minor != 0 ||
        model.schema_major != model.opset_major || model.schema_major < 1 ||
        model.schema_major > 7 || model.entry_kind != EntryKind::TokenIds) {
        return false;
    }
    return model.schema_major == 1
        ? model.maximum_context == 32768
        : model.maximum_context > 0 && model.maximum_context <= 262144;
}

struct ArtifactIdentity {
    ArtifactId id;
    ArtifactRole role = ArtifactRole::Primary;
    uint64_t size = 0;
    Sha256Digest digest;
};

bool index_artifacts_match(const ArtifactIndex& physical,
                           std::span<const ArtifactIdentity> expected) {
    if (physical.artifacts().size() != expected.size()) return false;
    for (size_t index = 0; index != expected.size(); ++index) {
        const PackageView& actual = physical.artifacts()[index];
        const ArtifactIdentity& wanted = expected[index];
        if (actual.artifact_id() != wanted.id || actual.role() != wanted.role ||
            actual.bytes().size() != wanted.size || actual.digest() != wanted.digest) {
            return false;
        }
        if (index != 0 && actual.artifact_id().value <= physical.artifacts()[index - 1].artifact_id().value) {
            return false;
        }
    }
    return true;
}

SemanticEncodeResult canonical_graph_bytes(const SemanticModel& model) {
    // encode_semantic_model is also the canonical graph encoder.  Give it a
    // deliberately inert physical projection and neutral interaction values;
    // those fields remain in the payload for binding validation, but cannot
    // influence the graph identity.
    SemanticModel graph = model;
    graph.vocabulary_size = 1;
    graph.bos_id = 0;
    graph.eos_id = 0;
    graph.stop_ids.clear();
    graph.tokenizer_digest = {};
    graph.template_digest = {};
    for (SemanticTensor& tensor : graph.tensors) {
        const uint8_t rank = static_cast<uint8_t>(tensor.dimensions.size());
        uint64_t element_count = 1;
        for (const Dimension& dimension : tensor.dimensions) {
            if (dimension.kind == DimensionKind::Constant &&
                !checked_multiply(element_count, dimension.constant_or_symbol, element_count)) {
                return manifest_error(CompatibilityError::IR_CONSTRAINT_FAILED,
                                      "SemanticManifest graph canonicalization has invalid dimensions");
            }
        }
        for (uint8_t axis = 0; axis < rank; ++axis) tensor.layout.strides[axis] = 1;
        if (tensor.layout.kind == PhysicalLayoutKind::GgufBlocked) tensor.layout.block_bytes = 1;
        if (tensor.quantization.kind != QuantizationKind::None) tensor.quantization.block_bytes = 1;
        if (tensor.expert_axis.kind == ExpertAxisKind::ExpertBank) {
            tensor.expert_axis.per_expert_byte_stride = 1;
        }
        for (TensorPlane& plane : tensor.planes) {
            plane.artifact_id = ArtifactId{0};
            plane.offset = 0;
            plane.length = tensor.expert_axis.kind == ExpertAxisKind::ExpertBank &&
                                   plane.kind == PlaneKind::Values
                               ? tensor.expert_axis.expert_count
                               : 1;
            plane.alignment = 1;
            plane.flags = 0;
        }
        std::sort(tensor.planes.begin(), tensor.planes.end(), [](const TensorPlane& left, const TensorPlane& right) {
            return static_cast<uint16_t>(left.kind) < static_cast<uint16_t>(right.kind);
        });
    }
    auto encoded = encode_semantic_model(graph);
    return encoded;
}

std::vector<uint8_t> token_contract_bytes(const TokenContract& contract) {
    return token_contract_canonical_bytes(contract);
}

std::vector<uint8_t> physical_binding_bytes(const ArtifactIndex& physical,
                                             const SemanticModel& model) {
    std::vector<uint8_t> bytes;
    constexpr char domain[] = "laplace-physical-binding-set-v1";
    bytes.insert(bytes.end(), domain, domain + sizeof(domain));
    append_digest(bytes, physical.normalized_digest());
    append_u32(bytes, static_cast<uint32_t>(model.tensors.size()));
    std::vector<const SemanticTensor*> tensors;
    tensors.reserve(model.tensors.size());
    for (const SemanticTensor& tensor : model.tensors) tensors.push_back(&tensor);
    std::sort(tensors.begin(), tensors.end(), [](const SemanticTensor* left, const SemanticTensor* right) {
        return left->id < right->id;
    });
    for (const SemanticTensor* tensor : tensors) {
        append_u32(bytes, tensor->id);
        std::vector<const TensorPlane*> planes;
        planes.reserve(tensor->planes.size());
        for (const TensorPlane& plane : tensor->planes) planes.push_back(&plane);
        std::sort(planes.begin(), planes.end(), [](const TensorPlane* left, const TensorPlane* right) {
            return static_cast<uint16_t>(left->kind) < static_cast<uint16_t>(right->kind);
        });
        append_u32(bytes, static_cast<uint32_t>(planes.size()));
        for (const TensorPlane* plane : planes) {
            append_u16(bytes, static_cast<uint16_t>(plane->kind));
            append_u16(bytes, static_cast<uint16_t>(plane->storage_type));
            append_u32(bytes, plane->artifact_id.value);
            append_u64(bytes, plane->offset);
            append_u64(bytes, plane->length);
            append_u32(bytes, plane->alignment);
            append_u32(bytes, plane->flags);
        }
    }
    return bytes;
}

std::vector<uint8_t> artifact_closure_bytes(const ArtifactIndex& physical) {
    std::vector<uint8_t> bytes;
    constexpr char domain[] = "laplace-artifact-closure-v1";
    bytes.insert(bytes.end(), domain, domain + sizeof(domain));
    std::vector<const PackageView*> artifacts;
    artifacts.reserve(physical.artifacts().size());
    for (const PackageView& artifact : physical.artifacts()) artifacts.push_back(&artifact);
    std::sort(artifacts.begin(), artifacts.end(), [](const PackageView* left, const PackageView* right) {
        return left->artifact_id().value < right->artifact_id().value;
    });
    append_u32(bytes, static_cast<uint32_t>(artifacts.size()));
    for (const PackageView* artifact : artifacts) {
        append_u32(bytes, artifact->artifact_id().value);
        append_u16(bytes, static_cast<uint16_t>(artifact->role()));
        append_u16(bytes, 0);
        append_u64(bytes, artifact->bytes().size());
        append_digest(bytes, artifact->digest());
    }
    return bytes;
}

SemanticGraphDigestResult compute_graph_digest(const SemanticModel& model) {
    auto encoded = canonical_graph_bytes(model);
    if (const auto* report = std::get_if<CompatibilityReport>(&encoded)) return *report;
    const auto& bytes = std::get<std::vector<uint8_t>>(encoded);
    return digest_bytes(bytes);
}

Sha256Digest compute_binding_digest(const ArtifactIndex& physical, const SemanticModel& model) {
    const std::vector<uint8_t> bytes = physical_binding_bytes(physical, model);
    return digest_bytes(bytes);
}

Sha256Digest compute_interaction_digest(const TokenContract& contract) {
    std::vector<uint8_t> bytes;
    constexpr char domain[] = "laplace-interaction-contract-v1";
    bytes.insert(bytes.end(), domain, domain + sizeof(domain));
    const std::vector<uint8_t> token = token_contract_bytes(contract);
    bytes.insert(bytes.end(), token.begin(), token.end());
    return digest_bytes(bytes);
}

Sha256Digest compute_package_digest(const ArtifactIndex& physical, const Sha256Digest& graph,
                                    const SemanticModel& model, const TokenContract& contract) {
    const Sha256Digest binding = compute_binding_digest(physical, model);
    const Sha256Digest interaction = compute_interaction_digest(contract);
    const std::vector<uint8_t> closure = artifact_closure_bytes(physical);
    const Sha256Digest closure_digest = digest_bytes(closure);
    std::vector<uint8_t> bytes;
    constexpr char domain[] = "laplace-package-fingerprint-v1";
    bytes.insert(bytes.end(), domain, domain + sizeof(domain));
    append_digest(bytes, graph);
    append_digest(bytes, binding);
    append_digest(bytes, interaction);
    append_digest(bytes, closure_digest);
    return digest_bytes(bytes);
}

std::vector<uint8_t> make_body(const ArtifactIndex& physical, const TokenContract& contract,
                               const Sha256Digest& graph_digest_value,
                               const Sha256Digest& binding_digest_value,
                               const Sha256Digest& interaction_digest_value,
                               const Sha256Digest& package_digest_value,
                               const std::vector<uint8_t>& semantic_bytes) {
    std::vector<uint8_t> body;
    append_u32(body, kBodyVersion);
    append_u32(body, static_cast<uint32_t>(physical.artifacts().size()));
    for (const PackageView& artifact : physical.artifacts()) {
        append_u32(body, artifact.artifact_id().value);
        append_u32(body, static_cast<uint32_t>(artifact.role()));
        append_u64(body, artifact.bytes().size());
        append_digest(body, artifact.digest());
    }
    append_digest(body, graph_digest_value);
    append_digest(body, binding_digest_value);
    append_digest(body, interaction_digest_value);
    append_digest(body, package_digest_value);
    append_u16(body, static_cast<uint16_t>(EntryKind::TokenIds));
    append_u16(body, 0);
    append_u32(body, contract.vocabulary_size);
    append_u32(body, contract.bos_id);
    append_u32(body, contract.eos_id);
    append_u32(body, static_cast<uint32_t>(contract.stop_ids.size()));
    for (uint32_t id : contract.stop_ids) append_u32(body, id);
    append_digest(body, contract.authoritative_tokenizer_digest);
    append_digest(body, contract.authoritative_template_digest);
    append_digest(body, semantic_manifest_token_contract_digest(contract));
    append_u64(body, semantic_bytes.size());
    body.insert(body.end(), semantic_bytes.begin(), semantic_bytes.end());
    const std::vector<uint8_t> token_authority = token_contract_canonical_bytes(contract);
    append_u64(body, token_authority.size());
    body.insert(body.end(), token_authority.begin(), token_authority.end());
    return body;
}

std::shared_ptr<const SemanticManifest::Data> make_data(const ArtifactIndex& physical,
                                                         const SemanticModel& model,
                                                         const TokenContract& contract,
                                                         std::vector<uint8_t> semantic_bytes,
                                                         std::vector<uint8_t> envelope_bytes,
                                                         const Sha256Digest& graph,
                                                         const Sha256Digest& binding,
                                                         const Sha256Digest& interaction,
                                                         const Sha256Digest& package,
                                                         std::optional<SourceCompilerGraphProof> graph_proof) {
    auto data = std::make_shared<SemanticManifest::Data>();
    data->physical = physical;
    data->model = model;
    data->contract = contract;
    data->graph_proof = std::move(graph_proof);
    data->bytes = std::move(envelope_bytes);
    data->semantic_bytes = std::move(semantic_bytes);
    data->semantic_graph_digest = graph;
    data->physical_binding_set_digest = binding;
    data->interaction_contract_digest = interaction;
    data->package_fingerprint = package;
    data->body_digest = digest_bytes(std::span<const uint8_t>(data->bytes).subspan(kHeaderBytes,
                                                                                   data->bytes.size() - kHeaderBytes - kEnvelopeDigestBytes));
    data->record_digest = digest_bytes(std::span<const uint8_t>(data->bytes).first(data->bytes.size() - kEnvelopeDigestBytes));
    const auto primary = std::find_if(data->physical.artifacts().begin(), data->physical.artifacts().end(),
                                      [](const PackageView& artifact) {
                                          return artifact.role() == ArtifactRole::Primary;
                                      });
    if (primary == data->physical.artifacts().end()) return data;
    const PackageView& artifact = *primary;
    data->artifact_id = artifact.artifact_id();
    data->artifact_size = artifact.bytes().size();
    data->artifact_digest = artifact.digest();
    return data;
}

} // namespace

Sha256Digest semantic_manifest_token_contract_digest(const TokenContract& contract) {
    const std::vector<uint8_t> bytes = token_contract_bytes(contract);
    return digest_bytes(bytes);
}

SemanticGraphDigestResult semantic_manifest_graph_digest(const SemanticModel& model) {
    return compute_graph_digest(model);
}

SemanticManifestResult SemanticManifest::build(const ArtifactIndex& physical,
                                               const SemanticModel& model,
                                               const TokenContract& contract) {
    try {
    if (physical.artifacts().empty() || physical.artifacts().size() > kMaxArtifacts) {
        return manifest_error(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                              "SemanticManifest artifact closure is empty or exceeds its bound");
    }
    if (!model_contract_supported(model) || !valid_token_contract(contract) ||
        !token_fields_match_model(contract, model)) {
        return manifest_error(CompatibilityError::IR_VERSION_UNSUPPORTED,
                              "SemanticManifest requires a supported complete TokenIds contract");
    }
    if (!token_contract_package_bound(physical, contract)) {
        return manifest_error(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                              "tokenizer descriptor is not bound to the complete package closure");
    }
    if (!physical_bindings_match(physical, model)) {
        return manifest_error(CompatibilityError::IR_LAYOUT_MISMATCH,
                              "semantic tensor planes do not exactly bind the physical ArtifactIndex");
    }
    const SemanticModel transport_model = payload_model(model);
    auto encoded = encode_semantic_model(transport_model);
    if (const auto* report = std::get_if<CompatibilityReport>(&encoded)) return *report;
    const std::vector<uint8_t> semantic_bytes = std::get<std::vector<uint8_t>>(std::move(encoded));
    auto graph_result = compute_graph_digest(model);
    if (const auto* report = std::get_if<CompatibilityReport>(&graph_result)) return *report;
    const Sha256Digest graph = std::get<Sha256Digest>(graph_result);
    std::optional<SourceCompilerGraphProof> graph_proof;
    if (!model.operators.empty()) {
        auto proof_result = prove_source_candidate_graph(model);
        if (const auto* report = std::get_if<CompatibilityReport>(&proof_result)) return *report;
        graph_proof = std::get<SourceCompilerGraphProof>(std::move(proof_result));
    }
    const Sha256Digest binding = compute_binding_digest(physical, model);
    const Sha256Digest interaction = compute_interaction_digest(contract);
    const Sha256Digest package = compute_package_digest(physical, graph, model, contract);
    const std::vector<uint8_t> body = make_body(physical, contract, graph, binding, interaction, package,
                                                semantic_bytes);
    if (body.size() > kMaxBodyBytes || body.size() > std::numeric_limits<uint64_t>::max() - kHeaderBytes - kEnvelopeDigestBytes) {
        return manifest_error(CompatibilityError::IR_CONSTRAINT_FAILED, "SemanticManifest body exceeds its bound");
    }
    std::vector<uint8_t> envelope;
    envelope.reserve(kHeaderBytes + body.size() + kEnvelopeDigestBytes);
    envelope.insert(envelope.end(), kMagic.begin(), kMagic.end());
    append_u16(envelope, kMajor);
    append_u16(envelope, kMinor);
    append_u32(envelope, kHeaderBytes);
    append_u64(envelope, body.size());
    append_u64(envelope, kHeaderBytes + body.size() + kEnvelopeDigestBytes);
    append_digest(envelope, digest_bytes(body));
    envelope.insert(envelope.end(), body.begin(), body.end());
    append_digest(envelope, digest_bytes(envelope));
    return SemanticManifest(make_data(physical, model, contract, semantic_bytes, std::move(envelope),
                                      graph, binding, interaction, package, std::move(graph_proof)));
    } catch (const std::bad_alloc&) {
        return manifest_error(CompatibilityError::IR_CONSTRAINT_FAILED, "SemanticManifest allocation exceeds its resource budget");
    }
}

SemanticManifestResult SemanticManifest::decode(const ArtifactIndex& physical,
                                                std::span<const uint8_t> bytes) {
    try {
    if (bytes.size() < kHeaderBytes + kEnvelopeDigestBytes || bytes.size() > kMaxBodyBytes + kHeaderBytes + kEnvelopeDigestBytes ||
        !std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
        return manifest_error(CompatibilityError::IR_VERSION_UNSUPPORTED, "SemanticManifest header is invalid");
    }
    Reader header(bytes);
    std::span<const uint8_t> magic;
    uint16_t major = 0, minor = 0;
    uint32_t header_bytes = 0;
    uint64_t body_length = 0, total_length = 0;
    if (!header.take(8, magic) || !header.u16(major) || !header.u16(minor) || !header.u32(header_bytes) ||
        !header.u64(body_length) || !header.u64(total_length) || header_bytes != kHeaderBytes ||
        major != kMajor || minor != kMinor || total_length != bytes.size() ||
        body_length > kMaxBodyBytes || body_length != bytes.size() - kHeaderBytes - kEnvelopeDigestBytes) {
        return manifest_error(CompatibilityError::IR_VERSION_UNSUPPORTED, "SemanticManifest version or length is invalid");
    }
    std::span<const uint8_t> body_digest_bytes;
    if (!header.take(32, body_digest_bytes) ||
        !std::equal(body_digest_bytes.begin(), body_digest_bytes.end(),
                    digest_bytes(bytes.subspan(kHeaderBytes, static_cast<size_t>(body_length))).bytes.begin())) {
        return manifest_error(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH, "SemanticManifest body digest is invalid");
    }
    const Sha256Digest record_digest = digest_bytes(bytes.first(bytes.size() - kEnvelopeDigestBytes));
    if (!std::equal(bytes.end() - kEnvelopeDigestBytes, bytes.end(), record_digest.bytes.begin())) {
        return manifest_error(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH, "SemanticManifest record digest is invalid");
    }

    Reader body(bytes.subspan(kHeaderBytes, static_cast<size_t>(body_length)));
    uint32_t body_version = 0, artifact_count = 0;
    if (!body.u32(body_version) || !body.u32(artifact_count) || body_version != kBodyVersion ||
        artifact_count == 0 || artifact_count > kMaxArtifacts) {
        return manifest_error(CompatibilityError::IR_VERSION_UNSUPPORTED, "SemanticManifest body header is invalid");
    }
    std::vector<ArtifactIdentity> artifact_identities;
    artifact_identities.reserve(artifact_count);
    for (uint32_t index = 0; index != artifact_count; ++index) {
        uint32_t artifact_id_raw = UINT32_MAX, role_raw = 0;
        uint64_t artifact_size = 0;
        std::span<const uint8_t> artifact_digest_span;
        if (!body.u32(artifact_id_raw) || !body.u32(role_raw) || !body.u64(artifact_size) ||
            !body.take(32, artifact_digest_span) || artifact_id_raw == UINT32_MAX || artifact_size == 0 ||
            role_raw < static_cast<uint32_t>(ArtifactRole::Primary) ||
            role_raw > static_cast<uint32_t>(ArtifactRole::Sidecar) ||
            (index != 0 && artifact_identities.back().id.value >= artifact_id_raw)) {
            return manifest_error(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                                  "SemanticManifest artifact closure is invalid or not canonically ordered");
        }
        ArtifactIdentity identity;
        identity.id = ArtifactId{artifact_id_raw};
        identity.role = static_cast<ArtifactRole>(role_raw);
        identity.size = artifact_size;
        std::copy(artifact_digest_span.begin(), artifact_digest_span.end(), identity.digest.bytes.begin());
        artifact_identities.push_back(identity);
    }
    if (!index_artifacts_match(physical, artifact_identities)) {
        return manifest_error(CompatibilityError::PACKAGE_SOURCE_CHANGED,
                              "manifest artifact closure does not match ArtifactIndex");
    }
    Sha256Digest encoded_graph_digest;
    Sha256Digest encoded_binding_digest;
    Sha256Digest encoded_interaction_digest;
    Sha256Digest encoded_package_digest;
    std::span<const uint8_t> graph_digest_span, binding_digest_span, interaction_digest_span, package_digest_span;
    if (!body.take(32, graph_digest_span) || !body.take(32, binding_digest_span) ||
        !body.take(32, interaction_digest_span) || !body.take(32, package_digest_span)) {
        return manifest_error(CompatibilityError::PACKAGE_BOUNDS_INVALID, "manifest identity digests are truncated");
    }
    std::copy(graph_digest_span.begin(), graph_digest_span.end(), encoded_graph_digest.bytes.begin());
    std::copy(binding_digest_span.begin(), binding_digest_span.end(), encoded_binding_digest.bytes.begin());
    std::copy(interaction_digest_span.begin(), interaction_digest_span.end(), encoded_interaction_digest.bytes.begin());
    std::copy(package_digest_span.begin(), package_digest_span.end(), encoded_package_digest.bytes.begin());
    uint16_t entry_raw = 0, token_reserved = 0;
    TokenContract projection;
    uint32_t stop_count = 0;
    if (!body.u16(entry_raw) || !body.u16(token_reserved) || !body.u32(projection.vocabulary_size) ||
        !body.u32(projection.bos_id) || !body.u32(projection.eos_id) || !body.u32(stop_count) ||
        token_reserved != 0 || entry_raw != static_cast<uint16_t>(EntryKind::TokenIds) || stop_count > kMaxStops) {
        return manifest_error(CompatibilityError::IR_VERSION_UNSUPPORTED, "manifest token contract header is invalid");
    }
    uint64_t stop_bytes = 0;
    if (!checked_multiply(stop_count, sizeof(uint32_t), stop_bytes) || !fits_size(stop_bytes) || body.remaining() < stop_bytes) {
        return manifest_error(CompatibilityError::PACKAGE_BOUNDS_INVALID, "manifest stop-ID array is out of bounds");
    }
    projection.stop_ids.resize(stop_count);
    for (uint32_t& id : projection.stop_ids) if (!body.u32(id)) return manifest_error(CompatibilityError::PACKAGE_BOUNDS_INVALID, "manifest stop-ID array is truncated");
    std::span<const uint8_t> tokenizer_digest_span, template_digest_span, token_digest_span;
    if (!body.take(32, tokenizer_digest_span) || !body.take(32, template_digest_span) || !body.take(32, token_digest_span)) {
        return manifest_error(CompatibilityError::PACKAGE_BOUNDS_INVALID, "manifest token digests are truncated");
    }
    std::copy(tokenizer_digest_span.begin(), tokenizer_digest_span.end(), projection.authoritative_tokenizer_digest.bytes.begin());
    std::copy(template_digest_span.begin(), template_digest_span.end(), projection.authoritative_template_digest.bytes.begin());
    Sha256Digest expected_token_digest;
    std::copy(token_digest_span.begin(), token_digest_span.end(), expected_token_digest.bytes.begin());
    uint64_t semantic_length = 0;
    if (!body.u64(semantic_length) || semantic_length > body.remaining() || !fits_size(semantic_length)) {
        return manifest_error(CompatibilityError::PACKAGE_BOUNDS_INVALID, "manifest semantic bytes are out of bounds");
    }
    std::span<const uint8_t> semantic_span;
    if (!body.take(static_cast<size_t>(semantic_length), semantic_span)) {
        return manifest_error(CompatibilityError::PACKAGE_BOUNDS_INVALID, "manifest semantic bytes are truncated");
    }
    uint64_t token_authority_length = 0;
    if (!body.u64(token_authority_length) || token_authority_length == 0 ||
        token_authority_length > body.remaining() || !fits_size(token_authority_length)) {
        return manifest_error(CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED,
                              "manifest token authority descriptor is missing or out of bounds");
    }
    std::span<const uint8_t> token_authority_span;
    if (!body.take(static_cast<size_t>(token_authority_length), token_authority_span) || body.remaining() != 0) {
        return manifest_error(CompatibilityError::IR_VERSION_UNSUPPORTED,
                              "manifest has unknown fields or trailing bytes");
    }
    auto decoded_contract = TokenContract::deserialize(token_authority_span);
    if (const auto* status = std::get_if<TokenContractStatus>(&decoded_contract)) {
        return manifest_error(CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED,
                              std::string("manifest token authority is invalid: ") +
                                  std::string(token_contract_error_name(status->error)));
    }
    TokenContract contract = std::get<TokenContract>(std::move(decoded_contract));
    if (projection.vocabulary_size != contract.vocabulary_size ||
        projection.bos_id != contract.bos_id || projection.eos_id != contract.eos_id ||
        projection.stop_ids != contract.stop_ids ||
        projection.authoritative_tokenizer_digest != contract.authoritative_tokenizer_digest ||
        projection.authoritative_template_digest != contract.authoritative_template_digest ||
        expected_token_digest != semantic_manifest_token_contract_digest(contract) ||
        !valid_token_contract(contract) || !token_contract_package_bound(physical, contract)) {
        return manifest_error(CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED,
                              "manifest token contract authority, digest, or package binding is invalid");
    }
    std::vector<uint8_t> semantic_bytes(semantic_span.begin(), semantic_span.end());
    auto decoded = decode_semantic_model(semantic_bytes);
    if (const auto* report = std::get_if<CompatibilityReport>(&decoded)) return *report;
    SemanticModel model = std::get<SemanticModel>(decoded);
    if (!model_contract_supported(model) || !token_fields_match_model(contract, model) ||
        !transport_bindings_match(physical, model) || !bind_model_to_index(physical, model) ||
        !physical_bindings_match(physical, model)) {
        return manifest_error(CompatibilityError::IR_REFERENCE_INVALID,
                              "manifest semantic graph, token contract, or physical bindings are inconsistent");
    }
    const SemanticModel transport_model = payload_model(model);
    auto expected_encoded = encode_semantic_model(transport_model);
    if (const auto* report = std::get_if<CompatibilityReport>(&expected_encoded)) return *report;
    std::vector<uint8_t> expected_semantic = std::get<std::vector<uint8_t>>(std::move(expected_encoded));
    if (expected_semantic != semantic_bytes) {
        return manifest_error(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH, "manifest semantic bytes are not canonical");
    }
    auto graph_result = compute_graph_digest(model);
    if (const auto* report = std::get_if<CompatibilityReport>(&graph_result)) return *report;
    const Sha256Digest graph = std::get<Sha256Digest>(graph_result);
    std::optional<SourceCompilerGraphProof> graph_proof;
    if (!model.operators.empty()) {
        auto proof_result = prove_source_candidate_graph(model);
        if (const auto* report = std::get_if<CompatibilityReport>(&proof_result)) return *report;
        graph_proof = std::get<SourceCompilerGraphProof>(std::move(proof_result));
    }
    const Sha256Digest binding = compute_binding_digest(physical, model);
    const Sha256Digest interaction = compute_interaction_digest(contract);
    const Sha256Digest package = compute_package_digest(physical, graph, model, contract);
    if (encoded_graph_digest != graph || encoded_binding_digest != binding ||
        encoded_interaction_digest != interaction || encoded_package_digest != package) {
        return manifest_error(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH,
                              "manifest graph, binding, interaction, or package identity is invalid");
    }
    std::vector<uint8_t> envelope(bytes.begin(), bytes.end());
    return SemanticManifest(make_data(physical, model, contract, std::move(semantic_bytes), std::move(envelope),
                                      graph, binding, interaction, package, std::move(graph_proof)));
    } catch (const std::bad_alloc&) {
        return manifest_error(CompatibilityError::IR_CONSTRAINT_FAILED, "SemanticManifest allocation exceeds its resource budget");
    }
}

SemanticManifestResult SemanticManifest::decode_carried(const ArtifactIndex& physical,
                                                        const PackageView& carrier) {
    try {
        if (carrier.role() != ArtifactRole::Sidecar || carrier.artifact_id().value == UINT32_MAX) {
            return carrier_error(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                                 "carried semantic manifest must be a uniquely identified sidecar");
        }
        if (carrier.bytes().empty()) {
            return carrier_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                 "carried semantic manifest sidecar is empty");
        }
        if (carrier.bytes().size() > kMaxBodyBytes + kHeaderBytes + kEnvelopeDigestBytes) {
            return carrier_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                 "carried semantic manifest sidecar exceeds its bound");
        }
        if (carrier.bytes().size() < kHeaderBytes + kEnvelopeDigestBytes) {
            return carrier_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                 "carried semantic manifest sidecar is truncated");
        }
        if (!std::equal(kMagic.begin(), kMagic.end(), carrier.bytes().begin())) {
            return carrier_error(CompatibilityError::PACKAGE_BAD_MAGIC,
                                 "carried semantic manifest sidecar magic is invalid");
        }
        if (digest_bytes(carrier.bytes()) != carrier.digest()) {
            return carrier_error(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH,
                                 "carried semantic manifest sidecar digest is invalid");
        }
        for (const PackageView& artifact : physical.artifacts()) {
            if (artifact.artifact_id() == carrier.artifact_id()) {
                return carrier_error(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                                     "manifest sidecar collides with a package artifact ID");
            }
            if (spans_overlap(artifact.bytes(), carrier.bytes())) {
                return carrier_error(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                                     "manifest sidecar aliases a package artifact span");
            }
        }
        auto decoded = decode(physical, carrier.bytes());
        if (const auto* report = std::get_if<CompatibilityReport>(&decoded)) return *report;
        SemanticManifest manifest = std::get<SemanticManifest>(std::move(decoded));
        if (!manifest.graph_proof()) {
            return carrier_error(CompatibilityError::IMPORT_CLOSURE_INCOMPLETE,
                                 "authoritative manifest lacks a complete graph proof");
        }
        auto data = std::make_shared<Data>(*manifest.data_);
        data->carrier = carrier;
        return SemanticManifest(std::move(data));
    } catch (const std::bad_alloc&) {
        return manifest_error(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "carried semantic manifest allocation exceeds its resource budget");
    }
}

std::span<const uint8_t> SemanticManifest::bytes() const noexcept { return data_->bytes; }
std::span<const uint8_t> SemanticManifest::semantic_bytes() const noexcept { return data_->semantic_bytes; }
const SemanticModel& SemanticManifest::semantic_model() const noexcept { return data_->model; }
const TokenContract& SemanticManifest::token_contract() const noexcept { return data_->contract; }
const ArtifactIndex& SemanticManifest::physical_index() const noexcept { return data_->physical; }
ArtifactId SemanticManifest::artifact_id() const noexcept { return data_->artifact_id; }
uint64_t SemanticManifest::artifact_size() const noexcept { return data_->artifact_size; }
const Sha256Digest& SemanticManifest::artifact_digest() const noexcept { return data_->artifact_digest; }
const Sha256Digest& SemanticManifest::semantic_graph_digest() const noexcept { return data_->semantic_graph_digest; }
const Sha256Digest& SemanticManifest::physical_binding_set_digest() const noexcept {
    return data_->physical_binding_set_digest;
}
const Sha256Digest& SemanticManifest::interaction_contract_digest() const noexcept {
    return data_->interaction_contract_digest;
}
const Sha256Digest& SemanticManifest::package_fingerprint() const noexcept { return data_->package_fingerprint; }
const Sha256Digest& SemanticManifest::semantic_digest() const noexcept { return data_->semantic_graph_digest; }
const Sha256Digest& SemanticManifest::body_digest() const noexcept { return data_->body_digest; }
const Sha256Digest& SemanticManifest::record_digest() const noexcept { return data_->record_digest; }
const std::optional<SourceCompilerGraphProof>& SemanticManifest::graph_proof() const noexcept {
    return data_->graph_proof;
}
bool SemanticManifest::has_carrier() const noexcept { return data_ && data_->carrier.has_value(); }

SemanticManifestResult compile_semantic_manifest(const ArtifactIndex& physical,
                                                 const SemanticModel& model,
                                                 const TokenContract& token_contract) {
    return SemanticManifest::build(physical, model, token_contract);
}

SemanticManifestResult decode_semantic_manifest(const ArtifactIndex& physical,
                                                std::span<const uint8_t> bytes) {
    return SemanticManifest::decode(physical, bytes);
}

SemanticManifestResult decode_carried_semantic_manifest(const ArtifactIndex& physical,
                                                        const PackageView& carrier) {
    return SemanticManifest::decode_carried(physical, carrier);
}

} // namespace Laplace
