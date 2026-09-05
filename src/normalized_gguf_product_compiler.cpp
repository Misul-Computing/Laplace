#include "normalized_gguf_product_compiler.h"
#include "chat_framing.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <CommonCrypto/CommonDigest.h>

#include "gguf.h"
#include "gguf_fact_keys.h"
#include "gguf_index.h"
#include "gguf_normalized_source_adapter.h"
#include "codec_certificate.h"
#include "physical_codec.h"

namespace Laplace {
namespace {

CompatibilityReport failure(CompatibilityError code, std::string detail) {
    return package_report(code, std::move(detail));
}

std::optional<uint32_t> fact_u32(const ArtifactIndex& physical, CanonicalFactKey key,
                                 bool allow_singleton_vector = false) {
    for (const ArtifactFact& fact : physical.metadata_facts()) {
        if (fact.key != key || fact.state != ArtifactFactState::Present) continue;
        if (const auto* scalar = std::get_if<uint64_t>(&fact.value)) {
            if (*scalar <= UINT32_MAX) return static_cast<uint32_t>(*scalar);
            return std::nullopt;
        }
        if (allow_singleton_vector) {
            if (const auto* vector = std::get_if<std::vector<uint64_t>>(&fact.value);
                vector && vector->size() == 1 && vector->front() <= UINT32_MAX) {
                return static_cast<uint32_t>(vector->front());
            }
        }
        return std::nullopt;
    }
    return std::nullopt;
}

std::optional<float> fact_f32(const ArtifactIndex& physical, CanonicalFactKey key) {
    for (const ArtifactFact& fact : physical.metadata_facts()) {
        if (fact.key != key || fact.state != ArtifactFactState::Present) continue;
        const auto* bits = std::get_if<ArtifactF32Bits>(&fact.value);
        if (!bits) return std::nullopt;
        const float value = std::bit_cast<float>(bits->value);
        if (!std::isfinite(value) || value <= 0.0f) return std::nullopt;
        return value;
    }
    return std::nullopt;
}

std::optional<std::vector<uint32_t>> fact_u32_vector(const ArtifactIndex& physical,
                                                     CanonicalFactKey key) {
    for (const ArtifactFact& fact : physical.metadata_facts()) {
        if (fact.key != key || fact.state != ArtifactFactState::Present) continue;
        const auto* vector = std::get_if<std::vector<uint64_t>>(&fact.value);
        if (!vector || vector->size() > 4096) return std::nullopt;
        std::vector<uint32_t> values;
        values.reserve(vector->size());
        for (uint64_t value : *vector) {
            if (value > UINT32_MAX) return std::nullopt;
            values.push_back(static_cast<uint32_t>(value));
        }
        return values;
    }
    return std::nullopt;
}

const ArtifactTensorRecord* physical_tensor(const ArtifactIndex& index, uint32_t id) {
    for (const ArtifactTensorRecord& tensor : index.tensors()) if (tensor.id == id) return &tensor;
    return nullptr;
}

const NormalizedTensorEvidence* evidence_for(const NormalizedSourceEvidence& evidence,
                                             uint32_t source_id) {
    for (size_t index = 0; index != evidence.bindings().size(); ++index)
        if (evidence.bindings()[index].source_tensor_id == source_id) return &evidence.tensors()[index];
    return nullptr;
}

const ArtifactTensorRecord* unique_role_tensor(const ArtifactIndex& physical,
                                               TensorRole role, uint32_t layer = UINT32_MAX,
                                               bool expert = false, uint32_t slot = UINT32_MAX) {
    const ArtifactTensorRecord* result = nullptr;
    for (const ArtifactTensorRecord& tensor : physical.tensors()) {
        if (tensor.coordinate.layer != layer ||
            (tensor.coordinate.bank_axis != UINT8_MAX) != expert ||
            (slot != UINT32_MAX && tensor.coordinate.slot != slot)) continue;
        if (std::none_of(tensor.role_evidence.begin(), tensor.role_evidence.end(),
                         [role](const ArtifactTensorRoleEvidence& evidence) {
                             return evidence.role == role;
                         })) continue;
        if (result) return nullptr;
        result = &tensor;
    }
    return result;
}

ScalarType semantic_scalar(ArtifactScalarType type) {
    switch (type) {
    case ArtifactScalarType::F32: return ScalarType::F32;
    case ArtifactScalarType::F16: return ScalarType::F16;
    case ArtifactScalarType::U32: return ScalarType::U32;
    case ArtifactScalarType::I32: return ScalarType::I32;
    case ArtifactScalarType::U8:
    case ArtifactScalarType::Packed: return ScalarType::U8;
    default: return static_cast<ScalarType>(0);
    }
}

PhysicalLayout semantic_layout(const ArtifactTensorRecord& tensor) {
    return tensor.layout;
}

Quantization semantic_quantization(const ArtifactTensorRecord& tensor) {
    Quantization result;
    result.kind = static_cast<QuantizationKind>(tensor.quantization.kind);
    result.version = tensor.quantization.version;
    result.accumulation_type = tensor.quantization.accumulation_type;
    result.scale_type = tensor.quantization.scale_type;
    result.zero_type = tensor.quantization.zero_type;
    result.bias_type = tensor.quantization.bias_type;
    result.block_elements = tensor.quantization.block_elements;
    result.block_bytes = tensor.quantization.block_bytes;
    result.group_size = tensor.quantization.group_size;
    result.required_plane_mask = tensor.quantization.required_plane_mask;
    result.flags = tensor.quantization.flags;
    return result;
}

std::variant<ArtifactIndex, CompatibilityReport> normalized_physical_index(
    const ArtifactIndex& source, const NormalizedSourceEvidence& evidence) {
    ArtifactIndexInput input;
    input.artifacts.assign(source.artifacts().begin(), source.artifacts().end());
    input.metadata_facts.assign(source.metadata_facts().begin(), source.metadata_facts().end());
    input.package_facts.assign(source.package_facts().begin(), source.package_facts().end());
    input.aliases.assign(source.aliases().begin(), source.aliases().end());
    input.diagnostics.assign(source.diagnostics().begin(), source.diagnostics().end());
    input.tensors.assign(source.tensors().begin(), source.tensors().end());

    for (ArtifactTensorRecord& tensor : input.tensors) {
        const NormalizedTensorEvidence* typed = evidence_for(evidence, tensor.id);
        if (!typed) return failure(CompatibilityError::IMPORT_TENSOR_UNMAPPED, "normalized source tensor binding is missing");
        tensor.coordinate = typed->coordinate;
        if (tensor.coordinate.slot == UINT32_MAX) {
            // Source coordinates for ordinary layer tensors have no semantic
            // slot.  Keep that fact intact in the normalized evidence, while
            // giving the physical index a collision-free local coordinate for
            // repeated roles and global tensors.  The high-bit namespace is
            // deliberately disjoint from the importer-assigned post-norm
            // slots (0..3).
            if (tensor.id >= 0x80000000u) {
                return failure(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                               "normalized source tensor count exceeds coordinate namespace");
            }
            tensor.coordinate.slot = 0x80000000u | tensor.id;
        }
        tensor.role_evidence.push_back({typed->role, {0x4e530000u + tensor.id}, ArtifactFactAuthority::Structural,
                                        typed->coordinate});
        const bool expert = typed->coordinate.bank_axis != UINT8_MAX;
        const bool router = typed->role == TensorRole::NextnProjectionWeight &&
                            tensor.logical_dimensions.size() == 2;
        if (expert) {
            if (tensor.logical_dimensions.size() != 3 || tensor.logical_dimensions[2] > UINT32_MAX ||
                tensor.logical_dimensions[0] > UINT32_MAX || tensor.logical_dimensions[1] > UINT32_MAX ||
                tensor.planes.size() != 1 || tensor.planes[0].source.length == 0)
                return failure(CompatibilityError::IR_LAYOUT_MISMATCH, "normalized expert tensor shape is invalid");
            const uint64_t experts = tensor.logical_dimensions[2];
            if (tensor.planes[0].source.length % experts != 0)
                return failure(CompatibilityError::IR_LAYOUT_MISMATCH, "expert tensor bytes do not divide by expert extent");
            tensor.coordinate.bank_axis = 2;
            tensor.coordinate.bank_extent = static_cast<uint32_t>(experts);
            tensor.coordinate.bank_stride = tensor.planes[0].source.length / experts;
            if (!tensor.coordinate.bank_stride)
                return failure(CompatibilityError::IR_LAYOUT_MISMATCH, "expert tensor has zero physical bank stride");
        } else if (router) {
            if (tensor.logical_dimensions[0] > UINT32_MAX || tensor.logical_dimensions[1] > UINT32_MAX)
                return failure(CompatibilityError::IR_SHAPE_MISMATCH, "router tensor dimensions exceed ABI bounds");
            // The source record remains in its declared [hidden, experts]
            // order; the semantic matrix view below is [experts, hidden].
        }
    }

    // A source may intentionally omit a value projection while retaining a
    // key projection.  Represent that declared shared KV geometry as an
    // exact-span alias with a distinct physical coordinate; this keeps the
    // semantic graph explicit without copying bytes or selecting a source
    // variant.  Ambiguous key/value candidates fail closed.
    const size_t source_tensor_count = input.tensors.size();
    for (size_t index = 0; index != source_tensor_count; ++index) {
        const ArtifactTensorRecord& key = input.tensors[index];
        if (key.coordinate.layer == UINT32_MAX ||
            !std::any_of(key.role_evidence.begin(), key.role_evidence.end(),
                         [](const ArtifactTensorRoleEvidence& evidence) {
                             return evidence.role == TensorRole::KeyWeight;
                         })) continue;
        const bool has_value = std::any_of(
            input.tensors.begin(), input.tensors.begin() + source_tensor_count,
            [&](const ArtifactTensorRecord& candidate) {
                return candidate.coordinate.layer == key.coordinate.layer &&
                       std::any_of(candidate.role_evidence.begin(), candidate.role_evidence.end(),
                                   [](const ArtifactTensorRoleEvidence& evidence) {
                                       return evidence.role == TensorRole::ValueWeight;
                                   });
            });
        if (has_value) continue;
        ArtifactTensorRecord value = key;
        value.id = static_cast<uint32_t>(input.tensors.size());
        if (value.id >= 0x80000000u) {
            return failure(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                           "normalized source exceeds shared-KV alias namespace");
        }
        value.coordinate.slot = 0x80000000u | value.id;
        value.role_evidence.clear();
        value.role_evidence.push_back({TensorRole::ValueWeight,
                                        {0x56414c00u + key.id},
                                        ArtifactFactAuthority::Derived,
                                        key.coordinate});
        input.tensors.push_back(value);
        input.aliases.push_back({ArtifactAliasKind::ExactSharedSpan,
                                 ArtifactAliasDirection::SourceToTarget,
                                 key.id, value.id, TensorRole::ValueWeight,
                                 {0x56414c00u + key.id},
                                 ArtifactFactAuthority::Derived, key.coordinate});
    }

    // Resolve output authority from normalized roles.  A separate output
    // tensor is already authoritative; only an absent output may be carried
    // by an explicitly-derived tied span.  Never synthesize an output when
    // neither evidence path exists.
    uint32_t embedding_id = UINT32_MAX;
    uint32_t output_id = UINT32_MAX;
    for (const ArtifactTensorRecord& tensor : input.tensors) {
        if (std::any_of(tensor.role_evidence.begin(), tensor.role_evidence.end(),
                        [](const ArtifactTensorRoleEvidence& evidence) {
                            return evidence.role == TensorRole::TokenEmbedding;
                        })) {
            if (embedding_id != UINT32_MAX) {
                return failure(CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS,
                               "normalized source has multiple token embedding candidates");
            }
            embedding_id = tensor.id;
        }
        if (std::any_of(tensor.role_evidence.begin(), tensor.role_evidence.end(),
                        [](const ArtifactTensorRoleEvidence& evidence) {
                            return evidence.role == TensorRole::OutputWeight;
                        })) {
            if (output_id != UINT32_MAX) {
                return failure(CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS,
                               "normalized source has multiple output tensor candidates");
            }
            output_id = tensor.id;
        }
    }
    if (embedding_id == UINT32_MAX || input.tensors.size() >= UINT32_MAX)
        return failure(CompatibilityError::IMPORT_TENSOR_UNMAPPED, "GGUF source has no token embedding tensor");
    if (output_id != UINT32_MAX) return ArtifactIndex::build(std::move(input));

    // The importer has established a tied-output candidate only when the
    // source contains no declared output role and the token embedding is a
    // complete, directly-bindable matrix.  The alias itself is retained as
    // canonical evidence and is verified by ArtifactIndex against exact
    // shared spans.
    const ArtifactTensorRecord* embedding = nullptr;
    for (const auto& tensor : input.tensors) if (tensor.id == embedding_id) embedding = &tensor;
    if (!embedding) return failure(CompatibilityError::IMPORT_TENSOR_UNMAPPED, "GGUF token embedding binding is missing");
    ArtifactTensorRecord output = *embedding;
    output.id = static_cast<uint32_t>(input.tensors.size());
    output.coordinate = {};
    output.role_evidence.clear();
    output.role_evidence.push_back({TensorRole::OutputWeight, {0x4f555450u}, ArtifactFactAuthority::Declared, {}});
    input.tensors.push_back(std::move(output));
    input.aliases.push_back({ArtifactAliasKind::TiedOutput, ArtifactAliasDirection::SourceToTarget,
                             embedding_id, static_cast<uint32_t>(input.tensors.size() - 1),
                             TensorRole::OutputWeight, {0x4f555450u}, ArtifactFactAuthority::Declared, {}});
    for (size_t left = 0; left != input.tensors.size(); ++left) {
        const auto& coordinate = input.tensors[left].coordinate;
        const bool set = coordinate.root != 0 || coordinate.layer != UINT32_MAX ||
                         coordinate.slot != UINT32_MAX || coordinate.instance != UINT32_MAX ||
                         coordinate.expert != UINT32_MAX || coordinate.bank_axis != UINT8_MAX;
        if (!set) continue;
        for (size_t right = left + 1; right != input.tensors.size(); ++right) {
            if (input.tensors[right].coordinate == coordinate) {
                return failure(CompatibilityError::IMPORT_TENSOR_DUPLICATE,
                               "normalized source coordinates collide: " +
                               std::to_string(input.tensors[left].id) + "/" +
                               std::to_string(input.tensors[right].id) + " layer=" +
                               std::to_string(coordinate.layer) + " slot=" +
                               std::to_string(coordinate.slot));
            }
        }
    }
    auto rebuilt = ArtifactIndex::build(std::move(input));
    if (auto* report = std::get_if<CompatibilityReport>(&rebuilt)) {
        report->detail += " (normalized source physical rebuild)";
    }
    return rebuilt;
}

std::optional<SemanticTensor> semantic_tensor(const ArtifactTensorRecord& physical,
                                              const NormalizedTensorEvidence& typed) {
    SemanticTensor result;
    result.id = physical.id;
    result.role = typed.role;
    result.logical_type = physical.logical_type == ArtifactScalarType::F16
        ? ScalarType::F16 : ScalarType::F32;
    std::vector<uint64_t> semantic_dimensions = physical.logical_dimensions;
    if (physical.coordinate.bank_axis != UINT8_MAX) {
        if (semantic_dimensions.size() != 3) return std::nullopt;
        std::rotate(semantic_dimensions.begin(), semantic_dimensions.begin() + 2, semantic_dimensions.end());
    }
    const bool router_matrix = typed.role == TensorRole::NextnProjectionWeight &&
                               typed.dimensions.size() == 2;
    if (router_matrix) std::swap(semantic_dimensions[0], semantic_dimensions[1]);
    for (uint64_t dimension : semantic_dimensions)
        result.dimensions.push_back({DimensionKind::Constant, dimension});
    result.layout = semantic_layout(physical);
    if (physical.coordinate.bank_axis != UINT8_MAX) {
        result.layout.axis_order = {1, 2, 0, 0xff, 0xff, 0xff, 0xff, 0xff};
    } else if (router_matrix) {
        result.layout.axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    }
    result.quantization = semantic_quantization(physical);
    if (physical.coordinate.bank_axis != UINT8_MAX) {
        if (physical.logical_dimensions.size() != 3 || physical.coordinate.bank_axis != 2) return std::nullopt;
        result.expert_axis = {ExpertAxisKind::ExpertBank, 0, 0xff, 1, 2,
                              physical.coordinate.bank_extent, physical.coordinate.bank_stride, 0};
    }
    for (const ArtifactTensorPlane& plane : physical.planes) {
        const ScalarType storage = semantic_scalar(plane.storage_type);
        if (storage == static_cast<ScalarType>(0)) return std::nullopt;
        result.planes.push_back({plane.kind, storage, plane.source.artifact_id,
                                 plane.source.offset, plane.source.length,
                                 plane.alignment, 0});
    }
    if (typed.role == TensorRole::NextnEmbeddingNormWeight ||
        (typed.role == TensorRole::PostNormWeight && typed.coordinate.slot == 0)) {
        // Historical programs left these unconsumed; activity is now decided
        // by the built graph, which marks every unreferenced tensor
        // inactive-program instead of pre-guessing by role.
    }
    return result;
}

struct LayerRefs {
    uint32_t attn_norm, q, k, v, q_norm, k_norm, attn_out, ffn_norm, gate, up, down;
    uint32_t q_bias, k_bias, v_bias;
    uint32_t post_attn, post_ffw;
    uint32_t route_scale, router, expert_norm, expert_up, expert_down, reduce;
    uint32_t post_dense, post_moe, post_output, output_scale;
};

uint32_t find_tensor(const SemanticModel& model, uint32_t layer, TensorRole role,
                     bool expert = false, uint32_t slot = UINT32_MAX) {
    (void)layer;
    (void)slot;
    for (const SemanticTensor& tensor : model.tensors)
        if (tensor.role == role && (tensor.expert_axis.kind == ExpertAxisKind::ExpertBank) == expert)
            return tensor.id;
    return UINT32_MAX;
}

// Resolve layer-local tensors from the normalized coordinate captured in the
// source evidence.  This helper is kept separate from semantic tensor roles:
// role and coordinates are both required to avoid ambiguous repeated roles.
uint32_t find_layer_tensor(const SemanticModel& model, const ArtifactIndex& physical,
                           uint32_t layer, TensorRole role, bool expert = false,
                           uint32_t slot = UINT32_MAX) {
    for (const SemanticTensor& tensor : model.tensors) {
        if (tensor.role != role || (tensor.expert_axis.kind == ExpertAxisKind::ExpertBank) != expert) continue;
        const ArtifactTensorRecord* record = physical_tensor(physical, tensor.id);
        if (!record || record->coordinate.layer != layer) continue;
        if (slot != UINT32_MAX && record->coordinate.slot != slot) continue;
        return tensor.id;
    }
    return UINT32_MAX;
}

std::vector<Dimension> rows(uint32_t width) {
    return {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, width}};
}

std::vector<Dimension> routed(uint32_t selected, uint32_t width) {
    return {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, selected},
            {DimensionKind::Constant, width}};
}

bool has_gguf_physical_tuple(const SemanticTensor& tensor, uint32_t elements,
                             uint32_t bytes, ScalarType zero_type) {
    if (tensor.logical_type != ScalarType::F32 ||
        tensor.layout.kind != PhysicalLayoutKind::GgufBlocked ||
        tensor.layout.version != 1 || tensor.layout.packing != PackingKind::Gguf ||
        tensor.layout.block_rank != 1 || tensor.layout.block_elements != elements ||
        tensor.layout.block_bytes != bytes || tensor.layout.flags != 0 ||
        tensor.quantization.kind != QuantizationKind::BlockedAffine ||
        tensor.quantization.version != 1 ||
        tensor.quantization.accumulation_type != ScalarType::F32 ||
        tensor.quantization.scale_type != ScalarType::F16 ||
        tensor.quantization.zero_type != zero_type ||
        tensor.quantization.bias_type != static_cast<ScalarType>(0) ||
        tensor.quantization.block_elements != elements ||
        tensor.quantization.block_bytes != bytes ||
        tensor.quantization.group_size != elements ||
        tensor.quantization.required_plane_mask != artifact_plane_mask(PlaneKind::Values) ||
        tensor.quantization.flags != 0 || tensor.planes.size() != 1 ||
        tensor.planes[0].kind != PlaneKind::Values ||
        tensor.planes[0].storage_type != ScalarType::U8 ||
        tensor.planes[0].flags != 0 ||
        tensor.planes[0].length == 0 ||
        tensor.planes[0].length % bytes != 0) return false;
    uint64_t constant_elements = 1;
    for (const Dimension& dimension : tensor.dimensions) {
        if (dimension.kind != DimensionKind::Constant || dimension.constant_or_symbol == 0 ||
            constant_elements > UINT64_MAX / dimension.constant_or_symbol) return false;
        constant_elements *= dimension.constant_or_symbol;
    }
    return constant_elements % elements == 0 &&
           tensor.planes[0].length / (constant_elements / elements) == bytes;
}

// Codec selection is data: one row per blocked codec ABI (block geometry
// plus zero-point type) with its arithmetic certificate. Extending codec
// coverage means adding a row here and its certificate constructor, not
// another branch in the loader.
using CodecCertificateConstructor = std::vector<uint8_t> (*)();
struct GgufCodecSelector {
    uint32_t block_elements;
    uint32_t block_bytes;
    ScalarType zero_type;
    CodecCertificateConstructor certificate;
};

constexpr GgufCodecSelector kGgufCodecSelectors[] = {
    {256, 144, ScalarType::F16, make_q4_k_codec_certificate},
    {32, 22, static_cast<ScalarType>(0), make_q5_0_codec_certificate},
    {32, 34, static_cast<ScalarType>(0), make_q8_0_codec_certificate},
    {256, 210, static_cast<ScalarType>(0), make_q6_k_codec_certificate},
};

std::vector<uint8_t> certificate_for_tensor(const SemanticTensor& tensor) {
    if (tensor.quantization.kind == QuantizationKind::None) {
        if (tensor.logical_type == ScalarType::F16 &&
            tensor.layout.kind == PhysicalLayoutKind::ContiguousRowMajor &&
            tensor.layout.version == 1 && tensor.layout.packing == PackingKind::None &&
            tensor.layout.block_rank == 0 && tensor.layout.block_elements == 0 &&
            tensor.layout.block_bytes == 0 && tensor.quantization.version == 1 &&
            tensor.quantization.accumulation_type == ScalarType::F32 &&
            tensor.quantization.scale_type == static_cast<ScalarType>(0) &&
            tensor.quantization.zero_type == static_cast<ScalarType>(0) &&
            tensor.quantization.bias_type == static_cast<ScalarType>(0) &&
            tensor.quantization.block_elements == 0 && tensor.quantization.block_bytes == 0 &&
            tensor.quantization.group_size == 0 && tensor.quantization.required_plane_mask == 0 &&
            tensor.quantization.flags == 0 && tensor.planes.size() == 1 &&
            tensor.planes[0].kind == PlaneKind::Values &&
            tensor.planes[0].storage_type == ScalarType::F16 &&
            tensor.planes[0].flags == 0)
            return make_raw_f16_codec_certificate();
        if (tensor.logical_type == ScalarType::F32 &&
            tensor.layout.kind == PhysicalLayoutKind::ContiguousRowMajor &&
            tensor.layout.version == 1 && tensor.layout.packing == PackingKind::None &&
            tensor.layout.block_rank == 0 && tensor.layout.block_elements == 0 &&
            tensor.layout.block_bytes == 0 && tensor.quantization.version == 1 &&
            tensor.quantization.accumulation_type == ScalarType::F32 &&
            tensor.quantization.scale_type == static_cast<ScalarType>(0) &&
            tensor.quantization.zero_type == static_cast<ScalarType>(0) &&
            tensor.quantization.bias_type == static_cast<ScalarType>(0) &&
            tensor.quantization.block_elements == 0 && tensor.quantization.block_bytes == 0 &&
            tensor.quantization.group_size == 0 && tensor.quantization.required_plane_mask == 0 &&
            tensor.quantization.flags == 0 && tensor.planes.size() == 1 &&
            tensor.planes[0].kind == PlaneKind::Values &&
            tensor.planes[0].storage_type == ScalarType::F32 &&
            tensor.planes[0].flags == 0)
            return make_raw_f32_codec_certificate();
        return {};
    }
    for (const GgufCodecSelector& selector : kGgufCodecSelectors) {
        if (has_gguf_physical_tuple(tensor, selector.block_elements,
                                    selector.block_bytes, selector.zero_type)) {
            return selector.certificate();
        }
    }
    return {};
}

uint32_t add_value(SemanticModel& model, ScalarType type, std::vector<Dimension> dimensions) {
    const uint32_t id = static_cast<uint32_t>(model.values.size());
    model.values.push_back({id, type, std::move(dimensions), 0});
    return id;
}

void add_operator(SemanticModel& model, OperatorKind kind, std::vector<uint32_t> inputs,
                  std::vector<uint32_t> outputs, std::vector<uint32_t> tensors,
                  std::vector<uint32_t> states, OperatorPayload payload) {
    const uint32_t id = static_cast<uint32_t>(model.operators.size());
    model.operators.push_back({id, kind, 7, std::move(inputs), std::move(outputs),
                               std::move(tensors), std::move(states), std::move(payload)});
}

std::variant<PhysicalCodecRegistry, CompatibilityReport>
build_codec_registry(const SemanticModel& model) {
    PhysicalCodecRegistry registry;
    for (const SemanticTensor& tensor : model.tensors) {
        const std::vector<uint8_t> certificate = certificate_for_tensor(tensor);
        if (certificate.empty()) {
            return failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                           "normalized source has no authoritative codec certificate for a tensor");
        }
        const PhysicalIdentityDigest arithmetic_digest = codec_certificate_digest(certificate);
        const auto identity = physical_codec_identity(tensor, 1, arithmetic_digest);
        if (!identity) {
            return failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                           "normalized source tensor cannot produce a complete codec identity");
        }
        const auto parsed = parse_codec_certificate(certificate);
        const auto* parsed_certificate = std::get_if<CodecCertificate>(&parsed);
        if (!parsed_certificate || !parsed_certificate->matches_physical_identity(*identity)) {
            return failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                           "normalized source codec certificate does not match its physical identity");
        }
        auto codec = std::find_if(registry.codecs.begin(), registry.codecs.end(),
                                 [&](const PhysicalCodecSpec& candidate) {
                                     return candidate.identity == *identity;
                                 });
        if (codec == registry.codecs.end()) {
            registry.codecs.push_back({*identity, certificate});
        }
        registry.tensors.push_back({tensor.id, *identity});
    }
    std::sort(registry.codecs.begin(), registry.codecs.end(),
              [](const PhysicalCodecSpec& left, const PhysicalCodecSpec& right) {
                  return physical_codec_identity_less(left.identity, right.identity);
              });
    std::sort(registry.tensors.begin(), registry.tensors.end(),
              [](const PhysicalTensorCodecDeclaration& left,
                 const PhysicalTensorCodecDeclaration& right) {
                  return left.tensor_id < right.tensor_id;
              });
    if (!validate_physical_codec_registry(registry, true) ||
        !physical_codec_registry_is_canonical(registry) ||
        !physical_codec_registry_matches_model(registry, model)) {
        return failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                       "normalized source codec registry is incomplete or non-canonical");
    }
    return registry;
}

std::optional<SemanticModel> build_model(const ArtifactIndex& physical,
                                         const NormalizedSourceEvidence& evidence) {
    using namespace gguf_fact_keys;
    const auto layers = fact_u32(physical, block_count);
    const auto hidden = fact_u32(physical, embedding_length);
    const auto experts = fact_u32(physical, expert_count);
    const auto selected = fact_u32(physical, expert_used_count);
    const auto expert_inter = fact_u32(physical, expert_feed_forward_length);
    const auto epsilon = fact_f32(physical, attention_layer_norm_rms_epsilon);
    const auto maximum_context = fact_u32(physical, context_length);
    const auto q_heads_fact = fact_u32(physical, attention_head_count);
    const auto kv_heads_fact = fact_u32(physical, attention_head_count_kv, true);
    const auto rope_base = fact_f32(physical, rope_freq_base);
    const auto declared_attention_scale = fact_f32(physical, attention_scale);
    for (const ArtifactFact& fact : physical.metadata_facts()) {
        if (fact.key == attention_scale && fact.state != ArtifactFactState::Missing &&
            !declared_attention_scale)
            return std::nullopt;
    }
    const auto bos = fact_u32(physical, bos_token_id);
    const auto eos = fact_u32(physical, eos_token_id);
    if (!layers || !hidden || !epsilon ||
        !maximum_context || !q_heads_fact || !rope_base || !bos || !eos ||
        *layers == 0 || *layers > 4096 || *hidden == 0 ||
        *maximum_context == 0 || *maximum_context > 262144 || *q_heads_fact == 0) {
        return std::nullopt;
    }
    // Routed-expert facts are all-or-nothing: a source either carries the
    // complete routed contract or is a plain dense model.
    const bool moe_source = experts.has_value() || selected.has_value() || expert_inter.has_value();
    if (moe_source && (!experts || !selected || !expert_inter ||
                   *experts == 0 || *selected == 0 ||
                   *selected > *experts || *selected > 16 || *expert_inter == 0)) {
        return std::nullopt;
    }
    // Windowed attention: the scalar window fact applies to the layers the
    // pattern marks. Without a declared pattern every layer takes the
    // window, which is exact within the window and upgradeable by declaring
    // the pattern fact; a pattern that disagrees with the layer count is a
    // contradiction and fails closed.
    const auto sliding_window = fact_u32(physical, attention_sliding_window);
    const auto window_pattern = fact_u32_vector(physical, attention_sliding_window_pattern);
    std::vector<bool> layer_windowed;
    if (sliding_window && *sliding_window != 0) {
        if (!window_pattern) {
            layer_windowed.assign(*layers, true);
        } else if (window_pattern->size() == *layers) {
            layer_windowed.reserve(*layers);
            for (uint32_t marked : *window_pattern) layer_windowed.push_back(marked != 0);
        } else {
            return std::nullopt;
        }
    } else {
        layer_windowed.assign(*layers, false);
    }
    // Rope and embedding geometry that GGUF cannot express are declared
    // facts with safe defaults: sources that need another value declare it
    // (sidecar or a converter that writes the key) instead of the loader
    // guessing from architecture names.
    const float embed_scale = fact_f32(physical, embedding_scale).value_or(1.0f);
    const uint32_t activation_fact = fact_u32(physical, feed_forward_activation)
                                         .value_or(static_cast<uint32_t>(ActivationKind::Silu));
    if (activation_fact != static_cast<uint32_t>(ActivationKind::Silu) &&
        activation_fact != static_cast<uint32_t>(ActivationKind::GeluTanh)) {
        return std::nullopt;
    }
    const ActivationKind activation = static_cast<ActivationKind>(activation_fact);
    const float sliding_rope_base = fact_f32(physical, rope_freq_base_swa).value_or(10000.0f);
    const auto sliding_rope_dim = fact_u32(physical, rope_dimension_count_swa);
    const float global_rope_scale = fact_f32(physical, rope_scaling_freq_scale).value_or(1.0f);

    // Vocabulary and projection geometry come from typed tensor coordinates
    // and declared dimensions.  This is also the cross-check that prevents a
    // metadata value from silently changing the executable graph shape.
    const ArtifactTensorRecord* embedding = unique_role_tensor(physical, TensorRole::TokenEmbedding);
    const ArtifactTensorRecord* q_tensor = unique_role_tensor(physical, TensorRole::QueryWeight, 0);
    const ArtifactTensorRecord* k_tensor = unique_role_tensor(physical, TensorRole::KeyWeight, 0);
    const ArtifactTensorRecord* v_tensor = unique_role_tensor(physical, TensorRole::ValueWeight, 0);
    const ArtifactTensorRecord* q_norm_tensor =
        unique_role_tensor(physical, TensorRole::AttentionQueryNormWeight, 0);
    if (!embedding || embedding->logical_dimensions.size() != 2 ||
        embedding->logical_dimensions[1] == 0 || embedding->logical_dimensions[1] > UINT32_MAX ||
        !q_tensor || !k_tensor || !v_tensor || q_tensor->logical_dimensions.size() != 2 ||
        k_tensor->logical_dimensions.size() != 2 || v_tensor->logical_dimensions.size() != 2 ||
        q_tensor->logical_dimensions[0] != *hidden ||
        k_tensor->logical_dimensions[0] != *hidden || v_tensor->logical_dimensions[0] != *hidden ||
        (q_norm_tensor && (q_norm_tensor->logical_dimensions.size() != 1 ||
                           q_norm_tensor->logical_dimensions[0] == 0 ||
                           q_norm_tensor->logical_dimensions[0] > UINT32_MAX))) {
        return std::nullopt;
    }
    const uint32_t q_width = static_cast<uint32_t>(q_tensor->logical_dimensions[1]);
    const uint32_t kv_width = static_cast<uint32_t>(k_tensor->logical_dimensions[1]);
    if (q_width == 0 || kv_width == 0 || q_width % *q_heads_fact != 0) return std::nullopt;
    const uint32_t head_dim = q_width / *q_heads_fact;
    if (head_dim == 0 || (q_norm_tensor && q_norm_tensor->logical_dimensions[0] != head_dim) ||
        v_tensor->logical_dimensions[1] != kv_width) return std::nullopt;
    const uint32_t kv_heads = kv_heads_fact.value_or(
        (kv_width % head_dim == 0 ? kv_width / head_dim : 0));
    if (kv_heads == 0 || kv_width != kv_heads * head_dim || *q_heads_fact % kv_heads != 0) {
        return std::nullopt;
    }

    SemanticModel model;
    model.schema_major = 7; model.opset_major = 7; model.maximum_context =
        *maximum_context;
    model.vocabulary_size = static_cast<uint32_t>(embedding->logical_dimensions[1]);
    model.bos_id = *bos < model.vocabulary_size ? *bos : UINT32_MAX;
    model.eos_id = *eos < model.vocabulary_size ? *eos : UINT32_MAX;
    if (model.bos_id == UINT32_MAX || model.eos_id == UINT32_MAX) return std::nullopt;
    model.tokenizer_digest = physical.artifacts().front().digest().bytes;
    model.template_digest = physical.artifacts().front().digest().bytes;
    model.stop_ids = {model.eos_id};

    for (const ArtifactTensorRecord& record : physical.tensors()) {
        const NormalizedTensorEvidence* typed = evidence_for(evidence, record.id);
        if (!typed) {
            const bool alias_target = std::any_of(
                physical.aliases().begin(), physical.aliases().end(),
                [&](const ArtifactAlias& alias) { return alias.target_tensor_id == record.id; });
            if (alias_target) {
                continue;
            }
            return std::nullopt;
        }
        auto converted = semantic_tensor(record, *typed);
        if (!converted) return std::nullopt;
        model.tensors.push_back(std::move(*converted));
    }
    if (model.tensors.empty()) return std::nullopt;
    // The alias target is a semantic copy of the token embedding, with the
    // same exact physical binding and the output role.
    // Materialize semantic views for exact-span aliases (shared KV and tied
    // output).  The source tensor remains the sole byte owner; the alias is
    // only a role-bearing view with the same physical binding.
    for (const ArtifactAlias& alias : physical.aliases()) {
        const SemanticTensor* source = nullptr;
        for (const SemanticTensor& tensor : model.tensors) {
            if (tensor.id == alias.source_tensor_id) {
                if (source) return std::nullopt;
                source = &tensor;
            }
        }
        if (!source) return std::nullopt;
        SemanticTensor target = *source;
        target.id = alias.target_tensor_id;
        target.role = alias.semantic_role;
        target.flags = 0;
        model.tensors.push_back(std::move(target));
    }
    std::sort(model.tensors.begin(), model.tensors.end(),
              [](const SemanticTensor& left, const SemanticTensor& right) {
                  return left.id < right.id;
              });

    const uint32_t token = add_value(model, ScalarType::U32, {{DimensionKind::Symbol, 1}});
    uint32_t current = add_value(model, ScalarType::F32, rows(*hidden));
    const uint32_t embedding_tensor = find_tensor(model, 0, TensorRole::TokenEmbedding);
    if (embedding_tensor == UINT32_MAX) return std::nullopt;
    add_operator(model, OperatorKind::EmbeddingLookup, {token}, {current}, {embedding_tensor}, {},
                 EmbeddingLookupPayload{std::bit_cast<uint32_t>(embed_scale),
                                        model.vocabulary_size, *hidden, 0});
    model.input_values_first = token; model.input_values_count = 1;

    const uint32_t q_heads = *q_heads_fact;
    const uint32_t intermediate = fact_u32(physical, feed_forward_length).value_or(0);
    if (intermediate == 0) return std::nullopt;
    std::vector<LayerRefs> refs;
    refs.reserve(*layers);
    const uint32_t norm_bits = std::bit_cast<uint32_t>(*epsilon);
    for (uint32_t layer = 0; layer != *layers; ++layer) {
        const ArtifactTensorRecord* layer_q = unique_role_tensor(
            physical, TensorRole::QueryWeight, layer);
        const ArtifactTensorRecord* layer_k = unique_role_tensor(
            physical, TensorRole::KeyWeight, layer);
        const ArtifactTensorRecord* layer_v = unique_role_tensor(
            physical, TensorRole::ValueWeight, layer);
        const ArtifactTensorRecord* layer_q_norm = unique_role_tensor(
            physical, TensorRole::AttentionQueryNormWeight, layer);
        if (!layer_q || !layer_k || !layer_v ||
            layer_q->logical_dimensions.size() != 2 ||
            layer_k->logical_dimensions.size() != 2 ||
            layer_v->logical_dimensions.size() != 2 ||
            (layer_q_norm && layer_q_norm->logical_dimensions.size() != 1) ||
            layer_q->logical_dimensions[0] != *hidden ||
            layer_k->logical_dimensions[0] != *hidden ||
            layer_v->logical_dimensions[0] != *hidden ||
            layer_v->logical_dimensions[1] != layer_k->logical_dimensions[1] ||
            layer_q->logical_dimensions[1] == 0 ||
            layer_k->logical_dimensions[1] == 0 ||
            (layer_q_norm && layer_q_norm->logical_dimensions[0] == 0) ||
            layer_q->logical_dimensions[1] % *q_heads_fact != 0) {
            return std::nullopt;
        }
        const uint32_t layer_q_width = static_cast<uint32_t>(layer_q->logical_dimensions[1]);
        const uint32_t layer_kv_width = static_cast<uint32_t>(layer_k->logical_dimensions[1]);
        const uint32_t layer_head_dim = layer_q_width / *q_heads_fact;
        if (layer_head_dim == 0 ||
            (layer_q_norm && layer_q_norm->logical_dimensions[0] != layer_head_dim)) {
            return std::nullopt;
        }
        const uint32_t layer_kv_heads = kv_heads_fact.value_or(
            (layer_kv_width % layer_head_dim == 0 ? layer_kv_width / layer_head_dim : 0));
        if (layer_kv_heads == 0 || layer_kv_heads * layer_head_dim != layer_kv_width ||
            *q_heads_fact % layer_kv_heads != 0) {
            return std::nullopt;
        }
        const uint32_t key_state_id = static_cast<uint32_t>(model.states.size());
        StateFormat key_format;
        key_format.encoded_domain = TransformDomain::RopeApplied;
        key_format.alignment = 64;
        StateFormat value_format;
        value_format.alignment = 64;
        model.states.push_back({key_state_id, StateKind::KeyCache, 7,
                                StateUpdateKind::AppendKey, PositionPolicy::AppendOnly,
                                {{DimensionKind::Symbol, 1},
                                 {DimensionKind::Constant, layer_kv_heads},
                                 {DimensionKind::Constant, layer_head_dim}},
                                {key_format}, 0});
        model.states.push_back({key_state_id + 1, StateKind::ValueCache, 7,
                                StateUpdateKind::AppendValue, PositionPolicy::AppendOnly,
                                {{DimensionKind::Symbol, 1},
                                 {DimensionKind::Constant, layer_kv_heads},
                                 {DimensionKind::Constant, layer_head_dim}},
                                {value_format}, 0});
        const uint32_t first_operator = static_cast<uint32_t>(model.operators.size());
        LayerRefs r{};
        r.attn_norm = find_layer_tensor(model, physical, layer, TensorRole::AttentionNormWeight);
        r.q = find_layer_tensor(model, physical, layer, TensorRole::QueryWeight);
        r.k = find_layer_tensor(model, physical, layer, TensorRole::KeyWeight);
        r.v = find_layer_tensor(model, physical, layer, TensorRole::ValueWeight);
        r.q_norm = find_layer_tensor(model, physical, layer, TensorRole::AttentionQueryNormWeight);
        r.k_norm = find_layer_tensor(model, physical, layer, TensorRole::AttentionKeyNormWeight);
        r.attn_out = find_layer_tensor(model, physical, layer, TensorRole::AttentionOutputWeight);
        r.ffn_norm = find_layer_tensor(model, physical, layer, TensorRole::FfnNormWeight);
        r.gate = find_layer_tensor(model, physical, layer, TensorRole::FfnGateWeight);
        r.up = find_layer_tensor(model, physical, layer, TensorRole::FfnUpWeight);
        r.down = find_layer_tensor(model, physical, layer, TensorRole::FfnDownWeight);
        r.q_bias = find_layer_tensor(model, physical, layer, TensorRole::QueryBias);
        r.k_bias = find_layer_tensor(model, physical, layer, TensorRole::KeyBias);
        r.v_bias = find_layer_tensor(model, physical, layer, TensorRole::ValueBias);
        r.post_attn = find_layer_tensor(model, physical, layer, TensorRole::PostNormWeight, false, 0);
        r.post_ffw = find_layer_tensor(model, physical, layer, TensorRole::PostNormWeight, false, 3);
        r.route_scale = find_layer_tensor(model, physical, layer, TensorRole::RouterScaleWeight);
        r.router = find_layer_tensor(model, physical, layer, TensorRole::NextnProjectionWeight);
        r.expert_norm = find_layer_tensor(model, physical, layer, TensorRole::ExpertNormWeight);
        r.expert_up = find_layer_tensor(model, physical, layer, TensorRole::FfnUpWeight, true);
        r.expert_down = find_layer_tensor(model, physical, layer, TensorRole::FfnDownWeight, true);
        r.reduce = find_layer_tensor(model, physical, layer, TensorRole::ReduceScaleWeight);
        r.post_dense = find_layer_tensor(model, physical, layer, TensorRole::PostNormWeight, false, 1);
        r.post_moe = find_layer_tensor(model, physical, layer, TensorRole::PostNormWeight, false, 2);
        r.post_output = find_layer_tensor(model, physical, layer, TensorRole::PostNormWeight, false, 3);
        r.output_scale = find_layer_tensor(model, physical, layer, TensorRole::OutputScaleWeight);
        if (r.attn_norm == UINT32_MAX || r.q == UINT32_MAX || r.k == UINT32_MAX ||
            r.v == UINT32_MAX || r.attn_out == UINT32_MAX || r.ffn_norm == UINT32_MAX ||
            r.gate == UINT32_MAX || r.up == UINT32_MAX || r.down == UINT32_MAX) {
            return std::nullopt;
        }
        if (moe_source && (r.route_scale == UINT32_MAX || r.router == UINT32_MAX ||
                       r.expert_norm == UINT32_MAX || r.expert_up == UINT32_MAX ||
                       r.expert_down == UINT32_MAX || r.reduce == UINT32_MAX ||
                       r.post_dense == UINT32_MAX || r.post_moe == UINT32_MAX ||
                       r.post_output == UINT32_MAX || r.output_scale == UINT32_MAX)) {
            return std::nullopt;
        }

        const uint32_t normed = add_value(model, ScalarType::F32, rows(*hidden));
        const uint32_t qv = add_value(model, ScalarType::F32, rows(layer_q_width));
        const uint32_t kv = add_value(model, ScalarType::F32, rows(layer_kv_width));
        const uint32_t vv = add_value(model, ScalarType::F32, rows(layer_kv_width));
        // Without a source q/k norm, Rope reads the projected values
        // directly; no unused intermediate values enter the graph.
        const uint32_t qn = r.q_norm != UINT32_MAX
            ? add_value(model, ScalarType::F32, rows(layer_q_width)) : qv;
        const uint32_t kn = r.k_norm != UINT32_MAX
            ? add_value(model, ScalarType::F32, rows(layer_kv_width)) : kv;
        const uint32_t qr = add_value(model, ScalarType::F32, rows(layer_q_width));
        const uint32_t ar = add_value(model, ScalarType::F32, rows(layer_kv_width));
        const uint32_t ao = add_value(model, ScalarType::F32, rows(layer_q_width));
        const uint32_t attn_proj = add_value(model, ScalarType::F32, rows(*hidden));
        const uint32_t attn_res = add_value(model, ScalarType::F32, rows(*hidden));
        const uint32_t fn = add_value(model, ScalarType::F32, rows(*hidden));
        const uint32_t dg = add_value(model, ScalarType::F32, rows(intermediate));
        const uint32_t du = add_value(model, ScalarType::F32, rows(intermediate));
        const uint32_t dh = add_value(model, ScalarType::F32, rows(intermediate));
        const uint32_t dd = add_value(model, ScalarType::F32, rows(*hidden));
        // Sandwich-norm sources (the gemma shape) norm both after the
        // attention residual and after the FFN; without those tensors the
        // graph is unchanged.
        const uint32_t normed2 = r.post_attn != UINT32_MAX
            ? add_value(model, ScalarType::F32, rows(*hidden)) : attn_res;
        const uint32_t dd_out = r.post_ffw != UINT32_MAX
            ? add_value(model, ScalarType::F32, rows(*hidden)) : dd;
        uint32_t dense_post = UINT32_MAX, router_norm = UINT32_MAX, route_scaled = UINT32_MAX;
        uint32_t route_normalized = UINT32_MAX, scores = UINT32_MAX, ids = UINT32_MAX;
        uint32_t weights = UINT32_MAX, expert_norm = UINT32_MAX, expert_up = UINT32_MAX;
        uint32_t expert_gate = UINT32_MAX, expert_value = UINT32_MAX, expert_act = UINT32_MAX;
        uint32_t expert_down = UINT32_MAX, reduced = UINT32_MAX, moe_post = UINT32_MAX;
        uint32_t branch = UINT32_MAX, output_post = UINT32_MAX, next_current = UINT32_MAX;
        if (moe_source) {
            dense_post = add_value(model, ScalarType::F32, rows(*hidden));
            router_norm = add_value(model, ScalarType::F32, rows(*hidden));
            route_scaled = add_value(model, ScalarType::F32, rows(*hidden));
            route_normalized = add_value(model, ScalarType::F32, rows(*hidden));
            scores = add_value(model, ScalarType::F32, rows(*experts));
            ids = add_value(model, ScalarType::U32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, *selected}});
            weights = add_value(model, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, *selected}});
            expert_norm = add_value(model, ScalarType::F32, rows(*hidden));
            expert_up = add_value(model, ScalarType::F32, routed(*selected, 2 * *expert_inter));
            expert_gate = add_value(model, ScalarType::F32, routed(*selected, *expert_inter));
            expert_value = add_value(model, ScalarType::F32, routed(*selected, *expert_inter));
            expert_act = add_value(model, ScalarType::F32, routed(*selected, *expert_inter));
            expert_down = add_value(model, ScalarType::F32, routed(*selected, *hidden));
            reduced = add_value(model, ScalarType::F32, rows(*hidden));
            moe_post = add_value(model, ScalarType::F32, rows(*hidden));
            branch = add_value(model, ScalarType::F32, rows(*hidden));
            output_post = add_value(model, ScalarType::F32, rows(*hidden));
            next_current = add_value(model, ScalarType::F32, rows(*hidden));
        }
        const uint32_t layer_out = add_value(model, ScalarType::F32, rows(*hidden));

        add_operator(model, OperatorKind::RmsNorm, {current}, {normed}, {r.attn_norm}, {}, RmsNormPayload{norm_bits, -1, 1});
        // Q/K/V projections carry their bias tensor when the source has one;
        // has_bias keeps the contract explicit in the operator payload.
        if (r.q_bias != UINT32_MAX)
            add_operator(model, OperatorKind::Linear, {normed}, {qv}, {r.q, r.q_bias}, {},
                         LinearPayload{false, true, ScalarType::F32});
        else
            add_operator(model, OperatorKind::Linear, {normed}, {qv}, {r.q}, {}, LinearPayload{});
        if (r.k_bias != UINT32_MAX)
            add_operator(model, OperatorKind::Linear, {normed}, {kv}, {r.k, r.k_bias}, {},
                         LinearPayload{false, true, ScalarType::F32});
        else
            add_operator(model, OperatorKind::Linear, {normed}, {kv}, {r.k}, {}, LinearPayload{});
        if (r.v_bias != UINT32_MAX)
            add_operator(model, OperatorKind::Linear, {normed}, {vv}, {r.v, r.v_bias}, {},
                         LinearPayload{false, true, ScalarType::F32});
        else
            add_operator(model, OperatorKind::Linear, {normed}, {vv}, {r.v}, {}, LinearPayload{});
        if (r.q_norm != UINT32_MAX)
            add_operator(model, OperatorKind::RmsNorm, {qv}, {qn}, {r.q_norm}, {}, RmsNormPayload{norm_bits, -1, 1});
        if (r.k_norm != UINT32_MAX)
            add_operator(model, OperatorKind::RmsNorm, {kv}, {kn}, {r.k_norm}, {}, RmsNormPayload{norm_bits, -1, 1});
        const bool windowed = layer < layer_windowed.size() && layer_windowed[layer];
        const float layer_rope_base = windowed ? sliding_rope_base : *rope_base;
        const float layer_rope_scale = windowed ? 1.0f : global_rope_scale;
        const uint32_t layer_rope_dim =
            (windowed && sliding_rope_dim && *sliding_rope_dim % 2 == 0 &&
             *sliding_rope_dim <= layer_head_dim)
                ? *sliding_rope_dim : layer_head_dim;
        add_operator(model, OperatorKind::Rope, {qn, kn}, {qr, ar}, {}, {},
                     RopePayload{RopePairing::HalfSplit, true, layer_rope_dim,
                                 std::bit_cast<uint32_t>(layer_rope_base),
                                 std::bit_cast<uint32_t>(layer_rope_scale)});
        const float layer_attention_scale = declared_attention_scale.value_or(
            1.0f / std::sqrt(static_cast<float>(layer_head_dim)));
        CausalAttentionPayload attention{q_heads, layer_kv_heads, layer_head_dim,
                                         std::bit_cast<uint32_t>(layer_attention_scale), AttentionMask::Causal,
                                         CachePolicy::Global,
                                         windowed ? AttentionWindowKind::Sliding : AttentionWindowKind::Global,
                                         windowed && sliding_window ? *sliding_window : 0,
                                         ValueSource::SeparateProjection, vv};
        add_operator(model, OperatorKind::CausalAttention, {qr, ar, vv}, {ao}, {},
                     {key_state_id, key_state_id + 1}, attention);
        add_operator(model, OperatorKind::Linear, {ao}, {attn_proj}, {r.attn_out}, {}, LinearPayload{});
        add_operator(model, OperatorKind::Add, {current, attn_proj}, {attn_res}, {}, {}, AddPayload{});
        if (r.post_attn != UINT32_MAX)
            add_operator(model, OperatorKind::RmsNorm, {attn_res}, {normed2}, {r.post_attn}, {},
                         RmsNormPayload{norm_bits, -1, 1});
        add_operator(model, OperatorKind::RmsNorm, {normed2}, {fn}, {r.ffn_norm}, {}, RmsNormPayload{norm_bits, -1, 1});
        add_operator(model, OperatorKind::Linear, {fn}, {dg}, {r.gate}, {}, LinearPayload{});
        add_operator(model, OperatorKind::Linear, {fn}, {du}, {r.up}, {}, LinearPayload{});
        add_operator(model, OperatorKind::SwiGlu, {dg, du}, {dh}, {}, {},
                     SwiGluPayload{activation});
        add_operator(model, OperatorKind::Linear, {dh}, {dd}, {r.down}, {}, LinearPayload{});
        if (r.post_ffw != UINT32_MAX)
            add_operator(model, OperatorKind::RmsNorm, {dd}, {dd_out}, {r.post_ffw}, {},
                         RmsNormPayload{norm_bits, -1, 1});
        if (moe_source) {
        add_operator(model, OperatorKind::RmsNorm, {dd}, {dense_post}, {r.post_dense}, {}, RmsNormPayload{norm_bits, -1, 1});
        add_operator(model, OperatorKind::RmsNorm, {attn_res}, {router_norm}, {}, {}, RmsNormPayload{norm_bits, -1, 0});
        add_operator(model, OperatorKind::Scale, {router_norm}, {route_scaled}, {r.route_scale}, {}, ScalePayload{ScaleSource::Tensor, 0});
        add_operator(model, OperatorKind::Scale, {route_scaled}, {route_normalized}, {}, {}, ScalePayload{ScaleSource::LiteralF32, std::bit_cast<uint32_t>(1.0f / std::sqrt(static_cast<float>(*hidden)))});
        add_operator(model, OperatorKind::Linear, {route_normalized}, {scores}, {r.router}, {}, LinearPayload{});
        RouterTopKPayload router{*experts, *selected, RouterScoreDomain::Logits,
                                 RouterNormalizationOrder::NormalizeThenSelect,
                                 SelectedWeightNormalization::RenormalizeSelectedProbabilities,
                                 RouterTiePolicy::LowestExpertId, RouterWeightSource::SelectedNormalizedScore, 0};
        add_operator(model, OperatorKind::RouterTopK, {scores}, {ids, weights}, {}, {}, router);
        add_operator(model, OperatorKind::RmsNorm, {attn_res}, {expert_norm}, {r.expert_norm}, {}, RmsNormPayload{norm_bits, -1, 1});
        add_operator(model, OperatorKind::RoutedLinear, {expert_norm, ids, weights}, {expert_up}, {r.expert_up}, {}, RoutedLinearPayload{ScalarType::F32});
        add_operator(model, OperatorKind::AxisSplit, {expert_up}, {expert_gate, expert_value}, {}, {}, AxisSplitPayload{*expert_inter, *expert_inter});
        add_operator(model, OperatorKind::GatedActivation, {expert_gate, expert_value}, {expert_act}, {}, {}, GatedActivationPayload{ActivationKind::GeluTanh});
        add_operator(model, OperatorKind::RoutedLinear, {expert_act, ids, weights}, {expert_down}, {r.expert_down}, {}, RoutedLinearPayload{ScalarType::F32});
        add_operator(model, OperatorKind::WeightedExpertReduce, {expert_down, ids, weights}, {reduced}, {r.reduce}, {}, WeightedExpertReducePayload{ExpertReduceAssociation::SelectedOrderLeftToRight, ExpertScaleSource::PerExpertTensor, ScalarType::F32});
        add_operator(model, OperatorKind::RmsNorm, {reduced}, {moe_post}, {r.post_moe}, {}, RmsNormPayload{norm_bits, -1, 1});
        add_operator(model, OperatorKind::Add, {dense_post, moe_post}, {branch}, {}, {}, AddPayload{});
        add_operator(model, OperatorKind::RmsNorm, {branch}, {output_post}, {r.post_output}, {}, RmsNormPayload{norm_bits, -1, 1});
        add_operator(model, OperatorKind::Add, {attn_res, output_post}, {layer_out}, {}, {}, AddPayload{});
        add_operator(model, OperatorKind::Scale, {layer_out}, {next_current}, {r.output_scale}, {}, ScalePayload{ScaleSource::Tensor, 0});
        current = next_current;
        } else {
            // Plain dense pre-norm block: the FFN output joins the attention
            // residual directly, with no post-norm or routed path.
            add_operator(model, OperatorKind::Add, {attn_res, dd_out}, {layer_out}, {}, {}, AddPayload{});
            current = layer_out;
        }
        model.layers.push_back({layer, first_operator,
                                static_cast<uint32_t>(model.operators.size()) - first_operator, 0});
    }
    const uint32_t final_norm = find_tensor(model, 0, TensorRole::FinalNormWeight);
    const uint32_t output_weight = find_tensor(model, 0, TensorRole::OutputWeight);
    if (final_norm == UINT32_MAX || output_weight == UINT32_MAX) return std::nullopt;
    const uint32_t final = add_value(model, ScalarType::F32, rows(*hidden));
    const uint32_t logits = add_value(model, ScalarType::F32, rows(model.vocabulary_size));
    add_operator(model, OperatorKind::RmsNorm, {current}, {final}, {final_norm}, {}, RmsNormPayload{norm_bits, -1, 1});
    add_operator(model, OperatorKind::Linear, {final}, {logits}, {output_weight}, {}, LinearPayload{});
    model.output_values_first = logits; model.output_values_count = 1;

    // The program defines tensor activity: a tensor no operator references
    // is inactive-program, and a referenced tensor is active. This replaces
    // role-based pre-guessing, so optional tensors (q/k norms, sandwich
    // norms, the rope table) are consumed or parked by the graph itself.
    {
        std::vector<bool> used(model.tensors.size(), false);
        for (const SemanticOperator& operation : model.operators) {
            for (uint32_t tensor_id : operation.tensors) {
                if (tensor_id < used.size()) used[tensor_id] = true;
            }
        }
        for (size_t tensor_id = 0; tensor_id != model.tensors.size(); ++tensor_id) {
            if (!used[tensor_id] &&
                (model.tensors[tensor_id].flags & kSemanticTensorFlagInactiveProgram) == 0) {
                model.tensors[tensor_id].flags |= kSemanticTensorFlagInactiveProgram;
            }
        }
    }
    return model;
}

// Compiles the package's own tokenizer facts into an executable token
// program. The "tokenizer.ggml.model" value is a format-level algorithm
// fact, not a family name: gpt2 selects byte-level BPE, and every
// tokenizer that stores gpt2 facts compiles through this one path.
// Algorithm kinds without an executable mapping keep the token-ids-only
// contract and fail closed at text prompts instead of guessing.
std::optional<TokenProgram> compile_gguf_tokenizer(const GGUFContext& context,
                                                   const SemanticModel& model) {
    const auto& metadata = context.metadata();
    const std::string* algorithm = meta_str(metadata, "tokenizer.ggml.model");
    if (!algorithm) return std::nullopt;
    const bool byte_bpe = *algorithm == "gpt2";
    const bool sentence_piece = *algorithm == "llama";
    // The "tokenizer.ggml.model" value is a format-level algorithm fact,
    // not a family name: gpt2 selects byte-level BPE and llama selects
    // SentencePiece, and every tokenizer storing those facts compiles
    // through one of these two paths. Other kinds fail closed.
    if (!byte_bpe && !sentence_piece) return std::nullopt;
    const auto* tokens = meta_as<MetaArrayStr>(metadata, "tokenizer.ggml.tokens");
    if (!tokens || tokens->empty() || tokens->size() != model.vocabulary_size) {
        return std::nullopt;
    }
    const auto* merges = meta_as<MetaArrayStr>(metadata, "tokenizer.ggml.merges");
    if (byte_bpe && !merges) return std::nullopt;
    const auto* token_types_u32 = meta_as<MetaArrayU32>(metadata, "tokenizer.ggml.token_type");
    const auto* token_types_i32 = meta_as<MetaArrayI32>(metadata, "tokenizer.ggml.token_type");
    if ((token_types_u32 && token_types_u32->size() != tokens->size()) ||
        (token_types_i32 && token_types_i32->size() != tokens->size())) {
        return std::nullopt;
    }
    const auto type_at = [token_types_u32, token_types_i32](size_t id) -> uint32_t {
        if (token_types_u32 && id < token_types_u32->size()) return (*token_types_u32)[id];
        if (token_types_i32 && id < token_types_i32->size() &&
            (*token_types_i32)[id] >= 0) return static_cast<uint32_t>((*token_types_i32)[id]);
        return static_cast<uint32_t>(TokenPieceType::Normal);
    };
    const auto* scores = meta_as<MetaArrayF32>(metadata, "tokenizer.ggml.scores");
    if (scores && scores->size() != tokens->size()) return std::nullopt;

    TokenProgramDefinition definition;
    definition.stop_ids = {model.eos_id};
    if (byte_bpe) definition.model_kind = TokenProgramModelKind::ByteBpe;
    if (sentence_piece) definition.model_kind = TokenProgramModelKind::SentencePiece;

    if (byte_bpe) {
        // GPT-2 byte alphabet: printable bytes stay themselves, the rest
        // continue at 256. This is the tokenizer data's own convention.
        uint32_t next = 256;
        for (int byte = 0; byte != 256; ++byte) {
            const bool printable = (byte >= 0x21 && byte <= 0x7e) ||
                                   (byte >= 0xa1 && byte <= 0xac) ||
                                   (byte >= 0xae && byte <= 0xff);
            definition.byte_to_unicode[static_cast<size_t>(byte)] =
                printable ? static_cast<uint32_t>(byte) : next++;
        }
    }

    definition.vocabulary.reserve(tokens->size());
    std::map<std::string, uint32_t> piece_ids;
    for (size_t id = 0; id != tokens->size(); ++id) {
        VocabEntry entry;
        entry.piece = (*tokens)[id];
        const uint32_t type = type_at(id);
        if (type < static_cast<uint32_t>(TokenPieceType::Normal) ||
            type > static_cast<uint32_t>(TokenPieceType::Byte)) {
            return std::nullopt;
        }
        entry.type = static_cast<uint8_t>(type);
        entry.flags = type == static_cast<uint32_t>(TokenPieceType::Normal)
                          ? 0
                          : static_cast<uint16_t>(VocabFlags::Special);
        // SentencePiece segmentation is scored; byte-level BPE ignores it.
        entry.score = scores ? (*scores)[id] : 0.0f;
        if (!piece_ids.emplace(entry.piece, static_cast<uint32_t>(id)).second) {
            return std::nullopt;
        }
        if (type == static_cast<uint32_t>(TokenPieceType::Unknown))
            definition.unknown_token_id = static_cast<uint32_t>(id);
        if (type == static_cast<uint32_t>(TokenPieceType::Byte))
            definition.byte_fallback = true;
        definition.vocabulary.push_back(std::move(entry));
    }

    if (byte_bpe) {
        definition.merges.reserve(merges->size());
        for (size_t rank = 0; rank != merges->size(); ++rank) {
            const std::string& text = (*merges)[rank];
            const size_t split = text.find(' ');
            if (split == std::string::npos ||
                text.find(' ', split + 1) != std::string::npos) {
                return std::nullopt;
            }
            const std::string left_text = text.substr(0, split);
            const std::string right_text = text.substr(split + 1);
            const auto left = piece_ids.find(left_text);
            const auto right = piece_ids.find(right_text);
            const auto result = piece_ids.find(left_text + right_text);
            if (left == piece_ids.end() || right == piece_ids.end() ||
                result == piece_ids.end()) {
                return std::nullopt;
            }
            definition.merges.push_back({left->second, right->second, result->second,
                                         static_cast<uint32_t>(rank)});
        }
    }

    if (byte_bpe) {
        definition.pretokenizer.kind = PretokenizerKind::ByteLevel;
        // The grouping contract comes from the package's own
        // "tokenizer.ggml.pre" fact. Every spelling without a verified
        // Qwen-style newline grouping keeps the default GPT-2 backtracking
        // rule; extending coverage is adding a spelling to this data list.
        if (const std::string* pre = meta_str(metadata, "tokenizer.ggml.pre");
            pre && *pre == "qwen2") {
            definition.pretokenizer.flags =
                static_cast<uint8_t>(PretokenizerFlags::GroupNewlineRuns);
        }
        definition.decoder.kind = DecoderKind::ByteLevel;
        definition.decoder.flags = static_cast<uint8_t>(DecoderFlags::SkipSpecial);
    } else {
        definition.normalizer.kind = NormalizerKind::SentencePiece;
        // The dummy-prefix contract is the package's own
        // "tokenizer.ggml.add_space_prefix" fact (BOOL or UINT32 per
        // converter); whitespace escaping is the SP default.
        bool dummy_prefix = false;
        if (const auto* flag = meta_as<bool>(metadata, "tokenizer.ggml.add_space_prefix")) {
            dummy_prefix = *flag;
        } else if (const auto* flag_u32 = meta_as<uint32_t>(metadata, "tokenizer.ggml.add_space_prefix")) {
            dummy_prefix = *flag_u32 == 1;
        }
        definition.normalizer.flags =
            static_cast<uint8_t>(SentencePieceNormalizerFlags::EscapeWhitespaces) |
            (dummy_prefix ? static_cast<uint8_t>(SentencePieceNormalizerFlags::AddDummyPrefix) : 0);
        definition.pretokenizer.kind = PretokenizerKind::SentencePiece;
        definition.decoder.kind = DecoderKind::Identity;
    }
    definition.postprocessor.kind = PostprocessorKind::None;
    // GGUF stores this flag as BOOL in some converters and UINT32 in others.
    bool bos_prefix = false;
    if (const auto* flag = meta_as<bool>(metadata, "tokenizer.ggml.add_bos_token")) {
        bos_prefix = *flag;
    } else if (const auto* flag_u32 = meta_as<uint32_t>(metadata, "tokenizer.ggml.add_bos_token")) {
        bos_prefix = *flag_u32 == 1;
    }
    if (bos_prefix && model.bos_id != UINT32_MAX) {
        definition.postprocessor.kind = PostprocessorKind::AddBosEos;
        definition.postprocessor.flags = static_cast<uint8_t>(PostprocessorFlags::AddBos);
        definition.postprocessor.bos_token_id = model.bos_id;
    }
    if (definition.unknown_token_id == kTokenProgramNoTokenId) {
        if (const auto* declared_unknown = meta_as<uint32_t>(metadata, "tokenizer.ggml.unknown_token_id");
            declared_unknown && *declared_unknown < tokens->size() &&
            type_at(*declared_unknown) == static_cast<uint32_t>(TokenPieceType::Unknown)) {
            definition.unknown_token_id = *declared_unknown;
        }
    }
    // The package's own chat template compiles into the first-turn and
    // continuation-turn programs. When the template is missing or outside
    // the recognized conversational shapes, the neutral universal marker
    // stays a newline and no turn program is emitted.
    const std::string* chat_template = meta_str(metadata, "tokenizer.chat_template");
    const ChatFraming framing =
        chat_template ? compile_chat_framing(*chat_template) : ChatFraming{};
    if (framing.matched) {
        std::string system_prefix = framing.system_prefix;
        // The template itself opens with the BOS token: when the package
        // does not declare an automatic BOS prefix, the framing owns it and
        // places the token's own text; otherwise the postprocessor already
        // adds it and the framing stays out of the way.
        if (framing.template_emits_bos && !bos_prefix &&
            model.bos_id != UINT32_MAX && model.bos_id < tokens->size()) {
            system_prefix = (*tokens)[model.bos_id] + system_prefix;
        }
        definition.prompt = {};
        if (!system_prefix.empty())
            definition.prompt.push_back({PromptOpcode::EmitLiteralUtf8, system_prefix});
        definition.prompt.push_back({PromptOpcode::EmitLiteralUtf8, framing.user_open});
        definition.prompt.push_back({PromptOpcode::EmitUserText, {}});
        definition.prompt.push_back({PromptOpcode::EmitLiteralUtf8, framing.turn_close});
        definition.prompt.push_back(
            {PromptOpcode::EmitGenerationPrompt, framing.generation_open});
        definition.prompt.push_back({PromptOpcode::End, {}});
        definition.turn = {
            {PromptOpcode::EmitLiteralUtf8, framing.assistant_close},
            {PromptOpcode::EmitLiteralUtf8, framing.user_open},
            {PromptOpcode::EmitUserText, {}},
            {PromptOpcode::EmitLiteralUtf8, framing.turn_close},
            {PromptOpcode::EmitGenerationPrompt, framing.generation_open},
            {PromptOpcode::End, {}},
        };
    } else {
        definition.prompt = {{PromptOpcode::EmitUserText, {}},
                             {PromptOpcode::EmitGenerationPrompt, "\n"},
                             {PromptOpcode::End, {}}};
    }

    auto serialized = serialize_token_program_v3(definition);
    if (std::get_if<TokenProgramStatus>(&serialized)) return std::nullopt;
    const auto* bytes = std::get_if<std::vector<uint8_t>>(&serialized);
    auto compiled = TokenProgram::compile(*bytes);
    if (std::get_if<TokenProgramStatus>(&compiled)) return std::nullopt;
    const auto* program = std::get_if<TokenProgram>(&compiled);
    if (!program) return std::nullopt;
    return std::move(*program);
}

} // namespace

GgufProductCompilationResult compile_normalized_gguf_product(const PackageView& package) {
    GGUFContext context;
    if (!context.parse(package)) return failure(CompatibilityError::PACKAGE_BAD_MAGIC, "GGUF source could not be parsed");
    auto imported = build_gguf_artifact_index(package);
    if (const auto* report = std::get_if<CompatibilityReport>(&imported)) return *report;
    ArtifactIndex source = std::get<ArtifactIndex>(std::move(imported));
    auto evidence_result = extract_gguf_normalized_source_evidence(source);
    if (const auto* report = std::get_if<CompatibilityReport>(&evidence_result)) return *report;
    NormalizedSourceEvidence evidence = std::get<NormalizedSourceEvidence>(std::move(evidence_result));
    auto physical_result = normalized_physical_index(source, evidence);
    if (const auto* report = std::get_if<CompatibilityReport>(&physical_result)) {
        CompatibilityReport enriched = *report;
        enriched.detail += " tensor=" + std::to_string(report->artifact_index);
        return enriched;
    }
    ArtifactIndex physical = std::get<ArtifactIndex>(std::move(physical_result));
    auto model_result = build_model(physical, evidence);
    if (!model_result) return failure(CompatibilityError::IMPORT_SCHEMA_NOT_FOUND, "typed normalized GGUF facts do not form an executable semantic graph");
    SemanticModel model = std::move(*model_result);
    auto token_program = compile_gguf_tokenizer(context, model);
    TokenContract contract;
    if (token_program) {
        // The tokenizer is compiled from the package's own metadata span
        // (header plus every key/value, which is where the tokenizer
        // arrays live), so the contract binds to that exact byte range
        // instead of re-hashing the whole file.
        const uint64_t span = context.data_section_offset();
        Sha256Digest span_digest{};
        {
            CC_SHA256_CTX digest_context;
            CC_SHA256_Init(&digest_context);
            size_t offset = 0;
            while (offset != span) {
                const size_t chunk = std::min<size_t>(
                    span - offset, std::numeric_limits<CC_LONG>::max());
                CC_SHA256_Update(&digest_context,
                                 package.bytes().data() + offset,
                                 static_cast<CC_LONG>(chunk));
                offset += chunk;
            }
            CC_SHA256_Final(span_digest.bytes.data(), &digest_context);
        }
        model.tokenizer_digest = span_digest.bytes;
        model.template_digest = token_program->prompt_digest().bytes;
        contract.tokenizer_algorithm =
            token_program->definition().model_kind == TokenProgramModelKind::SentencePiece
                ? TokenizerAlgorithm::SentencePiece
                : TokenizerAlgorithm::ByteBpe;
        contract.tokenizer_version = token_program->wire_major_version();
        contract.vocabulary_digest = token_program->vocabulary_digest();
        contract.tokenizer_data = {physical.artifacts().front().artifact_id(),
                                   0, span, span_digest};
        contract.authoritative_tokenizer_digest = span_digest;
        contract.authoritative_template_digest = token_program->prompt_digest();
        contract.stop_ids = model.stop_ids;
        PromptTemplate prompt;
        prompt.version = 1;
        for (const auto& instruction : token_program->definition().prompt) {
            if (instruction.opcode == PromptOpcode::End) break;
            const auto kind = instruction.opcode == PromptOpcode::EmitUserText
                ? PromptOperationKind::AppendInputText : PromptOperationKind::AppendLiteral;
            prompt.operations.push_back({kind,
                {instruction.literal.begin(), instruction.literal.end()}, kNoTokenId});
        }
        contract.prompt = std::move(prompt);
    } else {
        contract.tokenizer_algorithm = TokenizerAlgorithm::TokenIdsOnly;
        contract.tokenizer_version = 0;
        contract.vocabulary_digest = package.digest();
        contract.authoritative_tokenizer_digest.bytes = model.tokenizer_digest;
        contract.authoritative_template_digest.bytes = model.template_digest;
        contract.stop_ids = model.stop_ids;
    }
    contract.vocabulary_size = model.vocabulary_size;
    contract.bos_id = model.bos_id; contract.eos_id = model.eos_id;
    auto registry_result = build_codec_registry(model);
    if (const auto* report = std::get_if<CompatibilityReport>(&registry_result))
        return *report;
    PhysicalCodecRegistry registry =
        std::get<PhysicalCodecRegistry>(std::move(registry_result));
    auto manifest_result = SemanticManifest::build(physical, model, contract, registry);
    if (const auto* report = std::get_if<CompatibilityReport>(&manifest_result))
        return *report;
    return GgufProductCompilation{std::get<SemanticManifest>(std::move(manifest_result)),
                                  token_program
                                      ? std::move(*token_program)
                                      : TokenProgram::token_ids_only(model.vocabulary_size)};
}

} // namespace Laplace
