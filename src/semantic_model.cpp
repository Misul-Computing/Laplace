#include "semantic_model.h"

#include <CommonCrypto/CommonDigest.h>

#include <cmath>
#include <cstddef>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>

namespace Laplace {

namespace {

float fp32(float value) {
    volatile float rounded = value;
    return rounded;
}

float fp_add(float left, float right) {
    return fp32(fp32(left) + fp32(right));
}

float fp_mul(float left, float right) {
    return fp32(fp32(left) * fp32(right));
}

CompatibilityReport shape_error() {
    return package_report(CompatibilityError::IR_SHAPE_MISMATCH);
}

CompatibilityReport constraint_error() {
    return package_report(CompatibilityError::IR_CONSTRAINT_FAILED);
}

CompatibilityReport runtime_error() {
    return package_report(CompatibilityError::RUNTIME_INPUT_INVALID);
}

bool finite(const std::vector<float>& values) {
    for (float value : values) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

float sigmoid(float value) {
    if (value >= 0.0f) return fp32(1.0f / fp_add(1.0f, fp32(std::exp(-value))));
    float exponent = fp32(std::exp(value));
    return fp32(exponent / fp_add(1.0f, exponent));
}

bool product(uint32_t left, uint32_t right, size_t& result) {
    if (left == 0 || right == 0 || left > std::numeric_limits<size_t>::max() / right) return false;
    result = static_cast<size_t>(left) * right;
    return true;
}

void append_u16(std::vector<uint8_t>& output, uint16_t value) {
    output.push_back(static_cast<uint8_t>(value));
    output.push_back(static_cast<uint8_t>(value >> 8));
}

void append_u32(std::vector<uint8_t>& output, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8) output.push_back(static_cast<uint8_t>(value >> shift));
}

void append_u64(std::vector<uint8_t>& output, uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8) output.push_back(static_cast<uint8_t>(value >> shift));
}

bool read_u16(const std::vector<uint8_t>& bytes, size_t& offset, uint16_t& value) {
    if (bytes.size() - offset < 2) return false;
    value = static_cast<uint16_t>(bytes[offset]) | (static_cast<uint16_t>(bytes[offset + 1]) << 8);
    offset += 2;
    return true;
}

bool read_u32(const std::vector<uint8_t>& bytes, size_t& offset, uint32_t& value) {
    if (bytes.size() - offset < 4) return false;
    value = 0;
    for (unsigned shift = 0; shift != 32; shift += 8) value |= static_cast<uint32_t>(bytes[offset++]) << shift;
    return true;
}

bool read_u64(const std::vector<uint8_t>& bytes, size_t& offset, uint64_t& value) {
    if (bytes.size() - offset < 8) return false;
    value = 0;
    for (unsigned shift = 0; shift != 64; shift += 8) value |= static_cast<uint64_t>(bytes[offset++]) << shift;
    return true;
}

std::array<uint8_t, 32> sha256(const uint8_t* bytes, size_t size) {
    std::array<uint8_t, 32> digest{};
    CC_SHA256(bytes, static_cast<CC_LONG>(size), digest.data());
    return digest;
}

bool valid_token_id(uint32_t id, uint32_t vocabulary) {
    return id == UINT32_MAX || id < vocabulary;
}

CompatibilityReport wire_error(CompatibilityError code) {
    return package_report(code);
}

constexpr uint32_t kMaxTensors = 65536;
constexpr uint32_t kMaxValues = 262144;
constexpr uint32_t kMaxOperators = 1048576;
constexpr uint32_t kMaxStates = 4096;
constexpr uint32_t kMaxConstraints = 65536;
constexpr uint32_t kMaxVector = 1048576;
constexpr uint8_t kMaxRank = 8;
constexpr uint8_t kMaxPlanes = 16;

void append_u8(std::vector<uint8_t>& output, uint8_t value) {
    output.push_back(value);
}

bool read_u8(const std::vector<uint8_t>& bytes, size_t& offset, uint8_t& value) {
    if (offset == bytes.size()) return false;
    value = bytes[offset++];
    return true;
}

void append_zeros(std::vector<uint8_t>& output, size_t count) {
    output.insert(output.end(), count, 0);
}

void append_padding(std::vector<uint8_t>& output) {
    while (output.size() % 8 != 0) output.push_back(0);
}

bool consume_padding(const std::vector<uint8_t>& bytes, size_t& offset) {
    while (offset % 8 != 0) {
        if (offset == bytes.size() || bytes[offset++] != 0) return false;
    }
    return true;
}

template <typename E>
bool enum_between(E value, uint16_t first, uint16_t last) {
    uint16_t raw = static_cast<uint16_t>(value);
    return raw >= first && raw <= last;
}

bool valid_scalar_type(ScalarType type, bool allow_zero = false) {
    uint16_t raw = static_cast<uint16_t>(type);
    return (allow_zero && raw == 0) || (raw >= 1 && raw <= 5);
}

bool valid_dimension(const Dimension& dimension) {
    return (dimension.kind == DimensionKind::Constant || dimension.kind == DimensionKind::Symbol) &&
           dimension.constant_or_symbol != 0;
}

bool valid_dimensions(const std::vector<Dimension>& dimensions) {
    if (dimensions.size() > kMaxRank) return false;
    for (const Dimension& dimension : dimensions) {
        if (!valid_dimension(dimension)) return false;
    }
    return true;
}

bool valid_layout(const PhysicalLayout& layout, uint8_t rank) {
    if (!enum_between(layout.kind, 1, 4) || layout.version != 1 ||
        !enum_between(layout.packing, 0, 2) || layout.rank != rank || layout.block_rank > rank ||
        layout.flags != 0) {
        return false;
    }
    std::array<bool, 8> seen{};
    for (uint8_t index = 0; index != 8; ++index) {
        uint8_t axis = layout.axis_order[index];
        if (index < rank) {
            if (axis >= rank || seen[axis]) return false;
            seen[axis] = true;
        } else if (axis != 0xff || layout.strides[index] != 0) {
            return false;
        }
    }
    if (layout.kind == PhysicalLayoutKind::ContiguousRowMajor) {
        return layout.packing == PackingKind::None && layout.block_rank == 0 &&
               layout.block_elements == 0 && layout.block_bytes == 0;
    }
    if (layout.kind == PhysicalLayoutKind::GgufBlocked) {
        return layout.packing == PackingKind::Gguf && layout.block_elements != 0 &&
               layout.block_bytes != 0;
    }
    if (layout.kind == PhysicalLayoutKind::GroupedAffine) {
        return layout.packing == PackingKind::LsbBitPacked && layout.block_rank == 1 &&
               layout.block_elements != 0 && layout.block_bytes != 0;
    }
    if (layout.kind == PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip) {
        return layout.packing == PackingKind::LsbBitPacked && layout.block_rank == 1 &&
               layout.block_elements == 256 && layout.block_bytes == 64;
    }
    return false;
}

bool valid_quantization(const Quantization& quantization) {
    if (!enum_between(quantization.kind, 0, 1) || quantization.version != 1 ||
        !valid_scalar_type(quantization.accumulation_type) ||
        !valid_scalar_type(quantization.scale_type, true) ||
        !valid_scalar_type(quantization.zero_type, true) ||
        !valid_scalar_type(quantization.bias_type, true) || quantization.flags != 0) {
        return false;
    }
    if (quantization.kind == QuantizationKind::None) {
        return quantization.scale_type == static_cast<ScalarType>(0) &&
               quantization.zero_type == static_cast<ScalarType>(0) &&
               quantization.bias_type == static_cast<ScalarType>(0) &&
               quantization.block_elements == 0 && quantization.block_bytes == 0 &&
               quantization.group_size == 0 && quantization.required_plane_mask == 0;
    }
    return quantization.block_elements != 0 && quantization.block_bytes != 0 &&
           quantization.group_size != 0 && quantization.required_plane_mask != 0;
}

bool valid_expert_axis(const ExpertAxis& expert_axis, const SemanticTensor& tensor) {
    if (expert_axis.kind == ExpertAxisKind::None) {
        return expert_axis.expert_axis == 0xff && expert_axis.member_axis == 0xff &&
               expert_axis.input_axis == 0xff && expert_axis.output_axis == 0xff &&
               expert_axis.expert_count == 0 && expert_axis.per_expert_byte_stride == 0 &&
               expert_axis.flags == 0;
    }
    const uint8_t rank = static_cast<uint8_t>(tensor.dimensions.size());
    if (expert_axis.kind != ExpertAxisKind::ExpertBank || expert_axis.flags != 0 ||
        expert_axis.expert_axis >= rank || expert_axis.input_axis >= rank || expert_axis.output_axis >= rank ||
        expert_axis.input_axis == expert_axis.expert_axis || expert_axis.output_axis == expert_axis.expert_axis ||
        expert_axis.input_axis == expert_axis.output_axis || expert_axis.expert_count == 0 ||
        expert_axis.per_expert_byte_stride == 0 || tensor.dimensions[expert_axis.expert_axis].kind != DimensionKind::Constant ||
        tensor.dimensions[expert_axis.expert_axis].constant_or_symbol != expert_axis.expert_count) {
        return false;
    }
    if (expert_axis.member_axis != 0xff &&
        (expert_axis.member_axis >= rank || expert_axis.member_axis == expert_axis.expert_axis ||
         expert_axis.member_axis == expert_axis.input_axis || expert_axis.member_axis == expert_axis.output_axis)) {
        return false;
    }
    const auto values = std::find_if(tensor.planes.begin(), tensor.planes.end(), [](const TensorPlane& plane) {
        return plane.kind == PlaneKind::Values;
    });
    return values != tensor.planes.end() &&
           expert_axis.per_expert_byte_stride <= std::numeric_limits<uint64_t>::max() / expert_axis.expert_count &&
           values->length == expert_axis.per_expert_byte_stride * expert_axis.expert_count;
}

bool valid_plane(const TensorPlane& plane) {
    if (!enum_between(plane.kind, 1, 6) || !valid_scalar_type(plane.storage_type) ||
        plane.artifact_id.value != 0 || plane.length == 0 || plane.alignment == 0 ||
        (plane.alignment & (plane.alignment - 1)) != 0 || plane.offset % plane.alignment != 0 ||
        (plane.flags & ~1u) != 0) {
        return false;
    }
    return plane.offset <= std::numeric_limits<uint64_t>::max() - plane.length;
}

bool valid_tensor(const SemanticTensor& tensor, bool supports_expert_axis,
                  bool supports_inactive_program) {
    const uint16_t allowed_flags = supports_inactive_program
        ? kSemanticTensorFlagInactiveProgram : 0;
    if (!enum_between(tensor.role, 1, 31) || !valid_scalar_type(tensor.logical_type) ||
        !valid_dimensions(tensor.dimensions) || tensor.dimensions.empty() || tensor.planes.empty() ||
        tensor.planes.size() > kMaxPlanes || (tensor.flags & ~allowed_flags) != 0 ||
        !valid_layout(tensor.layout, static_cast<uint8_t>(tensor.dimensions.size())) ||
        !valid_quantization(tensor.quantization)) {
        return false;
    }
    uint32_t seen_mask = 0;
    for (const TensorPlane& plane : tensor.planes) {
        uint32_t bit = 1u << (static_cast<uint16_t>(plane.kind) - 1);
        if (!valid_plane(plane) || (seen_mask & bit) != 0) return false;
        seen_mask |= bit;
    }
    if (!supports_expert_axis && tensor.expert_axis != ExpertAxis{}) return false;
    if (supports_expert_axis && !valid_expert_axis(tensor.expert_axis, tensor)) return false;
    return tensor.quantization.kind == QuantizationKind::None
               ? (seen_mask == 1)
               : ((seen_mask & tensor.quantization.required_plane_mask) == tensor.quantization.required_plane_mask);
}

bool f32_bits(uint32_t bits, float& value) {
    std::memcpy(&value, &bits, sizeof(value));
    return std::isfinite(value);
}

bool valid_router_payload(const RouterTopKPayload& payload) {
    if (payload.expert_count == 0 || payload.selected_count == 0 || payload.selected_count > payload.expert_count ||
        payload.tie_policy != RouterTiePolicy::LowestExpertId ||
        payload.weight_source != RouterWeightSource::SelectedNormalizedScore || payload.flags != 0) {
        return false;
    }
    if (payload.normalization_order == RouterNormalizationOrder::SelectThenNormalize) {
        return payload.score_domain == RouterScoreDomain::Logits &&
               payload.selected_weight_normalization == SelectedWeightNormalization::Softmax;
    }
    if (payload.normalization_order != RouterNormalizationOrder::NormalizeThenSelect) return false;
    if (payload.selected_weight_normalization == SelectedWeightNormalization::PreserveSource) {
        return payload.score_domain == RouterScoreDomain::Logits ||
               payload.score_domain == RouterScoreDomain::Probabilities;
    }
    return payload.score_domain == RouterScoreDomain::Logits &&
           payload.selected_weight_normalization ==
               SelectedWeightNormalization::RenormalizeSelectedProbabilities;
}

bool valid_payload(const SemanticOperator& op) {
    if (op.inputs.size() > 64 || op.outputs.size() > 64 || op.tensors.size() > 64 || op.states.size() > 64) {
        return false;
    }
    if (!semantic_operator_signature_valid(op)) return false;
    if (op.semantic_version >= 2 && op.semantic_version <= 7) {
        switch (op.kind) {
        case OperatorKind::EmbeddingLookup: {
            const auto* payload = std::get_if<EmbeddingLookupPayload>(&op.payload);
            float scale = 0.0f;
            return payload && f32_bits(payload->scale_f32_bits, scale) && scale > 0.0f &&
                   payload->vocabulary != 0 && payload->width != 0 && payload->flags == 0 &&
                   op.inputs.size() == 1 && op.outputs.size() == 1;
        }
        case OperatorKind::RmsNorm: {
            const auto* payload = std::get_if<RmsNormPayload>(&op.payload);
            float epsilon = 0.0f;
            if (!payload || payload->axis != -1 ||
                !f32_bits(payload->epsilon_f32_bits, epsilon) || epsilon < 0.0f ||
                op.inputs.size() != 1 || op.outputs.size() != 1 || !op.states.empty()) {
                return false;
            }
            if (payload->weight_mode == 1) return op.tensors.size() == 1;
            return op.semantic_version == 7 && payload->weight_mode == 0 &&
                   op.tensors.empty();
        }
        case OperatorKind::Linear: {
            const auto* payload = std::get_if<LinearPayload>(&op.payload);
            return payload && payload->accumulation_type == ScalarType::F32 &&
                   op.inputs.size() == 1 && op.outputs.size() == 1;
        }
        case OperatorKind::Rope: {
            const auto* payload = std::get_if<RopePayload>(&op.payload);
            float base = 0.0f;
            float scale = 0.0f;
            if (!payload || !payload->position_from_cursor || payload->rotary_dimension == 0 ||
                payload->rotary_dimension % 2 != 0 ||
                !f32_bits(payload->base_f32_bits, base) || base <= 0.0f ||
                !f32_bits(payload->scale_f32_bits, scale) || scale <= 0.0f) {
                return false;
            }
            const uint32_t frequency_dimension = payload->frequency_dimension == 0
                ? payload->rotary_dimension : payload->frequency_dimension;
            if ((op.semantic_version < 7 && payload->frequency_dimension != 0) ||
                frequency_dimension < payload->rotary_dimension ||
                frequency_dimension % 2 != 0) {
                return false;
            }
            if (op.semantic_version < 5) {
                return (payload->pairing == RopePairing::HalfSplit || payload->pairing == RopePairing::Interleaved) &&
                       payload->position_sections == std::array<uint32_t, 4>{};
            }
            if (payload->pairing == RopePairing::HalfSplit || payload->pairing == RopePairing::Interleaved) {
                return payload->position_sections == std::array<uint32_t, 4>{};
            }
            uint64_t total = 0;
            for (uint32_t section : payload->position_sections) total += section;
            return payload->pairing == RopePairing::MultiSectionHalfSplit &&
                   payload->position_sections[0] != 0 && payload->position_sections[1] != 0 &&
                   total == payload->rotary_dimension / 2;
        }
        case OperatorKind::CausalAttention: {
            const auto* payload = std::get_if<CausalAttentionPayload>(&op.payload);
            float scale = 0.0f;
            if (!payload || payload->query_heads == 0 || payload->kv_heads == 0 ||
                payload->head_dimension == 0 || payload->query_heads % payload->kv_heads != 0 ||
                !f32_bits(payload->scale_f32_bits, scale) || scale <= 0.0f ||
                payload->mask != AttentionMask::Causal || payload->cache_policy != CachePolicy::Global) {
                return false;
            }
            if (op.semantic_version < 7) {
                return payload->window == AttentionWindowKind::Global && payload->window_tokens == 0 &&
                       payload->value_source == ValueSource::SeparateProjection &&
                       payload->value_source_value == UINT32_MAX;
            }
            return ((payload->window == AttentionWindowKind::Global && payload->window_tokens == 0) ||
                    (payload->window == AttentionWindowKind::Sliding && payload->window_tokens != 0)) &&
                   enum_between(payload->value_source, 1, 4) && payload->value_source_value != UINT32_MAX;
        }
        case OperatorKind::SwiGlu: {
            const auto* payload = std::get_if<SwiGluPayload>(&op.payload);
            return payload && payload->activation == ActivationKind::Silu;
        }
        case OperatorKind::Add:
            return std::holds_alternative<AddPayload>(op.payload);
        case OperatorKind::DepthwiseConvSilu: {
            const auto* payload = std::get_if<DepthwiseConvSiluPayload>(&op.payload);
            return payload && payload->qk_heads != 0 && payload->value_heads != 0 &&
                   payload->value_heads % payload->qk_heads == 0 && payload->head_dimension != 0 &&
                   payload->kernel >= 2 && op.inputs.size() == 1 && op.outputs.size() == 3 &&
                   op.tensors.size() == 1 && op.states.size() == 1;
        }
        case OperatorKind::GatedDeltaNet: {
            const auto* payload = std::get_if<GatedDeltaNetPayload>(&op.payload);
            return payload && payload->qk_heads != 0 && payload->value_heads != 0 &&
                   payload->value_heads % payload->qk_heads == 0 && payload->head_dimension != 0 &&
                   payload->qk_mapping == QkHeadMapping::ValueHeadModulo &&
                   payload->beta_transform == BetaTransform::Sigmoid &&
                   payload->decay_transform == DecayTransform::NegativeSoftplus &&
                   (payload->state_layout == DeltaStateLayout::ValueHeadOutputRowKeyColumn ||
                    payload->state_layout == DeltaStateLayout::ValueHeadKeyRowOutputColumn) && payload->flags == 0 &&
                   op.inputs.size() == 5 && op.outputs.size() == 1 && op.tensors.size() == 2 && op.states.size() == 1;
        }
        case OperatorKind::GatedAttention:
            return std::holds_alternative<GatedAttentionPayload>(op.payload) && op.inputs.size() == 2 &&
                   op.outputs.size() == 1 && op.tensors.empty() && op.states.empty();
        case OperatorKind::GatedRmsNorm: {
            const auto* payload = std::get_if<GatedRmsNormPayload>(&op.payload);
            float epsilon = 0.0f;
            return payload && f32_bits(payload->epsilon_f32_bits, epsilon) && epsilon >= 0.0f &&
                   payload->gate_activation == ActivationKind::Silu && payload->weight_mode == 1 &&
                   op.inputs.size() == 2 && op.outputs.size() == 1 && op.tensors.size() == 1 && op.states.empty();
        }
        case OperatorKind::L2Normalize: {
            const auto* payload = std::get_if<L2NormalizePayload>(&op.payload);
            float epsilon = 0.0f;
            return payload && f32_bits(payload->epsilon_f32_bits, epsilon) && epsilon >= 0.0f &&
                   op.inputs.size() == 1 && op.outputs.size() == 1 && op.tensors.empty() && op.states.empty();
        }
        case OperatorKind::AxisSplit: {
            const auto* payload = std::get_if<AxisSplitPayload>(&op.payload);
            return op.semantic_version >= 4 && payload &&
                   payload->first_width != 0 && payload->second_width != 0 &&
                   op.inputs.size() == 1 && op.outputs.size() == 2 && op.tensors.empty() && op.states.empty();
        }
        case OperatorKind::Concat: {
            const auto* payload = std::get_if<ConcatPayload>(&op.payload);
            return op.semantic_version >= 6 && payload && payload->axis == -1 &&
                   op.inputs.size() == 2 && op.outputs.size() == 1 && op.tensors.empty() && op.states.empty();
        }
        case OperatorKind::RouterTopK:
            return op.semantic_version == 7 && std::get_if<RouterTopKPayload>(&op.payload) &&
                   valid_router_payload(std::get<RouterTopKPayload>(op.payload)) && op.inputs.size() == 1 &&
                   op.outputs.size() == 2 && op.tensors.empty() && op.states.empty();
        case OperatorKind::RoutedLinear: {
            const auto* payload = std::get_if<RoutedLinearPayload>(&op.payload);
            return op.semantic_version == 7 && payload && payload->accumulation_type == ScalarType::F32 &&
                   op.inputs.size() == 3 && op.outputs.size() == 1 && op.tensors.size() == 1 && op.states.empty();
        }
        case OperatorKind::GatedActivation: {
            const auto* payload = std::get_if<GatedActivationPayload>(&op.payload);
            return op.semantic_version == 7 && payload &&
                   (payload->activation == ActivationKind::Silu || payload->activation == ActivationKind::GeluTanh) &&
                   op.inputs.size() == 2 && op.outputs.size() == 1 && op.tensors.empty() && op.states.empty();
        }
        case OperatorKind::WeightedExpertReduce: {
            const auto* payload = std::get_if<WeightedExpertReducePayload>(&op.payload);
            return op.semantic_version == 7 && payload &&
                   payload->association == ExpertReduceAssociation::SelectedOrderLeftToRight &&
                   (payload->scale_source == ExpertScaleSource::None || payload->scale_source == ExpertScaleSource::PerExpertTensor) &&
                   payload->accumulation_type == ScalarType::F32 && op.inputs.size() == 3 && op.outputs.size() == 1 &&
                   op.tensors.size() == (payload->scale_source == ExpertScaleSource::PerExpertTensor ? 1u : 0u) && op.states.empty();
        }
        case OperatorKind::Scale: {
            const auto* payload = std::get_if<ScalePayload>(&op.payload);
            float literal = 0.0f;
            return op.semantic_version == 7 && payload && op.inputs.size() == 1 && op.outputs.size() == 1 && op.states.empty() &&
                   ((payload->source == ScaleSource::LiteralF32 && f32_bits(payload->literal_f32_bits, literal) && op.tensors.empty()) ||
                    (payload->source == ScaleSource::Tensor && payload->literal_f32_bits == 0 && op.tensors.size() == 1));
        }
        case OperatorKind::TanhSoftcap: {
            const auto* payload = std::get_if<TanhSoftcapPayload>(&op.payload);
            float cap = 0.0f;
            return op.semantic_version == 7 && payload && f32_bits(payload->cap_f32_bits, cap) && cap > 0.0f &&
                   op.inputs.size() == 1 && op.outputs.size() == 1 && op.tensors.empty() && op.states.empty();
        }
        default:
            return false;
        }
    }
    if (!enum_between(op.kind, 1, 7) || op.semantic_version != 1) return false;
    switch (op.kind) {
    case OperatorKind::EmbeddingLookup: {
        auto* payload = std::get_if<EmbeddingLookupPayload>(&op.payload);
        float scale = 0.0f;
        return payload && f32_bits(payload->scale_f32_bits, scale) && scale > 0.0f &&
               payload->vocabulary != 0 && payload->width != 0 && payload->flags == 0 &&
               op.inputs.size() == 1 && op.outputs.size() == 1;
    }
    case OperatorKind::RmsNorm: {
        auto* payload = std::get_if<RmsNormPayload>(&op.payload);
        float epsilon = 0.0f;
        return payload && f32_bits(payload->epsilon_f32_bits, epsilon) && epsilon >= 0.0f &&
               payload->axis == -1 && payload->weight_mode == 1;
    }
    case OperatorKind::Linear: {
        auto* payload = std::get_if<LinearPayload>(&op.payload);
        return payload && payload->accumulation_type == ScalarType::F32 &&
               op.inputs.size() == 1 && op.outputs.size() == 1;
    }
    case OperatorKind::Rope: {
        auto* payload = std::get_if<RopePayload>(&op.payload);
        float base = 0.0f;
        float scale = 0.0f;
        return payload && payload->pairing == RopePairing::HalfSplit && payload->position_from_cursor &&
               payload->rotary_dimension != 0 && payload->rotary_dimension % 2 == 0 &&
               payload->frequency_dimension == 0 &&
               payload->position_sections == std::array<uint32_t, 4>{} &&
               f32_bits(payload->base_f32_bits, base) && base > 0.0f &&
               f32_bits(payload->scale_f32_bits, scale) && scale > 0.0f;
    }
    case OperatorKind::CausalAttention: {
        auto* payload = std::get_if<CausalAttentionPayload>(&op.payload);
        float scale = 0.0f;
        return payload && payload->query_heads != 0 && payload->kv_heads != 0 &&
               payload->head_dimension != 0 && payload->query_heads % payload->kv_heads == 0 &&
               f32_bits(payload->scale_f32_bits, scale) && scale > 0.0f &&
               payload->mask == AttentionMask::Causal && payload->cache_policy == CachePolicy::Global &&
               payload->window == AttentionWindowKind::Global && payload->window_tokens == 0 &&
               payload->value_source == ValueSource::SeparateProjection &&
               payload->value_source_value == UINT32_MAX;
    }
    case OperatorKind::SwiGlu: {
        auto* payload = std::get_if<SwiGluPayload>(&op.payload);
        return payload && payload->activation == ActivationKind::Silu;
    }
    case OperatorKind::Add:
        return std::holds_alternative<AddPayload>(op.payload);
    case OperatorKind::DepthwiseConvSilu:
    case OperatorKind::GatedDeltaNet:
    case OperatorKind::GatedAttention:
    case OperatorKind::GatedRmsNorm:
    case OperatorKind::L2Normalize:
    case OperatorKind::AxisSplit:
    case OperatorKind::Concat:
    case OperatorKind::RouterTopK:
    case OperatorKind::RoutedLinear:
    case OperatorKind::GatedActivation:
    case OperatorKind::WeightedExpertReduce:
    case OperatorKind::Scale:
    case OperatorKind::TanhSoftcap:
        return false;
    }
    return false;
}

bool valid_state(const SemanticState& state) {
    if (!valid_dimensions(state.dimensions) || state.formats.empty() || state.formats.size() > kMaxPlanes || state.flags != 0) {
        return false;
    }
    if (state.semantic_version >= 2 && state.semantic_version <= 7) {
        if (state.kind == StateKind::KeyCache || state.kind == StateKind::ValueCache) {
            if (state.position_policy != PositionPolicy::AppendOnly || state.formats.size() != 1 ||
                state.dimensions.size() != 3 || state.dimensions[0].kind != DimensionKind::Symbol) return false;
            const StateFormat& format = state.formats[0];
            const TransformDomain encoded = state.kind == StateKind::KeyCache ? TransformDomain::RopeApplied : TransformDomain::Untransformed;
            const StateUpdateKind update = state.kind == StateKind::KeyCache ? StateUpdateKind::AppendKey : StateUpdateKind::AppendValue;
            return state.update_kind == update && format.kind == StateFormatKind::GlobalContiguous && format.version == 1 &&
                   format.logical_type == ScalarType::F32 && format.encoded_type == ScalarType::F32 &&
                   format.logical_domain == TransformDomain::Untransformed && format.encoded_domain == encoded &&
                   format.codec == CodecKind::Fp32 && format.cache_policy == CachePolicy::Global &&
                   format.layout_policy == LayoutPolicy::TokenMajorContiguous && format.flags == 0 &&
                   format.tile_tokens == 0 && format.mutable_tokens == 0 && format.alignment == 64 && format.reserved == 0;
        }
        if (state.formats.size() != 1 || state.position_policy != PositionPolicy::ReplaceAtCursor) return false;
        const StateFormat& format = state.formats[0];
        if (format.kind != StateFormatKind::RecurrentContiguous || format.version != 1 ||
            format.logical_type != ScalarType::F32 || format.encoded_type != ScalarType::F32 ||
            format.logical_domain != TransformDomain::Untransformed || format.encoded_domain != TransformDomain::Untransformed ||
            format.codec != CodecKind::Fp32 || format.cache_policy != CachePolicy::Recurrent || format.flags != 0 ||
            format.tile_tokens != 0 || format.mutable_tokens != 0 || format.alignment != 64 || format.reserved != 0) {
            return false;
        }
        if (state.kind == StateKind::RecurrentConvHistory) {
            return state.update_kind == StateUpdateKind::ShiftHistory &&
                   format.layout_policy == LayoutPolicy::ChannelMajorHistory && state.dimensions.size() == 2;
        }
        return state.kind == StateKind::RecurrentDeltaMatrix && state.update_kind == StateUpdateKind::DeltaMatrix &&
               (format.layout_policy == LayoutPolicy::ValueHeadOutputRowKeyColumn ||
                format.layout_policy == LayoutPolicy::ValueHeadKeyRowOutputColumn) && state.dimensions.size() == 3;
    }
    if (!enum_between(state.kind, 1, 2) || state.semantic_version != 1 ||
        !enum_between(state.update_kind, 1, 2) || state.position_policy != PositionPolicy::AppendOnly) return false;
    for (const StateFormat& format : state.formats) {
        if (format.kind != StateFormatKind::GlobalContiguous || format.version != 1 ||
            format.logical_type != ScalarType::F32 || format.encoded_type != ScalarType::F32 ||
            !enum_between(format.logical_domain, 1, 2) || !enum_between(format.encoded_domain, 1, 2) ||
            format.codec != CodecKind::Fp32 || format.cache_policy != CachePolicy::Global ||
            format.layout_policy != LayoutPolicy::TokenMajorContiguous || format.flags != 0 ||
            format.tile_tokens != 0 || format.mutable_tokens != 0 || format.alignment != 64 ||
            format.reserved != 0) {
            return false;
        }
    }
    return true;
}

bool dense_ids(const auto& records) {
    if (records.size() > std::numeric_limits<uint32_t>::max()) return false;
    for (size_t index = 0; index != records.size(); ++index) {
        if (records[index].id != index) return false;
    }
    return true;
}

bool constant_dimensions(const SemanticState& state, std::initializer_list<uint64_t> expected) {
    if (state.dimensions.size() != expected.size()) return false;
    size_t index = 0;
    for (uint64_t extent : expected) {
        const Dimension& dimension = state.dimensions[index++];
        if (dimension.kind != DimensionKind::Constant || dimension.constant_or_symbol != extent) return false;
    }
    return true;
}

const SemanticState* model_state(const SemanticModel& model, uint32_t id) {
    return id < model.states.size() && model.states[id].id == id ? &model.states[id] : nullptr;
}

const SemanticTensor* model_tensor(const SemanticModel& model, uint32_t id) {
    return id < model.tensors.size() && model.tensors[id].id == id ? &model.tensors[id] : nullptr;
}

bool valid_recurrent_bindings(const SemanticModel& model, const SemanticOperator& op) {
    if (op.kind == OperatorKind::DepthwiseConvSilu) {
        const auto& payload = std::get<DepthwiseConvSiluPayload>(op.payload);
        const SemanticState* state = model_state(model, op.states[0]);
        const SemanticTensor* weight = model_tensor(model, op.tensors[0]);
        const uint64_t channels = static_cast<uint64_t>(payload.head_dimension) *
                                  (2ull * payload.qk_heads + payload.value_heads);
        return channels <= UINT32_MAX && state && weight && state->kind == StateKind::RecurrentConvHistory &&
               constant_dimensions(*state, {channels, payload.kernel - 1}) &&
               weight->role == TensorRole::RecurrentConvWeight && weight->dimensions.size() == 2 &&
               weight->dimensions[0] == Dimension{DimensionKind::Constant, channels} &&
               weight->dimensions[1] == Dimension{DimensionKind::Constant, payload.kernel};
    }
    if (op.kind == OperatorKind::GatedDeltaNet) {
        const auto& payload = std::get<GatedDeltaNetPayload>(op.payload);
        const SemanticState* state = model_state(model, op.states[0]);
        const SemanticTensor* dt = model_tensor(model, op.tensors[0]);
        const SemanticTensor* decay = model_tensor(model, op.tensors[1]);
        return state && dt && decay && state->kind == StateKind::RecurrentDeltaMatrix &&
               constant_dimensions(*state, {payload.value_heads, payload.head_dimension, payload.head_dimension}) &&
               dt->role == TensorRole::RecurrentDtBias && decay->role == TensorRole::RecurrentDecayWeight;
    }
    return true;
}

const SemanticValue* model_value(const SemanticModel& model, uint32_t id) {
    return id < model.values.size() && model.values[id].id == id ? &model.values[id] : nullptr;
}

bool same_dimensions(const std::vector<Dimension>& left, const std::vector<Dimension>& right) {
    return left == right;
}

bool valid_router_outputs(const SemanticModel& model, const SemanticOperator& op) {
    const auto* payload = std::get_if<RouterTopKPayload>(&op.payload);
    if (!payload || op.inputs.size() != 1 || op.outputs.size() != 2) return false;
    const SemanticValue* scores = model_value(model, op.inputs[0]);
    const SemanticValue* ids = model_value(model, op.outputs[0]);
    const SemanticValue* weights = model_value(model, op.outputs[1]);
    if (!scores || !ids || !weights || scores->logical_type != ScalarType::F32 || ids->logical_type != ScalarType::U32 ||
        weights->logical_type != ScalarType::F32 || scores->dimensions.empty() || scores->dimensions.size() != ids->dimensions.size() ||
        !same_dimensions(ids->dimensions, weights->dimensions)) {
        return false;
    }
    const size_t last = scores->dimensions.size() - 1;
    if (scores->dimensions[last] != Dimension{DimensionKind::Constant, payload->expert_count} ||
        ids->dimensions[last] != Dimension{DimensionKind::Constant, payload->selected_count}) {
        return false;
    }
    for (size_t index = 0; index != last; ++index) if (scores->dimensions[index] != ids->dimensions[index]) return false;
    return true;
}

const RouterTopKPayload* router_for_worklist(const SemanticModel& model, uint32_t ids, uint32_t weights) {
    const RouterTopKPayload* result = nullptr;
    for (const SemanticOperator& candidate : model.operators) {
        if (candidate.kind != OperatorKind::RouterTopK || candidate.outputs.size() != 2 ||
            candidate.outputs[0] != ids || candidate.outputs[1] != weights) continue;
        const auto* payload = std::get_if<RouterTopKPayload>(&candidate.payload);
        if (!payload || result) return nullptr;
        result = payload;
    }
    return result;
}

bool valid_routed_linear_bindings(const SemanticModel& model, const SemanticOperator& op) {
    const SemanticValue* input = model_value(model, op.inputs[0]);
    const SemanticValue* ids = model_value(model, op.inputs[1]);
    const SemanticValue* weights = model_value(model, op.inputs[2]);
    const SemanticValue* output = model_value(model, op.outputs[0]);
    const SemanticTensor* tensor = model_tensor(model, op.tensors[0]);
    const RouterTopKPayload* router = router_for_worklist(model, op.inputs[1], op.inputs[2]);
    if (!input || !ids || !weights || !output || !tensor || !router || input->logical_type != ScalarType::F32 ||
        output->logical_type != ScalarType::F32 || input->dimensions.empty() || ids->dimensions.empty() ||
        !same_dimensions(ids->dimensions, weights->dimensions) ||
        tensor->expert_axis.kind != ExpertAxisKind::ExpertBank || tensor->expert_axis.expert_count != router->expert_count) {
        return false;
    }
    if (ids->logical_type != ScalarType::U32 || weights->logical_type != ScalarType::F32 ||
        ids->dimensions.back() != Dimension{DimensionKind::Constant, router->selected_count}) {
        return false;
    }
    const Dimension& input_width = tensor->dimensions[tensor->expert_axis.input_axis];
    const Dimension& output_width = tensor->dimensions[tensor->expert_axis.output_axis];
    const size_t ids_last = ids->dimensions.size() - 1;
    const bool input_is_unrouted = input->dimensions.size() == ids->dimensions.size() &&
                                    input->dimensions.back() == input_width &&
                                    output->dimensions.size() == input->dimensions.size() + 1 &&
                                    output->dimensions[ids_last] == Dimension{DimensionKind::Constant, router->selected_count} &&
                                    output->dimensions.back() == output_width;
    if (input_is_unrouted) {
        for (size_t index = 0; index != ids_last; ++index) {
            if (ids->dimensions[index] != input->dimensions[index] || output->dimensions[index] != input->dimensions[index]) return false;
        }
        return true;
    }
    const bool input_is_routed = input->dimensions.size() == ids->dimensions.size() + 1 &&
                                  output->dimensions.size() == input->dimensions.size() &&
                                  input->dimensions.back() == input_width && output->dimensions.back() == output_width;
    if (!input_is_routed) return false;
    for (size_t index = 0; index != ids->dimensions.size(); ++index) {
        if (input->dimensions[index] != ids->dimensions[index] || output->dimensions[index] != input->dimensions[index]) return false;
    }
    return true;
}

bool valid_weighted_reduce_bindings(const SemanticModel& model, const SemanticOperator& op) {
    const auto* payload = std::get_if<WeightedExpertReducePayload>(&op.payload);
    const SemanticValue* expert_outputs = model_value(model, op.inputs[0]);
    const SemanticValue* ids = model_value(model, op.inputs[1]);
    const SemanticValue* weights = model_value(model, op.inputs[2]);
    const SemanticValue* output = model_value(model, op.outputs[0]);
    const RouterTopKPayload* router = router_for_worklist(model, op.inputs[1], op.inputs[2]);
    if (!payload || !expert_outputs || !ids || !weights || !output || !router || expert_outputs->logical_type != ScalarType::F32 ||
        output->logical_type != ScalarType::F32 || ids->logical_type != ScalarType::U32 || weights->logical_type != ScalarType::F32 ||
        !same_dimensions(ids->dimensions, weights->dimensions) || ids->dimensions.empty() ||
        expert_outputs->dimensions.size() != ids->dimensions.size() + 1 ||
        output->dimensions.size() != ids->dimensions.size() ||
        ids->dimensions.back() != Dimension{DimensionKind::Constant, router->selected_count}) {
        return false;
    }
    const size_t route_last = ids->dimensions.size() - 1;
    for (size_t index = 0; index != route_last; ++index) {
        if (expert_outputs->dimensions[index] != ids->dimensions[index] || output->dimensions[index] != ids->dimensions[index]) return false;
    }
    if (expert_outputs->dimensions[route_last] != ids->dimensions[route_last] ||
        output->dimensions.back() != expert_outputs->dimensions.back()) return false;
    if (payload->scale_source == ExpertScaleSource::None) return op.tensors.empty();
    const SemanticTensor* scale = model_tensor(model, op.tensors[0]);
    return scale && scale->dimensions == std::vector<Dimension>{{DimensionKind::Constant, router->expert_count}};
}

bool valid_attention_value_source(const SemanticModel& model, const SemanticOperator& op) {
    const auto& payload = std::get<CausalAttentionPayload>(op.payload);
    if (payload.window == AttentionWindowKind::Sliding && payload.window_tokens > model.maximum_context) return false;
    const SemanticState* key_state = op.states.empty() ? nullptr : model_state(model, op.states[0]);
    if (!key_state || key_state->kind != StateKind::KeyCache) return false;
    if (payload.value_source == ValueSource::SeparateProjection) {
        return op.inputs.size() == 3 && op.states.size() == 2 && payload.value_source_value == op.inputs[2] &&
               model_state(model, op.states[1]) && model_state(model, op.states[1])->kind == StateKind::ValueCache;
    }
    if (payload.value_source == ValueSource::KeyStateAlias) {
        return op.inputs.size() == 2 && op.states.size() == 1 && payload.value_source_value == op.inputs[1];
    }
    if (op.inputs.size() != 2 || op.states.size() != 2 || !model_state(model, op.states[1]) ||
        model_state(model, op.states[1])->kind != StateKind::ValueCache) {
        return false;
    }
    if (payload.value_source == ValueSource::KeyPostRope) return payload.value_source_value == op.inputs[1];
    if (payload.value_source != ValueSource::KeyPreRope) return false;
    uint32_t matches = 0;
    for (const SemanticOperator& candidate : model.operators) {
        if (candidate.kind == OperatorKind::Rope && candidate.inputs.size() == 2 && candidate.outputs.size() == 2 &&
            candidate.inputs[1] == payload.value_source_value && candidate.outputs[1] == op.inputs[1]) {
            ++matches;
        }
    }
    return matches == 1;
}

bool valid_v7_bindings(const SemanticModel& model, const SemanticOperator& op) {
    switch (op.kind) {
    case OperatorKind::RouterTopK:
        return valid_router_outputs(model, op);
    case OperatorKind::RoutedLinear:
        return valid_routed_linear_bindings(model, op);
    case OperatorKind::GatedActivation:
        return same_dimensions(model_value(model, op.inputs[0])->dimensions, model_value(model, op.inputs[1])->dimensions) &&
               same_dimensions(model_value(model, op.inputs[0])->dimensions, model_value(model, op.outputs[0])->dimensions);
    case OperatorKind::WeightedExpertReduce:
        return valid_weighted_reduce_bindings(model, op);
    case OperatorKind::Scale:
    case OperatorKind::TanhSoftcap:
        return same_dimensions(model_value(model, op.inputs[0])->dimensions, model_value(model, op.outputs[0])->dimensions);
    case OperatorKind::CausalAttention:
        return valid_attention_value_source(model, op);
    default:
        return true;
    }
}

bool valid_model(const SemanticModel& model) {
    const bool v1 = model.schema_major == 1 && model.schema_minor == 0 &&
                    model.opset_major == 1 && model.opset_minor == 0;
    const bool v2 = model.schema_major == 2 && model.schema_minor == 0 &&
                    model.opset_major == 2 && model.opset_minor == 0;
    const bool v3 = model.schema_major == 3 && model.schema_minor == 0 &&
                    model.opset_major == 3 && model.opset_minor == 0;
    const bool v4 = model.schema_major == 4 && model.schema_minor == 0 &&
                    model.opset_major == 4 && model.opset_minor == 0;
    const bool v5 = model.schema_major == 5 && model.schema_minor == 0 &&
                    model.opset_major == 5 && model.opset_minor == 0;
    const bool v6 = model.schema_major == 6 && model.schema_minor == 0 &&
                    model.opset_major == 6 && model.opset_minor == 0;
    const bool v7 = model.schema_major == 7 && model.schema_minor == 0 &&
                    model.opset_major == 7 && model.opset_minor == 0;
    if ((!v1 && !v2 && !v3 && !v4 && !v5 && !v6 && !v7) || (v1 && model.maximum_context != 32768) ||
        ((v2 || v3 || v4 || v5 || v6 || v7) && (model.maximum_context == 0 || model.maximum_context > 262144)) ||
        model.entry_kind != EntryKind::TokenIds ||
        model.vocabulary_size == 0 || !valid_token_id(model.bos_id, model.vocabulary_size) ||
        !valid_token_id(model.eos_id, model.vocabulary_size) || model.stop_ids.size() > kMaxVector ||
        model.tensors.size() > kMaxTensors || model.values.size() > kMaxValues ||
        model.operators.size() > kMaxOperators || model.layers.size() > kSemanticModelMaximumLayers ||
        model.states.size() > kMaxStates || model.constraints.size() > kMaxConstraints ||
        model.capabilities.size() > kMaxVector || model.fallbacks.size() > kMaxVector ||
        !dense_ids(model.tensors) || !dense_ids(model.values) || !dense_ids(model.operators) ||
        !dense_ids(model.states)) {
        return false;
    }
    for (uint32_t id : model.stop_ids) if (id >= model.vocabulary_size) return false;
    if (model.input_values_first > model.values.size() || model.input_values_count > model.values.size() - model.input_values_first ||
        model.output_values_first > model.values.size() || model.output_values_count > model.values.size() - model.output_values_first) {
        return false;
    }
    for (const SemanticTensor& tensor : model.tensors)
        if (!valid_tensor(tensor, v7, v6 || v7)) return false;
    for (const SemanticValue& value : model.values) {
        if (!valid_scalar_type(value.logical_type) || !valid_dimensions(value.dimensions) ||
            value.dimensions.empty() || value.flags != 0) return false;
    }
    for (const SemanticOperator& op : model.operators) {
        if ((v1 && op.semantic_version != 1) || (v2 && op.semantic_version != 2) ||
            (v3 && op.semantic_version != 3) || (v4 && op.semantic_version != 4) ||
            (v5 && op.semantic_version != 5) || (v6 && op.semantic_version != 6) ||
            (v7 && op.semantic_version != 7) || !valid_payload(op)) return false;
        for (uint32_t id : op.inputs) if (id >= model.values.size()) return false;
        for (uint32_t id : op.outputs) if (id >= model.values.size()) return false;
        for (uint32_t id : op.tensors) if (id >= model.tensors.size()) return false;
        for (uint32_t id : op.states) if (id >= model.states.size()) return false;
        if ((v2 || v3 || v4 || v5 || v6 || v7) && !valid_recurrent_bindings(model, op)) return false;
    }
    uint32_t operator_cursor = 0;
    bool speculative_started = false;
    for (size_t index = 0; index != model.layers.size(); ++index) {
        const SemanticLayer& layer = model.layers[index];
        const bool speculative = (layer.flags & kSemanticLayerFlagSpeculative) != 0;
        if (layer.layer_index != index || layer.first_operator < operator_cursor ||
            (layer.flags & ~((v6 || v7) ? kSemanticLayerFlagSpeculative : 0u)) != 0 ||
            (!speculative && speculative_started) ||
            layer.operator_count > model.operators.size() - operator_cursor) return false;
        if (layer.first_operator > model.operators.size() || layer.operator_count > model.operators.size() - layer.first_operator) return false;
        speculative_started = speculative_started || speculative;
        operator_cursor = layer.first_operator + layer.operator_count;
    }
    for (const SemanticState& state : model.states) {
        if (state.semantic_version != model.opset_major || !valid_state(state)) return false;
        if (state.kind == StateKind::RecurrentDeltaMatrix) {
            const LayoutPolicy expected = (v3 || v4 || v5 || v6 || v7) ? LayoutPolicy::ValueHeadKeyRowOutputColumn
                                              : LayoutPolicy::ValueHeadOutputRowKeyColumn;
            if (state.formats[0].layout_policy != expected) return false;
        }
    }
    for (const SemanticOperator& op : model.operators) {
        if (op.kind != OperatorKind::GatedDeltaNet) continue;
        const auto& payload = std::get<GatedDeltaNetPayload>(op.payload);
        const DeltaStateLayout expected = (v3 || v4 || v5 || v6 || v7) ? DeltaStateLayout::ValueHeadKeyRowOutputColumn
                                             : DeltaStateLayout::ValueHeadOutputRowKeyColumn;
        if (payload.state_layout != expected) return false;
    }
    if (v7) for (const SemanticOperator& op : model.operators) if (!valid_v7_bindings(model, op)) return false;
    for (const SemanticConstraint& constraint : model.constraints) {
        if (!enum_between(constraint.kind, 1, 6) || !enum_between(constraint.lhs_kind, 1, 4) ||
            !enum_between(constraint.rhs_kind, 1, 4)) return false;
    }
    for (const CapabilityRequirement& capability : model.capabilities) {
        if (!enum_between(capability.capability, 1, 8) || capability.flags != 0) return false;
    }
    for (const SemanticFallback& fallback : model.fallbacks) {
        if (!enum_between(fallback.kind, 0, 1) || !enum_between(fallback.phase, 1, 5) ||
            fallback.numerical_class != NumericalClass::ExactFp32 || fallback.flags != 0) return false;
    }
    return true;
}

void append_dimension(std::vector<uint8_t>& output, const Dimension& dimension) {
    append_u8(output, static_cast<uint8_t>(dimension.kind));
    append_zeros(output, 7);
    append_u64(output, dimension.constant_or_symbol);
}

bool read_dimension(const std::vector<uint8_t>& bytes, size_t& offset, Dimension& dimension) {
    uint8_t kind = 0;
    if (!read_u8(bytes, offset, kind) || bytes.size() - offset < 7) return false;
    for (unsigned index = 0; index != 7; ++index) if (bytes[offset++] != 0) return false;
    if (!read_u64(bytes, offset, dimension.constant_or_symbol)) return false;
    dimension.kind = static_cast<DimensionKind>(kind);
    return valid_dimension(dimension);
}

void append_layout(std::vector<uint8_t>& output, const PhysicalLayout& layout) {
    append_u16(output, static_cast<uint16_t>(layout.kind));
    append_u16(output, layout.version);
    append_u16(output, static_cast<uint16_t>(layout.packing));
    append_u8(output, layout.rank);
    append_u8(output, layout.block_rank);
    output.insert(output.end(), layout.axis_order.begin(), layout.axis_order.end());
    for (uint64_t stride : layout.strides) append_u64(output, stride);
    append_u32(output, layout.block_elements);
    append_u32(output, layout.block_bytes);
    append_u32(output, layout.flags);
}

bool read_layout(const std::vector<uint8_t>& bytes, size_t& offset, PhysicalLayout& layout) {
    uint16_t kind = 0, packing = 0;
    if (!read_u16(bytes, offset, kind) || !read_u16(bytes, offset, layout.version) ||
        !read_u16(bytes, offset, packing) || !read_u8(bytes, offset, layout.rank) ||
        !read_u8(bytes, offset, layout.block_rank) || bytes.size() - offset < 8) return false;
    layout.kind = static_cast<PhysicalLayoutKind>(kind);
    layout.packing = static_cast<PackingKind>(packing);
    for (uint8_t& axis : layout.axis_order) axis = bytes[offset++];
    for (uint64_t& stride : layout.strides) if (!read_u64(bytes, offset, stride)) return false;
    return read_u32(bytes, offset, layout.block_elements) && read_u32(bytes, offset, layout.block_bytes) &&
           read_u32(bytes, offset, layout.flags);
}

void append_quantization(std::vector<uint8_t>& output, const Quantization& quantization) {
    append_u16(output, static_cast<uint16_t>(quantization.kind));
    append_u16(output, quantization.version);
    append_u16(output, static_cast<uint16_t>(quantization.accumulation_type));
    append_u16(output, static_cast<uint16_t>(quantization.scale_type));
    append_u16(output, static_cast<uint16_t>(quantization.zero_type));
    append_u16(output, static_cast<uint16_t>(quantization.bias_type));
    append_u32(output, quantization.block_elements);
    append_u32(output, quantization.block_bytes);
    append_u32(output, quantization.group_size);
    append_u32(output, quantization.required_plane_mask);
    append_u32(output, quantization.flags);
}

bool read_quantization(const std::vector<uint8_t>& bytes, size_t& offset, Quantization& quantization) {
    uint16_t kind = 0, accumulation = 0, scale = 0, zero = 0, bias = 0;
    if (!read_u16(bytes, offset, kind) || !read_u16(bytes, offset, quantization.version) ||
        !read_u16(bytes, offset, accumulation) || !read_u16(bytes, offset, scale) ||
        !read_u16(bytes, offset, zero) || !read_u16(bytes, offset, bias)) return false;
    quantization.kind = static_cast<QuantizationKind>(kind);
    quantization.accumulation_type = static_cast<ScalarType>(accumulation);
    quantization.scale_type = static_cast<ScalarType>(scale);
    quantization.zero_type = static_cast<ScalarType>(zero);
    quantization.bias_type = static_cast<ScalarType>(bias);
    return read_u32(bytes, offset, quantization.block_elements) && read_u32(bytes, offset, quantization.block_bytes) &&
           read_u32(bytes, offset, quantization.group_size) && read_u32(bytes, offset, quantization.required_plane_mask) &&
           read_u32(bytes, offset, quantization.flags);
}

void append_expert_axis(std::vector<uint8_t>& output, const ExpertAxis& expert_axis) {
    append_u8(output, static_cast<uint8_t>(expert_axis.kind));
    append_u8(output, expert_axis.expert_axis);
    append_u8(output, expert_axis.member_axis);
    append_u8(output, expert_axis.input_axis);
    append_u8(output, expert_axis.output_axis);
    append_zeros(output, 3);
    append_u32(output, expert_axis.expert_count);
    append_u64(output, expert_axis.per_expert_byte_stride);
    append_u32(output, expert_axis.flags);
}

bool read_expert_axis(const std::vector<uint8_t>& bytes, size_t& offset, ExpertAxis& expert_axis) {
    uint8_t kind = 0;
    if (!read_u8(bytes, offset, kind) || !read_u8(bytes, offset, expert_axis.expert_axis) ||
        !read_u8(bytes, offset, expert_axis.member_axis) || !read_u8(bytes, offset, expert_axis.input_axis) ||
        !read_u8(bytes, offset, expert_axis.output_axis) || bytes.size() - offset < 3) {
        return false;
    }
    for (unsigned index = 0; index != 3; ++index) if (bytes[offset++] != 0) return false;
    expert_axis.kind = static_cast<ExpertAxisKind>(kind);
    return read_u32(bytes, offset, expert_axis.expert_count) &&
           read_u64(bytes, offset, expert_axis.per_expert_byte_stride) && read_u32(bytes, offset, expert_axis.flags);
}

void append_plane(std::vector<uint8_t>& output, const TensorPlane& plane) {
    append_u16(output, static_cast<uint16_t>(plane.kind));
    append_u16(output, static_cast<uint16_t>(plane.storage_type));
    append_u32(output, plane.artifact_id.value);
    append_u64(output, plane.offset);
    append_u64(output, plane.length);
    append_u32(output, plane.alignment);
    append_u32(output, plane.flags);
    append_zeros(output, 32);
}

bool read_plane(const std::vector<uint8_t>& bytes, size_t& offset, TensorPlane& plane) {
    uint16_t kind = 0, type = 0;
    if (!read_u16(bytes, offset, kind) || !read_u16(bytes, offset, type) ||
        !read_u32(bytes, offset, plane.artifact_id.value) || !read_u64(bytes, offset, plane.offset) ||
        !read_u64(bytes, offset, plane.length) || !read_u32(bytes, offset, plane.alignment) ||
        !read_u32(bytes, offset, plane.flags) || bytes.size() - offset < 32) return false;
    plane.kind = static_cast<PlaneKind>(kind);
    plane.storage_type = static_cast<ScalarType>(type);
    for (unsigned index = 0; index != 32; ++index) if (bytes[offset++] != 0) return false;
    return true;
}

uint32_t payload_tag(const SemanticOperator& op) {
    return static_cast<uint16_t>(op.kind);
}

void append_payload(std::vector<uint8_t>& output, const SemanticOperator& op) {
    switch (op.kind) {
    case OperatorKind::EmbeddingLookup: {
        const auto& payload = std::get<EmbeddingLookupPayload>(op.payload);
        append_u32(output, payload.scale_f32_bits); append_u32(output, payload.vocabulary);
        append_u32(output, payload.width); append_u32(output, payload.flags); break;
    }
    case OperatorKind::RmsNorm: {
        const auto& payload = std::get<RmsNormPayload>(op.payload);
        append_u32(output, payload.epsilon_f32_bits); append_u32(output, static_cast<uint32_t>(payload.axis));
        append_u8(output, payload.weight_mode); append_zeros(output, 7); break;
    }
    case OperatorKind::Linear: {
        const auto& payload = std::get<LinearPayload>(op.payload);
        append_u8(output, payload.transpose_weight ? 1 : 0); append_u8(output, payload.has_bias ? 1 : 0);
        append_u16(output, static_cast<uint16_t>(payload.accumulation_type)); append_u32(output, 0); break;
    }
    case OperatorKind::Rope: {
        const auto& payload = std::get<RopePayload>(op.payload);
        append_u8(output, static_cast<uint8_t>(payload.pairing)); append_u8(output, payload.position_from_cursor ? 1 : 0);
        append_u16(output, 0); append_u32(output, payload.rotary_dimension);
        append_u32(output, payload.base_f32_bits); append_u32(output, payload.scale_f32_bits);
        if (op.semantic_version >= 5) {
            for (uint32_t section : payload.position_sections) append_u32(output, section);
        }
        if (op.semantic_version >= 7) append_u32(output, payload.frequency_dimension);
        break;
    }
    case OperatorKind::CausalAttention: {
        const auto& payload = std::get<CausalAttentionPayload>(op.payload);
        append_u32(output, payload.query_heads); append_u32(output, payload.kv_heads);
        append_u32(output, payload.head_dimension); append_u32(output, payload.scale_f32_bits);
        append_u8(output, static_cast<uint8_t>(payload.mask)); append_u8(output, static_cast<uint8_t>(payload.cache_policy));
        append_u16(output, 0); append_u32(output, 0);
        if (op.semantic_version >= 7) {
            append_u8(output, static_cast<uint8_t>(payload.window));
            append_u8(output, static_cast<uint8_t>(payload.value_source));
            append_u16(output, 0);
            append_u32(output, payload.window_tokens);
            append_u32(output, payload.value_source_value);
            append_u32(output, 0);
        }
        break;
    }
    case OperatorKind::SwiGlu:
        append_u16(output, static_cast<uint16_t>(std::get<SwiGluPayload>(op.payload).activation));
        append_u16(output, 0); append_u32(output, 0); break;
    case OperatorKind::Add:
        break;
    case OperatorKind::DepthwiseConvSilu: {
        const auto& payload = std::get<DepthwiseConvSiluPayload>(op.payload);
        append_u32(output, payload.qk_heads); append_u32(output, payload.value_heads);
        append_u32(output, payload.head_dimension); append_u32(output, payload.kernel); break;
    }
    case OperatorKind::GatedDeltaNet: {
        const auto& payload = std::get<GatedDeltaNetPayload>(op.payload);
        append_u32(output, payload.qk_heads); append_u32(output, payload.value_heads);
        append_u32(output, payload.head_dimension);
        append_u8(output, static_cast<uint8_t>(payload.qk_mapping));
        append_u8(output, static_cast<uint8_t>(payload.beta_transform));
        append_u8(output, static_cast<uint8_t>(payload.decay_transform));
        append_u8(output, static_cast<uint8_t>(payload.state_layout));
        append_u32(output, payload.flags); append_u32(output, 0); break;
    }
    case OperatorKind::GatedAttention:
        break;
    case OperatorKind::GatedRmsNorm: {
        const auto& payload = std::get<GatedRmsNormPayload>(op.payload);
        append_u32(output, payload.epsilon_f32_bits);
        append_u16(output, static_cast<uint16_t>(payload.gate_activation));
        append_u8(output, payload.weight_mode); append_u8(output, 0); break;
    }
    case OperatorKind::L2Normalize:
        append_u32(output, std::get<L2NormalizePayload>(op.payload).epsilon_f32_bits);
        append_u32(output, 0); break;
    case OperatorKind::AxisSplit: {
        const auto& payload = std::get<AxisSplitPayload>(op.payload);
        append_u32(output, payload.first_width); append_u32(output, payload.second_width); break;
    }
    case OperatorKind::Concat:
        append_u32(output, static_cast<uint32_t>(std::get<ConcatPayload>(op.payload).axis));
        append_u32(output, 0);
        break;
    case OperatorKind::RouterTopK: {
        const auto& payload = std::get<RouterTopKPayload>(op.payload);
        append_u32(output, payload.expert_count); append_u32(output, payload.selected_count);
        append_u8(output, static_cast<uint8_t>(payload.score_domain));
        append_u8(output, static_cast<uint8_t>(payload.normalization_order));
        append_u8(output, static_cast<uint8_t>(payload.selected_weight_normalization));
        append_u8(output, static_cast<uint8_t>(payload.tie_policy));
        append_u8(output, static_cast<uint8_t>(payload.weight_source));
        append_zeros(output, 3); append_u32(output, payload.flags); append_u32(output, 0);
        break;
    }
    case OperatorKind::RoutedLinear:
        append_u16(output, static_cast<uint16_t>(std::get<RoutedLinearPayload>(op.payload).accumulation_type));
        append_u16(output, 0); append_u32(output, 0);
        break;
    case OperatorKind::GatedActivation:
        append_u16(output, static_cast<uint16_t>(std::get<GatedActivationPayload>(op.payload).activation));
        append_u16(output, 0); append_u32(output, 0);
        break;
    case OperatorKind::WeightedExpertReduce: {
        const auto& payload = std::get<WeightedExpertReducePayload>(op.payload);
        append_u8(output, static_cast<uint8_t>(payload.association));
        append_u8(output, static_cast<uint8_t>(payload.scale_source));
        append_u16(output, static_cast<uint16_t>(payload.accumulation_type));
        append_u32(output, 0);
        break;
    }
    case OperatorKind::Scale: {
        const auto& payload = std::get<ScalePayload>(op.payload);
        append_u8(output, static_cast<uint8_t>(payload.source)); append_zeros(output, 3);
        append_u32(output, payload.literal_f32_bits);
        break;
    }
    case OperatorKind::TanhSoftcap:
        append_u32(output, std::get<TanhSoftcapPayload>(op.payload).cap_f32_bits);
        append_u32(output, 0);
        break;
    }
}

uint32_t payload_length(OperatorKind kind, uint16_t semantic_version) {
    switch (kind) {
    case OperatorKind::EmbeddingLookup: case OperatorKind::RmsNorm: return 16;
    case OperatorKind::Rope: return semantic_version >= 7 ? 36 : semantic_version >= 5 ? 32 : 16;
    case OperatorKind::Linear: case OperatorKind::SwiGlu: return 8;
    case OperatorKind::CausalAttention: return semantic_version >= 7 ? 40 : 24;
    case OperatorKind::Add: return 0;
    case OperatorKind::DepthwiseConvSilu: return 16;
    case OperatorKind::GatedDeltaNet: return 24;
    case OperatorKind::GatedAttention: return 0;
    case OperatorKind::GatedRmsNorm: return 8;
    case OperatorKind::L2Normalize: return 8;
    case OperatorKind::AxisSplit: return 8;
    case OperatorKind::Concat: return 8;
    case OperatorKind::RouterTopK: return 24;
    case OperatorKind::RoutedLinear: case OperatorKind::GatedActivation:
    case OperatorKind::WeightedExpertReduce: case OperatorKind::Scale: case OperatorKind::TanhSoftcap:
        return 8;
    }
    return 0;
}

bool read_bool(const std::vector<uint8_t>& bytes, size_t& offset, bool& value) {
    uint8_t raw = 0;
    if (!read_u8(bytes, offset, raw) || raw > 1) return false;
    value = raw != 0;
    return true;
}

bool read_payload(const std::vector<uint8_t>& bytes, size_t& offset, OperatorKind kind,
                  uint16_t semantic_version, OperatorPayload& payload) {
    switch (kind) {
    case OperatorKind::EmbeddingLookup: {
        EmbeddingLookupPayload value;
        if (!read_u32(bytes, offset, value.scale_f32_bits) || !read_u32(bytes, offset, value.vocabulary) ||
            !read_u32(bytes, offset, value.width) || !read_u32(bytes, offset, value.flags)) return false;
        payload = value; return true;
    }
    case OperatorKind::RmsNorm: {
        RmsNormPayload value; uint32_t axis = 0;
        if (!read_u32(bytes, offset, value.epsilon_f32_bits) || !read_u32(bytes, offset, axis) ||
            !read_u8(bytes, offset, value.weight_mode) || bytes.size() - offset < 7) return false;
        value.axis = static_cast<int32_t>(axis);
        for (unsigned index = 0; index != 7; ++index) if (bytes[offset++] != 0) return false;
        payload = value; return true;
    }
    case OperatorKind::Linear: {
        LinearPayload value; uint16_t accumulation = 0; uint32_t flags = 0;
        if (!read_bool(bytes, offset, value.transpose_weight) || !read_bool(bytes, offset, value.has_bias) ||
            !read_u16(bytes, offset, accumulation) || !read_u32(bytes, offset, flags) || flags != 0) return false;
        value.accumulation_type = static_cast<ScalarType>(accumulation); payload = value; return true;
    }
    case OperatorKind::Rope: {
        RopePayload value; uint8_t pairing = 0; uint16_t reserved = 0;
        if (!read_u8(bytes, offset, pairing) || !read_bool(bytes, offset, value.position_from_cursor) ||
            !read_u16(bytes, offset, reserved) || reserved != 0 || !read_u32(bytes, offset, value.rotary_dimension) ||
            !read_u32(bytes, offset, value.base_f32_bits) || !read_u32(bytes, offset, value.scale_f32_bits)) return false;
        if (semantic_version >= 5) {
            for (uint32_t& section : value.position_sections) if (!read_u32(bytes, offset, section)) return false;
        }
        if (semantic_version >= 7 && !read_u32(bytes, offset, value.frequency_dimension)) return false;
        value.pairing = static_cast<RopePairing>(pairing); payload = value; return true;
    }
    case OperatorKind::CausalAttention: {
        CausalAttentionPayload value; uint8_t mask = 0, policy = 0; uint16_t flags = 0; uint32_t reserved = 0;
        if (!read_u32(bytes, offset, value.query_heads) || !read_u32(bytes, offset, value.kv_heads) ||
            !read_u32(bytes, offset, value.head_dimension) || !read_u32(bytes, offset, value.scale_f32_bits) ||
            !read_u8(bytes, offset, mask) || !read_u8(bytes, offset, policy) || !read_u16(bytes, offset, flags) ||
            !read_u32(bytes, offset, reserved) || flags != 0 || reserved != 0) return false;
        if (semantic_version >= 7) {
            uint8_t window = 0, source = 0;
            uint16_t extension_reserved = 0;
            uint32_t extension_zero = 0;
            if (!read_u8(bytes, offset, window) || !read_u8(bytes, offset, source) || !read_u16(bytes, offset, extension_reserved) ||
                !read_u32(bytes, offset, value.window_tokens) || !read_u32(bytes, offset, value.value_source_value) ||
                !read_u32(bytes, offset, extension_zero) || extension_reserved != 0 || extension_zero != 0) return false;
            value.window = static_cast<AttentionWindowKind>(window);
            value.value_source = static_cast<ValueSource>(source);
        }
        value.mask = static_cast<AttentionMask>(mask); value.cache_policy = static_cast<CachePolicy>(policy);
        payload = value; return true;
    }
    case OperatorKind::SwiGlu: {
        SwiGluPayload value; uint16_t activation = 0, reserved = 0; uint32_t flags = 0;
        if (!read_u16(bytes, offset, activation) || !read_u16(bytes, offset, reserved) || !read_u32(bytes, offset, flags) ||
            reserved != 0 || flags != 0) return false;
        value.activation = static_cast<ActivationKind>(activation); payload = value; return true;
    }
    case OperatorKind::Add:
        payload = AddPayload{}; return true;
    case OperatorKind::DepthwiseConvSilu: {
        DepthwiseConvSiluPayload value;
        if (!read_u32(bytes, offset, value.qk_heads) || !read_u32(bytes, offset, value.value_heads) ||
            !read_u32(bytes, offset, value.head_dimension) || !read_u32(bytes, offset, value.kernel)) return false;
        payload = value; return true;
    }
    case OperatorKind::GatedDeltaNet: {
        GatedDeltaNetPayload value; uint8_t qk_mapping = 0, beta = 0, decay = 0, layout = 0; uint32_t reserved = 0;
        if (!read_u32(bytes, offset, value.qk_heads) || !read_u32(bytes, offset, value.value_heads) ||
            !read_u32(bytes, offset, value.head_dimension) || !read_u8(bytes, offset, qk_mapping) ||
            !read_u8(bytes, offset, beta) || !read_u8(bytes, offset, decay) || !read_u8(bytes, offset, layout) ||
            !read_u32(bytes, offset, value.flags) || !read_u32(bytes, offset, reserved) || reserved != 0) return false;
        value.qk_mapping = static_cast<QkHeadMapping>(qk_mapping);
        value.beta_transform = static_cast<BetaTransform>(beta);
        value.decay_transform = static_cast<DecayTransform>(decay);
        value.state_layout = static_cast<DeltaStateLayout>(layout);
        payload = value; return true;
    }
    case OperatorKind::GatedAttention:
        payload = GatedAttentionPayload{}; return true;
    case OperatorKind::GatedRmsNorm: {
        GatedRmsNormPayload value; uint16_t activation = 0; uint8_t reserved = 0;
        if (!read_u32(bytes, offset, value.epsilon_f32_bits) || !read_u16(bytes, offset, activation) ||
            !read_u8(bytes, offset, value.weight_mode) || !read_u8(bytes, offset, reserved) || reserved != 0) return false;
        value.gate_activation = static_cast<ActivationKind>(activation); payload = value; return true;
    }
    case OperatorKind::L2Normalize: {
        L2NormalizePayload value; uint32_t reserved = 0;
        if (!read_u32(bytes, offset, value.epsilon_f32_bits) || !read_u32(bytes, offset, reserved) || reserved != 0) return false;
        payload = value; return true;
    }
    case OperatorKind::AxisSplit: {
        AxisSplitPayload value;
        if (!read_u32(bytes, offset, value.first_width) || !read_u32(bytes, offset, value.second_width)) return false;
        payload = value; return true;
    }
    case OperatorKind::Concat: {
        ConcatPayload value;
        uint32_t axis = 0, reserved = 0;
        if (!read_u32(bytes, offset, axis) || !read_u32(bytes, offset, reserved) || reserved != 0) return false;
        value.axis = static_cast<int32_t>(axis);
        payload = value;
        return true;
    }
    case OperatorKind::RouterTopK: {
        RouterTopKPayload value;
        uint8_t domain = 0, order = 0, normalization = 0, tie = 0, source = 0;
        uint32_t reserved = 0;
        if (!read_u32(bytes, offset, value.expert_count) || !read_u32(bytes, offset, value.selected_count) ||
            !read_u8(bytes, offset, domain) || !read_u8(bytes, offset, order) || !read_u8(bytes, offset, normalization) ||
            !read_u8(bytes, offset, tie) || !read_u8(bytes, offset, source) || bytes.size() - offset < 3) return false;
        for (unsigned index = 0; index != 3; ++index) if (bytes[offset++] != 0) return false;
        if (!read_u32(bytes, offset, value.flags) || !read_u32(bytes, offset, reserved) || reserved != 0) return false;
        value.score_domain = static_cast<RouterScoreDomain>(domain);
        value.normalization_order = static_cast<RouterNormalizationOrder>(order);
        value.selected_weight_normalization = static_cast<SelectedWeightNormalization>(normalization);
        value.tie_policy = static_cast<RouterTiePolicy>(tie);
        value.weight_source = static_cast<RouterWeightSource>(source);
        payload = value;
        return true;
    }
    case OperatorKind::RoutedLinear: {
        RoutedLinearPayload value;
        uint16_t accumulation = 0, reserved = 0;
        uint32_t flags = 0;
        if (!read_u16(bytes, offset, accumulation) || !read_u16(bytes, offset, reserved) ||
            !read_u32(bytes, offset, flags) || reserved != 0 || flags != 0) return false;
        value.accumulation_type = static_cast<ScalarType>(accumulation);
        payload = value;
        return true;
    }
    case OperatorKind::GatedActivation: {
        GatedActivationPayload value;
        uint16_t activation = 0, reserved = 0;
        uint32_t flags = 0;
        if (!read_u16(bytes, offset, activation) || !read_u16(bytes, offset, reserved) ||
            !read_u32(bytes, offset, flags) || reserved != 0 || flags != 0) return false;
        value.activation = static_cast<ActivationKind>(activation);
        payload = value;
        return true;
    }
    case OperatorKind::WeightedExpertReduce: {
        WeightedExpertReducePayload value;
        uint8_t association = 0, scale = 0;
        uint16_t accumulation = 0;
        uint32_t flags = 0;
        if (!read_u8(bytes, offset, association) || !read_u8(bytes, offset, scale) || !read_u16(bytes, offset, accumulation) ||
            !read_u32(bytes, offset, flags) || flags != 0) return false;
        value.association = static_cast<ExpertReduceAssociation>(association);
        value.scale_source = static_cast<ExpertScaleSource>(scale);
        value.accumulation_type = static_cast<ScalarType>(accumulation);
        payload = value;
        return true;
    }
    case OperatorKind::Scale: {
        ScalePayload value;
        uint8_t source = 0;
        if (!read_u8(bytes, offset, source) || bytes.size() - offset < 3) return false;
        for (unsigned index = 0; index != 3; ++index) if (bytes[offset++] != 0) return false;
        if (!read_u32(bytes, offset, value.literal_f32_bits)) return false;
        value.source = static_cast<ScaleSource>(source);
        payload = value;
        return true;
    }
    case OperatorKind::TanhSoftcap: {
        TanhSoftcapPayload value;
        uint32_t reserved = 0;
        if (!read_u32(bytes, offset, value.cap_f32_bits) || !read_u32(bytes, offset, reserved) || reserved != 0) return false;
        payload = value;
        return true;
    }
    }
    return false;
}

} // namespace

bool semantic_operator_contract_valid(const SemanticOperator& op) {
    return valid_payload(op);
}

SemanticEncodeResult encode_semantic_model(const SemanticModel& model) {
    if (!valid_model(model)) return wire_error(CompatibilityError::IR_CONSTRAINT_FAILED);

    std::vector<uint8_t> body;
    body.reserve(136 + model.stop_ids.size() * 4 + model.tensors.size() * 256);
    append_u32(body, model.maximum_context);
    append_u16(body, static_cast<uint16_t>(model.entry_kind));
    append_u16(body, 0);
    append_u32(body, static_cast<uint32_t>(model.tensors.size()));
    append_u32(body, static_cast<uint32_t>(model.values.size()));
    append_u32(body, static_cast<uint32_t>(model.operators.size()));
    append_u32(body, static_cast<uint32_t>(model.layers.size()));
    append_u32(body, static_cast<uint32_t>(model.states.size()));
    append_u32(body, static_cast<uint32_t>(model.constraints.size()));
    append_u32(body, static_cast<uint32_t>(model.capabilities.size()));
    append_u32(body, static_cast<uint32_t>(model.fallbacks.size()));
    append_u32(body, model.input_values_first);
    append_u32(body, model.input_values_count);
    append_u32(body, model.output_values_first);
    append_u32(body, model.output_values_count);
    append_u32(body, model.vocabulary_size);
    append_u32(body, model.bos_id);
    append_u32(body, model.eos_id);
    append_u32(body, static_cast<uint32_t>(model.stop_ids.size()));
    body.insert(body.end(), model.tokenizer_digest.begin(), model.tokenizer_digest.end());
    body.insert(body.end(), model.template_digest.begin(), model.template_digest.end());
    for (uint32_t id : model.stop_ids) append_u32(body, id);
    append_padding(body);

    for (const SemanticTensor& tensor : model.tensors) {
        append_u32(body, tensor.id);
        append_u16(body, static_cast<uint16_t>(tensor.role));
        append_u16(body, static_cast<uint16_t>(tensor.logical_type));
        append_u8(body, static_cast<uint8_t>(tensor.dimensions.size()));
        append_u8(body, static_cast<uint8_t>(tensor.planes.size()));
        append_u16(body, tensor.flags);
        for (const Dimension& dimension : tensor.dimensions) append_dimension(body, dimension);
        append_layout(body, tensor.layout);
        append_quantization(body, tensor.quantization);
        if (model.schema_major >= 7) append_expert_axis(body, tensor.expert_axis);
        for (const TensorPlane& plane : tensor.planes) append_plane(body, plane);
        append_padding(body);
    }
    for (const SemanticValue& value : model.values) {
        append_u32(body, value.id);
        append_u16(body, static_cast<uint16_t>(value.logical_type));
        append_u8(body, static_cast<uint8_t>(value.dimensions.size()));
        append_u8(body, value.flags);
        for (const Dimension& dimension : value.dimensions) append_dimension(body, dimension);
        append_padding(body);
    }
    for (const SemanticOperator& op : model.operators) {
        append_u32(body, op.id);
        append_u16(body, static_cast<uint16_t>(op.kind));
        append_u16(body, op.semantic_version);
        append_u16(body, static_cast<uint16_t>(op.inputs.size()));
        append_u16(body, static_cast<uint16_t>(op.outputs.size()));
        append_u16(body, static_cast<uint16_t>(op.tensors.size()));
        append_u16(body, static_cast<uint16_t>(op.states.size()));
        append_u16(body, static_cast<uint16_t>(payload_tag(op)));
        append_u16(body, 0);
        append_u32(body, payload_length(op.kind, op.semantic_version));
        append_u64(body, 0);
        for (uint32_t id : op.inputs) append_u32(body, id);
        for (uint32_t id : op.outputs) append_u32(body, id);
        for (uint32_t id : op.tensors) append_u32(body, id);
        for (uint32_t id : op.states) append_u32(body, id);
        append_payload(body, op);
        append_padding(body);
    }
    for (const SemanticLayer& layer : model.layers) {
        append_u32(body, layer.layer_index);
        append_u32(body, layer.first_operator);
        append_u32(body, layer.operator_count);
        append_u32(body, layer.flags);
    }
    for (const SemanticState& state : model.states) {
        append_u32(body, state.id);
        append_u16(body, static_cast<uint16_t>(state.kind));
        append_u16(body, state.semantic_version);
        append_u16(body, static_cast<uint16_t>(state.update_kind));
        append_u16(body, static_cast<uint16_t>(state.position_policy));
        append_u8(body, static_cast<uint8_t>(state.dimensions.size()));
        append_u8(body, static_cast<uint8_t>(state.formats.size()));
        append_u16(body, state.flags);
        for (const Dimension& dimension : state.dimensions) append_dimension(body, dimension);
        for (const StateFormat& format : state.formats) {
            append_u16(body, static_cast<uint16_t>(format.kind));
            append_u16(body, format.version);
            append_u16(body, static_cast<uint16_t>(format.logical_type));
            append_u16(body, static_cast<uint16_t>(format.encoded_type));
            append_u16(body, static_cast<uint16_t>(format.logical_domain));
            append_u16(body, static_cast<uint16_t>(format.encoded_domain));
            append_u16(body, static_cast<uint16_t>(format.codec));
            append_u16(body, static_cast<uint16_t>(format.cache_policy));
            append_u16(body, static_cast<uint16_t>(format.layout_policy));
            append_u16(body, format.flags);
            append_u32(body, format.tile_tokens);
            append_u32(body, format.mutable_tokens);
            append_u32(body, format.alignment);
            append_u32(body, format.reserved);
        }
        append_padding(body);
    }
    for (const SemanticConstraint& constraint : model.constraints) {
        append_u16(body, static_cast<uint16_t>(constraint.kind));
        append_u16(body, 0);
        append_u8(body, static_cast<uint8_t>(constraint.lhs_kind));
        append_u8(body, static_cast<uint8_t>(constraint.rhs_kind));
        append_zeros(body, 6);
        append_u32(body, constraint.lhs_id);
        append_u32(body, constraint.rhs_id);
        append_u32(body, constraint.lhs_axis);
        append_u32(body, constraint.rhs_axis);
        append_u64(body, constraint.constant);
        append_u64(body, constraint.divisor);
    }
    for (const CapabilityRequirement& capability : model.capabilities) {
        append_u16(body, static_cast<uint16_t>(capability.capability));
        append_u16(body, capability.minimum_version);
        append_u32(body, capability.flags);
    }
    for (const SemanticFallback& fallback : model.fallbacks) {
        append_u16(body, static_cast<uint16_t>(fallback.kind));
        append_u16(body, static_cast<uint16_t>(fallback.phase));
        append_u16(body, static_cast<uint16_t>(fallback.numerical_class));
        append_u16(body, fallback.flags);
    }
    if (body.size() > 16 * 1024 * 1024) return wire_error(CompatibilityError::IR_CONSTRAINT_FAILED);

    std::vector<uint8_t> output;
    output.reserve(64 + body.size());
    output.insert(output.end(), model.schema_major == 1
        ? std::initializer_list<uint8_t>{'L', 'A', 'P', 'I', 'R', '0', '0', '1'}
        : model.schema_major == 2
            ? std::initializer_list<uint8_t>{'L', 'A', 'P', 'I', 'R', '0', '0', '2'}
            : model.schema_major == 3
                ? std::initializer_list<uint8_t>{'L', 'A', 'P', 'I', 'R', '0', '0', '3'}
                : model.schema_major == 4
                    ? std::initializer_list<uint8_t>{'L', 'A', 'P', 'I', 'R', '0', '0', '4'}
                    : model.schema_major == 5
                        ? std::initializer_list<uint8_t>{'L', 'A', 'P', 'I', 'R', '0', '0', '5'}
                        : model.schema_major == 6
                            ? std::initializer_list<uint8_t>{'L', 'A', 'P', 'I', 'R', '0', '0', '6'}
                            : std::initializer_list<uint8_t>{'L', 'A', 'P', 'I', 'R', '0', '0', '7'});
    append_u16(output, model.schema_major);
    append_u16(output, model.schema_minor);
    append_u16(output, model.opset_major);
    append_u16(output, model.opset_minor);
    append_u64(output, body.size());
    append_u64(output, 64 + body.size());
    auto digest = sha256(body.data(), body.size());
    output.insert(output.end(), digest.begin(), digest.end());
    output.insert(output.end(), body.begin(), body.end());
    return output;
}

SemanticDecodeResult decode_semantic_model(const std::vector<uint8_t>& bytes) {
    const bool v1_magic = bytes.size() >= 64 && std::memcmp(bytes.data(), "LAPIR001", 8) == 0;
    const bool v2_magic = bytes.size() >= 64 && std::memcmp(bytes.data(), "LAPIR002", 8) == 0;
    const bool v3_magic = bytes.size() >= 64 && std::memcmp(bytes.data(), "LAPIR003", 8) == 0;
    const bool v4_magic = bytes.size() >= 64 && std::memcmp(bytes.data(), "LAPIR004", 8) == 0;
    const bool v5_magic = bytes.size() >= 64 && std::memcmp(bytes.data(), "LAPIR005", 8) == 0;
    const bool v6_magic = bytes.size() >= 64 && std::memcmp(bytes.data(), "LAPIR006", 8) == 0;
    const bool v7_magic = bytes.size() >= 64 && std::memcmp(bytes.data(), "LAPIR007", 8) == 0;
    if (!v1_magic && !v2_magic && !v3_magic && !v4_magic && !v5_magic && !v6_magic && !v7_magic) {
        return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
    }
    size_t offset = 8;
    uint16_t schema_major = 0, schema_minor = 0, opset_major = 0, opset_minor = 0;
    uint64_t body_length = 0, total_length = 0;
    if (!read_u16(bytes, offset, schema_major) || !read_u16(bytes, offset, schema_minor) ||
        !read_u16(bytes, offset, opset_major) || !read_u16(bytes, offset, opset_minor) ||
        !read_u64(bytes, offset, body_length) || !read_u64(bytes, offset, total_length) ||
        schema_major != (v1_magic ? 1 : v2_magic ? 2 : v3_magic ? 3 : v4_magic ? 4 : v5_magic ? 5 : v6_magic ? 6 : 7) ||
        schema_minor != 0 ||
        opset_major != (v1_magic ? 1 : v2_magic ? 2 : v3_magic ? 3 : v4_magic ? 4 : v5_magic ? 5 : v6_magic ? 6 : 7) ||
        opset_minor != 0 ||
        total_length != bytes.size() || body_length != bytes.size() - 64) {
        return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
    }
    std::array<uint8_t, 32> expected_digest{};
    std::memcpy(expected_digest.data(), bytes.data() + offset, expected_digest.size());
    if (expected_digest != sha256(bytes.data() + 64, static_cast<size_t>(body_length))) {
        return wire_error(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH);
    }
    offset = 64;
    if (body_length < 136 || body_length > 16 * 1024 * 1024) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
    SemanticModel model;
    model.schema_major = schema_major;
    model.schema_minor = schema_minor;
    model.opset_major = opset_major;
    model.opset_minor = opset_minor;
    std::array<uint32_t, 8> counts{};
    uint16_t entry = 0, reserved = 0;
    if (!read_u32(bytes, offset, model.maximum_context) || !read_u16(bytes, offset, entry) ||
        !read_u16(bytes, offset, reserved) || reserved != 0 || entry != static_cast<uint16_t>(EntryKind::TokenIds)) {
        return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
    }
    model.entry_kind = static_cast<EntryKind>(entry);
    for (uint32_t& count : counts) if (!read_u32(bytes, offset, count)) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
    if (!read_u32(bytes, offset, model.input_values_first) || !read_u32(bytes, offset, model.input_values_count) ||
        !read_u32(bytes, offset, model.output_values_first) || !read_u32(bytes, offset, model.output_values_count)) {
        return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
    }
    if (counts[0] > kMaxTensors || counts[1] > kMaxValues || counts[2] > kMaxOperators ||
        counts[3] > kSemanticModelMaximumLayers ||
        counts[4] > kMaxStates || counts[5] > kMaxConstraints || counts[6] > kMaxVector || counts[7] > kMaxVector) {
        return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
    }
    uint32_t stop_count = 0;
    if (!read_u32(bytes, offset, model.vocabulary_size) || !read_u32(bytes, offset, model.bos_id) ||
        !read_u32(bytes, offset, model.eos_id) || !read_u32(bytes, offset, stop_count) ||
        stop_count > kMaxVector || bytes.size() - offset < 64) {
        return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
    }
    std::memcpy(model.tokenizer_digest.data(), bytes.data() + offset, 32);
    offset += 32;
    std::memcpy(model.template_digest.data(), bytes.data() + offset, 32);
    offset += 32;
    if (bytes.size() - offset < static_cast<size_t>(stop_count) * 4) {
        return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
    }
    model.stop_ids.resize(stop_count);
    for (uint32_t& id : model.stop_ids) {
        if (!read_u32(bytes, offset, id)) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
    }
    if (!consume_padding(bytes, offset)) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
    model.tensors.reserve(counts[0]);
    for (uint32_t index = 0; index != counts[0]; ++index) {
        SemanticTensor tensor;
        uint16_t role = 0, type = 0;
        uint8_t rank = 0, planes = 0;
        if (!read_u32(bytes, offset, tensor.id) || !read_u16(bytes, offset, role) || !read_u16(bytes, offset, type) ||
            !read_u8(bytes, offset, rank) || !read_u8(bytes, offset, planes) || !read_u16(bytes, offset, tensor.flags) ||
            rank == 0 || rank > kMaxRank || planes == 0 || planes > kMaxPlanes) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        tensor.role = static_cast<TensorRole>(role);
        tensor.logical_type = static_cast<ScalarType>(type);
        tensor.dimensions.resize(rank);
        for (Dimension& dimension : tensor.dimensions) if (!read_dimension(bytes, offset, dimension)) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        if (!read_layout(bytes, offset, tensor.layout) || !read_quantization(bytes, offset, tensor.quantization) ||
            (schema_major >= 7 && !read_expert_axis(bytes, offset, tensor.expert_axis))) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        tensor.planes.resize(planes);
        for (TensorPlane& plane : tensor.planes) if (!read_plane(bytes, offset, plane)) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        if (!consume_padding(bytes, offset)) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        model.tensors.push_back(std::move(tensor));
    }
    model.values.reserve(counts[1]);
    for (uint32_t index = 0; index != counts[1]; ++index) {
        SemanticValue value;
        uint16_t type = 0; uint8_t rank = 0;
        if (!read_u32(bytes, offset, value.id) || !read_u16(bytes, offset, type) || !read_u8(bytes, offset, rank) ||
            !read_u8(bytes, offset, value.flags) || rank == 0 || rank > kMaxRank) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        value.logical_type = static_cast<ScalarType>(type); value.dimensions.resize(rank);
        for (Dimension& dimension : value.dimensions) if (!read_dimension(bytes, offset, dimension)) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        if (!consume_padding(bytes, offset)) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        model.values.push_back(std::move(value));
    }
    model.operators.reserve(counts[2]);
    for (uint32_t index = 0; index != counts[2]; ++index) {
        SemanticOperator op;
        uint16_t kind = 0, input_count = 0, output_count = 0, tensor_count = 0, state_count = 0, tag = 0, flags = 0;
        uint32_t length = 0; uint64_t zero = 0;
        if (!read_u32(bytes, offset, op.id) || !read_u16(bytes, offset, kind) || !read_u16(bytes, offset, op.semantic_version) ||
            !read_u16(bytes, offset, input_count) || !read_u16(bytes, offset, output_count) || !read_u16(bytes, offset, tensor_count) ||
            !read_u16(bytes, offset, state_count) || !read_u16(bytes, offset, tag) || !read_u16(bytes, offset, flags) ||
            !read_u32(bytes, offset, length) || !read_u64(bytes, offset, zero) || flags != 0 || zero != 0) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        op.kind = static_cast<OperatorKind>(kind);
        if (tag != kind || length != payload_length(op.kind, op.semantic_version) ||
            input_count > 64 || output_count > 64 || tensor_count > 64 || state_count > 64) {
            return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        }
        auto read_ids = [&](std::vector<uint32_t>& ids, uint16_t count) {
            ids.resize(count);
            for (uint32_t& id : ids) if (!read_u32(bytes, offset, id)) return false;
            return true;
        };
        if (!read_ids(op.inputs, input_count) || !read_ids(op.outputs, output_count) || !read_ids(op.tensors, tensor_count) ||
            !read_ids(op.states, state_count) || !read_payload(bytes, offset, op.kind, op.semantic_version, op.payload) ||
            !consume_padding(bytes, offset)) {
            return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        }
        model.operators.push_back(std::move(op));
    }
    model.layers.reserve(counts[3]);
    for (uint32_t index = 0; index != counts[3]; ++index) {
        SemanticLayer layer;
        if (!read_u32(bytes, offset, layer.layer_index) || !read_u32(bytes, offset, layer.first_operator) ||
            !read_u32(bytes, offset, layer.operator_count) || !read_u32(bytes, offset, layer.flags)) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        model.layers.push_back(layer);
    }
    model.states.reserve(counts[4]);
    for (uint32_t index = 0; index != counts[4]; ++index) {
        SemanticState state;
        uint16_t kind = 0, update = 0, position = 0; uint8_t rank = 0, format_count = 0;
        if (!read_u32(bytes, offset, state.id) || !read_u16(bytes, offset, kind) || !read_u16(bytes, offset, state.semantic_version) ||
            !read_u16(bytes, offset, update) || !read_u16(bytes, offset, position) || !read_u8(bytes, offset, rank) ||
            !read_u8(bytes, offset, format_count) || !read_u16(bytes, offset, state.flags) || rank > kMaxRank ||
            format_count == 0 || format_count > kMaxPlanes) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        state.kind = static_cast<StateKind>(kind); state.update_kind = static_cast<StateUpdateKind>(update);
        state.position_policy = static_cast<PositionPolicy>(position); state.dimensions.resize(rank);
        for (Dimension& dimension : state.dimensions) if (!read_dimension(bytes, offset, dimension)) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        state.formats.resize(format_count);
        for (StateFormat& format : state.formats) {
            uint16_t format_kind = 0, logical = 0, encoded = 0, logical_domain = 0, encoded_domain = 0, codec = 0, cache = 0, layout = 0;
            if (!read_u16(bytes, offset, format_kind) || !read_u16(bytes, offset, format.version) ||
                !read_u16(bytes, offset, logical) || !read_u16(bytes, offset, encoded) || !read_u16(bytes, offset, logical_domain) ||
                !read_u16(bytes, offset, encoded_domain) || !read_u16(bytes, offset, codec) || !read_u16(bytes, offset, cache) ||
                !read_u16(bytes, offset, layout) || !read_u16(bytes, offset, format.flags) || !read_u32(bytes, offset, format.tile_tokens) ||
                !read_u32(bytes, offset, format.mutable_tokens) || !read_u32(bytes, offset, format.alignment) || !read_u32(bytes, offset, format.reserved)) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
            format.kind = static_cast<StateFormatKind>(format_kind); format.logical_type = static_cast<ScalarType>(logical);
            format.encoded_type = static_cast<ScalarType>(encoded); format.logical_domain = static_cast<TransformDomain>(logical_domain);
            format.encoded_domain = static_cast<TransformDomain>(encoded_domain); format.codec = static_cast<CodecKind>(codec);
            format.cache_policy = static_cast<CachePolicy>(cache); format.layout_policy = static_cast<LayoutPolicy>(layout);
        }
        if (!consume_padding(bytes, offset)) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        model.states.push_back(std::move(state));
    }
    model.constraints.reserve(counts[5]);
    for (uint32_t index = 0; index != counts[5]; ++index) {
        SemanticConstraint constraint; uint16_t kind = 0, flags = 0; uint8_t lhs = 0, rhs = 0;
        if (!read_u16(bytes, offset, kind) || !read_u16(bytes, offset, flags) || !read_u8(bytes, offset, lhs) || !read_u8(bytes, offset, rhs) ||
            flags != 0 || bytes.size() - offset < 6) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        for (unsigned zero = 0; zero != 6; ++zero) if (bytes[offset++] != 0) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        if (!read_u32(bytes, offset, constraint.lhs_id) || !read_u32(bytes, offset, constraint.rhs_id) ||
            !read_u32(bytes, offset, constraint.lhs_axis) || !read_u32(bytes, offset, constraint.rhs_axis) ||
            !read_u64(bytes, offset, constraint.constant) || !read_u64(bytes, offset, constraint.divisor)) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        constraint.kind = static_cast<ConstraintKind>(kind); constraint.lhs_kind = static_cast<ConstraintOperandKind>(lhs); constraint.rhs_kind = static_cast<ConstraintOperandKind>(rhs);
        model.constraints.push_back(constraint);
    }
    model.capabilities.reserve(counts[6]);
    for (uint32_t index = 0; index != counts[6]; ++index) {
        CapabilityRequirement capability; uint16_t kind = 0;
        if (!read_u16(bytes, offset, kind) || !read_u16(bytes, offset, capability.minimum_version) || !read_u32(bytes, offset, capability.flags)) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        capability.capability = static_cast<Capability>(kind); model.capabilities.push_back(capability);
    }
    model.fallbacks.reserve(counts[7]);
    for (uint32_t index = 0; index != counts[7]; ++index) {
        SemanticFallback fallback; uint16_t kind = 0, phase = 0, numerical = 0;
        if (!read_u16(bytes, offset, kind) || !read_u16(bytes, offset, phase) || !read_u16(bytes, offset, numerical) || !read_u16(bytes, offset, fallback.flags)) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
        fallback.kind = static_cast<FallbackKind>(kind); fallback.phase = static_cast<ExecutionPhase>(phase); fallback.numerical_class = static_cast<NumericalClass>(numerical);
        model.fallbacks.push_back(fallback);
    }
    if (offset != bytes.size()) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
    auto validated = encode_semantic_model(model);
    if (auto* report = std::get_if<CompatibilityReport>(&validated)) return *report;
    if (std::get<std::vector<uint8_t>>(validated) != bytes) return wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
    return model;
}

Sha256Digest semantic_model_digest(const SemanticModel& model) {
    Sha256Digest digest;
    auto encoded = encode_semantic_model(model);
    if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) return digest;
    const auto& bytes = std::get<std::vector<uint8_t>>(encoded);
    digest.bytes = sha256(bytes.data(), bytes.size());
    return digest;
}

SemanticVectorResult semantic_embedding_lookup(const std::vector<float>& table,
                                               uint32_t vocabulary, uint32_t width,
                                               const std::vector<uint32_t>& token_ids,
                                               float scale) {
    size_t table_size = 0;
    if (!product(vocabulary, width, table_size) || table.size() != table_size ||
        !std::isfinite(scale) || scale <= 0 || !finite(table)) {
        return constraint_error();
    }
    std::vector<float> output;
    output.reserve(token_ids.size() * width);
    for (uint32_t token : token_ids) {
        if (token >= vocabulary) return runtime_error();
        for (uint32_t column = 0; column < width; ++column) {
            output.push_back(fp_mul(table[static_cast<size_t>(token) * width + column], scale));
        }
    }
    return output;
}

SemanticVectorResult semantic_rms_norm(const std::vector<float>& input,
                                       const std::vector<float>& weight, float epsilon) {
    if (input.empty() || (!weight.empty() && input.size() != weight.size())) return shape_error();
    if (!std::isfinite(epsilon) || epsilon < 0 || !finite(input) || !finite(weight)) {
        return constraint_error();
    }
    float lanes[16] = {};
    size_t index = 0;
    for (; index + 16 <= input.size(); index += 16) {
        for (size_t lane = 0; lane != 16; ++lane) lanes[lane] = fp_add(lanes[lane], fp_mul(input[index + lane], input[index + lane]));
    }
    float sum = 0.0f;
    for (float lane : lanes) sum = fp_add(sum, lane);
    for (; index != input.size(); ++index) sum = fp_add(sum, fp_mul(input[index], input[index]));
    float mean = fp32(sum / static_cast<float>(input.size()));
    float reciprocal_root = fp32(1.0f / std::sqrt(fp_add(mean, epsilon)));
    if (!std::isfinite(reciprocal_root)) return constraint_error();
    std::vector<float> output(input.size());
    for (size_t index = 0; index < input.size(); ++index) {
        const float scale = weight.empty() ? 1.0f : weight[index];
        output[index] = fp_mul(fp_mul(input[index], reciprocal_root), scale);
    }
    return output;
}

SemanticVectorResult semantic_linear(const std::vector<float>& input,
                                     const std::vector<float>& weight,
                                     uint32_t output_width, uint32_t input_width,
                                     const std::vector<float>& bias) {
    size_t weight_size = 0;
    if (!product(output_width, input_width, weight_size) || input.size() != input_width ||
        weight.size() != weight_size || (!bias.empty() && bias.size() != output_width)) {
        return shape_error();
    }
    if (!finite(input) || !finite(weight) || !finite(bias)) return constraint_error();
    std::vector<float> output(output_width);
    for (uint32_t output_index = 0; output_index < output_width; ++output_index) {
        float lanes[16] = {};
        uint32_t input_index = 0;
        for (; input_index + 16 <= input_width; input_index += 16) {
            for (uint32_t lane = 0; lane != 16; ++lane) {
                lanes[lane] = fp_add(lanes[lane],
                    fp_mul(input[input_index + lane], weight[static_cast<size_t>(output_index) * input_width + input_index + lane]));
            }
        }
        float accumulator = 0.0f;
        for (float lane : lanes) accumulator = fp_add(accumulator, lane);
        for (; input_index < input_width; ++input_index) {
            accumulator = fp_add(accumulator,
                                 fp_mul(input[input_index],
                                        weight[static_cast<size_t>(output_index) * input_width + input_index]));
        }
        if (!bias.empty()) accumulator = fp_add(accumulator, bias[output_index]);
        output[output_index] = accumulator;
    }
    return output;
}

SemanticVectorResult semantic_rope_half_split(const std::vector<float>& query,
                                              const std::vector<float>& key,
                                              uint32_t position, uint32_t rotary_dimension,
                                              float base, float scale,
                                              uint32_t frequency_dimension) {
    if (frequency_dimension == 0) frequency_dimension = rotary_dimension;
    if (query.size() != key.size() || query.empty() || rotary_dimension == 0 ||
        rotary_dimension > query.size() || rotary_dimension % 2 != 0 ||
        frequency_dimension < rotary_dimension || frequency_dimension % 2 != 0) {
        return shape_error();
    }
    if (!std::isfinite(base) || !std::isfinite(scale) || base <= 0 || scale <= 0 ||
        !finite(query) || !finite(key)) {
        return constraint_error();
    }
    auto rotate = [&](const std::vector<float>& input, std::vector<float>& output) {
        output = input;
        for (uint32_t pair = 0; pair < rotary_dimension / 2; ++pair) {
            float exponent = fp32(-2.0f * static_cast<float>(pair) / static_cast<float>(frequency_dimension));
            float angle = fp_mul(fp_mul(static_cast<float>(position), scale), std::pow(base, exponent));
            float cosine = fp32(std::cos(angle));
            float sine = fp32(std::sin(angle));
            float first = input[pair];
            float second = input[pair + rotary_dimension / 2];
            output[pair] = fp32(fp_mul(first, cosine) - fp_mul(second, sine));
            output[pair + rotary_dimension / 2] = fp_add(fp_mul(second, cosine), fp_mul(first, sine));
        }
    };
    std::vector<float> output;
    rotate(query, output);
    std::vector<float> rotated_key;
    rotate(key, rotated_key);
    output.insert(output.end(), rotated_key.begin(), rotated_key.end());
    return output;
}

SemanticVectorResult semantic_rope_interleaved(const std::vector<float>& query,
                                                const std::vector<float>& key,
                                                uint32_t position, uint32_t rotary_dimension,
                                                float base, float scale,
                                                uint32_t frequency_dimension) {
    if (frequency_dimension == 0) frequency_dimension = rotary_dimension;
    if (query.size() != key.size() || query.empty() || rotary_dimension == 0 ||
        rotary_dimension > query.size() || rotary_dimension % 2 != 0 ||
        frequency_dimension < rotary_dimension || frequency_dimension % 2 != 0) {
        return shape_error();
    }
    if (!std::isfinite(base) || !std::isfinite(scale) || base <= 0 || scale <= 0 ||
        !finite(query) || !finite(key)) {
        return constraint_error();
    }
    auto rotate = [&](const std::vector<float>& input, std::vector<float>& output) {
        output = input;
        for (uint32_t pair = 0; pair < rotary_dimension / 2; ++pair) {
            float exponent = fp32(-2.0f * static_cast<float>(pair) / static_cast<float>(frequency_dimension));
            float angle = fp_mul(fp_mul(static_cast<float>(position), scale), std::pow(base, exponent));
            float cosine = fp32(std::cos(angle));
            float sine = fp32(std::sin(angle));
            const uint32_t first_index = pair * 2;
            const uint32_t second_index = first_index + 1;
            float first = input[first_index];
            float second = input[second_index];
            output[first_index] = fp32(fp_mul(first, cosine) - fp_mul(second, sine));
            output[second_index] = fp_add(fp_mul(second, cosine), fp_mul(first, sine));
        }
    };
    std::vector<float> output;
    rotate(query, output);
    std::vector<float> rotated_key;
    rotate(key, rotated_key);
    output.insert(output.end(), rotated_key.begin(), rotated_key.end());
    return output;
}

SemanticVectorResult semantic_rope_multi_section_half_split(const std::vector<float>& query,
                                                             const std::vector<float>& key,
                                                             const std::array<uint32_t, 4>& positions,
                                                             const std::array<uint32_t, 4>& sections,
                                                             uint32_t rotary_dimension, float base, float scale,
                                                             uint32_t frequency_dimension) {
    if (frequency_dimension == 0) frequency_dimension = rotary_dimension;
    if (query.size() != key.size() || query.empty() || rotary_dimension == 0 ||
        rotary_dimension > query.size() || rotary_dimension % 2 != 0 ||
        frequency_dimension < rotary_dimension || frequency_dimension % 2 != 0) {
        return shape_error();
    }
    uint64_t total_sections = 0;
    for (uint32_t section : sections) total_sections += section;
    if (sections[0] == 0 || sections[1] == 0 || total_sections != rotary_dimension / 2 ||
        !std::isfinite(base) || !std::isfinite(scale) || base <= 0 || scale <= 0 ||
        !finite(query) || !finite(key)) {
        return constraint_error();
    }
    auto rotate = [&](const std::vector<float>& input, std::vector<float>& output) {
        output = input;
        for (uint32_t pair = 0; pair < rotary_dimension / 2; ++pair) {
            const uint32_t sector = pair % static_cast<uint32_t>(total_sections);
            const uint32_t axis = sector < sections[0] ? 0
                                : sector < sections[0] + sections[1] ? 1
                                : sector < sections[0] + sections[1] + sections[2] ? 2
                                : 3;
            const float exponent = fp32(-2.0f * static_cast<float>(pair) / static_cast<float>(frequency_dimension));
            const float angle = fp_mul(fp_mul(static_cast<float>(positions[axis]), scale), std::pow(base, exponent));
            const float cosine = fp32(std::cos(angle));
            const float sine = fp32(std::sin(angle));
            const float first = input[pair];
            const float second = input[pair + rotary_dimension / 2];
            output[pair] = fp32(fp_mul(first, cosine) - fp_mul(second, sine));
            output[pair + rotary_dimension / 2] = fp_add(fp_mul(second, cosine), fp_mul(first, sine));
        }
    };
    std::vector<float> output;
    rotate(query, output);
    std::vector<float> rotated_key;
    rotate(key, rotated_key);
    output.insert(output.end(), rotated_key.begin(), rotated_key.end());
    return output;
}

SemanticVectorResult semantic_l2_normalize(const std::vector<float>& input, float epsilon) {
    if (input.empty()) return shape_error();
    if (!std::isfinite(epsilon) || epsilon < 0.0f || !finite(input)) return constraint_error();
    float sum = 0.0f;
    for (float value : input) sum = fp_add(sum, fp_mul(value, value));
    float length = fp32(std::sqrt(fp_add(sum, epsilon)));
    if (!std::isfinite(length)) return constraint_error();
    if (length == 0.0f) return constraint_error();
    std::vector<float> output(input.size());
    for (size_t index = 0; index != input.size(); ++index) output[index] = fp32(input[index] / length);
    return output;
}

SemanticAxisSplitResult semantic_axis_split(const std::vector<float>& input,
                                            uint32_t row_width, uint32_t first_width) {
    if (row_width == 0 || first_width == 0 || first_width >= row_width) return constraint_error();
    if (input.empty() || input.size() % row_width != 0) return shape_error();
    if (!finite(input)) return constraint_error();
    const uint32_t second_width = row_width - first_width;
    SemanticAxisSplit output;
    output.first.reserve(input.size() / row_width * first_width);
    output.second.reserve(input.size() / row_width * second_width);
    for (size_t row = 0; row != input.size(); row += row_width) {
        output.first.insert(output.first.end(), input.begin() + static_cast<ptrdiff_t>(row),
                            input.begin() + static_cast<ptrdiff_t>(row + first_width));
        output.second.insert(output.second.end(), input.begin() + static_cast<ptrdiff_t>(row + first_width),
                             input.begin() + static_cast<ptrdiff_t>(row + row_width));
    }
    return output;
}

SemanticVectorResult semantic_concat_last_axis(const std::vector<float>& left,
                                               const std::vector<float>& right,
                                               uint32_t left_width, uint32_t right_width) {
    if (left_width == 0 || right_width == 0 || left.empty() || right.empty() ||
        left.size() % left_width != 0 || right.size() % right_width != 0 ||
        left.size() / left_width != right.size() / right_width) {
        return shape_error();
    }
    if (!finite(left) || !finite(right)) return constraint_error();
    const size_t rows = left.size() / left_width;
    std::vector<float> output;
    output.reserve(rows * (static_cast<size_t>(left_width) + right_width));
    for (size_t row = 0; row != rows; ++row) {
        const size_t left_start = row * left_width;
        const size_t right_start = row * right_width;
        output.insert(output.end(), left.begin() + static_cast<ptrdiff_t>(left_start),
                      left.begin() + static_cast<ptrdiff_t>(left_start + left_width));
        output.insert(output.end(), right.begin() + static_cast<ptrdiff_t>(right_start),
                      right.begin() + static_cast<ptrdiff_t>(right_start + right_width));
    }
    return output;
}

SemanticVectorResult semantic_depthwise_conv1d_silu_step(const std::vector<float>& input,
                                                          const std::vector<float>& weight,
                                                          uint32_t channels, uint32_t kernel,
                                                          SemanticConvState& state) {
    size_t weight_size = 0;
    size_t history_size = 0;
    if (!product(channels, kernel, weight_size) || kernel == 0 ||
        !product(channels, kernel - 1, history_size) || input.size() != channels ||
        weight.size() != weight_size || state.history.size() != history_size) {
        return shape_error();
    }
    if (!finite(input) || !finite(weight) || !finite(state.history)) return constraint_error();

    std::vector<float> output(channels);
    std::vector<float> history = state.history;
    for (uint32_t channel = 0; channel < channels; ++channel) {
        const size_t weight_base = static_cast<size_t>(channel) * kernel;
        const size_t history_base = static_cast<size_t>(channel) * (kernel - 1);
        float sum = 0.0f;
        for (uint32_t tap = 0; tap + 1 < kernel; ++tap) {
            sum = fp_add(sum, fp_mul(state.history[history_base + tap], weight[weight_base + tap]));
        }
        sum = fp_add(sum, fp_mul(input[channel], weight[weight_base + kernel - 1]));
        output[channel] = fp_mul(sum, sigmoid(sum));
        for (uint32_t tap = 0; tap + 2 < kernel; ++tap) {
            history[history_base + tap] = state.history[history_base + tap + 1];
        }
        if (kernel > 1) history[history_base + kernel - 2] = input[channel];
    }
    state.history = std::move(history);
    return output;
}

SemanticVectorResult semantic_gated_delta_net_step(const std::vector<float>& query,
                                                    const std::vector<float>& key,
                                                    const std::vector<float>& value,
                                                    const std::vector<float>& log_decay,
                                                    const std::vector<float>& beta,
                                                    uint32_t qk_heads, uint32_t value_heads,
                                                    uint32_t head_dimension,
                                                    SemanticGatedDeltaState& state) {
    size_t qk_size = 0;
    size_t value_size = 0;
    size_t matrix_per_head = 0;
    size_t matrix_size = 0;
    if (!product(qk_heads, head_dimension, qk_size) ||
        !product(value_heads, head_dimension, value_size) ||
        !product(head_dimension, head_dimension, matrix_per_head) ||
        !product(value_heads, static_cast<uint32_t>(matrix_per_head), matrix_size) ||
        qk_heads == 0 || value_heads == 0 || head_dimension == 0 || value_heads % qk_heads != 0 ||
        query.size() != qk_size || key.size() != qk_size || value.size() != value_size ||
        log_decay.size() != value_heads || beta.size() != value_heads || state.matrix.size() != matrix_size) {
        return shape_error();
    }
    if (!finite(query) || !finite(key) || !finite(value) || !finite(log_decay) || !finite(beta) ||
        !finite(state.matrix)) {
        return constraint_error();
    }
    for (uint32_t head = 0; head < value_heads; ++head) {
        if (log_decay[head] > 0.0f || beta[head] < 0.0f || beta[head] > 1.0f) return constraint_error();
    }

    const float scale = fp32(1.0f / std::sqrt(static_cast<float>(head_dimension)));
    std::vector<float> output(value_size);
    std::vector<float> matrix = state.matrix;
    for (uint32_t value_head = 0; value_head < value_heads; ++value_head) {
        const size_t qk_base = static_cast<size_t>(value_head % qk_heads) * head_dimension;
        const size_t value_base = static_cast<size_t>(value_head) * head_dimension;
        const size_t state_base = static_cast<size_t>(value_head) * matrix_per_head;
        const float decay = fp32(std::exp(log_decay[value_head]));
        std::vector<float> delta(head_dimension);
        for (uint32_t key_row = 0; key_row < head_dimension; ++key_row) {
            const size_t row_base = state_base + static_cast<size_t>(key_row) * head_dimension;
            for (uint32_t output_column = 0; output_column < head_dimension; ++output_column) {
                matrix[row_base + output_column] = fp_mul(matrix[row_base + output_column], decay);
            }
        }
        for (uint32_t output_column = 0; output_column < head_dimension; ++output_column) {
            float retrieved = 0.0f;
            for (uint32_t key_row = 0; key_row < head_dimension; ++key_row) {
                const size_t index = state_base + static_cast<size_t>(key_row) * head_dimension + output_column;
                retrieved = fp_add(retrieved, fp_mul(matrix[index], key[qk_base + key_row]));
            }
            delta[output_column] = fp_mul(fp32(value[value_base + output_column] - retrieved), beta[value_head]);
            if (!std::isfinite(delta[output_column])) return constraint_error();
        }
        for (uint32_t key_row = 0; key_row < head_dimension; ++key_row) {
            const size_t row_base = state_base + static_cast<size_t>(key_row) * head_dimension;
            for (uint32_t output_column = 0; output_column < head_dimension; ++output_column) {
                matrix[row_base + output_column] = fp_add(matrix[row_base + output_column],
                                                           fp_mul(key[qk_base + key_row], delta[output_column]));
            }
        }
        for (uint32_t output_column = 0; output_column < head_dimension; ++output_column) {
            float result = 0.0f;
            for (uint32_t key_row = 0; key_row < head_dimension; ++key_row) {
                const size_t index = state_base + static_cast<size_t>(key_row) * head_dimension + output_column;
                result = fp_add(result, fp_mul(matrix[index], query[qk_base + key_row]));
            }
            output[value_base + output_column] = fp_mul(result, scale);
            if (!std::isfinite(output[value_base + output_column])) return constraint_error();
        }
    }
    state.matrix = std::move(matrix);
    return output;
}

SemanticVectorResult semantic_gated_attention_output(const std::vector<float>& attention,
                                                      const std::vector<float>& gate) {
    if (attention.empty() || attention.size() != gate.size()) return shape_error();
    if (!finite(attention) || !finite(gate)) return constraint_error();
    std::vector<float> output(attention.size());
    for (size_t index = 0; index < attention.size(); ++index) {
        output[index] = fp_mul(attention[index], sigmoid(gate[index]));
    }
    return output;
}

SemanticVectorResult semantic_gated_rms_norm(const std::vector<float>& input,
                                             const std::vector<float>& weight,
                                             const std::vector<float>& gate, float epsilon) {
    if (gate.size() != input.size() || !finite(gate)) return shape_error();
    SemanticVectorResult normalized = semantic_rms_norm(input, weight, epsilon);
    if (const auto* report = std::get_if<CompatibilityReport>(&normalized)) return *report;
    std::vector<float> output = std::get<std::vector<float>>(std::move(normalized));
    for (size_t index = 0; index < output.size(); ++index) {
        output[index] = fp_mul(output[index], fp_mul(gate[index], sigmoid(gate[index])));
    }
    return output;
}

SemanticVectorResult semantic_causal_attention(const std::vector<float>& query,
                                               const std::vector<float>& key,
                                               const std::vector<float>& value,
                                               uint32_t rows, uint32_t query_heads,
                                               uint32_t kv_heads, uint32_t head_dimension,
                                               float scale, SemanticKvState& state) {
    return semantic_causal_attention_windowed(query, key, value, rows, query_heads, kv_heads, head_dimension,
                                              scale, AttentionWindowKind::Global, 0, state);
}

SemanticVectorResult semantic_causal_attention_windowed(const std::vector<float>& query,
                                                        const std::vector<float>& key,
                                                        const std::vector<float>& value,
                                                        uint32_t rows, uint32_t query_heads,
                                                        uint32_t kv_heads, uint32_t head_dimension,
                                                        float scale, AttentionWindowKind window,
                                                        uint32_t window_tokens, SemanticKvState& state) {
    size_t query_row = 0;
    size_t kv_row = 0;
    if (!product(query_heads, head_dimension, query_row) || !product(kv_heads, head_dimension, kv_row) ||
        rows == 0 || query_heads == 0 || kv_heads == 0 || head_dimension == 0 ||
        query_heads % kv_heads != 0 || query.size() != static_cast<size_t>(rows) * query_row ||
        key.size() != static_cast<size_t>(rows) * kv_row || value.size() != static_cast<size_t>(rows) * kv_row ||
        state.key.size() != static_cast<size_t>(state.tokens) * kv_row ||
        state.value.size() != static_cast<size_t>(state.tokens) * kv_row) {
        return query_heads % kv_heads ? constraint_error() : shape_error();
    }
    if ((window != AttentionWindowKind::Global && window != AttentionWindowKind::Sliding) ||
        (window == AttentionWindowKind::Global && window_tokens != 0) ||
        (window == AttentionWindowKind::Sliding && window_tokens == 0) ||
        !std::isfinite(scale) || scale <= 0.0f || !finite(query) || !finite(key) || !finite(value) ||
        !finite(state.key) || !finite(state.value)) {
        return constraint_error();
    }

    SemanticKvState candidate = state;
    candidate.key.insert(candidate.key.end(), key.begin(), key.end());
    candidate.value.insert(candidate.value.end(), value.begin(), value.end());
    candidate.tokens += rows;
    std::vector<float> output(static_cast<size_t>(rows) * query_row);
    uint32_t group = query_heads / kv_heads;
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t total_source_count = state.tokens + row + 1;
        const uint32_t first_source = window == AttentionWindowKind::Sliding && total_source_count > window_tokens
            ? total_source_count - window_tokens : 0;
        const uint32_t source_count = total_source_count - first_source;
        for (uint32_t query_head = 0; query_head < query_heads; ++query_head) {
            uint32_t kv_head = query_head / group;
            std::vector<float> scores(source_count);
            float maximum = -std::numeric_limits<float>::infinity();
            for (uint32_t local_source = 0; local_source < source_count; ++local_source) {
                const uint32_t source = first_source + local_source;
                float accumulator = 0.0f;
                float lanes[16] = {};
                uint32_t dimension = 0;
                for (; dimension + 16 <= head_dimension; dimension += 16) {
                    for (uint32_t lane = 0; lane != 16; ++lane) {
                        lanes[lane] = fp_add(lanes[lane],
                            fp_mul(query[(static_cast<size_t>(row) * query_heads + query_head) * head_dimension + dimension + lane],
                                   candidate.key[(static_cast<size_t>(source) * kv_heads + kv_head) * head_dimension + dimension + lane]));
                    }
                }
                for (float lane : lanes) accumulator = fp_add(accumulator, lane);
                for (; dimension < head_dimension; ++dimension) {
                    accumulator = fp_add(accumulator,
                        fp_mul(query[(static_cast<size_t>(row) * query_heads + query_head) * head_dimension + dimension],
                               candidate.key[(static_cast<size_t>(source) * kv_heads + kv_head) * head_dimension + dimension]));
                }
                scores[local_source] = fp_mul(accumulator, scale);
                maximum = std::max(maximum, scores[local_source]);
            }
            float normalizer = 0.0f;
            for (float score : scores) normalizer = fp_add(normalizer, fp32(std::exp(fp32(score - maximum))));
            if (!std::isfinite(normalizer) || normalizer <= 0) return constraint_error();
            for (uint32_t dimension = 0; dimension < head_dimension; ++dimension) {
                float accumulator = 0.0f;
                for (uint32_t local_source = 0; local_source < source_count; ++local_source) {
                    const uint32_t source = first_source + local_source;
                    float probability = fp32(fp32(std::exp(fp32(scores[local_source] - maximum))) / normalizer);
                    accumulator = fp_add(accumulator,
                        fp_mul(probability,
                               candidate.value[(static_cast<size_t>(source) * kv_heads + kv_head) * head_dimension + dimension]));
                }
                output[(static_cast<size_t>(row) * query_heads + query_head) * head_dimension + dimension] = accumulator;
            }
        }
    }
    if (window == AttentionWindowKind::Sliding && candidate.tokens > window_tokens) {
        const uint32_t removed = candidate.tokens - window_tokens;
        const size_t removed_values = static_cast<size_t>(removed) * kv_row;
        candidate.key.erase(candidate.key.begin(), candidate.key.begin() + removed_values);
        candidate.value.erase(candidate.value.begin(), candidate.value.begin() + removed_values);
        candidate.tokens = window_tokens;
    }
    state = std::move(candidate);
    return output;
}

SemanticVectorResult semantic_swiglu(const std::vector<float>& gate,
                                     const std::vector<float>& up) {
    if (gate.size() != up.size()) return shape_error();
    if (!finite(gate) || !finite(up)) return constraint_error();
    std::vector<float> output(gate.size());
    for (size_t index = 0; index < gate.size(); ++index) {
        float silu = fp32(gate[index] / fp_add(1.0f, fp32(std::exp(-gate[index]))));
        output[index] = fp_mul(silu, up[index]);
    }
    return output;
}

SemanticRouterResultOrError semantic_router_top_k(const std::vector<float>& scores,
                                                   const RouterTopKPayload& payload) {
    if (!valid_router_payload(payload) || scores.size() != payload.expert_count || !finite(scores)) {
        return constraint_error();
    }
    if (payload.score_domain == RouterScoreDomain::Probabilities) {
        float sum = 0.0f;
        for (float score : scores) {
            if (score < 0.0f) return constraint_error();
            sum = fp_add(sum, score);
        }
        if (!std::isfinite(sum) || sum <= 0.0f) return constraint_error();
    }
    std::vector<uint32_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t left, uint32_t right) {
        return scores[left] != scores[right] ? scores[left] > scores[right] : left < right;
    });
    SemanticRouterResult result;
    result.ids.assign(order.begin(), order.begin() + payload.selected_count);
    result.weights.resize(payload.selected_count);
    if (payload.normalization_order == RouterNormalizationOrder::SelectThenNormalize) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (uint32_t id : result.ids) maximum = std::max(maximum, scores[id]);
        float normalizer = 0.0f;
        for (uint32_t id : result.ids) normalizer = fp_add(normalizer, fp32(std::exp(fp32(scores[id] - maximum))));
        if (!std::isfinite(normalizer) || normalizer <= 0.0f) return constraint_error();
        for (size_t index = 0; index != result.ids.size(); ++index) {
            result.weights[index] = fp32(fp32(std::exp(fp32(scores[result.ids[index]] - maximum))) / normalizer);
        }
        return result;
    }
    std::vector<float> normalized = scores;
    if (payload.score_domain == RouterScoreDomain::Logits) {
        float maximum = *std::max_element(scores.begin(), scores.end());
        float normalizer = 0.0f;
        for (float score : scores) normalizer = fp_add(normalizer, fp32(std::exp(fp32(score - maximum))));
        if (!std::isfinite(normalizer) || normalizer <= 0.0f) return constraint_error();
        for (size_t index = 0; index != normalized.size(); ++index) {
            normalized[index] = fp32(fp32(std::exp(fp32(scores[index] - maximum))) / normalizer);
        }
    }
    for (size_t index = 0; index != result.ids.size(); ++index) result.weights[index] = normalized[result.ids[index]];
    if (payload.selected_weight_normalization ==
        SelectedWeightNormalization::RenormalizeSelectedProbabilities) {
        float selected_sum = 0.0f;
        for (float weight : result.weights) selected_sum = fp_add(selected_sum, weight);
        if (!std::isfinite(selected_sum) || selected_sum <= 0.0f) return constraint_error();
        for (float& weight : result.weights) weight = fp32(weight / selected_sum);
    }
    return result;
}

SemanticVectorResult semantic_gated_activation(const std::vector<float>& gate,
                                               const std::vector<float>& up,
                                               ActivationKind activation) {
    if (gate.size() != up.size()) return shape_error();
    if (!finite(gate) || !finite(up) || (activation != ActivationKind::Silu && activation != ActivationKind::GeluTanh)) {
        return constraint_error();
    }
    std::vector<float> output(gate.size());
    for (size_t index = 0; index != gate.size(); ++index) {
        float activated = 0.0f;
        if (activation == ActivationKind::Silu) {
            activated = fp32(gate[index] / fp_add(1.0f, fp32(std::exp(-gate[index]))));
        } else {
            const float cube = fp_mul(fp_mul(gate[index], gate[index]), gate[index]);
            const float inner = fp_mul(0.7978845608028654f, fp_add(gate[index], fp_mul(0.044715f, cube)));
            activated = fp_mul(0.5f, fp_mul(gate[index], fp_add(1.0f, fp32(std::tanh(inner)))));
        }
        output[index] = fp_mul(activated, up[index]);
    }
    return output;
}

SemanticVectorResult semantic_weighted_expert_reduce(const std::vector<float>& expert_outputs,
                                                      const std::vector<float>& route_weights,
                                                      const std::vector<float>& expert_scales,
                                                      uint32_t output_width,
                                                      const WeightedExpertReducePayload& payload) {
    if (output_width == 0 || !finite(expert_outputs) || !finite(route_weights) ||
        payload.association != ExpertReduceAssociation::SelectedOrderLeftToRight ||
        payload.accumulation_type != ScalarType::F32 || route_weights.empty() ||
        expert_outputs.size() != static_cast<size_t>(output_width) * route_weights.size()) {
        return shape_error();
    }
    if ((payload.scale_source == ExpertScaleSource::None && !expert_scales.empty()) ||
        (payload.scale_source == ExpertScaleSource::PerExpertTensor &&
         (expert_scales.size() != route_weights.size() || !finite(expert_scales))) ||
        (payload.scale_source != ExpertScaleSource::None && payload.scale_source != ExpertScaleSource::PerExpertTensor)) {
        return constraint_error();
    }
    std::vector<float> output(output_width, 0.0f);
    for (size_t expert = 0; expert != route_weights.size(); ++expert) {
        const float scale = payload.scale_source == ExpertScaleSource::PerExpertTensor ? expert_scales[expert] : 1.0f;
        for (uint32_t column = 0; column != output_width; ++column) {
            output[column] = fp_add(output[column], fp_mul(expert_outputs[expert * output_width + column],
                                                            fp_mul(route_weights[expert], scale)));
        }
    }
    return output;
}

SemanticVectorResult semantic_scale(const std::vector<float>& input, float scale) {
    if (!std::isfinite(scale) || !finite(input)) return constraint_error();
    std::vector<float> output(input.size());
    for (size_t index = 0; index != input.size(); ++index) output[index] = fp_mul(input[index], scale);
    return output;
}

SemanticVectorResult semantic_tanh_softcap(const std::vector<float>& input, float cap) {
    if (!std::isfinite(cap) || cap <= 0.0f || !finite(input)) return constraint_error();
    std::vector<float> output(input.size());
    for (size_t index = 0; index != input.size(); ++index) output[index] = fp_mul(cap, fp32(std::tanh(fp32(input[index] / cap))));
    return output;
}

SemanticVectorResult semantic_add(const std::vector<float>& lhs,
                                  const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size()) return shape_error();
    if (!finite(lhs) || !finite(rhs)) return constraint_error();
    std::vector<float> output(lhs.size());
    for (size_t index = 0; index < lhs.size(); ++index) output[index] = fp_add(lhs[index], rhs[index]);
    return output;
}

} // namespace Laplace
