#include "execution_plan.h"

#include "compat_rule.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

namespace Laplace {

namespace {

CompatibilityReport plan_error(CompatibilityError code) {
    CompatibilityReport report = package_report(code);
    report.stage = CompatibilityStage::Plan;
    return report;
}

template<class T>
bool matches(const ExactOrAny<T>& pattern, T value) {
    return pattern.any || pattern.exact == value;
}

uint64_t objective_cost(const KernelDescriptor& descriptor, RuntimeObjective objective) {
    return objective == RuntimeObjective::Latency ? descriptor.cost.latency : descriptor.cost.throughput;
}

KernelImplementation scalar_implementation(OperatorKind operation) {
    switch (operation) {
    case OperatorKind::EmbeddingLookup: return KernelImplementation::ScalarEmbedding;
    case OperatorKind::RmsNorm: return KernelImplementation::ScalarRmsNorm;
    case OperatorKind::Linear: return KernelImplementation::ScalarLinear;
    case OperatorKind::Rope: return KernelImplementation::ScalarRope;
    case OperatorKind::CausalAttention: return KernelImplementation::ScalarCausalAttention;
    case OperatorKind::SwiGlu: return KernelImplementation::ScalarSwiGlu;
    case OperatorKind::Add: return KernelImplementation::ScalarAdd;
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
        return KernelImplementation::Unavailable;
    }
    return KernelImplementation::Unavailable;
}

KernelQuery query_for(const SemanticModel& model, const SemanticOperator& op, ExecutionPhase phase);
bool match_dense_graph_impl(const SemanticModel& model, const SemanticLayer& layer,
                            DenseGraphWitness& witness);

bool implementation_matches(OperatorKind operation, KernelImplementation implementation) {
    const KernelImplementation scalar = scalar_implementation(operation);
    if (scalar != KernelImplementation::Unavailable && implementation == scalar) return true;
    if (operation == OperatorKind::CausalAttention &&
        (implementation == KernelImplementation::MetalDenseToken ||
         implementation == KernelImplementation::MetalDensePrefillBatch)) return true;
    if (operation == OperatorKind::GatedDeltaNet && implementation == KernelImplementation::MetalRecurrentToken) return true;
#if defined(LAPLACE_TESTING)
    return operation == OperatorKind::EmbeddingLookup && implementation == KernelImplementation::ScalarEmbeddingUnitOffset;
#else
    return false;
#endif
}

const SemanticLayer* layer_for_operator(const SemanticModel& model, uint32_t operator_id) {
    for (const SemanticLayer& layer : model.layers) {
        if (layer.first_operator > model.operators.size() || layer.operator_count > model.operators.size() - layer.first_operator) continue;
        for (uint32_t index = 0; index != layer.operator_count; ++index) {
            if (model.operators[layer.first_operator + index].id == operator_id) return &layer;
        }
    }
    return nullptr;
}

const SemanticOperator* layer_operator(const SemanticModel& model, const SemanticLayer& layer,
                                      OperatorKind kind, TensorRole role, bool expert = false) {
    const SemanticOperator* found = nullptr;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind != kind) continue;
        bool carries_role = false;
        for (uint32_t tensor_id : op.tensors) {
            carries_role |= tensor_id < model.tensors.size() && model.tensors[tensor_id].role == role &&
                            (model.tensors[tensor_id].expert_axis.kind == ExpertAxisKind::ExpertBank) == expert;
        }
        if (!carries_role) continue;
        if (found) return nullptr;
        found = &op;
    }
    return found;
}

const SemanticTensor* role_tensor(const SemanticModel& model, const SemanticOperator& op, TensorRole role,
                                  bool expert = false) {
    const SemanticTensor* found = nullptr;
    for (uint32_t tensor_id : op.tensors) {
        if (tensor_id >= model.tensors.size() || model.tensors[tensor_id].role != role ||
            (model.tensors[tensor_id].expert_axis.kind == ExpertAxisKind::ExpertBank) != expert) continue;
        if (found) return nullptr;
        found = &model.tensors[tensor_id];
    }
    return found;
}

bool exact_dense_tensor(const SemanticModel& model, const SemanticOperator& op, TensorRole role,
                        ScalarType storage, bool optional = false) {
    const SemanticTensor* tensor = role_tensor(model, op, role);
    if (!tensor) return optional;
    return tensor->logical_type == ScalarType::F32 && tensor->layout.kind == PhysicalLayoutKind::ContiguousRowMajor &&
           tensor->quantization.kind == QuantizationKind::None && tensor->planes.size() == 1 &&
           tensor->planes[0].kind == PlaneKind::Values && tensor->planes[0].storage_type == storage &&
           tensor->planes[0].alignment != 0;
}

struct MetalBlockedFormat {
    uint32_t format = 0;
    uint32_t elements = 0;
    uint32_t bytes = 0;
    QuantizationKind quantization = QuantizationKind::BlockedAffine;
};

constexpr std::array kMetalBlockedFormats = {
    MetalBlockedFormat{MetalWeightFormatIQ2XXS, 256, 66, QuantizationKind::Codebook},
    MetalBlockedFormat{MetalWeightFormatQ4_0, 32, 18},
    MetalBlockedFormat{MetalWeightFormatQ4K, 256, 144},
    MetalBlockedFormat{MetalWeightFormatQ5_0, 32, 22},
    MetalBlockedFormat{MetalWeightFormatQ6K, 256, 210},
    MetalBlockedFormat{MetalWeightFormatQ8_0, 32, 34},
    MetalBlockedFormat{MetalWeightFormatQ2K, 256, 84},
};

const MetalBlockedFormat* metal_blocked_format(uint32_t format) {
    const auto found = std::find_if(kMetalBlockedFormats.begin(), kMetalBlockedFormats.end(), [&](const MetalBlockedFormat& candidate) {
        return candidate.format == format;
    });
    return found == kMetalBlockedFormats.end() ? nullptr : &*found;
}

bool metal_dense_matrix_format(const SemanticModel& model, const SemanticOperator& op, TensorRole role,
                               uint32_t& format) {
    const SemanticTensor* tensor = role_tensor(model, op, role);
    if (!tensor || tensor->logical_type != ScalarType::F32 || tensor->dimensions.size() != 2 ||
        tensor->planes.empty()) return false;
    // The affine capability is a complete physical contract, not a hint from
    // an importer or a tensor/role name.  Column-grouped tensors preserve the
    // importer [K,N] dimensions and expose an output-group view with row
    // stride K.  The legacy grouped-affine contract keeps its existing stride.
    const bool column_grouped =
        tensor->layout.kind == PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip;
    const bool grouped_affine_layout =
        tensor->layout.kind == PhysicalLayoutKind::GroupedAffine || column_grouped;
    if (tensor->planes.size() == 3 && grouped_affine_layout &&
        tensor->layout.packing == PackingKind::LsbBitPacked && tensor->layout.rank == 2 &&
        tensor->layout.block_rank == 1 && tensor->layout.axis_order[0] == 1 &&
        tensor->layout.axis_order[1] == 0 && tensor->layout.strides[0] == 1 &&
        tensor->layout.strides[1] == (column_grouped
            ? tensor->dimensions[0].constant_or_symbol
            : tensor->dimensions[1].constant_or_symbol) &&
        tensor->layout.flags == 0 && tensor->flags == 0 &&
        tensor->expert_axis == ExpertAxis{} &&
        tensor->layout.block_elements == 256 && tensor->layout.block_bytes == 64 &&
        tensor->quantization.kind == QuantizationKind::BlockedAffine &&
        tensor->quantization.accumulation_type == ScalarType::F32 &&
        tensor->quantization.scale_type == ScalarType::F16 &&
        tensor->quantization.zero_type == static_cast<ScalarType>(0) &&
        tensor->quantization.bias_type == ScalarType::F16 &&
        tensor->quantization.block_elements == 256 && tensor->quantization.block_bytes == 64 &&
        tensor->quantization.group_size == 256 && tensor->quantization.required_plane_mask == 7 &&
        tensor->quantization.flags == 0 &&
        tensor->dimensions[0].kind == DimensionKind::Constant &&
        tensor->dimensions[1].kind == DimensionKind::Constant &&
        tensor->dimensions[0].constant_or_symbol != 0 &&
        tensor->dimensions[1].constant_or_symbol != 0) {
        const uint64_t n = tensor->dimensions[0].constant_or_symbol;
        const uint64_t k = tensor->dimensions[1].constant_or_symbol;
        const bool shape_supported = column_grouped
            ? n % 256 == 0
            : (k % 256 == 0 && k % 512 == 0 && n % 8 == 0);
        uint64_t elements = 0;
        uint64_t blocks = 0;
        if (shape_supported && n <= std::numeric_limits<uint64_t>::max() / k &&
            (elements = n * k) != 0 && (blocks = elements / 256) != 0 &&
            blocks <= UINT64_MAX / 2) {
            const TensorPlane* values = nullptr;
            const TensorPlane* scales = nullptr;
            const TensorPlane* biases = nullptr;
            const ScalarType expected_value_type = column_grouped
                ? ScalarType::U8
                : ScalarType::U32;
            for (const TensorPlane& candidate : tensor->planes) {
                if (candidate.alignment != 128 || candidate.offset % 128 != 0 || candidate.flags != 0) return false;
                if (candidate.kind == PlaneKind::Values && candidate.storage_type == expected_value_type)
                    values = values ? nullptr : &candidate;
                else if (candidate.kind == PlaneKind::Scales && candidate.storage_type == ScalarType::F16)
                    scales = scales ? nullptr : &candidate;
                else if (candidate.kind == PlaneKind::Biases && candidate.storage_type == ScalarType::F16)
                    biases = biases ? nullptr : &candidate;
                else return false;
            }
            if (values && scales && biases && values->length == elements / 4 &&
                scales->length == blocks * 2 && biases->length == blocks * 2) {
                format = tensor->layout.kind == PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip
                    ? MetalWeightFormatColumnGroupedAffineUInt2Skip256
                    : MetalWeightFormatAffineUInt2_256;
                return true;
            }
        }
    }
    if (tensor->planes.size() != 1 || tensor->planes[0].kind != PlaneKind::Values ||
        tensor->planes[0].alignment == 0) return false;
    const TensorPlane& plane = tensor->planes[0];
    if (tensor->layout.kind == PhysicalLayoutKind::ContiguousRowMajor &&
        tensor->quantization.kind == QuantizationKind::None && plane.storage_type == ScalarType::F16) {
        format = MetalWeightFormatF16;
        return true;
    }
    if (tensor->layout.kind != PhysicalLayoutKind::GgufBlocked || tensor->layout.packing != PackingKind::Gguf ||
        tensor->layout.block_rank != 1 ||
        (tensor->quantization.kind != QuantizationKind::BlockedAffine &&
         tensor->quantization.kind != QuantizationKind::Codebook) ||
        plane.storage_type != ScalarType::U8 || tensor->quantization.required_plane_mask != 1) return false;
    uint64_t elements = 1;
    for (const Dimension& dimension : tensor->dimensions) {
        if (dimension.kind != DimensionKind::Constant || dimension.constant_or_symbol == 0 ||
            dimension.constant_or_symbol > std::numeric_limits<uint64_t>::max() / elements) return false;
        elements *= dimension.constant_or_symbol;
    }
    for (const MetalBlockedFormat& candidate : kMetalBlockedFormats) {
        if (tensor->quantization.kind != candidate.quantization ||
            tensor->layout.block_elements != candidate.elements || tensor->layout.block_bytes != candidate.bytes ||
            tensor->quantization.block_elements != candidate.elements || tensor->quantization.block_bytes != candidate.bytes ||
            tensor->quantization.group_size != (candidate.quantization == QuantizationKind::Codebook ? 32u : candidate.elements) ||
            elements % candidate.elements != 0 ||
            elements / candidate.elements > std::numeric_limits<uint64_t>::max() / candidate.bytes ||
            plane.length != elements / candidate.elements * candidate.bytes) continue;
        format = candidate.format;
        return true;
    }
    return false;
}

const SemanticState* state_with_id(const SemanticModel& model, uint32_t id) {
    const SemanticState* found = nullptr;
    for (const SemanticState& state : model.states) {
        if (state.id != id) continue;
        if (found) return nullptr;
        found = &state;
    }
    return found;
}

bool dense_kv_states(const SemanticModel& model, const SemanticOperator& attention) {
    if (attention.states.size() != 2) return false;
    const SemanticState* key = nullptr;
    const SemanticState* value = nullptr;
    for (uint32_t id : attention.states) {
        const SemanticState* state = state_with_id(model, id);
        if (!state || state->formats.size() != 1) return false;
        if (state->kind == StateKind::KeyCache) key = key ? nullptr : state;
        if (state->kind == StateKind::ValueCache) value = value ? nullptr : state;
    }
    return key && value && key->update_kind == StateUpdateKind::AppendKey &&
           value->update_kind == StateUpdateKind::AppendValue &&
           key->formats[0].kind == StateFormatKind::GlobalContiguous &&
           value->formats[0].kind == StateFormatKind::GlobalContiguous &&
           key->formats[0].logical_type == ScalarType::F32 && value->formats[0].logical_type == ScalarType::F32 &&
           key->formats[0].encoded_type == ScalarType::F32 && value->formats[0].encoded_type == ScalarType::F32 &&
           key->formats[0].encoded_domain == TransformDomain::RopeApplied &&
           value->formats[0].encoded_domain == TransformDomain::Untransformed &&
           key->formats[0].codec == CodecKind::Fp32 && value->formats[0].codec == CodecKind::Fp32 &&
           key->formats[0].alignment == 64 && value->formats[0].alignment == 64;
}

bool canonical_attention_window_is_global(const SemanticModel& model, std::string& detail) {
    for (const SemanticOperator& op : model.operators) {
        if (op.kind != OperatorKind::CausalAttention) continue;
        const auto* payload = std::get_if<CausalAttentionPayload>(&op.payload);
        if (!payload) {
            detail = "canonical Metal attention window payload is invalid at operator " +
                     std::to_string(op.id);
            return false;
        }
        if (payload->window != AttentionWindowKind::Global || payload->window_tokens != 0) {
            detail = "canonical Metal attention requires a global window: operator " +
                     std::to_string(op.id) + " declares " +
                     (payload->window == AttentionWindowKind::Sliding ? "sliding" : "invalid") +
                     " window of " + std::to_string(payload->window_tokens) + " tokens";
            return false;
        }
    }
    return true;
}

bool dense_metal_layer(const SemanticModel& model, const SemanticLayer& layer, const SemanticOperator*& attention,
                       uint32_t* weight_format = nullptr, bool allow_moe_tail = false) {
    if (layer.first_operator > model.operators.size() || layer.operator_count > model.operators.size() - layer.first_operator) return false;
    const SemanticOperator* attn_norm = layer_operator(model, layer, OperatorKind::RmsNorm, TensorRole::AttentionNormWeight);
    const SemanticOperator* query = layer_operator(model, layer, OperatorKind::Linear, TensorRole::QueryWeight);
    const SemanticOperator* query_gate = layer_operator(model, layer, OperatorKind::Linear, TensorRole::AttentionQueryGateWeight);
    const SemanticOperator* key = layer_operator(model, layer, OperatorKind::Linear, TensorRole::KeyWeight);
    const SemanticOperator* value = layer_operator(model, layer, OperatorKind::Linear, TensorRole::ValueWeight);
    const SemanticOperator* output = layer_operator(model, layer, OperatorKind::Linear, TensorRole::AttentionOutputWeight);
    const SemanticOperator* query_norm = layer_operator(model, layer, OperatorKind::RmsNorm, TensorRole::AttentionQueryNormWeight);
    const SemanticOperator* key_norm = layer_operator(model, layer, OperatorKind::RmsNorm, TensorRole::AttentionKeyNormWeight);
    const SemanticOperator* ffn_norm = layer_operator(model, layer, OperatorKind::RmsNorm, TensorRole::FfnNormWeight);
    const SemanticOperator* gate = layer_operator(model, layer, OperatorKind::Linear, TensorRole::FfnGateWeight);
    const SemanticOperator* up = layer_operator(model, layer, OperatorKind::Linear, TensorRole::FfnUpWeight);
    const SemanticOperator* down = layer_operator(model, layer, OperatorKind::Linear, TensorRole::FfnDownWeight);
    const SemanticOperator* rope = nullptr;
    const SemanticOperator* swiglu = nullptr;
    std::vector<const SemanticOperator*> splits;
    const SemanticOperator* gated_attention = nullptr;
    uint32_t gated_attention_count = 0;
    attention = nullptr;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind == OperatorKind::Rope) rope = rope ? nullptr : &op;
        if (op.kind == OperatorKind::SwiGlu) swiglu = swiglu ? nullptr : &op;
        // Classify splits from their typed producer edge below.  Operator
        // order is deliberately irrelevant: a routed expert split may appear
        // before the attention split in a valid serialized graph.
        if (op.kind == OperatorKind::AxisSplit) splits.push_back(&op);
        if (op.kind == OperatorKind::GatedAttention) {
            gated_attention = gated_attention_count++ == 0 ? &op : nullptr;
        }
        if (op.kind == OperatorKind::CausalAttention) {
            attention = attention ? nullptr : &op;
        }
    }
    const bool fused_query_gate = query_gate != nullptr;
    const SemanticOperator* split = nullptr;
    uint32_t split_count = 0;
    if (fused_query_gate) {
        for (const SemanticOperator* candidate : splits) {
            if (candidate->inputs != query_gate->outputs) continue;
            if (split) {
                split_count = 2;
                break;
            }
            split = candidate;
            split_count = 1;
        }
    } else if (!allow_moe_tail) {
        split_count = splits.size() > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(splits.size());
    }
    const SemanticOperator* query_projection = fused_query_gate ? query_gate : query;
    const auto* rope_payload = rope ? std::get_if<RopePayload>(&rope->payload) : nullptr;
    const auto* attention_payload = attention ? std::get_if<CausalAttentionPayload>(&attention->payload) : nullptr;
    const auto* swiglu_payload = swiglu ? std::get_if<SwiGluPayload>(&swiglu->payload) : nullptr;
    const float rope_base = rope_payload ? std::bit_cast<float>(rope_payload->base_f32_bits) : 0.0f;
    const float rope_scale = rope_payload ? std::bit_cast<float>(rope_payload->scale_f32_bits) : 0.0f;
    const bool multi_section_rope = rope_payload && rope_payload->pairing == RopePairing::MultiSectionHalfSplit;
    if (!attn_norm || !query_projection || !key || !value || !output || !ffn_norm ||
        !gate || !up || !down || !rope_payload || !attention_payload || !swiglu_payload ||
        (swiglu_payload->activation != ActivationKind::Silu &&
         !(allow_moe_tail && swiglu_payload->activation == ActivationKind::GeluTanh)) ||
        (rope_payload->pairing != RopePairing::HalfSplit && rope_payload->pairing != RopePairing::Interleaved &&
         rope_payload->pairing != RopePairing::MultiSectionHalfSplit) ||
        !rope_payload->position_from_cursor ||
        !std::isfinite(rope_base) || rope_base <= 0.0f ||
        !std::isfinite(rope_scale) || rope_scale <= 0.0f ||
        rope_payload->rotary_dimension == 0 ||
        rope_payload->rotary_dimension > attention_payload->head_dimension ||
        (rope_payload->rotary_dimension % 2) != 0 ||
        attention_payload->mask != AttentionMask::Causal || attention_payload->cache_policy != CachePolicy::Global ||
        attention_payload->window != AttentionWindowKind::Global || attention_payload->window_tokens != 0 ||
        attention_payload->head_dimension < 32 || attention_payload->head_dimension > 512 ||
        attention_payload->head_dimension % 16 != 0 ||
        attention_payload->value_source != ValueSource::SeparateProjection ||
        !dense_kv_states(model, *attention) ||
        !exact_dense_tensor(model, *attn_norm, TensorRole::AttentionNormWeight, ScalarType::F32) ||
        (!fused_query_gate && !exact_dense_tensor(model, *query, TensorRole::QueryBias, ScalarType::F32, true)) ||
        !exact_dense_tensor(model, *key, TensorRole::KeyBias, ScalarType::F32, true) ||
        !exact_dense_tensor(model, *value, TensorRole::ValueBias, ScalarType::F32, true) ||
        !exact_dense_tensor(model, *ffn_norm, TensorRole::FfnNormWeight, ScalarType::F32)) {
        return false;
    }
    if (multi_section_rope) {
        uint64_t total_sections = 0;
        for (uint32_t section : rope_payload->position_sections) total_sections += section;
        if (rope_payload->position_sections[0] == 0 || rope_payload->position_sections[1] == 0 ||
            total_sections != rope_payload->rotary_dimension / 2) return false;
    }
    if ((query_norm == nullptr) != (key_norm == nullptr)) return false;
    if (query_norm && !fused_query_gate &&
        (!exact_dense_tensor(model, *query_norm, TensorRole::AttentionQueryNormWeight, ScalarType::F32) ||
         !exact_dense_tensor(model, *key_norm, TensorRole::AttentionKeyNormWeight, ScalarType::F32) ||
         query->outputs.size() != 1 || key->outputs.size() != 1 || query_norm->inputs.size() != 1 ||
         key_norm->inputs.size() != 1 || query_norm->outputs.size() != 1 || key_norm->outputs.size() != 1 ||
         rope->inputs.size() != 2 || query_norm->inputs[0] != query->outputs[0] ||
         key_norm->inputs[0] != key->outputs[0] || rope->inputs[0] != query_norm->outputs[0] ||
         rope->inputs[1] != key_norm->outputs[0])) return false;
    if (fused_query_gate) {
        const auto* split_payload = split ? std::get_if<AxisSplitPayload>(&split->payload) : nullptr;
        uint64_t query_width = static_cast<uint64_t>(attention_payload->query_heads) * attention_payload->head_dimension;
        if (query || split_count != 1 || gated_attention_count != 1 || !split || !gated_attention || !split_payload ||
            query_width > UINT32_MAX || split_payload->first_width != query_width ||
            split_payload->second_width != query_width || query_gate->outputs.size() != 1 ||
            role_tensor(model, *query_gate, TensorRole::QueryBias) != nullptr ||
            split->inputs != query_gate->outputs || split->outputs.size() != 2 || key->outputs.size() != 1 ||
            value->outputs.size() != 1 || rope->inputs.size() != 2 || rope->outputs.size() != 2 ||
            attention->inputs != std::vector<uint32_t>{rope->outputs[0], rope->outputs[1], value->outputs[0]} ||
            attention->outputs.size() != 1 || gated_attention->inputs != std::vector<uint32_t>{attention->outputs[0], split->outputs[1]} ||
            gated_attention->outputs.size() != 1 || output->inputs != gated_attention->outputs) return false;
        if (query_norm &&
            (!exact_dense_tensor(model, *query_norm, TensorRole::AttentionQueryNormWeight, ScalarType::F32) ||
             !exact_dense_tensor(model, *key_norm, TensorRole::AttentionKeyNormWeight, ScalarType::F32) ||
             query_norm->inputs != std::vector<uint32_t>{split->outputs[0]} ||
             key_norm->inputs != key->outputs || query_norm->outputs.size() != 1 || key_norm->outputs.size() != 1 ||
             rope->inputs != std::vector<uint32_t>{query_norm->outputs[0], key_norm->outputs[0]})) return false;
        if (!query_norm && rope->inputs != std::vector<uint32_t>{split->outputs[0], key->outputs[0]}) return false;
    } else if (split_count != 0 || gated_attention_count != 0) {
        return false;
    }
    uint32_t format = 0;
    const auto append_format = [&](const SemanticOperator* op, TensorRole role) {
        uint32_t candidate = 0;
        if (!op || !metal_dense_matrix_format(model, *op, role, candidate)) return false;
        format |= candidate;
        return true;
    };
    if (!append_format(query_projection, fused_query_gate ? TensorRole::AttentionQueryGateWeight : TensorRole::QueryWeight) ||
        !append_format(key, TensorRole::KeyWeight) ||
        !append_format(value, TensorRole::ValueWeight) ||
        !append_format(output, TensorRole::AttentionOutputWeight) || !append_format(gate, TensorRole::FfnGateWeight) ||
        !append_format(up, TensorRole::FfnUpWeight) || !append_format(down, TensorRole::FfnDownWeight)) return false;
    if (!allow_moe_tail) {
        DenseGraphWitness witness;
        if (!match_dense_graph_impl(model, layer, witness)) return false;
    }
    if (weight_format) *weight_format = format;
    return true;
}

bool exact_expert_format(const SemanticModel& model, const SemanticOperator& op, TensorRole role,
                         uint32_t experts, uint32_t input, uint32_t output, uint32_t format) {
    const MetalBlockedFormat* blocked = metal_blocked_format(format);
    const SemanticTensor* tensor = role_tensor(model, op, role, true);
    if (!blocked || op.tensors.size() != 1 || !tensor || tensor->logical_type != ScalarType::F32 || tensor->dimensions.size() != 3 ||
        tensor->dimensions[0] != Dimension{DimensionKind::Constant, experts} ||
        tensor->dimensions[1] != Dimension{DimensionKind::Constant, input} ||
        tensor->dimensions[2] != Dimension{DimensionKind::Constant, output} ||
        tensor->layout.kind != PhysicalLayoutKind::GgufBlocked || tensor->layout.packing != PackingKind::Gguf ||
        tensor->layout.rank != 3 || tensor->layout.axis_order[0] != 1 || tensor->layout.axis_order[1] != 2 ||
        tensor->layout.axis_order[2] != 0 || tensor->layout.block_rank != 1 ||
        tensor->layout.block_elements != blocked->elements || tensor->layout.block_bytes != blocked->bytes ||
        tensor->quantization.kind != blocked->quantization ||
        tensor->quantization.block_elements != blocked->elements || tensor->quantization.block_bytes != blocked->bytes ||
        tensor->quantization.group_size != blocked->elements || tensor->quantization.required_plane_mask != 1 ||
        tensor->planes.size() != 1 || tensor->planes[0].kind != PlaneKind::Values ||
        tensor->planes[0].storage_type != ScalarType::U8 || tensor->planes[0].alignment < 32 ||
        tensor->expert_axis.kind != ExpertAxisKind::ExpertBank || tensor->expert_axis.expert_axis != 0 ||
        tensor->expert_axis.member_axis != 0xff || tensor->expert_axis.input_axis != 1 ||
        tensor->expert_axis.output_axis != 2 || tensor->expert_axis.expert_count != experts ||
        input == 0 || output == 0 || input % blocked->elements != 0 ||
        static_cast<uint64_t>(input) > UINT64_MAX / output ||
        static_cast<uint64_t>(input) * output / blocked->elements > UINT64_MAX / blocked->bytes ||
        tensor->expert_axis.per_expert_byte_stride !=
            static_cast<uint64_t>(input) * output / blocked->elements * blocked->bytes ||
        tensor->planes[0].length != static_cast<uint64_t>(experts) * tensor->expert_axis.per_expert_byte_stride ||
        tensor->layout.strides[0] != 1 || tensor->layout.strides[1] != input ||
        tensor->layout.strides[2] != static_cast<uint64_t>(input) * output) return false;
    return true;
}

struct MoeDescriptorFacts {
    std::vector<uint32_t> covered_operator_ids;
    const SemanticOperator* dense_attn_norm = nullptr;
    const SemanticOperator* dense_query = nullptr;
    const SemanticOperator* dense_query_gate = nullptr;
    const SemanticOperator* dense_key = nullptr;
    const SemanticOperator* dense_value = nullptr;
    const SemanticOperator* dense_output = nullptr;
    const SemanticOperator* dense_query_norm = nullptr;
    const SemanticOperator* dense_key_norm = nullptr;
    const SemanticOperator* dense_ffn_norm = nullptr;
    const SemanticOperator* dense_gate = nullptr;
    const SemanticOperator* dense_up = nullptr;
    const SemanticOperator* dense_down = nullptr;
    const SemanticOperator* rope = nullptr;
    const SemanticOperator* swiglu = nullptr;
    const SemanticOperator* attention = nullptr;
    const SemanticOperator* router = nullptr;
    const SemanticOperator* router_linear = nullptr;
    const SemanticOperator* router_scale = nullptr;
    const SemanticOperator* router_normalization_scale = nullptr;
    const SemanticOperator* router_norm = nullptr;
    const SemanticOperator* expert_norm = nullptr;
    const SemanticOperator* expert_up = nullptr;
    const SemanticOperator* expert_split = nullptr;
    const SemanticOperator* expert_activation = nullptr;
    const SemanticOperator* expert_down = nullptr;
    const SemanticOperator* expert_reduce = nullptr;
    const SemanticOperator* attention_residual = nullptr;
    const SemanticOperator* residual = nullptr;
    const SemanticOperator* final_add = nullptr;
    const SemanticOperator* dense_post_norm = nullptr;
    const SemanticOperator* moe_post_norm = nullptr;
    const SemanticOperator* output_post_norm = nullptr;
    const SemanticOperator* output_scale = nullptr;
    const SemanticTensor* gate_up_tensor = nullptr;
    const SemanticTensor* down_tensor = nullptr;
    uint32_t hidden = 0;
    uint32_t intermediate = 0;
    uint32_t experts = 0;
    uint32_t selected = 0;
    uint32_t gate_up_format = 0;
    uint32_t down_format = 0;
    PhysicalLayoutKind gate_up_layout = static_cast<PhysicalLayoutKind>(0);
    PhysicalLayoutKind down_layout = static_cast<PhysicalLayoutKind>(0);
    QuantizationKind gate_up_quantization = static_cast<QuantizationKind>(0);
    QuantizationKind down_quantization = static_cast<QuantizationKind>(0);
    uint32_t gate_up_plane_mask = 0;
    uint32_t down_plane_mask = 0;
    uint32_t gate_up_block_elements = 0;
    uint32_t gate_up_block_bytes = 0;
    uint32_t down_block_elements = 0;
    uint32_t down_block_bytes = 0;
    uint32_t gate_up_input = 0;
    uint32_t gate_up_output = 0;
    uint32_t down_input = 0;
    uint32_t down_output = 0;
    uint64_t gate_up_expert_stride = 0;
    uint64_t down_expert_stride = 0;
    ActivationKind activation = ActivationKind::Silu;
    ExpertScaleSource scale_source = ExpertScaleSource::None;
    ValueSource value_source = ValueSource::SeparateProjection;
    uint32_t router_normalization_scale_bits = 0;
};

const SemanticValue* value_with_id(const SemanticModel& model, uint32_t id) {
    return id < model.values.size() && model.values[id].id == id ? &model.values[id] : nullptr;
}

bool value_last_width(const SemanticModel& model, uint32_t id, ScalarType type, uint32_t width) {
    const SemanticValue* value = value_with_id(model, id);
    return value && value->logical_type == type && !value->dimensions.empty() &&
           value->dimensions.back() == Dimension{DimensionKind::Constant, width};
}

bool value_flat_width(const SemanticModel& model, uint32_t id, ScalarType type, uint32_t width) {
    const SemanticValue* value = value_with_id(model, id);
    if (!value || value->logical_type != type || value->dimensions.size() < 2) return false;
    uint64_t product = 1;
    for (size_t index = 1; index < value->dimensions.size(); ++index) {
        const Dimension& dimension = value->dimensions[index];
        if (dimension.kind != DimensionKind::Constant || !dimension.constant_or_symbol ||
            product > UINT64_MAX / dimension.constant_or_symbol) return false;
        product *= dimension.constant_or_symbol;
    }
    return product == width;
}

bool same_value_prefix(const SemanticModel& model, uint32_t left_id, uint32_t right_id) {
    const SemanticValue* left = value_with_id(model, left_id);
    const SemanticValue* right = value_with_id(model, right_id);
    if (!left || !right || left->dimensions.size() != right->dimensions.size()) return false;
    if (left->dimensions.empty()) return true;
    for (size_t index = 0; index + 1 < left->dimensions.size(); ++index) {
        if (left->dimensions[index] != right->dimensions[index]) return false;
    }
    return true;
}

bool value_matches_routed(const SemanticModel& model, uint32_t id, uint32_t route_id,
                          ScalarType type, uint32_t width) {
    const SemanticValue* value = value_with_id(model, id);
    const SemanticValue* route = value_with_id(model, route_id);
    return value && route && value->logical_type == type &&
           value->dimensions.size() == route->dimensions.size() + 1 &&
           value_last_width(model, id, type, width) &&
           std::equal(route->dimensions.begin(), route->dimensions.end(), value->dimensions.begin());
}

bool exact_f32_vector(const SemanticModel& model, const SemanticOperator& op, uint32_t tensor_id, uint32_t width);

bool exact_norm_binding(const SemanticModel& model, const SemanticOperator& op, uint32_t width,
                        bool allow_no_scale) {
    const auto* payload = std::get_if<RmsNormPayload>(&op.payload);
    const float epsilon = payload ? std::bit_cast<float>(payload->epsilon_f32_bits) : 0.0f;
    if (!payload || !std::isfinite(epsilon) || epsilon < 0.0f || payload->axis != -1 ||
        op.inputs.size() != 1 || op.outputs.size() != 1 ||
        !value_last_width(model, op.inputs[0], ScalarType::F32, width) ||
        !value_last_width(model, op.outputs[0], ScalarType::F32, width)) return false;
    if (payload->weight_mode == 0) {
        return allow_no_scale && op.tensors.empty() &&
               payload->affine_geometry == RmsNormAffineGeometry::FullWidth &&
               payload->reduction_extent == 0;
    }
    if (payload->weight_mode != 1 || op.tensors.size() != 1) return false;
    uint32_t affine_width = 0;
    if (payload->affine_geometry == RmsNormAffineGeometry::FullWidth) {
        if (payload->reduction_extent != 0) return false;
        affine_width = width;
    } else if (payload->affine_geometry == RmsNormAffineGeometry::SharedAcrossGroups) {
        if (op.semantic_version < 8 || payload->reduction_extent == 0 ||
            width % payload->reduction_extent != 0) return false;
        affine_width = payload->reduction_extent;
    } else {
        return false;
    }
    return exact_f32_vector(model, op, op.tensors[0], affine_width);
}

bool exact_f32_vector(const SemanticModel& model, const SemanticOperator& op, uint32_t tensor_id, uint32_t width) {
    if (tensor_id >= model.tensors.size()) return false;
    const SemanticTensor& tensor = model.tensors[tensor_id];
    return tensor.logical_type == ScalarType::F32 && tensor.dimensions ==
               std::vector<Dimension>{{DimensionKind::Constant, width}} &&
           tensor.layout.kind == PhysicalLayoutKind::ContiguousRowMajor && tensor.layout.rank == 1 &&
           tensor.layout.strides[0] == 1 && tensor.quantization.kind == QuantizationKind::None &&
           tensor.planes.size() == 1 && tensor.planes[0].kind == PlaneKind::Values &&
           tensor.planes[0].storage_type == ScalarType::F32 && tensor.planes[0].length == uint64_t{width} * 4 &&
           tensor.planes[0].alignment != 0 && op.tensors.size() == 1 && op.tensors[0] == tensor_id;
}

bool exact_f32_matrix(const SemanticModel& model, const SemanticOperator& op, uint32_t tensor_id,
                      uint32_t rows, uint32_t columns) {
    if (tensor_id >= model.tensors.size()) return false;
    const SemanticTensor& tensor = model.tensors[tensor_id];
    return tensor.logical_type == ScalarType::F32 && tensor.dimensions ==
               std::vector<Dimension>{{DimensionKind::Constant, rows}, {DimensionKind::Constant, columns}} &&
           tensor.layout.kind == PhysicalLayoutKind::ContiguousRowMajor && tensor.layout.rank == 2 &&
           tensor.layout.axis_order[0] == 1 && tensor.layout.axis_order[1] == 0 &&
           tensor.layout.strides[0] == 1 && tensor.layout.strides[1] == columns &&
           tensor.quantization.kind == QuantizationKind::None && tensor.planes.size() == 1 &&
           tensor.planes[0].kind == PlaneKind::Values && tensor.planes[0].storage_type == ScalarType::F32 &&
           tensor.planes[0].length == uint64_t{rows} * columns * 4 && tensor.planes[0].alignment != 0 &&
           op.tensors.size() == 1 && op.tensors[0] == tensor_id;
}

bool semantic_flat_width(const SemanticModel& model, uint32_t id, ScalarType type,
                         uint32_t& width) {
    const SemanticValue* value = value_with_id(model, id);
    if (!value || value->logical_type != type || value->dimensions.size() < 2) return false;
    uint64_t product = 1;
    for (size_t index = 1; index < value->dimensions.size(); ++index) {
        const Dimension& dimension = value->dimensions[index];
        if (dimension.kind != DimensionKind::Constant || dimension.constant_or_symbol == 0 ||
            product > UINT32_MAX / dimension.constant_or_symbol) return false;
        product *= dimension.constant_or_symbol;
    }
    width = static_cast<uint32_t>(product);
    return true;
}

bool exact_linear_binding(const SemanticModel& model, const SemanticOperator& op,
                         uint32_t input_id, uint32_t output_id) {
    const auto* payload = std::get_if<LinearPayload>(&op.payload);
    if (op.kind != OperatorKind::Linear || !payload || payload->transpose_weight ||
        op.tensors.size() != 1 ||
        op.inputs.size() != 1 || op.outputs.size() != 1 || op.inputs[0] != input_id ||
        op.outputs[0] != output_id || op.tensors[0] >= model.tensors.size()) return false;
    uint32_t input_width = 0;
    uint32_t output_width = 0;
    if (!semantic_flat_width(model, input_id, ScalarType::F32, input_width) ||
        !semantic_flat_width(model, output_id, ScalarType::F32, output_width)) return false;
    const SemanticTensor& tensor = model.tensors[op.tensors[0]];
    return tensor.dimensions == std::vector<Dimension>{
               {DimensionKind::Constant, input_width},
               {DimensionKind::Constant, output_width}};
}

bool match_dense_graph_impl(const SemanticModel& model, const SemanticLayer& layer,
                            DenseGraphWitness& witness) {
    witness = {};
    if (layer.first_operator > model.operators.size() ||
        layer.operator_count > model.operators.size() - layer.first_operator ||
        layer.operator_count == 0) return false;

    const auto unique_kind = [&](OperatorKind kind) -> const SemanticOperator* {
        const SemanticOperator* found = nullptr;
        for (uint32_t index = 0; index != layer.operator_count; ++index) {
            const SemanticOperator& op = model.operators[layer.first_operator + index];
            if (op.kind != kind) continue;
            if (found) return nullptr;
            found = &op;
        }
        return found;
    };
    const SemanticOperator* attention_norm = layer_operator(model, layer, OperatorKind::RmsNorm,
                                                             TensorRole::AttentionNormWeight);
    const SemanticOperator* query = layer_operator(model, layer, OperatorKind::Linear,
                                                    TensorRole::QueryWeight);
    const SemanticOperator* query_gate = layer_operator(model, layer, OperatorKind::Linear,
                                                        TensorRole::AttentionQueryGateWeight);
    const SemanticOperator* key = layer_operator(model, layer, OperatorKind::Linear,
                                                 TensorRole::KeyWeight);
    const SemanticOperator* value = layer_operator(model, layer, OperatorKind::Linear,
                                                    TensorRole::ValueWeight);
    const SemanticOperator* query_norm = layer_operator(model, layer, OperatorKind::RmsNorm,
                                                        TensorRole::AttentionQueryNormWeight);
    const SemanticOperator* key_norm = layer_operator(model, layer, OperatorKind::RmsNorm,
                                                      TensorRole::AttentionKeyNormWeight);
    const SemanticOperator* attention_output = layer_operator(model, layer, OperatorKind::Linear,
                                                              TensorRole::AttentionOutputWeight);
    const SemanticOperator* ffn_norm = layer_operator(model, layer, OperatorKind::RmsNorm,
                                                      TensorRole::FfnNormWeight);
    const SemanticOperator* gate = layer_operator(model, layer, OperatorKind::Linear,
                                                  TensorRole::FfnGateWeight);
    const SemanticOperator* up = layer_operator(model, layer, OperatorKind::Linear,
                                                TensorRole::FfnUpWeight);
    const SemanticOperator* down = layer_operator(model, layer, OperatorKind::Linear,
                                                  TensorRole::FfnDownWeight);
    const SemanticOperator* rope = unique_kind(OperatorKind::Rope);
    const SemanticOperator* attention = unique_kind(OperatorKind::CausalAttention);
    const SemanticOperator* swiglu = unique_kind(OperatorKind::SwiGlu);
    const SemanticOperator* gated_attention = unique_kind(OperatorKind::GatedAttention);
    const SemanticOperator* split = unique_kind(OperatorKind::AxisSplit);
    if (!attention_norm || (!query && !query_gate) || (query && query_gate) || !key || !value ||
        !attention_output || !ffn_norm || !gate || !up || !down || !rope || !attention ||
        !swiglu || !std::get_if<CausalAttentionPayload>(&attention->payload) ||
        !std::get_if<SwiGluPayload>(&swiglu->payload) || (query_gate ? (!gated_attention || !split) :
                                                          (gated_attention || split)) ||
        (query_norm == nullptr) != (key_norm == nullptr)) return false;

    const auto one_input_one_output = [](const SemanticOperator* op) {
        return op && op->inputs.size() == 1 && op->outputs.size() == 1;
    };
    const auto two_input_one_output = [](const SemanticOperator* op) {
        return op && op->inputs.size() == 2 && op->outputs.size() == 1;
    };
    const auto two_input_two_output = [](const SemanticOperator* op) {
        return op && op->inputs.size() == 2 && op->outputs.size() == 2;
    };
    const auto one_input_two_output = [](const SemanticOperator* op) {
        return op && op->inputs.size() == 1 && op->outputs.size() == 2;
    };
    const auto three_input_one_output = [](const SemanticOperator* op) {
        return op && op->inputs.size() == 3 && op->outputs.size() == 1;
    };
    if (!one_input_one_output(attention_norm) || !one_input_one_output(key) ||
        !one_input_one_output(value) || !one_input_one_output(attention_output) ||
        !one_input_one_output(ffn_norm) || !one_input_one_output(gate) ||
        !one_input_one_output(up) || !one_input_one_output(down) ||
        !two_input_two_output(rope) || !three_input_one_output(attention) ||
        !two_input_one_output(swiglu)) return false;

    const auto* rope_payload = std::get_if<RopePayload>(&rope->payload);
    const auto* attention_payload = std::get_if<CausalAttentionPayload>(&attention->payload);
    const auto* swiglu_payload = std::get_if<SwiGluPayload>(&swiglu->payload);
    const float rope_base = rope_payload ? std::bit_cast<float>(rope_payload->base_f32_bits) : 0.0f;
    const float rope_scale = rope_payload ? std::bit_cast<float>(rope_payload->scale_f32_bits) : 0.0f;
    const float attention_scale = attention_payload
        ? std::bit_cast<float>(attention_payload->scale_f32_bits) : 0.0f;
    const auto* attention_norm_payload = std::get_if<RmsNormPayload>(&attention_norm->payload);
    const auto* ffn_norm_payload = std::get_if<RmsNormPayload>(&ffn_norm->payload);
    if (!rope_payload || !attention_payload || !swiglu_payload ||
        !std::isfinite(rope_base) || rope_base <= 0.0f ||
        !std::isfinite(rope_scale) || rope_scale != 1.0f ||
        !std::isfinite(attention_scale) || attention_scale <= 0.0f ||
        !attention_norm_payload || !ffn_norm_payload ||
        attention_norm_payload->epsilon_f32_bits != ffn_norm_payload->epsilon_f32_bits ||
        rope_payload->rotary_dimension == 0 ||
        rope_payload->rotary_dimension > attention_payload->head_dimension ||
        (rope_payload->rotary_dimension % 2) != 0 ||
        swiglu_payload->activation != ActivationKind::Silu ||
        attention->states.size() != 2 || attention_payload->value_source != ValueSource::SeparateProjection ||
        attention_payload->window != AttentionWindowKind::Global || attention_payload->window_tokens != 0) return false;
    const uint64_t query_width = static_cast<uint64_t>(attention_payload->query_heads) *
                                 attention_payload->head_dimension;
    const uint64_t kv_width = static_cast<uint64_t>(attention_payload->kv_heads) *
                              attention_payload->head_dimension;
    const uint64_t query_projection_width = query_gate ? query_width * 2 : query_width;
    uint32_t residual_width = 0;
    uint32_t normalized_width = 0;
    if (!query_width || query_width > UINT32_MAX || !kv_width || kv_width > UINT32_MAX ||
        query_projection_width > UINT32_MAX ||
        (query_norm &&
         (!exact_norm_binding(model, *query_norm, static_cast<uint32_t>(query_width), false) ||
          !exact_norm_binding(model, *key_norm, static_cast<uint32_t>(kv_width), false))) ||
        !semantic_flat_width(model, attention_norm->inputs[0], ScalarType::F32,
                             residual_width) ||
        !semantic_flat_width(model, attention_norm->outputs[0], ScalarType::F32,
                             normalized_width) ||
        residual_width != normalized_width ||
        !value_last_width(model, query ? query->outputs[0] : query_gate->outputs[0], ScalarType::F32,
                          static_cast<uint32_t>(query_projection_width)) ||
        !value_last_width(model, key->outputs[0], ScalarType::F32, static_cast<uint32_t>(kv_width)) ||
        !value_last_width(model, value->outputs[0], ScalarType::F32, static_cast<uint32_t>(kv_width)) ||
        !exact_norm_binding(model, *attention_norm, residual_width, false) ||
        !exact_norm_binding(model, *ffn_norm, residual_width, false)) return false;

    const uint32_t norm_output = attention_norm->outputs[0];
    if (query && (query->inputs != std::vector<uint32_t>{norm_output} ||
                  !value_last_width(model, query->outputs[0], ScalarType::F32,
                                    static_cast<uint32_t>(query_width)))) return false;
    if (key->inputs != std::vector<uint32_t>{norm_output} || value->inputs != std::vector<uint32_t>{norm_output}) return false;

    if (query_gate) {
        const auto* split_payload = std::get_if<AxisSplitPayload>(&split->payload);
        if (!one_input_one_output(query_gate) || query_gate->inputs != std::vector<uint32_t>{norm_output} ||
            query_gate->outputs.size() != 1 || !one_input_two_output(split) || !split_payload ||
            split->inputs != query_gate->outputs || split_payload->first_width != query_width ||
            split_payload->second_width != query_width || role_tensor(model, *query_gate, TensorRole::QueryBias) != nullptr ||
            gated_attention->inputs.size() != 2 || gated_attention->outputs.size() != 1) return false;
        if (query_norm) {
            if (!one_input_one_output(query_norm) || !one_input_one_output(key_norm) ||
                query_norm->inputs != std::vector<uint32_t>{split->outputs[0]} || query_norm->outputs.size() != 1 ||
                key_norm->inputs != key->outputs || key_norm->outputs.size() != 1 ||
                rope->inputs != std::vector<uint32_t>{query_norm->outputs[0], key_norm->outputs[0]}) return false;
        } else if (rope->inputs != std::vector<uint32_t>{split->outputs[0], key->outputs[0]}) return false;
        if (gated_attention->inputs != std::vector<uint32_t>{attention->outputs[0], split->outputs[1]}) return false;
    } else {
        if (query_norm) {
            if (!one_input_one_output(query_norm) || !one_input_one_output(key_norm) ||
                query_norm->inputs != query->outputs || key_norm->inputs != key->outputs ||
                rope->inputs != std::vector<uint32_t>{query_norm->outputs[0], key_norm->outputs[0]}) return false;
        } else if (rope->inputs != std::vector<uint32_t>{query->outputs[0], key->outputs[0]}) return false;
    }
    if (attention->inputs != std::vector<uint32_t>{rope->outputs[0], rope->outputs[1], value->outputs[0]} ||
        (query_gate ? attention_output->inputs != gated_attention->outputs
                    : attention_output->inputs != attention->outputs) ||
        !(value_last_width(model, attention->outputs[0], ScalarType::F32, static_cast<uint32_t>(query_width)) ||
          value_flat_width(model, attention->outputs[0], ScalarType::F32, static_cast<uint32_t>(query_width)))) return false;

    const uint32_t residual_input = attention_norm->inputs[0];
    const uint32_t attention_output_value = attention_output->outputs[0];
    const SemanticOperator* attention_residual = nullptr;
    const SemanticOperator* final_residual = nullptr;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind != OperatorKind::Add || op.inputs.size() != 2 || op.outputs.size() != 1) continue;
        if (op.inputs == std::vector<uint32_t>{residual_input, attention_output_value}) {
            if (attention_residual) return false;
            attention_residual = &op;
        }
    }
    if (!attention_residual || !value_last_width(model, attention_residual->outputs[0], ScalarType::F32,
                                                  residual_width) ||
        ffn_norm->inputs != attention_residual->outputs || gate->inputs != ffn_norm->outputs ||
        up->inputs != ffn_norm->outputs || !two_input_one_output(swiglu) ||
        swiglu->inputs != std::vector<uint32_t>{gate->outputs[0], up->outputs[0]} ||
        down->inputs != swiglu->outputs ||
        !value_last_width(model, down->outputs[0], ScalarType::F32, residual_width)) return false;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind != OperatorKind::Add || op.inputs.size() != 2 || op.outputs.size() != 1) continue;
        if (op.inputs == std::vector<uint32_t>{attention_residual->outputs[0], down->outputs[0]}) {
            if (final_residual) return false;
            final_residual = &op;
        }
    }
    if (!final_residual || !value_last_width(model, final_residual->outputs[0], ScalarType::F32,
                                              residual_width)) return false;

    // The Metal binder consumes a matrix as [input_width, output_width].
    // Physical format validation alone is insufficient: a valid quantized
    // plane with the wrong semantic shape must not be bound to a different
    // graph edge.
    if (!exact_linear_binding(model, query ? *query : *query_gate, norm_output,
                              query ? query->outputs[0] : query_gate->outputs[0]) ||
        !exact_linear_binding(model, *key, norm_output, key->outputs[0]) ||
        !exact_linear_binding(model, *value, norm_output, value->outputs[0]) ||
        !exact_linear_binding(model, *attention_output, attention_output->inputs[0],
                              attention_output->outputs[0]) ||
        !exact_linear_binding(model, *gate, ffn_norm->outputs[0], gate->outputs[0]) ||
        !exact_linear_binding(model, *up, ffn_norm->outputs[0], up->outputs[0]) ||
        !exact_linear_binding(model, *down, swiglu->outputs[0], down->outputs[0])) return false;

    std::vector<const SemanticOperator*> covered = {
        attention_norm, query ? query : query_gate, key, value, rope, attention,
        attention_output, attention_residual, ffn_norm, gate, up, swiglu, down, final_residual,
    };
    if (query_gate) {
        covered.push_back(split);
        covered.push_back(gated_attention);
    }
    if (query_norm) {
        covered.push_back(query_norm);
        covered.push_back(key_norm);
    }
    std::sort(covered.begin(), covered.end());
    if (std::adjacent_find(covered.begin(), covered.end()) != covered.end() ||
        covered.size() != layer.operator_count) return false;
    for (const SemanticOperator* op : covered) {
        if (!op || op->outputs.empty()) return false;
        if (std::find_if(model.operators.begin(), model.operators.end(), [&](const SemanticOperator& candidate) {
                return &candidate != op && candidate.id == op->id;
            }) != model.operators.end()) return false;
    }
    for (size_t i = 0; i != covered.size(); ++i) {
        for (size_t j = i + 1; j != covered.size(); ++j) {
            for (uint32_t output : covered[i]->outputs) {
                if (std::find(covered[j]->outputs.begin(), covered[j]->outputs.end(), output) != covered[j]->outputs.end()) return false;
            }
        }
    }

    std::vector<uint32_t> indegree(covered.size(), 0);
    std::vector<std::vector<size_t>> successors(covered.size());
    for (size_t consumer = 0; consumer != covered.size(); ++consumer) {
        for (uint32_t input : covered[consumer]->inputs) {
            if (!value_with_id(model, input)) return false;
            size_t producer = covered.size();
            for (size_t candidate = 0; candidate != covered.size(); ++candidate) {
                if (std::find(covered[candidate]->outputs.begin(), covered[candidate]->outputs.end(), input) ==
                    covered[candidate]->outputs.end()) continue;
                if (producer != covered.size()) return false;
                producer = candidate;
            }
            if (producer == covered.size()) continue;
            if (producer == consumer) return false;
            successors[producer].push_back(consumer);
            ++indegree[consumer];
        }
    }
    std::vector<size_t> ready;
    for (size_t index = 0; index != covered.size(); ++index) if (!indegree[index]) ready.push_back(index);
    size_t processed = 0;
    for (size_t cursor = 0; cursor != ready.size(); ++cursor) {
        ++processed;
        for (size_t successor : successors[ready[cursor]]) if (--indegree[successor] == 0) ready.push_back(successor);
    }
    if (processed != covered.size()) return false;

    witness.covered_operator_ids.reserve(covered.size());
    for (const SemanticOperator* op : covered) witness.covered_operator_ids.push_back(op->id);
    witness.attention_norm = attention_norm;
    witness.query = query;
    witness.query_gate = query_gate;
    witness.key = key;
    witness.value = value;
    witness.query_norm = query_norm;
    witness.key_norm = key_norm;
    witness.rope = rope;
    witness.attention = attention;
    witness.attention_output = attention_output;
    witness.attention_residual = attention_residual;
    witness.ffn_norm = ffn_norm;
    witness.gate = gate;
    witness.up = up;
    witness.swiglu = swiglu;
    witness.down = down;
    witness.final_residual = final_residual;
    witness.query_split = split;
    witness.gated_attention = gated_attention;
    return true;
}

bool dense_attention_geometry_is_valid(const SemanticModel& model,
                                       std::string& detail) {
    for (const SemanticLayer& layer : model.layers) {
        if ((layer.flags & kSemanticLayerFlagSpeculative) != 0) continue;
        if (layer.first_operator > model.operators.size() ||
            layer.operator_count > model.operators.size() - layer.first_operator) {
            detail = "canonical Metal attention layer range is invalid";
            return false;
        }
        bool has_router = false;
        for (uint32_t index = 0; index != layer.operator_count; ++index) {
            if (model.operators[layer.first_operator + index].kind == OperatorKind::RouterTopK) {
                has_router = true;
                break;
            }
        }
        if (has_router) continue;
        const bool has_dense_attention =
            layer_operator(model, layer, OperatorKind::RmsNorm, TensorRole::AttentionNormWeight) &&
            (layer_operator(model, layer, OperatorKind::Linear, TensorRole::QueryWeight) ||
             layer_operator(model, layer, OperatorKind::Linear, TensorRole::AttentionQueryGateWeight));
        if (!has_dense_attention) continue;
        DenseGraphWitness witness;
        if (!match_dense_graph_impl(model, layer, witness) || !witness.attention) {
            detail = "canonical Metal dense attention graph mismatch at layer " +
                     std::to_string(layer.layer_index);
            return false;
        }
        const auto* payload = std::get_if<CausalAttentionPayload>(&witness.attention->payload);
        if (!payload || payload->query_heads == 0 || payload->kv_heads == 0 || payload->head_dimension == 0) {
            detail = "canonical Metal dense attention geometry is invalid at layer " +
                     std::to_string(layer.layer_index);
            return false;
        }
        const uint64_t query_width = static_cast<uint64_t>(payload->query_heads) *
                                     payload->head_dimension;
        const uint64_t key_value_width = static_cast<uint64_t>(payload->kv_heads) *
                                         payload->head_dimension;
        if (query_width > INT_MAX || key_value_width > INT_MAX) {
            detail = "canonical Metal dense attention geometry exceeds token capacity at layer " +
                     std::to_string(layer.layer_index);
            return false;
        }
    }
    return true;
}

bool canonical_moe_capabilities(const RuntimeCapabilities& capabilities) {
    return capabilities.metal_device && capabilities.metal_library && capabilities.metal_pipeline &&
           capabilities.global_fp32_kv && capabilities.transactional_state &&
           capabilities.metal_moe_router_topk && capabilities.metal_moe_gate_up && capabilities.metal_moe_reduce;
}

bool canonical_moe_q5_capabilities(const RuntimeCapabilities& capabilities) {
    return canonical_moe_capabilities(capabilities) && capabilities.metal_moe_down_q5_0;
}

bool canonical_moe_q8_capabilities(const RuntimeCapabilities& capabilities) {
    return canonical_moe_capabilities(capabilities) && capabilities.metal_moe_down_q8_0;
}

bool moe_metal_layer(const SemanticModel& model, const SemanticLayer& layer,
                     const SemanticOperator*& attention, uint32_t* weight_format = nullptr,
                     MoeDescriptorFacts* facts_out = nullptr) {
    uint32_t dense_format = 0;
    if (layer.first_operator > model.operators.size() ||
        layer.operator_count > model.operators.size() - layer.first_operator ||
        !dense_metal_layer(model, layer, attention, &dense_format, true) || !attention) return false;

    auto unique_kind = [&](OperatorKind kind) -> const SemanticOperator* {
        const SemanticOperator* result = nullptr;
        for (uint32_t index = 0; index != layer.operator_count; ++index) {
            const SemanticOperator& op = model.operators[layer.first_operator + index];
            if (op.kind != kind) continue;
            if (result) return nullptr;
            result = &op;
        }
        return result;
    };

    // The MoE leaf shares the same dense attention and dense-FFN prefix as
    // the dense leaf.  Physical-format admission is not enough: freeze the
    // exact value edges that the Metal implementation executes before
    // matching the routed branch.
    const SemanticOperator* dense_attention_norm = layer_operator(
        model, layer, OperatorKind::RmsNorm, TensorRole::AttentionNormWeight);
    const SemanticOperator* dense_query = layer_operator(
        model, layer, OperatorKind::Linear, TensorRole::QueryWeight);
    const SemanticOperator* dense_query_gate = layer_operator(
        model, layer, OperatorKind::Linear, TensorRole::AttentionQueryGateWeight);
    const SemanticOperator* dense_key = layer_operator(
        model, layer, OperatorKind::Linear, TensorRole::KeyWeight);
    const SemanticOperator* dense_value = layer_operator(
        model, layer, OperatorKind::Linear, TensorRole::ValueWeight);
    const SemanticOperator* dense_attention_output = layer_operator(
        model, layer, OperatorKind::Linear, TensorRole::AttentionOutputWeight);
    const SemanticOperator* dense_ffn_norm = layer_operator(
        model, layer, OperatorKind::RmsNorm, TensorRole::FfnNormWeight);
    const SemanticOperator* dense_gate_for_edges = layer_operator(
        model, layer, OperatorKind::Linear, TensorRole::FfnGateWeight);
    const SemanticOperator* dense_up_for_edges = layer_operator(
        model, layer, OperatorKind::Linear, TensorRole::FfnUpWeight);
    const SemanticOperator* dense_down_for_edges = layer_operator(
        model, layer, OperatorKind::Linear, TensorRole::FfnDownWeight);
    const SemanticOperator* dense_swiglu = unique_kind(OperatorKind::SwiGlu);
    const SemanticOperator* dense_query_projection = dense_query_gate ? dense_query_gate : dense_query;
    if (!dense_attention_norm || !dense_query_projection || !dense_key || !dense_value ||
        !dense_attention_output || !dense_ffn_norm || !dense_gate_for_edges || !dense_up_for_edges ||
        !dense_down_for_edges || !dense_swiglu || dense_attention_norm->inputs.size() != 1 ||
        dense_attention_norm->outputs.size() != 1 || dense_query_projection->outputs.size() != 1 ||
        dense_key->outputs.size() != 1 || dense_value->outputs.size() != 1 ||
        dense_attention_output->outputs.size() != 1 || dense_ffn_norm->outputs.size() != 1 ||
        dense_gate_for_edges->outputs.size() != 1 || dense_up_for_edges->outputs.size() != 1 ||
        dense_swiglu->outputs.size() != 1 || dense_down_for_edges->outputs.size() != 1) return false;
    const uint32_t dense_residual_input = dense_attention_norm->inputs[0];
    const uint32_t dense_normalized = dense_attention_norm->outputs[0];
    if (dense_query_projection->inputs != std::vector<uint32_t>{dense_normalized} ||
        dense_key->inputs != std::vector<uint32_t>{dense_normalized} ||
        dense_value->inputs != std::vector<uint32_t>{dense_normalized}) return false;
    const SemanticOperator* exact_attention_residual = nullptr;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind != OperatorKind::Add ||
            op.inputs != std::vector<uint32_t>{dense_residual_input,
                                                dense_attention_output->outputs[0]} ||
            op.outputs.size() != 1) continue;
        if (exact_attention_residual) return false;
        exact_attention_residual = &op;
    }
    if (!exact_attention_residual || dense_ffn_norm->inputs != exact_attention_residual->outputs ||
        dense_gate_for_edges->inputs != dense_ffn_norm->outputs ||
        dense_up_for_edges->inputs != dense_ffn_norm->outputs ||
        dense_swiglu->inputs != std::vector<uint32_t>{dense_gate_for_edges->outputs[0],
                                                       dense_up_for_edges->outputs[0]} ||
        dense_down_for_edges->inputs != dense_swiglu->outputs ||
        !exact_linear_binding(model, *dense_query_projection, dense_normalized,
                              dense_query_projection->outputs[0]) ||
        !exact_linear_binding(model, *dense_key, dense_normalized, dense_key->outputs[0]) ||
        !exact_linear_binding(model, *dense_value, dense_normalized, dense_value->outputs[0]) ||
        !exact_linear_binding(model, *dense_attention_output,
                              dense_attention_output->inputs[0],
                              dense_attention_output->outputs[0]) ||
        !exact_linear_binding(model, *dense_gate_for_edges, dense_ffn_norm->outputs[0],
                              dense_gate_for_edges->outputs[0]) ||
        !exact_linear_binding(model, *dense_up_for_edges, dense_ffn_norm->outputs[0],
                              dense_up_for_edges->outputs[0]) ||
        !exact_linear_binding(model, *dense_down_for_edges, dense_swiglu->outputs[0],
                              dense_down_for_edges->outputs[0])) return false;

    const SemanticOperator* router = unique_kind(OperatorKind::RouterTopK);
    if (!router || router->inputs.size() != 1 || router->outputs.size() != 2) return false;
    const auto* router_payload = std::get_if<RouterTopKPayload>(&router->payload);
    if (!router_payload || router_payload->expert_count == 0 || router_payload->expert_count > 512 ||
        router_payload->selected_count == 0 || router_payload->selected_count > 16 ||
        router_payload->selected_count > router_payload->expert_count ||
        router_payload->score_domain != RouterScoreDomain::Logits ||
        router_payload->normalization_order != RouterNormalizationOrder::NormalizeThenSelect ||
        router_payload->selected_weight_normalization !=
            SelectedWeightNormalization::RenormalizeSelectedProbabilities ||
        router_payload->tie_policy != RouterTiePolicy::LowestExpertId ||
        router_payload->weight_source != RouterWeightSource::SelectedNormalizedScore ||
        !value_last_width(model, router->inputs[0], ScalarType::F32, router_payload->expert_count) ||
        !value_last_width(model, router->outputs[0], ScalarType::U32, router_payload->selected_count) ||
        !value_last_width(model, router->outputs[1], ScalarType::F32, router_payload->selected_count) ||
        !same_value_prefix(model, router->inputs[0], router->outputs[0]) ||
        !same_value_prefix(model, router->outputs[0], router->outputs[1])) return false;

    const SemanticOperator* router_linear = nullptr;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind != OperatorKind::Linear || op.outputs != router->inputs) continue;
        if (router_linear) return false;
        router_linear = &op;
    }
    const auto* router_linear_payload = router_linear
        ? std::get_if<LinearPayload>(&router_linear->payload) : nullptr;
    if (!router_linear || !router_linear_payload || router_linear_payload->transpose_weight ||
        router_linear_payload->has_bias || router_linear_payload->accumulation_type != ScalarType::F32 ||
        router_linear->inputs.size() != 1 || router_linear->outputs.size() != 1 ||
        !value_with_id(model, router_linear->inputs[0]) ||
        value_with_id(model, router_linear->inputs[0])->logical_type != ScalarType::F32 ||
        router_linear->outputs[0] != router->inputs[0]) return false;
    const SemanticValue* router_input = value_with_id(model, router_linear->inputs[0]);
    if (!router_input || router_input->dimensions.empty()) return false;
    const Dimension& hidden_dimension = router_input->dimensions.back();
    if (hidden_dimension.kind != DimensionKind::Constant || hidden_dimension.constant_or_symbol == 0 ||
        hidden_dimension.constant_or_symbol > UINT32_MAX) return false;
    const uint32_t hidden = static_cast<uint32_t>(hidden_dimension.constant_or_symbol);
    if (router_linear->tensors.size() != 1 ||
        !exact_f32_matrix(model, *router_linear, router_linear->tensors[0], router_payload->expert_count, hidden)) return false;

    // The router has two semantically distinct scales: the learned tensor
    // scale and the explicit H^-1/2 literal immediately before projection.
    // The native leaf must consume the latter's exact bits; it may not infer
    // or default the factor from the hidden width.
    const SemanticOperator* normalization_scale = nullptr;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind != OperatorKind::Scale || op.outputs != router_linear->inputs) continue;
        if (normalization_scale) return false;
        normalization_scale = &op;
    }
    const SemanticOperator* route_scale = nullptr;
    if (!normalization_scale || normalization_scale->inputs.size() != 1 ||
        normalization_scale->outputs.size() != 1) return false;
    const auto* normalization_payload = std::get_if<ScalePayload>(&normalization_scale->payload);
    if (!normalization_payload || normalization_payload->source != ScaleSource::LiteralF32 ||
        normalization_payload->literal_f32_bits == 0 || !normalization_scale->tensors.empty() ||
        !std::isfinite(std::bit_cast<float>(normalization_payload->literal_f32_bits)) ||
        std::bit_cast<float>(normalization_payload->literal_f32_bits) <= 0.0f ||
        std::fabs(std::bit_cast<float>(normalization_payload->literal_f32_bits) -
                  1.0f / std::sqrt(static_cast<float>(hidden))) > 1.0e-6f) return false;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind != OperatorKind::Scale || op.outputs != normalization_scale->inputs) continue;
        if (route_scale) return false;
        route_scale = &op;
    }
    if (!route_scale || route_scale->inputs.size() != 1 || route_scale->outputs.size() != 1 ||
        !role_tensor(model, *route_scale, TensorRole::RouterScaleWeight)) return false;
    const auto* route_payload = std::get_if<ScalePayload>(&route_scale->payload);
    if (!route_payload || route_payload->source != ScaleSource::Tensor ||
        route_payload->literal_f32_bits != 0 ||
        route_scale->tensors.size() != 1 ||
        !exact_f32_vector(model, *route_scale, route_scale->tensors[0], hidden)) return false;
    const SemanticOperator* router_norm = nullptr;
    const uint32_t norm_output = route_scale->inputs[0];
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind != OperatorKind::RmsNorm || op.outputs != std::vector<uint32_t>{norm_output}) continue;
        if (router_norm) return false;
        router_norm = &op;
    }
    if (!router_norm || !exact_norm_binding(model, *router_norm, hidden, true) ||
        std::get_if<RmsNormPayload>(&router_norm->payload)->weight_mode != 0 ||
        route_scale->inputs[0] != router_norm->outputs[0] ||
        normalization_scale->inputs[0] != route_scale->outputs[0]) return false;

    const SemanticOperator* expert_up = nullptr;
    const SemanticOperator* split = nullptr;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& candidate_split = model.operators[layer.first_operator + index];
        if (candidate_split.kind != OperatorKind::AxisSplit || candidate_split.inputs.size() != 1 ||
            candidate_split.outputs.size() != 2) continue;
        const auto* split_payload = std::get_if<AxisSplitPayload>(&candidate_split.payload);
        if (!split_payload || split_payload->first_width == 0 ||
            split_payload->first_width != split_payload->second_width) return false;
        const SemanticOperator* candidate_up = nullptr;
        for (uint32_t routed_index = 0; routed_index != layer.operator_count; ++routed_index) {
            const SemanticOperator& candidate = model.operators[layer.first_operator + routed_index];
            if (candidate.kind != OperatorKind::RoutedLinear || candidate.outputs != candidate_split.inputs ||
                candidate.inputs.size() != 3 || candidate.inputs[1] != router->outputs[0] ||
                candidate.inputs[2] != router->outputs[1]) continue;
            if (candidate_up) return false;
            candidate_up = &candidate;
        }
        if (!candidate_up) continue;
        if (split || expert_up) return false;
        split = &candidate_split;
        expert_up = candidate_up;
    }
    if (!expert_up || !split) return false;
    const auto* split_payload = std::get_if<AxisSplitPayload>(&split->payload);
    const auto* up_payload = std::get_if<RoutedLinearPayload>(&expert_up->payload);
    if (!split_payload || !up_payload || up_payload->accumulation_type != ScalarType::F32 ||
        expert_up->outputs.size() != 1 || !value_last_width(model, expert_up->inputs[0], ScalarType::F32, hidden) ||
        !value_matches_routed(model, expert_up->outputs[0], router->outputs[0], ScalarType::F32,
                              2 * split_payload->first_width) ||
        !value_matches_routed(model, split->outputs[0], router->outputs[0], ScalarType::F32,
                              split_payload->first_width) ||
        !value_matches_routed(model, split->outputs[1], router->outputs[0], ScalarType::F32,
                              split_payload->second_width)) return false;

    const SemanticOperator* expert_norm = nullptr;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind != OperatorKind::RmsNorm || expert_up->inputs.empty() ||
            op.outputs != std::vector<uint32_t>{expert_up->inputs[0]}) continue;
        if (expert_norm) return false;
        expert_norm = &op;
    }
    if (!expert_norm || !role_tensor(model, *expert_norm, TensorRole::ExpertNormWeight) ||
        !exact_norm_binding(model, *expert_norm, hidden, false) ||
        expert_norm->inputs.size() != 1 || router_norm->inputs != expert_norm->inputs) return false;

    const SemanticOperator* activation = nullptr;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind != OperatorKind::GatedActivation || op.inputs != split->outputs) continue;
        if (activation) return false;
        activation = &op;
    }
    const auto* activation_payload = activation ? std::get_if<GatedActivationPayload>(&activation->payload) : nullptr;
    if (!activation || !activation_payload || activation_payload->activation != ActivationKind::GeluTanh ||
        activation->outputs.size() != 1 ||
        !value_matches_routed(model, activation->outputs[0], router->outputs[0], ScalarType::F32,
                              split_payload->first_width)) return false;

    const SemanticOperator* expert_down = nullptr;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind != OperatorKind::RoutedLinear ||
            op.inputs != std::vector<uint32_t>{activation->outputs[0], router->outputs[0], router->outputs[1]}) continue;
        if (expert_down) return false;
        expert_down = &op;
    }
    const auto* down_payload = expert_down ? std::get_if<RoutedLinearPayload>(&expert_down->payload) : nullptr;
    if (!expert_down || !down_payload || down_payload->accumulation_type != ScalarType::F32 ||
        !value_matches_routed(model, expert_down->outputs.size() == 1 ? expert_down->outputs[0] : UINT32_MAX,
                              router->outputs[0], ScalarType::F32, hidden)) return false;

    const SemanticOperator* reduce = nullptr;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind != OperatorKind::WeightedExpertReduce ||
            op.inputs != std::vector<uint32_t>{expert_down->outputs[0], router->outputs[0], router->outputs[1]}) continue;
        if (reduce) return false;
        reduce = &op;
    }
    const auto* reduce_payload = reduce ? std::get_if<WeightedExpertReducePayload>(&reduce->payload) : nullptr;
    if (!reduce || !reduce_payload || reduce_payload->association != ExpertReduceAssociation::SelectedOrderLeftToRight ||
        reduce_payload->accumulation_type != ScalarType::F32 || reduce->outputs.size() != 1 ||
        !value_last_width(model, reduce->outputs[0], ScalarType::F32, hidden)) return false;
    if (reduce_payload->scale_source != ExpertScaleSource::PerExpertTensor ||
        reduce->tensors.size() != 1 ||
        !role_tensor(model, *reduce, TensorRole::ReduceScaleWeight) ||
        !exact_f32_vector(model, *reduce, reduce->tensors[0], router_payload->expert_count)) {
        return false;
    }

    const SemanticOperator* attention_residual = nullptr;
    const SemanticOperator* ffn_norm_for_edges = layer_operator(model, layer, OperatorKind::RmsNorm,
                                                                 TensorRole::FfnNormWeight);
    const SemanticOperator* attention_output_for_edges = layer_operator(
        model, layer, OperatorKind::Linear, TensorRole::AttentionOutputWeight);
    const uint32_t attention_value =
        ffn_norm_for_edges && ffn_norm_for_edges->inputs.size() == 1
            ? ffn_norm_for_edges->inputs[0] : UINT32_MAX;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind != OperatorKind::Add || op.outputs != std::vector<uint32_t>{attention_value}) continue;
        if (attention_residual) return false;
        attention_residual = &op;
    }
    if (!attention_residual || attention_residual->inputs.size() != 2 ||
        attention_residual->outputs.size() != 1 || router_norm->inputs != std::vector<uint32_t>{attention_value} ||
        expert_norm->inputs != std::vector<uint32_t>{attention_value} || !attention_output_for_edges ||
        attention_output_for_edges->outputs.size() != 1 ||
        std::find(attention_residual->inputs.begin(), attention_residual->inputs.end(),
                  attention_output_for_edges->outputs[0]) ==
            attention_residual->inputs.end()) return false;

    const auto unique_role_norm = [&](uint32_t input, TensorRole role,
                                      bool& ambiguous) -> const SemanticOperator* {
        const SemanticOperator* found = nullptr;
        ambiguous = false;
        for (uint32_t index = 0; index != layer.operator_count; ++index) {
            const SemanticOperator& op = model.operators[layer.first_operator + index];
            if (op.kind != OperatorKind::RmsNorm || op.inputs != std::vector<uint32_t>{input} ||
                !role_tensor(model, op, role, false)) continue;
            if (found) {
                ambiguous = true;
                return nullptr;
            }
            found = &op;
        }
        return found;
    };
    bool ambiguous_dense_post = false;
    bool ambiguous_moe_post = false;
    bool ambiguous_output_post = false;
    // The dense branch terminates at the role-tagged FfnDownWeight operator
    // found by the dense matcher, while the routed branch terminates at the
    // routed down operator above.
    const SemanticOperator* dense_down = layer_operator(model, layer, OperatorKind::Linear,
                                                        TensorRole::FfnDownWeight);
    if (!dense_down || dense_down->outputs.size() != 1) return false;
    const SemanticOperator* dense_post_norm =
        unique_role_norm(dense_down->outputs[0], TensorRole::PostNormWeight,
                         ambiguous_dense_post);
    const SemanticOperator* moe_post_norm =
        unique_role_norm(reduce->outputs[0], TensorRole::PostNormWeight,
                         ambiguous_moe_post);
    const uint32_t dense_terminal = dense_post_norm && dense_post_norm->outputs.size() == 1
        ? dense_post_norm->outputs[0] : dense_down->outputs[0];
    const uint32_t moe_terminal = moe_post_norm && moe_post_norm->outputs.size() == 1
        ? moe_post_norm->outputs[0] : reduce->outputs[0];
    const SemanticOperator* residual = nullptr;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind != OperatorKind::Add || op.inputs != std::vector<uint32_t>{dense_terminal, moe_terminal}) continue;
        if (residual) return false;
        residual = &op;
    }
    if (!residual || residual->outputs.size() != 1) return false;
    const SemanticOperator* output_post_norm =
        unique_role_norm(residual->outputs[0], TensorRole::PostNormWeight,
                         ambiguous_output_post);
    const uint32_t final_input = output_post_norm && output_post_norm->outputs.size() == 1
        ? output_post_norm->outputs[0] : residual->outputs[0];
    const SemanticOperator* final_add = nullptr;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind != OperatorKind::Add || op.inputs != std::vector<uint32_t>{attention_value, final_input}) continue;
        if (final_add) return false;
        final_add = &op;
    }
    const SemanticValue* residual_output = value_with_id(model, residual->outputs[0]);
    const SemanticValue* final_output = final_add ? value_with_id(model, final_add->outputs[0]) : nullptr;
    if (!final_add || residual->inputs.size() != 2 || final_add->inputs.size() != 2 ||
        !residual_output || !final_output || residual_output->logical_type != ScalarType::F32 ||
        final_output->logical_type != ScalarType::F32 ||
        !value_last_width(model, residual->outputs[0], ScalarType::F32, hidden) ||
        !value_last_width(model, final_add->outputs[0], ScalarType::F32, hidden) ||
        residual_output->dimensions != final_output->dimensions) return false;
    if (ambiguous_dense_post || ambiguous_moe_post || ambiguous_output_post ||
        (dense_post_norm && !exact_norm_binding(model, *dense_post_norm, hidden, false)) ||
        (moe_post_norm && !exact_norm_binding(model, *moe_post_norm, hidden, false)) ||
        (output_post_norm && !exact_norm_binding(model, *output_post_norm, hidden, false))) return false;
    const SemanticOperator* output_scale = nullptr;
    const uint32_t output_scale_input = final_add->outputs[0];
    bool ambiguous_output_scale = false;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind != OperatorKind::Scale || op.inputs != std::vector<uint32_t>{output_scale_input} ||
            !role_tensor(model, op, TensorRole::OutputScaleWeight, false)) continue;
        if (output_scale) {
            ambiguous_output_scale = true;
            break;
        }
        output_scale = &op;
    }
    if (ambiguous_output_scale) return false;
    if (output_scale) {
        const auto* scale_payload = std::get_if<ScalePayload>(&output_scale->payload);
        if (!scale_payload || scale_payload->source != ScaleSource::Tensor ||
            scale_payload->literal_f32_bits != 0 || output_scale->inputs.size() != 1 ||
            output_scale->outputs.size() != 1 || output_scale->tensors.size() != 1 ||
            !exact_f32_vector(model, *output_scale,
                              output_scale->tensors[0], 1)) return false;
    }

    // All norms fused by the native leaf share one epsilon payload.  Keeping
    // this as an exact-bit gate prevents the Metal binding from silently
    // applying one norm's value to another semantic edge.
    const SemanticOperator* attn_norm_witness = layer_operator(model, layer, OperatorKind::RmsNorm,
                                                               TensorRole::AttentionNormWeight);
    const SemanticOperator* ffn_norm_witness = layer_operator(model, layer, OperatorKind::RmsNorm,
                                                              TensorRole::FfnNormWeight);
    const auto* attn_norm_payload = attn_norm_witness
        ? std::get_if<RmsNormPayload>(&attn_norm_witness->payload) : nullptr;
    const auto* ffn_norm_payload = ffn_norm_witness
        ? std::get_if<RmsNormPayload>(&ffn_norm_witness->payload) : nullptr;
    if (!attn_norm_payload || !ffn_norm_payload ||
        attn_norm_payload->epsilon_f32_bits != ffn_norm_payload->epsilon_f32_bits) return false;
    const uint32_t fused_epsilon_bits = attn_norm_payload->epsilon_f32_bits;
    const auto same_fused_epsilon = [&](const SemanticOperator* op) {
        if (!op) return true;
        const auto* payload = std::get_if<RmsNormPayload>(&op->payload);
        return payload && payload->epsilon_f32_bits == fused_epsilon_bits;
    };
    if (!same_fused_epsilon(router_norm) || !same_fused_epsilon(expert_norm) ||
        !same_fused_epsilon(dense_post_norm) || !same_fused_epsilon(moe_post_norm) ||
        !same_fused_epsilon(output_post_norm)) return false;

    const auto* attention_payload = std::get_if<CausalAttentionPayload>(&attention->payload);
    if (!attention_payload || attention_payload->value_source != ValueSource::SeparateProjection) return false;
    const SemanticTensor* up_tensor = role_tensor(model, *expert_up, TensorRole::FfnUpWeight, true);
    const SemanticTensor* down_tensor = role_tensor(model, *expert_down, TensorRole::FfnDownWeight, true);
    if (!up_tensor || !down_tensor ||
        !exact_expert_format(model, *expert_up, TensorRole::FfnUpWeight, router_payload->expert_count,
                             hidden, 2 * split_payload->first_width, MetalWeightFormatQ4K)) return false;
    uint32_t down_format = 0;
    if (down_tensor->layout.block_elements == 32 && down_tensor->layout.block_bytes == 22) down_format = MetalWeightFormatQ5_0;
    if (down_tensor->layout.block_elements == 32 && down_tensor->layout.block_bytes == 34) down_format = MetalWeightFormatQ8_0;
    if (!down_format || !exact_expert_format(model, *expert_down, TensorRole::FfnDownWeight, router_payload->expert_count,
                                              split_payload->first_width, hidden, down_format)) return false;

    // Freeze one exact witness for both planner and lowerer.  The matcher may
    // discover edges in any order, but every operator in the layer must be one
    // of these typed nodes; an extra disconnected node or a cycle is not an
    // admissible native leaf.
    const SemanticOperator* embedding = unique_kind(OperatorKind::EmbeddingLookup);
    const SemanticOperator* rope = unique_kind(OperatorKind::Rope);
    const SemanticOperator* swiglu = unique_kind(OperatorKind::SwiGlu);
    const SemanticOperator* gated_attention = unique_kind(OperatorKind::GatedAttention);
    const SemanticOperator* query = layer_operator(model, layer, OperatorKind::Linear,
                                                    TensorRole::QueryWeight);
    const SemanticOperator* query_gate = layer_operator(model, layer, OperatorKind::Linear,
                                                        TensorRole::AttentionQueryGateWeight);
    const SemanticOperator* key = layer_operator(model, layer, OperatorKind::Linear,
                                                 TensorRole::KeyWeight);
    const SemanticOperator* value = layer_operator(model, layer, OperatorKind::Linear,
                                                   TensorRole::ValueWeight);
    const SemanticOperator* output = layer_operator(model, layer, OperatorKind::Linear,
                                                    TensorRole::AttentionOutputWeight);
    const SemanticOperator* query_norm = layer_operator(model, layer, OperatorKind::RmsNorm,
                                                        TensorRole::AttentionQueryNormWeight);
    const SemanticOperator* key_norm = layer_operator(model, layer, OperatorKind::RmsNorm,
                                                      TensorRole::AttentionKeyNormWeight);
    const SemanticOperator* dense_gate = layer_operator(model, layer, OperatorKind::Linear,
                                                        TensorRole::FfnGateWeight);
    const SemanticOperator* dense_up = layer_operator(model, layer, OperatorKind::Linear,
                                                      TensorRole::FfnUpWeight);
    std::vector<const SemanticOperator*> covered;
    const auto cover = [&](const SemanticOperator* op) {
        if (!op) return true;
        if (std::find(covered.begin(), covered.end(), op) != covered.end()) return true;
        covered.push_back(op);
        return true;
    };
    if (!cover(embedding) || !cover(attn_norm_witness) || !cover(query_gate ? query_gate : query) ||
        !cover(key) || !cover(value) || !cover(output) || !cover(query_norm) || !cover(key_norm) ||
        !cover(ffn_norm_witness) || !cover(dense_gate) || !cover(dense_up) || !cover(dense_down) ||
        !cover(rope) || !cover(attention) || !cover(swiglu) || !cover(gated_attention) ||
        !cover(router) || !cover(router_linear) || !cover(route_scale) ||
        !cover(normalization_scale) || !cover(router_norm) || !cover(expert_norm) ||
        !cover(expert_up) || !cover(split) || !cover(activation) || !cover(expert_down) ||
        !cover(reduce) || !cover(attention_residual) || !cover(residual) || !cover(final_add) || !cover(dense_post_norm) ||
        !cover(moe_post_norm) || !cover(output_post_norm) || !cover(output_scale) ||
        covered.size() != layer.operator_count) {
        return false;
    }

    std::vector<uint32_t> covered_operator_ids;
    covered_operator_ids.reserve(covered.size());
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (std::find(covered.begin(), covered.end(), &op) == covered.end() ||
            std::find(covered_operator_ids.begin(), covered_operator_ids.end(), op.id) !=
                covered_operator_ids.end()) return false;
        covered_operator_ids.push_back(op.id);
    }

    // Kahn's test is deliberately value-edge based, not serialized-index
    // based.  Self-edges fail closed; values entering from outside this layer
    // are graph inputs.  Ambiguous producers are handled by the typed edge
    // checks above (compact qualification fixtures may reuse value IDs for
    // unrelated dense edges).
    std::vector<uint32_t> indegree(covered.size(), 0);
    std::vector<std::vector<size_t>> successors(covered.size());
    for (size_t consumer = 0; consumer != covered.size(); ++consumer) {
        const SemanticOperator& op = *covered[consumer];
        for (uint32_t input : op.inputs) {
            size_t producer = covered.size();
            bool ambiguous = false;
            for (size_t candidate = 0; candidate != covered.size(); ++candidate) {
                if (std::find(covered[candidate]->outputs.begin(), covered[candidate]->outputs.end(), input) ==
                    covered[candidate]->outputs.end()) continue;
                if (producer != covered.size()) {
                    ambiguous = true;
                    break;
                }
                producer = candidate;
            }
            if (ambiguous) continue;
            if (producer == covered.size()) continue; // graph input owned outside this layer
            if (producer == consumer) return false;
            if (std::find(successors[producer].begin(), successors[producer].end(), consumer) ==
                successors[producer].end()) {
                successors[producer].push_back(consumer);
                ++indegree[consumer];
            }
        }
    }
    std::vector<size_t> ready;
    for (size_t index = 0; index != covered.size(); ++index)
        if (indegree[index] == 0) ready.push_back(index);
    size_t processed = 0;
    for (size_t cursor = 0; cursor != ready.size(); ++cursor) {
        ++processed;
        for (size_t successor : successors[ready[cursor]])
            if (--indegree[successor] == 0) ready.push_back(successor);
    }
    if (processed != covered.size()) {
        return false;
    }

    if (weight_format) *weight_format = dense_format | MetalWeightFormatQ4K | down_format;
    if (facts_out) {
        facts_out->covered_operator_ids = std::move(covered_operator_ids);
        facts_out->dense_attn_norm = attn_norm_witness;
        facts_out->dense_query = query;
        facts_out->dense_query_gate = query_gate;
        facts_out->dense_key = key;
        facts_out->dense_value = value;
        facts_out->dense_output = output;
        facts_out->dense_query_norm = query_norm;
        facts_out->dense_key_norm = key_norm;
        facts_out->dense_ffn_norm = ffn_norm_witness;
        facts_out->dense_gate = dense_gate;
        facts_out->dense_up = dense_up;
        facts_out->dense_down = dense_down;
        facts_out->rope = rope;
        facts_out->swiglu = swiglu;
        facts_out->attention = attention;
        facts_out->router = router;
        facts_out->router_linear = router_linear;
        facts_out->router_scale = route_scale;
        facts_out->router_normalization_scale = normalization_scale;
        facts_out->router_norm = router_norm;
        facts_out->expert_norm = expert_norm;
        facts_out->expert_up = expert_up;
        facts_out->expert_split = split;
        facts_out->expert_activation = activation;
        facts_out->expert_down = expert_down;
        facts_out->expert_reduce = reduce;
        facts_out->attention_residual = attention_residual;
        facts_out->residual = residual;
        facts_out->final_add = final_add;
        facts_out->dense_post_norm = dense_post_norm;
        facts_out->moe_post_norm = moe_post_norm;
        facts_out->output_post_norm = output_post_norm;
        facts_out->output_scale = output_scale;
        facts_out->gate_up_tensor = up_tensor;
        facts_out->down_tensor = down_tensor;
        facts_out->hidden = hidden;
        facts_out->intermediate = split_payload->first_width;
        facts_out->experts = router_payload->expert_count;
        facts_out->selected = router_payload->selected_count;
        facts_out->gate_up_format = MetalWeightFormatQ4K;
        facts_out->down_format = down_format;
        facts_out->gate_up_layout = up_tensor->layout.kind;
        facts_out->down_layout = down_tensor->layout.kind;
        facts_out->gate_up_quantization = up_tensor->quantization.kind;
        facts_out->down_quantization = down_tensor->quantization.kind;
        facts_out->gate_up_plane_mask = up_tensor->quantization.required_plane_mask;
        facts_out->down_plane_mask = down_tensor->quantization.required_plane_mask;
        facts_out->gate_up_block_elements = up_tensor->layout.block_elements;
        facts_out->gate_up_block_bytes = up_tensor->layout.block_bytes;
        facts_out->down_block_elements = down_tensor->layout.block_elements;
        facts_out->down_block_bytes = down_tensor->layout.block_bytes;
        facts_out->gate_up_input = hidden;
        facts_out->gate_up_output = 2 * split_payload->first_width;
        facts_out->down_input = split_payload->first_width;
        facts_out->down_output = hidden;
        facts_out->gate_up_expert_stride = up_tensor->expert_axis.per_expert_byte_stride;
        facts_out->down_expert_stride = down_tensor->expert_axis.per_expert_byte_stride;
        facts_out->activation = activation_payload->activation;
        facts_out->scale_source = reduce_payload->scale_source;
        facts_out->value_source = attention_payload->value_source;
        facts_out->router_normalization_scale_bits = normalization_payload->literal_f32_bits;
    }
    return true;
}

KernelQuery dense_metal_query(const SemanticModel& model, const SemanticLayer& layer,
                              const SemanticOperator& attention, ExecutionPhase phase,
                              uint32_t batch_rows = 1) {
    KernelQuery query = query_for(model, attention, phase);
    query.batch_rows = batch_rows;
    const SemanticOperator* candidate = nullptr;
    uint32_t format = 0;
    query.metal_dense_token_pattern = dense_metal_layer(model, layer, candidate, &format) && candidate == &attention;
    query.metal_weight_format_mask = format;
    query.metal_affine_u2_256 = (format & MetalWeightFormatAffineUInt2_256) != 0;
    query.metal_column_grouped_affine_u2_skip_256 =
        (format & MetalWeightFormatColumnGroupedAffineUInt2Skip256) != 0;
    if (format == MetalWeightFormatF16) {
        query.storage_type = ScalarType::F16;
        query.layout = PhysicalLayoutKind::ContiguousRowMajor;
        query.quantization = QuantizationKind::None;
        query.alignment = 64;
    } else if ((format & MetalWeightFormatF16) == 0) {
        query.storage_type = format == MetalWeightFormatAffineUInt2_256
            ? ScalarType::U32 : ScalarType::U8;
        query.layout = format == MetalWeightFormatAffineUInt2_256
            ? PhysicalLayoutKind::GroupedAffine
            : format == MetalWeightFormatColumnGroupedAffineUInt2Skip256
                ? PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip
                : PhysicalLayoutKind::GgufBlocked;
        query.quantization = QuantizationKind::BlockedAffine;
        query.alignment = (query.metal_affine_u2_256 ||
                           query.metal_column_grouped_affine_u2_skip_256) ? 128 : 32;
        if (const MetalBlockedFormat* blocked = metal_blocked_format(format)) {
            query.quantization = blocked->quantization;
            query.block_elements = blocked->elements;
            query.block_bytes = blocked->bytes;
        } else if (format == MetalWeightFormatAffineUInt2_256 ||
                   format == MetalWeightFormatColumnGroupedAffineUInt2Skip256) {
            query.block_elements = 256;
            query.block_bytes = 64;
        } else if ((format & MetalWeightFormatIQ2XXS) != 0) {
            query.quantization = QuantizationKind::Codebook;
        }
    } else if (format != MetalWeightFormatF16) {
        // A mixed layer has no single physical block tuple.  Keep the query
        // role-free and let the format mask carry the per-role evidence.
        query.storage_type = static_cast<ScalarType>(0);
        query.layout = static_cast<PhysicalLayoutKind>(0);
        query.quantization = static_cast<QuantizationKind>(0);
        query.alignment = (query.metal_affine_u2_256 ||
                           query.metal_column_grouped_affine_u2_skip_256) ? 128 : 32;
    }
    return query;
}

KernelQuery moe_metal_query(const SemanticModel& model, const SemanticLayer& layer,
                            const SemanticOperator& attention, ExecutionPhase phase) {
    KernelQuery query = query_for(model, attention, phase);
    const SemanticOperator* candidate = nullptr;
    uint32_t format = 0;
    MoeDescriptorFacts facts;
    query.metal_moe_token_pattern = moe_metal_layer(model, layer, candidate, &format, &facts) && candidate == &attention;
    query.metal_weight_format_mask = format;
    query.metal_affine_u2_256 = (format & MetalWeightFormatAffineUInt2_256) != 0;
    query.metal_column_grouped_affine_u2_skip_256 =
        (format & MetalWeightFormatColumnGroupedAffineUInt2Skip256) != 0;
    query.storage_type = ScalarType::U8;
    query.layout = PhysicalLayoutKind::GgufBlocked;
    query.quantization = QuantizationKind::BlockedAffine;
    query.alignment = 32;
    // A fused MoE layer can contain several independent physical block
    // contracts (dense attention, Q4_K gate/up, and Q5_0 or Q8_0 down).  Do
    // not collapse those into one arbitrary top-level block tuple; the
    // per-role facts below are the authoritative physical contract.
    query.block_elements = 0;
    query.block_bytes = 0;
    if (query.metal_moe_token_pattern) {
        query.moe_expert_count = facts.experts;
        query.moe_selected_count = facts.selected;
        query.moe_gate_up_format = facts.gate_up_format;
        query.moe_down_format = facts.down_format;
        query.moe_gate_up_layout = facts.gate_up_layout;
        query.moe_down_layout = facts.down_layout;
        query.moe_gate_up_quantization = facts.gate_up_quantization;
        query.moe_down_quantization = facts.down_quantization;
        query.moe_gate_up_plane_mask = facts.gate_up_plane_mask;
        query.moe_down_plane_mask = facts.down_plane_mask;
        query.moe_gate_up_block_elements = facts.gate_up_block_elements;
        query.moe_gate_up_block_bytes = facts.gate_up_block_bytes;
        query.moe_down_block_elements = facts.down_block_elements;
        query.moe_down_block_bytes = facts.down_block_bytes;
        query.moe_gate_up_input = facts.gate_up_input;
        query.moe_gate_up_output = facts.gate_up_output;
        query.moe_down_input = facts.down_input;
        query.moe_down_output = facts.down_output;
        query.moe_gate_up_expert_stride = facts.gate_up_expert_stride;
        query.moe_down_expert_stride = facts.down_expert_stride;
        query.moe_router_normalization_scale_bits = facts.router_normalization_scale_bits;
        query.moe_activation = facts.activation;
        query.moe_scale_source = facts.scale_source;
        query.moe_value_source = facts.value_source;
    }
    return query;
}

bool dense_prefill_batch_layer(const SemanticModel& model, const SemanticLayer& layer,
                               const SemanticOperator*& attention, uint32_t* weight_format = nullptr) {
    uint32_t format = 0;
    if (!dense_metal_layer(model, layer, attention, &format) || !attention ||
        format != MetalWeightFormatF16) return false;
    const auto exact_f16_64 = [](const SemanticTensor* tensor) {
        return tensor && tensor->logical_type == ScalarType::F32 &&
               tensor->dimensions.size() == 2 &&
               tensor->layout.kind == PhysicalLayoutKind::ContiguousRowMajor &&
               tensor->quantization.kind == QuantizationKind::None && tensor->planes.size() == 1 &&
               tensor->planes[0].kind == PlaneKind::Values &&
               tensor->planes[0].storage_type == ScalarType::F16 &&
               tensor->planes[0].alignment == 64;
    };
    const auto unique_role = [&](TensorRole role) {
        const SemanticTensor* found = nullptr;
        for (const SemanticTensor& tensor : model.tensors) {
            if (tensor.role != role) continue;
            if (found) return static_cast<const SemanticTensor*>(nullptr);
            found = &tensor;
        }
        return found;
    };
    if (!exact_f16_64(unique_role(TensorRole::TokenEmbedding)) ||
        !exact_f16_64(unique_role(TensorRole::OutputWeight))) return false;
    if (layer.first_operator > model.operators.size() ||
        layer.operator_count > model.operators.size() - layer.first_operator) return false;
    uint32_t rms_count = 0;
    uint32_t linear_count = 0;
    uint32_t rope_count = 0;
    uint32_t attention_count = 0;
    uint32_t swiglu_count = 0;
    uint32_t add_count = 0;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        switch (op.kind) {
        case OperatorKind::RmsNorm: ++rms_count; break;
        case OperatorKind::Linear: ++linear_count; break;
        case OperatorKind::Rope: ++rope_count; break;
        case OperatorKind::CausalAttention: ++attention_count; break;
        case OperatorKind::SwiGlu: ++swiglu_count; break;
        case OperatorKind::Add: ++add_count; break;
        default: return false;
        }
        for (uint32_t tensor_id : op.tensors) {
            if (tensor_id >= model.tensors.size()) return false;
            switch (model.tensors[tensor_id].role) {
            case TensorRole::QueryWeight:
            case TensorRole::KeyWeight:
            case TensorRole::ValueWeight:
            case TensorRole::AttentionOutputWeight:
            case TensorRole::FfnGateWeight:
            case TensorRole::FfnUpWeight:
            case TensorRole::FfnDownWeight:
                if (!exact_f16_64(&model.tensors[tensor_id])) return false;
                break;
            case TensorRole::QueryBias:
            case TensorRole::KeyBias:
            case TensorRole::ValueBias:
            case TensorRole::AttentionQueryGateWeight:
            case TensorRole::AttentionQueryNormWeight:
            case TensorRole::AttentionKeyNormWeight:
                return false;
            default:
                break;
            }
        }
        if (op.kind == OperatorKind::Rope) {
            const auto* rope = std::get_if<RopePayload>(&op.payload);
            if (!rope || rope->pairing != RopePairing::HalfSplit) return false;
        }
    }
    if (rms_count != 2 || linear_count != 7 || rope_count != 1 || attention_count != 1 ||
        swiglu_count != 1 || add_count != 2) return false;
    if (weight_format) *weight_format = format;
    return true;
}

KernelQuery dense_prefill_batch_query(const SemanticModel& model, const SemanticLayer& layer,
                                      const SemanticOperator& attention, uint32_t batch_rows) {
    KernelQuery query = dense_metal_query(model, layer, attention, ExecutionPhase::Prefill, batch_rows);
    const SemanticOperator* candidate = nullptr;
    query.metal_dense_prefill_batch_pattern =
        batch_rows == 2 && dense_prefill_batch_layer(model, layer, candidate) && candidate == &attention;
    return query;
}

bool recurrent_state(const SemanticState& state, StateKind kind, LayoutPolicy layout, uint16_t semantic_version,
                     std::initializer_list<uint64_t> dimensions) {
    if (state.kind != kind || state.semantic_version != semantic_version ||
        state.position_policy != PositionPolicy::ReplaceAtCursor ||
        state.formats.size() != 1 || state.dimensions.size() != dimensions.size()) return false;
    const StateFormat& format = state.formats[0];
    if (format.kind != StateFormatKind::RecurrentContiguous || format.version != 1 ||
        format.logical_type != ScalarType::F32 || format.encoded_type != ScalarType::F32 ||
        format.logical_domain != TransformDomain::Untransformed || format.encoded_domain != TransformDomain::Untransformed ||
        format.codec != CodecKind::Fp32 || format.cache_policy != CachePolicy::Recurrent ||
        format.layout_policy != layout || format.flags != 0 || format.tile_tokens != 0 ||
        format.mutable_tokens != 0 || format.alignment != 64 || format.reserved != 0) return false;
    size_t index = 0;
    for (uint64_t expected : dimensions) {
        const Dimension& dimension = state.dimensions[index++];
        if (dimension.kind != DimensionKind::Constant || dimension.constant_or_symbol != expected) return false;
    }
    return true;
}

bool recurrent_metal_layer(const SemanticModel& model, const SemanticLayer& layer,
                           const SemanticOperator*& delta_out, uint32_t* weight_format = nullptr) {
    if (layer.first_operator > model.operators.size() || layer.operator_count != 18 ||
        layer.operator_count > model.operators.size() - layer.first_operator) return false;
    const SemanticOperator* input_norm = layer_operator(model, layer, OperatorKind::RmsNorm, TensorRole::AttentionNormWeight);
    const SemanticOperator* qkv = layer_operator(model, layer, OperatorKind::Linear, TensorRole::RecurrentQkvWeight);
    const SemanticOperator* gate = layer_operator(model, layer, OperatorKind::Linear, TensorRole::RecurrentGateWeight);
    const SemanticOperator* beta = layer_operator(model, layer, OperatorKind::Linear, TensorRole::RecurrentBetaWeight);
    const SemanticOperator* alpha = layer_operator(model, layer, OperatorKind::Linear, TensorRole::RecurrentAlphaWeight);
    const SemanticOperator* output = layer_operator(model, layer, OperatorKind::Linear, TensorRole::RecurrentOutputWeight);
    const SemanticOperator* ffn_norm = layer_operator(model, layer, OperatorKind::RmsNorm, TensorRole::FfnNormWeight);
    const SemanticOperator* ffn_gate = layer_operator(model, layer, OperatorKind::Linear, TensorRole::FfnGateWeight);
    const SemanticOperator* ffn_up = layer_operator(model, layer, OperatorKind::Linear, TensorRole::FfnUpWeight);
    const SemanticOperator* ffn_down = layer_operator(model, layer, OperatorKind::Linear, TensorRole::FfnDownWeight);
    const SemanticOperator* conv = nullptr;
    std::vector<const SemanticOperator*> l2_operators;
    const SemanticOperator* gated_rms = nullptr;
    const SemanticOperator* swiglu = nullptr;
    const SemanticOperator* recurrent_residual = nullptr;
    const SemanticOperator* ffn_residual = nullptr;
    delta_out = nullptr;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind == OperatorKind::DepthwiseConvSilu) conv = conv ? nullptr : &op;
        if (op.kind == OperatorKind::L2Normalize) l2_operators.push_back(&op);
        if (op.kind == OperatorKind::GatedDeltaNet) delta_out = delta_out ? nullptr : &op;
        if (op.kind == OperatorKind::GatedRmsNorm) gated_rms = gated_rms ? nullptr : &op;
        if (op.kind == OperatorKind::SwiGlu) swiglu = swiglu ? nullptr : &op;
        if (op.kind == OperatorKind::Add) {
            if (!recurrent_residual) recurrent_residual = &op;
            else if (!ffn_residual) ffn_residual = &op;
            else ffn_residual = nullptr;
        }
    }
    const SemanticOperator* l2_query = nullptr;
    const SemanticOperator* l2_key = nullptr;
    if (conv && conv->outputs.size() >= 2 && l2_operators.size() == 2) {
        for (const SemanticOperator* candidate : l2_operators) {
            if (candidate->inputs == std::vector<uint32_t>{conv->outputs[0]}) {
                if (l2_query) {
                    l2_query = nullptr;
                    break;
                }
                l2_query = candidate;
            } else if (candidate->inputs == std::vector<uint32_t>{conv->outputs[1]}) {
                if (l2_key) {
                    l2_key = nullptr;
                    break;
                }
                l2_key = candidate;
            }
        }
    }
    const auto* input_norm_payload = input_norm ? std::get_if<RmsNormPayload>(&input_norm->payload) : nullptr;
    const auto* conv_payload = conv ? std::get_if<DepthwiseConvSiluPayload>(&conv->payload) : nullptr;
    const auto* delta_payload = delta_out ? std::get_if<GatedDeltaNetPayload>(&delta_out->payload) : nullptr;
    const auto* l2_query_payload = l2_query ? std::get_if<L2NormalizePayload>(&l2_query->payload) : nullptr;
    const auto* l2_key_payload = l2_key ? std::get_if<L2NormalizePayload>(&l2_key->payload) : nullptr;
    const auto* gated_rms_payload = gated_rms ? std::get_if<GatedRmsNormPayload>(&gated_rms->payload) : nullptr;
    const auto* ffn_norm_payload = ffn_norm ? std::get_if<RmsNormPayload>(&ffn_norm->payload) : nullptr;
    const auto* swiglu_payload = swiglu ? std::get_if<SwiGluPayload>(&swiglu->payload) : nullptr;
    const float input_norm_epsilon = input_norm_payload
        ? std::bit_cast<float>(input_norm_payload->epsilon_f32_bits) : -1.0f;
    const float l2_query_epsilon = l2_query_payload
        ? std::bit_cast<float>(l2_query_payload->epsilon_f32_bits) : -1.0f;
    const float l2_key_epsilon = l2_key_payload
        ? std::bit_cast<float>(l2_key_payload->epsilon_f32_bits) : -1.0f;
    const float gated_rms_epsilon = gated_rms_payload
        ? std::bit_cast<float>(gated_rms_payload->epsilon_f32_bits) : -1.0f;
    const float ffn_norm_epsilon = ffn_norm_payload
        ? std::bit_cast<float>(ffn_norm_payload->epsilon_f32_bits) : -1.0f;
    const uint16_t semantic_version = delta_out ? delta_out->semantic_version : 0;
    if (semantic_version != 3 && semantic_version != 4 && semantic_version != 6 &&
        semantic_version != 8) return false;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        if (model.operators[layer.first_operator + index].semantic_version != semantic_version) return false;
    }
    if (!input_norm || !qkv || !gate || !beta || !alpha || !output || !ffn_norm || !ffn_gate || !ffn_up || !ffn_down ||
        !conv || !l2_query || !l2_key || !delta_out || !gated_rms || !swiglu || !recurrent_residual || !ffn_residual ||
        !input_norm_payload || !conv_payload || !delta_payload || !l2_query_payload || !l2_key_payload ||
        !gated_rms_payload || !ffn_norm_payload || !swiglu_payload ||
        !std::isfinite(input_norm_epsilon) || input_norm_epsilon < 0.0f ||
        !std::isfinite(l2_query_epsilon) || l2_query_epsilon < 0.0f ||
        !std::isfinite(l2_key_epsilon) || l2_key_epsilon < 0.0f ||
        !std::isfinite(gated_rms_epsilon) || gated_rms_epsilon < 0.0f ||
        !std::isfinite(ffn_norm_epsilon) || ffn_norm_epsilon < 0.0f ||
        conv_payload->qk_heads != delta_payload->qk_heads || conv_payload->value_heads != delta_payload->value_heads ||
        conv_payload->head_dimension != delta_payload->head_dimension || conv->states.size() != 1 || delta_out->states.size() != 1 ||
        input_norm->tensors.size() != 1 || conv->tensors.size() != 1 || delta_out->tensors.size() != 2 ||
        gated_rms->tensors.size() != 1 || ffn_norm->tensors.size() != 1 ||
        recurrent_residual->tensors.size() != 0 || ffn_residual->tensors.size() != 0 || swiglu->tensors.size() != 0 ||
        l2_query_payload->epsilon_f32_bits != l2_key_payload->epsilon_f32_bits ||
        input_norm_payload->axis != -1 || input_norm_payload->weight_mode != 1 ||
        ffn_norm_payload->axis != -1 || ffn_norm_payload->weight_mode != 1 ||
        input_norm_payload->epsilon_f32_bits != ffn_norm_payload->epsilon_f32_bits ||
        input_norm_payload->epsilon_f32_bits != gated_rms_payload->epsilon_f32_bits ||
        gated_rms_payload->gate_activation != ActivationKind::Silu || gated_rms_payload->weight_mode != 1 ||
        swiglu_payload->activation != ActivationKind::Silu ||
        input_norm->inputs.size() != 1 || input_norm->outputs.size() != 1 ||
        qkv->inputs != input_norm->outputs || gate->inputs != qkv->inputs || beta->inputs != qkv->inputs || alpha->inputs != qkv->inputs ||
        qkv->outputs.size() != 1 || gate->outputs.size() != 1 || beta->outputs.size() != 1 || alpha->outputs.size() != 1 ||
        conv->inputs != qkv->outputs || conv->outputs.size() != 3 ||
        l2_query->inputs != std::vector<uint32_t>{conv->outputs[0]} || l2_query->outputs.size() != 1 ||
        l2_key->inputs != std::vector<uint32_t>{conv->outputs[1]} || l2_key->outputs.size() != 1 ||
        delta_out->inputs != std::vector<uint32_t>{l2_query->outputs[0], l2_key->outputs[0], conv->outputs[2],
                                                    beta->outputs[0], alpha->outputs[0]} || delta_out->outputs.size() != 1 ||
        gated_rms->inputs != std::vector<uint32_t>{delta_out->outputs[0], gate->outputs[0]} || gated_rms->outputs.size() != 1 ||
        output->inputs != gated_rms->outputs || output->outputs.size() != 1 || output->tensors.size() != 1 ||
        recurrent_residual->inputs != std::vector<uint32_t>{input_norm->inputs[0], output->outputs[0]} ||
        recurrent_residual->outputs.size() != 1 || ffn_norm->inputs != recurrent_residual->outputs ||
        ffn_norm->outputs.size() != 1 || ffn_gate->inputs != ffn_norm->outputs || ffn_up->inputs != ffn_norm->outputs ||
        ffn_gate->outputs.size() != 1 || ffn_up->outputs.size() != 1 ||
        swiglu->inputs != std::vector<uint32_t>{ffn_gate->outputs[0], ffn_up->outputs[0]} || swiglu->outputs.size() != 1 ||
        ffn_down->inputs != swiglu->outputs || ffn_down->outputs.size() != 1 ||
        ffn_residual->inputs != std::vector<uint32_t>{recurrent_residual->outputs[0], ffn_down->outputs[0]} ||
        ffn_residual->outputs.size() != 1) return false;
    const uint64_t channels = static_cast<uint64_t>(delta_payload->head_dimension) *
                              (2ull * delta_payload->qk_heads + delta_payload->value_heads);
    if (channels > UINT32_MAX) return false;
    const SemanticState* conv_state = state_with_id(model, conv->states[0]);
    const SemanticState* delta_state = state_with_id(model, delta_out->states[0]);
    if (!conv_state || !delta_state ||
        !recurrent_state(*conv_state, StateKind::RecurrentConvHistory, LayoutPolicy::ChannelMajorHistory,
                         semantic_version, {channels, conv_payload->kernel - 1}) ||
        !recurrent_state(*delta_state, StateKind::RecurrentDeltaMatrix, LayoutPolicy::ValueHeadKeyRowOutputColumn,
                         semantic_version,
                         {delta_payload->value_heads, delta_payload->head_dimension, delta_payload->head_dimension}) ||
        !exact_dense_tensor(model, *input_norm, TensorRole::AttentionNormWeight, ScalarType::F32) ||
        !exact_dense_tensor(model, *conv, TensorRole::RecurrentConvWeight, ScalarType::F32) ||
        !exact_dense_tensor(model, *delta_out, TensorRole::RecurrentDtBias, ScalarType::F32) ||
        !exact_dense_tensor(model, *delta_out, TensorRole::RecurrentDecayWeight, ScalarType::F32) ||
        !exact_dense_tensor(model, *gated_rms, TensorRole::RecurrentNormWeight, ScalarType::F32) ||
        !exact_dense_tensor(model, *ffn_norm, TensorRole::FfnNormWeight, ScalarType::F32)) return false;
    uint32_t format = 0;
    for (const auto [op, role] : std::array<std::pair<const SemanticOperator*, TensorRole>, 8>{{
             {qkv, TensorRole::RecurrentQkvWeight}, {gate, TensorRole::RecurrentGateWeight},
             {beta, TensorRole::RecurrentBetaWeight}, {alpha, TensorRole::RecurrentAlphaWeight},
             {output, TensorRole::RecurrentOutputWeight}, {ffn_gate, TensorRole::FfnGateWeight},
             {ffn_up, TensorRole::FfnUpWeight}, {ffn_down, TensorRole::FfnDownWeight}}}) {
        uint32_t candidate = 0;
        if (!metal_dense_matrix_format(model, *op, role, candidate)) return false;
        format |= candidate;
    }
    if (weight_format) *weight_format = format;
    return true;
}

KernelQuery recurrent_metal_query(const SemanticModel& model, const SemanticLayer& layer,
                                  const SemanticOperator& delta, ExecutionPhase phase) {
    KernelQuery query = query_for(model, delta, phase);
    const auto* payload = std::get_if<GatedDeltaNetPayload>(&delta.payload);
    uint32_t format = 0;
    const SemanticOperator* candidate = nullptr;
    query.metal_recurrent_token_pattern = recurrent_metal_layer(model, layer, candidate, &format) && candidate == &delta;
    query.metal_weight_format_mask = format;
    query.metal_affine_u2_256 = (format & MetalWeightFormatAffineUInt2_256) != 0;
    query.metal_column_grouped_affine_u2_skip_256 =
        (format & MetalWeightFormatColumnGroupedAffineUInt2Skip256) != 0;
    if (format && (format & MetalWeightFormatF16) == 0) {
        query.storage_type = format == MetalWeightFormatAffineUInt2_256
            ? ScalarType::U32 : ScalarType::U8;
        query.layout = format == MetalWeightFormatAffineUInt2_256
            ? PhysicalLayoutKind::GroupedAffine
            : format == MetalWeightFormatColumnGroupedAffineUInt2Skip256
                ? PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip
                : PhysicalLayoutKind::GgufBlocked;
        query.quantization = QuantizationKind::BlockedAffine;
        query.alignment = (query.metal_affine_u2_256 ||
                           query.metal_column_grouped_affine_u2_skip_256) ? 128 : 32;
        if (const MetalBlockedFormat* blocked = metal_blocked_format(format)) {
            query.quantization = blocked->quantization;
            query.block_elements = blocked->elements;
            query.block_bytes = blocked->bytes;
        } else if ((format & MetalWeightFormatIQ2XXS) != 0) {
            query.quantization = QuantizationKind::Codebook;
        } else if (format == MetalWeightFormatAffineUInt2_256 ||
                   format == MetalWeightFormatColumnGroupedAffineUInt2Skip256) {
            query.block_elements = 256;
            query.block_bytes = 64;
        }
    } else if (format != MetalWeightFormatF16) {
        query.storage_type = static_cast<ScalarType>(0);
        query.layout = static_cast<PhysicalLayoutKind>(0);
        query.quantization = static_cast<QuantizationKind>(0);
        query.alignment = (query.metal_affine_u2_256 ||
                           query.metal_column_grouped_affine_u2_skip_256) ? 128 : 32;
    }
    if (payload) query.head_dimension = payload->head_dimension;
    return query;
}

bool canonical_metal_operator_coverage(const SemanticModel& model,
                                       uint32_t& unexpected_operator,
                                       std::string& detail) {
    std::vector<bool> covered(model.operators.size(), false);
    for (const SemanticLayer& layer : model.layers) {
        if (layer.first_operator > model.operators.size() ||
            layer.operator_count > model.operators.size() - layer.first_operator) {
            detail = "canonical Metal layer operator range is invalid";
            return false;
        }
        for (uint32_t index = 0; index != layer.operator_count; ++index) {
            if (covered[layer.first_operator + index]) {
                unexpected_operator = layer.first_operator + index;
                detail = "canonical Metal layer operator ranges overlap";
                return false;
            }
            covered[layer.first_operator + index] = true;
        }
    }
    uint32_t embedding = 0;
    uint32_t final_norm = 0;
    uint32_t output = 0;
    for (uint32_t index = 0; index != model.operators.size(); ++index) {
        if (covered[index]) continue;
        const SemanticOperator& op = model.operators[index];
        if (op.kind == OperatorKind::EmbeddingLookup) {
            uint32_t embedding_format = 0;
            if (exact_dense_tensor(model, op, TensorRole::TokenEmbedding, ScalarType::F16) ||
                (metal_dense_matrix_format(model, op, TensorRole::TokenEmbedding, embedding_format) &&
                 (embedding_format == MetalWeightFormatQ4K || embedding_format == MetalWeightFormatQ6K))) {
                ++embedding;
                continue;
            }
        }
        if (op.kind == OperatorKind::RmsNorm && exact_dense_tensor(model, op, TensorRole::FinalNormWeight, ScalarType::F32)) {
            ++final_norm;
            continue;
        }
        if (op.kind == OperatorKind::Linear) {
            uint32_t output_format = 0;
            if (exact_dense_tensor(model, op, TensorRole::OutputWeight, ScalarType::F16) ||
                (metal_dense_matrix_format(model, op, TensorRole::OutputWeight, output_format) &&
                 output_format == MetalWeightFormatQ6K)) {
                ++output;
                continue;
            }
        }
        unexpected_operator = op.id;
        detail = "canonical Metal graph-level operator is unsupported";
        return false;
    }
    if (embedding != 1 || final_norm != 1 || output != 1) {
        detail = "canonical Metal requires exactly one graph-level embedding, final norm, and output projection; observed " +
                 std::to_string(embedding) + "/" + std::to_string(final_norm) + "/" +
                 std::to_string(output);
        return false;
    }
    return true;
}

bool checked_multiply(uint64_t left, uint64_t right, uint64_t& result) {
    if (left && right > std::numeric_limits<uint64_t>::max() / left) return false;
    result = left * right;
    return true;
}

bool checked_add(uint64_t left, uint64_t right, uint64_t& result) {
    if (right > std::numeric_limits<uint64_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool value_bytes(const SemanticValue& value, const SessionRequest& request, uint64_t& bytes) {
    uint64_t elements = 1;
    for (const Dimension& dimension : value.dimensions) {
        const uint64_t extent = dimension.kind == DimensionKind::Symbol ? request.max_context : dimension.constant_or_symbol;
        if (!extent || !checked_multiply(elements, extent, elements)) return false;
    }
    return checked_multiply(elements, sizeof(float), bytes);
}

bool peak_bytes(const SemanticModel& model, const SessionRequest& request, uint64_t state, uint64_t& result) {
    uint64_t values = 0;
    uint64_t scratch = 0;
    for (const SemanticValue& value : model.values) {
        uint64_t bytes = 0;
        if (!value_bytes(value, request, bytes) || !checked_add(values, bytes, values)) return false;
        scratch = std::max(scratch, bytes);
    }
    uint64_t total = 0;
    return checked_add(state, values, total) && checked_add(total, scratch, total) && (result = total, true);
}

KernelQuery query_for(const SemanticModel& model, const SemanticOperator& op, ExecutionPhase phase) {
    KernelQuery query;
    query.operation = op.kind;
    query.semantic_version = op.semantic_version;
    query.phase = phase;
    if (!op.outputs.empty() && op.outputs[0] < model.values.size()) {
        const SemanticValue& value = model.values[op.outputs[0]];
        query.logical_type = value.logical_type;
        query.rank = static_cast<uint32_t>(value.dimensions.size());
    }
    bool first_tensor = true;
    for (uint32_t tensor_id : op.tensors) {
        if (tensor_id >= model.tensors.size()) continue;
        const SemanticTensor& tensor = model.tensors[tensor_id];
        if (first_tensor) {
            const auto values = std::find_if(
                tensor.planes.begin(), tensor.planes.end(), [](const TensorPlane& plane) {
                    return plane.kind == PlaneKind::Values;
                });
            query.layout = tensor.layout.kind;
            query.quantization = tensor.quantization.kind;
            query.alignment = values == tensor.planes.end() ? 0 : values->alignment;
            query.storage_type = values == tensor.planes.end()
                ? static_cast<ScalarType>(0) : values->storage_type;
            first_tensor = false;
        } else if (query.layout != tensor.layout.kind || query.quantization != tensor.quantization.kind) {
            query.layout = static_cast<PhysicalLayoutKind>(0);
            query.quantization = static_cast<QuantizationKind>(UINT16_MAX);
        }
    }
    if (!op.states.empty()) {
        const SemanticState* state = state_with_id(model, op.states[0]);
        if (!state) return query;
        query.state_kind = state->kind;
        if (!state->formats.empty()) {
            query.state_format = state->formats[0].kind;
            query.alignment = state->formats[0].alignment;
            query.tile_tokens = state->formats[0].tile_tokens;
        }
    }
    if (const auto* attention = std::get_if<CausalAttentionPayload>(&op.payload)) query.head_dimension = attention->head_dimension;
    if (const auto* delta = std::get_if<GatedDeltaNetPayload>(&op.payload)) query.head_dimension = delta->head_dimension;
    return query;
}

uint64_t state_bytes(const SemanticModel& model, const SessionRequest& request, bool& valid) {
    uint64_t total = static_cast<uint64_t>(request.max_batch) * sizeof(float);
    valid = true;
    for (const SemanticState& state : model.states) {
        uint64_t elements = 1;
        for (const Dimension& dimension : state.dimensions) {
            const uint64_t extent = dimension.kind == DimensionKind::Symbol ? request.max_context : dimension.constant_or_symbol;
            if (!extent || !checked_multiply(elements, extent, elements)) {
                valid = false;
                return 0;
            }
        }
        uint64_t bytes = 0;
        if (!checked_multiply(elements, sizeof(float), bytes) || !checked_add(total, bytes, total)) {
            valid = false;
            return 0;
        }
    }
    return total;
}

} // namespace

namespace {

struct CodecRegistryIndex {
    std::vector<const PhysicalCodecSpec*> codecs;
    std::vector<const PhysicalTensorCodecDeclaration*> tensors;
};

std::optional<CodecRegistryIndex> build_codec_registry_index(
    const PhysicalCodecRegistry& registry) {
    if (!validate_physical_codec_registry(registry)) return std::nullopt;
    CodecRegistryIndex index;
    index.codecs.reserve(registry.codecs.size());
    for (const PhysicalCodecSpec& codec : registry.codecs) {
        index.codecs.push_back(&codec);
    }
    std::sort(index.codecs.begin(), index.codecs.end(),
              [](const PhysicalCodecSpec* left, const PhysicalCodecSpec* right) {
        return physical_codec_identity_less(left->identity, right->identity);
    });
    index.tensors.reserve(registry.tensors.size());
    for (const PhysicalTensorCodecDeclaration& tensor : registry.tensors) {
        index.tensors.push_back(&tensor);
    }
    std::sort(index.tensors.begin(), index.tensors.end(),
              [](const PhysicalTensorCodecDeclaration* left,
                 const PhysicalTensorCodecDeclaration* right) {
        return left->tensor_id < right->tensor_id;
    });
    return index;
}

bool valid_tensor_strides(const SemanticTensor& tensor) {
    if (tensor.dimensions.empty() || tensor.dimensions.size() != tensor.layout.rank) return false;
    uint64_t maximum_address = 0;
    for (size_t index = 0; index != tensor.dimensions.size(); ++index) {
        const Dimension& dimension = tensor.dimensions[index];
        if (dimension.kind != DimensionKind::Constant || dimension.constant_or_symbol == 0 ||
            tensor.layout.strides[index] == 0) return false;
        uint64_t contribution = 0;
        if (!checked_multiply(dimension.constant_or_symbol - 1,
                              tensor.layout.strides[index], contribution) ||
            !checked_add(maximum_address, contribution, maximum_address)) return false;
    }
    for (size_t index = tensor.dimensions.size(); index != tensor.layout.strides.size(); ++index) {
        if (tensor.layout.strides[index] != 0) return false;
    }
    return true;
}

bool power_of_two(uint32_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

std::optional<PhysicalTensorCodecBinding> binding_for_identity(
    const SemanticTensor& tensor, const PhysicalCodecIdentity& identity) {
    if (!valid_physical_codec_identity(identity) || !valid_tensor_strides(tensor)) return std::nullopt;

    const auto actual_identity = physical_codec_identity(
        tensor, identity.arithmetic_version, identity.arithmetic_digest,
        identity.codebook_digest);
    if (!actual_identity || *actual_identity != identity) return std::nullopt;

    PhysicalTensorCodecBinding binding;
    binding.tensor_id = tensor.id;
    binding.identity = identity;
    binding.dimensions = tensor.dimensions;
    binding.strides = tensor.layout.strides;
    binding.planes.reserve(identity.planes.size());
    for (const PhysicalPlaneSchema& schema : identity.planes) {
        const auto plane = std::find_if(tensor.planes.begin(), tensor.planes.end(),
                                        [&](const TensorPlane& candidate) {
            return candidate.kind == schema.kind;
        });
        if (plane == tensor.planes.end() || plane->storage_type != schema.storage_type ||
            plane->flags != schema.flags || !power_of_two(plane->alignment) ||
            plane->length == 0 || plane->offset > UINT64_MAX - plane->length ||
            plane->offset % plane->alignment != 0) return std::nullopt;
        binding.planes.push_back({plane->kind, plane->storage_type, plane->artifact_id,
                                  plane->offset, plane->length, plane->alignment,
                                  plane->flags});
    }
    if (binding.planes.size() != tensor.planes.size()) return std::nullopt;
    return binding;
}

std::optional<PhysicalTensorCodecBinding> bind_physical_codec(
    const SemanticTensor& tensor, const CodecRegistryIndex& registry) {
    const auto declaration = std::lower_bound(
        registry.tensors.begin(), registry.tensors.end(), tensor.id,
        [](const PhysicalTensorCodecDeclaration* candidate, uint32_t tensor_id) {
            return candidate->tensor_id < tensor_id;
        });
    if (declaration == registry.tensors.end() || (*declaration)->tensor_id != tensor.id) {
        return std::nullopt;
    }
    return binding_for_identity(tensor, (*declaration)->identity);
}

bool binding_matches_tensor(const SemanticTensor& tensor,
                            const PhysicalTensorCodecBinding& binding) {
    if (binding.tensor_id != tensor.id) return false;
    const auto expected = binding_for_identity(tensor, binding.identity);
    return expected && *expected == binding;
}

} // namespace

bool canonical_layer_chain_witness(const SemanticModel& model,
                                   std::vector<CanonicalLayerBoundary>& witness,
                                   std::string& detail) {
    constexpr uint32_t unassigned = std::numeric_limits<uint32_t>::max();
    witness.clear();
    detail.clear();
    if (model.layers.empty() || model.values.empty() || model.operators.empty()) {
        detail = "canonical Metal layer residual chain is empty";
        return false;
    }

    std::vector<uint32_t> operator_layer(model.operators.size(), unassigned);
    std::vector<bool> disabled_operator(model.operators.size(), false);
    std::vector<const SemanticLayer*> active_layers;
    uint32_t previous_end = 0;
    bool have_previous = false;
    for (const SemanticLayer& layer : model.layers) {
        if (layer.first_operator > model.operators.size() ||
            layer.operator_count == 0 ||
            layer.operator_count > model.operators.size() - layer.first_operator) {
            detail = "canonical Metal layer residual chain has an invalid layer range";
            return false;
        }
        const bool disabled = (layer.flags & kSemanticLayerFlagSpeculative) != 0;
        if (!disabled && have_previous && layer.first_operator != previous_end) {
            detail = "canonical Metal layer residual chain has an inter-layer operator gap";
            return false;
        }
        const uint32_t active_index = static_cast<uint32_t>(active_layers.size());
        for (uint32_t offset = 0; offset != layer.operator_count; ++offset) {
            const uint32_t operator_index = layer.first_operator + offset;
            if (operator_layer[operator_index] != unassigned || disabled_operator[operator_index]) {
                detail = "canonical Metal layer residual chain has overlapping layer ranges";
                return false;
            }
            if (disabled) disabled_operator[operator_index] = true;
            else operator_layer[operator_index] = active_index;
        }
        if (!disabled) {
            active_layers.push_back(&layer);
            previous_end = layer.first_operator + layer.operator_count;
            have_previous = true;
        }
    }
    if (active_layers.empty()) {
        detail = "canonical Metal layer residual chain has no executable layer";
        return false;
    }

    std::vector<uint32_t> producer(model.values.size(), unassigned);
    std::vector<std::vector<uint32_t>> consumers(model.values.size());
    for (uint32_t operator_index = 0; operator_index != model.operators.size();
         ++operator_index) {
        if (disabled_operator[operator_index]) continue;
        const SemanticOperator& op = model.operators[operator_index];
        for (uint32_t value_id : op.outputs) {
            if (value_id >= model.values.size() || producer[value_id] != unassigned) {
                detail = "canonical Metal layer residual chain has an invalid value producer";
                return false;
            }
            producer[value_id] = operator_index;
        }
        for (uint32_t value_id : op.inputs) {
            if (value_id >= model.values.size()) {
                detail = "canonical Metal layer residual chain has an invalid value consumer";
                return false;
            }
            consumers[value_id].push_back(operator_index);
        }
    }

    const auto graph_output = [&](uint32_t value_id) {
        return model.output_values_count != 0 &&
               value_id >= model.output_values_first &&
               value_id - model.output_values_first < model.output_values_count;
    };
    uint32_t previous_output = unassigned;
    for (uint32_t active_index = 0; active_index != active_layers.size(); ++active_index) {
        const SemanticLayer& layer = *active_layers[active_index];
        std::vector<uint32_t> external_inputs;
        std::vector<uint32_t> external_outputs;
        for (uint32_t offset = 0; offset != layer.operator_count; ++offset) {
            const SemanticOperator& op = model.operators[layer.first_operator + offset];
            for (uint32_t value_id : op.inputs) {
                const uint32_t source = producer[value_id];
                if (source == unassigned || operator_layer[source] != active_index)
                    external_inputs.push_back(value_id);
            }
            for (uint32_t value_id : op.outputs) {
                bool escapes = graph_output(value_id) || consumers[value_id].empty();
                for (uint32_t consumer : consumers[value_id])
                    escapes = escapes || operator_layer[consumer] != active_index;
                if (escapes) external_outputs.push_back(value_id);
            }
        }
        const auto unique = [](std::vector<uint32_t>& values) {
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
        };
        unique(external_inputs);
        unique(external_outputs);
        if (external_inputs.size() != 1 || external_outputs.size() != 1) {
            detail = "canonical Metal layer residual chain requires one input and one output at layer " +
                     std::to_string(layer.layer_index);
            return false;
        }
        if (previous_output != unassigned && external_inputs.front() != previous_output) {
            detail = "canonical Metal layer residual chain bypasses the previous layer at layer " +
                     std::to_string(layer.layer_index);
            return false;
        }
        witness.push_back({layer.layer_index, external_inputs.front(), external_outputs.front()});
        previous_output = external_outputs.front();
    }
    return true;
}

bool canonical_program_witness(const SemanticModel& model,
                               CanonicalProgramWitness& witness,
                               std::string& detail) {
    witness = {};
    detail.clear();
    if (!canonical_layer_chain_witness(model, witness.layers, detail)) return false;

    std::vector<bool> layer_operator(model.operators.size(), false);
    for (size_t layer_index = 0; layer_index != model.layers.size(); ++layer_index) {
        const SemanticLayer& layer = model.layers[layer_index];
        if (layer.layer_index != layer_index) {
            detail = "canonical Metal layer table has a non-canonical layer index";
            return false;
        }
        if (layer.flags != 0) {
            detail = "canonical Metal primary program cannot skip a declared layer";
            return false;
        }
        for (uint32_t offset = 0; offset != layer.operator_count; ++offset)
            layer_operator[layer.first_operator + offset] = true;
    }
    for (uint32_t index = 0; index != model.operators.size(); ++index) {
        if (model.operators[index].id != index) {
            detail = "canonical Metal operator table is not canonical";
            return false;
        }
    }

    const SemanticOperator* embedding = nullptr;
    const SemanticOperator* final_norm = nullptr;
    const SemanticOperator* output = nullptr;
    for (uint32_t index = 0; index != model.operators.size(); ++index) {
        if (layer_operator[index]) continue;
        const SemanticOperator& op = model.operators[index];
        if (op.kind == OperatorKind::EmbeddingLookup &&
            role_tensor(model, op, TensorRole::TokenEmbedding)) {
            if (embedding) {
                detail = "canonical Metal primary program has multiple embeddings";
                return false;
            }
            embedding = &op;
        }
        if (op.kind == OperatorKind::RmsNorm &&
            role_tensor(model, op, TensorRole::FinalNormWeight)) {
            if (final_norm) {
                detail = "canonical Metal primary program has multiple final norms";
                return false;
            }
            final_norm = &op;
        }
        if (op.kind == OperatorKind::Linear &&
            role_tensor(model, op, TensorRole::OutputWeight)) {
            if (output) {
                detail = "canonical Metal primary program has multiple output projections";
                return false;
            }
            output = &op;
        }
    }
    if (!embedding || !final_norm || !output || embedding->inputs.size() != 1 ||
        embedding->outputs.size() != 1 || final_norm->inputs.size() != 1 ||
        final_norm->outputs.size() != 1 || output->inputs.size() != 1 ||
        output->outputs.size() != 1) {
        detail = "canonical Metal primary program spine is incomplete";
        return false;
    }
    const SemanticLayer& last_layer = model.layers.back();
    const uint32_t last_layer_end = last_layer.first_operator + last_layer.operator_count;
    if (embedding->id >= model.layers.front().first_operator ||
        final_norm->id < last_layer_end || output->id <= final_norm->id) {
        detail = "canonical Metal primary program spine is out of serialized order";
        return false;
    }

    const uint32_t token_input = embedding->inputs[0];
    const bool declared_input = model.input_values_count != 0 &&
        token_input >= model.input_values_first &&
        token_input - model.input_values_first < model.input_values_count;
    const SemanticValue* token_value = value_with_id(model, token_input);
    if (!declared_input || !token_value || token_value->logical_type != ScalarType::U32 ||
        embedding->outputs[0] != witness.layers.front().input_value_id ||
        final_norm->inputs[0] != witness.layers.back().output_value_id ||
        output->inputs != final_norm->outputs || model.output_values_count != 1 ||
        output->outputs[0] != model.output_values_first) {
        detail = "canonical Metal primary program edges differ from the executable spine";
        return false;
    }

    uint32_t hidden = 0;
    uint32_t vocabulary = 0;
    const auto* embedding_payload = std::get_if<EmbeddingLookupPayload>(&embedding->payload);
    const float embedding_scale = embedding_payload
        ? std::bit_cast<float>(embedding_payload->scale_f32_bits) : 0.0f;
    const SemanticTensor* embedding_tensor = role_tensor(
        model, *embedding, TensorRole::TokenEmbedding);
    if (!embedding_payload || !std::isfinite(embedding_scale) || embedding_scale <= 0.0f ||
        !semantic_flat_width(
            model, embedding->outputs[0], ScalarType::F32, hidden) ||
        !semantic_flat_width(model, output->outputs[0], ScalarType::F32, vocabulary) ||
        hidden == 0 || vocabulary == 0 || embedding_payload->width != hidden ||
        embedding_payload->vocabulary != vocabulary || vocabulary != model.vocabulary_size ||
        !embedding_tensor || embedding_tensor->dimensions != std::vector<Dimension>{
            {DimensionKind::Constant, hidden}, {DimensionKind::Constant, vocabulary}} ||
        !exact_norm_binding(model, *final_norm, hidden, false) ||
        !exact_linear_binding(model, *output, final_norm->outputs[0], output->outputs[0])) {
        detail = "canonical Metal primary program tensor bindings differ from the executable spine";
        return false;
    }

    witness.token_input_value_id = token_input;
    witness.embedding_operator_id = embedding->id;
    witness.embedding_output_value_id = embedding->outputs[0];
    witness.final_norm_operator_id = final_norm->id;
    witness.final_norm_output_value_id = final_norm->outputs[0];
    witness.output_operator_id = output->id;
    witness.output_value_id = output->outputs[0];
    return true;
}

bool match_canonical_dense_operator_edges(const SemanticModel& model,
                                          const SemanticLayer& layer,
                                          DenseGraphWitness& witness) {
    return match_dense_graph_impl(model, layer, witness);
}

bool match_canonical_moe_operator_edges(const SemanticModel& model,
                                        const SemanticLayer& layer,
                                        CanonicalMoeOperatorEdges& edges) {
    edges = {};
    const SemanticOperator* attention = nullptr;
    MoeDescriptorFacts facts;
    if (!moe_metal_layer(model, layer, attention, nullptr, &facts)) return false;
    edges.covered_operator_ids = facts.covered_operator_ids;
    edges.dense_attn_norm = facts.dense_attn_norm;
    edges.dense_query = facts.dense_query;
    edges.dense_query_gate = facts.dense_query_gate;
    edges.dense_key = facts.dense_key;
    edges.dense_value = facts.dense_value;
    edges.dense_output = facts.dense_output;
    edges.dense_query_norm = facts.dense_query_norm;
    edges.dense_key_norm = facts.dense_key_norm;
    edges.dense_ffn_norm = facts.dense_ffn_norm;
    edges.dense_gate = facts.dense_gate;
    edges.dense_up = facts.dense_up;
    edges.dense_down = facts.dense_down;
    edges.rope = facts.rope;
    edges.swiglu = facts.swiglu;
    edges.attention = facts.attention;
    edges.router = facts.router;
    edges.router_linear = facts.router_linear;
    edges.router_scale = facts.router_scale;
    edges.router_normalization_scale = facts.router_normalization_scale;
    edges.router_norm = facts.router_norm;
    edges.expert_norm = facts.expert_norm;
    edges.expert_up = facts.expert_up;
    edges.expert_split = facts.expert_split;
    edges.expert_activation = facts.expert_activation;
    edges.expert_down = facts.expert_down;
    edges.expert_reduce = facts.expert_reduce;
    edges.attention_residual = facts.attention_residual;
    edges.residual = facts.residual;
    edges.final_add = facts.final_add;
    edges.dense_post_norm = facts.dense_post_norm;
    edges.moe_post_norm = facts.moe_post_norm;
    edges.output_post_norm = facts.output_post_norm;
    edges.output_scale = facts.output_scale;
    edges.gate_up_tensor = facts.gate_up_tensor;
    edges.down_tensor = facts.down_tensor;
    edges.hidden = facts.hidden;
    edges.intermediate = facts.intermediate;
    edges.expert_count = facts.experts;
    edges.selected_count = facts.selected;
    edges.gate_up_format = facts.gate_up_format;
    edges.down_format = facts.down_format;
    edges.gate_up_layout = facts.gate_up_layout;
    edges.down_layout = facts.down_layout;
    edges.gate_up_quantization = facts.gate_up_quantization;
    edges.down_quantization = facts.down_quantization;
    edges.gate_up_plane_mask = facts.gate_up_plane_mask;
    edges.down_plane_mask = facts.down_plane_mask;
    edges.gate_up_block_elements = facts.gate_up_block_elements;
    edges.gate_up_block_bytes = facts.gate_up_block_bytes;
    edges.down_block_elements = facts.down_block_elements;
    edges.down_block_bytes = facts.down_block_bytes;
    edges.gate_up_expert_stride = facts.gate_up_expert_stride;
    edges.down_expert_stride = facts.down_expert_stride;
    edges.activation = facts.activation;
    edges.scale_source = facts.scale_source;
    edges.value_source = facts.value_source;
    edges.router_normalization_scale_bits = facts.router_normalization_scale_bits;
    return !edges.covered_operator_ids.empty() && edges.attention && edges.router && edges.router_linear &&
           edges.router_scale && edges.router_normalization_scale && edges.router_normalization_scale_bits != 0 &&
           edges.router_norm && edges.expert_norm && edges.expert_up &&
           edges.expert_split && edges.expert_activation && edges.expert_down &&
           edges.expert_reduce && edges.residual && edges.final_add;
}

bool requires_scalar_fp32(const RuntimeCapabilities& capabilities) {
    return capabilities.scalar_fp32;
}

bool pattern_matches(const KernelPattern& pattern, const KernelQuery& query) {
    const bool moe_pattern = !pattern.require_moe_descriptor ||
                             (query.metal_moe_token_pattern &&
                              pattern.moe_expert_count.contains(query.moe_expert_count) &&
                              pattern.moe_selected_count.contains(query.moe_selected_count) &&
                              matches(pattern.moe_gate_up_format, query.moe_gate_up_format) &&
                              matches(pattern.moe_down_format, query.moe_down_format) &&
                              matches(pattern.moe_gate_up_layout, query.moe_gate_up_layout) &&
                              matches(pattern.moe_down_layout, query.moe_down_layout) &&
                              matches(pattern.moe_gate_up_quantization, query.moe_gate_up_quantization) &&
                              matches(pattern.moe_down_quantization, query.moe_down_quantization) &&
                              matches(pattern.moe_gate_up_plane_mask, query.moe_gate_up_plane_mask) &&
                              matches(pattern.moe_down_plane_mask, query.moe_down_plane_mask) &&
                              pattern.moe_gate_up_block_elements.contains(query.moe_gate_up_block_elements) &&
                              pattern.moe_gate_up_block_bytes.contains(query.moe_gate_up_block_bytes) &&
                              pattern.moe_down_block_elements.contains(query.moe_down_block_elements) &&
                              pattern.moe_down_block_bytes.contains(query.moe_down_block_bytes) &&
                              pattern.moe_gate_up_input.contains(query.moe_gate_up_input) &&
                              pattern.moe_gate_up_output.contains(query.moe_gate_up_output) &&
                              pattern.moe_down_input.contains(query.moe_down_input) &&
                              pattern.moe_down_output.contains(query.moe_down_output) &&
                              matches(pattern.moe_gate_up_expert_stride, query.moe_gate_up_expert_stride) &&
                              matches(pattern.moe_down_expert_stride, query.moe_down_expert_stride) &&
                              matches(pattern.moe_activation, query.moe_activation) &&
                              matches(pattern.moe_scale_source, query.moe_scale_source) &&
                              matches(pattern.moe_value_source, query.moe_value_source));
    return pattern.semantic_version.contains(query.semantic_version) &&
           pattern.rank.contains(query.rank) &&
           pattern.alignment.contains(query.alignment) &&
           pattern.head_dimension.contains(query.head_dimension) &&
           pattern.batch_rows.contains(query.batch_rows) &&
           (!pattern.head_dimension_multiple || query.head_dimension % pattern.head_dimension_multiple == 0) &&
           pattern.tile_tokens.contains(query.tile_tokens) &&
           pattern.block_elements.contains(query.block_elements) &&
           pattern.block_bytes.contains(query.block_bytes) &&
           pattern.required_physical_codecs == query.physical_codecs &&
           (!pattern.allowed_metal_weight_formats ||
            (query.metal_weight_format_mask != 0 &&
             (query.metal_weight_format_mask & ~pattern.allowed_metal_weight_formats) == 0)) &&
           (!pattern.require_mixed_metal_weight_formats ||
            (query.metal_weight_format_mask & (query.metal_weight_format_mask - 1)) != 0) &&
           (!pattern.require_metal_dense_token_pattern || query.metal_dense_token_pattern) &&
           (!pattern.require_metal_moe_token_pattern || query.metal_moe_token_pattern) &&
           (!pattern.require_metal_dense_prefill_batch_pattern || query.metal_dense_prefill_batch_pattern) &&
           (!pattern.require_metal_recurrent_token_pattern || query.metal_recurrent_token_pattern) &&
           (!pattern.require_metal_affine_u2_256 || query.metal_affine_u2_256) &&
           (!pattern.require_metal_column_grouped_affine_u2_skip_256 ||
            query.metal_column_grouped_affine_u2_skip_256) &&
           moe_pattern &&
           matches(pattern.operation, query.operation) &&
           matches(pattern.phase, query.phase) &&
           matches(pattern.logical_type, query.logical_type) &&
           matches(pattern.storage_type, query.storage_type) &&
           matches(pattern.layout, query.layout) &&
           matches(pattern.quantization, query.quantization) &&
           matches(pattern.state_kind, query.state_kind) &&
           matches(pattern.state_format, query.state_format);
}

namespace {

bool valid_operator_id_sequence(const std::array<uint32_t, 5>& ids) {
    for (uint32_t id : ids) {
        if (id == UINT32_MAX) return false;
    }
    for (size_t left = 0; left != ids.size(); ++left) {
        for (size_t right = left + 1; right != ids.size(); ++right) {
            if (ids[left] == ids[right]) return false;
        }
    }
    return true;
}

bool valid_worklist_dimensions(const MoeWorklistTensorContract& tensor,
                               uint32_t experts, uint32_t input, uint32_t output,
                               const PhysicalCodecIdentity& codec) {
    if (tensor.input_width != input || tensor.output_width != output ||
        tensor.expert_count != experts || tensor.dimensions.size() != 3 ||
        tensor.dimensions[0] != Dimension{DimensionKind::Constant, experts} ||
        tensor.dimensions[1] != Dimension{DimensionKind::Constant, input} ||
        tensor.dimensions[2] != Dimension{DimensionKind::Constant, output} ||
        input == 0 || output == 0 ||
        tensor.strides[0] != 1 || tensor.strides[1] != output ||
        tensor.strides[2] != static_cast<uint64_t>(input) * output) return false;
    for (size_t index = 3; index != tensor.strides.size(); ++index) {
        if (tensor.strides[index] != 0) return false;
    }
    if (codec.layout.block_elements != 0) {
        if (input % codec.layout.block_elements != 0) return false;
        uint64_t expert_stride = 0;
        const uint64_t block_count = static_cast<uint64_t>(input) * output /
                                     codec.layout.block_elements;
        if (!checked_multiply(block_count, codec.layout.block_bytes, expert_stride) ||
            expert_stride == 0 || tensor.expert_stride != expert_stride) return false;
    } else if (tensor.expert_stride == 0) {
        return false;
    }
    return tensor.layout == codec.layout && tensor.quantization == codec.quantization &&
           tensor.planes == codec.planes;
}

bool valid_worklist_codec(const PhysicalCodecIdentity& codec) {
    return valid_physical_codec_identity(codec) && codec.layout.rank == 3 &&
           !codec.planes.empty();
}

} // namespace

bool valid_moe_worklist_descriptor(const MoeWorklistDescriptor& descriptor) {
    if (descriptor.version != 1 ||
        (descriptor.phase != ExecutionPhase::Prefill &&
         descriptor.phase != ExecutionPhase::Decode) ||
        !valid_operator_id_sequence(descriptor.semantic_operator_ids) ||
        descriptor.expert_count == 0 || descriptor.selected_count == 0 ||
        descriptor.selected_count > descriptor.expert_count || descriptor.hidden == 0 ||
        descriptor.intermediate == 0 || descriptor.intermediate > UINT32_MAX / 2 ||
        descriptor.gate_up.input_width != descriptor.hidden ||
        descriptor.gate_up.output_width != descriptor.intermediate * 2 ||
        descriptor.down.input_width != descriptor.intermediate ||
        descriptor.down.output_width != descriptor.hidden ||
        descriptor.physical_codecs.size() != 3) return false;

    switch (descriptor.activation) {
    case ActivationKind::Silu:
    case ActivationKind::GeluTanh:
        break;
    default:
        return false;
    }
    switch (descriptor.scale_source) {
    case ExpertScaleSource::None:
    case ExpertScaleSource::PerExpertTensor:
        break;
    default:
        return false;
    }
    switch (descriptor.value_source) {
    case ValueSource::SeparateProjection:
    case ValueSource::KeyPreRope:
    case ValueSource::KeyPostRope:
        break;
    case ValueSource::KeyStateAlias:
        return false;
    }
    for (size_t index = 0; index != descriptor.physical_codecs.size(); ++index) {
        if (!valid_worklist_codec(descriptor.physical_codecs[index])) return false;
    }
    if (!valid_worklist_dimensions(descriptor.gate_up, descriptor.expert_count,
                                   descriptor.hidden, descriptor.intermediate * 2,
                                   descriptor.physical_codecs[0]) ||
        !valid_worklist_dimensions(descriptor.down, descriptor.expert_count,
                                   descriptor.intermediate, descriptor.hidden,
                                   descriptor.physical_codecs[2])) return false;
    for (uint32_t first : {descriptor.router_scale_tensor_id,
                           descriptor.expert_norm_tensor_id,
                           descriptor.reduce_scale_tensor_id}) {
        if (first == UINT32_MAX) return false;
    }
    const std::array<uint32_t, 5> tensor_ids = {
        descriptor.router_scale_tensor_id, descriptor.expert_norm_tensor_id,
        descriptor.reduce_scale_tensor_id, descriptor.post_norm_tensor_id,
        descriptor.output_scale_tensor_id};
    for (size_t left = 0; left != tensor_ids.size(); ++left) {
        if (tensor_ids[left] == UINT32_MAX) continue;
        for (size_t right = left + 1; right != tensor_ids.size(); ++right) {
            if (tensor_ids[left] == tensor_ids[right]) return false;
        }
    }
    return true;
}

bool moe_worklist_matches(const MoeWorklistDescriptor& query,
                          const MoeWorklistDescriptor& candidate) {
    if (!valid_moe_worklist_descriptor(query) ||
        !valid_moe_worklist_descriptor(candidate)) return false;
    // Operator and tensor ids are package-local binding addresses.  They are
    // deliberately excluded here; the complete semantic/physical contract is
    // what identifies a reusable implementation descriptor.
    return query.version == candidate.version && query.phase == candidate.phase &&
           query.expert_count == candidate.expert_count &&
           query.selected_count == candidate.selected_count && query.hidden == candidate.hidden &&
           query.intermediate == candidate.intermediate &&
           query.activation == candidate.activation && query.scale_source == candidate.scale_source &&
           query.value_source == candidate.value_source && query.gate_up == candidate.gate_up &&
           query.down == candidate.down && query.physical_codecs == candidate.physical_codecs;
}

MoeWorklistSelection select_moe_worklist(
    const MoeWorklistDescriptor& query, const RuntimeCapabilities& capabilities,
    const std::vector<MoeWorklistDescriptor>& registry) {
    if (!valid_moe_worklist_descriptor(query))
        return plan_error(CompatibilityError::IR_REFERENCE_INVALID);
    if (!capabilities.metal_moe_worklist)
        return plan_error(CompatibilityError::CAPABILITY_MISSING);
    std::vector<const MoeWorklistDescriptor*> matches;
    for (const MoeWorklistDescriptor& descriptor : registry) {
        if (moe_worklist_matches(query, descriptor)) matches.push_back(&descriptor);
    }
    if (matches.empty()) return plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
    if (matches.size() != 1) return plan_error(CompatibilityError::KERNEL_AMBIGUOUS);
    // Keep the caller's package-local operator/tensor addresses.  Registry
    // rows are implementation capabilities and must not overwrite bindings
    // for the package being planned.
    return query;
}

KernelSelection select_kernel(const KernelQuery& query, const SessionRequest& request,
                              const RuntimeCapabilities& capabilities,
                              const std::vector<KernelDescriptor>& registry) {
    // The affine bit is emitted only after the complete three-plane physical
    // tuple has been checked.  Once present, a real queried pipeline
    // capability is mandatory; no other descriptor or CPU fallback may
    // silently consume the same semantic layer.
    if ((query.metal_weight_format_mask & MetalWeightFormatAffineUInt2_256) != 0 &&
        !capabilities.metal_affine_u2_256) {
        return plan_error(CompatibilityError::CAPABILITY_MISSING);
    }
    if ((query.metal_weight_format_mask & MetalWeightFormatColumnGroupedAffineUInt2Skip256) != 0 &&
        !capabilities.metal_column_grouped_affine_u2_skip_256) {
        return plan_error(CompatibilityError::CAPABILITY_MISSING);
    }
    std::vector<const KernelDescriptor*> matches;
    for (const KernelDescriptor& descriptor : registry) {
        if (!pattern_matches(descriptor.pattern, query) || descriptor.numerical_class != request.minimum_class ||
            (descriptor.capability_predicate && !descriptor.capability_predicate(capabilities))) continue;
        matches.push_back(&descriptor);
    }
    if (matches.empty()) return plan_error(CompatibilityError::KERNEL_UNAVAILABLE);

    uint64_t best_cost = objective_cost(*matches.front(), request.objective);
    for (const KernelDescriptor* descriptor : matches) best_cost = std::min(best_cost, objective_cost(*descriptor, request.objective));
    matches.erase(std::remove_if(matches.begin(), matches.end(), [&](const KernelDescriptor* descriptor) {
        return objective_cost(*descriptor, request.objective) != best_cost;
    }), matches.end());

    uint16_t best_priority = matches.front()->priority;
    for (const KernelDescriptor* descriptor : matches) best_priority = std::max(best_priority, descriptor->priority);
    matches.erase(std::remove_if(matches.begin(), matches.end(), [&](const KernelDescriptor* descriptor) {
        return descriptor->priority != best_priority;
    }), matches.end());

    const PlanEffectKey effects = matches.front()->effects;
    for (const KernelDescriptor* descriptor : matches) {
        if (!(descriptor->effects == effects)) return plan_error(CompatibilityError::KERNEL_AMBIGUOUS);
    }
    return **std::min_element(matches.begin(), matches.end(), [](const KernelDescriptor* left, const KernelDescriptor* right) {
        return left->id < right->id;
    });
}

std::vector<KernelDescriptor> builtin_cpu_registry() {
    constexpr std::array operations = {OperatorKind::EmbeddingLookup, OperatorKind::RmsNorm, OperatorKind::Linear,
                                       OperatorKind::Rope, OperatorKind::CausalAttention, OperatorKind::SwiGlu,
                                       OperatorKind::Add};
    std::vector<KernelDescriptor> registry;
    uint32_t id = 1;
    auto add = [&](OperatorKind operation, ExecutionPhase phase) {
        KernelDescriptor descriptor;
        descriptor.id = id++;
        descriptor.priority = 1;
        descriptor.implementation = scalar_implementation(operation);
        descriptor.pattern.operation = exact(operation);
        descriptor.pattern.semantic_version = {1, 1};
        descriptor.pattern.phase = exact(phase);
        descriptor.pattern.logical_type = exact(ScalarType::F32);
        descriptor.pattern.storage_type = any<ScalarType>();
        descriptor.pattern.layout = exact(PhysicalLayoutKind::ContiguousRowMajor);
        descriptor.pattern.quantization = exact(QuantizationKind::None);
        descriptor.pattern.state_kind = exact(StateKind::KeyCache);
        descriptor.pattern.state_format = exact(StateFormatKind::GlobalContiguous);
        // Linear preserves the leading token axes. V1's scalar linear loop
        // is shape-invariant for its only admitted rank-2 and rank-3 value
        // views; all other operations have one exact output rank.
        if (operation == OperatorKind::Linear) descriptor.pattern.rank = {2, 3};
        else {
            const uint32_t rank = operation == OperatorKind::Rope || operation == OperatorKind::CausalAttention ? 3 : 2;
            descriptor.pattern.rank = {rank, rank};
        }
        descriptor.pattern.alignment = {0, UINT32_MAX};
        descriptor.pattern.head_dimension = operation == OperatorKind::CausalAttention ? ClosedRange<uint32_t>{32, 512} : ClosedRange<uint32_t>{0, 0};
        descriptor.pattern.head_dimension_multiple = operation == OperatorKind::CausalAttention ? 16 : 0;
        descriptor.pattern.tile_tokens = {0, 0};
        descriptor.capability_predicate = &requires_scalar_fp32;
        descriptor.numerical_class = NumericalClass::ExactFp32;
        descriptor.transactional = true;
        descriptor.effects = {BackendWriter::Cpu, NumericalClass::ExactFp32, StateFormatKind::GlobalContiguous,
                              FallbackKind::ExactCpu, true, ScratchLifetime::Phase};
        descriptor.cost = {1, 1};
        registry.push_back(descriptor);
    };
    for (OperatorKind operation : operations) {
        add(operation, ExecutionPhase::Prefill);
        add(operation, ExecutionPhase::Decode);
        add(operation, ExecutionPhase::Output);
    }
    add(OperatorKind::CausalAttention, ExecutionPhase::StateUpdate);
    add(OperatorKind::CausalAttention, ExecutionPhase::Rollback);
    return registry;
}

bool requires_canonical_metal(const RuntimeCapabilities& capabilities) {
    return capabilities.metal_device && capabilities.metal_library && capabilities.metal_pipeline &&
           capabilities.global_fp32_kv && capabilities.transactional_state;
}

bool requires_canonical_moe(const RuntimeCapabilities& capabilities) {
    return canonical_moe_capabilities(capabilities);
}

std::vector<KernelDescriptor> builtin_canonical_metal_registry() {
    std::vector<KernelDescriptor> registry;
    for (ExecutionPhase phase : {ExecutionPhase::Prefill, ExecutionPhase::Decode}) {
        KernelDescriptor descriptor;
        descriptor.id = phase == ExecutionPhase::Prefill ? 1001 : 1002;
        descriptor.priority = 100;
        descriptor.implementation = KernelImplementation::MetalDenseToken;
        descriptor.pattern.operation = exact(OperatorKind::CausalAttention);
        descriptor.pattern.semantic_version = {1, 1};
        descriptor.pattern.phase = exact(phase);
        descriptor.pattern.logical_type = exact(ScalarType::F32);
        descriptor.pattern.storage_type = exact(ScalarType::F16);
        descriptor.pattern.layout = exact(PhysicalLayoutKind::ContiguousRowMajor);
        descriptor.pattern.quantization = exact(QuantizationKind::None);
        descriptor.pattern.state_kind = exact(StateKind::KeyCache);
        descriptor.pattern.state_format = exact(StateFormatKind::GlobalContiguous);
        descriptor.pattern.rank = {2, 3};
        descriptor.pattern.alignment = {64, 64};
        descriptor.pattern.head_dimension = {32, 512};
        descriptor.pattern.batch_rows = {1, 1};
        descriptor.pattern.head_dimension_multiple = 16;
        descriptor.pattern.tile_tokens = {0, 0};
        descriptor.pattern.block_elements = {0, 0};
        descriptor.pattern.block_bytes = {0, 0};
        descriptor.pattern.allowed_metal_weight_formats = MetalWeightFormatF16;
        descriptor.pattern.require_metal_dense_token_pattern = true;
        descriptor.capability_predicate = &requires_canonical_metal;
        descriptor.numerical_class = NumericalClass::ExactFp32;
        descriptor.transactional = true;
        descriptor.effects = {BackendWriter::Metal, NumericalClass::ExactFp32, StateFormatKind::GlobalContiguous,
                              FallbackKind::None, true, ScratchLifetime::Session};
        descriptor.cost = {1, 1};
        registry.push_back(descriptor);

        for (size_t index = 0; index != kMetalBlockedFormats.size(); ++index) {
            const MetalBlockedFormat& blocked = kMetalBlockedFormats[index];
            KernelDescriptor quantized = descriptor;
            quantized.id = (phase == ExecutionPhase::Prefill ? 1011 : 1012) + static_cast<uint32_t>(index) * 10;
            quantized.pattern.storage_type = exact(ScalarType::U8);
            quantized.pattern.layout = exact(PhysicalLayoutKind::GgufBlocked);
            quantized.pattern.quantization = exact(blocked.quantization);
            quantized.pattern.alignment = {32, 32};
            quantized.pattern.block_elements = {blocked.elements, blocked.elements};
            quantized.pattern.block_bytes = {blocked.bytes, blocked.bytes};
            quantized.pattern.allowed_metal_weight_formats = blocked.format;
            registry.push_back(std::move(quantized));
        }
        KernelDescriptor mixed = descriptor;
        mixed.id = phase == ExecutionPhase::Prefill ? 1091 : 1092;
        mixed.priority = 99;
        mixed.pattern.storage_type = exact(ScalarType::U8);
        mixed.pattern.layout = exact(PhysicalLayoutKind::GgufBlocked);
        mixed.pattern.quantization = any<QuantizationKind>();
        mixed.pattern.alignment = {32, 32};
        mixed.pattern.block_elements = {0, 256};
        mixed.pattern.block_bytes = {0, 210};
        mixed.pattern.allowed_metal_weight_formats = MetalWeightFormatQ4_0 | MetalWeightFormatQ4K | MetalWeightFormatQ5_0 |
                                                      MetalWeightFormatQ6K | MetalWeightFormatQ8_0 |
                                                      MetalWeightFormatQ2K | MetalWeightFormatIQ2XXS;
        mixed.pattern.require_mixed_metal_weight_formats = true;
        KernelDescriptor mixed_affine = mixed;
        mixed_affine.id = phase == ExecutionPhase::Prefill ? 1093 : 1094;
        mixed_affine.pattern.allowed_metal_weight_formats |=
            MetalWeightFormatF16 | MetalWeightFormatAffineUInt2_256;
        mixed_affine.pattern.storage_type = any<ScalarType>();
        mixed_affine.pattern.layout = any<PhysicalLayoutKind>();
        mixed_affine.pattern.quantization = any<QuantizationKind>();
        mixed_affine.pattern.alignment = {32, 128};
        mixed_affine.pattern.require_metal_affine_u2_256 = true;
        KernelDescriptor mixed_column_grouped = mixed;
        mixed_column_grouped.id = phase == ExecutionPhase::Prefill ? 1099 : 1100;
        mixed_column_grouped.pattern.allowed_metal_weight_formats |=
            MetalWeightFormatF16 | MetalWeightFormatColumnGroupedAffineUInt2Skip256;
        mixed_column_grouped.pattern.storage_type = any<ScalarType>();
        mixed_column_grouped.pattern.layout = any<PhysicalLayoutKind>();
        mixed_column_grouped.pattern.quantization = any<QuantizationKind>();
        mixed_column_grouped.pattern.alignment = {32, 128};
        mixed_column_grouped.pattern.require_metal_column_grouped_affine_u2_skip_256 = true;
        registry.push_back(std::move(mixed));
        registry.push_back(std::move(mixed_affine));
        registry.push_back(std::move(mixed_column_grouped));

        KernelDescriptor affine = descriptor;
        affine.id = phase == ExecutionPhase::Prefill ? 1095 : 1096;
        affine.pattern.storage_type = exact(ScalarType::U32);
        affine.pattern.layout = exact(PhysicalLayoutKind::GroupedAffine);
        affine.pattern.quantization = exact(QuantizationKind::BlockedAffine);
        affine.pattern.alignment = {128, 128};
        affine.pattern.block_elements = {256, 256};
        affine.pattern.block_bytes = {64, 64};
        affine.pattern.allowed_metal_weight_formats = MetalWeightFormatAffineUInt2_256;
        affine.pattern.require_mixed_metal_weight_formats = false;
        affine.pattern.require_metal_affine_u2_256 = true;
        registry.push_back(std::move(affine));

        KernelDescriptor column_grouped = descriptor;
        column_grouped.id = phase == ExecutionPhase::Prefill ? 1097 : 1098;
        column_grouped.pattern.storage_type = exact(ScalarType::U8);
        column_grouped.pattern.layout = exact(PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip);
        column_grouped.pattern.quantization = exact(QuantizationKind::BlockedAffine);
        column_grouped.pattern.alignment = {128, 128};
        column_grouped.pattern.block_elements = {256, 256};
        column_grouped.pattern.block_bytes = {64, 64};
        column_grouped.pattern.allowed_metal_weight_formats =
            MetalWeightFormatColumnGroupedAffineUInt2Skip256;
        column_grouped.pattern.require_mixed_metal_weight_formats = false;
        column_grouped.pattern.require_metal_column_grouped_affine_u2_skip_256 = true;
        registry.push_back(std::move(column_grouped));
    }
    const size_t dense_descriptor_count = registry.size();
    for (size_t index = 0; index != dense_descriptor_count; ++index) {
        KernelDescriptor descriptor = registry[index];
        descriptor.id += 2000;
        descriptor.pattern.semantic_version = {4, 4};
        registry.push_back(std::move(descriptor));
    }
    for (size_t index = 0; index != dense_descriptor_count; ++index) {
        KernelDescriptor descriptor = registry[index];
        descriptor.id += 4000;
        descriptor.pattern.semantic_version = {6, 6};
        registry.push_back(std::move(descriptor));
    }
    for (size_t index = 0; index != dense_descriptor_count; ++index) {
        KernelDescriptor descriptor = registry[index];
        descriptor.id += 12000;
        descriptor.pattern.semantic_version = {8, 8};
        registry.push_back(std::move(descriptor));
    }
    for (ExecutionPhase phase : {ExecutionPhase::Prefill, ExecutionPhase::Decode}) {
        const auto add_moe = [&](uint32_t id, uint32_t down_format, uint32_t down_elements,
                                 uint32_t down_bytes, CapabilityPredicate capability) {
            KernelDescriptor moe;
            moe.id = id;
            moe.priority = 101;
            moe.implementation = KernelImplementation::MetalDenseToken;
            moe.pattern.operation = exact(OperatorKind::CausalAttention);
            // MoE admission is a semantic/physical contract.  The schema and
            // operator numbering are intentionally absent from this matcher.
            moe.pattern.semantic_version = {1, UINT16_MAX};
            moe.pattern.phase = exact(phase);
            moe.pattern.logical_type = exact(ScalarType::F32);
            moe.pattern.storage_type = exact(ScalarType::U8);
            moe.pattern.layout = exact(PhysicalLayoutKind::GgufBlocked);
            moe.pattern.quantization = exact(QuantizationKind::BlockedAffine);
            moe.pattern.state_kind = exact(StateKind::KeyCache);
            moe.pattern.state_format = exact(StateFormatKind::GlobalContiguous);
            moe.pattern.rank = {2, 3};
            moe.pattern.alignment = {32, 32};
            moe.pattern.head_dimension = {32, 512};
            moe.pattern.batch_rows = {1, 1};
            moe.pattern.head_dimension_multiple = 16;
            moe.pattern.tile_tokens = {0, 0};
            moe.pattern.block_elements = {0, UINT32_MAX};
            moe.pattern.block_bytes = {0, UINT32_MAX};
            moe.pattern.allowed_metal_weight_formats = MetalWeightFormatF16 | MetalWeightFormatQ4K |
                                                        MetalWeightFormatQ5_0 | MetalWeightFormatQ6K |
                                                        MetalWeightFormatQ8_0 | MetalWeightFormatQ2K |
                                                        MetalWeightFormatIQ2XXS;
            moe.pattern.require_mixed_metal_weight_formats = true;
            moe.pattern.require_metal_moe_token_pattern = true;
            moe.pattern.require_moe_descriptor = true;
            moe.pattern.moe_expert_count = {1, 512};
            moe.pattern.moe_selected_count = {1, 16};
            moe.pattern.moe_gate_up_format = exact(static_cast<uint32_t>(MetalWeightFormatQ4K));
            moe.pattern.moe_down_format = exact(down_format);
            moe.pattern.moe_gate_up_layout = exact(PhysicalLayoutKind::GgufBlocked);
            moe.pattern.moe_down_layout = exact(PhysicalLayoutKind::GgufBlocked);
            moe.pattern.moe_gate_up_quantization = exact(QuantizationKind::BlockedAffine);
            moe.pattern.moe_down_quantization = exact(QuantizationKind::BlockedAffine);
            moe.pattern.moe_gate_up_plane_mask = exact(1u);
            moe.pattern.moe_down_plane_mask = exact(1u);
            moe.pattern.moe_gate_up_block_elements = {256, 256};
            moe.pattern.moe_gate_up_block_bytes = {144, 144};
            moe.pattern.moe_down_block_elements = {down_elements, down_elements};
            moe.pattern.moe_down_block_bytes = {down_bytes, down_bytes};
            moe.pattern.moe_gate_up_input = {256, UINT32_MAX};
            moe.pattern.moe_gate_up_output = {256, UINT32_MAX};
            moe.pattern.moe_down_input = {256, UINT32_MAX};
            moe.pattern.moe_down_output = {256, UINT32_MAX};
            moe.pattern.moe_gate_up_expert_stride = any<uint64_t>();
            moe.pattern.moe_down_expert_stride = any<uint64_t>();
            moe.pattern.moe_activation = exact(ActivationKind::GeluTanh);
            moe.pattern.moe_scale_source = exact(ExpertScaleSource::PerExpertTensor);
            moe.pattern.moe_value_source = exact(ValueSource::SeparateProjection);
            moe.capability_predicate = capability;
            moe.numerical_class = NumericalClass::ExactFp32;
            moe.transactional = true;
            moe.effects = {BackendWriter::Metal, NumericalClass::ExactFp32, StateFormatKind::GlobalContiguous,
                           FallbackKind::None, true, ScratchLifetime::Session};
            moe.cost = {1, 1};
            registry.push_back(std::move(moe));
        };
        add_moe(phase == ExecutionPhase::Prefill ? 9001 : 9002,  MetalWeightFormatQ5_0, 32, 22,
                &canonical_moe_q5_capabilities);
        add_moe(phase == ExecutionPhase::Prefill ? 9011 : 9012, MetalWeightFormatQ8_0, 32, 34,
                &canonical_moe_q8_capabilities);
    }
    KernelDescriptor prefill_batch;
    prefill_batch.id = 7001;
    prefill_batch.priority = 101;
    prefill_batch.implementation = KernelImplementation::MetalDensePrefillBatch;
    prefill_batch.pattern.operation = exact(OperatorKind::CausalAttention);
    prefill_batch.pattern.semantic_version = {1, 1};
    prefill_batch.pattern.phase = exact(ExecutionPhase::Prefill);
    prefill_batch.pattern.logical_type = exact(ScalarType::F32);
    prefill_batch.pattern.storage_type = exact(ScalarType::F16);
    prefill_batch.pattern.layout = exact(PhysicalLayoutKind::ContiguousRowMajor);
    prefill_batch.pattern.quantization = exact(QuantizationKind::None);
    prefill_batch.pattern.state_kind = exact(StateKind::KeyCache);
    prefill_batch.pattern.state_format = exact(StateFormatKind::GlobalContiguous);
    prefill_batch.pattern.rank = {2, 3};
    prefill_batch.pattern.alignment = {64, 64};
    prefill_batch.pattern.head_dimension = {32, 512};
    prefill_batch.pattern.batch_rows = {2, 2};
    prefill_batch.pattern.head_dimension_multiple = 16;
    prefill_batch.pattern.tile_tokens = {0, 0};
    prefill_batch.pattern.block_elements = {0, 0};
    prefill_batch.pattern.block_bytes = {0, 0};
    prefill_batch.pattern.allowed_metal_weight_formats = MetalWeightFormatF16;
    prefill_batch.pattern.require_metal_dense_token_pattern = true;
    prefill_batch.pattern.require_metal_dense_prefill_batch_pattern = true;
    prefill_batch.capability_predicate = &requires_canonical_metal;
    prefill_batch.numerical_class = NumericalClass::ExactFp32;
    prefill_batch.transactional = true;
    prefill_batch.effects = {BackendWriter::Metal, NumericalClass::ExactFp32, StateFormatKind::GlobalContiguous,
                             FallbackKind::None, true, ScratchLifetime::Session};
    prefill_batch.cost = {1, 1};
    registry.push_back(std::move(prefill_batch));
    KernelDescriptor prefill_batch_v8 = registry.back();
    prefill_batch_v8.id = 19001;
    prefill_batch_v8.pattern.semantic_version = {8, 8};
    registry.push_back(std::move(prefill_batch_v8));
    const size_t recurrent_descriptor_begin = registry.size();
    for (ExecutionPhase phase : {ExecutionPhase::Prefill, ExecutionPhase::Decode}) {
        KernelDescriptor descriptor;
        descriptor.id = phase == ExecutionPhase::Prefill ? 1101 : 1102;
        descriptor.priority = 101;
        descriptor.implementation = KernelImplementation::MetalRecurrentToken;
        descriptor.pattern.operation = exact(OperatorKind::GatedDeltaNet);
        descriptor.pattern.semantic_version = {3, 3};
        descriptor.pattern.phase = exact(phase);
        descriptor.pattern.logical_type = exact(ScalarType::F32);
        descriptor.pattern.storage_type = exact(ScalarType::U8);
        descriptor.pattern.layout = exact(PhysicalLayoutKind::GgufBlocked);
        descriptor.pattern.quantization = any<QuantizationKind>();
        descriptor.pattern.state_kind = exact(StateKind::RecurrentDeltaMatrix);
        descriptor.pattern.state_format = exact(StateFormatKind::RecurrentContiguous);
        descriptor.pattern.rank = {2, 2};
        descriptor.pattern.alignment = {32, 32};
        descriptor.pattern.head_dimension = {16, 512};
        descriptor.pattern.head_dimension_multiple = 16;
        descriptor.pattern.tile_tokens = {0, 0};
        descriptor.pattern.block_elements = {0, 256};
        descriptor.pattern.block_bytes = {0, 210};
        descriptor.pattern.allowed_metal_weight_formats = MetalWeightFormatQ4K | MetalWeightFormatQ6K |
                                                          MetalWeightFormatQ2K | MetalWeightFormatIQ2XXS;
        descriptor.pattern.require_mixed_metal_weight_formats = true;
        descriptor.pattern.require_metal_recurrent_token_pattern = true;
        descriptor.capability_predicate = &requires_canonical_metal;
        descriptor.numerical_class = NumericalClass::ExactFp32;
        descriptor.transactional = true;
        descriptor.effects = {BackendWriter::Metal, NumericalClass::ExactFp32, StateFormatKind::RecurrentContiguous,
                              FallbackKind::None, true, ScratchLifetime::Session};
        descriptor.cost = {1, 1};
        registry.push_back(descriptor);

        KernelDescriptor mixed_affine = descriptor;
        mixed_affine.id = phase == ExecutionPhase::Prefill ? 1103 : 1104;
        mixed_affine.pattern.allowed_metal_weight_formats |=
            MetalWeightFormatF16 | MetalWeightFormatAffineUInt2_256;
        mixed_affine.pattern.storage_type = any<ScalarType>();
        mixed_affine.pattern.layout = any<PhysicalLayoutKind>();
        mixed_affine.pattern.quantization = any<QuantizationKind>();
        mixed_affine.pattern.alignment = {32, 128};
        mixed_affine.pattern.require_metal_affine_u2_256 = true;
        registry.push_back(std::move(mixed_affine));

        KernelDescriptor mixed_column_grouped = descriptor;
        mixed_column_grouped.id = phase == ExecutionPhase::Prefill ? 1105 : 1106;
        mixed_column_grouped.pattern.allowed_metal_weight_formats |=
            MetalWeightFormatF16 | MetalWeightFormatColumnGroupedAffineUInt2Skip256;
        mixed_column_grouped.pattern.storage_type = any<ScalarType>();
        mixed_column_grouped.pattern.layout = any<PhysicalLayoutKind>();
        mixed_column_grouped.pattern.quantization = any<QuantizationKind>();
        mixed_column_grouped.pattern.alignment = {32, 128};
        mixed_column_grouped.pattern.require_metal_column_grouped_affine_u2_skip_256 = true;
        registry.push_back(std::move(mixed_column_grouped));

        KernelDescriptor q4k = descriptor;
        q4k.id = phase == ExecutionPhase::Prefill ? 1111 : 1112;
        q4k.pattern.block_elements = {256, 256};
        q4k.pattern.block_bytes = {144, 144};
        q4k.pattern.quantization = exact(QuantizationKind::BlockedAffine);
        q4k.pattern.allowed_metal_weight_formats = MetalWeightFormatQ4K;
        q4k.pattern.require_mixed_metal_weight_formats = false;
        registry.push_back(std::move(q4k));

        KernelDescriptor q6k = descriptor;
        q6k.id = phase == ExecutionPhase::Prefill ? 1121 : 1122;
        q6k.pattern.block_elements = {256, 256};
        q6k.pattern.block_bytes = {210, 210};
        q6k.pattern.quantization = exact(QuantizationKind::BlockedAffine);
        q6k.pattern.allowed_metal_weight_formats = MetalWeightFormatQ6K;
        q6k.pattern.require_mixed_metal_weight_formats = false;
        registry.push_back(std::move(q6k));

        KernelDescriptor q2k = descriptor;
        q2k.id = phase == ExecutionPhase::Prefill ? 1131 : 1132;
        q2k.pattern.block_elements = {256, 256};
        q2k.pattern.block_bytes = {84, 84};
        q2k.pattern.quantization = exact(QuantizationKind::BlockedAffine);
        q2k.pattern.allowed_metal_weight_formats = MetalWeightFormatQ2K;
        q2k.pattern.require_mixed_metal_weight_formats = false;
        registry.push_back(std::move(q2k));

        KernelDescriptor iq2_xxs = descriptor;
        iq2_xxs.id = phase == ExecutionPhase::Prefill ? 1141 : 1142;
        iq2_xxs.pattern.quantization = exact(QuantizationKind::Codebook);
        iq2_xxs.pattern.block_elements = {256, 256};
        iq2_xxs.pattern.block_bytes = {66, 66};
        iq2_xxs.pattern.allowed_metal_weight_formats = MetalWeightFormatIQ2XXS;
        iq2_xxs.pattern.require_mixed_metal_weight_formats = false;
        registry.push_back(std::move(iq2_xxs));

        KernelDescriptor affine = descriptor;
        affine.id = phase == ExecutionPhase::Prefill ? 1151 : 1152;
        affine.pattern.alignment = {128, 128};
        affine.pattern.storage_type = exact(ScalarType::U32);
        affine.pattern.layout = exact(PhysicalLayoutKind::GroupedAffine);
        affine.pattern.block_elements = {256, 256};
        affine.pattern.block_bytes = {64, 64};
        affine.pattern.quantization = exact(QuantizationKind::BlockedAffine);
        affine.pattern.allowed_metal_weight_formats = MetalWeightFormatAffineUInt2_256;
        affine.pattern.require_mixed_metal_weight_formats = false;
        affine.pattern.require_metal_affine_u2_256 = true;
        registry.push_back(std::move(affine));

        KernelDescriptor column_grouped = descriptor;
        column_grouped.id = phase == ExecutionPhase::Prefill ? 1153 : 1154;
        column_grouped.pattern.alignment = {128, 128};
        column_grouped.pattern.storage_type = exact(ScalarType::U8);
        column_grouped.pattern.layout = exact(PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip);
        column_grouped.pattern.block_elements = {256, 256};
        column_grouped.pattern.block_bytes = {64, 64};
        column_grouped.pattern.quantization = exact(QuantizationKind::BlockedAffine);
        column_grouped.pattern.allowed_metal_weight_formats =
            MetalWeightFormatColumnGroupedAffineUInt2Skip256;
        column_grouped.pattern.require_mixed_metal_weight_formats = false;
        column_grouped.pattern.require_metal_column_grouped_affine_u2_skip_256 = true;
        registry.push_back(std::move(column_grouped));
    }
    const size_t recurrent_descriptor_end = registry.size();
    for (size_t index = recurrent_descriptor_begin; index != recurrent_descriptor_end; ++index) {
        KernelDescriptor descriptor = registry[index];
        descriptor.id += 2000;
        descriptor.pattern.semantic_version = {4, 4};
        registry.push_back(std::move(descriptor));

    }
    for (size_t index = recurrent_descriptor_begin; index != recurrent_descriptor_end; ++index) {
        KernelDescriptor descriptor = registry[index];
        descriptor.id += 4000;
        descriptor.pattern.semantic_version = {6, 6};
        registry.push_back(std::move(descriptor));
    }
    for (size_t index = recurrent_descriptor_begin; index != recurrent_descriptor_end; ++index) {
        KernelDescriptor descriptor = registry[index];
        descriptor.id += 12000;
        descriptor.pattern.semantic_version = {8, 8};
        registry.push_back(std::move(descriptor));
    }
    return registry;
}

PlanResult plan_canonical_metal(const SemanticModel& model, const SessionRequest& request,
                                const RuntimeCapabilities& capabilities,
                                const std::vector<KernelDescriptor>& registry) {
    const bool schema1 = model.schema_major == 1 && model.schema_minor == 0 &&
                         model.opset_major == 1 && model.opset_minor == 0;
    const bool schema3 = model.schema_major == 3 && model.schema_minor == 0 &&
                         model.opset_major == 3 && model.opset_minor == 0;
    const bool schema4 = model.schema_major == 4 && model.schema_minor == 0 &&
                         model.opset_major == 4 && model.opset_minor == 0;
    const bool schema6 = model.schema_major == 6 && model.schema_minor == 0 &&
                         model.opset_major == 6 && model.opset_minor == 0;
    const bool schema7 = model.schema_major == 7 && model.schema_minor == 0 &&
                         model.opset_major == 7 && model.opset_minor == 0;
    const bool schema8 = model.schema_major == 8 && model.schema_minor == 0 &&
                         model.opset_major == 8 && model.opset_minor == 0;
    if (!schema1 && !schema3 && !schema4 && !schema6 && !schema7 && !schema8)
        return plan_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
    if (request.enable_streaming) return plan_error(CompatibilityError::STREAMING_UNSUPPORTED);
    if (request.enable_speculation) return plan_error(CompatibilityError::FALLBACK_FORBIDDEN);
    if ((!request.enable_prefill && !request.enable_decode) || !request.max_context ||
        request.max_context > model.maximum_context || !request.max_batch) {
        return plan_error(!request.max_context || request.max_context > model.maximum_context
                              ? CompatibilityError::PLAN_CONTEXT_EXCEEDED
                              : CompatibilityError::RUNTIME_INPUT_INVALID);
    }
    CanonicalProgramWitness program;
    std::string program_detail;
    if (!canonical_program_witness(model, program, program_detail)) {
        CompatibilityReport report = plan_error(CompatibilityError::IR_REFERENCE_INVALID);
        report.detail = std::move(program_detail);
        return report;
    }
    std::string attention_window_detail;
    if (!canonical_attention_window_is_global(model, attention_window_detail)) {
        CompatibilityReport report = plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
        report.detail = std::move(attention_window_detail);
        return report;
    }
    bool state_memory_valid = false;
    const uint64_t reserved_bytes = state_bytes(model, request, state_memory_valid);
    uint64_t total_peak = 0;
    if (!state_memory_valid || !peak_bytes(model, request, reserved_bytes, total_peak) || total_peak > request.memory_limit) {
        return plan_error(CompatibilityError::PLAN_MEMORY_EXCEEDED);
    }
    uint32_t unexpected_operator = UINT32_MAX;
    std::string coverage_detail;
    if (!canonical_metal_operator_coverage(model, unexpected_operator,
                                           coverage_detail)) {
        CompatibilityReport report = plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
        report.operator_id = unexpected_operator;
        report.detail = std::move(coverage_detail);
        return report;
    }
    std::string geometry_detail;
    if (!dense_attention_geometry_is_valid(model, geometry_detail)) {
        CompatibilityReport report =
            plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
        report.detail = std::move(geometry_detail);
        return report;
    }

    ExecutionPlan plan;
    plan.layer_chain = program.layers;
    plan.program = std::move(program);
    plan.reserved_bytes = reserved_bytes;
    plan.peak_bytes = total_peak;
    auto append_phase = [&](ExecutionPhase phase) -> std::optional<CompatibilityReport> {
        for (const SemanticLayer& layer : model.layers) {
            if ((layer.flags & kSemanticLayerFlagSpeculative) != 0) continue;
            const SemanticOperator* dense_anchor = nullptr;
            const SemanticOperator* recurrent_anchor = nullptr;
            // Layer admission is driven by the graph and physical contracts.
            // Schema versions may describe the IR, but they are not a model
            // identity selector and must not decide whether a routed graph is
            // MoE-capable.
            const bool moe = moe_metal_layer(model, layer, dense_anchor);
            const bool dense = !moe && dense_metal_layer(model, layer, dense_anchor);
            const bool recurrent = recurrent_metal_layer(model, layer, recurrent_anchor);
            if ((dense || moe) == recurrent || (!dense_anchor && !recurrent_anchor)) {
                CompatibilityReport report = plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
                report.layer = layer.layer_index;
                report.detail = "canonical Metal layer does not match an admitted full-token pattern";
                return report;
            }
            const SemanticOperator& anchor = (dense || moe) ? *dense_anchor : *recurrent_anchor;
            if (!moe && anchor.semantic_version != model.opset_major) {
                CompatibilityReport report = plan_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
                report.layer = layer.layer_index;
                report.operator_id = anchor.id;
                return report;
            }
            const KernelQuery query = moe
                ? moe_metal_query(model, layer, anchor, phase)
                : dense
                ? (phase == ExecutionPhase::Prefill && request.max_batch != 1
                       ? dense_prefill_batch_query(model, layer, anchor, request.max_batch)
                       : dense_metal_query(model, layer, anchor, phase))
                : recurrent_metal_query(model, layer, anchor, phase);
            KernelSelection selected = select_kernel(query, request, capabilities, registry);
            if (const auto* report = std::get_if<CompatibilityReport>(&selected)) {
                CompatibilityReport annotated = *report;
                annotated.layer = layer.layer_index;
                annotated.operator_id = anchor.id;
                if (annotated.detail.empty()) {
                    annotated.detail = std::string(moe ? "canonical Metal MoE" :
                                                    dense ? "canonical Metal dense" : "canonical Metal recurrent") +
                                       " descriptor is unavailable: semantic=" + std::to_string(query.semantic_version) +
                                       " formats=" + std::to_string(query.metal_weight_format_mask) +
                                       " block=" + std::to_string(query.block_elements) + "x" +
                                       std::to_string(query.block_bytes) + " rank=" + std::to_string(query.rank) +
                                       " alignment=" + std::to_string(query.alignment) +
                                       " head=" + std::to_string(query.head_dimension) +
                                       " state=" + std::to_string(static_cast<uint16_t>(query.state_kind)) + "/" +
                                       std::to_string(static_cast<uint16_t>(query.state_format));
                }
                return annotated;
            }
            PlanEntry entry;
            entry.phase = phase;
            entry.operator_id = anchor.id;
            entry.descriptor = std::get<KernelDescriptor>(selected);
            entry.kernel_id = entry.descriptor.id;
            for (uint32_t index = 0; index != layer.operator_count; ++index) {
                const SemanticOperator& op = model.operators[layer.first_operator + index];
                for (uint32_t tensor_id : op.tensors) {
                    if (tensor_id >= model.tensors.size()) return plan_error(CompatibilityError::IR_REFERENCE_INVALID);
                    const SemanticTensor& tensor = model.tensors[tensor_id];
                    for (const TensorPlane& plane : tensor.planes) {
                        if (plane.offset > UINT64_MAX - plane.length) return plan_error(CompatibilityError::PACKAGE_BOUNDS_INVALID);
                        entry.tensors.push_back({tensor.id, plane.artifact_id, plane.offset, plane.length, plane.alignment,
                                                 tensor.logical_type, tensor.layout.kind, tensor.quantization.kind});
                    }
                }
                for (uint32_t state_id : op.states) {
                    const SemanticState* state = state_with_id(model, state_id);
                    if (!state || state->formats.empty()) return plan_error(CompatibilityError::IR_STATE_INVALID);
                    const auto duplicate = std::find_if(entry.states.begin(), entry.states.end(), [&](const CheckedStateBinding& binding) {
                        return binding.state_id == state->id;
                    });
                    if (duplicate == entry.states.end()) {
                        entry.states.push_back({state->id, state->kind, state->formats[0].kind,
                                                state->formats[0].alignment, state->formats[0].tile_tokens});
                    }
                }
            }
            plan.entries.push_back(std::move(entry));
        }
        return std::nullopt;
    };
    if (request.enable_prefill) {
        if (auto report = append_phase(ExecutionPhase::Prefill)) return *report;
    }
    if (request.enable_decode) {
        if (auto report = append_phase(ExecutionPhase::Decode)) return *report;
    }
    return plan;
}

PlanResult plan_canonical_metal_dense(const SemanticModel& model, const SessionRequest& request,
                                      const RuntimeCapabilities& capabilities,
                                      const std::vector<KernelDescriptor>& registry) {
    if (request.enable_streaming) return plan_error(CompatibilityError::STREAMING_UNSUPPORTED);
    if (request.enable_speculation) return plan_error(CompatibilityError::FALLBACK_FORBIDDEN);
    if ((!request.enable_prefill && !request.enable_decode) || !request.max_context ||
        request.max_context > model.maximum_context || !request.max_batch) {
        return plan_error(!request.max_context || request.max_context > model.maximum_context
                              ? CompatibilityError::PLAN_CONTEXT_EXCEEDED
                              : CompatibilityError::RUNTIME_INPUT_INVALID);
    }
    std::vector<CanonicalLayerBoundary> layer_chain;
    std::string layer_chain_detail;
    if (!canonical_layer_chain_witness(model, layer_chain, layer_chain_detail)) {
        CompatibilityReport report = plan_error(CompatibilityError::IR_REFERENCE_INVALID);
        report.detail = std::move(layer_chain_detail);
        return report;
    }
    std::string attention_window_detail;
    if (!canonical_attention_window_is_global(model, attention_window_detail)) {
        CompatibilityReport report = plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
        report.detail = std::move(attention_window_detail);
        return report;
    }
    bool state_memory_valid = false;
    const uint64_t reserved_bytes = state_bytes(model, request, state_memory_valid);
    uint64_t total_peak = 0;
    if (!state_memory_valid || !peak_bytes(model, request, reserved_bytes, total_peak) || total_peak > request.memory_limit) {
        return plan_error(CompatibilityError::PLAN_MEMORY_EXCEEDED);
    }
    uint32_t unexpected_operator = UINT32_MAX;
    std::string coverage_detail;
    if (!canonical_metal_operator_coverage(model, unexpected_operator,
                                           coverage_detail)) {
        CompatibilityReport report = plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
        report.operator_id = unexpected_operator;
        report.detail = std::move(coverage_detail);
        return report;
    }
    std::string geometry_detail;
    if (!dense_attention_geometry_is_valid(model, geometry_detail)) {
        CompatibilityReport report =
            plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
        report.detail = std::move(geometry_detail);
        return report;
    }
    ExecutionPlan plan;
    plan.layer_chain = std::move(layer_chain);
    plan.reserved_bytes = reserved_bytes;
    plan.peak_bytes = total_peak;
    auto append_phase = [&](ExecutionPhase phase) -> std::optional<CompatibilityReport> {
        for (const SemanticLayer& layer : model.layers) {
            const SemanticOperator* attention = nullptr;
            if (!dense_metal_layer(model, layer, attention) || !attention) {
                CompatibilityReport report = plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
                report.layer = layer.layer_index;
                return report;
            }
            const KernelQuery query = phase == ExecutionPhase::Prefill && request.max_batch != 1
                ? dense_prefill_batch_query(model, layer, *attention, request.max_batch)
                : dense_metal_query(model, layer, *attention, phase);
            KernelSelection selected = select_kernel(query, request, capabilities, registry);
            if (const auto* report = std::get_if<CompatibilityReport>(&selected)) {
                CompatibilityReport annotated = *report;
                annotated.layer = layer.layer_index;
                annotated.operator_id = attention->id;
                return annotated;
            }
            PlanEntry entry;
            entry.phase = phase;
            entry.operator_id = attention->id;
            entry.descriptor = std::get<KernelDescriptor>(selected);
            entry.kernel_id = entry.descriptor.id;
            for (uint32_t index = 0; index != layer.operator_count; ++index) {
                const SemanticOperator& op = model.operators[layer.first_operator + index];
                for (uint32_t tensor_id : op.tensors) {
                    if (tensor_id >= model.tensors.size()) return plan_error(CompatibilityError::IR_REFERENCE_INVALID);
                    const SemanticTensor& tensor = model.tensors[tensor_id];
                    for (const TensorPlane& plane : tensor.planes) {
                        if (plane.offset > UINT64_MAX - plane.length) return plan_error(CompatibilityError::PACKAGE_BOUNDS_INVALID);
                        entry.tensors.push_back({tensor.id, plane.artifact_id, plane.offset, plane.length, plane.alignment,
                                                 tensor.logical_type, tensor.layout.kind, tensor.quantization.kind});
                    }
                }
            }
            for (uint32_t state_id : attention->states) {
                const SemanticState* state = state_with_id(model, state_id);
                if (!state || state->formats.empty()) return plan_error(CompatibilityError::IR_STATE_INVALID);
                entry.states.push_back({state->id, state->kind, state->formats[0].kind,
                                        state->formats[0].alignment, state->formats[0].tile_tokens});
            }
            plan.entries.push_back(std::move(entry));
        }
        return std::nullopt;
    };
    if (request.enable_prefill) {
        if (auto report = append_phase(ExecutionPhase::Prefill)) return *report;
    }
    if (request.enable_decode) {
        if (auto report = append_phase(ExecutionPhase::Decode)) return *report;
    }
    return plan;
}

PlanResult plan_canonical_metal_recurrent(const SemanticModel& model, const SessionRequest& request,
                                          const RuntimeCapabilities& capabilities,
                                          const std::vector<KernelDescriptor>& registry) {
    const bool schema3 = model.schema_major == 3 && model.schema_minor == 0 &&
                         model.opset_major == 3 && model.opset_minor == 0;
    const bool schema4 = model.schema_major == 4 && model.schema_minor == 0 &&
                         model.opset_major == 4 && model.opset_minor == 0;
    if (!schema3 && !schema4) {
        return plan_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
    }
    if (request.enable_streaming) return plan_error(CompatibilityError::STREAMING_UNSUPPORTED);
    if (request.enable_speculation) return plan_error(CompatibilityError::FALLBACK_FORBIDDEN);
    if ((!request.enable_prefill && !request.enable_decode) || !request.max_context ||
        request.max_context > model.maximum_context || !request.max_batch) {
        return plan_error(!request.max_context || request.max_context > model.maximum_context
                              ? CompatibilityError::PLAN_CONTEXT_EXCEEDED
                              : CompatibilityError::RUNTIME_INPUT_INVALID);
    }
    std::vector<CanonicalLayerBoundary> layer_chain;
    std::string layer_chain_detail;
    if (!canonical_layer_chain_witness(model, layer_chain, layer_chain_detail)) {
        CompatibilityReport report = plan_error(CompatibilityError::IR_REFERENCE_INVALID);
        report.detail = std::move(layer_chain_detail);
        return report;
    }
    std::string attention_window_detail;
    if (!canonical_attention_window_is_global(model, attention_window_detail)) {
        CompatibilityReport report = plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
        report.detail = std::move(attention_window_detail);
        return report;
    }
    bool state_memory_valid = false;
    const uint64_t reserved_bytes = state_bytes(model, request, state_memory_valid);
    uint64_t total_peak = 0;
    if (!state_memory_valid || !peak_bytes(model, request, reserved_bytes, total_peak) || total_peak > request.memory_limit) {
        return plan_error(CompatibilityError::PLAN_MEMORY_EXCEEDED);
    }
    ExecutionPlan plan;
    plan.layer_chain = std::move(layer_chain);
    plan.reserved_bytes = reserved_bytes;
    plan.peak_bytes = total_peak;
    auto append_phase = [&](ExecutionPhase phase) -> std::optional<CompatibilityReport> {
        for (const SemanticLayer& layer : model.layers) {
            const SemanticOperator* delta = nullptr;
            if (!recurrent_metal_layer(model, layer, delta) || !delta) {
                CompatibilityReport report = plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
                report.layer = layer.layer_index;
                return report;
            }
            KernelSelection selected = select_kernel(recurrent_metal_query(model, layer, *delta, phase), request,
                                                     capabilities, registry);
            if (const auto* report = std::get_if<CompatibilityReport>(&selected)) {
                CompatibilityReport annotated = *report;
                annotated.layer = layer.layer_index;
                annotated.operator_id = delta->id;
                return annotated;
            }
            PlanEntry entry;
            entry.phase = phase;
            entry.operator_id = delta->id;
            entry.descriptor = std::get<KernelDescriptor>(selected);
            entry.kernel_id = entry.descriptor.id;
            for (uint32_t index = 0; index != layer.operator_count; ++index) {
                const SemanticOperator& op = model.operators[layer.first_operator + index];
                for (uint32_t tensor_id : op.tensors) {
                    if (tensor_id >= model.tensors.size()) return plan_error(CompatibilityError::IR_REFERENCE_INVALID);
                    const SemanticTensor& tensor = model.tensors[tensor_id];
                    for (const TensorPlane& plane : tensor.planes) {
                        if (plane.offset > UINT64_MAX - plane.length) return plan_error(CompatibilityError::PACKAGE_BOUNDS_INVALID);
                        entry.tensors.push_back({tensor.id, plane.artifact_id, plane.offset, plane.length, plane.alignment,
                                                 tensor.logical_type, tensor.layout.kind, tensor.quantization.kind});
                    }
                }
                for (uint32_t state_id : op.states) {
                    const SemanticState* state = state_with_id(model, state_id);
                    if (!state || state->formats.empty()) return plan_error(CompatibilityError::IR_STATE_INVALID);
                    const auto duplicate = std::find_if(entry.states.begin(), entry.states.end(), [&](const CheckedStateBinding& binding) {
                        return binding.state_id == state->id;
                    });
                    if (duplicate == entry.states.end()) {
                        entry.states.push_back({state->id, state->kind, state->formats[0].kind,
                                                state->formats[0].alignment, state->formats[0].tile_tokens});
                    }
                }
            }
            plan.entries.push_back(std::move(entry));
        }
        return std::nullopt;
    };
    if (request.enable_prefill) {
        if (auto report = append_phase(ExecutionPhase::Prefill)) return *report;
    }
    if (request.enable_decode) {
        if (auto report = append_phase(ExecutionPhase::Decode)) return *report;
    }
    return plan;
}

PlanResult plan_session(const SemanticModel& model, const SessionRequest& request,
                        const RuntimeCapabilities& capabilities,
                        const std::vector<KernelDescriptor>& registry,
                        const PhysicalCodecRegistry& codec_registry) {
    if (request.enable_streaming) return plan_error(CompatibilityError::STREAMING_UNSUPPORTED);
    if (!request.max_context || request.max_context > model.maximum_context) return plan_error(CompatibilityError::PLAN_CONTEXT_EXCEEDED);
    if (!request.max_batch) return plan_error(CompatibilityError::RUNTIME_INPUT_INVALID);
    if (!model.states.empty() && !capabilities.global_fp32_kv) return plan_error(CompatibilityError::CAPABILITY_MISSING);
    if (request.enable_speculation && (!capabilities.transactional_state || model.states.empty())) {
        return plan_error(CompatibilityError::FALLBACK_FORBIDDEN);
    }
    const bool codec_aware = !codec_registry.codecs.empty() || !codec_registry.tensors.empty();
    std::optional<CodecRegistryIndex> codec_index;
    if (codec_aware) {
        if (!physical_codec_registry_matches_model(codec_registry, model)) {
            return plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
        }
        codec_index = build_codec_registry_index(codec_registry);
        if (!codec_index) return plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
    }

    bool state_memory_valid = false;
    const uint64_t reserved_bytes = state_bytes(model, request, state_memory_valid);
    uint64_t total_peak = 0;
    if (!state_memory_valid || !peak_bytes(model, request, reserved_bytes, total_peak) || total_peak > request.memory_limit) {
        return plan_error(CompatibilityError::PLAN_MEMORY_EXCEEDED);
    }

    ExecutionPlan plan;
    plan.codec_aware = codec_aware;
    plan.reserved_bytes = reserved_bytes;
    plan.peak_bytes = total_peak;
    auto add_entry = [&](const SemanticOperator& op, ExecutionPhase phase) -> std::optional<CompatibilityReport> {
        KernelQuery query = query_for(model, op, phase);
        std::vector<PhysicalTensorCodecBinding> codec_bindings;
        if (codec_aware) {
            codec_bindings.reserve(op.tensors.size());
            for (uint32_t tensor_index : op.tensors) {
                if (tensor_index >= model.tensors.size()) {
                    return plan_error(CompatibilityError::IR_REFERENCE_INVALID);
                }
                const auto binding = bind_physical_codec(model.tensors[tensor_index], *codec_index);
                if (!binding) return plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
                query.physical_codecs.push_back(binding->identity);
                codec_bindings.push_back(*binding);
            }
        }
        KernelSelection selected = select_kernel(query, request, capabilities, registry);
        if (const auto* report = std::get_if<CompatibilityReport>(&selected)) {
            CompatibilityReport annotated = *report;
            annotated.operator_id = op.id;
            return annotated;
        }
        const KernelDescriptor descriptor = std::get<KernelDescriptor>(selected);
        PlanEntry entry;
        entry.codec_aware = codec_aware;
        entry.phase = phase;
        entry.operator_id = op.id;
        entry.kernel_id = descriptor.id;
        entry.descriptor = descriptor;
        entry.codec_bindings = std::move(codec_bindings);
        for (uint32_t tensor_id : op.tensors) {
            if (tensor_id >= model.tensors.size()) return plan_error(CompatibilityError::IR_REFERENCE_INVALID);
            const SemanticTensor& tensor = model.tensors[tensor_id];
            for (const TensorPlane& plane : tensor.planes) {
                if (plane.offset > UINT64_MAX - plane.length) return plan_error(CompatibilityError::PACKAGE_BOUNDS_INVALID);
                entry.tensors.push_back({tensor.id, plane.artifact_id, plane.offset, plane.length, plane.alignment,
                                         tensor.logical_type, tensor.layout.kind, tensor.quantization.kind});
            }
        }
        for (uint32_t state_id : op.states) {
            const auto found = std::find_if(model.states.begin(), model.states.end(), [&](const SemanticState& state) { return state.id == state_id; });
            if (found == model.states.end() || found->formats.empty()) return plan_error(CompatibilityError::IR_STATE_INVALID);
            entry.states.push_back({found->id, found->kind, found->formats.front().kind,
                                    found->formats.front().alignment, found->formats.front().tile_tokens});
        }
        plan.entries.push_back(std::move(entry));
        return std::nullopt;
    };
    auto append_phase = [&](ExecutionPhase phase) -> std::optional<CompatibilityReport> {
        for (const SemanticOperator& op : model.operators) {
            if (auto report = add_entry(op, phase)) return report;
        }
        return std::nullopt;
    };

    if (request.enable_prefill) {
        if (auto report = append_phase(ExecutionPhase::Prefill)) return *report;
    }
    if (request.enable_decode) {
        if (auto report = append_phase(ExecutionPhase::Decode)) return *report;
    }

    const auto state_operator = std::find_if(model.operators.begin(), model.operators.end(), [](const SemanticOperator& op) {
        return !op.states.empty();
    });
    if (!model.states.empty()) {
        if (state_operator == model.operators.end()) return plan_error(CompatibilityError::IR_STATE_INVALID);
        if (auto report = add_entry(*state_operator, ExecutionPhase::StateUpdate)) return *report;
        if (request.enable_speculation) {
            if (auto report = add_entry(*state_operator, ExecutionPhase::Rollback)) return *report;
        }
    }
    if ((request.enable_prefill || request.enable_decode) && !model.operators.empty()) {
        if (auto report = add_entry(model.operators.back(), ExecutionPhase::Output)) return *report;
    }
    return plan;
}

bool plan_entry_matches(const SemanticModel& model, const PlanEntry& entry) {
    if (entry.kernel_id == 0 || entry.kernel_id != entry.descriptor.id) return false;
    const auto operator_it = std::find_if(model.operators.begin(), model.operators.end(), [&](const SemanticOperator& op) {
        return op.id == entry.operator_id;
    });
    if (operator_it == model.operators.end() || !implementation_matches(operator_it->kind, entry.descriptor.implementation)) {
        return false;
    }
    const SemanticLayer* fused_layer = nullptr;
    KernelQuery query = query_for(model, *operator_it, entry.phase);
    if (entry.descriptor.implementation == KernelImplementation::MetalDenseToken) {
        fused_layer = layer_for_operator(model, operator_it->id);
        const SemanticOperator* attention = nullptr;
        if (!fused_layer) return false;
        if (moe_metal_layer(model, *fused_layer, attention) && attention == &*operator_it) {
            query = moe_metal_query(model, *fused_layer, *operator_it, entry.phase);
        } else {
            if (!dense_metal_layer(model, *fused_layer, attention) || attention != &*operator_it) return false;
            query = dense_metal_query(model, *fused_layer, *operator_it, entry.phase);
        }
    }
    if (entry.descriptor.implementation == KernelImplementation::MetalDensePrefillBatch) {
        fused_layer = layer_for_operator(model, operator_it->id);
        const SemanticOperator* attention = nullptr;
        if (!fused_layer || !dense_prefill_batch_layer(model, *fused_layer, attention) || attention != &*operator_it) return false;
        query = dense_prefill_batch_query(model, *fused_layer, *operator_it, 2);
    }
    if (entry.descriptor.implementation == KernelImplementation::MetalRecurrentToken) {
        fused_layer = layer_for_operator(model, operator_it->id);
        const SemanticOperator* delta = nullptr;
        if (!fused_layer || !recurrent_metal_layer(model, *fused_layer, delta) || delta != &*operator_it) return false;
        query = recurrent_metal_query(model, *fused_layer, *operator_it, entry.phase);
    }
    if (entry.codec_aware) {
        if (entry.codec_bindings.size() != operator_it->tensors.size()) return false;
        query.physical_codecs.reserve(entry.codec_bindings.size());
        for (size_t index = 0; index != entry.codec_bindings.size(); ++index) {
            const uint32_t tensor_index = operator_it->tensors[index];
            if (tensor_index >= model.tensors.size() ||
                !binding_matches_tensor(model.tensors[tensor_index], entry.codec_bindings[index])) {
                return false;
            }
            query.physical_codecs.push_back(entry.codec_bindings[index].identity);
        }
        if (!operator_it->tensors.empty() &&
            entry.descriptor.pattern.required_physical_codecs.empty()) return false;
    } else if (!entry.codec_bindings.empty() ||
               !entry.descriptor.pattern.required_physical_codecs.empty()) {
        return false;
    }
    if (!pattern_matches(entry.descriptor.pattern, query)) return false;
    std::vector<CheckedTensorSpan> expected_tensors;
    const uint32_t first = fused_layer ? fused_layer->first_operator : 0;
    const uint32_t count = fused_layer ? fused_layer->operator_count : 1;
    for (uint32_t operator_index = 0; operator_index != count; ++operator_index) {
        const SemanticOperator& expected_operator = fused_layer ? model.operators[first + operator_index] : *operator_it;
        for (uint32_t tensor_id : expected_operator.tensors) {
            if (tensor_id >= model.tensors.size()) return false;
            const SemanticTensor& tensor = model.tensors[tensor_id];
            for (const TensorPlane& plane : tensor.planes) {
                if (plane.offset > UINT64_MAX - plane.length) return false;
                expected_tensors.push_back({tensor.id, plane.artifact_id, plane.offset, plane.length, plane.alignment,
                                            tensor.logical_type, tensor.layout.kind, tensor.quantization.kind});
            }
        }
    }
    if (expected_tensors.size() != entry.tensors.size()) return false;
    for (size_t index = 0; index != expected_tensors.size(); ++index) {
        const auto& expected = expected_tensors[index];
        const auto& actual = entry.tensors[index];
        if (expected.tensor_id != actual.tensor_id || expected.artifact_id.value != actual.artifact_id.value ||
            expected.offset != actual.offset || expected.length != actual.length || expected.alignment != actual.alignment ||
            expected.logical_type != actual.logical_type || expected.layout != actual.layout || expected.quantization != actual.quantization) return false;
    }
    std::vector<CheckedStateBinding> expected_states;
    for (uint32_t operator_index = 0; operator_index != count; ++operator_index) {
        const SemanticOperator& expected_operator = fused_layer ? model.operators[first + operator_index] : *operator_it;
        for (uint32_t state_id : expected_operator.states) {
            const auto found = std::find_if(model.states.begin(), model.states.end(), [&](const SemanticState& state) { return state.id == state_id; });
            if (found == model.states.end() || found->formats.empty()) return false;
            const auto duplicate = std::find_if(expected_states.begin(), expected_states.end(), [&](const CheckedStateBinding& binding) {
                return binding.state_id == found->id;
            });
            if (duplicate == expected_states.end()) {
                expected_states.push_back({found->id, found->kind, found->formats.front().kind,
                                           found->formats.front().alignment, found->formats.front().tile_tokens});
            }
        }
    }
    if (expected_states.size() != entry.states.size()) return false;
    for (size_t index = 0; index != expected_states.size(); ++index) {
        const auto& expected = expected_states[index];
        const auto& actual = entry.states[index];
        if (expected.state_id != actual.state_id || expected.kind != actual.kind || expected.format != actual.format ||
            expected.alignment != actual.alignment || expected.tile_tokens != actual.tile_tokens) return false;
    }
    return true;
}

} // namespace Laplace
