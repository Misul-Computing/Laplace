#include "execution_plan.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>

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
        tensor->planes.size() != 1 || tensor->planes[0].kind != PlaneKind::Values ||
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

bool dense_metal_layer(const SemanticModel& model, const SemanticLayer& layer, const SemanticOperator*& attention,
                       uint32_t* weight_format = nullptr) {
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
    const SemanticOperator* split = nullptr;
    const SemanticOperator* gated_attention = nullptr;
    uint32_t split_count = 0;
    uint32_t gated_attention_count = 0;
    attention = nullptr;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind == OperatorKind::Rope) rope = rope ? nullptr : &op;
        if (op.kind == OperatorKind::SwiGlu) swiglu = swiglu ? nullptr : &op;
        if (op.kind == OperatorKind::AxisSplit) {
            split = split_count++ == 0 ? &op : nullptr;
        }
        if (op.kind == OperatorKind::GatedAttention) {
            gated_attention = gated_attention_count++ == 0 ? &op : nullptr;
        }
        if (op.kind == OperatorKind::CausalAttention) {
            attention = attention ? nullptr : &op;
        }
    }
    const bool fused_query_gate = query_gate != nullptr;
    const SemanticOperator* query_projection = fused_query_gate ? query_gate : query;
    const auto* rope_payload = rope ? std::get_if<RopePayload>(&rope->payload) : nullptr;
    const auto* attention_payload = attention ? std::get_if<CausalAttentionPayload>(&attention->payload) : nullptr;
    const auto* swiglu_payload = swiglu ? std::get_if<SwiGluPayload>(&swiglu->payload) : nullptr;
    const bool multi_section_rope = rope_payload && rope_payload->pairing == RopePairing::MultiSectionHalfSplit;
    if (!attn_norm || !query_projection || !key || !value || !output || !ffn_norm || !gate || !up || !down || !rope_payload ||
        !attention_payload || !swiglu_payload || swiglu_payload->activation != ActivationKind::Silu ||
        (rope_payload->pairing != RopePairing::HalfSplit && rope_payload->pairing != RopePairing::Interleaved &&
         rope_payload->pairing != RopePairing::MultiSectionHalfSplit) ||
        !rope_payload->position_from_cursor ||
        attention_payload->mask != AttentionMask::Causal || attention_payload->cache_policy != CachePolicy::Global ||
        attention_payload->head_dimension < 32 || attention_payload->head_dimension > 512 ||
        attention_payload->head_dimension % 16 != 0 || !dense_kv_states(model, *attention) ||
        !exact_dense_tensor(model, *attn_norm, TensorRole::AttentionNormWeight, ScalarType::F32) ||
        (!fused_query_gate && !exact_dense_tensor(model, *query, TensorRole::QueryBias, ScalarType::F32, true)) ||
        !exact_dense_tensor(model, *key, TensorRole::KeyBias, ScalarType::F32, true) ||
        !exact_dense_tensor(model, *value, TensorRole::ValueBias, ScalarType::F32, true) ||
        !exact_dense_tensor(model, *ffn_norm, TensorRole::FfnNormWeight, ScalarType::F32)) return false;
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
    for (const auto [op, role] : std::array<std::pair<const SemanticOperator*, TensorRole>, 7>{{
             {query_projection, fused_query_gate ? TensorRole::AttentionQueryGateWeight : TensorRole::QueryWeight},
             {key, TensorRole::KeyWeight}, {value, TensorRole::ValueWeight},
             {output, TensorRole::AttentionOutputWeight}, {gate, TensorRole::FfnGateWeight},
             {up, TensorRole::FfnUpWeight}, {down, TensorRole::FfnDownWeight}}}) {
        uint32_t candidate = 0;
        if (!metal_dense_matrix_format(model, *op, role, candidate)) return false;
        format |= candidate;
    }
    if (weight_format) *weight_format = format;
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
    if (format == MetalWeightFormatF16) {
        query.storage_type = ScalarType::F16;
        query.layout = PhysicalLayoutKind::ContiguousRowMajor;
        query.quantization = QuantizationKind::None;
        query.alignment = 64;
    } else if ((format & MetalWeightFormatF16) == 0) {
        query.storage_type = ScalarType::U8;
        query.layout = PhysicalLayoutKind::GgufBlocked;
        query.quantization = QuantizationKind::BlockedAffine;
        query.alignment = 32;
        if (const MetalBlockedFormat* blocked = metal_blocked_format(format)) {
            query.quantization = blocked->quantization;
            query.block_elements = blocked->elements;
            query.block_bytes = blocked->bytes;
        } else if ((format & MetalWeightFormatIQ2XXS) != 0) {
            query.quantization = QuantizationKind::Codebook;
        }
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
    const SemanticOperator* l2_query = nullptr;
    const SemanticOperator* l2_key = nullptr;
    const SemanticOperator* gated_rms = nullptr;
    const SemanticOperator* swiglu = nullptr;
    const SemanticOperator* recurrent_residual = nullptr;
    const SemanticOperator* ffn_residual = nullptr;
    delta_out = nullptr;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind == OperatorKind::DepthwiseConvSilu) conv = conv ? nullptr : &op;
        if (op.kind == OperatorKind::L2Normalize) {
            if (!l2_query) l2_query = &op;
            else if (!l2_key) l2_key = &op;
            else l2_key = nullptr;
        }
        if (op.kind == OperatorKind::GatedDeltaNet) delta_out = delta_out ? nullptr : &op;
        if (op.kind == OperatorKind::GatedRmsNorm) gated_rms = gated_rms ? nullptr : &op;
        if (op.kind == OperatorKind::SwiGlu) swiglu = swiglu ? nullptr : &op;
        if (op.kind == OperatorKind::Add) {
            if (!recurrent_residual) recurrent_residual = &op;
            else if (!ffn_residual) ffn_residual = &op;
            else ffn_residual = nullptr;
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
    const uint16_t semantic_version = delta_out ? delta_out->semantic_version : 0;
    if (semantic_version != 3 && semantic_version != 4 && semantic_version != 6) return false;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        if (model.operators[layer.first_operator + index].semantic_version != semantic_version) return false;
    }
    if (!input_norm || !qkv || !gate || !beta || !alpha || !output || !ffn_norm || !ffn_gate || !ffn_up || !ffn_down ||
        !conv || !l2_query || !l2_key || !delta_out || !gated_rms || !swiglu || !recurrent_residual || !ffn_residual ||
        !input_norm_payload || !conv_payload || !delta_payload || !l2_query_payload || !l2_key_payload ||
        !gated_rms_payload || !ffn_norm_payload || !swiglu_payload ||
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
    if (format && (format & MetalWeightFormatF16) == 0) {
        query.storage_type = ScalarType::U8;
        query.layout = PhysicalLayoutKind::GgufBlocked;
        query.quantization = QuantizationKind::BlockedAffine;
        query.alignment = 32;
        if (const MetalBlockedFormat* blocked = metal_blocked_format(format)) {
            query.quantization = blocked->quantization;
            query.block_elements = blocked->elements;
            query.block_bytes = blocked->bytes;
        } else if ((format & MetalWeightFormatIQ2XXS) != 0) {
            query.quantization = QuantizationKind::Codebook;
        }
    }
    if (payload) query.head_dimension = payload->head_dimension;
    return query;
}

bool canonical_metal_operator_coverage(const SemanticModel& model, uint32_t& unexpected_operator) {
    std::vector<bool> covered(model.operators.size(), false);
    for (const SemanticLayer& layer : model.layers) {
        if (layer.first_operator > model.operators.size() || layer.operator_count > model.operators.size() - layer.first_operator) return false;
        for (uint32_t index = 0; index != layer.operator_count; ++index) {
            if (covered[layer.first_operator + index]) return false;
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
        return false;
    }
    return embedding == 1 && final_norm == 1 && output == 1;
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
            query.layout = tensor.layout.kind;
            query.quantization = tensor.quantization.kind;
            query.alignment = tensor.planes.empty() ? 0 : tensor.planes.front().alignment;
            query.storage_type = tensor.planes.empty() ? static_cast<ScalarType>(0) : tensor.planes.front().storage_type;
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

bool requires_scalar_fp32(const RuntimeCapabilities& capabilities) {
    return capabilities.scalar_fp32;
}

bool pattern_matches(const KernelPattern& pattern, const KernelQuery& query) {
    return pattern.semantic_version.contains(query.semantic_version) &&
           pattern.rank.contains(query.rank) &&
           pattern.alignment.contains(query.alignment) &&
           pattern.head_dimension.contains(query.head_dimension) &&
           pattern.batch_rows.contains(query.batch_rows) &&
           (!pattern.head_dimension_multiple || query.head_dimension % pattern.head_dimension_multiple == 0) &&
           pattern.tile_tokens.contains(query.tile_tokens) &&
           pattern.block_elements.contains(query.block_elements) &&
           pattern.block_bytes.contains(query.block_bytes) &&
           (!pattern.allowed_metal_weight_formats ||
            (query.metal_weight_format_mask != 0 &&
             (query.metal_weight_format_mask & ~pattern.allowed_metal_weight_formats) == 0)) &&
           (!pattern.require_mixed_metal_weight_formats ||
            (query.metal_weight_format_mask & (query.metal_weight_format_mask - 1)) != 0) &&
           (!pattern.require_metal_dense_token_pattern || query.metal_dense_token_pattern) &&
           (!pattern.require_metal_dense_prefill_batch_pattern || query.metal_dense_prefill_batch_pattern) &&
           (!pattern.require_metal_recurrent_token_pattern || query.metal_recurrent_token_pattern) &&
           matches(pattern.operation, query.operation) &&
           matches(pattern.phase, query.phase) &&
           matches(pattern.logical_type, query.logical_type) &&
           matches(pattern.storage_type, query.storage_type) &&
           matches(pattern.layout, query.layout) &&
           matches(pattern.quantization, query.quantization) &&
           matches(pattern.state_kind, query.state_kind) &&
           matches(pattern.state_format, query.state_format);
}

KernelSelection select_kernel(const KernelQuery& query, const SessionRequest& request,
                              const RuntimeCapabilities& capabilities,
                              const std::vector<KernelDescriptor>& registry) {
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
        mixed.pattern.allowed_metal_weight_formats = MetalWeightFormatQ4K | MetalWeightFormatQ5_0 |
                                                      MetalWeightFormatQ6K | MetalWeightFormatQ8_0 |
                                                      MetalWeightFormatQ2K | MetalWeightFormatIQ2XXS;
        mixed.pattern.require_mixed_metal_weight_formats = true;
        registry.push_back(std::move(mixed));
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
    for (const SemanticOperator& op : model.operators) {
        if (op.kind != OperatorKind::RouterTopK && op.kind != OperatorKind::RoutedLinear &&
            op.kind != OperatorKind::WeightedExpertReduce) continue;
        CompatibilityReport report = plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
        report.operator_id = op.id;
        report.detail = "canonical Metal MoE operators are not admitted";
        return report;
    }
    if (!schema1 && !schema3 && !schema4 && !schema6)
        return plan_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
    if (request.enable_streaming) return plan_error(CompatibilityError::STREAMING_UNSUPPORTED);
    if (request.enable_speculation) return plan_error(CompatibilityError::FALLBACK_FORBIDDEN);
    if ((!request.enable_prefill && !request.enable_decode) || !request.max_context ||
        request.max_context > model.maximum_context || !request.max_batch) {
        return plan_error(!request.max_context || request.max_context > model.maximum_context
                              ? CompatibilityError::PLAN_CONTEXT_EXCEEDED
                              : CompatibilityError::RUNTIME_INPUT_INVALID);
    }
    bool state_memory_valid = false;
    const uint64_t reserved_bytes = state_bytes(model, request, state_memory_valid);
    uint64_t total_peak = 0;
    if (!state_memory_valid || !peak_bytes(model, request, reserved_bytes, total_peak) || total_peak > request.memory_limit) {
        return plan_error(CompatibilityError::PLAN_MEMORY_EXCEEDED);
    }
    uint32_t unexpected_operator = UINT32_MAX;
    if (!canonical_metal_operator_coverage(model, unexpected_operator)) {
        CompatibilityReport report = plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
        report.operator_id = unexpected_operator;
        return report;
    }

    ExecutionPlan plan;
    plan.reserved_bytes = reserved_bytes;
    plan.peak_bytes = total_peak;
    auto append_phase = [&](ExecutionPhase phase) -> std::optional<CompatibilityReport> {
        for (const SemanticLayer& layer : model.layers) {
            if ((layer.flags & kSemanticLayerFlagSpeculative) != 0) continue;
            const SemanticOperator* dense_anchor = nullptr;
            const SemanticOperator* recurrent_anchor = nullptr;
            const bool dense = dense_metal_layer(model, layer, dense_anchor);
            const bool recurrent = recurrent_metal_layer(model, layer, recurrent_anchor);
            if (dense == recurrent || (!dense_anchor && !recurrent_anchor)) {
                CompatibilityReport report = plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
                report.layer = layer.layer_index;
                report.detail = "canonical Metal layer does not match an admitted full-token pattern";
                return report;
            }
            const SemanticOperator& anchor = dense ? *dense_anchor : *recurrent_anchor;
            if (anchor.semantic_version != model.opset_major) {
                CompatibilityReport report = plan_error(CompatibilityError::IR_VERSION_UNSUPPORTED);
                report.layer = layer.layer_index;
                report.operator_id = anchor.id;
                return report;
            }
            const KernelQuery query = dense
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
                    annotated.detail = std::string(dense ? "canonical Metal dense" : "canonical Metal recurrent") +
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
    bool state_memory_valid = false;
    const uint64_t reserved_bytes = state_bytes(model, request, state_memory_valid);
    uint64_t total_peak = 0;
    if (!state_memory_valid || !peak_bytes(model, request, reserved_bytes, total_peak) || total_peak > request.memory_limit) {
        return plan_error(CompatibilityError::PLAN_MEMORY_EXCEEDED);
    }
    uint32_t unexpected_operator = UINT32_MAX;
    if (!canonical_metal_operator_coverage(model, unexpected_operator)) {
        CompatibilityReport report = plan_error(CompatibilityError::KERNEL_UNAVAILABLE);
        report.operator_id = unexpected_operator;
        return report;
    }
    ExecutionPlan plan;
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
    bool state_memory_valid = false;
    const uint64_t reserved_bytes = state_bytes(model, request, state_memory_valid);
    uint64_t total_peak = 0;
    if (!state_memory_valid || !peak_bytes(model, request, reserved_bytes, total_peak) || total_peak > request.memory_limit) {
        return plan_error(CompatibilityError::PLAN_MEMORY_EXCEEDED);
    }
    ExecutionPlan plan;
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
                        const std::vector<KernelDescriptor>& registry) {
    if (request.enable_streaming) return plan_error(CompatibilityError::STREAMING_UNSUPPORTED);
    if (!request.max_context || request.max_context > model.maximum_context) return plan_error(CompatibilityError::PLAN_CONTEXT_EXCEEDED);
    if (!request.max_batch) return plan_error(CompatibilityError::RUNTIME_INPUT_INVALID);
    if (!model.states.empty() && !capabilities.global_fp32_kv) return plan_error(CompatibilityError::CAPABILITY_MISSING);
    if (request.enable_speculation && (!capabilities.transactional_state || model.states.empty())) {
        return plan_error(CompatibilityError::FALLBACK_FORBIDDEN);
    }

    bool state_memory_valid = false;
    const uint64_t reserved_bytes = state_bytes(model, request, state_memory_valid);
    uint64_t total_peak = 0;
    if (!state_memory_valid || !peak_bytes(model, request, reserved_bytes, total_peak) || total_peak > request.memory_limit) {
        return plan_error(CompatibilityError::PLAN_MEMORY_EXCEEDED);
    }

    ExecutionPlan plan;
    plan.reserved_bytes = reserved_bytes;
    plan.peak_bytes = total_peak;
    auto add_entry = [&](const SemanticOperator& op, ExecutionPhase phase) -> std::optional<CompatibilityReport> {
        KernelSelection selected = select_kernel(query_for(model, op, phase), request, capabilities, registry);
        if (const auto* report = std::get_if<CompatibilityReport>(&selected)) {
            CompatibilityReport annotated = *report;
            annotated.operator_id = op.id;
            return annotated;
        }
        const KernelDescriptor descriptor = std::get<KernelDescriptor>(selected);
        PlanEntry entry;
        entry.phase = phase;
        entry.operator_id = op.id;
        entry.kernel_id = descriptor.id;
        entry.descriptor = descriptor;
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
        if (!dense_metal_layer(model, *fused_layer, attention) || attention != &*operator_it) return false;
        query = dense_metal_query(model, *fused_layer, *operator_it, entry.phase);
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
