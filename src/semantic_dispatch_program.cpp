#include "semantic_dispatch_program.h"

#include <algorithm>
#include <map>
#include <new>
#include <span>
#include <type_traits>

#include <CommonCrypto/CommonDigest.h>

namespace Laplace {
namespace {

constexpr size_t kMaximumDispatchSteps = 1u << 20;

void append_u8(std::vector<uint8_t>& bytes, uint8_t value) {
    bytes.push_back(value);
}

void append_u16(std::vector<uint8_t>& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8));
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8) {
        bytes.push_back(static_cast<uint8_t>(value >> shift));
    }
}

void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8) {
        bytes.push_back(static_cast<uint8_t>(value >> shift));
    }
}

void append_bool(std::vector<uint8_t>& bytes, bool value) {
    append_u8(bytes, value ? 1 : 0);
}

void append_digest(std::vector<uint8_t>& bytes, const Sha256Digest& digest) {
    bytes.insert(bytes.end(), digest.bytes.begin(), digest.bytes.end());
}

Sha256Digest digest_bytes(std::span<const uint8_t> bytes) {
    Sha256Digest digest;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.bytes.data());
    return digest;
}

CompatibilityReport failure(CompatibilityError code, std::string detail,
                            uint32_t operator_id = kSemanticDispatchUnresolved,
                            uint32_t state_id = kSemanticDispatchUnresolved) {
    CompatibilityReport report = compatibility_report(code, std::move(detail));
    report.operator_id = operator_id;
    report.state_id = state_id;
    return report;
}

bool set_failure(CompatibilityReport* output, CompatibilityReport report) {
    if (output) *output = std::move(report);
    return false;
}

bool request_valid(const SemanticDispatchRequest& request, CompatibilityReport* report) {
    if ((request.phase != ExecutionPhase::Prefill && request.phase != ExecutionPhase::Decode) ||
        request.batch_rows == 0 || request.numerical_class != NumericalClass::ExactFp32 ||
        request.include_speculative) {
        return set_failure(report, failure(CompatibilityError::RUNTIME_INPUT_INVALID,
                                           "semantic dispatch request is unsupported"));
    }
    return true;
}

bool has_input_value(const SemanticModel& model, uint32_t id) {
    return id >= model.input_values_first &&
           id - model.input_values_first < model.input_values_count;
}

bool excluded_operator(const std::vector<bool>& excluded, uint32_t id) {
    return id < excluded.size() && excluded[id];
}

bool derive_excluded_operators(const SemanticModel& model,
                               const SemanticDispatchRequest& request,
                               std::vector<bool>& excluded,
                               CompatibilityReport* report) {
    excluded.assign(model.operators.size(), false);
    for (const SemanticLayer& layer : model.layers) {
        const bool speculative = (layer.flags & kSemanticLayerFlagSpeculative) != 0;
        if (!speculative || request.include_speculative) continue;
        if (layer.first_operator > model.operators.size() ||
            layer.operator_count > model.operators.size() - layer.first_operator) {
            return set_failure(report, failure(CompatibilityError::IR_REFERENCE_INVALID,
                                               "semantic layer range is outside the operator graph"));
        }
        for (uint32_t offset = 0; offset < layer.operator_count; ++offset) {
            excluded[layer.first_operator + offset] = true;
        }
    }
    return true;
}

bool validate_graph(const SemanticModel& model, const std::vector<bool>& excluded,
                    std::vector<int64_t>& producers, CompatibilityReport* report) {
    if (model.operators.size() > kMaximumDispatchSteps) {
        return set_failure(report, failure(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                                           "semantic graph has too many dispatch steps"));
    }
    producers.assign(model.values.size(), -1);
    for (uint32_t offset = 0; offset < model.input_values_count; ++offset) {
        const uint32_t value = model.input_values_first + offset;
        if (value >= producers.size() || producers[value] != -1) {
            return set_failure(report, failure(CompatibilityError::IR_REFERENCE_INVALID,
                                               "semantic input values overlap"));
        }
        producers[value] = -2;
    }

    for (size_t index = 0; index < model.operators.size(); ++index) {
        const SemanticOperator& op = model.operators[index];
        std::vector<uint32_t> outputs;
        outputs.reserve(op.outputs.size());
        for (uint32_t value : op.inputs) {
            if (value >= producers.size() || (producers[value] == -1 && !has_input_value(model, value))) {
                return set_failure(report, failure(CompatibilityError::IR_REFERENCE_INVALID,
                                                   "operator input has no producer", op.id));
            }
        }
        for (uint32_t value : op.outputs) {
            if (value >= producers.size() || has_input_value(model, value) ||
                producers[value] != -1 ||
                std::find(outputs.begin(), outputs.end(), value) != outputs.end()) {
                return set_failure(report, failure(CompatibilityError::IR_REFERENCE_INVALID,
                                                   "operator output is replayed or overlaps an input", op.id));
            }
            outputs.push_back(value);
        }
        for (uint32_t value : outputs) producers[value] = static_cast<int64_t>(index);
    }
    for (uint32_t value = 0; value < producers.size(); ++value) {
        if (producers[value] == -1) {
            return set_failure(report, failure(CompatibilityError::IR_REFERENCE_INVALID,
                                               "semantic value is not produced", kSemanticDispatchUnresolved));
        }
    }
    for (size_t index = 0; index < model.operators.size(); ++index) {
        if (excluded_operator(excluded, static_cast<uint32_t>(index))) continue;
        const SemanticOperator& op = model.operators[index];
        for (uint32_t value : op.inputs) {
            const int64_t producer = producers[value];
            if (producer >= 0 && excluded_operator(excluded, static_cast<uint32_t>(producer))) {
                return set_failure(report, failure(CompatibilityError::IR_REFERENCE_INVALID,
                                                   "active operator consumes a speculative value", op.id));
            }
        }
    }
    return true;
}

bool derive_state_effects(const SemanticModel& model, const SemanticOperator& op,
                          std::vector<SemanticDispatchStateEffect>& effects,
                          CompatibilityReport* report) {
    effects.clear();
    if (op.states.empty()) return true;

    const auto require_state = [&](size_t index, StateKind kind, StateUpdateKind update,
                                   PositionPolicy position) {
        if (index >= op.states.size() || op.states[index] >= model.states.size()) {
            return false;
        }
        const SemanticState& state = model.states[op.states[index]];
        return state.kind == kind && state.update_kind == update &&
               state.position_policy == position;
    };

    switch (op.kind) {
    case OperatorKind::CausalAttention:
        if (op.states.size() != 2 ||
            !require_state(0, StateKind::KeyCache, StateUpdateKind::AppendKey,
                           PositionPolicy::AppendOnly) ||
            !require_state(1, StateKind::ValueCache, StateUpdateKind::AppendValue,
                           PositionPolicy::AppendOnly)) {
            return set_failure(report, failure(CompatibilityError::STATE_ABI_MISMATCH,
                                               "causal attention state effects do not match the state ABI", op.id));
        }
        break;
    case OperatorKind::DepthwiseConvSilu:
        if (op.states.size() != 1 ||
            !require_state(0, StateKind::RecurrentConvHistory, StateUpdateKind::ShiftHistory,
                           PositionPolicy::ReplaceAtCursor)) {
            return set_failure(report, failure(CompatibilityError::STATE_ABI_MISMATCH,
                                               "convolution state effect does not match the state ABI", op.id));
        }
        break;
    case OperatorKind::GatedDeltaNet:
        if (op.states.size() != 1 ||
            !require_state(0, StateKind::RecurrentDeltaMatrix, StateUpdateKind::DeltaMatrix,
                           PositionPolicy::ReplaceAtCursor)) {
            return set_failure(report, failure(CompatibilityError::STATE_ABI_MISMATCH,
                                               "recurrent state effect does not match the state ABI", op.id));
        }
        break;
    default:
        return set_failure(report, failure(CompatibilityError::IR_STATE_INVALID,
                                           "operator carries an unsupported state effect", op.id));
    }
    for (uint32_t id : op.states) {
        const SemanticState& state = model.states[id];
        const StateUpdateKind update = state.update_kind;
        effects.push_back({id, SemanticDispatchStateAccess::ReadWrite, update});
    }
    return true;
}

bool derive_execution_roots(const SemanticModel& model, const std::vector<bool>& excluded,
                            const std::vector<int64_t>& producers,
                            std::vector<uint32_t>& roots,
                            CompatibilityReport* report) {
    roots.clear();
    if (model.output_values_count == 0 ||
        model.output_values_first > model.values.size() ||
        model.output_values_count > model.values.size() - model.output_values_first) {
        return set_failure(report, failure(CompatibilityError::IR_REFERENCE_INVALID,
                                           "semantic graph has no valid output values"));
    }
    std::vector<bool> reachable(model.operators.size(), false);
    std::vector<uint32_t> pending;
    pending.reserve(model.output_values_count + model.states.size());
    for (uint32_t offset = 0; offset < model.output_values_count; ++offset) {
        const uint32_t value = model.output_values_first + offset;
        const int64_t producer = producers[value];
        if (producer < 0 || excluded_operator(excluded, static_cast<uint32_t>(producer))) {
            return set_failure(report, failure(CompatibilityError::IR_REFERENCE_INVALID,
                                               "graph output is not produced by an active operator"));
        }
        pending.push_back(static_cast<uint32_t>(producer));
        roots.push_back(model.operators[static_cast<size_t>(producer)].id);
    }
    for (uint32_t index = 0; index < model.operators.size(); ++index) {
        if (!excluded_operator(excluded, index) && !model.operators[index].states.empty()) {
            pending.push_back(index);
            roots.push_back(model.operators[index].id);
        }
    }
    while (!pending.empty()) {
        const uint32_t index = pending.back();
        pending.pop_back();
        if (index >= model.operators.size() || reachable[index]) continue;
        reachable[index] = true;
        for (uint32_t value : model.operators[index].inputs) {
            if (value >= producers.size()) {
                return set_failure(report, failure(CompatibilityError::IR_REFERENCE_INVALID,
                                                   "execution root input is outside the value table"));
            }
            const int64_t producer = producers[value];
            if (producer >= 0) pending.push_back(static_cast<uint32_t>(producer));
        }
    }
    for (uint32_t index = 0; index < model.operators.size(); ++index) {
        if (!excluded_operator(excluded, index) && !reachable[index]) {
            return set_failure(report, failure(CompatibilityError::IR_REFERENCE_INVALID,
                                               "active pure operator is unreachable from an output or state effect",
                                               model.operators[index].id));
        }
    }
    std::sort(roots.begin(), roots.end());
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
    if (roots.empty()) {
        return set_failure(report, failure(CompatibilityError::IR_REFERENCE_INVALID,
                                           "graph has no active terminal operator"));
    }
    return true;
}

void append_vector(std::vector<uint8_t>& bytes, const std::vector<uint32_t>& values) {
    append_u32(bytes, static_cast<uint32_t>(values.size()));
    for (uint32_t value : values) append_u32(bytes, value);
}

void append_effects(std::vector<uint8_t>& bytes,
                    const std::vector<SemanticDispatchStateEffect>& effects) {
    append_u32(bytes, static_cast<uint32_t>(effects.size()));
    for (const auto& effect : effects) {
        append_u32(bytes, effect.state_id);
        append_u8(bytes, static_cast<uint8_t>(effect.access));
        append_u16(bytes, static_cast<uint16_t>(effect.update_kind));
    }
}

void append_session_effects(
    std::vector<uint8_t>& bytes,
    const std::vector<SemanticDispatchSessionEffect>& effects) {
    append_u32(bytes, static_cast<uint32_t>(effects.size()));
    for (const auto& effect : effects) append_u8(bytes, static_cast<uint8_t>(effect.kind));
}

void append_dimensions(std::vector<uint8_t>& bytes,
                       const std::vector<Dimension>& dimensions) {
    append_u32(bytes, static_cast<uint32_t>(dimensions.size()));
    for (const Dimension& dimension : dimensions) {
        append_u8(bytes, static_cast<uint8_t>(dimension.kind));
        append_u64(bytes, dimension.constant_or_symbol);
    }
}

void append_value_contract(std::vector<uint8_t>& bytes, const SemanticValue& value) {
    append_u16(bytes, static_cast<uint16_t>(value.logical_type));
    append_dimensions(bytes, value.dimensions);
    append_u8(bytes, value.flags);
}

void append_tensor_contract(std::vector<uint8_t>& bytes, const SemanticTensor& tensor) {
    append_u16(bytes, static_cast<uint16_t>(tensor.logical_type));
    append_dimensions(bytes, tensor.dimensions);
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
    append_u32(bytes, static_cast<uint32_t>(tensor.planes.size()));
    for (const TensorPlane& plane : tensor.planes) {
        append_u16(bytes, static_cast<uint16_t>(plane.kind));
        append_u16(bytes, static_cast<uint16_t>(plane.storage_type));
        append_u64(bytes, plane.length);
        append_u32(bytes, plane.alignment);
        append_u32(bytes, plane.flags);
    }
    append_u16(bytes, tensor.flags);
}

void append_state_contract(std::vector<uint8_t>& bytes, const SemanticState& state) {
    append_u16(bytes, static_cast<uint16_t>(state.kind));
    append_u16(bytes, state.semantic_version);
    append_u16(bytes, static_cast<uint16_t>(state.update_kind));
    append_u16(bytes, static_cast<uint16_t>(state.position_policy));
    append_dimensions(bytes, state.dimensions);
    append_u32(bytes, static_cast<uint32_t>(state.formats.size()));
    for (const StateFormat& format : state.formats) {
        append_u16(bytes, static_cast<uint16_t>(format.kind));
        append_u16(bytes, format.version);
        append_u16(bytes, static_cast<uint16_t>(format.logical_type));
        append_u16(bytes, static_cast<uint16_t>(format.encoded_type));
        append_u16(bytes, static_cast<uint16_t>(format.logical_domain));
        append_u16(bytes, static_cast<uint16_t>(format.encoded_domain));
        append_u16(bytes, static_cast<uint16_t>(format.codec));
        append_u16(bytes, static_cast<uint16_t>(format.cache_policy));
        append_u16(bytes, static_cast<uint16_t>(format.layout_policy));
        append_u16(bytes, format.flags);
        append_u32(bytes, format.tile_tokens);
        append_u32(bytes, format.mutable_tokens);
        append_u32(bytes, format.alignment);
        append_u32(bytes, format.reserved);
    }
    append_u16(bytes, state.flags);
}

void append_payload(std::vector<uint8_t>& bytes, const OperatorPayload& payload) {
    append_u8(bytes, static_cast<uint8_t>(payload.index()));
    std::visit([&](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, EmbeddingLookupPayload>) {
            append_u32(bytes, value.scale_f32_bits); append_u32(bytes, value.vocabulary);
            append_u32(bytes, value.width); append_u32(bytes, value.flags);
        } else if constexpr (std::is_same_v<T, RmsNormPayload>) {
            append_u32(bytes, value.epsilon_f32_bits);
            append_u32(bytes, static_cast<uint32_t>(value.axis));
            append_u8(bytes, value.weight_mode);
            append_u8(bytes, static_cast<uint8_t>(value.affine_geometry));
            append_u32(bytes, value.reduction_extent);
        } else if constexpr (std::is_same_v<T, LinearPayload>) {
            append_bool(bytes, value.transpose_weight); append_bool(bytes, value.has_bias);
            append_u16(bytes, static_cast<uint16_t>(value.accumulation_type));
        } else if constexpr (std::is_same_v<T, RopePayload>) {
            append_u8(bytes, static_cast<uint8_t>(value.pairing));
            append_bool(bytes, value.position_from_cursor); append_u32(bytes, value.rotary_dimension);
            append_u32(bytes, value.base_f32_bits); append_u32(bytes, value.scale_f32_bits);
            for (uint32_t section : value.position_sections) append_u32(bytes, section);
            append_u32(bytes, value.frequency_dimension);
        } else if constexpr (std::is_same_v<T, CausalAttentionPayload>) {
            append_u32(bytes, value.query_heads); append_u32(bytes, value.kv_heads);
            append_u32(bytes, value.head_dimension); append_u32(bytes, value.scale_f32_bits);
            append_u8(bytes, static_cast<uint8_t>(value.mask));
            append_u16(bytes, static_cast<uint16_t>(value.cache_policy));
            append_u8(bytes, static_cast<uint8_t>(value.window));
            append_u32(bytes, value.window_tokens);
            append_u8(bytes, static_cast<uint8_t>(value.value_source));
            append_u32(bytes, value.value_source_value);
        } else if constexpr (std::is_same_v<T, SwiGluPayload>) {
            append_u16(bytes, static_cast<uint16_t>(value.activation));
        } else if constexpr (std::is_same_v<T, AddPayload> ||
                             std::is_same_v<T, GatedAttentionPayload>) {
        } else if constexpr (std::is_same_v<T, DepthwiseConvSiluPayload>) {
            append_u32(bytes, value.qk_heads); append_u32(bytes, value.value_heads);
            append_u32(bytes, value.head_dimension); append_u32(bytes, value.kernel);
        } else if constexpr (std::is_same_v<T, GatedDeltaNetPayload>) {
            append_u32(bytes, value.qk_heads); append_u32(bytes, value.value_heads);
            append_u32(bytes, value.head_dimension);
            append_u8(bytes, static_cast<uint8_t>(value.qk_mapping));
            append_u8(bytes, static_cast<uint8_t>(value.beta_transform));
            append_u8(bytes, static_cast<uint8_t>(value.decay_transform));
            append_u8(bytes, static_cast<uint8_t>(value.state_layout));
            append_u32(bytes, value.flags);
        } else if constexpr (std::is_same_v<T, GatedRmsNormPayload>) {
            append_u32(bytes, value.epsilon_f32_bits);
            append_u16(bytes, static_cast<uint16_t>(value.gate_activation));
            append_u8(bytes, value.weight_mode);
        } else if constexpr (std::is_same_v<T, L2NormalizePayload>) {
            append_u32(bytes, value.epsilon_f32_bits);
        } else if constexpr (std::is_same_v<T, AxisSplitPayload>) {
            append_u32(bytes, value.first_width); append_u32(bytes, value.second_width);
        } else if constexpr (std::is_same_v<T, ConcatPayload>) {
            append_u32(bytes, static_cast<uint32_t>(value.axis));
        } else if constexpr (std::is_same_v<T, RouterTopKPayload>) {
            append_u32(bytes, value.expert_count); append_u32(bytes, value.selected_count);
            append_u8(bytes, static_cast<uint8_t>(value.score_domain));
            append_u8(bytes, static_cast<uint8_t>(value.normalization_order));
            append_u8(bytes, static_cast<uint8_t>(value.selected_weight_normalization));
            append_u8(bytes, static_cast<uint8_t>(value.tie_policy));
            append_u8(bytes, static_cast<uint8_t>(value.weight_source));
            append_u32(bytes, value.flags);
        } else if constexpr (std::is_same_v<T, RoutedLinearPayload>) {
            append_u16(bytes, static_cast<uint16_t>(value.accumulation_type));
        } else if constexpr (std::is_same_v<T, GatedActivationPayload>) {
            append_u16(bytes, static_cast<uint16_t>(value.activation));
        } else if constexpr (std::is_same_v<T, WeightedExpertReducePayload>) {
            append_u8(bytes, static_cast<uint8_t>(value.association));
            append_u8(bytes, static_cast<uint8_t>(value.scale_source));
            append_u16(bytes, static_cast<uint16_t>(value.accumulation_type));
        } else if constexpr (std::is_same_v<T, ScalePayload>) {
            append_u8(bytes, static_cast<uint8_t>(value.source));
            append_u32(bytes, value.literal_f32_bits);
        } else if constexpr (std::is_same_v<T, TanhSoftcapPayload>) {
            append_u32(bytes, value.cap_f32_bits);
        }
    }, payload);
}

std::vector<uint8_t> requirement_key(const SemanticModel& model,
                                     const SemanticOperator& op,
                                     const SemanticDispatchRequest& request,
                                     const SemanticDispatchStep& step) {
    std::vector<uint8_t> bytes;
    bytes.reserve(512);
    static constexpr std::array<uint8_t, 8> domain = {'L','P','D','R','E','Q',2,0};
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    append_u8(bytes, static_cast<uint8_t>(step.kind));
    append_u16(bytes, static_cast<uint16_t>(step.operation));
    append_u16(bytes, op.semantic_version);
    append_u16(bytes, static_cast<uint16_t>(step.phase));
    append_u32(bytes, request.batch_rows);
    append_u16(bytes, static_cast<uint16_t>(step.numerical_class));
    append_payload(bytes, op.payload);
    append_u32(bytes, static_cast<uint32_t>(op.inputs.size()));
    for (uint32_t id : op.inputs) append_value_contract(bytes, model.values[id]);
    append_u32(bytes, static_cast<uint32_t>(op.outputs.size()));
    for (uint32_t id : op.outputs) append_value_contract(bytes, model.values[id]);
    append_u32(bytes, static_cast<uint32_t>(op.tensors.size()));
    for (uint32_t id : op.tensors) append_tensor_contract(bytes, model.tensors[id]);
    append_u32(bytes, static_cast<uint32_t>(op.states.size()));
    for (uint32_t id : op.states) append_state_contract(bytes, model.states[id]);
    append_effects(bytes, step.state_effects);
    return bytes;
}

std::vector<uint8_t> sampler_requirement_key(
    const SemanticDispatchRequest& request,
    const SemanticDispatchSamplerBinding& binding,
    const SemanticValue& logits) {
    std::vector<uint8_t> bytes;
    static constexpr std::array<uint8_t, 8> domain = {'L','P','S','A','M','P',1,0};
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    append_u8(bytes, static_cast<uint8_t>(SemanticDispatchStepKind::GreedySampler));
    append_u16(bytes, static_cast<uint16_t>(ExecutionPhase::Output));
    append_u32(bytes, request.batch_rows);
    append_u16(bytes, static_cast<uint16_t>(request.numerical_class));
    append_value_contract(bytes, logits);
    append_u32(bytes, binding.selected_row);
    append_u32(bytes, binding.vocabulary_size);
    return bytes;
}

void append_request(std::vector<uint8_t>& bytes, const SemanticDispatchRequest& request) {
    append_u16(bytes, static_cast<uint16_t>(request.phase));
    append_u32(bytes, request.batch_rows);
    append_u16(bytes, static_cast<uint16_t>(request.numerical_class));
    append_u8(bytes, request.include_speculative ? 1 : 0);
    append_u8(bytes, request.include_greedy_sampler ? 1 : 0);
}

void append_layer_views(std::vector<uint8_t>& bytes,
                        const std::vector<SemanticDispatchLayerView>& views) {
    append_u32(bytes, static_cast<uint32_t>(views.size()));
    for (const auto& view : views) {
        append_u32(bytes, view.layer_index);
        append_u32(bytes, view.first_step);
        append_u32(bytes, view.step_count);
        append_u32(bytes, view.flags);
    }
}

Sha256Digest program_digest(const SemanticDispatchProgram& program) {
    std::vector<uint8_t> bytes;
    bytes.reserve(64 + program.steps.size() * 80);
    append_digest(bytes, program.model_digest);
    append_request(bytes, program.request);
    append_vector(bytes, program.terminal_operator_ids);
    append_u8(bytes, program.output_binding ? 1 : 0);
    if (program.output_binding) {
        append_u32(bytes, program.output_binding->logits_value_id);
        append_u32(bytes, program.output_binding->selected_row);
        append_u32(bytes, program.output_binding->vocabulary_size);
        append_u32(bytes, program.output_binding->terminal_operator_id);
    }
    append_layer_views(bytes, program.layer_views);
    append_u32(bytes, static_cast<uint32_t>(program.steps.size()));
    for (const auto& step : program.steps) {
        append_u32(bytes, step.ordinal);
        append_u8(bytes, static_cast<uint8_t>(step.kind));
        append_u32(bytes, step.operator_id);
        append_u16(bytes, static_cast<uint16_t>(step.operation));
        append_u16(bytes, static_cast<uint16_t>(step.phase));
        append_u16(bytes, static_cast<uint16_t>(step.numerical_class));
        append_vector(bytes, step.covered_operator_ids);
        append_vector(bytes, step.input_values);
        append_vector(bytes, step.output_values);
        append_vector(bytes, step.tensor_ids);
        append_effects(bytes, step.state_effects);
        append_session_effects(bytes, step.session_effects);
        append_u8(bytes, step.sampler_binding ? 1 : 0);
        if (step.sampler_binding) {
            append_u32(bytes, step.sampler_binding->logits_value_id);
            append_u32(bytes, step.sampler_binding->selected_row);
            append_u32(bytes, step.sampler_binding->vocabulary_size);
            append_u32(bytes, step.sampler_binding->candidate_output_id);
        }
        append_u32(bytes, step.workspace_id);
        append_digest(bytes, step.requirement.identity);
        append_u32(bytes, step.requirement_index);
    }
    append_u32(bytes, static_cast<uint32_t>(program.requirements.size()));
    for (const auto& requirement : program.requirements) {
        append_u16(bytes, requirement.version);
        append_u8(bytes, static_cast<uint8_t>(requirement.step_kind));
        append_u16(bytes, static_cast<uint16_t>(requirement.operation));
        append_u16(bytes, requirement.semantic_version);
        append_u16(bytes, static_cast<uint16_t>(requirement.phase));
        append_u32(bytes, requirement.batch_rows);
        append_u16(bytes, static_cast<uint16_t>(requirement.numerical_class));
        append_digest(bytes, requirement.identity);
    }
    return digest_bytes(bytes);
}

bool build_unchecked(const SemanticModel& model, const SemanticDispatchRequest& request,
                     SemanticDispatchProgram& program, CompatibilityReport* report) {
    if (!request_valid(request, report)) return false;
    auto encoded = encode_semantic_model(model);
    if (const auto* error = std::get_if<CompatibilityReport>(&encoded)) {
        if (report) *report = *error;
        return false;
    }

    std::vector<bool> excluded;
    if (!derive_excluded_operators(model, request, excluded, report)) return false;
    std::vector<int64_t> producers;
    if (!validate_graph(model, excluded, producers, report)) return false;
    std::vector<uint32_t> terminals;
    if (!derive_execution_roots(model, excluded, producers, terminals, report)) return false;

    program = {};
    program.model_digest = semantic_model_digest(model);
    program.request = request;
    program.terminal_operator_ids = terminals;
    if (model.output_values_count == 1 &&
        model.output_values_first < model.values.size()) {
        const uint32_t logits_value_id = model.output_values_first;
        const SemanticValue& logits = model.values[logits_value_id];
        const int64_t terminal_index = producers[logits_value_id];
        if (logits.logical_type == ScalarType::F32 &&
            !logits.dimensions.empty() &&
            logits.dimensions.back() ==
                Dimension{DimensionKind::Constant, model.vocabulary_size} &&
            terminal_index >= 0 &&
            static_cast<size_t>(terminal_index) < model.operators.size()) {
            program.output_binding = SemanticDispatchOutputBinding{
                logits_value_id, request.batch_rows - 1,
                model.vocabulary_size,
                model.operators[static_cast<size_t>(terminal_index)].id};
        }
    }
    std::vector<int64_t> step_for_operator(model.operators.size(), -1);
    std::map<std::vector<uint8_t>, uint32_t> requirement_indices;
    for (uint32_t index = 0; index < model.operators.size(); ++index) {
        if (excluded_operator(excluded, index)) continue;
        const SemanticOperator& op = model.operators[index];
        SemanticDispatchStep step;
        step.ordinal = static_cast<uint32_t>(program.steps.size());
        step.operator_id = op.id;
        step.operation = op.kind;
        step.phase = request.phase;
        step.numerical_class = request.numerical_class;
        step.covered_operator_ids = {op.id};
        step.input_values = op.inputs;
        step.output_values = op.outputs;
        step.tensor_ids = op.tensors;
        if (!derive_state_effects(model, op, step.state_effects, report)) return false;
        step.workspace_id = step.ordinal + 1;
        step.requirement.step_kind = SemanticDispatchStepKind::Operator;
        step.requirement.operation = op.kind;
        step.requirement.semantic_version = op.semantic_version;
        step.requirement.phase = request.phase;
        step.requirement.batch_rows = request.batch_rows;
        step.requirement.numerical_class = request.numerical_class;
        std::vector<uint8_t> key = requirement_key(model, op, request, step);
        step.requirement.identity = digest_bytes(key);
        const auto requirement = requirement_indices.find(key);
        if (requirement == requirement_indices.end()) {
            step.requirement_index = static_cast<uint32_t>(program.requirements.size());
            requirement_indices.emplace(std::move(key), step.requirement_index);
            program.requirements.push_back(step.requirement);
        } else {
            step.requirement_index = requirement->second;
            step.requirement = program.requirements[step.requirement_index];
        }
        step_for_operator[index] = static_cast<int64_t>(program.steps.size());
        program.steps.push_back(std::move(step));
    }

    for (const SemanticLayer& layer : model.layers) {
        SemanticDispatchLayerView view;
        view.layer_index = layer.layer_index;
        view.flags = layer.flags;
        const bool speculative = (layer.flags & kSemanticLayerFlagSpeculative) != 0;
        if (speculative && !request.include_speculative) {
            view.first_step = kSemanticDispatchUnresolved;
            view.step_count = 0;
        } else if (layer.operator_count != 0) {
            if (layer.first_operator >= step_for_operator.size() ||
                step_for_operator[layer.first_operator] < 0 ||
                layer.operator_count > step_for_operator.size() - layer.first_operator) {
                return set_failure(report, failure(CompatibilityError::IR_REFERENCE_INVALID,
                                                   "layer view cannot address its dispatch steps"));
            }
            view.first_step = static_cast<uint32_t>(step_for_operator[layer.first_operator]);
            view.step_count = layer.operator_count;
            for (uint32_t offset = 0; offset < view.step_count; ++offset) {
                const int64_t step = step_for_operator[layer.first_operator + offset];
                if (step < 0 || step != static_cast<int64_t>(view.first_step + offset)) {
                    return set_failure(report, failure(CompatibilityError::IR_REFERENCE_INVALID,
                                                       "layer view is not a contiguous program view"));
                }
            }
        }
        program.layer_views.push_back(std::move(view));
    }

    if (request.include_greedy_sampler) {
        if (!program.output_binding) {
            return set_failure(report, failure(CompatibilityError::RUNTIME_INPUT_INVALID,
                                               "greedy sampling requires one logits output"));
        }
        const SemanticValue& logits =
            model.values[program.output_binding->logits_value_id];
        SemanticDispatchStep sampler;
        sampler.ordinal = static_cast<uint32_t>(program.steps.size());
        sampler.kind = SemanticDispatchStepKind::GreedySampler;
        sampler.operation = kSemanticDispatchNoOperator;
        sampler.phase = ExecutionPhase::Output;
        sampler.numerical_class = request.numerical_class;
        for (uint32_t offset = 0; offset < model.output_values_count; ++offset) {
            sampler.input_values.push_back(model.output_values_first + offset);
        }
        sampler.workspace_id = sampler.ordinal + 1;
        sampler.sampler_binding = SemanticDispatchSamplerBinding{
            program.output_binding->logits_value_id,
            program.output_binding->selected_row,
            program.output_binding->vocabulary_size,
            sampler.workspace_id};
        sampler.session_effects = {
            {SemanticDispatchSessionEffectKind::PositionCandidateAdvance},
            {SemanticDispatchSessionEffectKind::TokenHistoryCandidateAppend},
            {SemanticDispatchSessionEffectKind::CandidateOutputWrite},
        };
        sampler.requirement.step_kind = SemanticDispatchStepKind::GreedySampler;
        sampler.requirement.operation = sampler.operation;
        sampler.requirement.semantic_version = 1;
        sampler.requirement.phase = sampler.phase;
        sampler.requirement.batch_rows = request.batch_rows;
        sampler.requirement.numerical_class = sampler.numerical_class;
        std::vector<uint8_t> key = sampler_requirement_key(
            request, *sampler.sampler_binding, logits);
        sampler.requirement.identity = digest_bytes(key);
        const auto requirement = requirement_indices.find(key);
        if (requirement == requirement_indices.end()) {
            sampler.requirement_index = static_cast<uint32_t>(program.requirements.size());
            requirement_indices.emplace(std::move(key), sampler.requirement_index);
            program.requirements.push_back(sampler.requirement);
        } else {
            sampler.requirement_index = requirement->second;
            sampler.requirement = program.requirements[sampler.requirement_index];
        }
        program.steps.push_back(std::move(sampler));
    }
    program.program_digest = program_digest(program);
    return true;
}

} // namespace

SemanticDispatchProgramResult build_semantic_dispatch_program(
    const SemanticModel& model, const SemanticDispatchRequest& request) {
    try {
        SemanticDispatchProgram program;
        CompatibilityReport report;
        if (!build_unchecked(model, request, program, &report)) return report;
        return program;
    } catch (const std::bad_alloc&) {
        return failure(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                       "semantic dispatch program allocation exceeded its bound");
    }
}

bool validate_semantic_dispatch_program(
    const SemanticModel& model, const SemanticDispatchRequest& request,
    const SemanticDispatchProgram& program, CompatibilityReport* report) {
    try {
        SemanticDispatchProgram expected;
        if (!build_unchecked(model, request, expected, report)) return false;
        if (!(expected == program)) {
            return set_failure(report, failure(CompatibilityError::IR_REFERENCE_INVALID,
                                               "semantic dispatch program does not match the graph"));
        }
        return true;
    } catch (const std::bad_alloc&) {
        return set_failure(report, failure(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                                           "semantic dispatch validation allocation exceeded its bound"));
    }
}

} // namespace Laplace
