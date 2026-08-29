#include "source_compiler_graph_proof.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <deque>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace Laplace {
namespace {

constexpr uint32_t kUnassigned = std::numeric_limits<uint32_t>::max();

CompatibilityReport graph_error(CompatibilityError code, std::string detail,
                                uint32_t operator_id = kUnassigned,
                                uint32_t tensor_id = kUnassigned,
                                uint32_t state_id = kUnassigned) {
    CompatibilityReport report = compatibility_report(code, std::move(detail));
    report.operator_id = operator_id;
    report.tensor_id = tensor_id;
    report.state_id = state_id;
    return report;
}

template <typename Record>
bool dense_ids(const std::vector<Record>& records) {
    for (uint32_t id = 0; id != records.size(); ++id) {
        if (records[id].id != id) return false;
    }
    return true;
}

bool in_range(uint32_t id, size_t size) {
    return id < size;
}

bool v1_norm_role(TensorRole role) {
    return role == TensorRole::AttentionNormWeight ||
           role == TensorRole::FfnNormWeight || role == TensorRole::FinalNormWeight;
}

bool v1_linear_weight_role(TensorRole role) {
    switch (role) {
    case TensorRole::QueryWeight:
    case TensorRole::KeyWeight:
    case TensorRole::ValueWeight:
    case TensorRole::AttentionOutputWeight:
    case TensorRole::FfnGateWeight:
    case TensorRole::FfnUpWeight:
    case TensorRole::FfnDownWeight:
    case TensorRole::OutputWeight:
        return true;
    default:
        return false;
    }
}

bool v1_linear_bias_role(TensorRole role) {
    return role == TensorRole::QueryBias || role == TensorRole::KeyBias ||
           role == TensorRole::ValueBias;
}

bool v1_operator_tensor_role_valid(const SemanticOperator& op, size_t tensor_index,
                                   TensorRole role) {
    if (op.semantic_version != 1) return true;
    switch (op.kind) {
    case OperatorKind::EmbeddingLookup:
        return tensor_index == 0 && role == TensorRole::TokenEmbedding;
    case OperatorKind::RmsNorm:
        return tensor_index == 0 && v1_norm_role(role);
    case OperatorKind::Linear:
        return tensor_index == 0 ? v1_linear_weight_role(role)
                                 : tensor_index == 1 && v1_linear_bias_role(role);
    default:
        return false;
    }
}

struct StructuralDigestBuilder {
    std::vector<uint8_t> bytes;

    void u8(uint8_t value) { bytes.push_back(value); }

    void u16(uint16_t value) {
        bytes.push_back(static_cast<uint8_t>(value));
        bytes.push_back(static_cast<uint8_t>(value >> 8));
    }

    void u32(uint32_t value) {
        for (unsigned shift = 0; shift != 32; shift += 8) {
            bytes.push_back(static_cast<uint8_t>(value >> shift));
        }
    }

    void u64(uint64_t value) {
        for (unsigned shift = 0; shift != 64; shift += 8) {
            bytes.push_back(static_cast<uint8_t>(value >> shift));
        }
    }

    void dimensions(const std::vector<Dimension>& dimensions) {
        u32(static_cast<uint32_t>(dimensions.size()));
        for (const Dimension& dimension : dimensions) {
            u8(static_cast<uint8_t>(dimension.kind));
            u64(dimension.constant_or_symbol);
        }
    }

    template <size_t N>
    void fixed_bytes(const std::array<uint8_t, N>& value) {
        bytes.insert(bytes.end(), value.begin(), value.end());
    }

    void payload(const SemanticOperator& op) {
        u32(static_cast<uint32_t>(op.payload.index()));
        std::visit(
            [this](const auto& value) {
                using Payload = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<Payload, EmbeddingLookupPayload>) {
                    u32(value.scale_f32_bits);
                    u32(value.vocabulary);
                    u32(value.width);
                    u32(value.flags);
                } else if constexpr (std::is_same_v<Payload, RmsNormPayload>) {
                    u32(value.epsilon_f32_bits);
                    u32(static_cast<uint32_t>(value.axis));
                    u8(value.weight_mode);
                } else if constexpr (std::is_same_v<Payload, LinearPayload>) {
                    u8(value.transpose_weight ? 1 : 0);
                    u8(value.has_bias ? 1 : 0);
                    u16(static_cast<uint16_t>(value.accumulation_type));
                } else if constexpr (std::is_same_v<Payload, RopePayload>) {
                    u8(static_cast<uint8_t>(value.pairing));
                    u8(value.position_from_cursor ? 1 : 0);
                    u32(value.rotary_dimension);
                    u32(value.base_f32_bits);
                    u32(value.scale_f32_bits);
                    for (uint32_t section : value.position_sections) u32(section);
                    u32(value.frequency_dimension);
                } else if constexpr (std::is_same_v<Payload, CausalAttentionPayload>) {
                    u32(value.query_heads);
                    u32(value.kv_heads);
                    u32(value.head_dimension);
                    u32(value.scale_f32_bits);
                    u8(static_cast<uint8_t>(value.mask));
                    u8(static_cast<uint8_t>(value.cache_policy));
                    u8(static_cast<uint8_t>(value.window));
                    u8(static_cast<uint8_t>(value.value_source));
                    u32(value.window_tokens);
                    u32(value.value_source_value);
                } else if constexpr (std::is_same_v<Payload, SwiGluPayload>) {
                    u16(static_cast<uint16_t>(value.activation));
                } else if constexpr (std::is_same_v<Payload, AddPayload>) {
                    // No payload fields.
                } else if constexpr (std::is_same_v<Payload, DepthwiseConvSiluPayload>) {
                    u32(value.qk_heads);
                    u32(value.value_heads);
                    u32(value.head_dimension);
                    u32(value.kernel);
                } else if constexpr (std::is_same_v<Payload, GatedDeltaNetPayload>) {
                    u32(value.qk_heads);
                    u32(value.value_heads);
                    u32(value.head_dimension);
                    u8(static_cast<uint8_t>(value.qk_mapping));
                    u8(static_cast<uint8_t>(value.beta_transform));
                    u8(static_cast<uint8_t>(value.decay_transform));
                    u8(static_cast<uint8_t>(value.state_layout));
                    u32(value.flags);
                } else if constexpr (std::is_same_v<Payload, GatedAttentionPayload>) {
                    // No payload fields.
                } else if constexpr (std::is_same_v<Payload, GatedRmsNormPayload>) {
                    u32(value.epsilon_f32_bits);
                    u16(static_cast<uint16_t>(value.gate_activation));
                    u8(value.weight_mode);
                } else if constexpr (std::is_same_v<Payload, L2NormalizePayload>) {
                    u32(value.epsilon_f32_bits);
                } else if constexpr (std::is_same_v<Payload, AxisSplitPayload>) {
                    u32(value.first_width);
                    u32(value.second_width);
                } else if constexpr (std::is_same_v<Payload, ConcatPayload>) {
                    u32(static_cast<uint32_t>(value.axis));
                } else if constexpr (std::is_same_v<Payload, RouterTopKPayload>) {
                    u32(value.expert_count);
                    u32(value.selected_count);
                    u8(static_cast<uint8_t>(value.score_domain));
                    u8(static_cast<uint8_t>(value.normalization_order));
                    u8(static_cast<uint8_t>(value.selected_weight_normalization));
                    u8(static_cast<uint8_t>(value.tie_policy));
                    u8(static_cast<uint8_t>(value.weight_source));
                    u32(value.flags);
                } else if constexpr (std::is_same_v<Payload, RoutedLinearPayload>) {
                    u16(static_cast<uint16_t>(value.accumulation_type));
                } else if constexpr (std::is_same_v<Payload, GatedActivationPayload>) {
                    u16(static_cast<uint16_t>(value.activation));
                } else if constexpr (std::is_same_v<Payload, WeightedExpertReducePayload>) {
                    u8(static_cast<uint8_t>(value.association));
                    u8(static_cast<uint8_t>(value.scale_source));
                    u16(static_cast<uint16_t>(value.accumulation_type));
                } else if constexpr (std::is_same_v<Payload, ScalePayload>) {
                    u8(static_cast<uint8_t>(value.source));
                    u32(value.literal_f32_bits);
                } else if constexpr (std::is_same_v<Payload, TanhSoftcapPayload>) {
                    u32(value.cap_f32_bits);
                }
            },
            op.payload);
    }

    Sha256Digest finish() const {
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
};

Sha256Digest structural_graph_digest(const SemanticModel& model) {
    StructuralDigestBuilder builder;
    constexpr char domain[] = "laplace-source-compiler-graph-v1";
    builder.bytes.insert(builder.bytes.end(), domain, domain + sizeof(domain) - 1);
    builder.u16(model.schema_major);
    builder.u16(model.schema_minor);
    builder.u16(model.opset_major);
    builder.u16(model.opset_minor);
    builder.u32(model.maximum_context);
    builder.u32(model.vocabulary_size);
    builder.u32(model.bos_id);
    builder.u32(model.eos_id);
    builder.u32(static_cast<uint32_t>(model.stop_ids.size()));
    for (uint32_t stop_id : model.stop_ids) builder.u32(stop_id);
    builder.fixed_bytes(model.tokenizer_digest);
    builder.fixed_bytes(model.template_digest);
    builder.u16(static_cast<uint16_t>(model.entry_kind));
    builder.u32(model.input_values_first);
    builder.u32(model.input_values_count);
    builder.u32(model.output_values_first);
    builder.u32(model.output_values_count);

    builder.u32(static_cast<uint32_t>(model.tensors.size()));
    for (const SemanticTensor& tensor : model.tensors) {
        builder.u32(tensor.id);
        builder.u16(static_cast<uint16_t>(tensor.role));
        builder.u16(static_cast<uint16_t>(tensor.logical_type));
        builder.u16(tensor.flags);
        builder.dimensions(tensor.dimensions);
        builder.u16(static_cast<uint16_t>(tensor.layout.kind));
        builder.u16(tensor.layout.version);
        builder.u16(static_cast<uint16_t>(tensor.layout.packing));
        builder.u8(tensor.layout.rank);
        builder.u8(tensor.layout.block_rank);
        for (uint8_t axis : tensor.layout.axis_order) builder.u8(axis);
        for (uint64_t stride : tensor.layout.strides) builder.u64(stride);
        builder.u32(tensor.layout.block_elements);
        builder.u32(tensor.layout.block_bytes);
        builder.u32(tensor.layout.flags);
        builder.u16(static_cast<uint16_t>(tensor.quantization.kind));
        builder.u16(tensor.quantization.version);
        builder.u16(static_cast<uint16_t>(tensor.quantization.accumulation_type));
        builder.u16(static_cast<uint16_t>(tensor.quantization.scale_type));
        builder.u16(static_cast<uint16_t>(tensor.quantization.zero_type));
        builder.u16(static_cast<uint16_t>(tensor.quantization.bias_type));
        builder.u32(tensor.quantization.block_elements);
        builder.u32(tensor.quantization.block_bytes);
        builder.u32(tensor.quantization.group_size);
        builder.u32(tensor.quantization.required_plane_mask);
        builder.u32(tensor.quantization.flags);
        builder.u8(static_cast<uint8_t>(tensor.expert_axis.kind));
        builder.u8(tensor.expert_axis.expert_axis);
        builder.u8(tensor.expert_axis.member_axis);
        builder.u8(tensor.expert_axis.input_axis);
        builder.u8(tensor.expert_axis.output_axis);
        builder.u32(tensor.expert_axis.expert_count);
        builder.u64(tensor.expert_axis.per_expert_byte_stride);
        builder.u32(tensor.expert_axis.flags);
        builder.u32(static_cast<uint32_t>(tensor.planes.size()));
        for (const TensorPlane& plane : tensor.planes) {
            builder.u16(static_cast<uint16_t>(plane.kind));
            builder.u16(static_cast<uint16_t>(plane.storage_type));
            // Artifact identity and absolute offsets are authenticated by the
            // manifest's physical-binding digest. The source graph proof binds
            // only the execution-relevant plane structure.
            builder.u64(plane.length);
            builder.u32(plane.alignment);
            builder.u32(plane.flags);
        }
    }

    builder.u32(static_cast<uint32_t>(model.values.size()));
    for (const SemanticValue& value : model.values) {
        builder.u32(value.id);
        builder.u16(static_cast<uint16_t>(value.logical_type));
        builder.u8(value.flags);
        builder.dimensions(value.dimensions);
    }

    builder.u32(static_cast<uint32_t>(model.operators.size()));
    for (const SemanticOperator& op : model.operators) {
        builder.u32(op.id);
        builder.u16(static_cast<uint16_t>(op.kind));
        builder.u16(op.semantic_version);
        const auto append_ids = [&builder](const std::vector<uint32_t>& ids) {
            builder.u32(static_cast<uint32_t>(ids.size()));
            for (uint32_t id : ids) builder.u32(id);
        };
        append_ids(op.inputs);
        append_ids(op.outputs);
        append_ids(op.tensors);
        append_ids(op.states);
        builder.payload(op);
    }

    builder.u32(static_cast<uint32_t>(model.layers.size()));
    for (const SemanticLayer& layer : model.layers) {
        builder.u32(layer.layer_index);
        builder.u32(layer.first_operator);
        builder.u32(layer.operator_count);
        builder.u32(layer.flags);
    }

    builder.u32(static_cast<uint32_t>(model.states.size()));
    for (const SemanticState& state : model.states) {
        builder.u32(state.id);
        builder.u16(static_cast<uint16_t>(state.kind));
        builder.u16(state.semantic_version);
        builder.u16(static_cast<uint16_t>(state.update_kind));
        builder.u16(static_cast<uint16_t>(state.position_policy));
        builder.u16(state.flags);
        builder.dimensions(state.dimensions);
        builder.u32(static_cast<uint32_t>(state.formats.size()));
        for (const StateFormat& format : state.formats) {
            builder.u16(static_cast<uint16_t>(format.kind));
            builder.u16(format.version);
            builder.u16(static_cast<uint16_t>(format.logical_type));
            builder.u16(static_cast<uint16_t>(format.encoded_type));
            builder.u16(static_cast<uint16_t>(format.logical_domain));
            builder.u16(static_cast<uint16_t>(format.encoded_domain));
            builder.u16(static_cast<uint16_t>(format.codec));
            builder.u16(static_cast<uint16_t>(format.cache_policy));
            builder.u16(static_cast<uint16_t>(format.layout_policy));
            builder.u16(format.flags);
            builder.u32(format.tile_tokens);
            builder.u32(format.mutable_tokens);
            builder.u32(format.alignment);
            builder.u32(format.reserved);
        }
    }

    builder.u32(static_cast<uint32_t>(model.constraints.size()));
    for (const SemanticConstraint& constraint : model.constraints) {
        builder.u16(static_cast<uint16_t>(constraint.kind));
        builder.u8(static_cast<uint8_t>(constraint.lhs_kind));
        builder.u8(static_cast<uint8_t>(constraint.rhs_kind));
        builder.u32(constraint.lhs_id);
        builder.u32(constraint.rhs_id);
        builder.u32(constraint.lhs_axis);
        builder.u32(constraint.rhs_axis);
        builder.u64(constraint.constant);
        builder.u64(constraint.divisor);
    }

    builder.u32(static_cast<uint32_t>(model.capabilities.size()));
    for (const CapabilityRequirement& capability : model.capabilities) {
        builder.u16(static_cast<uint16_t>(capability.capability));
        builder.u16(capability.minimum_version);
        builder.u32(capability.flags);
    }
    builder.u32(static_cast<uint32_t>(model.fallbacks.size()));
    for (const SemanticFallback& fallback : model.fallbacks) {
        builder.u16(static_cast<uint16_t>(fallback.kind));
        builder.u16(static_cast<uint16_t>(fallback.phase));
        builder.u16(static_cast<uint16_t>(fallback.numerical_class));
        builder.u16(fallback.flags);
    }
    return builder.finish();
}

} // namespace

SourceCompilerGraphProofResult prove_source_candidate_graph(const SemanticModel& model) {
    if (model.layers.size() > kSemanticModelMaximumLayers) {
        return graph_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                           "candidate graph layer count exceeds the semantic limit");
    }
    if (model.values.empty() || model.operators.empty()) {
        return graph_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                           "candidate graph has no values or operators");
    }
    if (model.input_values_count == 0 || model.input_values_first > model.values.size() ||
        model.input_values_count > model.values.size() - model.input_values_first ||
        model.output_values_count == 0 || model.output_values_first > model.values.size() ||
        model.output_values_count > model.values.size() - model.output_values_first) {
        return graph_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                           "candidate graph input or output range is invalid");
    }
    if (!dense_ids(model.values) || !dense_ids(model.tensors) ||
        !dense_ids(model.operators) || !dense_ids(model.states)) {
        return graph_error(CompatibilityError::IR_REFERENCE_INVALID,
                           "candidate graph identifiers are not dense");
    }

    // Validate the complete versioned operator contract before interpreting
    // references. A source proof must not authenticate a payload that the
    // semantic runtime would reject.
    for (const SemanticOperator& op : model.operators) {
        if (!semantic_operator_signature_valid(op)) {
            return graph_error(CompatibilityError::IR_REFERENCE_INVALID,
                               "operator signature has invalid arity", op.id);
        }
        if (!semantic_operator_contract_valid(op)) {
            return graph_error(CompatibilityError::IR_REFERENCE_INVALID,
                               "operator semantic contract is invalid", op.id);
        }
    }

    const size_t value_count = model.values.size();
    const size_t operator_count = model.operators.size();
    const size_t tensor_count = model.tensors.size();
    const size_t state_count = model.states.size();
    std::vector<bool> is_input(value_count, false);
    std::vector<bool> is_output(value_count, false);
    for (uint32_t id = model.input_values_first;
         id != model.input_values_first + model.input_values_count; ++id) {
        is_input[id] = true;
    }
    for (uint32_t id = model.output_values_first;
         id != model.output_values_first + model.output_values_count; ++id) {
        is_output[id] = true;
        if (is_input[id]) {
            return graph_error(CompatibilityError::IR_REFERENCE_INVALID,
                               "candidate graph logits overlap graph input");
        }
    }

    std::vector<uint32_t> producer(value_count, kUnassigned);
    std::vector<uint32_t> value_uses(value_count, 0);
    std::vector<bool> tensor_used(tensor_count, false);
    std::vector<uint32_t> state_uses(state_count, 0);

    for (uint32_t operator_id = 0; operator_id != operator_count; ++operator_id) {
        const SemanticOperator& op = model.operators[operator_id];
        for (uint32_t value_id : op.outputs) {
            if (!in_range(value_id, value_count)) {
                return graph_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "operator output value is out of range", operator_id);
            }
            if (is_input[value_id]) {
                return graph_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "operator output overwrites graph input", operator_id);
            }
            if (producer[value_id] != kUnassigned) {
                return graph_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "duplicate producer for value", operator_id);
            }
            producer[value_id] = operator_id;
        }
        for (uint32_t tensor_id : op.tensors) {
            if (!in_range(tensor_id, tensor_count)) {
                return graph_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "operator tensor reference is out of range", operator_id);
            }
            tensor_used[tensor_id] = true;
        }
        for (size_t tensor_index = 0; tensor_index != op.tensors.size(); ++tensor_index) {
            const uint32_t tensor_id = op.tensors[tensor_index];
            if (!v1_operator_tensor_role_valid(op, tensor_index, model.tensors[tensor_id].role)) {
                return graph_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "operator tensor role mismatch", operator_id, tensor_id);
            }
        }
        for (uint32_t state_id : op.states) {
            if (!in_range(state_id, state_count)) {
                return graph_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "operator state reference is out of range", operator_id);
            }
            if (++state_uses[state_id] > 1) {
                return graph_error(
                    CompatibilityError::IR_STATE_INVALID,
                    "mutable state is shared without an explicit access/effect contract",
                    operator_id, kUnassigned, state_id);
            }
        }
    }

    for (uint32_t operator_id = 0; operator_id != operator_count; ++operator_id) {
        const SemanticOperator& op = model.operators[operator_id];
        for (uint32_t value_id : op.inputs) {
            if (!in_range(value_id, value_count)) {
                return graph_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "operator input value is out of range", operator_id);
            }
            ++value_uses[value_id];
            if (std::find(op.outputs.begin(), op.outputs.end(), value_id) != op.outputs.end()) {
                return graph_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "operator has a self-edge", operator_id);
            }
            if (!is_input[value_id] && producer[value_id] == kUnassigned) {
                return graph_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "unresolved input value", operator_id);
            }
        }
    }

    for (uint32_t value_id = 0; value_id != value_count; ++value_id) {
        if (!is_input[value_id] && producer[value_id] == kUnassigned) {
            return graph_error(CompatibilityError::IR_REFERENCE_INVALID,
                               "unresolved produced value");
        }
    }

    std::vector<uint32_t> indegree(operator_count, 0);
    std::vector<std::vector<uint32_t>> consumers(operator_count);
    for (uint32_t operator_id = 0; operator_id != operator_count; ++operator_id) {
        for (uint32_t value_id : model.operators[operator_id].inputs) {
            if (is_input[value_id]) continue;
            const uint32_t producer_id = producer[value_id];
            ++indegree[operator_id];
            consumers[producer_id].push_back(operator_id);
        }
    }

    std::deque<uint32_t> ready;
    for (uint32_t operator_id = 0; operator_id != operator_count; ++operator_id) {
        if (indegree[operator_id] == 0) ready.push_back(operator_id);
    }
    SourceCompilerGraphProof proof;
    proof.input_value_count = model.input_values_count;
    proof.output_value_count = model.output_values_count;
    proof.value_count = static_cast<uint32_t>(value_count);
    proof.operator_count = static_cast<uint32_t>(operator_count);
    proof.layer_count = static_cast<uint32_t>(model.layers.size());
    while (!ready.empty()) {
        const uint32_t operator_id = ready.front();
        ready.pop_front();
        proof.topological_operator_ids.push_back(operator_id);
        for (uint32_t consumer : consumers[operator_id]) {
            if (--indegree[consumer] == 0) ready.push_back(consumer);
        }
    }
    if (proof.topological_operator_ids.size() != operator_count) {
        return graph_error(CompatibilityError::IR_REFERENCE_INVALID,
                           "candidate graph contains an operator dependency cycle");
    }
    for (uint32_t operator_id = 0; operator_id != operator_count; ++operator_id) {
        for (uint32_t value_id : model.operators[operator_id].inputs) {
            if (!is_input[value_id] && producer[value_id] >= operator_id) {
                return graph_error(
                    CompatibilityError::IR_REFERENCE_INVALID,
                    "value dependency reverses the serialized operator program",
                    operator_id);
            }
        }
    }

    std::vector<bool> reachable_values(value_count, false);
    for (uint32_t id = model.input_values_first;
         id != model.input_values_first + model.input_values_count; ++id) {
        reachable_values[id] = true;
    }
    for (uint32_t operator_id : proof.topological_operator_ids) {
        const SemanticOperator& op = model.operators[operator_id];
        bool reachable = !op.inputs.empty();
        for (uint32_t value_id : op.inputs) reachable = reachable && reachable_values[value_id];
        if (reachable) {
            for (uint32_t value_id : op.outputs) reachable_values[value_id] = true;
        }
    }
    for (uint32_t value_id = model.output_values_first;
         value_id != model.output_values_first + model.output_values_count; ++value_id) {
        if (!reachable_values[value_id]) {
            return graph_error(CompatibilityError::IMPORT_CLOSURE_INCOMPLETE,
                               "unreachable output value");
        }
    }
    for (uint32_t value_id = 0; value_id != value_count; ++value_id) {
        if (is_input[value_id] && value_uses[value_id] == 0) {
            return graph_error(CompatibilityError::IMPORT_CLOSURE_INCOMPLETE,
                               "unconsumed graph input");
        }
        if (!is_input[value_id] && !is_output[value_id] && value_uses[value_id] == 0) {
            return graph_error(CompatibilityError::IMPORT_CLOSURE_INCOMPLETE,
                               "dead produced value " + std::to_string(value_id));
        }
    }

    // Every admitted operator must contribute to a requested graph output.
    // State-only side effects remain unsupported until the semantic IR carries
    // explicit state access and effect edges.
    std::vector<bool> output_closure(operator_count, false);
    std::deque<uint32_t> closure;
    for (uint32_t value_id = model.output_values_first;
         value_id != model.output_values_first + model.output_values_count; ++value_id) {
        const uint32_t producer_id = producer[value_id];
        if (!output_closure[producer_id]) {
            output_closure[producer_id] = true;
            closure.push_back(producer_id);
        }
    }
    while (!closure.empty()) {
        const uint32_t operator_id = closure.front();
        closure.pop_front();
        for (uint32_t value_id : model.operators[operator_id].inputs) {
            if (is_input[value_id]) continue;
            const uint32_t producer_id = producer[value_id];
            if (!output_closure[producer_id]) {
                output_closure[producer_id] = true;
                closure.push_back(producer_id);
            }
        }
    }
    if (std::find(output_closure.begin(), output_closure.end(), false) !=
        output_closure.end()) {
        return graph_error(CompatibilityError::IMPORT_CLOSURE_INCOMPLETE,
                           "operator is outside the output closure");
    }

    if (model.layers.empty()) {
        return graph_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                           "layer coverage is empty");
    }
    uint32_t operator_cursor = 0;
    for (size_t index = 0; index != model.layers.size(); ++index) {
        const SemanticLayer& layer = model.layers[index];
        if (layer.layer_index != index || layer.operator_count == 0 ||
            layer.first_operator < operator_cursor ||
            layer.first_operator > operator_count ||
            layer.operator_count > operator_count - layer.first_operator) {
            return graph_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                               "layer coverage overlaps, is unordered, or is out of range");
        }
        operator_cursor = layer.first_operator + layer.operator_count;
    }

    for (uint32_t tensor_id = 0; tensor_id != tensor_count; ++tensor_id) {
        const bool inactive =
            (model.tensors[tensor_id].flags & kSemanticTensorFlagInactiveProgram) != 0;
        if (inactive && tensor_used[tensor_id]) {
            return graph_error(CompatibilityError::IR_REFERENCE_INVALID,
                               "inactive-program tensor is referenced", kUnassigned,
                               tensor_id);
        }
        if (!inactive && !tensor_used[tensor_id]) {
            return graph_error(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                               "unconsumed tensor", kUnassigned, tensor_id);
        }
    }
    for (uint32_t state_id = 0; state_id != state_count; ++state_id) {
        if (state_uses[state_id] == 0) {
            return graph_error(CompatibilityError::IR_STATE_INVALID,
                               "unconsumed state", kUnassigned, kUnassigned, state_id);
        }
    }
    proof.structural_digest = structural_graph_digest(model);
    return proof;
}

bool source_compiler_graph_proof_matches(const SemanticModel& model,
                                         const SourceCompilerGraphProof& proof) {
    auto result = prove_source_candidate_graph(model);
    const auto* candidate = std::get_if<SourceCompilerGraphProof>(&result);
    return candidate != nullptr && candidate->topological_operator_ids == proof.topological_operator_ids &&
           candidate->structural_digest == proof.structural_digest &&
           candidate->input_value_count == proof.input_value_count &&
           candidate->output_value_count == proof.output_value_count &&
           candidate->value_count == proof.value_count &&
           candidate->operator_count == proof.operator_count &&
           candidate->layer_count == proof.layer_count;
}

} // namespace Laplace
