#include "scalar_executor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>

#include "fp16.h"

namespace Laplace {

namespace {

float fp32(float value) {
    volatile float rounded = value;
    return rounded;
}

float fp_add(float left, float right) { return fp32(fp32(left) + fp32(right)); }
float fp_mul(float left, float right) { return fp32(fp32(left) * fp32(right)); }

CompatibilityReport scalar_error(CompatibilityError code) {
    CompatibilityReport report = package_report(code);
    report.stage = CompatibilityStage::Semantic;
    return report;
}

bool dimensions(const SemanticTensor& tensor, std::vector<uint64_t>& result) {
    if (tensor.dimensions.empty() || tensor.dimensions.size() > 2) return false;
    result.clear();
    result.reserve(tensor.dimensions.size());
    for (const Dimension& dimension : tensor.dimensions) {
        if (dimension.kind != DimensionKind::Constant || dimension.constant_or_symbol == 0) return false;
        result.push_back(dimension.constant_or_symbol);
    }
    return true;
}

class TensorReader {
public:
    TensorReader(const RuntimePackage& package, const SemanticTensor& tensor) : tensor_(tensor) {
        if (tensor.planes.size() != 1 || tensor.planes[0].kind != PlaneKind::Values ||
            tensor.quantization.kind != QuantizationKind::None || !dimensions(tensor, dimensions_)) return;
        const TensorPlane& plane = tensor.planes[0];
        bytes_ = package.artifact_bytes(plane.artifact_id);
        element_bytes_ = plane.storage_type == ScalarType::F32 ? 4 : plane.storage_type == ScalarType::F16 ? 2 : 0;
        if (element_bytes_ == 0 || plane.offset > bytes_.size() || plane.length > bytes_.size() - plane.offset) return;
        uint64_t elements = 1;
        for (uint64_t dimension : dimensions_) {
            if (dimension > std::numeric_limits<uint64_t>::max() / elements) return;
            elements *= dimension;
        }
        if (elements > plane.length / element_bytes_) return;
        plane_ = &plane;
    }

    bool valid() const { return plane_ != nullptr; }
    uint64_t dimension(size_t index) const { return dimensions_.at(index); }
    size_t rank() const { return dimensions_.size(); }

    bool value(std::initializer_list<uint64_t> logical_index, float& output) const {
        if (!valid() || logical_index.size() != dimensions_.size()) return false;
        uint64_t physical_index = 0;
        uint64_t stride = 1;
        for (size_t physical_axis = 0; physical_axis != dimensions_.size(); ++physical_axis) {
            const uint8_t logical_axis = tensor_.layout.axis_order[physical_axis];
            if (logical_axis >= dimensions_.size() || logical_index.begin()[logical_axis] >= dimensions_[logical_axis]) return false;
            physical_index += logical_index.begin()[logical_axis] * stride;
            if (dimensions_[logical_axis] > std::numeric_limits<uint64_t>::max() / stride) return false;
            stride *= dimensions_[logical_axis];
        }
        const uint64_t byte_offset = plane_->offset + physical_index * element_bytes_;
        if (byte_offset > bytes_.size() || element_bytes_ > bytes_.size() - byte_offset) return false;
        if (plane_->storage_type == ScalarType::F32) {
            std::memcpy(&output, bytes_.data() + byte_offset, sizeof(output));
        } else {
            uint16_t half = 0;
            std::memcpy(&half, bytes_.data() + byte_offset, sizeof(half));
            output = fp16_to_fp32(half);
        }
        return std::isfinite(output);
    }

private:
    const SemanticTensor& tensor_;
    const TensorPlane* plane_ = nullptr;
    std::span<const uint8_t> bytes_;
    std::vector<uint64_t> dimensions_;
    uint64_t element_bytes_ = 0;
};

bool f32_from_bits(uint32_t bits, float& value) {
    std::memcpy(&value, &bits, sizeof(value));
    return std::isfinite(value);
}

bool linear(const RuntimePackage& package, const SemanticModel& model, const SemanticOperator& op,
            const std::vector<float>& input, std::vector<float>& output) {
    if (op.tensors.empty() || op.tensors.size() > 2) return false;
    const auto* payload = std::get_if<LinearPayload>(&op.payload);
    if (!payload || payload->accumulation_type != ScalarType::F32 || op.tensors.size() != (payload->has_bias ? 2u : 1u) ||
        op.tensors[0] >= model.tensors.size()) return false;
    TensorReader weight(package, model.tensors[op.tensors[0]]);
    if (!weight.valid() || weight.rank() != 2) return false;
    const uint64_t output_width = weight.dimension(0);
    const uint64_t input_width = weight.dimension(1);
    if (input_width == 0 || input.size() % input_width != 0 ||
        output_width > std::numeric_limits<size_t>::max() / (input.size() / input_width)) return false;
    std::unique_ptr<TensorReader> bias;
    if (payload->has_bias) {
        if (op.tensors[1] >= model.tensors.size()) return false;
        bias = std::make_unique<TensorReader>(package, model.tensors[op.tensors[1]]);
        if (!bias->valid() || bias->rank() != 1 || bias->dimension(0) != output_width) return false;
    }
    output.assign((input.size() / input_width) * output_width, 0.0f);
    for (size_t row = 0; row != input.size() / input_width; ++row) {
        for (uint64_t out = 0; out != output_width; ++out) {
            float lanes[16] = {};
            uint64_t in = 0;
            for (; in + 16 <= input_width; in += 16) {
                for (uint64_t lane = 0; lane != 16; ++lane) {
                    float weight_value = 0.0f;
                    if (!weight.value({out, in + lane}, weight_value)) return false;
                    lanes[lane] = fp_add(lanes[lane], fp_mul(input[row * input_width + in + lane], weight_value));
                }
            }
            float accumulator = 0.0f;
            for (float lane : lanes) accumulator = fp_add(accumulator, lane);
            for (; in != input_width; ++in) {
                float weight_value = 0.0f;
                if (!weight.value({out, in}, weight_value)) return false;
                accumulator = fp_add(accumulator, fp_mul(input[row * input_width + in], weight_value));
            }
            if (payload->has_bias) {
                float bias_value = 0.0f;
                if (!bias->value({out}, bias_value)) return false;
                accumulator = fp_add(accumulator, bias_value);
            }
            if (!std::isfinite(accumulator)) return false;
            output[row * output_width + out] = accumulator;
        }
    }
    return true;
}

bool rms_norm(const RuntimePackage& package, const SemanticModel& model, const SemanticOperator& op,
              const std::vector<float>& input, std::vector<float>& output) {
    if (op.tensors.size() != 1) return false;
    const auto* payload = std::get_if<RmsNormPayload>(&op.payload);
    if (!payload || payload->axis != -1 || payload->weight_mode != 1 || op.tensors[0] >= model.tensors.size()) return false;
    float epsilon = 0.0f;
    if (!f32_from_bits(payload->epsilon_f32_bits, epsilon) || epsilon < 0) return false;
    TensorReader weight(package, model.tensors[op.tensors[0]]);
    if (!weight.valid() || weight.rank() != 1 || weight.dimension(0) == 0) return false;
    const uint64_t width = payload->affine_geometry == RmsNormAffineGeometry::FullWidth
        ? weight.dimension(0) : payload->reduction_extent;
    if ((payload->affine_geometry == RmsNormAffineGeometry::FullWidth && payload->reduction_extent != 0) ||
        (payload->affine_geometry == RmsNormAffineGeometry::SharedAcrossGroups &&
         (payload->reduction_extent == 0 || weight.dimension(0) != payload->reduction_extent)) ||
        (payload->affine_geometry != RmsNormAffineGeometry::FullWidth &&
         payload->affine_geometry != RmsNormAffineGeometry::SharedAcrossGroups) ||
        input.size() % width != 0) return false;
    output.resize(input.size());
    for (size_t row = 0; row != input.size() / width; ++row) {
        float lanes[16] = {};
        uint64_t column = 0;
        for (; column + 16 <= width; column += 16) {
            for (uint64_t lane = 0; lane != 16; ++lane) {
                lanes[lane] = fp_add(lanes[lane], fp_mul(input[row * width + column + lane], input[row * width + column + lane]));
            }
        }
        float sum = 0.0f;
        for (float lane : lanes) sum = fp_add(sum, lane);
        for (; column != width; ++column) sum = fp_add(sum, fp_mul(input[row * width + column], input[row * width + column]));
        const float reciprocal = fp32(1.0f / std::sqrt(fp_add(fp32(sum / static_cast<float>(width)), epsilon)));
        if (!std::isfinite(reciprocal)) return false;
        for (uint64_t column = 0; column != width; ++column) {
            float scale = 0.0f;
            if (!weight.value({column}, scale)) return false;
            output[row * width + column] = fp_mul(fp_mul(input[row * width + column], reciprocal), scale);
        }
    }
    return true;
}

bool embedding(const RuntimePackage& package, const SemanticModel& model, const SemanticOperator& op,
               std::span<const uint32_t> tokens, std::vector<float>& output) {
    if (op.tensors.size() != 1) return false;
    const auto* payload = std::get_if<EmbeddingLookupPayload>(&op.payload);
    if (!payload || op.tensors[0] >= model.tensors.size()) return false;
    float scale = 0.0f;
    TensorReader table(package, model.tensors[op.tensors[0]]);
    if (!f32_from_bits(payload->scale_f32_bits, scale) || scale <= 0 || !table.valid() || table.rank() != 2 ||
        table.dimension(0) != payload->vocabulary || table.dimension(1) != payload->width) return false;
    output.resize(static_cast<size_t>(payload->width) * tokens.size());
    for (size_t row = 0; row != tokens.size(); ++row) {
        if (tokens[row] >= payload->vocabulary) return false;
        for (uint64_t column = 0; column != payload->width; ++column) {
            float value = 0.0f;
            if (!table.value({tokens[row], column}, value)) return false;
            output[row * payload->width + column] = fp_mul(value, scale);
        }
    }
    return true;
}

bool rope(const SemanticOperator& op, const std::vector<float>& query, const std::vector<float>& key,
          uint32_t first_position, uint32_t rows, uint32_t head_dimension,
          std::vector<float>& query_output, std::vector<float>& key_output) {
    const auto* payload = std::get_if<RopePayload>(&op.payload);
    if (!payload || payload->pairing != RopePairing::HalfSplit || !payload->position_from_cursor ||
        payload->rotary_dimension == 0 || payload->rotary_dimension % 2 != 0 || rows == 0 || head_dimension == 0 ||
        query.size() % head_dimension != 0 || key.size() % head_dimension != 0 || payload->rotary_dimension > head_dimension) return false;
    float base = 0.0f, scale = 0.0f;
    if (!f32_from_bits(payload->base_f32_bits, base) || !f32_from_bits(payload->scale_f32_bits, scale) || base <= 0 || scale <= 0) return false;
    auto rotate = [&](const std::vector<float>& input, std::vector<float>& output) {
        const size_t heads = input.size() / head_dimension / rows;
        if (heads == 0 || heads * rows * head_dimension != input.size()) return false;
        output = input;
        for (uint32_t row = 0; row != rows; ++row) {
            for (size_t head = 0; head != heads; ++head) {
                const size_t base_index = (static_cast<size_t>(row) * heads + head) * head_dimension;
                for (uint32_t pair = 0; pair != payload->rotary_dimension / 2; ++pair) {
                    const float exponent = fp32(-2.0f * static_cast<float>(pair) / static_cast<float>(payload->rotary_dimension));
                    const float angle = fp_mul(fp_mul(static_cast<float>(first_position + row), scale), std::pow(base, exponent));
                    const float cosine = fp32(std::cos(angle));
                    const float sine = fp32(std::sin(angle));
                    const float first = input[base_index + pair];
                    const float second = input[base_index + pair + payload->rotary_dimension / 2];
                    output[base_index + pair] = fp32(fp_mul(first, cosine) - fp_mul(second, sine));
                    output[base_index + pair + payload->rotary_dimension / 2] = fp_add(fp_mul(second, cosine), fp_mul(first, sine));
                }
            }
        }
        return true;
    };
    return rotate(query, query_output) && rotate(key, key_output);
}

bool rope_state_and_dimension(const SemanticModel& model, const SemanticOperator& rope_op,
                              size_t& state_slot, uint32_t& head_dimension) {
    for (const SemanticOperator& candidate : model.operators) {
        if (candidate.kind != OperatorKind::CausalAttention || candidate.inputs.size() < 2 || candidate.states.size() != 2 ||
            candidate.inputs[0] != rope_op.outputs[0] || candidate.inputs[1] != rope_op.outputs[1]) continue;
        const auto* attention = std::get_if<CausalAttentionPayload>(&candidate.payload);
        if (!attention || candidate.states[1] <= candidate.states[0]) return false;
        for (size_t index = 0; index + 1 < model.states.size(); index += 2) {
            if (model.states[index].id == candidate.states[0] && model.states[index + 1].id == candidate.states[1]) {
                state_slot = index / 2;
                head_dimension = attention->head_dimension;
                return true;
            }
        }
        return false;
    }
    return false;
}

KernelImplementation scalar_implementation(OperatorKind kind) {
    switch (kind) {
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
        return KernelImplementation::Unavailable;
    }
    return KernelImplementation::Unavailable;
}

bool implementation_matches(OperatorKind kind, KernelImplementation implementation) {
    const KernelImplementation scalar = scalar_implementation(kind);
    if (scalar != KernelImplementation::Unavailable && implementation == scalar) return true;
#if defined(LAPLACE_TESTING)
    return kind == OperatorKind::EmbeddingLookup && implementation == KernelImplementation::ScalarEmbeddingUnitOffset;
#else
    return false;
#endif
}

bool execute_scalar_operator(const RuntimePackage& package, const SemanticModel& model,
                             const SemanticOperator& op, KernelImplementation implementation,
                             std::span<const uint32_t> token_ids,
                             std::vector<std::vector<float>>& values,
                             ScalarExecutionOutput& result,
                             std::vector<SemanticKvState>& states) {
    if (!implementation_matches(op.kind, implementation) || op.inputs.empty() || op.outputs.empty() ||
        op.inputs[0] >= values.size() || op.outputs[0] >= values.size()) return false;
    std::vector<float> output;
    switch (implementation) {
    case KernelImplementation::Unavailable:
        return false;
    case KernelImplementation::ScalarEmbedding:
#if defined(LAPLACE_TESTING)
    case KernelImplementation::ScalarEmbeddingUnitOffset:
#endif
        if (!embedding(package, model, op, token_ids, output)) return false;
#if defined(LAPLACE_TESTING)
        if (implementation == KernelImplementation::ScalarEmbeddingUnitOffset) {
            for (float& value : output) value = fp_add(value, 1.0f);
        }
#endif
        values[op.outputs[0]] = std::move(output);
        break;
    case KernelImplementation::ScalarRmsNorm:
        if (!rms_norm(package, model, op, values[op.inputs[0]], output)) return false;
        values[op.outputs[0]] = std::move(output);
        break;
    case KernelImplementation::ScalarLinear:
        if (!linear(package, model, op, values[op.inputs[0]], output)) return false;
        values[op.outputs[0]] = std::move(output);
        break;
    case KernelImplementation::ScalarRope: {
        if (op.inputs.size() != 2 || op.outputs.size() != 2 || op.inputs[1] >= values.size() || op.outputs[1] >= values.size()) return false;
        size_t slot = 0;
        uint32_t dimension = 0;
        if (!rope_state_and_dimension(model, op, slot, dimension) || slot >= states.size() ||
            !rope(op, values[op.inputs[0]], values[op.inputs[1]], states[slot].tokens,
                  static_cast<uint32_t>(token_ids.size()), dimension, values[op.outputs[0]], values[op.outputs[1]])) return false;
        break;
    }
    case KernelImplementation::ScalarCausalAttention: {
        if (op.inputs.size() != 3 || op.states.size() != 2) return false;
        const auto state = std::find_if(model.states.begin(), model.states.end(), [&](const SemanticState& value) {
            return value.id == op.states[0];
        });
        if (state == model.states.end()) return false;
        const size_t slot = static_cast<size_t>(std::distance(model.states.begin(), state)) / 2;
        const auto* payload = std::get_if<CausalAttentionPayload>(&op.payload);
        float scale = 0.0f;
        if (slot >= states.size() || !payload || !f32_from_bits(payload->scale_f32_bits, scale)) return false;
        auto attention = semantic_causal_attention(values[op.inputs[0]], values[op.inputs[1]], values[op.inputs[2]],
                                                   static_cast<uint32_t>(token_ids.size()), payload->query_heads,
                                                   payload->kv_heads, payload->head_dimension, scale, states[slot]);
        if (!std::holds_alternative<std::vector<float>>(attention)) return false;
        values[op.outputs[0]] = std::get<std::vector<float>>(std::move(attention));
        break;
    }
    case KernelImplementation::ScalarSwiGlu: {
        if (op.inputs.size() != 2) return false;
        auto activated = semantic_swiglu(values[op.inputs[0]], values[op.inputs[1]]);
        if (!std::holds_alternative<std::vector<float>>(activated)) return false;
        values[op.outputs[0]] = std::get<std::vector<float>>(std::move(activated));
        break;
    }
    case KernelImplementation::ScalarAdd: {
        if (op.inputs.size() != 2) return false;
        auto sum = semantic_add(values[op.inputs[0]], values[op.inputs[1]]);
        if (!std::holds_alternative<std::vector<float>>(sum)) return false;
        values[op.outputs[0]] = std::get<std::vector<float>>(std::move(sum));
        break;
    }
    case KernelImplementation::MetalDenseToken:
    case KernelImplementation::MetalRecurrentToken:
        return false;
    }
    if (op.kind == OperatorKind::Rope) {
        result.operator_outputs.push_back(values[op.outputs[0]]);
        result.operator_outputs.back().insert(result.operator_outputs.back().end(), values[op.outputs[1]].begin(), values[op.outputs[1]].end());
    } else {
        result.operator_outputs.push_back(values[op.outputs[0]]);
    }
    return true;
}

ScalarExecutionResult execute_scalar_sequence(const RuntimePackage& package, std::span<const uint32_t> token_ids,
                                              const std::vector<KernelImplementation>& implementations,
                                              std::vector<SemanticKvState>& states) {
    const SemanticModel& model = package.semantics();
    if (token_ids.empty() || token_ids.size() > model.maximum_context || implementations.size() != model.operators.size()) {
        return scalar_error(CompatibilityError::RUNTIME_INPUT_INVALID);
    }
    if (std::find(implementations.begin(), implementations.end(), KernelImplementation::Unavailable) != implementations.end()) {
        return scalar_error(CompatibilityError::KERNEL_UNAVAILABLE);
    }
    for (uint32_t token : token_ids) if (token >= model.vocabulary_size) return scalar_error(CompatibilityError::RUNTIME_INPUT_INVALID);
    if (states.size() != model.states.size() / 2 ||
        model.operators.size() > std::numeric_limits<uint64_t>::max() / token_ids.size()) {
        return scalar_error(CompatibilityError::RUNTIME_INPUT_INVALID);
    }
    std::vector<std::vector<float>> values(model.values.size());
    ScalarExecutionOutput result;
    for (size_t index = 0; index != model.operators.size(); ++index) {
        if (!execute_scalar_operator(package, model, model.operators[index], implementations[index], token_ids, values, result, states)) {
            return scalar_error(CompatibilityError::IR_REFERENCE_INVALID);
        }
    }
    if (model.operators.empty() || model.operators.back().outputs.empty()) return scalar_error(CompatibilityError::IR_REFERENCE_INVALID);
    const auto& all_logits = values[model.operators.back().outputs[0]];
    if (all_logits.size() != static_cast<size_t>(model.vocabulary_size) * token_ids.size()) return scalar_error(CompatibilityError::IR_SHAPE_MISMATCH);
    result.logits.assign(all_logits.end() - model.vocabulary_size, all_logits.end());
    if (result.logits.size() != model.vocabulary_size) return scalar_error(CompatibilityError::IR_SHAPE_MISMATCH);
    for (float value : result.logits) if (!std::isfinite(value)) return scalar_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE);
    result.states = states;
    result.operator_token_work = static_cast<uint64_t>(model.operators.size()) * token_ids.size();
    return result;
}

ScalarExecutionResult execute_planned_scalar_with_states(const RuntimePackage& package, const ExecutionPlan& plan,
                                                         ExecutionPhase phase, std::span<const uint32_t> token_ids,
                                                         std::vector<SemanticKvState>& states) {
    const SemanticModel& model = package.semantics();
    std::vector<KernelImplementation> implementations;
    implementations.reserve(model.operators.size());
    size_t operator_index = 0;
    for (const PlanEntry& entry : plan.entries) {
        if (entry.phase != phase) continue;
        if (operator_index >= model.operators.size() || entry.operator_id != model.operators[operator_index].id ||
            !plan_entry_matches(model, entry)) return scalar_error(CompatibilityError::KERNEL_UNAVAILABLE);
        implementations.push_back(entry.descriptor.implementation);
        ++operator_index;
    }
    if (operator_index != model.operators.size()) return scalar_error(CompatibilityError::KERNEL_UNAVAILABLE);
    return execute_scalar_sequence(package, token_ids, implementations, states);
}

} // namespace

ScalarExecutionResult scalar_execute(const RuntimePackage& package, std::span<const uint32_t> token_ids) {
    std::vector<KernelImplementation> implementations;
    implementations.reserve(package.semantics().operators.size());
    for (const SemanticOperator& op : package.semantics().operators) implementations.push_back(scalar_implementation(op.kind));
    std::vector<SemanticKvState> states(package.semantics().states.size() / 2);
    return execute_scalar_sequence(package, token_ids, implementations, states);
}

ScalarExecutionResult execute_planned_scalar(const RuntimePackage& package, const ExecutionPlan& plan,
                                             ExecutionPhase phase, std::span<const uint32_t> token_ids) {
    std::vector<SemanticKvState> states(package.semantics().states.size() / 2);
    return execute_planned_scalar_with_states(package, plan, phase, token_ids, states);
}

ScalarExecutionResult execute_planned_scalar_incremental(const RuntimePackage& package, const ExecutionPlan& plan,
                                                         ExecutionPhase phase, std::span<const uint32_t> token_ids,
                                                         std::vector<SemanticKvState>& states) {
    return execute_planned_scalar_with_states(package, plan, phase, token_ids, states);
}

} // namespace Laplace
