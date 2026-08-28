#include "reference_fp32.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <limits>

#include "fp16.h"

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

CompatibilityReport reference_error(CompatibilityError code) {
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
            uint8_t logical_axis = tensor_.layout.axis_order[physical_axis];
            if (logical_axis >= dimensions_.size() || logical_index.begin()[logical_axis] >= dimensions_[logical_axis]) return false;
            physical_index += logical_index.begin()[logical_axis] * stride;
            if (dimensions_[logical_axis] > std::numeric_limits<uint64_t>::max() / stride) return false;
            stride *= dimensions_[logical_axis];
        }
        uint64_t byte_offset = plane_->offset + physical_index * element_bytes_;
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

bool finite_values(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

bool linear(const RuntimePackage& package, const SemanticModel& model, const SemanticOperator& op,
            const std::vector<float>& input, std::vector<float>& output) {
    if (op.tensors.empty() || op.tensors.size() > 2) return false;
    const auto* payload = std::get_if<LinearPayload>(&op.payload);
    if (!payload || payload->accumulation_type != ScalarType::F32 || op.tensors.size() != (payload->has_bias ? 2u : 1u)) return false;
    if (op.tensors[0] >= model.tensors.size()) return false;
    TensorReader weight(package, model.tensors[op.tensors[0]]);
    if (!weight.valid() || weight.rank() != 2) return false;
    const uint64_t output_width = weight.dimension(0);
    const uint64_t input_width = weight.dimension(1);
    if (input_width == 0 || input.size() % input_width != 0 || output_width > std::numeric_limits<size_t>::max() / (input.size() / input_width)) return false;
    std::unique_ptr<TensorReader> bias;
    if (payload->has_bias) {
        if (op.tensors[1] >= model.tensors.size()) return false;
        bias = std::make_unique<TensorReader>(package, model.tensors[op.tensors[1]]);
        if (!bias->valid() || bias->rank() != 1 || bias->dimension(0) != output_width) return false;
    }
    output.assign((input.size() / input_width) * output_width, 0.0f);
    for (size_t row = 0; row != input.size() / input_width; ++row) {
        for (uint64_t out = 0; out != output_width; ++out) {
            float accumulator = 0.0f;
            float bias_value = 0.0f;
            if (payload->has_bias && !bias->value({out}, bias_value)) return false;
            float lanes[16] = {};
            uint64_t in = 0;
            for (; in + 16 <= input_width; in += 16) {
                for (uint64_t lane = 0; lane != 16; ++lane) {
                    float weight_value = 0.0f;
                    if (!weight.value({out, in + lane}, weight_value)) return false;
                    lanes[lane] = fp_add(lanes[lane], fp_mul(input[row * input_width + in + lane], weight_value));
                }
            }
            for (float lane : lanes) accumulator = fp_add(accumulator, lane);
            for (; in != input_width; ++in) {
                float weight_value = 0.0f;
                if (!weight.value({out, in}, weight_value)) return false;
                accumulator = fp_add(accumulator, fp_mul(input[row * input_width + in], weight_value));
            }
            if (payload->has_bias) accumulator = fp_add(accumulator, bias_value);
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
    if (!weight.valid() || weight.rank() != 1 || weight.dimension(0) == 0 || input.size() % weight.dimension(0) != 0) return false;
    const uint64_t width = weight.dimension(0);
    output.resize(input.size());
    for (size_t row = 0; row != input.size() / width; ++row) {
        float lanes[16] = {};
        uint64_t column = 0;
        for (; column + 16 <= width; column += 16) {
            for (uint64_t lane = 0; lane != 16; ++lane) lanes[lane] = fp_add(lanes[lane], fp_mul(input[row * width + column + lane], input[row * width + column + lane]));
        }
        float sum = 0;
        for (float lane : lanes) sum = fp_add(sum, lane);
        for (; column != width; ++column) sum = fp_add(sum, fp_mul(input[row * width + column], input[row * width + column]));
        const float mean = fp32(sum / static_cast<float>(width));
        const float reciprocal = fp32(1.0f / std::sqrt(fp_add(mean, epsilon)));
        if (!std::isfinite(reciprocal)) return false;
        for (uint64_t column = 0; column != width; ++column) {
            float scale = 0;
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
    float scale = 0;
    TensorReader table(package, model.tensors[op.tensors[0]]);
    if (!f32_from_bits(payload->scale_f32_bits, scale) || scale <= 0 || !table.valid() || table.rank() != 2 ||
        table.dimension(0) != payload->vocabulary || table.dimension(1) != payload->width) return false;
    output.resize(static_cast<size_t>(payload->width) * tokens.size());
    for (size_t row = 0; row != tokens.size(); ++row) {
        if (tokens[row] >= payload->vocabulary) return false;
        for (uint64_t column = 0; column != payload->width; ++column) {
            float value = 0;
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
    const bool multi_section = payload && payload->pairing == RopePairing::MultiSectionHalfSplit;
    if (!payload || (payload->pairing != RopePairing::HalfSplit && payload->pairing != RopePairing::Interleaved &&
                     payload->pairing != RopePairing::MultiSectionHalfSplit) ||
        !payload->position_from_cursor ||
        payload->rotary_dimension == 0 || payload->rotary_dimension % 2 != 0 || rows == 0 || head_dimension == 0 ||
        query.size() % head_dimension != 0 || key.size() % head_dimension != 0 || payload->rotary_dimension > head_dimension) return false;
    if (multi_section) {
        uint64_t total_sections = 0;
        for (uint32_t section : payload->position_sections) total_sections += section;
        if (payload->position_sections[0] == 0 || payload->position_sections[1] == 0 ||
            total_sections != payload->rotary_dimension / 2) return false;
    }
    float base = 0, scale = 0;
    if (!f32_from_bits(payload->base_f32_bits, base) || !f32_from_bits(payload->scale_f32_bits, scale) || base <= 0 || scale <= 0) return false;
    auto rotate = [&](const std::vector<float>& input, std::vector<float>& output) {
        const size_t heads = input.size() / head_dimension / rows;
        if (heads == 0 || heads * rows * head_dimension != input.size()) return false;
        output = input;
        for (uint32_t row = 0; row != rows; ++row) {
            for (size_t head = 0; head != heads; ++head) {
                const size_t base_index = (static_cast<size_t>(row) * heads + head) * head_dimension;
                for (uint32_t pair = 0; pair != payload->rotary_dimension / 2; ++pair) {
                    if (multi_section) {
                        const uint32_t first = payload->position_sections[0];
                        const uint32_t second = first + payload->position_sections[1];
                        const uint32_t third = second + payload->position_sections[2];
                        const uint32_t position_axis = pair < first ? 0 : pair < second ? 1 : pair < third ? 2 : 3;
                        if (position_axis > 3) return false;
                    }
                    const float exponent = fp32(-2.0f * static_cast<float>(pair) / static_cast<float>(payload->rotary_dimension));
                    const float angle = fp_mul(fp_mul(static_cast<float>(first_position + row), scale), std::pow(base, exponent));
                    const float cosine = fp32(std::cos(angle));
                    const float sine = fp32(std::sin(angle));
                    const uint32_t first_index = payload->pairing == RopePairing::Interleaved ? pair * 2 : pair;
                    const uint32_t second_index = payload->pairing == RopePairing::Interleaved
                                                      ? first_index + 1
                                                      : pair + payload->rotary_dimension / 2;
                    const float first = input[base_index + first_index];
                    const float second = input[base_index + second_index];
                    output[base_index + first_index] = fp32(fp_mul(first, cosine) - fp_mul(second, sine));
                    output[base_index + second_index] = fp_add(fp_mul(second, cosine), fp_mul(first, sine));
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

bool causal_attention(const std::vector<float>& query, const std::vector<float>& key, const std::vector<float>& value,
                      uint32_t rows, const CausalAttentionPayload& payload, float scale, SemanticKvState& state,
                      std::vector<float>& output) {
    const uint32_t query_heads = payload.query_heads;
    const uint32_t kv_heads = payload.kv_heads;
    const uint32_t head_dimension = payload.head_dimension;
    if (!rows || !query_heads || !kv_heads || !head_dimension || query_heads % kv_heads || !std::isfinite(scale) ||
        query.size() != static_cast<size_t>(rows) * query_heads * head_dimension ||
        key.size() != static_cast<size_t>(rows) * kv_heads * head_dimension ||
        value.size() != key.size() || state.key.size() != static_cast<size_t>(state.tokens) * kv_heads * head_dimension ||
        state.value.size() != state.key.size() || !finite_values(query) || !finite_values(key) || !finite_values(value) ||
        !finite_values(state.key) || !finite_values(state.value)) return false;
    SemanticKvState candidate = state;
    candidate.key.insert(candidate.key.end(), key.begin(), key.end());
    candidate.value.insert(candidate.value.end(), value.begin(), value.end());
    candidate.tokens += rows;
    output.assign(static_cast<size_t>(rows) * query_heads * head_dimension, 0.0f);
    const uint32_t group = query_heads / kv_heads;
    for (uint32_t row = 0; row != rows; ++row) {
        const uint32_t sources = state.tokens + row + 1;
        for (uint32_t qh = 0; qh != query_heads; ++qh) {
            const uint32_t kvh = qh / group;
            std::vector<float> scores(sources);
            float maximum = -std::numeric_limits<float>::infinity();
            for (uint32_t source = 0; source != sources; ++source) {
                float lanes[16] = {};
                uint32_t dimension = 0;
                for (; dimension + 16 <= head_dimension; dimension += 16) {
                    for (uint32_t lane = 0; lane != 16; ++lane) {
                        lanes[lane] = fp_add(lanes[lane], fp_mul(query[(static_cast<size_t>(row) * query_heads + qh) * head_dimension + dimension + lane],
                                                                  candidate.key[(static_cast<size_t>(source) * kv_heads + kvh) * head_dimension + dimension + lane]));
                    }
                }
                float dot = 0.0f;
                for (float lane : lanes) dot = fp_add(dot, lane);
                for (; dimension != head_dimension; ++dimension) dot = fp_add(dot, fp_mul(query[(static_cast<size_t>(row) * query_heads + qh) * head_dimension + dimension], candidate.key[(static_cast<size_t>(source) * kv_heads + kvh) * head_dimension + dimension]));
                scores[source] = fp_mul(dot, scale);
                maximum = std::max(maximum, scores[source]);
            }
            float normalizer = 0.0f;
            for (float score : scores) normalizer = fp_add(normalizer, fp32(std::exp(fp32(score - maximum))));
            if (!std::isfinite(normalizer) || normalizer <= 0) return false;
            for (uint32_t dimension = 0; dimension != head_dimension; ++dimension) {
                float context = 0.0f;
                for (uint32_t source = 0; source != sources; ++source) {
                    const float probability = fp32(fp32(std::exp(fp32(scores[source] - maximum))) / normalizer);
                    context = fp_add(context, fp_mul(probability, candidate.value[(static_cast<size_t>(source) * kv_heads + kvh) * head_dimension + dimension]));
                }
                output[(static_cast<size_t>(row) * query_heads + qh) * head_dimension + dimension] = context;
            }
        }
    }
    state = std::move(candidate);
    return true;
}

bool swiglu(const std::vector<float>& gate, const std::vector<float>& up, std::vector<float>& output) {
    if (gate.size() != up.size() || !finite_values(gate) || !finite_values(up)) return false;
    output.resize(gate.size());
    for (size_t index = 0; index != gate.size(); ++index) output[index] = fp_mul(fp32(gate[index] / fp_add(1.0f, fp32(std::exp(-gate[index])))), up[index]);
    return true;
}

bool add(const std::vector<float>& left, const std::vector<float>& right, std::vector<float>& output) {
    if (left.size() != right.size() || !finite_values(left) || !finite_values(right)) return false;
    output.resize(left.size());
    for (size_t index = 0; index != left.size(); ++index) output[index] = fp_add(left[index], right[index]);
    return true;
}

} // namespace

ReferenceResult reference_fp32(const RuntimePackage& package, std::span<const uint32_t> token_ids) {
    const SemanticModel& model = package.semantics();
    if (token_ids.empty() || token_ids.size() > model.maximum_context) return reference_error(CompatibilityError::RUNTIME_INPUT_INVALID);
    for (uint32_t token : token_ids) if (token >= model.vocabulary_size) return reference_error(CompatibilityError::RUNTIME_INPUT_INVALID);
    std::vector<std::vector<float>> values(model.values.size());
    std::vector<SemanticKvState> states(model.states.size() / 2);
    ReferenceOutput result;
    for (const SemanticOperator& op : model.operators) {
        if (op.inputs.empty() || op.outputs.empty() || op.inputs[0] >= values.size() || op.outputs[0] >= values.size()) return reference_error(CompatibilityError::IR_REFERENCE_INVALID);
        std::vector<float> output;
        switch (op.kind) {
        case OperatorKind::EmbeddingLookup:
            if (!embedding(package, model, op, token_ids, output)) return reference_error(CompatibilityError::IR_LAYOUT_MISMATCH);
            values[op.outputs[0]] = std::move(output);
            break;
        case OperatorKind::RmsNorm:
            if (!rms_norm(package, model, op, values[op.inputs[0]], output)) return reference_error(CompatibilityError::IR_SHAPE_MISMATCH);
            values[op.outputs[0]] = std::move(output);
            break;
        case OperatorKind::Linear:
            if (!linear(package, model, op, values[op.inputs[0]], output)) return reference_error(CompatibilityError::IR_LAYOUT_MISMATCH);
            values[op.outputs[0]] = std::move(output);
            break;
        case OperatorKind::Rope: {
            if (op.inputs.size() != 2 || op.outputs.size() != 2 || op.inputs[1] >= values.size() || op.outputs[1] >= values.size() ||
                [&] { size_t slot = 0; uint32_t dimension = 0; return !rope_state_and_dimension(model, op, slot, dimension) || slot >= states.size() ||
                    !rope(op, values[op.inputs[0]], values[op.inputs[1]], states[slot].tokens,
                          static_cast<uint32_t>(token_ids.size()), dimension, values[op.outputs[0]], values[op.outputs[1]]); }()) return reference_error(CompatibilityError::IR_SHAPE_MISMATCH);
            break;
        }
        case OperatorKind::CausalAttention: {
            if (op.inputs.size() != 3 || op.states.size() != 2 || op.states[0] % 2 != 0 || op.states[1] != op.states[0] + 1 || op.states[0] / 2 >= states.size()) return reference_error(CompatibilityError::IR_REFERENCE_INVALID);
            const auto* payload = std::get_if<CausalAttentionPayload>(&op.payload);
            float scale = 0;
            if (!payload || !f32_from_bits(payload->scale_f32_bits, scale)) return reference_error(CompatibilityError::IR_CONSTRAINT_FAILED);
            size_t slot = 0;
            for (; slot + 1 < model.states.size(); slot += 2) if (model.states[slot].id == op.states[0] && model.states[slot + 1].id == op.states[1]) break;
            if (slot + 1 >= model.states.size() || !causal_attention(values[op.inputs[0]], values[op.inputs[1]], values[op.inputs[2]],
                                                                       static_cast<uint32_t>(token_ids.size()), *payload, scale, states[slot / 2], output)) return reference_error(CompatibilityError::IR_CONSTRAINT_FAILED);
            values[op.outputs[0]] = std::move(output);
            break;
        }
        case OperatorKind::SwiGlu: {
            if (op.inputs.size() != 2) return reference_error(CompatibilityError::IR_REFERENCE_INVALID);
            if (!swiglu(values[op.inputs[0]], values[op.inputs[1]], output)) return reference_error(CompatibilityError::IR_SHAPE_MISMATCH);
            values[op.outputs[0]] = std::move(output);
            break;
        }
        case OperatorKind::Add: {
            if (op.inputs.size() != 2) return reference_error(CompatibilityError::IR_REFERENCE_INVALID);
            if (!add(values[op.inputs[0]], values[op.inputs[1]], output)) return reference_error(CompatibilityError::IR_SHAPE_MISMATCH);
            values[op.outputs[0]] = std::move(output);
            break;
        }
        case OperatorKind::AxisSplit: {
            if (op.inputs.size() != 1 || op.outputs.size() != 2 || op.inputs[0] >= values.size() ||
                op.outputs[1] >= values.size()) return reference_error(CompatibilityError::IR_REFERENCE_INVALID);
            const auto* payload = std::get_if<AxisSplitPayload>(&op.payload);
            if (!payload || !payload->first_width || !payload->second_width ||
                payload->first_width > UINT32_MAX - payload->second_width) {
                return reference_error(CompatibilityError::IR_SHAPE_MISMATCH);
            }
            const size_t row_width = static_cast<size_t>(payload->first_width) + payload->second_width;
            const std::vector<float>& input = values[op.inputs[0]];
            if (input.empty() || input.size() % row_width || !finite_values(input)) {
                return reference_error(CompatibilityError::IR_SHAPE_MISMATCH);
            }
            const size_t rows = input.size() / row_width;
            std::vector<float> first;
            std::vector<float> second;
            first.reserve(rows * payload->first_width);
            second.reserve(rows * payload->second_width);
            for (size_t row = 0; row != rows; ++row) {
                const auto begin = input.begin() + row * row_width;
                first.insert(first.end(), begin, begin + payload->first_width);
                second.insert(second.end(), begin + payload->first_width, begin + row_width);
            }
            values[op.outputs[0]] = std::move(first);
            values[op.outputs[1]] = std::move(second);
            break;
        }
        case OperatorKind::GatedAttention: {
            if (op.inputs.size() != 2 || values[op.inputs[0]].size() != values[op.inputs[1]].size() ||
                !finite_values(values[op.inputs[0]]) || !finite_values(values[op.inputs[1]])) {
                return reference_error(CompatibilityError::IR_SHAPE_MISMATCH);
            }
            output.resize(values[op.inputs[0]].size());
            for (size_t index = 0; index != output.size(); ++index) {
                const float gate = values[op.inputs[1]][index];
                const float sigmoid = gate >= 0.0f
                    ? fp32(1.0f / fp_add(1.0f, fp32(std::exp(-gate))))
                    : fp32(std::exp(gate) / fp_add(1.0f, fp32(std::exp(gate))));
                output[index] = fp_mul(values[op.inputs[0]][index], sigmoid);
            }
            values[op.outputs[0]] = std::move(output);
            break;
        }
        case OperatorKind::DepthwiseConvSilu:
        case OperatorKind::GatedDeltaNet:
        case OperatorKind::GatedRmsNorm:
        case OperatorKind::L2Normalize:
        case OperatorKind::Concat:
        case OperatorKind::RouterTopK:
        case OperatorKind::RoutedLinear:
        case OperatorKind::GatedActivation:
        case OperatorKind::WeightedExpertReduce:
        case OperatorKind::Scale:
        case OperatorKind::TanhSoftcap:
            return reference_error(CompatibilityError::KERNEL_UNAVAILABLE);
        }
        if (op.kind == OperatorKind::Rope || op.kind == OperatorKind::AxisSplit) {
            result.operator_outputs.push_back(values[op.outputs[0]]);
            result.operator_outputs.back().insert(result.operator_outputs.back().end(), values[op.outputs[1]].begin(), values[op.outputs[1]].end());
        } else {
            result.operator_outputs.push_back(values[op.outputs[0]]);
        }
    }
    if (model.operators.empty() || model.operators.back().outputs.empty()) return reference_error(CompatibilityError::IR_REFERENCE_INVALID);
    const auto& all_logits = values[model.operators.back().outputs[0]];
    if (all_logits.size() != static_cast<size_t>(model.vocabulary_size) * token_ids.size()) return reference_error(CompatibilityError::IR_SHAPE_MISMATCH);
    result.logits.assign(all_logits.end() - model.vocabulary_size, all_logits.end());
    result.states = states;
    if (!states.empty()) result.key_state = states.front();
    if (result.logits.size() != model.vocabulary_size) return reference_error(CompatibilityError::IR_SHAPE_MISMATCH);
    for (float value : result.logits) if (!std::isfinite(value)) return reference_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE);
    return result;
}

} // namespace Laplace
