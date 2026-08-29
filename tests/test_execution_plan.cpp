#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "execution_plan.h"
#include "test_util.h"

using namespace Laplace;

namespace {

RuntimeCapabilities cpu_capabilities() {
    RuntimeCapabilities capabilities;
    capabilities.scalar_fp32 = true;
    capabilities.global_fp32_kv = true;
    capabilities.transactional_state = true;
    return capabilities;
}

SessionRequest request() {
    SessionRequest result;
    result.max_context = 4;
    result.max_batch = 1;
    result.memory_limit = UINT64_MAX;
    result.enable_prefill = true;
    result.enable_decode = true;
    result.minimum_class = NumericalClass::ExactFp32;
    result.objective = RuntimeObjective::Latency;
    return result;
}

SemanticModel add_model() {
    SemanticModel model;
    model.maximum_context = 32768;
    model.values = {
        {0, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1}}, 0},
        {1, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1}}, 0},
        {2, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1}}, 0},
    };
    model.operators.push_back({0, OperatorKind::Add, 1, {0, 1}, {2}});
    return model;
}

SemanticModel activation_model() {
    SemanticModel model;
    model.maximum_context = 32768;
    model.values = {
        {0, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 64}}, 0},
        {1, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 64}}, 0},
        {2, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 64}}, 0},
    };
    model.operators.push_back({0, OperatorKind::Add, 1, {0, 1}, {2}});
    return model;
}

SemanticModel state_model() {
    SemanticModel model;
    model.maximum_context = 32768;
    model.operators.push_back({0, OperatorKind::CausalAttention, 1});
    model.states.push_back({0, StateKind::KeyCache, 1, StateUpdateKind::AppendKey, PositionPolicy::AppendOnly,
                            {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}, {DimensionKind::Constant, 64}}, {StateFormat{}}, 0});
    model.states.push_back({1, StateKind::ValueCache, 1, StateUpdateKind::AppendValue, PositionPolicy::AppendOnly,
                            {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}, {DimensionKind::Constant, 64}}, {StateFormat{}}, 0});
    return model;
}

SemanticTensor dense_tensor(uint32_t id, TensorRole role, std::vector<Dimension> dimensions,
                            ScalarType storage_type, bool q4k = false) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = std::move(dimensions);
    tensor.layout.rank = static_cast<uint8_t>(tensor.dimensions.size());
    tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides[0] = 1;
    tensor.layout.strides[1] = tensor.dimensions.size() == 2 ? tensor.dimensions[0].constant_or_symbol : 0;
    uint64_t elements = 1;
    for (const Dimension& dimension : tensor.dimensions) elements *= dimension.constant_or_symbol;
    uint64_t length = elements * (storage_type == ScalarType::F16 ? 2 : 4);
    uint32_t alignment = 64;
    if (q4k) {
        tensor.layout.kind = PhysicalLayoutKind::GgufBlocked;
        tensor.layout.packing = PackingKind::Gguf;
        tensor.layout.block_rank = 1;
        tensor.layout.block_elements = 256;
        tensor.layout.block_bytes = 144;
        tensor.quantization.kind = QuantizationKind::BlockedAffine;
        tensor.quantization.accumulation_type = ScalarType::F32;
        tensor.quantization.scale_type = ScalarType::F16;
        tensor.quantization.zero_type = ScalarType::F16;
        tensor.quantization.block_elements = 256;
        tensor.quantization.block_bytes = 144;
        tensor.quantization.group_size = 256;
        tensor.quantization.required_plane_mask = 1;
        storage_type = ScalarType::U8;
        length = elements / 256 * 144;
        alignment = 32;
    }
    tensor.planes = {{PlaneKind::Values, storage_type, ArtifactId{0}, 0, length, alignment, 0}};
    return tensor;
}

SemanticModel q4k_dense_model() {
    SemanticModel model;
    model.maximum_context = 32768;
    model.vocabulary_size = 3;
    model.bos_id = 1;
    model.eos_id = 2;
    const std::vector<Dimension> hidden = {{DimensionKind::Constant, 256}};
    const std::vector<Dimension> matrix = {{DimensionKind::Constant, 256}, {DimensionKind::Constant, 256}};
    const std::vector<Dimension> embedding = {{DimensionKind::Constant, 256}, {DimensionKind::Constant, 3}};
    model.tensors = {
        dense_tensor(0, TensorRole::TokenEmbedding, embedding, ScalarType::F16),
        dense_tensor(1, TensorRole::AttentionNormWeight, hidden, ScalarType::F32),
        dense_tensor(2, TensorRole::QueryWeight, matrix, ScalarType::U8, true),
        dense_tensor(3, TensorRole::KeyWeight, matrix, ScalarType::U8, true),
        dense_tensor(4, TensorRole::ValueWeight, matrix, ScalarType::U8, true),
        dense_tensor(5, TensorRole::AttentionOutputWeight, matrix, ScalarType::U8, true),
        dense_tensor(6, TensorRole::FfnNormWeight, hidden, ScalarType::F32),
        dense_tensor(7, TensorRole::FfnGateWeight, matrix, ScalarType::U8, true),
        dense_tensor(8, TensorRole::FfnUpWeight, matrix, ScalarType::U8, true),
        dense_tensor(9, TensorRole::FfnDownWeight, matrix, ScalarType::U8, true),
        dense_tensor(10, TensorRole::FinalNormWeight, hidden, ScalarType::F32),
        dense_tensor(11, TensorRole::OutputWeight, embedding, ScalarType::F16),
    };
    model.values.clear();
    const auto value = [&](uint32_t id, ScalarType type, std::vector<Dimension> dimensions) {
        model.values.push_back({id, type, std::move(dimensions), 0});
    };
    const auto rows = [](uint32_t width) {
        return std::vector<Dimension>{{DimensionKind::Symbol, 1}, {DimensionKind::Constant, width}};
    };
    value(0, ScalarType::F32, rows(256)); // embedding output / residual input
    value(1, ScalarType::F32, rows(256)); // attention norm
    value(2, ScalarType::F32, rows(256)); // query
    value(3, ScalarType::F32, rows(256)); // key
    value(4, ScalarType::F32, rows(256)); // value
    value(5, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}, {DimensionKind::Constant, 64}});
    value(6, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}, {DimensionKind::Constant, 64}});
    value(7, ScalarType::F32, rows(256)); // attention context
    value(8, ScalarType::F32, rows(256)); // attention output projection
    value(9, ScalarType::F32, rows(256)); // attention residual
    value(10, ScalarType::F32, rows(256)); // FFN norm
    value(11, ScalarType::F32, rows(256)); // gate
    value(12, ScalarType::F32, rows(256)); // up
    value(13, ScalarType::F32, rows(256)); // SwiGLU
    value(14, ScalarType::F32, rows(256)); // FFN down
    value(15, ScalarType::F32, rows(256)); // final residual
    value(16, ScalarType::F32, rows(256)); // final norm
    value(17, ScalarType::F32, rows(3)); // output logits
    value(18, ScalarType::U32, {{DimensionKind::Symbol, 1}}); // token IDs
    auto add = [&](OperatorKind kind, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                   std::vector<uint32_t> tensors, OperatorPayload payload) {
        SemanticOperator op;
        op.id = static_cast<uint32_t>(model.operators.size());
        op.kind = kind;
        op.inputs = std::move(inputs);
        op.tensors = std::move(tensors);
        op.outputs = std::move(outputs);
        op.payload = std::move(payload);
        model.operators.push_back(std::move(op));
    };
    add(OperatorKind::EmbeddingLookup, {18}, {0}, {0}, EmbeddingLookupPayload{0x3f800000u, 3, 256, 0});
    add(OperatorKind::RmsNorm, {0}, {1}, {1}, RmsNormPayload{0x358637bdu, -1, 1});
    add(OperatorKind::Linear, {1}, {2}, {2}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {3}, {3}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {4}, {4}, LinearPayload{});
    add(OperatorKind::Rope, {2, 3}, {5, 6}, {}, RopePayload{RopePairing::HalfSplit, true, 64, 0x49742400u, 0x3f800000u});
    SemanticOperator attention;
    attention.id = static_cast<uint32_t>(model.operators.size());
    attention.kind = OperatorKind::CausalAttention;
    attention.inputs = {5, 6, 4};
    attention.outputs = {7};
    attention.states = {0, 1};
    attention.payload = CausalAttentionPayload{4, 4, 64, 0x3e800000u, AttentionMask::Causal, CachePolicy::Global};
    model.operators.push_back(std::move(attention));
    add(OperatorKind::Linear, {7}, {8}, {5}, LinearPayload{});
    add(OperatorKind::Add, {0, 8}, {9}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {9}, {10}, {6}, RmsNormPayload{0x358637bdu, -1, 1});
    add(OperatorKind::Linear, {10}, {11}, {7}, LinearPayload{});
    add(OperatorKind::Linear, {10}, {12}, {8}, LinearPayload{});
    add(OperatorKind::SwiGlu, {11, 12}, {13}, {}, SwiGluPayload{});
    add(OperatorKind::Linear, {13}, {14}, {9}, LinearPayload{});
    add(OperatorKind::Add, {9, 14}, {15}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {15}, {16}, {10}, RmsNormPayload{0x358637bdu, -1, 1});
    add(OperatorKind::Linear, {16}, {17}, {11}, LinearPayload{});
    model.input_values_first = 18;
    model.input_values_count = 1;
    model.output_values_first = 17;
    model.output_values_count = 1;
    model.layers = {{0, 1, 14, 0}};
    StateFormat key_format;
    key_format.encoded_domain = TransformDomain::RopeApplied;
    key_format.alignment = 64;
    SemanticState key_state;
    key_state.id = 0;
    key_state.kind = StateKind::KeyCache;
    key_state.update_kind = StateUpdateKind::AppendKey;
    key_state.dimensions = {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}, {DimensionKind::Constant, 64}};
    key_state.formats = {key_format};
    SemanticState value_state;
    value_state.id = 1;
    value_state.kind = StateKind::ValueCache;
    value_state.update_kind = StateUpdateKind::AppendValue;
    value_state.dimensions = key_state.dimensions;
    StateFormat value_format;
    value_format.alignment = 64;
    value_state.formats = {value_format};
    model.states = {key_state, value_state};
    return model;
}

SemanticModel q4k_query_gate_dense_model() {
    SemanticModel model = q4k_dense_model();
    model.schema_major = 4;
    model.opset_major = 4;
    for (SemanticOperator& op : model.operators) op.semantic_version = 4;
    for (SemanticState& state : model.states) state.semantic_version = 4;

    SemanticTensor& fused_query_gate = model.tensors[2];
    fused_query_gate.role = TensorRole::AttentionQueryGateWeight;
    fused_query_gate.dimensions[0].constant_or_symbol = 256;
    fused_query_gate.dimensions[1].constant_or_symbol = 512;
    fused_query_gate.layout.strides[1] = 256;
    fused_query_gate.planes[0].length = 512ull * 256 / 256 * 144;

    const auto set_value = [&](uint32_t id, uint32_t width) {
        model.values[id].dimensions = {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, width}};
        model.values[id].logical_type = ScalarType::F32;
    };
    set_value(5, 512); // fused query and gate
    set_value(6, 256); // query
    set_value(7, 256); // attention gate
    set_value(8, 256); // key
    set_value(9, 256); // value
    set_value(10, 256); // rotated query
    set_value(11, 256); // rotated key
    model.values[12].dimensions = {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}, {DimensionKind::Constant, 64}};
    model.values[13].dimensions = model.values[12].dimensions;
    set_value(14, 256); // attention output projection
    set_value(15, 256); // attention residual
    set_value(16, 256); // FFN norm
    set_value(17, 256); // FFN gate
    model.values.push_back({19, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 256}}, 0});
    model.values.push_back({20, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 256}}, 0});
    model.values.push_back({21, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 256}}, 0});
    model.values.push_back({22, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 256}}, 0});
    model.values.push_back({23, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 256}}, 0});
    model.values.push_back({24, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 3}}, 0});

    SemanticOperator split;
    split.kind = OperatorKind::AxisSplit;
    split.semantic_version = 4;
    split.inputs = {5};
    split.outputs = {6, 7};
    split.payload = AxisSplitPayload{256, 256};
    model.operators.insert(model.operators.begin() + 3, std::move(split));

    const auto find = [&](OperatorKind kind, TensorRole role) -> SemanticOperator& {
        const auto found = std::find_if(model.operators.begin(), model.operators.end(), [&](const SemanticOperator& op) {
            return op.kind == kind && !op.tensors.empty() && model.tensors[op.tensors[0]].role == role;
        });
        CHECK(found != model.operators.end());
        return *found;
    };
    SemanticOperator& query_gate = find(OperatorKind::Linear, TensorRole::AttentionQueryGateWeight);
    SemanticOperator& key = find(OperatorKind::Linear, TensorRole::KeyWeight);
    SemanticOperator& value_op = find(OperatorKind::Linear, TensorRole::ValueWeight);
    SemanticOperator& attention_output = find(OperatorKind::Linear, TensorRole::AttentionOutputWeight);
    query_gate.inputs = {1};
    query_gate.outputs = {5};
    key.inputs = {1};
    key.outputs = {8};
    value_op.inputs = {1};
    value_op.outputs = {9};
    const auto rope = std::find_if(model.operators.begin(), model.operators.end(), [](const SemanticOperator& op) {
        return op.kind == OperatorKind::Rope;
    });
    CHECK(rope != model.operators.end());
    rope->inputs = {6, 8};
    rope->outputs = {10, 11};
    rope->payload = RopePayload{RopePairing::Interleaved, true, 64, 0x49742400u, 0x3f800000u};
    const auto attention = std::find_if(model.operators.begin(), model.operators.end(), [](const SemanticOperator& op) {
        return op.kind == OperatorKind::CausalAttention;
    });
    CHECK(attention != model.operators.end());
    attention->inputs = {10, 11, 9};
    attention->outputs = {12};

    SemanticOperator gated;
    gated.kind = OperatorKind::GatedAttention;
    gated.semantic_version = 4;
    gated.inputs = {12, 7};
    gated.outputs = {13};
    gated.payload = GatedAttentionPayload{};
    attention_output.inputs = {13};
    attention_output.outputs = {14};
    model.operators.insert(attention + 1, std::move(gated));

    auto ffn_norm = std::find_if(model.operators.begin(), model.operators.end(), [&](const SemanticOperator& op) {
        return op.kind == OperatorKind::RmsNorm && !op.tensors.empty() && op.tensors[0] == 6;
    });
    auto gate = std::find_if(model.operators.begin(), model.operators.end(), [&](const SemanticOperator& op) {
        return op.kind == OperatorKind::Linear && !op.tensors.empty() && op.tensors[0] == 7;
    });
    auto up = std::find_if(model.operators.begin(), model.operators.end(), [&](const SemanticOperator& op) {
        return op.kind == OperatorKind::Linear && !op.tensors.empty() && op.tensors[0] == 8;
    });
    auto swiglu = std::find_if(model.operators.begin(), model.operators.end(), [](const SemanticOperator& op) {
        return op.kind == OperatorKind::SwiGlu;
    });
    auto down = std::find_if(model.operators.begin(), model.operators.end(), [&](const SemanticOperator& op) {
        return op.kind == OperatorKind::Linear && !op.tensors.empty() && op.tensors[0] == 9;
    });
    CHECK(ffn_norm != model.operators.end()); CHECK(gate != model.operators.end());
    CHECK(up != model.operators.end()); CHECK(swiglu != model.operators.end()); CHECK(down != model.operators.end());
    if (ffn_norm != model.operators.end()) { ffn_norm->inputs = {15}; ffn_norm->outputs = {16}; }
    if (gate != model.operators.end()) { gate->inputs = {16}; gate->outputs = {17}; }
    if (up != model.operators.end()) { up->inputs = {16}; up->outputs = {19}; }
    if (swiglu != model.operators.end()) { swiglu->inputs = {17, 19}; swiglu->outputs = {20}; }
    if (down != model.operators.end()) { down->inputs = {20}; down->outputs = {21}; }
    const auto adds = std::find_if(model.operators.begin(), model.operators.end(), [](const SemanticOperator& op) {
        return op.kind == OperatorKind::Add && op.inputs == std::vector<uint32_t>{0, 8};
    });
    CHECK(adds != model.operators.end());
    if (adds != model.operators.end()) { adds->inputs = {0, 14}; adds->outputs = {15}; }
    const auto final_add = std::find_if(model.operators.begin(), model.operators.end(), [](const SemanticOperator& op) {
        return op.kind == OperatorKind::Add && op.inputs == std::vector<uint32_t>{9, 14};
    });
    CHECK(final_add != model.operators.end());
    if (final_add != model.operators.end()) { final_add->inputs = {15, 21}; final_add->outputs = {22}; }
    const auto final_norm = std::find_if(model.operators.begin(), model.operators.end(), [](const SemanticOperator& op) {
        return op.kind == OperatorKind::RmsNorm && !op.tensors.empty() && op.tensors[0] == 10;
    });
    const auto output = std::find_if(model.operators.begin(), model.operators.end(), [](const SemanticOperator& op) {
        return op.kind == OperatorKind::Linear && !op.tensors.empty() && op.tensors[0] == 11;
    });
    CHECK(final_norm != model.operators.end()); CHECK(output != model.operators.end());
    if (final_norm != model.operators.end()) { final_norm->inputs = {22}; final_norm->outputs = {23}; }
    if (output != model.operators.end()) { output->inputs = {23}; output->outputs = {24}; }
    model.output_values_first = 24;

    for (uint32_t id = 0; id != model.operators.size(); ++id) model.operators[id].id = id;
    model.layers[0].first_operator = 1;
    model.layers[0].operator_count = static_cast<uint32_t>(model.operators.size() - 3);
    return model;
}

SemanticModel q4k_decoupled_query_width_dense_model() {
    SemanticModel model = q4k_query_gate_dense_model();
    const auto set_width = [&](uint32_t value_id, uint32_t width) {
        model.values[value_id].dimensions = {
            {DimensionKind::Symbol, 1}, {DimensionKind::Constant, width}};
    };

    SemanticTensor& query_gate = model.tensors[2];
    query_gate.dimensions = {{DimensionKind::Constant, 256},
                             {DimensionKind::Constant, 1024}};
    query_gate.layout.strides[1] = 256;
    query_gate.planes[0].length = 256ull * 1024 / 256 * 144;
    SemanticTensor& attention_output = model.tensors[5];
    attention_output.dimensions = {{DimensionKind::Constant, 512},
                                   {DimensionKind::Constant, 256}};
    attention_output.layout.strides[1] = 512;
    attention_output.planes[0].length = 512ull * 256 / 256 * 144;

    set_width(5, 1024);
    set_width(6, 512);
    set_width(7, 512);
    set_width(10, 512);
    model.values[12].dimensions = {{DimensionKind::Symbol, 1},
                                   {DimensionKind::Constant, 8},
                                   {DimensionKind::Constant, 64}};
    model.values[13].dimensions = model.values[12].dimensions;

    for (SemanticOperator& op : model.operators) {
        if (op.kind == OperatorKind::AxisSplit) {
            op.payload = AxisSplitPayload{512, 512};
        } else if (op.kind == OperatorKind::CausalAttention) {
            auto* payload = std::get_if<CausalAttentionPayload>(&op.payload);
            CHECK(payload != nullptr);
            if (payload) {
                payload->query_heads = 8;
                payload->kv_heads = 4;
                payload->head_dimension = 64;
            }
        }
    }
    return model;
}

SemanticTensor recurrent_vector_tensor(uint32_t id, TensorRole role, uint32_t length) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, length}};
    tensor.layout.rank = 1;
    tensor.layout.axis_order = {0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides[0] = 1;
    tensor.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, 0, uint64_t{length} * 4, 64, 0}};
    return tensor;
}

SemanticModel recurrent_q4k_q6k_model() {
    SemanticModel model;
    model.schema_major = 3;
    model.opset_major = 3;
    model.maximum_context = 256;
    model.vocabulary_size = 1;
    constexpr uint32_t hidden = 256;
    constexpr uint32_t head = 256;
    constexpr uint32_t channels = 3 * head;
    const std::vector<Dimension> matrix = {{DimensionKind::Constant, hidden}, {DimensionKind::Constant, hidden}};
    const std::vector<Dimension> qkv_matrix = {{DimensionKind::Constant, hidden}, {DimensionKind::Constant, channels}};
    model.tensors = {
        recurrent_vector_tensor(0, TensorRole::AttentionNormWeight, hidden),
        dense_tensor(1, TensorRole::RecurrentQkvWeight, qkv_matrix, ScalarType::U8, true),
        dense_tensor(2, TensorRole::RecurrentGateWeight, matrix, ScalarType::U8, true),
        dense_tensor(3, TensorRole::RecurrentBetaWeight, matrix, ScalarType::U8, true),
        dense_tensor(4, TensorRole::RecurrentAlphaWeight, matrix, ScalarType::U8, true),
        dense_tensor(5, TensorRole::RecurrentConvWeight,
                     {{DimensionKind::Constant, channels}, {DimensionKind::Constant, 2}}, ScalarType::F32),
        recurrent_vector_tensor(6, TensorRole::RecurrentDtBias, 1),
        recurrent_vector_tensor(7, TensorRole::RecurrentDecayWeight, 1),
        recurrent_vector_tensor(8, TensorRole::RecurrentNormWeight, head),
        dense_tensor(9, TensorRole::RecurrentOutputWeight, matrix, ScalarType::U8, true),
        recurrent_vector_tensor(10, TensorRole::FfnNormWeight, hidden),
        dense_tensor(11, TensorRole::FfnGateWeight, matrix, ScalarType::U8, true),
        dense_tensor(12, TensorRole::FfnUpWeight, matrix, ScalarType::U8, true),
        dense_tensor(13, TensorRole::FfnDownWeight, matrix, ScalarType::U8, true),
    };
    const auto q6k = [&](uint32_t id) {
        SemanticTensor& tensor = model.tensors[id];
        tensor.layout.block_bytes = 210;
        tensor.quantization.block_bytes = 210;
        tensor.planes[0].length = tensor.dimensions[0].constant_or_symbol * tensor.dimensions[1].constant_or_symbol / 256 * 210;
    };
    q6k(3);
    q6k(9);
    const auto value = [&](uint32_t id, uint32_t width) {
        model.values.push_back({id, ScalarType::F32,
                                {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, width}}, 0});
    };
    value(0, hidden);     // token input
    value(1, hidden);     // recurrent input RMS
    value(2, channels);   // fused QKV
    value(3, head);       // recurrent gate
    value(4, head);       // beta
    value(5, head);       // alpha
    value(6, head);       // convolved Q
    value(7, head);       // convolved K
    value(8, head);       // convolved V
    value(9, head);       // normalized Q
    value(10, head);      // normalized K
    value(11, head);      // DeltaNet output
    value(12, head);      // gated RMS output
    value(13, hidden);    // recurrent output projection
    value(14, hidden);    // recurrent residual
    value(15, hidden);    // FFN RMS
    value(16, hidden);    // FFN gate
    value(17, hidden);    // FFN up
    value(18, hidden);    // SwiGLU
    value(19, hidden);    // FFN down
    value(20, hidden);    // FFN residual

    StateFormat recurrent;
    recurrent.kind = StateFormatKind::RecurrentContiguous;
    recurrent.logical_type = ScalarType::F32;
    recurrent.encoded_type = ScalarType::F32;
    recurrent.logical_domain = TransformDomain::Untransformed;
    recurrent.encoded_domain = TransformDomain::Untransformed;
    recurrent.codec = CodecKind::Fp32;
    recurrent.cache_policy = CachePolicy::Recurrent;
    recurrent.layout_policy = LayoutPolicy::ChannelMajorHistory;
    recurrent.alignment = 64;
    model.states.push_back({0, StateKind::RecurrentConvHistory, 3, StateUpdateKind::ShiftHistory,
                            PositionPolicy::ReplaceAtCursor,
                            {{DimensionKind::Constant, channels}, {DimensionKind::Constant, 1}}, {recurrent}, 0});
    recurrent.layout_policy = LayoutPolicy::ValueHeadKeyRowOutputColumn;
    model.states.push_back({1, StateKind::RecurrentDeltaMatrix, 3, StateUpdateKind::DeltaMatrix,
                            PositionPolicy::ReplaceAtCursor,
                            {{DimensionKind::Constant, 1}, {DimensionKind::Constant, head}, {DimensionKind::Constant, head}},
                            {recurrent}, 0});

    auto linear = [&](TensorRole role, uint32_t tensor, uint32_t input, uint32_t output) {
        SemanticOperator op;
        op.id = static_cast<uint32_t>(model.operators.size());
        op.kind = OperatorKind::Linear;
        op.semantic_version = 3;
        op.inputs = {input};
        op.outputs = {output};
        op.tensors = {tensor};
        op.payload = LinearPayload{};
        model.operators.push_back(std::move(op));
    };
    SemanticOperator input_norm;
    input_norm.id = static_cast<uint32_t>(model.operators.size());
    input_norm.kind = OperatorKind::RmsNorm;
    input_norm.semantic_version = 3;
    input_norm.inputs = {0}; input_norm.outputs = {1}; input_norm.tensors = {0};
    input_norm.payload = RmsNormPayload{0x358637bdu, -1, 1};
    model.operators.push_back(std::move(input_norm));
    linear(TensorRole::RecurrentQkvWeight, 1, 1, 2);
    linear(TensorRole::RecurrentGateWeight, 2, 1, 3);
    linear(TensorRole::RecurrentBetaWeight, 3, 1, 4);
    linear(TensorRole::RecurrentAlphaWeight, 4, 1, 5);
    SemanticOperator conv;
    conv.id = static_cast<uint32_t>(model.operators.size());
    conv.kind = OperatorKind::DepthwiseConvSilu;
    conv.semantic_version = 3;
    conv.inputs = {2}; conv.outputs = {6, 7, 8}; conv.tensors = {5}; conv.states = {0};
    conv.payload = DepthwiseConvSiluPayload{1, 1, head, 2};
    model.operators.push_back(std::move(conv));

    auto l2 = [&](uint32_t input, uint32_t output) {
        SemanticOperator op;
        op.id = static_cast<uint32_t>(model.operators.size());
        op.kind = OperatorKind::L2Normalize;
        op.semantic_version = 3;
        op.inputs = {input};
        op.outputs = {output};
        op.payload = L2NormalizePayload{0x358637bdu};
        model.operators.push_back(std::move(op));
    };
    l2(6, 9);
    l2(7, 10);
    SemanticOperator delta;
    delta.id = static_cast<uint32_t>(model.operators.size());
    delta.kind = OperatorKind::GatedDeltaNet;
    delta.semantic_version = 3;
    delta.inputs = {9, 10, 8, 4, 5}; delta.outputs = {11}; delta.tensors = {6, 7}; delta.states = {1};
    delta.payload = GatedDeltaNetPayload{1, 1, head, QkHeadMapping::ValueHeadModulo,
                                         BetaTransform::Sigmoid, DecayTransform::NegativeSoftplus,
                                         DeltaStateLayout::ValueHeadKeyRowOutputColumn, 0};
    model.operators.push_back(std::move(delta));
    SemanticOperator gated_rms;
    gated_rms.id = static_cast<uint32_t>(model.operators.size());
    gated_rms.kind = OperatorKind::GatedRmsNorm;
    gated_rms.semantic_version = 3;
    gated_rms.inputs = {11, 3};
    gated_rms.outputs = {12};
    gated_rms.tensors = {8};
    gated_rms.payload = GatedRmsNormPayload{0x358637bdu, ActivationKind::Silu, 1};
    model.operators.push_back(std::move(gated_rms));
    linear(TensorRole::RecurrentOutputWeight, 9, 12, 13);
    auto add = [&](uint32_t left, uint32_t right, uint32_t output) {
        SemanticOperator op;
        op.id = static_cast<uint32_t>(model.operators.size());
        op.kind = OperatorKind::Add;
        op.semantic_version = 3;
        op.inputs = {left, right};
        op.outputs = {output};
        op.payload = AddPayload{};
        model.operators.push_back(std::move(op));
    };
    add(0, 13, 14);
    SemanticOperator ffn_norm;
    ffn_norm.id = static_cast<uint32_t>(model.operators.size());
    ffn_norm.kind = OperatorKind::RmsNorm;
    ffn_norm.semantic_version = 3;
    ffn_norm.inputs = {14}; ffn_norm.outputs = {15}; ffn_norm.tensors = {10};
    ffn_norm.payload = RmsNormPayload{0x358637bdu, -1, 1};
    model.operators.push_back(std::move(ffn_norm));
    linear(TensorRole::FfnGateWeight, 11, 15, 16);
    linear(TensorRole::FfnUpWeight, 12, 15, 17);
    SemanticOperator swiglu;
    swiglu.id = static_cast<uint32_t>(model.operators.size());
    swiglu.kind = OperatorKind::SwiGlu;
    swiglu.semantic_version = 3;
    swiglu.inputs = {16, 17}; swiglu.outputs = {18};
    swiglu.payload = SwiGluPayload{ActivationKind::Silu};
    model.operators.push_back(std::move(swiglu));
    linear(TensorRole::FfnDownWeight, 13, 18, 19);
    add(14, 19, 20);
    model.layers = {{0, 0, static_cast<uint32_t>(model.operators.size()), 0}};
    return model;
}

SemanticModel mixed_query_gate_recurrent_model() {
    SemanticModel model = q4k_query_gate_dense_model();
    SemanticModel recurrent = recurrent_q4k_q6k_model();
    const uint32_t tensor_base = static_cast<uint32_t>(model.tensors.size());
    const uint32_t value_base = static_cast<uint32_t>(model.values.size());
    const uint32_t state_base = static_cast<uint32_t>(model.states.size());
    const uint32_t operator_base = model.layers.front().first_operator +
                                   model.layers.front().operator_count;
    const uint32_t dense_residual = 22;
    for (SemanticTensor tensor : recurrent.tensors) {
        tensor.id += tensor_base;
        model.tensors.push_back(std::move(tensor));
    }
    for (SemanticValue value : recurrent.values) {
        if (value.id == 0) continue;
        value.id = value_base + value.id - 1;
        model.values.push_back(std::move(value));
    }
    for (SemanticState state : recurrent.states) {
        state.id += state_base;
        state.semantic_version = 4;
        model.states.push_back(std::move(state));
    }
    std::vector<SemanticOperator> recurrent_operators;
    recurrent_operators.reserve(recurrent.operators.size());
    for (SemanticOperator op : recurrent.operators) {
        op.semantic_version = 4;
        for (uint32_t& value : op.inputs)
            value = value == 0 ? dense_residual : value_base + value - 1;
        for (uint32_t& value : op.outputs)
            value = value == 0 ? dense_residual : value_base + value - 1;
        for (uint32_t& tensor : op.tensors) tensor += tensor_base;
        for (uint32_t& state : op.states) state += state_base;
        recurrent_operators.push_back(std::move(op));
    }
    model.operators.insert(model.operators.begin() + operator_base,
                           recurrent_operators.begin(), recurrent_operators.end());
    for (uint32_t id = 0; id != model.operators.size(); ++id) model.operators[id].id = id;
    model.layers.push_back({1, operator_base,
                            static_cast<uint32_t>(recurrent_operators.size()), 0});

    const uint32_t recurrent_residual = value_base + 19;
    SemanticOperator& final_norm = model.operators[operator_base + recurrent_operators.size()];
    CHECK(final_norm.kind == OperatorKind::RmsNorm);
    final_norm.inputs = {recurrent_residual};
    return model;
}

SemanticModel schema6_model_with_disabled_terminal_layer() {
    SemanticModel model = mixed_query_gate_recurrent_model();
    model.schema_major = 6;
    model.schema_minor = 0;
    model.opset_major = 6;
    model.opset_minor = 0;
    for (SemanticOperator& op : model.operators) op.semantic_version = 6;
    for (SemanticState& state : model.states) state.semantic_version = 6;
    for (SemanticOperator& op : model.operators) {
        if (op.kind != OperatorKind::Rope) continue;
        auto* payload = std::get_if<RopePayload>(&op.payload);
        CHECK(payload != nullptr);
        if (!payload) return {};
        payload->pairing = RopePairing::MultiSectionHalfSplit;
        payload->position_sections = {1, 1, 1, payload->rotary_dimension / 2 - 3};
    }

    const uint32_t first = static_cast<uint32_t>(model.operators.size());
    SemanticOperator concat;
    concat.id = first;
    concat.kind = OperatorKind::Concat;
    concat.semantic_version = 6;
    concat.inputs = {0, 0};
    concat.outputs = {0};
    concat.payload = ConcatPayload{-1};
    model.operators.push_back(std::move(concat));
    model.layers.push_back({static_cast<uint32_t>(model.layers.size()), first, 1,
                            kSemanticLayerFlagSpeculative});
    return model;
}

KernelQuery query() {
    KernelQuery result;
    result.operation = OperatorKind::Add;
    result.semantic_version = 1;
    result.phase = ExecutionPhase::Decode;
    result.logical_type = ScalarType::F32;
    result.layout = PhysicalLayoutKind::ContiguousRowMajor;
    result.quantization = QuantizationKind::None;
    result.state_kind = StateKind::KeyCache;
    result.state_format = StateFormatKind::GlobalContiguous;
    result.rank = 2;
    result.alignment = 64;
    result.head_dimension = 64;
    result.tile_tokens = 0;
    return result;
}

KernelDescriptor descriptor(uint32_t id, uint16_t priority, PlanEffectKey effects) {
    KernelDescriptor result;
    result.id = id;
    result.priority = priority;
    result.pattern.operation = exact(OperatorKind::Add);
    result.pattern.semantic_version = {1, 1};
    result.pattern.phase = exact(ExecutionPhase::Decode);
    result.pattern.logical_type = exact(ScalarType::F32);
    result.pattern.storage_type = exact(ScalarType::F32);
    result.pattern.layout = exact(PhysicalLayoutKind::ContiguousRowMajor);
    result.pattern.quantization = exact(QuantizationKind::None);
    result.pattern.state_kind = exact(StateKind::KeyCache);
    result.pattern.state_format = exact(StateFormatKind::GlobalContiguous);
    result.pattern.rank = {2, 2};
    result.pattern.alignment = {64, 64};
    result.pattern.head_dimension = {64, 64};
    result.pattern.tile_tokens = {0, 0};
    result.capability_predicate = &requires_scalar_fp32;
    result.numerical_class = NumericalClass::ExactFp32;
    result.transactional = true;
    result.effects = effects;
    result.cost = {10, 10};
    return result;
}

void test_pattern_matrix() {
    const KernelQuery candidate = query();
    KernelPattern exact_pattern = descriptor(1, 1, {}).pattern;
    CHECK(pattern_matches(exact_pattern, candidate));

    exact_pattern.operation = any<OperatorKind>();
    exact_pattern.logical_type = any<ScalarType>();
    exact_pattern.state_kind = any<StateKind>();
    CHECK(pattern_matches(exact_pattern, candidate));

    exact_pattern.rank = {1, 2};
    exact_pattern.alignment = {64, 128};
    exact_pattern.head_dimension = {32, 64};
    exact_pattern.tile_tokens = {0, 0};
    CHECK(pattern_matches(exact_pattern, candidate));
    exact_pattern.rank = {3, 4};
    CHECK(!pattern_matches(exact_pattern, candidate));
    exact_pattern.rank = {2, 1};
    CHECK(!pattern_matches(exact_pattern, candidate));
}

void test_selection_matrix() {
    PlanEffectKey cpu_effects{BackendWriter::Cpu, NumericalClass::ExactFp32, StateFormatKind::GlobalContiguous,
                              FallbackKind::ExactCpu, true, ScratchLifetime::Phase};
    auto low = descriptor(9, 1, cpu_effects);
    auto high = descriptor(5, 2, cpu_effects);
    auto selected = select_kernel(query(), request(), cpu_capabilities(), {low, high});
    CHECK(std::holds_alternative<KernelDescriptor>(selected));
    if (auto* value = std::get_if<KernelDescriptor>(&selected)) CHECK(value->id == 5);

    low.priority = high.priority;
    low.id = 3;
    selected = select_kernel(query(), request(), cpu_capabilities(), {low, high});
    CHECK(std::holds_alternative<KernelDescriptor>(selected));
    if (auto* value = std::get_if<KernelDescriptor>(&selected)) CHECK(value->id == 3);

    high.effects.backend = BackendWriter::Metal;
    selected = select_kernel(query(), request(), cpu_capabilities(), {low, high});
    CHECK(std::holds_alternative<CompatibilityReport>(selected));
    if (auto* report = std::get_if<CompatibilityReport>(&selected)) CHECK(report->code == CompatibilityError::KERNEL_AMBIGUOUS);

    RuntimeCapabilities no_scalar;
    selected = select_kernel(query(), request(), no_scalar, {low});
    CHECK(std::holds_alternative<CompatibilityReport>(selected));
    if (auto* report = std::get_if<CompatibilityReport>(&selected)) CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
}

void test_plan_failures() {
    const auto registry = builtin_cpu_registry();
    auto missing_decode = registry;
    missing_decode.erase(std::remove_if(missing_decode.begin(), missing_decode.end(), [](const KernelDescriptor& item) {
        return item.pattern.operation.exact == OperatorKind::Add && item.pattern.phase.exact == ExecutionPhase::Decode;
    }), missing_decode.end());
    auto result = plan_session(add_model(), request(), cpu_capabilities(), missing_decode);
    CHECK(std::holds_alternative<CompatibilityReport>(result));
    if (auto* report = std::get_if<CompatibilityReport>(&result)) CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);

    auto streaming = request();
    streaming.enable_streaming = true;
    result = plan_session(add_model(), streaming, cpu_capabilities(), registry);
    CHECK(std::holds_alternative<CompatibilityReport>(result));
    if (auto* report = std::get_if<CompatibilityReport>(&result)) CHECK(report->code == CompatibilityError::STREAMING_UNSUPPORTED);

    auto long_context = request();
    long_context.max_context = 32769;
    result = plan_session(add_model(), long_context, cpu_capabilities(), registry);
    CHECK(std::holds_alternative<CompatibilityReport>(result));
    if (auto* report = std::get_if<CompatibilityReport>(&result)) CHECK(report->code == CompatibilityError::PLAN_CONTEXT_EXCEEDED);

    auto speculative = request();
    speculative.enable_speculation = true;
    auto no_undo = cpu_capabilities();
    no_undo.transactional_state = false;
    result = plan_session(state_model(), speculative, no_undo, registry);
    CHECK(std::holds_alternative<CompatibilityReport>(result));
    if (auto* report = std::get_if<CompatibilityReport>(&result)) CHECK(report->code == CompatibilityError::FALLBACK_FORBIDDEN);

    auto limited = request();
    limited.memory_limit = 1;
    result = plan_session(add_model(), limited, cpu_capabilities(), registry);
    CHECK(std::holds_alternative<CompatibilityReport>(result));
    if (auto* report = std::get_if<CompatibilityReport>(&result)) CHECK(report->code == CompatibilityError::PLAN_MEMORY_EXCEEDED);

    auto cpu_only = cpu_capabilities();
    cpu_only.metal_device = false;
    cpu_only.metal_library = false;
    cpu_only.metal_pipeline = false;
    result = plan_session(add_model(), request(), cpu_only, registry);
    CHECK(std::holds_alternative<ExecutionPlan>(result));
}

void test_materialized_entries_and_peak_memory() {
    auto result = plan_session(activation_model(), request(), cpu_capabilities(), builtin_cpu_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(result));
    if (auto* plan = std::get_if<ExecutionPlan>(&result)) {
        CHECK(plan->peak_bytes > plan->reserved_bytes);
        CHECK(!plan->entries.empty());
        CHECK(plan->entries.front().descriptor.id == plan->entries.front().kernel_id);
        CHECK(plan->entries.front().descriptor.pattern.operation.exact == OperatorKind::Add);
    }

    auto constrained = request();
    constrained.memory_limit = 1024;
    result = plan_session(activation_model(), constrained, cpu_capabilities(), builtin_cpu_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(result));
    if (auto* report = std::get_if<CompatibilityReport>(&result)) {
        CHECK(report->code == CompatibilityError::PLAN_MEMORY_EXCEEDED);
    }
}

void test_metal_blocked_descriptor_requires_the_exact_q4k_contract() {
    auto blocked = query();
    blocked.operation = OperatorKind::CausalAttention;
    blocked.phase = ExecutionPhase::Decode;
    blocked.storage_type = ScalarType::U8;
    blocked.layout = PhysicalLayoutKind::GgufBlocked;
    blocked.quantization = QuantizationKind::BlockedAffine;
    blocked.alignment = 32;
    blocked.rank = 3;
    blocked.head_dimension = 256;
    blocked.metal_dense_token_pattern = true;
    blocked.metal_weight_format_mask = MetalWeightFormatQ4K;
    blocked.block_elements = 256;
    blocked.block_bytes = 144;

    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    auto selected = select_kernel(blocked, request(), metal, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<KernelDescriptor>(selected));
    if (const auto* descriptor = std::get_if<KernelDescriptor>(&selected)) {
        CHECK(descriptor->implementation == KernelImplementation::MetalDenseToken);
        CHECK(descriptor->pattern.allowed_metal_weight_formats == MetalWeightFormatQ4K);
    }

    blocked.block_bytes = 143;
    selected = select_kernel(blocked, request(), metal, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(selected));
    if (const auto* report = std::get_if<CompatibilityReport>(&selected)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
    }
}

void test_metal_quantized_descriptor_signatures_are_closed() {
    struct Contract { uint32_t format; uint32_t elements; uint32_t bytes; };
    const Contract contracts[] = {
        {MetalWeightFormatQ4_0, 32, 18},
        {MetalWeightFormatQ5_0, 32, 22},
        {MetalWeightFormatQ6K, 256, 210},
        {MetalWeightFormatQ8_0, 32, 34},
    };
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    for (const Contract& contract : contracts) {
        KernelQuery blocked;
        blocked.operation = OperatorKind::CausalAttention;
        blocked.semantic_version = 1;
        blocked.phase = ExecutionPhase::Decode;
        blocked.logical_type = ScalarType::F32;
        blocked.storage_type = ScalarType::U8;
        blocked.layout = PhysicalLayoutKind::GgufBlocked;
        blocked.quantization = QuantizationKind::BlockedAffine;
        blocked.state_kind = StateKind::KeyCache;
        blocked.state_format = StateFormatKind::GlobalContiguous;
        blocked.rank = 3;
        blocked.alignment = 32;
        blocked.head_dimension = 256;
        blocked.block_elements = contract.elements;
        blocked.block_bytes = contract.bytes;
        blocked.metal_weight_format_mask = contract.format;
        blocked.metal_dense_token_pattern = true;
        auto selected = select_kernel(blocked, request(), metal, builtin_canonical_metal_registry());
        CHECK(std::holds_alternative<KernelDescriptor>(selected));
        if (const auto* descriptor = std::get_if<KernelDescriptor>(&selected)) {
            CHECK(descriptor->implementation == KernelImplementation::MetalDenseToken);
            CHECK(descriptor->pattern.allowed_metal_weight_formats == contract.format);
        }
    }
}

void test_metal_iq2_xxs_descriptor_requires_the_exact_codebook_contract() {
    KernelQuery blocked;
    blocked.operation = OperatorKind::CausalAttention;
    blocked.semantic_version = 1;
    blocked.phase = ExecutionPhase::Decode;
    blocked.logical_type = ScalarType::F32;
    blocked.storage_type = ScalarType::U8;
    blocked.layout = PhysicalLayoutKind::GgufBlocked;
    blocked.quantization = QuantizationKind::Codebook;
    blocked.state_kind = StateKind::KeyCache;
    blocked.state_format = StateFormatKind::GlobalContiguous;
    blocked.rank = 3;
    blocked.alignment = 32;
    blocked.head_dimension = 256;
    blocked.block_elements = 256;
    blocked.block_bytes = 66;
    blocked.metal_weight_format_mask = MetalWeightFormatIQ2XXS;
    blocked.metal_dense_token_pattern = true;

    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    auto selected = select_kernel(blocked, request(), metal,
                                  builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<KernelDescriptor>(selected));
    if (const auto* descriptor = std::get_if<KernelDescriptor>(&selected)) {
        CHECK(descriptor->implementation == KernelImplementation::MetalDenseToken);
        CHECK(descriptor->pattern.allowed_metal_weight_formats == MetalWeightFormatIQ2XXS);
    }

    blocked.block_bytes = 65;
    selected = select_kernel(blocked, request(), metal,
                             builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(selected));
}

void test_affine_u2_requires_a_queried_capability() {
    constexpr uint32_t affine_format = MetalWeightFormatAffineUInt2_256;
    KernelDescriptor descriptor;
    descriptor.id = 90000;
    descriptor.implementation = KernelImplementation::MetalDenseToken;
    descriptor.pattern.operation = exact(OperatorKind::CausalAttention);
    descriptor.pattern.semantic_version = {1, 1};
    descriptor.pattern.phase = exact(ExecutionPhase::Prefill);
    descriptor.pattern.logical_type = exact(ScalarType::F32);
    descriptor.pattern.storage_type = exact(ScalarType::U32);
    descriptor.pattern.layout = exact(PhysicalLayoutKind::GroupedAffine);
    descriptor.pattern.quantization = exact(QuantizationKind::BlockedAffine);
    descriptor.pattern.state_kind = exact(StateKind::KeyCache);
    descriptor.pattern.state_format = exact(StateFormatKind::GlobalContiguous);
    descriptor.pattern.rank = {2, 2};
    descriptor.pattern.alignment = {128, 128};
    descriptor.pattern.head_dimension = {64, 64};
    descriptor.pattern.batch_rows = {1, 1};
    descriptor.pattern.block_elements = {256, 256};
    descriptor.pattern.block_bytes = {64, 64};
    descriptor.pattern.allowed_metal_weight_formats = affine_format;
    descriptor.pattern.require_metal_dense_token_pattern = true;
    descriptor.capability_predicate = &requires_canonical_metal;
    descriptor.numerical_class = NumericalClass::ExactFp32;
    descriptor.transactional = true;
    descriptor.effects = {BackendWriter::Metal, NumericalClass::ExactFp32,
                          StateFormatKind::GlobalContiguous, FallbackKind::None,
                          true, ScratchLifetime::Session};
    descriptor.cost = {1, 1};

    KernelQuery query;
    query.operation = OperatorKind::CausalAttention;
    query.semantic_version = 1;
    query.phase = ExecutionPhase::Prefill;
    query.logical_type = ScalarType::F32;
    query.storage_type = ScalarType::U32;
    query.layout = PhysicalLayoutKind::GroupedAffine;
    query.quantization = QuantizationKind::BlockedAffine;
    query.state_kind = StateKind::KeyCache;
    query.state_format = StateFormatKind::GlobalContiguous;
    query.rank = 2;
    query.alignment = 128;
    query.head_dimension = 64;
    query.batch_rows = 1;
    query.block_elements = 256;
    query.block_bytes = 64;
    query.metal_weight_format_mask = affine_format;
    query.metal_dense_token_pattern = true;
    query.metal_affine_u2_256 = true;

    RuntimeCapabilities capabilities;
    capabilities.metal_device = true;
    capabilities.metal_library = true;
    capabilities.metal_pipeline = true;
    capabilities.global_fp32_kv = true;
    capabilities.transactional_state = true;
    const KernelSelection selected = select_kernel(query, request(), capabilities, {descriptor});
    CHECK(std::holds_alternative<CompatibilityReport>(selected));
    CHECK(std::get<CompatibilityReport>(selected).code == CompatibilityError::CAPABILITY_MISSING);
    capabilities.metal_affine_u2_256 = true;
    const KernelSelection admitted = select_kernel(query, request(), capabilities,
                                                   builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<KernelDescriptor>(admitted));
    if (const auto* descriptor = std::get_if<KernelDescriptor>(&admitted)) {
        CHECK(descriptor->pattern.allowed_metal_weight_formats == affine_format);
        CHECK(descriptor->pattern.require_metal_affine_u2_256);
    }

    KernelQuery mixed = query;
    mixed.storage_type = static_cast<ScalarType>(0);
    mixed.layout = static_cast<PhysicalLayoutKind>(0);
    mixed.quantization = static_cast<QuantizationKind>(0);
    mixed.block_elements = 0;
    mixed.block_bytes = 0;
    mixed.metal_weight_format_mask = MetalWeightFormatF16 | affine_format;
    const KernelSelection mixed_admitted = select_kernel(
        mixed, request(), capabilities, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<KernelDescriptor>(mixed_admitted));
    if (const auto* selected_descriptor = std::get_if<KernelDescriptor>(&mixed_admitted)) {
        CHECK(selected_descriptor->pattern.require_mixed_metal_weight_formats);
        CHECK(selected_descriptor->pattern.require_metal_affine_u2_256);
        CHECK((selected_descriptor->pattern.allowed_metal_weight_formats &
               mixed.metal_weight_format_mask) == mixed.metal_weight_format_mask);
    }
}

void test_column_grouped_affine_u2_skip_has_a_distinct_typed_route() {
    constexpr uint32_t legacy_format = MetalWeightFormatAffineUInt2_256;
    constexpr uint32_t format = MetalWeightFormatColumnGroupedAffineUInt2Skip256;
    CHECK(format != legacy_format);
    CHECK(PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip != PhysicalLayoutKind::GroupedAffine);

    KernelDescriptor descriptor;
    descriptor.id = 90001;
    descriptor.implementation = KernelImplementation::MetalDenseToken;
    descriptor.pattern.operation = exact(OperatorKind::CausalAttention);
    descriptor.pattern.semantic_version = {1, 1};
    descriptor.pattern.phase = exact(ExecutionPhase::Prefill);
    descriptor.pattern.logical_type = exact(ScalarType::F32);
    descriptor.pattern.storage_type = exact(ScalarType::U8);
    descriptor.pattern.layout = exact(PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip);
    descriptor.pattern.quantization = exact(QuantizationKind::BlockedAffine);
    descriptor.pattern.state_kind = exact(StateKind::KeyCache);
    descriptor.pattern.state_format = exact(StateFormatKind::GlobalContiguous);
    descriptor.pattern.rank = {2, 2};
    descriptor.pattern.alignment = {128, 128};
    descriptor.pattern.head_dimension = {64, 64};
    descriptor.pattern.batch_rows = {1, 1};
    descriptor.pattern.block_elements = {256, 256};
    descriptor.pattern.block_bytes = {64, 64};
    descriptor.pattern.allowed_metal_weight_formats = format;
    descriptor.pattern.require_metal_dense_token_pattern = true;
    descriptor.pattern.require_metal_column_grouped_affine_u2_skip_256 = true;
    descriptor.capability_predicate = &requires_canonical_metal;
    descriptor.numerical_class = NumericalClass::ExactFp32;
    descriptor.transactional = true;
    descriptor.effects = {BackendWriter::Metal, NumericalClass::ExactFp32,
                          StateFormatKind::GlobalContiguous, FallbackKind::None,
                          true, ScratchLifetime::Session};
    descriptor.cost = {1, 1};

    KernelQuery query;
    query.operation = OperatorKind::CausalAttention;
    query.semantic_version = 1;
    query.phase = ExecutionPhase::Prefill;
    query.logical_type = ScalarType::F32;
    query.storage_type = ScalarType::U8;
    query.layout = PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip;
    query.quantization = QuantizationKind::BlockedAffine;
    query.state_kind = StateKind::KeyCache;
    query.state_format = StateFormatKind::GlobalContiguous;
    query.rank = 2;
    query.alignment = 128;
    query.head_dimension = 64;
    query.batch_rows = 1;
    query.block_elements = 256;
    query.block_bytes = 64;
    query.metal_weight_format_mask = format;
    query.metal_dense_token_pattern = true;
    query.metal_column_grouped_affine_u2_skip_256 = true;

    RuntimeCapabilities capabilities;
    capabilities.metal_device = true;
    capabilities.metal_library = true;
    capabilities.metal_pipeline = true;
    capabilities.global_fp32_kv = true;
    capabilities.transactional_state = true;
    const KernelSelection unavailable = select_kernel(query, request(), capabilities, {descriptor});
    CHECK(std::holds_alternative<CompatibilityReport>(unavailable));
    if (const auto* report = std::get_if<CompatibilityReport>(&unavailable))
        CHECK(report->code == CompatibilityError::CAPABILITY_MISSING);

    capabilities.metal_column_grouped_affine_u2_skip_256 = true;
    const KernelSelection admitted = select_kernel(query, request(), capabilities, {descriptor});
    CHECK(std::holds_alternative<KernelDescriptor>(admitted));
    if (const auto* selected = std::get_if<KernelDescriptor>(&admitted))
        CHECK(selected->pattern.layout.exact == PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip);

    const KernelSelection builtin = select_kernel(
        query, request(), capabilities, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<KernelDescriptor>(builtin));
    if (const auto* selected = std::get_if<KernelDescriptor>(&builtin)) {
        CHECK(selected->pattern.allowed_metal_weight_formats == format);
        CHECK(selected->pattern.require_metal_column_grouped_affine_u2_skip_256);
    }

    KernelQuery mixed = query;
    mixed.storage_type = static_cast<ScalarType>(0);
    mixed.layout = static_cast<PhysicalLayoutKind>(0);
    mixed.quantization = static_cast<QuantizationKind>(0);
    mixed.block_elements = 0;
    mixed.block_bytes = 0;
    mixed.metal_weight_format_mask = MetalWeightFormatQ4K | MetalWeightFormatQ6K | format;
    const KernelSelection mixed_builtin = select_kernel(
        mixed, request(), capabilities, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<KernelDescriptor>(mixed_builtin));
    if (const auto* selected = std::get_if<KernelDescriptor>(&mixed_builtin)) {
        CHECK(selected->pattern.require_mixed_metal_weight_formats);
        CHECK(selected->pattern.require_metal_column_grouped_affine_u2_skip_256);
        CHECK((selected->pattern.allowed_metal_weight_formats &
               mixed.metal_weight_format_mask) == mixed.metal_weight_format_mask);
    }

    capabilities.metal_column_grouped_affine_u2_skip_256 = false;
    const KernelSelection missing_mixed_capability = select_kernel(
        mixed, request(), capabilities, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(missing_mixed_capability));
    capabilities.metal_column_grouped_affine_u2_skip_256 = true;

    query.layout = PhysicalLayoutKind::GroupedAffine;
    const KernelSelection legacy_layout = select_kernel(query, request(), capabilities, {descriptor});
    CHECK(std::holds_alternative<CompatibilityReport>(legacy_layout));
    if (const auto* report = std::get_if<CompatibilityReport>(&legacy_layout))
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);

    query.layout = PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip;
    query.metal_column_grouped_affine_u2_skip_256 = false;
    const KernelSelection missing_marker = select_kernel(query, request(), capabilities, {descriptor});
    CHECK(std::holds_alternative<CompatibilityReport>(missing_marker));
    if (const auto* report = std::get_if<CompatibilityReport>(&missing_marker))
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);

    query.metal_column_grouped_affine_u2_skip_256 = true;
    query.block_bytes = 65;
    const KernelSelection malformed_tuple = select_kernel(query, request(), capabilities, {descriptor});
    CHECK(std::holds_alternative<CompatibilityReport>(malformed_tuple));
    if (const auto* report = std::get_if<CompatibilityReport>(&malformed_tuple))
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
}

void test_f16_prefill_batch_descriptor_is_exactly_two_rows() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    auto batch_request = request();
    batch_request.max_batch = 2;
    KernelQuery candidate;
    candidate.operation = OperatorKind::CausalAttention;
    candidate.semantic_version = 1;
    candidate.phase = ExecutionPhase::Prefill;
    candidate.logical_type = ScalarType::F32;
    candidate.storage_type = ScalarType::F16;
    candidate.layout = PhysicalLayoutKind::ContiguousRowMajor;
    candidate.quantization = QuantizationKind::None;
    candidate.state_kind = StateKind::KeyCache;
    candidate.state_format = StateFormatKind::GlobalContiguous;
    candidate.rank = 3;
    candidate.alignment = 64;
    candidate.head_dimension = 64;
    candidate.batch_rows = 2;
    candidate.metal_weight_format_mask = MetalWeightFormatF16;
    candidate.metal_dense_token_pattern = true;
    candidate.metal_dense_prefill_batch_pattern = true;
    auto selected = select_kernel(candidate, batch_request, metal,
                                  builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<KernelDescriptor>(selected));
    if (const auto* descriptor = std::get_if<KernelDescriptor>(&selected)) {
        CHECK(descriptor->implementation == KernelImplementation::MetalDensePrefillBatch);
        CHECK(descriptor->effects.fallback == FallbackKind::None);
        CHECK(descriptor->pattern.batch_rows.minimum == 2);
        CHECK(descriptor->pattern.batch_rows.maximum == 2);
    }

    candidate.batch_rows = 3;
    candidate.metal_dense_prefill_batch_pattern = false;
    batch_request.max_batch = 3;
    selected = select_kernel(candidate, batch_request, metal,
                             builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(selected));
    if (const auto* report = std::get_if<CompatibilityReport>(&selected)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
    }

    candidate.batch_rows = 1;
    candidate.phase = ExecutionPhase::Prefill;
    batch_request.max_batch = 1;
    selected = select_kernel(candidate, batch_request, metal,
                              builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<KernelDescriptor>(selected));
    if (const auto* descriptor = std::get_if<KernelDescriptor>(&selected)) {
        CHECK(descriptor->implementation == KernelImplementation::MetalDenseToken);
    }

    candidate.batch_rows = 2;
    candidate.phase = ExecutionPhase::Prefill;
    candidate.alignment = 32;
    candidate.metal_dense_prefill_batch_pattern = true;
    batch_request.max_batch = 2;
    selected = select_kernel(candidate, batch_request, metal,
                             builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(selected));

    candidate.alignment = 64;
    candidate.storage_type = ScalarType::U8;
    candidate.layout = PhysicalLayoutKind::GgufBlocked;
    candidate.quantization = QuantizationKind::BlockedAffine;
    candidate.block_elements = 256;
    candidate.block_bytes = 144;
    candidate.metal_weight_format_mask = MetalWeightFormatQ4K;
    selected = select_kernel(candidate, batch_request, metal,
                             builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(selected));

    candidate.block_bytes = 210;
    candidate.metal_weight_format_mask = MetalWeightFormatQ6K;
    selected = select_kernel(candidate, batch_request, metal,
                             builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(selected));
}

void test_canonical_planner_materializes_q4k_dense_entries() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    auto planned = plan_canonical_metal_dense(q4k_dense_model(), request(), metal,
                                              builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(planned));
    if (const auto* plan = std::get_if<ExecutionPlan>(&planned)) {
        CHECK(plan->entries.size() == 2);
        for (const PlanEntry& entry : plan->entries) {
            CHECK(entry.descriptor.implementation == KernelImplementation::MetalDenseToken);
            CHECK(entry.descriptor.pattern.allowed_metal_weight_formats == MetalWeightFormatQ4K);
            CHECK(plan_entry_matches(q4k_dense_model(), entry));
        }
    }
}

void test_canonical_planner_admits_flattened_attention_context() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    SemanticModel model = q4k_dense_model();
    model.values[3].dimensions = {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 256}};
    const auto planned = plan_canonical_metal_dense(model, request(), metal,
                                                    builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(planned));
    if (const auto* plan = std::get_if<ExecutionPlan>(&planned)) {
        CHECK(plan->entries.size() == 2);
        for (const PlanEntry& entry : plan->entries) CHECK(plan_entry_matches(model, entry));
    }
}

void test_canonical_planner_rejects_sliding_attention_until_windowed_metal_exists() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    SemanticModel model = q4k_dense_model();
    const auto attention = std::find_if(model.operators.begin(), model.operators.end(),
                                        [](const SemanticOperator& op) {
                                            return op.kind == OperatorKind::CausalAttention;
                                        });
    CHECK(attention != model.operators.end());
    if (attention == model.operators.end()) return;
    auto* payload = std::get_if<CausalAttentionPayload>(&attention->payload);
    CHECK(payload != nullptr);
    if (!payload) return;
    payload->window = AttentionWindowKind::Sliding;
    payload->window_tokens = 128;
    DenseGraphWitness witness;
    CHECK(!match_canonical_dense_operator_edges(model, model.layers.front(), witness));
    const auto planned = plan_canonical_metal_dense(model, request(), metal,
                                                    builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(planned));
    if (const auto* report = std::get_if<CompatibilityReport>(&planned)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
        CHECK(report->detail.find("canonical Metal attention requires a global window") !=
              std::string::npos);
        CHECK(report->detail.find("operator " + std::to_string(attention->id)) !=
              std::string::npos);
    }
}

void test_canonical_planner_rejects_invalid_rope_and_rms_contracts() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    const auto registry = builtin_canonical_metal_registry();
    const auto rejected = [&](SemanticModel model) {
        DenseGraphWitness witness;
        CHECK(!match_canonical_dense_operator_edges(model, model.layers.front(), witness));
        CHECK(std::holds_alternative<CompatibilityReport>(
            plan_canonical_metal_dense(model, request(), metal, registry)));
    };

    SemanticModel zero_base = q4k_dense_model();
    auto rope = std::find_if(zero_base.operators.begin(), zero_base.operators.end(),
                             [](const SemanticOperator& op) {
                                 return op.kind == OperatorKind::Rope;
                             });
    CHECK(rope != zero_base.operators.end());
    if (rope != zero_base.operators.end()) {
        auto* payload = std::get_if<RopePayload>(&rope->payload);
        CHECK(payload != nullptr);
        if (payload) payload->base_f32_bits = 0;
    }
    rejected(std::move(zero_base));

    SemanticModel oversized_rotary = q4k_dense_model();
    rope = std::find_if(oversized_rotary.operators.begin(), oversized_rotary.operators.end(),
                        [](const SemanticOperator& op) {
                            return op.kind == OperatorKind::Rope;
                        });
    CHECK(rope != oversized_rotary.operators.end());
    if (rope != oversized_rotary.operators.end()) {
        auto* payload = std::get_if<RopePayload>(&rope->payload);
        CHECK(payload != nullptr);
        if (payload) {
            payload->rotary_dimension = 128;
            payload->frequency_dimension = 128;
        }
    }
    rejected(std::move(oversized_rotary));

    SemanticModel negative_rms = q4k_dense_model();
    for (SemanticOperator& op : negative_rms.operators) {
        if (auto* payload = std::get_if<RmsNormPayload>(&op.payload))
            payload->epsilon_f32_bits = 0xbf800000u;
    }
    rejected(std::move(negative_rms));

    SemanticModel transposed_linear = q4k_dense_model();
    for (SemanticOperator& op : transposed_linear.operators) {
        if (auto* payload = std::get_if<LinearPayload>(&op.payload)) {
            payload->transpose_weight = true;
            break;
        }
    }
    rejected(std::move(transposed_linear));

    SemanticModel mismatched_rms = q4k_dense_model();
    uint32_t rms_index = 0;
    for (SemanticOperator& op : mismatched_rms.operators) {
        if (auto* payload = std::get_if<RmsNormPayload>(&op.payload)) {
            if (rms_index++ == 1) payload->epsilon_f32_bits = 0x3a83126fu;
        }
    }
    rejected(std::move(mismatched_rms));

    SemanticModel scaled_rope = q4k_dense_model();
    for (SemanticOperator& op : scaled_rope.operators) {
        if (auto* payload = std::get_if<RopePayload>(&op.payload)) {
            payload->scale_f32_bits = 0x40000000u;
            break;
        }
    }
    rejected(std::move(scaled_rope));

    for (uint32_t scale_bits : {0u, 0xbf800000u, 0x7f800000u, 0x7fc00000u}) {
        SemanticModel invalid_attention_scale = q4k_dense_model();
        const auto attention = std::find_if(
            invalid_attention_scale.operators.begin(),
            invalid_attention_scale.operators.end(),
            [](const SemanticOperator& op) {
                return op.kind == OperatorKind::CausalAttention;
            });
        CHECK(attention != invalid_attention_scale.operators.end());
        if (attention != invalid_attention_scale.operators.end()) {
            auto* payload = std::get_if<CausalAttentionPayload>(&attention->payload);
            CHECK(payload != nullptr);
            if (payload) payload->scale_f32_bits = scale_bits;
        }
        rejected(std::move(invalid_attention_scale));
    }

    const auto with_query_key_norms = [] {
        SemanticModel model = q4k_dense_model();
        model.tensors.push_back(dense_tensor(
            12, TensorRole::AttentionQueryNormWeight,
            {{DimensionKind::Constant, 256}}, ScalarType::F32));
        model.tensors.push_back(dense_tensor(
            13, TensorRole::AttentionKeyNormWeight,
            {{DimensionKind::Constant, 256}}, ScalarType::F32));
        model.values.push_back({19, ScalarType::F32,
                                {{DimensionKind::Symbol, 1},
                                 {DimensionKind::Constant, 256}}, 0});
        model.values.push_back({20, ScalarType::F32,
                                {{DimensionKind::Symbol, 1},
                                 {DimensionKind::Constant, 256}}, 0});
        const auto rope = std::find_if(
            model.operators.begin(), model.operators.end(),
            [](const SemanticOperator& op) { return op.kind == OperatorKind::Rope; });
        CHECK(rope != model.operators.end());
        if (rope == model.operators.end()) return model;
        const size_t rope_index = static_cast<size_t>(rope - model.operators.begin());
        SemanticOperator query_norm;
        query_norm.kind = OperatorKind::RmsNorm;
        query_norm.inputs = {2};
        query_norm.outputs = {19};
        query_norm.tensors = {12};
        query_norm.payload = RmsNormPayload{0x358637bdu, -1, 1};
        SemanticOperator key_norm = query_norm;
        key_norm.inputs = {3};
        key_norm.outputs = {20};
        key_norm.tensors = {13};
        model.operators.insert(model.operators.begin() + rope_index,
                               std::move(query_norm));
        model.operators.insert(model.operators.begin() + rope_index + 1,
                               std::move(key_norm));
        model.operators[rope_index + 2].inputs = {19, 20};
        for (uint32_t id = 0; id != model.operators.size(); ++id) {
            model.operators[id].id = id;
        }
        model.layers.front().operator_count += 2;
        return model;
    };

    SemanticModel normalized_query_key = with_query_key_norms();
    DenseGraphWitness normalized_witness;
    CHECK(match_canonical_dense_operator_edges(
        normalized_query_key, normalized_query_key.layers.front(), normalized_witness));
    for (uint32_t epsilon_bits : {0xbf800000u, 0x7f800000u, 0x7fc00000u}) {
        SemanticModel invalid_query_key_norm = with_query_key_norms();
        const auto query_norm = std::find_if(
            invalid_query_key_norm.operators.begin(),
            invalid_query_key_norm.operators.end(),
            [&](const SemanticOperator& op) {
                return op.kind == OperatorKind::RmsNorm && !op.tensors.empty() &&
                       invalid_query_key_norm.tensors[op.tensors[0]].role ==
                           TensorRole::AttentionQueryNormWeight;
            });
        CHECK(query_norm != invalid_query_key_norm.operators.end());
        if (query_norm != invalid_query_key_norm.operators.end()) {
            auto* payload = std::get_if<RmsNormPayload>(&query_norm->payload);
            CHECK(payload != nullptr);
            if (payload) payload->epsilon_f32_bits = epsilon_bits;
        }
        rejected(std::move(invalid_query_key_norm));
    }
}

void test_canonical_planner_materializes_query_gate_dense_entries() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    const SemanticModel model = q4k_query_gate_dense_model();
    auto planned = plan_canonical_metal_dense(model, request(), metal,
                                              builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(planned));
    if (const auto* plan = std::get_if<ExecutionPlan>(&planned)) {
        CHECK(plan->entries.size() == 2);
        for (const PlanEntry& entry : plan->entries) {
            CHECK(entry.descriptor.implementation == KernelImplementation::MetalDenseToken);
            CHECK(entry.descriptor.pattern.semantic_version.minimum == 4);
            CHECK(entry.descriptor.pattern.semantic_version.maximum == 4);
            CHECK(plan_entry_matches(model, entry));
        }
    }

    SemanticModel malformed = model;
    const auto gated = std::find_if(malformed.operators.begin(), malformed.operators.end(), [](const SemanticOperator& op) {
        return op.kind == OperatorKind::GatedAttention;
    });
    CHECK(gated != malformed.operators.end());
    gated->inputs[1] = 6; // Query is not the declared gate output.
    const auto rejected = plan_canonical_metal_dense(malformed, request(), metal,
                                                     builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    if (const auto* report = std::get_if<CompatibilityReport>(&rejected)) {
        CHECK(report->code == CompatibilityError::IR_REFERENCE_INVALID);
    }
}

void test_canonical_planner_materializes_mixed_quantized_dense_entries() {
    SemanticModel model = q4k_dense_model();
    const auto set_format = [&](uint32_t tensor_id, uint32_t elements, uint32_t bytes) {
        SemanticTensor& tensor = model.tensors[tensor_id];
        tensor.layout.block_elements = elements;
        tensor.layout.block_bytes = bytes;
        tensor.quantization.block_elements = elements;
        tensor.quantization.block_bytes = bytes;
        tensor.quantization.group_size = elements;
        tensor.planes[0].length = 256ull * 256 / elements * bytes;
    };
    set_format(2, 32, 18);   // Query Q4_0.
    set_format(3, 256, 210); // Key Q6_K.
    set_format(4, 32, 34);   // Value Q8_0.
    set_format(8, 256, 84);  // FFN up Q2_K.
    set_format(9, 32, 22);   // FFN down Q5_0.

    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    auto planned = plan_canonical_metal_dense(model, request(), metal,
                                              builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(planned));
    if (const auto* plan = std::get_if<ExecutionPlan>(&planned)) {
        const uint32_t expected_formats = MetalWeightFormatQ4_0 | MetalWeightFormatQ4K | MetalWeightFormatQ5_0 |
                                          MetalWeightFormatQ6K | MetalWeightFormatQ8_0 |
                                          MetalWeightFormatQ2K | MetalWeightFormatIQ2XXS;
        CHECK(plan->entries.size() == 2);
        for (const PlanEntry& entry : plan->entries) {
            CHECK(entry.descriptor.implementation == KernelImplementation::MetalDenseToken);
            CHECK(entry.descriptor.pattern.allowed_metal_weight_formats == expected_formats);
            CHECK(plan_entry_matches(model, entry));
        }
    }

    SemanticModel malformed = model;
    malformed.tensors[8].layout.block_bytes = 85;
    malformed.tensors[8].quantization.block_bytes = 85;
    const auto rejected = plan_canonical_metal_dense(malformed, request(), metal,
                                                      builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
}

void test_canonical_planner_materializes_recurrent_q4k_q6k_entries() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    SemanticModel model = recurrent_q4k_q6k_model();
    model.tensors[3].layout.block_bytes = 84;
    model.tensors[3].quantization.block_bytes = 84;
    model.tensors[3].planes[0].length = model.tensors[3].dimensions[0].constant_or_symbol *
                                        model.tensors[3].dimensions[1].constant_or_symbol / 256 * 84;
    auto planned = plan_canonical_metal_recurrent(model, request(), metal,
                                                  builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(planned));
    if (const auto* plan = std::get_if<ExecutionPlan>(&planned)) {
        CHECK(plan->entries.size() == 2);
        for (const PlanEntry& entry : plan->entries) {
            CHECK(entry.descriptor.implementation == KernelImplementation::MetalRecurrentToken);
            CHECK(entry.descriptor.pattern.allowed_metal_weight_formats ==
                  (MetalWeightFormatQ4K | MetalWeightFormatQ6K | MetalWeightFormatQ2K |
                   MetalWeightFormatIQ2XXS));
            CHECK(plan_entry_matches(model, entry));
        }
    }

    SemanticModel reordered_l2 = model;
    std::vector<size_t> l2_indices;
    for (size_t index = 0; index != reordered_l2.operators.size(); ++index) {
        if (reordered_l2.operators[index].kind == OperatorKind::L2Normalize)
            l2_indices.push_back(index);
    }
    CHECK(l2_indices.size() == 2);
    if (l2_indices.size() == 2) {
        std::swap(reordered_l2.operators[l2_indices[0]],
                  reordered_l2.operators[l2_indices[1]]);
        auto reordered_plan = plan_canonical_metal_recurrent(
            reordered_l2, request(), metal, builtin_canonical_metal_registry());
        CHECK(std::holds_alternative<ExecutionPlan>(reordered_plan));
    }

    SemanticModel malformed = model;
    malformed.tensors[2].layout.block_bytes = 209;
    malformed.tensors[2].quantization.block_bytes = 209;
    auto rejected = plan_canonical_metal_recurrent(malformed, request(), metal,
                                                   builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    if (const auto* report = std::get_if<CompatibilityReport>(&rejected)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
    }

    SemanticModel missing_l2 = model;
    missing_l2.operators.erase(missing_l2.operators.begin() + 5);
    for (uint32_t id = 0; id != missing_l2.operators.size(); ++id) missing_l2.operators[id].id = id;
    missing_l2.layers[0].operator_count = static_cast<uint32_t>(missing_l2.operators.size());
    auto no_l2_plan = plan_canonical_metal_recurrent(missing_l2, request(), metal,
                                                      builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(no_l2_plan));
}

void test_canonical_planner_admits_uniform_recurrent_q4k_and_q6k() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    const auto set_format = [](SemanticTensor& tensor, uint32_t bytes) {
        tensor.layout.block_bytes = bytes;
        tensor.quantization.block_bytes = bytes;
        tensor.planes[0].length = tensor.dimensions[0].constant_or_symbol *
                                  tensor.dimensions[1].constant_or_symbol / 256 * bytes;
    };

    SemanticModel q4k = recurrent_q4k_q6k_model();
    set_format(q4k.tensors[3], 144);
    set_format(q4k.tensors[9], 144);
    auto q4k_plan = plan_canonical_metal_recurrent(q4k, request(), metal,
                                                    builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(q4k_plan));
    if (const auto* plan = std::get_if<ExecutionPlan>(&q4k_plan)) {
        CHECK(plan->entries.size() == 2);
        for (const PlanEntry& entry : plan->entries) {
            CHECK(entry.descriptor.pattern.allowed_metal_weight_formats == MetalWeightFormatQ4K);
            CHECK(plan_entry_matches(q4k, entry));
        }
    }

    SemanticModel q6k = recurrent_q4k_q6k_model();
    for (uint32_t tensor_id : {1u, 2u, 3u, 4u, 9u, 11u, 12u, 13u}) set_format(q6k.tensors[tensor_id], 210);
    auto q6k_plan = plan_canonical_metal_recurrent(q6k, request(), metal,
                                                    builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(q6k_plan));
    if (const auto* plan = std::get_if<ExecutionPlan>(&q6k_plan)) {
        CHECK(plan->entries.size() == 2);
        for (const PlanEntry& entry : plan->entries) {
            CHECK(entry.descriptor.pattern.allowed_metal_weight_formats == MetalWeightFormatQ6K);
            CHECK(plan_entry_matches(q6k, entry));
        }
    }
}

void test_canonical_planner_materializes_schema4_recurrent_entries() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;

    SemanticModel model = recurrent_q4k_q6k_model();
    model.schema_major = 4;
    model.opset_major = 4;
    for (SemanticOperator& op : model.operators) op.semantic_version = 4;
    for (SemanticState& state : model.states) state.semantic_version = 4;

    auto planned = plan_canonical_metal_recurrent(model, request(), metal,
                                                  builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(planned));
    if (const auto* plan = std::get_if<ExecutionPlan>(&planned)) {
        CHECK(plan->entries.size() == 2);
        for (const PlanEntry& entry : plan->entries) {
            CHECK(entry.descriptor.implementation == KernelImplementation::MetalRecurrentToken);
            CHECK(entry.descriptor.pattern.semantic_version.minimum == 4);
            CHECK(entry.descriptor.pattern.semantic_version.maximum == 4);
            CHECK(plan_entry_matches(model, entry));
        }
    }

    SemanticModel mismatched_state = model;
    mismatched_state.states[0].semantic_version = 3;
    auto rejected = plan_canonical_metal_recurrent(mismatched_state, request(), metal,
                                                   builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    if (const auto* report = std::get_if<CompatibilityReport>(&rejected)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
    }
}

void test_canonical_planner_materializes_mixed_dense_recurrent_entries() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    const SemanticModel model = mixed_query_gate_recurrent_model();
    const auto planned = plan_canonical_metal(model, request(), metal, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(planned));
    if (const auto* plan = std::get_if<ExecutionPlan>(&planned)) {
        CHECK(plan->entries.size() == 4);
        CHECK(plan->entries[0].descriptor.implementation == KernelImplementation::MetalDenseToken);
        CHECK(plan->entries[1].descriptor.implementation == KernelImplementation::MetalRecurrentToken);
        CHECK(plan->entries[2].descriptor.implementation == KernelImplementation::MetalDenseToken);
        CHECK(plan->entries[3].descriptor.implementation == KernelImplementation::MetalRecurrentToken);
        for (const PlanEntry& entry : plan->entries) CHECK(plan_entry_matches(model, entry));
    }

    SemanticModel mismatched = model;
    mismatched.states.back().semantic_version = 3;
    const auto rejected = plan_canonical_metal(mismatched, request(), metal, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    if (const auto* report = std::get_if<CompatibilityReport>(&rejected)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
        CHECK(report->detail == "canonical Metal layer does not match an admitted full-token pattern");
    }
}

void test_canonical_planner_rejects_a_layer_that_bypasses_the_previous_residual() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;

    SemanticModel model = mixed_query_gate_recurrent_model();
    CHECK(model.layers.size() == 2);
    if (model.layers.size() != 2) return;
    const SemanticLayer& first = model.layers[0];
    const SemanticLayer& second = model.layers[1];
    CHECK(first.first_operator < model.operators.size());
    CHECK(second.first_operator < model.operators.size());
    if (first.first_operator >= model.operators.size() ||
        second.first_operator >= model.operators.size()) return;

    const uint32_t original_residual = model.operators[first.first_operator].inputs.front();
    const uint32_t chained_residual = model.operators[second.first_operator].inputs.front();
    for (uint32_t offset = 0; offset != second.operator_count; ++offset) {
        SemanticOperator& op = model.operators[second.first_operator + offset];
        for (uint32_t& input : op.inputs) {
            if (input == chained_residual) input = original_residual;
        }
    }
    const auto planned = plan_canonical_metal(
        model, request(), metal, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(planned));
    if (const auto* report = std::get_if<CompatibilityReport>(&planned)) {
        CHECK(report->code == CompatibilityError::IR_REFERENCE_INVALID);
        CHECK(report->detail.find("layer residual chain") != std::string::npos);
    }
}

void test_canonical_planner_rejects_a_graph_spine_that_differs_from_metal() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    const auto registry = builtin_canonical_metal_registry();

    const SemanticModel valid = q4k_dense_model();
    CHECK(std::holds_alternative<ExecutionPlan>(
        plan_canonical_metal(valid, request(), metal, registry)));

    SemanticModel wrong_embedding = valid;
    wrong_embedding.operators.front().outputs = {1};
    CHECK(std::holds_alternative<CompatibilityReport>(
        plan_canonical_metal(wrong_embedding, request(), metal, registry)));

    SemanticModel wrong_final_norm = valid;
    wrong_final_norm.operators[15].inputs = {0};
    CHECK(std::holds_alternative<CompatibilityReport>(
        plan_canonical_metal(wrong_final_norm, request(), metal, registry)));

    SemanticModel wrong_output_input = valid;
    wrong_output_input.operators[16].inputs = {15};
    CHECK(std::holds_alternative<CompatibilityReport>(
        plan_canonical_metal(wrong_output_input, request(), metal, registry)));

    SemanticModel wrong_declared_output = valid;
    wrong_declared_output.output_values_first = 16;
    CHECK(std::holds_alternative<CompatibilityReport>(
        plan_canonical_metal(wrong_declared_output, request(), metal, registry)));
}

void test_canonical_planner_skips_disabled_schema6_terminal_layer() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    const SemanticModel model = schema6_model_with_disabled_terminal_layer();
    const auto planned = plan_canonical_metal(model, request(), metal, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(planned));
    if (const auto* report = std::get_if<CompatibilityReport>(&planned)) {
        CHECK(report->code == CompatibilityError::IR_REFERENCE_INVALID);
        CHECK(report->detail.find("skip") != std::string::npos);
    }
}

void test_canonical_planner_admits_q6k_output_projection() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    SemanticModel model = q4k_dense_model();
    SemanticTensor& output = model.tensors[11];
    output.layout.kind = PhysicalLayoutKind::GgufBlocked;
    output.layout.packing = PackingKind::Gguf;
    output.layout.block_rank = 1;
    output.layout.block_elements = 256;
    output.layout.block_bytes = 210;
    output.quantization.kind = QuantizationKind::BlockedAffine;
    output.quantization.accumulation_type = ScalarType::F32;
    output.quantization.scale_type = ScalarType::F16;
    output.quantization.block_elements = 256;
    output.quantization.block_bytes = 210;
    output.quantization.group_size = 256;
    output.quantization.required_plane_mask = 1;
    output.planes[0].storage_type = ScalarType::U8;
    output.planes[0].alignment = 32;
    output.planes[0].length = 3 * 210;
    const auto planned = plan_canonical_metal(model, request(), metal, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(planned));

    output.layout.block_bytes = 144;
    output.quantization.block_bytes = 144;
    output.planes[0].length = 3 * 144;
    const auto rejected = plan_canonical_metal(model, request(), metal, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    if (const auto* report = std::get_if<CompatibilityReport>(&rejected)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
    }
}

void test_canonical_planner_admits_q6k_token_embedding() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    SemanticModel model = q4k_dense_model();
    SemanticTensor& embedding = model.tensors[0];
    embedding.layout.kind = PhysicalLayoutKind::GgufBlocked;
    embedding.layout.packing = PackingKind::Gguf;
    embedding.layout.block_rank = 1;
    embedding.layout.block_elements = 256;
    embedding.layout.block_bytes = 210;
    embedding.quantization.kind = QuantizationKind::BlockedAffine;
    embedding.quantization.accumulation_type = ScalarType::F32;
    embedding.quantization.scale_type = ScalarType::F16;
    embedding.quantization.block_elements = 256;
    embedding.quantization.block_bytes = 210;
    embedding.quantization.group_size = 256;
    embedding.quantization.required_plane_mask = 1;
    embedding.planes[0].storage_type = ScalarType::U8;
    embedding.planes[0].alignment = 32;
    embedding.planes[0].length = 3 * 210;
    const auto planned = plan_canonical_metal(model, request(), metal, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(planned));

    embedding.quantization.block_bytes = 143;
    embedding.layout.block_bytes = 143;
    embedding.planes[0].length = 3 * 143;
    const auto rejected = plan_canonical_metal(model, request(), metal, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    if (const auto* report = std::get_if<CompatibilityReport>(&rejected)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
    }
}

void test_canonical_planner_admits_q4k_token_embedding() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    SemanticModel model = q4k_dense_model();
    SemanticTensor& embedding = model.tensors[0];
    embedding.layout.kind = PhysicalLayoutKind::GgufBlocked;
    embedding.layout.packing = PackingKind::Gguf;
    embedding.layout.block_rank = 1;
    embedding.layout.block_elements = 256;
    embedding.layout.block_bytes = 144;
    embedding.quantization.kind = QuantizationKind::BlockedAffine;
    embedding.quantization.accumulation_type = ScalarType::F32;
    embedding.quantization.scale_type = ScalarType::F16;
    embedding.quantization.block_elements = 256;
    embedding.quantization.block_bytes = 144;
    embedding.quantization.group_size = 256;
    embedding.quantization.required_plane_mask = 1;
    embedding.planes[0].storage_type = ScalarType::U8;
    embedding.planes[0].alignment = 32;
    embedding.planes[0].length = 3 * 144;
    const auto planned = plan_canonical_metal(model, request(), metal, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(planned));

    embedding.quantization.block_bytes = 143;
    embedding.layout.block_bytes = 143;
    embedding.planes[0].length = 3 * 143;
    const auto rejected = plan_canonical_metal(model, request(), metal, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    if (const auto* report = std::get_if<CompatibilityReport>(&rejected)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
    }
}

void test_moe_registry_matches_generic_physical_contract() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    metal.metal_moe_router_topk = true;
    metal.metal_moe_gate_up = true;
    metal.metal_moe_down_q5_0 = true;
    metal.metal_moe_down_q8_0 = true;
    metal.metal_moe_reduce = true;

    KernelQuery query;
    query.operation = OperatorKind::CausalAttention;
    query.semantic_version = 99;
    query.phase = ExecutionPhase::Decode;
    query.logical_type = ScalarType::F32;
    query.storage_type = ScalarType::U8;
    query.layout = PhysicalLayoutKind::GgufBlocked;
    query.quantization = QuantizationKind::BlockedAffine;
    query.state_kind = StateKind::KeyCache;
    query.state_format = StateFormatKind::GlobalContiguous;
    query.rank = 2;
    query.alignment = 32;
    query.head_dimension = 256;
    query.batch_rows = 1;
    query.block_elements = 256;
    query.block_bytes = 144;
    query.metal_weight_format_mask = MetalWeightFormatF16 | MetalWeightFormatQ4K | MetalWeightFormatQ5_0;
    query.metal_moe_token_pattern = true;
    query.moe_expert_count = 128;
    query.moe_selected_count = 8;
    query.moe_gate_up_format = MetalWeightFormatQ4K;
    query.moe_down_format = MetalWeightFormatQ5_0;
    query.moe_gate_up_layout = PhysicalLayoutKind::GgufBlocked;
    query.moe_down_layout = PhysicalLayoutKind::GgufBlocked;
    query.moe_gate_up_quantization = QuantizationKind::BlockedAffine;
    query.moe_down_quantization = QuantizationKind::BlockedAffine;
    query.moe_gate_up_plane_mask = 1;
    query.moe_down_plane_mask = 1;
    query.moe_gate_up_block_elements = 256;
    query.moe_gate_up_block_bytes = 144;
    query.moe_down_block_elements = 32;
    query.moe_down_block_bytes = 22;
    query.moe_gate_up_input = 2816;
    query.moe_gate_up_output = 1408;
    query.moe_down_input = 704;
    query.moe_down_output = 2816;
    query.moe_gate_up_expert_stride = 2230272;
    query.moe_down_expert_stride = 1362944;
    query.moe_activation = ActivationKind::GeluTanh;
    query.moe_scale_source = ExpertScaleSource::PerExpertTensor;
    query.moe_value_source = ValueSource::SeparateProjection;

    const auto selected = select_kernel(query, request(), metal, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<KernelDescriptor>(selected));
    if (const auto* descriptor = std::get_if<KernelDescriptor>(&selected)) {
        CHECK(descriptor->pattern.moe_down_format.exact == MetalWeightFormatQ5_0);
        CHECK(descriptor->pattern.moe_activation.exact == ActivationKind::GeluTanh);
        CHECK(descriptor->pattern.moe_scale_source.exact == ExpertScaleSource::PerExpertTensor);
    }

    query.moe_down_format = MetalWeightFormatQ8_0;
    query.moe_down_block_bytes = 34;
    query.moe_down_expert_stride = 2106368;
    const auto q8 = select_kernel(query, request(), metal, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<KernelDescriptor>(q8));

    query.moe_down_format = MetalWeightFormatQ4K;
    query.moe_down_block_elements = 256;
    query.moe_down_block_bytes = 144;
    query.moe_down_expert_stride = 2230272;
    const auto unsupported = select_kernel(query, request(), metal, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(unsupported));
}

SemanticTensor moe_f32_matrix(uint32_t id, TensorRole role, uint32_t rows, uint32_t columns) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, rows}, {DimensionKind::Constant, columns}};
    tensor.layout.kind = PhysicalLayoutKind::ContiguousRowMajor;
    tensor.layout.rank = 2;
    tensor.layout.axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides[0] = 1;
    tensor.layout.strides[1] = columns;
    tensor.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, 0,
                      static_cast<uint64_t>(rows) * columns * sizeof(float), 64, 0}};
    return tensor;
}

SemanticTensor moe_expert_tensor(uint32_t id, TensorRole role, uint32_t experts, uint32_t input,
                                 uint32_t output, uint32_t block_elements, uint32_t block_bytes) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, experts}, {DimensionKind::Constant, input},
                         {DimensionKind::Constant, output}};
    tensor.layout.kind = PhysicalLayoutKind::GgufBlocked;
    tensor.layout.version = 1;
    tensor.layout.packing = PackingKind::Gguf;
    tensor.layout.rank = 3;
    tensor.layout.block_rank = 1;
    tensor.layout.axis_order = {1, 2, 0, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides[0] = 1;
    tensor.layout.strides[1] = input;
    tensor.layout.strides[2] = static_cast<uint64_t>(input) * output;
    tensor.layout.block_elements = block_elements;
    tensor.layout.block_bytes = block_bytes;
    tensor.quantization.kind = QuantizationKind::BlockedAffine;
    tensor.quantization.accumulation_type = ScalarType::F32;
    tensor.quantization.block_elements = block_elements;
    tensor.quantization.block_bytes = block_bytes;
    tensor.quantization.group_size = block_elements;
    tensor.quantization.required_plane_mask = 1;
    tensor.expert_axis = {ExpertAxisKind::ExpertBank, 0, 0xff, 1, 2, experts,
                          static_cast<uint64_t>(input) * output / block_elements * block_bytes, 0};
    tensor.planes = {{PlaneKind::Values, ScalarType::U8, ArtifactId{0}, 0,
                      static_cast<uint64_t>(experts) * tensor.expert_axis.per_expert_byte_stride, 64, 0}};
    return tensor;
}

SemanticModel generic_moe_planner_model(uint32_t down_block_bytes = 22, bool q8_down = false,
                                        bool add_extra_scale = false, bool duplicate_router = false,
                                        bool permute_subgraph = false, bool key_alias = false) {
    constexpr uint32_t hidden = 256;
    constexpr uint32_t experts = 4;
    constexpr uint32_t selected = 2;
    constexpr uint32_t expert_intermediate = 256;
    SemanticModel model = q4k_dense_model();
    const uint32_t value_base = static_cast<uint32_t>(model.values.size());
    // Keep the dense attention/FFN branch intact and replace only its final
    // residual with the explicit dense/MoE branch merge below.  The dense
    // matcher is deliberately relaxed for this fixture because MoE admission
    // owns the larger graph witness.
    model.schema_major = 7;
    model.opset_major = 7;
    for (SemanticOperator& op : model.operators) op.semantic_version = 7;
    for (SemanticState& state : model.states) state.semantic_version = 7;
    model.tensors.push_back(dense_tensor(12, TensorRole::NextnEmbeddingNormWeight,
                                         {{DimensionKind::Constant, hidden}}, ScalarType::F32));
    model.tensors.push_back(moe_f32_matrix(13, TensorRole::NextnProjectionWeight, experts, hidden));
    model.tensors.push_back(moe_expert_tensor(14, TensorRole::FfnUpWeight, experts, hidden,
                                              2 * expert_intermediate, 256, 144));
    model.tensors.push_back(moe_expert_tensor(15, TensorRole::FfnDownWeight, experts, expert_intermediate,
                                              hidden, q8_down ? 32 : 32, down_block_bytes));
    model.tensors.push_back(dense_tensor(16, TensorRole::NextnEmbeddingNormWeight,
                                         {{DimensionKind::Constant, experts}}, ScalarType::F32));
    model.tensors.push_back(dense_tensor(17, TensorRole::NextnHiddenNormWeight,
                                         {{DimensionKind::Constant, hidden}}, ScalarType::F32));
    const auto rows = [](uint32_t width) {
        return std::vector<Dimension>{{DimensionKind::Symbol, 1}, {DimensionKind::Constant, width}};
    };
    const auto routed = [](uint32_t width) {
        return std::vector<Dimension>{{DimensionKind::Symbol, 1}, {DimensionKind::Constant, selected},
                                      {DimensionKind::Constant, width}};
    };
    model.values.push_back({value_base + 0, ScalarType::F32, rows(hidden), 0}); // router norm
    model.values.push_back({value_base + 1, ScalarType::F32, rows(hidden), 0}); // route scale
    model.values.push_back({value_base + 2, ScalarType::F32, rows(hidden), 0}); // H^-1/2 scale
    model.values.push_back({value_base + 3, ScalarType::F32, rows(experts), 0}); // router logits
    model.values.push_back({value_base + 4, ScalarType::U32, rows(selected), 0}); // expert IDs
    model.values.push_back({value_base + 5, ScalarType::F32, rows(selected), 0}); // expert weights
    model.values.push_back({value_base + 6, ScalarType::F32, rows(hidden), 0}); // expert norm
    model.values.push_back({value_base + 7, ScalarType::F32, routed(2 * expert_intermediate), 0});
    model.values.push_back({value_base + 8, ScalarType::F32, routed(expert_intermediate), 0});
    model.values.push_back({value_base + 9, ScalarType::F32, routed(expert_intermediate), 0});
    model.values.push_back({value_base + 10, ScalarType::F32, routed(expert_intermediate), 0});
    model.values.push_back({value_base + 11, ScalarType::F32, routed(hidden), 0});
    model.values.push_back({value_base + 12, ScalarType::F32, rows(hidden), 0}); // branch merge
    model.values.push_back({value_base + 13, ScalarType::F32, rows(hidden), 0}); // final add

    const auto make = [](uint32_t id, OperatorKind kind, std::vector<uint32_t> inputs,
                         std::vector<uint32_t> outputs, std::vector<uint32_t> tensors,
                         OperatorPayload payload) {
        SemanticOperator op;
        op.id = id;
        op.kind = kind;
        op.semantic_version = 7;
        op.inputs = std::move(inputs);
        op.outputs = std::move(outputs);
        op.tensors = std::move(tensors);
        op.payload = std::move(payload);
        return op;
    };
    RouterTopKPayload router;
    router.expert_count = experts;
    router.selected_count = selected;
    router.normalization_order = RouterNormalizationOrder::NormalizeThenSelect;
    router.selected_weight_normalization =
        SelectedWeightNormalization::RenormalizeSelectedProbabilities;
    const uint32_t router_norm_value = value_base + 0;
    const uint32_t route_scale_value = value_base + 1;
    const uint32_t normalized_route_value = value_base + 2;
    const uint32_t router_logits = value_base + 3;
    const uint32_t expert_ids = value_base + 4;
    const uint32_t expert_weights = value_base + 5;
    const uint32_t expert_norm_value = value_base + 6;
    const uint32_t expert_up_value = value_base + 7;
    const uint32_t expert_gate_value = value_base + 8;
    const uint32_t expert_up_split_value = value_base + 9;
    const uint32_t expert_act_value = value_base + 10;
    const uint32_t expert_down_value = value_base + 11;
    const uint32_t moe_reduce_value = value_base + 12;
    const uint32_t branch_merge_value = 15;
    const uint32_t final_add_value = value_base + 13;
    model.operators[14].inputs = {14, moe_reduce_value};
    model.operators[14].outputs = {branch_merge_value};
    model.operators[15].inputs = {final_add_value};
    std::vector<SemanticOperator> tail = {
        make(0, OperatorKind::RmsNorm, {9}, {router_norm_value}, {}, RmsNormPayload{0x358637bdu, -1, 0}),
        make(0, OperatorKind::Scale, {router_norm_value}, {route_scale_value}, {12}, ScalePayload{ScaleSource::Tensor, 0}),
        make(0, OperatorKind::Scale, {route_scale_value}, {normalized_route_value}, {}, ScalePayload{ScaleSource::LiteralF32, 0x3d800000u}),
        make(0, OperatorKind::Linear, {normalized_route_value}, {router_logits}, {13}, LinearPayload{}),
        make(0, OperatorKind::RouterTopK, {router_logits}, {expert_ids, expert_weights}, {}, router),
        make(0, OperatorKind::RmsNorm, {9}, {expert_norm_value}, {17}, RmsNormPayload{0x358637bdu, -1, 1}),
        make(0, OperatorKind::RoutedLinear, {expert_norm_value, expert_ids, expert_weights}, {expert_up_value}, {14}, RoutedLinearPayload{ScalarType::F32}),
        make(0, OperatorKind::AxisSplit, {expert_up_value}, {expert_gate_value, expert_up_split_value}, {}, AxisSplitPayload{expert_intermediate, expert_intermediate}),
        make(0, OperatorKind::GatedActivation, {expert_gate_value, expert_up_split_value}, {expert_act_value}, {}, GatedActivationPayload{ActivationKind::GeluTanh}),
        make(0, OperatorKind::RoutedLinear, {expert_act_value, expert_ids, expert_weights}, {expert_down_value}, {15}, RoutedLinearPayload{ScalarType::F32}),
        make(0, OperatorKind::WeightedExpertReduce, {expert_down_value, expert_ids, expert_weights}, {moe_reduce_value}, {16},
             WeightedExpertReducePayload{ExpertReduceAssociation::SelectedOrderLeftToRight,
                                         ExpertScaleSource::PerExpertTensor, ScalarType::F32}),
        make(0, OperatorKind::Add, {9, branch_merge_value}, {final_add_value}, {}, AddPayload{}),
    };
    if (add_extra_scale) tail.push_back(make(0, OperatorKind::Scale, {9}, {final_add_value + 1}, {},
                                             ScalePayload{ScaleSource::LiteralF32, 0x3f800000u}));
    if (duplicate_router) tail.push_back(make(0, OperatorKind::RouterTopK, {router_logits}, {expert_ids, expert_weights}, {}, router));
    if (permute_subgraph) std::rotate(tail.begin(), tail.begin() + 2, tail.end());
    const size_t tail_begin = 15;
    model.operators.insert(model.operators.begin() + static_cast<ptrdiff_t>(tail_begin), tail.begin(), tail.end());
    for (uint32_t id = 0; id != model.operators.size(); ++id) model.operators[id].id = id;
    model.layers[0].operator_count = static_cast<uint32_t>(14 + tail.size());
    if (key_alias) {
        model.tensors[4].role = TensorRole::NextnProjectionWeight;
        const auto attention = std::find_if(model.operators.begin(), model.operators.end(), [](const SemanticOperator& op) {
            return op.kind == OperatorKind::CausalAttention;
        });
        CHECK(attention != model.operators.end());
        if (attention != model.operators.end()) {
            attention->inputs = {1, 2};
            attention->states = {0};
            auto* payload = std::get_if<CausalAttentionPayload>(&attention->payload);
            CHECK(payload != nullptr);
            if (payload) {
                payload->value_source = ValueSource::KeyStateAlias;
                payload->value_source_value = 2;
            }
        }
    }
    return model;
}

RuntimeCapabilities moe_capabilities() {
    RuntimeCapabilities capabilities;
    capabilities.global_fp32_kv = true;
    capabilities.transactional_state = true;
    capabilities.metal_device = true;
    capabilities.metal_library = true;
    capabilities.metal_pipeline = true;
    capabilities.metal_moe_router_topk = true;
    capabilities.metal_moe_gate_up = true;
    capabilities.metal_moe_down_q5_0 = true;
    capabilities.metal_moe_down_q8_0 = true;
    capabilities.metal_moe_reduce = true;
    return capabilities;
}

void test_moe_planner_discovers_permuted_graph_and_formats() {
    const RuntimeCapabilities capabilities = moe_capabilities();
    const auto registry = builtin_canonical_metal_registry();
    const SemanticModel q5 = generic_moe_planner_model(22, false, false, false, true);
    const PlanResult q5_plan = plan_canonical_metal(q5, request(), capabilities, registry);
    CHECK(std::holds_alternative<ExecutionPlan>(q5_plan));
    if (const auto* plan = std::get_if<ExecutionPlan>(&q5_plan)) {
        CHECK(plan->entries.size() == 2);
        for (const PlanEntry& entry : plan->entries) {
            CHECK(entry.descriptor.pattern.require_moe_descriptor);
            CHECK(entry.descriptor.pattern.moe_down_format.exact == MetalWeightFormatQ5_0);
            CHECK(plan_entry_matches(q5, entry));
        }
    }

    const SemanticModel q8 = generic_moe_planner_model(34, true, false, false, true);
    const PlanResult q8_plan = plan_canonical_metal(q8, request(), capabilities, registry);
    CHECK(std::holds_alternative<ExecutionPlan>(q8_plan));
    if (const auto* plan = std::get_if<ExecutionPlan>(&q8_plan)) {
        CHECK(plan->entries.size() == 2);
        for (const PlanEntry& entry : plan->entries) {
            CHECK(entry.descriptor.pattern.moe_down_format.exact == MetalWeightFormatQ8_0);
            CHECK(plan_entry_matches(q8, entry));
        }
    }

    const SemanticModel alias = generic_moe_planner_model(22, false, false, false, true, true);
    const PlanResult alias_plan = plan_canonical_metal(alias, request(), capabilities, registry);
    CHECK(std::holds_alternative<CompatibilityReport>(alias_plan));
    if (const auto* report = std::get_if<CompatibilityReport>(&alias_plan))
        CHECK(report->code == CompatibilityError::IR_REFERENCE_INVALID);
}

void test_moe_planner_rejects_ambiguous_and_nonmatching_graphs() {
    const RuntimeCapabilities capabilities = moe_capabilities();
    const auto registry = builtin_canonical_metal_registry();
    const PlanResult bad_format = plan_canonical_metal(generic_moe_planner_model(144), request(), capabilities, registry);
    CHECK(std::holds_alternative<CompatibilityReport>(bad_format));
    const PlanResult duplicate_router = plan_canonical_metal(generic_moe_planner_model(22, false, false, true),
                                                             request(), capabilities, registry);
    CHECK(std::holds_alternative<CompatibilityReport>(duplicate_router));
    RuntimeCapabilities no_q8 = capabilities;
    no_q8.metal_moe_down_q8_0 = false;
    const PlanResult unavailable = plan_canonical_metal(generic_moe_planner_model(34, true), request(), no_q8, registry);
    CHECK(std::holds_alternative<CompatibilityReport>(unavailable));

    SemanticModel swapped = generic_moe_planner_model();
    auto final_add = std::find_if(swapped.operators.begin(), swapped.operators.end(), [](const SemanticOperator& op) {
        return op.kind == OperatorKind::Add && op.inputs.size() == 2 && op.inputs[0] == 9 && op.inputs[1] == 15;
    });
    CHECK(final_add != swapped.operators.end());
    if (final_add != swapped.operators.end()) {
        std::swap(final_add->inputs[0], final_add->inputs[1]);
        const PlanResult rejected_order = plan_canonical_metal(swapped, request(), capabilities, registry);
        CHECK(std::holds_alternative<CompatibilityReport>(rejected_order));
    }

    const auto operator_with_role = [](SemanticModel& model, OperatorKind kind,
                                       TensorRole role) -> SemanticOperator* {
        SemanticOperator* found = nullptr;
        for (SemanticOperator& op : model.operators) {
            if (op.kind != kind) continue;
            const bool carries = std::any_of(op.tensors.begin(), op.tensors.end(), [&](uint32_t tensor_id) {
                return tensor_id < model.tensors.size() && model.tensors[tensor_id].role == role;
            });
            if (!carries) continue;
            if (found) return nullptr;
            found = &op;
        }
        return found;
    };

    SemanticModel wrong_qkv_input = generic_moe_planner_model();
    SemanticOperator* attention_norm = operator_with_role(
        wrong_qkv_input, OperatorKind::RmsNorm, TensorRole::AttentionNormWeight);
    CHECK(attention_norm != nullptr);
    if (attention_norm && attention_norm->inputs.size() == 1) {
        const uint32_t raw_residual = attention_norm->inputs[0];
        for (TensorRole role : {TensorRole::QueryWeight, TensorRole::KeyWeight,
                                TensorRole::ValueWeight}) {
            SemanticOperator* projection = operator_with_role(
                wrong_qkv_input, OperatorKind::Linear, role);
            CHECK(projection != nullptr);
            if (projection) projection->inputs = {raw_residual};
        }
        const PlanResult rejected = plan_canonical_metal(
            wrong_qkv_input, request(), capabilities, registry);
        CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    }

    SemanticModel wrong_attention_residual = generic_moe_planner_model();
    attention_norm = operator_with_role(
        wrong_attention_residual, OperatorKind::RmsNorm, TensorRole::AttentionNormWeight);
    SemanticOperator* attention_output = operator_with_role(
        wrong_attention_residual, OperatorKind::Linear, TensorRole::AttentionOutputWeight);
    SemanticOperator* ffn_norm = operator_with_role(
        wrong_attention_residual, OperatorKind::RmsNorm, TensorRole::FfnNormWeight);
    CHECK(attention_norm != nullptr);
    CHECK(attention_output != nullptr);
    CHECK(ffn_norm != nullptr);
    if (attention_norm && attention_output && ffn_norm &&
        attention_norm->outputs.size() == 1 && attention_output->outputs.size() == 1 &&
        ffn_norm->inputs.size() == 1) {
        auto residual = std::find_if(
            wrong_attention_residual.operators.begin(), wrong_attention_residual.operators.end(),
            [&](const SemanticOperator& op) {
                return op.kind == OperatorKind::Add && op.outputs == ffn_norm->inputs;
            });
        CHECK(residual != wrong_attention_residual.operators.end());
        if (residual != wrong_attention_residual.operators.end()) {
            residual->inputs = {attention_norm->outputs[0], attention_output->outputs[0]};
            const PlanResult rejected = plan_canonical_metal(
                wrong_attention_residual, request(), capabilities, registry);
            CHECK(std::holds_alternative<CompatibilityReport>(rejected));
        }
    }
}

void test_dense_graph_witness_closes_the_executed_dag() {
    const SemanticModel model = q4k_dense_model();
    DenseGraphWitness witness;
    CHECK(match_canonical_dense_operator_edges(model, model.layers.front(), witness));
    CHECK(witness.covered_operator_ids.size() == model.layers.front().operator_count);
    CHECK(witness.attention_residual != nullptr);
    CHECK(witness.ffn_norm != nullptr);
    CHECK(witness.gate != nullptr);
    CHECK(witness.up != nullptr);
    CHECK(witness.swiglu != nullptr);
    CHECK(witness.down != nullptr);
    CHECK(witness.final_residual != nullptr);

    const SemanticModel decoupled = q4k_decoupled_query_width_dense_model();
    DenseGraphWitness decoupled_witness;
    CHECK(match_canonical_dense_operator_edges(
        decoupled, decoupled.layers.front(), decoupled_witness));
    const PlanResult decoupled_plan = plan_canonical_metal(
        decoupled, request(), moe_capabilities(),
        builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(decoupled_plan));
    SemanticModel reversed_query = decoupled;
    reversed_query.tensors[2].dimensions = {
        {DimensionKind::Constant, 1024}, {DimensionKind::Constant, 256}};
    CHECK(!match_canonical_dense_operator_edges(
        reversed_query, reversed_query.layers.front(), decoupled_witness));

    SemanticModel disconnected = model;
    SemanticOperator extra;
    extra.kind = OperatorKind::Scale;
    extra.inputs = {0};
    extra.outputs = {18};
    extra.payload = ScalePayload{ScaleSource::LiteralF32, 0x3f800000u};
    const size_t insertion = disconnected.layers.front().first_operator +
                             disconnected.layers.front().operator_count;
    disconnected.operators.insert(disconnected.operators.begin() + static_cast<ptrdiff_t>(insertion), extra);
    for (uint32_t id = 0; id != disconnected.operators.size(); ++id) disconnected.operators[id].id = id;
    ++disconnected.layers.front().operator_count;
    DenseGraphWitness rejected;
    CHECK(!match_canonical_dense_operator_edges(disconnected, disconnected.layers.front(), rejected));

    SemanticModel wrong_swiglu = model;
    auto swiglu = std::find_if(wrong_swiglu.operators.begin(), wrong_swiglu.operators.end(), [](const SemanticOperator& op) {
        return op.kind == OperatorKind::SwiGlu;
    });
    CHECK(swiglu != wrong_swiglu.operators.end());
    if (swiglu != wrong_swiglu.operators.end()) {
        swiglu->inputs[0] = 10;
        CHECK(!match_canonical_dense_operator_edges(wrong_swiglu, wrong_swiglu.layers.front(), rejected));
    }

    SemanticModel wrong_ffn_residual = model;
    auto final_add = std::find_if(wrong_ffn_residual.operators.begin(), wrong_ffn_residual.operators.end(), [](const SemanticOperator& op) {
        return op.kind == OperatorKind::Add && op.inputs == std::vector<uint32_t>{9, 14};
    });
    CHECK(final_add != wrong_ffn_residual.operators.end());
    if (final_add != wrong_ffn_residual.operators.end()) {
        final_add->inputs[0] = 0;
        CHECK(!match_canonical_dense_operator_edges(wrong_ffn_residual,
                                                    wrong_ffn_residual.layers.front(), rejected));
    }

    SemanticModel duplicate_producer = model;
    auto duplicate_add = std::find_if(duplicate_producer.operators.begin(), duplicate_producer.operators.end(), [](const SemanticOperator& op) {
        return op.kind == OperatorKind::Add && op.inputs == std::vector<uint32_t>{9, 14};
    });
    CHECK(duplicate_add != duplicate_producer.operators.end());
    if (duplicate_add != duplicate_producer.operators.end()) {
        duplicate_add->outputs[0] = 9;
        CHECK(!match_canonical_dense_operator_edges(duplicate_producer,
                                                    duplicate_producer.layers.front(), rejected));
    }

    SemanticModel self_edge = model;
    auto self_add = std::find_if(self_edge.operators.begin(), self_edge.operators.end(), [](const SemanticOperator& op) {
        return op.kind == OperatorKind::Add && op.inputs == std::vector<uint32_t>{9, 14};
    });
    CHECK(self_add != self_edge.operators.end());
    if (self_add != self_edge.operators.end()) {
        self_add->inputs[0] = self_add->outputs[0];
        CHECK(!match_canonical_dense_operator_edges(self_edge, self_edge.layers.front(), rejected));
    }

    SemanticModel truncated = model;
    --truncated.layers.front().operator_count;
    CHECK(!match_canonical_dense_operator_edges(truncated, truncated.layers.front(), rejected));

    SemanticModel wrong_gate_shape = model;
    wrong_gate_shape.tensors[7].dimensions[0].constant_or_symbol = 128;
    wrong_gate_shape.tensors[7].planes[0].length = 128ull * 256 / 256 * 144;
    CHECK(!match_canonical_dense_operator_edges(wrong_gate_shape,
                                                wrong_gate_shape.layers.front(), rejected));

    SemanticModel wrong_fused_shape = q4k_query_gate_dense_model();
    wrong_fused_shape.tensors[2].dimensions[1].constant_or_symbol = 256;
    wrong_fused_shape.tensors[2].planes[0].length = 256ull * 256 / 256 * 144;
    CHECK(!match_canonical_dense_operator_edges(wrong_fused_shape,
                                                wrong_fused_shape.layers.front(), rejected));

    SemanticModel mixed_geometry = model;
    const uint32_t value_base = static_cast<uint32_t>(mixed_geometry.values.size());
    for (const SemanticValue& value : model.values) {
        SemanticValue copy = value;
        copy.id += value_base;
        mixed_geometry.values.push_back(std::move(copy));
    }
    std::vector<SemanticOperator> second_layer;
    for (uint32_t index = model.layers.front().first_operator;
         index != model.layers.front().first_operator + model.layers.front().operator_count; ++index) {
        SemanticOperator copy = model.operators[index];
        for (uint32_t& value : copy.inputs) value += value_base;
        for (uint32_t& value : copy.outputs) value += value_base;
        second_layer.push_back(std::move(copy));
    }
    const size_t second_layer_begin = model.layers.front().first_operator +
                                      model.layers.front().operator_count;
    mixed_geometry.operators.insert(mixed_geometry.operators.begin() +
                                    static_cast<ptrdiff_t>(second_layer_begin),
                                    second_layer.begin(), second_layer.end());
    for (uint32_t id = 0; id != mixed_geometry.operators.size(); ++id) mixed_geometry.operators[id].id = id;
    mixed_geometry.layers.push_back({0, static_cast<uint32_t>(second_layer_begin),
                                     model.layers.front().operator_count, 0});
    auto second_attention = std::find_if(
        mixed_geometry.operators.begin() + static_cast<ptrdiff_t>(second_layer_begin),
        mixed_geometry.operators.begin() + static_cast<ptrdiff_t>(second_layer_begin +
                                                                  model.layers.front().operator_count),
        [](const SemanticOperator& op) { return op.kind == OperatorKind::CausalAttention; });
    CHECK(second_attention != mixed_geometry.operators.end());
    if (second_attention != mixed_geometry.operators.end()) {
        auto* payload = std::get_if<CausalAttentionPayload>(&second_attention->payload);
        CHECK(payload != nullptr);
        if (payload) {
            payload->query_heads = 8;
            payload->kv_heads = 8;
            payload->head_dimension = 32;
        }
    }
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    const auto mixed_plan = plan_canonical_metal_dense(
        mixed_geometry, request(), metal, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(mixed_plan));
}

} // namespace

int main() {
    test_pattern_matrix();
    test_selection_matrix();
    test_plan_failures();
    test_materialized_entries_and_peak_memory();
    test_metal_blocked_descriptor_requires_the_exact_q4k_contract();
    test_metal_quantized_descriptor_signatures_are_closed();
    test_metal_iq2_xxs_descriptor_requires_the_exact_codebook_contract();
    test_affine_u2_requires_a_queried_capability();
    test_column_grouped_affine_u2_skip_has_a_distinct_typed_route();
    test_f16_prefill_batch_descriptor_is_exactly_two_rows();
    test_canonical_planner_materializes_q4k_dense_entries();
    test_canonical_planner_admits_flattened_attention_context();
    test_canonical_planner_rejects_sliding_attention_until_windowed_metal_exists();
    test_canonical_planner_rejects_invalid_rope_and_rms_contracts();
    test_canonical_planner_materializes_query_gate_dense_entries();
    test_canonical_planner_materializes_mixed_quantized_dense_entries();
    test_canonical_planner_materializes_recurrent_q4k_q6k_entries();
    test_canonical_planner_admits_uniform_recurrent_q4k_and_q6k();
    test_canonical_planner_materializes_schema4_recurrent_entries();
    test_canonical_planner_materializes_mixed_dense_recurrent_entries();
    test_canonical_planner_rejects_a_layer_that_bypasses_the_previous_residual();
    test_canonical_planner_rejects_a_graph_spine_that_differs_from_metal();
    test_canonical_planner_skips_disabled_schema6_terminal_layer();
    test_canonical_planner_admits_q6k_output_projection();
    test_canonical_planner_admits_q6k_token_embedding();
    test_canonical_planner_admits_q4k_token_embedding();
    test_moe_registry_matches_generic_physical_contract();
    test_moe_planner_discovers_permuted_graph_and_formats();
    test_moe_planner_rejects_ambiguous_and_nonmatching_graphs();
    test_dense_graph_witness_closes_the_executed_dag();
    return test_summary("test_execution_plan");
}
