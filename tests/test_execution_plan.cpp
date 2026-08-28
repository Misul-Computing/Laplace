#include <algorithm>
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
    model.values = {
        {0, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 256}}, 0},
        {1, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 256}}, 0},
        {2, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 256}}, 0},
        {3, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}, {DimensionKind::Constant, 64}}, 0},
        {4, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 256}}, 0},
    };
    auto add = [&](OperatorKind kind, std::vector<uint32_t> tensors, std::vector<uint32_t> outputs, OperatorPayload payload) {
        SemanticOperator op;
        op.id = static_cast<uint32_t>(model.operators.size());
        op.kind = kind;
        op.tensors = std::move(tensors);
        op.outputs = std::move(outputs);
        op.payload = std::move(payload);
        model.operators.push_back(std::move(op));
    };
    add(OperatorKind::EmbeddingLookup, {0}, {0}, EmbeddingLookupPayload{0x3f800000u, 3, 256, 0});
    add(OperatorKind::RmsNorm, {1}, {1}, RmsNormPayload{0x358637bdu, -1, 1});
    add(OperatorKind::Linear, {2}, {1}, LinearPayload{});
    add(OperatorKind::Linear, {3}, {1}, LinearPayload{});
    add(OperatorKind::Linear, {4}, {1}, LinearPayload{});
    add(OperatorKind::Rope, {}, {1, 2}, RopePayload{RopePairing::HalfSplit, true, 64, 0x49742400u, 0x3f800000u});
    SemanticOperator attention;
    attention.id = static_cast<uint32_t>(model.operators.size());
    attention.kind = OperatorKind::CausalAttention;
    attention.outputs = {3};
    attention.states = {0, 1};
    attention.payload = CausalAttentionPayload{4, 4, 64, 0x3e800000u, AttentionMask::Causal, CachePolicy::Global};
    model.operators.push_back(std::move(attention));
    add(OperatorKind::Linear, {5}, {1}, LinearPayload{});
    add(OperatorKind::RmsNorm, {6}, {1}, RmsNormPayload{0x358637bdu, -1, 1});
    add(OperatorKind::Linear, {7}, {1}, LinearPayload{});
    add(OperatorKind::Linear, {8}, {1}, LinearPayload{});
    add(OperatorKind::SwiGlu, {}, {1}, SwiGluPayload{});
    add(OperatorKind::Linear, {9}, {1}, LinearPayload{});
    add(OperatorKind::RmsNorm, {10}, {1}, RmsNormPayload{0x358637bdu, -1, 1});
    add(OperatorKind::Linear, {11}, {4}, LinearPayload{});
    model.layers = {{0, 1, 12, 0}};
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
    fused_query_gate.dimensions[1].constant_or_symbol = 512;
    fused_query_gate.planes[0].length = 256ull * 512 / 256 * 144;

    const auto value = [&](uint32_t id, uint32_t width) {
        model.values.push_back({id, ScalarType::F32,
                                {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, width}}, 0});
    };
    value(5, 512); // fused query and gate
    value(6, 256); // query
    value(7, 256); // attention gate
    value(8, 256); // key
    value(9, 256); // value
    value(10, 256); // rotated query
    value(11, 256); // rotated key
    value(12, 256); // attention context
    value(13, 256); // gated attention
    value(14, 256); // attention output projection
    model.values[12].dimensions = {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}, {DimensionKind::Constant, 64}};
    model.values[13].dimensions = model.values[12].dimensions;

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

    for (uint32_t id = 0; id != model.operators.size(); ++id) model.operators[id].id = id;
    model.layers[0].first_operator = 1;
    model.layers[0].operator_count = static_cast<uint32_t>(model.operators.size() - 3);
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
    const uint32_t operator_base = static_cast<uint32_t>(model.operators.size());
    for (SemanticTensor tensor : recurrent.tensors) {
        tensor.id += tensor_base;
        model.tensors.push_back(std::move(tensor));
    }
    for (SemanticValue value : recurrent.values) {
        value.id += value_base;
        model.values.push_back(std::move(value));
    }
    for (SemanticState state : recurrent.states) {
        state.id += state_base;
        state.semantic_version = 4;
        model.states.push_back(std::move(state));
    }
    for (SemanticOperator op : recurrent.operators) {
        op.id += operator_base;
        op.semantic_version = 4;
        for (uint32_t& value : op.inputs) value += value_base;
        for (uint32_t& value : op.outputs) value += value_base;
        for (uint32_t& tensor : op.tensors) tensor += tensor_base;
        for (uint32_t& state : op.states) state += state_base;
        model.operators.push_back(std::move(op));
    }
    model.layers.push_back({1, operator_base, static_cast<uint32_t>(recurrent.operators.size()), 0});
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
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
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
        const uint32_t expected_formats = MetalWeightFormatQ4K | MetalWeightFormatQ5_0 |
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

void test_canonical_planner_skips_disabled_schema6_terminal_layer() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    const SemanticModel model = schema6_model_with_disabled_terminal_layer();
    const auto planned = plan_canonical_metal(model, request(), metal, builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(planned));
    if (const auto* plan = std::get_if<ExecutionPlan>(&planned)) {
        CHECK(plan->entries.size() == 4);
        for (const PlanEntry& entry : plan->entries) {
            CHECK(entry.operator_id != model.operators.back().id);
            CHECK(plan_entry_matches(model, entry));
        }
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

void test_canonical_planner_refuses_moe_operators() {
    RuntimeCapabilities metal;
    metal.global_fp32_kv = true;
    metal.transactional_state = true;
    metal.metal_device = true;
    metal.metal_library = true;
    metal.metal_pipeline = true;
    for (const OperatorKind kind : {OperatorKind::RouterTopK, OperatorKind::RoutedLinear,
                                    OperatorKind::WeightedExpertReduce}) {
        SemanticModel model = q4k_dense_model();
        const uint32_t operator_id = static_cast<uint32_t>(model.operators.size());
        model.operators.push_back({operator_id, kind, 1});
        const auto planned = plan_canonical_metal(model, request(), metal,
                                                  builtin_canonical_metal_registry());
        CHECK(std::holds_alternative<CompatibilityReport>(planned));
        if (const auto* report = std::get_if<CompatibilityReport>(&planned)) {
            CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
            CHECK(report->operator_id == operator_id);
            CHECK(report->detail == "canonical Metal MoE operators are not admitted");
        }
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

} // namespace

int main() {
    test_pattern_matrix();
    test_selection_matrix();
    test_plan_failures();
    test_materialized_entries_and_peak_memory();
    test_metal_blocked_descriptor_requires_the_exact_q4k_contract();
    test_metal_quantized_descriptor_signatures_are_closed();
    test_metal_iq2_xxs_descriptor_requires_the_exact_codebook_contract();
    test_f16_prefill_batch_descriptor_is_exactly_two_rows();
    test_canonical_planner_materializes_q4k_dense_entries();
    test_canonical_planner_admits_flattened_attention_context();
    test_canonical_planner_materializes_query_gate_dense_entries();
    test_canonical_planner_materializes_mixed_quantized_dense_entries();
    test_canonical_planner_materializes_recurrent_q4k_q6k_entries();
    test_canonical_planner_admits_uniform_recurrent_q4k_and_q6k();
    test_canonical_planner_materializes_schema4_recurrent_entries();
    test_canonical_planner_materializes_mixed_dense_recurrent_entries();
    test_canonical_planner_skips_disabled_schema6_terminal_layer();
    test_canonical_planner_refuses_moe_operators();
    test_canonical_planner_admits_q6k_output_projection();
    test_canonical_planner_admits_q6k_token_embedding();
    test_canonical_planner_admits_q4k_token_embedding();
    return test_summary("test_execution_plan");
}
