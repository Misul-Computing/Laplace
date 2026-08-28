#include <cmath>
#include <cstring>
#include <vector>

#include "semantic_model.h"
#include "test_util.h"

using namespace Laplace;

namespace {

bool is_ok(const SemanticVectorResult& result) {
    return std::holds_alternative<std::vector<float>>(result);
}

uint32_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

const std::vector<float>& values(const SemanticVectorResult& result) {
    return std::get<std::vector<float>>(result);
}

void test_operator_vectors() {
    auto embedding = semantic_embedding_lookup({1, 2, 3, 4, 5, 6}, 3, 2, {2, 0}, 0.5f);
    CHECK(is_ok(embedding));
    if (is_ok(embedding)) {
        CHECK(almost_equal(values(embedding)[0], 2.5f));
        CHECK(almost_equal(values(embedding)[1], 3.0f));
        CHECK(almost_equal(values(embedding)[2], 0.5f));
        CHECK(almost_equal(values(embedding)[3], 1.0f));
    }

    auto rms = semantic_rms_norm({3, 4}, {2, 0.5f}, 3.5f);
    CHECK(is_ok(rms));
    if (is_ok(rms)) {
        CHECK(almost_equal(values(rms)[0], 1.5f));
        CHECK(almost_equal(values(rms)[1], 0.5f));
    }

    std::vector<float> reduced_rms_input(32, 0.0f);
    reduced_rms_input[0] = 1.0f;
    reduced_rms_input[1] = 1.0f;
    reduced_rms_input[16] = 4096.0f;
    auto reduced_rms = semantic_rms_norm(reduced_rms_input, std::vector<float>(32, 1.0f), 0.0f);
    CHECK(is_ok(reduced_rms));
    if (is_ok(reduced_rms)) {
        const float expected = 4096.0f / std::sqrt(16777216.0f / 32.0f);
        CHECK(float_bits(values(reduced_rms)[16]) == float_bits(expected));
    }

    auto linear = semantic_linear({1, 2}, {3, 4, 5, 6}, 2, 2, {7, 8});
    CHECK(is_ok(linear));
    if (is_ok(linear)) {
        CHECK(almost_equal(values(linear)[0], 18.0f));
        CHECK(almost_equal(values(linear)[1], 25.0f));
    }

    std::vector<float> reduced_input(32, 0.0f);
    reduced_input[0] = 1e20f;
    reduced_input[1] = -1e20f;
    reduced_input[16] = 1.0f;
    auto reduced_linear = semantic_linear(reduced_input, std::vector<float>(32, 1.0f), 1, 32, {});
    CHECK(is_ok(reduced_linear));
    if (is_ok(reduced_linear)) CHECK(values(reduced_linear)[0] == 0.0f);

    auto bias_after_reduce = semantic_linear({1e20f, -1e20f, 1.0f}, {1.0f, 1.0f, 1.0f}, 1, 3, {-1.0f});
    CHECK(is_ok(bias_after_reduce));
    if (is_ok(bias_after_reduce)) CHECK(values(bias_after_reduce)[0] == 0.0f);

    std::vector<float> q = {1, 2, 9};
    std::vector<float> k = {1, 2, 9};
    auto rope = semantic_rope_half_split(q, k, 1, 2, 1.0f, 1.5707963267948966f);
    CHECK(is_ok(rope));
    if (is_ok(rope)) {
        const auto& rotated = values(rope);
        CHECK(almost_equal(rotated[0], -2.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[1], 1.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[2], 9.0f));
        CHECK(almost_equal(rotated[3], -2.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[4], 1.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[5], 9.0f));
    }

    auto interleaved_rope = semantic_rope_interleaved({1, 2, 3, 4}, {-1, 1, 2, -2},
                                                      1, 4, 1.0f, 1.5707963267948966f);
    CHECK(is_ok(interleaved_rope));
    if (is_ok(interleaved_rope)) {
        const auto& rotated = values(interleaved_rope);
        CHECK(almost_equal(rotated[0], -2.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[1], 1.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[2], -4.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[3], 3.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[4], -1.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[5], -1.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[6], 2.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[7], 2.0f, 1e-5f, 1e-5f));
    }

    auto multi_section_rope = semantic_rope_multi_section_half_split(
        {1, 2, 3, 4}, {-1, 1, 2, -2}, {1, 2, 0, 0}, {1, 1, 0, 0},
        4, 1.0f, 1.5707963267948966f);
    CHECK(is_ok(multi_section_rope));
    if (is_ok(multi_section_rope)) {
        const auto& rotated = values(multi_section_rope);
        CHECK(almost_equal(rotated[0], -3.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[1], -2.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[2], 1.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[3], -4.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[4], -2.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[5], -1.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[6], -1.0f, 1e-5f, 1e-5f));
        CHECK(almost_equal(rotated[7], 2.0f, 1e-5f, 1e-5f));
    }
    auto invalid_multi_section_rope = semantic_rope_multi_section_half_split(
        {1, 2, 3, 4}, {-1, 1, 2, -2}, {1, 2, 0, 0}, {2, 1, 0, 0},
        4, 1.0f, 1.5707963267948966f);
    CHECK(std::holds_alternative<CompatibilityReport>(invalid_multi_section_rope));

    auto concat = semantic_concat_last_axis({1, 2, 5, 6}, {3, 4, 7, 8}, 2, 2);
    CHECK(is_ok(concat));
    if (is_ok(concat)) CHECK(values(concat) == std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8}));
    auto invalid_concat = semantic_concat_last_axis({1, 2}, {3, 4, 5}, 2, 2);
    CHECK(std::holds_alternative<CompatibilityReport>(invalid_concat));

    SemanticConvState conv_state;
    conv_state.history = {1, 2, 3, 4};
    auto conv = semantic_depthwise_conv1d_silu_step({5, 6},
                                                    {0.1f, 0.2f, 0.3f, -0.5f, 0.25f, 0.5f},
                                                    2, 3, conv_state);
    CHECK(is_ok(conv));
    if (is_ok(conv)) {
        CHECK(almost_equal(values(conv)[0], 1.7615942f, 1e-5f, 1e-5f));
        CHECK(almost_equal(values(conv)[1], 2.3103545f, 1e-5f, 1e-5f));
        CHECK(conv_state.history == std::vector<float>({2, 5, 4, 6}));
    }

    SemanticGatedDeltaState delta_state;
    delta_state.matrix = {1, 0, 0, 1};
    auto delta = semantic_gated_delta_net_step({1, 0}, {1, 0}, {3, 5}, {0}, {0.5f},
                                               1, 1, 2, delta_state);
    CHECK(is_ok(delta));
    if (is_ok(delta)) {
        CHECK(almost_equal(values(delta)[0], 1.4142135f, 1e-5f, 1e-5f));
        CHECK(almost_equal(values(delta)[1], 1.7677670f, 1e-5f, 1e-5f));
        CHECK(delta_state.matrix == std::vector<float>({2, 2.5f, 0, 1}));
    }

    // The recurrent matrix is addressed [key_row][output_column]. The
    // update and both reads are S^T times a key/query vector.
    SemanticGatedDeltaState transpose_delta_state;
    transpose_delta_state.matrix = {1, 2, 3, 4};
    auto transpose_delta = semantic_gated_delta_net_step({1, 2}, {3, 5}, {7, 11}, {0}, {0.5f},
                                                         1, 1, 2, transpose_delta_state);
    CHECK(is_ok(transpose_delta));
    if (is_ok(transpose_delta)) {
        CHECK(almost_equal(values(transpose_delta)[0], -45.608387f, 1e-5f, 1e-5f));
        CHECK(almost_equal(values(transpose_delta)[1], -61.871843f, 1e-5f, 1e-5f));
        CHECK(transpose_delta_state.matrix == std::vector<float>({-15.5f, -20.5f, -24.5f, -33.5f}));
    }

    SemanticGatedDeltaState mapped_delta_state;
    mapped_delta_state.matrix = {0, 0};
    auto mapped_delta = semantic_gated_delta_net_step({2}, {3}, {4, 5}, {0, 0}, {1, 1},
                                                      1, 2, 1, mapped_delta_state);
    CHECK(is_ok(mapped_delta));
    if (is_ok(mapped_delta)) {
        CHECK(almost_equal(values(mapped_delta)[0], 24.0f));
        CHECK(almost_equal(values(mapped_delta)[1], 30.0f));
        CHECK(mapped_delta_state.matrix == std::vector<float>({12, 15}));
    }

    auto gated_attention = semantic_gated_attention_output({2, -2}, {0, std::log(3.0f)});
    CHECK(is_ok(gated_attention));
    if (is_ok(gated_attention)) {
        CHECK(almost_equal(values(gated_attention)[0], 1.0f));
        CHECK(almost_equal(values(gated_attention)[1], -1.5f, 1e-5f, 1e-5f));
    }

    auto gated_rms = semantic_gated_rms_norm({3, 4}, {2, 0.5f}, {0, std::log(3.0f)}, 3.5f);
    CHECK(is_ok(gated_rms));
    if (is_ok(gated_rms)) {
        CHECK(almost_equal(values(gated_rms)[0], 0.0f));
        CHECK(almost_equal(values(gated_rms)[1], 0.75f * std::log(3.0f) * 0.5f, 1e-5f, 1e-5f));
    }

    // Qwen3Next normalizes each recurrent Q/K head before DeltaNet.  This is
    // L2 length, not RMS normalization and must stay independent of any
    // architecture importer.
    auto l2 = semantic_l2_normalize({3, 4}, 0.0f);
    CHECK(is_ok(l2));
    if (is_ok(l2)) {
        CHECK(almost_equal(values(l2)[0], 0.6f));
        CHECK(almost_equal(values(l2)[1], 0.8f));
    }
    // Qwen3Next's recurrent L2 path adds epsilon inside the square root.
    // Clamping the already-computed length changes nonzero vectors.
    auto l2_epsilon = semantic_l2_normalize({3, 4}, 9.0f);
    CHECK(is_ok(l2_epsilon));
    if (is_ok(l2_epsilon)) {
        CHECK(almost_equal(values(l2_epsilon)[0], 3.0f / std::sqrt(34.0f), 1e-6f, 1e-6f));
        CHECK(almost_equal(values(l2_epsilon)[1], 4.0f / std::sqrt(34.0f), 1e-6f, 1e-6f));
    }

    SemanticKvState cache;
    cache.key = {0};
    cache.value = {2};
    cache.tokens = 1;
    auto attention = semantic_causal_attention({0, 0}, {0}, {4}, 1, 2, 1, 1, 1.0f, cache);
    CHECK(is_ok(attention));
    if (is_ok(attention)) {
        CHECK(almost_equal(values(attention)[0], 3.0f));
        CHECK(almost_equal(values(attention)[1], 3.0f));
        CHECK(cache.tokens == 2);
        CHECK(cache.value.size() == 2);
        CHECK(almost_equal(cache.value[0], 2.0f));
        CHECK(almost_equal(cache.value[1], 4.0f));
    }

    // Head width is a semantic payload, not a family or layer-number rule.
    SemanticKvState width_32_cache;
    std::vector<float> width_32_query(32, 1.0f);
    std::vector<float> width_32_key(32, 1.0f);
    std::vector<float> width_32_value(32, 2.0f);
    auto width_32_attention = semantic_causal_attention(width_32_query, width_32_key, width_32_value,
                                                         1, 1, 1, 32, 1.0f, width_32_cache);
    CHECK(is_ok(width_32_attention));
    if (is_ok(width_32_attention)) {
        CHECK(values(width_32_attention).size() == 32);
        CHECK(almost_equal(values(width_32_attention)[31], 2.0f));
        CHECK(width_32_cache.tokens == 1);
    }

    auto swiglu = semantic_swiglu({std::log(3.0f)}, {4});
    CHECK(is_ok(swiglu));
    if (is_ok(swiglu)) {
        CHECK(almost_equal(values(swiglu)[0], 3.0f * std::log(3.0f), 1e-5f, 1e-5f));
    }

    auto add = semantic_add({1, -2}, {3, 5});
    CHECK(is_ok(add));
    if (is_ok(add)) {
        CHECK(almost_equal(values(add)[0], 4.0f));
        CHECK(almost_equal(values(add)[1], 3.0f));
    }
}

void test_operator_errors() {
    auto bad_token = semantic_embedding_lookup({1, 2}, 1, 2, {1}, 1.0f);
    CHECK(std::holds_alternative<CompatibilityReport>(bad_token));
    if (auto* report = std::get_if<CompatibilityReport>(&bad_token)) {
        CHECK(report->code == CompatibilityError::RUNTIME_INPUT_INVALID);
    }

    auto bad_norm = semantic_rms_norm({1, 2}, {1}, 0.0f);
    CHECK(std::holds_alternative<CompatibilityReport>(bad_norm));
    if (auto* report = std::get_if<CompatibilityReport>(&bad_norm)) {
        CHECK(report->code == CompatibilityError::IR_SHAPE_MISMATCH);
    }

    auto bad_linear = semantic_linear({1, 2}, {1, 2, 3}, 2, 2, {});
    CHECK(std::holds_alternative<CompatibilityReport>(bad_linear));
    if (auto* report = std::get_if<CompatibilityReport>(&bad_linear)) {
        CHECK(report->code == CompatibilityError::IR_SHAPE_MISMATCH);
    }

    std::vector<float> q = {1, 2};
    std::vector<float> k = {1, 2};
    auto bad_rope = semantic_rope_half_split(q, k, 0, 2, 0.0f, 1.0f);
    CHECK(std::holds_alternative<CompatibilityReport>(bad_rope));
    if (auto* report = std::get_if<CompatibilityReport>(&bad_rope)) {
        CHECK(report->code == CompatibilityError::IR_CONSTRAINT_FAILED);
    }

    auto bad_interleaved_rope = semantic_rope_interleaved(q, k, 0, 3, 1.0f, 1.0f);
    CHECK(std::holds_alternative<CompatibilityReport>(bad_interleaved_rope));
    if (auto* report = std::get_if<CompatibilityReport>(&bad_interleaved_rope)) {
        CHECK(report->code == CompatibilityError::IR_SHAPE_MISMATCH);
    }

    SemanticConvState bad_conv_state;
    bad_conv_state.history = {1};
    const auto history_before = bad_conv_state.history;
    auto bad_conv = semantic_depthwise_conv1d_silu_step({3}, {1, 2, 3}, 1, 3, bad_conv_state);
    CHECK(std::holds_alternative<CompatibilityReport>(bad_conv));
    if (auto* report = std::get_if<CompatibilityReport>(&bad_conv)) {
        CHECK(report->code == CompatibilityError::IR_SHAPE_MISMATCH);
    }
    CHECK(bad_conv_state.history == history_before);

    SemanticGatedDeltaState bad_delta_state;
    bad_delta_state.matrix = {1};
    const auto delta_before = bad_delta_state.matrix;
    auto bad_delta = semantic_gated_delta_net_step({1}, {1}, {1}, {0}, {2},
                                                   1, 1, 1, bad_delta_state);
    CHECK(std::holds_alternative<CompatibilityReport>(bad_delta));
    if (auto* report = std::get_if<CompatibilityReport>(&bad_delta)) {
        CHECK(report->code == CompatibilityError::IR_CONSTRAINT_FAILED);
    }
    CHECK(bad_delta_state.matrix == delta_before);

    auto bad_gated_attention = semantic_gated_attention_output({1}, {1, 2});
    CHECK(std::holds_alternative<CompatibilityReport>(bad_gated_attention));
    if (auto* report = std::get_if<CompatibilityReport>(&bad_gated_attention)) {
        CHECK(report->code == CompatibilityError::IR_SHAPE_MISMATCH);
    }

    auto bad_gated_rms = semantic_gated_rms_norm({1}, {1}, {1, 2}, 0.0f);
    CHECK(std::holds_alternative<CompatibilityReport>(bad_gated_rms));
    if (auto* report = std::get_if<CompatibilityReport>(&bad_gated_rms)) {
        CHECK(report->code == CompatibilityError::IR_SHAPE_MISMATCH);
    }

    auto bad_l2 = semantic_l2_normalize({}, 0.0f);
    CHECK(std::holds_alternative<CompatibilityReport>(bad_l2));
    if (auto* report = std::get_if<CompatibilityReport>(&bad_l2)) {
        CHECK(report->code == CompatibilityError::IR_SHAPE_MISMATCH);
    }

    SemanticKvState cache;
    auto bad_attention = semantic_causal_attention({0, 0, 0}, {0, 0}, {0, 0}, 1, 3, 2, 1, 1.0f, cache);
    CHECK(std::holds_alternative<CompatibilityReport>(bad_attention));
    if (auto* report = std::get_if<CompatibilityReport>(&bad_attention)) {
        CHECK(report->code == CompatibilityError::IR_CONSTRAINT_FAILED);
    }

    auto bad_swiglu = semantic_swiglu({1}, {1, 2});
    CHECK(std::holds_alternative<CompatibilityReport>(bad_swiglu));
    if (auto* report = std::get_if<CompatibilityReport>(&bad_swiglu)) {
        CHECK(report->code == CompatibilityError::IR_SHAPE_MISMATCH);
    }

    auto bad_add = semantic_add({1}, {1, 2});
    CHECK(std::holds_alternative<CompatibilityReport>(bad_add));
    if (auto* report = std::get_if<CompatibilityReport>(&bad_add)) {
        CHECK(report->code == CompatibilityError::IR_SHAPE_MISMATCH);
    }
}

void test_semantic_wire() {
    SemanticModel model;
    model.maximum_context = 32768;
    model.vocabulary_size = 3;
    model.bos_id = 1;
    model.eos_id = 2;
    model.stop_ids = {2};
    model.tokenizer_digest[0] = 7;

    auto encoded = encode_semantic_model(model);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
    if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) return;
    const auto& bytes = std::get<std::vector<uint8_t>>(encoded);
    CHECK(bytes.size() == 64 + 136 + 4 + 4);
    CHECK(std::memcmp(bytes.data(), "LAPIR001", 8) == 0);
    auto decoded = decode_semantic_model(bytes);
    CHECK(std::holds_alternative<SemanticModel>(decoded));
    if (auto* roundtrip = std::get_if<SemanticModel>(&decoded)) {
        CHECK(roundtrip->maximum_context == 32768);
        CHECK(roundtrip->vocabulary_size == 3);
        CHECK(roundtrip->stop_ids == std::vector<uint32_t>({2}));
        CHECK(roundtrip->tokenizer_digest[0] == 7);
    }

    auto bad_version = bytes;
    bad_version[8] = 2;
    auto version_result = decode_semantic_model(bad_version);
    CHECK(std::holds_alternative<CompatibilityReport>(version_result));
    if (auto* report = std::get_if<CompatibilityReport>(&version_result)) {
        CHECK(report->code == CompatibilityError::IR_VERSION_UNSUPPORTED);
    }

    auto trailing = bytes;
    trailing.push_back(0);
    auto trailing_result = decode_semantic_model(trailing);
    CHECK(std::holds_alternative<CompatibilityReport>(trailing_result));
    if (auto* report = std::get_if<CompatibilityReport>(&trailing_result)) {
        CHECK(report->code == CompatibilityError::IR_VERSION_UNSUPPORTED);
    }

    model.maximum_context = 0;
    auto zero_context = encode_semantic_model(model);
    CHECK(std::holds_alternative<CompatibilityReport>(zero_context));
    model.maximum_context = 32769;
    auto long_context = encode_semantic_model(model);
    CHECK(std::holds_alternative<CompatibilityReport>(long_context));

    CHECK(static_cast<uint16_t>(ScalarType::F32) == 1);
    CHECK(static_cast<uint16_t>(TensorRole::OutputWeight) == 15);
    CHECK(static_cast<uint16_t>(OperatorKind::CausalAttention) == 5);
    CHECK(static_cast<uint16_t>(ExecutionPhase::Output) == 5);
}

void test_semantic_wire_v2_recurrent() {
    SemanticModel model;
    model.schema_major = 2;
    model.schema_minor = 0;
    model.opset_major = 2;
    model.opset_minor = 0;
    model.maximum_context = 256;
    model.vocabulary_size = 1;

    for (uint32_t id = 0; id != 11; ++id) {
        model.values.push_back({id, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0});
    }

    auto tensor = [](uint32_t id, TensorRole role) {
        SemanticTensor result;
        result.id = id;
        result.role = role;
        result.logical_type = ScalarType::F32;
        result.dimensions = {{DimensionKind::Constant, 1}, {DimensionKind::Constant, 1}};
        result.layout.rank = 2;
        result.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        result.layout.strides[0] = 1;
        result.layout.strides[1] = 1;
        result.planes.push_back({PlaneKind::Values, ScalarType::F32, ArtifactId{0},
                                 static_cast<uint64_t>(id) * 64, 4, 4, 0});
        return result;
    };
    model.tensors = {tensor(0, TensorRole::RecurrentConvWeight),
                     tensor(1, TensorRole::RecurrentDtBias),
                     tensor(2, TensorRole::RecurrentDecayWeight),
                     tensor(3, TensorRole::RecurrentNormWeight)};
    model.tensors[0].dimensions = {{DimensionKind::Constant, 3}, {DimensionKind::Constant, 2}};
    model.tensors[0].layout.strides[0] = 2;

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
    model.states.push_back({0, StateKind::RecurrentConvHistory, 2, StateUpdateKind::ShiftHistory,
                            PositionPolicy::ReplaceAtCursor,
                            {{DimensionKind::Constant, 3}, {DimensionKind::Constant, 1}}, {recurrent}, 0});
    recurrent.layout_policy = LayoutPolicy::ValueHeadOutputRowKeyColumn;
    model.states.push_back({1, StateKind::RecurrentDeltaMatrix, 2, StateUpdateKind::DeltaMatrix,
                            PositionPolicy::ReplaceAtCursor,
                            {{DimensionKind::Constant, 1}, {DimensionKind::Constant, 1}, {DimensionKind::Constant, 1}},
                            {recurrent}, 0});

    SemanticOperator conv;
    conv.id = 0;
    conv.kind = OperatorKind::DepthwiseConvSilu;
    conv.semantic_version = 2;
    conv.inputs = {0};
    conv.outputs = {1, 2, 3};
    conv.tensors = {0};
    conv.states = {0};
    conv.payload = DepthwiseConvSiluPayload{1, 1, 1, 2};

    SemanticOperator l2_query;
    l2_query.id = 1;
    l2_query.kind = OperatorKind::L2Normalize;
    l2_query.semantic_version = 2;
    l2_query.inputs = {1};
    l2_query.outputs = {9};
    l2_query.payload = L2NormalizePayload{float_bits(1.0f)};

    SemanticOperator l2_key = l2_query;
    l2_key.id = 2;
    l2_key.inputs = {2};
    l2_key.outputs = {10};

    SemanticOperator delta;
    delta.id = 3;
    delta.kind = OperatorKind::GatedDeltaNet;
    delta.semantic_version = 2;
    delta.inputs = {9, 10, 3, 4, 5};
    delta.outputs = {6};
    delta.tensors = {1, 2};
    delta.states = {1};
    delta.payload = GatedDeltaNetPayload{1, 1, 1, QkHeadMapping::ValueHeadModulo,
                                         BetaTransform::Sigmoid, DecayTransform::NegativeSoftplus,
                                         DeltaStateLayout::ValueHeadOutputRowKeyColumn, 0};

    SemanticOperator gated_attention;
    gated_attention.id = 4;
    gated_attention.kind = OperatorKind::GatedAttention;
    gated_attention.semantic_version = 2;
    gated_attention.inputs = {6, 4};
    gated_attention.outputs = {7};
    gated_attention.payload = GatedAttentionPayload{};

    SemanticOperator gated_rms;
    gated_rms.id = 5;
    gated_rms.kind = OperatorKind::GatedRmsNorm;
    gated_rms.semantic_version = 2;
    gated_rms.inputs = {7, 5};
    gated_rms.outputs = {8};
    gated_rms.tensors = {3};
    gated_rms.payload = GatedRmsNormPayload{float_bits(1.0f), ActivationKind::Silu, 1};
    model.operators = {conv, l2_query, l2_key, delta, gated_attention, gated_rms};

    auto encoded = encode_semantic_model(model);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
    if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) return;
    const auto& bytes = std::get<std::vector<uint8_t>>(encoded);
    CHECK(bytes.size() >= 64);
    if (bytes.size() < 64) return;
    CHECK(std::memcmp(bytes.data(), "LAPIR002", 8) == 0);
    auto decoded = decode_semantic_model(bytes);
    CHECK(std::holds_alternative<SemanticModel>(decoded));
    if (const auto* result = std::get_if<SemanticModel>(&decoded)) {
        CHECK(result->schema_major == 2);
        CHECK(result->opset_major == 2);
        CHECK(result->maximum_context == 256);
        CHECK(result->operators.size() == 6);
        CHECK(result->states.size() == 2);
        if (result->operators.size() == 6 && result->states.size() == 2) {
            CHECK(result->operators[0].semantic_version == 2);
            const auto* decoded_conv = std::get_if<DepthwiseConvSiluPayload>(&result->operators[0].payload);
            const auto* decoded_delta = std::get_if<GatedDeltaNetPayload>(&result->operators[3].payload);
            CHECK(decoded_conv != nullptr);
            CHECK(decoded_delta != nullptr);
            if (decoded_conv) CHECK(decoded_conv->kernel == 2);
            if (decoded_delta) CHECK(decoded_delta->state_layout == DeltaStateLayout::ValueHeadOutputRowKeyColumn);
            const auto* decoded_l2 = std::get_if<L2NormalizePayload>(&result->operators[1].payload);
            CHECK(decoded_l2 != nullptr);
            if (decoded_l2) CHECK(decoded_l2->epsilon_f32_bits == float_bits(1.0f));
            CHECK(std::holds_alternative<GatedAttentionPayload>(result->operators[4].payload));
            const auto* decoded_rms = std::get_if<GatedRmsNormPayload>(&result->operators[5].payload);
            CHECK(decoded_rms != nullptr);
            if (decoded_rms) CHECK(decoded_rms->gate_activation == ActivationKind::Silu);
            CHECK(result->states[0].kind == StateKind::RecurrentConvHistory);
            CHECK(result->states[1].formats[0].layout_policy == LayoutPolicy::ValueHeadOutputRowKeyColumn);
        }
    }

    auto bad_history = model;
    bad_history.states[0].dimensions[0].constant_or_symbol = 4;
    CHECK(std::holds_alternative<CompatibilityReport>(encode_semantic_model(bad_history)));
    auto bad_layout = model;
    bad_layout.states[1].formats[0].layout_policy = LayoutPolicy::ChannelMajorHistory;
    CHECK(std::holds_alternative<CompatibilityReport>(encode_semantic_model(bad_layout)));
    auto bad_mapping = model;
    std::get<GatedDeltaNetPayload>(bad_mapping.operators[3].payload).qk_mapping = static_cast<QkHeadMapping>(2);
    CHECK(std::holds_alternative<CompatibilityReport>(encode_semantic_model(bad_mapping)));

    auto v3 = model;
    v3.schema_major = 3;
    v3.opset_major = 3;
    for (SemanticOperator& op : v3.operators) op.semantic_version = 3;
    for (SemanticState& state : v3.states) state.semantic_version = 3;
    v3.states[1].formats[0].layout_policy = LayoutPolicy::ValueHeadKeyRowOutputColumn;
    std::get<GatedDeltaNetPayload>(v3.operators[3].payload).state_layout =
        DeltaStateLayout::ValueHeadKeyRowOutputColumn;
    auto v3_encoded = encode_semantic_model(v3);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(v3_encoded));
    if (!std::holds_alternative<std::vector<uint8_t>>(v3_encoded)) return;
    const auto& v3_bytes = std::get<std::vector<uint8_t>>(v3_encoded);
    CHECK(std::memcmp(v3_bytes.data(), "LAPIR003", 8) == 0);
    auto v3_decoded = decode_semantic_model(v3_bytes);
    CHECK(std::holds_alternative<SemanticModel>(v3_decoded));
    if (const auto* result = std::get_if<SemanticModel>(&v3_decoded)) {
        CHECK(result->schema_major == 3);
        CHECK(result->states[1].formats[0].layout_policy == LayoutPolicy::ValueHeadKeyRowOutputColumn);
        const auto* payload = std::get_if<GatedDeltaNetPayload>(&result->operators[3].payload);
        CHECK(payload != nullptr);
        if (payload) CHECK(payload->state_layout == DeltaStateLayout::ValueHeadKeyRowOutputColumn);
    }
}

void test_semantic_wire_v2_reuses_known_linear() {
    SemanticModel model;
    model.schema_major = 2;
    model.opset_major = 2;
    model.maximum_context = 2;
    model.vocabulary_size = 1;
    model.values = {
        {0, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0},
        {1, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0},
    };
    SemanticTensor weight;
    weight.id = 0;
    weight.role = TensorRole::RecurrentQkvWeight;
    weight.logical_type = ScalarType::F32;
    weight.dimensions = {{DimensionKind::Constant, 1}, {DimensionKind::Constant, 1}};
    weight.layout.rank = 2;
    weight.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    weight.layout.strides[0] = 1;
    weight.layout.strides[1] = 1;
    weight.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, 0, 4, 4, 0}};
    model.tensors = {weight};
    StateFormat key_format;
    key_format.encoded_domain = TransformDomain::RopeApplied;
    key_format.alignment = 64;
    StateFormat value_format = key_format;
    value_format.encoded_domain = TransformDomain::Untransformed;
    model.states = {
        {0, StateKind::KeyCache, 2, StateUpdateKind::AppendKey, PositionPolicy::AppendOnly,
         {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1}, {DimensionKind::Constant, 1}}, {key_format}, 0},
        {1, StateKind::ValueCache, 2, StateUpdateKind::AppendValue, PositionPolicy::AppendOnly,
         {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1}, {DimensionKind::Constant, 1}}, {value_format}, 0},
    };
    model.operators = {{0, OperatorKind::Linear, 2, {0}, {1}, {0}, {}, LinearPayload{}}};
    auto encoded = encode_semantic_model(model);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
}

void test_semantic_axis_split_wire() {
    auto split = semantic_axis_split({1, 2, 3, 4, 5, 6, 7, 8}, 4, 2);
    CHECK(std::holds_alternative<SemanticAxisSplit>(split));
    if (const auto* result = std::get_if<SemanticAxisSplit>(&split)) {
        CHECK(result->first == std::vector<float>({1, 2, 5, 6}));
        CHECK(result->second == std::vector<float>({3, 4, 7, 8}));
    }
    auto malformed = semantic_axis_split({1, 2, 3}, 4, 2);
    CHECK(std::holds_alternative<CompatibilityReport>(malformed));

    SemanticModel model;
    model.schema_major = 4;
    model.opset_major = 4;
    model.maximum_context = 1;
    model.vocabulary_size = 3;
    model.bos_id = 1;
    model.eos_id = 2;
    model.values = {
        {0, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}, 0},
        {1, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}}, 0},
        {2, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}}, 0},
    };
    model.operators = {{0, OperatorKind::AxisSplit, 4, {0}, {1, 2}, {}, {}, AxisSplitPayload{2, 2}}};

    auto encoded = encode_semantic_model(model);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
    if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) return;
    const auto& bytes = std::get<std::vector<uint8_t>>(encoded);
    CHECK(std::memcmp(bytes.data(), "LAPIR004", 8) == 0);
    auto decoded = decode_semantic_model(bytes);
    CHECK(std::holds_alternative<SemanticModel>(decoded));
    if (const auto* round_trip = std::get_if<SemanticModel>(&decoded)) {
        CHECK(round_trip->operators[0].kind == OperatorKind::AxisSplit);
        CHECK((std::get<AxisSplitPayload>(round_trip->operators[0].payload) == AxisSplitPayload{2, 2}));
    }
}

void test_semantic_wire_v5_multi_section_rope() {
    SemanticModel model;
    model.schema_major = 5;
    model.opset_major = 5;
    model.maximum_context = 2;
    model.vocabulary_size = 3;
    model.bos_id = 1;
    model.eos_id = 2;
    model.values = {
        {0, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}, 0},
        {1, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}, 0},
        {2, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}, 0},
        {3, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}, 0},
    };
    RopePayload rope;
    rope.pairing = RopePairing::MultiSectionHalfSplit;
    rope.position_from_cursor = true;
    rope.rotary_dimension = 4;
    rope.base_f32_bits = 0x461c4000u;
    rope.scale_f32_bits = 0x3f800000u;
    rope.position_sections = {1, 1, 0, 0};
    model.operators = {{0, OperatorKind::Rope, 5, {0, 1}, {2, 3}, {}, {}, rope}};

    auto encoded = encode_semantic_model(model);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
    if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) return;
    const auto& bytes = std::get<std::vector<uint8_t>>(encoded);
    CHECK(std::memcmp(bytes.data(), "LAPIR005", 8) == 0);
    auto decoded = decode_semantic_model(bytes);
    CHECK(std::holds_alternative<SemanticModel>(decoded));
    if (const auto* round_trip = std::get_if<SemanticModel>(&decoded)) {
        CHECK((std::get<RopePayload>(round_trip->operators[0].payload).position_sections ==
               std::array<uint32_t, 4>{1, 1, 0, 0}));
    }

    auto invalid = model;
    std::get<RopePayload>(invalid.operators[0].payload).position_sections = {2, 1, 0, 0};
    CHECK(std::holds_alternative<CompatibilityReport>(encode_semantic_model(invalid)));
}

void test_semantic_wire_v6_concat() {
    SemanticModel model;
    model.schema_major = 6;
    model.opset_major = 6;
    model.maximum_context = 2;
    model.vocabulary_size = 3;
    model.bos_id = 1;
    model.eos_id = 2;
    model.values = {
        {0, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}}, 0},
        {1, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}}, 0},
        {2, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}, 0},
        {3, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}, 0},
        {4, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}, 0},
        {5, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}, 0},
        {6, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}, 0},
    };
    RopePayload rope;
    rope.pairing = RopePairing::MultiSectionHalfSplit;
    rope.position_from_cursor = true;
    rope.rotary_dimension = 4;
    rope.base_f32_bits = 0x461c4000u;
    rope.scale_f32_bits = 0x3f800000u;
    rope.position_sections = {1, 1, 0, 0};
    model.operators = {
        {0, OperatorKind::Concat, 6, {0, 1}, {2}, {}, {}, ConcatPayload{-1}},
        {1, OperatorKind::Rope, 6, {3, 4}, {5, 6}, {}, {}, rope},
    };

    auto encoded = encode_semantic_model(model);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
    if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) return;
    const auto& bytes = std::get<std::vector<uint8_t>>(encoded);
    CHECK(std::memcmp(bytes.data(), "LAPIR006", 8) == 0);
    auto decoded = decode_semantic_model(bytes);
    CHECK(std::holds_alternative<SemanticModel>(decoded));
    if (const auto* round_trip = std::get_if<SemanticModel>(&decoded)) {
        CHECK(std::get<ConcatPayload>(round_trip->operators[0].payload).axis == -1);
        const auto& decoded_rope = std::get<RopePayload>(round_trip->operators[1].payload);
        CHECK(decoded_rope.pairing == RopePairing::MultiSectionHalfSplit);
        CHECK((decoded_rope.position_sections == std::array<uint32_t, 4>{1, 1, 0, 0}));
    }
}

void test_semantic_wire_v7_generic_moe() {
    auto dimensions = [](std::initializer_list<Dimension> values) {
        return std::vector<Dimension>(values);
    };
    auto tensor = [&](uint32_t id, std::vector<Dimension> dims, ExpertAxis expert_axis) {
        SemanticTensor result;
        result.id = id;
        result.role = TensorRole::FfnGateWeight;
        result.logical_type = ScalarType::F32;
        result.dimensions = std::move(dims);
        result.layout.rank = static_cast<uint8_t>(result.dimensions.size());
        for (uint8_t axis = 0; axis != result.layout.rank; ++axis) result.layout.axis_order[axis] = axis;
        uint64_t elements = 1;
        for (const Dimension& dimension : result.dimensions) elements *= dimension.constant_or_symbol;
        result.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, 0, elements * 4, 4, 0}};
        result.expert_axis = expert_axis;
        return result;
    };

    ExpertAxis expert_axis;
    expert_axis.kind = ExpertAxisKind::ExpertBank;
    expert_axis.expert_axis = 0;
    expert_axis.member_axis = 0xff;
    expert_axis.input_axis = 1;
    expert_axis.output_axis = 2;
    expert_axis.expert_count = 4;
    expert_axis.per_expert_byte_stride = 64;

    SemanticModel model;
    model.schema_major = 7;
    model.opset_major = 7;
    model.maximum_context = 128;
    model.vocabulary_size = 3;
    model.bos_id = 1;
    model.eos_id = 2;
    model.values = {
        {0, ScalarType::F32, dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}), 0},
        {1, ScalarType::F32, dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}), 0},
        {2, ScalarType::U32, dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}}), 0},
        {3, ScalarType::F32, dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}}), 0},
        {4, ScalarType::F32, dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}, {DimensionKind::Constant, 4}}), 0},
        {5, ScalarType::F32, dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}, {DimensionKind::Constant, 4}}), 0},
        {6, ScalarType::F32, dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}, {DimensionKind::Constant, 4}}), 0},
        {7, ScalarType::F32, dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}, {DimensionKind::Constant, 4}}), 0},
        {8, ScalarType::F32, dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}), 0},
        {9, ScalarType::F32, dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}), 0},
        {10, ScalarType::F32, dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}), 0},
        {11, ScalarType::F32, dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}), 0},
        {12, ScalarType::F32, dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}), 0},
        {13, ScalarType::F32, dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}), 0},
        {14, ScalarType::F32, dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}), 0},
        {15, ScalarType::F32, dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}), 0},
    };
    model.tensors = {
        tensor(0, dimensions({{DimensionKind::Constant, 4}, {DimensionKind::Constant, 4}, {DimensionKind::Constant, 4}}), expert_axis),
        tensor(1, dimensions({{DimensionKind::Constant, 4}, {DimensionKind::Constant, 4}, {DimensionKind::Constant, 4}}), expert_axis),
        tensor(2, dimensions({{DimensionKind::Constant, 4}, {DimensionKind::Constant, 4}, {DimensionKind::Constant, 4}}), expert_axis),
        tensor(3, dimensions({{DimensionKind::Constant, 4}}), ExpertAxis{}),
    };

    RouterTopKPayload router;
    router.expert_count = 4;
    router.selected_count = 2;
    router.score_domain = RouterScoreDomain::Logits;
    router.normalization_order = RouterNormalizationOrder::SelectThenNormalize;
    router.selected_weight_normalization = SelectedWeightNormalization::Softmax;
    router.tie_policy = RouterTiePolicy::LowestExpertId;
    router.weight_source = RouterWeightSource::SelectedNormalizedScore;

    CausalAttentionPayload attention;
    attention.query_heads = 1;
    attention.kv_heads = 1;
    attention.head_dimension = 4;
    attention.scale_f32_bits = float_bits(0.5f);
    attention.mask = AttentionMask::Causal;
    attention.cache_policy = CachePolicy::Global;
    attention.window = AttentionWindowKind::Sliding;
    attention.window_tokens = 64;
    attention.value_source = ValueSource::KeyPreRope;
    attention.value_source_value = 12;

    model.operators = {
        {0, OperatorKind::RouterTopK, 7, {1}, {2, 3}, {}, {}, router},
        {1, OperatorKind::RoutedLinear, 7, {0, 2, 3}, {4}, {0}, {}, RoutedLinearPayload{ScalarType::F32}},
        {2, OperatorKind::RoutedLinear, 7, {0, 2, 3}, {5}, {1}, {}, RoutedLinearPayload{ScalarType::F32}},
        {3, OperatorKind::GatedActivation, 7, {4, 5}, {6}, {}, {}, GatedActivationPayload{ActivationKind::GeluTanh}},
        {4, OperatorKind::RoutedLinear, 7, {6, 2, 3}, {7}, {2}, {}, RoutedLinearPayload{ScalarType::F32}},
        {5, OperatorKind::WeightedExpertReduce, 7, {7, 2, 3}, {8}, {3}, {},
         WeightedExpertReducePayload{ExpertReduceAssociation::SelectedOrderLeftToRight,
                                     ExpertScaleSource::PerExpertTensor, ScalarType::F32}},
        {6, OperatorKind::Scale, 7, {8}, {9}, {}, {}, ScalePayload{ScaleSource::LiteralF32, float_bits(0.5f)}},
        {7, OperatorKind::TanhSoftcap, 7, {9}, {10}, {}, {}, TanhSoftcapPayload{float_bits(30.0f)}},
        {8, OperatorKind::Rope, 7, {11, 12}, {13, 14}, {}, {},
         RopePayload{RopePairing::HalfSplit, true, 4, float_bits(10000.0f), float_bits(1.0f)}},
        {9, OperatorKind::CausalAttention, 7, {13, 14}, {15}, {}, {0, 1}, attention},
    };

    StateFormat key_format;
    key_format.encoded_domain = TransformDomain::RopeApplied;
    key_format.alignment = 64;
    StateFormat value_format = key_format;
    value_format.encoded_domain = TransformDomain::Untransformed;
    model.states = {
        {0, StateKind::KeyCache, 7, StateUpdateKind::AppendKey, PositionPolicy::AppendOnly,
         dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1}, {DimensionKind::Constant, 4}}), {key_format}, 0},
        {1, StateKind::ValueCache, 7, StateUpdateKind::AppendValue, PositionPolicy::AppendOnly,
         dimensions({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1}, {DimensionKind::Constant, 4}}), {value_format}, 0},
    };

    auto encoded = encode_semantic_model(model);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
    if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) return;
    const auto& bytes = std::get<std::vector<uint8_t>>(encoded);
    CHECK(std::memcmp(bytes.data(), "LAPIR007", 8) == 0);
    auto decoded = decode_semantic_model(bytes);
    CHECK(std::holds_alternative<SemanticModel>(decoded));
    if (const auto* roundtrip = std::get_if<SemanticModel>(&decoded)) {
        CHECK(roundtrip->tensors[0].expert_axis == expert_axis);
        CHECK(std::holds_alternative<RouterTopKPayload>(roundtrip->operators[0].payload));
        CHECK(std::holds_alternative<RoutedLinearPayload>(roundtrip->operators[1].payload));
        CHECK(std::get<GatedActivationPayload>(roundtrip->operators[3].payload).activation == ActivationKind::GeluTanh);
        CHECK(std::get<WeightedExpertReducePayload>(roundtrip->operators[5].payload).association ==
              ExpertReduceAssociation::SelectedOrderLeftToRight);
        const auto& decoded_attention = std::get<CausalAttentionPayload>(roundtrip->operators[9].payload);
        CHECK(decoded_attention.window == AttentionWindowKind::Sliding);
        CHECK(decoded_attention.value_source == ValueSource::KeyPreRope);
    }

    auto bad_top_k = model;
    std::get<RouterTopKPayload>(bad_top_k.operators[0].payload).selected_count = 5;
    CHECK(std::holds_alternative<CompatibilityReport>(encode_semantic_model(bad_top_k)));
    auto bad_axis = model;
    bad_axis.tensors[0].expert_axis.output_axis = 1;
    CHECK(std::holds_alternative<CompatibilityReport>(encode_semantic_model(bad_axis)));
    auto missing_route = model;
    missing_route.operators[1].inputs[1] = 0;
    CHECK(std::holds_alternative<CompatibilityReport>(encode_semantic_model(missing_route)));
    auto bad_tie = model;
    std::get<RouterTopKPayload>(bad_tie.operators[0].payload).tie_policy = static_cast<RouterTiePolicy>(2);
    CHECK(std::holds_alternative<CompatibilityReport>(encode_semantic_model(bad_tie)));
    auto bad_reduce = model;
    std::get<WeightedExpertReducePayload>(bad_reduce.operators[5].payload).association =
        static_cast<ExpertReduceAssociation>(2);
    CHECK(std::holds_alternative<CompatibilityReport>(encode_semantic_model(bad_reduce)));
    auto ambiguous_value_source = model;
    SemanticOperator duplicate_rope = ambiguous_value_source.operators[8];
    duplicate_rope.id = static_cast<uint32_t>(ambiguous_value_source.operators.size());
    ambiguous_value_source.operators.push_back(duplicate_rope);
    CHECK(std::holds_alternative<CompatibilityReport>(encode_semantic_model(ambiguous_value_source)));

    auto top_k = semantic_router_top_k({1.0f, 3.0f, 3.0f, 2.0f}, router);
    CHECK(std::holds_alternative<SemanticRouterResult>(top_k));
    if (const auto* selected = std::get_if<SemanticRouterResult>(&top_k)) {
        CHECK(selected->ids == std::vector<uint32_t>({1, 2}));
        CHECK(almost_equal(selected->weights[0], 0.5f));
        CHECK(almost_equal(selected->weights[1], 0.5f));
    }
    auto gated = semantic_gated_activation({1.0f}, {2.0f}, ActivationKind::GeluTanh);
    CHECK(is_ok(gated));
    if (is_ok(gated)) CHECK(almost_equal(values(gated)[0], 1.682384f, 1e-5f, 1e-5f));
    auto reduced = semantic_weighted_expert_reduce({1.0f, 2.0f, 3.0f, 4.0f}, {0.25f, 0.75f}, {2.0f, 0.5f},
                                                   2, WeightedExpertReducePayload{ExpertReduceAssociation::SelectedOrderLeftToRight,
                                                                                   ExpertScaleSource::PerExpertTensor, ScalarType::F32});
    CHECK(is_ok(reduced));
    if (is_ok(reduced)) CHECK(values(reduced) == std::vector<float>({1.625f, 2.5f}));
    auto scaled = semantic_scale({2.0f}, 0.5f);
    CHECK(is_ok(scaled));
    if (is_ok(scaled)) CHECK(values(scaled) == std::vector<float>({1.0f}));
    auto softcapped = semantic_tanh_softcap({2.0f}, 2.0f);
    CHECK(is_ok(softcapped));
    if (is_ok(softcapped)) CHECK(almost_equal(values(softcapped)[0], 2.0f * std::tanh(1.0f)));

    SemanticKvState global_cache;
    global_cache.key = {0.0f};
    global_cache.value = {2.0f};
    global_cache.tokens = 1;
    auto global_attention = semantic_causal_attention_windowed({0.0f}, {0.0f}, {4.0f}, 1, 1, 1, 1, 1.0f,
                                                                AttentionWindowKind::Global, 0, global_cache);
    CHECK(is_ok(global_attention));
    if (is_ok(global_attention)) CHECK(almost_equal(values(global_attention)[0], 3.0f));
    SemanticKvState sliding_cache;
    sliding_cache.key = {0.0f};
    sliding_cache.value = {2.0f};
    sliding_cache.tokens = 1;
    auto sliding_attention = semantic_causal_attention_windowed({0.0f}, {0.0f}, {4.0f}, 1, 1, 1, 1, 1.0f,
                                                                 AttentionWindowKind::Sliding, 1, sliding_cache);
    CHECK(is_ok(sliding_attention));
    if (is_ok(sliding_attention)) {
        CHECK(almost_equal(values(sliding_attention)[0], 4.0f));
        CHECK(sliding_cache.tokens == 1);
    }
}

void test_complete_semantic_wire() {
    SemanticModel model;
    model.maximum_context = 32768;
    model.vocabulary_size = 3;
    model.bos_id = 1;
    model.eos_id = 2;
    model.stop_ids = {2};
    model.input_values_first = 0;
    model.input_values_count = 1;
    model.output_values_first = 7;
    model.output_values_count = 1;

    SemanticTensor tensor;
    tensor.id = 0;
    tensor.role = TensorRole::TokenEmbedding;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, 3}, {DimensionKind::Constant, 2}};
    tensor.layout.kind = PhysicalLayoutKind::GgufBlocked;
    tensor.layout.packing = PackingKind::Gguf;
    tensor.layout.rank = 2;
    tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides[0] = 2;
    tensor.layout.strides[1] = 1;
    tensor.layout.block_elements = 2;
    tensor.layout.block_bytes = 8;
    tensor.quantization.kind = QuantizationKind::BlockedAffine;
    tensor.quantization.accumulation_type = ScalarType::F32;
    tensor.quantization.scale_type = ScalarType::F32;
    tensor.quantization.zero_type = ScalarType::F32;
    tensor.quantization.bias_type = ScalarType::F32;
    tensor.quantization.block_elements = 2;
    tensor.quantization.block_bytes = 8;
    tensor.quantization.group_size = 2;
    tensor.quantization.required_plane_mask = 0x3f;
    for (uint16_t kind = static_cast<uint16_t>(PlaneKind::Values);
         kind <= static_cast<uint16_t>(PlaneKind::LayoutMetadata); ++kind) {
        tensor.planes.push_back({static_cast<PlaneKind>(kind), ScalarType::F32, ArtifactId{0},
                                 static_cast<uint64_t>(kind - 1) * 64, 24, 8, 0});
    }
    model.tensors.push_back(tensor);

    for (uint32_t id = 0; id != 8; ++id) {
        model.values.push_back({id, ScalarType::F32, {{DimensionKind::Constant, 2}}, 0});
    }

    model.operators = {
        {0, OperatorKind::EmbeddingLookup, 1, {0}, {1}, {0}, {},
         EmbeddingLookupPayload{0x3f800000u, 3, 2, 0}},
        {1, OperatorKind::RmsNorm, 1, {1}, {2}, {0}, {},
         RmsNormPayload{0x358637bdu, -1, 1}},
        {2, OperatorKind::Linear, 1, {2}, {3}, {0}, {},
         LinearPayload{false, true, ScalarType::F32}},
        {3, OperatorKind::Rope, 1, {3}, {4}, {0}, {},
         RopePayload{RopePairing::HalfSplit, true, 2, 0x49742400u, 0x3f800000u}},
        {4, OperatorKind::CausalAttention, 1, {4}, {5}, {0}, {0, 1},
         CausalAttentionPayload{2, 1, 1, 0x3f800000u, AttentionMask::Causal, CachePolicy::Global}},
        {5, OperatorKind::SwiGlu, 1, {5}, {6}, {0}, {}, SwiGluPayload{ActivationKind::Silu}},
        {6, OperatorKind::Add, 1, {5, 6}, {7}, {}, {}, AddPayload{}},
    };
    model.layers.push_back({0, 0, 7, 0});

    StateFormat format;
    format.kind = StateFormatKind::GlobalContiguous;
    format.logical_type = ScalarType::F32;
    format.encoded_type = ScalarType::F32;
    format.logical_domain = TransformDomain::Untransformed;
    format.encoded_domain = TransformDomain::RopeApplied;
    format.codec = CodecKind::Fp32;
    format.cache_policy = CachePolicy::Global;
    format.layout_policy = LayoutPolicy::TokenMajorContiguous;
    format.alignment = 64;
    model.states.push_back({0, StateKind::KeyCache, 1, StateUpdateKind::AppendKey,
                            PositionPolicy::AppendOnly, {{DimensionKind::Constant, 1}}, {format}, 0});
    format.encoded_domain = TransformDomain::Untransformed;
    model.states.push_back({1, StateKind::ValueCache, 1, StateUpdateKind::AppendValue,
                            PositionPolicy::AppendOnly, {{DimensionKind::Constant, 1}}, {format}, 0});
    model.constraints.push_back({ConstraintKind::Equal, ConstraintOperandKind::Value,
                                 ConstraintOperandKind::Value, 0, 1, 0, 0, 0, 0});
    model.capabilities.push_back({Capability::ScalarFp32, 1, 0});
    model.fallbacks.push_back({FallbackKind::ExactCpu, ExecutionPhase::Decode,
                               NumericalClass::ExactFp32, 0});

    auto encoded = encode_semantic_model(model);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
    if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) return;
    auto decoded = decode_semantic_model(std::get<std::vector<uint8_t>>(encoded));
    CHECK(std::holds_alternative<SemanticModel>(decoded));
    if (auto* roundtrip = std::get_if<SemanticModel>(&decoded)) {
        CHECK(roundtrip->tensors.size() == 1);
        CHECK(roundtrip->tensors[0].planes.size() == 6);
        CHECK(roundtrip->tensors[0].layout.kind == PhysicalLayoutKind::GgufBlocked);
        CHECK(roundtrip->tensors[0].quantization.kind == QuantizationKind::BlockedAffine);
        CHECK(roundtrip->operators.size() == 7);
        CHECK(roundtrip->layers.size() == 1);
        CHECK(roundtrip->states.size() == 2);
        CHECK(roundtrip->constraints.size() == 1);
        CHECK(roundtrip->capabilities.size() == 1);
        CHECK(roundtrip->fallbacks.size() == 1);
        CHECK(roundtrip->input_values_first == 0);
        CHECK(roundtrip->output_values_first == 7);
    }
}

} // namespace

int main() {
    test_operator_vectors();
    test_operator_errors();
    test_semantic_wire();
    test_semantic_wire_v2_recurrent();
    test_semantic_wire_v2_reuses_known_linear();
    test_semantic_axis_split_wire();
    test_semantic_wire_v5_multi_section_rope();
    test_semantic_wire_v6_concat();
    test_semantic_wire_v7_generic_moe();
    test_complete_semantic_wire();
    return test_summary("test_semantic_model");
}
