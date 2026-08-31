#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <memory>
#include <numeric>
#include <thread>
#include <unistd.h>
#include <vector>

#include "artifact_set.h"
#include "canonical_metal.h"
#include "compat_rule.h"
#include "execution_plan.h"
#include "matmul.h"
#include "quant_ref.h"
#include "reference_fp32.h"
#include "tensor.h"
#include "test_util.h"
#include "tokenizer.h"

using namespace Laplace;

namespace Laplace {
bool metal_available();
bool metal_device_present();
bool metal_test_q2k_two_row_ab(const float*, const Tensor&, float*, float*,
                               int, int, int, double*, double*);
bool metal_test_q2k_streamed_ab(const float*, const Tensor&, float*, float*,
                                int, int, int, double*, double*);
void metal_test_select_q2k_two_row_pipeline(bool);
bool metal_test_q2k_pipeline_widths(uint32_t*, uint32_t*);
bool metal_test_q2k_streamed_pipeline_widths(uint32_t*, uint32_t*);
uint32_t metal_test_thermal_state();
}

namespace {

void test_column_grouped_affine_u2_skip_canonical_transaction();
int test_generic_metal_derived_column_grouped_u2_quality(const char* path,
                                                         const char* corpus_path,
                                                         const char* cache_path);

std::shared_ptr<const RuntimePackage> synthetic_f16_dense_package(
    uint32_t attention_scale_bits = 0x3e800000u,
    bool asymmetric_embeddings = false,
    bool truncated_query_weight = false,
    uint32_t embedding_scale_bits = 0x3f800000u,
    uint32_t hidden = 32) {
    constexpr uint32_t vocabulary = 3;
    const std::array<size_t, 12> tensor_bytes = {
        static_cast<size_t>(hidden) * vocabulary * sizeof(uint16_t),
        static_cast<size_t>(hidden) * sizeof(float),
        static_cast<size_t>(hidden) * hidden * sizeof(uint16_t),
        static_cast<size_t>(hidden) * hidden * sizeof(uint16_t),
        static_cast<size_t>(hidden) * hidden * sizeof(uint16_t),
        static_cast<size_t>(hidden) * hidden * sizeof(uint16_t),
        static_cast<size_t>(hidden) * sizeof(float),
        static_cast<size_t>(hidden) * hidden * sizeof(uint16_t),
        static_cast<size_t>(hidden) * hidden * sizeof(uint16_t),
        static_cast<size_t>(hidden) * hidden * sizeof(uint16_t),
        static_cast<size_t>(hidden) * sizeof(float),
        static_cast<size_t>(hidden) * vocabulary * sizeof(uint16_t),
    };
    std::array<uint64_t, 12> tensor_offsets{};
    size_t total = 0;
    for (size_t index = 0; index != tensor_bytes.size(); ++index) {
        total = (total + 63) & ~size_t{63};
        tensor_offsets[index] = total;
        total += tensor_bytes[index];
    }
    std::vector<uint8_t> bytes(total, 0);
    const auto put_f16 = [&](uint32_t tensor, size_t index, uint16_t value) {
        std::memcpy(bytes.data() + tensor_offsets[tensor] + index * sizeof(value),
                    &value, sizeof(value));
    };
    const auto put_f32 = [&](uint32_t tensor, size_t index, float value) {
        std::memcpy(bytes.data() + tensor_offsets[tensor] + index * sizeof(value),
                    &value, sizeof(value));
    };
    for (uint32_t token = 0; token != vocabulary; ++token) {
        for (uint32_t channel = 0; channel != hidden; ++channel) {
            uint16_t value = token == 0 ? 0x2800u : token == 1 ? 0x2c00u : 0x2e00u;
            if (asymmetric_embeddings && token == 0) value = channel % 2 == 0 ? 0x2800u : 0x3000u;
            if (asymmetric_embeddings && token == 1) {
                constexpr std::array<uint16_t, 3> pattern = {0x3400u, 0x2c00u, 0xb000u};
                value = pattern[channel % pattern.size()];
            }
            put_f16(0, static_cast<size_t>(token) * hidden + channel, value);
        }
    }
    for (uint32_t tensor : {1u, 6u, 10u})
        for (uint32_t channel = 0; channel != hidden; ++channel) put_f32(tensor, channel, 1.0f);
    const auto diagonal = [&](uint32_t tensor, uint16_t value) {
        for (uint32_t channel = 0; channel != hidden; ++channel)
            put_f16(tensor, static_cast<size_t>(channel) * hidden + channel, value);
    };
    for (uint32_t tensor : {2u, 3u, 4u}) diagonal(tensor, 0x3c00u);
    diagonal(5, 0x3800u);
    diagonal(7, 0x3800u);
    diagonal(8, 0x3800u);
    diagonal(9, 0x3400u);
    put_f16(11, 0, 0x3c00u);
    put_f16(11, static_cast<size_t>(hidden) + 1, 0x3800u);
    put_f16(11, static_cast<size_t>(2) * hidden + 2, 0xb400u);
    char path[] = "/private/tmp/laplace-f16-prefill-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) return {};
    const bool written = write(fd, bytes.data(), bytes.size()) ==
                         static_cast<ssize_t>(bytes.size());
    close(fd);
    if (!written) {
        unlink(path);
        return {};
    }
    auto artifacts = ArtifactSet::load_single_file(path);
    unlink(path);
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return {};
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    if (!std::holds_alternative<PackageView>(view)) return {};

    const auto f16 = [&](uint32_t id, TensorRole role, uint32_t columns) {
        SemanticTensor tensor;
        tensor.id = id;
        tensor.role = role;
        tensor.logical_type = ScalarType::F32;
        const uint32_t physical_columns = id == 2 && truncated_query_weight
            ? columns / 2 : columns;
        const size_t physical_bytes = id == 2 && truncated_query_weight
            ? tensor_bytes[id] / 2 : tensor_bytes[id];
        tensor.dimensions = {{DimensionKind::Constant, hidden},
                             {DimensionKind::Constant, physical_columns}};
        tensor.layout.rank = 2;
        tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        tensor.layout.strides[0] = 1;
        tensor.layout.strides[1] = hidden;
        tensor.planes = {{PlaneKind::Values, ScalarType::F16, ArtifactId{0}, tensor_offsets[id],
                          physical_bytes, 64, 0}};
        return tensor;
    };
    const auto f32 = [&](uint32_t id, TensorRole role) {
        SemanticTensor tensor;
        tensor.id = id;
        tensor.role = role;
        tensor.logical_type = ScalarType::F32;
        tensor.dimensions = {{DimensionKind::Constant, hidden}};
        tensor.layout.rank = 1;
        tensor.layout.axis_order = {0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        tensor.layout.strides[0] = 1;
        tensor.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, tensor_offsets[id],
                          tensor_bytes[id], 64, 0}};
        return tensor;
    };
    const auto value = [](uint32_t id, uint32_t width) {
        return SemanticValue{id, ScalarType::F32,
                             {{DimensionKind::Symbol, 1},
                              {DimensionKind::Constant, width}}, 0};
    };
    SemanticModel model;
    model.maximum_context = 4;
    model.vocabulary_size = vocabulary;
    model.bos_id = 0;
    model.eos_id = 2;
    model.tensors = {
        f16(0, TensorRole::TokenEmbedding, vocabulary),
        f32(1, TensorRole::AttentionNormWeight),
        f16(2, TensorRole::QueryWeight, hidden),
        f16(3, TensorRole::KeyWeight, hidden),
        f16(4, TensorRole::ValueWeight, hidden),
        f16(5, TensorRole::AttentionOutputWeight, hidden),
        f32(6, TensorRole::FfnNormWeight),
        f16(7, TensorRole::FfnGateWeight, hidden),
        f16(8, TensorRole::FfnUpWeight, hidden),
        f16(9, TensorRole::FfnDownWeight, hidden),
        f32(10, TensorRole::FinalNormWeight),
        f16(11, TensorRole::OutputWeight, vocabulary),
    };
    model.values = {
        value(0, hidden), value(1, hidden), value(2, hidden), value(3, hidden),
        value(4, hidden), value(5, hidden), value(6, hidden), value(7, hidden),
        value(8, hidden), value(9, hidden), value(10, hidden), value(11, hidden),
        value(12, hidden), value(13, hidden), value(14, hidden), value(15, hidden),
        value(16, hidden), value(17, vocabulary),
        SemanticValue{18, ScalarType::U32,
                      {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1}}, 0},
    };
    const auto add = [&](OperatorKind kind, std::vector<uint32_t> inputs,
                         std::vector<uint32_t> outputs, std::vector<uint32_t> tensors,
                         std::vector<uint32_t> states, OperatorPayload payload) {
        SemanticOperator op;
        op.id = static_cast<uint32_t>(model.operators.size());
        op.kind = kind;
        op.inputs = std::move(inputs);
        op.outputs = std::move(outputs);
        op.tensors = std::move(tensors);
        op.states = std::move(states);
        op.payload = std::move(payload);
        model.operators.push_back(std::move(op));
    };
    constexpr uint32_t one = 0x3f800000u;
    constexpr uint32_t epsilon = 0x358637bdu;
    add(OperatorKind::EmbeddingLookup, {18}, {0}, {0}, {},
        EmbeddingLookupPayload{embedding_scale_bits, vocabulary, hidden, 0});
    add(OperatorKind::RmsNorm, {0}, {1}, {1}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {1}, {2}, {2}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {3}, {3}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {4}, {4}, {}, LinearPayload{});
    add(OperatorKind::Rope, {2, 3}, {5, 6}, {}, {},
        RopePayload{RopePairing::HalfSplit, true, hidden, 0x49742400u, one});
    add(OperatorKind::CausalAttention, {5, 6, 4}, {7}, {}, {0, 1},
        CausalAttentionPayload{1, 1, hidden, attention_scale_bits,
                               AttentionMask::Causal, CachePolicy::Global});
    add(OperatorKind::Linear, {7}, {8}, {5}, {}, LinearPayload{});
    add(OperatorKind::Add, {0, 8}, {9}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {9}, {10}, {6}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {10}, {11}, {7}, {}, LinearPayload{});
    add(OperatorKind::Linear, {10}, {12}, {8}, {}, LinearPayload{});
    add(OperatorKind::SwiGlu, {11, 12}, {13}, {}, {}, SwiGluPayload{ActivationKind::Silu});
    add(OperatorKind::Linear, {13}, {14}, {9}, {}, LinearPayload{});
    add(OperatorKind::Add, {9, 14}, {15}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {15}, {16}, {10}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {16}, {17}, {11}, {}, LinearPayload{});
    model.input_values_first = 18;
    model.input_values_count = 1;
    model.output_values_first = 17;
    model.output_values_count = 1;
    model.layers = {{0, 1, 14, 0}};
    StateFormat key_format;
    key_format.encoded_domain = TransformDomain::RopeApplied;
    key_format.alignment = 64;
    StateFormat value_format;
    value_format.alignment = 64;
    model.states = {
        {0, StateKind::KeyCache, 1, StateUpdateKind::AppendKey, PositionPolicy::AppendOnly,
         {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1},
          {DimensionKind::Constant, hidden}}, {key_format}, 0},
        {1, StateKind::ValueCache, 1, StateUpdateKind::AppendValue, PositionPolicy::AppendOnly,
         {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1},
          {DimensionKind::Constant, hidden}}, {value_format}, 0},
    };
    const PackageView& package = std::get<PackageView>(view);
    return RuntimePackage::make_legacy_test_only(std::move(model), package, package.digest(),
                                                 package.digest(), 0, RuleQualificationState::Draft);
}

std::shared_ptr<const RuntimePackage> synthetic_mixed_attention_geometry_package() {
    const auto base = synthetic_f16_dense_package(
        0x3e800000u, false, false, 0x3f800000u, 64);
    if (!base || base->semantics().layers.size() != 1 ||
        base->semantics().operators.size() != 17 || base->semantics().states.size() != 2)
        return {};
    const std::span<const uint8_t> source = base->artifact_bytes(ArtifactId{0});
    if (source.empty()) return {};
    char path[] = "/private/tmp/laplace-mixed-attention-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) return {};
    const bool written = write(fd, source.data(), source.size()) ==
                         static_cast<ssize_t>(source.size());
    close(fd);
    if (!written) {
        unlink(path);
        return {};
    }
    auto artifacts = ArtifactSet::load_single_file(path);
    unlink(path);
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return {};
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    if (!std::holds_alternative<PackageView>(view)) return {};

    SemanticModel model = base->semantics();
    SemanticTensor narrow_key = model.tensors[3];
    narrow_key.id = static_cast<uint32_t>(model.tensors.size());
    narrow_key.dimensions[1] = {DimensionKind::Constant, 32};
    narrow_key.planes[0].length /= 2;
    const uint32_t narrow_key_tensor_id = narrow_key.id;
    model.tensors.push_back(std::move(narrow_key));
    SemanticTensor narrow_value = model.tensors[4];
    narrow_value.id = static_cast<uint32_t>(model.tensors.size());
    narrow_value.dimensions[1] = {DimensionKind::Constant, 32};
    narrow_value.planes[0].length /= 2;
    const uint32_t narrow_value_tensor_id = narrow_value.id;
    model.tensors.push_back(std::move(narrow_value));
    const std::vector<SemanticOperator> layer_template(
        model.operators.begin() + 1, model.operators.begin() + 15);
    SemanticOperator final_norm = model.operators[15];
    SemanticOperator output = model.operators[16];
    model.operators.resize(15);

    std::array<uint32_t, 16> value_map{};
    value_map[0] = 15;
    for (uint32_t old_id = 1; old_id <= 15; ++old_id) {
        SemanticValue value = model.values[old_id];
        value.id = static_cast<uint32_t>(model.values.size());
        if ((old_id == 3 || old_id == 4 || old_id == 6) && !value.dimensions.empty())
            value.dimensions.back() = {DimensionKind::Constant, 32};
        value_map[old_id] = value.id;
        model.values.push_back(std::move(value));
    }
    const auto remap_value = [&](uint32_t value_id) {
        return value_id <= 15 ? value_map[value_id] : value_id;
    };
    for (SemanticOperator op : layer_template) {
        for (uint32_t& input : op.inputs) input = remap_value(input);
        for (uint32_t& result : op.outputs) result = remap_value(result);
        for (uint32_t& tensor : op.tensors) {
            if (tensor == 3) tensor = narrow_key_tensor_id;
            if (tensor == 4) tensor = narrow_value_tensor_id;
        }
        for (uint32_t& state : op.states) state += 2;
        if (auto* rope = std::get_if<RopePayload>(&op.payload))
            rope->rotary_dimension = 32;
        if (auto* attention = std::get_if<CausalAttentionPayload>(&op.payload)) {
            attention->query_heads = 2;
            attention->kv_heads = 1;
            attention->head_dimension = 32;
        }
        op.id = static_cast<uint32_t>(model.operators.size());
        model.operators.push_back(std::move(op));
    }
    final_norm.inputs = {value_map[15]};
    final_norm.id = static_cast<uint32_t>(model.operators.size());
    model.operators.push_back(std::move(final_norm));
    output.id = static_cast<uint32_t>(model.operators.size());
    model.operators.push_back(std::move(output));
    model.layers.push_back({1, 15, 14, 0});

    SemanticState key = model.states[0];
    SemanticState value = model.states[1];
    key.id = 2;
    value.id = 3;
    key.dimensions[1] = {DimensionKind::Constant, 1};
    key.dimensions[2] = {DimensionKind::Constant, 32};
    value.dimensions[1] = {DimensionKind::Constant, 1};
    value.dimensions[2] = {DimensionKind::Constant, 32};
    model.states.push_back(std::move(key));
    model.states.push_back(std::move(value));

    const PackageView& package = std::get<PackageView>(view);
    return RuntimePackage::make_legacy_test_only(
        std::move(model), package, package.digest(), package.digest(), 0,
        RuleQualificationState::Draft);
}

enum class SyntheticU2FfnFormat {
    F16Nonzero,
    DirectU2Nonzero,
    F16Zero,
    Q4KZero,
};

std::shared_ptr<const RuntimePackage> synthetic_column_grouped_u2_dense_package(
    SyntheticU2FfnFormat ffn_format) {
    constexpr uint32_t hidden = 256;
    constexpr uint32_t intermediate = 256;
    constexpr uint32_t vocabulary = 3;
    struct Storage {
        uint64_t values_offset = 0;
        uint64_t values_bytes = 0;
        uint64_t scales_offset = 0;
        uint64_t scales_bytes = 0;
        uint64_t biases_offset = 0;
        uint64_t biases_bytes = 0;
    };
    std::array<Storage, 12> storage{};
    size_t total = 0;
    const auto reserve = [&](size_t bytes, size_t alignment) -> uint64_t {
        total = (total + alignment - 1u) & ~(alignment - 1u);
        const uint64_t offset = total;
        total += bytes;
        return offset;
    };
    const auto reserve_values = [&](uint32_t id, size_t bytes, size_t alignment = 64u) {
        storage[id].values_offset = reserve(bytes, alignment);
        storage[id].values_bytes = bytes;
    };
    reserve_values(0, static_cast<size_t>(hidden) * vocabulary * sizeof(uint16_t));
    reserve_values(1, static_cast<size_t>(hidden) * sizeof(float));
    for (uint32_t id : {2u, 3u, 4u, 5u})
        reserve_values(id, static_cast<size_t>(hidden) * hidden * sizeof(uint16_t));
    reserve_values(6, static_cast<size_t>(hidden) * sizeof(float));
    for (uint32_t id : {7u, 8u, 9u}) {
        if (ffn_format == SyntheticU2FfnFormat::DirectU2Nonzero) {
            ColumnGroupedAffineUInt2SkipV1Contract contract;
            ColumnGroupedAffineUInt2SkipV1Error error{};
            if (!column_grouped_affine_uint2_skip_v1_make_contract(
                    hidden, intermediate, &contract, &error))
                return {};
            storage[id].values_offset = reserve(contract.values_bytes, 128u);
            storage[id].values_bytes = contract.values_bytes;
            storage[id].scales_offset = reserve(contract.scale_bytes, 128u);
            storage[id].scales_bytes = contract.scale_bytes;
            storage[id].biases_offset = reserve(contract.bias_bytes, 128u);
            storage[id].biases_bytes = contract.bias_bytes;
        } else if (ffn_format == SyntheticU2FfnFormat::Q4KZero) {
            reserve_values(id, static_cast<size_t>(hidden) * intermediate / 256u * 144u,
                           256u);
        } else {
            reserve_values(id, static_cast<size_t>(hidden) * intermediate * sizeof(uint16_t));
        }
    }
    reserve_values(10, static_cast<size_t>(hidden) * sizeof(float));
    reserve_values(11, static_cast<size_t>(hidden) * vocabulary * sizeof(uint16_t));

    std::vector<uint8_t> bytes(total, 0);
    const auto put_f16 = [&](uint32_t id, size_t index, uint16_t value) {
        std::memcpy(bytes.data() + storage[id].values_offset + index * sizeof(value),
                    &value, sizeof(value));
    };
    const auto put_f32 = [&](uint32_t id, size_t index, float value) {
        std::memcpy(bytes.data() + storage[id].values_offset + index * sizeof(value),
                    &value, sizeof(value));
    };
    for (uint32_t token = 0; token != vocabulary; ++token) {
        for (uint32_t channel = 0; channel != hidden; ++channel) {
            constexpr std::array<uint16_t, 4> magnitudes = {
                0x2400u, 0x2800u, 0xa400u, 0x2c00u};
            put_f16(0, static_cast<size_t>(token) * hidden + channel,
                    magnitudes[(channel + token) % magnitudes.size()]);
        }
    }
    for (uint32_t id : {1u, 6u, 10u})
        for (uint32_t channel = 0; channel != hidden; ++channel)
            put_f32(id, channel, 1.0f);
    const auto diagonal = [&](uint32_t id, uint16_t value) {
        for (uint32_t channel = 0; channel != hidden; ++channel)
            put_f16(id, static_cast<size_t>(channel) * hidden + channel, value);
    };
    for (uint32_t id : {2u, 3u, 4u}) diagonal(id, 0x3c00u);
    diagonal(5, 0x3800u);

    constexpr std::array<uint16_t, 4> decoded_halves = {
        0x9c00u, 0x0000u, 0x1c00u, 0x2000u};
    for (uint32_t id : {7u, 8u, 9u}) {
        if (ffn_format == SyntheticU2FfnFormat::DirectU2Nonzero) {
            auto* packed = bytes.data() + storage[id].values_offset;
            auto* scales = reinterpret_cast<uint16_t*>(
                bytes.data() + storage[id].scales_offset);
            auto* biases = reinterpret_cast<uint16_t*>(
                bytes.data() + storage[id].biases_offset);
            for (uint32_t column = 0; column != hidden; ++column) {
                for (uint32_t row = 0; row != intermediate; ++row) {
                    const uint8_t code = static_cast<uint8_t>((row + column + id) & 3u);
                    packed[static_cast<size_t>(column) * 64u + row / 4u] |=
                        static_cast<uint8_t>(code << (2u * (row & 3u)));
                }
                scales[column] = 0x1c00u;
                biases[column] = 0x9c00u;
            }
        } else if (ffn_format == SyntheticU2FfnFormat::F16Nonzero) {
            for (uint32_t row = 0; row != intermediate; ++row) {
                for (uint32_t column = 0; column != hidden; ++column) {
                    const uint8_t code = static_cast<uint8_t>((row + column + id) & 3u);
                    put_f16(id, static_cast<size_t>(row) * hidden + column,
                            decoded_halves[code]);
                }
            }
        }
    }
    put_f16(11, 0, 0x3c00u);
    put_f16(11, static_cast<size_t>(hidden) + 1u, 0x3800u);
    put_f16(11, static_cast<size_t>(2u) * hidden + 2u, 0xb800u);

    char path[] = "/private/tmp/laplace-column-u2-canonical-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) return {};
    const bool written = write(fd, bytes.data(), bytes.size()) ==
                         static_cast<ssize_t>(bytes.size());
    close(fd);
    if (!written) {
        unlink(path);
        return {};
    }
    auto artifacts = ArtifactSet::load_single_file(path);
    unlink(path);
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return {};
    auto view = std::get<ArtifactSet>(std::move(artifacts)).view(ArtifactId{0});
    if (!std::holds_alternative<PackageView>(view)) return {};

    const auto f16 = [&](uint32_t id, TensorRole role, uint32_t output,
                         uint32_t input = hidden) {
        SemanticTensor tensor;
        tensor.id = id;
        tensor.role = role;
        tensor.logical_type = ScalarType::F32;
        tensor.dimensions = {{DimensionKind::Constant, output},
                             {DimensionKind::Constant, input}};
        tensor.layout.rank = 2;
        tensor.layout.axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        tensor.layout.strides[0] = 1;
        tensor.layout.strides[1] = input;
        tensor.planes = {{PlaneKind::Values, ScalarType::F16, ArtifactId{0},
                          storage[id].values_offset, storage[id].values_bytes, 64, 0}};
        return tensor;
    };
    const auto f32 = [&](uint32_t id, TensorRole role) {
        SemanticTensor tensor;
        tensor.id = id;
        tensor.role = role;
        tensor.logical_type = ScalarType::F32;
        tensor.dimensions = {{DimensionKind::Constant, hidden}};
        tensor.layout.rank = 1;
        tensor.layout.axis_order = {0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        tensor.layout.strides[0] = 1;
        tensor.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0},
                          storage[id].values_offset, storage[id].values_bytes, 64, 0}};
        return tensor;
    };
    const auto u2 = [&](uint32_t id, TensorRole role, uint32_t output,
                        uint32_t input) {
        SemanticTensor tensor;
        tensor.id = id;
        tensor.role = role;
        tensor.logical_type = ScalarType::F32;
        tensor.dimensions = {{DimensionKind::Constant, output},
                             {DimensionKind::Constant, input}};
        tensor.layout.kind = PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip;
        tensor.layout.packing = PackingKind::LsbBitPacked;
        tensor.layout.rank = 2;
        tensor.layout.block_rank = 1;
        tensor.layout.axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        tensor.layout.strides[0] = 1;
        tensor.layout.strides[1] = input;
        tensor.layout.block_elements = 256;
        tensor.layout.block_bytes = 64;
        tensor.quantization.kind = QuantizationKind::BlockedAffine;
        tensor.quantization.accumulation_type = ScalarType::F32;
        tensor.quantization.scale_type = ScalarType::F16;
        tensor.quantization.bias_type = ScalarType::F16;
        tensor.quantization.block_elements = 256;
        tensor.quantization.block_bytes = 64;
        tensor.quantization.group_size = 256;
        tensor.quantization.required_plane_mask = 7;
        tensor.planes = {
            {PlaneKind::Values, ScalarType::U8, ArtifactId{0},
             storage[id].values_offset, storage[id].values_bytes, 128, 0},
            {PlaneKind::Scales, ScalarType::F16, ArtifactId{0},
             storage[id].scales_offset, storage[id].scales_bytes, 128, 0},
            {PlaneKind::Biases, ScalarType::F16, ArtifactId{0},
             storage[id].biases_offset, storage[id].biases_bytes, 128, 0},
        };
        return tensor;
    };
    const auto q4k = [&](uint32_t id, TensorRole role, uint32_t output,
                         uint32_t input) {
        SemanticTensor tensor;
        tensor.id = id;
        tensor.role = role;
        tensor.logical_type = ScalarType::F32;
        tensor.dimensions = {{DimensionKind::Constant, output},
                             {DimensionKind::Constant, input}};
        tensor.layout.kind = PhysicalLayoutKind::GgufBlocked;
        tensor.layout.packing = PackingKind::Gguf;
        tensor.layout.rank = 2;
        tensor.layout.block_rank = 1;
        tensor.layout.axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        tensor.layout.strides[0] = 1;
        tensor.layout.strides[1] = input;
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
        tensor.planes = {{PlaneKind::Values, ScalarType::U8, ArtifactId{0},
                          storage[id].values_offset, storage[id].values_bytes, 256, 0}};
        return tensor;
    };
    const auto value = [](uint32_t id, uint32_t width) {
        return SemanticValue{id, ScalarType::F32,
                             {{DimensionKind::Symbol, 1},
                              {DimensionKind::Constant, width}}, 0};
    };

    SemanticModel model;
    model.maximum_context = 4;
    model.vocabulary_size = vocabulary;
    model.bos_id = 0;
    model.eos_id = 2;
    model.tensors = {
        f16(0, TensorRole::TokenEmbedding, vocabulary),
        f32(1, TensorRole::AttentionNormWeight),
        f16(2, TensorRole::QueryWeight, hidden),
        f16(3, TensorRole::KeyWeight, hidden),
        f16(4, TensorRole::ValueWeight, hidden),
        f16(5, TensorRole::AttentionOutputWeight, hidden),
        f32(6, TensorRole::FfnNormWeight),
        ffn_format == SyntheticU2FfnFormat::DirectU2Nonzero
            ? u2(7, TensorRole::FfnGateWeight, intermediate, hidden)
            : (ffn_format == SyntheticU2FfnFormat::Q4KZero
                   ? q4k(7, TensorRole::FfnGateWeight, intermediate, hidden)
                   : f16(7, TensorRole::FfnGateWeight, intermediate)),
        ffn_format == SyntheticU2FfnFormat::DirectU2Nonzero
            ? u2(8, TensorRole::FfnUpWeight, intermediate, hidden)
            : (ffn_format == SyntheticU2FfnFormat::Q4KZero
                   ? q4k(8, TensorRole::FfnUpWeight, intermediate, hidden)
                   : f16(8, TensorRole::FfnUpWeight, intermediate)),
        ffn_format == SyntheticU2FfnFormat::DirectU2Nonzero
            ? u2(9, TensorRole::FfnDownWeight, hidden, intermediate)
            : (ffn_format == SyntheticU2FfnFormat::Q4KZero
                   ? q4k(9, TensorRole::FfnDownWeight, hidden, intermediate)
                   : f16(9, TensorRole::FfnDownWeight, hidden, intermediate)),
        f32(10, TensorRole::FinalNormWeight),
        f16(11, TensorRole::OutputWeight, vocabulary),
    };
    model.values = {
        value(0, hidden), value(1, hidden), value(2, hidden), value(3, hidden),
        value(4, hidden), value(5, hidden), value(6, hidden), value(7, hidden),
        value(8, hidden), value(9, hidden), value(10, hidden), value(11, hidden),
        value(12, hidden), value(13, hidden), value(14, hidden), value(15, hidden),
        value(16, hidden), value(17, vocabulary),
    };
    const auto add = [&](OperatorKind kind, std::vector<uint32_t> inputs,
                         std::vector<uint32_t> outputs, std::vector<uint32_t> tensors,
                         std::vector<uint32_t> states, OperatorPayload payload) {
        SemanticOperator op;
        op.id = static_cast<uint32_t>(model.operators.size());
        op.kind = kind;
        op.inputs = std::move(inputs);
        op.outputs = std::move(outputs);
        op.tensors = std::move(tensors);
        op.states = std::move(states);
        op.payload = std::move(payload);
        model.operators.push_back(std::move(op));
    };
    constexpr uint32_t one = 0x3f800000u;
    constexpr uint32_t epsilon = 0x358637bdu;
    add(OperatorKind::EmbeddingLookup, {0}, {0}, {0}, {},
        EmbeddingLookupPayload{one, vocabulary, hidden, 0});
    add(OperatorKind::RmsNorm, {0}, {1}, {1}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {1}, {2}, {2}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {3}, {3}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {4}, {4}, {}, LinearPayload{});
    add(OperatorKind::Rope, {2, 3}, {5, 6}, {}, {},
        RopePayload{RopePairing::HalfSplit, true, hidden, 0x49742400u, one});
    add(OperatorKind::CausalAttention, {5, 6, 4}, {7}, {}, {0, 1},
        CausalAttentionPayload{1, 1, hidden, 0x3d800000u,
                               AttentionMask::Causal, CachePolicy::Global});
    add(OperatorKind::Linear, {7}, {8}, {5}, {}, LinearPayload{});
    add(OperatorKind::Add, {0, 8}, {9}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {9}, {10}, {6}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {10}, {11}, {7}, {}, LinearPayload{});
    add(OperatorKind::Linear, {10}, {12}, {8}, {}, LinearPayload{});
    add(OperatorKind::SwiGlu, {11, 12}, {13}, {}, {}, SwiGluPayload{ActivationKind::Silu});
    add(OperatorKind::Linear, {13}, {14}, {9}, {}, LinearPayload{});
    add(OperatorKind::Add, {9, 14}, {15}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {15}, {16}, {10}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {16}, {17}, {11}, {}, LinearPayload{});
    model.layers = {{0, 1, 14, 0}};
    StateFormat key_format;
    key_format.encoded_domain = TransformDomain::RopeApplied;
    key_format.alignment = 64;
    StateFormat value_format;
    value_format.alignment = 64;
    model.states = {
        {0, StateKind::KeyCache, 1, StateUpdateKind::AppendKey,
         PositionPolicy::AppendOnly,
         {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1},
          {DimensionKind::Constant, hidden}}, {key_format}, 0},
        {1, StateKind::ValueCache, 1, StateUpdateKind::AppendValue,
         PositionPolicy::AppendOnly,
         {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1},
          {DimensionKind::Constant, hidden}}, {value_format}, 0},
    };
    const PackageView& package = std::get<PackageView>(view);
    return RuntimePackage::make_legacy_test_only(
        std::move(model), package, package.digest(), package.digest(), 0,
        RuleQualificationState::Draft);
}

enum class CanonicalMoeFixtureDefect {
    None,
    OperatorOrder,
    SequentialResidual,
    ExtraIndependentBranch,
    Ambiguous,
    MissingEdge,
    Cyclic,
    KeyStateAlias,
    MissingRouterNormalization,
    WrongRouterNormalization,
    WrongActivation,
    WrongQ5Contract,
    WrongQ8Contract,
    SchemaVersion,
};

std::shared_ptr<const RuntimePackage> synthetic_q4k_moe_package(
    CanonicalMoeFixtureDefect defect = CanonicalMoeFixtureDefect::None) {
    constexpr uint32_t hidden = 512, vocabulary = 3, intermediate = 512;
    constexpr uint32_t expert_intermediate = 256, experts = 4, selected = 2;
    constexpr size_t q4_block_bytes = sizeof(quant_ref::block_q4_K);
    constexpr size_t q5_block_bytes = 22;
    const size_t q4_up_bytes = static_cast<size_t>(experts) * hidden * (2 * expert_intermediate) / 256 * q4_block_bytes;
    const size_t q5_down_bytes = static_cast<size_t>(experts) * expert_intermediate * hidden / 32 * q5_block_bytes;
    const std::array<size_t, 18> tensor_bytes = {
        static_cast<size_t>(hidden) * vocabulary * sizeof(uint16_t),
        static_cast<size_t>(hidden) * sizeof(float),
        static_cast<size_t>(hidden) * hidden * sizeof(uint16_t),
        static_cast<size_t>(hidden) * hidden * sizeof(uint16_t),
        static_cast<size_t>(hidden) * hidden * sizeof(uint16_t),
        static_cast<size_t>(hidden) * hidden * sizeof(uint16_t),
        static_cast<size_t>(hidden) * sizeof(float),
        static_cast<size_t>(hidden) * intermediate * sizeof(uint16_t),
        static_cast<size_t>(hidden) * intermediate * sizeof(uint16_t),
        static_cast<size_t>(intermediate) * hidden * sizeof(uint16_t),
        static_cast<size_t>(hidden) * sizeof(float),
        static_cast<size_t>(hidden) * sizeof(float),
        static_cast<size_t>(experts) * sizeof(float),
        static_cast<size_t>(hidden) * experts * sizeof(float), q4_up_bytes, q5_down_bytes,
        static_cast<size_t>(hidden) * sizeof(float),
        static_cast<size_t>(hidden) * vocabulary * sizeof(uint16_t),
    };
    std::array<uint64_t, 18> tensor_offsets{};
    size_t total = 0;
    for (size_t index = 0; index != tensor_bytes.size(); ++index) {
        total = (total + 63) & ~size_t{63};
        tensor_offsets[index] = total;
        total += tensor_bytes[index];
    }
    std::vector<uint8_t> bytes(total, 0);
    // Keep the projections zero so the small scalar oracle remains simple,
    // but make the bound F32 weight planes non-zero so registration and
    // native bindings cannot pass on an all-zero placeholder package.
    const float one_weight = 1.0f;
    for (uint32_t tensor_id : {1u, 6u, 10u, 11u, 12u, 13u, 16u})
        std::memcpy(bytes.data() + tensor_offsets[tensor_id], &one_weight, sizeof(one_weight));
    char path[] = "/private/tmp/laplace-canonical-moe-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) return {};
    const bool written = write(fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size());
    close(fd);
    if (!written) { unlink(path); return {}; }
    auto artifacts = ArtifactSet::load_single_file(path);
    unlink(path);
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return {};
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    if (!std::holds_alternative<PackageView>(view)) return {};

    const auto f16 = [&](uint32_t id, TensorRole role, uint32_t columns) {
        SemanticTensor tensor;
        tensor.id = id; tensor.role = role; tensor.logical_type = ScalarType::F32;
        tensor.dimensions = {{DimensionKind::Constant, hidden}, {DimensionKind::Constant, columns}};
        tensor.layout.rank = 2; tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        tensor.layout.strides[0] = 1; tensor.layout.strides[1] = hidden;
        tensor.planes = {{PlaneKind::Values, ScalarType::F16, ArtifactId{0}, tensor_offsets[id], tensor_bytes[id], 64, 0}};
        return tensor;
    };
    const auto f32 = [&](uint32_t id, TensorRole role, uint32_t columns = hidden) {
        SemanticTensor tensor;
        tensor.id = id; tensor.role = role; tensor.logical_type = ScalarType::F32;
        tensor.dimensions = columns == hidden
            ? std::vector<Dimension>{{DimensionKind::Constant, hidden}}
            : std::vector<Dimension>{{DimensionKind::Constant, columns}, {DimensionKind::Constant, hidden}};
        tensor.layout.rank = static_cast<uint8_t>(tensor.dimensions.size());
        tensor.layout.axis_order = tensor.layout.rank == 1
            ? std::array<uint8_t, 8>{0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}
            : std::array<uint8_t, 8>{1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        tensor.layout.strides[0] = 1;
        if (tensor.layout.rank == 2) tensor.layout.strides[1] = hidden;
        tensor.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, tensor_offsets[id], tensor_bytes[id], 64, 0}};
        return tensor;
    };
    const auto f32_vector = [&](uint32_t id, TensorRole role, uint32_t width) {
        SemanticTensor tensor;
        tensor.id = id; tensor.role = role; tensor.logical_type = ScalarType::F32;
        tensor.dimensions = {{DimensionKind::Constant, width}};
        tensor.layout.rank = 1;
        tensor.layout.axis_order = {0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        tensor.layout.strides[0] = 1;
        tensor.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, tensor_offsets[id], tensor_bytes[id], 64, 0}};
        return tensor;
    };
    const auto expert_q4 = [&](uint32_t id, TensorRole role, uint32_t input, uint32_t output,
                               uint32_t block_elements = 256, uint32_t block_bytes = q4_block_bytes) {
        SemanticTensor tensor;
        tensor.id = id; tensor.role = role; tensor.logical_type = ScalarType::F32;
        tensor.dimensions = {{DimensionKind::Constant, experts}, {DimensionKind::Constant, input},
                             {DimensionKind::Constant, output}};
        tensor.layout.kind = PhysicalLayoutKind::GgufBlocked; tensor.layout.rank = 3;
        tensor.layout.packing = PackingKind::Gguf; tensor.layout.block_rank = 1;
        tensor.layout.axis_order = {1, 2, 0, 0xff, 0xff, 0xff, 0xff, 0xff};
        tensor.layout.strides[0] = 1; tensor.layout.strides[1] = input;
        tensor.layout.strides[2] = static_cast<uint64_t>(input) * output;
        tensor.layout.block_elements = block_elements; tensor.layout.block_bytes = block_bytes;
        tensor.quantization.kind = QuantizationKind::BlockedAffine;
        tensor.quantization.block_elements = block_elements;
        tensor.quantization.block_bytes = block_bytes;
        tensor.quantization.group_size = block_elements;
        tensor.quantization.required_plane_mask = 1;
        tensor.expert_axis = {ExpertAxisKind::ExpertBank, 0, 0xff, 1, 2, experts,
                              static_cast<uint64_t>(input) * output / block_elements * block_bytes, 0};
        tensor.planes = {{PlaneKind::Values, ScalarType::U8, ArtifactId{0}, tensor_offsets[id], tensor_bytes[id], 64, 0}};
        return tensor;
    };
    const auto value = [](uint32_t id, ScalarType type, std::vector<Dimension> dimensions) {
        return SemanticValue{id, type, std::move(dimensions), 0};
    };
    const auto rows = [](uint32_t width) {
        return std::vector<Dimension>{{DimensionKind::Symbol, 1}, {DimensionKind::Constant, width}};
    };
    const auto routed = [](uint32_t width) {
        return std::vector<Dimension>{{DimensionKind::Symbol, 1}, {DimensionKind::Constant, selected},
                                      {DimensionKind::Constant, width}};
    };

    SemanticModel model;
    model.schema_major = 7; model.opset_major = 7; model.maximum_context = 4;
    model.vocabulary_size = vocabulary; model.bos_id = 0; model.eos_id = 2;
    model.tensors = {
        f16(0, TensorRole::TokenEmbedding, vocabulary), f32(1, TensorRole::AttentionNormWeight),
        f16(2, TensorRole::QueryWeight, hidden), f16(3, TensorRole::KeyWeight, hidden),
        f16(4, TensorRole::ValueWeight, hidden), f16(5, TensorRole::AttentionOutputWeight, hidden),
        f32(6, TensorRole::FfnNormWeight), f16(7, TensorRole::FfnGateWeight, intermediate),
        f16(8, TensorRole::FfnUpWeight, intermediate), f16(9, TensorRole::FfnDownWeight, hidden),
        f32(10, TensorRole::RouterScaleWeight), f32(11, TensorRole::ExpertNormWeight),
        f32_vector(12, TensorRole::ReduceScaleWeight, experts),
        f32(13, TensorRole::NextnProjectionWeight, experts),
        expert_q4(14, TensorRole::FfnUpWeight, hidden, 2 * expert_intermediate),
        expert_q4(15, TensorRole::FfnDownWeight, expert_intermediate, hidden, 32, q5_block_bytes),
        f32(16, TensorRole::FinalNormWeight), f16(17, TensorRole::OutputWeight, vocabulary),
    };
    model.values = {
        value(0, ScalarType::F32, rows(hidden)), value(1, ScalarType::F32, rows(hidden)),
        value(2, ScalarType::F32, rows(hidden)), value(3, ScalarType::F32, rows(hidden)),
        value(4, ScalarType::F32, rows(hidden)), value(5, ScalarType::F32, rows(hidden)),
        value(6, ScalarType::F32, rows(hidden)), value(7, ScalarType::F32, rows(hidden)),
        value(8, ScalarType::F32, rows(hidden)), value(9, ScalarType::F32, rows(hidden)),
        value(10, ScalarType::F32, rows(hidden)), value(11, ScalarType::F32, rows(intermediate)),
        value(12, ScalarType::F32, rows(intermediate)), value(13, ScalarType::F32, rows(intermediate)),
        value(14, ScalarType::F32, rows(hidden)), value(15, ScalarType::F32, rows(hidden)),
        value(16, ScalarType::F32, rows(hidden)), value(17, ScalarType::F32, rows(hidden)),
        value(18, ScalarType::F32, rows(experts)), value(19, ScalarType::U32, rows(selected)),
        value(20, ScalarType::F32, rows(selected)), value(21, ScalarType::F32, rows(hidden)),
        value(22, ScalarType::F32, routed(2 * expert_intermediate)),
        value(23, ScalarType::F32, routed(expert_intermediate)), value(24, ScalarType::F32, routed(expert_intermediate)),
        value(25, ScalarType::F32, routed(expert_intermediate)), value(26, ScalarType::F32, routed(hidden)),
        value(27, ScalarType::F32, rows(hidden)), value(28, ScalarType::F32, rows(hidden)),
        value(29, ScalarType::F32, rows(hidden)), value(30, ScalarType::F32, rows(vocabulary)),
        value(31, ScalarType::F32, rows(hidden)), value(32, ScalarType::F32, rows(hidden)),
        value(33, ScalarType::F32, rows(experts)), value(34, ScalarType::U32, rows(1)),
    };
    const auto add = [&](OperatorKind kind, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                         std::vector<uint32_t> tensors, std::vector<uint32_t> states, OperatorPayload payload) {
        SemanticOperator op;
        op.id = static_cast<uint32_t>(model.operators.size()); op.kind = kind; op.semantic_version = 7;
        op.inputs = std::move(inputs); op.outputs = std::move(outputs); op.tensors = std::move(tensors);
        op.states = std::move(states); op.payload = std::move(payload); model.operators.push_back(std::move(op));
    };
    constexpr uint32_t one = 0x3f800000u, epsilon = 0x358637bdu;
    CausalAttentionPayload attention;
    attention.query_heads = 1; attention.kv_heads = 1; attention.head_dimension = hidden;
    attention.scale_f32_bits = 0x3e800000u; attention.mask = AttentionMask::Causal;
    attention.cache_policy = CachePolicy::Global; attention.value_source = ValueSource::SeparateProjection;
    attention.value_source_value = 4;
    RouterTopKPayload router;
    router.expert_count = experts; router.selected_count = selected; router.score_domain = RouterScoreDomain::Logits;
    router.normalization_order = RouterNormalizationOrder::NormalizeThenSelect;
    router.selected_weight_normalization =
        SelectedWeightNormalization::RenormalizeSelectedProbabilities;
    router.tie_policy = RouterTiePolicy::LowestExpertId; router.weight_source = RouterWeightSource::SelectedNormalizedScore;
    add(OperatorKind::EmbeddingLookup, {34}, {0}, {0}, {}, EmbeddingLookupPayload{one, vocabulary, hidden, 0});
    add(OperatorKind::RmsNorm, {0}, {1}, {1}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {1}, {2}, {2}, {}, LinearPayload{}); add(OperatorKind::Linear, {1}, {3}, {3}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {4}, {4}, {}, LinearPayload{});
    add(OperatorKind::Rope, {2, 3}, {5, 6}, {}, {}, RopePayload{RopePairing::HalfSplit, true, hidden, 0x49742400u, one});
    add(OperatorKind::CausalAttention, {5, 6, 4}, {7}, {}, {0, 1}, attention);
    add(OperatorKind::Linear, {7}, {8}, {5}, {}, LinearPayload{}); add(OperatorKind::Add, {0, 8}, {9}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {9}, {10}, {6}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {10}, {11}, {7}, {}, LinearPayload{}); add(OperatorKind::Linear, {10}, {12}, {8}, {}, LinearPayload{});
    add(OperatorKind::SwiGlu, {11, 12}, {13}, {}, {}, SwiGluPayload{ActivationKind::Silu});
    add(OperatorKind::Linear, {13}, {14}, {9}, {}, LinearPayload{});
    add(OperatorKind::Add, {14, 27}, {28}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {9}, {16}, {}, {}, RmsNormPayload{epsilon, -1, 0});
    add(OperatorKind::Scale, {16}, {31}, {10}, {}, ScalePayload{ScaleSource::Tensor, 0});
    add(OperatorKind::Scale, {31}, {32}, {}, {}, ScalePayload{ScaleSource::LiteralF32, 0x3d3504f3u});
    add(OperatorKind::Linear, {32}, {33}, {13}, {}, LinearPayload{}); add(OperatorKind::RouterTopK, {33}, {19, 20}, {}, {}, router);
    add(OperatorKind::RmsNorm, {9}, {21}, {11}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::RoutedLinear, {21, 19, 20}, {22}, {14}, {}, RoutedLinearPayload{ScalarType::F32});
    add(OperatorKind::AxisSplit, {22}, {23, 24}, {}, {}, AxisSplitPayload{expert_intermediate, expert_intermediate});
    add(OperatorKind::GatedActivation, {23, 24}, {25}, {}, {}, GatedActivationPayload{ActivationKind::GeluTanh});
    add(OperatorKind::RoutedLinear, {25, 19, 20}, {26}, {15}, {}, RoutedLinearPayload{ScalarType::F32});
    add(OperatorKind::WeightedExpertReduce, {26, 19, 20}, {27}, {12}, {},
        WeightedExpertReducePayload{ExpertReduceAssociation::SelectedOrderLeftToRight, ExpertScaleSource::PerExpertTensor, ScalarType::F32});
    add(OperatorKind::Add, {9, 28}, {15}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {15}, {29}, {16}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {29}, {30}, {17}, {}, LinearPayload{});
    model.input_values_first = 34;
    model.input_values_count = 1;
    model.output_values_first = 30;
    model.output_values_count = 1;
    // The embedding and final output head are graph-level operators. The
    // executable token layer contains operator indices 1..26 inclusive:
    // 26 operators ending at the final residual add. The final norm and
    // output head remain graph-level.
    model.layers = {{0, 1, 26, 0}};
    StateFormat key_format; key_format.encoded_domain = TransformDomain::RopeApplied; key_format.alignment = 64;
    StateFormat value_format; value_format.alignment = 64;
    model.states = {
        {0, StateKind::KeyCache, 7, StateUpdateKind::AppendKey, PositionPolicy::AppendOnly,
         {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1}, {DimensionKind::Constant, hidden}}, {key_format}, 0},
        {1, StateKind::ValueCache, 7, StateUpdateKind::AppendValue, PositionPolicy::AppendOnly,
         {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1}, {DimensionKind::Constant, hidden}}, {value_format}, 0},
    };
    if (defect == CanonicalMoeFixtureDefect::OperatorOrder) {
        // These independent branches are serialized in the opposite order;
        // their typed producer/consumer edges remain unchanged.
        std::swap(model.operators[16], model.operators[20]);
    }
    if (defect == CanonicalMoeFixtureDefect::SequentialResidual) {
        model.operators[14].inputs = {9, 14};
        model.operators[14].outputs = {15};
        model.operators[15].inputs = {15};
        model.operators[20].inputs = {15};
        model.operators[26].inputs = {15, 27};
        model.operators[26].outputs = {28};
        model.operators[27].inputs = {28};
    }
    if (defect == CanonicalMoeFixtureDefect::ExtraIndependentBranch ||
        defect == CanonicalMoeFixtureDefect::Ambiguous) {
        if (defect == CanonicalMoeFixtureDefect::ExtraIndependentBranch) {
            model.values.push_back(value(35, ScalarType::F32, rows(hidden)));
            SemanticOperator extra;
            extra.id = 100;
            extra.kind = OperatorKind::Scale;
            extra.semantic_version = 7;
            extra.inputs = {0};
            extra.outputs = {35};
            extra.payload = ScalePayload{ScaleSource::LiteralF32, one};
            model.operators.insert(model.operators.begin() + 27, std::move(extra));
        } else {
            SemanticOperator duplicate = model.operators[19];
            duplicate.id = 101;
            model.operators.insert(model.operators.begin() + 27, std::move(duplicate));
        }
        model.layers.front().operator_count += 1;
    }
    if (defect == CanonicalMoeFixtureDefect::MissingEdge) {
        for (SemanticOperator& op : model.operators) {
            if (op.kind == OperatorKind::WeightedExpertReduce) {
                op.inputs[0] = 999;
                break;
            }
        }
    }
    if (defect == CanonicalMoeFixtureDefect::Cyclic) {
        for (SemanticOperator& op : model.operators) {
            // Feed the final residual back into the attention residual.  All
            // typed edge witnesses still exist, but the value graph now has
            // a real final-add -> attention-residual cycle for Kahn to reject.
            if (op.kind == OperatorKind::Add && op.outputs == std::vector<uint32_t>{9}) {
                op.inputs[0] = 28;
                break;
            }
        }
    }
    if (defect == CanonicalMoeFixtureDefect::KeyStateAlias) {
        auto& payload = std::get<CausalAttentionPayload>(model.operators[6].payload);
        payload.value_source = ValueSource::KeyStateAlias;
    }
    if (defect == CanonicalMoeFixtureDefect::MissingRouterNormalization) {
        for (SemanticOperator& op : model.operators) {
            if (op.kind == OperatorKind::Scale && op.outputs == std::vector<uint32_t>{32}) {
                op.kind = OperatorKind::Add;
                op.payload = AddPayload{};
                break;
            }
        }
    }
    if (defect == CanonicalMoeFixtureDefect::WrongRouterNormalization) {
        for (SemanticOperator& op : model.operators) {
            if (op.kind == OperatorKind::Scale && op.outputs == std::vector<uint32_t>{32}) {
                std::get<ScalePayload>(op.payload).literal_f32_bits = 0x3c23d70au;
                break;
            }
        }
    }
    if (defect == CanonicalMoeFixtureDefect::WrongActivation) {
        for (SemanticOperator& op : model.operators) {
            if (op.kind == OperatorKind::GatedActivation) {
                std::get<GatedActivationPayload>(op.payload).activation = ActivationKind::Silu;
                break;
            }
        }
    }
    if (defect == CanonicalMoeFixtureDefect::WrongQ5Contract) {
        model.tensors[15].layout.block_bytes = 144;
        model.tensors[15].quantization.block_bytes = 144;
    }
    if (defect == CanonicalMoeFixtureDefect::WrongQ8Contract) {
        model.tensors[15].layout.block_bytes = 34;
        model.tensors[15].quantization.block_bytes = 34;
    }
    if (defect == CanonicalMoeFixtureDefect::SchemaVersion) {
        model.schema_major = 99;
        model.opset_major = 99;
    }
    const PackageView& package = std::get<PackageView>(view);
    return RuntimePackage::make_legacy_test_only(std::move(model), package, package.digest(), package.digest(), 0,
                                                 RuleQualificationState::Draft);
}

void test_canonical_moe_edge_matcher_is_order_independent() {
    const auto admitted = [](CanonicalMoeFixtureDefect defect) {
        const auto package = synthetic_q4k_moe_package(defect);
        if (!package) return false;
        CanonicalMoeOperatorEdges edges;
        return match_canonical_moe_operator_edges(package->semantics(),
                                                  package->semantics().layers.front(), edges);
    };
    CHECK(admitted(CanonicalMoeFixtureDefect::None));
    CHECK(admitted(CanonicalMoeFixtureDefect::SchemaVersion));
    CHECK(admitted(CanonicalMoeFixtureDefect::OperatorOrder));
    CHECK(!admitted(CanonicalMoeFixtureDefect::SequentialResidual));
    CHECK(!admitted(CanonicalMoeFixtureDefect::ExtraIndependentBranch));
    CHECK(!admitted(CanonicalMoeFixtureDefect::Ambiguous));
    CHECK(!admitted(CanonicalMoeFixtureDefect::MissingEdge));
    CHECK(!admitted(CanonicalMoeFixtureDefect::Cyclic));
    CHECK(!admitted(CanonicalMoeFixtureDefect::KeyStateAlias));
    CHECK(!admitted(CanonicalMoeFixtureDefect::MissingRouterNormalization));
    CHECK(!admitted(CanonicalMoeFixtureDefect::WrongRouterNormalization));
    CHECK(!admitted(CanonicalMoeFixtureDefect::WrongActivation));
    CHECK(!admitted(CanonicalMoeFixtureDefect::WrongQ5Contract));
    CHECK(!admitted(CanonicalMoeFixtureDefect::WrongQ8Contract));
    const auto package = synthetic_q4k_moe_package();
    CHECK(package != nullptr);
    if (package) {
        CanonicalMoeOperatorEdges edges;
        CHECK(match_canonical_moe_operator_edges(package->semantics(),
                                                  package->semantics().layers.front(), edges));
        CHECK(edges.covered_operator_ids.size() == package->semantics().layers.front().operator_count);
        CHECK(edges.router_normalization_scale != nullptr);
        CHECK(edges.router_normalization_scale_bits == 0x3d3504f3u);
        CHECK(edges.value_source == ValueSource::SeparateProjection);
        MetalTokLayer layer;
        layer.moe_router_normalization_scale_bits = edges.router_normalization_scale_bits;
        CHECK(layer.moe_router_normalization_scale_bits == 0x3d3504f3u);
    }
}

void test_canonical_moe_routes_the_exact_lapir007_pattern() {
    if (!metal_device_present() || !metal_available()) {
        std::fprintf(stderr, "SKIP: canonical MoE native fixture requires a Metal device and library\n");
        return;
    }
    const auto package = synthetic_q4k_moe_package();
    CHECK(package != nullptr);
    if (!package) return;
    const auto readiness_session = metal_tok_session_create();
    CHECK(readiness_session != nullptr);
    if (!readiness_session) return;
    const MetalTokMoeCapabilities readiness =
        metal_tok_session_moe_capabilities(*readiness_session);
    CHECK(readiness.router_topk);
    CHECK(readiness.gate_up_q4_k);
    CHECK(readiness.down_q5_0);
    CHECK(readiness.down_q8_0);
    CHECK(readiness.reduce);
    auto created = create_qualification_canonical_metal_program(package, 4);
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) {
        const auto& report = std::get<CompatibilityReport>(created);
        std::fprintf(stderr, "canonical MoE planner rejected LAPIR007: code=%u detail=%s\n",
                     static_cast<unsigned>(report.code), report.detail.c_str());
    }
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return;
    CHECK(metal_device_present());
    CHECK(metal_available());
    if (!metal_device_present() || !metal_available()) return;

    // The projections and embeddings are zero; the bound F32 planes are
    // non-zero, so this still exercises real registration and binding.
    // Compute its router, SwiGLU, and weighted expert reduction independently:
    // zero scores select the two lowest expert IDs; softmax gives 1/2 each;
    // zero gate/up/down projections keep the full final-logit vector at zero.
    const std::array<float, 4> scalar_router_scores{};
    std::array<uint32_t, 2> scalar_router_ids{};
    std::array<float, 2> scalar_router_weights{};
    for (uint32_t slot = 0; slot != scalar_router_ids.size(); ++slot) {
        uint32_t best = UINT32_MAX;
        for (uint32_t expert = 0; expert != scalar_router_scores.size(); ++expert) {
            bool already_selected = false;
            for (uint32_t earlier = 0; earlier != slot; ++earlier)
                already_selected |= scalar_router_ids[earlier] == expert;
            if (!already_selected && (best == UINT32_MAX ||
                                      scalar_router_scores[expert] > scalar_router_scores[best] ||
                                      (scalar_router_scores[expert] == scalar_router_scores[best] && expert < best)))
                best = expert;
        }
        scalar_router_ids[slot] = best;
    }
    const float scalar_normalizer = std::exp(scalar_router_scores[scalar_router_ids[0]]) +
                                    std::exp(scalar_router_scores[scalar_router_ids[1]]);
    for (uint32_t slot = 0; slot != scalar_router_weights.size(); ++slot)
        scalar_router_weights[slot] = std::exp(scalar_router_scores[scalar_router_ids[slot]]) /
                                     scalar_normalizer;
    const float scalar_gate = 0.0f;
    const float scalar_up = 0.0f;
    const float scalar_swiglu = scalar_gate / (1.0f + std::exp(-scalar_gate)) * scalar_up;
    const float scalar_weighted_expert = scalar_router_weights[0] * scalar_swiglu +
                                         scalar_router_weights[1] * scalar_swiglu;
    const std::array<float, 3> scalar_logits = {scalar_weighted_expert,
                                                 scalar_weighted_expert,
                                                 scalar_weighted_expert};
    CHECK(scalar_router_ids[0] == 0 && scalar_router_ids[1] == 1);
    CHECK(std::abs(scalar_router_weights[0] - 0.5f) <= 1e-7f);
    CHECK(std::abs(scalar_router_weights[1] - 0.5f) <= 1e-7f);
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    metal_dispatch_metrics_reset();
    const auto actual = program.decode(package->semantics().bos_id);
    const MetalDispatchMetrics dispatch = metal_dispatch_metrics();
    CHECK(std::holds_alternative<CanonicalMetalOutput>(actual));
    if (!std::holds_alternative<CanonicalMetalOutput>(actual)) return;
    const auto& output = std::get<CanonicalMetalOutput>(actual);
    CHECK(output.completed);
    CHECK(output.command_buffers == 1);
    CHECK(dispatch.command_buffers == 1);
    CHECK(output.operator_count != 0);
    CHECK(program.position() == 1);
    CHECK(output.logits.size() == package->semantics().vocabulary_size);
    for (size_t index = 0; index != output.logits.size() && index != scalar_logits.size(); ++index) {
        const float logit = output.logits[index];
        CHECK(std::isfinite(logit));
        CHECK(std::abs(logit - scalar_logits[index]) <= 1e-6f);
    }
    const CanonicalMetalResourceDiagnostics resources = program.resource_diagnostics();
    CHECK(resources.implicit_weight_copies == 0);
    std::fprintf(stderr,
                 "canonical MoE Metal: one_cb=%u requested_projection_source_bytes=%llu "
                 "registered_source_bytes=%llu implicit_copies=%llu "
                 "gpu_ms=%.6f "
                 "router_ids=[%u,%u] scalar_abs_error<=1e-6\n",
                 output.command_buffers,
                 static_cast<unsigned long long>(output.requested_projection_source_bytes),
                 static_cast<unsigned long long>(resources.registered_source_bytes),
                 static_cast<unsigned long long>(resources.implicit_weight_copies),
                 output.gpu_time_ms, scalar_router_ids[0], scalar_router_ids[1]);

    canonical_metal_fail_after_completed_submission_for_testing(program);
    const auto failed = program.decode(package->semantics().bos_id);
    CHECK(std::holds_alternative<CompatibilityReport>(failed));
    if (const auto* report = std::get_if<CompatibilityReport>(&failed))
        CHECK(report->code == CompatibilityError::RUNTIME_NUMERICAL_FAILURE);
    CHECK(program.position() == 1);
    CHECK(std::holds_alternative<CanonicalMetalOutput>(program.decode(package->semantics().bos_id)));
    CHECK(program.position() == 2);

    for (CanonicalMoeFixtureDefect defect : {CanonicalMoeFixtureDefect::Ambiguous,
                                             CanonicalMoeFixtureDefect::MissingEdge,
                                             CanonicalMoeFixtureDefect::SequentialResidual,
                                             CanonicalMoeFixtureDefect::KeyStateAlias,
                                             CanonicalMoeFixtureDefect::WrongActivation,
                                             CanonicalMoeFixtureDefect::WrongQ5Contract,
                                             CanonicalMoeFixtureDefect::WrongQ8Contract}) {
        const auto rejected_package = synthetic_q4k_moe_package(defect);
        CHECK(rejected_package != nullptr);
        if (!rejected_package) continue;
        const auto rejected = create_qualification_canonical_metal_program(rejected_package, 4);
        CHECK(std::holds_alternative<CompatibilityReport>(rejected));
        if (const auto* report = std::get_if<CompatibilityReport>(&rejected)) {
            const CompatibilityError expected =
                defect == CanonicalMoeFixtureDefect::Ambiguous ||
                        defect == CanonicalMoeFixtureDefect::MissingEdge
                    ? CompatibilityError::IR_REFERENCE_INVALID
                    : CompatibilityError::KERNEL_UNAVAILABLE;
            CHECK_MSG(report->code == expected,
                      "canonical MoE defect %u rejected with code=%u detail=%s",
                      static_cast<unsigned>(defect), static_cast<unsigned>(report->code),
                      report->detail.c_str());
        }
    }
}

void test_two_token_prefill_uses_registered_f16_batch() {
    CHECK(metal_device_present());
    CHECK(metal_available());
    const auto package = synthetic_f16_dense_package();
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 4, {}, {}, {}, {}, 2);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return;
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    const CanonicalMetalResourceDiagnostics before = program.resource_diagnostics();
    metal_dispatch_metrics_reset();
    const auto actual = program.prefill(std::vector<uint32_t>{0, 1});
    const auto expected = reference_fp32(*package, std::vector<uint32_t>{0, 1});
    const MetalDispatchMetrics dispatch = metal_dispatch_metrics();
    CHECK(std::holds_alternative<CanonicalMetalOutput>(actual));
    CHECK(std::holds_alternative<ReferenceOutput>(expected));
    if (!std::holds_alternative<CanonicalMetalOutput>(actual) ||
        !std::holds_alternative<ReferenceOutput>(expected)) return;
    const auto& output = std::get<CanonicalMetalOutput>(actual);
    const auto& reference = std::get<ReferenceOutput>(expected);
    const CanonicalMetalResourceDiagnostics after = program.resource_diagnostics();
    CHECK(output.completed);
    CHECK(dispatch.command_buffers == 1);
    CHECK(output.gpu_time_ms > 0.0);
    CHECK(output.batched_projection_dispatches == 7u * program.layer_count());
    CHECK(before.implicit_weight_copies == 0);
    CHECK(after.implicit_weight_copies == before.implicit_weight_copies);
    CHECK(after.implicit_weight_copies == 0);
    CHECK(program.position() == 2);
    CHECK(output.logits.size() == reference.logits.size());
    for (size_t index = 0; index != output.logits.size() && index != reference.logits.size(); ++index) {
        CHECK(std::isfinite(output.logits[index]));
        CHECK(std::abs(output.logits[index] - reference.logits[index]) <=
              1e-4f + 1e-4f * std::abs(reference.logits[index]));
    }
    CHECK(std::max_element(output.logits.begin(), output.logits.end()) - output.logits.begin() ==
          std::max_element(reference.logits.begin(), reference.logits.end()) - reference.logits.begin());

    auto stepped_created = create_qualification_canonical_metal_program(package, 4, {}, {}, {}, {}, 2);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(stepped_created));
    if (!std::holds_alternative<CanonicalMetalProgram>(stepped_created)) return;
    CanonicalMetalProgram stepped = std::get<CanonicalMetalProgram>(std::move(stepped_created));
    const auto first = stepped.prefill(std::vector<uint32_t>{0});
    const auto second = stepped.decode(1);
    CHECK(std::holds_alternative<CanonicalMetalOutput>(first));
    CHECK(std::holds_alternative<CanonicalMetalOutput>(second));
    if (!std::holds_alternative<CanonicalMetalOutput>(second)) return;
    const auto& stepped_output = std::get<CanonicalMetalOutput>(second);
    CHECK(stepped_output.completed);
    CHECK(stepped.position() == program.position());
    CHECK(stepped_output.logits.size() == output.logits.size());
    for (size_t index = 0; index != stepped_output.logits.size() && index != output.logits.size(); ++index) {
        CHECK(std::abs(stepped_output.logits[index] - output.logits[index]) <=
              1e-4f + 1e-4f * std::abs(output.logits[index]));
    }

    auto rejected_created = create_qualification_canonical_metal_program(package, 4, {}, {}, {}, {}, 2);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(rejected_created));
    if (!std::holds_alternative<CanonicalMetalProgram>(rejected_created)) return;
    CanonicalMetalProgram rejected = std::get<CanonicalMetalProgram>(std::move(rejected_created));
    const auto too_wide = rejected.prefill(std::vector<uint32_t>{0, 1, 2});
    CHECK(std::holds_alternative<CompatibilityReport>(too_wide));
    if (const auto* report = std::get_if<CompatibilityReport>(&too_wide))
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
    CHECK(rejected.position() == 0);

    auto sequential_created = create_qualification_canonical_metal_program(package, 4);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(sequential_created));
    if (!std::holds_alternative<CanonicalMetalProgram>(sequential_created)) return;
    CanonicalMetalProgram sequential =
        std::get<CanonicalMetalProgram>(std::move(sequential_created));
    const std::vector<uint32_t> sequential_tokens = {0, 1, 2};
    const auto sequential_actual = sequential.prefill(sequential_tokens);
    const auto sequential_expected = reference_fp32(*package, sequential_tokens);
    CHECK(std::holds_alternative<CanonicalMetalOutput>(sequential_actual));
    CHECK(std::holds_alternative<ReferenceOutput>(sequential_expected));
    if (!std::holds_alternative<CanonicalMetalOutput>(sequential_actual) ||
        !std::holds_alternative<ReferenceOutput>(sequential_expected)) return;
    const auto& sequential_output = std::get<CanonicalMetalOutput>(sequential_actual);
    const auto& sequential_reference = std::get<ReferenceOutput>(sequential_expected);
    CHECK(sequential.position() == sequential_tokens.size());
    CHECK(sequential_output.batched_projection_dispatches == 0);
    CHECK(sequential_output.logits.size() == sequential_reference.logits.size());
    for (size_t index = 0; index != sequential_output.logits.size() &&
                           index != sequential_reference.logits.size(); ++index) {
        CHECK(std::isfinite(sequential_output.logits[index]));
        CHECK(std::abs(sequential_output.logits[index] - sequential_reference.logits[index]) <=
              1e-4f + 1e-4f * std::abs(sequential_reference.logits[index]));
    }
}

void test_canonical_session_executes_mixed_attention_geometry() {
    CHECK(metal_device_present());
    CHECK(metal_available());
    if (!metal_device_present() || !metal_available()) return;
    const auto package = synthetic_mixed_attention_geometry_package();
    CHECK(package != nullptr);
    if (!package) return;
    for (const SemanticLayer& layer : package->semantics().layers) {
        DenseGraphWitness witness;
        const bool matched = match_canonical_dense_operator_edges(
            package->semantics(), layer, witness);
        CHECK_MSG(matched, "mixed attention layer %u graph matcher rejected",
                  layer.layer_index);
    }
    auto created = create_qualification_canonical_metal_program(package, 4);
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) {
        const auto& report = std::get<CompatibilityReport>(created);
        std::fprintf(stderr,
                     "mixed attention geometry rejected: code=%u layer=%u operator=%u detail=%s\n",
                     static_cast<unsigned>(report.code), report.layer, report.operator_id,
                     report.detail.c_str());
    }
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return;
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    CHECK(program.layer_count() == 2);
    metal_dispatch_metrics_reset();
    const std::vector<uint32_t> tokens = {package->semantics().bos_id, 1};
    const auto actual = program.prefill(tokens);
    const auto expected = reference_fp32(*package, tokens);
    CHECK(std::holds_alternative<CanonicalMetalOutput>(actual));
    CHECK(std::holds_alternative<ReferenceOutput>(expected));
    if (!std::holds_alternative<CanonicalMetalOutput>(actual) ||
        !std::holds_alternative<ReferenceOutput>(expected)) return;
    const auto& output = std::get<CanonicalMetalOutput>(actual);
    const auto& reference = std::get<ReferenceOutput>(expected);
    CHECK(output.completed);
    CHECK(output.command_buffers == 1);
    CHECK(output.kv_cache_bytes == 3072);
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    CHECK(output.logits.size() == reference.logits.size());
    for (size_t index = 0; index != output.logits.size() && index != reference.logits.size(); ++index) {
        CHECK(std::isfinite(output.logits[index]));
        CHECK(std::abs(output.logits[index] - reference.logits[index]) <=
              1e-4f + 1e-4f * std::abs(reference.logits[index]));
    }
    CHECK(program.resource_diagnostics().implicit_weight_copies == 0);

    auto batch_created = create_qualification_canonical_metal_program(
        package, 4, {}, {}, {}, {}, 2);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(batch_created));
    if (!std::holds_alternative<CanonicalMetalProgram>(batch_created)) return;
    CanonicalMetalProgram batch = std::get<CanonicalMetalProgram>(std::move(batch_created));
    metal_dispatch_metrics_reset();
    const auto batch_actual = batch.prefill(tokens);
    CHECK(std::holds_alternative<CanonicalMetalOutput>(batch_actual));
    if (!std::holds_alternative<CanonicalMetalOutput>(batch_actual)) return;
    const auto& batch_output = std::get<CanonicalMetalOutput>(batch_actual);
    CHECK(batch_output.completed);
    CHECK(batch_output.command_buffers == 1);
    CHECK(batch_output.kv_cache_bytes == 3072);
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    CHECK(batch_output.batched_projection_dispatches == 14);
    CHECK(batch_output.logits.size() == output.logits.size());
    for (size_t index = 0; index != batch_output.logits.size() && index != output.logits.size(); ++index) {
        CHECK(std::abs(batch_output.logits[index] - output.logits[index]) <=
              1e-4f + 1e-4f * std::abs(output.logits[index]));
    }
    CHECK(batch.resource_diagnostics().implicit_weight_copies == 0);
    std::fprintf(stderr,
                 "canonical mixed attention geometry: layers=2 sequential_tokens=2 "
                 "batch_rows=2 kv_cache_bytes=3072 one_cb=1 implicit_copies=0\n");
}

void test_declared_attention_scale_controls_canonical_prefill_and_decode() {
    CHECK(metal_device_present());
    CHECK(metal_available());
    if (!metal_device_present() || !metal_available()) return;

    struct ScaleOutputs {
        std::vector<float> prefill;
        std::vector<float> decode;
    };
    const auto run = [&](uint32_t scale_bits, ScaleOutputs& outputs) {
        const auto package = synthetic_f16_dense_package(scale_bits, true);
        CHECK(package != nullptr);
        if (!package) return;
        const std::vector<uint32_t> tokens = {0, 1};
        const auto expected = reference_fp32(*package, tokens);
        CHECK(std::holds_alternative<ReferenceOutput>(expected));
        if (!std::holds_alternative<ReferenceOutput>(expected)) return;
        const auto& reference = std::get<ReferenceOutput>(expected);

        auto prefill_created = create_qualification_canonical_metal_program(package, 4);
        CHECK(std::holds_alternative<CanonicalMetalProgram>(prefill_created));
        if (!std::holds_alternative<CanonicalMetalProgram>(prefill_created)) return;
        CanonicalMetalProgram prefill =
            std::get<CanonicalMetalProgram>(std::move(prefill_created));
        auto prefill_result = prefill.prefill(tokens);
        CHECK(std::holds_alternative<CanonicalMetalOutput>(prefill_result));
        if (!std::holds_alternative<CanonicalMetalOutput>(prefill_result)) return;
        outputs.prefill = std::get<CanonicalMetalOutput>(std::move(prefill_result)).logits;

        auto decode_created = create_qualification_canonical_metal_program(package, 4);
        CHECK(std::holds_alternative<CanonicalMetalProgram>(decode_created));
        if (!std::holds_alternative<CanonicalMetalProgram>(decode_created)) return;
        CanonicalMetalProgram decode =
            std::get<CanonicalMetalProgram>(std::move(decode_created));
        CHECK(std::holds_alternative<CanonicalMetalOutput>(decode.prefill(std::vector<uint32_t>{0})));
        auto decode_result = decode.decode(1);
        CHECK(std::holds_alternative<CanonicalMetalOutput>(decode_result));
        if (!std::holds_alternative<CanonicalMetalOutput>(decode_result)) return;
        outputs.decode = std::get<CanonicalMetalOutput>(std::move(decode_result)).logits;

        CHECK(outputs.prefill.size() == reference.logits.size());
        CHECK(outputs.decode.size() == reference.logits.size());
        for (size_t index = 0; index != reference.logits.size() &&
                               index != outputs.prefill.size() &&
                               index != outputs.decode.size(); ++index) {
            const float tolerance = 1e-4f + 1e-4f * std::abs(reference.logits[index]);
            CHECK(std::abs(outputs.prefill[index] - reference.logits[index]) <= tolerance);
            CHECK(std::abs(outputs.decode[index] - reference.logits[index]) <= tolerance);
        }
    };

    ScaleOutputs low, high;
    run(0x3c23d70au, low);   // 0.01, deliberately not 1/sqrt(32)
    run(0x3e4ccccdu, high);  // 0.2
    CHECK(low.prefill.size() == high.prefill.size());
    CHECK(low.decode.size() == high.decode.size());
    float prefill_delta = 0.0f, decode_delta = 0.0f;
    for (size_t index = 0; index != low.prefill.size() && index != high.prefill.size(); ++index)
        prefill_delta = std::max(prefill_delta, std::abs(low.prefill[index] - high.prefill[index]));
    for (size_t index = 0; index != low.decode.size() && index != high.decode.size(); ++index)
        decode_delta = std::max(decode_delta, std::abs(low.decode[index] - high.decode[index]));
    CHECK(prefill_delta > 1e-6f);
    CHECK(decode_delta > 1e-6f);
}

void test_declared_embedding_scale_controls_canonical_decode() {
    CHECK(metal_device_present());
    CHECK(metal_available());
    if (!metal_device_present() || !metal_available()) return;

    const auto run = [](uint32_t scale_bits, std::vector<float>& logits) {
        const auto package = synthetic_f16_dense_package(
            0x3e800000u, true, false, scale_bits);
        CHECK(package != nullptr);
        if (!package) return;
        const std::vector<uint32_t> tokens = {0, 1};
        const auto expected = reference_fp32(*package, tokens);
        CHECK(std::holds_alternative<ReferenceOutput>(expected));
        if (!std::holds_alternative<ReferenceOutput>(expected)) return;
        auto created = create_qualification_canonical_metal_program(package, 4);
        CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
        if (!std::holds_alternative<CanonicalMetalProgram>(created)) return;
        CanonicalMetalProgram program =
            std::get<CanonicalMetalProgram>(std::move(created));
        auto actual = program.prefill(tokens);
        CHECK(std::holds_alternative<CanonicalMetalOutput>(actual));
        if (!std::holds_alternative<CanonicalMetalOutput>(actual)) return;
        logits = std::get<CanonicalMetalOutput>(actual).logits;
        const auto& reference = std::get<ReferenceOutput>(expected).logits;
        CHECK(logits.size() == reference.size());
        for (size_t index = 0; index != logits.size() && index != reference.size(); ++index) {
            CHECK(std::abs(logits[index] - reference[index]) <=
                  1e-4f + 1e-4f * std::abs(reference[index]));
        }
    };

    std::vector<float> unit, doubled;
    run(0x3f800000u, unit);
    run(0x40000000u, doubled);
    CHECK(unit.size() == doubled.size());
    float delta = 0.0f;
    for (size_t index = 0; index != unit.size() && index != doubled.size(); ++index)
        delta = std::max(delta, std::abs(unit[index] - doubled[index]));
    CHECK(delta > 1e-6f);
}

void test_failed_two_token_prefill_does_not_publish_position_or_kv() {
    CHECK(metal_device_present());
    CHECK(metal_available());
    const auto package = synthetic_f16_dense_package();
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 4, {}, {}, {}, {}, 2);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return;
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));

    canonical_metal_fail_after_completed_submission_for_testing(program);
    const auto failed = program.prefill(std::vector<uint32_t>{0, 1});
    CHECK(std::holds_alternative<CompatibilityReport>(failed));
    if (const auto* report = std::get_if<CompatibilityReport>(&failed)) {
        CHECK(report->code == CompatibilityError::RUNTIME_NUMERICAL_FAILURE);
        CHECK(report->detail.find("injected post-completion failure") != std::string::npos);
    }
    CHECK(program.position() == 0);

    const auto retried = program.prefill(std::vector<uint32_t>{0, 1});
    CHECK(std::holds_alternative<CanonicalMetalOutput>(retried));
    CHECK(program.position() == 2);
    const auto decoded = program.decode(2);
    const auto expected = reference_fp32(*package, std::vector<uint32_t>{0, 1, 2});
    CHECK(std::holds_alternative<CanonicalMetalOutput>(decoded));
    CHECK(std::holds_alternative<ReferenceOutput>(expected));
    if (!std::holds_alternative<CanonicalMetalOutput>(decoded) ||
        !std::holds_alternative<ReferenceOutput>(expected)) return;
    const auto& output = std::get<CanonicalMetalOutput>(decoded);
    const auto& reference = std::get<ReferenceOutput>(expected);
    CHECK(output.logits.size() == reference.logits.size());
    for (size_t index = 0; index != output.logits.size() && index != reference.logits.size(); ++index) {
        CHECK(std::isfinite(output.logits[index]));
        CHECK(std::abs(output.logits[index] - reference.logits[index]) <=
              1e-4f + 1e-4f * std::abs(reference.logits[index]));
    }
}

void test_dense_linear_shape_cannot_read_an_adjacent_tensor_span() {
    CHECK(metal_device_present());
    CHECK(metal_available());
    if (!metal_device_present() || !metal_available()) return;
    const auto package = synthetic_f16_dense_package(0x3e800000u, false, true);
    CHECK(package != nullptr);
    if (!package) return;
    const auto created = create_qualification_canonical_metal_program(package, 4);
    CHECK(std::holds_alternative<CompatibilityReport>(created));
    if (const auto* report = std::get_if<CompatibilityReport>(&created)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE ||
              report->code == CompatibilityError::IR_REFERENCE_INVALID ||
              report->code == CompatibilityError::SESSION_CONSTRUCTION_FAILED);
    }
}

std::shared_ptr<const RuntimePackage> load_package() {
    auto artifacts = ArtifactSet::load_single_file(LAPLACE_QUALIFICATION_GGUF);
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return {};
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    if (!std::holds_alternative<PackageView>(view)) return {};
    auto loaded = load_expected_fixture_gguf(std::get<PackageView>(view), bundled_compatibility_rules());
    if (!std::holds_alternative<ValidatedPackage>(loaded)) return {};
    return std::get<ValidatedPackage>(loaded).runtime_package();
}

std::shared_ptr<const RuntimePackage> load_package_with_semantic_mutation(
    const std::function<bool(SemanticModel&)>& mutate) {
    auto artifacts = ArtifactSet::load_single_file(LAPLACE_QUALIFICATION_GGUF);
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return {};
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    if (!std::holds_alternative<PackageView>(view)) return {};
    auto loaded = load_expected_fixture_gguf(std::get<PackageView>(view), bundled_compatibility_rules());
    if (!std::holds_alternative<ValidatedPackage>(loaded)) return {};
    const auto original = std::get<ValidatedPackage>(loaded).runtime_package();
    SemanticModel semantics = original->semantics();
    if (!mutate(semantics)) return {};
    return RuntimePackage::make_legacy_test_only(std::move(semantics), std::get<PackageView>(view),
                                                 original->fingerprint(), original->rule_fingerprint(),
                                                 original->rule_revision(), original->qualification_state());
}

std::shared_ptr<const RuntimePackage> load_package_with_final_epsilon(float epsilon) {
    uint32_t bits = 0;
    std::memcpy(&bits, &epsilon, sizeof(bits));
    return load_package_with_semantic_mutation([&](SemanticModel& semantics) {
        for (SemanticOperator& op : semantics.operators) {
            bool final_norm = false;
            for (uint32_t tensor_id : op.tensors) {
                final_norm |= tensor_id < semantics.tensors.size() &&
                              semantics.tensors[tensor_id].role == TensorRole::FinalNormWeight;
            }
            if (!final_norm) continue;
            auto* payload = std::get_if<RmsNormPayload>(&op.payload);
            if (!payload) return false;
            payload->epsilon_f32_bits = bits;
            return true;
        }
        return false;
    });
}

bool add_qk_norm_to_first_layer(SemanticModel& semantics) {
    if (semantics.layers.empty()) return false;
    semantics.schema_major = 8;
    semantics.schema_minor = 0;
    semantics.opset_major = 8;
    semantics.opset_minor = 0;
    for (SemanticOperator& op : semantics.operators) {
        op.semantic_version = 8;
        if (op.kind != OperatorKind::CausalAttention) continue;
        auto* attention = std::get_if<CausalAttentionPayload>(&op.payload);
        if (!attention || op.inputs.size() != 3) return false;
        attention->value_source = ValueSource::SeparateProjection;
        attention->value_source_value = op.inputs[2];
    }
    for (SemanticState& state : semantics.states) state.semantic_version = 8;
    const SemanticLayer first_layer = semantics.layers.front();
    if (first_layer.first_operator > semantics.operators.size() ||
        first_layer.operator_count > semantics.operators.size() - first_layer.first_operator) return false;
    uint32_t query_value = UINT32_MAX;
    uint32_t key_value = UINT32_MAX;
    uint32_t attention_norm_tensor = UINT32_MAX;
    uint32_t rope_index = UINT32_MAX;
    uint32_t head_dimension = 0;
    RmsNormPayload payload;
    bool have_payload = false;
    const auto has_role = [&](const SemanticOperator& op, TensorRole role) {
        for (uint32_t tensor_id : op.tensors) {
            if (tensor_id < semantics.tensors.size() && semantics.tensors[tensor_id].role == role) return true;
        }
        return false;
    };
    for (uint32_t index = 0; index != first_layer.operator_count; ++index) {
        const uint32_t operator_index = first_layer.first_operator + index;
        const SemanticOperator& op = semantics.operators[operator_index];
        if (op.kind == OperatorKind::Linear && has_role(op, TensorRole::QueryWeight) && op.outputs.size() == 1)
            query_value = op.outputs[0];
        if (op.kind == OperatorKind::Linear && has_role(op, TensorRole::KeyWeight) && op.outputs.size() == 1)
            key_value = op.outputs[0];
        if (op.kind == OperatorKind::RmsNorm && has_role(op, TensorRole::AttentionNormWeight) && op.tensors.size() == 1) {
            attention_norm_tensor = op.tensors[0];
            const auto* norm_payload = std::get_if<RmsNormPayload>(&op.payload);
            if (!norm_payload) return false;
            payload = *norm_payload;
            have_payload = true;
        }
        if (op.kind == OperatorKind::Rope) rope_index = operator_index;
        if (op.kind == OperatorKind::CausalAttention) {
            const auto* attention = std::get_if<CausalAttentionPayload>(&op.payload);
            if (!attention || attention->head_dimension == 0) return false;
            head_dimension = attention->head_dimension;
        }
    }
    if (!have_payload || query_value >= semantics.values.size() || key_value >= semantics.values.size() ||
        attention_norm_tensor >= semantics.tensors.size() || rope_index == UINT32_MAX || head_dimension == 0) return false;
    const SemanticValue& key_value_descriptor = semantics.values[key_value];
    if (key_value_descriptor.dimensions.empty() ||
        key_value_descriptor.dimensions.back().kind != DimensionKind::Constant ||
        key_value_descriptor.dimensions.back().constant_or_symbol == 0 ||
        key_value_descriptor.dimensions.back().constant_or_symbol > UINT32_MAX) return false;
    const uint32_t key_width = static_cast<uint32_t>(key_value_descriptor.dimensions.back().constant_or_symbol);
    SemanticTensor query_tensor = semantics.tensors[attention_norm_tensor];
    query_tensor.id = static_cast<uint32_t>(semantics.tensors.size());
    query_tensor.role = TensorRole::AttentionQueryNormWeight;
    if (query_tensor.planes.size() != 1 || query_tensor.planes[0].storage_type != ScalarType::F32 ||
        head_dimension > UINT32_MAX / sizeof(float)) return false;
    query_tensor.dimensions = {{DimensionKind::Constant, head_dimension}};
    query_tensor.layout.rank = 1;
    query_tensor.layout.strides[0] = 1;
    query_tensor.planes[0].length = static_cast<uint64_t>(head_dimension) * sizeof(float);
    semantics.tensors.push_back(query_tensor);
    SemanticTensor key_tensor = semantics.tensors[attention_norm_tensor];
    key_tensor.id = static_cast<uint32_t>(semantics.tensors.size());
    key_tensor.role = TensorRole::AttentionKeyNormWeight;
    if (key_tensor.planes.size() != 1 || key_tensor.planes[0].storage_type != ScalarType::F32 ||
        head_dimension > UINT32_MAX / sizeof(float) || key_width % head_dimension != 0) return false;
    key_tensor.dimensions = {{DimensionKind::Constant, head_dimension}};
    key_tensor.layout.rank = 1;
    key_tensor.layout.strides[0] = 1;
    key_tensor.planes[0].length = static_cast<uint64_t>(head_dimension) * sizeof(float);
    semantics.tensors.push_back(key_tensor);
    SemanticValue query_output = semantics.values[query_value];
    query_output.id = static_cast<uint32_t>(semantics.values.size());
    semantics.values.push_back(query_output);
    SemanticValue key_output = semantics.values[key_value];
    key_output.id = static_cast<uint32_t>(semantics.values.size());
    semantics.values.push_back(key_output);
    SemanticOperator query_norm;
    query_norm.id = static_cast<uint32_t>(semantics.operators.size());
    query_norm.kind = OperatorKind::RmsNorm;
    query_norm.semantic_version = 8;
    query_norm.inputs = {query_value};
    query_norm.outputs = {query_output.id};
    query_norm.tensors = {query_tensor.id};
    payload.affine_geometry = RmsNormAffineGeometry::SharedAcrossGroups;
    payload.reduction_extent = head_dimension;
    query_norm.payload = payload;
    SemanticOperator key_norm = query_norm;
    key_norm.id += 1;
    key_norm.inputs = {key_value};
    key_norm.outputs = {key_output.id};
    key_norm.tensors = {key_tensor.id};
    semantics.operators[rope_index].inputs = {query_output.id, key_output.id};
    semantics.operators.insert(semantics.operators.begin() + rope_index, {query_norm, key_norm});
    for (size_t index = 0; index != semantics.layers.size(); ++index) {
        if (index == 0) semantics.layers[index].operator_count += 2;
        else semantics.layers[index].first_operator += 2;
    }
    return true;
}

bool add_query_gate_to_first_layer(SemanticModel& semantics) {
    if (semantics.layers.empty()) return false;
    semantics.schema_major = 4;
    semantics.opset_major = 4;
    for (SemanticOperator& op : semantics.operators) op.semantic_version = 4;
    for (SemanticState& state : semantics.states) state.semantic_version = 4;
    const SemanticLayer first = semantics.layers.front();
    if (first.first_operator > semantics.operators.size() ||
        first.operator_count > semantics.operators.size() - first.first_operator) return false;
    const auto has_role = [&](const SemanticOperator& op, TensorRole role) {
        return std::any_of(op.tensors.begin(), op.tensors.end(), [&](uint32_t tensor_id) {
            return tensor_id < semantics.tensors.size() && semantics.tensors[tensor_id].role == role;
        });
    };
    const auto find = [&](OperatorKind kind, TensorRole role) {
        for (uint32_t offset = 0; offset != first.operator_count; ++offset) {
            const uint32_t index = first.first_operator + offset;
            if (semantics.operators[index].kind == kind && has_role(semantics.operators[index], role)) return index;
        }
        return UINT32_MAX;
    };
    const uint32_t query_index = find(OperatorKind::Linear, TensorRole::QueryWeight);
    const uint32_t key_index = find(OperatorKind::Linear, TensorRole::KeyWeight);
    const uint32_t value_index = find(OperatorKind::Linear, TensorRole::ValueWeight);
    const uint32_t output_index = find(OperatorKind::Linear, TensorRole::AttentionOutputWeight);
    uint32_t rope_index = UINT32_MAX;
    uint32_t attention_index = UINT32_MAX;
    for (uint32_t offset = 0; offset != first.operator_count; ++offset) {
        const uint32_t index = first.first_operator + offset;
        if (semantics.operators[index].kind == OperatorKind::Rope) rope_index = rope_index == UINT32_MAX ? index : UINT32_MAX - 1;
        if (semantics.operators[index].kind == OperatorKind::CausalAttention) attention_index = attention_index == UINT32_MAX ? index : UINT32_MAX - 1;
    }
    if (query_index == UINT32_MAX || key_index == UINT32_MAX || value_index == UINT32_MAX || output_index == UINT32_MAX ||
        rope_index >= semantics.operators.size() || attention_index >= semantics.operators.size()) return false;
    SemanticOperator& query = semantics.operators[query_index];
    SemanticOperator& key = semantics.operators[key_index];
    SemanticOperator& value = semantics.operators[value_index];
    SemanticOperator& output = semantics.operators[output_index];
    SemanticOperator& rope = semantics.operators[rope_index];
    SemanticOperator& attention = semantics.operators[attention_index];
    const auto query_weight = std::find_if(query.tensors.begin(), query.tensors.end(), [&](uint32_t tensor_id) {
        return tensor_id < semantics.tensors.size() && semantics.tensors[tensor_id].role == TensorRole::QueryWeight;
    });
    if (query_weight == query.tensors.end() || query.outputs.size() != 1 || key.outputs.size() != 1 || value.outputs.size() != 1 ||
        output.inputs.size() != 1 || rope.inputs.size() != 2 || rope.outputs.size() != 2 ||
        attention.inputs.size() != 3 || attention.outputs.size() != 1) return false;
    const auto* original_attention = std::get_if<CausalAttentionPayload>(&attention.payload);
    if (!original_attention || original_attention->query_heads < 2 || original_attention->query_heads % 2 != 0 ||
        original_attention->head_dimension == 0) return false;
    const uint32_t query_heads = original_attention->query_heads / 2;
    const uint32_t head_dimension = original_attention->head_dimension;
    if (query_heads > UINT32_MAX / head_dimension) return false;
    const uint32_t query_width = query_heads * head_dimension;
    const uint32_t query_weight_id = *query_weight;
    query.tensors.erase(std::remove_if(query.tensors.begin(), query.tensors.end(), [&](uint32_t tensor_id) {
        return tensor_id < semantics.tensors.size() && semantics.tensors[tensor_id].role == TensorRole::QueryBias;
    }), query.tensors.end());
    semantics.tensors[query_weight_id].role = TensorRole::AttentionQueryGateWeight;

    const auto append_value = [&](uint32_t width, uint32_t heads = 0) {
        SemanticValue value_descriptor;
        value_descriptor.id = static_cast<uint32_t>(semantics.values.size());
        value_descriptor.logical_type = ScalarType::F32;
        value_descriptor.dimensions = heads == 0
            ? std::vector<Dimension>{{DimensionKind::Symbol, 1}, {DimensionKind::Constant, width}}
            : std::vector<Dimension>{{DimensionKind::Symbol, 1}, {DimensionKind::Constant, heads},
                                     {DimensionKind::Constant, head_dimension}};
        semantics.values.push_back(std::move(value_descriptor));
        return semantics.values.back().id;
    };
    const uint32_t fused = append_value(query_width * 2);
    const uint32_t split_query = append_value(query_width, query_heads);
    const uint32_t split_gate = append_value(query_width, query_heads);
    const uint32_t rotated_query = append_value(query_width, query_heads);
    const uint32_t rotated_key = append_value(head_dimension, 1);
    const uint32_t attention_context = append_value(query_width, query_heads);
    const uint32_t gated_context = append_value(query_width, query_heads);

    query.outputs = {fused};
    rope.inputs = {split_query, key.outputs[0]};
    rope.outputs = {rotated_query, rotated_key};
    rope.payload = RopePayload{RopePairing::Interleaved, true, head_dimension, 0x49742400u, 0x3f800000u};
    attention.inputs = {rotated_query, rotated_key, value.outputs[0]};
    attention.outputs = {attention_context};
    attention.payload = CausalAttentionPayload{query_heads, 1, head_dimension, original_attention->scale_f32_bits,
                                               AttentionMask::Causal, CachePolicy::Global};
    output.inputs = {gated_context};
    for (uint32_t state_id : attention.states) {
        if (state_id >= semantics.states.size() || semantics.states[state_id].dimensions.size() != 3) return false;
        semantics.states[state_id].dimensions[1] = {DimensionKind::Constant, 1};
    }

    SemanticOperator split;
    split.kind = OperatorKind::AxisSplit;
    split.semantic_version = 4;
    split.inputs = {fused};
    split.outputs = {split_query, split_gate};
    split.payload = AxisSplitPayload{query_width, query_width};
    semantics.operators.insert(semantics.operators.begin() + rope_index, std::move(split));
    ++attention_index;
    SemanticOperator gated;
    gated.kind = OperatorKind::GatedAttention;
    gated.semantic_version = 4;
    gated.inputs = {attention_context, split_gate};
    gated.outputs = {gated_context};
    gated.payload = GatedAttentionPayload{};
    semantics.operators.insert(semantics.operators.begin() + attention_index + 1, std::move(gated));
    for (size_t index = 0; index != semantics.layers.size(); ++index) {
        if (index == 0) semantics.layers[index].operator_count += 2;
        else semantics.layers[index].first_operator += 2;
    }
    for (uint32_t id = 0; id != semantics.operators.size(); ++id) semantics.operators[id].id = id;
    return true;
}

bool replace_first_dense_layer_with_recurrent(SemanticModel& semantics) {
    if (semantics.layers.empty() || semantics.tensors.empty() || semantics.operators.empty()) return false;
    SemanticLayer& first_layer = semantics.layers.front();
    if (first_layer.first_operator > semantics.operators.size() ||
        first_layer.operator_count > semantics.operators.size() - first_layer.first_operator) return false;
    const uint32_t first_end = first_layer.first_operator + first_layer.operator_count;
    const SemanticOperator* embedding = &semantics.operators.front();
    const auto* embedding_payload = std::get_if<EmbeddingLookupPayload>(&embedding->payload);
    if (!embedding_payload || embedding_payload->width == 0 || embedding_payload->width > INT_MAX) return false;
    const uint32_t hidden = embedding_payload->width;
    constexpr uint32_t qk_heads = 4;
    constexpr uint32_t value_heads = 4;
    constexpr uint32_t head_dimension = 128;
    constexpr uint32_t kernel = 2;
    constexpr uint32_t ffn_intermediate = 256;
    constexpr uint32_t channels = head_dimension * (2 * qk_heads + value_heads);
    if (hidden > UINT32_MAX / channels) return false;
    const ArtifactId artifact = semantics.tensors.front().planes.empty() ? ArtifactId{UINT32_MAX}
                                                                           : semantics.tensors.front().planes.front().artifact_id;
    if (artifact.value == UINT32_MAX) return false;

    semantics.schema_major = 4;
    semantics.schema_minor = 0;
    semantics.opset_major = 4;
    semantics.opset_minor = 0;
    for (SemanticOperator& op : semantics.operators) op.semantic_version = 4;
    for (SemanticState& state : semantics.states) state.semantic_version = 4;

    const auto append_f32 = [&](TensorRole role, std::vector<Dimension> dimensions) {
        uint64_t elements = 1;
        for (const Dimension& dimension : dimensions) elements *= dimension.constant_or_symbol;
        SemanticTensor tensor;
        tensor.id = static_cast<uint32_t>(semantics.tensors.size());
        tensor.role = role;
        tensor.logical_type = ScalarType::F32;
        tensor.dimensions = std::move(dimensions);
        tensor.layout.rank = static_cast<uint8_t>(tensor.dimensions.size());
        tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        tensor.layout.strides[0] = 1;
        if (tensor.dimensions.size() == 2) tensor.layout.strides[1] = tensor.dimensions[0].constant_or_symbol;
        tensor.planes = {{PlaneKind::Values, ScalarType::F32, artifact, 0, elements * sizeof(float), 64, 0}};
        semantics.tensors.push_back(std::move(tensor));
        return static_cast<uint32_t>(semantics.tensors.size() - 1);
    };
    const auto append_blocked = [&](TensorRole role, uint32_t rows, uint32_t columns, uint32_t block_bytes) {
        const uint64_t elements = static_cast<uint64_t>(rows) * columns;
        if (elements % 256 != 0) return UINT32_MAX;
        SemanticTensor tensor;
        tensor.id = static_cast<uint32_t>(semantics.tensors.size());
        tensor.role = role;
        tensor.logical_type = ScalarType::F32;
        tensor.dimensions = {{DimensionKind::Constant, rows}, {DimensionKind::Constant, columns}};
        tensor.layout.kind = PhysicalLayoutKind::GgufBlocked;
        tensor.layout.packing = PackingKind::Gguf;
        tensor.layout.rank = 2;
        tensor.layout.block_rank = 1;
        tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        tensor.layout.strides[0] = 1;
        tensor.layout.strides[1] = rows;
        tensor.layout.block_elements = 256;
        tensor.layout.block_bytes = block_bytes;
        tensor.quantization.kind = QuantizationKind::BlockedAffine;
        tensor.quantization.accumulation_type = ScalarType::F32;
        tensor.quantization.scale_type = ScalarType::F16;
        tensor.quantization.zero_type = ScalarType::F16;
        tensor.quantization.block_elements = 256;
        tensor.quantization.block_bytes = block_bytes;
        tensor.quantization.group_size = 256;
        tensor.quantization.required_plane_mask = 1;
        tensor.planes = {{PlaneKind::Values, ScalarType::U8, artifact, 0, elements / 256 * block_bytes, 32, 0}};
        semantics.tensors.push_back(std::move(tensor));
        return static_cast<uint32_t>(semantics.tensors.size() - 1);
    };
    const uint32_t input_norm = append_f32(TensorRole::AttentionNormWeight, {{DimensionKind::Constant, hidden}});
    const uint32_t qkv = append_blocked(TensorRole::RecurrentQkvWeight, hidden, channels, 144);
    const uint32_t gate = append_blocked(TensorRole::RecurrentGateWeight, hidden, head_dimension, 144);
    const uint32_t beta = append_blocked(TensorRole::RecurrentBetaWeight, hidden, value_heads, 210);
    const uint32_t alpha = append_blocked(TensorRole::RecurrentAlphaWeight, hidden, value_heads, 144);
    const uint32_t conv = append_f32(TensorRole::RecurrentConvWeight,
                                     {{DimensionKind::Constant, kernel}, {DimensionKind::Constant, channels}});
    const uint32_t dt_bias = append_f32(TensorRole::RecurrentDtBias, {{DimensionKind::Constant, value_heads}});
    const uint32_t decay = append_f32(TensorRole::RecurrentDecayWeight, {{DimensionKind::Constant, value_heads}});
    const uint32_t recurrent_norm = append_f32(TensorRole::RecurrentNormWeight,
                                                {{DimensionKind::Constant, head_dimension}});
    const uint32_t output = append_blocked(TensorRole::RecurrentOutputWeight, head_dimension, hidden, 210);
    const uint32_t ffn_norm = append_f32(TensorRole::FfnNormWeight, {{DimensionKind::Constant, hidden}});
    const uint32_t ffn_gate = append_blocked(TensorRole::FfnGateWeight, hidden, ffn_intermediate, 144);
    const uint32_t ffn_up = append_blocked(TensorRole::FfnUpWeight, hidden, ffn_intermediate, 210);
    const uint32_t ffn_down = append_blocked(TensorRole::FfnDownWeight, ffn_intermediate, hidden, 144);
    if (qkv == UINT32_MAX || gate == UINT32_MAX || beta == UINT32_MAX || alpha == UINT32_MAX ||
        output == UINT32_MAX || ffn_gate == UINT32_MAX || ffn_up == UINT32_MAX || ffn_down == UINT32_MAX) return false;

    const auto append_value = [&](uint32_t width) {
        const uint32_t id = static_cast<uint32_t>(semantics.values.size());
        semantics.values.push_back({id, ScalarType::F32,
                                    {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, width}}, 0});
        return id;
    };
    const uint32_t input = semantics.operators[first_layer.first_operator].inputs.empty()
                               ? UINT32_MAX : semantics.operators[first_layer.first_operator].inputs.front();
    if (input == UINT32_MAX) return false;
    const uint32_t normalized = append_value(hidden);
    const uint32_t qkv_value = append_value(channels);
    const uint32_t gate_value = append_value(head_dimension);
    const uint32_t beta_value = append_value(value_heads);
    const uint32_t alpha_value = append_value(value_heads);
    const uint32_t conv_query = append_value(head_dimension);
    const uint32_t conv_key = append_value(head_dimension);
    const uint32_t conv_value = append_value(head_dimension);
    const uint32_t normalized_query = append_value(head_dimension);
    const uint32_t normalized_key = append_value(head_dimension);
    const uint32_t delta_value = append_value(head_dimension);
    const uint32_t gated_value = append_value(head_dimension);
    const uint32_t recurrent_output = append_value(hidden);
    const uint32_t residual = append_value(hidden);
    const uint32_t ffn_normalized = append_value(hidden);
    const uint32_t ffn_gate_value = append_value(ffn_intermediate);
    const uint32_t ffn_up_value = append_value(ffn_intermediate);
    const uint32_t swiglu_value = append_value(ffn_intermediate);
    const uint32_t ffn_down_value = append_value(hidden);
    const uint32_t final_residual = append_value(hidden);

    StateFormat recurrent_format;
    recurrent_format.kind = StateFormatKind::RecurrentContiguous;
    recurrent_format.logical_type = ScalarType::F32;
    recurrent_format.encoded_type = ScalarType::F32;
    recurrent_format.logical_domain = TransformDomain::Untransformed;
    recurrent_format.encoded_domain = TransformDomain::Untransformed;
    recurrent_format.codec = CodecKind::Fp32;
    recurrent_format.cache_policy = CachePolicy::Recurrent;
    recurrent_format.layout_policy = LayoutPolicy::ChannelMajorHistory;
    recurrent_format.alignment = 64;
    const uint32_t conv_state = static_cast<uint32_t>(semantics.states.size());
    semantics.states.push_back({conv_state, StateKind::RecurrentConvHistory, 4, StateUpdateKind::ShiftHistory,
                                PositionPolicy::ReplaceAtCursor,
                                {{DimensionKind::Constant, channels}, {DimensionKind::Constant, kernel - 1}},
                                {recurrent_format}, 0});
    recurrent_format.layout_policy = LayoutPolicy::ValueHeadKeyRowOutputColumn;
    const uint32_t delta_state = static_cast<uint32_t>(semantics.states.size());
    semantics.states.push_back({delta_state, StateKind::RecurrentDeltaMatrix, 4, StateUpdateKind::DeltaMatrix,
                                PositionPolicy::ReplaceAtCursor,
                                {{DimensionKind::Constant, value_heads}, {DimensionKind::Constant, head_dimension},
                                 {DimensionKind::Constant, head_dimension}}, {recurrent_format}, 0});

    std::vector<SemanticOperator> recurrent;
    const auto add = [&](OperatorKind kind, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                         std::vector<uint32_t> tensors, std::vector<uint32_t> states, OperatorPayload payload) {
        SemanticOperator op;
        op.kind = kind;
        op.semantic_version = 4;
        op.inputs = std::move(inputs);
        op.outputs = std::move(outputs);
        op.tensors = std::move(tensors);
        op.states = std::move(states);
        op.payload = std::move(payload);
        recurrent.push_back(std::move(op));
    };
    constexpr uint32_t epsilon_bits = 0x358637bdu;
    add(OperatorKind::RmsNorm, {input}, {normalized}, {input_norm}, {}, RmsNormPayload{epsilon_bits, -1, 1});
    add(OperatorKind::Linear, {normalized}, {qkv_value}, {qkv}, {}, LinearPayload{});
    add(OperatorKind::Linear, {normalized}, {gate_value}, {gate}, {}, LinearPayload{});
    add(OperatorKind::Linear, {normalized}, {beta_value}, {beta}, {}, LinearPayload{});
    add(OperatorKind::Linear, {normalized}, {alpha_value}, {alpha}, {}, LinearPayload{});
    add(OperatorKind::DepthwiseConvSilu, {qkv_value}, {conv_query, conv_key, conv_value}, {conv}, {conv_state},
        DepthwiseConvSiluPayload{qk_heads, value_heads, head_dimension, kernel});
    add(OperatorKind::L2Normalize, {conv_query}, {normalized_query}, {}, {}, L2NormalizePayload{epsilon_bits});
    add(OperatorKind::L2Normalize, {conv_key}, {normalized_key}, {}, {}, L2NormalizePayload{epsilon_bits});
    add(OperatorKind::GatedDeltaNet, {normalized_query, normalized_key, conv_value, beta_value, alpha_value}, {delta_value},
        {dt_bias, decay}, {delta_state},
        GatedDeltaNetPayload{qk_heads, value_heads, head_dimension, QkHeadMapping::ValueHeadModulo,
                             BetaTransform::Sigmoid, DecayTransform::NegativeSoftplus,
                             DeltaStateLayout::ValueHeadKeyRowOutputColumn, 0});
    add(OperatorKind::GatedRmsNorm, {delta_value, gate_value}, {gated_value}, {recurrent_norm}, {},
        GatedRmsNormPayload{epsilon_bits, ActivationKind::Silu, 1});
    add(OperatorKind::Linear, {gated_value}, {recurrent_output}, {output}, {}, LinearPayload{});
    add(OperatorKind::Add, {input, recurrent_output}, {residual}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {residual}, {ffn_normalized}, {ffn_norm}, {}, RmsNormPayload{epsilon_bits, -1, 1});
    add(OperatorKind::Linear, {ffn_normalized}, {ffn_gate_value}, {ffn_gate}, {}, LinearPayload{});
    add(OperatorKind::Linear, {ffn_normalized}, {ffn_up_value}, {ffn_up}, {}, LinearPayload{});
    add(OperatorKind::SwiGlu, {ffn_gate_value, ffn_up_value}, {swiglu_value}, {}, {}, SwiGluPayload{ActivationKind::Silu});
    add(OperatorKind::Linear, {swiglu_value}, {ffn_down_value}, {ffn_down}, {}, LinearPayload{});
    add(OperatorKind::Add, {residual, ffn_down_value}, {final_residual}, {}, {}, AddPayload{});
    if (recurrent.size() != 18) return false;

    std::vector<SemanticOperator> operators;
    operators.reserve(semantics.operators.size() - first_layer.operator_count + recurrent.size());
    operators.insert(operators.end(), semantics.operators.begin(), semantics.operators.begin() + first_layer.first_operator);
    operators.insert(operators.end(), recurrent.begin(), recurrent.end());
    operators.insert(operators.end(), semantics.operators.begin() + first_end, semantics.operators.end());
    semantics.operators = std::move(operators);
    const int32_t delta = static_cast<int32_t>(recurrent.size()) - static_cast<int32_t>(first_layer.operator_count);
    first_layer.operator_count = static_cast<uint32_t>(recurrent.size());
    for (size_t index = 1; index != semantics.layers.size(); ++index) {
        semantics.layers[index].first_operator = static_cast<uint32_t>(static_cast<int32_t>(semantics.layers[index].first_operator) + delta);
    }
    for (uint32_t index = 0; index != semantics.operators.size(); ++index) semantics.operators[index].id = index;
    return true;
}

bool use_q6k_tensor(SemanticModel& semantics, TensorRole role) {
    for (SemanticTensor& tensor : semantics.tensors) {
        if (tensor.role != role || tensor.dimensions.size() != 2 || tensor.planes.size() != 1) continue;
        uint64_t elements = 1;
        for (const Dimension& dimension : tensor.dimensions) {
            if (dimension.kind != DimensionKind::Constant || dimension.constant_or_symbol == 0 ||
                dimension.constant_or_symbol > UINT64_MAX / elements) return false;
            elements *= dimension.constant_or_symbol;
        }
        if (elements % 256 != 0) return false;
        tensor.layout.kind = PhysicalLayoutKind::GgufBlocked;
        tensor.layout.packing = PackingKind::Gguf;
        tensor.layout.rank = 2;
        tensor.layout.block_rank = 1;
        tensor.layout.block_elements = 256;
        tensor.layout.block_bytes = 210;
        tensor.quantization.kind = QuantizationKind::BlockedAffine;
        tensor.quantization.accumulation_type = ScalarType::F32;
        tensor.quantization.scale_type = ScalarType::F16;
        tensor.quantization.zero_type = ScalarType::F16;
        tensor.quantization.block_elements = 256;
        tensor.quantization.block_bytes = 210;
        tensor.quantization.group_size = 256;
        tensor.quantization.required_plane_mask = 1;
        tensor.planes[0].storage_type = ScalarType::U8;
        tensor.planes[0].alignment = 32;
        tensor.planes[0].length = elements / 256 * 210;
        return true;
    }
    return false;
}

bool use_q6k_output_projection(SemanticModel& semantics) {
    return use_q6k_tensor(semantics, TensorRole::OutputWeight);
}

bool use_q6k_token_embedding(SemanticModel& semantics) {
    return use_q6k_tensor(semantics, TensorRole::TokenEmbedding);
}

bool use_q4k_token_embedding(SemanticModel& semantics) {
    for (SemanticTensor& tensor : semantics.tensors) {
        if (tensor.role != TensorRole::TokenEmbedding || tensor.dimensions.size() != 2 || tensor.planes.size() != 1) continue;
        uint64_t elements = 1;
        for (const Dimension& dimension : tensor.dimensions) {
            if (dimension.kind != DimensionKind::Constant || dimension.constant_or_symbol == 0 ||
                dimension.constant_or_symbol > UINT64_MAX / elements) return false;
            elements *= dimension.constant_or_symbol;
        }
        if (elements % 256 != 0) return false;
        tensor.layout.kind = PhysicalLayoutKind::GgufBlocked;
        tensor.layout.packing = PackingKind::Gguf;
        tensor.layout.rank = 2;
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
        tensor.planes[0].storage_type = ScalarType::U8;
        tensor.planes[0].alignment = 32;
        tensor.planes[0].length = elements / 256 * 144;
        return true;
    }
    return false;
}

SessionRequest canonical_metal_request() {
    SessionRequest request;
    request.max_context = 8;
    request.max_batch = 1;
    request.memory_limit = UINT64_MAX;
    request.enable_prefill = true;
    request.enable_decode = true;
    request.minimum_class = NumericalClass::ExactFp32;
    return request;
}

RuntimeCapabilities canonical_metal_capabilities() {
    RuntimeCapabilities capabilities;
    capabilities.global_fp32_kv = true;
    capabilities.transactional_state = true;
    capabilities.metal_device = true;
    capabilities.metal_library = true;
    capabilities.metal_pipeline = true;
    return capabilities;
}

void test_calibration_target_is_versioned_and_maps_the_linear_k_axis() {
    SemanticModel model;
    model.values = {
        {7, ScalarType::F32,
         {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 7}}, 0},
        {8, ScalarType::F32,
         {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 3}}, 0},
    };
    SemanticTensor weight;
    weight.id = 11;
    weight.role = TensorRole::FfnGateWeight;
    weight.logical_type = ScalarType::F32;
    weight.dimensions = {{DimensionKind::Constant, 7}, {DimensionKind::Constant, 3}};
    weight.layout.rank = 2;
    weight.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    weight.layout.strides[0] = 1;
    weight.layout.strides[1] = 7;
    weight.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, 0,
                      21 * sizeof(float), 4, 0}};
    model.tensors = {weight};
    model.operators = {
        {13, OperatorKind::Linear, 1, {7}, {8}, {11}, {}, LinearPayload{}},
    };

    CalibrationTarget target;
    target.operator_id = 13;
    target.input_value_id = 7;
    target.weight_tensor_id = 11;
    target.k_mapping.input_axis = 1;
    target.k_mapping.weight_physical_axis = 0;
    target.k_mapping.weight_logical_axis = 0;
    target.k_mapping.width = 7;
    target.k_mapping.block_elements = 1;
    target.k_mapping.block_bytes = sizeof(float);

    auto accepted = validate_calibration_target(model, target, true);
    CHECK(std::holds_alternative<CalibrationTarget>(accepted));

    CalibrationTarget bad_version = target;
    bad_version.version = 2;
    auto rejected = validate_calibration_target(model, bad_version, true);
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    if (const auto* report = std::get_if<CompatibilityReport>(&rejected))
        CHECK(report->code == CompatibilityError::IR_VERSION_UNSUPPORTED);

    CalibrationTarget bad_axis = target;
    bad_axis.k_mapping.weight_physical_axis = 1;
    rejected = validate_calibration_target(model, bad_axis, true);
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    if (const auto* report = std::get_if<CompatibilityReport>(&rejected))
        CHECK(report->code == CompatibilityError::IR_LAYOUT_MISMATCH);

    rejected = validate_calibration_target(model, target, false);
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    if (const auto* report = std::get_if<CompatibilityReport>(&rejected))
        CHECK(report->code == CompatibilityError::CAPABILITY_MISSING);

    model.values.push_back({9, ScalarType::F32,
                            {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 7}}, 0});
    model.operators[0].inputs.push_back(9);
    CalibrationTarget mislabeled = target;
    mislabeled.input_value_id = 9;
    rejected = validate_calibration_target(model, mislabeled, true);
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    if (const auto* report = std::get_if<CompatibilityReport>(&rejected))
        CHECK(report->code == CompatibilityError::IR_REFERENCE_INVALID);
}

void test_calibration_cache_roundtrip_and_key_mismatch() {
    CalibrationTarget target;
    target.operator_id = 13;
    target.input_value_id = 7;
    target.weight_tensor_id = 11;
    target.k_mapping.input_axis = 1;
    target.k_mapping.weight_physical_axis = 0;
    target.k_mapping.weight_logical_axis = 0;
    target.k_mapping.width = 3;
    target.k_mapping.block_elements = 256;
    target.k_mapping.block_bytes = 144;

    CalibrationCacheBundle source;
    source.artifact_digest.bytes[0] = 1;
    source.semantic_fingerprint.bytes[0] = 5;
    source.corpus_digest.bytes[0] = 3;
    source.token_digest.bytes[0] = 4;
    CalibrationRecord record;
    record.target = target;
    record.sample_count = 2;
    record.sum_squares = {1.25f, 5.0f, 2.5f};
    source.records.push_back(record);
    source.target_set_digest = calibration_target_set_digest(
        std::span<const CalibrationTarget>(&source.records[0].target, 1));

    const std::array<uint32_t, 4> token_ids = {1, 2, 3, 4};
    const std::array<uint32_t, 1> one_window = {4};
    const std::array<uint32_t, 2> two_windows = {2, 2};
    Sha256Digest one_window_digest, two_window_digest;
    CHECK(calibration_token_window_digest(token_ids, one_window, one_window_digest));
    CHECK(calibration_token_window_digest(token_ids, two_windows, two_window_digest));
    CHECK(one_window_digest != two_window_digest);
    const std::array<uint32_t, 1> bad_window = {3};
    CHECK(!calibration_token_window_digest(token_ids, bad_window, two_window_digest));
    CHECK(calibration_cache_value_count_valid_for_testing(0, 1));
    CHECK(!calibration_cache_value_count_valid_for_testing(0, UINT64_MAX));
    CalibrationRecord normalization;
    normalization.target = target;
    normalization.sample_count = 2;
    normalization.sum_squares = {2.0f, 0.0f, 18.0f};
    std::vector<float> importance;
    CHECK(normalize_calibration_record(normalization, importance));
    CHECK(importance == std::vector<float>({1.0f, 0.0f, 9.0f}));
    CalibrationCacheBundle calibrated = source;
    calibrated.records = {normalization};
    calibrated.target_set_digest = calibration_target_set_digest(
        std::span<const CalibrationTarget>(&calibrated.records[0].target, 1));
    importance.clear();
    CHECK(calibration_importance_for_tensor(
        calibrated, calibrated.artifact_digest, calibrated.semantic_fingerprint,
        target.weight_tensor_id, target.k_mapping, importance));
    CHECK(importance == std::vector<float>({1.0f, 0.0f, 9.0f}));
    CalibrationCacheBundle qualified = calibrated;
    qualified.records[0].sum_squares = {2.0f, 4.0f, 18.0f};
    CHECK(calibration_importance_for_tensor(
        qualified, qualified.artifact_digest, qualified.semantic_fingerprint,
        target.weight_tensor_id, target.k_mapping, importance));
    CHECK(importance == std::vector<float>({1.0f, 2.0f, 9.0f}));
    CalibrationCacheBundle ambiguous = qualified;
    CalibrationRecord duplicate_tensor = qualified.records[0];
    ++duplicate_tensor.target.operator_id;
    ambiguous.records.push_back(duplicate_tensor);
    const std::array<CalibrationTarget, 2> ambiguous_targets = {
        ambiguous.records[0].target, ambiguous.records[1].target};
    ambiguous.target_set_digest = calibration_target_set_digest(ambiguous_targets);
    CHECK(!calibration_importance_for_tensor(
        ambiguous, ambiguous.artifact_digest, ambiguous.semantic_fingerprint,
        target.weight_tensor_id, target.k_mapping, importance));
    CalibrationKMapping wrong_mapping = target.k_mapping;
    ++wrong_mapping.width;
    CHECK(!calibration_importance_for_tensor(
        qualified, qualified.artifact_digest, qualified.semantic_fingerprint,
        target.weight_tensor_id, wrong_mapping, importance));
    Sha256Digest wrong_artifact = qualified.artifact_digest;
    wrong_artifact.bytes[0] ^= 1;
    CHECK(!calibration_importance_for_tensor(
        qualified, wrong_artifact, qualified.semantic_fingerprint,
        target.weight_tensor_id, target.k_mapping, importance));

    CalibrationRecord first = record;
    first.sum_squares = {1.0f, 2.0f, 0.0f};
    CalibrationRecord second = record;
    second.sample_count = 3;
    second.sum_squares = {3.0f, 4.0f, 0.0f};
    std::vector<CalibrationRecord> merged;
    CHECK(merge_calibration_records(merged, std::span<const CalibrationRecord>(&first, 1)));
    CHECK(merge_calibration_records(merged, std::span<const CalibrationRecord>(&second, 1)));
    CHECK(merged.size() == 1);
    CHECK(merged[0].sample_count == 5);
    CHECK(merged[0].sum_squares == std::vector<float>({4.0f, 6.0f, 0.0f}));

    const std::vector<CalibrationRecord> merged_before_rejection = merged;
    CalibrationRecord wrong_target = second;
    ++wrong_target.target.operator_id;
    CHECK(!merge_calibration_records(merged, std::span<const CalibrationRecord>(&wrong_target, 1)));
    CHECK(merged.size() == merged_before_rejection.size());
    CHECK(merged[0].target == merged_before_rejection[0].target);
    CHECK(merged[0].sample_count == merged_before_rejection[0].sample_count);
    CHECK(merged[0].sum_squares == merged_before_rejection[0].sum_squares);
    CalibrationRecord wrong_width = second;
    wrong_width.sum_squares.pop_back();
    CHECK(!merge_calibration_records(merged, std::span<const CalibrationRecord>(&wrong_width, 1)));
    CHECK(merged[0].sample_count == merged_before_rejection[0].sample_count);
    CHECK(merged[0].sum_squares == merged_before_rejection[0].sum_squares);

    std::vector<CalibrationRecord> count_overflow = {first};
    count_overflow[0].sample_count = UINT32_MAX;
    CHECK(!merge_calibration_records(count_overflow,
                                      std::span<const CalibrationRecord>(&second, 1)));
    CHECK(count_overflow[0].sample_count == UINT32_MAX);
    std::vector<CalibrationRecord> sum_overflow = {first};
    sum_overflow[0].sum_squares[0] = std::numeric_limits<float>::max();
    CalibrationRecord large = second;
    large.sum_squares[0] = std::numeric_limits<float>::max();
    CHECK(!merge_calibration_records(sum_overflow,
                                      std::span<const CalibrationRecord>(&large, 1)));
    CHECK(sum_overflow[0].sum_squares[0] == std::numeric_limits<float>::max());

    std::vector<uint8_t> bytes;
    CHECK(encode_calibration_cache(source, bytes));
    CHECK(!bytes.empty());
    auto decoded = decode_calibration_cache(bytes, source.artifact_digest,
                                            source.semantic_fingerprint,
                                            source.target_set_digest, source.corpus_digest,
                                            source.token_digest);
    CHECK(std::holds_alternative<CalibrationCacheBundle>(decoded));
    if (const auto* cache = std::get_if<CalibrationCacheBundle>(&decoded)) {
        CHECK(cache->records.size() == 1);
        CHECK(cache->records[0].target == target);
        CHECK(cache->records[0].sample_count == 2);
        CHECK(cache->records[0].sum_squares == record.sum_squares);
    }

    Sha256Digest stale = source.token_digest;
    stale.bytes[0] ^= 0xff;
    auto rejected = decode_calibration_cache(bytes, source.artifact_digest,
                                             source.semantic_fingerprint,
                                             source.target_set_digest, source.corpus_digest, stale);
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));

    CalibrationCacheBundle duplicate = source;
    duplicate.records.push_back(record);
    const std::array<CalibrationTarget, 2> duplicate_targets = {target, target};
    duplicate.target_set_digest = calibration_target_set_digest(duplicate_targets);
    std::vector<uint8_t> duplicate_bytes;
    CHECK(!encode_calibration_cache(duplicate, duplicate_bytes));
    if (const auto* report = std::get_if<CompatibilityReport>(&rejected))
        CHECK(report->code == CompatibilityError::PACKAGE_CHECKSUM_MISMATCH);

    Sha256Digest stale_semantics = source.semantic_fingerprint;
    stale_semantics.bytes[0] ^= 0xff;
    rejected = decode_calibration_cache(bytes, source.artifact_digest, stale_semantics,
                                        source.target_set_digest, source.corpus_digest,
                                        source.token_digest);
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));

    bytes.back() ^= 1;
    rejected = decode_calibration_cache(bytes, source.artifact_digest,
                                        source.semantic_fingerprint,
                                        source.target_set_digest, source.corpus_digest,
                                        source.token_digest);
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));

    const std::string cache_path = "/private/tmp/laplace-calibration-cache-roundtrip.bin";
    std::remove(cache_path.c_str());
    auto missing = load_calibration_cache(cache_path, source.artifact_digest,
                                          source.semantic_fingerprint,
                                          source.target_set_digest, source.corpus_digest,
                                          source.token_digest);
    CHECK(std::holds_alternative<CompatibilityReport>(missing));
    CompatibilityReport write_error;
    CHECK(write_calibration_cache_atomic(cache_path, source, &write_error));
    auto loaded = load_calibration_cache(cache_path, source.artifact_digest,
                                         source.semantic_fingerprint,
                                         source.target_set_digest, source.corpus_digest,
                                         source.token_digest);
    CHECK(std::holds_alternative<CalibrationCacheBundle>(loaded));
    if (const auto* cache = std::get_if<CalibrationCacheBundle>(&loaded))
        CHECK(cache->records[0].sum_squares == record.sum_squares);
    CHECK(std::remove(cache_path.c_str()) == 0);

    const std::string atlas_path = "/private/tmp/laplace-iq2-atlas-cache-roundtrip.bin";
    std::remove(atlas_path.c_str());
    uint32_t conversions = UINT32_MAX;
    CHECK(canonical_metal_iq2_atlas_cache_roundtrip_for_testing(atlas_path, &conversions));
    CHECK(conversions == 1);
    conversions = UINT32_MAX;
    CHECK(canonical_metal_iq2_atlas_cache_roundtrip_for_testing(atlas_path, &conversions));
    CHECK(conversions == 0);
    CHECK(std::remove(atlas_path.c_str()) == 0);
}

void test_sparse_layer_mask_policy_surface() {
    CanonicalSparseFfnPolicy policy;
    policy.layer_masks.push_back({7, {0, 2, 5}});
    CHECK(policy.layer_masks.size() == 1);
    CHECK(policy.layer_masks.front().layer_index == 7);
    CHECK(policy.layer_masks.front().block_ids == std::vector<uint32_t>({0, 2, 5}));
}

void test_dense_f16_prefill_uses_one_completed_metal_transaction() {
    CHECK(metal_device_present());
    CHECK(metal_available());
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) return;

    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return;
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    CHECK(program.layer_count() == package->semantics().layers.size());
    CHECK(program.position() == 0);

    const std::vector<uint32_t> token_ids = {1};
    auto actual = program.prefill(token_ids);
    auto expected = reference_fp32(*package, token_ids);
    CHECK(std::holds_alternative<CanonicalMetalOutput>(actual));
    if (const auto* report = std::get_if<CompatibilityReport>(&actual)) {
        std::fprintf(stderr, "canonical Metal failure: code=%u detail=%s\n",
                     static_cast<unsigned>(report->code), report->detail.c_str());
    }
    CHECK(std::holds_alternative<ReferenceOutput>(expected));
    if (!std::holds_alternative<CanonicalMetalOutput>(actual) || !std::holds_alternative<ReferenceOutput>(expected)) return;

    const auto& output = std::get<CanonicalMetalOutput>(actual);
    const auto& reference = std::get<ReferenceOutput>(expected);
    CHECK(output.command_buffers == 1);
    CHECK(output.completed);
    CHECK(output.cpu_wait_ms > 0.0);
    CHECK(output.gpu_time_ms > 0.0);
    CHECK(output.peak_session_bytes > 0);
    CHECK(output.operator_count == package->semantics().operators.size());
    CHECK(program.position() == token_ids.size());
    CHECK(output.logits.size() == reference.logits.size());
    for (size_t index = 0; index != output.logits.size() && index != reference.logits.size(); ++index) {
        CHECK(std::isfinite(output.logits[index]));
        CHECK(std::abs(output.logits[index] - reference.logits[index]) <=
              1e-4f + 1e-4f * std::abs(reference.logits[index]));
    }
    CHECK(std::max_element(output.logits.begin(), output.logits.end()) - output.logits.begin() ==
          std::max_element(reference.logits.begin(), reference.logits.end()) - reference.logits.begin());
}

void test_incremental_decode_matches_the_same_full_prefix_oracle() {
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) return;

    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return;
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    auto prefill = program.prefill(std::vector<uint32_t>{1});
    auto decode = program.decode(2);
    auto expected = reference_fp32(*package, std::vector<uint32_t>{1, 2});
    CHECK(std::holds_alternative<CanonicalMetalOutput>(prefill));
    CHECK(std::holds_alternative<CanonicalMetalOutput>(decode));
    CHECK(std::holds_alternative<ReferenceOutput>(expected));
    if (!std::holds_alternative<CanonicalMetalOutput>(decode) || !std::holds_alternative<ReferenceOutput>(expected)) return;
    const auto& output = std::get<CanonicalMetalOutput>(decode);
    const auto& reference = std::get<ReferenceOutput>(expected);
    CHECK(output.command_buffers == 1);
    CHECK(output.completed);
    CHECK(program.position() == 2);
    CHECK(output.logits.size() == reference.logits.size());
    for (size_t index = 0; index != output.logits.size() && index != reference.logits.size(); ++index) {
        CHECK(std::isfinite(output.logits[index]));
        CHECK(std::abs(output.logits[index] - reference.logits[index]) <=
              1e-4f + 1e-4f * std::abs(reference.logits[index]));
    }
    CHECK(std::max_element(output.logits.begin(), output.logits.end()) - output.logits.begin() ==
          std::max_element(reference.logits.begin(), reference.logits.end()) - reference.logits.begin());
}

void test_final_norm_epsilon_does_not_replace_layer_epsilon() {
    auto package = load_package_with_final_epsilon(0.125f);
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return;
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    auto actual = program.prefill(std::vector<uint32_t>{1});
    auto expected = reference_fp32(*package, std::vector<uint32_t>{1});
    CHECK(std::holds_alternative<CanonicalMetalOutput>(actual));
    CHECK(std::holds_alternative<ReferenceOutput>(expected));
    if (!std::holds_alternative<CanonicalMetalOutput>(actual) || !std::holds_alternative<ReferenceOutput>(expected)) return;
    const auto& output = std::get<CanonicalMetalOutput>(actual);
    const auto& reference = std::get<ReferenceOutput>(expected);
    CHECK(output.logits.size() == reference.logits.size());
    for (size_t index = 0; index != output.logits.size() && index != reference.logits.size(); ++index) {
        if (std::abs(output.logits[index] - reference.logits[index]) >
            1e-4f + 1e-4f * std::abs(reference.logits[index])) {
            CHECK(false);
            return;
        }
    }
}

void test_rejects_dense_pattern_without_the_declared_swiglu_activation() {
    auto package = load_package_with_semantic_mutation([](SemanticModel& semantics) {
        for (SemanticOperator& op : semantics.operators) {
            if (op.kind != OperatorKind::SwiGlu) continue;
            auto* payload = std::get_if<SwiGluPayload>(&op.payload);
            if (!payload) return false;
            payload->activation = static_cast<ActivationKind>(99);
            return true;
        }
        return false;
    });
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CompatibilityReport>(created));
    if (const auto* report = std::get_if<CompatibilityReport>(&created)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
        CHECK(report->detail == "canonical dense pattern requires SiLU SwiGLU");
    }
}

void test_rejects_dense_pattern_without_global_causal_attention() {
    auto package = load_package_with_semantic_mutation([](SemanticModel& semantics) {
        for (SemanticOperator& op : semantics.operators) {
            if (op.kind != OperatorKind::CausalAttention) continue;
            auto* payload = std::get_if<CausalAttentionPayload>(&op.payload);
            if (!payload) return false;
            payload->mask = static_cast<AttentionMask>(99);
            return true;
        }
        return false;
    });
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CompatibilityReport>(created));
    if (const auto* report = std::get_if<CompatibilityReport>(&created)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
        CHECK(report->detail == "canonical dense pattern requires global causal attention");
    }
}

void test_rejects_dense_pattern_without_declared_key_value_state_ownership() {
    auto package = load_package_with_semantic_mutation([](SemanticModel& semantics) {
        for (SemanticOperator& op : semantics.operators) {
            if (op.kind != OperatorKind::CausalAttention) continue;
            op.states.clear();
            return true;
        }
        return false;
    });
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CompatibilityReport>(created));
    if (const auto* report = std::get_if<CompatibilityReport>(&created)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
        CHECK(report->detail == "canonical dense pattern requires one FP32 key/value state pair");
    }
}

void test_rejects_unowned_referenced_artifact_without_assuming_artifact_zero() {
    auto package = load_package_with_semantic_mutation([](SemanticModel& semantics) {
        if (semantics.tensors.empty() || semantics.tensors[0].planes.empty()) return false;
        semantics.tensors[0].planes[0].artifact_id = ArtifactId{1};
        return true;
    });
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CompatibilityReport>(created));
    if (const auto* report = std::get_if<CompatibilityReport>(&created)) {
        CHECK(report->code == CompatibilityError::IR_REFERENCE_INVALID);
        CHECK(report->detail == "canonical Metal references unavailable artifact 1");
    }
}

void test_interleaved_programs_do_not_share_kv_or_weight_resources() {
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) return;
    auto first_created = create_qualification_canonical_metal_program(package, 8);
    auto second_created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(first_created));
    CHECK(std::holds_alternative<CanonicalMetalProgram>(second_created));
    if (!std::holds_alternative<CanonicalMetalProgram>(first_created) ||
        !std::holds_alternative<CanonicalMetalProgram>(second_created)) return;
    CanonicalMetalProgram first = std::get<CanonicalMetalProgram>(std::move(first_created));
    CanonicalMetalProgram second = std::get<CanonicalMetalProgram>(std::move(second_created));
    CHECK(std::holds_alternative<CanonicalMetalOutput>(first.prefill(std::vector<uint32_t>{1})));
    CHECK(std::holds_alternative<CanonicalMetalOutput>(second.prefill(std::vector<uint32_t>{3})));
    auto actual = first.decode(2);
    auto expected = reference_fp32(*package, std::vector<uint32_t>{1, 2});
    CHECK(std::holds_alternative<CanonicalMetalOutput>(actual));
    CHECK(std::holds_alternative<ReferenceOutput>(expected));
    if (!std::holds_alternative<CanonicalMetalOutput>(actual) || !std::holds_alternative<ReferenceOutput>(expected)) return;
    const auto& output = std::get<CanonicalMetalOutput>(actual);
    const auto& reference = std::get<ReferenceOutput>(expected);
    CHECK(output.logits.size() == reference.logits.size());
    for (size_t index = 0; index != output.logits.size() && index != reference.logits.size(); ++index) {
        if (std::abs(output.logits[index] - reference.logits[index]) >
            1e-4f + 1e-4f * std::abs(reference.logits[index])) {
            CHECK(false);
            return;
        }
    }
}

void test_completed_command_failure_does_not_publish_position_or_kv() {
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return;
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    CHECK(std::holds_alternative<CanonicalMetalOutput>(program.prefill(std::vector<uint32_t>{1})));
    CHECK(program.position() == 1);
    canonical_metal_fail_after_completed_submission_for_testing(program);
    auto failed = program.decode(2);
    CHECK(std::holds_alternative<CompatibilityReport>(failed));
    if (const auto* report = std::get_if<CompatibilityReport>(&failed)) {
        CHECK(report->code == CompatibilityError::RUNTIME_NUMERICAL_FAILURE);
        CHECK(report->detail.find("injected post-completion failure") != std::string::npos);
    }
    CHECK(program.position() == 1);
    auto actual = program.decode(2);
    auto expected = reference_fp32(*package, std::vector<uint32_t>{1, 2});
    CHECK(std::holds_alternative<CanonicalMetalOutput>(actual));
    CHECK(std::holds_alternative<ReferenceOutput>(expected));
    if (!std::holds_alternative<CanonicalMetalOutput>(actual) || !std::holds_alternative<ReferenceOutput>(expected)) return;
    const auto& output = std::get<CanonicalMetalOutput>(actual);
    const auto& reference = std::get<ReferenceOutput>(expected);
    CHECK(output.logits.size() == reference.logits.size());
    for (size_t index = 0; index != output.logits.size() && index != reference.logits.size(); ++index) {
        if (std::abs(output.logits[index] - reference.logits[index]) >
            1e-4f + 1e-4f * std::abs(reference.logits[index])) {
            CHECK(false);
            return;
        }
    }
}

void test_prefill_submits_the_prompt_as_one_metal_transaction() {
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return;
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    const std::vector<uint32_t> prompt = {1, 2};
    auto actual = program.prefill(prompt);
    auto expected = reference_fp32(*package, prompt);
    CHECK(std::holds_alternative<CanonicalMetalOutput>(actual));
    CHECK(std::holds_alternative<ReferenceOutput>(expected));
    if (!std::holds_alternative<CanonicalMetalOutput>(actual) || !std::holds_alternative<ReferenceOutput>(expected)) return;
    const auto& output = std::get<CanonicalMetalOutput>(actual);
    const auto& reference = std::get<ReferenceOutput>(expected);
    CHECK(output.command_buffers == 1);
    CHECK(output.completed);
    CHECK(program.position() == prompt.size());
    CHECK(output.logits.size() == reference.logits.size());
    for (size_t index = 0; index != output.logits.size() && index != reference.logits.size(); ++index) {
        if (std::abs(output.logits[index] - reference.logits[index]) >
            1e-4f + 1e-4f * std::abs(reference.logits[index])) {
            CHECK(false);
            return;
        }
    }
}

void test_canonical_metal_checkpoint_commit_and_rollback_are_transactional() {
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return;
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    CHECK(std::holds_alternative<CanonicalMetalOutput>(program.prefill(std::vector<uint32_t>{1})));
    const CanonicalMetalCursor cursor = program.checkpoint();
    CHECK(cursor.position == 1);
    CHECK(std::holds_alternative<CanonicalMetalOutput>(program.decode(2)));
    CHECK(program.commit(cursor));
    CHECK(program.position() == 2);
    CHECK(program.rollback(cursor));
    CHECK(program.position() == 1);
    CHECK(!program.commit(cursor));
    auto actual = program.decode(3);
    auto expected = reference_fp32(*package, std::vector<uint32_t>{1, 3});
    CHECK(std::holds_alternative<CanonicalMetalOutput>(actual));
    CHECK(std::holds_alternative<ReferenceOutput>(expected));
    if (!std::holds_alternative<CanonicalMetalOutput>(actual) || !std::holds_alternative<ReferenceOutput>(expected)) return;
    const auto& output = std::get<CanonicalMetalOutput>(actual);
    const auto& reference = std::get<ReferenceOutput>(expected);
    CHECK(output.logits.size() == reference.logits.size());
    for (size_t index = 0; index != output.logits.size() && index != reference.logits.size(); ++index) {
        if (std::abs(output.logits[index] - reference.logits[index]) >
            1e-4f + 1e-4f * std::abs(reference.logits[index])) {
            CHECK(false);
            return;
        }
    }
}

void test_planner_selects_only_the_exact_dense_metal_pattern() {
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) return;
    auto planned = plan_canonical_metal_dense(package->semantics(), canonical_metal_request(),
                                              canonical_metal_capabilities(), builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(planned));
    if (const auto* plan = std::get_if<ExecutionPlan>(&planned)) {
        CHECK(plan->entries.size() == package->semantics().layers.size() * 2);
        for (const PlanEntry& entry : plan->entries) {
            CHECK(entry.descriptor.implementation == KernelImplementation::MetalDenseToken);
            CHECK(plan_entry_matches(package->semantics(), entry));
        }
    }
    auto unsupported = load_package_with_semantic_mutation([](SemanticModel& semantics) {
        for (SemanticOperator& op : semantics.operators) {
            if (op.kind != OperatorKind::SwiGlu) continue;
            auto* payload = std::get_if<SwiGluPayload>(&op.payload);
            if (!payload) return false;
            payload->activation = static_cast<ActivationKind>(99);
            return true;
        }
        return false;
    });
    CHECK(unsupported != nullptr);
    if (!unsupported) return;
    planned = plan_canonical_metal_dense(unsupported->semantics(), canonical_metal_request(),
                                         canonical_metal_capabilities(), builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(planned));
    if (const auto* report = std::get_if<CompatibilityReport>(&planned)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
    }
    auto extra = load_package_with_semantic_mutation([](SemanticModel& semantics) {
        SemanticOperator unexpected;
        unexpected.id = 0xffffffffu;
        unexpected.kind = OperatorKind::Add;
        unexpected.semantic_version = 1;
        unexpected.payload = AddPayload{};
        semantics.operators.push_back(std::move(unexpected));
        return true;
    });
    CHECK(extra != nullptr);
    if (!extra) return;
    planned = plan_canonical_metal_dense(extra->semantics(), canonical_metal_request(),
                                         canonical_metal_capabilities(), builtin_canonical_metal_registry());
    CHECK(std::holds_alternative<CompatibilityReport>(planned));
    if (const auto* report = std::get_if<CompatibilityReport>(&planned)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
    }
    auto created = create_qualification_canonical_metal_program(extra, 8);
    CHECK(std::holds_alternative<CompatibilityReport>(created));
    if (const auto* report = std::get_if<CompatibilityReport>(&created)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
    }
}

void test_canonical_metal_interleaved_rope_matches_independent_reference() {
    auto package = load_package_with_semantic_mutation([](SemanticModel& semantics) {
        for (SemanticOperator& op : semantics.operators) {
            if (op.kind != OperatorKind::Rope) continue;
            auto* payload = std::get_if<RopePayload>(&op.payload);
            if (!payload) return false;
            payload->pairing = RopePairing::Interleaved;
        }
        return true;
    });
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return;
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    const std::vector<uint32_t> token_ids = {1, 2};
    auto actual = program.prefill(token_ids);
    auto expected = reference_fp32(*package, token_ids);
    CHECK(std::holds_alternative<CanonicalMetalOutput>(actual));
    CHECK(std::holds_alternative<ReferenceOutput>(expected));
    if (!std::holds_alternative<CanonicalMetalOutput>(actual) || !std::holds_alternative<ReferenceOutput>(expected)) return;
    const auto& actual_logits = std::get<CanonicalMetalOutput>(actual).logits;
    const auto& expected_logits = std::get<ReferenceOutput>(expected).logits;
    CHECK(actual_logits.size() == expected_logits.size());
    for (size_t index = 0; index != actual_logits.size() && index != expected_logits.size(); ++index) {
        CHECK(std::abs(actual_logits[index] - expected_logits[index]) <=
              1e-4f + 1e-4f * std::abs(expected_logits[index]));
    }
}

void test_canonical_metal_multisection_rope_matches_independent_reference() {
    auto package = load_package_with_semantic_mutation([](SemanticModel& semantics) {
        semantics.schema_major = 6;
        semantics.schema_minor = 0;
        semantics.opset_major = 6;
        semantics.opset_minor = 0;
        for (SemanticOperator& op : semantics.operators) {
            op.semantic_version = 6;
            if (op.kind != OperatorKind::Rope) continue;
            auto* payload = std::get_if<RopePayload>(&op.payload);
            if (!payload || payload->rotary_dimension < 8) return false;
            payload->pairing = RopePairing::MultiSectionHalfSplit;
            payload->position_sections = {1, 1, 1, payload->rotary_dimension / 2 - 3};
        }
        for (SemanticState& state : semantics.states) state.semantic_version = 6;
        return true;
    });
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return;
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    const std::vector<uint32_t> token_ids = {1, 2};
    auto actual = program.prefill(token_ids);
    auto expected = reference_fp32(*package, token_ids);
    CHECK(std::holds_alternative<CanonicalMetalOutput>(actual));
    CHECK(std::holds_alternative<ReferenceOutput>(expected));
    if (!std::holds_alternative<CanonicalMetalOutput>(actual) || !std::holds_alternative<ReferenceOutput>(expected)) return;
    const auto& actual_logits = std::get<CanonicalMetalOutput>(actual).logits;
    const auto& expected_logits = std::get<ReferenceOutput>(expected).logits;
    CHECK(actual_logits.size() == expected_logits.size());
    for (size_t index = 0; index != actual_logits.size() && index != expected_logits.size(); ++index) {
        CHECK(std::abs(actual_logits[index] - expected_logits[index]) <=
              1e-4f + 1e-4f * std::abs(expected_logits[index]));
    }
}

void test_canonical_metal_qk_norm_matches_independent_reference() {
    auto package = load_package_with_semantic_mutation(add_qk_norm_to_first_layer);
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return;
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    const std::vector<uint32_t> token_ids = {1, 2};
    auto actual = program.prefill(token_ids);
    auto expected = reference_fp32(*package, token_ids);
    CHECK(std::holds_alternative<CanonicalMetalOutput>(actual));
    CHECK(std::holds_alternative<ReferenceOutput>(expected));
    if (!std::holds_alternative<CanonicalMetalOutput>(actual) || !std::holds_alternative<ReferenceOutput>(expected)) return;
    const auto& actual_logits = std::get<CanonicalMetalOutput>(actual).logits;
    const auto& expected_logits = std::get<ReferenceOutput>(expected).logits;
    CHECK(actual_logits.size() == expected_logits.size());
    for (size_t index = 0; index != actual_logits.size() && index != expected_logits.size(); ++index) {
        if (std::abs(actual_logits[index] - expected_logits[index]) >
            1e-4f + 1e-4f * std::abs(expected_logits[index])) {
            CHECK(false);
            return;
        }
    }
}

void test_canonical_metal_binds_the_exact_query_gate_pattern() {
    auto package = load_package_with_semantic_mutation(add_query_gate_to_first_layer);
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));

    auto malformed = load_package_with_semantic_mutation([](SemanticModel& semantics) {
        if (!add_query_gate_to_first_layer(semantics)) return false;
        const auto gated = std::find_if(semantics.operators.begin(), semantics.operators.end(), [](const SemanticOperator& op) {
            return op.kind == OperatorKind::GatedAttention;
        });
        if (gated == semantics.operators.end() || gated->inputs.size() != 2) return false;
        gated->inputs[1] = gated->inputs[0];
        return true;
    });
    CHECK(malformed != nullptr);
    if (!malformed) return;
    created = create_qualification_canonical_metal_program(malformed, 8);
    CHECK(std::holds_alternative<CompatibilityReport>(created));
    if (const auto* report = std::get_if<CompatibilityReport>(&created)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
    }
}

void test_canonical_metal_lowers_a_planned_recurrent_layer() {
    auto package = load_package_with_semantic_mutation(replace_first_dense_layer_with_recurrent);
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (const auto* program = std::get_if<CanonicalMetalProgram>(&created)) {
        const char* preflight = canonical_metal_first_recurrent_preflight_for_testing(*program);
        fprintf(stderr, "compact canonical Metal recurrent preflight: %s\n", preflight ? preflight : "OK");
        CHECK(preflight != nullptr);
        CHECK(preflight && std::strcmp(preflight, "recurrent QKV matrix physical contract is unsupported") == 0);
    }

    auto malformed = load_package_with_semantic_mutation([](SemanticModel& semantics) {
        if (!replace_first_dense_layer_with_recurrent(semantics) || semantics.states.size() < 2) return false;
        semantics.states.back().semantic_version = 3;
        return true;
    });
    CHECK(malformed != nullptr);
    if (!malformed) return;
    created = create_qualification_canonical_metal_program(malformed, 8);
    CHECK(std::holds_alternative<CompatibilityReport>(created));
    if (const auto* report = std::get_if<CompatibilityReport>(&created)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
    }
}

void test_canonical_metal_constructs_a_q6k_output_projection() {
    auto package = load_package_with_semantic_mutation(use_q6k_output_projection);
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
}

void test_canonical_metal_constructs_a_q6k_token_embedding() {
    auto package = load_package_with_semantic_mutation(use_q6k_token_embedding);
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
}

void test_canonical_metal_constructs_a_q4k_token_embedding() {
    auto package = load_package_with_semantic_mutation(use_q4k_token_embedding);
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
}

void test_metal_prefill_and_incremental_decode_match_prefixes_one_through_eight() {
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) return;
    for (uint32_t prefix = 1; prefix <= 8; ++prefix) {
        std::vector<uint32_t> token_ids;
        for (uint32_t token = 1; token <= prefix; ++token) token_ids.push_back(token);
        auto one_shot_created = create_qualification_canonical_metal_program(package, 8);
        auto incremental_created = create_qualification_canonical_metal_program(package, 8);
        CHECK(std::holds_alternative<CanonicalMetalProgram>(one_shot_created));
        CHECK(std::holds_alternative<CanonicalMetalProgram>(incremental_created));
        if (!std::holds_alternative<CanonicalMetalProgram>(one_shot_created) ||
            !std::holds_alternative<CanonicalMetalProgram>(incremental_created)) return;
        CanonicalMetalProgram one_shot = std::get<CanonicalMetalProgram>(std::move(one_shot_created));
        CanonicalMetalProgram incremental = std::get<CanonicalMetalProgram>(std::move(incremental_created));
        auto one_shot_output = one_shot.prefill(token_ids);
        CanonicalMetalRunResult incremental_output = incremental.prefill(std::span<const uint32_t>(token_ids.data(), 1));
        for (size_t token = 1; token != token_ids.size(); ++token) incremental_output = incremental.decode(token_ids[token]);
        auto expected = reference_fp32(*package, token_ids);
        CHECK(std::holds_alternative<CanonicalMetalOutput>(one_shot_output));
        CHECK(std::holds_alternative<CanonicalMetalOutput>(incremental_output));
        CHECK(std::holds_alternative<ReferenceOutput>(expected));
        if (!std::holds_alternative<CanonicalMetalOutput>(one_shot_output) ||
            !std::holds_alternative<CanonicalMetalOutput>(incremental_output) ||
            !std::holds_alternative<ReferenceOutput>(expected)) return;
        const auto& one_shot_logits = std::get<CanonicalMetalOutput>(one_shot_output).logits;
        const auto& incremental_logits = std::get<CanonicalMetalOutput>(incremental_output).logits;
        const auto& reference_logits = std::get<ReferenceOutput>(expected).logits;
        CHECK(one_shot.position() == prefix);
        CHECK(incremental.position() == prefix);
        CHECK(one_shot_logits.size() == reference_logits.size());
        CHECK(incremental_logits.size() == reference_logits.size());
        for (size_t index = 0; index != reference_logits.size() && index != one_shot_logits.size() &&
                            index != incremental_logits.size(); ++index) {
            const float tolerance = 1e-4f + 1e-4f * std::abs(reference_logits[index]);
            if (std::abs(one_shot_logits[index] - reference_logits[index]) > tolerance ||
                std::abs(incremental_logits[index] - reference_logits[index]) > tolerance ||
                std::abs(one_shot_logits[index] - incremental_logits[index]) > tolerance) {
                CHECK(false);
                return;
            }
        }
    }
}

void test_profile_mode_reports_decode_segments_without_changing_results() {
    setenv("LAPLACE_CANONICAL_METAL_PROFILE", "1", 1);
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) {
        unsetenv("LAPLACE_CANONICAL_METAL_PROFILE");
        return;
    }
    auto created = create_qualification_canonical_metal_program(package, 8);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) {
        unsetenv("LAPLACE_CANONICAL_METAL_PROFILE");
        return;
    }
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    auto actual = program.prefill(std::vector<uint32_t>{1});
    auto expected = reference_fp32(*package, std::vector<uint32_t>{1});
    unsetenv("LAPLACE_CANONICAL_METAL_PROFILE");
    CHECK(std::holds_alternative<CanonicalMetalOutput>(actual));
    CHECK(std::holds_alternative<ReferenceOutput>(expected));
    if (!std::holds_alternative<CanonicalMetalOutput>(actual) || !std::holds_alternative<ReferenceOutput>(expected)) return;
    const auto& output = std::get<CanonicalMetalOutput>(actual);
    const auto& reference = std::get<ReferenceOutput>(expected);
    CHECK(!output.split_command_buffer_profile);
    if (output.counter_samples) {
        CHECK(output.profiled);
        CHECK(output.counter_sample_count == 3ull * program.layer_count() + 2ull);
        CHECK(output.qkv_gpu_ms > 0.0);
        CHECK(output.attention_gpu_ms > 0.0);
        CHECK(output.ffn_gpu_ms > 0.0);
        CHECK(output.final_gpu_ms > 0.0);
    } else {
        CHECK(!output.profiled);
        CHECK(output.counter_sample_count == 0);
    }
    CHECK(output.logits.size() == reference.logits.size());
    for (size_t index = 0; index != output.logits.size() && index != reference.logits.size(); ++index) {
        CHECK(std::abs(output.logits[index] - reference.logits[index]) <=
              1e-4f + 1e-4f * std::abs(reference.logits[index]));
    }
}

void test_q4k_tensor_lowering_requires_the_exact_block_contract() {
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) return;

    SemanticTensor semantic;
    semantic.logical_type = ScalarType::F32;
    semantic.dimensions = {{DimensionKind::Constant, 256}, {DimensionKind::Constant, 1}};
    semantic.layout.kind = PhysicalLayoutKind::GgufBlocked;
    semantic.layout.packing = PackingKind::Gguf;
    semantic.layout.rank = 2;
    semantic.layout.block_rank = 1;
    semantic.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    semantic.layout.strides[0] = 1;
    semantic.layout.strides[1] = 256;
    semantic.layout.block_elements = 256;
    semantic.layout.block_bytes = 144;
    semantic.quantization.kind = QuantizationKind::BlockedAffine;
    semantic.quantization.accumulation_type = ScalarType::F32;
    semantic.quantization.scale_type = ScalarType::F16;
    semantic.quantization.zero_type = ScalarType::F16;
    semantic.quantization.block_elements = 256;
    semantic.quantization.block_bytes = 144;
    semantic.quantization.group_size = 256;
    semantic.quantization.required_plane_mask = 1;
    semantic.planes = {{PlaneKind::Values, ScalarType::U8, ArtifactId{0}, 0, 144, 32, 0}};

    Tensor lowered;
    CHECK(canonical_metal_tensor_view_for_testing(*package, semantic, lowered));
    CHECK(lowered.type == GGMLType::Q4_K);
    CHECK(lowered.n_dims == 2);
    CHECK(lowered.dims[0] == 256);
    CHECK(lowered.dims[1] == 1);
    CHECK(lowered.nbytes() == 144);
    CHECK(lowered.data_bytes == 144);
    semantic.quantization.block_bytes = 143;
    CHECK(!canonical_metal_tensor_view_for_testing(*package, semantic, lowered));
}

void test_tensor_lowering_preserves_transposed_physical_layout() {
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) return;

    SemanticTensor semantic;
    semantic.logical_type = ScalarType::F32;
    semantic.dimensions = {{DimensionKind::Constant, 768}, {DimensionKind::Constant, 2}};
    semantic.layout.rank = 2;
    semantic.layout.axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    semantic.layout.strides[0] = 1;
    semantic.layout.strides[1] = 2;
    semantic.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, 0, 6144, 4, 0}};

    Tensor lowered;
    CHECK(canonical_metal_tensor_view_for_testing(*package, semantic, lowered));
    CHECK(lowered.n_dims == 2);
    CHECK(lowered.dims[0] == 2);
    CHECK(lowered.dims[1] == 768);
    CHECK(lowered.data_bytes == 6144);
}

void test_affine_u2_256_tensor_lowering_requires_all_three_planes() {
    auto package = synthetic_f16_dense_package();
    CHECK(package != nullptr);
    if (!package) return;

    SemanticTensor semantic;
    semantic.id = 91;
    semantic.role = TensorRole::FfnDownWeight;
    semantic.logical_type = ScalarType::F32;
    semantic.dimensions = {{DimensionKind::Constant, 8}, {DimensionKind::Constant, 512}};
    semantic.layout.kind = PhysicalLayoutKind::GroupedAffine;
    semantic.layout.packing = PackingKind::LsbBitPacked;
    semantic.layout.rank = 2;
    semantic.layout.block_rank = 1;
    semantic.layout.axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    semantic.layout.strides[0] = 1;
    semantic.layout.strides[1] = 512;
    semantic.layout.block_elements = 256;
    semantic.layout.block_bytes = 64;
    semantic.quantization.kind = QuantizationKind::BlockedAffine;
    semantic.quantization.accumulation_type = ScalarType::F32;
    semantic.quantization.scale_type = ScalarType::F16;
    semantic.quantization.bias_type = ScalarType::F16;
    semantic.quantization.block_elements = 256;
    semantic.quantization.block_bytes = 64;
    semantic.quantization.group_size = 256;
    semantic.quantization.required_plane_mask = 7;
    semantic.planes = {
        {PlaneKind::Values, ScalarType::U32, ArtifactId{0}, 0, 1024, 128, 0},
        {PlaneKind::Scales, ScalarType::F16, ArtifactId{0}, 1024, 32, 128, 0},
        {PlaneKind::Biases, ScalarType::F16, ArtifactId{0}, 1152, 32, 128, 0},
    };

    Tensor lowered;
    CHECK(canonical_metal_tensor_view_for_testing(*package, semantic, lowered));
    CHECK(lowered.type == GGMLType::GROUPED_AFFINE_U2_256);
    CHECK(lowered.n_dims == 2 && lowered.dims[0] == 512 && lowered.dims[1] == 8);
    CHECK(lowered.data != nullptr && lowered.scales != nullptr && lowered.biases != nullptr);
    CHECK(lowered.data_bytes == 1024 && lowered.scale_bytes == 32 && lowered.bias_bytes == 32);
    CHECK(lowered.mlx_bits == 2 && lowered.mlx_group_size == 256);

    semantic.planes[1].length = 30;
    CHECK(!canonical_metal_tensor_view_for_testing(*package, semantic, lowered));
    semantic.planes[1].length = 32;
    semantic.planes[2].alignment = 64;
    CHECK(!canonical_metal_tensor_view_for_testing(*package, semantic, lowered));
}

void test_column_grouped_affine_u2_skip_tensor_lowering_is_byte_typed() {
    auto package = synthetic_f16_dense_package();
    CHECK(package != nullptr);
    if (!package) return;

    SemanticTensor semantic;
    semantic.id = 92;
    semantic.role = TensorRole::FfnGateWeight;
    semantic.logical_type = ScalarType::F32;
    semantic.dimensions = {{DimensionKind::Constant, 256},
                           {DimensionKind::Constant, 8}};
    semantic.layout.kind = PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip;
    semantic.layout.packing = PackingKind::LsbBitPacked;
    semantic.layout.rank = 2;
    semantic.layout.block_rank = 1;
    semantic.layout.axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    semantic.layout.strides[0] = 1;
    semantic.layout.strides[1] = 8;
    semantic.layout.block_elements = 256;
    semantic.layout.block_bytes = 64;
    semantic.quantization.kind = QuantizationKind::BlockedAffine;
    semantic.quantization.accumulation_type = ScalarType::F32;
    semantic.quantization.scale_type = ScalarType::F16;
    semantic.quantization.bias_type = ScalarType::F16;
    semantic.quantization.block_elements = 256;
    semantic.quantization.block_bytes = 64;
    semantic.quantization.group_size = 256;
    semantic.quantization.required_plane_mask = 7;
    semantic.planes = {
        {PlaneKind::Values, ScalarType::U8, ArtifactId{0}, 0, 512, 128, 0},
        {PlaneKind::Scales, ScalarType::F16, ArtifactId{0}, 512, 16, 128, 0},
        {PlaneKind::Biases, ScalarType::F16, ArtifactId{0}, 640, 16, 128, 0},
    };

    Tensor lowered;
    CHECK(canonical_metal_tensor_view_for_testing(*package, semantic, lowered));
    CHECK(lowered.type == GGMLType::COLUMN_GROUPED_AFFINE_U2_SKIP_256);
    CHECK(lowered.n_dims == 2 && lowered.dims[0] == 8 && lowered.dims[1] == 256);
    CHECK(lowered.data != nullptr && lowered.scales != nullptr && lowered.biases != nullptr);
    CHECK(lowered.data_bytes == 512 && lowered.scale_bytes == 16 && lowered.bias_bytes == 16);
    CHECK(lowered.mlx_bits == 2 && lowered.mlx_group_size == 256);

    semantic.planes[0].storage_type = ScalarType::U32;
    CHECK(!canonical_metal_tensor_view_for_testing(*package, semantic, lowered));
    semantic.planes[0].storage_type = ScalarType::U8;
    semantic.layout.kind = PhysicalLayoutKind::GroupedAffine;
    CHECK(!canonical_metal_tensor_view_for_testing(*package, semantic, lowered));
    semantic.layout.kind = PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip;
    semantic.dimensions[0].constant_or_symbol = 255;
    CHECK(!canonical_metal_tensor_view_for_testing(*package, semantic, lowered));
}

void test_column_grouped_affine_u2_skip_canonical_transaction() {
    CHECK(metal_device_present());
    CHECK(metal_available());
    if (!metal_device_present() || !metal_available()) return;

    const auto candidate_package = synthetic_column_grouped_u2_dense_package(
        SyntheticU2FfnFormat::DirectU2Nonzero);
    const auto scalar_package = synthetic_column_grouped_u2_dense_package(
        SyntheticU2FfnFormat::F16Nonzero);
    CHECK(candidate_package != nullptr);
    CHECK(scalar_package != nullptr);
    if (!candidate_package || !scalar_package) return;

    const std::vector<uint32_t> tokens = {candidate_package->semantics().bos_id};
    const auto expected = reference_fp32(*scalar_package, tokens);
    CHECK(std::holds_alternative<ReferenceOutput>(expected));
    if (!std::holds_alternative<ReferenceOutput>(expected)) return;

    auto created = create_qualification_canonical_metal_program(candidate_package, 4);
    if (const auto* report = std::get_if<CompatibilityReport>(&created)) {
        std::fprintf(stderr,
                     "column-grouped U2 canonical construction: code=%u layer=%u "
                     "operator=%u detail=%s\n",
                     static_cast<unsigned>(report->code), report->layer,
                     report->operator_id, report->detail.c_str());
    }
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return;
    CanonicalMetalProgram program =
        std::get<CanonicalMetalProgram>(std::move(created));

    metal_dispatch_metrics_reset();
    const auto actual = program.prefill(tokens);
    const MetalDispatchMetrics dispatch = metal_dispatch_metrics();
    CHECK(std::holds_alternative<CanonicalMetalOutput>(actual));
    if (!std::holds_alternative<CanonicalMetalOutput>(actual)) return;
    const auto& output = std::get<CanonicalMetalOutput>(actual);
    const auto& reference = std::get<ReferenceOutput>(expected);
    CHECK(output.completed);
    CHECK(output.command_buffers == 1);
    CHECK(dispatch.command_buffers == 1);
    CHECK(output.column_grouped_affine_u2_skip_projection_dispatches == 3);
    CHECK(output.projection_dispatches == 8);
    CHECK(output.requested_projection_source_bytes > 0);
    CHECK(program.position() == 1);
    const CanonicalMetalResourceDiagnostics resources = program.resource_diagnostics();
    CHECK(resources.implicit_weight_copies == 0);
    CHECK(resources.registered_source_bytes > 0);
    CHECK(output.logits.size() == reference.logits.size());
    float max_delta = 0.0f;
    for (size_t index = 0; index != output.logits.size() &&
                           index != reference.logits.size(); ++index) {
        CHECK(std::isfinite(output.logits[index]));
        max_delta = std::max(max_delta,
                             std::abs(output.logits[index] - reference.logits[index]));
        CHECK(std::abs(output.logits[index] - reference.logits[index]) <=
              3.0e-3f + 3.0e-3f * std::abs(reference.logits[index]));
    }
    CHECK(std::max_element(output.logits.begin(), output.logits.end()) -
              output.logits.begin() ==
          std::max_element(reference.logits.begin(), reference.logits.end()) -
              reference.logits.begin());

    canonical_metal_fail_after_completed_submission_for_testing(program);
    const auto failed = program.decode(candidate_package->semantics().bos_id);
    CHECK(std::holds_alternative<CompatibilityReport>(failed));
    if (const auto* report = std::get_if<CompatibilityReport>(&failed))
        CHECK(report->code == CompatibilityError::RUNTIME_NUMERICAL_FAILURE);
    CHECK(program.position() == 1);
    const auto retried = program.decode(candidate_package->semantics().bos_id);
    CHECK(std::holds_alternative<CanonicalMetalOutput>(retried));
    CHECK(program.position() == 2);
    std::fprintf(stderr,
                 "column-grouped U2 canonical: one_cb=1 u2_dispatches=%u "
                 "registered_source=%llu implicit_copies=0 max_logit_delta=%.9g "
                 "rollback_retry=1 gpu_ms=%.6f\n",
                 output.column_grouped_affine_u2_skip_projection_dispatches,
                 static_cast<unsigned long long>(resources.registered_source_bytes),
                 max_delta, output.gpu_time_ms);
}

void test_derived_column_grouped_u2_atlas_canonical_transaction() {
    CHECK(metal_device_present());
    CHECK(metal_available());
    if (!metal_device_present() || !metal_available()) return;

    const auto candidate_package = synthetic_column_grouped_u2_dense_package(
        SyntheticU2FfnFormat::Q4KZero);
    const auto scalar_package = synthetic_column_grouped_u2_dense_package(
        SyntheticU2FfnFormat::F16Zero);
    CHECK(candidate_package != nullptr);
    CHECK(scalar_package != nullptr);
    if (!candidate_package || !scalar_package) return;

    CanonicalDerivedColumnGroupedU2Policy policy;
    policy.tensor_roles = {TensorRole::FfnGateWeight, TensorRole::FfnUpWeight,
                           TensorRole::FfnDownWeight};
    policy.uniform_importance_for_testing = true;
    auto created = create_qualification_canonical_metal_program(
        candidate_package, 4, {}, {}, {}, {}, 1, policy);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return;
    CanonicalMetalProgram program =
        std::get<CanonicalMetalProgram>(std::move(created));

    const std::vector<uint32_t> tokens = {candidate_package->semantics().bos_id};
    const auto expected = reference_fp32(*scalar_package, tokens);
    const auto actual = program.prefill(tokens);
    CHECK(std::holds_alternative<ReferenceOutput>(expected));
    CHECK(std::holds_alternative<CanonicalMetalOutput>(actual));
    if (!std::holds_alternative<ReferenceOutput>(expected) ||
        !std::holds_alternative<CanonicalMetalOutput>(actual)) return;
    const auto& reference = std::get<ReferenceOutput>(expected);
    const auto& output = std::get<CanonicalMetalOutput>(actual);
    CHECK(output.completed);
    CHECK(output.command_buffers == 1);
    CHECK(output.column_grouped_affine_u2_skip_projection_dispatches == 3);
    CHECK(program.derived_column_grouped_u2_tensor_count() == 3);
    CHECK(program.derived_column_grouped_u2_source_bytes() == 3u * 256u * 144u);
    const CanonicalMetalResourceDiagnostics resources = program.resource_diagnostics();
    CHECK(resources.excluded_replaced_bytes == 3u * 256u * 144u);
    CHECK(resources.atlas_bytes >= 3u * (16384u + 512u + 512u));
    CHECK(resources.implicit_weight_copies == 0);
    CHECK(output.logits.size() == reference.logits.size());
    for (size_t index = 0; index != output.logits.size() &&
                           index != reference.logits.size(); ++index) {
        CHECK(std::isfinite(output.logits[index]));
        CHECK(std::abs(output.logits[index] - reference.logits[index]) <=
              1.0e-4f + 1.0e-4f * std::abs(reference.logits[index]));
    }

    policy.uniform_importance_for_testing = false;
    const auto rejected = create_qualification_canonical_metal_program(
        candidate_package, 4, {}, {}, {}, {}, 1, policy);
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
}

void test_canonical_affine_u2_256_end_to_end() {
    constexpr uint64_t rows = 8;
    constexpr uint64_t columns = 512;
    constexpr uint64_t logical_elements = rows * columns;
    constexpr uint64_t values_bytes = logical_elements / 4;
    constexpr uint64_t plane_elements = logical_elements / 256;
    constexpr uint64_t plane_bytes = plane_elements * sizeof(uint16_t);
    constexpr uint64_t values_offset = 128;
    constexpr uint64_t scales_offset = values_offset + values_bytes + 128;
    constexpr uint64_t biases_offset = (scales_offset + plane_bytes + 127) & ~uint64_t{127};

    // Product-reachability witness: the source-neutral physical index binds
    // the three planes before the semantic manifest is admitted.
    std::vector<uint8_t> bytes(biases_offset + plane_bytes, 0);
    char path[] = "/private/tmp/laplace-affine-u2-index-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd < 0) return;
    CHECK(write(fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size()));
    close(fd);
    auto artifacts = ArtifactSet::load_single_file(path);
    unlink(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return;
    auto view = std::get<ArtifactSet>(std::move(artifacts)).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return;
    const PackageView& package = std::get<PackageView>(view);

    SemanticTensor semantic;
    semantic.id = 0;
    semantic.role = TensorRole::FfnDownWeight;
    semantic.logical_type = ScalarType::F32;
    semantic.dimensions = {{DimensionKind::Constant, rows},
                           {DimensionKind::Constant, columns}};
    semantic.layout.kind = PhysicalLayoutKind::GroupedAffine;
    semantic.layout.packing = PackingKind::LsbBitPacked;
    semantic.layout.rank = 2;
    semantic.layout.block_rank = 1;
    semantic.layout.axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    semantic.layout.strides[0] = 1;
    semantic.layout.strides[1] = columns;
    semantic.layout.block_elements = 256;
    semantic.layout.block_bytes = 64;
    semantic.quantization.kind = QuantizationKind::BlockedAffine;
    semantic.quantization.accumulation_type = ScalarType::F32;
    semantic.quantization.scale_type = ScalarType::F16;
    semantic.quantization.bias_type = ScalarType::F16;
    semantic.quantization.block_elements = 256;
    semantic.quantization.block_bytes = 64;
    semantic.quantization.group_size = 256;
    semantic.quantization.required_plane_mask = 7;
    semantic.planes = {
        {PlaneKind::Values, ScalarType::U32, ArtifactId{0}, values_offset, values_bytes, 128, 0},
        {PlaneKind::Scales, ScalarType::F16, ArtifactId{0}, scales_offset,
         plane_bytes, 128, 0},
        {PlaneKind::Biases, ScalarType::F16, ArtifactId{0}, biases_offset, plane_bytes, 128, 0},
    };

    ArtifactTensorRecord physical;
    physical.id = 0;
    physical.logical_type = ArtifactScalarType::F32;
    physical.logical_dimensions = {rows, columns};
    physical.layout = semantic.layout;
    physical.quantization = semantic.quantization;
    physical.axis.source_rank = 2;
    physical.axis.source_axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    physical.axis.block_axis = 1;
    physical.axis.block_elements = 256;
    physical.axis.bytes_per_block = 64;
    physical.axis.row_stride_bytes = columns / 4;
    physical.format = {2, ArtifactPhysicalEncoding::GroupedAffineU2_256,
                       ArtifactScalarType::U32, ArtifactScalarType::F16,
                       ArtifactScalarType::None, ArtifactScalarType::None,
                       ArtifactScalarType::None, 256, 64, 2, 0, 0, 0,
                       ArtifactScalarType::F16, 2};
    physical.planes = {
        {PlaneKind::Values, ArtifactScalarType::U32,
         {ArtifactId{0}, values_offset, values_bytes}, logical_elements, 4, 16, 128},
        {PlaneKind::Scales, ArtifactScalarType::F16,
         {ArtifactId{0}, scales_offset, plane_bytes}, plane_elements, 2, 1, 128},
        {PlaneKind::Biases, ArtifactScalarType::F16,
         {ArtifactId{0}, biases_offset, plane_bytes},
         plane_elements, 2, 1, 128},
    };

    ArtifactIndexInput input;
    input.artifacts = {package};
    input.tensors = {physical};
    auto index_result = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<ArtifactIndex>(index_result));
    if (!std::holds_alternative<ArtifactIndex>(index_result)) {
        if (const auto* report = std::get_if<CompatibilityReport>(&index_result))
            std::fprintf(stderr, "affine physical index setup: code=%u artifact=%u tensor=%u "
                         "offset=%llu length=%llu detail=%s\n",
                         static_cast<unsigned>(report->code), report->artifact_id.value,
                         report->tensor_id,
                         static_cast<unsigned long long>(report->source_offset),
                         static_cast<unsigned long long>(report->source_length), report->detail.c_str());
        return;
    }
    ArtifactIndex index = std::get<ArtifactIndex>(std::move(index_result));

    SemanticModel model;
    model.schema_major = 1;
    model.opset_major = 1;
    model.maximum_context = 32768;
    model.entry_kind = EntryKind::TokenIds;
    model.vocabulary_size = 1;
    model.bos_id = 0;
    model.eos_id = 0;
    model.tokenizer_digest[0] = 1;
    model.template_digest[0] = 2;
    model.tensors = {semantic};
    TokenContract contract;
    contract.vocabulary_size = model.vocabulary_size;
    contract.bos_id = model.bos_id;
    contract.eos_id = model.eos_id;
    contract.authoritative_tokenizer_digest = {model.tokenizer_digest};
    contract.authoritative_template_digest = {model.template_digest};
    const auto manifest = SemanticManifest::build(index, model, contract);
    CHECK(std::holds_alternative<SemanticManifest>(manifest));

    // Missing bias is rejected at the physical index boundary before any
    // session can be constructed.
    ArtifactIndexInput missing_input;
    missing_input.artifacts = {package};
    physical.planes.pop_back();
    missing_input.tensors = {physical};
    const auto missing_index = ArtifactIndex::build(std::move(missing_input));
    CHECK(std::holds_alternative<CompatibilityReport>(missing_index));
    if (const auto* report = std::get_if<CompatibilityReport>(&missing_index)) {
        CHECK(report->code == CompatibilityError::IR_QUANTIZATION_UNSUPPORTED);
        CHECK_MSG(report->detail.find("plane set is not exact") != std::string::npos,
                  "unexpected missing-plane detail: %s", report->detail.c_str());
    }
}

int test_generic_metal_artifact(const char* path, uint32_t decode_count = 1, uint32_t maximum_context = 128) {
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return test_summary("test_canonical_metal");
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return test_summary("test_canonical_metal");
    const auto load_started = std::chrono::steady_clock::now();
    auto loaded = load_validated_gguf(std::get<PackageView>(view));
    const double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - load_started).count();
    if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
        fprintf(stderr, "generic canonical import: code=%u detail=%s\n", static_cast<unsigned>(report->code), report->detail.c_str());
    }
    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
    if (!std::holds_alternative<ValidatedPackage>(loaded)) return test_summary("test_canonical_metal");
    const auto create_started = std::chrono::steady_clock::now();
    auto created = create_qualification_canonical_metal_program(std::get<ValidatedPackage>(loaded).runtime_package(), maximum_context);
    const double create_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - create_started).count();
    if (const auto* report = std::get_if<CompatibilityReport>(&created)) {
        fprintf(stderr, "generic canonical Metal: code=%u layer=%u operator=%u detail=%s\n",
                static_cast<unsigned>(report->code), report->layer, report->operator_id, report->detail.c_str());
    }
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return test_summary("test_canonical_metal");
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    fprintf(stderr, "generic canonical Metal setup: load_ms=%.3f create_ms=%.3f maximum_context=%u\n",
            load_ms, create_ms, maximum_context);
    uint32_t token = std::get<ValidatedPackage>(loaded).runtime_package()->semantics().bos_id;
    for (uint32_t index = 0; index != decode_count; ++index) {
        const auto start = std::chrono::steady_clock::now();
        auto executed = program.decode(token);
        const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (const auto* report = std::get_if<CompatibilityReport>(&executed)) {
            fprintf(stderr, "generic canonical Metal token: code=%u operator=%u detail=%s\n",
                    static_cast<unsigned>(report->code), report->operator_id, report->detail.c_str());
        }
        CHECK(std::holds_alternative<CanonicalMetalOutput>(executed));
        if (const auto* output = std::get_if<CanonicalMetalOutput>(&executed)) {
            CHECK(output->completed);
            CHECK(output->command_buffers == 1);
            CHECK(output->logits.size() == std::get<ValidatedPackage>(loaded).runtime_package()->semantics().vocabulary_size);
            CHECK(output->projection_dispatches > 0);
            CHECK(output->q4k_projection_dispatches > 0);
            CHECK(output->q6k_projection_dispatches > 0);
            CHECK(output->requested_projection_source_bytes > 0);
            if (decode_count > 1) {
                const uint32_t top1 = static_cast<uint32_t>(std::max_element(
                    output->logits.begin(), output->logits.end()) - output->logits.begin());
                fprintf(stderr, "generic canonical Metal decode %u: input_token=%u output_top1=%u wall_ms=%.3f gpu_ms=%.3f cpu_wait_ms=%.3f ops=%u projections=%u q4k=%u q6k=%u requested_projection_source_bytes=%llu command_buffers=%u\n",
                        index + 1, token, top1, wall_ms, output->gpu_time_ms, output->cpu_wait_ms, output->operator_count,
                        output->projection_dispatches,
                        output->q4k_projection_dispatches, output->q6k_projection_dispatches,
                        static_cast<unsigned long long>(output->requested_projection_source_bytes),
                        output->command_buffers);
                token = top1;
            }
            if (output->profiled) {
                fprintf(stderr, "generic canonical Metal segments: qkv_ms=%.3f attention_ms=%.3f ffn_ms=%.3f final_ms=%.3f counter_samples=%u split_cb=%u\n",
                        output->qkv_gpu_ms, output->attention_gpu_ms, output->ffn_gpu_ms, output->final_gpu_ms,
                        output->counter_samples, output->split_command_buffer_profile);
            }
        }
    }
    return test_summary("test_canonical_metal");
}

int test_generic_metal_profile(const char* path) {
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return test_summary("test_canonical_metal");
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return test_summary("test_canonical_metal");
    auto loaded = load_validated_gguf(std::get<PackageView>(view));
    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
    if (!std::holds_alternative<ValidatedPackage>(loaded)) {
        const auto& report = std::get<CompatibilityReport>(loaded);
        std::fprintf(stderr, "generic canonical Metal load failed: code=%u detail=%s\n",
                     static_cast<unsigned>(report.code), report.detail.c_str());
        return test_summary("test_canonical_metal");
    }
    const auto package = std::get<ValidatedPackage>(loaded).runtime_package();
    const uint32_t token = package->semantics().bos_id;

    struct Sample {
        CanonicalMetalOutput output;
        double wall_ms = 0.0;
        uint32_t layers = 0;
    };
    auto run_sample = [&](bool profile, Sample& sample) {
        if (profile) setenv("LAPLACE_CANONICAL_METAL_PROFILE", "1", 1);
        else unsetenv("LAPLACE_CANONICAL_METAL_PROFILE");
        auto created = create_qualification_canonical_metal_program(package, 1);
        if (!std::holds_alternative<CanonicalMetalProgram>(created)) {
            unsetenv("LAPLACE_CANONICAL_METAL_PROFILE");
            CHECK(false);
            return false;
        }
        CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
        sample.layers = program.layer_count();
        const auto start = std::chrono::steady_clock::now();
        auto executed = program.decode(token);
        sample.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        unsetenv("LAPLACE_CANONICAL_METAL_PROFILE");
        CHECK(std::holds_alternative<CanonicalMetalOutput>(executed));
        if (!std::holds_alternative<CanonicalMetalOutput>(executed)) return false;
        sample.output = std::get<CanonicalMetalOutput>(std::move(executed));
        CHECK(sample.output.completed);
        CHECK(sample.output.command_buffers == 1);
        CHECK(sample.output.operator_count == package->semantics().operators.size());
        return true;
    };

    Sample warm;
    if (!run_sample(false, warm)) return test_summary("test_canonical_metal");
    std::array<Sample, 5> controls;
    std::array<Sample, 5> profiled;
    for (size_t index = 0; index != controls.size(); ++index) {
        if (!run_sample(false, controls[index]) || !run_sample(true, profiled[index]))
            return test_summary("test_canonical_metal");
        CHECK(controls[index].layers == profiled[index].layers);
        CHECK(controls[index].output.logits == profiled[index].output.logits);
        const uint32_t control_top1 = static_cast<uint32_t>(std::max_element(
            controls[index].output.logits.begin(), controls[index].output.logits.end()) -
            controls[index].output.logits.begin());
        const uint32_t profiled_top1 = static_cast<uint32_t>(std::max_element(
            profiled[index].output.logits.begin(), profiled[index].output.logits.end()) -
            profiled[index].output.logits.begin());
        CHECK(control_top1 == profiled_top1);
        CHECK(!profiled[index].output.split_command_buffer_profile);
        if (profiled[index].output.counter_samples) {
            CHECK(profiled[index].output.profiled);
            CHECK(profiled[index].output.counter_sample_count ==
                  3ull * profiled[index].layers + 2ull);
        } else {
            CHECK(!profiled[index].output.profiled);
            CHECK(profiled[index].output.counter_sample_count == 0);
        }
    }
    auto median = [](auto values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };
    std::array<double, 5> control_wall, profiled_wall, qkv, attention, ffn, final;
    for (size_t index = 0; index != controls.size(); ++index) {
        control_wall[index] = controls[index].wall_ms;
        profiled_wall[index] = profiled[index].wall_ms;
        qkv[index] = profiled[index].output.qkv_gpu_ms;
        attention[index] = profiled[index].output.attention_gpu_ms;
        ffn[index] = profiled[index].output.ffn_gpu_ms;
        final[index] = profiled[index].output.final_gpu_ms;
    }
    std::fprintf(stderr,
                 "generic canonical Metal profile: control_median_ms=%.3f profiled_median_ms=%.3f overhead_ms=%.3f samples=%llu qkv_median_ms=%.3f attention_median_ms=%.3f ffn_median_ms=%.3f final_median_ms=%.3f\n",
                 median(control_wall), median(profiled_wall), median(profiled_wall) - median(control_wall),
                 static_cast<unsigned long long>(profiled.front().output.counter_sample_count), median(qkv),
                 median(attention), median(ffn), median(final));
    return test_summary("test_canonical_metal");
}

int test_generic_metal_recurrent_preflight(const char* path) {
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return test_summary("test_canonical_metal");
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return test_summary("test_canonical_metal");
    auto loaded = load_validated_gguf(std::get<PackageView>(view));
    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
    if (!std::holds_alternative<ValidatedPackage>(loaded)) {
        const auto& report = std::get<CompatibilityReport>(loaded);
        std::fprintf(stderr, "generic canonical Metal load failed: code=%u detail=%s\n",
                     static_cast<unsigned>(report.code), report.detail.c_str());
        return test_summary("test_canonical_metal");
    }
    const auto package = std::get<ValidatedPackage>(loaded).runtime_package();
    auto created = create_qualification_canonical_metal_program(package, 128);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return test_summary("test_canonical_metal");
    const char* preflight = canonical_metal_first_recurrent_preflight_for_testing(
        std::get<CanonicalMetalProgram>(created));
    fprintf(stderr, "generic canonical Metal recurrent preflight: %s\n",
            preflight ? preflight : "OK");
    CHECK(preflight == nullptr);
    return test_summary("test_canonical_metal");
}

int test_generic_metal_error_propagation(const char* path) {
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return test_summary("test_canonical_metal");
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return test_summary("test_canonical_metal");
    auto loaded = load_validated_gguf(std::get<PackageView>(view));
    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
    if (!std::holds_alternative<ValidatedPackage>(loaded)) return test_summary("test_canonical_metal");
    const auto package = std::get<ValidatedPackage>(loaded).runtime_package();
    auto created = create_qualification_canonical_metal_program(package, 128);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return test_summary("test_canonical_metal");
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    canonical_metal_fail_after_completed_submission_for_testing(program);
    auto failed = program.decode(package->semantics().bos_id);
    CHECK(std::holds_alternative<CompatibilityReport>(failed));
    if (const auto* report = std::get_if<CompatibilityReport>(&failed)) {
        CHECK(report->code == CompatibilityError::RUNTIME_NUMERICAL_FAILURE);
        CHECK(report->detail.find("injected post-completion failure") != std::string::npos);
    }
    return test_summary("test_canonical_metal");
}

int test_generic_metal_derived_q2_ownership(const char* path, bool streamed_only = false,
                                             bool retained_source_only = false) {
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return test_summary("test_canonical_metal");
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return test_summary("test_canonical_metal");
    auto loaded = load_validated_gguf(std::get<PackageView>(view));
    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
    if (!std::holds_alternative<ValidatedPackage>(loaded)) return test_summary("test_canonical_metal");
    const auto package = std::get<ValidatedPackage>(loaded).runtime_package();

    std::vector<bool> executed_tensor(package->semantics().tensors.size(), false);
    for (const SemanticLayer& layer : package->semantics().layers) {
        if ((layer.flags & kSemanticLayerFlagSpeculative) != 0) continue;
        CHECK(layer.first_operator <= package->semantics().operators.size());
        CHECK(layer.operator_count <= package->semantics().operators.size() - layer.first_operator);
        for (uint32_t index = 0; index != layer.operator_count; ++index) {
            for (uint32_t tensor_id : package->semantics().operators[layer.first_operator + index].tensors) {
                CHECK(tensor_id < executed_tensor.size());
                if (tensor_id < executed_tensor.size()) executed_tensor[tensor_id] = true;
            }
        }
    }
    uint32_t expected = 0;
    uint64_t expected_source_bytes = 0;
    uint64_t expected_derived_bytes = 0;
    for (const SemanticTensor& tensor : package->semantics().tensors) {
        const bool selected_role = tensor.id < executed_tensor.size() && executed_tensor[tensor.id] &&
                                   (tensor.role == TensorRole::RecurrentBetaWeight ||
                                    tensor.role == TensorRole::FfnGateWeight);
        const bool eligible = selected_role && tensor.dimensions.size() == 2 && tensor.planes.size() == 1 &&
                              tensor.layout.kind == PhysicalLayoutKind::GgufBlocked &&
                              tensor.quantization.kind == QuantizationKind::BlockedAffine &&
                              tensor.layout.block_elements == 256 &&
                              (tensor.layout.block_bytes == 144 || tensor.layout.block_bytes == 210);
        if (!eligible) continue;
        const uint64_t elements = tensor.dimensions[0].constant_or_symbol *
                                  tensor.dimensions[1].constant_or_symbol;
        ++expected;
        expected_source_bytes += tensor.planes[0].length;
        expected_derived_bytes += elements / 256 * 84;
    }
    CHECK(expected > 0);

    const SemanticTensor* real_shape = nullptr;
    for (const SemanticTensor& tensor : package->semantics().tensors) {
        if (tensor.role == TensorRole::FfnGateWeight && tensor.dimensions.size() == 2 &&
            tensor.planes.size() == 1 && tensor.layout.kind == PhysicalLayoutKind::GgufBlocked &&
            tensor.quantization.kind == QuantizationKind::BlockedAffine &&
            tensor.layout.block_elements == 256 &&
            (tensor.layout.block_bytes == 144 || tensor.layout.block_bytes == 210)) {
            real_shape = &tensor;
            break;
        }
    }
    CHECK(real_shape != nullptr);
    if (!real_shape) return test_summary("test_canonical_metal");
    Tensor source_weight;
    CHECK(canonical_metal_tensor_view_for_testing(*package, *real_shape, source_weight));
    const int real_k = static_cast<int>(real_shape->dimensions[0].constant_or_symbol);
    const int real_n = static_cast<int>(real_shape->dimensions[1].constant_or_symbol);
    DerivedQ2KStorage real_derived;
    DerivedStorageError real_error = DerivedStorageError::None;
    CHECK(derive_q2_k_from_gguf(
        source_weight.type,
        std::span<const uint8_t>(static_cast<const uint8_t*>(source_weight.data), source_weight.nbytes()),
        static_cast<uint64_t>(real_k), static_cast<uint64_t>(real_n), &real_derived, &real_error));
    Tensor real_q2;
    real_q2.type = GGMLType::Q2_K;
    real_q2.n_dims = 2;
    real_q2.dims[0] = real_k;
    real_q2.dims[1] = real_n;
    real_q2.data = real_derived.bytes.data();
    std::vector<float> real_input(real_k), real_baseline(real_n), real_candidate(real_n);
    for (int index = 0; index != real_k; ++index)
        real_input[index] = std::sin(static_cast<float>(index + 1) * 0.013f);
    std::vector<double> real_baseline_gpu;
    std::vector<double> real_candidate_gpu;
    double warm_baseline = 0.0;
    double warm_candidate = 0.0;
    CHECK(streamed_only
        ? metal_test_q2k_streamed_ab(real_input.data(), real_q2,
                                     real_baseline.data(), real_candidate.data(),
                                     real_k, real_n, 1, &warm_baseline, &warm_candidate)
        : metal_test_q2k_two_row_ab(real_input.data(), real_q2,
                                    real_baseline.data(), real_candidate.data(),
                                    real_k, real_n, 1, &warm_baseline, &warm_candidate));
    metal_dispatch_metrics_reset();
    uint32_t candidate_wins = 0;
    for (int trial = 0; trial != 5; ++trial) {
        double baseline_ms = 0.0;
        double candidate_ms = 0.0;
        CHECK(streamed_only
            ? metal_test_q2k_streamed_ab(real_input.data(), real_q2,
                                         real_baseline.data(), real_candidate.data(),
                                         real_k, real_n, 1, &baseline_ms, &candidate_ms)
            : metal_test_q2k_two_row_ab(real_input.data(), real_q2,
                                        real_baseline.data(), real_candidate.data(),
                                        real_k, real_n, 1, &baseline_ms, &candidate_ms));
        real_baseline_gpu.push_back(baseline_ms);
        real_candidate_gpu.push_back(candidate_ms);
        if (candidate_ms < baseline_ms) ++candidate_wins;
    }
    float real_max_delta = 0.0f;
    for (int row = 0; row != real_n; ++row)
        real_max_delta = std::max(real_max_delta,
                                  std::abs(real_baseline[row] - real_candidate[row]));
    CHECK(real_max_delta == 0.0f);
    const auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        const size_t middle = values.size() / 2;
        return values.size() % 2 ? values[middle] :
            (values[middle - 1] + values[middle]) / 2.0;
    };
    const auto baseline_range = std::minmax_element(real_baseline_gpu.begin(), real_baseline_gpu.end());
    const auto candidate_range = std::minmax_element(real_candidate_gpu.begin(), real_candidate_gpu.end());
    const MetalDispatchMetrics direct_metrics = metal_dispatch_metrics();
    CHECK(direct_metrics.command_buffers == 10);
    uint32_t baseline_width = 0;
    uint32_t candidate_width = 0;
    CHECK(streamed_only
        ? metal_test_q2k_streamed_pipeline_widths(&baseline_width, &candidate_width)
        : metal_test_q2k_pipeline_widths(&baseline_width, &candidate_width));
    const double baseline_median = median(real_baseline_gpu);
    const double candidate_median = median(real_candidate_gpu);
    std::fprintf(stderr,
                 "generic real-shape Q2 %s A/B: K=%d N=%d source_type=%u q2_bytes=%zu warm_pairs=1 measured_pairs=5 candidate_wins=%u baseline_gpu_median_ms=%.3f baseline_gpu_range_ms=%.3f..%.3f candidate_gpu_median_ms=%.3f candidate_gpu_range_ms=%.3f..%.3f baseline_width=%u candidate_width=%u one_cb_per_sample=1 max_delta=%.9g\n",
                 streamed_only ? "streamed-loader" : "two-row",
                 real_k, real_n, static_cast<unsigned>(source_weight.type), real_derived.bytes.size(),
                 candidate_wins, baseline_median, *baseline_range.first, *baseline_range.second,
                 candidate_median, *candidate_range.first, *candidate_range.second,
                 baseline_width, candidate_width,
                 real_max_delta);
    if (streamed_only) return test_summary("test_canonical_metal");
    if (!retained_source_only && !(candidate_median < baseline_median && candidate_wins >= 4)) {
        std::fprintf(stderr,
                     "generic real-shape Q2 two-row rejected: no repeatable median win; full-model derived run skipped\n");
        return test_summary("test_canonical_metal");
    }

    CanonicalMetalOutput control_output;
    double control_wall_ms = 0.0;
    if (!retained_source_only) {
        auto control_created = create_qualification_canonical_metal_program(package, 128);
        CHECK(std::holds_alternative<CanonicalMetalProgram>(control_created));
        if (!std::holds_alternative<CanonicalMetalProgram>(control_created))
            return test_summary("test_canonical_metal");
        CanonicalMetalProgram control = std::get<CanonicalMetalProgram>(std::move(control_created));
        const auto started = std::chrono::steady_clock::now();
        auto executed = control.decode(package->semantics().bos_id);
        control_wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        CHECK(std::holds_alternative<CanonicalMetalOutput>(executed));
        if (!std::holds_alternative<CanonicalMetalOutput>(executed))
            return test_summary("test_canonical_metal");
        control_output = std::get<CanonicalMetalOutput>(std::move(executed));
    }

    CanonicalDerivedQ2KPolicy policy;
    policy.tensor_roles = {TensorRole::RecurrentBetaWeight, TensorRole::FfnGateWeight};
    policy.register_retained_source_ranges_for_testing = retained_source_only;
    const auto create_started = std::chrono::steady_clock::now();
    auto created = create_qualification_canonical_metal_program(package, 128, policy);
    const double create_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - create_started).count();
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) {
        if (const auto* report = std::get_if<CompatibilityReport>(&created))
            std::fprintf(stderr, "generic derived Q2 ownership: code=%u detail=%s\n",
                         static_cast<unsigned>(report->code), report->detail.c_str());
        return test_summary("test_canonical_metal");
    }
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    CHECK(program.derived_q2_storage_count() == expected);
    CHECK(program.derived_q2_storage_bytes() == expected_derived_bytes);
    CHECK(expected_derived_bytes < expected_source_bytes);
    if (retained_source_only) {
        CHECK(program.original_source_registered_bytes() < package->artifact_bytes(ArtifactId{0}).size());
        CHECK(program.derived_q2_registered_bytes() >= expected_derived_bytes);
        CHECK(program.retained_boundary_bytes() < program.original_source_registered_bytes());

        auto paired_control_created = create_qualification_canonical_metal_program(package, 128);
        CHECK(std::holds_alternative<CanonicalMetalProgram>(paired_control_created));
        if (!std::holds_alternative<CanonicalMetalProgram>(paired_control_created))
            return test_summary("test_canonical_metal");
        CanonicalMetalProgram paired_control =
            std::get<CanonicalMetalProgram>(std::move(paired_control_created));

        uint32_t thermal_before = metal_test_thermal_state();
        uint32_t stable_samples = 0;
        for (uint32_t attempt = 0; attempt != 30 && stable_samples != 5; ++attempt) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            const uint32_t current = metal_test_thermal_state();
            stable_samples = current == thermal_before ? stable_samples + 1 : 0;
            thermal_before = current;
        }
        CHECK(stable_samples == 5);
        if (stable_samples != 5) return test_summary("test_canonical_metal");

        const auto run_once = [&](CanonicalMetalProgram& target, CanonicalMetalOutput& output,
                                  double& wall_ms) {
            const auto started = std::chrono::steady_clock::now();
            auto result = target.decode(package->semantics().bos_id);
            wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            if (!std::holds_alternative<CanonicalMetalOutput>(result)) return false;
            output = std::get<CanonicalMetalOutput>(std::move(result));
            return output.completed && output.command_buffers == 1;
        };

        CanonicalMetalOutput warm_control, warm_derived;
        double warm_control_wall = 0.0, warm_derived_wall = 0.0;
        CHECK(run_once(paired_control, warm_control, warm_control_wall));
        CHECK(run_once(program, warm_derived, warm_derived_wall));

        std::vector<double> paired_control_gpu, paired_derived_gpu;
        std::vector<double> paired_control_wall, paired_derived_wall;
        size_t matched_top1 = 0;
        float paired_max_delta = 0.0f;
        CanonicalMetalOutput last_control, last_derived;
        for (uint32_t trial = 0; trial != 5; ++trial) {
            double dense_wall = 0.0, q2_wall = 0.0;
            bool dense_ok = false, q2_ok = false;
            if ((trial & 1u) == 0) {
                dense_ok = run_once(paired_control, last_control, dense_wall);
                q2_ok = run_once(program, last_derived, q2_wall);
            } else {
                q2_ok = run_once(program, last_derived, q2_wall);
                dense_ok = run_once(paired_control, last_control, dense_wall);
            }
            CHECK(dense_ok);
            CHECK(q2_ok);
            if (!dense_ok || !q2_ok) return test_summary("test_canonical_metal");
            CHECK(last_control.projection_dispatches == last_derived.projection_dispatches);
            CHECK(last_derived.requested_projection_source_bytes == last_control.requested_projection_source_bytes -
                                                           expected_source_bytes + expected_derived_bytes);
            paired_control_gpu.push_back(last_control.gpu_time_ms);
            paired_derived_gpu.push_back(last_derived.gpu_time_ms);
            paired_control_wall.push_back(dense_wall);
            paired_derived_wall.push_back(q2_wall);
            for (size_t index = 0; index != last_control.logits.size() &&
                                   index != last_derived.logits.size(); ++index) {
                paired_max_delta = std::max(paired_max_delta,
                    std::abs(last_control.logits[index] - last_derived.logits[index]));
            }
            const size_t dense_top1 = static_cast<size_t>(std::max_element(
                last_control.logits.begin(), last_control.logits.end()) - last_control.logits.begin());
            const size_t q2_top1 = static_cast<size_t>(std::max_element(
                last_derived.logits.begin(), last_derived.logits.end()) - last_derived.logits.begin());
            matched_top1 += dense_top1 == q2_top1;
        }
        const uint32_t thermal_after = metal_test_thermal_state();
        const auto control_gpu_range = std::minmax_element(paired_control_gpu.begin(), paired_control_gpu.end());
        const auto derived_gpu_range = std::minmax_element(paired_derived_gpu.begin(), paired_derived_gpu.end());
        const auto control_wall_range = std::minmax_element(paired_control_wall.begin(), paired_control_wall.end());
        const auto derived_wall_range = std::minmax_element(paired_derived_wall.begin(), paired_derived_wall.end());
        std::fprintf(stderr,
                     "generic derived Q2 retained-source paired: warm_pairs=1 measured_pairs=5 alternating_order=1 thermal_before=%u thermal_after=%u matched_top1=%zu/5 control_gpu_median_ms=%.3f control_gpu_range_ms=%.3f..%.3f derived_gpu_median_ms=%.3f derived_gpu_range_ms=%.3f..%.3f control_wall_median_ms=%.3f control_wall_range_ms=%.3f..%.3f derived_wall_median_ms=%.3f derived_wall_range_ms=%.3f..%.3f original_source_registered_bytes=%llu derived_bytes=%llu derived_registered_bytes=%llu retained_boundary_bytes=%llu control_peak_session_bytes=%llu derived_peak_session_bytes=%llu max_logit_delta=%.9g one_cb_per_sample=1\n",
                     thermal_before, thermal_after, matched_top1,
                     median(paired_control_gpu), *control_gpu_range.first, *control_gpu_range.second,
                     median(paired_derived_gpu), *derived_gpu_range.first, *derived_gpu_range.second,
                     median(paired_control_wall), *control_wall_range.first, *control_wall_range.second,
                     median(paired_derived_wall), *derived_wall_range.first, *derived_wall_range.second,
                     static_cast<unsigned long long>(program.original_source_registered_bytes()),
                     static_cast<unsigned long long>(program.derived_q2_storage_bytes()),
                     static_cast<unsigned long long>(program.derived_q2_registered_bytes()),
                     static_cast<unsigned long long>(program.retained_boundary_bytes()),
                     static_cast<unsigned long long>(last_control.peak_session_bytes),
                     static_cast<unsigned long long>(last_derived.peak_session_bytes),
                     paired_max_delta);
        return test_summary("test_canonical_metal");
    }

    metal_test_select_q2k_two_row_pipeline(!retained_source_only);
    const auto decode_started = std::chrono::steady_clock::now();
    auto executed = program.decode(package->semantics().bos_id);
    const double decode_wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - decode_started).count();
    metal_test_select_q2k_two_row_pipeline(false);
    CHECK(std::holds_alternative<CanonicalMetalOutput>(executed));
    if (const auto* report = std::get_if<CompatibilityReport>(&executed))
        std::fprintf(stderr, "generic derived Q2 decode: code=%u detail=%s\n",
                     static_cast<unsigned>(report->code), report->detail.c_str());
    if (const auto* output = std::get_if<CanonicalMetalOutput>(&executed)) {
        CHECK(output->completed);
        CHECK(output->command_buffers == 1);
        CHECK(control_output.command_buffers == 1);
        CHECK(output->projection_dispatches == control_output.projection_dispatches);
        CHECK(output->requested_projection_source_bytes == control_output.requested_projection_source_bytes -
                                                   expected_source_bytes + expected_derived_bytes);
        CHECK(output->logits.size() == package->semantics().vocabulary_size);
        for (float value : output->logits) CHECK(std::isfinite(value));
        float max_logit_delta = 0.0f;
        for (size_t index = 0; index != output->logits.size() && index != control_output.logits.size(); ++index)
            max_logit_delta = std::max(max_logit_delta, std::abs(output->logits[index] - control_output.logits[index]));
        const size_t control_top1 = static_cast<size_t>(std::max_element(
            control_output.logits.begin(), control_output.logits.end()) - control_output.logits.begin());
        const size_t derived_top1 = static_cast<size_t>(std::max_element(
            output->logits.begin(), output->logits.end()) - output->logits.begin());
        std::fprintf(stderr,
                     "generic derived Q2 %s metrics: source_bytes=%llu derived_bytes=%llu original_source_registered_bytes=%llu derived_registered_bytes=%llu retained_boundary_bytes=%llu create_ms=%.3f control_wall_ms=%.3f derived_wall_ms=%.3f control_gpu_ms=%.3f derived_gpu_ms=%.3f control_projection_bytes=%llu derived_projection_bytes=%llu peak_session_bytes=%llu max_logit_delta=%.9g control_top1=%zu derived_top1=%zu\n",
                     retained_source_only ? "retained-source" : "two-row",
                     static_cast<unsigned long long>(expected_source_bytes),
                     static_cast<unsigned long long>(expected_derived_bytes),
                     static_cast<unsigned long long>(program.original_source_registered_bytes()),
                     static_cast<unsigned long long>(program.derived_q2_registered_bytes()),
                     static_cast<unsigned long long>(program.retained_boundary_bytes()),
                     create_ms, control_wall_ms,
                     decode_wall_ms, control_output.gpu_time_ms, output->gpu_time_ms,
                     static_cast<unsigned long long>(control_output.requested_projection_source_bytes),
                     static_cast<unsigned long long>(output->requested_projection_source_bytes),
                     static_cast<unsigned long long>(output->peak_session_bytes), max_logit_delta,
                     control_top1, derived_top1);
    }
    std::fprintf(stderr, "generic derived Q2 ownership: tensors=%u bytes=%llu\n", expected,
                 static_cast<unsigned long long>(program.derived_q2_storage_bytes()));
    return test_summary("test_canonical_metal");
}

int test_generic_metal_iq2_xxs_atlas(const char* path) {
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return test_summary("test_canonical_metal");
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return test_summary("test_canonical_metal");
    auto loaded = load_validated_gguf(std::get<PackageView>(view));
    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
    if (!std::holds_alternative<ValidatedPackage>(loaded)) return test_summary("test_canonical_metal");
    const auto package = std::get<ValidatedPackage>(loaded).runtime_package();

    uint32_t expected_tensors = 0;
    uint64_t expected_source_bytes = 0;
    uint64_t expected_iq_bytes = 0;
    for (const SemanticTensor& tensor : package->semantics().tensors) {
        if (tensor.role != TensorRole::RecurrentBetaWeight || tensor.dimensions.size() != 2 ||
            tensor.planes.size() != 1 || tensor.layout.block_elements != 256 ||
            (tensor.layout.block_bytes != 144 && tensor.layout.block_bytes != 210)) continue;
        const uint64_t elements = tensor.dimensions[0].constant_or_symbol *
                                  tensor.dimensions[1].constant_or_symbol;
        ++expected_tensors;
        expected_source_bytes += tensor.planes[0].length;
        expected_iq_bytes += elements / 256 * 66;
    }
    CHECK(expected_tensors > 0);

    CanonicalDerivedIQ2XXSPolicy policy;
    policy.tensor_roles = {TensorRole::RecurrentBetaWeight};
    const auto started = std::chrono::steady_clock::now();
    auto created = create_qualification_canonical_metal_program(package, 128, {}, {}, policy);
    const double construction_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    if (const auto* report = std::get_if<CompatibilityReport>(&created))
        std::fprintf(stderr, "generic IQ2_XXS atlas: code=%u detail=%s\n",
                     static_cast<unsigned>(report->code), report->detail.c_str());
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created))
        return test_summary("test_canonical_metal");
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    CHECK(program.derived_iq2_xxs_atlas_count() == 1);
    CHECK(program.derived_iq2_xxs_tensor_count() == expected_tensors);
    CHECK(program.derived_iq2_xxs_source_bytes() == expected_source_bytes);
    CHECK(program.derived_iq2_xxs_storage_bytes() == expected_iq_bytes);
    CHECK(program.derived_iq2_xxs_registered_bytes() >= expected_iq_bytes);
    CHECK(program.derived_iq2_xxs_registered_bytes() - expected_iq_bytes < 16384 + 128 * expected_tensors);

    auto executed = program.decode(package->semantics().bos_id);
    CHECK(std::holds_alternative<CanonicalMetalOutput>(executed));
    if (const auto* output = std::get_if<CanonicalMetalOutput>(&executed)) {
        CHECK(output->completed);
        CHECK(output->command_buffers == 1);
        std::fprintf(stderr,
                     "generic IQ2_XXS atlas: tensors=%u source_bytes=%llu iq_bytes=%llu registered_bytes=%llu construction_ms=%.3f gpu_ms=%.3f one_cb=1\n",
                     expected_tensors, static_cast<unsigned long long>(expected_source_bytes),
                     static_cast<unsigned long long>(expected_iq_bytes),
                     static_cast<unsigned long long>(program.derived_iq2_xxs_registered_bytes()),
                     construction_ms, output->gpu_time_ms);
    }
    return test_summary("test_canonical_metal");
}

std::vector<CalibrationTarget> ffn_calibration_targets(const SemanticModel& model) {
    std::vector<CalibrationTarget> targets;
    for (const SemanticOperator& op : model.operators) {
        if (op.kind != OperatorKind::Linear || op.inputs.size() != 1) continue;
        const size_t operator_index = static_cast<size_t>(&op - model.operators.data());
        const bool in_executed_layer = std::any_of(model.layers.begin(), model.layers.end(),
            [&](const SemanticLayer& layer) {
                return (layer.flags & kSemanticLayerFlagSpeculative) == 0 &&
                       operator_index >= layer.first_operator &&
                       operator_index - layer.first_operator < layer.operator_count;
            });
        if (!in_executed_layer) continue;
        const auto input = std::find_if(model.values.begin(), model.values.end(), [&](const SemanticValue& value) {
            return value.id == op.inputs.front();
        });
        if (input == model.values.end() || input->dimensions.empty()) continue;
        for (uint32_t tensor_id : op.tensors) {
            const auto tensor = std::find_if(model.tensors.begin(), model.tensors.end(), [&](const SemanticTensor& value) {
                return value.id == tensor_id;
            });
            if (tensor == model.tensors.end() || tensor->dimensions.size() != 2) continue;
            if (tensor->role != TensorRole::FfnGateWeight && tensor->role != TensorRole::FfnUpWeight &&
                tensor->role != TensorRole::FfnDownWeight) continue;
            uint8_t physical_k_axis = 0xff;
            for (uint8_t axis = 0; axis != tensor->layout.rank; ++axis) {
                if (tensor->layout.axis_order[axis] == 0) {
                    physical_k_axis = axis;
                    break;
                }
            }
            if (physical_k_axis == 0xff || tensor->planes.empty() ||
                tensor->dimensions[0].kind != DimensionKind::Constant ||
                tensor->dimensions[0].constant_or_symbol > UINT32_MAX) continue;
            CalibrationTarget target;
            target.operator_id = op.id;
            target.input_value_id = input->id;
            target.weight_tensor_id = tensor->id;
            target.k_mapping.input_axis = static_cast<uint8_t>(input->dimensions.size() - 1);
            target.k_mapping.weight_physical_axis = physical_k_axis;
            target.k_mapping.weight_logical_axis = 0;
            target.k_mapping.width = static_cast<uint32_t>(tensor->dimensions[0].constant_or_symbol);
            target.k_mapping.block_elements = tensor->layout.kind == PhysicalLayoutKind::GgufBlocked
                ? tensor->layout.block_elements : 1;
            target.k_mapping.block_bytes = tensor->layout.kind == PhysicalLayoutKind::GgufBlocked
                ? tensor->layout.block_bytes
                : tensor->planes.front().storage_type == ScalarType::F32 ? 4 : 2;
            targets.push_back(target);
        }
    }
    return targets;
}

int test_generic_metal_derived_column_grouped_u2_quality(
    const char* path, const char* corpus_path, const char* cache_path) {
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts))
        return test_summary("test_canonical_metal");
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view))
        return test_summary("test_canonical_metal");
    auto loaded = load_validated_gguf(std::get<PackageView>(view));
    if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
        std::fprintf(stderr, "derived UInt2 quality import: code=%u detail=%s\n",
                     static_cast<unsigned>(report->code), report->detail.c_str());
    }
    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
    if (!std::holds_alternative<ValidatedPackage>(loaded))
        return test_summary("test_canonical_metal");
    const auto package = std::get<ValidatedPackage>(loaded).runtime_package();

    std::ifstream corpus_file(corpus_path, std::ios::binary);
    CHECK(corpus_file.is_open());
    if (!corpus_file.is_open()) return test_summary("test_canonical_metal");
    const std::string corpus((std::istreambuf_iterator<char>(corpus_file)),
                             std::istreambuf_iterator<char>());
    CHECK(!corpus_file.bad() && !corpus.empty());
    if (corpus_file.bad() || corpus.empty()) return test_summary("test_canonical_metal");

    GGUFContext tokenizer_context;
    CHECK(tokenizer_context.open(path));
    Tokenizer tokenizer;
    CHECK(tokenizer.init(tokenizer_context));
    std::vector<uint32_t> token_ids;
    if (tokenizer.bos_id() >= 0)
        token_ids.push_back(static_cast<uint32_t>(tokenizer.bos_id()));
    for (int token : tokenizer.encode(corpus)) {
        CHECK(token >= 0);
        if (token < 0) return test_summary("test_canonical_metal");
        token_ids.push_back(static_cast<uint32_t>(token));
    }
    constexpr size_t prefix_tokens = 17;
    CHECK(token_ids.size() >= prefix_tokens);
    if (token_ids.size() < prefix_tokens)
        return test_summary("test_canonical_metal");

    constexpr uint32_t window_limit = 128;
    std::vector<uint32_t> window_lengths;
    for (size_t offset = 0; offset < token_ids.size(); offset += window_limit)
        window_lengths.push_back(static_cast<uint32_t>(
            std::min<size_t>(window_limit, token_ids.size() - offset)));
    const auto targets = ffn_calibration_targets(package->semantics());
    CHECK(!targets.empty());
    const Sha256Digest target_digest = calibration_target_set_digest(targets);
    const Sha256Digest corpus_digest = calibration_bytes_digest(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(corpus.data()), corpus.size()));
    Sha256Digest token_digest;
    CHECK(calibration_token_window_digest(token_ids, window_lengths, token_digest));
    auto decoded_cache = load_calibration_cache(
        cache_path, package->artifact_digest(), package->fingerprint(), target_digest,
        corpus_digest, token_digest);
    if (const auto* report = std::get_if<CompatibilityReport>(&decoded_cache)) {
        std::fprintf(stderr, "derived UInt2 quality cache: code=%u detail=%s\n",
                     static_cast<unsigned>(report->code), report->detail.c_str());
    }
    CHECK(std::holds_alternative<CalibrationCacheBundle>(decoded_cache));
    if (!std::holds_alternative<CalibrationCacheBundle>(decoded_cache))
        return test_summary("test_canonical_metal");
    const auto& cache = std::get<CalibrationCacheBundle>(decoded_cache);
    CHECK(cache.records.size() == targets.size());
    if (cache.records.size() != targets.size())
        return test_summary("test_canonical_metal");

    std::vector<std::vector<float>> dense_logits;
    std::vector<uint32_t> dense_top1;
    dense_logits.reserve(prefix_tokens);
    dense_top1.reserve(prefix_tokens);
    double dense_gpu_ms = 0.0;
    uint64_t dense_requested_bytes = 0;
    {
        auto dense_created = create_qualification_canonical_metal_program(package, 128);
        if (const auto* report = std::get_if<CompatibilityReport>(&dense_created)) {
            std::fprintf(stderr, "derived UInt2 quality dense construction: code=%u detail=%s\n",
                         static_cast<unsigned>(report->code), report->detail.c_str());
        }
        CHECK(std::holds_alternative<CanonicalMetalProgram>(dense_created));
        if (!std::holds_alternative<CanonicalMetalProgram>(dense_created))
            return test_summary("test_canonical_metal");
        CanonicalMetalProgram dense =
            std::get<CanonicalMetalProgram>(std::move(dense_created));
        for (size_t step = 0; step != prefix_tokens; ++step) {
            auto result = dense.decode(token_ids[step]);
            CHECK(std::holds_alternative<CanonicalMetalOutput>(result));
            if (!std::holds_alternative<CanonicalMetalOutput>(result))
                return test_summary("test_canonical_metal");
            const auto& output = std::get<CanonicalMetalOutput>(result);
            CHECK(output.completed);
            CHECK(output.command_buffers == 1);
            CHECK(!output.logits.empty());
            for (float value : output.logits) CHECK(std::isfinite(value));
            const uint32_t top1 = static_cast<uint32_t>(
                std::max_element(output.logits.begin(), output.logits.end()) -
                output.logits.begin());
            dense_top1.push_back(top1);
            dense_logits.push_back(output.logits);
            dense_gpu_ms += output.gpu_time_ms;
            dense_requested_bytes = output.requested_projection_source_bytes;
        }
    }

    CanonicalDerivedColumnGroupedU2Policy policy;
    policy.tensor_roles = {TensorRole::FfnGateWeight, TensorRole::FfnUpWeight,
                           TensorRole::FfnDownWeight};
    policy.calibration_cache = &cache;
    const auto construction_start = std::chrono::steady_clock::now();
    auto derived_created = create_qualification_canonical_metal_program(
        package, 128, {}, {}, {}, {}, 1, policy);
    const double construction_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - construction_start).count();
    if (const auto* report = std::get_if<CompatibilityReport>(&derived_created)) {
        std::fprintf(stderr,
                     "derived UInt2 quality construction: code=%u layer=%u operator=%u detail=%s\n",
                     static_cast<unsigned>(report->code), report->layer,
                     report->operator_id, report->detail.c_str());
    }
    CHECK(std::holds_alternative<CanonicalMetalProgram>(derived_created));
    if (!std::holds_alternative<CanonicalMetalProgram>(derived_created))
        return test_summary("test_canonical_metal");
    CanonicalMetalProgram derived =
        std::get<CanonicalMetalProgram>(std::move(derived_created));

    size_t matched_top1 = 0;
    size_t first_mismatch = SIZE_MAX;
    uint32_t first_dense_top1 = 0;
    uint32_t first_derived_top1 = 0;
    float maximum_delta = 0.0f;
    double derived_gpu_ms = 0.0;
    uint64_t derived_requested_bytes = 0;
    uint64_t derived_peak_session_bytes = 0;
    uint64_t u2_dispatches = 0;
    bool one_cb_per_token = true;
    for (size_t step = 0; step != prefix_tokens; ++step) {
        auto result = derived.decode(token_ids[step]);
        CHECK(std::holds_alternative<CanonicalMetalOutput>(result));
        if (!std::holds_alternative<CanonicalMetalOutput>(result))
            return test_summary("test_canonical_metal");
        const auto& output = std::get<CanonicalMetalOutput>(result);
        CHECK(output.completed);
        CHECK(output.command_buffers == 1);
        one_cb_per_token = one_cb_per_token && output.command_buffers == 1;
        CHECK(output.logits.size() == dense_logits[step].size());
        if (output.logits.size() != dense_logits[step].size())
            return test_summary("test_canonical_metal");
        for (size_t index = 0; index != output.logits.size(); ++index) {
            CHECK(std::isfinite(output.logits[index]));
            maximum_delta = std::max(maximum_delta,
                                     std::abs(output.logits[index] - dense_logits[step][index]));
        }
        const uint32_t top1 = static_cast<uint32_t>(
            std::max_element(output.logits.begin(), output.logits.end()) -
            output.logits.begin());
        if (top1 == dense_top1[step]) {
            ++matched_top1;
        } else if (first_mismatch == SIZE_MAX) {
            first_mismatch = step;
            first_dense_top1 = dense_top1[step];
            first_derived_top1 = top1;
        }
        derived_gpu_ms += output.gpu_time_ms;
        derived_requested_bytes = output.requested_projection_source_bytes;
        derived_peak_session_bytes = std::max(derived_peak_session_bytes,
                                              output.peak_session_bytes);
        u2_dispatches += output.column_grouped_affine_u2_skip_projection_dispatches;
    }
    const CanonicalMetalResourceDiagnostics resources = derived.resource_diagnostics();
    CHECK(one_cb_per_token);
    CHECK(resources.implicit_weight_copies == 0);
    CHECK(derived.derived_column_grouped_u2_tensor_count() > 0);
    CHECK(derived.derived_column_grouped_u2_source_bytes() > 0);
    CHECK(derived.derived_column_grouped_u2_storage_bytes() > 0);
    CHECK(u2_dispatches > 0);
    std::fprintf(stderr,
                 "generic derived column-grouped UInt2 quality: prefix_tokens=%zu "
                 "construction_ms=%.3f derived_tensors=%u derived_source_bytes=%llu "
                 "derived_storage_bytes=%llu registered_source_bytes=%llu "
                 "excluded_replaced_bytes=%llu retained_boundary_bytes=%llu atlas_bytes=%llu "
                 "registration_overlap_bytes=%llu recommended_max_working_set_size=%llu "
                 "before_source_registration=%llu after_source_registration=%llu "
                 "after_atlas_registration=%llu after_session_construction=%llu "
                 "peak_session_bytes=%llu implicit_copies=%llu matched_top1=%zu/17 "
                 "first_mismatch=%s dense_top1=%u derived_top1=%u max_logit_delta=%.9g "
                 "one_cb_per_token=%u u2_dispatches=%llu selected_bytes=%llu "
                 "dense_requested_bytes=%llu derived_requested_bytes=%llu "
                 "dense_gpu_ms=%.3f derived_gpu_ms=%.3f\n",
                 prefix_tokens, construction_ms,
                 derived.derived_column_grouped_u2_tensor_count(),
                 static_cast<unsigned long long>(derived.derived_column_grouped_u2_source_bytes()),
                 static_cast<unsigned long long>(derived.derived_column_grouped_u2_storage_bytes()),
                 static_cast<unsigned long long>(resources.registered_source_bytes),
                 static_cast<unsigned long long>(resources.excluded_replaced_bytes),
                 static_cast<unsigned long long>(resources.retained_boundary_bytes),
                 static_cast<unsigned long long>(resources.atlas_bytes),
                 static_cast<unsigned long long>(resources.registration_overlap_bytes),
                 static_cast<unsigned long long>(resources.recommended_max_working_set_size),
                 static_cast<unsigned long long>(resources.before_source_registration),
                 static_cast<unsigned long long>(resources.after_source_registration),
                 static_cast<unsigned long long>(resources.after_atlas_registration),
                 static_cast<unsigned long long>(resources.after_session_construction),
                 static_cast<unsigned long long>(derived_peak_session_bytes),
                 static_cast<unsigned long long>(resources.implicit_weight_copies), matched_top1,
                 first_mismatch == SIZE_MAX ? "none" : std::to_string(first_mismatch).c_str(),
                 first_dense_top1, first_derived_top1, maximum_delta,
                 one_cb_per_token ? 1u : 0u, static_cast<unsigned long long>(u2_dispatches),
                 static_cast<unsigned long long>(derived_requested_bytes),
                 static_cast<unsigned long long>(dense_requested_bytes),
                 static_cast<unsigned long long>(derived_requested_bytes), dense_gpu_ms,
                 derived_gpu_ms);
    CHECK(matched_top1 == prefix_tokens);
    return test_summary("test_canonical_metal");
}

int test_generic_metal_ffn_calibration(const char* path) {
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return test_summary("test_canonical_metal");
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return test_summary("test_canonical_metal");
    auto loaded = load_validated_gguf(std::get<PackageView>(view));
    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
    if (!std::holds_alternative<ValidatedPackage>(loaded)) return test_summary("test_canonical_metal");
    const auto package = std::get<ValidatedPackage>(loaded).runtime_package();
    CanonicalCalibrationPolicy calibration;
    calibration.targets = ffn_calibration_targets(package->semantics());
    CHECK(!calibration.targets.empty());
    uint32_t excluded_speculative_targets = 0;
    const SemanticModel& semantics = package->semantics();
    for (const SemanticLayer& layer : semantics.layers) {
        if ((layer.flags & kSemanticLayerFlagSpeculative) == 0 ||
            layer.first_operator > semantics.operators.size() ||
            layer.operator_count > semantics.operators.size() - layer.first_operator) continue;
        for (uint32_t index = 0; index != layer.operator_count; ++index) {
            const SemanticOperator& op = semantics.operators[layer.first_operator + index];
            if (op.kind != OperatorKind::Linear) continue;
            for (uint32_t tensor_id : op.tensors) {
                const auto tensor = std::find_if(semantics.tensors.begin(), semantics.tensors.end(),
                    [&](const SemanticTensor& candidate) { return candidate.id == tensor_id; });
                if (tensor == semantics.tensors.end() ||
                    (tensor->role != TensorRole::FfnGateWeight &&
                     tensor->role != TensorRole::FfnUpWeight &&
                     tensor->role != TensorRole::FfnDownWeight)) continue;
                ++excluded_speculative_targets;
                std::fprintf(stderr,
                             "calibration excluded speculative target: layer=%u operator=%u tensor=%u role=%u\n",
                             layer.layer_index, op.id, tensor->id,
                             static_cast<unsigned>(tensor->role));
            }
        }
    }

    auto created = create_qualification_canonical_metal_program(package, 128, {}, {}, {}, calibration);
    if (const auto* report = std::get_if<CompatibilityReport>(&created))
        std::fprintf(stderr, "FFN calibration construction: code=%u detail=%s\n",
                     static_cast<unsigned>(report->code), report->detail.c_str());
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return test_summary("test_canonical_metal");
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    auto decoded = program.decode(package->semantics().bos_id);
    CHECK(std::holds_alternative<CanonicalMetalOutput>(decoded));
    if (!std::holds_alternative<CanonicalMetalOutput>(decoded)) return test_summary("test_canonical_metal");
    CHECK(std::get<CanonicalMetalOutput>(decoded).command_buffers == 1);

    std::vector<CalibrationRecord> records;
    CHECK(program.read_calibration(records));
    CHECK(records.size() == calibration.targets.size());
    for (const CalibrationRecord& record : records) {
        CHECK(record.sample_count == 1);
        CHECK(record.sum_squares.size() == record.target.k_mapping.width);
        bool any_positive = false;
        bool nonuniform = false;
        for (float value : record.sum_squares) {
            CHECK(std::isfinite(value));
            CHECK(value >= 0.0f);
            any_positive |= value > 0.0f;
            nonuniform |= value != record.sum_squares.front();
        }
        CHECK(any_positive);
        CHECK(nonuniform);
    }
    std::fprintf(stderr,
                 "generic FFN calibration: targets=%zu excluded_speculative=%u samples=1 nonuniform=1 one_cb=1 final_readback=1\n",
                 records.size(), excluded_speculative_targets);
    return test_summary("test_canonical_metal");
}

int test_generic_metal_ffn_calibration_cache(const char* path, const char* corpus_path,
                                             const char* cache_path) {
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return test_summary("test_canonical_metal");
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return test_summary("test_canonical_metal");
    auto loaded = load_validated_gguf(std::get<PackageView>(view));
    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
    if (!std::holds_alternative<ValidatedPackage>(loaded)) return test_summary("test_canonical_metal");
    const auto package = std::get<ValidatedPackage>(loaded).runtime_package();

    std::ifstream corpus_file(corpus_path, std::ios::binary);
    CHECK(corpus_file.is_open());
    if (!corpus_file.is_open()) return test_summary("test_canonical_metal");
    const std::string corpus((std::istreambuf_iterator<char>(corpus_file)),
                             std::istreambuf_iterator<char>());
    CHECK(!corpus_file.bad() && !corpus.empty());
    if (corpus_file.bad() || corpus.empty()) return test_summary("test_canonical_metal");
    GGUFContext tokenizer_context;
    CHECK(tokenizer_context.open(path));
    Tokenizer tokenizer;
    CHECK(tokenizer.init(tokenizer_context));
    std::vector<uint32_t> token_ids;
    if (tokenizer.bos_id() >= 0) token_ids.push_back(static_cast<uint32_t>(tokenizer.bos_id()));
    for (int token : tokenizer.encode(corpus)) {
        CHECK(token >= 0);
        if (token < 0) return test_summary("test_canonical_metal");
        token_ids.push_back(static_cast<uint32_t>(token));
    }
    constexpr size_t kCalibrationTokenLimit = 128;
    CHECK(token_ids.size() > 1 && token_ids.size() <= package->semantics().maximum_context);
    if (token_ids.size() <= 1 || token_ids.size() > package->semantics().maximum_context)
        return test_summary("test_canonical_metal");

    CanonicalCalibrationPolicy calibration;
    calibration.targets = ffn_calibration_targets(package->semantics());
    const Sha256Digest target_digest = calibration_target_set_digest(calibration.targets);
    const Sha256Digest corpus_digest = calibration_bytes_digest(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(corpus.data()), corpus.size()));
    std::vector<uint32_t> window_lengths;
    for (size_t offset = 0; offset != token_ids.size(); ) {
        const uint32_t length = static_cast<uint32_t>(
            std::min(kCalibrationTokenLimit, token_ids.size() - offset));
        window_lengths.push_back(length);
        offset += length;
    }
    Sha256Digest token_digest;
    CHECK(calibration_token_window_digest(token_ids, window_lengths, token_digest));

    auto cached = load_calibration_cache(cache_path, package->artifact_digest(),
                                         package->fingerprint(), target_digest, corpus_digest,
                                         token_digest);
    if (const auto* cache = std::get_if<CalibrationCacheBundle>(&cached)) {
        CHECK(cache->records.size() == calibration.targets.size());
        const uint32_t sample_count = cache->records.empty() ? 0 : cache->records.front().sample_count;
        size_t zero_lanes = 0;
        bool stats_valid = sample_count != 0;
        for (const CalibrationRecord& record : cache->records) {
            std::vector<float> normalized;
            stats_valid &= record.sample_count == sample_count &&
                           normalize_calibration_record(record, normalized) && !normalized.empty();
            if (normalized.empty()) continue;
            const auto [minimum, maximum] = std::minmax_element(normalized.begin(), normalized.end());
            double sum = 0.0;
            size_t record_zero_lanes = 0;
            for (float value : normalized) {
                sum += value;
                record_zero_lanes += value == 0.0f;
            }
            zero_lanes += record_zero_lanes;
            std::fprintf(stderr,
                         "calibration record stats: operator=%u tensor=%u samples=%u lanes=%zu zero_lanes=%zu min=%.9g max=%.9g mean=%.9g\n",
                         record.target.operator_id, record.target.weight_tensor_id,
                         record.sample_count, normalized.size(), record_zero_lanes,
                         *minimum, *maximum, sum / normalized.size());
        }
        CHECK(stats_valid);
        std::fprintf(stderr,
                     "generic FFN calibration cache hit: records=%zu samples=%u windows=%zu zero_lanes=%zu artifact=%s semantic=%s corpus=%s tokens=%s\n",
                     cache->records.size(), sample_count, window_lengths.size(), zero_lanes,
                     cache->artifact_digest.hex().c_str(), cache->semantic_fingerprint.hex().c_str(),
                     cache->corpus_digest.hex().c_str(), cache->token_digest.hex().c_str());
        return test_summary("test_canonical_metal");
    }
    std::fprintf(stderr,
                 "generic FFN calibration cache miss: corpus_tokens=%zu windows=%zu window_limit=%zu\n",
                 token_ids.size(), window_lengths.size(), kCalibrationTokenLimit);
    std::vector<CalibrationRecord> records;
    size_t token_offset = 0;
    size_t readbacks = 0;
    for (size_t window_index = 0; window_index != window_lengths.size(); ++window_index) {
        const uint32_t window_length = window_lengths[window_index];
        auto created = create_qualification_canonical_metal_program(package, window_length, {}, {}, {}, calibration);
        CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
        if (!std::holds_alternative<CanonicalMetalProgram>(created)) return test_summary("test_canonical_metal");
        CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
        for (uint32_t local_index = 0; local_index != window_length; ++local_index) {
            const uint32_t token = token_ids[token_offset + local_index];
            auto accumulated = program.accumulate_calibration(token);
            CHECK(std::holds_alternative<CanonicalMetalOutput>(accumulated));
            if (!std::holds_alternative<CanonicalMetalOutput>(accumulated)) {
                const auto& report = std::get<CompatibilityReport>(accumulated);
                std::fprintf(stderr,
                             "generic FFN calibration failure: window=%zu local_index=%u token_index=%zu token_id=%u position=%u code=%u operator=%u detail=%s\n",
                             window_index, local_index, token_offset + local_index, token,
                             program.position(), static_cast<unsigned>(report.code),
                             report.operator_id, report.detail.c_str());
                return test_summary("test_canonical_metal");
            }
            CHECK(std::get<CanonicalMetalOutput>(accumulated).command_buffers == 1);
            CHECK(std::get<CanonicalMetalOutput>(accumulated).logits.empty());
        }
        std::vector<CalibrationRecord> window_records;
        CHECK(program.read_calibration(window_records));
        for (const CalibrationRecord& record : window_records)
            CHECK(record.sample_count == window_length);
        CHECK(merge_calibration_records(records, window_records));
        ++readbacks;
        token_offset += window_length;
    }
    CHECK(token_offset == token_ids.size());
    CHECK(records.size() == calibration.targets.size());
    for (const CalibrationRecord& record : records)
        CHECK(record.sample_count == token_ids.size());

    CalibrationCacheBundle cache;
    cache.artifact_digest = package->artifact_digest();
    cache.semantic_fingerprint = package->fingerprint();
    cache.target_set_digest = target_digest;
    cache.corpus_digest = corpus_digest;
    cache.token_digest = token_digest;
    cache.records = records;
    CompatibilityReport write_error;
    CHECK(write_calibration_cache_atomic(cache_path, cache, &write_error));
    auto reloaded = load_calibration_cache(cache_path, cache.artifact_digest,
                                           cache.semantic_fingerprint, cache.target_set_digest,
                                           cache.corpus_digest, cache.token_digest);
    CHECK(std::holds_alternative<CalibrationCacheBundle>(reloaded));
    if (const auto* roundtrip = std::get_if<CalibrationCacheBundle>(&reloaded)) {
        CHECK(roundtrip->records.size() == records.size());
        for (size_t index = 0; index != records.size(); ++index) {
            CHECK(roundtrip->records[index].target == records[index].target);
            CHECK(roundtrip->records[index].sample_count == records[index].sample_count);
            CHECK(roundtrip->records[index].sum_squares == records[index].sum_squares);
        }
    }
    std::fprintf(stderr,
                 "generic FFN calibration cache green: records=%zu samples=%zu windows=%zu readbacks=%zu one_cb_per_token=1 artifact=%s semantic=%s corpus=%s tokens=%s\n",
                 records.size(), token_ids.size(), window_lengths.size(), readbacks,
                 cache.artifact_digest.hex().c_str(),
                 cache.semantic_fingerprint.hex().c_str(), cache.corpus_digest.hex().c_str(),
                 cache.token_digest.hex().c_str());
    return test_summary("test_canonical_metal");
}

enum class CalibratedSparseMode {
    Fixed48,
    VariableWidthProbe,
    MarginalAllocate,
    SensitivityAllocate34,
    SequentialAllocate34,
};

int test_generic_metal_sparse_ffn_calibrated(
    const char* path, const char* corpus_path, const char* cache_path,
    CalibratedSparseMode mode = CalibratedSparseMode::Fixed48,
    uint32_t target_average = 34) {
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return test_summary("test_canonical_metal");
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return test_summary("test_canonical_metal");
    auto loaded = load_validated_gguf(std::get<PackageView>(view));
    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
    if (!std::holds_alternative<ValidatedPackage>(loaded)) return test_summary("test_canonical_metal");
    const auto package = std::get<ValidatedPackage>(loaded).runtime_package();

    std::ifstream corpus_file(corpus_path, std::ios::binary);
    CHECK(corpus_file.is_open());
    if (!corpus_file.is_open()) return test_summary("test_canonical_metal");
    const std::string corpus((std::istreambuf_iterator<char>(corpus_file)),
                             std::istreambuf_iterator<char>());
    CHECK(!corpus_file.bad() && !corpus.empty());
    GGUFContext tokenizer_context;
    CHECK(tokenizer_context.open(path));
    Tokenizer tokenizer;
    CHECK(tokenizer.init(tokenizer_context));
    std::vector<uint32_t> token_ids;
    if (tokenizer.bos_id() >= 0) token_ids.push_back(static_cast<uint32_t>(tokenizer.bos_id()));
    for (int token : tokenizer.encode(corpus)) {
        CHECK(token >= 0);
        if (token < 0) return test_summary("test_canonical_metal");
        token_ids.push_back(static_cast<uint32_t>(token));
    }
    constexpr uint32_t window_limit = 128;
    std::vector<uint32_t> window_lengths;
    for (size_t offset = 0; offset < token_ids.size(); offset += window_limit)
        window_lengths.push_back(static_cast<uint32_t>(
            std::min<size_t>(window_limit, token_ids.size() - offset)));
    const auto targets = ffn_calibration_targets(package->semantics());
    const Sha256Digest target_digest = calibration_target_set_digest(targets);
    const Sha256Digest corpus_digest = calibration_bytes_digest(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(corpus.data()), corpus.size()));
    Sha256Digest token_digest;
    CHECK(calibration_token_window_digest(token_ids, window_lengths, token_digest));
    auto decoded = load_calibration_cache(cache_path, package->artifact_digest(),
                                          package->fingerprint(), target_digest,
                                          corpus_digest, token_digest);
    CHECK(std::holds_alternative<CalibrationCacheBundle>(decoded));
    if (!std::holds_alternative<CalibrationCacheBundle>(decoded)) {
        if (const auto* report = std::get_if<CompatibilityReport>(&decoded))
            std::fprintf(stderr, "calibrated sparse FFN cache: code=%u detail=%s\n",
                         static_cast<unsigned>(report->code), report->detail.c_str());
        return test_summary("test_canonical_metal");
    }
    const auto& cache = std::get<CalibrationCacheBundle>(decoded);

    struct RankedLayer {
        uint32_t layer_index = 0;
        uint32_t total_blocks = 0;
        uint64_t source_bytes = 0;
        std::vector<uint32_t> block_ids;
        std::vector<double> block_scores;
    };
    std::vector<RankedLayer> rankings;
    CanonicalSparseFfnPolicy policy;
    constexpr uint32_t selected_blocks = 48;
    uint64_t selected_block_total = 0;
    uint64_t expected_requested_bytes = 0;
    uint64_t expected_source_bytes = 0;
    for (const SemanticLayer& layer : package->semantics().layers) {
        if ((layer.flags & kSemanticLayerFlagSpeculative) != 0) continue;
        const SemanticTensor* down = nullptr;
        const SemanticTensor* gate = nullptr;
        const SemanticTensor* up = nullptr;
        for (uint32_t index = 0; index != layer.operator_count; ++index) {
            const SemanticOperator& op = package->semantics().operators[layer.first_operator + index];
            if (op.kind != OperatorKind::Linear || op.tensors.size() != 1) continue;
            const SemanticTensor& tensor = package->semantics().tensors[op.tensors.front()];
            if (tensor.role == TensorRole::FfnGateWeight) gate = &tensor;
            if (tensor.role == TensorRole::FfnUpWeight) up = &tensor;
            if (tensor.role == TensorRole::FfnDownWeight) down = &tensor;
        }
        CHECK(gate && up && down);
        if (!gate || !up || !down) return test_summary("test_canonical_metal");
        const CalibrationRecord* record = nullptr;
        for (const CalibrationRecord& candidate : cache.records) {
            if (candidate.target.weight_tensor_id != down->id) continue;
            CHECK(record == nullptr);
            record = &candidate;
        }
        CHECK(record != nullptr);
        if (!record) return test_summary("test_canonical_metal");
        std::vector<float> importance;
        CHECK(normalize_calibration_record(*record, importance));
        CHECK(importance.size() == gate->dimensions[1].constant_or_symbol);
        if (importance.size() != gate->dimensions[1].constant_or_symbol ||
            importance.size() % 256 != 0) return test_summary("test_canonical_metal");
        const uint32_t total_blocks = static_cast<uint32_t>(importance.size() / 256);
        const uint32_t layer_selected_blocks =
            mode == CalibratedSparseMode::VariableWidthProbe
                ? (policy.layer_masks.size() % 2 == 0 ? 68u : 55u)
                : ((mode == CalibratedSparseMode::MarginalAllocate ||
                    mode == CalibratedSparseMode::SensitivityAllocate34)
                       ? 68u
                       : selected_blocks);
        CHECK(total_blocks >= layer_selected_blocks);
        std::vector<double> scores(total_blocks);
        for (uint32_t block = 0; block != total_blocks; ++block) {
            for (uint32_t lane = 0; lane != 256; ++lane)
                scores[block] += importance[static_cast<size_t>(block) * 256 + lane];
            CHECK(std::isfinite(scores[block]));
        }
        std::vector<uint32_t> ids(total_blocks);
        std::iota(ids.begin(), ids.end(), 0u);
        std::stable_sort(ids.begin(), ids.end(), [&](uint32_t left, uint32_t right) {
            return scores[left] != scores[right] ? scores[left] > scores[right] : left < right;
        });
        const uint64_t source_bytes = gate->planes.front().length + up->planes.front().length +
                                      down->planes.front().length;
        rankings.push_back({layer.layer_index, total_blocks, source_bytes, ids, scores});
        ids.resize(layer_selected_blocks);
        std::sort(ids.begin(), ids.end());
        policy.layer_masks.push_back({layer.layer_index, std::move(ids)});
        selected_block_total += layer_selected_blocks;
        expected_source_bytes += source_bytes;
        expected_requested_bytes += source_bytes * layer_selected_blocks / total_blocks;
    }
    CHECK(!policy.layer_masks.empty());
    if (mode == CalibratedSparseMode::VariableWidthProbe) {
        auto created = create_qualification_canonical_metal_program(package, 1, {}, policy);
        CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
        if (!std::holds_alternative<CanonicalMetalProgram>(created)) {
            if (const auto* report = std::get_if<CompatibilityReport>(&created))
                std::fprintf(stderr, "variable sparse FFN probe: code=%u detail=%s\n",
                             static_cast<unsigned>(report->code), report->detail.c_str());
            return test_summary("test_canonical_metal");
        }
        CanonicalMetalProgram program =
            std::get<CanonicalMetalProgram>(std::move(created));
        auto output = program.decode(package->semantics().bos_id);
        CHECK(std::holds_alternative<CanonicalMetalOutput>(output));
        if (const auto* result = std::get_if<CanonicalMetalOutput>(&output)) {
            CHECK(result->completed && result->command_buffers == 1);
            CHECK(program.sparse_ffn_worklist_bytes() ==
                  selected_block_total * sizeof(uint32_t));
            std::fprintf(stderr,
                         "variable sparse FFN probe: layers=%zu total_blocks=%llu average=%.3f worklist_bytes=%llu one_cb=1\n",
                         policy.layer_masks.size(),
                         static_cast<unsigned long long>(selected_block_total),
                         static_cast<double>(selected_block_total) / policy.layer_masks.size(),
                         static_cast<unsigned long long>(program.sparse_ffn_worklist_bytes()));
        }
        return test_summary("test_canonical_metal");
    }
    constexpr std::array<uint32_t, 7> retention_levels = {20, 27, 34, 41, 48, 55, 68};
    std::array<uint32_t, retention_levels.size()> budget_histogram{};
    auto policy_for_counts = [&](const std::vector<uint32_t>& counts) {
        CanonicalSparseFfnPolicy result;
        if (counts.size() != rankings.size()) return result;
        for (size_t layer = 0; layer != rankings.size(); ++layer) {
            if (counts[layer] == 0 || counts[layer] > rankings[layer].block_ids.size()) {
                result.layer_masks.clear();
                return result;
            }
            std::vector<uint32_t> ids(rankings[layer].block_ids.begin(),
                                      rankings[layer].block_ids.begin() + counts[layer]);
            std::sort(ids.begin(), ids.end());
            result.layer_masks.push_back({rankings[layer].layer_index, std::move(ids)});
        }
        return result;
    };
    std::array<uint32_t, 69> exact_budget_histogram{};
    if (mode == CalibratedSparseMode::MarginalAllocate) {
        CHECK(target_average >= 32 && target_average <= 34);
        if (target_average < 32 || target_average > 34)
            return test_summary("test_canonical_metal");
        std::vector<uint32_t> allocated_counts(rankings.size(), 32);
        std::vector<double> score_totals(rankings.size());
        for (size_t layer = 0; layer != rankings.size(); ++layer) {
            CHECK(rankings[layer].total_blocks >= 48);
            for (double score : rankings[layer].block_scores) score_totals[layer] += score;
            CHECK(std::isfinite(score_totals[layer]) && score_totals[layer] > 0.0);
            if (rankings[layer].total_blocks < 48 || !std::isfinite(score_totals[layer]) ||
                score_totals[layer] <= 0.0)
                return test_summary("test_canonical_metal");
        }
        const uint64_t target_total =
            static_cast<uint64_t>(target_average) * rankings.size();
        selected_block_total = 32u * rankings.size();
        while (selected_block_total < target_total) {
            bool found = false;
            size_t best_layer = 0;
            uint32_t best_block = 0;
            double best_marginal = 0.0;
            for (size_t layer = 0; layer != rankings.size(); ++layer) {
                if (allocated_counts[layer] >= 48) continue;
                const uint32_t block =
                    rankings[layer].block_ids[allocated_counts[layer]];
                const double marginal =
                    rankings[layer].block_scores[block] / score_totals[layer];
                if (!found || marginal > best_marginal ||
                    (marginal == best_marginal &&
                     (rankings[layer].layer_index < rankings[best_layer].layer_index ||
                      (rankings[layer].layer_index == rankings[best_layer].layer_index &&
                       block < best_block)))) {
                    found = true;
                    best_layer = layer;
                    best_block = block;
                    best_marginal = marginal;
                }
            }
            CHECK(found);
            if (!found) return test_summary("test_canonical_metal");
            ++allocated_counts[best_layer];
            ++selected_block_total;
        }
        CHECK(selected_block_total == target_total);
        policy = policy_for_counts(allocated_counts);
        CHECK(policy.layer_masks.size() == rankings.size());
        expected_source_bytes = 0;
        expected_requested_bytes = 0;
        for (size_t layer = 0; layer != rankings.size(); ++layer) {
            ++exact_budget_histogram[allocated_counts[layer]];
            expected_source_bytes += rankings[layer].source_bytes;
            expected_requested_bytes += rankings[layer].source_bytes *
                                        allocated_counts[layer] /
                                        rankings[layer].total_blocks;
        }
        std::fprintf(stderr,
                     "sparse marginal allocator target_average=%u total_blocks=%llu "
                     "hist32=%u hist33=%u hist34=%u hist35=%u hist36=%u hist37=%u "
                     "hist38=%u hist39=%u hist40=%u hist41=%u hist42=%u hist43=%u "
                     "hist44=%u hist45=%u hist46=%u hist47=%u hist48=%u\n",
                     target_average,
                     static_cast<unsigned long long>(selected_block_total),
                     exact_budget_histogram[32], exact_budget_histogram[33],
                     exact_budget_histogram[34], exact_budget_histogram[35],
                     exact_budget_histogram[36], exact_budget_histogram[37],
                     exact_budget_histogram[38], exact_budget_histogram[39],
                     exact_budget_histogram[40], exact_budget_histogram[41],
                     exact_budget_histogram[42], exact_budget_histogram[43],
                     exact_budget_histogram[44], exact_budget_histogram[45],
                     exact_budget_histogram[46], exact_budget_histogram[47],
                     exact_budget_histogram[48]);
    } else if (mode == CalibratedSparseMode::SensitivityAllocate34) {
        CHECK(token_ids.size() >= 3);
        if (token_ids.size() < 3) return test_summary("test_canonical_metal");
        const std::array<uint32_t, 2> calibration_inputs = {
            package->semantics().bos_id, token_ids[1]};
        std::vector<std::vector<float>> reference_logits;
        std::vector<uint32_t> reference_top1;
        {
            auto baseline_created = create_qualification_canonical_metal_program(
                package, static_cast<uint32_t>(calibration_inputs.size()), {}, policy);
            CHECK(std::holds_alternative<CanonicalMetalProgram>(baseline_created));
            if (!std::holds_alternative<CanonicalMetalProgram>(baseline_created))
                return test_summary("test_canonical_metal");
            CanonicalMetalProgram baseline =
                std::get<CanonicalMetalProgram>(std::move(baseline_created));
            for (uint32_t token : calibration_inputs) {
                auto result = baseline.decode(token);
                CHECK(std::holds_alternative<CanonicalMetalOutput>(result));
                if (!std::holds_alternative<CanonicalMetalOutput>(result))
                    return test_summary("test_canonical_metal");
                CanonicalMetalOutput output =
                    std::get<CanonicalMetalOutput>(std::move(result));
                CHECK(output.completed && output.command_buffers == 1);
                reference_top1.push_back(static_cast<uint32_t>(
                    std::max_element(output.logits.begin(), output.logits.end()) -
                    output.logits.begin()));
                reference_logits.push_back(std::move(output.logits));
            }
        }

        struct Sensitivity {
            double mean_squared_error = 0.0;
            uint32_t top1_mismatches = 0;
        };
        std::vector<std::array<Sensitivity, retention_levels.size()>> sensitivity(
            rankings.size());
        std::vector<uint32_t> isolated_counts(rankings.size(), retention_levels.back());
        for (size_t layer = 0; layer != rankings.size(); ++layer) {
            for (size_t level = 0; level + 1 != retention_levels.size(); ++level) {
                isolated_counts[layer] = retention_levels[level];
                CanonicalSparseFfnPolicy candidate_policy = policy_for_counts(isolated_counts);
                CHECK(candidate_policy.layer_masks.size() == rankings.size());
                auto candidate_created = create_qualification_canonical_metal_program(
                    package, static_cast<uint32_t>(calibration_inputs.size()), {},
                    candidate_policy);
                CHECK(std::holds_alternative<CanonicalMetalProgram>(candidate_created));
                if (!std::holds_alternative<CanonicalMetalProgram>(candidate_created))
                    return test_summary("test_canonical_metal");
                CanonicalMetalProgram candidate =
                    std::get<CanonicalMetalProgram>(std::move(candidate_created));
                double squared_error = 0.0;
                uint64_t compared_values = 0;
                uint32_t mismatches = 0;
                for (size_t step = 0; step != calibration_inputs.size(); ++step) {
                    auto result = candidate.decode(calibration_inputs[step]);
                    CHECK(std::holds_alternative<CanonicalMetalOutput>(result));
                    if (!std::holds_alternative<CanonicalMetalOutput>(result))
                        return test_summary("test_canonical_metal");
                    const auto& output = std::get<CanonicalMetalOutput>(result);
                    CHECK(output.completed && output.command_buffers == 1);
                    CHECK(output.logits.size() == reference_logits[step].size());
                    const uint32_t top1 = static_cast<uint32_t>(
                        std::max_element(output.logits.begin(), output.logits.end()) -
                        output.logits.begin());
                    mismatches += top1 != reference_top1[step];
                    for (size_t index = 0; index != output.logits.size(); ++index) {
                        if (!std::isfinite(output.logits[index])) {
                            CHECK(false);
                            return test_summary("test_canonical_metal");
                        }
                        const double difference = static_cast<double>(output.logits[index]) -
                                                  reference_logits[step][index];
                        squared_error += difference * difference;
                    }
                    compared_values += output.logits.size();
                }
                sensitivity[layer][level] = {
                    compared_values == 0 ? 0.0
                                         : squared_error / static_cast<double>(compared_values),
                    mismatches,
                };
                isolated_counts[layer] = retention_levels.back();
            }
            sensitivity[layer].back() = {};
            std::fprintf(stderr,
                         "sparse sensitivity layer=%u e20=%.9g/%u e27=%.9g/%u "
                         "e34=%.9g/%u e41=%.9g/%u e48=%.9g/%u e55=%.9g/%u e68=0/0\n",
                         rankings[layer].layer_index,
                         sensitivity[layer][0].mean_squared_error,
                         sensitivity[layer][0].top1_mismatches,
                         sensitivity[layer][1].mean_squared_error,
                         sensitivity[layer][1].top1_mismatches,
                         sensitivity[layer][2].mean_squared_error,
                         sensitivity[layer][2].top1_mismatches,
                         sensitivity[layer][3].mean_squared_error,
                         sensitivity[layer][3].top1_mismatches,
                         sensitivity[layer][4].mean_squared_error,
                         sensitivity[layer][4].top1_mismatches,
                         sensitivity[layer][5].mean_squared_error,
                         sensitivity[layer][5].top1_mismatches);
        }

        std::vector<size_t> chosen_level(rankings.size(), retention_levels.size() - 1);
        selected_block_total = rankings.size() * retention_levels.back();
        const uint64_t block_budget = rankings.size() * 34u;
        while (selected_block_total > block_budget) {
            bool found = false;
            size_t best_layer = 0;
            uint32_t best_top1_cost = UINT32_MAX;
            double best_error_cost = 0.0;
            for (size_t layer = 0; layer != rankings.size(); ++layer) {
                const size_t current = chosen_level[layer];
                if (current == 0) continue;
                const size_t lower = current - 1;
                const uint32_t removed = retention_levels[current] - retention_levels[lower];
                const uint32_t top1_cost =
                    sensitivity[layer][lower].top1_mismatches >
                            sensitivity[layer][current].top1_mismatches
                        ? sensitivity[layer][lower].top1_mismatches -
                              sensitivity[layer][current].top1_mismatches
                        : 0;
                const double error_delta = std::max(
                    0.0, sensitivity[layer][lower].mean_squared_error -
                             sensitivity[layer][current].mean_squared_error);
                const double error_cost = error_delta / removed;
                if (!found || top1_cost < best_top1_cost ||
                    (top1_cost == best_top1_cost &&
                     (error_cost < best_error_cost ||
                      (error_cost == best_error_cost &&
                       rankings[layer].layer_index < rankings[best_layer].layer_index)))) {
                    found = true;
                    best_layer = layer;
                    best_top1_cost = top1_cost;
                    best_error_cost = error_cost;
                }
            }
            CHECK(found);
            if (!found) return test_summary("test_canonical_metal");
            const size_t current = chosen_level[best_layer];
            const size_t lower = current - 1;
            selected_block_total -= retention_levels[current] - retention_levels[lower];
            chosen_level[best_layer] = lower;
        }
        CHECK(selected_block_total <= block_budget);
        std::vector<uint32_t> allocated_counts(rankings.size());
        for (size_t layer = 0; layer != rankings.size(); ++layer) {
            allocated_counts[layer] = retention_levels[chosen_level[layer]];
            ++budget_histogram[chosen_level[layer]];
        }
        policy = policy_for_counts(allocated_counts);
        CHECK(policy.layer_masks.size() == rankings.size());
        expected_source_bytes = 0;
        expected_requested_bytes = 0;
        for (size_t layer = 0; layer != rankings.size(); ++layer) {
            expected_source_bytes += rankings[layer].source_bytes;
            expected_requested_bytes += rankings[layer].source_bytes *
                                        allocated_counts[layer] /
                                        rankings[layer].total_blocks;
        }
        std::fprintf(stderr,
                     "sparse allocator calibration_tokens=%u,%u total_blocks=%llu "
                     "average=%.3f hist20=%u hist27=%u hist34=%u hist41=%u "
                     "hist48=%u hist55=%u hist68=%u\n",
                     calibration_inputs[0], calibration_inputs[1],
                     static_cast<unsigned long long>(selected_block_total),
                     static_cast<double>(selected_block_total) / rankings.size(),
                     budget_histogram[0], budget_histogram[1], budget_histogram[2],
                     budget_histogram[3], budget_histogram[4], budget_histogram[5],
                     budget_histogram[6]);
    } else if (mode == CalibratedSparseMode::SequentialAllocate34) {
        CHECK(token_ids.size() >= 2);
        if (token_ids.size() < 2) return test_summary("test_canonical_metal");
        std::vector<uint32_t> calibration_inputs;
        std::vector<uint32_t> reference_top1;
        {
            auto reference_created = create_qualification_canonical_metal_program(
                package, 4);
            CHECK(std::holds_alternative<CanonicalMetalProgram>(reference_created));
            if (!std::holds_alternative<CanonicalMetalProgram>(reference_created))
                return test_summary("test_canonical_metal");
            CanonicalMetalProgram reference =
                std::get<CanonicalMetalProgram>(std::move(reference_created));
            uint32_t token = package->semantics().bos_id;
            for (size_t step = 0; step != 4; ++step) {
                calibration_inputs.push_back(token);
                auto result = reference.decode(token);
                CHECK(std::holds_alternative<CanonicalMetalOutput>(result));
                if (!std::holds_alternative<CanonicalMetalOutput>(result))
                    return test_summary("test_canonical_metal");
                const auto& output = std::get<CanonicalMetalOutput>(result);
                CHECK(output.completed && output.command_buffers == 1);
                bool finite = true;
                for (float value : output.logits) finite = finite && std::isfinite(value);
                CHECK(finite);
                if (!finite) return test_summary("test_canonical_metal");
                reference_top1.push_back(static_cast<uint32_t>(
                    std::max_element(output.logits.begin(), output.logits.end()) -
                    output.logits.begin()));
                token = reference_top1.back();
            }
        }

        constexpr std::array<uint32_t, 5> sequential_levels = {20, 27, 34, 41, 48};
        std::vector<uint32_t> allocated_counts(rankings.size(), 48);
        for (size_t layer = 0; layer != rankings.size(); ++layer) {
            bool accepted = false;
            for (uint32_t count : sequential_levels) {
                std::vector<uint32_t> trial_counts = allocated_counts;
                trial_counts[layer] = count;
                CanonicalSparseFfnPolicy trial_policy = policy_for_counts(trial_counts);
                CHECK(trial_policy.layer_masks.size() == rankings.size());
                auto trial_created = create_qualification_canonical_metal_program(
                    package, static_cast<uint32_t>(calibration_inputs.size()), {},
                    trial_policy);
                CHECK(std::holds_alternative<CanonicalMetalProgram>(trial_created));
                if (!std::holds_alternative<CanonicalMetalProgram>(trial_created))
                    return test_summary("test_canonical_metal");
                CanonicalMetalProgram trial =
                    std::get<CanonicalMetalProgram>(std::move(trial_created));
                uint32_t mismatches = 0;
                bool finite = true;
                for (size_t step = 0; step != calibration_inputs.size(); ++step) {
                    auto result = trial.decode(calibration_inputs[step]);
                    CHECK(std::holds_alternative<CanonicalMetalOutput>(result));
                    if (!std::holds_alternative<CanonicalMetalOutput>(result))
                        return test_summary("test_canonical_metal");
                    const auto& output = std::get<CanonicalMetalOutput>(result);
                    CHECK(output.completed && output.command_buffers == 1);
                    for (float value : output.logits) finite = finite && std::isfinite(value);
                    const uint32_t top1 = static_cast<uint32_t>(
                        std::max_element(output.logits.begin(), output.logits.end()) -
                        output.logits.begin());
                    mismatches += top1 != reference_top1[step];
                }
                std::fprintf(stderr,
                             "sparse sequential layer=%u count=%u mismatches=%u finite=%u\n",
                             rankings[layer].layer_index, count, mismatches,
                             finite ? 1u : 0u);
                if (finite && mismatches == 0) {
                    allocated_counts[layer] = count;
                    accepted = true;
                    break;
                }
            }
            CHECK(accepted);
            if (!accepted) return test_summary("test_canonical_metal");
        }

        selected_block_total = 0;
        budget_histogram.fill(0);
        for (size_t layer = 0; layer != rankings.size(); ++layer) {
            selected_block_total += allocated_counts[layer];
            const auto found = std::find(retention_levels.begin(), retention_levels.end(),
                                         allocated_counts[layer]);
            CHECK(found != retention_levels.end());
            if (found == retention_levels.end()) return test_summary("test_canonical_metal");
            ++budget_histogram[static_cast<size_t>(found - retention_levels.begin())];
        }
        policy = policy_for_counts(allocated_counts);
        CHECK(policy.layer_masks.size() == rankings.size());
        expected_source_bytes = 0;
        expected_requested_bytes = 0;
        for (size_t layer = 0; layer != rankings.size(); ++layer) {
            expected_source_bytes += rankings[layer].source_bytes;
            expected_requested_bytes += rankings[layer].source_bytes *
                                        allocated_counts[layer] /
                                        rankings[layer].total_blocks;
        }
        std::fprintf(stderr,
                     "sparse sequential allocator calibration_prefix=%u,%u,%u,%u "
                     "total_blocks=%llu average=%.3f hist20=%u hist27=%u hist34=%u "
                     "hist41=%u hist48=%u\n",
                     calibration_inputs[0], calibration_inputs[1],
                     calibration_inputs[2], calibration_inputs[3],
                     static_cast<unsigned long long>(selected_block_total),
                     static_cast<double>(selected_block_total) / rankings.size(),
                     budget_histogram[0], budget_histogram[1], budget_histogram[2],
                     budget_histogram[3], budget_histogram[4]);
    } else {
        budget_histogram[4] = static_cast<uint32_t>(rankings.size());
    }
    std::vector<uint32_t> expected_layer_block_counts;
    expected_layer_block_counts.reserve(policy.layer_masks.size());
    for (const auto& mask : policy.layer_masks)
        expected_layer_block_counts.push_back(static_cast<uint32_t>(mask.block_ids.size()));
    std::vector<uint32_t> input_tokens;
    std::vector<uint32_t> control_top1s;
    std::vector<std::vector<float>> control_logits;
    input_tokens.reserve(17);
    control_top1s.reserve(17);
    control_logits.reserve(17);
    double control_gpu_ms = 0.0;
    uint64_t control_projection_bytes = 0;
    {
        auto control_created = create_qualification_canonical_metal_program(package, 128);
        CHECK(std::holds_alternative<CanonicalMetalProgram>(control_created));
        if (!std::holds_alternative<CanonicalMetalProgram>(control_created))
            return test_summary("test_canonical_metal");
        CanonicalMetalProgram control = std::get<CanonicalMetalProgram>(std::move(control_created));
        uint32_t token = package->semantics().bos_id;
        for (size_t step = 0; step != 17; ++step) {
            input_tokens.push_back(token);
            auto result = control.decode(token);
            CHECK(std::holds_alternative<CanonicalMetalOutput>(result));
            if (!std::holds_alternative<CanonicalMetalOutput>(result))
                return test_summary("test_canonical_metal");
            CanonicalMetalOutput output = std::get<CanonicalMetalOutput>(std::move(result));
            CHECK(output.completed && output.command_buffers == 1);
            token = static_cast<uint32_t>(std::max_element(output.logits.begin(), output.logits.end()) -
                                          output.logits.begin());
            control_top1s.push_back(token);
            control_gpu_ms += output.gpu_time_ms;
            control_projection_bytes = output.requested_projection_source_bytes;
            control_logits.push_back(std::move(output.logits));
        }
    }
    auto created = create_qualification_canonical_metal_program(package, 128, {}, policy);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return test_summary("test_canonical_metal");
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    CHECK(program.sparse_ffn_layer_count() == policy.layer_masks.size());
    CHECK(program.sparse_ffn_source_bytes() == expected_source_bytes);
    CHECK(program.sparse_ffn_requested_bytes() == expected_requested_bytes);
    CHECK(program.sparse_ffn_worklist_bytes() ==
          selected_block_total * sizeof(uint32_t));
    CHECK(program.sparse_ffn_block_counts_for_testing() ==
          expected_layer_block_counts);
    size_t matched_top1 = 0;
    size_t first_mismatch = SIZE_MAX;
    uint32_t first_control = 0;
    uint32_t first_candidate = 0;
    float maximum_delta = 0.0f;
    double candidate_gpu_ms = 0.0;
    uint64_t candidate_projection_bytes = 0;
    uint64_t lm_head_bytes = 0;
    for (const SemanticTensor& tensor : package->semantics().tensors)
        if (tensor.role == TensorRole::OutputWeight && tensor.planes.size() == 1)
            lm_head_bytes = tensor.planes.front().length;
    for (size_t step = 0; step != 17; ++step) {
        auto candidate_result = program.decode(input_tokens[step]);
        CHECK(std::holds_alternative<CanonicalMetalOutput>(candidate_result));
        if (!std::holds_alternative<CanonicalMetalOutput>(candidate_result))
            return test_summary("test_canonical_metal");
        const auto& candidate_output = std::get<CanonicalMetalOutput>(candidate_result);
        CHECK(candidate_output.completed && candidate_output.command_buffers == 1);
        CHECK(control_logits[step].size() == candidate_output.logits.size());
        const uint32_t control_top1 = control_top1s[step];
        const uint32_t candidate_top1 = static_cast<uint32_t>(std::max_element(
            candidate_output.logits.begin(), candidate_output.logits.end()) - candidate_output.logits.begin());
        matched_top1 += control_top1 == candidate_top1;
        if (control_top1 != candidate_top1 && first_mismatch == SIZE_MAX) {
            first_mismatch = step;
            first_control = control_top1;
            first_candidate = candidate_top1;
        }
        for (size_t index = 0; index != control_logits[step].size(); ++index) {
            CHECK(std::isfinite(candidate_output.logits[index]));
            maximum_delta = std::max(maximum_delta,
                                     std::abs(candidate_output.logits[index] -
                                              control_logits[step][index]));
        }
        CHECK(candidate_output.requested_projection_source_bytes ==
              control_projection_bytes - expected_source_bytes + expected_requested_bytes);
        candidate_projection_bytes = candidate_output.requested_projection_source_bytes;
        candidate_gpu_ms += candidate_output.gpu_time_ms;
    }
    const auto diagnostics = program.resource_diagnostics();
    CHECK(diagnostics.implicit_weight_copies == 0);
    CHECK(matched_top1 == 17);
    std::fprintf(stderr,
                 "calibrated sparse FFN %s: matched_top1=%zu/17 first_mismatch=%zu "
                 "control_top1=%u candidate_top1=%u max_logit_delta=%.9g "
                 "layers=%zu total_blocks=%llu average=%.3f "
                 "hist20=%u hist27=%u hist34=%u hist41=%u hist48=%u hist55=%u hist68=%u "
                 "worklist_bytes=%llu control_intermediate_bytes=%llu "
                 "candidate_intermediate_bytes=%llu lm_head_bytes=%llu "
                 "candidate_total_bytes=%llu control_gpu_ms=%.3f candidate_gpu_ms=%.3f "
                 "one_cb=1 implicit_copies=%llu production_routing=0 "
                 "calibration_all_32_windows=1 held_out=0\n",
                 mode == CalibratedSparseMode::MarginalAllocate
                     ? "marginal"
                     : (mode == CalibratedSparseMode::SensitivityAllocate34
                            ? "sensitivity<=34"
                            : (mode == CalibratedSparseMode::SequentialAllocate34
                                   ? "sequential<=34"
                                   : "48/68")),
                 matched_top1, first_mismatch, first_control, first_candidate,
                 maximum_delta, policy.layer_masks.size(),
                 static_cast<unsigned long long>(selected_block_total),
                 static_cast<double>(selected_block_total) / policy.layer_masks.size(),
                 budget_histogram[0], budget_histogram[1], budget_histogram[2],
                 budget_histogram[3], budget_histogram[4], budget_histogram[5],
                 budget_histogram[6],
                 static_cast<unsigned long long>(program.sparse_ffn_worklist_bytes()),
                 static_cast<unsigned long long>(control_projection_bytes),
                 static_cast<unsigned long long>(candidate_projection_bytes),
                 static_cast<unsigned long long>(lm_head_bytes),
                 static_cast<unsigned long long>(candidate_projection_bytes + lm_head_bytes),
                 control_gpu_ms, candidate_gpu_ms,
                 static_cast<unsigned long long>(diagnostics.implicit_weight_copies));
    return test_summary("test_canonical_metal");
}

int test_generic_metal_iq2_xxs_ffn_quality(const char* path) {
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return test_summary("test_canonical_metal");
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return test_summary("test_canonical_metal");
    auto loaded = load_validated_gguf(std::get<PackageView>(view));
    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
    if (!std::holds_alternative<ValidatedPackage>(loaded)) return test_summary("test_canonical_metal");
    const auto package = std::get<ValidatedPackage>(loaded).runtime_package();
    const SemanticModel& model = package->semantics();

    std::vector<bool> speculative_operator(model.operators.size(), false);
    for (const SemanticLayer& layer : model.layers) {
        if ((layer.flags & kSemanticLayerFlagSpeculative) == 0) continue;
        for (uint32_t index = 0; index != layer.operator_count; ++index) {
            if (layer.first_operator + index < speculative_operator.size())
                speculative_operator[layer.first_operator + index] = true;
        }
    }
    const auto selected_role = [](TensorRole role) {
        return role == TensorRole::FfnGateWeight || role == TensorRole::FfnUpWeight ||
               role == TensorRole::FfnDownWeight;
    };
    uint64_t baseline_projection_bytes = 0;
    uint64_t lm_head_bytes = 0;
    uint64_t selected_source_bytes = 0;
    uint64_t selected_iq_bytes = 0;
    uint32_t selected_tensors = 0;
    for (size_t operator_index = 0; operator_index != model.operators.size(); ++operator_index) {
        const SemanticOperator& op = model.operators[operator_index];
        if (speculative_operator[operator_index] || op.kind != OperatorKind::Linear) continue;
        for (uint32_t tensor_id : op.tensors) {
            CHECK(tensor_id < model.tensors.size());
            if (tensor_id >= model.tensors.size()) continue;
            const SemanticTensor& tensor = model.tensors[tensor_id];
            if (tensor.dimensions.size() != 2 || tensor.planes.size() != 1 ||
                tensor.layout.block_elements != 256 ||
                (tensor.layout.block_bytes != 144 && tensor.layout.block_bytes != 210)) continue;
            baseline_projection_bytes += tensor.planes[0].length;
            if (tensor.role == TensorRole::OutputWeight) lm_head_bytes += tensor.planes[0].length;
            if (!selected_role(tensor.role)) continue;
            const uint64_t elements = tensor.dimensions[0].constant_or_symbol *
                                      tensor.dimensions[1].constant_or_symbol;
            selected_source_bytes += tensor.planes[0].length;
            selected_iq_bytes += elements / 256 * 66;
            ++selected_tensors;
        }
    }
    CHECK(selected_tensors > 0);
    CHECK(selected_source_bytes > selected_iq_bytes);
    CHECK(baseline_projection_bytes >= selected_source_bytes);
    CHECK(baseline_projection_bytes >= lm_head_bytes);
    const uint64_t candidate_projection_bytes =
        baseline_projection_bytes - selected_source_bytes + selected_iq_bytes;
    const uint64_t baseline_intermediate_projection_bytes = baseline_projection_bytes - lm_head_bytes;
    const uint64_t candidate_intermediate_projection_bytes = candidate_projection_bytes - lm_head_bytes;
    constexpr uint64_t target_projection_bytes = 6140000000ull;
    std::fprintf(stderr,
                 "IQ2_XXS FFN byte ledger: total_baseline=%llu total_candidate=%llu intermediate_baseline=%llu intermediate_candidate=%llu lm_head=%llu selected_source=%llu selected_iq=%llu target=%llu crosses_target=%u tensors=%u\n",
                 static_cast<unsigned long long>(baseline_projection_bytes),
                 static_cast<unsigned long long>(candidate_projection_bytes),
                 static_cast<unsigned long long>(baseline_intermediate_projection_bytes),
                 static_cast<unsigned long long>(candidate_intermediate_projection_bytes),
                 static_cast<unsigned long long>(lm_head_bytes),
                 static_cast<unsigned long long>(selected_source_bytes),
                 static_cast<unsigned long long>(selected_iq_bytes),
                 static_cast<unsigned long long>(target_projection_bytes),
                 candidate_projection_bytes <= target_projection_bytes, selected_tensors);

    std::vector<std::vector<float>> control_logits;
    std::vector<uint32_t> input_tokens;
    control_logits.reserve(17);
    input_tokens.reserve(17);
    double control_gpu_ms = 0.0;
    uint64_t control_peak_session_bytes = 0;
    {
        auto control_created = create_qualification_canonical_metal_program(package, 128);
        CHECK(std::holds_alternative<CanonicalMetalProgram>(control_created));
        if (!std::holds_alternative<CanonicalMetalProgram>(control_created))
            return test_summary("test_canonical_metal");
        CanonicalMetalProgram control = std::get<CanonicalMetalProgram>(std::move(control_created));
        uint32_t token = model.bos_id;
        for (size_t step = 0; step != 17; ++step) {
            input_tokens.push_back(token);
            auto result = control.decode(token);
            if (const auto* report = std::get_if<CompatibilityReport>(&result))
                std::fprintf(stderr,
                             "IQ2_XXS FFN control: code=%u layer=%u operator=%u detail=%s\n",
                             static_cast<unsigned>(report->code), report->layer,
                             report->operator_id, report->detail.c_str());
            CHECK(std::holds_alternative<CanonicalMetalOutput>(result));
            if (!std::holds_alternative<CanonicalMetalOutput>(result))
                return test_summary("test_canonical_metal");
            CanonicalMetalOutput output = std::get<CanonicalMetalOutput>(std::move(result));
            CHECK(output.command_buffers == 1);
            CHECK(output.requested_projection_source_bytes == baseline_intermediate_projection_bytes);
            control_gpu_ms += output.gpu_time_ms;
            control_peak_session_bytes = std::max(control_peak_session_bytes,
                                                  output.peak_session_bytes);
            token = static_cast<uint32_t>(
                std::max_element(output.logits.begin(), output.logits.end()) - output.logits.begin());
            control_logits.push_back(std::move(output.logits));
        }
    }
    CHECK(control_logits.size() == 17);

    CanonicalDerivedIQ2XXSPolicy policy;
    policy.tensor_roles = {TensorRole::FfnGateWeight, TensorRole::FfnUpWeight,
                           TensorRole::FfnDownWeight};
    policy.enable_metal_error_diagnostics = true;
    const auto construction_started = std::chrono::steady_clock::now();
    auto candidate_created = create_qualification_canonical_metal_program(package, 128, {}, {}, policy);
    const double construction_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - construction_started).count();
    if (const auto* report = std::get_if<CompatibilityReport>(&candidate_created))
        std::fprintf(stderr, "IQ2_XXS FFN candidate: code=%u detail=%s\n",
                     static_cast<unsigned>(report->code), report->detail.c_str());
    CHECK(std::holds_alternative<CanonicalMetalProgram>(candidate_created));
    if (!std::holds_alternative<CanonicalMetalProgram>(candidate_created))
        return test_summary("test_canonical_metal");
    CanonicalMetalProgram candidate = std::get<CanonicalMetalProgram>(std::move(candidate_created));
    CHECK(candidate.derived_iq2_xxs_atlas_count() == 1);
    CHECK(candidate.derived_iq2_xxs_tensor_count() == selected_tensors);
    CHECK(candidate.derived_iq2_xxs_source_bytes() == selected_source_bytes);
    CHECK(candidate.derived_iq2_xxs_storage_bytes() == selected_iq_bytes);
    const CanonicalMetalResourceDiagnostics candidate_construction = candidate.resource_diagnostics();
    CHECK(candidate_construction.registered_source_bytes < package->artifact_bytes(ArtifactId{0}).size());
    CHECK(candidate_construction.excluded_replaced_bytes == selected_source_bytes);
    CHECK(candidate_construction.implicit_weight_copies == 0);

    size_t matched_top1 = 0;
    size_t first_mismatch = SIZE_MAX;
    uint32_t first_control = 0;
    uint32_t first_candidate = 0;
    float maximum_delta = 0.0f;
    double candidate_gpu_ms = 0.0;
    uint64_t candidate_peak_session_bytes = 0;
    for (size_t step = 0; step != 17; ++step) {
        auto candidate_result = candidate.decode(input_tokens[step]);
        if (const auto* report = std::get_if<CompatibilityReport>(&candidate_result))
            std::fprintf(stderr,
                         "IQ2_XXS FFN submission: code=%u layer=%u operator=%u detail=%s\n",
                         static_cast<unsigned>(report->code), report->layer,
                         report->operator_id, report->detail.c_str());
        CHECK(std::holds_alternative<CanonicalMetalOutput>(candidate_result));
        if (!std::holds_alternative<CanonicalMetalOutput>(candidate_result)) break;
        const auto& dense = control_logits[step];
        const auto& iq = std::get<CanonicalMetalOutput>(candidate_result);
        CHECK(iq.command_buffers == 1);
        CHECK(iq.requested_projection_source_bytes == candidate_intermediate_projection_bytes);
        CHECK(dense.size() == iq.logits.size());
        const uint32_t dense_top1 = static_cast<uint32_t>(
            std::max_element(dense.begin(), dense.end()) - dense.begin());
        const uint32_t iq_top1 = static_cast<uint32_t>(
            std::max_element(iq.logits.begin(), iq.logits.end()) - iq.logits.begin());
        if (dense_top1 == iq_top1) ++matched_top1;
        else if (first_mismatch == SIZE_MAX) {
            first_mismatch = step;
            first_control = dense_top1;
            first_candidate = iq_top1;
        }
        for (size_t index = 0; index != dense.size(); ++index) {
            CHECK(std::isfinite(dense[index]));
            CHECK(std::isfinite(iq.logits[index]));
            maximum_delta = std::max(maximum_delta,
                                     std::abs(dense[index] - iq.logits[index]));
        }
        candidate_gpu_ms += iq.gpu_time_ms;
        candidate_peak_session_bytes = std::max(candidate_peak_session_bytes,
                                                iq.peak_session_bytes);
    }
    std::fprintf(stderr,
                 "IQ2_XXS FFN quality: matched_top1=%zu/17 first_mismatch=%zu control_top1=%u iq_top1=%u max_logit_delta=%.9g construction_ms=%.3f control_gpu_ms=%.3f iq_gpu_ms=%.3f source_registered=%llu excluded_replaced=%llu boundary=%llu registration_overlap=%llu atlas_registered=%llu allocated_before_source=%llu allocated_after_source=%llu allocated_after_atlas=%llu allocated_after_session=%llu recommended_max_working_set=%llu implicit_copies=%llu control_peak_session_bytes=%llu candidate_peak_session_bytes=%llu dense_session_destroyed_before_candidate=1 uniform_importance=1 one_cb=1\n",
                 matched_top1, first_mismatch, first_control, first_candidate, maximum_delta,
                 construction_ms, control_gpu_ms, candidate_gpu_ms,
                 static_cast<unsigned long long>(candidate_construction.registered_source_bytes),
                 static_cast<unsigned long long>(candidate_construction.excluded_replaced_bytes),
                 static_cast<unsigned long long>(candidate_construction.retained_boundary_bytes),
                 static_cast<unsigned long long>(candidate_construction.registration_overlap_bytes),
                 static_cast<unsigned long long>(candidate_construction.atlas_bytes),
                 static_cast<unsigned long long>(candidate_construction.before_source_registration),
                 static_cast<unsigned long long>(candidate_construction.after_source_registration),
                 static_cast<unsigned long long>(candidate_construction.after_atlas_registration),
                 static_cast<unsigned long long>(candidate_construction.after_session_construction),
                 static_cast<unsigned long long>(candidate_construction.recommended_max_working_set_size),
                 static_cast<unsigned long long>(candidate.resource_diagnostics().implicit_weight_copies),
                 static_cast<unsigned long long>(control_peak_session_bytes),
                 static_cast<unsigned long long>(candidate_peak_session_bytes));
    CHECK(matched_top1 == 17);
    return test_summary("test_canonical_metal");
}

int test_generic_metal_iq2_xxs_zero_atlas(const char* path) {
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return test_summary("test_canonical_metal");
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return test_summary("test_canonical_metal");
    auto loaded = load_validated_gguf(std::get<PackageView>(view));
    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
    if (!std::holds_alternative<ValidatedPackage>(loaded)) return test_summary("test_canonical_metal");
    const auto package = std::get<ValidatedPackage>(loaded).runtime_package();
    CanonicalDerivedIQ2XXSPolicy policy;
    policy.tensor_roles = {TensorRole::FfnGateWeight, TensorRole::FfnUpWeight,
                           TensorRole::FfnDownWeight};
    policy.enable_metal_error_diagnostics = true;
    policy.zero_fill_for_testing = true;
    const auto construction_started = std::chrono::steady_clock::now();
    auto created = create_qualification_canonical_metal_program(package, 128, {}, {}, policy);
    const double construction_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - construction_started).count();
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) {
        if (const auto* report = std::get_if<CompatibilityReport>(&created))
            std::fprintf(stderr, "IQ2_XXS zero atlas construction: code=%u detail=%s\n",
                         static_cast<unsigned>(report->code), report->detail.c_str());
        return test_summary("test_canonical_metal");
    }
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    const CanonicalMetalResourceDiagnostics construction = program.resource_diagnostics();
    CHECK(construction.registered_source_bytes < package->artifact_bytes(ArtifactId{0}).size());
    CHECK(construction.excluded_replaced_bytes == program.derived_iq2_xxs_source_bytes());
    CHECK(construction.atlas_bytes == program.derived_iq2_xxs_registered_bytes());
    CHECK(construction.implicit_weight_copies == 0);
    auto result = program.decode(package->semantics().bos_id);
    const CanonicalMetalResourceDiagnostics completed = program.resource_diagnostics();
    CHECK(completed.implicit_weight_copies == 0);
    if (const auto* report = std::get_if<CompatibilityReport>(&result)) {
        std::fprintf(stderr,
                     "IQ2_XXS zero atlas submission: code=%u layer=%u operator=%u detail=%s construction_ms=%.3f source_registered=%llu excluded_replaced=%llu boundary=%llu registration_overlap=%llu atlas_registered=%llu allocated_before_source=%llu allocated_after_source=%llu allocated_after_atlas=%llu allocated_after_session=%llu recommended_max_working_set=%llu implicit_copies=%llu\n",
                     static_cast<unsigned>(report->code), report->layer, report->operator_id,
                     report->detail.c_str(), construction_ms,
                     static_cast<unsigned long long>(construction.registered_source_bytes),
                     static_cast<unsigned long long>(construction.excluded_replaced_bytes),
                     static_cast<unsigned long long>(construction.retained_boundary_bytes),
                     static_cast<unsigned long long>(construction.registration_overlap_bytes),
                     static_cast<unsigned long long>(construction.atlas_bytes),
                     static_cast<unsigned long long>(construction.before_source_registration),
                     static_cast<unsigned long long>(construction.after_source_registration),
                     static_cast<unsigned long long>(construction.after_atlas_registration),
                     static_cast<unsigned long long>(construction.after_session_construction),
                     static_cast<unsigned long long>(construction.recommended_max_working_set_size),
                     static_cast<unsigned long long>(completed.implicit_weight_copies));
        CHECK(report->detail.find("Metal command buffer") != std::string::npos);
    } else {
        const auto& output = std::get<CanonicalMetalOutput>(result);
        std::fprintf(stderr,
                     "IQ2_XXS zero atlas submission: completed=1 construction_ms=%.3f gpu_ms=%.3f peak_session_bytes=%llu source_registered=%llu excluded_replaced=%llu boundary=%llu registration_overlap=%llu atlas_registered=%llu allocated_before_source=%llu allocated_after_source=%llu allocated_after_atlas=%llu allocated_after_session=%llu recommended_max_working_set=%llu implicit_copies=%llu\n",
                     construction_ms, output.gpu_time_ms,
                     static_cast<unsigned long long>(output.peak_session_bytes),
                     static_cast<unsigned long long>(construction.registered_source_bytes),
                     static_cast<unsigned long long>(construction.excluded_replaced_bytes),
                     static_cast<unsigned long long>(construction.retained_boundary_bytes),
                     static_cast<unsigned long long>(construction.registration_overlap_bytes),
                     static_cast<unsigned long long>(construction.atlas_bytes),
                     static_cast<unsigned long long>(construction.before_source_registration),
                     static_cast<unsigned long long>(construction.after_source_registration),
                     static_cast<unsigned long long>(construction.after_atlas_registration),
                     static_cast<unsigned long long>(construction.after_session_construction),
                     static_cast<unsigned long long>(construction.recommended_max_working_set_size),
                     static_cast<unsigned long long>(completed.implicit_weight_copies));
        CHECK(output.command_buffers == 1);
    }
    return test_summary("test_canonical_metal");
}

int test_generic_metal_iq2_xxs_layout(const char* path) {
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return test_summary("test_canonical_metal");
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return test_summary("test_canonical_metal");
    auto loaded = load_validated_gguf(std::get<PackageView>(view));
    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
    if (!std::holds_alternative<ValidatedPackage>(loaded)) return test_summary("test_canonical_metal");
    const SemanticModel& model = std::get<ValidatedPackage>(loaded).runtime_package()->semantics();
    const SemanticLayer* first_recurrent = nullptr;
    uint32_t anchor = UINT32_MAX;
    for (const SemanticLayer& layer : model.layers) {
        for (uint32_t index = 0; index != layer.operator_count; ++index) {
            const SemanticOperator& op = model.operators[layer.first_operator + index];
            if (op.kind == OperatorKind::GatedDeltaNet) {
                first_recurrent = &layer;
                anchor = op.id;
                break;
            }
        }
        if (first_recurrent) break;
    }
    CHECK(first_recurrent != nullptr);
    if (!first_recurrent) return test_summary("test_canonical_metal");
    std::vector<bool> first_tensor(model.tensors.size(), false);
    for (uint32_t index = 0; index != first_recurrent->operator_count; ++index) {
        const SemanticOperator& op = model.operators[first_recurrent->first_operator + index];
        for (uint32_t tensor_id : op.tensors)
            if (tensor_id < first_tensor.size()) first_tensor[tensor_id] = true;
    }
    const auto selected = [](TensorRole role) {
        return role == TensorRole::FfnGateWeight || role == TensorRole::FfnUpWeight ||
               role == TensorRole::FfnDownWeight;
    };
    const auto role_name = [](TensorRole role) {
        if (role == TensorRole::FfnGateWeight) return "FfnGateWeight";
        if (role == TensorRole::FfnUpWeight) return "FfnUpWeight";
        if (role == TensorRole::FfnDownWeight) return "FfnDownWeight";
        return "unknown";
    };
    uint64_t atlas_length = 0;
    uint32_t found = 0;
    for (const SemanticTensor& tensor : model.tensors) {
        if (!selected(tensor.role) || tensor.dimensions.size() != 2) continue;
        const uint64_t K = tensor.dimensions[0].constant_or_symbol;
        const uint64_t N = tensor.dimensions[1].constant_or_symbol;
        CHECK(K != 0 && K % 256 == 0 && N != 0);
        const uint64_t offset = (atlas_length + 127) / 128 * 128;
        const uint64_t length = K / 256 * N * 66;
        atlas_length = offset + length;
        if (tensor.id < first_tensor.size() && first_tensor[tensor.id]) {
            std::fprintf(stderr,
                         "IQ2_XXS first recurrent binding: layer=%u operator=%u tensor=%u role=%s K=%llu N=%llu source_block=%u derived_offset=%llu derived_length=%llu\n",
                         first_recurrent->layer_index, anchor, tensor.id, role_name(tensor.role),
                         static_cast<unsigned long long>(K), static_cast<unsigned long long>(N),
                         tensor.layout.block_bytes, static_cast<unsigned long long>(offset),
                         static_cast<unsigned long long>(length));
            ++found;
        }
    }
    CHECK(found == 3);
    std::fprintf(stderr, "IQ2_XXS derived layout: tensors=%u atlas_length=%llu\n",
                 found, static_cast<unsigned long long>(atlas_length));
    return test_summary("test_canonical_metal");
}

int test_generic_metal_sparse_ffn_ownership(const char* path, bool all_blocks,
                                             bool projection_probe_only = false,
                                             bool device_selector = false,
                                             bool dense_oracle_selector = false) {
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return test_summary("test_canonical_metal");
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return test_summary("test_canonical_metal");
    auto loaded = load_validated_gguf(std::get<PackageView>(view));
    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
    if (!std::holds_alternative<ValidatedPackage>(loaded)) return test_summary("test_canonical_metal");
    const auto package = std::get<ValidatedPackage>(loaded).runtime_package();

    uint64_t intermediate = 0;
    uint64_t hidden_width = 0;
    uint32_t expected_layers = 0;
    uint64_t expected_source_bytes = 0;
    for (const SemanticLayer& layer : package->semantics().layers) {
        if ((layer.flags & kSemanticLayerFlagSpeculative) != 0) continue;
        const SemanticTensor* gate = nullptr;
        const SemanticTensor* up = nullptr;
        const SemanticTensor* down = nullptr;
        for (uint32_t index = 0; index != layer.operator_count; ++index) {
            const SemanticOperator& op = package->semantics().operators[layer.first_operator + index];
            if (op.kind != OperatorKind::Linear || op.tensors.size() != 1) continue;
            const SemanticTensor& tensor = package->semantics().tensors[op.tensors[0]];
            if (tensor.role == TensorRole::FfnGateWeight) gate = &tensor;
            if (tensor.role == TensorRole::FfnUpWeight) up = &tensor;
            if (tensor.role == TensorRole::FfnDownWeight) down = &tensor;
        }
        CHECK(gate != nullptr && up != nullptr && down != nullptr);
        if (!gate || !up || !down) return test_summary("test_canonical_metal");
        CHECK(gate->dimensions.size() == 2 && up->dimensions.size() == 2 && down->dimensions.size() == 2);
        CHECK(gate->planes.size() == 1 && up->planes.size() == 1 && down->planes.size() == 1);
        const uint64_t layer_intermediate = gate->dimensions[1].constant_or_symbol;
        CHECK(layer_intermediate != 0 && layer_intermediate % 256 == 0);
        CHECK(!hidden_width || hidden_width == gate->dimensions[0].constant_or_symbol);
        CHECK(!intermediate || intermediate == layer_intermediate);
        hidden_width = gate->dimensions[0].constant_or_symbol;
        intermediate = layer_intermediate;
        ++expected_layers;
        expected_source_bytes += gate->planes[0].length + up->planes[0].length + down->planes[0].length;
    }
    CHECK(expected_layers > 0 && intermediate >= 512);

    auto control_created = create_qualification_canonical_metal_program(package, 128);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(control_created));
    if (!std::holds_alternative<CanonicalMetalProgram>(control_created))
        return test_summary("test_canonical_metal");
    CanonicalMetalProgram control = std::get<CanonicalMetalProgram>(std::move(control_created));

    CanonicalSparseFfnPolicy policy;
    const uint32_t total_blocks = static_cast<uint32_t>(intermediate / 256);
    if (projection_probe_only) {
        CanonicalSparseFfnPolicy all_policy{{{0, total_blocks}}};
        CanonicalSparseFfnPolicy half_policy{{{0, total_blocks / 2}}};
        const size_t hidden = static_cast<size_t>(hidden_width);
        CHECK(hidden > 0);
        std::vector<float> input(hidden);
        for (size_t index = 0; index != input.size(); ++index)
            input[index] = std::sin(static_cast<float>(index + 1) * 0.001f);
        std::vector<float> dense_output, all_output, half_output;
        double dense_warm = 0.0, all_warm = 0.0, half_warm = 0.0;
        const bool dense_warm_ok = canonical_metal_sparse_ffn_probe_for_testing(
            control, input, {}, dense_output, dense_warm);
        const bool all_warm_ok = canonical_metal_sparse_ffn_probe_for_testing(
            control, input, all_policy, all_output, all_warm);
        const bool half_warm_ok = canonical_metal_sparse_ffn_probe_for_testing(
            control, input, half_policy, half_output, half_warm);
        CHECK(dense_warm_ok && all_warm_ok && half_warm_ok);
        if (!dense_warm_ok || !all_warm_ok || !half_warm_ok)
            return test_summary("test_canonical_metal");
        CHECK(dense_output == all_output);
        constexpr size_t probe_repetitions = 20;
        std::vector<double> dense_gpu(probe_repetitions), all_gpu(probe_repetitions),
                            half_gpu(probe_repetitions);
        for (size_t repeat = 0; repeat != probe_repetitions; ++repeat) {
            const auto run_dense = [&] {
                return canonical_metal_sparse_ffn_probe_for_testing(
                    control, input, {}, dense_output, dense_gpu[repeat]);
            };
            const auto run_all = [&] {
                return canonical_metal_sparse_ffn_probe_for_testing(
                    control, input, all_policy, all_output, all_gpu[repeat]);
            };
            const auto run_half = [&] {
                return canonical_metal_sparse_ffn_probe_for_testing(
                    control, input, half_policy, half_output, half_gpu[repeat]);
            };
            bool dense_ok = false, all_ok = false, half_ok = false;
            if (repeat % 3 == 0) {
                dense_ok = run_dense(); all_ok = run_all(); half_ok = run_half();
            } else if (repeat % 3 == 1) {
                all_ok = run_all(); half_ok = run_half(); dense_ok = run_dense();
            } else {
                half_ok = run_half(); dense_ok = run_dense(); all_ok = run_all();
            }
            CHECK(dense_ok && all_ok && half_ok);
            if (!dense_ok || !all_ok || !half_ok) return test_summary("test_canonical_metal");
            CHECK(dense_output == all_output);
        }
        const auto probe_median = [](std::vector<double> values) {
            std::sort(values.begin(), values.end());
            return (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2.0;
        };
        const auto dense_range = std::minmax_element(dense_gpu.begin(), dense_gpu.end());
        const auto all_range = std::minmax_element(all_gpu.begin(), all_gpu.end());
        const auto half_range = std::minmax_element(half_gpu.begin(), half_gpu.end());
        float half_max_delta = 0.0f;
        for (size_t index = 0; index != dense_output.size(); ++index)
            half_max_delta = std::max(half_max_delta,
                                      std::abs(dense_output[index] - half_output[index]));
        std::fprintf(stderr,
                     "generic sparse FFN fixed-input probe: layer=0 hidden=%zu full_blocks=%u half_blocks=%u warm_each=1 measured_each=%zu shared_session=1 persistent_state=0 command_buffers_per_probe=1 dense_gpu_median_ms=%.3f dense_gpu_range_ms=%.3f..%.3f all_affine_gpu_median_ms=%.3f all_affine_gpu_range_ms=%.3f..%.3f half_affine_gpu_median_ms=%.3f half_affine_gpu_range_ms=%.3f..%.3f all_max_delta=0 half_max_delta=%.9g\n",
                     hidden, total_blocks, total_blocks / 2, probe_repetitions,
                     probe_median(dense_gpu), *dense_range.first, *dense_range.second,
                     probe_median(all_gpu), *all_range.first, *all_range.second,
                     probe_median(half_gpu), *half_range.first, *half_range.second,
                     half_max_delta);
        return test_summary("test_canonical_metal");
    }
    const uint32_t selected_blocks = (device_selector || dense_oracle_selector) ? 34u :
                                     all_blocks ? total_blocks : total_blocks / 2;
    CHECK(selected_blocks <= total_blocks);
    if (dense_oracle_selector) policy.dense_oracle_selected_blocks = selected_blocks;
    else if (device_selector) policy.proxy_selected_blocks = selected_blocks;
    else policy.runs = {{0, selected_blocks}};
    const auto create_started = std::chrono::steady_clock::now();
    auto created = create_qualification_canonical_metal_program(package, 128, {}, policy);
    const double create_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - create_started).count();
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) {
        if (const auto* report = std::get_if<CompatibilityReport>(&created))
            std::fprintf(stderr, "generic sparse FFN ownership: code=%u detail=%s\n",
                         static_cast<unsigned>(report->code), report->detail.c_str());
        return test_summary("test_canonical_metal");
    }
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    const uint64_t expected_requested_bytes = expected_source_bytes / total_blocks * selected_blocks;
    CHECK(program.sparse_ffn_layer_count() == expected_layers);
    CHECK(program.sparse_ffn_source_bytes() == expected_source_bytes);
    CHECK(program.sparse_ffn_requested_bytes() == expected_requested_bytes);
    CHECK(program.sparse_ffn_worklist_bytes() ==
          ((device_selector || dense_oracle_selector) ? 2u : static_cast<uint64_t>(selected_blocks)) *
              sizeof(uint32_t));

    const auto run_decode = [&](CanonicalMetalProgram& candidate, CanonicalMetalOutput& output,
                                double& wall_ms, uint32_t token_id = UINT32_MAX) {
        const auto started = std::chrono::steady_clock::now();
        auto executed = candidate.decode(token_id == UINT32_MAX ? package->semantics().bos_id : token_id);
        wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        if (const auto* report = std::get_if<CompatibilityReport>(&executed))
            std::fprintf(stderr, "generic sparse FFN decode: code=%u detail=%s\n",
                         static_cast<unsigned>(report->code), report->detail.c_str());
        if (!std::holds_alternative<CanonicalMetalOutput>(executed)) return false;
        output = std::get<CanonicalMetalOutput>(std::move(executed));
        return true;
    };
    if (device_selector || dense_oracle_selector) {
        uint32_t token = package->semantics().bos_id;
        size_t matched_top1 = 0;
        float maximum_logit_delta = 0.0f;
        double control_gpu_ms = 0.0;
        double selected_gpu_ms = 0.0;
        uint64_t control_peak_session_bytes = 0;
        uint64_t selected_peak_session_bytes = 0;
        size_t evaluated_steps = 0;
        size_t first_mismatch_step = SIZE_MAX;
        uint32_t first_mismatch_token = 0;
        uint32_t first_control_top1 = 0;
        uint32_t first_selected_top1 = 0;
        float first_mismatch_delta = 0.0f;
        std::vector<CanonicalSparseFfnWindowDiagnostic> first_windows;
        std::vector<CanonicalSparseFfnWindowDiagnostic> initial_windows;
        for (size_t step = 0; step != 17; ++step) {
            CanonicalMetalOutput control_output, selected_output;
            double control_wall_ms = 0.0, selected_wall_ms = 0.0;
            const bool control_ok = run_decode(control, control_output, control_wall_ms, token);
            const bool selected_ok = run_decode(program, selected_output, selected_wall_ms, token);
            ++evaluated_steps;
            CHECK(control_ok && selected_ok);
            if (!control_ok || !selected_ok) return test_summary("test_canonical_metal");
            CHECK(control_output.completed && selected_output.completed);
            CHECK(control_output.command_buffers == 1 && selected_output.command_buffers == 1);
            CHECK(control_output.logits.size() == selected_output.logits.size());
            const uint32_t control_top1 = static_cast<uint32_t>(std::max_element(
                control_output.logits.begin(), control_output.logits.end()) - control_output.logits.begin());
            const uint32_t selected_top1 = static_cast<uint32_t>(std::max_element(
                selected_output.logits.begin(), selected_output.logits.end()) - selected_output.logits.begin());
            const bool top1_matches = control_top1 == selected_top1;
            matched_top1 += top1_matches;
            float step_max_delta = 0.0f;
            for (size_t index = 0; index != control_output.logits.size(); ++index) {
                CHECK(std::isfinite(selected_output.logits[index]));
                const float delta = std::abs(selected_output.logits[index] - control_output.logits[index]);
                step_max_delta = std::max(step_max_delta, delta);
                maximum_logit_delta = std::max(maximum_logit_delta, delta);
            }
            if (dense_oracle_selector && step == 0)
                CHECK(canonical_metal_sparse_ffn_windows_for_testing(program, initial_windows));
            if (!top1_matches && first_mismatch_step == SIZE_MAX) {
                first_mismatch_step = step;
                first_mismatch_token = token;
                first_control_top1 = control_top1;
                first_selected_top1 = selected_top1;
                first_mismatch_delta = step_max_delta;
                if (dense_oracle_selector)
                    CHECK(canonical_metal_sparse_ffn_windows_for_testing(program, first_windows));
            }
            if (!dense_oracle_selector)
                CHECK(selected_output.requested_projection_source_bytes ==
                      control_output.requested_projection_source_bytes - expected_source_bytes + expected_requested_bytes);
            control_gpu_ms += control_output.gpu_time_ms;
            selected_gpu_ms += selected_output.gpu_time_ms;
            control_peak_session_bytes = std::max(control_peak_session_bytes,
                                                  control_output.peak_session_bytes);
            selected_peak_session_bytes = std::max(selected_peak_session_bytes,
                                                   selected_output.peak_session_bytes);
            token = control_top1;
            if (dense_oracle_selector && step == 0 && !top1_matches) break;
        }
        CHECK(evaluated_steps == 17 && matched_top1 == 17);
        std::fprintf(stderr,
                     "generic device %s sparse FFN: matched_top1=%zu/%zu selected_blocks=%u layers=%u requested_bytes=%llu create_ms=%.3f control_gpu_ms=%.3f selected_gpu_ms=%.3f control_peak_session_bytes=%llu selected_peak_session_bytes=%llu diagnostic_extra_peak_bytes=%llu maximum_logit_delta=%.9g first_mismatch_step=%zu first_mismatch_token=%u control_top1=%u selected_top1=%u first_mismatch_delta=%.9g one_cb=1\n",
                     dense_oracle_selector ? "two-run-downstream-output-error-oracle" : "proxy",
                     matched_top1, evaluated_steps,
                     selected_blocks, expected_layers,
                     static_cast<unsigned long long>(expected_requested_bytes),
                     create_ms, control_gpu_ms, selected_gpu_ms,
                     static_cast<unsigned long long>(control_peak_session_bytes),
                     static_cast<unsigned long long>(selected_peak_session_bytes),
                     static_cast<unsigned long long>(selected_peak_session_bytes > control_peak_session_bytes
                         ? selected_peak_session_bytes - control_peak_session_bytes : 0),
                     maximum_logit_delta,
                     first_mismatch_step, first_mismatch_token, first_control_top1,
                     first_selected_top1, first_mismatch_delta);
        const std::vector<CanonicalSparseFfnWindowDiagnostic>& reported_windows =
            first_windows.empty() ? initial_windows : first_windows;
        uint64_t overlap_sum = 0;
        uint32_t overlap_minimum = selected_blocks;
        const uint32_t oracle_run_blocks = dense_oracle_selector ? selected_blocks / 2 : selected_blocks;
        for (const CanonicalSparseFfnWindowDiagnostic& window : reported_windows) {
            const auto interval_overlap = [&](uint32_t first, uint32_t count) {
                const uint32_t overlap_first = std::max(window.surrogate_first, first);
                const uint32_t overlap_end = std::min(window.surrogate_first + selected_blocks,
                                                      first + count);
                return overlap_end > overlap_first ? overlap_end - overlap_first : 0u;
            };
            const uint32_t overlap = interval_overlap(window.oracle_first, oracle_run_blocks) +
                                     (dense_oracle_selector
                                          ? interval_overlap(window.oracle_second, oracle_run_blocks)
                                          : 0u);
            overlap_sum += overlap;
            overlap_minimum = std::min(overlap_minimum, overlap);
            std::fprintf(stderr,
                         "generic sparse FFN window: step=%zu layer=%u surrogate_first=%u oracle_first=%u oracle_second=%u overlap=%u/%u\n",
                         first_mismatch_step, window.layer_index, window.surrogate_first,
                         window.oracle_first, window.oracle_second, overlap, selected_blocks);
        }
        if (!reported_windows.empty())
            std::fprintf(stderr,
                         "generic sparse FFN window summary: step=%zu layers=%zu overlap_min=%u/%u overlap_mean=%.3f/%u\n",
                         first_windows.empty() ? 0 : first_mismatch_step, reported_windows.size(),
                         overlap_minimum, selected_blocks,
                         static_cast<double>(overlap_sum) / reported_windows.size(), selected_blocks);
        return test_summary("test_canonical_metal");
    }
    CanonicalMetalOutput control_output, sparse_output;
    double ignored_wall = 0.0;
    const bool control_warm_ok = run_decode(control, control_output, ignored_wall);
    const bool sparse_warm_ok = run_decode(program, sparse_output, ignored_wall);
    CHECK(control_warm_ok);
    CHECK(sparse_warm_ok);
    if (!control_warm_ok || !sparse_warm_ok) return test_summary("test_canonical_metal");
    CHECK(control.position() == 1 && program.position() == 1);
    if (all_blocks) CHECK(control_output.logits == sparse_output.logits);
    const size_t repetitions = 20;
    std::vector<double> control_gpu(repetitions), sparse_gpu(repetitions);
    std::vector<double> control_wall(repetitions), sparse_wall(repetitions);
    std::vector<double> control_qkv(repetitions), sparse_qkv(repetitions);
    std::vector<double> control_attention(repetitions), sparse_attention(repetitions);
    std::vector<double> control_ffn(repetitions), sparse_ffn_time(repetitions);
    std::vector<double> control_final(repetitions), sparse_final(repetitions);
    for (size_t repeat = 0; repeat != repetitions; ++repeat) {
        const bool control_first = repeat % 2 == 0;
        bool control_ok = false;
        bool sparse_ok = false;
        if (control_first) {
            control_ok = run_decode(control, control_output, control_wall[repeat]);
            sparse_ok = run_decode(program, sparse_output, sparse_wall[repeat]);
        } else {
            sparse_ok = run_decode(program, sparse_output, sparse_wall[repeat]);
            control_ok = run_decode(control, control_output, control_wall[repeat]);
        }
        CHECK(control_ok);
        CHECK(sparse_ok);
        if (!control_ok || !sparse_ok) return test_summary("test_canonical_metal");
        control_gpu[repeat] = control_output.gpu_time_ms;
        sparse_gpu[repeat] = sparse_output.gpu_time_ms;
        control_qkv[repeat] = control_output.qkv_gpu_ms;
        sparse_qkv[repeat] = sparse_output.qkv_gpu_ms;
        control_attention[repeat] = control_output.attention_gpu_ms;
        sparse_attention[repeat] = sparse_output.attention_gpu_ms;
        control_ffn[repeat] = control_output.ffn_gpu_ms;
        sparse_ffn_time[repeat] = sparse_output.ffn_gpu_ms;
        control_final[repeat] = control_output.final_gpu_ms;
        sparse_final[repeat] = sparse_output.final_gpu_ms;
        CHECK(control_output.completed && sparse_output.completed);
        CHECK(control_output.command_buffers == 1 && sparse_output.command_buffers == 1);
        CHECK(control.position() == program.position());
        if (all_blocks) CHECK(control_output.logits == sparse_output.logits);
        CHECK(sparse_output.projection_dispatches == control_output.projection_dispatches);
        CHECK(sparse_output.requested_projection_source_bytes == control_output.requested_projection_source_bytes -
                                                        expected_source_bytes + expected_requested_bytes);
        CHECK(sparse_output.peak_session_bytes <= control_output.peak_session_bytes + 4096);
    }
    const auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        const size_t middle = values.size() / 2;
        return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
    };
    const auto range = [](const std::vector<double>& values) {
        return std::minmax_element(values.begin(), values.end());
    };
    const auto control_gpu_range = range(control_gpu);
    const auto sparse_gpu_range = range(sparse_gpu);
    const auto control_wall_range = range(control_wall);
    const auto sparse_wall_range = range(sparse_wall);
    float max_logit_delta = 0.0f;
    for (size_t index = 0; index != sparse_output.logits.size() && index != control_output.logits.size(); ++index)
        max_logit_delta = std::max(max_logit_delta,
                                   std::abs(sparse_output.logits[index] - control_output.logits[index]));
    const size_t control_top1 = static_cast<size_t>(std::max_element(
        control_output.logits.begin(), control_output.logits.end()) - control_output.logits.begin());
    const size_t sparse_top1 = static_cast<size_t>(std::max_element(
        sparse_output.logits.begin(), sparse_output.logits.end()) - sparse_output.logits.begin());
    std::fprintf(stderr,
                 "generic original-span sparse FFN metrics: mode=%s layers=%u source_bytes=%llu requested_bytes=%llu worklist_bytes=%llu create_ms=%.3f warm_pairs=1 measured_pairs=%zu final_position=%u control_gpu_median_ms=%.3f control_gpu_range_ms=%.3f..%.3f sparse_gpu_median_ms=%.3f sparse_gpu_range_ms=%.3f..%.3f control_wall_median_ms=%.3f control_wall_range_ms=%.3f..%.3f sparse_wall_median_ms=%.3f sparse_wall_range_ms=%.3f..%.3f control_projection_bytes=%llu sparse_projection_bytes=%llu peak_session_bytes=%llu max_logit_delta=%.9g control_top1=%zu sparse_top1=%zu\n",
                 all_blocks ? "all-block" : "first-half",
                 expected_layers, static_cast<unsigned long long>(expected_source_bytes),
                 static_cast<unsigned long long>(program.sparse_ffn_requested_bytes()),
                 static_cast<unsigned long long>(program.sparse_ffn_worklist_bytes()), create_ms,
                 repetitions, program.position(),
                 median(control_gpu), *control_gpu_range.first, *control_gpu_range.second,
                 median(sparse_gpu), *sparse_gpu_range.first, *sparse_gpu_range.second,
                 median(control_wall), *control_wall_range.first, *control_wall_range.second,
                 median(sparse_wall), *sparse_wall_range.first, *sparse_wall_range.second,
                 static_cast<unsigned long long>(control_output.requested_projection_source_bytes),
                 static_cast<unsigned long long>(sparse_output.requested_projection_source_bytes),
                 static_cast<unsigned long long>(sparse_output.peak_session_bytes), max_logit_delta,
                 control_top1, sparse_top1);
    if (control_output.counter_samples && sparse_output.counter_samples) {
        std::fprintf(stderr,
                     "generic original-span sparse FFN profile: samples=%llu control_qkv_ms=%.3f sparse_qkv_ms=%.3f control_attention_ms=%.3f sparse_attention_ms=%.3f control_ffn_ms=%.3f sparse_ffn_ms=%.3f control_final_ms=%.3f sparse_final_ms=%.3f\n",
                     static_cast<unsigned long long>(control_output.counter_sample_count),
                     median(control_qkv), median(sparse_qkv),
                     median(control_attention), median(sparse_attention),
                     median(control_ffn), median(sparse_ffn_time),
                     median(control_final), median(sparse_final));
    }
    return test_summary("test_canonical_metal");
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--calibration-target") == 0) {
        test_calibration_target_is_versioned_and_maps_the_linear_k_axis();
        return test_summary("test_canonical_metal_calibration_target");
    }
    if (argc == 2 && std::strcmp(argv[1], "--calibration-cache") == 0) {
        test_calibration_cache_roundtrip_and_key_mismatch();
        return test_summary("test_canonical_metal_calibration_cache");
    }
    if (argc == 2 && std::strcmp(argv[1], "--sparse-layer-mask-policy") == 0) {
        test_sparse_layer_mask_policy_surface();
        return test_summary("test_canonical_metal_sparse_layer_mask_policy");
    }
    if (argc == 2 && std::strcmp(argv[1], "--error-propagation") == 0) {
        test_completed_command_failure_does_not_publish_position_or_kv();
        return test_summary("test_canonical_metal");
    }
    if (argc == 2 && std::strcmp(argv[1], "--canonical-moe") == 0) {
        test_canonical_moe_edge_matcher_is_order_independent();
        test_canonical_moe_routes_the_exact_lapir007_pattern();
        return test_summary("test_canonical_metal");
    }
    if (argc == 2 && std::strcmp(argv[1], "--prefill-batch") == 0) {
        test_two_token_prefill_uses_registered_f16_batch();
        test_failed_two_token_prefill_does_not_publish_position_or_kv();
        return test_summary("test_canonical_metal_prefill_batch");
    }
    if (argc == 2 && std::strcmp(argv[1], "--mixed-attention-geometry") == 0) {
        test_canonical_session_executes_mixed_attention_geometry();
        return test_summary("test_canonical_metal_mixed_attention_geometry");
    }
    if (argc == 2 && std::strcmp(argv[1], "--attention-scale") == 0) {
        test_declared_attention_scale_controls_canonical_prefill_and_decode();
        return test_summary("test_canonical_metal_attention_scale");
    }
    if (argc == 2 && std::strcmp(argv[1], "--grouped-rms") == 0) {
        test_canonical_metal_qk_norm_matches_independent_reference();
        return test_summary("test_canonical_metal_grouped_rms");
    }
    if (argc == 2 && std::strcmp(argv[1], "--embedding-scale") == 0) {
        test_declared_embedding_scale_controls_canonical_decode();
        return test_summary("test_canonical_metal_embedding_scale");
    }
    if (argc == 2 && std::strcmp(argv[1], "--dense-span-boundary") == 0) {
        test_dense_linear_shape_cannot_read_an_adjacent_tensor_span();
        return test_summary("test_canonical_metal_dense_span_boundary");
    }
    if (argc == 2 && std::strcmp(argv[1], "--affine-u2-256-lowering") == 0) {
        test_affine_u2_256_tensor_lowering_requires_all_three_planes();
        test_column_grouped_affine_u2_skip_tensor_lowering_is_byte_typed();
        return test_summary("test_canonical_metal_affine_u2_256_lowering");
    }
    if (argc == 2 && std::strcmp(argv[1], "--affine-u2-256-e2e") == 0) {
        test_canonical_affine_u2_256_end_to_end();
        return test_summary("test_canonical_metal_affine_u2_256_e2e");
    }
    if (argc == 2 && std::strcmp(argv[1], "--column-grouped-u2-canonical") == 0) {
        test_column_grouped_affine_u2_skip_canonical_transaction();
        return test_summary("test_canonical_metal_column_grouped_u2_canonical");
    }
    if (argc == 2 && std::strcmp(argv[1], "--derived-column-grouped-u2-canonical") == 0) {
        test_derived_column_grouped_u2_atlas_canonical_transaction();
        return test_summary("test_canonical_metal_derived_column_grouped_u2_canonical");
    }
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal") == 0) return test_generic_metal_artifact(argv[2]);
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal-incremental") == 0)
        return test_generic_metal_artifact(argv[2], 6);
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal-profile") == 0)
        return test_generic_metal_profile(argv[2]);
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal-preflight") == 0)
        return test_generic_metal_recurrent_preflight(argv[2]);
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal-error-propagation") == 0)
        return test_generic_metal_error_propagation(argv[2]);
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal-derived-q2") == 0)
        return test_generic_metal_derived_q2_ownership(argv[2]);
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal-derived-iq2-xxs-atlas") == 0)
        return test_generic_metal_iq2_xxs_atlas(argv[2]);
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal-ffn-calibration") == 0)
        return test_generic_metal_ffn_calibration(argv[2]);
    if (argc == 5 && std::strcmp(argv[1], "--generic-metal-ffn-calibration-cache") == 0)
        return test_generic_metal_ffn_calibration_cache(argv[2], argv[3], argv[4]);
    if (argc == 5 && std::strcmp(argv[1], "--generic-metal-derived-column-grouped-u2-quality") == 0)
        return test_generic_metal_derived_column_grouped_u2_quality(argv[2], argv[3], argv[4]);
    if (argc == 5 && std::strcmp(argv[1], "--generic-metal-sparse-ffn-calibrated") == 0)
        return test_generic_metal_sparse_ffn_calibrated(argv[2], argv[3], argv[4]);
    if (argc == 5 && std::strcmp(argv[1], "--generic-metal-sparse-ffn-variable-probe") == 0)
        return test_generic_metal_sparse_ffn_calibrated(
            argv[2], argv[3], argv[4], CalibratedSparseMode::VariableWidthProbe);
    if (argc == 5 && std::strcmp(argv[1], "--generic-metal-sparse-ffn-allocate34") == 0)
        return test_generic_metal_sparse_ffn_calibrated(
            argv[2], argv[3], argv[4], CalibratedSparseMode::MarginalAllocate, 34);
    if (argc == 5 && std::strcmp(argv[1], "--generic-metal-sparse-ffn-allocate33") == 0)
        return test_generic_metal_sparse_ffn_calibrated(
            argv[2], argv[3], argv[4], CalibratedSparseMode::MarginalAllocate, 33);
    if (argc == 5 && std::strcmp(argv[1], "--generic-metal-sparse-ffn-allocate32") == 0)
        return test_generic_metal_sparse_ffn_calibrated(
            argv[2], argv[3], argv[4], CalibratedSparseMode::MarginalAllocate, 32);
    if (argc == 5 && std::strcmp(argv[1], "--generic-metal-sparse-ffn-sensitivity34") == 0)
        return test_generic_metal_sparse_ffn_calibrated(
            argv[2], argv[3], argv[4], CalibratedSparseMode::SensitivityAllocate34);
    if (argc == 5 && std::strcmp(argv[1], "--generic-metal-sparse-ffn-sequential34") == 0)
        return test_generic_metal_sparse_ffn_calibrated(
            argv[2], argv[3], argv[4], CalibratedSparseMode::SequentialAllocate34);
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal-iq2-xxs-ffn-quality") == 0)
        return test_generic_metal_iq2_xxs_ffn_quality(argv[2]);
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal-iq2-xxs-zero-atlas") == 0)
        return test_generic_metal_iq2_xxs_zero_atlas(argv[2]);
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal-iq2-xxs-layout") == 0)
        return test_generic_metal_iq2_xxs_layout(argv[2]);
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal-q2-streamed") == 0)
        return test_generic_metal_derived_q2_ownership(argv[2], true);
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal-derived-q2-retained") == 0)
        return test_generic_metal_derived_q2_ownership(argv[2], false, true);
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal-sparse-ffn") == 0)
        return test_generic_metal_sparse_ffn_ownership(argv[2], false);
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal-sparse-ffn-all") == 0)
        return test_generic_metal_sparse_ffn_ownership(argv[2], true);
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal-sparse-ffn-probe") == 0)
        return test_generic_metal_sparse_ffn_ownership(argv[2], false, true);
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal-sparse-ffn-selector") == 0)
        return test_generic_metal_sparse_ffn_ownership(argv[2], false, false, true);
    if (argc == 3 && std::strcmp(argv[1], "--generic-metal-sparse-ffn-oracle") == 0)
        return test_generic_metal_sparse_ffn_ownership(argv[2], false, false, false, true);
    if (!LAPLACE_HAS_QUALIFICATION_GGUF) {
        std::fprintf(stderr, "SKIP: pinned qualification GGUF is unavailable; pass an explicit --generic-metal mode\n");
        return test_summary("test_canonical_metal");
    }
    if (argc == 2 && std::strcmp(argv[1], "--physical-layout") == 0) {
        test_tensor_lowering_preserves_transposed_physical_layout();
        return test_summary("test_canonical_metal");
    }
    if (argc == 2 && std::strcmp(argv[1], "--recurrent-preflight") == 0) {
        test_canonical_metal_lowers_a_planned_recurrent_layer();
        return test_summary("test_canonical_metal");
    }
    test_dense_f16_prefill_uses_one_completed_metal_transaction();
    test_incremental_decode_matches_the_same_full_prefix_oracle();
    test_final_norm_epsilon_does_not_replace_layer_epsilon();
    test_rejects_dense_pattern_without_the_declared_swiglu_activation();
    test_rejects_dense_pattern_without_global_causal_attention();
    test_rejects_dense_pattern_without_declared_key_value_state_ownership();
    test_rejects_unowned_referenced_artifact_without_assuming_artifact_zero();
    test_interleaved_programs_do_not_share_kv_or_weight_resources();
    test_completed_command_failure_does_not_publish_position_or_kv();
    test_prefill_submits_the_prompt_as_one_metal_transaction();
    test_canonical_metal_checkpoint_commit_and_rollback_are_transactional();
    test_planner_selects_only_the_exact_dense_metal_pattern();
    test_canonical_metal_interleaved_rope_matches_independent_reference();
    test_canonical_metal_multisection_rope_matches_independent_reference();
    test_canonical_metal_qk_norm_matches_independent_reference();
    test_canonical_metal_binds_the_exact_query_gate_pattern();
    test_canonical_metal_lowers_a_planned_recurrent_layer();
    test_canonical_metal_constructs_a_q6k_output_projection();
    test_canonical_metal_constructs_a_q6k_token_embedding();
    test_canonical_metal_constructs_a_q4k_token_embedding();
    test_metal_prefill_and_incremental_decode_match_prefixes_one_through_eight();
    test_profile_mode_reports_decode_segments_without_changing_results();
    test_q4k_tensor_lowering_requires_the_exact_block_contract();
    test_tensor_lowering_preserves_transposed_physical_layout();
    return test_summary("test_canonical_metal");
}
