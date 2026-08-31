#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <sys/stat.h>
#include <variant>
#include <vector>

#include "artifact_index.h"
#include "artifact_set.h"
#include "codec_certificate.h"
#include "gguf_index.h"
#include "gguf_writer.h"
#include "mlx_package.h"
#include "product_package.h"
#include "semantic_manifest.h"
#include "source_schema.h"
#include "tensor.h"
#include "test_util.h"
#include "token_program.h"

using namespace Laplace;

namespace {

size_t g_product_advice_calls = 0;
size_t g_product_advice_bytes = 0;

void record_product_advice(size_t, size_t bytes) {
    ++g_product_advice_calls;
    g_product_advice_bytes += bytes;
}

std::string temporary_path() {
    char path[] = "/private/tmp/laplace-product-package-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd >= 0) {
        const std::array<uint8_t, 8> bytes = {'G', 'G', 'U', 'F', 3, 0, 0, 0};
        CHECK(write(fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size()));
        close(fd);
    }
    return path;
}

bool write_bytes(const std::string& path, std::span<const uint8_t> bytes) {
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(fd >= 0);
    if (fd < 0) return false;
    const ssize_t written = write(fd, bytes.data(), bytes.size());
    close(fd);
    CHECK(written == static_cast<ssize_t>(bytes.size()));
    return written == static_cast<ssize_t>(bytes.size());
}

bool write_text(const std::string& path, const std::string& text) {
    return write_bytes(path, std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(text.data()), text.size()));
}

bool write_tiny_gguf(const std::string& path, uint8_t seed) {
    gguf_writer::Writer writer;
    writer.kv_u32("general.quantization_version", 2);
    gguf_writer::TensorDecl tensor;
    tensor.name = "token_embd.weight";
    tensor.dims = {256, 1};
    tensor.type = static_cast<uint32_t>(GGMLType::Q4_K);
    tensor.data.resize(bytes_per_block(GGMLType::Q4_K));
    std::fill(tensor.data.begin(), tensor.data.end(), seed);
    writer.add_tensor(std::move(tensor));
    return writer.write_file(path);
}

gguf_writer::TensorDecl f32_tensor(const char* name,
                                   std::vector<uint64_t> dimensions,
                                   float value = 0.0f) {
    size_t count = 1;
    for (const uint64_t dimension : dimensions) count *= static_cast<size_t>(dimension);
    gguf_writer::TensorDecl tensor;
    tensor.name = name;
    tensor.dims = std::move(dimensions);
    tensor.type = static_cast<uint32_t>(GGMLType::F32);
    tensor.data.resize(count * sizeof(float));
    for (size_t index = 0; index != count; ++index) {
        std::memcpy(tensor.data.data() + index * sizeof(float), &value, sizeof(value));
    }
    return tensor;
}

gguf_writer::TensorDecl blocked_tensor(const char* name,
                                       std::vector<uint64_t> dimensions,
                                       GGMLType type) {
    const size_t rows = dimensions.size() == 2 ? static_cast<size_t>(dimensions[1]) : 1;
    gguf_writer::TensorDecl tensor;
    tensor.name = name;
    tensor.dims = std::move(dimensions);
    tensor.type = static_cast<uint32_t>(type);
    tensor.data.resize(rows * bytes_per_block(type));
    return tensor;
}

enum class RawTokenizerVariant {
    Supported,
    MissingSemanticRecord,
    SemanticRecordBindingMismatch,
    SemanticRecordWrongWireType,
    MergeTable,
    MissingTokenTypes,
    InvalidByteBasis,
    DynamicTemplate,
    ExtraTokenizerMetadata,
};

struct DenseSourceDeclaration {
    SourceSchema schema;
    uint32_t maximum_context = 32768;
    uint32_t attention_heads = 1;
    uint32_t attention_kv_heads = 1;
    uint32_t head_dimension = 256;
    uint32_t rotary_dimension = 256;
    float rope_base = 10000.0f;
    float rms_epsilon = 1.0e-5f;
    uint32_t bos_id = 256;
    uint32_t eos_id = 257;
};

SourceNamePattern exact_source_name(const char* spelling) {
    SourceNamePattern pattern;
    pattern.kind = SourceNamePatternKind::Exact;
    pattern.literal = spelling;
    return pattern;
}

DenseSourceDeclaration dense_source_declaration() {
    constexpr uint32_t kHidden = 256;
    constexpr uint32_t kVocabulary = 258;
    constexpr uint32_t kIntermediate = 256;
    DenseSourceDeclaration declaration;
    declaration.schema.source_format = SourceFormat::Gguf;
    declaration.schema.slots = {
        {0, 0, {kHidden, kVocabulary}, true, false},
        {1, 1, {kHidden}, true, false},
        {2, 2, {kHidden, kVocabulary}, true, true},
        {3, 3, {kHidden}, true, false},
        {4, 4, {kHidden, kHidden}, true, false},
        {5, 5, {kHidden, kHidden}, true, false},
        {6, 6, {kHidden, kHidden}, true, false},
        {7, 7, {kHidden, kHidden}, true, false},
        {8, 8, {kHidden}, true, false},
        {9, 9, {kHidden, kIntermediate}, true, false},
        {10, 10, {kHidden, kIntermediate}, true, false},
        {11, 11, {kIntermediate, kHidden}, true, false},
    };
    declaration.schema.selectors = {
        {0, 0, {}, {exact_source_name("token_embd.weight")}, {kHidden, kVocabulary}, 1},
        {1, 1, {}, {exact_source_name("output_norm.weight")}, {kHidden}, 1},
        {2, 2, {}, {exact_source_name("output.weight")}, {kHidden, kVocabulary}, 1},
        {3, 3, {}, {exact_source_name("blk.0.attn_norm.weight")}, {kHidden}, 1},
        {4, 4, {}, {exact_source_name("blk.0.attn_q.weight")}, {kHidden, kHidden}, 1},
        {5, 5, {}, {exact_source_name("blk.0.attn_k.weight")}, {kHidden, kHidden}, 1},
        {6, 6, {}, {exact_source_name("blk.0.attn_v.weight")}, {kHidden, kHidden}, 1},
        {7, 7, {}, {exact_source_name("blk.0.attn_output.weight")}, {kHidden, kHidden}, 1},
        {8, 8, {}, {exact_source_name("blk.0.post_attention_norm.weight")}, {kHidden}, 1},
        {9, 9, {}, {exact_source_name("blk.0.ffn_gate.weight")}, {kHidden, kIntermediate}, 1},
        {10, 10, {}, {exact_source_name("blk.0.ffn_up.weight")}, {kHidden, kIntermediate}, 1},
        {11, 11, {}, {exact_source_name("blk.0.ffn_down.weight")}, {kIntermediate, kHidden}, 1},
    };
    return declaration;
}

constexpr std::array<TensorRole, 12> kDenseTensorRoles = {
    TensorRole::TokenEmbedding,
    TensorRole::FinalNormWeight,
    TensorRole::OutputWeight,
    TensorRole::AttentionNormWeight,
    TensorRole::QueryWeight,
    TensorRole::KeyWeight,
    TensorRole::ValueWeight,
    TensorRole::AttentionOutputWeight,
    TensorRole::FfnNormWeight,
    TensorRole::FfnGateWeight,
    TensorRole::FfnUpWeight,
    TensorRole::FfnDownWeight,
};

uint32_t f32_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::string physical_tensor_spelling(const ArtifactIndex& physical, uint32_t tensor_id) {
    for (const ArtifactDiagnostic& diagnostic : physical.diagnostics()) {
        if (diagnostic.tensor_id == tensor_id && !diagnostic.tensor_spelling.empty()) {
            return diagnostic.tensor_spelling;
        }
    }
    return {};
}

const ArtifactTensorRecord* declared_tensor_for_slot(const DenseSourceDeclaration& declaration,
                                                     const ArtifactIndex& physical,
                                                     uint32_t slot_id) {
    const SchemaLogicalSlot* slot = nullptr;
    for (const SchemaLogicalSlot& candidate : declaration.schema.slots) {
        if (candidate.id == slot_id) {
            slot = &candidate;
            break;
        }
    }
    if (!slot) return nullptr;
    const SchemaTensorSelector* selector = nullptr;
    for (const SchemaTensorSelector& candidate : declaration.schema.selectors) {
        if (candidate.id == slot->selector_id) {
            selector = &candidate;
            break;
        }
    }
    if (!selector) return nullptr;
    for (const ArtifactTensorRecord& tensor : physical.tensors()) {
        if (tensor.logical_dimensions != selector->source_dimensions) continue;
        const std::string spelling = physical_tensor_spelling(physical, tensor.id);
        for (const SourceNamePattern& pattern : selector->names) {
            if (pattern.kind == SourceNamePatternKind::Exact && pattern.literal == spelling) {
                return &tensor;
            }
        }
    }
    return nullptr;
}

SemanticModel manual_dense_semantic_model(const DenseSourceDeclaration& declaration,
                                          const ArtifactIndex* physical = nullptr) {
    const SourceSchema& schema = declaration.schema;
    const uint32_t hidden = static_cast<uint32_t>(schema.slots[0].dimensions[0]);
    const uint32_t vocabulary = static_cast<uint32_t>(schema.slots[0].dimensions[1]);
    const uint32_t intermediate = static_cast<uint32_t>(schema.slots[9].dimensions[1]);

    SemanticModel model;
    model.maximum_context = declaration.maximum_context;
    model.vocabulary_size = vocabulary;
    model.bos_id = declaration.bos_id;
    model.eos_id = declaration.eos_id;
    model.tensors.resize(schema.slots.size());

    for (size_t slot_index = 0; slot_index != schema.slots.size(); ++slot_index) {
        const SchemaLogicalSlot& slot = schema.slots[slot_index];
        const ArtifactTensorRecord* source = physical
            ? declared_tensor_for_slot(declaration, *physical, slot.id) : nullptr;
        const uint32_t tensor_id = source ? source->id : slot.id;
        CHECK(tensor_id < model.tensors.size());
        if (tensor_id >= model.tensors.size()) continue;
        SemanticTensor tensor;
        tensor.id = tensor_id;
        tensor.role = kDenseTensorRoles[slot.id];
        for (uint64_t dimension : slot.dimensions) {
            tensor.dimensions.push_back({DimensionKind::Constant, dimension});
        }
        if (physical) {
            CHECK(source != nullptr);
            if (source) {
                CHECK(source->logical_dimensions == slot.dimensions);
                tensor.logical_type = source->logical_type == ArtifactScalarType::F16
                    ? ScalarType::F16 : ScalarType::F32;
                tensor.layout = source->layout;
                tensor.quantization = source->quantization;
                CHECK(source->coordinate.bank_axis == UINT8_MAX);
                tensor.expert_axis = {};
                for (const ArtifactTensorPlane& source_plane : source->planes) {
                    const ScalarType storage_type = source_plane.storage_type == ArtifactScalarType::Packed
                        ? ScalarType::U8 : static_cast<ScalarType>(source_plane.storage_type);
                    tensor.planes.push_back({source_plane.kind, storage_type, ArtifactId{0},
                                             source_plane.source.offset, source_plane.source.length,
                                             source_plane.alignment, 0});
                }
            }
        } else {
            tensor.logical_type = ScalarType::F32;
            tensor.layout.rank = static_cast<uint8_t>(tensor.dimensions.size());
            for (size_t axis = 0; axis != tensor.dimensions.size(); ++axis) {
                tensor.layout.axis_order[axis] = static_cast<uint8_t>(axis);
                tensor.layout.strides[axis] = 1;
            }
            tensor.planes.push_back({PlaneKind::Values, ScalarType::F32, ArtifactId{0}, 0, 1, 32, 0});
        }
        model.tensors[tensor_id] = std::move(tensor);
    }

    auto add_value = [&](uint64_t width, ScalarType type = ScalarType::F32) {
        SemanticValue value;
        value.id = static_cast<uint32_t>(model.values.size());
        value.logical_type = type;
        value.dimensions = {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, width}};
        model.values.push_back(std::move(value));
        return model.values.back().id;
    };
    auto add_operator = [&](OperatorKind kind, std::vector<uint32_t> inputs,
                            std::vector<uint32_t> outputs, std::vector<uint32_t> tensors,
                            std::vector<uint32_t> states, OperatorPayload payload) {
        SemanticOperator op;
        op.id = static_cast<uint32_t>(model.operators.size());
        op.kind = kind;
        op.semantic_version = model.schema_major;
        op.inputs = std::move(inputs);
        op.outputs = std::move(outputs);
        op.tensors = std::move(tensors);
        op.states = std::move(states);
        op.payload = std::move(payload);
        model.operators.push_back(std::move(op));
    };
    auto add_state = [&](StateKind kind, StateUpdateKind update) {
        StateFormat format;
        format.logical_domain = TransformDomain::Untransformed;
        format.encoded_domain = kind == StateKind::KeyCache
            ? TransformDomain::RopeApplied : TransformDomain::Untransformed;
        format.alignment = 64;
        SemanticState state;
        state.id = static_cast<uint32_t>(model.states.size());
        state.kind = kind;
        state.semantic_version = model.schema_major;
        state.update_kind = update;
        state.position_policy = PositionPolicy::AppendOnly;
        state.dimensions = {{DimensionKind::Symbol, 1},
                            {DimensionKind::Constant, declaration.attention_kv_heads},
                            {DimensionKind::Constant, declaration.head_dimension}};
        state.formats = {format};
        model.states.push_back(std::move(state));
        return model.states.back().id;
    };

    const uint32_t embedding = 0;
    const uint32_t final_norm = 1;
    const uint32_t output = 2;
    const uint32_t attn_norm = 3;
    const uint32_t query = 4;
    const uint32_t key = 5;
    const uint32_t value = 6;
    const uint32_t attention_output = 7;
    const uint32_t ffn_norm = 8;
    const uint32_t ffn_gate = 9;
    const uint32_t ffn_up = 10;
    const uint32_t ffn_down = 11;
    const uint32_t token_ids = add_value(1, ScalarType::U32);
    const uint32_t embedded = add_value(hidden);
    model.input_values_first = token_ids;
    model.input_values_count = 1;
    add_operator(OperatorKind::EmbeddingLookup, {token_ids}, {embedded}, {embedding}, {},
                 EmbeddingLookupPayload{f32_bits(1.0f), vocabulary, hidden, 0});

    const uint32_t normalized = add_value(hidden);
    const uint32_t q = add_value(declaration.attention_heads * declaration.head_dimension);
    const uint32_t k = add_value(declaration.attention_kv_heads * declaration.head_dimension);
    const uint32_t v = add_value(declaration.attention_kv_heads * declaration.head_dimension);
    const uint32_t rotated_q = add_value(declaration.attention_heads * declaration.head_dimension);
    const uint32_t rotated_k = add_value(declaration.attention_kv_heads * declaration.head_dimension);
    const uint32_t context = add_value(declaration.attention_heads * declaration.head_dimension);
    const uint32_t projected = add_value(hidden);
    const uint32_t residual = add_value(hidden);
    const uint32_t ffn_normalized = add_value(hidden);
    const uint32_t gate = add_value(intermediate);
    const uint32_t up = add_value(intermediate);
    const uint32_t activated = add_value(intermediate);
    const uint32_t down = add_value(hidden);
    const uint32_t next = add_value(hidden);
    const uint32_t final_normalized = add_value(hidden);
    const uint32_t logits = add_value(vocabulary);
    const uint32_t key_state = add_state(StateKind::KeyCache, StateUpdateKind::AppendKey);
    const uint32_t value_state = add_state(StateKind::ValueCache, StateUpdateKind::AppendValue);
    const uint32_t attention_scale = f32_bits(
        1.0f / std::sqrt(static_cast<float>(declaration.head_dimension)));
    const uint32_t rms_epsilon = f32_bits(declaration.rms_epsilon);
    add_operator(OperatorKind::RmsNorm, {embedded}, {normalized}, {attn_norm}, {},
                 RmsNormPayload{rms_epsilon, -1, 1});
    add_operator(OperatorKind::Linear, {normalized}, {q}, {query}, {}, LinearPayload{});
    add_operator(OperatorKind::Linear, {normalized}, {k}, {key}, {}, LinearPayload{});
    add_operator(OperatorKind::Linear, {normalized}, {v}, {value}, {}, LinearPayload{});
    add_operator(OperatorKind::Rope, {q, k}, {rotated_q, rotated_k}, {}, {},
                 RopePayload{RopePairing::HalfSplit, true, declaration.rotary_dimension,
                             f32_bits(declaration.rope_base), f32_bits(1.0f), {}});
    add_operator(OperatorKind::CausalAttention, {rotated_q, rotated_k, v}, {context}, {},
                 {key_state, value_state},
                 CausalAttentionPayload{declaration.attention_heads, declaration.attention_kv_heads,
                                        declaration.head_dimension, attention_scale,
                                        AttentionMask::Causal, CachePolicy::Global});
    add_operator(OperatorKind::Linear, {context}, {projected}, {attention_output}, {}, LinearPayload{});
    add_operator(OperatorKind::Add, {embedded, projected}, {residual}, {}, {}, AddPayload{});
    add_operator(OperatorKind::RmsNorm, {residual}, {ffn_normalized}, {ffn_norm}, {},
                 RmsNormPayload{rms_epsilon, -1, 1});
    add_operator(OperatorKind::Linear, {ffn_normalized}, {gate}, {ffn_gate}, {}, LinearPayload{});
    add_operator(OperatorKind::Linear, {ffn_normalized}, {up}, {ffn_up}, {}, LinearPayload{});
    add_operator(OperatorKind::SwiGlu, {gate, up}, {activated}, {}, {},
                 SwiGluPayload{ActivationKind::Silu});
    add_operator(OperatorKind::Linear, {activated}, {down}, {ffn_down}, {}, LinearPayload{});
    add_operator(OperatorKind::Add, {residual, down}, {next}, {}, {}, AddPayload{});
    model.layers = {{0, 1, 14, 0}};
    add_operator(OperatorKind::RmsNorm, {next}, {final_normalized}, {final_norm}, {},
                 RmsNormPayload{rms_epsilon, -1, 1});
    add_operator(OperatorKind::Linear, {final_normalized}, {logits}, {output}, {}, LinearPayload{});
    model.output_values_first = logits;
    model.output_values_count = 1;
    return model;
}

bool write_compilable_raw_gguf(
    const std::string& path,
    RawTokenizerVariant variant = RawTokenizerVariant::Supported) {
    const DenseSourceDeclaration declaration = dense_source_declaration();
    const auto write = [&](std::span<const uint8_t> semantic_record) {
        constexpr uint32_t kHidden = 256;
        constexpr uint32_t kVocabulary = 258;
        gguf_writer::Writer writer;
        writer.kv_str("general.architecture", "fixture");
        writer.kv_u32("general.quantization_version", 2);
        writer.kv_u32("fixture.block_count", 1);
        writer.kv_u32("fixture.context_length", declaration.maximum_context);
        writer.kv_u32("fixture.embedding_length", kHidden);
        writer.kv_u32("fixture.feed_forward_length", kHidden);
        writer.kv_u32("fixture.attention.head_count", declaration.attention_heads);
        writer.kv_u32("fixture.attention.head_count_kv", declaration.attention_kv_heads);
        writer.kv_u32("fixture.attention.key_length", declaration.head_dimension);
        writer.kv_u32("fixture.attention.value_length", declaration.head_dimension);
        writer.kv_u32("fixture.rope.dimension_count", declaration.rotary_dimension);
        writer.kv_f32("fixture.rope.freq_base", declaration.rope_base);
        writer.kv_f32("fixture.attention.layer_norm_rms_epsilon", declaration.rms_epsilon);
        if (!semantic_record.empty()) {
            if (variant == RawTokenizerVariant::SemanticRecordWrongWireType) {
                writer.kv_arr_u32("laplace.semantic_model",
                                  std::vector<uint32_t>(semantic_record.begin(), semantic_record.end()));
            } else {
                writer.kv_arr_u8("laplace.semantic_model",
                                 std::vector<uint8_t>(semantic_record.begin(), semantic_record.end()));
            }
        }
        writer.kv_str("tokenizer.ggml.model", "gpt2");
        std::vector<std::string> tokens;
        std::vector<int32_t> token_types;
        tokens.reserve(kVocabulary);
        token_types.reserve(kVocabulary);
        for (uint32_t byte = 0; byte != 256; ++byte) {
            tokens.emplace_back(1, static_cast<char>(byte));
            token_types.push_back(6);
        }
        if (variant == RawTokenizerVariant::InvalidByteBasis) tokens[0] = "xx";
        tokens.emplace_back("<eos>");
        tokens.emplace_back("<bos>");
        token_types.push_back(3);
        token_types.push_back(3);
        writer.kv_arr_str("tokenizer.ggml.tokens", tokens);
        if (variant != RawTokenizerVariant::MissingTokenTypes) {
            writer.kv_arr_i32("tokenizer.ggml.token_type", token_types);
        }
        if (variant == RawTokenizerVariant::MergeTable) writer.kv_arr_str("tokenizer.ggml.merges", {"a a"});
        writer.kv_u32("tokenizer.ggml.eos_token_id", declaration.eos_id);
        writer.kv_u32("tokenizer.ggml.bos_token_id", declaration.bos_id);
        writer.kv_bool("tokenizer.ggml.add_bos_token", false);
        writer.kv_bool("tokenizer.ggml.add_eos_token", false);
        if (variant == RawTokenizerVariant::ExtraTokenizerMetadata) {
            writer.kv_bool("tokenizer.ggml.add_space_prefix", true);
        }
        writer.kv_str("tokenizer.chat_template",
                      variant == RawTokenizerVariant::DynamicTemplate
                          ? "{% for message in messages %}{{ message['content'] }}{% endfor %}a"
                          : "{{ messages[0]['content'] }}a");
        writer.add_tensor(blocked_tensor("token_embd.weight", {kHidden, kVocabulary}, GGMLType::Q4_K));
        writer.add_tensor(f32_tensor("output_norm.weight", {kHidden}, 1.0f));
        writer.add_tensor(blocked_tensor("output.weight", {kHidden, kVocabulary}, GGMLType::Q6_K));
        writer.add_tensor(f32_tensor("blk.0.attn_norm.weight", {kHidden}, 1.0f));
        writer.add_tensor(blocked_tensor("blk.0.attn_q.weight", {kHidden, kHidden}, GGMLType::Q4_K));
        writer.add_tensor(blocked_tensor("blk.0.attn_k.weight", {kHidden, kHidden}, GGMLType::Q4_K));
        writer.add_tensor(blocked_tensor("blk.0.attn_v.weight", {kHidden, kHidden}, GGMLType::Q4_K));
        writer.add_tensor(blocked_tensor("blk.0.attn_output.weight", {kHidden, kHidden}, GGMLType::Q4_K));
        writer.add_tensor(f32_tensor("blk.0.post_attention_norm.weight", {kHidden}, 1.0f));
        writer.add_tensor(blocked_tensor("blk.0.ffn_gate.weight", {kHidden, kHidden}, GGMLType::Q4_K));
        writer.add_tensor(blocked_tensor("blk.0.ffn_up.weight", {kHidden, kHidden}, GGMLType::Q4_K));
        writer.add_tensor(blocked_tensor("blk.0.ffn_down.weight", {kHidden, kHidden}, GGMLType::Q4_K));
        return writer.write_file(path);
    };

    if (variant != RawTokenizerVariant::Supported &&
        variant != RawTokenizerVariant::SemanticRecordBindingMismatch &&
        variant != RawTokenizerVariant::SemanticRecordWrongWireType) {
        return write({});
    }

    const auto placeholder = encode_semantic_model(manual_dense_semantic_model(declaration));
    CHECK(std::holds_alternative<std::vector<uint8_t>>(placeholder));
    if (!std::holds_alternative<std::vector<uint8_t>>(placeholder)) return false;
    const std::vector<uint8_t>& placeholder_bytes = std::get<std::vector<uint8_t>>(placeholder);
    if (!write(placeholder_bytes)) return false;
    auto source = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(source));
    if (!std::holds_alternative<ArtifactSet>(source)) return false;
    auto package = std::get<ArtifactSet>(std::move(source)).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(package));
    if (!std::holds_alternative<PackageView>(package)) return false;
    auto physical_result = build_gguf_artifact_index(std::get<PackageView>(package));
    CHECK(std::holds_alternative<ArtifactIndex>(physical_result));
    if (!std::holds_alternative<ArtifactIndex>(physical_result)) return false;
    ArtifactIndex physical = std::get<ArtifactIndex>(std::move(physical_result));
    SemanticModel semantic_model = manual_dense_semantic_model(declaration, &physical);
    auto encoded = encode_semantic_model(semantic_model);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
    if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) return false;
    const std::vector<uint8_t>& encoded_bytes = std::get<std::vector<uint8_t>>(encoded);
    CHECK(encoded_bytes.size() == placeholder_bytes.size());
    if (encoded_bytes.size() != placeholder_bytes.size()) return false;
    if (variant == RawTokenizerVariant::SemanticRecordBindingMismatch) {
        CHECK(!semantic_model.tensors.empty() && !semantic_model.tensors[0].planes.empty());
        if (semantic_model.tensors.empty() || semantic_model.tensors[0].planes.empty()) return false;
        ++semantic_model.tensors[0].planes[0].length;
        encoded = encode_semantic_model(semantic_model);
        CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
        if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) return false;
    }
    const std::vector<uint8_t>& final_record = std::get<std::vector<uint8_t>>(encoded);
    if (!write(final_record)) return false;
    auto final_source = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(final_source));
    if (!std::holds_alternative<ArtifactSet>(final_source)) return false;
    auto final_package = std::get<ArtifactSet>(std::move(final_source)).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(final_package));
    if (!std::holds_alternative<PackageView>(final_package)) return false;
    auto final_physical_result = build_gguf_artifact_index(std::get<PackageView>(final_package));
    CHECK(std::holds_alternative<ArtifactIndex>(final_physical_result));
    if (!std::holds_alternative<ArtifactIndex>(final_physical_result)) return false;
    const ArtifactIndex& final_physical = std::get<ArtifactIndex>(final_physical_result);
    CHECK(final_physical.tensors().size() == physical.tensors().size());
    if (final_physical.tensors().size() != physical.tensors().size()) return false;
    for (size_t index = 0; index != physical.tensors().size(); ++index) {
        CHECK(final_physical.tensors()[index] == physical.tensors()[index]);
    }
    return true;
}

std::vector<uint8_t> token_program_bytes(uint32_t vocabulary_size = 256) {
    TokenProgramDefinition definition;
    for (unsigned value = 0; value != 256; ++value) {
        definition.byte_map[value] = static_cast<uint8_t>(value);
    }
    definition.vocabulary.reserve(vocabulary_size);
    for (uint32_t token = 0; token != vocabulary_size; ++token) {
        VocabEntry entry;
        if (token < 128) {
            entry.piece.assign(1, static_cast<char>(token));
        } else {
            entry.piece = "<reserved-" + std::to_string(token) + ">";
            entry.flags = static_cast<uint16_t>(VocabFlags::Special);
        }
        definition.vocabulary.push_back(std::move(entry));
    }
    definition.vocabulary[1].flags = static_cast<uint16_t>(VocabFlags::Special);
    definition.vocabulary[2].flags = static_cast<uint16_t>(VocabFlags::Special);
    definition.unknown_token_id = 0;
    definition.normalizer.kind = NormalizerKind::None;
    definition.pretokenizer.kind = PretokenizerKind::ByteLevel;
    definition.postprocessor.kind = PostprocessorKind::AddBosEos;
    definition.postprocessor.flags = static_cast<uint8_t>(PostprocessorFlags::AddBos);
    definition.postprocessor.bos_token_id = 2;
    definition.decoder.kind = DecoderKind::ByteLevel;
    definition.decoder.flags = static_cast<uint8_t>(DecoderFlags::SkipSpecial);
    definition.prompt = {
        {PromptOpcode::EmitUserText, {}},
        {PromptOpcode::EmitGenerationPrompt, "\n"},
        {PromptOpcode::End, {}},
    };
    const auto serialized = serialize_token_program(definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    return std::holds_alternative<std::vector<uint8_t>>(serialized)
        ? std::get<std::vector<uint8_t>>(serialized)
        : std::vector<uint8_t>{};
}

ArtifactIndex add_token_artifact(ArtifactIndex physical, const PackageView& token_data) {
    ArtifactIndexInput input;
    input.artifacts.assign(physical.artifacts().begin(), physical.artifacts().end());
    input.artifacts.push_back(token_data);
    input.metadata_facts.assign(physical.metadata_facts().begin(), physical.metadata_facts().end());
    input.package_facts.assign(physical.package_facts().begin(), physical.package_facts().end());
    input.tensors.assign(physical.tensors().begin(), physical.tensors().end());
    input.aliases.assign(physical.aliases().begin(), physical.aliases().end());
    input.diagnostics.assign(physical.diagnostics().begin(), physical.diagnostics().end());
    auto augmented = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<ArtifactIndex>(augmented));
    return std::holds_alternative<ArtifactIndex>(augmented)
        ? std::get<ArtifactIndex>(std::move(augmented))
        : ArtifactIndex{};
}

PhysicalCodecRegistry codec_registry_for_model(const SemanticModel& model) {
    PhysicalCodecRegistry registry;
    for (const SemanticTensor& tensor : model.tensors) {
        std::vector<uint8_t> certificate_bytes;
        if (tensor.quantization.kind == QuantizationKind::None &&
            tensor.planes.size() == 1 && tensor.planes[0].storage_type == ScalarType::F16) {
            certificate_bytes = make_raw_f16_codec_certificate();
        } else if (tensor.quantization.kind == QuantizationKind::None &&
                   tensor.planes.size() == 1 && tensor.planes[0].storage_type == ScalarType::F32) {
            certificate_bytes = make_raw_f32_codec_certificate();
        } else if (tensor.layout.kind == PhysicalLayoutKind::GgufBlocked &&
                   tensor.layout.block_elements == 256 && tensor.layout.block_bytes == 144) {
            certificate_bytes = make_q4_k_codec_certificate();
        } else if (tensor.layout.kind == PhysicalLayoutKind::GgufBlocked &&
                   tensor.layout.block_elements == 256 && tensor.layout.block_bytes == 210) {
            certificate_bytes = make_q6_k_codec_certificate();
        } else {
            CHECK_MSG(false, "product fixture tensor has no codec certificate: id=%u", tensor.id);
            return {};
        }
        auto parsed = parse_codec_certificate(certificate_bytes);
        const auto* certificate = std::get_if<CodecCertificate>(&parsed);
        CHECK(certificate != nullptr);
        if (!certificate) return {};
        auto identity = physical_codec_identity(
            tensor, certificate->identity().abi_version, certificate->identity().digest);
        CHECK(identity.has_value());
        if (!identity) return {};
        if (std::none_of(registry.codecs.begin(), registry.codecs.end(),
                         [&](const PhysicalCodecSpec& candidate) {
                             return candidate.identity == *identity;
                         })) {
            registry.codecs.push_back({*identity, certificate_bytes});
        }
        registry.tensors.push_back({tensor.id, *identity});
    }
    std::sort(registry.codecs.begin(), registry.codecs.end(),
              [](const auto& left, const auto& right) {
                  return physical_codec_identity_less(left.identity, right.identity);
              });
    std::sort(registry.tensors.begin(), registry.tensors.end(),
              [](const auto& left, const auto& right) {
                  return left.tensor_id < right.tensor_id;
              });
    return registry;
}

enum class TokenAuthorityMismatch {
    None,
    Bos,
    VocabularyDigest,
    PromptOperations,
    ProgramVersion,
};

std::vector<uint8_t> manifest_for_gguf(const std::string& path,
                                       const std::string& token_path,
                                       bool token_ids_only = false,
                                       TokenAuthorityMismatch mismatch = TokenAuthorityMismatch::None,
                                       bool include_codec_registry = true) {
    const std::array<ArtifactSource, 2> sources = {
        ArtifactSource{path, ArtifactRole::Primary, ArtifactId{0}},
        ArtifactSource{token_path, ArtifactRole::Shard, ArtifactId{2}},
    };
    auto artifacts = ArtifactSet::load_graph(sources);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return {};
    auto* set = std::get_if<ArtifactSet>(&artifacts);
    auto view = set->view(ArtifactId{0});
    auto token_view = set->view(ArtifactId{2});
    CHECK(std::holds_alternative<PackageView>(view));
    CHECK(std::holds_alternative<PackageView>(token_view));
    if (!std::holds_alternative<PackageView>(view) ||
        !std::holds_alternative<PackageView>(token_view)) return {};
    const PackageView& package = std::get<PackageView>(view);
    const PackageView& token_data = std::get<PackageView>(token_view);
    auto compiled_program = TokenProgram::compile(token_data.bytes());
    CHECK(std::holds_alternative<TokenProgram>(compiled_program));
    if (!std::holds_alternative<TokenProgram>(compiled_program)) return {};
    const TokenProgram& program = std::get<TokenProgram>(compiled_program);
    auto physical_result = build_gguf_artifact_index(package);
    CHECK(std::holds_alternative<ArtifactIndex>(physical_result));
    if (!std::holds_alternative<ArtifactIndex>(physical_result)) return {};
    ArtifactIndex physical = add_token_artifact(
        std::get<ArtifactIndex>(std::move(physical_result)), token_data);
    CHECK(physical.tensors().size() == 1);
    if (physical.tensors().size() != 1) return {};

    const ArtifactTensorRecord& source = physical.tensors()[0];
    SemanticModel model;
    model.maximum_context = 32768;
    model.vocabulary_size = 256;
    model.bos_id = mismatch == TokenAuthorityMismatch::Bos ? 3 : 2;
    model.eos_id = 1;
    model.stop_ids = {1};
    model.tokenizer_digest = token_data.digest().bytes;
    model.template_digest = program.prompt_digest().bytes;

    SemanticTensor tensor;
    tensor.id = source.id;
    tensor.role = TensorRole::TokenEmbedding;
    tensor.logical_type = static_cast<ScalarType>(source.logical_type);
    tensor.layout = source.layout;
    tensor.quantization = source.quantization;
    for (uint64_t dimension : source.logical_dimensions) {
        tensor.dimensions.push_back({DimensionKind::Constant, dimension});
    }
    for (const ArtifactTensorPlane& source_plane : source.planes) {
        tensor.planes.push_back({source_plane.kind,
                                 source_plane.storage_type == ArtifactScalarType::Packed
                                     ? ScalarType::U8
                                     : static_cast<ScalarType>(source_plane.storage_type),
                                 source_plane.source.artifact_id,
                                 source_plane.source.offset,
                                 source_plane.source.length,
                                 source_plane.alignment,
                                 0});
    }
    model.tensors.push_back(std::move(tensor));
    model.values = {
        {0, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0},
        {1, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0},
    };
    model.input_values_first = 0;
    model.input_values_count = 1;
    model.output_values_first = 1;
    model.output_values_count = 1;
    model.operators = {
        {0, OperatorKind::EmbeddingLookup, 1, {0}, {1}, {0}, {},
         EmbeddingLookupPayload{0x3f800000u, model.vocabulary_size, 1, 0}},
    };
    model.layers = {{0, 0, 1, 0}};

    TokenContract contract;
    contract.vocabulary_size = model.vocabulary_size;
    contract.bos_id = model.bos_id;
    contract.eos_id = model.eos_id;
    contract.stop_ids = model.stop_ids;
    contract.authoritative_tokenizer_digest = {model.tokenizer_digest};
    contract.authoritative_template_digest = {model.template_digest};
    if (!token_ids_only) {
        contract.tokenizer_algorithm = TokenizerAlgorithm::ByteBpe;
        contract.tokenizer_version = mismatch == TokenAuthorityMismatch::ProgramVersion ? 2 : 1;
        contract.tokenizer_data = {ArtifactId{2}, 0, token_data.bytes().size(), token_data.digest()};
        contract.vocabulary_digest = program.vocabulary_digest();
        if (mismatch == TokenAuthorityMismatch::VocabularyDigest) {
            contract.vocabulary_digest.bytes[0] ^= 0x80;
        }
        PromptTemplate prompt;
        prompt.version = 1;
        prompt.operations.push_back({PromptOperationKind::AppendInputText, {}, kNoTokenId});
        prompt.operations.push_back({PromptOperationKind::AppendLiteral,
                                     mismatch == TokenAuthorityMismatch::PromptOperations
                                         ? std::vector<uint8_t>{'\n', '\n'}
                                         : std::vector<uint8_t>{'\n'},
                                     kNoTokenId});
        contract.prompt = std::move(prompt);
    }
    auto serialized_contract = contract.serialize();
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized_contract));
    auto manifest_result = include_codec_registry
        ? SemanticManifest::build(physical, model, contract, codec_registry_for_model(model))
        : SemanticManifest::build(physical, model, contract);
    CHECK(std::holds_alternative<SemanticManifest>(manifest_result));
    if (!std::holds_alternative<SemanticManifest>(manifest_result)) return {};
    const SemanticManifest& manifest = std::get<SemanticManifest>(manifest_result);
    return {manifest.bytes().begin(), manifest.bytes().end()};
}

struct Fixture {
    std::string path;
    std::string carrier_path;
    std::string token_path;
    std::vector<uint8_t> token_program;
    std::vector<uint8_t> manifest;
};

Fixture make_fixture(uint8_t seed, bool token_ids_only = false,
                     uint32_t token_vocabulary_size = 256,
                     TokenAuthorityMismatch mismatch = TokenAuthorityMismatch::None,
                     bool include_codec_registry = true) {
    Fixture fixture;
    fixture.path = temporary_path();
    fixture.carrier_path = fixture.path + ".lapman";
    fixture.token_path = fixture.path + ".laptok";
    CHECK(write_tiny_gguf(fixture.path, seed));
    fixture.token_program = token_program_bytes(token_vocabulary_size);
    CHECK(write_bytes(fixture.token_path, fixture.token_program));
    fixture.manifest = manifest_for_gguf(fixture.path, fixture.token_path, token_ids_only,
                                         mismatch, include_codec_registry);
    CHECK(!fixture.manifest.empty());
    CHECK(write_bytes(fixture.carrier_path, fixture.manifest));
    return fixture;
}

void remove_fixture(Fixture& fixture) {
    unlink(fixture.path.c_str());
    unlink(fixture.carrier_path.c_str());
    unlink(fixture.token_path.c_str());
}

std::string temporary_directory() {
    char path[] = "/private/tmp/laplace-mlx-product-XXXXXX";
    CHECK(mkdtemp(path) != nullptr);
    return path;
}

std::vector<uint8_t> safetensors_file(const std::string& name,
                                      const std::string& dtype = "F16",
                                      const std::string& shape = "[2,2]",
                                      size_t payload_size = 8) {
    const std::string header =
        "{\"__metadata__\":{\"format\":\"mlx\"},\"" + name +
        "\":{\"dtype\":\"" + dtype + "\",\"shape\":" + shape +
        ",\"data_offsets\":[0," + std::to_string(payload_size) + "]}}";
    std::vector<uint8_t> bytes(8);
    for (size_t index = 0; index != 8; ++index) {
        bytes[index] = static_cast<uint8_t>(header.size() >> (8 * index));
    }
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.resize(bytes.size() + payload_size, 0x3d);
    return bytes;
}

struct MlxProductFixture {
    std::string directory;
    std::vector<uint8_t> token_program;
    std::vector<uint8_t> manifest;
};

std::vector<uint8_t> manifest_for_mlx(const MlxProductPhysicalPackage& package,
                                      const PackageView& token_data) {
    auto compiled = TokenProgram::compile(token_data.bytes());
    CHECK(std::holds_alternative<TokenProgram>(compiled));
    if (!std::holds_alternative<TokenProgram>(compiled)) return {};
    const TokenProgram& program = std::get<TokenProgram>(compiled);
    CHECK(package.physical_index.tensors().size() == 1);
    if (package.physical_index.tensors().size() != 1) return {};
    const ArtifactTensorRecord& source = package.physical_index.tensors()[0];

    SemanticModel model;
    model.maximum_context = 32768;
    model.vocabulary_size = 256;
    model.bos_id = 2;
    model.eos_id = 1;
    model.stop_ids = {1};
    model.tokenizer_digest = token_data.digest().bytes;
    model.template_digest = program.prompt_digest().bytes;

    SemanticTensor tensor;
    tensor.id = source.id;
    tensor.role = TensorRole::TokenEmbedding;
    tensor.logical_type = static_cast<ScalarType>(source.logical_type);
    tensor.layout = source.layout;
    tensor.quantization = source.quantization;
    for (uint64_t dimension : source.logical_dimensions) {
        tensor.dimensions.push_back({DimensionKind::Constant, dimension});
    }
    for (const ArtifactTensorPlane& source_plane : source.planes) {
        tensor.planes.push_back({source_plane.kind,
                                 source_plane.storage_type == ArtifactScalarType::Packed
                                     ? ScalarType::U8
                                     : static_cast<ScalarType>(source_plane.storage_type),
                                 source_plane.source.artifact_id,
                                 source_plane.source.offset,
                                 source_plane.source.length,
                                 source_plane.alignment,
                                 0});
    }
    model.tensors.push_back(std::move(tensor));
    model.values = {
        {0, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0},
        {1, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0},
    };
    model.input_values_first = 0;
    model.input_values_count = 1;
    model.output_values_first = 1;
    model.output_values_count = 1;
    model.operators = {
        {0, OperatorKind::EmbeddingLookup, 1, {0}, {1}, {0}, {},
         EmbeddingLookupPayload{0x3f800000u, model.vocabulary_size, 1, 0}},
    };
    model.layers = {{0, 0, 1, 0}};

    TokenContract contract;
    contract.tokenizer_algorithm = TokenizerAlgorithm::ByteBpe;
    contract.tokenizer_version = 1;
    contract.tokenizer_data = {kMlxProductTokenArtifactId, 0, token_data.bytes().size(),
                               token_data.digest()};
    contract.vocabulary_size = model.vocabulary_size;
    contract.vocabulary_digest = program.vocabulary_digest();
    contract.bos_id = model.bos_id;
    contract.eos_id = model.eos_id;
    contract.stop_ids = model.stop_ids;
    contract.authoritative_tokenizer_digest = {model.tokenizer_digest};
    contract.authoritative_template_digest = {model.template_digest};
    PromptTemplate prompt;
    prompt.version = 1;
    prompt.operations = {
        {PromptOperationKind::AppendInputText, {}, kNoTokenId},
        {PromptOperationKind::AppendLiteral, {'\n'}, kNoTokenId},
    };
    contract.prompt = std::move(prompt);
    CHECK(contract.validate().ok());
    auto manifest = SemanticManifest::build(
        package.physical_index, model, contract, codec_registry_for_model(model));
    const auto* manifest_error = std::get_if<CompatibilityReport>(&manifest);
    CHECK_MSG(std::holds_alternative<SemanticManifest>(manifest),
              "manifest code=%u operator=%u tensor=%u detail=%s",
              manifest_error ? static_cast<unsigned>(manifest_error->code) : 0,
              manifest_error ? manifest_error->operator_id : UINT32_MAX,
              manifest_error ? manifest_error->tensor_id : UINT32_MAX,
              manifest_error ? manifest_error->detail.c_str() : "none");
    if (!std::holds_alternative<SemanticManifest>(manifest)) return {};
    const SemanticManifest& value = std::get<SemanticManifest>(manifest);
    return {value.bytes().begin(), value.bytes().end()};
}

MlxProductFixture make_mlx_product_fixture(const char* dtype = "F16",
                                           bool padded_index = false) {
    MlxProductFixture fixture;
    fixture.directory = temporary_directory();
    const std::string config_path = fixture.directory + "/config.json";
    const std::string index_path = fixture.directory + "/model.safetensors.index.json";
    const std::string shard_path = fixture.directory + "/model.safetensors";
    const std::string manifest_path = fixture.directory + "/laplace.lapman";
    const std::string token_path = fixture.directory + "/laplace.laptok";
    write_text(config_path, "{\"hidden_size\":2}");
    write_text(index_path, padded_index
        ? "{ \"metadata\" : { \"total_size\" : 8 }, \"weight_map\" : { "
          "\"token_embd.weight\" : \"model.safetensors\" } }\n"
        : "{\"metadata\":{\"total_size\":8},\"weight_map\":{"
          "\"token_embd.weight\":\"model.safetensors\"}}");
    write_bytes(shard_path, safetensors_file(
        "token_embd.weight", dtype, "[2,2]", std::string(dtype) == "F32" ? 16 : 8));
    fixture.token_program = token_program_bytes();
    CHECK(write_bytes(token_path, fixture.token_program));
    CHECK(write_bytes(manifest_path, std::array<uint8_t, 1>{0}));

    auto loaded = load_mlx_product_physical_package(fixture.directory);
    CHECK(std::holds_alternative<MlxProductPhysicalPackage>(loaded));
    if (std::holds_alternative<MlxProductPhysicalPackage>(loaded)) {
        MlxProductPhysicalPackage& package = std::get<MlxProductPhysicalPackage>(loaded);
        auto token_view = package.closure.view(kMlxProductTokenArtifactId);
        CHECK(std::holds_alternative<PackageView>(token_view));
        if (std::holds_alternative<PackageView>(token_view)) {
            fixture.manifest = manifest_for_mlx(package, std::get<PackageView>(token_view));
            CHECK(!fixture.manifest.empty());
            if (!fixture.manifest.empty()) CHECK(write_bytes(manifest_path, fixture.manifest));
        }
    }
    return fixture;
}

void remove_mlx_product_fixture(MlxProductFixture& fixture) {
    for (const char* leaf : {"config.json", "model.safetensors.index.json",
                             "model.safetensors", "laplace.lapman", "laplace.laptok"}) {
        unlink((fixture.directory + "/" + leaf).c_str());
    }
    rmdir(fixture.directory.c_str());
}

void check_error(const std::string& path, CompatibilityError expected) {
    ProductPackageLoadResult loaded = load_product_package(path);
    CHECK(std::holds_alternative<CompatibilityReport>(loaded));
    if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
        CHECK(report->code == expected);
    }
}

void test_malformed_raw_package_fails_before_authority() {
    const std::string path = temporary_path();
    ProductPackageLoadResult loaded = load_product_package(path);
    CHECK(std::holds_alternative<CompatibilityReport>(loaded));
    if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
        CHECK(report->code == CompatibilityError::PACKAGE_BAD_MAGIC);
    }
    unlink(path.c_str());
}

void test_embedded_semantic_record_cannot_grant_closed_authority() {
    const std::string path = temporary_path();
    CHECK(write_compilable_raw_gguf(path));
    ProductPackageLoadResult loaded = load_product_package(path);
    unlink(path.c_str());
    CHECK(std::holds_alternative<CompatibilityReport>(loaded));
    if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
        CHECK(report->code == CompatibilityError::AUTHORITY_INVALID);
    }
}

void test_raw_gguf_without_declared_semantics_fails_closed() {
    const std::string path = temporary_path();
    CHECK(write_compilable_raw_gguf(path, RawTokenizerVariant::MissingSemanticRecord));
    ProductPackageLoadResult loaded = load_product_package(path);
    CHECK(std::holds_alternative<CompatibilityReport>(loaded));
    if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
        CHECK(report->code == CompatibilityError::IMPORT_SCHEMA_NOT_FOUND);
    }
    unlink(path.c_str());
}

void test_raw_gguf_reports_missing_executable_semantics() {
    const std::string path = temporary_path();
    CHECK(write_compilable_raw_gguf(path, RawTokenizerVariant::MissingSemanticRecord));
    ProductPackageLoadResult loaded = load_product_package(path);
    unlink(path.c_str());
    CHECK(std::holds_alternative<CompatibilityReport>(loaded));
    if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
        CHECK(report->code == CompatibilityError::IMPORT_SCHEMA_NOT_FOUND);
        CHECK(report->detail ==
              "typed normalized GGUF facts do not form an executable semantic graph");
    }
}

void test_raw_gguf_declared_semantics_must_match_physical_tensors() {
    const std::string path = temporary_path();
    CHECK(write_compilable_raw_gguf(
        path, RawTokenizerVariant::SemanticRecordBindingMismatch));
    ProductPackageLoadResult loaded = load_product_package(path);
    CHECK(std::holds_alternative<CompatibilityReport>(loaded));
    if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
        CHECK(report->code == CompatibilityError::AUTHORITY_INVALID);
    }
    unlink(path.c_str());
}

void test_raw_gguf_semantic_record_requires_exact_u8_wire_type() {
    const std::string path = temporary_path();
    CHECK(write_compilable_raw_gguf(
        path, RawTokenizerVariant::SemanticRecordWrongWireType));
    ProductPackageLoadResult loaded = load_product_package(path);
    CHECK(std::holds_alternative<CompatibilityReport>(loaded));
    if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
        CHECK(report->code == CompatibilityError::AUTHORITY_INVALID);
    }
    unlink(path.c_str());
}

void test_raw_gguf_without_a_closed_schema_rejects_tokenizer_variants() {
    for (const RawTokenizerVariant variant : {
             RawTokenizerVariant::MergeTable,
             RawTokenizerVariant::MissingTokenTypes,
             RawTokenizerVariant::InvalidByteBasis,
             RawTokenizerVariant::DynamicTemplate,
             RawTokenizerVariant::ExtraTokenizerMetadata,
         }) {
        const std::string path = temporary_path();
        CHECK(write_compilable_raw_gguf(path, variant));
        ProductPackageLoadResult loaded = load_product_package(path);
        CHECK(std::holds_alternative<CompatibilityReport>(loaded));
        if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
            CHECK(report->code == CompatibilityError::IMPORT_SCHEMA_NOT_FOUND);
        }
        unlink(path.c_str());
    }
}

void test_carried_manifest_factory_grants_product_authority() {
    Fixture fixture = make_fixture(0x11);
    ProductPackageLoadResult loaded = load_product_package(fixture.path);
    CHECK(std::holds_alternative<ProductPackage>(loaded));
    if (auto* package = std::get_if<ProductPackage>(&loaded)) {
        const auto runtime = package->runtime_package();
        CHECK(runtime != nullptr);
        if (runtime) {
            CHECK(runtime->authority_kind() == PackageAuthorityKind::CarriedManifest);
            CHECK(runtime->product_authoritative());
        }
    }
    remove_fixture(fixture);
}

void test_product_load_advises_referenced_tensor_ranges() {
    Fixture fixture = make_fixture(0x19);
    g_product_advice_calls = 0;
    g_product_advice_bytes = 0;
    ArtifactSet::set_test_advice_hook(record_product_advice);
    ProductPackageLoadResult loaded = load_product_package(fixture.path);
    ArtifactSet::set_test_advice_hook(nullptr);
    CHECK(std::holds_alternative<ProductPackage>(loaded));
    CHECK(g_product_advice_calls > 0);
    CHECK(g_product_advice_bytes > 0);
    remove_fixture(fixture);
}

void test_carried_manifest_decoder_does_not_grant_partial_product_authority() {
    Fixture fixture = make_fixture(0x13);
    const std::array<ArtifactSource, 3> sources = {
        ArtifactSource{fixture.path, ArtifactRole::Primary, ArtifactId{0}},
        ArtifactSource{fixture.carrier_path, ArtifactRole::Sidecar, ArtifactId{1}},
        ArtifactSource{fixture.token_path, ArtifactRole::Shard, ArtifactId{2}},
    };
    auto graph = ArtifactSet::load_graph(sources);
    CHECK(std::holds_alternative<ArtifactSet>(graph));
    if (auto* artifacts = std::get_if<ArtifactSet>(&graph)) {
        auto primary = artifacts->view(ArtifactId{0});
        auto carrier = artifacts->view(ArtifactId{1});
        auto token_program = artifacts->view(ArtifactId{2});
        CHECK(std::holds_alternative<PackageView>(primary));
        CHECK(std::holds_alternative<PackageView>(carrier));
        CHECK(std::holds_alternative<PackageView>(token_program));
        if (auto* primary_view = std::get_if<PackageView>(&primary)) {
            auto physical = build_gguf_artifact_index(*primary_view);
            CHECK(std::holds_alternative<ArtifactIndex>(physical));
            if (auto* index = std::get_if<ArtifactIndex>(&physical)) {
                if (auto* token_view = std::get_if<PackageView>(&token_program)) {
                    *index = add_token_artifact(std::move(*index), *token_view);
                }
                if (auto* carrier_view = std::get_if<PackageView>(&carrier)) {
                    auto loaded = load_carried_manifest(*index, *carrier_view);
                    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
                    if (auto* validated = std::get_if<ValidatedPackage>(&loaded)) {
                        const auto runtime = validated->runtime_package();
                        CHECK(runtime != nullptr);
                        if (runtime) {
                            CHECK(runtime->authority_kind() ==
                                  PackageAuthorityKind::DiagnosticRaw);
                            CHECK(!runtime->product_authoritative());
                        }
                    }
                }
            }
        }
    }
    remove_fixture(fixture);
}

void test_carried_manifest_without_codec_registry_fails_closed() {
    Fixture fixture = make_fixture(0x12, false, 256, TokenAuthorityMismatch::None, false);
    check_error(fixture.path, CompatibilityError::IR_QUANTIZATION_UNSUPPORTED);
    remove_fixture(fixture);
}

void test_present_invalid_carrier_does_not_downgrade_to_raw_compilation() {
    const std::string path = temporary_path();
    CHECK(write_compilable_raw_gguf(path));
    const std::string carrier_path = path + ".lapman";
    CHECK(mkdir(carrier_path.c_str(), 0700) == 0);
    ProductPackageLoadResult loaded = load_product_package(path);
    CHECK(std::holds_alternative<CompatibilityReport>(loaded));
    if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
        CHECK(report->code == CompatibilityError::PACKAGE_BOUNDS_INVALID);
        CHECK(report->artifact_id == ArtifactId{1});
    }
    rmdir(carrier_path.c_str());
    unlink(path.c_str());
}

void test_token_ids_only_carrier_is_not_product_authoritative() {
    Fixture fixture = make_fixture(0x71, true);
    check_error(fixture.path, CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED);
    remove_fixture(fixture);
}

void test_corrupt_carried_manifest_fails_closed() {
    Fixture fixture = make_fixture(0x21);
    std::vector<uint8_t> corrupt = fixture.manifest;
    corrupt.back() ^= 1;
    CHECK(write_bytes(fixture.carrier_path, corrupt));
    check_error(fixture.path, CompatibilityError::PACKAGE_CHECKSUM_MISMATCH);
    remove_fixture(fixture);
}

void test_carried_manifest_from_a_different_source_fails_closed() {
    Fixture source = make_fixture(0x31);
    Fixture wrong = make_fixture(0x41);
    CHECK(write_bytes(wrong.carrier_path, source.manifest));
    check_error(wrong.path, CompatibilityError::PACKAGE_SOURCE_CHANGED);
    remove_fixture(source);
    remove_fixture(wrong);
}

void test_trailing_carried_manifest_bytes_fail_closed() {
    Fixture fixture = make_fixture(0x51);
    std::vector<uint8_t> trailing = fixture.manifest;
    trailing.push_back(0);
    CHECK(write_bytes(fixture.carrier_path, trailing));
    check_error(fixture.path, CompatibilityError::IR_VERSION_UNSUPPORTED);
    remove_fixture(fixture);
}

void test_malformed_carrier_fails_closed() {
    Fixture fixture = make_fixture(0x61);
    std::vector<uint8_t> malformed(96, 0xa5);
    CHECK(write_bytes(fixture.carrier_path, malformed));
    check_error(fixture.path, CompatibilityError::PACKAGE_BAD_MAGIC);
    remove_fixture(fixture);
}

void test_token_program_vocabulary_mismatch_fails_closed() {
    Fixture fixture = make_fixture(0x81, false, 255);
    check_error(fixture.path, CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED);
    remove_fixture(fixture);
}

void test_missing_token_program_fails_closed() {
    Fixture fixture = make_fixture(0x91);
    unlink(fixture.token_path.c_str());
    check_error(fixture.path, CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED);
    remove_fixture(fixture);
}

void test_token_program_semantics_must_match_carried_authority() {
    Fixture bos = make_fixture(0xa1, false, 256, TokenAuthorityMismatch::Bos);
    check_error(bos.path, CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED);
    remove_fixture(bos);

    Fixture vocabulary = make_fixture(
        0xa2, false, 256, TokenAuthorityMismatch::VocabularyDigest);
    check_error(vocabulary.path, CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED);
    remove_fixture(vocabulary);

    Fixture prompt = make_fixture(
        0xa3, false, 256, TokenAuthorityMismatch::PromptOperations);
    check_error(prompt.path, CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED);
    remove_fixture(prompt);

    Fixture version = make_fixture(
        0xa4, false, 256, TokenAuthorityMismatch::ProgramVersion);
    check_error(version.path, CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED);
    remove_fixture(version);
}

void test_mlx_carried_manifest_factory_grants_product_authority() {
    MlxProductFixture fixture = make_mlx_product_fixture();
    ProductPackageLoadResult loaded = load_product_package(fixture.directory);
    CHECK(std::holds_alternative<ProductPackage>(loaded));
    if (auto* package = std::get_if<ProductPackage>(&loaded)) {
        const auto runtime = package->runtime_package();
        CHECK(runtime != nullptr);
        if (runtime) {
            CHECK(runtime->authority_kind() == PackageAuthorityKind::CarriedManifest);
            CHECK(runtime->product_authoritative());
        }
    }
    remove_mlx_product_fixture(fixture);
}

void test_mlx_index_is_retained_and_changes_package_identity() {
    MlxProductFixture compact = make_mlx_product_fixture("F16", false);
    MlxProductFixture padded = make_mlx_product_fixture("F16", true);
    auto compact_loaded = load_mlx_product_physical_package(compact.directory);
    auto padded_loaded = load_mlx_product_physical_package(padded.directory);
    CHECK(std::holds_alternative<MlxProductPhysicalPackage>(compact_loaded));
    CHECK(std::holds_alternative<MlxProductPhysicalPackage>(padded_loaded));
    if (auto* left = std::get_if<MlxProductPhysicalPackage>(&compact_loaded)) {
        if (auto* right = std::get_if<MlxProductPhysicalPackage>(&padded_loaded)) {
            CHECK(left->physical_index.artifacts().size() == 4);
            CHECK(right->physical_index.artifacts().size() == 4);
            CHECK(left->physical_index.artifacts()[1].digest() !=
                  right->physical_index.artifacts()[1].digest());
        }
    }
    remove_mlx_product_fixture(compact);
    remove_mlx_product_fixture(padded);
}

void test_mlx_runtime_alone_retains_complete_immutable_closure() {
    MlxProductFixture fixture = make_mlx_product_fixture();
    std::optional<ArtifactIndex> retained;
    std::vector<std::vector<uint8_t>> expected;
    {
        auto loaded = load_mlx_product_physical_package(fixture.directory);
        CHECK(std::holds_alternative<MlxProductPhysicalPackage>(loaded));
        if (auto* package = std::get_if<MlxProductPhysicalPackage>(&loaded)) {
            for (const PackageView& artifact : package->physical_index.artifacts()) {
                expected.emplace_back(artifact.bytes().begin(), artifact.bytes().end());
            }
            retained.emplace(std::move(package->physical_index));
        }
    }
    remove_mlx_product_fixture(fixture);
    CHECK(retained.has_value());
    if (retained) {
        CHECK(retained->artifacts().size() == expected.size());
        for (size_t index = 0; index != expected.size() &&
                               index != retained->artifacts().size(); ++index) {
            const PackageView& artifact = retained->artifacts()[index];
            CHECK(std::equal(artifact.bytes().begin(), artifact.bytes().end(),
                             expected[index].begin(), expected[index].end()));
        }
    }
}

void test_mlx_directory_accepts_f32_values() {
    MlxProductFixture fixture = make_mlx_product_fixture("F32");
    auto loaded = load_mlx_product_physical_package(fixture.directory);
    CHECK(std::holds_alternative<MlxProductPhysicalPackage>(loaded));
    if (auto* package = std::get_if<MlxProductPhysicalPackage>(&loaded)) {
            CHECK(package->physical_index.tensors().size() == 1);
            CHECK(package->physical_index.tensors()[0].logical_type == ArtifactScalarType::F32);
            CHECK(package->physical_index.tensors()[0].format.encoding ==
                  ArtifactPhysicalEncoding::F32);
    }
    remove_mlx_product_fixture(fixture);
}

void test_mlx_product_requires_both_carried_sidecars() {
    MlxProductFixture missing_manifest = make_mlx_product_fixture();
    unlink((missing_manifest.directory + "/laplace.lapman").c_str());
    check_error(missing_manifest.directory, CompatibilityError::PACKAGE_AUTHORITY_REQUIRED);
    remove_mlx_product_fixture(missing_manifest);

    MlxProductFixture missing_token = make_mlx_product_fixture();
    unlink((missing_token.directory + "/laplace.laptok").c_str());
    check_error(missing_token.directory, CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED);
    remove_mlx_product_fixture(missing_token);
}

void test_mlx_product_rejects_mutated_manifest() {
    MlxProductFixture fixture = make_mlx_product_fixture();
    std::vector<uint8_t> mutated = fixture.manifest;
    CHECK(!mutated.empty());
    if (!mutated.empty()) {
        mutated.back() ^= 1;
        CHECK(write_bytes(fixture.directory + "/laplace.lapman", mutated));
        check_error(fixture.directory, CompatibilityError::PACKAGE_CHECKSUM_MISMATCH);
    }
    remove_mlx_product_fixture(fixture);
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--residency") == 0) {
        test_product_load_advises_referenced_tensor_ranges();
        return test_summary("test_product_package_residency");
    }
    test_malformed_raw_package_fails_before_authority();
    test_embedded_semantic_record_cannot_grant_closed_authority();
    test_raw_gguf_without_declared_semantics_fails_closed();
    test_raw_gguf_reports_missing_executable_semantics();
    test_raw_gguf_declared_semantics_must_match_physical_tensors();
    test_raw_gguf_semantic_record_requires_exact_u8_wire_type();
    test_raw_gguf_without_a_closed_schema_rejects_tokenizer_variants();
    test_carried_manifest_decoder_does_not_grant_partial_product_authority();
    test_carried_manifest_factory_grants_product_authority();
    test_product_load_advises_referenced_tensor_ranges();
    test_carried_manifest_without_codec_registry_fails_closed();
    test_present_invalid_carrier_does_not_downgrade_to_raw_compilation();
    test_token_ids_only_carrier_is_not_product_authoritative();
    test_corrupt_carried_manifest_fails_closed();
    test_carried_manifest_from_a_different_source_fails_closed();
    test_trailing_carried_manifest_bytes_fail_closed();
    test_malformed_carrier_fails_closed();
    test_token_program_vocabulary_mismatch_fails_closed();
    test_missing_token_program_fails_closed();
    test_token_program_semantics_must_match_carried_authority();
    test_mlx_carried_manifest_factory_grants_product_authority();
    test_mlx_index_is_retained_and_changes_package_identity();
    test_mlx_runtime_alone_retains_complete_immutable_closure();
    test_mlx_directory_accepts_f32_values();
    test_mlx_product_requires_both_carried_sidecars();
    test_mlx_product_rejects_mutated_manifest();
    return test_summary("test_product_package");
}
