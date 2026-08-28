#include "compat_rule.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <optional>
#include <string_view>
#include <type_traits>
#include <unordered_map>

#include "gguf.h"

namespace Laplace {

namespace {

CompatibilityReport import_error(CompatibilityError error) {
    CompatibilityReport report = package_report(error);
    report.stage = CompatibilityStage::Import;
    return report;
}

bool metadata_value(const MetaValue& input, PackageMetadataValue& output) {
    return std::visit([&](const auto& value) -> bool {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, std::string>) {
            output = value;
            return true;
        } else if constexpr (std::is_same_v<T, bool>) {
            output = static_cast<uint64_t>(value ? 1 : 0);
            return true;
        } else if constexpr (std::is_integral_v<T>) {
            if constexpr (std::is_signed_v<T>) {
                if (value < 0) return false;
            }
            output = static_cast<uint64_t>(value);
            return true;
        } else if constexpr (std::is_same_v<T, float>) {
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            output = static_cast<uint64_t>(bits);
            return true;
        } else if constexpr (std::is_same_v<T, double>) {
            uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            output = bits;
            return true;
        } else if constexpr (std::is_same_v<T, MetaArrayU32>) {
            output = std::vector<uint64_t>(value.begin(), value.end());
            return true;
        } else if constexpr (std::is_same_v<T, MetaArrayU64>) {
            output = std::vector<uint64_t>(value.begin(), value.end());
            return true;
        } else if constexpr (std::is_same_v<T, MetaArrayI32> || std::is_same_v<T, MetaArrayI64>) {
            std::vector<uint64_t> converted;
            converted.reserve(value.size());
            for (const auto entry : value) {
                if (entry < 0) return false;
                converted.push_back(static_cast<uint64_t>(entry));
            }
            output = std::move(converted);
            return true;
        }
        return false;
    }, input);
}

bool physical_tensor_format(GGMLType input, PackageTensorEvidence& output) {
    switch (input) {
    case GGMLType::F32:
        output.storage_type = ScalarType::F32;
        return true;
    case GGMLType::F16:
        output.storage_type = ScalarType::F16;
        return true;
    case GGMLType::Q4_K:
    case GGMLType::Q5_0:
    case GGMLType::Q6_K:
    case GGMLType::Q8_0:
        output.storage_type = ScalarType::U8;
        output.layout = PhysicalLayoutKind::GgufBlocked;
        output.quantization = QuantizationKind::BlockedAffine;
        output.block_elements = static_cast<uint32_t>(elements_per_block(input));
        output.block_bytes = static_cast<uint32_t>(bytes_per_block(input));
        return output.block_elements != 0 && output.block_bytes != 0;
    default: return false;
    }
}

void digest_update_u16(CC_SHA256_CTX& context, uint16_t value) {
    uint8_t bytes[2] = {static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8)};
    CC_SHA256_Update(&context, bytes, sizeof(bytes));
}

void digest_update_u32(CC_SHA256_CTX& context, uint32_t value) {
    uint8_t bytes[4];
    for (unsigned index = 0; index != 4; ++index) bytes[index] = static_cast<uint8_t>(value >> (index * 8));
    CC_SHA256_Update(&context, bytes, sizeof(bytes));
}

void digest_update_u64(CC_SHA256_CTX& context, uint64_t value) {
    uint8_t bytes[8];
    for (unsigned index = 0; index != 8; ++index) bytes[index] = static_cast<uint8_t>(value >> (index * 8));
    CC_SHA256_Update(&context, bytes, sizeof(bytes));
}

Sha256Digest model_fingerprint(const SemanticModel& semantics, const PackageView& package,
                               const Sha256Digest& semantic_source) {
    Sha256Digest artifact_fingerprint;
    CC_SHA256_CTX artifact_context;
    CC_SHA256_Init(&artifact_context);
    constexpr char artifact_domain[] = "laplace-artifact-v1";
    CC_SHA256_Update(&artifact_context, artifact_domain, sizeof(artifact_domain));
    digest_update_u32(artifact_context, package.artifact_id().value);
    digest_update_u16(artifact_context, 1);
    digest_update_u16(artifact_context, 0);
    digest_update_u64(artifact_context, package.bytes().size());
    CC_SHA256_Update(&artifact_context, package.digest().bytes.data(), package.digest().bytes.size());
    CC_SHA256_Final(artifact_fingerprint.bytes.data(), &artifact_context);

    Sha256Digest fingerprint;
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    constexpr char model_domain[] = "laplace-model-v1";
    CC_SHA256_Update(&context, model_domain, sizeof(model_domain));
    const Sha256Digest semantic = semantic_model_digest(semantics);
    CC_SHA256_Update(&context, semantic.bytes.data(), semantic.bytes.size());
    CC_SHA256_Update(&context, semantic_source.bytes.data(), semantic_source.bytes.size());
    digest_update_u32(context, 1);
    CC_SHA256_Update(&context, artifact_fingerprint.bytes.data(), artifact_fingerprint.bytes.size());
    CC_SHA256_Final(fingerprint.bytes.data(), &context);
    return fingerprint;
}

Sha256Digest generic_resolver_fingerprint() {
    Sha256Digest fingerprint;
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    constexpr char domain[] = "laplace-gguf-semantic-resolver-v1";
    CC_SHA256_Update(&context, domain, sizeof(domain));
    digest_update_u16(context, 5);
    CC_SHA256_Final(fingerprint.bytes.data(), &context);
    return fingerprint;
}

CompatibilityReport resolver_error(CompatibilityError error, std::string detail) {
    CompatibilityReport report = import_error(error);
    report.detail = std::move(detail);
    return report;
}

bool key_has_suffix(const std::string& key, std::string_view suffix) {
    return key.size() > suffix.size() && key.ends_with(suffix) &&
           key[key.size() - suffix.size() - 1] == '.';
}

std::optional<uint64_t> unique_metadata_u64(const PackageEvidence& package, std::string_view suffix) {
    std::optional<uint64_t> result;
    for (const auto& [key, value] : package.metadata) {
        if (!key_has_suffix(key, suffix)) continue;
        const auto* number = std::get_if<uint64_t>(&value);
        if (!number || result) return {};
        result = *number;
    }
    return result;
}

const PackageTensorEvidence* tensor_by_name(const PackageEvidence& package, const std::string& name,
                                             std::vector<bool>& used) {
    const PackageTensorEvidence* result = nullptr;
    size_t result_index = package.tensors.size();
    for (size_t index = 0; index != package.tensors.size(); ++index) {
        if (package.tensors[index].name != name) continue;
        if (result) return nullptr;
        result = &package.tensors[index];
        result_index = index;
    }
    if (result) used[result_index] = true;
    return result;
}

bool parse_block_member(std::string_view name, uint32_t& layer, std::string_view& member) {
    if (!name.starts_with("blk.")) return false;
    const size_t dot = name.find('.', 4);
    if (dot == std::string_view::npos || dot == 4 || dot + 1 == name.size()) return false;
    const char* begin = name.data() + 4;
    const char* end = name.data() + dot;
    auto parsed = std::from_chars(begin, end, layer);
    if (parsed.ec != std::errc{} || parsed.ptr != end) return false;
    member = name.substr(dot + 1);
    return true;
}

uint32_t f32_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool add_resolved_tensor(SemanticModel& model, const PackageTensorEvidence& source, TensorRole role,
                         std::vector<uint64_t> logical_dimensions, uint32_t& output) {
    if (logical_dimensions.empty() || logical_dimensions.size() > 8 || source.length == 0 ||
        source.alignment == 0 || (source.alignment & (source.alignment - 1)) != 0 ||
        source.offset % source.alignment != 0) return false;
    SemanticTensor tensor;
    tensor.id = static_cast<uint32_t>(model.tensors.size());
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    for (uint64_t dimension : logical_dimensions) tensor.dimensions.push_back({DimensionKind::Constant, dimension});
    tensor.layout.kind = source.layout;
    tensor.layout.packing = source.layout == PhysicalLayoutKind::GgufBlocked ? PackingKind::Gguf : PackingKind::None;
    tensor.layout.rank = static_cast<uint8_t>(logical_dimensions.size());
    tensor.layout.block_rank = source.layout == PhysicalLayoutKind::GgufBlocked ? 1 : 0;
    for (size_t axis = 0; axis != logical_dimensions.size(); ++axis) tensor.layout.axis_order[axis] = static_cast<uint8_t>(axis);
    tensor.layout.strides[0] = 1;
    for (size_t axis = 1; axis != logical_dimensions.size(); ++axis) {
        if (tensor.layout.strides[axis - 1] > UINT64_MAX / logical_dimensions[axis - 1]) return false;
        tensor.layout.strides[axis] = tensor.layout.strides[axis - 1] * logical_dimensions[axis - 1];
    }
    tensor.quantization.kind = source.quantization;
    if (source.quantization == QuantizationKind::BlockedAffine) {
        if (source.block_elements == 0 || source.block_bytes == 0) return false;
        tensor.layout.block_elements = source.block_elements;
        tensor.layout.block_bytes = source.block_bytes;
        tensor.quantization.accumulation_type = ScalarType::F32;
        tensor.quantization.scale_type = ScalarType::F16;
        tensor.quantization.zero_type = ScalarType::F16;
        tensor.quantization.block_elements = source.block_elements;
        tensor.quantization.block_bytes = source.block_bytes;
        tensor.quantization.group_size = source.block_elements;
        tensor.quantization.required_plane_mask = 1;
    }
    tensor.planes = {{PlaneKind::Values, source.storage_type, source.artifact_id, source.offset,
                      source.length, source.alignment, 0}};
    output = tensor.id;
    model.tensors.push_back(std::move(tensor));
    return true;
}

uint32_t add_value(SemanticModel& model, uint64_t width, ScalarType type = ScalarType::F32) {
    SemanticValue value;
    value.id = static_cast<uint32_t>(model.values.size());
    value.logical_type = type;
    value.dimensions = {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, width}};
    model.values.push_back(std::move(value));
    return model.values.back().id;
}

void add_operator(SemanticModel& model, OperatorKind kind, std::vector<uint32_t> inputs,
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
}

uint32_t add_kv_state(SemanticModel& model, StateKind kind, uint64_t heads, uint64_t head_dimension) {
    StateFormat format;
    format.logical_domain = TransformDomain::Untransformed;
    format.encoded_domain = kind == StateKind::KeyCache ? TransformDomain::RopeApplied : TransformDomain::Untransformed;
    format.alignment = 64;
    SemanticState state;
    state.id = static_cast<uint32_t>(model.states.size());
    state.kind = kind;
    state.semantic_version = model.schema_major;
    state.update_kind = kind == StateKind::KeyCache ? StateUpdateKind::AppendKey : StateUpdateKind::AppendValue;
    state.position_policy = PositionPolicy::AppendOnly;
    state.dimensions = {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, heads},
                        {DimensionKind::Constant, head_dimension}};
    state.formats = {format};
    model.states.push_back(std::move(state));
    return model.states.back().id;
}

bool add_recurrent_states(SemanticModel& model, uint64_t channels, uint64_t value_heads,
                          uint64_t head_dimension, uint64_t kernel, uint32_t& history, uint32_t& matrix) {
    if (kernel < 2) return false;
    StateFormat format;
    format.kind = StateFormatKind::RecurrentContiguous;
    format.logical_type = ScalarType::F32;
    format.encoded_type = ScalarType::F32;
    format.logical_domain = TransformDomain::Untransformed;
    format.encoded_domain = TransformDomain::Untransformed;
    format.codec = CodecKind::Fp32;
    format.cache_policy = CachePolicy::Recurrent;
    format.layout_policy = LayoutPolicy::ChannelMajorHistory;
    format.alignment = 64;
    SemanticState history_state;
    history_state.id = static_cast<uint32_t>(model.states.size());
    history_state.kind = StateKind::RecurrentConvHistory;
    history_state.semantic_version = model.schema_major;
    history_state.update_kind = StateUpdateKind::ShiftHistory;
    history_state.position_policy = PositionPolicy::ReplaceAtCursor;
    history_state.dimensions = {{DimensionKind::Constant, channels}, {DimensionKind::Constant, kernel - 1}};
    history_state.formats = {format};
    history = history_state.id;
    model.states.push_back(std::move(history_state));
    format.layout_policy = LayoutPolicy::ValueHeadKeyRowOutputColumn;
    SemanticState matrix_state;
    matrix_state.id = static_cast<uint32_t>(model.states.size());
    matrix_state.kind = StateKind::RecurrentDeltaMatrix;
    matrix_state.semantic_version = model.schema_major;
    matrix_state.update_kind = StateUpdateKind::DeltaMatrix;
    matrix_state.position_policy = PositionPolicy::ReplaceAtCursor;
    matrix_state.dimensions = {{DimensionKind::Constant, value_heads}, {DimensionKind::Constant, head_dimension},
                               {DimensionKind::Constant, head_dimension}};
    matrix_state.formats = {format};
    matrix = matrix_state.id;
    model.states.push_back(std::move(matrix_state));
    return true;
}

} // namespace

RuleEvaluationResult resolve_gguf_semantics(const PackageEvidence& package) {
    const auto block_count = unique_metadata_u64(package, "block_count");
    const auto maximum_context = unique_metadata_u64(package, "context_length");
    const auto hidden = unique_metadata_u64(package, "embedding_length");
    const auto intermediate = unique_metadata_u64(package, "feed_forward_length");
    const auto attention_heads = unique_metadata_u64(package, "attention.head_count");
    const auto attention_kv_heads = unique_metadata_u64(package, "attention.head_count_kv");
    const auto attention_key = unique_metadata_u64(package, "attention.key_length");
    const auto attention_value = unique_metadata_u64(package, "attention.value_length");
    const auto rotary = unique_metadata_u64(package, "rope.dimension_count");
    const auto rope_base = unique_metadata_u64(package, "rope.freq_base");
    const auto epsilon = unique_metadata_u64(package, "attention.layer_norm_rms_epsilon");
    const auto bos = unique_metadata_u64(package, "ggml.bos_token_id");
    const auto eos = unique_metadata_u64(package, "ggml.eos_token_id");
    for (const auto& [key, value] : package.metadata) {
        if (key_has_suffix(key, "attention.head_count_kv") && std::holds_alternative<std::vector<uint64_t>>(value)) {
            return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                  "generic resolver has no per-layer attention.head_count_kv geometry");
        }
    }
    if (!block_count || !maximum_context || !hidden || !intermediate || !attention_heads || !attention_kv_heads ||
        !attention_key || !attention_value || !rotary || !rope_base || !epsilon || !bos || !eos ||
        *block_count == 0 || *block_count > 4096 || *maximum_context == 0 || *maximum_context > 262144 ||
        *hidden == 0 || *hidden > UINT32_MAX || *intermediate == 0 || *intermediate > UINT32_MAX ||
        *attention_heads == 0 || *attention_kv_heads == 0 || *attention_heads % *attention_kv_heads != 0 ||
        *attention_key == 0 || *attention_key != *attention_value || *rotary == 0 || *rotary > *attention_key ||
        *rotary % 2 != 0 || *rope_base > UINT32_MAX || *epsilon > UINT32_MAX || *bos > UINT32_MAX || *eos > UINT32_MAX) {
        return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                              "generic resolver is missing a required normalized metadata field");
    }
    std::optional<std::array<uint32_t, 4>> multi_rope_sections;
    for (const auto& [key, value] : package.metadata) {
        if (!key_has_suffix(key, "rope.dimension_sections")) continue;
        const auto* sections = std::get_if<std::vector<uint64_t>>(&value);
        if (!sections || sections->size() != 4 || multi_rope_sections) {
            return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                  "multi-section RoPE metadata is malformed");
        }
        std::array<uint32_t, 4> converted{};
        uint64_t total = 0;
        for (size_t index = 0; index != converted.size(); ++index) {
            if ((*sections)[index] > UINT32_MAX) {
                return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                      "multi-section RoPE metadata is malformed");
            }
            converted[index] = static_cast<uint32_t>((*sections)[index]);
            total += converted[index];
        }
        if (converted[0] == 0 || converted[1] == 0 || total != *rotary / 2) {
            return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                  "multi-section RoPE metadata is malformed");
        }
        multi_rope_sections = converted;
    }
    uint64_t nextn_layers = 0;
    bool has_nextn_layers = false;
    for (const auto& [key, value] : package.metadata) {
        if (!key_has_suffix(key, "nextn_predict_layers")) continue;
        const auto* count = std::get_if<uint64_t>(&value);
        if (!count || has_nextn_layers || *count >= *block_count) {
            return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                  "terminal speculative-layer metadata is malformed");
        }
        nextn_layers = *count;
        has_nextn_layers = true;
    }
    if (nextn_layers > 1) {
        return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                              "generic resolver supports one terminal speculative layer");
    }
    const uint32_t trunk_block_count = static_cast<uint32_t>(*block_count - nextn_layers);

    std::vector<bool> used(package.tensors.size());
    const auto take_root = [&](const char* name, TensorRole role, std::vector<uint64_t> dimensions,
                               SemanticModel& model, uint32_t& id) {
        const PackageTensorEvidence* source = tensor_by_name(package, name, used);
        return source && source->dimensions == dimensions && add_resolved_tensor(model, *source, role, std::move(dimensions), id);
    };
    const auto take_layer = [&](uint32_t layer, const char* member, TensorRole role,
                                std::vector<uint64_t> dimensions, SemanticModel& model, uint32_t& id,
                                bool transpose = false) {
        const std::string name = "blk." + std::to_string(layer) + "." + member;
        const PackageTensorEvidence* source = tensor_by_name(package, name, used);
        if (!source) return false;
        std::vector<uint64_t> expected = dimensions;
        if (transpose && expected.size() == 2) std::swap(expected[0], expected[1]);
        if (source->dimensions != expected || !add_resolved_tensor(model, *source, role, std::move(dimensions), id)) return false;
        if (transpose) {
            SemanticTensor& tensor = model.tensors[id];
            tensor.layout.axis_order[0] = 1;
            tensor.layout.axis_order[1] = 0;
            tensor.layout.strides[0] = 1;
            tensor.layout.strides[1] = expected[0];
        }
        return true;
    };
    std::vector<bool> recurrent(*block_count);
    std::vector<bool> attention(*block_count);
    for (const PackageTensorEvidence& tensor : package.tensors) {
        uint32_t layer = 0;
        std::string_view member;
        if (!parse_block_member(tensor.name, layer, member)) continue;
        if (layer >= *block_count) {
            return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                  "tensor layer index exceeds declared block count");
        }
        recurrent[layer] = recurrent[layer] || member == "ssm_a";
        attention[layer] = attention[layer] || member == "attn_q.weight";
    }
    for (uint32_t layer = 0; layer != trunk_block_count; ++layer) {
        if (recurrent[layer] && attention[layer]) {
            return resolver_error(CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS,
                                  "layer " + std::to_string(layer) + " declares both recurrent and attention compositions");
        }
        if (!recurrent[layer] && !attention[layer]) {
            return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                  "layer " + std::to_string(layer) + " has no known operator composition");
        }
    }
    if (nextn_layers != 0 && (recurrent[trunk_block_count] || !attention[trunk_block_count])) {
        return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                              "terminal speculative layer is not a dense attention composition");
    }

    SemanticModel model;
    model.schema_major = nextn_layers != 0 ? 6 : multi_rope_sections ? 5 : 4;
    model.opset_major = model.schema_major;
    model.maximum_context = static_cast<uint32_t>(*maximum_context);
    model.vocabulary_size = 0;
    model.bos_id = static_cast<uint32_t>(*bos);
    model.eos_id = static_cast<uint32_t>(*eos);
    uint32_t embedding = 0, final_norm = 0, output = 0;
    if (!take_root("token_embd.weight", TensorRole::TokenEmbedding, {*hidden, 1}, model, embedding)) {
        const PackageTensorEvidence* source = tensor_by_name(package, "token_embd.weight", used);
        if (!source || source->dimensions.size() != 2 || source->dimensions[0] != *hidden || source->dimensions[1] == 0 ||
            !add_resolved_tensor(model, *source, TensorRole::TokenEmbedding, source->dimensions, embedding)) {
            return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING, "token embedding does not have [hidden,vocabulary] shape");
        }
    }
    const uint64_t vocabulary = model.tensors[embedding].dimensions[1].constant_or_symbol;
    if (vocabulary == 0 || vocabulary > UINT32_MAX || model.bos_id >= vocabulary || model.eos_id >= vocabulary ||
        !take_root("output_norm.weight", TensorRole::FinalNormWeight, {*hidden}, model, final_norm) ||
        !take_root("output.weight", TensorRole::OutputWeight, {*hidden, vocabulary}, model, output)) {
        return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                              "root embedding, output norm, or output projection is inconsistent");
    }
    model.vocabulary_size = static_cast<uint32_t>(vocabulary);
    const uint32_t token_ids = add_value(model, 1, ScalarType::U32);
    const uint32_t mtp_token_ids = nextn_layers != 0 ? add_value(model, 1, ScalarType::U32) : UINT32_MAX;
    const uint32_t embedded = add_value(model, *hidden);
    model.input_values_first = token_ids;
    model.input_values_count = nextn_layers != 0 ? 2 : 1;
    add_operator(model, OperatorKind::EmbeddingLookup, {token_ids}, {embedded}, {embedding}, {},
                 EmbeddingLookupPayload{0x3f800000u, model.vocabulary_size, static_cast<uint32_t>(*hidden), 0});
    uint32_t current = embedded;
    const uint64_t attention_width64 = *attention_heads * *attention_key;
    const uint64_t key_width64 = *attention_kv_heads * *attention_key;
    if (attention_width64 > UINT32_MAX || key_width64 > UINT32_MAX || attention_width64 > UINT32_MAX / 2) {
        return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING, "attention dimensions exceed canonical limits");
    }
    const uint32_t head_dimension = static_cast<uint32_t>(*attention_key);
    const uint32_t attention_width = static_cast<uint32_t>(attention_width64);
    const uint32_t key_width = static_cast<uint32_t>(key_width64);
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));
    const uint32_t rms_epsilon = static_cast<uint32_t>(*epsilon);

    const auto add_ffn = [&](uint32_t layer, uint32_t input, uint32_t ffn_norm, uint32_t ffn_gate,
                             uint32_t ffn_up, uint32_t ffn_down, uint32_t& next) {
        const uint32_t normalized = add_value(model, *hidden);
        const uint32_t gate = add_value(model, *intermediate);
        const uint32_t up = add_value(model, *intermediate);
        const uint32_t activated = add_value(model, *intermediate);
        const uint32_t down = add_value(model, *hidden);
        next = add_value(model, *hidden);
        add_operator(model, OperatorKind::RmsNorm, {input}, {normalized}, {ffn_norm}, {},
                     RmsNormPayload{rms_epsilon, -1, 1});
        add_operator(model, OperatorKind::Linear, {normalized}, {gate}, {ffn_gate}, {}, LinearPayload{});
        add_operator(model, OperatorKind::Linear, {normalized}, {up}, {ffn_up}, {}, LinearPayload{});
        add_operator(model, OperatorKind::SwiGlu, {gate, up}, {activated}, {}, {}, SwiGluPayload{ActivationKind::Silu});
        add_operator(model, OperatorKind::Linear, {activated}, {down}, {ffn_down}, {}, LinearPayload{});
        add_operator(model, OperatorKind::Add, {input, down}, {next}, {}, {}, AddPayload{});
        (void)layer;
    };

    for (uint32_t layer = 0; layer != trunk_block_count; ++layer) {
        const uint32_t first = static_cast<uint32_t>(model.operators.size());
        uint32_t attn_norm = 0, post_norm = 0, ffn_gate = 0, ffn_up = 0, ffn_down = 0;
        if (!take_layer(layer, "attn_norm.weight", TensorRole::AttentionNormWeight, {*hidden}, model, attn_norm) ||
            !take_layer(layer, "post_attention_norm.weight", TensorRole::FfnNormWeight, {*hidden}, model, post_norm) ||
            !take_layer(layer, "ffn_gate.weight", TensorRole::FfnGateWeight, {*hidden, *intermediate}, model, ffn_gate) ||
            !take_layer(layer, "ffn_up.weight", TensorRole::FfnUpWeight, {*hidden, *intermediate}, model, ffn_up) ||
            !take_layer(layer, "ffn_down.weight", TensorRole::FfnDownWeight, {*intermediate, *hidden}, model, ffn_down)) {
            return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                  "layer " + std::to_string(layer) + " is missing a shared norm or feed-forward tensor");
        }
        uint32_t next = 0;
        if (attention[layer]) {
            const std::string query_name = "blk." + std::to_string(layer) + ".attn_q.weight";
            const PackageTensorEvidence* query_source = tensor_by_name(package, query_name, used);
            const bool fused_query_gate = query_source && query_source->dimensions == std::vector<uint64_t>{*hidden, 2ull * attention_width};
            if (!query_source || (!fused_query_gate && query_source->dimensions != std::vector<uint64_t>{*hidden, attention_width})) {
                return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                      "layer " + std::to_string(layer) + " has no supported query projection shape");
            }
            uint32_t query = 0, key = 0, value = 0, attention_out = 0, query_norm = 0, key_norm = 0;
            if (!add_resolved_tensor(model, *query_source,
                                     fused_query_gate ? TensorRole::AttentionQueryGateWeight : TensorRole::QueryWeight,
                                     query_source->dimensions, query) ||
                !take_layer(layer, "attn_k.weight", TensorRole::KeyWeight, {*hidden, key_width}, model, key) ||
                !take_layer(layer, "attn_v.weight", TensorRole::ValueWeight, {*hidden, key_width}, model, value) ||
                !take_layer(layer, "attn_output.weight", TensorRole::AttentionOutputWeight,
                            {attention_width, *hidden}, model, attention_out) ||
                (fused_query_gate &&
                 (!take_layer(layer, "attn_q_norm.weight", TensorRole::AttentionQueryNormWeight,
                              {head_dimension}, model, query_norm) ||
                  !take_layer(layer, "attn_k_norm.weight", TensorRole::AttentionKeyNormWeight,
                              {head_dimension}, model, key_norm)))) {
                return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                  "layer " + std::to_string(layer) + " is missing a complete attention tensor set");
            }
            const uint32_t normalized = add_value(model, *hidden);
            const uint32_t query_gate = fused_query_gate ? add_value(model, 2ull * attention_width) : UINT32_MAX;
            const uint32_t q = add_value(model, attention_width);
            const uint32_t gate = fused_query_gate ? add_value(model, attention_width) : UINT32_MAX;
            const uint32_t k = add_value(model, key_width);
            const uint32_t v = add_value(model, key_width);
            const uint32_t normalized_q = fused_query_gate ? add_value(model, attention_width) : UINT32_MAX;
            const uint32_t normalized_k = fused_query_gate ? add_value(model, key_width) : UINT32_MAX;
            const uint32_t rotated_q = add_value(model, attention_width);
            const uint32_t rotated_k = add_value(model, key_width);
            const uint32_t context = add_value(model, attention_width);
            const uint32_t gated_context = fused_query_gate ? add_value(model, attention_width) : UINT32_MAX;
            const uint32_t projected = add_value(model, *hidden);
            const uint32_t residual = add_value(model, *hidden);
            const uint32_t key_state = add_kv_state(model, StateKind::KeyCache, *attention_kv_heads, *attention_key);
            const uint32_t value_state = add_kv_state(model, StateKind::ValueCache, *attention_kv_heads, *attention_key);
            add_operator(model, OperatorKind::RmsNorm, {current}, {normalized}, {attn_norm}, {}, RmsNormPayload{rms_epsilon, -1, 1});
            add_operator(model, OperatorKind::Linear, {normalized}, {fused_query_gate ? query_gate : q}, {query}, {}, LinearPayload{});
            if (fused_query_gate) {
                add_operator(model, OperatorKind::AxisSplit, {query_gate}, {q, gate}, {}, {},
                             AxisSplitPayload{attention_width, attention_width});
            }
            add_operator(model, OperatorKind::Linear, {normalized}, {k}, {key}, {}, LinearPayload{});
            add_operator(model, OperatorKind::Linear, {normalized}, {v}, {value}, {}, LinearPayload{});
            if (fused_query_gate) {
                add_operator(model, OperatorKind::RmsNorm, {q}, {normalized_q}, {query_norm}, {},
                             RmsNormPayload{rms_epsilon, -1, 1});
                add_operator(model, OperatorKind::RmsNorm, {k}, {normalized_k}, {key_norm}, {},
                             RmsNormPayload{rms_epsilon, -1, 1});
            }
            add_operator(model, OperatorKind::Rope,
                         {fused_query_gate ? normalized_q : q, fused_query_gate ? normalized_k : k},
                         {rotated_q, rotated_k}, {}, {},
                         RopePayload{multi_rope_sections ? RopePairing::MultiSectionHalfSplit : RopePairing::HalfSplit,
                                     true, static_cast<uint32_t>(*rotary), static_cast<uint32_t>(*rope_base),
                                     0x3f800000u, multi_rope_sections.value_or(std::array<uint32_t, 4>{})});
            add_operator(model, OperatorKind::CausalAttention, {rotated_q, rotated_k, v}, {context}, {}, {key_state, value_state},
                         CausalAttentionPayload{static_cast<uint32_t>(*attention_heads), static_cast<uint32_t>(*attention_kv_heads),
                                                head_dimension, f32_bits(attention_scale), AttentionMask::Causal, CachePolicy::Global});
            if (fused_query_gate) {
                add_operator(model, OperatorKind::GatedAttention, {context, gate}, {gated_context}, {}, {},
                             GatedAttentionPayload{});
            }
            add_operator(model, OperatorKind::Linear, {fused_query_gate ? gated_context : context}, {projected},
                         {attention_out}, {}, LinearPayload{});
            add_operator(model, OperatorKind::Add, {current, projected}, {residual}, {}, {}, AddPayload{});
            add_ffn(layer, residual, post_norm, ffn_gate, ffn_up, ffn_down, next);
        } else {
            const auto groups = unique_metadata_u64(package, "ssm.group_count");
            const auto inner = unique_metadata_u64(package, "ssm.inner_size");
            const auto state_size = unique_metadata_u64(package, "ssm.state_size");
            const auto kernel = unique_metadata_u64(package, "ssm.conv_kernel");
            const auto rank = unique_metadata_u64(package, "ssm.time_step_rank");
            if (!groups || !inner || !state_size || !kernel || !rank || *groups == 0 || *inner == 0 || *state_size == 0 ||
                *kernel < 2 || *rank == 0 || *inner % *state_size != 0 || *groups > UINT32_MAX ||
                *inner > UINT32_MAX || *state_size > UINT32_MAX || *kernel > UINT32_MAX || *rank > UINT32_MAX) {
                return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING, "recurrent metadata is incomplete");
            }
            const uint64_t value_heads = *inner / *state_size;
            const uint64_t channels = (2 * *groups + value_heads) * *state_size;
            if (channels > UINT32_MAX || *groups > value_heads) {
                return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING, "recurrent dimensions are inconsistent");
            }
            uint32_t qkv = 0, gate = 0, beta = 0, alpha = 0, conv = 0, dt = 0, decay = 0, norm = 0, recurrent_out = 0;
            if (!take_layer(layer, "attn_qkv.weight", TensorRole::RecurrentQkvWeight, {*hidden, channels}, model, qkv) ||
                !take_layer(layer, "attn_gate.weight", TensorRole::RecurrentGateWeight, {*hidden, *inner}, model, gate) ||
                !take_layer(layer, "ssm_beta.weight", TensorRole::RecurrentBetaWeight, {*hidden, value_heads}, model, beta) ||
                !take_layer(layer, "ssm_alpha.weight", TensorRole::RecurrentAlphaWeight, {*hidden, value_heads}, model, alpha) ||
                !take_layer(layer, "ssm_conv1d.weight", TensorRole::RecurrentConvWeight, {channels, *kernel}, model, conv, true) ||
                !take_layer(layer, "ssm_dt.bias", TensorRole::RecurrentDtBias, {value_heads}, model, dt) ||
                !take_layer(layer, "ssm_a", TensorRole::RecurrentDecayWeight, {value_heads}, model, decay) ||
                !take_layer(layer, "ssm_norm.weight", TensorRole::RecurrentNormWeight, {*state_size}, model, norm) ||
                !take_layer(layer, "ssm_out.weight", TensorRole::RecurrentOutputWeight, {*inner, *hidden}, model, recurrent_out)) {
                return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                      "layer " + std::to_string(layer) + " is missing a complete recurrent tensor set");
            }
            uint32_t history = 0, matrix = 0;
            if (!add_recurrent_states(model, channels, value_heads, *state_size, *kernel, history, matrix)) {
                return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING, "recurrent state dimensions are invalid");
            }
            const uint32_t normalized = add_value(model, *hidden);
            const uint32_t qkv_value = add_value(model, channels);
            const uint32_t gate_value = add_value(model, *inner);
            const uint32_t beta_value = add_value(model, value_heads);
            const uint32_t alpha_value = add_value(model, value_heads);
            const uint32_t q = add_value(model, *groups * *state_size);
            const uint32_t k = add_value(model, *groups * *state_size);
            const uint32_t v = add_value(model, *inner);
            const uint32_t normalized_q = add_value(model, *groups * *state_size);
            const uint32_t normalized_k = add_value(model, *groups * *state_size);
            const uint32_t delta = add_value(model, *inner);
            const uint32_t gated = add_value(model, *inner);
            const uint32_t projected = add_value(model, *hidden);
            const uint32_t residual = add_value(model, *hidden);
            add_operator(model, OperatorKind::RmsNorm, {current}, {normalized}, {attn_norm}, {}, RmsNormPayload{rms_epsilon, -1, 1});
            add_operator(model, OperatorKind::Linear, {normalized}, {qkv_value}, {qkv}, {}, LinearPayload{});
            add_operator(model, OperatorKind::Linear, {normalized}, {gate_value}, {gate}, {}, LinearPayload{});
            add_operator(model, OperatorKind::Linear, {normalized}, {beta_value}, {beta}, {}, LinearPayload{});
            add_operator(model, OperatorKind::Linear, {normalized}, {alpha_value}, {alpha}, {}, LinearPayload{});
            add_operator(model, OperatorKind::DepthwiseConvSilu, {qkv_value}, {q, k, v}, {conv}, {history},
                         DepthwiseConvSiluPayload{static_cast<uint32_t>(*groups), static_cast<uint32_t>(value_heads),
                                                   static_cast<uint32_t>(*state_size), static_cast<uint32_t>(*kernel)});
            add_operator(model, OperatorKind::L2Normalize, {q}, {normalized_q}, {}, {}, L2NormalizePayload{rms_epsilon});
            add_operator(model, OperatorKind::L2Normalize, {k}, {normalized_k}, {}, {}, L2NormalizePayload{rms_epsilon});
            add_operator(model, OperatorKind::GatedDeltaNet, {normalized_q, normalized_k, v, beta_value, alpha_value}, {delta},
                         {dt, decay}, {matrix}, GatedDeltaNetPayload{static_cast<uint32_t>(*groups), static_cast<uint32_t>(value_heads),
                         static_cast<uint32_t>(*state_size), QkHeadMapping::ValueHeadModulo, BetaTransform::Sigmoid,
                         DecayTransform::NegativeSoftplus, DeltaStateLayout::ValueHeadKeyRowOutputColumn, 0});
            add_operator(model, OperatorKind::GatedRmsNorm, {delta, gate_value}, {gated}, {norm}, {},
                         GatedRmsNormPayload{rms_epsilon, ActivationKind::Silu, 1});
            add_operator(model, OperatorKind::Linear, {gated}, {projected}, {recurrent_out}, {}, LinearPayload{});
            add_operator(model, OperatorKind::Add, {current, projected}, {residual}, {}, {}, AddPayload{});
            add_ffn(layer, residual, post_norm, ffn_gate, ffn_up, ffn_down, next);
        }
        model.layers.push_back({layer, first, static_cast<uint32_t>(model.operators.size()) - first, 0});
        current = next;
    }
    const uint32_t normalized = add_value(model, *hidden);
    const uint32_t logits = add_value(model, vocabulary);
    add_operator(model, OperatorKind::RmsNorm, {current}, {normalized}, {final_norm}, {}, RmsNormPayload{rms_epsilon, -1, 1});
    add_operator(model, OperatorKind::Linear, {normalized}, {logits}, {output}, {}, LinearPayload{});
    if (nextn_layers != 0) {
        const uint32_t layer = trunk_block_count;
        const uint32_t first = static_cast<uint32_t>(model.operators.size());
        uint32_t hnorm = 0, enorm = 0, eh_projection = 0, shared_head_norm = 0;
        uint32_t attn_norm = 0, post_norm = 0, ffn_gate = 0, ffn_up = 0, ffn_down = 0;
        if (!take_layer(layer, "nextn.hnorm.weight", TensorRole::NextnHiddenNormWeight, {*hidden}, model, hnorm) ||
            !take_layer(layer, "nextn.enorm.weight", TensorRole::NextnEmbeddingNormWeight, {*hidden}, model, enorm) ||
            !take_layer(layer, "nextn.eh_proj.weight", TensorRole::NextnProjectionWeight, {2ull * *hidden, *hidden}, model, eh_projection) ||
            !take_layer(layer, "nextn.shared_head_norm.weight", TensorRole::NextnSharedHeadNormWeight, {*hidden}, model, shared_head_norm) ||
            !take_layer(layer, "attn_norm.weight", TensorRole::AttentionNormWeight, {*hidden}, model, attn_norm) ||
            !take_layer(layer, "post_attention_norm.weight", TensorRole::FfnNormWeight, {*hidden}, model, post_norm) ||
            !take_layer(layer, "ffn_gate.weight", TensorRole::FfnGateWeight, {*hidden, *intermediate}, model, ffn_gate) ||
            !take_layer(layer, "ffn_up.weight", TensorRole::FfnUpWeight, {*hidden, *intermediate}, model, ffn_up) ||
            !take_layer(layer, "ffn_down.weight", TensorRole::FfnDownWeight, {*intermediate, *hidden}, model, ffn_down)) {
            return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                  "terminal speculative layer is missing a complete tensor set");
        }
        const std::string query_name = "blk." + std::to_string(layer) + ".attn_q.weight";
        const PackageTensorEvidence* query_source = tensor_by_name(package, query_name, used);
        if (!query_source || query_source->dimensions != std::vector<uint64_t>{*hidden, 2ull * attention_width}) {
            return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                  "terminal speculative layer has no fused query/gate projection");
        }
        uint32_t query = 0, key = 0, value = 0, attention_out = 0, query_norm = 0, key_norm = 0;
        if (!add_resolved_tensor(model, *query_source, TensorRole::AttentionQueryGateWeight,
                                 query_source->dimensions, query) ||
            !take_layer(layer, "attn_k.weight", TensorRole::KeyWeight, {*hidden, key_width}, model, key) ||
            !take_layer(layer, "attn_v.weight", TensorRole::ValueWeight, {*hidden, key_width}, model, value) ||
            !take_layer(layer, "attn_output.weight", TensorRole::AttentionOutputWeight,
                        {attention_width, *hidden}, model, attention_out) ||
            !take_layer(layer, "attn_q_norm.weight", TensorRole::AttentionQueryNormWeight,
                        {head_dimension}, model, query_norm) ||
            !take_layer(layer, "attn_k_norm.weight", TensorRole::AttentionKeyNormWeight,
                        {head_dimension}, model, key_norm)) {
            return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                  "terminal speculative layer is missing a complete attention tensor set");
        }
        const uint32_t candidate_embedding = add_value(model, *hidden);
        const uint32_t normalized_embedding = add_value(model, *hidden);
        const uint32_t normalized_hidden = add_value(model, *hidden);
        const uint32_t concat = add_value(model, 2ull * *hidden);
        const uint32_t mtp_current = add_value(model, *hidden);
        const uint32_t attn_input = add_value(model, *hidden);
        const uint32_t query_gate = add_value(model, 2ull * attention_width);
        const uint32_t q = add_value(model, attention_width);
        const uint32_t gate = add_value(model, attention_width);
        const uint32_t k = add_value(model, key_width);
        const uint32_t v = add_value(model, key_width);
        const uint32_t normalized_q = add_value(model, attention_width);
        const uint32_t normalized_k = add_value(model, key_width);
        const uint32_t rotated_q = add_value(model, attention_width);
        const uint32_t rotated_k = add_value(model, key_width);
        const uint32_t context = add_value(model, attention_width);
        const uint32_t gated_context = add_value(model, attention_width);
        const uint32_t projected = add_value(model, *hidden);
        const uint32_t residual = add_value(model, *hidden);
        const uint32_t key_state = add_kv_state(model, StateKind::KeyCache, *attention_kv_heads, *attention_key);
        const uint32_t value_state = add_kv_state(model, StateKind::ValueCache, *attention_kv_heads, *attention_key);
        const uint32_t head_hidden = add_value(model, *hidden);
        const uint32_t mtp_logits = add_value(model, vocabulary);
        add_operator(model, OperatorKind::EmbeddingLookup, {mtp_token_ids}, {candidate_embedding}, {embedding}, {},
                     EmbeddingLookupPayload{0x3f800000u, model.vocabulary_size, static_cast<uint32_t>(*hidden), 0});
        add_operator(model, OperatorKind::RmsNorm, {candidate_embedding}, {normalized_embedding}, {enorm}, {},
                     RmsNormPayload{rms_epsilon, -1, 1});
        add_operator(model, OperatorKind::RmsNorm, {normalized}, {normalized_hidden}, {hnorm}, {},
                     RmsNormPayload{rms_epsilon, -1, 1});
        add_operator(model, OperatorKind::Concat, {normalized_embedding, normalized_hidden}, {concat}, {}, {},
                     ConcatPayload{-1});
        add_operator(model, OperatorKind::Linear, {concat}, {mtp_current}, {eh_projection}, {}, LinearPayload{});
        add_operator(model, OperatorKind::RmsNorm, {mtp_current}, {attn_input}, {attn_norm}, {},
                     RmsNormPayload{rms_epsilon, -1, 1});
        add_operator(model, OperatorKind::Linear, {attn_input}, {query_gate}, {query}, {}, LinearPayload{});
        add_operator(model, OperatorKind::AxisSplit, {query_gate}, {q, gate}, {}, {},
                     AxisSplitPayload{attention_width, attention_width});
        add_operator(model, OperatorKind::Linear, {attn_input}, {k}, {key}, {}, LinearPayload{});
        add_operator(model, OperatorKind::Linear, {attn_input}, {v}, {value}, {}, LinearPayload{});
        add_operator(model, OperatorKind::RmsNorm, {q}, {normalized_q}, {query_norm}, {}, RmsNormPayload{rms_epsilon, -1, 1});
        add_operator(model, OperatorKind::RmsNorm, {k}, {normalized_k}, {key_norm}, {}, RmsNormPayload{rms_epsilon, -1, 1});
        add_operator(model, OperatorKind::Rope, {normalized_q, normalized_k}, {rotated_q, rotated_k}, {}, {},
                     RopePayload{multi_rope_sections ? RopePairing::MultiSectionHalfSplit : RopePairing::HalfSplit,
                                 true, static_cast<uint32_t>(*rotary), static_cast<uint32_t>(*rope_base),
                                 0x3f800000u, multi_rope_sections.value_or(std::array<uint32_t, 4>{})});
        add_operator(model, OperatorKind::CausalAttention, {rotated_q, rotated_k, v}, {context}, {}, {key_state, value_state},
                     CausalAttentionPayload{static_cast<uint32_t>(*attention_heads), static_cast<uint32_t>(*attention_kv_heads),
                                            head_dimension, f32_bits(attention_scale), AttentionMask::Causal, CachePolicy::Global});
        add_operator(model, OperatorKind::GatedAttention, {context, gate}, {gated_context}, {}, {}, GatedAttentionPayload{});
        add_operator(model, OperatorKind::Linear, {gated_context}, {projected}, {attention_out}, {}, LinearPayload{});
        add_operator(model, OperatorKind::Add, {mtp_current, projected}, {residual}, {}, {}, AddPayload{});
        uint32_t mtp_next = 0;
        add_ffn(layer, residual, post_norm, ffn_gate, ffn_up, ffn_down, mtp_next);
        add_operator(model, OperatorKind::RmsNorm, {mtp_next}, {head_hidden}, {shared_head_norm}, {},
                     RmsNormPayload{rms_epsilon, -1, 1});
        add_operator(model, OperatorKind::Linear, {head_hidden}, {mtp_logits}, {output}, {}, LinearPayload{});
        model.layers.push_back({layer, first, static_cast<uint32_t>(model.operators.size()) - first,
                                kSemanticLayerFlagSpeculative});
    }
    model.output_values_first = logits;
    model.output_values_count = 1;
    for (size_t index = 0; index != used.size(); ++index) {
        if (!used[index]) {
            return resolver_error(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                                  "generic resolver has no semantic role for tensor " + package.tensors[index].name);
        }
    }
    auto encoded = encode_semantic_model(model);
    if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) {
        return resolver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                              "derived semantic graph violates a canonical invariant");
    }
    return model;
}

using EvidenceResult = std::variant<PackageEvidence, CompatibilityReport>;

EvidenceResult normalized_gguf_evidence(const PackageView& package) {
    if (package.artifact_id().value != 0 || package.bytes().empty()) return import_error(CompatibilityError::PACKAGE_BOUNDS_INVALID);
    GGUFContext context;
    if (!context.parse(package.bytes())) return import_error(CompatibilityError::PACKAGE_BAD_MAGIC);

    PackageEvidence evidence;
    evidence.metadata.emplace("artifact.content_sha256", package.digest());
    for (const auto& [key, value] : context.metadata()) {
        PackageMetadataValue converted;
        if (metadata_value(value, converted)) evidence.metadata.emplace(key, std::move(converted));
    }
    const auto& infos = context.tensor_infos();
    const auto& tensors = context.tensors();
    if (infos.size() != tensors.size()) return import_error(CompatibilityError::PACKAGE_BOUNDS_INVALID);
    evidence.tensors.reserve(infos.size());
    for (size_t index = 0; index != infos.size(); ++index) {
        if (infos[index].n_dims == 0 || infos[index].n_dims > 4 || tensors[index].nbytes() == 0 ||
            context.data_section_offset() > UINT64_MAX - infos[index].offset) return import_error(CompatibilityError::PACKAGE_BOUNDS_INVALID);
        PackageTensorEvidence tensor;
        if (!physical_tensor_format(infos[index].type, tensor)) return import_error(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED);
        tensor.name = infos[index].name;
        tensor.dimensions.assign(infos[index].dims, infos[index].dims + infos[index].n_dims);
        tensor.artifact_id = package.artifact_id();
        tensor.offset = context.data_section_offset() + infos[index].offset;
        tensor.length = tensors[index].nbytes();
        tensor.alignment = static_cast<uint32_t>(context.alignment());
        evidence.tensors.push_back(std::move(tensor));
    }
    return evidence;
}

RuleEvaluationResult import_gguf(const PackageView& package) {
    auto evidence = normalized_gguf_evidence(package);
    if (const auto* report = std::get_if<CompatibilityReport>(&evidence)) return *report;
    return resolve_gguf_semantics(std::get<PackageEvidence>(std::move(evidence)));
}

RuleEvaluationResult import_expected_fixture_gguf(const PackageView& package,
                                                  const std::vector<CompatibilityRule>& rules) {
    auto evidence = normalized_gguf_evidence(package);
    if (const auto* report = std::get_if<CompatibilityReport>(&evidence)) return *report;
    return evaluate_rules(rules, std::get<PackageEvidence>(std::move(evidence)));
}

ValidatedLoadResult load_validated_gguf(const PackageView& package) {
    auto imported = import_gguf(package);
    if (const auto* report = std::get_if<CompatibilityReport>(&imported)) return *report;
    SemanticModel semantics = std::get<SemanticModel>(std::move(imported));
    const Sha256Digest resolver = generic_resolver_fingerprint();
    auto runtime = std::make_shared<RuntimePackage>(semantics, package, model_fingerprint(semantics, package, resolver),
                                                    resolver, 2, RuleQualificationState::Draft);
    DiagnosticProvenance diagnostics;
    diagnostics.format_name = "GGUF";
    diagnostics.importer_name = "gguf-v3";
    diagnostics.rule_id = "gguf-semantic-resolver-v1";
    diagnostics.rule_revision = 2;
    return ValidatedPackage(std::move(runtime), std::move(diagnostics));
}

} // namespace Laplace
