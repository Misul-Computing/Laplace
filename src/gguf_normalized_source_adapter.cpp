#include "gguf_normalized_source_adapter.h"

#include <charconv>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace Laplace {
namespace {

struct ParsedRole {
    TensorRole role = TensorRole::TokenEmbedding;
    ArtifactCoordinate coordinate;
};

enum class ParseState : uint8_t { Missing, Matched, Ambiguous };

struct ParseResult {
    ParseState state = ParseState::Missing;
    ParsedRole value;
};

std::vector<std::string_view> split(std::string_view value) {
    std::vector<std::string_view> parts;
    size_t begin = 0;
    while (begin <= value.size()) {
        const size_t end = value.find('.', begin);
        parts.push_back(value.substr(begin, end == std::string_view::npos
                                             ? value.size() - begin : end - begin));
        if (end == std::string_view::npos) break;
        begin = end + 1;
    }
    return parts;
}

bool decimal(std::string_view value, uint32_t& result) {
    if (value.empty()) return false;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
}

std::optional<TensorRole> role_marker(std::string_view value) {
    if (value == "attn_norm" || value == "attention_norm" || value == "input_layernorm")
        return TensorRole::AttentionNormWeight;
    if (value == "attn_q" || value == "q_proj" || value == "query")
        return TensorRole::QueryWeight;
    if (value == "attn_k" || value == "k_proj" || value == "key")
        return TensorRole::KeyWeight;
    if (value == "attn_v" || value == "v_proj" || value == "value")
        return TensorRole::ValueWeight;
    if (value == "attn_o" || value == "attn_output" || value == "o_proj" || value == "attention_output")
        return TensorRole::AttentionOutputWeight;
    if (value == "ffn_norm" || value == "post_attention_layernorm")
        return TensorRole::FfnNormWeight;
    if (value == "post_attention_norm") return TensorRole::PostNormWeight;
    if (value == "ffn_gate" || value == "gate_proj") return TensorRole::FfnGateWeight;
    if (value == "ffn_up" || value == "up_proj") return TensorRole::FfnUpWeight;
    if (value == "ffn_down" || value == "down_proj") return TensorRole::FfnDownWeight;
    return std::nullopt;
}

// These are typed source-schema markers, not model/family selectors.  Keep
// them here so the compiler consumes normalized role evidence rather than
// making a second spelling-based decision.
std::optional<TensorRole> full_role_marker(std::string_view value) {
    if (value == "ffn_gate_inp.scale") return TensorRole::RouterScaleWeight;
    if (value == "ffn_gate_inp.weight") return TensorRole::NextnProjectionWeight;
    if (value == "ffn_down_exps.scale") return TensorRole::ReduceScaleWeight;
    if (value == "ffn_gate_up_exps.weight") return TensorRole::FfnUpWeight;
    if (value == "ffn_down_exps.weight") return TensorRole::FfnDownWeight;
    if (value == "pre_ffw_norm_2.weight") return TensorRole::ExpertNormWeight;
    if (value == "post_ffw_norm_1.weight" || value == "post_ffw_norm_2.weight" ||
        value == "post_ffw_norm.weight") return TensorRole::PostNormWeight;
    if (value == "layer_output_scale.weight") return TensorRole::OutputScaleWeight;
    if (value == "attn_q_norm.weight") return TensorRole::AttentionQueryNormWeight;
    if (value == "attn_k_norm.weight") return TensorRole::AttentionKeyNormWeight;
    if (value == "rope_freqs.weight") return TensorRole::NextnEmbeddingNormWeight;
    return std::nullopt;
}

ParseResult parse_spelling(std::string_view spelling) {
    if (spelling == "token_embd.weight" || spelling == "embedding.weight" ||
        spelling == "embed_tokens.weight") {
        return {ParseState::Matched, {TensorRole::TokenEmbedding, {0}}};
    }
    if (spelling == "output_norm.weight" || spelling == "final_norm.weight") {
        return {ParseState::Matched, {TensorRole::FinalNormWeight, {0}}};
    }
    if (spelling == "output.weight" || spelling == "lm_head.weight") {
        return {ParseState::Matched, {TensorRole::OutputWeight, {0}}};
    }
    if (const auto compound = full_role_marker(spelling)) {
        ParsedRole parsed;
        parsed.role = *compound;
        parsed.coordinate.root = 0;
        return {ParseState::Matched, parsed};
    }

    const auto parts = split(spelling);
    uint32_t layer = UINT32_MAX;
    size_t after_layer = parts.size();
    for (size_t index = 0; index + 1 < parts.size(); ++index) {
        if (parts[index] == "blk" || parts[index] == "layer" || parts[index] == "layers") {
            if (layer != UINT32_MAX || !decimal(parts[index + 1], layer)) {
                return {ParseState::Ambiguous, {}};
            }
            after_layer = index + 2;
        }
    }
    if (layer == UINT32_MAX || after_layer == parts.size()) return {ParseState::Missing, {}};

    std::optional<TensorRole> role;
    const size_t marker_begin = after_layer;
    if (marker_begin < parts.size()) {
        std::string_view suffix = spelling.substr(0);
        // Match the complete post-layer spelling so compound source markers
        // cannot be mistaken for two independent role candidates.
        size_t begin = 0;
        for (size_t index = 0; index != marker_begin; ++index) {
            begin = spelling.find('.', begin);
            if (begin == std::string_view::npos) break;
            ++begin;
        }
        if (begin != std::string_view::npos) suffix = spelling.substr(begin);
        if (const auto compound = full_role_marker(suffix)) {
            ParsedRole parsed;
            parsed.role = *compound;
            parsed.coordinate.root = 0;
            parsed.coordinate.layer = layer;
            if (*compound == TensorRole::PostNormWeight) {
                if (suffix == "post_ffw_norm_1.weight") parsed.coordinate.slot = 1;
                else if (suffix == "post_ffw_norm_2.weight") parsed.coordinate.slot = 2;
                else parsed.coordinate.slot = 3;
            }
            if (suffix == "ffn_gate_up_exps.weight" || suffix == "ffn_down_exps.weight") {
                parsed.coordinate.bank_axis = 2;
            }
            return {ParseState::Matched, parsed};
        }
    }
    size_t matches = 0;
    bool bias_suffix = false;
    for (size_t index = after_layer; index != parts.size(); ++index) {
        if (parts[index] == "weight") continue;
        if (parts[index] == "bias") {
            bias_suffix = true;
            continue;
        }
        const auto candidate = role_marker(parts[index]);
        if (!candidate) continue;
        role = candidate;
        ++matches;
    }
    if (matches == 0) return {ParseState::Missing, {}};
    if (matches != 1) return {ParseState::Ambiguous, {}};
    if (bias_suffix) {
        // A bias projection is a distinct typed fact. Mapping it onto the
        // weight role would put two candidates on one coordinate.
        if (*role == TensorRole::QueryWeight) *role = TensorRole::QueryBias;
        else if (*role == TensorRole::KeyWeight) *role = TensorRole::KeyBias;
        else if (*role == TensorRole::ValueWeight) *role = TensorRole::ValueBias;
        else return {ParseState::Ambiguous, {}};
    }
    ParsedRole parsed;
    parsed.role = *role;
    parsed.coordinate.root = 0;
    parsed.coordinate.layer = layer;
    if (parsed.role == TensorRole::PostNormWeight) {
        for (size_t index = after_layer; index != parts.size(); ++index) {
            if (parts[index] == "post_attention_norm") {
                parsed.coordinate.slot = 0;
                break;
            }
        }
    }
    return {ParseState::Matched, parsed};
}

NormalizedEvidenceCandidate candidate(const GgufNormalizedSourceTensor& input,
                                      const ParsedRole& parsed) {
    NormalizedEvidenceCandidate result;
    result.evidence.role = parsed.role;
    result.evidence.coordinate = parsed.coordinate;
    result.evidence.strength = NormalizedEvidenceStrength::Structural;
    result.evidence.logical_type = input.physical.logical_type;
    result.evidence.dimensions = input.physical.logical_dimensions;
    result.evidence.layout = input.physical.layout;
    result.evidence.format = input.physical.format;
    result.evidence.quantization = input.physical.quantization;
    result.binding.source_tensor_id = input.source_tensor_id;
    for (const ArtifactTensorPlane& plane : input.physical.planes) {
        result.evidence.planes.push_back({plane.kind, plane.storage_type,
                                          plane.logical_elements, plane.bytes_per_block,
                                          plane.elements_per_block, plane.alignment});
        result.binding.planes.push_back(plane.source);
    }
    if (parsed.coordinate.bank_axis != UINT8_MAX) {
        if (parsed.coordinate.bank_axis >= result.evidence.dimensions.size() ||
            result.evidence.planes.empty() || result.evidence.dimensions[parsed.coordinate.bank_axis] == 0) {
            // Leave the candidate structurally invalid; normalization will
            // reject it closed with a typed error.
            result.evidence.coordinate.bank_extent = 0;
        } else {
            result.evidence.coordinate.bank_extent = static_cast<uint32_t>(
                result.evidence.dimensions[parsed.coordinate.bank_axis]);
            const uint64_t length = result.binding.planes.front().length;
            result.evidence.coordinate.bank_stride =
                result.evidence.coordinate.bank_extent == 0 ? 0 : length / result.evidence.coordinate.bank_extent;
        }
    }
    return result;
}

GgufNormalizedSourceEvidenceResult extract(
    std::span<const GgufNormalizedSourceTensor> tensors) {
    if (tensors.empty()) {
        return normalize_source_evidence({});
    }
    if (tensors.size() > kNormalizedSourceEvidenceMaximumCandidates) {
        return compatibility_report(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                                    "GGUF source evidence exceeds its bounded candidate limit");
    }
    std::vector<NormalizedEvidenceCandidate> candidates;
    candidates.reserve(tensors.size());
    for (const GgufNormalizedSourceTensor& input : tensors) {
        if (input.physical.logical_dimensions.size() > kNormalizedSourceEvidenceMaximumDimensions ||
            input.physical.planes.size() > kNormalizedSourceEvidenceMaximumPlanes) {
            return compatibility_report(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                                        "GGUF physical evidence exceeds its bounded field limits");
        }
        const ParseResult parsed = parse_spelling(input.spelling);
        if (parsed.state == ParseState::Ambiguous) {
            return compatibility_report(CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS,
                                         "GGUF source spelling maps to multiple typed roles: " + input.spelling);
        }
        if (parsed.state == ParseState::Missing) {
            return compatibility_report(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                                         "GGUF source tensor has no typed role evidence: " + input.spelling);
        }
        candidates.push_back(candidate(input, parsed.value));
    }
    return normalize_source_evidence(candidates);
}

} // namespace

GgufNormalizedSourceEvidenceResult
extract_gguf_normalized_source_evidence(
    std::span<const GgufNormalizedSourceTensor> tensors) {
    return extract(tensors);
}

GgufNormalizedSourceEvidenceResult
extract_gguf_normalized_source_evidence(const ArtifactIndex& physical) {
    std::vector<GgufNormalizedSourceTensor> inputs;
    for (const ArtifactTensorRecord& tensor : physical.tensors()) {
        bool found = false;
        for (const ArtifactDiagnostic& diagnostic : physical.diagnostics()) {
            if (diagnostic.tensor_id != tensor.id || diagnostic.tensor_spelling.empty()) continue;
            inputs.push_back({tensor.id, diagnostic.tensor_spelling, tensor});
            found = true;
        }
        if (!found) inputs.push_back({tensor.id, {}, tensor});
    }
    return extract(inputs);
}

} // namespace Laplace
