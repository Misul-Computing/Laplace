#include "mlx_normalized_source_adapter.h"

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
    if (value == "input_layernorm" || value == "norm1")
        return TensorRole::AttentionNormWeight;
    if (value == "q_proj" || value == "query" || value == "q") return TensorRole::QueryWeight;
    if (value == "k_proj" || value == "key" || value == "k") return TensorRole::KeyWeight;
    if (value == "v_proj" || value == "value" || value == "v") return TensorRole::ValueWeight;
    if (value == "o_proj" || value == "output" || value == "out_proj")
        return TensorRole::AttentionOutputWeight;
    if (value == "post_attention_layernorm" || value == "norm2")
        return TensorRole::FfnNormWeight;
    if (value == "gate_proj" || value == "gate") return TensorRole::FfnGateWeight;
    if (value == "up_proj" || value == "up") return TensorRole::FfnUpWeight;
    if (value == "down_proj" || value == "down") return TensorRole::FfnDownWeight;
    return std::nullopt;
}

ParseResult parse_spelling(std::string_view spelling) {
    if (spelling == "embeddings.word.weight" || spelling == "model.embed_tokens.weight") {
        return {ParseState::Matched, {TensorRole::TokenEmbedding, {0}}};
    }
    if (spelling == "model.norm.weight" || spelling == "final_norm.weight") {
        return {ParseState::Matched, {TensorRole::FinalNormWeight, {0}}};
    }
    if (spelling == "lm_head.weight" || spelling == "output.weight") {
        return {ParseState::Matched, {TensorRole::OutputWeight, {0}}};
    }

    const auto parts = split(spelling);
    uint32_t layer = UINT32_MAX;
    size_t after_layer = parts.size();
    for (size_t index = 0; index + 1 < parts.size(); ++index) {
        if (parts[index] != "layers" && parts[index] != "blocks") continue;
        if (layer != UINT32_MAX || !decimal(parts[index + 1], layer)) {
            return {ParseState::Ambiguous, {}};
        }
        after_layer = index + 2;
    }
    if (layer == UINT32_MAX || after_layer == parts.size()) return {ParseState::Missing, {}};

    std::optional<TensorRole> role;
    size_t matches = 0;
    for (size_t index = after_layer; index != parts.size(); ++index) {
        if (parts[index] == "weight") continue;
        const auto candidate = role_marker(parts[index]);
        if (!candidate) continue;
        role = candidate;
        ++matches;
    }
    if (matches == 0) return {ParseState::Missing, {}};
    if (matches != 1) return {ParseState::Ambiguous, {}};
    ParsedRole parsed;
    parsed.role = *role;
    parsed.coordinate.root = 0;
    parsed.coordinate.layer = layer;
    return {ParseState::Matched, parsed};
}

NormalizedEvidenceCandidate candidate(const MlxNormalizedSourceTensor& input,
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
    return result;
}

MlxNormalizedSourceEvidenceResult extract(
    std::span<const MlxNormalizedSourceTensor> tensors) {
    if (tensors.empty()) {
        return normalize_source_evidence({});
    }
    if (tensors.size() > kNormalizedSourceEvidenceMaximumCandidates) {
        return compatibility_report(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                                    "MLX source evidence exceeds its bounded candidate limit");
    }
    std::vector<NormalizedEvidenceCandidate> candidates;
    candidates.reserve(tensors.size());
    for (const MlxNormalizedSourceTensor& input : tensors) {
        if (input.physical.logical_dimensions.size() > kNormalizedSourceEvidenceMaximumDimensions ||
            input.physical.planes.size() > kNormalizedSourceEvidenceMaximumPlanes) {
            return compatibility_report(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                                        "MLX physical evidence exceeds its bounded field limits");
        }
        const ParseResult parsed = parse_spelling(input.spelling);
        if (parsed.state == ParseState::Ambiguous) {
            return compatibility_report(CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS,
                                         "MLX source spelling maps to multiple typed roles");
        }
        if (parsed.state == ParseState::Missing) {
            return compatibility_report(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                                         "MLX source tensor has no typed role evidence");
        }
        candidates.push_back(candidate(input, parsed.value));
    }
    return normalize_source_evidence(candidates);
}

} // namespace

MlxNormalizedSourceEvidenceResult
extract_mlx_normalized_source_evidence(
    std::span<const MlxNormalizedSourceTensor> tensors) {
    return extract(tensors);
}

MlxNormalizedSourceEvidenceResult
extract_mlx_normalized_source_evidence(const ArtifactIndex& physical) {
    std::vector<MlxNormalizedSourceTensor> inputs;
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
