#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "gguf_normalized_source_adapter.h"
#include "mlx_normalized_source_adapter.h"
#include "normalized_source_evidence.h"
#include "test_util.h"

using namespace Laplace;

namespace {

ArtifactTensorRecord physical(uint32_t id, ArtifactScalarType scalar,
                              std::vector<uint64_t> dimensions,
                              uint64_t offset) {
    ArtifactTensorRecord record;
    record.id = id;
    record.logical_type = scalar;
    record.logical_dimensions = std::move(dimensions);
    record.layout.kind = PhysicalLayoutKind::ContiguousRowMajor;
    record.layout.version = 1;
    record.layout.packing = PackingKind::None;
    record.layout.rank = static_cast<uint8_t>(record.logical_dimensions.size());
    uint64_t stride = 1;
    for (size_t axis = 0; axis != record.logical_dimensions.size(); ++axis) {
        record.layout.axis_order[axis] = static_cast<uint8_t>(axis);
        record.layout.strides[axis] = stride;
        stride *= record.logical_dimensions[axis];
    }
    record.format.version = 1;
    record.format.encoding = scalar == ArtifactScalarType::F16
        ? ArtifactPhysicalEncoding::F16 : ArtifactPhysicalEncoding::F32;
    record.format.value_type = scalar;
    record.format.block_elements = 1;
    record.format.block_bytes = scalar == ArtifactScalarType::F16 ? 2 : 4;
    record.axis.source_rank = record.layout.rank;
    record.axis.block_axis = UINT8_MAX;
    record.axis.row_stride_bytes = stride * record.format.block_bytes;
    record.planes.push_back({PlaneKind::Values, scalar,
                             {ArtifactId{0}, offset,
                              stride * record.format.block_bytes},
                             stride, record.format.block_bytes, 1, 32});
    return record;
}

bool success(const NormalizedSourceEvidenceResult& result) {
    return std::holds_alternative<NormalizedSourceEvidence>(result);
}

NormalizedEvidenceCandidate normalized_candidate(TensorRole role) {
    NormalizedEvidenceCandidate candidate;
    candidate.evidence.role = role;
    candidate.evidence.coordinate.root = 0;
    candidate.evidence.logical_type = ArtifactScalarType::F32;
    candidate.evidence.dimensions = {8};
    candidate.evidence.layout.kind = PhysicalLayoutKind::ContiguousRowMajor;
    candidate.evidence.layout.version = 1;
    candidate.evidence.layout.packing = PackingKind::None;
    candidate.evidence.layout.rank = 1;
    candidate.evidence.layout.axis_order[0] = 0;
    candidate.evidence.layout.strides[0] = 1;
    candidate.evidence.format.version = 1;
    candidate.evidence.format.encoding = ArtifactPhysicalEncoding::F32;
    candidate.evidence.format.value_type = ArtifactScalarType::F32;
    candidate.evidence.format.block_elements = 1;
    candidate.evidence.format.block_bytes = 4;
    candidate.evidence.quantization.kind = QuantizationKind::None;
    candidate.evidence.planes = {{PlaneKind::Values, ArtifactScalarType::F32,
                                  8, 4, 1, 32}};
    candidate.binding.source_tensor_id = 1;
    candidate.binding.planes = {{ArtifactId{0}, 0, 32}};
    return candidate;
}

const CompatibilityReport& failure(const NormalizedSourceEvidenceResult& result) {
    return std::get<CompatibilityReport>(result);
}

void test_equivalent_adapters_strip_spellings() {
    const ArtifactTensorRecord q = physical(10, ArtifactScalarType::F16, {8, 8}, 128);
    const ArtifactTensorRecord gate = physical(11, ArtifactScalarType::F16, {8, 16}, 256);
    const ArtifactTensorRecord embedding = physical(12, ArtifactScalarType::F16, {32, 8}, 512);

    const std::vector<GgufNormalizedSourceTensor> gguf = {
        {10, "blk.3.attn_q.weight", q},
        {11, "blk.3.ffn_gate.weight", gate},
        {12, "token_embd.weight", embedding},
    };
    const std::vector<MlxNormalizedSourceTensor> mlx = {
        {212, "embeddings.word.weight", physical(212, ArtifactScalarType::F16, {32, 8}, 12288)},
        {210, "model.layers.3.self_attn.q_proj.weight", physical(210, ArtifactScalarType::F16, {8, 8}, 4096)},
        {211, "model.layers.3.mlp.gate_proj.weight", physical(211, ArtifactScalarType::F16, {8, 16}, 8192)},
    };

    auto left = extract_gguf_normalized_source_evidence(gguf);
    auto right = extract_mlx_normalized_source_evidence(mlx);
    CHECK(success(left));
    CHECK(success(right));
    if (success(left) && success(right)) {
        const auto& a = std::get<NormalizedSourceEvidence>(left);
        const auto& b = std::get<NormalizedSourceEvidence>(right);
        CHECK(a.tensors().size() == b.tensors().size());
        CHECK(std::equal(a.tensors().begin(), a.tensors().end(), b.tensors().begin()));
        CHECK(a.canonical_digest() == b.canonical_digest());
        CHECK(a.canonical_bytes().size() != 0);
        const auto contains = [&](std::string_view spelling) {
            const auto& bytes = a.canonical_bytes();
            return std::search(bytes.begin(), bytes.end(), spelling.begin(), spelling.end()) != bytes.end();
        };
        CHECK(!contains("blk.3"));
        CHECK(!contains("self_attn"));
        CHECK(a.bindings().size() == 3);
        CHECK(b.bindings().size() == 3);
        CHECK(a.bindings()[0].source_tensor_id != b.bindings()[0].source_tensor_id);
    }
}

void test_missing_evidence_fails_closed() {
    const std::vector<GgufNormalizedSourceTensor> input = {
        {1, "unrelated.tensor", physical(1, ArtifactScalarType::F16, {8}, 0)},
    };
    auto result = extract_gguf_normalized_source_evidence(input);
    CHECK(!success(result));
    if (!success(result)) CHECK(failure(result).code == CompatibilityError::IMPORT_TENSOR_UNMAPPED);
}

void test_ambiguous_spelling_fails_closed() {
    const std::vector<GgufNormalizedSourceTensor> input = {
        {1, "blk.0.attn_q.q_proj.weight", physical(1, ArtifactScalarType::F16, {8, 8}, 0)},
    };
    auto result = extract_gguf_normalized_source_evidence(input);
    CHECK(!success(result));
    if (!success(result)) CHECK(failure(result).code == CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS);
}

void test_duplicate_evidence_fails_closed() {
    const auto record = physical(1, ArtifactScalarType::F16, {8, 8}, 0);
    const std::vector<GgufNormalizedSourceTensor> input = {
        {1, "blk.0.attn_q.weight", record},
        {2, "blk.0.attn_q.weight", record},
    };
    auto result = extract_gguf_normalized_source_evidence(input);
    CHECK(!success(result));
    if (!success(result)) CHECK(failure(result).code == CompatibilityError::IMPORT_TENSOR_DUPLICATE);
}

void test_conflicting_evidence_fails_closed() {
    const std::vector<GgufNormalizedSourceTensor> input = {
        {1, "blk.0.attn_q.weight", physical(1, ArtifactScalarType::F16, {8, 8}, 0)},
        {2, "blk.0.attn_q.weight", physical(2, ArtifactScalarType::F32, {8, 8}, 256)},
    };
    auto result = extract_gguf_normalized_source_evidence(input);
    CHECK(!success(result));
    if (!success(result)) CHECK(failure(result).code == CompatibilityError::IMPORT_RULE_CONFLICT);
}

void test_empty_input_fails_closed() {
    auto result = extract_mlx_normalized_source_evidence(
        std::span<const MlxNormalizedSourceTensor>{});
    CHECK(!success(result));
    if (!success(result)) CHECK(failure(result).code == CompatibilityError::IMPORT_SEMANTICS_MISSING);
}

void test_oversized_physical_evidence_fails_closed() {
    ArtifactTensorRecord oversized = physical(1, ArtifactScalarType::F16, {8}, 0);
    oversized.logical_dimensions.resize(kNormalizedSourceEvidenceMaximumDimensions + 1, 1);
    const std::vector<GgufNormalizedSourceTensor> input = {
        {1, "token_embd.weight", std::move(oversized)},
    };
    auto result = extract_gguf_normalized_source_evidence(input);
    CHECK(!success(result));
    if (!success(result)) CHECK(failure(result).code == CompatibilityError::IMPORT_SCHEMA_LIMIT);
}

void test_moe_auxiliary_roles_are_typed_evidence() {
    for (TensorRole role : {TensorRole::RouterScaleWeight,
                            TensorRole::ExpertNormWeight,
                            TensorRole::ReduceScaleWeight,
                            TensorRole::PostNormWeight,
                            TensorRole::OutputScaleWeight}) {
        const NormalizedEvidenceCandidate candidate = normalized_candidate(role);
        const auto result = normalize_source_evidence(
            std::span<const NormalizedEvidenceCandidate>(&candidate, 1));
        CHECK(success(result));
    }
}

} // namespace

int main() {
    test_equivalent_adapters_strip_spellings();
    test_missing_evidence_fails_closed();
    test_ambiguous_spelling_fails_closed();
    test_duplicate_evidence_fails_closed();
    test_conflicting_evidence_fails_closed();
    test_empty_input_fails_closed();
    test_oversized_physical_evidence_fails_closed();
    test_moe_auxiliary_roles_are_typed_evidence();
    return test_summary("test_normalized_source_evidence");
}
