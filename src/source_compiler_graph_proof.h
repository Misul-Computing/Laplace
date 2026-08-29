#pragma once

#include <cstdint>
#include <variant>
#include <vector>

#include "semantic_model.h"

namespace Laplace {

// This is a structural import proof only. It does not carry package authority
// and cannot be converted to a RuntimePackage or an execution plan.
struct SourceCompilerGraphProof {
    // Canonical SHA-256 of the semantic structural graph, excluding package
    // and artifact identity.
    Sha256Digest structural_digest;
    std::vector<uint32_t> topological_operator_ids;
    uint32_t input_value_count = 0;
    uint32_t output_value_count = 0;
    uint32_t value_count = 0;
    uint32_t operator_count = 0;
    uint32_t layer_count = 0;
};

using SourceCompilerGraphProofResult =
    std::variant<SourceCompilerGraphProof, CompatibilityReport>;

SourceCompilerGraphProofResult prove_source_candidate_graph(const SemanticModel& model);

// Re-proves the immutable semantic model and compares the complete witness.
// Downstream consumers can use this to reject a stale or mismatched proof.
bool source_compiler_graph_proof_matches(const SemanticModel& model,
                                         const SourceCompilerGraphProof& proof);

} // namespace Laplace
