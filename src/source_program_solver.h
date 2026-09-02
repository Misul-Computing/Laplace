#pragma once

#include "source_evidence.h"

namespace Laplace {

using SourceCandidateSetDigestResult =
    std::variant<SourceCandidateSetDigest, CompatibilityReport>;
using SourceProgramProofResult =
    std::variant<SourceProgramProof, CompatibilityReport>;

SourceCandidateSetDigestResult source_candidate_set_digest(
    std::span<const SourceProgramSummary> candidates);

SourceProgramProofResult solve_source_program(
    SourceEvidence evidence,
    std::span<const SourceProgramSummary> candidates);

bool source_program_proof_matches(
    const SourceProgramProof& proof, SourceEvidence evidence,
    const SourceProgramSummary& candidate,
    std::span<const SourceProgramSummary> candidate_set);

} // namespace Laplace
