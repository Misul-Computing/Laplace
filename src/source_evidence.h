#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include "artifact_set.h"
#include "compatibility_report.h"
#include "program_ir.h"

namespace Laplace {

class VerifiedProgramPackage;

enum class SourceProgramProperty : uint8_t {
    PackageDigest = 1,
    SemanticProgramDigest = 2,
    StateSchemaDigest = 3,
    PhysicalPackageDigest = 4,
    TokenVocabularyDigest = 5,
    TokenPromptDigest = 6,
    FunctionCount = 7,
    ExportCount = 8,
    StateReferenceCount = 9,
    PhysicalProgramCount = 10,
    PhysicalResourceCount = 11,
    PrimitiveOccurrenceCount = 12,
};

struct SourceProgramPredicate {
    SourceProgramProperty property = SourceProgramProperty::PackageDigest;
    // Used only by PrimitiveOccurrenceCount. It is the serialized Primitive
    // value, not a model, tensor, format, or source-container identifier.
    uint16_t selector = 0;
    uint64_t scalar = 0;
    Sha256Digest digest{};
    friend bool operator==(const SourceProgramPredicate&,
                           const SourceProgramPredicate&) = default;
};

struct SourceProgramSummary {
    Sha256Digest package_digest{};
    Sha256Digest semantic_program_digest{};
    Sha256Digest state_schema_digest{};
    Sha256Digest physical_package_digest{};
    Sha256Digest token_vocabulary_digest{};
    Sha256Digest token_prompt_digest{};
    uint64_t function_count = 0;
    uint64_t export_count = 0;
    uint64_t state_reference_count = 0;
    uint64_t physical_program_count = 0;
    uint64_t physical_resource_count = 0;
    std::array<uint64_t, kPrimitiveCount> primitive_occurrences{};
    friend bool operator==(const SourceProgramSummary&,
                           const SourceProgramSummary&) = default;
};

struct SourceEvidence {
    uint16_t major = 1;
    uint16_t minor = 0;
    Sha256Digest candidate_set_digest{};
    std::vector<SourceProgramPredicate> predicates;
};

struct SourceEvidenceDigest {
    Sha256Digest bytes{};
    friend bool operator==(SourceEvidenceDigest, SourceEvidenceDigest) = default;
};

struct SourceCandidateSetDigest {
    Sha256Digest bytes{};
    friend bool operator==(SourceCandidateSetDigest,
                           SourceCandidateSetDigest) = default;
};

struct SourceProgramProof {
    SourceEvidenceDigest evidence_digest{};
    SourceCandidateSetDigest candidate_set_digest{};
    Sha256Digest package_digest{};
    std::vector<SourceProgramPredicate> matched_predicates;
};

using SourceProgramSummaryResult =
    std::variant<SourceProgramSummary, CompatibilityReport>;

SourceProgramSummaryResult
summarize_source_program(const VerifiedProgramPackage& package);

} // namespace Laplace
