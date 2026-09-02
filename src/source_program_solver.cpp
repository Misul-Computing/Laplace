#include "source_program_solver.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <tuple>

namespace Laplace {
namespace {

constexpr size_t kMaximumCandidates = 256;
constexpr size_t kMaximumPredicates = 4096;

CompatibilityReport solver_error(CompatibilityError code, const char* detail) {
    CompatibilityReport report = compatibility_report(code, detail);
    report.stage = CompatibilityStage::Import;
    report.phase = CompatibilityPhase::Import;
    return report;
}

bool zero_digest(const Sha256Digest& digest) {
    return std::all_of(digest.bytes.begin(), digest.bytes.end(),
                       [](uint8_t value) { return value == 0; });
}

class DigestWriter {
public:
    DigestWriter() { CC_SHA256_Init(&context_); }
    void bytes(std::span<const uint8_t> value) {
        while (!value.empty()) {
            const size_t count = std::min<size_t>(value.size(), 1u << 20);
            CC_SHA256_Update(&context_, value.data(),
                             static_cast<CC_LONG>(count));
            value = value.subspan(count);
        }
    }
    void u8(uint8_t value) { bytes({&value, 1}); }
    void u16(uint16_t value) {
        std::array<uint8_t, 2> wire = {
            static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8)};
        bytes(wire);
    }
    void u64(uint64_t value) {
        std::array<uint8_t, 8> wire{};
        for (unsigned shift = 0; shift != 64; shift += 8)
            wire[shift / 8] = static_cast<uint8_t>(value >> shift);
        bytes(wire);
    }
    Sha256Digest finish() {
        Sha256Digest result;
        CC_SHA256_Final(result.bytes.data(), &context_);
        return result;
    }

private:
    CC_SHA256_CTX context_{};
};

bool digest_property(SourceProgramProperty property) {
    return property >= SourceProgramProperty::PackageDigest &&
           property <= SourceProgramProperty::TokenPromptDigest;
}

bool scalar_property(SourceProgramProperty property) {
    return property >= SourceProgramProperty::FunctionCount &&
           property <= SourceProgramProperty::PrimitiveOccurrenceCount;
}

bool valid_predicate(const SourceProgramPredicate& predicate) {
    if (!digest_property(predicate.property) &&
        !scalar_property(predicate.property))
        return false;
    if (digest_property(predicate.property))
        return predicate.selector == 0 && predicate.scalar == 0 &&
               !zero_digest(predicate.digest);
    if (!zero_digest(predicate.digest)) return false;
    if (predicate.property == SourceProgramProperty::PrimitiveOccurrenceCount)
        return predicate.selector >= static_cast<uint16_t>(Primitive::Constant) &&
               predicate.selector < kPrimitiveCount;
    return predicate.selector == 0;
}

auto predicate_key(const SourceProgramPredicate& predicate) {
    return std::pair{static_cast<uint8_t>(predicate.property),
                     predicate.selector};
}

void append_predicate(DigestWriter& writer,
                      const SourceProgramPredicate& predicate) {
    writer.u8(static_cast<uint8_t>(predicate.property));
    writer.u16(predicate.selector);
    writer.u64(predicate.scalar);
    writer.bytes(predicate.digest.bytes);
}

Sha256Digest summary_digest(const SourceProgramSummary& summary) {
    DigestWriter writer;
    static constexpr std::array<uint8_t, 25> domain = {
        'l','a','p','l','a','c','e','-','s','o','u','r','c','e','-','s','u','m','m','a','r','y','-','v','1'};
    writer.bytes(domain);
    writer.bytes(summary.package_digest.bytes);
    writer.bytes(summary.semantic_program_digest.bytes);
    writer.bytes(summary.state_schema_digest.bytes);
    writer.bytes(summary.physical_package_digest.bytes);
    writer.bytes(summary.token_vocabulary_digest.bytes);
    writer.bytes(summary.token_prompt_digest.bytes);
    writer.u64(summary.function_count);
    writer.u64(summary.export_count);
    writer.u64(summary.state_reference_count);
    writer.u64(summary.physical_program_count);
    writer.u64(summary.physical_resource_count);
    for (uint64_t count : summary.primitive_occurrences) writer.u64(count);
    return writer.finish();
}

bool complete_summary(const SourceProgramSummary& summary) {
    return !zero_digest(summary.package_digest) &&
           !zero_digest(summary.semantic_program_digest) &&
           !zero_digest(summary.state_schema_digest) &&
           !zero_digest(summary.physical_package_digest) &&
           !zero_digest(summary.token_vocabulary_digest) &&
           !zero_digest(summary.token_prompt_digest) &&
           summary.function_count != 0 && summary.export_count != 0 &&
           summary.physical_program_count != 0 &&
           summary.physical_resource_count != 0;
}

bool matches(const SourceProgramSummary& summary,
             const SourceProgramPredicate& predicate) {
    switch (predicate.property) {
    case SourceProgramProperty::PackageDigest:
        return summary.package_digest == predicate.digest;
    case SourceProgramProperty::SemanticProgramDigest:
        return summary.semantic_program_digest == predicate.digest;
    case SourceProgramProperty::StateSchemaDigest:
        return summary.state_schema_digest == predicate.digest;
    case SourceProgramProperty::PhysicalPackageDigest:
        return summary.physical_package_digest == predicate.digest;
    case SourceProgramProperty::TokenVocabularyDigest:
        return summary.token_vocabulary_digest == predicate.digest;
    case SourceProgramProperty::TokenPromptDigest:
        return summary.token_prompt_digest == predicate.digest;
    case SourceProgramProperty::FunctionCount:
        return summary.function_count == predicate.scalar;
    case SourceProgramProperty::ExportCount:
        return summary.export_count == predicate.scalar;
    case SourceProgramProperty::StateReferenceCount:
        return summary.state_reference_count == predicate.scalar;
    case SourceProgramProperty::PhysicalProgramCount:
        return summary.physical_program_count == predicate.scalar;
    case SourceProgramProperty::PhysicalResourceCount:
        return summary.physical_resource_count == predicate.scalar;
    case SourceProgramProperty::PrimitiveOccurrenceCount:
        return predicate.selector < summary.primitive_occurrences.size() &&
               summary.primitive_occurrences[predicate.selector] ==
                   predicate.scalar;
    }
    return false;
}

std::variant<std::vector<SourceProgramPredicate>, CompatibilityReport>
canonical_predicates(SourceEvidence evidence) {
    if (evidence.major != 1 || evidence.minor != 0)
        return solver_error(CompatibilityError::IR_VERSION_UNSUPPORTED,
                            "source evidence version is unsupported");
    if (evidence.predicates.empty() ||
        evidence.predicates.size() > kMaximumPredicates)
        return solver_error(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                            "source evidence predicate count is invalid");
    for (const auto& predicate : evidence.predicates)
        if (!valid_predicate(predicate))
            return solver_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                                "source evidence predicate is malformed");
    std::sort(evidence.predicates.begin(), evidence.predicates.end(),
              [](const auto& left, const auto& right) {
                  return predicate_key(left) < predicate_key(right);
              });
    for (size_t index = 1; index < evidence.predicates.size(); ++index)
        if (predicate_key(evidence.predicates[index - 1]) ==
            predicate_key(evidence.predicates[index]))
            return solver_error(CompatibilityError::IMPORT_RULE_CONFLICT,
                                "source evidence repeats one property");
    return std::move(evidence.predicates);
}

SourceEvidenceDigest evidence_digest(
    const SourceEvidence& evidence,
    std::span<const SourceProgramPredicate> predicates) {
    DigestWriter writer;
    static constexpr std::array<uint8_t, 26> domain = {
        'l','a','p','l','a','c','e','-','s','o','u','r','c','e','-','e','v','i','d','e','n','c','e','-','v','1'};
    writer.bytes(domain);
    writer.u16(evidence.major);
    writer.u16(evidence.minor);
    writer.bytes(evidence.candidate_set_digest.bytes);
    writer.u64(predicates.size());
    for (const auto& predicate : predicates) append_predicate(writer, predicate);
    return {writer.finish()};
}

} // namespace

SourceCandidateSetDigestResult source_candidate_set_digest(
    std::span<const SourceProgramSummary> candidates) {
    if (candidates.empty() || candidates.size() > kMaximumCandidates)
        return solver_error(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                            "source candidate count is invalid");
    std::vector<std::pair<Sha256Digest, Sha256Digest>> ordered;
    ordered.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        if (!complete_summary(candidate))
            return solver_error(CompatibilityError::IMPORT_CLOSURE_INCOMPLETE,
                                "source candidate summary is incomplete");
        ordered.push_back({candidate.package_digest, summary_digest(candidate)});
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto& left,
                                                  const auto& right) {
        return left.first.bytes < right.first.bytes;
    });
    for (size_t index = 1; index < ordered.size(); ++index)
        if (ordered[index - 1].first == ordered[index].first)
            return solver_error(CompatibilityError::IMPORT_SCHEMA_AMBIGUOUS,
                                "source candidates repeat a package digest");
    DigestWriter writer;
    static constexpr std::array<uint8_t, 31> domain = {
        'l','a','p','l','a','c','e','-','s','o','u','r','c','e','-','c','a','n','d','i','d','a','t','e','-','s','e','t','-','v','1'};
    writer.bytes(domain);
    writer.u64(ordered.size());
    for (const auto& [package, summary] : ordered) {
        writer.bytes(package.bytes);
        writer.bytes(summary.bytes);
    }
    return SourceCandidateSetDigest{writer.finish()};
}

SourceProgramProofResult solve_source_program(
    SourceEvidence evidence,
    std::span<const SourceProgramSummary> candidates) {
    auto set = source_candidate_set_digest(candidates);
    if (const auto* report = std::get_if<CompatibilityReport>(&set))
        return *report;
    const auto set_digest = std::get<SourceCandidateSetDigest>(set);
    if (evidence.candidate_set_digest != set_digest.bytes)
        return solver_error(CompatibilityError::AUTHORITY_INVALID,
                            "source evidence candidate set digest does not match");
    auto canonical = canonical_predicates(evidence);
    if (const auto* report = std::get_if<CompatibilityReport>(&canonical))
        return *report;
    const auto& predicates =
        std::get<std::vector<SourceProgramPredicate>>(canonical);
    const SourceProgramSummary* selected = nullptr;
    for (const auto& candidate : candidates) {
        const bool candidate_matches = std::all_of(
            predicates.begin(), predicates.end(),
            [&](const auto& predicate) { return matches(candidate, predicate); });
        if (!candidate_matches) continue;
        if (selected != nullptr)
            return solver_error(CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS,
                                "source evidence admits multiple semantic programs");
        selected = &candidate;
    }
    if (selected == nullptr)
        return solver_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                            "source evidence admits no semantic program");
    SourceProgramProof proof;
    proof.evidence_digest = evidence_digest(evidence, predicates);
    proof.candidate_set_digest = set_digest;
    proof.package_digest = selected->package_digest;
    proof.matched_predicates = predicates;
    return proof;
}

bool source_program_proof_matches(
    const SourceProgramProof& proof, SourceEvidence evidence,
    const SourceProgramSummary& candidate,
    std::span<const SourceProgramSummary> candidate_set) {
    auto solved = solve_source_program(std::move(evidence), candidate_set);
    const auto* current = std::get_if<SourceProgramProof>(&solved);
    return current != nullptr &&
           current->evidence_digest == proof.evidence_digest &&
           current->candidate_set_digest == proof.candidate_set_digest &&
           current->package_digest == proof.package_digest &&
           current->matched_predicates == proof.matched_predicates &&
           candidate.package_digest == proof.package_digest &&
           std::find(candidate_set.begin(), candidate_set.end(), candidate) !=
               candidate_set.end();
}

} // namespace Laplace
