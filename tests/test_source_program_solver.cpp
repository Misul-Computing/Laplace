#include "source_program_solver.h"
#include "test_util.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>
#include <variant>
#include <vector>

using namespace Laplace;

namespace {

Sha256Digest digest(uint8_t seed) {
    Sha256Digest result;
    for (size_t index = 0; index < result.bytes.size(); ++index)
        result.bytes[index] = static_cast<uint8_t>(seed + index * 17);
    return result;
}

SourceProgramSummary summary(uint8_t seed, uint64_t multiply_count = 2) {
    SourceProgramSummary result;
    result.package_digest = digest(seed);
    result.semantic_program_digest = digest(seed + 1);
    result.state_schema_digest = digest(seed + 2);
    result.physical_package_digest = digest(seed + 3);
    result.token_vocabulary_digest = digest(seed + 4);
    result.token_prompt_digest = digest(seed + 5);
    result.function_count = 3;
    result.export_count = 2;
    result.state_reference_count = 1;
    result.physical_program_count = 2;
    result.physical_resource_count = 4;
    result.primitive_occurrences[static_cast<size_t>(Primitive::Multiply)] =
        multiply_count;
    result.primitive_occurrences[
        static_cast<size_t>(Primitive::StructuredTensor)] = 1;
    return result;
}

SourceProgramPredicate scalar(SourceProgramProperty property, uint64_t value,
                              Primitive primitive = Primitive::Constant) {
    SourceProgramPredicate result;
    result.property = property;
    result.scalar = value;
    if (property == SourceProgramProperty::PrimitiveOccurrenceCount)
        result.selector = static_cast<uint16_t>(primitive);
    return result;
}

SourceProgramPredicate exact(SourceProgramProperty property,
                             Sha256Digest value) {
    SourceProgramPredicate result;
    result.property = property;
    result.digest = value;
    return result;
}

SourceEvidence evidence_for(std::span<const SourceProgramSummary> candidates,
                            std::vector<SourceProgramPredicate> predicates) {
    const auto set = source_candidate_set_digest(candidates);
    CHECK(std::holds_alternative<SourceCandidateSetDigest>(set));
    SourceEvidence evidence;
    if (const auto* digest = std::get_if<SourceCandidateSetDigest>(&set))
        evidence.candidate_set_digest = digest->bytes;
    evidence.predicates = std::move(predicates);
    return evidence;
}

const CompatibilityReport* report(const SourceProgramProofResult& result) {
    return std::get_if<CompatibilityReport>(&result);
}

void test_weight_only_evidence_is_ambiguous() {
    const std::array candidates = {summary(1), summary(20)};
    auto evidence = evidence_for(
        candidates,
        {scalar(SourceProgramProperty::FunctionCount, 3),
         scalar(SourceProgramProperty::PhysicalResourceCount, 4)});
    const auto result = solve_source_program(evidence, candidates);
    CHECK(report(result) != nullptr);
    if (report(result)) {
        CHECK(report(result)->code ==
              CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS);
    }
}

void test_carried_equation_selects_one_complete_program() {
    const std::array candidates = {summary(1), summary(20)};
    auto evidence = evidence_for(
        candidates,
        {scalar(SourceProgramProperty::FunctionCount, 3),
         exact(SourceProgramProperty::SemanticProgramDigest,
               candidates[1].semantic_program_digest),
         scalar(SourceProgramProperty::PrimitiveOccurrenceCount, 2,
                Primitive::Multiply)});
    const auto result = solve_source_program(evidence, candidates);
    CHECK(std::holds_alternative<SourceProgramProof>(result));
    if (const auto* proof = std::get_if<SourceProgramProof>(&result)) {
        CHECK(proof->package_digest == candidates[1].package_digest);
        CHECK(proof->matched_predicates.size() == 3);
        CHECK(source_program_proof_matches(*proof, evidence, candidates[1],
                                           candidates));
        auto changed = candidates[1];
        changed.primitive_occurrences[
            static_cast<size_t>(Primitive::Multiply)] = 3;
        CHECK(!source_program_proof_matches(*proof, evidence, changed,
                                            candidates));
    }
}

void test_candidate_and_predicate_order_do_not_change_proof() {
    const std::array forward = {summary(1), summary(20)};
    const std::array reverse = {forward[1], forward[0]};
    auto left = evidence_for(
        forward,
        {scalar(SourceProgramProperty::FunctionCount, 3),
         exact(SourceProgramProperty::PackageDigest,
               forward[0].package_digest)});
    auto right = evidence_for(
        reverse,
        {exact(SourceProgramProperty::PackageDigest,
               forward[0].package_digest),
         scalar(SourceProgramProperty::FunctionCount, 3)});
    CHECK(left.candidate_set_digest == right.candidate_set_digest);
    const auto left_result = solve_source_program(left, forward);
    const auto right_result = solve_source_program(right, reverse);
    CHECK(std::holds_alternative<SourceProgramProof>(left_result));
    CHECK(std::holds_alternative<SourceProgramProof>(right_result));
    if (const auto* a = std::get_if<SourceProgramProof>(&left_result)) {
        if (const auto* b = std::get_if<SourceProgramProof>(&right_result)) {
            CHECK(a->evidence_digest == b->evidence_digest);
            CHECK(a->candidate_set_digest == b->candidate_set_digest);
            CHECK(a->package_digest == b->package_digest);
            CHECK(a->matched_predicates == b->matched_predicates);
        }
    }
}

void test_tamper_and_incomplete_inputs_fail_closed() {
    const std::array candidates = {summary(1), summary(20)};
    auto evidence = evidence_for(
        candidates,
        {exact(SourceProgramProperty::PackageDigest,
               candidates[0].package_digest)});
    evidence.candidate_set_digest.bytes[0] ^= 1;
    auto result = solve_source_program(evidence, candidates);
    CHECK(report(result) != nullptr);
    if (report(result))
        CHECK(report(result)->code == CompatibilityError::AUTHORITY_INVALID);

    evidence = evidence_for(
        candidates,
        {scalar(SourceProgramProperty::FunctionCount, 3),
         scalar(SourceProgramProperty::FunctionCount, 4)});
    result = solve_source_program(evidence, candidates);
    CHECK(report(result) != nullptr);
    if (report(result))
        CHECK(report(result)->code == CompatibilityError::IMPORT_RULE_CONFLICT);

    auto malformed = exact(SourceProgramProperty::SemanticProgramDigest, {});
    evidence = evidence_for(candidates, {malformed});
    result = solve_source_program(evidence, candidates);
    CHECK(report(result) != nullptr);
    if (report(result))
        CHECK(report(result)->code ==
              CompatibilityError::IMPORT_SCHEMA_INCOMPLETE);

    auto incomplete = candidates[0];
    incomplete.physical_package_digest = {};
    const std::array bad = {incomplete};
    CHECK(std::holds_alternative<CompatibilityReport>(
        source_candidate_set_digest(bad)));
}

void test_no_match_and_replayed_candidate_fail_closed() {
    const std::array candidates = {summary(1), summary(20)};
    auto evidence = evidence_for(
        candidates,
        {scalar(SourceProgramProperty::PrimitiveOccurrenceCount, 99,
                Primitive::StructuredTensor)});
    const auto missing = solve_source_program(evidence, candidates);
    CHECK(report(missing) != nullptr);
    if (report(missing))
        CHECK(report(missing)->code ==
              CompatibilityError::IMPORT_SEMANTICS_MISSING);

    const std::array replay = {candidates[0], candidates[0]};
    const auto replayed = source_candidate_set_digest(replay);
    CHECK(std::holds_alternative<CompatibilityReport>(replayed));
    if (const auto* failure = std::get_if<CompatibilityReport>(&replayed))
        CHECK(failure->code == CompatibilityError::IMPORT_SCHEMA_AMBIGUOUS);
}

} // namespace

int main() {
    static_assert(!std::is_convertible_v<SourceProgramProperty, const char*>);
    test_weight_only_evidence_is_ambiguous();
    test_carried_equation_selects_one_complete_program();
    test_candidate_and_predicate_order_do_not_change_proof();
    test_tamper_and_incomplete_inputs_fail_closed();
    test_no_match_and_replayed_candidate_fail_closed();
    return test_summary("test_source_program_solver");
}
