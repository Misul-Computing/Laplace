#include "source_evidence.h"

#include "program_package.h"

namespace Laplace {

SourceProgramSummaryResult
summarize_source_program(const VerifiedProgramPackage& package) {
    SourceProgramSummary summary;
    summary.package_digest = package.digest();
    summary.semantic_program_digest.bytes =
        program_digest(package.semantic_program()).bytes;
    summary.state_schema_digest.bytes =
        state_schema_digest(package.state_schema()).bytes;
    summary.physical_package_digest = package.physical_package().digest();
    summary.token_vocabulary_digest = package.token_program().vocabulary_digest();
    summary.token_prompt_digest = package.token_program().prompt_digest();

    const Program& program = program_definition(package.semantic_program());
    summary.function_count = program.functions.size();
    summary.export_count = program.exports.size();
    summary.state_reference_count = program.state_references.size();
    summary.physical_program_count = package.physical_package().programs().size();
    summary.physical_resource_count = package.physical_package().resources().size();
    for (const Function& function : program.functions) {
        for (const Region& region : function.regions) {
            for (const Instruction& instruction : region.instructions) {
                const uint16_t code =
                    static_cast<uint16_t>(instruction.primitive.code);
                if (code == 0 || code >= summary.primitive_occurrences.size()) {
                    CompatibilityReport report = compatibility_report(
                        CompatibilityError::IR_VERSION_UNSUPPORTED,
                        "semantic program contains an uncountable primitive");
                    report.stage = CompatibilityStage::Import;
                    return report;
                }
                ++summary.primitive_occurrences[code];
            }
        }
    }
    return summary;
}

} // namespace Laplace
