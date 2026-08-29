#include "compat_rule.h"

#include <utility>

namespace Laplace {

namespace {

} // namespace

ValidatedLoadResult load_carried_manifest(const ArtifactIndex& physical,
                                          const PackageView& carrier) {
    auto decoded = SemanticManifest::decode_carried(physical, carrier);
    if (const auto* report = std::get_if<CompatibilityReport>(&decoded)) return *report;
    SemanticManifest manifest = std::get<SemanticManifest>(std::move(decoded));
    // Validate the proof attached to the immutable manifest.
    const auto& proof = manifest.graph_proof();
    if (!proof || !source_compiler_graph_proof_matches(manifest.semantic_model(), *proof)) {
        return package_report(CompatibilityError::IMPORT_CLOSURE_INCOMPLETE,
                              "authoritative manifest graph proof is missing or stale");
    }
    auto runtime = std::shared_ptr<const RuntimePackage>(new RuntimePackage(
        std::move(manifest), Sha256Digest{}, 0,
        RuleQualificationState::Draft, PackageAuthorityKind::DiagnosticRaw));
    DiagnosticProvenance diagnostics;
    diagnostics.format_name = "LAPMAN";
    diagnostics.importer_name = "untrusted-carried-manifest-v1";
    return ValidatedPackage(std::move(runtime), std::move(diagnostics));
}

} // namespace Laplace
