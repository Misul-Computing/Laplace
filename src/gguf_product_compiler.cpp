#include "gguf_product_compiler.h"

#include <string>
#include <utility>

#include "gguf.h"
#include "gguf_index.h"

namespace Laplace {

namespace {

CompatibilityReport compiler_error(CompatibilityError code, std::string detail) {
    CompatibilityReport report = package_report(code, std::move(detail));
    report.stage = CompatibilityStage::Import;
    return report;
}

} // namespace

GgufProductCompilationResult compile_gguf_product_source(
    const PackageView& package) {
    GGUFContext context;
    if (!context.parse(package)) {
        return compiler_error(CompatibilityError::PACKAGE_BAD_MAGIC,
                              "raw GGUF source could not be parsed");
    }

    auto physical = build_gguf_artifact_index(package);
    if (const auto* report = std::get_if<CompatibilityReport>(&physical)) {
        return *report;
    }

    if (context.metadata().contains("laplace.semantic_model")) {
        return compiler_error(
            CompatibilityError::AUTHORITY_INVALID,
            "an embedded semantic record is untrusted input and cannot grant closed product authority");
    }

    return compiler_error(
        CompatibilityError::IMPORT_SCHEMA_NOT_FOUND,
        "no closed data-only semantic schema matches this GGUF source");
}

} // namespace Laplace
