#pragma once

#include <variant>

#include "semantic_manifest.h"
#include "token_program.h"

namespace Laplace {

struct GgufProductCompilation {
    SemanticManifest manifest;
    TokenProgram token_program;
};

using GgufProductCompilationResult =
    std::variant<GgufProductCompilation, CompatibilityReport>;

// Compiles only a complete graph emitted by the closed data-only source-schema
// evaluator. Embedded semantic bytes are diagnostic input and never authority.
GgufProductCompilationResult compile_gguf_product_source(const PackageView& package);

} // namespace Laplace
