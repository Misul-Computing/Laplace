#pragma once

#include "program_package.h"
#include "semantic_manifest.h"

namespace Laplace {

struct LoadedSourceProgram {
    VerifiedProgramPackage package;
    uint32_t max_context = 0;
    uint32_t eos_id = UINT32_MAX;
    std::vector<uint32_t> stop_ids;
};

using LoadedSourceProgramResult =
    std::variant<LoadedSourceProgram, CompatibilityReport>;

// Compile one-token execution from validated source semantics and immutable
// codec/token programs. max_context bounds the generated persistent state.
ProgramPackageResult compile_source_program_package(
    const SemanticManifest& manifest, const TokenProgram& tokens,
    uint32_t max_context);

LoadedSourceProgramResult load_source_program_package(
    std::string_view source_path, uint32_t max_context = 0);

} // namespace Laplace
