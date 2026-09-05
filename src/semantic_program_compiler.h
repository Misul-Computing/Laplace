#pragma once

#include "program_ir.h"
#include "semantic_model.h"

namespace Laplace {

struct SemanticDimensionBinding {
    uint64_t symbol;
    uint64_t extent;
};

struct SemanticProgramBinding {
    uint32_t semantic_id;
    uint32_t entry_value_id;
};

struct CompiledSemanticProgram {
    VerifiedProgram program;
    std::vector<SemanticProgramBinding> tensors;
    std::vector<SemanticProgramBinding> inputs;
    std::vector<uint32_t> outputs;
};

using SemanticProgramCompileResult =
    std::variant<CompiledSemanticProgram, CompatibilityReport>;

// Operation lowering. Physical storage is supplied independently
// through the returned tensor entry arguments. Symbol dimensions must be bound.
// Stateful graphs execute one token per call and own a persistent cursor.
// Floating physical tensors decode into F32 compute values; storage stays external.
SemanticProgramCompileResult compile_semantic_program(
    const SemanticModel& model,
    std::span<const SemanticDimensionBinding> dimensions = {},
    uint32_t cache_capacity = 0);

} // namespace Laplace
