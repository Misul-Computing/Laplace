#pragma once

#include <cstdint>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include "physical_program_package.h"
#include "physical_program_interpreter.h"

namespace Laplace {

struct ReferenceValue {
    ElementType type = ElementType::F32;
    std::vector<uint64_t> extents;
    std::vector<uint64_t> bits;
    friend bool operator==(const ReferenceValue&, const ReferenceValue&) = default;
};

struct ReferenceInput {
    uint32_t value_id = UINT32_MAX;
    ReferenceValue value;
};

using ProgramInputs = std::vector<ReferenceInput>;

struct ReferenceState {
    std::vector<std::pair<uint32_t, ReferenceValue>> slots;
    uint64_t generation = 0;
};

struct ReferenceResult {
    std::vector<ReferenceValue> exports;
    uint64_t generation = 0;
};

using ReferenceExecutionResult =
    std::variant<ReferenceResult, CompatibilityReport>;

// Qualification-only decode bridge. It binds an artifact-owned resource to
// the verified physical byte program and interprets one logical coordinate.
// The bridge is intentionally slow and copies into the interpreter's owned
// snapshot; product Metal execution must provide its own zero-copy binding.
using ReferencePhysicalResult =
    std::variant<ScalarValue, CompatibilityReport>;

ReferencePhysicalResult decode_reference_resource(
    const VerifiedPhysicalProgramPackage& package, uint32_t resource_id,
    std::span<const uint64_t> logical_coordinate);

ReferenceExecutionResult execute_reference_program(
    const VerifiedProgram& program,
    ReferenceState& state,
    std::span<const ReferenceInput> inputs);

ReferenceExecutionResult execute_reference(
    const VerifiedPhysicalProgramPackage& package,
    ReferenceState& state,
    std::span<const ReferenceInput> inputs);

} // namespace Laplace
