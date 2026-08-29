#pragma once

#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include "compat_rule.h"
#include "execution_plan.h"

namespace Laplace {

struct ScalarExecutionOutput {
    std::vector<float> logits;
    std::vector<std::vector<float>> operator_outputs;
    std::vector<SemanticKvState> states;
    uint64_t operator_token_work = 0;
};

using ScalarExecutionResult = std::variant<ScalarExecutionOutput, CompatibilityReport>;

ScalarExecutionResult scalar_execute(const RuntimePackage& package, std::span<const uint32_t> token_ids);
ScalarExecutionResult execute_planned_scalar(const RuntimePackage& package, const ExecutionPlan& plan,
                                             ExecutionPhase phase, std::span<const uint32_t> token_ids);
ScalarExecutionResult execute_planned_scalar_incremental(const RuntimePackage& package, const ExecutionPlan& plan,
                                                         ExecutionPhase phase, std::span<const uint32_t> token_ids,
                                                         std::vector<SemanticKvState>& states);

} // namespace Laplace
