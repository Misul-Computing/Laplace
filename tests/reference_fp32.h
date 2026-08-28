#pragma once

#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include "compat_rule.h"

namespace Laplace {

struct ReferenceOutput {
    std::vector<float> logits;
    std::vector<std::vector<float>> operator_outputs;
    std::vector<SemanticKvState> states;
    SemanticKvState key_state;
};

using ReferenceResult = std::variant<ReferenceOutput, CompatibilityReport>;

ReferenceResult reference_fp32(const RuntimePackage& package, std::span<const uint32_t> token_ids);

} // namespace Laplace
