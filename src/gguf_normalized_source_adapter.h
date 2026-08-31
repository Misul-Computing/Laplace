#pragma once

#include <span>
#include <string>
#include <variant>
#include <vector>

#include "gguf_index.h"
#include "normalized_source_evidence.h"

namespace Laplace {

struct GgufNormalizedSourceTensor {
    uint32_t source_tensor_id = UINT32_MAX;
    std::string spelling;
    ArtifactTensorRecord physical;
};

using GgufNormalizedSourceEvidenceResult = NormalizedSourceEvidenceResult;

GgufNormalizedSourceEvidenceResult
extract_gguf_normalized_source_evidence(
    std::span<const GgufNormalizedSourceTensor> tensors);

GgufNormalizedSourceEvidenceResult
extract_gguf_normalized_source_evidence(const ArtifactIndex& physical);

} // namespace Laplace
