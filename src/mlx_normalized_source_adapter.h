#pragma once

#include <span>
#include <string>
#include <variant>
#include <vector>

#include "mlx_package.h"
#include "normalized_source_evidence.h"

namespace Laplace {

struct MlxNormalizedSourceTensor {
    uint32_t source_tensor_id = UINT32_MAX;
    std::string spelling;
    ArtifactTensorRecord physical;
};

using MlxNormalizedSourceEvidenceResult = NormalizedSourceEvidenceResult;

MlxNormalizedSourceEvidenceResult
extract_mlx_normalized_source_evidence(
    std::span<const MlxNormalizedSourceTensor> tensors);

MlxNormalizedSourceEvidenceResult
extract_mlx_normalized_source_evidence(const ArtifactIndex& physical);

} // namespace Laplace
