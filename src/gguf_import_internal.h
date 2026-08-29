#pragma once

#include "artifact_index.h"
#include "semantic_model.h"

namespace Laplace::detail {

bool bind_gguf_semantics_to_physical_index(const ArtifactIndex& physical,
                                           SemanticModel& model);

} // namespace Laplace::detail
