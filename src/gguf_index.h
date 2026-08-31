#pragma once

#include <variant>

#include "artifact_index.h"
#include "gguf_fact_keys.h"

namespace Laplace {

using GgufArtifactIndexResult = std::variant<ArtifactIndex, CompatibilityReport>;

// Converts one immutable GGUF artifact into a physical index. This function
// records checked bytes and structural coordinates only. It does not resolve
// model semantics or create an execution plan.
GgufArtifactIndexResult build_gguf_artifact_index(const PackageView& artifact);

} // namespace Laplace
