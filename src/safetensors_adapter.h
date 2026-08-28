#pragma once

#include <variant>

#include "artifact_index.h"
#include "safetensors.h"

namespace Laplace {

using SafeTensorsArtifactIndexResult =
    std::variant<ArtifactIndex, CompatibilityReport>;

// Converts one immutable SafeTensors artifact into the common physical index.
// This layer deliberately assigns no tensor roles and emits no model-family
// facts. Semantic configuration and resolution are separate package stages.
SafeTensorsArtifactIndexResult
build_safetensors_artifact_index(const PackageView& artifact);

} // namespace Laplace
