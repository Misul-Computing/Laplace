#pragma once

#include <optional>
#include <utility>
#include <variant>

#include "artifact_index.h"
#include "safetensors.h"

namespace Laplace {

using SafeTensorsArtifactIndexResult =
    std::variant<ArtifactIndex, CompatibilityReport>;

// Maps a SafeTensors scalar storage declaration to the common artifact ABI.
// This is container syntax only: it does not assign numerical semantics to a
// group of planes or select an execution implementation.
std::optional<std::pair<ArtifactScalarType, uint32_t>>
safetensors_physical_scalar(SafeTensorsDtype dtype) noexcept;

// Converts one immutable SafeTensors artifact into the common physical index.
// This layer deliberately assigns no tensor roles and emits no model-family
// facts. Semantic configuration and resolution are separate package stages.
SafeTensorsArtifactIndexResult
build_safetensors_artifact_index(const PackageView& artifact);

} // namespace Laplace
