#pragma once

#include <string_view>
#include <utility>
#include <variant>

#include "artifact_index.h"

namespace Laplace {

// A closed MLX directory or one SafeTensors physical file. This type makes no
// graph claim: names remain diagnostics in ArtifactIndex and semantics need a
// separate certificate or unique declarative proof.
class MlxPhysicalPackage {
public:
    explicit MlxPhysicalPackage(ArtifactIndex physical_index)
        : physical_index_(std::move(physical_index)) {}

    const ArtifactIndex& physical_index() const noexcept { return physical_index_; }
    CompatibilityReport semantic_refusal() const;

private:
    ArtifactIndex physical_index_;
};

using MlxPhysicalPackageResult = std::variant<MlxPhysicalPackage, CompatibilityReport>;

// For a directory, require config.json, model.safetensors.index.json, and the
// exact canonical shard set named by weight_map. For one regular file, parse a
// role-free SafeTensors physical slice only. Neither path executes package code.
MlxPhysicalPackageResult load_mlx_physical_package(std::string_view path);

} // namespace Laplace
