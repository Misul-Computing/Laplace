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

// The product closure keeps every declared MLX member alive while exposing a
// runtime index that contains the configuration, shard index, weight shards,
// and the carried token program. The manifest remains a separate sidecar so
// SemanticManifest::decode_carried can enforce its authority boundary.
inline constexpr ArtifactId kMlxProductManifestArtifactId{UINT32_MAX - 1};
inline constexpr ArtifactId kMlxProductTokenArtifactId{UINT32_MAX - 2};

struct MlxProductPhysicalPackage {
    ArtifactSet closure;
    ArtifactIndex physical_index;
    PackageView manifest;
};

using MlxProductPhysicalPackageResult =
    std::variant<MlxProductPhysicalPackage, CompatibilityReport>;

// For a directory, require config.json, model.safetensors.index.json, and the
// exact canonical shard set named by weight_map. For one regular file, parse a
// role-free SafeTensors physical slice only. Neither path executes package code.
MlxPhysicalPackageResult load_mlx_physical_package(std::string_view path);

// Loads the declared MLX directory closure used by the product route. The
// directory must contain config.json, model.safetensors.index.json, the exact
// indexed shard set, laplace.lapman, and laplace.laptok. The returned physical
// index contains exact F32/F16 tensors only and does not infer semantics.
MlxProductPhysicalPackageResult
load_mlx_product_physical_package(std::string_view path);

} // namespace Laplace
