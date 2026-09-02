#pragma once

#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "artifact_index.h"

namespace Laplace {

struct SafeTensorsStorageResource {
    uint32_t id = UINT32_MAX;
    std::vector<uint32_t> tensor_ids;
    friend bool operator==(const SafeTensorsStorageResource&,
                           const SafeTensorsStorageResource&) = default;
};

// Import-time names are discarded after resolving the declared groups to the
// immutable physical index. Runtime behavior is identified only by these
// canonical IDs and digests; arithmetic remains a separate physical program.
class VerifiedSafeTensorsStorageResources {
public:
    std::span<const SafeTensorsStorageResource> resources() const noexcept {
        return resources_;
    }
    uint32_t declared_resource_count() const noexcept {
        return declared_resource_count_;
    }
    const Sha256Digest& canonical_digest() const noexcept {
        return canonical_digest_;
    }
    const Sha256Digest& declaration_digest() const noexcept {
        return declaration_digest_;
    }

private:
    friend std::variant<VerifiedSafeTensorsStorageResources, CompatibilityReport>
    compile_safetensors_storage_resources(const ArtifactIndex&, const PackageView&);

    std::vector<SafeTensorsStorageResource> resources_;
    uint32_t declared_resource_count_ = 0;
    Sha256Digest canonical_digest_{};
    Sha256Digest declaration_digest_{};
};

using SafeTensorsStorageResourcesResult =
    std::variant<VerifiedSafeTensorsStorageResources, CompatibilityReport>;

SafeTensorsStorageResourcesResult compile_safetensors_storage_resources(
    const ArtifactIndex& physical, const PackageView& declaration);

// A closed sharded directory or one SafeTensors physical file. This type makes
// no graph claim: names remain diagnostics in ArtifactIndex and semantics need
// a separate certificate or unique declarative proof.
class MlxPhysicalPackage {
public:
    explicit MlxPhysicalPackage(
        ArtifactIndex physical_index,
        std::optional<VerifiedSafeTensorsStorageResources> storage = std::nullopt)
        : physical_index_(std::move(physical_index)),
          storage_(std::move(storage)) {}

    const ArtifactIndex& physical_index() const noexcept { return physical_index_; }
    const VerifiedSafeTensorsStorageResources* storage_resources() const noexcept {
        return storage_ ? &*storage_ : nullptr;
    }
    CompatibilityReport semantic_refusal() const;

private:
    ArtifactIndex physical_index_;
    std::optional<VerifiedSafeTensorsStorageResources> storage_;
};

using MlxPhysicalPackageResult = std::variant<MlxPhysicalPackage, CompatibilityReport>;

// The product closure keeps every declared MLX member alive while exposing a
// runtime index that contains the configuration, shard index, weight shards,
// and the carried token program. The manifest remains a separate sidecar so
// SemanticManifest::decode_carried can enforce its authority boundary.
inline constexpr ArtifactId kMlxProductManifestArtifactId{UINT32_MAX - 1};
inline constexpr ArtifactId kMlxProductTokenArtifactId{UINT32_MAX - 2};
inline constexpr ArtifactId kSafeTensorsProductPhysicalPackageArtifactId{UINT32_MAX - 3};

struct MlxProductPhysicalPackage {
    ArtifactSet closure;
    ArtifactIndex physical_index;
    PackageView manifest;
    std::optional<PackageView> physical_package;
};

using MlxProductPhysicalPackageResult =
    std::variant<MlxProductPhysicalPackage, CompatibilityReport>;

// For a directory, require config.json, model.safetensors.index.json, and the
// exact canonical shard set named by weight_map. For one regular file, parse a
// role-free SafeTensors physical slice only. Neither path executes package code.
MlxPhysicalPackageResult load_safetensors_physical_package(std::string_view path);

// Compatibility spelling. It adds no format behavior.
MlxPhysicalPackageResult load_mlx_physical_package(std::string_view path);

// Loads the declared MLX directory closure used by the product route. The
// directory must contain config.json, model.safetensors.index.json, the exact
// indexed shard set, laplace.lapman, and laplace.laptok. The returned physical
// index contains exact F32/F16 tensors only and does not infer semantics.
MlxProductPhysicalPackageResult
load_safetensors_product_physical_package(std::string_view path);

MlxProductPhysicalPackageResult
load_mlx_product_physical_package(std::string_view path);

} // namespace Laplace
