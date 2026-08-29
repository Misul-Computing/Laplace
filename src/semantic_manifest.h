#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "artifact_index.h"
#include "source_compiler_graph_proof.h"
#include "token_contract.h"

namespace Laplace {

class RuntimePackage;

// Covers the complete token authority, including the typed package reference
// and bounded prompt operations.
Sha256Digest semantic_manifest_token_contract_digest(const TokenContract& contract);
using SemanticGraphDigestResult = std::variant<Sha256Digest, CompatibilityReport>;
SemanticGraphDigestResult semantic_manifest_graph_digest(const SemanticModel& model);

class SemanticManifest;
using SemanticManifestResult = std::variant<SemanticManifest, CompatibilityReport>;

// A validated immutable envelope.  Its copy retains the ArtifactIndex and its
// reference-counted package owners, so spans returned by physical_index() stay
// valid for the lifetime of this object.
class SemanticManifest {
public:
    struct Data;

    static SemanticManifestResult build(const ArtifactIndex& physical,
                                         const SemanticModel& model,
                                         const TokenContract& token_contract);
    static SemanticManifestResult compile(const ArtifactIndex& physical,
                                          const SemanticModel& model,
                                          const TokenContract& token_contract) {
        return build(physical, model, token_contract);
    }
    static SemanticManifestResult decode(const ArtifactIndex& physical,
                                         std::span<const uint8_t> bytes);
    // Decode a manifest from an immutable sidecar member. The returned
    // record retains the sidecar owner, but remains data-only until the
    // carried-manifest package factory accepts it.
    static SemanticManifestResult decode_carried(const ArtifactIndex& physical,
                                                 const PackageView& carrier);

    std::span<const uint8_t> bytes() const noexcept;
    // The canonical LAPIR payload carried by the envelope. Its physical
    // fields are validated by physical_binding_set_digest(), not used as the
    // semantic graph identity.
    std::span<const uint8_t> semantic_bytes() const noexcept;
    const SemanticModel& semantic_model() const noexcept;
    const TokenContract& token_contract() const noexcept;
    const ArtifactIndex& physical_index() const noexcept;
    ArtifactId artifact_id() const noexcept;
    uint64_t artifact_size() const noexcept;
    const Sha256Digest& artifact_digest() const noexcept;
    const Sha256Digest& semantic_graph_digest() const noexcept;
    const Sha256Digest& physical_binding_set_digest() const noexcept;
    const Sha256Digest& interaction_contract_digest() const noexcept;
    const Sha256Digest& package_fingerprint() const noexcept;
    const Sha256Digest& graph_digest() const noexcept { return semantic_graph_digest(); }
    const Sha256Digest& physical_binding_digest() const noexcept { return physical_binding_set_digest(); }
    const Sha256Digest& interaction_digest() const noexcept { return interaction_contract_digest(); }
    // Compatibility spelling for callers that used the pre-split accessor.
    const Sha256Digest& semantic_digest() const noexcept;
    const Sha256Digest& body_digest() const noexcept;
    const Sha256Digest& record_digest() const noexcept;
    const std::optional<SourceCompilerGraphProof>& graph_proof() const noexcept;
    bool has_carrier() const noexcept;

private:
    explicit SemanticManifest(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
    std::shared_ptr<const Data> data_;

    friend class RuntimePackage;
};

SemanticManifestResult compile_semantic_manifest(const ArtifactIndex& physical,
                                                 const SemanticModel& model,
                                                 const TokenContract& token_contract);
SemanticManifestResult decode_semantic_manifest(const ArtifactIndex& physical,
                                                std::span<const uint8_t> bytes);
SemanticManifestResult decode_carried_semantic_manifest(const ArtifactIndex& physical,
                                                        const PackageView& carrier);

} // namespace Laplace
