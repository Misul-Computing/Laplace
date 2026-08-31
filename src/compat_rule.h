#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "semantic_manifest.h"

namespace Laplace {

enum class PackageFormat : uint16_t { Gguf = 1 };
enum class RuleQualificationState : uint16_t { Draft = 1, ExternalGoldenPassed = 2, CanonicalRuntimePassed = 3, Qualified = 4 };
enum class PackageAuthorityKind : uint16_t {
    DiagnosticRaw = 1,
    CarriedManifest = 2,
    ClosedV1Compiled = 3,
};
enum class MetadataPredicateKind : uint16_t { Exists = 1, ExactU64 = 2, ExactString = 3, EnumU64 = 4, ListLength = 5, Digest = 6 };
enum class TensorPatternKind : uint16_t { AnchoredDecimalCapture = 1, ExactName = 2 };
enum class TensorTransformKind : uint16_t { Identity = 1, LogicalTranspose = 2, Reshape = 3, AxisSplit = 4, AxisConcatenate = 5, DeclareDequantize = 6 };

using PackageMetadataValue = std::variant<uint64_t, std::string, std::vector<uint64_t>, Sha256Digest>;

struct MetadataPredicate {
    MetadataPredicateKind kind = MetadataPredicateKind::Exists;
    std::string key;
    uint64_t exact_u64 = 0;
    std::string exact_string;
    std::vector<uint64_t> enum_u64;
    uint32_t list_length = 0;
    Sha256Digest digest{};
};

struct TensorPattern {
    uint32_t template_id = 0;
    TensorPatternKind kind = TensorPatternKind::AnchoredDecimalCapture;
    std::string pattern;
    uint32_t capture_symbol = 0;
    TensorRole role = TensorRole::TokenEmbedding;
    ScalarType logical_type = ScalarType::F32;
    PhysicalLayoutKind layout = PhysicalLayoutKind::ContiguousRowMajor;
    QuantizationKind quantization = QuantizationKind::None;
    std::vector<Dimension> dimensions;
    uint32_t required_plane_mask = 1;
    ScalarType storage_type = ScalarType::F32;
    TensorTransformKind transform = TensorTransformKind::Identity;
    std::vector<uint32_t> alias_template_ids;
    uint32_t repetition_count = 1;
    uint32_t template_stride = 1;
};

struct PackageTensorEvidence {
    std::string name;
    std::vector<uint64_t> dimensions;
    ScalarType storage_type = ScalarType::F32;
    PhysicalLayoutKind layout = PhysicalLayoutKind::ContiguousRowMajor;
    QuantizationKind quantization = QuantizationKind::None;
    uint32_t block_elements = 0;
    uint32_t block_bytes = 0;
    ArtifactId artifact_id{};
    uint64_t offset = 0;
    uint64_t length = 0;
    uint32_t alignment = 0;
};

struct PackageEvidence {
    std::map<std::string, PackageMetadataValue> metadata;
    std::vector<PackageTensorEvidence> tensors;
};

struct CompatibilityRule {
    uint16_t schema_major = 1;
    uint16_t schema_minor = 0;
    uint16_t evaluator_major = 1;
    uint16_t evaluator_minor = 0;
    uint32_t rule_revision = 0;
    std::string rule_id;
    PackageFormat package_format = PackageFormat::Gguf;
    RuleQualificationState qualification_state = RuleQualificationState::Draft;
    std::vector<MetadataPredicate> metadata;
    std::vector<TensorPattern> tensors;
    std::vector<SemanticConstraint> constraints;
    std::vector<CapabilityRequirement> capabilities;
    std::vector<SemanticFallback> fallbacks;
    SemanticModel semantic_template;
    Sha256Digest semantic_template_digest{};
};

class ValidatedPackage;
class ProductPackage;
class VerifiedPhysicalProgramPackage;

// Identity of the closed, data-only source compiler that produced a
// SemanticManifest. It is authority provenance, not a model or performance
// selector. V1 accepts the exact major/minor pair and a nonzero revision.
struct ClosedCompilerIdentity {
    uint16_t major = 0;
    uint16_t minor = 0;
    uint32_t revision = 0;
    Sha256Digest digest{};

    friend bool operator==(const ClosedCompilerIdentity&, const ClosedCompilerIdentity&) = default;
};

class RuntimePackage {
public:
    // Diagnostic packages retain a complete immutable closure, but runtime
    // inference or caller-provided rules cannot grant product authority.
    static std::shared_ptr<const RuntimePackage> make_diagnostic(
        SemanticManifest manifest, Sha256Digest source_fingerprint,
        uint32_t source_revision) {
        return std::shared_ptr<const RuntimePackage>(new RuntimePackage(
            std::move(manifest), source_fingerprint, source_revision,
            RuleQualificationState::Draft, PackageAuthorityKind::DiagnosticRaw));
    }

    const SemanticManifest& manifest() const noexcept { return *manifest_; }
    const SemanticModel& semantics() const noexcept {
        return manifest_ ? manifest_->semantic_model() : *legacy_semantics_;
    }
    // coding-minimalism-rung: expose the manifest's existing immutable witness;
    // downstream consumers do not reconstruct a second graph identity.
    const std::optional<SourceCompilerGraphProof>& graph_proof() const noexcept {
        static const std::optional<SourceCompilerGraphProof> empty;
        return manifest_ ? manifest_->graph_proof() : empty;
    }
    const Sha256Digest& fingerprint() const noexcept { return fingerprint_; }
    const Sha256Digest& artifact_digest() const noexcept {
        return manifest_ ? manifest_->artifact_digest() : legacy_artifact_->digest();
    }
    const Sha256Digest& semantic_graph_digest() const noexcept { return manifest_->semantic_graph_digest(); }
    const Sha256Digest& physical_binding_set_digest() const noexcept {
        return manifest_->physical_binding_set_digest();
    }
    const Sha256Digest& interaction_contract_digest() const noexcept {
        return manifest_->interaction_contract_digest();
    }
    const Sha256Digest& package_fingerprint() const noexcept { return manifest_->package_fingerprint(); }
    const ArtifactIndex& physical_index() const noexcept { return manifest_->physical_index(); }
    const PhysicalCodecRegistry& physical_codec_registry() const noexcept {
        static const PhysicalCodecRegistry empty;
        return manifest_ ? manifest_->physical_codec_registry() : empty;
    }
    const Sha256Digest& physical_codec_registry_digest() const noexcept {
        static const Sha256Digest empty;
        return manifest_ ? manifest_->physical_codec_registry_digest() : empty;
    }
    const std::shared_ptr<const VerifiedPhysicalProgramPackage>&
    physical_program_package() const noexcept {
        return physical_program_package_;
    }
    const Sha256Digest& rule_fingerprint() const noexcept { return rule_fingerprint_; }
    uint32_t rule_revision() const noexcept { return rule_revision_; }
    RuleQualificationState qualification_state() const noexcept { return qualification_state_; }
    PackageAuthorityKind authority_kind() const noexcept { return authority_kind_; }
    bool product_authoritative() const noexcept {
        return authority_kind_ == PackageAuthorityKind::CarriedManifest ||
               authority_kind_ == PackageAuthorityKind::ClosedV1Compiled;
    }
    const ClosedCompilerIdentity& closed_compiler_identity() const noexcept {
        return closed_compiler_identity_;
    }
    std::span<const uint8_t> artifact_bytes(ArtifactId id) const noexcept {
        if (manifest_) {
            for (const PackageView& artifact : manifest_->physical_index().artifacts()) {
                if (artifact.artifact_id() == id) return artifact.bytes();
            }
            return {};
        }
        return id == legacy_artifact_->artifact_id() ? legacy_artifact_->bytes() : std::span<const uint8_t>{};
    }

#ifdef LAPLACE_RUNTIME_PACKAGE_TESTING
    static std::shared_ptr<RuntimePackage> make_closed_v1_test_only(
        SemanticManifest manifest) {
        ClosedCompilerIdentity identity;
        identity.major = 1;
        identity.minor = 0;
        identity.revision = 1;
        identity.digest.bytes[0] = 1;
        return make_closed_v1_test_only(std::move(manifest), identity);
    }

    static std::shared_ptr<RuntimePackage> make_closed_v1_test_only(
        SemanticManifest manifest, ClosedCompilerIdentity identity) {
        return std::shared_ptr<RuntimePackage>(new RuntimePackage(
            std::move(manifest), Sha256Digest{}, 0, RuleQualificationState::Draft,
            PackageAuthorityKind::ClosedV1Compiled, identity));
    }

    static std::shared_ptr<RuntimePackage> make_closed_v1_test_only(
        SemanticManifest manifest, ClosedCompilerIdentity identity,
        std::shared_ptr<const VerifiedPhysicalProgramPackage> physical_package) {
        return std::shared_ptr<RuntimePackage>(new RuntimePackage(
            std::move(manifest), Sha256Digest{}, 0, RuleQualificationState::Draft,
            PackageAuthorityKind::ClosedV1Compiled, identity,
            std::move(physical_package)));
    }

    // Legacy semantic fixtures use pre-manifest schema versions. This symbol
    // is compiled only into test targets so product factories cannot select it.
    static std::shared_ptr<RuntimePackage> make_legacy_test_only(
        SemanticModel semantics, PackageView artifact, Sha256Digest fingerprint,
        Sha256Digest rule_fingerprint, uint32_t rule_revision,
        RuleQualificationState qualification_state) {
        return std::shared_ptr<RuntimePackage>(new RuntimePackage(
            LegacyTestOnlyTag{}, std::move(semantics), std::move(artifact), fingerprint,
            rule_fingerprint, rule_revision, qualification_state));
    }
#endif

private:
    struct LegacyTestOnlyTag {};

    static bool valid_closed_compiler_identity(
        const ClosedCompilerIdentity& identity) noexcept {
        return identity.major == 1 && identity.minor == 0 && identity.revision != 0 &&
               identity.digest != Sha256Digest{};
    }

    RuntimePackage(SemanticManifest manifest, Sha256Digest rule_fingerprint,
                   uint32_t rule_revision, RuleQualificationState qualification_state,
                   PackageAuthorityKind authority_kind,
                   ClosedCompilerIdentity closed_compiler_identity = {},
                   std::shared_ptr<const VerifiedPhysicalProgramPackage> physical_program_package = {})
        : manifest_(std::move(manifest)), fingerprint_(manifest_->package_fingerprint()),
          rule_fingerprint_(rule_fingerprint), rule_revision_(rule_revision),
          qualification_state_(qualification_state),
          authority_kind_(authority_kind == PackageAuthorityKind::CarriedManifest && manifest_->has_carrier()
                              ? PackageAuthorityKind::CarriedManifest
                              : authority_kind == PackageAuthorityKind::ClosedV1Compiled && !manifest_->has_carrier()
                                    ? PackageAuthorityKind::ClosedV1Compiled
                              : PackageAuthorityKind::DiagnosticRaw),
          closed_compiler_identity_(authority_kind_ == PackageAuthorityKind::ClosedV1Compiled
                                        ? closed_compiler_identity
                                        : ClosedCompilerIdentity{}),
          physical_program_package_(std::move(physical_program_package)) {
        if ((authority_kind_ == PackageAuthorityKind::ClosedV1Compiled ||
             authority_kind_ == PackageAuthorityKind::CarriedManifest) &&
            (!manifest_->has_physical_codec_authority() ||
             (authority_kind_ == PackageAuthorityKind::ClosedV1Compiled &&
              !valid_closed_compiler_identity(closed_compiler_identity_)))) {
            authority_kind_ = PackageAuthorityKind::DiagnosticRaw;
            closed_compiler_identity_ = {};
        }
    }

    RuntimePackage(LegacyTestOnlyTag, SemanticModel semantics, PackageView artifact,
                   Sha256Digest fingerprint, Sha256Digest rule_fingerprint,
                   uint32_t rule_revision, RuleQualificationState qualification_state)
        : legacy_semantics_(std::move(semantics)), legacy_artifact_(std::move(artifact)),
          fingerprint_(fingerprint), rule_fingerprint_(rule_fingerprint),
          rule_revision_(rule_revision), qualification_state_(qualification_state),
          authority_kind_(PackageAuthorityKind::DiagnosticRaw) {}

    friend std::variant<ValidatedPackage, CompatibilityReport>
    load_validated_gguf(const PackageView& package);
    friend std::variant<ValidatedPackage, CompatibilityReport>
    load_carried_manifest(const ArtifactIndex& physical, const PackageView& carrier);
    friend class ProductPackage;
    friend std::variant<ValidatedPackage, CompatibilityReport>
    load_expected_fixture_gguf(const PackageView& package,
                               const std::vector<CompatibilityRule>& rules);
    std::optional<SemanticManifest> manifest_;
    std::optional<SemanticModel> legacy_semantics_;
    std::optional<PackageView> legacy_artifact_;
    Sha256Digest fingerprint_;
    Sha256Digest rule_fingerprint_;
    uint32_t rule_revision_ = 0;
    RuleQualificationState qualification_state_ = RuleQualificationState::Draft;
    PackageAuthorityKind authority_kind_ = PackageAuthorityKind::DiagnosticRaw;
    ClosedCompilerIdentity closed_compiler_identity_;
    std::shared_ptr<const VerifiedPhysicalProgramPackage> physical_program_package_;
};

struct DiagnosticProvenance {
    std::string format_name;
    std::string importer_name;
    std::string family_text;
    std::string rule_id;
    uint32_t rule_revision = 0;
};

class ValidatedPackage {
public:
    std::shared_ptr<const RuntimePackage> runtime_package() const { return runtime_; }
    const DiagnosticProvenance& diagnostics() const { return diagnostics_; }

private:
    ValidatedPackage(std::shared_ptr<const RuntimePackage> runtime, DiagnosticProvenance diagnostics)
        : runtime_(std::move(runtime)), diagnostics_(std::move(diagnostics)) {}

    friend std::variant<ValidatedPackage, CompatibilityReport>
    load_validated_gguf(const PackageView& package);
    friend std::variant<ValidatedPackage, CompatibilityReport>
    load_carried_manifest(const ArtifactIndex& physical, const PackageView& carrier);
    friend std::variant<ValidatedPackage, CompatibilityReport>
    load_expected_fixture_gguf(const PackageView& package,
                               const std::vector<CompatibilityRule>& rules);

    std::shared_ptr<const RuntimePackage> runtime_;
    DiagnosticProvenance diagnostics_;
};

using RuleEncodeResult = std::variant<std::vector<uint8_t>, CompatibilityReport>;
using RuleDecodeResult = std::variant<CompatibilityRule, CompatibilityReport>;
using RuleEvaluationResult = std::variant<SemanticModel, CompatibilityReport>;
using ValidatedLoadResult = std::variant<ValidatedPackage, CompatibilityReport>;

RuleEncodeResult encode_compatibility_rule(const CompatibilityRule& rule);
RuleDecodeResult decode_compatibility_rule(const std::vector<uint8_t>& bytes);
Sha256Digest rule_fingerprint(const CompatibilityRule& rule);
RuleEvaluationResult evaluate_rules(const std::vector<CompatibilityRule>& rules,
                                    const PackageEvidence& package);
// Resolves known semantic compositions from normalized GGUF evidence. It does
// not inspect package provenance or model-family metadata.
RuleEvaluationResult resolve_gguf_semantics(const PackageEvidence& package);
RuleEvaluationResult import_gguf(const PackageView& package);
const std::vector<CompatibilityRule>& bundled_compatibility_rules();
std::span<const uint8_t> bundled_compatibility_rule_record();
ValidatedLoadResult load_validated_gguf(const PackageView& package);
// Structurally validates an immutable carried manifest. Product authority is
// granted only by the loader after the complete carrier closure is validated.
ValidatedLoadResult load_carried_manifest(const ArtifactIndex& physical,
                                           const PackageView& carrier);
RuleEvaluationResult import_expected_fixture_gguf(const PackageView& package,
                                                  const std::vector<CompatibilityRule>& rules);
ValidatedLoadResult load_expected_fixture_gguf(const PackageView& package,
                                               const std::vector<CompatibilityRule>& rules);

} // namespace Laplace
