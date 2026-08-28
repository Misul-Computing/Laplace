#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "semantic_model.h"

namespace Laplace {

enum class PackageFormat : uint16_t { Gguf = 1 };
enum class RuleQualificationState : uint16_t { Draft = 1, ExternalGoldenPassed = 2, CanonicalRuntimePassed = 3, Qualified = 4 };
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

class RuntimePackage {
public:
    RuntimePackage(SemanticModel semantics, PackageView artifact, Sha256Digest fingerprint,
                   Sha256Digest rule_fingerprint, uint32_t rule_revision,
                   RuleQualificationState qualification_state)
        : semantics_(std::move(semantics)), artifact_(std::move(artifact)), fingerprint_(fingerprint),
          rule_fingerprint_(rule_fingerprint), rule_revision_(rule_revision), qualification_state_(qualification_state) {}

    const SemanticModel& semantics() const noexcept { return semantics_; }
    const Sha256Digest& fingerprint() const noexcept { return fingerprint_; }
    const Sha256Digest& artifact_digest() const noexcept { return artifact_.digest(); }
    const Sha256Digest& rule_fingerprint() const noexcept { return rule_fingerprint_; }
    uint32_t rule_revision() const noexcept { return rule_revision_; }
    RuleQualificationState qualification_state() const noexcept { return qualification_state_; }
    std::span<const uint8_t> artifact_bytes(ArtifactId id) const noexcept {
        return id == artifact_.artifact_id() ? artifact_.bytes() : std::span<const uint8_t>{};
    }

private:
    SemanticModel semantics_;
    PackageView artifact_;
    Sha256Digest fingerprint_;
    Sha256Digest rule_fingerprint_;
    uint32_t rule_revision_ = 0;
    RuleQualificationState qualification_state_ = RuleQualificationState::Draft;
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
    ValidatedPackage(std::shared_ptr<const RuntimePackage> runtime, DiagnosticProvenance diagnostics)
        : runtime_(std::move(runtime)), diagnostics_(std::move(diagnostics)) {}

    std::shared_ptr<const RuntimePackage> runtime_package() const { return runtime_; }
    const DiagnosticProvenance& diagnostics() const { return diagnostics_; }

private:
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
ValidatedLoadResult load_validated_gguf(const PackageView& package);
// Explicit rule evaluation is retained for evaluator tests and external
// callers that provide their own evidence. The public CLI never loads a
// bundled model-specific rule.
RuleEvaluationResult import_expected_fixture_gguf(const PackageView& package,
                                                  const std::vector<CompatibilityRule>& rules);

} // namespace Laplace
