#include "codec_binding.h"

#include "codec_certificate.h"
#include "compat_rule.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <new>
#include <optional>
#include <utility>

namespace Laplace {

// The constructors are private so the only publication path is a successful
// preflight. The builder is deliberately tiny; manifest/index validation
// remains the single authority for the package contract.
struct CodecBindingBuilder {
    static ResolvedCodecPlane make_plane(
        PlaneKind kind, ArtifactScalarType storage_type, ScalarType semantic_storage_type,
        ArtifactId artifact_id,
        uint64_t offset, uint64_t length, uint32_t alignment, uint32_t flags,
        uint64_t logical_elements, uint32_t bytes_per_block,
        uint32_t elements_per_block, Sha256Digest artifact_digest,
        uint64_t artifact_size) {
        return ResolvedCodecPlane(kind, storage_type, semantic_storage_type, artifact_id,
                                  offset, length,
                                  alignment, flags, logical_elements, bytes_per_block,
                                  elements_per_block, artifact_digest, artifact_size);
    }

    static ResolvedCodecTensor make_tensor(
        uint32_t occurrence_index, uint32_t tensor_slot, uint32_t tensor_id,
        PhysicalCodecIdentity physical_identity,
        CodecProgramIdentity program_identity, std::vector<Dimension> dimensions,
        std::array<uint64_t, 8> strides, std::vector<ResolvedCodecPlane> planes) {
        return ResolvedCodecTensor(occurrence_index, tensor_slot, tensor_id,
                                   std::move(physical_identity),
                                   program_identity, std::move(dimensions), strides,
                                   std::move(planes));
    }

    static ResolvedCodecOperator make_operator(
        uint32_t operator_id, std::vector<ResolvedCodecTensor> tensors) {
        return ResolvedCodecOperator(operator_id, std::move(tensors));
    }

    static ResolvedCodecBindings make_bindings(
        Sha256Digest package_fingerprint, std::vector<ResolvedCodecOperator> operators) {
        return ResolvedCodecBindings(package_fingerprint, std::move(operators));
    }

    static CodecBindingPreflightResult run(const RuntimePackage& package);
};

namespace {

CompatibilityReport failure(CompatibilityError code, std::string detail,
                             uint32_t operator_id = UINT32_MAX,
                             uint32_t tensor_id = UINT32_MAX) {
    CompatibilityReport report = compatibility_report(code, std::move(detail));
    report.operator_id = operator_id;
    report.tensor_id = tensor_id;
    return report;
}

const ArtifactTensorRecord* physical_tensor_by_id(const ArtifactIndex& index, uint32_t id) {
    const auto tensors = index.tensors();
    const auto found = std::lower_bound(
        tensors.begin(), tensors.end(), id,
        [](const ArtifactTensorRecord& tensor, uint32_t wanted) {
            return tensor.id < wanted;
        });
    return found == tensors.end() || found->id != id ? nullptr : &*found;
}

const PhysicalTensorCodecDeclaration* codec_tensor_by_id(
    const PhysicalCodecRegistry& registry, uint32_t id) {
    const auto found = std::lower_bound(
        registry.tensors.begin(), registry.tensors.end(), id,
        [](const PhysicalTensorCodecDeclaration& declaration, uint32_t wanted) {
            return declaration.tensor_id < wanted;
        });
    return found == registry.tensors.end() || found->tensor_id != id ? nullptr : &*found;
}

const PhysicalCodecSpec* codec_spec_by_identity(
    const PhysicalCodecRegistry& registry, const PhysicalCodecIdentity& identity) {
    const auto found = std::lower_bound(
        registry.codecs.begin(), registry.codecs.end(), identity,
        [](const PhysicalCodecSpec& spec, const PhysicalCodecIdentity& wanted) {
            return physical_codec_identity_less(spec.identity, wanted);
        });
    return found == registry.codecs.end() || found->identity != identity ? nullptr : &*found;
}

const TensorPlane* plane_by_kind(const SemanticTensor& tensor, PlaneKind kind) {
    const auto found = std::find_if(tensor.planes.begin(), tensor.planes.end(),
                                    [kind](const TensorPlane& plane) {
                                        return plane.kind == kind;
                                    });
    return found == tensor.planes.end() ? nullptr : &*found;
}

const PackageView* artifact_by_id(const ArtifactIndex& index, ArtifactId id) {
    const auto artifacts = index.artifacts();
    const auto found = std::find_if(artifacts.begin(), artifacts.end(),
                                    [id](const PackageView& artifact) {
                                        return artifact.artifact_id() == id;
                                    });
    return found == artifacts.end() ? nullptr : &*found;
}

bool checked_add(uint64_t left, uint64_t right, uint64_t& result) {
    if (right > std::numeric_limits<uint64_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool checked_multiply(uint64_t left, uint64_t right, uint64_t& result) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) return false;
    result = left * right;
    return true;
}

ArtifactScalarType artifact_scalar_type(ScalarType type) {
    switch (type) {
    case ScalarType::F32: return ArtifactScalarType::F32;
    case ScalarType::F16: return ArtifactScalarType::F16;
    case ScalarType::U8: return ArtifactScalarType::U8;
    case ScalarType::U32: return ArtifactScalarType::U32;
    case ScalarType::I32: return ArtifactScalarType::I32;
    }
    return ArtifactScalarType::None;
}

ScalarType semantic_scalar_type(ArtifactScalarType type) {
    switch (type) {
    case ArtifactScalarType::F32: return ScalarType::F32;
    case ArtifactScalarType::F16: return ScalarType::F16;
    case ArtifactScalarType::U8:
    case ArtifactScalarType::Packed: return ScalarType::U8;
    case ArtifactScalarType::U32: return ScalarType::U32;
    case ArtifactScalarType::I32: return ScalarType::I32;
    default: return static_cast<ScalarType>(0);
    }
}

// ArtifactTensorPlane.storage_type is the physical storage type. A packed
// values plane is represented as semantic U8 in PhysicalCodecIdentity, while
// auxiliary planes retain their typed scalar storage.
ArtifactScalarType expected_artifact_storage(const ArtifactTensorRecord& physical,
                                             const PhysicalPlaneSchema& schema) {
    if (schema.kind == PlaneKind::Values) return physical.format.value_type;
    return artifact_scalar_type(schema.storage_type);
}

std::optional<CompatibilityReport> validate_authoritative_planes(
    const RuntimePackage& package, const SemanticTensor& semantic,
    const ArtifactTensorRecord& physical, const PhysicalCodecIdentity& identity,
    uint32_t operator_id, std::vector<ResolvedCodecPlane>& copied) {
    if (physical.planes.size() != identity.planes.size() ||
        semantic.planes.size() != identity.planes.size()) {
        return failure(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                       "authoritative physical plane set is incomplete", operator_id, semantic.id);
    }
    uint64_t semantic_elements = 1;
    for (const Dimension& dimension : semantic.dimensions) {
        if (dimension.kind != DimensionKind::Constant ||
            dimension.constant_or_symbol == 0 ||
            !checked_multiply(semantic_elements,
                              dimension.constant_or_symbol,
                              semantic_elements)) {
            return failure(CompatibilityError::IR_SHAPE_MISMATCH,
                           "semantic tensor element count is not finite",
                           operator_id, semantic.id);
        }
    }
    copied.reserve(physical.planes.size());
    for (size_t index = 0; index < identity.planes.size(); ++index) {
        const PhysicalPlaneSchema& schema = identity.planes[index];
        const ArtifactTensorPlane& plane = physical.planes[index];
        const TensorPlane* semantic_plane = plane_by_kind(semantic, schema.kind);
        const PackageView* artifact = artifact_by_id(package.physical_index(),
                                                      plane.source.artifact_id);
        uint64_t end = 0, expected_length = 0;
        if (!semantic_plane || plane.kind != schema.kind ||
            plane.storage_type != expected_artifact_storage(physical, schema) ||
            plane.logical_elements == 0 || plane.bytes_per_block == 0 ||
            plane.elements_per_block == 0 ||
            schema.logical_elements_covered == 0 ||
            semantic_elements % schema.logical_elements_covered != 0 ||
            !checked_multiply(
                semantic_elements / schema.logical_elements_covered,
                schema.bytes_per_block, expected_length) ||
            plane.source.length != expected_length ||
            semantic_plane->flags != schema.flags || !artifact ||
            !checked_add(plane.source.offset, plane.source.length, end) ||
            end > artifact->bytes().size()) {
            return failure(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                           "authoritative plane disagrees with the selected codec identity",
                           operator_id, semantic.id);
        }
        copied.push_back(CodecBindingBuilder::make_plane(
            plane.kind, plane.storage_type, semantic_scalar_type(plane.storage_type),
            plane.source.artifact_id,
            plane.source.offset, plane.source.length, plane.alignment, semantic_plane->flags,
            plane.logical_elements, plane.bytes_per_block, plane.elements_per_block,
            artifact->digest(), artifact->bytes().size()));
    }
    return std::nullopt;
}

std::optional<PlaneKind> plane_kind(CodecCertificatePlaneRole role) {
    switch (role) {
    case CodecCertificatePlaneRole::Values: return PlaneKind::Values;
    case CodecCertificatePlaneRole::Scales: return PlaneKind::Scales;
    case CodecCertificatePlaneRole::Biases: return PlaneKind::Biases;
    case CodecCertificatePlaneRole::Indexes: return PlaneKind::Indexes;
    }
    return std::nullopt;
}

std::optional<CompatibilityReport> validate_operator(
    const RuntimePackage& package, const SemanticOperator& operation,
    const PhysicalCodecRegistry& registry, uint32_t& next_occurrence_index,
    std::vector<ResolvedCodecTensor>& output) {
    if (!semantic_operator_signature_valid(operation) ||
        !semantic_operator_contract_valid(operation)) {
        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                       "semantic operator contract is malformed", operation.id);
    }
    std::vector<ResolvedCodecTensor> occurrences;
    occurrences.reserve(operation.tensors.size());
    for (size_t tensor_slot = 0; tensor_slot < operation.tensors.size(); ++tensor_slot) {
        const uint32_t tensor_index = operation.tensors[tensor_slot];
        const SemanticModel& model = package.semantics();
        const SemanticTensor* semantic = tensor_index < model.tensors.size()
            ? &model.tensors[tensor_index] : nullptr;
        if (!semantic) {
            return failure(CompatibilityError::IR_REFERENCE_INVALID,
                           "operator tensor reference is outside the authoritative model",
                           operation.id, tensor_index);
        }
        const PhysicalTensorCodecDeclaration* codec = codec_tensor_by_id(registry, semantic->id);
        const PhysicalCodecSpec* spec = codec ? codec_spec_by_identity(registry, codec->identity)
                                              : nullptr;
        const ArtifactTensorRecord* physical =
            physical_tensor_by_id(package.physical_index(), semantic->id);
        if (!codec || !spec || !physical) {
            return failure(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                           "operator tensor lacks an authoritative codec certificate",
                           operation.id, semantic->id);
        }
        const CodecCertificateParseResult parsed =
            parse_codec_certificate(spec->certificate_bytes);
        const CodecCertificate* certificate = std::get_if<CodecCertificate>(&parsed);
        if (!certificate || !certificate->matches_physical_identity(codec->identity)) {
            return failure(CompatibilityError::IMPORT_MANIFEST_INVALID,
                           "codec certificate does not match the complete physical identity",
                           operation.id, semantic->id);
        }
        std::vector<ResolvedCodecPlane> copied_planes;
        if (auto error = validate_authoritative_planes(
                package, *semantic, *physical, codec->identity, operation.id, copied_planes)) {
            return std::move(*error);
        }
        uint64_t element_count = 1;
        for (const Dimension& dimension : semantic->dimensions) {
            if (dimension.kind != DimensionKind::Constant || dimension.constant_or_symbol == 0 ||
                !checked_multiply(element_count, dimension.constant_or_symbol, element_count)) {
                return failure(CompatibilityError::IR_SHAPE_MISMATCH,
                               "semantic tensor element count is not finite", operation.id,
                               semantic->id);
            }
        }
        const uint32_t unit_elements = certificate->summary().unit_elements;
        if (unit_elements == 0 || element_count % unit_elements != 0) {
            return failure(CompatibilityError::IR_SHAPE_MISMATCH,
                           "semantic tensor cannot be divided into certificate units",
                           operation.id, semantic->id);
        }
        CodecCertificateBinding binding;
        binding.unit_count = element_count / unit_elements;
        binding.planes.reserve(certificate->plane_summaries().size());
        for (const CodecCertificatePlaneSummary& declared : certificate->plane_summaries()) {
            const std::optional<PlaneKind> kind = plane_kind(declared.role);
            const auto physical_plane = kind
                ? std::find_if(physical->planes.begin(), physical->planes.end(),
                               [&](const ArtifactTensorPlane& candidate) {
                                   return candidate.kind == *kind;
                               })
                : physical->planes.end();
            if (!kind || physical_plane == physical->planes.end()) {
                return failure(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                               "codec certificate plane is absent from the physical tensor",
                               operation.id, semantic->id);
            }
            binding.planes.push_back({
                package.artifact_bytes(physical_plane->source.artifact_id),
                physical_plane->source.offset, physical_plane->source.length,
                declared.bytes_per_unit});
        }
        if (certificate->validate(binding) != CodecCertificateError::None) {
            return failure(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                           "codec certificate rejects the authoritative tensor spans",
                           operation.id, semantic->id);
        }
        if (next_occurrence_index == UINT32_MAX) {
            return failure(CompatibilityError::RULE_LIMIT_EXCEEDED,
                           "codec occurrence index exceeds its canonical width",
                           operation.id, semantic->id);
        }
        const CodecProgramIdentity program_identity{
            certificate->identity().abi_version, certificate->identity().digest};
        occurrences.push_back(CodecBindingBuilder::make_tensor(
            next_occurrence_index++, static_cast<uint32_t>(tensor_slot), semantic->id,
            codec->identity, program_identity, semantic->dimensions,
            semantic->layout.strides, std::move(copied_planes)));
    }
    output.insert(output.end(), std::make_move_iterator(occurrences.begin()),
                  std::make_move_iterator(occurrences.end()));
    return std::nullopt;
}

} // namespace

bool ResolvedCodecBindings::matches_package(const RuntimePackage& package) const noexcept {
    if (!package.product_authoritative() ||
        package.package_fingerprint() != package_fingerprint_) return false;
    for (const ResolvedCodecOperator& operation : operators_) {
        for (const ResolvedCodecTensor& tensor : operation.tensors()) {
            for (const ResolvedCodecPlane& plane : tensor.planes()) {
                const PackageView* artifact = artifact_by_id(package.physical_index(),
                                                              plane.artifact_id());
                uint64_t end = 0;
                if (!artifact || artifact->digest() != plane.artifact_digest() ||
                    artifact->bytes().size() != plane.artifact_size() ||
                    !checked_add(plane.offset(), plane.length(), end) ||
                    end > artifact->bytes().size()) {
                    return false;
                }
            }
        }
    }
    return true;
}

CodecBindingPreflightResult CodecBindingBuilder::run(const RuntimePackage& package) {
    try {
        if (!package.product_authoritative() || !package.manifest().has_physical_codec_authority()) {
            return failure(CompatibilityError::PACKAGE_AUTHORITY_REQUIRED,
                           "codec binding requires product package and physical codec authority");
        }
        const PhysicalCodecRegistry& registry = package.physical_codec_registry();
        const SemanticModel& model = package.semantics();
        if (!validate_physical_codec_registry(registry, true) ||
            !physical_codec_registry_is_canonical(registry) ||
            !physical_codec_registry_matches_model(registry, model)) {
            return failure(CompatibilityError::IMPORT_MANIFEST_INVALID,
                           "package codec authority is incomplete, unused, or non-canonical");
        }
        std::vector<ResolvedCodecOperator> resolved;
        resolved.reserve(model.operators.size());
        uint32_t next_occurrence_index = 0;
        for (const SemanticOperator& operation : model.operators) {
            std::vector<ResolvedCodecTensor> occurrences;
            if (auto error = validate_operator(package, operation, registry,
                                               next_occurrence_index, occurrences)) {
                return std::move(*error);
            }
            resolved.push_back(CodecBindingBuilder::make_operator(
                operation.id, std::move(occurrences)));
        }
        return CodecBindingBuilder::make_bindings(package.package_fingerprint(),
                                                  std::move(resolved));
    } catch (const std::bad_alloc&) {
        CompatibilityReport report;
        report.stage = compatibility_stage(CompatibilityError::PACKAGE_BOUNDS_INVALID);
        report.code = CompatibilityError::PACKAGE_BOUNDS_INVALID;
        report.phase = compatibility_phase(CompatibilityError::PACKAGE_BOUNDS_INVALID);
        return report;
    }
}

CodecBindingPreflightResult preflight_codec_bindings(const RuntimePackage& package) {
    return CodecBindingBuilder::run(package);
}

} // namespace Laplace
