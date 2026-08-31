#include "bound_dispatch_requirements.h"

#include "codec_certificate.h"
#include "compat_rule.h"
#include "execution_plan.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace Laplace {
namespace {

constexpr size_t kMaximumPrograms = 8;
constexpr size_t kMaximumSteps = 1u << 20;
constexpr size_t kMaximumOccurrencesPerStep = 64;

bool zero_digest(const Sha256Digest& digest) noexcept {
    return std::all_of(digest.bytes.begin(), digest.bytes.end(),
                       [](uint8_t value) { return value == 0; });
}

bool zero_digest(const std::array<uint8_t, 32>& digest) noexcept {
    return std::all_of(digest.begin(), digest.end(),
                       [](uint8_t value) { return value == 0; });
}

CompatibilityReport failure(CompatibilityError code, std::string detail,
                            uint32_t operator_id = kSemanticDispatchUnresolved,
                            uint32_t tensor_id = kSemanticDispatchUnresolved) {
    CompatibilityReport report = compatibility_report(code, std::move(detail));
    report.operator_id = operator_id;
    report.tensor_id = tensor_id;
    return report;
}

void append_u16(std::vector<uint8_t>& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8));
}

void append_u8(std::vector<uint8_t>& bytes, uint8_t value) {
    bytes.push_back(value);
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void append_digest(std::vector<uint8_t>& bytes, const Sha256Digest& digest) {
    bytes.insert(bytes.end(), digest.bytes.begin(), digest.bytes.end());
}

void append_program_identity(std::vector<uint8_t>& bytes,
                             const CodecProgramIdentity& identity) {
    append_u16(bytes, identity.abi_version);
    bytes.insert(bytes.end(), identity.contract_digest.begin(),
                 identity.contract_digest.end());
}

void append_physical_identity(std::vector<uint8_t>& bytes,
                              const PhysicalCodecIdentity& identity) {
    append_u16(bytes, identity.identity_version);
    append_u16(bytes, identity.arithmetic_version);
    bytes.insert(bytes.end(), identity.arithmetic_digest.begin(),
                 identity.arithmetic_digest.end());
    bytes.insert(bytes.end(), identity.codebook_digest.begin(),
                 identity.codebook_digest.end());

    const PhysicalLayoutSchema& layout = identity.layout;
    append_u16(bytes, static_cast<uint16_t>(layout.kind));
    append_u16(bytes, layout.version);
    append_u16(bytes, static_cast<uint16_t>(layout.packing));
    bytes.push_back(layout.rank);
    bytes.push_back(layout.block_rank);
    for (uint8_t axis : layout.axis_order) bytes.push_back(axis);
    append_u32(bytes, layout.block_elements);
    append_u32(bytes, layout.block_bytes);
    append_u32(bytes, layout.flags);

    const PhysicalQuantizationSchema& quantization = identity.quantization;
    append_u16(bytes, static_cast<uint16_t>(quantization.kind));
    append_u16(bytes, quantization.version);
    append_u16(bytes, static_cast<uint16_t>(quantization.accumulation_type));
    append_u16(bytes, static_cast<uint16_t>(quantization.scale_type));
    append_u16(bytes, static_cast<uint16_t>(quantization.zero_type));
    append_u16(bytes, static_cast<uint16_t>(quantization.bias_type));
    append_u32(bytes, quantization.block_elements);
    append_u32(bytes, quantization.block_bytes);
    append_u32(bytes, quantization.group_size);
    append_u32(bytes, quantization.required_plane_mask);
    append_u32(bytes, quantization.flags);

    append_u32(bytes, static_cast<uint32_t>(identity.planes.size()));
    for (const PhysicalPlaneSchema& plane : identity.planes) {
        append_u16(bytes, static_cast<uint16_t>(plane.kind));
        append_u16(bytes, static_cast<uint16_t>(plane.storage_type));
        append_u32(bytes, plane.logical_elements_covered);
        append_u32(bytes, plane.bytes_per_block);
        append_u32(bytes, plane.flags);
    }
}

void append_requirement(std::vector<uint8_t>& bytes,
                        const SemanticDispatchRequirement& requirement) {
    append_u16(bytes, requirement.version);
    append_u8(bytes, static_cast<uint8_t>(requirement.step_kind));
    append_u16(bytes, static_cast<uint16_t>(requirement.operation));
    append_u16(bytes, requirement.semantic_version);
    append_u16(bytes, static_cast<uint16_t>(requirement.phase));
    append_u32(bytes, requirement.batch_rows);
    append_u16(bytes, static_cast<uint16_t>(requirement.numerical_class));
    append_digest(bytes, requirement.identity);
}

void append_dimensions(std::vector<uint8_t>& bytes,
                       std::span<const Dimension> dimensions) {
    append_u32(bytes, static_cast<uint32_t>(dimensions.size()));
    for (const Dimension& dimension : dimensions) {
        append_u8(bytes, static_cast<uint8_t>(dimension.kind));
        for (unsigned shift = 0; shift != 64; shift += 8)
            bytes.push_back(static_cast<uint8_t>(dimension.constant_or_symbol >> shift));
    }
}

Sha256Digest semantic_tensor_descriptor_digest(const SemanticTensor& tensor) {
    std::vector<uint8_t> bytes;
    static constexpr std::string_view domain =
        "laplace.bound-dispatch.semantic-tensor.v2";
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    append_u32(bytes, tensor.id);
    append_u16(bytes, static_cast<uint16_t>(tensor.role));
    append_u16(bytes, static_cast<uint16_t>(tensor.logical_type));
    append_dimensions(bytes, tensor.dimensions);
    const PhysicalLayout& layout = tensor.layout;
    append_u16(bytes, static_cast<uint16_t>(layout.kind));
    append_u16(bytes, layout.version);
    append_u16(bytes, static_cast<uint16_t>(layout.packing));
    append_u8(bytes, layout.rank);
    append_u8(bytes, layout.block_rank);
    for (uint8_t axis : layout.axis_order) append_u8(bytes, axis);
    for (uint64_t stride : layout.strides)
        for (unsigned shift = 0; shift != 64; shift += 8)
            bytes.push_back(static_cast<uint8_t>(stride >> shift));
    append_u32(bytes, layout.block_elements);
    append_u32(bytes, layout.block_bytes);
    append_u32(bytes, layout.flags);
    const Quantization& quantization = tensor.quantization;
    append_u16(bytes, static_cast<uint16_t>(quantization.kind));
    append_u16(bytes, quantization.version);
    append_u16(bytes, static_cast<uint16_t>(quantization.accumulation_type));
    append_u16(bytes, static_cast<uint16_t>(quantization.scale_type));
    append_u16(bytes, static_cast<uint16_t>(quantization.zero_type));
    append_u16(bytes, static_cast<uint16_t>(quantization.bias_type));
    append_u32(bytes, quantization.block_elements);
    append_u32(bytes, quantization.block_bytes);
    append_u32(bytes, quantization.group_size);
    append_u32(bytes, quantization.required_plane_mask);
    append_u32(bytes, quantization.flags);
    append_u8(bytes, static_cast<uint8_t>(tensor.expert_axis.kind));
    append_u8(bytes, tensor.expert_axis.expert_axis);
    append_u8(bytes, tensor.expert_axis.member_axis);
    append_u8(bytes, tensor.expert_axis.input_axis);
    append_u8(bytes, tensor.expert_axis.output_axis);
    append_u32(bytes, tensor.expert_axis.expert_count);
    for (unsigned shift = 0; shift != 64; shift += 8)
        bytes.push_back(static_cast<uint8_t>(tensor.expert_axis.per_expert_byte_stride >> shift));
    append_u32(bytes, tensor.expert_axis.flags);
    // Source artifact identity and byte locations belong only to the source
    // span digest below. They must not affect this semantic descriptor.
    append_u32(bytes, static_cast<uint32_t>(tensor.planes.size()));
    for (const TensorPlane& plane : tensor.planes) {
        append_u16(bytes, static_cast<uint16_t>(plane.kind));
        append_u16(bytes, static_cast<uint16_t>(plane.storage_type));
        append_u32(bytes, plane.flags);
    }
    append_u16(bytes, tensor.flags);
    Sha256Digest digest;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.bytes.data());
    return digest;
}

Sha256Digest source_span_digest(const ResolvedCodecTensor& tensor) {
    std::vector<uint8_t> bytes;
    static constexpr std::string_view domain =
        "laplace.bound-dispatch.source-span.v2";
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    append_u32(bytes, tensor.tensor_id());
    append_u32(bytes, static_cast<uint32_t>(tensor.planes().size()));
    for (const ResolvedCodecPlane& plane : tensor.planes()) {
        append_u16(bytes, static_cast<uint16_t>(plane.kind()));
        append_u16(bytes, static_cast<uint16_t>(plane.storage_type()));
        append_u16(bytes, static_cast<uint16_t>(plane.semantic_storage_type()));
        append_u32(bytes, plane.artifact_id().value);
        for (uint64_t value : {plane.offset(), plane.length(), plane.logical_elements(),
                               plane.artifact_size()})
            for (unsigned shift = 0; shift != 64; shift += 8)
                bytes.push_back(static_cast<uint8_t>(value >> shift));
        append_u32(bytes, plane.alignment());
        append_u32(bytes, plane.flags());
        append_u32(bytes, plane.bytes_per_block());
        append_u32(bytes, plane.elements_per_block());
        append_digest(bytes, plane.artifact_digest());
    }
    Sha256Digest digest;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.bytes.data());
    return digest;
}

Sha256Digest value_descriptor_digest(const SemanticValue& value) {
    std::vector<uint8_t> bytes;
    static constexpr std::string_view domain = "laplace.bound-dispatch.value.v2";
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    append_u32(bytes, value.id);
    append_u16(bytes, static_cast<uint16_t>(value.logical_type));
    append_dimensions(bytes, value.dimensions);
    append_u8(bytes, value.flags);
    Sha256Digest digest;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.bytes.data());
    return digest;
}

Sha256Digest state_descriptor_digest(const SemanticState& state) {
    std::vector<uint8_t> bytes;
    static constexpr std::string_view domain = "laplace.bound-dispatch.state.v2";
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    append_u32(bytes, state.id);
    append_u16(bytes, static_cast<uint16_t>(state.kind));
    append_u16(bytes, state.semantic_version);
    append_u16(bytes, static_cast<uint16_t>(state.update_kind));
    append_u16(bytes, static_cast<uint16_t>(state.position_policy));
    append_dimensions(bytes, state.dimensions);
    append_u32(bytes, static_cast<uint32_t>(state.formats.size()));
    for (const StateFormat& format : state.formats) {
        append_u16(bytes, static_cast<uint16_t>(format.kind));
        append_u16(bytes, format.version);
        append_u16(bytes, static_cast<uint16_t>(format.logical_type));
        append_u16(bytes, static_cast<uint16_t>(format.encoded_type));
        append_u16(bytes, static_cast<uint16_t>(format.logical_domain));
        append_u16(bytes, static_cast<uint16_t>(format.encoded_domain));
        append_u16(bytes, static_cast<uint16_t>(format.codec));
        append_u16(bytes, static_cast<uint16_t>(format.cache_policy));
        append_u16(bytes, static_cast<uint16_t>(format.layout_policy));
        append_u16(bytes, format.flags);
        append_u32(bytes, format.tile_tokens);
        append_u32(bytes, format.mutable_tokens);
        append_u32(bytes, format.alignment);
        append_u32(bytes, format.reserved);
    }
    append_u16(bytes, state.flags);
    Sha256Digest digest;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.bytes.data());
    return digest;
}

Sha256Digest output_descriptor_digest(const BoundDispatchOutputBinding& output) {
    std::vector<uint8_t> bytes;
    static constexpr std::string_view domain = "laplace.bound-dispatch.output.v2";
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    append_u32(bytes, output.logits_value_id);
    append_u32(bytes, output.selected_row);
    append_u32(bytes, output.vocabulary_size);
    append_u32(bytes, output.terminal_operator_id);
    Sha256Digest digest;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.bytes.data());
    return digest;
}

Sha256Digest workspace_descriptor_digest(const BoundDispatchWorkspace& workspace) {
    std::vector<uint8_t> bytes;
    static constexpr std::string_view domain = "laplace.bound-dispatch.workspace.v2";
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    append_u32(bytes, workspace.workspace_id);
    Sha256Digest digest;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.bytes.data());
    return digest;
}

Sha256Digest bound_step_digest(
    const SemanticDispatchRequirement& requirement,
    std::span<const BoundDispatchTensor> tensors,
    std::span<const BoundDispatchValue> input_values,
    std::span<const BoundDispatchValue> output_values,
    std::span<const BoundDispatchState> states,
    std::span<const SemanticDispatchSessionEffect> session_effects,
    const std::optional<SemanticDispatchSamplerBinding>& sampler_binding,
    const BoundDispatchWorkspace& workspace, uint32_t ordinal) {
    std::vector<uint8_t> bytes;
    static constexpr std::string_view domain = "laplace.bound-dispatch.step.v2";
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    append_u16(bytes, kBoundDispatchRequirementsVersionV2);
    append_u32(bytes, ordinal);
    append_requirement(bytes, requirement);
    append_u32(bytes, static_cast<uint32_t>(tensors.size()));
    for (const BoundDispatchTensor& tensor : tensors) {
        append_u32(bytes, tensor.occurrence_index);
        append_u32(bytes, tensor.tensor_id);
        append_u32(bytes, tensor.tensor_slot);
        append_program_identity(bytes, tensor.codec_program_identity);
        append_physical_identity(bytes, tensor.physical_identity);
        append_digest(bytes, tensor.semantic_tensor_digest);
        append_digest(bytes, tensor.source_span_digest);
    }
    const auto append_values = [&](std::span<const BoundDispatchValue> values) {
        append_u32(bytes, static_cast<uint32_t>(values.size()));
        for (const BoundDispatchValue& value : values) {
            append_u32(bytes, value.value_id);
            append_digest(bytes, value.descriptor_digest);
        }
    };
    append_values(input_values);
    append_values(output_values);
    append_u32(bytes, static_cast<uint32_t>(states.size()));
    for (const BoundDispatchState& state : states) {
        append_u32(bytes, state.state_id);
        append_u8(bytes, static_cast<uint8_t>(state.access));
        append_u16(bytes, static_cast<uint16_t>(state.update_kind));
        append_digest(bytes, state.descriptor_digest);
    }
    append_u32(bytes, static_cast<uint32_t>(session_effects.size()));
    for (const SemanticDispatchSessionEffect& effect : session_effects)
        append_u8(bytes, static_cast<uint8_t>(effect.kind));
    append_u8(bytes, sampler_binding ? 1 : 0);
    if (sampler_binding) {
        append_u32(bytes, sampler_binding->logits_value_id);
        append_u32(bytes, sampler_binding->selected_row);
        append_u32(bytes, sampler_binding->vocabulary_size);
        append_u32(bytes, sampler_binding->candidate_output_id);
    }
    append_u32(bytes, workspace.workspace_id);
    append_digest(bytes, workspace.descriptor_digest);
    Sha256Digest digest;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.bytes.data());
    return digest;
}

Sha256Digest bound_program_digest(
    const Sha256Digest& program_digest, const SemanticDispatchRequest& request,
    const std::optional<BoundDispatchOutputBinding>& output_binding,
    std::span<const BoundDispatchStep> steps) {
    std::vector<uint8_t> bytes;
    static constexpr std::string_view domain = "laplace.bound-dispatch.program.v2";
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    append_u16(bytes, kBoundDispatchRequirementsVersionV2);
    append_digest(bytes, program_digest);
    append_u16(bytes, static_cast<uint16_t>(request.phase));
    append_u32(bytes, request.batch_rows);
    append_u16(bytes, static_cast<uint16_t>(request.numerical_class));
    append_u8(bytes, request.include_speculative ? 1 : 0);
    append_u8(bytes, request.include_greedy_sampler ? 1 : 0);
    append_u8(bytes, output_binding ? 1 : 0);
    if (output_binding) {
        append_u32(bytes, output_binding->logits_value_id);
        append_u32(bytes, output_binding->selected_row);
        append_u32(bytes, output_binding->vocabulary_size);
        append_u32(bytes, output_binding->terminal_operator_id);
        append_digest(bytes, output_binding->descriptor_digest);
    }
    append_u32(bytes, static_cast<uint32_t>(steps.size()));
    for (const BoundDispatchStep& step : steps) append_digest(bytes, step.bound_digest());
    Sha256Digest digest;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.bytes.data());
    return digest;
}

Sha256Digest bound_requirements_digest(std::span<const BoundDispatchProgram> programs) {
    std::vector<uint8_t> bytes;
    static constexpr std::string_view domain = "laplace.bound-dispatch.requirements.v2";
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    append_u16(bytes, kBoundDispatchRequirementsVersionV2);
    append_u32(bytes, static_cast<uint32_t>(programs.size()));
    for (const BoundDispatchProgram& program : programs)
        append_digest(bytes, program.bound_digest());
    Sha256Digest digest;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.bytes.data());
    return digest;
}

bool same_plane(const ResolvedCodecPlane& resolved, const TensorPlane& semantic) {
    return resolved.kind() == semantic.kind &&
           resolved.semantic_storage_type() == semantic.storage_type &&
           resolved.artifact_id() == semantic.artifact_id &&
           resolved.offset() == semantic.offset &&
           resolved.length() == semantic.length &&
           resolved.alignment() == semantic.alignment &&
           resolved.flags() == semantic.flags;
}

bool same_tensor(const SemanticTensor& semantic, const ResolvedCodecTensor& resolved) {
    if (resolved.dimensions().size() != semantic.dimensions.size() ||
        !std::equal(resolved.dimensions().begin(), resolved.dimensions().end(),
                    semantic.dimensions.begin()) ||
        resolved.strides() != semantic.layout.strides ||
        resolved.planes().size() != semantic.planes.size()) return false;
    for (size_t index = 0; index < semantic.planes.size(); ++index) {
        if (!same_plane(resolved.planes()[index], semantic.planes[index])) return false;
    }
    return true;
}

const PhysicalCodecSpec* codec_spec(const RuntimePackage& package,
                                    const PhysicalCodecIdentity& identity) {
    for (const PhysicalCodecSpec& spec : package.physical_codec_registry().codecs) {
        if (spec.identity == identity) return &spec;
    }
    return nullptr;
}

const SemanticTensor* find_tensor(const SemanticModel& model, uint32_t id);

bool declared_for_tensor(const RuntimePackage& package, uint32_t tensor_id,
                         const PhysicalCodecIdentity& identity) {
    for (const PhysicalTensorCodecDeclaration& declaration :
         package.physical_codec_registry().tensors) {
        if (declaration.tensor_id == tensor_id) return declaration.identity == identity;
    }
    return false;
}

bool certificate_matches(const RuntimePackage& package,
                         const ResolvedCodecTensor& tensor) {
    const PhysicalCodecSpec* spec = codec_spec(package, tensor.physical_identity());
    if (!spec || spec->certificate_bytes.empty()) return false;
    const CodecCertificateParseResult parsed =
        parse_codec_certificate(spec->certificate_bytes);
    const auto* certificate = std::get_if<CodecCertificate>(&parsed);
    return certificate && certificate->identity().abi_version ==
                               tensor.program_identity().abi_version &&
           certificate->identity().digest ==
                               tensor.program_identity().contract_digest &&
           certificate->matches_physical_identity(tensor.physical_identity());
}

std::optional<CompatibilityReport> validate_bindings(
    const RuntimePackage& package, const ResolvedCodecBindings& bindings,
    std::vector<const ResolvedCodecTensor*>& occurrences) {
    if (!package.product_authoritative() ||
        !package.manifest().has_physical_codec_authority()) {
        return failure(CompatibilityError::PACKAGE_AUTHORITY_REQUIRED,
                       "bound dispatch requires an authoritative package");
    }
    if (zero_digest(package.package_fingerprint()) ||
        bindings.package_fingerprint() != package.package_fingerprint() ||
        !bindings.matches_package(package)) {
        return failure(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH,
                       "codec bindings do not belong to the authoritative package");
    }

    const SemanticModel& model = package.semantics();
    if (bindings.operators().size() != model.operators.size()) {
        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                       "codec bindings do not cover the complete operator graph");
    }
    uint64_t total = 0;
    for (const SemanticOperator& operation : model.operators) {
        if (operation.tensors.size() > 64 ||
            total > UINT32_MAX - operation.tensors.size()) {
            return failure(CompatibilityError::RULE_LIMIT_EXCEEDED,
                           "codec occurrence count exceeds the bounded index");
        }
        total += operation.tensors.size();
    }
    occurrences.assign(static_cast<size_t>(total), nullptr);

    uint32_t expected_occurrence = 0;
    for (size_t op_index = 0; op_index < model.operators.size(); ++op_index) {
        const SemanticOperator& operation = model.operators[op_index];
        const ResolvedCodecOperator& bound = bindings.operators()[op_index];
        if (bound.operator_id() != operation.id ||
            bound.tensors().size() != operation.tensors.size()) {
            return failure(CompatibilityError::IR_REFERENCE_INVALID,
                           "codec operator order or tensor count differs from the graph",
                           operation.id);
        }
        for (size_t slot = 0; slot < operation.tensors.size(); ++slot) {
            const uint32_t tensor_id = operation.tensors[slot];
            const SemanticTensor* semantic = find_tensor(model, tensor_id);
            if (!semantic) {
                return failure(CompatibilityError::IR_REFERENCE_INVALID,
                               "operator tensor reference is outside the tensor table",
                               operation.id, tensor_id);
            }
            const ResolvedCodecTensor& tensor = bound.tensors()[slot];
            if (tensor.tensor_slot() != slot || tensor.tensor_id() != tensor_id ||
                tensor.occurrence_index() != expected_occurrence ||
                tensor.occurrence_index() >= occurrences.size() ||
                occurrences[tensor.occurrence_index()] != nullptr ||
                !same_tensor(*semantic, tensor) ||
                !declared_for_tensor(package, tensor_id, tensor.physical_identity()) ||
                !certificate_matches(package, tensor)) {
                return failure(CompatibilityError::IR_REFERENCE_INVALID,
                               "codec occurrence is reordered, replayed, or not certificate-bound",
                               operation.id, tensor_id);
            }
            occurrences[tensor.occurrence_index()] = &tensor;
            ++expected_occurrence;
        }
    }
    if (expected_occurrence != occurrences.size() ||
        std::find(occurrences.begin(), occurrences.end(), nullptr) != occurrences.end()) {
        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                       "codec occurrence sequence is incomplete");
    }
    return std::nullopt;
}

const SemanticOperator* find_operator(const SemanticModel& model, uint32_t id,
                                      size_t* index) {
    for (size_t candidate = 0; candidate < model.operators.size(); ++candidate) {
        if (model.operators[candidate].id == id) {
            if (index) *index = candidate;
            return &model.operators[candidate];
        }
    }
    return nullptr;
}

const SemanticTensor* find_tensor(const SemanticModel& model, uint32_t id) {
    for (const SemanticTensor& tensor : model.tensors)
        if (tensor.id == id) return &tensor;
    return nullptr;
}

const SemanticValue* find_value(const SemanticModel& model, uint32_t id) {
    for (const SemanticValue& value : model.values)
        if (value.id == id) return &value;
    return nullptr;
}

const SemanticState* find_state(const SemanticModel& model, uint32_t id) {
    for (const SemanticState& state : model.states)
        if (state.id == id) return &state;
    return nullptr;
}

bool phase_enabled(const SessionRequest& request, ExecutionPhase phase) {
    return phase == ExecutionPhase::Prefill ? request.enable_prefill :
           phase == ExecutionPhase::Decode ? request.enable_decode : false;
}

std::optional<CompatibilityReport> validate_program_shape(
    const RuntimePackage& package, const SessionRequest& session,
    const SemanticDispatchProgram& program) {
    const SemanticModel& model = package.semantics();
    if (!phase_enabled(session, program.request.phase) ||
        program.request.batch_rows == 0 ||
        program.request.batch_rows > session.max_batch ||
        (program.request.phase == ExecutionPhase::Decode &&
         program.request.batch_rows != 1) ||
        program.request.numerical_class != session.minimum_class ||
        program.request.include_speculative ||
        (program.request.phase != ExecutionPhase::Prefill &&
         program.request.phase != ExecutionPhase::Decode)) {
        return failure(CompatibilityError::RUNTIME_INPUT_INVALID,
                       "dispatch program is outside the requested session");
    }
    CompatibilityReport program_failure;
    if (!validate_semantic_dispatch_program(model, program.request, program,
                                            &program_failure)) {
        program_failure.detail = "dispatch program does not match the authoritative semantic graph";
        return program_failure;
    }
    if (program.model_digest != semantic_model_digest(model) ||
        program.steps.size() > kMaximumSteps || zero_digest(program.program_digest)) {
        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                       "dispatch program has an invalid model digest, digest, or size");
    }
    for (size_t index = 0; index < program.steps.size(); ++index) {
        if (program.steps[index].ordinal != index) {
            return failure(CompatibilityError::IR_REFERENCE_INVALID,
                           "dispatch step ordinals are not contiguous");
        }
    }
    return std::nullopt;
}

std::optional<CompatibilityReport> validate_program_order(
    std::span<const SemanticDispatchProgram> programs) {
    struct Mode {
        ExecutionPhase phase;
        uint32_t batch_rows;
        bool sampled;
    };

    std::vector<Mode> seen;
    seen.reserve(programs.size());
    bool saw_prefill = false;
    bool saw_decode = false;
    uint32_t prefill_batch = 0;
    bool prefill_sampled = false;
    bool decode_sampled = false;
    for (const SemanticDispatchProgram& program : programs) {
        const Mode mode = {program.request.phase, program.request.batch_rows,
                           program.request.include_greedy_sampler};
        if (std::find_if(seen.begin(), seen.end(), [&](const Mode& prior) {
                return prior.phase == mode.phase &&
                       prior.batch_rows == mode.batch_rows &&
                       prior.sampled == mode.sampled;
            }) != seen.end())
            return failure(CompatibilityError::IR_REFERENCE_INVALID,
                           "session contains a duplicate dispatch phase, batch, or output mode");
        seen.push_back(mode);

        if (program.request.phase == ExecutionPhase::Prefill) {
            if (saw_decode || (saw_prefill && mode.batch_rows < prefill_batch) ||
                (saw_prefill && mode.batch_rows > prefill_batch && !prefill_sampled) ||
                (saw_prefill && mode.batch_rows == prefill_batch &&
                 prefill_sampled) ||
                (saw_prefill && mode.batch_rows > prefill_batch && mode.sampled) ||
                (!saw_prefill && mode.sampled)) {
                return failure(CompatibilityError::IR_REFERENCE_INVALID,
                               "dispatch programs are not in canonical prefill order");
            }
            saw_prefill = true;
            prefill_batch = mode.batch_rows;
            prefill_sampled = mode.sampled;
        } else {
            if ((!saw_decode && mode.sampled) || (saw_decode && decode_sampled) ||
                (saw_prefill && !prefill_sampled)) {
                return failure(CompatibilityError::IR_REFERENCE_INVALID,
                               "dispatch programs are not in canonical decode order");
            }
            saw_decode = true;
            decode_sampled = mode.sampled;
        }
    }
    return std::nullopt;
}

} // namespace

class BoundDispatchRequirementsBuilder {
public:
    static BoundDispatchRequirementsResult run(
        const RuntimePackage& package, const ResolvedCodecBindings& bindings,
        const SessionRequest& session_request,
        std::span<const SemanticDispatchProgram> programs) {
        try {
            if (programs.empty() || programs.size() > kMaximumPrograms ||
                session_request.max_context == 0 ||
                session_request.max_context > package.semantics().maximum_context ||
                session_request.max_batch == 0 ||
                (!session_request.enable_prefill && !session_request.enable_decode) ||
                session_request.enable_streaming || session_request.enable_speculation) {
                return failure(CompatibilityError::RUNTIME_INPUT_INVALID,
                               "session request cannot produce bound dispatch programs");
            }

            std::vector<const ResolvedCodecTensor*> occurrences;
            if (const auto report = validate_bindings(package, bindings, occurrences)) {
                return *report;
            }

            if (const auto report = validate_program_order(programs)) return *report;

            bool saw_prefill = false;
            bool saw_decode = false;
            std::vector<BoundDispatchProgram> bound_programs;
            bound_programs.reserve(programs.size());
            for (const SemanticDispatchProgram& program : programs) {
                if (const auto report = validate_program_shape(package, session_request, program)) {
                    return *report;
                }
                bool& saw_phase = program.request.phase == ExecutionPhase::Prefill
                    ? saw_prefill : saw_decode;
                saw_phase = true;

                std::vector<BoundDispatchStep> steps;
                steps.reserve(program.steps.size());
                for (const SemanticDispatchStep& step : program.steps) {
                    std::vector<BoundDispatchTensor> tensors;
                    std::vector<BoundDispatchValue> input_values;
                    std::vector<BoundDispatchValue> output_values;
                    std::vector<BoundDispatchState> states;
                    tensors.reserve(step.tensor_ids.size());
                    input_values.reserve(step.input_values.size());
                    output_values.reserve(step.output_values.size());
                    states.reserve(step.state_effects.size());
                    const auto append_value_records = [&](std::span<const uint32_t> ids,
                                                           std::vector<BoundDispatchValue>& target) {
                        for (uint32_t id : ids) {
                            const SemanticValue* value = find_value(package.semantics(), id);
                            if (!value) return false;
                            target.push_back({id, value_descriptor_digest(*value)});
                        }
                        return true;
                    };
                    if (!append_value_records(step.input_values, input_values) ||
                        !append_value_records(step.output_values, output_values)) {
                        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                       "dispatch value is outside the semantic value table",
                                       step.operator_id);
                    }
                    std::vector<uint32_t> state_ids;
                    state_ids.reserve(step.state_effects.size());
                    for (const SemanticDispatchStateEffect& effect : step.state_effects) {
                        const SemanticState* state = find_state(package.semantics(), effect.state_id);
                        if (!state || effect.access < SemanticDispatchStateAccess::Read ||
                            effect.access > SemanticDispatchStateAccess::ReadWrite ||
                            std::find(state_ids.begin(), state_ids.end(), effect.state_id) !=
                                state_ids.end()) {
                            return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                           "dispatch state descriptor is malformed",
                                           step.operator_id, effect.state_id);
                        }
                        state_ids.push_back(effect.state_id);
                        states.push_back({effect.state_id, effect.access,
                                          effect.update_kind,
                                          state_descriptor_digest(*state)});
                    }
                    for (const SemanticDispatchSessionEffect& effect : step.session_effects) {
                        if (effect.kind < SemanticDispatchSessionEffectKind::PositionCandidateAdvance ||
                            effect.kind > SemanticDispatchSessionEffectKind::CandidateOutputWrite) {
                            return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                           "dispatch session effect is malformed",
                                           step.operator_id);
                        }
                    }
                    std::optional<SemanticDispatchSamplerBinding> sampler_binding;
                    if (step.sampler_binding) sampler_binding = step.sampler_binding;
                    if (step.kind == SemanticDispatchStepKind::GreedySampler &&
                        !sampler_binding) {
                        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                       "greedy sampler is missing its output contract");
                    }
                    if (step.kind == SemanticDispatchStepKind::Operator) {
                        size_t operator_index = 0;
                        const SemanticOperator* operation = find_operator(
                            package.semantics(), step.operator_id, &operator_index);
                        if (!operation) {
                            return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                           "dispatch operator is outside the semantic graph",
                                           step.operator_id);
                        }
                        if (step.tensor_ids != operation->tensors ||
                            step.tensor_ids.size() > kMaximumOccurrencesPerStep) {
                            return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                           "dispatch step tensor order differs from the graph",
                                           step.operator_id);
                        }
                        uint32_t occurrence = 0;
                        for (size_t op = 0; op < operator_index; ++op) {
                            occurrence += static_cast<uint32_t>(
                                package.semantics().operators[op].tensors.size());
                        }
                        const ResolvedCodecOperator& bound =
                            bindings.operators()[operator_index];
                        for (size_t slot = 0; slot < step.tensor_ids.size(); ++slot) {
                            const ResolvedCodecTensor& tensor = bound.tensors()[slot];
                            const uint32_t index = occurrence + static_cast<uint32_t>(slot);
                            if (index >= occurrences.size() || occurrences[index] != &tensor ||
                                tensor.occurrence_index() != index) {
                                return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                               "dispatch step replays or skips a codec occurrence",
                                               step.operator_id, step.tensor_ids[slot]);
                            }
                            const SemanticTensor* semantic =
                                find_tensor(package.semantics(), tensor.tensor_id());
                            if (!semantic) {
                                return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                               "dispatch tensor is outside the semantic tensor table",
                                               step.operator_id, tensor.tensor_id());
                            }
                            tensors.push_back({index, tensor.tensor_id(),
                                               tensor.tensor_slot(),
                                               tensor.program_identity(),
                                               tensor.physical_identity(),
                                               semantic_tensor_descriptor_digest(*semantic),
                                               source_span_digest(tensor)});
                        }
                    } else if (step.kind != SemanticDispatchStepKind::GreedySampler ||
                               !step.tensor_ids.empty() ||
                               step.operator_id != kSemanticDispatchUnresolved) {
                        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                       "dispatch step kind or sampler binding is invalid");
                    }
                    BoundDispatchWorkspace workspace{step.workspace_id, {}};
                    if (workspace.workspace_id == 0) {
                        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                       "dispatch step has no workspace identity",
                                       step.operator_id);
                    }
                    workspace.descriptor_digest = workspace_descriptor_digest(workspace);
                    const Sha256Digest digest = bound_step_digest(
                        step.requirement, tensors, input_values, output_values, states,
                        step.session_effects, sampler_binding, workspace, step.ordinal);
                    steps.push_back(BoundDispatchStep(
                        step.ordinal, step.requirement, digest, std::move(tensors),
                        std::move(input_values), std::move(output_values),
                        std::move(states), step.session_effects,
                        std::move(sampler_binding), std::move(workspace)));
                }
                std::optional<BoundDispatchOutputBinding> output_binding;
                if (program.output_binding) {
                    const SemanticDispatchOutputBinding& output = *program.output_binding;
                    if (!find_value(package.semantics(), output.logits_value_id) ||
                        output.selected_row >= program.request.batch_rows ||
                        output.vocabulary_size != package.semantics().vocabulary_size ||
                        !find_operator(package.semantics(), output.terminal_operator_id, nullptr)) {
                        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                       "dispatch output binding is malformed");
                    }
                    BoundDispatchOutputBinding record{
                        output.logits_value_id, output.selected_row,
                        output.vocabulary_size, output.terminal_operator_id, {}};
                    record.descriptor_digest = output_descriptor_digest(record);
                    output_binding = record;
                }
                if (program.request.include_greedy_sampler && !output_binding) {
                    return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                   "greedy sampler has no bound output row");
                }
                const Sha256Digest program_binding_digest = bound_program_digest(
                    program.program_digest, program.request, output_binding, steps);
                bound_programs.push_back(BoundDispatchProgram(
                    program.program_digest, program.request, std::move(steps),
                    std::move(output_binding), program_binding_digest));
            }
            if (saw_prefill != session_request.enable_prefill ||
                saw_decode != session_request.enable_decode) {
                return failure(CompatibilityError::IR_REFERENCE_INVALID,
                               "session does not contain every requested dispatch phase");
            }
            const Sha256Digest requirements_digest =
                bound_requirements_digest(bound_programs);
            return BoundDispatchRequirements(package.package_fingerprint(),
                                            std::move(bound_programs),
                                            requirements_digest);
        } catch (const std::bad_alloc&) {
            return failure(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                           "bound dispatch allocation exceeded its bound");
        }
    }
};

std::optional<CompatibilityReport> validate_bound_dispatch_step(
    const BoundDispatchStep& step) {
    if (step.version() != kBoundDispatchRequirementsVersionV2) {
        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                       "bound dispatch step uses an unsupported record version",
                       kSemanticDispatchUnresolved);
    }
    if (zero_digest(step.bound_digest())) {
        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                       "bound dispatch step has no binding digest",
                       kSemanticDispatchUnresolved);
    }
    if (step.tensors().size() > kMaximumOccurrencesPerStep) {
        return failure(CompatibilityError::RULE_LIMIT_EXCEEDED,
                       "bound dispatch step exceeds its tensor bound",
                       kSemanticDispatchUnresolved);
    }
    uint32_t previous_occurrence = 0;
    for (size_t index = 0; index < step.tensors().size(); ++index) {
        const BoundDispatchTensor& tensor = step.tensors()[index];
        if (tensor.occurrence_index == kSemanticDispatchUnresolved ||
            tensor.tensor_id == kSemanticDispatchUnresolved ||
            tensor.tensor_slot != index ||
            (index != 0 && tensor.occurrence_index <= previous_occurrence) ||
            tensor.codec_program_identity.abi_version == 0 ||
            zero_digest(tensor.codec_program_identity.contract_digest) ||
            zero_digest(tensor.semantic_tensor_digest) ||
            zero_digest(tensor.source_span_digest)) {
            return failure(CompatibilityError::IR_REFERENCE_INVALID,
                           "bound dispatch tensor record is malformed",
                           kSemanticDispatchUnresolved,
                           tensor.tensor_id);
        }
        previous_occurrence = tensor.occurrence_index;
    }
    const auto validate_values = [&](std::span<const BoundDispatchValue> values) {
        for (const BoundDispatchValue& value : values) {
            if (value.value_id == kSemanticDispatchUnresolved ||
                zero_digest(value.descriptor_digest))
                return false;
        }
        return true;
    };
    if (!validate_values(step.input_values()) ||
        !validate_values(step.output_values())) {
        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                       "bound dispatch value record is malformed",
                       kSemanticDispatchUnresolved);
    }
    for (const BoundDispatchState& state : step.states()) {
        if (state.state_id == kSemanticDispatchUnresolved ||
            state.access < SemanticDispatchStateAccess::Read ||
            state.access > SemanticDispatchStateAccess::ReadWrite ||
            zero_digest(state.descriptor_digest)) {
            return failure(CompatibilityError::IR_REFERENCE_INVALID,
                           "bound dispatch state record is malformed",
                           kSemanticDispatchUnresolved, state.state_id);
        }
    }
    for (const SemanticDispatchSessionEffect& effect : step.session_effects()) {
        if (effect.kind < SemanticDispatchSessionEffectKind::PositionCandidateAdvance ||
            effect.kind > SemanticDispatchSessionEffectKind::CandidateOutputWrite) {
            return failure(CompatibilityError::IR_REFERENCE_INVALID,
                           "bound dispatch session effect is malformed",
                           kSemanticDispatchUnresolved);
        }
    }
    if (step.requirement().step_kind == SemanticDispatchStepKind::GreedySampler &&
        !step.sampler_binding()) {
        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                       "bound greedy sampler is missing its output contract",
                       kSemanticDispatchUnresolved);
    }
    if (step.sampler_binding() &&
        (step.sampler_binding()->logits_value_id == kSemanticDispatchUnresolved ||
         step.sampler_binding()->candidate_output_id == kSemanticDispatchUnresolved)) {
        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                       "bound sampler contract is malformed",
                       kSemanticDispatchUnresolved);
    }
    if (step.workspace().workspace_id == 0 ||
        step.workspace().descriptor_digest != workspace_descriptor_digest(step.workspace())) {
        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                       "bound workspace record is malformed",
                       kSemanticDispatchUnresolved);
    }
    const Sha256Digest expected = bound_step_digest(
        step.requirement(), step.tensors(), step.input_values(), step.output_values(),
        step.states(), step.session_effects(), step.sampler_binding(), step.workspace(),
        step.ordinal());
    if (expected != step.bound_digest()) {
        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                       "bound dispatch step digest does not match its immutable record",
                       kSemanticDispatchUnresolved);
    }
    return std::nullopt;
}

std::optional<CompatibilityReport> validate_bound_dispatch_program(
    const BoundDispatchProgram& program) {
    if (program.version() != kBoundDispatchRequirementsVersionV2 ||
        zero_digest(program.program_digest()) || zero_digest(program.bound_digest()) ||
        program.steps().empty()) {
        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                       "bound dispatch program is malformed or uses an unsupported version");
    }
    for (size_t index = 0; index < program.steps().size(); ++index) {
        const BoundDispatchStep& step = program.steps()[index];
        if (step.ordinal() != index) {
            return failure(CompatibilityError::IR_REFERENCE_INVALID,
                           "bound dispatch program has a non-contiguous step sequence",
                           kSemanticDispatchUnresolved);
        }
        if (const auto report = validate_bound_dispatch_step(step)) return report;
    }
    if (program.output_binding()) {
        const BoundDispatchOutputBinding& output = *program.output_binding();
        if (output.logits_value_id == kSemanticDispatchUnresolved ||
            output.vocabulary_size == 0 ||
            output.terminal_operator_id == kSemanticDispatchUnresolved ||
            output.descriptor_digest != output_descriptor_digest(output)) {
            return failure(CompatibilityError::IR_REFERENCE_INVALID,
                           "bound dispatch output record is malformed");
        }
    }
    const Sha256Digest expected = bound_program_digest(
        program.program_digest(), program.request(), program.output_binding(),
        program.steps());
    if (expected != program.bound_digest()) {
        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                       "bound dispatch program digest does not match its immutable records");
    }
    return std::nullopt;
}

std::optional<CompatibilityReport> validate_bound_dispatch_requirements(
    const BoundDispatchRequirements& requirements) {
    if (requirements.version() != kBoundDispatchRequirementsVersionV2 ||
        zero_digest(requirements.package_fingerprint()) ||
        zero_digest(requirements.bound_digest()) || requirements.programs().empty()) {
        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                       "bound dispatch requirements are malformed or use an unsupported version");
    }
    for (const BoundDispatchProgram& program : requirements.programs()) {
        if (const auto report = validate_bound_dispatch_program(program)) return report;
    }
    const Sha256Digest expected = bound_requirements_digest(requirements.programs());
    if (expected != requirements.bound_digest()) {
        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                       "bound dispatch requirements digest does not match its immutable records");
    }
    return std::nullopt;
}

BoundDispatchRequirementsResult bind_dispatch_requirements(
    const RuntimePackage& package, const ResolvedCodecBindings& bindings,
    const SessionRequest& session_request,
    std::span<const SemanticDispatchProgram> programs) {
    return BoundDispatchRequirementsBuilder::run(package, bindings, session_request, programs);
}

} // namespace Laplace
