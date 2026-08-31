#include "product_metal_codec_capabilities.h"

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <utility>

#include "codec_certificate.h"
#include "compat_rule.h"

namespace Laplace {
namespace {

constexpr size_t kMaximumPrograms = 8;
constexpr size_t kMaximumRecords = 1u << 20;
constexpr size_t kMaximumDimensions = 4;

enum class KnownDecoder : uint8_t {
    RawF16 = 1,
    RawF32 = 2,
    Blocked4 = 3,
    Blocked6 = 4,
    GroupedU2 = 5,
};

struct Occurrence {
    const SemanticOperator* operation = nullptr;
    const SemanticTensor* tensor = nullptr;
    const ResolvedCodecTensor* binding = nullptr;
};

struct BoundOccurrence {
    const SemanticOperator* operation = nullptr;
    const SemanticTensor* tensor = nullptr;
    PhysicalCodecIdentity physical;
    CodecProgramIdentity program;
    uint32_t occurrence_index = 0;
};

CompatibilityReport failure(CompatibilityError code, std::string detail,
                            uint32_t operator_id = kSemanticDispatchUnresolved,
                            uint32_t tensor_id = kSemanticDispatchUnresolved) {
    CompatibilityReport report = compatibility_report(code, std::move(detail));
    report.operator_id = operator_id;
    report.tensor_id = tensor_id;
    return report;
}

bool zero(const CodecProgramIdentity& identity) noexcept {
    return identity.abi_version == 0 ||
           std::all_of(identity.contract_digest.begin(),
                       identity.contract_digest.end(),
                       [](uint8_t value) { return value == 0; });
}

bool zero(const Sha256Digest& digest) noexcept {
    return std::all_of(digest.bytes.begin(), digest.bytes.end(),
                       [](uint8_t value) { return value == 0; });
}

const SemanticOperator* operation_by_id(const SemanticModel& model,
                                        uint32_t id) noexcept {
    const auto found = std::find_if(
        model.operators.begin(), model.operators.end(),
        [id](const SemanticOperator& operation) { return operation.id == id; });
    return found == model.operators.end() ? nullptr : &*found;
}

const SemanticTensor* tensor_by_index(const SemanticModel& model,
                                      uint32_t index) noexcept {
    return index < model.tensors.size() ? &model.tensors[index] : nullptr;
}

const PhysicalCodecSpec* unique_spec(const RuntimePackage& package,
                                     const PhysicalCodecIdentity& identity,
                                     bool& duplicate) noexcept {
    duplicate = false;
    const PhysicalCodecSpec* result = nullptr;
    for (const PhysicalCodecSpec& candidate :
         package.physical_codec_registry().codecs) {
        if (!(candidate.identity == identity)) continue;
        if (result != nullptr) {
            duplicate = true;
            return nullptr;
        }
        result = &candidate;
    }
    return result;
}

std::optional<KnownDecoder> known_decoder(
    const CodecCertificate& actual,
    const PhysicalCodecIdentity& physical) {
    struct Candidate {
        KnownDecoder kind;
        std::vector<uint8_t> (*make)();
    };
    const std::array<Candidate, 5> candidates = {{
        {KnownDecoder::RawF16, make_raw_f16_codec_certificate},
        {KnownDecoder::RawF32, make_raw_f32_codec_certificate},
        {KnownDecoder::Blocked4, make_q4_k_codec_certificate},
        {KnownDecoder::Blocked6, make_q6_k_codec_certificate},
        {KnownDecoder::GroupedU2, make_grouped_affine_u2_codec_certificate},
    }};

    const NormalizedCodecProgramResult actual_normalized =
        normalize_codec_program(actual);
    const auto* actual_program =
        std::get_if<NormalizedCodecProgram>(&actual_normalized);
    if (actual_program == nullptr) return std::nullopt;

    std::optional<KnownDecoder> result;
    for (const Candidate& candidate : candidates) {
        const std::vector<uint8_t> bytes = candidate.make();
        const CodecCertificateParseResult parsed = parse_codec_certificate(bytes);
        const CodecCertificate* certificate =
            std::get_if<CodecCertificate>(&parsed);
        if (certificate == nullptr) continue;
        const NormalizedCodecProgramResult normalized =
            normalize_codec_program(*certificate);
        const auto* program = std::get_if<NormalizedCodecProgram>(&normalized);
        if (program == nullptr ||
            program->semantic_signature != actual_program->semantic_signature) {
            continue;
        }

        // Compare the complete physical tuple through the certificate's own
        // typed contract.  Only the arithmetic identity is replaced with the
        // candidate's known identity; source provenance remains excluded.
        PhysicalCodecIdentity expected = physical;
        expected.identity_version = 1;
        expected.arithmetic_version = certificate->identity().abi_version;
        expected.arithmetic_digest = certificate->identity().digest;
        expected.codebook_digest = {};
        if (!certificate->matches_physical_identity(expected)) continue;
        if (result.has_value()) return std::nullopt;
        result = candidate.kind;
    }
    return result;
}

std::optional<MetalCodecLoweringStrategy> strategy_for(
    OperatorKind operation, KnownDecoder decoder) noexcept {
    switch (decoder) {
    case KnownDecoder::RawF16:
        if (operation == OperatorKind::EmbeddingLookup)
            return MetalCodecLoweringStrategy::StructuralEmbeddingF16;
        if (operation == OperatorKind::Linear ||
            operation == OperatorKind::RoutedLinear)
            return MetalCodecLoweringStrategy::StructuralGemvF16;
        return std::nullopt;
    case KnownDecoder::RawF32:
        if (operation == OperatorKind::Linear ||
            operation == OperatorKind::RoutedLinear)
            return MetalCodecLoweringStrategy::StructuralGemvF32;
        // Raw F32 tensors consumed by fused structural operators are data
        // planes, not standalone GEMV recipes.  The structural compiler still
        // validates the exact operator, slot, shape, and invocation graph
        // before it can use this capability.
        return MetalCodecLoweringStrategy::StructuralTensorF32;
    case KnownDecoder::Blocked4:
        if (operation == OperatorKind::EmbeddingLookup)
            return MetalCodecLoweringStrategy::StructuralEmbeddingQ4;
        if (operation == OperatorKind::Linear ||
            operation == OperatorKind::RoutedLinear)
            return MetalCodecLoweringStrategy::StructuralGemvQ4;
        return std::nullopt;
    case KnownDecoder::Blocked6:
        if (operation == OperatorKind::EmbeddingLookup)
            return MetalCodecLoweringStrategy::StructuralEmbeddingQ6;
        if (operation == OperatorKind::Linear ||
            operation == OperatorKind::RoutedLinear)
            return MetalCodecLoweringStrategy::StructuralGemvQ6;
        return std::nullopt;
    case KnownDecoder::GroupedU2:
        if (operation == OperatorKind::Linear ||
            operation == OperatorKind::RoutedLinear)
            return MetalCodecLoweringStrategy::StructuralGemvAffineU2;
        return std::nullopt;
    }
    return std::nullopt;
}

MetalCodecRequirement make_requirement(const SemanticOperator& operation,
                                       const SemanticDispatchStep& step,
                                       const SemanticTensor& tensor,
                                       const NormalizedCodecProgram& decoder,
                                       const PhysicalCodecIdentity& physical,
                                       CompatibilityReport& error) {
    MetalCodecRequirement result;
    result.operation_abi = static_cast<uint32_t>(operation.kind);
    result.phase = static_cast<uint16_t>(step.phase);
    result.numerical_class = static_cast<uint16_t>(step.numerical_class);
    result.semantic_signature = decoder.semantic_signature;
    result.physical = {physical.layout, physical.quantization, physical.planes};
    if (result.operation_abi == 0 || result.phase == 0 ||
        result.numerical_class == 0 || tensor.dimensions.empty() ||
        tensor.dimensions.size() > kMaximumDimensions) {
        error = failure(CompatibilityError::IR_SHAPE_MISMATCH,
                        "tensor shape cannot form a bounded Metal capability requirement",
                        operation.id, tensor.id);
        return {};
    }
    for (size_t index = 0; index < tensor.dimensions.size(); ++index) {
        const Dimension& dimension = tensor.dimensions[index];
        if (dimension.kind != DimensionKind::Constant ||
            dimension.constant_or_symbol == 0 ||
            dimension.constant_or_symbol > std::numeric_limits<uint32_t>::max()) {
            error = failure(CompatibilityError::IR_SHAPE_MISMATCH,
                            "Metal capability shape must be finite constants",
                            operation.id, tensor.id);
            return {};
        }
        result.shape_class[index] =
            static_cast<uint32_t>(dimension.constant_or_symbol);
    }
    return result;
}

uint64_t lowering_identity(MetalCodecLoweringStrategy strategy) noexcept {
    return UINT64_C(0x4c41504d00000000) |
           static_cast<uint32_t>(strategy);
}

bool valid_package(const RuntimePackage& package,
                   const SemanticModel& model,
                   CompatibilityReport& error) {
    if (!package.product_authoritative() ||
        !package.manifest().has_physical_codec_authority()) {
        error = failure(CompatibilityError::PACKAGE_AUTHORITY_REQUIRED,
                        "Metal codec compilation requires authoritative package codec data");
        return false;
    }
    if (package.package_fingerprint() == Sha256Digest{}) {
        error = failure(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH,
                        "authoritative package fingerprint is empty");
        return false;
    }
    if (semantic_model_digest(package.semantics()) !=
        semantic_model_digest(model)) {
        error = failure(CompatibilityError::IR_REFERENCE_INVALID,
                        "Metal codec compilation model does not match package semantics");
        return false;
    }
    const PhysicalCodecRegistry& registry = package.physical_codec_registry();
    if (!validate_physical_codec_registry(registry, true) ||
        !physical_codec_registry_is_canonical(registry) ||
        !physical_codec_registry_matches_model(registry, model) ||
        package.physical_codec_registry_digest() !=
            physical_codec_registry_digest(registry, model)) {
        error = failure(CompatibilityError::IMPORT_MANIFEST_INVALID,
                        "package physical codec authority is incomplete or non-canonical");
        return false;
    }
    return true;
}

bool make_occurrences(const SemanticModel& model,
                      const ResolvedCodecBindings& bindings,
                      std::vector<Occurrence>& output,
                      CompatibilityReport& error) {
    if (bindings.operators().size() != model.operators.size()) {
        error = failure(CompatibilityError::IR_REFERENCE_INVALID,
                        "resolved codec bindings do not cover the operator graph");
        return false;
    }
    uint32_t expected_occurrence = 0;
    try {
        for (size_t op_index = 0; op_index < model.operators.size(); ++op_index) {
            const SemanticOperator& operation = model.operators[op_index];
            const ResolvedCodecOperator& bound = bindings.operators()[op_index];
            if (bound.operator_id() != operation.id ||
                bound.tensors().size() != operation.tensors.size()) {
                error = failure(CompatibilityError::IR_REFERENCE_INVALID,
                                "resolved codec binding operator order is not canonical",
                                operation.id);
                return false;
            }
            for (size_t slot = 0; slot < operation.tensors.size(); ++slot) {
                const SemanticTensor* tensor =
                    tensor_by_index(model, operation.tensors[slot]);
                const ResolvedCodecTensor& tensor_binding = bound.tensors()[slot];
                if (tensor == nullptr || tensor_binding.tensor_slot() != slot ||
                    tensor_binding.tensor_id() != tensor->id ||
                    tensor_binding.occurrence_index() != expected_occurrence) {
                    error = failure(CompatibilityError::IR_REFERENCE_INVALID,
                                    "resolved codec occurrence order is not canonical",
                                    operation.id,
                                    tensor == nullptr ? operation.tensors[slot]
                                                       : tensor->id);
                    return false;
                }
                if (expected_occurrence == UINT32_MAX) {
                    error = failure(CompatibilityError::RULE_LIMIT_EXCEEDED,
                                    "codec occurrence index exceeds its canonical width",
                                    operation.id, tensor->id);
                    return false;
                }
                output.push_back({&operation, tensor, &tensor_binding});
                ++expected_occurrence;
            }
        }
    } catch (const std::bad_alloc&) {
        error = failure(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                        "Metal codec occurrence map allocation exceeded its bound");
        return false;
    }
    return true;
}

bool check_programs(const SemanticModel& model,
                    std::span<const SemanticDispatchProgram> programs,
                    CompatibilityReport& error) {
    if (programs.empty() || programs.size() > kMaximumPrograms) {
        error = failure(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                        "Metal codec compilation program count is outside its bound");
        return false;
    }
    for (const SemanticDispatchProgram& program : programs) {
        if (!validate_semantic_dispatch_program(model, program.request, program,
                                                &error))
            return false;
    }
    return true;
}

bool append_record(
    const RuntimePackage& package, size_t program_index,
    const SemanticDispatchStep& step, uint32_t tensor_slot,
    const Occurrence& occurrence, const PhysicalCodecIdentity& physical,
    const CodecProgramIdentity& program_identity, uint32_t occurrence_index,
    MetalCodecCapabilityRegistry& registry,
    std::vector<MetalCodecRequirement>& seen,
    std::vector<ProductMetalCodecCapabilityRecord>& records,
    CompatibilityReport& error) {
    const auto declaration = std::find_if(
        package.physical_codec_registry().tensors.begin(),
        package.physical_codec_registry().tensors.end(),
        [&](const PhysicalTensorCodecDeclaration& candidate) {
            return candidate.tensor_id == occurrence.tensor->id;
        });
    if (declaration == package.physical_codec_registry().tensors.end() ||
        declaration->identity != physical) {
        error = failure(CompatibilityError::IMPORT_MANIFEST_INVALID,
                        "codec occurrence physical identity is not package-bound",
                        occurrence.operation->id, occurrence.tensor->id);
        return false;
    }
    bool duplicate_spec = false;
    const PhysicalCodecSpec* spec =
        unique_spec(package, physical, duplicate_spec);
    if (duplicate_spec) {
        error = failure(CompatibilityError::IMPORT_MANIFEST_INVALID,
                        "physical codec identity has duplicate certificate specs",
                        occurrence.operation->id, occurrence.tensor->id);
        return false;
    }
    if (spec == nullptr || spec->certificate_bytes.empty()) {
        error = failure(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                        "tensor occurrence lacks an authoritative decoder certificate",
                        occurrence.operation->id, occurrence.tensor->id);
        return false;
    }
    const CodecCertificateParseResult parsed =
        parse_codec_certificate(spec->certificate_bytes);
    const CodecCertificate* certificate =
        std::get_if<CodecCertificate>(&parsed);
    if (certificate == nullptr ||
        !certificate->matches_physical_identity(physical)) {
        error = failure(CompatibilityError::IMPORT_MANIFEST_INVALID,
                        "decoder certificate does not match its physical tuple",
                        occurrence.operation->id, occurrence.tensor->id);
        return false;
    }
    if (program_identity.abi_version != certificate->identity().abi_version ||
        program_identity.contract_digest != certificate->identity().digest) {
        error = failure(CompatibilityError::IMPORT_MANIFEST_INVALID,
                        "bound decoder identity does not match the package certificate",
                        occurrence.operation->id, occurrence.tensor->id);
        return false;
    }
    const NormalizedCodecProgramResult normalized =
        normalize_codec_program(*certificate);
    const auto* decoder = std::get_if<NormalizedCodecProgram>(&normalized);
    if (decoder == nullptr) {
        error = failure(CompatibilityError::CAPABILITY_MISSING,
                        "decoder semantics cannot be normalized",
                        occurrence.operation->id, occurrence.tensor->id);
        return false;
    }
    const std::optional<KnownDecoder> known =
        known_decoder(*certificate, physical);
    if (!known.has_value()) {
        error = failure(CompatibilityError::CAPABILITY_MISSING,
                        "decoder semantics are unknown or physically ambiguous",
                        occurrence.operation->id, occurrence.tensor->id);
        return false;
    }
    const std::optional<MetalCodecLoweringStrategy> strategy =
        strategy_for(occurrence.operation->kind, *known);
    if (!strategy.has_value()) {
        error = failure(CompatibilityError::CAPABILITY_MISSING,
                        "no Metal lowering exists for this semantic operator and decoder",
                        occurrence.operation->id, occurrence.tensor->id);
        return false;
    }
    const MetalCodecRequirement requirement = make_requirement(
        *occurrence.operation, step, *occurrence.tensor, *decoder, physical,
        error);
    if (requirement.operation_abi == 0) return false;

    const MetalCodecCapability capability{
        requirement,
        // These are portable minimum requirements, not observations about a
        // device. The transaction validates real same-device limits later.
        {0, 1, 1, 1},
        {lowering_identity(*strategy), static_cast<uint32_t>(*strategy)}};
    if (std::find(seen.begin(), seen.end(), requirement) == seen.end()) {
        if (!registry.add(capability, &error)) return false;
        seen.push_back(requirement);
    }
    const MetalCodecResolutionResult resolution =
        registry.resolve_portable(requirement);
    const auto* resolved = std::get_if<MetalCodecResolution>(&resolution);
    if (resolved == nullptr) {
        error = std::get<CompatibilityReport>(resolution);
        return false;
    }

    ProductMetalCodecCapabilityRecord record;
    record.program_index = static_cast<uint32_t>(program_index);
    record.step_ordinal = step.ordinal;
    record.tensor_slot = tensor_slot;
    record.codec_occurrence_index = occurrence_index;
    record.operator_id = occurrence.operation->id;
    record.tensor_id = occurrence.tensor->id;
    record.requirement = requirement;
    record.lowering = resolved->lowering;
    record.capability_digest = resolved->capability_digest;
    record.provenance = normalized_codec_provenance(*certificate);
    record.program_identity = program_identity;
    record.physical_identity = physical;
    try {
        records.push_back(std::move(record));
    } catch (const std::bad_alloc&) {
        error = failure(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                        "Metal codec capability record allocation exceeded its bound",
                        occurrence.operation->id, occurrence.tensor->id);
        return false;
    }
    return true;
}

ProductMetalCodecCapabilityResult compile_common(
    const RuntimePackage& package, const SemanticModel& model,
    std::span<const SemanticDispatchProgram> programs,
    std::span<const Occurrence> occurrences,
    std::span<const BoundOccurrence> bound_occurrences,
    bool bound_mode) {
    try {
        CompatibilityReport error;
        if (!valid_package(package, model, error) ||
            !check_programs(model, programs, error))
            return error;
        MetalCodecCapabilityRegistry registry;
        std::vector<MetalCodecRequirement> seen;
        std::vector<ProductMetalCodecCapabilityRecord> records;
        size_t expected_records = 0;
        for (const SemanticDispatchProgram& program : programs)
            for (const SemanticDispatchStep& step : program.steps) {
                if (step.kind == SemanticDispatchStepKind::Operator) {
                    if (expected_records > kMaximumRecords ||
                        step.tensor_ids.size() > kMaximumRecords - expected_records) {
                        return failure(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                                       "Metal codec capability record count exceeds its bound");
                    }
                    expected_records += step.tensor_ids.size();
                }
            }
        if (expected_records > kMaximumRecords) {
            return failure(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                           "Metal codec capability record count exceeds its bound");
        }
        records.reserve(expected_records);
        size_t bound_cursor = 0;
        for (size_t program_index = 0; program_index < programs.size();
             ++program_index) {
            const SemanticDispatchProgram& program = programs[program_index];
            for (const SemanticDispatchStep& step : program.steps) {
                if (step.kind != SemanticDispatchStepKind::Operator) continue;
                const SemanticOperator* operation =
                    operation_by_id(model, step.operator_id);
                if (operation == nullptr ||
                    step.tensor_ids.size() != operation->tensors.size()) {
                    return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                   "dispatch step tensor coverage is not canonical",
                                   step.operator_id);
                }
                for (size_t slot = 0; slot < step.tensor_ids.size(); ++slot) {
                    if (!bound_mode && occurrences.empty()) {
                        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                       "dispatch step references an absent codec occurrence",
                                       operation->id, step.tensor_ids[slot]);
                    }
                    const SemanticTensor* tensor =
                        tensor_by_index(model, operation->tensors[slot]);
                    if (tensor == nullptr || step.tensor_ids[slot] != operation->tensors[slot]) {
                        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                       "dispatch tensor reference does not match graph order",
                                       operation->id, step.tensor_ids[slot]);
                    }
                    const Occurrence* direct = nullptr;
                    BoundOccurrence bound;
                    if (bound_mode) {
                        if (bound_cursor >= bound_occurrences.size()) {
                            return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                           "bound dispatch occurrence count is incomplete",
                                           operation->id, step.tensor_ids[slot]);
                        }
                        bound = bound_occurrences[bound_cursor];
                    } else {
                        const auto found = std::find_if(
                            occurrences.begin(), occurrences.end(),
                            [&](const Occurrence& candidate) {
                                return candidate.operation == operation &&
                                       candidate.tensor == tensor &&
                                       candidate.binding->tensor_slot() == slot;
                            });
                        if (found == occurrences.end()) {
                            return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                           "dispatch step references an absent codec occurrence",
                                           operation->id, step.tensor_ids[slot]);
                        }
                        direct = &*found;
                    }
                    Occurrence occurrence;
                    PhysicalCodecIdentity physical;
                    CodecProgramIdentity identity;
                    uint32_t occurrence_index = 0;
                    if (direct != nullptr) {
                        occurrence = *direct;
                        physical = direct->binding->physical_identity();
                        identity = direct->binding->program_identity();
                        occurrence_index = direct->binding->occurrence_index();
                    } else {
                        occurrence.operation = bound.operation;
                        occurrence.tensor = bound.tensor;
                        physical = bound.physical;
                        identity = bound.program;
                        occurrence_index = bound.occurrence_index;
                    }
                    if (occurrence.operation != operation ||
                        occurrence.tensor != tensor ||
                        occurrence_index !=
                            (direct != nullptr
                                 ? direct->binding->occurrence_index()
                                 : bound.occurrence_index)) {
                        return failure(CompatibilityError::IR_REFERENCE_INVALID,
                                       "dispatch occurrence does not match semantic tensor",
                                       operation->id, tensor->id);
                    }
                    if (!append_record(package, program_index, step,
                                       static_cast<uint32_t>(slot), occurrence,
                                       physical, identity, occurrence_index,
                                       registry, seen, records, error))
                        return error;
                    if (bound_mode) ++bound_cursor;
                }
            }
        }
        if (records.size() != expected_records) {
            return failure(CompatibilityError::IR_REFERENCE_INVALID,
                           "Metal codec capabilities do not cover every tensor occurrence");
        }
        if (bound_mode && bound_cursor != bound_occurrences.size()) {
            return failure(CompatibilityError::IR_REFERENCE_INVALID,
                           "bound dispatch requirements contain unused occurrences");
        }
        return ProductMetalCodecCapabilityCompilation(std::move(registry),
                                                      std::move(records));
    } catch (const std::bad_alloc&) {
        return failure(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                       "Metal codec capability compilation allocation exceeded its bound");
    }
}

bool make_bound_occurrences(const SemanticModel& model,
                            const BoundDispatchRequirements& bound,
                            std::span<const SemanticDispatchProgram> programs,
                            std::vector<BoundOccurrence>& result,
                            CompatibilityReport& error) {
    if (bound.programs().size() != programs.size() ||
        bound.package_fingerprint() == Sha256Digest{}) {
        error = failure(CompatibilityError::IR_REFERENCE_INVALID,
                        "bound dispatch package or program count is invalid");
        return false;
    }
    std::vector<uint32_t> occurrence_bases(model.operators.size(), 0);
    uint32_t total_occurrences = 0;
    for (size_t operation_index = 0; operation_index < model.operators.size();
         ++operation_index) {
        occurrence_bases[operation_index] = total_occurrences;
        if (model.operators[operation_index].tensors.size() >
            UINT32_MAX - total_occurrences) {
            error = failure(CompatibilityError::RULE_LIMIT_EXCEEDED,
                            "codec occurrence count exceeds its canonical width");
            return false;
        }
        total_occurrences += static_cast<uint32_t>(
            model.operators[operation_index].tensors.size());
    }
    for (size_t program_index = 0; program_index < programs.size(); ++program_index) {
        const SemanticDispatchProgram& program = programs[program_index];
        const BoundDispatchProgram& bound_program = bound.programs()[program_index];
        if (bound_program.program_digest() != program.program_digest ||
            !(bound_program.request() == program.request) ||
            bound_program.steps().size() != program.steps.size()) {
            error = failure(CompatibilityError::IR_REFERENCE_INVALID,
                            "bound dispatch program does not match its semantic program");
            return false;
        }
        for (size_t step_index = 0; step_index < program.steps.size(); ++step_index) {
            const SemanticDispatchStep& step = program.steps[step_index];
            const BoundDispatchStep& bound_step = bound_program.steps()[step_index];
            if (bound_step.ordinal() != step.ordinal ||
                !(bound_step.requirement() == step.requirement) ||
                zero(bound_step.bound_digest())) {
                error = failure(CompatibilityError::IR_REFERENCE_INVALID,
                                "bound dispatch step does not match its semantic step",
                                step.operator_id);
                return false;
            }
            if (step.kind != SemanticDispatchStepKind::Operator) continue;
            const auto occurrences = bound_step.codec_occurrence_indices();
            const auto programs_for_step = bound_step.codec_program_identities();
            const auto physical_for_step = bound_step.physical_identities();
            if (occurrences.size() != step.tensor_ids.size() ||
                programs_for_step.size() != occurrences.size() ||
                physical_for_step.size() != occurrences.size()) {
                error = failure(CompatibilityError::IR_REFERENCE_INVALID,
                                "bound dispatch step tensor coverage is incomplete",
                                step.operator_id);
                return false;
            }
            size_t operation_index = 0;
            const SemanticOperator* operation = nullptr;
            for (; operation_index < model.operators.size(); ++operation_index) {
                if (model.operators[operation_index].id == step.operator_id) {
                    operation = &model.operators[operation_index];
                    break;
                }
            }
            if (operation == nullptr || operation->tensors.size() != occurrences.size()) {
                error = failure(CompatibilityError::IR_REFERENCE_INVALID,
                                "bound dispatch step operator is absent from the model",
                                step.operator_id);
                return false;
            }
            for (size_t slot = 0; slot < occurrences.size(); ++slot) {
                const uint32_t expected = occurrence_bases[operation_index] +
                    static_cast<uint32_t>(slot);
                if (occurrences[slot] != expected ||
                    zero(programs_for_step[slot]) ||
                    !valid_physical_codec_identity(physical_for_step[slot])) {
                    error = failure(CompatibilityError::IR_REFERENCE_INVALID,
                                    "bound dispatch occurrence identity is not canonical",
                                    operation->id, operation->tensors[slot]);
                    return false;
                }
                const SemanticTensor* tensor =
                    tensor_by_index(model, operation->tensors[slot]);
                if (tensor == nullptr) {
                    error = failure(CompatibilityError::IR_REFERENCE_INVALID,
                                    "bound dispatch tensor is absent from the model",
                                    operation->id, operation->tensors[slot]);
                    return false;
                }
                result.push_back({operation, tensor, physical_for_step[slot],
                                  programs_for_step[slot], occurrences[slot]});
            }
        }
    }
    return true;
}

} // namespace

ProductMetalCodecCapabilityResult compile_product_metal_codec_capabilities(
    const RuntimePackage& package, const ResolvedCodecBindings& bindings,
    std::span<const SemanticDispatchProgram> programs,
    const SemanticModel& semantic_model) {
    try {
        CompatibilityReport error;
        if (!valid_package(package, semantic_model, error)) return error;
        if (bindings.package_fingerprint() != package.package_fingerprint() ||
            !bindings.matches_package(package)) {
            return failure(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH,
                           "resolved codec bindings do not match package identity");
        }
        std::vector<Occurrence> occurrences;
        if (!make_occurrences(semantic_model, bindings, occurrences, error))
            return error;
        return compile_common(package, semantic_model, programs, occurrences, {}, false);
    } catch (const std::bad_alloc&) {
        return failure(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                       "Metal codec capability compilation allocation exceeded its bound");
    }
}

ProductMetalCodecCapabilityResult compile_product_metal_codec_capabilities(
    const RuntimePackage& package, const BoundDispatchRequirements& bound,
    std::span<const SemanticDispatchProgram> programs,
    const SemanticModel& semantic_model) {
    try {
        CompatibilityReport error;
        if (!valid_package(package, semantic_model, error)) return error;
        if (bound.package_fingerprint() != package.package_fingerprint()) {
            return failure(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH,
                           "bound dispatch requirements do not match package identity");
        }
        std::vector<BoundOccurrence> occurrences;
        if (!make_bound_occurrences(semantic_model, bound, programs, occurrences, error))
            return error;
        return compile_common(package, semantic_model, programs, {}, occurrences, true);
    } catch (const std::bad_alloc&) {
        return failure(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                       "Metal codec capability compilation allocation exceeded its bound");
    }
}

} // namespace Laplace
