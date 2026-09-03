#include "structural_metal_compiler.h"
#include "prefill_tile.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <optional>
#include <string_view>
#include <utility>

namespace Laplace {
namespace {

constexpr size_t kMaximumPrograms = 8;
constexpr size_t kMaximumSteps = 1u << 20;
constexpr size_t kMaximumRecipes = 1u << 20;
constexpr size_t kMaximumLibraries = 8;

struct CertificateFact {
    NormalizedCodecProgram program;
    NormalizedCodecProvenance provenance;
    const PhysicalCodecSpec* spec = nullptr;
    MetalCodecLowering lowering;
};

struct PrimitiveSpec {
    StructuralMetalPrimitive primitive = StructuralMetalPrimitive::VecAdd;
    StructuralMetalLibraryId library = StructuralMetalLibraryId::Core;
    std::string_view function;
    std::string_view abi;
    std::vector<MetalFunctionConstant> constants;
    MetalDispatchConstraints dispatch;
};

struct CompileState {
    const StructuralMetalCompilerInput& input;
    std::vector<StructuralMetalProgramBundle> programs;
    std::vector<MetalPipelineRecipe> recipes;
};

bool zero_digest(const Sha256Digest& digest) noexcept {
    return std::all_of(digest.bytes.begin(), digest.bytes.end(),
                       [](uint8_t value) { return value == 0; });
}

bool zero_digest(const MetalPipelineDigest& digest) noexcept {
    return std::all_of(digest.begin(), digest.end(),
                       [](uint8_t value) { return value == 0; });
}

CompatibilityReport fail(CompatibilityError code, std::string detail,
                         uint32_t operator_id = kSemanticDispatchUnresolved,
                         uint32_t tensor_id = kSemanticDispatchUnresolved) {
    CompatibilityReport result = compatibility_report(code, std::move(detail));
    result.operator_id = operator_id;
    result.tensor_id = tensor_id;
    return result;
}

void append_u8(std::vector<uint8_t>& bytes, uint8_t value) {
    bytes.push_back(value);
}

void append_u16(std::vector<uint8_t>& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8u));
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void append_digest(std::vector<uint8_t>& bytes, const Sha256Digest& digest) {
    bytes.insert(bytes.end(), digest.bytes.begin(), digest.bytes.end());
}

void append_digest(std::vector<uint8_t>& bytes,
                   const MetalPipelineDigest& digest) {
    bytes.insert(bytes.end(), digest.begin(), digest.end());
}

void append_string(std::vector<uint8_t>& bytes, std::string_view value) {
    append_u32(bytes, static_cast<uint32_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
}

MetalPipelineDigest metal_digest(std::span<const uint8_t> bytes) noexcept {
    MetalPipelineDigest result{};
    if (bytes.size() > static_cast<size_t>(std::numeric_limits<CC_LONG>::max()))
        return result;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), result.data());
    return result;
}

void append_physical_tuple(std::vector<uint8_t>& bytes,
                           const MetalCodecPhysicalTuple& physical) {
    const PhysicalLayoutSchema& layout = physical.layout;
    append_u16(bytes, static_cast<uint16_t>(layout.kind));
    append_u16(bytes, layout.version);
    append_u16(bytes, static_cast<uint16_t>(layout.packing));
    append_u8(bytes, layout.rank);
    append_u8(bytes, layout.block_rank);
    for (uint8_t axis : layout.axis_order) append_u8(bytes, axis);
    append_u32(bytes, layout.block_elements);
    append_u32(bytes, layout.block_bytes);
    append_u32(bytes, layout.flags);
    const PhysicalQuantizationSchema& quantization = physical.quantization;
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
    append_u32(bytes, static_cast<uint32_t>(physical.planes.size()));
    for (const PhysicalPlaneSchema& plane : physical.planes) {
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

void append_dispatch(std::vector<uint8_t>& bytes,
                     const MetalDispatchConstraints& dispatch) {
    append_u32(bytes, dispatch.min_threads_per_threadgroup);
    append_u32(bytes, dispatch.max_threads_per_threadgroup);
    append_u32(bytes, dispatch.min_thread_execution_width);
    append_u32(bytes, dispatch.max_thread_execution_width);
    append_u32(bytes, dispatch.required_simdgroups);
}

void append_constants(std::vector<uint8_t>& bytes,
                      const std::vector<MetalFunctionConstant>& constants) {
    append_u32(bytes, static_cast<uint32_t>(constants.size()));
    for (const MetalFunctionConstant& constant : constants) {
        append_u32(bytes, constant.index);
        append_u8(bytes, static_cast<uint8_t>(constant.type));
        append_u64(bytes, constant.value_bits);
    }
}

MetalCodecPhysicalTuple physical_tuple(const PhysicalCodecIdentity& identity);

MetalPipelineDigest recipe_digest(
    const SemanticDispatchStep& step,
    StructuralMetalPrimitive primitive, uint32_t order,
    std::string_view abi, const PrimitiveSpec& spec,
    const StructuralMetalLibraryIdentity& library,
    const std::vector<CertificateFact>& facts) {
    std::vector<uint8_t> bytes;
    static constexpr std::string_view domain = "laplace.structural-metal.recipe.v1";
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    append_requirement(bytes, step.requirement);
    append_u32(bytes, order);
    append_u8(bytes, static_cast<uint8_t>(primitive));
    append_string(bytes, abi);
    append_string(bytes, spec.function);
    append_u8(bytes, static_cast<uint8_t>(spec.library));
    append_digest(bytes, library.source_digest);
    append_constants(bytes, spec.constants);
    append_dispatch(bytes, spec.dispatch);
    append_u32(bytes, static_cast<uint32_t>(facts.size()));
    for (const CertificateFact& fact : facts) {
        append_digest(bytes, fact.program.semantic_signature);
        append_physical_tuple(bytes, physical_tuple(fact.spec->identity));
        append_u64(bytes, fact.lowering.identity);
        append_u32(bytes, fact.lowering.strategy);
    }
    return metal_digest(bytes);
}

const SemanticOperator* operator_for(const SemanticModel& model, uint32_t id,
                                     size_t* index = nullptr) {
    for (size_t candidate = 0; candidate < model.operators.size(); ++candidate) {
        if (model.operators[candidate].id == id) {
            if (index) *index = candidate;
            return &model.operators[candidate];
        }
    }
    return nullptr;
}

const PhysicalCodecSpec* unique_spec(
    std::span<const PhysicalCodecSpec> specs,
    const PhysicalCodecIdentity& identity) {
    const PhysicalCodecSpec* found = nullptr;
    for (const PhysicalCodecSpec& spec : specs) {
        if (!(spec.identity == identity)) continue;
        if (found) return nullptr;
        found = &spec;
    }
    return found;
}

bool checked_u32_dimension(const Dimension& dimension, uint32_t* value) noexcept {
    if (dimension.kind != DimensionKind::Constant ||
        dimension.constant_or_symbol == 0 ||
        dimension.constant_or_symbol > UINT32_MAX)
        return false;
    *value = static_cast<uint32_t>(dimension.constant_or_symbol);
    return true;
}

MetalCodecPhysicalTuple physical_tuple(const PhysicalCodecIdentity& identity) {
    return {identity.layout, identity.quantization, identity.planes};
}

uint32_t operation_abi(const SemanticOperator& operation) noexcept {
    return static_cast<uint32_t>(operation.kind);
}

std::optional<MetalCodecRequirement> codec_requirement(
    const SemanticDispatchStep& step, const SemanticOperator& operation,
    const SemanticTensor& tensor,
    const NormalizedCodecProgram& normalized,
    const PhysicalCodecIdentity& physical) {
    MetalCodecRequirement requirement;
    requirement.operation_abi = operation_abi(operation);
    requirement.phase = static_cast<uint16_t>(step.phase);
    requirement.numerical_class =
        static_cast<uint16_t>(step.numerical_class);
    if (requirement.phase == 0 || requirement.numerical_class == 0 ||
        tensor.dimensions.size() > requirement.shape_class.size())
        return std::nullopt;
    for (size_t index = 0; index < tensor.dimensions.size(); ++index)
        if (!checked_u32_dimension(tensor.dimensions[index],
                                   &requirement.shape_class[index]))
            return std::nullopt;
    requirement.semantic_signature = normalized.semantic_signature;
    requirement.physical = physical_tuple(physical);
    return requirement;
}

std::optional<CertificateFact> certificate_for(
    const StructuralMetalCompilerInput& input,
    const BoundDispatchStep& step, size_t index) {
    const auto physical = step.physical_identities();
    const auto programs = step.codec_program_identities();
    if (index >= physical.size() || index >= programs.size()) return std::nullopt;
    const PhysicalCodecSpec* spec = unique_spec(input.certificate_specs,
                                                 physical[index]);
    if (!spec || spec->certificate_bytes.empty()) return std::nullopt;
    const auto parsed = parse_codec_certificate(spec->certificate_bytes);
    const auto* certificate = std::get_if<CodecCertificate>(&parsed);
    if (!certificate || certificate->identity().abi_version !=
                            programs[index].abi_version ||
        certificate->identity().digest != programs[index].contract_digest ||
        certificate->identity().digest != spec->identity.arithmetic_digest ||
        !certificate->matches_physical_identity(spec->identity))
        return std::nullopt;
    const auto normalized = normalize_codec_program(*certificate);
    if (!std::holds_alternative<NormalizedCodecProgram>(normalized))
        return std::nullopt;
    CertificateFact result;
    result.program = std::get<NormalizedCodecProgram>(normalized);
    result.provenance = normalized_codec_provenance(*certificate);
    result.spec = spec;
    return result;
}

bool checked_mul(uint64_t left, uint64_t right, uint64_t* result) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
        return false;
    *result = left * right;
    return true;
}

ScalarType scalar_for(CodecCertificateStorageScalar scalar) {
    switch (scalar) {
    case CodecCertificateStorageScalar::Binary16: return ScalarType::F16;
    case CodecCertificateStorageScalar::Binary32: return ScalarType::F32;
    case CodecCertificateStorageScalar::Unsigned8: return ScalarType::U8;
    case CodecCertificateStorageScalar::Unsigned16: return ScalarType::U32;
    case CodecCertificateStorageScalar::Unsigned32: return ScalarType::U32;
    case CodecCertificateStorageScalar::Signed8: return ScalarType::U8;
    }
    return static_cast<ScalarType>(0);
}

bool valid_tensor_geometry(const SemanticTensor& tensor,
                           const NormalizedCodecProgram& decoder) {
    if (tensor.dimensions.empty() || tensor.dimensions.size() > 8 ||
        tensor.layout.rank != tensor.dimensions.size()) return false;
    const PhysicalLayoutKind layout_kind = tensor.layout.kind;
    const bool grouped_layout =
        layout_kind == PhysicalLayoutKind::GroupedAffine ||
        layout_kind == PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip;
    if (grouped_layout &&
        (tensor.dimensions.size() != 2 || tensor.layout.axis_order[0] != 1 ||
         tensor.layout.axis_order[1] != 0 || tensor.layout.strides[0] != 1 ||
         tensor.layout.strides[1] != tensor.dimensions[1].constant_or_symbol))
        return false;
    uint64_t elements = 1;
    for (size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
        const Dimension& dimension = tensor.dimensions[axis];
        if (dimension.kind != DimensionKind::Constant ||
            dimension.constant_or_symbol == 0 ||
            dimension.constant_or_symbol > UINT32_MAX)
            return false;
        if (!checked_mul(elements, dimension.constant_or_symbol, &elements))
            return false;
    }
    if (!grouped_layout) {
        const auto exact_contiguous = [&](bool first_axis_fastest) {
            uint64_t stride = 1;
            for (size_t ordinal = 0; ordinal != tensor.dimensions.size(); ++ordinal) {
                const size_t physical_axis = first_axis_fastest
                    ? ordinal : tensor.dimensions.size() - ordinal - 1;
                const uint8_t logical_axis =
                    tensor.layout.axis_order[physical_axis];
                if (logical_axis >= tensor.dimensions.size() ||
                    tensor.layout.strides[physical_axis] != stride ||
                    !checked_mul(stride,
                                 tensor.dimensions[logical_axis]
                                     .constant_or_symbol,
                                 &stride))
                    return false;
            }
            return true;
        };
        const bool exact_forward = exact_contiguous(true);
        const bool exact_reverse = exact_contiguous(false);
        if (layout_kind == PhysicalLayoutKind::GgufBlocked
                ? !exact_forward
                : !exact_forward && !exact_reverse)
            return false;
    }
    const CodecCertificateSummary& summary = decoder.summary;
    if (summary.unit_elements == 0 || elements % summary.unit_elements != 0 ||
        elements / summary.unit_elements > summary.maximum_units ||
        tensor.planes.size() != decoder.planes.size())
        return false;
    const uint64_t units = elements / summary.unit_elements;
    for (size_t index = 0; index < tensor.planes.size(); ++index) {
        const TensorPlane& plane = tensor.planes[index];
        const CodecCertificatePlaneSummary& declared =
            decoder.planes[index];
        if (plane.storage_type != scalar_for(declared.storage_scalar) ||
            plane.alignment < declared.alignment ||
            declared.bytes_per_unit == 0)
            return false;
        uint64_t bytes = 0;
        if (!checked_mul(units, declared.bytes_per_unit, &bytes) ||
            plane.length < bytes)
            return false;
    }
    return true;
}

std::optional<std::vector<CertificateFact>> facts_for(
    const StructuralMetalCompilerInput& input, const SemanticModel& model,
    const SemanticDispatchStep& step, const BoundDispatchStep& bound,
    const SemanticOperator& operation) {
    const auto physical = bound.physical_identities();
    const auto programs = bound.codec_program_identities();
    if (physical.size() != operation.tensors.size() ||
        programs.size() != operation.tensors.size() ||
        step.tensor_ids != operation.tensors)
        return std::nullopt;
    std::vector<CertificateFact> facts;
    facts.reserve(physical.size());
    for (size_t index = 0; index < physical.size(); ++index) {
        if (operation.tensors[index] >= model.tensors.size()) return std::nullopt;
        auto fact = certificate_for(input, bound, index);
        if (!fact) return std::nullopt;
        const SemanticTensor& tensor = model.tensors[operation.tensors[index]];
        if (!valid_tensor_geometry(tensor, fact->program) ||
            input.codec_capabilities == nullptr)
            return std::nullopt;
        const auto requirement = codec_requirement(
            step, operation, tensor, fact->program, physical[index]);
        if (!requirement) return std::nullopt;
        const auto resolution = input.codec_capabilities->resolve_portable(
            *requirement);
        if (!std::holds_alternative<MetalCodecResolution>(resolution))
            return std::nullopt;
        fact->lowering = std::get<MetalCodecResolution>(resolution).lowering;
        facts.push_back(std::move(*fact));
    }
    return facts;
}

MetalDispatchConstraints dispatch(uint32_t minimum, uint32_t maximum) {
    return {minimum, maximum, 0, 0, 0};
}

PrimitiveSpec primitive_spec(StructuralMetalPrimitive primitive) {
    PrimitiveSpec result;
    result.primitive = primitive;
    result.dispatch = dispatch(1, 1024);
    switch (primitive) {
    case StructuralMetalPrimitive::EmbeddingF16:
        result.function = "embedding_f16"; result.abi = "embedding.f16.v1";
        result.dispatch = dispatch(1, 1024); break;
    case StructuralMetalPrimitive::EmbeddingQ4:
        result.function = "embedding_q4k"; result.abi = "embedding.q4.v1";
        result.dispatch = dispatch(1, 1024); break;
    case StructuralMetalPrimitive::EmbeddingQ6:
        result.function = "embedding_q6k"; result.abi = "embedding.q6.v1";
        result.dispatch = dispatch(1, 1024); break;
    case StructuralMetalPrimitive::RmsNormF32:
        result.function = "rmsnorm_f32"; result.abi = "rmsnorm.f32.v1";
        result.dispatch = dispatch(256, 1024); break;
    case StructuralMetalPrimitive::RmsNormRowsF32:
        result.function = "rmsnorm_rows_f32";
        result.abi = "rmsnorm.rows.f32.v1";
        result.dispatch = dispatch(64, 1024); break;
    case StructuralMetalPrimitive::RmsNormNoScale:
        result.function = "rmsnorm_noscale"; result.abi = "rmsnorm.noscale.v1";
        result.dispatch = dispatch(256, 1024); break;
    case StructuralMetalPrimitive::GemvF16:
        result.function = "gemv"; result.abi = "gemv.f16.v1";
        result.constants = {{0, MetalFunctionConstantType::Int32, 1}};
        result.dispatch = dispatch(32, 1024); break;
    case StructuralMetalPrimitive::GemvF32:
        result.function = "gemv"; result.abi = "gemv.f32.v1";
        result.constants = {{0, MetalFunctionConstantType::Int32, 0}};
        result.dispatch = dispatch(32, 1024); break;
    case StructuralMetalPrimitive::GemvQ4:
        result.function = "gemv_q4k"; result.abi = "gemv.q4.v1";
        result.dispatch = dispatch(64, 1024); break;
    case StructuralMetalPrimitive::GemvQ6:
        result.function = "gemv_q6k"; result.abi = "gemv.q6.v1";
        result.dispatch = dispatch(64, 1024); break;
    case StructuralMetalPrimitive::GemvAffineU2:
        result.function = "gemv_affine_u2_256"; result.abi = "gemv.affine.u2.v1";
        result.constants = {
            {2, MetalFunctionConstantType::UInt32, 4},
            {3, MetalFunctionConstantType::UInt32, 2},
            {4, MetalFunctionConstantType::Bool, 0},
            {5, MetalFunctionConstantType::Bool, 0},
            {6, MetalFunctionConstantType::UInt32, 16},
        };
        result.dispatch = dispatch(64, 1024);
        result.dispatch.required_simdgroups = 2;
        break;
    case StructuralMetalPrimitive::PrefillF16Rows:
        result.library = StructuralMetalLibraryId::Prefill;
        result.function = "prefill_f16_rows"; result.abi = "prefill.f16.rows.v1";
        result.dispatch = dispatch(2, 1024); break;
    case StructuralMetalPrimitive::PrefillF16Tile:
        // One simdgroup of exactly 32 threads computes each 32x32 output
        // tile; threadgroup memory is set by the executor at dispatch time.
        result.library = StructuralMetalLibraryId::Prefill;
        result.function = "prefill_f16_tile"; result.abi = "prefill.f16.tile.v1";
        result.dispatch = dispatch(32, 32); break;
    case StructuralMetalPrimitive::VecAdd:
        result.function = "vec_add"; result.abi = "vec.add.v1";
        result.dispatch = dispatch(1, 1024); break;
    case StructuralMetalPrimitive::SwiGlu:
        result.function = "act_glu"; result.abi = "activation.glu.v1";
        result.dispatch = dispatch(64, 1024); break;
    case StructuralMetalPrimitive::RopeHalfSplit:
        result.function = "rope_f32"; result.abi = "rope.half.v1";
        result.dispatch = dispatch(1, 1024); break;
    case StructuralMetalPrimitive::RopeInterleaved:
        result.function = "rope_interleaved_f32"; result.abi = "rope.interleaved.v1";
        result.dispatch = dispatch(1, 1024); break;
    case StructuralMetalPrimitive::RopeMultiSection:
        result.function = "rope_multisection_f32"; result.abi = "rope.multisection.v1";
        result.dispatch = dispatch(1, 1024); break;
    case StructuralMetalPrimitive::KvWrite:
        result.function = "kv_write"; result.abi = "kv.write.v1";
        result.dispatch = dispatch(1, 1024); break;
    case StructuralMetalPrimitive::Attention:
        result.function = "attn_decode"; result.abi = "attention.decode.v1";
        result.dispatch = dispatch(32, 1024); break;
    case StructuralMetalPrimitive::GatedAttention:
        result.function = "gated_attention_f32"; result.abi = "attention.gated.v1";
        result.dispatch = dispatch(1, 1024); break;
    case StructuralMetalPrimitive::AxisSplit:
        result.function = "axis_split_f32"; result.abi = "axis.split.v1";
        result.dispatch = dispatch(1, 1024); break;
    case StructuralMetalPrimitive::DnetConvSilu:
        result.function = "dnet_conv_silu"; result.abi = "dnet.conv.v1";
        result.dispatch = dispatch(64, 1024); break;
    case StructuralMetalPrimitive::DnetL2:
        result.function = "dnet_l2_qk"; result.abi = "dnet.l2.v1";
        result.dispatch = dispatch(32, 1024); break;
    case StructuralMetalPrimitive::DnetUpdate:
        result.function = "dnet_update"; result.abi = "dnet.update.v1";
        result.dispatch = dispatch(32, 1024); break;
    case StructuralMetalPrimitive::RouterTopK:
        result.function = "router_topk"; result.abi = "router.topk.v1";
        result.dispatch = dispatch(256, 1024); break;
    case StructuralMetalPrimitive::GemvQ4Expert:
        result.function = "gemv_q4k_id"; result.abi = "gemv.expert.q4.v1";
        result.dispatch = dispatch(64, 1024); break;
    case StructuralMetalPrimitive::GatedActivationExperts:
        result.function = "act_glu_experts"; result.abi = "activation.experts.v1";
        result.dispatch = dispatch(64, 1024); break;
    case StructuralMetalPrimitive::ExpertReduce:
        result.function = "moe_combine"; result.abi = "expert.reduce.v1";
        result.dispatch = dispatch(64, 1024); break;
    case StructuralMetalPrimitive::ApplyDownScale:
        result.function = "apply_down_scale"; result.abi = "expert.scale.v1";
        result.dispatch = dispatch(1, 1024); break;
    case StructuralMetalPrimitive::VecScale:
        result.function = "vec_scale"; result.abi = "vec.scale.v1";
        result.dispatch = dispatch(1, 1024); break;
    case StructuralMetalPrimitive::VecMul:
        result.function = "vec_mul"; result.abi = "vec.mul.v1";
        result.dispatch = dispatch(1, 1024); break;
    case StructuralMetalPrimitive::SamplerGreedy:
        result.library = StructuralMetalLibraryId::Sampler;
        result.function = "sampler_greedy_f32"; result.abi = "sampler.greedy.v1";
        result.dispatch = dispatch(256, 256); break;
    case StructuralMetalPrimitive::ColumnGroupedSelect:
        result.function = "column_grouped_affine_u2_skip_256_select";
        result.abi = "column.grouped.select.v1"; result.dispatch = dispatch(256, 256); break;
    case StructuralMetalPrimitive::ColumnGroupedPartial:
        result.function = "column_grouped_affine_u2_skip_256_partial";
        result.abi = "column.grouped.partial.v1"; result.dispatch = dispatch(64, 64); break;
    case StructuralMetalPrimitive::ColumnGroupedReduce:
        result.function = "column_grouped_affine_u2_skip_256_reduce";
        result.abi = "column.grouped.reduce.v1"; result.dispatch = dispatch(256, 256); break;
    }
    return result;
}

bool has_library(const StructuralMetalCompilerInput& input,
                 StructuralMetalLibraryId id) {
    return input.libraries.find(id) != nullptr;
}

std::optional<CompatibilityReport> validate_libraries(
    const StructuralMetalCompilerInput& input) {
    if (input.libraries.identities.empty() ||
        input.libraries.identities.size() > kMaximumLibraries)
        return fail(CompatibilityError::CAPABILITY_MISSING,
                    "structural Metal compilation has no library identity set");
    for (size_t index = 0; index < input.libraries.identities.size(); ++index) {
        const auto& library = input.libraries.identities[index];
        if (library.id != StructuralMetalLibraryId::Core &&
            library.id != StructuralMetalLibraryId::Prefill &&
            library.id != StructuralMetalLibraryId::Sampler)
            return fail(CompatibilityError::CAPABILITY_MISSING,
                        "structural Metal compilation has an unknown library identity");
        if (zero_digest(library.source_digest))
            return fail(CompatibilityError::CAPABILITY_MISSING,
                        "structural Metal compilation has an incomplete library identity");
        for (size_t prior = 0; prior != index; ++prior)
            if (input.libraries.identities[prior].id == library.id)
                return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                            "structural Metal compilation has duplicate library identity");
    }
    if (!has_library(input, StructuralMetalLibraryId::Core))
        return fail(CompatibilityError::CAPABILITY_MISSING,
                    "structural Metal compilation requires the core library identity");
    return std::nullopt;
}

std::optional<StructuralMetalPrimitive> primitive_for_strategy(
    uint32_t value) noexcept {
    // Bounded universal slice: only app-owned recipes with a known structural
    // lowering are executable. Generic Metal JIT/emitter support remains
    // pending, so every unknown normalized decoder fails closed.
    switch (static_cast<MetalCodecLoweringStrategy>(value)) {
    case MetalCodecLoweringStrategy::StructuralEmbeddingF16:
        return StructuralMetalPrimitive::EmbeddingF16;
    case MetalCodecLoweringStrategy::StructuralEmbeddingQ4:
        return StructuralMetalPrimitive::EmbeddingQ4;
    case MetalCodecLoweringStrategy::StructuralEmbeddingQ6:
        return StructuralMetalPrimitive::EmbeddingQ6;
    case MetalCodecLoweringStrategy::StructuralGemvF16:
        return StructuralMetalPrimitive::GemvF16;
    case MetalCodecLoweringStrategy::StructuralGemvQ4:
        return StructuralMetalPrimitive::GemvQ4;
    case MetalCodecLoweringStrategy::StructuralGemvQ6:
        return StructuralMetalPrimitive::GemvQ6;
    case MetalCodecLoweringStrategy::StructuralGemvAffineU2:
        return StructuralMetalPrimitive::GemvAffineU2;
    case MetalCodecLoweringStrategy::StructuralGemvF32:
        return StructuralMetalPrimitive::GemvF32;
    case MetalCodecLoweringStrategy::StructuralGemvColumnGroupedU2:
    case MetalCodecLoweringStrategy::StructuralTensorF32:
        return std::nullopt;
    }
    return std::nullopt;
}

bool has_strategy(const CertificateFact& fact,
                  MetalCodecLoweringStrategy strategy) noexcept {
    return fact.lowering.strategy == static_cast<uint32_t>(strategy);
}

// Exact admitted batched prefill widths: two (the legacy rows kernel) and
// the declared tile width (the tiled kernel). The plan pattern admits the
// full interval between them; this is the exact structural gate.
bool admitted_batch_prefill(const SemanticDispatchProgram& program) noexcept {
    return program.request.phase == ExecutionPhase::Prefill &&
           (program.request.batch_rows == 2 ||
            program.request.batch_rows == kPrefillTileRows);
}

bool is_final_output_operator(const SemanticModel& model,
                              const SemanticDispatchProgram& program,
                              const SemanticOperator& operation) {
    if (model.output_values_count == 0 ||
        model.output_values_first > model.values.size() ||
        model.output_values_count > model.values.size() - model.output_values_first ||
        operation.outputs.size() != model.output_values_count)
        return false;
    for (size_t index = 0; index < operation.outputs.size(); ++index)
        if (operation.outputs[index] != model.output_values_first + index)
            return false;
    return std::find(program.terminal_operator_ids.begin(),
                     program.terminal_operator_ids.end(), operation.id) !=
           program.terminal_operator_ids.end();
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
            return fail(CompatibilityError::IR_REFERENCE_INVALID,
                        "session contains a duplicate dispatch phase, batch, or output mode");
        seen.push_back(mode);

        if (program.request.phase == ExecutionPhase::Prefill) {
            if (saw_decode || (saw_prefill && mode.batch_rows < prefill_batch) ||
                (saw_prefill && mode.batch_rows > prefill_batch && !prefill_sampled) ||
                (saw_prefill && mode.batch_rows == prefill_batch && prefill_sampled) ||
                (saw_prefill && mode.batch_rows > prefill_batch && mode.sampled) ||
                (!saw_prefill && mode.sampled)) {
                return fail(CompatibilityError::IR_REFERENCE_INVALID,
                            "dispatch programs are not in canonical prefill order");
            }
            saw_prefill = true;
            prefill_batch = mode.batch_rows;
            prefill_sampled = mode.sampled;
        } else {
            if ((!saw_decode && mode.sampled) || (saw_decode && decode_sampled) ||
                (saw_prefill && !prefill_sampled)) {
                return fail(CompatibilityError::IR_REFERENCE_INVALID,
                            "dispatch programs are not in canonical decode order");
            }
            saw_decode = true;
            decode_sampled = mode.sampled;
        }
    }
    return std::nullopt;
}

std::optional<CompatibilityReport> validate_batched_program(
    const SemanticModel& model, const SemanticDispatchProgram& program,
    const std::vector<std::vector<CertificateFact>>& facts) {
    if (!admitted_batch_prefill(program)) return std::nullopt;
    for (size_t index = 0; index < program.steps.size(); ++index) {
        const SemanticDispatchStep& step = program.steps[index];
        if (step.kind != SemanticDispatchStepKind::Operator) continue;
        const SemanticOperator* operation = operator_for(model, step.operator_id);
        if (!operation) return fail(CompatibilityError::IR_REFERENCE_INVALID,
                                    "batched prefill program has an unknown operator",
                                    step.operator_id);
        switch (operation->kind) {
        case OperatorKind::EmbeddingLookup:
            if (facts[index].size() != 1 ||
                !has_strategy(facts[index][0],
                              MetalCodecLoweringStrategy::StructuralEmbeddingF16))
                return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                            "batched F16 prefill requires an F16 embedding",
                            operation->id);
            break;
        case OperatorKind::Linear:
            if (!is_final_output_operator(model, program, *operation) &&
                (std::get<LinearPayload>(operation->payload).has_bias ||
                facts[index].empty() ||
                !has_strategy(facts[index][0],
                              MetalCodecLoweringStrategy::StructuralGemvF16)))
                return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                            "batched F16 prefill requires unbiased F16 linears",
                            operation->id);
            break;
        case OperatorKind::RmsNorm:
        case OperatorKind::Rope:
        case OperatorKind::CausalAttention:
        case OperatorKind::SwiGlu:
        case OperatorKind::Add:
            break;
        default:
            return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                        "batched F16 prefill contains an unsupported semantic operation",
                        operation->id);
        }
    }
    return std::nullopt;
}

bool fused_operator(OperatorKind kind) noexcept {
    return kind == OperatorKind::CausalAttention ||
           kind == OperatorKind::GatedDeltaNet ||
           kind == OperatorKind::GatedRmsNorm ||
           kind == OperatorKind::GatedActivation ||
           kind == OperatorKind::WeightedExpertReduce ||
           kind == OperatorKind::Linear;
}

bool find_recurrent_layer_owner(const SemanticModel& model,
                                const SemanticDispatchProgram& program,
                                uint32_t covered_step_ordinal, uint32_t* owner) {
    const SemanticDispatchLayerView* containing = nullptr;
    for (const SemanticDispatchLayerView& view : program.layer_views) {
        if (view.step_count == 0) continue;
        if (view.first_step == kSemanticDispatchUnresolved ||
            covered_step_ordinal >= program.steps.size()) continue;
        if (covered_step_ordinal >= view.first_step &&
            covered_step_ordinal - view.first_step < view.step_count) {
            if (containing) return false;
            containing = &view;
        }
    }
    if (!containing) return false;
    uint32_t found = kSemanticDispatchUnresolved;
    for (uint32_t index = containing->first_step;
         index < containing->first_step + containing->step_count; ++index) {
        const SemanticDispatchStep& step = program.steps[index];
        if (step.kind != SemanticDispatchStepKind::Operator ||
            step.operation != OperatorKind::GatedDeltaNet ||
            step.ordinal == covered_step_ordinal)
            continue;
        if (found != kSemanticDispatchUnresolved) return false;
        found = step.ordinal;
    }
    if (found == kSemanticDispatchUnresolved) return false;
    *owner = found;
    (void)model;
    return true;
}

bool second_recurrent_l2_step(const SemanticDispatchProgram& program,
                              uint32_t step_ordinal) {
    const SemanticDispatchLayerView* containing = nullptr;
    for (const SemanticDispatchLayerView& view : program.layer_views) {
        if (view.step_count == 0 || view.first_step == kSemanticDispatchUnresolved)
            continue;
        if (step_ordinal >= view.first_step &&
            step_ordinal - view.first_step < view.step_count) {
            if (containing) return false;
            containing = &view;
        }
    }
    if (!containing) return false;
    std::array<uint32_t, 2> l2_steps = {
        kSemanticDispatchUnresolved, kSemanticDispatchUnresolved};
    size_t count = 0;
    for (uint32_t index = containing->first_step;
         index < containing->first_step + containing->step_count; ++index) {
        if (program.steps[index].kind != SemanticDispatchStepKind::Operator ||
            program.steps[index].operation != OperatorKind::L2Normalize)
            continue;
        if (count >= l2_steps.size()) return false;
        l2_steps[count++] = program.steps[index].ordinal;
    }
    return count == 2 && step_ordinal == l2_steps[1];
}

bool find_moe_activation_owner(const SemanticDispatchProgram& program,
                               uint32_t covered_step_ordinal, uint32_t* owner) {
    const SemanticDispatchLayerView* containing = nullptr;
    for (const SemanticDispatchLayerView& view : program.layer_views) {
        if (view.step_count == 0 || view.first_step == kSemanticDispatchUnresolved)
            continue;
        if (covered_step_ordinal >= view.first_step &&
            covered_step_ordinal - view.first_step < view.step_count) {
            if (containing) return false;
            containing = &view;
        }
    }
    if (!containing) return false;
    uint32_t activation = kSemanticDispatchUnresolved;
    bool has_router = false;
    for (uint32_t index = containing->first_step;
         index < containing->first_step + containing->step_count; ++index) {
        const SemanticDispatchStep& step = program.steps[index];
        if (step.kind != SemanticDispatchStepKind::Operator) continue;
        has_router |= step.operation == OperatorKind::RouterTopK;
        if (step.operation != OperatorKind::GatedActivation) continue;
        if (activation != kSemanticDispatchUnresolved) return false;
        activation = step.ordinal;
    }
    if (!has_router || activation == kSemanticDispatchUnresolved ||
        activation == covered_step_ordinal)
        return false;
    *owner = activation;
    return true;
}

std::optional<CompatibilityReport> add_recipe(
    CompileState& state, const SemanticDispatchStep& step,
    StructuralMetalPrimitive primitive,
    uint32_t order, const std::vector<CertificateFact>& facts,
    StructuralMetalPrimitiveInvocation* invocation) {
    const PrimitiveSpec spec = primitive_spec(primitive);
    const auto* library = state.input.libraries.find(spec.library);
    if (!library)
        return fail(CompatibilityError::CAPABILITY_MISSING,
                    "structural Metal primitive has no library identity",
                    step.operator_id);
    MetalPipelineRecipe recipe;
    recipe.normalized_requirement_digest = recipe_digest(
        step, primitive, order, spec.abi, spec, *library, facts);
    recipe.function_name = std::string(spec.function);
    recipe.function_constants = spec.constants;
    recipe.library_source_digest = library->source_digest;
    recipe.dispatch = spec.dispatch;
    const uint32_t recipe_index = [&]() {
        for (uint32_t index = 0; index < state.recipes.size(); ++index)
            if (state.recipes[index].normalized_requirement_digest ==
                recipe.normalized_requirement_digest)
                return index;
        if (state.recipes.size() >= kMaximumRecipes) return kSemanticDispatchUnresolved;
        state.recipes.push_back(recipe);
        return static_cast<uint32_t>(state.recipes.size() - 1);
    }();
    if (recipe_index >= state.recipes.size() ||
        !(state.recipes[recipe_index] == recipe))
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "structural Metal recipe set exceeds its deterministic bound or has conflicting contracts",
                    step.operator_id);
    invocation->primitive = primitive;
    invocation->order = order;
    invocation->recipe_index = recipe_index;
    return std::nullopt;
}

std::optional<CompatibilityReport> add_primitives(
    CompileState& state, const SemanticModel& model,
    const SemanticDispatchProgram& program, const SemanticDispatchStep& step,
    const SemanticOperator& operation,
    const std::vector<CertificateFact>& facts,
    std::vector<StructuralMetalPrimitiveInvocation>& invocations) {
    const auto add = [&](StructuralMetalPrimitive primitive)
        -> std::optional<CompatibilityReport> {
        StructuralMetalPrimitiveInvocation invocation;
        if (const auto report = add_recipe(state, step, primitive,
                                           static_cast<uint32_t>(invocations.size()),
                                           facts, &invocation))
            return report;
        invocations.push_back(invocation);
        return std::nullopt;
    };
    switch (operation.kind) {
    case OperatorKind::EmbeddingLookup: {
        if (facts.size() != 1) return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                                           "embedding requires one exact certificate",
                                           operation.id);
        const auto embedding = primitive_for_strategy(facts[0].lowering.strategy);
        if (!embedding ||
            (*embedding != StructuralMetalPrimitive::EmbeddingF16 &&
             *embedding != StructuralMetalPrimitive::EmbeddingQ4 &&
             *embedding != StructuralMetalPrimitive::EmbeddingQ6))
            return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                        "embedding certificate is not an executable structural pattern",
                        operation.id);
        if (const auto report = add(*embedding))
            return report;
        break;
    }
    case OperatorKind::RmsNorm: {
        const auto* payload = std::get_if<RmsNormPayload>(&operation.payload);
        if (!payload) return fail(CompatibilityError::IR_REFERENCE_INVALID,
                                  "RMS norm payload is not canonical", operation.id);
        StructuralMetalPrimitive primitive = StructuralMetalPrimitive::RmsNormNoScale;
        if (payload->weight_mode != 0) {
            if (facts.size() != 1)
                return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                            "RMS norm requires one exact affine certificate", operation.id);
            const uint32_t strategy = facts[0].lowering.strategy;
            if (strategy == static_cast<uint32_t>(
                                MetalCodecLoweringStrategy::StructuralTensorF32)) {
                primitive = payload->affine_geometry ==
                                    RmsNormAffineGeometry::SharedAcrossGroups
                                ? StructuralMetalPrimitive::RmsNormRowsF32
                                : StructuralMetalPrimitive::RmsNormF32;
            } else {
                return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                            "RMS norm requires an exact supported certificate", operation.id);
            }
        }
        if (payload->weight_mode == 0 && !facts.empty())
            return fail(CompatibilityError::IR_REFERENCE_INVALID,
                        "unscaled RMS norm must not carry a tensor", operation.id);
        if (const auto report = add(primitive)) return report;
        break;
    }
    case OperatorKind::Linear: {
        const bool batched_linear = admitted_batch_prefill(program) &&
            !is_final_output_operator(model, program, operation);
        const bool router_projection = operation.outputs.size() == 1 &&
            std::any_of(model.operators.begin(), model.operators.end(),
                        [&](const SemanticOperator& candidate) {
                            return candidate.kind == OperatorKind::RouterTopK &&
                                   candidate.inputs.size() == 1 &&
                                   candidate.inputs[0] == operation.outputs[0];
                        });
        if (facts.empty())
            return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                        "linear has no exact supported structural codec", operation.id);
        const auto selected = primitive_for_strategy(facts[0].lowering.strategy);
        const bool column_grouped = has_strategy(
            facts[0], MetalCodecLoweringStrategy::StructuralGemvColumnGroupedU2);
        if (!selected && !column_grouped)
            return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                        "linear has no exact supported structural codec", operation.id);
        if (router_projection &&
            (!selected || *selected != StructuralMetalPrimitive::GemvF32)) {
            return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                        batched_linear
                            ? "batched prefill does not support an F32 router projection"
                            : "router projection requires an exact F32 structural codec",
                        operation.id);
        } else if (!router_projection && selected &&
                   *selected == StructuralMetalPrimitive::GemvF32) {
            return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                        "F32 linear requires a router projection", operation.id);
        }
        if (batched_linear) {
            if (!selected || *selected != StructuralMetalPrimitive::GemvF16)
                return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                            "batched prefill only supports an F16 structural codec",
                            operation.id);
        }
        if (column_grouped) {
            if (batched_linear)
                return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                            "batched prefill does not support column-grouped affine",
                            operation.id);
            if (const auto report = add(StructuralMetalPrimitive::ColumnGroupedSelect))
                return report;
            if (const auto report = add(StructuralMetalPrimitive::ColumnGroupedPartial))
                return report;
            if (const auto report = add(StructuralMetalPrimitive::ColumnGroupedReduce))
                return report;
        } else if (const auto report = add(
                       batched_linear
                           ? (program.request.batch_rows == 2
                                  ? StructuralMetalPrimitive::PrefillF16Rows
                                  : StructuralMetalPrimitive::PrefillF16Tile)
                           : *selected)) {
            return report;
        }
        const auto* payload = std::get_if<LinearPayload>(&operation.payload);
        if (!payload) return fail(CompatibilityError::IR_REFERENCE_INVALID,
                                  "linear payload is not canonical", operation.id);
        if (payload->has_bias) {
            if (facts.size() != 2 ||
                !has_strategy(facts[1],
                              MetalCodecLoweringStrategy::StructuralTensorF32))
                return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                            "linear bias requires an exact F32 certificate", operation.id);
            if (const auto report = add(StructuralMetalPrimitive::VecAdd)) return report;
        }
        break;
    }
    case OperatorKind::SwiGlu:
        if (std::get<SwiGluPayload>(operation.payload).activation != ActivationKind::Silu)
            return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                        "only the exact SiLU gated activation is executable", operation.id);
        if (const auto report = add(StructuralMetalPrimitive::SwiGlu)) return report;
        break;
    case OperatorKind::Add:
        if (const auto report = add(StructuralMetalPrimitive::VecAdd)) return report;
        break;
    case OperatorKind::Rope: {
        const auto* payload = std::get_if<RopePayload>(&operation.payload);
        if (!payload) return fail(CompatibilityError::IR_REFERENCE_INVALID,
                                  "RoPE payload is not canonical", operation.id);
        StructuralMetalPrimitive primitive = StructuralMetalPrimitive::RopeHalfSplit;
        if (payload->pairing == RopePairing::Interleaved)
            primitive = StructuralMetalPrimitive::RopeInterleaved;
        else if (payload->pairing == RopePairing::MultiSectionHalfSplit)
            primitive = StructuralMetalPrimitive::RopeMultiSection;
        else if (payload->pairing != RopePairing::HalfSplit)
            return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                        "RoPE pairing has no exact primitive", operation.id);
        if (const auto report = add(primitive)) return report;
        if (const auto report = add(primitive)) return report;
        break;
    }
    case OperatorKind::CausalAttention: {
        const auto* payload = std::get_if<CausalAttentionPayload>(&operation.payload);
        if (!payload || payload->window != AttentionWindowKind::Global ||
            payload->cache_policy != CachePolicy::Global)
            return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                        "attention window or state policy is not executable", operation.id);
        if (const auto report = add(StructuralMetalPrimitive::KvWrite)) return report;
        if (const auto report = add(StructuralMetalPrimitive::KvWrite)) return report;
        if (const auto report = add(StructuralMetalPrimitive::Attention)) return report;
        break;
    }
    case OperatorKind::GatedAttention:
        if (const auto report = add(StructuralMetalPrimitive::GatedAttention)) return report;
        break;
    case OperatorKind::AxisSplit:
        if (const auto report = add(StructuralMetalPrimitive::AxisSplit)) return report;
        break;
    case OperatorKind::DepthwiseConvSilu:
        if (const auto report = add(StructuralMetalPrimitive::DnetConvSilu)) return report;
        break;
    case OperatorKind::L2Normalize:
        if (const auto report = add(StructuralMetalPrimitive::DnetL2)) return report;
        break;
    case OperatorKind::GatedDeltaNet:
        if (const auto report = add(StructuralMetalPrimitive::DnetUpdate)) return report;
        break;
    case OperatorKind::GatedRmsNorm:
        // The operation is covered by the recurrent update owner. A standalone
        // PSO would execute the fused state equation twice.
        break;
    case OperatorKind::RouterTopK: {
        const auto* payload = std::get_if<RouterTopKPayload>(&operation.payload);
        if (!payload || payload->expert_count == 0 || payload->expert_count > 512 ||
            payload->selected_count == 0 || payload->selected_count > 16)
            return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                        "router shape is outside the exact executable contract", operation.id);
        if (const auto report = add(StructuralMetalPrimitive::RouterTopK)) return report;
        break;
    }
    case OperatorKind::RoutedLinear:
        if (facts.size() != 1 ||
            !has_strategy(facts[0],
                          MetalCodecLoweringStrategy::StructuralGemvQ4))
            return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                        "routed linear requires the exact supported expert codec", operation.id);
        if (const auto report = add(StructuralMetalPrimitive::GemvQ4Expert)) return report;
        break;
    case OperatorKind::GatedActivation:
        if (std::get<GatedActivationPayload>(operation.payload).activation != ActivationKind::Silu)
            return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                        "expert activation has no exact SiLU primitive", operation.id);
        if (const auto report = add(StructuralMetalPrimitive::GatedActivationExperts)) return report;
        break;
    case OperatorKind::WeightedExpertReduce: {
        const auto* payload = std::get_if<WeightedExpertReducePayload>(&operation.payload);
        if (!payload) return fail(CompatibilityError::IR_REFERENCE_INVALID,
                                  "expert reduce payload is not canonical", operation.id);
        if (payload->scale_source == ExpertScaleSource::PerExpertTensor) {
            if (facts.size() != 1 ||
                !has_strategy(facts[0],
                              MetalCodecLoweringStrategy::StructuralTensorF32))
                return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                            "expert scale requires an exact F32 tensor", operation.id);
            if (const auto report = add(StructuralMetalPrimitive::ApplyDownScale)) return report;
        } else if (!facts.empty()) {
            return fail(CompatibilityError::IR_REFERENCE_INVALID,
                        "unscaled expert reduce must not carry a tensor", operation.id);
        }
        if (const auto report = add(StructuralMetalPrimitive::ExpertReduce)) return report;
        break;
    }
    case OperatorKind::Scale: {
        const auto* payload = std::get_if<ScalePayload>(&operation.payload);
        if (!payload) return fail(CompatibilityError::IR_REFERENCE_INVALID,
                                  "scale payload is not canonical", operation.id);
        if (payload->source == ScaleSource::Tensor &&
            (facts.size() != 1 ||
             !has_strategy(facts[0],
                           MetalCodecLoweringStrategy::StructuralTensorF32)))
            return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                        "tensor scale requires an exact F32 tensor", operation.id);
        if (payload->source == ScaleSource::LiteralF32 && !facts.empty())
            return fail(CompatibilityError::IR_REFERENCE_INVALID,
                        "literal scale must not carry a tensor", operation.id);
        if (const auto report = add(
                payload->source == ScaleSource::Tensor
                    ? StructuralMetalPrimitive::VecMul
                    : StructuralMetalPrimitive::VecScale))
            return report;
        break;
    }
    case OperatorKind::Concat:
        return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                    "Concat has no executable structural Metal primitive", operation.id);
    case OperatorKind::TanhSoftcap:
        return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                    "TanhSoftcap has no executable structural Metal primitive", operation.id);
    }
    (void)model;
    return std::nullopt;
}

bool contains_id(std::span<const uint32_t> values, uint32_t value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

bool contains_operation(std::span<const OperatorKind> values, OperatorKind value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

bool operator_inputs_equal(const SemanticOperator& operation,
                           std::initializer_list<uint32_t> expected) {
    return std::equal(operation.inputs.begin(), operation.inputs.end(),
                      expected.begin(), expected.end()) &&
           operation.inputs.size() == expected.size();
}

bool ordered_kinds(std::span<const SemanticOperator*> operations,
                   std::span<const OperatorKind> expected) {
    if (operations.size() != expected.size()) return false;
    for (size_t index = 0; index < expected.size(); ++index)
        if (!operations[index] || operations[index]->kind != expected[index]) return false;
    return true;
}

std::optional<CompatibilityReport> validate_dense_layer_edges(
    std::span<const SemanticOperator*> operations,
    bool gated_attention) {
    const std::array<OperatorKind, 14> base = {
        OperatorKind::RmsNorm, OperatorKind::Linear, OperatorKind::Linear,
        OperatorKind::Linear, OperatorKind::Rope, OperatorKind::CausalAttention,
        OperatorKind::Linear, OperatorKind::Add, OperatorKind::RmsNorm,
        OperatorKind::Linear, OperatorKind::Linear, OperatorKind::SwiGlu,
        OperatorKind::Linear, OperatorKind::Add};
    const std::array<OperatorKind, 16> gated = {
        OperatorKind::RmsNorm, OperatorKind::Linear, OperatorKind::Linear,
        OperatorKind::Linear, OperatorKind::AxisSplit, OperatorKind::Rope,
        OperatorKind::CausalAttention, OperatorKind::GatedAttention,
        OperatorKind::Linear, OperatorKind::Add, OperatorKind::RmsNorm,
        OperatorKind::Linear, OperatorKind::Linear, OperatorKind::SwiGlu,
        OperatorKind::Linear, OperatorKind::Add};
    if (gated_attention ? !ordered_kinds(operations, gated)
                        : !ordered_kinds(operations, base))
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "dense layer operation order is not an executable structural DAG");
    const size_t rope = gated_attention ? 5 : 4;
    const size_t attention = gated_attention ? 6 : 5;
    const size_t projection = gated_attention ? 8 : 6;
    const size_t residual = gated_attention ? 9 : 7;
    const size_t ffn_norm = gated_attention ? 10 : 8;
    const size_t gate = gated_attention ? 11 : 9;
    const size_t up = gated_attention ? 12 : 10;
    const size_t activation = gated_attention ? 13 : 11;
    const size_t down = gated_attention ? 14 : 12;
    const size_t final_add = gated_attention ? 15 : 13;
    const SemanticOperator& norm = *operations[0];
    const SemanticOperator& query = *operations[1];
    const SemanticOperator& key = *operations[2];
    const SemanticOperator& value = *operations[3];
    const SemanticOperator& rope_op = *operations[rope];
    const SemanticOperator& attention_op = *operations[attention];
    const SemanticOperator& projection_op = *operations[projection];
    const SemanticOperator& residual_op = *operations[residual];
    const SemanticOperator& ffn_norm_op = *operations[ffn_norm];
    const SemanticOperator& gate_op = *operations[gate];
    const SemanticOperator& up_op = *operations[up];
    const SemanticOperator& activation_op = *operations[activation];
    const SemanticOperator& down_op = *operations[down];
    const SemanticOperator& final_add_op = *operations[final_add];
    if (norm.outputs.size() != 1 || query.outputs.size() != 1 ||
        key.outputs.size() != 1 || value.outputs.size() != 1 ||
        !operator_inputs_equal(query, {norm.outputs[0]}) ||
        !operator_inputs_equal(key, {norm.outputs[0]}) ||
        !operator_inputs_equal(value, {norm.outputs[0]}))
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "dense QKV producers do not share the normalized input");
    uint32_t query_input = query.outputs[0];
    if (gated_attention) {
        const SemanticOperator& split = *operations[4];
        if (split.outputs.size() != 2 ||
            !operator_inputs_equal(split, {query.outputs[0]}))
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "gated attention split is not fed by the query projection");
        query_input = split.outputs[0];
        if (!operator_inputs_equal(rope_op, {query_input, key.outputs[0]}))
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "gated attention RoPE inputs are not structurally bound");
    } else if (!operator_inputs_equal(rope_op, {query_input, key.outputs[0]})) {
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "RoPE inputs are not structurally bound to Q and K");
    }
    if (rope_op.outputs.size() != 2 || attention_op.inputs.size() != 3 ||
        attention_op.inputs[0] != rope_op.outputs[0] ||
        attention_op.inputs[1] != rope_op.outputs[1] ||
        attention_op.inputs[2] != value.outputs[0] || attention_op.outputs.size() != 1 ||
        attention_op.states.size() != 2)
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "attention is not structurally bound to RoPE, V, and KV state");
    uint32_t context = attention_op.outputs[0];
    if (gated_attention) {
        const SemanticOperator& gated_op = *operations[7];
        const SemanticOperator& split = *operations[4];
        if (!operator_inputs_equal(gated_op, {context, split.outputs[1]}) ||
            gated_op.outputs.size() != 1)
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "gated attention does not consume context and its gate");
        context = gated_op.outputs[0];
    }
    if (!operator_inputs_equal(projection_op, {context}) || projection_op.outputs.size() != 1 ||
        residual_op.inputs.size() != 2 || residual_op.inputs[1] != projection_op.outputs[0] ||
        residual_op.inputs[0] != norm.inputs[0] || residual_op.outputs.size() != 1 ||
        !operator_inputs_equal(ffn_norm_op, {residual_op.outputs[0]}) ||
        !operator_inputs_equal(gate_op, {ffn_norm_op.outputs[0]}) ||
        !operator_inputs_equal(up_op, {ffn_norm_op.outputs[0]}) ||
        !operator_inputs_equal(activation_op, {gate_op.outputs[0], up_op.outputs[0]}) ||
        !operator_inputs_equal(down_op, {activation_op.outputs[0]}) ||
        final_add_op.inputs.size() != 2 || final_add_op.inputs[0] != residual_op.outputs[0] ||
        final_add_op.inputs[1] != down_op.outputs[0] || final_add_op.outputs.size() != 1)
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "dense FFN or residual edges do not match the executor contract");
    return std::nullopt;
}

std::optional<CompatibilityReport> validate_recurrent_layer_edges(
    std::span<const SemanticOperator*> operations,
    const SemanticModel& model) {
    const std::array<OperatorKind, 18> expected = {
        OperatorKind::RmsNorm, OperatorKind::Linear, OperatorKind::Linear,
        OperatorKind::Linear, OperatorKind::Linear, OperatorKind::DepthwiseConvSilu,
        OperatorKind::L2Normalize, OperatorKind::L2Normalize,
        OperatorKind::GatedDeltaNet, OperatorKind::GatedRmsNorm, OperatorKind::Linear,
        OperatorKind::Add, OperatorKind::RmsNorm, OperatorKind::Linear,
        OperatorKind::Linear, OperatorKind::SwiGlu, OperatorKind::Linear,
        OperatorKind::Add};
    if (!ordered_kinds(operations, expected))
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "recurrent layer operation order is not an executable structural DAG");
    const SemanticOperator& norm = *operations[0];
    if (norm.outputs.size() != 1 || norm.inputs.empty())
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "recurrent norm does not expose its normalized value");
    for (size_t index = 1; index <= 4; ++index)
        if (!operator_inputs_equal(*operations[index], {norm.outputs[0]}))
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "recurrent projections do not share the normalized input");
    const SemanticOperator& conv = *operations[5];
    if (!operator_inputs_equal(conv, {operations[1]->outputs[0]}) || conv.outputs.size() != 3 ||
        conv.states.size() != 1)
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "recurrent convolution is not bound to QKV and history state");
    if (!operator_inputs_equal(*operations[6], {conv.outputs[0]}) ||
        !operator_inputs_equal(*operations[7], {conv.outputs[1]}) ||
        !operator_inputs_equal(*operations[8], {operations[6]->outputs[0],
                                                 operations[7]->outputs[0], conv.outputs[2],
                                                 operations[3]->outputs[0], operations[4]->outputs[0]}) ||
        operations[8]->states.size() != 1 || operations[8]->outputs.size() != 1 ||
        !operator_inputs_equal(*operations[9], {operations[8]->outputs[0], operations[2]->outputs[0]}) ||
        operations[9]->outputs.size() != 1 || !operator_inputs_equal(*operations[10], {operations[9]->outputs[0]}) ||
        operations[11]->inputs.size() != 2 || operations[11]->inputs[0] != norm.inputs[0] ||
        operations[11]->inputs[1] != operations[10]->outputs[0] || operations[11]->outputs.size() != 1 ||
        !operator_inputs_equal(*operations[12], {operations[11]->outputs[0]}) ||
        !operator_inputs_equal(*operations[13], {operations[12]->outputs[0]}) ||
        !operator_inputs_equal(*operations[14], {operations[12]->outputs[0]}) ||
        !operator_inputs_equal(*operations[15], {operations[13]->outputs[0], operations[14]->outputs[0]}) ||
        !operator_inputs_equal(*operations[16], {operations[15]->outputs[0]}) ||
        operations[17]->inputs.size() != 2 || operations[17]->inputs[0] != operations[11]->outputs[0] ||
        operations[17]->inputs[1] != operations[16]->outputs[0] || operations[17]->outputs.size() != 1)
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "recurrent state, gated norm, or residual edges are not executable");
    if (conv.states[0] >= model.states.size() || operations[8]->states[0] >= model.states.size() ||
        conv.states[0] == operations[8]->states[0] ||
        model.states[conv.states[0]].kind != StateKind::RecurrentConvHistory ||
        model.states[operations[8]->states[0]].kind != StateKind::RecurrentDeltaMatrix)
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "recurrent state ownership is not a typed history/update pair");
    return std::nullopt;
}

// Match the complete MoE execution DAG by typed value edges.  Serialized
// operator order is not semantic identity; this matcher accepts any valid
// topological declaration while rejecting the count-compatible serial
// residual rewire that the composite executor cannot bind.
std::optional<CompatibilityReport> validate_moe_layer_edges(
    std::span<const SemanticOperator*> operations) {
    const auto count = [&](OperatorKind kind) {
        return static_cast<size_t>(std::count_if(
            operations.begin(), operations.end(), [kind](const SemanticOperator* operation) {
                return operation && operation->kind == kind;
            }));
    };
    if (operations.size() != 26 || count(OperatorKind::RmsNorm) != 4 ||
        count(OperatorKind::Linear) != 8 || count(OperatorKind::Add) != 3 ||
        count(OperatorKind::Scale) != 2 || count(OperatorKind::SwiGlu) != 1 ||
        count(OperatorKind::Rope) != 1 || count(OperatorKind::CausalAttention) != 1 ||
        count(OperatorKind::RouterTopK) != 1 || count(OperatorKind::RoutedLinear) != 2 ||
        count(OperatorKind::AxisSplit) != 1 || count(OperatorKind::GatedActivation) != 1 ||
        count(OperatorKind::WeightedExpertReduce) != 1)
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "MoE layer does not have the exact executable operator cardinality");

    const auto unique = [&](OperatorKind kind, auto predicate) -> const SemanticOperator* {
        const SemanticOperator* found = nullptr;
        for (const SemanticOperator* operation : operations) {
            if (!operation || operation->kind != kind || !predicate(*operation)) continue;
            if (found) return nullptr;
            found = operation;
        }
        return found;
    };
    const SemanticOperator* attention = unique(
        OperatorKind::CausalAttention, [](const SemanticOperator& operation) {
            return operation.inputs.size() == 3 && operation.outputs.size() == 1 &&
                   operation.states.size() == 2;
        });
    const SemanticOperator* rope = unique(
        OperatorKind::Rope, [](const SemanticOperator& operation) {
            return operation.inputs.size() == 2 && operation.outputs.size() == 2;
        });
    if (!attention || !rope || attention->inputs[0] != rope->outputs[0] ||
        attention->inputs[1] != rope->outputs[1])
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "MoE attention root is not structurally bound to one RoPE pair");

    const SemanticOperator* attention_norm = unique(
        OperatorKind::RmsNorm, [&](const SemanticOperator& operation) {
            if (operation.inputs.size() != 1 || operation.outputs.size() != 1) return false;
            size_t users = 0;
            for (const SemanticOperator* candidate : operations)
                if (candidate && candidate->kind == OperatorKind::Linear &&
                    operator_inputs_equal(*candidate, {operation.outputs[0]})) ++users;
            return users == 3;
        });
    if (!attention_norm) return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                                     "MoE attention norm does not have exactly three QKV users");
    std::vector<const SemanticOperator*> qkv;
    for (const SemanticOperator* operation : operations)
        if (operation && operation->kind == OperatorKind::Linear &&
            operator_inputs_equal(*operation, {attention_norm->outputs[0]}))
            qkv.push_back(operation);
    if (qkv.size() != 3 || !operator_inputs_equal(*rope, {qkv[0]->outputs[0], qkv[1]->outputs[0]}) ||
        attention->inputs[2] != qkv[2]->outputs[0])
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "MoE QKV and RoPE edges are not structurally complete");
    const SemanticOperator* projection = unique(
        OperatorKind::Linear, [&](const SemanticOperator& operation) {
            return operation.outputs.size() == 1 &&
                   operator_inputs_equal(operation, {attention->outputs[0]});
        });
    if (!projection) return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                                 "MoE attention output projection is ambiguous");
    const SemanticOperator* attention_residual = unique(
        OperatorKind::Add, [&](const SemanticOperator& operation) {
            return operation.outputs.size() == 1 &&
                   operator_inputs_equal(operation, {attention_norm->inputs[0], projection->outputs[0]});
        });
    if (!attention_residual) return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                                         "MoE attention residual edge is not exact");
    const uint32_t residual = attention_residual->outputs[0];

    const SemanticOperator* dense_norm = unique(
        OperatorKind::RmsNorm, [&](const SemanticOperator& operation) {
            if (!operator_inputs_equal(operation, {residual}) || operation.outputs.size() != 1) return false;
            size_t users = 0;
            for (const SemanticOperator* candidate : operations)
                if (candidate && candidate->kind == OperatorKind::Linear &&
                    operator_inputs_equal(*candidate, {operation.outputs[0]})) ++users;
            return users == 2;
        });
    if (!dense_norm) return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                                 "MoE dense branch norm does not have exactly two users");
    std::vector<const SemanticOperator*> dense_gate_up;
    for (const SemanticOperator* operation : operations)
        if (operation && operation->kind == OperatorKind::Linear &&
            operator_inputs_equal(*operation, {dense_norm->outputs[0]}))
            dense_gate_up.push_back(operation);
    const SemanticOperator* swiglu = unique(
        OperatorKind::SwiGlu, [&](const SemanticOperator& operation) {
            return dense_gate_up.size() == 2 && operation.outputs.size() == 1 &&
                   operator_inputs_equal(operation, {dense_gate_up[0]->outputs[0], dense_gate_up[1]->outputs[0]});
        });
    const SemanticOperator* dense_down = unique(
        OperatorKind::Linear, [&](const SemanticOperator& operation) {
            return swiglu && operation.outputs.size() == 1 &&
                   operator_inputs_equal(operation, {swiglu->outputs[0]});
        });
    if (dense_gate_up.size() != 2 || !swiglu || !dense_down)
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "MoE dense branch gate/up/activation/down edges are not exact");

    const SemanticOperator* router = unique(
        OperatorKind::RouterTopK, [](const SemanticOperator& operation) {
            return operation.inputs.size() == 1 && operation.outputs.size() == 2;
        });
    const SemanticOperator* route_scale = unique(
        OperatorKind::Scale, [](const SemanticOperator& operation) {
            const auto* payload = std::get_if<ScalePayload>(&operation.payload);
            return payload && payload->source == ScaleSource::Tensor &&
                   operation.inputs.size() == 1 && operation.outputs.size() == 1;
        });
    const SemanticOperator* router_norm = unique(
        OperatorKind::RmsNorm, [&](const SemanticOperator& operation) {
            return route_scale && operation.outputs.size() == 1 &&
                   operator_inputs_equal(operation, {residual}) &&
                   operator_inputs_equal(*route_scale, {operation.outputs[0]});
        });
    const SemanticOperator* normalization_scale = unique(
        OperatorKind::Scale, [&](const SemanticOperator& operation) {
            const auto* payload = std::get_if<ScalePayload>(&operation.payload);
            return payload && payload->source == ScaleSource::LiteralF32 && route_scale &&
                   operation.inputs.size() == 1 && operation.outputs.size() == 1 &&
                   operator_inputs_equal(operation, {route_scale->outputs[0]});
        });
    const SemanticOperator* router_linear = unique(
        OperatorKind::Linear, [&](const SemanticOperator& operation) {
            return router && operation.inputs.size() == 1 && operation.outputs.size() == 1 &&
                   operation.outputs[0] == router->inputs[0];
        });
    if (!router || !route_scale || !router_norm || !normalization_scale || !router_linear ||
        normalization_scale->outputs.size() != 1 || router_linear->inputs.size() != 1 ||
        normalization_scale->outputs[0] != router_linear->inputs[0])
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "MoE router normalization and projection chain is not exact");

    const SemanticOperator* expert_norm = unique(
        OperatorKind::RmsNorm, [&](const SemanticOperator& operation) {
            if (operation.outputs.size() != 1 || !operator_inputs_equal(operation, {residual})) return false;
            size_t routed_users = 0;
            for (const SemanticOperator* candidate : operations)
                if (candidate && candidate->kind == OperatorKind::RoutedLinear &&
                    candidate->inputs.size() == 3 && candidate->inputs[0] == operation.outputs[0] &&
                    candidate->inputs[1] == router->outputs[0] && candidate->inputs[2] == router->outputs[1])
                    ++routed_users;
            return routed_users == 1;
        });
    const SemanticOperator* expert_up = unique(
        OperatorKind::RoutedLinear, [&](const SemanticOperator& operation) {
            return expert_norm && operation.inputs.size() == 3 && operation.outputs.size() == 1 &&
                   operation.inputs[0] == expert_norm->outputs[0] &&
                   operation.inputs[1] == router->outputs[0] &&
                   operation.inputs[2] == router->outputs[1];
        });
    const SemanticOperator* split = unique(
        OperatorKind::AxisSplit, [&](const SemanticOperator& operation) {
            return expert_up && operation.inputs.size() == 1 && operation.outputs.size() == 2 &&
                   operator_inputs_equal(operation, {expert_up->outputs[0]});
        });
    const SemanticOperator* activation = unique(
        OperatorKind::GatedActivation, [&](const SemanticOperator& operation) {
            return split && operation.outputs.size() == 1 &&
                   operator_inputs_equal(operation, {split->outputs[0], split->outputs[1]});
        });
    const SemanticOperator* expert_down = unique(
        OperatorKind::RoutedLinear, [&](const SemanticOperator& operation) {
            return activation && operation.inputs.size() == 3 && operation.outputs.size() == 1 &&
                   operation.inputs[0] == activation->outputs[0] &&
                   operation.inputs[1] == router->outputs[0] && operation.inputs[2] == router->outputs[1];
        });
    const SemanticOperator* reduce = unique(
        OperatorKind::WeightedExpertReduce, [&](const SemanticOperator& operation) {
            return expert_down && operation.inputs.size() == 3 && operation.outputs.size() == 1 &&
                   operation.inputs[0] == expert_down->outputs[0] &&
                   operation.inputs[1] == router->outputs[0] && operation.inputs[2] == router->outputs[1];
        });
    const SemanticOperator* branch_merge = unique(
        OperatorKind::Add, [&](const SemanticOperator& operation) {
            return reduce && operation.outputs.size() == 1 &&
                   operator_inputs_equal(operation, {dense_down->outputs[0], reduce->outputs[0]});
        });
    const SemanticOperator* final_add = unique(
        OperatorKind::Add, [&](const SemanticOperator& operation) {
            return branch_merge && operation.outputs.size() == 1 &&
                   operator_inputs_equal(operation, {residual, branch_merge->outputs[0]});
        });
    if (!expert_up || !expert_norm || !split || !activation || !expert_down || !reduce ||
        !branch_merge || !final_add)
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "MoE routed branch, merge, or final residual edges are not exact");

    std::vector<const SemanticOperator*> covered = {
        attention_norm, qkv[0], qkv[1], qkv[2], rope, attention, projection,
        attention_residual, dense_norm, dense_gate_up[0], dense_gate_up[1], swiglu,
        dense_down, branch_merge, router_norm, route_scale, normalization_scale,
        router_linear, router, expert_norm, expert_up, split, activation,
        expert_down, reduce, final_add};
    if (covered.size() != operations.size())
        return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                    "MoE layer contains an uncovered operator");
    for (size_t index = 0; index != covered.size(); ++index) {
        if (!covered[index] || std::find(covered.begin(), covered.begin() + index,
                                         covered[index]) != covered.begin() + index)
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "MoE layer contains a duplicated operator");
    }
    for (const SemanticOperator* operation : operations)
        if (std::find(covered.begin(), covered.end(), operation) == covered.end())
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "MoE layer contains an uncovered operator");
    return std::nullopt;
}

std::optional<CompatibilityReport> derive_layer_shape(
    const SemanticModel& model, const SemanticDispatchProgram& program,
    const SemanticDispatchLayerView& view,
    StructuralMetalExecutionShape* shape) {
    if (!shape || view.step_count == 0 ||
        view.first_step == kSemanticDispatchUnresolved ||
        view.first_step >= program.steps.size() ||
        view.step_count > program.steps.size() - view.first_step)
        return fail(CompatibilityError::IR_REFERENCE_INVALID,
                    "layer execution shape has an invalid semantic range");

    std::vector<OperatorKind> operations;
    std::vector<const SemanticOperator*> operation_views;
    operations.reserve(view.step_count);
    operation_views.reserve(view.step_count);
    for (uint32_t offset = 0; offset < view.step_count; ++offset) {
        const SemanticDispatchStep& step = program.steps[view.first_step + offset];
        if (step.kind != SemanticDispatchStepKind::Operator)
            return fail(CompatibilityError::IR_REFERENCE_INVALID,
                        "layer execution shape contains a non-operator step",
                        step.operator_id);
        const SemanticOperator* operation = operator_for(model, step.operator_id);
        if (!operation || operation->kind != step.operation)
            return fail(CompatibilityError::IR_REFERENCE_INVALID,
                        "layer execution shape cannot resolve its typed operator",
                        step.operator_id);
        operations.push_back(operation->kind);
        operation_views.push_back(operation);
    }

    const auto count = [&](OperatorKind kind) {
        return static_cast<size_t>(std::count(operations.begin(), operations.end(), kind));
    };
    const auto first = [&](OperatorKind kind) {
        const auto it = std::find(operations.begin(), operations.end(), kind);
        return it == operations.end() ? operations.size()
                                      : static_cast<size_t>(it - operations.begin());
    };
    const std::array<OperatorKind, 13> dense_allowed = {
        OperatorKind::RmsNorm, OperatorKind::Linear, OperatorKind::Rope,
        OperatorKind::CausalAttention, OperatorKind::GatedAttention,
        OperatorKind::Add, OperatorKind::SwiGlu, OperatorKind::AxisSplit,
        OperatorKind::Scale, OperatorKind::GatedRmsNorm, OperatorKind::L2Normalize,
        OperatorKind::DepthwiseConvSilu, OperatorKind::GatedDeltaNet};
    const bool has_router = count(OperatorKind::RouterTopK) != 0 ||
                            count(OperatorKind::RoutedLinear) != 0 ||
                            count(OperatorKind::WeightedExpertReduce) != 0 ||
                            count(OperatorKind::GatedActivation) != 0;
    const bool has_recurrent = count(OperatorKind::GatedDeltaNet) != 0 ||
                               count(OperatorKind::DepthwiseConvSilu) != 0;
    const bool has_attention = count(OperatorKind::CausalAttention) != 0 ||
                               count(OperatorKind::GatedAttention) != 0;
    if (has_router) {
        if (has_recurrent || count(OperatorKind::RouterTopK) != 1 ||
            count(OperatorKind::RoutedLinear) < 2 ||
            count(OperatorKind::GatedActivation) != 1 ||
            count(OperatorKind::WeightedExpertReduce) != 1 ||
            count(OperatorKind::AxisSplit) != 1 || operations.front() != OperatorKind::RmsNorm ||
            first(OperatorKind::RouterTopK) >= first(OperatorKind::RoutedLinear) ||
            first(OperatorKind::AxisSplit) <= first(OperatorKind::RouterTopK) ||
            first(OperatorKind::GatedActivation) <= first(OperatorKind::AxisSplit) ||
            first(OperatorKind::WeightedExpertReduce) <= first(OperatorKind::GatedActivation))
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "MoE layer does not have one exact router/branch/reduce shape");
        *shape = StructuralMetalExecutionShape::MoeRoutedFeedForward;
        return validate_moe_layer_edges(operation_views);
    }
    if (has_recurrent) {
        if (has_attention || count(OperatorKind::DepthwiseConvSilu) != 1 ||
            count(OperatorKind::GatedDeltaNet) != 1 ||
            count(OperatorKind::GatedRmsNorm) != 1 ||
            count(OperatorKind::L2Normalize) != 2 || operations.front() != OperatorKind::RmsNorm ||
            first(OperatorKind::DepthwiseConvSilu) <= first(OperatorKind::Linear) ||
            first(OperatorKind::L2Normalize) <= first(OperatorKind::DepthwiseConvSilu) ||
            first(OperatorKind::GatedDeltaNet) <= first(OperatorKind::L2Normalize) ||
            first(OperatorKind::GatedRmsNorm) <= first(OperatorKind::GatedDeltaNet))
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "recurrent layer does not have one exact DeltaNet shape");
        *shape = StructuralMetalExecutionShape::RecurrentDelta;
        return validate_recurrent_layer_edges(operation_views, model);
    }
    if (has_attention) {
        if (count(OperatorKind::CausalAttention) != 1 ||
            count(OperatorKind::Rope) != 1 || count(OperatorKind::Linear) < 5 ||
            operations.front() != OperatorKind::RmsNorm ||
            first(OperatorKind::Rope) <= first(OperatorKind::RmsNorm) ||
            first(OperatorKind::CausalAttention) <= first(OperatorKind::Rope))
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "attention layer does not have one exact attention shape");
        for (OperatorKind operation : operations)
            if (!contains_operation(dense_allowed, operation))
                return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                            "attention layer contains an unsupported semantic operator");
        *shape = StructuralMetalExecutionShape::DenseAttention;
        return validate_dense_layer_edges(
            operation_views, count(OperatorKind::GatedAttention) != 0);
    }
    return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                "layer has no exact structural executor shape");
}

std::optional<CompatibilityReport> validate_group_primitives(
    const CompileState& state, StructuralMetalBundleGroupKind kind,
    StructuralMetalExecutionShape shape,
    std::span<const StructuralMetalPrimitiveInvocation> primitives) {
    if (primitives.empty())
        return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                    "execution group has no primitive invocation");
    for (size_t index = 0; index < primitives.size(); ++index) {
        const auto& invocation = primitives[index];
        if (invocation.order != index || invocation.recipe_index >= state.recipes.size())
            return fail(CompatibilityError::IR_REFERENCE_INVALID,
                        "execution group primitive order or recipe binding is invalid");
        const PrimitiveSpec expected = primitive_spec(invocation.primitive);
        const MetalPipelineRecipe& recipe = state.recipes[invocation.recipe_index];
        const auto* library = state.input.libraries.find(expected.library);
        if (!library || recipe.function_name != expected.function ||
            recipe.function_constants != expected.constants ||
            recipe.dispatch != expected.dispatch ||
            recipe.library_source_digest != library->source_digest ||
            zero_digest(recipe.normalized_requirement_digest))
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "execution group primitive has an incomplete ABI/library binding");
    }
    if (kind == StructuralMetalBundleGroupKind::Graph) {
        if (shape != StructuralMetalExecutionShape::GraphEmbedding || primitives.size() != 1 ||
            (primitives[0].primitive != StructuralMetalPrimitive::EmbeddingF16 &&
             primitives[0].primitive != StructuralMetalPrimitive::EmbeddingQ4 &&
             primitives[0].primitive != StructuralMetalPrimitive::EmbeddingQ6))
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "graph group does not have one exact embedding ABI");
    } else if (kind == StructuralMetalBundleGroupKind::Sampler) {
        if (shape != StructuralMetalExecutionShape::GreedySampler || primitives.size() != 1 ||
            primitives[0].primitive != StructuralMetalPrimitive::SamplerGreedy)
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "sampler group does not have one exact sampler ABI");
    } else if (kind == StructuralMetalBundleGroupKind::Final) {
        if (shape != StructuralMetalExecutionShape::FinalOutput)
            return fail(CompatibilityError::IR_REFERENCE_INVALID,
                        "final group has the wrong execution shape");
    } else if (kind == StructuralMetalBundleGroupKind::Layer) {
        const auto has = [&](StructuralMetalPrimitive primitive) {
            return std::any_of(primitives.begin(), primitives.end(),
                               [primitive](const auto& value) {
                                   return value.primitive == primitive;
                               });
        };
        const auto first_index = [&](StructuralMetalPrimitive primitive) {
            for (size_t index = 0; index < primitives.size(); ++index)
                if (primitives[index].primitive == primitive) return index;
            return primitives.size();
        };
        if (shape == StructuralMetalExecutionShape::DenseAttention &&
            (!has(StructuralMetalPrimitive::KvWrite) ||
             !has(StructuralMetalPrimitive::Attention) ||
             (!has(StructuralMetalPrimitive::RopeHalfSplit) &&
              !has(StructuralMetalPrimitive::RopeInterleaved) &&
              !has(StructuralMetalPrimitive::RopeMultiSection))))
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "dense layer group is missing its ordered attention ABI");
        if (shape == StructuralMetalExecutionShape::DenseAttention &&
            (first_index(StructuralMetalPrimitive::RopeHalfSplit) >=
                 first_index(StructuralMetalPrimitive::KvWrite) &&
             first_index(StructuralMetalPrimitive::RopeInterleaved) >=
                 first_index(StructuralMetalPrimitive::KvWrite) &&
             first_index(StructuralMetalPrimitive::RopeMultiSection) >=
                 first_index(StructuralMetalPrimitive::KvWrite)))
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "dense layer group has reordered RoPE and KV invocations");
        if (shape == StructuralMetalExecutionShape::DenseAttention &&
            first_index(StructuralMetalPrimitive::KvWrite) >=
                first_index(StructuralMetalPrimitive::Attention))
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "dense layer group has reordered KV and attention invocations");
        if (shape == StructuralMetalExecutionShape::RecurrentDelta &&
            (!has(StructuralMetalPrimitive::DnetConvSilu) ||
             !has(StructuralMetalPrimitive::DnetL2) ||
             !has(StructuralMetalPrimitive::DnetUpdate)))
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "recurrent layer group is missing its ordered state-update ABI");
        if (shape == StructuralMetalExecutionShape::RecurrentDelta &&
            (first_index(StructuralMetalPrimitive::DnetConvSilu) >=
                 first_index(StructuralMetalPrimitive::DnetL2) ||
             first_index(StructuralMetalPrimitive::DnetL2) >=
                 first_index(StructuralMetalPrimitive::DnetUpdate)))
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "recurrent layer group has reordered state-update invocations");
        if (shape == StructuralMetalExecutionShape::MoeRoutedFeedForward &&
            (!has(StructuralMetalPrimitive::RouterTopK) ||
             !has(StructuralMetalPrimitive::GatedActivationExperts) ||
             !has(StructuralMetalPrimitive::ExpertReduce)))
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "MoE layer group is missing its ordered branch ABI");
        if (shape == StructuralMetalExecutionShape::MoeRoutedFeedForward &&
            (first_index(StructuralMetalPrimitive::RouterTopK) >=
                 first_index(StructuralMetalPrimitive::GatedActivationExperts) ||
             first_index(StructuralMetalPrimitive::GatedActivationExperts) >=
                 first_index(StructuralMetalPrimitive::ExpertReduce)))
            return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                        "MoE layer group has reordered router, activation, or reduce invocations");
    }
    return std::nullopt;
}

std::optional<CompatibilityReport> validate_final_group(
    const SemanticModel& model, const SemanticDispatchProgram& program,
    uint32_t first, uint32_t count) {
    if (count != 2 || first == 0 || first >= program.steps.size() ||
        first + count > program.steps.size())
        return fail(CompatibilityError::IR_REFERENCE_INVALID,
                    "final execution group is not the exact output chain");
    const SemanticDispatchStep& norm_step = program.steps[first];
    const SemanticDispatchStep& output_step = program.steps[first + 1];
    const SemanticOperator* norm = operator_for(model, norm_step.operator_id);
    const SemanticOperator* output = operator_for(model, output_step.operator_id);
    if (!norm || !output || norm->kind != OperatorKind::RmsNorm ||
        output->kind != OperatorKind::Linear || norm->outputs.size() != 1 ||
        output->outputs.size() != 1 || norm->inputs != program.steps[first - 1].output_values ||
        !operator_inputs_equal(*output, {norm->outputs[0]}) ||
        model.output_values_count != 1 || output->outputs[0] != model.output_values_first ||
        !contains_id(program.terminal_operator_ids, output->id))
        return fail(CompatibilityError::IR_REFERENCE_INVALID,
                    "final execution group is not a bound norm-to-logits chain",
                    output ? output->id : kSemanticDispatchUnresolved);
    return std::nullopt;
}

std::optional<CompatibilityReport> build_execution_groups(
    const CompileState& state, const SemanticModel& model,
    const SemanticDispatchProgram& program,
    const std::vector<StructuralMetalBundleStep>& steps,
    std::vector<StructuralMetalBundleGroup>* output) {
    if (steps.size() != program.steps.size())
        return fail(CompatibilityError::IR_REFERENCE_INVALID,
                    "structural execution groups lost semantic step alignment");

    std::vector<const SemanticDispatchLayerView*> active_views;
    std::vector<StructuralMetalExecutionShape> active_shapes;
    std::vector<uint8_t> covered(steps.size(), 0);
    std::vector<uint8_t> assigned(steps.size(), 0);
    uint32_t prior_layer = kSemanticDispatchUnresolved;
    uint32_t prior_first = kSemanticDispatchUnresolved;
    for (const SemanticDispatchLayerView& view : program.layer_views) {
        if (view.step_count == 0) {
            if (view.first_step != kSemanticDispatchUnresolved)
                return fail(CompatibilityError::IR_REFERENCE_INVALID,
                            "inactive layer view has an executable step range");
            continue;
        }
        if (view.first_step == kSemanticDispatchUnresolved ||
            view.first_step >= steps.size() ||
            view.step_count > steps.size() - view.first_step ||
            (prior_layer != kSemanticDispatchUnresolved &&
             view.layer_index <= prior_layer) ||
            (prior_first != kSemanticDispatchUnresolved &&
             view.first_step <= prior_first))
            return fail(CompatibilityError::IR_REFERENCE_INVALID,
                        "layer views are not ordered executable ranges");
        for (uint32_t offset = 0; offset < view.step_count; ++offset) {
            const size_t index = view.first_step + offset;
            if (covered[index] || program.steps[index].kind != SemanticDispatchStepKind::Operator)
                return fail(CompatibilityError::IR_REFERENCE_INVALID,
                            "layer view overlaps or contains a non-operator step");
            covered[index] = 1;
        }
        active_views.push_back(&view);
        StructuralMetalExecutionShape shape = StructuralMetalExecutionShape::DenseAttention;
        if (const auto report = derive_layer_shape(model, program, view, &shape))
            return report;
        active_shapes.push_back(shape);
        prior_layer = view.layer_index;
        prior_first = view.first_step;
    }

    output->clear();
    output->reserve(active_views.size() + 4);
    bool saw_final = false;
    bool saw_sampler = false;
    bool saw_embedding = false;
    size_t active_index = 0;
    uint32_t index = 0;
    const auto append_group = [&](StructuralMetalBundleGroupKind kind,
                                  StructuralMetalExecutionShape shape,
                                  uint32_t first, uint32_t count)
        -> std::optional<CompatibilityReport> {
        if (count == 0 || first >= steps.size() || count > steps.size() - first)
            return fail(CompatibilityError::IR_REFERENCE_INVALID,
                        "structural execution group has an invalid step range");
        std::vector<uint32_t> step_ordinals;
        std::vector<uint32_t> operator_ids;
        std::vector<StructuralMetalPrimitiveInvocation> primitives;
        std::vector<StructuralMetalSemanticCoverage> coverage_records;
        std::vector<SemanticDispatchStateEffect> state_effects;
        step_ordinals.reserve(count);
        for (uint32_t offset = 0; offset < count; ++offset) {
            const StructuralMetalBundleStep& step = steps[first + offset];
            const SemanticDispatchStep& semantic = program.steps[first + offset];
            const ExecutionPhase expected_phase =
                kind == StructuralMetalBundleGroupKind::Sampler
                    ? ExecutionPhase::Output : program.request.phase;
            if (step.ordinal() != first + offset || semantic.ordinal != first + offset ||
                step.phase() != expected_phase ||
                step.batch_rows() != program.request.batch_rows ||
                step.coverage().size() != 1 ||
                (kind == StructuralMetalBundleGroupKind::Sampler) !=
                    (semantic.kind == SemanticDispatchStepKind::GreedySampler))
                return fail(CompatibilityError::IR_REFERENCE_INVALID,
                            "structural execution group has incomplete step coverage",
                            semantic.operator_id);
            if (kind == StructuralMetalBundleGroupKind::Sampler &&
                (step.primitives().size() != 1 ||
                 step.primitives()[0].primitive != StructuralMetalPrimitive::SamplerGreedy ||
                 !step.covered_operator_ids().empty() ||
                 !semantic.sampler_binding.has_value() ||
                 semantic.input_values.size() != 1 ||
                 semantic.input_values[0] != model.output_values_first ||
                 semantic.sampler_binding->logits_value_id != model.output_values_first ||
                 semantic.sampler_binding->vocabulary_size != model.vocabulary_size))
                return fail(CompatibilityError::IR_REFERENCE_INVALID,
                            "sampler group is not an exact post-graph boundary");
            if (kind != StructuralMetalBundleGroupKind::Sampler &&
                semantic.kind != SemanticDispatchStepKind::Operator)
                return fail(CompatibilityError::IR_REFERENCE_INVALID,
                            "non-sampler group contains a post-graph step");
            if (semantic.kind == SemanticDispatchStepKind::Operator) {
                if (semantic.covered_operator_ids.size() != 1 ||
                    semantic.covered_operator_ids[0] != semantic.operator_id ||
                    contains_id(operator_ids, semantic.operator_id))
                    return fail(CompatibilityError::IR_REFERENCE_INVALID,
                                "execution group has duplicate or reordered operator coverage",
                                semantic.operator_id);
                operator_ids.push_back(semantic.operator_id);
            }
            step_ordinals.push_back(step.ordinal());
            const uint32_t primitive_base = static_cast<uint32_t>(primitives.size());
            for (const auto& source : step.primitives()) {
                auto invocation = source;
                invocation.order = static_cast<uint32_t>(primitives.size());
                primitives.push_back(invocation);
            }
            const auto step_coverage = step.coverage();
            for (const auto& source : step_coverage) {
                if (source.primitive_first != 0 ||
                    source.primitive_count != step.primitives().size() ||
                    source.primitive_count > primitives.size() - primitive_base)
                    return fail(CompatibilityError::IR_REFERENCE_INVALID,
                                "execution group primitive coverage is not exact",
                                source.operator_id);
                auto record = source;
                record.primitive_first += primitive_base;
                coverage_records.push_back(record);
            }
            state_effects.insert(state_effects.end(), step.state_effects().begin(),
                                 step.state_effects().end());
        }
        for (const auto& coverage : coverage_records) {
            if (coverage.kind == StructuralMetalCoverageKind::FusedCovered) {
                if (coverage.owner_step_ordinal < first ||
                    coverage.owner_step_ordinal >= first + count)
                    return fail(CompatibilityError::IR_REFERENCE_INVALID,
                                "fused coverage owner is outside its execution group",
                                coverage.operator_id);
                size_t owner_count = 0;
                for (const auto& owner : coverage_records)
                    if (owner.kind == StructuralMetalCoverageKind::FusedOwner &&
                        owner.owner_step_ordinal == coverage.owner_step_ordinal)
                        ++owner_count;
                if (owner_count != 1)
                    return fail(CompatibilityError::IR_REFERENCE_INVALID,
                                "fused coverage does not have one exact owner",
                                coverage.operator_id);
            }
        }
        uint32_t group_owner = first;
        if (kind == StructuralMetalBundleGroupKind::Layer) {
            group_owner = kSemanticDispatchUnresolved;
            for (const auto& coverage : coverage_records) {
                if (coverage.kind != StructuralMetalCoverageKind::FusedOwner) continue;
                if (group_owner == kSemanticDispatchUnresolved)
                    group_owner = coverage.owner_step_ordinal;
            }
            if (group_owner == kSemanticDispatchUnresolved)
                return fail(CompatibilityError::KERNEL_AMBIGUOUS,
                            "layer execution group has no fused executor owner");
        }
        if (const auto report = validate_group_primitives(
                state, kind, shape, primitives))
            return report;
        if (kind == StructuralMetalBundleGroupKind::Graph &&
            (count != 1 || program.steps[first].operation != OperatorKind::EmbeddingLookup))
            return fail(CompatibilityError::IR_REFERENCE_INVALID,
                        "graph execution group is not the embedding boundary");
        if (kind == StructuralMetalBundleGroupKind::Final &&
            program.steps[first].kind != SemanticDispatchStepKind::Operator)
            return fail(CompatibilityError::IR_REFERENCE_INVALID,
                        "final execution group is not an operator range");
        if (kind == StructuralMetalBundleGroupKind::Final) {
            if (const auto report = validate_final_group(model, program, first, count))
                return report;
        }
        for (uint32_t offset = 0; offset < count; ++offset)
            assigned[first + offset] = 1;
        output->emplace_back(static_cast<uint32_t>(output->size()), kind, shape,
                             first, count, group_owner,
                             program.request.phase,
                             program.request.batch_rows, std::move(step_ordinals),
                             std::move(operator_ids), std::move(primitives),
                             std::move(coverage_records), std::move(state_effects));
        return std::nullopt;
    };

    while (index < steps.size()) {
        if (active_index < active_views.size() &&
            active_views[active_index]->first_step == index) {
            if (saw_final || saw_sampler) return fail(
                CompatibilityError::IR_REFERENCE_INVALID,
                "layer execution group appears after a terminal group");
            const auto& view = *active_views[active_index++];
            if (const auto report = append_group(StructuralMetalBundleGroupKind::Layer,
                                                  active_shapes[active_index - 1],
                                                  view.first_step, view.step_count))
                return report;
            index += view.step_count;
            continue;
        }
        if (covered[index])
            return fail(CompatibilityError::IR_REFERENCE_INVALID,
                        "execution group range cursor skipped a layer step");
        const SemanticDispatchStep& step = program.steps[index];
        if (index == 0 && step.kind == SemanticDispatchStepKind::Operator &&
            step.operation == OperatorKind::EmbeddingLookup) {
            if (const auto report = append_group(StructuralMetalBundleGroupKind::Graph,
                                                  StructuralMetalExecutionShape::GraphEmbedding,
                                                  index, 1))
                return report;
            saw_embedding = true;
            ++index;
            continue;
        }
        if (step.kind == SemanticDispatchStepKind::GreedySampler) {
            if (saw_sampler || index + 1 != steps.size() || saw_final == false)
                return fail(CompatibilityError::IR_REFERENCE_INVALID,
                            "greedy sampler is not the final graph boundary");
            if (const auto report = append_group(StructuralMetalBundleGroupKind::Sampler,
                                                  StructuralMetalExecutionShape::GreedySampler,
                                                  index, 1))
                return report;
            saw_sampler = true;
            ++index;
            continue;
        }
        if (step.kind != SemanticDispatchStepKind::Operator || saw_final || saw_sampler)
            return fail(CompatibilityError::IR_REFERENCE_INVALID,
                        "uncovered semantic step is not an executable graph boundary",
                        step.operator_id);
        const uint32_t first = index;
        bool has_terminal = false;
        while (index < steps.size() && !covered[index] &&
               program.steps[index].kind == SemanticDispatchStepKind::Operator) {
            has_terminal = has_terminal || contains_id(
                program.terminal_operator_ids, program.steps[index].operator_id);
            ++index;
        }
        if (!has_terminal)
            return fail(CompatibilityError::IR_REFERENCE_INVALID,
                        "uncovered final range has no terminal output operator");
        if (const auto report = append_group(StructuralMetalBundleGroupKind::Final,
                                              StructuralMetalExecutionShape::FinalOutput,
                                              first, index - first))
            return report;
        saw_final = true;
    }
    if (active_index != active_views.size() ||
        std::any_of(assigned.begin(), assigned.end(), [](uint8_t value) { return value == 0; }) ||
        !saw_embedding)
        return fail(CompatibilityError::IR_REFERENCE_INVALID,
                    "execution groups do not cover every semantic step");
    if (!saw_final && !saw_sampler)
        return fail(CompatibilityError::IR_REFERENCE_INVALID,
                    "execution graph has no final output boundary");
    return std::nullopt;
}

std::optional<CompatibilityReport> compile_program(
    CompileState& state, const SemanticModel& model,
    const SemanticDispatchProgram& program, const BoundDispatchProgram& bound,
    StructuralMetalProgramBundle* output) {
    if (program.steps.size() > kMaximumSteps ||
        program.steps.size() != bound.steps().size() ||
        program.program_digest != bound.program_digest() ||
        program.request != bound.request())
        return fail(CompatibilityError::IR_REFERENCE_INVALID,
                    "semantic and bound dispatch programs are not aligned");
    CompatibilityReport validation;
    if (!validate_semantic_dispatch_program(model, program.request, program,
                                            &validation)) {
        validation.detail = "structural Metal compiler received a non-authoritative semantic program";
        return validation;
    }
    std::vector<std::vector<CertificateFact>> facts(program.steps.size());
    std::vector<const SemanticOperator*> operations(program.steps.size(), nullptr);
    for (size_t index = 0; index < program.steps.size(); ++index) {
        const SemanticDispatchStep& step = program.steps[index];
        const BoundDispatchStep& bound_step = bound.steps()[index];
        if (const auto report = validate_bound_dispatch_step(bound_step))
            return *report;
        if (step.ordinal != index || bound_step.ordinal() != index ||
            !(step.requirement == bound_step.requirement()) ||
            zero_digest(bound_step.bound_digest()))
            return fail(CompatibilityError::IR_REFERENCE_INVALID,
                        "semantic and bound step contracts are not aligned",
                        step.operator_id);
        if (step.kind == SemanticDispatchStepKind::Operator) {
            operations[index] = operator_for(model, step.operator_id);
            if (!operations[index])
                return fail(CompatibilityError::IR_REFERENCE_INVALID,
                            "dispatch step operator is outside the semantic graph",
                            step.operator_id);
            auto bound_facts = facts_for(state.input, model, step, bound_step,
                                         *operations[index]);
            if (!bound_facts)
                return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                            "dispatch tensor lacks a complete structural certificate",
                            step.operator_id);
            facts[index] = std::move(*bound_facts);
        } else if (step.kind != SemanticDispatchStepKind::GreedySampler ||
                   !step.sampler_binding.has_value() ||
                   !bound_step.codec_occurrence_indices().empty()) {
            return fail(CompatibilityError::IR_REFERENCE_INVALID,
                        "dispatch step kind is not a supported immutable bundle boundary",
                        step.operator_id);
        }
    }
    if (const auto report = validate_batched_program(model, program, facts)) return report;

    std::vector<StructuralMetalBundleStep> steps;
    steps.reserve(program.steps.size());
    for (size_t index = 0; index < program.steps.size(); ++index) {
        const SemanticDispatchStep& step = program.steps[index];
        const BoundDispatchStep& bound_step = bound.steps()[index];
        std::vector<StructuralMetalPrimitiveInvocation> invocations;
        std::vector<StructuralMetalSemanticCoverage> coverage;
        bool fused = false;
        if (step.kind == SemanticDispatchStepKind::GreedySampler) {
            if (step.phase != ExecutionPhase::Output ||
                (program.request.phase != ExecutionPhase::Prefill &&
                 program.request.phase != ExecutionPhase::Decode))
                return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                            "greedy sampler requires an admitted output boundary");
            if (const auto report = add_recipe(
                    state, step, StructuralMetalPrimitive::SamplerGreedy, 0,
                    facts[index], &invocations.emplace_back()))
                return report;
            coverage.push_back({kSemanticDispatchUnresolved,
                                StructuralMetalCoverageKind::Standalone,
                                kSemanticDispatchUnresolved, 0, 1});
        } else if (operations[index]->kind == OperatorKind::GatedRmsNorm ||
                   (operations[index]->kind == OperatorKind::L2Normalize &&
                    second_recurrent_l2_step(program, step.ordinal))) {
            uint32_t owner = kSemanticDispatchUnresolved;
            if (!find_recurrent_layer_owner(model, program, step.ordinal, &owner))
                return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                            "fused recurrent operation has no unique update owner",
                            step.operator_id);
            fused = true;
            coverage.push_back({step.operator_id,
                                StructuralMetalCoverageKind::FusedCovered, owner, 0, 0});
        } else if (operations[index]->kind == OperatorKind::AxisSplit) {
            uint32_t owner = kSemanticDispatchUnresolved;
            if (find_moe_activation_owner(program, step.ordinal, &owner)) {
                fused = true;
                coverage.push_back({step.operator_id,
                                    StructuralMetalCoverageKind::FusedCovered,
                                    owner, 0, 0});
            } else {
                if (const auto report = add_primitives(
                        state, model, program, step, *operations[index],
                        facts[index], invocations))
                    return report;
                coverage.push_back({step.operator_id,
                                    StructuralMetalCoverageKind::Standalone,
                                    step.ordinal, 0,
                                    static_cast<uint32_t>(invocations.size())});
            }
        } else {
            if (const auto report = add_primitives(
                    state, model, program, step, *operations[index],
                    facts[index], invocations))
                return report;
            const bool owner = fused_operator(operations[index]->kind) &&
                               (invocations.size() > 1 ||
                                operations[index]->kind == OperatorKind::GatedDeltaNet ||
                                operations[index]->kind == OperatorKind::GatedActivation);
            fused = owner;
            coverage.push_back({step.operator_id,
                                owner ? StructuralMetalCoverageKind::FusedOwner
                                      : StructuralMetalCoverageKind::Standalone,
                                step.ordinal, 0,
                                static_cast<uint32_t>(invocations.size())});
        }
        StructuralMetalBundleStep built_step(
            step.ordinal, step.requirement, bound_step.bound_digest(),
            step.covered_operator_ids, step.input_values, step.output_values,
            step.tensor_ids, step.state_effects, std::move(invocations),
            std::move(coverage), step.phase, program.request.batch_rows,
            step.operation, fused);
        steps.push_back(std::move(built_step));
    }
    std::vector<StructuralMetalBundleGroup> groups;
    if (const auto report = build_execution_groups(state, model, program, steps,
                                                  &groups))
        return report;
    *output = StructuralMetalProgramBundle(
        program.request.phase, program.request.batch_rows,
        program.program_digest, std::move(steps), std::move(groups));
    return std::nullopt;
}

MetalPipelineDigest compilation_digest(
    const std::vector<StructuralMetalProgramBundle>& programs,
    const std::vector<MetalPipelineRecipe>& recipes) {
    std::vector<uint8_t> bytes;
    static constexpr std::string_view domain = "laplace.structural-metal.compilation.v1";
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    append_u32(bytes, static_cast<uint32_t>(programs.size()));
    for (const auto& program : programs) {
        append_u16(bytes, static_cast<uint16_t>(program.phase()));
        append_u32(bytes, program.batch_rows());
        append_digest(bytes, program.program_digest());
        append_u32(bytes, static_cast<uint32_t>(program.steps().size()));
        for (const auto& step : program.steps()) {
            append_u32(bytes, step.ordinal());
            append_u32(bytes, static_cast<uint32_t>(step.primitives().size()));
            for (const auto& invocation : step.primitives()) {
                append_u8(bytes, static_cast<uint8_t>(invocation.primitive));
                append_u32(bytes, invocation.order);
                append_u32(bytes, invocation.recipe_index);
            }
        }
        append_u32(bytes, static_cast<uint32_t>(program.groups().size()));
        for (const auto& group : program.groups()) {
            append_u32(bytes, group.ordinal());
            append_u8(bytes, static_cast<uint8_t>(group.kind()));
            append_u8(bytes, static_cast<uint8_t>(group.shape()));
            append_u32(bytes, group.first_step());
            append_u32(bytes, group.step_count());
            append_u32(bytes, group.owner_step_ordinal());
            append_u16(bytes, static_cast<uint16_t>(group.phase()));
            append_u32(bytes, group.batch_rows());
            append_u32(bytes, static_cast<uint32_t>(group.covered_step_ordinals().size()));
            for (uint32_t ordinal : group.covered_step_ordinals()) append_u32(bytes, ordinal);
            append_u32(bytes, static_cast<uint32_t>(group.covered_operator_ids().size()));
            for (uint32_t id : group.covered_operator_ids()) append_u32(bytes, id);
            append_u32(bytes, static_cast<uint32_t>(group.primitives().size()));
            for (const auto& invocation : group.primitives()) {
                append_u8(bytes, static_cast<uint8_t>(invocation.primitive));
                append_u32(bytes, invocation.order);
                append_u32(bytes, invocation.recipe_index);
            }
            append_u32(bytes, static_cast<uint32_t>(group.coverage().size()));
            for (const auto& coverage : group.coverage()) {
                append_u32(bytes, coverage.operator_id);
                append_u8(bytes, static_cast<uint8_t>(coverage.kind));
                append_u32(bytes, coverage.owner_step_ordinal);
                append_u32(bytes, coverage.primitive_first);
                append_u32(bytes, coverage.primitive_count);
            }
            append_u32(bytes, static_cast<uint32_t>(group.state_effects().size()));
            for (const auto& effect : group.state_effects()) {
                append_u32(bytes, effect.state_id);
                append_u8(bytes, static_cast<uint8_t>(effect.access));
                append_u16(bytes, static_cast<uint16_t>(effect.update_kind));
            }
        }
    }
    append_u32(bytes, static_cast<uint32_t>(recipes.size()));
    for (const auto& recipe : recipes)
        append_digest(bytes, recipe.normalized_requirement_digest);
    return metal_digest(bytes);
}

} // namespace

const StructuralMetalLibraryIdentity* StructuralMetalLibraryIdentitySet::find(
    StructuralMetalLibraryId id) const noexcept {
    for (const auto& identity : identities)
        if (identity.id == id) return &identity;
    return nullptr;
}

StructuralMetalPrimitiveInvocation StructuralMetalBundleStep::primitive(
    size_t index) const noexcept {
    return index < primitives_.size() ? primitives_[index]
                                      : StructuralMetalPrimitiveInvocation{};
}

const MetalPipelineRecipe* StructuralMetalCompilation::recipe(
    uint32_t index) const noexcept {
    return index < recipes_.size() ? &recipes_[index] : nullptr;
}

class StructuralMetalCompiler {
public:
    static StructuralMetalCompilerResult run(
        const StructuralMetalCompilerInput& input) {
        try {
            if (!input.bound_requirements || !input.semantic_model ||
                input.programs.empty() ||
                input.programs.size() > kMaximumPrograms ||
                input.programs.size() != input.bound_requirements->programs().size())
                return fail(CompatibilityError::RUNTIME_INPUT_INVALID,
                            "structural Metal compiler input is incomplete");
            if (const auto report =
                    validate_bound_dispatch_requirements(*input.bound_requirements))
                return *report;
            if (const auto report = validate_libraries(input)) return *report;
            if (const auto report = validate_program_order(input.programs)) return *report;
            CompileState state{input, {}, {}};
            state.programs.reserve(input.programs.size());
            for (size_t index = 0; index < input.programs.size(); ++index) {
                const SemanticDispatchProgram& program = input.programs[index];
                const BoundDispatchProgram& bound =
                    input.bound_requirements->programs()[index];
                StructuralMetalProgramBundle bundle;
                if (const auto report = compile_program(
                        state, *input.semantic_model, program, bound, &bundle))
                    return *report;
                state.programs.push_back(std::move(bundle));
            }
            if (state.recipes.empty())
                return fail(CompatibilityError::KERNEL_UNAVAILABLE,
                            "structural Metal compiler emitted no primitive recipes");
            const MetalPipelineDigest digest =
                compilation_digest(state.programs, state.recipes);
            return StructuralMetalCompilation(
                std::move(state.programs), std::move(state.recipes), digest);
        } catch (const std::bad_alloc&) {
            return fail(CompatibilityError::PLAN_MEMORY_EXCEEDED,
                        "structural Metal compiler allocation exceeded its bound");
        }
    }
};

StructuralMetalCompilerResult compile_structural_metal(
    const StructuralMetalCompilerInput& input) {
    return StructuralMetalCompiler::run(input);
}

StructuralMetalCompilerResult compile_structural_metal(
    const BoundDispatchRequirements& bound_requirements,
    std::span<const SemanticDispatchProgram> programs,
    const SemanticModel& semantic_model,
    std::span<const PhysicalCodecSpec> certificate_specs,
    const StructuralMetalLibraryIdentitySet& libraries,
    const MetalCodecCapabilityRegistry* codec_capabilities) {
    StructuralMetalCompilerInput input;
    input.bound_requirements = &bound_requirements;
    input.programs = programs;
    input.semantic_model = &semantic_model;
    input.certificate_specs = certificate_specs;
    input.codec_capabilities = codec_capabilities;
    input.libraries = libraries;
    return compile_structural_metal(input);
}

#if defined(LAPLACE_TESTING)
MetalPipelineRecipe structural_metal_primitive_recipe_for_testing(
    StructuralMetalPrimitive primitive) {
    const PrimitiveSpec spec = primitive_spec(primitive);
    MetalPipelineRecipe recipe;
    recipe.function_name = std::string(spec.function);
    recipe.function_constants = spec.constants;
    recipe.dispatch = spec.dispatch;
    return recipe;
}
#endif

} // namespace Laplace
