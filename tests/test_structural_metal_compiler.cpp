#define LAPLACE_RUNTIME_PACKAGE_TESTING 1
#define main structural_embedded_bound_dispatch_main
#include "test_bound_dispatch_requirements.cpp"
#undef main

#include "structural_metal_compiler.h"
#include "metal_codec_capability.h"
#include "normalized_codec_program.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

#include "codec_certificate.h"
#include "test_util.h"

using namespace Laplace;

namespace {

StructuralMetalLibraryIdentity library(StructuralMetalLibraryId id,
                                        uint8_t source_byte) {
    StructuralMetalLibraryIdentity result;
    result.id = id;
    result.source_digest.fill(source_byte);
    return result;
}

const MetalPipelineRecipe* recipe_named(
    const StructuralMetalCompilation& compilation, std::string_view name,
    std::span<const MetalFunctionConstant> constants = {}) {
    const auto recipes = compilation.recipes();
    const auto found = std::find_if(
        recipes.begin(), recipes.end(), [&](const MetalPipelineRecipe& recipe) {
            return recipe.function_name == name &&
                   std::equal(recipe.function_constants.begin(),
                              recipe.function_constants.end(), constants.begin(),
                              constants.end());
        });
    return found == recipes.end() ? nullptr : &*found;
}

MetalCodecPhysicalTuple physical_tuple_for(
    const PhysicalCodecIdentity& identity) {
    return {identity.layout, identity.quantization, identity.planes};
}

const PhysicalCodecSpec* spec_for(
    std::span<const PhysicalCodecSpec> specs,
    const PhysicalCodecIdentity& identity) {
    for (const PhysicalCodecSpec& spec : specs)
        if (spec.identity == identity) return &spec;
    return nullptr;
}

bool tensor_is_router_projection(const SemanticModel& model,
                                 const SemanticOperator& operation) {
    return operation.kind == OperatorKind::Linear &&
           operation.outputs.size() == 1 &&
           std::any_of(model.operators.begin(), model.operators.end(),
                       [&](const SemanticOperator& candidate) {
                           return candidate.kind == OperatorKind::RouterTopK &&
                                  candidate.inputs.size() == 1 &&
                                  candidate.inputs[0] == operation.outputs[0];
                       });
}

bool same_primitives(std::span<const StructuralMetalPrimitiveInvocation> left,
                     std::span<const StructuralMetalPrimitiveInvocation> right) {
    return left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin());
}

MetalCodecLoweringStrategy strategy_for(
    const SemanticModel& model, const SemanticOperator& operation,
    size_t tensor_index, const PhysicalCodecIdentity& identity) {
    switch (operation.kind) {
    case OperatorKind::EmbeddingLookup:
        if (identity.layout.block_bytes == 210)
            return MetalCodecLoweringStrategy::StructuralEmbeddingQ6;
        if (identity.layout.block_bytes == 144)
            return MetalCodecLoweringStrategy::StructuralEmbeddingQ4;
        return MetalCodecLoweringStrategy::StructuralEmbeddingF16;
    case OperatorKind::RmsNorm:
        return MetalCodecLoweringStrategy::StructuralTensorF32;
    case OperatorKind::Linear:
        if (tensor_index != 0)
            return MetalCodecLoweringStrategy::StructuralTensorF32;
        if (tensor_is_router_projection(model, operation))
            return MetalCodecLoweringStrategy::StructuralGemvF32;
        if (identity.layout.kind == PhysicalLayoutKind::GroupedAffine)
            return MetalCodecLoweringStrategy::StructuralGemvAffineU2;
        if (identity.layout.kind == PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip)
            return MetalCodecLoweringStrategy::StructuralGemvColumnGroupedU2;
        if (identity.layout.block_bytes == 144)
            return MetalCodecLoweringStrategy::StructuralGemvQ4;
        if (identity.layout.block_bytes == 210)
            return MetalCodecLoweringStrategy::StructuralGemvQ6;
        return MetalCodecLoweringStrategy::StructuralGemvF16;
    case OperatorKind::RoutedLinear:
        return MetalCodecLoweringStrategy::StructuralGemvQ4;
    case OperatorKind::WeightedExpertReduce:
    case OperatorKind::Scale:
        return MetalCodecLoweringStrategy::StructuralTensorF32;
    default:
        return MetalCodecLoweringStrategy::StructuralTensorF32;
    }
}

MetalCodecCapabilityRegistry make_codec_capabilities(
    const SemanticModel& model, std::span<const SemanticDispatchProgram> programs,
    const BoundDispatchRequirements& bound,
    std::span<const PhysicalCodecSpec> specs) {
    MetalCodecCapabilityRegistry registry;
    std::vector<MetalCodecRequirement> seen;
    uint64_t lowering_identity = 100;
    for (size_t program_index = 0; program_index < programs.size(); ++program_index) {
        const SemanticDispatchProgram& program = programs[program_index];
        const BoundDispatchProgram& bound_program = bound.programs()[program_index];
        for (size_t step_index = 0; step_index < program.steps.size(); ++step_index) {
            const SemanticDispatchStep& step = program.steps[step_index];
            if (step.kind != SemanticDispatchStepKind::Operator) continue;
            const auto operation = std::find_if(
                model.operators.begin(), model.operators.end(),
                [&](const SemanticOperator& candidate) {
                    return candidate.id == step.operator_id;
                });
            CHECK(operation != model.operators.end());
            if (operation == model.operators.end()) continue;
            const BoundDispatchStep& bound_step = bound_program.steps()[step_index];
            const auto physical = bound_step.physical_identities();
            for (size_t tensor_index = 0; tensor_index < physical.size(); ++tensor_index) {
                const PhysicalCodecSpec* spec = spec_for(specs, physical[tensor_index]);
                CHECK(spec != nullptr);
                if (!spec) continue;
                const auto parsed = parse_codec_certificate(spec->certificate_bytes);
                CHECK(std::holds_alternative<CodecCertificate>(parsed));
                if (!std::holds_alternative<CodecCertificate>(parsed)) continue;
                const auto normalized = normalize_codec_program(
                    std::get<CodecCertificate>(parsed));
                CHECK(std::holds_alternative<NormalizedCodecProgram>(normalized));
                if (!std::holds_alternative<NormalizedCodecProgram>(normalized)) continue;
                const auto& decoder = std::get<NormalizedCodecProgram>(normalized);
                MetalCodecRequirement requirement;
                requirement.operation_abi = static_cast<uint32_t>(operation->kind);
                requirement.phase = static_cast<uint16_t>(step.phase);
                requirement.numerical_class = static_cast<uint16_t>(step.numerical_class);
                CHECK(tensor_index < operation->tensors.size());
                if (tensor_index >= operation->tensors.size()) continue;
                const SemanticTensor& tensor = model.tensors[operation->tensors[tensor_index]];
                CHECK(tensor.dimensions.size() <= requirement.shape_class.size());
                if (tensor.dimensions.size() > requirement.shape_class.size()) continue;
                for (size_t dimension = 0; dimension < tensor.dimensions.size(); ++dimension)
                    requirement.shape_class[dimension] =
                        static_cast<uint32_t>(tensor.dimensions[dimension].constant_or_symbol);
                requirement.semantic_signature = decoder.semantic_signature;
                requirement.physical = physical_tuple_for(physical[tensor_index]);
                if (std::find(seen.begin(), seen.end(), requirement) != seen.end()) continue;
                seen.push_back(requirement);
                const auto strategy = strategy_for(model, *operation, tensor_index,
                                                   physical[tensor_index]);
                CompatibilityReport error;
                CHECK(registry.add({requirement,
                                    {1u, 32u, 32u, 4u},
                                    {lowering_identity++, static_cast<uint32_t>(strategy)}},
                                   &error));
            }
        }
    }
    return registry;
}

ArtifactTensorRecord structural_physical_record(const SemanticTensor& tensor) {
    ArtifactTensorRecord record;
    record.id = tensor.id;
    record.coordinate.root = 0;
    const bool q4 = tensor.layout.kind == PhysicalLayoutKind::GgufBlocked;
    if (q4) {
        record.logical_type = ArtifactScalarType::F32;
        // The expert-bank axis is outermost in the source-neutral semantic
        // view.  GGUF block axis 0 is the packed input axis, so the expert
        // dimension is represented explicitly at physical axis 2.
        for (const Dimension& dimension : tensor.dimensions)
            record.logical_dimensions.push_back(dimension.constant_or_symbol);
        record.layout = tensor.layout;
        record.quantization = tensor.quantization;
        record.format = {1, ArtifactPhysicalEncoding::Q4_K,
                         ArtifactScalarType::Packed, ArtifactScalarType::F16,
                         ArtifactScalarType::F16, ArtifactScalarType::Packed,
                         ArtifactScalarType::None, 256, 144, 2, 2, 12, 0};
        if (tensor.expert_axis.kind == ExpertAxisKind::ExpertBank) {
            record.coordinate.bank_axis = tensor.expert_axis.expert_axis;
            record.coordinate.bank_extent = tensor.expert_axis.expert_count;
            record.coordinate.bank_stride = tensor.expert_axis.per_expert_byte_stride;
        }
        record.axis.source_rank = 3;
        record.axis.source_axis_order = {0, 1, 2, 0xff, 0xff, 0xff, 0xff, 0xff};
        const TensorPlane& source = tensor.planes.front();
        uint64_t elements = 1;
        for (uint64_t dimension : record.logical_dimensions) elements *= dimension;
        record.axis.block_axis = 0;
        record.axis.block_elements = 256;
        record.axis.bytes_per_block = 144;
        record.axis.row_stride_bytes =
            record.logical_dimensions[record.axis.block_axis] / 256u * 144u;
        record.planes.push_back({PlaneKind::Values, ArtifactScalarType::Packed,
                                 {source.artifact_id, source.offset, source.length},
                                 elements, 144, 256, source.alignment});
        return record;
    }
    const bool f16 = !tensor.planes.empty() &&
                     tensor.planes.front().storage_type == ScalarType::F16;
    const ArtifactScalarType scalar = f16 ? ArtifactScalarType::F16
                                          : ArtifactScalarType::F32;
    const uint32_t bytes_per_element = f16 ? 2 : 4;
    record.logical_type = tensor.logical_type == ScalarType::F16
        ? ArtifactScalarType::F16 : ArtifactScalarType::F32;
    for (const Dimension& dimension : tensor.dimensions)
        record.logical_dimensions.push_back(dimension.constant_or_symbol);
    record.layout = tensor.layout;
    record.quantization = tensor.quantization;
    record.format.version = 1;
    record.format.encoding = f16 ? ArtifactPhysicalEncoding::F16
                                 : ArtifactPhysicalEncoding::F32;
    record.format.value_type = scalar;
    record.format.block_elements = 1;
    record.format.block_bytes = bytes_per_element;
    const TensorPlane& source = tensor.planes.front();
    uint64_t elements = 1;
    for (uint64_t dimension : record.logical_dimensions) elements *= dimension;
    record.planes.push_back({PlaneKind::Values, scalar,
                             {source.artifact_id, source.offset, source.length},
                             elements, bytes_per_element, 1, source.alignment});
    record.axis.row_stride_bytes = record.logical_dimensions.empty()
        ? 0 : record.logical_dimensions.front() * bytes_per_element;
    return record;
}

Fixture make_structural_fixture(uint8_t fill, bool include_equivalent_declaration = true) {
    Fixture fixture;
    fixture.model = dense_model();
    for (uint32_t id : {0u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 11u}) {
        SemanticTensor& tensor = fixture.model.tensors[id];
        tensor.logical_type = ScalarType::F16;
        tensor.planes[0].storage_type = ScalarType::F16;
        tensor.planes[0].alignment = 2;
        uint64_t elements = 1;
        for (const Dimension& dimension : tensor.dimensions)
            elements *= dimension.constant_or_symbol;
        tensor.planes[0].length = elements * 2;
    }
    std::vector<uint8_t> bytes(4096, fill);
    auto artifact = ArtifactSet::make_owned_blob(ArtifactId{0}, ArtifactRole::Primary, bytes);
    CHECK(std::holds_alternative<PackageView>(artifact));
    if (!std::holds_alternative<PackageView>(artifact)) return fixture;
    ArtifactIndexInput index_input;
    index_input.artifacts.push_back(std::get<PackageView>(std::move(artifact)));
    for (const SemanticTensor& tensor : fixture.model.tensors)
        index_input.tensors.push_back(structural_physical_record(tensor));
    auto built_index = ArtifactIndex::build(std::move(index_input));
    CHECK_MSG(std::holds_alternative<ArtifactIndex>(built_index),
              "shape index error=%u detail=%s",
              std::holds_alternative<CompatibilityReport>(built_index)
                  ? static_cast<unsigned>(std::get<CompatibilityReport>(built_index).code) : 0u,
              std::holds_alternative<CompatibilityReport>(built_index)
                  ? std::get<CompatibilityReport>(built_index).detail.c_str() : "");
    if (!std::holds_alternative<ArtifactIndex>(built_index)) return fixture;
    ArtifactIndex index = std::get<ArtifactIndex>(std::move(built_index));
    TokenContract contract;
    contract.vocabulary_size = fixture.model.vocabulary_size;
    contract.bos_id = fixture.model.bos_id;
    contract.eos_id = fixture.model.eos_id;
    contract.stop_ids = fixture.model.stop_ids;
    contract.authoritative_tokenizer_digest = {fixture.model.tokenizer_digest};
    contract.authoritative_template_digest = {fixture.model.template_digest};
    PhysicalCodecRegistry registry;
    const auto f16_bytes = make_raw_f16_codec_certificate();
    const auto f32_bytes = make_raw_f32_codec_certificate();
    const auto alternate_f32_bytes = equivalent_f32_certificate();
    const auto f16_parsed = parse_codec_certificate(f16_bytes);
    const auto f32_parsed = parse_codec_certificate(f32_bytes);
    const auto alternate_f32_parsed = parse_codec_certificate(alternate_f32_bytes);
    const auto* f16_certificate = std::get_if<CodecCertificate>(&f16_parsed);
    const auto* f32_certificate = std::get_if<CodecCertificate>(&f32_parsed);
    const auto* alternate_f32_certificate =
        std::get_if<CodecCertificate>(&alternate_f32_parsed);
    CHECK(f16_certificate != nullptr);
    CHECK(f32_certificate != nullptr);
    CHECK(alternate_f32_certificate != nullptr);
    if (!f16_certificate || !f32_certificate || !alternate_f32_certificate) return fixture;
    for (const SemanticTensor& tensor : fixture.model.tensors) {
        const bool is_f16 = tensor.logical_type == ScalarType::F16;
        const bool alternate = include_equivalent_declaration && tensor.id == 1;
        const auto& certificate = is_f16
            ? *f16_certificate
            : alternate ? *alternate_f32_certificate : *f32_certificate;
        const auto identity = physical_codec_identity(
            tensor, certificate.identity().abi_version, certificate.identity().digest);
        CHECK(identity.has_value());
        if (!identity) return fixture;
        if (std::none_of(registry.codecs.begin(), registry.codecs.end(),
                         [&](const PhysicalCodecSpec& spec) {
                             return spec.identity == *identity;
                         }))
            registry.codecs.push_back({*identity,
                is_f16 ? f16_bytes : alternate ? alternate_f32_bytes : f32_bytes});
        registry.tensors.push_back({tensor.id, *identity});
    }
    std::sort(registry.codecs.begin(), registry.codecs.end(),
              [](const auto& left, const auto& right) {
                  return physical_codec_identity_less(left.identity, right.identity);
              });
    std::sort(registry.tensors.begin(), registry.tensors.end(),
              [](const auto& left, const auto& right) {
                  return left.tensor_id < right.tensor_id;
              });
    auto manifest = SemanticManifest::build(index, fixture.model, contract, registry);
    CHECK_MSG(std::holds_alternative<SemanticManifest>(manifest),
              "manifest error=%u detail=%s",
              std::holds_alternative<CompatibilityReport>(manifest)
                  ? static_cast<unsigned>(std::get<CompatibilityReport>(manifest).code) : 0u,
              std::holds_alternative<CompatibilityReport>(manifest)
                  ? std::get<CompatibilityReport>(manifest).detail.c_str() : "");
    if (!std::holds_alternative<SemanticManifest>(manifest)) return fixture;
    fixture.package = RuntimePackage::make_closed_v1_test_only(
        std::get<SemanticManifest>(std::move(manifest)));
    CHECK(fixture.package != nullptr);
    return fixture;
}

SemanticTensor structural_shape_tensor(uint32_t id, TensorRole role,
                                       std::initializer_list<uint32_t> dimensions,
                                       ScalarType storage, uint64_t offset) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    for (uint32_t dimension : dimensions)
        tensor.dimensions.push_back({DimensionKind::Constant, dimension});
    tensor.layout.rank = static_cast<uint8_t>(tensor.dimensions.size());
    uint64_t stride = 1;
    for (size_t reverse = tensor.dimensions.size(); reverse != 0; --reverse) {
        const size_t axis = reverse - 1;
        tensor.layout.axis_order[axis] = static_cast<uint8_t>(axis);
        tensor.layout.strides[axis] = stride;
        stride *= tensor.dimensions[axis].constant_or_symbol;
    }
    if (tensor.dimensions.size() == 2) {
        tensor.layout.strides[0] = tensor.dimensions[1].constant_or_symbol;
        tensor.layout.strides[1] = 1;
    }
    const uint32_t bytes = storage == ScalarType::F16 ? 2 : 4;
    tensor.planes = {{PlaneKind::Values, storage, ArtifactId{0}, offset,
                      stride * bytes, 64, 0}};
    return tensor;
}

SemanticValue structural_shape_value(uint32_t id, uint32_t width) {
    return {id, ScalarType::F32,
            {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, width}}, 0};
}

SemanticModel recurrent_shape_model() {
    constexpr uint32_t width = 4;
    constexpr uint32_t vocabulary = 3;
    constexpr uint32_t channels = 12;
    constexpr uint32_t intermediate = 4;
    SemanticModel model;
    model.schema_major = 3;
    model.opset_major = 3;
    model.maximum_context = 32;
    model.entry_kind = EntryKind::TokenIds;
    model.vocabulary_size = vocabulary;
    model.bos_id = 0;
    model.eos_id = 2;
    model.stop_ids = {2};
    for (size_t index = 0; index != model.tokenizer_digest.size(); ++index) {
        model.tokenizer_digest[index] = static_cast<uint8_t>(index + 1);
        model.template_digest[index] = static_cast<uint8_t>(index + 33);
    }
    const std::array<std::initializer_list<uint32_t>, 17> dimensions = {
        std::initializer_list<uint32_t>{vocabulary, width}, {width},
        {width, channels}, {width, width}, {width, 1}, {width, 1},
        {channels, 2}, {1}, {1}, {width}, {width, width}, {width},
        {width, intermediate}, {width, intermediate}, {intermediate, width},
        {width}, {width, vocabulary}};
    const std::array<TensorRole, 17> roles = {
        TensorRole::TokenEmbedding, TensorRole::AttentionNormWeight,
        TensorRole::RecurrentQkvWeight, TensorRole::RecurrentGateWeight,
        TensorRole::RecurrentBetaWeight, TensorRole::RecurrentAlphaWeight,
        TensorRole::RecurrentConvWeight, TensorRole::RecurrentDtBias,
        TensorRole::RecurrentDecayWeight, TensorRole::RecurrentNormWeight,
        TensorRole::RecurrentOutputWeight, TensorRole::FfnNormWeight,
        TensorRole::FfnGateWeight, TensorRole::FfnUpWeight,
        TensorRole::FfnDownWeight, TensorRole::FinalNormWeight,
        TensorRole::OutputWeight};
    uint64_t offset = 0;
    for (size_t index = 0; index != dimensions.size(); ++index) {
        const bool vector = dimensions[index].size() == 1;
        const ScalarType storage = vector ? ScalarType::F32 : ScalarType::F16;
        model.tensors.push_back(structural_shape_tensor(
            static_cast<uint32_t>(index), roles[index], dimensions[index], storage,
            offset));
        offset = (offset + model.tensors.back().planes.front().length + 63u) & ~uint64_t{63};
    }
    const std::array<uint32_t, 23> value_widths = {
        width, width, channels, width, 1, 1, width, width, width, width,
        width, width, width, width, width, width, intermediate, intermediate,
        intermediate, width, width, width, vocabulary};
    for (uint32_t id = 0; id != value_widths.size(); ++id)
        model.values.push_back(structural_shape_value(id, value_widths[id]));
    model.values.push_back({23, ScalarType::U32, {{DimensionKind::Constant, 1}}, 0});
    model.input_values_first = 23;
    model.input_values_count = 1;
    model.output_values_first = 22;
    model.output_values_count = 1;
    const auto add = [&](OperatorKind kind, std::vector<uint32_t> inputs,
                         std::vector<uint32_t> outputs, std::vector<uint32_t> tensors,
                         std::vector<uint32_t> states, OperatorPayload payload) {
        SemanticOperator operation;
        operation.id = static_cast<uint32_t>(model.operators.size());
        operation.kind = kind;
        operation.semantic_version = 3;
        operation.inputs = std::move(inputs);
        operation.outputs = std::move(outputs);
        operation.tensors = std::move(tensors);
        operation.states = std::move(states);
        operation.payload = std::move(payload);
        model.operators.push_back(std::move(operation));
    };
    constexpr uint32_t epsilon = 0x358637bdu;
    add(OperatorKind::EmbeddingLookup, {23}, {0}, {0}, {},
        EmbeddingLookupPayload{0x3f800000u, vocabulary, width, 0});
    add(OperatorKind::RmsNorm, {0}, {1}, {1}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {1}, {2}, {2}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {3}, {3}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {4}, {4}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {5}, {5}, {}, LinearPayload{});
    add(OperatorKind::DepthwiseConvSilu, {2}, {6, 7, 8}, {6}, {0},
        DepthwiseConvSiluPayload{1, 1, width, 2});
    add(OperatorKind::L2Normalize, {6}, {9}, {}, {}, L2NormalizePayload{epsilon});
    add(OperatorKind::L2Normalize, {7}, {10}, {}, {}, L2NormalizePayload{epsilon});
    add(OperatorKind::GatedDeltaNet, {9, 10, 8, 4, 5}, {11}, {7, 8}, {1},
        GatedDeltaNetPayload{1, 1, width, QkHeadMapping::ValueHeadModulo,
                             BetaTransform::Sigmoid, DecayTransform::NegativeSoftplus,
                             DeltaStateLayout::ValueHeadKeyRowOutputColumn, 0});
    add(OperatorKind::GatedRmsNorm, {11, 3}, {12}, {9}, {},
        GatedRmsNormPayload{epsilon, ActivationKind::Silu, 1});
    add(OperatorKind::Linear, {12}, {13}, {10}, {}, LinearPayload{});
    add(OperatorKind::Add, {0, 13}, {14}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {14}, {15}, {11}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {15}, {16}, {12}, {}, LinearPayload{});
    add(OperatorKind::Linear, {15}, {17}, {13}, {}, LinearPayload{});
    add(OperatorKind::SwiGlu, {16, 17}, {18}, {}, {}, SwiGluPayload{ActivationKind::Silu});
    add(OperatorKind::Linear, {18}, {19}, {14}, {}, LinearPayload{});
    add(OperatorKind::Add, {14, 19}, {20}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {20}, {21}, {15}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {21}, {22}, {16}, {}, LinearPayload{});
    model.layers = {{0, 1, 18, 0}};
    StateFormat recurrent;
    recurrent.kind = StateFormatKind::RecurrentContiguous;
    recurrent.logical_type = ScalarType::F32;
    recurrent.encoded_type = ScalarType::F32;
    recurrent.codec = CodecKind::Fp32;
    recurrent.cache_policy = CachePolicy::Recurrent;
    recurrent.layout_policy = LayoutPolicy::ChannelMajorHistory;
    recurrent.alignment = 64;
    model.states.push_back({0, StateKind::RecurrentConvHistory, 3,
                            StateUpdateKind::ShiftHistory,
                            PositionPolicy::ReplaceAtCursor,
                            {{DimensionKind::Constant, channels}, {DimensionKind::Constant, 1}},
                            {recurrent}, 0});
    recurrent.layout_policy = LayoutPolicy::ValueHeadKeyRowOutputColumn;
    model.states.push_back({1, StateKind::RecurrentDeltaMatrix, 3,
                            StateUpdateKind::DeltaMatrix,
                            PositionPolicy::ReplaceAtCursor,
                            {{DimensionKind::Constant, 1}, {DimensionKind::Constant, width},
                             {DimensionKind::Constant, width}}, {recurrent}, 0});
    return model;
}

SemanticTensor structural_q4_expert_tensor(uint32_t id, TensorRole role,
                                           uint32_t experts, uint32_t input,
                                           uint32_t output, uint64_t offset) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, input},
                         {DimensionKind::Constant, output},
                         {DimensionKind::Constant, experts}};
    tensor.layout.kind = PhysicalLayoutKind::GgufBlocked;
    tensor.layout.rank = 3;
    tensor.layout.packing = PackingKind::Gguf;
    tensor.layout.block_rank = 1;
    tensor.layout.axis_order = {0, 1, 2, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides = {1, input, static_cast<uint64_t>(input) * output,
                             0, 0, 0, 0, 0};
    tensor.layout.block_elements = 256;
    tensor.layout.block_bytes = 144;
    tensor.quantization.kind = QuantizationKind::BlockedAffine;
    tensor.quantization.scale_type = ScalarType::F16;
    tensor.quantization.zero_type = ScalarType::F16;
    tensor.quantization.block_elements = 256;
    tensor.quantization.block_bytes = 144;
    tensor.quantization.group_size = 256;
    tensor.quantization.required_plane_mask = 1;
    tensor.expert_axis = {ExpertAxisKind::ExpertBank, 2, 0xff, 0, 1,
                          experts, static_cast<uint64_t>(input) * output / 256u * 144u, 0};
    const uint64_t elements = static_cast<uint64_t>(experts) * input * output;
    tensor.planes = {{PlaneKind::Values, ScalarType::U8, ArtifactId{0}, offset,
                      elements / 256u * 144u, 64, 0}};
    return tensor;
}

SemanticModel moe_shape_model() {
    constexpr uint32_t hidden = 256;
    constexpr uint32_t vocabulary = 3;
    constexpr uint32_t intermediate = 256;
    constexpr uint32_t expert_intermediate = 256;
    constexpr uint32_t experts = 1;
    constexpr uint32_t selected = 1;
    SemanticModel model;
    model.schema_major = 7;
    model.opset_major = 7;
    model.maximum_context = 4;
    model.entry_kind = EntryKind::TokenIds;
    model.vocabulary_size = vocabulary;
    model.bos_id = 0;
    model.eos_id = 2;
    model.stop_ids = {2};
    for (size_t index = 0; index != model.tokenizer_digest.size(); ++index) {
        model.tokenizer_digest[index] = static_cast<uint8_t>(index + 1);
        model.template_digest[index] = static_cast<uint8_t>(index + 33);
    }
    uint64_t offset = 0;
    const auto add_tensor = [&](SemanticTensor tensor) {
        offset = (offset + 63u) & ~uint64_t{63};
        tensor.planes.front().offset = offset;
        offset += tensor.planes.front().length;
        model.tensors.push_back(std::move(tensor));
    };
    const auto f16 = [&](uint32_t id, TensorRole role,
                         std::initializer_list<uint32_t> dimensions) {
        add_tensor(structural_shape_tensor(id, role, dimensions, ScalarType::F16, offset));
    };
    const auto f32 = [&](uint32_t id, TensorRole role,
                         std::initializer_list<uint32_t> dimensions) {
        add_tensor(structural_shape_tensor(id, role, dimensions, ScalarType::F32, offset));
    };
    f16(0, TensorRole::TokenEmbedding, {vocabulary, hidden});
    f32(1, TensorRole::AttentionNormWeight, {hidden});
    f16(2, TensorRole::QueryWeight, {hidden, hidden});
    f16(3, TensorRole::KeyWeight, {hidden, hidden});
    f16(4, TensorRole::ValueWeight, {hidden, hidden});
    f16(5, TensorRole::AttentionOutputWeight, {hidden, hidden});
    f32(6, TensorRole::FfnNormWeight, {hidden});
    f16(7, TensorRole::FfnGateWeight, {hidden, intermediate});
    f16(8, TensorRole::FfnUpWeight, {hidden, intermediate});
    f16(9, TensorRole::FfnDownWeight, {intermediate, hidden});
    f32(10, TensorRole::RouterScaleWeight, {hidden});
    f32(11, TensorRole::ExpertNormWeight, {hidden});
    f32(12, TensorRole::ReduceScaleWeight, {experts});
    f32(13, TensorRole::NextnProjectionWeight, {experts, hidden});
    add_tensor(structural_q4_expert_tensor(14, TensorRole::FfnUpWeight, experts,
                                           hidden, 2 * expert_intermediate, offset));
    add_tensor(structural_q4_expert_tensor(15, TensorRole::FfnDownWeight, experts,
                                           expert_intermediate, hidden, offset));
    f32(16, TensorRole::FinalNormWeight, {hidden});
    f16(17, TensorRole::OutputWeight, {hidden, vocabulary});

    const auto value = [](uint32_t id, ScalarType type,
                          std::vector<Dimension> dimensions) {
        SemanticValue result;
        result.id = id;
        result.logical_type = type;
        result.dimensions = std::move(dimensions);
        return result;
    };
    const auto rows = [](uint32_t width) {
        return std::vector<Dimension>{{DimensionKind::Symbol, 1},
                                      {DimensionKind::Constant, width}};
    };
    const auto routed = [](uint32_t width) {
        return std::vector<Dimension>{{DimensionKind::Symbol, 1},
                                      {DimensionKind::Constant, selected},
                                      {DimensionKind::Constant, width}};
    };
    for (uint32_t id = 0; id != 19; ++id)
        model.values.push_back(value(id, ScalarType::F32, rows(id == 18 ? experts : hidden)));
    model.values.push_back(value(19, ScalarType::U32, rows(selected)));
    model.values.push_back(value(20, ScalarType::F32, rows(selected)));
    model.values.push_back(value(21, ScalarType::F32, rows(hidden)));
    for (uint32_t id = 22; id != 27; ++id)
        model.values.push_back(value(id, ScalarType::F32,
                                     routed(id == 22 ? 2 * expert_intermediate
                                           : id == 26 ? hidden : expert_intermediate)));
    for (uint32_t id = 27; id != 34; ++id)
        model.values.push_back(value(id, ScalarType::F32,
                                     rows(id == 30 ? vocabulary : id == 33 ? experts : hidden)));
    model.values.push_back(value(34, ScalarType::U32, rows(1)));

    const auto add = [&](OperatorKind kind, std::vector<uint32_t> inputs,
                         std::vector<uint32_t> outputs, std::vector<uint32_t> tensors,
                         std::vector<uint32_t> states, OperatorPayload payload) {
        SemanticOperator operation;
        operation.id = static_cast<uint32_t>(model.operators.size());
        operation.kind = kind;
        operation.semantic_version = 7;
        operation.inputs = std::move(inputs);
        operation.outputs = std::move(outputs);
        operation.tensors = std::move(tensors);
        operation.states = std::move(states);
        operation.payload = std::move(payload);
        model.operators.push_back(std::move(operation));
    };
    constexpr uint32_t epsilon = 0x358637bdu;
    constexpr uint32_t one = 0x3f800000u;
    CausalAttentionPayload attention{1, 1, hidden, 0x3e800000u,
                                     AttentionMask::Causal, CachePolicy::Global};
    attention.value_source_value = 4;
    RouterTopKPayload router;
    router.expert_count = experts;
    router.selected_count = selected;
    router.score_domain = RouterScoreDomain::Logits;
    router.normalization_order = RouterNormalizationOrder::NormalizeThenSelect;
    router.selected_weight_normalization = SelectedWeightNormalization::RenormalizeSelectedProbabilities;
    router.tie_policy = RouterTiePolicy::LowestExpertId;
    router.weight_source = RouterWeightSource::SelectedNormalizedScore;
    add(OperatorKind::EmbeddingLookup, {34}, {0}, {0}, {},
        EmbeddingLookupPayload{one, vocabulary, hidden, 0});
    add(OperatorKind::RmsNorm, {0}, {1}, {1}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {1}, {2}, {2}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {3}, {3}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {4}, {4}, {}, LinearPayload{});
    add(OperatorKind::Rope, {2, 3}, {5, 6}, {}, {},
        RopePayload{RopePairing::HalfSplit, true, hidden, 0x49742400u, one});
    add(OperatorKind::CausalAttention, {5, 6, 4}, {7}, { }, {0, 1}, attention);
    add(OperatorKind::Linear, {7}, {8}, {5}, {}, LinearPayload{});
    add(OperatorKind::Add, {0, 8}, {9}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {9}, {10}, {6}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {10}, {11}, {7}, {}, LinearPayload{});
    add(OperatorKind::Linear, {10}, {12}, {8}, {}, LinearPayload{});
    add(OperatorKind::SwiGlu, {11, 12}, {13}, {}, {}, SwiGluPayload{ActivationKind::Silu});
    add(OperatorKind::Linear, {13}, {14}, {9}, {}, LinearPayload{});
    add(OperatorKind::Add, {14, 27}, {28}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {9}, {16}, {}, {}, RmsNormPayload{epsilon, -1, 0});
    add(OperatorKind::Scale, {16}, {31}, {10}, {}, ScalePayload{ScaleSource::Tensor, 0});
    add(OperatorKind::Scale, {31}, {32}, {}, {}, ScalePayload{ScaleSource::LiteralF32, 0x3d3504f3u});
    add(OperatorKind::Linear, {32}, {33}, {13}, {}, LinearPayload{});
    add(OperatorKind::RouterTopK, {33}, {19, 20}, {}, {}, router);
    add(OperatorKind::RmsNorm, {9}, {21}, {11}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::RoutedLinear, {21, 19, 20}, {22}, {14}, {}, RoutedLinearPayload{ScalarType::F32});
    add(OperatorKind::AxisSplit, {22}, {23, 24}, {}, {}, AxisSplitPayload{expert_intermediate, expert_intermediate});
    add(OperatorKind::GatedActivation, {23, 24}, {25}, {}, {}, GatedActivationPayload{ActivationKind::Silu});
    add(OperatorKind::RoutedLinear, {25, 19, 20}, {26}, {15}, {}, RoutedLinearPayload{ScalarType::F32});
    add(OperatorKind::WeightedExpertReduce, {26, 19, 20}, {27}, {12}, {},
        WeightedExpertReducePayload{ExpertReduceAssociation::SelectedOrderLeftToRight,
                                    ExpertScaleSource::PerExpertTensor, ScalarType::F32});
    add(OperatorKind::Add, {9, 28}, {15}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {15}, {29}, {16}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {29}, {30}, {17}, {}, LinearPayload{});
    model.input_values_first = 34;
    model.input_values_count = 1;
    model.output_values_first = 30;
    model.output_values_count = 1;
    model.layers = {{0, 1, 26, 0}};
    StateFormat key_format;
    key_format.encoded_domain = TransformDomain::RopeApplied;
    key_format.alignment = 64;
    StateFormat value_format;
    value_format.alignment = 64;
    model.states = {
        {0, StateKind::KeyCache, 7, StateUpdateKind::AppendKey, PositionPolicy::AppendOnly,
         {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1}, {DimensionKind::Constant, hidden}},
         {key_format}, 0},
        {1, StateKind::ValueCache, 7, StateUpdateKind::AppendValue, PositionPolicy::AppendOnly,
         {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1}, {DimensionKind::Constant, hidden}},
         {value_format}, 0},
    };
    // The canonical fixture uses two scratch value IDs (17 and 18) that are
    // not produced by any operator.  A closed manifest correctly rejects
    // those dangling declarations, so compact the value table while
    // preserving every typed edge and the graph input/output boundaries.
    std::vector<uint32_t> value_map(35, kSemanticDispatchUnresolved);
    uint32_t compact_id = 0;
    std::vector<SemanticValue> compact_values;
    compact_values.reserve(model.values.size() - 2);
    for (const SemanticValue& value_record : model.values) {
        if (value_record.id == 17 || value_record.id == 18) continue;
        value_map[value_record.id] = compact_id;
        SemanticValue compact = value_record;
        compact.id = compact_id++;
        compact_values.push_back(std::move(compact));
    }
    for (SemanticOperator& operation : model.operators) {
        for (uint32_t& value_id : operation.inputs) value_id = value_map[value_id];
        for (uint32_t& value_id : operation.outputs) value_id = value_map[value_id];
    }
    // Make the layer a valid serialized topological program.  The branch
    // merge consumes the routed reduce result, so it must be declared after
    // that result while the structural compiler still matches by edges.
    const std::array<uint32_t, 29> topological_order = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 14, 26, 27, 28};
    std::vector<SemanticOperator> reordered;
    reordered.reserve(model.operators.size());
    for (uint32_t old_id : topological_order)
        reordered.push_back(std::move(model.operators[old_id]));
    for (uint32_t id = 0; id != reordered.size(); ++id) reordered[id].id = id;
    model.operators = std::move(reordered);
    model.values = std::move(compact_values);
    model.input_values_first = value_map[34];
    model.output_values_first = value_map[30];
    return model;
}

Fixture make_raw_shape_fixture(SemanticModel model) {
    Fixture fixture;
    fixture.model = std::move(model);
    std::vector<uint8_t> bytes(1u << 22, 0x2a);
    auto artifact = ArtifactSet::make_owned_blob(ArtifactId{0}, ArtifactRole::Primary, bytes);
    CHECK(std::holds_alternative<PackageView>(artifact));
    if (!std::holds_alternative<PackageView>(artifact)) return fixture;
    ArtifactIndexInput index_input;
    index_input.artifacts.push_back(std::get<PackageView>(std::move(artifact)));
    for (const SemanticTensor& tensor : fixture.model.tensors)
        index_input.tensors.push_back(structural_physical_record(tensor));
    auto built_index = ArtifactIndex::build(std::move(index_input));
    CHECK_MSG(std::holds_alternative<ArtifactIndex>(built_index),
              "shape index error=%u detail=%s",
              std::holds_alternative<CompatibilityReport>(built_index)
                  ? static_cast<unsigned>(std::get<CompatibilityReport>(built_index).code) : 0u,
              std::holds_alternative<CompatibilityReport>(built_index)
                  ? std::get<CompatibilityReport>(built_index).detail.c_str() : "");
    if (!std::holds_alternative<ArtifactIndex>(built_index)) return fixture;
    ArtifactIndex index = std::get<ArtifactIndex>(std::move(built_index));
    TokenContract contract;
    contract.vocabulary_size = fixture.model.vocabulary_size;
    contract.bos_id = fixture.model.bos_id;
    contract.eos_id = fixture.model.eos_id;
    contract.stop_ids = fixture.model.stop_ids;
    contract.authoritative_tokenizer_digest = {fixture.model.tokenizer_digest};
    contract.authoritative_template_digest = {fixture.model.template_digest};
    PhysicalCodecRegistry registry;
    const auto f16_bytes = make_raw_f16_codec_certificate();
    const auto f32_bytes = make_raw_f32_codec_certificate();
    const auto q4_bytes = make_q4_k_codec_certificate();
    const auto f16_parsed = parse_codec_certificate(f16_bytes);
    const auto f32_parsed = parse_codec_certificate(f32_bytes);
    const auto q4_parsed = parse_codec_certificate(q4_bytes);
    const auto* f16_certificate = std::get_if<CodecCertificate>(&f16_parsed);
    const auto* f32_certificate = std::get_if<CodecCertificate>(&f32_parsed);
    const auto* q4_certificate = std::get_if<CodecCertificate>(&q4_parsed);
    CHECK(f16_certificate != nullptr);
    CHECK(f32_certificate != nullptr);
    CHECK(q4_certificate != nullptr);
    if (!f16_certificate || !f32_certificate || !q4_certificate) return fixture;
    for (const SemanticTensor& tensor : fixture.model.tensors) {
        const bool is_q4 = tensor.layout.kind == PhysicalLayoutKind::GgufBlocked;
        const bool is_f16 = !tensor.planes.empty() && tensor.planes[0].storage_type == ScalarType::F16;
        const auto& certificate = is_q4 ? *q4_certificate
            : is_f16 ? *f16_certificate : *f32_certificate;
        const auto identity = physical_codec_identity(
            tensor, certificate.identity().abi_version, certificate.identity().digest);
        CHECK(identity.has_value());
        if (!identity) return fixture;
        if (std::none_of(registry.codecs.begin(), registry.codecs.end(),
                         [&](const PhysicalCodecSpec& candidate) {
                             return candidate.identity == *identity;
                         }))
            registry.codecs.push_back({*identity, is_q4 ? q4_bytes : is_f16 ? f16_bytes : f32_bytes});
        registry.tensors.push_back({tensor.id, *identity});
    }
    std::sort(registry.codecs.begin(), registry.codecs.end(),
              [](const auto& left, const auto& right) {
                  return physical_codec_identity_less(left.identity, right.identity);
              });
    std::sort(registry.tensors.begin(), registry.tensors.end(),
              [](const auto& left, const auto& right) {
                  return left.tensor_id < right.tensor_id;
              });
    auto manifest = SemanticManifest::build(index, fixture.model, contract, registry);
    CHECK_MSG(std::holds_alternative<SemanticManifest>(manifest),
              "shape manifest error=%u message=%s detail=%s",
              std::holds_alternative<CompatibilityReport>(manifest)
                  ? static_cast<unsigned>(std::get<CompatibilityReport>(manifest).code) : 0u,
              std::holds_alternative<CompatibilityReport>(manifest)
                  ? std::get<CompatibilityReport>(manifest).message.c_str() : "",
              std::holds_alternative<CompatibilityReport>(manifest)
                  ? std::get<CompatibilityReport>(manifest).detail.c_str() : "");
    if (!std::holds_alternative<SemanticManifest>(manifest)) return fixture;
    fixture.package = RuntimePackage::make_closed_v1_test_only(
        std::get<SemanticManifest>(std::move(manifest)));
    CHECK(fixture.package != nullptr);
    return fixture;
}

StructuralMetalCompilerResult compile_fixture_programs(
    const Fixture& fixture, std::span<const SemanticDispatchProgram> programs,
    const SessionRequest& request,
    const StructuralMetalLibraryIdentitySet& libraries,
    const MetalCodecCapabilityRegistry* override_capabilities = nullptr) {
    const auto bindings_result = preflight_codec_bindings(*fixture.package);
    if (!std::holds_alternative<ResolvedCodecBindings>(bindings_result))
        return std::get<CompatibilityReport>(bindings_result);
    const auto bound_result = bind_dispatch_requirements(
        *fixture.package, std::get<ResolvedCodecBindings>(bindings_result),
        request, programs);
    if (!std::holds_alternative<BoundDispatchRequirements>(bound_result))
        return std::get<CompatibilityReport>(bound_result);
    const BoundDispatchRequirements& bound =
        std::get<BoundDispatchRequirements>(bound_result);
    MetalCodecCapabilityRegistry capabilities =
        make_codec_capabilities(fixture.model, programs, bound,
                                fixture.package->physical_codec_registry().codecs);
    const MetalCodecCapabilityRegistry* selected =
        override_capabilities == nullptr ? &capabilities : override_capabilities;
    return compile_structural_metal(
        bound, programs, fixture.model,
        fixture.package->physical_codec_registry().codecs, libraries, selected);
}

void test_library_identity_is_required() {
    StructuralMetalCompilerInput input;
    const auto result = compile_structural_metal(input);
    CHECK(std::holds_alternative<CompatibilityReport>(result));
}

void test_structural_codec_selection_accepts_portable_registry() {
    StructuralMetalCompilerInput input;
    MetalCodecCapabilityRegistry registry;
    input.codec_capabilities = &registry;
    const auto result = compile_structural_metal(input);
    CHECK(std::holds_alternative<CompatibilityReport>(result));
}

void test_recipe_digest_binds_contract() {
    const auto f16 = make_raw_f16_codec_certificate();
    const auto parsed = parse_codec_certificate(f16);
    CHECK(std::holds_alternative<CodecCertificate>(parsed));
    if (!std::holds_alternative<CodecCertificate>(parsed)) return;

    StructuralMetalLibraryIdentitySet libraries;
    libraries.identities = {
        library(StructuralMetalLibraryId::Core, 0x11),
        library(StructuralMetalLibraryId::Prefill, 0x12),
        library(StructuralMetalLibraryId::Sampler, 0x13),
    };
    CHECK(libraries.identities.size() == 3);
}

void test_equivalent_codec_declarations_share_structural_lowering() {
    Fixture reordered = make_structural_fixture(0x31, true);
    Fixture canonical = make_structural_fixture(0x31, false);
    CHECK(reordered.package != nullptr);
    CHECK(canonical.package != nullptr);
    if (!reordered.package || !canonical.package) return;

    const SessionRequest request{4, 1, 0, true, true, false, false,
                                 NumericalClass::ExactFp32, RuntimeObjective::Latency};
    const auto prefill_result = build_semantic_dispatch_program(
        reordered.model, {ExecutionPhase::Prefill, 1, NumericalClass::ExactFp32,
                          false, false});
    const auto prefill_sample_result = build_semantic_dispatch_program(
        reordered.model, {ExecutionPhase::Prefill, 1, NumericalClass::ExactFp32,
                          false, true});
    const auto decode_result = build_semantic_dispatch_program(
        reordered.model, {ExecutionPhase::Decode, 1, NumericalClass::ExactFp32,
                          false, false});
    const auto decode_sample_result = build_semantic_dispatch_program(
        reordered.model, {ExecutionPhase::Decode, 1, NumericalClass::ExactFp32,
                          false, true});
    CHECK(std::holds_alternative<SemanticDispatchProgram>(prefill_result));
    CHECK(std::holds_alternative<SemanticDispatchProgram>(prefill_sample_result));
    CHECK(std::holds_alternative<SemanticDispatchProgram>(decode_result));
    CHECK(std::holds_alternative<SemanticDispatchProgram>(decode_sample_result));
    if (!std::holds_alternative<SemanticDispatchProgram>(prefill_result) ||
        !std::holds_alternative<SemanticDispatchProgram>(prefill_sample_result) ||
        !std::holds_alternative<SemanticDispatchProgram>(decode_result) ||
        !std::holds_alternative<SemanticDispatchProgram>(decode_sample_result)) return;
    const std::array<SemanticDispatchProgram, 4> programs = {
        std::get<SemanticDispatchProgram>(prefill_result),
        std::get<SemanticDispatchProgram>(prefill_sample_result),
        std::get<SemanticDispatchProgram>(decode_result),
        std::get<SemanticDispatchProgram>(decode_sample_result)};
    StructuralMetalLibraryIdentitySet libraries;
    libraries.identities = {
        library(StructuralMetalLibraryId::Core, 0x41),
        library(StructuralMetalLibraryId::Prefill, 0x42),
        library(StructuralMetalLibraryId::Sampler, 0x43),
    };
    const auto first = compile_fixture_programs(reordered, programs, request, libraries);
    const auto second = compile_fixture_programs(canonical, programs, request, libraries);
    CHECK(std::holds_alternative<StructuralMetalCompilation>(first));
    CHECK(std::holds_alternative<StructuralMetalCompilation>(second));
    if (!std::holds_alternative<StructuralMetalCompilation>(first) ||
        !std::holds_alternative<StructuralMetalCompilation>(second)) return;
    const auto& first_compiled = std::get<StructuralMetalCompilation>(first);
    const auto& second_compiled = std::get<StructuralMetalCompilation>(second);
    CHECK(first_compiled.compilation_digest() == second_compiled.compilation_digest());
    CHECK(first_compiled.recipes().size() == second_compiled.recipes().size());
    if (first_compiled.recipes().size() == second_compiled.recipes().size())
        for (size_t index = 0; index < first_compiled.recipes().size(); ++index)
            CHECK(first_compiled.recipes()[index] == second_compiled.recipes()[index]);
    CHECK(first_compiled.programs().size() == second_compiled.programs().size());
    if (first_compiled.programs().size() == second_compiled.programs().size())
        for (size_t program = 0; program < first_compiled.programs().size(); ++program) {
            CHECK(first_compiled.programs()[program].steps().size() ==
                  second_compiled.programs()[program].steps().size());
            if (first_compiled.programs()[program].steps().size() !=
                second_compiled.programs()[program].steps().size()) continue;
            for (size_t step = 0; step < first_compiled.programs()[program].steps().size(); ++step)
                CHECK(same_primitives(
                    first_compiled.programs()[program].steps()[step].primitives(),
                    second_compiled.programs()[program].steps()[step].primitives()));
        }
    CHECK(first_compiled.programs()[0].steps()[1].bound_digest() !=
          second_compiled.programs()[0].steps()[1].bound_digest());
}

void test_unknown_valid_codec_fails_closed() {
    const Fixture fixture = make_structural_fixture(0x31, false);
    CHECK(fixture.package != nullptr);
    if (!fixture.package) return;
    const auto prefill_result = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Prefill, 1, NumericalClass::ExactFp32,
                        false, false});
    const auto prefill_sample_result = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Prefill, 1, NumericalClass::ExactFp32,
                        false, true});
    const auto decode_result = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Decode, 1, NumericalClass::ExactFp32,
                        false, false});
    const auto decode_sample_result = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Decode, 1, NumericalClass::ExactFp32,
                        false, true});
    CHECK(std::holds_alternative<SemanticDispatchProgram>(prefill_result));
    CHECK(std::holds_alternative<SemanticDispatchProgram>(prefill_sample_result));
    CHECK(std::holds_alternative<SemanticDispatchProgram>(decode_result));
    CHECK(std::holds_alternative<SemanticDispatchProgram>(decode_sample_result));
    if (!std::holds_alternative<SemanticDispatchProgram>(prefill_result) ||
        !std::holds_alternative<SemanticDispatchProgram>(prefill_sample_result) ||
        !std::holds_alternative<SemanticDispatchProgram>(decode_result) ||
        !std::holds_alternative<SemanticDispatchProgram>(decode_sample_result)) return;
    const std::array<SemanticDispatchProgram, 4> programs = {
        std::get<SemanticDispatchProgram>(prefill_result),
        std::get<SemanticDispatchProgram>(prefill_sample_result),
        std::get<SemanticDispatchProgram>(decode_result),
        std::get<SemanticDispatchProgram>(decode_sample_result)};
    StructuralMetalLibraryIdentitySet libraries;
    libraries.identities = {library(StructuralMetalLibraryId::Core, 0x41)};
    MetalCodecCapabilityRegistry unknown;
    const SessionRequest request{4, 1, 0, true, true, false, false,
                                 NumericalClass::ExactFp32, RuntimeObjective::Latency};
    const auto result = compile_fixture_programs(
        fixture, programs, request, libraries, &unknown);
    CHECK(std::holds_alternative<CompatibilityReport>(result));
    if (std::holds_alternative<CompatibilityReport>(result))
        CHECK(std::get<CompatibilityReport>(result).code ==
              CompatibilityError::KERNEL_UNAVAILABLE);
}

void test_dense_program_compiles_to_bundles() {
    const Fixture fixture = make_structural_fixture(0x31);
    CHECK(fixture.package != nullptr);
    if (!fixture.package) return;
    const auto bindings_result = preflight_codec_bindings(*fixture.package);
    CHECK(std::holds_alternative<ResolvedCodecBindings>(bindings_result));
    if (!std::holds_alternative<ResolvedCodecBindings>(bindings_result)) return;
    const ResolvedCodecBindings bindings = std::get<ResolvedCodecBindings>(bindings_result);
    SessionRequest request;
    request.max_context = 4;
    request.max_batch = 2;
    request.enable_prefill = true;
    request.enable_decode = true;
    const auto prefill_result = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Prefill, 1, NumericalClass::ExactFp32, false, false});
    const auto prefill_sample_result = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Prefill, 1, NumericalClass::ExactFp32, false, true});
    const auto prefill_batch2_result = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Prefill, 2, NumericalClass::ExactFp32, false, false});
    const auto prefill_batch2_sample_result = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Prefill, 2, NumericalClass::ExactFp32, false, true});
    const auto decode_result = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Decode, 1, NumericalClass::ExactFp32, false, false});
    const auto decode_sample_result = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Decode, 1, NumericalClass::ExactFp32, false, true});
    CHECK(std::holds_alternative<SemanticDispatchProgram>(prefill_result));
    CHECK(std::holds_alternative<SemanticDispatchProgram>(prefill_sample_result));
    CHECK(std::holds_alternative<SemanticDispatchProgram>(prefill_batch2_result));
    CHECK(std::holds_alternative<SemanticDispatchProgram>(prefill_batch2_sample_result));
    CHECK(std::holds_alternative<SemanticDispatchProgram>(decode_result));
    CHECK(std::holds_alternative<SemanticDispatchProgram>(decode_sample_result));
    if (!std::holds_alternative<SemanticDispatchProgram>(prefill_result) ||
        !std::holds_alternative<SemanticDispatchProgram>(prefill_sample_result) ||
        !std::holds_alternative<SemanticDispatchProgram>(prefill_batch2_result) ||
        !std::holds_alternative<SemanticDispatchProgram>(prefill_batch2_sample_result) ||
        !std::holds_alternative<SemanticDispatchProgram>(decode_result) ||
        !std::holds_alternative<SemanticDispatchProgram>(decode_sample_result)) return;
    const std::array<SemanticDispatchProgram, 6> programs = {
        std::get<SemanticDispatchProgram>(prefill_result),
        std::get<SemanticDispatchProgram>(prefill_sample_result),
        std::get<SemanticDispatchProgram>(prefill_batch2_result),
        std::get<SemanticDispatchProgram>(prefill_batch2_sample_result),
        std::get<SemanticDispatchProgram>(decode_result),
        std::get<SemanticDispatchProgram>(decode_sample_result)};
    const auto bound_result = bind_dispatch_requirements(
        *fixture.package, bindings, request, programs);
    CHECK(std::holds_alternative<BoundDispatchRequirements>(bound_result));
    if (!std::holds_alternative<BoundDispatchRequirements>(bound_result)) return;
    StructuralMetalLibraryIdentitySet libraries;
    libraries.identities = {
        library(StructuralMetalLibraryId::Core, 0x41),
        library(StructuralMetalLibraryId::Prefill, 0x42),
        library(StructuralMetalLibraryId::Sampler, 0x43),
    };
    const MetalCodecCapabilityRegistry codec_capabilities =
        make_codec_capabilities(fixture.model, programs,
                                std::get<BoundDispatchRequirements>(bound_result),
                                fixture.package->physical_codec_registry().codecs);
    const auto compiled = compile_structural_metal(
        std::get<BoundDispatchRequirements>(bound_result), programs, fixture.model,
        fixture.package->physical_codec_registry().codecs, libraries,
        &codec_capabilities);
    CHECK_MSG(std::holds_alternative<StructuralMetalCompilation>(compiled),
              "compiler error=%u op=%u detail=%s",
              std::holds_alternative<CompatibilityReport>(compiled)
                  ? static_cast<unsigned>(std::get<CompatibilityReport>(compiled).code) : 0u,
              std::holds_alternative<CompatibilityReport>(compiled)
                  ? std::get<CompatibilityReport>(compiled).operator_id : 0u,
              std::holds_alternative<CompatibilityReport>(compiled)
                  ? std::get<CompatibilityReport>(compiled).detail.c_str() : "");
    if (!std::holds_alternative<StructuralMetalCompilation>(compiled)) return;
    const auto& output = std::get<StructuralMetalCompilation>(compiled);
    CHECK(output.programs().size() == 6);
    CHECK(!output.recipes().empty());
    CHECK(output.programs()[0].phase() == ExecutionPhase::Prefill);
    CHECK(output.programs()[1].phase() == ExecutionPhase::Prefill);
    CHECK(output.programs()[2].phase() == ExecutionPhase::Prefill);
    CHECK(output.programs()[3].phase() == ExecutionPhase::Prefill);
    CHECK(output.programs()[4].phase() == ExecutionPhase::Decode);
    CHECK(output.programs()[5].phase() == ExecutionPhase::Decode);
    CHECK(output.programs()[2].batch_rows() == 2);
    CHECK(output.programs()[0].groups().size() == 3);
    CHECK(output.programs()[1].groups().size() == 4);
    CHECK(output.programs()[2].groups().size() == 3);
    CHECK(output.programs()[3].groups().size() == 4);
    CHECK(output.programs()[4].groups().size() == 3);
    CHECK(output.programs()[5].groups().size() == 4);
    const auto& prefill_layer = output.programs()[0].groups()[1];
    CHECK(prefill_layer.kind() == StructuralMetalBundleGroupKind::Layer);
    CHECK(prefill_layer.shape() == StructuralMetalExecutionShape::DenseAttention);
    CHECK(prefill_layer.first_step() == 1);
    CHECK(prefill_layer.step_count() == 14);
    CHECK(prefill_layer.covered_step_ordinals().size() == 14);
    CHECK(prefill_layer.covered_operator_ids().size() == 14);
    CHECK(prefill_layer.coverage().size() == 14);
    const auto& two_row_final = output.programs()[2].groups()[2];
    CHECK(two_row_final.primitives().size() == 2);
    if (two_row_final.primitives().size() == 2)
        CHECK(two_row_final.primitives()[1].primitive ==
              StructuralMetalPrimitive::GemvF16);
    const auto& decode_layer = output.programs()[4].groups()[1];
    CHECK(decode_layer.kind() == StructuralMetalBundleGroupKind::Layer);
    CHECK(decode_layer.shape() == StructuralMetalExecutionShape::DenseAttention);
    CHECK(decode_layer.first_step() == 1);
    CHECK(decode_layer.step_count() == 14);
    CHECK(decode_layer.covered_step_ordinals().size() == 14);
    CHECK(decode_layer.primitives().size() > decode_layer.step_count());
    size_t rope_invocations = 0;
    size_t kv_write_invocations = 0;
    for (const auto& invocation : decode_layer.primitives())
        if (invocation.primitive == StructuralMetalPrimitive::RopeHalfSplit)
            ++rope_invocations;
        else if (invocation.primitive == StructuralMetalPrimitive::KvWrite)
            ++kv_write_invocations;
    CHECK(rope_invocations == 2);
    CHECK(kv_write_invocations == 2);
    CHECK(output.programs()[4].groups()[0].shape() ==
          StructuralMetalExecutionShape::GraphEmbedding);
    CHECK(output.programs()[4].groups()[2].shape() ==
          StructuralMetalExecutionShape::FinalOutput);
    CHECK(output.programs()[5].groups()[3].shape() ==
          StructuralMetalExecutionShape::GreedySampler);
    CHECK(output.programs()[5].steps().back().primitives().size() == 1);
    CHECK(output.programs()[5].steps().back().primitives()[0].primitive ==
          StructuralMetalPrimitive::SamplerGreedy);

    const auto expect_rejected = [&](auto mutate) {
        auto candidate = programs;
        mutate(candidate);
        const auto rejected = compile_structural_metal(
            std::get<BoundDispatchRequirements>(bound_result), candidate,
            fixture.model, fixture.package->physical_codec_registry().codecs,
            libraries, &codec_capabilities);
        CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    };
    expect_rejected([](auto& candidate) {
        --candidate[1].layer_views[0].step_count;
    });
    expect_rejected([](auto& candidate) {
        candidate[1].layer_views[0].first_step = 0;
    });
    expect_rejected([](auto& candidate) {
        candidate[1].steps[5].covered_operator_ids[0] =
            candidate[1].steps[6].operator_id;
    });
    expect_rejected([](auto& candidate) {
        std::swap(candidate[1].steps[5], candidate[1].steps[6]);
    });
}

void test_recurrent_group_owns_every_fused_step() {
    Fixture fixture = make_raw_shape_fixture(recurrent_shape_model());
    CHECK(fixture.package != nullptr);
    if (!fixture.package) return;
    const auto bindings_result = preflight_codec_bindings(*fixture.package);
    CHECK(std::holds_alternative<ResolvedCodecBindings>(bindings_result));
    if (!std::holds_alternative<ResolvedCodecBindings>(bindings_result)) return;
    const SessionRequest session{32, 1, 0, false, true, false, false,
                                 NumericalClass::ExactFp32, RuntimeObjective::Latency};
    const auto program_result = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Decode, 1, NumericalClass::ExactFp32, false, false});
    CHECK(std::holds_alternative<SemanticDispatchProgram>(program_result));
    if (!std::holds_alternative<SemanticDispatchProgram>(program_result)) return;
    const SemanticDispatchProgram program = std::get<SemanticDispatchProgram>(program_result);
    const std::array<SemanticDispatchProgram, 1> programs = {program};
    const auto bound_result = bind_dispatch_requirements(
        *fixture.package, std::get<ResolvedCodecBindings>(bindings_result), session, programs);
    CHECK(std::holds_alternative<BoundDispatchRequirements>(bound_result));
    if (!std::holds_alternative<BoundDispatchRequirements>(bound_result)) return;
    StructuralMetalLibraryIdentitySet libraries;
    libraries.identities = {library(StructuralMetalLibraryId::Core, 0x51)};
    const MetalCodecCapabilityRegistry codec_capabilities =
        make_codec_capabilities(fixture.model, programs,
                                std::get<BoundDispatchRequirements>(bound_result),
                                fixture.package->physical_codec_registry().codecs);
    const auto compiled = compile_structural_metal(
        std::get<BoundDispatchRequirements>(bound_result), programs, fixture.model,
        fixture.package->physical_codec_registry().codecs, libraries,
        &codec_capabilities);
    CHECK_MSG(std::holds_alternative<StructuralMetalCompilation>(compiled),
              "recurrent compiler error=%u detail=%s",
              std::holds_alternative<CompatibilityReport>(compiled)
                  ? static_cast<unsigned>(std::get<CompatibilityReport>(compiled).code) : 0u,
              std::holds_alternative<CompatibilityReport>(compiled)
                  ? std::get<CompatibilityReport>(compiled).detail.c_str() : "");
    if (!std::holds_alternative<StructuralMetalCompilation>(compiled)) return;
    const auto& output = std::get<StructuralMetalCompilation>(compiled);
    CHECK(output.programs().size() == 1);
    CHECK(output.programs()[0].groups().size() == 3);
    const auto& layer = output.programs()[0].groups()[1];
    CHECK(layer.shape() == StructuralMetalExecutionShape::RecurrentDelta);
    CHECK(layer.first_step() == 1);
    CHECK(layer.step_count() == 18);
    CHECK(layer.coverage().size() == 18);
    CHECK(layer.covered_operator_ids().size() == 18);
    size_t owners = 0;
    size_t covered = 0;
    size_t l2_invocations = 0;
    for (const auto& invocation : layer.primitives())
        l2_invocations += invocation.primitive == StructuralMetalPrimitive::DnetL2;
    for (const auto& record : layer.coverage()) {
        owners += record.kind == StructuralMetalCoverageKind::FusedOwner;
        covered += record.kind == StructuralMetalCoverageKind::FusedCovered;
    }
    CHECK(l2_invocations == 1);
    CHECK(owners == 1);
    CHECK(covered == 2);
    CHECK(layer.owner_step_ordinal() == 9);
    CHECK(layer.primitives().size() == layer.step_count() - 2);
}

void test_moe_parallel_group_and_rewired_serial_rejection() {
    Fixture fixture = make_raw_shape_fixture(moe_shape_model());
    CHECK(fixture.package != nullptr);
    if (!fixture.package) return;
    const auto bindings_result = preflight_codec_bindings(*fixture.package);
    CHECK(std::holds_alternative<ResolvedCodecBindings>(bindings_result));
    if (!std::holds_alternative<ResolvedCodecBindings>(bindings_result)) return;
    const SessionRequest session{4, 1, 0, false, true, false, false,
                                 NumericalClass::ExactFp32, RuntimeObjective::Latency};
    const auto program_result = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Decode, 1, NumericalClass::ExactFp32, false, false});
    CHECK(std::holds_alternative<SemanticDispatchProgram>(program_result));
    if (!std::holds_alternative<SemanticDispatchProgram>(program_result)) return;
    const SemanticDispatchProgram program = std::get<SemanticDispatchProgram>(program_result);
    const std::array<SemanticDispatchProgram, 1> programs = {program};
    const auto bound_result = bind_dispatch_requirements(
        *fixture.package, std::get<ResolvedCodecBindings>(bindings_result), session, programs);
    CHECK(std::holds_alternative<BoundDispatchRequirements>(bound_result));
    if (!std::holds_alternative<BoundDispatchRequirements>(bound_result)) return;
    StructuralMetalLibraryIdentitySet libraries;
    libraries.identities = {library(StructuralMetalLibraryId::Core, 0x61)};
    const MetalCodecCapabilityRegistry codec_capabilities =
        make_codec_capabilities(fixture.model, programs,
                                std::get<BoundDispatchRequirements>(bound_result),
                                fixture.package->physical_codec_registry().codecs);
    const auto compiled = compile_structural_metal(
        std::get<BoundDispatchRequirements>(bound_result), programs, fixture.model,
        fixture.package->physical_codec_registry().codecs, libraries,
        &codec_capabilities);
    CHECK_MSG(std::holds_alternative<StructuralMetalCompilation>(compiled),
              "MoE compiler error=%u op=%u detail=%s",
              std::holds_alternative<CompatibilityReport>(compiled)
                  ? static_cast<unsigned>(std::get<CompatibilityReport>(compiled).code) : 0u,
              std::holds_alternative<CompatibilityReport>(compiled)
                  ? std::get<CompatibilityReport>(compiled).operator_id : 0u,
              std::holds_alternative<CompatibilityReport>(compiled)
                  ? std::get<CompatibilityReport>(compiled).detail.c_str() : "");
    if (!std::holds_alternative<StructuralMetalCompilation>(compiled)) return;
    const auto& output = std::get<StructuralMetalCompilation>(compiled);
    CHECK(output.programs().size() == 1);
    CHECK(output.programs()[0].groups().size() == 3);
    const auto& layer = output.programs()[0].groups()[1];
    CHECK(layer.shape() == StructuralMetalExecutionShape::MoeRoutedFeedForward);
    CHECK(layer.first_step() == 1);
    CHECK(layer.step_count() == 26);
    CHECK(layer.coverage().size() == 26);
    CHECK(layer.covered_operator_ids().size() == 26);
    size_t router = 0;
    size_t expert_activation = 0;
    size_t reduce = 0;
    size_t axis_split = 0;
    size_t vector_scale = 0;
    size_t vector_multiply = 0;
    size_t apply_down_scale = layer.primitives().size();
    size_t expert_reduce = layer.primitives().size();
    size_t primitive_index = 0;
    for (const auto& invocation : layer.primitives()) {
        router += invocation.primitive == StructuralMetalPrimitive::RouterTopK;
        expert_activation += invocation.primitive == StructuralMetalPrimitive::GatedActivationExperts;
        reduce += invocation.primitive == StructuralMetalPrimitive::ExpertReduce;
        axis_split += invocation.primitive == StructuralMetalPrimitive::AxisSplit;
        vector_scale += invocation.primitive == StructuralMetalPrimitive::VecScale;
        vector_multiply += invocation.primitive == StructuralMetalPrimitive::VecMul;
        if (invocation.primitive == StructuralMetalPrimitive::ApplyDownScale)
            apply_down_scale = primitive_index;
        if (invocation.primitive == StructuralMetalPrimitive::ExpertReduce)
            expert_reduce = primitive_index;
        ++primitive_index;
    }
    CHECK(router == 1);
    CHECK(expert_activation == 1);
    CHECK(reduce == 1);
    CHECK(axis_split == 0);
    CHECK(vector_scale == 1);
    CHECK(vector_multiply == 1);
    CHECK(apply_down_scale + 1 == expert_reduce);
    CHECK(recipe_named(output, "gemv_q4k_id") != nullptr);
    const std::array<MetalFunctionConstant, 1> f32 = {
        MetalFunctionConstant{0, MetalFunctionConstantType::Int32, 0},
    };
    const MetalPipelineRecipe* router_recipe = recipe_named(output, "gemv", f32);
    CHECK(router_recipe != nullptr);
    if (router_recipe)
        CHECK(router_recipe->dispatch.min_threads_per_threadgroup == 32);

    auto rewired_model = fixture.model;
    // Preserve the operation count and serialized order, but make the dense
    // branch consume its own residual result.  The exact MoE contract must
    // reject this serial-vs-parallel rewire.
    const auto branch_merge = std::find_if(
        rewired_model.operators.begin(), rewired_model.operators.end(),
        [](const SemanticOperator& operation) {
            return operation.kind == OperatorKind::Add && operation.inputs.size() == 2 &&
                   operation.inputs[0] == 14;
        });
    CHECK(branch_merge != rewired_model.operators.end());
    if (branch_merge != rewired_model.operators.end()) branch_merge->inputs = {9, 14};
    const auto rejected = compile_structural_metal(
        std::get<BoundDispatchRequirements>(bound_result), programs, rewired_model,
        fixture.package->physical_codec_registry().codecs, libraries,
        &codec_capabilities);
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
}

} // namespace

int main() {
    test_library_identity_is_required();
    test_structural_codec_selection_accepts_portable_registry();
    test_recipe_digest_binds_contract();
    test_equivalent_codec_declarations_share_structural_lowering();
    test_unknown_valid_codec_fails_closed();
    test_dense_program_compiles_to_bundles();
    test_recurrent_group_owns_every_fused_step();
    test_moe_parallel_group_and_rewired_serial_rejection();
    return test_summary("test_structural_metal_compiler");
}
