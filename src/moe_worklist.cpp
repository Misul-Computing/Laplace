#include "moe_worklist.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace Laplace {

namespace {

CompatibilityReport worklist_error(CompatibilityError code, std::string detail) {
    CompatibilityReport report = package_report(code, std::move(detail));
    report.stage = CompatibilityStage::Plan;
    report.phase = CompatibilityPhase::Plan;
    return report;
}

bool digest_is_zero(const Sha256Digest& digest) {
    return std::all_of(digest.bytes.begin(), digest.bytes.end(), [](uint8_t byte) {
        return byte == 0;
    });
}

bool checked_multiply(uint64_t left, uint64_t right, uint64_t& result) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) return false;
    result = left * right;
    return true;
}

bool checked_add(uint64_t left, uint64_t right, uint64_t& result) {
    if (right > std::numeric_limits<uint64_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool align_up(uint64_t value, uint64_t alignment, uint64_t& result) {
    if (alignment == 0 || value > std::numeric_limits<uint64_t>::max() - alignment + 1) return false;
    result = (value + alignment - 1) / alignment * alignment;
    return true;
}

bool append_section(uint64_t count, uint64_t element_size, uint64_t& cursor,
                    uint64_t& offset) {
    if (!align_up(cursor, kMoeWorklistPayloadAlignment, offset)) return false;
    uint64_t bytes = 0;
    uint64_t end = 0;
    return checked_multiply(count, element_size, bytes) &&
           checked_add(offset, bytes, end) && (cursor = end, true);
}

bool build_device_layout(uint32_t token_capacity, uint32_t expert_count,
                         uint32_t selected_count, MoeWorklistDeviceLayout& layout) {
    if (token_capacity == 0 || expert_count == 0 || selected_count == 0) return false;
    uint64_t route_capacity = 0;
    if (!checked_multiply(token_capacity, selected_count, route_capacity) ||
        route_capacity > UINT32_MAX) return false;

    layout = {};
    layout.token_capacity = token_capacity;
    layout.route_capacity = static_cast<uint32_t>(route_capacity);
    layout.expert_capacity = expert_count;
    layout.source_span_capacity = kMoeWorklistMaximumSourceSpans;
    uint64_t cursor = 0;
    if (!append_section(route_capacity, sizeof(uint32_t), cursor, layout.routed_token_offset) ||
        !append_section(route_capacity, sizeof(uint32_t), cursor, layout.logical_expert_offset) ||
        !append_section(route_capacity, sizeof(uint32_t), cursor, layout.route_weight_offset) ||
        !append_section(expert_count, sizeof(uint32_t), cursor, layout.segment_count_offset) ||
        !append_section(expert_count, sizeof(uint32_t), cursor, layout.segment_offset_offset) ||
        !append_section(expert_count, sizeof(MoeWorklistIndirectDispatch), cursor,
                        layout.indirect_dispatch_offset) ||
        !append_section(kMoeWorklistMaximumSourceSpans, sizeof(MoeWorklistSourceSpanRef),
                        cursor, layout.source_span_offset) ||
        !align_up(cursor, kMoeWorklistPayloadAlignment, layout.byte_size)) return false;
    return layout.byte_size != 0;
}

const SemanticValue* value_for(const SemanticModel& model, uint32_t id) {
    for (const SemanticValue& value : model.values) if (value.id == id) return &value;
    return nullptr;
}

const SemanticTensor* tensor_for(const SemanticModel& model, uint32_t id) {
    for (const SemanticTensor& tensor : model.tensors) if (tensor.id == id) return &tensor;
    return nullptr;
}

std::vector<const SemanticOperator*> layer_operators(const SemanticModel& model,
                                                     const SemanticLayer& layer) {
    if (layer.first_operator > model.operators.size() ||
        layer.operator_count > model.operators.size() - layer.first_operator) return {};
    std::vector<const SemanticOperator*> result;
    result.reserve(layer.operator_count);
    for (uint32_t index = 0; index != layer.operator_count; ++index)
        result.push_back(&model.operators[layer.first_operator + index]);
    return result;
}

template<class Predicate>
const SemanticOperator* unique_operator(const std::vector<const SemanticOperator*>& operators,
                                        Predicate predicate, bool& ambiguous) {
    const SemanticOperator* result = nullptr;
    ambiguous = false;
    for (const SemanticOperator* op : operators) {
        if (!predicate(*op)) continue;
        if (result) {
            ambiguous = true;
            return nullptr;
        }
        result = op;
    }
    return result;
}

bool has_tensor_role(const SemanticModel& model, const SemanticOperator& op,
                     TensorRole role, const SemanticTensor*& tensor) {
    if (op.tensors.size() != 1) return false;
    tensor = tensor_for(model, op.tensors[0]);
    return tensor && tensor->role == role;
}

bool dimensions_prefix_equal(const std::vector<Dimension>& left,
                             const std::vector<Dimension>& right, size_t count) {
    if (left.size() < count || right.size() < count) return false;
    for (size_t index = 0; index != count; ++index)
        if (left[index] != right[index]) return false;
    return true;
}

bool last_constant_width(const SemanticValue& value, ScalarType type,
                         uint32_t width) {
    return value.logical_type == type && !value.dimensions.empty() &&
           value.dimensions.back() == Dimension{DimensionKind::Constant, width};
}

bool routed_shapes(const SemanticModel& model, const SemanticOperator& router,
                   const SemanticOperator& gate_up, const SemanticOperator& split,
                   const SemanticOperator& activation, const SemanticOperator& down,
                   const SemanticOperator& reduce, uint32_t hidden,
                   uint32_t intermediate, uint32_t experts, uint32_t selected) {
    const SemanticValue* scores = value_for(model, router.inputs[0]);
    const SemanticValue* ids = value_for(model, router.outputs[0]);
    const SemanticValue* weights = value_for(model, router.outputs[1]);
    const SemanticValue* gate_input = value_for(model, gate_up.inputs[0]);
    const SemanticValue* gate_output = value_for(model, gate_up.outputs[0]);
    const SemanticValue* split_gate = value_for(model, split.outputs[0]);
    const SemanticValue* split_up = value_for(model, split.outputs[1]);
    const SemanticValue* activated = value_for(model, activation.outputs[0]);
    const SemanticValue* down_output = value_for(model, down.outputs[0]);
    const SemanticValue* reduced = value_for(model, reduce.outputs[0]);
    if (!scores || !ids || !weights || !gate_input || !gate_output || !split_gate ||
        !split_up || !activated || !down_output || !reduced ||
        !last_constant_width(*scores, ScalarType::F32, experts) ||
        !last_constant_width(*ids, ScalarType::U32, selected) ||
        !last_constant_width(*weights, ScalarType::F32, selected) ||
        ids->dimensions != weights->dimensions ||
        !dimensions_prefix_equal(scores->dimensions, ids->dimensions, ids->dimensions.size() - 1) ||
        !dimensions_prefix_equal(gate_input->dimensions, ids->dimensions, ids->dimensions.size() - 1) ||
        gate_input->dimensions.back() != Dimension{DimensionKind::Constant, hidden}) return false;

    const std::vector<Dimension> routed_prefix(ids->dimensions.begin(), ids->dimensions.end() - 1);
    const auto routed = [&](const SemanticValue& value, uint32_t width) {
        std::vector<Dimension> expected = routed_prefix;
        expected.push_back({DimensionKind::Constant, selected});
        expected.push_back({DimensionKind::Constant, width});
        return value.logical_type == ScalarType::F32 && value.dimensions == expected;
    };
    if (!routed(*gate_output, intermediate * 2) ||
        !routed(*split_gate, intermediate) || !routed(*split_up, intermediate) ||
        !routed(*activated, intermediate) || !routed(*down_output, hidden) ||
        reduced->logical_type != ScalarType::F32 ||
        !([&] {
            std::vector<Dimension> expected = routed_prefix;
            expected.push_back({DimensionKind::Constant, hidden});
            return reduced->dimensions == expected;
        }()) ||
        split.inputs != std::vector<uint32_t>{gate_up.outputs[0]} ||
        activation.inputs != split.outputs ||
        down.inputs != std::vector<uint32_t>{activation.outputs[0], router.outputs[0],
                                             router.outputs[1]} ||
        reduce.inputs != std::vector<uint32_t>{down.outputs[0], router.outputs[0],
                                               router.outputs[1]}) return false;
    return true;
}

bool valid_expert_tensor(const SemanticTensor& tensor, const PhysicalCodecIdentity& identity,
                         TensorRole role, uint32_t experts, uint32_t input,
                         uint32_t output) {
    const std::array<uint8_t, 8> logical_order = {0, 1, 2, 0xff, 0xff, 0xff, 0xff, 0xff};
    const std::array<uint8_t, 8> gguf_order = {1, 2, 0, 0xff, 0xff, 0xff, 0xff, 0xff};
    const bool supported_order = tensor.layout.axis_order == logical_order ||
                                  tensor.layout.axis_order == gguf_order;
    const bool supported_strides = tensor.layout.axis_order == logical_order
        ? tensor.layout.strides[1] == output &&
          tensor.layout.strides[2] == static_cast<uint64_t>(input) * output
        : tensor.layout.strides[1] == input &&
          tensor.layout.strides[2] == static_cast<uint64_t>(input) * output;
    if (tensor.role != role || tensor.logical_type != ScalarType::F32 ||
        tensor.dimensions != std::vector<Dimension>{{DimensionKind::Constant, experts},
                                                     {DimensionKind::Constant, input},
                                                     {DimensionKind::Constant, output}} ||
        tensor.layout.rank != 3 || !supported_order || tensor.layout.strides[0] != 1 ||
        !supported_strides ||
        std::any_of(tensor.layout.strides.begin() + 3, tensor.layout.strides.end(),
                    [](uint64_t stride) { return stride != 0; }) ||
        tensor.expert_axis.kind != ExpertAxisKind::ExpertBank ||
        tensor.expert_axis.expert_axis != 0 || tensor.expert_axis.member_axis != 0xff ||
        tensor.expert_axis.input_axis != 1 || tensor.expert_axis.output_axis != 2 ||
        tensor.expert_axis.expert_count != experts || tensor.expert_axis.per_expert_byte_stride == 0 ||
        !valid_physical_codec_identity(identity)) return false;
    const auto expected = physical_codec_identity(
        tensor, identity.arithmetic_version, identity.arithmetic_digest,
        identity.codebook_digest);
    return expected && *expected == identity;
}

bool valid_aux_tensor(const SemanticTensor& tensor, TensorRole role, uint32_t width) {
    return tensor.role == role && tensor.logical_type == ScalarType::F32 &&
           tensor.dimensions == std::vector<Dimension>{{DimensionKind::Constant, width}} &&
           tensor.layout.rank == 1 && tensor.layout.axis_order[0] == 0 &&
           tensor.layout.strides[0] == 1 &&
           std::all_of(tensor.layout.strides.begin() + 1, tensor.layout.strides.end(),
                       [](uint64_t stride) { return stride == 0; }) &&
           tensor.planes.size() == 1 && tensor.planes[0].kind == PlaneKind::Values &&
           tensor.planes[0].storage_type == ScalarType::F32 && tensor.planes[0].length != 0;
}

const PhysicalCodecIdentity* registry_identity(const PhysicalCodecRegistry& registry,
                                               uint32_t tensor_id) {
    for (const PhysicalTensorCodecDeclaration& declaration : registry.tensors)
        if (declaration.tensor_id == tensor_id) return &declaration.identity;
    return nullptr;
}

PhysicalLayoutSchema layout_schema(const PhysicalLayout& layout) {
    return {layout.kind, layout.version, layout.packing, layout.rank, layout.block_rank,
            layout.axis_order, layout.block_elements, layout.block_bytes, layout.flags};
}

PhysicalQuantizationSchema quantization_schema(const Quantization& quantization) {
    return {quantization.kind, quantization.version, quantization.accumulation_type,
            quantization.scale_type, quantization.zero_type, quantization.bias_type,
            quantization.block_elements, quantization.block_bytes, quantization.group_size,
            quantization.required_plane_mask, quantization.flags};
}

MoeWorklistTensorContract tensor_contract(const SemanticTensor& tensor,
                                          const PhysicalCodecIdentity& identity) {
    MoeWorklistTensorContract result;
    result.input_width = tensor.expert_axis.input_axis < tensor.dimensions.size()
        ? static_cast<uint32_t>(tensor.dimensions[tensor.expert_axis.input_axis].constant_or_symbol) : 0;
    result.output_width = tensor.expert_axis.output_axis < tensor.dimensions.size()
        ? static_cast<uint32_t>(tensor.dimensions[tensor.expert_axis.output_axis].constant_or_symbol) : 0;
    result.expert_count = tensor.expert_axis.expert_count;
    result.dimensions = tensor.dimensions;
    result.strides = tensor.layout.strides;
    result.layout = layout_schema(tensor.layout);
    result.quantization = quantization_schema(tensor.quantization);
    result.planes = identity.planes;
    result.expert_stride = tensor.expert_axis.per_expert_byte_stride;
    return result;
}

bool contract_matches(const MoeWorklistTensorContract& contract,
                      const PhysicalCodecIdentity& identity, uint32_t experts,
                      uint32_t input, uint32_t output) {
    const std::array<uint8_t, 8> logical_order = {0, 1, 2, 0xff, 0xff, 0xff, 0xff, 0xff};
    const std::array<uint8_t, 8> gguf_order = {1, 2, 0, 0xff, 0xff, 0xff, 0xff, 0xff};
    const bool logical_strides = contract.layout.axis_order == logical_order &&
        contract.strides[0] == 1 && contract.strides[1] == output &&
        contract.strides[2] == static_cast<uint64_t>(input) * output;
    const bool gguf_strides = contract.layout.axis_order == gguf_order &&
        contract.strides[0] == 1 && contract.strides[1] == input &&
        contract.strides[2] == static_cast<uint64_t>(input) * output;
    return contract.input_width == input && contract.output_width == output &&
           contract.expert_count == experts && contract.dimensions ==
               std::vector<Dimension>{{DimensionKind::Constant, experts},
                                       {DimensionKind::Constant, input},
                                       {DimensionKind::Constant, output}} &&
           (logical_strides || gguf_strides) &&
           std::all_of(contract.strides.begin() + 3, contract.strides.end(),
                       [](uint64_t stride) { return stride == 0; }) &&
           contract.layout == identity.layout && contract.quantization == identity.quantization &&
           contract.planes == identity.planes && contract.expert_stride != 0;
}

bool valid_descriptor(const MoeWorklistDescriptor& descriptor) {
    if (descriptor.version != 1 ||
        (descriptor.phase != ExecutionPhase::Prefill && descriptor.phase != ExecutionPhase::Decode) ||
        descriptor.expert_count == 0 || descriptor.selected_count == 0 ||
        descriptor.selected_count > descriptor.expert_count || descriptor.hidden == 0 ||
        descriptor.intermediate == 0 || descriptor.intermediate > UINT32_MAX / 2 ||
        std::find(descriptor.semantic_operator_ids.begin(), descriptor.semantic_operator_ids.end(), UINT32_MAX) !=
            descriptor.semantic_operator_ids.end() ||
        descriptor.gate_up.input_width != descriptor.hidden ||
        descriptor.gate_up.output_width != descriptor.intermediate * 2 ||
        descriptor.down.input_width != descriptor.intermediate ||
        descriptor.down.output_width != descriptor.hidden || descriptor.physical_codecs.size() != 3) {
        return false;
    }
    for (size_t left = 0; left != descriptor.semantic_operator_ids.size(); ++left) {
        const uint32_t id = descriptor.semantic_operator_ids[left];
        if (id == UINT32_MAX) return false;
        for (size_t right = left + 1; right != descriptor.semantic_operator_ids.size(); ++right)
            if (id == descriptor.semantic_operator_ids[right]) return false;
    }
    if (descriptor.activation != ActivationKind::Silu && descriptor.activation != ActivationKind::GeluTanh)
        return false;
    if (descriptor.scale_source != ExpertScaleSource::PerExpertTensor) return false;
    if (descriptor.value_source == ValueSource::KeyStateAlias ||
        (descriptor.value_source != ValueSource::SeparateProjection &&
         descriptor.value_source != ValueSource::KeyPreRope &&
         descriptor.value_source != ValueSource::KeyPostRope)) return false;
    if (descriptor.physical_codecs.size() != 3 ||
        descriptor.physical_codecs[0] != descriptor.physical_codecs[1]) return false;
    for (const PhysicalCodecIdentity& identity : descriptor.physical_codecs)
        if (!valid_physical_codec_identity(identity)) return false;
    if (!contract_matches(descriptor.gate_up, descriptor.physical_codecs[0],
                          descriptor.expert_count, descriptor.hidden,
                          descriptor.intermediate * 2) ||
        !contract_matches(descriptor.down, descriptor.physical_codecs[2],
                          descriptor.expert_count, descriptor.intermediate,
                          descriptor.hidden)) return false;
    const std::array<uint32_t, 5> tensors = {
        descriptor.router_scale_tensor_id, descriptor.expert_norm_tensor_id,
        descriptor.reduce_scale_tensor_id, descriptor.post_norm_tensor_id,
        descriptor.output_scale_tensor_id};
    if (tensors[0] == UINT32_MAX || tensors[1] == UINT32_MAX || tensors[2] == UINT32_MAX) return false;
    for (size_t left = 0; left != tensors.size(); ++left) {
        if (tensors[left] == UINT32_MAX) continue;
        for (size_t right = left + 1; right != tensors.size(); ++right)
            if (tensors[left] == tensors[right]) return false;
    }
    return true;
}

bool descriptor_layout_matches(const MoeWorklistPlan& plan) {
    MoeWorklistDeviceLayout expected;
    if (!build_device_layout(plan.device_layout().token_capacity,
                             plan.descriptor().expert_count,
                             plan.descriptor().selected_count, expected)) return false;
    return expected == plan.device_layout();
}

void append_u16(std::vector<uint8_t>& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8));
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void append_identity(std::vector<uint8_t>& bytes, const PhysicalCodecIdentity& identity) {
    append_u16(bytes, identity.identity_version);
    append_u16(bytes, identity.arithmetic_version);
    bytes.insert(bytes.end(), identity.arithmetic_digest.begin(), identity.arithmetic_digest.end());
    bytes.insert(bytes.end(), identity.codebook_digest.begin(), identity.codebook_digest.end());
    append_u16(bytes, static_cast<uint16_t>(identity.layout.kind));
    append_u16(bytes, identity.layout.version);
    append_u16(bytes, static_cast<uint16_t>(identity.layout.packing));
    bytes.push_back(identity.layout.rank);
    bytes.push_back(identity.layout.block_rank);
    for (uint8_t axis : identity.layout.axis_order) bytes.push_back(axis);
    append_u32(bytes, identity.layout.block_elements);
    append_u32(bytes, identity.layout.block_bytes);
    append_u32(bytes, identity.layout.flags);
    append_u16(bytes, static_cast<uint16_t>(identity.quantization.kind));
    append_u16(bytes, identity.quantization.version);
    append_u16(bytes, static_cast<uint16_t>(identity.quantization.accumulation_type));
    append_u16(bytes, static_cast<uint16_t>(identity.quantization.scale_type));
    append_u16(bytes, static_cast<uint16_t>(identity.quantization.zero_type));
    append_u16(bytes, static_cast<uint16_t>(identity.quantization.bias_type));
    append_u32(bytes, identity.quantization.block_elements);
    append_u32(bytes, identity.quantization.block_bytes);
    append_u32(bytes, identity.quantization.group_size);
    append_u32(bytes, identity.quantization.required_plane_mask);
    append_u32(bytes, identity.quantization.flags);
    append_u32(bytes, static_cast<uint32_t>(identity.planes.size()));
    for (const PhysicalPlaneSchema& plane : identity.planes) {
        append_u16(bytes, static_cast<uint16_t>(plane.kind));
        append_u16(bytes, static_cast<uint16_t>(plane.storage_type));
        append_u32(bytes, plane.logical_elements_covered);
        append_u32(bytes, plane.bytes_per_block);
        append_u32(bytes, plane.flags);
    }
}

void append_tensor_contract(std::vector<uint8_t>& bytes,
                            const MoeWorklistTensorContract& contract) {
    append_u32(bytes, contract.input_width);
    append_u32(bytes, contract.output_width);
    append_u32(bytes, contract.expert_count);
    append_u32(bytes, static_cast<uint32_t>(contract.dimensions.size()));
    for (const Dimension& dimension : contract.dimensions) {
        bytes.push_back(static_cast<uint8_t>(dimension.kind));
        append_u64(bytes, dimension.constant_or_symbol);
    }
    for (uint64_t stride : contract.strides) append_u64(bytes, stride);
    append_u16(bytes, static_cast<uint16_t>(contract.layout.kind));
    append_u16(bytes, contract.layout.version);
    append_u16(bytes, static_cast<uint16_t>(contract.layout.packing));
    bytes.push_back(contract.layout.rank);
    bytes.push_back(contract.layout.block_rank);
    for (uint8_t axis : contract.layout.axis_order) bytes.push_back(axis);
    append_u32(bytes, contract.layout.block_elements);
    append_u32(bytes, contract.layout.block_bytes);
    append_u32(bytes, contract.layout.flags);
    append_u16(bytes, static_cast<uint16_t>(contract.quantization.kind));
    append_u16(bytes, contract.quantization.version);
    append_u16(bytes, static_cast<uint16_t>(contract.quantization.accumulation_type));
    append_u16(bytes, static_cast<uint16_t>(contract.quantization.scale_type));
    append_u16(bytes, static_cast<uint16_t>(contract.quantization.zero_type));
    append_u16(bytes, static_cast<uint16_t>(contract.quantization.bias_type));
    append_u32(bytes, contract.quantization.block_elements);
    append_u32(bytes, contract.quantization.block_bytes);
    append_u32(bytes, contract.quantization.group_size);
    append_u32(bytes, contract.quantization.required_plane_mask);
    append_u32(bytes, contract.quantization.flags);
    append_u32(bytes, static_cast<uint32_t>(contract.planes.size()));
    for (const PhysicalPlaneSchema& plane : contract.planes) {
        append_u16(bytes, static_cast<uint16_t>(plane.kind));
        append_u16(bytes, static_cast<uint16_t>(plane.storage_type));
        append_u32(bytes, plane.logical_elements_covered);
        append_u32(bytes, plane.bytes_per_block);
        append_u32(bytes, plane.flags);
    }
    append_u64(bytes, contract.expert_stride);
}

Sha256Digest plan_digest(const SemanticModel& model, const SemanticLayer& layer,
                         const MoeWorklistDescriptor& descriptor,
                         std::span<const MoeWorklistSourceSpanRef> spans,
                         const PhysicalCodecRegistry& registry,
                         uint32_t token_capacity) {
    const Sha256Digest semantic = semantic_model_digest(model);
    const Sha256Digest physical = physical_codec_registry_digest(registry, model);
    Sha256Digest result;
    if (digest_is_zero(semantic) || digest_is_zero(physical)) return result;
    std::vector<uint8_t> bytes;
    constexpr char domain[] = "laplace-moe-worklist-v1";
    bytes.insert(bytes.end(), domain, domain + sizeof(domain));
    append_u32(bytes, token_capacity);
    append_u32(bytes, layer.layer_index);
    append_u32(bytes, layer.first_operator);
    append_u32(bytes, layer.operator_count);
    append_u32(bytes, layer.flags);
    bytes.insert(bytes.end(), semantic.bytes.begin(), semantic.bytes.end());
    bytes.insert(bytes.end(), physical.bytes.begin(), physical.bytes.end());
    append_u16(bytes, descriptor.version);
    append_u16(bytes, static_cast<uint16_t>(descriptor.phase));
    for (uint32_t id : descriptor.semantic_operator_ids) append_u32(bytes, id);
    append_u32(bytes, descriptor.expert_count);
    append_u32(bytes, descriptor.selected_count);
    append_u32(bytes, descriptor.hidden);
    append_u32(bytes, descriptor.intermediate);
    bytes.push_back(static_cast<uint8_t>(descriptor.activation));
    bytes.push_back(static_cast<uint8_t>(descriptor.scale_source));
    bytes.push_back(static_cast<uint8_t>(descriptor.value_source));
    append_tensor_contract(bytes, descriptor.gate_up);
    append_tensor_contract(bytes, descriptor.down);
    append_u32(bytes, static_cast<uint32_t>(descriptor.physical_codecs.size()));
    for (const PhysicalCodecIdentity& identity : descriptor.physical_codecs)
        append_identity(bytes, identity);
    append_u32(bytes, descriptor.router_scale_tensor_id);
    append_u32(bytes, descriptor.expert_norm_tensor_id);
    append_u32(bytes, descriptor.reduce_scale_tensor_id);
    append_u32(bytes, descriptor.post_norm_tensor_id);
    append_u32(bytes, descriptor.output_scale_tensor_id);
    append_u32(bytes, static_cast<uint32_t>(spans.size()));
    for (const MoeWorklistSourceSpanRef& span : spans) {
        append_u32(bytes, span.tensor_id);
        append_u16(bytes, static_cast<uint16_t>(span.plane));
        append_u16(bytes, span.reserved);
        append_u64(bytes, span.offset);
        append_u64(bytes, span.length);
    }
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), result.bytes.data());
    return result;
}

bool source_spans_valid(const MoeWorklistPlan& plan) {
    const auto spans = plan.source_spans();
    if (spans.empty() || spans.size() > kMoeWorklistMaximumSourceSpans) return false;
    for (size_t left = 0; left != spans.size(); ++left) {
        const MoeWorklistSourceSpanRef& span = spans[left];
        if (span.tensor_id == UINT32_MAX || span.reserved != 0 || span.length == 0 ||
            span.offset > UINT64_MAX - span.length) return false;
        for (size_t right = left + 1; right != spans.size(); ++right)
            if (span.tensor_id == spans[right].tensor_id && span.plane == spans[right].plane) return false;
    }
    return true;
}

} // namespace

MoeWorklistPlanResult build_moe_worklist_plan(
    const SemanticModel& model, const SemanticLayer& layer,
    ExecutionPhase phase, uint32_t token_capacity,
    const PhysicalCodecRegistry& codec_registry) {
    if (phase != ExecutionPhase::Prefill && phase != ExecutionPhase::Decode)
        return worklist_error(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "MoE worklist phase is not executable");
    if (token_capacity == 0)
        return worklist_error(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "MoE worklist token capacity is zero");
    const auto operators = layer_operators(model, layer);
    if (operators.empty() || (layer.flags & kSemanticLayerFlagSpeculative) != 0)
        return worklist_error(CompatibilityError::IR_REFERENCE_INVALID,
                              "MoE worklist layer range is invalid");
    for (const SemanticOperator* op : operators) {
        if (!semantic_operator_signature_valid(*op) ||
            !semantic_operator_contract_valid(*op))
            return worklist_error(CompatibilityError::IR_REFERENCE_INVALID,
                                  "MoE worklist contains an invalid semantic operator");
    }

    bool ambiguous = false;
    const SemanticOperator* router = unique_operator(
        operators, [](const SemanticOperator& op) { return op.kind == OperatorKind::RouterTopK; }, ambiguous);
    if (!router || ambiguous)
        return worklist_error(ambiguous ? CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS
                                        : CompatibilityError::IMPORT_SEMANTICS_MISSING,
                              "MoE worklist router is missing or ambiguous");
    const auto* router_payload = std::get_if<RouterTopKPayload>(&router->payload);
    if (!router_payload || router->inputs.size() != 1 || router->outputs.size() != 2)
        return worklist_error(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "MoE worklist router contract is invalid");
    const uint32_t experts = router_payload->expert_count;
    const uint32_t selected = router_payload->selected_count;

    const SemanticOperator* gate_up = unique_operator(
        operators, [&](const SemanticOperator& op) {
            if (op.kind != OperatorKind::RoutedLinear || op.inputs.size() != 3) return false;
            const SemanticTensor* tensor = nullptr;
            return op.inputs[1] == router->outputs[0] && op.inputs[2] == router->outputs[1] &&
                   has_tensor_role(model, op, TensorRole::FfnUpWeight, tensor) && tensor->expert_axis.kind == ExpertAxisKind::ExpertBank;
        }, ambiguous);
    if (!gate_up || ambiguous)
        return worklist_error(ambiguous ? CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS
                                        : CompatibilityError::IMPORT_SEMANTICS_MISSING,
                              "MoE worklist gate/up projection is missing or ambiguous");
    const SemanticTensor* gate_up_tensor = nullptr;
    if (!has_tensor_role(model, *gate_up, TensorRole::FfnUpWeight, gate_up_tensor))
        return worklist_error(CompatibilityError::IR_REFERENCE_INVALID,
                              "MoE worklist gate/up tensor is not bound");
    if (gate_up->outputs.size() != 1 || gate_up_tensor->expert_axis.input_axis >= gate_up_tensor->dimensions.size() ||
        gate_up_tensor->expert_axis.output_axis >= gate_up_tensor->dimensions.size())
        return worklist_error(CompatibilityError::IR_SHAPE_MISMATCH,
                              "MoE worklist gate/up shape is invalid");
    const Dimension& hidden_dimension = gate_up_tensor->dimensions[gate_up_tensor->expert_axis.input_axis];
    const Dimension& gate_up_output_dimension = gate_up_tensor->dimensions[gate_up_tensor->expert_axis.output_axis];
    if (hidden_dimension.kind != DimensionKind::Constant || hidden_dimension.constant_or_symbol == 0 ||
        hidden_dimension.constant_or_symbol > UINT32_MAX ||
        gate_up_output_dimension.kind != DimensionKind::Constant ||
        gate_up_output_dimension.constant_or_symbol == 0 ||
        gate_up_output_dimension.constant_or_symbol > UINT32_MAX ||
        gate_up_output_dimension.constant_or_symbol % 2 != 0 ||
        gate_up_tensor->expert_axis.expert_count != experts) {
        return worklist_error(CompatibilityError::IR_SHAPE_MISMATCH,
                              "MoE worklist gate/up dimensions do not match the router");
    }
    const uint32_t hidden = static_cast<uint32_t>(hidden_dimension.constant_or_symbol);
    const uint32_t gate_up_output = static_cast<uint32_t>(gate_up_output_dimension.constant_or_symbol);
    const uint32_t intermediate = gate_up_output / 2;

    const SemanticOperator* split = unique_operator(
        operators, [&](const SemanticOperator& op) {
            return op.kind == OperatorKind::AxisSplit && op.inputs == std::vector<uint32_t>{gate_up->outputs[0]};
        }, ambiguous);
    if (!split || ambiguous)
        return worklist_error(ambiguous ? CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS
                                        : CompatibilityError::IMPORT_SEMANTICS_MISSING,
                              "MoE worklist gate/up split is missing or ambiguous");
    const auto* split_payload = std::get_if<AxisSplitPayload>(&split->payload);
    if (!split_payload || split_payload->first_width != intermediate ||
        split_payload->second_width != intermediate)
        return worklist_error(CompatibilityError::IR_SHAPE_MISMATCH,
                              "MoE worklist split width differs from the expert tensor");

    const SemanticOperator* activation = unique_operator(
        operators, [&](const SemanticOperator& op) {
            return op.kind == OperatorKind::GatedActivation && op.inputs == split->outputs;
        }, ambiguous);
    if (!activation || ambiguous)
        return worklist_error(ambiguous ? CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS
                                        : CompatibilityError::IMPORT_SEMANTICS_MISSING,
                              "MoE worklist activation is missing or ambiguous");
    const auto* activation_payload = std::get_if<GatedActivationPayload>(&activation->payload);
    if (!activation_payload)
        return worklist_error(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "MoE worklist activation payload is invalid");

    const SemanticOperator* down = unique_operator(
        operators, [&](const SemanticOperator& op) {
            if (op.kind != OperatorKind::RoutedLinear || op.inputs.size() != 3) return false;
            const SemanticTensor* tensor = nullptr;
            return op.inputs == std::vector<uint32_t>{activation->outputs[0], router->outputs[0], router->outputs[1]} &&
                   has_tensor_role(model, op, TensorRole::FfnDownWeight, tensor) && tensor->expert_axis.kind == ExpertAxisKind::ExpertBank;
        }, ambiguous);
    if (!down || ambiguous)
        return worklist_error(ambiguous ? CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS
                                        : CompatibilityError::IMPORT_SEMANTICS_MISSING,
                              "MoE worklist down projection is missing or ambiguous");
    const SemanticTensor* down_tensor = nullptr;
    if (!has_tensor_role(model, *down, TensorRole::FfnDownWeight, down_tensor) || down->outputs.size() != 1)
        return worklist_error(CompatibilityError::IR_REFERENCE_INVALID,
                              "MoE worklist down tensor is not bound");
    if (down_tensor->dimensions.size() != 3 || down_tensor->expert_axis.input_axis >= 3 ||
        down_tensor->expert_axis.output_axis >= 3) {
        return worklist_error(CompatibilityError::IR_SHAPE_MISMATCH,
                              "MoE worklist down tensor rank is invalid");
    }
    const Dimension& down_input_dimension = down_tensor->dimensions[down_tensor->expert_axis.input_axis];
    const Dimension& down_output_dimension = down_tensor->dimensions[down_tensor->expert_axis.output_axis];
    if (down_input_dimension.kind != DimensionKind::Constant || down_output_dimension.kind != DimensionKind::Constant ||
        down_input_dimension.constant_or_symbol > UINT32_MAX || down_output_dimension.constant_or_symbol > UINT32_MAX) {
        return worklist_error(CompatibilityError::IR_SHAPE_MISMATCH,
                              "MoE worklist down dimensions exceed host capacity");
    }
    const uint32_t down_input = static_cast<uint32_t>(down_input_dimension.constant_or_symbol);
    const uint32_t down_output = static_cast<uint32_t>(down_output_dimension.constant_or_symbol);
    if (down_input != intermediate || down_output != hidden)
        return worklist_error(CompatibilityError::IR_SHAPE_MISMATCH,
                              "MoE worklist down dimensions do not match gate/up");

    const SemanticOperator* reduce = unique_operator(
        operators, [&](const SemanticOperator& op) {
            return op.kind == OperatorKind::WeightedExpertReduce &&
                   op.inputs == std::vector<uint32_t>{down->outputs[0], router->outputs[0], router->outputs[1]};
        }, ambiguous);
    if (!reduce || ambiguous)
        return worklist_error(ambiguous ? CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS
                                        : CompatibilityError::IMPORT_SEMANTICS_MISSING,
                              "MoE worklist reduction is missing or ambiguous");
    const auto* reduce_payload = std::get_if<WeightedExpertReducePayload>(&reduce->payload);
    if (!reduce_payload || reduce_payload->scale_source != ExpertScaleSource::PerExpertTensor ||
        reduce->tensors.size() != 1)
        return worklist_error(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "MoE worklist reduction scale is incomplete");

    const SemanticOperator* router_scale = unique_operator(
        operators, [&](const SemanticOperator& op) {
            const SemanticTensor* tensor = nullptr;
            return op.kind == OperatorKind::Scale && op.outputs == router->inputs &&
                   has_tensor_role(model, op, TensorRole::RouterScaleWeight, tensor);
        }, ambiguous);
    if (!router_scale || ambiguous)
        return worklist_error(ambiguous ? CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS
                                        : CompatibilityError::IMPORT_SEMANTICS_MISSING,
                              "MoE worklist router scale is missing or ambiguous");
    const SemanticOperator* expert_norm = unique_operator(
        operators, [&](const SemanticOperator& op) {
            const SemanticTensor* tensor = nullptr;
            return op.kind == OperatorKind::RmsNorm && op.outputs == std::vector<uint32_t>{gate_up->inputs[0]} &&
                   has_tensor_role(model, op, TensorRole::ExpertNormWeight, tensor);
        }, ambiguous);
    if (!expert_norm || ambiguous)
        return worklist_error(ambiguous ? CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS
                                        : CompatibilityError::IMPORT_SEMANTICS_MISSING,
                              "MoE worklist expert norm is missing or ambiguous");

    const std::array<const SemanticOperator*, 8> required = {
        router, gate_up, split, activation, down, reduce, router_scale, expert_norm};
    if (operators.size() != required.size())
        return worklist_error(CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS,
                              "MoE worklist layer contains an unsupported or disconnected operator");
    for (size_t left = 0; left != required.size(); ++left) {
        if (!required[left]) return worklist_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                                   "MoE worklist required operator is absent");
        for (size_t right = left + 1; right != required.size(); ++right)
            if (required[left] == required[right])
                return worklist_error(CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS,
                                      "MoE worklist operator is used for two semantic roles");
    }

    // Check value-edge order independently of serialized operator order. Any
    // cycle or multiply-produced internal value is refused before planning.
    std::vector<uint32_t> indegree(required.size(), 0);
    std::vector<std::vector<size_t>> successors(required.size());
    for (size_t consumer = 0; consumer != required.size(); ++consumer) {
        for (uint32_t input : required[consumer]->inputs) {
            size_t producer = required.size();
            for (size_t candidate = 0; candidate != required.size(); ++candidate) {
                if (std::find(required[candidate]->outputs.begin(), required[candidate]->outputs.end(), input) ==
                    required[candidate]->outputs.end()) continue;
                if (producer != required.size())
                    return worklist_error(CompatibilityError::IR_CONSTRAINT_FAILED,
                                          "MoE worklist value has multiple internal producers");
                producer = candidate;
            }
            if (producer == consumer)
                return worklist_error(CompatibilityError::IR_CONSTRAINT_FAILED,
                                      "MoE worklist contains a self edge");
            if (producer != required.size()) {
                if (std::find(successors[producer].begin(), successors[producer].end(), consumer) ==
                    successors[producer].end()) {
                    successors[producer].push_back(consumer);
                    ++indegree[consumer];
                }
            }
        }
    }
    std::vector<size_t> ready;
    for (size_t index = 0; index != required.size(); ++index)
        if (indegree[index] == 0) ready.push_back(index);
    size_t processed = 0;
    for (size_t cursor = 0; cursor != ready.size(); ++cursor) {
        ++processed;
        for (size_t successor : successors[ready[cursor]]) if (--indegree[successor] == 0) ready.push_back(successor);
    }
    if (processed != required.size())
        return worklist_error(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "MoE worklist semantic edge order contains a cycle");

    const auto* router_scale_payload = std::get_if<ScalePayload>(&router_scale->payload);
    const auto* expert_norm_payload = std::get_if<RmsNormPayload>(&expert_norm->payload);
    const SemanticTensor* router_scale_tensor = nullptr;
    const SemanticTensor* expert_norm_tensor = nullptr;
    const SemanticTensor* reduce_scale_tensor = nullptr;
    if (!router_scale_payload || router_scale_payload->source != ScaleSource::Tensor ||
        !expert_norm_payload || !has_tensor_role(model, *router_scale, TensorRole::RouterScaleWeight, router_scale_tensor) ||
        !has_tensor_role(model, *expert_norm, TensorRole::ExpertNormWeight, expert_norm_tensor) ||
        !has_tensor_role(model, *reduce, TensorRole::ReduceScaleWeight, reduce_scale_tensor) ||
        !valid_aux_tensor(*router_scale_tensor, TensorRole::RouterScaleWeight, experts) ||
        !valid_aux_tensor(*expert_norm_tensor, TensorRole::ExpertNormWeight, hidden) ||
        !valid_aux_tensor(*reduce_scale_tensor, TensorRole::ReduceScaleWeight, experts) ||
        !routed_shapes(model, *router, *gate_up, *split, *activation, *down, *reduce,
                       hidden, intermediate, experts, selected)) {
        // The expert identities are checked after registry admission below.
        if (!routed_shapes(model, *router, *gate_up, *split, *activation, *down, *reduce,
                           hidden, intermediate, experts, selected))
            return worklist_error(CompatibilityError::IR_SHAPE_MISMATCH,
                                  "MoE worklist value shapes do not form one routed subgraph");
        return worklist_error(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "MoE worklist auxiliary tensor contract is invalid");
    }

    if (!validate_physical_codec_registry(codec_registry, true) ||
        !physical_codec_registry_matches_model(codec_registry, model))
        return worklist_error(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                              "MoE worklist physical codec registry is incomplete");
    const PhysicalCodecIdentity* gate_up_identity = registry_identity(codec_registry, gate_up_tensor->id);
    const PhysicalCodecIdentity* down_identity = registry_identity(codec_registry, down_tensor->id);
    if (!gate_up_identity || !down_identity ||
        !valid_expert_tensor(*gate_up_tensor, *gate_up_identity, TensorRole::FfnUpWeight,
                             experts, hidden, gate_up_output) ||
        !valid_expert_tensor(*down_tensor, *down_identity, TensorRole::FfnDownWeight,
                             experts, down_input, down_output))
        return worklist_error(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                              "MoE worklist expert codec identity is incomplete");

    MoeWorklistDescriptor descriptor;
    descriptor.phase = phase;
    descriptor.semantic_operator_ids = {router->id, gate_up->id, activation->id, down->id, reduce->id};
    descriptor.expert_count = experts;
    descriptor.selected_count = selected;
    descriptor.hidden = hidden;
    descriptor.intermediate = intermediate;
    descriptor.activation = activation_payload->activation;
    descriptor.scale_source = reduce_payload->scale_source;
    descriptor.value_source = ValueSource::SeparateProjection;
    descriptor.gate_up = tensor_contract(*gate_up_tensor, *gate_up_identity);
    descriptor.down = tensor_contract(*down_tensor, *down_identity);
    descriptor.physical_codecs = {*gate_up_identity, *gate_up_identity, *down_identity};
    descriptor.router_scale_tensor_id = router_scale_tensor->id;
    descriptor.expert_norm_tensor_id = expert_norm_tensor->id;
    descriptor.reduce_scale_tensor_id = reduce_scale_tensor->id;

    MoeWorklistDeviceLayout device_layout;
    if (!valid_descriptor(descriptor) ||
        !build_device_layout(token_capacity, experts, selected, device_layout))
        return worklist_error(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "MoE worklist descriptor or payload layout is invalid");

    std::array<MoeWorklistSourceSpanRef, kMoeWorklistMaximumSourceSpans> spans{};
    uint32_t span_count = 0;
    for (const SemanticTensor* tensor : std::array<const SemanticTensor*, 5>{
             gate_up_tensor, down_tensor, router_scale_tensor, expert_norm_tensor, reduce_scale_tensor}) {
        for (const TensorPlane& plane : tensor->planes) {
            if (span_count == spans.size() || plane.length == 0 ||
                plane.offset > UINT64_MAX - plane.length)
                return worklist_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                      "MoE worklist source plane span is out of bounds");
            spans[span_count++] = {tensor->id, plane.kind, 0, plane.offset, plane.length};
        }
    }
    const Sha256Digest digest = plan_digest(model, layer, descriptor,
                                            {spans.data(), span_count},
                                            codec_registry, token_capacity);
    if (digest_is_zero(digest))
        return worklist_error(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH,
                              "MoE worklist source contract has no authenticated digest");
    MoeWorklistPlan plan(std::move(descriptor), digest, device_layout, std::move(spans), span_count);
    if (!valid_moe_worklist_plan(plan))
        return worklist_error(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "MoE worklist plan failed its immutable self-check");
    return plan;
}

bool valid_moe_worklist_plan(const MoeWorklistPlan& plan) {
    return valid_descriptor(plan.descriptor()) && !digest_is_zero(plan.contract_digest()) &&
           descriptor_layout_matches(plan) && source_spans_valid(plan);
}

} // namespace Laplace
