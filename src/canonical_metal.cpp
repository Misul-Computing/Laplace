#include "canonical_metal.h"

#include <CommonCrypto/CommonDigest.h>

#include <fcntl.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "execution_plan.h"
#include "column_grouped_u2_atlas.h"
#include "fp16.h"
#include "kernels.h"
#include "matmul.h"
#include "sparse_selector_coeff.h"

namespace Laplace {

namespace {

constexpr int kMetalUnboundedAttentionWindow = 0;
std::atomic<uint64_t> g_next_metal_session_id{1};

CompatibilityReport metal_error(CompatibilityError code, std::string detail = {}) {
    CompatibilityReport report = package_report(code, std::move(detail));
    report.stage = CompatibilityStage::Session;
    return report;
}

bool f32_bits(uint32_t bits, float& value) {
    std::memcpy(&value, &bits, sizeof(value));
    return std::isfinite(value);
}

bool constant_dimension(const Dimension& dimension, uint64_t& value) {
    if (dimension.kind != DimensionKind::Constant || dimension.constant_or_symbol == 0) return false;
    value = dimension.constant_or_symbol;
    return true;
}

CompatibilityReport calibration_error(CompatibilityError code, const CalibrationTarget& target,
                                      std::string detail) {
    CompatibilityReport report = package_report(code, std::move(detail));
    report.stage = CompatibilityStage::Plan;
    report.operator_id = target.operator_id;
    report.tensor_id = target.weight_tensor_id;
    return report;
}

CompatibilityReport calibration_cache_error(CompatibilityError code, std::string detail) {
    CompatibilityReport report = package_report(code, std::move(detail));
    report.stage = CompatibilityStage::Package;
    return report;
}

void append_u16(std::vector<uint8_t>& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8));
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void append_target(std::vector<uint8_t>& bytes, const CalibrationTarget& target) {
    append_u16(bytes, target.version);
    append_u16(bytes, target.flags);
    append_u32(bytes, target.operator_id);
    append_u32(bytes, target.input_value_id);
    append_u32(bytes, target.weight_tensor_id);
    bytes.push_back(target.k_mapping.input_axis);
    bytes.push_back(target.k_mapping.weight_physical_axis);
    bytes.push_back(target.k_mapping.weight_logical_axis);
    bytes.push_back(target.k_mapping.reserved);
    append_u32(bytes, target.k_mapping.width);
    append_u32(bytes, target.k_mapping.block_elements);
    append_u32(bytes, target.k_mapping.block_bytes);
}

Sha256Digest sha256_bytes(std::span<const uint8_t> bytes) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    constexpr size_t chunk_size = 1024 * 1024;
    for (size_t offset = 0; offset < bytes.size(); offset += chunk_size) {
        const size_t length = std::min(chunk_size, bytes.size() - offset);
        CC_SHA256_Update(&context, bytes.data() + offset, static_cast<CC_LONG>(length));
    }
    Sha256Digest digest;
    CC_SHA256_Final(digest.bytes.data(), &context);
    return digest;
}

bool read_u16(std::span<const uint8_t> bytes, size_t& offset, uint16_t& value) {
    if (offset > bytes.size() || bytes.size() - offset < 2) return false;
    value = static_cast<uint16_t>(bytes[offset]) |
            static_cast<uint16_t>(bytes[offset + 1]) << 8;
    offset += 2;
    return true;
}

bool read_u32(std::span<const uint8_t> bytes, size_t& offset, uint32_t& value) {
    if (offset > bytes.size() || bytes.size() - offset < 4) return false;
    value = static_cast<uint32_t>(bytes[offset]) |
            static_cast<uint32_t>(bytes[offset + 1]) << 8 |
            static_cast<uint32_t>(bytes[offset + 2]) << 16 |
            static_cast<uint32_t>(bytes[offset + 3]) << 24;
    offset += 4;
    return true;
}

bool read_digest(std::span<const uint8_t> bytes, size_t& offset, Sha256Digest& digest) {
    if (offset > bytes.size() || bytes.size() - offset < digest.bytes.size()) return false;
    std::memcpy(digest.bytes.data(), bytes.data() + offset, digest.bytes.size());
    offset += digest.bytes.size();
    return true;
}

bool read_target(std::span<const uint8_t> bytes, size_t& offset, CalibrationTarget& target) {
    if (!read_u16(bytes, offset, target.version) || !read_u16(bytes, offset, target.flags) ||
        !read_u32(bytes, offset, target.operator_id) ||
        !read_u32(bytes, offset, target.input_value_id) ||
        !read_u32(bytes, offset, target.weight_tensor_id) ||
        offset > bytes.size() || bytes.size() - offset < 4) return false;
    target.k_mapping.input_axis = bytes[offset++];
    target.k_mapping.weight_physical_axis = bytes[offset++];
    target.k_mapping.weight_logical_axis = bytes[offset++];
    target.k_mapping.reserved = bytes[offset++];
    return read_u32(bytes, offset, target.k_mapping.width) &&
           read_u32(bytes, offset, target.k_mapping.block_elements) &&
           read_u32(bytes, offset, target.k_mapping.block_bytes);
}

bool gguf_block_type(const SemanticTensor& semantic, const TensorPlane& plane, GGMLType& type) {
    if (semantic.layout.kind != PhysicalLayoutKind::GgufBlocked ||
        semantic.layout.packing != PackingKind::Gguf || semantic.layout.block_rank != 1 ||
        (semantic.quantization.kind != QuantizationKind::BlockedAffine &&
         semantic.quantization.kind != QuantizationKind::Codebook) ||
        semantic.planes.size() != 1 || plane.kind != PlaneKind::Values ||
        plane.storage_type != ScalarType::U8 || semantic.layout.block_elements == 0 ||
        semantic.layout.block_bytes == 0 ||
        semantic.layout.block_elements != semantic.quantization.block_elements ||
        semantic.layout.block_bytes != semantic.quantization.block_bytes ||
        (semantic.quantization.kind == QuantizationKind::BlockedAffine &&
         semantic.quantization.group_size != semantic.layout.block_elements) ||
        (semantic.quantization.kind == QuantizationKind::Codebook &&
         semantic.quantization.group_size != 32) ||
        semantic.quantization.required_plane_mask != 1) return false;
    struct BlockFormat { uint32_t elements; uint32_t bytes; GGMLType type; };
    static constexpr BlockFormat formats[] = {
        {256, 66, GGMLType::IQ2_XXS},
        {32, 18, GGMLType::Q4_0},
        {256, 84, GGMLType::Q2_K},
        {256, 144, GGMLType::Q4_K},
        {32, 22, GGMLType::Q5_0},
        {256, 210, GGMLType::Q6_K},
        {32, 34, GGMLType::Q8_0},
    };
    for (const BlockFormat& format : formats) {
        if (semantic.layout.block_elements == format.elements && semantic.layout.block_bytes == format.bytes) {
            type = format.type;
            return true;
        }
    }
    return false;
}

bool affine_u2_256_tensor_view(const RuntimePackage& package, const SemanticTensor& semantic,
                               Tensor& tensor) {
    if (semantic.logical_type != ScalarType::F32 || semantic.dimensions.size() != 2 ||
        semantic.layout.kind != PhysicalLayoutKind::GroupedAffine ||
        semantic.layout.version != 1 ||
        semantic.layout.packing != PackingKind::LsbBitPacked ||
        semantic.layout.rank != 2 || semantic.layout.block_rank != 1 ||
        semantic.layout.axis_order[0] != 1 || semantic.layout.axis_order[1] != 0 ||
        semantic.layout.strides[0] != 1 ||
        semantic.layout.strides[1] != semantic.dimensions[1].constant_or_symbol ||
        semantic.layout.flags != 0 || semantic.flags != 0 || semantic.expert_axis != ExpertAxis{} ||
        semantic.layout.block_elements != 256 || semantic.layout.block_bytes != 64 ||
        semantic.quantization.kind != QuantizationKind::BlockedAffine ||
        semantic.quantization.version != 1 ||
        semantic.quantization.accumulation_type != ScalarType::F32 ||
        semantic.quantization.scale_type != ScalarType::F16 ||
        semantic.quantization.zero_type != static_cast<ScalarType>(0) ||
        semantic.quantization.bias_type != ScalarType::F16 ||
        semantic.quantization.block_elements != 256 ||
        semantic.quantization.block_bytes != 64 || semantic.quantization.group_size != 256 ||
        semantic.quantization.required_plane_mask != 7 || semantic.quantization.flags != 0 ||
        semantic.planes.size() != 3) return false;
    uint64_t n = 0;
    uint64_t k = 0;
    if (!constant_dimension(semantic.dimensions[0], n) ||
        !constant_dimension(semantic.dimensions[1], k) || n % 8 != 0 || k % 512 != 0 ||
        k % 256 != 0 || n > UINT64_MAX / k) return false;
    const uint64_t blocks = n * (k / 256);
    if (!blocks || blocks > UINT64_MAX / 2) return false;
    const uint64_t values_length = n * (k / 4);
    const uint64_t plane_length = blocks * 2;
    const TensorPlane* values = nullptr;
    const TensorPlane* scales = nullptr;
    const TensorPlane* biases = nullptr;
    for (const TensorPlane& plane : semantic.planes) {
        if (plane.alignment != 128 || plane.offset % 128 != 0 || plane.flags != 0) return false;
        if (plane.kind == PlaneKind::Values && plane.storage_type == ScalarType::U32) {
            if (values) return false;
            values = &plane;
        } else if (plane.kind == PlaneKind::Scales && plane.storage_type == ScalarType::F16) {
            if (scales) return false;
            scales = &plane;
        } else if (plane.kind == PlaneKind::Biases && plane.storage_type == ScalarType::F16) {
            if (biases) return false;
            biases = &plane;
        }
        else
            return false;
    }
    if (!values || !scales || !biases || values->length != values_length ||
        scales->length != plane_length || biases->length != plane_length) return false;
    const auto pointer_for = [&](const TensorPlane& plane) -> const uint8_t* {
        const std::span<const uint8_t> artifact = package.artifact_bytes(plane.artifact_id);
        if (artifact.empty() || plane.offset > artifact.size() ||
            plane.length > artifact.size() - plane.offset) return nullptr;
        return artifact.data() + plane.offset;
    };
    const uint8_t* values_pointer = pointer_for(*values);
    const uint8_t* scales_pointer = pointer_for(*scales);
    const uint8_t* biases_pointer = pointer_for(*biases);
    if (!values_pointer || !scales_pointer || !biases_pointer) return false;
    tensor = {};
    tensor.type = GGMLType::GROUPED_AFFINE_U2_256;
    tensor.n_dims = 2;
    tensor.dims[0] = k;
    tensor.dims[1] = n;
    tensor.data = values_pointer;
    tensor.scales = scales_pointer;
    tensor.biases = biases_pointer;
    tensor.data_bytes = values_length;
    tensor.scale_bytes = plane_length;
    tensor.bias_bytes = plane_length;
    tensor.mlx_bits = 2;
    tensor.mlx_group_size = 256;
    return true;
}

bool column_grouped_affine_u2_skip_256_tensor_view(
    const RuntimePackage& package, const SemanticTensor& semantic, Tensor& tensor) {
    if (semantic.logical_type != ScalarType::F32 || semantic.dimensions.size() != 2 ||
        semantic.layout.kind != PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip ||
        semantic.layout.version != 1 ||
        semantic.layout.packing != PackingKind::LsbBitPacked ||
        semantic.layout.rank != 2 || semantic.layout.block_rank != 1 ||
        semantic.layout.axis_order[0] != 1 || semantic.layout.axis_order[1] != 0 ||
        semantic.layout.strides[0] != 1 ||
        semantic.layout.strides[1] != semantic.dimensions[1].constant_or_symbol ||
        semantic.layout.flags != 0 || semantic.flags != 0 ||
        semantic.expert_axis != ExpertAxis{} ||
        semantic.layout.block_elements != 256 || semantic.layout.block_bytes != 64 ||
        semantic.quantization.kind != QuantizationKind::BlockedAffine ||
        semantic.quantization.version != 1 ||
        semantic.quantization.accumulation_type != ScalarType::F32 ||
        semantic.quantization.scale_type != ScalarType::F16 ||
        semantic.quantization.zero_type != static_cast<ScalarType>(0) ||
        semantic.quantization.bias_type != ScalarType::F16 ||
        semantic.quantization.block_elements != 256 ||
        semantic.quantization.block_bytes != 64 ||
        semantic.quantization.group_size != 256 ||
        semantic.quantization.required_plane_mask != 7 ||
        semantic.quantization.flags != 0 || semantic.planes.size() != 3)
        return false;

    uint64_t n = 0;
    uint64_t k = 0;
    if (!constant_dimension(semantic.dimensions[0], n) ||
        !constant_dimension(semantic.dimensions[1], k) || n % 256 != 0 ||
        n > UINT64_MAX / k)
        return false;
    const uint64_t elements = n * k;
    const uint64_t groups = elements / 256;
    if (groups == 0 || groups > UINT64_MAX / 2) return false;
    const uint64_t values_length = elements / 4;
    const uint64_t metadata_length = groups * 2;

    const TensorPlane* values = nullptr;
    const TensorPlane* scales = nullptr;
    const TensorPlane* biases = nullptr;
    for (const TensorPlane& plane : semantic.planes) {
        if (plane.alignment != 128 || plane.offset % 128 != 0 || plane.flags != 0)
            return false;
        if (plane.kind == PlaneKind::Values && plane.storage_type == ScalarType::U8) {
            if (values) return false;
            values = &plane;
        } else if (plane.kind == PlaneKind::Scales &&
                   plane.storage_type == ScalarType::F16) {
            if (scales) return false;
            scales = &plane;
        } else if (plane.kind == PlaneKind::Biases &&
                   plane.storage_type == ScalarType::F16) {
            if (biases) return false;
            biases = &plane;
        } else {
            return false;
        }
    }
    if (!values || !scales || !biases || values->length != values_length ||
        scales->length != metadata_length || biases->length != metadata_length)
        return false;

    const auto pointer_for = [&](const TensorPlane& plane) -> const uint8_t* {
        const std::span<const uint8_t> artifact = package.artifact_bytes(plane.artifact_id);
        if (artifact.empty() || plane.offset > artifact.size() ||
            plane.length > artifact.size() - plane.offset)
            return nullptr;
        return artifact.data() + plane.offset;
    };
    const uint8_t* values_pointer = pointer_for(*values);
    const uint8_t* scales_pointer = pointer_for(*scales);
    const uint8_t* biases_pointer = pointer_for(*biases);
    if (!values_pointer || !scales_pointer || !biases_pointer) return false;

    tensor = {};
    tensor.type = GGMLType::COLUMN_GROUPED_AFFINE_U2_SKIP_256;
    tensor.n_dims = 2;
    tensor.dims[0] = k;
    tensor.dims[1] = n;
    tensor.data = values_pointer;
    tensor.scales = scales_pointer;
    tensor.biases = biases_pointer;
    tensor.data_bytes = values_length;
    tensor.scale_bytes = metadata_length;
    tensor.bias_bytes = metadata_length;
    tensor.mlx_bits = 2;
    tensor.mlx_group_size = 256;
    return true;
}

bool tensor_view(const RuntimePackage& package, const SemanticTensor& semantic, Tensor& tensor) {
    if (semantic.planes.size() == 3 &&
        column_grouped_affine_u2_skip_256_tensor_view(package, semantic, tensor))
        return true;
    if (semantic.planes.size() == 3 && affine_u2_256_tensor_view(package, semantic, tensor)) return true;
    if (semantic.planes.size() != 1 ||
        semantic.planes[0].kind != PlaneKind::Values || semantic.dimensions.empty() || semantic.dimensions.size() > 4) return false;
    const TensorPlane& plane = semantic.planes[0];
    const std::span<const uint8_t> artifact = package.artifact_bytes(plane.artifact_id);
    if (artifact.empty() || plane.offset > artifact.size() || plane.length > artifact.size() - plane.offset) return false;
    uint64_t elements = 1;
    tensor = {};
    tensor.n_dims = static_cast<uint32_t>(semantic.dimensions.size());
    for (uint32_t physical_axis = 0; physical_axis != tensor.n_dims; ++physical_axis) {
        const uint8_t logical_axis = semantic.layout.axis_order[physical_axis];
        if (logical_axis >= tensor.n_dims ||
            !constant_dimension(semantic.dimensions[logical_axis], tensor.dims[physical_axis]) ||
            tensor.dims[physical_axis] > std::numeric_limits<uint64_t>::max() / elements) return false;
        elements *= tensor.dims[physical_axis];
    }
    if (semantic.layout.kind == PhysicalLayoutKind::ContiguousRowMajor &&
        semantic.quantization.kind == QuantizationKind::None &&
        (plane.storage_type == ScalarType::F16 || plane.storage_type == ScalarType::F32)) {
        tensor.type = plane.storage_type == ScalarType::F16 ? GGMLType::F16 : GGMLType::F32;
        const uint64_t bytes_per_element = plane.storage_type == ScalarType::F16 ? 2 : 4;
        if (elements > std::numeric_limits<uint64_t>::max() / bytes_per_element ||
            elements * bytes_per_element != plane.length) return false;
    } else {
        if (!gguf_block_type(semantic, plane, tensor.type) || elements % semantic.layout.block_elements != 0 ||
            elements / semantic.layout.block_elements > std::numeric_limits<uint64_t>::max() / semantic.layout.block_bytes ||
            elements / semantic.layout.block_elements * semantic.layout.block_bytes != plane.length) return false;
    }
    tensor.data = artifact.data() + plane.offset;
    // Keep the exact source span with the non-owning view.  Canonical Metal
    // registration is allowed to cover a larger artifact range, but a
    // binder must never infer permission to read past this tensor plane.
    tensor.data_bytes = plane.length;
    return true;
}

const SemanticTensor* tensor_with_role(const SemanticModel& model, const SemanticOperator& op, TensorRole role,
                                       bool expert = false) {
    const SemanticTensor* found = nullptr;
    for (uint32_t id : op.tensors) {
        if (id >= model.tensors.size() || model.tensors[id].role != role ||
            (model.tensors[id].expert_axis.kind == ExpertAxisKind::ExpertBank) != expert) continue;
        if (found) return nullptr;
        found = &model.tensors[id];
    }
    return found;
}

const SemanticOperator* operator_with_role(const SemanticModel& model, const SemanticLayer& layer,
                                           OperatorKind kind, TensorRole role, bool expert = false) {
    if (layer.first_operator > model.operators.size() || layer.operator_count > model.operators.size() - layer.first_operator) return nullptr;
    const SemanticOperator* found = nullptr;
    for (uint32_t index = 0; index != layer.operator_count; ++index) {
        const SemanticOperator& op = model.operators[layer.first_operator + index];
        if (op.kind != kind || !tensor_with_role(model, op, role, expert)) continue;
        if (found) return nullptr;
        found = &op;
    }
    return found;
}

bool output_width(const SemanticModel& model, const SemanticOperator& op, uint32_t& width) {
    if (op.outputs.size() != 1 || op.outputs[0] >= model.values.size()) return false;
    const SemanticValue& output = model.values[op.outputs[0]];
    if (output.dimensions.empty()) return false;
    uint64_t value = 0;
    if (!constant_dimension(output.dimensions.back(), value) || value > UINT32_MAX) return false;
    width = static_cast<uint32_t>(value);
    return true;
}

bool value_width(const SemanticModel& model, uint32_t value_id, uint32_t& width) {
    if (value_id >= model.values.size()) return false;
    const SemanticValue& value = model.values[value_id];
    if (value.dimensions.empty() || value.dimensions.size() > 4) return false;
    uint64_t resolved = 0;
    if (!constant_dimension(value.dimensions.back(), resolved) || resolved == 0 || resolved > UINT32_MAX)
        return false;
    width = static_cast<uint32_t>(resolved);
    return true;
}

bool validate_linear_tensor_shape(const SemanticModel& model, const SemanticOperator& op,
                                  const Tensor& tensor, TensorRole role, bool expert,
                                  std::string& error) {
    if (expert || op.kind != OperatorKind::Linear) return true;
    const auto* payload = std::get_if<LinearPayload>(&op.payload);
    uint32_t input_width = 0;
    uint32_t output_width_value = 0;
    if (!payload || payload->transpose_weight || op.inputs.size() != 1 || op.outputs.size() != 1 ||
        !value_width(model, op.inputs.front(), input_width) ||
        !value_width(model, op.outputs.front(), output_width_value) || tensor.n_dims != 2 ||
        tensor.dims[0] != input_width || tensor.dims[1] != output_width_value) {
        error = "canonical Metal linear tensor shape does not match semantic input/output widths for role " +
                std::to_string(static_cast<unsigned>(role));
        return false;
    }
    return true;
}

bool fp32_global_state_format(const StateFormat& format, TransformDomain encoded_domain) {
    return format.kind == StateFormatKind::GlobalContiguous && format.version == 1 &&
           format.logical_type == ScalarType::F32 && format.encoded_type == ScalarType::F32 &&
           format.logical_domain == TransformDomain::Untransformed && format.encoded_domain == encoded_domain &&
           format.codec == CodecKind::Fp32 && format.cache_policy == CachePolicy::Global &&
           format.layout_policy == LayoutPolicy::TokenMajorContiguous && format.flags == 0 &&
           format.tile_tokens == 0 && format.mutable_tokens == 0 && format.alignment == 64 && format.reserved == 0;
}

bool kv_state_shape(const SemanticState& state, uint32_t kv_heads, uint32_t head_dimension) {
    uint64_t heads = 0;
    uint64_t dimension = 0;
    return state.dimensions.size() == 3 && state.dimensions[0].kind == DimensionKind::Symbol &&
           constant_dimension(state.dimensions[1], heads) && constant_dimension(state.dimensions[2], dimension) &&
           heads == kv_heads && dimension == head_dimension;
}

const SemanticState* state_by_id(const SemanticModel& model, uint32_t id) {
    const SemanticState* found = nullptr;
    for (const SemanticState& state : model.states) {
        if (state.id != id) continue;
        if (found) return nullptr;
        found = &state;
    }
    return found;
}

bool owns_fp32_global_kv(const SemanticModel& model, const SemanticOperator& attention,
                         uint32_t kv_heads, uint32_t head_dimension,
                         uint32_t* key_state_id = nullptr,
                         uint32_t* value_state_id = nullptr) {
    if (attention.states.size() != 2) return false;
    const SemanticState* key = nullptr;
    const SemanticState* value = nullptr;
    for (uint32_t state_id : attention.states) {
        const SemanticState* state = state_by_id(model, state_id);
        if (!state) return false;
        if (state->kind == StateKind::KeyCache) key = key ? nullptr : state;
        if (state->kind == StateKind::ValueCache) value = value ? nullptr : state;
    }
    const bool valid = key && value && key->id != value->id &&
           key->semantic_version == value->semantic_version &&
           (key->semantic_version == 1 || key->semantic_version == 4 ||
            key->semantic_version == 5 || key->semantic_version == 6 || key->semantic_version == 7) &&
           key->update_kind == StateUpdateKind::AppendKey && value->update_kind == StateUpdateKind::AppendValue &&
           key->position_policy == PositionPolicy::AppendOnly && value->position_policy == PositionPolicy::AppendOnly &&
           key->flags == 0 && value->flags == 0 && key->formats.size() == 1 && value->formats.size() == 1 &&
           fp32_global_state_format(key->formats[0], TransformDomain::RopeApplied) &&
           fp32_global_state_format(value->formats[0], TransformDomain::Untransformed) &&
           kv_state_shape(*key, kv_heads, head_dimension) && kv_state_shape(*value, kv_heads, head_dimension);
    if (valid) {
        if (key_state_id) *key_state_id = key->id;
        if (value_state_id) *value_state_id = value->id;
    }
    return valid;
}

struct DenseLayer {
    Tensor attn_norm;
    Tensor query;
    Tensor query_bias;
    Tensor key;
    Tensor key_bias;
    Tensor query_norm;
    Tensor key_norm;
    Tensor value;
    Tensor value_bias;
    Tensor attention_output;
    Tensor ffn_norm;
    Tensor ffn_gate;
    Tensor ffn_up;
    Tensor ffn_down;
    Tensor moe_gate;
    Tensor moe_gate_scale;
    Tensor moe_pre_norm;
    Tensor moe_up;
    Tensor moe_down;
    Tensor moe_down_scale;
    Tensor post_ffw_1;
    Tensor post_ffw_2;
    Tensor post_ffw;
    Tensor out_scale;
    MetalTokLayer metal;
    uint32_t operator_id = 0;
};

struct RecurrentLayer {
    Tensor input_norm;
    Tensor qkv;
    Tensor gate;
    Tensor beta;
    Tensor alpha;
    Tensor conv;
    Tensor dt_bias;
    Tensor decay;
    Tensor norm;
    Tensor output;
    Tensor ffn_norm;
    Tensor ffn_gate;
    Tensor ffn_up;
    Tensor ffn_down;
    MetalTokRecurrentLayer metal;
    uint32_t operator_id = 0;
};

struct DerivedTensorOwner {
    uint32_t tensor_id = UINT32_MAX;
    ArtifactId artifact_id{};
    DerivedQ2KContract contract;
    std::array<uint8_t, 32> source_digest{};
    std::array<uint8_t, 32> storage_digest{};
    std::shared_ptr<uint8_t> mapping;
    size_t logical_length = 0;
    size_t mapped_length = 0;
    Tensor tensor;
};

struct DerivedIQ2XXSEntry {
    uint32_t tensor_id = UINT32_MAX;
    uint64_t offset = 0;
    uint64_t logical_length = 0;
    CalibrationKMapping calibration_mapping;
    std::vector<float> importance;
    DerivedIQ2XXSRecord record;
    Tensor tensor;
};

struct DerivedIQ2XXSAtlas {
    ArtifactId artifact_id{0x90000000u};
    std::shared_ptr<uint8_t> mapping;
    size_t logical_length = 0;
    size_t mapped_length = 0;
    uint64_t source_bytes = 0;
    std::vector<DerivedIQ2XXSEntry> entries;
};

struct SparseFfnPlan {
    uint32_t packed_intermediate = 0;
    uint64_t source_bytes = 0;
    uint64_t requested_bytes = 0;
};

struct SparseFfnProxyOwner {
    uint32_t layer_index = 0;
    uint32_t slot = UINT32_MAX;
    uint32_t selected_blocks = 0;
    SparseSelectorCoefficients coefficients;
};

bool policy_selects(const CanonicalDerivedQ2KPolicy& policy, TensorRole role) {
    return std::find(policy.tensor_roles.begin(), policy.tensor_roles.end(), role) != policy.tensor_roles.end();
}

bool policy_selects(const CanonicalDerivedIQ2XXSPolicy& policy, TensorRole role) {
    return std::find(policy.tensor_roles.begin(), policy.tensor_roles.end(), role) != policy.tensor_roles.end();
}

bool policy_selects(const CanonicalDerivedColumnGroupedU2Policy& policy, TensorRole role) {
    return std::find(policy.tensor_roles.begin(), policy.tensor_roles.end(), role) !=
           policy.tensor_roles.end();
}

bool build_derived_column_grouped_u2_atlas(
    const RuntimePackage& package, const SemanticModel& model,
    const CanonicalDerivedColumnGroupedU2Policy& policy,
    std::optional<ColumnGroupedU2Atlas>& atlas, std::string& error) {
    if (policy.tensor_roles.empty()) return true;
    for (size_t index = 0; index != policy.tensor_roles.size(); ++index) {
        if (std::find(policy.tensor_roles.begin(), policy.tensor_roles.begin() + index,
                      policy.tensor_roles[index]) != policy.tensor_roles.begin() + index) {
            error = "derived column-grouped UInt2 policy repeats a tensor role";
            return false;
        }
    }
    bool uniform_importance = false;
#if defined(LAPLACE_METAL_TESTING)
    uniform_importance = policy.uniform_importance_for_testing;
#endif
    if (!uniform_importance && !policy.calibration_cache) {
        error = "derived column-grouped UInt2 conversion requires a validated calibration cache";
        return false;
    }

    std::vector<bool> executable_operator(model.operators.size(), true);
    for (const SemanticLayer& layer : model.layers) {
        if ((layer.flags & kSemanticLayerFlagSpeculative) == 0) continue;
        if (layer.first_operator > model.operators.size() ||
            layer.operator_count > model.operators.size() - layer.first_operator) {
            error = "derived column-grouped UInt2 found an invalid speculative layer range";
            return false;
        }
        for (uint32_t offset = 0; offset != layer.operator_count; ++offset)
            executable_operator[layer.first_operator + offset] = false;
    }

    struct Candidate {
        uint32_t tensor_id = UINT32_MAX;
        Tensor source;
        std::vector<float> importance;
    };
    std::vector<Candidate> candidates;
    try {
        candidates.reserve(model.tensors.size());
    } catch (const std::bad_alloc&) {
        error = "derived column-grouped UInt2 candidate allocation failed";
        return false;
    }
    for (const SemanticTensor& semantic : model.tensors) {
        if ((semantic.flags & kSemanticTensorFlagInactiveProgram) != 0 ||
            !policy_selects(policy, semantic.role)) continue;
        const SemanticOperator* linear = nullptr;
        for (size_t operator_index = 0; operator_index != model.operators.size();
             ++operator_index) {
            if (!executable_operator[operator_index]) continue;
            const SemanticOperator& candidate = model.operators[operator_index];
            if (candidate.kind != OperatorKind::Linear ||
                std::count(candidate.tensors.begin(), candidate.tensors.end(), semantic.id) != 1)
                continue;
            if (linear) {
                error = "derived column-grouped UInt2 tensor " +
                        std::to_string(semantic.id) + " has ambiguous Linear ownership";
                return false;
            }
            linear = &candidate;
        }
        if (!linear || linear->inputs.size() != 1 || linear->outputs.size() != 1) {
            error = "derived column-grouped UInt2 tensor " +
                    std::to_string(semantic.id) + " is not one executable Linear weight";
            return false;
        }
        Candidate candidate;
        candidate.tensor_id = semantic.id;
        if (!tensor_view(package, semantic, candidate.source) ||
            candidate.source.n_dims != 2 ||
            (candidate.source.type != GGMLType::Q4_K &&
             candidate.source.type != GGMLType::Q6_K) ||
            candidate.source.data_bytes == 0 || candidate.source.data_bytes > SIZE_MAX ||
            !candidate.source.data ||
            linear->inputs.front() >= model.values.size() ||
            model.values[linear->inputs.front()].dimensions.size() != 2 ||
            semantic.layout.axis_order[0] >= semantic.dimensions.size()) {
            error = "derived column-grouped UInt2 rejected tensor " +
                    std::to_string(semantic.id) + " physical or Linear input contract";
            return false;
        }
        CalibrationKMapping mapping;
        mapping.input_axis = 1;
        mapping.weight_physical_axis = 0;
        mapping.weight_logical_axis = semantic.layout.axis_order[0];
        mapping.width = static_cast<uint32_t>(candidate.source.dims[0]);
        mapping.block_elements = semantic.layout.block_elements;
        mapping.block_bytes = semantic.layout.block_bytes;
        if (uniform_importance) {
            try {
                candidate.importance.assign(mapping.width, 1.0f);
            } catch (const std::bad_alloc&) {
                error = "derived column-grouped UInt2 importance allocation failed";
                return false;
            }
        } else if (!calibration_importance_for_tensor(
                       *policy.calibration_cache, package.artifact_digest(),
                       package.fingerprint(), semantic.id, mapping,
                       candidate.importance)) {
            error = "derived column-grouped UInt2 calibration rejected tensor " +
                    std::to_string(semantic.id);
            return false;
        }
        candidates.push_back(std::move(candidate));
    }
    if (candidates.empty()) {
        error = "derived column-grouped UInt2 policy selected no eligible tensors";
        return false;
    }

    std::vector<ColumnGroupedU2AtlasSource> sources;
    try {
        sources.reserve(candidates.size());
    } catch (const std::bad_alloc&) {
        error = "derived column-grouped UInt2 source allocation failed";
        return false;
    }
    for (const Candidate& candidate : candidates) {
        sources.push_back({candidate.tensor_id, candidate.source.type,
                           std::span<const uint8_t>(candidate.source.data,
                                                    static_cast<size_t>(candidate.source.data_bytes)),
                           candidate.source.dims[0], candidate.source.dims[1],
                           candidate.importance});
    }
    ColumnGroupedU2AtlasResult built = build_column_grouped_u2_atlas(sources);
    if (const auto* atlas_error = std::get_if<ColumnGroupedU2AtlasError>(&built)) {
        error = "derived column-grouped UInt2 atlas failed with code " +
                std::to_string(static_cast<uint16_t>(*atlas_error));
        return false;
    }
    atlas.emplace(std::get<ColumnGroupedU2Atlas>(std::move(built)));
    return true;
}

const ColumnGroupedU2AtlasEntry* derived_column_grouped_u2_entry(
    const std::optional<ColumnGroupedU2Atlas>& atlas, uint32_t tensor_id) {
    if (!atlas) return nullptr;
    const auto& entries = atlas->entries();
    const auto found = std::find_if(entries.begin(), entries.end(), [&](const auto& entry) {
        return entry.binding_id == tensor_id;
    });
    return found == entries.end() ? nullptr : &*found;
}

Tensor derived_column_grouped_u2_tensor(const ColumnGroupedU2AtlasEntry& entry) {
    Tensor tensor;
    tensor.type = GGMLType::COLUMN_GROUPED_AFFINE_U2_SKIP_256;
    tensor.n_dims = 2;
    tensor.dims[0] = entry.storage.contract.logical_k;
    tensor.dims[1] = entry.storage.contract.logical_n;
    tensor.data = entry.storage.planes.values;
    tensor.scales = reinterpret_cast<const uint8_t*>(entry.storage.planes.scales);
    tensor.biases = reinterpret_cast<const uint8_t*>(entry.storage.planes.biases);
    tensor.data_bytes = entry.storage.contract.values_bytes;
    tensor.scale_bytes = entry.storage.contract.scale_bytes;
    tensor.bias_bytes = entry.storage.contract.bias_bytes;
    tensor.mlx_bits = 2;
    tensor.mlx_group_size = 256;
    return tensor;
}

bool build_derived_iq2_xxs_atlas(const RuntimePackage& package, const SemanticModel& model,
                                 const CanonicalDerivedIQ2XXSPolicy& policy,
                                 DerivedIQ2XXSAtlas& atlas, std::string& error) {
    if (policy.tensor_roles.empty()) return true;
    bool require_calibration = true;
#if defined(LAPLACE_METAL_TESTING)
    require_calibration = !policy.zero_fill_for_testing;
#endif
    if (require_calibration && !policy.calibration_cache) {
        error = "derived IQ2_XXS conversion requires a validated calibration cache";
        return false;
    }
    std::vector<bool> executed(model.tensors.size(), false);
    for (const SemanticLayer& layer : model.layers) {
        if ((layer.flags & kSemanticLayerFlagSpeculative) != 0 ||
            layer.first_operator > model.operators.size() ||
            layer.operator_count > model.operators.size() - layer.first_operator) continue;
        for (uint32_t index = 0; index != layer.operator_count; ++index) {
            for (uint32_t tensor_id : model.operators[layer.first_operator + index].tensors) {
                if (tensor_id < executed.size()) executed[tensor_id] = true;
            }
        }
    }

    uint64_t logical_length = 0;
    for (const SemanticTensor& semantic : model.tensors) {
        if (semantic.id >= executed.size() || !executed[semantic.id] ||
            !policy_selects(policy, semantic.role)) continue;
        Tensor source;
        if (!tensor_view(package, semantic, source) || source.n_dims != 2 ||
            source.dims[0] == 0 || source.dims[1] == 0 || source.dims[0] % 256 != 0 ||
            (source.type != GGMLType::Q4_K && source.type != GGMLType::Q6_K)) {
            error = "derived IQ2_XXS atlas rejected semantic tensor " + std::to_string(semantic.id) +
                    " role " + std::to_string(static_cast<uint16_t>(semantic.role));
            return false;
        }
        const uint64_t blocks_per_row = source.dims[0] / 256;
        constexpr uint64_t iq2_xxs_block_bytes = 66;
        if (source.dims[1] > UINT64_MAX / blocks_per_row ||
            source.dims[1] * blocks_per_row > UINT64_MAX / iq2_xxs_block_bytes) {
            error = "derived IQ2_XXS atlas tensor length overflows";
            return false;
        }
        const uint64_t length = source.dims[1] * blocks_per_row * iq2_xxs_block_bytes;
        if (logical_length > UINT64_MAX - 127 || (logical_length + 127) / 128 > UINT64_MAX / 128) {
            error = "derived IQ2_XXS atlas alignment overflows";
            return false;
        }
        const uint64_t offset = (logical_length + 127) / 128 * 128;
        if (offset > UINT64_MAX - length) {
            error = "derived IQ2_XXS atlas size overflows";
            return false;
        }
        DerivedIQ2XXSEntry entry;
        entry.tensor_id = semantic.id;
        entry.offset = offset;
        entry.logical_length = length;
        entry.calibration_mapping.input_axis = 1;
        entry.calibration_mapping.weight_physical_axis = 0;
        entry.calibration_mapping.weight_logical_axis = semantic.layout.axis_order[0];
        entry.calibration_mapping.width = static_cast<uint32_t>(source.dims[0]);
        entry.calibration_mapping.block_elements = semantic.layout.block_elements;
        entry.calibration_mapping.block_bytes = semantic.layout.block_bytes;
        if (require_calibration && !calibration_importance_for_tensor(
                *policy.calibration_cache, package.artifact_digest(), package.fingerprint(),
                entry.tensor_id, entry.calibration_mapping, entry.importance)) {
            error = "derived IQ2_XXS calibration rejected semantic tensor " +
                    std::to_string(entry.tensor_id);
            return false;
        }
        entry.tensor = source;
        atlas.entries.push_back(std::move(entry));
        logical_length = offset + length;
        if (atlas.source_bytes > UINT64_MAX - source.nbytes()) {
            error = "derived IQ2_XXS source byte count overflows";
            return false;
        }
        atlas.source_bytes += source.nbytes();
    }
    if (atlas.entries.empty()) {
        error = "derived IQ2_XXS policy selected no executed eligible tensors";
        return false;
    }
    const long page_value = sysconf(_SC_PAGESIZE);
    if (page_value <= 0 || logical_length > std::numeric_limits<size_t>::max()) {
        error = "derived IQ2_XXS atlas page contract is unavailable";
        return false;
    }
    const size_t page = static_cast<size_t>(page_value);
    if (static_cast<size_t>(logical_length) > std::numeric_limits<size_t>::max() - (page - 1)) {
        error = "derived IQ2_XXS atlas mapped length overflows";
        return false;
    }
    const size_t mapped_length = (static_cast<size_t>(logical_length) + page - 1) / page * page;
    void* mapped = mmap(nullptr, mapped_length, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mapped == MAP_FAILED) {
        error = "derived IQ2_XXS atlas allocation failed";
        return false;
    }
    atlas.mapping = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(mapped), [mapped_length](uint8_t* pointer) {
        munmap(pointer, mapped_length);
    });
    atlas.logical_length = static_cast<size_t>(logical_length);
    atlas.mapped_length = mapped_length;

    for (DerivedIQ2XXSEntry& entry : atlas.entries) {
        std::span<uint8_t> destination(atlas.mapping.get() + entry.offset,
                                       static_cast<size_t>(entry.logical_length));
#if defined(LAPLACE_METAL_TESTING)
        if (policy.zero_fill_for_testing) {
            entry.tensor.type = GGMLType::IQ2_XXS;
            entry.tensor.data = destination.data();
            entry.tensor.data_bytes = entry.logical_length;
            continue;
        }
#endif
        DerivedStorageError derived_error = DerivedStorageError::None;
        if (!derive_iq2_xxs_from_gguf(
                entry.tensor.type,
                std::span<const uint8_t>(entry.tensor.data, entry.tensor.nbytes()),
                entry.tensor.dims[0], entry.tensor.dims[1], entry.importance,
                destination, &entry.record, &derived_error)) {
            error = "derived IQ2_XXS conversion rejected semantic tensor " +
                    std::to_string(entry.tensor_id) + " error " +
                    std::to_string(static_cast<uint16_t>(derived_error));
            return false;
        }
        entry.tensor.type = GGMLType::IQ2_XXS;
        entry.tensor.data = destination.data();
        entry.tensor.data_bytes = entry.logical_length;
    }
    if (mprotect(atlas.mapping.get(), atlas.mapped_length, PROT_READ) != 0) {
        error = "derived IQ2_XXS atlas could not be sealed read-only";
        return false;
    }
    return true;
}

bool materialize_derived_q2(const RuntimePackage& package, const SemanticTensor& semantic,
                            const CanonicalDerivedQ2KPolicy& policy,
                            const DerivedIQ2XXSAtlas& iq2_xxs_atlas,
                            const std::optional<ColumnGroupedU2Atlas>& column_u2_atlas,
                            std::vector<DerivedTensorOwner>& owners, Tensor& output,
                            std::string& error) {
    if (!tensor_view(package, semantic, output)) return false;
    // The native affine UInt2 leaf consumes its three checked planes directly;
    // it is not a source for the derived Q2_K/Codebook policies.
    if (output.type == GGMLType::GROUPED_AFFINE_U2_256 ||
        output.type == GGMLType::COLUMN_GROUPED_AFFINE_U2_SKIP_256)
        return true;
    const ColumnGroupedU2AtlasEntry* column_u2 =
        derived_column_grouped_u2_entry(column_u2_atlas, semantic.id);
    const auto iq2_xxs = std::find_if(iq2_xxs_atlas.entries.begin(), iq2_xxs_atlas.entries.end(),
                                      [&](const DerivedIQ2XXSEntry& entry) {
                                          return entry.tensor_id == semantic.id;
                                      });
    if (column_u2) {
        if (policy_selects(policy, semantic.role) || iq2_xxs != iq2_xxs_atlas.entries.end()) {
            error = "semantic tensor " + std::to_string(semantic.id) +
                    " is selected by multiple derived quantization policies";
            return false;
        }
        output = derived_column_grouped_u2_tensor(*column_u2);
        return true;
    }
    if (iq2_xxs != iq2_xxs_atlas.entries.end()) {
        if (policy_selects(policy, semantic.role)) {
            error = "semantic tensor " + std::to_string(semantic.id) +
                    " is selected by both derived quantization policies";
            return false;
        }
        output = iq2_xxs->tensor;
        return true;
    }
    if (!policy_selects(policy, semantic.role)) return true;
    const auto existing = std::find_if(owners.begin(), owners.end(), [&](const DerivedTensorOwner& owner) {
        return owner.tensor_id == semantic.id;
    });
    if (existing != owners.end()) {
        output = existing->tensor;
        return true;
    }
    if (output.n_dims != 2) {
        error = "derived Q2_K requires an exact rank-2 tensor for semantic tensor " + std::to_string(semantic.id);
        return false;
    }
    const size_t source_length = output.nbytes();
    if (!source_length || !output.data) {
        error = "derived Q2_K has no checked source span for semantic tensor " + std::to_string(semantic.id);
        return false;
    }
    DerivedQ2KStorage derived;
    DerivedStorageError derived_error = DerivedStorageError::None;
    if (!derive_q2_k_from_gguf(output.type, std::span<const uint8_t>(output.data, source_length),
                               output.dims[0], output.dims[1], &derived, &derived_error)) {
        error = "derived Q2_K rejected semantic tensor " + std::to_string(semantic.id) +
                " source format " + type_name(output.type) + " error " +
                std::to_string(static_cast<uint16_t>(derived_error));
        return false;
    }
    const long page_value = sysconf(_SC_PAGESIZE);
    if (page_value <= 0 || derived.bytes.empty()) {
        error = "derived Q2_K could not determine a page-aligned storage size";
        return false;
    }
    const size_t page = static_cast<size_t>(page_value);
    if (derived.bytes.size() > std::numeric_limits<size_t>::max() - (page - 1)) {
        error = "derived Q2_K storage length overflows";
        return false;
    }
    const size_t mapped_length = (derived.bytes.size() + page - 1) / page * page;
    void* mapped = mmap(nullptr, mapped_length, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mapped == MAP_FAILED) {
        error = "derived Q2_K anonymous mapping failed";
        return false;
    }
    std::memcpy(mapped, derived.bytes.data(), derived.bytes.size());
    std::shared_ptr<uint8_t> mapping(static_cast<uint8_t*>(mapped), [mapped_length](uint8_t* pointer) {
        munmap(pointer, mapped_length);
    });

    DerivedTensorOwner owner;
    owner.tensor_id = semantic.id;
    owner.artifact_id = ArtifactId{0x80000000u + static_cast<uint32_t>(owners.size())};
    owner.contract = derived.contract;
    owner.source_digest = derived.source_digest;
    owner.storage_digest = derived.storage_digest;
    owner.mapping = std::move(mapping);
    owner.logical_length = derived.bytes.size();
    owner.mapped_length = mapped_length;
    owner.tensor = output;
    owner.tensor.type = GGMLType::Q2_K;
    owner.tensor.data = owner.mapping.get();
    owner.tensor.data_bytes = owner.logical_length;
    owners.push_back(std::move(owner));
    output = owners.back().tensor;
    return true;
}

bool rewrite_derived_semantics(SemanticModel& model, const std::vector<DerivedTensorOwner>& owners) {
    for (const DerivedTensorOwner& owner : owners) {
        if (owner.tensor_id >= model.tensors.size()) return false;
        SemanticTensor& tensor = model.tensors[owner.tensor_id];
        if (tensor.planes.size() != 1) return false;
        tensor.layout.kind = PhysicalLayoutKind::GgufBlocked;
        tensor.layout.packing = PackingKind::Gguf;
        tensor.layout.block_rank = 1;
        tensor.layout.block_elements = owner.contract.block_width;
        tensor.layout.block_bytes = owner.contract.bytes_per_block;
        tensor.quantization.kind = QuantizationKind::BlockedAffine;
        tensor.quantization.accumulation_type = ScalarType::F32;
        tensor.quantization.scale_type = ScalarType::F16;
        tensor.quantization.zero_type = ScalarType::F16;
        tensor.quantization.block_elements = owner.contract.block_width;
        tensor.quantization.block_bytes = owner.contract.bytes_per_block;
        tensor.quantization.group_size = owner.contract.block_width;
        tensor.quantization.required_plane_mask = 1;
        tensor.planes[0] = {PlaneKind::Values, ScalarType::U8, owner.artifact_id, 0,
                            owner.logical_length, owner.contract.alignment, 0};
    }
    return true;
}

bool rewrite_derived_semantics(SemanticModel& model, const DerivedIQ2XXSAtlas& atlas) {
    for (const DerivedIQ2XXSEntry& entry : atlas.entries) {
        if (entry.tensor_id >= model.tensors.size()) return false;
        SemanticTensor& tensor = model.tensors[entry.tensor_id];
        if (tensor.planes.size() != 1) return false;
        tensor.layout.kind = PhysicalLayoutKind::GgufBlocked;
        tensor.layout.packing = PackingKind::Gguf;
        tensor.layout.block_rank = 1;
        tensor.layout.block_elements = entry.record.contract.block_width;
        tensor.layout.block_bytes = entry.record.contract.bytes_per_block;
        tensor.quantization.kind = QuantizationKind::Codebook;
        tensor.quantization.accumulation_type = ScalarType::F32;
        tensor.quantization.scale_type = ScalarType::F16;
        tensor.quantization.zero_type = static_cast<ScalarType>(0);
        tensor.quantization.block_elements = entry.record.contract.block_width;
        tensor.quantization.block_bytes = entry.record.contract.bytes_per_block;
        tensor.quantization.group_size = 32;
        tensor.quantization.required_plane_mask = 1;
        tensor.planes[0] = {PlaneKind::Values, ScalarType::U8, atlas.artifact_id, entry.offset,
                            entry.logical_length, entry.record.contract.alignment, 0};
    }
    return true;
}

bool rewrite_derived_semantics(SemanticModel& model,
                               const std::optional<ColumnGroupedU2Atlas>& atlas) {
    if (!atlas) return true;
    constexpr ArtifactId kAtlasArtifact{0xa0000000u};
    for (const ColumnGroupedU2AtlasEntry& entry : atlas->entries()) {
        if (entry.binding_id >= model.tensors.size()) return false;
        SemanticTensor& tensor = model.tensors[entry.binding_id];
        if (tensor.planes.size() != 1 || tensor.dimensions.size() != 2) return false;
        const auto& contract = entry.storage.contract;
        tensor.layout.kind = PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip;
        tensor.layout.version = 1;
        tensor.layout.packing = PackingKind::LsbBitPacked;
        tensor.layout.rank = 2;
        tensor.layout.block_rank = 1;
        tensor.layout.axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        tensor.layout.strides = {1, contract.logical_k, 0, 0, 0, 0, 0, 0};
        tensor.layout.block_elements = contract.group_elements;
        tensor.layout.block_bytes = contract.packed_bytes_per_group;
        tensor.layout.flags = 0;
        tensor.quantization.kind = QuantizationKind::BlockedAffine;
        tensor.quantization.version = 1;
        tensor.quantization.accumulation_type = ScalarType::F32;
        tensor.quantization.scale_type = ScalarType::F16;
        tensor.quantization.zero_type = static_cast<ScalarType>(0);
        tensor.quantization.bias_type = ScalarType::F16;
        tensor.quantization.block_elements = contract.group_elements;
        tensor.quantization.block_bytes = contract.packed_bytes_per_group;
        tensor.quantization.group_size = contract.group_elements;
        tensor.quantization.required_plane_mask = 7;
        tensor.quantization.flags = 0;
        tensor.planes = {
            {PlaneKind::Values, ScalarType::U8, kAtlasArtifact,
             entry.values_offset, contract.values_bytes, contract.plane_alignment, 0},
            {PlaneKind::Scales, ScalarType::F16, kAtlasArtifact,
             entry.scales_offset, contract.scale_bytes, contract.plane_alignment, 0},
            {PlaneKind::Biases, ScalarType::F16, kAtlasArtifact,
             entry.biases_offset, contract.bias_bytes, contract.plane_alignment, 0},
        };
    }
    return true;
}

bool validate_sparse_ffn(uint32_t layer_index, const Tensor& gate, const Tensor& up,
                         const Tensor& down, const CanonicalSparseFfnPolicy& policy,
                         SparseFfnPlan& plan, std::string& error) {
    if (policy.runs.empty()) return true;
    const auto supported = [](GGMLType type) {
        return type == GGMLType::Q4_K || type == GGMLType::Q6_K;
    };
    if (!gate.data || !up.data || !down.data || gate.n_dims != 2 || up.n_dims != 2 || down.n_dims != 2 ||
        gate.dims[0] == 0 || gate.dims[1] == 0 || gate.dims[0] != up.dims[0] ||
        gate.dims[1] != up.dims[1] || down.dims[0] != gate.dims[1] ||
        down.dims[1] != gate.dims[0] || gate.dims[0] % 256 != 0 || gate.dims[1] % 256 != 0 ||
        !supported(gate.type) || !supported(up.type) || !supported(down.type)) {
        error = "sparse FFN original-span contract rejected layer " + std::to_string(layer_index);
        return false;
    }
    const uint64_t total_blocks = gate.dims[1] / 256;
    uint64_t selected_blocks = 0;
    uint64_t previous_end = 0;
    for (const SparseBlockRun& run : policy.runs) {
        const uint64_t end = static_cast<uint64_t>(run.first) + run.count;
        if (run.count == 0 || run.first < previous_end || end > total_blocks) {
            error = "sparse FFN run table rejected layer " + std::to_string(layer_index);
            return false;
        }
        if (selected_blocks > UINT32_MAX - run.count) {
            error = "sparse FFN selected block count overflows";
            return false;
        }
        selected_blocks += run.count;
        previous_end = end;
    }
    if (selected_blocks == 0 || selected_blocks > static_cast<uint64_t>(INT_MAX) / 256) {
        error = "sparse FFN selected width is unsupported";
        return false;
    }
    const uint64_t packed_intermediate = selected_blocks * 256;
    const auto projection_bytes = [](GGMLType type, uint64_t K, uint64_t N, uint64_t& bytes) {
        const int block = elements_per_block(type);
        const size_t block_bytes = bytes_per_block(type);
        if (block <= 0 || block_bytes == 0 || K % static_cast<uint64_t>(block) != 0) return false;
        const uint64_t blocks = K / static_cast<uint64_t>(block);
        if (blocks > UINT64_MAX / block_bytes) return false;
        const uint64_t row_bytes = blocks * block_bytes;
        if (N > UINT64_MAX / row_bytes) return false;
        bytes = N * row_bytes;
        return true;
    };
    uint64_t gate_bytes = 0, up_bytes = 0, down_bytes = 0;
    if (!projection_bytes(gate.type, gate.dims[0], packed_intermediate, gate_bytes) ||
        !projection_bytes(up.type, up.dims[0], packed_intermediate, up_bytes) ||
        !projection_bytes(down.type, packed_intermediate, down.dims[1], down_bytes) ||
        gate_bytes > UINT64_MAX - up_bytes || gate_bytes + up_bytes > UINT64_MAX - down_bytes) {
        error = "sparse FFN requested byte count overflows";
        return false;
    }
    const uint64_t gate_source = gate.nbytes();
    const uint64_t up_source = up.nbytes();
    const uint64_t down_source = down.nbytes();
    if (gate_source == 0 || up_source == 0 || down_source == 0 ||
        gate_source > UINT64_MAX - up_source || gate_source + up_source > UINT64_MAX - down_source) {
        error = "sparse FFN source byte count is invalid";
        return false;
    }
    plan.packed_intermediate = static_cast<uint32_t>(packed_intermediate);
    plan.source_bytes = gate_source + up_source + down_source;
    plan.requested_bytes = gate_bytes + up_bytes + down_bytes;
    return true;
}

bool build_sparse_ffn_proxy(uint32_t layer_index, const Tensor& gate, const Tensor& up,
                            const Tensor& down, uint32_t selected_blocks,
                            std::vector<SparseFfnProxyOwner>& owners, uint32_t& slot,
                            std::string& error) {
    if (owners.size() >= 256 || gate.nbytes() == 0 || up.nbytes() == 0 || down.nbytes() == 0) {
        error = "sparse FFN proxy resource limit rejected layer " + std::to_string(layer_index);
        return false;
    }
    SparseFfnProxyOwner owner;
    SparseSelectorCoeffError coefficient_error = SparseSelectorCoeffError::None;
    if (!build_sparse_selector_coefficients(
            gate, std::span<const uint8_t>(gate.data, gate.nbytes()),
            up, std::span<const uint8_t>(up.data, up.nbytes()),
            down, std::span<const uint8_t>(down.data, down.nbytes()),
            owner.coefficients, coefficient_error) ||
        selected_blocks == 0 || selected_blocks > owner.coefficients.output_blocks) {
        error = "sparse FFN proxy coefficients rejected layer " + std::to_string(layer_index) +
                " with code " + std::to_string(static_cast<uint16_t>(coefficient_error));
        return false;
    }
    owner.slot = static_cast<uint32_t>(owners.size());
    owner.layer_index = layer_index;
    owner.selected_blocks = selected_blocks;
    slot = owner.slot;
    owners.push_back(std::move(owner));
    return true;
}

bool assign_tensor(const RuntimePackage& package, const SemanticModel& model, const SemanticOperator& op,
                   TensorRole role, Tensor& output, bool required,
                   const CanonicalDerivedQ2KPolicy& policy,
                   const DerivedIQ2XXSAtlas& iq2_xxs_atlas,
                   const std::optional<ColumnGroupedU2Atlas>& column_u2_atlas,
                   std::vector<DerivedTensorOwner>& owners, std::string& error, bool expert = false) {
    const SemanticTensor* semantic = tensor_with_role(model, op, role, expert);
    if (!semantic) return !required;
    if (!materialize_derived_q2(package, *semantic, policy, iq2_xxs_atlas,
                                column_u2_atlas, owners, output, error))
        return false;
    return validate_linear_tensor_shape(model, op, output, role, expert, error);
}

struct ArtifactSpan {
    const uint8_t* base = nullptr;
    size_t length = 0;
};

struct RetainedArtifactRange {
    const uint8_t* base = nullptr;
    size_t length = 0;
};

struct PlaneRange {
    uint32_t artifact_id = UINT32_MAX;
    uint64_t begin = 0;
    uint64_t end = 0;
};

bool collect_retained_artifact_ranges(const RuntimePackage& package, const SemanticModel& model,
                                      std::span<const uint32_t> used_tensor_ids,
                                      std::span<const uint32_t> replaced_tensor_ids,
                                      std::vector<RetainedArtifactRange>& ranges,
                                      uint64_t& registered_bytes, uint64_t& boundary_bytes,
                                      CompatibilityReport& error) {
    const long page_value = sysconf(_SC_PAGESIZE);
    if (page_value <= 0) {
        error = metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                            "canonical Metal could not determine source registration page size");
        return false;
    }
    const uint64_t page = static_cast<uint64_t>(page_value);
    std::vector<PlaneRange> exact;
    for (uint32_t tensor_id : used_tensor_ids) {
        if (tensor_id >= model.tensors.size()) {
            error = metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                "canonical Metal retained source tensor ID is invalid");
            return false;
        }
        if (std::find(replaced_tensor_ids.begin(), replaced_tensor_ids.end(), tensor_id) !=
            replaced_tensor_ids.end()) continue;
        for (const TensorPlane& plane : model.tensors[tensor_id].planes) {
            const std::span<const uint8_t> artifact = package.artifact_bytes(plane.artifact_id);
            if (artifact.empty() || plane.offset > artifact.size() ||
                plane.length > artifact.size() - plane.offset || plane.length == 0) {
                error = metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                    "canonical Metal retained source plane is outside its artifact");
                return false;
            }
            exact.push_back({plane.artifact_id.value, plane.offset, plane.offset + plane.length});
        }
    }
    if (exact.empty()) {
        error = metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                            "canonical Metal retained source set is empty");
        return false;
    }
    std::sort(exact.begin(), exact.end(), [](const PlaneRange& left, const PlaneRange& right) {
        return left.artifact_id < right.artifact_id ||
               (left.artifact_id == right.artifact_id && left.begin < right.begin);
    });
    std::vector<PlaneRange> exact_union;
    for (const PlaneRange& range : exact) {
        if (!exact_union.empty() && exact_union.back().artifact_id == range.artifact_id &&
            range.begin <= exact_union.back().end) {
            exact_union.back().end = std::max(exact_union.back().end, range.end);
        } else {
            exact_union.push_back(range);
        }
    }
    std::vector<PlaneRange> page_union;
    for (PlaneRange range : exact_union) {
        if (range.end > UINT64_MAX - (page - 1)) {
            error = metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                                "canonical Metal retained source page range overflows");
            return false;
        }
        range.begin = range.begin / page * page;
        range.end = (range.end + page - 1) / page * page;
        if (!page_union.empty() && page_union.back().artifact_id == range.artifact_id &&
            range.begin <= page_union.back().end) {
            page_union.back().end = std::max(page_union.back().end, range.end);
        } else {
            page_union.push_back(range);
        }
    }
    uint64_t logical_bytes = 0;
    for (const PlaneRange& range : exact_union) {
        const uint64_t length = range.end - range.begin;
        if (logical_bytes > UINT64_MAX - length) {
            error = metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                                "canonical Metal retained source logical byte count overflows");
            return false;
        }
        logical_bytes += length;
    }
    for (const PlaneRange& range : page_union) {
        const std::span<const uint8_t> artifact = package.artifact_bytes(ArtifactId{range.artifact_id});
        if (artifact.empty() || range.begin >= artifact.size() || range.end <= range.begin ||
            range.begin > SIZE_MAX || artifact.size() - static_cast<size_t>(range.begin) == 0) {
            error = metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                "canonical Metal retained source page range is invalid");
            return false;
        }
        const uint64_t registered = range.end - range.begin;
        if (registered_bytes > UINT64_MAX - registered) {
            error = metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                                "canonical Metal retained source registered byte count overflows");
            return false;
        }
        registered_bytes += registered;
        const size_t begin = static_cast<size_t>(range.begin);
        const size_t length = artifact.size() - begin < registered
            ? artifact.size() - begin : static_cast<size_t>(registered);
        ranges.push_back({artifact.data() + begin, length});
    }
    if (registered_bytes < logical_bytes) {
        error = metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                            "canonical Metal retained source byte accounting is invalid");
        return false;
    }
    boundary_bytes = registered_bytes - logical_bytes;
    return true;
}

bool collect_referenced_artifacts(const RuntimePackage& package, const SemanticModel& model,
                                  std::vector<ArtifactSpan>& spans, CompatibilityReport& error) {
    std::vector<uint32_t> ids;
    for (const SemanticTensor& tensor : model.tensors) {
        if ((tensor.flags & kSemanticTensorFlagInactiveProgram) != 0) continue;
        for (const TensorPlane& plane : tensor.planes) {
            if (plane.artifact_id.value == UINT32_MAX) {
                error = metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                    "canonical Metal references an invalid artifact ID");
                return false;
            }
            bool known = false;
            for (uint32_t id : ids) known |= id == plane.artifact_id.value;
            if (known) continue;
            const std::span<const uint8_t> bytes = package.artifact_bytes(plane.artifact_id);
            if (bytes.empty()) {
                error = metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                    "canonical Metal references unavailable artifact " +
                                    std::to_string(plane.artifact_id.value));
                return false;
            }
            ids.push_back(plane.artifact_id.value);
            spans.push_back({bytes.data(), bytes.size()});
        }
    }
    if (spans.empty()) {
        error = metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                            "canonical Metal model has no artifact-backed tensors");
        return false;
    }
    return true;
}

} // namespace

Sha256Digest calibration_bytes_digest(std::span<const uint8_t> bytes) {
    return sha256_bytes(bytes);
}

Sha256Digest calibration_target_set_digest(std::span<const CalibrationTarget> targets) {
    std::vector<uint8_t> bytes = {'L','A','P','C','A','L','T','1'};
    append_u32(bytes, static_cast<uint32_t>(targets.size()));
    for (const CalibrationTarget& target : targets) append_target(bytes, target);
    return sha256_bytes(bytes);
}

bool calibration_token_window_digest(std::span<const uint32_t> token_ids,
                                     std::span<const uint32_t> window_lengths,
                                     Sha256Digest& digest) {
    if (token_ids.empty() || window_lengths.empty() || token_ids.size() > UINT32_MAX ||
        window_lengths.size() > UINT32_MAX) return false;
    uint64_t total = 0;
    for (uint32_t length : window_lengths) {
        if (length == 0 || total > UINT64_MAX - length) return false;
        total += length;
    }
    if (total != token_ids.size()) return false;
    std::vector<uint8_t> bytes = {'L','A','P','C','A','L','W','1'};
    append_u32(bytes, static_cast<uint32_t>(token_ids.size()));
    for (uint32_t token : token_ids) append_u32(bytes, token);
    append_u32(bytes, static_cast<uint32_t>(window_lengths.size()));
    for (uint32_t length : window_lengths) append_u32(bytes, length);
    digest = sha256_bytes(bytes);
    return true;
}

namespace {
constexpr uint64_t kMaximumCalibrationValues = 1ull << 28;

bool calibration_value_count_valid(uint64_t total, uint64_t next) {
    return total <= kMaximumCalibrationValues && next <= kMaximumCalibrationValues &&
           total <= kMaximumCalibrationValues - next;
}
}

#if defined(LAPLACE_METAL_TESTING)
bool calibration_cache_value_count_valid_for_testing(uint64_t total, uint64_t next) {
    return calibration_value_count_valid(total, next);
}

bool canonical_metal_iq2_atlas_cache_roundtrip_for_testing(std::string_view path,
                                                           uint32_t* conversions) {
    if (!conversions || path.empty()) return false;
    *conversions = 0;

    CalibrationTarget target;
    target.operator_id = 3;
    target.input_value_id = 5;
    target.weight_tensor_id = 7;
    target.k_mapping.input_axis = 1;
    target.k_mapping.weight_physical_axis = 0;
    target.k_mapping.weight_logical_axis = 0;
    target.k_mapping.width = 256;
    target.k_mapping.block_elements = 256;
    target.k_mapping.block_bytes = sizeof(kernels::block_q4_K);

    CalibrationCacheBundle calibration;
    calibration.artifact_digest.bytes[0] = 0x11;
    calibration.semantic_fingerprint.bytes[0] = 0x22;
    calibration.corpus_digest.bytes[0] = 0x33;
    calibration.token_digest.bytes[0] = 0x44;
    CalibrationRecord sample;
    sample.target = target;
    sample.sample_count = 4;
    sample.sum_squares.resize(256);
    for (size_t index = 0; index != sample.sum_squares.size(); ++index)
        sample.sum_squares[index] = index == 17 ? 0.0f : 1.0f + static_cast<float>(index % 13);
    calibration.records.push_back(sample);
    calibration.target_set_digest = calibration_target_set_digest(
        std::span<const CalibrationTarget>(&calibration.records[0].target, 1));

    std::vector<float> importance;
    if (!calibration_importance_for_tensor(
            calibration, calibration.artifact_digest, calibration.semantic_fingerprint,
            target.weight_tensor_id, target.k_mapping, importance) ||
        importance.size() != 256 || importance[17] != 0.0f) return false;

    kernels::block_q4_K source{};
    source.d = fp32_to_fp16(0.5f);
    source.dmin = fp32_to_fp16(0.125f);
    for (size_t index = 0; index != sizeof(source.scales); ++index)
        source.scales[index] = static_cast<uint8_t>(index * 9 + 3);
    for (size_t index = 0; index != sizeof(source.qs); ++index)
        source.qs[index] = static_cast<uint8_t>(index * 17 + 5);
    const std::span<const uint8_t> source_bytes(
        reinterpret_cast<const uint8_t*>(&source), sizeof(source));
    const std::span<const uint8_t> importance_bytes(
        reinterpret_cast<const uint8_t*>(importance.data()),
        importance.size() * sizeof(importance[0]));
    const Sha256Digest source_digest = sha256_bytes(source_bytes);
    const Sha256Digest importance_digest = sha256_bytes(importance_bytes);

    std::vector<uint8_t> expected_prefix = {'L','A','P','I','Q','2','C','1'};
    append_u16(expected_prefix, 1);
    append_u16(expected_prefix, calibration.converter_version);
    append_u16(expected_prefix, calibration.format_version);
    append_u16(expected_prefix, 0);
    append_u32(expected_prefix, static_cast<uint32_t>(GGMLType::Q4_K));
    append_u32(expected_prefix, 256);
    append_u32(expected_prefix, 1);
    append_u32(expected_prefix, sizeof(kernels::block_iq2_xxs));
    const std::array<const Sha256Digest*, 7> cache_digests = {
        &calibration.artifact_digest, &calibration.semantic_fingerprint,
        &calibration.target_set_digest, &calibration.corpus_digest,
        &calibration.token_digest, &source_digest, &importance_digest};
    for (const Sha256Digest* digest : cache_digests)
        expected_prefix.insert(expected_prefix.end(), digest->bytes.begin(), digest->bytes.end());

    const std::string cache_path(path);
    const int cached_fd = open(cache_path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (cached_fd >= 0) {
        struct stat state{};
        bool ok = fstat(cached_fd, &state) == 0 && S_ISREG(state.st_mode) &&
                  state.st_size > 0 && state.st_size <= 4096;
        std::vector<uint8_t> cached(ok ? static_cast<size_t>(state.st_size) : 0);
        size_t read_offset = 0;
        while (ok && read_offset != cached.size()) {
            const ssize_t count = read(cached_fd, cached.data() + read_offset,
                                       cached.size() - read_offset);
            if (count <= 0) ok = false;
            else read_offset += static_cast<size_t>(count);
        }
        close(cached_fd);
        constexpr size_t storage_digest_bytes = 32;
        constexpr size_t checksum_bytes = 32;
        ok = ok && cached.size() == expected_prefix.size() + storage_digest_bytes +
                                  sizeof(kernels::block_iq2_xxs) + checksum_bytes &&
             std::equal(expected_prefix.begin(), expected_prefix.end(), cached.begin());
        if (ok) {
            const size_t payload_size = cached.size() - checksum_bytes;
            Sha256Digest stored_checksum;
            std::memcpy(stored_checksum.bytes.data(), cached.data() + payload_size,
                        checksum_bytes);
            ok = sha256_bytes(std::span<const uint8_t>(cached.data(), payload_size)) ==
                 stored_checksum;
        }
        return ok;
    }

    std::array<uint8_t, sizeof(kernels::block_iq2_xxs)> converted{};
    DerivedIQ2XXSRecord record;
    DerivedStorageError derived_error = DerivedStorageError::None;
    if (!derive_iq2_xxs_from_gguf(GGMLType::Q4_K, source_bytes, 256, 1, importance,
                                  converted, &record, &derived_error) ||
        derived_error != DerivedStorageError::None ||
        record.source_digest != source_digest.bytes ||
        record.importance_digest != importance_digest.bytes) return false;
    *conversions = 1;

    std::vector<uint8_t> encoded = expected_prefix;
    encoded.insert(encoded.end(), record.storage_digest.begin(), record.storage_digest.end());
    encoded.insert(encoded.end(), converted.begin(), converted.end());
    const Sha256Digest checksum = sha256_bytes(encoded);
    encoded.insert(encoded.end(), checksum.bytes.begin(), checksum.bytes.end());

    const std::string temporary = cache_path + ".tmp." + std::to_string(getpid());
    const int output = open(temporary.c_str(),
                            O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (output < 0) return false;
    size_t written = 0;
    bool ok = true;
    while (written != encoded.size()) {
        const ssize_t count = write(output, encoded.data() + written, encoded.size() - written);
        if (count <= 0) { ok = false; break; }
        written += static_cast<size_t>(count);
    }
    if (ok) ok = fsync(output) == 0;
    if (close(output) != 0) ok = false;
    if (ok) ok = rename(temporary.c_str(), cache_path.c_str()) == 0;
    if (!ok) unlink(temporary.c_str());
    return ok;
}
#endif

bool encode_calibration_cache(const CalibrationCacheBundle& cache, std::vector<uint8_t>& bytes) {
    bytes.clear();
    if (cache.version != 1 || cache.accumulator_version != 1 ||
        cache.converter_version != 1 || cache.format_version != 1 ||
        cache.records.empty() || cache.records.size() > 4096) return false;
    std::vector<CalibrationTarget> targets;
    targets.reserve(cache.records.size());
    uint64_t total_values = 0;
    for (const CalibrationRecord& record : cache.records) {
        if (record.target.version != 1 || record.target.flags != 0 ||
            record.target.k_mapping.reserved != 0 || record.sample_count == 0 ||
            record.target.k_mapping.width == 0 ||
            record.sum_squares.size() != record.target.k_mapping.width ||
            !calibration_value_count_valid(total_values, record.sum_squares.size()) ||
            std::find(targets.begin(), targets.end(), record.target) != targets.end()) return false;
        for (float value : record.sum_squares)
            if (!std::isfinite(value) || value < 0.0f) return false;
        total_values += record.sum_squares.size();
        targets.push_back(record.target);
    }
    if (calibration_target_set_digest(targets) != cache.target_set_digest) return false;
    static constexpr uint8_t magic[] = {'L','A','P','C','A','L','0','1'};
    bytes.insert(bytes.end(), std::begin(magic), std::end(magic));
    append_u16(bytes, cache.version);
    append_u16(bytes, cache.accumulator_version);
    append_u16(bytes, cache.converter_version);
    append_u16(bytes, cache.format_version);
    append_u32(bytes, static_cast<uint32_t>(cache.records.size()));
    for (const Sha256Digest* digest : {&cache.artifact_digest, &cache.semantic_fingerprint,
                                       &cache.target_set_digest, &cache.corpus_digest,
                                       &cache.token_digest}) {
        bytes.insert(bytes.end(), digest->bytes.begin(), digest->bytes.end());
    }
    for (const CalibrationRecord& record : cache.records) {
        append_target(bytes, record.target);
        append_u32(bytes, record.sample_count);
        append_u32(bytes, static_cast<uint32_t>(record.sum_squares.size()));
        for (float value : record.sum_squares) {
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            append_u32(bytes, bits);
        }
    }
    const Sha256Digest checksum = sha256_bytes(bytes);
    bytes.insert(bytes.end(), checksum.bytes.begin(), checksum.bytes.end());
    return true;
}

bool normalize_calibration_record(const CalibrationRecord& record,
                                  std::vector<float>& importance) {
    importance.clear();
    if (record.sample_count == 0 || record.target.k_mapping.width == 0 ||
        record.sum_squares.size() != record.target.k_mapping.width) return false;
    importance.reserve(record.sum_squares.size());
    const float inverse_count = 1.0f / static_cast<float>(record.sample_count);
    for (float sum : record.sum_squares) {
        if (!std::isfinite(sum) || sum < 0.0f) {
            importance.clear();
            return false;
        }
        const float value = sum * inverse_count;
        if (!std::isfinite(value)) {
            importance.clear();
            return false;
        }
        importance.push_back(value);
    }
    return true;
}

bool calibration_importance_for_tensor(const CalibrationCacheBundle& cache,
                                       const Sha256Digest& artifact_digest,
                                       const Sha256Digest& semantic_fingerprint,
                                       uint32_t weight_tensor_id,
                                       const CalibrationKMapping& expected_mapping,
                                       std::vector<float>& importance) {
    importance.clear();
    if (cache.version != 1 || cache.accumulator_version != 1 ||
        cache.converter_version != 1 || cache.format_version != 1 ||
        cache.artifact_digest != artifact_digest ||
        cache.semantic_fingerprint != semantic_fingerprint || cache.records.empty()) return false;
    std::vector<CalibrationTarget> targets;
    targets.reserve(cache.records.size());
    const CalibrationRecord* match = nullptr;
    for (const CalibrationRecord& record : cache.records) {
        targets.push_back(record.target);
        if (record.target.weight_tensor_id != weight_tensor_id) continue;
        if (match) return false;
        match = &record;
    }
    if (!match || calibration_target_set_digest(targets) != cache.target_set_digest ||
        match->target.k_mapping != expected_mapping ||
        !normalize_calibration_record(*match, importance)) {
        importance.clear();
        return false;
    }
    return true;
}

bool merge_calibration_records(std::vector<CalibrationRecord>& accumulated,
                               std::span<const CalibrationRecord> window) {
    if (window.empty()) return false;
    const auto valid = [](const CalibrationRecord& record) {
        if (record.sample_count == 0 || record.target.k_mapping.width == 0 ||
            record.sum_squares.size() != record.target.k_mapping.width) return false;
        for (float value : record.sum_squares)
            if (!std::isfinite(value) || value < 0.0f) return false;
        return true;
    };
    for (size_t index = 0; index != window.size(); ++index) {
        if (!valid(window[index])) return false;
        for (size_t prior = 0; prior != index; ++prior)
            if (window[prior].target == window[index].target) return false;
    }
    if (accumulated.empty()) {
        accumulated.assign(window.begin(), window.end());
        return true;
    }
    if (accumulated.size() != window.size()) return false;
    for (size_t index = 0; index != window.size(); ++index) {
        const CalibrationRecord& current = accumulated[index];
        const CalibrationRecord& incoming = window[index];
        if (!valid(current) || current.target != incoming.target ||
            current.sample_count > UINT32_MAX - incoming.sample_count) return false;
        for (size_t value = 0; value != current.sum_squares.size(); ++value)
            if (!std::isfinite(current.sum_squares[value] + incoming.sum_squares[value])) return false;
    }
    for (size_t index = 0; index != window.size(); ++index) {
        CalibrationRecord& current = accumulated[index];
        const CalibrationRecord& incoming = window[index];
        current.sample_count += incoming.sample_count;
        for (size_t value = 0; value != current.sum_squares.size(); ++value)
            current.sum_squares[value] += incoming.sum_squares[value];
    }
    return true;
}

bool write_calibration_cache_atomic(std::string_view path, const CalibrationCacheBundle& cache,
                                    CompatibilityReport* error) {
    const auto fail = [&](CompatibilityError code, const char* detail) {
        if (error) *error = calibration_cache_error(code, detail);
        return false;
    };
    if (path.empty()) return fail(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                  "calibration cache path is empty");
    std::vector<uint8_t> bytes;
    if (!encode_calibration_cache(cache, bytes))
        return fail(CompatibilityError::IR_REFERENCE_INVALID,
                    "calibration cache bundle is invalid");
    const std::string destination(path);
    const std::string temporary = destination + ".tmp." + std::to_string(getpid());
    const int fd = open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) return fail(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                            "calibration cache temporary file could not be created");
    size_t written = 0;
    bool ok = true;
    while (written != bytes.size()) {
        const ssize_t count = write(fd, bytes.data() + written, bytes.size() - written);
        if (count <= 0) { ok = false; break; }
        written += static_cast<size_t>(count);
    }
    if (ok) ok = fsync(fd) == 0;
    if (close(fd) != 0) ok = false;
    if (ok) ok = rename(temporary.c_str(), destination.c_str()) == 0;
    if (!ok) {
        unlink(temporary.c_str());
        return fail(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                    "calibration cache atomic write failed");
    }
    return true;
}

CalibrationCacheDecode load_calibration_cache(std::string_view path,
                                              const Sha256Digest& artifact_digest,
                                              const Sha256Digest& semantic_fingerprint,
                                              const Sha256Digest& target_set_digest,
                                              const Sha256Digest& corpus_digest,
                                              const Sha256Digest& token_digest) {
    if (path.empty()) return calibration_cache_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                                      "calibration cache path is empty");
    const std::string source(path);
    const int fd = open(source.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return calibration_cache_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                               "calibration cache is absent");
    struct stat before{};
    if (fstat(fd, &before) != 0 || !S_ISREG(before.st_mode) || before.st_size <= 0 ||
        static_cast<uint64_t>(before.st_size) > (1ull << 30)) {
        close(fd);
        return calibration_cache_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                       "calibration cache source is invalid");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(before.st_size));
    size_t offset = 0;
    bool ok = true;
    while (offset != bytes.size()) {
        const ssize_t count = read(fd, bytes.data() + offset, bytes.size() - offset);
        if (count <= 0) { ok = false; break; }
        offset += static_cast<size_t>(count);
    }
    struct stat after{};
    ok = ok && fstat(fd, &after) == 0 && before.st_dev == after.st_dev &&
         before.st_ino == after.st_ino && before.st_size == after.st_size &&
         before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec &&
         before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec;
    close(fd);
    if (!ok) return calibration_cache_error(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH,
                                             "calibration cache changed while reading");
    return decode_calibration_cache(bytes, artifact_digest, semantic_fingerprint,
                                    target_set_digest, corpus_digest, token_digest);
}

CalibrationCacheDecode decode_calibration_cache(std::span<const uint8_t> bytes,
                                                const Sha256Digest& artifact_digest,
                                                const Sha256Digest& semantic_fingerprint,
                                                const Sha256Digest& target_set_digest,
                                                const Sha256Digest& corpus_digest,
                                                const Sha256Digest& token_digest) {
    constexpr size_t minimum_size = 8 + 4 * sizeof(uint16_t) + sizeof(uint32_t) +
                                    5 * 32 + 32;
    if (bytes.size() < minimum_size) {
        return calibration_cache_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                       "calibration cache is truncated");
    }
    const size_t payload_size = bytes.size() - 32;
    Sha256Digest stored_checksum;
    std::memcpy(stored_checksum.bytes.data(), bytes.data() + payload_size, 32);
    if (sha256_bytes(bytes.first(payload_size)) != stored_checksum) {
        return calibration_cache_error(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH,
                                       "calibration cache checksum differs");
    }
    static constexpr uint8_t magic[] = {'L','A','P','C','A','L','0','1'};
    if (!std::equal(std::begin(magic), std::end(magic), bytes.begin())) {
        return calibration_cache_error(CompatibilityError::IR_VERSION_UNSUPPORTED,
                                       "calibration cache magic differs");
    }
    size_t offset = sizeof(magic);
    CalibrationCacheBundle cache;
    uint32_t record_count = 0;
    if (!read_u16(bytes, offset, cache.version) ||
        !read_u16(bytes, offset, cache.accumulator_version) ||
        !read_u16(bytes, offset, cache.converter_version) ||
        !read_u16(bytes, offset, cache.format_version) ||
        !read_u32(bytes, offset, record_count) ||
        !read_digest(bytes, offset, cache.artifact_digest) ||
        !read_digest(bytes, offset, cache.semantic_fingerprint) ||
        !read_digest(bytes, offset, cache.target_set_digest) ||
        !read_digest(bytes, offset, cache.corpus_digest) ||
        !read_digest(bytes, offset, cache.token_digest)) {
        return calibration_cache_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                       "calibration cache header is truncated");
    }
    if (cache.version != 1 || cache.accumulator_version != 1 ||
        cache.converter_version != 1 || cache.format_version != 1) {
        return calibration_cache_error(CompatibilityError::IR_VERSION_UNSUPPORTED,
                                       "calibration cache contract version differs");
    }
    if (record_count == 0 || record_count > 4096 ||
        cache.artifact_digest != artifact_digest ||
        cache.semantic_fingerprint != semantic_fingerprint ||
        cache.target_set_digest != target_set_digest ||
        cache.corpus_digest != corpus_digest || cache.token_digest != token_digest) {
        return calibration_cache_error(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH,
                                       "calibration cache key differs");
    }
    cache.records.reserve(record_count);
    std::vector<CalibrationTarget> targets;
    targets.reserve(record_count);
    for (uint32_t index = 0; index != record_count; ++index) {
        CalibrationRecord record;
        uint32_t value_count = 0;
        if (!read_target(bytes.first(payload_size), offset, record.target) ||
            !read_u32(bytes.first(payload_size), offset, record.sample_count) ||
            !read_u32(bytes.first(payload_size), offset, value_count) ||
            record.target.version != 1 || record.target.flags != 0 ||
            record.target.k_mapping.reserved != 0 || record.sample_count == 0 ||
            value_count == 0 || value_count != record.target.k_mapping.width ||
            value_count > (1u << 24) || offset > payload_size ||
            static_cast<size_t>(value_count) > (payload_size - offset) / sizeof(uint32_t)) {
            return calibration_cache_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                           "calibration cache record is invalid");
        }
        record.sum_squares.resize(value_count);
        for (uint32_t value_index = 0; value_index != value_count; ++value_index) {
            uint32_t bits = 0;
            if (!read_u32(bytes.first(payload_size), offset, bits)) {
                return calibration_cache_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                               "calibration cache values are truncated");
            }
            std::memcpy(&record.sum_squares[value_index], &bits, sizeof(bits));
            if (!std::isfinite(record.sum_squares[value_index]) ||
                record.sum_squares[value_index] < 0.0f) {
                return calibration_cache_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE,
                                               "calibration cache contains an invalid accumulator");
            }
        }
        targets.push_back(record.target);
        cache.records.push_back(std::move(record));
    }
    if (offset != payload_size || calibration_target_set_digest(targets) != cache.target_set_digest) {
        return calibration_cache_error(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH,
                                       "calibration cache target set differs");
    }
    return cache;
}

CalibrationTargetValidation validate_calibration_target(const SemanticModel& model,
                                                         const CalibrationTarget& target,
                                                         bool metal_capability_available) {
    if (target.version != 1 || target.flags != 0 || target.k_mapping.reserved != 0) {
        return calibration_error(CompatibilityError::IR_VERSION_UNSUPPORTED, target,
                                 "calibration target version or flags are unsupported");
    }
    if (!metal_capability_available) {
        return calibration_error(CompatibilityError::CAPABILITY_MISSING, target,
                                 "calibration target requires a Metal accumulation pipeline");
    }
    const SemanticOperator* op = nullptr;
    for (const SemanticOperator& candidate : model.operators) {
        if (candidate.id != target.operator_id) continue;
        if (op) return calibration_error(CompatibilityError::IR_REFERENCE_INVALID, target,
                                         "calibration operator id is not unique");
        op = &candidate;
    }
    const SemanticValue* input = nullptr;
    for (const SemanticValue& candidate : model.values) {
        if (candidate.id != target.input_value_id) continue;
        if (input) return calibration_error(CompatibilityError::IR_REFERENCE_INVALID, target,
                                            "calibration input value id is not unique");
        input = &candidate;
    }
    const SemanticTensor* weight = nullptr;
    for (const SemanticTensor& candidate : model.tensors) {
        if (candidate.id != target.weight_tensor_id) continue;
        if (weight) return calibration_error(CompatibilityError::IR_REFERENCE_INVALID, target,
                                             "calibration weight tensor id is not unique");
        weight = &candidate;
    }
    if (!op || !input || !weight || op->kind != OperatorKind::Linear ||
        !std::holds_alternative<LinearPayload>(op->payload) ||
        op->inputs.size() != 1 || op->inputs.front() != input->id ||
        std::count(op->tensors.begin(), op->tensors.end(), weight->id) != 1) {
        return calibration_error(CompatibilityError::IR_REFERENCE_INVALID, target,
                                 "calibration target does not identify one linear input and weight");
    }
    const auto* linear = std::get_if<LinearPayload>(&op->payload);
    if (!linear || linear->transpose_weight || target.k_mapping.input_axis >= input->dimensions.size() ||
        target.k_mapping.weight_physical_axis >= weight->layout.rank ||
        target.k_mapping.weight_physical_axis >= weight->layout.axis_order.size() ||
        target.k_mapping.weight_logical_axis >= weight->dimensions.size() ||
        weight->layout.axis_order[target.k_mapping.weight_physical_axis] !=
            target.k_mapping.weight_logical_axis || target.k_mapping.weight_logical_axis != 0) {
        return calibration_error(CompatibilityError::IR_LAYOUT_MISMATCH, target,
                                 "calibration target K-axis mapping does not match the linear weight");
    }
    uint64_t input_width = 0;
    uint64_t weight_width = 0;
    if (!constant_dimension(input->dimensions[target.k_mapping.input_axis], input_width) ||
        !constant_dimension(weight->dimensions[target.k_mapping.weight_logical_axis], weight_width) ||
        input_width != weight_width || input_width != target.k_mapping.width || input_width > UINT32_MAX) {
        return calibration_error(CompatibilityError::IR_SHAPE_MISMATCH, target,
                                 "calibration target width does not match the linear K dimension");
    }
    uint32_t block_elements = 1;
    uint32_t block_bytes = 0;
    if (weight->layout.kind == PhysicalLayoutKind::GgufBlocked) {
        block_elements = weight->layout.block_elements;
        block_bytes = weight->layout.block_bytes;
    } else if (weight->layout.kind == PhysicalLayoutKind::ContiguousRowMajor &&
               weight->planes.size() == 1 && weight->planes[0].kind == PlaneKind::Values) {
        block_bytes = weight->planes[0].storage_type == ScalarType::F32 ? 4 :
                      weight->planes[0].storage_type == ScalarType::F16 ? 2 : 0;
    }
    if (block_elements == 0 || block_bytes == 0 ||
        target.k_mapping.block_elements != block_elements ||
        target.k_mapping.block_bytes != block_bytes) {
        return calibration_error(CompatibilityError::IR_LAYOUT_MISMATCH, target,
                                 "calibration target physical block mapping differs from the weight");
    }
    return target;
}

struct CanonicalMetalProgram::Impl {
    struct CalibrationBinding {
        CalibrationTarget target;
        uint32_t slot = UINT32_MAX;
        bool attached = false;
    };

    std::shared_ptr<const RuntimePackage> package;
    Tensor embedding;
    Tensor final_norm;
    Tensor output;
    std::vector<std::variant<DenseLayer, RecurrentLayer>> layers;
    uint32_t hidden = 0;
    uint32_t vocabulary = 0;
    uint32_t maximum_context = 0;
    uint32_t maximum_batch_rows = 0;
    float embedding_scale = 1.0f;
    bool enable_prefill = false;
    bool enable_decode = false;
    uint32_t token_intermediate = 0;
    uint32_t attention_query_capacity = 0;
    uint32_t attention_key_value_capacity = 0;
    uint64_t attention_key_value_width_sum = 0;
    uint32_t moe_exp_intermediate = 0;
    uint32_t moe_selected_experts = 0;
    uint32_t moe_experts = 0;
    bool has_recurrent = false;
    bool has_attention = false;
    bool has_moe = false;
    bool prefill_batch = false;
    uint32_t position = 0;
    float final_epsilon = 0.0f;
    uint64_t session_id = 0;
    uint64_t generation = 1;
    std::shared_ptr<MetalTokSession> metal_session;
    ExecutionPlan plan;
    ExecutionPlan token_plan;
    std::vector<DerivedTensorOwner> derived_q2;
    DerivedIQ2XXSAtlas derived_iq2_xxs;
    std::optional<ColumnGroupedU2Atlas> derived_column_grouped_u2;
    std::vector<MetalSparseBlockRun> sparse_ffn_runs;
    std::vector<uint32_t> sparse_ffn_block_ids;
    std::vector<uint32_t> sparse_ffn_block_offsets;
    std::vector<uint32_t> sparse_ffn_block_counts;
    std::vector<SparseFfnProxyOwner> sparse_ffn_proxies;
    std::vector<CalibrationBinding> calibration;
    std::vector<uint32_t> calibration_widths;
    uint32_t sparse_ffn_layers = 0;
    uint64_t sparse_ffn_source = 0;
    uint64_t sparse_ffn_requested = 0;
    uint64_t original_source_registered = 0;
    uint64_t derived_q2_registered = 0;
    uint64_t derived_iq2_xxs_registered = 0;
    uint64_t derived_column_grouped_u2_registered = 0;
    uint64_t retained_boundary = 0;
    CanonicalMetalResourceDiagnostics resource_diagnostics;
    std::vector<const uint8_t*> registered_artifact_bases;

    ~Impl() {
        if (metal_session) {
            for (const uint8_t* base : registered_artifact_bases) {
                metal_tok_session_unregister_weights(*metal_session, base);
            }
        }
    }
};

CanonicalMetalProgram::CanonicalMetalProgram(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CanonicalMetalProgram::CanonicalMetalProgram(CanonicalMetalProgram&&) noexcept = default;
CanonicalMetalProgram& CanonicalMetalProgram::operator=(CanonicalMetalProgram&&) noexcept = default;
CanonicalMetalProgram::~CanonicalMetalProgram() = default;

uint32_t CanonicalMetalProgram::layer_count() const noexcept {
    return impl_ ? static_cast<uint32_t>(impl_->layers.size()) : 0;
}

uint32_t CanonicalMetalProgram::position() const noexcept {
    return impl_ ? impl_->position : 0;
}

const ExecutionPlan& CanonicalMetalProgram::plan() const noexcept {
    static const ExecutionPlan empty;
    return impl_ ? impl_->plan : empty;
}

bool CanonicalMetalProgram::has_recurrent_layers() const noexcept {
    return impl_ && impl_->has_recurrent;
}

bool CanonicalMetalProgram::read_calibration(std::vector<CalibrationRecord>& records) const {
    records.clear();
    if (!impl_ || !impl_->metal_session || impl_->calibration.empty()) return false;
    records.reserve(impl_->calibration.size());
    for (const Impl::CalibrationBinding& binding : impl_->calibration) {
        CalibrationRecord record;
        record.target = binding.target;
        record.sum_squares.resize(binding.target.k_mapping.width);
        if (!metal_tok_session_read_importance(*impl_->metal_session, binding.slot,
                                               record.sum_squares.data(),
                                               binding.target.k_mapping.width,
                                               &record.sample_count)) {
            records.clear();
            return false;
        }
        records.push_back(std::move(record));
    }
    return true;
}

uint32_t CanonicalMetalProgram::derived_q2_storage_count() const noexcept {
    return impl_ ? static_cast<uint32_t>(impl_->derived_q2.size()) : 0;
}

uint64_t CanonicalMetalProgram::derived_q2_storage_bytes() const noexcept {
    uint64_t bytes = 0;
    if (impl_) for (const DerivedTensorOwner& owner : impl_->derived_q2) bytes += owner.logical_length;
    return bytes;
}

uint64_t CanonicalMetalProgram::original_source_registered_bytes() const noexcept {
    return impl_ ? impl_->original_source_registered : 0;
}

uint64_t CanonicalMetalProgram::derived_q2_registered_bytes() const noexcept {
    return impl_ ? impl_->derived_q2_registered : 0;
}

uint64_t CanonicalMetalProgram::retained_boundary_bytes() const noexcept {
    return impl_ ? impl_->retained_boundary : 0;
}

uint32_t CanonicalMetalProgram::derived_iq2_xxs_atlas_count() const noexcept {
    return impl_ && impl_->derived_iq2_xxs.mapping ? 1u : 0u;
}

uint32_t CanonicalMetalProgram::derived_iq2_xxs_tensor_count() const noexcept {
    return impl_ ? static_cast<uint32_t>(impl_->derived_iq2_xxs.entries.size()) : 0;
}

uint64_t CanonicalMetalProgram::derived_iq2_xxs_source_bytes() const noexcept {
    return impl_ ? impl_->derived_iq2_xxs.source_bytes : 0;
}

uint64_t CanonicalMetalProgram::derived_iq2_xxs_storage_bytes() const noexcept {
    if (!impl_) return 0;
    uint64_t bytes = 0;
    for (const DerivedIQ2XXSEntry& entry : impl_->derived_iq2_xxs.entries) bytes += entry.logical_length;
    return bytes;
}

uint64_t CanonicalMetalProgram::derived_iq2_xxs_registered_bytes() const noexcept {
    return impl_ ? impl_->derived_iq2_xxs_registered : 0;
}

uint32_t CanonicalMetalProgram::derived_column_grouped_u2_tensor_count() const noexcept {
    return impl_ && impl_->derived_column_grouped_u2
        ? static_cast<uint32_t>(impl_->derived_column_grouped_u2->entries().size()) : 0;
}

uint64_t CanonicalMetalProgram::derived_column_grouped_u2_source_bytes() const noexcept {
    return impl_ && impl_->derived_column_grouped_u2
        ? impl_->derived_column_grouped_u2->source_bytes() : 0;
}

uint64_t CanonicalMetalProgram::derived_column_grouped_u2_storage_bytes() const noexcept {
    return impl_ && impl_->derived_column_grouped_u2
        ? impl_->derived_column_grouped_u2->mapped_bytes() : 0;
}

CanonicalMetalResourceDiagnostics CanonicalMetalProgram::resource_diagnostics() const noexcept {
    if (!impl_) return {};
    CanonicalMetalResourceDiagnostics diagnostics = impl_->resource_diagnostics;
    if (impl_->metal_session) {
        const MetalResourceSnapshot current = metal_tok_session_resource_snapshot(*impl_->metal_session);
        diagnostics.implicit_weight_copies = current.implicit_weight_copies;
    }
    return diagnostics;
}

uint32_t CanonicalMetalProgram::sparse_ffn_layer_count() const noexcept {
    return impl_ ? impl_->sparse_ffn_layers : 0;
}

uint64_t CanonicalMetalProgram::sparse_ffn_source_bytes() const noexcept {
    return impl_ ? impl_->sparse_ffn_source : 0;
}

uint64_t CanonicalMetalProgram::sparse_ffn_requested_bytes() const noexcept {
    return impl_ ? impl_->sparse_ffn_requested : 0;
}

uint64_t CanonicalMetalProgram::sparse_ffn_worklist_bytes() const noexcept {
    if (impl_ && !impl_->sparse_ffn_proxies.empty()) return 2 * sizeof(uint32_t);
    if (impl_ && !impl_->sparse_ffn_block_ids.empty())
        return impl_->sparse_ffn_block_ids.size() * sizeof(uint32_t);
    uint64_t blocks = 0;
    if (impl_) for (const MetalSparseBlockRun& run : impl_->sparse_ffn_runs) blocks += run.count;
    return blocks * sizeof(uint32_t);
}

#if defined(LAPLACE_METAL_TESTING)
std::vector<uint32_t> CanonicalMetalProgram::sparse_ffn_block_counts_for_testing() const {
    std::vector<uint32_t> counts;
    if (!impl_) return counts;
    for (const auto& layer : impl_->layers) {
        if (const auto* dense = std::get_if<DenseLayer>(&layer)) {
            if (dense->metal.sparse_ffn)
                counts.push_back(dense->metal.sparse_ffn_block_count);
        } else if (const auto* recurrent = std::get_if<RecurrentLayer>(&layer)) {
            if (recurrent->metal.sparse_ffn)
                counts.push_back(recurrent->metal.sparse_ffn_block_count);
        }
    }
    return counts;
}
#endif

CanonicalMetalCursor CanonicalMetalProgram::checkpoint() const noexcept {
    if (!impl_) return {};
    return {impl_->session_id, impl_->generation, impl_->position};
}

bool CanonicalMetalProgram::commit(CanonicalMetalCursor cursor) noexcept {
    return impl_ && cursor.session_id == impl_->session_id && cursor.generation == impl_->generation &&
           cursor.position <= impl_->position;
}

bool CanonicalMetalProgram::rollback(CanonicalMetalCursor cursor) noexcept {
    if (!impl_ || impl_->has_recurrent || cursor.session_id != impl_->session_id || cursor.generation != impl_->generation ||
        cursor.position > impl_->position) return false;
    impl_->position = cursor.position;
    ++impl_->generation;
    return true;
}

bool CanonicalMetalProgram::rollback_to_position(uint32_t position) noexcept {
    if (!impl_ || impl_->has_recurrent || position > impl_->position) return false;
    impl_->position = position;
    ++impl_->generation;
    return true;
}

CanonicalMetalCreateResult create_canonical_metal_program_internal(
    std::shared_ptr<const RuntimePackage> package, SessionRequest request,
    const CanonicalDerivedQ2KPolicy& derived_q2,
    const CanonicalSparseFfnPolicy& sparse_ffn,
    const CanonicalDerivedIQ2XXSPolicy& derived_iq2_xxs,
    const CanonicalCalibrationPolicy& calibration,
    const CanonicalDerivedColumnGroupedU2Policy& derived_column_u2) {
    if (!package || request.max_context == 0 || request.max_context > UINT32_MAX ||
        request.max_context > package->semantics().maximum_context) {
        return metal_error(CompatibilityError::PLAN_CONTEXT_EXCEEDED);
    }
    if (request.enable_streaming) return metal_error(CompatibilityError::STREAMING_UNSUPPORTED);
    if (request.enable_speculation) return metal_error(CompatibilityError::FALLBACK_FORBIDDEN);
    if ((!request.enable_prefill && !request.enable_decode) || request.max_batch == 0 ||
        request.minimum_class != NumericalClass::ExactFp32 ||
        (request.objective != RuntimeObjective::Latency && request.objective != RuntimeObjective::Throughput)) {
        return metal_error(CompatibilityError::RUNTIME_INPUT_INVALID);
    }
    if (request.memory_limit == 0) return metal_error(CompatibilityError::PLAN_MEMORY_EXCEEDED);
    const uint32_t maximum_context = static_cast<uint32_t>(request.max_context);
    const SemanticModel& model = package->semantics();
    if (model.layers.empty() || model.vocabulary_size == 0 || model.layers.size() > INT_MAX) {
        return metal_error(CompatibilityError::IR_REFERENCE_INVALID);
    }
    CanonicalProgramWitness lowering_program;
    std::string lowering_program_detail;
    if (!canonical_program_witness(model, lowering_program,
                                   lowering_program_detail)) {
        return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                           lowering_program_detail);
    }
    if (!calibration.targets.empty() &&
        (!sparse_ffn.runs.empty() || !sparse_ffn.layer_masks.empty() ||
         sparse_ffn.proxy_selected_blocks != 0 ||
         sparse_ffn.dense_oracle_selected_blocks != 0)) {
        return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                           "canonical calibration requires the full FFN activation geometry");
    }
#if !defined(LAPLACE_METAL_TESTING)
    if (derived_q2.register_retained_source_ranges_for_testing) {
        return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                           "canonical Metal retained source registration is test-only");
    }
#endif
    const uint32_t selector_count = static_cast<uint32_t>(!sparse_ffn.runs.empty()) +
                                    static_cast<uint32_t>(!sparse_ffn.layer_masks.empty()) +
                                    static_cast<uint32_t>(sparse_ffn.proxy_selected_blocks != 0) +
                                    static_cast<uint32_t>(sparse_ffn.dense_oracle_selected_blocks != 0);
    if (selector_count > 1) {
        return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                           "canonical Metal sparse FFN policy has multiple selectors");
    }
#if !defined(LAPLACE_METAL_TESTING)
    if (sparse_ffn.dense_oracle_selected_blocks != 0) {
        return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                           "canonical Metal dense-oracle FFN selector is test-only");
    }
#endif
    CanonicalSparseFfnPolicy admitted_sparse_ffn = sparse_ffn;
    const uint32_t dynamic_selected_blocks = sparse_ffn.proxy_selected_blocks != 0
        ? sparse_ffn.proxy_selected_blocks : sparse_ffn.dense_oracle_selected_blocks;
    if (dynamic_selected_blocks != 0) {
        admitted_sparse_ffn.runs = {{0, dynamic_selected_blocks}};
        admitted_sparse_ffn.proxy_selected_blocks = 0;
        admitted_sparse_ffn.dense_oracle_selected_blocks = 0;
    }
    auto impl = std::make_unique<CanonicalMetalProgram::Impl>();
    impl->package = std::move(package);
    impl->session_id = g_next_metal_session_id.fetch_add(1, std::memory_order_relaxed);
    impl->maximum_context = maximum_context;
    impl->maximum_batch_rows = request.max_batch;
    impl->enable_prefill = request.enable_prefill;
    impl->enable_decode = request.enable_decode;
    impl->vocabulary = model.vocabulary_size;
    impl->calibration.reserve(calibration.targets.size());
    for (const CalibrationTarget& target : calibration.targets) {
        CalibrationTargetValidation validated = validate_calibration_target(model, target, true);
        if (const auto* report = std::get_if<CompatibilityReport>(&validated)) return *report;
        if (std::find_if(impl->calibration.begin(), impl->calibration.end(), [&](const auto& binding) {
                return binding.target == target;
            }) != impl->calibration.end()) {
            return calibration_error(CompatibilityError::IR_REFERENCE_INVALID, target,
                                     "calibration target is duplicated");
        }
        uint32_t slot = UINT32_MAX;
        for (uint32_t candidate = 0; candidate != impl->calibration.size(); ++candidate) {
            const CalibrationTarget& existing = impl->calibration[candidate].target;
            if (existing.input_value_id == target.input_value_id &&
                existing.k_mapping == target.k_mapping) {
                slot = impl->calibration[candidate].slot;
                break;
            }
        }
        if (slot == UINT32_MAX) {
            slot = static_cast<uint32_t>(impl->calibration_widths.size());
            impl->calibration_widths.push_back(target.k_mapping.width);
        }
        impl->calibration.push_back({target, slot, false});
    }
    impl->sparse_ffn_runs.reserve(sparse_ffn.runs.size());
    for (const SparseBlockRun& run : sparse_ffn.runs)
        impl->sparse_ffn_runs.push_back({run.first, run.count});
    std::vector<ArtifactSpan> artifacts;
    CompatibilityReport artifact_error;
    if (!collect_referenced_artifacts(*impl->package, model, artifacts, artifact_error)) return artifact_error;
    std::vector<uint32_t> used_tensor_ids;
    std::string derived_error;
    if (!build_derived_column_grouped_u2_atlas(
            *impl->package, model, derived_column_u2,
            impl->derived_column_grouped_u2, derived_error)) {
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE, derived_error);
    }
    if (!build_derived_iq2_xxs_atlas(*impl->package, model, derived_iq2_xxs,
                                     impl->derived_iq2_xxs, derived_error)) {
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE, derived_error);
    }
    const auto assign = [&](const SemanticOperator& op, TensorRole role, Tensor& tensor, bool required,
                            bool expert = false) {
        const SemanticTensor* semantic = tensor_with_role(model, op, role, expert);
        const bool assigned = assign_tensor(*impl->package, model, op, role, tensor, required,
                                            derived_q2, impl->derived_iq2_xxs,
                                            impl->derived_column_grouped_u2,
                                            impl->derived_q2, derived_error, expert);
        if (!assigned && required && derived_error.empty()) {
            derived_error = "canonical Metal could not materialize tensor role " +
                            std::to_string(static_cast<unsigned>(role)) +
                            (expert ? " from an expert bank" : "");
        }
        if (assigned && semantic && std::find(used_tensor_ids.begin(), used_tensor_ids.end(), semantic->id) == used_tensor_ids.end())
            used_tensor_ids.push_back(semantic->id);
        return assigned;
    };
    const auto assign_single = [&](const SemanticOperator& op, Tensor& tensor) {
        if (op.tensors.size() != 1 || op.tensors[0] >= model.tensors.size()) {
            derived_error = "canonical Metal expected exactly one materialized tensor";
            return false;
        }
        const SemanticTensor& semantic = model.tensors[op.tensors[0]];
        const bool assigned = materialize_derived_q2(*impl->package, semantic, derived_q2,
                                                      impl->derived_iq2_xxs,
                                                      impl->derived_column_grouped_u2,
                                                      impl->derived_q2,
                                                      tensor, derived_error);
        if (!assigned && derived_error.empty())
            derived_error = "canonical Metal could not materialize a single-tensor operator";
        if (assigned && std::find(used_tensor_ids.begin(), used_tensor_ids.end(), semantic.id) == used_tensor_ids.end())
            used_tensor_ids.push_back(semantic.id);
        return assigned;
    };
    const auto assignment_error = [&]() {
        return metal_error(derived_error.empty() ? CompatibilityError::IR_REFERENCE_INVALID
                                                 : CompatibilityError::KERNEL_UNAVAILABLE,
                           derived_error);
    };
    const auto calibration_slot = [&](const SemanticOperator& op, TensorRole role,
                                      uint32_t expected_input_value_id) {
        const SemanticTensor* tensor = tensor_with_role(model, op, role);
        if (!tensor) return UINT32_MAX;
        uint32_t slot = UINT32_MAX;
        for (auto& binding : impl->calibration) {
            if (binding.target.operator_id != op.id ||
                binding.target.weight_tensor_id != tensor->id ||
                binding.target.input_value_id != expected_input_value_id) continue;
            if (slot != UINT32_MAX && slot != binding.slot) return UINT32_MAX;
            slot = binding.slot;
            binding.attached = true;
        }
        return slot;
    };
    size_t admitted_layer_masks = 0;
    const auto sparse_policy_for_layer = [&](uint32_t layer_index,
                                             CanonicalSparseFfnPolicy& policy,
                                             const CanonicalSparseFfnPolicy::LayerMask*& mask) {
        policy = admitted_sparse_ffn;
        policy.layer_masks.clear();
        mask = nullptr;
        if (sparse_ffn.layer_masks.empty()) return true;
        for (const auto& candidate : sparse_ffn.layer_masks) {
            if (candidate.layer_index != layer_index) continue;
            if (mask) return false;
            mask = &candidate;
        }
        if (!mask || mask->block_ids.empty()) return false;
        policy.runs.clear();
        uint32_t first = mask->block_ids.front();
        uint32_t previous = first;
        for (size_t index = 1; index != mask->block_ids.size(); ++index) {
            const uint32_t current = mask->block_ids[index];
            if (current <= previous) return false;
            if (current != previous + 1) {
                policy.runs.push_back({first, previous - first + 1});
                first = current;
            }
            previous = current;
        }
        policy.runs.push_back({first, previous - first + 1});
        return true;
    };

    const SemanticOperator& embedding_op = model.operators[lowering_program.embedding_operator_id];
    const auto* embedding_payload = std::get_if<EmbeddingLookupPayload>(&embedding_op.payload);
    if (embedding_op.kind != OperatorKind::EmbeddingLookup || !embedding_payload ||
        embedding_payload->width == 0 || embedding_payload->vocabulary != model.vocabulary_size ||
        !f32_bits(embedding_payload->scale_f32_bits, impl->embedding_scale) ||
        impl->embedding_scale <= 0.0f ||
        !assign(embedding_op, TensorRole::TokenEmbedding, impl->embedding, true)) {
        return assignment_error();
    }
    impl->hidden = embedding_payload->width;

    impl->layers.reserve(model.layers.size());
    uint32_t recurrent_state_slot = 0;
    std::vector<uint32_t> claimed_attention_states;
    for (const SemanticLayer& layer : model.layers) {
        if ((layer.flags & kSemanticLayerFlagSpeculative) != 0) continue;
        const SemanticOperator* delta = operator_with_role(model, layer, OperatorKind::GatedDeltaNet,
                                                            TensorRole::RecurrentDtBias);
        if (delta) {
            const SemanticOperator* input_norm = operator_with_role(model, layer, OperatorKind::RmsNorm,
                                                                      TensorRole::AttentionNormWeight);
            const SemanticOperator* qkv = operator_with_role(model, layer, OperatorKind::Linear,
                                                              TensorRole::RecurrentQkvWeight);
            const SemanticOperator* gate = operator_with_role(model, layer, OperatorKind::Linear,
                                                               TensorRole::RecurrentGateWeight);
            const SemanticOperator* beta = operator_with_role(model, layer, OperatorKind::Linear,
                                                               TensorRole::RecurrentBetaWeight);
            const SemanticOperator* alpha = operator_with_role(model, layer, OperatorKind::Linear,
                                                                TensorRole::RecurrentAlphaWeight);
            const SemanticOperator* conv = operator_with_role(model, layer, OperatorKind::DepthwiseConvSilu,
                                                               TensorRole::RecurrentConvWeight);
            const SemanticOperator* recurrent_norm = operator_with_role(model, layer, OperatorKind::GatedRmsNorm,
                                                                         TensorRole::RecurrentNormWeight);
            const SemanticOperator* recurrent_output = operator_with_role(model, layer, OperatorKind::Linear,
                                                                           TensorRole::RecurrentOutputWeight);
            const SemanticOperator* ffn_norm = operator_with_role(model, layer, OperatorKind::RmsNorm,
                                                                    TensorRole::FfnNormWeight);
            const SemanticOperator* ffn_gate = operator_with_role(model, layer, OperatorKind::Linear,
                                                                    TensorRole::FfnGateWeight);
            const SemanticOperator* ffn_up = operator_with_role(model, layer, OperatorKind::Linear,
                                                                  TensorRole::FfnUpWeight);
            const SemanticOperator* ffn_down = operator_with_role(model, layer, OperatorKind::Linear,
                                                                    TensorRole::FfnDownWeight);
            const SemanticOperator* l2_query = nullptr;
            const SemanticOperator* ffn_swiglu = nullptr;
            if (layer.first_operator > model.operators.size() ||
                layer.operator_count > model.operators.size() - layer.first_operator) {
                return metal_error(CompatibilityError::IR_REFERENCE_INVALID);
            }
            for (uint32_t index = 0; index != layer.operator_count; ++index) {
                const SemanticOperator& op = model.operators[layer.first_operator + index];
                if (op.kind == OperatorKind::L2Normalize && !l2_query) l2_query = &op;
                if (op.kind == OperatorKind::SwiGlu)
                    ffn_swiglu = ffn_swiglu ? nullptr : &op;
            }
            const auto* delta_payload = std::get_if<GatedDeltaNetPayload>(&delta->payload);
            const auto* conv_payload = conv ? std::get_if<DepthwiseConvSiluPayload>(&conv->payload) : nullptr;
            const auto* input_norm_payload = input_norm ? std::get_if<RmsNormPayload>(&input_norm->payload) : nullptr;
            const auto* l2_payload = l2_query ? std::get_if<L2NormalizePayload>(&l2_query->payload) : nullptr;
            uint32_t ffn_intermediate = 0;
            float rms_epsilon = 0.0f;
            float l2_epsilon = 0.0f;
            if (!input_norm || !qkv || !gate || !beta || !alpha || !conv || !recurrent_norm || !recurrent_output ||
                !ffn_norm || !ffn_gate || !ffn_up || !ffn_down || !ffn_swiglu ||
                ffn_norm->outputs.size() != 1 || ffn_swiglu->outputs.size() != 1 ||
                !delta_payload || !conv_payload ||
                !input_norm_payload || !l2_payload ||
                !f32_bits(input_norm_payload->epsilon_f32_bits, rms_epsilon) || rms_epsilon < 0.0f ||
                !f32_bits(l2_payload->epsilon_f32_bits, l2_epsilon) || l2_epsilon < 0.0f ||
                !output_width(model, *ffn_gate, ffn_intermediate) ||
                delta_payload->qk_heads == 0 || delta_payload->value_heads == 0 || delta_payload->head_dimension == 0 ||
                delta_payload->qk_heads > INT_MAX || delta_payload->value_heads > INT_MAX ||
                delta_payload->head_dimension > INT_MAX || conv_payload->kernel == 0 ||
                conv_payload->kernel > INT_MAX || ffn_intermediate > INT_MAX) {
                return metal_error(CompatibilityError::IR_REFERENCE_INVALID);
            }
            impl->layers.emplace_back(RecurrentLayer{});
            RecurrentLayer& recurrent = std::get<RecurrentLayer>(impl->layers.back());
            if (!assign(*input_norm, TensorRole::AttentionNormWeight, recurrent.input_norm, true) ||
                !assign(*qkv, TensorRole::RecurrentQkvWeight, recurrent.qkv, true) ||
                !assign(*gate, TensorRole::RecurrentGateWeight, recurrent.gate, true) ||
                !assign(*beta, TensorRole::RecurrentBetaWeight, recurrent.beta, true) ||
                !assign(*alpha, TensorRole::RecurrentAlphaWeight, recurrent.alpha, true) ||
                !assign(*conv, TensorRole::RecurrentConvWeight, recurrent.conv, true) ||
                !assign(*delta, TensorRole::RecurrentDtBias, recurrent.dt_bias, true) ||
                !assign(*delta, TensorRole::RecurrentDecayWeight, recurrent.decay, true) ||
                !assign(*recurrent_norm, TensorRole::RecurrentNormWeight, recurrent.norm, true) ||
                !assign(*recurrent_output, TensorRole::RecurrentOutputWeight, recurrent.output, true) ||
                !assign(*ffn_norm, TensorRole::FfnNormWeight, recurrent.ffn_norm, true) ||
                !assign(*ffn_gate, TensorRole::FfnGateWeight, recurrent.ffn_gate, true) ||
                !assign(*ffn_up, TensorRole::FfnUpWeight, recurrent.ffn_up, true) ||
                !assign(*ffn_down, TensorRole::FfnDownWeight, recurrent.ffn_down, true)) {
                return assignment_error();
            }
            CanonicalSparseFfnPolicy layer_sparse_ffn;
            const CanonicalSparseFfnPolicy::LayerMask* layer_mask = nullptr;
            if (!sparse_policy_for_layer(layer.layer_index, layer_sparse_ffn, layer_mask)) {
                return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "canonical Metal sparse FFN layer mask is invalid");
            }
            SparseFfnPlan sparse_plan;
            if (!validate_sparse_ffn(layer.layer_index, recurrent.ffn_gate, recurrent.ffn_up,
                                     recurrent.ffn_down, layer_sparse_ffn, sparse_plan, derived_error)) {
                return assignment_error();
            }
            if (!layer_sparse_ffn.runs.empty()) {
                if (impl->sparse_ffn_source > UINT64_MAX - sparse_plan.source_bytes ||
                    impl->sparse_ffn_requested > UINT64_MAX - sparse_plan.requested_bytes) {
                    return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                       "sparse FFN program byte count overflows");
                }
                impl->sparse_ffn_source += sparse_plan.source_bytes;
                impl->sparse_ffn_requested += sparse_plan.requested_bytes;
                ++impl->sparse_ffn_layers;
            }
            if (layer_mask) {
                if (layer_mask->block_ids.size() > UINT32_MAX ||
                    impl->sparse_ffn_block_ids.size() >
                        UINT32_MAX - layer_mask->block_ids.size()) {
                    return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                       "canonical Metal sparse FFN layer mask table overflows");
                }
                recurrent.metal.sparse_ffn_block_offset =
                    static_cast<uint32_t>(impl->sparse_ffn_block_ids.size());
                recurrent.metal.sparse_ffn_block_count =
                    static_cast<uint32_t>(layer_mask->block_ids.size());
                impl->sparse_ffn_block_offsets.push_back(
                    recurrent.metal.sparse_ffn_block_offset);
                impl->sparse_ffn_block_counts.push_back(
                    recurrent.metal.sparse_ffn_block_count);
                impl->sparse_ffn_block_ids.insert(impl->sparse_ffn_block_ids.end(),
                                                  layer_mask->block_ids.begin(),
                                                  layer_mask->block_ids.end());
                ++admitted_layer_masks;
            }
            recurrent.metal.input_norm = &recurrent.input_norm;
            recurrent.metal.qkv = &recurrent.qkv;
            recurrent.metal.gate = &recurrent.gate;
            recurrent.metal.beta = &recurrent.beta;
            recurrent.metal.alpha = &recurrent.alpha;
            recurrent.metal.conv = &recurrent.conv;
            recurrent.metal.dt_bias = &recurrent.dt_bias;
            recurrent.metal.decay = &recurrent.decay;
            recurrent.metal.norm = &recurrent.norm;
            recurrent.metal.output = &recurrent.output;
            recurrent.metal.ffn_norm = &recurrent.ffn_norm;
            recurrent.metal.ffn_gate = &recurrent.ffn_gate;
            recurrent.metal.ffn_up = &recurrent.ffn_up;
            recurrent.metal.ffn_down = &recurrent.ffn_down;
            const uint32_t gate_importance = calibration_slot(
                *ffn_gate, TensorRole::FfnGateWeight, ffn_norm->outputs.front());
            const uint32_t up_importance = calibration_slot(
                *ffn_up, TensorRole::FfnUpWeight, ffn_norm->outputs.front());
            if (gate_importance != UINT32_MAX && up_importance != UINT32_MAX &&
                gate_importance != up_importance) {
                return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "canonical recurrent FFN gate/up calibration inputs differ");
            }
            recurrent.metal.ffn_input_importance_slot = gate_importance != UINT32_MAX
                ? gate_importance : up_importance;
            recurrent.metal.ffn_down_importance_slot =
                calibration_slot(*ffn_down, TensorRole::FfnDownWeight,
                                 ffn_swiglu->outputs.front());
            if (dynamic_selected_blocks != 0 &&
                !build_sparse_ffn_proxy(layer.layer_index, recurrent.ffn_gate, recurrent.ffn_up,
                                        recurrent.ffn_down, dynamic_selected_blocks,
                                        impl->sparse_ffn_proxies,
                                        recurrent.metal.sparse_ffn_proxy_slot, derived_error)) {
                return assignment_error();
            }
            recurrent.metal.state_slot = recurrent_state_slot++;
            recurrent.metal.H = static_cast<int>(impl->hidden);
            recurrent.metal.qk_heads = static_cast<int>(delta_payload->qk_heads);
            recurrent.metal.value_heads = static_cast<int>(delta_payload->value_heads);
            recurrent.metal.head_dimension = static_cast<int>(delta_payload->head_dimension);
            recurrent.metal.kernel = static_cast<int>(conv_payload->kernel);
            recurrent.metal.ffn_intermediate = static_cast<int>(
                layer_sparse_ffn.runs.empty() ? ffn_intermediate : sparse_plan.packed_intermediate);
            recurrent.metal.sparse_ffn_full_intermediate = static_cast<int>(ffn_intermediate);
            recurrent.metal.sparse_ffn = !layer_sparse_ffn.runs.empty();
            recurrent.metal.sparse_ffn_dense_oracle = sparse_ffn.dense_oracle_selected_blocks != 0;
            recurrent.metal.l2_epsilon = l2_epsilon;
            recurrent.metal.rms_epsilon = rms_epsilon;
            recurrent.operator_id = delta->id;
            const uint64_t channels = static_cast<uint64_t>(delta_payload->head_dimension) *
                                      (2ull * delta_payload->qk_heads + delta_payload->value_heads);
            if (channels > INT_MAX) return metal_error(CompatibilityError::IR_REFERENCE_INVALID);
            impl->token_intermediate = std::max(
                impl->token_intermediate,
                std::max(recurrent.metal.sparse_ffn_dense_oracle
                             ? ffn_intermediate
                             : static_cast<uint32_t>(recurrent.metal.ffn_intermediate),
                         static_cast<uint32_t>(channels)));
            impl->has_recurrent = true;
            continue;
        }
        const SemanticOperator* attn_norm = operator_with_role(model, layer, OperatorKind::RmsNorm, TensorRole::AttentionNormWeight);
        const SemanticOperator* query = operator_with_role(model, layer, OperatorKind::Linear, TensorRole::QueryWeight);
        const SemanticOperator* query_gate = operator_with_role(model, layer, OperatorKind::Linear,
                                                                 TensorRole::AttentionQueryGateWeight);
        const SemanticOperator* key = operator_with_role(model, layer, OperatorKind::Linear, TensorRole::KeyWeight);
        const SemanticOperator* query_norm = operator_with_role(model, layer, OperatorKind::RmsNorm,
                                                                 TensorRole::AttentionQueryNormWeight);
        const SemanticOperator* key_norm = operator_with_role(model, layer, OperatorKind::RmsNorm,
                                                               TensorRole::AttentionKeyNormWeight);
        const SemanticOperator* value = operator_with_role(model, layer, OperatorKind::Linear, TensorRole::ValueWeight);
        const SemanticOperator* attention_output = operator_with_role(model, layer, OperatorKind::Linear, TensorRole::AttentionOutputWeight);
        const SemanticOperator* ffn_norm = operator_with_role(model, layer, OperatorKind::RmsNorm, TensorRole::FfnNormWeight);
        const SemanticOperator* ffn_gate = operator_with_role(model, layer, OperatorKind::Linear, TensorRole::FfnGateWeight);
        const SemanticOperator* ffn_up = operator_with_role(model, layer, OperatorKind::Linear, TensorRole::FfnUpWeight);
        const SemanticOperator* ffn_down = operator_with_role(model, layer, OperatorKind::Linear, TensorRole::FfnDownWeight);
        CanonicalMoeOperatorEdges moe_edges;
        bool has_moe_operator = false;
        if (layer.first_operator <= model.operators.size() &&
            layer.operator_count <= model.operators.size() - layer.first_operator) {
            for (uint32_t index = 0; index != layer.operator_count; ++index) {
                const OperatorKind kind = model.operators[layer.first_operator + index].kind;
                if (kind == OperatorKind::RouterTopK || kind == OperatorKind::RoutedLinear ||
                    kind == OperatorKind::GatedActivation || kind == OperatorKind::WeightedExpertReduce) {
                    has_moe_operator = true;
                    break;
                }
            }
        }
        const bool is_moe = has_moe_operator &&
                            match_canonical_moe_operator_edges(model, layer, moe_edges);
        const SemanticOperator* moe_router = nullptr;
        const SemanticOperator* moe_norm = nullptr;
        const SemanticOperator* moe_up = nullptr;
        const SemanticOperator* moe_down = nullptr;
        const RouterTopKPayload* moe_payload = nullptr;
        uint32_t moe_intermediate = 0;
        if (has_moe_operator && !is_moe) {
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "canonical Metal MoE layer has an unsupported or ambiguous semantic edge set");
        }
        if (is_moe) {
            // The matcher returned one complete typed witness.  Use its
            // dense nodes as well as its routed nodes so lowering cannot
            // select a different operator after admission.
            attn_norm = moe_edges.dense_attn_norm;
            query = moe_edges.dense_query;
            query_gate = moe_edges.dense_query_gate;
            key = moe_edges.dense_key;
            value = moe_edges.dense_value;
            attention_output = moe_edges.dense_output;
            query_norm = moe_edges.dense_query_norm;
            key_norm = moe_edges.dense_key_norm;
            ffn_norm = moe_edges.dense_ffn_norm;
            ffn_gate = moe_edges.dense_gate;
            ffn_up = moe_edges.dense_up;
            ffn_down = moe_edges.dense_down;
            moe_router = moe_edges.router_linear;
            moe_norm = moe_edges.expert_norm;
            moe_up = moe_edges.expert_up;
            moe_down = moe_edges.expert_down;
            moe_payload = std::get_if<RouterTopKPayload>(&moe_edges.router->payload);
            if (!moe_payload || moe_payload->expert_count > INT_MAX ||
                moe_payload->selected_count > INT_MAX || moe_edges.intermediate > INT_MAX) {
                return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                                   "canonical Metal MoE layer has an unsupported router or expert geometry");
            }
            moe_intermediate = moe_edges.intermediate;
        }
        const auto dense_role_error = [&](const char* role) {
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "canonical Metal dense layer " + std::to_string(layer.layer_index) +
                               " lacks " + role);
        };
        if (!attn_norm) return dense_role_error("AttentionNormWeight");
        if ((query == nullptr) == (query_gate == nullptr)) return dense_role_error("one query projection role");
        if (!key) return dense_role_error("KeyWeight");
        if (!attention_output) return dense_role_error("AttentionOutputWeight");
        if (!ffn_norm) return dense_role_error("FfnNormWeight");
        if (!ffn_gate) return dense_role_error("FfnGateWeight");
        if (!ffn_up) return dense_role_error("FfnUpWeight");
        if (!ffn_down) return dense_role_error("FfnDownWeight");
        if (is_moe) {
            const auto witness_contains = [&](const SemanticOperator* op) {
                return op && std::find(moe_edges.covered_operator_ids.begin(),
                                       moe_edges.covered_operator_ids.end(), op->id) !=
                                   moe_edges.covered_operator_ids.end();
            };
            if (!witness_contains(attn_norm) || !witness_contains(query_gate ? query_gate : query) ||
                !witness_contains(key) || !witness_contains(value) ||
                !witness_contains(attention_output) || !witness_contains(ffn_norm) ||
                !witness_contains(ffn_gate) || !witness_contains(ffn_up) ||
                !witness_contains(ffn_down)) {
                return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                                   "canonical Metal lowerer received a dense edge outside the MoE witness");
            }
        }
        const bool fused_query_gate = query_gate != nullptr;
        const SemanticOperator* query_projection = fused_query_gate ? query_gate : query;
        const SemanticOperator* rope = nullptr;
        const SemanticOperator* attention = nullptr;
        const SemanticOperator* swiglu = nullptr;
        if (layer.first_operator > model.operators.size() || layer.operator_count > model.operators.size() - layer.first_operator) {
            return metal_error(CompatibilityError::IR_REFERENCE_INVALID);
        }
        if (is_moe) {
            rope = moe_edges.rope;
            attention = moe_edges.attention;
            swiglu = moe_edges.swiglu;
        } else {
            for (uint32_t index = 0; index != layer.operator_count; ++index) {
                const SemanticOperator& op = model.operators[layer.first_operator + index];
                if (op.kind == OperatorKind::Rope) rope = rope ? nullptr : &op;
                if (op.kind == OperatorKind::CausalAttention) attention = attention ? nullptr : &op;
                if (op.kind == OperatorKind::SwiGlu) swiglu = swiglu ? nullptr : &op;
            }
        }
        const auto* rope_payload = rope ? std::get_if<RopePayload>(&rope->payload) : nullptr;
        const auto* attention_payload = attention ? std::get_if<CausalAttentionPayload>(&attention->payload) : nullptr;
        const auto* swiglu_payload = swiglu ? std::get_if<SwiGluPayload>(&swiglu->payload) : nullptr;
        const auto* attn_rms = std::get_if<RmsNormPayload>(&attn_norm->payload);
        const auto* ffn_rms = std::get_if<RmsNormPayload>(&ffn_norm->payload);
        const auto* query_norm_payload = query_norm ? std::get_if<RmsNormPayload>(&query_norm->payload) : nullptr;
        const auto* key_norm_payload = key_norm ? std::get_if<RmsNormPayload>(&key_norm->payload) : nullptr;
        uint32_t inter = 0;
        float epsilon = 0.0f;
        float query_norm_epsilon = 0.0f;
        float key_norm_epsilon = 0.0f;
        float rope_base = 0.0f;
        float rope_scale = 0.0f;
        float attention_scale = 0.0f;
        const uint32_t rope_frequency_dimension = rope_payload
            ? (rope_payload->frequency_dimension == 0
                ? rope_payload->rotary_dimension : rope_payload->frequency_dimension)
            : 0;
        const bool multi_section_rope = rope_payload && rope_payload->pairing == RopePairing::MultiSectionHalfSplit;
        if (!swiglu_payload || swiglu_payload->activation != ActivationKind::Silu) {
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "canonical dense pattern requires SiLU SwiGLU");
        }
        if (!attention || !attention_payload || attention_payload->mask != AttentionMask::Causal ||
            attention_payload->cache_policy != CachePolicy::Global) {
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "canonical dense pattern requires global causal attention");
        }
        if (attention_payload->value_source == ValueSource::SeparateProjection && !value) {
            return dense_role_error("ValueWeight");
        }
        if (attention_payload->value_source != ValueSource::SeparateProjection) {
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "canonical attention value source requires a separate projection");
        }
        uint32_t key_state_id = UINT32_MAX;
        uint32_t value_state_id = UINT32_MAX;
        const bool owns_kv = owns_fp32_global_kv(
            model, *attention, attention_payload->kv_heads,
            attention_payload->head_dimension, &key_state_id, &value_state_id);
        if (!owns_kv) {
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "canonical dense pattern requires one FP32 key/value state pair");
        }
        if (!rope_payload || !attention_payload || !attn_rms || !ffn_rms ||
            (rope_payload->pairing != RopePairing::HalfSplit && rope_payload->pairing != RopePairing::Interleaved &&
             rope_payload->pairing != RopePairing::MultiSectionHalfSplit) ||
            ((query_norm == nullptr) != (key_norm == nullptr)) ||
            (query_norm && (!query_norm_payload || !key_norm_payload ||
                            !f32_bits(query_norm_payload->epsilon_f32_bits, query_norm_epsilon) ||
                            query_norm_epsilon < 0.0f ||
                            !f32_bits(key_norm_payload->epsilon_f32_bits, key_norm_epsilon) ||
                            key_norm_epsilon < 0.0f)) ||
            attn_rms->epsilon_f32_bits != ffn_rms->epsilon_f32_bits ||
            !output_width(model, *ffn_gate, inter) ||
            !f32_bits(attn_rms->epsilon_f32_bits, epsilon) || epsilon < 0.0f ||
            !f32_bits(rope_payload->base_f32_bits, rope_base) || rope_base <= 0.0f ||
            !f32_bits(rope_payload->scale_f32_bits, rope_scale) || rope_scale <= 0.0f ||
            !f32_bits(attention_payload->scale_f32_bits, attention_scale) || attention_scale <= 0.0f ||
            rope_scale != 1.0f || attention_payload->head_dimension == 0 ||
            rope_payload->rotary_dimension == 0 ||
            rope_payload->rotary_dimension > attention_payload->head_dimension ||
            (rope_payload->rotary_dimension % 2) != 0 ||
            rope_frequency_dimension < rope_payload->rotary_dimension ||
            (rope_frequency_dimension % 2) != 0 || rope_frequency_dimension > INT_MAX ||
            attention_payload->query_heads > INT_MAX || attention_payload->kv_heads > INT_MAX ||
            attention_payload->head_dimension > INT_MAX || inter > INT_MAX || impl->hidden > INT_MAX) {
            return metal_error(CompatibilityError::IR_REFERENCE_INVALID);
        }
        if (multi_section_rope) {
            uint64_t total_sections = 0;
            for (uint32_t section : rope_payload->position_sections) total_sections += section;
            if (rope_payload->position_sections[0] == 0 || rope_payload->position_sections[1] == 0 ||
                total_sections != rope_payload->rotary_dimension / 2) {
                return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "canonical Metal MRoPE sections do not cover the rotary pairs");
            }
        }

        impl->layers.emplace_back(DenseLayer{});
        DenseLayer& dense = std::get<DenseLayer>(impl->layers.back());
        if (!assign(*attn_norm, TensorRole::AttentionNormWeight, dense.attn_norm, true) ||
            !assign(*query_projection,
                    fused_query_gate ? TensorRole::AttentionQueryGateWeight : TensorRole::QueryWeight,
                    dense.query, true) ||
            (!fused_query_gate && !assign(*query, TensorRole::QueryBias, dense.query_bias, false)) ||
            !assign(*key, TensorRole::KeyWeight, dense.key, true) ||
            !assign(*key, TensorRole::KeyBias, dense.key_bias, false) ||
            (query_norm && (!assign(*query_norm,
                                    TensorRole::AttentionQueryNormWeight, dense.query_norm, true) ||
                            !assign(*key_norm,
                                    TensorRole::AttentionKeyNormWeight, dense.key_norm, true))) ||
            (value && !assign(*value, TensorRole::ValueWeight, dense.value, true)) ||
            (value && !assign(*value, TensorRole::ValueBias, dense.value_bias, false)) ||
            !assign(*attention_output, TensorRole::AttentionOutputWeight, dense.attention_output, true) ||
            !assign(*ffn_norm, TensorRole::FfnNormWeight, dense.ffn_norm, true) ||
            !assign(*ffn_gate, TensorRole::FfnGateWeight, dense.ffn_gate, true) ||
            !assign(*ffn_up, TensorRole::FfnUpWeight, dense.ffn_up, true) ||
            !assign(*ffn_down, TensorRole::FfnDownWeight, dense.ffn_down, true)) {
            return assignment_error();
        }
        if (is_moe &&
            (!assign_single(*moe_router, dense.moe_gate) ||
             !assign(*moe_edges.router_scale, TensorRole::NextnEmbeddingNormWeight,
                     dense.moe_gate_scale, true, false) ||
             !assign(*moe_norm, TensorRole::NextnEmbeddingNormWeight, dense.moe_pre_norm, true, false) ||
             !assign(*moe_up, TensorRole::FfnUpWeight, dense.moe_up, true, true) ||
             !assign(*moe_down, TensorRole::FfnDownWeight, dense.moe_down, true, true) ||
             !assign(*moe_edges.expert_reduce, TensorRole::NextnEmbeddingNormWeight,
                     dense.moe_down_scale, true, false) ||
             (moe_edges.dense_post_norm &&
              !assign(*moe_edges.dense_post_norm, TensorRole::NextnEmbeddingNormWeight,
                      dense.post_ffw_1, true, false)) ||
             (moe_edges.moe_post_norm &&
              !assign(*moe_edges.moe_post_norm, TensorRole::NextnEmbeddingNormWeight,
                      dense.post_ffw_2, true, false)) ||
             (moe_edges.output_post_norm &&
              !assign(*moe_edges.output_post_norm, TensorRole::NextnEmbeddingNormWeight,
                      dense.post_ffw, true, false)) ||
             (moe_edges.output_scale &&
              !assign(*moe_edges.output_scale, TensorRole::NextnEmbeddingNormWeight,
                      dense.out_scale, true, false)))) {
            return assignment_error();
        }
        CanonicalSparseFfnPolicy layer_sparse_ffn;
        const CanonicalSparseFfnPolicy::LayerMask* layer_mask = nullptr;
        if (!sparse_policy_for_layer(layer.layer_index, layer_sparse_ffn, layer_mask)) {
            return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                               "canonical Metal sparse FFN layer mask is invalid");
        }
        SparseFfnPlan sparse_plan;
        if (!validate_sparse_ffn(layer.layer_index, dense.ffn_gate, dense.ffn_up, dense.ffn_down,
                                 layer_sparse_ffn, sparse_plan, derived_error)) {
            return assignment_error();
        }
        if (!layer_sparse_ffn.runs.empty()) {
            if (impl->sparse_ffn_source > UINT64_MAX - sparse_plan.source_bytes ||
                impl->sparse_ffn_requested > UINT64_MAX - sparse_plan.requested_bytes) {
                return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "sparse FFN program byte count overflows");
            }
            impl->sparse_ffn_source += sparse_plan.source_bytes;
            impl->sparse_ffn_requested += sparse_plan.requested_bytes;
            ++impl->sparse_ffn_layers;
        }
        if (layer_mask) {
            if (layer_mask->block_ids.size() > UINT32_MAX ||
                impl->sparse_ffn_block_ids.size() >
                    UINT32_MAX - layer_mask->block_ids.size()) {
                return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                                   "canonical Metal sparse FFN layer mask table overflows");
            }
            dense.metal.sparse_ffn_block_offset =
                static_cast<uint32_t>(impl->sparse_ffn_block_ids.size());
            dense.metal.sparse_ffn_block_count =
                static_cast<uint32_t>(layer_mask->block_ids.size());
            impl->sparse_ffn_block_offsets.push_back(
                dense.metal.sparse_ffn_block_offset);
            impl->sparse_ffn_block_counts.push_back(
                dense.metal.sparse_ffn_block_count);
            impl->sparse_ffn_block_ids.insert(impl->sparse_ffn_block_ids.end(),
                                              layer_mask->block_ids.begin(),
                                              layer_mask->block_ids.end());
            ++admitted_layer_masks;
        }
        dense.metal.attn_norm = &dense.attn_norm;
        dense.metal.attn_q = &dense.query;
        dense.metal.attn_q_bias = dense.query_bias.data ? &dense.query_bias : nullptr;
        dense.metal.attn_k = &dense.key;
        dense.metal.attn_k_bias = dense.key_bias.data ? &dense.key_bias : nullptr;
        dense.metal.q_norm = dense.query_norm.data ? &dense.query_norm : nullptr;
        dense.metal.k_norm = dense.key_norm.data ? &dense.key_norm : nullptr;
        dense.metal.attn_v = dense.value.data ? &dense.value : nullptr;
        dense.metal.attn_v_bias = dense.value_bias.data ? &dense.value_bias : nullptr;
        dense.metal.attn_o = &dense.attention_output;
        dense.metal.ffn_norm = &dense.ffn_norm;
        dense.metal.ffn_gate = &dense.ffn_gate;
        dense.metal.ffn_up = &dense.ffn_up;
        dense.metal.ffn_down = &dense.ffn_down;
        if (is_moe) {
            dense.metal.moe_gate = &dense.moe_gate;
            dense.metal.moe_gate_scale = &dense.moe_gate_scale;
            dense.metal.moe_up = &dense.moe_up;
            dense.metal.moe_dn = &dense.moe_down;
            dense.metal.moe_dn_scale = &dense.moe_down_scale;
            dense.metal.pre_ffw_2 = &dense.moe_pre_norm;
            dense.metal.post_ffw_1 = dense.post_ffw_1.data ? &dense.post_ffw_1 : nullptr;
            dense.metal.post_ffw_2 = dense.post_ffw_2.data ? &dense.post_ffw_2 : nullptr;
            dense.metal.post_ffw = dense.post_ffw.data ? &dense.post_ffw : nullptr;
            dense.metal.out_scale = dense.out_scale.data ? &dense.out_scale : nullptr;
            dense.metal.exp_inter = static_cast<int>(moe_intermediate);
            dense.metal.n_used = static_cast<int>(moe_payload->selected_count);
            dense.metal.n_experts = static_cast<int>(moe_payload->expert_count);
            dense.metal.moe_router_normalization_scale_bits =
                moe_edges.router_normalization_scale_bits;
            impl->moe_exp_intermediate = std::max(impl->moe_exp_intermediate, moe_intermediate);
            impl->moe_selected_experts = std::max(impl->moe_selected_experts, moe_payload->selected_count);
            impl->moe_experts = std::max(impl->moe_experts, moe_payload->expert_count);
            impl->has_moe = true;
        }
        if (ffn_norm->outputs.size() != 1 || !swiglu || swiglu->outputs.size() != 1) {
            return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                               "canonical dense FFN calibration edges are invalid");
        }
        const uint32_t gate_importance = calibration_slot(
            *ffn_gate, TensorRole::FfnGateWeight, ffn_norm->outputs.front());
        const uint32_t up_importance = calibration_slot(
            *ffn_up, TensorRole::FfnUpWeight, ffn_norm->outputs.front());
        if (gate_importance != UINT32_MAX && up_importance != UINT32_MAX &&
            gate_importance != up_importance) {
            return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                               "canonical dense FFN gate/up calibration inputs differ");
        }
        dense.metal.ffn_input_importance_slot = gate_importance != UINT32_MAX
            ? gate_importance : up_importance;
        dense.metal.ffn_down_importance_slot =
            calibration_slot(*ffn_down, TensorRole::FfnDownWeight,
                             swiglu->outputs.front());
        if (dynamic_selected_blocks != 0 &&
            !build_sparse_ffn_proxy(layer.layer_index, dense.ffn_gate, dense.ffn_up,
                                    dense.ffn_down, dynamic_selected_blocks,
                                    impl->sparse_ffn_proxies,
                                    dense.metal.sparse_ffn_proxy_slot, derived_error)) {
            return assignment_error();
        }
        dense.metal.H = static_cast<int>(impl->hidden);
        dense.metal.inter = static_cast<int>(
            layer_sparse_ffn.runs.empty() ? inter : sparse_plan.packed_intermediate);
        dense.metal.sparse_ffn_full_intermediate = static_cast<int>(inter);
        dense.metal.sparse_ffn = !layer_sparse_ffn.runs.empty();
        dense.metal.sparse_ffn_dense_oracle = sparse_ffn.dense_oracle_selected_blocks != 0;
        dense.metal.Hq = static_cast<int>(attention_payload->query_heads);
        dense.metal.Hk = static_cast<int>(attention_payload->kv_heads);
        dense.metal.Dh = static_cast<int>(attention_payload->head_dimension);
        dense.metal.rope_dim = static_cast<int>(rope_payload->rotary_dimension);
        dense.metal.rope_frequency_dimension = static_cast<int>(rope_frequency_dimension);
        dense.metal.rope_base = rope_base;
        dense.metal.attention_scale = attention_scale;
        dense.metal.rope_interleaved = rope_payload->pairing == RopePairing::Interleaved;
        dense.metal.rope_multi_section = multi_section_rope;
        std::copy(rope_payload->position_sections.begin(), rope_payload->position_sections.end(),
                  dense.metal.rope_sections);
        dense.metal.query_gate_split = fused_query_gate;
        dense.metal.rms_eps = epsilon;
        dense.metal.q_norm_eps = query_norm_epsilon;
        dense.metal.k_norm_eps = key_norm_epsilon;
        dense.metal.window = kMetalUnboundedAttentionWindow;
        dense.metal.swiglu = swiglu_payload->activation == ActivationKind::Silu;
        dense.metal.key_state_alias = false;
        dense.metal.moe_gelu_tanh = is_moe && moe_edges.activation == ActivationKind::GeluTanh;
        dense.metal.moe_reduce_left_to_right =
            is_moe && moe_edges.expert_reduce != nullptr;
        dense.metal.owns_kv = owns_kv;
        dense.metal.is_global = attention_payload->cache_policy == CachePolicy::Global;
        dense.operator_id = attention->id;
        const uint64_t query_width = static_cast<uint64_t>(attention_payload->query_heads) *
                                     attention_payload->head_dimension;
        const uint64_t key_value_width = static_cast<uint64_t>(attention_payload->kv_heads) *
                                         attention_payload->head_dimension;
        const uint64_t query_workspace = fused_query_gate ? 2 * query_width : query_width;
        if (query_workspace > INT_MAX || key_value_width > INT_MAX) {
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "canonical Metal attention geometry exceeds the admitted token buffer");
        }
        if (std::find(claimed_attention_states.begin(), claimed_attention_states.end(),
                      key_state_id) != claimed_attention_states.end() ||
            std::find(claimed_attention_states.begin(), claimed_attention_states.end(),
                      value_state_id) != claimed_attention_states.end()) {
            return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                               "canonical Metal attention state is owned by more than one layer");
        }
        if (impl->attention_key_value_width_sum > UINT64_MAX - key_value_width) {
            return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                               "canonical Metal attention cache geometry overflows");
        }
        dense.metal.cache_width_offset = impl->attention_key_value_width_sum;
        impl->attention_key_value_width_sum += key_value_width;
        claimed_attention_states.push_back(key_state_id);
        claimed_attention_states.push_back(value_state_id);
        impl->has_attention = true;
        impl->attention_query_capacity = std::max(
            impl->attention_query_capacity, static_cast<uint32_t>(query_width));
        impl->attention_key_value_capacity = std::max(
            impl->attention_key_value_capacity, static_cast<uint32_t>(key_value_width));
        impl->token_intermediate = std::max(impl->token_intermediate,
                                             std::max(dense.metal.sparse_ffn_dense_oracle
                                                          ? inter
                                                          : static_cast<uint32_t>(dense.metal.inter),
                                                      static_cast<uint32_t>(query_workspace)));
    }

    if (admitted_layer_masks != sparse_ffn.layer_masks.size()) {
        return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                           "canonical Metal sparse FFN layer mask is not executable");
    }
    if ((!impl->has_attention && !impl->has_recurrent) || impl->token_intermediate == 0) {
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                           "canonical Metal requires an admitted token workspace geometry");
    }

    const SemanticOperator* final_norm =
        &model.operators[lowering_program.final_norm_operator_id];
    const SemanticOperator* output = &model.operators[lowering_program.output_operator_id];
    const auto* final_payload = final_norm ? std::get_if<RmsNormPayload>(&final_norm->payload) : nullptr;
    float final_epsilon = 0.0f;
    if (!final_norm || !output || !final_payload ||
        !f32_bits(final_payload->epsilon_f32_bits, final_epsilon) || final_epsilon < 0.0f ||
        !assign(*final_norm, TensorRole::FinalNormWeight, impl->final_norm, true) ||
        !assign(*output, TensorRole::OutputWeight, impl->output, true)) {
        return assignment_error();
    }
    impl->final_epsilon = final_epsilon;
    for (const CanonicalMetalProgram::Impl::CalibrationBinding& binding : impl->calibration) {
        if (!binding.attached) {
            return calibration_error(CompatibilityError::KERNEL_UNAVAILABLE, binding.target,
                                     "calibration target is outside the admitted FFN Metal pattern");
        }
    }
    impl->metal_session = metal_tok_session_create();
    if (!impl->metal_session) {
        return metal_error(CompatibilityError::CAPABILITY_MISSING,
                           "canonical Metal session resources are unavailable");
    }
    metal_tok_session_enable_error_diagnostics(*impl->metal_session,
                                               derived_iq2_xxs.enable_metal_error_diagnostics);
    uint32_t column_grouped_affine_u2_resources = 0;
    for (uint32_t tensor_id : used_tensor_ids) {
        if (tensor_id >= model.tensors.size()) {
            return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                               "canonical Metal referenced tensor ID is invalid");
        }
        const SemanticTensor& semantic = model.tensors[tensor_id];
        if (semantic.layout.kind != PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip)
            continue;
        Tensor resource;
        if (!column_grouped_affine_u2_skip_256_tensor_view(*impl->package, semantic,
                                                           resource) ||
            !metal_tok_session_register_column_grouped_affine_u2_skip_256(
                *impl->metal_session, resource)) {
            return metal_error(
                CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                "canonical Metal column-grouped UInt2 resource construction failed for tensor " +
                    std::to_string(tensor_id));
        }
        ++column_grouped_affine_u2_resources;
    }
    if (impl->derived_column_grouped_u2) {
        for (const ColumnGroupedU2AtlasEntry& entry :
             impl->derived_column_grouped_u2->entries()) {
            const Tensor resource = derived_column_grouped_u2_tensor(entry);
            if (!metal_tok_session_register_column_grouped_affine_u2_skip_256(
                    *impl->metal_session, resource)) {
                return metal_error(
                    CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                    "canonical Metal derived column-grouped UInt2 resource construction failed for tensor " +
                        std::to_string(entry.binding_id));
            }
            ++column_grouped_affine_u2_resources;
        }
    }
    if (!impl->calibration_widths.empty() &&
        !metal_tok_session_set_importance_slots(*impl->metal_session,
                                                impl->calibration_widths.data(),
                                                static_cast<uint32_t>(impl->calibration_widths.size()))) {
        return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                           "canonical Metal calibration resources are unavailable");
    }
    if (!impl->sparse_ffn_block_ids.empty() &&
        !metal_tok_session_set_sparse_ffn_layer_ids(
            *impl->metal_session, impl->sparse_ffn_block_ids.data(),
            impl->sparse_ffn_block_offsets.data(),
            impl->sparse_ffn_block_counts.data(),
            static_cast<uint32_t>(sparse_ffn.layer_masks.size()))) {
        return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                           "canonical Metal sparse FFN layer-mask construction failed");
    }
    if (impl->sparse_ffn_block_ids.empty() && !impl->sparse_ffn_runs.empty() &&
        !metal_tok_session_set_sparse_ffn_runs(*impl->metal_session, impl->sparse_ffn_runs.data(),
                                               static_cast<uint32_t>(impl->sparse_ffn_runs.size()))) {
        return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                           "canonical Metal sparse FFN run-table construction failed");
    }
    for (const SparseFfnProxyOwner& proxy : impl->sparse_ffn_proxies) {
        if (!metal_tok_session_set_sparse_ffn_proxy(
                *impl->metal_session, proxy.slot, proxy.coefficients.values.data(),
                proxy.coefficients.input_blocks, proxy.coefficients.output_blocks,
                proxy.selected_blocks)) {
            return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                               "canonical Metal sparse FFN proxy resource construction failed");
        }
    }
#if defined(LAPLACE_METAL_TESTING)
    if (sparse_ffn.dense_oracle_selected_blocks != 0 &&
        !metal_tok_session_enable_sparse_ffn_dense_oracle_for_testing(
            *impl->metal_session, static_cast<uint32_t>(impl->sparse_ffn_proxies.size()))) {
        return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                           "canonical Metal dense-oracle FFN diagnostic construction failed");
    }
#endif
    if (!metal_tok_session_dense_ready(*impl->metal_session)) {
        return metal_error(CompatibilityError::CAPABILITY_MISSING,
                           "canonical Metal required pipeline capabilities are unavailable");
    }
    if (impl->has_recurrent && !metal_tok_session_recurrent_ready(*impl->metal_session)) {
        return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                           "canonical Metal recurrent pipeline construction failed");
    }
    RuntimeCapabilities capabilities;
    capabilities.global_fp32_kv = true;
    capabilities.transactional_state = true;
    capabilities.metal_device = true;
    capabilities.metal_library = true;
    capabilities.metal_pipeline = true;
    capabilities.metal_affine_u2_256 = metal_tok_session_affine_u2_256_ready(*impl->metal_session);
    capabilities.metal_column_grouped_affine_u2_skip_256 =
        metal_tok_session_column_grouped_affine_u2_skip_256_ready(*impl->metal_session);
    if (column_grouped_affine_u2_resources != 0 &&
        !capabilities.metal_column_grouped_affine_u2_skip_256) {
        return metal_error(CompatibilityError::CAPABILITY_MISSING,
                           "canonical Metal column-grouped UInt2 pipelines are unavailable");
    }
    const MetalTokMoeCapabilities moe_capabilities =
        metal_tok_session_moe_capabilities(*impl->metal_session);
    capabilities.metal_moe_router_topk = moe_capabilities.router_topk;
    capabilities.metal_moe_gate_up = moe_capabilities.gate_up_q4_k;
    capabilities.metal_moe_down_q5_0 = moe_capabilities.down_q5_0;
    capabilities.metal_moe_down_q8_0 = moe_capabilities.down_q8_0;
    capabilities.metal_moe_reduce = moe_capabilities.reduce;
    SemanticModel execution_model = model;
    if (!rewrite_derived_semantics(execution_model, impl->derived_q2) ||
        !rewrite_derived_semantics(execution_model, impl->derived_iq2_xxs) ||
        !rewrite_derived_semantics(execution_model,
                                   impl->derived_column_grouped_u2)) {
        return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                           "canonical Metal could not bind derived quantization semantics");
    }
    PlanResult planned = plan_canonical_metal(execution_model, request, capabilities,
                                              builtin_canonical_metal_registry());
    if (const auto* report = std::get_if<CompatibilityReport>(&planned)) return *report;
    const auto& selected_plan = std::get<ExecutionPlan>(planned);
    if (selected_plan.program != lowering_program) {
        return metal_error(CompatibilityError::IR_REFERENCE_INVALID,
                           "canonical Metal plan and lowerer program witnesses differ");
    }
    impl->prefill_batch = impl->sparse_ffn_layers == 0 && !selected_plan.entries.empty() && std::all_of(
        selected_plan.entries.begin(), selected_plan.entries.end(), [](const PlanEntry& entry) {
            return entry.phase != ExecutionPhase::Prefill ||
                   entry.descriptor.implementation == KernelImplementation::MetalDensePrefillBatch;
        });
    if (impl->prefill_batch) {
        SessionRequest token_request = request;
        token_request.max_batch = 1;
        const PlanResult token_planned = plan_canonical_metal(
            execution_model, token_request, capabilities, builtin_canonical_metal_registry());
        if (const auto* report = std::get_if<CompatibilityReport>(&token_planned)) return *report;
        impl->plan = std::get<ExecutionPlan>(planned);
        impl->token_plan = std::get<ExecutionPlan>(token_planned);
    } else {
        impl->plan = selected_plan;
        impl->token_plan = impl->plan;
    }
    const MetalResourceSnapshot before_source =
        metal_tok_session_resource_snapshot(*impl->metal_session);
    impl->resource_diagnostics.recommended_max_working_set_size =
        before_source.recommended_max_working_set_size;
    impl->resource_diagnostics.before_source_registration = before_source.current_allocated_size;
    std::vector<uint32_t> replaced_tensor_ids;
    if (derived_q2.register_retained_source_ranges_for_testing) {
        for (const DerivedTensorOwner& owner : impl->derived_q2)
            replaced_tensor_ids.push_back(owner.tensor_id);
    }
    for (const DerivedIQ2XXSEntry& entry : impl->derived_iq2_xxs.entries)
        replaced_tensor_ids.push_back(entry.tensor_id);
    if (impl->derived_column_grouped_u2) {
        for (const ColumnGroupedU2AtlasEntry& entry :
             impl->derived_column_grouped_u2->entries())
            replaced_tensor_ids.push_back(entry.binding_id);
    }
    if (!replaced_tensor_ids.empty()) {
        std::vector<RetainedArtifactRange> retained;
        if (!collect_retained_artifact_ranges(*impl->package, model, used_tensor_ids, replaced_tensor_ids,
                                              retained, impl->original_source_registered,
                                              impl->retained_boundary, artifact_error)) return artifact_error;
        for (const RetainedArtifactRange& range : retained) {
            if (!metal_tok_session_register_weights(*impl->metal_session, range.base, range.length)) {
                return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                                   "canonical Metal could not register a retained artifact range");
            }
            impl->registered_artifact_bases.push_back(range.base);
        }
    } else {
        const long page_value = sysconf(_SC_PAGESIZE);
        if (page_value <= 0) {
            return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                               "canonical Metal could not determine artifact registration page size");
        }
        const uint64_t page = static_cast<uint64_t>(page_value);
        for (const ArtifactSpan& artifact : artifacts) {
            if (!metal_tok_session_register_weights(*impl->metal_session, artifact.base, artifact.length)) {
                return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                                   "canonical Metal could not register an artifact span");
            }
            impl->registered_artifact_bases.push_back(artifact.base);
            if (artifact.length > UINT64_MAX - (page - 1)) {
                return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                                   "canonical Metal artifact registration byte count overflows");
            }
            impl->original_source_registered += (artifact.length + page - 1) / page * page;
        }
    }
    const MetalResourceSnapshot after_source =
        metal_tok_session_resource_snapshot(*impl->metal_session);
    impl->resource_diagnostics.after_source_registration = after_source.current_allocated_size;
    for (const DerivedTensorOwner& owner : impl->derived_q2) {
        if (!metal_tok_session_register_weights(*impl->metal_session, owner.mapping.get(), owner.mapped_length)) {
            return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                               "canonical Metal could not register derived Q2_K storage");
        }
        impl->registered_artifact_bases.push_back(owner.mapping.get());
        if (impl->derived_q2_registered > UINT64_MAX - owner.mapped_length) {
            return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                               "canonical Metal derived Q2_K registered byte count overflows");
        }
        impl->derived_q2_registered += owner.mapped_length;
    }
    if (impl->derived_iq2_xxs.mapping) {
        if (!metal_tok_session_register_weights(*impl->metal_session,
                                                impl->derived_iq2_xxs.mapping.get(),
                                                impl->derived_iq2_xxs.mapped_length)) {
            return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                               "canonical Metal could not register the derived IQ2_XXS atlas");
        }
        impl->registered_artifact_bases.push_back(impl->derived_iq2_xxs.mapping.get());
        impl->derived_iq2_xxs_registered = impl->derived_iq2_xxs.mapped_length;
    }
    if (impl->derived_column_grouped_u2) {
        if (!metal_tok_session_register_weights(
                *impl->metal_session, impl->derived_column_grouped_u2->data(),
                impl->derived_column_grouped_u2->mapped_bytes())) {
            return metal_error(
                CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                "canonical Metal could not register the derived column-grouped UInt2 atlas");
        }
        impl->registered_artifact_bases.push_back(
            impl->derived_column_grouped_u2->data());
        impl->derived_column_grouped_u2_registered =
            impl->derived_column_grouped_u2->mapped_bytes();
    }
    const MetalResourceSnapshot after_atlas =
        metal_tok_session_resource_snapshot(*impl->metal_session);
    impl->resource_diagnostics.after_atlas_registration = after_atlas.current_allocated_size;
    if (!replaced_tensor_ids.empty()) {
        const auto replaced = [&](uint32_t tensor_id) {
            return std::find(replaced_tensor_ids.begin(), replaced_tensor_ids.end(), tensor_id) !=
                   replaced_tensor_ids.end();
        };
        for (uint32_t tensor_id : used_tensor_ids) {
            if (tensor_id >= model.tensors.size() || replaced(tensor_id)) continue;
            for (const TensorPlane& plane : model.tensors[tensor_id].planes) {
                const std::span<const uint8_t> artifact = impl->package->artifact_bytes(plane.artifact_id);
                const uint32_t coverage = plane.offset <= artifact.size() &&
                    plane.length <= artifact.size() - plane.offset && plane.length <= SIZE_MAX
                    ? metal_tok_session_weight_span_coverage(
                          *impl->metal_session, artifact.data() + static_cast<size_t>(plane.offset),
                          static_cast<size_t>(plane.length))
                    : 0;
                if (coverage != 1) {
                    return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                                       "canonical Metal source tensor " + std::to_string(tensor_id) +
                                       " plane offset=" + std::to_string(plane.offset) +
                                       " length=" + std::to_string(plane.length) +
                                       " has retained coverage=" + std::to_string(coverage));
                }
            }
        }
        for (const DerivedTensorOwner& owner : impl->derived_q2) {
            if (metal_tok_session_weight_span_coverage(
                    *impl->metal_session, owner.mapping.get(), owner.logical_length) != 1) {
                return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                                   "canonical Metal derived Q2_K tensor is not covered exactly once");
            }
        }
        for (const DerivedIQ2XXSEntry& entry : impl->derived_iq2_xxs.entries) {
            if (metal_tok_session_weight_span_coverage(
                    *impl->metal_session, entry.tensor.data,
                    static_cast<size_t>(entry.logical_length)) != 1) {
                return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                                   "canonical Metal derived IQ2_XXS tensor " +
                                   std::to_string(entry.tensor_id) + " is not covered exactly once");
            }
        }
        if (impl->derived_column_grouped_u2) {
            for (const ColumnGroupedU2AtlasEntry& entry :
                 impl->derived_column_grouped_u2->entries()) {
                const Tensor tensor = derived_column_grouped_u2_tensor(entry);
                const struct {
                    const uint8_t* data;
                    uint64_t bytes;
                    const char* name;
                } planes[] = {
                    {tensor.data, tensor.data_bytes, "values"},
                    {tensor.scales, tensor.scale_bytes, "scales"},
                    {tensor.biases, tensor.bias_bytes, "biases"},
                };
                for (const auto& plane : planes) {
                    if (plane.bytes > SIZE_MAX ||
                        metal_tok_session_weight_span_coverage(
                            *impl->metal_session, plane.data,
                            static_cast<size_t>(plane.bytes)) != 1) {
                        return metal_error(
                            CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                            "canonical Metal derived column-grouped UInt2 tensor " +
                                std::to_string(entry.binding_id) + " " + plane.name +
                                " plane is not covered exactly once");
                    }
                }
            }
        }
    }
    // Every externally constructible canonical session is strict about weight
    // residency.  Product execution must fail at construction or binding if a
    // tensor is not covered; it must never create an implicit per-dispatch copy.
    metal_tok_session_require_registered_weights(*impl->metal_session, true);
    impl->resource_diagnostics.after_session_construction =
        metal_tok_session_resource_snapshot(*impl->metal_session).current_allocated_size;
    impl->resource_diagnostics.registered_source_bytes = impl->original_source_registered;
    const uint64_t column_u2_source_bytes = impl->derived_column_grouped_u2
        ? impl->derived_column_grouped_u2->source_bytes() : 0;
    if (impl->derived_iq2_xxs.source_bytes > UINT64_MAX - column_u2_source_bytes ||
        impl->derived_iq2_xxs_registered > UINT64_MAX -
            impl->derived_column_grouped_u2_registered) {
        return metal_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                           "canonical Metal derived atlas diagnostics overflow");
    }
    impl->resource_diagnostics.excluded_replaced_bytes =
        impl->derived_iq2_xxs.source_bytes + column_u2_source_bytes;
    impl->resource_diagnostics.retained_boundary_bytes = impl->retained_boundary;
    impl->resource_diagnostics.atlas_bytes =
        impl->derived_iq2_xxs_registered +
        impl->derived_column_grouped_u2_registered;
    const uint64_t unique_registered = impl->original_source_registered +
        impl->derived_q2_registered + impl->derived_iq2_xxs_registered +
        impl->derived_column_grouped_u2_registered;
    impl->resource_diagnostics.registration_overlap_bytes =
        after_atlas.registered_weight_bytes > unique_registered
            ? after_atlas.registered_weight_bytes - unique_registered : 0;
    return CanonicalMetalProgram(std::move(impl));
}

CanonicalMetalCreateResult create_canonical_metal_program(
    std::shared_ptr<const RuntimePackage> package, const SessionRequest& request,
    const CanonicalDerivedQ2KPolicy& derived_q2,
    const CanonicalSparseFfnPolicy& sparse_ffn,
    const CanonicalDerivedIQ2XXSPolicy& derived_iq2_xxs,
    const CanonicalCalibrationPolicy& calibration,
    const CanonicalDerivedColumnGroupedU2Policy& derived_column_u2) {
    if (!package || !package->product_authoritative()) {
        return metal_error(CompatibilityError::PACKAGE_AUTHORITY_REQUIRED,
                           "canonical Metal product construction requires a closed authoritative package");
    }
    bool unqualified_transform =
        !derived_q2.tensor_roles.empty() ||
        derived_q2.register_retained_source_ranges_for_testing ||
        !sparse_ffn.runs.empty() || !sparse_ffn.layer_masks.empty() ||
        sparse_ffn.proxy_selected_blocks != 0 ||
        sparse_ffn.dense_oracle_selected_blocks != 0 ||
        !derived_iq2_xxs.tensor_roles.empty() ||
        derived_iq2_xxs.calibration_cache != nullptr ||
        derived_iq2_xxs.enable_metal_error_diagnostics ||
        !calibration.targets.empty() ||
        !derived_column_u2.tensor_roles.empty() ||
        derived_column_u2.calibration_cache != nullptr;
#if defined(LAPLACE_METAL_TESTING)
    unqualified_transform = unqualified_transform ||
                            derived_iq2_xxs.zero_fill_for_testing ||
                            derived_column_u2.uniform_importance_for_testing;
#endif
    if (unqualified_transform) {
        return metal_error(
            CompatibilityError::PACKAGE_AUTHORITY_REQUIRED,
            "runtime transform or calibration policy requires an authoritative transform certificate");
    }
    return create_canonical_metal_program_internal(std::move(package), request, derived_q2,
                                                   sparse_ffn, derived_iq2_xxs, calibration,
                                                   derived_column_u2);
}

#if defined(LAPLACE_QUALIFICATION_RUNTIME) || defined(LAPLACE_METAL_TESTING)
CanonicalMetalCreateResult create_qualification_canonical_metal_program(
    std::shared_ptr<const RuntimePackage> package, uint32_t maximum_context,
    const CanonicalDerivedQ2KPolicy& derived_q2,
    const CanonicalSparseFfnPolicy& sparse_ffn,
    const CanonicalDerivedIQ2XXSPolicy& derived_iq2_xxs,
    const CanonicalCalibrationPolicy& calibration,
    uint32_t maximum_batch_rows,
    const CanonicalDerivedColumnGroupedU2Policy& derived_column_u2) {
    SessionRequest request;
    request.max_context = maximum_context;
    request.max_batch = maximum_batch_rows;
    request.memory_limit = UINT64_MAX;
    request.enable_prefill = true;
    request.enable_decode = true;
    request.minimum_class = NumericalClass::ExactFp32;
    request.objective = RuntimeObjective::Latency;
    return create_canonical_metal_program_internal(std::move(package), request, derived_q2,
                                                   sparse_ffn, derived_iq2_xxs, calibration,
                                                   derived_column_u2);
}
#endif

CanonicalMetalRunResult CanonicalMetalProgram::run(std::span<const uint32_t> token_ids,
                                                   ExecutionPhase phase,
                                                   OutputMode output_mode) {
    const bool produce_logits = output_mode == OutputMode::Logits;
    const bool produce_sample = output_mode == OutputMode::GreedySample;
    if (!impl_ || token_ids.empty() ||
        token_ids.size() > impl_->maximum_context - impl_->position) {
        return metal_error(CompatibilityError::PLAN_CONTEXT_EXCEEDED);
    }
    if ((phase == ExecutionPhase::Prefill && !impl_->enable_prefill) ||
        (phase == ExecutionPhase::Decode && !impl_->enable_decode)) {
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                           "canonical Metal execution phase was not admitted by the session request");
    }
    if (output_mode == OutputMode::None &&
        (token_ids.size() != 1 ||
         (phase != ExecutionPhase::Prefill && impl_->calibration.empty()))) {
        return metal_error(CompatibilityError::RUNTIME_INPUT_INVALID,
                           "canonical state-only execution requires one prefill token or configured calibration");
    }
    if (impl_->has_recurrent && token_ids.size() > 1) {
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                           "canonical Metal recurrent prefill requires explicit candidate-state chaining");
    }
    if (impl_->prefill_batch && phase == ExecutionPhase::Prefill &&
        (token_ids.size() > impl_->maximum_batch_rows || token_ids.size() > 2 ||
         (token_ids.size() == 2 && impl_->position != 0))) {
        return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                           "canonical Metal F16 prefill batch is exactly two initial tokens");
    }
    CanonicalMetalOutput output;
    if (produce_logits) output.logits.resize(impl_->vocabulary);
    MetalSamplerResult sampled_result;
    uint32_t next_position = impl_->position;
    if (impl_->prefill_batch && phase == ExecutionPhase::Prefill && token_ids.size() == 2) {
        if (token_ids[0] >= impl_->vocabulary || token_ids[1] >= impl_->vocabulary ||
            !metal_tok_session_begin_prefill_batch_with_attention_capacity(
                *impl_->metal_session, static_cast<int>(impl_->hidden),
                static_cast<int>(impl_->token_intermediate), 0, 0, 0,
                MetalTokAttentionCapacity{
                    static_cast<int>(impl_->attention_query_capacity),
                    static_cast<int>(impl_->attention_key_value_capacity),
                    impl_->attention_key_value_width_sum},
                static_cast<int>(impl_->maximum_context), static_cast<int>(impl_->layers.size()),
                static_cast<int>(next_position), 2) ||
            !metal_tok_session_upload_embeddings_batch(*impl_->metal_session, impl_->embedding,
                                                       token_ids.data(), 2,
                                                       static_cast<int>(impl_->hidden),
                                                       static_cast<int>(impl_->vocabulary),
                                                       impl_->embedding_scale)) {
            metal_tok_session_abort(*impl_->metal_session);
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "canonical Metal F16 prefill batch submission failed");
        }
        for (const auto& program_layer : impl_->layers) {
            const uint32_t operator_id = std::visit([](const auto& layer) { return layer.operator_id; }, program_layer);
            const auto planned = std::find_if(impl_->plan.entries.begin(), impl_->plan.entries.end(), [&](const PlanEntry& entry) {
                return entry.phase == ExecutionPhase::Prefill && entry.operator_id == operator_id;
            });
            if (planned == impl_->plan.entries.end() ||
                planned->descriptor.implementation != KernelImplementation::MetalDensePrefillBatch ||
                !std::holds_alternative<DenseLayer>(program_layer) ||
                !metal_tok_session_dense_prefill_batch_layer(
                    *impl_->metal_session, std::get<DenseLayer>(program_layer).metal, 2)) {
                const char* failure = metal_tok_session_last_failure(*impl_->metal_session);
                metal_tok_session_abort(*impl_->metal_session);
                CompatibilityReport report = metal_error(
                    CompatibilityError::KERNEL_UNAVAILABLE,
                    failure ? std::string("canonical Metal F16 prefill batch failed: ") + failure
                            : "canonical Metal F16 prefill batch selected an incompatible layer");
                report.operator_id = operator_id;
                return report;
            }
        }
        const bool row_selected = metal_tok_session_select_prefill_batch_row(
            *impl_->metal_session, 1);
        const bool finalized = row_selected &&
            (produce_sample
                 ? metal_tok_session_final_sampled(
                       *impl_->metal_session, impl_->final_norm, impl_->output,
                       MetalSamplerDescriptor{}, &sampled_result,
                       static_cast<int>(impl_->hidden),
                       static_cast<int>(impl_->vocabulary), impl_->final_epsilon)
                 : produce_logits
                       ? metal_tok_session_final(
                             *impl_->metal_session, impl_->final_norm, impl_->output,
                             output.logits.data(), static_cast<int>(impl_->hidden),
                             static_cast<int>(impl_->vocabulary), impl_->final_epsilon)
                       : metal_tok_session_commit_token(*impl_->metal_session));
        if (!finalized) {
            const char* failure = metal_tok_session_last_failure(*impl_->metal_session);
            metal_tok_session_abort(*impl_->metal_session);
            std::string detail = "canonical Metal F16 prefill batch final output did not complete";
            if (failure) detail += ": " + std::string(failure);
            return metal_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE, std::move(detail));
        }
        next_position += 2;
    } else {
    const ExecutionPlan& active_plan =
        impl_->prefill_batch && phase == ExecutionPhase::Prefill ? impl_->token_plan : impl_->plan;
    const MetalTokAttentionCapacity attention_capacity{
        static_cast<int>(impl_->attention_query_capacity),
        static_cast<int>(impl_->attention_key_value_capacity),
        impl_->attention_key_value_width_sum};
    const auto begin_first_token = [&]() {
        if (impl_->has_attention) {
            return metal_tok_session_begin_with_attention_capacity(
                *impl_->metal_session, static_cast<int>(impl_->hidden),
                static_cast<int>(impl_->token_intermediate),
                static_cast<int>(impl_->moe_exp_intermediate),
                static_cast<int>(impl_->moe_selected_experts),
                static_cast<int>(impl_->moe_experts), attention_capacity,
                static_cast<int>(impl_->maximum_context),
                static_cast<int>(impl_->layers.size()), static_cast<int>(next_position),
                static_cast<uint32_t>(token_ids.size()));
        }
        return metal_tok_session_begin(
            *impl_->metal_session, static_cast<int>(impl_->hidden),
            static_cast<int>(impl_->token_intermediate),
            static_cast<int>(impl_->moe_exp_intermediate),
            static_cast<int>(impl_->moe_selected_experts),
            static_cast<int>(impl_->moe_experts), 0, 0, 0,
            static_cast<int>(impl_->maximum_context),
            static_cast<int>(impl_->layers.size()), static_cast<int>(next_position),
            static_cast<uint32_t>(token_ids.size()));
    };
    const auto begin_continuing_token = [&]() {
        if (impl_->has_attention) {
            return metal_tok_session_begin_continuing_with_attention_capacity(
                *impl_->metal_session, static_cast<int>(impl_->hidden),
                static_cast<int>(impl_->token_intermediate),
                static_cast<int>(impl_->moe_exp_intermediate),
                static_cast<int>(impl_->moe_selected_experts),
                static_cast<int>(impl_->moe_experts), attention_capacity,
                static_cast<int>(impl_->maximum_context),
                static_cast<int>(impl_->layers.size()), static_cast<int>(next_position));
        }
        return metal_tok_session_begin_continuing(
            *impl_->metal_session, static_cast<int>(impl_->hidden),
            static_cast<int>(impl_->token_intermediate),
            static_cast<int>(impl_->moe_exp_intermediate),
            static_cast<int>(impl_->moe_selected_experts),
            static_cast<int>(impl_->moe_experts), 0, 0, 0,
            static_cast<int>(impl_->maximum_context),
            static_cast<int>(impl_->layers.size()), static_cast<int>(next_position));
    };
    for (size_t index = 0; index != token_ids.size(); ++index) {
        const uint32_t token = token_ids[index];
        const bool final_token = index + 1 == token_ids.size();
        if (token >= impl_->vocabulary ||
            !(index == 0 ? begin_first_token() : begin_continuing_token()) ||
            !metal_tok_session_upload_embedding(*impl_->metal_session, impl_->embedding, token,
                                                static_cast<int>(impl_->hidden), static_cast<int>(impl_->vocabulary),
                                                impl_->embedding_scale)) {
            metal_tok_session_abort(*impl_->metal_session);
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE, "canonical Metal begin or embedding submission failed");
        }
        for (const auto& program_layer : impl_->layers) {
            const uint32_t operator_id = std::visit([](const auto& layer) { return layer.operator_id; }, program_layer);
            const auto planned = std::find_if(active_plan.entries.begin(), active_plan.entries.end(), [&](const PlanEntry& entry) {
                return entry.phase == phase && entry.operator_id == operator_id;
            });
            if (planned == active_plan.entries.end() || planned->kernel_id != planned->descriptor.id) {
                metal_tok_session_abort(*impl_->metal_session);
                return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                                   "canonical Metal has no selected descriptor for a layer");
            }
            const bool expected_dense = std::holds_alternative<DenseLayer>(program_layer);
            const KernelImplementation expected_implementation = expected_dense
                ? KernelImplementation::MetalDenseToken : KernelImplementation::MetalRecurrentToken;
            if (planned->descriptor.implementation != expected_implementation) {
                metal_tok_session_abort(*impl_->metal_session);
                CompatibilityReport report = metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                                                         "canonical Metal selected an incompatible execution identity");
                report.operator_id = operator_id;
                return report;
            }
            const bool dispatched = std::visit([&](const auto& layer) {
                using Layer = std::decay_t<decltype(layer)>;
                if constexpr (std::is_same_v<Layer, DenseLayer>) {
                    return metal_tok_session_layer(*impl_->metal_session, layer.metal);
                } else {
                    return metal_tok_session_recurrent_layer(*impl_->metal_session, layer.metal);
                }
            }, program_layer);
            if (!dispatched) {
                const char* failure = metal_tok_session_last_failure(*impl_->metal_session);
                metal_tok_session_abort(*impl_->metal_session);
                std::string detail = "canonical Metal selected descriptor submission failed";
                if (failure) detail += ": " + std::string(failure);
                CompatibilityReport report = metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                                                         std::move(detail));
                report.operator_id = operator_id;
                return report;
            }
        }
        if (!final_token && !metal_tok_session_seal_token(*impl_->metal_session)) {
            metal_tok_session_abort(*impl_->metal_session);
            return metal_error(CompatibilityError::KERNEL_UNAVAILABLE,
                               "canonical Metal could not continue a prompt transaction");
        }
        if (final_token && output_mode == OutputMode::None &&
            !metal_tok_session_commit_token(*impl_->metal_session)) {
            const char* failure = metal_tok_session_last_failure(*impl_->metal_session);
            metal_tok_session_abort(*impl_->metal_session);
            return metal_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE,
                               failure ? failure : "canonical Metal calibration token did not complete");
        }
        if (final_token && produce_logits &&
            !metal_tok_session_final(*impl_->metal_session, impl_->final_norm, impl_->output,
                                     output.logits.data(), static_cast<int>(impl_->hidden),
                                     static_cast<int>(impl_->vocabulary), impl_->final_epsilon)) {
            const char* failure = metal_tok_session_last_failure(*impl_->metal_session);
            metal_tok_session_abort(*impl_->metal_session);
            std::string detail = "canonical Metal final OutputWeight K=" + std::to_string(impl_->hidden) +
                                 " N=" + std::to_string(impl_->vocabulary) +
                                 " format=" + type_name(impl_->output.type) + " did not complete";
            if (failure) detail += ": " + std::string(failure);
            return metal_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE, std::move(detail));
        }
        if (final_token && produce_sample &&
            !metal_tok_session_final_sampled(
                *impl_->metal_session, impl_->final_norm, impl_->output,
                MetalSamplerDescriptor{}, &sampled_result,
                static_cast<int>(impl_->hidden),
                static_cast<int>(impl_->vocabulary), impl_->final_epsilon)) {
            const char* failure = metal_tok_session_last_failure(*impl_->metal_session);
            metal_tok_session_abort(*impl_->metal_session);
            std::string detail = "canonical Metal sampled final OutputWeight K=" +
                                 std::to_string(impl_->hidden) + " N=" +
                                 std::to_string(impl_->vocabulary) + " format=" +
                                 type_name(impl_->output.type) + " did not complete";
            if (failure) detail += ": " + std::string(failure);
            return metal_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE,
                               std::move(detail));
        }
        ++next_position;
    }
    }
    impl_->position = next_position;
    const MetalTokMetrics metrics = metal_tok_session_metrics(*impl_->metal_session);
    output.command_buffers = 1;
    output.operator_count = static_cast<uint32_t>(impl_->package->semantics().operators.size());
    output.cpu_wait_ms = metrics.cpu_wait_ms;
    output.gpu_time_ms = metrics.gpu_time_ms;
    output.peak_session_bytes = metrics.peak_session_bytes;
    output.kv_cache_bytes = metrics.kv_cache_bytes;
    output.requested_projection_source_bytes = metrics.requested_projection_source_bytes;
    output.projection_dispatches = metrics.projection_dispatches;
    output.batched_projection_dispatches = metrics.batched_projection_dispatches;
    output.q4k_projection_dispatches = metrics.q4k_projection_dispatches;
    output.q6k_projection_dispatches = metrics.q6k_projection_dispatches;
    output.grouped_affine_u2_projection_dispatches =
        metrics.grouped_affine_u2_projection_dispatches;
    output.column_grouped_affine_u2_skip_projection_dispatches =
        metrics.column_grouped_affine_u2_skip_projection_dispatches;
    output.counter_sample_count = metrics.counter_sample_count;
    output.profiled = metrics.profiled;
    output.counter_samples = metrics.counter_samples;
    output.split_command_buffer_profile = metrics.split_command_buffer_profile;
    output.qkv_gpu_ms = metrics.qkv_gpu_ms;
    output.attention_gpu_ms = metrics.attention_gpu_ms;
    output.ffn_gpu_ms = metrics.ffn_gpu_ms;
    output.final_gpu_ms = metrics.final_gpu_ms;
    if (produce_logits) {
        output.host_result_bytes = static_cast<uint64_t>(impl_->vocabulary) * sizeof(float);
    } else if (produce_sample) {
        output.sampled = true;
        output.sampled_token_id = sampled_result.token_id;
        output.sampled_logit = sampled_result.logit;
        output.host_result_bytes = sizeof(MetalSamplerResult);
    }
    output.completed = true;
    return output;
}

CanonicalMetalRunResult CanonicalMetalProgram::prefill(std::span<const uint32_t> token_ids) {
    return run(token_ids, ExecutionPhase::Prefill, OutputMode::Logits);
}

CanonicalMetalRunResult CanonicalMetalProgram::decode(uint32_t token_id) {
    return run(std::span<const uint32_t>(&token_id, 1), ExecutionPhase::Decode,
               OutputMode::Logits);
}

CanonicalMetalRunResult CanonicalMetalProgram::prefill_sampled(
    std::span<const uint32_t> token_ids) {
    return run(token_ids, ExecutionPhase::Prefill, OutputMode::GreedySample);
}

CanonicalMetalRunResult CanonicalMetalProgram::decode_sampled(uint32_t token_id) {
    return run(std::span<const uint32_t>(&token_id, 1), ExecutionPhase::Decode,
               OutputMode::GreedySample);
}

CanonicalMetalRunResult CanonicalMetalProgram::advance_prefill(uint32_t token_id) {
    return run(std::span<const uint32_t>(&token_id, 1), ExecutionPhase::Prefill,
               OutputMode::None);
}

CanonicalMetalRunResult CanonicalMetalProgram::accumulate_calibration(uint32_t token_id) {
    return run(std::span<const uint32_t>(&token_id, 1), ExecutionPhase::Decode,
               OutputMode::None);
}

#if defined(LAPLACE_METAL_TESTING)
void canonical_metal_fail_after_completed_submission_for_testing(CanonicalMetalProgram& program) {
    if (program.impl_ && program.impl_->metal_session) {
        metal_tok_session_fail_after_completed_submission_for_testing(*program.impl_->metal_session);
    }
}

bool canonical_metal_tensor_view_for_testing(const RuntimePackage& package, const SemanticTensor& semantic,
                                              Tensor& tensor) {
    return tensor_view(package, semantic, tensor);
}

const char* canonical_metal_first_recurrent_preflight_for_testing(const CanonicalMetalProgram& program) {
    if (!program.impl_) return "canonical Metal program is empty";
    for (const auto& layer : program.impl_->layers) {
        if (const auto* recurrent = std::get_if<RecurrentLayer>(&layer)) {
            return metal_tok_recurrent_layer_preflight_for_testing(recurrent->metal,
                                                                    static_cast<int>(program.impl_->hidden),
                                                                    static_cast<int>(program.impl_->token_intermediate),
                                                                    static_cast<int>(program.impl_->hidden));
        }
    }
    return "canonical Metal program has no recurrent layer";
}

bool canonical_metal_sparse_ffn_probe_for_testing(CanonicalMetalProgram& program,
                                                   std::span<const float> input,
                                                   const CanonicalSparseFfnPolicy& policy,
                                                   std::vector<float>& output,
                                                   double& gpu_ms) {
    if (!program.impl_ || !program.impl_->metal_session ||
        input.size() != program.impl_->hidden || program.impl_->layers.empty()) return false;
    const Tensor* gate = nullptr;
    const Tensor* up = nullptr;
    const Tensor* down = nullptr;
    uint32_t full_intermediate = 0;
    uint32_t layer_index = 0;
    std::visit([&](const auto& layer) {
        gate = &layer.ffn_gate;
        up = &layer.ffn_up;
        down = &layer.ffn_down;
        full_intermediate = static_cast<uint32_t>(layer.metal.sparse_ffn_full_intermediate);
    }, program.impl_->layers.front());
    if (!gate || !up || !down || full_intermediate == 0) return false;
    SparseFfnPlan plan;
    std::string error;
    if (!validate_sparse_ffn(layer_index, *gate, *up, *down, policy, plan, error)) return false;
    std::vector<MetalSparseBlockRun> runs;
    runs.reserve(policy.runs.size());
    for (const SparseBlockRun& run : policy.runs) runs.push_back({run.first, run.count});
    output.resize(program.impl_->hidden);
    return metal_tok_session_probe_ffn_for_testing(
        *program.impl_->metal_session, input.data(), *gate, *up, *down,
        runs.data(), static_cast<uint32_t>(runs.size()), output.data(),
        static_cast<int>(program.impl_->hidden), static_cast<int>(full_intermediate), &gpu_ms);
}

bool canonical_metal_sparse_ffn_windows_for_testing(
    const CanonicalMetalProgram& program,
    std::vector<CanonicalSparseFfnWindowDiagnostic>& windows) {
    windows.clear();
    if (!program.impl_ || !program.impl_->metal_session || program.impl_->sparse_ffn_proxies.empty())
        return false;
    std::vector<uint32_t> starts(program.impl_->sparse_ffn_proxies.size() * 3);
    if (!metal_tok_session_sparse_ffn_windows_for_testing(
            *program.impl_->metal_session, starts.data(),
            static_cast<uint32_t>(program.impl_->sparse_ffn_proxies.size()))) return false;
    windows.reserve(program.impl_->sparse_ffn_proxies.size());
    for (const SparseFfnProxyOwner& owner : program.impl_->sparse_ffn_proxies) {
        windows.push_back({owner.layer_index, starts[3 * owner.slot],
                           starts[3 * owner.slot + 1], starts[3 * owner.slot + 2]});
    }
    return true;
}
#endif

} // namespace Laplace
