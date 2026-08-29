#include "compat_rule.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "gguf.h"
#include "gguf_fact_keys.h"
#include "gguf_index.h"

namespace Laplace {

namespace {

bool checked_add_u64(uint64_t left, uint64_t right, uint64_t& result) {
    if (left > UINT64_MAX - right) return false;
    result = left + right;
    return true;
}

bool checked_multiply_u64(uint64_t left, uint64_t right, uint64_t& result) {
    if (left != 0 && right > UINT64_MAX / left) return false;
    result = left * right;
    return true;
}

bool checked_tensor_bytes(const GGUFTensorInfo& info, uint64_t& bytes, uint64_t& elements) {
    elements = 1;
    for (uint32_t index = 0; index != info.n_dims; ++index) {
        if (!checked_multiply_u64(elements, info.dims[index], elements)) return false;
    }
    const int block_elements = elements_per_block(info.type);
    const size_t block_bytes = bytes_per_block(info.type);
    if (block_elements == 0 || block_bytes == 0 || elements == 0) {
        bytes = 0;
        return true;
    }
    uint64_t blocks = elements / static_cast<uint64_t>(block_elements);
    if (elements % static_cast<uint64_t>(block_elements) != 0) {
        if (blocks == UINT64_MAX) return false;
        ++blocks;
    }
    return checked_multiply_u64(blocks, block_bytes, bytes);
}

CompatibilityReport import_error(CompatibilityError error) {
    CompatibilityReport report = package_report(error);
    report.stage = CompatibilityStage::Import;
    return report;
}

CompatibilityReport physical_import_error(CompatibilityError error, const PackageView& artifact,
                                           std::string detail, uint32_t tensor = UINT32_MAX,
                                           CanonicalFactKey fact = {}) {
    CompatibilityReport report = import_error(error);
    report.detail = std::move(detail);
    report.artifact_id = artifact.artifact_id();
    report.artifact_index = artifact.artifact_id().value;
    report.tensor_id = tensor;
    report.fact_key = fact;
    return report;
}

bool physical_tensor_format(GGMLType input, PackageTensorEvidence& output) {
    switch (input) {
    case GGMLType::F32:
        output.storage_type = ScalarType::F32;
        return true;
    case GGMLType::F16:
        output.storage_type = ScalarType::F16;
        return true;
    case GGMLType::Q4_0:
    case GGMLType::Q4_K:
    case GGMLType::Q5_0:
    case GGMLType::Q6_K:
    case GGMLType::Q8_0:
        output.storage_type = ScalarType::U8;
        output.layout = PhysicalLayoutKind::GgufBlocked;
        output.quantization = QuantizationKind::BlockedAffine;
        output.block_elements = static_cast<uint32_t>(elements_per_block(input));
        output.block_bytes = static_cast<uint32_t>(bytes_per_block(input));
        return output.block_elements != 0 && output.block_bytes != 0;
    default: return false;
    }
}

bool gguf_physical_format(GGMLType input, ArtifactPhysicalFormat& format) {
    format = {};
    format.version = 1;
    switch (input) {
    case GGMLType::F32:
        format.encoding = ArtifactPhysicalEncoding::F32;
        format.value_type = ArtifactScalarType::F32;
        format.block_elements = 1;
        format.block_bytes = 4;
        return true;
    case GGMLType::F16:
        format.encoding = ArtifactPhysicalEncoding::F16;
        format.value_type = ArtifactScalarType::F16;
        format.block_elements = 1;
        format.block_bytes = 2;
        return true;
    case GGMLType::Q4_K:
        format.encoding = ArtifactPhysicalEncoding::Q4_K;
        format.value_type = ArtifactScalarType::Packed;
        format.scale_type = ArtifactScalarType::F16;
        format.zero_type = ArtifactScalarType::F16;
        format.subscale_type = ArtifactScalarType::Packed;
        format.block_elements = 256;
        format.block_bytes = 144;
        format.scale_bytes = 2;
        format.zero_bytes = 2;
        format.subscale_bytes = 12;
        return true;
    case GGMLType::Q4_0:
        format.encoding = ArtifactPhysicalEncoding::Q4_0;
        format.value_type = ArtifactScalarType::Packed;
        format.scale_type = ArtifactScalarType::F16;
        format.block_elements = 32;
        format.block_bytes = 18;
        format.scale_bytes = 2;
        return true;
    case GGMLType::Q5_0:
        format.encoding = ArtifactPhysicalEncoding::Q5_0;
        format.value_type = ArtifactScalarType::Packed;
        format.scale_type = ArtifactScalarType::F16;
        format.block_elements = 32;
        format.block_bytes = 22;
        format.scale_bytes = 2;
        return true;
    case GGMLType::Q6_K:
        format.encoding = ArtifactPhysicalEncoding::Q6_K;
        format.value_type = ArtifactScalarType::Packed;
        format.scale_type = ArtifactScalarType::F16;
        format.subscale_type = ArtifactScalarType::I8;
        format.block_elements = 256;
        format.block_bytes = 210;
        format.scale_bytes = 2;
        format.subscale_bytes = 16;
        return true;
    case GGMLType::Q8_0:
        format.encoding = ArtifactPhysicalEncoding::Q8_0;
        format.value_type = ArtifactScalarType::Packed;
        format.scale_type = ArtifactScalarType::F16;
        format.block_elements = 32;
        format.block_bytes = 34;
        format.scale_bytes = 2;
        return true;
    default:
        return false;
    }
}

bool key_has_suffix(const std::string& key, std::string_view suffix) {
    return key.size() > suffix.size() && key.ends_with(suffix) &&
           key[key.size() - suffix.size() - 1] == '.';
}

bool key_matches_suffix(const std::string& key, std::string_view suffix) {
    return key == suffix || key_has_suffix(key, suffix);
}

bool metadata_unsigned(const MetaValue& value, uint64_t& output) {
    return std::visit([&](const auto& item) -> bool {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
            if constexpr (std::is_signed_v<T>) {
                if (item < 0) return false;
            }
            output = static_cast<uint64_t>(item);
            return true;
        }
        return false;
    }, value);
}

bool metadata_unsigned_vector(const MetaValue& value, std::vector<uint64_t>& output) {
    return std::visit([&](const auto& item) -> bool {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, MetaArrayU32> || std::is_same_v<T, MetaArrayU64> ||
                      std::is_same_v<T, MetaArrayI32> || std::is_same_v<T, MetaArrayI64>) {
            output.clear();
            output.reserve(item.size());
            for (const auto entry : item) {
                if constexpr (std::is_signed_v<typename T::value_type>) {
                    if (entry < 0) return false;
                }
                output.push_back(static_cast<uint64_t>(entry));
            }
            return true;
        }
        return false;
    }, value);
}

bool metadata_float32(const MetaValue& value, ArtifactF32Bits& output) {
    const auto* item = std::get_if<float>(&value);
    if (!item) return false;
    std::memcpy(&output.value, item, sizeof(output.value));
    return true;
}

void append_gguf_metadata_facts(const GGUFContext& context, ArtifactId artifact_id,
                                std::vector<ArtifactFact>& facts) {
    facts.reserve(facts.size() + gguf_fact_keys::descriptors.size());
    for (const auto& descriptor : gguf_fact_keys::descriptors) {
        std::vector<const GGUFMetadataEntry*> matches;
        for (const GGUFMetadataEntry& entry : context.metadata_entries()) {
            if (key_matches_suffix(entry.key, descriptor.suffix)) matches.push_back(&entry);
        }

        ArtifactFact fact;
        fact.key = descriptor.key;
        fact.authority = ArtifactFactAuthority::Declared;
        fact.scope = {};
        if (descriptor.value_kind == gguf_fact_keys::ValueKind::Float32) {
            fact.value = ArtifactF32Bits{};
        } else {
            fact.value = uint64_t{0};
        }

        if (matches.empty()) {
            fact.state = ArtifactFactState::Missing;
        } else {
            const GGUFMetadataEntry& entry = *matches.front();
            fact.source = {artifact_id, entry.source_offset, entry.source_length};
            if (matches.size() > 1) {
                fact.state = ArtifactFactState::Ambiguous;
            } else if (descriptor.value_kind == gguf_fact_keys::ValueKind::Float32) {
                fact.state = metadata_float32(entry.value, std::get<ArtifactF32Bits>(fact.value))
                    ? ArtifactFactState::Present : ArtifactFactState::WrongType;
            } else {
                uint64_t scalar = 0;
                if (metadata_unsigned(entry.value, scalar)) {
                    fact.value = scalar;
                    fact.state = ArtifactFactState::Present;
                } else if (descriptor.value_kind == gguf_fact_keys::ValueKind::UnsignedOrVector) {
                    std::vector<uint64_t> values;
                    if (metadata_unsigned_vector(entry.value, values)) {
                        fact.value = std::move(values);
                        fact.state = ArtifactFactState::Present;
                    } else {
                        fact.state = ArtifactFactState::WrongType;
                    }
                } else {
                    fact.state = ArtifactFactState::WrongType;
                }
            }
        }
        facts.push_back(std::move(fact));
    }
}

} // namespace

GgufArtifactIndexResult build_gguf_artifact_index(const PackageView& artifact) {
    if (artifact.artifact_id().value == UINT32_MAX || artifact.bytes().empty()) {
        return physical_import_error(CompatibilityError::PACKAGE_BOUNDS_INVALID, artifact,
                                     "GGUF artifact is empty or has no stable ID");
    }

    GGUFContext context;
    if (!context.parse(artifact)) {
        return physical_import_error(CompatibilityError::PACKAGE_BOUNDS_INVALID, artifact,
                                     "GGUF parser rejected malformed package");
    }
    if (context.alignment() == 0 || context.alignment() > UINT32_MAX) {
        return physical_import_error(CompatibilityError::IR_LAYOUT_MISMATCH, artifact,
                                     "GGUF alignment is outside the physical index ABI");
    }

    ArtifactIndexInput input;
    input.artifacts.push_back(artifact);
    append_gguf_metadata_facts(context, artifact.artifact_id(), input.metadata_facts);
    for (const GGUFMetadataEntry& entry : context.metadata_entries()) {
        input.diagnostics.push_back({artifact.artifact_id(), {}, UINT32_MAX, entry.key, {}});
    }

    const auto& infos = context.tensor_infos();
    const auto& tensors = context.tensors();
    if (infos.size() != tensors.size() || infos.empty() || infos.size() > UINT32_MAX) {
        return physical_import_error(CompatibilityError::PACKAGE_BOUNDS_INVALID, artifact,
                                     "GGUF tensor table is missing or inconsistent");
    }
    bool has_quantized_tensor = false;
    for (const GGUFTensorInfo& info : infos) {
        has_quantized_tensor = has_quantized_tensor ||
            (info.type != GGMLType::F32 && info.type != GGMLType::F16);
    }
    if (has_quantized_tensor) {
        const auto quantization_version = context.metadata().find("general.quantization_version");
        if (quantization_version == context.metadata().end() ||
            !std::holds_alternative<uint32_t>(quantization_version->second)) {
            return physical_import_error(CompatibilityError::PACKAGE_BOUNDS_INVALID, artifact,
                                         "quantized GGUF is missing general.quantization_version");
        }
    }

    struct BuiltTensor {
        ArtifactTensorRecord record;
        std::string spelling;
        uint64_t source_offset = 0;
        uint64_t source_length = 0;
    };
    std::vector<BuiltTensor> built;
    built.reserve(infos.size());
    std::vector<size_t> physical_order(infos.size());
    for (size_t index = 0; index != infos.size(); ++index) physical_order[index] = index;
    std::sort(physical_order.begin(), physical_order.end(), [&](size_t left, size_t right) {
        const GGUFTensorInfo& lhs = infos[left];
        const GGUFTensorInfo& rhs = infos[right];
        if (lhs.offset != rhs.offset) return lhs.offset < rhs.offset;
        if (lhs.type != rhs.type) return static_cast<uint32_t>(lhs.type) < static_cast<uint32_t>(rhs.type);
        if (lhs.n_dims != rhs.n_dims) return lhs.n_dims < rhs.n_dims;
        for (uint32_t axis = 0; axis != lhs.n_dims; ++axis) {
            if (lhs.dims[axis] != rhs.dims[axis]) return lhs.dims[axis] < rhs.dims[axis];
        }
        // Tensor IDs are local lookup handles only. The final tie-break keeps
        // those handles deterministic without making them canonical identity.
        return left < right;
    });

    const uint64_t artifact_size = static_cast<uint64_t>(artifact.bytes().size());
    const uint32_t alignment = static_cast<uint32_t>(context.alignment());
    for (size_t physical_rank = 0; physical_rank != physical_order.size(); ++physical_rank) {
        const size_t index = physical_order[physical_rank];
        const GGUFTensorInfo& info = infos[index];
        PackageTensorEvidence physical;
        if (!physical_tensor_format(info.type, physical)) {
            return physical_import_error(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED, artifact,
                                         "GGUF tensor uses an unsupported physical storage format",
                                         static_cast<uint32_t>(index));
        }
        uint64_t bytes = 0;
        uint64_t elements = 0;
        if (!checked_tensor_bytes(info, bytes, elements) || bytes == 0 ||
            info.n_dims > 8 || (info.n_dims != 0 &&
                                std::any_of(info.dims, info.dims + info.n_dims,
                                            [](uint64_t dimension) { return dimension == 0; }))) {
            return physical_import_error(CompatibilityError::PACKAGE_BOUNDS_INVALID, artifact,
                                         "GGUF tensor dimensions or byte count are invalid",
                                         static_cast<uint32_t>(index));
        }
        if (info.offset % context.alignment() != 0) {
            return physical_import_error(CompatibilityError::IR_LAYOUT_MISMATCH, artifact,
                                         "GGUF tensor offset violates the declared alignment",
                                         static_cast<uint32_t>(index));
        }
        uint64_t absolute_offset = 0;
        if (!checked_add_u64(context.data_section_offset(), info.offset, absolute_offset) ||
            absolute_offset % context.alignment() != 0 || absolute_offset > artifact_size ||
            bytes > artifact_size - absolute_offset) {
            CompatibilityReport report = physical_import_error(
                CompatibilityError::PACKAGE_BOUNDS_INVALID, artifact,
                "GGUF tensor span is outside the immutable artifact", static_cast<uint32_t>(index));
            report.source_offset = absolute_offset;
            report.source_length = bytes;
            return report;
        }
        const uint32_t block_elements = static_cast<uint32_t>(elements_per_block(info.type));
        const uint32_t block_bytes = static_cast<uint32_t>(bytes_per_block(info.type));
        const bool blocked = physical.layout == PhysicalLayoutKind::GgufBlocked;
        if (blocked && (info.n_dims == 0 || info.dims[0] % block_elements != 0)) {
            return physical_import_error(CompatibilityError::IR_SHAPE_MISMATCH, artifact,
                                         "GGUF blocked tensor has a partial innermost block",
                                         static_cast<uint32_t>(index));
        }
        uint64_t row_stride = block_bytes;
        if (info.n_dims != 0) {
            const uint64_t row_units = blocked ? info.dims[0] / block_elements : info.dims[0];
            if (!checked_multiply_u64(row_units, block_bytes, row_stride)) {
                return physical_import_error(CompatibilityError::IR_SHAPE_MISMATCH, artifact,
                                             "GGUF tensor row stride overflows uint64",
                                             static_cast<uint32_t>(index));
            }
        }

        BuiltTensor result;
        result.spelling = info.name;
        result.source_offset = absolute_offset;
        result.source_length = bytes;
        if (!gguf_physical_format(info.type, result.record.format)) {
            return physical_import_error(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED, artifact,
                                         "GGUF tensor has no exact supported physical descriptor",
                                         static_cast<uint32_t>(index));
        }
        result.record.coordinate.root = 0;
        result.record.logical_type = blocked ? ArtifactScalarType::F32 :
            (info.type == GGMLType::F16 ? ArtifactScalarType::F16 : ArtifactScalarType::F32);
        result.record.logical_dimensions.assign(info.dims, info.dims + info.n_dims);
        result.record.layout.kind = physical.layout;
        result.record.layout.version = 1;
        result.record.layout.packing = blocked ? PackingKind::Gguf : PackingKind::None;
        result.record.layout.rank = static_cast<uint8_t>(info.n_dims);
        result.record.layout.block_rank = blocked ? 1 : 0;
        for (uint32_t axis = 0; axis != info.n_dims; ++axis) {
            result.record.layout.axis_order[axis] = static_cast<uint8_t>(axis);
        }
        uint64_t stride = 1;
        for (size_t axis = 0; axis != info.n_dims; ++axis) {
            result.record.layout.strides[axis] = stride;
            if (!checked_multiply_u64(stride, info.dims[axis], stride)) {
                return physical_import_error(CompatibilityError::IR_SHAPE_MISMATCH, artifact,
                                             "GGUF tensor logical stride overflows uint64",
                                             static_cast<uint32_t>(index));
            }
        }
        result.record.layout.block_elements = blocked ? block_elements : 0;
        result.record.layout.block_bytes = blocked ? block_bytes : 0;
        result.record.quantization.kind = blocked ? QuantizationKind::BlockedAffine : QuantizationKind::None;
        result.record.quantization.version = 1;
        result.record.quantization.accumulation_type = ScalarType::F32;
        result.record.quantization.scale_type = blocked
            ? static_cast<ScalarType>(result.record.format.scale_type) : static_cast<ScalarType>(0);
        result.record.quantization.zero_type = blocked
            ? static_cast<ScalarType>(result.record.format.zero_type) : static_cast<ScalarType>(0);
        result.record.quantization.block_elements = blocked ? block_elements : 0;
        result.record.quantization.block_bytes = blocked ? block_bytes : 0;
        result.record.quantization.group_size = blocked ? block_elements : 0;
        result.record.quantization.required_plane_mask =
            blocked ? artifact_plane_mask(PlaneKind::Values) : 0;
        result.record.axis.source_rank = static_cast<uint8_t>(info.n_dims);
        for (uint32_t axis = 0; axis != info.n_dims; ++axis) {
            result.record.axis.source_axis_order[axis] = static_cast<uint8_t>(axis);
        }
        result.record.axis.block_axis = blocked ? 0 : UINT8_MAX;
        result.record.axis.block_elements = blocked ? block_elements : 0;
        result.record.axis.bytes_per_block = blocked ? block_bytes : 0;
        result.record.axis.row_stride_bytes = row_stride;
        result.record.axis.plane_order = 0;
        result.record.planes.push_back({PlaneKind::Values,
                                        result.record.format.value_type,
                                        {artifact.artifact_id(), absolute_offset, bytes}, elements,
                                        result.record.format.block_bytes, result.record.format.block_elements,
                                        alignment});
        built.push_back(std::move(result));
    }

    for (size_t index = 0; index != built.size(); ++index) {
        built[index].record.id = static_cast<uint32_t>(index);
        input.diagnostics.push_back({artifact.artifact_id(), {}, static_cast<uint32_t>(index), {},
                                     built[index].spelling});
        input.tensors.push_back(std::move(built[index].record));
    }

    auto result = ArtifactIndex::build(std::move(input));
    if (auto* error = std::get_if<CompatibilityReport>(&result)) {
        if (error->artifact_id.value == UINT32_MAX) error->artifact_id = artifact.artifact_id();
        if (error->artifact_index == UINT32_MAX) error->artifact_index = artifact.artifact_id().value;
    }
    return result;
}

} // namespace Laplace
