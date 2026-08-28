#include "safetensors_adapter.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace Laplace {
namespace {

bool checked_add(uint64_t left, uint64_t right, uint64_t& result) {
    if (left > std::numeric_limits<uint64_t>::max() - right) return false;
    result = left + right;
    return true;
}

bool checked_multiply(uint64_t left, uint64_t right, uint64_t& result) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) return false;
    result = left * right;
    return true;
}

std::optional<std::pair<ArtifactScalarType, uint32_t>>
artifact_scalar_type(SafeTensorsDtype dtype) {
    switch (dtype) {
        case SafeTensorsDtype::BOOL: return std::pair{ArtifactScalarType::Bool, 1u};
        case SafeTensorsDtype::U8: return std::pair{ArtifactScalarType::U8, 1u};
        case SafeTensorsDtype::I8: return std::pair{ArtifactScalarType::I8, 1u};
        case SafeTensorsDtype::I16: return std::pair{ArtifactScalarType::I16, 2u};
        case SafeTensorsDtype::U16: return std::pair{ArtifactScalarType::U16, 2u};
        case SafeTensorsDtype::F16: return std::pair{ArtifactScalarType::F16, 2u};
        case SafeTensorsDtype::BF16: return std::pair{ArtifactScalarType::BF16, 2u};
        case SafeTensorsDtype::I32: return std::pair{ArtifactScalarType::I32, 4u};
        case SafeTensorsDtype::U32: return std::pair{ArtifactScalarType::U32, 4u};
        case SafeTensorsDtype::F32: return std::pair{ArtifactScalarType::F32, 4u};
        case SafeTensorsDtype::I64: return std::pair{ArtifactScalarType::I64, 8u};
        case SafeTensorsDtype::U64: return std::pair{ArtifactScalarType::U64, 8u};
        default: return std::nullopt;
    }
}

CompatibilityReport adapter_failure(CompatibilityError code,
                                    const PackageView& artifact,
                                    uint32_t tensor_id,
                                    std::string detail) {
    CompatibilityReport report = compatibility_report(code, std::move(detail));
    report.artifact_id = artifact.artifact_id();
    report.artifact_index = artifact.artifact_id().value;
    report.tensor_id = tensor_id;
    return report;
}

} // namespace

SafeTensorsArtifactIndexResult
build_safetensors_artifact_index(const PackageView& artifact) {
    SafeTensorsParseResult parsed = parse_safetensors(artifact.bytes());
    if (const auto* error = std::get_if<SafeTensorsParseError>(&parsed)) {
        CompatibilityReport report = error->report;
        report.artifact_id = artifact.artifact_id();
        report.artifact_index = artifact.artifact_id().value;
        return report;
    }

    const SafeTensorsFile& file = std::get<SafeTensorsFile>(parsed);
    std::vector<const SafeTensorsTensor*> ordered;
    ordered.reserve(file.tensors().size());
    for (const SafeTensorsTensor& tensor : file.tensors()) ordered.push_back(&tensor);
    std::sort(ordered.begin(), ordered.end(), [](const auto* left, const auto* right) {
        return std::tie(left->data_offset, left->data_length, left->dtype, left->shape) <
               std::tie(right->data_offset, right->data_length, right->dtype, right->shape);
    });

    ArtifactIndexInput input;
    input.artifacts.push_back(artifact);
    input.tensors.reserve(ordered.size());
    input.diagnostics.reserve(ordered.size());

    uint64_t data_base = 0;
    if (!checked_add(8, file.header_length(), data_base)) {
        return adapter_failure(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                               artifact, UINT32_MAX,
                               "SafeTensors data base overflows uint64");
    }
    for (size_t index = 0; index != ordered.size(); ++index) {
        const SafeTensorsTensor& source = *ordered[index];
        const uint32_t tensor_id = static_cast<uint32_t>(index);
        if (index != 0) {
            const SafeTensorsTensor& previous = *ordered[index - 1];
            if (source.data_offset == previous.data_offset &&
                source.data_length == previous.data_length &&
                source.dtype == previous.dtype && source.shape == previous.shape) {
                return adapter_failure(CompatibilityError::IMPORT_TENSOR_DUPLICATE,
                                       artifact, tensor_id,
                                       "SafeTensors contains physically indistinguishable tensor records");
            }
        }
        const auto scalar = artifact_scalar_type(source.dtype);
        if (!scalar) {
            return adapter_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                   artifact, tensor_id,
                                   "SafeTensors dtype has no complete physical plane contract");
        }
        if (source.shape.size() > 8) {
            return adapter_failure(CompatibilityError::IR_SHAPE_MISMATCH,
                                   artifact, tensor_id,
                                   "SafeTensors rank exceeds the physical index ABI");
        }

        ArtifactTensorRecord tensor;
        tensor.id = tensor_id;
        tensor.logical_type = scalar->first;
        tensor.logical_dimensions = source.shape;
        tensor.layout.kind = PhysicalLayoutKind::ContiguousRowMajor;
        tensor.layout.version = 1;
        tensor.layout.packing = PackingKind::None;
        tensor.layout.rank = static_cast<uint8_t>(source.shape.size());
        uint64_t element_count = 1;
        for (size_t reverse = source.shape.size(); reverse != 0; --reverse) {
            const size_t axis = reverse - 1;
            tensor.layout.axis_order[axis] = static_cast<uint8_t>(axis);
            tensor.layout.strides[axis] = element_count;
            if (!checked_multiply(element_count, source.shape[axis], element_count)) {
                return adapter_failure(CompatibilityError::IR_SHAPE_MISMATCH,
                                       artifact, tensor_id,
                                       "SafeTensors row-major stride overflows uint64");
            }
        }
        tensor.quantization.kind = QuantizationKind::None;
        tensor.quantization.version = 1;
        tensor.quantization.required_plane_mask =
            artifact_plane_mask(PlaneKind::Values);
        uint64_t source_offset = 0;
        if (!checked_add(data_base, source.data_offset, source_offset)) {
            return adapter_failure(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                   artifact, tensor_id,
                                   "SafeTensors absolute tensor offset overflows uint64");
        }
        tensor.planes.push_back({PlaneKind::Values, scalar->first,
                                 {artifact.artifact_id(), source_offset,
                                  source.data_length},
                                 element_count, scalar->second, 1, 1});
        input.tensors.push_back(std::move(tensor));
        input.diagnostics.push_back({artifact.artifact_id(), {}, tensor_id,
                                     {}, source.name});
    }
    return ArtifactIndex::build(std::move(input));
}

} // namespace Laplace
