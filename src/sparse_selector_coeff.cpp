#include "sparse_selector_coeff.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "matmul.h"

namespace Laplace {

namespace {

constexpr uint64_t kBlockWidth = 256;
constexpr uint64_t kMaxDecodedRowElements = 1u << 20;
constexpr uint64_t kMaxCoefficientEntries = 1u << 22;

bool supported(GGMLType type) {
    return type == GGMLType::Q4_K || type == GGMLType::Q6_K;
}

bool matrix_bytes(const Tensor& tensor, uint64_t K, uint64_t N, size_t& row_bytes,
                  size_t& total_bytes) {
    if (!tensor.data || tensor.n_dims != 2 || tensor.dims[0] != K || tensor.dims[1] != N ||
        K == 0 || N == 0 || K % kBlockWidth != 0 || K > std::numeric_limits<int>::max() ||
        N > std::numeric_limits<int>::max() || K > kMaxDecodedRowElements) {
        return false;
    }
    const size_t blocks = static_cast<size_t>(K / kBlockWidth);
    const size_t block_bytes = bytes_per_block(tensor.type);
    if (block_bytes == 0 || blocks > std::numeric_limits<size_t>::max() / block_bytes) return false;
    row_bytes = blocks * block_bytes;
    if (static_cast<size_t>(N) > std::numeric_limits<size_t>::max() / row_bytes) return false;
    total_bytes = static_cast<size_t>(N) * row_bytes;
    return true;
}

bool valid_source(const Tensor& tensor, std::span<const uint8_t> bytes,
                  uint64_t K, uint64_t N, size_t& row_bytes,
                  SparseSelectorCoeffError& error) {
    if (!supported(tensor.type)) {
        error = SparseSelectorCoeffError::UnsupportedFormat;
        return false;
    }
    size_t total_bytes = 0;
    if (!matrix_bytes(tensor, K, N, row_bytes, total_bytes)) {
        error = SparseSelectorCoeffError::InvalidShape;
        return false;
    }
    if (bytes.data() != tensor.data || bytes.size() != total_bytes) {
        error = SparseSelectorCoeffError::InvalidSpan;
        return false;
    }
    return true;
}

Tensor row_view(const Tensor& source, uint64_t K, const uint8_t* data) {
    Tensor row;
    row.type = source.type;
    row.n_dims = 2;
    row.dims[0] = K;
    row.dims[1] = 1;
    row.data = data;
    return row;
}

bool decode_row(const Tensor& source, uint64_t K, const uint8_t* data,
                std::vector<float>& values, SparseSelectorCoeffError& error) {
    const Tensor row = row_view(source, K, data);
    dequantize(row, values.data(), static_cast<int>(K));
    for (const float value : values) {
        if (!std::isfinite(value)) {
            error = SparseSelectorCoeffError::NonFiniteSource;
            return false;
        }
    }
    return true;
}

} // namespace

float SparseSelectorCoefficients::at(uint32_t output_block, uint32_t input_block) const noexcept {
    if (output_block >= output_blocks || input_block >= input_blocks) return 0.0f;
    return values[static_cast<size_t>(output_block) * input_blocks + input_block];
}

bool build_sparse_selector_coefficients(
    const Tensor& gate, std::span<const uint8_t> gate_bytes,
    const Tensor& up, std::span<const uint8_t> up_bytes,
    const Tensor& down, std::span<const uint8_t> down_bytes,
    SparseSelectorCoefficients& output, SparseSelectorCoeffError& error) {
    output = {};
    error = SparseSelectorCoeffError::None;

    if (!supported(gate.type) || !supported(up.type) || !supported(down.type)) {
        error = SparseSelectorCoeffError::UnsupportedFormat;
        return false;
    }
    if (gate.n_dims != 2 || up.n_dims != 2 || down.n_dims != 2 ||
        gate.dims[0] == 0 || gate.dims[1] == 0 ||
        gate.dims[0] % kBlockWidth != 0 || gate.dims[1] % kBlockWidth != 0 ||
        up.dims[0] != gate.dims[0] || up.dims[1] != gate.dims[1] ||
        down.dims[0] != gate.dims[1] || down.dims[1] != gate.dims[0]) {
        error = SparseSelectorCoeffError::InvalidShape;
        return false;
    }

    const uint64_t hidden = gate.dims[0];
    const uint64_t intermediate = gate.dims[1];
    const uint64_t input_blocks = hidden / kBlockWidth;
    const uint64_t output_blocks = intermediate / kBlockWidth;
    if (input_blocks == 0 || output_blocks == 0 ||
        input_blocks > std::numeric_limits<uint32_t>::max() ||
        output_blocks > std::numeric_limits<uint32_t>::max() ||
        output_blocks > std::numeric_limits<uint64_t>::max() / input_blocks ||
        output_blocks * input_blocks > kMaxCoefficientEntries) {
        error = SparseSelectorCoeffError::OutputTooLarge;
        return false;
    }

    size_t gate_row_bytes = 0, up_row_bytes = 0, down_row_bytes = 0;
    size_t ignored_total = 0;
    if (!valid_source(gate, gate_bytes, hidden, intermediate, gate_row_bytes, error) ||
        !valid_source(up, up_bytes, hidden, intermediate, up_row_bytes, error) ||
        !valid_source(down, down_bytes, intermediate, hidden, down_row_bytes, error)) {
        return false;
    }
    (void)ignored_total;

    SparseSelectorCoefficients candidate;
    candidate.hidden = hidden;
    candidate.intermediate = intermediate;
    candidate.input_blocks = static_cast<uint32_t>(input_blocks);
    candidate.output_blocks = static_cast<uint32_t>(output_blocks);
    try {
        candidate.values.assign(static_cast<size_t>(output_blocks * input_blocks), 0.0f);
        std::vector<double> gate_up_norms(candidate.values.size(), 0.0);
        std::vector<double> down_norms(static_cast<size_t>(output_blocks), 0.0);
        std::vector<float> gate_row(static_cast<size_t>(hidden));
        std::vector<float> up_row(static_cast<size_t>(hidden));
        std::vector<float> down_row(static_cast<size_t>(intermediate));

        for (uint64_t output_block = 0; output_block < output_blocks; ++output_block) {
            const uint64_t first_row = output_block * kBlockWidth;
            for (uint64_t row = first_row; row < first_row + kBlockWidth; ++row) {
                if (!decode_row(gate, hidden,
                                gate.data + static_cast<size_t>(row) * gate_row_bytes,
                                gate_row, error) ||
                    !decode_row(up, hidden,
                                up.data + static_cast<size_t>(row) * up_row_bytes,
                                up_row, error)) {
                    return false;
                }
                for (uint64_t input_block = 0; input_block < input_blocks; ++input_block) {
                    double& norm = gate_up_norms[static_cast<size_t>(output_block * input_blocks + input_block)];
                    const uint64_t first_value = input_block * kBlockWidth;
                    for (uint64_t k = first_value; k < first_value + kBlockWidth; ++k) {
                        const double gate_value = gate_row[static_cast<size_t>(k)];
                        const double up_value = up_row[static_cast<size_t>(k)];
                        norm += gate_value * gate_value + up_value * up_value;
                    }
                    if (!std::isfinite(norm)) {
                        error = SparseSelectorCoeffError::NonFiniteCoefficient;
                        return false;
                    }
                }
            }
        }

        for (uint64_t row = 0; row < hidden; ++row) {
            if (!decode_row(down, intermediate,
                            down.data + static_cast<size_t>(row) * down_row_bytes,
                            down_row, error)) {
                return false;
            }
            for (uint64_t output_block = 0; output_block < output_blocks; ++output_block) {
                const uint64_t first_value = output_block * kBlockWidth;
                double& norm = down_norms[static_cast<size_t>(output_block)];
                for (uint64_t k = first_value; k < first_value + kBlockWidth; ++k) {
                    const double value = down_row[static_cast<size_t>(k)];
                    norm += value * value;
                }
                if (!std::isfinite(norm)) {
                    error = SparseSelectorCoeffError::NonFiniteCoefficient;
                    return false;
                }
            }
        }

        for (uint64_t output_block = 0; output_block < output_blocks; ++output_block) {
            for (uint64_t input_block = 0; input_block < input_blocks; ++input_block) {
                const double value = gate_up_norms[static_cast<size_t>(output_block * input_blocks + input_block)] *
                                     down_norms[static_cast<size_t>(output_block)];
                if (!std::isfinite(value) || value < 0.0 || value > std::numeric_limits<float>::max()) {
                    error = SparseSelectorCoeffError::NonFiniteCoefficient;
                    return false;
                }
                candidate.values[static_cast<size_t>(output_block * input_blocks + input_block)] =
                    static_cast<float>(value);
            }
        }
    } catch (const std::bad_alloc&) {
        error = SparseSelectorCoeffError::AllocationFailure;
        return false;
    }

    output = std::move(candidate);
    return true;
}

} // namespace Laplace
