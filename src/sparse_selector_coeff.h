#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "tensor.h"

namespace Laplace {

// Coefficient[o * input_blocks + i] is
//   (||gate[o, i]||_2^2 + ||up[o, i]||_2^2) * ||down[:, o]||_2^2.
// The first two norms cover one 256-row output block and one 256-element
// input block. The downstream norm covers the matching 256-element K block
// across every output row of down. This score is a selector input only; it
// does not change model values or promise approximation quality.
struct SparseSelectorCoefficients {
    uint16_t version = 1;
    uint64_t hidden = 0;
    uint64_t intermediate = 0;
    uint32_t input_blocks = 0;
    uint32_t output_blocks = 0;
    std::vector<float> values;

    float at(uint32_t output_block, uint32_t input_block) const noexcept;
};

enum class SparseSelectorCoeffError : uint16_t {
    None = 0,
    UnsupportedFormat = 1,
    InvalidShape = 2,
    InvalidSpan = 3,
    OutputTooLarge = 4,
    AllocationFailure = 5,
    NonFiniteSource = 6,
    NonFiniteCoefficient = 7,
};

// Build a bounded, model-name-free score matrix from three GGUF tensor
// spans. Each span must be the exact contiguous byte owner for its Tensor.
// Gate/up have logical shape [hidden, intermediate]; down is [intermediate,
// hidden]. Q4_K and Q6_K may be mixed across the three tensors.
bool build_sparse_selector_coefficients(
    const Tensor& gate, std::span<const uint8_t> gate_bytes,
    const Tensor& up, std::span<const uint8_t> up_bytes,
    const Tensor& down, std::span<const uint8_t> down_bytes,
    SparseSelectorCoefficients& output, SparseSelectorCoeffError& error);

} // namespace Laplace
