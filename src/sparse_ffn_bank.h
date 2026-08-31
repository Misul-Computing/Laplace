#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "tensor.h"

namespace Laplace {

struct SparseBlockRun {
    uint32_t first = 0;
    uint32_t count = 0;
};

enum class SparseBankError : uint16_t {
    None = 0,
    UnsupportedFormat = 1,
    InvalidShape = 2,
    InvalidRuns = 3,
    SizeOverflow = 4,
};

struct PackedQuantTensor {
    GGMLType type = GGMLType::F32;
    uint64_t K = 0;
    uint64_t N = 0;
    std::vector<uint8_t> bytes;

    Tensor tensor() const noexcept;
};

struct SparseFfnBank {
    uint16_t version = 1;
    uint64_t hidden = 0;
    uint64_t full_intermediate = 0;
    uint64_t packed_intermediate = 0;
    std::vector<SparseBlockRun> runs;
    PackedQuantTensor gate;
    PackedQuantTensor up;
    PackedQuantTensor down;
    uint64_t dense_source_bytes = 0;
    uint64_t physical_bytes = 0;
};

bool pack_sparse_ffn_bank(const Tensor& gate, const Tensor& up, const Tensor& down,
                          std::span<const SparseBlockRun> runs,
                          SparseFfnBank& output, SparseBankError& error);

} // namespace Laplace
