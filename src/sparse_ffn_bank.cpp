#include "sparse_ffn_bank.h"

#include <cstring>
#include <limits>

#include "kernels.h"

namespace Laplace {

namespace {

bool supported(GGMLType type) {
    return type == GGMLType::Q4_K || type == GGMLType::Q6_K;
}

size_t block_bytes(GGMLType type) {
    return type == GGMLType::Q4_K ? sizeof(kernels::block_q4_K) : sizeof(kernels::block_q6_K);
}

bool matrix_bytes(const Tensor& tensor, uint64_t K, uint64_t N, uint64_t& bytes) {
    if (!tensor.data || tensor.n_dims != 2 || tensor.dims[0] != K || tensor.dims[1] != N || K % 256 != 0)
        return false;
    const uint64_t blocks = K / 256;
    const uint64_t bpb = block_bytes(tensor.type);
    if (blocks > std::numeric_limits<uint64_t>::max() / bpb ||
        N > std::numeric_limits<uint64_t>::max() / (blocks * bpb)) return false;
    bytes = N * blocks * bpb;
    return bytes <= std::numeric_limits<size_t>::max();
}

} // namespace

Tensor PackedQuantTensor::tensor() const noexcept {
    Tensor result;
    result.type = type;
    result.n_dims = 2;
    result.dims[0] = K;
    result.dims[1] = N;
    result.data = bytes.data();
    return result;
}

bool pack_sparse_ffn_bank(const Tensor& gate, const Tensor& up, const Tensor& down,
                          std::span<const SparseBlockRun> runs,
                          SparseFfnBank& output, SparseBankError& error) {
    output = {};
    error = SparseBankError::None;
    if (!supported(gate.type) || !supported(up.type) || !supported(down.type)) {
        error = SparseBankError::UnsupportedFormat;
        return false;
    }
    if (gate.n_dims != 2 || up.n_dims != 2 || down.n_dims != 2 ||
        gate.dims[0] == 0 || gate.dims[0] % 256 != 0 ||
        gate.dims[1] == 0 || gate.dims[1] % 256 != 0 ||
        up.dims[0] != gate.dims[0] || up.dims[1] != gate.dims[1] ||
        down.dims[0] != gate.dims[1] || down.dims[1] != gate.dims[0]) {
        error = SparseBankError::InvalidShape;
        return false;
    }
    const uint64_t hidden = gate.dims[0];
    const uint64_t intermediate = gate.dims[1];
    const uint64_t intermediate_blocks = intermediate / 256;
    uint64_t retained_blocks = 0;
    uint64_t end = 0;
    for (const SparseBlockRun run : runs) {
        if (run.count == 0 || run.first < end || run.first >= intermediate_blocks ||
            run.count > intermediate_blocks - run.first) {
            error = SparseBankError::InvalidRuns;
            return false;
        }
        end = static_cast<uint64_t>(run.first) + run.count;
        retained_blocks += run.count;
    }
    if (retained_blocks == 0 || retained_blocks > std::numeric_limits<uint64_t>::max() / 256) {
        error = SparseBankError::InvalidRuns;
        return false;
    }

    uint64_t gate_bytes = 0, up_bytes = 0, down_bytes = 0;
    if (!matrix_bytes(gate, hidden, intermediate, gate_bytes) ||
        !matrix_bytes(up, hidden, intermediate, up_bytes) ||
        !matrix_bytes(down, intermediate, hidden, down_bytes) ||
        gate_bytes > std::numeric_limits<uint64_t>::max() - up_bytes ||
        gate_bytes + up_bytes > std::numeric_limits<uint64_t>::max() - down_bytes) {
        error = SparseBankError::SizeOverflow;
        return false;
    }
    const uint64_t packed_intermediate = retained_blocks * 256;
    SparseFfnBank candidate;
    candidate.hidden = hidden;
    candidate.full_intermediate = intermediate;
    candidate.packed_intermediate = packed_intermediate;
    candidate.runs.assign(runs.begin(), runs.end());
    candidate.gate = {gate.type, hidden, packed_intermediate, {}};
    candidate.up = {up.type, hidden, packed_intermediate, {}};
    candidate.down = {down.type, packed_intermediate, hidden, {}};

    const size_t gate_row = static_cast<size_t>(hidden / 256) * block_bytes(gate.type);
    const size_t up_row = static_cast<size_t>(hidden / 256) * block_bytes(up.type);
    const size_t down_source_row = static_cast<size_t>(intermediate_blocks) * block_bytes(down.type);
    const size_t down_packed_row = static_cast<size_t>(retained_blocks) * block_bytes(down.type);
    candidate.gate.bytes.resize(static_cast<size_t>(packed_intermediate) * gate_row);
    candidate.up.bytes.resize(static_cast<size_t>(packed_intermediate) * up_row);
    candidate.down.bytes.resize(static_cast<size_t>(hidden) * down_packed_row);

    size_t destination_row = 0;
    for (const SparseBlockRun run : runs) {
        const size_t first_row = static_cast<size_t>(run.first) * 256;
        const size_t row_count = static_cast<size_t>(run.count) * 256;
        std::memcpy(candidate.gate.bytes.data() + destination_row * gate_row,
                    gate.data + first_row * gate_row, row_count * gate_row);
        std::memcpy(candidate.up.bytes.data() + destination_row * up_row,
                    up.data + first_row * up_row, row_count * up_row);
        destination_row += row_count;
    }
    for (size_t row = 0; row != hidden; ++row) {
        size_t destination_block = 0;
        for (const SparseBlockRun run : runs) {
            const size_t bytes = static_cast<size_t>(run.count) * block_bytes(down.type);
            std::memcpy(candidate.down.bytes.data() + row * down_packed_row + destination_block * block_bytes(down.type),
                        down.data + row * down_source_row + static_cast<size_t>(run.first) * block_bytes(down.type), bytes);
            destination_block += run.count;
        }
    }
    candidate.dense_source_bytes = gate_bytes + up_bytes + down_bytes;
    candidate.physical_bytes = candidate.gate.bytes.size() + candidate.up.bytes.size() + candidate.down.bytes.size();
    output = std::move(candidate);
    return true;
}

} // namespace Laplace
