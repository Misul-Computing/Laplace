#include "sparse_ffn_bank.h"

#include <algorithm>
#include <cstdio>
#include <vector>

#include "kernels.h"

using namespace Laplace;

namespace {

int checks = 0;

#define CHECK(condition) do { \
    ++checks; \
    if (!(condition)) { \
        std::fprintf(stderr, "FAIL:%d: %s\n", __LINE__, #condition); \
        return 1; \
    } \
} while (0)

Tensor tensor(GGMLType type, int K, int N, const std::vector<uint8_t>& bytes) {
    Tensor value;
    value.type = type;
    value.n_dims = 2;
    value.dims[0] = static_cast<uint64_t>(K);
    value.dims[1] = static_cast<uint64_t>(N);
    value.data = bytes.data();
    return value;
}

} // namespace

int main() {
    constexpr int H = 256;
    constexpr int I = 512;
    const size_t q4_row_h = sizeof(kernels::block_q4_K);
    const size_t q6_row_h = sizeof(kernels::block_q6_K);
    const size_t q4_row_i = 2 * sizeof(kernels::block_q4_K);
    std::vector<uint8_t> gate(static_cast<size_t>(I) * q4_row_h);
    std::vector<uint8_t> up(static_cast<size_t>(I) * q6_row_h);
    std::vector<uint8_t> down(static_cast<size_t>(H) * q4_row_i);
    for (size_t i = 0; i != gate.size(); ++i) gate[i] = static_cast<uint8_t>(i * 3 + 1);
    for (size_t i = 0; i != up.size(); ++i) up[i] = static_cast<uint8_t>(i * 5 + 2);
    for (size_t i = 0; i != down.size(); ++i) down[i] = static_cast<uint8_t>(i * 7 + 3);

    const Tensor gate_tensor = tensor(GGMLType::Q4_K, H, I, gate);
    const Tensor up_tensor = tensor(GGMLType::Q6_K, H, I, up);
    const Tensor down_tensor = tensor(GGMLType::Q4_K, I, H, down);
    const SparseBlockRun selected[] = {{1, 1}};
    SparseFfnBank bank;
    SparseBankError error = SparseBankError::None;
    CHECK(pack_sparse_ffn_bank(gate_tensor, up_tensor, down_tensor, selected, bank, error));
    CHECK(error == SparseBankError::None);
    CHECK(bank.hidden == H && bank.full_intermediate == I && bank.packed_intermediate == 256);
    CHECK(bank.dense_source_bytes == gate.size() + up.size() + down.size());
    CHECK(bank.physical_bytes == bank.gate.bytes.size() + bank.up.bytes.size() + bank.down.bytes.size());
    CHECK(bank.physical_bytes * 2 == bank.dense_source_bytes);
    CHECK(bank.gate.tensor().dims[0] == H && bank.gate.tensor().dims[1] == 256);
    CHECK(bank.up.tensor().type == GGMLType::Q6_K);
    CHECK(bank.down.tensor().dims[0] == 256 && bank.down.tensor().dims[1] == H);
    CHECK(std::equal(bank.gate.bytes.begin(), bank.gate.bytes.end(), gate.begin() + 256 * q4_row_h));
    CHECK(std::equal(bank.up.bytes.begin(), bank.up.bytes.end(), up.begin() + 256 * q6_row_h));
    for (int row = 0; row != H; ++row) {
        CHECK(std::equal(bank.down.bytes.begin() + static_cast<size_t>(row) * q4_row_h,
                         bank.down.bytes.begin() + static_cast<size_t>(row + 1) * q4_row_h,
                         down.begin() + static_cast<size_t>(row) * q4_row_i + q4_row_h));
    }

    const SparseBlockRun overlap[] = {{0, 2}, {1, 1}};
    CHECK(!pack_sparse_ffn_bank(gate_tensor, up_tensor, down_tensor, overlap, bank, error));
    CHECK(error == SparseBankError::InvalidRuns);
    Tensor unsupported = gate_tensor;
    unsupported.type = GGMLType::F16;
    CHECK(!pack_sparse_ffn_bank(unsupported, up_tensor, down_tensor, selected, bank, error));
    CHECK(error == SparseBankError::UnsupportedFormat);

    std::printf("test_sparse_ffn_bank: OK (%d checks)\n", checks);
    return 0;
}
