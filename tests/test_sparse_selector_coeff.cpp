#include "sparse_selector_coeff.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <span>
#include <vector>

#include "fp16.h"
#include "kernels.h"
#include "quant_ref.h"

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

Tensor tensor(GGMLType type, int K, int N, const uint8_t* data) {
    Tensor value;
    value.type = type;
    value.n_dims = 2;
    value.dims[0] = static_cast<uint64_t>(K);
    value.dims[1] = static_cast<uint64_t>(N);
    value.data = data;
    return value;
}

std::vector<uint8_t> make_q4(int K, int N) {
    const size_t bytes = static_cast<size_t>(K / 256) * sizeof(kernels::block_q4_K) * N;
    std::vector<uint8_t> output(bytes);
    auto* blocks = reinterpret_cast<kernels::block_q4_K*>(output.data());
    for (size_t block = 0; block < bytes / sizeof(kernels::block_q4_K); ++block) {
        blocks[block].d = fp32_to_fp16(0.125f + 0.01f * static_cast<float>(block % 3));
        blocks[block].dmin = fp32_to_fp16(0.015f);
        for (int i = 0; i < 12; ++i) blocks[block].scales[i] = static_cast<uint8_t>(1 + (i + block) % 4);
        for (int i = 0; i < 128; ++i) blocks[block].qs[i] = static_cast<uint8_t>(i * 13 + block * 7);
    }
    return output;
}

std::vector<uint8_t> make_q6(int K, int N) {
    const size_t bytes = static_cast<size_t>(K / 256) * sizeof(kernels::block_q6_K) * N;
    std::vector<uint8_t> output(bytes);
    auto* blocks = reinterpret_cast<kernels::block_q6_K*>(output.data());
    for (size_t block = 0; block < bytes / sizeof(kernels::block_q6_K); ++block) {
        blocks[block].d = fp32_to_fp16(0.02f + 0.005f * static_cast<float>(block % 4));
        for (int i = 0; i < 16; ++i) blocks[block].scales[i] = static_cast<int8_t>((i % 5) - 2);
        for (int i = 0; i < 128; ++i) blocks[block].ql[i] = static_cast<uint8_t>(i * 17 + block * 3);
        for (int i = 0; i < 64; ++i) blocks[block].qh[i] = static_cast<uint8_t>(i * 11 + block * 5);
    }
    return output;
}

void decode_row(const Tensor& source, int K, const uint8_t* data, std::vector<float>& values) {
    values.resize(static_cast<size_t>(K));
    if (source.type == GGMLType::Q4_K) {
        const auto* blocks = reinterpret_cast<const quant_ref::block_q4_K*>(data);
        for (int block = 0; block < K / 256; ++block)
            quant_ref::dequant_q4_K(&blocks[block], values.data() + block * 256);
    } else {
        const auto* blocks = reinterpret_cast<const quant_ref::block_q6_K*>(data);
        for (int block = 0; block < K / 256; ++block)
            quant_ref::dequant_q6_K(&blocks[block], values.data() + block * 256);
    }
}

std::vector<float> reference(const Tensor& gate, const Tensor& up, const Tensor& down,
                             int H, int I) {
    const int input_blocks = H / 256;
    const int output_blocks = I / 256;
    const size_t gate_row_bytes = static_cast<size_t>(H / 256) * bytes_per_block(gate.type);
    const size_t up_row_bytes = static_cast<size_t>(H / 256) * bytes_per_block(up.type);
    const size_t down_row_bytes = static_cast<size_t>(I / 256) * bytes_per_block(down.type);
    std::vector<double> combined(static_cast<size_t>(input_blocks) * output_blocks, 0.0);
    std::vector<double> downstream(output_blocks, 0.0);
    std::vector<float> gate_row, up_row, down_row;
    for (int ob = 0; ob < output_blocks; ++ob) {
        for (int row = ob * 256; row < (ob + 1) * 256; ++row) {
            decode_row(gate, H, gate.data + static_cast<size_t>(row) * gate_row_bytes, gate_row);
            decode_row(up, H, up.data + static_cast<size_t>(row) * up_row_bytes, up_row);
            for (int ib = 0; ib < input_blocks; ++ib)
                for (int k = ib * 256; k < (ib + 1) * 256; ++k)
                    combined[static_cast<size_t>(ob) * input_blocks + ib] +=
                        static_cast<double>(gate_row[k]) * gate_row[k] +
                        static_cast<double>(up_row[k]) * up_row[k];
        }
    }
    for (int row = 0; row < H; ++row) {
        decode_row(down, I, down.data + static_cast<size_t>(row) * down_row_bytes, down_row);
        for (int ob = 0; ob < output_blocks; ++ob)
            for (int k = ob * 256; k < (ob + 1) * 256; ++k)
                downstream[ob] += static_cast<double>(down_row[k]) * down_row[k];
    }
    std::vector<float> output(combined.size());
    for (int ob = 0; ob < output_blocks; ++ob)
        for (int ib = 0; ib < input_blocks; ++ib)
            output[static_cast<size_t>(ob) * input_blocks + ib] = static_cast<float>(
                combined[static_cast<size_t>(ob) * input_blocks + ib] * downstream[ob]);
    return output;
}

} // namespace

int main() {
    constexpr int H = 512;
    constexpr int I = 1024;
    std::vector<uint8_t> gate_bytes = make_q4(H, I);
    std::vector<uint8_t> up_bytes = make_q6(H, I);
    std::vector<uint8_t> down_bytes = make_q4(I, H);
    const Tensor gate = tensor(GGMLType::Q4_K, H, I, gate_bytes.data());
    const Tensor up = tensor(GGMLType::Q6_K, H, I, up_bytes.data());
    const Tensor down = tensor(GGMLType::Q4_K, I, H, down_bytes.data());

    SparseSelectorCoefficients coefficients;
    SparseSelectorCoeffError error = SparseSelectorCoeffError::None;
    CHECK(build_sparse_selector_coefficients(gate, gate_bytes, up, up_bytes, down, down_bytes,
                                             coefficients, error));
    CHECK(error == SparseSelectorCoeffError::None);
    CHECK(coefficients.hidden == H && coefficients.intermediate == I);
    CHECK(coefficients.input_blocks == 2 && coefficients.output_blocks == 4);
    CHECK(coefficients.values.size() == 8);
    const std::vector<float> expected = reference(gate, up, down, H, I);
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(std::isfinite(coefficients.values[i]));
        CHECK(coefficients.values[i] >= 0.0f);
        CHECK(std::fabs(coefficients.values[i] - expected[i]) <=
              std::max(1e-4f, std::fabs(expected[i]) * 2e-6f));
    }
    CHECK(coefficients.at(0, 0) == coefficients.values[0]);
    CHECK(coefficients.at(99, 0) == 0.0f);

    std::vector<uint8_t> short_span(gate_bytes.begin(), gate_bytes.end() - 1);
    CHECK(!build_sparse_selector_coefficients(gate, short_span, up, up_bytes, down, down_bytes,
                                              coefficients, error));
    CHECK(error == SparseSelectorCoeffError::InvalidSpan);
    Tensor wrong_type = gate;
    wrong_type.type = GGMLType::Q5_K;
    CHECK(!build_sparse_selector_coefficients(wrong_type, gate_bytes, up, up_bytes, down, down_bytes,
                                              coefficients, error));
    CHECK(error == SparseSelectorCoeffError::UnsupportedFormat);
    Tensor wrong_shape = gate;
    wrong_shape.dims[1] = I - 256;
    CHECK(!build_sparse_selector_coefficients(wrong_shape, gate_bytes, up, up_bytes, down, down_bytes,
                                              coefficients, error));
    CHECK(error == SparseSelectorCoeffError::InvalidShape);
    Tensor null_data = gate;
    null_data.data = nullptr;
    CHECK(!build_sparse_selector_coefficients(null_data, gate_bytes, up, up_bytes, down, down_bytes,
                                              coefficients, error));
    CHECK(error == SparseSelectorCoeffError::InvalidSpan || error == SparseSelectorCoeffError::InvalidShape);

    std::vector<uint8_t> nonfinite_gate = gate_bytes;
    reinterpret_cast<kernels::block_q4_K*>(nonfinite_gate.data())[0].d = 0x7e00;
    Tensor nonfinite_gate_tensor = tensor(GGMLType::Q4_K, H, I, nonfinite_gate.data());
    CHECK(!build_sparse_selector_coefficients(nonfinite_gate_tensor, nonfinite_gate, up, up_bytes,
                                              down, down_bytes, coefficients, error));
    CHECK(error == SparseSelectorCoeffError::NonFiniteSource);

    Tensor oversized_gate = gate;
    oversized_gate.dims[1] = (1u << 30) + 256;
    Tensor oversized_up = up;
    oversized_up.dims[1] = oversized_gate.dims[1];
    Tensor oversized_down = down;
    oversized_down.dims[0] = oversized_gate.dims[1];
    CHECK(!build_sparse_selector_coefficients(oversized_gate, gate_bytes, oversized_up, up_bytes,
                                              oversized_down, down_bytes, coefficients, error));
    CHECK(error == SparseSelectorCoeffError::OutputTooLarge);

    std::printf("test_sparse_selector_coeff: OK (%d checks)\n", checks);
    return 0;
}
