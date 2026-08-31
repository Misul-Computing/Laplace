#include "column_grouped_q4.h"
#include "test_util.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

using namespace Laplace;

namespace {

float load_le_f32(const uint8_t* bytes) {
    uint32_t word = static_cast<uint32_t>(bytes[0]) |
                    (static_cast<uint32_t>(bytes[1]) << 8u) |
                    (static_cast<uint32_t>(bytes[2]) << 16u) |
                    (static_cast<uint32_t>(bytes[3]) << 24u);
    float value = 0.0f;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

float independent_value(const ColumnGroupedQ4V1Storage& storage, uint32_t k, uint32_t n) {
    const auto& c = storage.contract;
    const uint32_t output_blocks = c.logical_n / c.output_block_elements;
    const uint32_t block = k * output_blocks + n / c.output_block_elements;
    const uint8_t* bytes = storage.bytes.data() + static_cast<size_t>(block) * c.block_bytes;
    const uint8_t packed = bytes[(n % c.output_block_elements) / 2u];
    const uint8_t q = static_cast<uint8_t>((packed >> (4u * (n & 1u))) & 0x0fu);
    return load_le_f32(bytes + c.packed_bytes) * static_cast<float>(q) +
           load_le_f32(bytes + c.packed_bytes + sizeof(float));
}

void independent_gemv(const ColumnGroupedQ4V1Storage& storage, std::span<const float> input,
                      std::span<float> output, std::span<const uint32_t> active) {
    std::fill(output.begin(), output.end(), 0.0f);
    const bool selected = !active.empty();
    for (uint32_t n = 0; n < storage.contract.logical_n; ++n) {
        double sum = 0.0;
        if (selected) {
            for (uint32_t k : active) sum += static_cast<double>(input[k]) * independent_value(storage, k, n);
        } else {
            for (uint32_t k = 0; k < storage.contract.logical_k; ++k)
                sum += static_cast<double>(input[k]) * independent_value(storage, k, n);
        }
        output[n] = static_cast<float>(sum);
    }
}

std::vector<float> source_matrix(uint32_t k, uint32_t n) {
    std::vector<float> source(static_cast<size_t>(k) * n);
    for (uint32_t row = 0; row < n; ++row) {
        for (uint32_t column = 0; column < k; ++column) {
            source[static_cast<size_t>(row) * k + column] =
                static_cast<float>((static_cast<int>((row * 37u + column * 13u) % 71u) - 35)) / 9.0f;
        }
    }
    return source;
}

void check_outputs(std::span<const float> actual, std::span<const float> expected) {
    CHECK(actual.size() == expected.size());
    for (size_t index = 0; index < actual.size(); ++index)
        CHECK_MSG(almost_equal(actual[index], expected[index], 2.0e-6f, 2.0e-6f),
                  "output %zu actual=%g expected=%g", index, actual[index], expected[index]);
}

void test_column_grouped_q4_layout_contract() {
    constexpr uint32_t K = 8;
    constexpr uint32_t N = 512;
    const std::vector<float> source = source_matrix(K, N);
    ColumnGroupedQ4V1Storage storage;
    ColumnGroupedQ4Error error = ColumnGroupedQ4Error::None;
    CHECK(column_grouped_q4_v1_from_f32(source, K, N, &storage, &error));
    CHECK(error == ColumnGroupedQ4Error::None);
    CHECK(storage.contract.version == 1u);
    CHECK(storage.contract.logical_k == K);
    CHECK(storage.contract.logical_n == N);
    CHECK(storage.contract.output_block_elements == 256u);
    CHECK(storage.contract.packed_bytes == 128u);
    CHECK(storage.contract.block_bytes == 136u);
    CHECK(storage.bytes.size() == static_cast<size_t>(K) * (N / 256u) * 136u);
    CHECK(validate_column_grouped_q4_v1(storage, storage.source_digest, &error));
    CHECK(error == ColumnGroupedQ4Error::None);

    ColumnGroupedQ4V1Storage bad_layout = storage;
    bad_layout.contract.logical_n = 511;
    CHECK(!validate_column_grouped_q4_v1(bad_layout, storage.source_digest, &error));
    CHECK(error == ColumnGroupedQ4Error::InvalidContract);

    ColumnGroupedQ4V1Storage truncated = storage;
    truncated.bytes.pop_back();
    CHECK(!validate_column_grouped_q4_v1(truncated, storage.source_digest, &error));
    CHECK(error == ColumnGroupedQ4Error::StorageLengthMismatch);

    std::array<uint8_t, 32> wrong_digest = storage.source_digest;
    wrong_digest[0] ^= 0x80u;
    CHECK(!validate_column_grouped_q4_v1(storage, wrong_digest, &error));
    CHECK(error == ColumnGroupedQ4Error::SourceDigestMismatch);

    ColumnGroupedQ4V1Storage tampered = storage;
    tampered.bytes[0] ^= 0x01u;
    CHECK(!validate_column_grouped_q4_v1(tampered, storage.source_digest, &error));
    CHECK(error == ColumnGroupedQ4Error::DerivedDigestMismatch);

    CHECK(!column_grouped_q4_v1_from_f32(std::span<const float>(source.data(), source.size() - 1u),
                                          K, N, &storage, &error));
    CHECK(error == ColumnGroupedQ4Error::SourceLengthMismatch);
}

void test_column_grouped_q4_scalar_gemv_matches_independent_decoder() {
    constexpr uint32_t K = 8;
    constexpr uint32_t N = 512;
    const std::vector<float> source = source_matrix(K, N);
    ColumnGroupedQ4V1Storage storage;
    ColumnGroupedQ4Error error = ColumnGroupedQ4Error::None;
    CHECK(column_grouped_q4_v1_from_f32(source, K, N, &storage, &error));

    const std::array<float, K> input = {0.0f, -0.5f, 0.0f, 1.0f, 0.0f, -1.5f, 0.0f, 2.0f};
    const std::array<uint32_t, 4> active = {1u, 3u, 5u, 7u};
    std::vector<float> expected_dense(N), actual_dense(N);
    std::vector<float> expected_sparse(N), actual_sparse(N);
    independent_gemv(storage, input, expected_dense, {});
    independent_gemv(storage, input, expected_sparse, active);
    CHECK(column_grouped_q4_v1_gemv(storage, input, actual_dense, &error));
    CHECK(error == ColumnGroupedQ4Error::None);
    CHECK(column_grouped_q4_v1_selected_gemv(storage, input, active, actual_sparse, &error));
    CHECK(error == ColumnGroupedQ4Error::None);
    check_outputs(actual_dense, expected_dense);
    check_outputs(actual_sparse, expected_sparse);
}

}  // namespace

int main() {
    test_column_grouped_q4_layout_contract();
    test_column_grouped_q4_scalar_gemv_matches_independent_decoder();
    return test_summary("test_column_grouped_q4");
}
