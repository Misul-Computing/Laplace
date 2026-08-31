#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace Laplace {

enum class ColumnGroupedQ4Error : uint8_t {
    None,
    SourceLengthMismatch,
    ShapeUnsupported,
    NonFiniteSource,
    InvalidContract,
    StorageLengthMismatch,
    SourceDigestMismatch,
    DerivedDigestMismatch,
    ActiveIndexInvalid,
};

// LaplaceColumnGroupedQ4V1 stores each input column as consecutive blocks of
// 256 output rows. Each block is 128 packed 4-bit values, then LE FP32 scale
// and LE FP32 bias. The decoded value is scale * q + bias.
struct ColumnGroupedQ4V1Contract {
    uint16_t version = 1;
    uint32_t logical_k = 0;
    uint32_t logical_n = 0;
    uint32_t output_block_elements = 256;
    uint32_t packed_bytes = 128;
    uint32_t block_bytes = 136;
    friend bool operator==(const ColumnGroupedQ4V1Contract&,
                           const ColumnGroupedQ4V1Contract&) = default;
};

struct ColumnGroupedQ4V1Storage {
    ColumnGroupedQ4V1Contract contract;
    std::array<uint8_t, 32> source_digest{};
    std::array<uint8_t, 32> derived_digest{};
    std::vector<uint8_t> bytes;
};

bool column_grouped_q4_v1_from_f32(std::span<const float> source,
                                    uint32_t logical_k, uint32_t logical_n,
                                    ColumnGroupedQ4V1Storage* output,
                                    ColumnGroupedQ4Error* error);
bool validate_column_grouped_q4_v1(const ColumnGroupedQ4V1Storage& storage,
                                    const std::array<uint8_t, 32>& expected_source_digest,
                                    ColumnGroupedQ4Error* error);
bool decode_column_grouped_q4_v1(const ColumnGroupedQ4V1Storage& storage,
                                  std::vector<float>* output,
                                  ColumnGroupedQ4Error* error);
bool column_grouped_q4_v1_gemv(const ColumnGroupedQ4V1Storage& storage,
                                std::span<const float> input, std::span<float> output,
                                ColumnGroupedQ4Error* error);
bool column_grouped_q4_v1_selected_gemv(const ColumnGroupedQ4V1Storage& storage,
                                         std::span<const float> input,
                                         std::span<const uint32_t> active_columns,
                                         std::span<float> output,
                                         ColumnGroupedQ4Error* error);

}  // namespace Laplace
