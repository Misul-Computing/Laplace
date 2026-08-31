#include "column_grouped_q4.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace Laplace {
namespace {

constexpr uint32_t kOutputBlockElements = 256;
constexpr uint32_t kPackedBytes = 128;
constexpr uint32_t kBlockBytes = 136;

void set_error(ColumnGroupedQ4Error* error, ColumnGroupedQ4Error value) {
    if (error) *error = value;
}

void sha_u32(CC_SHA256_CTX& context, uint32_t value) {
    const uint8_t bytes[4] = {
        static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8u),
        static_cast<uint8_t>(value >> 16u), static_cast<uint8_t>(value >> 24u),
    };
    CC_SHA256_Update(&context, bytes, sizeof(bytes));
}

std::array<uint8_t, 32> source_digest(std::span<const float> source) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    for (float value : source) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        sha_u32(context, bits);
    }
    std::array<uint8_t, 32> digest{};
    CC_SHA256_Final(digest.data(), &context);
    return digest;
}

std::array<uint8_t, 32> derived_digest(const ColumnGroupedQ4V1Storage& storage) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    sha_u32(context, storage.contract.version);
    sha_u32(context, storage.contract.logical_k);
    sha_u32(context, storage.contract.logical_n);
    sha_u32(context, storage.contract.output_block_elements);
    sha_u32(context, storage.contract.packed_bytes);
    sha_u32(context, storage.contract.block_bytes);
    CC_SHA256_Update(&context, storage.source_digest.data(), storage.source_digest.size());
    CC_SHA256_Update(&context, storage.bytes.data(), storage.bytes.size());
    std::array<uint8_t, 32> digest{};
    CC_SHA256_Final(digest.data(), &context);
    return digest;
}

void store_le_f32(uint8_t* bytes, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bytes[0] = static_cast<uint8_t>(bits);
    bytes[1] = static_cast<uint8_t>(bits >> 8u);
    bytes[2] = static_cast<uint8_t>(bits >> 16u);
    bytes[3] = static_cast<uint8_t>(bits >> 24u);
}

float load_le_f32(const uint8_t* bytes) {
    const uint32_t bits = static_cast<uint32_t>(bytes[0]) |
                          (static_cast<uint32_t>(bytes[1]) << 8u) |
                          (static_cast<uint32_t>(bytes[2]) << 16u) |
                          (static_cast<uint32_t>(bytes[3]) << 24u);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint8_t round_to_nearest_even(float value) {
    const float floor_value = std::floor(value);
    const float fraction = value - floor_value;
    uint32_t result = static_cast<uint32_t>(floor_value);
    if (fraction > 0.5f || (fraction == 0.5f && (result & 1u) != 0u)) ++result;
    return static_cast<uint8_t>(std::min(result, 15u));
}

bool contract_valid(const ColumnGroupedQ4V1Contract& contract, size_t* byte_count) {
    if (contract.version != 1 || contract.logical_k == 0 || contract.logical_n == 0 ||
        contract.logical_n % kOutputBlockElements != 0 ||
        contract.output_block_elements != kOutputBlockElements ||
        contract.packed_bytes != kPackedBytes || contract.block_bytes != kBlockBytes) return false;
    const uint64_t blocks = static_cast<uint64_t>(contract.logical_k) *
                            (contract.logical_n / kOutputBlockElements);
    if (blocks > std::numeric_limits<size_t>::max() / kBlockBytes) return false;
    *byte_count = static_cast<size_t>(blocks) * kBlockBytes;
    return true;
}

const uint8_t* block_at(const ColumnGroupedQ4V1Storage& storage, uint32_t k, uint32_t n) {
    const uint32_t output_blocks = storage.contract.logical_n / kOutputBlockElements;
    const uint64_t block = static_cast<uint64_t>(k) * output_blocks + n / kOutputBlockElements;
    return storage.bytes.data() + block * kBlockBytes;
}

float decode_value(const ColumnGroupedQ4V1Storage& storage, uint32_t k, uint32_t n) {
    const uint8_t* block = block_at(storage, k, n);
    const uint8_t packed = block[(n & 255u) / 2u];
    const uint8_t q = static_cast<uint8_t>((packed >> (4u * (n & 1u))) & 0x0fu);
    return load_le_f32(block + kPackedBytes) * static_cast<float>(q) +
           load_le_f32(block + kPackedBytes + sizeof(float));
}

bool validate_for_compute(const ColumnGroupedQ4V1Storage& storage,
                          std::span<const float> input, std::span<float> output,
                          ColumnGroupedQ4Error* error) {
    if (!validate_column_grouped_q4_v1(storage, storage.source_digest, error)) return false;
    if (input.size() != storage.contract.logical_k || output.size() != storage.contract.logical_n) {
        set_error(error, ColumnGroupedQ4Error::ShapeUnsupported);
        return false;
    }
    return true;
}

}  // namespace

bool column_grouped_q4_v1_from_f32(std::span<const float> source, uint32_t logical_k,
                                    uint32_t logical_n, ColumnGroupedQ4V1Storage* output,
                                    ColumnGroupedQ4Error* error) {
    if (!output) {
        set_error(error, ColumnGroupedQ4Error::InvalidContract);
        return false;
    }
    if (logical_k == 0 || logical_n == 0 || logical_n % kOutputBlockElements != 0) {
        set_error(error, ColumnGroupedQ4Error::ShapeUnsupported);
        return false;
    }
    const uint64_t expected_values = static_cast<uint64_t>(logical_k) * logical_n;
    if (expected_values != source.size()) {
        set_error(error, ColumnGroupedQ4Error::SourceLengthMismatch);
        return false;
    }
    for (float value : source) {
        if (!std::isfinite(value)) {
            set_error(error, ColumnGroupedQ4Error::NonFiniteSource);
            return false;
        }
    }

    ColumnGroupedQ4V1Storage candidate;
    candidate.contract.logical_k = logical_k;
    candidate.contract.logical_n = logical_n;
    size_t total_bytes = 0;
    if (!contract_valid(candidate.contract, &total_bytes)) {
        set_error(error, ColumnGroupedQ4Error::InvalidContract);
        return false;
    }
    candidate.bytes.resize(total_bytes);
    const uint32_t output_blocks = logical_n / kOutputBlockElements;
    for (uint32_t k = 0; k < logical_k; ++k) {
        for (uint32_t output_block = 0; output_block < output_blocks; ++output_block) {
            uint8_t* block = candidate.bytes.data() +
                             (static_cast<size_t>(k) * output_blocks + output_block) * kBlockBytes;
            const uint32_t first_row = output_block * kOutputBlockElements;
            float minimum = source[static_cast<size_t>(first_row) * logical_k + k];
            float maximum = minimum;
            for (uint32_t lane = 1; lane < kOutputBlockElements; ++lane) {
                const float value = source[static_cast<size_t>(first_row + lane) * logical_k + k];
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }
            const float scale = (maximum - minimum) / 15.0f;
            std::memset(block, 0, kPackedBytes);
            if (scale != 0.0f) {
                for (uint32_t lane = 0; lane < kOutputBlockElements; ++lane) {
                    const float value = source[static_cast<size_t>(first_row + lane) * logical_k + k];
                    const uint8_t q = round_to_nearest_even((value - minimum) / scale);
                    block[lane / 2u] |= static_cast<uint8_t>(q << (4u * (lane & 1u)));
                }
            }
            store_le_f32(block + kPackedBytes, scale);
            store_le_f32(block + kPackedBytes + sizeof(float), minimum);
        }
    }
    candidate.source_digest = source_digest(source);
    candidate.derived_digest = derived_digest(candidate);
    *output = std::move(candidate);
    set_error(error, ColumnGroupedQ4Error::None);
    return true;
}

bool validate_column_grouped_q4_v1(const ColumnGroupedQ4V1Storage& storage,
                                    const std::array<uint8_t, 32>& expected_source_digest,
                                    ColumnGroupedQ4Error* error) {
    size_t expected_bytes = 0;
    if (!contract_valid(storage.contract, &expected_bytes)) {
        set_error(error, ColumnGroupedQ4Error::InvalidContract);
        return false;
    }
    if (storage.bytes.size() != expected_bytes) {
        set_error(error, ColumnGroupedQ4Error::StorageLengthMismatch);
        return false;
    }
    if (storage.source_digest != expected_source_digest) {
        set_error(error, ColumnGroupedQ4Error::SourceDigestMismatch);
        return false;
    }
    if (storage.derived_digest != derived_digest(storage)) {
        set_error(error, ColumnGroupedQ4Error::DerivedDigestMismatch);
        return false;
    }
    set_error(error, ColumnGroupedQ4Error::None);
    return true;
}

bool decode_column_grouped_q4_v1(const ColumnGroupedQ4V1Storage& storage,
                                  std::vector<float>* output, ColumnGroupedQ4Error* error) {
    if (!output || !validate_column_grouped_q4_v1(storage, storage.source_digest, error)) return false;
    output->resize(static_cast<size_t>(storage.contract.logical_k) * storage.contract.logical_n);
    for (uint32_t n = 0; n < storage.contract.logical_n; ++n)
        for (uint32_t k = 0; k < storage.contract.logical_k; ++k)
            (*output)[static_cast<size_t>(n) * storage.contract.logical_k + k] = decode_value(storage, k, n);
    set_error(error, ColumnGroupedQ4Error::None);
    return true;
}

bool column_grouped_q4_v1_gemv(const ColumnGroupedQ4V1Storage& storage,
                                std::span<const float> input, std::span<float> output,
                                ColumnGroupedQ4Error* error) {
    if (!validate_for_compute(storage, input, output, error)) return false;
    for (uint32_t n = 0; n < storage.contract.logical_n; ++n) {
        double sum = 0.0;
        for (uint32_t k = 0; k < storage.contract.logical_k; ++k)
            sum += static_cast<double>(input[k]) * decode_value(storage, k, n);
        output[n] = static_cast<float>(sum);
    }
    set_error(error, ColumnGroupedQ4Error::None);
    return true;
}

bool column_grouped_q4_v1_selected_gemv(const ColumnGroupedQ4V1Storage& storage,
                                         std::span<const float> input,
                                         std::span<const uint32_t> active_columns,
                                         std::span<float> output,
                                         ColumnGroupedQ4Error* error) {
    if (!validate_for_compute(storage, input, output, error)) return false;
    uint32_t previous = 0;
    for (size_t index = 0; index < active_columns.size(); ++index) {
        const uint32_t k = active_columns[index];
        if (k >= storage.contract.logical_k || (index != 0 && k <= previous)) {
            set_error(error, ColumnGroupedQ4Error::ActiveIndexInvalid);
            return false;
        }
        previous = k;
    }
    for (uint32_t n = 0; n < storage.contract.logical_n; ++n) {
        double sum = 0.0;
        for (uint32_t k : active_columns)
            sum += static_cast<double>(input[k]) * decode_value(storage, k, n);
        output[n] = static_cast<float>(sum);
    }
    set_error(error, ColumnGroupedQ4Error::None);
    return true;
}

}  // namespace Laplace
