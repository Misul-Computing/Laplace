#include "column_grouped_affine_uint2_skip.h"

#include "fp16.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace Laplace {
namespace {

using Contract = ColumnGroupedAffineUInt2SkipV1Contract;
using Error = ColumnGroupedAffineUInt2SkipV1Error;
using Planes = ColumnGroupedAffineUInt2SkipV1Planes;
using Storage = ColumnGroupedAffineUInt2SkipV1Storage;

constexpr size_t kDigestBytes = 32;

void set_error(Error* error, Error value) {
    if (error) *error = value;
}

bool checked_multiply(uint64_t left, uint64_t right, uint64_t* result) {
    if (right != 0 && left > std::numeric_limits<uint64_t>::max() / right) return false;
    *result = left * right;
    return true;
}

bool checked_size(uint64_t value, size_t* result) {
    if (value > std::numeric_limits<size_t>::max()) return false;
    *result = static_cast<size_t>(value);
    return true;
}

bool make_contract(uint64_t logical_k, uint64_t logical_n, Contract* contract) {
    if (!contract || logical_k == 0 || logical_n == 0 || logical_n % 256u != 0) return false;

    uint64_t logical_elements = 0;
    uint64_t group_count = 0;
    uint64_t values_bytes = 0;
    uint64_t scale_bytes = 0;
    uint64_t bias_bytes = 0;
    if (!checked_multiply(logical_n, logical_k, &logical_elements) ||
        !checked_multiply(logical_n / 256u, logical_k, &group_count) ||
        !checked_multiply(group_count, 64u, &values_bytes) ||
        !checked_multiply(group_count, 2u, &scale_bytes) ||
        !checked_multiply(group_count, 2u, &bias_bytes))
        return false;

    size_t ignored = 0;
    if (!checked_size(logical_elements, &ignored) ||
        !checked_size(values_bytes, &ignored) || !checked_size(scale_bytes, &ignored) ||
        !checked_size(bias_bytes, &ignored) || group_count > std::numeric_limits<size_t>::max())
        return false;

    Contract candidate;
    candidate.logical_k = logical_k;
    candidate.logical_n = logical_n;
    candidate.group_count = group_count;
    candidate.values_bytes = values_bytes;
    candidate.scale_bytes = scale_bytes;
    candidate.bias_bytes = bias_bytes;
    *contract = candidate;
    return true;
}

bool valid_contract(const Contract& contract) {
    Contract expected;
    return make_contract(contract.logical_k, contract.logical_n, &expected) &&
           contract == expected;
}

bool aligned(const void* pointer, uint32_t alignment) {
    return pointer != nullptr && (reinterpret_cast<uintptr_t>(pointer) % alignment) == 0;
}

bool range_end(const void* pointer, size_t bytes, uintptr_t* end) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(pointer);
    if (bytes > std::numeric_limits<uintptr_t>::max() - start) return false;
    *end = start + bytes;
    return true;
}

bool overlaps(const void* left, size_t left_bytes, const void* right, size_t right_bytes) {
    uintptr_t left_end = 0;
    uintptr_t right_end = 0;
    if (!range_end(left, left_bytes, &left_end) || !range_end(right, right_bytes, &right_end))
        return true;
    const uintptr_t left_start = reinterpret_cast<uintptr_t>(left);
    const uintptr_t right_start = reinterpret_cast<uintptr_t>(right);
    return left_start < right_end && right_start < left_end;
}

bool plane_layout_valid(const Contract& contract, const Planes& planes, Error* error) {
    if (!planes.values || !planes.scales || !planes.biases) {
        set_error(error, Error::PlanePointerNull);
        return false;
    }
    if (planes.values_bytes != static_cast<size_t>(contract.values_bytes) ||
        planes.scale_count != static_cast<size_t>(contract.group_count) ||
        planes.bias_count != static_cast<size_t>(contract.group_count)) {
        set_error(error, Error::PlaneLengthMismatch);
        return false;
    }
    if (!aligned(planes.values, contract.plane_alignment) ||
        !aligned(planes.scales, contract.plane_alignment) ||
        !aligned(planes.biases, contract.plane_alignment)) {
        set_error(error, Error::PlaneAlignmentMismatch);
        return false;
    }

    const size_t scalar_bytes = planes.scale_count * sizeof(uint16_t);
    if (overlaps(planes.values, planes.values_bytes, planes.scales, scalar_bytes) ||
        overlaps(planes.values, planes.values_bytes, planes.biases, scalar_bytes) ||
        overlaps(planes.scales, scalar_bytes, planes.biases, scalar_bytes)) {
        set_error(error, Error::PlaneOverlap);
        return false;
    }
    return true;
}

bool digest_present(const std::array<uint8_t, kDigestBytes>& digest) {
    return std::any_of(digest.begin(), digest.end(), [](uint8_t value) { return value != 0; });
}

void sha_bytes(CC_SHA256_CTX& context, const void* data, size_t size) {
    constexpr size_t kChunk = 1u << 20;
    const auto* bytes = static_cast<const uint8_t*>(data);
    while (size != 0) {
        const size_t count = std::min(size, kChunk);
        CC_SHA256_Update(&context, bytes, static_cast<CC_LONG>(count));
        bytes += count;
        size -= count;
    }
}

void sha_u16(CC_SHA256_CTX& context, uint16_t value) {
    const uint8_t bytes[2] = {static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8u)};
    sha_bytes(context, bytes, sizeof(bytes));
}

void sha_u32(CC_SHA256_CTX& context, uint32_t value) {
    const uint8_t bytes[4] = {
        static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8u),
        static_cast<uint8_t>(value >> 16u), static_cast<uint8_t>(value >> 24u),
    };
    sha_bytes(context, bytes, sizeof(bytes));
}

void sha_u64(CC_SHA256_CTX& context, uint64_t value) {
    const uint8_t bytes[8] = {
        static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8u),
        static_cast<uint8_t>(value >> 16u), static_cast<uint8_t>(value >> 24u),
        static_cast<uint8_t>(value >> 32u), static_cast<uint8_t>(value >> 40u),
        static_cast<uint8_t>(value >> 48u), static_cast<uint8_t>(value >> 56u),
    };
    sha_bytes(context, bytes, sizeof(bytes));
}

uint16_t load_le_u16(const uint16_t* values, size_t index) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(values) + index * sizeof(uint16_t);
    return static_cast<uint16_t>(static_cast<uint16_t>(bytes[0]) |
                                 (static_cast<uint16_t>(bytes[1]) << 8u));
}

std::array<uint8_t, kDigestBytes> finish_digest(CC_SHA256_CTX& context) {
    std::array<uint8_t, kDigestBytes> digest{};
    CC_SHA256_Final(digest.data(), &context);
    return digest;
}

bool canonical_storage_view(const Storage& storage) {
    Error ignored = Error::None;
    return valid_contract(storage.contract) &&
           plane_layout_valid(storage.contract, storage.planes, &ignored);
}

std::array<uint8_t, kDigestBytes> storage_digest(const Storage& storage) {
    if (!canonical_storage_view(storage)) return {};

    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    static constexpr uint8_t tag[] = {'L', 'A', 'P', 'S', 'P', 'Q', 'T', '1'};
    sha_bytes(context, tag, sizeof(tag));
    const Contract& contract = storage.contract;
    sha_u16(context, contract.version);
    sha_u16(context, contract.reserved);
    sha_u32(context, contract.group_elements);
    sha_u32(context, contract.packed_bytes_per_group);
    sha_u32(context, contract.scale_bytes_per_group);
    sha_u32(context, contract.bias_bytes_per_group);
    sha_u32(context, contract.plane_alignment);
    sha_u64(context, contract.logical_k);
    sha_u64(context, contract.logical_n);
    sha_u64(context, contract.group_count);
    sha_u64(context, contract.values_bytes);
    sha_u64(context, contract.scale_bytes);
    sha_u64(context, contract.bias_bytes);
    sha_bytes(context, storage.source_digest.data(), storage.source_digest.size());
    sha_bytes(context, storage.provenance_digest.data(), storage.provenance_digest.size());
    sha_bytes(context, storage.planes.values, storage.planes.values_bytes);
    sha_bytes(context, storage.planes.scales,
              storage.planes.scale_count * sizeof(uint16_t));
    sha_bytes(context, storage.planes.biases,
              storage.planes.bias_count * sizeof(uint16_t));
    return finish_digest(context);
}

}  // namespace

bool column_grouped_affine_uint2_skip_v1_make_contract(
    uint64_t logical_k, uint64_t logical_n, Contract* contract, Error* error) {
    set_error(error, Error::None);
    if (!contract) {
        set_error(error, Error::NullOutput);
        return false;
    }
    if (logical_k == 0 || logical_n == 0 || logical_n % 256u != 0) {
        set_error(error, Error::ShapeUnsupported);
        return false;
    }
    if (!make_contract(logical_k, logical_n, contract)) {
        set_error(error, Error::Overflow);
        return false;
    }
    return true;
}

std::array<uint8_t, 32> column_grouped_affine_uint2_skip_v1_storage_digest(
    const Storage& storage) {
    return storage_digest(storage);
}

bool column_grouped_affine_uint2_skip_v1_validate(
    const Storage& storage, const std::array<uint8_t, 32>& expected_source_digest,
    const std::array<uint8_t, 32>& expected_provenance_digest, Error* error) {
    set_error(error, Error::None);
    if (!valid_contract(storage.contract)) {
        set_error(error, Error::ContractMismatch);
        return false;
    }
    if (!plane_layout_valid(storage.contract, storage.planes, error)) return false;
    if (!digest_present(storage.source_digest) || !digest_present(expected_source_digest) ||
        storage.source_digest != expected_source_digest) {
        set_error(error, digest_present(storage.source_digest) && digest_present(expected_source_digest)
                               ? Error::SourceDigestMismatch
                               : Error::DigestMissing);
        return false;
    }
    if (!digest_present(storage.provenance_digest) || !digest_present(expected_provenance_digest) ||
        storage.provenance_digest != expected_provenance_digest) {
        set_error(error, digest_present(storage.provenance_digest) && digest_present(expected_provenance_digest)
                               ? Error::ProvenanceDigestMismatch
                               : Error::DigestMissing);
        return false;
    }
    for (size_t index = 0; index < storage.planes.scale_count; ++index) {
        const float scale = fp16_to_fp32(load_le_u16(storage.planes.scales, index));
        const float bias = fp16_to_fp32(load_le_u16(storage.planes.biases, index));
        if (!std::isfinite(scale) || !std::isfinite(bias)) {
            set_error(error, Error::NonFinitePlane);
            return false;
        }
    }
    if (!digest_present(storage.storage_digest)) {
        set_error(error, Error::DigestMissing);
        return false;
    }
    if (storage.storage_digest != storage_digest(storage)) {
        set_error(error, Error::StorageDigestMismatch);
        return false;
    }
    return true;
}

bool column_grouped_affine_uint2_skip_v1_decode(
    const Storage& storage, const std::array<uint8_t, 32>& expected_source_digest,
    const std::array<uint8_t, 32>& expected_provenance_digest,
    std::span<float> output, Error* error) {
    set_error(error, Error::None);
    if (!output.data()) {
        set_error(error, Error::OutputPointerNull);
        return false;
    }
    uint64_t value_count = 0;
    if (!checked_multiply(storage.contract.logical_k, storage.contract.logical_n, &value_count) ||
        value_count > std::numeric_limits<size_t>::max() ||
        output.size() != static_cast<size_t>(value_count)) {
        set_error(error, Error::OutputLengthMismatch);
        return false;
    }
    if (!column_grouped_affine_uint2_skip_v1_validate(
            storage, expected_source_digest, expected_provenance_digest, error))
        return false;

    const size_t scale_bytes = storage.planes.scale_count * sizeof(uint16_t);
    const size_t bias_bytes = storage.planes.bias_count * sizeof(uint16_t);
    if (overlaps(output.data(), output.size_bytes(), storage.planes.values,
                 storage.planes.values_bytes) ||
        overlaps(output.data(), output.size_bytes(), storage.planes.scales, scale_bytes) ||
        overlaps(output.data(), output.size_bytes(), storage.planes.biases, bias_bytes)) {
        set_error(error, Error::OutputOverlap);
        return false;
    }

    const uint64_t logical_k = storage.contract.logical_k;
    for (uint64_t row = 0; row < storage.contract.logical_n; ++row) {
        const uint64_t row_group = row / storage.contract.group_elements;
        for (uint64_t column = 0; column < logical_k; ++column) {
            const uint64_t group = row_group * logical_k + column;
            const uint8_t* packed = storage.planes.values + group * 64u;
            const uint8_t code = static_cast<uint8_t>(
                (packed[(row % 256u) / 4u] >> (2u * (row & 3u))) & 3u);
            output[static_cast<size_t>(row * logical_k + column)] =
                fp16_to_fp32(load_le_u16(storage.planes.scales, static_cast<size_t>(group))) *
                    static_cast<float>(code) +
                fp16_to_fp32(load_le_u16(storage.planes.biases, static_cast<size_t>(group)));
        }
    }
    return true;
}

}  // namespace Laplace
