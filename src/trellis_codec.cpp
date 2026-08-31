// Generic physical 16x16 K-bit trellis codec.
//
// Adapted in part from PonyExl3 commit
// 8e7fa6b1556f59fc669e25087903b279b9b0346f, Apache License 2.0. See the
// repository LICENSE and NOTICE files for redistribution terms.
// PonyExl3 copyright 2026 Theinruj Toranavikrai.

#include "trellis_codec.h"

#include "fp16.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace Laplace {
namespace {

constexpr uint32_t kStateMask = 0xffffu;

bool valid_bits(uint8_t bits) { return bits >= 1 && bits <= 8; }

void swap16_per_uint32(std::span<uint16_t> words) {
    for (size_t i = 0; i < words.size(); i += 2)
        std::swap(words[i], words[i + 1]);
}

void write_bit(std::span<uint16_t> words, size_t bit, uint32_t value) {
    const size_t word = bit / 16;
    const uint32_t shift = 15u - static_cast<uint32_t>(bit % 16);
    words[word] = static_cast<uint16_t>(words[word] | ((value & 1u) << shift));
}

uint32_t read_bits(std::span<const uint16_t> words, size_t bit, uint8_t bits) {
    uint32_t value = 0;
    for (uint8_t b = 0; b != bits; ++b) {
        const size_t absolute = bit + b;
        const size_t word = absolute / 16;
        const uint32_t shift = 15u - static_cast<uint32_t>(absolute % 16);
        value = (value << 1) | ((words[word] >> shift) & 1u);
    }
    return value;
}

uint32_t lop3_6a(uint32_t a, uint32_t b, uint32_t c) {
    // PTX imm 0x6a: c ^ (a & b). This is the bit operation used by the
    // upstream procedural 3INST decoder; spelling it directly is scalar and
    // independent of any runtime or model family.
    return c ^ (a & b);
}

float half_add_from_u32(uint32_t bits) {
    const _Float16 lo = static_cast<_Float16>(fp16_to_fp32(static_cast<uint16_t>(bits)));
    const _Float16 hi = static_cast<_Float16>(fp16_to_fp32(static_cast<uint16_t>(bits >> 16)));
    return static_cast<float>(lo + hi);
}

} // namespace

size_t trellis_packed_bytes_per_tile(uint8_t bits) {
    return valid_bits(bits) ? static_cast<size_t>(16u * bits * sizeof(uint16_t)) : 0;
}

bool trellis_select_descriptor(TrellisTileLayout layout, uint8_t plane_count,
                               uint8_t bits, uint32_t logical_k, uint32_t logical_n,
                               TrellisCodebook codebook, TrellisKernelFacts kernel,
                               TrellisPhysicalDescriptor* out) {
    if (!out || layout != TrellisTileLayout::TensorCore16x16 || plane_count != 1 ||
        !valid_bits(bits) ||
        logical_k == 0 || logical_n == 0 || logical_k % 16 != 0 || logical_n % 16 != 0 ||
        static_cast<uint8_t>(codebook) > static_cast<uint8_t>(TrellisCodebook::Mul1) ||
        kernel.thread_execution_width == 0 || kernel.max_threads_per_threadgroup < 128)
        return false;
    // K is a physical bit depth. The selection is intentionally bounded while
    // still covering the complete EXL3 trellis bit-depth range.
    const size_t tile_count = static_cast<size_t>(logical_k / 16) * (logical_n / 16);
    const size_t bytes_per_tile = trellis_packed_bytes_per_tile(bits);
    if (tile_count > std::numeric_limits<uint32_t>::max() ||
        bytes_per_tile > std::numeric_limits<uint32_t>::max() ||
        tile_count > std::numeric_limits<uint64_t>::max() / bytes_per_tile)
        return false;
    TrellisPhysicalDescriptor candidate{};
    candidate.layout = layout;
    candidate.bits = bits;
    candidate.plane_count = plane_count;
    candidate.codebook = codebook;
    candidate.logical_k = logical_k;
    candidate.logical_n = logical_n;
    candidate.tile_count = static_cast<uint32_t>(tile_count);
    candidate.packed_bytes_per_tile = static_cast<uint32_t>(bytes_per_tile);
    candidate.tile_stride_bytes = candidate.packed_bytes_per_tile;
    candidate.packed_plane_bytes = tile_count * bytes_per_tile;
    if (!trellis_validate_descriptor(candidate)) return false;
    *out = candidate;
    return true;
}

bool trellis_validate_descriptor(const TrellisPhysicalDescriptor& descriptor) {
    if (descriptor.version != 1 || descriptor.layout != TrellisTileLayout::TensorCore16x16 ||
        !valid_bits(descriptor.bits) || descriptor.plane_count != 1 ||
        descriptor.logical_k == 0 || descriptor.logical_n == 0 ||
        descriptor.logical_k % 16 != 0 || descriptor.logical_n % 16 != 0 ||
        static_cast<uint8_t>(descriptor.codebook) >
            static_cast<uint8_t>(TrellisCodebook::Mul1))
        return false;
    const size_t bytes_per_tile = trellis_packed_bytes_per_tile(descriptor.bits);
    const size_t tile_count = static_cast<size_t>(descriptor.logical_k / 16) *
                              static_cast<size_t>(descriptor.logical_n / 16);
    if (bytes_per_tile == 0 || descriptor.packed_bytes_per_tile != bytes_per_tile ||
        descriptor.tile_stride_bytes != bytes_per_tile ||
        descriptor.tile_count != tile_count ||
        tile_count > std::numeric_limits<uint64_t>::max() / bytes_per_tile ||
        descriptor.packed_plane_bytes != tile_count * bytes_per_tile)
        return false;
    return true;
}

bool trellis_pack_tile(std::span<const uint16_t> encoded,
                       std::span<uint8_t> packed, uint8_t bits) {
    const size_t packed_bytes = trellis_packed_bytes_per_tile(bits);
    if (encoded.size() != TrellisTileValueCount || packed.size() != packed_bytes)
        return false;
    std::array<uint16_t, TrellisTileValueCount / 16 * 8> words{};
    const uint32_t mask = (1u << bits) - 1u;
    const size_t words_per_span = bits;
    for (size_t span = 0; span != 16; ++span) {
        for (size_t i = 0; i != 16; ++i) {
            const uint32_t value = encoded[span * 16 + i] & mask;
            const size_t start = span * 16 * bits + i * bits;
            for (uint8_t b = 0; b != bits; ++b)
                write_bit(std::span<uint16_t>(words), start + b,
                          value >> (bits - 1u - b));
        }
        (void)words_per_span;
    }
    swap16_per_uint32(std::span<uint16_t>(words));
    std::memcpy(packed.data(), words.data(), packed_bytes);
    return true;
}

bool trellis_unpack_tile(std::span<const uint8_t> packed,
                         std::span<uint16_t> states, uint8_t bits) {
    const size_t packed_bytes = trellis_packed_bytes_per_tile(bits);
    if (packed.size() != packed_bytes || states.size() != TrellisTileValueCount)
        return false;
    std::array<uint16_t, TrellisTileValueCount / 16 * 8> words{};
    std::memcpy(words.data(), packed.data(), packed_bytes);
    swap16_per_uint32(std::span<uint16_t>(words));
    std::array<uint16_t, TrellisTileValueCount> fresh{};
    const uint32_t mask = (1u << bits) - 1u;
    for (size_t span = 0; span != 16; ++span)
        for (size_t i = 0; i != 16; ++i)
            fresh[span * 16 + i] = static_cast<uint16_t>(
                read_bits(std::span<const uint16_t>(words),
                          span * 16 * bits + i * bits, bits) & mask);

    uint32_t state = 0;
    for (size_t lap = 0; lap != 2; ++lap) {
        for (size_t i = 0; i != TrellisTileValueCount; ++i) {
            state = ((state << bits) | fresh[i]) & kStateMask;
            states[i] = static_cast<uint16_t>(state);
        }
    }
    return true;
}

float trellis_decode_codeword(uint16_t codeword, TrellisCodebook codebook) {
    uint32_t x = codeword;
    switch (codebook) {
        case TrellisCodebook::Default:
            x = x * 89226354u + 64248484u;
            x = lop3_6a(x, 0x8fff8fffu, 0x3b603b60u);
            return half_add_from_u32(x);
        case TrellisCodebook::Mcg:
            x = x * 0xcbac1fedu;
            x = lop3_6a(x, 0x8fff8fffu, 0x3b603b60u);
            return half_add_from_u32(x);
        case TrellisCodebook::Mul1: {
            x = x * 0x83dcd12du;
            uint32_t sum = 0x6400u;
            for (uint32_t lane = 0; lane != 4; ++lane)
                sum += (x >> (8u * lane)) & 0xffu;
            const float h = fp16_to_fp32(static_cast<uint16_t>(sum));
            const _Float16 value = static_cast<_Float16>(
                h * fp16_to_fp32(0x1eeeu) + fp16_to_fp32(0xc931u));
            return static_cast<float>(value);
        }
    }
    return std::numeric_limits<float>::quiet_NaN();
}

} // namespace Laplace
