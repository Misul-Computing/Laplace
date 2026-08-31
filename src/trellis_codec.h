// Generic physical 16x16 K-bit trellis codec.
//
// The packing/state convention is adapted from PonyExl3 commit
// 8e7fa6b1556f59fc669e25087903b279b9b0346f (Apache-2.0, see LICENSE in the
// pinned source tree). This file intentionally contains no model, artifact,
// importer, or family selection.
// PonyExl3 copyright 2026 Theinruj Toranavikrai.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace Laplace {

constexpr size_t TrellisTileValueCount = 16u * 16u;

enum class TrellisTileLayout : uint8_t {
    TensorCore16x16 = 1,
};

enum class TrellisCodebook : uint8_t {
    Default = 0,
    Mcg = 1,
    Mul1 = 2,
};

// Runtime facts used only to decide whether the physical kernel can launch.
// They are deliberately excluded from the serialized/immutable byte identity.
struct TrellisKernelFacts {
    uint32_t thread_execution_width = 0;
    uint32_t max_threads_per_threadgroup = 0;
};

// Physical facts required to select and validate the bounded primitive.
// `logical_k` and `logical_n` are element dimensions, not model metadata.
struct TrellisPhysicalDescriptor {
    uint16_t version = 1;
    TrellisTileLayout layout = TrellisTileLayout::TensorCore16x16;
    uint8_t bits = 0;
    uint8_t plane_count = 1;
    TrellisCodebook codebook = TrellisCodebook::Default;
    uint32_t logical_k = 0;
    uint32_t logical_n = 0;
    uint32_t tile_count = 0;
    uint32_t packed_bytes_per_tile = 0;
    uint32_t tile_stride_bytes = 0;
    uint64_t packed_plane_bytes = 0;
};

size_t trellis_packed_bytes_per_tile(uint8_t bits);

bool trellis_select_descriptor(TrellisTileLayout layout, uint8_t plane_count,
                               uint8_t bits, uint32_t logical_k, uint32_t logical_n,
                               TrellisCodebook codebook, TrellisKernelFacts kernel,
                               TrellisPhysicalDescriptor* out);
bool trellis_validate_descriptor(const TrellisPhysicalDescriptor& descriptor);

// The encoded values contain only fresh low-K bits. The packed bytes follow
// the EXL3 16-span stream and uint32 SWAP16 convention.
bool trellis_pack_tile(std::span<const uint16_t> encoded,
                       std::span<uint8_t> packed, uint8_t bits);
bool trellis_unpack_tile(std::span<const uint8_t> packed,
                         std::span<uint16_t> states, uint8_t bits);

// Independent scalar oracle for the procedural 3INST codebooks used by the
// physical decode test. Returned values are finite for every 16-bit input.
float trellis_decode_codeword(uint16_t codeword, TrellisCodebook codebook);

} // namespace Laplace
