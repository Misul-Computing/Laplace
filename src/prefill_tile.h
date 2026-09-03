// prefill_tile.h - shared declared capacity for the tiled prefill path.
#pragma once

#include <cstdint>

// Row tile width of the tiled prefill projection kernel (prefill_f16_tile in
// metal_library_sources.inc). One batched prefill transaction evaluates up to
// this many prompt rows. Every admission layer (plan pattern, product Metal
// contract, canonical driver, session begin) references this single declared
// fact instead of hardcoding a batch width.
inline constexpr uint32_t kPrefillTileRows = 32;

// The tiled kernel stages activations and weights as 128-bit vectors, so the
// shared reduction width K must be a multiple of this value. Widths that do
// not satisfy it fail closed to the scalar per-row path.
inline constexpr uint32_t kPrefillTileKMultiple = 8;
