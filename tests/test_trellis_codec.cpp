#include "trellis_codec.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace Laplace;

static void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::abort();
    }
}

static void round_trip(uint8_t bits) {
    std::mt19937 rng(1000u + bits);
    const uint16_t mask = static_cast<uint16_t>((1u << bits) - 1u);
    std::array<uint16_t, TrellisTileValueCount> values{};
    for (uint16_t& value : values) value = static_cast<uint16_t>(rng()) & mask;
    std::vector<uint8_t> packed(trellis_packed_bytes_per_tile(bits));
    std::array<uint16_t, TrellisTileValueCount> decoded{};
    check(trellis_pack_tile(values, packed, bits), "pack succeeds");
    check(trellis_unpack_tile(packed, decoded, bits), "unpack succeeds");
    for (size_t i = 0; i != values.size(); ++i)
        check((decoded[i] & mask) == values[i], "fresh bits survive round trip");
    check(trellis_pack_tile(decoded, packed, bits), "repack succeeds");
    std::vector<uint8_t> repacked(trellis_packed_bytes_per_tile(bits));
    check(trellis_pack_tile(decoded, repacked, bits), "second pack succeeds");
    check(packed == repacked, "full sliding states repack identically");
}

int main() {
    round_trip(1);
    round_trip(2);
    round_trip(3);
    round_trip(4);
    round_trip(5);
    round_trip(6);
    round_trip(7);
    round_trip(8);

    TrellisPhysicalDescriptor descriptor{};
    check(trellis_select_descriptor(TrellisTileLayout::TensorCore16x16, 1, 2,
                                    5120, 17408, TrellisCodebook::Default,
                                    {32, 1024}, &descriptor),
          "descriptor selection succeeds");
    check(descriptor.packed_bytes_per_tile == 64, "K=2 tile has 64 packed bytes");
    check(trellis_validate_descriptor(descriptor), "selected descriptor validates");

    TrellisPhysicalDescriptor malformed = descriptor;
    malformed.logical_k = 5119;
    check(!trellis_validate_descriptor(malformed), "non-tile shape rejected");
    malformed = descriptor;
    malformed.plane_count = 2;
    check(!trellis_validate_descriptor(malformed), "multi-plane descriptor rejected");
    malformed = descriptor;
    malformed.packed_bytes_per_tile = 63;
    check(!trellis_validate_descriptor(malformed), "wrong packed size rejected");
    malformed = descriptor;
    malformed.layout = static_cast<TrellisTileLayout>(99);
    check(!trellis_validate_descriptor(malformed), "unknown layout rejected");
    malformed = descriptor;
    malformed.codebook = static_cast<TrellisCodebook>(99);
    check(!trellis_validate_descriptor(malformed), "unknown codebook rejected");

    for (TrellisCodebook codebook : {TrellisCodebook::Default,
                                     TrellisCodebook::Mcg,
                                     TrellisCodebook::Mul1}) {
        const float value = trellis_decode_codeword(0x1234u, codebook);
        check(std::isfinite(value), "scalar codebook decode is finite");
    }
    std::puts("PASS trellis codec k=1..8 descriptor and scalar decoder");
}
