// fp16.h - IEEE 754 binary16 <-> binary32 conversions
#pragma once

#include <cstdint>
#include <cstring>

namespace Laplace {

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            // Subnormal: normalize.
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp  = static_cast<uint32_t>(static_cast<int32_t>(exp) - 1);
            }
            exp  = static_cast<uint32_t>(static_cast<int32_t>(exp) + 1);
            mant &= 0x3ffu;
            f = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        // Inf / NaN
        f = (sign << 31) | (0xffu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }

    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

inline uint16_t fp32_to_fp16(float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant = (mant | 0x800000u) >> (1 - exp);
        if (mant & 0x00001000u) mant += 0x00002000u;  // round
        return static_cast<uint16_t>(sign | (mant >> 13));
    }
    if (exp == 0xff - 127 + 15) {
        if (mant == 0) return static_cast<uint16_t>(sign | 0x7c00);
        return static_cast<uint16_t>(sign | 0x7c00 | (mant >> 13));
    }
    if (exp > 30) {
        return static_cast<uint16_t>(sign | 0x7c00);  // overflow -> Inf
    }
    if (mant & 0x00001000u) {
        mant += 0x00002000u;
        if (mant & 0x00800000u) {
            mant = 0;
            exp += 1;
            if (exp > 30) return static_cast<uint16_t>(sign | 0x7c00);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

inline float bf16_to_fp32(uint16_t b) {
    uint32_t f = static_cast<uint32_t>(b) << 16;
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

inline uint16_t fp32_to_bf16(float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    // round-to-nearest-even: add 0x7FFF + (bits>>16 & 1) before truncating
    uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    bits += rounding;
    return static_cast<uint16_t>(bits >> 16);
}

} // namespace Laplace
