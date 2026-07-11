// tensor.h - tensor view backed by a memory-mapped GGUF file
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace Laplace {

// GGML tensor types. Only the ones we actually decode.
enum class GGMLType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    BF16    = 30,
    I8      = 31,
    I32     = 32,
    I64     = 33,
    U8      = 34,
    U32     = 35,
    BOOL    = 36,
    MLX_AFFINE = 100,
};

// Block / super-block element counts
inline constexpr int QK_LEGACY = 32;   // Q4_0/1, Q5_0/1, Q8_0/1
inline constexpr int QK_KQUANT = 256;  // Q2_K..Q8_K

// Bytes per quantization block (excluding alignment padding)
inline constexpr size_t bytes_per_block(GGMLType t) {
    switch (t) {
        case GGMLType::F32:  return 4;
        case GGMLType::F16:  return 2;
        case GGMLType::BF16: return 2;
        case GGMLType::I8:   return 1;
        case GGMLType::I32:  return 4;
        case GGMLType::I64:  return 8;
        case GGMLType::U8:   return 1;
        case GGMLType::U32:  return 4;
        case GGMLType::BOOL: return 1;
        case GGMLType::Q4_0: return 2 + 16;                  // fp16 d + 32 * 4-bit
        case GGMLType::Q4_1: return 2 + 2 + 16;              // d, m + 32 * 4-bit
        case GGMLType::Q5_0: return 2 + 4 + 16;              // d, qh + 32 * (4+1)-bit
        case GGMLType::Q5_1: return 2 + 2 + 4 + 16;          // d, m, qh + 32 * (4+1)-bit
        case GGMLType::Q8_0: return 2 + 32;                  // d + 32 int8
        case GGMLType::Q8_1: return 2 + 2 + 32;              // d, s + 32 int8
        case GGMLType::Q2_K: return 2 + 2 + 16 + 64;         // d, dmin, scales, 256 * 2-bit
        case GGMLType::Q3_K: return 2 + 12 + 64 + 32;        // d, scales, 256 * 3-bit, hmask
        case GGMLType::Q4_K: return 2 + 2 + 12 + 128;        // d, dmin, scales, 256 * 4-bit
        case GGMLType::Q5_K: return 2 + 2 + 12 + 32 + 128;   // d, dmin, scales, qh, 256 * (4+1)-bit
        case GGMLType::Q6_K: return 2 + 16 + 128 + 64;       // d, scales, ql, qh
        case GGMLType::Q8_K: return 2 + 256 + 64 + 32;       // d, qs, bsums, scales (placeholder)
        case GGMLType::MLX_AFFINE: return 4;                  // packed uint32 (32/bits elements)
        default: return 0;
    }
}

inline constexpr int elements_per_block(GGMLType t) {
    switch (t) {
        case GGMLType::F32:
        case GGMLType::F16:
        case GGMLType::BF16:
        case GGMLType::I8:
        case GGMLType::I32:
        case GGMLType::I64:
        case GGMLType::U8:
        case GGMLType::U32:
        case GGMLType::BOOL:
            return 1;
        case GGMLType::Q4_0:
        case GGMLType::Q4_1:
        case GGMLType::Q5_0:
        case GGMLType::Q5_1:
        case GGMLType::Q8_0:
        case GGMLType::Q8_1:
            return QK_LEGACY;
        case GGMLType::Q2_K:
        case GGMLType::Q3_K:
        case GGMLType::Q4_K:
        case GGMLType::Q5_K:
        case GGMLType::Q6_K:
        case GGMLType::Q8_K:
            return QK_KQUANT;
        case GGMLType::MLX_AFFINE:
            return 1;  // variable, handled by mlx_bits/mlx_group_size
        default:
            return 0;
    }
}

inline const char* type_name(GGMLType t) {
    switch (t) {
        case GGMLType::F32:  return "F32";
        case GGMLType::F16:  return "F16";
        case GGMLType::BF16: return "BF16";
        case GGMLType::I8:   return "I8";
        case GGMLType::I32:  return "I32";
        case GGMLType::I64:  return "I64";
        case GGMLType::U8:   return "U8";
        case GGMLType::U32:  return "U32";
        case GGMLType::BOOL: return "BOOL";
        case GGMLType::Q4_0: return "Q4_0";
        case GGMLType::Q4_1: return "Q4_1";
        case GGMLType::Q5_0: return "Q5_0";
        case GGMLType::Q5_1: return "Q5_1";
        case GGMLType::Q8_0: return "Q8_0";
        case GGMLType::Q8_1: return "Q8_1";
        case GGMLType::Q2_K: return "Q2_K";
        case GGMLType::Q3_K: return "Q3_K";
        case GGMLType::Q4_K: return "Q4_K";
        case GGMLType::Q5_K: return "Q5_K";
        case GGMLType::Q6_K: return "Q6_K";
        case GGMLType::Q8_K: return "Q8_K";
        case GGMLType::MLX_AFFINE: return "MLX_AFFINE";
        default: return "??";
    }
}

struct Tensor {
    std::string name;
    GGMLType type = GGMLType::F32;
    uint32_t n_dims = 0;
    uint64_t dims[4] = {0, 0, 0, 0};
    const uint8_t* data = nullptr;
    const uint8_t* scales = nullptr;
    const uint8_t* biases = nullptr;
    int mlx_bits = 0;
    int mlx_group_size = 0;

    uint64_t n_elements() const {
        uint64_t n = 1;
        for (uint32_t i = 0; i < n_dims; i++) {
            if (dims[i] == 0) return 0;
            n *= dims[i];
        }
        return n;
    }

    size_t nbytes() const {
        size_t bpb = bytes_per_block(type);
        if (bpb == 0) return 0;
        int epb = elements_per_block(type);
        if (epb == 0) return 0;
        size_t nblocks = (static_cast<size_t>(n_elements()) + epb - 1) / epb;
        return nblocks * bpb;
    }

    // Indexed accessors (reversed: dims[0] is innermost, like GGUF)
    int64_t ne(int i) const {
        return static_cast<int>(n_dims) > i ? static_cast<int64_t>(dims[i]) : 1;
    }
};

} // namespace Laplace
