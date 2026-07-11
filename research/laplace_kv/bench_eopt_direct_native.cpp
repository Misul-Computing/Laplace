// Standalone direct-factor residual attention benchmark for Apple Silicon.
// Build: clang++ -O3 -std=c++20 -mcpu=native -Wall -Wextra -Werror \
//   -Isrc -o /tmp/bench-eopt-direct \
//   research/laplace_kv/bench_eopt_direct_native.cpp \
//   src/laplace_kv.cpp src/ops.cpp -framework Accelerate

#include "../../src/fp16.h"
#include "../../src/laplace_kv.h"

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_MATMUL_INT8)
#error "This benchmark requires AArch64 I8MM"
#endif

namespace {

constexpr int kTokens = 128;
constexpr int kCodebookSize = 16;
constexpr int kHeaderBytes = 64;
constexpr int kAlignment = 64;

struct Rng {
    uint32_t state;
    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
    float unit() { return (next() >> 8) * (1.0f / 16777216.0f); }
};

size_t align64(size_t bytes) {
    return (bytes + kAlignment - 1) & ~(static_cast<size_t>(kAlignment) - 1);
}

struct RecordSize {
    size_t header = kHeaderBytes;
    size_t residuals = 0;
    size_t norms = 0;
    size_t factor_codes = 0;
    size_t factor_metadata = 0;
    size_t total = 0;
    double bits_per_scalar = 0.0;
};

RecordSize record_size(int dimension, int rank) {
    RecordSize result;
    const size_t residual = static_cast<size_t>(kTokens) * dimension / 4;
    const size_t norms = static_cast<size_t>(kTokens) * sizeof(uint16_t);
    const size_t u_codes = static_cast<size_t>(kTokens) * rank / 2;
    const size_t b_codes = (static_cast<size_t>(rank) * dimension + 1) / 2;
    result.residuals = 2 * align64(residual);
    result.norms = 2 * align64(norms);
    result.factor_codes = 2 * align64(u_codes) + 2 * align64(b_codes);
    // K and V each store r FP16 singular values. Four 16-entry FP16
    // codebooks cover K-U, K-B, V-U, and V-B.
    result.factor_metadata = align64(
        static_cast<size_t>(2 * rank + 4 * kCodebookSize)
        * sizeof(uint16_t));
    result.total = result.header + result.residuals + result.norms
                 + result.factor_codes + result.factor_metadata;
    result.bits_per_scalar = result.total * 8.0
        / (2.0 * kTokens * dimension);
    return result;
}

int maximum_rank(int dimension) {
    int rank = 0;
    while (record_size(dimension, rank + 1).bits_per_scalar <= 3.0) rank++;
    return rank;
}

uint8_t packed2(const std::vector<uint8_t>& values, size_t index) {
    return static_cast<uint8_t>((values[index / 4] >> ((index & 3) * 2)) & 3);
}

uint8_t packed4(const std::vector<uint8_t>& values, size_t index) {
    return static_cast<uint8_t>((values[index / 2] >> ((index & 1) * 4)) & 15);
}

void set2(std::vector<uint8_t>& values, size_t index, uint8_t code) {
    values[index / 4] |= static_cast<uint8_t>(code << ((index & 3) * 2));
}

void set4(std::vector<uint8_t>& values, size_t index, uint8_t code) {
    values[index / 2] |= static_cast<uint8_t>(code << ((index & 1) * 4));
}

struct Tile {
    int dimension = 0;
    int rank = 0;
    std::vector<uint8_t> key_residual;
    std::vector<uint8_t> value_residual;
    std::array<uint16_t, kTokens> key_norm{};
    std::array<uint16_t, kTokens> value_norm{};
    std::vector<uint8_t> key_u;
    std::vector<uint8_t> key_b;
    std::vector<uint8_t> value_u;
    std::vector<uint8_t> value_b;
    std::vector<uint16_t> key_singular;
    std::vector<uint16_t> value_singular;
    std::array<uint16_t, kCodebookSize> key_u_codebook{};
    std::array<uint16_t, kCodebookSize> key_b_codebook{};
    std::array<uint16_t, kCodebookSize> value_u_codebook{};
    std::array<uint16_t, kCodebookSize> value_b_codebook{};
};

float codebook_value(int index, float amplitude) {
    return amplitude * (2.0f * index - 15.0f) / 15.0f;
}

Tile make_tile(int dimension, int rank, Rng& random) {
    Tile tile;
    tile.dimension = dimension;
    tile.rank = rank;
    tile.key_residual.assign(static_cast<size_t>(kTokens) * dimension / 4, 0);
    tile.value_residual.assign(static_cast<size_t>(kTokens) * dimension / 4, 0);
    tile.key_u.assign(static_cast<size_t>(kTokens) * rank / 2, 0);
    tile.key_b.assign((static_cast<size_t>(rank) * dimension + 1) / 2, 0);
    tile.value_u.assign(static_cast<size_t>(kTokens) * rank / 2, 0);
    tile.value_b.assign((static_cast<size_t>(rank) * dimension + 1) / 2, 0);
    tile.key_singular.resize(rank);
    tile.value_singular.resize(rank);
    for (int code = 0; code < kCodebookSize; code++) {
        tile.key_u_codebook[code] = Laplace::fp32_to_fp16(
            codebook_value(code, 0.35f));
        tile.key_b_codebook[code] = Laplace::fp32_to_fp16(
            codebook_value(code, 0.35f));
        tile.value_u_codebook[code] = Laplace::fp32_to_fp16(
            codebook_value(code, 0.35f));
        tile.value_b_codebook[code] = Laplace::fp32_to_fp16(
            codebook_value(code, 0.35f));
    }
    for (int token = 0; token < kTokens; token++) {
        tile.key_norm[token] = Laplace::fp32_to_fp16(
            0.035f + 0.025f * random.unit());
        tile.value_norm[token] = Laplace::fp32_to_fp16(
            0.035f + 0.025f * random.unit());
        for (int dim = 0; dim < dimension; dim++) {
            set2(tile.key_residual,
                 static_cast<size_t>(token) * dimension + dim,
                 random.next() & 3);
            set2(tile.value_residual,
                 static_cast<size_t>(dim) * kTokens + token,
                 random.next() & 3);
        }
        for (int component = 0; component < rank; component++) {
            set4(tile.key_u,
                 static_cast<size_t>(token) * rank + component,
                 random.next() & 15);
            set4(tile.value_u,
                 static_cast<size_t>(token) * rank + component,
                 random.next() & 15);
        }
    }
    for (int component = 0; component < rank; component++) {
        tile.key_singular[component] = Laplace::fp32_to_fp16(
            0.35f / (1.0f + component));
        tile.value_singular[component] = Laplace::fp32_to_fp16(
            0.35f / (1.0f + component));
        for (int dim = 0; dim < dimension; dim++) {
            set4(tile.key_b,
                 static_cast<size_t>(component) * dimension + dim,
                 random.next() & 15);
            set4(tile.value_b,
                 static_cast<size_t>(component) * dimension + dim,
                 random.next() & 15);
        }
    }
    return tile;
}

struct Scratch {
    alignas(16) std::array<float, kTokens> scores{};
    alignas(16) std::array<float, kTokens> weights{};
    alignas(16) std::array<uint8_t, kTokens> weights_q8{};
    alignas(16) std::array<int8_t, 512> query_q8{};
    std::vector<float> local;
    std::vector<float> key_projection;
    std::vector<float> value_projection;
    explicit Scratch(int dimension, int rank)
        : local(dimension), key_projection(rank), value_projection(rank) {}
};

float quantize_query(const float* input, int size, int8_t* output) {
    float32x4_t maximum = vdupq_n_f32(0.0f);
    for (int index = 0; index < size; index += 4) {
        maximum = vmaxq_f32(maximum, vabsq_f32(vld1q_f32(input + index)));
    }
    float scale = vmaxvq_f32(maximum) / 127.0f;
    if (!(scale > 0.0f)) scale = 1.0f;
    float32x4_t inverse = vdupq_n_f32(1.0f / scale);
    for (int index = 0; index < size; index += 16) {
        int32x4_t q0 = vcvtnq_s32_f32(vmulq_f32(
            vld1q_f32(input + index), inverse));
        int32x4_t q1 = vcvtnq_s32_f32(vmulq_f32(
            vld1q_f32(input + index + 4), inverse));
        int32x4_t q2 = vcvtnq_s32_f32(vmulq_f32(
            vld1q_f32(input + index + 8), inverse));
        int32x4_t q3 = vcvtnq_s32_f32(vmulq_f32(
            vld1q_f32(input + index + 12), inverse));
        int16x8_t q01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        int16x8_t q23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
        vst1q_s8(output + index,
                 vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23)));
    }
    return scale;
}

float quantize_weights(const float* input, uint8_t* output) {
    float32x4_t maximum = vdupq_n_f32(0.0f);
    for (int index = 0; index < kTokens; index += 4) {
        maximum = vmaxq_f32(maximum, vld1q_f32(input + index));
    }
    float scale = vmaxvq_f32(maximum) / 255.0f;
    if (!(scale > 0.0f)) scale = 1.0f;
    float32x4_t inverse = vdupq_n_f32(1.0f / scale);
    for (int index = 0; index < kTokens; index += 16) {
        uint32x4_t q0 = vcvtnq_u32_f32(vmulq_f32(
            vld1q_f32(input + index), inverse));
        uint32x4_t q1 = vcvtnq_u32_f32(vmulq_f32(
            vld1q_f32(input + index + 4), inverse));
        uint32x4_t q2 = vcvtnq_u32_f32(vmulq_f32(
            vld1q_f32(input + index + 8), inverse));
        uint32x4_t q3 = vcvtnq_u32_f32(vmulq_f32(
            vld1q_f32(input + index + 12), inverse));
        uint16x8_t q01 = vcombine_u16(vqmovn_u32(q0), vqmovn_u32(q1));
        uint16x8_t q23 = vcombine_u16(vqmovn_u32(q2), vqmovn_u32(q3));
        vst1q_u8(output + index,
                 vcombine_u8(vqmovn_u16(q01), vqmovn_u16(q23)));
    }
    return scale;
}

void unpack_2bit_64(const uint8_t* packed, int8x16_t output[4]) {
    const uint8x16_t source = vld1q_u8(packed);
    const uint8x16_t mask = vdupq_n_u8(3);
    const uint8x16_t c0 = vandq_u8(source, mask);
    const uint8x16_t c1 = vandq_u8(vshrq_n_u8(source, 2), mask);
    const uint8x16_t c2 = vandq_u8(vshrq_n_u8(source, 4), mask);
    const uint8x16_t c3 = vshrq_n_u8(source, 6);
    const uint8x16_t a0 = vzip1q_u8(c0, c2);
    const uint8x16_t b0 = vzip1q_u8(c1, c3);
    const uint8x16_t a1 = vzip2q_u8(c0, c2);
    const uint8x16_t b1 = vzip2q_u8(c1, c3);
    const int8x16_t offset = vdupq_n_s8(-3);
    output[0] = vaddq_s8(offset, vreinterpretq_s8_u8(
        vshlq_n_u8(vzip1q_u8(a0, b0), 1)));
    output[1] = vaddq_s8(offset, vreinterpretq_s8_u8(
        vshlq_n_u8(vzip2q_u8(a0, b0), 1)));
    output[2] = vaddq_s8(offset, vreinterpretq_s8_u8(
        vshlq_n_u8(vzip1q_u8(a1, b1), 1)));
    output[3] = vaddq_s8(offset, vreinterpretq_s8_u8(
        vshlq_n_u8(vzip2q_u8(a1, b1), 1)));
}

void unpack_2bit_32(const uint8_t* packed, int8x16_t output[2]) {
    const uint8x16_t source = vcombine_u8(
        vld1_u8(packed), vdup_n_u8(0));
    const uint8x16_t mask = vdupq_n_u8(3);
    const uint8x16_t c0 = vandq_u8(source, mask);
    const uint8x16_t c1 = vandq_u8(vshrq_n_u8(source, 2), mask);
    const uint8x16_t c2 = vandq_u8(vshrq_n_u8(source, 4), mask);
    const uint8x16_t c3 = vshrq_n_u8(source, 6);
    const uint8x16_t a = vzip1q_u8(c0, c2);
    const uint8x16_t b = vzip1q_u8(c1, c3);
    const int8x16_t offset = vdupq_n_s8(-3);
    output[0] = vaddq_s8(offset, vreinterpretq_s8_u8(
        vshlq_n_u8(vzip1q_u8(a, b), 1)));
    output[1] = vaddq_s8(offset, vreinterpretq_s8_u8(
        vshlq_n_u8(vzip2q_u8(a, b), 1)));
}

float decode4(const std::vector<uint8_t>& codes, size_t index,
              const float* codebook) {
    return codebook[packed4(codes, index)];
}

void decode_codebook(const std::array<uint16_t, kCodebookSize>& stored,
                     float* output) {
    for (int index = 0; index < kCodebookSize; index++) {
        output[index] = Laplace::fp16_to_fp32(stored[index]);
    }
}

void key_scores(const Tile& tile, const float* query,
                const int8_t* query_q8, float query_scale,
                float logit_scale, float* scores, Scratch& scratch) {
    alignas(16) float u_codebook[kCodebookSize];
    alignas(16) float b_codebook[kCodebookSize];
    decode_codebook(tile.key_u_codebook, u_codebook);
    decode_codebook(tile.key_b_codebook, b_codebook);
    for (int component = 0; component < tile.rank; component++) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        const size_t base = static_cast<size_t>(component) * tile.dimension;
        for (int dim = 0; dim < tile.dimension; dim += 4) {
            alignas(16) float decoded[4];
            for (int lane = 0; lane < 4; lane++) {
                decoded[lane] = decode4(
                    tile.key_b, base + dim + lane, b_codebook);
            }
            sum = vfmaq_f32(sum, vld1q_f32(query + dim),
                            vld1q_f32(decoded));
        }
        scratch.key_projection[component] = vaddvq_f32(sum)
            * Laplace::fp16_to_fp32(tile.key_singular[component]);
    }
    for (int token = 0; token < kTokens; token++) {
        const uint8_t* packed = tile.key_residual.data()
            + static_cast<size_t>(token) * tile.dimension / 4;
        int32x4_t residual = vdupq_n_s32(0);
        int dim = 0;
        for (; dim + 64 <= tile.dimension; dim += 64) {
            int8x16_t codes[4];
            unpack_2bit_64(packed + dim / 4, codes);
            residual = vdotq_s32(residual, codes[0],
                                 vld1q_s8(query_q8 + dim));
            residual = vdotq_s32(residual, codes[1],
                                 vld1q_s8(query_q8 + dim + 16));
            residual = vdotq_s32(residual, codes[2],
                                 vld1q_s8(query_q8 + dim + 32));
            residual = vdotq_s32(residual, codes[3],
                                 vld1q_s8(query_q8 + dim + 48));
        }
        if (dim < tile.dimension) {
            int8x16_t codes[2];
            unpack_2bit_32(packed + dim / 4, codes);
            residual = vdotq_s32(residual, codes[0],
                                 vld1q_s8(query_q8 + dim));
            residual = vdotq_s32(residual, codes[1],
                                 vld1q_s8(query_q8 + dim + 16));
        }
        float factor = 0.0f;
        const size_t base = static_cast<size_t>(token) * tile.rank;
        for (int component = 0; component < tile.rank; component++) {
            factor += decode4(tile.key_u, base + component, u_codebook)
                    * scratch.key_projection[component];
        }
        const float norm = Laplace::fp16_to_fp32(tile.key_norm[token]);
        scores[token] = (norm * vaddvq_s32(residual)
            * (query_scale / 3.0f) + factor) * logit_scale;
    }
}

void add_values(const Tile& tile, const float* weights,
                float* output, Scratch& scratch) {
    alignas(16) float u_codebook[kCodebookSize];
    alignas(16) float b_codebook[kCodebookSize];
    decode_codebook(tile.value_u_codebook, u_codebook);
    decode_codebook(tile.value_b_codebook, b_codebook);
    for (int token = 0; token < kTokens; token++) {
        scratch.scores[token] = weights[token]
            * Laplace::fp16_to_fp32(tile.value_norm[token]);
    }
    const float weight_scale = quantize_weights(
        scratch.scores.data(), scratch.weights_q8.data());
    for (int component = 0; component < tile.rank; component++) {
        float sum = 0.0f;
        for (int token = 0; token < kTokens; token++) {
            sum += weights[token] * decode4(
                tile.value_u,
                static_cast<size_t>(token) * tile.rank + component,
                u_codebook);
        }
        scratch.value_projection[component] = sum
            * Laplace::fp16_to_fp32(tile.value_singular[component]);
    }
    for (int dim = 0; dim < tile.dimension; dim++) {
        const uint8_t* packed = tile.value_residual.data()
            + static_cast<size_t>(dim) * kTokens / 4;
        int32x4_t residual = vdupq_n_s32(0);
        for (int token = 0; token < kTokens; token += 64) {
            int8x16_t codes[4];
            unpack_2bit_64(packed + token / 4, codes);
            residual = vusdotq_s32(residual,
                vld1q_u8(scratch.weights_q8.data() + token), codes[0]);
            residual = vusdotq_s32(residual,
                vld1q_u8(scratch.weights_q8.data() + token + 16), codes[1]);
            residual = vusdotq_s32(residual,
                vld1q_u8(scratch.weights_q8.data() + token + 32), codes[2]);
            residual = vusdotq_s32(residual,
                vld1q_u8(scratch.weights_q8.data() + token + 48), codes[3]);
        }
        float factor = 0.0f;
        for (int component = 0; component < tile.rank; component++) {
            factor += scratch.value_projection[component] * decode4(
                tile.value_b,
                static_cast<size_t>(component) * tile.dimension + dim,
                b_codebook);
        }
        output[dim] = vaddvq_s32(residual) * (weight_scale / 3.0f) + factor;
    }
}

void merge(float local_max, float local_sum, const float* local,
           int dimension, bool& initialized, float& global_max,
           float& global_sum, float* output) {
    if (!initialized) {
        std::copy_n(local, dimension, output);
        global_max = local_max;
        global_sum = local_sum;
        initialized = true;
        return;
    }
    if (local_max <= global_max) {
        const float factor = std::exp(local_max - global_max);
        for (int dim = 0; dim < dimension; dim++) output[dim] += factor * local[dim];
        global_sum += factor * local_sum;
    } else {
        const float factor = std::exp(global_max - local_max);
        for (int dim = 0; dim < dimension; dim++) {
            output[dim] = factor * output[dim] + local[dim];
        }
        global_sum = factor * global_sum + local_sum;
        global_max = local_max;
    }
}

void direct_attention(const std::vector<Tile>& tiles, const float* query,
                      int dimension, int rank, float* output,
                      Scratch& scratch) {
    const float query_scale = quantize_query(
        query, dimension, scratch.query_q8.data());
    const float logit_scale = 1.0f / std::sqrt(static_cast<float>(dimension));
    bool initialized = false;
    float global_max = 0.0f;
    float global_sum = 0.0f;
    for (const Tile& tile : tiles) {
        if (tile.rank != rank) std::abort();
        key_scores(tile, query, scratch.query_q8.data(), query_scale,
                   logit_scale, scratch.scores.data(), scratch);
        const float local_max = *std::max_element(
            scratch.scores.begin(), scratch.scores.end());
        for (int token = 0; token < kTokens; token++) {
            scratch.weights[token] = scratch.scores[token] - local_max;
        }
        int count = kTokens;
        vvexpf(scratch.weights.data(), scratch.weights.data(), &count);
        float local_sum = 0.0f;
        for (float weight : scratch.weights) local_sum += weight;
        add_values(tile, scratch.weights.data(), scratch.local.data(), scratch);
        merge(local_max, local_sum, scratch.local.data(), dimension,
              initialized, global_max, global_sum, output);
    }
    const float inverse = 1.0f / global_sum;
    for (int dim = 0; dim < dimension; dim++) output[dim] *= inverse;
}

float dense_key(const Tile& tile, int token, int dim) {
    alignas(16) float u_codebook[kCodebookSize];
    alignas(16) float b_codebook[kCodebookSize];
    decode_codebook(tile.key_u_codebook, u_codebook);
    decode_codebook(tile.key_b_codebook, b_codebook);
    const float residual_code = (2.0f * packed2(tile.key_residual,
        static_cast<size_t>(token) * tile.dimension + dim) - 3.0f) / 3.0f;
    const float residual = residual_code * Laplace::fp16_to_fp32(
        tile.key_norm[token]);
    float factor = 0.0f;
    for (int component = 0; component < tile.rank; component++) {
        factor += decode4(tile.key_u,
            static_cast<size_t>(token) * tile.rank + component, u_codebook)
            * Laplace::fp16_to_fp32(tile.key_singular[component])
            * decode4(tile.key_b,
                static_cast<size_t>(component) * tile.dimension + dim,
                b_codebook);
    }
    return residual + factor;
}

float dense_value(const Tile& tile, int token, int dim) {
    alignas(16) float u_codebook[kCodebookSize];
    alignas(16) float b_codebook[kCodebookSize];
    decode_codebook(tile.value_u_codebook, u_codebook);
    decode_codebook(tile.value_b_codebook, b_codebook);
    const float residual = (2.0f * packed2(tile.value_residual,
        static_cast<size_t>(dim) * kTokens + token) - 3.0f) / 3.0f;
    float factor = 0.0f;
    for (int component = 0; component < tile.rank; component++) {
        factor += decode4(tile.value_u,
            static_cast<size_t>(token) * tile.rank + component, u_codebook)
            * Laplace::fp16_to_fp32(tile.value_singular[component])
            * decode4(tile.value_b,
                static_cast<size_t>(component) * tile.dimension + dim,
                b_codebook);
    }
    return residual * Laplace::fp16_to_fp32(tile.value_norm[token]) + factor;
}

void materialize(const std::vector<Tile>& tiles, int dimension,
                 std::vector<uint16_t>& keys, std::vector<uint16_t>& values) {
    const int context = static_cast<int>(tiles.size()) * kTokens;
    keys.resize(static_cast<size_t>(context) * dimension);
    values.resize(static_cast<size_t>(context) * dimension);
    for (size_t tile_index = 0; tile_index < tiles.size(); tile_index++) {
        const Tile& tile = tiles[tile_index];
        for (int token = 0; token < kTokens; token++) {
            const size_t row = (tile_index * kTokens + token) * dimension;
            for (int dim = 0; dim < dimension; dim++) {
                keys[row + dim] = Laplace::fp32_to_fp16(
                    dense_key(tile, token, dim));
                values[row + dim] = Laplace::fp32_to_fp16(
                    dense_value(tile, token, dim));
            }
        }
    }
}

void fp16_attention(const std::vector<uint16_t>& keys,
                    const std::vector<uint16_t>& values,
                    const float* query, int context, int dimension,
                    float* output, Scratch& scratch) {
    bool initialized = false;
    float global_max = 0.0f;
    float global_sum = 0.0f;
    const float logit_scale = 1.0f / std::sqrt(static_cast<float>(dimension));
    for (int first = 0; first < context; first += kTokens) {
        for (int token = 0; token < kTokens; token++) {
            const uint16_t* key = keys.data()
                + static_cast<size_t>(first + token) * dimension;
            float32x4_t sum = vdupq_n_f32(0.0f);
            for (int dim = 0; dim < dimension; dim += 4) {
                const float32x4_t value = vcvt_f32_f16(vld1_f16(
                    reinterpret_cast<const __fp16*>(key + dim)));
                sum = vfmaq_f32(sum, vld1q_f32(query + dim), value);
            }
            scratch.scores[token] = vaddvq_f32(sum) * logit_scale;
        }
        const float local_max = *std::max_element(
            scratch.scores.begin(), scratch.scores.end());
        for (int token = 0; token < kTokens; token++) {
            scratch.weights[token] = scratch.scores[token] - local_max;
        }
        int count = kTokens;
        vvexpf(scratch.weights.data(), scratch.weights.data(), &count);
        float local_sum = 0.0f;
        for (float weight : scratch.weights) local_sum += weight;
        std::fill(scratch.local.begin(), scratch.local.end(), 0.0f);
        for (int token = 0; token < kTokens; token++) {
            const float32x4_t weight = vdupq_n_f32(scratch.weights[token]);
            const uint16_t* value = values.data()
                + static_cast<size_t>(first + token) * dimension;
            for (int dim = 0; dim < dimension; dim += 4) {
                const float32x4_t decoded = vcvt_f32_f16(vld1_f16(
                    reinterpret_cast<const __fp16*>(value + dim)));
                const float32x4_t current = vld1q_f32(
                    scratch.local.data() + dim);
                vst1q_f32(scratch.local.data() + dim,
                    vfmaq_f32(current, weight, decoded));
            }
        }
        merge(local_max, local_sum, scratch.local.data(), dimension,
              initialized, global_max, global_sum, output);
    }
    const float inverse = 1.0f / global_sum;
    for (int dim = 0; dim < dimension; dim++) output[dim] *= inverse;
}

double relative_error(const std::vector<float>& actual,
                      const std::vector<float>& expected) {
    double error = 0.0;
    double norm = 0.0;
    for (size_t index = 0; index < actual.size(); index++) {
        const double delta = actual[index] - expected[index];
        error += delta * delta;
        norm += static_cast<double>(expected[index]) * expected[index];
    }
    return std::sqrt(error / std::max(norm, 1e-30));
}

volatile float sink = 0.0f;

template <class Function>
double median_ms(Function&& function, int trials) {
    function();
    function();
    std::vector<double> samples;
    for (int trial = 0; trial < trials; trial++) {
        const auto start = std::chrono::steady_clock::now();
        sink = function();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(
            end - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void check_unpack() {
    alignas(16) std::array<uint8_t, 16> packed{};
    for (int index = 0; index < 64; index++) {
        packed[index / 4] |= static_cast<uint8_t>(
            (index & 3) << ((index & 3) * 2));
    }
    int8x16_t decoded[4];
    unpack_2bit_64(packed.data(), decoded);
    alignas(16) std::array<int8_t, 64> values{};
    for (int group = 0; group < 4; group++) {
        vst1q_s8(values.data() + group * 16, decoded[group]);
    }
    for (int index = 0; index < 64; index++) {
        if (values[index] != 2 * (index & 3) - 3) std::abort();
    }
}

void benchmark(int context, int dimension, int trials) {
    const int max_rank = maximum_rank(dimension);
    if (max_rank < 1) std::abort();
    Rng control_random{0x81f34a21u
        + static_cast<uint32_t>(context + 17 * dimension)};
    std::vector<Tile> control_tiles;
    control_tiles.reserve(context / kTokens);
    for (int token = 0; token < context; token += kTokens) {
        control_tiles.push_back(make_tile(
            dimension, max_rank, control_random));
    }
    std::vector<float> query(dimension);
    for (float& value : query) value = 2.0f * control_random.unit() - 1.0f;
    std::vector<uint16_t> fp16_keys;
    std::vector<uint16_t> fp16_values;
    materialize(control_tiles, dimension, fp16_keys, fp16_values);

    Laplace::LaplaceKV control;
    if (!control.init(1, 1, dimension, context, false)) std::abort();
    std::vector<float> row(dimension);
    for (int token = 0; token < context; token++) {
        for (int dim = 0; dim < dimension; dim++) {
            row[dim] = Laplace::fp16_to_fp32(
                fp16_keys[static_cast<size_t>(token) * dimension + dim]);
        }
        control.store_k_wh(0, 0, token, row.data());
        for (int dim = 0; dim < dimension; dim++) {
            row[dim] = Laplace::fp16_to_fp32(
                fp16_values[static_cast<size_t>(token) * dimension + dim]);
        }
        control.store_v_wh(0, 0, token, row.data());
    }

    std::vector<float> fp16_output(dimension);
    std::vector<float> control_output(dimension);
    std::vector<float> direct_max_output(dimension);
    Scratch fp16_scratch(dimension, max_rank);
    Scratch direct_max_scratch(dimension, max_rank);
    auto fp16_call = [&] {
        fp16_attention(fp16_keys, fp16_values, query.data(), context,
                       dimension, fp16_output.data(), fp16_scratch);
        return fp16_output[0];
    };
    auto control_call = [&] {
        control.attention_wh(0, 0, context, query.data(),
            1.0f / std::sqrt(static_cast<float>(dimension)),
            control_output.data());
        return control_output[0];
    };
    auto direct_max_call = [&] {
        direct_attention(control_tiles, query.data(), dimension, max_rank,
                         direct_max_output.data(), direct_max_scratch);
        return direct_max_output[0];
    };
    fp16_call();
    direct_max_call();
    const double direct_max_error = relative_error(
        direct_max_output, fp16_output);
    const double fp16_ms = median_ms(fp16_call, trials);
    const double control_ms = median_ms(control_call, trials);
    const double control_error = relative_error(control_output, fp16_output);

    for (int rank = 1; rank <= max_rank; rank++) {
        Rng random{0x519ceb72u + static_cast<uint32_t>(
            context + 257 * dimension + 65537 * rank)};
        std::vector<Tile> tiles;
        tiles.reserve(context / kTokens);
        for (int token = 0; token < context; token += kTokens) {
            tiles.push_back(make_tile(dimension, rank, random));
        }
        std::vector<float> output(dimension);
        Scratch scratch(dimension, rank);
        auto direct_call = [&] {
            direct_attention(tiles, query.data(), dimension, rank,
                             output.data(), scratch);
            return output[0];
        };
        const double direct_ms = median_ms(direct_call, trials);
        const RecordSize bytes = record_size(dimension, rank);
        std::printf(
            "context=%d D=%d rank=%d tile_bytes=%zu bits=%.6f "
            "header=%zu residual=%zu norms=%zu factor_codes=%zu "
            "factor_metadata=%zu direct_ms=%.6f fp16_ms=%.6f "
            "k8v6_ms=%.6f direct_vs_fp16=%.3fx direct_vs_k8v6=%.3fx "
            "max_rank_direct_fp16_relerr=%.6g k8v6_fp16_relerr=%.6g "
            "synthetic=true\n",
            context, dimension, rank, bytes.total, bytes.bits_per_scalar,
            bytes.header, bytes.residuals, bytes.norms, bytes.factor_codes,
            bytes.factor_metadata, direct_ms, fp16_ms, control_ms,
            fp16_ms / direct_ms, control_ms / direct_ms,
            direct_max_error, control_error);
    }
}

} // namespace

int main(int argc, char** argv) {
    const int context = argc > 1 ? std::atoi(argv[1]) : 16384;
    const int trials = argc > 2 ? std::atoi(argv[2]) : 9;
    if (context < kTokens || context % kTokens || trials < 1) {
        std::fprintf(stderr,
            "usage: %s [context_multiple_of_128] [trials]\n", argv[0]);
        return 2;
    }
    std::puts(
        "format=residual2+factor4 tile=128 norms=fp16 singular=fp16 "
        "factor_codebooks=4x16xfp16 header=64 alignment=64 "
        "single_query=true warm_cache=true synthetic=true");
    check_unpack();
    for (int dimension : {64, 96, 128, 256}) {
        benchmark(context, dimension, trials);
    }
    return 0;
}
