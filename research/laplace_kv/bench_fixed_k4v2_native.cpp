// Standalone fixed-width K4/V2 attention benchmark for Apple Silicon.
// Build: clang++ -O3 -std=c++20 -mcpu=native -Wall -Wextra -Werror \
//   -Isrc -o /tmp/bench-fixed-k4v2 \
//   research/laplace_kv/bench_fixed_k4v2_native.cpp \
//   src/laplace_kv.cpp src/ops.cpp -framework Accelerate

#include "../../src/fp16.h"
#include "../../src/laplace_kv.h"

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_MATMUL_INT8)
#error "This benchmark requires AArch64 I8MM"
#endif

namespace {

constexpr int kTokens = 128;
constexpr int kSyndromeKeyGroups = 1170;
constexpr int kSyndromeValueGroups = 46;
constexpr int kSyndromeBits = 3648;
constexpr int kSyndromeBytes = kSyndromeBits / 8;
static_assert((kSyndromeKeyGroups + kSyndromeValueGroups) * 3
              == kSyndromeBits);

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

struct Tile {
    std::vector<uint8_t> keys;
    std::vector<uint8_t> values;
    std::vector<float> ka;
    std::vector<float> kb;
    std::array<float, kTokens> kc{};
    std::array<float, kTokens> va{};
    std::array<float, kTokens> vb{};
    std::vector<float> vc;
};

struct PreparedKey {
    std::vector<int8_t> query;
    float scale = 1.0f;
    float bias = 0.0f;
};

struct Scratch {
    alignas(16) std::array<float, kTokens> scores{};
    alignas(16) std::array<float, kTokens> weights{};
    alignas(16) std::array<uint8_t, kTokens> weights_q8{};
    alignas(16) std::array<int8_t, 512> query_q8{};
    std::array<uint8_t, 1024> key_lsb{};
    std::array<uint8_t, 41> value_lsb{};
    std::array<uint8_t, kSyndromeKeyGroups + kSyndromeValueGroups> syndromes{};
    std::array<uint8_t, kSyndromeBytes> metadata{};
    std::vector<float> local;
    explicit Scratch(int dimension) : local(dimension) {}
};

void set_k4(std::vector<uint8_t>& packed, int index, uint8_t code);
void set_v2(std::vector<uint8_t>& packed, int index, uint8_t code);

constexpr std::array<uint8_t, 256> make_lsb_table(int bits) {
    std::array<uint8_t, 256> table{};
    for (int value = 0; value < 256; value++) {
        for (int shift = 0; shift < 8; shift += bits) {
            table[value] |= static_cast<uint8_t>(
                ((value >> shift) & 1) << (shift / bits));
        }
    }
    return table;
}

constexpr std::array<uint8_t, 128> make_syndrome_table() {
    std::array<uint8_t, 128> table{};
    for (int pattern = 0; pattern < 128; pattern++) {
        for (int bit = 0; bit < 7; bit++) {
            if (pattern & (1 << bit)) {
                table[pattern] ^= static_cast<uint8_t>(bit + 1);
            }
        }
    }
    return table;
}

constexpr auto kK4Lsb = make_lsb_table(4);
constexpr auto kV2Lsb = make_lsb_table(2);
constexpr auto kHammingSyndrome = make_syndrome_table();

__attribute__((noinline))
uint32_t extract_syndrome_metadata(const Tile& tile, Scratch& scratch) {
    for (int byte = 0; byte < 1024; byte++) {
        const uint8_t* source = tile.keys.data() + byte * 4;
        scratch.key_lsb[byte] = static_cast<uint8_t>(
              kK4Lsb[source[0]]
            | (kK4Lsb[source[1]] << 2)
            | (kK4Lsb[source[2]] << 4)
            | (kK4Lsb[source[3]] << 6));
    }
    for (int byte = 0; byte < 40; byte++) {
        const uint8_t* source = tile.values.data() + byte * 2;
        scratch.value_lsb[byte] = static_cast<uint8_t>(
              kV2Lsb[source[0]]
            | (kV2Lsb[source[1]] << 4));
    }
    scratch.value_lsb[40] = kV2Lsb[tile.values[80]];

    auto extract = [&](const uint8_t* bits, int groups, uint8_t* output) {
        for (int group = 0; group < groups; group++) {
            int bit = group * 7;
            uint16_t window = static_cast<uint16_t>(bits[bit >> 3])
                            | static_cast<uint16_t>(bits[(bit >> 3) + 1]) << 8;
            output[group] = kHammingSyndrome[(window >> (bit & 7)) & 127];
        }
    };
    extract(scratch.key_lsb.data(), kSyndromeKeyGroups,
            scratch.syndromes.data());
    extract(scratch.value_lsb.data(), kSyndromeValueGroups,
            scratch.syndromes.data() + kSyndromeKeyGroups);

    uint32_t checksum = 0;
    for (int group = 0; group < kSyndromeKeyGroups + kSyndromeValueGroups;
         group += 8) {
        uint32_t packed = 0;
        for (int lane = 0; lane < 8; lane++) {
            uint8_t syndrome = scratch.syndromes[group + lane];
            packed |= static_cast<uint32_t>(syndrome) << (lane * 3);
            checksum += syndrome;
        }
        scratch.metadata[group * 3 / 8] = static_cast<uint8_t>(packed);
        scratch.metadata[group * 3 / 8 + 1] = static_cast<uint8_t>(packed >> 8);
        scratch.metadata[group * 3 / 8 + 2] = static_cast<uint8_t>(packed >> 16);
    }
    return checksum;
}

void check_syndrome_extraction() {
    Tile tile;
    tile.keys.assign(kTokens * 64 / 2, 0);
    tile.values.assign(kTokens * 64 / 4, 0);
    set_k4(tile.keys, 0, 1);
    set_k4(tile.keys, 1, 1);
    set_k4(tile.keys, 3, 1);
    set_v2(tile.values, 2, 1);
    Scratch scratch(64);
    uint32_t checksum = extract_syndrome_metadata(tile, scratch);
    if ((scratch.metadata[0] & 7) != 7
        || (scratch.metadata[438] & 0xc0) != 0xc0
        || (scratch.metadata[439] & 1) != 0 || checksum != 10) {
        std::abort();
    }
}

void set_k4(std::vector<uint8_t>& packed, int index, uint8_t code) {
    packed[index / 2] |= static_cast<uint8_t>(code << ((index & 1) * 4));
}

void set_v2(std::vector<uint8_t>& packed, int index, uint8_t code) {
    packed[index / 4] |= static_cast<uint8_t>(code << ((index & 3) * 2));
}

Tile make_tile(int dimension, Rng& random,
               std::vector<uint16_t>& fp16_keys,
               std::vector<uint16_t>& fp16_values) {
    Tile tile;
    tile.keys.assign(static_cast<size_t>(kTokens) * dimension / 2, 0);
    tile.values.assign(static_cast<size_t>(kTokens) * dimension / 4, 0);
    tile.ka.resize(dimension);
    tile.kb.resize(dimension);
    tile.vc.resize(dimension);
    for (int dim = 0; dim < dimension; dim++) {
        tile.ka[dim] = 0.010f + 0.020f * random.unit();
        tile.kb[dim] = -0.150f + 0.100f * random.unit();
        tile.vc[dim] = 0.750f + 0.500f * random.unit();
    }
    for (int token = 0; token < kTokens; token++) {
        tile.kc[token] = 0.750f + 0.500f * random.unit();
        tile.va[token] = 0.025f + 0.040f * random.unit();
        tile.vb[token] = -0.100f + 0.080f * random.unit();
        for (int dim = 0; dim < dimension; dim++) {
            uint8_t key_code = random.next() & 15;
            uint8_t value_code = random.next() & 3;
            set_k4(tile.keys, token * dimension + dim, key_code);
            set_v2(tile.values, dim * kTokens + token, value_code);
            float key = (key_code * tile.ka[dim] + tile.kb[dim])
                      * tile.kc[token];
            float value = (value_code * tile.va[token] + tile.vb[token])
                        * tile.vc[dim];
            fp16_keys.push_back(Laplace::fp32_to_fp16(key));
            fp16_values.push_back(Laplace::fp32_to_fp16(value));
        }
    }
    return tile;
}

float quantize_s8(const float* input, int size, int8_t* output) {
    float32x4_t maximum = vdupq_n_f32(0.0f);
    for (int index = 0; index < size; index += 4) {
        maximum = vmaxq_f32(maximum, vabsq_f32(vld1q_f32(input + index)));
    }
    float scale = vmaxvq_f32(maximum) / 127.0f;
    if (!(scale > 0.0f)) scale = 1.0f;
    float32x4_t inverse = vdupq_n_f32(1.0f / scale);
    for (int index = 0; index < size; index += 16) {
        int32x4_t q0 = vcvtnq_s32_f32(
            vmulq_f32(vld1q_f32(input + index), inverse));
        int32x4_t q1 = vcvtnq_s32_f32(
            vmulq_f32(vld1q_f32(input + index + 4), inverse));
        int32x4_t q2 = vcvtnq_s32_f32(
            vmulq_f32(vld1q_f32(input + index + 8), inverse));
        int32x4_t q3 = vcvtnq_s32_f32(
            vmulq_f32(vld1q_f32(input + index + 12), inverse));
        int16x8_t q01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        int16x8_t q23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
        vst1q_s8(output + index,
                 vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23)));
    }
    return scale;
}

float quantize_u8(const float* input, uint8_t* output) {
    float32x4_t maximum = vdupq_n_f32(0.0f);
    for (int index = 0; index < kTokens; index += 4) {
        maximum = vmaxq_f32(maximum, vld1q_f32(input + index));
    }
    float scale = vmaxvq_f32(maximum) / 255.0f;
    if (!(scale > 0.0f)) scale = 1.0f;
    float32x4_t inverse = vdupq_n_f32(1.0f / scale);
    for (int index = 0; index < kTokens; index += 16) {
        uint32x4_t q0 = vcvtnq_u32_f32(
            vmulq_f32(vld1q_f32(input + index), inverse));
        uint32x4_t q1 = vcvtnq_u32_f32(
            vmulq_f32(vld1q_f32(input + index + 4), inverse));
        uint32x4_t q2 = vcvtnq_u32_f32(
            vmulq_f32(vld1q_f32(input + index + 8), inverse));
        uint32x4_t q3 = vcvtnq_u32_f32(
            vmulq_f32(vld1q_f32(input + index + 12), inverse));
        uint16x8_t q01 = vcombine_u16(vqmovn_u32(q0), vqmovn_u32(q1));
        uint16x8_t q23 = vcombine_u16(vqmovn_u32(q2), vqmovn_u32(q3));
        vst1q_u8(output + index,
                 vcombine_u8(vqmovn_u16(q01), vqmovn_u16(q23)));
    }
    return scale;
}

void prepare_key(const Tile& tile, const float* query, int dimension,
                 int8_t* query_q8, float& query_scale, float& bias) {
    alignas(16) float adjusted[512];
    float32x4_t bias_sum = vdupq_n_f32(0.0f);
    for (int dim = 0; dim < dimension; dim += 4) {
        float32x4_t q = vld1q_f32(query + dim);
        vst1q_f32(adjusted + dim,
                  vmulq_f32(q, vld1q_f32(tile.ka.data() + dim)));
        bias_sum = vfmaq_f32(
            bias_sum, q, vld1q_f32(tile.kb.data() + dim));
    }
    bias = vaddvq_f32(bias_sum);
    query_scale = quantize_s8(adjusted, dimension, query_q8);
}

void dot_keys(const Tile& tile, const int8_t* query_q8,
              float query_scale, float bias, int dimension,
              float logit_scale, float* scores) {
    const uint8x16_t mask = vdupq_n_u8(15);
    for (int token = 0; token < kTokens; token++) {
        const uint8_t* row = tile.keys.data()
                           + static_cast<size_t>(token) * dimension / 2;
        int32x4_t sum = vdupq_n_s32(0);
        for (int dim = 0; dim < dimension; dim += 32) {
            uint8x16_t packed = vld1q_u8(row + dim / 2);
            uint8x16_t low = vandq_u8(packed, mask);
            uint8x16_t high = vshrq_n_u8(packed, 4);
            sum = vusdotq_s32(
                sum, vzip1q_u8(low, high), vld1q_s8(query_q8 + dim));
            sum = vusdotq_s32(
                sum, vzip2q_u8(low, high), vld1q_s8(query_q8 + dim + 16));
        }
        scores[token] = (vaddvq_s32(sum) * query_scale + bias)
                      * tile.kc[token] * logit_scale;
    }
}

void add_values(const Tile& tile, const float* weights, int dimension,
                float* output, Scratch& scratch) {
    float32x4_t bias_sum = vdupq_n_f32(0.0f);
    for (int token = 0; token < kTokens; token += 4) {
        float32x4_t weight = vld1q_f32(weights + token);
        vst1q_f32(scratch.scores.data() + token,
                  vmulq_f32(weight, vld1q_f32(tile.va.data() + token)));
        bias_sum = vfmaq_f32(
            bias_sum, weight, vld1q_f32(tile.vb.data() + token));
    }
    float bias = vaddvq_f32(bias_sum);
    float weight_scale = quantize_u8(
        scratch.scores.data(), scratch.weights_q8.data());

    alignas(16) static constexpr uint8_t indices_data[16] = {
        0, 0, 0, 0, 1, 1, 1, 1,
        2, 2, 2, 2, 3, 3, 3, 3
    };
    alignas(16) static constexpr int8_t shifts_data[16] = {
        0, -2, -4, -6, 0, -2, -4, -6,
        0, -2, -4, -6, 0, -2, -4, -6
    };
    const uint8x16_t base_indices = vld1q_u8(indices_data);
    const int8x16_t shifts = vld1q_s8(shifts_data);
    const uint8x16_t mask = vdupq_n_u8(3);
    for (int dim = 0; dim < dimension; dim++) {
        const uint8_t* packed = tile.values.data()
                              + static_cast<size_t>(dim) * (kTokens / 4);
        int32x4_t sum = vdupq_n_s32(0);
        for (int half = 0; half < 2; half++) {
            uint8x16_t source = vld1q_u8(packed + half * 16);
            for (int block = 0; block < 4; block++) {
                uint8x16_t indices = vaddq_u8(
                    base_indices, vdupq_n_u8(block * 4));
                uint8x16_t repeated = vqtbl1q_u8(source, indices);
                uint8x16_t codes = vandq_u8(
                    vshlq_u8(repeated, shifts), mask);
                sum = vusdotq_s32(
                    sum,
                    vld1q_u8(scratch.weights_q8.data()
                              + (half * 4 + block) * 16),
                    vreinterpretq_s8_u8(codes));
            }
        }
        output[dim] = (vaddvq_s32(sum) * weight_scale + bias) * tile.vc[dim];
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
        float factor = std::exp(local_max - global_max);
        for (int dim = 0; dim < dimension; dim++) {
            output[dim] += factor * local[dim];
        }
        global_sum += factor * local_sum;
    } else {
        float factor = std::exp(global_max - local_max);
        for (int dim = 0; dim < dimension; dim++) {
            output[dim] = factor * output[dim] + local[dim];
        }
        global_sum = factor * global_sum + local_sum;
        global_max = local_max;
    }
}

volatile uint32_t metadata_sink = 0;

void fixed_attention(const std::vector<Tile>& tiles,
                     const std::vector<PreparedKey>* prepared,
                     bool extract_metadata,
                     const float* query, int dimension,
                     float* output, Scratch& scratch) {
    bool initialized = false;
    float global_max = 0.0f;
    float global_sum = 0.0f;
    uint32_t metadata_checksum = 0;
    float logit_scale = 1.0f / std::sqrt(static_cast<float>(dimension));
    for (size_t index = 0; index < tiles.size(); index++) {
        const Tile& tile = tiles[index];
        if (extract_metadata) {
            metadata_checksum += extract_syndrome_metadata(
                tile, scratch);
        }
        const int8_t* query_q8;
        float query_scale;
        float bias;
        if (prepared) {
            query_q8 = (*prepared)[index].query.data();
            query_scale = (*prepared)[index].scale;
            bias = (*prepared)[index].bias;
        } else {
            prepare_key(tile, query, dimension, scratch.query_q8.data(),
                        query_scale, bias);
            query_q8 = scratch.query_q8.data();
        }
        dot_keys(tile, query_q8, query_scale, bias, dimension,
                 logit_scale, scratch.scores.data());
        float local_max = *std::max_element(
            scratch.scores.begin(), scratch.scores.end());
        for (int token = 0; token < kTokens; token++) {
            scratch.weights[token] = scratch.scores[token] - local_max;
        }
        int count = kTokens;
        vvexpf(scratch.weights.data(), scratch.weights.data(), &count);
        float local_sum = 0.0f;
        for (float weight : scratch.weights) local_sum += weight;
        add_values(tile, scratch.weights.data(), dimension,
                   scratch.local.data(), scratch);
        merge(local_max, local_sum, scratch.local.data(), dimension,
              initialized, global_max, global_sum, output);
    }
    float inverse = 1.0f / global_sum;
    for (int dim = 0; dim < dimension; dim++) output[dim] *= inverse;
    if (extract_metadata) metadata_sink = metadata_checksum;
}

void fp16_attention(const std::vector<uint16_t>& keys,
                    const std::vector<uint16_t>& values,
                    const float* query, int context, int dimension,
                    float* output, Scratch& scratch) {
    bool initialized = false;
    float global_max = 0.0f;
    float global_sum = 0.0f;
    float logit_scale = 1.0f / std::sqrt(static_cast<float>(dimension));
    for (int first = 0; first < context; first += kTokens) {
        for (int token = 0; token < kTokens; token++) {
            const uint16_t* key = keys.data()
                + static_cast<size_t>(first + token) * dimension;
            float32x4_t sum = vdupq_n_f32(0.0f);
            for (int dim = 0; dim < dimension; dim += 4) {
                float32x4_t value = vcvt_f32_f16(vld1_f16(
                    reinterpret_cast<const __fp16*>(key + dim)));
                sum = vfmaq_f32(sum, vld1q_f32(query + dim), value);
            }
            scratch.scores[token] = vaddvq_f32(sum) * logit_scale;
        }
        float local_max = *std::max_element(
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
            float32x4_t weight = vdupq_n_f32(scratch.weights[token]);
            const uint16_t* value = values.data()
                + static_cast<size_t>(first + token) * dimension;
            for (int dim = 0; dim < dimension; dim += 4) {
                float32x4_t decoded = vcvt_f32_f16(vld1_f16(
                    reinterpret_cast<const __fp16*>(value + dim)));
                float32x4_t current = vld1q_f32(scratch.local.data() + dim);
                vst1q_f32(scratch.local.data() + dim,
                    vfmaq_f32(current, weight, decoded));
            }
        }
        merge(local_max, local_sum, scratch.local.data(), dimension,
              initialized, global_max, global_sum, output);
    }
    float inverse = 1.0f / global_sum;
    for (int dim = 0; dim < dimension; dim++) output[dim] *= inverse;
}

double relative_error(const std::vector<float>& actual,
                      const std::vector<float>& expected) {
    double error = 0.0;
    double norm = 0.0;
    for (size_t index = 0; index < actual.size(); index++) {
        double delta = actual[index] - expected[index];
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
        auto start = std::chrono::steady_clock::now();
        sink = function();
        auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(
            end - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void benchmark(int context, int dimension, int trials) {
    Rng random{0x12345678u + static_cast<uint32_t>(context + dimension)};
    std::vector<uint16_t> fp16_keys;
    std::vector<uint16_t> fp16_values;
    fp16_keys.reserve(static_cast<size_t>(context) * dimension);
    fp16_values.reserve(static_cast<size_t>(context) * dimension);
    std::vector<Tile> tiles;
    tiles.reserve(context / kTokens);
    for (int first = 0; first < context; first += kTokens) {
        tiles.push_back(make_tile(
            dimension, random, fp16_keys, fp16_values));
    }
    std::vector<float> query(dimension);
    for (float& value : query) value = 2.0f * random.unit() - 1.0f;

    std::vector<PreparedKey> prepared(tiles.size());
    for (size_t index = 0; index < tiles.size(); index++) {
        prepared[index].query.resize(dimension);
        prepare_key(tiles[index], query.data(), dimension,
                    prepared[index].query.data(), prepared[index].scale,
                    prepared[index].bias);
    }

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

    std::vector<float> fixed_output(dimension);
    std::vector<float> full_output(dimension);
    std::vector<float> syndrome_output(dimension);
    std::vector<float> fp16_output(dimension);
    std::vector<float> control_output(dimension);
    Scratch fixed_scratch(dimension);
    Scratch full_scratch(dimension);
    Scratch syndrome_scratch(dimension);
    Scratch fp16_scratch(dimension);

    auto fixed_call = [&] {
        fixed_attention(tiles, &prepared, false, query.data(), dimension,
                        fixed_output.data(), fixed_scratch);
        return fixed_output[0];
    };
    auto full_call = [&] {
        fixed_attention(tiles, nullptr, false, query.data(), dimension,
                        full_output.data(), full_scratch);
        return full_output[0];
    };
    auto syndrome_call = [&] {
        fixed_attention(tiles, nullptr, true, query.data(), dimension,
                        syndrome_output.data(), syndrome_scratch);
        return syndrome_output[0];
    };
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

    fixed_call();
    full_call();
    if (dimension == 64) syndrome_call();
    fp16_call();
    control_call();
    double full_error = relative_error(full_output, fixed_output);
    double syndrome_error = dimension == 64
        ? relative_error(syndrome_output, full_output) : 0.0;
    double fixed_fp16_error = relative_error(fixed_output, fp16_output);
    double control_fp16_error = relative_error(control_output, fp16_output);
    if (full_error > 1e-6 || syndrome_error > 1e-6) {
        std::fprintf(stderr, "path mismatch: prepared=%.9g syndrome=%.9g\n",
                     full_error, syndrome_error);
        std::abort();
    }

    double fixed_ms = median_ms(fixed_call, trials);
    double full_ms = median_ms(full_call, trials);
    double syndrome_ms = dimension == 64
        ? median_ms(syndrome_call, trials) : NAN;
    double fp16_ms = median_ms(fp16_call, trials);
    double control_ms = median_ms(control_call, trials);
    double actual_fp32_metadata_bits = 3.0
        + 3.0 * (dimension + kTokens) * sizeof(float) * 8.0
          / (2.0 * kTokens * dimension);
    double modeled_q8_metadata_bits = 3.0
        + (3.0 * (dimension + kTokens) + 24.0) * 8.0
          / (2.0 * kTokens * dimension);
    size_t prepared_scratch_bytes = tiles.size()
        * (static_cast<size_t>(dimension) + 2 * sizeof(float));
    std::printf(
        "D=%d context=%d trials=%d synthetic=true code_bits=3.000000 "
        "actual_fp32_metadata_bits=%.6f modeled_q8_metadata_bits=%.6f "
        "prepared_scratch_bytes=%zu fixed_kernel_ms=%.6f fixed_full_ms=%.6f "
        "syndrome_applicable=%s syndrome_groups=%d+%d syndrome_bits=%d "
        "syndrome_ms=%.6f fp16_ms=%.6f k8v6_ms=%.6f kernel_vs_fp16=%.3fx "
        "full_vs_fp16=%.3fx kernel_vs_k8v6=%.3fx full_vs_k8v6=%.3fx "
        "syndrome_vs_fp16=%.3fx syndrome_vs_k8v6=%.3fx "
        "prepared_relerr=%.3g syndrome_relerr=%.3g fixed_fp16_relerr=%.3g "
        "k8v6_fp16_relerr=%.3g\n",
        dimension, context, trials, actual_fp32_metadata_bits,
        modeled_q8_metadata_bits, prepared_scratch_bytes,
        fixed_ms, full_ms, dimension == 64 ? "true" : "false",
        kSyndromeKeyGroups, kSyndromeValueGroups, kSyndromeBits,
        syndrome_ms, fp16_ms, control_ms,
        fp16_ms / fixed_ms, fp16_ms / full_ms,
        control_ms / fixed_ms, control_ms / full_ms,
        fp16_ms / syndrome_ms, control_ms / syndrome_ms,
        full_error, syndrome_error, fixed_fp16_error, control_fp16_error);
}

} // namespace

int main(int argc, char** argv) {
    int context = argc > 1 ? std::atoi(argv[1]) : 16384;
    int trials = argc > 2 ? std::atoi(argv[2]) : 7;
    if (context < kTokens || context % kTokens || trials < 1) {
        std::fprintf(stderr,
            "usage: %s [context_multiple_of_128] [trials]\n", argv[0]);
        return 2;
    }
    std::puts(
        "fixed_format=K4V2 tile=128 metadata=decoded_fp32 "
        "prepared_path=query_conditioned_key_metadata_excluded "
        "full_path=query_conditioned_key_metadata_included "
        "single_query=true warm_cache=true");
    check_syndrome_extraction();
    benchmark(context, 64, trials);
    benchmark(context, 96, trials);
    return 0;
}
