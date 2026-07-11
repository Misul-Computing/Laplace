// Standalone Apple Silicon microbenchmark for the two-context K4/V2 codec.
// Build: clang++ -O3 -std=c++20 -mcpu=native -o /tmp/bench-rans \
//        research/laplace_kv/bench_context_rans_native.cpp

#include "../../src/fp16.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#if !defined(__aarch64__)
#error "This benchmark requires AArch64 NEON"
#endif
#include <arm_neon.h>

namespace {

constexpr int kTokens = 128;
constexpr int kScaleBits = 12;
constexpr uint32_t kScaleTotal = 1u << kScaleBits;
constexpr uint32_t kRansLow = 1u << 23;

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

void append_u16(std::vector<uint8_t>& out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value));
    out.push_back(static_cast<uint8_t>(value >> 8));
}

void append_u32(std::vector<uint8_t>& out, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<uint8_t>(value >> shift));
    }
}

uint16_t read_u16(const uint8_t* source) {
    return static_cast<uint16_t>(source[0] | (source[1] << 8));
}

uint32_t read_u32(const uint8_t* source) {
    return static_cast<uint32_t>(source[0])
         | (static_cast<uint32_t>(source[1]) << 8)
         | (static_cast<uint32_t>(source[2]) << 16)
         | (static_cast<uint32_t>(source[3]) << 24);
}

uint64_t read_u64(const uint8_t* source) {
    uint64_t value = 0;
    for (int shift = 0; shift < 64; shift += 8) {
        value |= static_cast<uint64_t>(source[shift / 8]) << shift;
    }
    return value;
}

void write_u64(uint8_t* destination, uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        destination[shift / 8] = static_cast<uint8_t>(value >> shift);
    }
}

struct Model {
    int levels = 0;
    std::array<uint16_t, 16> frequency{};
    std::array<uint16_t, 16> end{};
};

Model normalize(const std::array<uint32_t, 16>& counts, int levels) {
    Model model;
    model.levels = levels;
    uint32_t count_sum = 0;
    for (int symbol = 0; symbol < levels; symbol++) count_sum += counts[symbol];
    std::array<double, 16> exact{};
    int total = 0;
    for (int symbol = 0; symbol < levels; symbol++) {
        exact[symbol] = counts[symbol] * (double(kScaleTotal) / count_sum);
        int frequency = static_cast<int>(std::floor(exact[symbol]));
        if (counts[symbol] && frequency == 0) frequency = 1;
        model.frequency[symbol] = static_cast<uint16_t>(frequency);
        total += frequency;
    }
    while (total < static_cast<int>(kScaleTotal)) {
        int best = 0;
        for (int symbol = 1; symbol < levels; symbol++) {
            if (!counts[symbol]) continue;
            double candidate = exact[symbol] - model.frequency[symbol];
            double current = exact[best] - model.frequency[best];
            if (!counts[best] || candidate > current) best = symbol;
        }
        model.frequency[best]++;
        total++;
    }
    while (total > static_cast<int>(kScaleTotal)) {
        int best = -1;
        for (int symbol = 0; symbol < levels; symbol++) {
            if (model.frequency[symbol] <= 1) continue;
            if (best < 0 || model.frequency[symbol] - exact[symbol]
                            > model.frequency[best] - exact[best]) {
                best = symbol;
            }
        }
        model.frequency[best]--;
        total--;
    }
    int cumulative = 0;
    for (int symbol = 0; symbol < levels; symbol++) {
        cumulative += model.frequency[symbol];
        model.end[symbol] = static_cast<uint16_t>(cumulative);
    }
    return model;
}

std::array<Model, 2> build_models(const std::vector<uint8_t>& codes,
                                  int dimension, int levels, int shift) {
    std::array<std::array<uint32_t, 16>, 2> counts{};
    for (int token = 0; token < kTokens; token++) {
        for (int dim = 0; dim < dimension; dim++) {
            int index = token * dimension + dim;
            int context = token ? codes[index - dimension] >> shift : 0;
            counts[context][codes[index]]++;
        }
    }
    return {normalize(counts[0], levels), normalize(counts[1], levels)};
}

std::vector<uint8_t> encode_lane(const std::vector<uint8_t>& codes,
                                 int dimension, int shift,
                                 const std::array<Model, 2>& models,
                                 int lane) {
    uint32_t state = kRansLow;
    std::vector<uint8_t> emitted;
    for (int index = static_cast<int>(codes.size()) - 1; index >= 0; index--) {
        if ((index & 3) != lane) continue;
        int token = index / dimension;
        int context = token ? codes[index - dimension] >> shift : 0;
        int symbol = codes[index];
        uint32_t frequency = models[context].frequency[symbol];
        uint32_t cumulative = models[context].end[symbol] - frequency;
        uint32_t threshold = ((kRansLow >> kScaleBits) << 8) * frequency;
        while (state >= threshold) {
            emitted.push_back(static_cast<uint8_t>(state));
            state >>= 8;
        }
        state = ((state / frequency) << kScaleBits)
              + state % frequency + cumulative;
    }
    std::vector<uint8_t> payload;
    append_u32(payload, state);
    payload.insert(payload.end(), emitted.rbegin(), emitted.rend());
    return payload;
}

struct Metadata {
    std::vector<float> ka, kb, kc;
    std::vector<float> va, vb, vc;
};

void append_field(std::vector<uint8_t>& bytes, const std::vector<float>& input,
                  std::vector<float>& decoded) {
    auto [low_it, high_it] = std::minmax_element(input.begin(), input.end());
    float low = Laplace::fp16_to_fp32(Laplace::fp32_to_fp16(*low_it));
    float step = (*high_it - *low_it) / 255.0f;
    step = Laplace::fp16_to_fp32(Laplace::fp32_to_fp16(step));
    if (!(step > 0.0f)) step = std::ldexp(1.0f, -24);
    append_u16(bytes, Laplace::fp32_to_fp16(low));
    append_u16(bytes, Laplace::fp32_to_fp16(step));
    decoded.resize(input.size());
    for (size_t index = 0; index < input.size(); index++) {
        int code = static_cast<int>(std::nearbyint((input[index] - low) / step));
        code = std::clamp(code, 0, 255);
        bytes.push_back(static_cast<uint8_t>(code));
        decoded[index] = low + code * step;
    }
}

const uint8_t* read_field(const uint8_t* source, int size,
                          std::vector<float>& output) {
    float low = Laplace::fp16_to_fp32(read_u16(source));
    float step = Laplace::fp16_to_fp32(read_u16(source + 2));
    source += 4;
    output.resize(size);
    for (int index = 0; index < size; index++) {
        output[index] = low + source[index] * step;
    }
    return source + size;
}

std::vector<uint8_t> make_metadata(int dimension, Rng& rng,
                                   Metadata& decoded) {
    auto field = [&](int size, float low, float span) {
        std::vector<float> values(size);
        for (float& value : values) value = low + span * rng.unit();
        return values;
    };
    std::vector<uint8_t> bytes;
    append_field(bytes, field(dimension, 0.010f, 0.020f), decoded.ka);
    append_field(bytes, field(dimension, -0.15f, 0.10f), decoded.kb);
    append_field(bytes, field(kTokens, 0.75f, 0.50f), decoded.kc);
    append_field(bytes, field(kTokens, 0.025f, 0.040f), decoded.va);
    append_field(bytes, field(kTokens, -0.10f, 0.08f), decoded.vb);
    append_field(bytes, field(dimension, 0.75f, 0.50f), decoded.vc);
    return bytes;
}

void parse_metadata(const uint8_t* source, int dimension, Metadata& metadata) {
    source = read_field(source, dimension, metadata.ka);
    source = read_field(source, dimension, metadata.kb);
    source = read_field(source, kTokens, metadata.kc);
    source = read_field(source, kTokens, metadata.va);
    source = read_field(source, kTokens, metadata.vb);
    read_field(source, dimension, metadata.vc);
}

void append_models(std::vector<uint8_t>& record,
                   const std::array<Model, 2>& models) {
    for (const Model& model : models) {
        for (int symbol = 0; symbol < model.levels - 1; symbol++) {
            append_u16(record, model.frequency[symbol]);
        }
    }
}

const uint8_t* read_models(const uint8_t* source, int levels,
                           std::array<Model, 2>& models) {
    for (Model& model : models) {
        model.levels = levels;
        int total = 0;
        for (int symbol = 0; symbol < levels - 1; symbol++) {
            model.frequency[symbol] = read_u16(source);
            source += 2;
            total += model.frequency[symbol];
        }
        model.frequency[levels - 1] = static_cast<uint16_t>(kScaleTotal - total);
        int cumulative = 0;
        for (int symbol = 0; symbol < levels; symbol++) {
            cumulative += model.frequency[symbol];
            model.end[symbol] = static_cast<uint16_t>(cumulative);
        }
    }
    return source;
}

struct PackedTile {
    std::vector<uint8_t> metadata;
    std::vector<uint8_t> key;
    std::vector<uint8_t> value;
    std::vector<uint16_t> key_fp16;
    std::vector<uint16_t> value_fp16;
};

std::vector<uint8_t> pack_codes(const std::vector<uint8_t>& codes, int bits) {
    std::vector<uint8_t> output((codes.size() * bits + 7) / 8);
    int bit = 0;
    for (uint8_t code : codes) {
        output[bit >> 3] |= static_cast<uint8_t>(code << (bit & 7));
        bit += bits;
    }
    return output;
}

struct BuiltTile {
    std::vector<uint8_t> record;
    PackedTile packed;
};

BuiltTile build_tile(int dimension, uint32_t seed) {
    Rng rng{seed};
    std::vector<uint8_t> keys(kTokens * dimension);
    std::vector<uint8_t> values(kTokens * dimension);
    for (int token = 0; token < kTokens; token++) {
        for (int dim = 0; dim < dimension; dim++) {
            int index = token * dimension + dim;
            if (!token) {
                keys[index] = rng.next() & 15;
                values[index] = rng.next() & 3;
                continue;
            }
            int key_high = keys[index - dimension] >> 3;
            if (rng.next() % 100 >= 80) key_high ^= 1;
            keys[index] = static_cast<uint8_t>((key_high << 3) | (rng.next() & 7));
            int value_high = values[index - dimension] >> 1;
            if (rng.next() % 100 >= 90) value_high ^= 1;
            values[index] = static_cast<uint8_t>((value_high << 1) | (rng.next() & 1));
        }
    }

    Metadata metadata;
    std::vector<uint8_t> metadata_bytes = make_metadata(dimension, rng, metadata);
    auto key_models = build_models(keys, dimension, 16, 3);
    auto value_models = build_models(values, dimension, 4, 1);
    std::array<std::vector<uint8_t>, 8> payloads;
    for (int lane = 0; lane < 4; lane++) {
        payloads[lane] = encode_lane(keys, dimension, 3, key_models, lane);
        payloads[4 + lane] = encode_lane(
            values, dimension, 1, value_models, lane);
    }

    BuiltTile tile;
    tile.record.resize(16);
    for (int lane = 0; lane < 8; lane++) {
        uint16_t size = static_cast<uint16_t>(payloads[lane].size());
        tile.record[lane * 2] = static_cast<uint8_t>(size);
        tile.record[lane * 2 + 1] = static_cast<uint8_t>(size >> 8);
    }
    append_models(tile.record, key_models);
    append_models(tile.record, value_models);
    tile.record.insert(tile.record.end(), metadata_bytes.begin(), metadata_bytes.end());
    for (const auto& payload : payloads) {
        tile.record.insert(tile.record.end(), payload.begin(), payload.end());
    }
    tile.record.resize((tile.record.size() + 15) & ~size_t(15));

    tile.packed.metadata = std::move(metadata_bytes);
    tile.packed.key = pack_codes(keys, 4);
    tile.packed.value = pack_codes(values, 2);
    tile.packed.key_fp16.resize(keys.size());
    tile.packed.value_fp16.resize(values.size());
    for (int token = 0; token < kTokens; token++) {
        for (int dim = 0; dim < dimension; dim++) {
            int index = token * dimension + dim;
            float key = (keys[index] * metadata.ka[dim] + metadata.kb[dim])
                      * metadata.kc[token];
            float value = (values[index] * metadata.va[token] + metadata.vb[token])
                        * metadata.vc[dim];
            tile.packed.key_fp16[index] = Laplace::fp32_to_fp16(key);
            tile.packed.value_fp16[index] = Laplace::fp32_to_fp16(value);
        }
    }
    return tile;
}

struct Archive {
    std::vector<uint8_t> bytes;
};

Archive build_archive(const std::vector<BuiltTile>& tiles) {
    Archive archive;
    archive.bytes.resize(tiles.size() * sizeof(uint64_t));
    archive.bytes.resize((archive.bytes.size() + 15) & ~size_t(15));
    for (size_t index = 0; index < tiles.size(); index++) {
        uint64_t offset = archive.bytes.size();
        write_u64(archive.bytes.data() + index * sizeof(uint64_t), offset);
        archive.bytes.insert(
            archive.bytes.end(), tiles[index].record.begin(), tiles[index].record.end());
    }
    return archive;
}

struct Lane {
    uint32_t state = 0;
    const uint8_t* next = nullptr;
    const uint8_t* end = nullptr;
};

uint8_t decode_symbol(Lane& lane, const Model& model) {
    uint32_t slot = lane.state & (kScaleTotal - 1);
    int symbol;
    if (model.levels == 4) {
        symbol = (slot >= model.end[0])
               + (slot >= model.end[1])
               + (slot >= model.end[2]);
    } else {
        alignas(16) static constexpr uint8_t indices_data[16] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        };
        uint16x8_t slots = vdupq_n_u16(static_cast<uint16_t>(slot));
        uint8x16_t matches = vcombine_u8(
            vmovn_u16(vcgtq_u16(vld1q_u16(model.end.data()), slots)),
            vmovn_u16(vcgtq_u16(vld1q_u16(model.end.data() + 8), slots)));
        uint8x16_t selected = vbslq_u8(
            matches, vld1q_u8(indices_data), vdupq_n_u8(16));
        symbol = vminvq_u8(selected);
    }
    uint32_t frequency = model.frequency[symbol];
    uint32_t cumulative = model.end[symbol] - frequency;
    lane.state = frequency * (lane.state >> kScaleBits) + slot - cumulative;
    while (lane.state < kRansLow && lane.next < lane.end) {
        lane.state = (lane.state << 8) | *lane.next++;
    }
    return static_cast<uint8_t>(symbol);
}

struct TileView {
    std::array<Model, 2> key_models;
    std::array<Model, 2> value_models;
    Metadata metadata;
    std::array<Lane, 4> key_lanes;
    std::array<Lane, 4> value_lanes;
};

void parse_record(const uint8_t* record, int dimension, TileView& view) {
    std::array<uint16_t, 8> lengths{};
    for (int lane = 0; lane < 8; lane++) lengths[lane] = read_u16(record + lane * 2);
    const uint8_t* cursor = record + 16;
    cursor = read_models(cursor, 16, view.key_models);
    cursor = read_models(cursor, 4, view.value_models);
    parse_metadata(cursor, dimension, view.metadata);
    cursor += 3 * (dimension + kTokens) + 24;
    for (int lane = 0; lane < 8; lane++) {
        Lane decoded{read_u32(cursor), cursor + 4, cursor + lengths[lane]};
        if (lane < 4) view.key_lanes[lane] = decoded;
        else view.value_lanes[lane - 4] = decoded;
        cursor += lengths[lane];
    }
}

struct Scratch {
    TileView view;
    std::array<float, kTokens> scores{};
    std::array<float, kTokens> weights{};
    std::vector<uint8_t> previous;
    std::vector<float> local;
    explicit Scratch(int dimension) : previous(dimension), local(dimension) {}
};

void decode_key(TileView& view, int dimension, const float* query,
                float scale, float* scores, std::vector<uint8_t>& previous) {
    std::fill(previous.begin(), previous.end(), 0);
    for (int token = 0; token < kTokens; token++) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        for (int dim = 0; dim < dimension; dim += 4) {
            alignas(16) float code_values[4];
            for (int lane = 0; lane < 4; lane++) {
                int context = previous[dim + lane] >> 3;
                uint8_t code = decode_symbol(
                    view.key_lanes[lane], view.key_models[context]);
                previous[dim + lane] = code;
                code_values[lane] = code;
            }
            float32x4_t code = vld1q_f32(code_values);
            float32x4_t key = vfmaq_f32(
                vld1q_f32(view.metadata.kb.data() + dim), code,
                vld1q_f32(view.metadata.ka.data() + dim));
            sum = vfmaq_f32(sum, vld1q_f32(query + dim), key);
        }
        scores[token] = vaddvq_f32(sum) * view.metadata.kc[token] * scale;
    }
}

void decode_value(TileView& view, int dimension, const float* weights,
                  float* output, std::vector<uint8_t>& previous) {
    std::fill(previous.begin(), previous.end(), 0);
    std::fill_n(output, dimension, 0.0f);
    for (int token = 0; token < kTokens; token++) {
        float32x4_t weight = vdupq_n_f32(weights[token]);
        float32x4_t row_scale = vdupq_n_f32(view.metadata.va[token]);
        float32x4_t row_zero = vdupq_n_f32(view.metadata.vb[token]);
        for (int dim = 0; dim < dimension; dim += 4) {
            alignas(16) float code_values[4];
            for (int lane = 0; lane < 4; lane++) {
                int context = previous[dim + lane] >> 1;
                uint8_t code = decode_symbol(
                    view.value_lanes[lane], view.value_models[context]);
                previous[dim + lane] = code;
                code_values[lane] = code;
            }
            float32x4_t value = vfmaq_f32(
                row_zero, vld1q_f32(code_values), row_scale);
            value = vmulq_f32(value, vld1q_f32(view.metadata.vc.data() + dim));
            float32x4_t current = vld1q_f32(output + dim);
            vst1q_f32(output + dim, vfmaq_f32(current, weight, value));
        }
    }
}

void merge_tile(float local_max, float local_sum, const float* local,
                int dimension, bool& have_global, float& global_max,
                float& global_sum, float* output) {
    if (!have_global) {
        std::copy_n(local, dimension, output);
        global_max = local_max;
        global_sum = local_sum;
        have_global = true;
    } else if (local_max <= global_max) {
        float factor = std::exp(local_max - global_max);
        for (int dim = 0; dim < dimension; dim++) output[dim] += local[dim] * factor;
        global_sum += local_sum * factor;
    } else {
        float factor = std::exp(global_max - local_max);
        for (int dim = 0; dim < dimension; dim++) {
            output[dim] = output[dim] * factor + local[dim];
        }
        global_sum = global_sum * factor + local_sum;
        global_max = local_max;
    }
}

void entropy_attention(const Archive& archive, int tiles, int dimension,
                       const float* query, float* output, Scratch& scratch) {
    bool have_global = false;
    float global_max = 0.0f;
    float global_sum = 0.0f;
    std::fill_n(output, dimension, 0.0f);
    float scale = 1.0f / std::sqrt(static_cast<float>(dimension));
    for (int tile = 0; tile < tiles; tile++) {
        uint64_t offset = read_u64(archive.bytes.data() + tile * sizeof(uint64_t));
        parse_record(archive.bytes.data() + offset, dimension, scratch.view);
        decode_key(scratch.view, dimension, query, scale,
                   scratch.scores.data(), scratch.previous);
        float local_max = *std::max_element(
            scratch.scores.begin(), scratch.scores.end());
        float local_sum = 0.0f;
        for (int token = 0; token < kTokens; token++) {
            scratch.weights[token] = std::exp(scratch.scores[token] - local_max);
            local_sum += scratch.weights[token];
        }
        decode_value(scratch.view, dimension, scratch.weights.data(),
                     scratch.local.data(), scratch.previous);
        merge_tile(local_max, local_sum, scratch.local.data(), dimension,
                   have_global, global_max, global_sum, output);
    }
    float inverse = 1.0f / global_sum;
    for (int dim = 0; dim < dimension; dim++) output[dim] *= inverse;
}

uint8_t packed_code(const std::vector<uint8_t>& source, int index, int bits) {
    return static_cast<uint8_t>(
        (source[(index * bits) >> 3] >> ((index * bits) & 7))
        & ((1 << bits) - 1));
}

bool verify_record(const BuiltTile& tile, int dimension) {
    TileView view;
    parse_record(tile.record.data(), dimension, view);
    std::vector<uint8_t> previous(dimension);
    for (int index = 0; index < kTokens * dimension; index++) {
        int dim = index % dimension;
        int context = previous[dim] >> 3;
        uint8_t code = decode_symbol(
            view.key_lanes[index & 3], view.key_models[context]);
        if (code != packed_code(tile.packed.key, index, 4)) return false;
        previous[dim] = code;
    }
    std::fill(previous.begin(), previous.end(), 0);
    for (int index = 0; index < kTokens * dimension; index++) {
        int dim = index % dimension;
        int context = previous[dim] >> 1;
        uint8_t code = decode_symbol(
            view.value_lanes[index & 3], view.value_models[context]);
        if (code != packed_code(tile.packed.value, index, 2)) return false;
        previous[dim] = code;
    }
    for (const Lane& lane : view.key_lanes) if (lane.next != lane.end) return false;
    for (const Lane& lane : view.value_lanes) if (lane.next != lane.end) return false;
    return true;
}

void packed_attention(const std::vector<BuiltTile>& tiles, int dimension,
                      const float* query, float* output, Scratch& scratch) {
    bool have_global = false;
    float global_max = 0.0f;
    float global_sum = 0.0f;
    std::fill_n(output, dimension, 0.0f);
    float scale = 1.0f / std::sqrt(static_cast<float>(dimension));
    for (const BuiltTile& built : tiles) {
        const PackedTile& tile = built.packed;
        parse_metadata(tile.metadata.data(), dimension, scratch.view.metadata);
        for (int token = 0; token < kTokens; token++) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            for (int dim = 0; dim < dimension; dim += 4) {
                alignas(16) float codes[4];
                for (int lane = 0; lane < 4; lane++) {
                    codes[lane] = packed_code(
                        tile.key, token * dimension + dim + lane, 4);
                }
                float32x4_t key = vfmaq_f32(
                    vld1q_f32(scratch.view.metadata.kb.data() + dim),
                    vld1q_f32(codes),
                    vld1q_f32(scratch.view.metadata.ka.data() + dim));
                sum = vfmaq_f32(sum, vld1q_f32(query + dim), key);
            }
            scratch.scores[token] = vaddvq_f32(sum)
                                  * scratch.view.metadata.kc[token] * scale;
        }
        float local_max = *std::max_element(
            scratch.scores.begin(), scratch.scores.end());
        float local_sum = 0.0f;
        for (int token = 0; token < kTokens; token++) {
            scratch.weights[token] = std::exp(scratch.scores[token] - local_max);
            local_sum += scratch.weights[token];
        }
        std::fill(scratch.local.begin(), scratch.local.end(), 0.0f);
        for (int token = 0; token < kTokens; token++) {
            float32x4_t weight = vdupq_n_f32(scratch.weights[token]);
            float32x4_t row_scale = vdupq_n_f32(scratch.view.metadata.va[token]);
            float32x4_t row_zero = vdupq_n_f32(scratch.view.metadata.vb[token]);
            for (int dim = 0; dim < dimension; dim += 4) {
                alignas(16) float codes[4];
                for (int lane = 0; lane < 4; lane++) {
                    codes[lane] = packed_code(
                        tile.value, token * dimension + dim + lane, 2);
                }
                float32x4_t value = vfmaq_f32(
                    row_zero, vld1q_f32(codes), row_scale);
                value = vmulq_f32(
                    value, vld1q_f32(scratch.view.metadata.vc.data() + dim));
                float32x4_t current = vld1q_f32(scratch.local.data() + dim);
                vst1q_f32(scratch.local.data() + dim,
                           vfmaq_f32(current, weight, value));
            }
        }
        merge_tile(local_max, local_sum, scratch.local.data(), dimension,
                   have_global, global_max, global_sum, output);
    }
    float inverse = 1.0f / global_sum;
    for (int dim = 0; dim < dimension; dim++) output[dim] *= inverse;
}

void fp16_attention(const std::vector<BuiltTile>& tiles, int dimension,
                    const float* query, float* output, Scratch& scratch) {
    bool have_global = false;
    float global_max = 0.0f;
    float global_sum = 0.0f;
    std::fill_n(output, dimension, 0.0f);
    float scale = 1.0f / std::sqrt(static_cast<float>(dimension));
    for (const BuiltTile& built : tiles) {
        const PackedTile& tile = built.packed;
        for (int token = 0; token < kTokens; token++) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            const uint16_t* key = tile.key_fp16.data() + token * dimension;
            for (int dim = 0; dim < dimension; dim += 4) {
                float32x4_t value = vcvt_f32_f16(vld1_f16(
                    reinterpret_cast<const __fp16*>(key + dim)));
                sum = vfmaq_f32(sum, vld1q_f32(query + dim), value);
            }
            scratch.scores[token] = vaddvq_f32(sum) * scale;
        }
        float local_max = *std::max_element(
            scratch.scores.begin(), scratch.scores.end());
        float local_sum = 0.0f;
        for (int token = 0; token < kTokens; token++) {
            scratch.weights[token] = std::exp(scratch.scores[token] - local_max);
            local_sum += scratch.weights[token];
        }
        std::fill(scratch.local.begin(), scratch.local.end(), 0.0f);
        for (int token = 0; token < kTokens; token++) {
            float32x4_t weight = vdupq_n_f32(scratch.weights[token]);
            const uint16_t* value = tile.value_fp16.data() + token * dimension;
            for (int dim = 0; dim < dimension; dim += 4) {
                float32x4_t decoded = vcvt_f32_f16(vld1_f16(
                    reinterpret_cast<const __fp16*>(value + dim)));
                float32x4_t current = vld1q_f32(scratch.local.data() + dim);
                vst1q_f32(scratch.local.data() + dim,
                           vfmaq_f32(current, weight, decoded));
            }
        }
        merge_tile(local_max, local_sum, scratch.local.data(), dimension,
                   have_global, global_max, global_sum, output);
    }
    float inverse = 1.0f / global_sum;
    for (int dim = 0; dim < dimension; dim++) output[dim] *= inverse;
}

double relative_error(const std::vector<float>& left,
                      const std::vector<float>& right) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t index = 0; index < left.size(); index++) {
        double delta = left[index] - right[index];
        numerator += delta * delta;
        denominator += right[index] * right[index];
    }
    return std::sqrt(numerator / std::max(denominator, 1e-30));
}

volatile float benchmark_sink = 0.0f;

template <class Function>
double time_ms(Function&& function, int trials) {
    std::vector<double> samples;
    function();
    function();
    for (int trial = 0; trial < trials; trial++) {
        auto start = std::chrono::steady_clock::now();
        benchmark_sink = function();
        auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void benchmark(int context, int dimension, int trials) {
    int tile_count = context / kTokens;
    std::vector<BuiltTile> tiles;
    tiles.reserve(tile_count);
    auto encode_start = std::chrono::steady_clock::now();
    for (int tile = 0; tile < tile_count; tile++) {
        tiles.push_back(build_tile(dimension, 0x12345678u + tile * 0x9e3779b9u));
        if (!verify_record(tiles.back(), dimension)) {
            std::fprintf(stderr, "record roundtrip failed at tile %d\n", tile);
            std::abort();
        }
    }
    Archive archive = build_archive(tiles);
    auto encode_end = std::chrono::steady_clock::now();

    Rng rng{0xc001d00du};
    std::vector<float> query(dimension);
    for (float& value : query) value = 2.0f * rng.unit() - 1.0f;
    std::vector<float> entropy_output(dimension);
    std::vector<float> packed_output(dimension);
    std::vector<float> fp16_output(dimension);
    Scratch entropy_scratch(dimension);
    Scratch packed_scratch(dimension);
    Scratch fp16_scratch(dimension);

    entropy_attention(archive, tile_count, dimension, query.data(),
                      entropy_output.data(), entropy_scratch);
    packed_attention(tiles, dimension, query.data(),
                     packed_output.data(), packed_scratch);
    fp16_attention(tiles, dimension, query.data(),
                   fp16_output.data(), fp16_scratch);
    double exact_error = relative_error(entropy_output, packed_output);
    double fp16_error = relative_error(entropy_output, fp16_output);
    if (exact_error > 2e-5) {
        std::fprintf(stderr, "roundtrip fusion mismatch: %.9g\n", exact_error);
        std::abort();
    }

    auto entropy_call = [&] {
        entropy_attention(archive, tile_count, dimension, query.data(),
                          entropy_output.data(), entropy_scratch);
        return entropy_output[0];
    };
    auto packed_call = [&] {
        packed_attention(tiles, dimension, query.data(),
                         packed_output.data(), packed_scratch);
        return packed_output[0];
    };
    auto fp16_call = [&] {
        fp16_attention(tiles, dimension, query.data(),
                       fp16_output.data(), fp16_scratch);
        return fp16_output[0];
    };
    double entropy_ms = time_ms(entropy_call, trials);
    double packed_ms = time_ms(packed_call, trials);
    double fp16_ms = time_ms(fp16_call, trials);

    size_t packed_bytes = 0;
    for (const BuiltTile& tile : tiles) {
        packed_bytes += tile.packed.metadata.size()
                      + tile.packed.key.size() + tile.packed.value.size();
    }
    double scalars = 2.0 * context * dimension;
    double entropy_bits = archive.bytes.size() * 8.0 / scalars;
    double packed_bits = packed_bytes * 8.0 / scalars;
    double encode_ms = std::chrono::duration<double, std::milli>(
        encode_end - encode_start).count();
    std::printf(
        "D=%d context=%d tiles=%d synthetic=true trials=%d "
        "entropy_bits=%.6f packed_bits=%.6f build_verify_ms=%.3f "
        "entropy_ms=%.6f packed_k4v2_ms=%.6f fp16_ms=%.6f "
        "entropy_vs_fp16=%.3fx packed_vs_fp16=%.3fx "
        "fusion_relerr=%.3g fp16_relerr=%.3g\n",
        dimension, context, tile_count, trials, entropy_bits, packed_bits,
        encode_ms, entropy_ms, packed_ms, fp16_ms,
        fp16_ms / entropy_ms, fp16_ms / packed_ms, exact_error, fp16_error);
}

} // namespace

int main(int argc, char** argv) {
    int context = argc > 1 ? std::atoi(argv[1]) : 16384;
    int trials = argc > 2 ? std::atoi(argv[2]) : 7;
    if (context < kTokens || context % kTokens || trials < 1) {
        std::fprintf(stderr, "usage: %s [context_multiple_of_128] [trials]\n", argv[0]);
        return 2;
    }
    std::puts("turboquant=unavailable reason=removed_source packed_k4v2_is_proxy=true");
    benchmark(context, 64, trials);
    benchmark(context, 96, trials);
    return 0;
}
