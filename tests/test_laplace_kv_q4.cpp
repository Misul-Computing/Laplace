#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "fp16.h"
#include "laplace_kv_q4.h"
#include "test_util.h"

using namespace Laplace;

namespace {

void set_k4(uint8_t* codes, int index, uint8_t code) {
    codes[index / 2] |= static_cast<uint8_t>(code << ((index & 1) * 4));
}

void set_v2(uint8_t* codes, int index, uint8_t code) {
    codes[index / 4] |= static_cast<uint8_t>(code << ((index & 3) * 2));
}

void test_fixed_equations(int dimension) {
    const int tokens = LaplaceKVQ4Tile::kTokens;
    const size_t metadata_values = static_cast<size_t>(3) * dimension +
                                   static_cast<size_t>(3) * tokens;
    const size_t metadata_bytes = metadata_values * sizeof(uint16_t);
    const size_t key_bytes = static_cast<size_t>(tokens) * dimension / 2;
    const size_t value_bytes = static_cast<size_t>(tokens) * dimension / 4;
    const size_t exact_bytes = metadata_bytes + key_bytes + value_bytes;
    CHECK(LaplaceKVQ4Tile::encoded_bytes(dimension) == exact_bytes);
    CHECK(LaplaceKVQ4Tile::storage_words(dimension) * sizeof(uint32_t) ==
          (exact_bytes + 3) / 4 * 4);

    std::vector<uint32_t> storage(LaplaceKVQ4Tile::storage_words(dimension), 0);
    uint8_t* raw = reinterpret_cast<uint8_t*>(storage.data());
    auto* ka = reinterpret_cast<uint16_t*>(raw);
    auto* kb = ka + dimension;
    auto* kc = kb + dimension;
    auto* va = kc + tokens;
    auto* vb = va + tokens;
    auto* vc = vb + tokens;
    uint8_t* key_codes = reinterpret_cast<uint8_t*>(vc + dimension);
    uint8_t* value_codes = key_codes + key_bytes;

    for (int d = 0; d < dimension; d++) {
        ka[d] = fp32_to_fp16(0.01f + d * 0.00001f);
        kb[d] = fp32_to_fp16(-0.1f + d * 0.00002f);
        vc[d] = fp32_to_fp16(0.8f + d * 0.0001f);
    }
    for (int t = 0; t < tokens; t++) {
        kc[t] = fp32_to_fp16(0.75f + t * 0.001f);
        va[t] = fp32_to_fp16(0.02f + t * 0.00002f);
        vb[t] = fp32_to_fp16(-0.08f + t * 0.00001f);
        for (int d = 0; d < dimension; d++) {
            set_k4(key_codes, t * dimension + d,
                   static_cast<uint8_t>((t + d) & 15));
            set_v2(value_codes, d * tokens + t,
                   static_cast<uint8_t>((t + 2 * d) & 3));
        }
    }

    LaplaceKVQ4Tile tile;
    CHECK(tile.init(dimension, storage.data(), true));
    std::vector<float> key(dimension), value(dimension);
    for (int t : {0, 63, 127}) {
        tile.load_key_wh(t, key.data());
        tile.load_value_wh(t, value.data());
        for (int d = 0; d < dimension; d++) {
            int kcode = (t + d) & 15;
            int vcode = (t + 2 * d) & 3;
            float expected_k =
                (kcode * fp16_to_fp32(ka[d]) + fp16_to_fp32(kb[d])) *
                fp16_to_fp32(kc[t]);
            float expected_v =
                (vcode * fp16_to_fp32(va[t]) + fp16_to_fp32(vb[t])) *
                fp16_to_fp32(vc[d]);
            CHECK(almost_equal(key[d], expected_k, 1e-6f, 1e-6f));
            CHECK(almost_equal(value[d], expected_v, 1e-6f, 1e-6f));
        }
    }
}

void test_cache_accounting(bool streaming) {
    constexpr int layers = 2;
    constexpr int heads = 3;
    constexpr int dimension = 96;
    constexpr int capacity = 256;
    LaplaceKVQ4 cache;
    CHECK(cache.init(layers, heads, dimension, capacity, streaming));
    const size_t head_count = static_cast<size_t>(layers) * heads;
    const size_t tile_bytes =
        LaplaceKVQ4Tile::storage_words(dimension) * sizeof(uint32_t);
    const size_t archive = head_count * 2 * tile_bytes;
    CHECK(cache.archive_bytes() == (streaming ? archive : 0));
    CHECK(cache.archive_read_buffer_bytes() == (streaming ? tile_bytes : 0));
    CHECK(cache.encoded_bytes(128) ==
          head_count * (LaplaceKVQ4Tile::encoded_bytes(dimension) + 1));
    CHECK(cache.encoded_bytes(129) ==
          head_count * (LaplaceKVQ4Tile::encoded_bytes(dimension) +
                        2 * dimension * sizeof(float) + 2));
}

void test_cache_attention(bool streaming) {
    constexpr int dimension = 64;
    constexpr int tokens = 129;
    LaplaceKVQ4 cache;
    CHECK(cache.init(1, 1, dimension, tokens, streaming));
    std::vector<float> key(dimension), value(dimension), query(dimension);
    for (int dim = 0; dim < dimension; dim++)
        query[dim] = std::sin(dim * 0.17f);
    for (int token = 0; token < tokens; token++) {
        for (int dim = 0; dim < dimension; dim++) {
            key[dim] = std::sin(token * 0.03f + dim * 0.11f);
            value[dim] = std::cos(token * 0.07f - dim * 0.05f);
        }
        cache.store_k_wh(0, 0, token, key.data());
        cache.store_v_wh(0, 0, token, value.data());
    }

    std::vector<float> scores(tokens), decoded(dimension);
    float maximum = -INFINITY;
    for (int token = 0; token < tokens; token++) {
        cache.load_k_wh(0, 0, token, decoded.data());
        float score = 0.0f;
        for (int dim = 0; dim < dimension; dim++)
            score += query[dim] * decoded[dim];
        scores[token] = score / std::sqrt(static_cast<float>(dimension));
        maximum = std::max(maximum, scores[token]);
    }
    float sum = 0.0f;
    for (float& score : scores) {
        score = std::exp(score - maximum);
        sum += score;
    }
    std::vector<float> expected(dimension, 0.0f);
    for (int token = 0; token < tokens; token++) {
        cache.load_v_wh(0, 0, token, decoded.data());
        for (int dim = 0; dim < dimension; dim++)
            expected[dim] += scores[token] / sum * decoded[dim];
    }

    std::vector<float> actual(dimension);
    cache.attention_wh(0, 0, tokens, query.data(),
                       1.0f / std::sqrt(static_cast<float>(dimension)),
                       actual.data());
    for (int dim = 0; dim < dimension; dim++)
        CHECK(almost_equal(actual[dim], expected[dim], 1e-5f, 1e-5f));
    if (streaming) {
        CHECK(cache.stream_calls() == 1);
        CHECK(cache.archive_read_bytes() > 0);
        CHECK(cache.archive_write_bytes() > 0);
    }
}

} // namespace

int main() {
    for (int dimension : {64, 96, 256, 512})
        test_fixed_equations(dimension);
    test_cache_accounting(false);
    test_cache_accounting(true);
    test_cache_attention(false);
    test_cache_attention(true);
    return test_summary("test_laplace_kv_q4");
}
