#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "laplace_kv_adaptive.h"
#include "test_util.h"

using namespace Laplace;

namespace {

void fill_tile(LaplaceKVAdaptive& cache, int start, bool zero,
               float amplitude = 1.0f, bool zero_values = false) {
    constexpr int dimension = 64;
    std::mt19937 rng(91 + start);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> key(dimension);
    std::vector<float> value(dimension);
    for (int token = 0; token < LaplaceKVAdaptive::kTokens; token++) {
        for (int dim = 0; dim < dimension; dim++) {
            key[dim] = zero ? 0.0f : amplitude * normal(rng);
            value[dim] = (zero || zero_values) ? 0.0f
                                                 : amplitude * normal(rng);
        }
        cache.store_k_wh(0, 0, start + token, key.data());
        cache.store_v_wh(0, 0, start + token, value.data());
    }
}

void test_selector() {
    LaplaceKVTileError exact{};
    LaplaceKVTileError control{0.01, 0.02, 0.03, 0.04};
    CHECK(laplace_kv_prefer_q4(exact, control));

    LaplaceKVTileError lossy{0.01, 0.03, 0.03, 0.04};
    CHECK(!laplace_kv_prefer_q4(lossy, control));

    LaplaceKVTileError invalid = exact;
    invalid.key_rms = std::numeric_limits<double>::quiet_NaN();
    CHECK(!laplace_kv_prefer_q4(invalid, control));

    CHECK(laplace_kv_accept_k8({0.01, 0.02, 0.025, 0.05}));
    CHECK(!laplace_kv_accept_k8({0.01, 0.02, 0.026, 0.05}));

    CHECK(LaplaceKVAdaptive::kTokens == LaplaceKVQ4Tile::kTokens);
    CHECK(codec_accepts_tile(LaplaceKVAdaptiveFormat::K4_V2,
                             LaplaceKVAdaptive::kTokens));
    CHECK(codec_accepts_tile(LaplaceKVAdaptiveFormat::K8_V6,
                             LaplaceKVAdaptive::kTokens));
    CHECK(!codec_accepts_tile(LaplaceKVAdaptiveFormat::K4_V2,
                              LaplaceKVAdaptive::kTokens - 1));
    CHECK(!codec_accepts_tile(LaplaceKVAdaptiveFormat::K8_V6,
                              LaplaceKVAdaptive::kTokens - 1));
}

void test_selection_and_accounting(bool streaming) {
    constexpr int dimension = 64;
    constexpr int capacity = LaplaceKVAdaptive::kTokens;
    LaplaceKVAdaptive q4;
    CHECK(q4.init(1, 1, dimension, capacity, streaming,
                  CompressionEligibility{true, false}));
    fill_tile(q4, 0, true);
    CHECK(q4.tile_format(0, 0, 0) == LaplaceKVAdaptiveFormat::K4_V2);
    CHECK(q4.q4_tiles() == 1);
    CHECK(q4.k8_tiles() == 0);
    const size_t q4_expected =
        LaplaceKVQ4Tile::storage_words(dimension) * sizeof(uint32_t) + 1;
    CHECK(q4.encoded_bytes(capacity) == q4_expected);
    CHECK(q4.archive_bytes() == (streaming ? q4_expected - 1 : 0));

    LaplaceKVAdaptive k8;
    CHECK(k8.init(1, 1, dimension, capacity, streaming,
                  CompressionEligibility{false, true}));
    fill_tile(k8, 0, false, 1.0f, true);
    CHECK_MSG(k8.tile_format(0, 0, 0) == LaplaceKVAdaptiveFormat::K8_V6,
              "format=%d", static_cast<int>(k8.tile_format(0, 0, 0)));
    CHECK(k8.q4_tiles() == 0);
    CHECK(k8.k8_tiles() == 1);
    const size_t k8_expected =
        2 * LaplaceKVTile::storage_words(dimension) * sizeof(uint32_t) + 1;
    CHECK(k8.encoded_bytes(capacity) == k8_expected);
    CHECK(k8.archive_bytes() == (streaming ? k8_expected - 1 : 0));
    CHECK(k8.archive_write_bytes() == (streaming ? k8_expected - 1 : 0));
}

void test_production_eligibility_is_fp16_only() {
    constexpr int dimension = 64;
    LaplaceKVAdaptive cache;
    CHECK(cache.init(1, 1, dimension, LaplaceKVAdaptive::kTokens, false));
    fill_tile(cache, 0, false);
    CHECK(cache.tile_format(0, 0, 0) == LaplaceKVAdaptiveFormat::FP16);
    CHECK(cache.q4_tiles() == 0);
    CHECK(cache.k8_tiles() == 0);
    CHECK(cache.fp16_tiles() == 1);
}

void test_short_sequence_encodes() {
    constexpr int dimension = 64;
    LaplaceKVAdaptive cache;
    CHECK(cache.init(1, 1, dimension, LaplaceKVAdaptive::kTokens, false));
    fill_tile(cache, 0, false);
    CHECK(cache.fp16_tiles() + cache.q4_tiles() + cache.k8_tiles() >= 1);
    CHECK(cache.tile_format(0, 0, 0) != LaplaceKVAdaptiveFormat::MUTABLE);
}

void test_roundtrip_and_attention(bool streaming) {
    constexpr int dimension = 64;
    constexpr int capacity = 257;
    LaplaceKVAdaptive cache;
    CHECK(cache.init(1, 1, dimension, capacity, streaming));
    fill_tile(cache, 0, true);
    fill_tile(cache, LaplaceKVAdaptive::kTokens, false);

    std::vector<float> tail_key(dimension);
    std::vector<float> tail_value(dimension);
    std::vector<float> query(dimension);
    for (int dim = 0; dim < dimension; dim++) {
        tail_key[dim] = std::sin(dim * 0.07f);
        tail_value[dim] = std::cos(dim * 0.11f);
        query[dim] = std::sin(dim * 0.13f);
    }
    cache.store_k_wh(0, 0, capacity - 1, tail_key.data());
    cache.store_v_wh(0, 0, capacity - 1, tail_value.data());

    std::vector<float> loaded(dimension);
    cache.load_k_wh(0, 0, capacity - 1, loaded.data());
    for (int dim = 0; dim < dimension; dim++)
        CHECK(loaded[dim] == tail_key[dim]);

    std::vector<float> scores(capacity);
    cache.dot_keys_wh(0, 0, capacity, query.data(), scores.data());
    float maximum = scores[0] / std::sqrt(static_cast<float>(dimension));
    for (float score : scores)
        maximum = std::max(
            maximum, score / std::sqrt(static_cast<float>(dimension)));
    float sum = 0.0f;
    for (float& score : scores) {
        score = std::exp(
            score / std::sqrt(static_cast<float>(dimension)) - maximum);
        sum += score;
    }
    for (float& score : scores) score /= sum;

    std::vector<float> expected(dimension, 0.0f);
    cache.add_values_wh(
        0, 0, capacity, scores.data(), expected.data());
    std::vector<float> actual(dimension);
    cache.attention_wh(
        0, 0, capacity, query.data(),
        1.0f / std::sqrt(static_cast<float>(dimension)), actual.data());
    for (int dim = 0; dim < dimension; dim++)
        CHECK(almost_equal(actual[dim], expected[dim], 1e-5f, 1e-5f));

    if (streaming) {
        CHECK(cache.archive_read_bytes() > 0);
        CHECK(cache.stream_calls() == 1);
    }
}

} // namespace

int main() {
    test_selector();
    test_selection_and_accounting(false);
    test_selection_and_accounting(true);
    test_production_eligibility_is_fp16_only();
    test_short_sequence_encodes();
    test_roundtrip_and_attention(false);
    test_roundtrip_and_attention(true);
    return test_summary("test_laplace_kv_adaptive");
}
