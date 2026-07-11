// test_laplace_kv - sealed chunk accuracy and storage checks.
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "laplace_kv.h"
#include "ops.h"
#include "test_util.h"

using namespace Laplace;

namespace {

float scalar_quantize_q8(const float* input, int size, int8_t* output) {
    float max_abs = 0.0f;
    for (int i = 0; i < size; i++) {
        max_abs = std::max(max_abs, std::fabs(input[i]));
    }
    float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    float inv = 1.0f / scale;
    for (int i = 0; i < size; i++) {
        int value = static_cast<int>(std::nearbyint(input[i] * inv));
        output[i] = static_cast<int8_t>(std::clamp(value, -127, 127));
    }
    return scale;
}

void test_quantizer() {
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, 3.0f);
    for (int dim : {32, 64, 96, 128, 256, 512}) {
        std::vector<float> input(dim);
        for (float& value : input) value = normal(rng);
        std::vector<int8_t> expected(dim), actual(dim);
        float expected_scale = scalar_quantize_q8(
            input.data(), dim, expected.data());
        float actual_scale = laplace_kv_quantize_q8(
            input.data(), dim, actual.data());
        CHECK(expected_scale == actual_scale);
        CHECK(expected == actual);
    }

    std::vector<float> ties(32, 0.0f);
    ties[0] = 127.0f;
    for (int i = 1; i < 16; i++) ties[i] = i - 0.5f;
    std::vector<int8_t> expected(32), actual(32);
    scalar_quantize_q8(ties.data(), 32, expected.data());
    laplace_kv_quantize_q8(ties.data(), 32, actual.data());
    CHECK(expected == actual);
}

void fill_unit(std::vector<float>& data, int rows, int dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (int row = 0; row < rows; row++) {
        float* value = data.data() + static_cast<size_t>(row) * dim;
        float norm2 = 0.0f;
        for (int d = 0; d < dim; d++) {
            value[d] = normal(rng);
            norm2 += value[d] * value[d];
        }
        float inv = 1.0f / std::sqrt(norm2);
        for (int d = 0; d < dim; d++) value[d] *= inv;
    }
}

void test_chunk() {
    constexpr int dim = 128;
    constexpr int tokens = LaplaceKVTile::kTokens;
    std::vector<float> keys(static_cast<size_t>(tokens) * dim);
    std::vector<float> values(static_cast<size_t>(tokens) * dim);
    std::vector<float> query(dim);
    fill_unit(keys, tokens, dim, 10);
    fill_unit(values, tokens, dim, 20);
    fill_unit(query, 1, dim, 30);

    std::vector<float> keys_wh = keys;
    std::vector<float> values_wh = values;
    std::vector<float> query_wh = query;
    for (int token = 0; token < tokens; token++) {
        walsh_hadamard(keys_wh.data() + static_cast<size_t>(token) * dim, dim);
        walsh_hadamard(values_wh.data() + static_cast<size_t>(token) * dim, dim);
    }
    walsh_hadamard(query_wh.data(), dim);

    LaplaceKVTile chunk;
    CHECK(chunk.init(dim));
    CHECK(chunk.seal(keys_wh.data(), values_wh.data()));

    std::vector<int8_t> query_q8(dim);
    float query_scale = laplace_kv_quantize_q8(query_wh.data(), dim, query_q8.data());
    std::vector<float> scores(tokens);
    chunk.dot_keys(query_q8.data(), query_scale, scores.data());

    double score_error = 0.0, score_reference = 0.0;
    std::vector<float> exact_scores(tokens);
    for (int token = 0; token < tokens; token++) {
        exact_scores[token] = ops::dot(query.data(),
                                       keys.data() + static_cast<size_t>(token) * dim,
                                       dim);
        double delta = scores[token] - exact_scores[token];
        score_error += delta * delta;
        score_reference += static_cast<double>(exact_scores[token]) * exact_scores[token];
    }
    CHECK_MSG(std::sqrt(score_error / score_reference) < 0.02,
              "score relative error %.4f", std::sqrt(score_error / score_reference));

    float max_score = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& score : scores) {
        score = std::exp(score - max_score);
        sum += score;
    }
    for (float& score : scores) score /= sum;

    std::vector<float> output_wh(dim, 0.0f);
    chunk.add_values(scores.data(), output_wh.data());
    inverse_walsh_hadamard(output_wh.data(), dim);

    std::vector<float> exact_output(dim, 0.0f);
    max_score = *std::max_element(exact_scores.begin(), exact_scores.end());
    sum = 0.0f;
    for (float& score : exact_scores) {
        score = std::exp(score - max_score);
        sum += score;
    }
    for (int token = 0; token < tokens; token++) {
        ops::axpy(exact_output.data(), exact_scores[token] / sum,
                  values.data() + static_cast<size_t>(token) * dim, dim);
    }

    double output_error = 0.0, output_reference = 0.0;
    for (int d = 0; d < dim; d++) {
        double delta = output_wh[d] - exact_output[d];
        output_error += delta * delta;
        output_reference += static_cast<double>(exact_output[d]) * exact_output[d];
    }
    CHECK_MSG(std::sqrt(output_error / output_reference) < 0.03,
              "output relative error %.4f", std::sqrt(output_error / output_reference));

    size_t fp16_bytes = static_cast<size_t>(2) * tokens * dim * sizeof(uint16_t);
    CHECK_MSG(chunk.storage_bytes() * 5 < fp16_bytes * 3,
              "storage %zu is not at least 40%% smaller than FP16 %zu",
              chunk.storage_bytes(), fp16_bytes);
}

void test_cache() {
    constexpr int dim = 128;
    constexpr int context = 130;
    std::vector<float> keys(static_cast<size_t>(context) * dim);
    std::vector<float> values(static_cast<size_t>(context) * dim);
    std::vector<float> query(dim);
    fill_unit(keys, context, dim, 40);
    fill_unit(values, context, dim, 50);
    fill_unit(query, 1, dim, 60);

    std::vector<float> keys_wh = keys;
    std::vector<float> values_wh = values;
    std::vector<float> query_wh = query;
    for (int token = 0; token < context; token++) {
        walsh_hadamard(keys_wh.data() + static_cast<size_t>(token) * dim, dim);
        walsh_hadamard(values_wh.data() + static_cast<size_t>(token) * dim, dim);
    }
    walsh_hadamard(query_wh.data(), dim);

    LaplaceKV cache;
    CHECK(cache.init(1, 1, dim, context));
    for (int token = 0; token < context; token++) {
        cache.store_k_wh(0, 0, token,
                         keys_wh.data() + static_cast<size_t>(token) * dim);
        cache.store_v_wh(0, 0, token,
                         values_wh.data() + static_cast<size_t>(token) * dim);
    }

    for (int n_tokens : {1, 63, 64, 65, 129, 130}) {
        std::vector<float> scores(n_tokens);
        cache.dot_keys_wh(0, 0, n_tokens, query_wh.data(), scores.data());

        std::vector<float> exact_scores(n_tokens);
        double score_error = 0.0;
        for (int token = 0; token < n_tokens; token++) {
            exact_scores[token] = ops::dot(
                query.data(), keys.data() + static_cast<size_t>(token) * dim, dim);
            double delta = scores[token] - exact_scores[token];
            score_error += delta * delta;
        }
        double score_rmse = std::sqrt(score_error / n_tokens);
        CHECK_MSG(score_rmse < 0.02,
                  "cache score RMSE %.4f at %d tokens",
                  score_rmse, n_tokens);

        float max_score = *std::max_element(scores.begin(), scores.end());
        float sum = 0.0f;
        for (float& score : scores) {
            score = std::exp(score - max_score);
            sum += score;
        }
        for (float& score : scores) score /= sum;

        std::vector<float> output_wh(dim, 0.0f);
        cache.add_values_wh(0, 0, n_tokens, scores.data(), output_wh.data());
        inverse_walsh_hadamard(output_wh.data(), dim);
        std::vector<float> fused_output(dim);
        cache.attention_wh(
            0, 0, n_tokens, query_wh.data(), 1.0f, fused_output.data());
        inverse_walsh_hadamard(fused_output.data(), dim);

        max_score = *std::max_element(exact_scores.begin(), exact_scores.end());
        sum = 0.0f;
        for (float& score : exact_scores) {
            score = std::exp(score - max_score);
            sum += score;
        }
        std::vector<float> exact_output(dim, 0.0f);
        for (int token = 0; token < n_tokens; token++) {
            ops::axpy(exact_output.data(), exact_scores[token] / sum,
                      values.data() + static_cast<size_t>(token) * dim, dim);
        }

        double output_error = 0.0, output_reference = 0.0;
        for (int d = 0; d < dim; d++) {
            CHECK(std::fabs(fused_output[d] - output_wh[d]) < 1e-5f);
            double delta = output_wh[d] - exact_output[d];
            output_error += delta * delta;
            output_reference += static_cast<double>(exact_output[d])
                              * exact_output[d];
        }
        CHECK_MSG(std::sqrt(output_error / (output_reference + 1e-20)) < 0.13,
                  "cache output relative error %.4f at %d tokens",
                  std::sqrt(output_error / (output_reference + 1e-20)), n_tokens);
    }

    constexpr int first = 65;
    constexpr int end = 129;
    constexpr float sentinel = 1234.0f;
    std::vector<float> range_scores(context, sentinel);
    cache.dot_keys_wh(0, 0, end, query_wh.data(), range_scores.data(), first);
    for (int token = 0; token < context; token++) {
        if (token < first || token >= end) {
            CHECK(range_scores[token] == sentinel);
        } else {
            float exact = ops::dot(
                query.data(), keys.data() + static_cast<size_t>(token) * dim, dim);
            CHECK(std::fabs(range_scores[token] - exact) < 0.03f);
        }
    }

    std::vector<float> range_weights(context, 1.0f / (end - first));
    std::vector<float> range_output(dim, 0.0f), exact_range_output(dim, 0.0f);
    cache.add_values_wh(
        0, 0, end, range_weights.data(), range_output.data(), first);
    inverse_walsh_hadamard(range_output.data(), dim);
    for (int token = first; token < end; token++) {
        ops::axpy(exact_range_output.data(), range_weights[token],
                  values.data() + static_cast<size_t>(token) * dim, dim);
    }
    for (int d = 0; d < dim; d++) {
        CHECK(std::fabs(range_output[d] - exact_range_output[d]) < 0.01f);
    }
    std::vector<float> fused_range_output(dim);
    cache.attention_wh(
        0, 0, end, query_wh.data(), 1.0f, fused_range_output.data(), first);
    inverse_walsh_hadamard(fused_range_output.data(), dim);
    for (int d = 0; d < dim; d++) {
        CHECK(std::fabs(fused_range_output[d] - exact_range_output[d]) < 0.01f);
    }
}

void test_dimensions() {
    constexpr int tokens = LaplaceKVTile::kTokens;
    for (int dim : {64, 96, 256, 512}) {
        std::vector<float> keys(static_cast<size_t>(tokens) * dim);
        std::vector<float> values(keys.size());
        std::vector<float> query(dim);
        fill_unit(keys, tokens, dim, 70 + dim);
        fill_unit(values, tokens, dim, 80 + dim);
        fill_unit(query, 1, dim, 90 + dim);
        std::vector<float> keys_real = keys;
        std::vector<float> values_real = values;
        bool rotate = (dim & (dim - 1)) == 0;
        if (rotate) {
            for (int token = 0; token < tokens; token++) {
                walsh_hadamard(
                    keys.data() + static_cast<size_t>(token) * dim, dim);
                walsh_hadamard(
                    values.data() + static_cast<size_t>(token) * dim, dim);
            }
            walsh_hadamard(query.data(), dim);
        }
        std::vector<float> query_real = query;
        if (rotate) walsh_hadamard(query_real.data(), dim);

        LaplaceKVTile chunk;
        CHECK_MSG(chunk.init(dim), "chunk init failed at dim=%d", dim);
        CHECK(chunk.seal(keys.data(), values.data()));
        std::vector<int8_t> query_q8(dim);
        float query_scale = laplace_kv_quantize_q8(
            query.data(), dim, query_q8.data());
        std::vector<float> scores(tokens);
        chunk.dot_keys(query_q8.data(), query_scale, scores.data());

        double error = 0.0;
        for (int token = 0; token < tokens; token++) {
            float exact = ops::dot(
                query_real.data(),
                keys_real.data() + static_cast<size_t>(token) * dim, dim);
            double delta = scores[token] - exact;
            error += delta * delta;
        }
        CHECK_MSG(std::sqrt(error / tokens) < 0.02,
                  "chunk score RMSE %.4f at dim=%d",
                  std::sqrt(error / tokens), dim);

        std::vector<float> weights(tokens);
        float weight_sum = 0.0f;
        for (int token = 0; token < tokens; token++) {
            weights[token] = 1.0f + 0.25f * std::sin(static_cast<float>(token));
            weight_sum += weights[token];
        }
        for (float& weight : weights) weight /= weight_sum;
        std::vector<float> output(dim, 0.0f), exact_output(dim, 0.0f);
        chunk.add_values(weights.data(), output.data());
        if (rotate) inverse_walsh_hadamard(output.data(), dim);
        for (int token = 0; token < tokens; token++) {
            ops::axpy(exact_output.data(), weights[token],
                      values_real.data() + static_cast<size_t>(token) * dim, dim);
        }
        double output_error = 0.0, output_reference = 0.0;
        for (int d = 0; d < dim; d++) {
            double delta = output[d] - exact_output[d];
            output_error += delta * delta;
            output_reference += static_cast<double>(exact_output[d]) * exact_output[d];
        }
        double relative_error = std::sqrt(output_error / output_reference);
        CHECK_MSG(relative_error < 0.04,
                  "chunk output error %.4f at dim=%d",
                  relative_error, dim);
    }
}

void test_head_indexing() {
    constexpr int dim = 64;
    constexpr int context = 65;
    LaplaceKV cache;
    CHECK(cache.init(2, 2, dim, context));
    std::vector<float> key(dim), value(dim), loaded(dim);

    for (int layer = 0; layer < 2; layer++) {
        for (int head = 0; head < 2; head++) {
            for (int token = 0; token < context; token++) {
                float base = 0.1f * (1 + layer * 2 + head) + token * 0.001f;
                for (int d = 0; d < dim; d++) {
                    key[d] = base + d * 0.0001f;
                    value[d] = -base + d * 0.0001f;
                }
                cache.store_k_wh(layer, head, token, key.data());
                cache.store_v_wh(layer, head, token, value.data());
            }
        }
    }

    for (int layer = 0; layer < 2; layer++) {
        for (int head = 0; head < 2; head++) {
            for (int token : {0, 63, 64}) {
                float base = 0.1f * (1 + layer * 2 + head) + token * 0.001f;
                cache.load_k_wh(layer, head, token, loaded.data());
                for (int d = 0; d < dim; d++) {
                    CHECK(std::fabs(loaded[d] - (base + d * 0.0001f)) < 0.005f);
                }
                cache.load_v_wh(layer, head, token, loaded.data());
                for (int d = 0; d < dim; d++) {
                    CHECK(std::fabs(loaded[d] - (-base + d * 0.0001f)) < 0.01f);
                }
            }
        }
    }
}

void test_streaming_cache() {
    constexpr int dim = 64;
    constexpr int capacity = 65536;
    constexpr int context = 129 * LaplaceKVTile::kTokens + 1;
    LaplaceKV cache;
    CHECK(cache.init(1, 1, dim, capacity, true));
    CHECK(cache.streaming());
    CHECK(cache.archive_read_buffer_bytes()
          == 512 * LaplaceKVTile::storage_words(dim) * sizeof(uint32_t));

    std::vector<float> keys(static_cast<size_t>(context) * dim);
    std::vector<float> values(keys.size());
    std::vector<float> query(dim);
    fill_unit(keys, context, dim, 120);
    fill_unit(values, context, dim, 121);
    fill_unit(query, 1, dim, 122);
    for (int token = 0; token < context; token++) {
        cache.store_k_wh(0, 0, token,
                         keys.data() + static_cast<size_t>(token) * dim);
        cache.store_v_wh(0, 0, token,
                         values.data() + static_cast<size_t>(token) * dim);
    }
    size_t tile_bytes = LaplaceKVTile::storage_words(dim) * sizeof(uint32_t);
    CHECK(cache.archive_write_bytes() == 129 * tile_bytes);
    CHECK(cache.archive_read_bytes() == 0);

    std::vector<float> loaded(dim);
    cache.load_k_wh(0, 0, LaplaceKVTile::kTokens, loaded.data());
    CHECK(cache.archive_read_bytes() == tile_bytes);
    double load_error = 0.0;
    for (int d = 0; d < dim; d++) {
        double delta = loaded[d]
                     - keys[static_cast<size_t>(LaplaceKVTile::kTokens) * dim + d];
        load_error += delta * delta;
    }
    CHECK_MSG(std::sqrt(load_error / dim) < 0.01,
              "streaming archive RMSE %.4f", std::sqrt(load_error / dim));

    float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    std::vector<float> output(dim);
    cache.attention_wh(0, 0, context, query.data(), scale, output.data());
    CHECK(cache.archive_read_bytes() == 130 * tile_bytes);
    std::vector<float> scores(context);
    float max_score = -1e30f;
    for (int token = 0; token < context; token++) {
        scores[token] = ops::dot(
            query.data(), keys.data() + static_cast<size_t>(token) * dim, dim)
            * scale;
        max_score = std::max(max_score, scores[token]);
    }
    float sum = 0.0f;
    for (float& score : scores) {
        score = std::exp(score - max_score);
        sum += score;
    }
    std::vector<float> exact(dim, 0.0f);
    for (int token = 0; token < context; token++) {
        ops::axpy(exact.data(), scores[token] / sum,
                  values.data() + static_cast<size_t>(token) * dim, dim);
    }
    double output_error = 0.0;
    double output_reference = 0.0;
    for (int d = 0; d < dim; d++) {
        double delta = output[d] - exact[d];
        output_error += delta * delta;
        output_reference += static_cast<double>(exact[d]) * exact[d];
    }
    double relative_error = std::sqrt(
        output_error / (output_reference + 1e-20));
    CHECK_MSG(relative_error < 0.05,
              "streaming output error %.4f", relative_error);
    CHECK_MSG(cache.storage_bytes() * 4 < cache.archive_bytes(),
              "active storage %zu is not 4x smaller than archive %zu",
              cache.storage_bytes(), cache.archive_bytes());
}

void test_streaming_batch_crosses_seal() {
    constexpr int dim = 64;
    constexpr int context = 70;
    constexpr int count = 4;
    const int ends[count] = {61, 64, 65, 70};
    LaplaceKV cache;
    LaplaceKV resident;
    CHECK(cache.init(1, 1, dim, 128, true));
    CHECK(resident.init(1, 1, dim, 128));

    std::vector<float> keys(static_cast<size_t>(context) * dim);
    std::vector<float> values(keys.size());
    std::vector<float> queries(static_cast<size_t>(count) * dim);
    fill_unit(keys, context, dim, 130);
    fill_unit(values, context, dim, 131);
    fill_unit(queries, count, dim, 132);
    for (int token = 0; token < context; token++) {
        cache.store_k_wh(0, 0, token,
                         keys.data() + static_cast<size_t>(token) * dim);
        cache.store_v_wh(0, 0, token,
                         values.data() + static_cast<size_t>(token) * dim);
        resident.store_k_wh(0, 0, token,
                            keys.data() + static_cast<size_t>(token) * dim);
        resident.store_v_wh(0, 0, token,
                            values.data() + static_cast<size_t>(token) * dim);
    }

    std::vector<float> outputs(static_cast<size_t>(count) * dim);
    const float* query_ptrs[count];
    float* output_ptrs[count];
    for (int query = 0; query < count; query++) {
        query_ptrs[query] = queries.data() + static_cast<size_t>(query) * dim;
        output_ptrs[query] = outputs.data() + static_cast<size_t>(query) * dim;
    }
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    cache.attention_batch_wh(0, 0, count, ends, query_ptrs,
                             scale, output_ptrs);
    std::vector<float> resident_outputs(static_cast<size_t>(count) * dim);
    float* resident_ptrs[count];
    for (int query = 0; query < count; query++) {
        resident_ptrs[query] = resident_outputs.data()
                             + static_cast<size_t>(query) * dim;
    }
    resident.attention_batch_wh(0, 0, count, ends, query_ptrs,
                                scale, resident_ptrs);

    for (int query = 0; query < count; query++) {
        for (int d = 0; d < dim; d++) {
            CHECK(std::fabs(output_ptrs[query][d]
                            - resident_ptrs[query][d]) < 1e-6f);
        }
        std::vector<float> scores(ends[query]);
        float max_score = -1e30f;
        for (int token = 0; token < ends[query]; token++) {
            scores[token] = ops::dot(
                query_ptrs[query],
                keys.data() + static_cast<size_t>(token) * dim, dim) * scale;
            max_score = std::max(max_score, scores[token]);
        }
        float sum = 0.0f;
        for (float& score : scores) {
            score = std::exp(score - max_score);
            sum += score;
        }
        std::vector<float> exact(dim, 0.0f);
        for (int token = 0; token < ends[query]; token++) {
            ops::axpy(exact.data(), scores[token] / sum,
                      values.data() + static_cast<size_t>(token) * dim, dim);
        }
        double error = 0.0;
        double reference = 0.0;
        for (int d = 0; d < dim; d++) {
            double delta = output_ptrs[query][d] - exact[d];
            error += delta * delta;
            reference += static_cast<double>(exact[d]) * exact[d];
        }
        CHECK_MSG(std::sqrt(error / reference) < 0.05,
                  "batch query %d relative error %.4f", query,
                  std::sqrt(error / reference));
    }
}

} // namespace

int main() {
    test_quantizer();
    test_chunk();
    test_cache();
    test_dimensions();
    test_head_indexing();
    test_streaming_cache();
    test_streaming_batch_crosses_seal();
    return test_summary("test_laplace_kv");
}
