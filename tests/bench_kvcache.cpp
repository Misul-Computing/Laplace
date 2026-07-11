// Fixed-context KV attention benchmark for the real Laplace kernels and a
// local CPU proxy for the MLX affine Q2/G64 format.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <arm_neon.h>

#include "fp16.h"
#include "laplace_kv.h"
#include "ops.h"

using namespace Laplace;

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kGroup = 64;
constexpr int kModes = 4;
volatile float sink = 0.0f;

struct Q2Cache {
    int tokens;
    int dim;
    int groups;
    std::vector<uint8_t> keys;
    std::vector<uint8_t> values;
    std::vector<uint16_t> key_scales;
    std::vector<uint16_t> key_biases;
    std::vector<uint16_t> value_scales;
    std::vector<uint16_t> value_biases;

    size_t bytes() const {
        return keys.size() + values.size()
             + (key_scales.size() + key_biases.size()
                + value_scales.size() + value_biases.size())
                 * sizeof(uint16_t);
    }
};

const std::array<std::array<float, 4>, 256>& q2_table() {
    static const auto table = [] {
        std::array<std::array<float, 4>, 256> result{};
        for (int packed = 0; packed < 256; packed++) {
            for (int lane = 0; lane < 4; lane++) {
                result[packed][lane] = (packed >> (2 * lane)) & 3;
            }
        }
        return result;
    }();
    return table;
}

void fill(std::vector<float>& values, unsigned seed) {
    std::mt19937 random(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (float& value : values) value = normal(random);
}

void quantize_q2(const std::vector<float>& source, int tokens, int dim,
                 std::vector<uint8_t>& codes,
                 std::vector<uint16_t>& scales,
                 std::vector<uint16_t>& biases) {
    int groups = dim / kGroup;
    codes.assign(static_cast<size_t>(tokens) * dim / 4, 0);
    scales.resize(static_cast<size_t>(tokens) * groups);
    biases.resize(scales.size());
    for (int token = 0; token < tokens; token++) {
        for (int group = 0; group < groups; group++) {
            size_t first = static_cast<size_t>(token) * dim + group * kGroup;
            float low = std::numeric_limits<float>::max();
            float high = 0.0f;
            for (int index = 0; index < kGroup; index++) {
                low = std::min(low, source[first + index]);
                high = std::max(high, source[first + index]);
            }
            bool low_side = std::abs(low) > std::abs(high);
            float scale = std::max((high - low) / 3.0f, 1e-7f);
            scale = low_side ? scale : -scale;
            float edge = low_side ? low : high;
            float edge_code = std::round(edge / scale);
            float bias = 0.0f;
            if (edge_code != 0.0f) {
                scale = edge / edge_code;
                bias = edge;
            }
            size_t metadata = static_cast<size_t>(token) * groups + group;
            scales[metadata] = fp32_to_fp16(scale);
            biases[metadata] = fp32_to_fp16(bias);
            for (int index = 0; index < kGroup; index++) {
                int code = static_cast<int>(std::round(
                    (source[first + index] - bias) / scale));
                code = std::clamp(code, 0, 3);
                size_t flat = first + index;
                codes[flat / 4] |= static_cast<uint8_t>(
                    code << (2 * (flat & 3)));
            }
        }
    }
}

Q2Cache make_q2(const std::vector<float>& keys,
                const std::vector<float>& values, int tokens, int dim) {
    Q2Cache cache{tokens, dim, dim / kGroup};
    quantize_q2(keys, tokens, dim, cache.keys,
                cache.key_scales, cache.key_biases);
    quantize_q2(values, tokens, dim, cache.values,
                cache.value_scales, cache.value_biases);
    return cache;
}

void softmax(std::vector<float>& scores) {
    float maximum = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& score : scores) {
        score = std::exp(score - maximum);
        sum += score;
    }
    for (float& score : scores) score /= sum;
}

void fp32_attention(const std::vector<float>& keys,
                    const std::vector<float>& values,
                    const std::vector<float>& query,
                    int tokens, int dim, std::vector<float>& output) {
    std::vector<float> scores(tokens);
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    for (int token = 0; token < tokens; token++) {
        scores[token] = ops::dot(
            query.data(), keys.data() + static_cast<size_t>(token) * dim, dim)
            * scale;
    }
    softmax(scores);
    std::fill(output.begin(), output.end(), 0.0f);
    for (int token = 0; token < tokens; token++) {
        ops::axpy(output.data(), scores[token],
                  values.data() + static_cast<size_t>(token) * dim, dim);
    }
}

void fp16_attention(const std::vector<uint16_t>& keys,
                    const std::vector<uint16_t>& values,
                    const std::vector<float>& query,
                    int tokens, int dim, std::vector<float>& output,
                    std::vector<float>& scores) {
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    for (int token = 0; token < tokens; token++) {
        scores[token] = ops::dot_f16(
            query.data(), keys.data() + static_cast<size_t>(token) * dim, dim)
            * scale;
    }
    softmax(scores);
    std::fill(output.begin(), output.end(), 0.0f);
    for (int token = 0; token < tokens; token++) {
        ops::axpy_f16(output.data(), scores[token],
                      values.data() + static_cast<size_t>(token) * dim, dim);
    }
}

void q2_attention(const Q2Cache& cache, const std::vector<float>& query,
                  std::vector<float>& output, std::vector<float>& scores) {
    const auto& table = q2_table();
    std::array<float, 8> query_sums{};
    for (int group = 0; group < cache.groups; group++) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        for (int index = 0; index < kGroup; index += 4) {
            sum = vaddq_f32(
                sum, vld1q_f32(query.data() + group * kGroup + index));
        }
        query_sums[group] = vaddvq_f32(sum);
    }
    float attention_scale = 1.0f / std::sqrt(static_cast<float>(cache.dim));
    for (int token = 0; token < cache.tokens; token++) {
        float score = 0.0f;
        for (int group = 0; group < cache.groups; group++) {
            size_t metadata = static_cast<size_t>(token) * cache.groups + group;
            float scale = fp16_to_fp32(cache.key_scales[metadata]);
            float bias = fp16_to_fp32(cache.key_biases[metadata]);
            float32x4_t code_dot = vdupq_n_f32(0.0f);
            size_t first = (static_cast<size_t>(token) * cache.dim
                           + group * kGroup) / 4;
            const float* q = query.data() + group * kGroup;
            for (int byte = 0; byte < kGroup / 4; byte++) {
                code_dot = vfmaq_f32(
                    code_dot, vld1q_f32(q + 4 * byte),
                    vld1q_f32(table[cache.keys[first + byte]].data()));
            }
            score += scale * vaddvq_f32(code_dot)
                   + bias * query_sums[group];
        }
        scores[token] = score * attention_scale;
    }
    softmax(scores);
    std::fill(output.begin(), output.end(), 0.0f);
    for (int token = 0; token < cache.tokens; token++) {
        float weight = scores[token];
        for (int group = 0; group < cache.groups; group++) {
            size_t metadata = static_cast<size_t>(token) * cache.groups + group;
            float ws = weight * fp16_to_fp32(cache.value_scales[metadata]);
            float wb = weight * fp16_to_fp32(cache.value_biases[metadata]);
            float32x4_t scale = vdupq_n_f32(ws);
            float32x4_t bias = vdupq_n_f32(wb);
            size_t first = (static_cast<size_t>(token) * cache.dim
                           + group * kGroup) / 4;
            float* destination = output.data() + group * kGroup;
            for (int byte = 0; byte < kGroup / 4; byte++) {
                float32x4_t current = vld1q_f32(destination + 4 * byte);
                float32x4_t code = vld1q_f32(
                    table[cache.values[first + byte]].data());
                vst1q_f32(destination + 4 * byte,
                          vaddq_f32(vfmaq_f32(current, code, scale), bias));
            }
        }
    }
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

template <typename Function>
double time_ms(Function&& function) {
    auto start = Clock::now();
    auto& output = function();
    sink = output[0];
    return std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
}

struct Summary {
    double p5;
    double median;
    double p95;
};

Summary summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    size_t last = samples.size() - 1;
    return {samples[static_cast<size_t>(0.05 * last)],
            samples[last / 2],
            samples[static_cast<size_t>(0.95 * last)]};
}

void benchmark(int tokens, int dim, int trials) {
    if (dim % kGroup != 0) {
        std::fprintf(stderr, "Q2/G64 CPU proxy requires dim divisible by 64\n");
        std::exit(2);
    }
    std::vector<float> keys(static_cast<size_t>(tokens) * dim);
    std::vector<float> values(keys.size());
    std::vector<float> query(dim);
    std::vector<float> expected(dim);
    fill(keys, 100 + tokens + dim);
    fill(values, 200 + tokens + dim);
    fill(query, 300 + tokens + dim);
    fp32_attention(keys, values, query, tokens, dim, expected);

    std::vector<uint16_t> keys16(keys.size());
    std::vector<uint16_t> values16(values.size());
    for (size_t index = 0; index < keys.size(); index++) {
        keys16[index] = fp32_to_fp16(keys[index]);
        values16[index] = fp32_to_fp16(values[index]);
    }
    Q2Cache q2 = make_q2(keys, values, tokens, dim);
    if (q2.bytes() != static_cast<size_t>(tokens) * dim * 5 / 8) {
        std::abort();
    }

    LaplaceKV resident;
    LaplaceKV stream;
    if (!resident.init(1, 1, dim, tokens, false)
        || !stream.init(1, 1, dim, tokens, true)) {
        std::abort();
    }
    std::vector<float> row(dim);
    for (int token = 0; token < tokens; token++) {
        std::copy_n(keys.data() + static_cast<size_t>(token) * dim,
                    dim, row.data());
        walsh_hadamard(row.data(), dim);
        resident.store_k_wh(0, 0, token, row.data());
        stream.store_k_wh(0, 0, token, row.data());
        std::copy_n(values.data() + static_cast<size_t>(token) * dim,
                    dim, row.data());
        walsh_hadamard(row.data(), dim);
        resident.store_v_wh(0, 0, token, row.data());
        stream.store_v_wh(0, 0, token, row.data());
    }
    std::vector<float> query_work(dim);

    std::array<std::vector<float>, kModes> outputs;
    for (auto& output : outputs) output.resize(dim);
    std::vector<float> fp16_scores(tokens);
    std::vector<float> q2_scores(tokens);
    float attention_scale = 1.0f / std::sqrt(static_cast<float>(dim));
    auto call = [&](int mode) -> std::vector<float>& {
        if (mode == 0) {
            fp16_attention(keys16, values16, query, tokens, dim,
                           outputs[mode], fp16_scores);
        } else if (mode == 1) {
            q2_attention(q2, query, outputs[mode], q2_scores);
        } else {
            LaplaceKV& cache = mode == 2 ? resident : stream;
            std::copy(query.begin(), query.end(), query_work.begin());
            walsh_hadamard(query_work.data(), dim);
            cache.attention_wh(0, 0, tokens, query_work.data(),
                               attention_scale, outputs[mode].data());
            inverse_walsh_hadamard(outputs[mode].data(), dim);
        }
        return outputs[mode];
    };

    for (int warmup = 0; warmup < 2; warmup++) {
        for (int mode = 0; mode < kModes; mode++) call(mode);
    }
    std::array<std::vector<double>, kModes> samples;
    for (int trial = 0; trial < trials; trial++) {
        for (int offset = 0; offset < kModes; offset++) {
            int mode = (trial + offset) % kModes;
            samples[mode].push_back(time_ms(
                [&]() -> std::vector<float>& { return call(mode); }));
        }
    }

    const char* names[kModes] = {
        "fp16", "mlx_affine_q2_g64_cpu_proxy", "laplace_k8v6_resident",
        "laplace_k8v6_stream"
    };
    size_t bytes[kModes] = {
        2 * keys16.size() * sizeof(uint16_t), q2.bytes(),
        resident.encoded_bytes(tokens), stream.encoded_bytes(tokens)
    };
    for (int mode = 0; mode < kModes; mode++) {
        Summary stats = summarize(samples[mode]);
        std::printf(
            "{\"context\":%d,\"dim\":%d,\"trials\":%d,"
            "\"mode\":\"%s\",\"cache_record_bytes\":%zu,"
            "\"decoder_constant_bytes\":%zu,"
            "\"p5_ms\":%.9g,\"median_ms\":%.9g,\"p95_ms\":%.9g,"
            "\"output_relative_error\":%.9g,"
            "\"archive_read_bytes\":%llu,\"archive_write_bytes\":%llu}\n",
            tokens, dim, trials, names[mode], bytes[mode],
            mode == 1 ? sizeof(float) * 256 * 4 : 0,
            stats.p5, stats.median, stats.p95,
            relative_error(outputs[mode], expected),
            static_cast<unsigned long long>(
                mode == 3 ? stream.archive_read_bytes() : 0),
            static_cast<unsigned long long>(
                mode == 3 ? stream.archive_write_bytes() : 0));
    }
}

} // namespace

int main(int argc, char** argv) {
    int context = argc > 1 ? std::atoi(argv[1]) : 0;
    int dim = argc > 2 ? std::atoi(argv[2]) : 128;
    int trials = argc > 3 ? std::atoi(argv[3]) : 21;
    if (dim < 64 || dim > 512 || (dim & (dim - 1)) || trials < 20
        || (context && (context < 64 || context % 64))) {
        std::fprintf(stderr,
            "usage: %s [context_multiple_of_64] [dim_multiple_of_64] "
            "[trials_at_least_20]\n", argv[0]);
        return 2;
    }
    std::puts(
        "{\"benchmark\":\"kv_attention\",\"synthetic\":true,"
        "\"single_head\":true,\"warmups\":2,\"interleaved\":true,"
        "\"laplace_transforms_timed\":true,"
        "\"q2_baseline\":\"local_cpu_proxy_not_official_mlx_kernel\","
        "\"mlx_lm_commit\":\"15b522f593b7ca5fbc0cac6f7572d40859d2d8fe\","
        "\"mlx_core\":\"v0.31.2\"}");
    if (context) {
        benchmark(context, dim, trials);
    } else {
        for (int tokens : {512, 4096, 16384, 65536}) {
            benchmark(tokens, dim, trials);
        }
    }
}
