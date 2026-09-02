// Host sampling cost per generated token, across vocabulary sizes and
// truncation filters. Portable: no model file and no Metal required.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "sampler.h"

using namespace Laplace;

namespace {

struct XorShift32 {
    uint32_t state;
    explicit XorShift32(uint32_t s) : state(s ? s : 1u) {}
    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
    float next_float() { return static_cast<float>(next() & 0xFFFFFF) / 16777216.0f; }
};

volatile int sink = 0;

double bench_ms(const std::vector<float>& logits, const SamplerParams& params) {
    Sampler sampler(params);
    const int n = static_cast<int>(logits.size());
    for (int i = 0; i < 200; i++) sink = sampler.sample(logits.data(), n);
    constexpr int kSamples = 500;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kSamples; i++) sink = sampler.sample(logits.data(), n);
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / kSamples;
}

} // namespace

int main() {
    struct Config {
        const char* shape;
        int vocab;
        float temperature;
        int top_k;
        float top_p;
    };
    const Config configs[] = {
        {"peaked", 151936, 0.7f, 40, 0.9f},
        {"tail", 151936, 0.7f, 40, 0.9f},
        {"peaked", 151936, 0.7f, 0, 0.9f},
        {"peaked", 151936, 0.7f, 0, 1.0f},
        {"flat", 151936, 1.0f, 0, 0.9f},
        {"peaked", 151936, 0.0f, 40, 0.9f},
        {"peaked", 32768, 0.7f, 40, 0.9f},
        {"tail", 151936, 1.0f, 250, 0.95f},
    };
    for (const Config& config : configs) {
        std::vector<float> logits(static_cast<size_t>(config.vocab));
        XorShift32 gen(0xC0FFEEu);
        for (int i = 0; i < config.vocab; i++) {
            if (std::strcmp(config.shape, "peaked") == 0) {
                logits[i] = i < 64 ? 8.0f + gen.next_float() * 4.0f
                                   : gen.next_float() * 6.0f - 14.0f;
            } else if (std::strcmp(config.shape, "tail") == 0) {
                const double u = 1.0 - gen.next_float();
                logits[i] = static_cast<float>(-std::log(-std::log(u)) * 1.5 - 2.0);
            } else {
                logits[i] = 1.0f;
            }
        }
        SamplerParams params;
        params.temperature = config.temperature;
        params.top_k = config.top_k;
        params.top_p = config.top_p;
        params.seed = 12345;
        printf("%-6s vocab=%6d temp=%.2f top_k=%3d top_p=%.2f : %8.3f ms/sample\n",
               config.shape, config.vocab, config.temperature, config.top_k,
               config.top_p, bench_ms(logits, params));
    }
    return 0;
}
