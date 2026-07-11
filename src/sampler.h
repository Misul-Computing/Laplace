// sampler.h - token sampling with temperature, top-k, top-p
#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace Laplace {

struct SamplerParams {
    float temperature = 1.0f;  // 0 = greedy
    int   top_k       = 0;     // 0 = off
    float top_p       = 1.0f;  // 1.0 = off
    uint32_t seed     = 0;     // 0 = nondeterministic
};

class Sampler {
public:
    explicit Sampler(SamplerParams p)
        : params_(p),
          rng_(p.seed != 0 ? p.seed : std::random_device{}()) {}

    // Sample one token from logits. Greedy if temperature == 0.
    int sample(const float* logits, int n);

private:
    SamplerParams params_;
    std::mt19937 rng_;
    std::vector<float> probs_;
    std::vector<int>   indices_;
};

} // namespace Laplace
