// sampler.cpp - token sampling
#include "sampler.h"

#include <algorithm>
#include <cmath>

namespace Laplace {

int Sampler::sample(const float* logits, int n) {
    // Greedy
    if (params_.temperature <= 0.0f) {
        int best = 0;
        float bestv = logits[0];
        for (int i = 1; i < n; i++) {
            if (logits[i] > bestv) { bestv = logits[i]; best = i; }
        }
        return best;
    }

    // Softmax with temperature.
    probs_.assign(n, 0.0f);
    float maxv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
    float invT = 1.0f / params_.temperature;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs_[i] = std::exp((logits[i] - maxv) * invT);
        sum += probs_[i];
    }
    for (int i = 0; i < n; i++) probs_[i] /= sum;

    // Candidate token IDs, sorted by probability (descending) when a
    // truncation filter is active.
    indices_.resize(n);
    for (int i = 0; i < n; i++) indices_[i] = i;
    int n_cand = n;

    bool need_sort = (params_.top_k > 0 && params_.top_k < n) || params_.top_p < 1.0f;
    if (need_sort) {
        std::sort(indices_.begin(), indices_.end(),
            [&](int a, int b) { return probs_[a] > probs_[b]; });
    }

    // Top-k: keep only the k most probable token IDs.
    if (params_.top_k > 0 && params_.top_k < n_cand) {
        n_cand = params_.top_k;
    }

    // Top-p (nucleus): keep the smallest prefix whose mass reaches top_p.
    if (params_.top_p < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < n_cand; i++) {
            cum += probs_[indices_[i]];
            if (cum >= params_.top_p) { n_cand = i + 1; break; }
        }
    }

    // Renormalize over the candidate set and sample from it.
    float csum = 0.0f;
    for (int i = 0; i < n_cand; i++) csum += probs_[indices_[i]];
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_) * csum;
    float acc = 0.0f;
    for (int i = 0; i < n_cand; i++) {
        acc += probs_[indices_[i]];
        if (r < acc) return indices_[i];
    }
    return indices_[n_cand - 1];
}

} // namespace Laplace
