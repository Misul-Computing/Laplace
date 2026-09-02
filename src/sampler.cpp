// sampler.cpp - token sampling
#include "sampler.h"

#include <algorithm>
#include <cmath>

namespace Laplace {

int Sampler::remember(int token) {
    if (token == last_token_) {
        repeat_run_++;
    } else {
        last_token_ = token;
        repeat_run_ = 1;
    }
    return token;
}

int Sampler::sample(const float* logits, int n) {
    const int blocked = repeat_run_ >= 8 && n > 1 ? last_token_ : -1;

    // Greedy
    if (params_.temperature <= 0.0f) {
        int best = blocked == 0 && n > 1 ? 1 : 0;
        float bestv = logits[best];
        for (int i = 0; i < n; i++) {
            if (i == blocked) continue;
            if (logits[i] > bestv) { bestv = logits[i]; best = i; }
        }
        return remember(best);
    }

    // Softmax with temperature. probs_ holds unnormalized exponentials
    // until the candidate set is known: only candidates are read past
    // selection, so only they are divided by the sum.
    probs_.resize(n);
    if (blocked >= 0) probs_[blocked] = 0.0f;
    int first = blocked == 0 && n > 1 ? 1 : 0;
    float maxv = logits[first];
    for (int i = 0; i < n; i++) {
        if (i != blocked && logits[i] > maxv) maxv = logits[i];
    }
    float invT = 1.0f / params_.temperature;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        if (i == blocked) continue;
        probs_[i] = std::exp((logits[i] - maxv) * invT);
        sum += probs_[i];
    }

    // Candidates are ranked by probability (descending) with the token id
    // as tie-break. The total order makes the candidate sequence identical
    // for any selection strategy.
    const auto ranked_before = [this](int a, int b) {
        if (probs_[a] != probs_[b]) return probs_[a] > probs_[b];
        return a < b;
    };

    int n_cand = n;
    const bool top_k_active = params_.top_k > 0 && params_.top_k < n;
    const bool top_p_active = params_.top_p < 1.0f;

    if (top_k_active || top_p_active) {
        // Bounded selection instead of a full-vocabulary sort: keep the m
        // best-ranked tokens in a heap of size m, then order just those.
        // m is the top-k size, or a nucleus window that grows only when
        // the distribution is flat enough to need more mass. The heap is
        // ordered by ranked_before, so its root is the worst kept token
        // and pop_heap evicts exactly that.
        int m = top_k_active ? params_.top_k : std::min(n, 256);
        const auto order_all = [&]() {
            indices_.resize(static_cast<size_t>(n));
            for (int i = 0; i < n; i++) indices_[i] = i;
            std::sort(indices_.begin(), indices_.end(), ranked_before);
        };
        for (;;) {
            if (m >= n) {
                // Full window (small vocabulary or flat distribution):
                // ordering everything directly beats heap-filling it.
                order_all();
                break;
            }
            indices_.clear();
            indices_.reserve(static_cast<size_t>(m));
            for (int i = 0; i < n; i++) {
                if (static_cast<int>(indices_.size()) < m) {
                    indices_.push_back(i);
                    std::push_heap(indices_.begin(), indices_.end(), ranked_before);
                } else if (ranked_before(i, indices_.front())) {
                    std::pop_heap(indices_.begin(), indices_.end(), ranked_before);
                    indices_.back() = i;
                    std::push_heap(indices_.begin(), indices_.end(), ranked_before);
                }
            }
            std::sort(indices_.begin(), indices_.end(), ranked_before);
            if (top_k_active) break;
            float mass = 0.0f;
            for (int i = 0; i < m; i++) mass += probs_[indices_[i]];
            const float covered = mass / sum;
            if (covered >= params_.top_p) break;
            // A window holding this little mass is a near-flat
            // distribution: take the full ordering in one pass instead of
            // climbing the ladder.
            m = (covered < 0.25f || m >= 16384) ? n : std::min(n, m * 8);
        }
        n_cand = m;
        for (int i = 0; i < n_cand; i++) probs_[indices_[i]] /= sum;
    } else {
        // No filter: every token is a candidate, in id order.
        indices_.resize(n);
        for (int i = 0; i < n; i++) indices_[i] = i;
        for (int i = 0; i < n; i++) probs_[i] /= sum;
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
        if (r < acc) return remember(indices_[i]);
    }
    return remember(indices_[n_cand - 1]);
}

} // namespace Laplace
