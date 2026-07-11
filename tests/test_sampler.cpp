// test_sampler - greedy argmax, top-k support restriction, top-p restriction,
// and seed reproducibility.

#include <set>
#include <vector>

#include "sampler.h"
#include "test_util.h"

using namespace Laplace;

namespace {

void test_greedy() {
    SamplerParams p;
    p.temperature = 0.0f;
    Sampler s(p);
    std::vector<float> logits(100, 0.0f);
    logits[42] = 5.0f;
    CHECK(s.sample(logits.data(), 100) == 42);
}

void test_top_k_support() {
    // Token IDs 90..99 have overwhelmingly higher logits. With top_k = 10,
    // every sample must come from {90..99} — never from IDs 0..89.
    SamplerParams p;
    p.temperature = 1.0f;
    p.top_k = 10;
    p.top_p = 1.0f;
    p.seed = 1234;
    Sampler s(p);

    std::vector<float> logits(100, 0.0f);
    for (int i = 90; i < 100; i++) logits[i] = 10.0f;

    int outside = 0;
    std::set<int> seen;
    for (int i = 0; i < 500; i++) {
        int id = s.sample(logits.data(), 100);
        if (id < 90) outside++;
        seen.insert(id);
    }
    CHECK_MSG(outside == 0, "%d/500 samples outside the top-k set", outside);
    // The 10 candidates are equiprobable; 500 draws must hit more than one.
    CHECK_MSG(seen.size() > 1, "RNG stuck: only %zu distinct samples", seen.size());
}

void test_top_p_support() {
    // One token holds ~97% of the mass. top_p = 0.5 must restrict to it.
    SamplerParams p;
    p.temperature = 1.0f;
    p.top_k = 0;
    p.top_p = 0.5f;
    p.seed = 99;
    Sampler s(p);

    std::vector<float> logits(50, 0.0f);
    logits[7] = 8.0f;
    for (int i = 0; i < 200; i++) {
        int id = s.sample(logits.data(), 50);
        CHECK_MSG(id == 7, "top-p sampled %d, expected 7", id);
        if (id != 7) break;
    }
}

void test_seed_reproducible() {
    SamplerParams p;
    p.temperature = 1.0f;
    p.top_k = 0;
    p.top_p = 1.0f;
    p.seed = 777;

    std::vector<float> logits(100);
    XorShift32 rng(5);
    for (auto& v : logits) v = rng.next_float() * 3.0f;

    Sampler a(p), b(p);
    bool same = true;
    for (int i = 0; i < 50; i++) {
        if (a.sample(logits.data(), 100) != b.sample(logits.data(), 100)) same = false;
    }
    CHECK_MSG(same, "same seed produced different sequences");

    // And the sequence itself must not be constant.
    Sampler c(p);
    std::set<int> seen;
    for (int i = 0; i < 50; i++) seen.insert(c.sample(logits.data(), 100));
    CHECK_MSG(seen.size() > 1, "seeded RNG produced a constant sequence");
}

} // namespace

int main() {
    test_greedy();
    test_top_k_support();
    test_top_p_support();
    test_seed_reproducible();
    return test_summary("test_sampler");
}
