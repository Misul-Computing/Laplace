// test_util.h - minimal check macros for Laplace unit tests
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        g_checks++;                                                        \
        if (!(cond)) {                                                     \
            g_failures++;                                                  \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);\
        }                                                                  \
    } while (0)

#define CHECK_MSG(cond, ...)                                               \
    do {                                                                   \
        g_checks++;                                                        \
        if (!(cond)) {                                                     \
            g_failures++;                                                  \
            fprintf(stderr, "FAIL %s:%d: %s  (", __FILE__, __LINE__, #cond); \
            fprintf(stderr, __VA_ARGS__);                                  \
            fprintf(stderr, ")\n");                                        \
        }                                                                  \
    } while (0)

// Relative comparison for float accumulations (SIMD reorder tolerance).
inline bool almost_equal(float a, float b, float rel = 1e-3f, float abs_tol = 1e-4f) {
    float diff = std::fabs(a - b);
    if (diff <= abs_tol) return true;
    float scale = std::fmax(std::fabs(a), std::fabs(b));
    return diff <= rel * scale;
}

inline int test_summary(const char* name) {
    if (g_failures == 0) {
        printf("%s: OK (%d checks)\n", name, g_checks);
        return 0;
    }
    printf("%s: %d/%d checks FAILED\n", name, g_failures, g_checks);
    return 1;
}

// Deterministic PRNG (don't depend on libc rand).
struct XorShift32 {
    uint32_t s;
    explicit XorShift32(uint32_t seed) : s(seed ? seed : 0xdeadbeefu) {}
    uint32_t next() {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return s;
    }
    // uniform in [-1, 1)
    float next_float() {
        return (next() >> 8) * (2.0f / 16777216.0f) - 1.0f;
    }
    uint8_t next_byte() { return static_cast<uint8_t>(next() & 0xff); }
};
