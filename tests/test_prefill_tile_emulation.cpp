// Lane-exact emulation of prefill_f16_tile (src/metal_library_sources.inc).
//
// The sandbox cannot compile Metal, so this test mirrors the kernel's index
// arithmetic, staging, bounds guards, accumulation order, and store pattern
// one-for-one in plain C++ and validates it against reference GEMMs on Linux.
// Every buffer access goes through a bounds-checking accessor: a tripped
// check means the kernel's guard logic would read out of bounds on device.
// Keep the two implementations in lockstep when editing either.

#include "fp16.h"
#include "prefill_tile.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using Laplace::fp16_to_fp32;
using Laplace::fp32_to_fp16;

namespace {

constexpr uint32_t kTile = kPrefillTileRows;

uint64_t g_out_of_bounds_attempts = 0;

float checked_read(const std::vector<float>& buffer, uint64_t index) {
    if (index >= buffer.size()) {
        ++g_out_of_bounds_attempts;
        return 0.0f;
    }
    return buffer[index];
}

float checked_read_f16(const std::vector<uint16_t>& buffer, uint64_t index) {
    if (index >= buffer.size()) {
        ++g_out_of_bounds_attempts;
        return 0.0f;
    }
    return fp16_to_fp32(buffer[index]);
}

void checked_write(std::vector<float>& buffer, uint64_t index, float value) {
    if (index >= buffer.size()) {
        ++g_out_of_bounds_attempts;
        return;
    }
    buffer[index] = value;
}

struct Problem {
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t K = 0;
    std::vector<uint16_t> weights;  // [N][K] f16 bits, row major
    std::vector<float> input;       // [M][K] f32, row major
};

Problem make_problem(uint32_t M, uint32_t N, uint32_t K, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> activation(-2.0f, 2.0f);
    std::uniform_real_distribution<float> weight(-1.0f, 1.0f);
    Problem problem;
    problem.M = M;
    problem.N = N;
    problem.K = K;
    problem.weights.reserve(static_cast<size_t>(N) * K);
    for (size_t i = 0; i < static_cast<size_t>(N) * K; ++i)
        problem.weights.push_back(fp32_to_fp16(weight(rng)));
    problem.input.reserve(static_cast<size_t>(M) * K);
    for (size_t i = 0; i < static_cast<size_t>(M) * K; ++i)
        problem.input.push_back(activation(rng));
    return problem;
}

// Reference 1: k-ascending fused multiply-add. This is the kernel's exact
// accumulation order (k-block outer, 128-bit group, element inner), so the
// emulation must match it bitwise.
std::vector<float> reference_fused(const Problem& problem) {
    const uint32_t M = problem.M, N = problem.N, K = problem.K;
    std::vector<float> output(static_cast<size_t>(M) * N);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
                acc = std::fma(problem.input[static_cast<size_t>(m) * K + k],
                               fp16_to_fp32(problem.weights[static_cast<size_t>(n) * K + k]),
                               acc);
            output[static_cast<size_t>(m) * N + n] = acc;
        }
    }
    return output;
}

// Reference 2: the legacy scalar kernel prefill_f16_rows (multiply then add,
// per-element weight loads), for numerics comparison on the widths it admits.
std::vector<float> reference_legacy_scalar(const Problem& problem) {
    const uint32_t M = problem.M, N = problem.N, K = problem.K;
    std::vector<float> output(static_cast<size_t>(M) * N);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
                sum += fp16_to_fp32(problem.weights[static_cast<size_t>(n) * K + k]) *
                       problem.input[static_cast<size_t>(m) * K + k];
            output[static_cast<size_t>(m) * N + n] = sum;
        }
    }
    return output;
}

// Reference 3: fp64 ground truth for magnitude sanity.
std::vector<double> reference_fp64(const Problem& problem) {
    const uint32_t M = problem.M, N = problem.N, K = problem.K;
    std::vector<double> output(static_cast<size_t>(M) * N);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            double sum = 0.0;
            for (uint32_t k = 0; k < K; ++k)
                sum += static_cast<double>(
                           fp16_to_fp32(problem.weights[static_cast<size_t>(n) * K + k])) *
                       static_cast<double>(problem.input[static_cast<size_t>(m) * K + k]);
            output[static_cast<size_t>(m) * N + n] = sum;
        }
    }
    return output;
}

// The emulation. Mirrors prefill_f16_tile phase by phase: per threadgroup
// (grid x = ceil(N/32) column tiles, y = ceil(M/32) row tiles), per k-block:
// cooperative activation stage (lane owns stage row `lane`), then every lane
// accumulates its column against all 32 staged rows, then guarded stores.
std::vector<float> emulate_prefill_f16_tile(const Problem& problem) {
    const uint32_t M = problem.M, N = problem.N, K = problem.K;
    std::vector<float> output(static_cast<size_t>(M) * N,
                              std::numeric_limits<float>::quiet_NaN());
    const uint32_t k_blocks = (K + kTile - 1) / kTile;
    for (uint32_t tg_y = 0; tg_y * kTile < M; ++tg_y) {
        for (uint32_t tg_x = 0; tg_x * kTile < N; ++tg_x) {
            const uint32_t row_start = tg_y * kTile;
            const uint32_t col_start = tg_x * kTile;
            float stage[kTile][kTile];
            float acc[kTile][kTile];  // acc[lane][row] within the tile
            for (uint32_t lane = 0; lane < kTile; ++lane)
                for (uint32_t r = 0; r < kTile; ++r) acc[lane][r] = 0.0f;
            for (uint32_t kb = 0; kb < k_blocks; ++kb) {
                const uint32_t k_base = kb * kTile;
                // Stage phase, all lanes (each lane owns row `lane`).
                for (uint32_t lane = 0; lane < kTile; ++lane) {
                    const uint64_t source_row =
                        static_cast<uint64_t>(row_start + lane) * K;
                    const bool row_valid = row_start + lane < M;
                    if (row_valid) {
                        for (uint32_t v = 0; v < 8; ++v) {
                            if (k_base + v * 4 + 4 <= K) {
                                for (uint32_t e = 0; e < 4; ++e)
                                    stage[lane][v * 4 + e] = checked_read(
                                        problem.input, source_row + k_base + v * 4 + e);
                            } else {
                                for (uint32_t e = 0; e < 4; ++e)
                                    stage[lane][v * 4 + e] =
                                        (k_base + v * 4 + e < K)
                                            ? checked_read(problem.input,
                                                           source_row + k_base + v * 4 + e)
                                            : 0.0f;
                            }
                        }
                    } else {
                        for (uint32_t t = 0; t < kTile; ++t) stage[lane][t] = 0.0f;
                    }
                }
                // Compute phase, all lanes (lane owns column col_start+lane).
                for (uint32_t lane = 0; lane < kTile; ++lane) {
                    const uint32_t column = col_start + lane;
                    const bool column_valid = column < N;
                    const uint64_t weight_row =
                        column_valid ? static_cast<uint64_t>(column) * K : 0;
                    float w[8][4];
                    if (column_valid) {
                        for (uint32_t v = 0; v < 8; ++v) {
                            if (k_base + v * 4 + 4 <= K) {
                                for (uint32_t e = 0; e < 4; ++e)
                                    w[v][e] = checked_read_f16(
                                        problem.weights, weight_row + k_base + v * 4 + e);
                            } else {
                                for (uint32_t e = 0; e < 4; ++e)
                                    w[v][e] = (k_base + v * 4 + e < K)
                                        ? checked_read_f16(problem.weights,
                                                           weight_row + k_base + v * 4 + e)
                                        : 0.0f;
                            }
                        }
                    } else {
                        for (uint32_t v = 0; v < 8; ++v)
                            for (uint32_t e = 0; e < 4; ++e) w[v][e] = 0.0f;
                    }
                    for (uint32_t r = 0; r < kTile; ++r) {
                        for (uint32_t v = 0; v < 8; ++v) {
                            for (uint32_t e = 0; e < 4; ++e)
                                acc[lane][r] = std::fma(stage[r][v * 4 + e], w[v][e],
                                                        acc[lane][r]);
                        }
                    }
                }
            }
            // Store phase.
            for (uint32_t lane = 0; lane < kTile; ++lane) {
                const uint32_t column = col_start + lane;
                if (column >= N) continue;
                for (uint32_t r = 0; r < kTile; ++r) {
                    const uint32_t row = row_start + r;
                    if (row < M)
                        checked_write(output, static_cast<uint64_t>(row) * N + column,
                                      acc[lane][r]);
                }
            }
        }
    }
    return output;
}

int g_failures = 0;

void expect(bool condition, const std::string& detail) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", detail.c_str());
        ++g_failures;
    }
}

void run_case(uint32_t M, uint32_t N, uint32_t K, uint32_t seed) {
    const Problem problem = make_problem(M, N, K, seed);
    const std::vector<float> emulated = emulate_prefill_f16_tile(problem);
    const std::vector<float> fused = reference_fused(problem);
    const std::vector<double> exact = reference_fp64(problem);
    const std::string label = "M=" + std::to_string(M) + " N=" + std::to_string(N) +
                              " K=" + std::to_string(K);
    expect(emulated.size() == fused.size(), label + " output size");
    // Every output element must be written exactly once (NaN prefill).
    for (size_t i = 0; i < emulated.size(); ++i)
        expect(!std::isnan(emulated[i]), label + " unwritten output at " + std::to_string(i));
    // Bitwise equality with the kernel's accumulation order.
    for (size_t i = 0; i < emulated.size(); ++i)
        expect(std::memcmp(&emulated[i], &fused[i], sizeof(float)) == 0,
               label + " bitwise drift vs fused reference at " + std::to_string(i) +
                   " (" + std::to_string(emulated[i]) + " vs " + std::to_string(fused[i]) + ")");
    // fp64 sanity: fp32 fused accumulation stays well inside fp16-product noise.
    for (size_t i = 0; i < emulated.size(); ++i) {
        const double scale = 1.0 + std::fabs(exact[i]);
        expect(std::fabs(static_cast<double>(emulated[i]) - exact[i]) < 1e-3 * scale,
               label + " fp64 drift at " + std::to_string(i));
    }
    // Parity with the legacy scalar kernel on the widths it admits.
    if (M <= 2) {
        const std::vector<float> legacy = reference_legacy_scalar(problem);
        for (size_t i = 0; i < emulated.size(); ++i) {
            const float scale = 1.0f + std::fabs(legacy[i]);
            expect(std::fabs(emulated[i] - legacy[i]) < 1e-4f * scale,
                   label + " legacy parity drift at " + std::to_string(i));
        }
    }
}

}  // namespace

int main() {
    // Edge geometries: tile boundaries off by one in every dimension, tails
    // in K (partial k-blocks, 128-bit groups), and single-element shapes.
    const uint32_t m_values[] = {1, 2, 3, 31, 32, 33, 64, 65, 96};
    const uint32_t n_values[] = {1, 2, 3, 31, 32, 33, 64};
    const uint32_t k_values[] = {8, 16, 24, 32, 40, 64, 72, 96, 136, 264};
    uint32_t seed = 1;
    for (uint32_t M : m_values)
        for (uint32_t N : n_values)
            for (uint32_t K : k_values)
                run_case(M, N, K, seed++);
    // K values that violate the declared K-multiple admission still exercise
    // the tail arithmetic here; hardware never reaches the kernel with them.
    const uint32_t k_unaligned[] = {1, 5, 7, 13, 33, 65};
    for (uint32_t M : {1u, 2u, 33u})
        for (uint32_t N : {1u, 33u})
            for (uint32_t K : k_unaligned)
                run_case(M, N, K, seed++);
    // A tall practical reduction width.
    run_case(32, 64, 2048, seed++);
    run_case(33, 31, 2048, seed++);

    expect(g_out_of_bounds_attempts == 0,
           "emulation performed " + std::to_string(g_out_of_bounds_attempts) +
               " out-of-bounds accesses");
    if (g_failures == 0) {
        std::printf("test_prefill_tile_emulation: all cases passed\n");
        return 0;
    }
    std::printf("test_prefill_tile_emulation: %d failures\n", g_failures);
    return 1;
}
