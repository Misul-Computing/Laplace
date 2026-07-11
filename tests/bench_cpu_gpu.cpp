// bench_cpu_gpu.cpp - per-kernel CPU vs GPU timing for model matmul sizes
#include "../src/tensor.h"
#include "../src/kernels.h"
#include "../src/matmul.h"
#include "../src/fp16.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>

using namespace Laplace;
using namespace Laplace::kernels;

namespace Laplace {
    extern bool metal_available();
    extern bool metal_gemv(const float* x, const Tensor& w, float* y, int K, int N);
    extern bool metal_gemm(const float* x, const Tensor& w, float* y, int M, int K, int N);
}

static void rng_fill(float* v, int n, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < n; i++) { s = s*1103515245 + 12345; v[i] = (float)((s >> 16) % 200 - 100) / 10.0f; }
}

static uint16_t f2h(float f) { return fp32_to_fp16(f); }

static void pack_q4_K(const float* w, int K, int N, uint8_t* out) {
    int nb = K/256; auto blk = (block_q4_K*)out;
    unsigned rs = 999;
    for (int j = 0; j < N; j++) for (int b = 0; b < nb; b++) {
        const float* x = w + j*K + b*256; float amax = 0;
        for (int i = 0; i < 256; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax/15; auto& bb = blk[(uint64_t)j*nb+b]; bb.d = f2h(d); bb.dmin = f2h(amax/240);
        for (int s = 0; s < 12; s++) { rs = rs*1103515245+12345; bb.scales[s] = (uint8_t)(rs >> 16); }
        float id = d > 0 ? 1/d : 0;
        for (int i = 0; i < 128; i++) {
            int q0 = roundf(x[i]*id), q1 = roundf(x[i+128]*id);
            bb.qs[i] = (q0&0xF)|((q1&0xF)<<4);
        }
    }
}

int main() {
    if (!metal_available()) { fprintf(stderr, "SKIP: no Metal\n"); return 0; }

    // Actual matmul sizes from Gemma4 26B (Q4_K weights, M=1 decode)
    struct Size { const char* name; int K, N; GGMLType type; };
    Size sizes[] = {
        {"QKV proj",      2816,  8192, GGMLType::Q4_K},
        {"O proj",        8192,  2816, GGMLType::Q4_K},
        {"FFN gate",      2816, 16384, GGMLType::Q4_K},
        {"FFN up",        2816, 16384, GGMLType::Q4_K},
        {"FFN down",     16384,  2816, GGMLType::Q4_K},
        {"MoE router",    2816,   128, GGMLType::F32},
        {"Expert gate_up",2816,  1408, GGMLType::Q4_K},
        {"Expert down",    704,  2816, GGMLType::Q4_K},
        {"LM head",       2816,262144, GGMLType::Q4_K},
    };

    const int ITERS = 50;
    printf("%-16s  %8s  %8s  %8s  %s\n", "kernel", "CPU(ms)", "GPU(ms)", "ratio", "winner");
    printf("%-16s  %8s  %8s  %8s  %s\n", "------", "-------", "-------", "-----", "------");

    for (auto& sz : sizes) {
        int K = sz.K, N = sz.N;
        std::vector<float> x(K), y_cpu(N), y_gpu(N);
        rng_fill(x.data(), K, 42);

        // Pack weights
        size_t bpb = bytes_per_block(sz.type), epb = elements_per_block(sz.type);
        size_t nblocks = ((size_t)K*N + epb - 1) / epb;
        std::vector<uint8_t> storage(nblocks * bpb, 0);
        std::vector<float> w(K * N);
        rng_fill(w.data(), K*N, 123);

        Tensor wt; wt.type = sz.type; wt.n_dims = 2; wt.dims[0] = K; wt.dims[1] = N; wt.data = storage.data();
        if (sz.type == GGMLType::F32) memcpy(storage.data(), w.data(), (size_t)K*N*4);
        else if (sz.type == GGMLType::Q4_K) pack_q4_K(w.data(), K, N, storage.data());

        // Warmup CPU
        setenv("LAPLACE_NOMETAL", "1", 1);
        matmul_rows(x.data(), wt, y_cpu.data(), 1, K, N);
        unsetenv("LAPLACE_NOMETAL");

        // Warmup GPU
        metal_gemv(x.data(), wt, y_gpu.data(), K, N);

        // Bench CPU
        setenv("LAPLACE_NOMETAL", "1", 1);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < ITERS; i++)
            matmul_rows(x.data(), wt, y_cpu.data(), 1, K, N);
        auto t1 = std::chrono::steady_clock::now();
        unsetenv("LAPLACE_NOMETAL");
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;

        // Bench GPU
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < ITERS; i++)
            metal_gemv(x.data(), wt, y_gpu.data(), K, N);
        t1 = std::chrono::steady_clock::now();
        double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;

        double ratio = gpu_ms / cpu_ms;
        const char* winner = ratio < 0.9 ? "GPU" : (ratio > 1.1 ? "CPU" : "tie");
        printf("%-16s  %8.2f  %8.2f  %8.2f  %s\n", sz.name, cpu_ms, gpu_ms, ratio, winner);
    }
    return 0;
}
