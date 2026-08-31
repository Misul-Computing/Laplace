// Compare one mapped GGUF GEMV tensor against an independent scalar decode.
#include "../src/gguf.h"
#include "../src/matmul.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <vector>

namespace Laplace {
extern bool metal_gemv_repeat(const float* x, const Tensor& w, float* y,
                              int K, int N, int reps);
extern void metal_register_weights(const void* base, size_t size);
}

static double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s model.gguf tensor-name [repetitions-per-command-buffer]\n", argv[0]);
        return 1;
    }
    const int reps = argc >= 4 ? std::atoi(argv[3]) : 8;
    if (reps < 1) {
        fprintf(stderr, "repetitions must be positive\n");
        return 1;
    }

    Laplace::GGUFContext ctx;
    if (!ctx.open(argv[1])) {
        fprintf(stderr, "open failed\n");
        return 1;
    }
    const Laplace::Tensor* tensor = ctx.find_tensor(argv[2]);
    if (!tensor || !tensor->data) {
        fprintf(stderr, "missing tensor %s\n", argv[2]);
        return 1;
    }
    if (tensor->n_dims != 2 || tensor->dims[0] > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        tensor->dims[1] > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "tensor %s is not a supported two-dimensional GEMV matrix\n", argv[2]);
        return 1;
    }
    const int K = static_cast<int>(tensor->dims[0]);
    const int N = static_cast<int>(tensor->dims[1]);
    if (tensor->type != Laplace::GGMLType::Q4_K && tensor->type != Laplace::GGMLType::Q6_K) {
        fprintf(stderr, "tensor %s has unsupported microbenchmark format %s\n",
                argv[2], Laplace::type_name(tensor->type));
        return 1;
    }
    const size_t elements_per_block = Laplace::elements_per_block(tensor->type);
    const size_t bytes_per_block = Laplace::bytes_per_block(tensor->type);
    if (elements_per_block == 0 || bytes_per_block == 0 || K % static_cast<int>(elements_per_block) != 0) {
        fprintf(stderr, "tensor %s has unsupported type/layout %s for exact mapped GEMV\n",
                argv[2], Laplace::type_name(tensor->type));
        return 1;
    }
    const size_t blocks_per_row = static_cast<size_t>(K) / elements_per_block;
    if (blocks_per_row > std::numeric_limits<size_t>::max() / bytes_per_block ||
        static_cast<size_t>(N) > std::numeric_limits<size_t>::max() / (blocks_per_row * bytes_per_block)) {
        fprintf(stderr, "tensor %s byte count overflows\n", argv[2]);
        return 1;
    }
    const size_t expected_bytes = static_cast<size_t>(N) * blocks_per_row * bytes_per_block;
    const uintptr_t base = reinterpret_cast<uintptr_t>(ctx.file_data());
    const uintptr_t data = reinterpret_cast<uintptr_t>(tensor->data);
    if (data < base || data - base > ctx.file_size() || expected_bytes > ctx.file_size() - (data - base) ||
        tensor->nbytes() != expected_bytes) {
        fprintf(stderr, "tensor %s does not occupy its exact mapped GGUF byte span\n", argv[2]);
        return 1;
    }
    const size_t offset = static_cast<size_t>(data - base);
    fprintf(stderr, "%s type=%s K=%d N=%d mapped_offset=%zu mapped_bytes=%zu\n",
            argv[2], Laplace::type_name(tensor->type), K, N, offset, expected_bytes);

    Laplace::metal_register_weights(ctx.file_data(), ctx.file_size());
    std::vector<float> input(K), scalar(N), metal(N);
    for (int index = 0; index != K; ++index)
        input[index] = static_cast<float>((index * 17) % 200 - 100) / 10.0f;

    // This scalar path intentionally shares no Metal unpack helper.
    setenv("LAPLACE_NOSIMD", "1", 1);
    Laplace::matmul_rows(input.data(), *tensor, scalar.data(), 1, K, N);
    unsetenv("LAPLACE_NOSIMD");
    const auto run_repeat = [&] (int count) {
        return Laplace::metal_gemv_repeat(input.data(), *tensor, metal.data(), K, N, count);
    };
    if (!run_repeat(1)) {
        fprintf(stderr, "one-command-buffer Metal GEMV is unsupported\n");
        return 1;
    }
    float max_error = 0.0f, max_value = 0.0f;
    for (int row = 0; row != N; ++row) {
        max_error = fmaxf(max_error, fabsf(scalar[row] - metal[row]));
        max_value = fmaxf(max_value, fabsf(scalar[row]));
    }
    const float relative_error = max_value > 0.0f ? max_error / max_value : 0.0f;
    fprintf(stderr, "scalar-metal relative_error=%.6e max_value=%.6f\n", relative_error, max_value);
    if (!std::isfinite(relative_error) || relative_error > 1.0e-2f) return 1;

    // One warm command, then five independently submitted, repeated-kernel
    // samples. Each measured sample has exactly one command buffer.
    if (!run_repeat(reps)) {
        fprintf(stderr, "warm repeated Metal GEMV failed\n");
        return 1;
    }
    Laplace::metal_dispatch_metrics_reset();
    std::vector<double> wall_ms, gpu_ms;
    for (int sample = 0; sample != 5; ++sample) {
        const Laplace::MetalDispatchMetrics before = Laplace::metal_dispatch_metrics();
        const auto start = std::chrono::steady_clock::now();
        const bool ok = run_repeat(reps);
        const auto end = std::chrono::steady_clock::now();
        const Laplace::MetalDispatchMetrics after = Laplace::metal_dispatch_metrics();
        if (!ok || after.command_buffers != before.command_buffers + 1) {
            fprintf(stderr, "sample %d did not complete exactly one command buffer\n", sample);
            return 1;
        }
        wall_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count() / reps);
        if (after.gpu_time_ms > before.gpu_time_ms)
            gpu_ms.push_back((after.gpu_time_ms - before.gpu_time_ms) / reps);
    }
    const Laplace::MetalDispatchMetrics metrics = Laplace::metal_dispatch_metrics();
    const auto range = std::minmax_element(wall_ms.begin(), wall_ms.end());
    const double wall_median_ms = median(wall_ms);
    const double requested_gbps = static_cast<double>(expected_bytes) / wall_median_ms / 1.0e6;
    fprintf(stderr, "repeat reps=%d samples=5 command_buffers=%llu wall_ms_per_gemv median=%.3f range=[%.3f,%.3f] requested_mapped_GBps=%.3f\n",
            reps, static_cast<unsigned long long>(metrics.command_buffers), wall_median_ms,
            *range.first, *range.second, requested_gbps);
    if (gpu_ms.size() == wall_ms.size()) {
        const auto gpu_range = std::minmax_element(gpu_ms.begin(), gpu_ms.end());
        const double gpu_median_ms = median(gpu_ms);
        fprintf(stderr, "gpu_ms_per_gemv median=%.3f range=[%.3f,%.3f] requested_mapped_GBps=%.3f\n",
                gpu_median_ms, *gpu_range.first, *gpu_range.second,
                static_cast<double>(expected_bytes) / gpu_median_ms / 1.0e6);
    } else {
        fprintf(stderr, "gpu timestamp unavailable; requested_mapped_GBps is host-wait effective throughput\n");
    }
    return 0;
}
