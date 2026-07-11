// Extract frozen, weight-only attention error metrics from a Qwen2 GGUF.
//
// Build from the repository root (the stubs below keep the research tool on
// the scalar runtime path while reusing GGUFContext and dequantize):
//   xcrun clang++ -std=c++20 -O3 -Wall -Wextra -Werror -Isrc \
//     research/laplace_kv/extract_weight_metrics.cpp \
//     src/gguf.cpp src/mmap.cpp src/matmul.cpp -framework Accelerate \
//     -framework IOKit -framework CoreFoundation -o /tmp/extract_weight_metrics

#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "gguf.h"
#include "kernels.h"
#include "matmul.h"

// matmul.cpp references the optional native and Metal dispatchers. This tool
// needs only its scalar dequantizer and scalar matmul validation path.
namespace Laplace::kernels {
gemm_fn get_simd_gemm() { return nullptr; }
moe_gemv_fn get_simd_moe_gemv() { return nullptr; }
moe_gemv_multi_fn get_simd_moe_gemv_multi() { return nullptr; }
}
namespace Laplace {
bool metal_gemv(const float*, const Tensor&, float*, int, int) { return false; }
bool metal_gemm(const float*, const Tensor&, float*, int, int, int) { return false; }
void metal_register_weights(const void*, size_t) {}
}

namespace {

using Matrix = std::vector<double>;

struct Header {
    char magic[8];
    uint32_t version;
    uint32_t layers;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t q_heads;
    uint32_t hidden;
    uint32_t context;
    uint32_t rope_dim;
    double rope_base;
};
static_assert(sizeof(Header) == 48);

struct RopeMeans {
    int half = 0;
    Matrix cc;
    Matrix ss;
    Matrix cs;
    Matrix sc;
};

struct Spectrum {
    Matrix values;
    Matrix vectors;
};

[[noreturn]] void fail(const std::string& message) {
    std::fprintf(stderr, "weight-metrics: %s\n", message.c_str());
    std::exit(1);
}

std::string tensor_name(int layer, const char* suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}

const Laplace::Tensor& require_tensor(const Laplace::GGUFContext& gguf,
                                      int layer, const char* suffix) {
    std::string name = tensor_name(layer, suffix);
    const Laplace::Tensor* tensor = gguf.find_tensor(name);
    if (!tensor) fail("missing " + name);
    return *tensor;
}

std::vector<float> decode(const Laplace::Tensor& tensor) {
    if (tensor.n_elements() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        fail(tensor.name + " is too large for Laplace::dequantize");
    std::vector<float> values(tensor.n_elements());
    Laplace::dequantize(tensor, values.data(), static_cast<int>(values.size()));
    for (float value : values)
        if (!std::isfinite(value)) fail(tensor.name + " decoded a non-finite value");
    return values;
}

void validate_runtime_row(const Laplace::Tensor& tensor,
                          const std::vector<float>& decoded, int input, int output) {
    std::vector<float> x(input), actual(output);
    for (int i = 0; i < input; ++i)
        x[i] = std::sin(0.013 * (i + 1)) + 0.25f * std::cos(0.031 * (i + 1));
    Laplace::matmul_row(x.data(), tensor, actual.data(), input, output);
    double worst = 0.0;
    for (int row = 0; row < output; ++row) {
        double expected = 0.0;
        const float* weights = decoded.data() + static_cast<size_t>(row) * input;
        for (int column = 0; column < input; ++column)
            expected += static_cast<double>(x[column]) * weights[column];
        worst = std::max(worst, std::abs(expected - actual[row]));
    }
    if (worst > 2e-3) fail(tensor.name + " dequant/matmul disagreement");
}

RopeMeans rope_means(int dimension, int positions, double base) {
    if (dimension <= 0 || dimension % 2 || positions <= 0)
        fail("RoPE averaging requires an even positive dimension and context");
    const int half = dimension / 2;
    Matrix cosine(static_cast<size_t>(positions) * half);
    Matrix sine(cosine.size());
    for (int position = 0; position < positions; ++position) {
        for (int pair = 0; pair < half; ++pair) {
            double frequency = std::pow(base, -2.0 * pair / dimension);
            double angle = position * frequency;
            cosine[static_cast<size_t>(position) * half + pair] = std::cos(angle);
            sine[static_cast<size_t>(position) * half + pair] = std::sin(angle);
        }
    }
    RopeMeans means{half, Matrix(half * half), Matrix(half * half),
                    Matrix(half * half), Matrix(half * half)};
    const double scale = 1.0 / positions;
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                half, half, positions, scale, cosine.data(), half,
                cosine.data(), half, 0.0, means.cc.data(), half);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                half, half, positions, scale, sine.data(), half,
                sine.data(), half, 0.0, means.ss.data(), half);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                half, half, positions, scale, cosine.data(), half,
                sine.data(), half, 0.0, means.cs.data(), half);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                half, half, positions, scale, sine.data(), half,
                cosine.data(), half, 0.0, means.sc.data(), half);
    return means;
}

Matrix rope_average(const Matrix& source, const RopeMeans& means) {
    const int n = means.half;
    const int dimension = 2 * n;
    if (source.size() != static_cast<size_t>(dimension * dimension))
        fail("internal RoPE matrix shape mismatch");
    Matrix result(source.size());
    auto at = [&](const Matrix& matrix, int row, int column) {
        return matrix[static_cast<size_t>(row) * n + column];
    };
    auto h = [&](int row, int column) {
        return source[static_cast<size_t>(row) * dimension + column];
    };
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double cc = at(means.cc, i, j), ss = at(means.ss, i, j);
            double cs = at(means.cs, i, j), sc = at(means.sc, i, j);
            double a = h(i, j), b = h(i, j + n);
            double c = h(i + n, j), d = h(i + n, j + n);
            result[static_cast<size_t>(i) * dimension + j] =
                cc * a - sc * c - cs * b + ss * d;
            result[static_cast<size_t>(i) * dimension + j + n] =
                cs * a - ss * c + cc * b - sc * d;
            result[static_cast<size_t>(i + n) * dimension + j] =
                sc * a + cc * c - ss * b - cs * d;
            result[static_cast<size_t>(i + n) * dimension + j + n] =
                ss * a + cs * c + sc * b + cc * d;
        }
    }
    return result;
}

void self_check_rope_average() {
    constexpr int dimension = 4;
    constexpr int positions = 17;
    constexpr double base = 10000.0;
    Matrix source = {3.0, 0.2, -0.4, 0.1,
                     0.2, 2.0, 0.3, -0.2,
                    -0.4, 0.3, 4.0, 0.5,
                     0.1, -0.2, 0.5, 1.5};
    Matrix fast = rope_average(source, rope_means(dimension, positions, base));
    Matrix direct(dimension * dimension);
    for (int position = 0; position < positions; ++position) {
        Matrix rotation(dimension * dimension);
        for (int pair = 0; pair < dimension / 2; ++pair) {
            double angle = position * std::pow(base, -2.0 * pair / dimension);
            double c = std::cos(angle), s = std::sin(angle);
            rotation[pair * dimension + pair] = c;
            rotation[pair * dimension + pair + dimension / 2] = -s;
            rotation[(pair + dimension / 2) * dimension + pair] = s;
            rotation[(pair + dimension / 2) * dimension + pair + dimension / 2] = c;
        }
        Matrix temp(dimension * dimension), conjugated(dimension * dimension);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    dimension, dimension, dimension, 1.0, rotation.data(), dimension,
                    source.data(), dimension, 0.0, temp.data(), dimension);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    dimension, dimension, dimension, 1.0, temp.data(), dimension,
                    rotation.data(), dimension, 1.0, conjugated.data(), dimension);
        for (size_t i = 0; i < direct.size(); ++i) direct[i] += conjugated[i] / positions;
    }
    double worst = 0.0;
    for (size_t i = 0; i < direct.size(); ++i)
        worst = std::max(worst, std::abs(direct[i] - fast[i]));
    if (worst > 1e-12) fail("RoPE coefficient self-check failed");
}

double validate_positive_definite(const Matrix& matrix, int dimension,
                                  double* symmetry_error) {
    Matrix lower(dimension * dimension);
    double max_diagonal = 0.0;
    *symmetry_error = 0.0;
    for (int row = 0; row < dimension; ++row) {
        max_diagonal = std::max(max_diagonal,
                                std::abs(matrix[static_cast<size_t>(row) * dimension + row]));
        for (int column = 0; column < dimension; ++column)
            *symmetry_error = std::max(*symmetry_error,
                std::abs(matrix[static_cast<size_t>(row) * dimension + column]
                       - matrix[static_cast<size_t>(column) * dimension + row]));
    }
    if (*symmetry_error > 1e-9 * std::max(1.0, max_diagonal))
        fail("metric is not symmetric");

    double minimum_pivot = std::numeric_limits<double>::infinity();
    for (int row = 0; row < dimension; ++row) {
        for (int column = 0; column <= row; ++column) {
            double value = 0.5 * (
                matrix[static_cast<size_t>(row) * dimension + column]
              + matrix[static_cast<size_t>(column) * dimension + row]);
            for (int k = 0; k < column; ++k)
                value -= lower[static_cast<size_t>(row) * dimension + k]
                       * lower[static_cast<size_t>(column) * dimension + k];
            if (row == column) {
                minimum_pivot = std::min(minimum_pivot, value);
                if (!(value > 0.0)) fail("metric failed unjittered Cholesky PSD check");
                lower[static_cast<size_t>(row) * dimension + column] = std::sqrt(value);
            } else {
                lower[static_cast<size_t>(row) * dimension + column] = value /
                    lower[static_cast<size_t>(column) * dimension + column];
            }
        }
    }
    return minimum_pivot;
}

void add_outer(Matrix* matrix, const float* vector, int dimension) {
    for (int row = 0; row < dimension; ++row)
        for (int column = 0; column < dimension; ++column)
            (*matrix)[static_cast<size_t>(row) * dimension + column] +=
                static_cast<double>(vector[row]) * vector[column];
}

void scale(Matrix* matrix, double value) {
    for (double& element : *matrix) element *= value;
}

double trace(const Matrix& matrix, int dimension) {
    double result = 0.0;
    for (int i = 0; i < dimension; ++i)
        result += matrix[static_cast<size_t>(i) * dimension + i];
    return result;
}

Spectrum spectrum(const Matrix& matrix, int dimension) {
    Matrix lapack = matrix;
    Matrix ascending(dimension);
    const __LAPACK_int n = dimension;
    const __LAPACK_int lda = dimension;
    __LAPACK_int info = 0;
    __LAPACK_int lwork = -1;
    double query = 0.0;
    const char jobz = 'V';
    const char uplo = 'U';
    dsyev_(&jobz, &uplo, &n, lapack.data(), &lda, ascending.data(),
           &query, &lwork, &info);
    if (info != 0) fail("LAPACK eigen workspace query failed");
    lwork = static_cast<__LAPACK_int>(query);
    Matrix work(static_cast<size_t>(lwork));
    dsyev_(&jobz, &uplo, &n, lapack.data(), &lda, ascending.data(),
           work.data(), &lwork, &info);
    if (info != 0) fail("LAPACK symmetric eigendecomposition failed");

    Spectrum result{Matrix(dimension), Matrix(dimension * dimension)};
    double total = 0.0;
    for (double value : ascending) total += value;
    if (!(total > 0.0)) fail("metric has non-positive trace");
    for (int component = 0; component < dimension; ++component) {
        int source = dimension - 1 - component;
        result.values[component] = ascending[source] / total;
        double* vector = result.vectors.data() + static_cast<size_t>(component) * dimension;
        for (int coordinate = 0; coordinate < dimension; ++coordinate)
            vector[coordinate] = lapack[static_cast<size_t>(source) * dimension + coordinate];
        int pivot = 0;
        for (int coordinate = 1; coordinate < dimension; ++coordinate)
            if (std::abs(vector[coordinate]) > std::abs(vector[pivot])) pivot = coordinate;
        if (vector[pivot] < 0.0)
            for (int coordinate = 0; coordinate < dimension; ++coordinate)
                vector[coordinate] = -vector[coordinate];
    }

    double worst = 0.0;
    for (int row = 0; row < dimension; ++row) {
        for (int column = 0; column < dimension; ++column) {
            double reconstructed = 0.0;
            for (int component = 0; component < dimension; ++component) {
                const double* vector = result.vectors.data()
                                     + static_cast<size_t>(component) * dimension;
                reconstructed += result.values[component] * vector[row] * vector[column];
            }
            worst = std::max(worst, std::abs(
                reconstructed - matrix[static_cast<size_t>(row) * dimension + column] / total));
        }
    }
    if (worst > 1e-11) fail("eigenbasis reconstruction check failed");
    return result;
}

void write_all(FILE* file, const void* data, size_t bytes) {
    if (std::fwrite(data, 1, bytes, file) != bytes) fail("failed to write output");
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf OUTPUT.lwm\n", argv[0]);
        return 2;
    }
    self_check_rope_average();

    Laplace::GGUFContext gguf;
    if (!gguf.open(argv[1])) fail("could not open GGUF");
    const auto& metadata = gguf.metadata();
    const std::string* architecture = Laplace::meta_str(metadata, "general.architecture");
    if (!architecture || *architecture != "qwen2") fail("requires a qwen2 GGUF");
    const std::string prefix = *architecture + ".";
    const int layers = static_cast<int>(Laplace::meta_int(metadata, (prefix + "block_count").c_str()));
    const int hidden = static_cast<int>(Laplace::meta_int(metadata, (prefix + "embedding_length").c_str()));
    const int q_heads = static_cast<int>(Laplace::meta_int(metadata, (prefix + "attention.head_count").c_str()));
    const int kv_heads = static_cast<int>(Laplace::meta_int(metadata, (prefix + "attention.head_count_kv").c_str()));
    const int head_dim = static_cast<int>(Laplace::meta_int(
        metadata, (prefix + "attention.key_length").c_str(), hidden / q_heads));
    const int context = static_cast<int>(Laplace::meta_int(metadata, (prefix + "context_length").c_str()));
    const int rope_dim = static_cast<int>(Laplace::meta_int(
        metadata, (prefix + "rope.dimension_count").c_str(), head_dim));
    const double rope_base = Laplace::meta_float(metadata, (prefix + "rope.freq_base").c_str(), 10000.0);
    if (layers <= 0 || hidden <= 0 || q_heads <= 0 || kv_heads <= 0 ||
        q_heads % kv_heads || head_dim != hidden / q_heads || rope_dim != head_dim)
        fail("unsupported or inconsistent attention metadata");
    const int group = q_heads / kv_heads;
    const int q_dimension = q_heads * head_dim;
    RopeMeans means = rope_means(rope_dim, context, rope_base);

    Header header{{'L','W','P','S','D','1','\0','\0'}, 2,
                  static_cast<uint32_t>(layers), static_cast<uint32_t>(kv_heads),
                  static_cast<uint32_t>(head_dim), static_cast<uint32_t>(q_heads),
                  static_cast<uint32_t>(hidden), static_cast<uint32_t>(context),
                  static_cast<uint32_t>(rope_dim), rope_base};
    FILE* output = std::fopen(argv[2], "wb");
    if (!output) fail("could not create output");
    write_all(output, &header, sizeof(header));

    std::vector<Matrix> value_metrics;
    value_metrics.reserve(static_cast<size_t>(layers) * kv_heads);
    std::array<std::vector<Matrix>, 2> eigenvalues;
    std::array<std::vector<Matrix>, 2> eigenvectors;
    for (int kind = 0; kind < 2; ++kind) {
        eigenvalues[kind].reserve(static_cast<size_t>(layers) * kv_heads);
        eigenvectors[kind].reserve(static_cast<size_t>(layers) * kv_heads);
    }
    double minimum_pivot = std::numeric_limits<double>::infinity();
    double maximum_symmetry_error = 0.0;
    double minimum_key_trace = std::numeric_limits<double>::infinity();
    double maximum_key_trace = 0.0;
    double minimum_value_trace = std::numeric_limits<double>::infinity();
    double maximum_value_trace = 0.0;

    for (int layer = 0; layer < layers; ++layer) {
        const Laplace::Tensor& q_tensor = require_tensor(gguf, layer, "attn_q.weight");
        const Laplace::Tensor& o_tensor = require_tensor(gguf, layer, "attn_output.weight");
        const Laplace::Tensor& norm_tensor = require_tensor(gguf, layer, "attn_norm.weight");
        const Laplace::Tensor& bias_tensor = require_tensor(gguf, layer, "attn_q.bias");
        if (gguf.find_tensor(tensor_name(layer, "attn_q_norm.weight")))
            fail("per-head Q RMSNorm is outside this frozen linear proxy");
        if (q_tensor.n_dims != 2 || q_tensor.dims[0] != static_cast<uint64_t>(hidden) ||
            q_tensor.dims[1] != static_cast<uint64_t>(q_dimension))
            fail(q_tensor.name + " violates arch_llama [input, output] shape");
        if (o_tensor.n_dims != 2 || o_tensor.dims[0] != static_cast<uint64_t>(q_dimension) ||
            o_tensor.dims[1] != static_cast<uint64_t>(hidden))
            fail(o_tensor.name + " violates arch_llama [input, output] shape");
        if (norm_tensor.n_dims != 1 || norm_tensor.dims[0] != static_cast<uint64_t>(hidden) ||
            bias_tensor.n_dims != 1 || bias_tensor.dims[0] != static_cast<uint64_t>(q_dimension))
            fail("attention norm or Q bias shape mismatch");

        std::vector<float> q = decode(q_tensor);
        std::vector<float> o = decode(o_tensor);
        std::vector<float> norm = decode(norm_tensor);
        std::vector<float> bias = decode(bias_tensor);
        if (layer == 0) {
            validate_runtime_row(q_tensor, q, hidden, q_dimension);
            validate_runtime_row(o_tensor, o, q_dimension, hidden);
            std::printf("decoder validation: %s and %s agree with scalar runtime matmul\n",
                        Laplace::type_name(q_tensor.type), Laplace::type_name(o_tensor.type));
        }
        std::printf("layer=%d Q=%s[%llu,%llu] O=%s[%llu,%llu] norm=%s bias=%s\n",
                    layer, Laplace::type_name(q_tensor.type),
                    static_cast<unsigned long long>(q_tensor.dims[0]),
                    static_cast<unsigned long long>(q_tensor.dims[1]),
                    Laplace::type_name(o_tensor.type),
                    static_cast<unsigned long long>(o_tensor.dims[0]),
                    static_cast<unsigned long long>(o_tensor.dims[1]),
                    Laplace::type_name(norm_tensor.type), Laplace::type_name(bias_tensor.type));

        std::vector<double> q_scaled(q.size());
        for (int row = 0; row < q_dimension; ++row)
            for (int column = 0; column < hidden; ++column)
                q_scaled[static_cast<size_t>(row) * hidden + column] =
                    static_cast<double>(q[static_cast<size_t>(row) * hidden + column]) * norm[column];
        std::vector<double> o_double(o.begin(), o.end());

        for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
            Matrix key(head_dim * head_dim);
            Matrix value(head_dim * head_dim);
            for (int local = 0; local < group; ++local) {
                const int q_head = kv_head * group + local;
                const double* q_block = q_scaled.data() + static_cast<size_t>(q_head) * head_dim * hidden;
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            head_dim, head_dim, hidden, 1.0, q_block, hidden,
                            q_block, hidden, 1.0, key.data(), head_dim);
                add_outer(&key, bias.data() + static_cast<size_t>(q_head) * head_dim, head_dim);

                const double* o_block = o_double.data() + q_head * head_dim;
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            head_dim, head_dim, hidden, 1.0, o_block, q_dimension,
                            o_block, q_dimension, 1.0, value.data(), head_dim);
            }
            // Squared score error includes (1/sqrt(D))^2. Keeping this factor
            // makes the raw K metric comparable before optional trace scaling.
            scale(&key, 1.0 / (group * head_dim));
            scale(&value, 1.0 / group);
            key = rope_average(key, means);

            double key_symmetry = 0.0, value_symmetry = 0.0;
            minimum_pivot = std::min(minimum_pivot,
                validate_positive_definite(key, head_dim, &key_symmetry));
            minimum_pivot = std::min(minimum_pivot,
                validate_positive_definite(value, head_dim, &value_symmetry));
            maximum_symmetry_error = std::max({maximum_symmetry_error,
                                               key_symmetry, value_symmetry});
            double key_trace = trace(key, head_dim);
            double value_trace = trace(value, head_dim);
            minimum_key_trace = std::min(minimum_key_trace, key_trace);
            maximum_key_trace = std::max(maximum_key_trace, key_trace);
            minimum_value_trace = std::min(minimum_value_trace, value_trace);
            maximum_value_trace = std::max(maximum_value_trace, value_trace);
            write_all(output, key.data(), key.size() * sizeof(double));
            Spectrum key_spectrum = spectrum(key, head_dim);
            Spectrum value_spectrum = spectrum(value, head_dim);
            eigenvalues[0].push_back(std::move(key_spectrum.values));
            eigenvectors[0].push_back(std::move(key_spectrum.vectors));
            eigenvalues[1].push_back(std::move(value_spectrum.values));
            eigenvectors[1].push_back(std::move(value_spectrum.vectors));
            value_metrics.push_back(std::move(value));
        }
    }
    for (const Matrix& value : value_metrics)
        write_all(output, value.data(), value.size() * sizeof(double));
    for (const auto& kind : eigenvalues)
        for (const Matrix& values : kind)
            write_all(output, values.data(), values.size() * sizeof(double));
    for (const auto& kind : eigenvectors)
        for (const Matrix& vectors : kind)
            write_all(output, vectors.data(), vectors.size() * sizeof(double));
    if (std::fclose(output) != 0) fail("failed to close output");

    std::printf("metadata: layers=%d hidden=%d Q=%d KV=%d group=%d D=%d "
                "context=%d rope_dim=%d rope_base=%.0f\n",
                layers, hidden, q_heads, kv_heads, group, head_dim,
                context, rope_dim, rope_base);
    std::printf("layout: metrics, eigenvalues, row-eigenvectors; each starts "
                "[key,value][layer][kv_head], header=%zu bytes\n", sizeof(Header));
    std::printf("validation: max_symmetry_error=%.3e min_cholesky_pivot=%.6e "
                "key_trace=[%.6e,%.6e] value_trace=[%.6e,%.6e]\n",
                maximum_symmetry_error, minimum_pivot,
                minimum_key_trace, maximum_key_trace,
                minimum_value_trace, maximum_value_trace);
    return 0;
}
