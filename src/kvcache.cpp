#include "kvcache.h"

#include "fp16.h"
#include "ops.h"

#include <algorithm>
#if defined(LAPLACE_KV_CAPTURE)
#include <cmath>
#include <limits>
#endif
#include <cstring>

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_neon.h>
#endif

namespace Laplace {

namespace {

void to_f16(const float* source, uint16_t* destination, int size) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    int index = 0;
    for (; index + 4 <= size; index += 4) {
        vst1_f16(reinterpret_cast<__fp16*>(destination + index),
                 vcvt_f16_f32(vld1q_f32(source + index)));
    }
    for (; index < size; index++) {
        destination[index] = fp32_to_fp16(source[index]);
    }
#else
    for (int index = 0; index < size; index++) {
        destination[index] = fp32_to_fp16(source[index]);
    }
#endif
}

void from_f16(const uint16_t* source, float* destination, int size) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    int index = 0;
    for (; index + 4 <= size; index += 4) {
        vst1q_f32(destination + index,
                  vcvt_f32_f16(vld1_f16(
                      reinterpret_cast<const __fp16*>(source + index))));
    }
    for (; index < size; index++) {
        destination[index] = fp16_to_fp32(source[index]);
    }
#else
    for (int index = 0; index < size; index++) {
        destination[index] = fp16_to_fp32(source[index]);
    }
#endif
}

#if defined(LAPLACE_KV_CAPTURE)
float half_round(float value) {
    return fp16_to_fp32(fp32_to_fp16(value));
}

double standard_deviation(const std::vector<double>& matrix, int rows,
                          int columns, bool column, int index) {
    const int count = column ? rows : columns;
    double mean = 0.0;
    for (int item = 0; item < count; item++) {
        int row = column ? item : index;
        int col = column ? index : item;
        mean += matrix[static_cast<size_t>(row) * columns + col];
    }
    mean /= count;
    double variance = 0.0;
    for (int item = 0; item < count; item++) {
        int row = column ? item : index;
        int col = column ? index : item;
        double delta = matrix[static_cast<size_t>(row) * columns + col] - mean;
        variance += delta * delta;
    }
    return std::sqrt(variance / std::max(1, count - 1));
}

double imbalance(const std::vector<double>& matrix, int rows, int columns) {
    double row_min = std::numeric_limits<double>::infinity();
    double row_max = 0.0;
    double column_min = std::numeric_limits<double>::infinity();
    double column_max = 0.0;
    for (int row = 0; row < rows; row++) {
        double value = standard_deviation(matrix, rows, columns, false, row);
        row_min = std::min(row_min, value);
        row_max = std::max(row_max, value);
    }
    for (int column = 0; column < columns; column++) {
        double value = standard_deviation(
            matrix, rows, columns, true, column);
        column_min = std::min(column_min, value);
        column_max = std::max(column_max, value);
    }
    return row_max / std::max(row_min, 1e-8)
         + column_max / std::max(column_min, 1e-8);
}

void compact_scale_field(std::vector<float>& values, int bits) {
    auto [low_iterator, high_iterator] = std::minmax_element(
        values.begin(), values.end());
    float low = half_round(*low_iterator);
    int maximum = (1 << bits) - 1;
    float step = half_round(
        (*high_iterator - *low_iterator) / maximum);
    step = std::max(step, std::ldexp(1.0f, -24));
    for (float& value : values) {
        int code = static_cast<int>(std::nearbyint((value - low) / step));
        value = std::clamp(code, 0, maximum) * step + low;
    }
}

float binary_float_round(float value, int minimum_exponent,
                         int maximum_exponent, int mantissa_bits,
                         bool signed_value) {
    float sign = signed_value && std::signbit(value) ? -1.0f : 1.0f;
    float magnitude = signed_value ? std::abs(value) : std::max(value, 0.0f);
    float minimum_normal = std::ldexp(1.0f, minimum_exponent);
    float step = std::ldexp(1.0f, minimum_exponent - mantissa_bits);
    if (magnitude >= minimum_normal) {
        int exponent = std::ilogb(magnitude);
        exponent = std::clamp(
            exponent, minimum_exponent, maximum_exponent);
        step = std::ldexp(1.0f, exponent - mantissa_bits);
    }
    float maximum = (2.0f - std::ldexp(1.0f, -mantissa_bits))
                  * std::ldexp(1.0f, maximum_exponent);
    return sign * std::min(std::nearbyint(magnitude / step) * step, maximum);
}

void polarity_fp8_field(std::vector<float>& values, bool signed_value) {
    for (float& value : values) {
        value = signed_value
              ? binary_float_round(value, -6, 7, 3, true)
              : binary_float_round(value, -14, 15, 3, false);
    }
}

void variance_normalized_rtn(std::vector<double>& matrix, int rows,
                             int columns, int bits, int metadata_bits) {
    const std::vector<double> original = matrix;
    std::vector<double> log_rows(rows, 0.0);
    std::vector<double> log_columns(columns, 0.0);
    std::vector<double> best_rows = log_rows;
    std::vector<double> best_columns = log_columns;

    auto rebuild = [&] {
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                matrix[static_cast<size_t>(row) * columns + column] =
                    original[static_cast<size_t>(row) * columns + column]
                    / std::exp(log_rows[row] + log_columns[column]);
            }
        }
    };

    rebuild();
    double best_imbalance = imbalance(matrix, rows, columns);
    for (int iteration = 0; iteration < 8; iteration++) {
        for (int column = 0; column < columns; column++) {
            double deviation = std::clamp(
                standard_deviation(matrix, rows, columns, true, column),
                1e-3, 1e3);
            log_columns[column] = std::clamp(
                log_columns[column] + std::log(deviation), -0.3, 10.0);
        }
        rebuild();
        for (int row = 0; row < rows; row++) {
            double deviation = std::clamp(
                standard_deviation(matrix, rows, columns, false, row),
                1e-3, 1e3);
            log_rows[row] = std::clamp(
                log_rows[row] + std::log(deviation), -0.3, 10.0);
        }
        rebuild();
        double candidate = imbalance(matrix, rows, columns);
        if (candidate <= best_imbalance) {
            best_imbalance = candidate;
            best_rows = log_rows;
            best_columns = log_columns;
        }
    }

    std::vector<float> column_scale(columns);
    std::vector<float> absorbed_scale(rows);
    std::vector<float> absorbed_zero(rows);
    std::vector<double> steps(rows);
    std::vector<double> lows(rows);
    std::vector<int> maxima(rows);
    for (int column = 0; column < columns; column++) {
        column_scale[column] = static_cast<float>(
            std::exp(best_columns[column]));
    }
    for (int row = 0; row < rows; row++) {
        int row_bits = bits == 5 ? 3 + (row & 1) : bits;
        maxima[row] = (1 << row_bits) - 1;
        double low = std::numeric_limits<double>::infinity();
        double high = -std::numeric_limits<double>::infinity();
        double row_scale = std::exp(best_rows[row]);
        for (int column = 0; column < columns; column++) {
            double value = original[static_cast<size_t>(row) * columns + column]
                         / row_scale / std::exp(best_columns[column]);
            matrix[static_cast<size_t>(row) * columns + column] = value;
            low = std::min(low, value);
            high = std::max(high, value);
        }
        steps[row] = std::max((high - low) / maxima[row], 1e-10);
        lows[row] = low;
        absorbed_scale[row] = static_cast<float>(row_scale * steps[row]);
        absorbed_zero[row] = static_cast<float>(row_scale * low);
    }
    if (metadata_bits == 9) {
        polarity_fp8_field(column_scale, false);
        polarity_fp8_field(absorbed_scale, false);
        polarity_fp8_field(absorbed_zero, true);
    } else if (metadata_bits > 0) {
        compact_scale_field(column_scale, metadata_bits);
        compact_scale_field(absorbed_scale, metadata_bits);
        compact_scale_field(absorbed_zero, metadata_bits);
    } else {
        for (float& value : column_scale) value = half_round(value);
        for (float& value : absorbed_scale) value = half_round(value);
        for (float& value : absorbed_zero) value = half_round(value);
    }
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < columns; column++) {
            double value = matrix[static_cast<size_t>(row) * columns + column];
            int code = static_cast<int>(std::nearbyint(
                (value - lows[row]) / steps[row]));
            code = std::clamp(code, 0, maxima[row]);
            matrix[static_cast<size_t>(row) * columns + column] =
                (code * absorbed_scale[row] + absorbed_zero[row])
                * column_scale[column];
        }
    }
}

void store_hadamard(float* values, int dimension) {
    for (int dim = 0; dim < dimension; dim++) {
        values[dim] = half_round(values[dim]);
    }
    for (int width = 1; width < dimension; width *= 2) {
        for (int start = 0; start < dimension; start += 2 * width) {
            for (int dim = start; dim < start + width; dim++) {
                float left = values[dim];
                float right = values[dim + width];
                values[dim] = left + right;
                values[dim + width] = left - right;
            }
        }
    }
    float scale = half_round(1.0f / std::sqrt(static_cast<float>(dimension)));
    for (int dim = 0; dim < dimension; dim++) {
        values[dim] = half_round(values[dim] * scale);
    }
}

void store_even_transform(float* values, int dimension) {
    int power = dimension & -dimension;
    if (power == dimension) {
        store_hadamard(values, dimension);
        return;
    }
    int blocks = dimension / power;
    for (int block = 0; block < blocks; block++) {
        store_hadamard(values + block * power, power);
    }
    float factor = 2.0f / blocks;
    for (int dim = 0; dim < power; dim++) {
        float sum = 0.0f;
        for (int block = 0; block < blocks; block++) {
            sum += values[block * power + dim];
        }
        for (int block = 0; block < blocks; block++) {
            int index = block * power + dim;
            values[index] = half_round(values[index] - factor * sum);
        }
    }
}

void inverse_even_transform(float* values, int dimension) {
    int power = dimension & -dimension;
    for (int block = 0; block < dimension / power; block++) {
        inverse_walsh_hadamard(values + block * power, power);
    }
    if (power == dimension) return;
    int blocks = dimension / power;
    float factor = 2.0f / blocks;
    for (int dim = 0; dim < power; dim++) {
        float sum = 0.0f;
        for (int block = 0; block < blocks; block++) {
            sum += values[block * power + dim];
        }
        for (int block = 0; block < blocks; block++) {
            int index = block * power + dim;
            values[index] -= factor * sum;
        }
    }
}

void simulate_tail_vector(float* source, int dimension, int bits) {
    std::vector<float> transformed(source, source + dimension);
    store_even_transform(transformed.data(), dimension);
    float maximum = 0.0f;
    for (float value : transformed) {
        maximum = std::max(maximum, std::abs(value));
    }
    int levels = 1 << bits;
    float peak = levels / 2.0f - 0.5f;
    float scale = half_round(maximum > 0.0f ? maximum / peak : 1.0f);
    for (float& value : transformed) {
        int code = static_cast<int>(std::nearbyint(
            value / scale + levels / 2.0f - 0.5f));
        code = std::clamp(code, 0, levels - 1);
        value = (code - levels / 2.0f + 0.5f) * scale;
    }
    inverse_even_transform(transformed.data(), dimension);
    for (int dim = 0; dim < dimension; dim++) {
        source[dim] = half_round(transformed[dim]);
    }
}

// MLX 0.31.2 Metal affine Q2/G64. The signed scale and endpoint snap match
// affine_quantize in mlx/backend/metal/kernels/quantized.h.
void simulate_mlx_q2(float* source, int dimension) {
    constexpr int group = 64;
    constexpr float bins = 3.0f;
    for (int start = 0; start < dimension; start += group) {
        float low = std::numeric_limits<float>::max();
        float high = 0.0f;
        for (int index = 0; index < group; index++) {
            low = std::min(low, source[start + index]);
            high = std::max(high, source[start + index]);
        }
        bool low_side = std::abs(low) > std::abs(high);
        float scale = std::max((high - low) / bins, 1e-7f);
        scale = low_side ? scale : -scale;
        float edge = low_side ? low : high;
        float edge_code = std::round(edge / scale);
        float bias = 0.0f;
        if (edge_code != 0.0f) {
            scale = edge / edge_code;
            bias = edge;
        }
        float stored_scale = half_round(scale);
        float stored_bias = half_round(bias);
        for (int index = 0; index < group; index++) {
            float code = std::round(
                (source[start + index] - bias) / scale);
            code = std::clamp(code, 0.0f, bins);
            source[start + index] = code * stored_scale + stored_bias;
        }
    }
}

// MLX-VLM TurboQuant 2.5-bit mode at D64: K2/V3, per-vector FP16 norm,
// seeded randomized Hadamard transform, and the fixed spherical codebooks.
void simulate_turboquant_d64(float* source, int bits, uint64_t sign_mask) {
    static constexpr float kCodebook2[] = {
        -0.187454432f, -0.0564936958f, 0.0564936921f, 0.187454432f
    };
    static constexpr float kCodebook3[] = {
        -0.263751388f, -0.16599451f, -0.0936825648f, -0.0304045826f,
         0.030404577f, 0.0936825722f, 0.16599454f, 0.263751417f
    };
    const float* codebook = bits == 2 ? kCodebook2 : kCodebook3;
    int levels = 1 << bits;
    float norm_squared = 0.0f;
    for (int index = 0; index < 64; index++) {
        norm_squared += source[index] * source[index];
    }
    float norm = std::sqrt(norm_squared);
    float divisor = std::max(norm, 1e-6f);
    float rotated[64];
    for (int index = 0; index < 64; index++) {
        float sign = sign_mask & (uint64_t{1} << index) ? 1.0f : -1.0f;
        rotated[index] = source[index] * sign / divisor;
    }
    walsh_hadamard(rotated, 64);
    for (float& value : rotated) {
        int code = 0;
        while (code + 1 < levels
               && value > 0.5f * (codebook[code] + codebook[code + 1])) {
            code++;
        }
        value = codebook[code];
    }
    walsh_hadamard(rotated, 64);
    float stored_norm = half_round(norm);
    for (int index = 0; index < 64; index++) {
        float sign = sign_mask & (uint64_t{1} << index) ? 1.0f : -1.0f;
        source[index] = stored_norm * sign * rotated[index];
    }
}

void simulate_bfp3_vector(float* source, int dimension, int position) {
    std::vector<float> transformed(source, source + dimension);
    store_even_transform(transformed.data(), dimension);
    for (int start = 0; start < dimension; start += 32) {
        int count = std::min(32, dimension - start);
        int stride = count / 4;
        int rotation = position % stride;
        auto low_precision = [&](int index) {
            return (index - rotation + count) % stride == 0;
        };
        float required_scale = 0.0f;
        for (int index = 0; index < count; index++) {
            float peak = low_precision(index) ? 1.5f : 3.5f;
            required_scale = std::max(
                required_scale, std::abs(transformed[start + index]) / peak);
        }
        int exponent = required_scale > 0.0f
                     ? static_cast<int>(std::ceil(std::log2(required_scale)))
                     : -8;
        exponent = std::clamp(exponent, -8, 7);
        float scale = std::ldexp(1.0f, exponent);
        for (int index = 0; index < count; index++) {
            int bits = low_precision(index) ? 2 : 3;
            int levels = 1 << bits;
            int code = static_cast<int>(std::nearbyint(
                transformed[start + index] / scale
                + levels / 2.0f - 0.5f));
            code = std::clamp(code, 0, levels - 1);
            transformed[start + index] =
                (code - levels / 2.0f + 0.5f) * scale;
        }
    }
    inverse_even_transform(transformed.data(), dimension);
    for (int dim = 0; dim < dimension; dim++) {
        source[dim] = half_round(transformed[dim]);
    }
}

void simulate_kivi_group(float* source, int count, int stride) {
    float low = source[0];
    float high = source[0];
    for (int index = 1; index < count; index++) {
        low = std::min(low, source[index * stride]);
        high = std::max(high, source[index * stride]);
    }
    float scale = half_round(half_round(high - low) / 3.0f);
    if (scale == 0.0f) return;
    for (int index = 0; index < count; index++) {
        float normalized = half_round(
            half_round(source[index * stride] - low) / scale);
        int code = std::clamp(
            static_cast<int>(std::nearbyint(normalized)), 0, 3);
        source[index * stride] = code * scale + low;
    }
}

void simulate_kivi_keys(float* source, int tokens, int dimension) {
    constexpr int group = 32;
    for (int start = 0; start < tokens; start += group) {
        for (int dim = 0; dim < dimension; dim++) {
            simulate_kivi_group(source + start * dimension + dim,
                                group, dimension);
        }
    }
}

void simulate_kivi_value(float* source, int dimension) {
    constexpr int group = 32;
    for (int start = 0; start < dimension; start += group) {
        simulate_kivi_group(source + start, group, 1);
    }
}

void simulate_kvarn_tile(float* source, int tokens, int dimension, int bits,
                         bool key, int metadata_bits) {
    std::vector<double> rotated(static_cast<size_t>(tokens) * dimension);
    std::vector<float> vector(dimension);
    for (int token = 0; token < tokens; token++) {
        std::copy_n(source + static_cast<size_t>(token) * dimension,
                    dimension, vector.data());
        store_even_transform(vector.data(), dimension);
        for (int dim = 0; dim < dimension; dim++) {
            size_t index = key
                         ? static_cast<size_t>(dim) * tokens + token
                         : static_cast<size_t>(token) * dimension + dim;
            rotated[index] = vector[dim];
        }
    }
    variance_normalized_rtn(
        rotated, key ? dimension : tokens, key ? tokens : dimension,
        bits, metadata_bits);
    for (int token = 0; token < tokens; token++) {
        for (int dim = 0; dim < dimension; dim++) {
            size_t index = key
                         ? static_cast<size_t>(dim) * tokens + token
                         : static_cast<size_t>(token) * dimension + dim;
            vector[dim] = static_cast<float>(rotated[index]);
        }
        inverse_even_transform(vector.data(), dimension);
        for (int dim = 0; dim < dimension; dim++) {
            vector[dim] = half_round(vector[dim]);
        }
        std::copy_n(vector.data(), dimension,
                    source + static_cast<size_t>(token) * dimension);
    }
}
#endif

} // namespace

bool KVCache::init(int n_layers, int n_kv_heads, int head_dim, int capacity,
                   KVCacheMode mode) {
    free();
    if (n_layers <= 0 || n_kv_heads <= 0 || head_dim <= 0 || capacity <= 0) {
        return false;
    }
    n_layers_ = n_layers;
    n_kv_heads_ = n_kv_heads;
    head_dim_ = head_dim;
    capacity_ = capacity;
    mode_ = mode;
    size_t elements = static_cast<size_t>(n_layers) * n_kv_heads
                    * capacity * head_dim;
    if (mode == KVCacheMode::FP32) {
        k32_.assign(elements, 0.0f);
        v32_.assign(elements, 0.0f);
        return true;
    }
    if (mode == KVCacheMode::FP16) {
        k16_.assign(elements, 0);
        v16_.assign(elements, 0);
        return true;
    }
    laplace_ = std::make_unique<LaplaceKV>();
    return laplace_->init(
        n_layers, n_kv_heads, head_dim, capacity, streaming_);
}

void KVCache::free() {
    std::vector<float>().swap(k32_);
    std::vector<float>().swap(v32_);
    std::vector<uint16_t>().swap(k16_);
    std::vector<uint16_t>().swap(v16_);
    laplace_.reset();
    n_layers_ = 0;
    n_kv_heads_ = 0;
    head_dim_ = 0;
    capacity_ = 0;
#if defined(LAPLACE_KV_CAPTURE)
    research_key_bits_ = 0;
    research_value_bits_ = 0;
    research_group_ = 0;
    research_sink_tokens_ = 0;
    research_metadata_bits_ = 0;
    research_tail_key_bits_ = 0;
    research_tail_value_bits_ = 0;
    research_bfp3_ = false;
    research_kivi_2_ = false;
    research_mlx_q2_ = false;
    research_turboquant_2_5_ = false;
#endif
}

#if defined(LAPLACE_KV_CAPTURE)
bool KVCache::set_research_bfp3() {
    if (mode_ != KVCacheMode::FP32 || laplace_
        || head_dim_ < 16 || head_dim_ % 16 != 0) {
        return false;
    }
    research_bfp3_ = true;
    return true;
}

bool KVCache::set_research_kivi_2() {
    if (mode_ != KVCacheMode::FP32 || laplace_
        || head_dim_ < 32 || head_dim_ % 32 != 0) {
        return false;
    }
    research_kivi_2_ = true;
    return true;
}

bool KVCache::set_research_mlx_q2() {
    if (mode_ != KVCacheMode::FP32 || head_dim_ % 64 != 0) return false;
    research_mlx_q2_ = true;
    return true;
}

bool KVCache::set_research_turboquant_2_5() {
    if (mode_ != KVCacheMode::FP32 || head_dim_ != 64) return false;
    research_turboquant_2_5_ = true;
    return true;
}

bool KVCache::set_research_baseline(int key_bits, int value_bits,
                                    int group, int sink_tokens,
                                    int metadata_bits, int tail_key_bits,
                                    int tail_value_bits) {
    if (mode_ != KVCacheMode::FP32 || laplace_
        || (key_bits != 2 && key_bits != 3
            && key_bits != 4 && key_bits != 5)
        || (value_bits != 2 && value_bits != 4)
        || group <= 1 || sink_tokens < 0
        || (metadata_bits != 0
            && (metadata_bits < 2 || metadata_bits > 9))
        || ((tail_key_bits != 0 || tail_value_bits != 0)
            && ((tail_key_bits != 3 && tail_key_bits != 4)
                || (tail_value_bits != 2 && tail_value_bits != 3)))
        || head_dim_ <= 1 || (head_dim_ & 1) != 0) {
        return false;
    }
    research_key_bits_ = key_bits;
    research_value_bits_ = value_bits;
    research_group_ = group;
    research_sink_tokens_ = sink_tokens;
    research_metadata_bits_ = metadata_bits;
    research_tail_key_bits_ = tail_key_bits;
    research_tail_value_bits_ = tail_value_bits;
    return true;
}
#endif

void KVCache::load_k(int layer, int head, int pos, float* output) const {
    if (laplace_) {
        laplace_->load_k_wh(layer, head, pos, output);
        if (laplace_->uses_rotation()) inverse_walsh_hadamard(output, head_dim_);
    } else if (mode_ == KVCacheMode::FP16) {
        from_f16(k16_.data() + slot_index(layer, head, pos), output, head_dim_);
    } else {
        std::memcpy(output, slot_k(layer, head, pos), sizeof(float) * head_dim_);
    }
}

void KVCache::load_v(int layer, int head, int pos, float* output) const {
    if (laplace_) {
        laplace_->load_v_wh(layer, head, pos, output);
        if (laplace_->uses_rotation()) inverse_walsh_hadamard(output, head_dim_);
    } else if (mode_ == KVCacheMode::FP16) {
        from_f16(v16_.data() + slot_index(layer, head, pos), output, head_dim_);
    } else {
        std::memcpy(output, slot_v(layer, head, pos), sizeof(float) * head_dim_);
    }
}

void KVCache::store_k(int layer, int head, int pos, const float* input) {
    if (laplace_) {
        float transformed[512];
        std::memcpy(transformed, input, sizeof(float) * head_dim_);
        if (laplace_->uses_rotation()) walsh_hadamard(transformed, head_dim_);
        laplace_->store_k_wh(layer, head, pos, transformed);
    } else if (mode_ == KVCacheMode::FP16) {
        to_f16(input, k16_.data() + slot_index(layer, head, pos), head_dim_);
    } else {
        std::memcpy(slot_k(layer, head, pos), input, sizeof(float) * head_dim_);
    }
}

void KVCache::store_v(int layer, int head, int pos, const float* input) {
    if (laplace_) {
        float transformed[512];
        std::memcpy(transformed, input, sizeof(float) * head_dim_);
        if (laplace_->uses_rotation()) walsh_hadamard(transformed, head_dim_);
        laplace_->store_v_wh(layer, head, pos, transformed);
    } else if (mode_ == KVCacheMode::FP16) {
        to_f16(input, v16_.data() + slot_index(layer, head, pos), head_dim_);
    } else {
        std::memcpy(slot_v(layer, head, pos), input, sizeof(float) * head_dim_);
#if defined(LAPLACE_KV_CAPTURE)
        if (research_kivi_2_) {
            for (int dim = 0; dim < head_dim_; dim++) {
                slot_k(layer, head, pos)[dim] = half_round(
                    slot_k(layer, head, pos)[dim]);
                slot_v(layer, head, pos)[dim] = half_round(
                    slot_v(layer, head, pos)[dim]);
            }
            constexpr int residual = 128;
            if (pos >= residual && pos % residual == 0) {
                simulate_kivi_keys(
                    slot_k(layer, head, pos - residual),
                    residual, head_dim_);
            }
            if (pos > residual) {
                simulate_kivi_value(
                    slot_v(layer, head, pos - residual - 1), head_dim_);
            }
        } else if (research_mlx_q2_) {
            simulate_mlx_q2(slot_k(layer, head, pos), head_dim_);
            simulate_mlx_q2(slot_v(layer, head, pos), head_dim_);
        } else if (research_turboquant_2_5_) {
            simulate_turboquant_d64(slot_k(layer, head, pos), 2,
                                    0x601bf8ee3d23b3c0ULL);
            simulate_turboquant_d64(slot_v(layer, head, pos), 3,
                                    0x5d98147ac5be444eULL);
        } else if (research_bfp3_) {
            simulate_bfp3_vector(slot_k(layer, head, pos), head_dim_, pos);
            simulate_bfp3_vector(slot_v(layer, head, pos), head_dim_, pos);
        } else if (research_tail_key_bits_ != 0) {
            simulate_tail_vector(slot_k(layer, head, pos), head_dim_,
                                 research_tail_key_bits_);
            simulate_tail_vector(slot_v(layer, head, pos), head_dim_,
                                 research_tail_value_bits_);
        } else if (research_group_ > 0) {
            for (int dim = 0; dim < head_dim_; dim++) {
                slot_k(layer, head, pos)[dim] = half_round(
                    slot_k(layer, head, pos)[dim]);
                slot_v(layer, head, pos)[dim] = half_round(
                    slot_v(layer, head, pos)[dim]);
            }
        }
        int completed = pos + 1 - research_sink_tokens_;
        if (research_group_ > 0 && completed > 0
            && completed % research_group_ == 0) {
            int start = pos + 1 - research_group_;
            simulate_kvarn_tile(slot_k(layer, head, start), research_group_,
                                head_dim_, research_key_bits_, true,
                                research_metadata_bits_);
            simulate_kvarn_tile(slot_v(layer, head, start), research_group_,
                                head_dim_, research_value_bits_, false,
                                research_metadata_bits_);
        }
#endif
    }
}

void KVCache::load_k_wh(int layer, int head, int pos, float* output) const {
    if (laplace_) {
        laplace_->load_k_wh(layer, head, pos, output);
        return;
    }
    load_k(layer, head, pos, output);
    walsh_hadamard(output, head_dim_);
}

void KVCache::load_v_wh(int layer, int head, int pos, float* output) const {
    if (laplace_) {
        laplace_->load_v_wh(layer, head, pos, output);
        return;
    }
    load_v(layer, head, pos, output);
    walsh_hadamard(output, head_dim_);
}

void KVCache::store_k_wh(int layer, int head, int pos, const float* input) {
    if (laplace_) {
        laplace_->store_k_wh(layer, head, pos, input);
        return;
    }
    float value[512];
    std::memcpy(value, input, sizeof(float) * head_dim_);
    inverse_walsh_hadamard(value, head_dim_);
    store_k(layer, head, pos, value);
}

void KVCache::store_v_wh(int layer, int head, int pos, const float* input) {
    if (laplace_) {
        laplace_->store_v_wh(layer, head, pos, input);
        return;
    }
    float value[512];
    std::memcpy(value, input, sizeof(float) * head_dim_);
    inverse_walsh_hadamard(value, head_dim_);
    store_v(layer, head, pos, value);
}

float KVCache::dot_k_wh(int layer, int head, int pos,
                        const float* query_wh) const {
    float key[512];
    load_k_wh(layer, head, pos, key);
    return ops::dot(query_wh, key, head_dim_);
}

void KVCache::weighted_add_v_wh(int layer, int head, int pos, float weight,
                                float* output_wh) const {
    float value[512];
    load_v_wh(layer, head, pos, value);
    ops::axpy(output_wh, weight, value, head_dim_);
}

void KVCache::dot_k_all_wh(int layer, int head, int n_tokens,
                           const float* query_wh, float logit_scale,
                           float* scores, int first_token) const {
    if (!laplace_) return;
    laplace_->dot_keys_wh(
        layer, head, n_tokens, query_wh, scores, first_token);
    for (int token = first_token; token < n_tokens; token++) {
        scores[token] *= logit_scale;
    }
}

void KVCache::weighted_add_v_all_wh(int layer, int head, int n_tokens,
                                    const float* weights, float* output_wh,
                                    int first_token) const {
    if (laplace_) {
        laplace_->add_values_wh(
            layer, head, n_tokens, weights, output_wh, first_token);
    }
}

void KVCache::attention_all_wh(int layer, int head, int n_tokens,
                               const float* query_wh, float logit_scale,
                               float* output_wh, int first_token) const {
    if (laplace_) {
        laplace_->attention_wh(
            layer, head, n_tokens, query_wh, logit_scale,
            output_wh, first_token);
    }
}

void KVCache::attention_batch_all_wh(
        int layer, int head, int count, const int* n_tokens,
        const float* const* queries_wh, float logit_scale,
        float* const* outputs_wh, int first_token) const {
    if (laplace_) {
        laplace_->attention_batch_wh(
            layer, head, count, n_tokens, queries_wh,
            logit_scale, outputs_wh, first_token);
    }
}

float* KVCache::slot_k(int layer, int head, int pos) {
    return k32_.data() + slot_index(layer, head, pos);
}
float* KVCache::slot_v(int layer, int head, int pos) {
    return v32_.data() + slot_index(layer, head, pos);
}
const float* KVCache::slot_k(int layer, int head, int pos) const {
    return k32_.data() + slot_index(layer, head, pos);
}
const float* KVCache::slot_v(int layer, int head, int pos) const {
    return v32_.data() + slot_index(layer, head, pos);
}
float* KVCache::head_k(int layer, int head) { return slot_k(layer, head, 0); }
float* KVCache::head_v(int layer, int head) { return slot_v(layer, head, 0); }
const float* KVCache::head_k(int layer, int head) const {
    return slot_k(layer, head, 0);
}
const float* KVCache::head_v(int layer, int head) const {
    return slot_v(layer, head, 0);
}
const uint16_t* KVCache::head_k16(int layer, int head) const {
    return k16_.data() + slot_index(layer, head, 0);
}
const uint16_t* KVCache::head_v16(int layer, int head) const {
    return v16_.data() + slot_index(layer, head, 0);
}

} // namespace Laplace
