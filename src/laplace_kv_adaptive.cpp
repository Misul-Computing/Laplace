#include "laplace_kv_adaptive.h"

#include "fp16.h"
#include "ops.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>

namespace Laplace {

namespace {

bool finite_error(const LaplaceKVTileError& error) {
    return std::isfinite(error.key_rms) &&
           std::isfinite(error.key_max) &&
           std::isfinite(error.value_rms) &&
           std::isfinite(error.value_max);
}

struct ErrorAccumulator {
    double squared_error = 0.0;
    double squared_source = 0.0;
    double maximum = 0.0;
};

void add_vector_error(const float* source, const float* decoded, int size,
                      ErrorAccumulator& accumulator) {
    double vector_error = 0.0;
    double vector_source = 0.0;
    for (int index = 0; index < size; index++) {
        double delta = static_cast<double>(decoded[index]) - source[index];
        vector_error += delta * delta;
        vector_source += static_cast<double>(source[index]) * source[index];
    }
    accumulator.squared_error += vector_error;
    accumulator.squared_source += vector_source;
    double relative = vector_source == 0.0
                    ? (vector_error == 0.0 ? 0.0 : INFINITY)
                    : std::sqrt(vector_error / vector_source);
    accumulator.maximum = std::max(accumulator.maximum, relative);
}

double rms_error(const ErrorAccumulator& accumulator) {
    if (accumulator.squared_source == 0.0)
        return accumulator.squared_error == 0.0 ? 0.0 : INFINITY;
    return std::sqrt(
        accumulator.squared_error / accumulator.squared_source);
}

LaplaceKVTileError q4_error(
        const LaplaceKVQ4Tile& tile, const float* keys,
        const float* values, int dimension) {
    ErrorAccumulator key_error;
    ErrorAccumulator value_error;
    std::vector<float> decoded(dimension);
    for (int token = 0; token < LaplaceKVAdaptive::kTokens; token++) {
        tile.load_key_wh(token, decoded.data());
        add_vector_error(
            keys + static_cast<size_t>(token) * dimension,
            decoded.data(), dimension, key_error);
        tile.load_value_wh(token, decoded.data());
        add_vector_error(
            values + static_cast<size_t>(token) * dimension,
            decoded.data(), dimension, value_error);
    }
    return {
        rms_error(key_error), key_error.maximum,
        rms_error(value_error), value_error.maximum,
    };
}

LaplaceKVTileError k8_error(
        const LaplaceKVTile& first, const LaplaceKVTile& second,
        const float* keys, const float* values, int dimension) {
    ErrorAccumulator key_error;
    ErrorAccumulator value_error;
    std::vector<float> decoded(dimension);
    for (int token = 0; token < LaplaceKVAdaptive::kTokens; token++) {
        const LaplaceKVTile& tile =
            token < LaplaceKVTile::kTokens ? first : second;
        int offset = token % LaplaceKVTile::kTokens;
        tile.load_key_wh(offset, decoded.data());
        add_vector_error(
            keys + static_cast<size_t>(token) * dimension,
            decoded.data(), dimension, key_error);
        tile.load_value_wh(offset, decoded.data());
        add_vector_error(
            values + static_cast<size_t>(token) * dimension,
            decoded.data(), dimension, value_error);
    }
    return {
        rms_error(key_error), key_error.maximum,
        rms_error(value_error), value_error.maximum,
    };
}

} // namespace

bool laplace_kv_prefer_q4(const LaplaceKVTileError& q4,
                          const LaplaceKVTileError& k8) {
    return finite_error(q4) && finite_error(k8) &&
           q4.key_rms <= k8.key_rms &&
           q4.key_max <= k8.key_max &&
           q4.value_rms <= k8.value_rms &&
           q4.value_max <= k8.value_max;
}

bool laplace_kv_accept_k8(const LaplaceKVTileError& error) {
    return finite_error(error) &&
           error.key_rms <= 0.01 &&
           error.key_max <= 0.02 &&
           error.value_rms <= 0.025 &&
           error.value_max <= 0.05;
}

bool codec_accepts_tile(LaplaceKVAdaptiveFormat codec, int tile_tokens) {
    switch (codec) {
    case LaplaceKVAdaptiveFormat::K4_V2:
        return tile_tokens == LaplaceKVQ4Tile::kTokens;
    case LaplaceKVAdaptiveFormat::K8_V6:
        return tile_tokens == 2 * LaplaceKVTile::kTokens;
    case LaplaceKVAdaptiveFormat::FP16:
        return tile_tokens == LaplaceKVAdaptive::kTokens;
    case LaplaceKVAdaptiveFormat::MUTABLE:
        return false;
    }
    return false;
}

bool LaplaceKVAdaptive::init(
        int n_layers, int n_kv_heads, int head_dim, int capacity,
        bool streaming, CompressionEligibility eligibility) {
    clear();
    if (n_layers <= 0 || n_kv_heads <= 0 || head_dim < 32 ||
        head_dim > 512 || head_dim % 16 != 0 || capacity <= 0) {
        return false;
    }
    n_layers_ = n_layers;
    n_kv_heads_ = n_kv_heads;
    head_dim_ = head_dim;
    capacity_ = capacity;
    streaming_ = streaming;
    eligibility_ = eligibility;
    tiles_per_head_ =
        (capacity + kTokens - 1) / kTokens;
    q4_words_ = LaplaceKVQ4Tile::storage_words(head_dim);
    k8_words_ = LaplaceKVTile::storage_words(head_dim);
    fp16_words_ = static_cast<size_t>(kTokens) * head_dim;
    size_t heads = static_cast<size_t>(n_layers) * n_kv_heads;
    descriptors_.resize(heads * tiles_per_head_);
    if (streaming_) {
        const char* temp = std::getenv("TMPDIR");
        std::string pattern = temp && *temp ? temp : "/tmp";
        if (pattern.back() != '/') pattern.push_back('/');
        pattern += "laplace-kv-adaptive-XXXXXX";
        std::vector<char> path(pattern.begin(), pattern.end());
        path.push_back('\0');
        archive_fd_ = mkstemp(path.data());
        if (archive_fd_ < 0) {
            clear();
            return false;
        }
        unlink(path.data());
#if defined(__APPLE__) && defined(F_NOCACHE)
        int no_cache = 1;
        if (fcntl(archive_fd_, F_NOCACHE, no_cache) != 0) {
            clear();
            return false;
        }
#endif
    } else {
        resident_arenas_.resize(heads);
    }
    size_t tail_values = heads * kTokens * head_dim;
    k_tail_.assign(tail_values, 0.0f);
    v_tail_.assign(tail_values, 0.0f);
    return true;
}

void LaplaceKVAdaptive::clear() {
    if (archive_fd_ >= 0) close(archive_fd_);
    archive_fd_ = -1;
    archive_bytes_ = 0;
    resident_arenas_.clear();
    descriptors_.clear();
    k_tail_.clear();
    v_tail_.clear();
    q4_tiles_.store(0);
    k8_tiles_.store(0);
    fp16_tiles_.store(0);
    stream_calls_.store(0);
    archive_read_bytes_.store(0);
    archive_write_bytes_.store(0);
    n_layers_ = 0;
    n_kv_heads_ = 0;
    head_dim_ = 0;
    capacity_ = 0;
    tiles_per_head_ = 0;
    q4_words_ = 0;
    k8_words_ = 0;
    fp16_words_ = 0;
    streaming_ = false;
    eligibility_ = {};
}

size_t LaplaceKVAdaptive::head_index(int layer, int head) const {
    return static_cast<size_t>(layer) * n_kv_heads_ + head;
}

size_t LaplaceKVAdaptive::tile_index(
        int layer, int head, int tile) const {
    return head_index(layer, head) * tiles_per_head_ + tile;
}

float* LaplaceKVAdaptive::tail_k(
        int layer, int head, int offset) {
    size_t index =
        (head_index(layer, head) * kTokens + offset) * head_dim_;
    return k_tail_.data() + index;
}

float* LaplaceKVAdaptive::tail_v(
        int layer, int head, int offset) {
    size_t index =
        (head_index(layer, head) * kTokens + offset) * head_dim_;
    return v_tail_.data() + index;
}

const float* LaplaceKVAdaptive::tail_k(
        int layer, int head, int offset) const {
    size_t index =
        (head_index(layer, head) * kTokens + offset) * head_dim_;
    return k_tail_.data() + index;
}

const float* LaplaceKVAdaptive::tail_v(
        int layer, int head, int offset) const {
    size_t index =
        (head_index(layer, head) * kTokens + offset) * head_dim_;
    return v_tail_.data() + index;
}

bool LaplaceKVAdaptive::write_archive(
        const uint32_t* input, size_t words, uint64_t& offset) {
    std::lock_guard<std::mutex> lock(archive_mutex_);
    offset = archive_bytes_;
    size_t bytes = words * sizeof(uint32_t);
    size_t done = 0;
    while (done < bytes) {
        ssize_t result = pwrite(
            archive_fd_, reinterpret_cast<const uint8_t*>(input) + done,
            bytes - done, static_cast<off_t>(offset + done));
        if (result > 0) {
            done += static_cast<size_t>(result);
            archive_write_bytes_.fetch_add(
                static_cast<uint64_t>(result), std::memory_order_relaxed);
        } else if (result < 0 && errno == EINTR) {
            continue;
        } else {
            return false;
        }
    }
    archive_bytes_ += bytes;
    return true;
}

bool LaplaceKVAdaptive::read_archive(
        const TileDescriptor& descriptor, uint32_t* output) const {
    size_t bytes = static_cast<size_t>(descriptor.words) * sizeof(uint32_t);
    size_t done = 0;
    while (done < bytes) {
        ssize_t result = pread(
            archive_fd_, reinterpret_cast<uint8_t*>(output) + done,
            bytes - done,
            static_cast<off_t>(descriptor.offset + done));
        if (result > 0) {
            done += static_cast<size_t>(result);
            archive_read_bytes_.fetch_add(
                static_cast<uint64_t>(result), std::memory_order_relaxed);
        } else if (result < 0 && errno == EINTR) {
            continue;
        } else {
            return false;
        }
    }
    return true;
}

const uint32_t* LaplaceKVAdaptive::resident_payload(
        size_t head, const TileDescriptor& descriptor) const {
    return resident_arenas_[head].data() +
           descriptor.offset / sizeof(uint32_t);
}

bool LaplaceKVAdaptive::seal_tile(
        int layer, int head, int tile) {
    const float* keys = tail_k(layer, head, 0);
    const float* values = tail_v(layer, head, 0);
    const bool q4_eligible = eligibility_.q4 &&
        codec_accepts_tile(LaplaceKVAdaptiveFormat::K4_V2, kTokens);
    const bool k8_eligible = eligibility_.k8 &&
        codec_accepts_tile(LaplaceKVAdaptiveFormat::K8_V6, kTokens);

    std::vector<uint32_t> q4_storage;
    bool q4_ready = false;
    LaplaceKVQ4Tile q4;
    if (q4_eligible) {
        q4_storage.resize(q4_words_);
        q4_ready =
            q4.init(head_dim_, q4_storage.data(), false) &&
            q4.seal(keys, values);
    }

    std::vector<uint32_t> k8_storage;
    bool k8_ready = false;
    LaplaceKVTile k8_first;
    LaplaceKVTile k8_second;
    if (k8_eligible) {
        k8_storage.resize(2 * k8_words_);
        k8_ready =
            k8_first.init(head_dim_, k8_storage.data(), false) &&
            k8_second.init(
                head_dim_, k8_storage.data() + k8_words_, false) &&
            k8_first.seal(keys, values) &&
            k8_second.seal(
                keys + static_cast<size_t>(LaplaceKVTile::kTokens) * head_dim_,
                values + static_cast<size_t>(LaplaceKVTile::kTokens) * head_dim_);
        if (!k8_ready) return false;
    }

    const LaplaceKVTileError q4_measure =
        q4_ready ? q4_error(q4, keys, values, head_dim_)
                 : LaplaceKVTileError{INFINITY, INFINITY, INFINITY, INFINITY};
    const LaplaceKVTileError k8_measure =
        k8_ready ? k8_error(k8_first, k8_second, keys, values, head_dim_)
                 : LaplaceKVTileError{INFINITY, INFINITY, INFINITY, INFINITY};
    const bool use_q4 =
        q4_ready && (!k8_ready ||
                     laplace_kv_prefer_q4(q4_measure, k8_measure));
    const bool use_k8 =
        k8_ready && !use_q4 && laplace_kv_accept_k8(k8_measure);
    std::vector<uint32_t> fp16_storage;
    if (!use_q4 && !use_k8) {
        fp16_storage.resize(fp16_words_);
        auto* packed = reinterpret_cast<uint16_t*>(fp16_storage.data());
        const size_t values_offset =
            static_cast<size_t>(kTokens) * head_dim_;
        for (size_t index = 0; index < values_offset; index++) {
            packed[index] = fp32_to_fp16(keys[index]);
            packed[values_offset + index] = fp32_to_fp16(values[index]);
        }
    }
    const std::vector<uint32_t>& selected =
        use_q4 ? q4_storage : use_k8 ? k8_storage : fp16_storage;
    TileDescriptor descriptor;
    descriptor.words = static_cast<uint32_t>(selected.size());
    descriptor.format = use_q4 ? LaplaceKVAdaptiveFormat::K4_V2
                      : use_k8 ? LaplaceKVAdaptiveFormat::K8_V6
                               : LaplaceKVAdaptiveFormat::FP16;
    size_t head_id = head_index(layer, head);
    if (streaming_) {
        if (!write_archive(
                selected.data(), selected.size(), descriptor.offset)) {
            return false;
        }
    } else {
        auto& arena = resident_arenas_[head_id];
        descriptor.offset = arena.size() * sizeof(uint32_t);
        arena.insert(arena.end(), selected.begin(), selected.end());
    }
    descriptors_[tile_index(layer, head, tile)] = descriptor;
    if (use_q4)
        q4_tiles_.fetch_add(1, std::memory_order_relaxed);
    else if (use_k8)
        k8_tiles_.fetch_add(1, std::memory_order_relaxed);
    else
        fp16_tiles_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void LaplaceKVAdaptive::store_k_wh(
        int layer, int head, int pos, const float* key_wh) {
    if (pos < 0 || pos >= capacity_) return;
    int offset = pos % kTokens;
    std::memcpy(
        tail_k(layer, head, offset), key_wh, sizeof(float) * head_dim_);
}

void LaplaceKVAdaptive::store_v_wh(
        int layer, int head, int pos, const float* value_wh) {
    if (pos < 0 || pos >= capacity_) return;
    int offset = pos % kTokens;
    std::memcpy(
        tail_v(layer, head, offset), value_wh, sizeof(float) * head_dim_);
    if (offset == kTokens - 1 &&
        !seal_tile(layer, head, pos / kTokens)) {
        std::abort();
    }
}

LaplaceKVAdaptiveFormat LaplaceKVAdaptive::tile_format(
        int layer, int head, int tile) const {
    if (layer < 0 || layer >= n_layers_ || head < 0 ||
        head >= n_kv_heads_ || tile < 0 || tile >= tiles_per_head_) {
        return LaplaceKVAdaptiveFormat::MUTABLE;
    }
    return descriptors_[tile_index(layer, head, tile)].format;
}

void LaplaceKVAdaptive::load_k_wh(
        int layer, int head, int pos, float* key_wh) const {
    if (pos < 0 || pos >= capacity_) return;
    int tile = pos / kTokens;
    int offset = pos % kTokens;
    const TileDescriptor& descriptor =
        descriptors_[tile_index(layer, head, tile)];
    if (descriptor.format == LaplaceKVAdaptiveFormat::MUTABLE) {
        std::memcpy(
            key_wh, tail_k(layer, head, offset), sizeof(float) * head_dim_);
        return;
    }
    static thread_local std::vector<uint32_t> archive;
    if (streaming_) archive.resize(descriptor.words);
    const uint32_t* data = streaming_
        ? (read_archive(descriptor, archive.data()) ? archive.data() : nullptr)
        : resident_payload(head_index(layer, head), descriptor);
    if (!data) std::abort();
    if (descriptor.format == LaplaceKVAdaptiveFormat::K4_V2) {
        LaplaceKVQ4Tile view;
        view.init(head_dim_, const_cast<uint32_t*>(data), true);
        view.load_key_wh(offset, key_wh);
    } else if (descriptor.format == LaplaceKVAdaptiveFormat::K8_V6) {
        LaplaceKVTile view;
        int half = offset / LaplaceKVTile::kTokens;
        view.init(
            head_dim_, const_cast<uint32_t*>(data) + half * k8_words_, true);
        view.load_key_wh(offset % LaplaceKVTile::kTokens, key_wh);
    } else {
        const auto* packed = reinterpret_cast<const uint16_t*>(data);
        const uint16_t* source =
            packed + static_cast<size_t>(offset) * head_dim_;
        for (int dim = 0; dim < head_dim_; dim++)
            key_wh[dim] = fp16_to_fp32(source[dim]);
    }
}

void LaplaceKVAdaptive::load_v_wh(
        int layer, int head, int pos, float* value_wh) const {
    if (pos < 0 || pos >= capacity_) return;
    int tile = pos / kTokens;
    int offset = pos % kTokens;
    const TileDescriptor& descriptor =
        descriptors_[tile_index(layer, head, tile)];
    if (descriptor.format == LaplaceKVAdaptiveFormat::MUTABLE) {
        std::memcpy(
            value_wh, tail_v(layer, head, offset), sizeof(float) * head_dim_);
        return;
    }
    static thread_local std::vector<uint32_t> archive;
    if (streaming_) archive.resize(descriptor.words);
    const uint32_t* data = streaming_
        ? (read_archive(descriptor, archive.data()) ? archive.data() : nullptr)
        : resident_payload(head_index(layer, head), descriptor);
    if (!data) std::abort();
    if (descriptor.format == LaplaceKVAdaptiveFormat::K4_V2) {
        LaplaceKVQ4Tile view;
        view.init(head_dim_, const_cast<uint32_t*>(data), true);
        view.load_value_wh(offset, value_wh);
    } else if (descriptor.format == LaplaceKVAdaptiveFormat::K8_V6) {
        LaplaceKVTile view;
        int half = offset / LaplaceKVTile::kTokens;
        view.init(
            head_dim_, const_cast<uint32_t*>(data) + half * k8_words_, true);
        view.load_value_wh(offset % LaplaceKVTile::kTokens, value_wh);
    } else {
        const auto* packed = reinterpret_cast<const uint16_t*>(data);
        const uint16_t* source =
            packed + static_cast<size_t>(kTokens + offset) * head_dim_;
        for (int dim = 0; dim < head_dim_; dim++)
            value_wh[dim] = fp16_to_fp32(source[dim]);
    }
}

void LaplaceKVAdaptive::dot_keys_wh(
        int layer, int head, int n_tokens, const float* query_wh,
        float* scores, int first_token) const {
    n_tokens = std::clamp(n_tokens, 0, capacity_);
    first_token = std::clamp(first_token, 0, n_tokens);
    float tile_scores[kTokens];
    int token = first_token;
    while (token < n_tokens) {
        int tile = token / kTokens;
        int offset = token % kTokens;
        int count = std::min(kTokens - offset, n_tokens - token);
        const TileDescriptor& descriptor =
            descriptors_[tile_index(layer, head, tile)];
        if (descriptor.format == LaplaceKVAdaptiveFormat::MUTABLE) {
            for (int index = 0; index < count; index++) {
                scores[token + index] = ops::dot(
                    query_wh, tail_k(layer, head, offset + index), head_dim_);
            }
            token += count;
            continue;
        }
        static thread_local std::vector<uint32_t> archive;
        if (streaming_) archive.resize(descriptor.words);
        const uint32_t* data = streaming_
            ? (read_archive(descriptor, archive.data())
                ? archive.data() : nullptr)
            : resident_payload(head_index(layer, head), descriptor);
        if (!data) std::abort();
        if (descriptor.format == LaplaceKVAdaptiveFormat::K4_V2) {
            LaplaceKVQ4Tile view;
            view.init(head_dim_, const_cast<uint32_t*>(data), true);
            view.dot_keys(query_wh, tile_scores);
        } else if (descriptor.format == LaplaceKVAdaptiveFormat::K8_V6) {
            std::vector<int8_t> query_q8(head_dim_);
            float query_scale = laplace_kv_quantize_q8(
                query_wh, head_dim_, query_q8.data());
            LaplaceKVTile first;
            LaplaceKVTile second;
            first.init(head_dim_, const_cast<uint32_t*>(data), true);
            second.init(
                head_dim_,
                const_cast<uint32_t*>(data) + k8_words_, true);
            first.dot_keys(
                query_q8.data(), query_scale, tile_scores);
            second.dot_keys(
                query_q8.data(), query_scale,
                tile_scores + LaplaceKVTile::kTokens);
        } else {
            const auto* packed = reinterpret_cast<const uint16_t*>(data);
            for (int index = 0; index < kTokens; index++) {
                tile_scores[index] = ops::dot_f16(
                    query_wh,
                    packed + static_cast<size_t>(index) * head_dim_,
                    head_dim_);
            }
        }
        std::copy_n(tile_scores + offset, count, scores + token);
        token += count;
    }
}

void LaplaceKVAdaptive::add_values_wh(
        int layer, int head, int n_tokens, const float* weights,
        float* output_wh, int first_token) const {
    n_tokens = std::clamp(n_tokens, 0, capacity_);
    first_token = std::clamp(first_token, 0, n_tokens);
    float tile_weights[kTokens];
    int token = first_token;
    while (token < n_tokens) {
        int tile = token / kTokens;
        int offset = token % kTokens;
        int count = std::min(kTokens - offset, n_tokens - token);
        const TileDescriptor& descriptor =
            descriptors_[tile_index(layer, head, tile)];
        if (descriptor.format == LaplaceKVAdaptiveFormat::MUTABLE) {
            for (int index = 0; index < count; index++) {
                ops::axpy(
                    output_wh, weights[token + index],
                    tail_v(layer, head, offset + index), head_dim_);
            }
            token += count;
            continue;
        }
        static thread_local std::vector<uint32_t> archive;
        if (streaming_) archive.resize(descriptor.words);
        const uint32_t* data = streaming_
            ? (read_archive(descriptor, archive.data())
                ? archive.data() : nullptr)
            : resident_payload(head_index(layer, head), descriptor);
        if (!data) std::abort();
        std::fill_n(tile_weights, kTokens, 0.0f);
        std::copy_n(weights + token, count, tile_weights + offset);
        if (descriptor.format == LaplaceKVAdaptiveFormat::K4_V2) {
            LaplaceKVQ4Tile view;
            view.init(head_dim_, const_cast<uint32_t*>(data), true);
            view.add_values(tile_weights, output_wh);
        } else if (descriptor.format == LaplaceKVAdaptiveFormat::K8_V6) {
            LaplaceKVTile first;
            LaplaceKVTile second;
            first.init(head_dim_, const_cast<uint32_t*>(data), true);
            second.init(
                head_dim_,
                const_cast<uint32_t*>(data) + k8_words_, true);
            first.add_values(tile_weights, output_wh);
            second.add_values(
                tile_weights + LaplaceKVTile::kTokens, output_wh);
        } else {
            const auto* packed = reinterpret_cast<const uint16_t*>(data);
            const uint16_t* values =
                packed + static_cast<size_t>(kTokens) * head_dim_;
            for (int index = 0; index < kTokens; index++) {
                ops::axpy_f16(
                    output_wh, tile_weights[index],
                    values + static_cast<size_t>(index) * head_dim_,
                    head_dim_);
            }
        }
        token += count;
    }
}

void LaplaceKVAdaptive::attention_wh(
        int layer, int head, int n_tokens, const float* query_wh,
        float logit_scale, float* output_wh, int first_token) const {
    n_tokens = std::clamp(n_tokens, 0, capacity_);
    first_token = std::clamp(first_token, 0, n_tokens);
    std::fill_n(output_wh, head_dim_, 0.0f);
    if (first_token == n_tokens) return;
    std::vector<float> scores(n_tokens, 0.0f);
    dot_keys_wh(
        layer, head, n_tokens, query_wh, scores.data(), first_token);
    float maximum = scores[first_token] * logit_scale;
    for (int token = first_token + 1; token < n_tokens; token++)
        maximum = std::max(maximum, scores[token] * logit_scale);
    float sum = 0.0f;
    for (int token = first_token; token < n_tokens; token++) {
        scores[token] = std::exp(
            scores[token] * logit_scale - maximum);
        sum += scores[token];
    }
    float inverse = 1.0f / sum;
    for (int token = first_token; token < n_tokens; token++)
        scores[token] *= inverse;
    add_values_wh(
        layer, head, n_tokens, scores.data(), output_wh, first_token);
    if (streaming_)
        stream_calls_.fetch_add(1, std::memory_order_relaxed);
}

void LaplaceKVAdaptive::attention_batch_wh(
        int layer, int head, int count, const int* n_tokens,
        const float* const* queries_wh, float logit_scale,
        float* const* outputs_wh, int first_token) const {
    for (int query = 0; query < count; query++) {
        attention_wh(
            layer, head, n_tokens[query], queries_wh[query],
            logit_scale, outputs_wh[query], first_token);
    }
}

size_t LaplaceKVAdaptive::encoded_bytes(int n_tokens) const {
    int tokens = std::clamp(n_tokens, 0, capacity_);
    int sealed_tiles = tokens / kTokens;
    int tail_tokens = tokens % kTokens;
    size_t bytes = 0;
    size_t heads = static_cast<size_t>(n_layers_) * n_kv_heads_;
    for (size_t head = 0; head < heads; head++) {
        for (int tile = 0; tile < sealed_tiles; tile++) {
            bytes += static_cast<size_t>(
                descriptors_[head * tiles_per_head_ + tile].words) *
                sizeof(uint32_t);
        }
        bytes += static_cast<size_t>(tail_tokens) * head_dim_ *
                 2 * sizeof(float);
        bytes += static_cast<size_t>(
            (tokens + kTokens - 1) / kTokens);
    }
    return bytes;
}

size_t LaplaceKVAdaptive::storage_bytes() const {
    size_t bytes =
        (k_tail_.capacity() + v_tail_.capacity()) * sizeof(float) +
        descriptors_.capacity() * sizeof(TileDescriptor) +
        resident_arenas_.capacity() * sizeof(resident_arenas_[0]);
    for (const auto& arena : resident_arenas_)
        bytes += arena.capacity() * sizeof(uint32_t);
    return bytes;
}

size_t LaplaceKVAdaptive::archive_read_buffer_bytes() const {
    return streaming_
         ? std::max({q4_words_, 2 * k8_words_, fp16_words_}) *
               sizeof(uint32_t)
         : 0;
}

} // namespace Laplace
