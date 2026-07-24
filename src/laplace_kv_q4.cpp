#include "laplace_kv_q4.h"

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
#include <vector>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace Laplace {

namespace {

void set_k4(uint8_t* codes, size_t index, uint8_t code) {
    uint8_t& byte = codes[index / 2];
    int shift = static_cast<int>(index & 1) * 4;
    byte = static_cast<uint8_t>((byte & ~(15 << shift)) | (code << shift));
}

uint8_t get_k4(const uint8_t* codes, size_t index) {
    return static_cast<uint8_t>((codes[index / 2] >> ((index & 1) * 4)) & 15);
}

void set_v2(uint8_t* codes, size_t index, uint8_t code) {
    uint8_t& byte = codes[index / 4];
    int shift = static_cast<int>(index & 3) * 2;
    byte = static_cast<uint8_t>((byte & ~(3 << shift)) | (code << shift));
}

uint8_t get_v2(const uint8_t* codes, size_t index) {
    return static_cast<uint8_t>((codes[index / 4] >> ((index & 3) * 2)) & 3);
}

double standard_deviation(const std::vector<double>& matrix, int rows,
                          int columns, bool column, int index) {
    int count = column ? rows : columns;
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
        double delta =
            matrix[static_cast<size_t>(row) * columns + col] - mean;
        variance += delta * delta;
    }
    return std::sqrt(variance / std::max(1, count - 1));
}

double imbalance(const std::vector<double>& matrix, int rows, int columns) {
    double row_min = INFINITY;
    double row_max = 0.0;
    double column_min = INFINITY;
    double column_max = 0.0;
    for (int row = 0; row < rows; row++) {
        double value =
            standard_deviation(matrix, rows, columns, false, row);
        row_min = std::min(row_min, value);
        row_max = std::max(row_max, value);
    }
    for (int column = 0; column < columns; column++) {
        double value =
            standard_deviation(matrix, rows, columns, true, column);
        column_min = std::min(column_min, value);
        column_max = std::max(column_max, value);
    }
    return row_max / std::max(row_min, 1e-8) +
           column_max / std::max(column_min, 1e-8);
}

template <class StoreCode>
void encode_affine(const std::vector<double>& original, int rows, int columns,
                   int maximum, uint16_t* row_scale, uint16_t* row_zero,
                   uint16_t* column_scale, StoreCode store_code) {
    std::vector<double> matrix(original.size());
    std::vector<double> log_rows(rows, 0.0);
    std::vector<double> log_columns(columns, 0.0);
    std::vector<double> best_rows = log_rows;
    std::vector<double> best_columns = log_columns;
    auto rebuild = [&] {
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
                matrix[static_cast<size_t>(row) * columns + column] =
                    original[static_cast<size_t>(row) * columns + column] /
                    std::exp(log_rows[row] + log_columns[column]);
    };

    rebuild();
    double best_imbalance = imbalance(matrix, rows, columns);
    for (int iteration = 0; iteration < 8; iteration++) {
        for (int column = 0; column < columns; column++) {
            double deviation = std::clamp(
                standard_deviation(matrix, rows, columns, true, column),
                1e-3, 1e3);
            log_columns[column] =
                std::clamp(log_columns[column] + std::log(deviation),
                           -0.3, 10.0);
        }
        rebuild();
        for (int row = 0; row < rows; row++) {
            double deviation = std::clamp(
                standard_deviation(matrix, rows, columns, false, row),
                1e-3, 1e3);
            log_rows[row] =
                std::clamp(log_rows[row] + std::log(deviation), -0.3, 10.0);
        }
        rebuild();
        double candidate = imbalance(matrix, rows, columns);
        if (candidate <= best_imbalance) {
            best_imbalance = candidate;
            best_rows = log_rows;
            best_columns = log_columns;
        }
    }

    log_rows = best_rows;
    log_columns = best_columns;
    rebuild();
    for (int column = 0; column < columns; column++)
        column_scale[column] =
            fp32_to_fp16(static_cast<float>(std::exp(log_columns[column])));
    for (int row = 0; row < rows; row++) {
        double low = INFINITY;
        double high = -INFINITY;
        for (int column = 0; column < columns; column++) {
            double value =
                matrix[static_cast<size_t>(row) * columns + column];
            low = std::min(low, value);
            high = std::max(high, value);
        }
        double step = std::max((high - low) / maximum, 1e-10);
        double scale = std::exp(log_rows[row]);
        row_scale[row] = fp32_to_fp16(static_cast<float>(scale * step));
        row_zero[row] = fp32_to_fp16(static_cast<float>(scale * low));
        for (int column = 0; column < columns; column++) {
            int code = static_cast<int>(std::nearbyint(
                (matrix[static_cast<size_t>(row) * columns + column] - low) /
                step));
            store_code(row, column, std::clamp(code, 0, maximum));
        }
    }
}

} // namespace

size_t LaplaceKVQ4Tile::encoded_bytes(int head_dim) {
    size_t metadata = (static_cast<size_t>(3) * head_dim +
                       static_cast<size_t>(3) * kTokens) * sizeof(uint16_t);
    size_t key_codes = static_cast<size_t>(kTokens) * head_dim / 2;
    size_t value_codes = static_cast<size_t>(kTokens) * head_dim / 4;
    return metadata + key_codes + value_codes;
}

size_t LaplaceKVQ4Tile::storage_words(int head_dim) {
    return (encoded_bytes(head_dim) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
}

bool LaplaceKVQ4Tile::init(int head_dim) {
    if (head_dim < 32 || head_dim > 512 || head_dim % 16 != 0) return false;
    storage_.assign(storage_words(head_dim), 0);
    return init(head_dim, storage_.data(), false);
}

bool LaplaceKVQ4Tile::init(int head_dim, uint32_t* storage, bool sealed) {
    if (head_dim < 32 || head_dim > 512 || head_dim % 16 != 0 || !storage) {
        return false;
    }
    head_dim_ = head_dim;
    sealed_ = sealed;
    uint16_t* cursor = reinterpret_cast<uint16_t*>(storage);
    ka_ = cursor;
    cursor += head_dim;
    kb_ = cursor;
    cursor += head_dim;
    kc_ = cursor;
    cursor += kTokens;
    va_ = cursor;
    cursor += kTokens;
    vb_ = cursor;
    cursor += kTokens;
    vc_ = cursor;
    cursor += head_dim;
    key_codes_ = reinterpret_cast<uint8_t*>(cursor);
    value_codes_ = key_codes_ + static_cast<size_t>(kTokens) * head_dim / 2;
    return true;
}

bool LaplaceKVQ4Tile::seal(const float* keys_wh, const float* values_wh) {
    if (!keys_wh || !values_wh || head_dim_ == 0) return false;
    std::memset(key_codes_, 0,
                static_cast<size_t>(kTokens) * head_dim_ / 2);
    std::memset(value_codes_, 0,
                static_cast<size_t>(kTokens) * head_dim_ / 4);

    std::vector<double> key_matrix(
        static_cast<size_t>(head_dim_) * kTokens);
    for (int dim = 0; dim < head_dim_; dim++)
        for (int token = 0; token < kTokens; token++)
            key_matrix[static_cast<size_t>(dim) * kTokens + token] =
                keys_wh[static_cast<size_t>(token) * head_dim_ + dim];
    encode_affine(
        key_matrix, head_dim_, kTokens, 15, ka_, kb_, kc_,
        [&](int dim, int token, int code) {
            set_k4(
                key_codes_, static_cast<size_t>(token) * head_dim_ + dim,
                static_cast<uint8_t>(code));
        });

    std::vector<double> value_matrix(
        values_wh, values_wh + static_cast<size_t>(kTokens) * head_dim_);
    encode_affine(
        value_matrix, kTokens, head_dim_, 3, va_, vb_, vc_,
        [&](int token, int dim, int code) {
            set_v2(
                value_codes_, static_cast<size_t>(dim) * kTokens + token,
                static_cast<uint8_t>(code));
        });
    sealed_ = true;
    return true;
}

void LaplaceKVQ4Tile::load_key_wh(int token, float* output_wh) const {
    for (int dim = 0; dim < head_dim_; dim++) {
        int code = get_k4(
            key_codes_, static_cast<size_t>(token) * head_dim_ + dim);
        output_wh[dim] =
            (code * fp16_to_fp32(ka_[dim]) + fp16_to_fp32(kb_[dim])) *
            fp16_to_fp32(kc_[token]);
    }
}

void LaplaceKVQ4Tile::load_value_wh(int token, float* output_wh) const {
    for (int dim = 0; dim < head_dim_; dim++) {
        int code = get_v2(
            value_codes_, static_cast<size_t>(dim) * kTokens + token);
        output_wh[dim] =
            (code * fp16_to_fp32(va_[token]) + fp16_to_fp32(vb_[token])) *
            fp16_to_fp32(vc_[dim]);
    }
}

void LaplaceKVQ4Tile::dot_keys(const float* query_wh, float* scores) const {
    for (int token = 0; token < kTokens; token++) {
        float sum = 0.0f;
        for (int dim = 0; dim < head_dim_; dim++) {
            int code = get_k4(
                key_codes_, static_cast<size_t>(token) * head_dim_ + dim);
            sum += query_wh[dim] *
                (code * fp16_to_fp32(ka_[dim]) + fp16_to_fp32(kb_[dim]));
        }
        scores[token] = sum * fp16_to_fp32(kc_[token]);
    }
}

void LaplaceKVQ4Tile::add_values(
        const float* weights, float* output_wh) const {
    float bias = 0.0f;
    for (int token = 0; token < kTokens; token++)
        bias += weights[token] * fp16_to_fp32(vb_[token]);
    for (int dim = 0; dim < head_dim_; dim++) {
        float sum = bias;
        for (int token = 0; token < kTokens; token++) {
            int code = get_v2(
                value_codes_, static_cast<size_t>(dim) * kTokens + token);
            sum += weights[token] * fp16_to_fp32(va_[token]) * code;
        }
        output_wh[dim] += sum * fp16_to_fp32(vc_[dim]);
    }
}

size_t LaplaceKVQ4Tile::storage_bytes() const {
    return storage_.capacity() * sizeof(uint32_t);
}

bool LaplaceKVQ4::init(int n_layers, int n_kv_heads, int head_dim,
                       int capacity, bool streaming) {
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
    tiles_per_head_ =
        (capacity + LaplaceKVQ4Tile::kTokens - 1) /
        LaplaceKVQ4Tile::kTokens;
    tile_words_ = LaplaceKVQ4Tile::storage_words(head_dim);
    size_t heads = static_cast<size_t>(n_layers) * n_kv_heads;

    if (streaming_) {
        const char* temp = std::getenv("TMPDIR");
        std::string pattern = temp && *temp ? temp : "/tmp";
        if (pattern.back() != '/') pattern.push_back('/');
        pattern += "laplace-kv-q4-XXXXXX";
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
        archive_bytes_ = heads * tiles_per_head_ * tile_words_ *
                         sizeof(uint32_t);
        if (ftruncate(archive_fd_, static_cast<off_t>(archive_bytes_)) != 0) {
            clear();
            return false;
        }
    } else {
        resident_storage_.resize(heads);
    }
    sealed_tiles_.assign(heads * tiles_per_head_, 0);
    size_t tail_values =
        heads * LaplaceKVQ4Tile::kTokens * head_dim;
    k_tail_.assign(tail_values, 0.0f);
    v_tail_.assign(tail_values, 0.0f);
    return true;
}

void LaplaceKVQ4::clear() {
    if (archive_fd_ >= 0) close(archive_fd_);
    archive_fd_ = -1;
    archive_bytes_ = 0;
    resident_storage_.clear();
    sealed_tiles_.clear();
    k_tail_.clear();
    v_tail_.clear();
    stream_calls_.store(0);
    archive_read_bytes_.store(0);
    archive_write_bytes_.store(0);
    n_layers_ = 0;
    n_kv_heads_ = 0;
    head_dim_ = 0;
    capacity_ = 0;
    tiles_per_head_ = 0;
    tile_words_ = 0;
    streaming_ = false;
}

size_t LaplaceKVQ4::head_index(int layer, int head) const {
    return static_cast<size_t>(layer) * n_kv_heads_ + head;
}

size_t LaplaceKVQ4::tile_index(int layer, int head, int tile) const {
    return head_index(layer, head) * tiles_per_head_ + tile;
}

float* LaplaceKVQ4::tail_k(int layer, int head, int offset) {
    size_t index =
        (head_index(layer, head) * LaplaceKVQ4Tile::kTokens + offset) *
        head_dim_;
    return k_tail_.data() + index;
}

float* LaplaceKVQ4::tail_v(int layer, int head, int offset) {
    size_t index =
        (head_index(layer, head) * LaplaceKVQ4Tile::kTokens + offset) *
        head_dim_;
    return v_tail_.data() + index;
}

const float* LaplaceKVQ4::tail_k(int layer, int head, int offset) const {
    size_t index =
        (head_index(layer, head) * LaplaceKVQ4Tile::kTokens + offset) *
        head_dim_;
    return k_tail_.data() + index;
}

const float* LaplaceKVQ4::tail_v(int layer, int head, int offset) const {
    size_t index =
        (head_index(layer, head) * LaplaceKVQ4Tile::kTokens + offset) *
        head_dim_;
    return v_tail_.data() + index;
}

size_t LaplaceKVQ4::archive_offset(size_t head, int tile) const {
    return (head * tiles_per_head_ + tile) * tile_words_ * sizeof(uint32_t);
}

bool LaplaceKVQ4::read_archive(
        size_t head, int tile, uint32_t* output) const {
    size_t bytes = tile_words_ * sizeof(uint32_t);
    size_t done = 0;
    while (done < bytes) {
        ssize_t result = pread(
            archive_fd_, reinterpret_cast<uint8_t*>(output) + done,
            bytes - done,
            static_cast<off_t>(archive_offset(head, tile) + done));
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

bool LaplaceKVQ4::write_archive(
        size_t head, int tile, const uint32_t* input) {
    size_t bytes = tile_words_ * sizeof(uint32_t);
    size_t done = 0;
    while (done < bytes) {
        ssize_t result = pwrite(
            archive_fd_, reinterpret_cast<const uint8_t*>(input) + done,
            bytes - done,
            static_cast<off_t>(archive_offset(head, tile) + done));
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
    return true;
}

void LaplaceKVQ4::store_k_wh(
        int layer, int head, int pos, const float* key_wh) {
    if (pos < 0 || pos >= capacity_) return;
    int offset = pos % LaplaceKVQ4Tile::kTokens;
    std::memcpy(
        tail_k(layer, head, offset), key_wh, sizeof(float) * head_dim_);
}

void LaplaceKVQ4::store_v_wh(
        int layer, int head, int pos, const float* value_wh) {
    if (pos < 0 || pos >= capacity_) return;
    int offset = pos % LaplaceKVQ4Tile::kTokens;
    std::memcpy(
        tail_v(layer, head, offset), value_wh, sizeof(float) * head_dim_);
    if (offset != LaplaceKVQ4Tile::kTokens - 1) return;

    int tile = pos / LaplaceKVQ4Tile::kTokens;
    size_t head_id = head_index(layer, head);
    if (streaming_) {
        std::vector<uint32_t> storage(tile_words_);
        LaplaceKVQ4Tile view;
        if (!view.init(head_dim_, storage.data(), false) ||
            !view.seal(tail_k(layer, head, 0), tail_v(layer, head, 0)) ||
            !write_archive(head_id, tile, storage.data())) {
            std::abort();
        }
    } else {
        auto& storage = resident_storage_[head_id];
        storage.resize(static_cast<size_t>(tile + 1) * tile_words_);
        LaplaceKVQ4Tile view;
        if (!view.init(
                head_dim_,
                storage.data() + static_cast<size_t>(tile) * tile_words_,
                false) ||
            !view.seal(tail_k(layer, head, 0), tail_v(layer, head, 0))) {
            std::abort();
        }
    }
    sealed_tiles_[tile_index(layer, head, tile)] = 1;
}

void LaplaceKVQ4::load_k_wh(
        int layer, int head, int pos, float* key_wh) const {
    if (pos < 0 || pos >= capacity_) return;
    int tile = pos / LaplaceKVQ4Tile::kTokens;
    int offset = pos % LaplaceKVQ4Tile::kTokens;
    if (!sealed_tiles_[tile_index(layer, head, tile)]) {
        std::memcpy(
            key_wh, tail_k(layer, head, offset), sizeof(float) * head_dim_);
        return;
    }
    std::vector<uint32_t> archive;
    uint32_t* data;
    if (streaming_) {
        archive.resize(tile_words_);
        if (!read_archive(head_index(layer, head), tile, archive.data()))
            std::abort();
        data = archive.data();
    } else {
        data = const_cast<uint32_t*>(
            resident_storage_[head_index(layer, head)].data()) +
            static_cast<size_t>(tile) * tile_words_;
    }
    LaplaceKVQ4Tile view;
    view.init(head_dim_, data, true);
    view.load_key_wh(offset, key_wh);
}

void LaplaceKVQ4::load_v_wh(
        int layer, int head, int pos, float* value_wh) const {
    if (pos < 0 || pos >= capacity_) return;
    int tile = pos / LaplaceKVQ4Tile::kTokens;
    int offset = pos % LaplaceKVQ4Tile::kTokens;
    if (!sealed_tiles_[tile_index(layer, head, tile)]) {
        std::memcpy(
            value_wh, tail_v(layer, head, offset), sizeof(float) * head_dim_);
        return;
    }
    std::vector<uint32_t> archive;
    uint32_t* data;
    if (streaming_) {
        archive.resize(tile_words_);
        if (!read_archive(head_index(layer, head), tile, archive.data()))
            std::abort();
        data = archive.data();
    } else {
        data = const_cast<uint32_t*>(
            resident_storage_[head_index(layer, head)].data()) +
            static_cast<size_t>(tile) * tile_words_;
    }
    LaplaceKVQ4Tile view;
    view.init(head_dim_, data, true);
    view.load_value_wh(offset, value_wh);
}

void LaplaceKVQ4::dot_keys_wh(
        int layer, int head, int n_tokens, const float* query_wh,
        float* scores, int first_token) const {
    n_tokens = std::clamp(n_tokens, 0, capacity_);
    first_token = std::clamp(first_token, 0, n_tokens);
    float tile_scores[LaplaceKVQ4Tile::kTokens];
    std::vector<uint32_t> archive(streaming_ ? tile_words_ : 0);
    int token = first_token;
    while (token < n_tokens) {
        int tile = token / LaplaceKVQ4Tile::kTokens;
        int offset = token % LaplaceKVQ4Tile::kTokens;
        int count = std::min(
            LaplaceKVQ4Tile::kTokens - offset, n_tokens - token);
        if (sealed_tiles_[tile_index(layer, head, tile)]) {
            uint32_t* data;
            if (streaming_) {
                if (!read_archive(
                        head_index(layer, head), tile, archive.data()))
                    std::abort();
                data = archive.data();
            } else {
                data = const_cast<uint32_t*>(
                    resident_storage_[head_index(layer, head)].data()) +
                    static_cast<size_t>(tile) * tile_words_;
            }
            LaplaceKVQ4Tile view;
            view.init(head_dim_, data, true);
            view.dot_keys(query_wh, tile_scores);
            std::copy_n(tile_scores + offset, count, scores + token);
        } else {
            for (int index = 0; index < count; index++) {
                scores[token + index] = ops::dot(
                    query_wh, tail_k(layer, head, offset + index), head_dim_);
            }
        }
        token += count;
    }
}

void LaplaceKVQ4::add_values_wh(
        int layer, int head, int n_tokens, const float* weights,
        float* output_wh, int first_token) const {
    n_tokens = std::clamp(n_tokens, 0, capacity_);
    first_token = std::clamp(first_token, 0, n_tokens);
    float tile_weights[LaplaceKVQ4Tile::kTokens];
    std::vector<uint32_t> archive(streaming_ ? tile_words_ : 0);
    int token = first_token;
    while (token < n_tokens) {
        int tile = token / LaplaceKVQ4Tile::kTokens;
        int offset = token % LaplaceKVQ4Tile::kTokens;
        int count = std::min(
            LaplaceKVQ4Tile::kTokens - offset, n_tokens - token);
        if (sealed_tiles_[tile_index(layer, head, tile)]) {
            uint32_t* data;
            if (streaming_) {
                if (!read_archive(
                        head_index(layer, head), tile, archive.data()))
                    std::abort();
                data = archive.data();
            } else {
                data = const_cast<uint32_t*>(
                    resident_storage_[head_index(layer, head)].data()) +
                    static_cast<size_t>(tile) * tile_words_;
            }
            std::fill_n(
                tile_weights, LaplaceKVQ4Tile::kTokens, 0.0f);
            std::copy_n(weights + token, count, tile_weights + offset);
            LaplaceKVQ4Tile view;
            view.init(head_dim_, data, true);
            view.add_values(tile_weights, output_wh);
        } else {
            for (int index = 0; index < count; index++) {
                ops::axpy(
                    output_wh, weights[token + index],
                    tail_v(layer, head, offset + index), head_dim_);
            }
        }
        token += count;
    }
}

void LaplaceKVQ4::attention_wh(
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
        scores[token] = std::exp(scores[token] * logit_scale - maximum);
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

void LaplaceKVQ4::attention_batch_wh(
        int layer, int head, int count, const int* n_tokens,
        const float* const* queries_wh, float logit_scale,
        float* const* outputs_wh, int first_token) const {
    for (int query = 0; query < count; query++)
        attention_wh(layer, head, n_tokens[query], queries_wh[query],
                     logit_scale, outputs_wh[query], first_token);
}

size_t LaplaceKVQ4::encoded_bytes(int n_tokens) const {
    int tokens = std::clamp(n_tokens, 0, capacity_);
    size_t heads = static_cast<size_t>(n_layers_) * n_kv_heads_;
    size_t sealed =
        static_cast<size_t>(tokens / LaplaceKVQ4Tile::kTokens) *
        LaplaceKVQ4Tile::encoded_bytes(head_dim_);
    size_t tail =
        static_cast<size_t>(tokens % LaplaceKVQ4Tile::kTokens) *
        head_dim_ * 2 * sizeof(float);
    size_t states = static_cast<size_t>(
        (tokens + LaplaceKVQ4Tile::kTokens - 1) /
        LaplaceKVQ4Tile::kTokens);
    return heads * (sealed + tail + states);
}

size_t LaplaceKVQ4::storage_bytes() const {
    size_t bytes =
        (k_tail_.capacity() + v_tail_.capacity()) * sizeof(float) +
        sealed_tiles_.capacity() * sizeof(sealed_tiles_[0]) +
        resident_storage_.capacity() * sizeof(resident_storage_[0]);
    for (const auto& storage : resident_storage_)
        bytes += storage.capacity() * sizeof(uint32_t);
    return bytes;
}

size_t LaplaceKVQ4::archive_read_buffer_bytes() const {
    return streaming_ && tile_words_ > 0
         ? tile_words_ * sizeof(uint32_t) : 0;
}

} // namespace Laplace
