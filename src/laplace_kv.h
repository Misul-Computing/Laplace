// laplace_kv.h - sealed 64-token K8/V6 chunks.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Laplace {

// Quantize a float vector to signed int8. Returns the dequantization scale.
float laplace_kv_quantize_q8(const float* input, int size, int8_t* output);

class LaplaceKVTile {
public:
    static constexpr int kTokens = 64;

    LaplaceKVTile() = default;
    LaplaceKVTile(const LaplaceKVTile&) = delete;
    LaplaceKVTile& operator=(const LaplaceKVTile&) = delete;
    LaplaceKVTile(LaplaceKVTile&&) = default;
    LaplaceKVTile& operator=(LaplaceKVTile&&) = default;

    bool init(int head_dim);
    bool init(int head_dim, uint32_t* storage, bool sealed);
    bool seal(const float* keys_wh, const float* values_wh);
    static size_t storage_words(int head_dim);

    // Keys are token-major. Query quantization is shared across chunks.
    void dot_keys(const int8_t* query_q8, float query_scale,
                  float* scores) const;

    // Values are coordinate-major. Softmax weights are quantized once here.
    void add_values(const float* weights, float* output_wh) const;
    void load_key_wh(int token, float* output_wh) const;
    void load_value_wh(int token, float* output_wh) const;

    size_t storage_bytes() const;

private:
    int head_dim_ = 0;
    bool sealed_ = false;
    std::vector<uint32_t> storage_;
    float* k_scales_ = nullptr;
    float* v_scales_ = nullptr;
    uint8_t* k_codes_ = nullptr;
    uint8_t* v_codes_ = nullptr;
};

// Whole-cache storage built from immutable K8/V6 chunks and one mutable WH tail
// per layer and KV head. Streaming mode keeps only the tail in RAM and reads the
// full archive from an unlinked file once per query batch.
class LaplaceKV {
public:
    ~LaplaceKV() { clear(); }
    bool init(int n_layers, int n_kv_heads, int head_dim, int capacity,
              bool streaming = false);
    void clear();

    void store_k_wh(int layer, int head, int pos, const float* key_wh);
    void store_v_wh(int layer, int head, int pos, const float* value_wh);
    void load_k_wh(int layer, int head, int pos, float* key_wh) const;
    void load_v_wh(int layer, int head, int pos, float* value_wh) const;

    void dot_keys_wh(int layer, int head, int n_tokens,
                     const float* query_wh, float* scores,
                     int first_token = 0) const;
    void add_values_wh(int layer, int head, int n_tokens,
                       const float* weights, float* output_wh,
                       int first_token = 0) const;
    void attention_wh(int layer, int head, int n_tokens,
                      const float* query_wh, float logit_scale,
                      float* output_wh, int first_token = 0) const;
    void attention_batch_wh(int layer, int head, int count,
                            const int* n_tokens,
                            const float* const* queries_wh,
                            float logit_scale, float* const* outputs_wh,
                            int first_token = 0) const;
    bool uses_rotation() const {
        return (head_dim_ & (head_dim_ - 1)) == 0;
    }
    bool streaming() const { return streaming_; }
    // Logical codec bytes: sealed tile payloads, the populated part of the
    // mutable FP32 tail, and one state byte per live tile. Tile padding and
    // codec metadata are already included in chunk_words_. Allocator slack is
    // reported by storage_bytes() instead.
    size_t encoded_bytes(int n_tokens) const {
        int tokens = n_tokens < 0 ? 0
                   : n_tokens > capacity_ ? capacity_ : n_tokens;
        size_t heads = static_cast<size_t>(n_layers_) * n_kv_heads_;
        size_t sealed = static_cast<size_t>(tokens / LaplaceKVTile::kTokens)
                      * chunk_words_ * sizeof(uint32_t);
        size_t tail = static_cast<size_t>(tokens % LaplaceKVTile::kTokens)
                    * head_dim_ * 2 * sizeof(float);
        size_t states = static_cast<size_t>(
            (tokens + LaplaceKVTile::kTokens - 1) / LaplaceKVTile::kTokens);
        return heads * (sealed + tail + states);
    }
    size_t storage_bytes() const;
    size_t archive_read_buffer_bytes() const;
    size_t archive_bytes() const { return archive_bytes_; }
    uint64_t stream_calls() const { return stream_calls_.load(); }
    uint64_t archive_read_bytes() const { return archive_read_bytes_.load(); }
    uint64_t archive_write_bytes() const { return archive_write_bytes_.load(); }

private:
    size_t head_index(int layer, int head) const;
    size_t chunk_index(int layer, int head, int chunk) const;
    float* tail_k(int layer, int head, int offset);
    float* tail_v(int layer, int head, int offset);
    const float* tail_k(int layer, int head, int offset) const;
    const float* tail_v(int layer, int head, int offset) const;
    size_t archive_offset(size_t head, int chunk) const;
    bool read_archive(size_t head, int chunk, uint32_t* output) const;
    bool read_archive(size_t head, int chunk, int count,
                      uint32_t* output) const;
    bool write_archive(size_t head, int chunk, const uint32_t* input);

    int n_layers_ = 0;
    int n_kv_heads_ = 0;
    int head_dim_ = 0;
    int capacity_ = 0;
    int chunks_per_head_ = 0;
    size_t chunk_words_ = 0;
    bool streaming_ = false;
    std::vector<std::vector<uint32_t>> resident_storage_;
    std::vector<uint8_t> sealed_chunks_;
    std::vector<float> k_tail_;
    std::vector<float> v_tail_;

    int archive_fd_ = -1;
    size_t archive_bytes_ = 0;
    mutable std::atomic<uint64_t> stream_calls_{0};
    mutable std::atomic<uint64_t> archive_read_bytes_{0};
    std::atomic<uint64_t> archive_write_bytes_{0};
};

} // namespace Laplace
