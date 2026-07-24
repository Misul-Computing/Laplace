// laplace_kv_q4.h - fully counted fixed-width K4/V2 research codec.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Laplace {

class LaplaceKVQ4Tile {
public:
    static constexpr int kTokens = 128;

    bool init(int head_dim);
    bool init(int head_dim, uint32_t* storage, bool sealed);
    bool seal(const float* keys_wh, const float* values_wh);

    void dot_keys(const float* query_wh, float* scores) const;
    void add_values(const float* weights, float* output_wh) const;
    void load_key_wh(int token, float* output_wh) const;
    void load_value_wh(int token, float* output_wh) const;

    static size_t encoded_bytes(int head_dim);
    static size_t storage_words(int head_dim);
    size_t storage_bytes() const;

private:
    int head_dim_ = 0;
    bool sealed_ = false;
    std::vector<uint32_t> storage_;
    uint16_t* ka_ = nullptr;
    uint16_t* kb_ = nullptr;
    uint16_t* kc_ = nullptr;
    uint16_t* va_ = nullptr;
    uint16_t* vb_ = nullptr;
    uint16_t* vc_ = nullptr;
    uint8_t* key_codes_ = nullptr;
    uint8_t* value_codes_ = nullptr;
};

class LaplaceKVQ4 {
public:
    ~LaplaceKVQ4() { clear(); }

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
    size_t encoded_bytes(int n_tokens) const;
    size_t storage_bytes() const;
    size_t archive_read_buffer_bytes() const;
    size_t archive_bytes() const { return archive_bytes_; }
    uint64_t stream_calls() const { return stream_calls_.load(); }
    uint64_t archive_read_bytes() const { return archive_read_bytes_.load(); }
    uint64_t archive_write_bytes() const { return archive_write_bytes_.load(); }

private:
    size_t head_index(int layer, int head) const;
    size_t tile_index(int layer, int head, int tile) const;
    float* tail_k(int layer, int head, int offset);
    float* tail_v(int layer, int head, int offset);
    const float* tail_k(int layer, int head, int offset) const;
    const float* tail_v(int layer, int head, int offset) const;
    size_t archive_offset(size_t head, int tile) const;
    bool read_archive(size_t head, int tile, uint32_t* output) const;
    bool write_archive(size_t head, int tile, const uint32_t* input);

    int n_layers_ = 0;
    int n_kv_heads_ = 0;
    int head_dim_ = 0;
    int capacity_ = 0;
    int tiles_per_head_ = 0;
    size_t tile_words_ = 0;
    bool streaming_ = false;
    std::vector<std::vector<uint32_t>> resident_storage_;
    std::vector<uint8_t> sealed_tiles_;
    std::vector<float> k_tail_;
    std::vector<float> v_tail_;
    int archive_fd_ = -1;
    size_t archive_bytes_ = 0;
    mutable std::atomic<uint64_t> stream_calls_{0};
    mutable std::atomic<uint64_t> archive_read_bytes_{0};
    std::atomic<uint64_t> archive_write_bytes_{0};
};

} // namespace Laplace
