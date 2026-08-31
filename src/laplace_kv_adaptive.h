// laplace_kv_adaptive.h - data-driven K4/V2 with K8/V6 and FP16 promotion.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "laplace_kv.h"
#include "laplace_kv_q4.h"

namespace Laplace {

enum class LaplaceKVAdaptiveFormat : uint8_t {
    MUTABLE,
    K4_V2,
    K8_V6,
    FP16,
};

struct LaplaceKVTileError {
    double key_rms = 0.0;
    double key_max = 0.0;
    double value_rms = 0.0;
    double value_max = 0.0;
};

// Compression is opt-in until model-level parity qualifies a codec. The
// public KVCache factory uses the default, which is FP16-only.
struct CompressionEligibility {
    bool q4 = false;
    bool k8 = false;
};

bool laplace_kv_prefer_q4(const LaplaceKVTileError& q4,
                          const LaplaceKVTileError& k8);
bool laplace_kv_accept_k8(const LaplaceKVTileError& error);
bool codec_accepts_tile(LaplaceKVAdaptiveFormat codec, int tile_tokens);

class LaplaceKVAdaptive {
public:
    // One adaptive tile is the complete Q4 contract and contains two K8
    // blocks. A mutable tail remains uncompressed until all 128 tokens exist.
    static constexpr int kTokens = LaplaceKVQ4Tile::kTokens;

    ~LaplaceKVAdaptive() { clear(); }

    bool init(int n_layers, int n_kv_heads, int head_dim, int capacity,
              bool streaming = false,
              CompressionEligibility eligibility = {});
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
    LaplaceKVAdaptiveFormat tile_format(
        int layer, int head, int tile) const;
    uint64_t q4_tiles() const { return q4_tiles_.load(); }
    uint64_t k8_tiles() const { return k8_tiles_.load(); }
    uint64_t fp16_tiles() const { return fp16_tiles_.load(); }
    size_t encoded_bytes(int n_tokens) const;
    size_t storage_bytes() const;
    size_t archive_read_buffer_bytes() const;
    size_t archive_bytes() const { return archive_bytes_; }
    uint64_t stream_calls() const { return stream_calls_.load(); }
    uint64_t archive_read_bytes() const {
        return archive_read_bytes_.load();
    }
    uint64_t archive_write_bytes() const {
        return archive_write_bytes_.load();
    }

private:
    struct TileDescriptor {
        uint64_t offset = 0;
        uint32_t words = 0;
        LaplaceKVAdaptiveFormat format =
            LaplaceKVAdaptiveFormat::MUTABLE;
    };

    size_t head_index(int layer, int head) const;
    size_t tile_index(int layer, int head, int tile) const;
    float* tail_k(int layer, int head, int offset);
    float* tail_v(int layer, int head, int offset);
    const float* tail_k(int layer, int head, int offset) const;
    const float* tail_v(int layer, int head, int offset) const;
    bool seal_tile(int layer, int head, int tile);
    bool write_archive(const uint32_t* input, size_t words,
                       uint64_t& offset);
    bool read_archive(const TileDescriptor& descriptor,
                      uint32_t* output) const;
    const uint32_t* resident_payload(
        size_t head, const TileDescriptor& descriptor) const;

    int n_layers_ = 0;
    int n_kv_heads_ = 0;
    int head_dim_ = 0;
    int capacity_ = 0;
    int tiles_per_head_ = 0;
    size_t q4_words_ = 0;
    size_t k8_words_ = 0;
    size_t fp16_words_ = 0;
    bool streaming_ = false;
    CompressionEligibility eligibility_;
    std::vector<std::vector<uint32_t>> resident_arenas_;
    std::vector<TileDescriptor> descriptors_;
    std::vector<float> k_tail_;
    std::vector<float> v_tail_;
    int archive_fd_ = -1;
    size_t archive_bytes_ = 0;
    mutable std::mutex archive_mutex_;
    std::atomic<uint64_t> q4_tiles_{0};
    std::atomic<uint64_t> k8_tiles_{0};
    std::atomic<uint64_t> fp16_tiles_{0};
    mutable std::atomic<uint64_t> stream_calls_{0};
    mutable std::atomic<uint64_t> archive_read_bytes_{0};
    std::atomic<uint64_t> archive_write_bytes_{0};
};

} // namespace Laplace
