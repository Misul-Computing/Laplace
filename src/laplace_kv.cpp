// laplace_kv.cpp - sealed K8/V6 chunks with SDOT decode kernels.
#include "laplace_kv.h"

#include "fp16.h"
#include "ops.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#if defined(LAPLACE_KV_CAPTURE)
#include <mutex>
#endif
#include <string>
#include <unistd.h>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include <sys/sysctl.h>
#endif

namespace Laplace {

namespace {
constexpr size_t kArchiveReadBytes = 16 * 1024 * 1024;
constexpr int kMaxArchiveReadChunks = 512;

#if defined(LAPLACE_KV_CAPTURE)
struct KVTraceHeader {
    uint32_t magic;
    uint32_t type;
    int32_t layer;
    int32_t head;
    int32_t position;
    int32_t head_dim;
    int32_t first_token;
    float logit_scale;
    int32_t query_index;
};

static_assert(sizeof(KVTraceHeader) == 36);

class KVTraceSink {
public:
    KVTraceSink() {
        const char* path = std::getenv("LAPLACE_KV_CAPTURE");
        if (!path || !path[0]) return;
        file_ = std::fopen(path, "wb");
        const char* at = std::getenv("LAPLACE_KV_CAPTURE_AT");
        if (at) capture_at_ = std::atoi(at);
    }

    ~KVTraceSink() {
        if (file_) std::fclose(file_);
    }

    void vector(uint32_t type, int layer, int head, int position,
                int head_dim, const float* data) {
        if (!file_) return;
        KVTraceHeader header{
            0x4c4b5654, type, layer, head, position, head_dim, 0, 0.0f, 0
        };
        std::lock_guard<std::mutex> lock(mutex_);
        std::fwrite(&header, sizeof(header), 1, file_);
        std::fwrite(data, sizeof(float), head_dim, file_);
    }

    void query(int layer, int head, int n_tokens, int head_dim,
               int first_token, float logit_scale, int query_index,
               const float* query, const float* output) {
        if (!file_ || (capture_at_ > 0 && n_tokens != capture_at_)) return;
        KVTraceHeader header{
            0x4c4b5654, 3, layer, head, n_tokens, head_dim,
            first_token, logit_scale, query_index
        };
        std::lock_guard<std::mutex> lock(mutex_);
        std::fwrite(&header, sizeof(header), 1, file_);
        std::fwrite(query, sizeof(float), head_dim, file_);
        std::fwrite(output, sizeof(float), head_dim, file_);
    }

private:
    FILE* file_ = nullptr;
    int capture_at_ = 0;
    std::mutex mutex_;
};

KVTraceSink& kv_trace_sink() {
    static KVTraceSink sink;
    return sink;
}
#endif
}

float laplace_kv_quantize_q8(const float* input, int size, int8_t* output) {
    float max_abs = 0.0f;
#if defined(__aarch64__)
    float32x4_t vmax = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= size; i += 4) {
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(input + i)));
    }
    max_abs = vmaxvq_f32(vmax);
    for (; i < size; i++) max_abs = std::max(max_abs, std::fabs(input[i]));
#else
    for (int i = 0; i < size; i++) {
        max_abs = std::max(max_abs, std::fabs(input[i]));
    }
#endif
    float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    float inv = 1.0f / scale;
#if defined(__aarch64__)
    i = 0;
    float32x4_t vinv = vdupq_n_f32(inv);
    for (; i + 16 <= size; i += 16) {
        int32x4_t q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(input + i), vinv));
        int32x4_t q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(input + i + 4), vinv));
        int32x4_t q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(input + i + 8), vinv));
        int32x4_t q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(input + i + 12), vinv));
        int16x8_t q01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        int16x8_t q23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
        vst1q_s8(output + i,
                 vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23)));
    }
    for (; i < size; i++) {
        int value = static_cast<int>(std::nearbyint(input[i] * inv));
        output[i] = static_cast<int8_t>(std::clamp(value, -127, 127));
    }
#else
    for (int i = 0; i < size; i++) {
        int value = static_cast<int>(std::nearbyint(input[i] * inv));
        output[i] = static_cast<int8_t>(std::clamp(value, -127, 127));
    }
#endif
    return scale;
}

static bool cpu_has_i8mm() {
#if defined(__APPLE__) && defined(__aarch64__)
    if (std::getenv("LAPLACE_NOSIMD")) return false;
    static const bool supported = [] {
        int value = 0;
        size_t size = sizeof(value);
        return sysctlbyname("hw.optional.arm.FEAT_I8MM",
                            &value, &size, nullptr, 0) == 0 && value != 0;
    }();
    return supported;
#else
    return false;
#endif
}

static float quantize_u8(const float* input, int size, uint8_t* output) {
    float max_value = 0.0f;
#if defined(__aarch64__)
    float32x4_t vmax = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= size; i += 4) {
        vmax = vmaxq_f32(vmax, vld1q_f32(input + i));
    }
    max_value = vmaxvq_f32(vmax);
    for (; i < size; i++) max_value = std::max(max_value, input[i]);
#else
    for (int i = 0; i < size; i++) max_value = std::max(max_value, input[i]);
#endif
    float scale = max_value > 0.0f ? max_value / 255.0f : 1.0f;
    float inv = 1.0f / scale;
#if defined(__aarch64__)
    i = 0;
    float32x4_t vinv = vdupq_n_f32(inv);
    for (; i + 16 <= size; i += 16) {
        uint32x4_t q0 = vcvtnq_u32_f32(vmulq_f32(vld1q_f32(input + i), vinv));
        uint32x4_t q1 = vcvtnq_u32_f32(vmulq_f32(vld1q_f32(input + i + 4), vinv));
        uint32x4_t q2 = vcvtnq_u32_f32(vmulq_f32(vld1q_f32(input + i + 8), vinv));
        uint32x4_t q3 = vcvtnq_u32_f32(vmulq_f32(vld1q_f32(input + i + 12), vinv));
        uint16x8_t q01 = vcombine_u16(vqmovn_u32(q0), vqmovn_u32(q1));
        uint16x8_t q23 = vcombine_u16(vqmovn_u32(q2), vqmovn_u32(q3));
        vst1q_u8(output + i,
                 vcombine_u8(vqmovn_u16(q01), vqmovn_u16(q23)));
    }
    for (; i < size; i++) {
        int value = static_cast<int>(std::nearbyint(input[i] * inv));
        output[i] = static_cast<uint8_t>(std::clamp(value, 0, 255));
    }
#else
    for (int i = 0; i < size; i++) {
        int value = static_cast<int>(std::nearbyint(input[i] * inv));
        output[i] = static_cast<uint8_t>(std::clamp(value, 0, 255));
    }
#endif
    return scale;
}

size_t LaplaceKVTile::storage_words(int head_dim) {
    size_t scale_bytes = static_cast<size_t>(kTokens) * sizeof(float);
    size_t k_code_bytes = static_cast<size_t>(kTokens) * head_dim;
    size_t v_code_bytes = static_cast<size_t>(head_dim) * kTokens * 3 / 4;
    size_t storage_bytes = 2 * scale_bytes + k_code_bytes + v_code_bytes;
    return (storage_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
}

bool LaplaceKVTile::init(int head_dim) {
    if (head_dim < 32 || head_dim > 512 || head_dim % 16 != 0) return false;
    storage_.assign(storage_words(head_dim), 0);
    return init(head_dim, storage_.data(), false);
}

bool LaplaceKVTile::init(int head_dim, uint32_t* storage, bool sealed) {
    if (head_dim < 32 || head_dim > 512 || head_dim % 16 != 0 || !storage) {
        return false;
    }
    head_dim_ = head_dim;
    size_t scale_bytes = static_cast<size_t>(kTokens) * sizeof(float);
    size_t k_code_bytes = static_cast<size_t>(kTokens) * head_dim;
    uint8_t* cursor = reinterpret_cast<uint8_t*>(storage);
    k_scales_ = reinterpret_cast<float*>(cursor);
    cursor += scale_bytes;
    v_scales_ = reinterpret_cast<float*>(cursor);
    cursor += scale_bytes;
    k_codes_ = cursor;
    cursor += k_code_bytes;
    v_codes_ = cursor;
    sealed_ = sealed;
    return true;
}

bool LaplaceKVTile::seal(const float* keys_wh, const float* values_wh) {
    if (head_dim_ == 0 || !keys_wh || !values_wh
        || !k_scales_ || !v_scales_) return false;

    alignas(16) int8_t q8_values[512];
    std::memset(v_codes_, 0,
                static_cast<size_t>(head_dim_) * kTokens * 3 / 4);

    for (int token = 0; token < kTokens; token++) {
        const float* key = keys_wh + static_cast<size_t>(token) * head_dim_;
        int8_t* codes = reinterpret_cast<int8_t*>(
            k_codes_ + static_cast<size_t>(token) * head_dim_);
        k_scales_[token] = laplace_kv_quantize_q8(key, head_dim_, codes);

        const float* value = values_wh + static_cast<size_t>(token) * head_dim_;
        float value_scale_q8 = laplace_kv_quantize_q8(
            value, head_dim_, q8_values);
        v_scales_[token] = value_scale_q8 * (127.0f / 31.0f);
        for (int dim = 0; dim < head_dim_; dim++) {
            int q6 = static_cast<int>(std::nearbyint(
                q8_values[dim] * (31.0f / 127.0f)));
            uint8_t code = static_cast<uint8_t>(q6) & 63;
            uint8_t& high = v_codes_[
                static_cast<size_t>(dim) * (kTokens / 2) + token / 2];
            high |= static_cast<uint8_t>(
                (code >> 2) << ((token & 1) * 4));
            uint8_t& low = v_codes_[
                static_cast<size_t>(head_dim_) * (kTokens / 2)
                + static_cast<size_t>(dim) * (kTokens / 4) + token / 4];
            low |= static_cast<uint8_t>(
                (code & 3) << ((token & 3) * 2));
        }
    }
    sealed_ = true;
    return true;
}

void LaplaceKVTile::dot_keys(const int8_t* query_q8, float query_scale,
                               float* scores) const {
    if (!sealed_ || !k_scales_) return;
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    for (int token = 0; token < kTokens; token += 4) {
        const int8_t* key0 = reinterpret_cast<const int8_t*>(
            k_codes_ + static_cast<size_t>(token) * head_dim_);
        const int8_t* key1 = key0 + head_dim_;
        const int8_t* key2 = key1 + head_dim_;
        const int8_t* key3 = key2 + head_dim_;
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        for (int dim = 0; dim < head_dim_; dim += 16) {
            int8x16_t query = vld1q_s8(query_q8 + dim);
            acc0 = vdotq_s32(acc0, query, vld1q_s8(key0 + dim));
            acc1 = vdotq_s32(acc1, query, vld1q_s8(key1 + dim));
            acc2 = vdotq_s32(acc2, query, vld1q_s8(key2 + dim));
            acc3 = vdotq_s32(acc3, query, vld1q_s8(key3 + dim));
        }
        scores[token] = static_cast<float>(vaddvq_s32(acc0))
                      * query_scale * k_scales_[token];
        scores[token + 1] = static_cast<float>(vaddvq_s32(acc1))
                          * query_scale * k_scales_[token + 1];
        scores[token + 2] = static_cast<float>(vaddvq_s32(acc2))
                          * query_scale * k_scales_[token + 2];
        scores[token + 3] = static_cast<float>(vaddvq_s32(acc3))
                          * query_scale * k_scales_[token + 3];
    }
#else
    for (int token = 0; token < kTokens; token++) {
        const int8_t* key = reinterpret_cast<const int8_t*>(
            k_codes_ + static_cast<size_t>(token) * head_dim_);
        int sum = 0;
        for (int dim = 0; dim < head_dim_; dim++) {
            sum += query_q8[dim] * key[dim];
        }
        scores[token] = static_cast<float>(sum) * query_scale * k_scales_[token];
    }
#endif
}

void LaplaceKVTile::add_values(const float* weights, float* output_wh) const {
    if (!sealed_ || !v_scales_) return;
    alignas(16) uint8_t weights_q8[kTokens];
    alignas(16) float scaled_weights[kTokens];
    if (std::all_of(weights, weights + kTokens,
                    [](float weight) { return weight == 0.0f; })) {
        return;
    }
    for (int token = 0; token < kTokens; token++) {
        scaled_weights[token] = weights[token] * v_scales_[token];
    }
    float weight_scale = quantize_u8(
        scaled_weights, kTokens, weights_q8);
#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
    if (cpu_has_i8mm()) {
        alignas(16) static constexpr uint8_t low_indices_data[16] = {
            0, 0, 0, 0, 1, 1, 1, 1,
            2, 2, 2, 2, 3, 3, 3, 3
        };
        alignas(16) static constexpr int8_t low_shifts_data[16] = {
            0, -2, -4, -6, 0, -2, -4, -6,
            0, -2, -4, -6, 0, -2, -4, -6
        };
        const uint8x16_t nibble_mask = vdupq_n_u8(15);
        const uint8x16_t low_mask = vdupq_n_u8(3);
        const uint8x16_t low_indices = vld1q_u8(low_indices_data);
        const int8x16_t low_shifts = vld1q_s8(low_shifts_data);
        uint8x16_t weights_vec[4];
        for (int block = 0; block < 4; block++) {
            weights_vec[block] = vld1q_u8(weights_q8 + block * 16);
        }
        for (int dim = 0; dim < head_dim_; dim += 4) {
            int32x4_t high_acc[4] = {
                vdupq_n_s32(0), vdupq_n_s32(0),
                vdupq_n_s32(0), vdupq_n_s32(0)
            };
            int32x4_t low_acc[4] = {
                vdupq_n_s32(0), vdupq_n_s32(0),
                vdupq_n_s32(0), vdupq_n_s32(0)
            };
            for (int lane = 0; lane < 4; lane++) {
                const uint8_t* high = v_codes_
                    + static_cast<size_t>(dim + lane) * (kTokens / 2);
                for (int token = 0; token < kTokens; token += 32) {
                    uint8x16_t packed = vld1q_u8(high + token / 2);
                    uint8x16_t lo = vandq_u8(packed, nibble_mask);
                    uint8x16_t hi = vshrq_n_u8(packed, 4);
                    int8x16_t slo = vshrq_n_s8(
                        vreinterpretq_s8_u8(vshlq_n_u8(lo, 4)), 4);
                    int8x16_t shi = vshrq_n_s8(
                        vreinterpretq_s8_u8(vshlq_n_u8(hi, 4)), 4);
                    high_acc[lane] = vusdotq_s32(
                        high_acc[lane], weights_vec[token / 16],
                        vzip1q_s8(slo, shi));
                    high_acc[lane] = vusdotq_s32(
                        high_acc[lane], weights_vec[token / 16 + 1],
                        vzip2q_s8(slo, shi));
                }
                const uint8_t* low = v_codes_
                    + static_cast<size_t>(head_dim_) * (kTokens / 2)
                    + static_cast<size_t>(dim + lane) * (kTokens / 4);
                uint8x16_t packed = vld1q_u8(low);
                for (int block = 0; block < 4; block++) {
                    uint8x16_t indices = vaddq_u8(
                        low_indices, vdupq_n_u8(block * 4));
                    uint8x16_t repeated = vqtbl1q_u8(packed, indices);
                    uint8x16_t codes = vandq_u8(
                        vshlq_u8(repeated, low_shifts), low_mask);
                    low_acc[lane] = vusdotq_s32(
                        low_acc[lane], weights_vec[block],
                        vreinterpretq_s8_u8(codes));
                }
            }
            for (int lane = 0; lane < 4; lane++) {
                int sum = 4 * vaddvq_s32(high_acc[lane])
                        + vaddvq_s32(low_acc[lane]);
                output_wh[dim + lane] += static_cast<float>(sum)
                                       * weight_scale;
            }
        }
        return;
    }
#endif
    for (int dim = 0; dim < head_dim_; dim++) {
        const uint8_t* high = v_codes_
            + static_cast<size_t>(dim) * (kTokens / 2);
        const uint8_t* low = v_codes_
            + static_cast<size_t>(head_dim_) * (kTokens / 2)
            + static_cast<size_t>(dim) * (kTokens / 4);
        int sum = 0;
        for (int token = 0; token < kTokens; token++) {
            int code = (high[token / 2] >> ((token & 1) * 4)) & 15;
            if (code >= 8) code -= 16;
            code = 4 * code
                 + ((low[token / 4] >> ((token & 3) * 2)) & 3);
            sum += weights_q8[token] * code;
        }
        output_wh[dim] += static_cast<float>(sum) * weight_scale;
    }
}

void LaplaceKVTile::load_key_wh(int token, float* output_wh) const {
    if (!sealed_ || !k_scales_ || token < 0 || token >= kTokens) return;
    const int8_t* codes = reinterpret_cast<const int8_t*>(
        k_codes_ + static_cast<size_t>(token) * head_dim_);
    for (int dim = 0; dim < head_dim_; dim++) {
        output_wh[dim] = codes[dim] * k_scales_[token];
    }
}

void LaplaceKVTile::load_value_wh(int token, float* output_wh) const {
    if (!sealed_ || !v_scales_ || token < 0 || token >= kTokens) return;
    for (int dim = 0; dim < head_dim_; dim++) {
        uint8_t packed_high = v_codes_[
            static_cast<size_t>(dim) * (kTokens / 2) + token / 2];
        int code = (packed_high >> ((token & 1) * 4)) & 15;
        if (code >= 8) code -= 16;
        uint8_t packed_low = v_codes_[
            static_cast<size_t>(head_dim_) * (kTokens / 2)
            + static_cast<size_t>(dim) * (kTokens / 4) + token / 4];
        code = 4 * code
             + ((packed_low >> ((token & 3) * 2)) & 3);
        output_wh[dim] = code * v_scales_[token];
    }
}

size_t LaplaceKVTile::storage_bytes() const {
    return storage_.capacity() * sizeof(uint32_t);
}

bool LaplaceKV::init(int n_layers, int n_kv_heads, int head_dim, int capacity,
                        bool streaming) {
    clear();
    if (n_layers <= 0 || n_kv_heads <= 0 || head_dim < 32 || head_dim > 512
        || head_dim % 16 != 0 || capacity <= 0) {
        return false;
    }

    n_layers_ = n_layers;
    n_kv_heads_ = n_kv_heads;
    head_dim_ = head_dim;
    capacity_ = capacity;
    streaming_ = streaming;
    chunks_per_head_ = (capacity + LaplaceKVTile::kTokens - 1)
                     / LaplaceKVTile::kTokens;
    size_t heads = static_cast<size_t>(n_layers_) * n_kv_heads_;
    chunk_words_ = LaplaceKVTile::storage_words(head_dim_);
    if (streaming_) {
        const char* temp = std::getenv("TMPDIR");
        std::string pattern = temp && *temp ? temp : "/tmp";
        if (pattern.back() != '/') pattern.push_back('/');
        pattern += "laplace-kv-XXXXXX";
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
        archive_bytes_ = heads * static_cast<size_t>(chunks_per_head_)
                       * chunk_words_ * sizeof(uint32_t);
        if (ftruncate(archive_fd_, static_cast<off_t>(archive_bytes_)) != 0) {
            clear();
            return false;
        }
    } else {
        resident_storage_.resize(heads);
    }
    sealed_chunks_.assign(heads * chunks_per_head_, 0);
    size_t tail_values = heads * LaplaceKVTile::kTokens * head_dim_;
    k_tail_.assign(tail_values, 0.0f);
    v_tail_.assign(tail_values, 0.0f);
    return true;
}

void LaplaceKV::clear() {
    if (archive_fd_ >= 0) close(archive_fd_);
    archive_fd_ = -1;
    archive_bytes_ = 0;
    resident_storage_.clear();
    sealed_chunks_.clear();
    k_tail_.clear();
    v_tail_.clear();
    stream_calls_.store(0);
    archive_read_bytes_.store(0);
    archive_write_bytes_.store(0);
    n_layers_ = 0;
    n_kv_heads_ = 0;
    head_dim_ = 0;
    capacity_ = 0;
    chunks_per_head_ = 0;
    chunk_words_ = 0;
    streaming_ = false;
}

size_t LaplaceKV::head_index(int layer, int head) const {
    return static_cast<size_t>(layer) * n_kv_heads_ + head;
}

size_t LaplaceKV::chunk_index(int layer, int head, int chunk) const {
    return head_index(layer, head) * chunks_per_head_ + chunk;
}

float* LaplaceKV::tail_k(int layer, int head, int offset) {
    size_t index = (head_index(layer, head) * LaplaceKVTile::kTokens + offset)
                 * head_dim_;
    return k_tail_.data() + index;
}

float* LaplaceKV::tail_v(int layer, int head, int offset) {
    size_t index = (head_index(layer, head) * LaplaceKVTile::kTokens + offset)
                 * head_dim_;
    return v_tail_.data() + index;
}

const float* LaplaceKV::tail_k(int layer, int head, int offset) const {
    size_t index = (head_index(layer, head) * LaplaceKVTile::kTokens + offset)
                 * head_dim_;
    return k_tail_.data() + index;
}

const float* LaplaceKV::tail_v(int layer, int head, int offset) const {
    size_t index = (head_index(layer, head) * LaplaceKVTile::kTokens + offset)
                 * head_dim_;
    return v_tail_.data() + index;
}

size_t LaplaceKV::archive_offset(size_t head, int chunk) const {
    return (head * static_cast<size_t>(chunks_per_head_) + chunk)
         * chunk_words_ * sizeof(uint32_t);
}

bool LaplaceKV::read_archive(size_t head, int chunk, uint32_t* output) const {
    return read_archive(head, chunk, 1, output);
}

bool LaplaceKV::read_archive(size_t head, int chunk, int count,
                                uint32_t* output) const {
    size_t bytes = static_cast<size_t>(count) * chunk_words_ * sizeof(uint32_t);
    size_t done = 0;
    while (done < bytes) {
        ssize_t result = pread(archive_fd_,
                               reinterpret_cast<uint8_t*>(output) + done,
                               bytes - done,
                               static_cast<off_t>(archive_offset(head, chunk) + done));
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

bool LaplaceKV::write_archive(size_t head, int chunk, const uint32_t* input) {
    size_t bytes = chunk_words_ * sizeof(uint32_t);
    size_t done = 0;
    while (done < bytes) {
        ssize_t result = pwrite(
            archive_fd_, reinterpret_cast<const uint8_t*>(input) + done,
            bytes - done,
            static_cast<off_t>(archive_offset(head, chunk) + done));
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

void LaplaceKV::store_k_wh(int layer, int head, int pos, const float* key_wh) {
    if (pos < 0 || pos >= capacity_) return;
#if defined(LAPLACE_KV_CAPTURE)
    kv_trace_sink().vector(1, layer, head, pos, head_dim_, key_wh);
#endif
    int offset = pos % LaplaceKVTile::kTokens;
    std::memcpy(tail_k(layer, head, offset), key_wh, sizeof(float) * head_dim_);
}

void LaplaceKV::store_v_wh(int layer, int head, int pos, const float* value_wh) {
    if (pos < 0 || pos >= capacity_) return;
#if defined(LAPLACE_KV_CAPTURE)
    kv_trace_sink().vector(2, layer, head, pos, head_dim_, value_wh);
#endif
    int offset = pos % LaplaceKVTile::kTokens;
    std::memcpy(tail_v(layer, head, offset), value_wh, sizeof(float) * head_dim_);
    if (offset != LaplaceKVTile::kTokens - 1) return;

    int chunk = pos / LaplaceKVTile::kTokens;
    size_t head_id = head_index(layer, head);
    if (streaming_) {
        std::vector<uint32_t> storage(chunk_words_);
        LaplaceKVTile view;
        if (!view.init(head_dim_, storage.data(), false)
            || !view.seal(tail_k(layer, head, 0), tail_v(layer, head, 0))) {
            return;
        }
        if (!write_archive(head_id, chunk, storage.data())) std::abort();
        sealed_chunks_[chunk_index(layer, head, chunk)] = 1;
        return;
    }
    auto& storage = resident_storage_[head_id];
    storage.resize(static_cast<size_t>(chunk + 1) * chunk_words_);
    LaplaceKVTile view;
    if (!view.init(head_dim_,
                   storage.data() + static_cast<size_t>(chunk) * chunk_words_,
                   false)
        || !view.seal(tail_k(layer, head, 0), tail_v(layer, head, 0))) {
        return;
    }
    sealed_chunks_[chunk_index(layer, head, chunk)] = 1;
}

void LaplaceKV::load_k_wh(int layer, int head, int pos, float* key_wh) const {
    if (pos < 0 || pos >= capacity_) return;
    int chunk = pos / LaplaceKVTile::kTokens;
    int offset = pos % LaplaceKVTile::kTokens;
    size_t index = chunk_index(layer, head, chunk);
    if (sealed_chunks_[index]) {
        if (streaming_) {
            std::vector<uint32_t> storage(chunk_words_);
            if (!read_archive(head_index(layer, head), chunk, storage.data())) {
                std::abort();
            }
            LaplaceKVTile view;
            view.init(head_dim_, storage.data(), true);
            view.load_key_wh(offset, key_wh);
            return;
        }
        const auto& storage = resident_storage_[head_index(layer, head)];
        LaplaceKVTile view;
        view.init(head_dim_, const_cast<uint32_t*>(storage.data())
                           + static_cast<size_t>(chunk) * chunk_words_, true);
        view.load_key_wh(offset, key_wh);
    } else {
        std::memcpy(key_wh, tail_k(layer, head, offset), sizeof(float) * head_dim_);
    }
}

void LaplaceKV::load_v_wh(int layer, int head, int pos, float* value_wh) const {
    if (pos < 0 || pos >= capacity_) return;
    int chunk = pos / LaplaceKVTile::kTokens;
    int offset = pos % LaplaceKVTile::kTokens;
    size_t index = chunk_index(layer, head, chunk);
    if (sealed_chunks_[index]) {
        if (streaming_) {
            std::vector<uint32_t> storage(chunk_words_);
            if (!read_archive(head_index(layer, head), chunk, storage.data())) {
                std::abort();
            }
            LaplaceKVTile view;
            view.init(head_dim_, storage.data(), true);
            view.load_value_wh(offset, value_wh);
            return;
        }
        const auto& storage = resident_storage_[head_index(layer, head)];
        LaplaceKVTile view;
        view.init(head_dim_, const_cast<uint32_t*>(storage.data())
                           + static_cast<size_t>(chunk) * chunk_words_, true);
        view.load_value_wh(offset, value_wh);
    } else {
        std::memcpy(value_wh, tail_v(layer, head, offset), sizeof(float) * head_dim_);
    }
}

void LaplaceKV::dot_keys_wh(int layer, int head, int n_tokens,
                               const float* query_wh, float* scores,
                               int first_token) const {
    n_tokens = std::clamp(n_tokens, 0, capacity_);
    first_token = std::clamp(first_token, 0, n_tokens);
    if (streaming_) {
        alignas(16) float key[512];
        for (int token = first_token; token < n_tokens; token++) {
            load_k_wh(layer, head, token, key);
            scores[token] = ops::dot(query_wh, key, head_dim_);
        }
        return;
    }
    alignas(16) int8_t query_q8[512];
    float query_scale = laplace_kv_quantize_q8(
        query_wh, head_dim_, query_q8);
    alignas(16) float chunk_scores[LaplaceKVTile::kTokens];

    int token = first_token;
    while (token < n_tokens) {
        int chunk = token / LaplaceKVTile::kTokens;
        int offset = token % LaplaceKVTile::kTokens;
        int count = std::min(LaplaceKVTile::kTokens - offset, n_tokens - token);
        size_t index = chunk_index(layer, head, chunk);
        bool sealed = sealed_chunks_[index] != 0;
        LaplaceKVTile view;
        if (sealed) {
            const auto& storage = resident_storage_[head_index(layer, head)];
            view.init(head_dim_, const_cast<uint32_t*>(storage.data())
                               + static_cast<size_t>(chunk) * chunk_words_, true);
            float* destination = offset == 0 && count == LaplaceKVTile::kTokens
                               ? scores + token : chunk_scores;
            view.dot_keys(query_q8, query_scale, destination);
            if (destination == chunk_scores) {
                std::copy_n(chunk_scores + offset, count, scores + token);
            }
        } else {
            for (int index = 0; index < count; index++) {
                scores[token + index] = ops::dot(
                    query_wh, tail_k(layer, head, offset + index), head_dim_);
            }
        }
        token += count;
    }
}

void LaplaceKV::add_values_wh(int layer, int head, int n_tokens,
                                 const float* weights, float* output_wh,
                                 int first_token) const {
    n_tokens = std::clamp(n_tokens, 0, capacity_);
    first_token = std::clamp(first_token, 0, n_tokens);
    if (streaming_) {
        alignas(16) float value[512];
        for (int token = first_token; token < n_tokens; token++) {
            load_v_wh(layer, head, token, value);
            ops::axpy(output_wh, weights[token], value, head_dim_);
        }
        return;
    }
    alignas(16) float chunk_weights[LaplaceKVTile::kTokens];

    int token = first_token;
    while (token < n_tokens) {
        int chunk = token / LaplaceKVTile::kTokens;
        int offset = token % LaplaceKVTile::kTokens;
        int count = std::min(LaplaceKVTile::kTokens - offset, n_tokens - token);
        size_t index = chunk_index(layer, head, chunk);
        bool sealed = sealed_chunks_[index] != 0;
        LaplaceKVTile view;
        if (sealed) {
            const auto& storage = resident_storage_[head_index(layer, head)];
            view.init(head_dim_, const_cast<uint32_t*>(storage.data())
                               + static_cast<size_t>(chunk) * chunk_words_, true);
            const float* source = weights + token;
            if (offset != 0 || count != LaplaceKVTile::kTokens) {
                std::fill_n(chunk_weights, LaplaceKVTile::kTokens, 0.0f);
                std::copy_n(source, count, chunk_weights + offset);
                source = chunk_weights;
            }
            view.add_values(source, output_wh);
        } else {
            for (int index = 0; index < count; index++) {
                ops::axpy(output_wh, weights[token + index],
                          tail_v(layer, head, offset + index), head_dim_);
            }
        }
        token += count;
    }
}


void LaplaceKV::attention_batch_wh(
        int layer, int head, int count, const int* n_tokens,
        const float* const* queries_wh, float logit_scale,
        float* const* outputs_wh, int first_token) const {
    if (!streaming_) {
        for (int query = 0; query < count; query++) {
            attention_wh(layer, head, n_tokens[query], queries_wh[query],
                         logit_scale, outputs_wh[query], first_token);
        }
        return;
    }

    int max_tokens = 0;
    std::vector<int> ends(count);
    for (int query = 0; query < count; query++) {
        ends[query] = std::clamp(n_tokens[query], 0, capacity_);
        max_tokens = std::max(max_tokens, ends[query]);
        std::fill_n(outputs_wh[query], head_dim_, 0.0f);
    }
    first_token = std::clamp(first_token, 0, max_tokens);
    if (first_token == max_tokens) return;

    std::vector<int8_t> queries_q8(static_cast<size_t>(count) * head_dim_);
    std::vector<float> query_scales(count);
    for (int query = 0; query < count; query++) {
        query_scales[query] = laplace_kv_quantize_q8(
            queries_wh[query], head_dim_,
            queries_q8.data() + static_cast<size_t>(query) * head_dim_);
    }

    std::vector<float> global_max(count, 0.0f);
    std::vector<float> global_sum(count, 0.0f);
    std::vector<uint8_t> have_global(count, 0);
    alignas(16) float scores[LaplaceKVTile::kTokens];
    alignas(16) float weights[LaplaceKVTile::kTokens];
    alignas(16) float local_output[512];

    auto merge = [&](int query, float local_max, float local_sum) {
        float* output = outputs_wh[query];
        if (!have_global[query]) {
            std::copy_n(local_output, head_dim_, output);
            global_max[query] = local_max;
            global_sum[query] = local_sum;
            have_global[query] = 1;
        } else if (local_max <= global_max[query]) {
            float scale = std::exp(local_max - global_max[query]);
            for (int dim = 0; dim < head_dim_; dim++) {
                output[dim] += local_output[dim] * scale;
            }
            global_sum[query] += local_sum * scale;
        } else {
            float scale = std::exp(global_max[query] - local_max);
            for (int dim = 0; dim < head_dim_; dim++) {
                output[dim] = output[dim] * scale + local_output[dim];
            }
            global_sum[query] = global_sum[query] * scale + local_sum;
            global_max[query] = local_max;
        }
    };

    auto process_sealed = [&](int chunk, const uint32_t* storage) {
        LaplaceKVTile view;
        view.init(head_dim_, const_cast<uint32_t*>(storage), true);
        const int chunk_start = chunk * LaplaceKVTile::kTokens;
        for (int query = 0; query < count; query++) {
            int begin = std::max(0, first_token - chunk_start);
            int end = std::min(LaplaceKVTile::kTokens,
                               ends[query] - chunk_start);
            if (begin >= end) continue;
            view.dot_keys(
                queries_q8.data() + static_cast<size_t>(query) * head_dim_,
                query_scales[query], scores);
            float local_max = scores[begin] * logit_scale;
            for (int token = begin + 1; token < end; token++) {
                local_max = std::max(local_max,
                                     scores[token] * logit_scale);
            }
            std::fill_n(weights, LaplaceKVTile::kTokens, 0.0f);
            float local_sum = 0.0f;
            for (int token = begin; token < end; token++) {
                float weight = std::exp(
                    scores[token] * logit_scale - local_max);
                weights[token] = weight;
                local_sum += weight;
            }
            std::fill_n(local_output, head_dim_, 0.0f);
            view.add_values(weights, local_output);
            merge(query, local_max, local_sum);
        }
    };

    auto process_tail = [&](int chunk) {
        const int chunk_start = chunk * LaplaceKVTile::kTokens;
        for (int query = 0; query < count; query++) {
            int begin = std::max(0, first_token - chunk_start);
            int end = std::min(LaplaceKVTile::kTokens,
                               ends[query] - chunk_start);
            if (begin >= end) continue;
            float local_max = ops::dot(
                queries_wh[query], tail_k(layer, head, begin), head_dim_)
                * logit_scale;
            for (int token = begin; token < end; token++) {
                scores[token] = ops::dot(
                    queries_wh[query], tail_k(layer, head, token), head_dim_)
                    * logit_scale;
                local_max = std::max(local_max, scores[token]);
            }
            float local_sum = 0.0f;
            std::fill_n(local_output, head_dim_, 0.0f);
            for (int token = begin; token < end; token++) {
                float weight = std::exp(scores[token] - local_max);
                local_sum += weight;
                ops::axpy(local_output, weight,
                          tail_v(layer, head, token), head_dim_);
            }
            merge(query, local_max, local_sum);
        }
    };

    const size_t head_id = head_index(layer, head);
    const int first_chunk = first_token / LaplaceKVTile::kTokens;
    const int last_chunk = (max_tokens - 1) / LaplaceKVTile::kTokens;
    const int read_chunks = std::clamp(
        static_cast<int>(kArchiveReadBytes
                         / (chunk_words_ * sizeof(uint32_t))),
        1, kMaxArchiveReadChunks);
    std::vector<uint32_t> archive(
        static_cast<size_t>(read_chunks) * chunk_words_);
    int chunk = first_chunk;
    while (chunk <= last_chunk) {
        if (!sealed_chunks_[chunk_index(layer, head, chunk)]) {
            process_tail(chunk++);
            continue;
        }
        int end = chunk;
        while (end <= last_chunk && end < chunk + read_chunks
               && sealed_chunks_[chunk_index(layer, head, end)]) {
            end++;
        }
        if (!read_archive(head_id, chunk, end - chunk, archive.data())) {
            std::abort();
        }
        for (int sealed = chunk; sealed < end; sealed++) {
            process_sealed(
                sealed, archive.data()
                    + static_cast<size_t>(sealed - chunk) * chunk_words_);
        }
        chunk = end;
    }

    for (int query = 0; query < count; query++) {
        if (!have_global[query] || global_sum[query] <= 0.0f) continue;
        float inverse = 1.0f / global_sum[query];
        for (int dim = 0; dim < head_dim_; dim++) {
            outputs_wh[query][dim] *= inverse;
        }
#if defined(LAPLACE_KV_CAPTURE)
        kv_trace_sink().query(layer, head, ends[query], head_dim_, first_token,
                              logit_scale, query, queries_wh[query],
                              outputs_wh[query]);
#endif
    }
    stream_calls_.fetch_add(count, std::memory_order_relaxed);
}

void LaplaceKV::attention_wh(int layer, int head, int n_tokens,
                                const float* query_wh, float logit_scale,
                                float* output_wh, int first_token) const {
    if (streaming_) {
        const float* queries[] = {query_wh};
        float* outputs[] = {output_wh};
        attention_batch_wh(layer, head, 1, &n_tokens, queries,
                           logit_scale, outputs, first_token);
        return;
    }
    n_tokens = std::clamp(n_tokens, 0, capacity_);
    first_token = std::clamp(first_token, 0, n_tokens);
    std::fill_n(output_wh, head_dim_, 0.0f);
    if (first_token == n_tokens) return;

    alignas(16) int8_t query_q8[512];
    float query_scale = laplace_kv_quantize_q8(query_wh, head_dim_, query_q8);
    alignas(16) float scores[LaplaceKVTile::kTokens];
    alignas(16) float weights[LaplaceKVTile::kTokens];
    alignas(16) float local_output[512];
    float global_max = 0.0f;
    float global_sum = 0.0f;
    bool have_global = false;

    int token = first_token;
    while (token < n_tokens) {
        int chunk = token / LaplaceKVTile::kTokens;
        int offset = token % LaplaceKVTile::kTokens;
        int count = std::min(LaplaceKVTile::kTokens - offset, n_tokens - token);
        size_t index = chunk_index(layer, head, chunk);
        bool sealed = sealed_chunks_[index] != 0;
        LaplaceKVTile view;
        if (sealed) {
            const auto& storage = resident_storage_[head_index(layer, head)];
            view.init(head_dim_, const_cast<uint32_t*>(storage.data())
                               + static_cast<size_t>(chunk) * chunk_words_, true);
            view.dot_keys(query_q8, query_scale, scores);
        } else {
            for (int index = 0; index < count; index++) {
                scores[offset + index] = ops::dot(
                    query_wh, tail_k(layer, head, offset + index), head_dim_);
            }
        }

        float local_max = scores[offset] * logit_scale;
        for (int index = 1; index < count; index++) {
            local_max = std::max(
                local_max, scores[offset + index] * logit_scale);
        }
        std::fill_n(weights, LaplaceKVTile::kTokens, 0.0f);
        for (int index = 0; index < count; index++) {
            weights[offset + index] =
                scores[offset + index] * logit_scale - local_max;
        }
#if defined(__APPLE__)
        vvexpf(weights + offset, weights + offset, &count);
#else
        for (int index = 0; index < count; index++) {
            weights[offset + index] = std::exp(weights[offset + index]);
        }
#endif
        float local_sum = 0.0f;
        for (int index = 0; index < count; index++) {
            local_sum += weights[offset + index];
        }

        std::fill_n(local_output, head_dim_, 0.0f);
        if (sealed) {
            view.add_values(weights, local_output);
        } else {
            for (int index = 0; index < count; index++) {
                ops::axpy(local_output, weights[offset + index],
                          tail_v(layer, head, offset + index), head_dim_);
            }
        }

        if (!have_global) {
            std::copy_n(local_output, head_dim_, output_wh);
            global_max = local_max;
            global_sum = local_sum;
            have_global = true;
        } else if (local_max <= global_max) {
            float local_scale = std::exp(local_max - global_max);
            for (int dim = 0; dim < head_dim_; dim++) {
                output_wh[dim] += local_output[dim] * local_scale;
            }
            global_sum += local_sum * local_scale;
        } else {
            float global_scale = std::exp(global_max - local_max);
            for (int dim = 0; dim < head_dim_; dim++) {
                output_wh[dim] = output_wh[dim] * global_scale
                               + local_output[dim];
            }
            global_sum = global_sum * global_scale + local_sum;
            global_max = local_max;
        }
        token += count;
    }

    float inverse_sum = 1.0f / global_sum;
    for (int dim = 0; dim < head_dim_; dim++) output_wh[dim] *= inverse_sum;
#if defined(LAPLACE_KV_CAPTURE)
    kv_trace_sink().query(layer, head, n_tokens, head_dim_, first_token,
                          logit_scale, 0, query_wh, output_wh);
#endif
}

size_t LaplaceKV::storage_bytes() const {
    size_t bytes = (k_tail_.capacity() + v_tail_.capacity()) * sizeof(float)
                 + sealed_chunks_.capacity() * sizeof(sealed_chunks_[0])
                 + resident_storage_.capacity()
                     * sizeof(resident_storage_[0]);
    for (const auto& storage : resident_storage_) {
        bytes += storage.capacity() * sizeof(uint32_t);
    }
    return bytes;
}

size_t LaplaceKV::archive_read_buffer_bytes() const {
    if (!streaming_ || chunk_words_ == 0) return 0;
    size_t chunk_bytes = chunk_words_ * sizeof(uint32_t);
    size_t chunks = std::clamp(
        kArchiveReadBytes / chunk_bytes,
        static_cast<size_t>(1),
        static_cast<size_t>(kMaxArchiveReadChunks));
    return chunks * chunk_bytes;
}

} // namespace Laplace
