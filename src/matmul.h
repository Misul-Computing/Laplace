// matmul.h - GEMM dispatch over quantization formats
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "tensor.h"

namespace Laplace {

enum class DerivedStorageError : uint16_t {
    None = 0,
    SourceFormatUnsupported = 1,
    ShapeUnsupported = 2,
    SourceLengthMismatch = 3,
    NonFiniteSource = 4,
    IntegrityMismatch = 5,
    ImportanceLengthMismatch = 6,
    DestinationLengthMismatch = 7,
    InvalidImportance = 8,
    ContractMismatch = 9,
    DestinationAlignmentMismatch = 10,
};

// Derived affine UInt2 storage. Each logical 256-value block has one 64-byte
// LSB-first UInt2 payload, one FP16 scale, and one FP16 bias in separate
// row-major planes. Values decode as scale * q + bias.
struct DerivedAffineUInt2Contract {
    uint16_t version = 1;
    uint64_t logical_k = 0;
    uint64_t logical_n = 0;
    uint32_t block_width = 256;
    uint32_t packed_bytes_per_block = 64;
    uint32_t scale_bytes_per_block = 2;
    uint32_t bias_bytes_per_block = 2;
    uint32_t plane_alignment = 128;
    friend bool operator==(const DerivedAffineUInt2Contract&,
                           const DerivedAffineUInt2Contract&) = default;
};

struct DerivedAffineUInt2Storage {
    DerivedAffineUInt2Contract contract;
    std::vector<uint8_t> packed_weights;
    std::vector<uint16_t> scales;
    std::vector<uint16_t> biases;
};

bool validate_affine_u2_block256(const DerivedAffineUInt2Storage& storage,
                                 DerivedStorageError* error);
bool decode_affine_u2_block256(const DerivedAffineUInt2Storage& storage,
                               std::vector<float>& output,
                               DerivedStorageError* error);

struct DerivedAffineUInt2Record {
    DerivedAffineUInt2Contract contract;
    GGMLType source_format = GGMLType::F32;
    std::array<uint8_t, 32> source_digest{};
    std::array<uint8_t, 32> importance_digest{};
    std::array<uint8_t, 32> storage_digest{};
    friend bool operator==(const DerivedAffineUInt2Record&,
                           const DerivedAffineUInt2Record&) = default;
};

bool derive_affine_u2_block256_from_gguf(
    GGMLType source_format, std::span<const uint8_t> source,
    uint64_t logical_k, uint64_t logical_n, std::span<const float> importance,
    std::span<uint8_t> packed_weights, std::span<uint16_t> scales,
    std::span<uint16_t> biases, DerivedAffineUInt2Record* record,
    DerivedStorageError* error);

struct DerivedQ2KContract {
    uint16_t version = 1;
    GGMLType source_format = GGMLType::F32;
    uint64_t logical_k = 0;
    uint64_t logical_n = 0;
    uint32_t block_width = 256;
    uint32_t bytes_per_block = 84;
    uint32_t alignment = 128;
    friend bool operator==(const DerivedQ2KContract&, const DerivedQ2KContract&) = default;
};

struct DerivedQ2KStorage {
    DerivedQ2KContract contract;
    std::array<uint8_t, 32> source_digest{};
    std::array<uint8_t, 32> storage_digest{};
    std::vector<uint8_t> bytes;
};

bool derive_q2_k_from_gguf(GGMLType source_format, std::span<const uint8_t> source,
                           uint64_t logical_k, uint64_t logical_n,
                           DerivedQ2KStorage* output, DerivedStorageError* error);
bool decode_derived_q2_k(const DerivedQ2KStorage& storage, std::vector<float>& output);

struct DerivedIQ2XXSContract {
    uint16_t version = 1;
    GGMLType source_format = GGMLType::F32;
    uint64_t logical_k = 0;
    uint64_t logical_n = 0;
    uint32_t block_width = 256;
    uint32_t bytes_per_block = 66;
    uint32_t alignment = 128;
    friend bool operator==(const DerivedIQ2XXSContract&, const DerivedIQ2XXSContract&) = default;
};

struct DerivedIQ2XXSRecord {
    DerivedIQ2XXSContract contract;
    std::array<uint8_t, 32> source_digest{};
    std::array<uint8_t, 32> importance_digest{};
    std::array<uint8_t, 32> storage_digest{};
    friend bool operator==(const DerivedIQ2XXSRecord&, const DerivedIQ2XXSRecord&) = default;
};

bool derive_iq2_xxs_from_gguf(GGMLType source_format, std::span<const uint8_t> source,
                              uint64_t logical_k, uint64_t logical_n,
                              std::span<const float> importance,
                              std::span<uint8_t> destination,
                              DerivedIQ2XXSRecord* record, DerivedStorageError* error);

struct DerivedIQ1SContract {
    uint16_t version = 1;
    GGMLType source_format = GGMLType::F32;
    uint64_t logical_k = 0;
    uint64_t logical_n = 0;
    uint32_t block_width = 256;
    uint32_t bytes_per_block = 50;
    uint32_t alignment = 128;
    friend bool operator==(const DerivedIQ1SContract&, const DerivedIQ1SContract&) = default;
};

struct DerivedIQ1SRecord {
    DerivedIQ1SContract contract;
    std::array<uint8_t, 32> source_digest{};
    std::array<uint8_t, 32> importance_digest{};
    std::array<uint8_t, 32> storage_digest{};
    friend bool operator==(const DerivedIQ1SRecord&, const DerivedIQ1SRecord&) = default;
};

bool derive_iq1_s_from_gguf(GGMLType source_format, std::span<const uint8_t> source,
                            uint64_t logical_k, uint64_t logical_n,
                            std::span<const float> importance,
                            std::span<uint8_t> destination,
                            DerivedIQ1SRecord* record, DerivedStorageError* error);
#if defined(LAPLACE_TESTING)
uint8_t iq2_xxs_upstream_neighbor_for_testing(uint16_t pattern, const float* values,
                                              const float* optimization_weights, float scale,
                                              uint8_t* selected_levels);
uint16_t iq1_s_upstream_neighbor_for_testing(uint16_t pattern, const float* values,
                                             const float* weights, float scale,
                                             const float* quant_values,
                                             int8_t* selected_levels);
bool iq1_s_candidate_order_is_upstream_for_testing();
#endif

// y[M,N] += x[M,K] @ w[K,N]  (row-major, FP32 activations and accumulators)
// w must already be cast to the right layout in the GGUF file; this routine
// just dispatches to the kernel for the weight's quantization type.
//
// We support the [1, K] x [K, N] -> [1, N] "decode" case fast (one token
// against a single row of activations), and the general [M, K] x [K, N] case
// for prefill.
void matmul_row(const float* x, const Tensor& w, float* y, int K, int N);
void matmul_rows(const float* x, const Tensor& w, float* y, int M, int K, int N);

// LM head: tries GPU first (GPU is 2.5x faster for the large vocab projection),
// falls back to CPU. Used only for the final logits projection.
void matmul_lm_head(const float* x, const Tensor& w, float* y, int M, int K, int N);

// Register mmap'd weight region for zero-copy GPU access.
void matmul_register_weights(const void* base, size_t size);
void matmul_gpu_pack(const Tensor& w, int K, int N);

struct MatmulBatchSpec {
    const float* x;
    const Tensor* w;
    float* y;
    int K, N;
};

// Submit M=1 GEMVs to Metal and return immediately. matmul_gpu_end() waits
// and writes the y buffers. Returns false if GPU is off or the submit failed.
bool matmul_gpu_begin(const MatmulBatchSpec* specs, int n);
void matmul_gpu_end();
bool matmul_gpu_available();
bool matmul_gpu_batch(const MatmulBatchSpec* specs, int n);

// One GPU command buffer for a decode token. Residual stays on device.
struct MetalTokLayer {
    const Tensor* attn_norm = nullptr;
    const Tensor* attn_q = nullptr;
    const Tensor* attn_q_bias = nullptr;
    const Tensor* attn_k = nullptr;
    const Tensor* attn_k_bias = nullptr;
    const Tensor* attn_v = nullptr;
    const Tensor* attn_v_bias = nullptr;
    const Tensor* attn_o = nullptr;
    const Tensor* q_norm = nullptr;
    const Tensor* k_norm = nullptr;
    const Tensor* post_attn_norm = nullptr;
    const Tensor* ffn_norm = nullptr;
    const Tensor* ffn_gate = nullptr;
    const Tensor* ffn_up = nullptr;
    const Tensor* ffn_down = nullptr;
    const Tensor* pre_ffw_2 = nullptr;
    const Tensor* post_ffw_1 = nullptr;
    const Tensor* post_ffw_2 = nullptr;
    const Tensor* post_ffw = nullptr;
    const Tensor* out_scale = nullptr;
    const float* rope_freqs = nullptr;
    int n_rope_freqs = 0;
    uint32_t rope_sections[4]{};
    int H = 0, inter = 0;
    int Hq = 0, Hk = 0, Dh = 0, rope_dim = 0, window = 0, cache_id = 0;
    float rope_base = 0, rms_eps = 0, q_norm_eps = 0, k_norm_eps = 0;
    bool swiglu = false, owns_kv = true, is_global = false, query_gate_split = false,
         sparse_ffn = false,
         sparse_ffn_dense_oracle = false,
         rope_interleaved = false, rope_multi_section = false;
    int sparse_ffn_full_intermediate = 0;
    uint32_t sparse_ffn_block_offset = 0;
    uint32_t sparse_ffn_block_count = 0;
    uint32_t sparse_ffn_proxy_slot = UINT32_MAX;
    uint32_t ffn_input_importance_slot = UINT32_MAX;
    uint32_t ffn_down_importance_slot = UINT32_MAX;
};

// One exact recurrent semantic layer. Canonical lowering fills this from
// tensor roles and state references; it carries no importer or family text.
struct MetalTokRecurrentLayer {
    const Tensor* input_norm = nullptr;
    const Tensor* qkv = nullptr;
    const Tensor* gate = nullptr;
    const Tensor* beta = nullptr;
    const Tensor* alpha = nullptr;
    const Tensor* conv = nullptr;
    const Tensor* dt_bias = nullptr;
    const Tensor* decay = nullptr;
    const Tensor* norm = nullptr;
    const Tensor* output = nullptr;
    const Tensor* ffn_norm = nullptr;
    const Tensor* ffn_gate = nullptr;
    const Tensor* ffn_up = nullptr;
    const Tensor* ffn_down = nullptr;
    uint32_t state_slot = 0;
    int H = 0;
    int qk_heads = 0;
    int value_heads = 0;
    int head_dimension = 0;
    int kernel = 0;
    int ffn_intermediate = 0;
    int sparse_ffn_full_intermediate = 0;
    bool sparse_ffn = false;
    bool sparse_ffn_dense_oracle = false;
    uint32_t sparse_ffn_block_offset = 0;
    uint32_t sparse_ffn_block_count = 0;
    uint32_t sparse_ffn_proxy_slot = UINT32_MAX;
    uint32_t ffn_input_importance_slot = UINT32_MAX;
    uint32_t ffn_down_importance_slot = UINT32_MAX;
    float l2_epsilon = 0.0f;
    float rms_epsilon = 0.0f;
};

// Private token resources for one canonical runtime/session. The legacy
// token-graph wrappers below retain their process-default context.
class MetalTokSession;
struct MetalSparseBlockRun {
    uint32_t first = 0;
    uint32_t count = 0;
};
struct MetalTokMetrics {
    double cpu_wait_ms = 0.0;
    double gpu_time_ms = 0.0;
    uint64_t peak_session_bytes = 0;
    uint64_t projection_weight_bytes = 0;
    uint32_t projection_dispatches = 0;
    // A projection dispatched with more than one activation row. This is
    // intentionally separate from the legacy total so a one-command-buffer
    // serial prefill cannot masquerade as a batch implementation.
    uint32_t batched_projection_dispatches = 0;
    uint32_t q4k_projection_dispatches = 0;
    uint32_t q6k_projection_dispatches = 0;
    uint64_t counter_sample_count = 0;
    bool profiled = false;
    bool counter_samples = false;
    bool split_command_buffer_profile = false;
    double qkv_gpu_ms = 0.0;
    double attention_gpu_ms = 0.0;
    double ffn_gpu_ms = 0.0;
    double final_gpu_ms = 0.0;
};
struct MetalDispatchMetrics {
    uint64_t command_buffers = 0;
    double cpu_wait_ms = 0.0;
    double gpu_time_ms = 0.0;
};
struct MetalResourceSnapshot {
    uint64_t current_allocated_size = 0;
    uint64_t recommended_max_working_set_size = 0;
    uint64_t registered_weight_bytes = 0;
    uint64_t implicit_weight_copies = 0;
};
void metal_dispatch_metrics_reset();
MetalDispatchMetrics metal_dispatch_metrics();
std::shared_ptr<MetalTokSession> metal_tok_session_create();
void metal_tok_session_enable_error_diagnostics(MetalTokSession&, bool);
const char* metal_tok_session_last_failure(const MetalTokSession&);
void metal_tok_session_require_registered_weights(MetalTokSession&, bool);
uint32_t metal_tok_session_weight_span_coverage(const MetalTokSession&, const void*, size_t);
MetalResourceSnapshot metal_tok_session_resource_snapshot(const MetalTokSession&);
bool metal_tok_session_dense_ready(MetalTokSession&);
bool metal_tok_session_recurrent_ready(MetalTokSession&);
bool metal_tok_session_register_weights(MetalTokSession&, const void* base, size_t size);
void metal_tok_session_unregister_weights(MetalTokSession&, const void* base);
bool metal_tok_session_set_sparse_ffn_runs(MetalTokSession&, const MetalSparseBlockRun*, uint32_t count);
bool metal_tok_session_set_sparse_ffn_layer_ids(MetalTokSession&, const uint32_t* ids,
                                                const uint32_t* offsets,
                                                const uint32_t* counts,
                                                uint32_t layer_count);
bool metal_tok_session_set_sparse_ffn_proxy(MetalTokSession&, uint32_t slot,
                                            const float* coefficients, uint32_t input_blocks,
                                            uint32_t output_blocks, uint32_t selected_blocks);
bool metal_tok_session_set_importance_slots(MetalTokSession&, const uint32_t* widths,
                                            uint32_t slot_count);
bool metal_tok_session_read_importance(const MetalTokSession&, uint32_t slot,
                                       float* sum_squares, uint32_t width,
                                       uint32_t* sample_count);
bool metal_tok_session_begin(MetalTokSession&, int H, int inter, int exp_inter, int n_used, int n_experts,
                             int Hq, int Hk, int Dh, int max_seq, int n_layers, int pos,
                             uint32_t profile_token_count = 1);
bool metal_tok_session_begin_prefill_batch(MetalTokSession&, int H, int inter, int exp_inter,
                                           int n_used, int n_experts, int Hq, int Hk, int Dh,
                                           int max_seq, int n_layers, int pos, uint32_t rows);
bool metal_tok_session_begin_continuing(MetalTokSession&, int H, int inter, int exp_inter, int n_used, int n_experts,
                                        int Hq, int Hk, int Dh, int max_seq, int n_layers, int pos);
bool metal_tok_session_upload_embedding(MetalTokSession&, const Tensor&, uint32_t token, int H, int vocab);
bool metal_tok_session_upload_embeddings_batch(MetalTokSession&, const Tensor&, const uint32_t* tokens,
                                               uint32_t rows, int H, int vocab);
bool metal_tok_session_upload_x(MetalTokSession&, const float*, int H);
bool metal_tok_session_layer(MetalTokSession&, const MetalTokLayer&);
bool metal_tok_session_dense_prefill_batch_layer(MetalTokSession&, const MetalTokLayer&, uint32_t rows);
bool metal_tok_session_select_prefill_batch_row(MetalTokSession&, uint32_t row);
bool metal_tok_session_recurrent_layer(MetalTokSession&, const MetalTokRecurrentLayer&);
bool metal_tok_session_recurrent_commit(MetalTokSession&);
bool metal_tok_session_commit_token(MetalTokSession&);
bool metal_tok_session_seal_token(MetalTokSession&);
bool metal_tok_session_final(MetalTokSession&, const Tensor& norm, const Tensor& lm, float* logits,
                             int H, int vocab, float eps);
MetalTokMetrics metal_tok_session_metrics(const MetalTokSession&);
void metal_tok_session_abort(MetalTokSession&);
#if defined(LAPLACE_METAL_TESTING)
bool metal_tok_session_enable_sparse_ffn_dense_oracle_for_testing(MetalTokSession&, uint32_t slots);
bool metal_tok_session_sparse_ffn_windows_for_testing(const MetalTokSession&, uint32_t* starts,
                                                       uint32_t slots);
const char* metal_tok_recurrent_layer_preflight_for_testing(const MetalTokRecurrentLayer&, int H, int inter, int Dh);
#endif
#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
bool metal_tok_profile_sample_count_for_testing(uint32_t token_count, uint32_t layer_count,
                                                uint64_t* sample_count);
#endif

// Recurrent state stays session-owned. These low-level bindings exist for the
// canonical recurrent lowering and device conformance tests; callers publish a
// token only when step returns true.
bool metal_tok_session_recurrent_seed(MetalTokSession&, const float* history, const float* state,
                                      int qk_heads, int value_heads, int head_dimension, int kernel);
bool metal_tok_session_recurrent_step(MetalTokSession&, const float* qkv, const float* conv_weight,
                                      const float* gate, const float* beta, const float* alpha,
                                      const float* dt_bias, const float* decay, const float* norm,
                                      float* output, int qk_heads, int value_heads, int head_dimension,
                                      int kernel, float l2_epsilon, float rms_epsilon);
bool metal_tok_session_recurrent_snapshot(const MetalTokSession&, float* history, float* state,
                                          int qk_heads, int value_heads, int head_dimension, int kernel);
bool metal_tok_session_recurrent_snapshot_slot(const MetalTokSession&, uint32_t state_slot,
                                               float* history, float* state,
                                               int qk_heads, int value_heads, int head_dimension, int kernel);
#if defined(LAPLACE_METAL_TESTING)
void metal_tok_session_fail_after_completed_submission_for_testing(MetalTokSession&);
#endif
#if defined(LAPLACE_TESTING)
void metal_tok_session_recurrent_fail_after_completed_submission_for_testing(MetalTokSession&);
bool metal_tok_session_download_x_for_testing(const MetalTokSession&, float*, int H);
#endif
#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
bool metal_tok_session_probe_ffn_for_testing(MetalTokSession&, const float*, const Tensor&,
                                              const Tensor&, const Tensor&,
                                              const MetalSparseBlockRun*, uint32_t,
                                              float*, int, int, double*);
#endif

bool metal_tok_begin(int H, int inter, int exp_inter, int n_used, int n_experts,
                     int Hq, int Hk, int Dh, int max_seq, int n_layers, int pos);
bool metal_tok_active();
void metal_tok_upload_x(const float* x, int H);
bool metal_tok_upload_embedding(const Tensor& embedding, uint32_t token, int H, int vocab);
void metal_tok_import_kv(int cache_id, int t, const float* k, const float* v, int kn);
bool metal_tok_kv_needs_seed();
bool metal_tok_layer(const MetalTokLayer& L);
bool metal_tok_flush(double* ms_out);
bool metal_tok_end(float* x, int H);
void metal_tok_abort();
bool metal_tok_lm(const Tensor& w, float* logits, int H, int vocab);
bool metal_tok_final(const Tensor& norm, const Tensor& lm, float* logits,
                     int H, int vocab, float eps);
void metal_register_weights(const void* base, size_t size);
void metal_unregister_weights(const void* base);
bool metal_test_attn(const float* Q, const float* Kc, const float* Vc,
                     float* out, int Hq, int Hk, int Dh, int pos,
                     int window, int max_seq, float scale, int kn_stride = 0);
bool metal_test_q4k_embedding(const Tensor& embedding, uint32_t token, float* output,
                              int width, int vocabulary);
bool metal_test_q6k_embedding(const Tensor& embedding, uint32_t token, float* output,
                              int width, int vocabulary);

// Standalone dequantize (for testing and verification).
// dst must hold exactly `n` floats.
void dequantize(const Tensor& w, float* dst, int n);

// Batch GEMV: dispatch multiple M=1 matmuls in one Metal command buffer.
// Falls back to sequential dispatch when Metal is unavailable.
bool matmul_gemm_batch(const MatmulBatchSpec* specs, int n);

} // namespace Laplace
