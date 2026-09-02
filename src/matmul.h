// matmul.h - GEMM dispatch over quantization formats
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "column_grouped_affine_uint2_skip.h"
#include "tensor.h"

#if defined(LAPLACE_TESTING)
#include "trellis_codec.h"
#endif

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
    DestinationOverlap = 11,
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

// Converts a row-major GGUF Q4_K/Q6_K matrix W[N,K] into the output-block-
// major ColumnGroupedAffineUInt2SkipV1 physical layout. The K-length
// importance vector is the validated activation statistic and is recorded in
// the provenance digest; zero-observation columns receive an unweighted fit
// so the storage remains complete. The caller owns all three output planes.
bool derive_column_grouped_affine_u2_skip_v1_from_gguf(
    GGMLType source_format, std::span<const uint8_t> source,
    uint64_t logical_k, uint64_t logical_n, std::span<const float> importance,
    std::span<uint8_t> packed_weights, std::span<uint16_t> scales,
    std::span<uint16_t> biases,
    ColumnGroupedAffineUInt2SkipV1Storage* storage,
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
// Test-only Metal 4 tensor resource probe. This carries physical facts only;
// it has no model, artifact, or production dispatch ownership.
enum class MetalTensorOpsProbeStorage : uint8_t {
    UInt2 = 1,
    Int2 = 2,
};

enum class MetalTensorOpsProbeStatus : uint8_t {
    Rejected = 0,
    SkippedSdkUnavailable = 1,
    Ran = 2,
    Failed = 3,
};

struct MetalTensorOpsProbePlane {
    uint64_t byte_length = 0;
    uint64_t byte_offset = 0;
};

struct MetalTensorOpsProbeAttachments {
    const void* data = nullptr;
    uint64_t data_bytes = 0;
    const void* scales = nullptr;
    uint64_t scale_bytes = 0;
};

struct MetalTensorOpsProbeDescriptor {
    uint16_t version = 1;
    uint16_t rank = 2;
    MetalTensorOpsProbeStorage storage = MetalTensorOpsProbeStorage::UInt2;
    std::array<uint64_t, 2> dimensions{};
    std::array<uint64_t, 2> strides{};
    MetalTensorOpsProbePlane data_plane;
    MetalTensorOpsProbePlane scale_plane;
    std::array<uint64_t, 2> block_factors{};
    MetalTensorOpsProbeAttachments attachments;
};

struct MetalTensorOpsProbeResult {
    MetalTensorOpsProbeStatus status = MetalTensorOpsProbeStatus::Rejected;
    uint32_t attached_planes = 0;
    uint32_t command_buffers = 0;
    uint32_t implicit_copies = 0;
};

MetalTensorOpsProbeResult metal_test_tensorops_descriptor_probe(
    const MetalTensorOpsProbeDescriptor& descriptor);

// Test-only generic physical trellis primitive. The descriptor carries its
// immutable byte arithmetic and tensor geometry; launch facts stay outside it.
bool metal_test_trellis_decode_dot(
    const TrellisPhysicalDescriptor& descriptor, const uint8_t* packed,
    const float* input, float* output, TrellisCodebook codebook,
    uint32_t warmups, uint32_t samples, double* sample_gpu_ms,
    uint64_t* requested_bytes, uint32_t* command_buffers,
    uint32_t* implicit_copies);

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

// Versioned sampler contract. V1 implements only deterministic greedy
// selection. The temperature, top-k, top-p, and counter-RNG fields are
// reserved in the descriptor so later policies do not change the call ABI.
enum class MetalSamplerMode : uint8_t { Greedy = 1 };
enum class MetalSamplerTiePolicy : uint8_t { FirstIndex = 1 };
enum class MetalSamplerNonFinitePolicy : uint8_t { Reject = 1 };
enum class MetalSamplerResultStatus : uint32_t {
    Success = 0,
    InvalidInput = 1,
    NonFiniteLogit = 2,
    MetalFailure = 3,
};

struct MetalSamplerDescriptor {
    uint16_t version = 1;
    MetalSamplerMode mode = MetalSamplerMode::Greedy;
    MetalSamplerTiePolicy tie_policy = MetalSamplerTiePolicy::FirstIndex;
    MetalSamplerNonFinitePolicy nonfinite_policy = MetalSamplerNonFinitePolicy::Reject;
    uint8_t reserved = 0;
    float temperature = 1.0f;
    uint32_t top_k = 0;
    float top_p = 1.0f;
    uint64_t rng_seed = 0;
    uint64_t rng_counter = 0;
};

// Borrowed MTLBuffer view. `buffer` is an opaque id<MTLBuffer>; the caller
// keeps it alive until metal_sampler_greedy returns.
struct MetalSamplerDeviceLogits {
    const void* buffer = nullptr;
    size_t byte_offset = 0;
    uint32_t vocabulary = 0;
};

struct MetalSamplerResult {
    uint32_t token_id = 0;
    float logit = 0.0f;
    MetalSamplerResultStatus status = MetalSamplerResultStatus::InvalidInput;
    uint32_t reserved = 0;
};
static_assert(sizeof(MetalSamplerResult) == 16, "Metal sampler result ABI");

// Consumes device-resident F32 logits and copies only MetalSamplerResult back
// to the caller. A failed command leaves `result` unchanged.
bool metal_sampler_greedy(const MetalSamplerDescriptor& descriptor,
                          const MetalSamplerDeviceLogits& logits,
                          MetalSamplerResult* result);

#if defined(LAPLACE_TESTING)
// Test-only upload seam. It uploads input logits, then exercises the device
// buffer API; it does not download the full vocabulary.
bool metal_test_sampler_greedy(const MetalSamplerDescriptor& descriptor,
                               const float* logits, uint32_t vocabulary,
                               MetalSamplerResult* result,
                               uint32_t* command_buffers);
#endif

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
    const Tensor* moe_gate = nullptr;
    const Tensor* moe_gate_scale = nullptr;
    const Tensor* moe_up = nullptr;
    const Tensor* moe_dn = nullptr;
    const Tensor* moe_dn_scale = nullptr;
    const Tensor* pre_ffw_2 = nullptr;
    const Tensor* post_ffw_1 = nullptr;
    const Tensor* post_ffw_2 = nullptr;
    const Tensor* post_ffw = nullptr;
    const Tensor* out_scale = nullptr;
    const float* rope_freqs = nullptr;
    int n_rope_freqs = 0;
    uint32_t rope_sections[4]{};
    int H = 0, inter = 0, exp_inter = 0, n_experts = 0, n_used = 0;
    int Hq = 0, Hk = 0, Dh = 0, rope_dim = 0, rope_frequency_dimension = 0,
        window = 0, cache_id = 0;
    // Offset in cached scalar values per sequence position. UINT64_MAX keeps
    // the legacy fixed-stride cache addressing used outside canonical sessions.
    uint64_t cache_width_offset = UINT64_MAX;
    uint32_t moe_router_normalization_scale_bits = 0;
    // Construction-time copy of the authoritative scalar. Keeping the bits
    // here avoids reading a mapped model tensor from the host on every token.
    uint32_t moe_output_scale_bits = 0;
    float rope_base = 0, attention_scale = 0, rms_eps = 0, q_norm_eps = 0, k_norm_eps = 0;
    bool swiglu = false, owns_kv = true, is_global = false, query_gate_split = false,
         moe_gelu_tanh = false, moe_reduce_left_to_right = false, key_state_alias = false,
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
class MetalPipelineLease;
struct MetalPipelineRecipe;
struct MetalFunctionConstant;
// Opaque execution-mode range over the immutable flattened recipe stream.
// The ID has no model, format, tensor, or artifact meaning.
struct MetalTokProgramRange {
    uint32_t id = 0;
    uint32_t first_invocation = 0;
    uint32_t invocation_count = 0;
    uint32_t batch_rows = 1;
    uint32_t selected_output_row = UINT32_MAX;
    std::array<uint8_t, 32> semantic_program_digest{};

    friend bool operator==(const MetalTokProgramRange&,
                           const MetalTokProgramRange&) = default;
};
// Product-only immutable authority for one consumed pipeline invocation. The
// numeric kind fields are the wire values of the structural compiler enums;
// keeping them numeric avoids a dependency from the Metal ABI onto the
// compiler implementation. pipeline_slot is the transaction-resolved slot;
// recipe_index remains the authenticated structural recipe index.
struct MetalTokInvocationAuthority {
    uint16_t version = 1;
    uint16_t group_kind = 0;
    uint16_t group_shape = 0;
    uint16_t primitive = 0;
    uint32_t program_id = UINT32_MAX;
    uint32_t invocation_ordinal = UINT32_MAX;
    uint32_t group_ordinal = UINT32_MAX;
    uint32_t primitive_order = UINT32_MAX;
    uint32_t recipe_index = UINT32_MAX;
    uint32_t pipeline_slot = UINT32_MAX;
    uint32_t batch_rows = 1;
    uint32_t row_index = UINT32_MAX;
    uint32_t row_count = 1;
    std::array<uint8_t, 32> semantic_program_digest{};
    std::array<uint8_t, 32> bound_program_digest{};
    std::array<uint8_t, 32> invocation_digest{};

    friend bool operator==(const MetalTokInvocationAuthority&,
                           const MetalTokInvocationAuthority&) = default;
};
struct MetalTokAttentionCapacity {
    int query_width = 0;
    int key_value_width = 0;
    // Sum of the exact per-attention-layer K/V widths. Zero selects the
    // legacy layer-count times maximum-width allocation.
    uint64_t key_value_width_sum = 0;
};
struct MetalSparseBlockRun {
    uint32_t first = 0;
    uint32_t count = 0;
};
struct MetalTokMetrics {
    double cpu_wait_ms = 0.0;
    double gpu_time_ms = 0.0;
    uint32_t command_buffers = 0;
    uint64_t peak_session_bytes = 0;
    uint64_t kv_cache_bytes = 0;
    uint64_t requested_projection_source_bytes = 0;
    uint32_t projection_dispatches = 0;
    // A projection dispatched with more than one activation row. This is
    // intentionally separate from the legacy total so a one-command-buffer
    // serial prefill cannot masquerade as a batch implementation.
    uint32_t batched_projection_dispatches = 0;
    uint32_t q4k_projection_dispatches = 0;
    uint32_t q6k_projection_dispatches = 0;
    uint32_t grouped_affine_u2_projection_dispatches = 0;
    uint32_t column_grouped_affine_u2_skip_projection_dispatches = 0;
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
    uint64_t session_owned_metadata_bytes = 0;
    uint64_t transient_workspace_bytes = 0;
    uint64_t residency_allocated_size = 0;
    uint32_t residency_allocation_count = 0;
    bool residency_set_supported = false;
    bool residency_set_committed = false;
    bool residency_requested = false;
};
struct MetalTokMoeCapabilities {
    bool router_topk = false;
    bool gate_up_q4_k = false;
    bool down_q5_0 = false;
    bool down_q8_0 = false;
    bool reduce = false;
};
void metal_dispatch_metrics_reset();
MetalDispatchMetrics metal_dispatch_metrics();
std::shared_ptr<MetalTokSession> metal_tok_session_create();
// Product-only construction boundary. The lease remains untouched when
// validation or queue creation fails, and moves into the published session on
// success. Leased dispatch consumes the direct slot table retained by the
// session; legacy zero-argument diagnostics retain their existing path.
std::shared_ptr<MetalTokSession> metal_tok_session_create_from_pipeline_lease(
    MetalPipelineLease&& lease,
    std::span<const MetalPipelineRecipe> ordered_recipes);
std::shared_ptr<MetalTokSession> metal_tok_session_create_from_pipeline_lease(
    MetalPipelineLease&& lease,
    std::span<const MetalPipelineRecipe> ordered_recipes,
    std::span<const uint32_t> flattened_invocation_recipe_indices,
    std::span<const MetalTokProgramRange> program_ranges);
std::shared_ptr<MetalTokSession> metal_tok_session_create_from_pipeline_lease(
    MetalPipelineLease&& lease,
    std::span<const MetalPipelineRecipe> ordered_recipes,
    std::span<const uint32_t> flattened_invocation_recipe_indices,
    std::span<const MetalTokProgramRange> program_ranges,
    std::span<const MetalTokInvocationAuthority> invocation_authorities);
void metal_tok_session_enable_error_diagnostics(MetalTokSession&, bool);
const char* metal_tok_session_last_failure(const MetalTokSession&);
void metal_tok_session_require_registered_weights(MetalTokSession&, bool);
uint32_t metal_tok_session_weight_span_coverage(const MetalTokSession&, const void*, size_t);
MetalResourceSnapshot metal_tok_session_resource_snapshot(const MetalTokSession&);
bool metal_tok_session_dense_ready(MetalTokSession&);
// Capability snapshot for the canonical three-plane UInt2/256 GEMV leaf.
// This is a query of the loaded Metal library and pipeline limits, not a
// device-family or model-name assumption.
bool metal_tok_session_affine_u2_256_ready(MetalTokSession&);
// Capability and session-owned metadata binding for the distinct
// output-block-major UInt2 layout. This never aliases the legacy row-major
// GROUPED_AFFINE_U2_256 contract.
bool metal_tok_session_column_grouped_affine_u2_skip_256_ready(MetalTokSession&);
bool metal_tok_session_register_column_grouped_affine_u2_skip_256(
    MetalTokSession&, const Tensor&);
MetalTokMoeCapabilities metal_tok_session_moe_capabilities(MetalTokSession&);
bool metal_tok_session_moe_ready(MetalTokSession&);
bool metal_tok_session_recurrent_ready(MetalTokSession&);
bool metal_tok_session_register_weights(MetalTokSession&, const void* base, size_t size);
bool metal_tok_session_prepare_weight_residency(MetalTokSession&);
void metal_tok_session_unregister_weights(MetalTokSession&, const void* base);
#if defined(LAPLACE_TESTING)
// Test-only registration seam. The chunk limit forces split-buffer coverage;
// fail_after_chunk uses zero-based indexing and UINT32_MAX disables failure.
bool metal_tok_session_register_weights_for_testing(MetalTokSession&, const void* base,
                                                    size_t size, size_t chunk_limit,
                                                    uint32_t fail_after_chunk);
#endif
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
bool metal_tok_session_begin_program(MetalTokSession&, uint32_t program_id);
bool metal_tok_session_select_program(MetalTokSession&, uint32_t program_id);
bool metal_tok_session_begin_with_attention_capacity(
    MetalTokSession&, int H, int inter, int exp_inter, int n_used, int n_experts,
    MetalTokAttentionCapacity, int max_seq, int n_layers, int pos,
    uint32_t profile_token_count = 1);
bool metal_tok_session_begin_prefill_batch(MetalTokSession&, int H, int inter, int exp_inter,
                                           int n_used, int n_experts, int Hq, int Hk, int Dh,
                                           int max_seq, int n_layers, int pos, uint32_t rows);
bool metal_tok_session_begin_prefill_batch_with_attention_capacity(
    MetalTokSession&, int H, int inter, int exp_inter, int n_used, int n_experts,
    MetalTokAttentionCapacity, int max_seq, int n_layers, int pos, uint32_t rows);
bool metal_tok_session_begin_continuing(MetalTokSession&, int H, int inter, int exp_inter, int n_used, int n_experts,
                                        int Hq, int Hk, int Dh, int max_seq, int n_layers, int pos);
bool metal_tok_session_begin_continuing_with_attention_capacity(
    MetalTokSession&, int H, int inter, int exp_inter, int n_used, int n_experts,
    MetalTokAttentionCapacity, int max_seq, int n_layers, int pos);
bool metal_tok_session_upload_embedding(MetalTokSession&, const Tensor&, uint32_t token, int H, int vocab,
                                        float scale);
bool metal_tok_session_upload_embeddings_batch(MetalTokSession&, const Tensor&, const uint32_t* tokens,
                                               uint32_t rows, int H, int vocab, float scale);
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
bool metal_tok_session_final_sampled(MetalTokSession&, const Tensor& norm, const Tensor& lm,
                                     const MetalSamplerDescriptor&, MetalSamplerResult*,
                                     int H, int vocab, float eps);
MetalTokMetrics metal_tok_session_metrics(const MetalTokSession&);
void metal_tok_session_abort(MetalTokSession&);
#if defined(LAPLACE_TESTING)
uintptr_t metal_tok_session_queue_identity_for_testing(const MetalTokSession&);
uintptr_t metal_tok_session_queue_device_identity_for_testing(const MetalTokSession&);
uint32_t metal_tok_session_pipeline_lease_slot_count_for_testing(
    const MetalTokSession&);
bool metal_tok_session_dispatch_leased_slot_for_testing(MetalTokSession&, uint32_t slot);
bool metal_tok_session_probe_global_pipeline_for_testing(MetalTokSession&);
bool metal_tok_session_dispatch_program_invocation_for_testing(
    MetalTokSession&, const char* function,
    std::span<const MetalFunctionConstant> constants = {});
uintptr_t metal_tok_session_pipeline_device_identity_for_testing(const MetalTokSession&, uint32_t slot);
struct MetalPipelineGlobalLookupMetrics {
    uint64_t lookup_attempts = 0;
    uint64_t compilation_attempts = 0;
};
void metal_pipeline_global_lookup_metrics_reset_for_testing();
MetalPipelineGlobalLookupMetrics metal_pipeline_global_lookup_metrics_for_testing();
#endif
#if defined(LAPLACE_METAL_TESTING)
bool metal_tok_session_enable_sparse_ffn_dense_oracle_for_testing(MetalTokSession&, uint32_t slots);
bool metal_tok_session_sparse_ffn_windows_for_testing(const MetalTokSession&, uint32_t* starts,
                                                       uint32_t slots);
const char* metal_tok_recurrent_layer_preflight_for_testing(const MetalTokRecurrentLayer&, int H, int inter, int Dh);
#endif
#if defined(LAPLACE_METAL_TESTING) || defined(LAPLACE_TESTING)
enum class MetalRouterScoreDomain : uint8_t { Logits = 1 };
enum class MetalRouterNormalization : uint8_t { SelectThenNormalizeSoftmax = 1 };
enum class MetalRouterTiePolicy : uint8_t { LowestExpertId = 1 };

enum class MetalMoeActivationTestPath : uint8_t {
    PerExpertLoop = 1,
    Batched2D = 2,
};

struct MetalMoeActivationTestMetrics {
    double gpu_ms = 0.0;
    uint32_t command_buffers = 0;
    uint32_t activation_dispatches = 0;
    bool expert_ids_unchanged = false;
    bool gate_up_unchanged = false;
};

// Test-only A/B seam for the selected-expert activation slice. gate_up is
// laid out as [selected][gate, up][intermediate]. Both paths execute in one
// command buffer; Batched2D must encode exactly one activation dispatch.
bool metal_test_moe_batched_activation(
    const float* gate_up, size_t gate_up_values,
    const uint32_t* expert_ids, uint32_t selected_count,
    uint32_t intermediate, bool swiglu, MetalMoeActivationTestPath path,
    float* output, size_t output_values,
    MetalMoeActivationTestMetrics* metrics);

struct MetalRouterTopKSpec {
    uint32_t expert_count = 0;
    uint32_t selected_count = 0;
    MetalRouterScoreDomain score_domain = MetalRouterScoreDomain::Logits;
    MetalRouterNormalization normalization =
        MetalRouterNormalization::SelectThenNormalizeSoftmax;
    MetalRouterTiePolicy tie_policy = MetalRouterTiePolicy::LowestExpertId;
};

struct MetalRouterPipelineCaps {
    uint32_t thread_execution_width = 0;
    uint32_t max_total_threads_per_threadgroup = 0;
};

bool metal_test_router_top_k(const MetalRouterTopKSpec&, const float* logits,
                             uint32_t* ids, float* weights,
                             MetalRouterPipelineCaps* capabilities);

struct MetalMoeDownReduceSpec {
    uint32_t input_width = 0;
    uint32_t output_width = 0;
    uint32_t expert_count = 0;
    uint32_t selected_count = 0;
};

struct MetalMoeDownReducePipelineCaps {
    uint32_t thread_execution_width = 0;
    uint32_t max_total_threads_per_threadgroup = 0;
};

// Test-only gathered down projection and deterministic weighted reduction.
// The worklist owns expert order. expert_scales is optional, but when present
// it must contain one finite F32 value for every expert.
bool metal_test_moe_down_reduce(
    const MetalMoeDownReduceSpec&, const Tensor&, size_t source_bytes,
    const float* input, size_t input_values, const uint32_t* expert_ids,
    const float* route_weights, const float* expert_scales,
    size_t expert_scale_values, float* output, size_t output_values,
    double* gpu_ms, MetalMoeDownReducePipelineCaps* capabilities);

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

// One GPU command buffer for dense FFN + routed experts. QKV stays on CPU.
bool matmul_decode_ffn_moe(
    const float* x_norm, const Tensor& ffn_gate, const Tensor& ffn_up,
    const Tensor& ffn_down, float* xb, int H, int inter, bool swiglu,
    const float* moe_in, const Tensor* moe_up_stack, const Tensor* moe_dn_stack,
    const int* expert_ids, int n_exp, int exp_inter, const float* route_w,
    float* moe_out);

// Run subsequent matmul_rows on a GCD queue (E-cores) instead of the
// P-core ThreadPool. For overlapping dense FFN with MoE.
void matmul_use_gcd(bool on);

// Fused MoE GEMV with indirect expert access. expert_idx[k] selects which
// expert from the stacked weight tensor to use. y[k * N + j] = output of
// expert expert_idx[k], column j. One parallel_for for all experts.
void fused_moe_gemm_idx(const float* x, const Tensor& w, float* y,
                        const int* expert_idx, int n_experts,
                        int K, int N);

// Fused MoE GEMV with per-expert activations. x[k * K + i] is expert k's
// input. y[k * N + j] = output of expert expert_idx[k], column j.
// One parallel_for for all experts.
void fused_moe_gemm_multi(const float* x, const Tensor& w, float* y,
                          const int* expert_idx, int n_experts,
                          int K, int N);

// Computes the selected experts' gate and up projections and applies GeGLU
// without materializing the 2 * hidden_dim projection.
bool fused_moe_gate_up_geglu(const float* x, const Tensor& w, float* hidden,
                             const int* expert_idx, int n_experts,
                             int K, int hidden_dim,
                             const uint8_t* const* bases = nullptr);

// Adds the route-weighted selected expert projections directly to output,
// without materializing one output vector per expert.
bool fused_moe_down_accumulate(const float* x, const Tensor& w,
                               const int* expert_idx, const float* route_weight,
                               int n_experts, int K, int N, float* output,
                               const uint8_t* const* bases = nullptr);

// Standalone dequantize (for testing and verification).
// dst must hold exactly `n` floats.
void dequantize(const Tensor& w, float* dst, int n);

// Batch GEMV: dispatch multiple M=1 matmuls in one Metal command buffer.
// Falls back to sequential dispatch when Metal is unavailable.
bool matmul_gemm_batch(const MatmulBatchSpec* specs, int n);

} // namespace Laplace
