#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <variant>
#include <vector>

#include "compat_rule.h"
#include "execution_plan.h"
#include "sparse_ffn_bank.h"
#if defined(LAPLACE_METAL_TESTING)
#include "tensor.h"
#endif

namespace Laplace {

class ProductMetalPreparedSession;

struct CanonicalMetalOutput {
    std::vector<float> logits;
    uint32_t sampled_token_id = 0;
    float sampled_logit = 0.0f;
    uint64_t host_result_bytes = 0;
    bool sampled = false;
    uint32_t command_buffers = 0;
    uint32_t operator_count = 0;
    double cpu_wait_ms = 0.0;
    double gpu_time_ms = 0.0;
    uint64_t peak_session_bytes = 0;
    uint64_t kv_cache_bytes = 0;
    uint64_t requested_projection_source_bytes = 0;
    uint32_t projection_dispatches = 0;
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
    bool completed = false;
};

struct CanonicalMetalResourceDiagnostics {
    uint64_t recommended_max_working_set_size = 0;
    uint64_t before_source_registration = 0;
    uint64_t after_source_registration = 0;
    uint64_t after_atlas_registration = 0;
    uint64_t after_session_construction = 0;
    uint64_t registered_source_bytes = 0;
    uint64_t excluded_replaced_bytes = 0;
    uint64_t retained_boundary_bytes = 0;
    uint64_t registration_overlap_bytes = 0;
    uint64_t atlas_bytes = 0;
    uint64_t implicit_weight_copies = 0;
    uint64_t residency_allocated_size = 0;
    uint32_t residency_allocation_count = 0;
    bool residency_set_supported = false;
    bool residency_set_committed = false;
    bool residency_requested = false;
};

struct CanonicalMetalCursor {
    uint64_t session_id = 0;
    uint64_t generation = 0;
    uint32_t position = 0;
    friend bool operator==(const CanonicalMetalCursor&, const CanonicalMetalCursor&) = default;
};

struct CanonicalDerivedQ2KPolicy {
    std::vector<TensorRole> tensor_roles;
    bool register_retained_source_ranges_for_testing = false;
};

struct CalibrationCacheBundle;

struct CanonicalDerivedIQ2XXSPolicy {
    std::vector<TensorRole> tensor_roles;
    const CalibrationCacheBundle* calibration_cache = nullptr;
    bool enable_metal_error_diagnostics = false;
#if defined(LAPLACE_METAL_TESTING)
    bool zero_fill_for_testing = false;
#endif
};

struct CanonicalDerivedColumnGroupedU2Policy {
    std::vector<TensorRole> tensor_roles;
    const CalibrationCacheBundle* calibration_cache = nullptr;
#if defined(LAPLACE_METAL_TESTING)
    bool uniform_importance_for_testing = false;
#endif
};

struct CalibrationKMapping {
    uint8_t input_axis = 0xff;
    uint8_t weight_physical_axis = 0xff;
    uint8_t weight_logical_axis = 0xff;
    uint8_t reserved = 0;
    uint32_t width = 0;
    uint32_t block_elements = 0;
    uint32_t block_bytes = 0;
    friend bool operator==(const CalibrationKMapping&, const CalibrationKMapping&) = default;
};

struct CalibrationTarget {
    uint16_t version = 1;
    uint16_t flags = 0;
    uint32_t operator_id = UINT32_MAX;
    uint32_t input_value_id = UINT32_MAX;
    uint32_t weight_tensor_id = UINT32_MAX;
    CalibrationKMapping k_mapping;
    friend bool operator==(const CalibrationTarget&, const CalibrationTarget&) = default;
};

struct CalibrationRecord {
    CalibrationTarget target;
    uint32_t sample_count = 0;
    std::vector<float> sum_squares;
};

struct CanonicalCalibrationPolicy {
    std::vector<CalibrationTarget> targets;
};

struct CalibrationCacheBundle {
    uint16_t version = 1;
    uint16_t accumulator_version = 1;
    uint16_t converter_version = 1;
    uint16_t format_version = 1;
    Sha256Digest artifact_digest;
    Sha256Digest semantic_fingerprint;
    Sha256Digest target_set_digest;
    Sha256Digest corpus_digest;
    Sha256Digest token_digest;
    std::vector<CalibrationRecord> records;
};

using CalibrationCacheDecode = std::variant<CalibrationCacheBundle, CompatibilityReport>;
Sha256Digest calibration_target_set_digest(std::span<const CalibrationTarget> targets);
Sha256Digest calibration_bytes_digest(std::span<const uint8_t> bytes);
bool calibration_token_window_digest(std::span<const uint32_t> token_ids,
                                     std::span<const uint32_t> window_lengths,
                                     Sha256Digest& digest);
bool encode_calibration_cache(const CalibrationCacheBundle&, std::vector<uint8_t>& bytes);
bool normalize_calibration_record(const CalibrationRecord&, std::vector<float>& importance);
bool calibration_importance_for_tensor(const CalibrationCacheBundle&,
                                       const Sha256Digest& artifact_digest,
                                       const Sha256Digest& semantic_fingerprint,
                                       uint32_t weight_tensor_id,
                                       const CalibrationKMapping& expected_mapping,
                                       std::vector<float>& importance);
bool merge_calibration_records(std::vector<CalibrationRecord>& accumulated,
                               std::span<const CalibrationRecord> window);
bool write_calibration_cache_atomic(std::string_view path, const CalibrationCacheBundle&,
                                    CompatibilityReport* error = nullptr);
CalibrationCacheDecode decode_calibration_cache(std::span<const uint8_t> bytes,
                                                const Sha256Digest& artifact_digest,
                                                const Sha256Digest& semantic_fingerprint,
                                                const Sha256Digest& target_set_digest,
                                                const Sha256Digest& corpus_digest,
                                                const Sha256Digest& token_digest);
CalibrationCacheDecode load_calibration_cache(std::string_view path,
                                              const Sha256Digest& artifact_digest,
                                              const Sha256Digest& semantic_fingerprint,
                                              const Sha256Digest& target_set_digest,
                                              const Sha256Digest& corpus_digest,
                                              const Sha256Digest& token_digest);
#if defined(LAPLACE_METAL_TESTING)
bool calibration_cache_value_count_valid_for_testing(uint64_t total, uint64_t next);
bool canonical_metal_iq2_atlas_cache_roundtrip_for_testing(std::string_view path,
                                                           uint32_t* conversions);
#endif

using CalibrationTargetValidation = std::variant<CalibrationTarget, CompatibilityReport>;
CalibrationTargetValidation validate_calibration_target(const SemanticModel&, const CalibrationTarget&,
                                                         bool metal_capability_available);

struct CanonicalSparseFfnPolicy {
    struct LayerMask {
        uint32_t layer_index = 0;
        std::vector<uint32_t> block_ids;
    };
    std::vector<SparseBlockRun> runs;
    std::vector<LayerMask> layer_masks;
    uint32_t proxy_selected_blocks = 0;
    uint32_t dense_oracle_selected_blocks = 0;
};

struct CanonicalSparseFfnWindowDiagnostic {
    uint32_t layer_index = 0;
    uint32_t surrogate_first = 0;
    uint32_t oracle_first = 0;
    uint32_t oracle_second = 0;
};

class CanonicalMetalProgram {
public:
    CanonicalMetalProgram(CanonicalMetalProgram&&) noexcept;
    CanonicalMetalProgram& operator=(CanonicalMetalProgram&&) noexcept;
    ~CanonicalMetalProgram();
    CanonicalMetalProgram(const CanonicalMetalProgram&) = delete;
    CanonicalMetalProgram& operator=(const CanonicalMetalProgram&) = delete;

    std::variant<CanonicalMetalOutput, CompatibilityReport> prefill(std::span<const uint32_t> token_ids);
    std::variant<CanonicalMetalOutput, CompatibilityReport> decode(uint32_t token_id);
    std::variant<CanonicalMetalOutput, CompatibilityReport> prefill_sampled(
        std::span<const uint32_t> token_ids);
    std::variant<CanonicalMetalOutput, CompatibilityReport> decode_sampled(uint32_t token_id);
    std::variant<CanonicalMetalOutput, CompatibilityReport> accumulate_calibration(uint32_t token_id);
    uint32_t layer_count() const noexcept;
    uint32_t position() const noexcept;
    const ExecutionPlan& plan() const noexcept;
    bool has_recurrent_layers() const noexcept;
    uint32_t derived_q2_storage_count() const noexcept;
    uint64_t derived_q2_storage_bytes() const noexcept;
    uint64_t original_source_registered_bytes() const noexcept;
    uint64_t derived_q2_registered_bytes() const noexcept;
    uint64_t retained_boundary_bytes() const noexcept;
    uint32_t derived_iq2_xxs_atlas_count() const noexcept;
    uint32_t derived_iq2_xxs_tensor_count() const noexcept;
    uint64_t derived_iq2_xxs_source_bytes() const noexcept;
    uint64_t derived_iq2_xxs_storage_bytes() const noexcept;
    uint64_t derived_iq2_xxs_registered_bytes() const noexcept;
    uint32_t derived_column_grouped_u2_tensor_count() const noexcept;
    uint64_t derived_column_grouped_u2_source_bytes() const noexcept;
    uint64_t derived_column_grouped_u2_storage_bytes() const noexcept;
    CanonicalMetalResourceDiagnostics resource_diagnostics() const noexcept;
    uint32_t sparse_ffn_layer_count() const noexcept;
    uint64_t sparse_ffn_source_bytes() const noexcept;
    uint64_t sparse_ffn_requested_bytes() const noexcept;
    uint64_t sparse_ffn_worklist_bytes() const noexcept;
#if defined(LAPLACE_METAL_TESTING)
    std::vector<uint32_t> sparse_ffn_block_counts_for_testing() const;
#endif
    bool read_calibration(std::vector<CalibrationRecord>& records) const;
    CanonicalMetalCursor checkpoint() const noexcept;
    bool commit(CanonicalMetalCursor) noexcept;
    bool rollback(CanonicalMetalCursor) noexcept;
    bool rollback_to_position(uint32_t position) noexcept;

private:
    enum class OutputMode : uint8_t { None = 0, Logits = 1, GreedySample = 2 };
    struct Impl;
    explicit CanonicalMetalProgram(std::unique_ptr<Impl> impl);
    std::variant<CanonicalMetalOutput, CompatibilityReport> run(std::span<const uint32_t> token_ids,
                                                                  ExecutionPhase phase,
                                                                  OutputMode output_mode);
    std::variant<CanonicalMetalOutput, CompatibilityReport> advance_prefill(uint32_t token_id);
    std::variant<CanonicalMetalOutput, CompatibilityReport> advance_prefill_batch(
        std::span<const uint32_t> token_ids);
    std::unique_ptr<Impl> impl_;

    friend class RuntimeSession;

    friend std::variant<CanonicalMetalProgram, CompatibilityReport>
    create_canonical_metal_program(std::shared_ptr<const RuntimePackage>, const SessionRequest&,
                                   const CanonicalDerivedQ2KPolicy&, const CanonicalSparseFfnPolicy&,
                                   const CanonicalDerivedIQ2XXSPolicy&, const CanonicalCalibrationPolicy&,
                                   const CanonicalDerivedColumnGroupedU2Policy&);
    friend std::variant<CanonicalMetalProgram, CompatibilityReport>
    create_canonical_metal_program_internal(std::shared_ptr<const RuntimePackage>, SessionRequest,
                                            const CanonicalDerivedQ2KPolicy&, const CanonicalSparseFfnPolicy&,
                                            const CanonicalDerivedIQ2XXSPolicy&, const CanonicalCalibrationPolicy&,
                                            const CanonicalDerivedColumnGroupedU2Policy&,
                                            ProductMetalPreparedSession*);
#if defined(LAPLACE_QUALIFICATION_RUNTIME) || defined(LAPLACE_METAL_TESTING)
    friend std::variant<CanonicalMetalProgram, CompatibilityReport>
    create_qualification_canonical_metal_program(std::shared_ptr<const RuntimePackage>, uint32_t,
                                                 const CanonicalDerivedQ2KPolicy&, const CanonicalSparseFfnPolicy&,
                                                 const CanonicalDerivedIQ2XXSPolicy&, const CanonicalCalibrationPolicy&,
                                                 uint32_t, const CanonicalDerivedColumnGroupedU2Policy&);
#endif
#if defined(LAPLACE_METAL_TESTING)
    friend void canonical_metal_fail_after_completed_submission_for_testing(CanonicalMetalProgram&);
    friend const char* canonical_metal_first_recurrent_preflight_for_testing(const CanonicalMetalProgram&);
    friend bool canonical_metal_sparse_ffn_probe_for_testing(CanonicalMetalProgram&,
                                                              std::span<const float>,
                                                              const CanonicalSparseFfnPolicy&,
                                                              std::vector<float>&, double&);
    friend bool canonical_metal_sparse_ffn_windows_for_testing(
        const CanonicalMetalProgram&, std::vector<CanonicalSparseFfnWindowDiagnostic>&);
#endif
};

using CanonicalMetalCreateResult = std::variant<CanonicalMetalProgram, CompatibilityReport>;
using CanonicalMetalRunResult = std::variant<CanonicalMetalOutput, CompatibilityReport>;

CanonicalMetalCreateResult create_canonical_metal_program(std::shared_ptr<const RuntimePackage> package,
                                                            const SessionRequest& request,
                                                            const CanonicalDerivedQ2KPolicy& derived_q2 = {},
                                                            const CanonicalSparseFfnPolicy& sparse_ffn = {},
                                                            const CanonicalDerivedIQ2XXSPolicy& derived_iq2_xxs = {},
                                                            const CanonicalCalibrationPolicy& calibration = {},
                                                            const CanonicalDerivedColumnGroupedU2Policy& derived_column_u2 = {});

#if defined(LAPLACE_QUALIFICATION_RUNTIME) || defined(LAPLACE_METAL_TESTING)
// Qualification and device tests may exercise an unpromoted package. This API
// is absent from product builds.
CanonicalMetalCreateResult create_qualification_canonical_metal_program(
    std::shared_ptr<const RuntimePackage> package, uint32_t maximum_context,
    const CanonicalDerivedQ2KPolicy& derived_q2 = {},
    const CanonicalSparseFfnPolicy& sparse_ffn = {},
    const CanonicalDerivedIQ2XXSPolicy& derived_iq2_xxs = {},
    const CanonicalCalibrationPolicy& calibration = {},
    uint32_t maximum_batch_rows = 1,
    const CanonicalDerivedColumnGroupedU2Policy& derived_column_u2 = {});
#endif

#if defined(LAPLACE_METAL_TESTING)
void canonical_metal_fail_after_completed_submission_for_testing(CanonicalMetalProgram&);
bool canonical_metal_tensor_view_for_testing(const RuntimePackage&, const SemanticTensor&, Tensor&);
const char* canonical_metal_first_recurrent_preflight_for_testing(const CanonicalMetalProgram&);
bool canonical_metal_sparse_ffn_probe_for_testing(CanonicalMetalProgram&, std::span<const float>,
                                                   const CanonicalSparseFfnPolicy&,
                                                   std::vector<float>&, double&);
bool canonical_metal_sparse_ffn_windows_for_testing(
    const CanonicalMetalProgram&, std::vector<CanonicalSparseFfnWindowDiagnostic>&);
#endif

} // namespace Laplace
