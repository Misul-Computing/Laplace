#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <variant>
#include <vector>

#include "canonical_metal.h"
#include "compat_rule.h"
#include "execution_plan.h"
#include "program_metal.h"
#include "session_resources.h"
#include "state_abi.h"

namespace Laplace {

class ProductPackage;
class VerifiedProgramPackage;
class VerifiedPhysicalProgramPackage;

enum class SessionFaultPoint : uint8_t {
    None = 0, DeviceQuery = 1, CommandQueue = 2, Plan = 3, StateAllocation = 4,
};

struct RuntimeOutput {
    std::vector<float> logits;
    uint32_t sampled_token_id = 0;
    float sampled_logit = 0.0f;
    uint64_t host_result_bytes = 0;
    bool sampled = false;
    std::vector<std::vector<float>> checkpoints;
    std::vector<SemanticKvState> states;
    std::vector<uint32_t> token_history;
    uint32_t command_buffers = 0;
    uint32_t operator_count = 0;
    double cpu_wait_ms = 0.0;
    double gpu_time_ms = 0.0;
    uint64_t peak_session_bytes = 0;
    uint64_t requested_projection_source_bytes = 0;
    uint32_t projection_dispatches = 0;
    uint32_t grouped_affine_u2_projection_dispatches = 0;
    uint32_t column_grouped_affine_u2_skip_projection_dispatches = 0;
    bool completed = false;
    uint64_t operator_token_work = 0;
};

using RuntimeRunResult = std::variant<RuntimeOutput, CompatibilityReport>;

class RuntimeSession {
public:
    RuntimeSession(RuntimeSession&& other) noexcept;
    RuntimeSession& operator=(RuntimeSession&& other) noexcept;
    RuntimeSession(const RuntimeSession&) = delete;
    RuntimeSession& operator=(const RuntimeSession&) = delete;

    RuntimeRunResult prefill(std::span<const uint32_t> token_ids);
    RuntimeRunResult decode(uint32_t token_id);
    RuntimeRunResult prefill_sampled(std::span<const uint32_t> token_ids);
    RuntimeRunResult decode_sampled(uint32_t token_id);
    StateCursor checkpoint() const noexcept;
    StateMutationResult commit(StateCursor cursor);
    StateMutationResult rollback(StateCursor cursor);
    StateSaveResult save_state() const;
    StateMutationResult restore_state(std::span<const uint8_t> bytes);
    const std::vector<uint32_t>& token_history() const noexcept;
    const ExecutionPlan& plan() const noexcept { return plan_; }
#if defined(LAPLACE_METAL_TESTING)
    void fail_after_completed_submission_for_testing() noexcept;
    void fail_after_prefill_tokens_for_testing(uint32_t count) noexcept;
    uint64_t implicit_weight_copy_count_for_testing() const noexcept;
#endif

private:
    friend std::variant<RuntimeSession, CompatibilityReport> create_runtime_session(
        const ProductPackage&, SessionRequest, SessionFaultPoint);
    friend std::variant<RuntimeSession, CompatibilityReport> create_runtime_session(
        const VerifiedProgramPackage&, SessionRequest, SessionFaultPoint);
#if defined(LAPLACE_QUALIFICATION_RUNTIME)
    friend std::variant<RuntimeSession, CompatibilityReport> create_qualification_runtime_session(
        std::shared_ptr<const RuntimePackage>, SessionRequest, SessionFaultPoint);
    friend std::variant<RuntimeSession, CompatibilityReport> create_runtime_session_internal(
        std::shared_ptr<const RuntimePackage>, SessionRequest, SessionFaultPoint);
#endif
    friend std::variant<RuntimeSession, CompatibilityReport> create_runtime_session_internal(
        const ProductPackage&, SessionRequest, SessionFaultPoint);
    friend std::variant<RuntimeSession, CompatibilityReport>
    create_product_runtime_session_internal(std::shared_ptr<const RuntimePackage>,
                                            SessionRequest, SessionFaultPoint);
#if defined(LAPLACE_METAL_TESTING)
    friend std::variant<RuntimeSession, CompatibilityReport>
    create_product_runtime_session_for_testing(std::shared_ptr<const RuntimePackage>,
                                               SessionRequest, SessionFaultPoint);
#endif

    RuntimeSession(std::shared_ptr<const RuntimePackage> package, ExecutionPlan plan, SessionResources resources,
                   StateStore state, std::unique_ptr<CanonicalMetalProgram> metal = {}, bool product_route = false,
                   SessionRequest request = {},
                   std::shared_ptr<const VerifiedPhysicalProgramPackage> physical_package = {})
        : package_(std::move(package)), request_(request), plan_(std::move(plan)), resources_(std::move(resources)),
          state_(std::move(state)), metal_(std::move(metal)), product_route_(product_route),
          physical_package_(std::move(physical_package)) {}
    RuntimeSession(std::shared_ptr<const VerifiedProgramPackage> package,
                   SessionRequest request,
                   std::unique_ptr<MetalProgramExecutable> metal,
                   uint32_t token_input_value, uint32_t score_result_index)
        : request_(request), state_(SemanticModel{}), product_route_(true),
          program_package_(std::move(package)),
          program_metal_(std::move(metal)),
          program_token_input_value_(token_input_value),
          program_score_result_index_(score_result_index) {}

    RuntimeRunResult execute(std::span<const uint32_t> token_ids, ExecutionPhase phase,
                             bool sampled = false);
    bool restore_product_prefix();
    void clear_after_move() noexcept;

    std::shared_ptr<const RuntimePackage> package_;
    SessionRequest request_;
    ExecutionPlan plan_;
    SessionResources resources_;
    StateStore state_ = StateStore(SemanticModel{});
    std::unique_ptr<CanonicalMetalProgram> metal_;
    bool product_route_ = false;
    std::shared_ptr<const VerifiedPhysicalProgramPackage> physical_package_;
    std::shared_ptr<const VerifiedProgramPackage> program_package_;
    std::unique_ptr<MetalProgramExecutable> program_metal_;
    uint32_t program_token_input_value_ = UINT32_MAX;
    uint32_t program_score_result_index_ = UINT32_MAX;
    std::vector<uint64_t> program_token_extents_;
    std::vector<uint64_t> program_score_extents_;
    bool poisoned_ = false;
    uint64_t product_store_id_ = 0;
    uint64_t product_generation_ = 1;
    std::vector<uint32_t> product_history_;
#if defined(LAPLACE_METAL_TESTING)
    uint32_t product_prefill_failure_after_ = UINT32_MAX;
#endif
};

using SessionCreateResult = std::variant<RuntimeSession, CompatibilityReport>;

SessionCreateResult create_runtime_session(const ProductPackage& package, SessionRequest request,
                                           SessionFaultPoint fault = SessionFaultPoint::None);
SessionCreateResult create_runtime_session(
    const VerifiedProgramPackage& package, SessionRequest request,
    SessionFaultPoint fault = SessionFaultPoint::None);
#if defined(LAPLACE_METAL_TESTING)
SessionCreateResult create_product_runtime_session_for_testing(
    std::shared_ptr<const RuntimePackage> package, SessionRequest request,
    SessionFaultPoint fault = SessionFaultPoint::None);
#endif
#if defined(LAPLACE_QUALIFICATION_RUNTIME)
// Gate A/Gate B tests may intentionally exercise an unpromoted rule. Production
// callers and the CLI must use create_runtime_session.
SessionCreateResult create_qualification_runtime_session(std::shared_ptr<const RuntimePackage> package, SessionRequest request,
                                                         SessionFaultPoint fault = SessionFaultPoint::None);
#endif
uint32_t session_live_resource_count();

} // namespace Laplace
