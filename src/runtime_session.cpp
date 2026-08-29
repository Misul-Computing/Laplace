#include "runtime_session.h"

#include "product_package.h"

#include <algorithm>
#include <atomic>
#include <string>

#if defined(LAPLACE_QUALIFICATION_RUNTIME)
#include "scalar_executor.h"
#endif

namespace Laplace {

namespace {

std::atomic<uint64_t> g_next_product_store_id{1};

CompatibilityReport session_error(CompatibilityError code) {
    CompatibilityReport report = package_report(code);
    report.stage = CompatibilityStage::Session;
    return report;
}

CompatibilityReport session_error(CompatibilityError code, std::string detail) {
    CompatibilityReport report = session_error(code);
    report.detail = std::move(detail);
    return report;
}

#if defined(LAPLACE_QUALIFICATION_RUNTIME)
SessionResourceFault resource_fault(SessionFaultPoint fault) {
    switch (fault) {
    case SessionFaultPoint::DeviceQuery: return SessionResourceFault::DeviceQuery;
    case SessionFaultPoint::CommandQueue: return SessionResourceFault::CommandQueue;
    default: return SessionResourceFault::None;
    }
}
#endif

#if defined(LAPLACE_QUALIFICATION_RUNTIME)
RuntimeRunResult output_from(ScalarExecutionResult result, std::vector<uint32_t> history) {
    if (auto* report = std::get_if<CompatibilityReport>(&result)) return *report;
    ScalarExecutionOutput& scalar = std::get<ScalarExecutionOutput>(result);
    RuntimeOutput output;
    output.logits = std::move(scalar.logits);
    output.checkpoints = std::move(scalar.operator_outputs);
    output.states = std::move(scalar.states);
    output.token_history = std::move(history);
    output.operator_token_work = scalar.operator_token_work;
    return output;
}
#endif

RuntimeOutput output_from(CanonicalMetalOutput output, std::vector<uint32_t> history) {
    RuntimeOutput result;
    result.logits = std::move(output.logits);
    result.sampled_token_id = output.sampled_token_id;
    result.sampled_logit = output.sampled_logit;
    result.host_result_bytes = output.host_result_bytes;
    result.sampled = output.sampled;
    result.token_history = std::move(history);
    result.command_buffers = output.command_buffers;
    result.operator_count = output.operator_count;
    result.cpu_wait_ms = output.cpu_wait_ms;
    result.gpu_time_ms = output.gpu_time_ms;
    result.peak_session_bytes = output.peak_session_bytes;
    result.requested_projection_source_bytes = output.requested_projection_source_bytes;
    result.projection_dispatches = output.projection_dispatches;
    result.grouped_affine_u2_projection_dispatches =
        output.grouped_affine_u2_projection_dispatches;
    result.column_grouped_affine_u2_skip_projection_dispatches =
        output.column_grouped_affine_u2_skip_projection_dispatches;
    result.completed = output.completed;
    result.operator_token_work = output.operator_count;
    return result;
}

} // namespace

SessionCreateResult create_product_runtime_session_internal(
    std::shared_ptr<const RuntimePackage> package, SessionRequest request,
    SessionFaultPoint fault) {
    if (!package) return session_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED);
    if (fault != SessionFaultPoint::None) return session_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED);
    if (request.minimum_class != NumericalClass::ExactFp32 ||
        (request.objective != RuntimeObjective::Latency && request.objective != RuntimeObjective::Throughput)) {
        return session_error(CompatibilityError::RUNTIME_INPUT_INVALID);
    }
    CanonicalMetalCreateResult created = create_canonical_metal_program(package, request);
    if (const auto* report = std::get_if<CompatibilityReport>(&created)) return *report;
    auto metal = std::make_unique<CanonicalMetalProgram>(std::get<CanonicalMetalProgram>(std::move(created)));
    if (metal->plan().entries.empty()) return session_error(CompatibilityError::KERNEL_UNAVAILABLE);
    RuntimeSession result(std::move(package), metal->plan(), SessionResources{}, StateStore(SemanticModel{}),
                          std::move(metal), true, request);
    result.product_store_id_ = g_next_product_store_id.fetch_add(1, std::memory_order_relaxed);
    return result;
}

SessionCreateResult create_runtime_session_internal(const ProductPackage& product,
                                                     SessionRequest request,
                                                     SessionFaultPoint fault) {
    return create_product_runtime_session_internal(
        product.runtime_package(), request, fault);
}

#if defined(LAPLACE_QUALIFICATION_RUNTIME)
SessionCreateResult create_runtime_session_internal(std::shared_ptr<const RuntimePackage> package, SessionRequest request,
                                                    SessionFaultPoint fault) {
    if (!package) return session_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED);
    if (fault == SessionFaultPoint::Plan || fault == SessionFaultPoint::StateAllocation) {
        return session_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED);
    }

    SessionResourcesCandidate candidate(resource_fault(fault));
    candidate.query_optional_metal();
    RuntimeCapabilities capabilities = candidate.capabilities();
    capabilities.scalar_fp32 = true;
    capabilities.global_fp32_kv = true;
    capabilities.transactional_state = true;
    const PlanResult planned = plan_session(package->semantics(), request, capabilities, builtin_cpu_registry());
    if (const auto* report = std::get_if<CompatibilityReport>(&planned)) return *report;
    if (fault == SessionFaultPoint::StateAllocation) return session_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED);
    StateStore state(package->semantics());
    if (!state.valid()) return session_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED);
    return RuntimeSession(std::move(package), std::get<ExecutionPlan>(planned), candidate.finish_cpu_plan(), std::move(state));
}
#endif

SessionCreateResult create_runtime_session(const ProductPackage& package, SessionRequest request,
                                           SessionFaultPoint fault) {
    return create_runtime_session_internal(package, request, fault);
}

#if defined(LAPLACE_METAL_TESTING)
SessionCreateResult create_product_runtime_session_for_testing(
    std::shared_ptr<const RuntimePackage> package, SessionRequest request,
    SessionFaultPoint fault) {
    return create_product_runtime_session_internal(
        std::move(package), request, fault);
}
#endif

#if defined(LAPLACE_QUALIFICATION_RUNTIME)
SessionCreateResult create_qualification_runtime_session(std::shared_ptr<const RuntimePackage> package, SessionRequest request,
                                                         SessionFaultPoint fault) {
    return create_runtime_session_internal(std::move(package), request, fault);
}
#endif

uint32_t session_live_resource_count() {
    return session_resources_live_count();
}

RuntimeSession::RuntimeSession(RuntimeSession&& other) noexcept
    : package_(std::move(other.package_)), request_(other.request_), plan_(std::move(other.plan_)),
      resources_(std::move(other.resources_)), state_(std::move(other.state_)), metal_(std::move(other.metal_)),
      product_route_(other.product_route_), poisoned_(other.poisoned_), product_store_id_(other.product_store_id_),
      product_generation_(other.product_generation_), product_history_(std::move(other.product_history_))
#if defined(LAPLACE_METAL_TESTING)
      , product_prefill_failure_after_(other.product_prefill_failure_after_)
#endif
{
    other.clear_after_move();
}

RuntimeSession& RuntimeSession::operator=(RuntimeSession&& other) noexcept {
    if (this == &other) return *this;
    package_ = std::move(other.package_);
    request_ = other.request_;
    plan_ = std::move(other.plan_);
    resources_ = std::move(other.resources_);
    state_ = std::move(other.state_);
    metal_ = std::move(other.metal_);
    product_route_ = other.product_route_;
    poisoned_ = other.poisoned_;
    product_store_id_ = other.product_store_id_;
    product_generation_ = other.product_generation_;
    product_history_ = std::move(other.product_history_);
#if defined(LAPLACE_METAL_TESTING)
    product_prefill_failure_after_ = other.product_prefill_failure_after_;
#endif
    other.clear_after_move();
    return *this;
}

void RuntimeSession::clear_after_move() noexcept {
    package_.reset();
    request_ = {};
    plan_ = {};
    metal_.reset();
    product_route_ = false;
    poisoned_ = true;
    product_store_id_ = 0;
    product_generation_ = 0;
    product_history_.clear();
#if defined(LAPLACE_METAL_TESTING)
    product_prefill_failure_after_ = UINT32_MAX;
#endif
}

RuntimeRunResult RuntimeSession::execute(std::span<const uint32_t> token_ids,
                                         ExecutionPhase phase, bool sampled) {
    if (!package_) return session_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED);
    if (product_route_) {
        if (poisoned_ || !metal_ || token_ids.empty() ||
            token_ids.size() > package_->semantics().maximum_context ||
            product_history_.size() > package_->semantics().maximum_context - token_ids.size()) {
            return session_error(poisoned_ ? CompatibilityError::SESSION_CONSTRUCTION_FAILED
                                           : CompatibilityError::RUNTIME_INPUT_INVALID);
        }
        if (std::any_of(token_ids.begin(), token_ids.end(), [&](uint32_t token) {
                return token >= package_->semantics().vocabulary_size;
            })) {
            return session_error(CompatibilityError::RUNTIME_INPUT_INVALID);
        }
        if (metal_->position() != product_history_.size()) {
            poisoned_ = true;
            return session_error(CompatibilityError::STATE_ABI_MISMATCH);
        }
        if (phase == ExecutionPhase::Prefill && metal_->has_recurrent_layers() && token_ids.size() > 1) {
            CanonicalMetalOutput final_output;
            uint32_t command_buffers = 0;
            uint32_t projection_dispatches = 0;
            uint32_t batched_projection_dispatches = 0;
            uint32_t q4k_projection_dispatches = 0;
            uint32_t q6k_projection_dispatches = 0;
            uint32_t grouped_affine_u2_projection_dispatches = 0;
            uint32_t column_grouped_affine_u2_skip_projection_dispatches = 0;
            uint64_t counter_sample_count = 0;
            double cpu_wait_ms = 0.0;
            double gpu_time_ms = 0.0;
            double qkv_gpu_ms = 0.0;
            double attention_gpu_ms = 0.0;
            double ffn_gpu_ms = 0.0;
            double final_gpu_ms = 0.0;
            uint64_t peak_session_bytes = 0;
            uint64_t requested_projection_source_bytes = 0;
            bool profiled = false;
            bool counter_samples = false;
            bool split_command_buffer_profile = false;

            for (size_t index = 0; index != token_ids.size(); ++index) {
#if defined(LAPLACE_METAL_TESTING)
                if (product_prefill_failure_after_ == index) {
                    canonical_metal_fail_after_completed_submission_for_testing(*metal_);
                    product_prefill_failure_after_ = UINT32_MAX;
                }
#endif
                if (metal_->position() != product_history_.size() + index) {
                    poisoned_ = true;
                    return session_error(CompatibilityError::STATE_ABI_MISMATCH);
                }
                const uint32_t token = token_ids[index];
                const bool final_token = index + 1 == token_ids.size();
                CanonicalMetalRunResult chained = final_token
                    ? (sampled
                           ? metal_->prefill_sampled(
                                 std::span<const uint32_t>(&token, 1))
                           : metal_->prefill(
                                 std::span<const uint32_t>(&token, 1)))
                    : metal_->advance_prefill(token);
                if (const auto* report = std::get_if<CompatibilityReport>(&chained)) {
                    const bool restored = restore_product_prefix();
                    if (!restored) poisoned_ = true;
                    return *report;
                }
                CanonicalMetalOutput sample = std::get<CanonicalMetalOutput>(std::move(chained));
                if (!sample.completed) {
                    const bool restored = restore_product_prefix();
                    if (!restored) poisoned_ = true;
                    return session_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE);
                }
                command_buffers += sample.command_buffers;
                projection_dispatches += sample.projection_dispatches;
                batched_projection_dispatches += sample.batched_projection_dispatches;
                q4k_projection_dispatches += sample.q4k_projection_dispatches;
                q6k_projection_dispatches += sample.q6k_projection_dispatches;
                grouped_affine_u2_projection_dispatches += sample.grouped_affine_u2_projection_dispatches;
                column_grouped_affine_u2_skip_projection_dispatches +=
                    sample.column_grouped_affine_u2_skip_projection_dispatches;
                counter_sample_count += sample.counter_sample_count;
                cpu_wait_ms += sample.cpu_wait_ms;
                gpu_time_ms += sample.gpu_time_ms;
                qkv_gpu_ms += sample.qkv_gpu_ms;
                attention_gpu_ms += sample.attention_gpu_ms;
                ffn_gpu_ms += sample.ffn_gpu_ms;
                final_gpu_ms += sample.final_gpu_ms;
                peak_session_bytes = std::max(peak_session_bytes, sample.peak_session_bytes);
                requested_projection_source_bytes += sample.requested_projection_source_bytes;
                profiled = profiled || sample.profiled;
                counter_samples = counter_samples || sample.counter_samples;
                split_command_buffer_profile = split_command_buffer_profile || sample.split_command_buffer_profile;
                if (index + 1 == token_ids.size()) final_output = std::move(sample);
            }
            product_history_.insert(product_history_.end(), token_ids.begin(), token_ids.end());
            final_output.command_buffers = command_buffers;
            final_output.projection_dispatches = projection_dispatches;
            final_output.batched_projection_dispatches = batched_projection_dispatches;
            final_output.q4k_projection_dispatches = q4k_projection_dispatches;
            final_output.q6k_projection_dispatches = q6k_projection_dispatches;
            final_output.grouped_affine_u2_projection_dispatches = grouped_affine_u2_projection_dispatches;
            final_output.column_grouped_affine_u2_skip_projection_dispatches =
                column_grouped_affine_u2_skip_projection_dispatches;
            final_output.counter_sample_count = counter_sample_count;
            final_output.cpu_wait_ms = cpu_wait_ms;
            final_output.gpu_time_ms = gpu_time_ms;
            final_output.qkv_gpu_ms = qkv_gpu_ms;
            final_output.attention_gpu_ms = attention_gpu_ms;
            final_output.ffn_gpu_ms = ffn_gpu_ms;
            final_output.final_gpu_ms = final_gpu_ms;
            final_output.peak_session_bytes = peak_session_bytes;
            final_output.requested_projection_source_bytes = requested_projection_source_bytes;
            final_output.profiled = profiled;
            final_output.counter_samples = counter_samples;
            final_output.split_command_buffer_profile = split_command_buffer_profile;
            return output_from(std::move(final_output), product_history_);
        }
        CanonicalMetalRunResult result = phase == ExecutionPhase::Prefill
            ? (sampled ? metal_->prefill_sampled(token_ids) : metal_->prefill(token_ids))
            : (token_ids.size() == 1
                   ? (sampled ? metal_->decode_sampled(token_ids.front())
                              : metal_->decode(token_ids.front()))
                   : CanonicalMetalRunResult{
                         session_error(CompatibilityError::RUNTIME_INPUT_INVALID)});
        if (const auto* report = std::get_if<CompatibilityReport>(&result)) {
            poisoned_ = true;
            return *report;
        }
        CanonicalMetalOutput output = std::get<CanonicalMetalOutput>(std::move(result));
        if (!output.completed) {
            poisoned_ = true;
            return session_error(CompatibilityError::RUNTIME_NUMERICAL_FAILURE);
        }
        product_history_.insert(product_history_.end(), token_ids.begin(), token_ids.end());
        return output_from(std::move(output), product_history_);
    }
#if defined(LAPLACE_QUALIFICATION_RUNTIME)
    if (sampled) {
        return session_error(CompatibilityError::KERNEL_UNAVAILABLE,
                             "device sampling is available only on the product Metal runtime");
    }
    if (token_ids.empty() || !package_) return session_error(CompatibilityError::RUNTIME_INPUT_INVALID);
    const bool planned = std::any_of(plan_.entries.begin(), plan_.entries.end(), [&](const PlanEntry& entry) {
        return entry.phase == phase;
    });
    if (!planned) return session_error(CompatibilityError::KERNEL_UNAVAILABLE);
    if (token_ids.size() > package_->semantics().maximum_context ||
        state_.token_history().size() > package_->semantics().maximum_context - token_ids.size()) {
        CompatibilityReport report = session_error(CompatibilityError::PLAN_CONTEXT_EXCEEDED);
        report.stage = CompatibilityStage::Plan;
        return report;
    }
    const StateCursor cursor = state_.checkpoint();
    StateMutationResult opened = state_.begin_execution_append(token_ids);
    if (const auto* report = std::get_if<CompatibilityReport>(&opened)) return *report;
    std::vector<SemanticKvState>* slots = state_.execution_slots();
    if (!slots) return session_error(CompatibilityError::IR_STATE_INVALID);
    ScalarExecutionResult scalar = execute_planned_scalar_incremental(*package_, plan_, phase, token_ids, *slots);
    if (const auto* report = std::get_if<CompatibilityReport>(&scalar)) {
        StateMutationResult undone = state_.rollback(cursor);
        if (const auto* undo_report = std::get_if<CompatibilityReport>(&undone)) return *undo_report;
        return *report;
    }
    StateMutationResult finished = state_.finish_execution_append();
    if (const auto* report = std::get_if<CompatibilityReport>(&finished)) {
        StateMutationResult undone = state_.rollback(cursor);
        if (const auto* undo_report = std::get_if<CompatibilityReport>(&undone)) return *undo_report;
        return *report;
    }
    return output_from(std::move(scalar), state_.token_history());
#else
    (void)token_ids;
    (void)phase;
    return session_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                         "qualification scalar runtime is not linked into this product binary");
#endif
}

bool RuntimeSession::restore_product_prefix() {
    if (!product_route_ || !package_ || !metal_) return false;
    metal_.reset();
    CanonicalMetalCreateResult created = create_canonical_metal_program(package_, request_);
    if (std::holds_alternative<CompatibilityReport>(created)) return false;
    auto replacement = std::make_unique<CanonicalMetalProgram>(
        std::get<CanonicalMetalProgram>(std::move(created)));
    const auto replay = [&](std::span<const uint32_t> tokens) {
        for (const uint32_t token : tokens) {
            CanonicalMetalRunResult replay = replacement->prefill(
                std::span<const uint32_t>(&token, 1));
            if (std::holds_alternative<CompatibilityReport>(replay) ||
                !std::get<CanonicalMetalOutput>(std::move(replay)).completed) return false;
        }
        return true;
    };
    if (!replay(product_history_)) return false;
    metal_ = std::move(replacement);
    return true;
}

RuntimeRunResult RuntimeSession::prefill(std::span<const uint32_t> token_ids) {
    return execute(token_ids, ExecutionPhase::Prefill, false);
}

RuntimeRunResult RuntimeSession::decode(uint32_t token_id) {
    return execute(std::span<const uint32_t>(&token_id, 1), ExecutionPhase::Decode, false);
}

RuntimeRunResult RuntimeSession::prefill_sampled(std::span<const uint32_t> token_ids) {
    return execute(token_ids, ExecutionPhase::Prefill, true);
}

RuntimeRunResult RuntimeSession::decode_sampled(uint32_t token_id) {
    return execute(std::span<const uint32_t>(&token_id, 1), ExecutionPhase::Decode, true);
}

StateCursor RuntimeSession::checkpoint() const noexcept {
    if (!package_) return {};
    if (!product_route_) return state_.checkpoint();
    return {product_store_id_, product_generation_, static_cast<uint64_t>(product_history_.size()), 0};
}

StateMutationResult RuntimeSession::commit(StateCursor cursor) {
    if (!package_) return session_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED);
    if (!product_route_) return state_.commit(cursor);
    if (poisoned_ || cursor.store_id != product_store_id_ || cursor.generation != product_generation_ ||
        cursor.undo_count != 0 ||
        cursor.accepted_tokens > product_history_.size()) {
        return session_error(CompatibilityError::STATE_ABI_MISMATCH);
    }
    return std::monostate{};
}

StateMutationResult RuntimeSession::rollback(StateCursor cursor) {
    if (!package_) return session_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED);
    if (!product_route_) return state_.rollback(cursor);
    if (poisoned_ || !metal_ || cursor.store_id != product_store_id_ || cursor.generation != product_generation_ ||
        cursor.undo_count != 0 || cursor.accepted_tokens > UINT32_MAX ||
        cursor.accepted_tokens > product_history_.size()) {
        return session_error(CompatibilityError::STATE_ABI_MISMATCH);
    }
    if (metal_->has_recurrent_layers()) {
        return session_error(CompatibilityError::CACHE_MODE_UNQUALIFIED,
                             "product rollback requires a recurrent GPU state history");
    }
    if (!metal_->rollback_to_position(static_cast<uint32_t>(cursor.accepted_tokens))) {
        poisoned_ = true;
        return session_error(CompatibilityError::STATE_ABI_MISMATCH);
    }
    product_history_.resize(static_cast<size_t>(cursor.accepted_tokens));
    ++product_generation_;
    return std::monostate{};
}

const std::vector<uint32_t>& RuntimeSession::token_history() const noexcept {
    static const std::vector<uint32_t> empty;
    if (!package_) return empty;
    return product_route_ ? product_history_ : state_.token_history();
}

#if defined(LAPLACE_METAL_TESTING)
void RuntimeSession::fail_after_completed_submission_for_testing() noexcept {
    if (product_route_ && metal_) canonical_metal_fail_after_completed_submission_for_testing(*metal_);
}

void RuntimeSession::fail_after_prefill_tokens_for_testing(uint32_t count) noexcept {
    if (product_route_) product_prefill_failure_after_ = count;
}

uint64_t RuntimeSession::implicit_weight_copy_count_for_testing() const noexcept {
    return product_route_ && metal_ ? metal_->resource_diagnostics().implicit_weight_copies : UINT64_MAX;
}
#endif

StateSaveResult RuntimeSession::save_state() const {
    if (!package_) return session_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED);
    if (product_route_) {
        return session_error(CompatibilityError::CACHE_MODE_UNQUALIFIED,
                             "product Metal state save is not admitted by the current GPU state ABI");
    }
    return state_.save(package_->fingerprint(), semantic_model_digest(package_->semantics()), token_contract_digest(package_->semantics()));
}

StateMutationResult RuntimeSession::restore_state(std::span<const uint8_t> bytes) {
    if (!package_) return session_error(CompatibilityError::SESSION_CONSTRUCTION_FAILED);
    if (product_route_) {
        return session_error(CompatibilityError::CACHE_MODE_UNQUALIFIED,
                             "product Metal state restore is not admitted by the current GPU state ABI");
    }
    auto restored = StateStore::restore(package_->semantics(), package_->fingerprint(), semantic_model_digest(package_->semantics()),
                                        token_contract_digest(package_->semantics()), bytes);
    if (const auto* report = std::get_if<CompatibilityReport>(&restored)) return *report;
    state_ = std::get<StateStore>(std::move(restored));
    return std::monostate{};
}

} // namespace Laplace
