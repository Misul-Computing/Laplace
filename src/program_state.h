#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include "program_ir.h"

namespace Laplace {

struct StateDimensionBinding {
    uint32_t parameter_id = UINT32_MAX;
    uint64_t value = 0;
    friend bool operator==(const StateDimensionBinding&,
                           const StateDimensionBinding&) = default;
};

struct StateSlotSchema {
    uint32_t state_reference_id = UINT32_MAX;
    ElementType element_type = ElementType::F32;
    std::vector<uint64_t> extents;
    uint32_t alias_cell = UINT32_MAX;
};

struct StateSchema {
    uint16_t major = 1;
    uint16_t minor = 0;
    std::vector<StateDimensionBinding> dimension_bindings;
    std::vector<StateSlotSchema> slots;
};

struct StateSchemaDigest {
    std::array<uint8_t, 32> bytes{};
    friend bool operator==(StateSchemaDigest, StateSchemaDigest) = default;
};

struct VerifiedStateSlot {
    uint32_t state_reference_id = UINT32_MAX;
    uint32_t canonical_state_ordinal = UINT32_MAX;
    uint32_t canonical_cell_ordinal = UINT32_MAX;
    ElementType element_type = ElementType::F32;
    std::vector<uint64_t> extents;
    uint64_t byte_count = 0;
    bool writable = false;
};

struct VerifiedStateCell {
    uint32_t canonical_ordinal = UINT32_MAX;
    uint64_t byte_count = 0;
    bool writable = false;
    std::vector<uint32_t> slot_indices;
};

class VerifiedStateSchema {
public:
    VerifiedStateSchema(const VerifiedStateSchema&) = default;
    VerifiedStateSchema(VerifiedStateSchema&&) noexcept = default;
    VerifiedStateSchema& operator=(const VerifiedStateSchema&) = default;
    VerifiedStateSchema& operator=(VerifiedStateSchema&&) noexcept = default;

    const std::vector<VerifiedStateSlot>& slots() const noexcept { return slots_; }
    const std::vector<VerifiedStateCell>& cells() const noexcept { return cells_; }

private:
    ProgramDigest program_digest_;
    StateSchemaDigest digest_;
    StateSchema definition_;
    std::vector<VerifiedStateSlot> slots_;
    std::vector<VerifiedStateCell> cells_;

    VerifiedStateSchema(ProgramDigest program_digest, StateSchemaDigest digest,
                        StateSchema definition,
                        std::vector<VerifiedStateSlot> slots,
                        std::vector<VerifiedStateCell> cells);
    friend std::variant<VerifiedStateSchema, CompatibilityReport>
    verify_state_schema(StateSchema, const VerifiedProgram&);
    friend ProgramDigest state_program_digest(const VerifiedStateSchema&);
    friend StateSchemaDigest state_schema_digest(const VerifiedStateSchema&);
    friend std::variant<std::vector<uint8_t>, CompatibilityReport>
    encode_state_schema_wire(const VerifiedStateSchema&);
    friend class CandidateGeneration;
    friend class StateRoot;
    friend std::variant<class StateRoot, CompatibilityReport>
    make_state_root(const VerifiedStateSchema&,
                    std::vector<struct StateCellValue>);
    friend std::variant<class StateRoot, CompatibilityReport>
    restore_state_root(const VerifiedStateSchema&, std::span<const uint8_t>);
};

using StateSchemaVerificationResult =
    std::variant<VerifiedStateSchema, CompatibilityReport>;
using StateSchemaWireResult =
    std::variant<std::vector<uint8_t>, CompatibilityReport>;

StateSchemaVerificationResult
verify_state_schema(StateSchema schema, const VerifiedProgram& program);
ProgramDigest state_program_digest(const VerifiedStateSchema& schema);
StateSchemaDigest state_schema_digest(const VerifiedStateSchema& schema);
StateSchemaWireResult
encode_state_schema_wire(const VerifiedStateSchema& schema);
StateSchemaVerificationResult
decode_state_schema_wire(std::span<const uint8_t> wire,
                         const VerifiedProgram& program);

struct StateCellValue {
    // Any state reference in the cell may identify its one initial payload.
    uint32_t state_reference_id = UINT32_MAX;
    std::vector<uint8_t> bytes;
};

using ProgramStateMutationResult =
    std::variant<std::monostate, CompatibilityReport>;
using ProgramStateSaveResult =
    std::variant<std::vector<uint8_t>, CompatibilityReport>;

class StateReadView {
public:
    StateReadView() = default;
    bool valid() const noexcept { return static_cast<bool>(storage_); }
    bool empty() const noexcept { return !storage_ || storage_->empty(); }
    size_t size() const noexcept { return storage_ ? storage_->size() : 0; }
    const uint8_t* data() const noexcept {
        static const uint8_t empty_storage = 0;
        return storage_ ? (storage_->empty() ? &empty_storage : storage_->data())
                        : nullptr;
    }
    const uint8_t* begin() const noexcept { return data(); }
    const uint8_t* end() const noexcept {
        return storage_ ? data() + size() : nullptr;
    }
    std::span<const uint8_t> bytes() const noexcept {
        return storage_ ? std::span<const uint8_t>(data(), size())
                        : std::span<const uint8_t>();
    }
    operator std::span<const uint8_t>() const noexcept { return bytes(); }

private:
    std::shared_ptr<const std::vector<uint8_t>> storage_;

    explicit StateReadView(
        std::shared_ptr<const std::vector<uint8_t>> storage) noexcept
        : storage_(std::move(storage)) {}
    friend class CandidateGeneration;
    friend class StateRoot;
};

enum class ExecutionDisposition : uint8_t {
    RejectedBeforeSubmission = 0,
    Completed = 1,
    Failed = 2,
    Indeterminate = 3,
};

class ProgramStateExecutionAuthority final {
public:
    ProgramStateExecutionAuthority() = delete;
};
struct ProgramStateCandidateStorage;
struct ProgramStateRootStatus;

class ExecutionResult {
public:
    ExecutionResult(ExecutionResult&& other) noexcept;
    ExecutionResult& operator=(ExecutionResult&&) noexcept = delete;
    ExecutionResult(const ExecutionResult&) = delete;
    ExecutionResult& operator=(const ExecutionResult&) = delete;

    bool valid() const noexcept { return valid_ && !consumed_; }
    ExecutionDisposition disposition() const noexcept { return disposition_; }

private:
    uint64_t root_id_ = 0;
    uint64_t candidate_id_ = 0;
    uint64_t base_generation_ = 0;
    uint64_t effect_boundaries_ = 0;
    StateSchemaDigest schema_digest_;
    ExecutionDisposition disposition_ =
        ExecutionDisposition::RejectedBeforeSubmission;
    bool valid_ = false;
    bool consumed_ = false;

    ExecutionResult(uint64_t root_id, uint64_t candidate_id,
                    uint64_t base_generation, uint64_t effect_boundaries,
                    StateSchemaDigest schema_digest,
                    ExecutionDisposition disposition, bool valid) noexcept;
    friend class ProgramStateExecutionAuthority;
    friend class StateRoot;
#ifdef LAPLACE_TESTING
    friend ExecutionResult program_state_execution_result_for_testing(
        const class CandidateGeneration&, ExecutionDisposition);
#endif
};

class CandidateGeneration {
public:
    ~CandidateGeneration();
    CandidateGeneration(CandidateGeneration&& other) noexcept;
    CandidateGeneration& operator=(CandidateGeneration&&) noexcept = delete;
    CandidateGeneration(const CandidateGeneration&) = delete;
    CandidateGeneration& operator=(const CandidateGeneration&) = delete;

    bool valid() const noexcept;
    uint64_t base_generation() const noexcept;
    uint64_t effect_boundaries() const noexcept;
    StateReadView read_slot(uint32_t state_reference_id) const noexcept;
    ProgramStateMutationResult write_slot(uint32_t state_reference_id,
                                          std::span<const uint8_t> bytes);

private:
    std::shared_ptr<ProgramStateCandidateStorage> state_;

    explicit CandidateGeneration(
        std::shared_ptr<ProgramStateCandidateStorage> state) noexcept;
    size_t slot_index(uint32_t state_reference_id) const noexcept;
    friend class StateRoot;
#ifdef LAPLACE_TESTING
    friend ExecutionResult program_state_execution_result_for_testing(
        const CandidateGeneration&, ExecutionDisposition);
#endif
};

class StateRoot {
public:
    StateRoot(StateRoot&& other) noexcept;
    StateRoot& operator=(StateRoot&&) noexcept = delete;
    StateRoot(const StateRoot&) = delete;
    StateRoot& operator=(const StateRoot&) = delete;

    bool valid() const noexcept { return valid_; }
    bool poisoned() const noexcept;
    bool candidate_active() const noexcept;
    uint64_t generation() const noexcept { return generation_; }
    StateReadView current_slot(uint32_t state_reference_id) const noexcept;
    std::variant<CandidateGeneration, CompatibilityReport> begin_candidate();

    ProgramStateMutationResult resolve_candidate(CandidateGeneration&& candidate,
                                                  ExecutionResult&& result);

    ProgramStateSaveResult save() const;

private:
    VerifiedStateSchema schema_;
    std::vector<std::shared_ptr<const std::vector<uint8_t>>> current_;
    std::shared_ptr<ProgramStateRootStatus> status_;
    std::shared_ptr<ProgramStateCandidateStorage> active_candidate_;
    std::shared_ptr<ProgramStateCandidateStorage> quarantined_candidate_;
    uint64_t root_id_ = 0;
    uint64_t generation_ = 0;
    uint64_t next_candidate_id_ = 1;
    bool valid_ = false;

    StateRoot(VerifiedStateSchema schema,
              std::vector<std::shared_ptr<const std::vector<uint8_t>>> current,
              uint64_t generation);
    size_t slot_index(uint32_t state_reference_id) const noexcept;
    bool matches(const CandidateGeneration& candidate) const noexcept;
    bool matches(const ExecutionResult& result,
                 const ProgramStateCandidateStorage& candidate) const noexcept;
    void consume(CandidateGeneration& candidate,
                 ExecutionResult& result) noexcept;
    friend std::variant<StateRoot, CompatibilityReport>
    make_state_root(const VerifiedStateSchema&, std::vector<StateCellValue>);
    friend std::variant<StateRoot, CompatibilityReport>
    restore_state_root(const VerifiedStateSchema&, std::span<const uint8_t>);
#ifdef LAPLACE_TESTING
    friend bool program_state_has_retained_candidate_for_testing(
        const StateRoot& root);
#endif
};

using StateRootResult = std::variant<StateRoot, CompatibilityReport>;

StateRootResult make_state_root(const VerifiedStateSchema& schema,
                                std::vector<StateCellValue> cells);
StateRootResult restore_state_root(const VerifiedStateSchema& schema,
                                   std::span<const uint8_t> bytes);

#ifdef LAPLACE_TESTING
struct ProgramStateDigestChunk {
    uint64_t offset = 0;
    uint64_t length = 0;
};

std::vector<ProgramStateDigestChunk>
program_state_digest_chunks_for_testing(uint64_t length);
bool program_state_wire_size_for_testing(
    std::span<const uint32_t> slot_ranks,
    std::span<const uint64_t> cell_byte_counts, uint64_t* wire_bytes);
ExecutionResult program_state_execution_result_for_testing(
    const CandidateGeneration& candidate, ExecutionDisposition disposition);
bool program_state_has_retained_candidate_for_testing(const StateRoot& root);
#endif

} // namespace Laplace
