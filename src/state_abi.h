#pragma once

#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include "semantic_model.h"

namespace Laplace {

struct StateCursor {
    uint64_t store_id = 0;
    uint64_t generation = 0;
    uint64_t accepted_tokens = 0;
    uint64_t undo_count = 0;
    friend bool operator==(const StateCursor&, const StateCursor&) = default;
};

struct StateDigestChunk {
    uint64_t offset = 0;
    uint64_t length = 0;
};

// This bounded seam uses the same chunk schedule as the state ABI writer and
// lets tests cover 32-bit boundaries without allocating multi-gigabyte slots.
std::vector<StateDigestChunk> state_digest_chunk_ranges_for_testing(uint64_t length);
Sha256Digest state_digest_for_testing(std::span<const uint8_t> bytes);

using StateMutationResult = std::variant<std::monostate, CompatibilityReport>;
using StateSaveResult = std::variant<std::vector<uint8_t>, CompatibilityReport>;

Sha256Digest token_contract_digest(const SemanticModel& model);

class StateStore;
using StateRestoreResult = std::variant<StateStore, CompatibilityReport>;

class StateStore {
public:
    explicit StateStore(const SemanticModel& model);
    StateStore(StateStore&&) noexcept = default;
    StateStore& operator=(StateStore&&) noexcept = default;
    StateStore(const StateStore&) = delete;
    StateStore& operator=(const StateStore&) = delete;

    bool valid() const noexcept { return valid_; }
    uint64_t accepted_tokens() const noexcept { return token_history_.size(); }
    uint64_t full_copy_count() const noexcept { return 0; }
    const std::vector<uint32_t>& token_history() const noexcept { return token_history_; }
    const std::vector<SemanticKvState>& slots() const noexcept { return slots_; }

    // Opens one undo-protected append. The returned slots are valid only until
    // finish_execution_append(), rollback(), commit(), reset(), or restore().
    StateMutationResult begin_execution_append(std::span<const uint32_t> token_ids);
    std::vector<SemanticKvState>* execution_slots() noexcept { return append_active_ ? &slots_ : nullptr; }
    StateMutationResult finish_execution_append();
    StateMutationResult append(std::span<const uint32_t> token_ids,
                               const std::vector<SemanticKvState>& additions);
    StateMutationResult trim(uint64_t accepted_tokens);
    StateCursor checkpoint() const noexcept;
    StateMutationResult rollback(StateCursor cursor);
    StateMutationResult commit(StateCursor cursor);
    void reset() noexcept;

    StateSaveResult save(const Sha256Digest& model_fingerprint,
                         const Sha256Digest& semantic_digest,
                         const Sha256Digest& token_contract_digest) const;
    static StateRestoreResult restore(const SemanticModel& model,
                                      const Sha256Digest& model_fingerprint,
                                      const Sha256Digest& semantic_digest,
                                      const Sha256Digest& token_contract_digest,
                                      std::span<const uint8_t> bytes);

private:
    struct SlotDefinition {
        SemanticState key;
        SemanticState value;
        uint32_t heads = 0;
        uint32_t dimension = 0;
    };
    struct UndoEntry {
        uint64_t tokens = 0;
        uint64_t history_size = 0;
    };

    static std::variant<std::vector<SlotDefinition>, CompatibilityReport>
    definitions_from(const SemanticModel& model);
    StateMutationResult set_lengths(uint64_t accepted_tokens, uint64_t history_size);

    std::vector<SlotDefinition> definitions_;
    uint32_t maximum_context_ = 0;
    bool valid_ = false;
    std::vector<SemanticKvState> slots_;
    std::vector<uint32_t> token_history_;
    std::vector<UndoEntry> undo_;
    uint64_t undo_base_ = 0;
    uint64_t store_id_ = 0;
    uint64_t generation_ = 1;
    bool append_active_ = false;
};

} // namespace Laplace
