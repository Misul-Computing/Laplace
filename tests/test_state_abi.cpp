#include <cstring>
#include <cstdint>
#include <vector>

#include "state_abi.h"
#include "test_util.h"

using namespace Laplace;

namespace {

SemanticModel two_slot_model() {
    SemanticModel model;
    model.maximum_context = 8;
    StateFormat key_format;
    key_format.logical_domain = TransformDomain::Untransformed;
    key_format.encoded_domain = TransformDomain::RopeApplied;
    key_format.alignment = 64;
    StateFormat value_format = key_format;
    value_format.encoded_domain = TransformDomain::Untransformed;
    const std::vector<Dimension> dimensions = {
        {DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}, {DimensionKind::Constant, 4}};
    model.states.push_back({7, StateKind::KeyCache, 1, StateUpdateKind::AppendKey,
                            PositionPolicy::AppendOnly, dimensions, {key_format}, 0});
    model.states.push_back({8, StateKind::ValueCache, 1, StateUpdateKind::AppendValue,
                            PositionPolicy::AppendOnly, dimensions, {value_format}, 0});
    return model;
}

Sha256Digest digest(uint8_t value) {
    Sha256Digest result;
    result.bytes.fill(value);
    return result;
}

SemanticKvState kv(uint32_t tokens, float start) {
    SemanticKvState state;
    state.tokens = tokens;
    for (uint32_t index = 0; index != tokens * 8; ++index) {
        state.key.push_back(start + static_cast<float>(index));
        state.value.push_back(start + 100.0f + static_cast<float>(index));
    }
    return state;
}

uint16_t u16(const std::vector<uint8_t>& bytes, size_t offset) {
    return static_cast<uint16_t>(bytes[offset]) | (static_cast<uint16_t>(bytes[offset + 1]) << 8);
}

uint32_t u32(const std::vector<uint8_t>& bytes, size_t offset) {
    uint32_t value = 0;
    for (unsigned shift = 0; shift != 32; shift += 8) value |= static_cast<uint32_t>(bytes[offset + shift / 8]) << shift;
    return value;
}

uint64_t u64(const std::vector<uint8_t>& bytes, size_t offset) {
    uint64_t value = 0;
    for (unsigned shift = 0; shift != 64; shift += 8) value |= static_cast<uint64_t>(bytes[offset + shift / 8]) << shift;
    return value;
}

bool valid_file_digest(const std::vector<uint8_t>& bytes) {
    if (bytes.size() < 32) return false;
    const Sha256Digest digest = state_digest_for_testing(std::span<const uint8_t>(bytes.data(), bytes.size() - 32));
    return std::memcmp(digest.bytes.data(), bytes.data() + bytes.size() - 32, digest.bytes.size()) == 0;
}

void reseal(std::vector<uint8_t>& bytes) {
    const Sha256Digest digest = state_digest_for_testing(std::span<const uint8_t>(bytes.data(), bytes.size() - 32));
    std::memcpy(bytes.data() + bytes.size() - digest.bytes.size(), digest.bytes.data(), digest.bytes.size());
}

StateMutationResult append(StateStore& store, std::initializer_list<uint32_t> token_ids,
                           std::vector<SemanticKvState> additions) {
    const std::vector<uint32_t> ids(token_ids);
    return store.append(ids, additions);
}

void test_exact_save_restore() {
    const SemanticModel model = two_slot_model();
    StateStore store(model);
    CHECK(store.valid());
    CHECK(std::holds_alternative<std::monostate>(append(store, {11, 12}, {kv(2, 1.0f)})));
    auto saved = store.save(digest(1), digest(2), digest(3));
    CHECK(std::holds_alternative<std::vector<uint8_t>>(saved));
    if (!std::holds_alternative<std::vector<uint8_t>>(saved)) return;
    const std::vector<uint8_t>& bytes = std::get<std::vector<uint8_t>>(saved);
    CHECK_MSG(bytes.size() == 896, "actual=%zu", bytes.size());
    CHECK(std::memcmp(bytes.data(), "LAPST001", 8) == 0);
    CHECK(u16(bytes, 8) == 1 && u16(bytes, 10) == 0);
    CHECK_MSG(u32(bytes, 12) == 576 && u64(bytes, 16) == 576 && u64(bytes, 24) == 896,
              "header=%u payload=%llu total=%llu", u32(bytes, 12),
              static_cast<unsigned long long>(u64(bytes, 16)), static_cast<unsigned long long>(u64(bytes, 24)));
    CHECK(u64(bytes, 128) == 2 && u64(bytes, 136) == 2 && u64(bytes, 144) == 576 && u64(bytes, 152) == 8);
    CHECK(u32(bytes, 192) == 2 && u32(bytes, 196) == 0);
    CHECK(u32(bytes, 200) == 7 && u16(bytes, 216) == static_cast<uint16_t>(TransformDomain::Untransformed) &&
          u16(bytes, 218) == static_cast<uint16_t>(TransformDomain::RopeApplied));
    CHECK(u32(bytes, 368) == 8 && u16(bytes, 384) == static_cast<uint16_t>(TransformDomain::Untransformed) &&
          u16(bytes, 386) == static_cast<uint16_t>(TransformDomain::Untransformed));
    CHECK_MSG(u64(bytes, 320) == 640 && u64(bytes, 328) == 96 && u64(bytes, 488) == 768 && u64(bytes, 496) == 96,
              "first=%llu/%llu second=%llu/%llu", static_cast<unsigned long long>(u64(bytes, 320)),
              static_cast<unsigned long long>(u64(bytes, 328)), static_cast<unsigned long long>(u64(bytes, 488)),
              static_cast<unsigned long long>(u64(bytes, 496)));
    CHECK(u64(bytes, 640) == 2 && u64(bytes, 648) == 8 && u32(bytes, 656) == 2 && u32(bytes, 660) == 4 && u64(bytes, 664) == 64);
    CHECK_MSG(u64(bytes, 768) == 2 && u64(bytes, 776) == 8 && u32(bytes, 784) == 2 && u32(bytes, 788) == 4 && u64(bytes, 792) == 64,
              "second payload token=%llu capacity=%llu heads=%u dim=%u bytes=%llu",
              static_cast<unsigned long long>(u64(bytes, 768)), static_cast<unsigned long long>(u64(bytes, 776)),
              u32(bytes, 784), u32(bytes, 788), static_cast<unsigned long long>(u64(bytes, 792)));
    CHECK(valid_file_digest(bytes));
    CHECK(std::all_of(bytes.begin() + 536, bytes.begin() + 576, [](uint8_t value) { return value == 0; }));
    CHECK(std::all_of(bytes.begin() + 584, bytes.begin() + 640, [](uint8_t value) { return value == 0; }));

    auto restored = StateStore::restore(model, digest(1), digest(2), digest(3), bytes);
    CHECK(std::holds_alternative<StateStore>(restored));
    if (std::holds_alternative<StateStore>(restored)) {
        const StateStore& result = std::get<StateStore>(restored);
        CHECK(result.token_history() == std::vector<uint32_t>({11, 12}));
        CHECK(result.slots().size() == 1);
        CHECK(result.slots()[0].tokens == 2);
        CHECK(result.slots()[0].key == kv(2, 1.0f).key);
        CHECK(result.slots()[0].value == kv(2, 1.0f).value);
    }

    std::vector<uint8_t> corrupt = bytes;
    corrupt[196] = 1;
    auto rejected = StateStore::restore(model, digest(1), digest(2), digest(3), corrupt);
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    if (std::holds_alternative<CompatibilityReport>(rejected)) {
        CHECK(std::get<CompatibilityReport>(rejected).code == CompatibilityError::STATE_ABI_MISMATCH);
    }
    CHECK(store.token_history() == std::vector<uint32_t>({11, 12}));
}

void test_cursor_undo() {
    StateStore store(two_slot_model());
    CHECK(std::holds_alternative<std::monostate>(append(store, {1}, {kv(1, 1.0f)})));
    const StateCursor checkpoint = store.checkpoint();
    CHECK(std::holds_alternative<std::monostate>(append(store, {2}, {kv(1, 9.0f)})));
    CHECK(std::holds_alternative<std::monostate>(append(store, {3}, {kv(1, 17.0f)})));
    CHECK(store.token_history().size() == 3);
    CHECK(store.full_copy_count() == 0);
    CHECK(std::holds_alternative<std::monostate>(store.rollback(checkpoint)));
    CHECK(store.token_history() == std::vector<uint32_t>({1}));
    CHECK(store.slots()[0].tokens == 1);
    CHECK(store.slots()[0].key == kv(1, 1.0f).key);
    CHECK(store.full_copy_count() == 0);
}

void test_digest_boundaries_and_cursor_identity() {
    const uint64_t four_gib = uint64_t{1} << 32;
    const auto at_u32_max = state_digest_chunk_ranges_for_testing(UINT32_MAX);
    const auto at_four_gib = state_digest_chunk_ranges_for_testing(four_gib);
    const auto after_four_gib = state_digest_chunk_ranges_for_testing(four_gib + 1);
    CHECK(!at_u32_max.empty());
    CHECK(!at_four_gib.empty());
    CHECK(!after_four_gib.empty());
    CHECK(at_u32_max.back().offset + at_u32_max.back().length == UINT32_MAX);
    CHECK(at_four_gib.back().offset + at_four_gib.back().length == four_gib);
    CHECK(after_four_gib.back().offset + after_four_gib.back().length == four_gib + 1);
    CHECK(after_four_gib.size() > at_four_gib.size());

    std::array<uint8_t, 17> first{};
    std::array<uint8_t, 17> second{};
    second.back() = 1;
    CHECK(state_digest_for_testing(first) != state_digest_for_testing(second));

    StateStore left(two_slot_model());
    StateStore right(two_slot_model());
    CHECK(std::holds_alternative<std::monostate>(append(left, {1}, {kv(1, 1.0f)})));
    CHECK(std::holds_alternative<std::monostate>(append(right, {1}, {kv(1, 1.0f)})));
    const StateCursor first_checkpoint = left.checkpoint();
    const StateCursor foreign_checkpoint = right.checkpoint();
    CHECK(first_checkpoint.accepted_tokens == foreign_checkpoint.accepted_tokens);
    CHECK(first_checkpoint.undo_count == foreign_checkpoint.undo_count);
    CHECK(std::holds_alternative<CompatibilityReport>(left.rollback(foreign_checkpoint)));
    CHECK(std::holds_alternative<std::monostate>(append(left, {2}, {kv(1, 9.0f)})));
    CHECK(std::holds_alternative<std::monostate>(left.commit(first_checkpoint)));
    CHECK(left.token_history() == std::vector<uint32_t>({1, 2}));

    const StateCursor nested = left.checkpoint();
    CHECK(std::holds_alternative<std::monostate>(append(left, {3}, {kv(1, 17.0f)})));
    CHECK(std::holds_alternative<std::monostate>(left.rollback(nested)));
    left.reset();
    CHECK(std::holds_alternative<CompatibilityReport>(left.rollback(nested)));
}

void test_every_wire_region_rejects() {
    const SemanticModel model = two_slot_model();
    StateStore live(model);
    CHECK(std::holds_alternative<std::monostate>(append(live, {11, 12}, {kv(2, 1.0f)})));
    auto saved = live.save(digest(1), digest(2), digest(3));
    CHECK(std::holds_alternative<std::vector<uint8_t>>(saved));
    if (!std::holds_alternative<std::vector<uint8_t>>(saved)) return;
    const std::vector<uint8_t> baseline = std::get<std::vector<uint8_t>>(saved);
    const std::vector<size_t> offsets = {
        0, 8, 10, 12, 16, 24, 32, 64, 96, 128, 136, 144, 152, 160, 192, 196,
        200, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230,
        232, 240, 248, 296, 300, 304, 312, 320, 328, 336, 536, 576, 584,
        640, 648, 656, 660, 664, 672, 736, 768, 800};
    for (size_t offset : offsets) {
        std::vector<uint8_t> mutated = baseline;
        mutated[offset] ^= 1;
        reseal(mutated);
        auto restored = StateStore::restore(model, digest(1), digest(2), digest(3), mutated);
        CHECK_MSG(std::holds_alternative<CompatibilityReport>(restored), "offset=%zu", offset);
        if (std::holds_alternative<CompatibilityReport>(restored)) {
            CHECK(std::get<CompatibilityReport>(restored).code == CompatibilityError::STATE_ABI_MISMATCH);
        }
        auto current = live.save(digest(1), digest(2), digest(3));
        CHECK(std::holds_alternative<std::vector<uint8_t>>(current));
        if (std::holds_alternative<std::vector<uint8_t>>(current)) {
            CHECK(std::get<std::vector<uint8_t>>(current) == baseline);
        }
    }
    std::vector<uint8_t> trailing = baseline;
    trailing.back() ^= 1;
    auto trailing_result = StateStore::restore(model, digest(1), digest(2), digest(3), trailing);
    CHECK(std::holds_alternative<CompatibilityReport>(trailing_result));
}

} // namespace

int main() {
    test_exact_save_restore();
    test_cursor_undo();
    test_digest_boundaries_and_cursor_identity();
    test_every_wire_region_rejects();
    return test_summary("test_state_abi");
}
