#include "state_abi.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <atomic>

namespace Laplace {

namespace {

constexpr std::array<uint8_t, 8> kMagic = {'L', 'A', 'P', 'S', 'T', '0', '0', '1'};
constexpr uint16_t kMajor = 1;
constexpr uint16_t kMinor = 0;
constexpr uint64_t kHeaderBytes = 200;
constexpr uint64_t kDescriptorBytes = 168;
constexpr uint64_t kPayloadHeaderBytes = 32;
constexpr uint64_t kDigestBytes = 32;
constexpr uint64_t kAlignment = 64;
constexpr uint64_t kMaximumTokens = 1048576;
constexpr uint64_t kMaximumSlots = 4096;
constexpr uint64_t kMaximumSlotBytes = 32ull * 1024 * 1024 * 1024;
constexpr uint64_t kMaximumFileBytes = 64ull * 1024 * 1024 * 1024;
constexpr uint64_t kDigestChunkBytes = 1024 * 1024;

std::atomic<uint64_t> g_next_store_id{1};

CompatibilityReport state_error() {
    CompatibilityReport report = package_report(CompatibilityError::STATE_ABI_MISMATCH);
    report.stage = CompatibilityStage::State;
    return report;
}

bool checked_add(uint64_t left, uint64_t right, uint64_t& result) {
    if (left > std::numeric_limits<uint64_t>::max() - right) return false;
    result = left + right;
    return true;
}

bool checked_multiply(uint64_t left, uint64_t right, uint64_t& result) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) return false;
    result = left * right;
    return true;
}

bool align_up(uint64_t value, uint64_t alignment, uint64_t& result) {
    uint64_t expanded = 0;
    return checked_add(value, alignment - 1, expanded) && (result = expanded & ~(alignment - 1), true);
}

bool fits_size(uint64_t value) {
    return value <= std::numeric_limits<size_t>::max();
}

void append_u16(std::vector<uint8_t>& out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value));
    out.push_back(static_cast<uint8_t>(value >> 8));
}

void append_u32(std::vector<uint8_t>& out, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8) out.push_back(static_cast<uint8_t>(value >> shift));
}

void append_u64(std::vector<uint8_t>& out, uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8) out.push_back(static_cast<uint8_t>(value >> shift));
}

void append_f32(std::vector<uint8_t>& out, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    append_u32(out, bits);
}

bool read_u16(std::span<const uint8_t> bytes, uint64_t offset, uint16_t& value) {
    if (offset > bytes.size() || bytes.size() - offset < 2) return false;
    value = static_cast<uint16_t>(bytes[static_cast<size_t>(offset)]) |
            (static_cast<uint16_t>(bytes[static_cast<size_t>(offset + 1)]) << 8);
    return true;
}

bool read_u32(std::span<const uint8_t> bytes, uint64_t offset, uint32_t& value) {
    if (offset > bytes.size() || bytes.size() - offset < 4) return false;
    value = 0;
    for (unsigned shift = 0; shift != 32; shift += 8) {
        value |= static_cast<uint32_t>(bytes[static_cast<size_t>(offset + shift / 8)]) << shift;
    }
    return true;
}

bool read_u64(std::span<const uint8_t> bytes, uint64_t offset, uint64_t& value) {
    if (offset > bytes.size() || bytes.size() - offset < 8) return false;
    value = 0;
    for (unsigned shift = 0; shift != 64; shift += 8) {
        value |= static_cast<uint64_t>(bytes[static_cast<size_t>(offset + shift / 8)]) << shift;
    }
    return true;
}

bool read_f32(std::span<const uint8_t> bytes, uint64_t offset, float& value) {
    uint32_t bits = 0;
    if (!read_u32(bytes, offset, bits)) return false;
    std::memcpy(&value, &bits, sizeof(value));
    return std::isfinite(value);
}

std::vector<StateDigestChunk> digest_chunks(uint64_t length) {
    std::vector<StateDigestChunk> chunks;
    for (uint64_t offset = 0; offset < length;) {
        const uint64_t chunk = std::min(kDigestChunkBytes, length - offset);
        chunks.push_back({offset, chunk});
        offset += chunk;
    }
    return chunks;
}

Sha256Digest sha256(std::span<const uint8_t> bytes) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    for (const StateDigestChunk& chunk : digest_chunks(bytes.size())) {
        CC_SHA256_Update(&context, bytes.data() + static_cast<size_t>(chunk.offset),
                         static_cast<CC_LONG>(chunk.length));
    }
    Sha256Digest digest;
    CC_SHA256_Final(digest.bytes.data(), &context);
    return digest;
}

bool all_zero(std::span<const uint8_t> bytes, uint64_t begin, uint64_t end) {
    if (begin > end || end > bytes.size()) return false;
    return std::all_of(bytes.begin() + static_cast<size_t>(begin), bytes.begin() + static_cast<size_t>(end),
                       [](uint8_t value) { return value == 0; });
}

bool state_dimensions(const SemanticState& state, uint32_t& heads, uint32_t& dimension) {
    if (state.dimensions.size() != 3 || state.dimensions[0].kind != DimensionKind::Symbol ||
        state.dimensions[0].constant_or_symbol == 0 || state.dimensions[1].kind != DimensionKind::Constant ||
        state.dimensions[2].kind != DimensionKind::Constant || state.dimensions[1].constant_or_symbol == 0 ||
        state.dimensions[2].constant_or_symbol == 0 || state.dimensions[1].constant_or_symbol > UINT32_MAX ||
        state.dimensions[2].constant_or_symbol > UINT32_MAX) return false;
    heads = static_cast<uint32_t>(state.dimensions[1].constant_or_symbol);
    dimension = static_cast<uint32_t>(state.dimensions[2].constant_or_symbol);
    return true;
}

bool valid_format(const StateFormat& format, TransformDomain expected_domain) {
    return format.kind == StateFormatKind::GlobalContiguous && format.version == 1 &&
           format.logical_type == ScalarType::F32 && format.encoded_type == ScalarType::F32 &&
           format.logical_domain == TransformDomain::Untransformed && format.encoded_domain == expected_domain &&
           format.codec == CodecKind::Fp32 && format.cache_policy == CachePolicy::Global &&
           format.layout_policy == LayoutPolicy::TokenMajorContiguous && format.flags == 0 &&
           format.tile_tokens == 0 && format.mutable_tokens == 0 && format.alignment == 64 && format.reserved == 0;
}

bool slot_byte_count(uint64_t tokens, uint32_t heads, uint32_t dimension, uint64_t& data_bytes, uint64_t& payload_bytes) {
    uint64_t values = 0;
    return checked_multiply(tokens, heads, values) && checked_multiply(values, dimension, values) &&
           checked_multiply(values, sizeof(float), data_bytes) && checked_add(kPayloadHeaderBytes, data_bytes, payload_bytes) &&
           payload_bytes <= kMaximumSlotBytes;
}

bool valid_addition(const SemanticKvState& state, uint64_t tokens, uint32_t heads, uint32_t dimension) {
    uint64_t data_bytes = 0;
    uint64_t payload_bytes = 0;
    if (state.tokens != tokens || !slot_byte_count(tokens, heads, dimension, data_bytes, payload_bytes) ||
        data_bytes / sizeof(float) != state.key.size() || state.key.size() != state.value.size()) return false;
    return std::all_of(state.key.begin(), state.key.end(), [](float value) { return std::isfinite(value); }) &&
           std::all_of(state.value.begin(), state.value.end(), [](float value) { return std::isfinite(value); });
}

void append_descriptor(std::vector<uint8_t>& out, const SemanticState& state, uint32_t heads, uint32_t dimension,
                       uint64_t tokens, uint64_t payload_offset, uint64_t payload_length,
                       const Sha256Digest& payload_digest) {
    const StateFormat& format = state.formats.front();
    append_u32(out, state.id);
    for (uint16_t value : {static_cast<uint16_t>(state.kind), state.semantic_version,
                           static_cast<uint16_t>(format.kind), format.version,
                           static_cast<uint16_t>(format.logical_type), static_cast<uint16_t>(format.encoded_type),
                           static_cast<uint16_t>(format.logical_domain), static_cast<uint16_t>(format.encoded_domain),
                           static_cast<uint16_t>(format.codec), static_cast<uint16_t>(state.position_policy),
                           static_cast<uint16_t>(format.cache_policy), static_cast<uint16_t>(format.layout_policy),
                           static_cast<uint16_t>(state.dimensions.size()), static_cast<uint16_t>(0)}) append_u16(out, value);
    append_u64(out, 0);
    append_u64(out, heads);
    append_u64(out, dimension);
    for (unsigned index = 3; index != 8; ++index) append_u64(out, 0);
    append_u32(out, 0);
    append_u32(out, 0);
    append_u64(out, 0);
    append_u64(out, tokens);
    append_u64(out, payload_offset);
    append_u64(out, payload_length);
    out.insert(out.end(), payload_digest.bytes.begin(), payload_digest.bytes.end());
}

bool descriptor_matches(std::span<const uint8_t> bytes, uint64_t offset, const SemanticState& expected,
                        uint32_t heads, uint32_t dimension, uint64_t tokens,
                        uint64_t payload_offset, uint64_t payload_length, Sha256Digest payload_digest) {
    if (offset > bytes.size() || bytes.size() - offset < kDescriptorBytes || expected.formats.size() != 1) return false;
    const StateFormat& format = expected.formats.front();
    std::array<uint16_t, 14> actual{};
    uint32_t id = 0;
    if (!read_u32(bytes, offset, id) || id != expected.id) return false;
    for (size_t index = 0; index != actual.size(); ++index) if (!read_u16(bytes, offset + 4 + index * 2, actual[index])) return false;
    const std::array<uint16_t, 14> expected_values = {
        static_cast<uint16_t>(expected.kind), expected.semantic_version,
        static_cast<uint16_t>(format.kind), format.version,
        static_cast<uint16_t>(format.logical_type), static_cast<uint16_t>(format.encoded_type),
        static_cast<uint16_t>(format.logical_domain), static_cast<uint16_t>(format.encoded_domain),
        static_cast<uint16_t>(format.codec), static_cast<uint16_t>(expected.position_policy),
        static_cast<uint16_t>(format.cache_policy), static_cast<uint16_t>(format.layout_policy),
        static_cast<uint16_t>(expected.dimensions.size()), 0};
    if (actual != expected_values) return false;
    std::array<uint64_t, 8> dimensions{};
    for (size_t index = 0; index != dimensions.size(); ++index) if (!read_u64(bytes, offset + 32 + index * 8, dimensions[index])) return false;
    if (dimensions[0] != 0 || dimensions[1] != heads || dimensions[2] != dimension ||
        std::any_of(dimensions.begin() + 3, dimensions.end(), [](uint64_t value) { return value != 0; })) return false;
    uint32_t tile = 0, mutable_tokens = 0;
    uint64_t begin = 0, end = 0, actual_offset = 0, actual_length = 0;
    if (!read_u32(bytes, offset + 96, tile) || !read_u32(bytes, offset + 100, mutable_tokens) ||
        !read_u64(bytes, offset + 104, begin) || !read_u64(bytes, offset + 112, end) ||
        !read_u64(bytes, offset + 120, actual_offset) || !read_u64(bytes, offset + 128, actual_length) ||
        tile != 0 || mutable_tokens != 0 || begin != 0 || end != tokens ||
        actual_offset != payload_offset || actual_length != payload_length) return false;
    return std::equal(payload_digest.bytes.begin(), payload_digest.bytes.end(), bytes.begin() + static_cast<size_t>(offset + 136));
}

} // namespace

Sha256Digest token_contract_digest(const SemanticModel& model) {
    std::vector<uint8_t> bytes;
    constexpr char kDomain[] = "laplace-token-contract-v1";
    bytes.insert(bytes.end(), kDomain, kDomain + sizeof(kDomain));
    append_u16(bytes, static_cast<uint16_t>(model.entry_kind));
    append_u16(bytes, 0);
    append_u32(bytes, model.vocabulary_size);
    append_u32(bytes, model.bos_id);
    append_u32(bytes, model.eos_id);
    append_u32(bytes, static_cast<uint32_t>(model.stop_ids.size()));
    for (uint32_t stop_id : model.stop_ids) append_u32(bytes, stop_id);
    bytes.insert(bytes.end(), model.tokenizer_digest.begin(), model.tokenizer_digest.end());
    bytes.insert(bytes.end(), model.template_digest.begin(), model.template_digest.end());
    return sha256(bytes);
}

std::variant<std::vector<StateStore::SlotDefinition>, CompatibilityReport>
StateStore::definitions_from(const SemanticModel& model) {
    if (model.maximum_context == 0 || model.maximum_context > kMaximumTokens || model.states.empty() ||
        model.states.size() % 2 != 0 || model.states.size() > kMaximumSlots) return state_error();
    std::vector<SlotDefinition> definitions;
    definitions.reserve(model.states.size() / 2);
    uint32_t last_id = 0;
    for (size_t index = 0; index != model.states.size(); index += 2) {
        const SemanticState& key = model.states[index];
        const SemanticState& value = model.states[index + 1];
        uint32_t heads = 0, dimension = 0, value_heads = 0, value_dimension = 0;
        if ((index != 0 && key.id <= last_id) || value.id <= key.id || key.kind != StateKind::KeyCache ||
            value.kind != StateKind::ValueCache || key.semantic_version != 1 || value.semantic_version != 1 ||
            key.update_kind != StateUpdateKind::AppendKey || value.update_kind != StateUpdateKind::AppendValue ||
            key.position_policy != PositionPolicy::AppendOnly || value.position_policy != PositionPolicy::AppendOnly ||
            key.flags != 0 || value.flags != 0 || key.formats.size() != 1 || value.formats.size() != 1 ||
            !valid_format(key.formats[0], TransformDomain::RopeApplied) ||
            !valid_format(value.formats[0], TransformDomain::Untransformed) ||
            !state_dimensions(key, heads, dimension) || !state_dimensions(value, value_heads, value_dimension) ||
            heads != value_heads || dimension != value_dimension) return state_error();
        last_id = value.id;
        definitions.push_back({key, value, heads, dimension});
    }
    return definitions;
}

StateStore::StateStore(const SemanticModel& model)
    : maximum_context_(model.maximum_context), store_id_(g_next_store_id.fetch_add(1, std::memory_order_relaxed)) {
    auto definitions = definitions_from(model);
    if (!std::holds_alternative<std::vector<SlotDefinition>>(definitions)) return;
    definitions_ = std::get<std::vector<SlotDefinition>>(std::move(definitions));
    slots_.resize(definitions_.size());
    valid_ = true;
}

StateMutationResult StateStore::begin_execution_append(std::span<const uint32_t> token_ids) {
    if (!valid_ || append_active_ || token_ids.empty() || token_ids.size() > maximum_context_ ||
        token_history_.size() > maximum_context_ - token_ids.size()) return state_error();
    const uint64_t accepted = token_history_.size();
    for (size_t index = 0; index != slots_.size(); ++index) {
        if (slots_[index].tokens != accepted ||
            !valid_addition(slots_[index], slots_[index].tokens, definitions_[index].heads, definitions_[index].dimension)) {
            return state_error();
        }
    }
    undo_.push_back({slots_.empty() ? 0 : slots_.front().tokens, accepted});
    token_history_.insert(token_history_.end(), token_ids.begin(), token_ids.end());
    append_active_ = true;
    return std::monostate{};
}

StateMutationResult StateStore::finish_execution_append() {
    if (!valid_ || !append_active_) return state_error();
    const uint64_t accepted = token_history_.size();
    for (size_t index = 0; index != slots_.size(); ++index) {
        if (slots_[index].tokens != accepted ||
            !valid_addition(slots_[index], slots_[index].tokens, definitions_[index].heads, definitions_[index].dimension)) {
            return state_error();
        }
    }
    append_active_ = false;
    return std::monostate{};
}

StateMutationResult StateStore::append(std::span<const uint32_t> token_ids,
                                       const std::vector<SemanticKvState>& additions) {
    if (!valid_ || append_active_ || token_ids.empty() || additions.size() != slots_.size() || token_ids.size() > maximum_context_ ||
        token_history_.size() > maximum_context_ - token_ids.size()) return state_error();
    for (size_t index = 0; index != additions.size(); ++index) {
        if (!valid_addition(additions[index], token_ids.size(), definitions_[index].heads, definitions_[index].dimension)) return state_error();
    }
    undo_.push_back({slots_.empty() ? 0 : slots_.front().tokens, token_history_.size()});
    token_history_.insert(token_history_.end(), token_ids.begin(), token_ids.end());
    for (size_t index = 0; index != slots_.size(); ++index) {
        slots_[index].key.insert(slots_[index].key.end(), additions[index].key.begin(), additions[index].key.end());
        slots_[index].value.insert(slots_[index].value.end(), additions[index].value.begin(), additions[index].value.end());
        slots_[index].tokens += additions[index].tokens;
    }
    return std::monostate{};
}

StateMutationResult StateStore::set_lengths(uint64_t accepted_tokens, uint64_t history_size) {
    if (!valid_ || accepted_tokens > maximum_context_ || history_size > token_history_.size()) return state_error();
    for (size_t index = 0; index != slots_.size(); ++index) {
        uint64_t values = 0;
        if (!checked_multiply(accepted_tokens, definitions_[index].heads, values) ||
            !checked_multiply(values, definitions_[index].dimension, values) || values > slots_[index].key.size() ||
            values > slots_[index].value.size()) return state_error();
        slots_[index].key.resize(static_cast<size_t>(values));
        slots_[index].value.resize(static_cast<size_t>(values));
        slots_[index].tokens = static_cast<uint32_t>(accepted_tokens);
    }
    token_history_.resize(static_cast<size_t>(history_size));
    return std::monostate{};
}

StateMutationResult StateStore::trim(uint64_t accepted_tokens) {
    if (!valid_ || append_active_ || accepted_tokens > token_history_.size()) return state_error();
    undo_.push_back({slots_.empty() ? 0 : slots_.front().tokens, token_history_.size()});
    return set_lengths(accepted_tokens, accepted_tokens);
}

StateCursor StateStore::checkpoint() const noexcept {
    return {store_id_, generation_, static_cast<uint64_t>(token_history_.size()), undo_base_ + static_cast<uint64_t>(undo_.size())};
}

StateMutationResult StateStore::rollback(StateCursor cursor) {
    if (!valid_ || cursor.store_id != store_id_ || cursor.generation != generation_ ||
        cursor.undo_count < undo_base_ || cursor.undo_count > undo_base_ + undo_.size() ||
        cursor.accepted_tokens > token_history_.size()) return state_error();
    while (undo_base_ + undo_.size() > cursor.undo_count) {
        UndoEntry entry = undo_.back();
        undo_.pop_back();
        StateMutationResult restored = set_lengths(entry.tokens, entry.history_size);
        if (std::holds_alternative<CompatibilityReport>(restored)) return restored;
    }
    if (token_history_.size() != cursor.accepted_tokens) return state_error();
    append_active_ = false;
    return std::monostate{};
}

StateMutationResult StateStore::commit(StateCursor cursor) {
    if (!valid_ || append_active_ || cursor.store_id != store_id_ || cursor.generation != generation_ ||
        cursor.undo_count < undo_base_ || cursor.undo_count > undo_base_ + undo_.size() ||
        cursor.accepted_tokens > token_history_.size()) return state_error();
    const uint64_t discarded = cursor.undo_count - undo_base_;
    undo_.erase(undo_.begin(), undo_.begin() + static_cast<size_t>(discarded));
    undo_base_ = cursor.undo_count;
    return std::monostate{};
}

void StateStore::reset() noexcept {
    if (!valid_) return;
    for (SemanticKvState& slot : slots_) {
        slot.key.clear();
        slot.value.clear();
        slot.tokens = 0;
    }
    token_history_.clear();
    undo_.clear();
    undo_base_ = 0;
    append_active_ = false;
    ++generation_;
}

StateSaveResult StateStore::save(const Sha256Digest& model_fingerprint,
                                 const Sha256Digest& semantic_digest,
                                 const Sha256Digest& token_contract_digest) const {
    if (!valid_ || token_history_.size() > kMaximumTokens) return state_error();
    uint64_t descriptor_bytes = 0, descriptor_end = 0, header_length = 0, history_length = 0, payload_cursor = 0;
    if (!checked_multiply(definitions_.size() * 2, kDescriptorBytes, descriptor_bytes) ||
        !checked_add(kHeaderBytes, descriptor_bytes, descriptor_end) || !align_up(descriptor_end, kAlignment, header_length) ||
        !checked_multiply(token_history_.size(), sizeof(uint32_t), history_length) ||
        !checked_add(header_length, history_length, payload_cursor) || !align_up(payload_cursor, kAlignment, payload_cursor)) return state_error();
    struct SerializedSlot { uint64_t offset; uint64_t length; Sha256Digest digest; };
    std::vector<SerializedSlot> serialized;
    serialized.reserve(definitions_.size() * 2);
    for (size_t index = 0; index != slots_.size(); ++index) {
        const SemanticKvState& slot = slots_[index];
        if (slot.tokens != token_history_.size() || !valid_addition(slot, slot.tokens, definitions_[index].heads, definitions_[index].dimension)) return state_error();
        for (const std::vector<float>* values : {&slot.key, &slot.value}) {
            uint64_t data_bytes = 0, payload_length = 0, next = 0;
            if (!slot_byte_count(slot.tokens, definitions_[index].heads, definitions_[index].dimension, data_bytes, payload_length) ||
                !checked_add(payload_cursor, payload_length, next) || next > kMaximumFileBytes) return state_error();
            std::vector<uint8_t> payload;
            payload.reserve(static_cast<size_t>(payload_length));
            append_u64(payload, slot.tokens); append_u64(payload, maximum_context_);
            append_u32(payload, definitions_[index].heads); append_u32(payload, definitions_[index].dimension); append_u64(payload, data_bytes);
            for (float value : *values) append_f32(payload, value);
            serialized.push_back({payload_cursor, payload_length, sha256(payload)});
            if (!align_up(next, kAlignment, payload_cursor)) return state_error();
        }
    }
    uint64_t payload_end = serialized.empty() ? header_length + history_length : serialized.back().offset + serialized.back().length;
    uint64_t total_length = 0;
    if (!checked_add(payload_end, kDigestBytes, total_length) || total_length > kMaximumFileBytes || !fits_size(total_length)) return state_error();
    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(total_length));
    out.insert(out.end(), kMagic.begin(), kMagic.end());
    append_u16(out, kMajor); append_u16(out, kMinor); append_u32(out, static_cast<uint32_t>(header_length));
    append_u64(out, header_length); append_u64(out, total_length);
    out.insert(out.end(), model_fingerprint.bytes.begin(), model_fingerprint.bytes.end());
    out.insert(out.end(), semantic_digest.bytes.begin(), semantic_digest.bytes.end());
    out.insert(out.end(), token_contract_digest.bytes.begin(), token_contract_digest.bytes.end());
    append_u64(out, token_history_.size()); append_u64(out, token_history_.size());
    append_u64(out, header_length); append_u64(out, history_length);
    std::vector<uint8_t> history;
    history.reserve(static_cast<size_t>(history_length));
    for (uint32_t token : token_history_) append_u32(history, token);
    Sha256Digest history_digest = sha256(history);
    out.insert(out.end(), history_digest.bytes.begin(), history_digest.bytes.end());
    append_u32(out, static_cast<uint32_t>(serialized.size())); append_u32(out, 0);
    for (size_t index = 0; index != definitions_.size(); ++index) {
        append_descriptor(out, definitions_[index].key, definitions_[index].heads, definitions_[index].dimension,
                          slots_[index].tokens, serialized[index * 2].offset, serialized[index * 2].length, serialized[index * 2].digest);
        append_descriptor(out, definitions_[index].value, definitions_[index].heads, definitions_[index].dimension,
                          slots_[index].tokens, serialized[index * 2 + 1].offset, serialized[index * 2 + 1].length, serialized[index * 2 + 1].digest);
    }
    if (out.size() > header_length) return state_error();
    out.resize(static_cast<size_t>(header_length), 0);
    out.insert(out.end(), history.begin(), history.end());
    for (size_t index = 0; index != serialized.size(); ++index) {
        if (out.size() > serialized[index].offset) return state_error();
        out.resize(static_cast<size_t>(serialized[index].offset), 0);
        const SemanticKvState& slot = slots_[index / 2];
        const std::vector<float>& values = index % 2 == 0 ? slot.key : slot.value;
        uint64_t data_bytes = 0, payload_length = 0;
        if (!slot_byte_count(slot.tokens, definitions_[index / 2].heads, definitions_[index / 2].dimension, data_bytes, payload_length)) return state_error();
        append_u64(out, slot.tokens); append_u64(out, maximum_context_);
        append_u32(out, definitions_[index / 2].heads); append_u32(out, definitions_[index / 2].dimension); append_u64(out, data_bytes);
        for (float value : values) append_f32(out, value);
    }
    if (out.size() != payload_end) return state_error();
    Sha256Digest file_digest = sha256(out);
    out.insert(out.end(), file_digest.bytes.begin(), file_digest.bytes.end());
    return out;
}

StateRestoreResult StateStore::restore(const SemanticModel& model,
                                       const Sha256Digest& model_fingerprint,
                                       const Sha256Digest& semantic_digest,
                                       const Sha256Digest& token_contract_digest,
                                       std::span<const uint8_t> bytes) {
    StateStore result(model);
    if (!result.valid_ || bytes.size() < kHeaderBytes + kDigestBytes || bytes.size() > kMaximumFileBytes) return state_error();
    uint16_t major = 0, minor = 0;
    uint32_t header_length32 = 0, slot_count = 0, reserved = 0;
    uint64_t payload_offset = 0, total_length = 0, cursor = 0, token_count = 0, history_offset = 0, history_length = 0;
    if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin()) || !read_u16(bytes, 8, major) || !read_u16(bytes, 10, minor) ||
        !read_u32(bytes, 12, header_length32) || !read_u64(bytes, 16, payload_offset) || !read_u64(bytes, 24, total_length) ||
        !read_u64(bytes, 128, cursor) || !read_u64(bytes, 136, token_count) || !read_u64(bytes, 144, history_offset) ||
        !read_u64(bytes, 152, history_length) || !read_u32(bytes, 192, slot_count) || !read_u32(bytes, 196, reserved)) return state_error();
    const Sha256Digest whole_digest = sha256(bytes.first(bytes.size() - kDigestBytes));
    if (major != kMajor || minor != kMinor || total_length != bytes.size() || total_length > kMaximumFileBytes ||
        !std::equal(whole_digest.bytes.begin(), whole_digest.bytes.end(), bytes.end() - kDigestBytes) ||
        slot_count != result.definitions_.size() * 2 || slot_count > kMaximumSlots || reserved != 0 || cursor != token_count ||
        token_count > result.maximum_context_ || token_count > kMaximumTokens) return state_error();
    uint64_t descriptors = 0, header_minimum = 0, expected_header = 0, expected_history_length = 0;
    if (!checked_multiply(slot_count, kDescriptorBytes, descriptors) || !checked_add(kHeaderBytes, descriptors, header_minimum) ||
        !align_up(header_minimum, kAlignment, expected_header) || header_length32 != expected_header || payload_offset != expected_header ||
        history_offset != payload_offset || !checked_multiply(token_count, sizeof(uint32_t), expected_history_length) ||
        history_length != expected_history_length || !all_zero(bytes, header_minimum, expected_header)) return state_error();
    if (!std::equal(model_fingerprint.bytes.begin(), model_fingerprint.bytes.end(), bytes.begin() + 32) ||
        !std::equal(semantic_digest.bytes.begin(), semantic_digest.bytes.end(), bytes.begin() + 64) ||
        !std::equal(token_contract_digest.bytes.begin(), token_contract_digest.bytes.end(), bytes.begin() + 96)) return state_error();
    uint64_t history_end = 0;
    const Sha256Digest history_digest = sha256(bytes.subspan(static_cast<size_t>(history_offset), static_cast<size_t>(history_length)));
    if (!checked_add(history_offset, history_length, history_end) || history_end > bytes.size() - kDigestBytes ||
        !std::equal(history_digest.bytes.begin(), history_digest.bytes.end(), bytes.begin() + 160)) return state_error();
    result.token_history_.reserve(static_cast<size_t>(token_count));
    for (uint64_t index = 0; index != token_count; ++index) {
        uint32_t token = 0;
        if (!read_u32(bytes, history_offset + index * sizeof(uint32_t), token)) return state_error();
        result.token_history_.push_back(token);
    }
    uint64_t payload_cursor = 0;
    if (!align_up(history_end, kAlignment, payload_cursor) || !all_zero(bytes, history_end, payload_cursor)) return state_error();
    uint64_t last_payload_end = history_end;
    for (size_t slot = 0; slot != result.definitions_.size(); ++slot) {
        SemanticKvState restored;
        for (size_t member = 0; member != 2; ++member) {
            uint64_t data_bytes = 0, payload_length = 0;
            if (!slot_byte_count(token_count, result.definitions_[slot].heads, result.definitions_[slot].dimension, data_bytes, payload_length)) return state_error();
            uint64_t descriptor_offset = kHeaderBytes + (slot * 2 + member) * kDescriptorBytes;
            uint64_t payload_end = 0;
            if (!checked_add(payload_cursor, payload_length, payload_end) || payload_end > bytes.size() - kDigestBytes ||
                !descriptor_matches(bytes, descriptor_offset, member == 0 ? result.definitions_[slot].key : result.definitions_[slot].value,
                                    result.definitions_[slot].heads, result.definitions_[slot].dimension, token_count,
                                    payload_cursor, payload_length,
                                    sha256(bytes.subspan(static_cast<size_t>(payload_cursor), static_cast<size_t>(payload_length))))) return state_error();
            uint64_t stored_tokens = 0, capacity = 0, stored_data_bytes = 0;
            uint32_t heads = 0, dimension = 0;
            if (!read_u64(bytes, payload_cursor, stored_tokens) || !read_u64(bytes, payload_cursor + 8, capacity) ||
                !read_u32(bytes, payload_cursor + 16, heads) || !read_u32(bytes, payload_cursor + 20, dimension) ||
                !read_u64(bytes, payload_cursor + 24, stored_data_bytes) || stored_tokens != token_count ||
                capacity != result.maximum_context_ || heads != result.definitions_[slot].heads || dimension != result.definitions_[slot].dimension ||
                stored_data_bytes != data_bytes) return state_error();
            std::vector<float>& values = member == 0 ? restored.key : restored.value;
            values.reserve(static_cast<size_t>(data_bytes / sizeof(float)));
            for (uint64_t index = 0; index != data_bytes / sizeof(float); ++index) {
                float value = 0.0f;
                if (!read_f32(bytes, payload_cursor + kPayloadHeaderBytes + index * sizeof(float), value)) return state_error();
                values.push_back(value);
            }
            last_payload_end = payload_end;
            if (!align_up(payload_end, kAlignment, payload_cursor)) return state_error();
            if (slot * 2 + member + 1 < slot_count && !all_zero(bytes, payload_end, payload_cursor)) return state_error();
        }
        restored.tokens = static_cast<uint32_t>(token_count);
        result.slots_[slot] = std::move(restored);
    }
    uint64_t expected_total = 0;
    expected_total = last_payload_end;
    if (!checked_add(expected_total, kDigestBytes, expected_total) || expected_total != total_length) return state_error();
    return result;
}

std::vector<StateDigestChunk> state_digest_chunk_ranges_for_testing(uint64_t length) {
    return digest_chunks(length);
}

Sha256Digest state_digest_for_testing(std::span<const uint8_t> bytes) {
    return sha256(bytes);
}

} // namespace Laplace
