#include "program_state.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <new>
#include <unordered_map>
#include <unordered_set>

namespace Laplace {

namespace {

constexpr std::array<uint8_t, 8> kMagic = {
    'L', 'A', 'P', 'G', 'S', '0', '0', '1'};
constexpr uint16_t kMajor = 1;
constexpr uint16_t kMinor = 0;
constexpr uint32_t kHeaderBytes = 128;
constexpr uint32_t kSlotDescriptorBytes = 288;
constexpr uint32_t kCellDescriptorBytes = 64;
constexpr uint64_t kAlignment = 64;
constexpr uint64_t kMaximumSlots = 4096;
constexpr uint64_t kMaximumRank = 32;
constexpr uint64_t kMaximumCellBytes = uint64_t{32} << 30;
constexpr uint64_t kMaximumImageBytes = uint64_t{64} << 30;
constexpr uint64_t kMaximumEffectBoundaries = 65536;
constexpr uint64_t kDigestChunkBytes = 1024 * 1024;

std::atomic<uint64_t> g_next_root_id{1};

uint64_t allocate_root_id() noexcept {
    uint64_t candidate = g_next_root_id.load(std::memory_order_relaxed);
    while (candidate != UINT64_MAX) {
        if (g_next_root_id.compare_exchange_weak(
                candidate, candidate + 1, std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return candidate;
        }
    }
    return 0;
}

CompatibilityReport state_error(const char* detail) {
    CompatibilityReport report =
        compatibility_report(CompatibilityError::STATE_ABI_MISMATCH, detail);
    report.stage = CompatibilityStage::State;
    return report;
}

bool checked_add(uint64_t left, uint64_t right, uint64_t* result) {
    if (left > std::numeric_limits<uint64_t>::max() - right) return false;
    *result = left + right;
    return true;
}

bool checked_multiply(uint64_t left, uint64_t right, uint64_t* result) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        return false;
    }
    *result = left * right;
    return true;
}

bool align_up(uint64_t value, uint64_t alignment, uint64_t* result) {
    uint64_t expanded = 0;
    if (!checked_add(value, alignment - 1, &expanded)) return false;
    *result = expanded & ~(alignment - 1);
    return true;
}

void append_u8(std::vector<uint8_t>& bytes, uint8_t value) {
    bytes.push_back(value);
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<uint8_t>(value >> shift));
    }
}

void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        bytes.push_back(static_cast<uint8_t>(value >> shift));
    }
}

void put_u16(std::vector<uint8_t>& bytes, size_t offset, uint16_t value) {
    for (unsigned shift = 0; shift < 16; shift += 8) {
        bytes[offset + shift / 8] = static_cast<uint8_t>(value >> shift);
    }
}

void put_u32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        bytes[offset + shift / 8] = static_cast<uint8_t>(value >> shift);
    }
}

void put_u64(std::vector<uint8_t>& bytes, size_t offset, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        bytes[offset + shift / 8] = static_cast<uint8_t>(value >> shift);
    }
}

bool read_u16(std::span<const uint8_t> bytes, uint64_t offset,
              uint16_t* value) {
    if (offset > bytes.size() || bytes.size() - offset < 2) return false;
    *value = static_cast<uint16_t>(bytes[static_cast<size_t>(offset)]) |
             (static_cast<uint16_t>(bytes[static_cast<size_t>(offset + 1)])
              << 8);
    return true;
}

bool read_u32(std::span<const uint8_t> bytes, uint64_t offset,
              uint32_t* value) {
    if (offset > bytes.size() || bytes.size() - offset < 4) return false;
    *value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8) {
        *value |= static_cast<uint32_t>(
                      bytes[static_cast<size_t>(offset + shift / 8)])
                  << shift;
    }
    return true;
}

bool read_u64(std::span<const uint8_t> bytes, uint64_t offset,
              uint64_t* value) {
    if (offset > bytes.size() || bytes.size() - offset < 8) return false;
    *value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        *value |= static_cast<uint64_t>(
                      bytes[static_cast<size_t>(offset + shift / 8)])
                  << shift;
    }
    return true;
}

bool all_zero(std::span<const uint8_t> bytes, uint64_t begin, uint64_t end) {
    if (begin > end || end > bytes.size()) return false;
    return std::all_of(bytes.begin() + static_cast<size_t>(begin),
                       bytes.begin() + static_cast<size_t>(end),
                       [](uint8_t value) { return value == 0; });
}

std::array<uint8_t, 32> sha256(std::span<const uint8_t> bytes) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    for (uint64_t offset = 0; offset < bytes.size();) {
        const uint64_t length =
            std::min(kDigestChunkBytes, bytes.size() - offset);
        CC_SHA256_Update(&context, bytes.data() + static_cast<size_t>(offset),
                         static_cast<CC_LONG>(length));
        offset += length;
    }
    std::array<uint8_t, 32> digest{};
    CC_SHA256_Final(digest.data(), &context);
    return digest;
}

bool wire_size(std::span<const uint32_t> slot_ranks,
               std::span<const uint64_t> cell_byte_counts,
               uint64_t* wire_bytes) {
    if (slot_ranks.size() > kMaximumSlots ||
        cell_byte_counts.size() > kMaximumSlots) {
        return false;
    }
    uint64_t slot_table = 0;
    uint64_t cell_table = 0;
    uint64_t total = 0;
    if (!checked_multiply(slot_ranks.size(), kSlotDescriptorBytes,
                          &slot_table) ||
        !checked_multiply(cell_byte_counts.size(), kCellDescriptorBytes,
                          &cell_table) ||
        !checked_add(kHeaderBytes, slot_table, &total) ||
        !checked_add(total, cell_table, &total)) {
        return false;
    }
    for (uint32_t rank : slot_ranks) {
        if (rank > kMaximumRank) return false;
    }
    for (uint64_t byte_count : cell_byte_counts) {
        if (byte_count > kMaximumCellBytes ||
            !align_up(total, kAlignment, &total) ||
            !checked_add(total, byte_count, &total)) {
            return false;
        }
    }
    if (!checked_add(total, 32, &total) || total > kMaximumImageBytes) {
        return false;
    }
    *wire_bytes = total;
    return true;
}

uint64_t element_bytes(ElementType type) {
    switch (type) {
    case ElementType::I1:
        return 1;
    case ElementType::F16:
        return 2;
    case ElementType::I32:
    case ElementType::U32:
    case ElementType::F32:
        return 4;
    case ElementType::U64:
        return 8;
    }
    return 0;
}

struct DimensionContext {
    std::unordered_map<uint32_t, uint64_t> values;
};

void collect_parameters(const DimensionExpr& expression,
                        std::unordered_set<uint32_t>* parameters) {
    if (expression.expression == DimensionExpression::Parameter &&
        expression.value <= UINT32_MAX) {
        parameters->insert(static_cast<uint32_t>(expression.value));
    }
    for (const DimensionExpr& operand : expression.operands) {
        collect_parameters(operand, parameters);
    }
}

bool evaluate_dimension(const DimensionExpr& expression,
                        const DimensionContext& context, uint64_t* result) {
    switch (expression.expression) {
    case DimensionExpression::Constant:
        if (!expression.operands.empty()) return false;
        *result = expression.value;
        return true;
    case DimensionExpression::Parameter: {
        if (!expression.operands.empty() || expression.value > UINT32_MAX) {
            return false;
        }
        const auto found = context.values.find(
            static_cast<uint32_t>(expression.value));
        if (found == context.values.end()) return false;
        *result = found->second;
        return true;
    }
    case DimensionExpression::Add:
    case DimensionExpression::Multiply:
    case DimensionExpression::CeilDivide:
        break;
    }
    if (expression.operands.size() != 2) return false;
    uint64_t left = 0;
    uint64_t right = 0;
    if (!evaluate_dimension(expression.operands[0], context, &left) ||
        !evaluate_dimension(expression.operands[1], context, &right)) {
        return false;
    }
    if (expression.expression == DimensionExpression::Add) {
        return checked_add(left, right, result);
    }
    if (expression.expression == DimensionExpression::Multiply) {
        return checked_multiply(left, right, result);
    }
    if (right == 0) return false;
    *result = left / right + (left % right != 0 ? 1 : 0);
    return true;
}

bool concrete_type(const ValueType& type, const DimensionContext& context,
                   std::vector<uint64_t>* extents, uint64_t* byte_count) {
    const uint64_t width = element_bytes(type.element_type);
    if (width == 0 || type.dimensions.size() > kMaximumRank) return false;
    extents->clear();
    extents->reserve(type.dimensions.size());
    uint64_t elements = 1;
    for (const DimensionExpr& expression : type.dimensions) {
        uint64_t extent = 0;
        if (!evaluate_dimension(expression, context, &extent) ||
            !checked_multiply(elements, extent, &elements)) {
            return false;
        }
        extents->push_back(extent);
    }
    return checked_multiply(elements, width, byte_count) &&
           *byte_count <= kMaximumCellBytes;
}

uint64_t program_alias_key(const StateReference& reference,
                           uint32_t canonical_ordinal) {
    return reference.alias_group == UINT32_MAX
               ? (uint64_t{1} << 32) | canonical_ordinal
               : reference.alias_group;
}

StateSchemaDigest digest_schema(
    ProgramDigest program_digest,
    const std::vector<std::pair<uint32_t, uint64_t>>& bindings,
    const std::vector<VerifiedStateSlot>& slots,
    const std::vector<VerifiedStateCell>& cells) {
    std::vector<uint8_t> bytes;
    constexpr char kDomain[] = "laplace-program-state-schema-v1";
    bytes.insert(bytes.end(), kDomain, kDomain + sizeof(kDomain));
    bytes.insert(bytes.end(), program_digest.bytes.begin(),
                 program_digest.bytes.end());
    append_u32(bytes, static_cast<uint32_t>(bindings.size()));
    for (const auto& [ordinal, value] : bindings) {
        append_u32(bytes, ordinal);
        append_u64(bytes, value);
    }
    append_u32(bytes, static_cast<uint32_t>(slots.size()));
    for (const VerifiedStateSlot& slot : slots) {
        append_u32(bytes, slot.canonical_state_ordinal);
        append_u32(bytes, slot.canonical_cell_ordinal);
        append_u8(bytes, static_cast<uint8_t>(slot.element_type));
        append_u8(bytes, slot.writable ? 1 : 0);
        append_u32(bytes, static_cast<uint32_t>(slot.extents.size()));
        for (uint64_t extent : slot.extents) append_u64(bytes, extent);
        append_u64(bytes, slot.byte_count);
    }
    append_u32(bytes, static_cast<uint32_t>(cells.size()));
    for (const VerifiedStateCell& cell : cells) {
        append_u32(bytes, cell.canonical_ordinal);
        append_u64(bytes, cell.byte_count);
        append_u8(bytes, cell.writable ? 1 : 0);
        append_u32(bytes, static_cast<uint32_t>(cell.slot_indices.size()));
        for (uint32_t slot : cell.slot_indices) append_u32(bytes, slot);
    }
    StateSchemaDigest result;
    result.bytes = sha256(bytes);
    return result;
}

} // namespace

VerifiedStateSchema::VerifiedStateSchema(
    ProgramDigest program_digest, StateSchemaDigest digest,
    StateSchema definition,
    std::vector<VerifiedStateSlot> slots,
    std::vector<VerifiedStateCell> cells)
    : program_digest_(program_digest), digest_(digest),
      definition_(std::move(definition)), slots_(std::move(slots)),
      cells_(std::move(cells)) {}

StateSchemaVerificationResult
verify_state_schema(StateSchema schema, const VerifiedProgram& verified_program) {
    const Program& program = program_definition(verified_program);
    if (schema.major != 1 || schema.minor != 0 ||
        schema.slots.size() != program.state_references.size() ||
        schema.slots.size() > kMaximumSlots) {
        return state_error("state schema version or logical slot count is invalid");
    }

    std::unordered_map<uint32_t, const DimensionParameter*> parameters;
    for (const DimensionParameter& parameter : program.dimension_parameters) {
        parameters.emplace(parameter.id, &parameter);
    }
    std::unordered_map<uint32_t, uint32_t> canonical_dimensions;
    const auto dimension_ids =
        canonical_dimension_parameter_ids(verified_program);
    for (uint32_t ordinal = 0; ordinal < dimension_ids.size(); ++ordinal) {
        canonical_dimensions.emplace(dimension_ids[ordinal], ordinal);
    }
    std::unordered_set<uint32_t> used_parameters;
    for (const StateReference& reference : program.state_references) {
        for (const DimensionExpr& expression : reference.type.dimensions) {
            collect_parameters(expression, &used_parameters);
        }
    }

    DimensionContext context;
    std::vector<std::pair<uint32_t, uint64_t>> canonical_bindings;
    for (const StateDimensionBinding& binding : schema.dimension_bindings) {
        const auto parameter = parameters.find(binding.parameter_id);
        const auto ordinal = canonical_dimensions.find(binding.parameter_id);
        if (parameter == parameters.end() || ordinal == canonical_dimensions.end() ||
            used_parameters.find(binding.parameter_id) == used_parameters.end() ||
            binding.value < parameter->second->lower ||
            binding.value > parameter->second->upper ||
            !context.values.emplace(binding.parameter_id, binding.value).second) {
            return state_error("state dimension binding is invalid or repeated");
        }
        canonical_bindings.push_back({ordinal->second, binding.value});
    }
    if (context.values.size() != used_parameters.size()) {
        return state_error("state dimension bindings are incomplete");
    }
    std::sort(canonical_bindings.begin(), canonical_bindings.end());

    std::unordered_map<uint32_t, const StateReference*> references;
    for (const StateReference& reference : program.state_references) {
        references.emplace(reference.id, &reference);
    }
    std::unordered_map<uint32_t, uint32_t> canonical_states;
    const auto state_ids = canonical_state_reference_ids(verified_program);
    for (uint32_t ordinal = 0; ordinal < state_ids.size(); ++ordinal) {
        canonical_states.emplace(state_ids[ordinal], ordinal);
    }

    struct Draft {
        StateSlotSchema schema;
        const StateReference* reference = nullptr;
        uint32_t canonical_ordinal = UINT32_MAX;
        uint64_t alias_key = UINT64_MAX;
        uint64_t byte_count = 0;
    };
    std::vector<Draft> drafts;
    drafts.reserve(schema.slots.size());
    std::unordered_set<uint32_t> bound_references;
    std::unordered_map<uint64_t, uint32_t> program_to_schema_cell;
    std::unordered_map<uint32_t, uint64_t> schema_to_program_cell;
    for (StateSlotSchema& slot : schema.slots) {
        const auto reference = references.find(slot.state_reference_id);
        const auto ordinal = canonical_states.find(slot.state_reference_id);
        if (slot.alias_cell == UINT32_MAX || reference == references.end() ||
            ordinal == canonical_states.end() ||
            !bound_references.insert(slot.state_reference_id).second ||
            slot.element_type != reference->second->type.element_type) {
            return state_error("state slot identity or element type is invalid");
        }
        std::vector<uint64_t> extents;
        uint64_t byte_count = 0;
        if (!concrete_type(reference->second->type, context, &extents,
                           &byte_count) ||
            extents != slot.extents) {
            return state_error("state slot concrete shape does not match");
        }
        const uint64_t alias_key =
            program_alias_key(*reference->second, ordinal->second);
        const auto [program_cell, program_inserted] =
            program_to_schema_cell.emplace(alias_key, slot.alias_cell);
        const auto [schema_cell, schema_inserted] =
            schema_to_program_cell.emplace(slot.alias_cell, alias_key);
        if ((!program_inserted && program_cell->second != slot.alias_cell) ||
            (!schema_inserted && schema_cell->second != alias_key)) {
            return state_error("state alias-cell equivalence does not match");
        }
        drafts.push_back({std::move(slot), reference->second, ordinal->second,
                          alias_key, byte_count});
    }
    if (bound_references.size() != references.size()) {
        return state_error("state schema does not bind every state reference");
    }
    std::sort(drafts.begin(), drafts.end(),
              [](const Draft& left, const Draft& right) {
                  return left.canonical_ordinal < right.canonical_ordinal;
              });

    std::unordered_map<uint64_t, uint32_t> canonical_cells;
    std::vector<VerifiedStateSlot> slots;
    std::vector<VerifiedStateCell> cells;
    slots.reserve(drafts.size());
    for (Draft& draft : drafts) {
        auto [cell, inserted] = canonical_cells.emplace(
            draft.alias_key, static_cast<uint32_t>(canonical_cells.size()));
        if (inserted) {
            VerifiedStateCell new_cell;
            new_cell.canonical_ordinal = cell->second;
            new_cell.byte_count = draft.byte_count;
            new_cell.writable = draft.reference->writable;
            cells.push_back(std::move(new_cell));
        } else if (cells[cell->second].byte_count != draft.byte_count) {
            return state_error("aliased state references have different extents");
        }
        VerifiedStateSlot verified;
        verified.state_reference_id = draft.schema.state_reference_id;
        verified.canonical_state_ordinal = draft.canonical_ordinal;
        verified.canonical_cell_ordinal = cell->second;
        verified.element_type = draft.schema.element_type;
        verified.extents = std::move(draft.schema.extents);
        verified.byte_count = draft.byte_count;
        verified.writable = draft.reference->writable;
        const uint32_t slot_index = static_cast<uint32_t>(slots.size());
        slots.push_back(std::move(verified));
        cells[cell->second].slot_indices.push_back(slot_index);
        cells[cell->second].writable =
            cells[cell->second].writable || draft.reference->writable;
    }
    std::vector<uint32_t> slot_ranks;
    slot_ranks.reserve(slots.size());
    for (const VerifiedStateSlot& slot : slots) {
        slot_ranks.push_back(static_cast<uint32_t>(slot.extents.size()));
    }
    std::vector<uint64_t> cell_byte_counts;
    cell_byte_counts.reserve(cells.size());
    for (const VerifiedStateCell& cell : cells) {
        cell_byte_counts.push_back(cell.byte_count);
    }
    uint64_t ignored_wire_bytes = 0;
    if (!wire_size(slot_ranks, cell_byte_counts, &ignored_wire_bytes)) {
        return state_error("state image byte extent exceeds the bound");
    }

    const ProgramDigest program_identity = program_digest(verified_program);
    const StateSchemaDigest schema_identity =
        digest_schema(program_identity, canonical_bindings, slots, cells);
    StateSchema canonical_definition;
    canonical_definition.major = schema.major;
    canonical_definition.minor = schema.minor;
    canonical_definition.dimension_bindings.reserve(canonical_bindings.size());
    for (const auto& [ordinal, value] : canonical_bindings) {
        canonical_definition.dimension_bindings.push_back(
            {dimension_ids[ordinal], value});
    }
    canonical_definition.slots.reserve(slots.size());
    for (const VerifiedStateSlot& slot : slots) {
        canonical_definition.slots.push_back(
            {slot.state_reference_id, slot.element_type, slot.extents,
             slot.canonical_cell_ordinal});
    }
    return VerifiedStateSchema(program_identity, schema_identity,
                               std::move(canonical_definition),
                               std::move(slots), std::move(cells));
}

ProgramDigest state_program_digest(const VerifiedStateSchema& schema) {
    return schema.program_digest_;
}

StateSchemaDigest state_schema_digest(const VerifiedStateSchema& schema) {
    return schema.digest_;
}

namespace {

constexpr size_t kStateSchemaWireLimit = 4u * 1024u * 1024u;
constexpr size_t kStateSchemaBindingLimit = 4096;

CompatibilityReport state_wire_error(CompatibilityError code,
                                     const char* detail) {
    CompatibilityReport report = compatibility_report(code, detail);
    report.stage = CompatibilityStage::State;
    return report;
}

class StateSchemaWireWriter {
public:
    void u8(uint8_t value) { bytes_.push_back(value); }
    void u16(uint16_t value) {
        for (unsigned shift = 0; shift != 16; shift += 8)
            u8(static_cast<uint8_t>(value >> shift));
    }
    void u32(uint32_t value) {
        for (unsigned shift = 0; shift != 32; shift += 8)
            u8(static_cast<uint8_t>(value >> shift));
    }
    void u64(uint64_t value) {
        for (unsigned shift = 0; shift != 64; shift += 8)
            u8(static_cast<uint8_t>(value >> shift));
    }
    void magic() {
        static constexpr std::array<uint8_t, 8> value = {
            'L', 'A', 'P', 'S', 'T', 'W', '0', '1'};
        bytes_.insert(bytes_.end(), value.begin(), value.end());
    }
    std::vector<uint8_t> finish() {
        const uint32_t size = static_cast<uint32_t>(bytes_.size());
        for (unsigned shift = 0; shift != 32; shift += 8)
            bytes_[12 + shift / 8] = static_cast<uint8_t>(size >> shift);
        return std::move(bytes_);
    }

private:
    std::vector<uint8_t> bytes_;
};

class StateSchemaWireReader {
public:
    explicit StateSchemaWireReader(std::span<const uint8_t> wire) : wire_(wire) {}

    bool u8(uint8_t* value) {
        if (!value || remaining() < 1) return false;
        *value = wire_[at_++];
        return true;
    }
    bool u16(uint16_t* value) {
        if (!value || remaining() < 2) return false;
        *value = static_cast<uint16_t>(wire_[at_]) |
                 static_cast<uint16_t>(wire_[at_ + 1]) << 8;
        at_ += 2;
        return true;
    }
    bool u32(uint32_t* value) {
        if (!value || remaining() < 4) return false;
        *value = 0;
        for (unsigned shift = 0; shift != 32; shift += 8)
            *value |= static_cast<uint32_t>(wire_[at_++]) << shift;
        return true;
    }
    bool u64(uint64_t* value) {
        if (!value || remaining() < 8) return false;
        *value = 0;
        for (unsigned shift = 0; shift != 64; shift += 8)
            *value |= static_cast<uint64_t>(wire_[at_++]) << shift;
        return true;
    }
    size_t remaining() const noexcept { return wire_.size() - at_; }
    bool finished() const noexcept { return at_ == wire_.size(); }

private:
    std::span<const uint8_t> wire_;
    size_t at_ = 0;
};

} // namespace

StateSchemaWireResult
encode_state_schema_wire(const VerifiedStateSchema& verified) {
    try {
        const StateSchema& schema = verified.definition_;
        StateSchemaWireWriter writer;
        writer.magic();
        writer.u16(1);
        writer.u16(0);
        writer.u32(0);
        writer.u16(schema.major);
        writer.u16(schema.minor);
        writer.u32(static_cast<uint32_t>(schema.dimension_bindings.size()));
        for (const StateDimensionBinding& binding : schema.dimension_bindings) {
            writer.u32(binding.parameter_id);
            writer.u64(binding.value);
        }
        writer.u32(static_cast<uint32_t>(schema.slots.size()));
        for (const StateSlotSchema& slot : schema.slots) {
            writer.u32(slot.state_reference_id);
            writer.u8(static_cast<uint8_t>(slot.element_type));
            writer.u32(static_cast<uint32_t>(slot.extents.size()));
            for (uint64_t extent : slot.extents) writer.u64(extent);
            writer.u32(slot.alias_cell);
        }
        std::vector<uint8_t> wire = writer.finish();
        if (wire.size() > kStateSchemaWireLimit) {
            return state_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                    "state schema wire exceeds its bounded size");
        }
        return wire;
    } catch (const std::bad_alloc&) {
        return state_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                "state schema wire allocation failed");
    }
}

StateSchemaVerificationResult
decode_state_schema_wire(std::span<const uint8_t> wire,
                         const VerifiedProgram& program) {
    static constexpr std::array<uint8_t, 8> magic = {
        'L', 'A', 'P', 'S', 'T', 'W', '0', '1'};
    if (wire.size() < 24 || wire.size() > kStateSchemaWireLimit ||
        !std::equal(magic.begin(), magic.end(), wire.begin())) {
        return state_wire_error(CompatibilityError::PACKAGE_BAD_MAGIC,
                                "state schema wire header is invalid");
    }
    try {
        StateSchemaWireReader reader(wire.subspan(8));
        uint16_t version = 0, reserved = 0;
        uint32_t total = 0;
        StateSchema schema;
        if (!reader.u16(&version) || !reader.u16(&reserved) ||
            !reader.u32(&total) || version != 1 || reserved != 0 ||
            total != wire.size() || !reader.u16(&schema.major) ||
            !reader.u16(&schema.minor)) {
            return state_wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED,
                                    "state schema wire version or length is invalid");
        }
        uint32_t bindings = 0;
        if (!reader.u32(&bindings) || bindings > kStateSchemaBindingLimit ||
            bindings > reader.remaining() / 12) {
            return state_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                    "state schema dimension bindings are invalid");
        }
        schema.dimension_bindings.resize(bindings);
        for (StateDimensionBinding& binding : schema.dimension_bindings) {
            if (!reader.u32(&binding.parameter_id) || !reader.u64(&binding.value)) {
                return state_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                        "state schema dimension binding is truncated");
            }
        }
        uint32_t slots = 0;
        if (!reader.u32(&slots) || slots > kMaximumSlots) {
            return state_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                    "state schema slot count is invalid");
        }
        schema.slots.resize(slots);
        for (StateSlotSchema& slot : schema.slots) {
            uint8_t element = 0;
            uint32_t rank = 0;
            if (!reader.u32(&slot.state_reference_id) || !reader.u8(&element) ||
                !reader.u32(&rank) || rank > kMaximumRank ||
                rank > reader.remaining() / sizeof(uint64_t)) {
                return state_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                        "state schema slot is invalid");
            }
            slot.element_type = static_cast<ElementType>(element);
            slot.extents.resize(rank);
            for (uint64_t& extent : slot.extents) {
                if (!reader.u64(&extent)) {
                    return state_wire_error(
                        CompatibilityError::PACKAGE_BOUNDS_INVALID,
                        "state schema slot extent is truncated");
                }
            }
            if (!reader.u32(&slot.alias_cell)) {
                return state_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                        "state schema alias cell is truncated");
            }
        }
        if (!reader.finished()) {
            return state_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                    "state schema wire has trailing bytes");
        }
        return verify_state_schema(std::move(schema), program);
    } catch (const std::bad_alloc&) {
        return state_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                "state schema wire allocation failed");
    }
}

struct ProgramStateRootStatus {
    bool poisoned = false;
};

struct ProgramStateCandidateStorage {
    std::weak_ptr<ProgramStateRootStatus> root_status;
    uint64_t root_id = 0;
    uint64_t candidate_id = 0;
    uint64_t base_generation = 0;
    uint64_t effect_boundaries = 0;
    StateSchemaDigest schema_digest;
    std::vector<VerifiedStateSlot> slots;
    std::vector<std::shared_ptr<const std::vector<uint8_t>>> base;
    std::vector<std::shared_ptr<const std::vector<uint8_t>>> replacements;
    bool consumed = false;
    bool abandoned = false;
};

ExecutionResult::ExecutionResult(
    uint64_t root_id, uint64_t candidate_id, uint64_t base_generation,
    uint64_t effect_boundaries, StateSchemaDigest schema_digest,
    ExecutionDisposition disposition, bool valid) noexcept
    : root_id_(root_id), candidate_id_(candidate_id),
      base_generation_(base_generation), effect_boundaries_(effect_boundaries),
      schema_digest_(schema_digest), disposition_(disposition), valid_(valid) {}

ExecutionResult::ExecutionResult(ExecutionResult&& other) noexcept
    : root_id_(other.root_id_), candidate_id_(other.candidate_id_),
      base_generation_(other.base_generation_),
      effect_boundaries_(other.effect_boundaries_),
      schema_digest_(other.schema_digest_), disposition_(other.disposition_),
      valid_(other.valid_), consumed_(other.consumed_) {
    other.valid_ = false;
    other.consumed_ = true;
}

CandidateGeneration::CandidateGeneration(
    std::shared_ptr<ProgramStateCandidateStorage> state) noexcept
    : state_(std::move(state)) {}

CandidateGeneration::~CandidateGeneration() {
    if (!state_ || state_->consumed || state_->abandoned) return;
    state_->abandoned = true;
    if (const auto status = state_->root_status.lock()) status->poisoned = true;
}

CandidateGeneration::CandidateGeneration(CandidateGeneration&& other) noexcept
    : state_(std::move(other.state_)) {}

bool CandidateGeneration::valid() const noexcept {
    if (!state_ || state_->consumed || state_->abandoned) return false;
    const auto status = state_->root_status.lock();
    return status && !status->poisoned;
}

uint64_t CandidateGeneration::base_generation() const noexcept {
    return state_ ? state_->base_generation : 0;
}

uint64_t CandidateGeneration::effect_boundaries() const noexcept {
    return state_ ? state_->effect_boundaries : 0;
}

size_t CandidateGeneration::slot_index(
    uint32_t state_reference_id) const noexcept {
    if (!state_) return SIZE_MAX;
    for (size_t index = 0; index < state_->slots.size(); ++index) {
        if (state_->slots[index].state_reference_id == state_reference_id) {
            return index;
        }
    }
    return SIZE_MAX;
}

StateReadView
CandidateGeneration::read_slot(uint32_t state_reference_id) const noexcept {
    if (!valid()) return {};
    const size_t slot = slot_index(state_reference_id);
    if (slot == SIZE_MAX) return {};
    const uint32_t cell = state_->slots[slot].canonical_cell_ordinal;
    const auto& payload = state_->replacements[cell]
                              ? state_->replacements[cell]
                              : state_->base[cell];
    return StateReadView(payload);
}

ProgramStateMutationResult CandidateGeneration::write_slot(
    uint32_t state_reference_id, std::span<const uint8_t> bytes) {
    if (!valid()) return state_error("state candidate is not live");
    const size_t slot = slot_index(state_reference_id);
    if (slot == SIZE_MAX || !state_->slots[slot].writable ||
        bytes.size() != state_->slots[slot].byte_count ||
        state_->effect_boundaries == kMaximumEffectBoundaries) {
        return state_error("state candidate write contract is invalid");
    }
    const uint32_t cell = state_->slots[slot].canonical_cell_ordinal;
    auto replacement = std::make_shared<std::vector<uint8_t>>(bytes.size());
    if (!bytes.empty()) {
        std::memcpy(replacement->data(), bytes.data(), bytes.size());
    }
    state_->replacements[cell] = std::move(replacement);
    ++state_->effect_boundaries;
    return std::monostate{};
}

StateRoot::StateRoot(
    VerifiedStateSchema schema,
    std::vector<std::shared_ptr<const std::vector<uint8_t>>> current,
    uint64_t generation)
    : schema_(std::move(schema)), current_(std::move(current)),
      status_(std::make_shared<ProgramStateRootStatus>()),
      root_id_(allocate_root_id()),
      generation_(generation), valid_(root_id_ != 0) {}

StateRoot::StateRoot(StateRoot&& other) noexcept
    : schema_(std::move(other.schema_)), current_(std::move(other.current_)),
      status_(std::move(other.status_)),
      active_candidate_(std::move(other.active_candidate_)),
      quarantined_candidate_(std::move(other.quarantined_candidate_)),
      root_id_(other.root_id_),
      generation_(other.generation_), next_candidate_id_(other.next_candidate_id_),
      valid_(other.valid_) {
    other.root_id_ = 0;
    other.valid_ = false;
}

bool StateRoot::poisoned() const noexcept {
    return !status_ || status_->poisoned;
}

bool StateRoot::candidate_active() const noexcept {
    return active_candidate_ && !active_candidate_->consumed &&
           !active_candidate_->abandoned;
}

size_t StateRoot::slot_index(uint32_t state_reference_id) const noexcept {
    for (size_t index = 0; index < schema_.slots_.size(); ++index) {
        if (schema_.slots_[index].state_reference_id == state_reference_id) {
            return index;
        }
    }
    return SIZE_MAX;
}

StateReadView
StateRoot::current_slot(uint32_t state_reference_id) const noexcept {
    if (!valid_ || poisoned()) return {};
    const size_t slot = slot_index(state_reference_id);
    if (slot == SIZE_MAX) return {};
    return StateReadView(
        current_[schema_.slots_[slot].canonical_cell_ordinal]);
}

std::variant<CandidateGeneration, CompatibilityReport>
StateRoot::begin_candidate() {
    if (!valid_ || poisoned() || active_candidate_ ||
        generation_ == UINT64_MAX || next_candidate_id_ == UINT64_MAX) {
        return state_error("state root cannot begin a candidate");
    }
    auto state = std::make_shared<ProgramStateCandidateStorage>();
    state->root_status = status_;
    state->root_id = root_id_;
    state->candidate_id = next_candidate_id_++;
    state->base_generation = generation_;
    state->schema_digest = schema_.digest_;
    state->slots = schema_.slots_;
    state->base = current_;
    state->replacements.resize(schema_.cells_.size());
    active_candidate_ = state;
    return CandidateGeneration(std::move(state));
}

bool StateRoot::matches(const CandidateGeneration& candidate) const noexcept {
    return valid_ && !poisoned() && candidate.valid() &&
           candidate.state_ == active_candidate_ &&
           candidate.state_->root_id == root_id_ &&
           candidate.state_->base_generation == generation_ &&
           candidate.state_->schema_digest == schema_.digest_;
}

bool StateRoot::matches(
    const ExecutionResult& result,
    const ProgramStateCandidateStorage& candidate) const noexcept {
    return result.valid() && result.root_id_ == candidate.root_id &&
           result.candidate_id_ == candidate.candidate_id &&
           result.base_generation_ == candidate.base_generation &&
           result.effect_boundaries_ == candidate.effect_boundaries &&
           result.schema_digest_ == candidate.schema_digest;
}

void StateRoot::consume(CandidateGeneration& candidate,
                        ExecutionResult& result) noexcept {
    candidate.state_->consumed = true;
    active_candidate_.reset();
    candidate.state_.reset();
    result.consumed_ = true;
}

ProgramStateMutationResult
StateRoot::resolve_candidate(CandidateGeneration&& candidate,
                             ExecutionResult&& result) {
    if (!matches(candidate) || !matches(result, *candidate.state_)) {
        return state_error("state execution result does not match the candidate");
    }

    switch (result.disposition_) {
    case ExecutionDisposition::RejectedBeforeSubmission:
        consume(candidate, result);
        return std::monostate{};
    case ExecutionDisposition::Completed:
        if (generation_ == UINT64_MAX) {
            return state_error("state generation is exhausted");
        }
        for (size_t cell = 0; cell < current_.size(); ++cell) {
            if (candidate.state_->replacements[cell]) {
                current_[cell] = candidate.state_->replacements[cell];
            }
        }
        ++generation_;
        consume(candidate, result);
        return std::monostate{};
    case ExecutionDisposition::Failed:
    case ExecutionDisposition::Indeterminate:
        status_->poisoned = true;
        quarantined_candidate_ = active_candidate_;
        active_candidate_.reset();
        candidate.state_->consumed = true;
        candidate.state_.reset();
        result.consumed_ = true;
        return std::monostate{};
    }
    return state_error("state execution disposition is invalid");
}

#ifdef LAPLACE_TESTING
ExecutionResult program_state_execution_result_for_testing(
    const CandidateGeneration& candidate, ExecutionDisposition disposition) {
    if (!candidate.valid()) {
        return ExecutionResult(0, 0, 0, 0, {}, disposition, false);
    }
    return ExecutionResult(
        candidate.state_->root_id, candidate.state_->candidate_id,
        candidate.state_->base_generation, candidate.state_->effect_boundaries,
        candidate.state_->schema_digest, disposition, true);
}

bool program_state_has_retained_candidate_for_testing(const StateRoot& root) {
    return static_cast<bool>(root.active_candidate_) ||
           static_cast<bool>(root.quarantined_candidate_);
}
#endif

StateRootResult make_state_root(const VerifiedStateSchema& schema,
                                std::vector<StateCellValue> values) {
    if (values.size() != schema.cells_.size()) {
        return state_error("initial state cell count does not match the schema");
    }
    std::vector<std::shared_ptr<const std::vector<uint8_t>>> cells(
        schema.cells_.size());
    for (StateCellValue& value : values) {
        size_t slot = SIZE_MAX;
        for (size_t index = 0; index < schema.slots_.size(); ++index) {
            if (schema.slots_[index].state_reference_id ==
                value.state_reference_id) {
                slot = index;
                break;
            }
        }
        if (slot == SIZE_MAX) {
            return state_error("initial state reference is unresolved");
        }
        const uint32_t cell = schema.slots_[slot].canonical_cell_ordinal;
        if (cells[cell] || value.bytes.size() != schema.cells_[cell].byte_count) {
            return state_error("initial state cell is repeated or has wrong size");
        }
        cells[cell] = std::make_shared<const std::vector<uint8_t>>(
            std::move(value.bytes));
    }
    if (std::any_of(cells.begin(), cells.end(),
                    [](const auto& cell) { return !cell; })) {
        return state_error("initial state cells are incomplete");
    }
    StateRoot root(schema, std::move(cells), 0);
    if (!root.valid()) return state_error("state root identity is exhausted");
    return root;
}

ProgramStateSaveResult StateRoot::save() const {
    if (!valid_ || poisoned() || active_candidate_ ||
        current_.size() != schema_.cells_.size()) {
        return state_error("state root is not saveable");
    }
    uint64_t slot_table_bytes = 0;
    uint64_t cell_table_bytes = 0;
    uint64_t payload_floor = 0;
    if (!checked_multiply(schema_.slots_.size(), kSlotDescriptorBytes,
                          &slot_table_bytes) ||
        !checked_multiply(schema_.cells_.size(), kCellDescriptorBytes,
                          &cell_table_bytes) ||
        !checked_add(kHeaderBytes, slot_table_bytes, &payload_floor) ||
        !checked_add(payload_floor, cell_table_bytes, &payload_floor) ||
        payload_floor > SIZE_MAX) {
        return state_error("state descriptor tables overflow");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(payload_floor), 0);

    for (size_t index = 0; index < schema_.slots_.size(); ++index) {
        const VerifiedStateSlot& slot = schema_.slots_[index];
        const size_t descriptor =
            kHeaderBytes + index * kSlotDescriptorBytes;
        put_u32(bytes, descriptor, static_cast<uint32_t>(index));
        put_u32(bytes, descriptor + 4, slot.canonical_cell_ordinal);
        bytes[descriptor + 8] = static_cast<uint8_t>(slot.element_type);
        bytes[descriptor + 9] = slot.writable ? 1 : 0;
        put_u16(bytes, descriptor + 10,
                static_cast<uint16_t>(slot.extents.size()));
        for (size_t extent = 0; extent < slot.extents.size(); ++extent) {
            put_u64(bytes, descriptor + 16 + extent * 8,
                    slot.extents[extent]);
        }
        put_u64(bytes, descriptor + 272, slot.byte_count);
    }

    const size_t cell_table =
        kHeaderBytes + schema_.slots_.size() * kSlotDescriptorBytes;
    for (size_t index = 0; index < schema_.cells_.size(); ++index) {
        uint64_t aligned = 0;
        if (!align_up(bytes.size(), kAlignment, &aligned) ||
            aligned > SIZE_MAX) {
            return state_error("state payload alignment overflows");
        }
        bytes.resize(static_cast<size_t>(aligned), 0);
        const uint64_t payload_offset = bytes.size();
        const auto& payload = *current_[index];
        if (payload.size() != schema_.cells_[index].byte_count ||
            payload.size() > SIZE_MAX - bytes.size()) {
            return state_error("state payload size does not match the schema");
        }
        bytes.insert(bytes.end(), payload.begin(), payload.end());
        const auto payload_digest = sha256(payload);
        const size_t descriptor = cell_table + index * kCellDescriptorBytes;
        put_u32(bytes, descriptor, static_cast<uint32_t>(index));
        bytes[descriptor + 4] = schema_.cells_[index].writable ? 1 : 0;
        put_u64(bytes, descriptor + 8, payload.size());
        put_u64(bytes, descriptor + 16, payload_offset);
        std::copy(payload_digest.begin(), payload_digest.end(),
                  bytes.begin() + static_cast<ptrdiff_t>(descriptor + 24));
    }
    if (bytes.size() > kMaximumImageBytes - 32 ||
        bytes.size() > SIZE_MAX - 32) {
        return state_error("state image exceeds the wire bound");
    }
    const uint64_t total_bytes = bytes.size() + 32;
    std::copy(kMagic.begin(), kMagic.end(), bytes.begin());
    put_u16(bytes, 8, kMajor);
    put_u16(bytes, 10, kMinor);
    put_u32(bytes, 12, kHeaderBytes);
    put_u64(bytes, 16, total_bytes);
    put_u64(bytes, 24, generation_);
    put_u32(bytes, 32, static_cast<uint32_t>(schema_.slots_.size()));
    put_u32(bytes, 36, kSlotDescriptorBytes);
    put_u32(bytes, 40, static_cast<uint32_t>(schema_.cells_.size()));
    put_u32(bytes, 44, kCellDescriptorBytes);
    std::copy(schema_.program_digest_.bytes.begin(),
              schema_.program_digest_.bytes.end(), bytes.begin() + 48);
    std::copy(schema_.digest_.bytes.begin(), schema_.digest_.bytes.end(),
              bytes.begin() + 80);
    const auto file_digest = sha256(bytes);
    bytes.insert(bytes.end(), file_digest.begin(), file_digest.end());
    return bytes;
}

StateRootResult restore_state_root(const VerifiedStateSchema& schema,
                                   std::span<const uint8_t> bytes) {
    if (bytes.size() < kHeaderBytes + 32 ||
        bytes.size() > kMaximumImageBytes ||
        !std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
        return state_error("state image magic or length is invalid");
    }
    uint16_t major = 0;
    uint16_t minor = 0;
    uint32_t header_bytes = 0;
    uint32_t slot_count = 0;
    uint32_t slot_descriptor_bytes = 0;
    uint32_t cell_count = 0;
    uint32_t cell_descriptor_bytes = 0;
    uint64_t total_bytes = 0;
    uint64_t generation = 0;
    if (!read_u16(bytes, 8, &major) || !read_u16(bytes, 10, &minor) ||
        !read_u32(bytes, 12, &header_bytes) ||
        !read_u64(bytes, 16, &total_bytes) ||
        !read_u64(bytes, 24, &generation) ||
        !read_u32(bytes, 32, &slot_count) ||
        !read_u32(bytes, 36, &slot_descriptor_bytes) ||
        !read_u32(bytes, 40, &cell_count) ||
        !read_u32(bytes, 44, &cell_descriptor_bytes) ||
        major != kMajor || minor != kMinor || header_bytes != kHeaderBytes ||
        slot_descriptor_bytes != kSlotDescriptorBytes ||
        cell_descriptor_bytes != kCellDescriptorBytes ||
        total_bytes != bytes.size() || slot_count != schema.slots_.size() ||
        cell_count != schema.cells_.size() ||
        !all_zero(bytes, 112, kHeaderBytes) ||
        !std::equal(schema.program_digest_.bytes.begin(),
                    schema.program_digest_.bytes.end(), bytes.begin() + 48) ||
        !std::equal(schema.digest_.bytes.begin(), schema.digest_.bytes.end(),
                    bytes.begin() + 80)) {
        return state_error("state image header does not match the schema");
    }
    const auto file_digest = sha256(bytes.first(bytes.size() - 32));
    if (!std::equal(file_digest.begin(), file_digest.end(), bytes.end() - 32)) {
        return state_error("state image digest does not match");
    }

    uint64_t slot_table_bytes = 0;
    uint64_t cell_table_bytes = 0;
    uint64_t cell_table = 0;
    uint64_t payload_floor = 0;
    if (!checked_multiply(slot_count, slot_descriptor_bytes,
                          &slot_table_bytes) ||
        !checked_multiply(cell_count, cell_descriptor_bytes,
                          &cell_table_bytes) ||
        !checked_add(header_bytes, slot_table_bytes, &cell_table) ||
        !checked_add(cell_table, cell_table_bytes, &payload_floor) ||
        payload_floor > bytes.size() - 32) {
        return state_error("state image descriptor tables are invalid");
    }

    std::vector<uint32_t> slot_ranks;
    std::vector<uint64_t> cell_byte_counts;
    slot_ranks.reserve(schema.slots_.size());
    cell_byte_counts.reserve(schema.cells_.size());
    for (const VerifiedStateSlot& slot : schema.slots_) {
        slot_ranks.push_back(static_cast<uint32_t>(slot.extents.size()));
    }
    for (const VerifiedStateCell& cell : schema.cells_) {
        cell_byte_counts.push_back(cell.byte_count);
    }
    uint64_t canonical_wire_bytes = 0;
    if (!wire_size(slot_ranks, cell_byte_counts, &canonical_wire_bytes) ||
        canonical_wire_bytes != bytes.size()) {
        return state_error("state image is not canonically sized");
    }

    for (uint32_t index = 0; index < slot_count; ++index) {
        const VerifiedStateSlot& expected = schema.slots_[index];
        const uint64_t descriptor =
            header_bytes + uint64_t{index} * slot_descriptor_bytes;
        uint32_t ordinal = UINT32_MAX;
        uint32_t cell = UINT32_MAX;
        uint16_t rank = 0;
        uint64_t byte_count = 0;
        if (!read_u32(bytes, descriptor, &ordinal) || ordinal != index ||
            !read_u32(bytes, descriptor + 4, &cell) ||
            cell != expected.canonical_cell_ordinal ||
            bytes[static_cast<size_t>(descriptor + 8)] !=
                static_cast<uint8_t>(expected.element_type) ||
            bytes[static_cast<size_t>(descriptor + 9)] !=
                (expected.writable ? 1 : 0) ||
            !read_u16(bytes, descriptor + 10, &rank) ||
            rank != expected.extents.size() ||
            !all_zero(bytes, descriptor + 12, descriptor + 16) ||
            !read_u64(bytes, descriptor + 272, &byte_count) ||
            byte_count != expected.byte_count ||
            !all_zero(bytes, descriptor + 280, descriptor + 288)) {
            return state_error("state logical descriptor does not match");
        }
        for (uint32_t extent = 0; extent < kMaximumRank; ++extent) {
            uint64_t actual = 0;
            if (!read_u64(bytes, descriptor + 16 + uint64_t{extent} * 8,
                          &actual) ||
                actual !=
                    (extent < expected.extents.size()
                         ? expected.extents[extent]
                         : 0)) {
                return state_error("state logical extent does not match");
            }
        }
    }

    std::vector<StateCellValue> restored;
    restored.reserve(cell_count);
    uint64_t previous_end = payload_floor;
    for (uint32_t index = 0; index < cell_count; ++index) {
        const VerifiedStateCell& expected = schema.cells_[index];
        const uint64_t descriptor =
            cell_table + uint64_t{index} * cell_descriptor_bytes;
        uint32_t ordinal = UINT32_MAX;
        uint64_t byte_count = 0;
        uint64_t payload_offset = 0;
        uint64_t expected_offset = 0;
        if (!align_up(previous_end, kAlignment, &expected_offset)) {
            return state_error("state payload alignment overflows");
        }
        if (!read_u32(bytes, descriptor, &ordinal) || ordinal != index ||
            bytes[static_cast<size_t>(descriptor + 4)] !=
                (expected.writable ? 1 : 0) ||
            !all_zero(bytes, descriptor + 5, descriptor + 8) ||
            !read_u64(bytes, descriptor + 8, &byte_count) ||
            byte_count != expected.byte_count ||
            !read_u64(bytes, descriptor + 16, &payload_offset) ||
            !all_zero(bytes, descriptor + 56, descriptor + 64) ||
            payload_offset != expected_offset ||
            !all_zero(bytes, previous_end, payload_offset) ||
            payload_offset > bytes.size() - 32 ||
            byte_count > bytes.size() - 32 - payload_offset) {
            return state_error("state cell descriptor is invalid");
        }
        const auto payload =
            bytes.subspan(static_cast<size_t>(payload_offset),
                          static_cast<size_t>(byte_count));
        const auto payload_digest = sha256(payload);
        if (!std::equal(payload_digest.begin(), payload_digest.end(),
                        bytes.begin() +
                            static_cast<ptrdiff_t>(descriptor + 24))) {
            return state_error("state cell payload digest does not match");
        }
        const uint32_t representative =
            schema.slots_[expected.slot_indices.front()].state_reference_id;
        restored.push_back(
            {representative,
             std::vector<uint8_t>(payload.begin(), payload.end())});
        if (!checked_add(payload_offset, byte_count, &previous_end)) {
            return state_error("state cell payload extent overflows");
        }
    }
    if (previous_end != bytes.size() - 32) {
        return state_error("state image has unbound trailing bytes");
    }
    StateRootResult result = make_state_root(schema, std::move(restored));
    if (!std::holds_alternative<StateRoot>(result)) return result;
    StateRoot root = std::get<StateRoot>(std::move(result));
    root.generation_ = generation;
    return root;
}

#ifdef LAPLACE_TESTING
std::vector<ProgramStateDigestChunk>
program_state_digest_chunks_for_testing(uint64_t length) {
    std::vector<ProgramStateDigestChunk> chunks;
    for (uint64_t offset = 0; offset < length;) {
        const uint64_t chunk = std::min(kDigestChunkBytes, length - offset);
        chunks.push_back({offset, chunk});
        offset += chunk;
    }
    return chunks;
}

bool program_state_wire_size_for_testing(
    std::span<const uint32_t> slot_ranks,
    std::span<const uint64_t> cell_byte_counts, uint64_t* wire_bytes) {
    return wire_bytes != nullptr &&
           wire_size(slot_ranks, cell_byte_counts, wire_bytes);
}
#endif

} // namespace Laplace
