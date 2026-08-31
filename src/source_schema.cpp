#include "source_schema.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Laplace {
namespace {

constexpr size_t kMaxSchemaBytes = 1u << 20;
constexpr size_t kMaxSourceStringBytes = 4096;
constexpr size_t kMaxSourceTensors = 65536;
constexpr size_t kMaxSlots = 65536;
constexpr size_t kMaxSelectors = 65536;
constexpr size_t kMaxAliases = 65536;
constexpr size_t kMaxSchemas = 64;
constexpr size_t kMaxPatternsPerSelector = 256;
constexpr size_t kMaxRank = 8;
constexpr uint32_t kMaxPatternDigits = 10;
constexpr uint32_t kMaxExpansionSteps = 4'000'000;

SourceSchemaResult failure(SourceSchemaError error, uint32_t selector = kSourceSchemaNoId,
                           uint32_t slot = kSourceSchemaNoId,
                           uint32_t tensor = kSourceSchemaNoId) {
    SourceSchemaResult result;
    result.error = error;
    result.selector_id = selector;
    result.slot_id = slot;
    result.tensor_id = tensor;
    return result;
}

struct WorkBudget {
    size_t remaining;
    explicit WorkBudget(uint32_t maximum) : remaining(maximum) {}
    bool take(size_t amount = 1) {
        if (amount > remaining) return false;
        remaining -= amount;
        return true;
    }
};

bool dimensions_equal(std::span<const uint64_t> left, std::span<const uint64_t> right) {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin());
}

bool checked_mul(uint64_t left, uint64_t right, uint64_t& result) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) return false;
    result = left * right;
    return true;
}

bool checked_add(uint64_t left, uint64_t right, uint64_t& result) {
    if (right > std::numeric_limits<uint64_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool dimensions_valid(std::span<const uint64_t> dimensions) {
    if (dimensions.empty() || dimensions.size() > kMaxRank) return false;
    return std::all_of(dimensions.begin(), dimensions.end(), [](uint64_t dimension) {
        return dimension != 0;
    });
}

bool valid_source_format(SourceFormat format) {
    return format == SourceFormat::Gguf || format == SourceFormat::Mlx;
}

bool has_prefix_and_suffix(std::string_view value, std::string_view prefix, std::string_view suffix) {
    if (value.size() < prefix.size() + suffix.size() ||
        value.substr(0, prefix.size()) != prefix) {
        return false;
    }
    if (suffix.empty()) return true;
    return value.substr(value.size() - suffix.size()) == suffix;
}

struct PatternMatch {
    bool matched = false;
    bool has_capture = false;
    uint32_t capture = 0;
};

PatternMatch pattern_matches(const SourceNamePattern& pattern, std::string_view spelling) {
    if (spelling.size() > kMaxSourceStringBytes) return {};
    if (pattern.kind == SourceNamePatternKind::Exact) {
        return {pattern.prefix.empty() && pattern.suffix.empty() &&
                    !pattern.literal.empty() && pattern.literal.size() <= kMaxSourceStringBytes &&
                    spelling == pattern.literal,
                false, 0};
    }
    if (pattern.kind != SourceNamePatternKind::AnchoredDecimal ||
        pattern.literal.size() != 0 || pattern.prefix.size() > kMaxSourceStringBytes ||
        pattern.suffix.size() > kMaxSourceStringBytes || pattern.maximum_digits == 0 ||
        pattern.maximum_digits > kMaxPatternDigits ||
        !has_prefix_and_suffix(spelling, pattern.prefix, pattern.suffix)) {
        return {};
    }

    const size_t begin = pattern.prefix.size();
    const size_t end = spelling.size() - pattern.suffix.size();
    if (begin >= end) return {};
    const std::string_view digits = spelling.substr(begin, end - begin);
    if (digits.size() > pattern.maximum_digits ||
        (digits.size() > 1 && digits.front() == '0')) {
        return {};
    }
    uint64_t value = 0;
    for (char digit : digits) {
        if (digit < '0' || digit > '9') return {};
        value = value * 10 + static_cast<uint64_t>(digit - '0');
        if (value > std::numeric_limits<uint32_t>::max()) return {};
    }
    return {true, true, static_cast<uint32_t>(value)};
}

std::span<const uint32_t> selector_targets(const SchemaTensorSelector& selector) {
    if (!selector.binding_slots.empty()) return selector.binding_slots;
    return {&selector.binding_slot, 1};
}

uint64_t alias_key(uint32_t left, uint32_t right) {
    const uint32_t first = std::min(left, right);
    const uint32_t second = std::max(left, right);
    return (static_cast<uint64_t>(first) << 32) | second;
}

struct SpanKey {
    uint32_t artifact_id = kSourceSchemaNoId;
    uint64_t offset = 0;
    uint64_t length = 0;
    friend bool operator==(const SpanKey&, const SpanKey&) = default;
};

struct SpanKeyHash {
    size_t operator()(const SpanKey& key) const noexcept {
        size_t hash = static_cast<size_t>(key.artifact_id) * 0x9e3779b1u;
        hash ^= static_cast<size_t>(key.offset ^ (key.offset >> 32)) + 0x9e3779b9u +
                (hash << 6) + (hash >> 2);
        hash ^= static_cast<size_t>(key.length ^ (key.length >> 32)) + 0x9e3779b9u +
                (hash << 6) + (hash >> 2);
        return hash;
    }
};

bool append_byte(std::vector<uint8_t>& bytes, uint8_t value) {
    if (bytes.size() >= kMaxSchemaBytes) return false;
    bytes.push_back(value);
    return true;
}

bool append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    if (bytes.size() > kMaxSchemaBytes - sizeof(uint32_t)) return false;
    for (unsigned shift = 0; shift != 32; shift += 8) {
        bytes.push_back(static_cast<uint8_t>(value >> shift));
    }
    return true;
}

bool append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
    if (bytes.size() > kMaxSchemaBytes - sizeof(uint64_t)) return false;
    for (unsigned shift = 0; shift != 64; shift += 8) {
        bytes.push_back(static_cast<uint8_t>(value >> shift));
    }
    return true;
}

bool append_count(std::vector<uint8_t>& bytes, size_t value, WorkBudget& budget) {
    return budget.take() && value <= std::numeric_limits<uint32_t>::max() &&
           append_u32(bytes, static_cast<uint32_t>(value));
}

struct BindingWork {
    bool bound = false;
    size_t tensor_index = 0;
};

bool alias_is_declared(const std::unordered_set<uint64_t>& aliases,
                       uint32_t left, uint32_t right) {
    return aliases.find(alias_key(left, right)) != aliases.end();
}

size_t alias_root(std::vector<size_t>& parent, size_t index) {
    while (parent[index] != index) {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    return index;
}

SourceSchemaResult compile_one(const SourceSchema& schema,
                               const SourcePackageEvidence& package) {
    const SourceSchemaVersion schema_version{schema.schema_major, schema.schema_minor};
    const SourceSchemaVersion evaluator_version{schema.evaluator_major, schema.evaluator_minor};
    if (schema_version.major != SourceSchema::version().major ||
        schema_version.minor > SourceSchema::version().minor) {
        return failure(SourceSchemaError::SchemaVersionUnsupported);
    }
    if (evaluator_version.major != SourceSchemaEvaluator::version().major ||
        evaluator_version.minor > SourceSchemaEvaluator::version().minor) {
        return failure(SourceSchemaError::EvaluatorVersionUnsupported);
    }
    if (!valid_source_format(schema.source_format) || !valid_source_format(package.format) ||
        schema.source_format != package.format) {
        return failure(SourceSchemaError::SourceFormatMismatch);
    }
    if (schema.maximum_expansion_steps == 0 ||
        schema.maximum_expansion_steps > kMaxExpansionSteps ||
        package.artifacts.size() > kMaxSourceTensors ||
        package.tensors.size() > kMaxSourceTensors || schema.slots.size() > kMaxSlots ||
        schema.selectors.size() > kMaxSelectors || schema.aliases.size() > kMaxAliases) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    WorkBudget budget(schema.maximum_expansion_steps);

    std::unordered_map<uint32_t, uint64_t> artifact_sizes;
    if (!budget.take(package.artifacts.size())) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    artifact_sizes.reserve(package.artifacts.size());
    for (const SourceArtifactEvidence& artifact : package.artifacts) {
        if (!budget.take()) return failure(SourceSchemaError::SchemaLimitExceeded);
        if (artifact.artifact_id == kSourceSchemaNoId || artifact.byte_length == 0 ||
            !artifact_sizes.emplace(artifact.artifact_id, artifact.byte_length).second) {
            return failure(SourceSchemaError::InvalidPhysicalSpan);
        }
    }

    std::unordered_map<uint32_t, size_t> tensor_by_id;
    if (!budget.take(package.tensors.size())) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    tensor_by_id.reserve(package.tensors.size());
    if (!budget.take(package.tensors.size() * 2)) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    std::vector<uint64_t> tensor_ends(package.tensors.size());
    for (size_t tensor_index = 0; tensor_index != package.tensors.size(); ++tensor_index) {
        const SourceTensorEvidence& tensor = package.tensors[tensor_index];
        if (!budget.take(1 + tensor.dimensions.size())) {
            return failure(SourceSchemaError::SchemaLimitExceeded, kSourceSchemaNoId,
                           kSourceSchemaNoId, tensor.physical_tensor_id);
        }
        if (tensor.physical_tensor_id == kSourceSchemaNoId) {
            return failure(SourceSchemaError::DuplicateId, kSourceSchemaNoId,
                           kSourceSchemaNoId, tensor.physical_tensor_id);
        }
        if (!tensor_by_id.emplace(tensor.physical_tensor_id, tensor_index).second) {
            return failure(SourceSchemaError::DuplicateId, kSourceSchemaNoId,
                           kSourceSchemaNoId, tensor.physical_tensor_id);
        }
        if (tensor.source_spelling.size() > kMaxSourceStringBytes ||
            tensor.dimensions.size() > kMaxRank ||
            tensor.physical_block_axis >= tensor.dimensions.size() ||
            tensor.physical_block_elements == 0 || tensor.physical_block_bytes == 0 ||
            !dimensions_valid(tensor.dimensions)) {
            return failure(SourceSchemaError::InvalidPhysicalSpan, kSourceSchemaNoId,
                           kSourceSchemaNoId, tensor.physical_tensor_id);
        }
        const auto artifact = artifact_sizes.find(tensor.span.artifact_id);
        uint64_t elements = 1;
        for (uint64_t dimension : tensor.dimensions) {
            if (!checked_mul(elements, dimension, elements)) {
                return failure(SourceSchemaError::InvalidPhysicalSpan, kSourceSchemaNoId,
                               kSourceSchemaNoId, tensor.physical_tensor_id);
            }
        }
        uint64_t expected_bytes = 0;
        uint64_t end = 0;
        if (artifact == artifact_sizes.end() || tensor.span.length == 0 ||
            tensor.dimensions[tensor.physical_block_axis] %
                    tensor.physical_block_elements != 0 ||
            !checked_mul(elements / tensor.physical_block_elements,
                         tensor.physical_block_bytes, expected_bytes) ||
            expected_bytes != tensor.span.length ||
            !checked_add(tensor.span.offset, tensor.span.length, end) ||
            end > artifact->second) {
            return failure(SourceSchemaError::InvalidPhysicalSpan, kSourceSchemaNoId,
                           kSourceSchemaNoId, tensor.physical_tensor_id);
        }
        tensor_ends[tensor_index] = end;
    }

    std::vector<size_t> span_order(package.tensors.size());
    for (size_t index = 0; index != span_order.size(); ++index) span_order[index] = index;
    if (!budget.take(span_order.size() * 8)) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    std::sort(span_order.begin(), span_order.end(), [&](size_t left, size_t right) {
        const PhysicalSpan& a = package.tensors[left].span;
        const PhysicalSpan& b = package.tensors[right].span;
        return std::tie(a.artifact_id, a.offset, a.length) <
               std::tie(b.artifact_id, b.offset, b.length);
    });
    for (size_t index = 1; index != span_order.size(); ++index) {
        const size_t previous = span_order[index - 1];
        const size_t current = span_order[index];
        const PhysicalSpan& previous_span = package.tensors[previous].span;
        const PhysicalSpan& current_span = package.tensors[current].span;
        if (previous_span.artifact_id == current_span.artifact_id &&
            tensor_ends[previous] > current_span.offset && previous_span != current_span) {
            return failure(SourceSchemaError::InvalidPhysicalSpan, kSourceSchemaNoId,
                           kSourceSchemaNoId, package.tensors[current].physical_tensor_id);
        }
    }

    std::unordered_map<uint32_t, size_t> slot_by_id;
    if (!budget.take(schema.slots.size())) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    slot_by_id.reserve(schema.slots.size());
    bool has_required_output = false;
    for (size_t slot_index = 0; slot_index != schema.slots.size(); ++slot_index) {
        const SchemaLogicalSlot& slot = schema.slots[slot_index];
        if (!budget.take(1 + slot.dimensions.size())) {
            return failure(SourceSchemaError::SchemaLimitExceeded, kSourceSchemaNoId, slot.id);
        }
        if (slot.id == kSourceSchemaNoId ||
            !dimensions_valid(slot.dimensions) ||
            !slot_by_id.emplace(slot.id, slot_index).second) {
            return failure(SourceSchemaError::DuplicateId, kSourceSchemaNoId, slot.id);
        }
        if (slot.output && !slot.required) {
            return failure(SourceSchemaError::MissingOutput, kSourceSchemaNoId, slot.id);
        }
        has_required_output |= slot.output && slot.required;
    }
    if (!has_required_output) return failure(SourceSchemaError::MissingOutput);

    std::unordered_map<uint32_t, SourceAliasProofEvidence> proofs;
    if (!budget.take(package.alias_proofs.size())) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    proofs.reserve(package.alias_proofs.size());
    for (const SourceAliasProofEvidence& proof : package.alias_proofs) {
        if (!budget.take()) return failure(SourceSchemaError::SchemaLimitExceeded);
        if (proof.proof_key == kSourceSchemaNoId ||
            !proofs.emplace(proof.proof_key, proof).second) {
            return failure(SourceSchemaError::AliasInvalid);
        }
    }

    std::unordered_map<uint32_t, size_t> selector_by_id;
    if (!budget.take(schema.selectors.size())) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    selector_by_id.reserve(schema.selectors.size());
    std::vector<int32_t> slot_selector(schema.slots.size(), -1);
    size_t schema_string_bytes = 0;
    for (size_t selector_index = 0; selector_index != schema.selectors.size(); ++selector_index) {
        const SchemaTensorSelector& selector = schema.selectors[selector_index];
        const std::span<const uint32_t> targets = selector_targets(selector);
        if (!budget.take(1 + selector.names.size() + selector.source_dimensions.size())) {
            return failure(SourceSchemaError::SchemaLimitExceeded, selector.id);
        }
        if (selector.id == kSourceSchemaNoId ||
            !selector_by_id.emplace(selector.id, selector_index).second ||
            selector.names.empty() || selector.exact_cardinality == 0 ||
            selector.exact_cardinality > kMaxSourceTensors ||
            selector.names.size() > kMaxPatternsPerSelector ||
            !dimensions_valid(selector.source_dimensions) || targets.empty() ||
            targets.size() != selector.exact_cardinality ||
            (selector.binding_slots.empty() && selector.binding_slot == kSourceSchemaNoId) ||
            (!selector.binding_slots.empty() && selector.binding_slot != kSourceSchemaNoId)) {
            return failure(SourceSchemaError::InvalidSelector, selector.id, selector.binding_slot);
        }
        if (targets.size() > 1 &&
            (selector.names.size() != 1 ||
             selector.names.front().kind != SourceNamePatternKind::AnchoredDecimal)) {
            return failure(SourceSchemaError::InvalidSelector, selector.id);
        }
        std::unordered_set<uint32_t> target_ids;
        target_ids.reserve(targets.size());
        for (uint32_t slot_id : targets) {
            if (!budget.take()) return failure(SourceSchemaError::SchemaLimitExceeded, selector.id);
            const auto slot = slot_by_id.find(slot_id);
            if (slot == slot_by_id.end() || !target_ids.insert(slot_id).second ||
                slot_selector[slot->second] != -1 ||
                (schema.slots[slot->second].selector_id != kSourceSchemaNoId &&
                 schema.slots[slot->second].selector_id != selector.id) ||
                !dimensions_equal(selector.source_dimensions, schema.slots[slot->second].dimensions)) {
                return failure(SourceSchemaError::InvalidSelector, selector.id, slot_id);
            }
            slot_selector[slot->second] = static_cast<int32_t>(selector_index);
        }
        for (const SourceNamePattern& pattern : selector.names) {
            const bool exact_pattern = pattern.kind == SourceNamePatternKind::Exact;
            const bool decimal_pattern = pattern.kind == SourceNamePatternKind::AnchoredDecimal;
            if (!budget.take(1 + pattern.literal.size() + pattern.prefix.size() +
                             pattern.suffix.size())) {
                return failure(SourceSchemaError::SchemaLimitExceeded, selector.id);
            }
            if ((!exact_pattern && !decimal_pattern) ||
                pattern.literal.size() > kMaxSourceStringBytes ||
                pattern.prefix.size() > kMaxSourceStringBytes ||
                pattern.suffix.size() > kMaxSourceStringBytes ||
                (exact_pattern && (pattern.literal.empty() || !pattern.prefix.empty() ||
                                   !pattern.suffix.empty())) ||
                (decimal_pattern && (!pattern.literal.empty() || pattern.maximum_digits == 0 ||
                                     pattern.maximum_digits > kMaxPatternDigits))) {
                return failure(SourceSchemaError::InvalidSelector, selector.id, selector.binding_slot);
            }
            uint64_t pattern_bytes = static_cast<uint64_t>(pattern.literal.size()) +
                                     pattern.prefix.size() + pattern.suffix.size();
            uint64_t next_string_bytes = 0;
            if (!checked_add(schema_string_bytes, pattern_bytes, next_string_bytes) ||
                next_string_bytes > kMaxSchemaBytes) {
                return failure(SourceSchemaError::SchemaLimitExceeded, selector.id);
            }
            schema_string_bytes = static_cast<size_t>(next_string_bytes);
        }
    }

    for (size_t slot_index = 0; slot_index != schema.slots.size(); ++slot_index) {
        const uint32_t selector_id = schema.slots[slot_index].selector_id;
        if (selector_id == kSourceSchemaNoId) continue;
        const auto selector_index = selector_by_id.find(selector_id);
        if (!budget.take() || selector_index == selector_by_id.end() ||
            slot_selector[slot_index] != static_cast<int32_t>(selector_index->second)) {
            return failure(SourceSchemaError::InvalidSelector, selector_id,
                           schema.slots[slot_index].id);
        }
    }

    if (!budget.take(schema.slots.size() + package.tensors.size())) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    std::vector<BindingWork> bindings(schema.slots.size());
    std::vector<std::vector<size_t>> tensor_assignments(package.tensors.size());
    for (size_t selector_index = 0; selector_index != schema.selectors.size(); ++selector_index) {
        const SchemaTensorSelector& selector = schema.selectors[selector_index];
        const std::span<const uint32_t> targets = selector_targets(selector);
        std::vector<size_t> matches;
        std::vector<PatternMatch> captures;
        if (!budget.take(selector.exact_cardinality * 2)) {
            return failure(SourceSchemaError::SchemaLimitExceeded, selector.id);
        }
        matches.reserve(selector.exact_cardinality);
        captures.reserve(selector.exact_cardinality);
        for (size_t tensor_index = 0; tensor_index != package.tensors.size(); ++tensor_index) {
            const SourceTensorEvidence& tensor = package.tensors[tensor_index];
            if (!budget.take(1 + selector.names.size() + tensor.source_spelling.size())) {
                return failure(SourceSchemaError::SchemaLimitExceeded, selector.id);
            }
            PatternMatch matched;
            bool found = false;
            for (const SourceNamePattern& pattern : selector.names) {
                const PatternMatch current = pattern_matches(pattern, tensor.source_spelling);
                if (!current.matched) continue;
                if (found && (matched.has_capture != current.has_capture ||
                              (matched.has_capture && matched.capture != current.capture))) {
                    return failure(SourceSchemaError::SelectorNotMatched, selector.id,
                                   kSourceSchemaNoId, tensor.physical_tensor_id);
                }
                matched = current;
                found = true;
            }
            if (found) {
                matches.push_back(tensor_index);
                captures.push_back(matched);
            }
        }
        if (matches.size() != selector.exact_cardinality) {
            return failure(SourceSchemaError::SelectorCardinalityMismatch, selector.id);
        }

        std::vector<size_t> ordered_matches;
        if (targets.size() > 1) {
            if (!budget.take(targets.size())) {
                return failure(SourceSchemaError::SchemaLimitExceeded, selector.id);
            }
            ordered_matches.assign(targets.size(), package.tensors.size());
            for (size_t match_index = 0; match_index != matches.size(); ++match_index) {
                const PatternMatch capture = captures[match_index];
                if (!capture.has_capture || capture.capture >= targets.size() ||
                    ordered_matches[capture.capture] != package.tensors.size()) {
                    return failure(SourceSchemaError::SelectorNotMatched, selector.id);
                }
                ordered_matches[capture.capture] = matches[match_index];
            }
            if (!budget.take(ordered_matches.size())) {
                return failure(SourceSchemaError::SchemaLimitExceeded, selector.id);
            }
            if (std::find(ordered_matches.begin(), ordered_matches.end(), package.tensors.size()) !=
                ordered_matches.end()) {
                return failure(SourceSchemaError::SelectorCardinalityMismatch, selector.id);
            }
        } else {
            ordered_matches = std::move(matches);
        }
        for (size_t target_index = 0; target_index != targets.size(); ++target_index) {
            if (!budget.take()) return failure(SourceSchemaError::SchemaLimitExceeded, selector.id);
            const size_t tensor_index = ordered_matches[target_index];
            const SourceTensorEvidence& tensor = package.tensors[tensor_index];
            const auto slot = slot_by_id.find(targets[target_index]);
            if (slot == slot_by_id.end() ||
                !dimensions_equal(selector.source_dimensions, tensor.dimensions)) {
                return failure(SourceSchemaError::SelectorNotMatched, selector.id,
                               targets[target_index], tensor.physical_tensor_id);
            }
            if (bindings[slot->second].bound) {
                return failure(SourceSchemaError::TensorConsumedTwice, selector.id,
                               targets[target_index], tensor.physical_tensor_id);
            }
            bindings[slot->second] = {true, tensor_index};
            tensor_assignments[tensor_index].push_back(slot->second);
        }
    }

    std::unordered_set<uint64_t> declared_aliases;
    if (!budget.take(schema.aliases.size() + schema.slots.size() * 3)) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    declared_aliases.reserve(schema.aliases.size());
    std::vector<int32_t> alias_next(schema.slots.size(), -1);
    std::vector<int32_t> alias_previous(schema.slots.size(), -1);
    std::vector<size_t> alias_parent(schema.slots.size());
    for (size_t index = 0; index != alias_parent.size(); ++index) alias_parent[index] = index;
    for (const SchemaAliasTemplate& alias : schema.aliases) {
        if (!budget.take()) return failure(SourceSchemaError::SchemaLimitExceeded);
        const auto source = slot_by_id.find(alias.source_slot_id);
        const auto target = slot_by_id.find(alias.target_slot_id);
        if (source == slot_by_id.end() || target == slot_by_id.end() ||
            alias.source_slot_id == alias.target_slot_id ||
            (alias.kind != SourceAliasKind::ExactSharedSpan &&
             alias.kind != SourceAliasKind::TiedOutput) ||
            !dimensions_equal(schema.slots[source->second].dimensions,
                              schema.slots[target->second].dimensions) ||
            !declared_aliases.insert(alias_key(alias.source_slot_id, alias.target_slot_id)).second ||
            alias_next[source->second] != -1 || alias_previous[target->second] != -1) {
            return failure(SourceSchemaError::AliasInvalid, kSourceSchemaNoId,
                           alias.target_slot_id);
        }
        if (alias.kind == SourceAliasKind::TiedOutput) {
            const auto proof = proofs.find(alias.proof_key);
            if (alias.proof_key == kSourceSchemaNoId || proof == proofs.end() ||
                proof->second.source_slot_id != alias.source_slot_id ||
                proof->second.target_slot_id != alias.target_slot_id ||
                proof->second.kind != alias.kind || schema.slots[source->second].output ||
                !schema.slots[target->second].output || !schema.slots[target->second].required) {
                return failure(SourceSchemaError::AliasInvalid, kSourceSchemaNoId,
                               alias.target_slot_id);
            }
        }
        alias_next[source->second] = static_cast<int32_t>(target->second);
        alias_previous[target->second] = static_cast<int32_t>(source->second);
        const size_t source_root = alias_root(alias_parent, source->second);
        const size_t target_root = alias_root(alias_parent, target->second);
        alias_parent[target_root] = source_root;
    }
    for (size_t start = 0; start != alias_next.size(); ++start) {
        size_t hops = 0;
        int32_t cursor = static_cast<int32_t>(start);
        while (cursor != -1) {
            if (!budget.take() || hops++ > schema.aliases.size()) {
                return failure(SourceSchemaError::SchemaLimitExceeded);
            }
            cursor = alias_next[static_cast<size_t>(cursor)];
            if (cursor == static_cast<int32_t>(start)) {
                return failure(SourceSchemaError::AliasInvalid, kSourceSchemaNoId,
                               schema.slots[start].id);
            }
        }
    }

    // Aliases are declarations, not an inference pass. A short fixed-point loop
    // permits a bounded alias chain while refusing an unbound source.
    for (size_t pass = 0; pass != schema.aliases.size() + 1; ++pass) {
        bool changed = false;
        for (const SchemaAliasTemplate& alias : schema.aliases) {
            if (!budget.take()) return failure(SourceSchemaError::SchemaLimitExceeded);
            const size_t source_slot = slot_by_id.find(alias.source_slot_id)->second;
            const size_t target_slot = slot_by_id.find(alias.target_slot_id)->second;
            if (!bindings[source_slot].bound) continue;
            if (!bindings[target_slot].bound) {
                bindings[target_slot] = bindings[source_slot];
                changed = true;
            } else if (package.tensors[bindings[source_slot].tensor_index].span !=
                package.tensors[bindings[target_slot].tensor_index].span) {
                return failure(SourceSchemaError::AliasMismatch, kSourceSchemaNoId,
                               alias.target_slot_id,
                               package.tensors[bindings[target_slot].tensor_index].physical_tensor_id);
            }
        }
        if (!changed) break;
    }

    for (const SchemaAliasTemplate& alias : schema.aliases) {
        const size_t source_slot = slot_by_id.find(alias.source_slot_id)->second;
        if (!bindings[source_slot].bound) {
            return failure(SourceSchemaError::AliasInvalid, kSourceSchemaNoId,
                           alias.source_slot_id);
        }
    }

    for (size_t slot_index = 0; slot_index != schema.slots.size(); ++slot_index) {
        const SchemaLogicalSlot& slot = schema.slots[slot_index];
        if (!bindings[slot_index].bound && slot.required) {
            return failure(slot.output ? SourceSchemaError::MissingOutput
                                       : SourceSchemaError::MissingRequiredSlot,
                           slot.selector_id, slot.id);
        }
    }

    if (!budget.take(package.tensors.size())) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    std::vector<bool> final_tensor_use(package.tensors.size(), false);
    for (const BindingWork& binding : bindings) {
        if (binding.bound) final_tensor_use[binding.tensor_index] = true;
    }
    for (size_t tensor_index = 0; tensor_index != package.tensors.size(); ++tensor_index) {
        if (!final_tensor_use[tensor_index]) {
            return failure(SourceSchemaError::UnmatchedTensor, kSourceSchemaNoId,
                           kSourceSchemaNoId, package.tensors[tensor_index].physical_tensor_id);
        }
        const auto& assignments = tensor_assignments[tensor_index];
        for (size_t left = 0; left != assignments.size(); ++left) {
            for (size_t right = left + 1; right != assignments.size(); ++right) {
                if (!budget.take()) return failure(SourceSchemaError::SchemaLimitExceeded);
                const uint32_t left_id = schema.slots[assignments[left]].id;
                const uint32_t right_id = schema.slots[assignments[right]].id;
                if (!alias_is_declared(declared_aliases, left_id, right_id)) {
                    return failure(SourceSchemaError::AliasRequired, kSourceSchemaNoId,
                                   right_id, package.tensors[tensor_index].physical_tensor_id);
                }
            }
        }
    }

    std::unordered_map<SpanKey, size_t, SpanKeyHash> used_spans;
    if (!budget.take(schema.aliases.size() + 1)) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    if (!budget.take(bindings.size())) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    used_spans.reserve(bindings.size());
    for (size_t slot_index = 0; slot_index != bindings.size(); ++slot_index) {
        if (!bindings[slot_index].bound) continue;
        if (!budget.take()) return failure(SourceSchemaError::SchemaLimitExceeded);
        const SourceTensorEvidence& tensor = package.tensors[bindings[slot_index].tensor_index];
        const SpanKey span{tensor.span.artifact_id, tensor.span.offset, tensor.span.length};
        const auto existing = used_spans.find(span);
        if (existing != used_spans.end() &&
            alias_root(alias_parent, existing->second) != alias_root(alias_parent, slot_index)) {
            return failure(SourceSchemaError::AliasRequired, kSourceSchemaNoId,
                           schema.slots[slot_index].id, tensor.physical_tensor_id);
        }
        used_spans.emplace(span, slot_index);
    }

    SourceSchemaResult result;
    if (!budget.take(schema.slots.size() * 2 + schema.aliases.size())) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    result.compiled.graph_slots.reserve(schema.slots.size());
    result.compiled.bindings.reserve(schema.slots.size());
    for (size_t slot_index = 0; slot_index != schema.slots.size(); ++slot_index) {
        if (!bindings[slot_index].bound) continue;
        if (!budget.take(2)) return failure(SourceSchemaError::SchemaLimitExceeded);
        const SchemaLogicalSlot& slot = schema.slots[slot_index];
        const SourceTensorEvidence& tensor = package.tensors[bindings[slot_index].tensor_index];
        result.compiled.graph_slots.push_back({slot.id, slot.dimensions, slot.required, slot.output});
        result.compiled.bindings.push_back({slot.id, tensor.physical_tensor_id, tensor.span,
                                            tensor.physical_type_code,
                                            tensor.physical_block_axis,
                                            tensor.physical_block_elements,
                                            tensor.physical_block_bytes});
    }
    for (const SchemaAliasTemplate& alias : schema.aliases) {
        if (!budget.take()) return failure(SourceSchemaError::SchemaLimitExceeded);
        result.compiled.aliases.push_back({alias.source_slot_id, alias.target_slot_id,
                                           alias.kind, alias.proof_key});
    }
    if (!budget.take(result.compiled.graph_slots.size() * 8) ||
        !budget.take(result.compiled.bindings.size() * 8) ||
        !budget.take(result.compiled.aliases.size() * 8)) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    std::sort(result.compiled.graph_slots.begin(), result.compiled.graph_slots.end(),
              [](const CompiledGraphSlot& left, const CompiledGraphSlot& right) {
                  return left.logical_slot_id < right.logical_slot_id;
              });
    std::sort(result.compiled.bindings.begin(), result.compiled.bindings.end(),
              [](const CompiledBinding& left, const CompiledBinding& right) {
                  return left.logical_slot_id < right.logical_slot_id;
              });
    std::sort(result.compiled.aliases.begin(), result.compiled.aliases.end(),
              [](const CompiledAlias& left, const CompiledAlias& right) {
                  return std::pair{left.source_slot_id, left.target_slot_id} <
                         std::pair{right.source_slot_id, right.target_slot_id};
              });

    if (!append_count(result.compiled.graph_bytes, result.compiled.graph_slots.size(), budget)) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    for (const CompiledGraphSlot& slot : result.compiled.graph_slots) {
        if (!append_u32(result.compiled.graph_bytes, slot.logical_slot_id) ||
            !append_count(result.compiled.graph_bytes, slot.dimensions.size(), budget) ||
            !append_byte(result.compiled.graph_bytes, slot.required ? 1 : 0) ||
            !append_byte(result.compiled.graph_bytes, slot.output ? 1 : 0)) {
            return failure(SourceSchemaError::SchemaLimitExceeded);
        }
        for (uint64_t dimension : slot.dimensions) {
            if (!budget.take() || !append_u64(result.compiled.graph_bytes, dimension)) {
                return failure(SourceSchemaError::SchemaLimitExceeded);
            }
        }
    }
    if (!append_count(result.compiled.graph_bytes, result.compiled.aliases.size(), budget)) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    for (const CompiledAlias& alias : result.compiled.aliases) {
        if (!budget.take() || !append_u32(result.compiled.graph_bytes, alias.source_slot_id) ||
            !append_u32(result.compiled.graph_bytes, alias.target_slot_id) ||
            !append_byte(result.compiled.graph_bytes, static_cast<uint8_t>(alias.kind))) {
            return failure(SourceSchemaError::SchemaLimitExceeded);
        }
    }

    if (!append_count(result.compiled.binding_bytes, result.compiled.bindings.size(), budget)) {
        return failure(SourceSchemaError::SchemaLimitExceeded);
    }
    for (const CompiledBinding& binding : result.compiled.bindings) {
        if (!budget.take() || !append_u32(result.compiled.binding_bytes, binding.logical_slot_id) ||
            !append_u32(result.compiled.binding_bytes, binding.physical_tensor_id) ||
            !append_u32(result.compiled.binding_bytes, binding.span.artifact_id) ||
            !append_u64(result.compiled.binding_bytes, binding.span.offset) ||
            !append_u64(result.compiled.binding_bytes, binding.span.length) ||
            !append_u32(result.compiled.binding_bytes, binding.physical_type_code) ||
            !append_u32(result.compiled.binding_bytes, binding.physical_block_axis) ||
            !append_u32(result.compiled.binding_bytes, binding.physical_block_elements) ||
            !append_u32(result.compiled.binding_bytes, binding.physical_block_bytes)) {
            return failure(SourceSchemaError::SchemaLimitExceeded);
        }
    }
    result.error = SourceSchemaError::None;
    result.matching_schema_count = 1;
    return result;
}

} // namespace

SourceSchemaResult SourceSchemaEvaluator::evaluate(const SourceSchema& schema,
                                                   const SourcePackageEvidence& package) {
    return compile_one(schema, package);
}

SourceSchemaResult SourceSchemaEvaluator::evaluate(std::span<const SourceSchema> schemas,
                                                   const SourcePackageEvidence& package) {
    if (schemas.size() > kMaxSchemas) return failure(SourceSchemaError::SchemaLimitExceeded);
    SourceSchemaResult winner;
    uint32_t matches = 0;
    for (const SourceSchema& schema : schemas) {
        SourceSchemaResult candidate = compile_one(schema, package);
        if (candidate.success()) {
            ++matches;
            if (matches == 1) winner = std::move(candidate);
            continue;
        }
        switch (candidate.error) {
        case SourceSchemaError::SchemaVersionUnsupported:
        case SourceSchemaError::EvaluatorVersionUnsupported:
        case SourceSchemaError::SchemaLimitExceeded:
        case SourceSchemaError::DuplicateId:
        case SourceSchemaError::InvalidSelector:
        case SourceSchemaError::InvalidPhysicalSpan:
        case SourceSchemaError::MissingOutput:
        case SourceSchemaError::AliasInvalid:
            candidate.matching_schema_count = matches;
            return candidate;
        default:
            break;
        }
    }
    if (matches == 0) return failure(SourceSchemaError::NoMatchingSchema);
    if (matches != 1) {
        SourceSchemaResult ambiguous = failure(SourceSchemaError::AmbiguousSchemas);
        ambiguous.matching_schema_count = matches;
        return ambiguous;
    }
    winner.matching_schema_count = matches;
    return winner;
}

SourceSchemaResult compile_source_schema(const SourceSchema& schema,
                                         const SourcePackageEvidence& package) {
    return SourceSchemaEvaluator::evaluate(schema, package);
}

SourceSchemaResult evaluate_source_schemas(std::span<const SourceSchema> schemas,
                                           const SourcePackageEvidence& package) {
    return SourceSchemaEvaluator::evaluate(schemas, package);
}

} // namespace Laplace
