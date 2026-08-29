#include "closed_v1_source_schema.h"

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Laplace {
namespace {

using IdSet = std::unordered_set<uint32_t>;
using SymbolValues = std::unordered_map<uint32_t, uint64_t>;

bool checked_add(uint64_t left, uint64_t right, uint64_t& result) {
    if (right > std::numeric_limits<uint64_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool checked_mul(uint64_t left, uint64_t right, uint64_t& result) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) return false;
    result = left * right;
    return true;
}

bool add_estimate(uint64_t& estimate, uint64_t amount) {
    uint64_t next = 0;
    if (!checked_add(estimate, amount, next)) return false;
    estimate = next;
    return true;
}

bool add_size_estimate(uint64_t& estimate, size_t amount) {
    return add_estimate(estimate, static_cast<uint64_t>(amount));
}

bool add_aligned_estimate(uint64_t& estimate, uint64_t amount) {
    const uint64_t padding = (8 - (amount % 8)) % 8;
    return add_estimate(estimate, amount) && add_estimate(estimate, padding);
}

struct ClosedV1ExpansionEstimate {
    uint64_t steps = 0;
    uint64_t encoded_bytes = 0;
    uint64_t digest_bytes = 0;
};

bool estimate_closed_v1_work(const ClosedV1Schema& schema,
                             const ClosedV1SourcePackage& package,
                             ClosedV1ExpansionEstimate& estimate) {
    const ClosedV1GraphTemplate& graph = schema.graph;
    const auto add_vector_items = [&](size_t count) {
        return add_size_estimate(estimate.steps, count);
    };
    if (!add_vector_items(schema.predicates.size()) ||
        !add_vector_items(graph.tokenizer.predicates.size()) ||
        !add_vector_items(graph.prompt.predicates.size()) ||
        !add_vector_items(graph.stop_ids.size()) ||
        !add_vector_items(graph.symbols.size()) ||
        !add_vector_items(graph.tensors.size()) ||
        !add_vector_items(graph.values.size()) ||
        !add_vector_items(graph.operators.size()) ||
        !add_vector_items(graph.layers.size()) ||
        !add_vector_items(graph.states.size()) ||
        !add_vector_items(graph.aliases.size()) ||
        !add_vector_items(package.facts.size()) ||
        !add_vector_items(package.source.artifacts.size()) ||
        !add_vector_items(package.source.tensors.size()) ||
        !add_vector_items(package.source.alias_proofs.size())) {
        return false;
    }
    for (const ClosedV1TensorTemplate& tensor : graph.tensors) {
        if (!add_vector_items(tensor.dimensions.size()) ||
            !add_vector_items(tensor.source_dimensions.size()) ||
            !add_vector_items(tensor.source_names.size()) ||
            !add_vector_items(tensor.control.has_variant ? 1 : 0)) return false;
        for (const SourceNamePattern& pattern : tensor.source_names) {
            if (!add_size_estimate(estimate.steps, pattern.literal.size()) ||
                !add_size_estimate(estimate.steps, pattern.prefix.size()) ||
                !add_size_estimate(estimate.steps, pattern.suffix.size())) return false;
        }
    }
    for (const ClosedV1ValueTemplate& value : graph.values)
        if (!add_vector_items(value.dimensions.size()) ||
            !add_vector_items(value.control.has_variant ? 1 : 0)) return false;
    for (const ClosedV1OperatorTemplate& op : graph.operators) {
        if (!add_vector_items(op.inputs.size()) || !add_vector_items(op.outputs.size()) ||
            !add_vector_items(op.tensors.size()) || !add_vector_items(op.states.size()) ||
            !add_vector_items(op.control.has_variant ? 1 : 0)) return false;
    }
    for (const ClosedV1LayerTemplate& layer : graph.layers)
        if (!add_vector_items(layer.control.has_variant ? 1 : 0)) return false;
    for (const ClosedV1StateTemplate& state : graph.states)
        if (!add_vector_items(state.dimensions.size()) || !add_vector_items(state.formats.size()) ||
            !add_vector_items(state.control.has_variant ? 1 : 0)) return false;
    for (const SourceTensorEvidence& tensor : package.source.tensors) {
        if (!add_size_estimate(estimate.steps, tensor.dimensions.size()) ||
            !add_size_estimate(estimate.steps, tensor.source_spelling.size())) return false;
    }

    // Charge each output byte before the compiler allocates its wire or digest
    // buffers. The closed V1 model emits one values plane per tensor and no
    // constraints, capabilities, or fallbacks.
    uint64_t stop_bytes = 0;
    if (!checked_mul(graph.stop_ids.size(), 4, stop_bytes) ||
        !add_estimate(estimate.encoded_bytes, 64 + 136) ||
        !add_estimate(estimate.encoded_bytes, stop_bytes) ||
        !add_estimate(estimate.encoded_bytes, 7)) return false;
    for (const ClosedV1TensorTemplate& tensor : graph.tensors) {
        uint64_t record = 207;
        uint64_t dimensions = 0;
        if (!checked_mul(tensor.dimensions.size(), 16, dimensions) ||
            !checked_add(record, dimensions, record) ||
            !add_aligned_estimate(estimate.encoded_bytes, record)) return false;
    }
    for (const ClosedV1ValueTemplate& value : graph.values) {
        uint64_t record = 15;
        uint64_t dimensions = 0;
        if (!checked_mul(value.dimensions.size(), 16, dimensions) ||
            !checked_add(record, dimensions, record) ||
            !add_aligned_estimate(estimate.encoded_bytes, record)) return false;
    }
    for (const ClosedV1OperatorTemplate& op : graph.operators) {
        uint64_t ids = 0;
        uint64_t record = 32;
        if (!checked_add(op.inputs.size(), op.outputs.size(), ids) ||
            !checked_add(ids, op.tensors.size(), ids) ||
            !checked_add(ids, op.states.size(), ids) ||
            !checked_mul(ids, 4, ids) || !checked_add(record, ids, record) ||
            !checked_add(record, 40, record) ||
            !add_aligned_estimate(estimate.encoded_bytes, record)) return false;
    }
    for (size_t layer_index = 0; layer_index != graph.layers.size(); ++layer_index)
        if (!add_estimate(estimate.encoded_bytes, 16)) return false;
    for (const ClosedV1StateTemplate& state : graph.states) {
        uint64_t record = 23;
        uint64_t dimensions = 0;
        uint64_t formats = 0;
        if (!checked_mul(state.dimensions.size(), 16, dimensions) ||
            !checked_mul(state.formats.size(), 36, formats) ||
            !checked_add(record, dimensions, record) || !checked_add(record, formats, record) ||
            !add_aligned_estimate(estimate.encoded_bytes, record)) return false;
    }
    // The graph proof's structural digest omits physical spans and source
    // spellings, while the wire estimate above includes those fixed-size
    // records. Use the larger wire bound for its input as well, so its
    // growable digest buffer is covered before proof construction starts.
    estimate.digest_bytes = estimate.encoded_bytes;
    return true;
}

ClosedV1SourceSchemaResult failure(ClosedV1SchemaError error, uint32_t schema_id,
                                   CanonicalFactKey fact_key = {}, uint32_t id = kClosedV1NoId,
                                   uint32_t source_tensor_id = kClosedV1NoId) {
    ClosedV1SourceSchemaResult result;
    result.error = error;
    result.schema_id = schema_id;
    result.fact_key = fact_key;
    result.id = id;
    result.source_tensor_id = source_tensor_id;
    return result;
}

enum class FactMatch : uint8_t { Match, Missing, Wrong, Ambiguous };

FactMatch fact_matches(const ClosedV1FactPredicate& predicate,
                       std::span<const ClosedV1FactPredicate> facts) {
    const ClosedV1FactPredicate* found = nullptr;
    for (const ClosedV1FactPredicate& fact : facts) {
        if (fact.key != predicate.key) continue;
        if (found != nullptr) return FactMatch::Ambiguous;
        found = &fact;
    }
    if (found == nullptr) return FactMatch::Missing;
    return found->value == predicate.value ? FactMatch::Match : FactMatch::Wrong;
}

FactMatch fact_value(CanonicalFactKey key, std::span<const ClosedV1FactPredicate> facts,
                    uint64_t& value) {
    const ClosedV1FactPredicate* found = nullptr;
    for (const ClosedV1FactPredicate& fact : facts) {
        if (fact.key != key) continue;
        if (found != nullptr) return FactMatch::Ambiguous;
        found = &fact;
    }
    if (found == nullptr) return FactMatch::Missing;
    const auto* scalar = std::get_if<uint64_t>(&found->value);
    if (scalar == nullptr) return FactMatch::Wrong;
    value = *scalar;
    return FactMatch::Match;
}

bool valid_id(uint32_t id) {
    return id != kClosedV1NoId;
}

bool dimensions_equal(std::span<const Dimension> left, std::span<const Dimension> right) {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin());
}

bool physical_bytes_match(std::span<const Dimension> dimensions, uint32_t block_axis,
                          uint32_t block_elements, uint32_t block_bytes,
                          uint64_t span_length) {
    if (dimensions.empty() || block_axis >= dimensions.size() || block_elements == 0 ||
        block_bytes == 0) return false;
    uint64_t elements = 1;
    for (const Dimension& dimension : dimensions) {
        if (dimension.kind != DimensionKind::Constant ||
            !checked_mul(elements, dimension.constant_or_symbol, elements)) return false;
    }
    if (dimensions[block_axis].constant_or_symbol % block_elements != 0) return false;
    uint64_t expected = 0;
    return checked_mul(elements / block_elements, block_bytes, expected) &&
           expected == span_length;
}

template <typename T, typename GetId>
ClosedV1SchemaError dense_ids(std::span<const T> records, GetId get_id, uint32_t schema_id,
                              uint32_t& bad_id) {
    IdSet seen;
    seen.reserve(records.size());
    for (size_t index = 0; index != records.size(); ++index) {
        const uint32_t id = get_id(records[index]);
        if (!valid_id(id) || !seen.insert(id).second || id != index) {
            bad_id = id;
            return ClosedV1SchemaError::DuplicateId;
        }
    }
    (void)schema_id;
    return ClosedV1SchemaError::None;
}

ClosedV1SchemaError validate_repeat(const ClosedV1Repeat& repeat) {
    if (repeat.count == 0 || repeat.maximum == 0 || repeat.count > repeat.maximum || repeat.count != 1) {
        return ClosedV1SchemaError::RepeatBoundExceeded;
    }
    return ClosedV1SchemaError::None;
}

ClosedV1SchemaError resolve_symbols(const ClosedV1GraphTemplate& graph,
                                    std::span<const ClosedV1FactPredicate> facts,
                                    SymbolValues& values, CanonicalFactKey& bad_fact,
                                    uint32_t& bad_id) {
    IdSet seen;
    seen.reserve(graph.symbols.size());
    for (const ClosedV1Symbol& symbol : graph.symbols) {
        if (!valid_id(symbol.id) || !seen.insert(symbol.id).second || symbol.id == 0) {
            bad_id = symbol.id;
            return ClosedV1SchemaError::DuplicateId;
        }
        uint64_t value = symbol.literal;
        if (symbol.fact_key.value != UINT32_MAX) {
            const FactMatch match = fact_value(symbol.fact_key, facts, value);
            if (match == FactMatch::Missing) {
                bad_fact = symbol.fact_key;
                return ClosedV1SchemaError::MissingFact;
            }
            if (match == FactMatch::Wrong) {
                bad_fact = symbol.fact_key;
                return ClosedV1SchemaError::WrongFact;
            }
            if (match == FactMatch::Ambiguous) {
                bad_fact = symbol.fact_key;
                return ClosedV1SchemaError::AmbiguousSchema;
            }
        } else if (value == 0) {
            bad_id = symbol.id;
            return ClosedV1SchemaError::InvalidTemplate;
        }
        if (symbol.minimum == 0 || symbol.minimum > symbol.maximum || value < symbol.minimum ||
            value > symbol.maximum) {
            bad_id = symbol.id;
            return ClosedV1SchemaError::InvalidTemplate;
        }
        values.emplace(symbol.id, value);
    }
    return ClosedV1SchemaError::None;
}

ClosedV1SchemaError resolve_dimensions(std::span<const ClosedV1Dimension> input,
                                      const SymbolValues& symbols,
                                      std::vector<Dimension>& output, uint32_t& bad_id) {
    if (input.empty() || input.size() > 8) return ClosedV1SchemaError::InvalidTemplate;
    output.clear();
    output.reserve(input.size());
    for (const ClosedV1Dimension& dimension : input) {
        uint64_t extent = dimension.value;
        if (dimension.kind == DimensionKind::Symbol) {
            const auto found = symbols.find(dimension.value);
            if (found == symbols.end()) {
                bad_id = static_cast<uint32_t>(dimension.value);
                return ClosedV1SchemaError::InvalidReference;
            }
            extent = found->second;
        } else if (dimension.kind != DimensionKind::Constant) {
            return ClosedV1SchemaError::InvalidTemplate;
        }
        if (extent == 0) return ClosedV1SchemaError::InvalidTemplate;
        output.push_back({DimensionKind::Constant, extent});
    }
    return ClosedV1SchemaError::None;
}

bool controls_match(const ClosedV1TemplateControl& control,
                    std::span<const ClosedV1FactPredicate> facts) {
    if (control.has_variant && fact_matches(control.variant, facts) != FactMatch::Match) return false;
    return validate_repeat(control.repeat) == ClosedV1SchemaError::None;
}

ClosedV1SchemaError validate_controls(const ClosedV1GraphTemplate& graph,
                                      std::span<const ClosedV1FactPredicate> facts) {
    if (validate_repeat(graph.repeat) != ClosedV1SchemaError::None) {
        return ClosedV1SchemaError::RepeatBoundExceeded;
    }
    for (const ClosedV1TensorTemplate& value : graph.tensors)
        if (!controls_match(value.control, facts)) return value.control.has_variant
            ? ClosedV1SchemaError::NoMatchingSchema : ClosedV1SchemaError::RepeatBoundExceeded;
    for (const ClosedV1ValueTemplate& value : graph.values)
        if (!controls_match(value.control, facts)) return value.control.has_variant
            ? ClosedV1SchemaError::NoMatchingSchema : ClosedV1SchemaError::RepeatBoundExceeded;
    for (const ClosedV1OperatorTemplate& value : graph.operators)
        if (!controls_match(value.control, facts)) return value.control.has_variant
            ? ClosedV1SchemaError::NoMatchingSchema : ClosedV1SchemaError::RepeatBoundExceeded;
    for (const ClosedV1LayerTemplate& value : graph.layers)
        if (!controls_match(value.control, facts)) return value.control.has_variant
            ? ClosedV1SchemaError::NoMatchingSchema : ClosedV1SchemaError::RepeatBoundExceeded;
    for (const ClosedV1StateTemplate& value : graph.states)
        if (!controls_match(value.control, facts)) return value.control.has_variant
            ? ClosedV1SchemaError::NoMatchingSchema : ClosedV1SchemaError::RepeatBoundExceeded;
    return ClosedV1SchemaError::None;
}

ClosedV1SchemaError validate_predicates(std::span<const ClosedV1FactPredicate> predicates,
                                        std::span<const ClosedV1FactPredicate> facts,
                                        CanonicalFactKey& bad_fact) {
    for (const ClosedV1FactPredicate& predicate : predicates) {
        const FactMatch match = fact_matches(predicate, facts);
        if (match == FactMatch::Missing) {
            bad_fact = predicate.key;
            return ClosedV1SchemaError::MissingFact;
        }
        if (match == FactMatch::Wrong) {
            bad_fact = predicate.key;
            return ClosedV1SchemaError::WrongFact;
        }
        if (match == FactMatch::Ambiguous) {
            bad_fact = predicate.key;
            return ClosedV1SchemaError::AmbiguousSchema;
        }
    }
    return ClosedV1SchemaError::None;
}

ClosedV1SourceSchemaResult compile_matched(const ClosedV1Schema& schema,
                                           const ClosedV1SourcePackage& package) {
    const ClosedV1GraphTemplate& graph = schema.graph;
    if (schema.schema_major != 1 || schema.schema_minor != 0 || graph.schema_major != 1 ||
        graph.schema_minor != 0 || graph.opset_major != 1 || graph.opset_minor != 0) {
        return failure(ClosedV1SchemaError::SchemaVersionUnsupported, schema.schema_id);
    }
    if (schema.source_format != package.source.format) {
        return failure(ClosedV1SchemaError::SourceFormatMismatch, schema.schema_id);
    }
    if (schema.maximum_expansion_steps == 0 || schema.maximum_expansion_steps > 4'000'000) {
        return failure(ClosedV1SchemaError::RepeatBoundExceeded, schema.schema_id);
    }
    if (graph.tokenizer.vocabulary_size == 0)
        return failure(ClosedV1SchemaError::InvalidTemplate, schema.schema_id);
    ClosedV1ExpansionEstimate estimate;
    if (!estimate_closed_v1_work(schema, package, estimate) ||
        estimate.steps > schema.maximum_expansion_steps ||
        estimate.encoded_bytes > schema.maximum_expansion_steps ||
        estimate.digest_bytes > schema.maximum_expansion_steps) {
        return failure(ClosedV1SchemaError::RepeatBoundExceeded, schema.schema_id);
    }

    CanonicalFactKey bad_fact{};
    uint32_t bad_id = kClosedV1NoId;
    ClosedV1SchemaError fact_error = validate_predicates(graph.tokenizer.predicates, package.facts, bad_fact);
    if (fact_error != ClosedV1SchemaError::None)
        return failure(fact_error, schema.schema_id, bad_fact);
    fact_error = validate_predicates(graph.prompt.predicates, package.facts, bad_fact);
    if (fact_error != ClosedV1SchemaError::None)
        return failure(fact_error, schema.schema_id, bad_fact);
    const ClosedV1SchemaError control_error = validate_controls(graph, package.facts);
    if (control_error != ClosedV1SchemaError::None) {
        return failure(control_error, schema.schema_id, bad_fact, bad_id);
    }
    SymbolValues symbols;
    const ClosedV1SchemaError symbol_error =
        resolve_symbols(graph, package.facts, symbols, bad_fact, bad_id);
    if (symbol_error != ClosedV1SchemaError::None) {
        return failure(symbol_error, schema.schema_id, bad_fact, bad_id);
    }

    ClosedV1SchemaError id_error = dense_ids<ClosedV1TensorTemplate>(
        graph.tensors, [](const auto& value) { return value.id; }, schema.schema_id, bad_id);
    if (id_error != ClosedV1SchemaError::None) return failure(id_error, schema.schema_id, {}, bad_id);
    id_error = dense_ids<ClosedV1ValueTemplate>(
        graph.values, [](const auto& value) { return value.id; }, schema.schema_id, bad_id);
    if (id_error != ClosedV1SchemaError::None) return failure(id_error, schema.schema_id, {}, bad_id);
    id_error = dense_ids<ClosedV1OperatorTemplate>(
        graph.operators, [](const auto& value) { return value.id; }, schema.schema_id, bad_id);
    if (id_error != ClosedV1SchemaError::None) return failure(id_error, schema.schema_id, {}, bad_id);
    id_error = dense_ids<ClosedV1StateTemplate>(
        graph.states, [](const auto& value) { return value.id; }, schema.schema_id, bad_id);
    if (id_error != ClosedV1SchemaError::None) return failure(id_error, schema.schema_id, {}, bad_id);
    id_error = dense_ids<ClosedV1LayerTemplate>(
        graph.layers, [](const auto& value) { return value.layer_index; }, schema.schema_id, bad_id);
    if (id_error != ClosedV1SchemaError::None) return failure(id_error, schema.schema_id, {}, bad_id);

    if (graph.tensors.empty() || graph.values.empty() || graph.operators.empty() ||
        graph.layers.empty() || graph.states.empty()) {
        return failure(ClosedV1SchemaError::InvalidTemplate, schema.schema_id);
    }
    if (graph.tensors.size() + graph.values.size() + graph.operators.size() + graph.layers.size() +
            graph.states.size() + graph.symbols.size() > schema.maximum_expansion_steps) {
        return failure(ClosedV1SchemaError::RepeatBoundExceeded, schema.schema_id);
    }

    SourceSchema binding_schema;
    binding_schema.source_format = schema.source_format;
    binding_schema.maximum_expansion_steps = schema.maximum_expansion_steps;
    IdSet source_slots;
    for (size_t index = 0; index != graph.tensors.size(); ++index) {
        const ClosedV1TensorTemplate& tensor = graph.tensors[index];
        if (!valid_id(tensor.source_slot_id) || !source_slots.insert(tensor.source_slot_id).second ||
            tensor.source_names.empty()) {
            return failure(ClosedV1SchemaError::InvalidTemplate, schema.schema_id, {}, tensor.id);
        }
        std::vector<Dimension> logical_dimensions;
        std::vector<Dimension> source_dimensions;
        ClosedV1SchemaError error = resolve_dimensions(tensor.dimensions, symbols, logical_dimensions, bad_id);
        if (error != ClosedV1SchemaError::None)
            return failure(error, schema.schema_id, {}, bad_id);
        const auto source_input = tensor.source_dimensions.empty()
            ? tensor.dimensions : tensor.source_dimensions;
        error = resolve_dimensions(source_input, symbols, source_dimensions, bad_id);
        if (error != ClosedV1SchemaError::None)
            return failure(error, schema.schema_id, {}, bad_id);
        if (!dimensions_equal(logical_dimensions, source_dimensions) ||
            tensor.physical_block_axis >= source_dimensions.size()) {
            return failure(ClosedV1SchemaError::PhysicalContractMismatch,
                           schema.schema_id, {}, tensor.id);
        }
        std::vector<uint64_t> source_extents;
        source_extents.reserve(source_dimensions.size());
        for (const Dimension& dimension : source_dimensions) source_extents.push_back(dimension.constant_or_symbol);
        bool source_output = index == 0;
        for (const ClosedV1AliasTemplate& alias : graph.aliases)
            source_output = (source_output && alias.source_slot_id != tensor.source_slot_id) ||
                            alias.target_slot_id == tensor.source_slot_id;
        binding_schema.slots.push_back({tensor.source_slot_id, tensor.source_slot_id, source_extents, true,
                                        source_output});
        binding_schema.selectors.push_back({tensor.source_slot_id, tensor.source_slot_id, {}, tensor.source_names,
                                            source_extents, 1});
    }
    for (const ClosedV1AliasTemplate& alias : graph.aliases) {
        if (!source_slots.contains(alias.source_slot_id) || !source_slots.contains(alias.target_slot_id) ||
            alias.source_slot_id == alias.target_slot_id) {
            return failure(ClosedV1SchemaError::InvalidReference, schema.schema_id);
        }
        binding_schema.aliases.push_back({alias.source_slot_id, alias.target_slot_id, alias.kind, alias.proof_key});
    }
    const SourceSchemaResult binding = compile_source_schema(binding_schema, package.source);
    if (!binding.success()) {
        if (binding.error == SourceSchemaError::UnmatchedTensor)
            return failure(ClosedV1SchemaError::UnmatchedSourceTensor, schema.schema_id, {},
                           kClosedV1NoId, binding.tensor_id);
        if (binding.error == SourceSchemaError::DuplicateId)
            return failure(ClosedV1SchemaError::DuplicateId, schema.schema_id, {}, binding.tensor_id);
        return failure(ClosedV1SchemaError::MissingTensorBinding, schema.schema_id, {},
                       binding.slot_id, binding.tensor_id);
    }

    SemanticModel model;
    model.schema_major = graph.schema_major;
    model.schema_minor = graph.schema_minor;
    model.opset_major = graph.opset_major;
    model.opset_minor = graph.opset_minor;
    model.maximum_context = graph.maximum_context;
    model.vocabulary_size = graph.tokenizer.vocabulary_size;
    model.bos_id = graph.bos_id;
    model.eos_id = graph.eos_id;
    model.stop_ids = graph.stop_ids;
    model.tokenizer_digest = graph.tokenizer.digest;
    model.template_digest = graph.prompt.digest;
    model.input_values_first = graph.input_values_first;
    model.input_values_count = graph.input_values_count;
    model.output_values_first = graph.output_values_first;
    model.output_values_count = graph.output_values_count;

    model.tensors.reserve(graph.tensors.size());
    for (const ClosedV1TensorTemplate& tensor : graph.tensors) {
        std::vector<Dimension> dimensions;
        const ClosedV1SchemaError error = resolve_dimensions(tensor.dimensions, symbols, dimensions, bad_id);
        if (error != ClosedV1SchemaError::None) return failure(error, schema.schema_id, {}, bad_id);
        if (tensor.layout.rank != dimensions.size() || tensor.plane_alignment == 0 ||
            tensor.physical_block_elements == 0 || tensor.physical_block_bytes == 0) {
            return failure(ClosedV1SchemaError::InvalidTemplate, schema.schema_id, {}, tensor.id);
        }
        const auto binding_it = std::find_if(binding.compiled.bindings.begin(), binding.compiled.bindings.end(),
            [&](const CompiledBinding& value) { return value.logical_slot_id == tensor.source_slot_id; });
        if (binding_it == binding.compiled.bindings.end())
            return failure(ClosedV1SchemaError::MissingTensorBinding, schema.schema_id, {}, tensor.id);
        const SourceTensorEvidence* source_tensor = nullptr;
        for (const SourceTensorEvidence& candidate : package.source.tensors) {
            if (candidate.physical_tensor_id == binding_it->physical_tensor_id) {
                source_tensor = &candidate;
                break;
            }
        }
        if (source_tensor == nullptr || source_tensor->physical_type_code != tensor.physical_type_code ||
            source_tensor->physical_block_axis != tensor.physical_block_axis ||
            source_tensor->physical_block_elements != tensor.physical_block_elements ||
            source_tensor->physical_block_bytes != tensor.physical_block_bytes ||
            !physical_bytes_match(dimensions, tensor.physical_block_axis,
                                  tensor.physical_block_elements,
                                  tensor.physical_block_bytes, source_tensor->span.length)) {
            return failure(ClosedV1SchemaError::PhysicalContractMismatch, schema.schema_id, {}, tensor.id,
                           binding_it->physical_tensor_id);
        }
        SemanticTensor output;
        output.id = tensor.id;
        output.role = tensor.role;
        output.logical_type = tensor.logical_type;
        output.dimensions = std::move(dimensions);
        output.layout = tensor.layout;
        output.quantization = tensor.quantization;
        output.flags = tensor.flags;
        output.planes.push_back({PlaneKind::Values, tensor.plane_storage_type, ArtifactId{0},
                                 binding_it->span.offset, binding_it->span.length,
                                 tensor.plane_alignment, 0});
        model.tensors.push_back(std::move(output));
    }

    model.values.reserve(graph.values.size());
    for (const ClosedV1ValueTemplate& value : graph.values) {
        std::vector<Dimension> dimensions;
        const ClosedV1SchemaError error = resolve_dimensions(value.dimensions, symbols, dimensions, bad_id);
        if (error != ClosedV1SchemaError::None) return failure(error, schema.schema_id, {}, bad_id);
        model.values.push_back({value.id, value.logical_type, std::move(dimensions), value.flags});
    }
    model.operators.reserve(graph.operators.size());
    for (const ClosedV1OperatorTemplate& op : graph.operators) {
        model.operators.push_back({op.id, op.kind, op.semantic_version, op.inputs, op.outputs,
                                   op.tensors, op.states, op.payload});
    }
    model.layers.reserve(graph.layers.size());
    for (const ClosedV1LayerTemplate& layer : graph.layers)
        model.layers.push_back({layer.layer_index, layer.first_operator, layer.operator_count, layer.flags});
    model.states.reserve(graph.states.size());
    for (const ClosedV1StateTemplate& state : graph.states) {
        std::vector<Dimension> dimensions;
        const ClosedV1SchemaError error = resolve_dimensions(state.dimensions, symbols, dimensions, bad_id);
        if (error != ClosedV1SchemaError::None) return failure(error, schema.schema_id, {}, bad_id);
        model.states.push_back({state.id, state.kind, state.semantic_version, state.update_kind,
                                state.position_policy, std::move(dimensions), state.formats, state.flags});
    }

    if (!std::holds_alternative<std::vector<uint8_t>>(encode_semantic_model(model))) {
        return failure(ClosedV1SchemaError::GraphProofFailed, schema.schema_id);
    }
    const SourceCompilerGraphProofResult proof_result = prove_source_candidate_graph(model);
    if (const auto* proof = std::get_if<SourceCompilerGraphProof>(&proof_result)) {
        ClosedV1SourceSchemaResult result;
        result.schema_id = schema.schema_id;
        result.compiled = ClosedV1SourceCompilation{std::move(model), *proof};
        return result;
    }
    if (const auto* report = std::get_if<CompatibilityReport>(&proof_result)) {
        if (report->detail.find("cycle") != std::string::npos)
            return failure(ClosedV1SchemaError::GraphCycle, schema.schema_id, {}, report->operator_id);
        if (report->code == CompatibilityError::IR_REFERENCE_INVALID)
            return failure(ClosedV1SchemaError::InvalidReference, schema.schema_id, {}, report->operator_id);
    }
    return failure(ClosedV1SchemaError::GraphProofFailed, schema.schema_id);
}

} // namespace

ClosedV1SourceSchemaResult compile_closed_v1_source_schema(
    const ClosedV1Schema& schema, const ClosedV1SourcePackage& package) {
    if (schema.source_format != package.source.format)
        return failure(ClosedV1SchemaError::SourceFormatMismatch, schema.schema_id);
    for (const ClosedV1FactPredicate& predicate : schema.predicates) {
        const FactMatch match = fact_matches(predicate, package.facts);
        if (match == FactMatch::Missing)
            return failure(ClosedV1SchemaError::MissingFact, schema.schema_id, predicate.key);
        if (match == FactMatch::Wrong)
            return failure(ClosedV1SchemaError::WrongFact, schema.schema_id, predicate.key);
        if (match == FactMatch::Ambiguous)
            return failure(ClosedV1SchemaError::AmbiguousSchema, schema.schema_id, predicate.key);
    }
    return compile_matched(schema, package);
}

ClosedV1SourceSchemaResult compile_closed_v1_source_schemas(
    std::span<const ClosedV1Schema> schemas, const ClosedV1SourcePackage& package) {
    ClosedV1SourceSchemaResult winner;
    ClosedV1SourceSchemaResult first_failure;
    bool have_failure = false;
    uint32_t candidates = 0;
    uint32_t matches = 0;
    const auto predicates_match = [&](std::span<const ClosedV1FactPredicate> predicates) {
        for (const ClosedV1FactPredicate& predicate : predicates)
            if (fact_matches(predicate, package.facts) != FactMatch::Match) return false;
        return true;
    };
    for (const ClosedV1Schema& schema : schemas) {
        if (schema.source_format != package.source.format || schema.schema_major != 1 ||
            schema.schema_minor != 0) continue;
        if (!predicates_match(schema.predicates) ||
            !predicates_match(schema.graph.tokenizer.predicates) ||
            !predicates_match(schema.graph.prompt.predicates)) continue;
        ++candidates;
        ClosedV1SourceSchemaResult candidate =
            compile_closed_v1_source_schema(schema, package);
        if (candidate.success()) {
            ++matches;
            if (matches == 1) winner = std::move(candidate);
        } else if (!have_failure) {
            first_failure = std::move(candidate);
            have_failure = true;
        }
    }
    if (matches == 0) {
        if (candidates == 1 && have_failure) return first_failure;
        return failure(ClosedV1SchemaError::NoMatchingSchema, kClosedV1NoId);
    }
    if (matches != 1) return failure(ClosedV1SchemaError::AmbiguousSchema, kClosedV1NoId);
    return winner;
}

} // namespace Laplace
