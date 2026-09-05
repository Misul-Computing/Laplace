#include "semantic_program_compiler.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace Laplace {
namespace {

using Shape = std::vector<DimensionExpr>;
using Map = TensorIndexMap;
using Expr = TensorIndexExpr;

[[noreturn]] void fail(const char* detail) {
    auto report = package_report(CompatibilityError::IR_SHAPE_MISMATCH);
    report.stage = CompatibilityStage::Semantic;
    report.detail = detail;
    throw report;
}
DimensionExpr dim(uint64_t n) { return {DimensionExpression::Constant, n, {}}; }
Expr index(uint32_t n) { return {TensorIndexExpression::Iterator, n, {}}; }
Expr literal(int64_t n) { return {TensorIndexExpression::Constant, n, {}}; }
Expr expr(TensorIndexExpression op, Expr a, Expr b) {
    return {op, 0, {std::move(a), std::move(b)}};
}
Map identity(size_t rank) {
    Map m;
    for (size_t i = 0; i < rank; ++i) m.results.push_back(index(i));
    return m;
}
float number(uint32_t bits) { return std::bit_cast<float>(bits); }

class Compiler {
    const SemanticModel& model;
    std::unordered_map<uint64_t, uint64_t> dimensions;
    std::unordered_map<uint32_t, ValueType> value_types, tensor_types;
    std::unordered_map<uint32_t, TypedValue> values, tensors;
    std::vector<SemanticProgramBinding> tensor_bindings, input_bindings;
    std::unordered_set<uint32_t> finite_sources;
    Region entry{1, {}, {}, {}};
    std::vector<Region> bodies;
    uint32_t next = 2;
    uint32_t capacity;
    std::vector<StateReference> state_references;
    std::unordered_map<uint32_t, TypedValue> cache_values;
    std::unordered_map<uint32_t, uint32_t> cache_state_ids;
    std::unordered_set<uint32_t> written_caches;
    uint32_t last_effect = UINT32_MAX, cursor_state = UINT32_MAX;
    TypedValue cursor;

    uint32_t id() {
        if (next == UINT32_MAX) fail("semantic program exceeds SSA ID space");
        return next++;
    }
    ValueType type(ScalarType scalar, const std::vector<Dimension>& source) {
        ElementType element;
        switch (scalar) {
        case ScalarType::F32: element = ElementType::F32; break;
        case ScalarType::F16: element = ElementType::F16; break;
        case ScalarType::U32: element = ElementType::U32; break;
        case ScalarType::I32: element = ElementType::I32; break;
        default: fail("semantic scalar type has no ProgramIR representation");
        }
        ValueType result{element, {}};
        uint64_t count = 1;
        for (const auto& d : source) {
            uint64_t n = d.constant_or_symbol;
            if (d.kind == DimensionKind::Symbol) {
                auto found = dimensions.find(n);
                if (found == dimensions.end()) fail("unbound semantic dimension");
                n = found->second;
            } else if (d.kind != DimensionKind::Constant) fail("invalid semantic dimension kind");
            if (!n || n > INT64_MAX || count > uint64_t(INT64_MAX) / n)
                fail("invalid or overflowing semantic shape");
            count *= n;
            result.dimensions.push_back(dim(n));
        }
        return result;
    }
    TypedValue constant(Region& r, ValueType t, uint64_t bits) {
        TypedValue v{id(), std::move(t)};
        r.instructions.push_back({id(), {Primitive::Constant, 1, 0}, {}, {v}, {}, {}, ConstantAttributes{bits}});
        return v;
    }
    TypedValue scalar(Region& r, float value) {
        return constant(r, {ElementType::F32, {}}, std::bit_cast<uint32_t>(value));
    }
    TypedValue operation(Region& r, Primitive op, std::initializer_list<TypedValue> args,
                         ElementType result = ElementType::F32) {
        TypedValue v{id(), {result, {}}};
        Instruction item{id(), {op, 1, 0}, {}, {v}, {}, {}, NoAttributes{}};
        for (const auto& arg : args) item.inputs.push_back(arg.id);
        if (result == ElementType::F32 && op != Primitive::Select && op != Primitive::Convert && op != Primitive::RequireFinite &&
            op != Primitive::Less && op != Primitive::Equal && op != Primitive::Require)
            item.attributes = ArithmeticAttributes{true, true};
        r.instructions.push_back(std::move(item));
        return v;
    }
    using Body = std::function<TypedValue(Region&, const std::vector<TypedValue>&)>;
    TypedValue stage(const std::vector<TypedValue>& sources, std::vector<Map> maps,
                     ValueType output, Shape iterations, Body body,
                     TypedValue initial = {UINT32_MAX, {} }) {
        if (initial.id == UINT32_MAX) initial = constant(entry, output, 0);
        StructuredTensorAttributes attrs;
        attrs.source_count = sources.size();
        attrs.iteration_dimensions = std::move(iterations);
        if (attrs.iteration_dimensions.empty()) attrs.iteration_dimensions.push_back(dim(1));
        attrs.iterator_kinds.assign(output.dimensions.size(), TensorIteratorKind::Parallel);
        attrs.iterator_kinds.resize(attrs.iteration_dimensions.size(), TensorIteratorKind::Reduction);
        attrs.indexing_maps = std::move(maps);
        attrs.indexing_maps.push_back(identity(output.dimensions.size()));
        Region region{id(), {}, {}, {}};
        for (const auto& s : sources) region.arguments.push_back({id(), {s.type.element_type, {}}});
        region.arguments.push_back({id(), {output.element_type, {}}});
        auto arguments = region.arguments;
        for (size_t i = 0; i < sources.size(); ++i)
            if (finite_sources.contains(sources[i].id))
                arguments[i] = operation(region, Primitive::RequireFinite, {arguments[i]});
        region.yields = {body(region, arguments).id};
        TypedValue result{id(), output};
        Instruction item{id(), {Primitive::StructuredTensor, 1, 0}, {}, {result}, {region.id}, {}, attrs};
        for (const auto& s : sources) item.inputs.push_back(s.id);
        item.inputs.push_back(initial.id);
        entry.instructions.push_back(std::move(item));
        bodies.push_back(std::move(region));
        return result;
    }
    TypedValue elementwise(const std::vector<TypedValue>& sources, Body body) {
        if (sources.empty()) fail("missing elementwise source");
        std::vector<Map> maps(sources.size(), identity(sources.front().type.dimensions.size()));
        for (const auto& s : sources) if (s.type != sources.front().type) fail("elementwise shape/type mismatch");
        return stage(sources, maps, sources.front().type, sources.front().type.dimensions, body);
    }
    TypedValue finite(TypedValue source) {
        finite_sources.insert(source.id);
        return source;
    }
    TypedValue floating(TypedValue source) {
        if (source.type.element_type != ElementType::F32)
            fail("semantic arithmetic requires F32 values");
        return source;
    }
    TypedValue input(uint32_t semantic_id) {
        auto found = values.find(semantic_id);
        if (found == values.end()) fail("semantic input is not available in graph order");
        return found->second;
    }
    TypedValue tensor(uint32_t semantic_id) {
        if (auto found = tensors.find(semantic_id); found != tensors.end()) return found->second;
        auto found = tensor_types.find(semantic_id);
        if (found == tensor_types.end()) fail("unknown semantic tensor ID");
        TypedValue v{id(), found->second};
        entry.arguments.push_back(v);
        tensor_bindings.push_back({semantic_id, v.id});
        v = finite(floating(v));
        tensors.emplace(semantic_id, v);
        return v;
    }
    void output(uint32_t semantic_id, TypedValue v) {
        const auto found = value_types.find(semantic_id);
        if (found == value_types.end()) fail("unknown semantic output value ID");
        if (found->second != v.type) {
            const auto describe = [](const ValueType& t) {
                std::string result = "type=" + std::to_string(static_cast<unsigned>(t.element_type)) + " [";
                for (const auto& d : t.dimensions) result += std::to_string(d.value) + ",";
                return result + "]";
            };
            const std::string detail = "semantic output value " + std::to_string(semantic_id) +
                " expected " + describe(found->second) + " actual " + describe(v.type);
            fail(detail.c_str());
        }
        if (!values.emplace(semantic_id, v).second) fail("semantic value has multiple definitions");
    }
    static uint64_t width(TypedValue v) {
        if (v.type.dimensions.empty()) fail("semantic operation needs a last axis");
        return v.type.dimensions.back().value;
    }
    TypedValue sigmoid(Region& r, TypedValue x) {
        auto zero = scalar(r, 0), one = scalar(r, 1);
        auto negative = operation(r, Primitive::Less, {x, zero}, ElementType::I1);
        auto minus = operation(r, Primitive::Negate, {x});
        auto exponent = operation(r, Primitive::Exp,
            {operation(r, Primitive::Select, {negative, x, minus})});
        auto numerator = operation(r, Primitive::Select, {negative, exponent, one});
        return operation(r, Primitive::Divide,
            {numerator, operation(r, Primitive::Add, {one, exponent})});
    }
    TypedValue activation(Region& r, TypedValue x, ActivationKind kind) {
        if (kind == ActivationKind::Silu) {
            auto e = operation(r, Primitive::Exp, {operation(r, Primitive::Negate, {x})});
            return operation(r, Primitive::Divide, {x, operation(r, Primitive::Add, {scalar(r, 1), e})});
        }
        if (kind != ActivationKind::GeluTanh) fail("unsupported activation");
        auto cube = operation(r, Primitive::Multiply, {operation(r, Primitive::Multiply, {x, x}), x});
        auto inner = operation(r, Primitive::Multiply, {scalar(r, 0.7978845608028654f),
            operation(r, Primitive::Add, {x, operation(r, Primitive::Multiply, {scalar(r, 0.044715f), cube})})});
        auto t = operation(r, Primitive::Tanh, {inner});
        return operation(r, Primitive::Multiply, {scalar(r, 0.5f), operation(r, Primitive::Multiply,
            {x, operation(r, Primitive::Add, {scalar(r, 1), t})})});
    }

    // Sixteen independent lanes, then lanes in order, then the scalar tail.
    // This is the association specified by semantic_linear/semantic_rms_norm.
    TypedValue sum_products(const std::vector<TypedValue>& sources, ValueType output,
                           uint64_t extent, const std::function<std::vector<Map>(Expr)>& maps,
                           bool lanes) {
        auto sum = constant(entry, output, 0);
        const size_t rank = output.dimensions.size();
        uint64_t base = 0;
        const auto reduce_body = [&](Region& r, const auto& a) {
            return operation(r, Primitive::Add, {a.back(), operation(r, Primitive::Multiply, {a[0], a[1]})});
        };
        if (lanes && extent >= 16) {
            ValueType partial = output; partial.dimensions.push_back(dim(16));
            Shape iterations = partial.dimensions; iterations.push_back(dim(extent / 16));
            auto k = expr(TensorIndexExpression::Add,
                expr(TensorIndexExpression::Multiply, index(rank + 1), literal(16)), index(rank));
            auto lane_sums = stage(sources, maps(k), partial, iterations, reduce_body);
            Shape fold = output.dimensions; fold.push_back(dim(16));
            sum = stage({lane_sums}, {identity(rank + 1)}, output, fold,
                [&](Region& r, const auto& a) { return operation(r, Primitive::Add, {a.back(), a[0]}); }, sum);
            base = extent / 16 * 16;
        }
        if (base < extent) {
            Shape tail = output.dimensions; tail.push_back(dim(extent - base));
            auto k = expr(TensorIndexExpression::Add, literal(base), index(rank));
            sum = stage(sources, maps(k), output, tail, reduce_body, sum);
        }
        return sum;
    }

    TypedValue coordinate(Region& r, uint32_t axis) {
        TypedValue v{id(), {ElementType::U32, {}}};
        r.instructions.push_back({id(), {Primitive::TensorCoordinate, 1, 0}, {}, {v}, {}, {}, CoordinateAttributes{axis}});
        return v;
    }
    TypedValue integer(Region& r, uint32_t value) { return constant(r, {ElementType::U32, {}}, value); }
    TypedValue less(Region& r, TypedValue a, TypedValue b) { return operation(r, Primitive::Less, {a, b}, ElementType::I1); }
    TypedValue equal(Region& r, TypedValue a, TypedValue b) { return operation(r, Primitive::Equal, {a, b}, ElementType::I1); }
    TypedValue choose(Region& r, TypedValue condition, TypedValue yes, TypedValue no) {
        return operation(r, Primitive::Select, {condition, yes, no}, yes.type.element_type);
    }
    TypedValue require(Region& r, TypedValue predicate) { return operation(r, Primitive::Require, {predicate}, ElementType::I1); }
    TypedValue state_read(uint32_t state, ValueType t) {
        TypedValue result{id(), t};
        Instruction item{id(), {Primitive::StateRead, 1, 0}, {}, {result}, {}, {}, StateAttributes{state}};
        if (last_effect != UINT32_MAX) item.effect_predecessors = {last_effect};
        last_effect = item.id; entry.instructions.push_back(std::move(item));
        return result;
    }
    void state_write(uint32_t state, TypedValue value) {
        Instruction item{id(), {Primitive::StateWrite, 1, 0}, {value.id}, {}, {}, {}, StateAttributes{state}};
        if (last_effect != UINT32_MAX) item.effect_predecessors = {last_effect};
        last_effect = item.id; entry.instructions.push_back(std::move(item));
    }
    void initialize_cursor() {
        if (cursor_state != UINT32_MAX) return;
        if (!capacity || (model.maximum_context && capacity > model.maximum_context)) fail("invalid semantic cache capacity");
        cursor_state = id();
        ValueType t{ElementType::U32, {}};
        state_references.push_back({cursor_state, t, cursor_state, true});
        cursor = state_read(cursor_state, t);
    }
    TypedValue cache(uint32_t semantic_id, StateKind kind, uint32_t heads, uint32_t head_dimension) {
        if (auto found = cache_values.find(semantic_id); found != cache_values.end()) {
            if (found->second.type.dimensions != Shape{dim(capacity), dim(heads), dim(head_dimension)})
                fail("semantic cache reused with different geometry");
            return found->second;
        }
        auto found = std::find_if(model.states.begin(), model.states.end(), [&](const auto& s) { return s.id == semantic_id; });
        if (found == model.states.end() || found->kind != kind || found->dimensions.size() != 3 ||
            found->semantic_version < 1 || found->semantic_version > 8 || found->flags != 0 ||
            found->position_policy != PositionPolicy::AppendOnly ||
            found->update_kind != (kind == StateKind::KeyCache ? StateUpdateKind::AppendKey : StateUpdateKind::AppendValue))
            fail("unsupported semantic cache declaration");
        auto dimensions = found->dimensions;
        if (dimensions[0].kind != DimensionKind::Constant && dimensions[0].kind != DimensionKind::Symbol)
            fail("invalid semantic cache capacity dimension");
        if (dimensions[0].kind == DimensionKind::Constant && dimensions[0].constant_or_symbol < capacity)
            fail("semantic cache capacity exceeds declaration");
        dimensions[0] = {DimensionKind::Constant, capacity};
        auto t = type(ScalarType::F32, dimensions);
        if (t.dimensions != Shape{dim(capacity), dim(heads), dim(head_dimension)}) fail("semantic cache dimensions mismatch");
        const uint32_t state = id();
        state_references.push_back({state, t, state, true});
        cache_state_ids.emplace(semantic_id, state);
        auto value = finite(state_read(state, t));
        cache_values.emplace(semantic_id, value);
        return value;
    }
    static Map vector_map(TypedValue source, Expr column) {
        Map result;
        for (size_t i = 0; i + 1 < source.type.dimensions.size(); ++i) {
            if (source.type.dimensions[i].value != 1) fail("stateful semantic execution requires one token row");
            result.results.push_back(literal(0));
        }
        if (source.type.dimensions.empty()) fail("stateful source needs a channel axis");
        result.results.push_back(std::move(column));
        return result;
    }
    TypedValue flatten_heads(TypedValue source, ValueType output, uint32_t head_dimension) {
        Map m; m.results = {
            expr(TensorIndexExpression::FloorDivide, index(output.dimensions.size() - 1), literal(head_dimension)),
            expr(TensorIndexExpression::Remainder, index(output.dimensions.size() - 1), literal(head_dimension))};
        return stage({source}, {m}, output, output.dimensions, [](Region&, const auto& a) { return a[0]; });
    }
    void rope(const SemanticOperator& op, TypedValue query) {
        initialize_cursor();
        const auto& p = std::get<RopePayload>(op.payload);
        if (p.pairing != RopePairing::HalfSplit && p.pairing != RopePairing::Interleaved)
            fail("multi-position RoPE is not implemented");
        uint32_t head_dimension = 0;
        for (const auto& consumer : model.operators) {
            if (consumer.kind != OperatorKind::CausalAttention || !semantic_operator_contract_valid(consumer)) continue;
            const auto& attention = std::get<CausalAttentionPayload>(consumer.payload);
            if (consumer.inputs[0] != op.outputs[0] || consumer.inputs[1] != op.outputs[1]) continue;
            if (head_dimension && head_dimension != attention.head_dimension) fail("ambiguous RoPE head geometry");
            head_dimension = attention.head_dimension;
        }
        if (!head_dimension) fail("RoPE requires an explicit attention head geometry consumer");
        if (p.rotary_dimension > head_dimension) fail("RoPE prefix exceeds head width");
        const uint32_t frequency_dimension = p.frequency_dimension ? p.frequency_dimension : p.rotary_dimension;
        for (size_t operand = 0; operand < 2; ++operand) {
            auto x = operand ? finite(floating(input(op.inputs[1]))) : query;
            if (width(x) % head_dimension) fail("RoPE channels do not divide into heads");
            const uint32_t heads = width(x) / head_dimension;
            ValueType rotated{ElementType::F32, {dim(heads), dim(head_dimension)}};
            auto local = expr(TensorIndexExpression::Remainder, index(1), literal(p.rotary_dimension));
            auto pair = p.pairing == RopePairing::HalfSplit ?
                expr(TensorIndexExpression::Remainder, local, literal(p.rotary_dimension / 2)) :
                expr(TensorIndexExpression::FloorDivide, local, literal(2));
            auto first = p.pairing == RopePairing::HalfSplit ? pair : expr(TensorIndexExpression::Multiply, pair, literal(2));
            auto second = expr(TensorIndexExpression::Add, first, literal(p.pairing == RopePairing::HalfSplit ? p.rotary_dimension / 2 : 1));
            const auto channel = [&](Expr local) { return expr(TensorIndexExpression::Add,
                expr(TensorIndexExpression::Multiply, index(0), literal(head_dimension)), local); };
            // A coordinate tensor makes the pair index available as a scalar
            // without replacing the declared pow schedule with another formula.
            ValueType pair_type{ElementType::U32, {dim(p.rotary_dimension / 2)}};
            auto pair_ids = stage({cursor}, {Map{}}, pair_type, pair_type.dimensions,
                [&](Region& r, const auto&) { return coordinate(r, 0); });
            Map pm; pm.results = {pair};
            auto result = stage({x, x, x, cursor, pair_ids},
                {vector_map(x, channel(first)), vector_map(x, channel(second)), vector_map(x, channel(index(1))), Map{}, pm},
                rotated, rotated.dimensions, [&](Region& r, const auto& a) {
                    auto col = coordinate(r, 1);
                    auto pair_float = operation(r, Primitive::Convert, {a[4]});
                    auto exponent = operation(r, Primitive::Divide, {
                        operation(r, Primitive::Multiply, {scalar(r, -2), pair_float}), scalar(r, static_cast<float>(frequency_dimension))});
                    auto frequency = operation(r, Primitive::Pow, {scalar(r, number(p.base_f32_bits)), exponent});
                    auto valid = require(r, less(r, a[3], integer(r, capacity)));
                    auto position = operation(r, Primitive::Convert, {choose(r, valid, a[3], integer(r, 0))});
                    auto angle = operation(r, Primitive::Multiply, {
                        operation(r, Primitive::Multiply, {position, scalar(r, number(p.scale_f32_bits))}), frequency});
                    auto cosine = operation(r, Primitive::Cos, {angle}), sine = operation(r, Primitive::Sin, {angle});
                    auto first_rotated = operation(r, Primitive::Subtract, {
                        operation(r, Primitive::Multiply, {a[0], cosine}), operation(r, Primitive::Multiply, {a[1], sine})});
                    auto second_rotated = operation(r, Primitive::Add, {
                        operation(r, Primitive::Multiply, {a[1], cosine}), operation(r, Primitive::Multiply, {a[0], sine})});
                    TypedValue first_half;
                    if (p.pairing == RopePairing::HalfSplit) first_half = less(r, col, integer(r, p.rotary_dimension / 2));
                    else {
                        auto doubled_pair = operation(r, Primitive::Multiply, {a[4], integer(r, 2)}, ElementType::U32);
                        first_half = equal(r, col, doubled_pair);
                    }
                    return choose(r, less(r, col, integer(r, p.rotary_dimension)),
                        choose(r, first_half, first_rotated, second_rotated), a[2]);
                });
            output(op.outputs[operand], flatten_heads(result, x.type, head_dimension));
        }
    }

    TypedValue active_source(Region& r, TypedValue position, TypedValue source,
                             const CausalAttentionPayload& payload) {
        auto no = constant(r, {ElementType::I1, {}}, 0);
        auto yes = constant(r, {ElementType::I1, {}}, 1);
        auto active = choose(r, less(r, position, source), no, yes);
        if (payload.window == AttentionWindowKind::Sliding && payload.window_tokens < capacity) {
            auto end = operation(r, Primitive::Add, {source, integer(r, payload.window_tokens)}, ElementType::U32);
            active = choose(r, active, less(r, position, end), no);
        }
        return active;
    }
    TypedValue attention(const SemanticOperator& op, TypedValue query) {
        initialize_cursor();
        const auto& p = std::get<CausalAttentionPayload>(op.payload);
        if (p.window == AttentionWindowKind::Sliding &&
            (p.window_tokens > model.maximum_context ||
             (p.window_tokens < capacity && uint64_t(capacity) - 1 + p.window_tokens > UINT32_MAX)))
            fail("sliding attention window exceeds supported position bounds");
        if (!p.kv_heads || p.query_heads % p.kv_heads || width(query) != uint64_t(p.query_heads) * p.head_dimension)
            fail("attention query geometry mismatch");
        const bool alias = p.value_source == ValueSource::KeyStateAlias;
        uint32_t value_id = p.value_source_value;
        if (p.value_source == ValueSource::SeparateProjection) {
            if (op.semantic_version >= 7 && value_id != op.inputs[2]) fail("attention value-source binding mismatch");
            value_id = op.inputs[2];
        } else if (p.value_source == ValueSource::KeyPostRope || alias) {
            if (value_id != op.inputs[1]) fail("attention key alias binding mismatch");
        } else if (p.value_source == ValueSource::KeyPreRope) {
            size_t matches = 0;
            for (const auto& candidate : model.operators)
                if (candidate.kind == OperatorKind::Rope && semantic_operator_contract_valid(candidate) &&
                    candidate.inputs[1] == value_id && candidate.outputs[1] == op.inputs[1]) ++matches;
            if (matches != 1) fail("attention pre-RoPE value binding is not unique");
        } else fail("unsupported attention value source");
        if (!written_caches.insert(op.states[0]).second ||
            (!alias && !written_caches.insert(op.states[1]).second))
            fail("multiple append operations share a semantic cache in one invocation");
        auto key = finite(floating(input(op.inputs[1]))), value = finite(floating(input(value_id)));
        if (key.type != value.type || width(key) != uint64_t(p.kv_heads) * p.head_dimension)
            fail("attention key/value geometry mismatch");
        const auto append = [&](uint32_t state_id, StateKind kind, TypedValue token) {
            auto old = cache(state_id, kind, p.kv_heads, p.head_dimension);
            auto col = expr(TensorIndexExpression::Add, expr(TensorIndexExpression::Multiply, index(1), literal(p.head_dimension)), index(2));
            auto updated = stage({old, token, cursor}, {identity(3), vector_map(token, col), Map{}}, old.type, old.type.dimensions,
                [&](Region& r, const auto& a) {
                    auto valid = require(r, less(r, a[2], integer(r, capacity)));
                    auto position = choose(r, valid, a[2], integer(r, 0));
                    return choose(r, equal(r, coordinate(r, 0), position), a[1], a[0]);
                });
            cache_values[state_id] = updated;
            return updated;
        };
        auto keys = append(op.states[0], StateKind::KeyCache, key);
        auto vals = alias ? keys : append(op.states[1], StateKind::ValueCache, value);
        ValueType score_type{ElementType::F32, {dim(p.query_heads), dim(capacity)}};
        auto scores = sum_products({query, keys}, score_type, p.head_dimension, [&](Expr k) {
            auto qc = expr(TensorIndexExpression::Add, expr(TensorIndexExpression::Multiply, index(0), literal(p.head_dimension)), k);
            Map km; km.results = {index(1), expr(TensorIndexExpression::FloorDivide, index(0), literal(p.query_heads / p.kv_heads)), k};
            return std::vector<Map>{vector_map(query, qc), km};
        }, true);
        scores = elementwise({scores}, [&](Region& r, const auto& a) { return operation(r, Primitive::Multiply, {a[0], scalar(r, number(p.scale_f32_bits))}); });
        ValueType heads{ElementType::F32, {dim(p.query_heads)}};
        auto maximum_init = constant(entry, heads, std::bit_cast<uint32_t>(-std::numeric_limits<float>::infinity()));
        auto maximum = stage({scores, cursor}, {identity(2), Map{}}, heads, score_type.dimensions,
            [&](Region& r, const auto& a) {
                auto active = active_source(r, a[1], coordinate(r, 1), p);
                return choose(r, active, operation(r, Primitive::Maximum, {a.back(), a[0]}), a.back());
            }, maximum_init);
        Map head_map; head_map.results = {index(0)};
        auto probabilities = stage({scores, maximum}, {identity(2), head_map}, score_type, score_type.dimensions,
            [&](Region& r, const auto& a) { return operation(r, Primitive::Exp, {operation(r, Primitive::Subtract, {a[0], a[1]})}); });
        auto normalizer = stage({probabilities, cursor}, {identity(2), Map{}}, heads, score_type.dimensions,
            [&](Region& r, const auto& a) {
                return choose(r, active_source(r, a[1], coordinate(r, 1), p), operation(r, Primitive::Add, {a.back(), a[0]}), a.back());
            });
        normalizer = elementwise({normalizer}, [&](Region& r, const auto& a) {
            auto n = operation(r, Primitive::RequireFinite, {a[0]});
            return choose(r, require(r, less(r, scalar(r, 0), n)), n, scalar(r, 1));
        });
        ValueType attention_type{ElementType::F32, {dim(p.query_heads), dim(p.head_dimension)}};
        Shape iterations = attention_type.dimensions; iterations.push_back(dim(capacity));
        Map pm; pm.results = {index(0), index(2)};
        Map vm; vm.results = {index(2), expr(TensorIndexExpression::FloorDivide, index(0), literal(p.query_heads / p.kv_heads)), index(1)};
        auto attended = stage({probabilities, normalizer, vals, cursor}, {pm, head_map, vm, Map{}}, attention_type, iterations,
            [&](Region& r, const auto& a) {
                auto probability = operation(r, Primitive::Divide, {a[0], a[1]});
                auto next = operation(r, Primitive::Add, {a.back(), operation(r, Primitive::Multiply, {probability, a[2]})});
                return choose(r, active_source(r, a[3], coordinate(r, 2), p), next, a.back());
            });
        state_write(cache_state_ids.at(op.states[0]), keys);
        if (!alias) state_write(cache_state_ids.at(op.states[1]), vals);
        return flatten_heads(attended, query.type, p.head_dimension);
    }

    void lower(const SemanticOperator& op) {
        if (!semantic_operator_signature_valid(op) || !semantic_operator_contract_valid(op)) fail("invalid semantic operator payload or signature");
        if (!op.states.empty() && op.kind != OperatorKind::CausalAttention) fail("stateful semantic operator is not implemented");
        auto x = input(op.inputs[0]);
        if (op.kind != OperatorKind::EmbeddingLookup) x = finite(floating(x));
        const auto same = [&](uint32_t other) { auto y = finite(floating(input(other))); if (y.type != x.type) fail("semantic operand shape mismatch"); return y; };
        TypedValue result;
        switch (op.kind) {
        case OperatorKind::Rope:
            rope(op, x);
            return;
        case OperatorKind::CausalAttention:
            result = attention(op, x);
            break;
        case OperatorKind::EmbeddingLookup: {
            const auto& p = std::get<EmbeddingLookupPayload>(op.payload);
            auto table = tensor(op.tensors[0]);
            if (x.type.element_type != ElementType::U32 || table.type.dimensions != Shape{dim(p.width), dim(p.vocabulary)})
                fail("embedding dimensions or token type mismatch");
            ValueType out{ElementType::F32, x.type.dimensions}; out.dimensions.push_back(dim(p.width));
            Map table_map; table_map.results = {index(x.type.dimensions.size()), {TensorIndexExpression::SourceElement, 1, identity(x.type.dimensions.size()).results}};
            result = stage({table, x}, {table_map, identity(x.type.dimensions.size())}, out, out.dimensions,
                [&](Region& r, const auto& a) { return operation(r, Primitive::Multiply, {a[0], scalar(r, number(p.scale_f32_bits))}); });
            break;
        }
        case OperatorKind::Linear: {
            const auto& p = std::get<LinearPayload>(op.payload);
            auto weight = tensor(op.tensors[0]);
            if (weight.type.dimensions.size() != 2) fail("linear weight must have rank two");
            const uint64_t k = width(x);
            const size_t in_axis = p.transpose_weight ? 1 : 0;
            if (weight.type.dimensions[in_axis].value != k) fail("linear reduction dimension mismatch");
            auto out = x.type; out.dimensions.back() = weight.type.dimensions[1 - in_axis];
            result = sum_products({x, weight}, out, k, [&](Expr reduce) {
                auto xm = identity(out.dimensions.size()); xm.results.back() = reduce;
                Map wm; wm.results = !p.transpose_weight ? std::vector<Expr>{reduce, index(out.dimensions.size() - 1)} :
                    std::vector<Expr>{index(out.dimensions.size() - 1), reduce};
                return std::vector<Map>{xm, wm};
            }, true);
            if (p.has_bias) {
                auto bias = tensor(op.tensors[1]);
                if (bias.type.dimensions != Shape{out.dimensions.back()}) fail("linear bias width mismatch");
                Map bm; bm.results = {index(out.dimensions.size() - 1)};
                result = stage({result, bias}, {identity(out.dimensions.size()), bm}, out, out.dimensions,
                    [&](Region& r, const auto& a) { return operation(r, Primitive::Add, {a[0], a[1]}); });
            }
            break;
        }
        case OperatorKind::Add:
            result = elementwise({x, same(op.inputs[1])}, [&](Region& r, const auto& a) { return operation(r, Primitive::Add, {a[0], a[1]}); });
            break;
        case OperatorKind::Scale: {
            const auto& p = std::get<ScalePayload>(op.payload);
            if (p.source == ScaleSource::LiteralF32) {
                result = elementwise({x}, [&](Region& r, const auto& a) { return operation(r, Primitive::Multiply, {a[0], scalar(r, number(p.literal_f32_bits))}); });
            } else {
                auto scale = tensor(op.tensors[0]);
                Map sm;
                for (const auto& d : scale.type.dimensions) { if (d.value != 1) fail("scale tensor must contain one element"); sm.results.push_back(literal(0)); }
                result = stage({x, scale}, {identity(x.type.dimensions.size()), sm}, x.type, x.type.dimensions,
                    [&](Region& r, const auto& a) { return operation(r, Primitive::Multiply, {a[0], a[1]}); });
            }
            break;
        }
        case OperatorKind::SwiGlu:
        case OperatorKind::GatedActivation: {
            const auto kind = op.kind == OperatorKind::SwiGlu ? std::get<SwiGluPayload>(op.payload).activation : std::get<GatedActivationPayload>(op.payload).activation;
            result = elementwise({x, same(op.inputs[1])}, [&](Region& r, const auto& a) {
                return operation(r, Primitive::Multiply, {activation(r, a[0], kind), a[1]}); });
            break;
        }
        case OperatorKind::GatedAttention:
            result = elementwise({x, same(op.inputs[1])}, [&](Region& r, const auto& a) { return operation(r, Primitive::Multiply, {a[0], sigmoid(r, a[1])}); });
            break;
        case OperatorKind::TanhSoftcap: {
            float cap = number(std::get<TanhSoftcapPayload>(op.payload).cap_f32_bits);
            result = elementwise({x}, [&](Region& r, const auto& a) {
                auto c = scalar(r, cap); return operation(r, Primitive::Multiply,
                    {c, operation(r, Primitive::Tanh, {operation(r, Primitive::Divide, {a[0], c})})}); });
            break;
        }
        case OperatorKind::RmsNorm:
        case OperatorKind::GatedRmsNorm:
        case OperatorKind::L2Normalize: {
            uint64_t reduction = width(x);
            float epsilon;
            bool affine = false;
            const bool l2 = op.kind == OperatorKind::L2Normalize;
            if (l2) epsilon = number(std::get<L2NormalizePayload>(op.payload).epsilon_f32_bits);
            else if (op.kind == OperatorKind::RmsNorm) {
                const auto& p = std::get<RmsNormPayload>(op.payload);
                epsilon = number(p.epsilon_f32_bits); affine = p.weight_mode == 1;
                if (p.affine_geometry == RmsNormAffineGeometry::SharedAcrossGroups) reduction = p.reduction_extent;
            } else { epsilon = number(std::get<GatedRmsNormPayload>(op.payload).epsilon_f32_bits); affine = true; }
            if (!reduction || width(x) % reduction) fail("normalization group width mismatch");
            ValueType sums = x.type; sums.dimensions.back() = dim(width(x) / reduction);
            auto sum = sum_products({x, x}, sums, reduction, [&](Expr k) {
                auto m = identity(sums.dimensions.size());
                m.results.back() = expr(TensorIndexExpression::Add,
                    expr(TensorIndexExpression::Multiply, index(sums.dimensions.size() - 1), literal(reduction)), k);
                return std::vector<Map>{m, m};
            }, !l2);
            auto factor = elementwise({sum}, [&](Region& r, const auto& a) {
                auto mean = l2 ? a[0] : operation(r, Primitive::Divide, {a[0], scalar(r, static_cast<float>(reduction))});
                auto root = operation(r, Primitive::Sqrt, {operation(r, Primitive::Add, {mean, scalar(r, epsilon)})});
                return operation(r, Primitive::RequireFinite, {l2 ? root : operation(r, Primitive::Divide, {scalar(r, 1), root})});
            });
            std::vector<TypedValue> sources{x, factor};
            Map fm = identity(x.type.dimensions.size());
            fm.results.back() = expr(TensorIndexExpression::FloorDivide, index(x.type.dimensions.size() - 1), literal(reduction));
            std::vector<Map> maps{identity(x.type.dimensions.size()), fm};
            if (affine) {
                auto weight = tensor(op.tensors[0]);
                if (weight.type.dimensions != Shape{dim(reduction)}) fail("normalization weight width mismatch");
                sources.push_back(weight); Map wm;
                wm.results = {expr(TensorIndexExpression::Remainder, index(x.type.dimensions.size() - 1), literal(reduction))}; maps.push_back(wm);
            }
            if (op.kind == OperatorKind::GatedRmsNorm) { sources.push_back(same(op.inputs[1])); maps.push_back(identity(x.type.dimensions.size())); }
            result = stage(sources, maps, x.type, x.type.dimensions, [&](Region& r, const auto& a) {
                auto denominator = a[1];
                if (l2) {
                    auto nonzero = operation(r, Primitive::Less, {scalar(r, 0), denominator}, ElementType::I1);
                    auto required = operation(r, Primitive::Require, {nonzero}, ElementType::I1);
                    denominator = operation(r, Primitive::Select, {required, denominator, scalar(r, 1)});
                }
                auto value = operation(r, l2 ? Primitive::Divide : Primitive::Multiply, {a[0], denominator});
                if (!l2) value = operation(r, Primitive::Multiply, {value, affine ? a[2] : scalar(r, 1)});
                if (op.kind == OperatorKind::GatedRmsNorm) {
                    auto gate = a[3]; value = operation(r, Primitive::Multiply,
                        {value, operation(r, Primitive::Multiply, {gate, sigmoid(r, gate)})});
                }
                return value;
            });
            break;
        }
        case OperatorKind::AxisSplit: {
            const auto& p = std::get<AxisSplitPayload>(op.payload);
            if (uint64_t(p.first_width) + p.second_width != width(x)) fail("split widths mismatch");
            for (size_t part = 0; part < 2; ++part) {
                auto out = x.type; out.dimensions.back() = dim(part ? p.second_width : p.first_width);
                auto m = identity(out.dimensions.size());
                m.results.back() = expr(TensorIndexExpression::Add, m.results.back(), literal(part ? p.first_width : 0));
                auto v = stage({x}, {m}, out, out.dimensions, [](Region&, const auto& a) { return a[0]; });
                output(op.outputs[part], v);
            }
            return;
        }
        case OperatorKind::Concat: {
            auto y = finite(floating(input(op.inputs[1])));
            auto out = x.type;
            if (y.type.dimensions.size() != out.dimensions.size()) fail("concat rank mismatch");
            for (size_t i = 0; i + 1 < out.dimensions.size(); ++i) if (out.dimensions[i] != y.type.dimensions[i]) fail("concat prefix mismatch");
            const uint64_t left = width(x), right = width(y);
            if (left > uint64_t(INT64_MAX) - right) fail("concat width overflow");
            out.dimensions.back() = dim(left + right);
            auto mask_type = x.type; mask_type.element_type = ElementType::I1;
            auto mask = constant(entry, mask_type, 1);
            auto xm = identity(out.dimensions.size()); xm.bounds = TensorBoundsMode::Zero;
            auto ym = xm; ym.results.back() = expr(TensorIndexExpression::Add, ym.results.back(), literal(-static_cast<int64_t>(left)));
            result = stage({x, y, mask}, {xm, ym, xm}, out, out.dimensions,
                [&](Region& r, const auto& a) { return operation(r, Primitive::Select, {a[2], a[0], a[1]}); });
            break;
        }
        default: fail("semantic operator is not implemented by this compiler");
        }
        output(op.outputs[0], result);
    }

public:
    Compiler(const SemanticModel& m, std::span<const SemanticDimensionBinding> bindings, uint32_t requested_capacity)
        : model(m), capacity(requested_capacity ? requested_capacity : m.maximum_context) {
        for (const auto& d : bindings) if (!d.extent || !dimensions.emplace(d.symbol, d.extent).second) fail("invalid duplicate dimension binding");
    }
    CompiledSemanticProgram compile() {

        if (model.input_values_first > model.values.size() || model.input_values_count > model.values.size() - model.input_values_first ||
            model.output_values_first >= model.values.size() || model.output_values_count != 1) fail("invalid semantic entry/output range");
        for (const auto& v : model.values) if (!value_types.emplace(v.id, type(v.logical_type, v.dimensions)).second) fail("duplicate semantic value ID");
        for (const auto& t : model.tensors) {
            if (t.logical_type != ScalarType::F32 && t.logical_type != ScalarType::F16)
                fail("semantic tensor must decode into floating compute values");
            if (!tensor_types.emplace(t.id, type(ScalarType::F32, t.dimensions)).second)
                fail("duplicate semantic tensor ID");
        }
        for (size_t i = 0; i < model.input_values_count; ++i) {
            auto semantic_id = model.values[model.input_values_first + i].id;
            TypedValue v{id(), value_types.at(semantic_id)}; entry.arguments.push_back(v);
            input_bindings.push_back({semantic_id, v.id}); values.emplace(semantic_id, v);
        }
        std::unordered_set<uint32_t> operators;
        for (const auto& op : model.operators) {
            if (!operators.insert(op.id).second) fail("duplicate semantic operator ID");
            lower(op);
        }
        const uint32_t out_id = model.values[model.output_values_first].id;
        auto out = input(out_id);
        if (cursor_state != UINT32_MAX) {
            auto next_cursor = stage({cursor}, {Map{}}, cursor.type, {}, [&](Region& r, const auto& a) {
                return operation(r, Primitive::Add, {a[0], integer(r, 1)}, ElementType::U32);
            });
            state_write(cursor_state, next_cursor);
        }
        if (cache_state_ids.size() != model.states.size()) fail("semantic state declaration has no supported executable consumer");
        entry.yields = {out.id};
        bodies.push_back(std::move(entry));
        Program program; program.minor = 1;
        program.functions = {{0, 1, std::move(bodies), {out.type}}};
        program.exports = {{0, 0, out.type}};
        program.state_references = std::move(state_references);
        auto verified = verify_and_canonicalize_program(std::move(program));
        if (const auto* error = std::get_if<CompatibilityReport>(&verified)) throw *error;
        return {std::get<VerifiedProgram>(std::move(verified)), std::move(tensor_bindings), std::move(input_bindings), {out_id}};
    }
};
} // namespace

SemanticProgramCompileResult compile_semantic_program(const SemanticModel& model,
    std::span<const SemanticDimensionBinding> dimensions, uint32_t cache_capacity) {
    try { return Compiler(model, dimensions, cache_capacity).compile(); }
    catch (const CompatibilityReport& report) { return report; }
}
} // namespace Laplace
