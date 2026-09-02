#include "program_ir.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Laplace {
namespace {

constexpr uint64_t kMaximumDimension = uint64_t{1} << 48;
constexpr uint64_t kMaximumLoopIterations = uint64_t{1} << 32;
constexpr size_t kMaximumFunctions = 1024;
constexpr size_t kMaximumRegions = 65536;
// Canonicalization follows typed dependencies recursively. Keep the complete
// V1 program below the verified host-stack boundary; repetition belongs in the
// bounded-control form rather than in an unrolled authority object.
constexpr size_t kMaximumInstructions = 4096;
constexpr size_t kMaximumEffectEdges = 1u << 16;
constexpr size_t kMaximumValues = 1u << 22;
constexpr size_t kMaximumStateReferences = 1u << 20;
constexpr size_t kMaximumExports = 1u << 20;
constexpr size_t kMaximumShapeDepth = 32;
constexpr size_t kMaximumDimensions = 32;
constexpr size_t kMaximumStructuredOperands = 64;
constexpr size_t kMaximumIndexExpressionNodes = 4096;

struct Bounds {
    uint64_t lower = 0;
    uint64_t upper = 0;
};

struct ValueDefinition {
    const ValueType* type = nullptr;
    uint32_t region_id = UINT32_MAX;
    uint32_t position = UINT32_MAX;
    uint32_t result_index = UINT32_MAX;
    bool argument = false;
};

struct InstructionLocation {
    const Instruction* instruction = nullptr;
    uint32_t region_id = UINT32_MAX;
    uint32_t position = UINT32_MAX;
};

struct RegionParent {
    uint32_t region_id = UINT32_MAX;
    uint32_t instruction_id = UINT32_MAX;
    uint32_t instruction_position = UINT32_MAX;
    uint32_t child_index = UINT32_MAX;
};

struct FunctionInfo {
    const Function* function = nullptr;
    std::unordered_map<uint32_t, const Region*> regions;
    std::unordered_map<uint32_t, RegionParent> parents;
    std::unordered_map<uint32_t, ValueDefinition> values;
    std::unordered_map<uint32_t, InstructionLocation> instructions;
    std::unordered_map<uint32_t, std::vector<uint32_t>> effect_successors;
};

bool valid_element_type(ElementType type) {
    switch (type) {
    case ElementType::I1:
    case ElementType::I32:
    case ElementType::U32:
    case ElementType::U64:
    case ElementType::F16:
    case ElementType::F32:
        return true;
    }
    return false;
}

bool constant_fits_element_type(ElementType type, uint64_t bits) {
    switch (type) {
    case ElementType::I1:
        return bits <= 1;
    case ElementType::F16:
        return (bits >> 16) == 0;
    case ElementType::I32:
    case ElementType::U32:
    case ElementType::F32:
        return (bits >> 32) == 0;
    case ElementType::U64:
        return true;
    }
    return false;
}

uint64_t ceil_divide(uint64_t value, uint64_t divisor) {
    return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

bool add_fits(uint64_t left, uint64_t right, uint64_t* result) {
    if (left > kMaximumDimension - right) return false;
    *result = left + right;
    return true;
}

bool multiply_fits(uint64_t left, uint64_t right, uint64_t* result) {
    if (left != 0 && right > kMaximumDimension / left) return false;
    *result = left * right;
    return true;
}

class Encoder {
public:
    void u8(uint8_t value) { bytes_.push_back(value); }

    void u16(uint16_t value) {
        for (unsigned shift = 0; shift != 16; shift += 8) {
            u8(static_cast<uint8_t>(value >> shift));
        }
    }

    void u32(uint32_t value) {
        for (unsigned shift = 0; shift != 32; shift += 8) {
            u8(static_cast<uint8_t>(value >> shift));
        }
    }

    void u64(uint64_t value) {
        for (unsigned shift = 0; shift != 64; shift += 8) {
            u8(static_cast<uint8_t>(value >> shift));
        }
    }

    void digest(ProgramDigest value) {
        bytes_.insert(bytes_.end(), value.bytes.begin(), value.bytes.end());
    }

    void domain(std::string_view value) {
        u32(static_cast<uint32_t>(value.size()));
        bytes_.insert(bytes_.end(), value.begin(), value.end());
    }

    ProgramDigest finish() const {
        ProgramDigest result;
        CC_SHA256_CTX context;
        CC_SHA256_Init(&context);
        size_t offset = 0;
        while (offset < bytes_.size()) {
            const size_t count = std::min<size_t>(
                bytes_.size() - offset, std::numeric_limits<CC_LONG>::max());
            CC_SHA256_Update(&context, bytes_.data() + offset,
                             static_cast<CC_LONG>(count));
            offset += count;
        }
        CC_SHA256_Final(result.bytes.data(), &context);
        return result;
    }

private:
    std::vector<uint8_t> bytes_;
};

class Verifier {
public:
    explicit Verifier(const Program& program) : program_(program) {}

    bool run() {
        if (program_.major != 1 || program_.minor > 1) {
            return reject(CompatibilityError::IR_VERSION_UNSUPPORTED,
                          "program version is unsupported");
        }
        if (program_.functions.empty() || program_.functions.size() > kMaximumFunctions) {
            return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                          "function count is outside the supported bound");
        }
        if (!validate_dimensions() || !validate_states()) return false;

        std::unordered_set<uint32_t> function_ids;
        infos_.reserve(program_.functions.size());
        for (const Function& function : program_.functions) {
            if (function.id == UINT32_MAX || !function_ids.insert(function.id).second) {
                return reject(CompatibilityError::IR_REFERENCE_INVALID,
                              "function identity is invalid or repeated");
            }
            infos_.push_back({});
            if (!build_function(function, infos_.back())) return false;
        }
        if (!validate_exports(function_ids)) return false;
        return true;
    }

    const std::vector<FunctionInfo>& infos() const { return infos_; }
    const std::unordered_map<uint32_t, uint32_t>& dimension_ordinals() const {
        return dimension_ordinals_;
    }
    const std::unordered_map<uint32_t, uint32_t>& state_ordinals() const {
        return state_ordinals_;
    }
    const CompatibilityReport& report() const { return report_; }

private:
    const Program& program_;
    CompatibilityReport report_;
    bool failed_ = false;
    std::unordered_map<uint32_t, uint32_t> dimension_ordinals_;
    std::unordered_map<uint32_t, uint32_t> state_ordinals_;
    std::vector<FunctionInfo> infos_;
    size_t instruction_count_ = 0;
    size_t effect_edge_count_ = 0;
    size_t region_count_ = 0;
    size_t value_count_ = 0;

    bool reject(CompatibilityError code, std::string detail) {
        if (!failed_) {
            report_ = compatibility_report(code, std::move(detail));
            failed_ = true;
        }
        return false;
    }

    bool validate_dimensions() {
        if (program_.dimension_parameters.size() > kMaximumDimensions) {
            return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                          "dimension parameter count exceeds the bound");
        }
        for (uint32_t index = 0; index < program_.dimension_parameters.size(); ++index) {
            const DimensionParameter& parameter = program_.dimension_parameters[index];
            if (parameter.id == UINT32_MAX || parameter.lower > parameter.upper ||
                parameter.upper > kMaximumDimension ||
                !dimension_ordinals_.emplace(parameter.id, index).second) {
                return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                              "dimension parameter is invalid or repeated");
            }
        }
        return true;
    }

    bool dimension_bounds(const DimensionExpr& expression, size_t depth,
                          Bounds* result) {
        if (depth > kMaximumShapeDepth) {
            return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                          "dimension expression nesting exceeds the bound");
        }
        switch (expression.expression) {
        case DimensionExpression::Constant:
            if (!expression.operands.empty() || expression.value > kMaximumDimension) {
                return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                              "constant dimension is invalid");
            }
            *result = {expression.value, expression.value};
            return true;
        case DimensionExpression::Parameter: {
            if (!expression.operands.empty() || expression.value > UINT32_MAX) {
                return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                              "dimension parameter reference is invalid");
            }
            const auto found = dimension_ordinals_.find(
                static_cast<uint32_t>(expression.value));
            if (found == dimension_ordinals_.end()) {
                return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                              "dimension parameter reference is unresolved");
            }
            const DimensionParameter& parameter =
                program_.dimension_parameters[found->second];
            *result = {parameter.lower, parameter.upper};
            return true;
        }
        case DimensionExpression::Add:
        case DimensionExpression::Multiply:
        case DimensionExpression::CeilDivide:
            break;
        default:
            return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                          "dimension expression is unsupported");
        }
        if (expression.operands.size() != 2) {
            return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                          "dimension expression arity is invalid");
        }
        Bounds left;
        Bounds right;
        if (!dimension_bounds(expression.operands[0], depth + 1, &left) ||
            !dimension_bounds(expression.operands[1], depth + 1, &right)) {
            return false;
        }
        if (expression.expression == DimensionExpression::Add) {
            return add_fits(left.lower, right.lower, &result->lower) &&
                       add_fits(left.upper, right.upper, &result->upper)
                       ? true
                       : reject(CompatibilityError::IR_SHAPE_MISMATCH,
                                "dimension addition exceeds the bound");
        }
        if (expression.expression == DimensionExpression::Multiply) {
            return multiply_fits(left.lower, right.lower, &result->lower) &&
                       multiply_fits(left.upper, right.upper, &result->upper)
                       ? true
                       : reject(CompatibilityError::IR_SHAPE_MISMATCH,
                                "dimension multiplication exceeds the bound");
        }
        if (right.lower == 0) {
            return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                          "dimension divisor can be zero");
        }
        result->lower = ceil_divide(left.lower, right.upper);
        result->upper = ceil_divide(left.upper, right.lower);
        return true;
    }

    bool valid_type(const ValueType& type) {
        if (!valid_element_type(type.element_type) ||
            type.dimensions.size() > kMaximumDimensions) {
            return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                          "value type is invalid");
        }
        for (const DimensionExpr& dimension : type.dimensions) {
            Bounds ignored;
            if (!dimension_bounds(dimension, 0, &ignored)) return false;
        }
        return true;
    }

    bool validate_states() {
        if (program_.state_references.size() > kMaximumStateReferences) {
            return reject(CompatibilityError::IR_STATE_INVALID,
                          "state reference count exceeds the bound");
        }
        for (uint32_t index = 0; index < program_.state_references.size(); ++index) {
            const StateReference& state = program_.state_references[index];
            if (state.id == UINT32_MAX ||
                !state_ordinals_.emplace(state.id, index).second) {
                return reject(CompatibilityError::IR_STATE_INVALID,
                              "state reference identity is invalid or repeated");
            }
            if (!valid_type(state.type)) return false;
        }
        return true;
    }

    bool build_function(const Function& function, FunctionInfo& info) {
        info.function = &function;
        if (function.regions.empty() ||
            function.regions.size() > kMaximumRegions - region_count_) {
            return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                          "region count is outside the supported bound");
        }
        region_count_ += function.regions.size();
        if (function.result_types.size() > kMaximumValues) {
            return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                          "function result count exceeds the bound");
        }
        for (const ValueType& result : function.result_types) {
            if (!valid_type(result)) return false;
        }
        for (const Region& region : function.regions) {
            if (region.id == UINT32_MAX || !info.regions.emplace(region.id, &region).second) {
                return reject(CompatibilityError::IR_REFERENCE_INVALID,
                              "region identity is invalid or repeated");
            }
        }
        if (info.regions.find(function.entry_region_id) == info.regions.end()) {
            return reject(CompatibilityError::IR_REFERENCE_INVALID,
                          "entry region is unresolved");
        }

        for (const Region& region : function.regions) {
            if (region.instructions.size() >
                    kMaximumInstructions - instruction_count_ ||
                region.arguments.size() > kMaximumValues - value_count_ ||
                region.yields.size() > kMaximumValues) {
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "instruction or value count exceeds the bound");
            }
            instruction_count_ += region.instructions.size();
            value_count_ += region.arguments.size();
            for (uint32_t index = 0; index < region.arguments.size(); ++index) {
                const TypedValue& argument = region.arguments[index];
                if (argument.id == UINT32_MAX || !valid_type(argument.type) ||
                    !info.values.emplace(
                        argument.id,
                        ValueDefinition{&argument.type, region.id, index, UINT32_MAX, true})
                         .second) {
                    return reject(CompatibilityError::IR_REFERENCE_INVALID,
                                  "region argument identity is invalid or repeated");
                }
            }
            for (uint32_t position = 0; position < region.instructions.size(); ++position) {
                const Instruction& item = region.instructions[position];
                if (item.inputs.size() > kMaximumValues ||
                    item.outputs.size() > kMaximumValues - value_count_ ||
                    item.regions.size() > kMaximumRegions ||
                    item.effect_predecessors.size() > kMaximumInstructions ||
                    item.effect_predecessors.size() >
                        kMaximumEffectEdges - effect_edge_count_) {
                    return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                                  "instruction arity exceeds the bound");
                }
                effect_edge_count_ += item.effect_predecessors.size();
                if (item.id == UINT32_MAX ||
                    !info.instructions.emplace(
                        item.id, InstructionLocation{&item, region.id, position})
                         .second) {
                    return reject(CompatibilityError::IR_REFERENCE_INVALID,
                                  "instruction identity is invalid or repeated");
                }
                for (uint32_t result_index = 0; result_index < item.outputs.size();
                     ++result_index) {
                    const TypedValue& output = item.outputs[result_index];
                    ++value_count_;
                    if (value_count_ > kMaximumValues || output.id == UINT32_MAX ||
                        !valid_type(output.type) ||
                        !info.values.emplace(
                            output.id,
                            ValueDefinition{&output.type, region.id, position,
                                            result_index, false})
                             .second) {
                        return reject(CompatibilityError::IR_REFERENCE_INVALID,
                                      "instruction result identity is invalid or repeated");
                    }
                }
                for (uint32_t child_index = 0; child_index < item.regions.size();
                     ++child_index) {
                    const uint32_t child_id = item.regions[child_index];
                    if (info.regions.find(child_id) == info.regions.end() ||
                        !info.parents.emplace(
                            child_id,
                            RegionParent{region.id, item.id, position, child_index})
                             .second) {
                        return reject(CompatibilityError::IR_REFERENCE_INVALID,
                                      "nested region ownership is invalid");
                    }
                }
            }
        }
        if (info.parents.find(function.entry_region_id) != info.parents.end()) {
            return reject(CompatibilityError::IR_REFERENCE_INVALID,
                          "entry region cannot have a parent");
        }
        for (const Region& region : function.regions) {
            if (region.id != function.entry_region_id &&
                info.parents.find(region.id) == info.parents.end()) {
                return reject(CompatibilityError::IR_REFERENCE_INVALID,
                              "nested region is not owned");
            }
        }

        std::unordered_map<uint32_t, uint8_t> region_colors;
        if (!visit_region(function.entry_region_id, info, region_colors)) return false;
        if (region_colors.size() != function.regions.size()) {
            return reject(CompatibilityError::IR_REFERENCE_INVALID,
                          "region is not reachable from the entry");
        }

        for (const Region& region : function.regions) {
            for (uint32_t position = 0; position < region.instructions.size(); ++position) {
                const Instruction& item = region.instructions[position];
                for (uint32_t input : item.inputs) {
                    const auto definition = info.values.find(input);
                    if (definition == info.values.end() ||
                        definition->second.region_id != region.id ||
                        !dominates(definition->second, region.id, position, info)) {
                        return reject(CompatibilityError::IR_REFERENCE_INVALID,
                                      "value is unresolved, captured, or does not dominate its use");
                    }
                }
                if (!validate_instruction(item, info)) return false;
                std::unordered_set<uint32_t> predecessors;
                for (uint32_t predecessor : item.effect_predecessors) {
                    const auto found = info.instructions.find(predecessor);
                    if (!predecessors.insert(predecessor).second ||
                        found == info.instructions.end() ||
                        found->second.region_id != region.id ||
                        !is_state_access(*found->second.instruction) ||
                        !is_state_access(item)) {
                        return reject(CompatibilityError::IR_STATE_INVALID,
                                      "effect dependency is invalid");
                    }
                    info.effect_successors[predecessor].push_back(item.id);
                }
            }
            if (!validate_yields(region, info)) return false;
        }
        const Region& entry = *info.regions.at(function.entry_region_id);
        if (entry.yields.size() != function.result_types.size()) {
            return reject(CompatibilityError::IR_REFERENCE_INVALID,
                          "function result count does not match entry yields");
        }
        for (uint32_t index = 0; index < entry.yields.size(); ++index) {
            const ValueDefinition& definition = info.values.at(entry.yields[index]);
            if (*definition.type != function.result_types[index]) {
                return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                              "function result type does not match entry yield");
            }
        }

        std::unordered_map<uint32_t, uint8_t> dependency_colors;
        for (const auto& [id, location] : info.instructions) {
            if (!visit_instruction_dependencies(id, info, dependency_colors)) return false;
        }
        if (!validate_alias_order(info)) return false;
        return validate_effect_roots(info);
    }

    bool visit_region(uint32_t region_id, const FunctionInfo& info,
                      std::unordered_map<uint32_t, uint8_t>& colors) {
        const uint8_t color = colors[region_id];
        if (color == 1) {
            return reject(CompatibilityError::IR_REFERENCE_INVALID,
                          "region ownership contains a cycle");
        }
        if (color == 2) return true;
        colors[region_id] = 1;
        const Region& region = *info.regions.at(region_id);
        for (const Instruction& item : region.instructions) {
            for (uint32_t child : item.regions) {
                if (!visit_region(child, info, colors)) return false;
            }
        }
        colors[region_id] = 2;
        return true;
    }

    bool dominates(const ValueDefinition& definition, uint32_t use_region,
                   uint32_t use_position, const FunctionInfo& info) const {
        uint32_t current_region = use_region;
        uint32_t current_position = use_position;
        while (true) {
            if (definition.region_id == current_region) {
                return definition.argument || definition.position < current_position;
            }
            const auto parent = info.parents.find(current_region);
            if (parent == info.parents.end()) return false;
            current_region = parent->second.region_id;
            current_position = parent->second.instruction_position;
        }
    }

    const ValueType* input_type(const Instruction& item, uint32_t index,
                                const FunctionInfo& info) const {
        if (index >= item.inputs.size()) return nullptr;
        const auto found = info.values.find(item.inputs[index]);
        return found == info.values.end() ? nullptr : found->second.type;
    }

    struct IndexBounds {
        int64_t lower = 0;
        int64_t upper = 0;
        bool empty = false;
        bool dynamic = false;
    };

    bool checked_index_value(__int128 value, int64_t* result) {
        if (value < std::numeric_limits<int64_t>::min() ||
            value > std::numeric_limits<int64_t>::max()) {
            return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                          "tensor index expression overflows I64");
        }
        *result = static_cast<int64_t>(value);
        return true;
    }

    static int64_t floor_divide_index(int64_t value, int64_t divisor) {
        const int64_t quotient = value / divisor;
        const int64_t remainder = value % divisor;
        return quotient - static_cast<int64_t>(remainder < 0);
    }

    bool index_bounds(const TensorIndexExpr& expression,
                      std::span<const Bounds> iterator_bounds,
                      size_t depth, size_t* nodes, IndexBounds* result) {
        if (!nodes || !result || depth > kMaximumShapeDepth ||
            ++*nodes > kMaximumIndexExpressionNodes) {
            return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                          "tensor index expression exceeds its bound");
        }
        switch (expression.expression) {
        case TensorIndexExpression::Constant:
            if (!expression.operands.empty()) {
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "tensor constant index has operands");
            }
            result->lower = expression.value;
            result->upper = expression.value;
            return true;
        case TensorIndexExpression::Iterator:
            if (!expression.operands.empty() || expression.value < 0 ||
                static_cast<uint64_t>(expression.value) >= iterator_bounds.size()) {
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "tensor iterator index is invalid");
            }
            if (iterator_bounds[static_cast<size_t>(expression.value)].upper == 0) {
                result->empty = true;
                return true;
            }
            result->lower = 0;
            return checked_index_value(
                static_cast<__int128>(
                    iterator_bounds[static_cast<size_t>(expression.value)].upper) - 1,
                &result->upper);
        case TensorIndexExpression::SourceScalar:
            if (!expression.operands.empty() || expression.value < 0) {
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "tensor source scalar index is invalid");
            }
            result->dynamic = true;
            return true;
        case TensorIndexExpression::Add:
        case TensorIndexExpression::Multiply:
        case TensorIndexExpression::FloorDivide:
        case TensorIndexExpression::Remainder:
            break;
        default:
            return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                          "tensor index expression is unsupported");
        }
        if (expression.value != 0 || expression.operands.size() != 2) {
            return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                          "tensor index expression arity is invalid");
        }
        IndexBounds left;
        IndexBounds right;
        if (!index_bounds(expression.operands[0], iterator_bounds, depth + 1,
                          nodes, &left) ||
            !index_bounds(expression.operands[1], iterator_bounds, depth + 1,
                          nodes, &right)) {
            return false;
        }
        if (left.empty || right.empty) {
            result->empty = true;
            return true;
        }
        result->dynamic = left.dynamic || right.dynamic;
        if (expression.expression == TensorIndexExpression::Add) {
            if (result->dynamic) return true;
            return checked_index_value(static_cast<__int128>(left.lower) + right.lower,
                                       &result->lower) &&
                   checked_index_value(static_cast<__int128>(left.upper) + right.upper,
                                       &result->upper);
        }
        if (expression.expression == TensorIndexExpression::Multiply) {
            const bool left_constant =
                expression.operands[0].expression == TensorIndexExpression::Constant;
            const bool right_constant =
                expression.operands[1].expression == TensorIndexExpression::Constant;
            if (left_constant == right_constant) {
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "tensor index multiplication requires one constant");
            }
            if (result->dynamic) return true;
            const std::array<__int128, 4> products = {
                static_cast<__int128>(left.lower) * right.lower,
                static_cast<__int128>(left.lower) * right.upper,
                static_cast<__int128>(left.upper) * right.lower,
                static_cast<__int128>(left.upper) * right.upper};
            const auto [minimum, maximum] =
                std::minmax_element(products.begin(), products.end());
            return checked_index_value(*minimum, &result->lower) &&
                   checked_index_value(*maximum, &result->upper);
        }
        if (expression.operands[1].expression != TensorIndexExpression::Constant ||
            right.lower <= 0 || right.lower != right.upper) {
            return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                          "tensor index divisor must be a positive constant");
        }
        if (left.dynamic) return true;
        if (expression.expression == TensorIndexExpression::FloorDivide) {
            result->lower = floor_divide_index(left.lower, right.lower);
            result->upper = floor_divide_index(left.upper, right.lower);
            return true;
        }
        result->lower = 0;
        result->upper = right.lower - 1;
        return true;
    }

    static bool exact_iterator(const TensorIndexExpr& expression,
                               uint32_t* iterator) {
        if (!iterator || expression.expression != TensorIndexExpression::Iterator ||
            !expression.operands.empty() || expression.value < 0 ||
            expression.value > UINT32_MAX)
            return false;
        *iterator = static_cast<uint32_t>(expression.value);
        return true;
    }

    bool validate_source_scalar_indices(
        const TensorIndexExpr& expression, const Instruction& item,
        const StructuredTensorAttributes& attributes,
        const FunctionInfo& info, size_t depth = 0) {
        if (depth > kMaximumShapeDepth) {
            return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                          "tensor source scalar index exceeds its bound");
        }
        if (expression.expression == TensorIndexExpression::SourceScalar) {
            if (!expression.operands.empty() || expression.value < 0) {
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "tensor source scalar index is malformed");
            }
            if (static_cast<uint64_t>(expression.value) >=
                attributes.source_count) {
                return reject(CompatibilityError::IR_REFERENCE_INVALID,
                              "tensor source scalar reference is invalid");
            }
            const ValueType* type = input_type(
                item, static_cast<uint32_t>(expression.value), info);
            if (!type || *type != ValueType{ElementType::U32, {}}) {
                return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                              "tensor source scalar must be scalar U32");
            }
            return true;
        }
        for (const TensorIndexExpr& operand : expression.operands) {
            if (!validate_source_scalar_indices(operand, item, attributes,
                                                info, depth + 1))
                return false;
        }
        return true;
    }

    bool validate_structured_tensor(const Instruction& item,
                                    const FunctionInfo& info) {
        const auto* attributes =
            std::get_if<StructuredTensorAttributes>(&item.attributes);
        if (program_.minor < 1) {
            return reject(CompatibilityError::IR_VERSION_UNSUPPORTED,
                          "structured tensor requires program V1.1");
        }
        if (!attributes || attributes->source_count == 0 || item.outputs.empty() ||
            item.inputs.size() != attributes->source_count + item.outputs.size() ||
            item.inputs.size() > kMaximumStructuredOperands ||
            item.regions.size() != 1 || !item.effect_predecessors.empty() ||
            attributes->iteration_dimensions.empty() ||
            attributes->iteration_dimensions.size() > kMaximumDimensions ||
            attributes->iterator_kinds.size() !=
                attributes->iteration_dimensions.size() ||
            attributes->indexing_maps.size() != item.inputs.size()) {
            return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                          "structured tensor contract is invalid");
        }

        std::vector<Bounds> iterator_bounds(attributes->iteration_dimensions.size());
        size_t parallel_count = 0;
        for (size_t index = 0; index < attributes->iteration_dimensions.size();
             ++index) {
            if (!dimension_bounds(attributes->iteration_dimensions[index], 0,
                                  &iterator_bounds[index]))
                return false;
            switch (attributes->iterator_kinds[index]) {
            case TensorIteratorKind::Parallel:
                ++parallel_count;
                break;
            case TensorIteratorKind::Reduction:
                break;
            default:
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "structured tensor iterator kind is invalid");
            }
        }

        const Region& body = *info.regions.at(item.regions.front());
        if (body.arguments.size() != item.inputs.size() ||
            body.yields.size() != item.outputs.size()) {
            return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                          "structured tensor body signature is invalid");
        }
        for (const Instruction& body_item : body.instructions) {
            const Primitive code = body_item.primitive.code;
            if ((code != Primitive::Constant && code != Primitive::Add &&
                 code != Primitive::Multiply && code != Primitive::Subtract &&
                 code != Primitive::Divide && code != Primitive::Maximum &&
                 code != Primitive::Negate && code != Primitive::Exp &&
                 code != Primitive::Log && code != Primitive::Rsqrt &&
                 code != Primitive::Sin && code != Primitive::Cos) ||
                !body_item.regions.empty() || !body_item.effect_predecessors.empty()) {
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "structured tensor body is not scalar and pure");
            }
            for (const TypedValue& output : body_item.outputs) {
                if (!output.type.dimensions.empty()) {
                    return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                                  "structured tensor body output is not scalar");
                }
            }
        }

        for (size_t input = 0; input < item.inputs.size(); ++input) {
            const ValueType* type = input_type(item, static_cast<uint32_t>(input), info);
            if (!type || !body.arguments[input].type.dimensions.empty() ||
                body.arguments[input].type.element_type != type->element_type ||
                attributes->indexing_maps[input].results.size() !=
                    type->dimensions.size()) {
                return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                              "structured tensor operand or body type is invalid");
            }
            const TensorIndexMap& map = attributes->indexing_maps[input];
            if (map.bounds != TensorBoundsMode::Reject &&
                map.bounds != TensorBoundsMode::Zero) {
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "structured tensor bounds mode is invalid");
            }
            for (size_t axis = 0; axis < map.results.size(); ++axis) {
                size_t nodes = 0;
                IndexBounds bounds;
                if (!validate_source_scalar_indices(map.results[axis], item,
                                                    *attributes, info))
                    return false;
                if (!index_bounds(map.results[axis], iterator_bounds, 0, &nodes,
                                  &bounds))
                    return false;
                uint32_t iterator = UINT32_MAX;
                const bool identity_dimension =
                    exact_iterator(map.results[axis], &iterator) &&
                    iterator < attributes->iteration_dimensions.size() &&
                    type->dimensions[axis] ==
                        attributes->iteration_dimensions[iterator];
                Bounds operand_bounds;
                if (!dimension_bounds(type->dimensions[axis], 0, &operand_bounds))
                    return false;
                if (map.bounds == TensorBoundsMode::Reject && !bounds.empty &&
                    !bounds.dynamic &&
                    !identity_dimension &&
                    (bounds.lower < 0 ||
                     static_cast<uint64_t>(bounds.upper) >= operand_bounds.lower)) {
                    return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                                  "structured tensor source map is not provably in bounds");
                }
            }
        }

        for (size_t result = 0; result < item.outputs.size(); ++result) {
            const size_t destination = attributes->source_count + result;
            const ValueType* initial =
                input_type(item, static_cast<uint32_t>(destination), info);
            const ValueType& output = item.outputs[result].type;
            if (!initial || *initial != output ||
                output.dimensions.size() != parallel_count ||
                body.arguments[destination].type !=
                    ValueType{output.element_type, {}}) {
                return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                              "structured tensor destination type is invalid");
            }
            const TensorIndexMap& map = attributes->indexing_maps[destination];
            if (map.bounds != TensorBoundsMode::Reject ||
                map.results.size() != parallel_count) {
                return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                              "structured tensor destination map is invalid");
            }
            std::vector<bool> seen(attributes->iteration_dimensions.size(), false);
            for (size_t axis = 0; axis < map.results.size(); ++axis) {
                uint32_t iterator = UINT32_MAX;
                if (!exact_iterator(map.results[axis], &iterator) ||
                    iterator >= attributes->iterator_kinds.size() ||
                    attributes->iterator_kinds[iterator] !=
                        TensorIteratorKind::Parallel || seen[iterator] ||
                    output.dimensions[axis] !=
                        attributes->iteration_dimensions[iterator]) {
                    return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                                  "structured tensor destination is not an injective parallel map");
                }
                seen[iterator] = true;
            }
            for (size_t iterator = 0; iterator < seen.size(); ++iterator) {
                if (attributes->iterator_kinds[iterator] ==
                        TensorIteratorKind::Parallel &&
                    !seen[iterator]) {
                    return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                                  "structured tensor destination omits a parallel iterator");
                }
            }
            const auto yielded = info.values.find(body.yields[result]);
            if (yielded == info.values.end() ||
                *yielded->second.type != ValueType{output.element_type, {}}) {
                return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                              "structured tensor body result type is invalid");
            }
        }
        return true;
    }

    bool known_primitive(Primitive code) const {
        switch (code) {
        case Primitive::Constant:
        case Primitive::Add:
        case Primitive::Multiply:
        case Primitive::BoundedLoop:
        case Primitive::StateRead:
        case Primitive::StateWrite:
        case Primitive::StructuredTensor:
        case Primitive::Subtract:
        case Primitive::Divide:
        case Primitive::Maximum:
        case Primitive::Negate:
        case Primitive::Exp:
        case Primitive::Log:
        case Primitive::Rsqrt:
        case Primitive::Sin:
        case Primitive::Cos:
            return true;
        }
        return false;
    }

    bool is_state_access(const Instruction& item) const {
        return item.primitive.code == Primitive::StateRead ||
               item.primitive.code == Primitive::StateWrite;
    }

    bool validate_instruction(const Instruction& item, const FunctionInfo& info) {
        if (!known_primitive(item.primitive.code) || item.primitive.major != 1 ||
            item.primitive.minor != 0) {
            return reject(CompatibilityError::IR_VERSION_UNSUPPORTED,
                          "primitive version is unsupported");
        }
        switch (item.primitive.code) {
        case Primitive::Constant:
            if (!std::holds_alternative<ConstantAttributes>(item.attributes) ||
                !item.inputs.empty() || item.outputs.size() != 1 ||
                (program_.minor < 1 &&
                 !item.outputs.front().type.dimensions.empty()) ||
                !item.regions.empty() ||
                !item.effect_predecessors.empty()) {
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "constant primitive contract is invalid");
            }
            if (!constant_fits_element_type(
                    item.outputs.front().type.element_type,
                    std::get<ConstantAttributes>(item.attributes).bits)) {
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "constant payload exceeds its declared element width");
            }
            return true;
        case Primitive::Add:
        case Primitive::Multiply: {
            if (!std::holds_alternative<NoAttributes>(item.attributes) ||
                item.inputs.size() != 2 || item.outputs.size() != 1 ||
                !item.regions.empty() || !item.effect_predecessors.empty()) {
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "arithmetic primitive contract is invalid");
            }
            const ValueType* left = input_type(item, 0, info);
            const ValueType* right = input_type(item, 1, info);
            if (left == nullptr || right == nullptr || *left != *right ||
                *left != item.outputs.front().type) {
                return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                              "arithmetic value types do not match");
            }
            return true;
        }
        case Primitive::Subtract:
        case Primitive::Divide:
        case Primitive::Maximum: {
            if (!std::holds_alternative<NoAttributes>(item.attributes) ||
                item.inputs.size() != 2 || item.outputs.size() != 1 ||
                !item.regions.empty() || !item.effect_predecessors.empty()) {
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "F32 binary primitive contract is invalid");
            }
            const ValueType* left = input_type(item, 0, info);
            const ValueType* right = input_type(item, 1, info);
            if (left == nullptr || right == nullptr || *left != *right ||
                *left != item.outputs.front().type ||
                left->element_type != ElementType::F32) {
                return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                              "F32 binary value types do not match");
            }
            return true;
        }
        case Primitive::Negate:
        case Primitive::Exp:
        case Primitive::Log:
        case Primitive::Rsqrt:
        case Primitive::Sin:
        case Primitive::Cos: {
            if (!std::holds_alternative<NoAttributes>(item.attributes) ||
                item.inputs.size() != 1 || item.outputs.size() != 1 ||
                !item.regions.empty() || !item.effect_predecessors.empty()) {
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "F32 unary primitive contract is invalid");
            }
            const ValueType* input = input_type(item, 0, info);
            if (input == nullptr || *input != item.outputs.front().type ||
                input->element_type != ElementType::F32) {
                return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                              "F32 unary value types do not match");
            }
            return true;
        }
        case Primitive::BoundedLoop: {
            const auto* bounds = std::get_if<LoopAttributes>(&item.attributes);
            if (bounds == nullptr || item.regions.size() != 1 ||
                item.inputs.size() != item.outputs.size() || bounds->step == 0 ||
                bounds->upper < bounds->lower) {
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "bounded loop contract is invalid");
            }
            const uint64_t distance = bounds->upper - bounds->lower;
            const uint64_t iterations = ceil_divide(distance, bounds->step);
            if (iterations > kMaximumLoopIterations) {
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "bounded loop trip count exceeds the limit");
            }
            if (iterations != 0 &&
                bounds->step >
                    (std::numeric_limits<uint64_t>::max() - bounds->lower) /
                        iterations) {
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "bounded loop induction increment can overflow");
            }
            const Region& body = *info.regions.at(item.regions.front());
            if (body.arguments.size() != item.inputs.size() + 1 ||
                body.yields.size() != item.outputs.size() ||
                body.arguments.front().type != ValueType{ElementType::U64, {}}) {
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "bounded loop region signature is invalid");
            }
            for (uint32_t index = 0; index < item.inputs.size(); ++index) {
                const ValueType* input = input_type(item, index, info);
                const ValueType& output = item.outputs[index].type;
                const ValueType& argument = body.arguments[index + 1].type;
                const auto yielded = info.values.find(body.yields[index]);
                if (input == nullptr || yielded == info.values.end() ||
                    *input != output || *input != argument ||
                    *input != *yielded->second.type) {
                    return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                                  "bounded loop carried value types do not match");
                }
            }
            return true;
        }
        case Primitive::StateRead:
        case Primitive::StateWrite: {
            const auto* attributes = std::get_if<StateAttributes>(&item.attributes);
            if (attributes == nullptr || !item.regions.empty()) {
                return reject(CompatibilityError::IR_STATE_INVALID,
                              "state primitive attributes are invalid");
            }
            const auto state = state_ordinals_.find(attributes->state_id);
            if (state == state_ordinals_.end()) {
                return reject(CompatibilityError::IR_STATE_INVALID,
                              "state reference is unresolved");
            }
            const StateReference& reference = program_.state_references[state->second];
            if (item.primitive.code == Primitive::StateRead) {
                if (!item.inputs.empty() || item.outputs.size() != 1 ||
                    item.outputs.front().type != reference.type) {
                    return reject(CompatibilityError::IR_STATE_INVALID,
                                  "state read contract is invalid");
                }
            } else if (item.inputs.size() != 1 || !item.outputs.empty() ||
                       !reference.writable ||
                       *input_type(item, 0, info) != reference.type) {
                return reject(CompatibilityError::IR_STATE_INVALID,
                              "state write contract is invalid");
            }
            return true;
        }
        case Primitive::StructuredTensor:
            return validate_structured_tensor(item, info);
        }
        return false;
    }

    bool validate_yields(const Region& region, const FunctionInfo& info) {
        const uint32_t position = static_cast<uint32_t>(region.instructions.size());
        for (uint32_t value_id : region.yields) {
            const auto found = info.values.find(value_id);
            if (found == info.values.end() ||
                found->second.region_id != region.id ||
                !dominates(found->second, region.id, position, info)) {
                return reject(CompatibilityError::IR_REFERENCE_INVALID,
                              "region yield is captured or not dominated by its value");
            }
        }
        return true;
    }

    bool visit_instruction_dependencies(
        uint32_t id, const FunctionInfo& info,
        std::unordered_map<uint32_t, uint8_t>& colors) {
        const uint8_t color = colors[id];
        if (color == 1) {
            return reject(CompatibilityError::IR_STATE_INVALID,
                          "instruction dependencies contain a cycle");
        }
        if (color == 2) return true;
        colors[id] = 1;
        const InstructionLocation& location = info.instructions.at(id);
        for (uint32_t input : location.instruction->inputs) {
            const ValueDefinition& definition = info.values.at(input);
            if (!definition.argument && definition.region_id == location.region_id) {
                const Region& region = *info.regions.at(definition.region_id);
                const uint32_t producer = region.instructions[definition.position].id;
                if (!visit_instruction_dependencies(producer, info, colors)) return false;
            }
        }
        for (uint32_t predecessor : location.instruction->effect_predecessors) {
            if (!visit_instruction_dependencies(predecessor, info, colors)) return false;
        }
        colors[id] = 2;
        return true;
    }

    struct Access {
        uint32_t instruction_id = UINT32_MAX;
        uint32_t region_id = UINT32_MAX;
        uint64_t alias = UINT64_MAX;
        bool write = false;
        size_t node_index = SIZE_MAX;
    };

    bool validate_alias_order(const FunctionInfo& info) {
        std::vector<uint32_t> nodes;
        std::unordered_map<uint32_t, size_t> node_indices;
        for (const auto& [id, location] : info.instructions) {
            if (!is_state_access(*location.instruction)) continue;
            node_indices.emplace(id, nodes.size());
            nodes.push_back(id);
        }

        std::vector<size_t> indegrees(nodes.size(), 0);
        for (const auto& [source, successors] : info.effect_successors) {
            (void)source;
            for (uint32_t successor : successors) {
                ++indegrees[node_indices.at(successor)];
            }
        }
        std::vector<size_t> topological;
        topological.reserve(nodes.size());
        for (size_t index = 0; index < nodes.size(); ++index) {
            if (indegrees[index] == 0) topological.push_back(index);
        }
        for (size_t cursor = 0; cursor < topological.size(); ++cursor) {
            const size_t source_index = topological[cursor];
            const auto found = info.effect_successors.find(nodes[source_index]);
            if (found == info.effect_successors.end()) continue;
            for (uint32_t successor : found->second) {
                const size_t successor_index = node_indices.at(successor);
                if (--indegrees[successor_index] == 0) {
                    topological.push_back(successor_index);
                }
            }
        }
        if (topological.size() != nodes.size()) {
            return reject(CompatibilityError::IR_STATE_INVALID,
                          "effect dependency graph is not acyclic");
        }

        const size_t words = (nodes.size() + 63) / 64;
        std::vector<std::vector<uint64_t>> reachable(
            nodes.size(), std::vector<uint64_t>(words, 0));
        for (auto position = topological.rbegin(); position != topological.rend();
             ++position) {
            const size_t source_index = *position;
            const auto found = info.effect_successors.find(nodes[source_index]);
            if (found == info.effect_successors.end()) continue;
            for (uint32_t successor : found->second) {
                const size_t successor_index = node_indices.at(successor);
                reachable[source_index][successor_index / 64] |=
                    uint64_t{1} << (successor_index % 64);
                for (size_t word = 0; word < words; ++word) {
                    reachable[source_index][word] |= reachable[successor_index][word];
                }
            }
        }

        std::vector<Access> accesses;
        for (const auto& [id, location] : info.instructions) {
            if (!is_state_access(*location.instruction)) continue;
            const auto& attributes = std::get<StateAttributes>(location.instruction->attributes);
            const uint32_t state_index = state_ordinals_.at(attributes.state_id);
            const StateReference& state = program_.state_references[state_index];
            const uint64_t alias = state.alias_group == UINT32_MAX
                                       ? (uint64_t{1} << 32) | state_index
                                       : state.alias_group;
            accesses.push_back({id, location.region_id, alias,
                                location.instruction->primitive.code ==
                                    Primitive::StateWrite,
                                node_indices.at(id)});
        }
        for (size_t left = 0; left < accesses.size(); ++left) {
            for (size_t right = left + 1; right < accesses.size(); ++right) {
                const Access& a = accesses[left];
                const Access& b = accesses[right];
                if (a.alias != b.alias || (!a.write && !b.write)) continue;
                const bool a_reaches_b =
                    (reachable[a.node_index][b.node_index / 64] &
                     (uint64_t{1} << (b.node_index % 64))) != 0;
                const bool b_reaches_a =
                    (reachable[b.node_index][a.node_index / 64] &
                     (uint64_t{1} << (a.node_index % 64))) != 0;
                if (a.region_id != b.region_id ||
                    (!a_reaches_b && !b_reaches_a)) {
                    return reject(CompatibilityError::IR_STATE_INVALID,
                                  "mutable alias accesses are not explicitly ordered");
                }
            }
        }
        return true;
    }

    bool validate_effect_roots(const FunctionInfo& info) {
        size_t roots = 0;
        for (const auto& [id, location] : info.instructions) {
            if (location.instruction->primitive.code == Primitive::StateWrite &&
                info.effect_successors.find(id) == info.effect_successors.end()) {
                ++roots;
            }
        }
        return roots <= 1
                   ? true
                   : reject(CompatibilityError::IR_STATE_INVALID,
                            "independent state effects require one explicit ordered root");
    }

    bool validate_exports(const std::unordered_set<uint32_t>& function_ids) {
        if (program_.exports.size() > kMaximumExports) {
            return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                          "export count exceeds the bound");
        }
        std::set<std::pair<uint32_t, uint32_t>> seen;
        for (const ProgramExport& item : program_.exports) {
            if (function_ids.find(item.function_id) == function_ids.end() ||
                !seen.emplace(item.function_id, item.result_index).second) {
                return reject(CompatibilityError::IR_REFERENCE_INVALID,
                              "program export is unresolved or repeated");
            }
            const Function* function = nullptr;
            for (const Function& candidate : program_.functions) {
                if (candidate.id == item.function_id) {
                    function = &candidate;
                    break;
                }
            }
            if (function == nullptr || item.result_index >= function->result_types.size() ||
                item.type != function->result_types[item.result_index]) {
                return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                              "program export type does not match its result");
            }
        }
        return true;
    }
};

class Canonicalizer {
public:
    Canonicalizer(const Program& program, const std::vector<FunctionInfo>& infos,
                  const std::unordered_map<uint32_t, uint32_t>& dimension_ordinals,
                  const std::unordered_map<uint32_t, uint32_t>& state_ordinals)
        : program_(program), dimension_ordinals_(dimension_ordinals),
          state_ordinals_(state_ordinals) {
        for (const FunctionInfo& info : infos) infos_.emplace(info.function->id, &info);
    }

    std::variant<ProgramDigest, CompatibilityReport> finish() {
        if (const auto report = validate_effect_predecessor_ordering()) {
            return *report;
        }
        output_.domain("laplace-program-ir-v1");
        output_.u16(program_.major);
        output_.u16(program_.minor);
        output_.u32(static_cast<uint32_t>(program_.dimension_parameters.size()));
        output_.u32(static_cast<uint32_t>(program_.state_references.size()));

        output_.u32(static_cast<uint32_t>(program_.exports.size()));
        std::vector<const ProgramExport*> ordered_exports;
        ordered_exports.reserve(program_.exports.size());
        for (const ProgramExport& item : program_.exports)
            ordered_exports.push_back(&item);
        std::sort(ordered_exports.begin(), ordered_exports.end(),
                  [this](const ProgramExport* left,
                         const ProgramExport* right) {
                      const ProgramDigest left_function =
                          function_preview(*infos_.at(left->function_id));
                      const ProgramDigest right_function =
                          function_preview(*infos_.at(right->function_id));
                      if (left_function.bytes != right_function.bytes)
                          return left_function.bytes < right_function.bytes;
                      return left->result_index < right->result_index;
                  });
        for (const ProgramExport* item : ordered_exports) emit_export(*item);

        std::vector<const FunctionInfo*> side_only;
        for (const auto& [id, info] : infos_) {
            if (function_slots_.find(id) == function_slots_.end() &&
                !effect_roots(*info).empty()) {
                side_only.push_back(info);
            }
        }
        if (side_only.size() > 1) {
            return compatibility_report(
                CompatibilityError::IR_STATE_INVALID,
                "independent effect-only functions require one explicit ordered root");
        }
        std::sort(side_only.begin(), side_only.end(), [this](const FunctionInfo* left,
                                                            const FunctionInfo* right) {
            return function_preview(*left).bytes < function_preview(*right).bytes;
        });
        for (const FunctionInfo* info : side_only) ensure_function(*info);

        std::vector<std::pair<uint32_t, const FunctionInfo*>> ordered_functions;
        for (const auto& [id, slot] : function_slots_) {
            ordered_functions.push_back({slot, infos_.at(id)});
        }
        std::sort(ordered_functions.begin(), ordered_functions.end());
        output_.u32(static_cast<uint32_t>(ordered_functions.size()));
        for (const auto& [slot, info] : ordered_functions) {
            output_.u32(slot);
            const std::vector<uint32_t> roots = effect_roots(*info);
            output_.u32(static_cast<uint32_t>(roots.size()));
            for (uint32_t root : roots) {
                FunctionState& state = states_.at(info->function->id);
                ensure_instruction(*info, state, root);
                output_.u32(state.instruction_slots.at(root));
            }
        }

        for (const auto& [id, info] : infos_) {
            const auto state = states_.find(id);
            if (state == states_.end() ||
                state->second.visited_instructions.size() != info->instructions.size()) {
                return compatibility_report(
                    CompatibilityError::IR_REFERENCE_INVALID,
                    "program contains an unrooted instruction");
            }
        }
        if (dimension_slots_.size() != program_.dimension_parameters.size()) {
            return compatibility_report(
                CompatibilityError::IR_SHAPE_MISMATCH,
                "program contains an unused dimension parameter");
        }
        if (state_slots_.size() != program_.state_references.size()) {
            return compatibility_report(
                CompatibilityError::IR_STATE_INVALID,
                "program contains an unused state reference");
        }
        return output_.finish();
    }

    std::vector<uint32_t> canonical_dimension_parameter_ids() const {
        std::vector<uint32_t> result(dimension_slots_.size(), UINT32_MAX);
        for (const auto& [id, slot] : dimension_slots_) result.at(slot) = id;
        return result;
    }

    std::vector<uint32_t> canonical_state_reference_ids() const {
        std::vector<uint32_t> result(state_slots_.size(), UINT32_MAX);
        for (const auto& [id, slot] : state_slots_) result.at(slot) = id;
        return result;
    }

    std::vector<uint32_t> canonical_function_ids() const {
        std::vector<uint32_t> result(function_slots_.size(), UINT32_MAX);
        for (const auto& [id, slot] : function_slots_) result.at(slot) = id;
        return result;
    }

private:
    struct FunctionState {
        uint32_t function_slot = UINT32_MAX;
        uint32_t next_instruction_slot = 0;
        uint32_t next_region_slot = 1;
        std::unordered_map<uint32_t, uint32_t> instruction_slots;
        std::unordered_map<uint32_t, uint32_t> region_slots;
        std::unordered_set<uint32_t> visited_instructions;
    };

    const Program& program_;
    const std::unordered_map<uint32_t, uint32_t>& dimension_ordinals_;
    const std::unordered_map<uint32_t, uint32_t>& state_ordinals_;
    std::unordered_map<uint32_t, const FunctionInfo*> infos_;
    std::unordered_map<uint32_t, uint32_t> function_slots_;
    std::unordered_map<uint32_t, FunctionState> states_;
    std::unordered_map<uint32_t, uint32_t> dimension_slots_;
    std::unordered_map<uint32_t, uint32_t> state_slots_;
    std::unordered_map<uint32_t, uint32_t> alias_slots_;
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, ProgramDigest>>
        preview_memos_;
    uint32_t next_alias_slot_ = 0;
    Encoder output_;

    std::vector<std::pair<ProgramDigest, uint32_t>> ordered_effect_predecessors(
        const FunctionInfo& info, const Instruction& item,
        std::unordered_map<uint32_t, ProgramDigest>& memo) const {
        std::vector<std::pair<ProgramDigest, uint32_t>> ordered;
        ordered.reserve(item.effect_predecessors.size());
        for (uint32_t predecessor : item.effect_predecessors) {
            ordered.push_back(
                {preview_instruction(info, predecessor, memo), predecessor});
        }
        std::sort(ordered.begin(), ordered.end(), [](const auto& left,
                                                     const auto& right) {
            return left.first.bytes < right.first.bytes;
        });
        return ordered;
    }

    std::optional<CompatibilityReport> validate_effect_predecessor_ordering() const {
        for (const auto& [function_id, info] : infos_) {
            (void)function_id;
            std::unordered_map<uint32_t, ProgramDigest> memo;
            for (const auto& [instruction_id, location] : info->instructions) {
                (void)instruction_id;
                const auto ordered =
                    ordered_effect_predecessors(*info, *location.instruction, memo);
                for (size_t index = 1; index < ordered.size(); ++index) {
                    if (ordered[index - 1].first == ordered[index].first) {
                        return compatibility_report(
                            CompatibilityError::IR_STATE_INVALID,
                            "structurally ambiguous effect predecessors require "
                            "explicit chaining");
                    }
                }
            }
        }
        return std::nullopt;
    }

    void encode_dimension(const DimensionExpr& expression, Encoder& encoder) {
        encoder.u8(static_cast<uint8_t>(expression.expression));
        if (expression.expression == DimensionExpression::Parameter) {
            const uint32_t id = static_cast<uint32_t>(expression.value);
            const auto existing = dimension_slots_.find(id);
            if (existing != dimension_slots_.end()) {
                encoder.u8(0);
                encoder.u32(existing->second);
            } else {
                const uint32_t slot = static_cast<uint32_t>(dimension_slots_.size());
                dimension_slots_.emplace(id, slot);
                const DimensionParameter& parameter =
                    program_.dimension_parameters[dimension_ordinals_.at(id)];
                encoder.u8(1);
                encoder.u32(slot);
                encoder.u64(parameter.lower);
                encoder.u64(parameter.upper);
            }
        } else {
            encoder.u64(expression.value);
        }
        encoder.u32(static_cast<uint32_t>(expression.operands.size()));
        for (const DimensionExpr& operand : expression.operands) {
            encode_dimension(operand, encoder);
        }
    }

    void encode_type(const ValueType& type, Encoder& encoder) {
        encoder.u8(static_cast<uint8_t>(type.element_type));
        encoder.u32(static_cast<uint32_t>(type.dimensions.size()));
        for (const DimensionExpr& dimension : type.dimensions) {
            encode_dimension(dimension, encoder);
        }
    }

    static void encode_tensor_index(const TensorIndexExpr& expression,
                                    Encoder& encoder) {
        encoder.u8(static_cast<uint8_t>(expression.expression));
        encoder.u64(static_cast<uint64_t>(expression.value));
        encoder.u32(static_cast<uint32_t>(expression.operands.size()));
        for (const TensorIndexExpr& operand : expression.operands)
            encode_tensor_index(operand, encoder);
    }

    void encode_structured_tensor(const StructuredTensorAttributes& attributes,
                                  Encoder& encoder) {
        encoder.u32(attributes.source_count);
        encoder.u32(static_cast<uint32_t>(attributes.iteration_dimensions.size()));
        for (size_t index = 0; index < attributes.iteration_dimensions.size();
             ++index) {
            encode_dimension(attributes.iteration_dimensions[index], encoder);
            encoder.u8(static_cast<uint8_t>(attributes.iterator_kinds[index]));
        }
        encoder.u32(static_cast<uint32_t>(attributes.indexing_maps.size()));
        for (const TensorIndexMap& map : attributes.indexing_maps) {
            encoder.u8(static_cast<uint8_t>(map.bounds));
            encoder.u32(static_cast<uint32_t>(map.results.size()));
            for (const TensorIndexExpr& result : map.results)
                encode_tensor_index(result, encoder);
        }
    }

    void encode_attributes(const Instruction& item, Encoder& encoder) {
        encoder.u8(static_cast<uint8_t>(item.attributes.index()));
        if (const auto* value = std::get_if<ConstantAttributes>(&item.attributes)) {
            encoder.u64(value->bits);
        } else if (const auto* value = std::get_if<LoopAttributes>(&item.attributes)) {
            encoder.u64(value->lower);
            encoder.u64(value->upper);
            encoder.u64(value->step);
        } else if (const auto* value = std::get_if<StateAttributes>(&item.attributes)) {
            const auto existing = state_slots_.find(value->state_id);
            if (existing != state_slots_.end()) {
                encoder.u8(0);
                encoder.u32(existing->second);
            } else {
                const uint32_t slot = static_cast<uint32_t>(state_slots_.size());
                state_slots_.emplace(value->state_id, slot);
                const StateReference& state =
                    program_.state_references[state_ordinals_.at(value->state_id)];
                encoder.u8(1);
                encoder.u32(slot);
                encode_type(state.type, encoder);
                encoder.u8(state.writable ? 1 : 0);
                if (state.alias_group == UINT32_MAX) {
                    encoder.u32(next_alias_slot_++);
                } else {
                    const auto [alias, inserted] =
                        alias_slots_.emplace(state.alias_group, next_alias_slot_);
                    if (inserted) ++next_alias_slot_;
                    encoder.u32(alias->second);
                }
            }
        } else if (const auto* value =
                       std::get_if<StructuredTensorAttributes>(&item.attributes)) {
            encode_structured_tensor(*value, encoder);
        }
    }

    void preview_encode_dimension(const DimensionExpr& expression,
                                  Encoder& encoder) const {
        encoder.u8(static_cast<uint8_t>(expression.expression));
        if (expression.expression == DimensionExpression::Parameter) {
            const uint32_t id = static_cast<uint32_t>(expression.value);
            const DimensionParameter& parameter =
                program_.dimension_parameters[dimension_ordinals_.at(id)];
            encoder.u64(parameter.lower);
            encoder.u64(parameter.upper);
        } else {
            encoder.u64(expression.value);
        }
        encoder.u32(static_cast<uint32_t>(expression.operands.size()));
        for (const DimensionExpr& operand : expression.operands) {
            preview_encode_dimension(operand, encoder);
        }
    }

    void preview_encode_type(const ValueType& type, Encoder& encoder) const {
        encoder.u8(static_cast<uint8_t>(type.element_type));
        encoder.u32(static_cast<uint32_t>(type.dimensions.size()));
        for (const DimensionExpr& dimension : type.dimensions) {
            preview_encode_dimension(dimension, encoder);
        }
    }

    void preview_encode_structured_tensor(
        const StructuredTensorAttributes& attributes, Encoder& encoder) const {
        encoder.u32(attributes.source_count);
        encoder.u32(static_cast<uint32_t>(attributes.iteration_dimensions.size()));
        for (size_t index = 0; index < attributes.iteration_dimensions.size();
             ++index) {
            preview_encode_dimension(attributes.iteration_dimensions[index], encoder);
            encoder.u8(static_cast<uint8_t>(attributes.iterator_kinds[index]));
        }
        encoder.u32(static_cast<uint32_t>(attributes.indexing_maps.size()));
        for (const TensorIndexMap& map : attributes.indexing_maps) {
            encoder.u8(static_cast<uint8_t>(map.bounds));
            encoder.u32(static_cast<uint32_t>(map.results.size()));
            for (const TensorIndexExpr& result : map.results)
                encode_tensor_index(result, encoder);
        }
    }

    void preview_encode_attributes(const Instruction& item,
                                   Encoder& encoder) const {
        encoder.u8(static_cast<uint8_t>(item.attributes.index()));
        if (const auto* value = std::get_if<ConstantAttributes>(&item.attributes)) {
            encoder.u64(value->bits);
        } else if (const auto* value = std::get_if<LoopAttributes>(&item.attributes)) {
            encoder.u64(value->lower);
            encoder.u64(value->upper);
            encoder.u64(value->step);
        } else if (const auto* value = std::get_if<StateAttributes>(&item.attributes)) {
            const StateReference& state =
                program_.state_references[state_ordinals_.at(value->state_id)];
            preview_encode_type(state.type, encoder);
            encoder.u8(state.writable ? 1 : 0);
            encoder.u8(state.alias_group == UINT32_MAX ? 0 : 1);
        } else if (const auto* value =
                       std::get_if<StructuredTensorAttributes>(&item.attributes)) {
            preview_encode_structured_tensor(*value, encoder);
        }
    }

    const Region& entry_region(const FunctionInfo& info) const {
        return *info.regions.at(info.function->entry_region_id);
    }

    FunctionState& ensure_function(const FunctionInfo& info) {
        const auto found = states_.find(info.function->id);
        if (found != states_.end()) return found->second;
        const uint32_t slot = static_cast<uint32_t>(states_.size());
        FunctionState state;
        state.function_slot = slot;
        state.region_slots.emplace(info.function->entry_region_id, 0);
        states_.emplace(info.function->id, std::move(state));
        function_slots_.emplace(info.function->id, slot);
        FunctionState& stored = states_.at(info.function->id);

        output_.u8(1);
        output_.u32(slot);
        const Region& entry = entry_region(info);
        output_.u32(static_cast<uint32_t>(entry.arguments.size()));
        for (const TypedValue& argument : entry.arguments) encode_type(argument.type, output_);
        output_.u32(static_cast<uint32_t>(info.function->result_types.size()));
        for (const ValueType& result : info.function->result_types) encode_type(result, output_);
        output_.u32(static_cast<uint32_t>(entry.yields.size()));
        for (uint32_t result : entry.yields) emit_value(info, stored, result);
        return stored;
    }

    void emit_export(const ProgramExport& item) {
        const FunctionInfo& info = *infos_.at(item.function_id);
        FunctionState& state = ensure_function(info);
        output_.u8(2);
        output_.u32(state.function_slot);
        output_.u32(item.result_index);
        encode_type(item.type, output_);
        emit_value(info, state, entry_region(info).yields[item.result_index]);
    }

    void emit_value(const FunctionInfo& info, FunctionState& state, uint32_t value_id) {
        const ValueDefinition& definition = info.values.at(value_id);
        if (definition.argument) {
            output_.u8(3);
            output_.u32(state.region_slots.at(definition.region_id));
            output_.u32(definition.position);
            return;
        }
        const Region& region = *info.regions.at(definition.region_id);
        const uint32_t instruction_id = region.instructions[definition.position].id;
        ensure_instruction(info, state, instruction_id);
        output_.u8(4);
        output_.u32(state.instruction_slots.at(instruction_id));
        output_.u32(definition.result_index);
    }

    void ensure_instruction(const FunctionInfo& info, FunctionState& state,
                            uint32_t instruction_id) {
        if (state.instruction_slots.find(instruction_id) !=
            state.instruction_slots.end()) {
            return;
        }
        const InstructionLocation& location = info.instructions.at(instruction_id);
        ensure_region_owner(info, state, location.region_id);
        const uint32_t slot = state.next_instruction_slot++;
        state.instruction_slots.emplace(instruction_id, slot);
        state.visited_instructions.insert(instruction_id);
        const Instruction& item = *location.instruction;

        output_.u8(5);
        output_.u32(slot);
        output_.u32(state.region_slots.at(location.region_id));
        output_.u16(static_cast<uint16_t>(item.primitive.code));
        output_.u16(item.primitive.major);
        output_.u16(item.primitive.minor);
        encode_attributes(item, output_);
        output_.u32(static_cast<uint32_t>(item.outputs.size()));
        for (const TypedValue& output : item.outputs) encode_type(output.type, output_);
        output_.u32(static_cast<uint32_t>(item.inputs.size()));
        for (uint32_t input : item.inputs) emit_value(info, state, input);
        output_.u32(static_cast<uint32_t>(item.effect_predecessors.size()));
        auto& memo = preview_memos_[info.function->id];
        for (const auto& [digest, predecessor] :
             ordered_effect_predecessors(info, item, memo)) {
            (void)digest;
            ensure_instruction(info, state, predecessor);
            output_.u32(state.instruction_slots.at(predecessor));
        }
        output_.u32(static_cast<uint32_t>(item.regions.size()));
        for (uint32_t region_id : item.regions) emit_region(info, state, region_id);
    }

    void ensure_region_owner(const FunctionInfo& info, FunctionState& state,
                             uint32_t region_id) {
        if (state.region_slots.find(region_id) != state.region_slots.end()) return;
        const RegionParent& parent = info.parents.at(region_id);
        ensure_instruction(info, state, parent.instruction_id);
    }

    void emit_region(const FunctionInfo& info, FunctionState& state,
                     uint32_t region_id) {
        const uint32_t slot = state.next_region_slot++;
        state.region_slots.emplace(region_id, slot);
        const Region& region = *info.regions.at(region_id);
        output_.u8(6);
        output_.u32(slot);
        output_.u32(static_cast<uint32_t>(region.arguments.size()));
        for (const TypedValue& argument : region.arguments) encode_type(argument.type, output_);
        output_.u32(static_cast<uint32_t>(region.yields.size()));
        for (uint32_t yielded : region.yields) emit_value(info, state, yielded);
    }

    ProgramDigest preview_value(const FunctionInfo& info, uint32_t value_id,
                                std::unordered_map<uint32_t, ProgramDigest>& memo) const {
        const ValueDefinition& definition = info.values.at(value_id);
        Encoder encoder;
        if (definition.argument) {
            encoder.u8(1);
            encoder.u32(definition.position);
            preview_encode_type(*definition.type, encoder);
            return encoder.finish();
        }
        const Region& region = *info.regions.at(definition.region_id);
        encoder.u8(2);
        encoder.digest(preview_instruction(info, region.instructions[definition.position].id,
                                           memo));
        encoder.u32(definition.result_index);
        return encoder.finish();
    }

    ProgramDigest preview_instruction(
        const FunctionInfo& info, uint32_t instruction_id,
        std::unordered_map<uint32_t, ProgramDigest>& memo) const {
        const auto cached = memo.find(instruction_id);
        if (cached != memo.end()) return cached->second;
        const Instruction& item = *info.instructions.at(instruction_id).instruction;
        Encoder encoder;
        encoder.u16(static_cast<uint16_t>(item.primitive.code));
        encoder.u16(item.primitive.major);
        encoder.u16(item.primitive.minor);
        preview_encode_attributes(item, encoder);
        encoder.u32(static_cast<uint32_t>(item.outputs.size()));
        for (const TypedValue& output : item.outputs) {
            preview_encode_type(output.type, encoder);
        }
        encoder.u32(static_cast<uint32_t>(item.inputs.size()));
        for (uint32_t input : item.inputs) {
            encoder.digest(preview_value(info, input, memo));
        }
        encoder.u32(static_cast<uint32_t>(item.effect_predecessors.size()));
        for (const auto& [digest, predecessor] :
             ordered_effect_predecessors(info, item, memo)) {
            (void)predecessor;
            encoder.digest(digest);
        }
        encoder.u32(static_cast<uint32_t>(item.regions.size()));
        for (uint32_t region_id : item.regions) {
            const Region& region = *info.regions.at(region_id);
            encoder.u32(static_cast<uint32_t>(region.arguments.size()));
            for (const TypedValue& argument : region.arguments) {
                preview_encode_type(argument.type, encoder);
            }
            encoder.u32(static_cast<uint32_t>(region.yields.size()));
            for (uint32_t yielded : region.yields) {
                encoder.digest(preview_value(info, yielded, memo));
            }
        }
        const ProgramDigest result = encoder.finish();
        memo.emplace(instruction_id, result);
        return result;
    }

    std::vector<uint32_t> effect_roots(const FunctionInfo& info) const {
        std::vector<uint32_t> roots;
        for (const auto& [id, location] : info.instructions) {
            const Primitive code = location.instruction->primitive.code;
            if (code == Primitive::StateWrite &&
                info.effect_successors.find(id) == info.effect_successors.end()) {
                roots.push_back(id);
            }
        }
        std::unordered_map<uint32_t, ProgramDigest> memo;
        std::sort(roots.begin(), roots.end(), [&](uint32_t left, uint32_t right) {
            const ProgramDigest a = preview_instruction(info, left, memo);
            const ProgramDigest b = preview_instruction(info, right, memo);
            return a.bytes < b.bytes;
        });
        return roots;
    }

    ProgramDigest function_preview(const FunctionInfo& info) const {
        Encoder encoder;
        const Region& entry = entry_region(info);
        encoder.u32(static_cast<uint32_t>(entry.arguments.size()));
        for (const TypedValue& argument : entry.arguments) {
            preview_encode_type(argument.type, encoder);
        }
        encoder.u32(static_cast<uint32_t>(info.function->result_types.size()));
        for (const ValueType& result : info.function->result_types) {
            preview_encode_type(result, encoder);
        }
        encoder.u32(static_cast<uint32_t>(entry.yields.size()));
        std::unordered_map<uint32_t, ProgramDigest> result_memo;
        for (uint32_t result : entry.yields) {
            encoder.digest(preview_value(info, result, result_memo));
        }
        const std::vector<uint32_t> roots = effect_roots(info);
        encoder.u32(static_cast<uint32_t>(roots.size()));
        std::unordered_map<uint32_t, ProgramDigest> memo;
        for (uint32_t root : roots) {
            encoder.digest(preview_instruction(info, root, memo));
        }
        return encoder.finish();
    }
};

} // namespace

VerifiedProgram::VerifiedProgram(
    Program program, ProgramDigest digest,
    std::vector<uint32_t> canonical_dimension_parameter_ids,
    std::vector<uint32_t> canonical_state_reference_ids,
    std::vector<uint32_t> canonical_function_ids)
    : program_(std::move(program)), digest_(digest),
      canonical_dimension_parameter_ids_(
          std::move(canonical_dimension_parameter_ids)),
      canonical_state_reference_ids_(std::move(canonical_state_reference_ids)),
      canonical_function_ids_(std::move(canonical_function_ids)) {}

ProgramVerificationResult verify_and_canonicalize_program(Program program) {
    Verifier verifier(program);
    if (!verifier.run()) return verifier.report();
    Canonicalizer canonicalizer(program, verifier.infos(), verifier.dimension_ordinals(),
                                verifier.state_ordinals());
    auto result = canonicalizer.finish();
    if (const auto* report = std::get_if<CompatibilityReport>(&result)) {
        return *report;
    }
    return VerifiedProgram(
        std::move(program), std::get<ProgramDigest>(result),
        canonicalizer.canonical_dimension_parameter_ids(),
        canonicalizer.canonical_state_reference_ids(),
        canonicalizer.canonical_function_ids());
}

ProgramDigest program_digest(const VerifiedProgram& program) {
    return program.digest_;
}

const Program& program_definition(const VerifiedProgram& program) {
    return program.program_;
}

std::span<const uint32_t>
canonical_dimension_parameter_ids(const VerifiedProgram& program) {
    return program.canonical_dimension_parameter_ids_;
}

std::span<const uint32_t>
canonical_state_reference_ids(const VerifiedProgram& program) {
    return program.canonical_state_reference_ids_;
}

std::span<const uint32_t>
canonical_function_ids(const VerifiedProgram& program) {
    return program.canonical_function_ids_;
}

namespace {

constexpr size_t kProgramWireLimit = 16u * 1024u * 1024u;

CompatibilityReport program_wire_error(CompatibilityError code,
                                       const char* detail) {
    return compatibility_report(code, detail);
}

class ProgramWireWriter {
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
            'L', 'A', 'P', 'I', 'R', 'W', '0', '1'};
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

class ProgramWireReader {
public:
    explicit ProgramWireReader(std::span<const uint8_t> wire) : wire_(wire) {}

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

void write_dimension(ProgramWireWriter& writer, const DimensionExpr& dimension) {
    writer.u8(static_cast<uint8_t>(dimension.expression));
    writer.u64(dimension.value);
    writer.u32(static_cast<uint32_t>(dimension.operands.size()));
    for (const DimensionExpr& operand : dimension.operands)
        write_dimension(writer, operand);
}

void write_type(ProgramWireWriter& writer, const ValueType& type) {
    writer.u8(static_cast<uint8_t>(type.element_type));
    writer.u32(static_cast<uint32_t>(type.dimensions.size()));
    for (const DimensionExpr& dimension : type.dimensions)
        write_dimension(writer, dimension);
}

void write_value(ProgramWireWriter& writer, const TypedValue& value) {
    writer.u32(value.id);
    write_type(writer, value.type);
}

void write_ids(ProgramWireWriter& writer, std::span<const uint32_t> ids) {
    writer.u32(static_cast<uint32_t>(ids.size()));
    for (uint32_t id : ids) writer.u32(id);
}

void write_tensor_index(ProgramWireWriter& writer,
                        const TensorIndexExpr& expression) {
    writer.u8(static_cast<uint8_t>(expression.expression));
    writer.u64(static_cast<uint64_t>(expression.value));
    writer.u32(static_cast<uint32_t>(expression.operands.size()));
    for (const TensorIndexExpr& operand : expression.operands)
        write_tensor_index(writer, operand);
}

void write_structured_tensor(ProgramWireWriter& writer,
                             const StructuredTensorAttributes& attributes) {
    writer.u32(attributes.source_count);
    writer.u32(static_cast<uint32_t>(attributes.iteration_dimensions.size()));
    for (size_t index = 0; index < attributes.iteration_dimensions.size(); ++index) {
        write_dimension(writer, attributes.iteration_dimensions[index]);
        writer.u8(static_cast<uint8_t>(attributes.iterator_kinds[index]));
    }
    writer.u32(static_cast<uint32_t>(attributes.indexing_maps.size()));
    for (const TensorIndexMap& map : attributes.indexing_maps) {
        writer.u8(static_cast<uint8_t>(map.bounds));
        writer.u32(static_cast<uint32_t>(map.results.size()));
        for (const TensorIndexExpr& result : map.results)
            write_tensor_index(writer, result);
    }
}

void write_instruction(ProgramWireWriter& writer, const Instruction& instruction) {
    writer.u32(instruction.id);
    writer.u16(static_cast<uint16_t>(instruction.primitive.code));
    writer.u16(instruction.primitive.major);
    writer.u16(instruction.primitive.minor);
    writer.u8(static_cast<uint8_t>(instruction.attributes.index()));
    if (const auto* value = std::get_if<ConstantAttributes>(&instruction.attributes)) {
        writer.u64(value->bits);
    } else if (const auto* value = std::get_if<LoopAttributes>(&instruction.attributes)) {
        writer.u64(value->lower);
        writer.u64(value->upper);
        writer.u64(value->step);
    } else if (const auto* value = std::get_if<StateAttributes>(&instruction.attributes)) {
        writer.u32(value->state_id);
    } else if (const auto* value =
                   std::get_if<StructuredTensorAttributes>(&instruction.attributes)) {
        write_structured_tensor(writer, *value);
    }
    write_ids(writer, instruction.inputs);
    writer.u32(static_cast<uint32_t>(instruction.outputs.size()));
    for (const TypedValue& output : instruction.outputs) write_value(writer, output);
    write_ids(writer, instruction.regions);
    write_ids(writer, instruction.effect_predecessors);
}

void write_region(ProgramWireWriter& writer, const Region& region) {
    writer.u32(region.id);
    writer.u32(static_cast<uint32_t>(region.arguments.size()));
    for (const TypedValue& argument : region.arguments) write_value(writer, argument);
    writer.u32(static_cast<uint32_t>(region.instructions.size()));
    for (const Instruction& instruction : region.instructions)
        write_instruction(writer, instruction);
    write_ids(writer, region.yields);
}

bool read_dimension(ProgramWireReader& reader, DimensionExpr* dimension,
                    size_t depth) {
    if (!dimension || depth > kMaximumShapeDepth) return false;
    uint8_t expression = 0;
    uint32_t count = 0;
    if (!reader.u8(&expression) || !reader.u64(&dimension->value) ||
        !reader.u32(&count) || count > kMaximumDimensions)
        return false;
    dimension->expression = static_cast<DimensionExpression>(expression);
    dimension->operands.resize(count);
    for (DimensionExpr& operand : dimension->operands)
        if (!read_dimension(reader, &operand, depth + 1)) return false;
    return true;
}

bool read_type(ProgramWireReader& reader, ValueType* type) {
    if (!type) return false;
    uint8_t element = 0;
    uint32_t count = 0;
    if (!reader.u8(&element) || !reader.u32(&count) ||
        count > kMaximumDimensions)
        return false;
    type->element_type = static_cast<ElementType>(element);
    type->dimensions.resize(count);
    for (DimensionExpr& dimension : type->dimensions)
        if (!read_dimension(reader, &dimension, 0)) return false;
    return true;
}

bool read_value(ProgramWireReader& reader, TypedValue* value) {
    return value && reader.u32(&value->id) && read_type(reader, &value->type);
}

bool read_ids(ProgramWireReader& reader, std::vector<uint32_t>* ids,
              size_t maximum) {
    if (!ids) return false;
    uint32_t count = 0;
    if (!reader.u32(&count) || count > maximum ||
        count > reader.remaining() / sizeof(uint32_t))
        return false;
    ids->resize(count);
    for (uint32_t& id : *ids)
        if (!reader.u32(&id)) return false;
    return true;
}

bool read_tensor_index(ProgramWireReader& reader, TensorIndexExpr* expression,
                       size_t depth, size_t* nodes) {
    if (!expression || !nodes || depth > kMaximumShapeDepth ||
        ++*nodes > kMaximumIndexExpressionNodes)
        return false;
    uint8_t kind = 0;
    uint64_t value = 0;
    uint32_t operands = 0;
    if (!reader.u8(&kind) || !reader.u64(&value) || !reader.u32(&operands) ||
        operands > 2)
        return false;
    expression->expression = static_cast<TensorIndexExpression>(kind);
    expression->value = static_cast<int64_t>(value);
    expression->operands.resize(operands);
    for (TensorIndexExpr& operand : expression->operands)
        if (!read_tensor_index(reader, &operand, depth + 1, nodes)) return false;
    return true;
}

bool read_structured_tensor(ProgramWireReader& reader,
                            StructuredTensorAttributes* attributes) {
    if (!attributes || !reader.u32(&attributes->source_count)) return false;
    uint32_t iterations = 0;
    if (!reader.u32(&iterations) || iterations == 0 ||
        iterations > kMaximumDimensions)
        return false;
    attributes->iteration_dimensions.resize(iterations);
    attributes->iterator_kinds.resize(iterations);
    for (uint32_t index = 0; index < iterations; ++index) {
        uint8_t kind = 0;
        if (!read_dimension(reader, &attributes->iteration_dimensions[index], 0) ||
            !reader.u8(&kind))
            return false;
        attributes->iterator_kinds[index] =
            static_cast<TensorIteratorKind>(kind);
    }
    uint32_t maps = 0;
    if (!reader.u32(&maps) || maps == 0 || maps > kMaximumStructuredOperands)
        return false;
    attributes->indexing_maps.resize(maps);
    size_t nodes = 0;
    for (TensorIndexMap& map : attributes->indexing_maps) {
        uint8_t bounds = 0;
        uint32_t results = 0;
        if (!reader.u8(&bounds) || !reader.u32(&results) ||
            results > kMaximumDimensions)
            return false;
        map.bounds = static_cast<TensorBoundsMode>(bounds);
        map.results.resize(results);
        for (TensorIndexExpr& result : map.results)
            if (!read_tensor_index(reader, &result, 0, &nodes)) return false;
    }
    return true;
}

bool read_instruction(ProgramWireReader& reader, Instruction* instruction) {
    if (!instruction) return false;
    uint16_t primitive = 0;
    uint8_t attributes = 0;
    if (!reader.u32(&instruction->id) || !reader.u16(&primitive) ||
        !reader.u16(&instruction->primitive.major) ||
        !reader.u16(&instruction->primitive.minor) || !reader.u8(&attributes))
        return false;
    instruction->primitive.code = static_cast<Primitive>(primitive);
    switch (attributes) {
    case 0:
        instruction->attributes = NoAttributes{};
        break;
    case 1: {
        ConstantAttributes value;
        if (!reader.u64(&value.bits)) return false;
        instruction->attributes = value;
        break;
    }
    case 2: {
        LoopAttributes value;
        if (!reader.u64(&value.lower) || !reader.u64(&value.upper) ||
            !reader.u64(&value.step))
            return false;
        instruction->attributes = value;
        break;
    }
    case 3: {
        StateAttributes value;
        if (!reader.u32(&value.state_id)) return false;
        instruction->attributes = value;
        break;
    }
    case 4: {
        StructuredTensorAttributes value;
        if (!read_structured_tensor(reader, &value)) return false;
        instruction->attributes = std::move(value);
        break;
    }
    default:
        return false;
    }
    if (!read_ids(reader, &instruction->inputs, kMaximumValues)) return false;
    uint32_t outputs = 0;
    if (!reader.u32(&outputs) || outputs > kMaximumValues) return false;
    instruction->outputs.resize(outputs);
    for (TypedValue& output : instruction->outputs)
        if (!read_value(reader, &output)) return false;
    return read_ids(reader, &instruction->regions, kMaximumRegions) &&
           read_ids(reader, &instruction->effect_predecessors,
                    kMaximumEffectEdges);
}

bool read_region(ProgramWireReader& reader, Region* region,
                 size_t* instruction_count, size_t* value_count) {
    if (!region || !instruction_count || !value_count ||
        !reader.u32(&region->id))
        return false;
    uint32_t arguments = 0;
    if (!reader.u32(&arguments) ||
        arguments > kMaximumValues - std::min(*value_count, kMaximumValues))
        return false;
    region->arguments.resize(arguments);
    for (TypedValue& argument : region->arguments)
        if (!read_value(reader, &argument)) return false;
    *value_count += arguments;
    uint32_t instructions = 0;
    if (!reader.u32(&instructions) ||
        instructions > kMaximumInstructions -
                           std::min(*instruction_count, kMaximumInstructions))
        return false;
    region->instructions.resize(instructions);
    for (Instruction& instruction : region->instructions) {
        if (!read_instruction(reader, &instruction)) return false;
        if (instruction.outputs.size() >
            kMaximumValues - std::min(*value_count, kMaximumValues))
            return false;
        *value_count += instruction.outputs.size();
    }
    *instruction_count += instructions;
    return read_ids(reader, &region->yields, kMaximumValues);
}

} // namespace

ProgramWireResult encode_program_wire(const VerifiedProgram& verified) {
    try {
        const Program& program = program_definition(verified);
        ProgramWireWriter writer;
        writer.magic();
        writer.u16(1);
        writer.u16(0);
        writer.u32(0);
        writer.u16(program.major);
        writer.u16(program.minor);
        writer.u32(static_cast<uint32_t>(program.dimension_parameters.size()));
        for (const DimensionParameter& parameter : program.dimension_parameters) {
            writer.u32(parameter.id);
            writer.u64(parameter.lower);
            writer.u64(parameter.upper);
        }
        writer.u32(static_cast<uint32_t>(program.state_references.size()));
        for (const StateReference& state : program.state_references) {
            writer.u32(state.id);
            write_type(writer, state.type);
            writer.u32(state.alias_group);
            writer.u8(state.writable ? 1 : 0);
        }
        writer.u32(static_cast<uint32_t>(program.functions.size()));
        for (const Function& function : program.functions) {
            writer.u32(function.id);
            writer.u32(function.entry_region_id);
            writer.u32(static_cast<uint32_t>(function.result_types.size()));
            for (const ValueType& type : function.result_types)
                write_type(writer, type);
            writer.u32(static_cast<uint32_t>(function.regions.size()));
            for (const Region& region : function.regions) write_region(writer, region);
        }
        writer.u32(static_cast<uint32_t>(program.exports.size()));
        for (const ProgramExport& export_record : program.exports) {
            writer.u32(export_record.function_id);
            writer.u32(export_record.result_index);
            write_type(writer, export_record.type);
        }
        std::vector<uint8_t> wire = writer.finish();
        if (wire.size() > kProgramWireLimit)
            return program_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                      "program wire exceeds its bounded size");
        return wire;
    } catch (const std::bad_alloc&) {
        return program_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                  "program wire allocation failed");
    }
}

ProgramVerificationResult decode_program_wire(std::span<const uint8_t> wire) {
    static constexpr std::array<uint8_t, 8> magic = {
        'L', 'A', 'P', 'I', 'R', 'W', '0', '1'};
    if (wire.size() < 20 || wire.size() > kProgramWireLimit ||
        !std::equal(magic.begin(), magic.end(), wire.begin()))
        return program_wire_error(CompatibilityError::PACKAGE_BAD_MAGIC,
                                  "program wire header is invalid");
    try {
        ProgramWireReader reader(wire.subspan(8));
        uint16_t version = 0, reserved = 0;
        uint32_t total = 0;
        Program program;
        if (!reader.u16(&version) || !reader.u16(&reserved) ||
            !reader.u32(&total) || version != 1 || reserved != 0 ||
            total != wire.size() || !reader.u16(&program.major) ||
            !reader.u16(&program.minor))
            return program_wire_error(CompatibilityError::IR_VERSION_UNSUPPORTED,
                                      "program wire version or length is invalid");
        uint32_t dimensions = 0;
        if (!reader.u32(&dimensions) || dimensions > kMaximumDimensions)
            return program_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                      "program wire dimension declarations are invalid");
        program.dimension_parameters.resize(dimensions);
        for (DimensionParameter& parameter : program.dimension_parameters)
            if (!reader.u32(&parameter.id) || !reader.u64(&parameter.lower) ||
                !reader.u64(&parameter.upper))
                return program_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                          "program wire dimension declaration is truncated");
        uint32_t states = 0;
        if (!reader.u32(&states) || states > kMaximumStateReferences)
            return program_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                      "program wire state declarations are invalid");
        program.state_references.resize(states);
        for (StateReference& state : program.state_references) {
            uint8_t writable = 0;
            if (!reader.u32(&state.id) || !read_type(reader, &state.type) ||
                !reader.u32(&state.alias_group) || !reader.u8(&writable) ||
                writable > 1)
                return program_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                          "program wire state declaration is invalid");
            state.writable = writable != 0;
        }
        uint32_t functions = 0;
        if (!reader.u32(&functions) || functions == 0 ||
            functions > kMaximumFunctions)
            return program_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                      "program wire function declarations are invalid");
        program.functions.resize(functions);
        size_t region_count = 0;
        size_t instruction_count = 0;
        size_t value_count = 0;
        for (Function& function : program.functions) {
            if (!reader.u32(&function.id) ||
                !reader.u32(&function.entry_region_id))
                return program_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                          "program wire function is truncated");
            uint32_t results = 0;
            if (!reader.u32(&results) || results > kMaximumValues)
                return program_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                          "program wire function results are invalid");
            function.result_types.resize(results);
            for (ValueType& type : function.result_types)
                if (!read_type(reader, &type))
                    return program_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                              "program wire result type is invalid");
            uint32_t regions = 0;
            if (!reader.u32(&regions) || regions == 0 ||
                regions > kMaximumRegions - std::min(region_count, kMaximumRegions))
                return program_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                          "program wire regions are invalid");
            function.regions.resize(regions);
            for (Region& region : function.regions)
                if (!read_region(reader, &region, &instruction_count,
                                 &value_count))
                    return program_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                              "program wire region is invalid");
            region_count += regions;
        }
        uint32_t exports = 0;
        if (!reader.u32(&exports) || exports > kMaximumExports)
            return program_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                      "program wire exports are invalid");
        program.exports.resize(exports);
        for (ProgramExport& export_record : program.exports)
            if (!reader.u32(&export_record.function_id) ||
                !reader.u32(&export_record.result_index) ||
                !read_type(reader, &export_record.type))
                return program_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                          "program wire export is invalid");
        if (!reader.finished())
            return program_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                      "program wire has trailing bytes");
        return verify_and_canonicalize_program(std::move(program));
    } catch (const std::bad_alloc&) {
        return program_wire_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                  "program wire allocation failed");
    }
}

} // namespace Laplace
