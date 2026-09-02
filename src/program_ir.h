#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include "compatibility_report.h"

namespace Laplace {

enum class ElementType : uint8_t {
    I1 = 1,
    I32 = 2,
    U32 = 3,
    U64 = 4,
    F16 = 5,
    F32 = 6,
};

enum class DimensionExpression : uint8_t {
    Constant = 1,
    Parameter = 2,
    Add = 3,
    Multiply = 4,
    CeilDivide = 5,
};

struct DimensionExpr {
    DimensionExpression expression = DimensionExpression::Constant;
    uint64_t value = 0;
    std::vector<DimensionExpr> operands;
    friend bool operator==(const DimensionExpr&, const DimensionExpr&) = default;
};

struct DimensionParameter {
    uint32_t id = UINT32_MAX;
    uint64_t lower = 0;
    uint64_t upper = 0;
    friend bool operator==(const DimensionParameter&, const DimensionParameter&) = default;
};

struct ValueType {
    ElementType element_type = ElementType::F32;
    std::vector<DimensionExpr> dimensions;
    friend bool operator==(const ValueType&, const ValueType&) = default;
};

struct TypedValue {
    uint32_t id = UINT32_MAX;
    ValueType type;
    friend bool operator==(const TypedValue&, const TypedValue&) = default;
};

enum class Primitive : uint16_t {
    Constant = 1,
    Add = 2,
    Multiply = 3,
    BoundedLoop = 4,
    StateRead = 5,
    StateWrite = 6,
    StructuredTensor = 7,
    Subtract = 8,
    Divide = 9,
    Maximum = 10,
    Negate = 11,
    Exp = 12,
    Log = 13,
    Rsqrt = 14,
    Sin = 15,
    Cos = 16,
};

inline constexpr size_t kPrimitiveCount =
    static_cast<size_t>(Primitive::Cos) + 1;

struct PrimitiveVersion {
    Primitive code = Primitive::Constant;
    uint16_t major = 1;
    uint16_t minor = 0;
    friend bool operator==(PrimitiveVersion, PrimitiveVersion) = default;
};

struct NoAttributes {
    friend bool operator==(NoAttributes, NoAttributes) = default;
};

struct ConstantAttributes {
    uint64_t bits = 0;
    friend bool operator==(ConstantAttributes, ConstantAttributes) = default;
};

struct LoopAttributes {
    // The interval is [lower, upper) with a positive step. Verification also
    // requires the post-body induction increment to fit in U64.
    uint64_t lower = 0;
    uint64_t upper = 0;
    uint64_t step = 0;
    friend bool operator==(LoopAttributes, LoopAttributes) = default;
};

struct StateAttributes {
    uint32_t state_id = UINT32_MAX;
    friend bool operator==(StateAttributes, StateAttributes) = default;
};

enum class TensorIteratorKind : uint8_t {
    Parallel = 1,
    Reduction = 2,
};

enum class TensorIndexExpression : uint8_t {
    Constant = 1,
    Iterator = 2,
    Add = 3,
    Multiply = 4,
    FloorDivide = 5,
    Remainder = 6,
    SourceScalar = 7,
};

struct TensorIndexExpr {
    TensorIndexExpression expression = TensorIndexExpression::Constant;
    int64_t value = 0;
    std::vector<TensorIndexExpr> operands;
    friend bool operator==(const TensorIndexExpr&, const TensorIndexExpr&) = default;
};

enum class TensorBoundsMode : uint8_t {
    Reject = 1,
    Zero = 2,
};

struct TensorIndexMap {
    TensorBoundsMode bounds = TensorBoundsMode::Reject;
    std::vector<TensorIndexExpr> results;
    friend bool operator==(const TensorIndexMap&, const TensorIndexMap&) = default;
};

struct StructuredTensorAttributes {
    uint32_t source_count = 0;
    std::vector<DimensionExpr> iteration_dimensions;
    std::vector<TensorIteratorKind> iterator_kinds;
    std::vector<TensorIndexMap> indexing_maps;
    friend bool operator==(const StructuredTensorAttributes&,
                           const StructuredTensorAttributes&) = default;
};

using PrimitiveAttributes =
    std::variant<NoAttributes, ConstantAttributes, LoopAttributes, StateAttributes,
                 StructuredTensorAttributes>;

struct Instruction {
    uint32_t id = UINT32_MAX;
    PrimitiveVersion primitive;
    std::vector<uint32_t> inputs;
    std::vector<TypedValue> outputs;
    std::vector<uint32_t> regions;
    std::vector<uint32_t> effect_predecessors;
    PrimitiveAttributes attributes = NoAttributes{};
};

struct Region {
    uint32_t id = UINT32_MAX;
    std::vector<TypedValue> arguments;
    std::vector<Instruction> instructions;
    std::vector<uint32_t> yields;
};

struct Function {
    uint32_t id = UINT32_MAX;
    uint32_t entry_region_id = UINT32_MAX;
    std::vector<Region> regions;
    std::vector<ValueType> result_types;
};

struct StateReference {
    uint32_t id = UINT32_MAX;
    ValueType type;
    uint32_t alias_group = UINT32_MAX;
    bool writable = false;
};

struct ProgramExport {
    uint32_t function_id = UINT32_MAX;
    uint32_t result_index = UINT32_MAX;
    ValueType type;
};

struct Program {
    uint16_t major = 1;
    uint16_t minor = 0;
    std::vector<DimensionParameter> dimension_parameters;
    std::vector<StateReference> state_references;
    std::vector<Function> functions;
    std::vector<ProgramExport> exports;
};

struct ProgramDigest {
    std::array<uint8_t, 32> bytes{};
    friend bool operator==(ProgramDigest, ProgramDigest) = default;
};

struct VerifiedProgram {
    VerifiedProgram(const VerifiedProgram&) = default;
    VerifiedProgram(VerifiedProgram&&) noexcept = default;
    VerifiedProgram& operator=(const VerifiedProgram&) = default;
    VerifiedProgram& operator=(VerifiedProgram&&) noexcept = default;

private:
    Program program_;
    ProgramDigest digest_;
    std::vector<uint32_t> canonical_dimension_parameter_ids_;
    std::vector<uint32_t> canonical_state_reference_ids_;
    std::vector<uint32_t> canonical_function_ids_;

    VerifiedProgram(Program program, ProgramDigest digest,
                    std::vector<uint32_t> canonical_dimension_parameter_ids,
                    std::vector<uint32_t> canonical_state_reference_ids,
                    std::vector<uint32_t> canonical_function_ids);
    friend std::variant<VerifiedProgram, CompatibilityReport>
    verify_and_canonicalize_program(Program program);
    friend ProgramDigest program_digest(const VerifiedProgram& program);
    friend const Program& program_definition(const VerifiedProgram& program);
    friend std::span<const uint32_t>
    canonical_dimension_parameter_ids(const VerifiedProgram& program);
    friend std::span<const uint32_t>
    canonical_state_reference_ids(const VerifiedProgram& program);
    friend std::span<const uint32_t>
    canonical_function_ids(const VerifiedProgram& program);
};

using ProgramVerificationResult = std::variant<VerifiedProgram, CompatibilityReport>;
using ProgramWireResult =
    std::variant<std::vector<uint8_t>, CompatibilityReport>;

ProgramVerificationResult verify_and_canonicalize_program(Program program);
ProgramWireResult encode_program_wire(const VerifiedProgram& program);
ProgramVerificationResult decode_program_wire(std::span<const uint8_t> wire);
ProgramDigest program_digest(const VerifiedProgram& program);
const Program& program_definition(const VerifiedProgram& program);
std::span<const uint32_t>
canonical_dimension_parameter_ids(const VerifiedProgram& program);
std::span<const uint32_t>
canonical_state_reference_ids(const VerifiedProgram& program);
std::span<const uint32_t>
canonical_function_ids(const VerifiedProgram& program);

} // namespace Laplace
