#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <variant>
#include <vector>

#include "physical_program.h"

namespace Laplace {

// The reference interpreter owns every external byte plane for the lifetime of
// a verified program. Artifact-backed product bindings are introduced at the
// package boundary; this qualification API never retains a naked span.
struct PhysicalPlaneBinding {
    uint32_t plane = UINT32_MAX;
    std::shared_ptr<const std::vector<uint8_t>> bytes;
    uint64_t offset = 0;
    uint64_t length = 0;
};

struct ScalarValue {
    ElementType type = ElementType::F32;
    uint64_t bits = 0;
    friend bool operator==(ScalarValue, ScalarValue) = default;
};

enum class PhysicalInterpretError : uint8_t {
    CoordinateInvalid = 1,
    AddressOutOfBounds = 2,
    ArithmeticOverflow = 3,
    NumericalPolicyRejected = 4,
    InternalInvariant = 5,
};

using PhysicalInterpretResult =
    std::variant<ScalarValue, PhysicalInterpretError>;

class VerifiedPhysicalProgram {
  public:
    VerifiedPhysicalProgram(const VerifiedPhysicalProgram&) = default;
    VerifiedPhysicalProgram(VerifiedPhysicalProgram&&) noexcept = default;
    VerifiedPhysicalProgram& operator=(const VerifiedPhysicalProgram&) = default;
    VerifiedPhysicalProgram& operator=(VerifiedPhysicalProgram&&) noexcept =
        default;

    const PhysicalProgram& program() const noexcept { return program_; }
    const LogicalTensorType& logical_type() const noexcept { return logical_; }
    const PhysicalProgramDigest& digest() const noexcept { return digest_; }

  private:
    VerifiedPhysicalProgram() = default;
    PhysicalProgram program_;
    LogicalTensorType logical_;
    PhysicalProgramDigest digest_{};
    std::vector<PhysicalPlaneBinding> bindings_;

    friend std::variant<VerifiedPhysicalProgram, CompatibilityReport>
    verify_physical_program(PhysicalProgram,
                            std::span<const PhysicalPlaneBinding>,
                            const LogicalTensorType&);
    friend PhysicalInterpretResult interpret_physical_value(
        const VerifiedPhysicalProgram&, std::span<const uint64_t>);
};

using VerifiedPhysicalProgramResult =
    std::variant<VerifiedPhysicalProgram, CompatibilityReport>;

VerifiedPhysicalProgramResult verify_physical_program(
    PhysicalProgram program, std::span<const PhysicalPlaneBinding> planes,
    const LogicalTensorType& logical_type);

PhysicalInterpretResult interpret_physical_value(
    const VerifiedPhysicalProgram& program,
    std::span<const uint64_t> logical_coordinate);

} // namespace Laplace
