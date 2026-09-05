#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <variant>
#include <vector>

#include "program_ir.h"
#include "physical_program_interpreter.h"

namespace Laplace {

class RuntimePackage;
class MetalProgramCompiler;
class MetalPhysicalCompiler;
class VerifiedPhysicalProgramPackage;
class VerifiedProgramPackage;

struct MetalProgramValue {
    ElementType type = ElementType::F32;
    std::vector<uint64_t> extents;
    std::vector<uint64_t> bits;
    friend bool operator==(const MetalProgramValue&,
                           const MetalProgramValue&) = default;
};

struct MetalProgramInput {
    uint32_t value_id = UINT32_MAX;
    MetalProgramValue value;
};

struct MetalProgramInputStep {
    std::vector<MetalProgramInput> inputs;
};

struct MetalProgramExecutionAudit {
    ProgramDigest program_digest{};
    std::array<uint8_t, 32> lowering_digest{};
    uint64_t explicit_upload_bytes = 0;
    uint64_t zero_copy_plane_bytes = 0;
    uint64_t persistent_plane_bytes = 0;
    uint64_t explicit_download_bytes = 0;
    uint32_t command_buffers = 0;
    uint32_t implicit_weight_copies = 0;
    uint32_t mapped_artifact_buffer_count = 0;
    uint32_t thread_execution_width = 0;
    uint32_t max_total_threads_per_threadgroup = 0;
    double gpu_time_ms = 0.0;
    double cpu_wait_ms = 0.0;
    uint64_t state_generation = 0;
    // Intermediate outputs only; excludes inputs, weights and transactional state.
    uint64_t intermediate_buffer_allocations = 0;
    uint64_t intermediate_buffer_bytes = 0;
};

struct MetalProgramResult {
    std::vector<MetalProgramValue> exports;
    MetalProgramExecutionAudit audit;
};

using MetalProgramExecutionResult =
    std::variant<MetalProgramResult, CompatibilityReport>;

class MetalProgramExecutable {
public:
    ~MetalProgramExecutable();
    MetalProgramExecutable(MetalProgramExecutable&&) noexcept;
    MetalProgramExecutable& operator=(MetalProgramExecutable&&) noexcept;
    MetalProgramExecutable(const MetalProgramExecutable&) = delete;
    MetalProgramExecutable& operator=(const MetalProgramExecutable&) = delete;

    ProgramDigest program_digest() const noexcept;
    std::array<uint8_t, 32> lowering_digest() const noexcept;
    MetalProgramExecutionResult execute(
        std::span<const MetalProgramInput> inputs) const;
    MetalProgramExecutionResult execute_sequence(
        std::span<const MetalProgramInputStep> steps) const;

private:
    struct Impl;
    explicit MetalProgramExecutable(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;

    friend std::variant<MetalProgramExecutable, CompatibilityReport>
    compile_metal_program(const VerifiedProgram& program);
    friend std::variant<MetalProgramExecutable, CompatibilityReport>
    compile_metal_program(const VerifiedPhysicalProgramPackage& package);
    friend class MetalProgramCompiler;
};

using MetalProgramCompileResult =
    std::variant<MetalProgramExecutable, CompatibilityReport>;

MetalProgramCompileResult compile_metal_program(const VerifiedProgram& program);
MetalProgramCompileResult compile_metal_program(
    const VerifiedPhysicalProgramPackage& package);
MetalProgramCompileResult compile_metal_program(
    const VerifiedProgramPackage& package);

struct MetalPhysicalProgramAudit {
    PhysicalProgramDigest physical_program_digest{};
    std::array<uint8_t, 32> lowering_digest{};
    uint64_t explicit_upload_bytes = 0;
    uint64_t zero_copy_plane_bytes = 0;
    uint64_t persistent_plane_bytes = 0;
    uint64_t explicit_download_bytes = 0;
    uint32_t command_buffers = 0;
    uint32_t implicit_weight_copies = 0;
    uint32_t thread_execution_width = 0;
    uint32_t max_total_threads_per_threadgroup = 0;
};

struct MetalPhysicalProgramResult {
    MetalProgramValue value;
    MetalPhysicalProgramAudit audit;
};

using MetalPhysicalProgramExecutionResult =
    std::variant<MetalPhysicalProgramResult, CompatibilityReport>;

class MetalPhysicalProgramExecutable {
public:
    ~MetalPhysicalProgramExecutable();
    MetalPhysicalProgramExecutable(MetalPhysicalProgramExecutable&&) noexcept;
    MetalPhysicalProgramExecutable& operator=(
        MetalPhysicalProgramExecutable&&) noexcept;
    MetalPhysicalProgramExecutable(const MetalPhysicalProgramExecutable&) = delete;
    MetalPhysicalProgramExecutable& operator=(
        const MetalPhysicalProgramExecutable&) = delete;

    PhysicalProgramDigest physical_program_digest() const noexcept;
    std::array<uint8_t, 32> lowering_digest() const noexcept;
    MetalPhysicalProgramExecutionResult execute() const;

private:
    struct Impl;
    explicit MetalPhysicalProgramExecutable(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;

    friend std::variant<MetalPhysicalProgramExecutable, CompatibilityReport>
    compile_metal_physical_program(const VerifiedPhysicalProgram& program);
    friend std::variant<MetalPhysicalProgramExecutable, CompatibilityReport>
    compile_metal_physical_resource(const RuntimePackage& package,
                                    uint32_t resource_id);
    friend class MetalPhysicalCompiler;
};

using MetalPhysicalProgramCompileResult =
    std::variant<MetalPhysicalProgramExecutable, CompatibilityReport>;

MetalPhysicalProgramCompileResult compile_metal_physical_program(
    const VerifiedPhysicalProgram& program);

// Compiles one authority-bound package resource. External planes stay owned by
// the immutable package mapping and are bound to Metal without a byte copy.
MetalPhysicalProgramCompileResult compile_metal_physical_resource(
    const RuntimePackage& package, uint32_t resource_id);

} // namespace Laplace
