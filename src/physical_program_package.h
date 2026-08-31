#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "artifact_index.h"
#include "physical_program_interpreter.h"
#include "program_ir.h"
#include "semantic_manifest.h"

namespace Laplace {

class RuntimePackage;

// A physical program is identified by its canonical bytes.  Records carry
// no source-format vocabulary; planes are anonymous operands of the program.
struct PhysicalProgramRecord {
    PhysicalProgramDigest digest{};
    std::vector<uint8_t> wire;
    LogicalTensorType logical_type;
    friend bool operator==(const PhysicalProgramRecord&, const PhysicalProgramRecord&) = default;
};

struct PhysicalPlaneSource {
    uint32_t plane = UINT32_MAX;
    ArtifactId artifact_id{};
    uint64_t offset = 0;
    uint64_t length = 0;
    friend bool operator==(const PhysicalPlaneSource&, const PhysicalPlaneSource&) = default;
};

struct PhysicalResourceBinding {
    uint32_t resource_id = UINT32_MAX;
    PhysicalProgramDigest program_digest{};
    std::vector<PhysicalPlaneSource> planes;
    // A verified semantic package binds a physical resource to one entry value
    // in one function. Both fields are UINT32_MAX only for the transitional
    // manifest adapter, which cannot claim category-free semantic linkage.
    uint32_t semantic_function_id = UINT32_MAX;
    uint32_t semantic_value_id = UINT32_MAX;
    friend bool operator==(const PhysicalResourceBinding&, const PhysicalResourceBinding&) = default;
};

struct PhysicalResourcePlaneView {
    uint32_t plane = UINT32_MAX;
    ArtifactId artifact_id{};
    std::span<const uint8_t> bytes;
};

class VerifiedPhysicalProgramPackage {
  public:
    VerifiedPhysicalProgramPackage(const VerifiedPhysicalProgramPackage&) = default;
    VerifiedPhysicalProgramPackage(VerifiedPhysicalProgramPackage&&) noexcept = default;
    VerifiedPhysicalProgramPackage& operator=(const VerifiedPhysicalProgramPackage&) = default;
    VerifiedPhysicalProgramPackage& operator=(VerifiedPhysicalProgramPackage&&) noexcept = default;

    const ArtifactIndex& physical_index() const noexcept { return physical_; }
    const std::optional<VerifiedProgram>& semantic_program() const noexcept { return semantic_; }
    const std::optional<SemanticManifest>& semantic_manifest() const noexcept {
        return manifest_;
    }
    std::span<const PhysicalProgram> programs() const noexcept { return programs_; }
    std::span<const LogicalTensorType> program_logical_types() const noexcept {
        return logical_types_;
    }
    std::span<const PhysicalResourceBinding> resources() const noexcept { return resources_; }
    const Sha256Digest& digest() const noexcept { return digest_; }
    const PhysicalProgram* find_program(const PhysicalProgramDigest& digest) const noexcept;
    std::variant<std::vector<PhysicalResourcePlaneView>, CompatibilityReport>
    resolve_resource(uint32_t resource_id) const;

  private:
    ArtifactIndex physical_;
    std::optional<VerifiedProgram> semantic_;
    std::optional<SemanticManifest> manifest_;
    std::vector<PhysicalProgram> programs_;
    std::vector<LogicalTensorType> logical_types_;
    std::vector<PhysicalResourceBinding> resources_;
    Sha256Digest digest_{};

    VerifiedPhysicalProgramPackage(ArtifactIndex physical, VerifiedProgram semantic,
                                   std::vector<PhysicalProgram> programs,
                                   std::vector<LogicalTensorType> logical_types,
                                   std::vector<PhysicalResourceBinding> resources,
                                   Sha256Digest digest);
    VerifiedPhysicalProgramPackage(ArtifactIndex physical, SemanticManifest manifest,
                                   std::vector<PhysicalProgram> programs,
                                   std::vector<LogicalTensorType> logical_types,
                                   std::vector<PhysicalResourceBinding> resources,
                                   Sha256Digest digest);
    friend std::variant<VerifiedPhysicalProgramPackage, CompatibilityReport>
    load_physical_program_package(
        ArtifactIndex physical, VerifiedProgram semantic,
        std::span<const PhysicalProgramRecord> programs,
        std::span<const PhysicalResourceBinding> resources);
    friend std::variant<VerifiedPhysicalProgramPackage, CompatibilityReport>
    load_physical_program_package(
        ArtifactIndex physical, SemanticManifest manifest,
        std::span<const PhysicalProgramRecord> programs,
        std::span<const PhysicalResourceBinding> resources);
};

using PhysicalProgramPackageResult =
    std::variant<VerifiedPhysicalProgramPackage, CompatibilityReport>;

using PhysicalProgramPackageWireResult =
    std::variant<std::vector<uint8_t>, CompatibilityReport>;
using PhysicalProgramPackageRecords =
    std::pair<std::vector<PhysicalProgramRecord>,
              std::vector<PhysicalResourceBinding>>;
using PhysicalProgramPackageDecodeResult =
    std::variant<PhysicalProgramPackageRecords, CompatibilityReport>;

// Validates that a carried physical package belongs to the exact authoritative
// semantic package that is about to own the session.  This checks identity only;
// it never copies or reinterprets source bytes.
std::variant<std::monostate, CompatibilityReport>
validate_physical_program_package_for_runtime(
    const RuntimePackage& runtime,
    const VerifiedPhysicalProgramPackage& physical);

PhysicalProgramPackageWireResult encode_physical_program_package_records(
    std::span<const PhysicalProgramRecord> programs,
    std::span<const PhysicalResourceBinding> resources);
PhysicalProgramPackageDecodeResult decode_physical_program_package_records(
    std::span<const uint8_t> wire);

PhysicalProgramPackageResult load_physical_program_package(
    ArtifactIndex physical, VerifiedProgram semantic,
    std::span<const PhysicalProgramRecord> programs,
    std::span<const PhysicalResourceBinding> resources);

PhysicalProgramPackageResult load_physical_program_package(
    ArtifactIndex physical, SemanticManifest manifest,
    std::span<const PhysicalProgramRecord> programs,
    std::span<const PhysicalResourceBinding> resources);

PhysicalProgramPackageResult load_physical_program_package_wire(
    ArtifactIndex physical, SemanticManifest manifest,
    std::span<const uint8_t> wire);

} // namespace Laplace
