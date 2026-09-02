#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

#include "physical_program_package.h"
#include "container_schema_program.h"
#include "program_state.h"
#include "token_program.h"

namespace Laplace {

struct TokenProgramSource {
    ArtifactId artifact_id{};
    uint64_t offset = 0;
    uint64_t length = 0;
    Sha256Digest digest{};
    friend bool operator==(const TokenProgramSource&,
                           const TokenProgramSource&) = default;
};

enum class TokenEndpointKind : uint8_t {
    InputToken = 1,
    OutputScores = 2,
};

struct TokenEndpointBinding {
    TokenEndpointKind kind = TokenEndpointKind::InputToken;
    uint32_t semantic_function_id = UINT32_MAX;
    // Entry value ID for InputToken; function result index for OutputScores.
    uint32_t semantic_value = UINT32_MAX;
    friend bool operator==(const TokenEndpointBinding&,
                           const TokenEndpointBinding&) = default;
};

struct ContainerArtifactSection {
    uint32_t section_id = 0;
    ArtifactId artifact_id{};
    ArtifactRole role = ArtifactRole::Primary;
    friend bool operator==(const ContainerArtifactSection&,
                           const ContainerArtifactSection&) = default;
};

struct ProgramIngressManifest {
    uint16_t major = 1;
    uint16_t minor = 0;
    uint32_t package_section_id = 0;
    std::vector<ContainerSchemaProgram> schemas;
    std::vector<ContainerArtifactSection> artifact_sections;
};

class VerifiedProgramPackage {
public:
    VerifiedProgramPackage(const VerifiedProgramPackage&) = default;
    VerifiedProgramPackage(VerifiedProgramPackage&&) noexcept = default;
    VerifiedProgramPackage& operator=(const VerifiedProgramPackage&) = default;
    VerifiedProgramPackage& operator=(VerifiedProgramPackage&&) noexcept = default;

    bool complete() const noexcept { return true; }
    const VerifiedProgram& semantic_program() const noexcept;
    const VerifiedStateSchema& state_schema() const noexcept { return state_; }
    const TokenProgram& token_program() const noexcept { return token_; }
    const TokenProgramSource& token_source() const noexcept { return token_source_; }
    std::span<const TokenEndpointBinding> token_bindings() const noexcept {
        return token_bindings_;
    }
    const VerifiedPhysicalProgramPackage& physical_package() const noexcept {
        return physical_;
    }
    const Sha256Digest& digest() const noexcept { return digest_; }

private:
    VerifiedPhysicalProgramPackage physical_;
    VerifiedStateSchema state_;
    TokenProgram token_;
    TokenProgramSource token_source_;
    std::vector<TokenEndpointBinding> token_bindings_;
    Sha256Digest digest_{};

    VerifiedProgramPackage(VerifiedPhysicalProgramPackage physical,
                           VerifiedStateSchema state, TokenProgram token,
                           TokenProgramSource token_source,
                           std::vector<TokenEndpointBinding> token_bindings,
                           Sha256Digest digest);
    friend std::variant<VerifiedProgramPackage, CompatibilityReport>
    build_program_package(
        ArtifactIndex, VerifiedProgram, VerifiedStateSchema,
        TokenProgramSource, std::span<const TokenEndpointBinding>,
        std::span<const PhysicalProgramRecord>,
        std::span<const PhysicalResourceBinding>);
};

using ProgramPackageResult =
    std::variant<VerifiedProgramPackage, CompatibilityReport>;
using ProgramPackageWireResult =
    std::variant<std::vector<uint8_t>, CompatibilityReport>;
using ProgramIngressManifestResult =
    std::variant<ProgramIngressManifest, CompatibilityReport>;

ProgramPackageResult build_program_package(
    ArtifactIndex physical, VerifiedProgram semantic,
    VerifiedStateSchema state, TokenProgramSource token_source,
    std::span<const TokenEndpointBinding> token_bindings,
    std::span<const PhysicalProgramRecord> programs,
    std::span<const PhysicalResourceBinding> resources);

ProgramPackageWireResult
encode_program_package(const VerifiedProgramPackage& package);

ProgramPackageResult decode_program_package(
    ArtifactIndex physical, std::span<const uint8_t> wire);

ProgramPackageResult decode_container_program_package(
    ArtifactIndex physical,
    std::span<const ContainerSchemaProgram> schemas,
    std::span<const uint8_t> container,
    uint32_t package_section_id);

ProgramPackageResult load_container_program_package(
    const PackageView& container,
    std::span<const ContainerSchemaProgram> schemas,
    std::span<const ContainerArtifactSection> artifact_sections,
    uint32_t package_section_id);

ProgramPackageWireResult encode_program_ingress_manifest(
    const ProgramIngressManifest& manifest);

ProgramIngressManifestResult decode_program_ingress_manifest(
    std::span<const uint8_t> wire);

ProgramPackageResult load_program_package(
    const PackageView& container, std::span<const uint8_t> ingress_manifest);

ProgramPackageResult load_program_package(
    std::string_view container_path, std::string_view ingress_manifest_path);

} // namespace Laplace
