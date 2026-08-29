#include "product_package.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <sys/stat.h>
#include <utility>

#include "artifact_set.h"
#include "gguf_product_compiler.h"
#include "gguf_index.h"
#include "mlx_package.h"

namespace Laplace {

namespace {

CompatibilityReport tokenizer_error(std::string detail) {
    CompatibilityReport report = package_report(
        CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED, std::move(detail));
    report.artifact_id = ArtifactId{2};
    return report;
}

bool prompt_equivalent(const TokenProgram& program, const PromptTemplate& prompt) {
    const auto& instructions = program.definition().prompt;
    size_t instruction_index = 0;
    size_t operation_index = 0;
    while (instruction_index != instructions.size()) {
        const PromptInstruction& instruction = instructions[instruction_index++];
        if (instruction.opcode == PromptOpcode::End) {
            return instruction_index == instructions.size() && operation_index == prompt.operations.size();
        }
        if (operation_index == prompt.operations.size()) return false;
        const PromptOperation& operation = prompt.operations[operation_index++];
        if (instruction.opcode == PromptOpcode::EmitUserText) {
            if (operation.kind != PromptOperationKind::AppendInputText) return false;
            continue;
        }
        if (instruction.opcode != PromptOpcode::EmitLiteralUtf8 &&
            instruction.opcode != PromptOpcode::EmitGenerationPrompt) {
            return false;
        }
        if (operation.kind != PromptOperationKind::AppendLiteral ||
            operation.literal.size() != instruction.literal.size() ||
            !std::equal(operation.literal.begin(), operation.literal.end(),
                        reinterpret_cast<const uint8_t*>(instruction.literal.data()))) {
            return false;
        }
    }
    return false;
}

std::optional<CompatibilityReport> validate_token_program_authority(const TokenProgram& program,
                                                                    const TokenContract& contract) {
    const TokenProgramDefinition& definition = program.definition();
    if (contract.tokenizer_version != program.wire_major_version()) {
        return tokenizer_error("tokenizer program wire version does not match the semantic package");
    }
    if (definition.vocabulary.size() != contract.vocabulary_size) {
        return tokenizer_error("tokenizer program vocabulary size does not match the semantic package");
    }
    if (program.vocabulary_digest() != contract.vocabulary_digest) {
        return tokenizer_error("tokenizer program vocabulary digest does not match the semantic package");
    }
    if (program.prompt_digest() != contract.authoritative_template_digest) {
        return tokenizer_error("tokenizer program prompt digest does not match the semantic package");
    }
    if (definition.postprocessor.kind == PostprocessorKind::AddBosEos) {
        if ((definition.postprocessor.flags & static_cast<uint8_t>(PostprocessorFlags::AddBos)) != 0 &&
            definition.postprocessor.bos_token_id != contract.bos_id) {
            return tokenizer_error("tokenizer program BOS behavior does not match the semantic package");
        }
        if ((definition.postprocessor.flags & static_cast<uint8_t>(PostprocessorFlags::AddEos)) != 0 &&
            definition.postprocessor.eos_token_id != contract.eos_id) {
            return tokenizer_error("tokenizer program EOS behavior does not match the semantic package");
        }
    }
    const auto* prompt = std::get_if<PromptTemplate>(&contract.prompt_encoding());
    if (prompt == nullptr || !prompt_equivalent(program, *prompt)) {
        return tokenizer_error("tokenizer program prompt operations do not match the semantic package");
    }
    return std::nullopt;
}

std::variant<ArtifactIndex, CompatibilityReport>
add_token_program_artifact(ArtifactIndex physical, PackageView token_program) {
    ArtifactIndexInput input;
    input.artifacts.assign(physical.artifacts().begin(), physical.artifacts().end());
    input.artifacts.push_back(std::move(token_program));
    input.metadata_facts.assign(physical.metadata_facts().begin(), physical.metadata_facts().end());
    input.package_facts.assign(physical.package_facts().begin(), physical.package_facts().end());
    input.tensors.assign(physical.tensors().begin(), physical.tensors().end());
    input.aliases.assign(physical.aliases().begin(), physical.aliases().end());
    input.diagnostics.assign(physical.diagnostics().begin(), physical.diagnostics().end());
    return ArtifactIndex::build(std::move(input));
}

ClosedCompilerIdentity closed_v1_compiler_identity() {
    ClosedCompilerIdentity identity;
    identity.major = 1;
    identity.minor = 0;
    identity.revision = 2;
    identity.digest.bytes = {
        0x78, 0x7b, 0x09, 0x0f, 0x9e, 0xaa, 0x15, 0xe0,
        0x8a, 0xf9, 0xfa, 0x32, 0xce, 0x57, 0xa9, 0x32,
        0xda, 0xb6, 0x44, 0x1c, 0x26, 0x16, 0x97, 0x5e,
        0xc9, 0x86, 0xbd, 0xfe, 0xa2, 0x9d, 0xce, 0xb2,
    };
    return identity;
}

} // namespace

ProductPackageLoadResult ProductPackage::finish_closed_v1(
    GgufProductCompilation compilation) {
    if (compilation.manifest.has_carrier()) {
        return package_report(CompatibilityError::AUTHORITY_INVALID,
                              "closed source compilation unexpectedly carries a manifest sidecar");
    }
    const auto& proof = compilation.manifest.graph_proof();
    if (!proof || !source_compiler_graph_proof_matches(
                      compilation.manifest.semantic_model(), *proof)) {
        return package_report(CompatibilityError::IMPORT_CLOSURE_INCOMPLETE,
                              "closed source compilation graph proof is missing or stale");
    }
    const TokenContract& contract = compilation.manifest.token_contract();
    if (contract.tokenizer_algorithm != TokenizerAlgorithm::ByteBpe ||
        contract.prompt_mode() != TokenPromptMode::SerializedTemplate) {
        return tokenizer_error(
            "closed source compilation requires a ByteBpe tokenizer and input-text prompt contract");
    }
    if (auto token_authority =
            validate_token_program_authority(compilation.token_program, contract)) {
        return std::move(*token_authority);
    }
    auto runtime = std::shared_ptr<const RuntimePackage>(new RuntimePackage(
        std::move(compilation.manifest), Sha256Digest{}, 0,
        RuleQualificationState::Draft, PackageAuthorityKind::ClosedV1Compiled,
        closed_v1_compiler_identity()));
    return ProductPackage(std::move(runtime), std::move(compilation.token_program));
}

ProductPackageLoadResult ProductPackage::finish_product_package(ArtifactIndex physical,
                                                                PackageView carrier) {
    auto loaded = load_carried_manifest(physical, carrier);
    if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) return *report;
    auto validated = std::get<ValidatedPackage>(std::move(loaded));
    const auto runtime = validated.runtime_package();
    if (!runtime || runtime->manifest().token_contract().tokenizer_algorithm !=
                            TokenizerAlgorithm::ByteBpe ||
        runtime->manifest().token_contract().prompt_mode() !=
            TokenPromptMode::SerializedTemplate) {
        return tokenizer_error(
            "product package requires a carried ByteBpe tokenizer and input-text prompt contract");
    }
    const TokenContract& contract = runtime->manifest().token_contract();
    const std::span<const uint8_t> artifact = runtime->artifact_bytes(
        contract.tokenizer_data.artifact_id);
    if (contract.tokenizer_data.offset > artifact.size() ||
        contract.tokenizer_data.length > artifact.size() - contract.tokenizer_data.offset) {
        return tokenizer_error("tokenizer program reference is outside its immutable artifact");
    }
    const auto payload = artifact.subspan(
        static_cast<size_t>(contract.tokenizer_data.offset),
        static_cast<size_t>(contract.tokenizer_data.length));
    auto compiled = TokenProgram::compile(payload);
    if (const auto* status = std::get_if<TokenProgramStatus>(&compiled)) {
        return tokenizer_error("tokenizer program rejected: " + status->detail);
    }
    TokenProgram program = std::get<TokenProgram>(std::move(compiled));
    if (auto token_authority = validate_token_program_authority(program, contract)) {
        return std::move(*token_authority);
    }
    return package_report(
        CompatibilityError::AUTHORITY_INVALID,
        "carried manifests require a verified trust certificate; closed source compilation is the only current product authority");
}

ProductPackageLoadResult load_product_package(std::string_view package_path) {
    constexpr size_t kMaximumPathBytes = 16 * 1024;
    if (package_path.empty() || package_path.size() > kMaximumPathBytes - 32) {
        return package_report(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                              "product package path is empty or exceeds the bounded loader contract");
    }

    try {
        const std::string primary_path(package_path);
        struct stat status {};
        if (lstat(primary_path.c_str(), &status) == 0 && S_ISDIR(status.st_mode)) {
            auto mlx = load_mlx_product_physical_package(package_path);
            if (const auto* report = std::get_if<CompatibilityReport>(&mlx)) return *report;
            MlxProductPhysicalPackage package =
                std::get<MlxProductPhysicalPackage>(std::move(mlx));
            return ProductPackage::finish_product_package(std::move(package.physical_index),
                                                         std::move(package.manifest));
        }
        const std::string carrier_path = primary_path + ".lapman";
        const std::string token_path = primary_path + ".laptok";
        struct stat carrier_status {};
        if (lstat(carrier_path.c_str(), &carrier_status) != 0) {
            const int carrier_errno = errno;
            if (carrier_errno != ENOENT) {
                CompatibilityReport report = package_report(
                    CompatibilityError::PACKAGE_BOUNDS_INVALID,
                    "manifest sidecar presence could not be determined");
                report.artifact_id = ArtifactId{1};
                report.artifact_index = 1;
                return report;
            }
            auto raw_graph = ArtifactSet::load_single_file(primary_path);
            if (const auto* raw_report =
                    std::get_if<CompatibilityReport>(&raw_graph)) {
                return *raw_report;
            }
            ArtifactSet raw_artifacts =
                std::get<ArtifactSet>(std::move(raw_graph));
            auto raw_view = raw_artifacts.view(ArtifactId{0});
            if (const auto* raw_report =
                    std::get_if<CompatibilityReport>(&raw_view)) {
                return *raw_report;
            }
            auto compiled = compile_gguf_product_source(
                std::get<PackageView>(std::move(raw_view)));
            if (const auto* compile_report =
                    std::get_if<CompatibilityReport>(&compiled)) {
                return *compile_report;
            }
            return ProductPackage::finish_closed_v1(
                std::get<GgufProductCompilation>(std::move(compiled)));
        }
        const std::array<ArtifactSource, 3> sources = {
            ArtifactSource{primary_path, ArtifactRole::Primary, ArtifactId{0}},
            ArtifactSource{carrier_path, ArtifactRole::Sidecar, ArtifactId{1}},
            ArtifactSource{token_path, ArtifactRole::Shard, ArtifactId{2}},
        };
        auto graph = ArtifactSet::load_graph(sources);
        if (const auto* report = std::get_if<CompatibilityReport>(&graph)) {
            if (report->artifact_id == ArtifactId{2}) {
                return tokenizer_error("adjacent .laptok tokenizer program is missing or invalid");
            }
            return *report;
        }
        ArtifactSet artifacts = std::get<ArtifactSet>(std::move(graph));
        auto primary = artifacts.view(ArtifactId{0});
        if (const auto* report = std::get_if<CompatibilityReport>(&primary)) return *report;
        auto carrier = artifacts.view(ArtifactId{1});
        if (const auto* report = std::get_if<CompatibilityReport>(&carrier)) return *report;
        auto token_program = artifacts.view(ArtifactId{2});
        if (const auto* report = std::get_if<CompatibilityReport>(&token_program)) return *report;
        auto physical = build_gguf_artifact_index(std::get<PackageView>(primary));
        if (const auto* report = std::get_if<CompatibilityReport>(&physical)) return *report;
        auto augmented = add_token_program_artifact(
            std::get<ArtifactIndex>(std::move(physical)),
            std::get<PackageView>(std::move(token_program)));
        if (const auto* report = std::get_if<CompatibilityReport>(&augmented)) return *report;
        return ProductPackage::finish_product_package(std::get<ArtifactIndex>(std::move(augmented)),
                                                      std::get<PackageView>(std::move(carrier)));
    } catch (const std::bad_alloc&) {
        return package_report(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                              "product package loader allocation failed");
    }
}

} // namespace Laplace
