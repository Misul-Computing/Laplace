#include "source_program_compiler.h"

#include "codec_certificate_physical_program.h"
#include "gguf_index.h"
#include "gguf_product_compiler.h"
#include "mlx_package.h"
#include "semantic_program_compiler.h"

#include <CommonCrypto/CommonDigest.h>

#include <array>
#include <algorithm>
#include <cerrno>
#include <map>
#include <new>
#include <set>
#include <string>
#include <sys/stat.h>
#include <utility>

namespace Laplace {
namespace {
CompatibilityReport error(CompatibilityError code, const char* message) {
    return compatibility_report(code, message);
}

bool token_authority_matches(const TokenProgram& program, const TokenContract& contract) {
    const auto& definition = program.definition();
    const auto algorithm = definition.model_kind == TokenProgramModelKind::SentencePiece
        ? TokenizerAlgorithm::SentencePiece : TokenizerAlgorithm::ByteBpe;
    if (contract.tokenizer_algorithm != algorithm ||
        contract.tokenizer_version != program.wire_major_version() ||
        contract.vocabulary_size != definition.vocabulary.size() ||
        contract.vocabulary_digest != program.vocabulary_digest() ||
        contract.authoritative_template_digest != program.prompt_digest()) return false;
    const auto& post = definition.postprocessor;
    if (post.kind == PostprocessorKind::AddBosEos &&
        (((post.flags & static_cast<uint8_t>(PostprocessorFlags::AddBos)) && post.bos_token_id != contract.bos_id) ||
         ((post.flags & static_cast<uint8_t>(PostprocessorFlags::AddEos)) && post.eos_token_id != contract.eos_id))) return false;
    if (program.wire_major_version() == kTokenProgramV3MajorVersion &&
        definition.stop_ids != contract.stop_ids) return false;
    const auto* prompt = std::get_if<PromptTemplate>(&contract.prompt);
    if (!prompt || prompt->version != 1) return false;
    size_t operation = 0;
    for (size_t i = 0; i < definition.prompt.size(); ++i) {
        const auto& instruction = definition.prompt[i];
        if (instruction.opcode == PromptOpcode::End)
            return i + 1 == definition.prompt.size() && operation == prompt->operations.size();
        if (operation == prompt->operations.size()) return false;
        const auto& expected = prompt->operations[operation++];
        if (instruction.opcode == PromptOpcode::EmitUserText) {
            if (expected.kind != PromptOperationKind::AppendInputText) return false;
        } else if (instruction.opcode == PromptOpcode::EmitLiteralUtf8 ||
                   instruction.opcode == PromptOpcode::EmitGenerationPrompt) {
            if (expected.kind != PromptOperationKind::AppendLiteral ||
                expected.literal.size() != instruction.literal.size() ||
                !std::equal(expected.literal.begin(), expected.literal.end(),
                            reinterpret_cast<const uint8_t*>(instruction.literal.data()))) return false;
        } else return false;
    }
    return false;
}

PlaneKind plane_kind(CodecCertificatePlaneRole role) {
    switch (role) {
    case CodecCertificatePlaneRole::Values: return PlaneKind::Values;
    case CodecCertificatePlaneRole::Scales: return PlaneKind::Scales;
    case CodecCertificatePlaneRole::Biases: return PlaneKind::Biases;
    case CodecCertificatePlaneRole::Indexes: return PlaneKind::Indexes;
    }
    return static_cast<PlaneKind>(0);
}

ProgramPackageResult compile(const SemanticManifest& manifest, const TokenProgram& tokens,
                             uint32_t max_context) {
    const auto& model = manifest.semantic_model();
    if (!manifest.has_physical_codec_authority())
        return error(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                     "product manifest physical codec registry is incomplete");
    if (!max_context || (model.maximum_context && max_context > model.maximum_context))
        return error(CompatibilityError::IR_SHAPE_MISMATCH, "source context capacity is outside its declared range");
    if (!token_authority_matches(tokens, manifest.token_contract()))
        return error(CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED, "token program differs from the source token contract");
    if (model.input_values_count != 1 || model.input_values_first >= model.values.size() ||
        model.values[model.input_values_first].logical_type != ScalarType::U32)
        return error(CompatibilityError::IR_SHAPE_MISMATCH, "source token entry must be one U32 tensor");
    std::vector<SemanticDimensionBinding> dimensions;
    std::set<uint64_t> token_symbols;
    for (const auto& dimension : model.values[model.input_values_first].dimensions) {
        if (dimension.kind == DimensionKind::Symbol) token_symbols.insert(dimension.constant_or_symbol);
        else if (dimension.kind != DimensionKind::Constant || dimension.constant_or_symbol != 1)
            return error(CompatibilityError::IR_SHAPE_MISMATCH, "source token entry must contain one token per invocation");
    }
    for (auto symbol : token_symbols) dimensions.push_back({symbol, 1});
    auto result = compile_semantic_program(model, dimensions, max_context);
    if (auto* report = std::get_if<CompatibilityReport>(&result)) return *report;
    auto compiled = std::get<CompiledSemanticProgram>(std::move(result));
    const auto& program = program_definition(compiled.program);
    if (program.functions.size() != 1 || program.exports.size() != 1 || compiled.inputs.size() != 1)
        return error(CompatibilityError::IR_REFERENCE_INVALID, "source compiler did not produce one token endpoint pair");
    const auto function = program.functions.front().id;
    const auto& registry = manifest.physical_codec_registry();
    const auto& physical = manifest.physical_index();
    std::vector<PhysicalProgramRecord> records;
    std::vector<PhysicalResourceBinding> resources;
    for (const auto& entry : compiled.tensors) {
        const auto tensor = std::find_if(model.tensors.begin(), model.tensors.end(),
            [&](const auto& t) { return t.id == entry.semantic_id; });
        const auto declaration = std::find_if(registry.tensors.begin(), registry.tensors.end(),
            [&](const auto& t) { return t.tensor_id == entry.semantic_id; });
        if (tensor == model.tensors.end() || declaration == registry.tensors.end())
            return error(CompatibilityError::IMPORT_TENSOR_UNMAPPED, "compiled source tensor lacks a certificate declaration");
        const auto spec = std::find_if(registry.codecs.begin(), registry.codecs.end(),
            [&](const auto& c) { return c.identity == declaration->identity; });
        if (spec == registry.codecs.end())
            return error(CompatibilityError::AUTHORITY_INVALID, "source certificate identity is absent from registry");
        auto parsed = parse_codec_certificate(spec->certificate_bytes);
        if (!std::holds_alternative<CodecCertificate>(parsed))
            return error(CompatibilityError::AUTHORITY_INVALID, "source certificate grammar is invalid");
        const auto& certificate = std::get<CodecCertificate>(parsed);
        if (!certificate.matches_physical_identity(spec->identity))
            return error(CompatibilityError::AUTHORITY_INVALID, "source certificate does not match its complete physical identity");
        LogicalTensorType logical{ElementType::F32, {}};
        for (const auto& dimension : tensor->dimensions) {
            if (dimension.kind != DimensionKind::Constant || !dimension.constant_or_symbol)
                return error(CompatibilityError::IR_SHAPE_MISMATCH, "source weight dimensions must be constant");
            logical.extents.push_back(dimension.constant_or_symbol);
        }
        if (tensor->layout.rank != logical.extents.size() || logical.extents.size() > 8)
            return error(CompatibilityError::IR_LAYOUT_MISMATCH, "source tensor layout rank is inconsistent");
        std::vector<uint64_t> logical_strides(logical.extents.size());
        std::vector<bool> seen(logical.extents.size(), false);
        for (size_t position = 0; position < logical.extents.size(); ++position) {
            const auto axis = tensor->layout.axis_order[position];
            if (axis >= logical.extents.size() || seen[axis])
                return error(CompatibilityError::IR_LAYOUT_MISMATCH, "source tensor axis order is not a permutation");
            seen[axis] = true;
            logical_strides[axis] = tensor->layout.strides[position];
        }
        CodecCertificateBinding binding;
        std::vector<const TensorPlane*> source_planes;
        std::vector<uint64_t> strides;
        for (const auto& declared : certificate.plane_summaries()) {
            const auto plane = std::find_if(tensor->planes.begin(), tensor->planes.end(),
                [&](const auto& p) { return p.kind == plane_kind(declared.role); });
            if (plane == tensor->planes.end())
                return error(CompatibilityError::IMPORT_TENSOR_UNMAPPED, "source tensor lacks a declared certificate plane");
            const auto artifact = std::find_if(physical.artifacts().begin(), physical.artifacts().end(),
                [&](const auto& a) { return a.artifact_id() == plane->artifact_id; });
            if (artifact == physical.artifacts().end())
                return error(CompatibilityError::PACKAGE_BOUNDS_INVALID, "source tensor plane artifact is absent");
            source_planes.push_back(&*plane);
            strides.push_back(declared.stride);
            binding.planes.push_back({artifact->bytes(), plane->offset, plane->length, declared.stride});
        }
        auto translated = translate_codec_certificate(certificate, logical, logical_strides, strides);
        if (auto* report = std::get_if<CompatibilityReport>(&translated)) return *report;
        auto decoder = std::get<TranslatedCodecCertificate>(std::move(translated));
        binding.unit_count = decoder.required_units;
        if (certificate.validate_tensor(binding) != CodecCertificateError::None)
            return error(CompatibilityError::PACKAGE_BOUNDS_INVALID, "source certificate rejects aggregate tensor spans or aligned unit composition");
        auto digest = physical_program_digest(decoder.program);
        auto wire = encode_physical_program(decoder.program);
        if (auto* report = std::get_if<CompatibilityReport>(&digest)) return *report;
        if (auto* report = std::get_if<CompatibilityReport>(&wire)) return *report;
        const auto identity = std::get<PhysicalProgramDigest>(digest);
        const auto existing = std::find_if(records.begin(), records.end(),
            [&](const auto& record) { return record.digest == identity; });
        if (existing == records.end())
            records.push_back({identity, std::get<std::vector<uint8_t>>(std::move(wire)), logical});
        else if (existing->logical_type != logical)
            return error(CompatibilityError::IR_SHAPE_MISMATCH, "physical program identity has incompatible logical shapes");
        PhysicalResourceBinding resource;
        resource.resource_id = entry.semantic_id;
        resource.program_digest = identity;
        resource.semantic_function_id = function;
        resource.semantic_value_id = entry.entry_value_id;
        for (uint32_t plane = 0; plane < decoder.source_planes.size(); ++plane) {
            const auto source = decoder.source_planes[plane];
            if (source == kNoPhysicalPlane) continue;
            const auto& span = *source_planes[source];
            resource.planes.push_back({plane, span.artifact_id, span.offset, span.length});
        }
        resources.push_back(std::move(resource));
    }
    std::sort(records.begin(), records.end(),
              [](const PhysicalProgramRecord& left, const PhysicalProgramRecord& right) {
                  return left.digest < right.digest;
              });
    std::sort(resources.begin(), resources.end(),
              [](const PhysicalResourceBinding& left, const PhysicalResourceBinding& right) {
                  return left.resource_id < right.resource_id;
              });
    StateSchema schema;
    std::map<uint64_t, uint32_t> aliases;
    for (const auto& reference : program.state_references) {
        const uint64_t key = reference.alias_group == UINT32_MAX
            ? (uint64_t{1} << 32) | reference.id : reference.alias_group;
        const auto [cell, inserted] = aliases.emplace(key, static_cast<uint32_t>(aliases.size()));
        StateSlotSchema slot{reference.id, reference.type.element_type, {}, cell->second};
        for (const auto& dimension : reference.type.dimensions) {
            if (dimension.expression != DimensionExpression::Constant)
                return error(CompatibilityError::IR_STATE_INVALID, "compiled source state still has dynamic extents");
            slot.extents.push_back(dimension.value);
        }
        schema.slots.push_back(std::move(slot));
    }
    auto state = verify_state_schema(std::move(schema), compiled.program);
    if (auto* report = std::get_if<CompatibilityReport>(&state)) return *report;
    ArtifactIndexInput retained;
    retained.artifacts.assign(physical.artifacts().begin(), physical.artifacts().end());
    retained.metadata_facts.assign(physical.metadata_facts().begin(), physical.metadata_facts().end());
    retained.package_facts.assign(physical.package_facts().begin(), physical.package_facts().end());
    retained.tensors.assign(physical.tensors().begin(), physical.tensors().end());
    retained.aliases.assign(physical.aliases().begin(), physical.aliases().end());
    retained.diagnostics.assign(physical.diagnostics().begin(), physical.diagnostics().end());
    uint32_t next_id = 0;
    const auto append_blob = [&](std::span<const uint8_t> bytes) -> std::variant<PackageView, CompatibilityReport> {
        while (next_id != UINT32_MAX && std::any_of(retained.artifacts.begin(), retained.artifacts.end(),
               [&](const auto& a) { return a.artifact_id().value == next_id; })) ++next_id;
        if (next_id == UINT32_MAX)
            return error(CompatibilityError::IMPORT_SCHEMA_LIMIT, "source artifact ID space is exhausted");
        auto blob = ArtifactSet::make_owned_blob(ArtifactId{next_id++}, ArtifactRole::Sidecar, bytes);
        if (auto* view = std::get_if<PackageView>(&blob)) retained.artifacts.push_back(*view);
        return blob;
    };
    auto provenance = append_blob(manifest.bytes());
    if (auto* report = std::get_if<CompatibilityReport>(&provenance)) return *report;
    auto serialized = tokens.serialize();
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized))
        return error(CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED, "source token program cannot be serialized");
    auto token_blob = append_blob(std::get<std::vector<uint8_t>>(serialized));
    if (auto* report = std::get_if<CompatibilityReport>(&token_blob)) return *report;
    const auto& token_view = std::get<PackageView>(token_blob);
    TokenProgramSource token_source{token_view.artifact_id(), 0, token_view.bytes().size(), token_view.digest()};
    auto index = ArtifactIndex::build(std::move(retained));
    if (auto* report = std::get_if<CompatibilityReport>(&index)) return *report;
    const std::array<TokenEndpointBinding, 2> endpoints = {{
        {TokenEndpointKind::InputToken, function, compiled.inputs.front().entry_value_id},
        {TokenEndpointKind::OutputScores, function, program.exports.front().result_index}}};
    return build_program_package(std::get<ArtifactIndex>(std::move(index)), std::move(compiled.program),
        std::get<VerifiedStateSchema>(std::move(state)), token_source, endpoints, records, resources);
}

const PackageView* artifact_by_id(const ArtifactIndex& index, ArtifactId id) {
    const auto artifacts = index.artifacts();
    const auto found = std::find_if(artifacts.begin(), artifacts.end(),
        [&](const PackageView& artifact) { return artifact.artifact_id() == id; });
    return found == artifacts.end() ? nullptr : &*found;
}

Sha256Digest digest_bytes(std::span<const uint8_t> bytes) {
    Sha256Digest digest;
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    for (size_t offset = 0; offset < bytes.size();) {
        const size_t chunk = std::min<size_t>(1024 * 1024, bytes.size() - offset);
        CC_SHA256_Update(&context, bytes.data() + offset, static_cast<CC_LONG>(chunk));
        offset += chunk;
    }
    CC_SHA256_Final(digest.bytes.data(), &context);
    return digest;
}

std::variant<ArtifactIndex, CompatibilityReport>
add_token_program_artifact(ArtifactIndex physical, PackageView token) {
    ArtifactIndexInput input;
    input.artifacts.assign(physical.artifacts().begin(), physical.artifacts().end());
    input.artifacts.push_back(std::move(token));
    input.metadata_facts.assign(physical.metadata_facts().begin(), physical.metadata_facts().end());
    input.package_facts.assign(physical.package_facts().begin(), physical.package_facts().end());
    input.tensors.assign(physical.tensors().begin(), physical.tensors().end());
    input.aliases.assign(physical.aliases().begin(), physical.aliases().end());
    input.diagnostics.assign(physical.diagnostics().begin(), physical.diagnostics().end());
    return ArtifactIndex::build(std::move(input));
}

std::variant<TokenProgram, CompatibilityReport>
compile_token_program(const ArtifactIndex& physical, const SemanticManifest& manifest) {
    const TokenArtifactReference& reference = manifest.token_contract().tokenizer_data;
    const PackageView* artifact = artifact_by_id(physical, reference.artifact_id);
    if (!artifact || reference.length == 0 || reference.offset > artifact->bytes().size() ||
        reference.length > artifact->bytes().size() - reference.offset) {
        return error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                     "source tokenizer program reference is outside its immutable artifact");
    }
    const auto payload = artifact->bytes().subspan(
        static_cast<size_t>(reference.offset), static_cast<size_t>(reference.length));
    if (digest_bytes(payload) != reference.digest)
        return error(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH,
                     "source tokenizer program digest does not match its immutable artifact");
    auto compiled = TokenProgram::compile(payload);
    if (const auto* status = std::get_if<TokenProgramStatus>(&compiled))
        return error(CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED,
                     status->detail.c_str());
    return std::get<TokenProgram>(std::move(compiled));
}

LoadedSourceProgramResult finish_loaded_source(SemanticManifest manifest,
                                                TokenProgram tokens,
                                                uint32_t requested_context) {
    const auto& model = manifest.semantic_model();
    const uint32_t capacity = requested_context != 0
        ? requested_context
        : (model.maximum_context != 0 ? std::min<uint32_t>(2048, model.maximum_context) : 2048);
    auto compiled = compile_source_program_package(manifest, tokens, capacity);
    if (const auto* report = std::get_if<CompatibilityReport>(&compiled)) return *report;
    return LoadedSourceProgram{
        std::get<VerifiedProgramPackage>(std::move(compiled)), capacity,
        model.eos_id, model.stop_ids};
}

LoadedSourceProgramResult load_gguf_with_sidecars(std::string_view source_path,
                                                 uint32_t requested_context) {
    const std::string carrier_path = std::string(source_path) + ".lapman";
    const std::string token_path = std::string(source_path) + ".laptok";
    const std::array<ArtifactSource, 3> sources = {{
        {source_path, ArtifactRole::Primary, ArtifactId{0}},
        {carrier_path, ArtifactRole::Sidecar, ArtifactId{1}},
        {token_path, ArtifactRole::Shard, ArtifactId{2}},
    }};
    auto graph = ArtifactSet::load_graph(sources);
    if (const auto* report = std::get_if<CompatibilityReport>(&graph)) return *report;
    ArtifactSet artifacts = std::get<ArtifactSet>(std::move(graph));
    auto primary = artifacts.view(ArtifactId{0});
    auto carrier = artifacts.view(ArtifactId{1});
    auto token = artifacts.view(ArtifactId{2});
    if (const auto* report = std::get_if<CompatibilityReport>(&primary)) return *report;
    if (const auto* report = std::get_if<CompatibilityReport>(&carrier)) return *report;
    if (const auto* report = std::get_if<CompatibilityReport>(&token)) return *report;
    auto physical = build_gguf_artifact_index(std::get<PackageView>(primary));
    if (const auto* report = std::get_if<CompatibilityReport>(&physical)) return *report;
    auto augmented = add_token_program_artifact(
        std::get<ArtifactIndex>(std::move(physical)), std::get<PackageView>(token));
    if (const auto* report = std::get_if<CompatibilityReport>(&augmented)) return *report;
    ArtifactIndex index = std::get<ArtifactIndex>(std::move(augmented));
    auto manifest = SemanticManifest::decode_carried(index, std::get<PackageView>(carrier));
    if (const auto* report = std::get_if<CompatibilityReport>(&manifest)) return *report;
    SemanticManifest value = std::get<SemanticManifest>(std::move(manifest));
    auto program = compile_token_program(index, value);
    if (const auto* report = std::get_if<CompatibilityReport>(&program)) return *report;
    return finish_loaded_source(std::move(value),
        std::get<TokenProgram>(std::move(program)), requested_context);
}

LoadedSourceProgramResult load_raw_gguf(std::string_view source_path,
                                        uint32_t requested_context) {
    auto artifacts = ArtifactSet::load_single_file(source_path);
    if (const auto* report = std::get_if<CompatibilityReport>(&artifacts)) return *report;
    auto view = std::get<ArtifactSet>(std::move(artifacts)).view(ArtifactId{0});
    if (const auto* report = std::get_if<CompatibilityReport>(&view)) return *report;
    auto source = compile_gguf_product_source(std::get<PackageView>(view));
    if (const auto* report = std::get_if<CompatibilityReport>(&source)) return *report;
    auto compiled = std::get<GgufProductCompilation>(std::move(source));
    return finish_loaded_source(std::move(compiled.manifest),
        std::move(compiled.token_program), requested_context);
}

LoadedSourceProgramResult load_safetensors_source(std::string_view source_path,
                                                  uint32_t requested_context) {
    auto product = load_safetensors_product_physical_package(source_path);
    if (const auto* report = std::get_if<CompatibilityReport>(&product)) return *report;
    auto loaded = std::get<MlxProductPhysicalPackage>(std::move(product));
    auto manifest = SemanticManifest::decode_carried(loaded.physical_index, loaded.manifest);
    if (const auto* report = std::get_if<CompatibilityReport>(&manifest)) return *report;
    SemanticManifest value = std::get<SemanticManifest>(std::move(manifest));
    auto tokens = compile_token_program(loaded.physical_index, value);
    if (const auto* report = std::get_if<CompatibilityReport>(&tokens)) return *report;
    return finish_loaded_source(std::move(value),
        std::get<TokenProgram>(std::move(tokens)), requested_context);
}
} // namespace

ProgramPackageResult compile_source_program_package(const SemanticManifest& manifest,
    const TokenProgram& tokens, uint32_t max_context) {
    try { return compile(manifest, tokens, max_context); }
    catch (const std::bad_alloc&) {
        return error(CompatibilityError::IMPORT_SCHEMA_LIMIT, "source program compilation exceeded available memory");
    }
}

LoadedSourceProgramResult load_source_program_package(std::string_view path,
                                                      uint32_t max_context) {
    try {
        struct stat status {};
        if (lstat(std::string(path).c_str(), &status) == 0 && S_ISDIR(status.st_mode))
            return load_safetensors_source(path, max_context);
        const std::string carrier_path = std::string(path) + ".lapman";
        struct stat carrier_status {};
        if (lstat(carrier_path.c_str(), &carrier_status) == 0)
            return load_gguf_with_sidecars(path, max_context);
        if (errno != ENOENT)
            return error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                         "source manifest sidecar presence could not be determined");
        return load_raw_gguf(path, max_context);
    } catch (const std::bad_alloc&) {
        return error(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                     "source package loading exceeded available memory");
    }
}
} // namespace Laplace
