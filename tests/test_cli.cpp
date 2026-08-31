#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <sys/wait.h>
#include <utility>
#include <vector>

#include "artifact_index.h"
#include "artifact_set.h"
#include "compat_rule.h"
#include "gguf_index.h"
#include "gguf_writer.h"
#include "semantic_manifest.h"
#include "semantic_model.h"
#include "test_util.h"
#include "token_program.h"

using namespace Laplace;

namespace {

gguf_writer::TensorDecl f32_tensor(const char* name,
                                   std::vector<uint64_t> dims,
                                   float value = 0.0f) {
    size_t count = 1;
    for (uint64_t dim : dims) count *= static_cast<size_t>(dim);
    gguf_writer::TensorDecl tensor;
    tensor.name = name;
    tensor.dims = std::move(dims);
    tensor.type = 0;
    tensor.data.resize(count * sizeof(float));
    for (size_t i = 0; i < count; ++i)
        std::memcpy(tensor.data.data() + i * sizeof(float), &value,
                    sizeof(value));
    return tensor;
}

gguf_writer::TensorDecl blocked_tensor(const char* name, std::vector<uint64_t> dims,
                                       uint32_t type, size_t block_bytes) {
    size_t rows = dims.size() == 2 ? static_cast<size_t>(dims[1]) : 1;
    gguf_writer::TensorDecl tensor;
    tensor.name = name;
    tensor.dims = std::move(dims);
    tensor.type = type;
    tensor.data.resize(rows * block_bytes);
    return tensor;
}

void write_generic_canonical_model(const char* path) {
    constexpr uint32_t kQ4K = 12;
    constexpr uint32_t kQ6K = 14;
    constexpr uint32_t kHidden = 256;
    constexpr uint32_t kVocab = 3;
    gguf_writer::Writer writer;
    writer.kv_str("general.architecture", "fixture");
    writer.kv_u32("general.quantization_version", 2);
    writer.kv_u32("fixture.block_count", 1);
    writer.kv_u32("fixture.context_length", 32768);
    writer.kv_u32("fixture.embedding_length", kHidden);
    writer.kv_u32("fixture.feed_forward_length", kHidden);
    writer.kv_u32("fixture.attention.head_count", 1);
    writer.kv_u32("fixture.attention.head_count_kv", 1);
    writer.kv_u32("fixture.attention.key_length", kHidden);
    writer.kv_u32("fixture.attention.value_length", kHidden);
    writer.kv_u32("fixture.rope.dimension_count", kHidden);
    writer.kv_f32("fixture.rope.freq_base", 10000.0f);
    writer.kv_f32("fixture.attention.layer_norm_rms_epsilon", 1.0e-5f);
    writer.kv_u32("tokenizer.ggml.bos_token_id", 2);
    writer.kv_u32("tokenizer.ggml.eos_token_id", 1);
    writer.kv_arr_str("tokenizer.ggml.tokens", {"a", "<eos>", "<bos>"});

    writer.add_tensor(blocked_tensor("token_embd.weight", {kHidden, kVocab}, kQ4K, 144));
    writer.add_tensor(f32_tensor("output_norm.weight", {kHidden}, 1.0f));
    writer.add_tensor(blocked_tensor("output.weight", {kHidden, kVocab}, kQ6K, 210));
    writer.add_tensor(f32_tensor("blk.0.attn_norm.weight", {kHidden}, 1.0f));
    writer.add_tensor(blocked_tensor("blk.0.attn_q.weight", {kHidden, kHidden}, kQ4K, 144));
    writer.add_tensor(blocked_tensor("blk.0.attn_k.weight", {kHidden, kHidden}, kQ4K, 144));
    writer.add_tensor(blocked_tensor("blk.0.attn_v.weight", {kHidden, kHidden}, kQ4K, 144));
    writer.add_tensor(blocked_tensor("blk.0.attn_output.weight", {kHidden, kHidden}, kQ4K, 144));
    writer.add_tensor(f32_tensor("blk.0.post_attention_norm.weight", {kHidden}, 1.0f));
    writer.add_tensor(blocked_tensor("blk.0.ffn_gate.weight", {kHidden, kHidden}, kQ4K, 144));
    writer.add_tensor(blocked_tensor("blk.0.ffn_up.weight", {kHidden, kHidden}, kQ4K, 144));
    writer.add_tensor(blocked_tensor("blk.0.ffn_down.weight", {kHidden, kHidden}, kQ4K, 144));
    CHECK(writer.write_file(path));
}

bool write_bytes(const std::string& path, std::span<const uint8_t> bytes) {
    FILE* file = std::fopen(path.c_str(), "wb");
    CHECK(file != nullptr);
    if (!file) return false;
    const size_t written = bytes.empty() ? 0 : std::fwrite(bytes.data(), 1, bytes.size(), file);
    const bool closed = std::fclose(file) == 0;
    CHECK(written == bytes.size());
    CHECK(closed);
    return written == bytes.size() && closed;
}

std::vector<uint8_t> byte_level_program(uint32_t bos_id, uint32_t eos_id) {
    TokenProgramDefinition definition;
    for (unsigned value = 0; value != definition.byte_map.size(); ++value) {
        definition.byte_map[value] = static_cast<uint8_t>(value);
    }
    definition.vocabulary = {
        {"a", 0, 0},
        {"<eos>", static_cast<uint16_t>(VocabFlags::Special), 0},
        {"<bos>", static_cast<uint16_t>(VocabFlags::Special), 0},
    };
    definition.unknown_token_id = 0;
    definition.pretokenizer.kind = PretokenizerKind::ByteLevel;
    definition.postprocessor.kind = PostprocessorKind::AddBosEos;
    definition.postprocessor.flags = static_cast<uint8_t>(PostprocessorFlags::AddBos) |
                                     static_cast<uint8_t>(PostprocessorFlags::AddEos);
    definition.postprocessor.bos_token_id = bos_id;
    definition.postprocessor.eos_token_id = eos_id;
    definition.decoder.kind = DecoderKind::ByteLevel;
    definition.decoder.flags = static_cast<uint8_t>(DecoderFlags::SkipSpecial);
    definition.prompt = {
        {PromptOpcode::EmitUserText, {}},
        {PromptOpcode::EmitGenerationPrompt, "a"},
        {PromptOpcode::End, {}},
    };
    const auto serialized = serialize_token_program(definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized)) return {};
    return std::get<std::vector<uint8_t>>(serialized);
}

struct DiagnosticFixture {
    ArtifactIndex physical;
    SemanticModel model;
};

std::optional<DiagnosticFixture> load_diagnostic_fixture(const PackageView& primary) {
    // This resolver is a test-only compiler input. Its output is normalized to
    // the closed v1 manifest route below and never enters the product loader.
    auto validated = load_validated_gguf(primary);
    if (const auto* package = std::get_if<ValidatedPackage>(&validated)) {
        const auto runtime = package->runtime_package();
        if (runtime) return DiagnosticFixture{runtime->physical_index(), runtime->semantics()};
    }
    auto physical = build_gguf_artifact_index(primary);
    auto imported = import_gguf(primary);
    if (!std::holds_alternative<ArtifactIndex>(physical) ||
        !std::holds_alternative<SemanticModel>(imported)) {
        return std::nullopt;
    }
    return DiagnosticFixture{std::get<ArtifactIndex>(std::move(physical)),
                             std::get<SemanticModel>(std::move(imported))};
}

std::optional<SemanticManifest> build_carried_manifest(const PackageView& primary,
                                                        const PackageView& token_data) {
    auto diagnostic = load_diagnostic_fixture(primary);
    CHECK(diagnostic.has_value());
    if (!diagnostic) return std::nullopt;

    ArtifactIndexInput input;
    input.artifacts.assign(diagnostic->physical.artifacts().begin(), diagnostic->physical.artifacts().end());
    input.artifacts.push_back(token_data);
    input.metadata_facts.assign(diagnostic->physical.metadata_facts().begin(), diagnostic->physical.metadata_facts().end());
    input.package_facts.assign(diagnostic->physical.package_facts().begin(), diagnostic->physical.package_facts().end());
    input.tensors.assign(diagnostic->physical.tensors().begin(), diagnostic->physical.tensors().end());
    input.aliases.assign(diagnostic->physical.aliases().begin(), diagnostic->physical.aliases().end());
    input.diagnostics.assign(diagnostic->physical.diagnostics().begin(), diagnostic->physical.diagnostics().end());
    auto augmented = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<ArtifactIndex>(augmented));
    if (!std::holds_alternative<ArtifactIndex>(augmented)) return std::nullopt;
    ArtifactIndex augmented_index = std::get<ArtifactIndex>(std::move(augmented));
    auto compiled_token_program = TokenProgram::compile(token_data.bytes());
    CHECK(std::holds_alternative<TokenProgram>(compiled_token_program));
    if (!std::holds_alternative<TokenProgram>(compiled_token_program)) return std::nullopt;
    const TokenProgram& token_program = std::get<TokenProgram>(compiled_token_program);

    SemanticModel model = std::move(diagnostic->model);
    model.schema_major = 1;
    model.schema_minor = 0;
    model.opset_major = 1;
    model.opset_minor = 0;
    model.maximum_context = 32768;
    model.stop_ids = {model.eos_id};
    for (SemanticOperator& operation : model.operators) operation.semantic_version = 1;
    for (SemanticState& state : model.states) state.semantic_version = 1;
    std::vector<uint32_t> semantic_to_physical(model.tensors.size(), UINT32_MAX);
    std::vector<SemanticTensor> rebound_tensors(model.tensors.size());
    for (SemanticTensor& tensor : model.tensors) {
        CHECK(!tensor.planes.empty());
        if (tensor.planes.empty()) return std::nullopt;
        const TensorPlane& source = tensor.planes.front();
        const ArtifactTensorRecord* physical = nullptr;
        for (const ArtifactTensorRecord& candidate : augmented_index.tensors()) {
            if (candidate.planes.size() != 1 || candidate.planes.front().kind != source.kind) continue;
            const ArtifactSourceSpan& span = candidate.planes.front().source;
            if (span.artifact_id == source.artifact_id && span.offset == source.offset &&
                span.length == source.length) {
                physical = &candidate;
                break;
            }
        }
        CHECK(physical != nullptr);
        if (tensor.id >= semantic_to_physical.size() || !physical || physical->id >= rebound_tensors.size() ||
            !rebound_tensors[physical->id].planes.empty() ||
            semantic_to_physical[tensor.id] != UINT32_MAX) return std::nullopt;
        semantic_to_physical[tensor.id] = physical->id;
        tensor.id = physical->id;
        tensor.quantization = physical->quantization;
        rebound_tensors[physical->id] = std::move(tensor);
    }
    for (uint32_t mapped : semantic_to_physical) {
        if (mapped == UINT32_MAX) return std::nullopt;
    }
    for (SemanticOperator& operation : model.operators) {
        for (uint32_t& tensor_id : operation.tensors) {
            if (tensor_id >= semantic_to_physical.size() ||
                semantic_to_physical[tensor_id] == UINT32_MAX) return std::nullopt;
            tensor_id = semantic_to_physical[tensor_id];
        }
    }
    model.tensors = std::move(rebound_tensors);
    model.tokenizer_digest = token_data.digest().bytes;
    model.template_digest = token_program.prompt_digest().bytes;

    TokenContract contract;
    contract.tokenizer_algorithm = TokenizerAlgorithm::ByteBpe;
    contract.tokenizer_version = 1;
    contract.tokenizer_data = {token_data.artifact_id(), 0, token_data.bytes().size(), token_data.digest()};
    contract.vocabulary_size = model.vocabulary_size;
    contract.vocabulary_digest = token_program.vocabulary_digest();
    contract.bos_id = model.bos_id;
    contract.eos_id = model.eos_id;
    contract.stop_ids = model.stop_ids;
    contract.authoritative_tokenizer_digest = {model.tokenizer_digest};
    contract.authoritative_template_digest = {model.template_digest};
    PromptTemplate prompt;
    prompt.version = 1;
    prompt.operations = {
        {PromptOperationKind::AppendInputText, {}, kNoTokenId},
        {PromptOperationKind::AppendLiteral, {'a'}, kNoTokenId},
    };
    contract.prompt = std::move(prompt);
    CHECK(contract.validate().ok());
    auto manifest = SemanticManifest::build(augmented_index, model, contract);
    CHECK(std::holds_alternative<SemanticManifest>(manifest));
    if (!std::holds_alternative<SemanticManifest>(manifest)) return std::nullopt;
    return std::get<SemanticManifest>(std::move(manifest));
}

struct ProductFixture {
    std::string path;
    std::string carrier_path;
    std::string token_path;
};

ProductFixture make_product_fixture(const char* path) {
    ProductFixture fixture{path, std::string(path) + ".lapman", std::string(path) + ".laptok"};
    std::remove(fixture.path.c_str());
    std::remove(fixture.carrier_path.c_str());
    std::remove(fixture.token_path.c_str());
    write_generic_canonical_model(fixture.path.c_str());

    const std::vector<uint8_t> token_bytes = byte_level_program(2, 1);
    CHECK(write_bytes(fixture.token_path, token_bytes));
    const std::array<ArtifactSource, 2> sources = {
        ArtifactSource{fixture.path, ArtifactRole::Primary, ArtifactId{0}},
        ArtifactSource{fixture.token_path, ArtifactRole::Shard, ArtifactId{2}},
    };
    auto artifacts = ArtifactSet::load_graph(sources);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return fixture;
    ArtifactSet& set = std::get<ArtifactSet>(artifacts);
    auto primary = set.view(ArtifactId{0});
    auto token_data = set.view(ArtifactId{2});
    CHECK(std::holds_alternative<PackageView>(primary));
    CHECK(std::holds_alternative<PackageView>(token_data));
    if (!std::holds_alternative<PackageView>(primary) ||
        !std::holds_alternative<PackageView>(token_data)) return fixture;
    auto manifest = build_carried_manifest(std::get<PackageView>(primary),
                                           std::get<PackageView>(token_data));
    CHECK(manifest.has_value());
    if (manifest) {
        const auto bytes = manifest->bytes();
        CHECK(write_bytes(fixture.carrier_path, bytes));
    }
    return fixture;
}

void remove_product_fixture(const ProductFixture& fixture) {
    std::remove(fixture.path.c_str());
    std::remove(fixture.carrier_path.c_str());
    std::remove(fixture.token_path.c_str());
}

std::string run_cli(const std::string& arguments, int* status) {
    std::string command = "\"" LAPLACE_BINARY_PATH "\" " + arguments + " 2>&1";
    FILE* process = popen(command.c_str(), "r");
    CHECK(process != nullptr);
    std::string output;
    char buffer[256];
    while (process && std::fgets(buffer, sizeof(buffer), process)) output += buffer;
    int result = process ? pclose(process) : -1;
    if (result == -1) {
        *status = -1;
    } else if (WIFEXITED(result)) {
        *status = WEXITSTATUS(result);
    } else if (WIFSIGNALED(result)) {
        *status = 128 + WTERMSIG(result);
    } else {
        *status = 128;
    }
    return output;
}

bool native_runtime_required() {
    return std::getenv("LAPLACE_TEST_REQUIRE_NATIVE") != nullptr;
}

void test_help_lists_only_the_v1_product_route() {
    int status = 0;
    const std::string help = run_cli("--help", &status);
    CHECK(status == 0);
    for (const char* removed : {
             "--laplace-kv", "--laplace-stream", "--laplace-resident",
             "--laplace-kv-q4", "--kv-fp16", "--kv-fp32", "--eval-file",
             "--eval-limit", "--eval-preliminary", "--no-spec", "--draft",
             "--draft-mode", "--no-im-start", "--raw-channels", "-j"}) {
        CHECK(help.find(removed) == std::string::npos);
    }
}

void test_removed_flags_fail_in_the_model_position() {
    for (const char* option : {"--laplace-kv", "--eval-file", "--no-spec", "--draft"}) {
        int status = 0;
        const std::string output = run_cli(option, &status);
        CHECK(status != 0);
        CHECK(output.find("unknown option:") != std::string::npos);
        CHECK(output.find(option) != std::string::npos);
    }
}

void test_malformed_numeric_values_fail_closed() {
    const std::vector<std::string> invalid = {
        "-n not-a-number",
        "-n --bench",
        "-n 2147483648",
        "-t nan",
        "-t inf",
        "-t 1e999",
        "--top-k nope",
        "--top-k -1",
        "--top-p 2.5",
        "--seed -1",
        "--seed 4294967296",
        "--max-seq 16junk",
        "--max-seq 0",
        "--max-seq --bench",
    };
    for (const std::string& option : invalid) {
        int status = 0;
        const std::string output = run_cli("missing.gguf " + option, &status);
        CHECK(status != 0);
        CHECK(output.find("invalid value") != std::string::npos);
        CHECK(output.find("failed to open missing.gguf") == std::string::npos);
    }
}

void test_invalid_prompt_file_fails_before_model_open() {
    int status = 0;
    const std::string output = run_cli(
        "missing.gguf --prompt-file /private/tmp", &status);
    CHECK(status != 0);
    CHECK(output.find("prompt file must be a regular file") != std::string::npos);
    CHECK(output.find("failed to open missing.gguf") == std::string::npos);
}

void test_removed_product_flags_fail_as_unknown_options() {
    const std::vector<std::string> removed = {
        "--laplace-kv",
        "--laplace-stream",
        "--laplace-resident",
        "--laplace-kv-q4",
        "--kv-fp16",
        "--kv-fp32",
        "--eval-file eval.txt",
        "--eval-limit 1",
        "--eval-preliminary",
        "--no-spec",
        "--draft 1",
        "--draft-mode prompt",
        "--no-im-start",
        "--raw-channels",
        "-j 2",
    };
    for (const std::string& option : removed) {
        int status = 0;
        const std::string output = run_cli("missing.gguf " + option, &status);
        CHECK(status != 0);
        CHECK(output.find("unknown option:") != std::string::npos);
        CHECK(output.find(option.substr(0, option.find(' '))) != std::string::npos);
    }
}

void test_raw_gguf_with_incomplete_tokenizer_contract_fails_closed() {
    constexpr const char* path = "/private/tmp/laplace-test-cli-raw.gguf";
    const std::string carrier = std::string(path) + ".lapman";
    const std::string token = std::string(path) + ".laptok";
    std::remove(path);
    std::remove(carrier.c_str());
    std::remove(token.c_str());
    write_generic_canonical_model(path);
    int status = 0;
    const std::string output = run_cli(std::string(path) + " -p a -n 1 --greedy", &status);
    CHECK(status != 0);
    CHECK(output.find("typed normalized GGUF facts do not form an executable semantic graph") !=
          std::string::npos);
    std::remove(path);
}

void test_incomplete_codec_manifest_cannot_enter_default_generation() {
    const ProductFixture fixture = make_product_fixture("/private/tmp/laplace-test-cli-model.gguf");
    int status = 0;
    const std::string default_mode = run_cli(
        fixture.path + " -p a -n 1 --greedy", &status);
    CHECK(status != 0);
    CHECK(default_mode.find("code=20") != std::string::npos);
    CHECK(default_mode.find("product manifest physical codec registry is incomplete") !=
          std::string::npos);
    CHECK(default_mode.find("[canonical] Metal session") == std::string::npos);

    const std::string benchmark = run_cli(
        fixture.path + " -p a -n 2 --greedy --bench",
        &status);
    CHECK(status != 0);
    CHECK(benchmark.find("product manifest physical codec registry is incomplete") !=
          std::string::npos);
    CHECK(benchmark.find("[bench]") == std::string::npos);

    const std::string no_decode = run_cli(
        fixture.path + " -p a -n 0 --greedy --bench",
        &status);
    CHECK(status != 0);
    CHECK(no_decode.find("product manifest physical codec registry is incomplete") !=
          std::string::npos);
    remove_product_fixture(fixture);
}

void test_untrusted_package_cannot_supply_a_context_limit() {
    const ProductFixture fixture = make_product_fixture("/private/tmp/laplace-test-cli-context.gguf");
    int status = 0;
    const std::string output = run_cli(
        fixture.path + " -p a -n 1 --greedy --max-seq 32769", &status);
    CHECK(status != 0);
    CHECK(output.find("product manifest physical codec registry is incomplete") !=
          std::string::npos);
    CHECK(output.find("exceeds package maximum") == std::string::npos);
    remove_product_fixture(fixture);
}

void test_self_built_mixed_format_manifest_cannot_enter_metal() {
    const ProductFixture fixture = make_product_fixture("/private/tmp/laplace-test-cli-canonical.gguf");
    int status = 0;
    const std::string output = run_cli(
        fixture.path + " -p a -n 2 --greedy --bench", &status);
    CHECK(status != 0);
    CHECK(output.find("product manifest physical codec registry is incomplete") !=
          std::string::npos);
    CHECK(output.find("[canonical] Metal session") == std::string::npos);
    CHECK(output.find("model:") == std::string::npos);
    CHECK(output.find("kv-cache:") == std::string::npos);
    CHECK(output.find("[spec:") == std::string::npos);
    CHECK(output.find("[bench]") == std::string::npos);
    remove_product_fixture(fixture);
}

void test_native_required_mode_is_not_skippable() {
    if (!native_runtime_required()) return;
    const ProductFixture fixture = make_product_fixture("/private/tmp/laplace-test-cli-native-required.gguf");
    int status = 0;
    const std::string output = run_cli(
        fixture.path + " -p a -n 1 --greedy", &status);
    CHECK(status != 0);
    CHECK(output.find("product manifest physical codec registry is incomplete") !=
          std::string::npos);
    CHECK(output.find("[canonical] Metal session") == std::string::npos);
    remove_product_fixture(fixture);
}

} // namespace

int main() {
    test_help_lists_only_the_v1_product_route();
    test_removed_product_flags_fail_as_unknown_options();
    test_removed_flags_fail_in_the_model_position();
    test_malformed_numeric_values_fail_closed();
    test_invalid_prompt_file_fails_before_model_open();
    test_raw_gguf_with_incomplete_tokenizer_contract_fails_closed();
    test_untrusted_package_cannot_supply_a_context_limit();
    test_incomplete_codec_manifest_cannot_enter_default_generation();
    test_self_built_mixed_format_manifest_cannot_enter_metal();
    test_native_required_mode_is_not_skippable();
    return test_summary("test_cli");
}
