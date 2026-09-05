#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <optional>
#include <cstdint>
#include <utility>
#include <vector>
#include <unistd.h>

#include "artifact_set.h"
#include "gguf.h"
#include "gguf_product_compiler.h"
#include "gguf_writer.h"
#include "test_util.h"

using namespace Laplace;

namespace {

std::string write_source() {
    char path[] = "/private/tmp/laplace-product-compiler-source-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd >= 0) close(fd);

    gguf_writer::Writer writer;
    writer.kv_u32("general.quantization_version", 2);
    gguf_writer::TensorDecl tensor;
    tensor.name = "tensor";
    tensor.dims = {1};
    tensor.type = static_cast<uint32_t>(GGMLType::F32);
    tensor.data.resize(sizeof(float), 0);
    writer.add_tensor(std::move(tensor));
    CHECK(writer.write_file(path));
    return path;
}

void test_untyped_tensor_stays_fail_closed() {
    const std::string path = write_source();
    auto loaded = ArtifactSet::load_single_file(path);
    unlink(path.c_str());
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    if (!std::holds_alternative<ArtifactSet>(loaded)) return;

    ArtifactSet artifacts = std::get<ArtifactSet>(std::move(loaded));
    auto view = artifacts.view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return;

    const auto result = compile_gguf_product_source(
        std::get<PackageView>(std::move(view)));
    CHECK(std::holds_alternative<CompatibilityReport>(result));
    if (const auto* report = std::get_if<CompatibilityReport>(&result)) {
        CHECK(report->code == CompatibilityError::IMPORT_TENSOR_UNMAPPED);
        CHECK(report->stage == CompatibilityStage::Import);
        CHECK(report->detail ==
              "GGUF source tensor has no typed role evidence: tensor");
    }
}

std::string write_dense_source(std::optional<float> attention_scale = std::nullopt, uint32_t query_heads = 1, bool text_tokens = false, bool chat_template = false, bool cli_tokens = false) {
    // Plain dense llama-spelled source: no q/k norms, no routed experts,
    // and no output projection (tied embeddings). This shape must compile
    // through the universal product route.
    char path[] = "/private/tmp/laplace-product-compiler-dense-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd >= 0) close(fd);

    constexpr uint32_t kHidden = 8;
    const uint32_t kVocab = cli_tokens ? 5 : 4;
    constexpr uint32_t kFfn = 16;
    gguf_writer::Writer writer;
    writer.kv_u32("general.quantization_version", 2);
    writer.kv_u32("llama.block_count", 1);
    writer.kv_u32("llama.context_length", 64);
    writer.kv_u32("llama.embedding_length", kHidden);
    writer.kv_u32("llama.feed_forward_length", kFfn);
    writer.kv_u32("llama.attention.head_count", query_heads);
    writer.kv_u32("llama.attention.head_count_kv", 1);
    writer.kv_u32("llama.attention.key_length", kHidden / query_heads);
    writer.kv_u32("llama.attention.value_length", kHidden / query_heads);
    writer.kv_u32("llama.rope.dimension_count", kHidden / query_heads);
    writer.kv_f32("llama.rope.freq_base", 10000.0f);
    if (attention_scale) writer.kv_f32("llama.attention.scale", *attention_scale);
    writer.kv_f32("llama.attention.layer_norm_rms_epsilon", 1.0e-5f);
    if (chat_template)
        writer.kv_str("tokenizer.chat_template",
            "{% for message in messages %}{{ '<|' + message['role'] + '|>' + message['content'] + '</turn>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}");
    writer.kv_u32("tokenizer.ggml.bos_token_id", 2);
    writer.kv_u32("tokenizer.ggml.eos_token_id", 1);
    std::vector<std::string> vocabulary = {"a", "<eos>", "<bos>", "<pad>"};
    if (cli_tokens) vocabulary.push_back("\xc4\x8a");
    writer.kv_arr_str("tokenizer.ggml.tokens", vocabulary);
    if (text_tokens) {
        writer.kv_str("tokenizer.ggml.model", "gpt2");
        writer.kv_arr_str("tokenizer.ggml.merges", {});
    }


    auto f32 = [](const char* name, std::vector<uint64_t> dims) {
        gguf_writer::TensorDecl tensor;
        tensor.name = name;
        tensor.dims = std::move(dims);
        tensor.type = 0;
        size_t count = 1;
        for (uint64_t dim : tensor.dims) count *= static_cast<size_t>(dim);
        tensor.data.resize(count * sizeof(float), 0);
        return tensor;
    };
    writer.add_tensor(f32("token_embd.weight", {kHidden, kVocab}));
    writer.add_tensor(f32("output_norm.weight", {kHidden}));
    writer.add_tensor(f32("blk.0.attn_norm.weight", {kHidden}));
    writer.add_tensor(f32("blk.0.attn_q.weight", {kHidden, kHidden}));
    writer.add_tensor(f32("blk.0.attn_k.weight", {kHidden, kHidden / query_heads}));
    writer.add_tensor(f32("blk.0.attn_v.weight", {kHidden, kHidden / query_heads}));
    writer.add_tensor(f32("blk.0.attn_output.weight", {kHidden, kHidden}));
    writer.add_tensor(f32("blk.0.ffn_norm.weight", {kHidden}));
    writer.add_tensor(f32("blk.0.ffn_gate.weight", {kHidden, kFfn}));
    writer.add_tensor(f32("blk.0.ffn_up.weight", {kHidden, kFfn}));
    writer.add_tensor(f32("blk.0.ffn_down.weight", {kFfn, kHidden}));
    CHECK(writer.write_file(path));
    return path;
}

void test_dense_tied_model_compiles(uint32_t query_heads = 1) {
    const std::string path = write_dense_source(std::nullopt, query_heads);
    auto loaded = ArtifactSet::load_single_file(path);
    unlink(path.c_str());
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    if (!std::holds_alternative<ArtifactSet>(loaded)) return;

    ArtifactSet artifacts = std::get<ArtifactSet>(std::move(loaded));
    auto view = artifacts.view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return;

    const auto result = compile_gguf_product_source(
        std::get<PackageView>(std::move(view)));
    CHECK(std::holds_alternative<GgufProductCompilation>(result));
    if (const auto* report = std::get_if<CompatibilityReport>(&result)) {
        CHECK_MSG(false, "dense tied model failed to compile: %s (code=%u)",
                  report->detail.c_str(), static_cast<unsigned>(report->code));
        return;
    }
    const GgufProductCompilation& compiled = std::get<GgufProductCompilation>(result);
    const SemanticModel& model = compiled.manifest.semantic_model();
    CHECK(model.layers.size() == 1);
    CHECK(model.operators.size() >= 12);
    CHECK(model.vocabulary_size == 4);
    CHECK(model.maximum_context == 64);
    size_t attention_count = 0;
    for (const auto& operation : model.operators) {
        if (const auto* attention = std::get_if<CausalAttentionPayload>(&operation.payload)) {
            CHECK(std::bit_cast<float>(attention->scale_f32_bits) ==
                  1.0f / std::sqrt(static_cast<float>(attention->head_dimension)));
            ++attention_count;
        }
    }
    CHECK(attention_count == 1);
    for (const auto& operation : model.operators) {
        if (!std::holds_alternative<RopePayload>(operation.payload)) continue;
        CHECK(operation.inputs.size() == operation.outputs.size());
        for (size_t i = 0; i < operation.inputs.size(); ++i) {
            const auto& input = model.values[operation.inputs[i]];
            const auto& output = model.values[operation.outputs[i]];
            CHECK(input.dimensions == output.dimensions);
        }
    }

}

void test_attention_scale_override() {
    for (float scale : {0.125f, 0.0f, -1.0f,
                        std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
        const auto path = write_dense_source(scale);
        auto loaded = ArtifactSet::load_single_file(path);
        unlink(path.c_str());
        CHECK(std::holds_alternative<ArtifactSet>(loaded));
        if (!std::holds_alternative<ArtifactSet>(loaded)) continue;
        auto view = std::get<ArtifactSet>(loaded).view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (!std::holds_alternative<PackageView>(view)) continue;
        const auto result = compile_gguf_product_source(std::get<PackageView>(view));
        if (scale != 0.125f) {
            CHECK(std::holds_alternative<CompatibilityReport>(result));
            continue;
        }
        CHECK(std::holds_alternative<GgufProductCompilation>(result));
        if (const auto* compiled = std::get_if<GgufProductCompilation>(&result)) {
            size_t count = 0;
            for (const auto& operation : compiled->manifest.semantic_model().operators) {
                if (const auto* attention = std::get_if<CausalAttentionPayload>(&operation.payload)) {
                    CHECK(std::bit_cast<float>(attention->scale_f32_bits) == scale);
                    ++count;
                }
            }
            CHECK(count == 1);
        }
    }
}

} // namespace

int main() {
    test_untyped_tensor_stays_fail_closed();
    test_dense_tied_model_compiles();
    test_dense_tied_model_compiles(2);
    test_attention_scale_override();
    return test_summary("test_gguf_product_compiler");
}
