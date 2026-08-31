#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <variant>
#include <vector>

#include "artifact_set.h"
#include "canonical_metal.h"
#include "compat_rule.h"
#include "gguf.h"
#include "kvcache.h"
#include "matmul.h"
#include "model.h"
#include "test_util.h"
#include "tokenizer.h"

using namespace Laplace;

namespace {

int argmax(const std::vector<float>& logits) {
    return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

int test_qwen_greedy_token_parity(const char* path) {
    GGUFContext reference_context;
    const bool reference_open = reference_context.open(path);
    CHECK(reference_open);
    if (!reference_open) return test_summary("test_qwen_canonical_parity");
    matmul_register_weights(reference_context.file_data(), reference_context.file_size());
    Tokenizer tokenizer;
    Model reference;
    const bool tokenizer_ready = tokenizer.init(reference_context);
    const bool reference_ready = reference.init(reference_context);
    CHECK(tokenizer_ready);
    CHECK(reference_ready);
    if (!tokenizer_ready || !reference_ready)
        return test_summary("test_qwen_canonical_parity");
    constexpr int kMaximumContext = 128;
    const bool reserved = reference.reserve(kMaximumContext, 1);
    CHECK(reserved);
    if (!reserved) return test_summary("test_qwen_canonical_parity");
    const auto& config = reference.config();
    int attention_layers = 0;
    for (int layer = 0; layer != config.n_layers; ++layer)
        attention_layers += reference.is_attention_layer(layer);
    KVCache cache;
    const auto layouts = reference.kv_layer_configs(kMaximumContext, KVCacheMode::FP32);
    const bool cache_ready = layouts.empty()
        ? cache.init(attention_layers, config.n_kv_heads, config.head_dim, kMaximumContext, KVCacheMode::FP32)
        : cache.init(layouts);
    CHECK(cache_ready);
    if (!cache_ready) {
        return test_summary("test_qwen_canonical_parity");
    }

    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return test_summary("test_qwen_canonical_parity");
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return test_summary("test_qwen_canonical_parity");
    auto loaded = load_validated_gguf(std::get<PackageView>(view));
    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
    if (!std::holds_alternative<ValidatedPackage>(loaded)) return test_summary("test_qwen_canonical_parity");
    auto created = create_qualification_canonical_metal_program(
        std::get<ValidatedPackage>(loaded).runtime_package(), kMaximumContext);
    CHECK(std::holds_alternative<CanonicalMetalProgram>(created));
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return test_summary("test_qwen_canonical_parity");
    CanonicalMetalProgram canonical = std::get<CanonicalMetalProgram>(std::move(created));

    std::vector<int> prompt = tokenizer.encode_chat("Explain why Metal is fast on Apple Silicon.");
    CHECK(!prompt.empty());
    if (prompt.empty()) return test_summary("test_qwen_canonical_parity");
    std::vector<float> reference_logits(static_cast<size_t>(config.vocab));
    CanonicalMetalOutput canonical_output;
    int position = 0;
    for (int token : prompt) {
        const bool forwarded = reference.forward(token, position, cache, reference_logits.data());
        CHECK(forwarded);
        if (!forwarded)
            return test_summary("test_qwen_canonical_parity");
        const uint32_t canonical_token = static_cast<uint32_t>(token);
        auto executed = canonical.prefill(std::span<const uint32_t>(&canonical_token, 1));
        CHECK(std::holds_alternative<CanonicalMetalOutput>(executed));
        if (!std::holds_alternative<CanonicalMetalOutput>(executed)) return test_summary("test_qwen_canonical_parity");
        canonical_output = std::get<CanonicalMetalOutput>(std::move(executed));
        ++position;
    }

    std::vector<int> tokens;
    for (int index = 0; index != 16; ++index) {
        const int reference_token = argmax(reference_logits);
        const int canonical_token = argmax(canonical_output.logits);
        CHECK(canonical_token == reference_token);
        tokens.push_back(canonical_token);
        if (index + 1 == 16) break;
        const bool forwarded = reference.forward(reference_token, position, cache, reference_logits.data());
        CHECK(forwarded);
        if (!forwarded)
            return test_summary("test_qwen_canonical_parity");
        const uint32_t next = static_cast<uint32_t>(canonical_token);
        auto executed = canonical.decode(next);
        CHECK(std::holds_alternative<CanonicalMetalOutput>(executed));
        if (!std::holds_alternative<CanonicalMetalOutput>(executed)) return test_summary("test_qwen_canonical_parity");
        canonical_output = std::get<CanonicalMetalOutput>(std::move(executed));
        ++position;
    }
    std::fprintf(stderr, "Qwen canonical/CPU greedy token IDs:");
    for (int token : tokens) std::fprintf(stderr, " %d", token);
    std::fputc('\n', stderr);
    return test_summary("test_qwen_canonical_parity");
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "SKIP: pass the local Qwen GGUF path\n");
        return test_summary("test_qwen_canonical_parity");
    }
    setenv("LAPLACE_LM_CPU", "1", 1);
    return test_qwen_greedy_token_parity(argv[1]);
}
