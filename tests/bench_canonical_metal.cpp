#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

#include "artifact_set.h"
#include "canonical_metal.h"
#include "compat_rule.h"
#include "gguf.h"
#include "tokenizer.h"

using namespace Laplace;

namespace {

constexpr char kPrompt[] = "M5 performance baseline: Laplace routes semantic operators through Metal.";
constexpr uint32_t kMaximumContext = 128;
constexpr uint32_t kDecodeTokens = 31;

std::shared_ptr<const RuntimePackage> load_package() {
    auto artifacts = ArtifactSet::load_single_file(LAPLACE_QUALIFICATION_GGUF);
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return {};
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    if (!std::holds_alternative<PackageView>(view)) return {};
    auto loaded = load_expected_fixture_gguf(std::get<PackageView>(view), bundled_compatibility_rules());
    if (!std::holds_alternative<ValidatedPackage>(loaded)) return {};
    return std::get<ValidatedPackage>(loaded).runtime_package();
}

uint32_t argmax(const std::vector<float>& logits) {
    return static_cast<uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

double milliseconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

} // namespace

int main() {
    if (!LAPLACE_HAS_QUALIFICATION_GGUF) {
        std::fprintf(stderr, "SKIP: pinned qualification GGUF is unavailable\n");
        return 0;
    }
    GGUFContext gguf;
    Tokenizer tokenizer;
    if (!gguf.open(LAPLACE_QUALIFICATION_GGUF) || !tokenizer.init(gguf)) {
        std::fprintf(stderr, "canonical benchmark could not load the pinned tokenizer\n");
        return 1;
    }
    std::vector<int> encoded;
    if (tokenizer.bos_id() >= 0) encoded.push_back(tokenizer.bos_id());
    const std::vector<int> prompt = tokenizer.encode(kPrompt);
    encoded.insert(encoded.end(), prompt.begin(), prompt.end());
    std::vector<uint32_t> token_ids;
    token_ids.reserve(encoded.size());
    for (int token : encoded) {
        if (token < 0) return 1;
        token_ids.push_back(static_cast<uint32_t>(token));
    }
    if (token_ids.empty() || token_ids.size() + kDecodeTokens > kMaximumContext) return 1;

    const auto package = load_package();
    if (!package) return 1;

    auto warmed = create_qualification_canonical_metal_program(package, kMaximumContext);
    if (!std::holds_alternative<CanonicalMetalProgram>(warmed)) return 1;
    CanonicalMetalProgram warm = std::get<CanonicalMetalProgram>(std::move(warmed));
    if (!std::holds_alternative<CanonicalMetalOutput>(warm.prefill(std::span(token_ids.data(), size_t{1})))) return 1;

    auto created = create_qualification_canonical_metal_program(package, kMaximumContext);
    if (!std::holds_alternative<CanonicalMetalProgram>(created)) return 1;
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));

    const auto prefill_start = std::chrono::steady_clock::now();
    auto prefill = program.prefill(token_ids);
    const double prefill_ms = milliseconds_since(prefill_start);
    if (!std::holds_alternative<CanonicalMetalOutput>(prefill)) return 1;
    const CanonicalMetalOutput prefill_output = std::get<CanonicalMetalOutput>(std::move(prefill));
    uint32_t token = argmax(prefill_output.logits);
    double decode_gpu_ms = 0.0;
    double decode_wait_ms = 0.0;
    double decode_qkv_ms = 0.0;
    double decode_attention_ms = 0.0;
    double decode_ffn_ms = 0.0;
    double decode_final_ms = 0.0;
    uint64_t peak_session_bytes = prefill_output.peak_session_bytes;
    uint32_t command_buffers = 0;
    bool counter_samples = false;
    bool split_command_buffer_profile = false;

    const auto decode_start = std::chrono::steady_clock::now();
    for (uint32_t index = 0; index != kDecodeTokens; ++index) {
        auto decoded = program.decode(token);
        if (!std::holds_alternative<CanonicalMetalOutput>(decoded)) return 1;
        const CanonicalMetalOutput output = std::get<CanonicalMetalOutput>(std::move(decoded));
        token = argmax(output.logits);
        decode_gpu_ms += output.gpu_time_ms;
        decode_wait_ms += output.cpu_wait_ms;
        decode_qkv_ms += output.qkv_gpu_ms;
        decode_attention_ms += output.attention_gpu_ms;
        decode_ffn_ms += output.ffn_gpu_ms;
        decode_final_ms += output.final_gpu_ms;
        peak_session_bytes = std::max(peak_session_bytes, output.peak_session_bytes);
        command_buffers += output.command_buffers;
        counter_samples = counter_samples || output.counter_samples;
        split_command_buffer_profile = split_command_buffer_profile || output.split_command_buffer_profile;
    }
    const double decode_ms = milliseconds_since(decode_start);

    std::printf("benchmark=canonical_metal\n");
    std::printf("prompt=%s\n", kPrompt);
    std::printf("prompt_token_ids=");
    for (uint32_t id : token_ids) std::printf("%u,", id);
    std::printf("\n");
    std::printf("prefill_tokens=%zu prefill_ms=%.3f prefill_tok_s=%.3f ttft_ms=%.3f gpu_ms=%.3f cpu_wait_ms=%.3f command_buffers=%u\n",
                token_ids.size(), prefill_ms, 1000.0 * token_ids.size() / prefill_ms, prefill_ms,
                prefill_output.gpu_time_ms, prefill_output.cpu_wait_ms, prefill_output.command_buffers);
    std::printf("decode_tokens=%u decode_ms=%.3f decode_tok_s=%.3f gpu_ms_per_token=%.3f cpu_wait_ms_per_token=%.3f command_buffers_per_token=%.3f peak_session_bytes=%llu final_token_id=%u\n",
                kDecodeTokens, decode_ms, 1000.0 * kDecodeTokens / decode_ms,
                decode_gpu_ms / kDecodeTokens, decode_wait_ms / kDecodeTokens,
                static_cast<double>(command_buffers) / kDecodeTokens,
                static_cast<unsigned long long>(peak_session_bytes), token);
    if (counter_samples || split_command_buffer_profile) {
        std::printf("decode_profile=%s qkv_gpu_ms_per_token=%.3f attention_gpu_ms_per_token=%.3f ffn_gpu_ms_per_token=%.3f final_gpu_ms_per_token=%.3f\n",
                    counter_samples ? "timestamp_counters" : "split_command_buffers_perturbed",
                    decode_qkv_ms / kDecodeTokens, decode_attention_ms / kDecodeTokens,
                    decode_ffn_ms / kDecodeTokens, decode_final_ms / kDecodeTokens);
    }
    return 0;
}
