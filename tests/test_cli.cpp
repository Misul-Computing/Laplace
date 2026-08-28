#include <cstdio>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <vector>

#include "gguf_writer.h"
#include "test_util.h"

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
    writer.kv_u32("fixture.block_count", 1);
    writer.kv_u32("fixture.context_length", 16);
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

std::string run_cli(const std::string& arguments, int* status) {
    std::string command = "\"" LAPLACE_BINARY_PATH "\" " + arguments + " 2>&1";
    FILE* process = popen(command.c_str(), "r");
    CHECK(process != nullptr);
    std::string output;
    char buffer[256];
    while (process && std::fgets(buffer, sizeof(buffer), process)) output += buffer;
    int result = process ? pclose(process) : -1;
    *status = result == -1 ? -1 : WEXITSTATUS(result);
    return output;
}

bool metal_session_unavailable(int status, const std::string& output) {
    return status != 0 &&
           output.find("code=24") != std::string::npos &&
           output.find("canonical Metal session resources are unavailable") !=
               std::string::npos;
}

void test_public_cache_modes_fail_closed() {
    int status = 0;
    const std::string streaming =
        run_cli("missing.gguf --laplace-stream", &status);
    CHECK(status != 0);
    CHECK(streaming.find("STREAMING_UNSUPPORTED") != std::string::npos);

    const std::string fixed_q4 =
        run_cli("missing.gguf --laplace-kv-q4", &status);
    CHECK(status != 0);
    CHECK(fixed_q4.find("CACHE_MODE_UNQUALIFIED") != std::string::npos);
    const std::string resident =
        run_cli("missing.gguf --laplace-resident", &status);
    CHECK(status != 0);
    CHECK(resident.find("CACHE_MODE_UNQUALIFIED") != std::string::npos);
}

void test_cli_uses_the_canonical_metal_session_for_a_generic_mixed_format_package() {
    constexpr const char* path = "test_cli_canonical.gguf";
    write_generic_canonical_model(path);
    int status = 0;
    const std::string output = run_cli(
        std::string(path) + " -p a -n 2 --no-spec --greedy --max-seq 16 --bench", &status);
    if (metal_session_unavailable(status, output)) {
        std::fprintf(stderr, "SKIP: canonical Metal session is unavailable\n");
        std::remove(path);
        return;
    }
    CHECK_MSG(status == 0, "%s", output.c_str());
    CHECK(output.find("canonical Metal plan") != std::string::npos);
    CHECK(output.find("aa") != std::string::npos);
    CHECK(output.find("[bench] Metal command buffers: prefill=1 decode=1") != std::string::npos);
    std::remove(path);
}

} // namespace

int main() {
    test_public_cache_modes_fail_closed();
    test_cli_uses_the_canonical_metal_session_for_a_generic_mixed_format_package();
    return test_summary("test_cli");
}
