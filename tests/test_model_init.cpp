#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "gguf.h"
#include "gguf_writer.h"
#include "model.h"
#include "test_util.h"

using namespace Laplace;

namespace {

gguf_writer::TensorDecl f32_tensor(const char* name,
                                   std::vector<uint64_t> dims) {
    size_t count = 1;
    for (uint64_t dim : dims) count *= static_cast<size_t>(dim);
    gguf_writer::TensorDecl tensor;
    tensor.name = name;
    tensor.dims = std::move(dims);
    tensor.type = 0;
    tensor.data.resize(count * sizeof(float));
    return tensor;
}

void write_adaptive_failure_with_llama_fallback(const char* path) {
    gguf_writer::Writer writer;
    writer.kv_str("general.architecture", "llama");
    writer.kv_u32("llama.block_count", 1);
    writer.kv_u32("llama.embedding_length", 4);
    writer.kv_u32("llama.feed_forward_length", 8);
    writer.kv_u32("llama.context_length", 16);
    writer.kv_u32("llama.attention.head_count", 1);
    writer.kv_u32("llama.attention.head_count_kv", 1);
    writer.kv_u32("llama.attention.key_length", 4);
    writer.kv_u32("llama.expert_count", 2);
    writer.kv_u32("llama.expert_used_count", 1);
    writer.kv_u32("llama.expert_feed_forward_length", 4);

    writer.add_tensor(f32_tensor("token_embd.weight", {4, 3}));
    writer.add_tensor(f32_tensor("output_norm.weight", {4}));
    writer.add_tensor(f32_tensor("output.weight", {4, 3}));
    writer.add_tensor(f32_tensor("rope_freqs.weight", {2}));
    writer.add_tensor(f32_tensor("blk.0.attn_norm.weight", {4}));
    writer.add_tensor(f32_tensor("blk.0.post_attention_norm.weight", {4}));
    writer.add_tensor(f32_tensor("blk.0.attn_q.weight", {4, 4}));
    writer.add_tensor(f32_tensor("blk.0.attn_k.weight", {4, 4}));
    writer.add_tensor(f32_tensor("blk.0.attn_v.weight", {4, 4}));
    writer.add_tensor(f32_tensor("blk.0.attn_output.weight", {4, 4}));
    writer.add_tensor(f32_tensor("blk.0.ffn_gate.weight", {4, 8}));
    writer.add_tensor(f32_tensor("blk.0.ffn_up.weight", {4, 8}));
    writer.add_tensor(f32_tensor("blk.0.ffn_down.weight", {8, 4}));
    // Adaptive treats this as routed-MoE evidence and rejects the missing
    // expert tensors. Llama ignores it and can load the same dense weights.
    writer.add_tensor(f32_tensor("blk.0.ffn_gate_inp.weight", {4, 2}));
    CHECK(writer.write_file(path));
}

void test_named_fallback_discards_adaptive_attempt_state() {
    constexpr const char* path = "test_model_init_fallback.gguf";
    write_adaptive_failure_with_llama_fallback(path);

    GGUFContext gguf;
    CHECK(gguf.open(path));

    Model model;
    CHECK(model.init(gguf));
    CHECK(model.arch() != nullptr);
    if (model.arch()) CHECK(std::string(model.arch()->name()) == "llama");

    const ModelConfig& cfg = model.config();
    CHECK(cfg.n_experts == 0);
    CHECK(cfg.n_experts_used == 0);
    CHECK(cfg.expert_inter == 0);
    CHECK(cfg.embed_scale == 1.0f);
    CHECK(cfg.logit_softcap == 0.0f);
    CHECK(model.kv_layer_configs(16, KVCacheMode::FP16).empty());

    std::remove(path);
}

void write_named_q4_forward_model(const char* path) {
    constexpr uint32_t hidden = 32;
    gguf_writer::Writer writer;
    writer.kv_str("general.architecture", "llama");
    writer.kv_u32("llama.block_count", 1);
    writer.kv_u32("llama.embedding_length", hidden);
    writer.kv_u32("llama.feed_forward_length", hidden);
    writer.kv_u32("llama.context_length", 128);
    writer.kv_u32("llama.attention.head_count", 1);
    writer.kv_u32("llama.attention.head_count_kv", 1);
    writer.kv_u32("llama.attention.key_length", hidden);
    writer.kv_u32("llama.expert_count", 2);
    writer.kv_u32("llama.expert_used_count", 1);
    writer.kv_u32("llama.expert_feed_forward_length", hidden);

    writer.add_tensor(f32_tensor("token_embd.weight", {hidden, 3}));
    writer.add_tensor(f32_tensor("output_norm.weight", {hidden}));
    writer.add_tensor(f32_tensor("output.weight", {hidden, 3}));
    writer.add_tensor(f32_tensor("rope_freqs.weight", {hidden / 2}));
    writer.add_tensor(f32_tensor("blk.0.attn_norm.weight", {hidden}));
    writer.add_tensor(f32_tensor("blk.0.post_attention_norm.weight", {hidden}));
    writer.add_tensor(f32_tensor("blk.0.attn_q.weight", {hidden, hidden}));
    writer.add_tensor(f32_tensor("blk.0.attn_k.weight", {hidden, hidden}));
    writer.add_tensor(f32_tensor("blk.0.attn_v.weight", {hidden, hidden}));
    writer.add_tensor(f32_tensor("blk.0.attn_output.weight", {hidden, hidden}));
    writer.add_tensor(f32_tensor("blk.0.ffn_gate.weight", {hidden, hidden}));
    writer.add_tensor(f32_tensor("blk.0.ffn_up.weight", {hidden, hidden}));
    writer.add_tensor(f32_tensor("blk.0.ffn_down.weight", {hidden, hidden}));
    // Trigger adaptive failure so the real named Llama attention path runs.
    writer.add_tensor(f32_tensor("blk.0.ffn_gate_inp.weight", {hidden, 2}));
    CHECK(writer.write_file(path));
}

void test_named_attention_does_not_treat_fixed_q4_as_fp16() {
    constexpr const char* path = "test_model_init_fixed_q4.gguf";
    write_named_q4_forward_model(path);
    GGUFContext gguf;
    CHECK(gguf.open(path));
    Model model;
    CHECK(model.init(gguf));
    CHECK(model.reserve(128, 1));

    KVCache kv;
    CHECK(kv.init(1, 1, 32, 128, KVCacheMode::LAPLACE_Q4));
    float logits[3] = {};
    for (int pos = 0; pos < 128; ++pos) model.forward(0, pos, kv, logits);
    for (float logit : logits) CHECK(std::isfinite(logit));
    std::remove(path);
}

} // namespace

int main() {
    test_named_fallback_discards_adaptive_attempt_state();
    test_named_attention_does_not_treat_fixed_q4_as_fp16();
    return test_summary("test_model_init");
}
