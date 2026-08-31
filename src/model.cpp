// model.cpp - pluggable architecture dispatcher
#include "model.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>

#include "matmul.h"
#include "ops.h"
#include "token_graph_backend.h"
#include "trace.h"
#include "laplace_moe.h"
#include "topology.h"

#if defined(__APPLE__)
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>
#endif

namespace {
// LAPLACE_PROF=1: split out the non-matmul hot spots.
struct PhaseProf {
    bool on = std::getenv("LAPLACE_PROF") != nullptr;
    double fwd[9] = {};
    long   fwd_n[9] = {};
    double lm_head = 0.0;
    long   lm_n = 0;
    ~PhaseProf() {
        if (!on) return;
        for (int m = 1; m <= 8; m++) {
            if (fwd_n[m]) fprintf(stderr, "PROF forward_batch M=%d: %ld calls, %.1f ms avg\n",
                                  m, fwd_n[m], 1e3 * fwd[m] / fwd_n[m]);
        }
        if (lm_n) fprintf(stderr, "PROF lm_head: %ld calls, %.1f ms avg\n",
                          lm_n, 1e3 * lm_head / lm_n);
    }
};
PhaseProf g_pprof;
struct PhaseTimer {
    double* acc;
    std::chrono::steady_clock::time_point t0;
    explicit PhaseTimer(double* a) : acc(a) {
        if (g_pprof.on) t0 = std::chrono::steady_clock::now();
    }
    ~PhaseTimer() {
        if (g_pprof.on) *acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};
}

namespace Laplace {

void Model::restore_ssm_state(const float* checkpoint) {
    if (buffers_.ssm_conv_state.empty() && buffers_.ssm_recurrent.empty()) return;
    std::memcpy(buffers_.ssm_conv_state.data(), checkpoint,
                buffers_.ssm_conv_state.size() * sizeof(float));
    std::memcpy(buffers_.ssm_recurrent.data(),
                checkpoint + buffers_.ssm_conv_state.size(),
                buffers_.ssm_recurrent.size() * sizeof(float));
}

bool Model::init(const GGUFContext& gguf) {
    const auto& m = gguf.metadata();
    reset_load_attempt();

    LaplaceMoE::set_file_fd(gguf.fd());
    LaplaceMoE::set_mmap_base(gguf.file_data());

    auto arch_name = meta_str(m, "general.architecture");
    if (!arch_name) {
        fprintf(stderr, "model: missing general.architecture\n");
        return false;
    }
    TopologyPlan topology;
    std::string topology_error;
    if (synthesize_topology(gguf, &topology, &topology_error)) {
        topology_ = topology;
        arch_ = create_adaptive_arch(std::move(topology));
    } else {
        arch_ = create_arch(*arch_name);
        if (!arch_) {
            fprintf(stderr, "%s\n", topology_error.c_str());
        }
    }
    if (!arch_) {
        fprintf(stderr, "model: unsupported architecture '%s'\n", arch_name->c_str());
        return false;
    }

    if (!arch_->load_config(gguf, &cfg_)) {
        fprintf(stderr, "model: %s load_config failed\n", arch_->name());
        return false;
    }

    const auto load_shared_tensors = [&]() {
        token_embd_  = gguf.find_tensor("token_embd.weight");
        output_norm_ = gguf.find_tensor("output_norm.weight");
        output_      = gguf.find_tensor("output.weight");
        if (!token_embd_ || !output_norm_) {
            fprintf(stderr, "model: missing token_embd or output_norm\n");
            return false;
        }
        cfg_.vocab = static_cast<int>(token_embd_->dims[1]);
        cfg_.hidden = static_cast<int>(token_embd_->dims[0]);
        if (output_ && output_->dims[1] == (uint64_t)cfg_.vocab) {
            cfg_.tied_lm_head = false;
        } else {
            output_ = token_embd_;
            cfg_.tied_lm_head = true;
        }
        return true;
    };
    if (!load_shared_tensors()) return false;

    layers_.assign(cfg_.n_layers, {});
    is_attention_.assign(cfg_.n_layers, false);
    kv_layer_idx_.assign(cfg_.n_layers, -1);
    if (!arch_->load_weights(gguf, cfg_, &layers_, &is_attention_, &kv_layer_idx_)) {
        fprintf(stderr, "model: %s load_weights failed\n", arch_->name());
        if (std::string(arch_->name()) == "adaptive") {
            reset_load_attempt();
            arch_ = create_arch(*arch_name);
            if (arch_ && arch_->load_config(gguf, &cfg_) &&
                load_shared_tensors()) {
                layers_.assign(cfg_.n_layers, {});
                is_attention_.assign(cfg_.n_layers, false);
                kv_layer_idx_.assign(cfg_.n_layers, -1);
                if (arch_->load_weights(gguf, cfg_, &layers_,
                                        &is_attention_, &kv_layer_idx_)) {
                    fprintf(stderr, "model: fell back to %s\n", arch_->name());
                    auto t0 = std::chrono::steady_clock::now();
                    plan_residency();
                    fprintf(stderr, "[load] residency plan: %.1f ms\n",
                            std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0).count());
                    return true;
                }
            }
        }
        return false;
    }

    auto t0 = std::chrono::steady_clock::now();
    plan_residency();
    fprintf(stderr, "[load] residency plan: %.1f ms\n",
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count());
    return true;
}

void Model::reset_load_attempt() {
    cfg_ = {};
    arch_.reset();
    token_embd_ = nullptr;
    output_norm_ = nullptr;
    output_ = nullptr;
    layers_.clear();
    is_attention_.clear();
    kv_layer_idx_.clear();
    max_seq_ = 0;
    max_batch_ = 1;
    streaming_experts_ = false;
    buffers_ = {};
    topology_ = {};
}

std::vector<KVLayerConfig> Model::kv_layer_configs(
        int max_seq_len, KVCacheMode mode) const {
    return make_kv_layer_configs(topology_, max_seq_len, mode);
}

// Decide which weights stay resident and which stream from SSD.
// On Apple Silicon, all tensors are views into a single MAP_SHARED mmap.
// The OS LRU keeps recently-touched pages in RAM, but when the model
// barely fits, the OS thrashes: it evicts dense weights (attention,
// FFN, router) to make room for expert tensors, then re-faults them
// next layer. The fix: fault all dense weights into RAM at load so
// they stay resident, and only stream the expert tensors on demand
// through LaplaceMoE. This converts OS thrashing into controlled
// expert-only working set.
void Model::plan_residency() {
#if defined(__APPLE__)
    // Physical RAM.
    size_t ram_bytes = 0;
    size_t len = sizeof(ram_bytes);
    if (sysctlbyname("hw.memsize", &ram_bytes, &len, nullptr, 0) != 0 || ram_bytes == 0)
        return;

    // Total model size (the mmap spans the whole file).
    // We approximate model weight bytes from the tensors we hold.
    auto tensor_bytes = [](const Tensor* t) -> size_t {
        return t ? t->nbytes() : 0;
    };

    // Dense (always-resident) weight bytes: everything except the
    // stacked expert tensors.
    size_t dense_bytes = 0;
    size_t expert_bytes = 0;
    bool has_moe = false;

    dense_bytes += tensor_bytes(token_embd_);
    dense_bytes += tensor_bytes(output_norm_);
    if (!cfg_.tied_lm_head) dense_bytes += tensor_bytes(output_);

    for (const auto& W : layers_) {
        dense_bytes += tensor_bytes(W.attn_norm);
        dense_bytes += tensor_bytes(W.post_attn_norm);
        dense_bytes += tensor_bytes(W.ffn_norm);
        dense_bytes += tensor_bytes(W.ffn_gate);
        dense_bytes += tensor_bytes(W.ffn_up);
        dense_bytes += tensor_bytes(W.ffn_down);
        dense_bytes += tensor_bytes(W.attn_q);
        dense_bytes += tensor_bytes(W.attn_k);
        dense_bytes += tensor_bytes(W.attn_v);
        dense_bytes += tensor_bytes(W.attn_q_bias);
        dense_bytes += tensor_bytes(W.attn_k_bias);
        dense_bytes += tensor_bytes(W.attn_v_bias);
        dense_bytes += tensor_bytes(W.attn_q_norm);
        dense_bytes += tensor_bytes(W.attn_k_norm);
        dense_bytes += tensor_bytes(W.attn_output);
        dense_bytes += tensor_bytes(W.attn_qkv);
        dense_bytes += tensor_bytes(W.attn_gate);
        dense_bytes += tensor_bytes(W.ssm_a);
        dense_bytes += tensor_bytes(W.ssm_conv1d);
        dense_bytes += tensor_bytes(W.ssm_dt_bias);
        dense_bytes += tensor_bytes(W.ssm_alpha);
        dense_bytes += tensor_bytes(W.ssm_beta);
        dense_bytes += tensor_bytes(W.ssm_norm);
        dense_bytes += tensor_bytes(W.ssm_out);
        dense_bytes += tensor_bytes(W.moe_gate_inp);
        dense_bytes += tensor_bytes(W.moe_gate_inp_scale);
        dense_bytes += tensor_bytes(W.moe_down_exps_scale);
        dense_bytes += tensor_bytes(W.pre_ffw_norm_2);
        dense_bytes += tensor_bytes(W.post_ffw_norm_1);
        dense_bytes += tensor_bytes(W.post_ffw_norm_2);
        dense_bytes += tensor_bytes(W.post_ffw_norm);
        dense_bytes += tensor_bytes(W.layer_output_scale);
        if (W.moe_gate_up_exps) {
            has_moe = true;
            expert_bytes += tensor_bytes(W.moe_gate_up_exps);
        }
        if (W.moe_down_exps) {
            has_moe = true;
            expert_bytes += tensor_bytes(W.moe_down_exps);
        }
    }

    // Adaptive budget: estimate OS overhead, KV cache, and activations,
    // then give the rest to weights. This works for any model on any
    // Apple Silicon Mac without a magic percentage.
    size_t os_overhead = std::max<size_t>(2ULL << 30, ram_bytes / 10);
    size_t available = ram_bytes - os_overhead;

    // KV cache estimate: 2 (K+V) * n_kv_heads * head_dim * 2 bytes (FP16)
    // per token per layer. The KV cache grows with context, so reserve
    // for a moderate working context (1024 tokens), not the full max_seq.
    // The OS page cache will absorb the rest.
    int n_layers = static_cast<int>(layers_.size());
    int kv_heads = cfg_.n_kv_heads > 0 ? cfg_.n_kv_heads : 1;
    int head_dim = cfg_.head_dim > 0 ? cfg_.head_dim : 128;
    size_t kv_per_token = static_cast<size_t>(n_layers) * 2 * kv_heads * head_dim * 2;
    size_t kv_bytes = kv_per_token * 1024;  // 1024-token working context

    // Activation estimate: hidden * 4 bytes * ~10 intermediate buffers.
    // Not per-layer (buffers are reused across layers).
    int hidden = cfg_.hidden > 0 ? cfg_.hidden : 1;
    size_t act_bytes = static_cast<size_t>(hidden) * 4 * 10;

    // Safety margin: 512 MB.
    size_t safety = 512ULL << 20;

    size_t budget = (available > kv_bytes + act_bytes + safety)
                        ? available - kv_bytes - act_bytes - safety
                        : 0;
    size_t total = dense_bytes + expert_bytes;

    // 26B experts fit the numeric budget but pinning all of them was
    // measured slower than streaming (5.0 vs 9.6 tok/s decode, same
    // command, this M5 Pro). Keep streaming when experts dominate.
    bool prefer_stream = has_moe && expert_bytes > total / 2
                         && expert_bytes > budget / 2;

    if (!has_moe || (total <= budget && !prefer_stream)) {
        // Model fits: fault everything in so the OS treats the pages as
        // hot and keeps them resident. Touch ranges in parallel: a
        // single-threaded walk serializes ~1M soft faults and dominates
        // load time on multi-GB models.
        streaming_experts_ = false;
        auto fault_in = [](const Tensor* t) {
            if (!t || !t->data) return;
            (void)madvise(const_cast<uint8_t*>(t->data), t->nbytes(), MADV_WILLNEED);
        };
        fault_in(token_embd_);
        fault_in(output_norm_);
        if (!cfg_.tied_lm_head) fault_in(output_);
        for (const auto& W : layers_) {
            fault_in(W.attn_norm); fault_in(W.post_attn_norm);
            fault_in(W.ffn_norm); fault_in(W.ffn_gate); fault_in(W.ffn_up); fault_in(W.ffn_down);
            fault_in(W.attn_q); fault_in(W.attn_k); fault_in(W.attn_v);
            fault_in(W.attn_q_bias); fault_in(W.attn_k_bias); fault_in(W.attn_v_bias);
            fault_in(W.attn_q_norm); fault_in(W.attn_k_norm); fault_in(W.attn_output);
            fault_in(W.attn_qkv); fault_in(W.attn_gate);
            fault_in(W.ssm_a); fault_in(W.ssm_conv1d); fault_in(W.ssm_dt_bias);
            fault_in(W.ssm_alpha); fault_in(W.ssm_beta); fault_in(W.ssm_norm); fault_in(W.ssm_out);
            fault_in(W.moe_gate_inp); fault_in(W.moe_gate_inp_scale);
            fault_in(W.moe_down_exps_scale);
            fault_in(W.pre_ffw_norm_2); fault_in(W.post_ffw_norm_1);
            fault_in(W.post_ffw_norm_2); fault_in(W.post_ffw_norm);
            fault_in(W.layer_output_scale);
            fault_in(W.moe_gate_up_exps);
            fault_in(W.moe_down_exps);
        }
        {
            // One contiguous work list, split into equal byte spans, one
            // thread per span. Sequential order inside each span keeps
            // NVMe readahead effective.
            std::vector<std::pair<const uint8_t*, size_t>> ranges;
            const auto push = [&](const Tensor* t) {
                if (t && t->data && t->nbytes())
                    ranges.emplace_back((const uint8_t*)t->data, t->nbytes());
            };
            push(token_embd_); push(output_norm_);
            if (!cfg_.tied_lm_head) push(output_);
            for (const auto& W : layers_) {
                push(W.attn_norm); push(W.post_attn_norm);
                push(W.ffn_norm); push(W.ffn_gate); push(W.ffn_up); push(W.ffn_down);
                push(W.attn_q); push(W.attn_k); push(W.attn_v);
                push(W.attn_q_bias); push(W.attn_k_bias); push(W.attn_v_bias);
                push(W.attn_q_norm); push(W.attn_k_norm); push(W.attn_output);
                push(W.attn_qkv); push(W.attn_gate);
                push(W.ssm_a); push(W.ssm_conv1d); push(W.ssm_dt_bias);
                push(W.ssm_alpha); push(W.ssm_beta); push(W.ssm_norm); push(W.ssm_out);
                push(W.moe_gate_inp); push(W.moe_gate_inp_scale);
                push(W.moe_down_exps_scale);
                push(W.pre_ffw_norm_2); push(W.post_ffw_norm_1);
                push(W.post_ffw_norm_2); push(W.post_ffw_norm);
                push(W.layer_output_scale);
                push(W.moe_gate_up_exps);
                push(W.moe_down_exps);
            }
            size_t total_bytes = 0;
            for (auto& r : ranges) total_bytes += r.second;
            long ps = sysconf(_SC_PAGESIZE);
            size_t page = ps > 0 ? static_cast<size_t>(ps) : 4096;
            unsigned hw_threads = std::thread::hardware_concurrency();
            int nthreads = (int)(hw_threads ? hw_threads : 8);
            if (nthreads > 12) nthreads = 12;
            if ((size_t)nthreads > ranges.size()) nthreads = (int)ranges.size();
            if (nthreads < 1) nthreads = 1;
            size_t per = total_bytes / (size_t)nthreads + 1;
            std::vector<std::thread> workers;
            // Per span: readahead hint, fault the pages in by touching,
            // then wire immediately. Touching beats waiting on
            // MADV_WILLNEED alone, which livelocks when free memory is
            // smaller than the model: the kernel recycles early ranges
            // while we wait for late ones.
            auto lock_span = [&](size_t begin_byte, size_t end_byte) {
                size_t seen = 0;
                for (auto& r : ranges) {
                    size_t rs = r.second;
                    if (seen + rs <= begin_byte) { seen += rs; continue; }
                    size_t lo = begin_byte > seen ? begin_byte - seen : 0;
                    size_t hi = end_byte - seen < rs ? end_byte - seen : rs;
                    if (hi > lo) {
                        size_t alo = lo & ~(page - 1);
                        (void)madvise(
                            const_cast<uint8_t*>(r.first + alo), hi - alo,
                            MADV_WILLNEED);
                        const volatile uint8_t* p = r.first;
                        for (size_t off = lo; off < hi; off += page)
                            (void)p[off];
                        (void)mlock(const_cast<uint8_t*>(r.first + alo),
                                    hi - alo);
                    }
                    seen += rs;
                    if (seen >= end_byte) break;
                }
            };
            for (int i = 0; i < nthreads; i++) {
                size_t b = (size_t)i * per;
                size_t e = b + per;
                if (e > total_bytes) e = total_bytes;
                workers.emplace_back(lock_span, b, e);
            }
            for (auto& w : workers) w.join();
        }
        fprintf(stderr, "residency: model %.2f GB fits in %.2f GB budget, pinned all weights\n",
                total / 1e9, budget / 1e9);
        LaplaceMoE::set_cache_budget(0);
        return;
    }

    // If dense alone exceeds the budget, the model is too large for this
    // machine. We cannot pin the dense weights and still have room for KV
    // cache and activations.
    if (dense_bytes > budget) {
        fprintf(stderr, "residency: dense weights %.2f GB exceed %.2f GB budget, "
                        "model too large for this machine\n",
                dense_bytes / 1e9, budget / 1e9);
        LaplaceMoE::set_cache_budget(0);
        return;
    }

    // Model does not fit: pin dense, stream experts.
    streaming_experts_ = true;
    LaplaceMoE::set_streaming(true);
    auto pin = [](const Tensor* t) {
        if (!t || !t->data) return;
        // MADV_WILLNEED starts readahead. A sequential touch forces the
        // pages into the working set so the OS treats them as hot and
        // avoids evicting them under pressure.
        (void)madvise(const_cast<uint8_t*>(t->data), t->nbytes(), MADV_WILLNEED);
        const volatile uint8_t* p = t->data;
        size_t sz = t->nbytes();
        long ps = sysconf(_SC_PAGESIZE);
        size_t page = ps > 0 ? static_cast<size_t>(ps) : 4096;
        for (size_t off = 0; off + page <= sz; off += page)
            (void)p[off];
        if (sz) (void)p[sz - 1];
        (void)mlock(const_cast<uint8_t*>(t->data), sz);
    };
    pin(token_embd_);
    pin(output_norm_);
    if (!cfg_.tied_lm_head) pin(output_);
    for (const auto& W : layers_) {
        pin(W.attn_norm); pin(W.post_attn_norm);
        pin(W.ffn_norm); pin(W.ffn_gate); pin(W.ffn_up); pin(W.ffn_down);
        pin(W.attn_q); pin(W.attn_k); pin(W.attn_v);
        pin(W.attn_q_bias); pin(W.attn_k_bias); pin(W.attn_v_bias);
        pin(W.attn_q_norm); pin(W.attn_k_norm); pin(W.attn_output);
        pin(W.attn_qkv); pin(W.attn_gate);
        pin(W.ssm_a); pin(W.ssm_conv1d); pin(W.ssm_dt_bias);
        pin(W.ssm_alpha); pin(W.ssm_beta); pin(W.ssm_norm); pin(W.ssm_out);
        pin(W.moe_gate_inp); pin(W.moe_gate_inp_scale);
        pin(W.moe_down_exps_scale);
        pin(W.pre_ffw_norm_2); pin(W.post_ffw_norm_1);
        pin(W.post_ffw_norm_2); pin(W.post_ffw_norm);
        pin(W.layer_output_scale);
    }
    // Expert GEMV reads the file map. First touch faults during compute.
    // 4 GB is the seen-set budget for hit accounting, not a heap copy.
    int top_k = cfg_.n_experts_used > 0 ? cfg_.n_experts_used : 1;
    size_t active_bytes = 0;
    for (const auto& W : layers_) {
        if (W.moe_gate_up_exps)
            active_bytes += LaplaceMoE::per_expert_bytes(W.moe_gate_up_exps) *
                            static_cast<size_t>(top_k);
        if (W.moe_down_exps)
            active_bytes += LaplaceMoE::per_expert_bytes(W.moe_down_exps) *
                            static_cast<size_t>(top_k);
    }
    size_t free_after_dense = (budget > dense_bytes) ? budget - dense_bytes : 0;
    size_t cache_budget = 4ULL << 30;
    if (cache_budget > free_after_dense) cache_budget = free_after_dense;
    if (cache_budget < active_bytes) cache_budget = active_bytes;
    fprintf(stderr, "residency: model %.2f GB exceeds %.2f GB budget, "
                    "pinning %.2f GB dense, staging %.2f GB experts "
                    "(working set %.2f GB, top-%d token = %.2f GB)\n",
            total / 1e9, budget / 1e9, dense_bytes / 1e9, expert_bytes / 1e9,
            cache_budget / 1e9, top_k, active_bytes / 1e9);
    LaplaceMoE::set_cache_budget(cache_budget > 0 ? cache_budget : 1);
#endif
}

bool Model::reserve(int max_seq_len, int max_batch) {
    max_seq_ = max_seq_len;
    max_batch_ = max_batch > 0 ? max_batch : 1;

    arch_->reserve(cfg_, max_seq_, max_batch_, &buffers_);

    // RoPE cos/sin tables: [max_seq, rope_pairs].
    //   angle(pos, p) = pos * base^(-2p / rope_dim_count)
    if (arch_->needs_shared_rope_table()) {
        buffers_.rope_pairs = cfg_.rope_dim_count / 2;
        buffers_.rope_cos.resize(
            static_cast<size_t>(max_seq_) * buffers_.rope_pairs);
        buffers_.rope_sin.resize(
            static_cast<size_t>(max_seq_) * buffers_.rope_pairs);
        for (int p = 0; p < buffers_.rope_pairs; p++) {
            double inv_freq = std::pow(
                static_cast<double>(cfg_.rope_freq_base),
                -2.0 * p / cfg_.rope_dim_count);
            for (int pos = 0; pos < max_seq_; pos++) {
                double angle = pos * inv_freq;
                buffers_.rope_cos[
                    static_cast<size_t>(pos) * buffers_.rope_pairs + p] =
                    static_cast<float>(std::cos(angle));
                buffers_.rope_sin[
                    static_cast<size_t>(pos) * buffers_.rope_pairs + p] =
                    static_cast<float>(std::sin(angle));
            }
        }
    } else {
        buffers_.rope_pairs = 0;
        buffers_.rope_cos.clear();
        buffers_.rope_sin.clear();
    }

    return true;
}

bool Model::forward(int token, int pos, KVCache& kv, float* logits) {
    return forward_batch(&token, 1, pos, kv, logits, nullptr);
}

bool Model::forward_batch(const int* tokens, int M, int pos0, KVCache& kv,
                          float* logits, float* checkpoints) {
    PhaseTimer pt_fwd(&g_pprof.fwd[M <= 8 ? M : 8]);
    if (g_pprof.on && M <= 8) g_pprof.fwd_n[M]++;
    const int H = cfg_.hidden;
    const size_t row_bytes = token_embd_->nbytes() / token_embd_->dims[1];

    // Token embedding lookups — dequantize one row per token via a view
    // (rows are whole blocks: hidden is a multiple of the block size for
    // every supported format).
    for (int m = 0; m < M; m++) {
        Tensor row = *token_embd_;
        row.data += static_cast<size_t>(tokens[m]) * row_bytes;
        dequantize(row, buffers_.x.data() + static_cast<size_t>(m) * H, H);
    }
    // Gemma4 scales embeddings by sqrt(hidden).
    if (cfg_.embed_scale != 1.0f) {
        for (size_t j = 0; j < static_cast<size_t>(M) * H; j++)
            buffers_.x[j] *= cfg_.embed_scale;
    }
    if (M == 1) trace("input_embed", -1, buffers_.x.data(), H);

    TokenGraphBackend& token_graph = token_graph_backend_
        ? *token_graph_backend_ : metal_token_graph_backend();
    ScopedTokenGraphBackend token_graph_scope(token_graph);
    bool tok = false;
    if (M == 1 && metal_token_ && arch_->supports_token_graph() &&
        token_graph.available()) {
        const int inter = cfg_.intermediate;
        const int Hq = cfg_.n_q_heads;
        const int Hk = cfg_.n_kv_heads;
        const int Dh = cfg_.head_dim;
        tok = token_graph.begin(H, inter, cfg_.expert_inter, cfg_.n_experts_used,
                                cfg_.n_experts, Hq, Hk, Dh, max_seq_,
                                cfg_.n_layers, pos0);
        if (tok) token_graph.upload_x(buffers_.x.data(), H);
    }

    // Each block is handled by the architecture implementation.
    LaplaceMoE::set_current_token(pos0);
    for (int i = 0; i < cfg_.n_layers; i++) {
        arch_->forward_layer(i, layers_[i], cfg_, M, pos0, kv, &buffers_, checkpoints);
        if (tok && !token_graph.active()) return false;
        if (M == 1 && !(tok && token_graph.active()))
            trace("post_ffn", i, buffers_.x.data(), H);
    }
    // GPU LM is Task 5: only after a real-tensor microbench and Hello.
    // Token graph copies residual back; CPU does final RMSNorm + LM head.
    if (tok && token_graph.active() && !token_graph.end(buffers_.x.data(), H)) {
        token_graph.abort();
        return false;
    }

    // Final norm + LM head for every position (the batched LM-head matmul is
    // the single biggest weight stream, amortized across all M tokens).
    for (int m = 0; m < M; m++) {
        ops::rmsnorm(buffers_.x.data() + static_cast<size_t>(m) * H,
                     reinterpret_cast<const float*>(output_norm_->data),
                     buffers_.x_norm.data() + static_cast<size_t>(m) * H,
                     H, cfg_.rms_eps);
    }
    if (M == 1) trace("result_norm", -1, buffers_.x_norm.data(), H);
    {
        PhaseTimer pt_lm(&g_pprof.lm_head);
        if (g_pprof.on) g_pprof.lm_n++;
        matmul_lm_head(buffers_.x_norm.data(), *output_, logits, M, H, cfg_.vocab);
    }

    // Gemma4 final logit softcapping: logits = tanh(logits / cap) * cap
    if (cfg_.logit_softcap > 0.0f) {
        float cap = cfg_.logit_softcap;
        float inv_cap = 1.0f / cap;
        for (size_t j = 0; j < static_cast<size_t>(M) * cfg_.vocab; j++)
            logits[j] = std::tanh(logits[j] * inv_cap) * cap;
    }
    return true;
}

} // namespace Laplace
