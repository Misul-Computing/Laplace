#include "topology.h"

#include <algorithm>
#include <cmath>

namespace Laplace {

namespace {

bool fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

bool integer_array(const std::map<std::string, MetaValue>& metadata,
                   const std::string& key, std::vector<int>* output) {
    auto copy = [&](const auto* values) {
        if (!values) return false;
        output->assign(values->begin(), values->end());
        return true;
    };
    return copy(meta_as<MetaArrayU32>(metadata, key.c_str())) ||
           copy(meta_as<MetaArrayI32>(metadata, key.c_str())) ||
           copy(meta_as<MetaArrayU64>(metadata, key.c_str())) ||
           copy(meta_as<MetaArrayI64>(metadata, key.c_str()));
}

} // namespace

bool synthesize_topology(const GGUFContext& gguf, TopologyPlan* plan,
                         std::string* error) {
    if (!plan) return fail(error, "topology: null output plan");

    const auto& metadata = gguf.metadata();
    const std::string* architecture =
        meta_str(metadata, "general.architecture");
    if (!architecture || architecture->empty()) {
        return fail(error, "topology: missing general.architecture metadata namespace");
    }

    TopologyPlan candidate;
    candidate.metadata_namespace = *architecture + ".";
    const std::string& prefix = candidate.metadata_namespace;
    auto integer = [&](const char* suffix, int fallback = 0) {
        return static_cast<int>(
            meta_int(metadata, (prefix + suffix).c_str(), fallback));
    };
    auto real = [&](const char* suffix, double fallback = 0.0) {
        return static_cast<float>(
            meta_float(metadata, (prefix + suffix).c_str(), fallback));
    };

    const int layer_count = integer("block_count");
    candidate.hidden = integer("embedding_length");
    candidate.intermediate = integer("feed_forward_length");
    candidate.max_seq_len = integer("context_length");
    const int query_heads = integer("attention.head_count");
    const int global_dim = integer("attention.key_length");
    const int sliding_dim = integer("attention.key_length_swa", global_dim);
    const int window = integer("attention.sliding_window");

    if (layer_count <= 0 || candidate.hidden <= 0 ||
        candidate.intermediate <= 0 || candidate.max_seq_len <= 0 ||
        query_heads <= 0 || global_dim <= 0 || sliding_dim <= 0) {
        return fail(error, "topology: incomplete model or attention dimensions");
    }

    std::vector<int> kv_heads;
    std::vector<int> sliding_pattern;
    if (!integer_array(metadata, prefix + "attention.head_count_kv",
                       &kv_heads) ||
        static_cast<int>(kv_heads.size()) != layer_count) {
        return fail(error, "topology: attention.head_count_kv must have one entry per layer");
    }
    if (!integer_array(metadata, prefix + "attention.sliding_window_pattern",
                       &sliding_pattern) ||
        static_cast<int>(sliding_pattern.size()) != layer_count) {
        return fail(error,
                    "topology: attention.sliding_window_pattern must have one entry per layer");
    }

    candidate.n_experts = integer("expert_count");
    candidate.n_experts_used = integer("expert_used_count");
    candidate.expert_intermediate =
        integer("expert_feed_forward_length");
    if ((candidate.n_experts > 0) !=
        (candidate.n_experts_used > 0 &&
         candidate.expert_intermediate > 0)) {
        return fail(error, "topology: incomplete routed-expert dimensions");
    }
    if (candidate.n_experts_used > candidate.n_experts ||
        candidate.n_experts_used > 16) {
        return fail(error, "topology: routed expert count exceeds execution limit");
    }
    candidate.rms_eps =
        real("attention.layer_norm_rms_epsilon", 1e-6);
    candidate.logit_softcap = real("final_logit_softcapping");
    candidate.embed_scale = std::sqrt(static_cast<float>(candidate.hidden));

    const float global_rope = real("rope.freq_base", 1000000.0);
    const float sliding_rope = real("rope.freq_base_swa", 10000.0);
    const int global_rope_dim =
        integer("rope.dimension_count", global_dim);
    const int sliding_rope_dim =
        integer("rope.dimension_count_swa", sliding_dim);

    candidate.layers.resize(layer_count);
    for (int layer = 0; layer < layer_count; ++layer) {
        LayerTopology& output = candidate.layers[layer];
        const bool sliding = sliding_pattern[layer] != 0;
        output.n_q_heads = query_heads;
        output.n_kv_heads = kv_heads[layer];
        output.head_dim = sliding ? sliding_dim : global_dim;
        output.sliding_window = sliding ? window : 0;
        output.rope_dim = sliding ? sliding_rope_dim : global_rope_dim;
        output.rope_base = sliding ? sliding_rope : global_rope;
        output.moe = candidate.n_experts > 0;

        if (output.n_kv_heads <= 0 ||
            output.n_q_heads % output.n_kv_heads != 0 ||
            output.head_dim > 512 ||
            output.rope_dim <= 0 || output.rope_dim > output.head_dim ||
            (output.rope_dim & 1) != 0 ||
            output.rope_base <= 0.0f ||
            (sliding && output.sliding_window <= 0)) {
            return fail(error, "topology: invalid attention geometry at layer " +
                               std::to_string(layer));
        }
    }

    *plan = std::move(candidate);
    if (error) error->clear();
    return true;
}

std::vector<KVLayerConfig> make_kv_layer_configs(
        const TopologyPlan& plan, int max_seq_len, KVCacheMode mode) {
    std::vector<KVLayerConfig> output;
    if (max_seq_len <= 0) return output;
    output.reserve(plan.layers.size());
    for (const LayerTopology& layer : plan.layers) {
        KVLayerConfig config;
        config.n_kv_heads = layer.n_kv_heads;
        config.head_dim = layer.head_dim;
        config.sliding_window = layer.sliding_window;
        config.capacity = layer.sliding_window > 0
                        ? std::min(max_seq_len, layer.sliding_window)
                        : max_seq_len;
        config.mode = layer.sliding_window > 0 &&
                      (mode == KVCacheMode::LAPLACE ||
                       mode == KVCacheMode::LAPLACE_Q4)
                    ? KVCacheMode::FP16 : mode;
        output.push_back(config);
    }
    return output;
}

} // namespace Laplace
