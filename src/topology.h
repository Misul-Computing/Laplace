#pragma once

#include <string>
#include <vector>

#include "gguf.h"
#include "kvcache.h"

namespace Laplace {

struct LayerTopology {
    int n_q_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int intermediate = 0;
    int sliding_window = 0;
    int rope_dim = 0;
    float rope_base = 0.0f;
    bool shared_kv = false;
    bool moe = false;
    bool swiglu = false;
};

struct TopologyPlan {
    std::string metadata_namespace;
    int hidden = 0;
    int intermediate = 0;
    int max_seq_len = 0;
    int n_experts = 0;
    int n_experts_used = 0;
    int expert_intermediate = 0;
    float rms_eps = 1e-6f;
    float logit_softcap = 0.0f;
    float embed_scale = 1.0f;
    std::vector<LayerTopology> layers;
};

bool synthesize_topology(const GGUFContext&, TopologyPlan*, std::string* error);
std::vector<KVLayerConfig> make_kv_layer_configs(
    const TopologyPlan&, int max_seq_len, KVCacheMode mode);

} // namespace Laplace
