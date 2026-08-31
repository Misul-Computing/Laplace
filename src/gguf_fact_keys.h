#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include "compatibility_report.h"

namespace Laplace::gguf_fact_keys {

// These IDs are the adapter's fixed, source-independent vocabulary. The
// metadata spelling before each suffix is GGUF-specific diagnostic input.
enum class ValueKind : uint8_t {
    Unsigned = 1,
    UnsignedOrVector = 2,
    Float32 = 3,
};

struct Descriptor {
    CanonicalFactKey key;
    std::string_view suffix;
    ValueKind value_kind;
};

inline constexpr CanonicalFactKey block_count{0x47554601u};
inline constexpr CanonicalFactKey context_length{0x47554602u};
inline constexpr CanonicalFactKey embedding_length{0x47554603u};
inline constexpr CanonicalFactKey feed_forward_length{0x47554604u};
inline constexpr CanonicalFactKey attention_head_count{0x47554605u};
inline constexpr CanonicalFactKey attention_head_count_kv{0x47554606u};
inline constexpr CanonicalFactKey attention_key_length{0x47554607u};
inline constexpr CanonicalFactKey attention_value_length{0x47554608u};
inline constexpr CanonicalFactKey rope_dimension_count{0x47554609u};
inline constexpr CanonicalFactKey rope_freq_base{0x4755460au};
inline constexpr CanonicalFactKey attention_layer_norm_rms_epsilon{0x4755460bu};
inline constexpr CanonicalFactKey bos_token_id{0x4755460cu};
inline constexpr CanonicalFactKey eos_token_id{0x4755460du};
inline constexpr CanonicalFactKey expert_count{0x4755460eu};
inline constexpr CanonicalFactKey expert_used_count{0x4755460fu};
inline constexpr CanonicalFactKey expert_feed_forward_length{0x47554610u};

inline constexpr std::array<Descriptor, 16> descriptors = {{
    {block_count, "block_count", ValueKind::Unsigned},
    {context_length, "context_length", ValueKind::Unsigned},
    {embedding_length, "embedding_length", ValueKind::Unsigned},
    {feed_forward_length, "feed_forward_length", ValueKind::Unsigned},
    {attention_head_count, "attention.head_count", ValueKind::Unsigned},
    {attention_head_count_kv, "attention.head_count_kv", ValueKind::UnsignedOrVector},
    {attention_key_length, "attention.key_length", ValueKind::UnsignedOrVector},
    {attention_value_length, "attention.value_length", ValueKind::UnsignedOrVector},
    {rope_dimension_count, "rope.dimension_count", ValueKind::UnsignedOrVector},
    {rope_freq_base, "rope.freq_base", ValueKind::Float32},
    {attention_layer_norm_rms_epsilon, "attention.layer_norm_rms_epsilon", ValueKind::Float32},
    {bos_token_id, "ggml.bos_token_id", ValueKind::Unsigned},
    {eos_token_id, "ggml.eos_token_id", ValueKind::Unsigned},
    {expert_count, "expert_count", ValueKind::Unsigned},
    {expert_used_count, "expert_used_count", ValueKind::Unsigned},
    {expert_feed_forward_length, "expert_feed_forward_length", ValueKind::Unsigned},
}};

} // namespace Laplace::gguf_fact_keys
