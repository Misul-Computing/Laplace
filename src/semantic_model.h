#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "artifact_set.h"
#include "compatibility_report.h"

namespace Laplace {

enum class DimensionKind : uint8_t { Constant = 1, Symbol = 2 };
enum class ScalarType : uint16_t { F32 = 1, F16 = 2, U32 = 3, I32 = 4, U8 = 5 };
enum class TensorRole : uint16_t {
    TokenEmbedding = 1, AttentionNormWeight = 2, QueryWeight = 3, QueryBias = 4,
    KeyWeight = 5, KeyBias = 6, ValueWeight = 7, ValueBias = 8,
    AttentionOutputWeight = 9, FfnNormWeight = 10, FfnGateWeight = 11,
    FfnUpWeight = 12, FfnDownWeight = 13, FinalNormWeight = 14, OutputWeight = 15,
    AttentionQueryGateWeight = 16, AttentionQueryNormWeight = 17, AttentionKeyNormWeight = 18,
    RecurrentQkvWeight = 19, RecurrentGateWeight = 20, RecurrentBetaWeight = 21,
    RecurrentAlphaWeight = 22, RecurrentConvWeight = 23, RecurrentDtBias = 24,
    RecurrentDecayWeight = 25, RecurrentNormWeight = 26, RecurrentOutputWeight = 27,
    NextnProjectionWeight = 28, NextnEmbeddingNormWeight = 29,
    NextnHiddenNormWeight = 30, NextnSharedHeadNormWeight = 31,
};
enum class PhysicalLayoutKind : uint16_t { ContiguousRowMajor = 1, GgufBlocked = 2 };
enum class PackingKind : uint16_t { None = 0, Gguf = 1 };
enum class QuantizationKind : uint16_t { None = 0, BlockedAffine = 1, Codebook = 2 };
enum class PlaneKind : uint16_t { Values = 1, Scales = 2, Biases = 3, Zeros = 4, Indexes = 5, LayoutMetadata = 6 };
enum class OperatorKind : uint16_t {
    EmbeddingLookup = 1, RmsNorm = 2, Linear = 3, Rope = 4, CausalAttention = 5, SwiGlu = 6, Add = 7,
    DepthwiseConvSilu = 8, GatedDeltaNet = 9, GatedAttention = 10, GatedRmsNorm = 11, L2Normalize = 12,
    AxisSplit = 13, Concat = 14,
    RouterTopK = 15, RoutedLinear = 16, GatedActivation = 17, WeightedExpertReduce = 18,
    Scale = 19, TanhSoftcap = 20,
};
enum class ConstraintKind : uint16_t { Equal = 1, ProductEqual = 2, Divisible = 3, ExactValue = 4, Unique = 5, TensorBytesEqual = 6 };
enum class ConstraintOperandKind : uint8_t { Dimension = 1, Tensor = 2, Value = 3, Constant = 4 };
enum class RopePairing : uint8_t { HalfSplit = 1, Interleaved = 2, MultiSectionHalfSplit = 3 };
enum class AttentionMask : uint8_t { Causal = 1 };
enum class ActivationKind : uint16_t { Silu = 1, GeluTanh = 2 };
enum class StateKind : uint16_t { KeyCache = 1, ValueCache = 2, RecurrentConvHistory = 3, RecurrentDeltaMatrix = 4 };
enum class StateUpdateKind : uint16_t { AppendKey = 1, AppendValue = 2, ShiftHistory = 3, DeltaMatrix = 4 };
enum class StateFormatKind : uint16_t { GlobalContiguous = 1, RecurrentContiguous = 2 };
enum class TransformDomain : uint16_t { Untransformed = 1, RopeApplied = 2 };
enum class AttentionWindowKind : uint8_t { Global = 1, Sliding = 2 };
enum class ValueSource : uint8_t { SeparateProjection = 1, KeyPreRope = 2, KeyPostRope = 3, KeyStateAlias = 4 };
enum class CodecKind : uint16_t { None = 0, Fp32 = 1 };
enum class CachePolicy : uint16_t { Global = 1, Recurrent = 2 };
enum class LayoutPolicy : uint16_t {
    TokenMajorContiguous = 1,
    ChannelMajorHistory = 2,
    ValueHeadOutputRowKeyColumn = 3,
    ValueHeadKeyRowOutputColumn = 4,
};
enum class PositionPolicy : uint16_t { AppendOnly = 1, ReplaceAtCursor = 2 };
enum class EntryKind : uint16_t { TokenIds = 1 };
enum class ExecutionPhase : uint16_t { Prefill = 1, Decode = 2, StateUpdate = 3, Rollback = 4, Output = 5 };
enum class Capability : uint16_t { ScalarFp32 = 1, GlobalFp32Kv = 2, CpuNeon = 3, CpuDotProd = 4, MetalDevice = 5, MetalLibrary = 6, MetalPipeline = 7, TransactionalState = 8 };
enum class FallbackKind : uint16_t { None = 0, ExactCpu = 1 };
enum class NumericalClass : uint16_t { ExactFp32 = 1 };

struct Dimension {
    DimensionKind kind = DimensionKind::Constant;
    uint64_t constant_or_symbol = 0;
    friend bool operator==(const Dimension&, const Dimension&) = default;
};

struct PhysicalLayout {
    PhysicalLayoutKind kind = PhysicalLayoutKind::ContiguousRowMajor;
    uint16_t version = 1;
    PackingKind packing = PackingKind::None;
    uint8_t rank = 0;
    uint8_t block_rank = 0;
    std::array<uint8_t, 8> axis_order = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    std::array<uint64_t, 8> strides{};
    uint32_t block_elements = 0;
    uint32_t block_bytes = 0;
    uint32_t flags = 0;
    friend bool operator==(const PhysicalLayout&, const PhysicalLayout&) = default;
};

struct Quantization {
    QuantizationKind kind = QuantizationKind::None;
    uint16_t version = 1;
    ScalarType accumulation_type = ScalarType::F32;
    ScalarType scale_type = static_cast<ScalarType>(0);
    ScalarType zero_type = static_cast<ScalarType>(0);
    ScalarType bias_type = static_cast<ScalarType>(0);
    uint32_t block_elements = 0;
    uint32_t block_bytes = 0;
    uint32_t group_size = 0;
    uint32_t required_plane_mask = 0;
    uint32_t flags = 0;
    friend bool operator==(const Quantization&, const Quantization&) = default;
};

enum class ExpertAxisKind : uint8_t { None = 0, ExpertBank = 1 };

struct ExpertAxis {
    ExpertAxisKind kind = ExpertAxisKind::None;
    uint8_t expert_axis = 0xff;
    uint8_t member_axis = 0xff;
    uint8_t input_axis = 0xff;
    uint8_t output_axis = 0xff;
    uint32_t expert_count = 0;
    uint64_t per_expert_byte_stride = 0;
    uint32_t flags = 0;
    friend bool operator==(const ExpertAxis&, const ExpertAxis&) = default;
};

struct TensorPlane {
    PlaneKind kind = PlaneKind::Values;
    ScalarType storage_type = ScalarType::F32;
    ArtifactId artifact_id{};
    uint64_t offset = 0;
    uint64_t length = 0;
    uint32_t alignment = 0;
    uint32_t flags = 0;
    friend bool operator==(const TensorPlane&, const TensorPlane&) = default;
};

struct SemanticTensor {
    uint32_t id = 0;
    TensorRole role = TensorRole::TokenEmbedding;
    ScalarType logical_type = ScalarType::F32;
    std::vector<Dimension> dimensions;
    PhysicalLayout layout;
    Quantization quantization;
    ExpertAxis expert_axis;
    std::vector<TensorPlane> planes;
    uint16_t flags = 0;
    friend bool operator==(const SemanticTensor&, const SemanticTensor&) = default;
};

struct SemanticValue {
    uint32_t id = 0;
    ScalarType logical_type = ScalarType::F32;
    std::vector<Dimension> dimensions;
    uint8_t flags = 0;
    friend bool operator==(const SemanticValue&, const SemanticValue&) = default;
};

struct EmbeddingLookupPayload {
    uint32_t scale_f32_bits = 0;
    uint32_t vocabulary = 0;
    uint32_t width = 0;
    uint32_t flags = 0;
    friend bool operator==(const EmbeddingLookupPayload&, const EmbeddingLookupPayload&) = default;
};

struct RmsNormPayload {
    uint32_t epsilon_f32_bits = 0;
    int32_t axis = -1;
    uint8_t weight_mode = 1;
    friend bool operator==(const RmsNormPayload&, const RmsNormPayload&) = default;
};

struct LinearPayload {
    bool transpose_weight = false;
    bool has_bias = false;
    ScalarType accumulation_type = ScalarType::F32;
    friend bool operator==(const LinearPayload&, const LinearPayload&) = default;
};

struct RopePayload {
    RopePairing pairing = RopePairing::HalfSplit;
    bool position_from_cursor = true;
    uint32_t rotary_dimension = 0;
    uint32_t base_f32_bits = 0;
    uint32_t scale_f32_bits = 0;
    std::array<uint32_t, 4> position_sections{};
    friend bool operator==(const RopePayload&, const RopePayload&) = default;
};

struct CausalAttentionPayload {
    uint32_t query_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dimension = 0;
    uint32_t scale_f32_bits = 0;
    AttentionMask mask = AttentionMask::Causal;
    CachePolicy cache_policy = CachePolicy::Global;
    AttentionWindowKind window = AttentionWindowKind::Global;
    uint32_t window_tokens = 0;
    ValueSource value_source = ValueSource::SeparateProjection;
    uint32_t value_source_value = UINT32_MAX;
    friend bool operator==(const CausalAttentionPayload&, const CausalAttentionPayload&) = default;
};

struct SwiGluPayload {
    ActivationKind activation = ActivationKind::Silu;
    friend bool operator==(const SwiGluPayload&, const SwiGluPayload&) = default;
};

struct AddPayload {
    friend bool operator==(const AddPayload&, const AddPayload&) = default;
};

struct DepthwiseConvSiluPayload {
    uint32_t qk_heads = 0;
    uint32_t value_heads = 0;
    uint32_t head_dimension = 0;
    uint32_t kernel = 0;
    friend bool operator==(const DepthwiseConvSiluPayload&, const DepthwiseConvSiluPayload&) = default;
};

enum class QkHeadMapping : uint8_t { ValueHeadModulo = 1 };
enum class BetaTransform : uint8_t { Sigmoid = 1 };
enum class DecayTransform : uint8_t { NegativeSoftplus = 1 };
enum class DeltaStateLayout : uint8_t {
    ValueHeadOutputRowKeyColumn = 1,
    ValueHeadKeyRowOutputColumn = 2,
};

struct GatedDeltaNetPayload {
    uint32_t qk_heads = 0;
    uint32_t value_heads = 0;
    uint32_t head_dimension = 0;
    QkHeadMapping qk_mapping = QkHeadMapping::ValueHeadModulo;
    BetaTransform beta_transform = BetaTransform::Sigmoid;
    DecayTransform decay_transform = DecayTransform::NegativeSoftplus;
    DeltaStateLayout state_layout = DeltaStateLayout::ValueHeadOutputRowKeyColumn;
    uint32_t flags = 0;
    friend bool operator==(const GatedDeltaNetPayload&, const GatedDeltaNetPayload&) = default;
};

struct GatedAttentionPayload {
    friend bool operator==(const GatedAttentionPayload&, const GatedAttentionPayload&) = default;
};

struct GatedRmsNormPayload {
    uint32_t epsilon_f32_bits = 0;
    ActivationKind gate_activation = ActivationKind::Silu;
    uint8_t weight_mode = 1;
    friend bool operator==(const GatedRmsNormPayload&, const GatedRmsNormPayload&) = default;
};

struct L2NormalizePayload {
    uint32_t epsilon_f32_bits = 0;
    friend bool operator==(const L2NormalizePayload&, const L2NormalizePayload&) = default;
};

struct AxisSplitPayload {
    uint32_t first_width = 0;
    uint32_t second_width = 0;
    friend bool operator==(const AxisSplitPayload&, const AxisSplitPayload&) = default;
};

struct ConcatPayload {
    int32_t axis = -1;
    friend bool operator==(const ConcatPayload&, const ConcatPayload&) = default;
};

enum class RouterScoreDomain : uint8_t { Logits = 1, Probabilities = 2 };
enum class RouterNormalizationOrder : uint8_t { SelectThenNormalize = 1, NormalizeThenSelect = 2 };
enum class SelectedWeightNormalization : uint8_t { Softmax = 1, PreserveSource = 2 };
enum class RouterTiePolicy : uint8_t { LowestExpertId = 1 };
enum class RouterWeightSource : uint8_t { SelectedNormalizedScore = 1 };

struct RouterTopKPayload {
    uint32_t expert_count = 0;
    uint32_t selected_count = 0;
    RouterScoreDomain score_domain = RouterScoreDomain::Logits;
    RouterNormalizationOrder normalization_order = RouterNormalizationOrder::SelectThenNormalize;
    SelectedWeightNormalization selected_weight_normalization = SelectedWeightNormalization::Softmax;
    RouterTiePolicy tie_policy = RouterTiePolicy::LowestExpertId;
    RouterWeightSource weight_source = RouterWeightSource::SelectedNormalizedScore;
    uint32_t flags = 0;
    friend bool operator==(const RouterTopKPayload&, const RouterTopKPayload&) = default;
};

struct RoutedLinearPayload {
    ScalarType accumulation_type = ScalarType::F32;
    friend bool operator==(const RoutedLinearPayload&, const RoutedLinearPayload&) = default;
};

struct GatedActivationPayload {
    ActivationKind activation = ActivationKind::Silu;
    friend bool operator==(const GatedActivationPayload&, const GatedActivationPayload&) = default;
};

enum class ExpertReduceAssociation : uint8_t { SelectedOrderLeftToRight = 1 };
enum class ExpertScaleSource : uint8_t { None = 1, PerExpertTensor = 2 };

struct WeightedExpertReducePayload {
    ExpertReduceAssociation association = ExpertReduceAssociation::SelectedOrderLeftToRight;
    ExpertScaleSource scale_source = ExpertScaleSource::None;
    ScalarType accumulation_type = ScalarType::F32;
    friend bool operator==(const WeightedExpertReducePayload&, const WeightedExpertReducePayload&) = default;
};

enum class ScaleSource : uint8_t { LiteralF32 = 1, Tensor = 2 };

struct ScalePayload {
    ScaleSource source = ScaleSource::LiteralF32;
    uint32_t literal_f32_bits = 0;
    friend bool operator==(const ScalePayload&, const ScalePayload&) = default;
};

struct TanhSoftcapPayload {
    uint32_t cap_f32_bits = 0;
    friend bool operator==(const TanhSoftcapPayload&, const TanhSoftcapPayload&) = default;
};

using OperatorPayload = std::variant<EmbeddingLookupPayload, RmsNormPayload, LinearPayload,
                                     RopePayload, CausalAttentionPayload, SwiGluPayload, AddPayload,
                                     DepthwiseConvSiluPayload, GatedDeltaNetPayload,
                                     GatedAttentionPayload, GatedRmsNormPayload, L2NormalizePayload,
                                     AxisSplitPayload, ConcatPayload, RouterTopKPayload,
                                     RoutedLinearPayload, GatedActivationPayload,
                                     WeightedExpertReducePayload, ScalePayload, TanhSoftcapPayload>;

struct SemanticOperator {
    uint32_t id = 0;
    OperatorKind kind = OperatorKind::EmbeddingLookup;
    uint16_t semantic_version = 1;
    std::vector<uint32_t> inputs;
    std::vector<uint32_t> outputs;
    std::vector<uint32_t> tensors;
    std::vector<uint32_t> states;
    OperatorPayload payload = AddPayload{};
    friend bool operator==(const SemanticOperator&, const SemanticOperator&) = default;
};

struct SemanticLayer {
    uint32_t layer_index = 0;
    uint32_t first_operator = 0;
    uint32_t operator_count = 0;
    uint32_t flags = 0;
    friend bool operator==(const SemanticLayer&, const SemanticLayer&) = default;
};

constexpr uint32_t kSemanticLayerFlagSpeculative = 1;

struct StateFormat {
    StateFormatKind kind = StateFormatKind::GlobalContiguous;
    uint16_t version = 1;
    ScalarType logical_type = ScalarType::F32;
    ScalarType encoded_type = ScalarType::F32;
    TransformDomain logical_domain = TransformDomain::Untransformed;
    TransformDomain encoded_domain = TransformDomain::Untransformed;
    CodecKind codec = CodecKind::Fp32;
    CachePolicy cache_policy = CachePolicy::Global;
    LayoutPolicy layout_policy = LayoutPolicy::TokenMajorContiguous;
    uint16_t flags = 0;
    uint32_t tile_tokens = 0;
    uint32_t mutable_tokens = 0;
    uint32_t alignment = 0;
    uint32_t reserved = 0;
    friend bool operator==(const StateFormat&, const StateFormat&) = default;
};

struct SemanticState {
    uint32_t id = 0;
    StateKind kind = StateKind::KeyCache;
    uint16_t semantic_version = 1;
    StateUpdateKind update_kind = StateUpdateKind::AppendKey;
    PositionPolicy position_policy = PositionPolicy::AppendOnly;
    std::vector<Dimension> dimensions;
    std::vector<StateFormat> formats;
    uint16_t flags = 0;
    friend bool operator==(const SemanticState&, const SemanticState&) = default;
};

struct SemanticConstraint {
    ConstraintKind kind = ConstraintKind::Equal;
    ConstraintOperandKind lhs_kind = ConstraintOperandKind::Dimension;
    ConstraintOperandKind rhs_kind = ConstraintOperandKind::Dimension;
    uint32_t lhs_id = 0;
    uint32_t rhs_id = 0;
    uint32_t lhs_axis = 0;
    uint32_t rhs_axis = 0;
    uint64_t constant = 0;
    uint64_t divisor = 0;
    friend bool operator==(const SemanticConstraint&, const SemanticConstraint&) = default;
};

struct CapabilityRequirement {
    Capability capability = Capability::ScalarFp32;
    uint16_t minimum_version = 0;
    uint32_t flags = 0;
    friend bool operator==(const CapabilityRequirement&, const CapabilityRequirement&) = default;
};

struct SemanticFallback {
    FallbackKind kind = FallbackKind::None;
    ExecutionPhase phase = ExecutionPhase::Prefill;
    NumericalClass numerical_class = NumericalClass::ExactFp32;
    uint16_t flags = 0;
    friend bool operator==(const SemanticFallback&, const SemanticFallback&) = default;
};

struct SemanticModel {
    uint16_t schema_major = 1;
    uint16_t schema_minor = 0;
    uint16_t opset_major = 1;
    uint16_t opset_minor = 0;
    uint32_t maximum_context = 0;
    EntryKind entry_kind = EntryKind::TokenIds;
    uint32_t vocabulary_size = 0;
    uint32_t bos_id = UINT32_MAX;
    uint32_t eos_id = UINT32_MAX;
    std::vector<uint32_t> stop_ids;
    std::array<uint8_t, 32> tokenizer_digest{};
    std::array<uint8_t, 32> template_digest{};
    uint32_t input_values_first = 0;
    uint32_t input_values_count = 0;
    uint32_t output_values_first = 0;
    uint32_t output_values_count = 0;
    std::vector<SemanticTensor> tensors;
    std::vector<SemanticValue> values;
    std::vector<SemanticOperator> operators;
    std::vector<SemanticLayer> layers;
    std::vector<SemanticState> states;
    std::vector<SemanticConstraint> constraints;
    std::vector<CapabilityRequirement> capabilities;
    std::vector<SemanticFallback> fallbacks;
};

using SemanticEncodeResult = std::variant<std::vector<uint8_t>, CompatibilityReport>;
using SemanticDecodeResult = std::variant<SemanticModel, CompatibilityReport>;

SemanticEncodeResult encode_semantic_model(const SemanticModel& model);
SemanticDecodeResult decode_semantic_model(const std::vector<uint8_t>& bytes);
Sha256Digest semantic_model_digest(const SemanticModel& model);

using SemanticVectorResult = std::variant<std::vector<float>, CompatibilityReport>;
using SemanticAxisSplit = std::pair<std::vector<float>, std::vector<float>>;
using SemanticAxisSplitResult = std::variant<SemanticAxisSplit, CompatibilityReport>;

struct SemanticRouterResult {
    std::vector<uint32_t> ids;
    std::vector<float> weights;
    friend bool operator==(const SemanticRouterResult&, const SemanticRouterResult&) = default;
};

using SemanticRouterResultOrError = std::variant<SemanticRouterResult, CompatibilityReport>;

struct SemanticKvState {
    std::vector<float> key;
    std::vector<float> value;
    uint32_t tokens = 0;
};

struct SemanticConvState {
    std::vector<float> history;
};

struct SemanticGatedDeltaState {
    // Value-head-major, then output row, then key column.
    std::vector<float> matrix;
};

SemanticVectorResult semantic_embedding_lookup(const std::vector<float>& table,
                                               uint32_t vocabulary, uint32_t width,
                                               const std::vector<uint32_t>& token_ids,
                                               float scale);
SemanticVectorResult semantic_rms_norm(const std::vector<float>& input,
                                       const std::vector<float>& weight, float epsilon);
SemanticVectorResult semantic_linear(const std::vector<float>& input,
                                     const std::vector<float>& weight,
                                     uint32_t output_width, uint32_t input_width,
                                     const std::vector<float>& bias);
SemanticVectorResult semantic_rope_half_split(const std::vector<float>& query,
                                              const std::vector<float>& key,
                                              uint32_t position, uint32_t rotary_dimension,
                                              float base, float scale);
SemanticVectorResult semantic_rope_interleaved(const std::vector<float>& query,
                                                const std::vector<float>& key,
                                                uint32_t position, uint32_t rotary_dimension,
                                                float base, float scale);
SemanticVectorResult semantic_rope_multi_section_half_split(const std::vector<float>& query,
                                                             const std::vector<float>& key,
                                                             const std::array<uint32_t, 4>& positions,
                                                             const std::array<uint32_t, 4>& sections,
                                                             uint32_t rotary_dimension, float base, float scale);
SemanticVectorResult semantic_depthwise_conv1d_silu_step(const std::vector<float>& input,
                                                          const std::vector<float>& weight,
                                                          uint32_t channels, uint32_t kernel,
                                                          SemanticConvState& state);
SemanticVectorResult semantic_gated_delta_net_step(const std::vector<float>& query,
                                                    const std::vector<float>& key,
                                                    const std::vector<float>& value,
                                                    const std::vector<float>& log_decay,
                                                    const std::vector<float>& beta,
                                                    uint32_t qk_heads, uint32_t value_heads,
                                                    uint32_t head_dimension,
                                                    SemanticGatedDeltaState& state);
SemanticVectorResult semantic_gated_attention_output(const std::vector<float>& attention,
                                                      const std::vector<float>& gate);
SemanticVectorResult semantic_gated_rms_norm(const std::vector<float>& input,
                                             const std::vector<float>& weight,
                                             const std::vector<float>& gate, float epsilon);
SemanticVectorResult semantic_l2_normalize(const std::vector<float>& input, float epsilon);
SemanticAxisSplitResult semantic_axis_split(const std::vector<float>& input,
                                            uint32_t row_width, uint32_t first_width);
SemanticVectorResult semantic_concat_last_axis(const std::vector<float>& left,
                                               const std::vector<float>& right,
                                               uint32_t left_width, uint32_t right_width);
SemanticVectorResult semantic_causal_attention(const std::vector<float>& query,
                                               const std::vector<float>& key,
                                               const std::vector<float>& value,
                                               uint32_t rows, uint32_t query_heads,
                                               uint32_t kv_heads, uint32_t head_dimension,
                                               float scale, SemanticKvState& state);
SemanticVectorResult semantic_causal_attention_windowed(const std::vector<float>& query,
                                                        const std::vector<float>& key,
                                                        const std::vector<float>& value,
                                                        uint32_t rows, uint32_t query_heads,
                                                        uint32_t kv_heads, uint32_t head_dimension,
                                                        float scale, AttentionWindowKind window,
                                                        uint32_t window_tokens, SemanticKvState& state);
SemanticVectorResult semantic_swiglu(const std::vector<float>& gate,
                                     const std::vector<float>& up);
SemanticRouterResultOrError semantic_router_top_k(const std::vector<float>& scores,
                                                   const RouterTopKPayload& payload);
SemanticVectorResult semantic_gated_activation(const std::vector<float>& gate,
                                               const std::vector<float>& up,
                                               ActivationKind activation);
SemanticVectorResult semantic_weighted_expert_reduce(const std::vector<float>& expert_outputs,
                                                      const std::vector<float>& route_weights,
                                                      const std::vector<float>& expert_scales,
                                                      uint32_t output_width,
                                                      const WeightedExpertReducePayload& payload);
SemanticVectorResult semantic_scale(const std::vector<float>& input, float scale);
SemanticVectorResult semantic_tanh_softcap(const std::vector<float>& input, float cap);
SemanticVectorResult semantic_add(const std::vector<float>& lhs,
                                  const std::vector<float>& rhs);

} // namespace Laplace
