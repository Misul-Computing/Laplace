#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "compatibility_report.h"
#include "semantic_dispatch_program.h"
#include "physical_codec.h"
#include "semantic_model.h"

namespace Laplace {

enum class RuntimeObjective : uint16_t { Latency = 1, Throughput = 2 };
enum class BackendWriter : uint16_t { Cpu = 1, Metal = 2 };
enum class ScratchLifetime : uint16_t { None = 0, Phase = 1, Session = 2 };
// This is an executable identity, not a diagnostic label. Scalar identities
// remain qualification/reference-test execution only.
enum class KernelImplementation : uint16_t {
    Unavailable = 0,
    ScalarEmbedding = 1,
    ScalarRmsNorm = 2,
    ScalarLinear = 3,
    ScalarRope = 4,
    ScalarCausalAttention = 5,
    ScalarSwiGlu = 6,
    ScalarAdd = 7,
    MetalDenseToken = 8,
    MetalRecurrentToken = 9,
    MetalDensePrefillBatch = 10,
#if defined(LAPLACE_TESTING)
    ScalarEmbeddingUnitOffset = 255,
#endif
};

enum MetalWeightFormat : uint32_t {
    MetalWeightFormatF16 = 1u << 0,
    MetalWeightFormatQ4K = 1u << 1,
    MetalWeightFormatQ5_0 = 1u << 2,
    MetalWeightFormatQ6K = 1u << 3,
    MetalWeightFormatQ8_0 = 1u << 4,
    MetalWeightFormatQ2K = 1u << 5,
    MetalWeightFormatIQ2XXS = 1u << 6,
    MetalWeightFormatQ4_0 = 1u << 7,
    // Source-neutral three-plane grouped-affine storage: packed UInt2 values
    // plus FP16 scales and FP16 biases, one 256-value group per row/block.
    MetalWeightFormatAffineUInt2_256 = 1u << 8,
    // Output-block-major UInt2 skip storage. This has a distinct group order
    // from MetalWeightFormatAffineUInt2_256 and is never interchangeable.
    MetalWeightFormatColumnGroupedAffineUInt2Skip256 = 1u << 9,
};

template<class T>
struct ExactOrAny {
    bool any = false;
    T exact{};
};

template<class T>
constexpr ExactOrAny<T> exact(T value) {
    return {false, value};
}

template<class T>
constexpr ExactOrAny<T> any() {
    return {true, {}};
}

template<class T>
struct ClosedRange {
    T minimum = 0;
    T maximum = 0;

    constexpr bool valid() const { return minimum <= maximum; }
    constexpr bool contains(T value) const { return valid() && minimum <= value && value <= maximum; }
};

struct RuntimeCapabilities {
    bool scalar_fp32 = false;
    bool global_fp32_kv = false;
    bool transactional_state = false;
    bool cpu_neon = false;
    bool cpu_dotprod = false;
    bool metal_device = false;
    bool metal_library = false;
    bool metal_pipeline = false;
    bool metal_affine_u2_256 = false;
    bool metal_column_grouped_affine_u2_skip_256 = false;
    bool metal_moe_router_topk = false;
    bool metal_moe_gate_up = false;
    bool metal_moe_down_q5_0 = false;
    bool metal_moe_down_q8_0 = false;
    bool metal_moe_reduce = false;
    // Queried device capability for a device-resident routed-expert worklist.
    // This is independent of any particular model or codec.
    bool metal_moe_worklist = false;
};

// The canonical MoE admission witness.  Every consumer of the native MoE
// leaf uses this edge set; no consumer is allowed to infer meaning from
// serialized operator positions.
struct CanonicalMoeOperatorEdges {
    // Exact layer witness, in the serialized layer order.  The order is not
    // semantic; the set must contain every operator admitted by the matcher.
    std::vector<uint32_t> covered_operator_ids;
    // Dense attention/FFN nodes are part of the same witness.  Keeping these
    // pointers here makes the lowerer consume the match result instead of
    // rediscovering a second, potentially different graph.
    const SemanticOperator* dense_attn_norm = nullptr;
    const SemanticOperator* dense_query = nullptr;
    const SemanticOperator* dense_query_gate = nullptr;
    const SemanticOperator* dense_key = nullptr;
    const SemanticOperator* dense_value = nullptr;
    const SemanticOperator* dense_output = nullptr;
    const SemanticOperator* dense_query_norm = nullptr;
    const SemanticOperator* dense_key_norm = nullptr;
    const SemanticOperator* dense_ffn_norm = nullptr;
    const SemanticOperator* dense_gate = nullptr;
    const SemanticOperator* dense_up = nullptr;
    const SemanticOperator* dense_down = nullptr;
    const SemanticOperator* rope = nullptr;
    const SemanticOperator* swiglu = nullptr;
    const SemanticOperator* attention = nullptr;
    const SemanticOperator* router = nullptr;
    const SemanticOperator* router_linear = nullptr;
    const SemanticOperator* router_scale = nullptr;
    const SemanticOperator* router_normalization_scale = nullptr;
    const SemanticOperator* router_norm = nullptr;
    const SemanticOperator* expert_norm = nullptr;
    const SemanticOperator* expert_up = nullptr;
    const SemanticOperator* expert_split = nullptr;
    const SemanticOperator* expert_activation = nullptr;
    const SemanticOperator* expert_down = nullptr;
    const SemanticOperator* expert_reduce = nullptr;
    const SemanticOperator* attention_residual = nullptr;
    // Branch merge (dense output, MoE output) before the final residual add.
    const SemanticOperator* residual = nullptr;
    const SemanticOperator* final_add = nullptr;
    const SemanticOperator* dense_post_norm = nullptr;
    const SemanticOperator* moe_post_norm = nullptr;
    const SemanticOperator* output_post_norm = nullptr;
    const SemanticOperator* output_scale = nullptr;
    const SemanticTensor* gate_up_tensor = nullptr;
    const SemanticTensor* down_tensor = nullptr;
    uint32_t hidden = 0;
    uint32_t intermediate = 0;
    uint32_t expert_count = 0;
    uint32_t selected_count = 0;
    uint32_t gate_up_format = 0;
    uint32_t down_format = 0;
    PhysicalLayoutKind gate_up_layout = static_cast<PhysicalLayoutKind>(0);
    PhysicalLayoutKind down_layout = static_cast<PhysicalLayoutKind>(0);
    QuantizationKind gate_up_quantization = static_cast<QuantizationKind>(0);
    QuantizationKind down_quantization = static_cast<QuantizationKind>(0);
    uint32_t gate_up_plane_mask = 0;
    uint32_t down_plane_mask = 0;
    uint32_t gate_up_block_elements = 0;
    uint32_t gate_up_block_bytes = 0;
    uint32_t down_block_elements = 0;
    uint32_t down_block_bytes = 0;
    uint64_t gate_up_expert_stride = 0;
    uint64_t down_expert_stride = 0;
    ActivationKind activation = ActivationKind::Silu;
    ExpertScaleSource scale_source = ExpertScaleSource::None;
    ValueSource value_source = ValueSource::SeparateProjection;
    uint32_t router_normalization_scale_bits = 0;
};

// Exact witness for the dense attention/FFN subgraph executed by the native
// token leaf.  The serialized order is only an addressing detail; consumers
// use the typed operator pointers and covered ids captured by the matcher.
struct DenseGraphWitness {
    std::vector<uint32_t> covered_operator_ids;
    const SemanticOperator* attention_norm = nullptr;
    const SemanticOperator* query = nullptr;
    const SemanticOperator* query_gate = nullptr;
    const SemanticOperator* key = nullptr;
    const SemanticOperator* value = nullptr;
    const SemanticOperator* query_norm = nullptr;
    const SemanticOperator* key_norm = nullptr;
    const SemanticOperator* rope = nullptr;
    const SemanticOperator* attention = nullptr;
    const SemanticOperator* attention_output = nullptr;
    const SemanticOperator* attention_residual = nullptr;
    const SemanticOperator* ffn_norm = nullptr;
    const SemanticOperator* gate = nullptr;
    const SemanticOperator* up = nullptr;
    const SemanticOperator* swiglu = nullptr;
    const SemanticOperator* down = nullptr;
    const SemanticOperator* final_residual = nullptr;
    const SemanticOperator* query_split = nullptr;
    const SemanticOperator* gated_attention = nullptr;
};

// Match the complete dense graph consumed by the canonical Metal token
// program.  All values, producers, edges, and layer coverage are checked;
// unsupported or ambiguous graphs fail closed.
bool match_canonical_dense_operator_edges(const SemanticModel& model,
                                          const SemanticLayer& layer,
                                          DenseGraphWitness& witness);

// Matches the complete typed MoE edge contract for one semantic layer.
// Admission is closed on ambiguity, missing edges, unsupported activation,
// layouts, planes, quantization, or runtime-visible value/scale sources.
bool match_canonical_moe_operator_edges(const SemanticModel& model,
                                        const SemanticLayer& layer,
                                        CanonicalMoeOperatorEdges& edges);

struct SessionRequest {
    uint64_t max_context = 0;
    // Maximum rows evaluated in parallel by one kernel dispatch. A prompt is a
    // sequence of rows and may contain more tokens than this value.
    uint32_t max_batch = 0;
    uint64_t memory_limit = 0;
    bool enable_prefill = false;
    bool enable_decode = false;
    bool enable_streaming = false;
    bool enable_speculation = false;
    NumericalClass minimum_class = NumericalClass::ExactFp32;
    RuntimeObjective objective = RuntimeObjective::Latency;
};

// A source-neutral routed expert worklist contract.  The ids are package-local
// binding addresses only; matching is driven by the typed graph/physical
// contract below, never by a model name, tensor spelling, or artifact hash.
struct MoeWorklistTensorContract {
    uint32_t input_width = 0;
    uint32_t output_width = 0;
    uint32_t expert_count = 0;
    std::vector<Dimension> dimensions;
    std::array<uint64_t, 8> strides{};
    PhysicalLayoutSchema layout;
    PhysicalQuantizationSchema quantization;
    std::vector<PhysicalPlaneSchema> planes;
    uint64_t expert_stride = 0;
    friend bool operator==(const MoeWorklistTensorContract&,
                           const MoeWorklistTensorContract&) = default;
};

struct MoeWorklistDescriptor {
    uint16_t version = 1;
    ExecutionPhase phase = ExecutionPhase::Decode;
    // [router, gate/up, activation, down, reduce].  These are local graph
    // addresses supplied to the future session binder.
    std::array<uint32_t, 5> semantic_operator_ids = {
        UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    uint32_t expert_count = 0;
    uint32_t selected_count = 0;
    uint32_t hidden = 0;
    uint32_t intermediate = 0;
    ActivationKind activation = ActivationKind::Silu;
    ExpertScaleSource scale_source = ExpertScaleSource::None;
    ValueSource value_source = ValueSource::SeparateProjection;
    MoeWorklistTensorContract gate_up;
    MoeWorklistTensorContract down;
    // [gate, up, down], in the same order as the future Metal bindings.
    std::vector<PhysicalCodecIdentity> physical_codecs;
    uint32_t router_scale_tensor_id = UINT32_MAX;
    uint32_t expert_norm_tensor_id = UINT32_MAX;
    uint32_t reduce_scale_tensor_id = UINT32_MAX;
    uint32_t post_norm_tensor_id = UINT32_MAX;
    uint32_t output_scale_tensor_id = UINT32_MAX;
    friend bool operator==(const MoeWorklistDescriptor&,
                           const MoeWorklistDescriptor&) = default;
};

using MoeWorklistSelection = std::variant<MoeWorklistDescriptor, CompatibilityReport>;

bool valid_moe_worklist_descriptor(const MoeWorklistDescriptor& descriptor);
bool moe_worklist_matches(const MoeWorklistDescriptor& query,
                          const MoeWorklistDescriptor& candidate);
MoeWorklistSelection select_moe_worklist(
    const MoeWorklistDescriptor& query, const RuntimeCapabilities& capabilities,
    const std::vector<MoeWorklistDescriptor>& registry);

struct KernelQuery {
    OperatorKind operation = OperatorKind::EmbeddingLookup;
    uint16_t semantic_version = 1;
    ExecutionPhase phase = ExecutionPhase::Prefill;
    ScalarType logical_type = ScalarType::F32;
    ScalarType storage_type = ScalarType::F32;
    PhysicalLayoutKind layout = PhysicalLayoutKind::ContiguousRowMajor;
    QuantizationKind quantization = QuantizationKind::None;
    StateKind state_kind = StateKind::KeyCache;
    StateFormatKind state_format = StateFormatKind::GlobalContiguous;
    uint32_t rank = 0;
    uint32_t alignment = 0;
    uint32_t head_dimension = 0;
    uint32_t batch_rows = 1;
    uint32_t tile_tokens = 0;
    uint32_t block_elements = 0;
    uint32_t block_bytes = 0;
    std::vector<PhysicalCodecIdentity> physical_codecs;
    uint32_t metal_weight_format_mask = 0;
    bool metal_affine_u2_256 = false;
    bool metal_column_grouped_affine_u2_skip_256 = false;
    bool metal_dense_token_pattern = false;
    bool metal_moe_token_pattern = false;
    bool metal_dense_prefill_batch_pattern = false;
    bool metal_recurrent_token_pattern = false;
    uint32_t moe_expert_count = 0;
    uint32_t moe_selected_count = 0;
    uint32_t moe_gate_up_format = 0;
    uint32_t moe_down_format = 0;
    PhysicalLayoutKind moe_gate_up_layout = static_cast<PhysicalLayoutKind>(0);
    PhysicalLayoutKind moe_down_layout = static_cast<PhysicalLayoutKind>(0);
    QuantizationKind moe_gate_up_quantization = static_cast<QuantizationKind>(0);
    QuantizationKind moe_down_quantization = static_cast<QuantizationKind>(0);
    uint32_t moe_gate_up_plane_mask = 0;
    uint32_t moe_down_plane_mask = 0;
    uint32_t moe_gate_up_block_elements = 0;
    uint32_t moe_gate_up_block_bytes = 0;
    uint32_t moe_down_block_elements = 0;
    uint32_t moe_down_block_bytes = 0;
    uint32_t moe_gate_up_input = 0;
    uint32_t moe_gate_up_output = 0;
    uint32_t moe_down_input = 0;
    uint32_t moe_down_output = 0;
    uint64_t moe_gate_up_expert_stride = 0;
    uint64_t moe_down_expert_stride = 0;
    uint32_t moe_router_normalization_scale_bits = 0;
    ActivationKind moe_activation = ActivationKind::Silu;
    ExpertScaleSource moe_scale_source = ExpertScaleSource::None;
    ValueSource moe_value_source = ValueSource::SeparateProjection;
};

struct KernelPattern {
    ExactOrAny<OperatorKind> operation;
    ClosedRange<uint16_t> semantic_version;
    ExactOrAny<ExecutionPhase> phase;
    ExactOrAny<ScalarType> logical_type;
    ExactOrAny<ScalarType> storage_type;
    ExactOrAny<PhysicalLayoutKind> layout;
    ExactOrAny<QuantizationKind> quantization;
    ExactOrAny<StateKind> state_kind;
    ExactOrAny<StateFormatKind> state_format;
    ClosedRange<uint32_t> rank;
    ClosedRange<uint32_t> alignment;
    ClosedRange<uint32_t> head_dimension;
    ClosedRange<uint32_t> batch_rows = {1, UINT32_MAX};
    ClosedRange<uint32_t> tile_tokens;
    ClosedRange<uint32_t> block_elements;
    ClosedRange<uint32_t> block_bytes;
    // Codec-aware plans require this exact tensor-order sequence. Empty is
    // valid only when no physical registry was supplied.
    std::vector<PhysicalCodecIdentity> required_physical_codecs;
    // A zero divisor disables the constraint. Nonzero values require an exact
    // multiple after the closed-range check.
    uint32_t head_dimension_multiple = 0;
    uint32_t allowed_metal_weight_formats = 0;
    bool require_mixed_metal_weight_formats = false;
    bool require_metal_dense_token_pattern = false;
    bool require_metal_moe_token_pattern = false;
    bool require_metal_dense_prefill_batch_pattern = false;
    bool require_metal_recurrent_token_pattern = false;
    bool require_metal_affine_u2_256 = false;
    bool require_metal_column_grouped_affine_u2_skip_256 = false;
    bool require_moe_descriptor = false;
    ClosedRange<uint32_t> moe_expert_count = {0, UINT32_MAX};
    ClosedRange<uint32_t> moe_selected_count = {0, UINT32_MAX};
    ExactOrAny<uint32_t> moe_gate_up_format = any<uint32_t>();
    ExactOrAny<uint32_t> moe_down_format = any<uint32_t>();
    ExactOrAny<PhysicalLayoutKind> moe_gate_up_layout = any<PhysicalLayoutKind>();
    ExactOrAny<PhysicalLayoutKind> moe_down_layout = any<PhysicalLayoutKind>();
    ExactOrAny<QuantizationKind> moe_gate_up_quantization = any<QuantizationKind>();
    ExactOrAny<QuantizationKind> moe_down_quantization = any<QuantizationKind>();
    ExactOrAny<uint32_t> moe_gate_up_plane_mask = any<uint32_t>();
    ExactOrAny<uint32_t> moe_down_plane_mask = any<uint32_t>();
    ClosedRange<uint32_t> moe_gate_up_block_elements = {0, UINT32_MAX};
    ClosedRange<uint32_t> moe_gate_up_block_bytes = {0, UINT32_MAX};
    ClosedRange<uint32_t> moe_down_block_elements = {0, UINT32_MAX};
    ClosedRange<uint32_t> moe_down_block_bytes = {0, UINT32_MAX};
    ClosedRange<uint32_t> moe_gate_up_input = {0, UINT32_MAX};
    ClosedRange<uint32_t> moe_gate_up_output = {0, UINT32_MAX};
    ClosedRange<uint32_t> moe_down_input = {0, UINT32_MAX};
    ClosedRange<uint32_t> moe_down_output = {0, UINT32_MAX};
    ExactOrAny<uint64_t> moe_gate_up_expert_stride = any<uint64_t>();
    ExactOrAny<uint64_t> moe_down_expert_stride = any<uint64_t>();
    ExactOrAny<ActivationKind> moe_activation = any<ActivationKind>();
    ExactOrAny<ExpertScaleSource> moe_scale_source = any<ExpertScaleSource>();
    ExactOrAny<ValueSource> moe_value_source = any<ValueSource>();
};

struct PlanEffectKey {
    BackendWriter backend = BackendWriter::Cpu;
    NumericalClass result_class = NumericalClass::ExactFp32;
    StateFormatKind state_format = StateFormatKind::GlobalContiguous;
    FallbackKind fallback = FallbackKind::ExactCpu;
    bool transactional = false;
    ScratchLifetime scratch = ScratchLifetime::None;
    friend bool operator==(const PlanEffectKey&, const PlanEffectKey&) = default;
};

struct CostEstimate {
    uint64_t latency = 0;
    uint64_t throughput = 0;
};

using CapabilityPredicate = bool (*)(const RuntimeCapabilities&);

struct KernelDescriptor {
    uint32_t id = 0;
    uint16_t priority = 0;
    KernelImplementation implementation = KernelImplementation::ScalarEmbedding;
    KernelPattern pattern;
    CapabilityPredicate capability_predicate = nullptr;
    NumericalClass numerical_class = NumericalClass::ExactFp32;
    bool transactional = false;
    PlanEffectKey effects;
    CostEstimate cost;
};

struct CheckedTensorSpan {
    uint32_t tensor_id = 0;
    ArtifactId artifact_id{};
    uint64_t offset = 0;
    uint64_t length = 0;
    uint32_t alignment = 0;
    ScalarType logical_type = ScalarType::F32;
    PhysicalLayoutKind layout = PhysicalLayoutKind::ContiguousRowMajor;
    QuantizationKind quantization = QuantizationKind::None;
};

struct CheckedStateBinding {
    uint32_t state_id = 0;
    StateKind kind = StateKind::KeyCache;
    StateFormatKind format = StateFormatKind::GlobalContiguous;
    uint32_t alignment = 0;
    uint32_t tile_tokens = 0;
};

struct CheckedPhysicalPlaneSpan {
    PlaneKind kind = PlaneKind::Values;
    ScalarType storage_type = ScalarType::F32;
    ArtifactId artifact_id{};
    uint64_t offset = 0;
    uint64_t length = 0;
    uint32_t alignment = 0;
    uint32_t flags = 0;
    friend bool operator==(const CheckedPhysicalPlaneSpan&,
                           const CheckedPhysicalPlaneSpan&) = default;
};

struct PhysicalTensorCodecBinding {
    uint32_t tensor_id = 0;
    PhysicalCodecIdentity identity;
    std::vector<Dimension> dimensions;
    std::array<uint64_t, 8> strides{};
    std::vector<CheckedPhysicalPlaneSpan> planes;
    friend bool operator==(const PhysicalTensorCodecBinding&,
                           const PhysicalTensorCodecBinding&) = default;
};

struct PlanEntry {
    bool codec_aware = false;
    ExecutionPhase phase = ExecutionPhase::Prefill;
    uint32_t operator_id = 0;
    uint32_t kernel_id = 0;
    KernelDescriptor descriptor;
    std::vector<PhysicalTensorCodecBinding> codec_bindings;
    std::vector<CheckedTensorSpan> tensors;
    std::vector<CheckedStateBinding> states;
};

struct CanonicalLayerBoundary {
    uint32_t layer_index = 0;
    uint32_t input_value_id = 0;
    uint32_t output_value_id = 0;

    bool operator==(const CanonicalLayerBoundary&) const = default;
};

struct CanonicalProgramWitness {
    uint32_t token_input_value_id = 0;
    uint32_t embedding_operator_id = 0;
    uint32_t embedding_output_value_id = 0;
    std::vector<CanonicalLayerBoundary> layers;
    uint32_t final_norm_operator_id = 0;
    uint32_t final_norm_output_value_id = 0;
    uint32_t output_operator_id = 0;
    uint32_t output_value_id = 0;

    bool operator==(const CanonicalProgramWitness&) const = default;
};

struct ExecutionPlan {
    bool codec_aware = false;
    // Product sessions publish the authority fingerprint they were built from.
    Sha256Digest package_fingerprint{};
    std::vector<PlanEntry> entries;
    // Exact residual chain consumed by the serial canonical Metal executor.
    // This is derived from semantic value edges, never from layer or model names.
    std::vector<CanonicalLayerBoundary> layer_chain;
    // Exact graph-level spine consumed by the product Metal executor.
    CanonicalProgramWitness program;
    // Ordered source-neutral execution for every admitted phase. Kernel and
    // pipeline slots remain unresolved until the exact Metal transaction is
    // attached; the planner may not reconstruct or omit graph-level work.
    std::vector<SemanticDispatchProgram> dispatch_programs;
    // Pipeline transactions populate these plan-owned slots. The semantic
    // program remains immutable, and every slot starts unresolved.
    std::vector<std::vector<uint32_t>> dispatch_pipeline_slots;
    uint64_t reserved_bytes = 0;
    uint64_t peak_bytes = 0;
};

using KernelSelection = std::variant<KernelDescriptor, CompatibilityReport>;
using PlanResult = std::variant<ExecutionPlan, CompatibilityReport>;

bool requires_scalar_fp32(const RuntimeCapabilities& capabilities);
bool pattern_matches(const KernelPattern& pattern, const KernelQuery& query);
KernelSelection select_kernel(const KernelQuery& query, const SessionRequest& request,
                              const RuntimeCapabilities& capabilities,
                              const std::vector<KernelDescriptor>& registry);
std::vector<KernelDescriptor> builtin_cpu_registry();
bool requires_canonical_metal(const RuntimeCapabilities& capabilities);
bool requires_canonical_moe(const RuntimeCapabilities& capabilities);
bool canonical_layer_chain_witness(const SemanticModel& model,
                                   std::vector<CanonicalLayerBoundary>& witness,
                                   std::string& detail);
bool canonical_program_witness(const SemanticModel& model,
                               CanonicalProgramWitness& witness,
                               std::string& detail);
std::vector<KernelDescriptor> builtin_canonical_metal_registry();
PlanResult plan_canonical_metal(const SemanticModel& model, const SessionRequest& request,
                                const RuntimeCapabilities& capabilities,
                                const std::vector<KernelDescriptor>& registry);
PlanResult plan_canonical_metal_dense(const SemanticModel& model, const SessionRequest& request,
                                      const RuntimeCapabilities& capabilities,
                                      const std::vector<KernelDescriptor>& registry);
PlanResult plan_canonical_metal_recurrent(const SemanticModel& model, const SessionRequest& request,
                                          const RuntimeCapabilities& capabilities,
                                          const std::vector<KernelDescriptor>& registry);
PlanResult plan_session(const SemanticModel& model, const SessionRequest& request,
                        const RuntimeCapabilities& capabilities,
                        const std::vector<KernelDescriptor>& registry,
                        const PhysicalCodecRegistry& codec_registry = {});
bool plan_entry_matches(const SemanticModel& model, const PlanEntry& entry);

} // namespace Laplace
