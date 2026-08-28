#pragma once

#include <cstdint>
#include <variant>
#include <vector>

#include "compatibility_report.h"
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
};

struct SessionRequest {
    uint64_t max_context = 0;
    uint32_t max_batch = 0;
    uint64_t memory_limit = 0;
    bool enable_prefill = false;
    bool enable_decode = false;
    bool enable_streaming = false;
    bool enable_speculation = false;
    NumericalClass minimum_class = NumericalClass::ExactFp32;
    RuntimeObjective objective = RuntimeObjective::Latency;
};

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
    uint32_t metal_weight_format_mask = 0;
    bool metal_dense_token_pattern = false;
    bool metal_dense_prefill_batch_pattern = false;
    bool metal_recurrent_token_pattern = false;
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
    // A zero divisor disables the constraint. Nonzero values require an exact
    // multiple after the closed-range check.
    uint32_t head_dimension_multiple = 0;
    uint32_t allowed_metal_weight_formats = 0;
    bool require_mixed_metal_weight_formats = false;
    bool require_metal_dense_token_pattern = false;
    bool require_metal_dense_prefill_batch_pattern = false;
    bool require_metal_recurrent_token_pattern = false;
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

struct PlanEntry {
    ExecutionPhase phase = ExecutionPhase::Prefill;
    uint32_t operator_id = 0;
    uint32_t kernel_id = 0;
    KernelDescriptor descriptor;
    std::vector<CheckedTensorSpan> tensors;
    std::vector<CheckedStateBinding> states;
};

struct ExecutionPlan {
    std::vector<PlanEntry> entries;
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
                        const std::vector<KernelDescriptor>& registry);
bool plan_entry_matches(const SemanticModel& model, const PlanEntry& entry);

} // namespace Laplace
