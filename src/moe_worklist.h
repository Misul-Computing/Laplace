#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <variant>
#include <utility>

#include "execution_plan.h"

namespace Laplace {

inline constexpr uint32_t kMoeWorklistMaximumSourceSpans = 16;
inline constexpr uint64_t kMoeWorklistPayloadAlignment = 16;

// This is the host-to-device reference for one immutable source plane. It
// carries no artifact selector. The package-bound tensor id and contract
// digest select the source at the later session binding boundary.
struct MoeWorklistSourceSpanRef {
    uint32_t tensor_id = UINT32_MAX;
    PlaneKind plane = PlaneKind::Values;
    uint16_t reserved = 0;
    uint64_t offset = 0;
    uint64_t length = 0;
    friend bool operator==(const MoeWorklistSourceSpanRef&,
                           const MoeWorklistSourceSpanRef&) = default;
};
static_assert(sizeof(MoeWorklistSourceSpanRef) == 24);

// Metal indirect dispatch arguments are exactly three 32-bit threadgroup
// counts. Route and segment ranges remain in their fixed payload arrays.
struct MoeWorklistIndirectDispatch {
    uint32_t threadgroups_x = 0;
    uint32_t threadgroups_y = 0;
    uint32_t threadgroups_z = 0;
    friend bool operator==(const MoeWorklistIndirectDispatch&,
                           const MoeWorklistIndirectDispatch&) = default;
};
static_assert(sizeof(MoeWorklistIndirectDispatch) == 12);

// Exact fixed-capacity byte layout. Every offset is relative to the start of
// one device payload and each section starts at the declared 16-byte boundary.
// The payload sections are, in order: routed token rows, logical expert ids,
// route weights, segment counts, segment offsets, indirect dispatch arguments,
// and source span references.
struct MoeWorklistDeviceLayout {
    uint32_t token_capacity = 0;
    uint32_t route_capacity = 0;
    uint32_t expert_capacity = 0;
    uint32_t source_span_capacity = 0;
    uint64_t routed_token_offset = 0;
    uint64_t logical_expert_offset = 0;
    uint64_t route_weight_offset = 0;
    uint64_t segment_count_offset = 0;
    uint64_t segment_offset_offset = 0;
    uint64_t indirect_dispatch_offset = 0;
    uint64_t source_span_offset = 0;
    uint64_t byte_size = 0;
    friend bool operator==(const MoeWorklistDeviceLayout&,
                           const MoeWorklistDeviceLayout&) = default;
};
static_assert(sizeof(MoeWorklistDeviceLayout) == 80);

class MoeWorklistPlan;
using MoeWorklistPlanResult = std::variant<MoeWorklistPlan, CompatibilityReport>;

// A plan owns copies of all package-visible control data. It exposes no
// mutator and carries only neutral ids, source spans, and authenticated
// contract data. CPU execution and model selectors are not part of this ABI.
class MoeWorklistPlan {
public:
    const MoeWorklistDescriptor& descriptor() const noexcept { return descriptor_; }
    const Sha256Digest& contract_digest() const noexcept { return contract_digest_; }
    const MoeWorklistDeviceLayout& device_layout() const noexcept { return device_layout_; }
    std::span<const MoeWorklistSourceSpanRef> source_spans() const noexcept {
        return {source_spans_.data(), source_span_count_};
    }

private:
    MoeWorklistPlan(MoeWorklistDescriptor descriptor, Sha256Digest contract_digest,
                    MoeWorklistDeviceLayout device_layout,
                    std::array<MoeWorklistSourceSpanRef, kMoeWorklistMaximumSourceSpans> source_spans,
                    uint32_t source_span_count)
        : descriptor_(std::move(descriptor)), contract_digest_(contract_digest),
          device_layout_(device_layout), source_spans_(std::move(source_spans)),
          source_span_count_(source_span_count) {}

    MoeWorklistDescriptor descriptor_;
    Sha256Digest contract_digest_;
    MoeWorklistDeviceLayout device_layout_;
    std::array<MoeWorklistSourceSpanRef, kMoeWorklistMaximumSourceSpans> source_spans_{};
    uint32_t source_span_count_ = 0;

    friend MoeWorklistPlanResult build_moe_worklist_plan(
        const SemanticModel&, const SemanticLayer&, ExecutionPhase, uint32_t,
        const PhysicalCodecRegistry&);
};

MoeWorklistPlanResult build_moe_worklist_plan(
    const SemanticModel& model, const SemanticLayer& layer,
    ExecutionPhase phase, uint32_t token_capacity,
    const PhysicalCodecRegistry& codec_registry);

bool valid_moe_worklist_plan(const MoeWorklistPlan& plan);

} // namespace Laplace
