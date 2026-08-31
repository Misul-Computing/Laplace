#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <variant>
#include <vector>

#include "moe_worklist.h"
#include "test_util.h"

using namespace Laplace;

namespace {

constexpr uint32_t kExperts = 4;
constexpr uint32_t kTopK = 2;
constexpr uint32_t kHidden = 8;
constexpr uint32_t kIntermediate = 4;

std::vector<Dimension> dims(std::initializer_list<Dimension> values) {
    return {values};
}

SemanticValue value(uint32_t id, ScalarType type, std::vector<Dimension> dimensions) {
    return {id, type, std::move(dimensions), 0};
}

SemanticTensor f32_tensor(uint32_t id, TensorRole role, uint32_t width, uint64_t offset) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = dims({{DimensionKind::Constant, width}});
    tensor.layout.rank = 1;
    tensor.layout.axis_order[0] = 0;
    tensor.layout.strides[0] = 1;
    tensor.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, offset,
                      static_cast<uint64_t>(width) * sizeof(float), 16, 0}};
    return tensor;
}

SemanticTensor expert_tensor(uint32_t id, TensorRole role, uint32_t input,
                             uint32_t output, uint64_t offset) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = dims({{DimensionKind::Constant, kExperts},
                               {DimensionKind::Constant, input},
                               {DimensionKind::Constant, output}});
    tensor.layout.rank = 3;
    tensor.layout.axis_order = {0, 1, 2, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides[0] = 1;
    tensor.layout.strides[1] = output;
    tensor.layout.strides[2] = static_cast<uint64_t>(input) * output;
    const uint64_t expert_bytes = static_cast<uint64_t>(input) * output * sizeof(float);
    tensor.expert_axis = {ExpertAxisKind::ExpertBank, 0, 0xff, 1, 2,
                          kExperts, expert_bytes, 0};
    tensor.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, offset,
                      expert_bytes * kExperts, 16, 0}};
    return tensor;
}

PhysicalCodecIdentity codec_for(const SemanticTensor& tensor, uint8_t tag) {
    PhysicalIdentityDigest arithmetic{};
    arithmetic[0] = tag;
    const auto identity = physical_codec_identity(tensor, 1, arithmetic);
    CHECK(identity.has_value());
    return identity.value_or(PhysicalCodecIdentity{});
}

struct Fixture {
    SemanticModel model;
    SemanticLayer layer;
    PhysicalCodecRegistry registry;
};

Fixture fixture() {
    Fixture result;
    result.model.schema_major = 7;
    result.model.opset_major = 7;
    result.model.maximum_context = 32768;
    result.model.vocabulary_size = 1;
    result.model.bos_id = 0;
    result.model.eos_id = 0;
    result.model.values = {
        value(0, ScalarType::F32, dims({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, kHidden}})),
        value(1, ScalarType::F32, dims({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, kHidden}})),
        value(2, ScalarType::F32, dims({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, kExperts}})),
        value(3, ScalarType::F32, dims({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, kExperts}})),
        value(4, ScalarType::U32, dims({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, kTopK}})),
        value(5, ScalarType::F32, dims({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, kTopK}})),
        value(6, ScalarType::F32, dims({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, kTopK},
                                        {DimensionKind::Constant, 2 * kIntermediate}})),
        value(7, ScalarType::F32, dims({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, kTopK},
                                        {DimensionKind::Constant, kIntermediate}})),
        value(8, ScalarType::F32, dims({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, kTopK},
                                        {DimensionKind::Constant, kIntermediate}})),
        value(9, ScalarType::F32, dims({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, kTopK},
                                        {DimensionKind::Constant, kIntermediate}})),
        value(10, ScalarType::F32, dims({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, kTopK},
                                         {DimensionKind::Constant, kHidden}})),
        value(11, ScalarType::F32, dims({{DimensionKind::Symbol, 1}, {DimensionKind::Constant, kHidden}})),
    };
    result.model.tensors = {
        expert_tensor(0, TensorRole::FfnUpWeight, kHidden, 2 * kIntermediate, 0),
        expert_tensor(1, TensorRole::FfnDownWeight, kIntermediate, kHidden, 1024),
        f32_tensor(2, TensorRole::RouterScaleWeight, kExperts, 2048),
        f32_tensor(3, TensorRole::ExpertNormWeight, kHidden, 2112),
        f32_tensor(4, TensorRole::ReduceScaleWeight, kExperts, 2176),
    };
    const auto op = [](uint32_t id, OperatorKind kind, std::vector<uint32_t> inputs,
                       std::vector<uint32_t> outputs, std::vector<uint32_t> tensors,
                       OperatorPayload payload) {
        return SemanticOperator{id, kind, 7, std::move(inputs), std::move(outputs),
                                std::move(tensors), {}, std::move(payload)};
    };
    result.model.operators = {
        op(0, OperatorKind::RmsNorm, {0}, {1}, {3}, RmsNormPayload{0x358637bdu, -1, 1}),
        op(1, OperatorKind::Scale, {2}, {3}, {2}, ScalePayload{ScaleSource::Tensor, 0}),
        op(2, OperatorKind::RouterTopK, {3}, {4, 5}, {},
           RouterTopKPayload{kExperts, kTopK, RouterScoreDomain::Logits,
                             RouterNormalizationOrder::NormalizeThenSelect,
                             SelectedWeightNormalization::RenormalizeSelectedProbabilities,
                             RouterTiePolicy::LowestExpertId,
                             RouterWeightSource::SelectedNormalizedScore, 0}),
        op(3, OperatorKind::RoutedLinear, {1, 4, 5}, {6}, {0}, RoutedLinearPayload{}),
        op(4, OperatorKind::AxisSplit, {6}, {7, 8}, {}, AxisSplitPayload{kIntermediate, kIntermediate}),
        op(5, OperatorKind::GatedActivation, {7, 8}, {9}, {}, GatedActivationPayload{ActivationKind::GeluTanh}),
        op(6, OperatorKind::RoutedLinear, {9, 4, 5}, {10}, {1}, RoutedLinearPayload{}),
        op(7, OperatorKind::WeightedExpertReduce, {10, 4, 5}, {11}, {4},
           WeightedExpertReducePayload{ExpertReduceAssociation::SelectedOrderLeftToRight,
                                       ExpertScaleSource::PerExpertTensor, ScalarType::F32}),
    };
    result.layer = {0, 0, static_cast<uint32_t>(result.model.operators.size()), 0};

    const std::array<uint8_t, 5> tags = {1, 2, 3, 4, 5};
    std::vector<PhysicalCodecIdentity> identities;
    for (size_t index = 0; index != result.model.tensors.size(); ++index)
        identities.push_back(codec_for(result.model.tensors[index], tags[index]));
    result.registry.codecs.reserve(identities.size());
    for (const PhysicalCodecIdentity& identity : identities)
        result.registry.codecs.push_back({identity, {}});
    for (size_t index = 0; index != identities.size(); ++index)
        result.registry.tensors.push_back({static_cast<uint32_t>(index), identities[index]});
    return result;
}

bool rejected(const MoeWorklistPlanResult& result) {
    return std::holds_alternative<CompatibilityReport>(result);
}

void test_positive_dense_independent_plan() {
    const Fixture input = fixture();
    const auto result = build_moe_worklist_plan(input.model, input.layer,
                                                ExecutionPhase::Decode, 4, input.registry);
    CHECK(std::holds_alternative<MoeWorklistPlan>(result));
    if (const auto* plan = std::get_if<MoeWorklistPlan>(&result)) {
        CHECK(valid_moe_worklist_plan(*plan));
        CHECK(plan->contract_digest() != Sha256Digest{});
        CHECK((plan->descriptor().semantic_operator_ids == std::array<uint32_t, 5>{2, 3, 5, 6, 7}));
        CHECK(plan->descriptor().physical_codecs.size() == 3);
        CHECK(plan->descriptor().gate_up.dimensions.size() == 3);
        CHECK(plan->device_layout().token_capacity == 4);
        CHECK(plan->device_layout().route_capacity == 8);
        CHECK(plan->device_layout().source_span_capacity == kMoeWorklistMaximumSourceSpans);
        CHECK(plan->device_layout().byte_size % kMoeWorklistPayloadAlignment == 0);
        CHECK(plan->source_spans().size() == 5);
        CHECK(plan->source_spans()[0].tensor_id == 0);
        CHECK(plan->source_spans()[1].tensor_id == 1);
        CHECK(plan->source_spans()[4].tensor_id == 4);
    }
}

void test_codec_tamper_fails_closed() {
    Fixture input = fixture();
    input.registry.tensors[0].identity.arithmetic_digest[0] ^= 1;
    CHECK(rejected(build_moe_worklist_plan(input.model, input.layer,
                                           ExecutionPhase::Decode, 4, input.registry)));
}

void test_ambiguous_router_fails_closed() {
    Fixture input = fixture();
    input.model.operators.push_back(input.model.operators[2]);
    input.layer.operator_count = static_cast<uint32_t>(input.model.operators.size());
    CHECK(rejected(build_moe_worklist_plan(input.model, input.layer,
                                           ExecutionPhase::Decode, 4, input.registry)));
}

void test_capacity_overflow_fails_closed() {
    const Fixture input = fixture();
    CHECK(rejected(build_moe_worklist_plan(input.model, input.layer,
                                           ExecutionPhase::Decode,
                                           std::numeric_limits<uint32_t>::max(), input.registry)));
}

void test_route_input_order_fails_closed() {
    Fixture input = fixture();
    std::swap(input.model.operators[3].inputs[1], input.model.operators[3].inputs[2]);
    CHECK(rejected(build_moe_worklist_plan(input.model, input.layer,
                                           ExecutionPhase::Decode, 4, input.registry)));
}

void test_declared_layout_and_binding_are_authenticated() {
    Fixture input = fixture();
    const auto first = build_moe_worklist_plan(input.model, input.layer,
                                               ExecutionPhase::Decode, 4, input.registry);
    CHECK(std::holds_alternative<MoeWorklistPlan>(first));
    if (!std::holds_alternative<MoeWorklistPlan>(first)) return;
    const MoeWorklistPlan& first_plan = std::get<MoeWorklistPlan>(first);
    CHECK(first_plan.descriptor().gate_up.strides == input.model.tensors[0].layout.strides);

    Fixture other_layer = input;
    other_layer.layer.layer_index = 1;
    const auto second = build_moe_worklist_plan(other_layer.model, other_layer.layer,
                                                ExecutionPhase::Decode, 4, other_layer.registry);
    CHECK(std::holds_alternative<MoeWorklistPlan>(second));
    if (std::holds_alternative<MoeWorklistPlan>(second)) {
        CHECK(first_plan.contract_digest() != std::get<MoeWorklistPlan>(second).contract_digest());
    }

    Fixture other_binding = input;
    other_binding.model.tensors[0].planes[0].offset += 4096;
    const auto third = build_moe_worklist_plan(other_binding.model, other_binding.layer,
                                               ExecutionPhase::Decode, 4, other_binding.registry);
    CHECK(std::holds_alternative<MoeWorklistPlan>(third));
    if (std::holds_alternative<MoeWorklistPlan>(third)) {
        CHECK(first_plan.contract_digest() != std::get<MoeWorklistPlan>(third).contract_digest());
    }
}

} // namespace

int main() {
    test_positive_dense_independent_plan();
    test_codec_tamper_fails_closed();
    test_ambiguous_router_fails_closed();
    test_capacity_overflow_fails_closed();
    test_route_input_order_fails_closed();
    test_declared_layout_and_binding_are_authenticated();
    return test_summary("test_moe_worklist");
}
