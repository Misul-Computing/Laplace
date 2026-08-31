#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <fcntl.h>
#include <memory>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include "artifact_set.h"
#include "compat_rule.h"
#include "reference_fp32.h"
#include "runtime_session.h"
#include "scalar_executor.h"
#include "test_util.h"

using namespace Laplace;

static_assert(!std::is_invocable_v<decltype(&create_runtime_session),
                                   std::shared_ptr<const RuntimePackage>,
                                   SessionRequest, SessionFaultPoint>);

namespace {

struct GateBMaximum {
    float logit_error = 0.0f;
    uint32_t logit_prefix = 0;
    size_t logit_index = 0;
    float checkpoint_error = 0.0f;
    uint32_t checkpoint_prefix = 0;
    size_t checkpoint_index = 0;
};

GateBMaximum g_gate_b_maximum;

std::shared_ptr<const RuntimePackage> load_package() {
    auto artifacts = ArtifactSet::load_single_file(LAPLACE_QUALIFICATION_GGUF);
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return {};
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    if (!std::holds_alternative<PackageView>(view)) return {};
    auto loaded = load_expected_fixture_gguf(std::get<PackageView>(view), bundled_compatibility_rules());
    if (!std::holds_alternative<ValidatedPackage>(loaded)) return {};
    return std::get<ValidatedPackage>(loaded).runtime_package();
}

std::shared_ptr<const RuntimePackage> axis_split_package() {
    char path[] = "/private/tmp/laplace-axis-split-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) return {};
    // TensorReader uses the physical [row, column] order declared below.
    const std::array<float, 12> bytes = {1, 5, 2, 6, 3, 7, 4, 8, 1, 0, 0, 1};
    const bool written = write(fd, bytes.data(), sizeof(bytes)) == static_cast<ssize_t>(sizeof(bytes));
    close(fd);
    if (!written) {
        unlink(path);
        return {};
    }
    auto artifacts = ArtifactSet::load_single_file(path);
    unlink(path);
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return {};
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    if (!std::holds_alternative<PackageView>(view)) return {};

    auto tensor = [](uint32_t id, TensorRole role, uint32_t rows, uint32_t columns, uint64_t offset) {
        SemanticTensor value;
        value.id = id;
        value.role = role;
        value.logical_type = ScalarType::F32;
        value.dimensions = {{DimensionKind::Constant, rows}, {DimensionKind::Constant, columns}};
        value.layout.rank = 2;
        value.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        value.layout.strides[0] = 1;
        value.layout.strides[1] = rows;
        value.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, offset,
                         static_cast<uint64_t>(rows) * columns * sizeof(float), 4, 0}};
        return value;
    };
    SemanticModel semantics;
    semantics.schema_major = 4;
    semantics.opset_major = 4;
    semantics.maximum_context = 4;
    semantics.vocabulary_size = 2;
    semantics.tensors = {
        tensor(0, TensorRole::TokenEmbedding, 2, 4, 0),
        tensor(1, TensorRole::OutputWeight, 2, 2, 8 * sizeof(float)),
    };
    semantics.values = {
        {0, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 4}}, 0},
        {1, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}}, 0},
        {2, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}}, 0},
        {3, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}}, 0},
        {4, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}}, 0},
    };
    semantics.operators = {
        {0, OperatorKind::EmbeddingLookup, 1, {0}, {0}, {0}, {}, EmbeddingLookupPayload{0x3f800000u, 2, 4, 0}},
        {1, OperatorKind::AxisSplit, 4, {0}, {1, 2}, {}, {}, AxisSplitPayload{2, 2}},
        {2, OperatorKind::GatedAttention, 3, {1, 2}, {3}, {}, {}, GatedAttentionPayload{}},
        {3, OperatorKind::Linear, 1, {3}, {4}, {1}, {}, LinearPayload{}},
    };
    const PackageView& package = std::get<PackageView>(view);
    return RuntimePackage::make_legacy_test_only(std::move(semantics), package, package.digest(), package.digest(),
                                                 0, RuleQualificationState::Draft);
}

SessionRequest request() {
    SessionRequest result;
    result.max_context = 8;
    result.max_batch = 1;
    result.memory_limit = UINT64_MAX;
    result.enable_prefill = true;
    result.enable_decode = true;
    result.minimum_class = NumericalClass::ExactFp32;
    result.objective = RuntimeObjective::Latency;
    return result;
}

uint32_t argmax(const std::vector<float>& values) {
    return static_cast<uint32_t>(std::max_element(values.begin(), values.end()) - values.begin());
}

bool per_token_width(const SemanticValue& value, size_t& width) {
    width = 1;
    for (const Dimension& dimension : value.dimensions) {
        if (dimension.kind == DimensionKind::Symbol) continue;
        if (dimension.kind != DimensionKind::Constant || dimension.constant_or_symbol == 0 ||
            dimension.constant_or_symbol > SIZE_MAX / width) return false;
        width *= static_cast<size_t>(dimension.constant_or_symbol);
    }
    return width != 0;
}

void compare_output(const SemanticModel& model, const RuntimeOutput& output, const ReferenceOutput& reference, uint32_t prefix) {
    CHECK(output.logits.size() == reference.logits.size());
    CHECK(output.checkpoints.size() == reference.operator_outputs.size());
    CHECK(output.states.size() == reference.states.size());
    for (size_t index = 0; index != output.logits.size() && index != reference.logits.size(); ++index) {
        CHECK(std::isfinite(output.logits[index]));
        const float error = std::abs(output.logits[index] - reference.logits[index]);
        if (error > g_gate_b_maximum.logit_error) g_gate_b_maximum = {error, prefix, index,
                                                                         g_gate_b_maximum.checkpoint_error,
                                                                         g_gate_b_maximum.checkpoint_prefix,
                                                                         g_gate_b_maximum.checkpoint_index};
        if (error > 1e-4f + 1e-4f * std::abs(reference.logits[index])) {
            CHECK(false);
            return;
        }
    }
    CHECK(argmax(output.logits) == argmax(reference.logits));
    for (size_t checkpoint = 0; checkpoint != output.checkpoints.size() && checkpoint != reference.operator_outputs.size(); ++checkpoint) {
        const auto& actual = output.checkpoints[checkpoint];
        const auto& expected = reference.operator_outputs[checkpoint];
        // Incremental execution reports only the newly admitted token rows.
        // The independent full-prefix oracle reports all rows, so compare its tail.
        CHECK(actual.size() <= expected.size());
        if (actual.size() > expected.size()) continue;
        const SemanticOperator* op = checkpoint < model.operators.size() ? &model.operators[checkpoint] : nullptr;
        const bool rope = op && op->kind == OperatorKind::Rope;
        size_t query_width = 0, key_width = 0;
        if (rope && (op->outputs.size() != 2 || op->outputs[0] >= model.values.size() || op->outputs[1] >= model.values.size() ||
                     !per_token_width(model.values[op->outputs[0]], query_width) ||
                     !per_token_width(model.values[op->outputs[1]], key_width) ||
                     query_width > SIZE_MAX - key_width || actual.size() % (query_width + key_width) != 0 ||
                     expected.size() % (query_width + key_width) != 0)) {
            CHECK(false);
            return;
        }
        const size_t expected_offset = expected.size() - actual.size();
        for (size_t index = 0; index != actual.size(); ++index) {
            CHECK(std::isfinite(actual[index]));
            const size_t new_rows = rope ? actual.size() / (query_width + key_width) : 0;
            const size_t expected_rows = rope ? expected.size() / (query_width + key_width) : 0;
            const size_t expected_index = rope ? (index < new_rows * query_width
                ? (expected_rows - new_rows) * query_width + index
                : expected_rows * query_width + (expected_rows - new_rows) * key_width +
                  (index - new_rows * query_width)) : expected_offset + index;
            const float error = std::abs(actual[index] - expected[expected_index]);
            if (error > g_gate_b_maximum.checkpoint_error) {
                g_gate_b_maximum.checkpoint_error = error;
                g_gate_b_maximum.checkpoint_prefix = prefix;
                g_gate_b_maximum.checkpoint_index = checkpoint;
            }
            if (error > 2e-5f + 2e-5f * std::abs(expected[expected_index])) {
                CHECK(false);
                return;
            }
        }
    }
    for (size_t state = 0; state != output.states.size() && state != reference.states.size(); ++state) {
        CHECK(output.states[state].tokens == reference.states[state].tokens);
        CHECK(output.states[state].key == reference.states[state].key);
        CHECK(output.states[state].value == reference.states[state].value);
        for (float value : output.states[state].key) CHECK(std::isfinite(value));
        for (float value : output.states[state].value) CHECK(std::isfinite(value));
    }
}

void test_transactional_construction() {
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) return;
    for (SessionFaultPoint point : {SessionFaultPoint::DeviceQuery, SessionFaultPoint::CommandQueue}) {
        auto created = create_qualification_runtime_session(package, request(), point);
        CHECK(std::holds_alternative<RuntimeSession>(created));
        CHECK(session_live_resource_count() == 0);
    }
    for (SessionFaultPoint point : {SessionFaultPoint::Plan, SessionFaultPoint::StateAllocation}) {
        auto created = create_qualification_runtime_session(package, request(), point);
        CHECK(std::holds_alternative<CompatibilityReport>(created));
        CHECK(session_live_resource_count() == 0);
    }
}

void test_public_qualification_gate() {
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) return;
    auto qualification_result = create_qualification_runtime_session(package, request());
    CHECK(std::holds_alternative<RuntimeSession>(qualification_result));
}

void test_canonical_product_factory_gate() {
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_canonical_metal_program(package, request());
    CHECK(std::holds_alternative<CompatibilityReport>(created));
    if (const auto* report = std::get_if<CompatibilityReport>(&created)) {
        CHECK(report->code == CompatibilityError::PACKAGE_AUTHORITY_REQUIRED);
    }
}

void test_plan_dispatch_rejects_tampering() {
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_runtime_session(package, request());
    CHECK(std::holds_alternative<RuntimeSession>(created));
    if (!std::holds_alternative<RuntimeSession>(created)) return;
    ExecutionPlan forged = std::get<RuntimeSession>(std::move(created)).plan();
    CHECK(!forged.entries.empty());
    if (forged.entries.empty()) return;
    forged.entries.front().kernel_id += 1000;
    const std::vector<uint32_t> token_ids = {1};
    auto dispatched = execute_planned_scalar(*package, forged, ExecutionPhase::Prefill, token_ids);
    CHECK(std::holds_alternative<CompatibilityReport>(dispatched));
    if (const auto* report = std::get_if<CompatibilityReport>(&dispatched)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
    }
}

void test_reference_axis_split() {
    auto package = axis_split_package();
    CHECK(package != nullptr);
    if (!package) return;
    auto result = reference_fp32(*package, std::vector<uint32_t>{1});
    CHECK(std::holds_alternative<ReferenceOutput>(result));
    if (const auto* output = std::get_if<ReferenceOutput>(&result)) {
        CHECK(output->logits.size() == 2);
        if (output->logits.size() == 2) {
            CHECK(almost_equal(output->logits[0], 5.0f / (1.0f + std::exp(-7.0f)), 1.0e-6f, 1.0e-6f));
            CHECK(almost_equal(output->logits[1], 6.0f / (1.0f + std::exp(-8.0f)), 1.0e-6f, 1.0e-6f));
        }
        CHECK(output->operator_outputs.size() == 4);
        if (output->operator_outputs.size() > 1) {
            CHECK(output->operator_outputs[1] == std::vector<float>({5, 6, 7, 8}));
        }
    }
    auto scalar = scalar_execute(*package, std::vector<uint32_t>{1});
    CHECK(std::holds_alternative<CompatibilityReport>(scalar));
    if (const auto* report = std::get_if<CompatibilityReport>(&scalar)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
    }
}

void test_gate_b_detects_production_only_kernel_sabotage() {
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_runtime_session(package, request());
    CHECK(std::holds_alternative<RuntimeSession>(created));
    if (!std::holds_alternative<RuntimeSession>(created)) return;
    ExecutionPlan sabotaged = std::get<RuntimeSession>(std::move(created)).plan();
    CHECK(!sabotaged.entries.empty());
    if (sabotaged.entries.empty()) return;
    sabotaged.entries.front().descriptor.implementation = KernelImplementation::ScalarEmbeddingUnitOffset;
    const std::vector<uint32_t> token_ids = {1};
    auto produced = execute_planned_scalar(*package, sabotaged, ExecutionPhase::Prefill, token_ids);
    auto expected = reference_fp32(*package, token_ids);
    CHECK(std::holds_alternative<ScalarExecutionOutput>(produced));
    CHECK(std::holds_alternative<ReferenceOutput>(expected));
    if (!std::holds_alternative<ScalarExecutionOutput>(produced) || !std::holds_alternative<ReferenceOutput>(expected)) return;
    const auto& production = std::get<ScalarExecutionOutput>(produced);
    const auto& reference = std::get<ReferenceOutput>(expected);
    CHECK(production.logits.size() == reference.logits.size());
    bool comparator_detected_difference = false;
    for (size_t index = 0; index != production.logits.size(); ++index) {
        if (std::abs(production.logits[index] - reference.logits[index]) >
            1e-4f + 1e-4f * std::abs(reference.logits[index])) {
            comparator_detected_difference = true;
            break;
        }
    }
    CHECK(comparator_detected_difference);
}

void test_cpu_gate_b() {
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_runtime_session(package, request());
    CHECK(std::holds_alternative<RuntimeSession>(created));
    if (!std::holds_alternative<RuntimeSession>(created)) return;
    RuntimeSession session = std::get<RuntimeSession>(std::move(created));
    const std::vector<uint32_t> prefill_ids = {1};
    auto prefill = session.prefill(prefill_ids);
    CHECK(std::holds_alternative<RuntimeOutput>(prefill));
    auto expected_prefill = reference_fp32(*package, prefill_ids);
    CHECK(std::holds_alternative<ReferenceOutput>(expected_prefill));
    if (auto* output = std::get_if<RuntimeOutput>(&prefill); output && std::holds_alternative<ReferenceOutput>(expected_prefill)) {
        compare_output(package->semantics(), *output, std::get<ReferenceOutput>(expected_prefill), 1);
        CHECK(output->token_history == std::vector<uint32_t>({1}));
        CHECK(output->operator_token_work == package->semantics().operators.size());
    }
    std::vector<uint32_t> decode_ids = {1};
    for (uint32_t token = 2; token != 9; ++token) {
        decode_ids.push_back(token);
        auto decode = session.decode(token);
        CHECK(std::holds_alternative<RuntimeOutput>(decode));
        auto expected_decode = reference_fp32(*package, decode_ids);
        CHECK(std::holds_alternative<ReferenceOutput>(expected_decode));
        if (auto* output = std::get_if<RuntimeOutput>(&decode); output && std::holds_alternative<ReferenceOutput>(expected_decode)) {
            compare_output(package->semantics(), *output, std::get<ReferenceOutput>(expected_decode), token);
            CHECK(output->token_history == decode_ids);
            // A decode may consume its accumulated KV state, but it must never
            // reconstruct prior token operators to do so.
            CHECK(output->operator_token_work == package->semantics().operators.size());
        }
    }
    std::printf("Gate B maximums: logit %.8g prefix %u index %zu; checkpoint %.8g prefix %u operator %zu\n",
                g_gate_b_maximum.logit_error, g_gate_b_maximum.logit_prefix, g_gate_b_maximum.logit_index,
                g_gate_b_maximum.checkpoint_error, g_gate_b_maximum.checkpoint_prefix, g_gate_b_maximum.checkpoint_index);
}

void test_session_state_lifecycle() {
    auto package = load_package();
    CHECK(package != nullptr);
    if (!package) return;
    auto created = create_qualification_runtime_session(package, request());
    CHECK(std::holds_alternative<RuntimeSession>(created));
    if (!std::holds_alternative<RuntimeSession>(created)) return;
    RuntimeSession session = std::get<RuntimeSession>(std::move(created));
    CHECK(std::holds_alternative<RuntimeOutput>(session.prefill(std::vector<uint32_t>{1, 2})));
    auto before = session.save_state();
    CHECK(std::holds_alternative<std::vector<uint8_t>>(before));
    const StateCursor cursor = session.checkpoint();
    CHECK(std::holds_alternative<RuntimeOutput>(session.decode(3)));
    CHECK(std::holds_alternative<std::monostate>(session.commit(cursor)));
    CHECK(session.token_history() == std::vector<uint32_t>({1, 2, 3}));
    const StateCursor nested = session.checkpoint();
    CHECK(std::holds_alternative<RuntimeOutput>(session.decode(4)));
    CHECK(std::holds_alternative<std::monostate>(session.rollback(nested)));
    CHECK(session.token_history() == std::vector<uint32_t>({1, 2, 3}));
    CHECK(std::holds_alternative<std::monostate>(session.restore_state(std::get<std::vector<uint8_t>>(before))));
    CHECK(std::holds_alternative<CompatibilityReport>(session.rollback(nested)));
    const StateCursor rollback_cursor = session.checkpoint();
    CHECK(std::holds_alternative<RuntimeOutput>(session.decode(3)));
    CHECK(std::holds_alternative<std::monostate>(session.rollback(rollback_cursor)));
    CHECK(session.token_history() == std::vector<uint32_t>({1, 2}));
    auto after = session.save_state();
    CHECK(std::holds_alternative<std::vector<uint8_t>>(after));
    if (std::holds_alternative<std::vector<uint8_t>>(before) && std::holds_alternative<std::vector<uint8_t>>(after)) {
        CHECK(std::get<std::vector<uint8_t>>(before) == std::get<std::vector<uint8_t>>(after));
        std::vector<uint8_t> corrupt = std::get<std::vector<uint8_t>>(before);
        corrupt[196] = 1;
        CHECK(std::holds_alternative<CompatibilityReport>(session.restore_state(corrupt)));
        auto unchanged = session.save_state();
        CHECK(std::holds_alternative<std::vector<uint8_t>>(unchanged));
        if (std::holds_alternative<std::vector<uint8_t>>(unchanged)) {
            CHECK(std::get<std::vector<uint8_t>>(unchanged) == std::get<std::vector<uint8_t>>(before));
        }
    }
}

} // namespace

int main() {
    test_transactional_construction();
    test_public_qualification_gate();
    test_canonical_product_factory_gate();
    test_plan_dispatch_rejects_tampering();
    test_reference_axis_split();
    test_gate_b_detects_production_only_kernel_sabotage();
    test_cpu_gate_b();
    test_session_state_lifecycle();
    return test_summary("test_runtime_session");
}
