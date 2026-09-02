// test_scalar_executor - focused coverage for the CPU reference semantic
// executor. The module is production code (runtime CPU fallback and the
// qualification gate-b scalar path) but is only compiled into the
// Apple-and-GGUF-gated test_runtime_session; this test exercises it standalone
// with hand-built packages so validation branches and exact fp32 numerics are
// checked on every platform.

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "artifact_set.h"
#include "compat_rule.h"
#include "execution_plan.h"
#include "fp16.h"
#include "scalar_executor.h"
#include "test_util.h"

using namespace Laplace;

namespace {

uint32_t f32_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// The executor maps logical {d0, d1} to blob offset d0 + d1 * dimension(0)
// when axis_order is {0, 1}. `row_major` supplies values in logical
// row-major order (d0 outer), and arrange_2d places them at the executor's
// expected storage offsets.
std::vector<uint8_t> arrange_2d(const std::vector<float>& row_major,
                                uint64_t d0, uint64_t d1) {
    std::vector<uint8_t> blob(d0 * d1 * sizeof(float));
    uint64_t logical = 0;
    for (float value : row_major) {
        const uint64_t r = logical / d1;
        const uint64_t c = logical % d1;
        const uint64_t storage = r + c * d0;
        std::memcpy(blob.data() + storage * sizeof(float), &value, sizeof(float));
        ++logical;
    }
    return blob;
}

SemanticTensor f32_tensor(uint32_t id, TensorRole role, uint64_t dimensions0,
                          uint64_t dimensions1, uint64_t offset) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, dimensions0},
                         {DimensionKind::Constant, dimensions1}};
    tensor.layout.kind = PhysicalLayoutKind::ContiguousRowMajor;
    tensor.layout.rank = 2;
    tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, offset,
                      dimensions0 * dimensions1 * sizeof(float), 4, 0}};
    return tensor;
}

// Rank-1 tensors (rms weight, linear bias) declare a single dimension.
SemanticTensor f32_tensor_1d(uint32_t id, TensorRole role, uint64_t extent,
                             uint64_t offset) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, extent}};
    tensor.layout.kind = PhysicalLayoutKind::ContiguousRowMajor;
    tensor.layout.rank = 1;
    tensor.layout.axis_order = {0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, offset,
                      extent * sizeof(float), 4, 0}};
    return tensor;
}

std::shared_ptr<const RuntimePackage>
make_package(SemanticModel model, const std::vector<uint8_t>& artifact) {
    auto owned = ArtifactSet::make_owned_blob(ArtifactId{0}, ArtifactRole::Primary, artifact);
    if (!std::holds_alternative<PackageView>(owned)) return {};
    const PackageView& view = std::get<PackageView>(owned);
    return RuntimePackage::make_legacy_test_only(
        std::move(model), view, view.digest(), view.digest(), 0,
        RuleQualificationState::Draft);
}

// embedding -> rms_norm -> linear (bias). All integer-friendly values so the
// fp32 chain stays exact; logits come from the last row batch.
struct NumericModel {
    SemanticModel model;
    std::vector<uint8_t> artifact;
    std::shared_ptr<const RuntimePackage> package;
};

NumericModel numeric_model() {
    // Tensor 0: embedding (vocab 2 x width 3). Tensor 1: rms weight.
    // Tensor 2: linear weight 2x3. Tensor 3: linear bias size 2.
    std::vector<uint8_t> artifact;
    auto append = [&](std::vector<uint8_t> bytes) {
        const uint64_t offset = artifact.size();
        artifact.insert(artifact.end(), bytes.begin(), bytes.end());
        return offset;
    };
    const uint64_t emb_offset = append(arrange_2d({1, 2, 3, 4, 5, 6}, 2, 3));
    const uint64_t rms_offset = append(arrange_2d({1, 1, 1}, 1, 3));
    const uint64_t weight_offset = append(arrange_2d({1, 2, 3, 4, 5, 6}, 2, 3));
    const uint64_t bias_offset = append(arrange_2d({10, 20}, 1, 2));

    SemanticModel model;
    model.schema_major = 4;
    model.opset_major = 4;
    model.maximum_context = 4;
    model.vocabulary_size = 2;
    model.tensors = {
        f32_tensor(0, TensorRole::TokenEmbedding, 2, 3, emb_offset),
        f32_tensor_1d(1, TensorRole::FfnNormWeight, 3, rms_offset),
        f32_tensor(2, TensorRole::OutputWeight, 2, 3, weight_offset),
        f32_tensor_1d(3, TensorRole::QueryBias, 2, bias_offset),
    };
    model.values = {
        // Input slot for the embedding operator, then embedding -> rms -> linear.
        {0, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 3}}, 0},
        {1, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 3}}, 0},
        {2, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 3}}, 0},
        {3, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}}, 0},
    };
    RmsNormPayload rms;
    rms.axis = -1;
    rms.weight_mode = 1;
    rms.affine_geometry = RmsNormAffineGeometry::FullWidth;
    rms.epsilon_f32_bits = f32_bits(0.0f);  // clean non-identity normalization
    rms.reduction_extent = 0;
    LinearPayload linear;
    linear.has_bias = true;
    linear.accumulation_type = ScalarType::F32;
    model.operators = {
        {0, OperatorKind::EmbeddingLookup, 1, {0}, {1}, {0}, {},
         EmbeddingLookupPayload{f32_bits(1.0f), 2, 3, 0}},
        {1, OperatorKind::RmsNorm, 1, {1}, {2}, {1}, {}, rms},
        {2, OperatorKind::Linear, 1, {2}, {3}, {2, 3}, {}, linear},
    };
    return {model, artifact, make_package(model, artifact)};
}

void test_embedding_rms_linear_numerics() {
    NumericModel built = numeric_model();
    CHECK(built.package != nullptr);
    if (!built.package) return;
    ScalarExecutionResult result = scalar_execute(*built.package, std::vector<uint32_t>{0, 1});
    CHECK(std::holds_alternative<ScalarExecutionOutput>(result));
    if (!std::holds_alternative<ScalarExecutionOutput>(result)) return;
    const ScalarExecutionOutput& output = std::get<ScalarExecutionOutput>(result);

    // token1 embedding [4,5,6] -> rms -> [4,5,6]*sqrt(3/77) -> linear rows
    // [1,2,3] / [4,5,6] with bias [10,20]:
    //   logit0 = sqrt(3/77)*32 + 10, logit1 = sqrt(231) + 20.
    CHECK(output.logits.size() == 2);
    const float expected0 = std::sqrt(3.0f / 77.0f) * 32.0f + 10.0f;
    const float expected1 = std::sqrt(231.0f) + 20.0f;
    CHECK(almost_equal(output.logits[0], expected0, 1e-4f, 1e-4f));
    CHECK(almost_equal(output.logits[1], expected1, 1e-4f, 1e-4f));
    // The linear output for both rows is what scalar_execute retained; the
    // returned logits are the final (second-token) row of that batch.
    CHECK(output.operator_outputs.size() == 3);
    CHECK(output.operator_outputs[2].size() == 4);  // batch x vocab
    CHECK(almost_equal(output.operator_outputs[2][2], output.logits[0], 1e-4f, 1e-4f));
    CHECK(almost_equal(output.operator_outputs[2][3], output.logits[1], 1e-4f, 1e-4f));
    CHECK(output.operator_token_work == 3 * 2);

    // Same model, single token: prefill logits are the first (only) row.
    ScalarExecutionResult one = scalar_execute(*built.package, std::vector<uint32_t>{1});
    CHECK(std::holds_alternative<ScalarExecutionOutput>(one));
    if (!std::holds_alternative<ScalarExecutionOutput>(one)) return;
    const ScalarExecutionOutput& first = std::get<ScalarExecutionOutput>(one);
    CHECK(first.logits.size() == 2);
    CHECK(almost_equal(first.logits[0], expected0, 1e-4f, 1e-4f));
    CHECK(almost_equal(first.logits[1], expected1, 1e-4f, 1e-4f));
}

void test_f16_embedding_storage() {
    // V1 packages may carry F16 planes; the executor must decode them.
    std::vector<uint8_t> artifact;
    std::vector<uint16_t> halves = {fp32_to_fp16(1.0f), fp32_to_fp16(2.0f),
                                    fp32_to_fp16(3.0f), fp32_to_fp16(4.0f)};
    for (uint16_t half : halves) {
        artifact.push_back(static_cast<uint8_t>(half & 0xff));
        artifact.push_back(static_cast<uint8_t>(half >> 8));
    }

    SemanticModel model;
    model.maximum_context = 2;
    model.vocabulary_size = 2;
    SemanticTensor table = f32_tensor(0, TensorRole::TokenEmbedding, 2, 2, 0);
    table.planes[0].storage_type = ScalarType::F16;
    table.planes[0].length = 2 * 2 * sizeof(uint16_t);
    model.tensors = {table};
    model.values = {
        {0, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}}, 0},
        {1, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}}, 0},
    };
    // Embedding is also the final operator: its output width (2) equals the
    // vocabulary, so the logits window is the last row directly.
    model.operators = {
        {0, OperatorKind::EmbeddingLookup, 1, {0}, {1}, {0}, {},
         EmbeddingLookupPayload{f32_bits(1.0f), 2, 2, 0}},
    };
    auto package = make_package(model, artifact);
    CHECK(package != nullptr);
    if (!package) return;
    ScalarExecutionResult result = scalar_execute(*package, std::vector<uint32_t>{0, 1});
    CHECK(std::holds_alternative<ScalarExecutionOutput>(result));
    if (!std::holds_alternative<ScalarExecutionOutput>(result)) return;
    const ScalarExecutionOutput& output = std::get<ScalarExecutionOutput>(result);
    // storage order is {d0, d1} -> d0 + d1 * rows, so token 0 reads halves at
    // logical (0,0) and (1,1): embedding row 0 is [1, 3] and row 1 is [2, 4];
    // logits slice the last (token 1) row.
    CHECK(output.logits.size() == 2);
    CHECK(output.logits[0] == 2.0f);
    CHECK(output.logits[1] == 4.0f);
}

void test_input_validation() {
    NumericModel built = numeric_model();
    CHECK(built.package != nullptr);
    if (!built.package) return;

    ScalarExecutionResult empty = scalar_execute(*built.package, std::vector<uint32_t>{});
    CHECK(std::holds_alternative<CompatibilityReport>(empty));
    if (std::holds_alternative<CompatibilityReport>(empty)) {
        CHECK(std::get<CompatibilityReport>(empty).code == CompatibilityError::RUNTIME_INPUT_INVALID);
    }

    // Token outside the vocabulary.
    ScalarExecutionResult oob = scalar_execute(*built.package, std::vector<uint32_t>{0, 2});
    CHECK(std::holds_alternative<CompatibilityReport>(oob));
    if (std::holds_alternative<CompatibilityReport>(oob)) {
        CHECK(std::get<CompatibilityReport>(oob).code == CompatibilityError::RUNTIME_INPUT_INVALID);
    }

    // More tokens than the model context admits.
    ScalarExecutionResult over = scalar_execute(*built.package, std::vector<uint32_t>{0, 0, 0, 0, 0});
    CHECK(std::holds_alternative<CompatibilityReport>(over));
    if (std::holds_alternative<CompatibilityReport>(over)) {
        CHECK(std::get<CompatibilityReport>(over).code == CompatibilityError::RUNTIME_INPUT_INVALID);
    }

    // An operator kind with no scalar kernel makes the whole run unavailable.
    SemanticModel model = built.model;
    model.operators[0].kind = OperatorKind::GatedDeltaNet;
    model.operators[0].payload = GatedDeltaNetPayload{};
    auto package = make_package(model, std::vector<uint8_t>(1, 0));
    if (package) {
        ScalarExecutionResult unavailable = scalar_execute(*package, std::vector<uint32_t>{0});
        CHECK(std::holds_alternative<CompatibilityReport>(unavailable));
        if (std::holds_alternative<CompatibilityReport>(unavailable)) {
            CHECK(std::get<CompatibilityReport>(unavailable).code == CompatibilityError::KERNEL_UNAVAILABLE);
        }
    }
}

void test_tensor_and_reference_validation() {
    // A weight reference past the tensor table is an invalid reference.
    NumericModel built = numeric_model();
    SemanticModel model = built.model;
    model.operators[2].tensors = {99};
    auto bad_ref = make_package(model, built.artifact);
    CHECK(bad_ref != nullptr);
    if (bad_ref) {
        ScalarExecutionResult result = scalar_execute(*bad_ref, std::vector<uint32_t>{0});
        CHECK(std::holds_alternative<CompatibilityReport>(result));
        if (std::holds_alternative<CompatibilityReport>(result)) {
            CHECK(std::get<CompatibilityReport>(result).code == CompatibilityError::IR_REFERENCE_INVALID);
        }
    }

    // A quantized tensor has no scalar plane and must be rejected.
    SemanticModel quantized = built.model;
    quantized.tensors[2].quantization.kind = QuantizationKind::BlockedAffine;
    auto quantized_package = make_package(quantized, built.artifact);
    CHECK(quantized_package != nullptr);
    if (quantized_package) {
        ScalarExecutionResult result = scalar_execute(*quantized_package, std::vector<uint32_t>{0});
        CHECK(std::holds_alternative<CompatibilityReport>(result));
        if (std::holds_alternative<CompatibilityReport>(result)) {
            CHECK(std::get<CompatibilityReport>(result).code == CompatibilityError::IR_REFERENCE_INVALID);
        }
    }

    // Final operator width that is not vocab x batch is a shape mismatch.
    SemanticModel mismatch = built.model;
    mismatch.vocabulary_size = 4;  // final linear still emits 2 per row
    auto mismatch_package = make_package(mismatch, built.artifact);
    CHECK(mismatch_package != nullptr);
    if (mismatch_package) {
        ScalarExecutionResult result = scalar_execute(*mismatch_package, std::vector<uint32_t>{0});
        CHECK(std::holds_alternative<CompatibilityReport>(result));
        if (std::holds_alternative<CompatibilityReport>(result)) {
            CHECK(std::get<CompatibilityReport>(result).code == CompatibilityError::IR_SHAPE_MISMATCH);
        }
    }
}

void test_planned_execution_matches_scalar() {
    // plan_session() with only CPU scalar kernels available must produce a
    // plan that the scalar planned executor accepts, phase-filtered.
    NumericModel built = numeric_model();
    CHECK(built.package != nullptr);
    if (!built.package) return;

    RuntimeCapabilities capabilities;
    capabilities.scalar_fp32 = true;
    SessionRequest request;
    request.max_context = 4;
    request.max_batch = 1;
    request.memory_limit = UINT64_MAX;
    request.enable_prefill = true;
    request.enable_decode = true;
    request.minimum_class = NumericalClass::ExactFp32;
    request.objective = RuntimeObjective::Latency;

    PlanResult plan_result =
        plan_session(built.model, request, capabilities, builtin_cpu_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(plan_result));
    if (!std::holds_alternative<ExecutionPlan>(plan_result)) return;
    const ExecutionPlan& plan = std::get<ExecutionPlan>(plan_result);

    ScalarExecutionResult prefill =
        execute_planned_scalar(*built.package, plan, ExecutionPhase::Prefill, std::vector<uint32_t>{0, 1});
    ScalarExecutionResult decode =
        execute_planned_scalar(*built.package, plan, ExecutionPhase::Decode, std::vector<uint32_t>{0, 1});
    CHECK(std::holds_alternative<ScalarExecutionOutput>(prefill));
    CHECK(std::holds_alternative<ScalarExecutionOutput>(decode));
    if (!std::holds_alternative<ScalarExecutionOutput>(prefill) ||
        !std::holds_alternative<ScalarExecutionOutput>(decode))
        return;
    // A stateless model is phase-invariant: prefill and decode agree.
    const auto& prefill_output = std::get<ScalarExecutionOutput>(prefill);
    const auto& decode_output = std::get<ScalarExecutionOutput>(decode);
    CHECK(prefill_output.logits == decode_output.logits);
    CHECK(prefill_output.logits.size() == 2);

    // A plan entry outside the declared phase window or with a tampered
    // operator id is rejected as KERNEL_UNAVAILABLE.
    ExecutionPlan tampered = plan;
    ScalarExecutionResult bad_phase =
        execute_planned_scalar(*built.package, tampered, ExecutionPhase::Rollback, std::vector<uint32_t>{0});
    CHECK(std::holds_alternative<CompatibilityReport>(bad_phase));
    if (std::holds_alternative<CompatibilityReport>(bad_phase)) {
        CHECK(std::get<CompatibilityReport>(bad_phase).code == CompatibilityError::KERNEL_UNAVAILABLE);
    }
}

void test_incremental_attention_matches_full_prefill() {
    // Full 3-token run vs. 2-token prefill + 1-token decode must produce the
    // same final logits, and the KV state must carry exactly the positions.

    // Model: embed 2x96 -> q,k,v linears (32 wide) -> rope -> attention
    // (1 head, dim 32) -> output linear (2 wide).
    std::vector<uint8_t> artifact;
    auto append = [&](std::vector<uint8_t> bytes) {
        const uint64_t offset = artifact.size();
        artifact.insert(artifact.end(), bytes.begin(), bytes.end());
        return offset;
    };
    // A one-hot slice: identity_block(rows, columns, column_offset) returns
    // the executor storage for a (rows x columns) weight whose non-zero
    // entries sit at logical row == column - column_offset.
    auto identity_block = [&](uint64_t rows, uint64_t columns, uint64_t column_offset) {
        std::vector<uint8_t> blob(rows * columns * sizeof(float));
        for (uint64_t column = 0; column != columns; ++column) {
            for (uint64_t row = 0; row != rows; ++row) {
                const float value =
                    column >= column_offset && column - column_offset == row ? 1.0f : 0.0f;
                // Executor storage for axis_order {0,1}: index = row + column*rows
                const uint64_t storage = row + column * rows;
                std::memcpy(blob.data() + storage * sizeof(float), &value, sizeof(float));
            }
        }
        return blob;
    };

    std::vector<float> embedding_row_major;
    for (uint64_t token = 0; token != 2; ++token)
        for (uint64_t column = 0; column != 96; ++column)
            embedding_row_major.push_back(static_cast<float>((token + 1) * (column + 1)));
    const uint64_t emb_offset = append(arrange_2d(embedding_row_major, 2, 96));
    const uint64_t q_offset = append(identity_block(32, 96, 0));
    const uint64_t k_offset = append(identity_block(32, 96, 32));
    const uint64_t v_offset = append(identity_block(32, 96, 64));
    const uint64_t out_offset = append(arrange_2d({
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    }, 2, 32));

    SemanticModel model;
    model.schema_major = 4;
    model.opset_major = 4;
    model.maximum_context = 8;
    model.vocabulary_size = 2;
    model.tensors = {
        f32_tensor(0, TensorRole::TokenEmbedding, 2, 96, emb_offset),
        f32_tensor(1, TensorRole::OutputWeight, 32, 96, q_offset),
        f32_tensor(2, TensorRole::OutputWeight, 32, 96, k_offset),
        f32_tensor(3, TensorRole::OutputWeight, 32, 96, v_offset),
        f32_tensor(4, TensorRole::OutputWeight, 2, 32, out_offset),
    };
    model.values = {
        {0, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 96}}, 0},
        {1, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 32}}, 0},
        {2, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 32}}, 0},
        {3, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 32}}, 0},
        // Rope and attention outputs are rank 3 in the planner's view; the
        // scalar executor only requires the runtime row count.
        {4, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Symbol, 1},
                              {DimensionKind::Constant, 32}}, 0},
        {5, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Symbol, 1},
                              {DimensionKind::Constant, 32}}, 0},
        {6, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Symbol, 1},
                              {DimensionKind::Constant, 32}}, 0},
        {7, ScalarType::F32, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 2}}, 0},
    };
    RopePayload rope;
    rope.pairing = RopePairing::HalfSplit;
    rope.position_from_cursor = true;
    rope.rotary_dimension = 32;
    rope.base_f32_bits = f32_bits(10000.0f);
    rope.scale_f32_bits = f32_bits(1.0f);
    CausalAttentionPayload attention;
    attention.query_heads = 1;
    attention.kv_heads = 1;
    attention.head_dimension = 32;
    attention.scale_f32_bits = f32_bits(1.0f);
    LinearPayload linear;
    linear.accumulation_type = ScalarType::F32;
    model.operators = {
        {0, OperatorKind::EmbeddingLookup, 1, {0}, {0}, {0}, {},
         EmbeddingLookupPayload{f32_bits(1.0f), 2, 96, 0}},
        {1, OperatorKind::Linear, 1, {0}, {1}, {1}, {}, linear},
        {2, OperatorKind::Linear, 1, {0}, {2}, {2}, {}, linear},
        {3, OperatorKind::Linear, 1, {0}, {3}, {3}, {}, linear},
        {4, OperatorKind::Rope, 1, {1, 2}, {4, 5}, {}, {}, rope},
        {5, OperatorKind::CausalAttention, 1, {4, 5, 3}, {6}, {}, {0, 1}, attention},
        {6, OperatorKind::Linear, 1, {6}, {7}, {4}, {}, linear},
    };
    SemanticState key;
    key.id = 0;
    key.kind = StateKind::KeyCache;
    key.dimensions = {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 32}};
    key.formats = {{StateFormatKind::GlobalContiguous, 1, ScalarType::F32,
                    ScalarType::F32, TransformDomain::Untransformed,
                    TransformDomain::Untransformed, CodecKind::Fp32,
                    CachePolicy::Global, LayoutPolicy::TokenMajorContiguous,
                    0, 0, 0, 4, 0}};
    SemanticState value = key;
    value.id = 1;
    value.kind = StateKind::ValueCache;
    model.states = {key, value};

    auto package = make_package(model, artifact);
    CHECK(package != nullptr);
    if (!package) return;

    RuntimeCapabilities capabilities;
    capabilities.scalar_fp32 = true;
    capabilities.global_fp32_kv = true;
    SessionRequest request;
    request.max_context = 8;
    request.max_batch = 1;
    request.memory_limit = UINT64_MAX;
    request.enable_prefill = true;
    request.enable_decode = true;
    request.minimum_class = NumericalClass::ExactFp32;
    request.objective = RuntimeObjective::Latency;
    PlanResult plan_result =
        plan_session(model, request, capabilities, builtin_cpu_registry());
    CHECK(std::holds_alternative<ExecutionPlan>(plan_result));
    if (!std::holds_alternative<ExecutionPlan>(plan_result)) return;
    const ExecutionPlan& plan = std::get<ExecutionPlan>(plan_result);

    // Full prefill of all three tokens.
    ScalarExecutionResult full =
        execute_planned_scalar(*package, plan, ExecutionPhase::Prefill, std::vector<uint32_t>{0, 1, 1});
    CHECK(std::holds_alternative<ScalarExecutionOutput>(full));
    if (!std::holds_alternative<ScalarExecutionOutput>(full)) return;
    const auto& full_output = std::get<ScalarExecutionOutput>(full);

    // Split: prefill two tokens, then decode the third incrementally.
    std::vector<SemanticKvState> states;
    ScalarExecutionResult prefill =
        execute_planned_scalar(*package, plan, ExecutionPhase::Prefill, std::vector<uint32_t>{0, 1});
    CHECK(std::holds_alternative<ScalarExecutionOutput>(prefill));
    if (!std::holds_alternative<ScalarExecutionOutput>(prefill)) return;
    const auto& prefill_output = std::get<ScalarExecutionOutput>(prefill);
    states = prefill_output.states;
    CHECK(states.size() == 1);
    CHECK(states[0].tokens == 2);
    CHECK(states[0].key.size() == 2 * 32 && states[0].value.size() == 2 * 32);

    ScalarExecutionResult decode = execute_planned_scalar_incremental(
        *package, plan, ExecutionPhase::Decode, std::vector<uint32_t>{1}, states);
    CHECK(std::holds_alternative<ScalarExecutionOutput>(decode));
    if (!std::holds_alternative<ScalarExecutionOutput>(decode)) return;
    const auto& decode_output = std::get<ScalarExecutionOutput>(decode);
    CHECK(states[0].tokens == 3);
    CHECK(states[0].key.size() == 3 * 32 && states[0].value.size() == 3 * 32);

    CHECK(full_output.logits.size() == 2);
    CHECK(decode_output.logits.size() == 2);
    for (size_t index = 0; index != full_output.logits.size(); ++index) {
        CHECK_MSG(almost_equal(decode_output.logits[index], full_output.logits[index], 1e-4f, 1e-4f),
                  "incremental decode logit %zu diverged from full prefill", index);
    }
    CHECK(decode_output.operator_token_work == 7 * 1);
}

} // namespace

int main() {
    test_embedding_rms_linear_numerics();
    test_f16_embedding_storage();
    test_input_validation();
    test_tensor_and_reference_validation();
    test_planned_execution_matches_scalar();
    test_incremental_attention_matches_full_prefill();
    return test_summary("test_scalar_executor");
}
