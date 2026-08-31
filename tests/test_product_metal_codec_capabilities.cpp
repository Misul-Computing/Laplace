#define LAPLACE_RUNTIME_PACKAGE_TESTING 1
#define LAPLACE_METAL_TESTING 1
#define main product_fixture_helper_main
#include "test_runtime_session_product.cpp"
#undef main

#include <cstdint>
#include <string_view>
#include <variant>

#include "product_metal_codec_capabilities.h"
#include "product_metal_execution.h"
#include "test_util.h"

using namespace Laplace;

namespace {

const MetalPipelineRecipe* recipe_named(
    const ProductMetalExecutionContract& contract, std::string_view name,
    std::span<const MetalFunctionConstant> constants = {}) {
    const auto recipes = contract.recipes();
    const auto found = std::find_if(
        recipes.begin(), recipes.end(), [&](const MetalPipelineRecipe& recipe) {
            return recipe.function_name == name &&
                   std::equal(recipe.function_constants.begin(),
                              recipe.function_constants.end(), constants.begin(),
                              constants.end());
        });
    return found == recipes.end() ? nullptr : &*found;
}

std::vector<std::string> invocation_names(
    const ProductMetalExecutionContract& contract,
    ProductMetalProgramKind kind) {
    const ProductMetalProgramBinding* binding = contract.find(kind);
    if (!binding) return {};
    const MetalTokProgramRange* range = contract.range(binding->id);
    if (!range) return {};
    const auto flattened = contract.flattened_invocation_recipe_indices();
    const auto recipes = contract.recipes();
    if (range->first_invocation > flattened.size() ||
        range->invocation_count > flattened.size() - range->first_invocation)
        return {};
    std::vector<std::string> result;
    result.reserve(range->invocation_count);
    for (uint32_t offset = 0; offset < range->invocation_count; ++offset) {
        const uint32_t recipe_index =
            flattened[range->first_invocation + offset];
        if (recipe_index >= recipes.size()) return {};
        result.push_back(recipes[recipe_index].function_name);
    }
    return result;
}

SemanticDispatchRequest decode_request() {
    SemanticDispatchRequest request;
    request.phase = ExecutionPhase::Decode;
    request.batch_rows = 1;
    request.numerical_class = NumericalClass::ExactFp32;
    request.include_greedy_sampler = false;
    request.include_speculative = false;
    return request;
}

void test_product_compiler_covers_tensor_occurrences() {
    Storage storage;
    ProductFixture fixture = make_closed_route_fixture(
        make_dense_model(storage), std::move(storage));
    CHECK(fixture.closed_route != nullptr);
    if (!fixture.closed_route) {
        cleanup(fixture);
        return;
    }

    const auto preflight = preflight_codec_bindings(*fixture.closed_route);
    CHECK(std::holds_alternative<ResolvedCodecBindings>(preflight));
    if (!std::holds_alternative<ResolvedCodecBindings>(preflight)) {
        cleanup(fixture);
        return;
    }
    const ResolvedCodecBindings& bindings =
        std::get<ResolvedCodecBindings>(preflight);
    SemanticDispatchProgramResult built = build_semantic_dispatch_program(
        fixture.model, decode_request());
    CHECK(std::holds_alternative<SemanticDispatchProgram>(built));
    if (!std::holds_alternative<SemanticDispatchProgram>(built)) {
        cleanup(fixture);
        return;
    }
    const SemanticDispatchProgram program =
        std::get<SemanticDispatchProgram>(std::move(built));

    const auto compiled = compile_product_metal_codec_capabilities(
        *fixture.closed_route, bindings, {&program, 1}, fixture.model);
    CHECK(std::holds_alternative<ProductMetalCodecCapabilityCompilation>(compiled));
    if (std::holds_alternative<ProductMetalCodecCapabilityCompilation>(compiled)) {
        const auto& result =
            std::get<ProductMetalCodecCapabilityCompilation>(compiled);
        size_t tensor_occurrences = 0;
        for (const SemanticOperator& operation : fixture.model.operators)
            tensor_occurrences += operation.tensors.size();
        CHECK(result.records().size() == tensor_occurrences);
        CHECK(result.registry().size() != 0);
        for (const ProductMetalCodecCapabilityRecord& record : result.records()) {
            CHECK(record.requirement.operation_abi != 0);
            CHECK(record.requirement.phase ==
                  static_cast<uint16_t>(ExecutionPhase::Decode));
            CHECK(record.lowering.identity != 0);
            CHECK(record.capability_digest != MetalCodecCapabilityDigest{});
            CHECK(record.provenance.source_identity != CodecCertificateDigest{});
        }
    }
    cleanup(fixture);
}

void test_product_compiler_rejects_tampered_program() {
    Storage storage;
    ProductFixture fixture = make_closed_route_fixture(
        make_dense_model(storage), std::move(storage));
    CHECK(fixture.closed_route != nullptr);
    if (!fixture.closed_route) {
        cleanup(fixture);
        return;
    }
    const auto preflight = preflight_codec_bindings(*fixture.closed_route);
    CHECK(std::holds_alternative<ResolvedCodecBindings>(preflight));
    if (!std::holds_alternative<ResolvedCodecBindings>(preflight)) {
        cleanup(fixture);
        return;
    }
    const ResolvedCodecBindings& bindings =
        std::get<ResolvedCodecBindings>(preflight);
    SemanticDispatchProgramResult built = build_semantic_dispatch_program(
        fixture.model, decode_request());
    CHECK(std::holds_alternative<SemanticDispatchProgram>(built));
    if (!std::holds_alternative<SemanticDispatchProgram>(built)) {
        cleanup(fixture);
        return;
    }
    SemanticDispatchProgram tampered =
        std::get<SemanticDispatchProgram>(std::move(built));
    if (!tampered.steps.empty()) tampered.steps[0].operation = OperatorKind::Add;
    const auto compiled = compile_product_metal_codec_capabilities(
        *fixture.closed_route, bindings, {&tampered, 1}, fixture.model);
    CHECK(std::holds_alternative<CompatibilityReport>(compiled));
    if (const auto* report = std::get_if<CompatibilityReport>(&compiled))
        CHECK(report->code == CompatibilityError::IR_REFERENCE_INVALID);
    cleanup(fixture);
}

void test_product_compiler_reuses_recipe_across_programs() {
    Storage storage;
    ProductFixture fixture = make_closed_route_fixture(
        make_dense_model(storage), std::move(storage));
    CHECK(fixture.closed_route != nullptr);
    if (!fixture.closed_route) {
        cleanup(fixture);
        return;
    }
    const auto preflight = preflight_codec_bindings(*fixture.closed_route);
    CHECK(std::holds_alternative<ResolvedCodecBindings>(preflight));
    if (!std::holds_alternative<ResolvedCodecBindings>(preflight)) {
        cleanup(fixture);
        return;
    }
    const ResolvedCodecBindings& bindings =
        std::get<ResolvedCodecBindings>(preflight);
    SemanticDispatchProgramResult built = build_semantic_dispatch_program(
        fixture.model, decode_request());
    CHECK(std::holds_alternative<SemanticDispatchProgram>(built));
    if (!std::holds_alternative<SemanticDispatchProgram>(built)) {
        cleanup(fixture);
        return;
    }
    const SemanticDispatchProgram program =
        std::get<SemanticDispatchProgram>(std::move(built));
    const std::array<SemanticDispatchProgram, 2> programs = {program, program};
    const auto compiled = compile_product_metal_codec_capabilities(
        *fixture.closed_route, bindings, programs, fixture.model);
    CHECK(std::holds_alternative<ProductMetalCodecCapabilityCompilation>(compiled));
    if (std::holds_alternative<ProductMetalCodecCapabilityCompilation>(compiled)) {
        const auto& result =
            std::get<ProductMetalCodecCapabilityCompilation>(compiled);
        CHECK(result.records().size() == 24);
        CHECK(result.registry().size() < result.records().size());
        CHECK(result.records()[0].capability_digest ==
              result.records()[12].capability_digest);
    }
    cleanup(fixture);
}

void test_product_execution_contract_owns_exact_program_ranges() {
    Storage storage;
    ProductFixture fixture = make_closed_route_fixture(
        make_dense_model(storage), std::move(storage));
    CHECK(fixture.closed_route != nullptr);
    if (!fixture.closed_route) {
        cleanup(fixture);
        return;
    }

    ProductMetalExecutionContractResult built =
        compile_product_metal_execution_contract(*fixture.closed_route, request());
    CHECK(std::holds_alternative<ProductMetalExecutionContract>(built));
    if (!std::holds_alternative<ProductMetalExecutionContract>(built)) {
        if (const auto* report = std::get_if<CompatibilityReport>(&built)) {
            CHECK_MSG(false, "product Metal execution contract failed: code=%u detail=%s",
                      static_cast<unsigned>(report->code), report->detail.c_str());
        }
        cleanup(fixture);
        return;
    }

    const ProductMetalExecutionContract& contract =
        std::get<ProductMetalExecutionContract>(built);
    CHECK(contract.semantic_programs().size() == 4);
    CHECK(contract.structural_programs().size() == 4);
    CHECK(contract.program_bindings().size() == 5);
    CHECK(contract.program_ranges().size() == 5);
    CHECK(!contract.recipes().empty());
    CHECK(!contract.flattened_invocation_recipe_indices().empty());
    CHECK(contract.invocation_authorities().size() ==
          contract.flattened_invocation_recipe_indices().size());

    std::vector<Sha256Digest> authority_digests;
    for (size_t index = 0; index < contract.invocation_authorities().size(); ++index) {
        const ProductMetalInvocationAuthority& authority =
            contract.invocation_authorities()[index];
        CHECK(authority.version == 1);
        CHECK(authority.invocation_ordinal == index);
        CHECK(authority.recipe_index ==
              contract.flattened_invocation_recipe_indices()[index]);
        CHECK(authority.batch_rows == 1);
        CHECK(authority.row_index == 0);
        CHECK(authority.row_count == 1);
        CHECK(authority.bound_program_digest != Sha256Digest{});
        CHECK(authority.invocation_digest != Sha256Digest{});
        CHECK(std::find(authority_digests.begin(), authority_digests.end(),
                        authority.invocation_digest) == authority_digests.end());
        authority_digests.push_back(authority.invocation_digest);
    }

    std::vector<uint32_t> ids;
    for (const ProductMetalProgramBinding& binding : contract.program_bindings()) {
        CHECK(binding.id != UINT32_MAX);
        CHECK(std::find(ids.begin(), ids.end(), binding.id) == ids.end());
        ids.push_back(binding.id);
        const auto range = std::find_if(
            contract.program_ranges().begin(), contract.program_ranges().end(),
            [&](const MetalTokProgramRange& candidate) {
                return candidate.id == binding.id;
            });
        CHECK(range != contract.program_ranges().end());
        if (range == contract.program_ranges().end()) continue;
        CHECK(range->invocation_count != 0);
        CHECK(range->first_invocation <=
              contract.flattened_invocation_recipe_indices().size());
        CHECK(range->invocation_count <=
              contract.flattened_invocation_recipe_indices().size() -
                  range->first_invocation);
    }
    for (uint32_t recipe_index :
         contract.flattened_invocation_recipe_indices()) {
        CHECK(recipe_index < contract.recipes().size());
    }

    const ProductMetalProgramBinding* prefill_logits =
        contract.find(ProductMetalProgramKind::PrefillLogits);
    const ProductMetalProgramBinding* prefill_state =
        contract.find(ProductMetalProgramKind::PrefillStateOnly);
    CHECK(prefill_logits != nullptr);
    CHECK(prefill_state != nullptr);
    if (prefill_logits && prefill_state) {
        const MetalTokProgramRange* logits_range =
            contract.range(prefill_logits->id);
        const MetalTokProgramRange* state_range =
            contract.range(prefill_state->id);
        CHECK(logits_range != nullptr);
        CHECK(state_range != nullptr);
        if (logits_range && state_range) {
            CHECK(state_range->invocation_count < logits_range->invocation_count);
        }
    }
    cleanup(fixture);
}

void test_product_execution_contract_expands_exact_two_row_invocations() {
    Storage storage;
    ProductFixture fixture = make_closed_route_fixture(
        make_dense_model(storage), std::move(storage));
    CHECK(fixture.closed_route != nullptr);
    if (!fixture.closed_route) {
        cleanup(fixture);
        return;
    }

    SessionRequest two_row = request();
    two_row.max_batch = 2;
    ProductMetalExecutionContractResult built =
        compile_product_metal_execution_contract(*fixture.closed_route, two_row);
    if (const auto* report = std::get_if<CompatibilityReport>(&built)) {
        CHECK_MSG(false,
                  "two-row product Metal contract failed: code=%u operator=%u detail=%s",
                  static_cast<unsigned>(report->code), report->operator_id,
                  report->detail.c_str());
    }
    CHECK(std::holds_alternative<ProductMetalExecutionContract>(built));
    if (!std::holds_alternative<ProductMetalExecutionContract>(built)) {
        cleanup(fixture);
        return;
    }

    const ProductMetalExecutionContract& contract =
        std::get<ProductMetalExecutionContract>(built);
    CHECK(contract.semantic_programs().size() == 6);
    CHECK(contract.structural_programs().size() == 6);
    CHECK(contract.program_bindings().size() == 7);
    CHECK(contract.program_ranges().size() == 7);

    const std::vector<std::string> expected = {
        "embedding_f16", "embedding_f16",
        "rmsnorm_f32", "rmsnorm_f32",
        "prefill_f16_rows", "prefill_f16_rows", "prefill_f16_rows",
        "rope_f32", "rope_f32", "kv_write", "kv_write",
        "rope_f32", "rope_f32", "kv_write", "kv_write",
        "attn_decode", "attn_decode", "prefill_f16_rows",
        "vec_add", "rmsnorm_f32", "vec_add", "rmsnorm_f32",
        "prefill_f16_rows", "prefill_f16_rows",
        "act_glu", "act_glu", "prefill_f16_rows",
        "vec_add", "vec_add",
        "rmsnorm_f32", "gemv",
    };
    const std::vector<std::string> batch2_logits = invocation_names(
        contract, ProductMetalProgramKind::PrefillBatch2Logits);
    const std::vector<std::string> batch2_sample = invocation_names(
        contract, ProductMetalProgramKind::PrefillBatch2GreedySample);
    CHECK(batch2_logits == expected);
    CHECK(batch2_sample.size() == expected.size() + 1);
    if (batch2_sample.size() == expected.size() + 1) {
        CHECK(std::equal(expected.begin(), expected.end(),
                         batch2_sample.begin()));
        CHECK(batch2_sample.back() == "sampler_greedy_f32");
    }
    const ProductMetalProgramBinding* batch2_binding =
        contract.find(ProductMetalProgramKind::PrefillBatch2Logits);
    CHECK(batch2_binding != nullptr);
    if (batch2_binding) {
        const MetalTokProgramRange* range = contract.range(batch2_binding->id);
        CHECK(range != nullptr);
        CHECK(batch2_binding->semantic_program_index <
              contract.semantic_programs().size());
        if (range && batch2_binding->semantic_program_index <
                         contract.semantic_programs().size()) {
            const SemanticDispatchProgram& semantic =
                contract.semantic_programs()[batch2_binding->semantic_program_index];
            CHECK(semantic.output_binding.has_value());
            CHECK(range->batch_rows == 2);
            CHECK(range->selected_output_row == 1);
            CHECK(range->semantic_program_digest ==
                  semantic.program_digest.bytes);
            const auto authorities = contract.invocation_authorities();
            CHECK(range->invocation_count == expected.size());
            if (range->invocation_count == expected.size() &&
                range->first_invocation + range->invocation_count <=
                    authorities.size()) {
                const auto authority = [&](uint32_t offset)
                    -> const ProductMetalInvocationAuthority& {
                    return authorities[range->first_invocation + offset];
                };
                CHECK(authority(0).row_index == 0 && authority(0).row_count == 1);
                CHECK(authority(1).row_index == 1 && authority(1).row_count == 1);
                CHECK(authority(2).row_index == 0 && authority(2).row_count == 1);
                CHECK(authority(3).row_index == 1 && authority(3).row_count == 1);
                CHECK(authority(4).row_index == UINT32_MAX &&
                      authority(4).row_count == 2);
                CHECK(authority(5).row_index == UINT32_MAX &&
                      authority(5).row_count == 2);
                CHECK(authority(6).row_index == UINT32_MAX &&
                      authority(6).row_count == 2);
                CHECK(authority(29).row_index == 1 && authority(29).row_count == 1);
                CHECK(authority(30).row_index == 1 && authority(30).row_count == 1);
            }
        }
    }

    const std::vector<std::string> state_only = invocation_names(
        contract, ProductMetalProgramKind::PrefillStateOnly);
    CHECK(state_only.size() == 18);
    if (!state_only.empty()) CHECK(state_only.front() == "embedding_f16");
    const ProductMetalProgramBinding* state_binding =
        contract.find(ProductMetalProgramKind::PrefillStateOnly);
    CHECK(state_binding != nullptr);
    if (state_binding) {
        const MetalTokProgramRange* range = contract.range(state_binding->id);
        CHECK(range != nullptr);
        if (range) CHECK(range->selected_output_row == UINT32_MAX);
    }
    cleanup(fixture);
}

void test_grouped_affine_product_contract_uses_codec_units() {
    Storage storage;
    ProductFixture fixture = make_closed_route_fixture(
        make_dense_model(storage, 512, true), std::move(storage));
    CHECK(fixture.closed_route != nullptr);
    if (!fixture.closed_route) {
        cleanup(fixture);
        return;
    }

    const auto preflight = preflight_codec_bindings(*fixture.closed_route);
    CHECK(std::holds_alternative<ResolvedCodecBindings>(preflight));
    if (const auto* report = std::get_if<CompatibilityReport>(&preflight)) {
        CHECK_MSG(false, "grouped-affine codec preflight failed: code=%u operator=%u tensor=%u detail=%s",
                  static_cast<unsigned>(report->code), report->operator_id,
                  report->tensor_id, report->detail.c_str());
    }

    ProductMetalExecutionContractResult built =
        compile_product_metal_execution_contract(*fixture.closed_route, request());
    CHECK(std::holds_alternative<ProductMetalExecutionContract>(built));
    if (const auto* report = std::get_if<CompatibilityReport>(&built)) {
        CHECK_MSG(false, "grouped-affine product contract failed: code=%u operator=%u tensor=%u detail=%s",
                  static_cast<unsigned>(report->code), report->operator_id,
                  report->tensor_id, report->detail.c_str());
    }
    if (const auto* contract =
            std::get_if<ProductMetalExecutionContract>(&built)) {
        const std::array<MetalFunctionConstant, 5> constants = {
            MetalFunctionConstant{2, MetalFunctionConstantType::UInt32, 4},
            MetalFunctionConstant{3, MetalFunctionConstantType::UInt32, 2},
            MetalFunctionConstant{4, MetalFunctionConstantType::Bool, 0},
            MetalFunctionConstant{5, MetalFunctionConstantType::Bool, 0},
            MetalFunctionConstant{6, MetalFunctionConstantType::UInt32, 16},
        };
        const MetalPipelineRecipe* affine = recipe_named(
            *contract, "gemv_affine_u2_256", constants);
        CHECK(affine != nullptr);
        if (affine) {
            CHECK(affine->dispatch.min_threads_per_threadgroup == 64);
            CHECK(affine->dispatch.required_simdgroups == 2);
        }
    }
    cleanup(fixture);
}

void test_product_gemv_recipes_match_leased_binders() {
    Storage dense_storage;
    ProductFixture dense = make_closed_route_fixture(
        make_dense_model(dense_storage), std::move(dense_storage));
    CHECK(dense.closed_route != nullptr);
    if (dense.closed_route) {
        ProductMetalExecutionContractResult built =
            compile_product_metal_execution_contract(*dense.closed_route,
                                                     request());
        CHECK(std::holds_alternative<ProductMetalExecutionContract>(built));
        if (const auto* contract =
                std::get_if<ProductMetalExecutionContract>(&built)) {
            const std::array<MetalFunctionConstant, 1> f16 = {
                MetalFunctionConstant{0, MetalFunctionConstantType::Int32, 1},
            };
            const MetalPipelineRecipe* gemv = recipe_named(*contract, "gemv", f16);
            CHECK(gemv != nullptr);
            if (gemv)
                CHECK(gemv->dispatch.min_threads_per_threadgroup == 32);
        }
    }
    cleanup(dense);

}

void test_grouped_rms_recipe_matches_leased_binder() {
    const MetalPipelineRecipe recipe =
        structural_metal_primitive_recipe_for_testing(
            StructuralMetalPrimitive::RmsNormRowsF32);
    CHECK(recipe.function_name == "rmsnorm_rows_f32");
    CHECK(recipe.function_constants.empty());
    CHECK(recipe.dispatch.min_threads_per_threadgroup == 64);
    CHECK(recipe.dispatch.max_threads_per_threadgroup == 1024);
}

void test_recurrent_product_contract_accepts_bound_f32_state_tensors() {
    Storage storage;
    ProductFixture fixture = make_closed_route_fixture(
        make_recurrent_model(storage), std::move(storage));
    CHECK(fixture.closed_route != nullptr);
    if (!fixture.closed_route) {
        cleanup(fixture);
        return;
    }
    ProductMetalExecutionContractResult built =
        compile_product_metal_execution_contract(*fixture.closed_route,
                                                 request());
    if (const auto* report = std::get_if<CompatibilityReport>(&built))
        CHECK_MSG(false, "recurrent product contract failed: code=%u operator=%u tensor=%u detail=%s",
                  static_cast<unsigned>(report->code), report->operator_id,
                  report->tensor_id, report->detail.c_str());
    CHECK(std::holds_alternative<ProductMetalExecutionContract>(built));
    cleanup(fixture);
}

} // namespace

int main() {
    test_product_compiler_covers_tensor_occurrences();
    test_product_compiler_rejects_tampered_program();
    test_product_compiler_reuses_recipe_across_programs();
    test_product_execution_contract_owns_exact_program_ranges();
    test_product_execution_contract_expands_exact_two_row_invocations();
    test_grouped_affine_product_contract_uses_codec_units();
    test_product_gemv_recipes_match_leased_binders();
    test_grouped_rms_recipe_matches_leased_binder();
    test_recurrent_product_contract_accepts_bound_f32_state_tensors();
    return test_summary("test_product_metal_codec_capabilities");
}
