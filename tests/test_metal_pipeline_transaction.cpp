#include "metal_pipeline_transaction.h"
#include "matmul.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "test_util.h"

using namespace Laplace;

static_assert(std::is_move_constructible_v<MetalPipelineTransaction>);
static_assert(std::is_move_assignable_v<MetalPipelineTransaction>);
static_assert(!std::is_copy_constructible_v<MetalPipelineTransaction>);
static_assert(!std::is_copy_assignable_v<MetalPipelineTransaction>);
static_assert(!std::is_aggregate_v<MetalPipelineTransaction>);
static_assert(std::is_move_constructible_v<MetalPipelineLease>);
static_assert(std::is_move_assignable_v<MetalPipelineLease>);
static_assert(!std::is_copy_constructible_v<MetalPipelineLease>);
static_assert(!std::is_copy_assignable_v<MetalPipelineLease>);
static_assert(!std::is_aggregate_v<MetalPipelineLease>);

namespace {

void append_u16(std::vector<uint8_t>& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8u));
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

std::array<uint8_t, 32> invocation_digest(
    const MetalTokInvocationAuthority& authority) {
    std::vector<uint8_t> bytes = {'L', 'P', 'M', 'I', 'N', 'V', 1, 0};
    append_u16(bytes, authority.version);
    append_u32(bytes, authority.program_id);
    append_u32(bytes, authority.invocation_ordinal);
    append_u32(bytes, authority.group_ordinal);
    append_u16(bytes, authority.group_kind);
    append_u16(bytes, authority.group_shape);
    append_u16(bytes, authority.primitive);
    append_u32(bytes, authority.primitive_order);
    append_u32(bytes, authority.recipe_index);
    append_u32(bytes, authority.batch_rows);
    append_u32(bytes, authority.row_index);
    append_u32(bytes, authority.row_count);
    bytes.insert(bytes.end(), authority.semantic_program_digest.begin(),
                 authority.semantic_program_digest.end());
    bytes.insert(bytes.end(), authority.bound_program_digest.begin(),
                 authority.bound_program_digest.end());
    std::array<uint8_t, 32> result{};
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), result.data());
    return result;
}

std::vector<MetalTokInvocationAuthority> invocation_authorities(
    std::span<const uint32_t> flattened,
    std::span<const MetalTokProgramRange> ranges) {
    std::vector<MetalTokInvocationAuthority> result;
    result.reserve(flattened.size());
    for (const MetalTokProgramRange& range : ranges) {
        for (uint32_t offset = 0; offset != range.invocation_count; ++offset) {
            const uint32_t ordinal = range.first_invocation + offset;
            MetalTokInvocationAuthority authority;
            authority.group_kind = 1;
            authority.group_shape = 1;
            authority.primitive = 11;
            authority.program_id = range.id;
            authority.invocation_ordinal = ordinal;
            authority.group_ordinal = 0;
            authority.primitive_order = offset;
            authority.recipe_index = flattened[ordinal];
            authority.pipeline_slot = flattened[ordinal];
            authority.batch_rows = range.batch_rows;
            authority.row_index = 0;
            authority.row_count = 1;
            authority.semantic_program_digest = range.semantic_program_digest;
            authority.bound_program_digest.fill(
                static_cast<uint8_t>(0x40u + range.id));
            authority.invocation_digest = invocation_digest(authority);
            result.push_back(authority);
        }
    }
    return result;
}

constexpr char kLibrary[] = R"metal(
#include <metal_stdlib>
using namespace metal;
constant bool use_first [[function_constant(0)]];
constant uint tile_mode [[function_constant(1)]];
kernel void first_kernel(uint tid [[thread_position_in_grid]]) {
    if (use_first && tile_mode == 7 && tid == 0) { }
}
kernel void second_kernel(uint tid [[thread_position_in_grid]]) {
    if (tid == 0) { }
}
kernel void apply_down_scale(uint tid [[thread_position_in_grid]]) {
    if (tid == 0) { }
}
kernel void moe_combine(uint tid [[thread_position_in_grid]]) {
    if (tid == 0) { }
}
kernel void rmsnorm_f32(uint tid [[thread_position_in_grid]]) {
    if (tid == 0) { }
}
kernel void gemv_q2k(uint tid [[thread_position_in_grid]]) {
    if (tid == 0) { }
}
kernel void dnet_conv_silu(uint tid [[thread_position_in_grid]]) {
    if (tid == 0) { }
}
kernel void dnet_l2_qk(uint tid [[thread_position_in_grid]]) {
    if (tid == 0) { }
}
kernel void dnet_update(uint tid [[thread_position_in_grid]]) {
    if (tid == 0) { }
}
kernel void vec_add(uint tid [[thread_position_in_grid]]) {
    if (tid == 0) { }
}
kernel void act_glu(uint tid [[thread_position_in_grid]]) {
    if (tid == 0) { }
}
kernel void rope_f32(uint tid [[thread_position_in_grid]]) {
    if (tid == 0) { }
}
kernel void kv_write(uint tid [[thread_position_in_grid]]) {
    if (tid == 0) { }
}
kernel void attn_decode(uint tid [[thread_position_in_grid]]) {
    if (tid == 0) { }
}
kernel void prefill_f16_rows(uint tid [[thread_position_in_grid]]) {
    if (tid == 0) { }
}
)metal";

constexpr char kLibraryA[] = R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void library_a_kernel(uint tid [[thread_position_in_grid]]) {
    if (tid == 0) { }
}
)metal";

constexpr char kLibraryB[] = R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void library_b_kernel(uint tid [[thread_position_in_grid]]) {
    if (tid == 0) { }
}
)metal";

std::span<const uint8_t> library_bytes() {
    return {reinterpret_cast<const uint8_t*>(kLibrary), sizeof(kLibrary) - 1};
}

template <size_t N>
std::span<const uint8_t> bytes_of(const char (&source)[N]) {
    return {reinterpret_cast<const uint8_t*>(source), N - 1};
}

MetalPipelineDigest digest_text(std::string_view text) {
    return metal_pipeline_digest(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(text.data()), text.size()));
}

MetalPipelineRecipe recipe(std::string_view key, std::string_view function,
                           bool with_constant = false) {
    MetalPipelineRecipe result;
    result.normalized_requirement_digest = digest_text(key);
    result.function_name = std::string(function);
    if (with_constant)
        result.function_constants = {
            {0, MetalFunctionConstantType::Bool, 1},
            {1, MetalFunctionConstantType::UInt32, 7},
        };
    result.library_source_digest = metal_pipeline_digest(library_bytes());
    result.dispatch = {1, 1024, 1, 0, 0};
    return result;
}

MetalPipelineRecipe recipe_for(std::string_view key, std::string_view function,
                               std::span<const uint8_t> source) {
    MetalPipelineRecipe result = recipe(key, function);
    result.library_source_digest = metal_pipeline_digest(source);
    return result;
}

MetalPipelineLibrarySource library_source(
    std::span<const uint8_t> bytes, bool fast_math_enabled = true) {
    MetalPipelineLibrarySource result;
    result.bytes = bytes;
    result.source_digest = metal_pipeline_digest(bytes);
    result.compile_contract.fast_math_enabled = fast_math_enabled;
    return result;
}

bool is_error(const MetalPipelineTransactionResult& result,
              MetalPipelineTransactionError error) {
    const auto* failure = std::get_if<MetalPipelineTransactionFailure>(&result);
    return failure != nullptr && failure->code == error;
}

void test_contract_rejection() {
    constexpr MetalPipelineDigest abc_sha256 = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
    };
    CHECK(digest_text("abc") == abc_sha256);

    MetalPipelineRecipe valid = recipe("contract", "second_kernel");
    const std::array<MetalPipelineRecipe, 1> one = {valid};

    const auto empty = build_metal_pipeline_transaction({}, library_bytes());
    CHECK(is_error(empty, MetalPipelineTransactionError::InvalidRequest));

    valid.normalized_requirement_digest = {};
    const auto zero_requirement = build_metal_pipeline_transaction(
        std::span<const MetalPipelineRecipe>(
            &valid, 1), library_bytes());
    CHECK(is_error(zero_requirement, MetalPipelineTransactionError::InvalidRequest));

    valid = recipe("contract-zero-source", "second_kernel");
    valid.library_source_digest = {};
    const auto zero_source = build_metal_pipeline_transaction(
        std::span<const MetalPipelineRecipe>(&valid, 1), library_bytes());
    CHECK(is_error(zero_source, MetalPipelineTransactionError::InvalidRequest));

    valid = recipe("contract-zero-compile", "second_kernel");
    auto malformed_compile = library_source(library_bytes());
    malformed_compile.compile_contract.version = 0;
    const std::array<MetalPipelineLibrarySource, 1> malformed_sources = {
        malformed_compile};
    const auto malformed = build_metal_pipeline_transaction(
        std::span<const MetalPipelineRecipe>(&valid, 1), malformed_sources);
    CHECK(is_error(malformed, MetalPipelineTransactionError::InvalidRequest));

    valid = recipe("contract-duplicate-constant", "first_kernel", true);
    valid.function_constants.push_back(
        {0, MetalFunctionConstantType::Bool, 0});
    const auto duplicate_constant = build_metal_pipeline_transaction(
        std::span<const MetalPipelineRecipe>(&valid, 1), library_bytes());
    CHECK(is_error(duplicate_constant, MetalPipelineTransactionError::InvalidRequest));

    valid = recipe("contract-bad-limits", "second_kernel");
    valid.dispatch.min_threads_per_threadgroup = 1024;
    valid.dispatch.max_threads_per_threadgroup = 1;
    const auto bad_limits = build_metal_pipeline_transaction(
        std::span<const MetalPipelineRecipe>(&valid, 1), library_bytes());
    CHECK(is_error(bad_limits, MetalPipelineTransactionError::InvalidRequest));

    valid = recipe("contract-empty-function", "");
    const auto empty_function = build_metal_pipeline_transaction(
        std::span<const MetalPipelineRecipe>(&valid, 1), library_bytes());
    CHECK(is_error(empty_function, MetalPipelineTransactionError::InvalidRequest));

    valid = recipe("contract-source-mismatch", "second_kernel");
    std::array<uint8_t, 1> different_source = {0};
    const auto mismatch = build_metal_pipeline_transaction(
        std::span<const MetalPipelineRecipe>(&valid, 1), different_source);
    CHECK(is_error(mismatch, MetalPipelineTransactionError::InvalidRequest));

    const auto unused = one;
    CHECK(unused[0].function_name == "second_kernel");
}

void test_multi_library_contract() {
    const std::span<const uint8_t> source_a = bytes_of(kLibraryA);
    const std::span<const uint8_t> source_b = bytes_of(kLibraryB);
    const MetalPipelineLibrarySource library_a =
        library_source(source_a, false);
    const MetalPipelineLibrarySource library_b =
        library_source(source_b, true);
    const std::array<MetalPipelineLibrarySource, 2> libraries = {
        library_a, library_b};
    const MetalPipelineRecipe recipe_a =
        recipe_for("multi-a", "library_a_kernel", source_a);
    const MetalPipelineRecipe recipe_b =
        recipe_for("multi-b", "library_b_kernel", source_b);
    const std::array<MetalPipelineRecipe, 2> recipes = {recipe_a, recipe_b};

    const auto valid = build_metal_pipeline_transaction(recipes, libraries);
    if (const auto* failure = std::get_if<MetalPipelineTransactionFailure>(&valid))
        CHECK(failure->code == MetalPipelineTransactionError::NoDevice);

    const std::array<MetalPipelineLibrarySource, 1> only_a = {library_a};
    CHECK(is_error(build_metal_pipeline_transaction(recipes, only_a),
                   MetalPipelineTransactionError::InvalidRequest));

    MetalPipelineLibrarySource zero_digest_source = library_a;
    zero_digest_source.source_digest = {};
    const std::array<MetalPipelineLibrarySource, 2> zero_digest_sources = {
        zero_digest_source, library_b};
    CHECK(is_error(build_metal_pipeline_transaction(recipes, zero_digest_sources),
                   MetalPipelineTransactionError::InvalidRequest));

    const std::array<MetalPipelineLibrarySource, 2> duplicate_sources = {
        library_a, library_a};
    CHECK(is_error(build_metal_pipeline_transaction(recipes, duplicate_sources),
                   MetalPipelineTransactionError::InvalidRequest));

    MetalPipelineLibrarySource bad_bytes = library_a;
    bad_bytes.source_digest = metal_pipeline_digest(source_b);
    const std::array<MetalPipelineLibrarySource, 2> tampered_source = {
        bad_bytes, library_b};
    CHECK(is_error(build_metal_pipeline_transaction(recipes, tampered_source),
                   MetalPipelineTransactionError::InvalidRequest));

    const std::array<MetalPipelineRecipe, 1> only_a_recipe = {recipe_a};
    CHECK(is_error(build_metal_pipeline_transaction(only_a_recipe, libraries),
                   MetalPipelineTransactionError::InvalidRequest));

    MetalPipelineRecipe wrong_source_function = recipe_for(
        "cross-library", "library_a_kernel", source_b);
    const std::array<MetalPipelineRecipe, 2> wrong_source = {
        recipe_a, wrong_source_function};
    const auto cross_library =
        build_metal_pipeline_transaction(wrong_source, libraries);
    CHECK(std::get_if<MetalPipelineTransaction>(&cross_library) == nullptr);
    if (const auto* failure =
            std::get_if<MetalPipelineTransactionFailure>(&cross_library))
        CHECK(failure->code == MetalPipelineTransactionError::NoDevice ||
              failure->code == MetalPipelineTransactionError::FunctionLookupFailed);

    MetalPipelineRecipe conflicting = recipe_a;
    conflicting.function_name = "library_b_kernel";
    const std::array<MetalPipelineRecipe, 2> conflicting_recipes = {
        recipe_a, conflicting};
    CHECK(is_error(build_metal_pipeline_transaction(conflicting_recipes, libraries),
                   MetalPipelineTransactionError::InvalidRequest));

    const std::array<MetalPipelineRecipe, 2> duplicate_recipes = {
        recipe_a, recipe_a};
    CHECK(is_error(build_metal_pipeline_transaction(duplicate_recipes, libraries),
                   MetalPipelineTransactionError::InvalidRequest));

    auto wrong_compile = libraries;
    wrong_compile[1].compile_contract.fast_math_enabled = false;
    const auto wrong_compile_result =
        build_metal_pipeline_transaction(recipes, wrong_compile);
    if (const auto* failure = std::get_if<MetalPipelineTransactionFailure>(
            &wrong_compile_result))
        CHECK(failure->code == MetalPipelineTransactionError::NoDevice);
}

bool build_native_pair(MetalPipelineTransaction* transaction,
                       MetalPipelineTransactionFailure* failure) {
    MetalPipelineRecipe first = recipe("requirement-a", "first_kernel", true);
    MetalPipelineRecipe second = recipe("requirement-b", "second_kernel");
    const std::array<MetalPipelineRecipe, 3> input = {second, first, first};
    auto result = build_metal_pipeline_transaction(input, library_bytes());
    if (auto* value = std::get_if<MetalPipelineTransaction>(&result)) {
        *transaction = std::move(*value);
        return true;
    }
    if (failure) *failure = std::get<MetalPipelineTransactionFailure>(result);
    return false;
}

void test_transaction_and_lease() {
    MetalPipelineTransaction transaction;
    MetalPipelineTransactionFailure failure;
    if (!build_native_pair(&transaction, &failure)) {
        CHECK(failure.code == MetalPipelineTransactionError::NoDevice);
        return;
    }

    CHECK(transaction.valid());
    CHECK(transaction.slot_count() == 2);
    CHECK(transaction.audit().devices_created == 1);
    CHECK(transaction.audit().libraries_created == 1);
    CHECK(transaction.audit().functions_created == 2);
    CHECK(transaction.audit().pipelines_created == 2);
    CHECK(transaction.audit().command_queues_created == 0);
    CHECK(transaction.audit().command_buffers_created == 0);
    CHECK(transaction.audit().buffers_created == 0);
    CHECK(transaction.audit().sessions_created == 0);
    CHECK(transaction.audit().global_cache_mutations == 0);
    CHECK(transaction.audit().environment_mutations == 0);
    CHECK(transaction.audit().transaction_published);

    const MetalPipelineDigest generation = transaction.generation_digest();
    CHECK(generation != MetalPipelineDigest{});
    for (uint32_t index = 0; index != transaction.slot_count(); ++index) {
        const MetalPipelineSlotInfo* current = transaction.slot(index);
        CHECK(current != nullptr);
        if (!current) continue;
        CHECK(current->slot == index);
        CHECK(current->pipeline_identity != nullptr);
        CHECK(current->thread_execution_width != 0);
        CHECK(current->max_total_threads_per_threadgroup != 0);
        CHECK(current->dispatch_minimum <=
              current->max_total_threads_per_threadgroup);
        CHECK(current->dispatch_minimum <= current->dispatch_maximum);
        CHECK(current->dispatch_maximum <=
              current->max_total_threads_per_threadgroup);
        CHECK(transaction.find(current->requirement_digest) == current);
    }
    CHECK(transaction.slot(2) == nullptr);
    CHECK(transaction.find({}) == nullptr);

    MetalPipelineLease lease = std::move(transaction).take_lease();
    CHECK(!transaction.valid());
    CHECK(transaction.slot_count() == 0);
    CHECK(lease.valid());
    CHECK(lease.slot_count() == 2);
    CHECK(lease.generation_digest() == generation);
    CHECK(lease.device_token() != nullptr);
    CHECK(lease.matches_device_token(lease.device_token()));
    CHECK(!lease.matches_device_token(nullptr));
    int unrelated_device = 0;
    CHECK(!lease.matches_device_token(&unrelated_device));
    for (uint32_t index = 0; index != lease.slot_count(); ++index) {
        const MetalPipelineSlotInfo* current = lease.slot(index);
        CHECK(current != nullptr);
        if (!current) continue;
        CHECK(lease.pipeline_token(index) == current->pipeline_identity);
        CHECK(lease.pipeline_token(index) != nullptr);
        CHECK(lease.find(current->requirement_digest) == current);
    }

    uint32_t resolved_slot = UINT32_MAX;
    const void* resolved_pipeline = reinterpret_cast<const void*>(uintptr_t(1));
    CHECK(lease.resolve_recipe(recipe("requirement-a", "first_kernel", true),
                               &resolved_slot, &resolved_pipeline));
    CHECK(resolved_slot == 0);
    CHECK(resolved_pipeline == lease.pipeline_token(resolved_slot));

    MetalPipelineRecipe reordered = recipe("requirement-a", "first_kernel", true);
    std::swap(reordered.function_constants[0], reordered.function_constants[1]);
    resolved_slot = UINT32_MAX;
    resolved_pipeline = reinterpret_cast<const void*>(uintptr_t(1));
    CHECK(lease.resolve_recipe(reordered, &resolved_slot, &resolved_pipeline));
    CHECK(resolved_slot == 0);
    CHECK(resolved_pipeline == lease.pipeline_token(resolved_slot));

    const MetalPipelineRecipe queries[] = {
        recipe("missing-requirement", "first_kernel", true),
        recipe("requirement-a", "second_kernel", true),
        recipe_for("requirement-a", "first_kernel", bytes_of(kLibraryA)),
    };
    for (const MetalPipelineRecipe& query : queries) {
        resolved_slot = UINT32_MAX;
        resolved_pipeline = reinterpret_cast<const void*>(uintptr_t(1));
        CHECK(!lease.resolve_recipe(query, &resolved_slot, &resolved_pipeline));
        CHECK(resolved_slot == UINT32_MAX);
        CHECK(resolved_pipeline == reinterpret_cast<const void*>(uintptr_t(1)));
    }

    MetalPipelineRecipe changed_constant =
        recipe("requirement-a", "first_kernel", true);
    changed_constant.function_constants[1].value_bits = 8;
    resolved_slot = UINT32_MAX;
    resolved_pipeline = reinterpret_cast<const void*>(uintptr_t(1));
    CHECK(!lease.resolve_recipe(changed_constant, &resolved_slot,
                                &resolved_pipeline));
    CHECK(resolved_slot == UINT32_MAX);
    CHECK(resolved_pipeline == reinterpret_cast<const void*>(uintptr_t(1)));

    MetalPipelineRecipe changed_dispatch = recipe("requirement-a", "first_kernel", true);
    changed_dispatch.dispatch.max_threads_per_threadgroup = 512;
    resolved_slot = UINT32_MAX;
    resolved_pipeline = reinterpret_cast<const void*>(uintptr_t(1));
    CHECK(!lease.resolve_recipe(changed_dispatch, &resolved_slot, &resolved_pipeline));
    CHECK(resolved_slot == UINT32_MAX);
    CHECK(resolved_pipeline == reinterpret_cast<const void*>(uintptr_t(1)));
    CHECK(!lease.resolve_recipe(reordered, nullptr, &resolved_pipeline));
    CHECK(!lease.resolve_recipe(reordered, &resolved_slot, nullptr));

    MetalPipelineLease moved = std::move(lease);
    CHECK(!lease.valid());
    resolved_slot = UINT32_MAX;
    resolved_pipeline = reinterpret_cast<const void*>(uintptr_t(1));
    CHECK(!lease.resolve_recipe(reordered, &resolved_slot, &resolved_pipeline));
    CHECK(resolved_slot == UINT32_MAX);
    CHECK(resolved_pipeline == reinterpret_cast<const void*>(uintptr_t(1)));
    CHECK(moved.valid());
}

void test_dedup_and_determinism() {
    MetalPipelineTransaction first;
    MetalPipelineTransactionFailure first_failure;
    if (!build_native_pair(&first, &first_failure)) {
        CHECK(first_failure.code == MetalPipelineTransactionError::NoDevice);
        return;
    }

    MetalPipelineRecipe a = recipe("requirement-a", "first_kernel", true);
    MetalPipelineRecipe b = recipe("requirement-b", "second_kernel");
    const std::array<MetalPipelineRecipe, 3> reversed = {a, b, a};
    const auto result = build_metal_pipeline_transaction(reversed, library_bytes());
    const auto* second = std::get_if<MetalPipelineTransaction>(&result);
    CHECK(second != nullptr);
    if (!second) return;
    CHECK(second->slot_count() == first.slot_count());
    CHECK(second->generation_digest() == first.generation_digest());
    for (uint32_t index = 0; index != first.slot_count(); ++index) {
        CHECK(second->slot(index)->slot == index);
        CHECK(second->slot(index)->requirement_digest ==
              first.slot(index)->requirement_digest);
        CHECK(second->slot(index)->pipeline_identity != nullptr);
    }

    MetalPipelineRecipe changed_limit = a;
    changed_limit.dispatch.max_threads_per_threadgroup = 512;
    const std::array<MetalPipelineRecipe, 2> changed_recipes = {
        changed_limit, b,
    };
    const auto changed_result = build_metal_pipeline_transaction(
        changed_recipes, library_bytes());
    const auto* changed = std::get_if<MetalPipelineTransaction>(&changed_result);
    CHECK(changed != nullptr);
    if (changed)
        CHECK(changed->generation_digest() != first.generation_digest());

    std::array<MetalPipelineRecipe, 2> conflict = {a, a};
    conflict[1].function_name = "second_kernel";
    const auto conflicting = build_metal_pipeline_transaction(conflict, library_bytes());
    CHECK(is_error(conflicting, MetalPipelineTransactionError::InvalidRequest));

    MetalPipelineRecipe reordered = a;
    std::swap(reordered.function_constants[0], reordered.function_constants[1]);
    const std::array<MetalPipelineRecipe, 2> equivalent = {a, reordered};
    auto equivalent_result = build_metal_pipeline_transaction(
        equivalent, library_bytes());
    if (const auto* equivalent_transaction =
            std::get_if<MetalPipelineTransaction>(&equivalent_result))
        CHECK(equivalent_transaction->slot_count() == 1);
    else
        CHECK(first_failure.code == MetalPipelineTransactionError::NoDevice);
}

void test_effective_dispatch_limit() {
    MetalPipelineRecipe impossible = recipe(
        "effective-limit", "second_kernel");
    impossible.dispatch.min_threads_per_threadgroup = 1;
    impossible.dispatch.max_threads_per_threadgroup = 1;
    impossible.dispatch.required_simdgroups = 2;
    const std::array<MetalPipelineRecipe, 1> recipes = {impossible};
    const auto result = build_metal_pipeline_transaction(recipes, library_bytes());
    CHECK(is_error(result, MetalPipelineTransactionError::PipelineLimitsInvalid));
}

#if defined(LAPLACE_TESTING)
void test_pipeline_lease_session_boundary() {
    const std::array<MetalPipelineRecipe, 2> recipes = {
        recipe("requirement-a", "first_kernel", true),
        recipe("requirement-b", "second_kernel")};

    MetalPipelineTransaction transaction;
    MetalPipelineTransactionFailure failure;
    if (!build_native_pair(&transaction, &failure)) {
        CHECK(failure.code == MetalPipelineTransactionError::NoDevice);
        return;
    }
    MetalPipelineLease lease = std::move(transaction).take_lease();
    const void* device = lease.device_token();
    const MetalPipelineDigest generation = lease.generation_digest();

    auto missing = recipes;
    missing[1].normalized_requirement_digest = digest_text("missing");
    CHECK(!metal_tok_session_create_from_pipeline_lease(
        std::move(lease), std::span<const MetalPipelineRecipe>(missing)));
    CHECK(lease.valid());
    CHECK(lease.device_token() == device);
    CHECK(lease.generation_digest() == generation);
    CHECK(lease.slot_count() == recipes.size());

    const std::array<MetalPipelineRecipe, 1> subset = {recipes[0]};
    CHECK(!metal_tok_session_create_from_pipeline_lease(
        std::move(lease), std::span<const MetalPipelineRecipe>(subset)));
    CHECK(lease.valid());
    CHECK(lease.device_token() == device);

    auto reversed = recipes;
    std::swap(reversed[0], reversed[1]);
    CHECK(!metal_tok_session_create_from_pipeline_lease(
        std::move(lease), std::span<const MetalPipelineRecipe>(reversed)));
    CHECK(lease.valid());
    CHECK(lease.device_token() == device);

    auto duplicate = recipes;
    duplicate[1] = duplicate[0];
    CHECK(!metal_tok_session_create_from_pipeline_lease(
        std::move(lease), std::span<const MetalPipelineRecipe>(duplicate)));
    CHECK(lease.valid());
    CHECK(lease.device_token() == device);

    auto tampered = recipes;
    tampered[0].function_name = "second_kernel";
    CHECK(!metal_tok_session_create_from_pipeline_lease(
        std::move(lease), std::span<const MetalPipelineRecipe>(tampered)));
    CHECK(lease.valid());
    CHECK(lease.device_token() == device);

    MetalPipelineTransaction moved_transaction;
    if (!build_native_pair(&moved_transaction, &failure)) {
        CHECK(failure.code == MetalPipelineTransactionError::NoDevice);
        return;
    }
    MetalPipelineLease moved_from = std::move(moved_transaction).take_lease();
    MetalPipelineLease retained = std::move(moved_from);
    CHECK(!moved_from.valid());
    CHECK(!metal_tok_session_create_from_pipeline_lease(
        std::move(moved_from), std::span<const MetalPipelineRecipe>(recipes)));
    CHECK(!moved_from.valid());
    CHECK(retained.valid());

    MetalPipelineTransaction success_transaction;
    if (!build_native_pair(&success_transaction, &failure)) {
        CHECK(failure.code == MetalPipelineTransactionError::NoDevice);
        return;
    }
    MetalPipelineLease success_lease =
        std::move(success_transaction).take_lease();
    const uintptr_t success_device =
        reinterpret_cast<uintptr_t>(success_lease.device_token());
    metal_dispatch_metrics_reset();
    metal_pipeline_global_lookup_metrics_reset_for_testing();
    const std::shared_ptr<MetalTokSession> session =
        metal_tok_session_create_from_pipeline_lease(
            std::move(success_lease), std::span<const MetalPipelineRecipe>(recipes));
    CHECK(session != nullptr);
    CHECK(!success_lease.valid());
    CHECK(metal_tok_session_queue_identity_for_testing(*session) != 0);
    CHECK(metal_tok_session_queue_device_identity_for_testing(*session) ==
          success_device);
    CHECK(metal_tok_session_pipeline_lease_slot_count_for_testing(*session) ==
          recipes.size());
    CHECK(metal_tok_session_pipeline_device_identity_for_testing(*session, 0) ==
          success_device);
    CHECK(metal_dispatch_metrics().command_buffers == 0);
    CHECK(metal_tok_session_begin(*session, 1, 1, 1, 0, 0, 0, 0, 0,
                                  1, 1, 0, 1));
    metal_tok_session_abort(*session);
    CHECK(metal_dispatch_metrics().command_buffers == 0);
    CHECK(metal_pipeline_global_lookup_metrics_for_testing().lookup_attempts == 0);
    CHECK(!metal_tok_session_probe_global_pipeline_for_testing(*session));
    CHECK(metal_pipeline_global_lookup_metrics_for_testing().lookup_attempts == 1);
    CHECK(metal_pipeline_global_lookup_metrics_for_testing().compilation_attempts == 0);
    metal_pipeline_global_lookup_metrics_reset_for_testing();
    CHECK(!metal_tok_session_dispatch_leased_slot_for_testing(
        *session, std::numeric_limits<uint32_t>::max()));
    CHECK(metal_dispatch_metrics().command_buffers == 0);
    CHECK(metal_tok_session_dispatch_leased_slot_for_testing(*session, 0));
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    CHECK(metal_tok_session_pipeline_device_identity_for_testing(*session, 0) ==
          success_device);
    const MetalPipelineGlobalLookupMetrics pipeline_metrics =
        metal_pipeline_global_lookup_metrics_for_testing();
    CHECK(pipeline_metrics.lookup_attempts == 0);
    CHECK(pipeline_metrics.compilation_attempts == 0);
}

bool dispatch_program_invocation_for_test(
    MetalTokSession& session, const char* function,
    std::span<const MetalFunctionConstant> constants = {}) {
    return metal_tok_session_dispatch_program_invocation_for_testing(
        session, function, constants);
}

void test_ordered_leased_program_stream() {
    const std::array<MetalPipelineRecipe, 2> recipes = {
        recipe("requirement-a", "first_kernel", true),
        recipe("requirement-b", "second_kernel")};
    const std::array<uint32_t, 4> flattened = {0, 1, 1, 0};
    std::array<MetalTokProgramRange, 2> ranges = {
        MetalTokProgramRange{17, 0, 3}, MetalTokProgramRange{29, 3, 1}};
    ranges[0].semantic_program_digest.fill(0x11);
    ranges[1].semantic_program_digest.fill(0x12);
    const std::vector<MetalTokInvocationAuthority> authorities =
        invocation_authorities(flattened, ranges);

    MetalPipelineTransaction transaction;
    MetalPipelineTransactionFailure failure;
    if (!build_native_pair(&transaction, &failure)) {
        CHECK(failure.code == MetalPipelineTransactionError::NoDevice);
        return;
    }
    MetalPipelineLease lease = std::move(transaction).take_lease();
    metal_pipeline_global_lookup_metrics_reset_for_testing();
    metal_dispatch_metrics_reset();
    const std::array<uint32_t, 2> invalid_flattened = {0, 2};
    const MetalTokProgramRange invalid_range{17, 0, 2};
    CHECK(!metal_tok_session_create_from_pipeline_lease(
        std::move(lease), std::span<const MetalPipelineRecipe>(recipes),
        std::span<const uint32_t>(invalid_flattened),
        std::span<const MetalTokProgramRange>(&invalid_range, 1)));
    CHECK(lease.valid());
    MetalTokProgramRange invalid_batch_rows{17, 0, 4};
    invalid_batch_rows.batch_rows = 3;
    CHECK(!metal_tok_session_create_from_pipeline_lease(
        std::move(lease), std::span<const MetalPipelineRecipe>(recipes),
        std::span<const uint32_t>(flattened),
        std::span<const MetalTokProgramRange>(&invalid_batch_rows, 1)));
    CHECK(lease.valid());
    MetalTokProgramRange unauthenticated_output_row{17, 0, 4};
    unauthenticated_output_row.batch_rows = 2;
    unauthenticated_output_row.selected_output_row = 1;
    CHECK(!metal_tok_session_create_from_pipeline_lease(
        std::move(lease), std::span<const MetalPipelineRecipe>(recipes),
        std::span<const uint32_t>(flattened),
        std::span<const MetalTokProgramRange>(&unauthenticated_output_row, 1)));
    CHECK(lease.valid());
    MetalTokProgramRange out_of_range_output_row = unauthenticated_output_row;
    out_of_range_output_row.semantic_program_digest[0] = 1;
    out_of_range_output_row.selected_output_row = 2;
    CHECK(!metal_tok_session_create_from_pipeline_lease(
        std::move(lease), std::span<const MetalPipelineRecipe>(recipes),
        std::span<const uint32_t>(flattened),
        std::span<const MetalTokProgramRange>(&out_of_range_output_row, 1)));
    CHECK(lease.valid());
    const std::array<MetalTokProgramRange, 2> overlapping_ranges = {
        MetalTokProgramRange{17, 0, 3}, MetalTokProgramRange{29, 2, 2}};
    CHECK(!metal_tok_session_create_from_pipeline_lease(
        std::move(lease), std::span<const MetalPipelineRecipe>(recipes),
        std::span<const uint32_t>(flattened),
        std::span<const MetalTokProgramRange>(overlapping_ranges)));
    CHECK(lease.valid());
    const std::array<MetalTokProgramRange, 2> duplicate_id_ranges = {
        MetalTokProgramRange{17, 0, 2}, MetalTokProgramRange{17, 2, 2}};
    CHECK(!metal_tok_session_create_from_pipeline_lease(
        std::move(lease), std::span<const MetalPipelineRecipe>(recipes),
        std::span<const uint32_t>(flattened),
        std::span<const MetalTokProgramRange>(duplicate_id_ranges)));
    CHECK(lease.valid());
    const std::array<MetalTokProgramRange, 2> uncovered_ranges = {
        MetalTokProgramRange{17, 0, 1}, MetalTokProgramRange{29, 2, 1}};
    CHECK(!metal_tok_session_create_from_pipeline_lease(
        std::move(lease), std::span<const MetalPipelineRecipe>(recipes),
        std::span<const uint32_t>(flattened),
        std::span<const MetalTokProgramRange>(uncovered_ranges)));
    CHECK(lease.valid());
    CHECK(!metal_tok_session_create_from_pipeline_lease(
        std::move(lease), std::span<const MetalPipelineRecipe>(recipes),
        std::span<const uint32_t>(flattened),
        std::span<const MetalTokProgramRange>(ranges),
        std::span<const MetalTokInvocationAuthority>(
            authorities.data(), authorities.size() - 1)));
    CHECK(lease.valid());
    auto tampered_authorities = authorities;
    tampered_authorities[0].row_index = 1;
    CHECK(!metal_tok_session_create_from_pipeline_lease(
        std::move(lease), std::span<const MetalPipelineRecipe>(recipes),
        std::span<const uint32_t>(flattened),
        std::span<const MetalTokProgramRange>(ranges),
        std::span<const MetalTokInvocationAuthority>(tampered_authorities)));
    CHECK(lease.valid());
    tampered_authorities = authorities;
    std::swap(tampered_authorities[0], tampered_authorities[1]);
    CHECK(!metal_tok_session_create_from_pipeline_lease(
        std::move(lease), std::span<const MetalPipelineRecipe>(recipes),
        std::span<const uint32_t>(flattened),
        std::span<const MetalTokProgramRange>(ranges),
        std::span<const MetalTokInvocationAuthority>(tampered_authorities)));
    CHECK(lease.valid());
    tampered_authorities = authorities;
    tampered_authorities[0].pipeline_slot = 1;
    CHECK(!metal_tok_session_create_from_pipeline_lease(
        std::move(lease), std::span<const MetalPipelineRecipe>(recipes),
        std::span<const uint32_t>(flattened),
        std::span<const MetalTokProgramRange>(ranges),
        std::span<const MetalTokInvocationAuthority>(tampered_authorities)));
    CHECK(lease.valid());
    const std::shared_ptr<MetalTokSession> session =
        metal_tok_session_create_from_pipeline_lease(
            std::move(lease), std::span<const MetalPipelineRecipe>(recipes),
            std::span<const uint32_t>(flattened),
            std::span<const MetalTokProgramRange>(ranges),
            std::span<const MetalTokInvocationAuthority>(authorities));
    CHECK(session != nullptr);
    if (!session) return;

    const std::array<MetalFunctionConstant, 2> first_constants = {
        MetalFunctionConstant{0, MetalFunctionConstantType::Bool, 1},
        MetalFunctionConstant{1, MetalFunctionConstantType::UInt32, 7}};
    CHECK(metal_tok_session_begin_program(*session, 17));
    CHECK(metal_tok_session_begin(*session, 1, 1, 1, 0, 0, 0, 0, 0,
                                  1, 1, 0, 1));
    CHECK(dispatch_program_invocation_for_test(*session, "first_kernel",
                                               first_constants));
    CHECK(dispatch_program_invocation_for_test(*session, "second_kernel"));
    CHECK(dispatch_program_invocation_for_test(*session, "second_kernel"));
    CHECK(!dispatch_program_invocation_for_test(*session, "first_kernel",
                                                first_constants));
    CHECK(metal_tok_session_commit_token(*session));
    CHECK(metal_dispatch_metrics().command_buffers == 1);

    CHECK(metal_tok_session_select_program(*session, 29));
    CHECK(metal_tok_session_begin(*session, 1, 1, 1, 0, 0, 0, 0, 0,
                                  1, 1, 0, 1));
    CHECK(dispatch_program_invocation_for_test(*session, "first_kernel",
                                               first_constants));
    CHECK(metal_tok_session_commit_token(*session));
    CHECK(metal_dispatch_metrics().command_buffers == 2);
    CHECK(metal_pipeline_global_lookup_metrics_for_testing().lookup_attempts == 0);
    CHECK(metal_pipeline_global_lookup_metrics_for_testing().compilation_attempts == 0);

    CHECK(metal_tok_session_select_program(*session, 17));
    CHECK(metal_tok_session_begin(*session, 1, 1, 1, 0, 0, 0, 0, 0,
                                  1, 1, 0, 1));
    CHECK(!dispatch_program_invocation_for_test(*session, "second_kernel"));
    CHECK(!dispatch_program_invocation_for_test(*session, "first_kernel",
                                                std::array<MetalFunctionConstant, 2>{
                                                    first_constants[0],
                                                    MetalFunctionConstant{1, MetalFunctionConstantType::UInt32, 8}}));
    CHECK(dispatch_program_invocation_for_test(*session, "first_kernel",
                                               first_constants));
    CHECK(!metal_tok_session_seal_token(*session));
    metal_tok_session_abort(*session);

    CHECK(metal_tok_session_begin_program(*session, 17));
    CHECK(metal_tok_session_begin(*session, 1, 1, 1, 0, 0, 0, 0, 0,
                                  1, 1, 0, 1));
    CHECK(dispatch_program_invocation_for_test(*session, "first_kernel",
                                               first_constants));
    CHECK(dispatch_program_invocation_for_test(*session, "second_kernel"));
    CHECK(dispatch_program_invocation_for_test(*session, "second_kernel"));
    CHECK(metal_tok_session_commit_token(*session));
    CHECK(metal_dispatch_metrics().command_buffers == 3);
    CHECK(metal_pipeline_global_lookup_metrics_for_testing().lookup_attempts == 0);
    CHECK(metal_pipeline_global_lookup_metrics_for_testing().compilation_attempts == 0);
}

void test_moe_reduce_stream_order() {
    const std::array<MetalPipelineRecipe, 2> input = {
        recipe("moe-apply-down-scale", "apply_down_scale"),
        recipe("moe-combine", "moe_combine")};
    // The test transaction has two slots with deterministic identities, while
    // the production transaction is free to normalize its recipe order.
    // Rebind the borrowed recipe table to those exact slot indices before
    // constructing the session.
    auto result = build_metal_pipeline_transaction(input, library_bytes());
    auto* value = std::get_if<MetalPipelineTransaction>(&result);
    if (!value) {
        const auto* error = std::get_if<MetalPipelineTransactionFailure>(&result);
        CHECK(error != nullptr);
        if (error) CHECK(error->code == MetalPipelineTransactionError::NoDevice);
        return;
    }
    MetalPipelineLease lease = std::move(*value).take_lease();
    std::array<MetalPipelineRecipe, 2> ordered{};
    uint32_t apply_index = UINT32_MAX;
    uint32_t combine_index = UINT32_MAX;
    for (const MetalPipelineRecipe& candidate : input) {
        uint32_t slot = UINT32_MAX;
        const void* pipeline = nullptr;
        CHECK(lease.resolve_recipe(candidate, &slot, &pipeline));
        if (slot >= ordered.size()) continue;
        ordered[slot] = candidate;
        if (candidate.function_name == "apply_down_scale") apply_index = slot;
        if (candidate.function_name == "moe_combine") combine_index = slot;
    }
    CHECK(apply_index != UINT32_MAX);
    CHECK(combine_index != UINT32_MAX);
    if (apply_index == UINT32_MAX || combine_index == UINT32_MAX) return;

    const std::array<uint32_t, 2> flattened = {apply_index, combine_index};
    const MetalTokProgramRange range{41, 0, 2};
    const std::shared_ptr<MetalTokSession> session =
        metal_tok_session_create_from_pipeline_lease(
            std::move(lease), std::span<const MetalPipelineRecipe>(ordered),
            std::span<const uint32_t>(flattened),
            std::span<const MetalTokProgramRange>(&range, 1));
    CHECK(session != nullptr);
    if (!session) return;

    metal_dispatch_metrics_reset();
    metal_pipeline_global_lookup_metrics_reset_for_testing();
    CHECK(metal_tok_session_begin_program(*session, range.id));
    CHECK(metal_tok_session_begin(*session, 1, 1, 1, 0, 0, 0, 0, 0,
                                  1, 1, 0, 1));
    CHECK(!dispatch_program_invocation_for_test(*session, "moe_combine"));
    CHECK(dispatch_program_invocation_for_test(*session, "apply_down_scale"));
    CHECK(!metal_tok_session_seal_token(*session));
    metal_tok_session_abort(*session);

    CHECK(metal_tok_session_begin_program(*session, range.id));
    CHECK(metal_tok_session_begin(*session, 1, 1, 1, 0, 0, 0, 0, 0,
                                  1, 1, 0, 1));
    CHECK(dispatch_program_invocation_for_test(*session, "apply_down_scale"));
    CHECK(dispatch_program_invocation_for_test(*session, "moe_combine"));
    CHECK(metal_tok_session_commit_token(*session));
    CHECK(!dispatch_program_invocation_for_test(*session, "moe_combine"));
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    CHECK(metal_pipeline_global_lookup_metrics_for_testing().lookup_attempts == 0);
    CHECK(metal_pipeline_global_lookup_metrics_for_testing().compilation_attempts == 0);
}

Tensor recurrent_f32_vector(std::vector<float>& storage, int width) {
    storage.assign(static_cast<size_t>(width), 0.0f);
    Tensor tensor;
    tensor.type = GGMLType::F32;
    tensor.n_dims = 1;
    tensor.dims[0] = static_cast<uint64_t>(width);
    tensor.data = reinterpret_cast<const uint8_t*>(storage.data());
    tensor.data_bytes = storage.size() * sizeof(float);
    return tensor;
}

Tensor recurrent_f32_matrix(std::vector<float>& storage, int width, int height) {
    storage.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0.0f);
    Tensor tensor;
    tensor.type = GGMLType::F32;
    tensor.n_dims = 2;
    tensor.dims[0] = static_cast<uint64_t>(width);
    tensor.dims[1] = static_cast<uint64_t>(height);
    tensor.data = reinterpret_cast<const uint8_t*>(storage.data());
    tensor.data_bytes = storage.size() * sizeof(float);
    return tensor;
}

Tensor recurrent_q2_matrix(std::vector<uint8_t>& storage, int width, int height) {
    const size_t row_bytes = static_cast<size_t>(width / QK_KQUANT) *
                             bytes_per_block(GGMLType::Q2_K);
    storage.assign(static_cast<size_t>(height) * row_bytes, 0);
    Tensor tensor;
    tensor.type = GGMLType::Q2_K;
    tensor.n_dims = 2;
    tensor.dims[0] = static_cast<uint64_t>(width);
    tensor.dims[1] = static_cast<uint64_t>(height);
    tensor.data = storage.data();
    tensor.data_bytes = storage.size();
    return tensor;
}

Tensor recurrent_f16_matrix(std::vector<uint16_t>& storage, int width, int height) {
    storage.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0);
    Tensor tensor;
    tensor.type = GGMLType::F16;
    tensor.n_dims = 2;
    tensor.dims[0] = static_cast<uint64_t>(width);
    tensor.dims[1] = static_cast<uint64_t>(height);
    tensor.data = reinterpret_cast<const uint8_t*>(storage.data());
    tensor.data_bytes = storage.size() * sizeof(uint16_t);
    return tensor;
}

std::shared_ptr<MetalTokSession> recurrent_stream_session(
    std::span<const std::string_view> sequence, uint32_t program_id = 73,
    uint32_t batch_rows = 1,
    uint32_t selected_output_row = UINT32_MAX) {
    std::vector<MetalPipelineRecipe> input;
    std::vector<std::string_view> unique_functions;
    try {
        for (const std::string_view function : sequence) {
            auto found = std::find(unique_functions.begin(), unique_functions.end(), function);
            if (found == unique_functions.end()) {
                unique_functions.push_back(function);
                input.push_back(recipe("recurrent-" + std::to_string(input.size()), function));
            }
        }
    } catch (...) {
        CHECK(false);
        return {};
    }
    auto result = build_metal_pipeline_transaction(input, library_bytes());
    auto* transaction = std::get_if<MetalPipelineTransaction>(&result);
    if (!transaction) {
        const auto* failure = std::get_if<MetalPipelineTransactionFailure>(&result);
        CHECK(failure != nullptr);
        if (failure) CHECK(failure->code == MetalPipelineTransactionError::NoDevice);
        return {};
    }
    MetalPipelineLease lease = std::move(*transaction).take_lease();
    std::vector<MetalPipelineRecipe> ordered(input.size());
    std::vector<uint32_t> function_slots(unique_functions.size(), UINT32_MAX);
    for (const MetalPipelineRecipe& candidate : input) {
        uint32_t slot = UINT32_MAX;
        const void* pipeline = nullptr;
        CHECK(lease.resolve_recipe(candidate, &slot, &pipeline));
        CHECK(slot < ordered.size());
        if (slot < ordered.size()) {
            ordered[slot] = candidate;
            const auto found = std::find(unique_functions.begin(), unique_functions.end(),
                                         std::string_view(candidate.function_name));
            CHECK(found != unique_functions.end());
            if (found != unique_functions.end())
                function_slots[static_cast<size_t>(found - unique_functions.begin())] = slot;
        }
    }
    std::vector<uint32_t> flattened;
    flattened.reserve(sequence.size());
    for (const std::string_view function : sequence) {
        const auto found = std::find(unique_functions.begin(), unique_functions.end(), function);
        CHECK(found != unique_functions.end());
        if (found == unique_functions.end()) return {};
        const uint32_t slot = function_slots[static_cast<size_t>(found - unique_functions.begin())];
        CHECK(slot != UINT32_MAX);
        if (slot == UINT32_MAX) return {};
        flattened.push_back(slot);
    }
    MetalTokProgramRange range{
        program_id, 0, static_cast<uint32_t>(flattened.size())};
    range.batch_rows = batch_rows;
    range.selected_output_row = selected_output_row;
    if (selected_output_row != UINT32_MAX)
        range.semantic_program_digest[0] = 1;
    return metal_tok_session_create_from_pipeline_lease(
        std::move(lease), std::span<const MetalPipelineRecipe>(ordered),
        std::span<const uint32_t>(flattened),
        std::span<const MetalTokProgramRange>(&range, 1));
}

void test_dense_leased_repeated_dispatches() {
    constexpr int H = 256;
    std::vector<float> attn_norm_storage, ffn_norm_storage;
    const Tensor attn_norm = recurrent_f32_vector(attn_norm_storage, H);
    const Tensor ffn_norm = recurrent_f32_vector(ffn_norm_storage, H);
    std::array<std::vector<uint8_t>, 7> matrix_storage;
    std::array<Tensor, 7> matrices;
    const std::array<std::array<int, 2>, 7> dimensions = {
        std::array<int, 2>{H, H}, std::array<int, 2>{H, H},
        std::array<int, 2>{H, H}, std::array<int, 2>{H, H},
        std::array<int, 2>{H, H}, std::array<int, 2>{H, H},
        std::array<int, 2>{H, H}};
    for (size_t index = 0; index != matrices.size(); ++index)
        matrices[index] = recurrent_q2_matrix(matrix_storage[index],
                                              dimensions[index][0], dimensions[index][1]);

    MetalTokLayer layer;
    layer.attn_norm = &attn_norm;
    layer.attn_q = &matrices[0];
    layer.attn_k = &matrices[1];
    layer.attn_v = &matrices[2];
    layer.attn_o = &matrices[3];
    layer.ffn_norm = &ffn_norm;
    layer.ffn_gate = &matrices[4];
    layer.ffn_up = &matrices[5];
    layer.ffn_down = &matrices[6];
    layer.H = H;
    layer.inter = H;
    layer.Hq = 1;
    layer.Hk = 1;
    layer.Dh = H;
    layer.rope_dim = H;
    layer.rope_base = 10000.0f;
    layer.attention_scale = 1.0f;
    layer.rms_eps = 1.0e-5f;
    layer.swiglu = true;

    const std::array<std::string_view, 17> sequence = {
        "rmsnorm_f32", "gemv_q2k", "gemv_q2k", "gemv_q2k",
        "rope_f32", "rope_f32", "kv_write", "kv_write",
        "attn_decode", "gemv_q2k", "vec_add", "rmsnorm_f32", "gemv_q2k",
        "gemv_q2k", "act_glu", "gemv_q2k", "vec_add"};
    const std::shared_ptr<MetalTokSession> session =
        recurrent_stream_session(sequence);
    CHECK(session != nullptr);
    if (!session) return;
    const std::array<float, H> input{};
    metal_dispatch_metrics_reset();
    metal_pipeline_global_lookup_metrics_reset_for_testing();
    CHECK(metal_tok_session_begin_program(*session, 73));
    CHECK(metal_tok_session_begin(*session, H, H, 0, 0, 0, 1, 1, H,
                                  1, 1, 0, 1));
    CHECK(metal_tok_session_upload_x(*session, input.data(), H));
    CHECK(metal_tok_session_layer(*session, layer));
    CHECK(metal_tok_session_commit_token(*session));
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    CHECK(metal_pipeline_global_lookup_metrics_for_testing().lookup_attempts == 0);
    CHECK(metal_pipeline_global_lookup_metrics_for_testing().compilation_attempts == 0);
}

void test_dense_leased_prefill_repeated_dispatches() {
    constexpr int H = 256;
    std::vector<float> attn_norm_storage, ffn_norm_storage;
    const Tensor attn_norm = recurrent_f32_vector(attn_norm_storage, H);
    const Tensor ffn_norm = recurrent_f32_vector(ffn_norm_storage, H);
    std::array<std::vector<uint16_t>, 7> matrix_storage;
    std::array<Tensor, 7> matrices;
    for (size_t index = 0; index != matrices.size(); ++index)
        matrices[index] = recurrent_f16_matrix(matrix_storage[index], H, H);

    MetalTokLayer layer;
    layer.attn_norm = &attn_norm;
    layer.attn_q = &matrices[0];
    layer.attn_k = &matrices[1];
    layer.attn_v = &matrices[2];
    layer.attn_o = &matrices[3];
    layer.ffn_norm = &ffn_norm;
    layer.ffn_gate = &matrices[4];
    layer.ffn_up = &matrices[5];
    layer.ffn_down = &matrices[6];
    layer.H = H;
    layer.inter = H;
    layer.Hq = 1;
    layer.Hk = 1;
    layer.Dh = H;
    layer.rope_dim = H;
    layer.rope_base = 10000.0f;
    layer.attention_scale = 1.0f;
    layer.rms_eps = 1.0e-5f;
    layer.swiglu = true;

    const std::array<std::string_view, 27> sequence = {
        "rmsnorm_f32", "rmsnorm_f32", "prefill_f16_rows", "prefill_f16_rows",
        "prefill_f16_rows", "rope_f32", "rope_f32", "kv_write", "kv_write",
        "rope_f32", "rope_f32", "kv_write", "kv_write", "attn_decode",
        "attn_decode", "prefill_f16_rows", "vec_add", "rmsnorm_f32", "vec_add",
        "rmsnorm_f32", "prefill_f16_rows", "prefill_f16_rows", "act_glu", "act_glu",
        "prefill_f16_rows", "vec_add", "vec_add"};
    const std::shared_ptr<MetalTokSession> session =
        recurrent_stream_session(sequence, 74, 2, 1);
    CHECK(session != nullptr);
    if (!session) return;
    const std::array<float, H> input{};
    metal_dispatch_metrics_reset();
    metal_pipeline_global_lookup_metrics_reset_for_testing();
    CHECK(metal_tok_session_begin_program(*session, 74));
    CHECK(metal_tok_session_begin_prefill_batch(*session, H, H, 0, 0, 0, 1, 1, H,
                                                2, 1, 0, 2));
    CHECK(metal_tok_session_upload_x(*session, input.data(), H));
    CHECK(metal_tok_session_dense_prefill_batch_layer(*session, layer, 2));
    CHECK(!metal_tok_session_select_prefill_batch_row(*session, 0));
    CHECK(metal_tok_session_select_prefill_batch_row(*session, 1));
    CHECK(metal_tok_session_commit_token(*session));
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    CHECK(metal_pipeline_global_lookup_metrics_for_testing().lookup_attempts == 0);
    CHECK(metal_pipeline_global_lookup_metrics_for_testing().compilation_attempts == 0);
}

void test_recurrent_leased_exact_range() {
    constexpr int H = 512;
    constexpr int qk_heads = 2;
    constexpr int value_heads = 4;
    constexpr int head_dimension = 64;
    constexpr int kernel = 2;
    constexpr int channels = head_dimension * (2 * qk_heads + value_heads);
    constexpr int output_width = value_heads * head_dimension;
    constexpr int intermediate = 512;

    std::vector<float> input_norm_storage, conv_storage, dt_storage,
        decay_storage, norm_storage, ffn_norm_storage;
    const Tensor input_norm = recurrent_f32_vector(input_norm_storage, H);
    const Tensor conv = recurrent_f32_matrix(conv_storage, kernel, channels);
    const Tensor dt = recurrent_f32_vector(dt_storage, value_heads);
    const Tensor decay = recurrent_f32_vector(decay_storage, value_heads);
    const Tensor norm = recurrent_f32_vector(norm_storage, head_dimension);
    const Tensor ffn_norm = recurrent_f32_vector(ffn_norm_storage, H);
    std::array<std::vector<uint8_t>, 8> matrix_storage;
    std::array<Tensor, 8> matrices;
    const std::array<std::array<int, 2>, 8> dimensions = {
        std::array<int, 2>{H, channels}, std::array<int, 2>{H, output_width},
        std::array<int, 2>{H, value_heads}, std::array<int, 2>{H, value_heads},
        std::array<int, 2>{output_width, H}, std::array<int, 2>{H, intermediate},
        std::array<int, 2>{H, intermediate}, std::array<int, 2>{intermediate, H}};
    for (size_t index = 0; index != matrices.size(); ++index)
        matrices[index] = recurrent_q2_matrix(matrix_storage[index],
                                              dimensions[index][0], dimensions[index][1]);

    MetalTokRecurrentLayer layer;
    layer.input_norm = &input_norm;
    layer.qkv = &matrices[0];
    layer.gate = &matrices[1];
    layer.beta = &matrices[2];
    layer.alpha = &matrices[3];
    layer.conv = &conv;
    layer.dt_bias = &dt;
    layer.decay = &decay;
    layer.norm = &norm;
    layer.output = &matrices[4];
    layer.ffn_norm = &ffn_norm;
    layer.ffn_gate = &matrices[5];
    layer.ffn_up = &matrices[6];
    layer.ffn_down = &matrices[7];
    layer.H = H;
    layer.qk_heads = qk_heads;
    layer.value_heads = value_heads;
    layer.head_dimension = head_dimension;
    layer.kernel = kernel;
    layer.ffn_intermediate = intermediate;
    layer.l2_epsilon = 1.0e-5f;
    layer.rms_epsilon = 1.0e-5f;

    const std::array<std::string_view, 16> sequence = {
        "rmsnorm_f32", "gemv_q2k", "gemv_q2k", "gemv_q2k",
        "gemv_q2k", "dnet_conv_silu", "dnet_l2_qk", "dnet_update",
        "gemv_q2k", "vec_add", "rmsnorm_f32", "gemv_q2k",
        "gemv_q2k", "act_glu", "gemv_q2k", "vec_add"};
    const std::array<float, H> input{};
    const auto begin = [&](MetalTokSession& session) {
        CHECK(metal_tok_session_begin_program(session, 73));
        CHECK(metal_tok_session_begin(session, H, H, 0, 0, 0, 0, 0, 0, 1,
                                      1, 0, 1));
        CHECK(metal_tok_session_upload_x(session, input.data(), H));
    };

    const std::shared_ptr<MetalTokSession> valid =
        recurrent_stream_session(sequence);
    CHECK(valid != nullptr);
    if (!valid) return;
    metal_dispatch_metrics_reset();
    metal_pipeline_global_lookup_metrics_reset_for_testing();
    begin(*valid);
    CHECK(metal_tok_session_recurrent_layer(*valid, layer));
    metal_tok_session_abort(*valid);
    CHECK(metal_dispatch_metrics().command_buffers == 0);
    begin(*valid);
    CHECK(metal_tok_session_recurrent_layer(*valid, layer));
    CHECK(metal_tok_session_recurrent_commit(*valid));
    CHECK(metal_dispatch_metrics().command_buffers == 1);
    CHECK(metal_tok_session_metrics(*valid).projection_dispatches == 8);
    CHECK(metal_pipeline_global_lookup_metrics_for_testing().lookup_attempts == 0);
    CHECK(metal_pipeline_global_lookup_metrics_for_testing().compilation_attempts == 0);

    std::array<std::string_view, 15> missing;
    std::copy(sequence.begin(), sequence.end() - 1, missing.begin());
    const std::shared_ptr<MetalTokSession> missing_session =
        recurrent_stream_session(missing);
    CHECK(missing_session != nullptr);
    if (missing_session) {
        begin(*missing_session);
        CHECK(!metal_tok_session_recurrent_layer(*missing_session, layer));
        metal_tok_session_abort(*missing_session);
    }

    auto reordered = sequence;
    std::swap(reordered[0], reordered[1]);
    const std::shared_ptr<MetalTokSession> reordered_session =
        recurrent_stream_session(reordered);
    CHECK(reordered_session != nullptr);
    if (reordered_session) {
        begin(*reordered_session);
        CHECK(!metal_tok_session_recurrent_layer(*reordered_session, layer));
        metal_tok_session_abort(*reordered_session);
    }

    std::array<std::string_view, 17> extra;
    std::copy(sequence.begin(), sequence.end(), extra.begin());
    extra.back() = "gemv_q2k";
    const std::shared_ptr<MetalTokSession> extra_session =
        recurrent_stream_session(extra);
    CHECK(extra_session != nullptr);
    if (extra_session) {
        begin(*extra_session);
        CHECK(metal_tok_session_recurrent_layer(*extra_session, layer));
        CHECK(!metal_tok_session_recurrent_commit(*extra_session));
        metal_tok_session_abort(*extra_session);
    }
}

void test_allocation_failure_boundary() {
    const MetalPipelineRecipe only = recipe("allocation", "second_kernel");
    const auto result = build_metal_pipeline_transaction_for_testing(
        std::span<const MetalPipelineRecipe>(&only, 1), library_bytes(),
        MetalPipelineInjectedFailure::Allocation);
    const auto* failure = std::get_if<MetalPipelineTransactionFailure>(&result);
    CHECK(failure != nullptr);
    if (!failure) return;
    CHECK(failure->code == MetalPipelineTransactionError::AllocationFailed);
    CHECK(!failure->audit.transaction_published);
}

void test_injected_failures() {
    const MetalPipelineRecipe first = recipe("failure-a", "first_kernel", true);
    const MetalPipelineRecipe second = recipe("failure-b", "second_kernel");
    const std::array<MetalPipelineRecipe, 2> recipes = {first, second};
    const MetalPipelineInjectedFailure failures[] = {
        MetalPipelineInjectedFailure::Library,
        MetalPipelineInjectedFailure::Function,
        MetalPipelineInjectedFailure::Pipeline,
        MetalPipelineInjectedFailure::PipelineAfterFirst,
        MetalPipelineInjectedFailure::Allocation,
    };
    for (const MetalPipelineInjectedFailure injected : failures) {
        const uint32_t live_before =
            metal_pipeline_live_pipeline_owners_for_testing();
        const auto result = build_metal_pipeline_transaction_for_testing(
            recipes, library_bytes(), injected);
        const auto* transaction = std::get_if<MetalPipelineTransaction>(&result);
        CHECK(transaction == nullptr);
        const auto* failure = std::get_if<MetalPipelineTransactionFailure>(&result);
        CHECK(failure != nullptr);
        if (!failure) continue;
        CHECK(!failure->audit.transaction_published);
        CHECK(failure->audit.command_queues_created == 0);
        CHECK(failure->audit.command_buffers_created == 0);
        CHECK(failure->audit.buffers_created == 0);
        CHECK(failure->audit.sessions_created == 0);
        CHECK(failure->audit.global_cache_mutations == 0);
        CHECK(failure->audit.environment_mutations == 0);
        if (injected == MetalPipelineInjectedFailure::PipelineAfterFirst)
            CHECK(failure->audit.pipelines_created == 1);
        if (injected == MetalPipelineInjectedFailure::Allocation)
            CHECK(failure->code == MetalPipelineTransactionError::AllocationFailed);
        CHECK(metal_pipeline_live_pipeline_owners_for_testing() == live_before);
    }
}

void test_multi_library_native() {
    const std::span<const uint8_t> source_a = bytes_of(kLibraryA);
    const std::span<const uint8_t> source_b = bytes_of(kLibraryB);
    const std::array<MetalPipelineLibrarySource, 2> libraries = {
        library_source(source_a, false),
        library_source(source_b, true)};
    const std::array<MetalPipelineRecipe, 2> recipes = {
        recipe_for("multi-native-a", "library_a_kernel", source_a),
        recipe_for("multi-native-b", "library_b_kernel", source_b)};
    const uint32_t live_before = metal_pipeline_live_pipeline_owners_for_testing();
    const auto result = build_metal_pipeline_transaction_for_testing(
        recipes, libraries, MetalPipelineInjectedFailure::PipelineAfterFirst);
    const auto* failure = std::get_if<MetalPipelineTransactionFailure>(&result);
    CHECK(failure != nullptr);
    if (failure) {
        CHECK(failure->code == MetalPipelineTransactionError::PipelineBuildFailed);
        CHECK(failure->audit.libraries_created == 2);
        CHECK(failure->audit.functions_created == 1);
        CHECK(failure->audit.pipelines_created == 1);
        CHECK(!failure->audit.transaction_published);
    }
    CHECK(metal_pipeline_live_pipeline_owners_for_testing() == live_before);

    const auto success = build_metal_pipeline_transaction(recipes, libraries);
    const auto* transaction = std::get_if<MetalPipelineTransaction>(&success);
    CHECK(transaction != nullptr);
    if (!transaction) return;
    CHECK(transaction->slot_count() == 2);
    CHECK(transaction->audit().libraries_created == 2);
    CHECK(transaction->audit().functions_created == 2);
    CHECK(transaction->audit().pipelines_created == 2);
    CHECK(transaction->generation_digest() != MetalPipelineDigest{});

    const std::array<MetalPipelineLibrarySource, 2> reversed_libraries = {
        libraries[1], libraries[0]};
    const auto reversed =
        build_metal_pipeline_transaction(recipes, reversed_libraries);
    const auto* reversed_transaction =
        std::get_if<MetalPipelineTransaction>(&reversed);
    CHECK(reversed_transaction != nullptr);
    if (reversed_transaction) {
        CHECK(reversed_transaction->generation_digest() ==
              transaction->generation_digest());
        CHECK(reversed_transaction->slot_count() == transaction->slot_count());
        for (uint32_t index = 0; index < transaction->slot_count(); ++index) {
            CHECK(reversed_transaction->slot(index)->slot ==
                  transaction->slot(index)->slot);
            CHECK(reversed_transaction->slot(index)->requirement_digest ==
                  transaction->slot(index)->requirement_digest);
        }
    }

    auto changed_compile = libraries;
    changed_compile[0].compile_contract.fast_math_enabled =
        !changed_compile[0].compile_contract.fast_math_enabled;
    const auto changed = build_metal_pipeline_transaction(recipes, changed_compile);
    if (const auto* changed_transaction =
            std::get_if<MetalPipelineTransaction>(&changed))
        CHECK(changed_transaction->generation_digest() !=
              transaction->generation_digest());
}
#endif

} // namespace

int main() {
    test_contract_rejection();
    test_multi_library_contract();
#if defined(LAPLACE_TESTING)
    test_allocation_failure_boundary();
#endif
    MetalPipelineTransaction transaction;
    MetalPipelineTransactionFailure failure;
    if (!build_native_pair(&transaction, &failure)) {
        if (failure.code == MetalPipelineTransactionError::NoDevice) {
            printf("SKIP: no Metal device\n");
            return test_summary("test_metal_pipeline_transaction");
        }
        CHECK(false);
        return test_summary("test_metal_pipeline_transaction");
    }
    test_transaction_and_lease();
    test_dedup_and_determinism();
    test_effective_dispatch_limit();
#if defined(LAPLACE_TESTING)
    test_injected_failures();
    test_multi_library_native();
    test_pipeline_lease_session_boundary();
    test_ordered_leased_program_stream();
    test_moe_reduce_stream_order();
    test_dense_leased_repeated_dispatches();
    test_dense_leased_prefill_repeated_dispatches();
    test_recurrent_leased_exact_range();
#endif
    return test_summary("test_metal_pipeline_transaction");
}
