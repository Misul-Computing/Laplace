#include "metal_library_source_catalog.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

using namespace Laplace;

namespace {

int checks = 0;

void check(bool value, const char* message) {
    ++checks;
    if (!value) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

template <class T>
std::array<T, MetalLibrarySourceCatalog::kLibraryCount> copy_records(
    std::span<const T> records) {
    std::array<T, MetalLibrarySourceCatalog::kLibraryCount> copy{};
    std::copy(records.begin(), records.end(), copy.begin());
    return copy;
}

void test_catalog() {
    const auto& catalog = product_metal_library_source_catalog();
    check(catalog.valid(), "product catalog is valid");
    check(catalog.records().size() == 3, "catalog has three records");
    check(catalog.compiler_identities().size() == 3,
          "compiler identity count matches catalog");
    check(catalog.transaction_sources().size() == 3,
          "transaction source count matches catalog");

    for (size_t index = 0; index != catalog.records().size(); ++index) {
        const auto& record = catalog.records()[index];
        check(!record.bytes.empty(), "source bytes are present");
        check(record.bytes.back() != 0, "source span has no trailing NUL");
        check(record.source_digest == catalog.compiler_identities()[index].source_digest,
              "compiler identity uses record digest");
        check(record.bytes.data() == catalog.transaction_sources()[index].bytes.data() &&
                  record.bytes.size() == catalog.transaction_sources()[index].bytes.size(),
              "transaction source uses record span");
        check(record.source_digest ==
                  metal_pipeline_digest(record.bytes),
              "record digest matches source bytes");
        check(record.compile_contract.version == 1 &&
                  record.compile_contract.reserved == 0,
              "record has an explicit valid compile contract");
        check(record.compile_contract.version ==
                  catalog.transaction_sources()[index].compile_contract.version &&
                  record.compile_contract.language_version ==
                      catalog.transaction_sources()[index].compile_contract.language_version &&
                  record.compile_contract.fast_math_enabled ==
                      catalog.transaction_sources()[index].compile_contract.fast_math_enabled &&
                  record.compile_contract.reserved ==
                      catalog.transaction_sources()[index].compile_contract.reserved,
              "transaction source uses record compile contract");
        check(catalog.find(record.id) == &record, "record lookup is stable");
    }

    auto records = copy_records(catalog.records());
    std::reverse(records.begin(), records.end());
    const auto reordered = validate_metal_library_source_catalog(records);
    check(reordered.ok(), "caller order does not affect validation");

#if defined(LAPLACE_TESTING)
    const auto normalized = make_metal_library_source_catalog_for_testing(records);
    check(std::holds_alternative<MetalLibrarySourceCatalog>(normalized),
          "reordered records normalize");
    const auto& normalized_catalog = std::get<MetalLibrarySourceCatalog>(normalized);
    check(normalized_catalog.records()[0].id == StructuralMetalLibraryId::Core,
          "normalized order starts with core");
    check(normalized_catalog.records()[1].id == StructuralMetalLibraryId::Prefill,
          "normalized order puts prefill second");
    check(normalized_catalog.records()[2].id == StructuralMetalLibraryId::Sampler,
          "normalized order puts sampler third");

    auto malformed = records;
    malformed[0].source_digest[0] ^= 1;
    check(!validate_metal_library_source_catalog(malformed).ok(),
          "changed digest is rejected");

    malformed = records;
    malformed[0].compile_contract.version = 0;
    check(!validate_metal_library_source_catalog(malformed).ok(),
          "invalid compile contract is rejected");

    auto changed_contract = records;
    changed_contract[0].compile_contract.fast_math_enabled =
        !changed_contract[0].compile_contract.fast_math_enabled;
    const auto changed_catalog =
        make_metal_library_source_catalog_for_testing(changed_contract);
    check(std::holds_alternative<MetalLibrarySourceCatalog>(changed_catalog),
          "valid compile contract variant remains structurally valid");
    const auto unchanged_catalog =
        make_metal_library_source_catalog_for_testing(records);
    check(std::get<MetalLibrarySourceCatalog>(changed_catalog).catalog_digest() !=
              std::get<MetalLibrarySourceCatalog>(unchanged_catalog).catalog_digest(),
          "compile contract changes catalog identity");

    malformed = records;
    std::array<uint8_t, 8> replacement = {1, 2, 3, 4, 5, 6, 7, 8};
    malformed[0].bytes = replacement;
    check(!validate_metal_library_source_catalog(malformed).ok(),
          "changed source bytes are rejected");

    malformed = records;
    malformed[1].id = malformed[0].id;
    check(!validate_metal_library_source_catalog(malformed).ok(),
          "duplicate source ID is rejected");

    malformed = records;
    malformed[1].source_digest = malformed[0].source_digest;
    check(!validate_metal_library_source_catalog(malformed).ok(),
          "duplicate source digest is rejected");

    malformed = records;
    malformed[0].bytes = std::span<const uint8_t>(
        malformed[0].bytes.data(), malformed[0].bytes.size() - 1);
    check(!validate_metal_library_source_catalog(malformed).ok(),
          "truncated source is rejected");
#endif
}

void test_transaction_source_views() {
    const auto& catalog = product_metal_library_source_catalog();
    std::vector<MetalPipelineRecipe> recipes;
    recipes.reserve(catalog.records().size());
    const std::array<const char*, MetalLibrarySourceCatalog::kLibraryCount>
        functions = {"gemv", "prefill_f16_rows", "sampler_greedy_f32"};
    for (size_t index = 0; index != catalog.records().size(); ++index) {
        MetalPipelineRecipe recipe;
        recipe.normalized_requirement_digest[0] =
            static_cast<uint8_t>(index + 1);
        recipe.function_name = functions[index];
        recipe.library_source_digest = catalog.records()[index].source_digest;
        recipe.dispatch = {1, 1024, 0, 0, 0};
        if (index == 0)
            recipe.function_constants = {
                {0, MetalFunctionConstantType::Int32, 1}};
        recipes.push_back(std::move(recipe));
    }

    const auto result = build_metal_pipeline_transaction(
        recipes, catalog.transaction_sources());
    if (std::holds_alternative<MetalPipelineTransactionFailure>(result)) {
        const auto& failure = std::get<MetalPipelineTransactionFailure>(result);
        if (failure.code == MetalPipelineTransactionError::NoDevice) {
            std::cout << "transaction source compile: SKIP (no Metal device)\n";
            return;
        }
        std::cerr << "FAIL: transaction source compile: " << failure.detail << '\n';
        std::exit(1);
    }
    const auto& transaction = std::get<MetalPipelineTransaction>(result);
    check(transaction.valid(), "catalog sources build a Metal transaction");
    check(transaction.slot_count() == catalog.records().size(),
          "transaction has one slot per catalog source");
}

} // namespace

int main() {
    test_catalog();
    test_transaction_source_views();
    std::cout << "test_metal_library_source_catalog: OK (" << checks
              << " checks)\n";
    return 0;
}
