#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <variant>

namespace Laplace {

using MetalPipelineDigest = std::array<uint8_t, 32>;

enum class MetalPipelineLanguageVersion : uint8_t {
    V2_0 = 1,
    V2_1 = 2,
    V2_2 = 3,
    V2_3 = 4,
    V2_4 = 5,
    V3_0 = 6,
    V3_1 = 7,
    V3_2 = 8,
    V4_0 = 9,
    V4_1 = 10,
};

// These are the source-level settings that affect Metal library semantics.
// The transaction applies them and derives the resulting identity internally;
// callers never provide a compiler/build digest.
struct MetalPipelineCompileContract {
    uint8_t version = 1;
    MetalPipelineLanguageVersion language_version =
        MetalPipelineLanguageVersion::V2_0;
    bool fast_math_enabled = true;
    uint8_t reserved = 0;

    friend bool operator==(const MetalPipelineCompileContract&,
                           const MetalPipelineCompileContract&) = default;
};

enum class MetalFunctionConstantType : uint8_t {
    Bool = 1,
    Int32 = 2,
    UInt32 = 3,
    Float32 = 4,
};

// The bit pattern is interpreted according to `type`. Keeping the value in a
// fixed-width field makes recipe equality independent of host ABI padding.
struct MetalFunctionConstant {
    uint32_t index = 0;
    MetalFunctionConstantType type = MetalFunctionConstantType::UInt32;
    uint64_t value_bits = 0;

    friend bool operator==(const MetalFunctionConstant&,
                           const MetalFunctionConstant&) = default;
};

struct MetalDispatchConstraints {
    uint32_t min_threads_per_threadgroup = 1;
    uint32_t max_threads_per_threadgroup = 0;
    uint32_t min_thread_execution_width = 0;
    uint32_t max_thread_execution_width = 0;
    uint32_t required_simdgroups = 0;

    friend bool operator==(const MetalDispatchConstraints&,
                           const MetalDispatchConstraints&) = default;
};

// Immutable bytes and their construction identities for one separately
// compiled Metal library. The span is borrowed only for transaction build.
struct MetalPipelineLibrarySource {
    std::span<const uint8_t> bytes;
    MetalPipelineDigest source_digest{};
    MetalPipelineCompileContract compile_contract{};
};

// This is the private compiler output. It contains no package or external
// selector. The library source digest resolves it to a verified compile
// contract owned by the transaction input.
struct MetalPipelineRecipe {
    MetalPipelineDigest normalized_requirement_digest{};
    std::string function_name;
    std::vector<MetalFunctionConstant> function_constants;
    MetalPipelineDigest library_source_digest{};
    MetalDispatchConstraints dispatch;

    friend bool operator==(const MetalPipelineRecipe&,
                           const MetalPipelineRecipe&) = default;
};

enum class MetalPipelineTransactionError : uint8_t {
    InvalidRequest = 1,
    NoDevice = 2,
    LibraryCompileFailed = 3,
    FunctionLookupFailed = 4,
    PipelineBuildFailed = 5,
    PipelineLimitsInvalid = 6,
    DeviceMismatch = 7,
    AllocationFailed = 8,
};

struct MetalPipelineTransactionAudit {
    uint32_t devices_created = 0;
    uint32_t libraries_created = 0;
    uint32_t functions_created = 0;
    uint32_t pipelines_created = 0;
    uint32_t command_queues_created = 0;
    uint32_t command_buffers_created = 0;
    uint32_t buffers_created = 0;
    uint32_t sessions_created = 0;
    uint32_t global_cache_mutations = 0;
    uint32_t environment_mutations = 0;
    bool transaction_published = false;

    friend bool operator==(const MetalPipelineTransactionAudit&,
                           const MetalPipelineTransactionAudit&) = default;
};

struct MetalPipelineTransactionFailure {
    MetalPipelineTransactionError code = MetalPipelineTransactionError::InvalidRequest;
    std::string detail;
    MetalPipelineTransactionAudit audit;
};

#if defined(LAPLACE_TESTING)
enum class MetalPipelineInjectedFailure : uint8_t {
    None = 0,
    Library = 1,
    Function = 2,
    Pipeline = 3,
    PipelineAfterFirst = 4,
    Allocation = 5,
};
#endif

struct MetalPipelineSlotInfo {
    uint32_t slot = 0;
    MetalPipelineDigest requirement_digest{};
    uint32_t thread_execution_width = 0;
    uint32_t max_total_threads_per_threadgroup = 0;
    uint32_t dispatch_minimum = 0;
    uint32_t dispatch_maximum = 0;
    // Opaque identity token for diagnostics and the eventual session binder.
    // The transaction remains the owner of the underlying Metal object.
    const void* pipeline_identity = nullptr;
};

class MetalPipelineLease;
struct MetalPipelineStorage;

class MetalPipelineTransaction {
public:
    MetalPipelineTransaction() noexcept;
    ~MetalPipelineTransaction();
    MetalPipelineTransaction(MetalPipelineTransaction&&) noexcept;
    MetalPipelineTransaction& operator=(MetalPipelineTransaction&&) noexcept;
    MetalPipelineTransaction(const MetalPipelineTransaction&) = delete;
    MetalPipelineTransaction& operator=(const MetalPipelineTransaction&) = delete;

    bool valid() const noexcept;
    uint32_t slot_count() const noexcept;
    const MetalPipelineSlotInfo* slot(uint32_t index) const noexcept;
    const MetalPipelineSlotInfo* find(MetalPipelineDigest requirement) const noexcept;
    MetalPipelineDigest generation_digest() const noexcept;
    const MetalPipelineTransactionAudit& audit() const noexcept;
    // Consumes this transaction. The returned lease owns the exact device and
    // pipeline objects. A consumed transaction is invalid and cannot be reused.
    MetalPipelineLease take_lease() && noexcept;

private:
    explicit MetalPipelineTransaction(std::unique_ptr<MetalPipelineStorage> storage);
    std::unique_ptr<MetalPipelineStorage> storage_;

    friend std::variant<MetalPipelineTransaction, MetalPipelineTransactionFailure>
    build_metal_pipeline_transaction(std::span<const MetalPipelineRecipe>,
                                     std::span<const uint8_t>);
    friend std::variant<MetalPipelineTransaction, MetalPipelineTransactionFailure>
    build_metal_pipeline_transaction(
        std::span<const MetalPipelineRecipe>,
        std::span<const MetalPipelineLibrarySource>);
    friend std::variant<MetalPipelineTransaction, MetalPipelineTransactionFailure>
    build_metal_pipeline_transaction(
        std::span<const MetalPipelineRecipe>, std::span<const uint8_t>,
        const MetalPipelineCompileContract&);
    friend std::variant<MetalPipelineTransaction, MetalPipelineTransactionFailure>
    build_impl(std::span<const MetalPipelineRecipe>, std::span<const uint8_t>, uint8_t);
    friend std::variant<MetalPipelineTransaction, MetalPipelineTransactionFailure>
    build_impl(std::span<const MetalPipelineRecipe>, std::span<const uint8_t>,
               const MetalPipelineCompileContract&, uint8_t);
    friend std::variant<MetalPipelineTransaction, MetalPipelineTransactionFailure>
    build_impl(std::span<const MetalPipelineRecipe>,
               std::span<const MetalPipelineLibrarySource>, uint8_t, bool);
#if defined(LAPLACE_TESTING)
    friend std::variant<MetalPipelineTransaction, MetalPipelineTransactionFailure>
    build_metal_pipeline_transaction_for_testing(
        std::span<const MetalPipelineRecipe>, std::span<const uint8_t>,
        MetalPipelineInjectedFailure);
    friend std::variant<MetalPipelineTransaction, MetalPipelineTransactionFailure>
    build_metal_pipeline_transaction_for_testing(
        std::span<const MetalPipelineRecipe>, std::span<const uint8_t>,
        const MetalPipelineCompileContract&, MetalPipelineInjectedFailure);
    friend std::variant<MetalPipelineTransaction, MetalPipelineTransactionFailure>
    build_metal_pipeline_transaction_for_testing(
        std::span<const MetalPipelineRecipe>,
        std::span<const MetalPipelineLibrarySource>,
        MetalPipelineInjectedFailure);
#endif
};

class MetalPipelineLease {
public:
    ~MetalPipelineLease();
    MetalPipelineLease(MetalPipelineLease&&) noexcept;
    MetalPipelineLease& operator=(MetalPipelineLease&&) noexcept;
    MetalPipelineLease(const MetalPipelineLease&) = delete;
    MetalPipelineLease& operator=(const MetalPipelineLease&) = delete;

    bool valid() const noexcept;
    uint32_t slot_count() const noexcept;
    const MetalPipelineSlotInfo* slot(uint32_t index) const noexcept;
    const MetalPipelineSlotInfo* find(MetalPipelineDigest requirement) const noexcept;
    MetalPipelineDigest generation_digest() const noexcept;
    const MetalPipelineTransactionAudit& audit() const noexcept;

    // These are opaque Objective-C object tokens. They are intentionally
    // exposed as const void* so C++ planner code cannot bind Metal objects.
    const void* device_token() const noexcept;
    const void* pipeline_token(uint32_t index) const noexcept;
    bool matches_device_token(const void* device) const noexcept;
    // Resolves a complete canonical recipe to its owned slot and pipeline.
    // Failure leaves both output arguments unchanged.
    bool resolve_recipe(const MetalPipelineRecipe& recipe, uint32_t* slot,
                        const void** pipeline) const noexcept;

private:
    explicit MetalPipelineLease(std::unique_ptr<MetalPipelineStorage> storage);
    std::unique_ptr<MetalPipelineStorage> storage_;

    friend class MetalPipelineTransaction;
};

using MetalPipelineTransactionResult =
    std::variant<MetalPipelineTransaction, MetalPipelineTransactionFailure>;

MetalPipelineTransactionResult build_metal_pipeline_transaction(
    std::span<const MetalPipelineRecipe> recipes,
    std::span<const uint8_t> library_source);

MetalPipelineTransactionResult build_metal_pipeline_transaction(
    std::span<const MetalPipelineRecipe> recipes,
    std::span<const uint8_t> library_source,
    const MetalPipelineCompileContract& compile_contract);

MetalPipelineTransactionResult build_metal_pipeline_transaction(
    std::span<const MetalPipelineRecipe> recipes,
    std::span<const MetalPipelineLibrarySource> library_sources);

MetalPipelineDigest metal_pipeline_digest(std::span<const uint8_t> bytes) noexcept;

#if defined(LAPLACE_TESTING)
MetalPipelineTransactionResult build_metal_pipeline_transaction_for_testing(
    std::span<const MetalPipelineRecipe> recipes,
    std::span<const uint8_t> library_source,
    MetalPipelineInjectedFailure failure);
MetalPipelineTransactionResult build_metal_pipeline_transaction_for_testing(
    std::span<const MetalPipelineRecipe> recipes,
    std::span<const uint8_t> library_source,
    const MetalPipelineCompileContract& compile_contract,
    MetalPipelineInjectedFailure failure);
MetalPipelineTransactionResult build_metal_pipeline_transaction_for_testing(
    std::span<const MetalPipelineRecipe> recipes,
    std::span<const MetalPipelineLibrarySource> library_sources,
    MetalPipelineInjectedFailure failure);
uint32_t metal_pipeline_live_pipeline_owners_for_testing() noexcept;
#endif

} // namespace Laplace
