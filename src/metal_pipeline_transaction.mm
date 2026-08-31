#include "metal_pipeline_transaction.h"

#include <Metal/Metal.h>

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <limits>
#include <memory>
#include <new>
#include <string_view>
#include <utility>

namespace Laplace {

namespace {

constexpr size_t kMaxRecipes = 4096;
constexpr size_t kMaxLibraries = 256;
constexpr size_t kMaxFunctionName = 256;
constexpr size_t kMaxFunctionConstants = 64;
constexpr size_t kMaxLibrarySource = 16u * 1024u * 1024u;

enum class InjectedFailure : uint8_t {
    None = 0,
    Library = 1,
    Function = 2,
    Pipeline = 3,
    PipelineAfterFirst = 4,
    Allocation = 5,
};

#if defined(LAPLACE_TESTING)
std::atomic_uint32_t g_live_pipeline_owners{0};
#endif

bool zero_digest(const MetalPipelineDigest& digest) noexcept {
    for (const uint8_t byte : digest)
        if (byte != 0) return false;
    return true;
}

void append_u8(std::vector<uint8_t>& bytes, uint8_t value) {
    bytes.push_back(value);
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void append_digest(std::vector<uint8_t>& bytes, const MetalPipelineDigest& digest) {
    bytes.insert(bytes.end(), digest.begin(), digest.end());
}

bool append_string(std::vector<uint8_t>& bytes, std::string_view value) {
    if (value.size() > std::numeric_limits<uint32_t>::max()) return false;
    append_u32(bytes, static_cast<uint32_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
    return true;
}

MetalPipelineDigest digest_bytes(std::span<const uint8_t> bytes) noexcept {
    MetalPipelineDigest digest{};
    if (bytes.size() > static_cast<size_t>(std::numeric_limits<CC_LONG>::max()))
        return digest;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
    return digest;
}

bool compile_contract_valid(
    const MetalPipelineCompileContract& contract) noexcept {
    return contract.version == 1 && contract.reserved == 0 &&
           static_cast<uint8_t>(contract.language_version) >=
               static_cast<uint8_t>(MetalPipelineLanguageVersion::V2_0) &&
           static_cast<uint8_t>(contract.language_version) <=
               static_cast<uint8_t>(MetalPipelineLanguageVersion::V4_1);
}

MetalPipelineDigest compile_contract_digest(
    const MetalPipelineCompileContract& contract, bool modern_math) noexcept {
    constexpr std::string_view domain = "laplace.metal.compile-contract.v1";
    std::array<uint8_t, 64> bytes{};
    size_t offset = 0;
    const auto append = [&bytes, &offset](uint8_t value) noexcept {
        if (offset < bytes.size()) bytes[offset++] = value;
    };
    append(static_cast<uint8_t>(domain.size()));
    for (const char value : domain) append(static_cast<uint8_t>(value));
    append(contract.version);
    append(static_cast<uint8_t>(contract.language_version));
    append(contract.fast_math_enabled ? 1 : 0);
    append(contract.reserved);
    append(modern_math ? 1 : 0);
    append(modern_math ? (contract.fast_math_enabled ? 2 : 0)
                       : (contract.fast_math_enabled ? 1 : 0));
    append(modern_math ? (contract.fast_math_enabled ? 0 : 1) : 0);
    MetalPipelineDigest digest{};
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(offset), digest.data());
    return digest;
}

bool set_language_version(MTLCompileOptions* options,
                          MetalPipelineLanguageVersion version) {
    switch (version) {
    case MetalPipelineLanguageVersion::V2_0:
        options.languageVersion = MTLLanguageVersion2_0;
        return true;
    case MetalPipelineLanguageVersion::V2_1:
        if (@available(macOS 10.14, *)) {
            options.languageVersion = MTLLanguageVersion2_1;
            return true;
        }
        return false;
    case MetalPipelineLanguageVersion::V2_2:
        if (@available(macOS 10.15, *)) {
            options.languageVersion = MTLLanguageVersion2_2;
            return true;
        }
        return false;
    case MetalPipelineLanguageVersion::V2_3:
        if (@available(macOS 11.0, *)) {
            options.languageVersion = MTLLanguageVersion2_3;
            return true;
        }
        return false;
    case MetalPipelineLanguageVersion::V2_4:
        if (@available(macOS 12.0, *)) {
            options.languageVersion = MTLLanguageVersion2_4;
            return true;
        }
        return false;
    case MetalPipelineLanguageVersion::V3_0:
        if (@available(macOS 13.0, *)) {
            options.languageVersion = MTLLanguageVersion3_0;
            return true;
        }
        return false;
    case MetalPipelineLanguageVersion::V3_1:
        if (@available(macOS 14.0, *)) {
            options.languageVersion = MTLLanguageVersion3_1;
            return true;
        }
        return false;
    case MetalPipelineLanguageVersion::V3_2:
        if (@available(macOS 15.0, *)) {
            options.languageVersion = MTLLanguageVersion3_2;
            return true;
        }
        return false;
    case MetalPipelineLanguageVersion::V4_0:
        if (@available(macOS 26.0, *)) {
            options.languageVersion = MTLLanguageVersion4_0;
            return true;
        }
        return false;
    case MetalPipelineLanguageVersion::V4_1:
        if (@available(macOS 27.0, *)) {
            options.languageVersion = MTLLanguageVersion4_1;
            return true;
        }
        return false;
    }
    return false;
}

bool apply_compile_contract(MTLCompileOptions* options,
                            const MetalPipelineCompileContract& contract,
                            bool* modern_math) {
    if (!compile_contract_valid(contract) ||
        !set_language_version(options, contract.language_version))
        return false;
    if (@available(macOS 15.0, *)) {
        options.mathMode = contract.fast_math_enabled ? MTLMathModeFast
                                                       : MTLMathModeSafe;
        options.mathFloatingPointFunctions =
            contract.fast_math_enabled ? MTLMathFloatingPointFunctionsFast
                                       : MTLMathFloatingPointFunctionsPrecise;
        *modern_math = true;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = contract.fast_math_enabled;
#pragma clang diagnostic pop
        *modern_math = false;
    }
    return true;
}

bool constant_type_valid(MetalFunctionConstantType type) noexcept {
    switch (type) {
    case MetalFunctionConstantType::Bool:
    case MetalFunctionConstantType::Int32:
    case MetalFunctionConstantType::UInt32:
    case MetalFunctionConstantType::Float32:
        return true;
    }
    return false;
}

bool dispatch_valid(const MetalDispatchConstraints& dispatch) noexcept {
    if (dispatch.min_threads_per_threadgroup == 0 ||
        dispatch.max_threads_per_threadgroup == 0 ||
        dispatch.min_threads_per_threadgroup > dispatch.max_threads_per_threadgroup)
        return false;
    if (dispatch.max_thread_execution_width != 0 &&
        dispatch.min_thread_execution_width > dispatch.max_thread_execution_width)
        return false;
    if (dispatch.required_simdgroups != 0 &&
        dispatch.required_simdgroups >
            std::numeric_limits<uint32_t>::max() / 1024u)
        return false;
    return true;
}

bool recipe_shape_valid(const MetalPipelineRecipe& recipe) noexcept {
    if (zero_digest(recipe.normalized_requirement_digest) ||
        zero_digest(recipe.library_source_digest) || recipe.function_name.empty() ||
        recipe.function_name.size() > kMaxFunctionName ||
        recipe.function_name.find('\0') != std::string::npos ||
        recipe.function_constants.size() > kMaxFunctionConstants ||
        !dispatch_valid(recipe.dispatch))
        return false;

    for (size_t i = 0; i < recipe.function_constants.size(); ++i) {
        const MetalFunctionConstant& constant = recipe.function_constants[i];
        if (constant.index >= 1024 || !constant_type_valid(constant.type)) return false;
        if (constant.type == MetalFunctionConstantType::Bool &&
            constant.value_bits > 1)
            return false;
        if (constant.type != MetalFunctionConstantType::Bool &&
            constant.value_bits > std::numeric_limits<uint32_t>::max())
            return false;
        for (size_t prior = 0; prior != i; ++prior)
            if (recipe.function_constants[prior].index == constant.index)
                return false;
    }
    return true;
}

bool recipe_less(const MetalPipelineRecipe& left,
                 const MetalPipelineRecipe& right) noexcept {
    if (left.normalized_requirement_digest != right.normalized_requirement_digest)
        return std::lexicographical_compare(
            left.normalized_requirement_digest.begin(),
            left.normalized_requirement_digest.end(),
            right.normalized_requirement_digest.begin(),
            right.normalized_requirement_digest.end());
    if (left.function_name != right.function_name)
        return left.function_name < right.function_name;
    if (left.library_source_digest != right.library_source_digest)
        return std::lexicographical_compare(
            left.library_source_digest.begin(), left.library_source_digest.end(),
            right.library_source_digest.begin(), right.library_source_digest.end());
    if (left.dispatch.min_threads_per_threadgroup !=
        right.dispatch.min_threads_per_threadgroup)
        return left.dispatch.min_threads_per_threadgroup <
               right.dispatch.min_threads_per_threadgroup;
    if (left.dispatch.max_threads_per_threadgroup !=
        right.dispatch.max_threads_per_threadgroup)
        return left.dispatch.max_threads_per_threadgroup <
               right.dispatch.max_threads_per_threadgroup;
    if (left.dispatch.min_thread_execution_width !=
        right.dispatch.min_thread_execution_width)
        return left.dispatch.min_thread_execution_width <
               right.dispatch.min_thread_execution_width;
    if (left.dispatch.max_thread_execution_width !=
        right.dispatch.max_thread_execution_width)
        return left.dispatch.max_thread_execution_width <
               right.dispatch.max_thread_execution_width;
    if (left.dispatch.required_simdgroups != right.dispatch.required_simdgroups)
        return left.dispatch.required_simdgroups < right.dispatch.required_simdgroups;
    return std::lexicographical_compare(
        left.function_constants.begin(), left.function_constants.end(),
        right.function_constants.begin(), right.function_constants.end(),
        [](const MetalFunctionConstant& a, const MetalFunctionConstant& b) {
            if (a.index != b.index) return a.index < b.index;
            if (a.type != b.type)
                return static_cast<uint8_t>(a.type) < static_cast<uint8_t>(b.type);
            return a.value_bits < b.value_bits;
        });
}

bool recipe_equal(const MetalPipelineRecipe& left,
                  const MetalPipelineRecipe& right) noexcept {
    if (left.normalized_requirement_digest != right.normalized_requirement_digest)
        return false;
    if (left.function_name != right.function_name ||
        left.library_source_digest != right.library_source_digest ||
        left.dispatch != right.dispatch ||
        left.function_constants.size() != right.function_constants.size())
        return false;
    for (size_t index = 0; index != left.function_constants.size(); ++index)
        if (!(left.function_constants[index] == right.function_constants[index]))
            return false;
    return true;
}

bool recipe_equal_canonical(const MetalPipelineRecipe& stored,
                            const MetalPipelineRecipe& query) noexcept {
    if (!recipe_shape_valid(query) ||
        stored.normalized_requirement_digest != query.normalized_requirement_digest ||
        stored.function_name != query.function_name ||
        stored.library_source_digest != query.library_source_digest ||
        stored.dispatch != query.dispatch ||
        stored.function_constants.size() != query.function_constants.size())
        return false;

    for (const MetalFunctionConstant& stored_constant : stored.function_constants) {
        bool found = false;
        for (const MetalFunctionConstant& query_constant : query.function_constants) {
            if (stored_constant == query_constant) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

bool serialize_recipe(std::vector<uint8_t>& bytes,
                      const MetalPipelineRecipe& recipe) {
    append_digest(bytes, recipe.normalized_requirement_digest);
    if (!append_string(bytes, recipe.function_name)) return false;
    append_digest(bytes, recipe.library_source_digest);
    append_u32(bytes, recipe.dispatch.min_threads_per_threadgroup);
    append_u32(bytes, recipe.dispatch.max_threads_per_threadgroup);
    append_u32(bytes, recipe.dispatch.min_thread_execution_width);
    append_u32(bytes, recipe.dispatch.max_thread_execution_width);
    append_u32(bytes, recipe.dispatch.required_simdgroups);
    std::vector<MetalFunctionConstant> constants = recipe.function_constants;
    std::sort(constants.begin(), constants.end(), [](const auto& left, const auto& right) {
        if (left.index != right.index) return left.index < right.index;
        if (left.type != right.type)
            return static_cast<uint8_t>(left.type) < static_cast<uint8_t>(right.type);
        return left.value_bits < right.value_bits;
    });
    append_u32(bytes, static_cast<uint32_t>(constants.size()));
    for (const MetalFunctionConstant& constant : constants) {
        append_u32(bytes, constant.index);
        append_u8(bytes, static_cast<uint8_t>(constant.type));
        append_u64(bytes, constant.value_bits);
    }
    return true;
}

struct NormalizedRecipes {
    std::vector<MetalPipelineRecipe> recipes;
    std::vector<MetalPipelineLibrarySource> libraries;
};

std::variant<NormalizedRecipes, MetalPipelineTransactionFailure> normalize_recipes(
    std::span<const MetalPipelineRecipe> input,
    std::span<const MetalPipelineLibrarySource> input_libraries,
    bool allow_exact_duplicates) {
    MetalPipelineTransactionAudit audit;
    if (input.empty() || input.size() > kMaxRecipes || input_libraries.empty() ||
        input_libraries.size() > kMaxLibraries)
        return MetalPipelineTransactionFailure{
            MetalPipelineTransactionError::InvalidRequest,
            "recipe list and library sources must be nonempty and bounded", audit};

    NormalizedRecipes normalized;
    normalized.recipes.assign(input.begin(), input.end());
    normalized.libraries.assign(input_libraries.begin(), input_libraries.end());

    for (size_t index = 0; index < normalized.libraries.size(); ++index) {
        const MetalPipelineLibrarySource& library = normalized.libraries[index];
        if (library.bytes.empty() || library.bytes.size() > kMaxLibrarySource ||
            library.bytes.size() >
                static_cast<size_t>(std::numeric_limits<CC_LONG>::max()) ||
            zero_digest(library.source_digest) ||
            !compile_contract_valid(library.compile_contract) ||
            digest_bytes(library.bytes) != library.source_digest)
            return MetalPipelineTransactionFailure{
                MetalPipelineTransactionError::InvalidRequest,
                "library source identity does not match its bytes", audit};
        for (size_t prior = 0; prior < index; ++prior)
            if (normalized.libraries[prior].source_digest == library.source_digest)
                return MetalPipelineTransactionFailure{
                    MetalPipelineTransactionError::InvalidRequest,
                    "library source digest is supplied more than once", audit};
    }

    std::sort(normalized.libraries.begin(), normalized.libraries.end(),
              [](const MetalPipelineLibrarySource& left,
                 const MetalPipelineLibrarySource& right) {
                  if (left.source_digest != right.source_digest)
                      return std::lexicographical_compare(
                          left.source_digest.begin(), left.source_digest.end(),
                          right.source_digest.begin(), right.source_digest.end());
                  if (left.compile_contract.version !=
                      right.compile_contract.version)
                      return left.compile_contract.version <
                             right.compile_contract.version;
                  if (left.compile_contract.language_version !=
                      right.compile_contract.language_version)
                      return static_cast<uint8_t>(left.compile_contract.language_version) <
                             static_cast<uint8_t>(right.compile_contract.language_version);
                  if (left.compile_contract.fast_math_enabled !=
                      right.compile_contract.fast_math_enabled)
                      return left.compile_contract.fast_math_enabled <
                             right.compile_contract.fast_math_enabled;
                  return left.compile_contract.reserved <
                         right.compile_contract.reserved;
              });

    std::vector<bool> used_libraries(normalized.libraries.size(), false);
    for (const MetalPipelineRecipe& recipe : normalized.recipes) {
        if (!recipe_shape_valid(recipe))
            return MetalPipelineTransactionFailure{
                MetalPipelineTransactionError::InvalidRequest,
                "recipe has an invalid identity, constant, function, or limit", audit};
        size_t matching_library = normalized.libraries.size();
        for (size_t index = 0; index < normalized.libraries.size(); ++index) {
            if (normalized.libraries[index].source_digest ==
                recipe.library_source_digest) {
                matching_library = index;
                break;
            }
        }
        if (matching_library == normalized.libraries.size())
            return MetalPipelineTransactionFailure{
                MetalPipelineTransactionError::InvalidRequest,
                "recipe library source digest does not resolve to one library", audit};
        used_libraries[matching_library] = true;
    }

    for (bool used : used_libraries)
        if (!used)
            return MetalPipelineTransactionFailure{
                MetalPipelineTransactionError::InvalidRequest,
                "every supplied library must be used by a recipe", audit};

    for (MetalPipelineRecipe& recipe : normalized.recipes) {
        std::sort(recipe.function_constants.begin(), recipe.function_constants.end(),
                  [](const auto& left, const auto& right) {
                      if (left.index != right.index) return left.index < right.index;
                      if (left.type != right.type)
                          return static_cast<uint8_t>(left.type) <
                                 static_cast<uint8_t>(right.type);
                      return left.value_bits < right.value_bits;
                  });
    }

    std::sort(normalized.recipes.begin(), normalized.recipes.end(), recipe_less);
    std::vector<MetalPipelineRecipe> unique;
    unique.reserve(normalized.recipes.size());
    for (const MetalPipelineRecipe& recipe : normalized.recipes) {
        if (!unique.empty() && unique.back().normalized_requirement_digest ==
                                   recipe.normalized_requirement_digest) {
            if (!recipe_equal(unique.back(), recipe))
                return MetalPipelineTransactionFailure{
                    MetalPipelineTransactionError::InvalidRequest,
                    "one requirement digest has conflicting recipes", audit};
            if (!allow_exact_duplicates)
                return MetalPipelineTransactionFailure{
                    MetalPipelineTransactionError::InvalidRequest,
                    "one requirement has duplicate recipes", audit};
            continue;
        }
        unique.push_back(recipe);
    }
    normalized.recipes = std::move(unique);
    return normalized;
}

uint32_t checked_u32(NSUInteger value, bool* ok) noexcept {
    if (value > std::numeric_limits<uint32_t>::max()) {
        *ok = false;
        return 0;
    }
    return static_cast<uint32_t>(value);
}

uint32_t dispatch_minimum(const MetalPipelineRecipe& recipe,
                          uint32_t thread_width, bool* ok) noexcept {
    uint64_t required = recipe.dispatch.min_threads_per_threadgroup;
    if (recipe.dispatch.required_simdgroups != 0)
        required = std::max<uint64_t>(
            required, static_cast<uint64_t>(thread_width) *
                          recipe.dispatch.required_simdgroups);
    if (required > std::numeric_limits<uint32_t>::max()) {
        *ok = false;
        return 0;
    }
    return static_cast<uint32_t>(required);
}

uint32_t dispatch_maximum(const MetalPipelineRecipe& recipe,
                          uint32_t pipeline_max, uint32_t device_max) noexcept {
    return std::min({recipe.dispatch.max_threads_per_threadgroup,
                     pipeline_max, device_max});
}

void set_function_constant(MTLFunctionConstantValues* values,
                           const MetalFunctionConstant& constant) {
    switch (constant.type) {
    case MetalFunctionConstantType::Bool: {
        bool value = constant.value_bits != 0;
        [values setConstantValue:&value type:MTLDataTypeBool atIndex:constant.index];
        return;
    }
    case MetalFunctionConstantType::Int32: {
        int32_t value = static_cast<int32_t>(constant.value_bits);
        [values setConstantValue:&value type:MTLDataTypeInt atIndex:constant.index];
        return;
    }
    case MetalFunctionConstantType::UInt32: {
        uint32_t value = static_cast<uint32_t>(constant.value_bits);
        [values setConstantValue:&value type:MTLDataTypeUInt atIndex:constant.index];
        return;
    }
    case MetalFunctionConstantType::Float32: {
        uint32_t bits = static_cast<uint32_t>(constant.value_bits);
        float value = std::bit_cast<float>(bits);
        [values setConstantValue:&value type:MTLDataTypeFloat atIndex:constant.index];
        return;
    }
    }
}

MetalPipelineDigest generation_digest(const NormalizedRecipes& normalized,
                                      const std::vector<MetalPipelineSlotInfo>& slots,
                                      MTLSize device_max_threads,
                                      bool modern_math) {
    std::vector<uint8_t> bytes;
    const std::string_view domain = "laplace.metal-pipeline-transaction.v2";
    append_u32(bytes, static_cast<uint32_t>(domain.size()));
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    append_u32(bytes, static_cast<uint32_t>(normalized.libraries.size()));
    for (size_t index = 0; index < normalized.libraries.size(); ++index) {
        const MetalPipelineLibrarySource& library = normalized.libraries[index];
        append_u32(bytes, static_cast<uint32_t>(index));
        append_digest(bytes, library.source_digest);
        append_digest(bytes,
                      compile_contract_digest(library.compile_contract,
                                              modern_math));
    }
    append_u64(bytes, device_max_threads.width);
    append_u64(bytes, device_max_threads.height);
    append_u64(bytes, device_max_threads.depth);
    append_u32(bytes, static_cast<uint32_t>(slots.size()));
    for (size_t index = 0; index != slots.size(); ++index) {
        const MetalPipelineSlotInfo& slot = slots[index];
        append_u32(bytes, static_cast<uint32_t>(index));
        append_digest(bytes, slot.requirement_digest);
        append_u32(bytes, slot.thread_execution_width);
        append_u32(bytes, slot.max_total_threads_per_threadgroup);
        append_u32(bytes, slot.dispatch_minimum);
        append_u32(bytes, slot.dispatch_maximum);
        if (!serialize_recipe(bytes, normalized.recipes[index])) return {};
    }
    return digest_bytes(bytes);
}

struct BuildCandidate {
    id<MTLDevice> device = nil;
    std::vector<id<MTLLibrary>> libraries;
    std::vector<id<MTLFunction>> functions;
    std::vector<id<MTLComputePipelineState>> pipelines;
    MetalPipelineTransactionAudit audit;

    ~BuildCandidate() {
        for (id<MTLComputePipelineState> pipeline : pipelines) {
            [pipeline release];
#if defined(LAPLACE_TESTING)
            g_live_pipeline_owners.fetch_sub(1, std::memory_order_relaxed);
#endif
        }
        for (id<MTLFunction> function : functions) [function release];
        for (id<MTLLibrary> library : libraries) [library release];
        [device release];
    }
};

MetalPipelineTransactionResult allocation_failure() noexcept {
    MetalPipelineTransactionFailure failure;
    failure.code = MetalPipelineTransactionError::AllocationFailed;
    return failure;
}

template <typename Builder>
MetalPipelineTransactionResult guarded_build(Builder&& builder) noexcept {
    try {
        return builder();
    } catch (const std::bad_alloc&) {
        return allocation_failure();
    }
}

} // namespace

struct MetalPipelineStorage {
    std::vector<MetalPipelineSlotInfo> slots;
    std::vector<MetalPipelineRecipe> recipes;
    std::vector<id<MTLLibrary>> libraries;
    std::vector<id<MTLFunction>> functions;
    std::vector<id<MTLComputePipelineState>> pipelines;
    id<MTLDevice> device = nil;
    MetalPipelineDigest generation_digest{};
    MetalPipelineTransactionAudit audit;

    ~MetalPipelineStorage() {
        for (id<MTLComputePipelineState> pipeline : pipelines) {
            [pipeline release];
#if defined(LAPLACE_TESTING)
            g_live_pipeline_owners.fetch_sub(1, std::memory_order_relaxed);
#endif
        }
        for (id<MTLFunction> function : functions) [function release];
        for (id<MTLLibrary> library : libraries) [library release];
        [device release];
    }
};

MetalPipelineTransactionResult build_impl(
    std::span<const MetalPipelineRecipe> input,
    std::span<const MetalPipelineLibrarySource> library_sources,
    uint8_t injected_value,
    bool allow_exact_duplicates) {
    @autoreleasepool {
        const InjectedFailure injected = static_cast<InjectedFailure>(injected_value);
        auto normalized_result = normalize_recipes(input, library_sources,
                                                   allow_exact_duplicates);
        if (auto* failure = std::get_if<MetalPipelineTransactionFailure>(
                &normalized_result))
            return *failure;
        NormalizedRecipes normalized =
            std::move(std::get<NormalizedRecipes>(normalized_result));

        if (injected == InjectedFailure::Allocation)
            throw std::bad_alloc();

        BuildCandidate candidate;
        candidate.device = MTLCreateSystemDefaultDevice();
        if (!candidate.device)
            return MetalPipelineTransactionFailure{
                MetalPipelineTransactionError::NoDevice,
                "Metal has no system default device", candidate.audit};
        candidate.audit.devices_created = 1;

        candidate.libraries.reserve(normalized.libraries.size());
        bool modern_math = false;
        if (@available(macOS 15.0, *)) modern_math = true;
        for (size_t index = 0; index < normalized.libraries.size(); ++index) {
            if (injected == InjectedFailure::Library)
                return MetalPipelineTransactionFailure{
                    MetalPipelineTransactionError::LibraryCompileFailed,
                    "injected library failure", candidate.audit};
            const MetalPipelineLibrarySource& library_source =
                normalized.libraries[index];
            NSString* source = [[NSString alloc]
                initWithBytes:library_source.bytes.data()
                       length:library_source.bytes.size()
                     encoding:NSUTF8StringEncoding];
            if (!source)
                return MetalPipelineTransactionFailure{
                    MetalPipelineTransactionError::LibraryCompileFailed,
                    "library source is not valid UTF-8", candidate.audit};
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            if (!options) {
                [source release];
                return MetalPipelineTransactionFailure{
                    MetalPipelineTransactionError::AllocationFailed,
                    "Metal compile options allocation failed", candidate.audit};
            }
            bool applied_modern_math = false;
            const bool options_valid = apply_compile_contract(
                options, library_source.compile_contract, &applied_modern_math);
            if (!options_valid || applied_modern_math != modern_math) {
                [options release];
                [source release];
                return MetalPipelineTransactionFailure{
                    MetalPipelineTransactionError::LibraryCompileFailed,
                    "Metal compile contract is unavailable on this OS", candidate.audit};
            }
            NSError* error = nil;
            id<MTLLibrary> library = [candidate.device
                newLibraryWithSource:source options:options error:&error];
            [options release];
            [source release];
            if (!library)
                return MetalPipelineTransactionFailure{
                    MetalPipelineTransactionError::LibraryCompileFailed,
                    "Metal library compilation failed", candidate.audit};
            candidate.libraries.push_back(library);
            candidate.audit.libraries_created++;
        }

        const MTLSize device_max_threads = candidate.device.maxThreadsPerThreadgroup;
        if (device_max_threads.width == 0 || device_max_threads.height == 0 ||
            device_max_threads.depth == 0)
            return MetalPipelineTransactionFailure{
                MetalPipelineTransactionError::PipelineLimitsInvalid,
                "Metal device reports zero threadgroup limits", candidate.audit};

        std::vector<MetalPipelineSlotInfo> slots;
        slots.reserve(normalized.recipes.size());
        candidate.functions.reserve(normalized.recipes.size());
        candidate.pipelines.reserve(normalized.recipes.size());
        NSError* error = nil;
        for (const MetalPipelineRecipe& recipe : normalized.recipes) {
            if (injected == InjectedFailure::PipelineAfterFirst &&
                !candidate.pipelines.empty())
                return MetalPipelineTransactionFailure{
                    MetalPipelineTransactionError::PipelineBuildFailed,
                    "injected pipeline failure after a retained pipeline",
                    candidate.audit};
            if (injected == InjectedFailure::Function)
                return MetalPipelineTransactionFailure{
                    MetalPipelineTransactionError::FunctionLookupFailed,
                    "injected function failure", candidate.audit};

            size_t library_index = normalized.libraries.size();
            for (size_t index = 0; index < normalized.libraries.size(); ++index) {
                if (normalized.libraries[index].source_digest ==
                    recipe.library_source_digest) {
                    library_index = index;
                    break;
                }
            }
            if (library_index == normalized.libraries.size())
                return MetalPipelineTransactionFailure{
                    MetalPipelineTransactionError::InvalidRequest,
                    "recipe library source digest is not bound", candidate.audit};
            id<MTLLibrary> library = candidate.libraries[library_index];
            id<MTLFunction> function = nil;
            if (recipe.function_constants.empty()) {
                function = [library
                    newFunctionWithName:[NSString stringWithUTF8String:
                                                  recipe.function_name.c_str()]];
            } else {
                MTLFunctionConstantValues* values =
                    [[MTLFunctionConstantValues alloc] init];
                for (const MetalFunctionConstant& constant : recipe.function_constants)
                    set_function_constant(values, constant);
                function = [library
                    newFunctionWithName:[NSString stringWithUTF8String:
                                                  recipe.function_name.c_str()]
                         constantValues:values
                                  error:&error];
                [values release];
            }
            if (!function)
                return MetalPipelineTransactionFailure{
                    MetalPipelineTransactionError::FunctionLookupFailed,
                    "Metal function lookup or specialization failed", candidate.audit};
            candidate.audit.functions_created++;
            candidate.functions.push_back(function);

            id<MTLComputePipelineState> pipeline =
                [candidate.device newComputePipelineStateWithFunction:function
                                                                   error:&error];
            if (!pipeline)
                return MetalPipelineTransactionFailure{
                    MetalPipelineTransactionError::PipelineBuildFailed,
                    "Metal compute pipeline compilation failed", candidate.audit};
            candidate.audit.pipelines_created++;
            if (injected == InjectedFailure::Pipeline) {
                [pipeline release];
                return MetalPipelineTransactionFailure{
                    MetalPipelineTransactionError::PipelineBuildFailed,
                    "injected pipeline failure", candidate.audit};
            }
            if (pipeline.device != candidate.device) {
                [pipeline release];
                return MetalPipelineTransactionFailure{
                    MetalPipelineTransactionError::DeviceMismatch,
                    "pipeline was created by a different Metal device", candidate.audit};
            }

            bool limits_ok = true;
            const uint32_t thread_width = checked_u32(
                pipeline.threadExecutionWidth, &limits_ok);
            const uint32_t pipeline_max = checked_u32(
                pipeline.maxTotalThreadsPerThreadgroup, &limits_ok);
            const uint32_t required = dispatch_minimum(
                recipe, thread_width, &limits_ok);
            const uint32_t device_max = checked_u32(
                device_max_threads.width, &limits_ok);
            const uint32_t allowed = dispatch_maximum(
                recipe, pipeline_max, device_max);
            if (!limits_ok || thread_width == 0 || pipeline_max == 0 ||
                thread_width < recipe.dispatch.min_thread_execution_width ||
                (recipe.dispatch.max_thread_execution_width != 0 &&
                 thread_width > recipe.dispatch.max_thread_execution_width) ||
                required > allowed) {
                [pipeline release];
                return MetalPipelineTransactionFailure{
                    MetalPipelineTransactionError::PipelineLimitsInvalid,
                    "pipeline limits do not satisfy recipe constraints", candidate.audit};
            }
            candidate.pipelines.push_back(pipeline);
#if defined(LAPLACE_TESTING)
            g_live_pipeline_owners.fetch_add(1, std::memory_order_relaxed);
#endif
            slots.push_back({static_cast<uint32_t>(slots.size()),
                             recipe.normalized_requirement_digest, thread_width,
                             pipeline_max, required, allowed,
                             static_cast<const void*>(pipeline)});
        }

        auto storage = std::make_unique<MetalPipelineStorage>();
        storage->slots = std::move(slots);
        storage->libraries = std::move(candidate.libraries);
        storage->functions = std::move(candidate.functions);
        storage->pipelines = std::move(candidate.pipelines);
        candidate.libraries.clear();
        candidate.functions.clear();
        candidate.pipelines.clear();
        storage->device = candidate.device;
        candidate.device = nil;
        storage->generation_digest = generation_digest(
            normalized, storage->slots, device_max_threads, modern_math);
        if (zero_digest(storage->generation_digest))
            return MetalPipelineTransactionFailure{
                MetalPipelineTransactionError::AllocationFailed,
                "transaction generation digest is zero", candidate.audit};
        storage->recipes = std::move(normalized.recipes);
        candidate.audit.transaction_published = true;
        storage->audit = candidate.audit;
        return MetalPipelineTransaction(std::move(storage));
    }
}

MetalPipelineTransactionResult build_impl(
    std::span<const MetalPipelineRecipe> input,
    std::span<const uint8_t> library_source,
    const MetalPipelineCompileContract& compile_contract,
    uint8_t injected_value) {
    const MetalPipelineLibrarySource source = {
        library_source, digest_bytes(library_source), compile_contract};
    return build_impl(input, std::span<const MetalPipelineLibrarySource>(&source, 1),
                      injected_value, true);
}

MetalPipelineTransactionResult build_impl(
    std::span<const MetalPipelineRecipe> input,
    std::span<const uint8_t> library_source,
    uint8_t injected_value) {
    return build_impl(input, library_source, MetalPipelineCompileContract{},
                      injected_value);
}

MetalPipelineTransaction::MetalPipelineTransaction(
    std::unique_ptr<MetalPipelineStorage> storage)
    : storage_(std::move(storage)) {}

MetalPipelineTransaction::MetalPipelineTransaction() noexcept = default;

MetalPipelineTransaction::~MetalPipelineTransaction() = default;

MetalPipelineTransaction::MetalPipelineTransaction(
    MetalPipelineTransaction&&) noexcept = default;

MetalPipelineTransaction& MetalPipelineTransaction::operator=(
    MetalPipelineTransaction&&) noexcept = default;

bool MetalPipelineTransaction::valid() const noexcept {
    return storage_ != nullptr && !storage_->slots.empty() &&
           !storage_->libraries.empty() &&
           storage_->recipes.size() == storage_->slots.size() &&
           storage_->slots.size() == storage_->pipelines.size() &&
           storage_->pipelines.size() == storage_->functions.size() &&
           !zero_digest(storage_->generation_digest);
}

uint32_t MetalPipelineTransaction::slot_count() const noexcept {
    return storage_ && storage_->slots.size() <= std::numeric_limits<uint32_t>::max()
        ? static_cast<uint32_t>(storage_->slots.size())
        : 0;
}

const MetalPipelineSlotInfo* MetalPipelineTransaction::slot(
    uint32_t index) const noexcept {
    if (!storage_ || index >= storage_->slots.size()) return nullptr;
    return &storage_->slots[index];
}

const MetalPipelineSlotInfo* MetalPipelineTransaction::find(
    MetalPipelineDigest requirement) const noexcept {
    if (!storage_) return nullptr;
    const auto found = std::lower_bound(
        storage_->slots.begin(), storage_->slots.end(), requirement,
        [](const MetalPipelineSlotInfo& slot, const MetalPipelineDigest& digest) {
            return std::lexicographical_compare(
                slot.requirement_digest.begin(), slot.requirement_digest.end(),
                digest.begin(), digest.end());
        });
    if (found == storage_->slots.end() || found->requirement_digest != requirement)
        return nullptr;
    return &*found;
}

MetalPipelineDigest MetalPipelineTransaction::generation_digest() const noexcept {
    return storage_ ? storage_->generation_digest : MetalPipelineDigest{};
}

const MetalPipelineTransactionAudit& MetalPipelineTransaction::audit() const noexcept {
    static const MetalPipelineTransactionAudit empty{};
    return storage_ ? storage_->audit : empty;
}

MetalPipelineLease MetalPipelineTransaction::take_lease() && noexcept {
    return MetalPipelineLease(std::move(storage_));
}

MetalPipelineLease::MetalPipelineLease(
    std::unique_ptr<MetalPipelineStorage> storage)
    : storage_(std::move(storage)) {}

MetalPipelineLease::~MetalPipelineLease() = default;

MetalPipelineLease::MetalPipelineLease(MetalPipelineLease&&) noexcept = default;

MetalPipelineLease& MetalPipelineLease::operator=(MetalPipelineLease&&) noexcept = default;

bool MetalPipelineLease::valid() const noexcept {
    return storage_ != nullptr && !storage_->slots.empty() &&
           !storage_->libraries.empty() &&
           storage_->recipes.size() == storage_->slots.size() &&
           storage_->slots.size() == storage_->pipelines.size() &&
           storage_->pipelines.size() == storage_->functions.size() &&
           !zero_digest(storage_->generation_digest);
}

uint32_t MetalPipelineLease::slot_count() const noexcept {
    return storage_ && storage_->slots.size() <= std::numeric_limits<uint32_t>::max()
        ? static_cast<uint32_t>(storage_->slots.size())
        : 0;
}

const MetalPipelineSlotInfo* MetalPipelineLease::slot(uint32_t index) const noexcept {
    if (!storage_ || index >= storage_->slots.size()) return nullptr;
    return &storage_->slots[index];
}

const MetalPipelineSlotInfo* MetalPipelineLease::find(
    MetalPipelineDigest requirement) const noexcept {
    if (!storage_) return nullptr;
    const auto found = std::lower_bound(
        storage_->slots.begin(), storage_->slots.end(), requirement,
        [](const MetalPipelineSlotInfo& slot, const MetalPipelineDigest& digest) {
            return std::lexicographical_compare(
                slot.requirement_digest.begin(), slot.requirement_digest.end(),
                digest.begin(), digest.end());
        });
    if (found == storage_->slots.end() || found->requirement_digest != requirement)
        return nullptr;
    return &*found;
}

MetalPipelineDigest MetalPipelineLease::generation_digest() const noexcept {
    return storage_ ? storage_->generation_digest : MetalPipelineDigest{};
}

const MetalPipelineTransactionAudit& MetalPipelineLease::audit() const noexcept {
    static const MetalPipelineTransactionAudit empty{};
    return storage_ ? storage_->audit : empty;
}

const void* MetalPipelineLease::device_token() const noexcept {
    return storage_ ? static_cast<const void*>(storage_->device) : nullptr;
}

const void* MetalPipelineLease::pipeline_token(uint32_t index) const noexcept {
    if (!storage_ || index >= storage_->pipelines.size()) return nullptr;
    return static_cast<const void*>(storage_->pipelines[index]);
}

bool MetalPipelineLease::matches_device_token(const void* device) const noexcept {
    return device != nullptr && device_token() == device;
}

bool MetalPipelineLease::resolve_recipe(const MetalPipelineRecipe& recipe,
                                        uint32_t* slot,
                                        const void** pipeline) const noexcept {
    if (!storage_ || !slot || !pipeline ||
        storage_->recipes.size() != storage_->slots.size() ||
        storage_->slots.size() != storage_->pipelines.size())
        return false;
    for (size_t index = 0; index < storage_->recipes.size(); ++index) {
        if (!recipe_equal_canonical(storage_->recipes[index], recipe)) continue;
        const void* resolved_pipeline =
            static_cast<const void*>(storage_->pipelines[index]);
        if (!resolved_pipeline || index > std::numeric_limits<uint32_t>::max())
            return false;
        *slot = static_cast<uint32_t>(index);
        *pipeline = resolved_pipeline;
        return true;
    }
    return false;
}

MetalPipelineDigest metal_pipeline_digest(std::span<const uint8_t> bytes) noexcept {
    return digest_bytes(bytes);
}

MetalPipelineTransactionResult build_metal_pipeline_transaction(
    std::span<const MetalPipelineRecipe> recipes,
    std::span<const uint8_t> library_source) {
    return guarded_build([&] {
        return build_impl(recipes, library_source,
                          static_cast<uint8_t>(InjectedFailure::None));
    });
}

MetalPipelineTransactionResult build_metal_pipeline_transaction(
    std::span<const MetalPipelineRecipe> recipes,
    std::span<const uint8_t> library_source,
    const MetalPipelineCompileContract& compile_contract) {
    return guarded_build([&] {
        return build_impl(recipes, library_source, compile_contract,
                          static_cast<uint8_t>(InjectedFailure::None));
    });
}

MetalPipelineTransactionResult build_metal_pipeline_transaction(
    std::span<const MetalPipelineRecipe> recipes,
    std::span<const MetalPipelineLibrarySource> library_sources) {
    return guarded_build([&] {
        return build_impl(recipes, library_sources,
                          static_cast<uint8_t>(InjectedFailure::None), false);
    });
}

#if defined(LAPLACE_TESTING)
MetalPipelineTransactionResult build_metal_pipeline_transaction_for_testing(
    std::span<const MetalPipelineRecipe> recipes,
    std::span<const uint8_t> library_source,
    MetalPipelineInjectedFailure failure) {
    return guarded_build([&] {
        return build_impl(recipes, library_source,
                          static_cast<uint8_t>(failure));
    });
}

MetalPipelineTransactionResult build_metal_pipeline_transaction_for_testing(
    std::span<const MetalPipelineRecipe> recipes,
    std::span<const uint8_t> library_source,
    const MetalPipelineCompileContract& compile_contract,
    MetalPipelineInjectedFailure failure) {
    return guarded_build([&] {
        return build_impl(recipes, library_source, compile_contract,
                          static_cast<uint8_t>(failure));
    });
}

MetalPipelineTransactionResult build_metal_pipeline_transaction_for_testing(
    std::span<const MetalPipelineRecipe> recipes,
    std::span<const MetalPipelineLibrarySource> library_sources,
    MetalPipelineInjectedFailure failure) {
    return guarded_build([&] {
        return build_impl(recipes, library_sources,
                          static_cast<uint8_t>(failure), false);
    });
}

uint32_t metal_pipeline_live_pipeline_owners_for_testing() noexcept {
    return g_live_pipeline_owners.load(std::memory_order_relaxed);
}
#endif

} // namespace Laplace
