#include "metal_capability_snapshot.h"

#include <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "gemv_legacy.inc"

namespace Laplace {

struct MetalCapabilitySnapshot::Impl {
    CodecProgramIdentity identity;
    MetalCapabilityDigest capability_digest{};
    uint32_t thread_execution_width = 0;
    uint32_t max_total_threads_per_threadgroup = 0;
    uint32_t dispatch_minimum = 0;
    bool pipeline_constructed = false;
    // Retain the exact device used to create the pipeline. This is deliberately
    // private so a later execution/session binder can require device identity
    // without making Metal objects part of the planner-facing API.
    id<MTLDevice> device = nil;
    id<MTLComputePipelineState> pipeline = nil;
    MetalCapabilityProbeAudit audit;

    ~Impl() {
        [pipeline release];
        [device release];
    }
};

namespace {

// Private application mapping. It derives the exact identity from the
// application registry's canonical contract and never uses registry position
// or device marketing data. These tags are private contract fields.
constexpr uint16_t kDecodeContractTag = 1;
constexpr uint16_t kOptimizedContractTag = 2;
constexpr int kDecodeConstantValue = 1;
constexpr NSUInteger kDecodeConstantIndex = 0;
constexpr const char* kDecodeFunctionName = "gemv";
constexpr const char* kOptimizedFunctionName = "gemv_q4k";

constexpr std::string_view kDigestDomain =
    "laplace.metal-capability-snapshot.v1";
constexpr std::string_view kOperatorAbi =
    "laplace.decode-gemv-operator.v1;buffers=u8,f32,f32,i32,i32,u64,i32;"
    "phase=decode;output=binary32;";
constexpr std::string_view kDecodePhase = "decode";
constexpr std::string_view kNumericalPolicy =
    "ieee754-binary32-accumulate-v1;qualification=none";

enum class Recipe : uint8_t { Decode = 1, Optimized = 2 };

enum class InjectedFailure : uint8_t {
    None = 0,
    Library = 1,
    Function = 2,
    Pipeline = 3,
};

struct RecipeSelection {
    Recipe recipe = Recipe::Decode;
    MetalCapabilityProbeError error = MetalCapabilityProbeError::RegistryInvalid;
    bool ok = false;
};

void append_u16(std::vector<uint8_t>& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8u));
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void append_bytes(std::vector<uint8_t>& bytes, std::span<const uint8_t> input) {
    bytes.insert(bytes.end(), input.begin(), input.end());
}

void append_string(std::vector<uint8_t>& bytes, std::string_view value) {
    append_u32(bytes, static_cast<uint32_t>(value.size()));
    append_bytes(bytes, std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(value.data()), value.size()));
}

MetalCapabilityDigest digest_string(std::string_view value) {
    return codec_program_digest(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(value.data()), value.size()));
}

std::optional<Recipe> recipe_from_contract(const CodecProgram& program) {
    const std::span<const uint8_t> bytes = program.canonical_bytes();
    constexpr size_t algorithm_offset = 8 + 2 + 2;
    if (bytes.size() < algorithm_offset + 2) return std::nullopt;
    const uint16_t tag = static_cast<uint16_t>(bytes[algorithm_offset]) |
                         (static_cast<uint16_t>(bytes[algorithm_offset + 1]) << 8u);
    if (tag == kDecodeContractTag) return Recipe::Decode;
    if (tag == kOptimizedContractTag) return Recipe::Optimized;
    return std::nullopt;
}

bool exact_program(const CodecProgramRegistry& registry,
                  const CodecProgram& candidate,
                  const CodecProgramRegistry& expected) {
    const CodecProgram* expected_program = expected.resolve(candidate.identity());
    if (!expected_program || candidate.canonical_bytes().size() !=
                                 expected_program->canonical_bytes().size() ||
        !std::equal(candidate.canonical_bytes().begin(), candidate.canonical_bytes().end(),
                    expected_program->canonical_bytes().begin()))
        return false;
    return registry.resolve(candidate.identity()) == &candidate;
}

RecipeSelection select_recipe(const CodecProgramRegistry& registry,
                              const CodecProgramIdentity& requested) {
    const CodecProgramRegistry expected = make_application_codec_registry();
    // Resolve only the requested identity. Additional future registry entries
    // are permitted and do not change the exact application mapping.
    const CodecProgram* expected_program = nullptr;
    for (const CodecProgram& program : expected.programs()) {
        if (program.identity() != requested) continue;
        if (expected_program != nullptr) return {};
        expected_program = &program;
    }
    if (expected_program == nullptr) {
        RecipeSelection result;
        result.error = MetalCapabilityProbeError::UnknownProgram;
        return result;
    }
    const std::optional<Recipe> requested_recipe = recipe_from_contract(*expected_program);
    if (!requested_recipe) return {};

    const CodecProgram* registered_program = nullptr;
    for (const CodecProgram& program : registry.programs()) {
        if (program.identity() != requested) continue;
        if (registered_program != nullptr) {
            RecipeSelection result;
            result.error = MetalCapabilityProbeError::RegistryInvalid;
            return result;
        }
        registered_program = &program;
    }
    if (registered_program == nullptr ||
        !exact_program(registry, *registered_program, expected)) {
        RecipeSelection result;
        result.error = MetalCapabilityProbeError::UnknownProgram;
        return result;
    }
    RecipeSelection result;
    result.recipe = *requested_recipe;
    result.error = MetalCapabilityProbeError::NoDevice;
    result.ok = true;
    return result;
}

struct ProbeCandidate {
    id<MTLDevice> device = nil;
    id<MTLLibrary> library = nil;
    id<MTLFunction> function = nil;
    id<MTLComputePipelineState> pipeline = nil;
    MetalCapabilityProbeAudit audit;

    ~ProbeCandidate() {
        [pipeline release];
        [function release];
        [library release];
        [device release];
    }
};

bool checked_u32(NSUInteger value, uint32_t* result) {
    if (value > std::numeric_limits<uint32_t>::max()) return false;
    *result = static_cast<uint32_t>(value);
    return true;
}

MetalCapabilityDigest make_recipe_digest(Recipe recipe, uint32_t dispatch_minimum) {
    std::vector<uint8_t> bytes;
    append_string(bytes, "laplace.private-pipeline-recipe.v1");
    append_string(bytes, "library-options=nil");
    if (recipe == Recipe::Decode) {
        append_string(bytes, kDecodeFunctionName);
        append_u32(bytes, static_cast<uint32_t>(kDecodeConstantIndex));
        append_u32(bytes, static_cast<uint32_t>(kDecodeConstantValue));
        append_u32(bytes, static_cast<uint32_t>(MTLDataTypeInt));
        append_string(bytes, "constant-count=1");
        append_string(bytes, "simdgroups=1");
    } else {
        append_string(bytes, kOptimizedFunctionName);
        append_string(bytes, "constant-count=0");
        append_u32(bytes, 0);
        append_u32(bytes, 0);
        append_string(bytes, "simdgroups=2");
    }
    append_u32(bytes, dispatch_minimum);
    return codec_program_digest(std::span<const uint8_t>(bytes));
}

MetalCapabilityDigest make_capability_digest(
    const CodecProgramIdentity& identity,
    const MetalCapabilityDigest& operator_abi_digest,
    const MetalCapabilityDigest& source_digest,
    const MetalCapabilityDigest& recipe_digest,
    uint32_t thread_width, uint32_t pipeline_max,
    MTLSize device_max_threads, uint32_t family_bits, uint32_t api_facts) {
    std::vector<uint8_t> bytes;
    bytes.reserve(224);
    append_string(bytes, kDigestDomain);
    append_u16(bytes, identity.abi_version);
    append_bytes(bytes, std::span<const uint8_t>(identity.contract_digest));
    append_bytes(bytes, std::span<const uint8_t>(operator_abi_digest));
    append_string(bytes, kDecodePhase);
    append_string(bytes, kNumericalPolicy);
    append_bytes(bytes, std::span<const uint8_t>(source_digest));
    append_bytes(bytes, std::span<const uint8_t>(recipe_digest));
    append_u32(bytes, thread_width);
    append_u32(bytes, pipeline_max);
    append_u64(bytes, static_cast<uint64_t>(device_max_threads.width));
    append_u64(bytes, static_cast<uint64_t>(device_max_threads.height));
    append_u64(bytes, static_cast<uint64_t>(device_max_threads.depth));
    append_u32(bytes, api_facts);
    append_u32(bytes, family_bits);
    return codec_program_digest(std::span<const uint8_t>(bytes));
}

uint32_t queried_family_bits(id<MTLDevice> device) {
    uint32_t bits = 0;
    if (@available(macOS 10.15, *)) {
        constexpr MTLGPUFamily families[] = {
            MTLGPUFamilyApple1, MTLGPUFamilyApple2, MTLGPUFamilyApple3,
            MTLGPUFamilyApple4, MTLGPUFamilyApple5, MTLGPUFamilyApple6,
            MTLGPUFamilyApple7, MTLGPUFamilyApple8, MTLGPUFamilyApple9,
            MTLGPUFamilyApple10,
        };
        for (uint32_t index = 0; index != sizeof(families) / sizeof(families[0]); ++index)
            if ([device supportsFamily:families[index]]) bits |= 1u << index;
    }
    return bits;
}

} // namespace

MetalCapabilityProbeResult metal_capability_detail::probe(
    const CodecProgramRegistry& registry, const CodecProgramIdentity& identity,
    uint8_t injected_failure) {
    @autoreleasepool {
    const InjectedFailure injected = static_cast<InjectedFailure>(injected_failure);
    const RecipeSelection selection = select_recipe(registry, identity);
    if (!selection.ok) return MetalCapabilityProbeFailure{selection.error, {}};

    ProbeCandidate candidate;
    candidate.device = MTLCreateSystemDefaultDevice();
    if (!candidate.device)
        return MetalCapabilityProbeFailure{
            MetalCapabilityProbeError::NoDevice, candidate.audit};
    candidate.audit.devices_created = 1;

    if (injected == InjectedFailure::Library)
        return MetalCapabilityProbeFailure{
            MetalCapabilityProbeError::LibraryCompileFailed, candidate.audit};

    NSString* source = [NSString stringWithUTF8String:src_gemv];
    NSError* error = nil;
    candidate.library = [candidate.device newLibraryWithSource:source
                                                        options:nil
                                                          error:&error];
    if (!candidate.library)
        return MetalCapabilityProbeFailure{
            MetalCapabilityProbeError::LibraryCompileFailed, candidate.audit};
    candidate.audit.libraries_created = 1;

    if (injected == InjectedFailure::Function)
        return MetalCapabilityProbeFailure{
            MetalCapabilityProbeError::FunctionLookupFailed, candidate.audit};

    if (selection.recipe == Recipe::Decode) {
        MTLFunctionConstantValues* values = [[MTLFunctionConstantValues alloc] init];
        int decode_constant = kDecodeConstantValue;
        [values setConstantValue:&decode_constant
                            type:MTLDataTypeInt
                         atIndex:kDecodeConstantIndex];
        candidate.function = [candidate.library
            newFunctionWithName:[NSString stringWithUTF8String:kDecodeFunctionName]
                  constantValues:values
                           error:&error];
        [values release];
    } else {
        candidate.function = [candidate.library
            newFunctionWithName:[NSString stringWithUTF8String:kOptimizedFunctionName]];
    }
    if (!candidate.function)
        return MetalCapabilityProbeFailure{
            MetalCapabilityProbeError::FunctionLookupFailed, candidate.audit};
    candidate.audit.functions_created = 1;

    candidate.pipeline =
        [candidate.device newComputePipelineStateWithFunction:candidate.function
                                                         error:&error];
    if (!candidate.pipeline)
        return MetalCapabilityProbeFailure{
            MetalCapabilityProbeError::PipelineBuildFailed, candidate.audit};
    candidate.audit.pipelines_created = 1;
    if (injected == InjectedFailure::Pipeline)
        return MetalCapabilityProbeFailure{
            MetalCapabilityProbeError::PipelineBuildFailed, candidate.audit};

    uint32_t thread_width = 0;
    uint32_t pipeline_max = 0;
    if (!checked_u32(candidate.pipeline.threadExecutionWidth, &thread_width) ||
        !checked_u32(candidate.pipeline.maxTotalThreadsPerThreadgroup, &pipeline_max) ||
        thread_width == 0 || pipeline_max == 0 ||
        (selection.recipe == Recipe::Optimized &&
         thread_width > std::numeric_limits<uint32_t>::max() / 2u))
        return MetalCapabilityProbeFailure{
            MetalCapabilityProbeError::PipelineLimitsInvalid, candidate.audit};
    const uint32_t dispatch_minimum = selection.recipe == Recipe::Decode
        ? thread_width
        : thread_width * 2u;
    const MTLSize device_max_threads = candidate.device.maxThreadsPerThreadgroup;
    if (device_max_threads.width == 0 || device_max_threads.height == 0 ||
        device_max_threads.depth == 0 || dispatch_minimum > pipeline_max ||
        static_cast<uint64_t>(dispatch_minimum) >
            static_cast<uint64_t>(device_max_threads.width))
        return MetalCapabilityProbeFailure{
            MetalCapabilityProbeError::PipelineLimitsInvalid, candidate.audit};

    const size_t source_size = std::strlen(src_gemv);
    const MetalCapabilityDigest source_digest = codec_program_digest(
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(src_gemv), source_size));
    const MetalCapabilityDigest operator_abi_digest = digest_string(kOperatorAbi);
    const MetalCapabilityDigest recipe_digest =
        make_recipe_digest(selection.recipe, dispatch_minimum);
    constexpr uint32_t kSourceFunctionApi = 1u << 0;
    constexpr uint32_t kSpecializedFunctionApi = 1u << 1;
    constexpr uint32_t kPipelineApi = 1u << 2;
    constexpr uint32_t kFamilyQueryApi = 1u << 3;
    uint32_t api_facts = kSourceFunctionApi | kSpecializedFunctionApi | kPipelineApi;
    if (@available(macOS 10.15, *)) api_facts |= kFamilyQueryApi;
    const uint32_t family_bits = queried_family_bits(candidate.device);

    auto impl = std::make_unique<MetalCapabilitySnapshot::Impl>();
    impl->identity = identity;
    impl->capability_digest = make_capability_digest(
        identity, operator_abi_digest, source_digest, recipe_digest,
        thread_width, pipeline_max, device_max_threads, family_bits, api_facts);
    impl->thread_execution_width = thread_width;
    impl->max_total_threads_per_threadgroup = pipeline_max;
    impl->dispatch_minimum = dispatch_minimum;
    impl->pipeline_constructed = true;
    impl->device = candidate.device;
    candidate.device = nil;
    impl->pipeline = candidate.pipeline;
    candidate.pipeline = nil;
    impl->audit = candidate.audit;
    impl->audit.snapshot_published = true;
    return MetalCapabilitySnapshot(std::move(impl));
    }
}

MetalCapabilitySnapshot::MetalCapabilitySnapshot(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MetalCapabilitySnapshot::~MetalCapabilitySnapshot() = default;

MetalCapabilitySnapshot::MetalCapabilitySnapshot(MetalCapabilitySnapshot&&) noexcept = default;

MetalCapabilitySnapshot& MetalCapabilitySnapshot::operator=(MetalCapabilitySnapshot&&) noexcept =
    default;

CodecProgramIdentity MetalCapabilitySnapshot::program_identity() const noexcept {
    return impl_->identity;
}

MetalCapabilityDigest MetalCapabilitySnapshot::capability_digest() const noexcept {
    return impl_->capability_digest;
}

uint32_t MetalCapabilitySnapshot::thread_execution_width() const noexcept {
    return impl_->thread_execution_width;
}

uint32_t MetalCapabilitySnapshot::max_total_threads_per_threadgroup() const noexcept {
    return impl_->max_total_threads_per_threadgroup;
}

uint32_t MetalCapabilitySnapshot::dispatch_minimum() const noexcept {
    return impl_->dispatch_minimum;
}

bool MetalCapabilitySnapshot::pipeline_constructed() const noexcept {
    return impl_->pipeline_constructed;
}

const MetalCapabilityProbeAudit& MetalCapabilitySnapshot::audit() const noexcept {
    return impl_->audit;
}

MetalCapabilityProbeResult probe_metal_capability(
    const CodecProgramRegistry& registry, const CodecProgramIdentity& identity) {
    return metal_capability_detail::probe(
        registry, identity, 0);
}

#if defined(LAPLACE_TESTING)
MetalCapabilityProbeResult probe_metal_capability_for_testing(
    const CodecProgramRegistry& registry, const CodecProgramIdentity& identity,
    MetalCapabilityInjectedFailure injected) {
    return metal_capability_detail::probe(registry, identity,
                                          static_cast<uint8_t>(injected));
}
#endif

} // namespace Laplace
