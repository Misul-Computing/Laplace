#include <cstdio>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <variant>

#include "codec_program.h"
#include "metal_capability_snapshot.h"
#include "test_util.h"

using namespace Laplace;

namespace {

const MetalCapabilityProbeFailure* failure_of(const MetalCapabilityProbeResult& result) {
    return std::get_if<MetalCapabilityProbeFailure>(&result);
}

const MetalCapabilitySnapshot* snapshot_of(const MetalCapabilityProbeResult& result) {
    return std::get_if<MetalCapabilitySnapshot>(&result);
}

void check_no_forbidden_resources(const MetalCapabilityProbeAudit& audit) {
    CHECK(audit.command_queues_created == 0);
    CHECK(audit.command_buffers_created == 0);
    CHECK(audit.buffers_created == 0);
    CHECK(audit.token_sessions_created == 0);
    CHECK(audit.global_cache_mutations == 0);
    CHECK(audit.environment_mutations == 0);
}

void test_source_contract() {
#if defined(LAPLACE_SOURCE_DIR)
    const std::string source_path =
        std::string(LAPLACE_SOURCE_DIR) + "/src/metal_capability_snapshot.mm";
    std::ifstream source_file(source_path);
    CHECK(source_file.good());
    const std::string source((std::istreambuf_iterator<char>(source_file)),
                             std::istreambuf_iterator<char>());
    for (const char* forbidden : {
             "newCommandQueue", "newBuffer", "newCommandBuffer", "commandBuffer",
             "setenv(", "putenv(", "getenv(", "dispatch_once", "call_once",
             "pthread_once", "atexit("}) {
        CHECK(source.find(forbidden) == std::string::npos);
    }

    const std::string header_path =
        std::string(LAPLACE_SOURCE_DIR) + "/src/metal_capability_snapshot.h";
    std::ifstream header_file(header_path);
    CHECK(header_file.good());
    const std::string header((std::istreambuf_iterator<char>(header_file)),
                             std::istreambuf_iterator<char>());
    for (const char* forbidden : {
             "GGML", "F16", "Q4", "BF16", "MTLGPUFamily", "format", "family",
             "tensor", "artifact"}) {
        CHECK(header.find(forbidden) == std::string::npos);
    }
#else
    CHECK(false);
#endif
}

void test_api_is_opaque_and_move_only() {
    CHECK(!std::is_copy_constructible_v<MetalCapabilitySnapshot>);
    CHECK(!std::is_copy_assignable_v<MetalCapabilitySnapshot>);
    CHECK(std::is_move_constructible_v<MetalCapabilitySnapshot>);
    CHECK(std::is_move_assignable_v<MetalCapabilitySnapshot>);
}

void test_actual_pipeline_and_opaque_identity(
    const MetalCapabilitySnapshot& snapshot, const CodecProgramIdentity& identity) {
    CHECK(snapshot.program_identity() == identity);
    CHECK(snapshot.capability_digest() != MetalCapabilityDigest{});
    CHECK(snapshot.pipeline_constructed());
    CHECK(snapshot.thread_execution_width() != 0);
    CHECK(snapshot.max_total_threads_per_threadgroup() != 0);
    CHECK(snapshot.dispatch_minimum() >= snapshot.thread_execution_width());
    CHECK(snapshot.dispatch_minimum() <= snapshot.max_total_threads_per_threadgroup());
    CHECK(snapshot.dispatch_minimum() % snapshot.thread_execution_width() == 0);

    const MetalCapabilityProbeAudit& audit = snapshot.audit();
    CHECK(audit.devices_created == 1);
    CHECK(audit.libraries_created == 1);
    CHECK(audit.functions_created == 1);
    CHECK(audit.pipelines_created == 1);
    CHECK(audit.snapshot_published);
    check_no_forbidden_resources(audit);
}

void test_unknown_and_tampered_programs_fail_closed(
    const CodecProgramRegistry& registry) {
    CodecProgramIdentity unknown = registry.programs()[0].identity();
    unknown.contract_digest[0] ^= 0x80u;
    const MetalCapabilityProbeResult unknown_result =
        probe_metal_capability(registry, unknown);
    const auto* unknown_failure = failure_of(unknown_result);
    CHECK(unknown_failure != nullptr);
    if (unknown_failure)
        CHECK(unknown_failure->code == MetalCapabilityProbeError::UnknownProgram);

    CodecProgramIdentity tampered = registry.programs()[0].identity();
    ++tampered.abi_version;
    const MetalCapabilityProbeResult tampered_result =
        probe_metal_capability(registry, tampered);
    const auto* tampered_failure = failure_of(tampered_result);
    CHECK(tampered_failure != nullptr);
    if (tampered_failure)
        CHECK(tampered_failure->code == MetalCapabilityProbeError::UnknownProgram);
}

void test_distinct_program_capabilities(const CodecProgramRegistry& registry) {
    const CodecProgramIdentity first = registry.programs()[0].identity();
    const CodecProgramIdentity second = registry.programs()[1].identity();
    const MetalCapabilityProbeResult first_result =
        probe_metal_capability(registry, first);
    const MetalCapabilityProbeResult second_result =
        probe_metal_capability(registry, second);
    CHECK(failure_of(first_result) == nullptr);
    CHECK(failure_of(second_result) == nullptr);
    const auto* first_snapshot = snapshot_of(first_result);
    const auto* second_snapshot = snapshot_of(second_result);
    CHECK(first_snapshot != nullptr);
    CHECK(second_snapshot != nullptr);
    if (!first_snapshot || !second_snapshot) return;
    test_actual_pipeline_and_opaque_identity(*first_snapshot, first);
    test_actual_pipeline_and_opaque_identity(*second_snapshot, second);
    CHECK(first_snapshot->capability_digest() != second_snapshot->capability_digest());

    // Correlate each opaque identity with its private application contract
    // without exposing a recipe or representation enum in the public API.
    for (size_t index = 0; index != registry.programs().size(); ++index) {
        const std::span<const uint8_t> bytes = registry.programs()[index].canonical_bytes();
        CHECK(bytes.size() >= 14);
        if (bytes.size() < 14) continue;
        const uint16_t private_contract_tag =
            static_cast<uint16_t>(bytes[12]) |
            (static_cast<uint16_t>(bytes[13]) << 8u);
        CHECK(private_contract_tag == 1 || private_contract_tag == 2);
        const MetalCapabilitySnapshot* snapshot = index == 0 ? first_snapshot : second_snapshot;
        const uint32_t multiplier = private_contract_tag == 1 ? 1u : 2u;
        CHECK(snapshot->thread_execution_width() <=
              std::numeric_limits<uint32_t>::max() / multiplier);
        CHECK(snapshot->dispatch_minimum() ==
              snapshot->thread_execution_width() * multiplier);
    }
}

void test_repeat_is_deterministic(const CodecProgramRegistry& registry) {
    const CodecProgramIdentity identity = registry.programs()[0].identity();
    const MetalCapabilityProbeResult first = probe_metal_capability(registry, identity);
    const MetalCapabilityProbeResult second = probe_metal_capability(registry, identity);
    CHECK(failure_of(first) == nullptr);
    CHECK(failure_of(second) == nullptr);
    const auto* first_snapshot = snapshot_of(first);
    const auto* second_snapshot = snapshot_of(second);
    CHECK(first_snapshot != nullptr);
    CHECK(second_snapshot != nullptr);
    if (!first_snapshot || !second_snapshot) return;
    CHECK(first_snapshot->program_identity() == second_snapshot->program_identity());
    CHECK(first_snapshot->capability_digest() == second_snapshot->capability_digest());
    CHECK(first_snapshot->audit() == second_snapshot->audit());
    CHECK(first_snapshot->thread_execution_width() == second_snapshot->thread_execution_width());
    CHECK(first_snapshot->max_total_threads_per_threadgroup() ==
          second_snapshot->max_total_threads_per_threadgroup());
    CHECK(first_snapshot->dispatch_minimum() == second_snapshot->dispatch_minimum());
}

void test_failures_do_not_publish_partial_snapshots(const CodecProgramRegistry& registry) {
#if defined(LAPLACE_TESTING)
    const CodecProgramIdentity identity = registry.programs()[0].identity();
    for (const MetalCapabilityInjectedFailure fault : {
             MetalCapabilityInjectedFailure::Library,
             MetalCapabilityInjectedFailure::Function,
             MetalCapabilityInjectedFailure::Pipeline}) {
        const MetalCapabilityProbeResult result =
            probe_metal_capability_for_testing(registry, identity, fault);
        const auto* failure = failure_of(result);
        CHECK(failure != nullptr);
        CHECK(snapshot_of(result) == nullptr);
        if (failure) {
            CHECK(!failure->audit.snapshot_published);
            check_no_forbidden_resources(failure->audit);
            const uint32_t expected_libraries =
                fault == MetalCapabilityInjectedFailure::Library ? 0u : 1u;
            const uint32_t expected_functions =
                fault == MetalCapabilityInjectedFailure::Pipeline ? 1u : 0u;
            const uint32_t expected_pipelines =
                fault == MetalCapabilityInjectedFailure::Pipeline ? 1u : 0u;
            CHECK(failure->audit.devices_created == 1);
            CHECK(failure->audit.libraries_created == expected_libraries);
            CHECK(failure->audit.functions_created == expected_functions);
            CHECK(failure->audit.pipelines_created == expected_pipelines);
        }
    }
#else
    (void)registry;
#endif
}

} // namespace

int main() {
    test_api_is_opaque_and_move_only();
    test_source_contract();
    const CodecProgramRegistry registry = make_application_codec_registry();
    CHECK(registry.programs().size() == 2);

    const MetalCapabilityProbeResult result = probe_metal_capability(
        registry, registry.programs()[0].identity());
    const auto* failure = failure_of(result);
    if (failure && failure->code == MetalCapabilityProbeError::NoDevice) {
        CHECK(failure->audit.devices_created == 0);
        check_no_forbidden_resources(failure->audit);
        std::printf("SKIP: no Metal device\n");
        return 0;
    }
    CHECK(failure == nullptr);
    const auto* snapshot = snapshot_of(result);
    CHECK(snapshot != nullptr);
    if (!snapshot) return test_summary("metal capability snapshot");

    test_unknown_and_tampered_programs_fail_closed(registry);
    test_distinct_program_capabilities(registry);
    test_repeat_is_deterministic(registry);
    test_failures_do_not_publish_partial_snapshots(registry);
    return test_summary("metal capability snapshot");
}
