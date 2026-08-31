#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <variant>

#include "codec_program.h"

namespace Laplace {

using MetalCapabilityDigest = std::array<uint8_t, 32>;

enum class MetalCapabilityProbeError : uint8_t {
    NoDevice = 1,
    RegistryInvalid = 2,
    UnknownProgram = 3,
    LibraryCompileFailed = 4,
    FunctionLookupFailed = 5,
    PipelineBuildFailed = 6,
    PipelineLimitsInvalid = 7,
};

struct MetalCapabilityProbeAudit {
    uint32_t devices_created = 0;
    uint32_t libraries_created = 0;
    uint32_t functions_created = 0;
    uint32_t pipelines_created = 0;
    uint32_t command_queues_created = 0;
    uint32_t command_buffers_created = 0;
    uint32_t buffers_created = 0;
    uint32_t token_sessions_created = 0;
    uint32_t global_cache_mutations = 0;
    uint32_t environment_mutations = 0;
    bool snapshot_published = false;

    friend bool operator==(const MetalCapabilityProbeAudit&,
                           const MetalCapabilityProbeAudit&) = default;
};

struct MetalCapabilityProbeFailure {
    MetalCapabilityProbeError code = MetalCapabilityProbeError::NoDevice;
    MetalCapabilityProbeAudit audit;

    friend bool operator==(const MetalCapabilityProbeFailure&,
                           const MetalCapabilityProbeFailure&) = default;
};

#if defined(LAPLACE_TESTING)
enum class MetalCapabilityInjectedFailure : uint8_t {
    Library = 1,
    Function = 2,
    Pipeline = 3,
};
#endif

class MetalCapabilitySnapshot;
using MetalCapabilityProbeResult =
    std::variant<MetalCapabilitySnapshot, MetalCapabilityProbeFailure>;

namespace metal_capability_detail {
MetalCapabilityProbeResult probe(const CodecProgramRegistry&,
                                 const CodecProgramIdentity&,
                                 uint8_t injected_failure);
} // namespace metal_capability_detail

class MetalCapabilitySnapshot {
public:
    ~MetalCapabilitySnapshot();
    MetalCapabilitySnapshot(MetalCapabilitySnapshot&&) noexcept;
    MetalCapabilitySnapshot& operator=(MetalCapabilitySnapshot&&) noexcept;
    MetalCapabilitySnapshot(const MetalCapabilitySnapshot&) = delete;
    MetalCapabilitySnapshot& operator=(const MetalCapabilitySnapshot&) = delete;

    CodecProgramIdentity program_identity() const noexcept;
    MetalCapabilityDigest capability_digest() const noexcept;
    uint32_t thread_execution_width() const noexcept;
    uint32_t max_total_threads_per_threadgroup() const noexcept;
    uint32_t dispatch_minimum() const noexcept;
    bool pipeline_constructed() const noexcept;
    const MetalCapabilityProbeAudit& audit() const noexcept;

private:
    struct Impl;
    explicit MetalCapabilitySnapshot(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;

    friend MetalCapabilityProbeResult metal_capability_detail::probe(
        const CodecProgramRegistry&, const CodecProgramIdentity&,
        uint8_t injected_failure);
};

MetalCapabilityProbeResult probe_metal_capability(
    const CodecProgramRegistry&, const CodecProgramIdentity&);

#if defined(LAPLACE_TESTING)
MetalCapabilityProbeResult probe_metal_capability_for_testing(
    const CodecProgramRegistry&, const CodecProgramIdentity&,
    MetalCapabilityInjectedFailure);
#endif

} // namespace Laplace
