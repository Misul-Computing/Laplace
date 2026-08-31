#include "session_resources.h"

#include <Metal/Metal.h>

#include <atomic>

namespace Laplace {

namespace {

std::atomic<uint32_t> g_live_resources{0};

} // namespace

struct SessionResources::Impl {
    RuntimeCapabilities capabilities;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> command_queue = nil;

    ~Impl() {
        [command_queue release];
        [device release];
    }
};

SessionResources::SessionResources(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
    if (impl_) ++g_live_resources;
}

SessionResources::SessionResources() = default;

SessionResources::~SessionResources() {
    if (impl_) --g_live_resources;
}

SessionResources::SessionResources(SessionResources&&) noexcept = default;
SessionResources& SessionResources::operator=(SessionResources&&) noexcept = default;

SessionResourcesCandidate::SessionResourcesCandidate(SessionResourceFault fault)
    : fault_(fault), impl_(std::make_unique<SessionResources::Impl>()) {}

SessionResourcesCandidate::~SessionResourcesCandidate() = default;
SessionResourcesCandidate::SessionResourcesCandidate(SessionResourcesCandidate&&) noexcept = default;
SessionResourcesCandidate& SessionResourcesCandidate::operator=(SessionResourcesCandidate&&) noexcept = default;

void SessionResourcesCandidate::query_optional_metal() {
    if (!impl_ || fault_ == SessionResourceFault::DeviceQuery) return;
    impl_->device = MTLCreateSystemDefaultDevice();
    if (impl_->device == nil) return;
    impl_->capabilities.metal_device = true;

    // V1 registers no canonical Metal descriptor, so a library and pipeline are
    // not semantically eligible. The queue is created locally only to establish
    // device capability. CPU replanning owns no Metal resource.
    if (fault_ == SessionResourceFault::CommandQueue) return;
    impl_->command_queue = [impl_->device newCommandQueue];
}

const RuntimeCapabilities& SessionResourcesCandidate::capabilities() const noexcept {
    static const RuntimeCapabilities none;
    return impl_ ? impl_->capabilities : none;
}

SessionResources SessionResourcesCandidate::finish_cpu_plan() noexcept {
    impl_.reset();
    return SessionResources();
}

uint32_t session_resources_live_count() {
    return g_live_resources.load();
}

} // namespace Laplace
