#pragma once

#include <cstdint>
#include <memory>

#include "execution_plan.h"

namespace Laplace {

// V1 owns only the session-local device and command queue. Canonical Metal
// libraries and pipelines are not constructed or claimed yet.
enum class SessionResourceFault : uint8_t { None = 0, DeviceQuery = 1, CommandQueue = 2 };

class SessionResources {
public:
    SessionResources();
    ~SessionResources();
    SessionResources(SessionResources&&) noexcept;
    SessionResources& operator=(SessionResources&&) noexcept;
    SessionResources(const SessionResources&) = delete;
    SessionResources& operator=(const SessionResources&) = delete;

private:
    struct Impl;
    explicit SessionResources(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;

    friend class SessionResourcesCandidate;
};

class SessionResourcesCandidate {
public:
    explicit SessionResourcesCandidate(SessionResourceFault fault);
    ~SessionResourcesCandidate();
    SessionResourcesCandidate(SessionResourcesCandidate&&) noexcept;
    SessionResourcesCandidate& operator=(SessionResourcesCandidate&&) noexcept;
    SessionResourcesCandidate(const SessionResourcesCandidate&) = delete;
    SessionResourcesCandidate& operator=(const SessionResourcesCandidate&) = delete;

    void query_optional_metal();
    const RuntimeCapabilities& capabilities() const noexcept;
    SessionResources finish_cpu_plan() noexcept;

private:
    SessionResourceFault fault_ = SessionResourceFault::None;
    std::unique_ptr<SessionResources::Impl> impl_;
};

uint32_t session_resources_live_count();

} // namespace Laplace
