#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <variant>

#include "compatibility_report.h"
#include "mmap.h"

namespace Laplace {

struct Sha256Digest {
    std::array<uint8_t, 32> bytes{};

    std::string hex() const;
    friend bool operator==(const Sha256Digest&, const Sha256Digest&) = default;
};

enum class ArtifactRole : uint8_t {
    Primary = 1,
    Shard = 2,
    Sidecar = 3,
};

struct ArtifactSource {
    std::string_view path;
    ArtifactRole role = ArtifactRole::Primary;
    ArtifactId id{};
};

class PackageView {
public:
    ArtifactId artifact_id() const { return artifact_id_; }
    ArtifactRole role() const { return role_; }
    std::span<const uint8_t> bytes() const { return bytes_; }
    const Sha256Digest& digest() const { return digest_; }
    MappedReadAdviceResult advise_read_ranges(
        std::span<const MappedReadRange> ranges) const;

private:
    using ReadAdviceFunction = MappedReadAdviceResult (*)(
        const void*, std::span<const MappedReadRange>);
    friend class ArtifactSet;
    PackageView(ArtifactId artifact_id, ArtifactRole role, std::span<const uint8_t> bytes,
                Sha256Digest digest, std::shared_ptr<const void> owner,
                ReadAdviceFunction advice = nullptr)
        : artifact_id_(artifact_id), role_(role), bytes_(bytes), digest_(digest),
          owner_(std::move(owner)), advice_(advice) {}

    ArtifactId artifact_id_;
    ArtifactRole role_ = ArtifactRole::Primary;
    std::span<const uint8_t> bytes_;
    Sha256Digest digest_;
    std::shared_ptr<const void> owner_;
    ReadAdviceFunction advice_ = nullptr;
};

#ifdef LAPLACE_ARTIFACT_SET_TESTING
struct ArtifactIdentityForTesting {
    uint64_t device;
    uint64_t inode;
    uint64_t size;
    int64_t mtime_seconds;
    int64_t mtime_nanoseconds;
    int64_t ctime_seconds;
    int64_t ctime_nanoseconds;
    uint64_t generation;
};

bool artifact_identity_equal_for_testing(const ArtifactIdentityForTesting& left,
                                         const ArtifactIdentityForTesting& right);
#endif

class ArtifactSet {
public:
    // Copies caller bytes into one immutable, reference-counted owner. The
    // returned view remains valid after the input span and ArtifactSet have
    // gone out of scope; it never aliases mutable caller storage.
    static std::variant<PackageView, CompatibilityReport>
    make_owned_blob(ArtifactId id, ArtifactRole role, std::span<const uint8_t> bytes);

    static std::variant<ArtifactSet, CompatibilityReport>
    load_single_file(std::string_view path);
    static std::variant<ArtifactSet, CompatibilityReport>
    load_graph(std::span<const ArtifactSource> sources);

    std::variant<PackageView, CompatibilityReport> view(ArtifactId id) const;

#ifdef LAPLACE_ARTIFACT_SET_TESTING
    static void set_test_hook(void (*hook)(int));
    static void set_test_advice_hook(void (*hook)(size_t, size_t));
    static uint32_t live_mapping_count_for_testing();
#endif

private:
    explicit ArtifactSet(std::shared_ptr<const void> owner)
        : owner_(std::move(owner)) {}

    std::shared_ptr<const void> owner_;
};

} // namespace Laplace
