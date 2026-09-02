#include "artifact_set.h"

#include <CommonCrypto/CommonDigest.h>

#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "mmap.h"

namespace Laplace {

namespace {

struct ArtifactIdentity {
    uint64_t device;
    uint64_t inode;
    uint64_t size;
    int64_t mtime_seconds;
    int64_t mtime_nanoseconds;
    int64_t ctime_seconds;
    int64_t ctime_nanoseconds;
    uint64_t generation;
};

ArtifactIdentity identity_from_stat(const struct stat& st) {
    return {
        static_cast<uint64_t>(st.st_dev),
        static_cast<uint64_t>(st.st_ino),
        static_cast<uint64_t>(st.st_size),
        static_cast<int64_t>(st.st_mtimespec.tv_sec),
        static_cast<int64_t>(st.st_mtimespec.tv_nsec),
        static_cast<int64_t>(st.st_ctimespec.tv_sec),
        static_cast<int64_t>(st.st_ctimespec.tv_nsec),
#ifdef __APPLE__
        static_cast<uint64_t>(st.st_gen),
#else
        0,
#endif
    };
}

bool identity_equal(const ArtifactIdentity& left, const ArtifactIdentity& right) {
    return left.device == right.device && left.inode == right.inode &&
           left.size == right.size && left.mtime_seconds == right.mtime_seconds &&
           left.mtime_nanoseconds == right.mtime_nanoseconds &&
           left.ctime_seconds == right.ctime_seconds &&
           left.ctime_nanoseconds == right.ctime_nanoseconds &&
           left.generation == right.generation;
}

bool same_file(const ArtifactIdentity& left, const ArtifactIdentity& right) {
    return left.device == right.device && left.inode == right.inode;
}

Sha256Digest digest_bytes(std::span<const uint8_t> bytes) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    constexpr size_t chunk_size = 1024 * 1024;
    for (size_t offset = 0; offset < bytes.size(); offset += chunk_size) {
        size_t chunk = std::min(chunk_size, bytes.size() - offset);
        CC_SHA256_Update(&context, bytes.data() + offset, static_cast<CC_LONG>(chunk));
    }
    Sha256Digest digest;
    CC_SHA256_Final(digest.bytes.data(), &context);
    return digest;
}

bool digest_fd(int fd, size_t size, Sha256Digest& digest) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    std::unique_ptr<uint8_t[]> buffer(new (std::nothrow) uint8_t[1024 * 1024]);
    if (!buffer) return false;
    size_t offset = 0;
    while (offset < size) {
        const size_t requested = std::min<size_t>(1024 * 1024, size - offset);
        ssize_t received;
        do {
            received = ::pread(fd, buffer.get(), requested, static_cast<off_t>(offset));
        } while (received < 0 && errno == EINTR);
        if (received <= 0) return false;
        CC_SHA256_Update(&context, buffer.get(), static_cast<CC_LONG>(received));
        offset += static_cast<size_t>(received);
    }
    CC_SHA256_Final(digest.bytes.data(), &context);
    return true;
}

#ifdef LAPLACE_ARTIFACT_SET_TESTING
void (*g_test_hook)(int) = nullptr;
void (*g_advice_hook)(size_t, size_t) = nullptr;
uint32_t g_live_mapping_count = 0;
#endif

class ArtifactBytesOwner {
public:
#ifdef LAPLACE_ARTIFACT_SET_TESTING
    ~ArtifactBytesOwner() {
        if (mapped_) --g_live_mapping_count;
    }
#endif

    ArtifactId id;
    ArtifactRole role = ArtifactRole::Primary;
    Sha256Digest digest;
    MappedFile mapping;

#ifdef LAPLACE_ARTIFACT_SET_TESTING
    bool mapped_ = false;
#endif
};

MappedReadAdviceResult advise_mapped_owner(
    const void* owner, std::span<const MappedReadRange> ranges) {
    if (owner == nullptr) {
        MappedReadAdviceResult result;
        result.error = EINVAL;
        return result;
    }
    const MappedReadAdviceResult result =
        static_cast<const ArtifactBytesOwner*>(owner)->mapping.advise_read_ranges(ranges);
#ifdef LAPLACE_ARTIFACT_SET_TESTING
    if (g_advice_hook != nullptr) {
        g_advice_hook(result.requested_ranges, result.requested_bytes);
    }
#endif
    return result;
}

class ArtifactOwners {
public:
    std::vector<std::shared_ptr<const ArtifactBytesOwner>> artifacts;
};

class ArtifactBlobOwner {
public:
    std::vector<uint8_t> bytes;
};

class ArtifactSubviewOwner {
public:
    std::shared_ptr<const void> parent;
    size_t offset = 0;
    size_t length = 0;
    MappedReadAdviceResult (*advice)(
        const void*, std::span<const MappedReadRange>) = nullptr;
};

MappedReadAdviceResult advise_subview_owner(
    const void* raw_owner, std::span<const MappedReadRange> ranges) {
    MappedReadAdviceResult rejected;
    rejected.error = EINVAL;
    if (raw_owner == nullptr) return rejected;
    const auto& owner = *static_cast<const ArtifactSubviewOwner*>(raw_owner);
    if (!owner.parent || owner.advice == nullptr) {
        rejected.error = ENOTSUP;
        return rejected;
    }
    try {
        std::vector<MappedReadRange> translated;
        translated.reserve(ranges.size());
        for (const MappedReadRange& range : ranges) {
            if (range.offset > owner.length ||
                range.length > owner.length - range.offset ||
                range.offset > std::numeric_limits<size_t>::max() - owner.offset) {
                return rejected;
            }
            translated.push_back({owner.offset + range.offset, range.length});
        }
        return owner.advice(owner.parent.get(), translated);
    } catch (const std::bad_alloc&) {
        rejected.error = ENOMEM;
        return rejected;
    }
}

constexpr size_t kMaximumOwnedBlobBytes = size_t{1} << 30;

CompatibilityReport package_failure(CompatibilityError code, ArtifactId id, std::string detail) {
    CompatibilityReport report = package_report(code, std::move(detail));
    report.artifact_id = id;
    report.artifact_index = id.value;
    return report;
}

CompatibilityReport source_failure(ArtifactId id) {
    return package_failure(CompatibilityError::PACKAGE_BOUNDS_INVALID, id,
                           "source is not one nonempty regular file");
}

CompatibilityReport snapshot_failure(ArtifactId id, int snapshot_errno) {
    std::string detail = "immutable artifact snapshot unavailable";
    if (snapshot_errno == EFBIG) {
        detail += ": disk-copy fallback is limited to 1 GiB; source must be on a clone-capable same filesystem";
    }
    if (snapshot_errno != 0) {
        detail += ": ";
        detail += std::strerror(snapshot_errno);
    }
    errno = snapshot_errno;
    return package_failure(CompatibilityError::PACKAGE_SNAPSHOT_UNAVAILABLE, id, std::move(detail));
}

ArtifactId source_id(const ArtifactSource& source, size_t index) {
    return source.id.value == UINT32_MAX ? ArtifactId{static_cast<uint32_t>(index)} : source.id;
}

} // namespace

std::string Sha256Digest::hex() const {
    static constexpr char digits[] = "0123456789abcdef";
    std::string text(64, '0');
    for (size_t i = 0; i < bytes.size(); ++i) {
        text[2 * i] = digits[bytes[i] >> 4];
        text[2 * i + 1] = digits[bytes[i] & 0x0f];
    }
    return text;
}

MappedReadAdviceResult PackageView::advise_read_ranges(
    std::span<const MappedReadRange> ranges) const {
    if (advice_ == nullptr || !owner_) {
        MappedReadAdviceResult result;
        result.error = ENOTSUP;
        return result;
    }
    return advice_(owner_.get(), ranges);
}

std::variant<PackageView, CompatibilityReport>
ArtifactSet::make_owned_blob(ArtifactId id, ArtifactRole role,
                             std::span<const uint8_t> bytes) {
    if (id.value == UINT32_MAX ||
        (role != ArtifactRole::Primary && role != ArtifactRole::Shard &&
         role != ArtifactRole::Sidecar)) {
        return package_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED, id,
                               "owned artifact blob has an invalid identity or role");
    }
    if (bytes.empty() || bytes.size() > kMaximumOwnedBlobBytes) {
        return package_failure(CompatibilityError::PACKAGE_BOUNDS_INVALID, id,
                               "owned artifact blob is empty or exceeds its 1 GiB bound");
    }
    try {
        auto owner = std::make_shared<ArtifactBlobOwner>();
        owner->bytes.assign(bytes.begin(), bytes.end());
        const Sha256Digest digest = digest_bytes(owner->bytes);
        return PackageView(id, role,
                           std::span<const uint8_t>(owner->bytes.data(), owner->bytes.size()),
                           digest, std::move(owner));
    } catch (const std::bad_alloc&) {
        return package_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED, id,
                               "owned artifact blob allocation failed");
    }
}

std::variant<PackageView, CompatibilityReport>
ArtifactSet::make_subview(const PackageView& source, ArtifactId id,
                          ArtifactRole role, size_t offset, size_t length) {
    if (id.value == UINT32_MAX ||
        (role != ArtifactRole::Primary && role != ArtifactRole::Shard &&
         role != ArtifactRole::Sidecar)) {
        return package_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED, id,
                               "artifact subview has an invalid identity or role");
    }
    if (!source.owner_ || length == 0 || offset > source.bytes_.size() ||
        length > source.bytes_.size() - offset) {
        return package_failure(CompatibilityError::PACKAGE_BOUNDS_INVALID, id,
                               "artifact subview range is empty or outside its source");
    }
    try {
        auto owner = std::make_shared<ArtifactSubviewOwner>();
        owner->parent = source.owner_;
        owner->offset = offset;
        owner->length = length;
        owner->advice = source.advice_;
        const std::span<const uint8_t> bytes = source.bytes_.subspan(offset, length);
        return PackageView(id, role, bytes, digest_bytes(bytes), std::move(owner),
                           source.advice_ == nullptr ? nullptr : advise_subview_owner);
    } catch (const std::bad_alloc&) {
        return package_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED, id,
                               "artifact subview allocation failed");
    }
}

std::variant<ArtifactSet, CompatibilityReport>
ArtifactSet::load_single_file(std::string_view path) {
    ArtifactSource source{path, ArtifactRole::Primary, ArtifactId{0}};
    return load_graph(std::span<const ArtifactSource>(&source, 1));
}

std::variant<ArtifactSet, CompatibilityReport>
ArtifactSet::load_graph(std::span<const ArtifactSource> sources) {
    if (sources.empty() || sources.size() > UINT32_MAX) {
        return package_report(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                              "package graph has no stable artifact members");
    }
    size_t primary_count = 0;
    std::vector<ArtifactId> ids;
    ids.reserve(sources.size());
    for (size_t index = 0; index != sources.size(); ++index) {
        const ArtifactSource& source = sources[index];
        const ArtifactId id = source_id(source, index);
        if (source.path.empty() || id.value == UINT32_MAX ||
            (source.role != ArtifactRole::Primary && source.role != ArtifactRole::Shard &&
             source.role != ArtifactRole::Sidecar)) {
            return package_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED, id,
                                   "package member is not declared with an allowed role");
        }
        primary_count += source.role == ArtifactRole::Primary;
        if (std::find(ids.begin(), ids.end(), id) != ids.end()) {
            return package_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED, id,
                                   "package graph has a duplicate artifact member");
        }
        ids.push_back(id);
    }
    if (primary_count != 1) {
        return package_report(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                              "package graph must declare exactly one primary artifact");
    }
    auto owners = std::make_shared<ArtifactOwners>();
    owners->artifacts.reserve(sources.size());
    std::vector<ArtifactIdentity> identities;
    identities.reserve(sources.size());
    for (size_t index = 0; index != sources.size(); ++index) {
        const ArtifactSource& source = sources[index];
        const ArtifactId id = ids[index];
        const std::string path(source.path);
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) return source_failure(id);

        struct stat before_stat {};
        if (fstat(fd, &before_stat) != 0 || !S_ISREG(before_stat.st_mode) || before_stat.st_size <= 0 ||
            static_cast<uint64_t>(before_stat.st_size) > std::numeric_limits<size_t>::max()) {
            ::close(fd);
            return source_failure(id);
        }
        const ArtifactIdentity before = identity_from_stat(before_stat);
        if (std::find_if(identities.begin(), identities.end(), [&](const ArtifactIdentity& known) {
                return same_file(known, before);
            }) != identities.end()) {
            ::close(fd);
            return package_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED, id,
                                   "package graph has a duplicate source artifact");
        }

        const int validation_fd = ::fcntl(fd, F_DUPFD_CLOEXEC, 0);
        if (validation_fd < 0) {
            ::close(fd);
            return source_failure(id);
        }

        auto owner = std::make_shared<ArtifactBytesOwner>();
        if (!owner->mapping.map_snapshot_fd(fd, static_cast<size_t>(before.size))) {
            const int snapshot_errno = errno;
            ::close(validation_fd);
            errno = snapshot_errno;
            return snapshot_failure(id, snapshot_errno);
        }
#ifdef LAPLACE_ARTIFACT_SET_TESTING
        owner->mapped_ = true;
        ++g_live_mapping_count;
#endif
        owner->id = id;
        owner->role = source.role;
        // Hash through the owned descriptor. The mapped view is kept for the
        // package, but walking it here faults the same pages that Metal will
        // later stream. A sequential descriptor read lets the kernel schedule
        // file I/O without making the load-time digest walk the GPU working set.
        if (!digest_fd(owner->mapping.fd(), owner->mapping.size(), owner->digest)) {
            ::close(validation_fd);
            return package_failure(CompatibilityError::PACKAGE_BOUNDS_INVALID, id,
                                   "source digest could not be read from the validated snapshot");
        }

#ifdef LAPLACE_ARTIFACT_SET_TESTING
        if (g_test_hook) g_test_hook(owner->mapping.fd());
#endif

        if (owner->mapping.snapshot_kind() == SnapshotKind::DiskCopy) {
            Sha256Digest source_digest;
            if (!digest_fd(validation_fd, static_cast<size_t>(before.size), source_digest) ||
                source_digest != owner->digest) {
                ::close(validation_fd);
                return package_failure(CompatibilityError::PACKAGE_SOURCE_CHANGED, id,
                                       "source bytes changed while creating disk snapshot");
            }
        }

        struct stat after_stat {};
        if (fstat(validation_fd, &after_stat) != 0 ||
            !identity_equal(before, identity_from_stat(after_stat))) {
            ::close(validation_fd);
            return package_failure(CompatibilityError::PACKAGE_SOURCE_CHANGED, id,
                                   "source identity changed while validating");
        }
        ::close(validation_fd);
        identities.push_back(before);
        owners->artifacts.push_back(std::move(owner));
    }

    return ArtifactSet(std::move(owners));
}

std::variant<PackageView, CompatibilityReport> ArtifactSet::view(ArtifactId id) const {
    if (!owner_) return package_failure(CompatibilityError::IR_REFERENCE_INVALID, id,
                                        "artifact ID is not in the validated package");
    const auto* owners = static_cast<const ArtifactOwners*>(owner_.get());
    const auto entry = std::find_if(owners->artifacts.begin(), owners->artifacts.end(), [&](const auto& owner) {
        return owner->id == id;
    });
    if (entry == owners->artifacts.end()) return package_failure(CompatibilityError::IR_REFERENCE_INVALID, id,
                                                                  "artifact ID is not in the validated package");
    const std::shared_ptr<const ArtifactBytesOwner>& owner = *entry;
    return PackageView(id, owner->role, {owner->mapping.data(), owner->mapping.size()},
                       owner->digest, owner, advise_mapped_owner);
}

#ifdef LAPLACE_ARTIFACT_SET_TESTING
bool artifact_identity_equal_for_testing(const ArtifactIdentityForTesting& left,
                                         const ArtifactIdentityForTesting& right) {
    return left.device == right.device && left.inode == right.inode &&
           left.size == right.size && left.mtime_seconds == right.mtime_seconds &&
           left.mtime_nanoseconds == right.mtime_nanoseconds &&
           left.ctime_seconds == right.ctime_seconds &&
           left.ctime_nanoseconds == right.ctime_nanoseconds &&
           left.generation == right.generation;
}

void ArtifactSet::set_test_hook(void (*hook)(int)) {
    g_test_hook = hook;
}

void ArtifactSet::set_test_advice_hook(void (*hook)(size_t, size_t)) {
    g_advice_hook = hook;
}

uint32_t ArtifactSet::live_mapping_count_for_testing() {
    return g_live_mapping_count;
}
#endif

} // namespace Laplace
