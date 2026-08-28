#include "artifact_set.h"

#include <CommonCrypto/CommonDigest.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
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
};

ArtifactIdentity identity_from_stat(const struct stat& st) {
    return {
        static_cast<uint64_t>(st.st_dev),
        static_cast<uint64_t>(st.st_ino),
        static_cast<uint64_t>(st.st_size),
        static_cast<int64_t>(st.st_mtimespec.tv_sec),
        static_cast<int64_t>(st.st_mtimespec.tv_nsec),
    };
}

bool identity_equal(const ArtifactIdentity& left, const ArtifactIdentity& right) {
    return left.device == right.device && left.inode == right.inode &&
           left.size == right.size && left.mtime_seconds == right.mtime_seconds &&
           left.mtime_nanoseconds == right.mtime_nanoseconds;
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

#ifdef LAPLACE_ARTIFACT_SET_TESTING
void (*g_test_hook)(int) = nullptr;
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

class ArtifactOwners {
public:
    std::vector<std::shared_ptr<const ArtifactBytesOwner>> artifacts;
};

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
                return identity_equal(known, before);
            }) != identities.end()) {
            ::close(fd);
            return package_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED, id,
                                   "package graph has a duplicate source artifact");
        }

        auto owner = std::make_shared<ArtifactBytesOwner>();
        if (!owner->mapping.map_owned_fd(fd, static_cast<size_t>(before.size))) return source_failure(id);
#ifdef LAPLACE_ARTIFACT_SET_TESTING
        owner->mapped_ = true;
        ++g_live_mapping_count;
#endif
        owner->id = id;
        owner->role = source.role;
        owner->digest = digest_bytes({owner->mapping.data(), owner->mapping.size()});

#ifdef LAPLACE_ARTIFACT_SET_TESTING
        if (g_test_hook) g_test_hook(owner->mapping.fd());
#endif

        struct stat after_stat {};
        if (fstat(owner->mapping.fd(), &after_stat) != 0 ||
            !identity_equal(before, identity_from_stat(after_stat))) {
            return package_failure(CompatibilityError::PACKAGE_SOURCE_CHANGED, id,
                                   "source identity changed while validating");
        }
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
    return PackageView(id, owner->role, {owner->mapping.data(), owner->mapping.size()}, owner->digest, owner);
}

#ifdef LAPLACE_ARTIFACT_SET_TESTING
bool artifact_identity_equal_for_testing(const ArtifactIdentityForTesting& left,
                                         const ArtifactIdentityForTesting& right) {
    return left.device == right.device && left.inode == right.inode &&
           left.size == right.size && left.mtime_seconds == right.mtime_seconds &&
           left.mtime_nanoseconds == right.mtime_nanoseconds;
}

void ArtifactSet::set_test_hook(void (*hook)(int)) {
    g_test_hook = hook;
}

uint32_t ArtifactSet::live_mapping_count_for_testing() {
    return g_live_mapping_count;
}
#endif

} // namespace Laplace
