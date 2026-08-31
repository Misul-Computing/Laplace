#include "mmap.h"

#include <array>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>

#include <fcntl.h>
#ifdef __APPLE__
#include <sys/clonefile.h>
#include <sys/mount.h>
#endif
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace Laplace {

namespace {

constexpr size_t kSnapshotFallbackMaxBytes = 1ULL << 30;

#ifdef LAPLACE_ARTIFACT_SET_TESTING
bool g_force_snapshot_fallback = false;
void (*g_snapshot_hook)(int) = nullptr;
#endif

class PrivateTempDirectory {
public:
    PrivateTempDirectory() = default;
    ~PrivateTempDirectory() { close(); }

    PrivateTempDirectory(const PrivateTempDirectory&) = delete;
    PrivateTempDirectory& operator=(const PrivateTempDirectory&) = delete;

    bool open() {
        char path[] = "/private/tmp/laplace-artifact-snapshot-dir-XXXXXX";
        if (::mkdtemp(path) == nullptr) return false;
        path_ = path;
        dir_fd_ = ::open(path_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
        if (dir_fd_ < 0 || ::fchmod(dir_fd_, 0700) != 0) {
            const int saved_errno = errno;
            close();
            errno = saved_errno;
            return false;
        }
        return true;
    }

    int fd() const { return dir_fd_; }

    void remove_payload() const {
        const int saved_errno = errno;
        if (dir_fd_ >= 0) (void)::unlinkat(dir_fd_, "payload", 0);
        errno = saved_errno;
    }

    void close() {
        const int saved_errno = errno;
        if (dir_fd_ >= 0) {
            ::close(dir_fd_);
            dir_fd_ = -1;
        }
        if (!path_.empty()) {
            (void)::unlinkat(AT_FDCWD, path_.c_str(), AT_REMOVEDIR);
            path_.clear();
        }
        errno = saved_errno;
    }

private:
    int dir_fd_ = -1;
    std::string path_;
};

bool write_all(int fd, const uint8_t* data, size_t size) {
    size_t offset = 0;
    while (offset < size) {
        const ssize_t written = ::write(fd, data + offset, size - offset);
        if (written < 0 && errno == EINTR) continue;
        if (written == 0) {
            errno = EIO;
            return false;
        }
        if (written < 0) return false;
        offset += static_cast<size_t>(written);
    }
    return true;
}

int clone_snapshot_fd(int source_fd, size_t exact_size, int& failure_errno) {
#ifdef __APPLE__
    PrivateTempDirectory directory;
    if (!directory.open()) {
        failure_errno = errno;
        return -1;
    }

    constexpr char kPayloadName[] = "payload";
    if (::fclonefileat(source_fd, directory.fd(), kPayloadName,
                       CLONE_NOFOLLOW_ANY | CLONE_RESOLVE_BENEATH) != 0) {
        failure_errno = errno;
        return -1;
    }

    const int snapshot_fd = ::openat(directory.fd(), kPayloadName,
                                     O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (snapshot_fd < 0) {
        failure_errno = errno;
        directory.remove_payload();
        return -1;
    }
#ifdef LAPLACE_ARTIFACT_SET_TESTING
    if (g_snapshot_hook) g_snapshot_hook(directory.fd());
#endif
    struct stat entry {};
    struct stat st {};
    if (::fstatat(directory.fd(), kPayloadName, &entry, AT_SYMLINK_NOFOLLOW) != 0) {
        failure_errno = errno;
        ::close(snapshot_fd);
        directory.remove_payload();
        return -1;
    }
    if (::fstat(snapshot_fd, &st) != 0) {
        failure_errno = errno;
        ::close(snapshot_fd);
        directory.remove_payload();
        return -1;
    }
    if (entry.st_dev != st.st_dev || entry.st_ino != st.st_ino) {
        failure_errno = EIO;
        ::close(snapshot_fd);
        directory.remove_payload();
        return -1;
    }
    if (::unlinkat(directory.fd(), kPayloadName, 0) != 0) {
        failure_errno = errno;
        directory.remove_payload();
        ::close(snapshot_fd);
        return -1;
    }
    if (!S_ISREG(st.st_mode) || st.st_size <= 0 ||
        static_cast<uint64_t>(st.st_size) != exact_size) {
        failure_errno = EINVAL;
        ::close(snapshot_fd);
        return -1;
    }
    return snapshot_fd;
#else
    (void)source_fd;
    (void)exact_size;
    failure_errno = ENOTSUP;
    return -1;
#endif
}

bool fallback_space_available(size_t exact_size, int& failure_errno) {
#ifdef __APPLE__
    struct statfs volume {};
    if (::statfs("/private/tmp", &volume) != 0) {
        failure_errno = errno;
        return false;
    }
    const uint64_t free_blocks = static_cast<uint64_t>(volume.f_bavail);
    const uint64_t block_size = static_cast<uint64_t>(volume.f_bsize);
    if (block_size != 0 && free_blocks > std::numeric_limits<uint64_t>::max() / block_size) {
        failure_errno = EOVERFLOW;
        return false;
    }
    if (block_size == 0) {
        failure_errno = EINVAL;
        return false;
    }
    if (free_blocks * block_size < exact_size) {
        failure_errno = ENOSPC;
        return false;
    }
    return free_blocks * block_size >= exact_size;
#else
    (void)exact_size;
    (void)failure_errno;
    return true;
#endif
}

int copy_snapshot_fd(int source_fd, size_t exact_size) {
    if (exact_size > kSnapshotFallbackMaxBytes) {
        errno = EFBIG;
        return -1;
    }
    int space_errno = 0;
    if (!fallback_space_available(exact_size, space_errno)) {
        errno = space_errno == 0 ? ENOSPC : space_errno;
        return -1;
    }

    PrivateTempDirectory directory;
    if (!directory.open()) return -1;
    constexpr char kPayloadName[] = "payload";
    const int writable_fd = ::openat(directory.fd(), kPayloadName,
                                     O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (writable_fd < 0) return -1;
    if (exact_size > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
        const int saved_errno = EOVERFLOW;
        ::close(writable_fd);
        directory.remove_payload();
        errno = saved_errno;
        return -1;
    }
    if (::ftruncate(writable_fd, static_cast<off_t>(exact_size)) != 0) {
        const int saved_errno = errno;
        ::close(writable_fd);
        directory.remove_payload();
        errno = saved_errno;
        return -1;
    }

    std::unique_ptr<uint8_t[]> buffer(new (std::nothrow) uint8_t[1024 * 1024]);
    if (!buffer) {
        ::close(writable_fd);
        directory.remove_payload();
        errno = ENOMEM;
        return -1;
    }
    size_t offset = 0;
    while (offset < exact_size) {
        const size_t requested = std::min<size_t>(1024 * 1024, exact_size - offset);
        ssize_t received;
        do {
            received = ::pread(source_fd, buffer.get(), requested, static_cast<off_t>(offset));
        } while (received < 0 && errno == EINTR);
        if (received <= 0 || !write_all(writable_fd, buffer.get(), static_cast<size_t>(received))) {
            const int saved_errno = received < 0 ? errno : EIO;
            ::close(writable_fd);
            directory.remove_payload();
            errno = saved_errno;
            return -1;
        }
        offset += static_cast<size_t>(received);
    }
    if (::close(writable_fd) != 0) {
        const int saved_errno = errno;
        directory.remove_payload();
        errno = saved_errno;
        return -1;
    }

    const int snapshot_fd = ::openat(directory.fd(), kPayloadName,
                                     O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (snapshot_fd < 0) {
        directory.remove_payload();
        return -1;
    }
#ifdef LAPLACE_ARTIFACT_SET_TESTING
    if (g_snapshot_hook) g_snapshot_hook(directory.fd());
#endif
    struct stat entry {};
    struct stat st {};
    if (::fstatat(directory.fd(), kPayloadName, &entry, AT_SYMLINK_NOFOLLOW) != 0) {
        const int saved_errno = errno;
        ::close(snapshot_fd);
        directory.remove_payload();
        errno = saved_errno;
        return -1;
    }
    if (::fstat(snapshot_fd, &st) != 0) {
        const int saved_errno = errno;
        ::close(snapshot_fd);
        directory.remove_payload();
        errno = saved_errno;
        return -1;
    }
    if (entry.st_dev != st.st_dev || entry.st_ino != st.st_ino) {
        const int saved_errno = EIO;
        ::close(snapshot_fd);
        directory.remove_payload();
        errno = saved_errno;
        return -1;
    }
    if (!S_ISREG(st.st_mode) || st.st_size <= 0 || static_cast<uint64_t>(st.st_size) != exact_size) {
        const int saved_errno = EINVAL;
        ::close(snapshot_fd);
        directory.remove_payload();
        errno = saved_errno;
        return -1;
    }
    if (::unlinkat(directory.fd(), kPayloadName, 0) != 0) {
        const int saved_errno = errno;
        directory.remove_payload();
        ::close(snapshot_fd);
        errno = saved_errno;
        return -1;
    }
    return snapshot_fd;
}

} // namespace

MappedFile::~MappedFile() { close(); }

bool MappedFile::open(const char* path) {
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "mmap: open failed for %s\n", path);
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        ::close(fd);
        fprintf(stderr, "mmap: fstat failed for %s\n", path);
        return false;
    }
    if (!map_owned_fd(fd, static_cast<size_t>(st.st_size))) {
        fprintf(stderr, "mmap: mmap failed for %s\n", path);
        return false;
    }
    return true;
}

bool MappedFile::map_owned_fd(int fd, size_t exact_size) {
    if (fd < 0) {
        errno = EBADF;
        return false;
    }
    if (fd == fd_) {
        errno = EINVAL;
        return false;
    }
    struct stat st {};
    if (exact_size == 0) {
        const int saved_errno = EINVAL;
        ::close(fd);
        errno = saved_errno;
        return false;
    }
    if (fstat(fd, &st) != 0) {
        const int saved_errno = errno;
        ::close(fd);
        errno = saved_errno;
        return false;
    }
    if (st.st_size < 0 || static_cast<uint64_t>(st.st_size) != exact_size) {
        const int saved_errno = EINVAL;
        ::close(fd);
        errno = saved_errno;
        return false;
    }
    void* p = ::mmap(nullptr, exact_size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        const int saved_errno = errno;
        ::close(fd);
        errno = saved_errno;
        return false;
    }
    close();
    fd_ = fd;
    data_ = static_cast<const uint8_t*>(p);
    size_ = exact_size;
    snapshot_kind_ = SnapshotKind::None;
    return true;
}

bool MappedFile::map_snapshot_fd(int fd, size_t exact_size) {
    if (fd < 0) {
        errno = EBADF;
        return false;
    }
    struct stat source_stat {};
    if (exact_size == 0) {
        const int saved_errno = EINVAL;
        ::close(fd);
        errno = saved_errno;
        return false;
    }
    if (fstat(fd, &source_stat) != 0) {
        const int saved_errno = errno;
        ::close(fd);
        errno = saved_errno;
        return false;
    }
    if (!S_ISREG(source_stat.st_mode) || source_stat.st_size <= 0 ||
        static_cast<uint64_t>(source_stat.st_size) != exact_size) {
        const int saved_errno = EINVAL;
        ::close(fd);
        errno = saved_errno;
        return false;
    }
#ifdef __APPLE__
    unsigned long snapshot_blocking_flags = UF_IMMUTABLE | UF_APPEND;
#ifdef UF_NOUNLINK
    snapshot_blocking_flags |= UF_NOUNLINK;
#endif
    if ((source_stat.st_flags & snapshot_blocking_flags) != 0) {
        const int saved_errno = EPERM;
        ::close(fd);
        errno = saved_errno;
        return false;
    }
#endif

    int clone_error = ENOTSUP;
    int snapshot_fd = -1;
#ifdef LAPLACE_ARTIFACT_SET_TESTING
    if (!g_force_snapshot_fallback) snapshot_fd = clone_snapshot_fd(fd, exact_size, clone_error);
#else
    snapshot_fd = clone_snapshot_fd(fd, exact_size, clone_error);
#endif
    SnapshotKind kind = SnapshotKind::Clone;
    bool attempted_fallback = false;
    if (snapshot_fd < 0 && (clone_error == ENOTSUP || clone_error == EXDEV || clone_error == EINVAL)) {
        attempted_fallback = true;
        snapshot_fd = copy_snapshot_fd(fd, exact_size);
        kind = SnapshotKind::DiskCopy;
    }
    if (snapshot_fd < 0) {
        const int saved_errno = attempted_fallback ? errno : clone_error;
        ::close(fd);
        errno = saved_errno;
        return false;
    }
    ::close(fd);
    if (!map_owned_fd(snapshot_fd, exact_size)) return false;
    snapshot_kind_ = kind;
    return true;
}

MappedReadAdviceResult MappedFile::advise_read_ranges(
    std::span<const MappedReadRange> ranges) const {
    MappedReadAdviceResult result;
    if (!valid() || ranges.empty()) {
        result.error = EINVAL;
        return result;
    }

    size_t requested_ranges = 0;
    size_t requested_bytes = 0;
    for (const MappedReadRange& range : ranges) {
        if (range.length == 0 || range.offset > size_ ||
            range.length > size_ - range.offset) {
            result.error = EINVAL;
            return result;
        }
        if (requested_bytes > std::numeric_limits<size_t>::max() - range.length) {
            result.error = EOVERFLOW;
            return result;
        }
        ++requested_ranges;
        requested_bytes += range.length;
    }
    result.requested_ranges = requested_ranges;
    result.requested_bytes = requested_bytes;

#ifdef __APPLE__
    for (const MappedReadRange& range : ranges) {
        if (::madvise(const_cast<uint8_t*>(data_ + range.offset), range.length,
                      MADV_WILLNEED) == 0) {
            result.memory_hint_applied = true;
        } else if (result.error == 0) {
            result.error = errno;
        }
    }

    std::array<struct radvisory, READ_ADVISE_RANGES_MAX> batch{};
    size_t batch_count = 0;
    auto flush = [&]() {
        if (batch_count == 0) return;
        const unsigned int count = static_cast<unsigned int>(batch_count);
        struct radvisoryv vector{F_RDADVISEV_NOAGE, count, batch.data()};
        bool applied = ::fcntl(fd_, F_RDADVISEV, &vector) == 0;
        int failure = applied ? 0 : errno;
        if (!applied) {
            vector.rav_flags = 0;
            applied = ::fcntl(fd_, F_RDADVISEV, &vector) == 0;
            if (applied) failure = 0;
            else failure = errno;
        }
        if (applied) {
            result.file_hint_applied = true;
            result.file_advised_ranges += batch_count;
        } else {
            for (size_t index = 0; index < batch_count; ++index) {
                if (::fcntl(fd_, F_RDADVISE, &batch[index]) == 0) {
                    result.file_hint_applied = true;
                    ++result.file_advised_ranges;
                } else if (failure == 0) {
                    failure = errno;
                }
            }
        }
        if (!result.file_hint_applied && result.error == 0) result.error = failure;
        batch_count = 0;
    };

    const uint64_t max_off_t = static_cast<uint64_t>(std::numeric_limits<off_t>::max());
    for (const MappedReadRange& range : ranges) {
        size_t offset = range.offset;
        size_t remaining = range.length;
        while (remaining != 0) {
            const size_t count = std::min<size_t>(remaining, INT_MAX);
            if (static_cast<uint64_t>(offset) > max_off_t) {
                if (result.error == 0) result.error = EOVERFLOW;
                remaining = 0;
                break;
            }
            batch[batch_count++] = {
                static_cast<off_t>(offset), static_cast<int>(count)};
            if (batch_count == batch.size()) flush();
            offset += count;
            remaining -= count;
        }
    }
    flush();
#else
    result.error = ENOTSUP;
#endif
    return result;
}

#ifdef LAPLACE_ARTIFACT_SET_TESTING
void MappedFile::set_test_force_snapshot_fallback(bool force) {
    g_force_snapshot_fallback = force;
}

void MappedFile::set_test_snapshot_hook(void (*hook)(int)) {
    g_snapshot_hook = hook;
}

size_t MappedFile::snapshot_fallback_max_bytes_for_testing() {
    return kSnapshotFallbackMaxBytes;
}
#endif

void MappedFile::close() {
    if (data_) {
        ::munmap(const_cast<uint8_t*>(data_), size_);
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    size_ = 0;
    snapshot_kind_ = SnapshotKind::None;
}

} // namespace Laplace
