// mmap.h - portable memory-mapped file
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace Laplace {

enum class SnapshotKind : uint8_t {
    None = 0,
    Clone = 1,
    DiskCopy = 2,
};

struct MappedReadRange {
    size_t offset = 0;
    size_t length = 0;
};

struct MappedReadAdviceResult {
    size_t requested_ranges = 0;
    size_t requested_bytes = 0;
    size_t file_advised_ranges = 0;
    bool memory_hint_applied = false;
    bool file_hint_applied = false;
    int error = 0;
};

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    bool open(const char* path);
    // Consumes fd on every path except a rejected self-transfer. exact_size must
    // be the current regular-file size.
    bool map_owned_fd(int fd, size_t exact_size);
    // Consumes fd on every path. The mapped bytes come from an immutable
    // snapshot, not the caller's mutable source inode. The private directory
    // excludes other accounts; same-account processes are trusted and are not
    // an adversarial isolation boundary during snapshot creation.
    bool map_snapshot_fd(int fd, size_t exact_size);
    MappedReadAdviceResult advise_read_ranges(std::span<const MappedReadRange> ranges) const;
    void close();

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    bool valid() const { return data_ != nullptr; }
    int fd() const { return fd_; }
    SnapshotKind snapshot_kind() const { return snapshot_kind_; }

#ifdef LAPLACE_ARTIFACT_SET_TESTING
    static void set_test_force_snapshot_fallback(bool force);
    static void set_test_snapshot_hook(void (*hook)(int));
    static size_t snapshot_fallback_max_bytes_for_testing();
#endif

private:
    int fd_ = -1;
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    SnapshotKind snapshot_kind_ = SnapshotKind::None;
};

} // namespace Laplace
