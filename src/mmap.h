// mmap.h - portable memory-mapped file
#pragma once

#include <cstddef>
#include <cstdint>

namespace Laplace {

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    bool open(const char* path);
    // Consumes fd on every path. exact_size must be the current regular-file size.
    bool map_owned_fd(int fd, size_t exact_size);
    void close();

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    bool valid() const { return data_ != nullptr; }
    int fd() const { return fd_; }

private:
    int fd_ = -1;
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
};

} // namespace Laplace
