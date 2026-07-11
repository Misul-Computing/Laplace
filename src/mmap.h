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
    void close();

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    bool valid() const { return data_ != nullptr; }
    int fd() const { return static_cast<int>(reinterpret_cast<intptr_t>(file_handle_)); }

private:
    void* file_handle_ = nullptr;     // int fd stored as void*
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
};

} // namespace Laplace
