#include "mmap.h"

#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace Laplace {

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
    void* p = ::mmap(nullptr, static_cast<size_t>(st.st_size),
                     PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        ::close(fd);
        fprintf(stderr, "mmap: mmap failed for %s\n", path);
        return false;
    }
    file_handle_ = reinterpret_cast<void*>(static_cast<intptr_t>(fd));
    data_ = static_cast<const uint8_t*>(p);
    size_ = static_cast<size_t>(st.st_size);
    return true;
}

void MappedFile::close() {
    if (data_) {
        ::munmap(const_cast<uint8_t*>(data_), size_);
        data_ = nullptr;
    }
    if (file_handle_) {
        ::close(static_cast<int>(reinterpret_cast<intptr_t>(file_handle_)));
        file_handle_ = nullptr;
    }
    size_ = 0;
}

} // namespace Laplace
