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
    if (!map_owned_fd(fd, static_cast<size_t>(st.st_size))) {
        fprintf(stderr, "mmap: mmap failed for %s\n", path);
        return false;
    }
    return true;
}

bool MappedFile::map_owned_fd(int fd, size_t exact_size) {
    if (fd < 0) return false;
    struct stat st;
    if (exact_size == 0 || fstat(fd, &st) != 0 || st.st_size < 0 ||
        static_cast<uint64_t>(st.st_size) != exact_size) {
        ::close(fd);
        return false;
    }
    void* p = ::mmap(nullptr, exact_size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        ::close(fd);
        return false;
    }
    close();
    fd_ = fd;
    data_ = static_cast<const uint8_t*>(p);
    size_ = exact_size;
    return true;
}

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
}

} // namespace Laplace
