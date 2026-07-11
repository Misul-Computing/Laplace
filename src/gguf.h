// gguf.h - GGUF v3 parser
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "mmap.h"
#include "tensor.h"

namespace Laplace {

// GGUF metadata value types (subset we actually need)
using MetaArrayU32  = std::vector<uint32_t>;
using MetaArrayI32  = std::vector<int32_t>;
using MetaArrayU64  = std::vector<uint64_t>;
using MetaArrayI64  = std::vector<int64_t>;
using MetaArrayF32  = std::vector<float>;
using MetaArrayF64  = std::vector<double>;
using MetaArrayStr  = std::vector<std::string>;

using MetaValue = std::variant<
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, uint64_t, int64_t,
    float, double, bool, std::string,
    MetaArrayU32, MetaArrayI32, MetaArrayU64, MetaArrayI64,
    MetaArrayF32, MetaArrayF64, MetaArrayStr
>;

struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims = 0;
    uint64_t dims[4] = {0, 0, 0, 0};
    GGMLType type = GGMLType::F32;
    uint64_t offset = 0;  // relative to start of tensor data section
};

class GGUFContext {
public:
    GGUFContext() = default;
    ~GGUFContext() { close(); }

    GGUFContext(const GGUFContext&) = delete;
    GGUFContext& operator=(const GGUFContext&) = delete;

    bool open(const char* path);
    void close();

    const std::map<std::string, MetaValue>& metadata() const { return metadata_; }
    const std::vector<GGUFTensorInfo>& tensor_infos() const { return tensor_infos_; }
    const std::vector<Tensor>& tensors() const { return tensors_; }

    const Tensor* find_tensor(const std::string& name) const;
    const Tensor* find_tensor(const char* name) const;

    const uint8_t* file_data() const { return file_.data(); }
    size_t file_size() const { return file_.size(); }
    int fd() const { return file_.fd(); }

    uint64_t data_section_offset() const { return data_section_offset_; }
    uint64_t alignment() const { return alignment_; }
    uint32_t version() const { return version_; }
    const std::string& path() const { return path_; }

private:
    MappedFile file_;
    std::map<std::string, MetaValue> metadata_;
    std::vector<GGUFTensorInfo> tensor_infos_;
    std::vector<Tensor> tensors_;
    uint64_t data_section_offset_ = 0;
    uint64_t alignment_ = 32;
    uint32_t version_ = 0;
    std::string path_;
};

// Convenience accessors (tolerate multiple integer widths)
int64_t         meta_int  (const std::map<std::string, MetaValue>& m, const char* key, int64_t fallback = 0);
double          meta_float(const std::map<std::string, MetaValue>& m, const char* key, double fallback = 0.0);
const std::string* meta_str(const std::map<std::string, MetaValue>& m, const char* key);
const uint32_t* meta_uint32(const std::map<std::string, MetaValue>& m, const char* key);
const uint64_t* meta_uint64(const std::map<std::string, MetaValue>& m, const char* key);

// Direct variant access (returns nullptr if the key is missing or has a
// different variant type).
template <typename T>
const T* meta_as(const std::map<std::string, MetaValue>& m, const char* key) {
    auto it = m.find(key);
    if (it == m.end()) return nullptr;
    return std::get_if<T>(&it->second);
}

} // namespace Laplace
