// safetensors.h - SafeTensors parser for MLX-format and HuggingFace-format models
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <variant>

#include "compatibility_report.h"
#include "mmap.h"
#include "tensor.h"

namespace Laplace {

// SafeTensors on-disk layout:
//   [0..7]    uint64 little-endian header length N
//   [8..8+N)  JSON header (UTF-8, starts with '{')
//   [8+N..)   tensor data buffer (contiguous)
//
// The JSON header maps tensor names to descriptors:
//   "tensor_name": { "dtype": "F16", "shape": [4096,1024],
//                     "data_offsets": [0, 8388608] }
// The optional "__metadata__" key holds a flat string-to-string map.

enum class SafeTensorsDtype : uint8_t {
    BOOL,
    F4,
    F6_E2M3,
    F6_E3M2,
    U8,
    I8,
    F8_E5M2,
    F8_E4M3,
    F8_E8M0,
    F8_E4M3FNUZ,
    F8_E5M2FNUZ,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    C64,
    F64,
    I64,
    U64,
};

enum class SafeTensorsError : uint8_t {
    HeaderTooSmall,
    HeaderTooLarge,
    HeaderLength,
    HeaderUtf8,
    HeaderStart,
    HeaderJson,
    HeaderPadding,
    HeaderObject,
    DuplicateKey,
    MetadataNotObject,
    MetadataNotString,
    TensorObject,
    TensorField,
    Dtype,
    InvalidInteger,
    DataOffsets,
    TensorByteSize,
    IncompleteBuffer,
    ArithmeticOverflow,
};

struct SafeTensorsParseError {
    SafeTensorsError code = SafeTensorsError::HeaderJson;
    CompatibilityReport report;
};

struct SafeTensorsTensor {
    std::string name;
    SafeTensorsDtype dtype = SafeTensorsDtype::U8;
    std::vector<uint64_t> shape;
    uint64_t data_offset = 0;
    uint64_t data_length = 0;
    std::span<const uint8_t> data;
};

class SafeTensorsFile {
public:
    SafeTensorsFile() = default;

    uint64_t header_length() const noexcept { return header_length_; }
    std::span<const uint8_t> bytes() const noexcept { return bytes_; }
    const std::vector<SafeTensorsTensor>& tensors() const noexcept { return tensors_; }
    const std::map<std::string, std::string>& metadata() const noexcept { return metadata_; }

private:
    friend std::variant<SafeTensorsFile, SafeTensorsParseError>
        parse_safetensors(std::span<const uint8_t> bytes);

    uint64_t header_length_ = 0;
    std::span<const uint8_t> bytes_;
    std::vector<SafeTensorsTensor> tensors_;
    std::map<std::string, std::string> metadata_;
};

using SafeTensorsParseResult = std::variant<SafeTensorsFile, SafeTensorsParseError>;

// Parses one complete SafeTensors byte stream. Returned tensor spans borrow
// from bytes, so the caller must retain the source storage while using them.
SafeTensorsParseResult parse_safetensors(std::span<const uint8_t> bytes);

class SafeTensorsContext {
public:
    SafeTensorsContext() = default;
    ~SafeTensorsContext() { close(); }

    SafeTensorsContext(const SafeTensorsContext&) = delete;
    SafeTensorsContext& operator=(const SafeTensorsContext&) = delete;

    // Open a single .safetensors file.
    bool open(const char* path);

    // Open a sharded model via a .safetensors.index.json file.
    bool open_sharded(const char* index_path);

    void close();

    const Tensor* find_tensor(const std::string& name) const;
    const Tensor* find_tensor(const char* name) const;

    const std::vector<Tensor>& tensors() const { return tensors_; }
    const std::map<std::string, std::string>& metadata() const { return metadata_; }

    size_t num_shards() const { return shards_.size(); }
    bool is_sharded() const { return sharded_; }
    const std::string& path() const { return path_; }

private:
    struct Shard {
        std::unique_ptr<MappedFile> file;
        SafeTensorsFile parsed;
    };

    bool open_shard(const std::string& full_path);

    std::vector<Shard> shards_;
    std::vector<Tensor> tensors_;
    std::map<std::string, size_t> tensor_index_;  // name -> index in tensors_
    std::map<std::string, std::string> metadata_;
    std::string path_;
    std::string base_dir_;  // directory for resolving shard filenames
    bool sharded_ = false;
};

} // namespace Laplace
