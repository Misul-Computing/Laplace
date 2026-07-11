// safetensors.h - SafeTensors parser for MLX-format and HuggingFace-format models
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

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
//
// Sharded models use a <model>.safetensors.index.json whose "weight_map"
// maps each tensor name to the shard file that contains it.

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
        const uint8_t* data_base = nullptr;  // file->data() + 8 + header_len
    };

    bool open_shard(const std::string& full_path);
    bool parse_json_header(const char* json, size_t json_len, Shard& shard);

    std::vector<Shard> shards_;
    std::vector<Tensor> tensors_;
    std::map<std::string, size_t> tensor_index_;  // name -> index in tensors_
    std::map<std::string, std::string> metadata_;
    std::string path_;
    std::string base_dir_;  // directory for resolving shard filenames
    bool sharded_ = false;
};

} // namespace Laplace
