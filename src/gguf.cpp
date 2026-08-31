#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <new>

namespace Laplace {

namespace {

constexpr uint32_t GGUF_MAGIC = 0x46554747u;  // "GGUF" little-endian
constexpr uint64_t MAX_METADATA_ENTRIES = 1u << 20;
constexpr uint64_t MAX_TENSOR_ENTRIES = 1u << 20;
constexpr uint64_t MAX_ARRAY_ELEMENTS = 1u << 20;
constexpr uint64_t MAX_METADATA_STRING_BYTES = 1u << 24;

enum class GGUFValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

class Reader {
public:
    Reader(const uint8_t* p, size_t n) : base_(p), p_(p), end_(p + n) {}

    size_t pos() const { return static_cast<size_t>(p_ - base_); }
    size_t remaining() const { return static_cast<size_t>(end_ - p_); }
    const uint8_t* ptr() const { return p_; }

    template <typename T>
    bool read(T& out) {
        if (remaining() < sizeof(T)) return false;
        std::memcpy(&out, p_, sizeof(T));
        p_ += sizeof(T);
        return true;
    }

    bool read_bytes(void* dst, size_t n) {
        if (remaining() < n) return false;
        std::memcpy(dst, p_, n);
        p_ += n;
        return true;
    }

    bool skip(size_t n) {
        if (remaining() < n) return false;
        p_ += n;
        return true;
    }

    bool align_to(uint64_t alignment) {
        if (alignment == 0) return true;
        size_t mis = pos() % alignment;
        if (mis == 0) return true;
        const size_t padding = static_cast<size_t>(alignment - mis);
        if (remaining() < padding) return false;
        for (size_t index = 0; index != padding; ++index) {
            if (p_[index] != 0) return false;
        }
        p_ += padding;
        return true;
    }

private:
    const uint8_t* base_;
    const uint8_t* p_;
    const uint8_t* end_;
};

bool read_string(Reader& r, std::string& out,
                 uint64_t max_length = MAX_METADATA_STRING_BYTES) {
    uint64_t len = 0;
    if (!r.read(len)) return false;
    if (len > max_length || len > std::numeric_limits<size_t>::max()) return false;
    if (r.remaining() < len) return false;
    out.assign(reinterpret_cast<const char*>(r.ptr()), static_cast<size_t>(len));
    r.skip(static_cast<size_t>(len));
    return true;
}

bool valid_metadata_key(const std::string& key) {
    if (key.empty() || key.size() > 65535 || key.front() == '.' || key.back() == '.') return false;
    bool segment_start = true;
    for (unsigned char character : key) {
        if (character == '.') {
            if (segment_start) return false;
            segment_start = true;
            continue;
        }
        if (segment_start) {
            if (!((character >= 'a' && character <= 'z') ||
                  (character >= '0' && character <= '9'))) return false;
            segment_start = false;
        } else if (!((character >= 'a' && character <= 'z') ||
                     (character >= '0' && character <= '9') || character == '_')) {
            return false;
        }
    }
    return !segment_start;
}

bool valid_tensor_name(const std::string& name) {
    if (name.empty() || name.size() > 64) return false;
    for (size_t index = 0; index < name.size();) {
        const uint8_t first = static_cast<uint8_t>(name[index]);
        if (first == 0) return false;
        size_t length = 0;
        uint32_t codepoint = 0;
        if (first <= 0x7f) {
            length = 1;
            codepoint = first;
        } else if (first >= 0xc2 && first <= 0xdf) {
            length = 2;
            codepoint = first & 0x1f;
        } else if (first >= 0xe0 && first <= 0xef) {
            length = 3;
            codepoint = first & 0x0f;
        } else if (first >= 0xf0 && first <= 0xf4) {
            length = 4;
            codepoint = first & 0x07;
        } else {
            return false;
        }
        if (index + length > name.size()) return false;
        for (size_t offset = 1; offset < length; ++offset) {
            const uint8_t continuation = static_cast<uint8_t>(name[index + offset]);
            if ((continuation & 0xc0) != 0x80) return false;
            codepoint = (codepoint << 6) | (continuation & 0x3f);
        }
        if ((length == 2 && codepoint < 0x80) ||
            (length == 3 && codepoint < 0x800) ||
            (length == 4 && codepoint < 0x10000) ||
            codepoint > 0x10ffff || (codepoint >= 0xd800 && codepoint <= 0xdfff)) return false;
        index += length;
    }
    return true;
}

bool array_storage_fits(const Reader& r, uint64_t count, size_t element_size) {
    if (count > std::numeric_limits<size_t>::max()) return false;
    return element_size != 0 && count <= r.remaining() / element_size;
}

bool read_kv_value(Reader& r, GGUFValueType vt, MetaValue& out);

bool read_kv_array(Reader& r, MetaValue& out) {
    uint32_t et_raw = 0;
    uint64_t len = 0;
    if (!r.read(et_raw)) return false;
    if (!r.read(len)) return false;
    if (len > MAX_ARRAY_ELEMENTS) return false;
    auto et = static_cast<GGUFValueType>(et_raw);

    switch (et) {
        // Small integer and bool arrays are widened into the 32-bit array
        // variants so files containing them still parse.
        case GGUFValueType::UINT8:
        case GGUFValueType::BOOL: {
            if (!array_storage_fits(r, len, sizeof(uint8_t))) return false;
            MetaArrayU32 arr(len);
            for (uint64_t i = 0; i < len; i++) {
                uint8_t v;
                if (!r.read(v)) return false;
                if (et == GGUFValueType::BOOL && v > 1) return false;
                arr[i] = v;
            }
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::INT8: {
            if (!array_storage_fits(r, len, sizeof(int8_t))) return false;
            MetaArrayI32 arr(len);
            for (uint64_t i = 0; i < len; i++) {
                int8_t v;
                if (!r.read(v)) return false;
                arr[i] = v;
            }
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::UINT16: {
            if (!array_storage_fits(r, len, sizeof(uint16_t))) return false;
            MetaArrayU32 arr(len);
            for (uint64_t i = 0; i < len; i++) {
                uint16_t v;
                if (!r.read(v)) return false;
                arr[i] = v;
            }
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::INT16: {
            if (!array_storage_fits(r, len, sizeof(int16_t))) return false;
            MetaArrayI32 arr(len);
            for (uint64_t i = 0; i < len; i++) {
                int16_t v;
                if (!r.read(v)) return false;
                arr[i] = v;
            }
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::UINT32: {
            if (!array_storage_fits(r, len, sizeof(uint32_t))) return false;
            MetaArrayU32 arr(len);
            if (len && !r.read_bytes(arr.data(), static_cast<size_t>(len) * sizeof(uint32_t))) return false;
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::INT32: {
            if (!array_storage_fits(r, len, sizeof(int32_t))) return false;
            MetaArrayI32 arr(len);
            if (len && !r.read_bytes(arr.data(), static_cast<size_t>(len) * sizeof(int32_t))) return false;
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::FLOAT32: {
            if (!array_storage_fits(r, len, sizeof(float))) return false;
            MetaArrayF32 arr(len);
            if (len && !r.read_bytes(arr.data(), static_cast<size_t>(len) * sizeof(float))) return false;
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::UINT64: {
            if (!array_storage_fits(r, len, sizeof(uint64_t))) return false;
            MetaArrayU64 arr(len);
            if (len && !r.read_bytes(arr.data(), static_cast<size_t>(len) * sizeof(uint64_t))) return false;
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::INT64: {
            if (!array_storage_fits(r, len, sizeof(int64_t))) return false;
            MetaArrayI64 arr(len);
            if (len && !r.read_bytes(arr.data(), static_cast<size_t>(len) * sizeof(int64_t))) return false;
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::FLOAT64: {
            if (!array_storage_fits(r, len, sizeof(double))) return false;
            MetaArrayF64 arr(len);
            if (len && !r.read_bytes(arr.data(), static_cast<size_t>(len) * sizeof(double))) return false;
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::STRING: {
            if (len > std::numeric_limits<size_t>::max() || len > r.remaining() / sizeof(uint64_t)) return false;
            MetaArrayStr arr;
            arr.reserve(len);
            for (uint64_t i = 0; i < len; i++) {
                std::string s;
                if (!read_string(r, s)) return false;
                arr.push_back(std::move(s));
            }
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::ARRAY:
            // Nested arrays are not used in any LLaMA-family metadata we read.
            return false;
    }
    return false;
}

bool read_kv_value(Reader& r, GGUFValueType vt, MetaValue& out) {
    switch (vt) {
        case GGUFValueType::UINT8:   { uint8_t  v; if (!r.read(v)) return false; out = v; return true; }
        case GGUFValueType::INT8:    { int8_t   v; if (!r.read(v)) return false; out = v; return true; }
        case GGUFValueType::UINT16:  { uint16_t v; if (!r.read(v)) return false; out = v; return true; }
        case GGUFValueType::INT16:   { int16_t  v; if (!r.read(v)) return false; out = v; return true; }
        case GGUFValueType::UINT32:  { uint32_t v; if (!r.read(v)) return false; out = v; return true; }
        case GGUFValueType::INT32:   { int32_t  v; if (!r.read(v)) return false; out = v; return true; }
        case GGUFValueType::UINT64:  { uint64_t v; if (!r.read(v)) return false; out = v; return true; }
        case GGUFValueType::INT64:   { int64_t  v; if (!r.read(v)) return false; out = v; return true; }
        case GGUFValueType::FLOAT32: { float    v; if (!r.read(v)) return false; out = v; return true; }
        case GGUFValueType::FLOAT64: { double   v; if (!r.read(v)) return false; out = v; return true; }
        case GGUFValueType::BOOL: {
            uint8_t v; if (!r.read(v)) return false;
            if (v > 1) return false;
            out = (v != 0);
            return true;
        }
        case GGUFValueType::STRING: {
            std::string s;
            if (!read_string(r, s)) return false;
            out = std::move(s);
            return true;
        }
        case GGUFValueType::ARRAY:
            return read_kv_array(r, out);
    }
    return false;
}

bool checked_tensor_nbytes(const GGUFTensorInfo& info, uint64_t& bytes) {
    uint64_t elements = 1;
    for (uint32_t index = 0; index != info.n_dims; ++index) {
        if (info.dims[index] != 0 && elements > UINT64_MAX / info.dims[index]) return false;
        elements *= info.dims[index];
    }
    const size_t block_bytes = bytes_per_block(info.type);
    const int block_elements = elements_per_block(info.type);
    if (block_bytes == 0 || block_elements == 0 || elements == 0) {
        bytes = 0;
        return true;
    }
    uint64_t blocks = elements / static_cast<uint64_t>(block_elements);
    if (elements % static_cast<uint64_t>(block_elements) != 0) {
        if (blocks == UINT64_MAX) return false;
        ++blocks;
    }
    if (blocks > UINT64_MAX / block_bytes) return false;
    bytes = blocks * block_bytes;
    return true;
}

} // namespace

bool GGUFContext::open(const char* path) {
    close();
    if (!file_.open(path)) return false;
    path_ = path;

    try {
        if (!parse_bytes(file_.data(), file_.size())) {
            close();
            return false;
        }
        return true;
    } catch (const std::bad_alloc&) {
        fprintf(stderr, "gguf: resource budget exceeded while parsing '%s'\n", path_.c_str());
        close();
        return false;
    } catch (const std::exception& error) {
        fprintf(stderr, "gguf: parser exception for '%s': %s\n", path_.c_str(), error.what());
        close();
        return false;
    }
}

bool GGUFContext::parse(const PackageView& package) {
    close();
    if (package.bytes().empty()) return false;
    package_owner_ = package;
    try {
        const bool parsed = parse_bytes(package.bytes().data(), package.bytes().size());
        if (!parsed) close();
        return parsed;
    } catch (const std::bad_alloc&) {
        fprintf(stderr, "gguf: resource budget exceeded while parsing package\n");
        close();
        return false;
    } catch (const std::exception& error) {
        fprintf(stderr, "gguf: parser exception for package: %s\n", error.what());
        close();
        return false;
    }
}

bool GGUFContext::parse_borrowed(std::span<const uint8_t> bytes) {
    close();
    if (bytes.empty()) return false;
    try {
        const bool parsed = parse_bytes(bytes.data(), bytes.size());
        if (!parsed) close();
        return parsed;
    } catch (const std::bad_alloc&) {
        fprintf(stderr, "gguf: resource budget exceeded while parsing borrowed bytes\n");
        close();
        return false;
    } catch (const std::exception& error) {
        fprintf(stderr, "gguf: parser exception for borrowed bytes: %s\n", error.what());
        close();
        return false;
    }
}

bool GGUFContext::parse_bytes(const uint8_t* bytes, size_t size) {
    Reader r(bytes, size);

    uint32_t magic = 0;
    if (!r.read(magic)) {
        fprintf(stderr, "gguf: read past EOF on magic\n");
        return false;
    }
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "gguf: bad magic 0x%08x (expected 0x%08x)\n", magic, GGUF_MAGIC);
        return false;
    }

    if (!r.read(version_)) {
        fprintf(stderr, "gguf: read past EOF on version\n");
        return false;
    }
    if (version_ < 2 || version_ > 3) {
        fprintf(stderr, "gguf: unsupported version %u (need 2 or 3)\n", version_);
        return false;
    }

    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    if (!r.read(tensor_count) || !r.read(kv_count)) {
        fprintf(stderr, "gguf: read past EOF on counts\n");
        return false;
    }

    if (kv_count > MAX_METADATA_ENTRIES || tensor_count > MAX_TENSOR_ENTRIES ||
        kv_count > r.remaining() / 12 || tensor_count > size / 24) {
        fprintf(stderr, "gguf: declared record count exceeds the remaining file\n");
        return false;
    }

    metadata_.clear();
    metadata_entries_.clear();
    metadata_entries_.reserve(static_cast<size_t>(kv_count));
    for (uint64_t i = 0; i < kv_count; i++) {
        const size_t entry_offset = r.pos();
        std::string key;
        if (!read_string(r, key, 65535)) {
            fprintf(stderr, "gguf: read past EOF on kv key #%llu\n", (unsigned long long)i);
            return false;
        }
        if (!valid_metadata_key(key)) {
            fprintf(stderr, "gguf: invalid metadata key at entry #%llu\n", (unsigned long long)i);
            return false;
        }
        uint32_t vt = 0;
        if (!r.read(vt)) {
            fprintf(stderr, "gguf: read past EOF on kv value type for '%s'\n", key.c_str());
            return false;
        }
        if (metadata_.find(key) != metadata_.end()) {
            fprintf(stderr, "gguf: duplicate metadata key '%s'\n", key.c_str());
            return false;
        }
        MetaValue value;
        if (!read_kv_value(r, static_cast<GGUFValueType>(vt), value)) {
            // Non-fatal for unknown array element types: skip ahead is hard, so we abort.
            fprintf(stderr, "gguf: failed to read kv '%s' (value type %u)\n", key.c_str(), vt);
            return false;
        }
        metadata_[key] = std::move(value);
        metadata_entries_.push_back({key, metadata_.at(key), vt,
                                     entry_offset, static_cast<uint64_t>(r.pos() - entry_offset)});
    }

    tensor_infos_.clear();
    tensor_infos_.reserve(tensor_count);
    for (uint64_t i = 0; i < tensor_count; i++) {
        GGUFTensorInfo ti;
        if (!read_string(r, ti.name, 64)) {
            fprintf(stderr, "gguf: read past EOF on tensor name #%llu\n", (unsigned long long)i);
            return false;
        }
        if (!valid_tensor_name(ti.name)) {
            fprintf(stderr, "gguf: invalid tensor name at entry #%llu\n", (unsigned long long)i);
            return false;
        }
        if (!r.read(ti.n_dims) || ti.n_dims == 0 || ti.n_dims > 4) {
            fprintf(stderr, "gguf: bad n_dims for tensor '%s'\n", ti.name.c_str());
            return false;
        }
        for (uint32_t d = 0; d < ti.n_dims; d++) {
            if (!r.read(ti.dims[d])) {
                fprintf(stderr, "gguf: read past EOF on dims for tensor '%s'\n", ti.name.c_str());
                return false;
            }
            if (ti.dims[d] == 0) {
                fprintf(stderr, "gguf: zero tensor dimension for '%s'\n", ti.name.c_str());
                return false;
            }
        }
        uint32_t type_int = 0;
        if (!r.read(type_int)) {
            fprintf(stderr, "gguf: read past EOF on type for tensor '%s'\n", ti.name.c_str());
            return false;
        }
        ti.type = static_cast<GGMLType>(type_int);
        if (!r.read(ti.offset)) {
            fprintf(stderr, "gguf: read past EOF on offset for tensor '%s'\n", ti.name.c_str());
            return false;
        }
        if (std::find_if(tensor_infos_.begin(), tensor_infos_.end(), [&](const GGUFTensorInfo& prior) {
                return prior.name == ti.name;
            }) != tensor_infos_.end()) {
            fprintf(stderr, "gguf: duplicate tensor name '%s'\n", ti.name.c_str());
            return false;
        }
        tensor_infos_.push_back(std::move(ti));
    }

    if (auto entry = metadata_.find("general.alignment"); entry != metadata_.end() &&
        !std::holds_alternative<uint32_t>(entry->second) &&
        !std::holds_alternative<uint64_t>(entry->second)) {
        fprintf(stderr, "gguf: general.alignment has an invalid type\n");
        return false;
    }
    if (auto p = meta_uint32(metadata_, "general.alignment"); p) {
        alignment_ = *p;
    } else if (auto p = meta_uint64(metadata_, "general.alignment"); p) {
        alignment_ = *p;
    }

    if (alignment_ == 0 || alignment_ % 8 != 0 ||
        alignment_ > std::numeric_limits<size_t>::max()) {
        fprintf(stderr, "gguf: alignment must be a positive multiple of eight\n");
        return false;
    }
    for (const auto& info : tensor_infos_) {
        if (info.offset % alignment_ != 0) {
            fprintf(stderr, "gguf: tensor '%s' offset violates alignment\n", info.name.c_str());
            return false;
        }
    }

    if (!r.align_to(alignment_)) {
        fprintf(stderr, "gguf: failed to align to %llu\n", (unsigned long long)alignment_);
        return false;
    }
    data_section_offset_ = r.pos();

    tensors_.clear();
    tensors_.reserve(tensor_infos_.size());
    for (const auto& ti : tensor_infos_) {
        Tensor t;
        t.name   = ti.name;
        t.type   = ti.type;
        t.n_dims = ti.n_dims;
        for (uint32_t d = 0; d < ti.n_dims; d++) t.dims[d] = ti.dims[d];
        if (data_section_offset_ > UINT64_MAX - ti.offset) {
            fprintf(stderr, "gguf: tensor '%s' offset overflows uint64\n", ti.name.c_str());
            return false;
        }
        uint64_t file_off = data_section_offset_ + ti.offset;
        uint64_t need = 0;
        if (!checked_tensor_nbytes(ti, need)) {
            fprintf(stderr, "gguf: tensor '%s' size overflows uint64\n", ti.name.c_str());
            return false;
        }
        if (file_off > size || need > size - file_off) {
            fprintf(stderr, "gguf: tensor '%s' data [%llu, +%llu) past EOF (%zu)\n",
                    ti.name.c_str(), (unsigned long long)file_off,
                    (unsigned long long)need, size);
            return false;
        }
        t.data = bytes + file_off;
        tensors_.push_back(std::move(t));
    }

    return true;
}

void GGUFContext::close() {
    file_.close();
    package_owner_.reset();
    metadata_.clear();
    metadata_entries_.clear();
    tensor_infos_.clear();
    tensors_.clear();
    data_section_offset_ = 0;
    alignment_ = 32;
    version_ = 0;
    path_.clear();
}

const Tensor* GGUFContext::find_tensor(const std::string& name) const {
    for (const auto& t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

const Tensor* GGUFContext::find_tensor(const char* name) const {
    if (!name) return nullptr;
    return find_tensor(std::string(name));
}

const uint32_t* meta_uint32(const std::map<std::string, MetaValue>& m, const char* key) {
    return meta_as<uint32_t>(m, key);
}
const uint64_t* meta_uint64(const std::map<std::string, MetaValue>& m, const char* key) {
    return meta_as<uint64_t>(m, key);
}

int64_t meta_int(const std::map<std::string, MetaValue>& m, const char* key, int64_t fallback) {
    if (auto p = meta_as<uint32_t>(m, key)) return static_cast<int64_t>(*p);
    if (auto p = meta_as<int32_t>(m, key))  return static_cast<int64_t>(*p);
    if (auto p = meta_as<uint64_t>(m, key)) return static_cast<int64_t>(*p);
    if (auto p = meta_as<int64_t>(m, key))  return *p;
    if (auto p = meta_as<int16_t>(m, key))  return *p;
    if (auto p = meta_as<int8_t>(m, key))   return *p;
    if (auto p = meta_as<uint16_t>(m, key)) return *p;
    if (auto p = meta_as<uint8_t>(m, key))  return *p;
    return fallback;
}

double meta_float(const std::map<std::string, MetaValue>& m, const char* key, double fallback) {
    if (auto p = meta_as<float>(m, key))  return *p;
    if (auto p = meta_as<double>(m, key)) return *p;
    return fallback;
}

const std::string* meta_str(const std::map<std::string, MetaValue>& m, const char* key) {
    return meta_as<std::string>(m, key);
}

} // namespace Laplace
