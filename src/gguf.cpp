#include "gguf.h"

#include <cstdio>
#include <cstring>

namespace Laplace {

namespace {

constexpr uint32_t GGUF_MAGIC = 0x46554747u;  // "GGUF" little-endian

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
        return skip(alignment - mis);
    }

private:
    const uint8_t* base_;
    const uint8_t* p_;
    const uint8_t* end_;
};

bool read_string(Reader& r, std::string& out) {
    uint64_t len = 0;
    if (!r.read(len)) return false;
    if (r.remaining() < len) return false;
    out.assign(reinterpret_cast<const char*>(r.ptr()), static_cast<size_t>(len));
    r.skip(static_cast<size_t>(len));
    return true;
}

bool read_kv_value(Reader& r, GGUFValueType vt, MetaValue& out);

bool read_kv_array(Reader& r, MetaValue& out) {
    uint32_t et_raw = 0;
    uint64_t len = 0;
    if (!r.read(et_raw)) return false;
    if (!r.read(len)) return false;
    auto et = static_cast<GGUFValueType>(et_raw);

    switch (et) {
        // Small integer and bool arrays are widened into the 32-bit array
        // variants so files containing them still parse.
        case GGUFValueType::UINT8:
        case GGUFValueType::BOOL: {
            MetaArrayU32 arr(len);
            for (uint64_t i = 0; i < len; i++) {
                uint8_t v;
                if (!r.read(v)) return false;
                arr[i] = v;
            }
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::INT8: {
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
            MetaArrayU32 arr(len);
            if (len && !r.read_bytes(arr.data(), len * sizeof(uint32_t))) return false;
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::INT32: {
            MetaArrayI32 arr(len);
            if (len && !r.read_bytes(arr.data(), len * sizeof(int32_t))) return false;
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::FLOAT32: {
            MetaArrayF32 arr(len);
            if (len && !r.read_bytes(arr.data(), len * sizeof(float))) return false;
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::UINT64: {
            MetaArrayU64 arr(len);
            if (len && !r.read_bytes(arr.data(), len * sizeof(uint64_t))) return false;
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::INT64: {
            MetaArrayI64 arr(len);
            if (len && !r.read_bytes(arr.data(), len * sizeof(int64_t))) return false;
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::FLOAT64: {
            MetaArrayF64 arr(len);
            if (len && !r.read_bytes(arr.data(), len * sizeof(double))) return false;
            out = std::move(arr);
            return true;
        }
        case GGUFValueType::STRING: {
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

} // namespace

bool GGUFContext::open(const char* path) {
    close();
    if (!file_.open(path)) return false;
    path_ = path;

    Reader r(file_.data(), file_.size());

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

    metadata_.clear();
    for (uint64_t i = 0; i < kv_count; i++) {
        std::string key;
        if (!read_string(r, key)) {
            fprintf(stderr, "gguf: read past EOF on kv key #%llu\n", (unsigned long long)i);
            return false;
        }
        uint32_t vt = 0;
        if (!r.read(vt)) {
            fprintf(stderr, "gguf: read past EOF on kv value type for '%s'\n", key.c_str());
            return false;
        }
        MetaValue value;
        if (!read_kv_value(r, static_cast<GGUFValueType>(vt), value)) {
            // Non-fatal for unknown array element types: skip ahead is hard, so we abort.
            fprintf(stderr, "gguf: failed to read kv '%s' (value type %u)\n", key.c_str(), vt);
            return false;
        }
        metadata_[key] = std::move(value);
    }

    tensor_infos_.clear();
    tensor_infos_.reserve(tensor_count);
    for (uint64_t i = 0; i < tensor_count; i++) {
        GGUFTensorInfo ti;
        if (!read_string(r, ti.name)) {
            fprintf(stderr, "gguf: read past EOF on tensor name #%llu\n", (unsigned long long)i);
            return false;
        }
        if (!r.read(ti.n_dims) || ti.n_dims > 4) {
            fprintf(stderr, "gguf: bad n_dims for tensor '%s'\n", ti.name.c_str());
            return false;
        }
        for (uint32_t d = 0; d < ti.n_dims; d++) {
            if (!r.read(ti.dims[d])) {
                fprintf(stderr, "gguf: read past EOF on dims for tensor '%s'\n", ti.name.c_str());
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
        tensor_infos_.push_back(std::move(ti));
    }

    if (auto p = meta_uint32(metadata_, "general.alignment"); p) {
        alignment_ = *p;
    } else if (auto p = meta_uint64(metadata_, "general.alignment"); p) {
        alignment_ = *p;
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
        uint64_t file_off = data_section_offset_ + ti.offset;
        uint64_t need = t.nbytes();
        if (file_off > file_.size() || need > file_.size() - file_off) {
            fprintf(stderr, "gguf: tensor '%s' data [%llu, +%llu) past EOF (%zu)\n",
                    ti.name.c_str(), (unsigned long long)file_off,
                    (unsigned long long)need, file_.size());
            return false;
        }
        t.data = file_.data() + file_off;
        tensors_.push_back(std::move(t));
    }

    return true;
}

void GGUFContext::close() {
    file_.close();
    metadata_.clear();
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
