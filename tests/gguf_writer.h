// gguf_writer.h - minimal GGUF v3 serializer for building synthetic test files.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace gguf_writer {

enum ValueType : uint32_t {
    VT_UINT8 = 0, VT_INT8 = 1, VT_UINT16 = 2, VT_INT16 = 3,
    VT_UINT32 = 4, VT_INT32 = 5, VT_FLOAT32 = 6, VT_BOOL = 7,
    VT_STRING = 8, VT_ARRAY = 9, VT_UINT64 = 10, VT_INT64 = 11, VT_FLOAT64 = 12,
};

struct TensorDecl {
    std::string name;
    std::vector<uint64_t> dims;   // dims[0] innermost, per GGUF convention
    uint32_t type = 0;            // GGMLType as u32
    std::vector<uint8_t> data;
};

class Writer {
public:
    void kv_u32(const std::string& key, uint32_t v)  { begin_kv(key, VT_UINT32); put(v); }
    void kv_i32(const std::string& key, int32_t v)   { begin_kv(key, VT_INT32);  put(v); }
    void kv_f32(const std::string& key, float v)     { begin_kv(key, VT_FLOAT32);put(v); }
    void kv_bool(const std::string& key, bool v)     { begin_kv(key, VT_BOOL);   put<uint8_t>(v ? 1 : 0); }
    void kv_str(const std::string& key, const std::string& v) { begin_kv(key, VT_STRING); put_str(v); }

    void kv_arr_str(const std::string& key, const std::vector<std::string>& v) {
        begin_kv(key, VT_ARRAY);
        put<uint32_t>(VT_STRING);
        put<uint64_t>(v.size());
        for (const auto& s : v) put_str(s);
    }
    void kv_arr_u8(const std::string& key, const std::vector<uint8_t>& v) {
        begin_kv(key, VT_ARRAY);
        put<uint32_t>(VT_UINT8);
        put<uint64_t>(v.size());
        for (uint8_t b : v) put(b);
    }
    void kv_arr_i16(const std::string& key, const std::vector<int16_t>& v) {
        begin_kv(key, VT_ARRAY);
        put<uint32_t>(VT_INT16);
        put<uint64_t>(v.size());
        for (int16_t b : v) put(b);
    }
    void kv_arr_bool(const std::string& key, const std::vector<bool>& v) {
        begin_kv(key, VT_ARRAY);
        put<uint32_t>(VT_BOOL);
        put<uint64_t>(v.size());
        for (bool b : v) put<uint8_t>(b ? 1 : 0);
    }
    void kv_arr_i32(const std::string& key, const std::vector<int32_t>& v) {
        begin_kv(key, VT_ARRAY);
        put<uint32_t>(VT_INT32);
        put<uint64_t>(v.size());
        for (int32_t b : v) put(b);
    }

    void add_tensor(TensorDecl t) { tensors_.push_back(std::move(t)); }

    // If truncate_data_bytes > 0, that many bytes are cut from the end of the
    // tensor data section (to simulate a truncated file).
    bool write_file(const std::string& path, size_t truncate_data_bytes = 0) {
        std::string out;
        append(out, static_cast<uint32_t>(0x46554747u));  // "GGUF"
        append(out, static_cast<uint32_t>(3));            // version
        append(out, static_cast<uint64_t>(tensors_.size()));
        append(out, static_cast<uint64_t>(kv_count_));
        out += meta_;

        // Tensor infos with packed sequential offsets (32-byte aligned).
        const uint64_t alignment = 32;
        uint64_t offset = 0;
        std::string data;
        for (const auto& t : tensors_) {
            append_str(out, t.name);
            append(out, static_cast<uint32_t>(t.dims.size()));
            for (uint64_t d : t.dims) append(out, d);
            append(out, t.type);
            append(out, offset);
            data.append(reinterpret_cast<const char*>(t.data.data()), t.data.size());
            uint64_t padded = (t.data.size() + alignment - 1) / alignment * alignment;
            data.append(padded - t.data.size(), '\0');
            offset += padded;
        }

        // Align the data section start.
        size_t mis = out.size() % alignment;
        if (mis) out.append(alignment - mis, '\0');
        out += data;
        if (truncate_data_bytes > 0 && truncate_data_bytes < out.size()) {
            out.resize(out.size() - truncate_data_bytes);
        }

        FILE* f = fopen(path.c_str(), "wb");
        if (!f) return false;
        size_t n = fwrite(out.data(), 1, out.size(), f);
        fclose(f);
        return n == out.size();
    }

private:
    template <typename T>
    static void append(std::string& s, T v) {
        s.append(reinterpret_cast<const char*>(&v), sizeof(T));
    }
    static void append_str(std::string& s, const std::string& v) {
        append(s, static_cast<uint64_t>(v.size()));
        s += v;
    }

    void begin_kv(const std::string& key, ValueType vt) {
        append_str(meta_, key);
        append(meta_, static_cast<uint32_t>(vt));
        kv_count_++;
    }
    template <typename T>
    void put(T v) { append(meta_, v); }
    void put_str(const std::string& v) { append_str(meta_, v); }

    std::string meta_;
    uint64_t kv_count_ = 0;
    std::vector<TensorDecl> tensors_;
};

} // namespace gguf_writer
