#include "safetensors.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace Laplace {

// ---------------------------------------------------------------------------
// Minimal JSON parser (objects, arrays, strings, integers).
// SafeTensors headers only use these four kinds; true/false/null are
// accepted for robustness but never expected in practice.
// ---------------------------------------------------------------------------

namespace {

struct JSONValue {
    enum class Type { Null, Object, Array, String, Int };
    Type type = Type::Null;
    std::string str;
    int64_t num = 0;
    std::vector<std::pair<std::string, JSONValue>> obj;
    std::vector<JSONValue> arr;
};

class JSONParser {
public:
    JSONParser(const char* data, size_t len)
        : data_(data), len_(len), pos_(0) {}

    bool parse(JSONValue& out) {
        skip_ws();
        if (!parse_value(out)) return false;
        // Trailing whitespace or padding is allowed (some files pad the
        // header to an alignment boundary with spaces or NUL bytes).
        return true;
    }

private:
    bool parse_value(JSONValue& out) {
        skip_ws();
        if (pos_ >= len_) return false;
        char c = data_[pos_];
        if (c == '{') return parse_object(out);
        if (c == '[') return parse_array(out);
        if (c == '"') {
            out.type = JSONValue::Type::String;
            return parse_string(out.str);
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            out.type = JSONValue::Type::Int;
            return parse_number(out.num);
        }
        if (match_lit("true"))  { out.type = JSONValue::Type::Int; out.num = 1; return true; }
        if (match_lit("false")) { out.type = JSONValue::Type::Int; out.num = 0; return true; }
        if (match_lit("null"))  { out.type = JSONValue::Type::Null; return true; }
        return false;
    }

    bool parse_object(JSONValue& out) {
        out.type = JSONValue::Type::Object;
        pos_++;  // '{'
        skip_ws();
        if (pos_ < len_ && data_[pos_] == '}') { pos_++; return true; }
        while (true) {
            skip_ws();
            std::string key;
            if (!parse_string(key)) return false;
            skip_ws();
            if (pos_ >= len_ || data_[pos_] != ':') return false;
            pos_++;  // ':'
            JSONValue val;
            if (!parse_value(val)) return false;
            out.obj.emplace_back(std::move(key), std::move(val));
            skip_ws();
            if (pos_ >= len_) return false;
            if (data_[pos_] == ',') { pos_++; continue; }
            if (data_[pos_] == '}') { pos_++; return true; }
            return false;
        }
    }

    bool parse_array(JSONValue& out) {
        out.type = JSONValue::Type::Array;
        pos_++;  // '['
        skip_ws();
        if (pos_ < len_ && data_[pos_] == ']') { pos_++; return true; }
        while (true) {
            JSONValue val;
            if (!parse_value(val)) return false;
            out.arr.push_back(std::move(val));
            skip_ws();
            if (pos_ >= len_) return false;
            if (data_[pos_] == ',') { pos_++; continue; }
            if (data_[pos_] == ']') { pos_++; return true; }
            return false;
        }
    }

    bool parse_string(std::string& out) {
        if (pos_ >= len_ || data_[pos_] != '"') return false;
        pos_++;  // opening '"'
        out.clear();
        while (pos_ < len_) {
            char c = data_[pos_++];
            if (c == '"') return true;
            if (c == '\\') {
                if (pos_ >= len_) return false;
                char esc = data_[pos_++];
                switch (esc) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'n':  out += '\n'; break;
                    case 't':  out += '\t'; break;
                    case 'r':  out += '\r'; break;
                    case 'b':  out += '\b'; break;
                    case 'f':  out += '\f'; break;
                    case 'u':
                        // Skip 4 hex digits. SafeTensors keys are ASCII so
                        // this path is never exercised in practice.
                        if (pos_ + 4 > len_) return false;
                        pos_ += 4;
                        break;
                    default: out += esc; break;
                }
            } else {
                out += c;
            }
        }
        return false;  // unterminated
    }

    bool parse_number(int64_t& out) {
        size_t start = pos_;
        if (pos_ < len_ && data_[pos_] == '-') pos_++;
        size_t digits_start = pos_;
        while (pos_ < len_ && data_[pos_] >= '0' && data_[pos_] <= '9') pos_++;
        if (pos_ == digits_start) return false;  // no digits
        std::string s(data_ + start, pos_ - start);
        out = std::strtoll(s.c_str(), nullptr, 10);
        return true;
    }

    void skip_ws() {
        while (pos_ < len_) {
            char c = data_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') pos_++;
            else break;
        }
    }

    bool match_lit(const char* lit) {
        size_t n = std::strlen(lit);
        if (pos_ + n > len_) return false;
        if (std::strncmp(data_ + pos_, lit, n) != 0) return false;
        pos_ += n;
        return true;
    }

    const char* data_;
    size_t len_;
    size_t pos_;
};

// Look up a key in a JSON object. Returns nullptr if not found.
const JSONValue* obj_get(const JSONValue& obj, const std::string& key) {
    if (obj.type != JSONValue::Type::Object) return nullptr;
    for (const auto& [k, v] : obj.obj) {
        if (k == key) return &v;
    }
    return nullptr;
}

bool dtype_to_ggml(const std::string& dtype, GGMLType& out) {
    if (dtype == "F32")  { out = GGMLType::F32;  return true; }
    if (dtype == "F16")  { out = GGMLType::F16;  return true; }
    if (dtype == "BF16") { out = GGMLType::BF16; return true; }
    if (dtype == "I8")   { out = GGMLType::I8;   return true; }
    if (dtype == "I32")  { out = GGMLType::I32;  return true; }
    if (dtype == "I64")  { out = GGMLType::I64;  return true; }
    if (dtype == "U8")   { out = GGMLType::U8;   return true; }
    if (dtype == "U32")  { out = GGMLType::U32;  return true; }
    if (dtype == "BOOL") { out = GGMLType::BOOL; return true; }
    return false;
}

bool read_text_file(const std::string& path, std::string& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    if (sz < 0) { std::fclose(f); return false; }
    std::fseek(f, 0, SEEK_SET);
    out.resize(static_cast<size_t>(sz));
    size_t n = std::fread(out.data(), 1, static_cast<size_t>(sz), f);
    std::fclose(f);
    return n == static_cast<size_t>(sz);
}

std::string dir_of(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) return "";
    return path.substr(0, pos + 1);
}

} // namespace

// ---------------------------------------------------------------------------
// SafeTensorsContext
// ---------------------------------------------------------------------------

bool SafeTensorsContext::open_shard(const std::string& full_path) {
    Shard shard;
    shard.file = std::make_unique<MappedFile>();
    if (!shard.file->open(full_path.c_str())) {
        fprintf(stderr, "safetensors: cannot mmap %s\n", full_path.c_str());
        return false;
    }

    if (shard.file->size() < 8) {
        fprintf(stderr, "safetensors: file too small (%zu bytes): %s\n",
                shard.file->size(), full_path.c_str());
        return false;
    }

    uint64_t header_len = 0;
    std::memcpy(&header_len, shard.file->data(), 8);

    if (8 + header_len > shard.file->size()) {
        fprintf(stderr, "safetensors: header extends past EOF: %s\n",
                full_path.c_str());
        return false;
    }

    shard.data_base = shard.file->data() + 8 + header_len;

    const char* json = reinterpret_cast<const char*>(shard.file->data() + 8);
    if (!parse_json_header(json, static_cast<size_t>(header_len), shard)) {
        fprintf(stderr, "safetensors: failed to parse header: %s\n",
                full_path.c_str());
        return false;
    }

    shards_.push_back(std::move(shard));
    return true;
}

bool SafeTensorsContext::parse_json_header(const char* json, size_t json_len,
                                           Shard& shard) {
    JSONParser parser(json, json_len);
    JSONValue root;
    if (!parser.parse(root)) {
        fprintf(stderr, "safetensors: JSON parse error\n");
        return false;
    }
    if (root.type != JSONValue::Type::Object) {
        fprintf(stderr, "safetensors: JSON root is not an object\n");
        return false;
    }

    for (const auto& [key, val] : root.obj) {
        if (key == "__metadata__") {
            if (val.type == JSONValue::Type::Object) {
                for (const auto& [mk, mv] : val.obj) {
                    if (mv.type == JSONValue::Type::String)
                        metadata_[mk] = mv.str;
                    else if (mv.type == JSONValue::Type::Int)
                        metadata_[mk] = std::to_string(mv.num);
                }
            }
            continue;
        }

        // Tensor descriptor: { "dtype": "...", "shape": [...],
        //                       "data_offsets": [start, end] }
        if (val.type != JSONValue::Type::Object) {
            fprintf(stderr, "safetensors: tensor '%s' value is not an object\n",
                    key.c_str());
            return false;
        }

        const JSONValue* j_dtype = obj_get(val, "dtype");
        const JSONValue* j_shape = obj_get(val, "shape");
        const JSONValue* j_offsets = obj_get(val, "data_offsets");

        if (!j_dtype || j_dtype->type != JSONValue::Type::String) {
            fprintf(stderr, "safetensors: tensor '%s' missing dtype\n",
                    key.c_str());
            return false;
        }
        if (!j_shape || j_shape->type != JSONValue::Type::Array) {
            fprintf(stderr, "safetensors: tensor '%s' missing shape\n",
                    key.c_str());
            return false;
        }
        if (!j_offsets || j_offsets->type != JSONValue::Type::Array
            || j_offsets->arr.size() != 2) {
            fprintf(stderr, "safetensors: tensor '%s' missing data_offsets\n",
                    key.c_str());
            return false;
        }

        GGMLType gtype;
        if (!dtype_to_ggml(j_dtype->str, gtype)) {
            fprintf(stderr, "safetensors: tensor '%s' unknown dtype '%s'\n",
                    key.c_str(), j_dtype->str.c_str());
            return false;
        }

        if (j_shape->arr.size() > 4) {
            fprintf(stderr, "safetensors: tensor '%s' has %zu dims (> 4)\n",
                    key.c_str(), j_shape->arr.size());
            return false;
        }

        uint64_t data_start = static_cast<uint64_t>(j_offsets->arr[0].num);
        uint64_t data_end   = static_cast<uint64_t>(j_offsets->arr[1].num);
        if (data_end < data_start) {
            fprintf(stderr, "safetensors: tensor '%s' bad data_offsets\n",
                    key.c_str());
            return false;
        }

        // Bounds-check against file size. data_base points to the start
        // of the data section (file->data() + 8 + header_len), so the
        // absolute offset is (data_base - file->data()) + data_offset.
        size_t section_off = static_cast<size_t>(shard.data_base - shard.file->data());
        uint64_t abs_end = static_cast<uint64_t>(section_off) + data_end;
        if (abs_end > shard.file->size()) {
            fprintf(stderr, "safetensors: tensor '%s' data ends past EOF (%zu)\n",
                    key.c_str(), shard.file->size());
            return false;
        }

        Tensor t;
        t.name = key;
        t.type = gtype;
        t.n_dims = static_cast<uint32_t>(j_shape->arr.size());
        // SafeTensors shape is row-major (outermost first). GGML dims[0]
        // is innermost, so reverse the array.
        for (size_t i = 0; i < j_shape->arr.size(); i++) {
            t.dims[i] = static_cast<uint64_t>(
                j_shape->arr[j_shape->arr.size() - 1 - i].num);
        }
        t.data = shard.data_base + data_start;

        tensor_index_[key] = tensors_.size();
        tensors_.push_back(std::move(t));
    }

    return true;
}

bool SafeTensorsContext::open(const char* path) {
    close();
    path_ = path;
    base_dir_ = dir_of(path_);
    sharded_ = false;
    return open_shard(path);
}

bool SafeTensorsContext::open_sharded(const char* index_path) {
    close();
    path_ = index_path;
    base_dir_ = dir_of(index_path);
    sharded_ = true;

    std::string content;
    if (!read_text_file(index_path, content)) {
        fprintf(stderr, "safetensors: cannot read index: %s\n", index_path);
        return false;
    }

    JSONParser parser(content.data(), content.size());
    JSONValue root;
    if (!parser.parse(root) || root.type != JSONValue::Type::Object) {
        fprintf(stderr, "safetensors: failed to parse index JSON\n");
        return false;
    }

    const JSONValue* wm = obj_get(root, "weight_map");
    if (!wm || wm->type != JSONValue::Type::Object) {
        fprintf(stderr, "safetensors: index missing weight_map\n");
        return false;
    }

    // Collect unique shard filenames in sorted order for deterministic
    // shard opening.
    std::vector<std::string> shard_files;
    for (const auto& [tensor_name, shard_val] : wm->obj) {
        if (shard_val.type != JSONValue::Type::String) continue;
        shard_files.push_back(shard_val.str);
    }
    std::sort(shard_files.begin(), shard_files.end());
    shard_files.erase(std::unique(shard_files.begin(), shard_files.end()),
                      shard_files.end());

    for (const auto& fname : shard_files) {
        std::string full = base_dir_ + fname;
        if (!open_shard(full)) {
            fprintf(stderr, "safetensors: failed to open shard %s\n",
                    full.c_str());
            return false;
        }
    }

    return true;
}

void SafeTensorsContext::close() {
    shards_.clear();
    tensors_.clear();
    tensor_index_.clear();
    metadata_.clear();
    path_.clear();
    base_dir_.clear();
    sharded_ = false;
}

const Tensor* SafeTensorsContext::find_tensor(const std::string& name) const {
    auto it = tensor_index_.find(name);
    if (it == tensor_index_.end()) return nullptr;
    return &tensors_[it->second];
}

const Tensor* SafeTensorsContext::find_tensor(const char* name) const {
    if (!name) return nullptr;
    return find_tensor(std::string(name));
}

} // namespace Laplace
