#include "safetensors.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <utility>

namespace Laplace {
namespace {

constexpr uint64_t kMaxHeaderBytes = 100000000;

SafeTensorsParseError parse_error(SafeTensorsError code, std::string detail,
                                  uint64_t offset = UINT64_MAX) {
    CompatibilityReport report;
    report.stage = CompatibilityStage::Package;
    report.phase = CompatibilityPhase::Package;
    report.code = CompatibilityError::PACKAGE_BOUNDS_INVALID;
    report.message = "SafeTensors wire format is invalid";
    report.detail = std::move(detail);
    report.source_offset = offset;
    return {code, std::move(report)};
}

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t& out) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) return false;
    out = lhs + rhs;
    return true;
}

bool checked_mul(uint64_t lhs, uint64_t rhs, uint64_t& out) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) return false;
    out = lhs * rhs;
    return true;
}

bool is_valid_utf8(std::span<const uint8_t> bytes) {
    for (size_t i = 0; i < bytes.size();) {
        const uint8_t c = bytes[i++];
        if (c <= 0x7f) continue;

        size_t continuation_count = 0;
        uint32_t codepoint = 0;
        uint32_t minimum = 0;
        if ((c & 0xe0) == 0xc0) {
            continuation_count = 1;
            codepoint = c & 0x1f;
            minimum = 0x80;
        } else if ((c & 0xf0) == 0xe0) {
            continuation_count = 2;
            codepoint = c & 0x0f;
            minimum = 0x800;
        } else if ((c & 0xf8) == 0xf0) {
            continuation_count = 3;
            codepoint = c & 0x07;
            minimum = 0x10000;
        } else {
            return false;
        }
        if (i + continuation_count > bytes.size()) return false;
        for (size_t j = 0; j < continuation_count; ++j) {
            const uint8_t next = bytes[i++];
            if ((next & 0xc0) != 0x80) return false;
            codepoint = (codepoint << 6) | (next & 0x3f);
        }
        if (codepoint < minimum || codepoint > 0x10ffff ||
            (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
            return false;
        }
    }
    return true;
}

bool is_json_whitespace(uint8_t byte) {
    return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r';
}

struct JSONValue {
    enum class Type { Null, Boolean, Object, Array, String, Unsigned, Negative, Decimal };

    Type type = Type::Null;
    std::string string;
    uint64_t number = 0;
    std::vector<std::pair<std::string, JSONValue>> object;
    std::vector<JSONValue> array;
};

class JSONParser {
public:
    JSONParser(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    bool parse_value(JSONValue& out) {
        skip_ws();
        if (pos_ >= size_) return fail(SafeTensorsError::HeaderJson);
        switch (data_[pos_]) {
            case '{': return parse_object(out);
            case '[': return parse_array(out);
            case '"':
                out.type = JSONValue::Type::String;
                return parse_string(out.string);
            case 't':
                if (match("true")) { out.type = JSONValue::Type::Boolean; return true; }
                return fail(SafeTensorsError::HeaderJson);
            case 'f':
                if (match("false")) { out.type = JSONValue::Type::Boolean; return true; }
                return fail(SafeTensorsError::HeaderJson);
            case 'n':
                if (match("null")) { out.type = JSONValue::Type::Null; return true; }
                return fail(SafeTensorsError::HeaderJson);
            default:
                if (data_[pos_] == '-' || (data_[pos_] >= '0' && data_[pos_] <= '9')) {
                    return parse_number(out);
                }
                return fail(SafeTensorsError::HeaderJson);
        }
    }

    bool consumed() {
        skip_ws();
        return pos_ == size_;
    }

    SafeTensorsError error() const { return error_; }
    size_t position() const { return pos_; }

private:
    bool parse_object(JSONValue& out) {
        out.type = JSONValue::Type::Object;
        ++pos_;
        skip_ws();
        if (consume('}')) return true;

        std::set<std::string> keys;
        while (true) {
            skip_ws();
            std::string key;
            if (!parse_string(key)) return false;
            if (!keys.insert(key).second) return fail(SafeTensorsError::DuplicateKey);
            skip_ws();
            if (!consume(':')) return fail(SafeTensorsError::HeaderJson);
            JSONValue value;
            if (!parse_value(value)) return false;
            out.object.emplace_back(std::move(key), std::move(value));
            skip_ws();
            if (consume('}')) return true;
            if (!consume(',')) return fail(SafeTensorsError::HeaderJson);
        }
    }

    bool parse_array(JSONValue& out) {
        out.type = JSONValue::Type::Array;
        ++pos_;
        skip_ws();
        if (consume(']')) return true;
        while (true) {
            JSONValue value;
            if (!parse_value(value)) return false;
            out.array.push_back(std::move(value));
            skip_ws();
            if (consume(']')) return true;
            if (!consume(',')) return fail(SafeTensorsError::HeaderJson);
        }
    }

    static bool hex_value(uint8_t c, uint32_t& out) {
        if (c >= '0' && c <= '9') { out = c - '0'; return true; }
        if (c >= 'a' && c <= 'f') { out = c - 'a' + 10; return true; }
        if (c >= 'A' && c <= 'F') { out = c - 'A' + 10; return true; }
        return false;
    }

    bool parse_hex_codepoint(uint32_t& out) {
        if (pos_ + 4 > size_) return fail(SafeTensorsError::HeaderJson);
        out = 0;
        for (size_t i = 0; i < 4; ++i) {
            uint32_t nibble = 0;
            if (!hex_value(data_[pos_++], nibble)) return fail(SafeTensorsError::HeaderJson);
            out = (out << 4) | nibble;
        }
        return true;
    }

    static void append_utf8(std::string& out, uint32_t codepoint) {
        if (codepoint <= 0x7f) {
            out.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ff) {
            out.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else if (codepoint <= 0xffff) {
            out.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else {
            out.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
    }

    bool parse_string(std::string& out) {
        if (!consume('"')) return fail(SafeTensorsError::HeaderJson);
        out.clear();
        while (pos_ < size_) {
            const uint8_t c = data_[pos_++];
            if (c == '"') return true;
            if (c < 0x20) return fail(SafeTensorsError::HeaderJson);
            if (c != '\\') {
                out.push_back(static_cast<char>(c));
                continue;
            }
            if (pos_ == size_) return fail(SafeTensorsError::HeaderJson);
            switch (data_[pos_++]) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t codepoint = 0;
                    if (!parse_hex_codepoint(codepoint)) return false;
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                        if (pos_ + 2 > size_ || data_[pos_] != '\\' || data_[pos_ + 1] != 'u') {
                            return fail(SafeTensorsError::HeaderJson);
                        }
                        pos_ += 2;
                        uint32_t low = 0;
                        if (!parse_hex_codepoint(low) || low < 0xdc00 || low > 0xdfff) {
                            return fail(SafeTensorsError::HeaderJson);
                        }
                        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                    } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                        return fail(SafeTensorsError::HeaderJson);
                    }
                    append_utf8(out, codepoint);
                    break;
                }
                default: return fail(SafeTensorsError::HeaderJson);
            }
        }
        return fail(SafeTensorsError::HeaderJson);
    }

    bool parse_number(JSONValue& out) {
        const size_t start = pos_;
        const bool negative = consume('-');
        if (pos_ == size_) return fail(SafeTensorsError::HeaderJson);
        if (data_[pos_] == '0') {
            ++pos_;
            if (pos_ < size_ && data_[pos_] >= '0' && data_[pos_] <= '9') {
                return fail(SafeTensorsError::HeaderJson);
            }
        } else {
            if (data_[pos_] < '1' || data_[pos_] > '9') return fail(SafeTensorsError::HeaderJson);
            do { ++pos_; } while (pos_ < size_ && data_[pos_] >= '0' && data_[pos_] <= '9');
        }
        const size_t integer_end = pos_;
        bool decimal = false;
        if (pos_ < size_ && data_[pos_] == '.') {
            decimal = true;
            ++pos_;
            const size_t fraction_start = pos_;
            while (pos_ < size_ && data_[pos_] >= '0' && data_[pos_] <= '9') ++pos_;
            if (pos_ == fraction_start) return fail(SafeTensorsError::HeaderJson);
        }
        if (pos_ < size_ && (data_[pos_] == 'e' || data_[pos_] == 'E')) {
            decimal = true;
            ++pos_;
            if (pos_ < size_ && (data_[pos_] == '+' || data_[pos_] == '-')) ++pos_;
            const size_t exponent_start = pos_;
            while (pos_ < size_ && data_[pos_] >= '0' && data_[pos_] <= '9') ++pos_;
            if (pos_ == exponent_start) return fail(SafeTensorsError::HeaderJson);
        }
        if (decimal) {
            out.type = JSONValue::Type::Decimal;
            return true;
        }
        uint64_t value = 0;
        const size_t digit_start = negative ? start + 1 : start;
        for (size_t i = digit_start; i < integer_end; ++i) {
            const uint64_t digit = data_[i] - '0';
            if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
                return fail(SafeTensorsError::HeaderJson);
            }
            value = value * 10 + digit;
        }
        out.type = negative ? JSONValue::Type::Negative : JSONValue::Type::Unsigned;
        out.number = value;
        return true;
    }

    bool consume(uint8_t expected) {
        if (pos_ >= size_ || data_[pos_] != expected) return false;
        ++pos_;
        return true;
    }

    bool match(const char* literal) {
        const size_t length = std::strlen(literal);
        if (pos_ + length > size_ || std::memcmp(data_ + pos_, literal, length) != 0) return false;
        pos_ += length;
        return true;
    }

    void skip_ws() {
        while (pos_ < size_ && (data_[pos_] == ' ' || data_[pos_] == '\t' ||
                                data_[pos_] == '\n' || data_[pos_] == '\r')) {
            ++pos_;
        }
    }

    bool fail(SafeTensorsError error) {
        error_ = error;
        return false;
    }

    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;
    SafeTensorsError error_ = SafeTensorsError::HeaderJson;
};

const JSONValue* object_value(const JSONValue& object, const char* key) {
    if (object.type != JSONValue::Type::Object) return nullptr;
    for (const auto& [candidate, value] : object.object) {
        if (candidate == key) return &value;
    }
    return nullptr;
}

bool parse_dtype(const std::string& text, SafeTensorsDtype& dtype, uint64_t& bits) {
    struct DtypeEntry { const char* name; SafeTensorsDtype dtype; uint64_t bits; };
    static constexpr DtypeEntry kDtypes[] = {
        {"BOOL", SafeTensorsDtype::BOOL, 8},
        {"F4", SafeTensorsDtype::F4, 4},
        {"F6_E2M3", SafeTensorsDtype::F6_E2M3, 6},
        {"F6_E3M2", SafeTensorsDtype::F6_E3M2, 6},
        {"U8", SafeTensorsDtype::U8, 8},
        {"I8", SafeTensorsDtype::I8, 8},
        {"F8_E5M2", SafeTensorsDtype::F8_E5M2, 8},
        {"F8_E4M3", SafeTensorsDtype::F8_E4M3, 8},
        {"F8_E8M0", SafeTensorsDtype::F8_E8M0, 8},
        {"F8_E4M3FNUZ", SafeTensorsDtype::F8_E4M3FNUZ, 8},
        {"F8_E5M2FNUZ", SafeTensorsDtype::F8_E5M2FNUZ, 8},
        {"I16", SafeTensorsDtype::I16, 16},
        {"U16", SafeTensorsDtype::U16, 16},
        {"F16", SafeTensorsDtype::F16, 16},
        {"BF16", SafeTensorsDtype::BF16, 16},
        {"I32", SafeTensorsDtype::I32, 32},
        {"U32", SafeTensorsDtype::U32, 32},
        {"F32", SafeTensorsDtype::F32, 32},
        {"C64", SafeTensorsDtype::C64, 64},
        {"F64", SafeTensorsDtype::F64, 64},
        {"I64", SafeTensorsDtype::I64, 64},
        {"U64", SafeTensorsDtype::U64, 64},
    };
    for (const auto& entry : kDtypes) {
        if (text == entry.name) {
            dtype = entry.dtype;
            bits = entry.bits;
            return true;
        }
    }
    return false;
}

bool dtype_to_ggml(SafeTensorsDtype dtype, GGMLType& out) {
    switch (dtype) {
        case SafeTensorsDtype::F32: out = GGMLType::F32; return true;
        case SafeTensorsDtype::F16: out = GGMLType::F16; return true;
        case SafeTensorsDtype::BF16: out = GGMLType::BF16; return true;
        case SafeTensorsDtype::I8: out = GGMLType::I8; return true;
        case SafeTensorsDtype::I32: out = GGMLType::I32; return true;
        case SafeTensorsDtype::I64: out = GGMLType::I64; return true;
        case SafeTensorsDtype::U8: out = GGMLType::U8; return true;
        case SafeTensorsDtype::U32: out = GGMLType::U32; return true;
        case SafeTensorsDtype::BOOL: out = GGMLType::BOOL; return true;
        default: return false;
    }
}

bool read_text_file(const std::string& path, std::string& out) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (!file) return false;
    if (std::fseek(file, 0, SEEK_END) != 0) { std::fclose(file); return false; }
    const long size = std::ftell(file);
    if (size < 0 || std::fseek(file, 0, SEEK_SET) != 0) { std::fclose(file); return false; }
    out.resize(static_cast<size_t>(size));
    const size_t read = std::fread(out.data(), 1, out.size(), file);
    std::fclose(file);
    return read == out.size();
}

std::string dir_of(const std::string& path) {
    const size_t position = path.find_last_of('/');
    return position == std::string::npos ? "" : path.substr(0, position + 1);
}

} // namespace

SafeTensorsParseResult parse_safetensors(std::span<const uint8_t> bytes) {
    if (bytes.size() < 8) {
        return parse_error(SafeTensorsError::HeaderTooSmall, "SafeTensors preamble is shorter than 8 bytes");
    }

    uint64_t header_length = 0;
    for (size_t i = 0; i < 8; ++i) header_length |= static_cast<uint64_t>(bytes[i]) << (i * 8);
    if (header_length == std::numeric_limits<uint64_t>::max()) {
        return parse_error(SafeTensorsError::HeaderLength, "SafeTensors header length overflows its addressable range");
    }
    if (header_length > kMaxHeaderBytes) {
        return parse_error(SafeTensorsError::HeaderTooLarge, "SafeTensors header exceeds the 100 MB limit");
    }
    if (header_length > bytes.size() - 8) {
        return parse_error(SafeTensorsError::HeaderLength, "SafeTensors header extends past the supplied buffer");
    }

    const auto header = bytes.subspan(8, static_cast<size_t>(header_length));
    if (header.empty()) {
        return parse_error(SafeTensorsError::HeaderStart, "SafeTensors header must begin with '{'", 8);
    }
    if (!is_valid_utf8(header)) {
        return parse_error(SafeTensorsError::HeaderUtf8, "SafeTensors header is not valid UTF-8", 8);
    }
    const auto root_start = std::find_if(header.begin(), header.end(),
                                         [](uint8_t byte) {
                                             return !is_json_whitespace(byte);
                                         });
    if (root_start == header.end()) {
        return parse_error(SafeTensorsError::HeaderStart,
                           "SafeTensors header has no JSON object", 8);
    }
    if (*root_start != '{') {
        // Preserve a useful root-type diagnostic for an otherwise
        // recognizable JSON array.
        const SafeTensorsError code = *root_start == '[' ? SafeTensorsError::HeaderObject
                                                          : SafeTensorsError::HeaderStart;
        return parse_error(code, "SafeTensors header must contain a JSON object",
                           8 + static_cast<uint64_t>(root_start - header.begin()));
    }

    JSONParser parser(header.data(), header.size());
    JSONValue root;
    if (!parser.parse_value(root)) {
        return parse_error(parser.error(), "SafeTensors header is not valid JSON", 8 + parser.position());
    }
    for (size_t i = parser.position(); i < header.size(); ++i) {
        if (!is_json_whitespace(header[i])) {
            return parse_error(SafeTensorsError::HeaderPadding,
                               "only JSON whitespace may follow the JSON header", 8 + i);
        }
    }
    if (root.type != JSONValue::Type::Object) {
        return parse_error(SafeTensorsError::HeaderObject, "SafeTensors header root is not an object", 8);
    }

    const auto data = bytes.subspan(8 + static_cast<size_t>(header_length));
    SafeTensorsFile file;
    file.header_length_ = header_length;
    file.bytes_ = bytes;

    for (const auto& [name, descriptor] : root.object) {
        if (name == "__metadata__") {
            if (descriptor.type != JSONValue::Type::Object) {
                return parse_error(SafeTensorsError::MetadataNotObject, "__metadata__ must be an object");
            }
            for (const auto& [key, value] : descriptor.object) {
                if (value.type != JSONValue::Type::String) {
                    return parse_error(SafeTensorsError::MetadataNotString, "metadata values must be strings");
                }
                file.metadata_.emplace(key, value.string);
            }
            continue;
        }
        if (descriptor.type != JSONValue::Type::Object) {
            return parse_error(SafeTensorsError::TensorObject, "tensor descriptor is not an object");
        }
        for (const auto& [field, unused_value] : descriptor.object) {
            (void)unused_value;
            if (field != "dtype" && field != "shape" && field != "data_offsets") {
                return parse_error(SafeTensorsError::TensorField, "tensor descriptor contains an unknown field");
            }
        }
        const JSONValue* json_dtype = object_value(descriptor, "dtype");
        const JSONValue* json_shape = object_value(descriptor, "shape");
        const JSONValue* json_offsets = object_value(descriptor, "data_offsets");
        if (!json_dtype || !json_shape || !json_offsets || json_dtype->type != JSONValue::Type::String ||
            json_shape->type != JSONValue::Type::Array || json_offsets->type != JSONValue::Type::Array) {
            return parse_error(SafeTensorsError::TensorField, "tensor descriptor lacks dtype, shape, or data_offsets");
        }

        SafeTensorsTensor tensor;
        tensor.name = name;
        uint64_t dtype_bits = 0;
        if (!parse_dtype(json_dtype->string, tensor.dtype, dtype_bits)) {
            return parse_error(SafeTensorsError::Dtype, "unknown SafeTensors dtype");
        }

        uint64_t element_count = 1; // Scalar tensors have one element.
        for (const JSONValue& dimension : json_shape->array) {
            if (dimension.type != JSONValue::Type::Unsigned) {
                return parse_error(SafeTensorsError::InvalidInteger, "tensor shape entries must be bounded nonnegative integers");
            }
            tensor.shape.push_back(dimension.number);
            if (!checked_mul(element_count, dimension.number, element_count)) {
                return parse_error(SafeTensorsError::ArithmeticOverflow, "tensor element count overflows uint64");
            }
        }
        if (json_offsets->array.size() != 2) {
            return parse_error(SafeTensorsError::DataOffsets, "data_offsets must be two bounded nonnegative integers");
        }
        if (json_offsets->array[0].type != JSONValue::Type::Unsigned ||
            json_offsets->array[1].type != JSONValue::Type::Unsigned) {
            return parse_error(SafeTensorsError::InvalidInteger,
                               "data_offsets entries must be bounded nonnegative integers");
        }
        const uint64_t begin = json_offsets->array[0].number;
        const uint64_t end = json_offsets->array[1].number;
        if (end < begin) {
            return parse_error(SafeTensorsError::DataOffsets, "data_offsets end precedes its start");
        }
        uint64_t tensor_bits = 0;
        if (!checked_mul(element_count, dtype_bits, tensor_bits)) {
            return parse_error(SafeTensorsError::ArithmeticOverflow, "tensor byte count overflows uint64");
        }
        if (tensor_bits % 8 != 0) {
            return parse_error(SafeTensorsError::TensorByteSize, "tensor bit count is not byte aligned");
        }
        const uint64_t expected_length = tensor_bits / 8;
        if (end - begin != expected_length) {
            return parse_error(SafeTensorsError::TensorByteSize, "data_offsets do not match dtype and shape");
        }
        if (end > data.size()) {
            return parse_error(SafeTensorsError::IncompleteBuffer, "tensor data ends outside the data buffer");
        }
        tensor.data_offset = begin;
        tensor.data_length = expected_length;
        tensor.data = data.subspan(static_cast<size_t>(begin), static_cast<size_t>(expected_length));
        file.tensors_.push_back(std::move(tensor));
    }

    std::sort(file.tensors_.begin(), file.tensors_.end(), [](const SafeTensorsTensor& lhs,
                                                              const SafeTensorsTensor& rhs) {
        return lhs.data_offset < rhs.data_offset;
    });
    uint64_t next_offset = 0;
    for (const SafeTensorsTensor& tensor : file.tensors_) {
        if (tensor.data_offset != next_offset || !checked_add(next_offset, tensor.data_length, next_offset)) {
            return parse_error(SafeTensorsError::IncompleteBuffer,
                               "tensor intervals must exactly cover the data buffer without gaps or overlap");
        }
    }
    if (next_offset != data.size()) {
        return parse_error(SafeTensorsError::IncompleteBuffer,
                           "tensor intervals do not fully cover the data buffer");
    }
    return file;
}

bool SafeTensorsContext::open_shard(const std::string& full_path) {
    Shard shard;
    shard.file = std::make_unique<MappedFile>();
    if (!shard.file->open(full_path.c_str())) {
        std::fprintf(stderr, "safetensors: cannot mmap %s\n", full_path.c_str());
        return false;
    }
    const auto bytes = std::span<const uint8_t>(shard.file->data(), shard.file->size());
    SafeTensorsParseResult parsed = parse_safetensors(bytes);
    if (const auto* error = std::get_if<SafeTensorsParseError>(&parsed)) {
        std::fprintf(stderr, "safetensors: invalid wire format (%s): %s\n", full_path.c_str(),
                     error->report.detail.c_str());
        return false;
    }
    shard.parsed = std::get<SafeTensorsFile>(std::move(parsed));

    // These conditions are local adapter support, not SafeTensors corruption.
    for (const SafeTensorsTensor& source : shard.parsed.tensors()) {
        GGMLType type;
        if (!dtype_to_ggml(source.dtype, type)) {
            std::fprintf(stderr, "safetensors: unsupported adapter dtype for %s\n", source.name.c_str());
            return false;
        }
        if (source.shape.size() > 4) {
            std::fprintf(stderr, "safetensors: unsupported adapter rank for %s\n", source.name.c_str());
            return false;
        }
        Tensor tensor;
        tensor.name = source.name;
        tensor.type = type;
        tensor.n_dims = static_cast<uint32_t>(source.shape.size());
        for (size_t i = 0; i < source.shape.size(); ++i) {
            tensor.dims[i] = source.shape[source.shape.size() - 1 - i];
        }
        tensor.data = source.data.data();
        tensor_index_[tensor.name] = tensors_.size();
        tensors_.push_back(std::move(tensor));
    }
    for (const auto& [key, value] : shard.parsed.metadata()) metadata_[key] = value;
    shards_.push_back(std::move(shard));
    return true;
}

bool SafeTensorsContext::open(const char* path) {
    close();
    if (!path) return false;
    path_ = path;
    base_dir_ = dir_of(path_);
    sharded_ = false;
    return open_shard(path_);
}

bool SafeTensorsContext::open_sharded(const char* index_path) {
    close();
    if (!index_path) return false;
    path_ = index_path;
    base_dir_ = dir_of(path_);
    sharded_ = true;

    std::string content;
    if (!read_text_file(index_path, content)) {
        std::fprintf(stderr, "safetensors: cannot read index: %s\n", index_path);
        return false;
    }
    const auto index_bytes = std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(content.data()), content.size());
    if (!is_valid_utf8(index_bytes)) {
        std::fprintf(stderr, "safetensors: index is not UTF-8\n");
        return false;
    }
    JSONParser parser(index_bytes.data(), index_bytes.size());
    JSONValue root;
    if (!parser.parse_value(root) || !parser.consumed() || root.type != JSONValue::Type::Object) {
        std::fprintf(stderr, "safetensors: failed to parse index JSON\n");
        return false;
    }
    const JSONValue* weight_map = object_value(root, "weight_map");
    if (!weight_map || weight_map->type != JSONValue::Type::Object) {
        std::fprintf(stderr, "safetensors: index missing weight_map\n");
        return false;
    }
    std::vector<std::string> shard_files;
    for (const auto& [unused_tensor_name, shard_name] : weight_map->object) {
        (void)unused_tensor_name;
        if (shard_name.type == JSONValue::Type::String) shard_files.push_back(shard_name.string);
    }
    std::sort(shard_files.begin(), shard_files.end());
    shard_files.erase(std::unique(shard_files.begin(), shard_files.end()), shard_files.end());
    for (const std::string& shard_name : shard_files) {
        if (!open_shard(base_dir_ + shard_name)) return false;
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
    const auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

const Tensor* SafeTensorsContext::find_tensor(const char* name) const {
    return name ? find_tensor(std::string(name)) : nullptr;
}

} // namespace Laplace
