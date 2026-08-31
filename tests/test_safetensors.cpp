// test_safetensors - parser correctness on synthetic .safetensors files.
//
// Covers: single-file parsing with multiple dtypes, metadata round-trip,
// shape reversal (row-major -> GGML innermost-first), data pointer
// correctness, sharded models via index.json, and rejection of bad files.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <vector>
#include <variant>

#include "safetensors.h"
#include "tensor.h"

#include "test_util.h"

using namespace Laplace;

namespace {

template <typename T>
T load_unaligned(const void* bytes, size_t index) {
    T value{};
    std::memcpy(&value, static_cast<const uint8_t*>(bytes) + index * sizeof(T), sizeof(T));
    return value;
}

// ---- synthetic .safetensors file writer ----

struct STTensor {
    std::string name;
    std::string dtype;             // "F32", "F16", "BF16", "I8", "U32", ...
    std::vector<uint64_t> shape;   // row-major (outermost first)
    std::vector<uint8_t> data;
};

bool write_safetensors(const std::string& path,
                       const std::vector<std::pair<std::string, std::string>>& metadata,
                       const std::vector<STTensor>& tensors) {
    std::string json = "{";
    bool first = true;

    if (!metadata.empty()) {
        json += "\"__metadata__\":{";
        for (size_t i = 0; i < metadata.size(); i++) {
            if (i > 0) json += ",";
            json += "\"" + metadata[i].first + "\":\"" + metadata[i].second + "\"";
        }
        json += "}";
        first = false;
    }

    uint64_t offset = 0;
    for (const auto& t : tensors) {
        if (!first) json += ",";
        json += "\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
        for (size_t i = 0; i < t.shape.size(); i++) {
            if (i > 0) json += ",";
            json += std::to_string(t.shape[i]);
        }
        json += "],\"data_offsets\":[" + std::to_string(offset) + ","
                + std::to_string(offset + t.data.size()) + "]}";
        offset += t.data.size();
        first = false;
    }
    json += "}";

    uint64_t json_len = json.size();
    std::string file;
    file.append(reinterpret_cast<const char*>(&json_len), 8);
    file += json;
    for (const auto& t : tensors)
        file.append(reinterpret_cast<const char*>(t.data.data()), t.data.size());

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t n = fwrite(file.data(), 1, file.size(), f);
    fclose(f);
    return n == file.size();
}

bool write_index_json(const std::string& path,
                      const std::map<std::string, std::string>& weight_map,
                      uint64_t total_size) {
    std::string json = "{\"metadata\":{\"total_size\":" + std::to_string(total_size)
                       + "},\"weight_map\":{";
    bool first = true;
    for (const auto& [name, shard] : weight_map) {
        if (!first) json += ",";
        json += "\"" + name + "\":\"" + shard + "\"";
        first = false;
    }
    json += "}}";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t n = fwrite(json.data(), 1, json.size(), f);
    fclose(f);
    return n == json.size();
}

std::vector<uint8_t> wire_file(const std::string& header,
                               const std::vector<uint8_t>& payload = {}) {
    std::vector<uint8_t> bytes(8);
    uint64_t n = header.size();
    for (size_t i = 0; i < 8; ++i) bytes[i] = static_cast<uint8_t>(n >> (8 * i));
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    return bytes;
}

std::vector<uint8_t> wire_preamble(uint64_t n) {
    std::vector<uint8_t> bytes(8);
    for (size_t i = 0; i < 8; ++i) bytes[i] = static_cast<uint8_t>(n >> (8 * i));
    return bytes;
}

struct ParsedWire {
    std::vector<uint8_t> bytes;
    SafeTensorsParseResult result;

    explicit ParsedWire(std::vector<uint8_t> input)
        : bytes(std::move(input)), result(parse_safetensors(bytes)) {}
};

const SafeTensorsFile* parsed_file(const SafeTensorsParseResult& result) {
    return std::get_if<SafeTensorsFile>(&result);
}

void expect_wire_error(const std::vector<uint8_t>& bytes, SafeTensorsError expected) {
    const SafeTensorsParseResult result = parse_safetensors(bytes);
    const auto* error = std::get_if<SafeTensorsParseError>(&result);
    CHECK(error != nullptr);
    if (error) {
        if (error->code != expected) {
            fprintf(stderr, "wire error mismatch: expected %u, got %u\n",
                    static_cast<unsigned>(expected), static_cast<unsigned>(error->code));
        }
        CHECK(error->code == expected);
        CHECK(error->report.code == CompatibilityError::PACKAGE_BOUNDS_INVALID);
    }
}

// ---- test cases ----

void test_single_file() {
    std::vector<STTensor> tensors;

    // F32 [4, 3] -> 48 bytes. Values 0.5 .. 11.5.
    STTensor t1;
    t1.name = "layer1.weight";
    t1.dtype = "F32";
    t1.shape = {4, 3};
    t1.data.resize(48);
    float f32_vals[12] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f,
                          6.5f, 7.5f, 8.5f, 9.5f, 10.5f, 11.5f};
    memcpy(t1.data.data(), f32_vals, 48);
    tensors.push_back(std::move(t1));

    // F16 [2, 2] -> 8 bytes. Values 1.0, 2.0, 3.0, 4.0.
    STTensor t2;
    t2.name = "layer1.bias";
    t2.dtype = "F16";
    t2.shape = {2, 2};
    uint16_t f16_vals[4] = {0x3C00, 0x4000, 0x4200, 0x4400};
    t2.data.resize(8);
    memcpy(t2.data.data(), f16_vals, 8);
    tensors.push_back(std::move(t2));

    // BF16 [3] -> 6 bytes. Values 1.0, 2.0, 3.0 in BF16.
    STTensor t3;
    t3.name = "layer1.norm";
    t3.dtype = "BF16";
    t3.shape = {3};
    uint16_t bf16_vals[3] = {0x3F80, 0x4000, 0x4080};
    t3.data.resize(6);
    memcpy(t3.data.data(), bf16_vals, 6);
    tensors.push_back(std::move(t3));

    // I8 [4] -> 4 bytes.
    STTensor t4;
    t4.name = "layer1.scale";
    t4.dtype = "I8";
    t4.shape = {4};
    t4.data = {1, 2, 0xFF, 3};  // 0xFF = -1 as int8
    tensors.push_back(std::move(t4));

    // U32 [2] -> 8 bytes.
    STTensor t5;
    t5.name = "layer1.packed";
    t5.dtype = "U32";
    t5.shape = {2};
    uint32_t u32_vals[2] = {0xDEADBEEFu, 0x12345678u};
    t5.data.resize(8);
    memcpy(t5.data.data(), u32_vals, 8);
    tensors.push_back(std::move(t5));

    CHECK(write_safetensors("test_st_single.safetensors",
                            {{"format", "mlx"}}, tensors));

    SafeTensorsContext ctx;
    CHECK_MSG(ctx.open("test_st_single.safetensors"),
              "parser rejected a valid file");

    // Metadata
    const auto& meta = ctx.metadata();
    CHECK(meta.count("format") && meta.at("format") == "mlx");

    CHECK(ctx.tensors().size() == 4 + 1);  // 5 tensors
    CHECK(!ctx.is_sharded());
    CHECK(ctx.num_shards() == 1);

    // F32 tensor: shape [4,3] -> GGML dims[0]=3, dims[1]=4
    const Tensor* t = ctx.find_tensor("layer1.weight");
    CHECK(t != nullptr);
    if (t) {
        CHECK(t->type == GGMLType::F32);
        CHECK(t->n_dims == 2);
        CHECK(t->dims[0] == 3);
        CHECK(t->dims[1] == 4);
        CHECK(t->nbytes() == 48);
        CHECK(load_unaligned<float>(t->data, 0) == 0.5f);
        CHECK(load_unaligned<float>(t->data, 11) == 11.5f);
    }

    // F16 tensor: shape [2,2] -> dims[0]=2, dims[1]=2
    t = ctx.find_tensor("layer1.bias");
    CHECK(t != nullptr);
    if (t) {
        CHECK(t->type == GGMLType::F16);
        CHECK(t->n_dims == 2);
        CHECK(t->dims[0] == 2 && t->dims[1] == 2);
        CHECK(t->nbytes() == 8);
        CHECK(load_unaligned<uint16_t>(t->data, 0) == 0x3C00);
        CHECK(load_unaligned<uint16_t>(t->data, 3) == 0x4400);
    }

    // BF16 tensor: shape [3] -> dims[0]=3
    t = ctx.find_tensor("layer1.norm");
    CHECK(t != nullptr);
    if (t) {
        CHECK(t->type == GGMLType::BF16);
        CHECK(t->n_dims == 1);
        CHECK(t->dims[0] == 3);
        CHECK(t->nbytes() == 6);
        CHECK(load_unaligned<uint16_t>(t->data, 0) == 0x3F80);
        CHECK(load_unaligned<uint16_t>(t->data, 2) == 0x4080);
    }

    // I8 tensor: shape [4] -> dims[0]=4
    t = ctx.find_tensor("layer1.scale");
    CHECK(t != nullptr);
    if (t) {
        CHECK(t->type == GGMLType::I8);
        CHECK(t->n_dims == 1);
        CHECK(t->dims[0] == 4);
        CHECK(t->nbytes() == 4);
        CHECK(load_unaligned<int8_t>(t->data, 0) == 1);
        CHECK(load_unaligned<int8_t>(t->data, 1) == 2);
        CHECK(load_unaligned<int8_t>(t->data, 2) == -1);
        CHECK(load_unaligned<int8_t>(t->data, 3) == 3);
    }

    // U32 tensor: shape [2] -> dims[0]=2
    t = ctx.find_tensor("layer1.packed");
    CHECK(t != nullptr);
    if (t) {
        CHECK(t->type == GGMLType::U32);
        CHECK(t->n_dims == 1);
        CHECK(t->dims[0] == 2);
        CHECK(t->nbytes() == 8);
        CHECK(load_unaligned<uint32_t>(t->data, 0) == 0xDEADBEEFu);
        CHECK(load_unaligned<uint32_t>(t->data, 1) == 0x12345678u);
    }

    // Missing tensor returns nullptr.
    CHECK(ctx.find_tensor("does_not_exist") == nullptr);

    remove("test_st_single.safetensors");
}

void test_no_metadata() {
    std::vector<STTensor> tensors;
    STTensor t;
    t.name = "solo";
    t.dtype = "F32";
    t.shape = {1};
    float val = 42.0f;
    t.data.resize(4);
    memcpy(t.data.data(), &val, 4);
    tensors.push_back(std::move(t));

    CHECK(write_safetensors("test_st_nometa.safetensors", {}, tensors));

    SafeTensorsContext ctx;
    CHECK(ctx.open("test_st_nometa.safetensors"));
    CHECK(ctx.metadata().empty());
    CHECK(ctx.tensors().size() == 1);
    const Tensor* p = ctx.find_tensor("solo");
    CHECK(p != nullptr);
    if (p) {
        CHECK(p->type == GGMLType::F32);
        CHECK(load_unaligned<float>(p->data, 0) == 42.0f);
    }
    remove("test_st_nometa.safetensors");
}

void test_3d_shape() {
    // Shape [2, 3, 4] (row-major) -> GGML dims[0]=4, dims[1]=3, dims[2]=2.
    STTensor t;
    t.name = "cube";
    t.dtype = "U8";
    t.shape = {2, 3, 4};
    t.data.resize(24);
    for (int i = 0; i < 24; i++) t.data[i] = static_cast<uint8_t>(i);

    CHECK(write_safetensors("test_st_3d.safetensors", {}, {t}));

    SafeTensorsContext ctx;
    CHECK(ctx.open("test_st_3d.safetensors"));
    const Tensor* p = ctx.find_tensor("cube");
    CHECK(p != nullptr);
    if (p) {
        CHECK(p->type == GGMLType::U8);
        CHECK(p->n_dims == 3);
        CHECK(p->dims[0] == 4);
        CHECK(p->dims[1] == 3);
        CHECK(p->dims[2] == 2);
        CHECK(p->nbytes() == 24);
        CHECK(p->data[0] == 0);
        CHECK(p->data[23] == 23);
    }
    remove("test_st_3d.safetensors");
}

void test_sharded() {
    // Shard 1: block0.weight (F32 [2,2]) + block0.bias (F32 [2])
    std::vector<STTensor> s1;
    STTensor a;
    a.name = "block0.weight";
    a.dtype = "F32";
    a.shape = {2, 2};
    a.data.resize(16);
    float a_vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(a.data.data(), a_vals, 16);
    s1.push_back(std::move(a));

    STTensor b;
    b.name = "block0.bias";
    b.dtype = "F32";
    b.shape = {2};
    b.data.resize(8);
    float b_vals[2] = {0.5f, 1.5f};
    memcpy(b.data.data(), b_vals, 8);
    s1.push_back(std::move(b));

    CHECK(write_safetensors("test_st_shard1.safetensors", {}, s1));

    // Shard 2: block1.weight (F16 [3,2]) + block1.bias (BF16 [3])
    std::vector<STTensor> s2;
    STTensor c;
    c.name = "block1.weight";
    c.dtype = "F16";
    c.shape = {3, 2};
    c.data.resize(12);
    uint16_t c_vals[6] = {0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600};
    memcpy(c.data.data(), c_vals, 12);
    s2.push_back(std::move(c));

    STTensor d;
    d.name = "block1.bias";
    d.dtype = "BF16";
    d.shape = {3};
    d.data.resize(6);
    uint16_t d_vals[3] = {0x3F80, 0x4000, 0x4080};
    memcpy(d.data.data(), d_vals, 6);
    s2.push_back(std::move(d));

    CHECK(write_safetensors("test_st_shard2.safetensors", {}, s2));

    // Index JSON
    std::map<std::string, std::string> wm;
    wm["block0.weight"] = "test_st_shard1.safetensors";
    wm["block0.bias"]   = "test_st_shard1.safetensors";
    wm["block1.weight"] = "test_st_shard2.safetensors";
    wm["block1.bias"]   = "test_st_shard2.safetensors";
    CHECK(write_index_json("test_st_model.safetensors.index.json", wm, 42));

    SafeTensorsContext ctx;
    CHECK_MSG(ctx.open_sharded("test_st_model.safetensors.index.json"),
              "parser rejected sharded model");

    CHECK(ctx.is_sharded());
    CHECK(ctx.num_shards() == 2);
    CHECK(ctx.tensors().size() == 4);

    // Tensor from shard 1.
    const Tensor* t = ctx.find_tensor("block0.weight");
    CHECK(t != nullptr);
    if (t) {
        CHECK(t->type == GGMLType::F32);
        CHECK(t->n_dims == 2);
        CHECK(t->dims[0] == 2 && t->dims[1] == 2);
        CHECK(load_unaligned<float>(t->data, 0) == 1.0f &&
              load_unaligned<float>(t->data, 3) == 4.0f);
    }

    t = ctx.find_tensor("block0.bias");
    CHECK(t != nullptr);
    if (t) {
        CHECK(t->type == GGMLType::F32);
        CHECK(t->n_dims == 1 && t->dims[0] == 2);
        CHECK(load_unaligned<float>(t->data, 0) == 0.5f &&
              load_unaligned<float>(t->data, 1) == 1.5f);
    }

    // Tensor from shard 2. Shape [3,2] -> dims[0]=2, dims[1]=3.
    t = ctx.find_tensor("block1.weight");
    CHECK(t != nullptr);
    if (t) {
        CHECK(t->type == GGMLType::F16);
        CHECK(t->n_dims == 2);
        CHECK(t->dims[0] == 2);
        CHECK(t->dims[1] == 3);
        CHECK(load_unaligned<uint16_t>(t->data, 0) == 0x3C00);
        CHECK(load_unaligned<uint16_t>(t->data, 5) == 0x4600);
    }

    t = ctx.find_tensor("block1.bias");
    CHECK(t != nullptr);
    if (t) {
        CHECK(t->type == GGMLType::BF16);
        CHECK(t->n_dims == 1 && t->dims[0] == 3);
    }

    CHECK(ctx.find_tensor("nope") == nullptr);

    remove("test_st_shard1.safetensors");
    remove("test_st_shard2.safetensors");
    remove("test_st_model.safetensors.index.json");
}

void test_bad_file_rejected() {
    // Too small to contain the 8-byte header length.
    FILE* f = fopen("test_st_bad.safetensors", "wb");
    const char junk[] = "short";
    fwrite(junk, 1, 5, f);
    fclose(f);
    SafeTensorsContext ctx;
    CHECK(!ctx.open("test_st_bad.safetensors"));
    remove("test_st_bad.safetensors");
}

void test_bad_json_rejected() {
    // Valid 8-byte length but garbage JSON.
    FILE* f = fopen("test_st_badjson.safetensors", "wb");
    uint64_t hdr_len = 16;
    fwrite(&hdr_len, 8, 1, f);
    const char junk[] = "NOT_VALID_JSON!!!";
    fwrite(junk, 1, 16, f);
    fclose(f);
    SafeTensorsContext ctx;
    CHECK(!ctx.open("test_st_badjson.safetensors"));
    remove("test_st_badjson.safetensors");
}

void test_wire_format_contract() {
    // ST-01: LE header length and ASCII-space padding are valid.
    const ParsedWire padded(wire_file(
        "{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}   ", {7}));
    const SafeTensorsFile* padded_file = parsed_file(padded.result);
    CHECK(padded_file != nullptr);
    if (padded_file) {
        CHECK(padded_file->header_length() == 56);
        CHECK(padded_file->tensors().size() == 1);
        CHECK(padded_file->tensors()[0].data.size() == 1);
        CHECK(padded_file->tensors()[0].data[0] == 7);
    }

    // ST-02 and ST-03: scalar and empty tensors are valid.
    const ParsedWire scalar(wire_file(
        "{\"scalar\":{\"dtype\":\"BF16\",\"shape\":[],\"data_offsets\":[0,2]}}", {0x80, 0x3f}));
    const SafeTensorsFile* scalar_file = parsed_file(scalar.result);
    CHECK(scalar_file != nullptr);
    if (scalar_file) {
        CHECK(scalar_file->tensors()[0].dtype == SafeTensorsDtype::BF16);
        CHECK(scalar_file->tensors()[0].shape.empty());
        CHECK(scalar_file->tensors()[0].data.size() == 2);
    }
    const ParsedWire empty(wire_file(
        "{\"empty\":{\"dtype\":\"F16\",\"shape\":[0],\"data_offsets\":[0,0]}}"));
    const SafeTensorsFile* empty_file = parsed_file(empty.result);
    CHECK(empty_file != nullptr);
    if (empty_file) CHECK(empty_file->tensors()[0].data.empty());

    // The current format also has sub-byte dtypes; validate bit arithmetic
    // rather than truncating their element width to zero bytes.
    const ParsedWire packed_f4(wire_file(
        "{\"packed\":{\"dtype\":\"F4\",\"shape\":[2],\"data_offsets\":[0,1]}}", {0}));
    const SafeTensorsFile* packed_f4_file = parsed_file(packed_f4.result);
    CHECK(packed_f4_file != nullptr);
    if (packed_f4_file) CHECK(packed_f4_file->tensors()[0].dtype == SafeTensorsDtype::F4);
    const ParsedWire packed_f6(wire_file(
        "{\"packed\":{\"dtype\":\"F6_E2M3\",\"shape\":[4],\"data_offsets\":[0,3]}}", {0, 0, 0}));
    CHECK(parsed_file(packed_f6.result) != nullptr);
    expect_wire_error(wire_file(
        "{\"packed\":{\"dtype\":\"F4\",\"shape\":[1],\"data_offsets\":[0,1]}}", {0}),
                      SafeTensorsError::TensorByteSize);

    // ST-04: descriptor order is irrelevant; output is ordered by byte span.
    const ParsedWire unordered(wire_file(
        "{\"b\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]},\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", {3, 4}));
    const SafeTensorsFile* unordered_file = parsed_file(unordered.result);
    CHECK(unordered_file != nullptr);
    if (unordered_file) {
        CHECK(unordered_file->tensors()[0].name == "a");
        CHECK(unordered_file->tensors()[1].name == "b");
    }

    // Current upstream SafeTensors accepts JSON whitespace before and after
    // the root object. It is header content, so tensor data still begins after
    // the complete declared header length.
    const ParsedWire whitespace(wire_file(
        " \t\r\n{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}\n\t ",
        {9}));
    const SafeTensorsFile* whitespace_file = parsed_file(whitespace.result);
    CHECK(whitespace_file != nullptr);
    if (whitespace_file) CHECK(whitespace_file->tensors()[0].data[0] == 9);

    // ST-05 through ST-20: exact malformed-wire refusals.
    expect_wire_error(wire_preamble(100000001), SafeTensorsError::HeaderTooLarge);
    expect_wire_error(wire_preamble(std::numeric_limits<uint64_t>::max()), SafeTensorsError::HeaderLength);
    expect_wire_error(wire_file(std::string("{}\0", 3)), SafeTensorsError::HeaderPadding);
    expect_wire_error(wire_file("{\"x\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[0,0]},\"x\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[0,0]}}"),
                      SafeTensorsError::DuplicateKey);
    expect_wire_error(wire_file("[]"), SafeTensorsError::HeaderObject);
    expect_wire_error(wire_file("{\"__metadata__\":{\"a\":\"one\",\"a\":\"two\"}}"),
                      SafeTensorsError::DuplicateKey);
    expect_wire_error(wire_file("{\"__metadata__\":{\"a\":1}}"), SafeTensorsError::MetadataNotString);
    expect_wire_error(wire_file("{\"x\":{\"dtype\":\"U8\",\"shape\":[\"1\"],\"data_offsets\":[0,0]}}"),
                      SafeTensorsError::InvalidInteger);
    expect_wire_error(wire_file("{\"x\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,-1]}}"),
                      SafeTensorsError::InvalidInteger);
    expect_wire_error(wire_file("{\"x\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,1]}}", {0}),
                      SafeTensorsError::TensorByteSize);
    expect_wire_error(wire_file("{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},\"b\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[2,3]}}", {1, 0, 2}),
                      SafeTensorsError::IncompleteBuffer);
    expect_wire_error(wire_file("{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},\"b\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", {1}),
                      SafeTensorsError::IncompleteBuffer);
    expect_wire_error(wire_file("{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", {1, 2}),
                      SafeTensorsError::IncompleteBuffer);
    expect_wire_error(wire_file("{\"a\":{\"dtype\":\"F16\",\"shape\":[281474976710655,281474976710655],\"data_offsets\":[0,0]}}"),
                      SafeTensorsError::ArithmeticOverflow);
    // UINT64 values are format-valid integers. An offset outside the supplied
    // data buffer is a coverage error, not an invented smaller integer limit.
    expect_wire_error(wire_file(
        "{\"a\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[18446744073709551615,18446744073709551615]}}"),
                      SafeTensorsError::IncompleteBuffer);
    std::string invalid_utf8 = "{\"";
    invalid_utf8.push_back(static_cast<char>(0xff));
    invalid_utf8 += "\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[0,0]}}";
    expect_wire_error(wire_file(invalid_utf8), SafeTensorsError::HeaderUtf8);
}

void test_wire_parser_keeps_adapter_policy_separate() {
    const ParsedWire result(wire_file(
        "{\"rank_five\":{\"dtype\":\"U8\",\"shape\":[1,1,1,1,1],\"data_offsets\":[0,1]},\"i16\":{\"dtype\":\"I16\",\"shape\":[1],\"data_offsets\":[1,3]}}", {5, 0, 0}));
    const SafeTensorsFile* file = parsed_file(result.result);
    CHECK(file != nullptr);
    if (file) {
        CHECK(file->tensors().size() == 2);
        CHECK(file->tensors()[0].name == "rank_five");
        CHECK(file->tensors()[1].dtype == SafeTensorsDtype::I16);
    }
}

} // namespace

int main() {
    test_single_file();
    test_no_metadata();
    test_3d_shape();
    test_sharded();
    test_bad_file_rejected();
    test_bad_json_rejected();
    test_wire_format_contract();
    test_wire_parser_keeps_adapter_policy_separate();
    return test_summary("test_safetensors");
}
