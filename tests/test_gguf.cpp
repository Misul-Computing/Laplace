// test_gguf - parser correctness on synthetic GGUF files (no model needed).
//
// Covers: metadata round-trip for every value type incl. small-int and bool
// arrays, tensor info parsing with correct data pointers, and rejection of
// files whose tensor data extends past EOF.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "gguf.h"
#include "artifact_set.h"
#include "tensor.h"

#include "gguf_writer.h"
#include "test_util.h"

using namespace Laplace;

namespace {

gguf_writer::TensorDecl make_f32_tensor(const std::string& name, int K, int N, float fill) {
    gguf_writer::TensorDecl t;
    t.name = name;
    t.dims = {static_cast<uint64_t>(K), static_cast<uint64_t>(N)};
    t.type = static_cast<uint32_t>(GGMLType::F32);
    t.data.resize(sizeof(float) * K * N);
    std::vector<float> v(static_cast<size_t>(K) * N, fill);
    std::memcpy(t.data.data(), v.data(), t.data.size());
    return t;
}

void test_valid_file() {
    gguf_writer::Writer w;
    w.kv_str("general.architecture", "test");
    w.kv_u32("test.block_count", 24);
    w.kv_i32("test.signed_value", -7);
    w.kv_f32("test.eps", 1e-6f);
    w.kv_bool("test.flag", true);
    w.kv_arr_str("test.names", {"alpha", "beta"});
    w.kv_arr_i32("test.sections", {11, 11, 10, 0});
    w.kv_arr_u8("test.bytes", {1, 2, 3});          // u8 array must not abort parse
    w.kv_arr_i16("test.shorts", {-1, 2, -3});      // i16 array must not abort parse
    w.kv_arr_bool("test.flags", {true, false});    // bool array must not abort parse
    w.add_tensor(make_f32_tensor("token_embd.weight", 4, 3, 0.5f));
    w.add_tensor(make_f32_tensor("output_norm.weight", 4, 1, 1.0f));
    CHECK(w.write_file("/private/tmp/laplace-test-gguf-valid.gguf"));

    GGUFContext ctx;
    CHECK_MSG(ctx.open("/private/tmp/laplace-test-gguf-valid.gguf"),
              "parser rejected a valid file (small-int/bool arrays must be tolerated)");
    if (ctx.file_data()) {
        const auto& m = ctx.metadata();
        auto* arch = meta_str(m, "general.architecture");
        CHECK(arch && *arch == "test");
        CHECK(meta_int(m, "test.block_count") == 24);
        CHECK(meta_int(m, "test.signed_value") == -7);
        CHECK(almost_equal(static_cast<float>(meta_float(m, "test.eps")), 1e-6f, 1e-6f, 1e-12f));

        CHECK(ctx.tensors().size() == 2);
        const Tensor* t = ctx.find_tensor("token_embd.weight");
        CHECK(t != nullptr);
        if (t) {
            CHECK(t->n_dims == 2);
            CHECK(t->dims[0] == 4 && t->dims[1] == 3);
            CHECK(t->nbytes() == 4 * 3 * sizeof(float));
            // Data pointer must be valid and hold the fill value.
            const float* p = reinterpret_cast<const float*>(t->data);
            CHECK(p[0] == 0.5f && p[11] == 0.5f);
        }
    }
    remove("/private/tmp/laplace-test-gguf-valid.gguf");
}

void test_truncated_tensor_rejected() {
    gguf_writer::Writer w;
    w.kv_str("general.architecture", "test");
    w.add_tensor(make_f32_tensor("token_embd.weight", 64, 4, 1.0f));
    // Cut 512 bytes off the tensor data: offset is still inside the file but
    // offset + nbytes is past EOF. The parser must refuse.
    CHECK(w.write_file("/private/tmp/laplace-test-gguf-trunc.gguf", 512));

    GGUFContext ctx;
    CHECK_MSG(!ctx.open("/private/tmp/laplace-test-gguf-trunc.gguf"),
              "parser accepted a file with tensor data past EOF");
    remove("/private/tmp/laplace-test-gguf-trunc.gguf");
}

void test_bad_magic_rejected() {
    FILE* f = fopen("/private/tmp/laplace-test-gguf-bad.gguf", "wb");
    const char junk[] = "NOTAGGUFFILE----------------";
    fwrite(junk, 1, sizeof(junk), f);
    fclose(f);
    GGUFContext ctx;
    CHECK(!ctx.open("/private/tmp/laplace-test-gguf-bad.gguf"));
    remove("/private/tmp/laplace-test-gguf-bad.gguf");
}

void test_duplicate_metadata_rejected() {
    gguf_writer::Writer w;
    w.kv_u32("test.duplicate", 1);
    w.kv_u32("test.duplicate", 2);
    w.add_tensor(make_f32_tensor("token_embd.weight", 2, 2, 0.25f));
    CHECK(w.write_file("/private/tmp/laplace-test-gguf-duplicate.gguf"));
    GGUFContext ctx;
    CHECK(!ctx.open("/private/tmp/laplace-test-gguf-duplicate.gguf"));
    remove("/private/tmp/laplace-test-gguf-duplicate.gguf");
}

void test_duplicate_tensor_rejected() {
    gguf_writer::Writer w;
    w.kv_str("general.architecture", "test");
    w.add_tensor(make_f32_tensor("duplicate.weight", 2, 2, 0.25f));
    w.add_tensor(make_f32_tensor("duplicate.weight", 2, 2, 0.5f));
    const char* path = "/private/tmp/laplace-test-gguf-duplicate-tensor.gguf";
    CHECK(w.write_file(path));
    GGUFContext ctx;
    CHECK(!ctx.open(path));
    remove(path);
}

void test_malformed_metadata_key_rejected() {
    gguf_writer::Writer w;
    w.kv_str("General.Architecture", "test");
    w.add_tensor(make_f32_tensor("tensor.weight", 2, 2, 0.25f));
    const char* path = "/private/tmp/laplace-test-gguf-bad-key.gguf";
    CHECK(w.write_file(path));
    GGUFContext ctx;
    CHECK(!ctx.open(path));
    remove(path);
}

void test_invalid_bool_rejected() {
    const char* path = "/private/tmp/laplace-test-gguf-bad-bool.gguf";
    gguf_writer::Writer w;
    w.kv_bool("test.flag", true);
    w.add_tensor(make_f32_tensor("tensor.weight", 2, 2, 0.25f));
    CHECK(w.write_file(path));
    std::ifstream input(path, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
    const std::string key = "test.flag";
    const auto found = std::search(bytes.begin(), bytes.end(), key.begin(), key.end());
    CHECK(found != bytes.end());
    if (found != bytes.end()) {
        const size_t key_offset = static_cast<size_t>(found - bytes.begin());
        const size_t value_offset = key_offset + key.size() + sizeof(uint32_t);
        CHECK(value_offset < bytes.size());
        if (value_offset < bytes.size()) bytes[value_offset] = 2;
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    output.close();
    GGUFContext ctx;
    CHECK(!ctx.open(path));
    remove(path);
}

void test_oversized_tensor_name_rejected() {
    const char* path = "/private/tmp/laplace-test-gguf-long-name.gguf";
    gguf_writer::Writer w;
    w.kv_str("general.architecture", "test");
    w.add_tensor(make_f32_tensor(std::string(65, 'x'), 2, 2, 0.25f));
    CHECK(w.write_file(path));
    GGUFContext ctx;
    CHECK(!ctx.open(path));
    remove(path);
}

void test_checked_bytes_parse_without_mapping() {
    gguf_writer::Writer w;
    w.kv_str("general.architecture", "test");
    w.add_tensor(make_f32_tensor("token_embd.weight", 2, 2, 0.25f));
    CHECK(w.write_file("/private/tmp/laplace-test-gguf-checked.gguf"));

    auto loaded = ArtifactSet::load_single_file("/private/tmp/laplace-test-gguf-checked.gguf");
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    if (auto* artifacts = std::get_if<ArtifactSet>(&loaded)) {
        auto view = artifacts->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) {
            GGUFContext ctx;
            CHECK(ctx.parse(*package));
            const Tensor* tensor = ctx.find_tensor("token_embd.weight");
            CHECK(tensor != nullptr);
            if (tensor) {
                CHECK(reinterpret_cast<const float*>(tensor->data)[3] == 0.25f);
            }
        }
    }
    remove("/private/tmp/laplace-test-gguf-checked.gguf");
}

} // namespace

int main() {
    test_valid_file();
    test_truncated_tensor_rejected();
    test_bad_magic_rejected();
    test_duplicate_metadata_rejected();
    test_duplicate_tensor_rejected();
    test_malformed_metadata_key_rejected();
    test_invalid_bool_rejected();
    test_oversized_tensor_name_rejected();
    test_checked_bytes_parse_without_mapping();
    return test_summary("test_gguf");
}
