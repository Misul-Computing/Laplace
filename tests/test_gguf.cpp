// test_gguf - parser correctness on synthetic GGUF files (no model needed).
//
// Covers: metadata round-trip for every value type incl. small-int and bool
// arrays, tensor info parsing with correct data pointers, and rejection of
// files whose tensor data extends past EOF.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gguf.h"
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
    CHECK(w.write_file("test_gguf_valid.gguf"));

    GGUFContext ctx;
    CHECK_MSG(ctx.open("test_gguf_valid.gguf"),
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
    remove("test_gguf_valid.gguf");
}

void test_truncated_tensor_rejected() {
    gguf_writer::Writer w;
    w.kv_str("general.architecture", "test");
    w.add_tensor(make_f32_tensor("token_embd.weight", 64, 4, 1.0f));
    // Cut 512 bytes off the tensor data: offset is still inside the file but
    // offset + nbytes is past EOF. The parser must refuse.
    CHECK(w.write_file("test_gguf_trunc.gguf", 512));

    GGUFContext ctx;
    CHECK_MSG(!ctx.open("test_gguf_trunc.gguf"),
              "parser accepted a file with tensor data past EOF");
    remove("test_gguf_trunc.gguf");
}

void test_bad_magic_rejected() {
    FILE* f = fopen("test_gguf_bad.gguf", "wb");
    const char junk[] = "NOTAGGUFFILE----------------";
    fwrite(junk, 1, sizeof(junk), f);
    fclose(f);
    GGUFContext ctx;
    CHECK(!ctx.open("test_gguf_bad.gguf"));
    remove("test_gguf_bad.gguf");
}

} // namespace

int main() {
    test_valid_file();
    test_truncated_tensor_rejected();
    test_bad_magic_rejected();
    return test_summary("test_gguf");
}
