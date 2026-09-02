#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "artifact_index.h"
#include "artifact_set.h"
#include "gguf_fact_keys.h"
#include "gguf_index.h"
#include "gguf_writer.h"
#include "tensor.h"
#include "test_util.h"

using namespace Laplace;

namespace {

void add_f32(gguf_writer::Writer& writer, std::string name, std::vector<uint64_t> dims) {
    uint64_t elements = 1;
    for (uint64_t dimension : dims) elements *= dimension;
    gguf_writer::TensorDecl tensor;
    tensor.name = std::move(name);
    tensor.dims = std::move(dims);
    tensor.type = static_cast<uint32_t>(GGMLType::F32);
    tensor.data.resize(static_cast<size_t>(elements) * sizeof(float));
    writer.add_tensor(std::move(tensor));
}

void add_q4k(gguf_writer::Writer& writer, std::string name, std::vector<uint64_t> dims) {
    uint64_t elements = 1;
    for (uint64_t dimension : dims) elements *= dimension;
    CHECK(elements % 256 == 0);
    gguf_writer::TensorDecl tensor;
    tensor.name = std::move(name);
    tensor.dims = std::move(dims);
    tensor.type = static_cast<uint32_t>(GGMLType::Q4_K);
    tensor.data.resize(static_cast<size_t>(elements / 256) * bytes_per_block(GGMLType::Q4_K));
    writer.add_tensor(std::move(tensor));
}

bool write_fixture(const char* path, uint32_t block_count = 2, uint32_t vector_count = 2,
                   bool duplicate_block_count = false, bool wrong_block_count_type = false,
                   bool unsupported_tensor = false, bool out_of_range_layer = false) {
    gguf_writer::Writer writer;
    writer.kv_str("general.architecture", "fixture");
    writer.kv_u32("general.quantization_version", 2);
    if (wrong_block_count_type) writer.kv_str("fixture.block_count", "two");
    else writer.kv_u32("fixture.block_count", block_count);
    if (duplicate_block_count) writer.kv_u32("other.block_count", block_count);
    std::vector<uint32_t> heads(vector_count, 2);
    std::vector<uint32_t> keys(vector_count, 512);
    std::vector<uint32_t> values(vector_count, 512);
    std::vector<uint32_t> windows(vector_count, 0);
    std::vector<uint32_t> rotary(vector_count, 512);
    if (!heads.empty()) heads[0] = 8;
    if (!keys.empty()) keys[0] = 256;
    if (!values.empty()) values[0] = 256;
    if (!windows.empty()) windows[0] = 1024;
    if (!rotary.empty()) rotary[0] = 256;
    writer.kv_arr_u32("fixture.attention.head_count_kv", heads);
    writer.kv_arr_u32("fixture.attention.key_length", keys);
    writer.kv_arr_u32("fixture.attention.value_length", values);
    writer.kv_arr_u32("fixture.sliding_window", windows);
    writer.kv_arr_u32("fixture.rope.dimension_count", rotary);
    writer.kv_u32("fixture.expert_count", 128);
    add_f32(writer, "token_embd.weight", {4, 2});
    add_f32(writer, "output_norm.weight", {4});
    add_q4k(writer, "blk.0.attn_q.weight", {256, 4});
    add_q4k(writer, "blk.0.ffn_down_exps.weight", {256, 4, 128});
    add_q4k(writer, out_of_range_layer ? "blk.2.attn_q.weight" : "blk.1.attn_q.weight", {256, 4});
    if (unsupported_tensor) {
        gguf_writer::TensorDecl tensor;
        tensor.name = "blk.1.unsupported.weight";
        tensor.dims = {256, 4};
        tensor.type = static_cast<uint32_t>(GGMLType::Q2_K);
        tensor.data.resize(4 * bytes_per_block(GGMLType::Q2_K));
        writer.add_tensor(std::move(tensor));
    }
    return writer.write_file(path);
}

bool build_index(const char* path, ArtifactIndex& output) {
    auto artifacts = ArtifactSet::load_single_file(path);
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return false;
    auto* set = std::get_if<ArtifactSet>(&artifacts);
    auto view = set->view(ArtifactId{0});
    if (!std::holds_alternative<PackageView>(view)) return false;
    auto result = build_gguf_artifact_index(*std::get_if<PackageView>(&view));
    if (!std::holds_alternative<ArtifactIndex>(result)) return false;
    output = std::move(*std::get_if<ArtifactIndex>(&result));
    return true;
}

const ArtifactFact* find_fact(const ArtifactIndex& index, CanonicalFactKey key) {
    const auto found = std::find_if(index.metadata_facts().begin(), index.metadata_facts().end(),
                                    [&](const ArtifactFact& fact) { return fact.key == key; });
    return found == index.metadata_facts().end() ? nullptr : &*found;
}

void test_physical_index_preserves_structure() {
    const char* path = "/private/tmp/laplace-gguf-index-fixture.gguf";
    CHECK(write_fixture(path));
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (auto* set = std::get_if<ArtifactSet>(&artifacts)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) {
            auto result = build_gguf_artifact_index(*package);
            CHECK(std::holds_alternative<ArtifactIndex>(result));
            if (auto* index = std::get_if<ArtifactIndex>(&result)) {
                CHECK(index->tensors().size() == 5);
                CHECK(index->metadata_facts().size() == gguf_fact_keys::descriptors.size());
                const ArtifactFact* block_count = find_fact(*index, gguf_fact_keys::block_count);
                CHECK(block_count != nullptr);
                if (block_count) {
                    CHECK(block_count->state == ArtifactFactState::Present);
                    CHECK(std::holds_alternative<uint64_t>(block_count->value));
                    CHECK(std::get<uint64_t>(block_count->value) == 2);
                }
                const ArtifactFact* context = find_fact(*index, gguf_fact_keys::context_length);
                CHECK(context != nullptr && context->state == ArtifactFactState::Missing);
                for (const ArtifactTensorRecord& tensor : index->tensors()) {
                    CHECK(tensor.role_evidence.empty());
                    CHECK(!tensor.planes.empty());
                    CHECK(tensor.coordinate.layer == UINT32_MAX);
                    CHECK(tensor.coordinate.slot == UINT32_MAX);
                    CHECK(tensor.coordinate.expert == UINT32_MAX);
                    CHECK(tensor.coordinate.bank_axis == UINT8_MAX);
                }
                CHECK(index->tensors()[0].layout.strides[0] == 1);
                CHECK(index->tensors()[0].layout.strides[1] == 4);
                CHECK(index->tensors()[0].quantization.kind == QuantizationKind::None);
                CHECK(index->tensors()[0].quantization.required_plane_mask == 0);
                CHECK(index->tensors()[2].format.encoding == ArtifactPhysicalEncoding::Q4_K);
                CHECK(index->tensors()[2].format.block_elements == 256);
                CHECK(index->tensors()[2].format.block_bytes == 144);
                CHECK(index->tensors()[2].quantization.required_plane_mask ==
                      artifact_plane_mask(PlaneKind::Values));
            }
        }
    }
    remove(path);
}

void test_all_supported_physical_descriptors() {
    const char* path = "/private/tmp/laplace-gguf-index-formats.gguf";
    gguf_writer::Writer writer;
    writer.kv_str("general.architecture", "fixture");
    writer.kv_u32("general.quantization_version", 2);
    add_f32(writer, "format.f32", {4, 2});
    gguf_writer::TensorDecl f16;
    f16.name = "format.f16";
    f16.dims = {4, 2};
    f16.type = static_cast<uint32_t>(GGMLType::F16);
    f16.data.resize(16);
    writer.add_tensor(std::move(f16));
    gguf_writer::TensorDecl q40;
    q40.name = "format.q40";
    q40.dims = {32, 2};
    q40.type = static_cast<uint32_t>(GGMLType::Q4_0);
    q40.data.resize(2 * 18);
    writer.add_tensor(std::move(q40));
    add_q4k(writer, "format.q4k", {256, 2});
    gguf_writer::TensorDecl q50;
    q50.name = "format.q50";
    q50.dims = {32, 2};
    q50.type = static_cast<uint32_t>(GGMLType::Q5_0);
    q50.data.resize(2 * 22);
    writer.add_tensor(std::move(q50));
    gguf_writer::TensorDecl q6k;
    q6k.name = "format.q6k";
    q6k.dims = {256, 2};
    q6k.type = static_cast<uint32_t>(GGMLType::Q6_K);
    q6k.data.resize(2 * 210);
    writer.add_tensor(std::move(q6k));
    gguf_writer::TensorDecl q80;
    q80.name = "format.q80";
    q80.dims = {32, 2};
    q80.type = static_cast<uint32_t>(GGMLType::Q8_0);
    q80.data.resize(2 * 34);
    writer.add_tensor(std::move(q80));
    CHECK(writer.write_file(path));
    ArtifactIndex index;
    CHECK(build_index(path, index));
    const ArtifactPhysicalEncoding encodings[] = {
        ArtifactPhysicalEncoding::F32, ArtifactPhysicalEncoding::F16,
        ArtifactPhysicalEncoding::Q4_0,
        ArtifactPhysicalEncoding::Q4_K, ArtifactPhysicalEncoding::Q5_0,
        ArtifactPhysicalEncoding::Q6_K, ArtifactPhysicalEncoding::Q8_0,
    };
    const uint32_t block_elements[] = {1, 1, 32, 256, 32, 256, 32};
    const uint32_t block_bytes[] = {4, 2, 18, 144, 22, 210, 34};
    CHECK(index.tensors().size() == std::size(encodings));
    for (size_t i = 0; i != std::size(encodings) && i < index.tensors().size(); ++i) {
        CHECK(index.tensors()[i].format.encoding == encodings[i]);
        CHECK(index.tensors()[i].format.block_elements == block_elements[i]);
        CHECK(index.tensors()[i].format.block_bytes == block_bytes[i]);
    }
    if (index.tensors().size() == 7) {
        const auto& q40 = index.tensors()[2].format;
        const auto& q4 = index.tensors()[3].format;
        const auto& q50 = index.tensors()[4].format;
        const auto& q6 = index.tensors()[5].format;
        const auto& q8 = index.tensors()[6].format;
        CHECK(q40.scale_type == ArtifactScalarType::F16 && q40.zero_type == ArtifactScalarType::None);
        CHECK(q4.scale_type == ArtifactScalarType::F16 && q4.zero_type == ArtifactScalarType::F16);
        CHECK(q4.subscale_type == ArtifactScalarType::Packed && q4.subscale_bytes == 12);
        CHECK(q50.scale_type == ArtifactScalarType::F16 && q50.zero_type == ArtifactScalarType::None);
        CHECK(q6.scale_type == ArtifactScalarType::F16 && q6.zero_type == ArtifactScalarType::None);
        CHECK(q6.subscale_type == ArtifactScalarType::I8 && q6.subscale_bytes == 16);
        CHECK(q8.scale_type == ArtifactScalarType::F16 && q8.zero_type == ArtifactScalarType::None);
    }
    remove(path);
}

void test_rejects_ambiguous_or_malformed_metadata() {
    struct Case {
        const char* path;
        uint32_t block_count;
        uint32_t vector_count;
        bool duplicate;
        bool wrong_type;
        bool out_of_range;
    };
    const Case cases[] = {
        {"/private/tmp/laplace-gguf-index-29.gguf", 30, 29, false, false, false},
        {"/private/tmp/laplace-gguf-index-31.gguf", 30, 31, false, false, false},
        {"/private/tmp/laplace-gguf-index-ambiguous.gguf", 2, 2, true, false, false},
        {"/private/tmp/laplace-gguf-index-layer-range.gguf", 2, 2, false, false, true},
    };
    for (const Case& test : cases) {
        CHECK(write_fixture(test.path, test.block_count, test.vector_count,
                            test.duplicate, test.wrong_type, false, test.out_of_range));
        auto artifacts = ArtifactSet::load_single_file(test.path);
        CHECK(std::holds_alternative<ArtifactSet>(artifacts));
        if (auto* set = std::get_if<ArtifactSet>(&artifacts)) {
            auto view = set->view(ArtifactId{0});
            CHECK(std::holds_alternative<PackageView>(view));
            if (auto* package = std::get_if<PackageView>(&view)) {
                auto result = build_gguf_artifact_index(*package);
                CHECK(std::holds_alternative<ArtifactIndex>(result));
                if (auto* index = std::get_if<ArtifactIndex>(&result)) {
                    CHECK(index->metadata_facts().size() == gguf_fact_keys::descriptors.size());
                    const ArtifactFact* block_count = find_fact(*index, gguf_fact_keys::block_count);
                    CHECK(block_count != nullptr);
                    if (block_count) {
                        CHECK(block_count->state == (test.duplicate ? ArtifactFactState::Ambiguous
                                                                    : ArtifactFactState::Present));
                    }
                }
            }
        }
        remove(test.path);
    }
}

void test_unsupported_physical_format_fails_closed() {
    const char* path = "/private/tmp/laplace-gguf-index-unsupported.gguf";
    CHECK(write_fixture(path, 2, 2, false, false, true));
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (auto* set = std::get_if<ArtifactSet>(&artifacts)) {
        auto view = set->view(ArtifactId{0});
        if (auto* package = std::get_if<PackageView>(&view)) {
            auto result = build_gguf_artifact_index(*package);
            CHECK(std::holds_alternative<CompatibilityReport>(result));
            if (auto* report = std::get_if<CompatibilityReport>(&result)) {
                CHECK(report->code == CompatibilityError::IR_QUANTIZATION_UNSUPPORTED);
            }
        }
    }
    remove(path);
}

void test_missing_semantics_remain_physical() {
    const char* path = "/private/tmp/laplace-gguf-index-fixture-missing.gguf";
    CHECK(write_fixture(path));
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (auto* set = std::get_if<ArtifactSet>(&artifacts)) {
        auto view = set->view(ArtifactId{0});
        if (auto* package = std::get_if<PackageView>(&view)) {
            auto result = build_gguf_artifact_index(*package);
            CHECK(std::holds_alternative<ArtifactIndex>(result));
            if (auto* index = std::get_if<ArtifactIndex>(&result)) {
                CHECK(index->aliases().empty());
                CHECK(index->diagnostics().size() >= index->tensors().size());
            }
        }
    }
    remove(path);
}

void test_reviewed_physical_boundary_failures() {
    const char* path = "/private/tmp/laplace-gguf-index-review-red.gguf";
    CHECK(write_fixture(path));
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (auto* set = std::get_if<ArtifactSet>(&artifacts)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) {
            auto result = build_gguf_artifact_index(*package);
            CHECK(std::holds_alternative<ArtifactIndex>(result));
            if (auto* index = std::get_if<ArtifactIndex>(&result)) {
                CHECK(index->metadata_facts().size() == gguf_fact_keys::descriptors.size());
                for (const auto& tensor : index->tensors()) {
                    CHECK(tensor.coordinate.layer == UINT32_MAX);
                    CHECK(tensor.coordinate.bank_axis == UINT8_MAX);
                    if (tensor.logical_dimensions == std::vector<uint64_t>{4, 2}) {
                        CHECK(tensor.layout.strides[0] == 1);
                        CHECK(tensor.layout.strides[1] == 4);
                    }
                    if (tensor.logical_dimensions == std::vector<uint64_t>{256, 4}) {
                        CHECK(tensor.planes[0].storage_type == ArtifactScalarType::Packed);
                    }
                }
            }
        }
    }
    remove(path);
}

void test_diagnostic_spelling_is_not_normalized_identity() {
    const char* first_path = "/private/tmp/laplace-gguf-index-spelling-a.gguf";
    const char* second_path = "/private/tmp/laplace-gguf-index-spelling-b.gguf";
    auto build_one = [](const char* path, const char* tensor_name) -> std::optional<ArtifactIndex> {
        gguf_writer::Writer writer;
        writer.kv_str("general.architecture", "fixture");
        add_f32(writer, tensor_name, {2, 2});
        CHECK(writer.write_file(path));
        auto artifacts = ArtifactSet::load_single_file(path);
        CHECK(std::holds_alternative<ArtifactSet>(artifacts));
        if (auto* set = std::get_if<ArtifactSet>(&artifacts)) {
            auto view = set->view(ArtifactId{0});
            CHECK(std::holds_alternative<PackageView>(view));
            if (auto* package = std::get_if<PackageView>(&view)) {
                auto result = build_gguf_artifact_index(*package);
                CHECK(std::holds_alternative<ArtifactIndex>(result));
                if (auto* index = std::get_if<ArtifactIndex>(&result)) return std::move(*index);
            }
        }
        return std::nullopt;
    };
    auto first = build_one(first_path, "dense.a");
    auto second = build_one(second_path, "dense.b");
    CHECK(first.has_value() && second.has_value());
    if (first && second) {
        CHECK(first->normalized_digest() == second->normalized_digest());
        CHECK(first->provenance_digest() != second->provenance_digest());
    }
    remove(first_path);
    remove(second_path);
}

void test_bounded_real_physical_index(const char* path) {
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (auto* set = std::get_if<ArtifactSet>(&artifacts)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) {
            auto result = build_gguf_artifact_index(*package);
            CHECK(std::holds_alternative<ArtifactIndex>(result));
            if (auto* index = std::get_if<ArtifactIndex>(&result)) {
                CHECK(!index->tensors().empty());
                CHECK(index->tensors().size() <= 20000);
                for (const ArtifactTensorRecord& tensor : index->tensors()) {
                    CHECK(tensor.role_evidence.empty());
                    CHECK(!tensor.planes.empty());
                    CHECK(tensor.planes[0].source.offset <= package->bytes().size());
                    CHECK(tensor.planes[0].source.length <=
                          package->bytes().size() - tensor.planes[0].source.offset);
                }
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    test_physical_index_preserves_structure();
    test_all_supported_physical_descriptors();
    test_missing_semantics_remain_physical();
    test_rejects_ambiguous_or_malformed_metadata();
    test_unsupported_physical_format_fails_closed();
    test_reviewed_physical_boundary_failures();
    test_diagnostic_spelling_is_not_normalized_identity();
    if (argc == 2) test_bounded_real_physical_index(argv[1]);
    return test_summary("test_gguf_index");
}
