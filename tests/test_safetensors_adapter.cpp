#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <variant>
#include <vector>

#include "artifact_index.h"
#include "artifact_set.h"
#include "safetensors.h"
#include "safetensors_adapter.h"
#include "test_util.h"

using namespace Laplace;

namespace {

std::vector<uint8_t> wire_file(const std::string& header,
                               const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> bytes(8);
    const uint64_t length = header.size();
    for (size_t index = 0; index != 8; ++index) {
        bytes[index] = static_cast<uint8_t>(length >> (8 * index));
    }
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    return bytes;
}

std::string write_temporary(const std::vector<uint8_t>& bytes) {
    char path[] = "/private/tmp/laplace-safetensors-adapter-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd >= 0) {
        CHECK(write(fd, bytes.data(), bytes.size()) ==
              static_cast<ssize_t>(bytes.size()));
        close(fd);
    }
    return path;
}

std::variant<PackageView, CompatibilityReport>
load_view(const std::vector<uint8_t>& bytes, std::string& path) {
    path = write_temporary(bytes);
    auto loaded = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    if (auto* set = std::get_if<ArtifactSet>(&loaded)) return set->view(ArtifactId{0});
    return std::get<CompatibilityReport>(std::move(loaded));
}

void test_role_free_physical_index() {
    const std::string header =
        "{\"norm\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
        "\"weight\":{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[4,16]}}";
    std::string path;
    auto loaded = load_view(wire_file(header, std::vector<uint8_t>(16, 7)), path);
    CHECK(std::holds_alternative<PackageView>(loaded));
    if (const auto* view = std::get_if<PackageView>(&loaded)) {
        auto result = build_safetensors_artifact_index(*view);
        CHECK(std::holds_alternative<ArtifactIndex>(result));
        if (const auto* index = std::get_if<ArtifactIndex>(&result)) {
            CHECK(index->artifacts().size() == 1);
            CHECK(index->metadata_facts().empty());
            CHECK(index->package_facts().empty());
            CHECK(index->tensors().size() == 2);
            CHECK(index->diagnostics().size() == 2);
            CHECK(index->tensors()[0].role_evidence.empty());
            CHECK(index->tensors()[1].role_evidence.empty());
            CHECK(index->tensors()[0].quantization.kind == QuantizationKind::None);
            CHECK(index->tensors()[0].quantization.required_plane_mask == 0);
            CHECK(index->tensors()[1].quantization.required_plane_mask == 0);
            CHECK(index->tensors()[1].axis.source_rank == 2);
            CHECK(index->tensors()[1].axis.source_axis_order[0] == 0);
            CHECK(index->tensors()[1].axis.source_axis_order[1] == 1);
            CHECK(index->tensors()[1].format.encoding == ArtifactPhysicalEncoding::Unknown);
            CHECK(index->tensors()[0].planes[0].alignment == 1);
            CHECK(index->tensors()[0].planes[0].source.offset == 8 + header.size());
            CHECK(index->tensors()[1].logical_type == ArtifactScalarType::BF16);
            CHECK(index->tensors()[1].logical_dimensions ==
                  std::vector<uint64_t>({2, 3}));
            CHECK(index->tensors()[1].layout.strides[0] == 3);
            CHECK(index->tensors()[1].layout.strides[1] == 1);
        }
    }
    unlink(path.c_str());
}

void test_unsupported_wire_dtype_fails_after_parse() {
    const std::string header =
        "{\"packed\":{\"dtype\":\"F8_E4M3\",\"shape\":[4],\"data_offsets\":[0,4]}}";
    std::string path;
    auto loaded = load_view(wire_file(header, std::vector<uint8_t>(4, 1)), path);
    CHECK(std::holds_alternative<PackageView>(loaded));
    if (const auto* view = std::get_if<PackageView>(&loaded)) {
        auto result = build_safetensors_artifact_index(*view);
        const auto* report = std::get_if<CompatibilityReport>(&result);
        CHECK(report != nullptr);
        if (report) {
            CHECK(report->code == CompatibilityError::IR_QUANTIZATION_UNSUPPORTED);
            CHECK(report->artifact_id == ArtifactId{0});
            CHECK(report->tensor_id == 0);
        }
    }
    unlink(path.c_str());
}

void test_rank_beyond_physical_abi_fails_closed() {
    const std::string header =
        "{\"rank9\":{\"dtype\":\"U8\",\"shape\":[1,1,1,1,1,1,1,1,1],"
        "\"data_offsets\":[0,1]}}";
    std::string path;
    auto loaded = load_view(wire_file(header, {0}), path);
    CHECK(std::holds_alternative<PackageView>(loaded));
    if (const auto* view = std::get_if<PackageView>(&loaded)) {
        auto result = build_safetensors_artifact_index(*view);
        const auto* report = std::get_if<CompatibilityReport>(&result);
        CHECK(report != nullptr);
        if (report) CHECK(report->code == CompatibilityError::IR_SHAPE_MISMATCH);
    }
    unlink(path.c_str());
}

void test_existing_file(const char* path) {
    auto loaded = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    if (auto* set = std::get_if<ArtifactSet>(&loaded)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (const auto* package = std::get_if<PackageView>(&view)) {
            auto result = build_safetensors_artifact_index(*package);
            CHECK(std::holds_alternative<ArtifactIndex>(result));
            if (const auto* index = std::get_if<ArtifactIndex>(&result)) {
                CHECK(!index->tensors().empty());
                std::printf("existing SafeTensors physical index: tensors=%zu bytes=%zu digest=%s\n",
                            index->tensors().size(), package->bytes().size(),
                            index->digest().hex().c_str());
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2) {
        test_existing_file(argv[1]);
        return test_summary("test_safetensors_adapter");
    }
    test_role_free_physical_index();
    test_unsupported_wire_dtype_fails_after_parse();
    test_rank_beyond_physical_abi_fails_closed();
    return test_summary("test_safetensors_adapter");
}
