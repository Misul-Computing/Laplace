#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <variant>
#include <vector>

#include "mlx_package.h"
#include "test_util.h"

using namespace Laplace;

namespace {

std::string make_directory() {
    char path[] = "/private/tmp/laplace-mlx-package-XXXXXX";
    CHECK(mkdtemp(path) != nullptr);
    return path;
}

void write_bytes(const std::string& path, const std::vector<uint8_t>& bytes) {
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(fd >= 0);
    if (fd >= 0) {
        CHECK(write(fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size()));
        close(fd);
    }
}

void write_text(const std::string& path, const std::string& text) {
    write_bytes(path, {text.begin(), text.end()});
}

std::vector<uint8_t> safetensors_file(const std::string& name,
                                       const std::string& format = "mlx",
                                       const std::string& dtype = "F16",
                                       const std::string& shape = "[2]",
                                       size_t payload_size = 4) {
    const std::string header =
        "{\"__metadata__\":{\"format\":\"" + format + "\"},\"" + name +
        "\":{\"dtype\":\"" + dtype + "\",\"shape\":" + shape +
        ",\"data_offsets\":[0," + std::to_string(payload_size) + "]}}";
    std::vector<uint8_t> bytes(8);
    for (size_t index = 0; index != 8; ++index) {
        bytes[index] = static_cast<uint8_t>(header.size() >> (8 * index));
    }
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.resize(bytes.size() + payload_size, 7);
    return bytes;
}

void remove_tree(const std::string& directory, const std::vector<std::string>& leaves) {
    for (const std::string& leaf : leaves) unlink((directory + "/" + leaf).c_str());
    rmdir(directory.c_str());
}

void test_closed_sharded_package_is_role_free() {
    const std::string directory = make_directory();
    write_text(directory + "/config.json", "{\"model_type\":\"diagnostic-only\"}");
    write_text(directory + "/model.safetensors.index.json",
               "{\"metadata\":{\"total_size\":8},\"weight_map\":{"
               "\"raw.b\":\"model-00002-of-00002.safetensors\","
               "\"raw.a\":\"model-00001-of-00002.safetensors\"}}");
    write_bytes(directory + "/model-00001-of-00002.safetensors", safetensors_file("raw.a"));
    write_bytes(directory + "/model-00002-of-00002.safetensors", safetensors_file("raw.b"));

    auto result = load_mlx_physical_package(directory);
    CHECK(std::holds_alternative<MlxPhysicalPackage>(result));
    if (const auto* report = std::get_if<CompatibilityReport>(&result)) {
        std::printf("closed-shard diagnostic: code=%u detail=%s\n",
                    static_cast<unsigned>(report->code), report->detail.c_str());
    }
    if (const auto* package = std::get_if<MlxPhysicalPackage>(&result)) {
        const ArtifactIndex& index = package->physical_index();
        CHECK(index.artifacts().size() == 4);
        CHECK(index.tensors().size() == 2);
        CHECK(index.tensors()[0].role_evidence.empty());
        CHECK(index.tensors()[1].role_evidence.empty());
        CHECK(index.tensors()[0].quantization.kind == QuantizationKind::None);
        CHECK(index.tensors()[0].quantization.required_plane_mask == 0);
        CHECK(index.tensors()[1].quantization.required_plane_mask == 0);
        CHECK(index.diagnostics().size() == 2);
        CHECK(index.artifacts()[0].role() == ArtifactRole::Primary);
        CHECK(index.artifacts()[1].role() == ArtifactRole::Sidecar);
        const CompatibilityReport semantic = package->semantic_refusal();
        CHECK(semantic.code == CompatibilityError::IMPORT_SEMANTICS_MISSING);
        CHECK(semantic.stage == CompatibilityStage::Semantic);
    }
    remove_tree(directory, {"config.json", "model.safetensors.index.json",
                            "model-00001-of-00002.safetensors",
                            "model-00002-of-00002.safetensors"});
}

void test_unsafe_weight_map_leaf_is_rejected() {
    const std::string directory = make_directory();
    write_text(directory + "/config.json", "{}");
    write_text(directory + "/model.safetensors.index.json",
               "{\"weight_map\":{\"raw.a\":\"../escape.safetensors\"}}");
    auto result = load_mlx_physical_package(directory);
    const auto* report = std::get_if<CompatibilityReport>(&result);
    CHECK(report != nullptr);
    if (report) CHECK(report->code == CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED);
    remove_tree(directory, {"config.json", "model.safetensors.index.json"});
}

void test_unreferenced_shard_is_rejected() {
    const std::string directory = make_directory();
    write_text(directory + "/config.json", "{}");
    write_text(directory + "/model.safetensors.index.json",
               "{\"weight_map\":{\"raw.a\":\"model.safetensors\"}}");
    write_bytes(directory + "/model.safetensors", safetensors_file("raw.a"));
    write_bytes(directory + "/unused.safetensors", safetensors_file("raw.extra"));
    auto result = load_mlx_physical_package(directory);
    const auto* report = std::get_if<CompatibilityReport>(&result);
    CHECK(report != nullptr);
    if (report) CHECK(report->code == CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED);
    remove_tree(directory, {"config.json", "model.safetensors.index.json", "model.safetensors",
                            "unused.safetensors"});
}

void test_executable_model_file_is_rejected() {
    const std::string directory = make_directory();
    write_text(directory + "/config.json", "{\"model_file\":\"model.py\"}");
    write_text(directory + "/model.safetensors.index.json",
               "{\"weight_map\":{\"raw.a\":\"model.safetensors\"}}");
    write_bytes(directory + "/model.safetensors", safetensors_file("raw.a"));
    auto result = load_mlx_physical_package(directory);
    const auto* report = std::get_if<CompatibilityReport>(&result);
    CHECK(report != nullptr);
    if (report) CHECK(report->code == CompatibilityError::IMPORT_EXECUTABLE_CODE_REQUIRED);
    remove_tree(directory, {"config.json", "model.safetensors.index.json", "model.safetensors"});
}

void test_directory_unsupported_dtype_fails_closed() {
    const std::string directory = make_directory();
    write_text(directory + "/config.json", "{}");
    write_text(directory + "/model.safetensors.index.json",
               "{\"weight_map\":{\"raw.a\":\"model.safetensors\"}}");
    write_bytes(directory + "/model.safetensors",
                safetensors_file("raw.a", "mlx", "BF16", "[2]", 4));
    auto result = load_mlx_physical_package(directory);
    const auto* report = std::get_if<CompatibilityReport>(&result);
    CHECK(report != nullptr);
    if (report) CHECK(report->code == CompatibilityError::IR_QUANTIZATION_UNSUPPORTED);
    remove_tree(directory, {"config.json", "model.safetensors.index.json", "model.safetensors"});
}

void test_directory_normalizes_c_row_major_matrix_axes() {
    const std::string directory = make_directory();
    write_text(directory + "/config.json", "{}");
    write_text(directory + "/model.safetensors.index.json",
               "{\"weight_map\":{\"raw.weight\":\"model.safetensors\"}}");
    write_bytes(directory + "/model.safetensors",
                safetensors_file("raw.weight", "mlx", "F16", "[2,3]", 12));
    auto result = load_mlx_physical_package(directory);
    CHECK(std::holds_alternative<MlxPhysicalPackage>(result));
    if (const auto* package = std::get_if<MlxPhysicalPackage>(&result)) {
        const ArtifactIndex& index = package->physical_index();
        CHECK(index.tensors().size() == 1);
        if (index.tensors().size() == 1) {
            const ArtifactTensorRecord& tensor = index.tensors()[0];
            CHECK(tensor.logical_dimensions == std::vector<uint64_t>({3, 2}));
            CHECK(tensor.layout.strides[0] == 1);
            CHECK(tensor.layout.strides[1] == 3);
            CHECK(tensor.axis.source_rank == 2);
            CHECK(tensor.axis.source_axis_order[0] == 0);
            CHECK(tensor.axis.source_axis_order[1] == 1);
            CHECK(tensor.axis.row_stride_bytes == 6);
            CHECK(tensor.format.encoding == ArtifactPhysicalEncoding::F16);
        }
    }
    remove_tree(directory, {"config.json", "model.safetensors.index.json",
                            "model.safetensors"});
}

void test_single_file_physical_slice_never_claims_semantics() {
    char path[] = "/private/tmp/laplace-mlx-single-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd >= 0) close(fd);
    const std::string file = std::string(path) + ".safetensors";
    rename(path, file.c_str());
    write_bytes(file, safetensors_file("raw.weight", "pt", "BF16", "[2,3]", 12));
    auto result = load_mlx_physical_package(file);
    CHECK(std::holds_alternative<MlxPhysicalPackage>(result));
    if (const auto* package = std::get_if<MlxPhysicalPackage>(&result)) {
        CHECK(package->physical_index().artifacts().size() == 1);
        CHECK(package->physical_index().tensors().size() == 1);
        CHECK(package->physical_index().tensors()[0].logical_type == ArtifactScalarType::BF16);
        CHECK(package->physical_index().tensors()[0].logical_dimensions ==
              std::vector<uint64_t>({2, 3}));
        CHECK(package->physical_index().tensors()[0].layout.strides[0] == 3);
        CHECK(package->physical_index().tensors()[0].layout.strides[1] == 1);
        CHECK(package->physical_index().tensors()[0].axis.source_rank == 2);
        CHECK(package->physical_index().tensors()[0].axis.source_axis_order[0] == 0);
        CHECK(package->physical_index().tensors()[0].axis.source_axis_order[1] == 1);
        CHECK(package->physical_index().tensors()[0].planes[0].storage_type ==
              ArtifactScalarType::BF16);
        CHECK(package->physical_index().tensors()[0].format.encoding == ArtifactPhysicalEncoding::Unknown);
        CHECK(package->physical_index().tensors()[0].quantization.required_plane_mask == 0);
        CHECK(package->semantic_refusal().code == CompatibilityError::IMPORT_SEMANTICS_MISSING);
    }
    unlink(file.c_str());
}

void inventory(const char* path) {
    auto result = load_mlx_physical_package(path);
    if (const auto* report = std::get_if<CompatibilityReport>(&result)) {
        std::printf("MLX physical inventory refused: code=%u detail=%s\n",
                    static_cast<unsigned>(report->code), report->detail.c_str());
        return;
    }
    const auto& package = std::get<MlxPhysicalPackage>(result);
    const auto& index = package.physical_index();
    std::printf("MLX physical inventory: artifacts=%zu tensors=%zu digest=%s semantic=%u\n",
                index.artifacts().size(), index.tensors().size(), index.digest().hex().c_str(),
                static_cast<unsigned>(package.semantic_refusal().code));
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2) {
        inventory(argv[1]);
        return test_summary("test_mlx_package");
    }
    test_closed_sharded_package_is_role_free();
    test_unsafe_weight_map_leaf_is_rejected();
    test_unreferenced_shard_is_rejected();
    test_executable_model_file_is_rejected();
    test_directory_unsupported_dtype_fails_closed();
    test_directory_normalizes_c_row_major_matrix_axes();
    test_single_file_physical_slice_never_claims_semantics();
    return test_summary("test_mlx_package");
}
