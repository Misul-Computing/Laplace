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

    auto result = load_safetensors_physical_package(directory);
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
    auto result = load_safetensors_physical_package(directory);
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
    auto result = load_safetensors_physical_package(directory);
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
    auto result = load_safetensors_physical_package(directory);
    const auto* report = std::get_if<CompatibilityReport>(&result);
    CHECK(report != nullptr);
    if (report) CHECK(report->code == CompatibilityError::IMPORT_EXECUTABLE_CODE_REQUIRED);
    remove_tree(directory, {"config.json", "model.safetensors.index.json", "model.safetensors"});
}

void test_directory_mixed_source_dtype_is_indexed_without_format_policy() {
    const std::string directory = make_directory();
    write_text(directory + "/config.json", "{}");
    write_text(directory + "/model.safetensors.index.json",
               "{\"weight_map\":{\"raw.a\":\"model.safetensors\"}}");
    write_bytes(directory + "/model.safetensors",
                safetensors_file("raw.a", "pt", "BF16", "[2]", 4));
    auto result = load_safetensors_physical_package(directory);
    CHECK(std::holds_alternative<MlxPhysicalPackage>(result));
    if (const auto* package = std::get_if<MlxPhysicalPackage>(&result)) {
        const ArtifactIndex& index = package->physical_index();
        CHECK(index.tensors().size() == 1);
        if (index.tensors().size() == 1) {
            CHECK(index.tensors()[0].logical_type == ArtifactScalarType::BF16);
            CHECK(index.tensors()[0].planes.size() == 1);
            CHECK(index.tensors()[0].planes[0].storage_type ==
                  ArtifactScalarType::BF16);
            CHECK(index.tensors()[0].format.encoding ==
                  ArtifactPhysicalEncoding::Unknown);
        }
    }
    remove_tree(directory, {"config.json", "model.safetensors.index.json", "model.safetensors"});
}

void test_directory_preserves_c_row_major_matrix_axes() {
    const std::string directory = make_directory();
    write_text(directory + "/config.json", "{}");
    write_text(directory + "/model.safetensors.index.json",
               "{\"weight_map\":{\"raw.weight\":\"model.safetensors\"}}");
    write_bytes(directory + "/model.safetensors",
                safetensors_file("raw.weight", "mlx", "F16", "[2,3]", 12));
    auto result = load_safetensors_physical_package(directory);
    CHECK(std::holds_alternative<MlxPhysicalPackage>(result));
    if (const auto* package = std::get_if<MlxPhysicalPackage>(&result)) {
        const ArtifactIndex& index = package->physical_index();
        CHECK(index.tensors().size() == 1);
        if (index.tensors().size() == 1) {
            const ArtifactTensorRecord& tensor = index.tensors()[0];
            CHECK(tensor.logical_dimensions == std::vector<uint64_t>({2, 3}));
            CHECK(tensor.layout.strides[0] == 3);
            CHECK(tensor.layout.strides[1] == 1);
            CHECK(tensor.axis.source_rank == 2);
            CHECK(tensor.axis.source_axis_order[0] == 0);
            CHECK(tensor.axis.source_axis_order[1] == 1);
            CHECK(tensor.axis.row_stride_bytes == 6);
            CHECK(tensor.format.encoding == ArtifactPhysicalEncoding::F16);
            CHECK(tensor.format.value_type == ArtifactScalarType::F16);
            CHECK(tensor.format.block_elements == 1);
            CHECK(tensor.format.block_bytes == 2);
        }
    }
    remove_tree(directory, {"config.json", "model.safetensors.index.json",
                            "model.safetensors"});
}

PackageView owned_declaration(const std::string& text, uint32_t id) {
    auto view = ArtifactSet::make_owned_blob(
        ArtifactId{id}, ArtifactRole::Sidecar,
        std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(text.data()), text.size()));
    CHECK(std::holds_alternative<PackageView>(view));
    return std::get<PackageView>(std::move(view));
}

void test_declared_storage_groups_resolve_once_then_drop_source_names() {
    const std::string directory = make_directory();
    const std::string first =
        "{\"tensor_storage\":{\"diagnostic.one\":{\"stored_tensors\":{"
        "\"raw.a\":{},\"raw.b\":{}}}}}";
    write_text(directory + "/config.json", "{}");
    write_text(directory + "/quantization_config.json", first);
    write_text(directory + "/model.safetensors.index.json",
               "{\"weight_map\":{"
               "\"raw.a\":\"model-00001-of-00002.safetensors\","
               "\"raw.b\":\"model-00002-of-00002.safetensors\"}}");
    write_bytes(directory + "/model-00001-of-00002.safetensors",
                safetensors_file("raw.a", "pt", "I16", "[2]", 4));
    write_bytes(directory + "/model-00002-of-00002.safetensors",
                safetensors_file("raw.b", "pt", "I32", "[]", 4));
    auto loaded = load_safetensors_physical_package(directory);
    CHECK(std::holds_alternative<MlxPhysicalPackage>(loaded));
    if (const auto* package = std::get_if<MlxPhysicalPackage>(&loaded)) {
        CHECK(package->storage_resources() != nullptr);
        if (package->storage_resources()) {
            CHECK(package->storage_resources()->declared_resource_count() == 1);
            CHECK(package->storage_resources()->resources().size() == 1);
        }
        const PackageView declaration = owned_declaration(first, 20);
        auto compiled = compile_safetensors_storage_resources(
            package->physical_index(), declaration);
        CHECK(std::holds_alternative<VerifiedSafeTensorsStorageResources>(compiled));
        if (const auto* resources =
                std::get_if<VerifiedSafeTensorsStorageResources>(&compiled)) {
            CHECK(resources->declared_resource_count() == 1);
            CHECK(resources->resources().size() == 1);
            CHECK(resources->resources()[0].id == 0);
            CHECK(resources->resources()[0].tensor_ids ==
                  std::vector<uint32_t>({0, 1}));
            CHECK(resources->declaration_digest() == declaration.digest());

            const PackageView renamed = owned_declaration(
                "{\"tensor_storage\":{\"diagnostic.two\":{\"stored_tensors\":{"
                "\"raw.b\":{},\"raw.a\":{}}}}}", 21);
            auto renamed_result = compile_safetensors_storage_resources(
                package->physical_index(), renamed);
            CHECK(std::holds_alternative<VerifiedSafeTensorsStorageResources>(
                renamed_result));
            if (const auto* renamed_resources =
                    std::get_if<VerifiedSafeTensorsStorageResources>(
                        &renamed_result)) {
                CHECK(renamed_resources->canonical_digest() ==
                      resources->canonical_digest());
                CHECK(renamed_resources->declaration_digest() !=
                      resources->declaration_digest());
            }

            const PackageView regrouped = owned_declaration(
                "{\"tensor_storage\":{"
                "\"x\":{\"stored_tensors\":{\"raw.a\":{}}},"
                "\"y\":{\"stored_tensors\":{\"raw.b\":{}}}}}", 24);
            auto regrouped_result = compile_safetensors_storage_resources(
                package->physical_index(), regrouped);
            CHECK(std::holds_alternative<VerifiedSafeTensorsStorageResources>(
                regrouped_result));
            if (const auto* regrouped_resources =
                    std::get_if<VerifiedSafeTensorsStorageResources>(
                        &regrouped_result)) {
                CHECK(regrouped_resources->resources().size() == 2);
                CHECK(regrouped_resources->canonical_digest() !=
                      resources->canonical_digest());
            }
        }

        const PackageView missing = owned_declaration(
            "{\"tensor_storage\":{\"x\":{\"stored_tensors\":{"
            "\"raw.absent\":{}}}}}", 22);
        auto missing_result = compile_safetensors_storage_resources(
            package->physical_index(), missing);
        const auto* missing_report =
            std::get_if<CompatibilityReport>(&missing_result);
        CHECK(missing_report != nullptr);
        if (missing_report) {
            CHECK(missing_report->code ==
                  CompatibilityError::IMPORT_TENSOR_UNMAPPED);
        }

        const PackageView replayed = owned_declaration(
            "{\"tensor_storage\":{"
            "\"x\":{\"stored_tensors\":{\"raw.a\":{}}},"
            "\"y\":{\"stored_tensors\":{\"raw.a\":{}}}}}", 23);
        auto replayed_result = compile_safetensors_storage_resources(
            package->physical_index(), replayed);
        const auto* replayed_report =
            std::get_if<CompatibilityReport>(&replayed_result);
        CHECK(replayed_report != nullptr);
        if (replayed_report) {
            CHECK(replayed_report->code ==
                  CompatibilityError::IMPORT_TENSOR_DUPLICATE);
        }
    }
    remove_tree(directory, {"config.json", "quantization_config.json",
                            "model.safetensors.index.json",
                            "model-00001-of-00002.safetensors",
                            "model-00002-of-00002.safetensors"});
}

void test_storage_declaration_failure_closes_directory_load() {
    const std::string directory = make_directory();
    write_text(directory + "/config.json", "{}");
    write_text(directory + "/quantization_config.json",
               "{\"tensor_storage\":{\"x\":{\"stored_tensors\":{"
               "\"absent\":{}}}}}");
    write_text(directory + "/model.safetensors.index.json",
               "{\"weight_map\":{\"raw.a\":\"model.safetensors\"}}");
    write_bytes(directory + "/model.safetensors",
                safetensors_file("raw.a", "pt", "I16", "[2]", 4));
    auto loaded = load_safetensors_physical_package(directory);
    const auto* report = std::get_if<CompatibilityReport>(&loaded);
    CHECK(report != nullptr);
    if (report) {
        CHECK(report->code == CompatibilityError::IMPORT_TENSOR_UNMAPPED);
    }
    remove_tree(directory, {"config.json", "quantization_config.json",
                            "model.safetensors.index.json",
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
    auto result = load_safetensors_physical_package(file);
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

void inventory(const char* path, const char* declaration_path = nullptr) {
    auto result = load_safetensors_physical_package(path);
    if (const auto* report = std::get_if<CompatibilityReport>(&result)) {
        std::printf("SafeTensors physical inventory refused: code=%u detail=%s\n",
                    static_cast<unsigned>(report->code), report->detail.c_str());
        CHECK(false);
        return;
    }
    const auto& package = std::get<MlxPhysicalPackage>(result);
    const auto& index = package.physical_index();
    const CompatibilityReport semantic = package.semantic_refusal();
    std::printf("SafeTensors physical inventory: artifacts=%zu tensors=%zu digest=%s "
                "semantic=%u detail=%s\n",
                index.artifacts().size(), index.tensors().size(), index.digest().hex().c_str(),
                static_cast<unsigned>(semantic.code), semantic.detail.c_str());
    CHECK(!index.tensors().empty());
    if (const auto* resources = package.storage_resources()) {
        size_t bound_tensors = 0;
        for (const auto& resource : resources->resources()) {
            bound_tensors += resource.tensor_ids.size();
        }
        CHECK(bound_tensors == index.tensors().size());
        std::printf(
            "SafeTensors automatic storage resources: declared=%u total=%zu "
            "bound_tensors=%zu canonical=%s declaration=%s\n",
            resources->declared_resource_count(), resources->resources().size(),
            bound_tensors, resources->canonical_digest().hex().c_str(),
            resources->declaration_digest().hex().c_str());
    }
    if (declaration_path == nullptr) return;

    auto declaration_set = ArtifactSet::load_single_file(declaration_path);
    CHECK(std::holds_alternative<ArtifactSet>(declaration_set));
    if (const auto* set = std::get_if<ArtifactSet>(&declaration_set)) {
        auto declaration = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(declaration));
        if (const auto* view = std::get_if<PackageView>(&declaration)) {
            auto compiled = compile_safetensors_storage_resources(index, *view);
            CHECK(std::holds_alternative<VerifiedSafeTensorsStorageResources>(
                compiled));
            if (const auto* resources =
                    std::get_if<VerifiedSafeTensorsStorageResources>(&compiled)) {
                size_t bound_tensors = 0;
                for (const auto& resource : resources->resources()) {
                    bound_tensors += resource.tensor_ids.size();
                }
                CHECK(bound_tensors == index.tensors().size());
                std::printf(
                    "SafeTensors storage resources: declared=%u total=%zu "
                    "bound_tensors=%zu canonical=%s declaration=%s\n",
                    resources->declared_resource_count(),
                    resources->resources().size(), bound_tensors,
                    resources->canonical_digest().hex().c_str(),
                    resources->declaration_digest().hex().c_str());
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 || argc == 3) {
        inventory(argv[1], argc == 3 ? argv[2] : nullptr);
        return test_summary("test_mlx_package");
    }
    test_closed_sharded_package_is_role_free();
    test_unsafe_weight_map_leaf_is_rejected();
    test_unreferenced_shard_is_rejected();
    test_executable_model_file_is_rejected();
    test_directory_mixed_source_dtype_is_indexed_without_format_policy();
    test_directory_preserves_c_row_major_matrix_axes();
    test_declared_storage_groups_resolve_once_then_drop_source_names();
    test_storage_declaration_failure_closes_directory_load();
    test_single_file_physical_slice_never_claims_semantics();
    return test_summary("test_mlx_package");
}
