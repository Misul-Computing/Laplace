#include <array>
#include <algorithm>
#include <fcntl.h>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include "artifact_index.h"
#include "test_util.h"

using namespace Laplace;

namespace {

std::string temporary_path() {
    char path[] = "/private/tmp/laplace-index-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd >= 0) close(fd);
    return path;
}

void write_pattern(const std::string& path, uint8_t seed, size_t count) {
    std::vector<uint8_t> bytes(count);
    for (size_t index = 0; index != count; ++index) bytes[index] = static_cast<uint8_t>(seed + index);
    const int fd = open(path.c_str(), O_WRONLY | O_TRUNC);
    CHECK(fd >= 0);
    if (fd < 0) return;
    CHECK(write(fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size()));
    close(fd);
}

std::vector<PackageView> load_views(const std::string& primary, const std::string& shard) {
    const std::array<ArtifactSource, 2> sources = {{
        {primary, ArtifactRole::Primary, ArtifactId{7}},
        {shard, ArtifactRole::Shard, ArtifactId{11}},
    }};
    auto loaded = ArtifactSet::load_graph(sources);
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    std::vector<PackageView> views;
    if (auto* set = std::get_if<ArtifactSet>(&loaded)) {
        for (ArtifactId id : {ArtifactId{7}, ArtifactId{11}}) {
            auto view = set->view(id);
            CHECK(std::holds_alternative<PackageView>(view));
            if (auto* package = std::get_if<PackageView>(&view)) views.push_back(*package);
        }
    }
    return views;
}

ArtifactTensorRecord dense_tensor(uint32_t id, ArtifactId artifact, uint64_t offset,
                                  TensorRole role) {
    ArtifactTensorRecord tensor;
    tensor.id = id;
    tensor.logical_type = ArtifactScalarType::F32;
    tensor.logical_dimensions = {2, 4};
    tensor.layout.kind = PhysicalLayoutKind::ContiguousRowMajor;
    tensor.layout.rank = 2;
    tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides = {4, 1, 0, 0, 0, 0, 0, 0};
    tensor.quantization.kind = QuantizationKind::None;
    tensor.quantization.required_plane_mask = artifact_plane_mask(PlaneKind::Values);
    tensor.role_evidence.push_back({role, CanonicalFactKey{100 + id}, ArtifactFactAuthority::Structural});
    tensor.planes.push_back({PlaneKind::Values, ArtifactScalarType::F32,
                             {artifact, offset, 32}, 8, 4, 1, 4});
    return tensor;
}

ArtifactIndexInput valid_input(const std::vector<PackageView>& views) {
    ArtifactIndexInput input;
    input.artifacts = views;
    input.metadata_facts.push_back({CanonicalFactKey{1}, uint64_t{2},
                                    {ArtifactId{7}, 0, 8}, ArtifactFactAuthority::Declared});
    input.package_facts.push_back({CanonicalFactKey{2}, std::vector<uint64_t>{2, 4},
                                   {ArtifactId{11}, 8, 8}, ArtifactFactAuthority::Structural});
    input.tensors.push_back(dense_tensor(3, ArtifactId{7}, 64, TensorRole::TokenEmbedding));
    input.tensors.push_back(dense_tensor(9, ArtifactId{11}, 128, TensorRole::OutputWeight));
    input.diagnostics.push_back({ArtifactId{7}, CanonicalFactKey{1}, UINT32_MAX,
                                 "format-specific.key", "format-specific.tensor"});
    return input;
}

void test_canonicalization_excludes_order_and_diagnostics() {
    const std::string primary = temporary_path();
    const std::string shard = temporary_path();
    write_pattern(primary, 3, 512);
    write_pattern(shard, 17, 512);
    const std::vector<PackageView> views = load_views(primary, shard);
    CHECK(views.size() == 2);

    ArtifactIndexInput first = valid_input(views);
    auto first_result = ArtifactIndex::build(std::move(first));
    CHECK(std::holds_alternative<ArtifactIndex>(first_result));

    ArtifactIndexInput second = valid_input(views);
    std::reverse(second.artifacts.begin(), second.artifacts.end());
    std::reverse(second.tensors.begin(), second.tensors.end());
    second.diagnostics[0].metadata_spelling = "another-format.key";
    second.diagnostics[0].tensor_spelling = "another-format.tensor";
    auto second_result = ArtifactIndex::build(std::move(second));
    CHECK(std::holds_alternative<ArtifactIndex>(second_result));

    if (auto* left = std::get_if<ArtifactIndex>(&first_result)) {
        if (auto* right = std::get_if<ArtifactIndex>(&second_result)) {
            CHECK(left->canonical_bytes() == right->canonical_bytes());
            CHECK(left->digest() == right->digest());
            CHECK(left->artifacts().size() == 2);
            CHECK(left->tensors().size() == 2);
            CHECK(left->diagnostics()[0].metadata_spelling != right->diagnostics()[0].metadata_spelling);
        }
    }

    unlink(primary.c_str());
    unlink(shard.c_str());
}

void expect_error(ArtifactIndexInput input, CompatibilityError code) {
    auto result = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<CompatibilityReport>(result));
    if (auto* report = std::get_if<CompatibilityReport>(&result)) CHECK(report->code == code);
}

void test_span_plane_overlap_and_alias_validation() {
    const std::string primary = temporary_path();
    const std::string shard = temporary_path();
    write_pattern(primary, 3, 512);
    write_pattern(shard, 17, 512);
    const std::vector<PackageView> views = load_views(primary, shard);

    ArtifactIndexInput outside = valid_input(views);
    outside.tensors[0].planes[0].source = {ArtifactId{7}, 500, 32};
    expect_error(std::move(outside), CompatibilityError::IMPORT_TENSOR_UNMAPPED);

    ArtifactIndexInput wrong_bytes = valid_input(views);
    wrong_bytes.tensors[0].planes[0].source.length = 28;
    expect_error(std::move(wrong_bytes), CompatibilityError::IR_SHAPE_MISMATCH);

    ArtifactIndexInput overlap = valid_input(views);
    overlap.tensors[1] = dense_tensor(9, ArtifactId{7}, 80, TensorRole::OutputWeight);
    expect_error(std::move(overlap), CompatibilityError::IMPORT_TENSOR_DUPLICATE);

    ArtifactIndexInput tied = valid_input(views);
    tied.tensors[1] = dense_tensor(9, ArtifactId{7}, 64, TensorRole::OutputWeight);
    tied.aliases.push_back({ArtifactAliasKind::TiedOutput, ArtifactAliasDirection::SourceToTarget,
                            3, 9, TensorRole::OutputWeight});
    auto tied_result = ArtifactIndex::build(std::move(tied));
    CHECK(std::holds_alternative<ArtifactIndex>(tied_result));

    ArtifactIndexInput undeclared = valid_input(views);
    undeclared.tensors[1] = dense_tensor(9, ArtifactId{7}, 64, TensorRole::OutputWeight);
    expect_error(std::move(undeclared), CompatibilityError::IMPORT_TENSOR_DUPLICATE);

    ArtifactIndexInput false_alias = valid_input(views);
    false_alias.aliases.push_back({ArtifactAliasKind::TiedOutput, ArtifactAliasDirection::SourceToTarget,
                                  3, 9, TensorRole::OutputWeight});
    expect_error(std::move(false_alias), CompatibilityError::IR_REFERENCE_INVALID);

    unlink(primary.c_str());
    unlink(shard.c_str());
}

void test_checked_element_and_block_arithmetic() {
    const std::string primary = temporary_path();
    const std::string shard = temporary_path();
    write_pattern(primary, 3, 512);
    write_pattern(shard, 17, 512);
    const std::vector<PackageView> views = load_views(primary, shard);

    ArtifactIndexInput overflow = valid_input(views);
    overflow.tensors[0].logical_dimensions = {UINT64_MAX, 2};
    expect_error(std::move(overflow), CompatibilityError::IR_SHAPE_MISMATCH);

    ArtifactIndexInput bad_axis = valid_input(views);
    bad_axis.tensors[0].layout.axis_order[1] = 0;
    expect_error(std::move(bad_axis), CompatibilityError::IR_LAYOUT_MISMATCH);

    ArtifactIndexInput missing_plane = valid_input(views);
    missing_plane.tensors[0].quantization.required_plane_mask |= artifact_plane_mask(PlaneKind::Scales);
    expect_error(std::move(missing_plane), CompatibilityError::IR_QUANTIZATION_UNSUPPORTED);

    unlink(primary.c_str());
    unlink(shard.c_str());
}

void test_physical_index_does_not_claim_semantic_roles() {
    const std::string primary = temporary_path();
    const std::string shard = temporary_path();
    write_pattern(primary, 3, 512);
    write_pattern(shard, 17, 512);
    const std::vector<PackageView> views = load_views(primary, shard);

    ArtifactIndexInput input = valid_input(views);
    for (ArtifactTensorRecord& tensor : input.tensors) tensor.role_evidence.clear();
    auto result = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<ArtifactIndex>(result));
    if (const auto* index = std::get_if<ArtifactIndex>(&result)) {
        CHECK(index->tensors().size() == 2);
        CHECK(index->tensors()[0].role_evidence.empty());
        CHECK(index->tensors()[1].role_evidence.empty());
    }

    unlink(primary.c_str());
    unlink(shard.c_str());
}

} // namespace

int main() {
    static_assert(!std::is_constructible_v<ArtifactFactValue, std::string>);
    test_canonicalization_excludes_order_and_diagnostics();
    test_span_plane_overlap_and_alias_validation();
    test_checked_element_and_block_arithmetic();
    test_physical_index_does_not_claim_semantic_roles();
    return test_summary("test_artifact_index");
}
