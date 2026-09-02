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

    ArtifactIndexInput bad_bank = valid_input(views);
    bad_bank.tensors[0].coordinate = {0, 0, UINT32_MAX, 0, UINT32_MAX, 1, 2, 16};
    expect_error(std::move(bad_bank), CompatibilityError::IR_LAYOUT_MISMATCH);

    ArtifactIndexInput bad_source_axes = valid_input(views);
    bad_source_axes.tensors[0].axis.source_rank = 2;
    bad_source_axes.tensors[0].axis.source_axis_order = {0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    expect_error(std::move(bad_source_axes), CompatibilityError::IR_LAYOUT_MISMATCH);

    unlink(primary.c_str());
    unlink(shard.c_str());
}

void test_coordinate_namespace_collision_is_rejected() {
    const std::string primary = temporary_path();
    const std::string shard = temporary_path();
    write_pattern(primary, 3, 512);
    write_pattern(shard, 17, 512);
    const std::vector<PackageView> views = load_views(primary, shard);

    ArtifactIndexInput disjoint = valid_input(views);
    const ArtifactCoordinate fallback_namespace =
        {0, 0, 0x80000003u, UINT32_MAX, UINT32_MAX, UINT8_MAX, 0, 0};
    const ArtifactCoordinate explicit_slot =
        {0, 0, 3, UINT32_MAX, UINT32_MAX, UINT8_MAX, 0, 0};
    disjoint.tensors[0].coordinate = fallback_namespace;
    disjoint.tensors[1].coordinate = explicit_slot;
    auto disjoint_result = ArtifactIndex::build(std::move(disjoint));
    CHECK(std::holds_alternative<ArtifactIndex>(disjoint_result));

    ArtifactIndexInput collision = valid_input(views);
    collision.tensors[0].coordinate = fallback_namespace;
    collision.tensors[1].coordinate = fallback_namespace;
    expect_error(std::move(collision), CompatibilityError::IMPORT_TENSOR_DUPLICATE);

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

void test_complete_physical_contract_validation() {
    const std::string primary = temporary_path();
    const std::string shard = temporary_path();
    write_pattern(primary, 3, 1024);
    write_pattern(shard, 17, 512);
    const std::vector<PackageView> views = load_views(primary, shard);

    ArtifactIndexInput blocked = valid_input(views);
    ArtifactTensorRecord quantized;
    quantized.id = 3;
    quantized.logical_type = ArtifactScalarType::F32;
    quantized.logical_dimensions = {32, 1};
    quantized.format = {1, ArtifactPhysicalEncoding::Q5_0, ArtifactScalarType::Packed,
                        ArtifactScalarType::F16, ArtifactScalarType::None,
                        ArtifactScalarType::None, ArtifactScalarType::None, 32, 22, 2, 0, 0, 0};
    quantized.layout.kind = PhysicalLayoutKind::GgufBlocked;
    quantized.layout.packing = PackingKind::Gguf;
    quantized.layout.rank = 2;
    quantized.layout.block_rank = 1;
    quantized.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    quantized.layout.strides = {1, 32, 0, 0, 0, 0, 0, 0};
    quantized.layout.block_elements = 32;
    quantized.layout.block_bytes = 22;
    quantized.axis.source_rank = 2;
    quantized.axis.source_axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    quantized.axis.block_axis = 0;
    quantized.axis.block_elements = 32;
    quantized.axis.bytes_per_block = 22;
    quantized.axis.row_stride_bytes = 22;
    quantized.quantization.kind = QuantizationKind::BlockedAffine;
    quantized.quantization.scale_type = ScalarType::F16;
    quantized.quantization.block_elements = 32;
    quantized.quantization.block_bytes = 22;
    quantized.quantization.group_size = 32;
    quantized.quantization.required_plane_mask = 1;
    quantized.planes = {{PlaneKind::Values, ArtifactScalarType::Packed,
                         {ArtifactId{7}, 64, 22}, 32, 22, 32, 4}};
    quantized.role_evidence.push_back({TensorRole::TokenEmbedding, CanonicalFactKey{103},
                                       ArtifactFactAuthority::Structural});
    blocked.tensors[0] = quantized;
    auto valid_blocked = ArtifactIndex::build(std::move(blocked));
    CHECK(std::holds_alternative<ArtifactIndex>(valid_blocked));

    ArtifactIndexInput contradictory_axis = valid_input(views);
    ArtifactTensorRecord wrong_axis = quantized;
    wrong_axis.logical_dimensions = {32, 32};
    wrong_axis.layout.strides = {1, 32, 0, 0, 0, 0, 0, 0};
    wrong_axis.axis.block_axis = 1;
    wrong_axis.planes[0].source.length = 704;
    wrong_axis.planes[0].logical_elements = 1024;
    contradictory_axis.tensors[0] = std::move(wrong_axis);
    expect_error(std::move(contradictory_axis), CompatibilityError::IR_QUANTIZATION_UNSUPPORTED);

    ArtifactIndexInput mismatched = valid_input(views);
    mismatched.tensors[0] = quantized;
    mismatched.tensors[0].quantization.block_bytes = 23;
    expect_error(std::move(mismatched), CompatibilityError::IR_QUANTIZATION_UNSUPPORTED);

    ArtifactIndexInput alias_contract = valid_input(views);
    alias_contract.tensors[1] = dense_tensor(9, ArtifactId{7}, 64, TensorRole::OutputWeight);
    alias_contract.aliases.push_back({ArtifactAliasKind::TiedOutput, ArtifactAliasDirection::SourceToTarget,
                                      3, 9, TensorRole::OutputWeight});
    alias_contract.tensors[1].logical_type = ArtifactScalarType::F16;
    expect_error(std::move(alias_contract), CompatibilityError::IR_REFERENCE_INVALID);

    unlink(primary.c_str());
    unlink(shard.c_str());
}

void test_scalar_encoding_does_not_select_axis_order() {
    const std::string primary = temporary_path();
    const std::string shard = temporary_path();
    write_pattern(primary, 3, 512);
    write_pattern(shard, 17, 512);
    const std::vector<PackageView> views = load_views(primary, shard);

    auto scalar = [](ArtifactTensorRecord tensor) {
        tensor.axis.source_rank = 2;
        tensor.axis.source_axis_order =
            {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        tensor.format = {1, ArtifactPhysicalEncoding::F32,
                         ArtifactScalarType::F32,
                         ArtifactScalarType::None,
                         ArtifactScalarType::None,
                         ArtifactScalarType::None,
                         ArtifactScalarType::None, 1, 4, 0, 0, 0, 0};
        tensor.axis.row_stride_bytes = 16;
        return tensor;
    };

    ArtifactIndexInput last_axis_contiguous = valid_input(views);
    last_axis_contiguous.tensors[0] = scalar(last_axis_contiguous.tensors[0]);
    CHECK(std::holds_alternative<ArtifactIndex>(
        ArtifactIndex::build(std::move(last_axis_contiguous))));

    ArtifactIndexInput first_axis_contiguous = valid_input(views);
    first_axis_contiguous.tensors[0] = scalar(first_axis_contiguous.tensors[0]);
    first_axis_contiguous.tensors[0].layout.strides =
        {1, 2, 0, 0, 0, 0, 0, 0};
    first_axis_contiguous.tensors[0].axis.row_stride_bytes = 8;
    CHECK(std::holds_alternative<ArtifactIndex>(
        ArtifactIndex::build(std::move(first_axis_contiguous))));

    ArtifactIndexInput malformed = valid_input(views);
    malformed.tensors[0] = scalar(malformed.tensors[0]);
    malformed.tensors[0].layout.strides =
        {2, 2, 0, 0, 0, 0, 0, 0};
    expect_error(std::move(malformed), CompatibilityError::IR_LAYOUT_MISMATCH);

    unlink(primary.c_str());
    unlink(shard.c_str());
}

void test_grouped_affine_physical_contract_is_exact() {
    const std::string primary = temporary_path();
    const std::string shard = temporary_path();
    write_pattern(primary, 3, 4096);
    write_pattern(shard, 17, 512);
    const std::vector<PackageView> views = load_views(primary, shard);

    ArtifactTensorRecord affine;
    affine.id = 3;
    affine.logical_type = ArtifactScalarType::F32;
    affine.logical_dimensions = {8, 512};
    affine.layout.kind = PhysicalLayoutKind::GroupedAffine;
    affine.layout.packing = PackingKind::LsbBitPacked;
    affine.layout.rank = 2;
    affine.layout.block_rank = 1;
    affine.layout.axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    affine.layout.strides = {1, 512, 0, 0, 0, 0, 0, 0};
    affine.layout.block_elements = 256;
    affine.layout.block_bytes = 64;
    affine.axis.source_rank = 2;
    affine.axis.source_axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    affine.axis.block_axis = 1;
    affine.axis.block_elements = 256;
    affine.axis.bytes_per_block = 64;
    affine.axis.row_stride_bytes = 128;
    affine.format = {2, ArtifactPhysicalEncoding::GroupedAffineU2_256,
                     ArtifactScalarType::U32, ArtifactScalarType::F16,
                     ArtifactScalarType::None, ArtifactScalarType::None,
                     ArtifactScalarType::None, 256, 64, 2, 0, 0, 0,
                     ArtifactScalarType::F16, 2};
    affine.quantization.kind = QuantizationKind::BlockedAffine;
    affine.quantization.accumulation_type = ScalarType::F32;
    affine.quantization.scale_type = ScalarType::F16;
    affine.quantization.bias_type = ScalarType::F16;
    affine.quantization.block_elements = 256;
    affine.quantization.block_bytes = 64;
    affine.quantization.group_size = 256;
    affine.quantization.required_plane_mask = 7;
    affine.planes = {
        {PlaneKind::Values, ArtifactScalarType::U32,
         {ArtifactId{7}, 128, 1024}, 4096, 4, 16, 128},
        {PlaneKind::Scales, ArtifactScalarType::F16,
         {ArtifactId{7}, 1280, 32}, 16, 2, 1, 128},
        {PlaneKind::Biases, ArtifactScalarType::F16,
         {ArtifactId{7}, 1408, 32}, 16, 2, 1, 128},
    };

    ArtifactIndexInput valid = valid_input(views);
    valid.tensors[0] = affine;
    CHECK(std::holds_alternative<ArtifactIndex>(ArtifactIndex::build(std::move(valid))));

    ArtifactIndexInput missing_bias = valid_input(views);
    missing_bias.tensors[0] = affine;
    missing_bias.tensors[0].planes.pop_back();
    expect_error(std::move(missing_bias), CompatibilityError::IR_QUANTIZATION_UNSUPPORTED);

    ArtifactIndexInput missing_auxiliary = valid_input(views);
    missing_auxiliary.tensors[0] = affine;
    missing_auxiliary.tensors[0].quantization.required_plane_mask =
        artifact_plane_mask(PlaneKind::Values);
    missing_auxiliary.tensors[0].planes.resize(1);
    expect_error(std::move(missing_auxiliary), CompatibilityError::IR_QUANTIZATION_UNSUPPORTED);

    ArtifactIndexInput wrong_packing = valid_input(views);
    wrong_packing.tensors[0] = affine;
    wrong_packing.tensors[0].layout.packing = PackingKind::Gguf;
    expect_error(std::move(wrong_packing), CompatibilityError::IR_LAYOUT_MISMATCH);

    ArtifactIndexInput wrong_axis = valid_input(views);
    wrong_axis.tensors[0] = affine;
    wrong_axis.tensors[0].layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    expect_error(std::move(wrong_axis), CompatibilityError::IR_LAYOUT_MISMATCH);

    ArtifactIndexInput wrong_bias = valid_input(views);
    wrong_bias.tensors[0] = affine;
    wrong_bias.tensors[0].format.bias_bytes = 4;
    expect_error(std::move(wrong_bias), CompatibilityError::IR_QUANTIZATION_UNSUPPORTED);

    unlink(primary.c_str());
    unlink(shard.c_str());
}

void test_column_grouped_affine_u2_skip_uses_packed_byte_values() {
    const std::string primary = temporary_path();
    const std::string shard = temporary_path();
    write_pattern(primary, 3, 4096);
    write_pattern(shard, 17, 512);
    const std::vector<PackageView> views = load_views(primary, shard);

    ArtifactTensorRecord tensor;
    tensor.id = 3;
    tensor.logical_type = ArtifactScalarType::F32;
    tensor.logical_dimensions = {256, 8};
    tensor.layout.kind = PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip;
    tensor.layout.packing = PackingKind::LsbBitPacked;
    tensor.layout.rank = 2;
    tensor.layout.block_rank = 1;
    tensor.layout.axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides = {1, 8, 0, 0, 0, 0, 0, 0};
    tensor.layout.block_elements = 256;
    tensor.layout.block_bytes = 64;
    tensor.axis.source_rank = 2;
    tensor.axis.source_axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.axis.block_axis = 0;
    tensor.axis.block_elements = 256;
    tensor.axis.bytes_per_block = 64;
    tensor.axis.row_stride_bytes = 8 * 64;
    tensor.format = {1, ArtifactPhysicalEncoding::ColumnGroupedAffineU2Skip256,
                     ArtifactScalarType::U8, ArtifactScalarType::F16,
                     ArtifactScalarType::None, ArtifactScalarType::None,
                     ArtifactScalarType::None, 256, 64, 2, 0, 0, 0,
                     ArtifactScalarType::F16, 2};
    tensor.quantization.kind = QuantizationKind::BlockedAffine;
    tensor.quantization.accumulation_type = ScalarType::F32;
    tensor.quantization.scale_type = ScalarType::F16;
    tensor.quantization.bias_type = ScalarType::F16;
    tensor.quantization.block_elements = 256;
    tensor.quantization.block_bytes = 64;
    tensor.quantization.group_size = 256;
    tensor.quantization.required_plane_mask = 7;
    tensor.planes = {
        {PlaneKind::Values, ArtifactScalarType::U8,
         {ArtifactId{7}, 128, 512}, 2048, 64, 256, 128},
        {PlaneKind::Scales, ArtifactScalarType::F16,
         {ArtifactId{7}, 768, 16}, 8, 2, 1, 128},
        {PlaneKind::Biases, ArtifactScalarType::F16,
         {ArtifactId{7}, 896, 16}, 8, 2, 1, 128},
    };

    ArtifactIndexInput valid = valid_input(views);
    valid.tensors[0] = tensor;
    CHECK(std::holds_alternative<ArtifactIndex>(ArtifactIndex::build(std::move(valid))));

    ArtifactIndexInput reordered = valid_input(views);
    reordered.tensors[0] = tensor;
    std::reverse(reordered.tensors[0].planes.begin(), reordered.tensors[0].planes.end());
    CHECK(std::holds_alternative<ArtifactIndex>(ArtifactIndex::build(std::move(reordered))));

    ArtifactIndexInput u32_alias = valid_input(views);
    u32_alias.tensors[0] = tensor;
    u32_alias.tensors[0].format.value_type = ArtifactScalarType::U32;
    u32_alias.tensors[0].planes[0].storage_type = ArtifactScalarType::U32;
    expect_error(std::move(u32_alias), CompatibilityError::IR_QUANTIZATION_UNSUPPORTED);

    ArtifactIndexInput wrong_stride = valid_input(views);
    wrong_stride.tensors[0] = tensor;
    wrong_stride.tensors[0].axis.row_stride_bytes = 64;
    expect_error(std::move(wrong_stride), CompatibilityError::IR_QUANTIZATION_UNSUPPORTED);

    ArtifactIndexInput wrong_scale = valid_input(views);
    wrong_scale.tensors[0] = tensor;
    wrong_scale.tensors[0].planes[1].storage_type = ArtifactScalarType::U16;
    expect_error(std::move(wrong_scale), CompatibilityError::IR_QUANTIZATION_UNSUPPORTED);

    ArtifactIndexInput extra_plane = valid_input(views);
    extra_plane.tensors[0] = tensor;
    extra_plane.tensors[0].planes.push_back(
        {PlaneKind::Zeros, ArtifactScalarType::F16,
         {ArtifactId{7}, 1024, 16}, 8, 2, 1, 128});
    expect_error(std::move(extra_plane), CompatibilityError::IR_QUANTIZATION_UNSUPPORTED);

    ArtifactIndexInput legacy_layout = valid_input(views);
    legacy_layout.tensors[0] = tensor;
    legacy_layout.tensors[0].layout.kind = PhysicalLayoutKind::GroupedAffine;
    expect_error(std::move(legacy_layout), CompatibilityError::IR_QUANTIZATION_UNSUPPORTED);

    unlink(primary.c_str());
    unlink(shard.c_str());
}

} // namespace

int main() {
    static_assert(!std::is_constructible_v<ArtifactFactValue, std::string>);
    test_canonicalization_excludes_order_and_diagnostics();
    test_span_plane_overlap_and_alias_validation();
    test_checked_element_and_block_arithmetic();
    test_coordinate_namespace_collision_is_rejected();
    test_physical_index_does_not_claim_semantic_roles();
    test_complete_physical_contract_validation();
    test_scalar_encoding_does_not_select_axis_order();
    test_grouped_affine_physical_contract_is_exact();
    test_column_grouped_affine_u2_skip_uses_packed_byte_values();
    return test_summary("test_artifact_index");
}
