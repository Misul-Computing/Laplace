#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <variant>
#include <vector>

#include "artifact_index.h"
#include "artifact_set.h"
#include "codec_binding.h"
#include "codec_certificate.h"
#include "compat_rule.h"
#include "semantic_manifest.h"
#include "test_util.h"

namespace {
bool fail_all_allocations = false;
}

void* operator new(std::size_t size) {
    if (fail_all_allocations) throw std::bad_alloc();
    if (void* allocation = std::malloc(size ? size : 1)) return allocation;
    throw std::bad_alloc();
}

void operator delete(void* allocation) noexcept { std::free(allocation); }
void operator delete(void* allocation, std::size_t) noexcept { std::free(allocation); }

using namespace Laplace;

namespace {

struct Fixture {
    std::shared_ptr<const RuntimePackage> package;
    std::shared_ptr<const RuntimePackage> diagnostic;
    SemanticModel model;
    ArtifactIndex index;
};

SemanticTensor semantic_tensor(uint32_t id, uint64_t offset) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = id == 0 ? TensorRole::TokenEmbedding
                          : id == 1 ? TensorRole::AttentionNormWeight
                                    : TensorRole::OutputWeight;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, 1}};
    tensor.layout.rank = 1;
    tensor.layout.axis_order = {0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides[0] = 1;
    tensor.planes = {{PlaneKind::Values, ScalarType::F16, ArtifactId{0}, offset, 2, 4, 0}};
    return tensor;
}

ArtifactTensorRecord physical_tensor(const SemanticTensor& semantic) {
    ArtifactTensorRecord tensor;
    tensor.id = semantic.id;
    tensor.logical_type = ArtifactScalarType::F32;
    tensor.logical_dimensions = semantic.layout.kind == PhysicalLayoutKind::GgufBlocked
        ? std::vector<uint64_t>{256, 1} : std::vector<uint64_t>{1};
    tensor.layout = semantic.layout;
    tensor.quantization = semantic.quantization;
    tensor.axis.source_rank = semantic.layout.kind == PhysicalLayoutKind::GgufBlocked ? 2 : 0;
    if (tensor.axis.source_rank != 0) tensor.axis.source_axis_order = semantic.layout.axis_order;
    if (semantic.layout.kind != PhysicalLayoutKind::GgufBlocked) tensor.axis.row_stride_bytes = 2;
    const TensorPlane& plane = semantic.planes.front();
    if (semantic.layout.kind == PhysicalLayoutKind::GgufBlocked) {
        tensor.axis.block_axis = 0;
        tensor.axis.block_elements = 256;
        tensor.axis.bytes_per_block = 144;
        tensor.axis.row_stride_bytes = 144;
        tensor.format = {1, ArtifactPhysicalEncoding::Q4_K, ArtifactScalarType::Packed,
                         ArtifactScalarType::F16, ArtifactScalarType::F16,
                         ArtifactScalarType::Packed, ArtifactScalarType::None,
                         256, 144, 2, 2, 12, 0};
        tensor.planes = {{PlaneKind::Values, ArtifactScalarType::Packed,
                          {plane.artifact_id, plane.offset, plane.length}, 256, 144, 256,
                          plane.alignment}};
    } else {
        tensor.format = {1, ArtifactPhysicalEncoding::F16, ArtifactScalarType::F16,
                         ArtifactScalarType::None, ArtifactScalarType::None,
                         ArtifactScalarType::None, ArtifactScalarType::None,
                         1, 2, 0, 0, 0, 0};
        tensor.planes = {{PlaneKind::Values, ArtifactScalarType::F16,
                          {plane.artifact_id, plane.offset, plane.length}, 1, 2, 1,
                          plane.alignment}};
    }
    return tensor;
}

SemanticModel make_model() {
    SemanticModel model;
    model.maximum_context = 32768;
    model.vocabulary_size = 8;
    model.bos_id = 1;
    model.eos_id = 2;
    model.stop_ids = {2};
    for (size_t index = 0; index != model.tokenizer_digest.size(); ++index) {
        model.tokenizer_digest[index] = static_cast<uint8_t>(index + 1);
    }
    model.input_values_first = 0;
    model.input_values_count = 1;
    model.output_values_first = 4;
    model.output_values_count = 1;
    for (uint32_t id = 0; id != 5; ++id) {
        model.values.push_back({id, ScalarType::F32,
                                {{DimensionKind::Constant, 1}}, 0});
    }
    model.tensors.push_back(semantic_tensor(0, 0));
    model.tensors.push_back(semantic_tensor(1, 4));
    model.tensors.push_back(semantic_tensor(2, 8));
    model.operators = {
        {0, OperatorKind::EmbeddingLookup, 1, {0}, {1}, {0}, {},
         EmbeddingLookupPayload{0x3f800000u, 8, 1, 0}},
        {1, OperatorKind::RmsNorm, 1, {1}, {2}, {1}, {}, RmsNormPayload{}},
        {2, OperatorKind::RmsNorm, 1, {2}, {3}, {1}, {}, RmsNormPayload{}},
        {3, OperatorKind::Linear, 1, {3}, {4}, {2}, {}, LinearPayload{}},
    };
    model.layers.push_back({0, 1, 3, 0});
    return model;
}

Fixture make_fixture(bool alter_layout = false, uint64_t artifact_seed = 0,
                     bool certificate_misaligned = false) {
    Fixture fixture;
    fixture.model = make_model();
    if (certificate_misaligned) {
        fixture.model.tensors[0].planes[0].offset = 1;
        fixture.model.tensors[0].planes[0].alignment = 1;
    }
    if (alter_layout) {
        fixture.model.tensors[0].dimensions = {{DimensionKind::Constant, 256},
                                               {DimensionKind::Constant, 1}};
        fixture.model.tensors[0].layout.kind = PhysicalLayoutKind::GgufBlocked;
        fixture.model.tensors[0].layout.packing = PackingKind::Gguf;
        fixture.model.tensors[0].layout.rank = 2;
        fixture.model.tensors[0].layout.block_rank = 1;
        fixture.model.tensors[0].layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        fixture.model.tensors[0].layout.strides = {1, 256, 0, 0, 0, 0, 0, 0};
        fixture.model.tensors[0].layout.block_elements = 256;
        fixture.model.tensors[0].layout.block_bytes = 144;
        fixture.model.tensors[0].quantization.kind = QuantizationKind::BlockedAffine;
        fixture.model.tensors[0].quantization.block_elements = 256;
        fixture.model.tensors[0].quantization.block_bytes = 144;
        fixture.model.tensors[0].quantization.group_size = 256;
        fixture.model.tensors[0].quantization.required_plane_mask = 1;
        fixture.model.tensors[0].quantization.scale_type = ScalarType::F16;
        fixture.model.tensors[0].quantization.zero_type = ScalarType::F16;
        fixture.model.tensors[0].planes[0].storage_type = ScalarType::U8;
        fixture.model.tensors[0].planes[0].length = 144;
        fixture.model.tensors[1].planes[0].offset = 144;
        fixture.model.tensors[2].planes[0].offset = 148;
    }

    std::array<uint8_t, 256> bytes{};
    std::fill(bytes.begin(), bytes.end(), static_cast<uint8_t>(artifact_seed + 0x31));
    auto owned = ArtifactSet::make_owned_blob(ArtifactId{0}, ArtifactRole::Primary,
                                               std::span<const uint8_t>(bytes));
    CHECK(std::holds_alternative<PackageView>(owned));
    if (!std::holds_alternative<PackageView>(owned)) return fixture;

    ArtifactIndexInput index_input;
    index_input.artifacts.push_back(std::get<PackageView>(std::move(owned)));
    for (const SemanticTensor& tensor : fixture.model.tensors) {
        index_input.tensors.push_back(physical_tensor(tensor));
    }
    auto built_index = ArtifactIndex::build(std::move(index_input));
    CHECK(std::holds_alternative<ArtifactIndex>(built_index));
    if (!std::holds_alternative<ArtifactIndex>(built_index)) return fixture;
    fixture.index = std::get<ArtifactIndex>(std::move(built_index));

    TokenContract contract;
    contract.vocabulary_size = fixture.model.vocabulary_size;
    contract.bos_id = fixture.model.bos_id;
    contract.eos_id = fixture.model.eos_id;
    contract.stop_ids = fixture.model.stop_ids;
    contract.authoritative_tokenizer_digest.bytes = fixture.model.tokenizer_digest;

    PhysicalCodecRegistry codec_registry;
    for (const SemanticTensor& tensor : fixture.model.tensors) {
        const std::vector<uint8_t> certificate_bytes =
            tensor.layout.kind == PhysicalLayoutKind::GgufBlocked
                ? make_q4_k_codec_certificate()
                : make_raw_f16_codec_certificate();
        const CodecCertificateParseResult parsed =
            parse_codec_certificate(certificate_bytes);
        const auto* certificate = std::get_if<CodecCertificate>(&parsed);
        CHECK(certificate != nullptr);
        if (!certificate) return fixture;
        auto identity = physical_codec_identity(
            tensor, certificate->identity().abi_version,
            certificate->identity().digest);
        CHECK(identity.has_value());
        if (!identity) return fixture;
        codec_registry.tensors.push_back({tensor.id, *identity});
        const auto found = std::find_if(codec_registry.codecs.begin(), codec_registry.codecs.end(),
                                        [&](const auto& codec) {
                                            return codec.identity == *identity;
                                        });
        if (found == codec_registry.codecs.end())
            codec_registry.codecs.push_back({*identity, certificate_bytes});
    }
    std::sort(codec_registry.codecs.begin(), codec_registry.codecs.end(),
              [](const auto& left, const auto& right) {
                  return physical_codec_identity_less(left.identity, right.identity);
              });
    auto manifest_result = SemanticManifest::build(
        fixture.index, fixture.model, contract, codec_registry);
    CHECK(std::holds_alternative<SemanticManifest>(manifest_result));
    if (!std::holds_alternative<SemanticManifest>(manifest_result)) return fixture;
    SemanticManifest manifest = std::get<SemanticManifest>(std::move(manifest_result));
    fixture.diagnostic = RuntimePackage::make_diagnostic(manifest, Sha256Digest{}, 0);
    fixture.package = RuntimePackage::make_closed_v1_test_only(std::move(manifest));
    CHECK(fixture.package != nullptr);
    return fixture;
}

const CompatibilityReport* report_of(const CodecBindingPreflightResult& result) {
    return std::get_if<CompatibilityReport>(&result);
}

void test_valid_output_is_fingerprint_bound_and_ordered() {
    Fixture fixture = make_fixture();
    CHECK(fixture.package != nullptr);
    if (!fixture.package) return;

    const auto result = preflight_codec_bindings(*fixture.package);
    CHECK(std::holds_alternative<ResolvedCodecBindings>(result));
    if (!std::holds_alternative<ResolvedCodecBindings>(result)) return;
    const ResolvedCodecBindings& bindings = std::get<ResolvedCodecBindings>(result);
    CHECK(bindings.package_fingerprint() == fixture.package->package_fingerprint());
    CHECK(bindings.operators().size() == 4);
    const std::array<uint32_t, 4> expected_operator_ids = {0, 1, 2, 3};
    const std::array<uint32_t, 4> expected_tensor_ids = {0, 1, 1, 2};
    for (size_t index = 0; index != bindings.operators().size(); ++index) {
        CHECK(bindings.operators()[index].operator_id() == expected_operator_ids[index]);
        CHECK(bindings.operators()[index].tensors().size() == 1);
        if (bindings.operators()[index].tensors().size() != 1) continue;
        const ResolvedCodecTensor& tensor = bindings.operators()[index].tensors()[0];
        CHECK(tensor.occurrence_index() == index);
        CHECK(tensor.tensor_slot() == 0);
        CHECK(tensor.tensor_id() == expected_tensor_ids[index]);
        CHECK(tensor.program_identity().abi_version ==
              tensor.physical_identity().arithmetic_version);
        CHECK(tensor.program_identity().contract_digest ==
              tensor.physical_identity().arithmetic_digest);
        CHECK(std::vector<Dimension>(tensor.dimensions().begin(), tensor.dimensions().end()) ==
              fixture.model.tensors[tensor.tensor_id()].dimensions);
        CHECK(tensor.strides() == fixture.model.tensors[tensor.tensor_id()].layout.strides);
        CHECK(tensor.planes().size() == 1);
        if (tensor.planes().size() == 1) {
            CHECK(tensor.planes()[0].kind() == PlaneKind::Values);
            CHECK(tensor.planes()[0].storage_type() == ArtifactScalarType::F16);
            CHECK(tensor.planes()[0].semantic_storage_type() == ScalarType::F16);
            CHECK(tensor.planes()[0].logical_elements() == 1);
            CHECK(tensor.planes()[0].elements_per_block() == 1);
            CHECK(tensor.planes()[0].bytes_per_block() == 2);
            CHECK(tensor.planes()[0].offset() == fixture.model.tensors[tensor.tensor_id()].planes[0].offset);
        }
    }
    CHECK(bindings.matches_package(*fixture.package));
    CHECK(!bindings.matches_package(*fixture.diagnostic));
}

void test_certificate_identity_and_diagnostic_authority() {
    Fixture raw_fixture = make_fixture();
    Fixture blocked_fixture = make_fixture(true);
    CHECK(raw_fixture.package != nullptr);
    CHECK(blocked_fixture.package != nullptr);
    if (!raw_fixture.package || !blocked_fixture.package) return;
    const auto raw_result = preflight_codec_bindings(*raw_fixture.package);
    const auto blocked_result = preflight_codec_bindings(*blocked_fixture.package);
    CHECK(std::holds_alternative<ResolvedCodecBindings>(raw_result));
    CHECK(std::holds_alternative<ResolvedCodecBindings>(blocked_result));
    if (std::holds_alternative<ResolvedCodecBindings>(raw_result) &&
        std::holds_alternative<ResolvedCodecBindings>(blocked_result)) {
        const auto& raw = std::get<ResolvedCodecBindings>(raw_result);
        const auto& blocked = std::get<ResolvedCodecBindings>(blocked_result);
        CHECK(raw.operators()[0].tensors()[0].program_identity() !=
              blocked.operators()[0].tensors()[0].program_identity());
        CHECK(blocked.operators()[0].tensors()[0].planes()[0].storage_type() ==
              ArtifactScalarType::Packed);
        CHECK(blocked.operators()[0].tensors()[0].planes()[0].logical_elements() == 256);
        CHECK(blocked.operators()[0].tensors()[0].planes()[0].bytes_per_block() == 144);
    }

    Fixture valid_fixture = make_fixture();
    if (!valid_fixture.diagnostic) return;
    const auto diagnostic_result = preflight_codec_bindings(*valid_fixture.diagnostic);
    CHECK(report_of(diagnostic_result) != nullptr);
    if (const auto* report = report_of(diagnostic_result)) {
        CHECK(report->code == CompatibilityError::PACKAGE_AUTHORITY_REQUIRED);
    }
}

void test_package_fingerprint_changes_with_artifact_bytes() {
    Fixture first = make_fixture(false, 0);
    Fixture second = make_fixture(false, 1);
    if (!first.package || !second.package) return;
    const auto first_result = preflight_codec_bindings(*first.package);
    const auto second_result = preflight_codec_bindings(*second.package);
    CHECK(std::holds_alternative<ResolvedCodecBindings>(first_result));
    CHECK(std::holds_alternative<ResolvedCodecBindings>(second_result));
    if (std::holds_alternative<ResolvedCodecBindings>(first_result) &&
        std::holds_alternative<ResolvedCodecBindings>(second_result)) {
        CHECK(std::get<ResolvedCodecBindings>(first_result).package_fingerprint() !=
              std::get<ResolvedCodecBindings>(second_result).package_fingerprint());
        CHECK(std::get<ResolvedCodecBindings>(first_result).matches_package(*first.package));
        CHECK(!std::get<ResolvedCodecBindings>(first_result).matches_package(*second.package));
    }
}

void test_replay_is_rejected() {
    Fixture first = make_fixture(false, 0);
    Fixture second = make_fixture(false, 1);
    if (!first.package || !second.package) return;
    const auto result = preflight_codec_bindings(*first.package);
    CHECK(std::holds_alternative<ResolvedCodecBindings>(result));
    if (!std::holds_alternative<ResolvedCodecBindings>(result)) return;
    const ResolvedCodecBindings& bindings = std::get<ResolvedCodecBindings>(result);
    CHECK(!bindings.matches_package(*second.package));
}

void test_authority_rejects_malformed_span_before_preflight() {
    SemanticModel model = make_model();
    std::array<uint8_t, 32> bytes{};
    auto owned = ArtifactSet::make_owned_blob(ArtifactId{0}, ArtifactRole::Primary,
                                               std::span<const uint8_t>(bytes));
    CHECK(std::holds_alternative<PackageView>(owned));
    if (!std::holds_alternative<PackageView>(owned)) return;
    ArtifactIndexInput input;
    input.artifacts.push_back(std::get<PackageView>(std::move(owned)));
    for (const SemanticTensor& tensor : model.tensors)
        input.tensors.push_back(physical_tensor(tensor));
    input.tensors[0].planes[0].source.offset = UINT64_MAX;
    const auto result = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<CompatibilityReport>(result));
    if (const auto* report = std::get_if<CompatibilityReport>(&result)) {
        CHECK(report->code == CompatibilityError::IMPORT_TENSOR_UNMAPPED ||
              report->code == CompatibilityError::PACKAGE_BOUNDS_INVALID);
    }
}

void test_certificate_rejects_index_valid_misalignment() {
    Fixture fixture = make_fixture(false, 0, true);
    CHECK(fixture.package != nullptr);
    if (!fixture.package) return;

    const auto result = preflight_codec_bindings(*fixture.package);
    CHECK(std::holds_alternative<CompatibilityReport>(result));
    if (const auto* report = std::get_if<CompatibilityReport>(&result)) {
        CHECK(report->code == CompatibilityError::PACKAGE_BOUNDS_INVALID);
        CHECK(report->operator_id == 0);
        CHECK(report->tensor_id == 0);
    }
}

void test_persistent_allocation_failure_is_atomic() {
    Fixture fixture = make_fixture();
    if (!fixture.package) return;

    const auto bound = preflight_codec_bindings(*fixture.package);
    CHECK(std::holds_alternative<ResolvedCodecBindings>(bound));
    if (!std::holds_alternative<ResolvedCodecBindings>(bound)) return;

    fail_all_allocations = true;
    const bool matches = std::get<ResolvedCodecBindings>(bound).matches_package(*fixture.package);
    fail_all_allocations = false;
    CHECK(matches);

    std::optional<CodecBindingPreflightResult> failed;
    bool escaped = false;
    fail_all_allocations = true;
    try {
        failed.emplace(preflight_codec_bindings(*fixture.package));
    } catch (const std::bad_alloc&) {
        escaped = true;
    }
    fail_all_allocations = false;
    CHECK(!escaped);
    CHECK(failed.has_value());
    if (failed) {
        const auto* report = std::get_if<CompatibilityReport>(&*failed);
        CHECK(report != nullptr);
        if (report) CHECK(report->code == CompatibilityError::PACKAGE_BOUNDS_INVALID);
    }
}

} // namespace

int main() {
    test_valid_output_is_fingerprint_bound_and_ordered();
    test_certificate_identity_and_diagnostic_authority();
    test_package_fingerprint_changes_with_artifact_bytes();
    test_replay_is_rejected();
    test_authority_rejects_malformed_span_before_preflight();
    test_certificate_rejects_index_valid_misalignment();
    test_persistent_allocation_failure_is_atomic();
    return test_summary("test_codec_binding");
}
