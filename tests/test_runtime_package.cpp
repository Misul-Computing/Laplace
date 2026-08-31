#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include "artifact_set.h"
#include "compat_rule.h"
#include "semantic_manifest.h"
#include "test_util.h"

using namespace Laplace;

static_assert(!std::is_constructible_v<RuntimePackage, SemanticManifest, Sha256Digest,
                                       uint32_t, RuleQualificationState>);
static_assert(!std::is_constructible_v<ValidatedPackage,
                                       std::shared_ptr<const RuntimePackage>,
                                       DiagnosticProvenance>);

namespace {

struct Fixture {
    std::array<std::string, 2> paths;
    std::string carrier_path;
    ArtifactIndex index;
    SemanticModel model;
    TokenIdContract contract;
    std::shared_ptr<const RuntimePackage> package;
};

std::string temporary_path() {
    char path[] = "/private/tmp/laplace-runtime-package-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd >= 0) close(fd);
    return path;
}

ArtifactTensorRecord tensor_record(uint32_t id, uint32_t root) {
    ArtifactTensorRecord tensor;
    tensor.id = id;
    tensor.coordinate.root = root;
    tensor.logical_type = ArtifactScalarType::F32;
    tensor.logical_dimensions = {2, 2};
    tensor.layout.rank = 2;
    tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides = {2, 1, 0, 0, 0, 0, 0, 0};
    tensor.format = {1, ArtifactPhysicalEncoding::F32, ArtifactScalarType::F32,
                     ArtifactScalarType::None, ArtifactScalarType::None,
                     ArtifactScalarType::None, ArtifactScalarType::None, 1, 4, 0, 0, 0, 0};
    tensor.axis.row_stride_bytes = 8;
    tensor.planes.push_back({PlaneKind::Values, ArtifactScalarType::F32,
                             {ArtifactId{id}, 64, 16}, 4, 4, 1, 4});
    return tensor;
}

SemanticTensor semantic_tensor(uint32_t id, uint32_t artifact_id) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = id == 0 ? TensorRole::QueryWeight : TensorRole::OutputWeight;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, 2}, {DimensionKind::Constant, 2}};
    tensor.layout.rank = 2;
    tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides = {2, 1, 0, 0, 0, 0, 0, 0};
    tensor.planes.push_back({PlaneKind::Values, ScalarType::F32, ArtifactId{artifact_id}, 64, 16, 4, 0});
    return tensor;
}

Fixture make_fixture(uint8_t second_artifact_byte = 0x72) {
    Fixture fixture;
    std::array<std::vector<uint8_t>, 2> contents = {
        std::vector<uint8_t>(256, 0x31), std::vector<uint8_t>(256, second_artifact_byte)};
    std::array<ArtifactSource, 2> sources{};
    for (size_t index = 0; index != fixture.paths.size(); ++index) {
        fixture.paths[index] = temporary_path();
        const int fd = open(fixture.paths[index].c_str(), O_WRONLY | O_TRUNC);
        CHECK(fd >= 0);
        if (fd >= 0) {
            CHECK(write(fd, contents[index].data(), contents[index].size()) ==
                  static_cast<ssize_t>(contents[index].size()));
            close(fd);
        }
        sources[index] = {fixture.paths[index], index == 0 ? ArtifactRole::Primary : ArtifactRole::Shard,
                          ArtifactId{static_cast<uint32_t>(index)}};
    }
    fixture.carrier_path = temporary_path();

    auto loaded = ArtifactSet::load_graph(sources);
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    std::vector<PackageView> artifacts;
    if (auto* set = std::get_if<ArtifactSet>(&loaded)) {
        for (uint32_t id = 0; id != 2; ++id) {
            auto view = set->view(ArtifactId{id});
            CHECK(std::holds_alternative<PackageView>(view));
            if (auto* package = std::get_if<PackageView>(&view)) artifacts.push_back(*package);
        }
    }

    ArtifactIndexInput input;
    input.artifacts = artifacts;
    input.tensors.push_back(tensor_record(0, 0));
    input.tensors.push_back(tensor_record(1, 1));
    auto index_result = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<ArtifactIndex>(index_result));
    if (auto* report = std::get_if<CompatibilityReport>(&index_result)) {
        fprintf(stderr, "runtime package fixture index error: %u %s\n",
                static_cast<unsigned>(report->code), report->detail.c_str());
    }
    if (!std::holds_alternative<ArtifactIndex>(index_result)) return fixture;
    ArtifactIndex index = std::get<ArtifactIndex>(std::move(index_result));

    SemanticModel model;
    model.maximum_context = 32768;
    model.vocabulary_size = 4;
    model.bos_id = 1;
    model.eos_id = 2;
    model.stop_ids = {2};
    for (size_t i = 0; i != model.tokenizer_digest.size(); ++i) model.tokenizer_digest[i] = static_cast<uint8_t>(i + 1);
    model.tensors = {semantic_tensor(0, 0), semantic_tensor(1, 1)};
    for (uint32_t id = 0; id != 5; ++id) {
        model.values.push_back({id, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0});
    }
    model.input_values_first = 0;
    model.input_values_count = 2;
    model.output_values_first = 4;
    model.output_values_count = 1;
    model.operators = {
        {0, OperatorKind::Linear, 1, {0}, {2}, {0}, {}, LinearPayload{}},
        {1, OperatorKind::Linear, 1, {1}, {3}, {1}, {}, LinearPayload{}},
        {2, OperatorKind::Add, 1, {2, 3}, {4}, {}, {}, AddPayload{}},
    };
    model.layers = {{0, 0, 3, 0}};
    TokenIdContract contract;
    contract.vocabulary_size = model.vocabulary_size;
    contract.bos_id = model.bos_id;
    contract.eos_id = model.eos_id;
    contract.stop_ids = model.stop_ids;
    contract.authoritative_tokenizer_digest = {model.tokenizer_digest};
    contract.authoritative_template_digest = {model.template_digest};

    fixture.index = index;
    fixture.model = model;
    fixture.contract = contract;

    auto manifest_result = SemanticManifest::build(index, model, contract);
    CHECK(std::holds_alternative<SemanticManifest>(manifest_result));
    if (!std::holds_alternative<SemanticManifest>(manifest_result)) return fixture;
    SemanticManifest manifest = std::get<SemanticManifest>(std::move(manifest_result));
    fixture.package = RuntimePackage::make_diagnostic(
        std::move(manifest), Sha256Digest{}, 0);
    return fixture;
}

void remove_fixture(Fixture& fixture) {
    for (const std::string& path : fixture.paths) unlink(path.c_str());
    if (!fixture.carrier_path.empty()) unlink(fixture.carrier_path.c_str());
}

void write_bytes(const std::string& path, std::span<const uint8_t> bytes) {
    const int fd = open(path.c_str(), O_WRONLY | O_TRUNC);
    CHECK(fd >= 0);
    if (fd >= 0) {
        CHECK(write(fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size()));
        close(fd);
    }
}

uint64_t get64(const std::vector<uint8_t>& bytes, size_t offset) {
    uint64_t result = 0;
    for (unsigned shift = 0; shift != 64; shift += 8) {
        result |= static_cast<uint64_t>(bytes[offset + shift / 8]) << shift;
    }
    return result;
}

Sha256Digest sha256(std::span<const uint8_t> bytes) {
    Sha256Digest result;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), result.bytes.data());
    return result;
}

void reseal(std::vector<uint8_t>& bytes) {
    const size_t body_length = static_cast<size_t>(get64(bytes, 16));
    const Sha256Digest body = sha256(std::span<const uint8_t>(bytes).subspan(64, body_length));
    std::copy(body.bytes.begin(), body.bytes.end(), bytes.begin() + 32);
    const Sha256Digest record = sha256(std::span<const uint8_t>(bytes).first(bytes.size() - 32));
    std::copy(record.bytes.begin(), record.bytes.end(), bytes.end() - 32);
}

void reseal_semantic_payload(std::vector<uint8_t>& bytes) {
    constexpr std::array<uint8_t, 8> semantic_magic = {'L', 'A', 'P', 'I', 'R', '0', '0', '1'};
    const auto semantic = std::search(bytes.begin(), bytes.end(), semantic_magic.begin(), semantic_magic.end());
    CHECK(semantic != bytes.end());
    if (semantic == bytes.end()) return;
    const size_t semantic_start = static_cast<size_t>(semantic - bytes.begin());
    const size_t semantic_body_length = static_cast<size_t>(get64(bytes, semantic_start + 16));
    CHECK(semantic_body_length > 0);
    if (semantic_body_length == 0) return;
    bytes[semantic_start + 64 + semantic_body_length - 1] ^= 1;
    const Sha256Digest semantic_body = sha256(
        std::span<const uint8_t>(bytes).subspan(semantic_start + 64, semantic_body_length));
    std::copy(semantic_body.bytes.begin(), semantic_body.bytes.end(), bytes.begin() + semantic_start + 32);
    reseal(bytes);
}

std::variant<PackageView, CompatibilityReport> load_carrier_view(const Fixture& fixture) {
    std::array<ArtifactSource, 3> sources = {
        ArtifactSource{fixture.paths[0], ArtifactRole::Primary, ArtifactId{0}},
        ArtifactSource{fixture.paths[1], ArtifactRole::Shard, ArtifactId{1}},
        ArtifactSource{fixture.carrier_path, ArtifactRole::Sidecar, ArtifactId{2}},
    };
    auto loaded = ArtifactSet::load_graph(sources);
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    if (auto* set = std::get_if<ArtifactSet>(&loaded)) return set->view(ArtifactId{2});
    return package_report(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                          "runtime package test fixture sidecar failed to load");
}

void test_manifest_owns_complete_artifact_closure() {
    Fixture fixture = make_fixture();
    Fixture alternate = make_fixture(0x73);
    CHECK(fixture.package != nullptr);
    CHECK(alternate.package != nullptr);
    if (!fixture.package || !alternate.package) {
        remove_fixture(fixture);
        remove_fixture(alternate);
        return;
    }
    CHECK(fixture.package->manifest().physical_index().artifacts().size() == 2);
    CHECK(fixture.package->authority_kind() == PackageAuthorityKind::DiagnosticRaw);
    CHECK(!fixture.package->product_authoritative());
    CHECK(fixture.package->artifact_bytes(ArtifactId{0}).size() == 256);
    CHECK(fixture.package->artifact_bytes(ArtifactId{1}).size() == 256);
    CHECK(fixture.package->artifact_bytes(ArtifactId{0})[0] == 0x31);
    CHECK(fixture.package->artifact_bytes(ArtifactId{1})[0] == 0x72);
    CHECK(fixture.package->artifact_bytes(ArtifactId{99}).empty());
    CHECK(fixture.package->semantic_graph_digest() == fixture.package->manifest().semantic_graph_digest());
    CHECK(fixture.package->physical_binding_set_digest() == fixture.package->manifest().physical_binding_set_digest());
    CHECK(fixture.package->interaction_contract_digest() == fixture.package->manifest().interaction_contract_digest());
    CHECK(fixture.package->package_fingerprint() == fixture.package->manifest().package_fingerprint());
    CHECK(fixture.package->fingerprint() == fixture.package->package_fingerprint());
    CHECK(fixture.package->semantic_graph_digest() == alternate.package->semantic_graph_digest());
    CHECK(fixture.package->interaction_contract_digest() == alternate.package->interaction_contract_digest());
    CHECK(fixture.package->physical_binding_set_digest() == alternate.package->physical_binding_set_digest());
    CHECK(fixture.package->package_fingerprint() != alternate.package->package_fingerprint());
    remove_fixture(fixture);
    remove_fixture(alternate);
}

void test_carried_manifest_requires_a_sidecar_carrier() {
    Fixture fixture = make_fixture();
    CHECK(fixture.package != nullptr);
    if (fixture.package) {
        auto carried = SemanticManifest::decode_carried(
            fixture.package->physical_index(),
            fixture.package->physical_index().artifacts()[0]);
        CHECK(std::holds_alternative<CompatibilityReport>(carried));
    }
    remove_fixture(fixture);
}

void test_carried_manifest_requires_a_complete_graph_proof() {
    Fixture fixture = make_fixture();
    CHECK(fixture.package != nullptr);
    if (!fixture.package) {
        remove_fixture(fixture);
        return;
    }
    SemanticModel incomplete = fixture.model;
    incomplete.values.clear();
    incomplete.operators.clear();
    incomplete.layers.clear();
    incomplete.input_values_first = 0;
    incomplete.input_values_count = 0;
    incomplete.output_values_first = 0;
    incomplete.output_values_count = 0;
    auto manifest_result = SemanticManifest::build(fixture.index, incomplete, fixture.contract);
    CHECK(std::holds_alternative<SemanticManifest>(manifest_result));
    if (!std::holds_alternative<SemanticManifest>(manifest_result)) {
        remove_fixture(fixture);
        return;
    }
    const SemanticManifest& incomplete_manifest = std::get<SemanticManifest>(manifest_result);
    const std::vector<uint8_t> bytes(incomplete_manifest.bytes().begin(),
                                     incomplete_manifest.bytes().end());
    write_bytes(fixture.carrier_path, bytes);
    auto carrier_result = load_carrier_view(fixture);
    CHECK(std::holds_alternative<PackageView>(carrier_result));
    if (auto* carrier = std::get_if<PackageView>(&carrier_result)) {
        auto decoded = SemanticManifest::decode_carried(fixture.index, *carrier);
        CHECK(std::holds_alternative<CompatibilityReport>(decoded));
        if (auto* report = std::get_if<CompatibilityReport>(&decoded)) {
            CHECK(report->code == CompatibilityError::IMPORT_CLOSURE_INCOMPLETE);
        }
    }
    remove_fixture(fixture);
}

void test_carried_manifest_is_diagnostic_without_a_trust_certificate() {
    Fixture fixture = make_fixture();
    CHECK(fixture.package != nullptr);
    if (!fixture.package) {
        remove_fixture(fixture);
        return;
    }
    const std::vector<uint8_t> bytes(fixture.package->manifest().bytes().begin(),
                                     fixture.package->manifest().bytes().end());
    write_bytes(fixture.carrier_path, bytes);
    auto carrier_result = load_carrier_view(fixture);
    CHECK(std::holds_alternative<PackageView>(carrier_result));
    if (!std::holds_alternative<PackageView>(carrier_result)) {
        remove_fixture(fixture);
        return;
    }
    const PackageView carrier = std::get<PackageView>(carrier_result);
    auto manifest_result = SemanticManifest::decode_carried(fixture.index, carrier);
    CHECK(std::holds_alternative<SemanticManifest>(manifest_result));
    if (!std::holds_alternative<SemanticManifest>(manifest_result)) {
        remove_fixture(fixture);
        return;
    }
    CHECK(std::get<SemanticManifest>(manifest_result).has_carrier());

    auto loaded = load_carried_manifest(fixture.index, carrier);
    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
    if (auto* validated = std::get_if<ValidatedPackage>(&loaded)) {
        const auto runtime = validated->runtime_package();
        CHECK(runtime != nullptr);
        if (runtime) {
            CHECK(runtime->authority_kind() == PackageAuthorityKind::DiagnosticRaw);
            CHECK(!runtime->product_authoritative());
            CHECK(runtime->qualification_state() == RuleQualificationState::Draft);
            CHECK(runtime->rule_revision() == 0);
            CHECK(runtime->rule_fingerprint() == Sha256Digest{});
            CHECK(runtime->artifact_bytes(ArtifactId{0}).size() == 256);
            CHECK(runtime->artifact_bytes(ArtifactId{1}).size() == 256);
            CHECK(runtime->manifest().has_carrier());
            CHECK(std::vector<uint8_t>(runtime->manifest().bytes().begin(),
                                       runtime->manifest().bytes().end()) == bytes);
        }
    }
    unlink(fixture.carrier_path.c_str());
    unlink(fixture.paths[0].c_str());
    unlink(fixture.paths[1].c_str());
    if (auto* validated = std::get_if<ValidatedPackage>(&loaded)) {
        const auto runtime = validated->runtime_package();
        CHECK(runtime != nullptr);
        if (runtime) {
            CHECK(runtime->artifact_bytes(ArtifactId{0})[0] == 0x31);
            CHECK(runtime->artifact_bytes(ArtifactId{1})[0] == 0x72);
            CHECK(runtime->manifest().bytes().size() == bytes.size());
            CHECK(runtime->manifest().bytes()[0] == 'L');
        }
    }
    remove_fixture(fixture);
}

void test_carried_manifest_rejects_bad_sidecar_and_wrong_closure() {
    Fixture fixture = make_fixture();
    CHECK(fixture.package != nullptr);
    if (!fixture.package) {
        remove_fixture(fixture);
        return;
    }
    const std::vector<uint8_t> original(fixture.package->manifest().bytes().begin(),
                                        fixture.package->manifest().bytes().end());
    for (const auto& mutation : {
             std::vector<uint8_t>(original.begin(), original.end() - 1),
             [&] {
                 auto bytes = original;
                 bytes[64] ^= 1;
                 return bytes;
             }(),
             [&] {
                 auto bytes = original;
                 reseal_semantic_payload(bytes);
                 return bytes;
             }()}) {
        write_bytes(fixture.carrier_path, mutation);
        auto carrier_result = load_carrier_view(fixture);
        CHECK(std::holds_alternative<PackageView>(carrier_result));
        if (auto* carrier = std::get_if<PackageView>(&carrier_result)) {
            auto decoded = SemanticManifest::decode_carried(fixture.index, *carrier);
            CHECK(std::holds_alternative<CompatibilityReport>(decoded));
        }
    }

    write_bytes(fixture.carrier_path, original);
    auto carrier_result = load_carrier_view(fixture);
    CHECK(std::holds_alternative<PackageView>(carrier_result));
    Fixture alternate = make_fixture(0x73);
    if (auto* carrier = std::get_if<PackageView>(&carrier_result)) {
        auto wrong_closure = load_carried_manifest(alternate.index, *carrier);
        CHECK(std::holds_alternative<CompatibilityReport>(wrong_closure));
        if (auto* report = std::get_if<CompatibilityReport>(&wrong_closure)) {
            CHECK(report->code == CompatibilityError::PACKAGE_SOURCE_CHANGED);
        }
    }
    CHECK(std::holds_alternative<CompatibilityReport>(SemanticManifest::decode(fixture.index, {})));
    remove_fixture(fixture);
    remove_fixture(alternate);
}

} // namespace

int main() {
    test_manifest_owns_complete_artifact_closure();
    test_carried_manifest_requires_a_sidecar_carrier();
    test_carried_manifest_requires_a_complete_graph_proof();
    test_carried_manifest_is_diagnostic_without_a_trust_certificate();
    test_carried_manifest_rejects_bad_sidecar_and_wrong_closure();
    return test_summary("test_runtime_package");
}
