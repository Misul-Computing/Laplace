#include <array>
#include <cstdio>
#include <fcntl.h>
#include <memory>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include "artifact_set.h"
#include "compat_rule.h"
#include "semantic_manifest.h"
#include "source_schema.h"
#include "test_util.h"

#if __has_include("source_schema_authority.h")
#error "the removed source-schema authority surface must not be public"
#endif

using namespace Laplace;

static_assert(!std::is_constructible_v<RuntimePackage, ArtifactIndex>);
static_assert(!std::is_constructible_v<RuntimePackage, SemanticManifest>);
static_assert(!std::is_constructible_v<RuntimePackage, SourceSchemaResult>);
static_assert(!std::is_convertible_v<SourceSchemaResult, std::shared_ptr<const RuntimePackage>>);
static_assert(!std::is_convertible_v<ArtifactIndex, std::shared_ptr<const RuntimePackage>>);
static_assert(!std::is_convertible_v<SemanticManifest, std::shared_ptr<const RuntimePackage>>);

namespace {

struct Fixture {
    std::string path;
    std::shared_ptr<const RuntimePackage> package;
};

std::string temporary_path() {
    char path[] = "/private/tmp/laplace-authority-negative-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd >= 0) close(fd);
    return path;
}

Fixture make_fixture() {
    Fixture fixture;
    fixture.path = temporary_path();
    const std::array<uint8_t, 256> bytes = [] {
        std::array<uint8_t, 256> value{};
        value.fill(0x5a);
        return value;
    }();
    const int fd = open(fixture.path.c_str(), O_WRONLY | O_TRUNC);
    CHECK(fd >= 0);
    if (fd >= 0) {
        CHECK(write(fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size()));
        close(fd);
    }

    auto loaded = ArtifactSet::load_single_file(fixture.path);
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    std::vector<PackageView> artifacts;
    if (auto* set = std::get_if<ArtifactSet>(&loaded)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) artifacts.push_back(*package);
    }

    ArtifactTensorRecord physical;
    physical.id = 0;
    physical.coordinate.root = 0;
    physical.logical_type = ArtifactScalarType::F32;
    physical.logical_dimensions = {2, 2};
    physical.layout.rank = 2;
    physical.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    physical.layout.strides = {2, 1, 0, 0, 0, 0, 0, 0};
    physical.axis.row_stride_bytes = 8;
    physical.format = {1, ArtifactPhysicalEncoding::F32, ArtifactScalarType::F32,
                       ArtifactScalarType::None, ArtifactScalarType::None,
                       ArtifactScalarType::None, ArtifactScalarType::None, 1, 4, 0, 0, 0, 0};
    physical.planes.push_back({PlaneKind::Values, ArtifactScalarType::F32,
                               {ArtifactId{0}, 64, 16}, 4, 4, 1, 4});

    ArtifactIndexInput input;
    input.artifacts = std::move(artifacts);
    input.tensors.push_back(std::move(physical));
    auto built = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<ArtifactIndex>(built));
    if (!std::holds_alternative<ArtifactIndex>(built)) return fixture;
    ArtifactIndex index = std::get<ArtifactIndex>(std::move(built));

    SemanticModel model;
    model.maximum_context = 32768;
    model.vocabulary_size = 4;
    model.bos_id = 1;
    model.eos_id = 2;
    model.stop_ids = {2};
    for (size_t i = 0; i != model.tokenizer_digest.size(); ++i) {
        model.tokenizer_digest[i] = static_cast<uint8_t>(i + 1);
    }
    SemanticTensor tensor;
    tensor.id = 0;
    tensor.role = TensorRole::TokenEmbedding;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, 2}, {DimensionKind::Constant, 2}};
    tensor.layout.rank = 2;
    tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides = {2, 1, 0, 0, 0, 0, 0, 0};
    tensor.planes.push_back({PlaneKind::Values, ScalarType::F32, ArtifactId{0}, 64, 16, 4, 0});
    model.tensors.push_back(std::move(tensor));
    TokenIdContract contract;
    contract.vocabulary_size = model.vocabulary_size;
    contract.bos_id = model.bos_id;
    contract.eos_id = model.eos_id;
    contract.stop_ids = model.stop_ids;
    contract.authoritative_tokenizer_digest = {model.tokenizer_digest};
    contract.authoritative_template_digest = {model.template_digest};
    auto manifest = SemanticManifest::build(index, model, contract);
    CHECK(std::holds_alternative<SemanticManifest>(manifest));
    if (const auto* report = std::get_if<CompatibilityReport>(&manifest)) {
        std::fprintf(stderr, "authority-negative fixture: code=%u detail=%s\n",
                     static_cast<unsigned>(report->code), report->detail.c_str());
    }
    if (auto* value = std::get_if<SemanticManifest>(&manifest)) {
        fixture.package = RuntimePackage::make_diagnostic(std::move(*value), Sha256Digest{}, 0);
    }
    return fixture;
}

void test_caller_inputs_remain_diagnostic() {
    Fixture fixture = make_fixture();
    CHECK(fixture.package != nullptr);
    if (fixture.package) {
        CHECK(fixture.package->authority_kind() == PackageAuthorityKind::DiagnosticRaw);
        CHECK(!fixture.package->product_authoritative());
        CHECK(fixture.package->physical_index().tensors().size() == 1);
        remove(fixture.path.c_str());
        CHECK(fixture.package->artifact_bytes(ArtifactId{0}).size() == 256);
    }
    remove(fixture.path.c_str());
}

void test_public_source_schema_result_is_diagnostic_only() {
    SourceSchemaResult public_result;
    public_result.error = SourceSchemaError::NoMatchingSchema;
    CHECK(!public_result.success());
}

} // namespace

int main() {
    test_caller_inputs_remain_diagnostic();
    test_public_source_schema_result_is_diagnostic_only();
    return test_summary("test_source_schema_authority");
}
