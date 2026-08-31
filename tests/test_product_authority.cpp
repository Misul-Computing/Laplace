#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <string>
#include <variant>

#include "artifact_set.h"
#include "compat_rule.h"
#include "test_util.h"

using namespace Laplace;

namespace {

std::string temporary_path() {
    char path[] = "/private/tmp/laplace-product-authority-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd >= 0) {
        const std::array<uint8_t, 96> bytes{};
        CHECK(write(fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size()));
        close(fd);
    }
    return path;
}

void test_carried_manifest_factory_is_separate_from_diagnostics() {
    const std::string primary_path = temporary_path();
    const std::string carrier_path = temporary_path();
    const std::array<ArtifactSource, 2> sources = {
        ArtifactSource{primary_path, ArtifactRole::Primary, ArtifactId{0}},
        ArtifactSource{carrier_path, ArtifactRole::Sidecar, ArtifactId{1}},
    };
    auto graph = ArtifactSet::load_graph(sources);
    CHECK(std::holds_alternative<ArtifactSet>(graph));
    if (auto* artifacts = std::get_if<ArtifactSet>(&graph)) {
        auto carrier = artifacts->view(ArtifactId{1});
        CHECK(std::holds_alternative<PackageView>(carrier));
        if (auto* view = std::get_if<PackageView>(&carrier)) {
            auto loaded = load_carried_manifest(ArtifactIndex{}, *view);
            CHECK(std::holds_alternative<CompatibilityReport>(loaded));
            if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
                CHECK(report->code == CompatibilityError::PACKAGE_BAD_MAGIC);
            }
        }
    }
    unlink(primary_path.c_str());
    unlink(carrier_path.c_str());
}

} // namespace

int main() {
    test_carried_manifest_factory_is_separate_from_diagnostics();
    return test_summary("test_product_authority");
}
