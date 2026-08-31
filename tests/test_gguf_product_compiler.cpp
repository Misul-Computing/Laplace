#include <array>
#include <cstdint>
#include <unistd.h>

#include "artifact_set.h"
#include "gguf.h"
#include "gguf_product_compiler.h"
#include "gguf_writer.h"
#include "test_util.h"

using namespace Laplace;

namespace {

std::string write_source() {
    char path[] = "/private/tmp/laplace-product-compiler-source-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd >= 0) close(fd);

    gguf_writer::Writer writer;
    writer.kv_u32("general.quantization_version", 2);
    gguf_writer::TensorDecl tensor;
    tensor.name = "tensor";
    tensor.dims = {1};
    tensor.type = static_cast<uint32_t>(GGMLType::F32);
    tensor.data.resize(sizeof(float), 0);
    writer.add_tensor(std::move(tensor));
    CHECK(writer.write_file(path));
    return path;
}

void test_untyped_tensor_stays_fail_closed() {
    const std::string path = write_source();
    auto loaded = ArtifactSet::load_single_file(path);
    unlink(path.c_str());
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    if (!std::holds_alternative<ArtifactSet>(loaded)) return;

    ArtifactSet artifacts = std::get<ArtifactSet>(std::move(loaded));
    auto view = artifacts.view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return;

    const auto result = compile_gguf_product_source(
        std::get<PackageView>(std::move(view)));
    CHECK(std::holds_alternative<CompatibilityReport>(result));
    if (const auto* report = std::get_if<CompatibilityReport>(&result)) {
        CHECK(report->code == CompatibilityError::IMPORT_TENSOR_UNMAPPED);
        CHECK(report->stage == CompatibilityStage::Import);
        CHECK(report->detail ==
              "GGUF source tensor has no typed role evidence: tensor");
    }
}

} // namespace

int main() {
    test_untyped_tensor_stays_fail_closed();
    return test_summary("test_gguf_product_compiler");
}
