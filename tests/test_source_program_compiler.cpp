#define main source_fixture_tests_main
#include "test_gguf_product_compiler.cpp"
#undef main

#include "source_program_compiler.h"
#include <sys/wait.h>

void test_compiled_source_package(uint32_t query_heads, bool chat_template = false) {
    const auto path = write_dense_source(std::nullopt, query_heads, true, chat_template);
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return;
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return;
    auto source = compile_gguf_product_source(std::get<PackageView>(view));
    CHECK(std::holds_alternative<GgufProductCompilation>(source));
    if (!std::holds_alternative<GgufProductCompilation>(source)) return;
    const auto& compiled = std::get<GgufProductCompilation>(source);
    if (chat_template) {
        CHECK(!compiled.token_program.definition().turn.empty());
        const auto turn = compiled.token_program.render_turn("a");
        CHECK(std::holds_alternative<std::string>(turn));
        if (const auto* text = std::get_if<std::string>(&turn))
            CHECK(*text == "</turn><|user|>a</turn><|assistant|>");
    }
    auto result = compile_source_program_package(compiled.manifest,
                                                 compiled.token_program, 8);
    const auto* error = std::get_if<CompatibilityReport>(&result);
    CHECK_MSG(!error, "source package: %s", error ? error->detail.c_str() : "none");
    if (const auto* package = std::get_if<VerifiedProgramPackage>(&result)) {
        CHECK(package->complete());
        CHECK(package->state_schema().slots().size() == 3);
        CHECK(!package->physical_package().resources().empty());
        CHECK(package->physical_package().physical_index().artifacts().size() == 3);
        bool retained = false;
        for (const auto& artifact : package->physical_package().physical_index().artifacts())
            if (artifact.artifact_id().value == 0)
                retained = artifact.bytes().data() == std::get<PackageView>(view).bytes().data();
        CHECK(retained);
    }
    CHECK(std::holds_alternative<CompatibilityReport>(compile_source_program_package(
        compiled.manifest, compiled.token_program, 65)));
    CHECK(std::holds_alternative<CompatibilityReport>(compile_source_program_package(
        compiled.manifest, TokenProgram::token_ids_only(4), 8)));
    for (uint32_t requested : {0u, 8u, 65u}) {
        const auto loaded = load_source_program_package(path, requested);
        if (requested == 65) {
            CHECK(std::holds_alternative<CompatibilityReport>(loaded));
        } else {
            CHECK(std::holds_alternative<LoadedSourceProgram>(loaded));
            if (const auto* package = std::get_if<LoadedSourceProgram>(&loaded)) {
                CHECK(package->max_context == (requested ? requested : 64));
                CHECK(package->eos_id == 1);
                CHECK(package->stop_ids == compiled.manifest.semantic_model().stop_ids);
            }
        }
    }
    unlink(path.c_str());
}

void test_chat_cli() {
    const auto path = write_dense_source(std::nullopt, 1, true, false, true);
    const auto run = [&](const std::string& input, const std::string& args) {
        const std::string command = "printf '" + input + "' | '" LAPLACE_BINARY_PATH
            "' '" + path + "' " + args + " 2>/dev/null";
        FILE* pipe = popen(command.c_str(), "r");
        CHECK(pipe != nullptr);
        std::string output;
        char buffer[256];
        while (pipe && fgets(buffer, sizeof(buffer), pipe)) output += buffer;
        const int status = pipe ? pclose(pipe) : -1;
        CHECK(status >= 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0);
        return output;
    };
    CHECK(run("", "--raw-prompt --greedy --max-seq 8 -n 2").empty());
    CHECK(run("a\\na\\n", "--raw-prompt --greedy --max-seq 8 -n 2") == "aa\naa\n");
    CHECK(run("a\\na\\n", "--raw-prompt --greedy --max-seq 3 -n 10") == "aa\n");
    CHECK(run("aaaaaaaa\\na\\n", "--raw-prompt --greedy --max-seq 3 -n 10") == "aa\n");
    CHECK(run("", "-p a --raw-prompt --greedy --max-seq 3 -n 10") == "aa\n");
    CHECK(run("a\\n", "--greedy --max-seq 3 -n 10") == "a\n");
    CHECK(run("", "--info").find("file:") != std::string::npos);
    unlink(path.c_str());
}

int main() {
    test_compiled_source_package(1);
    test_compiled_source_package(2);
    test_compiled_source_package(1, true);
    test_chat_cli();
    return test_summary("test_source_program_compiler");
}
