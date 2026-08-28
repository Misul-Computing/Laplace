#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <variant>
#include <vector>

#include <sys/stat.h>

#include "artifact_set.h"
#include "canonical_metal.h"
#include "compat_rule.h"
#include "generation_loop.h"
#include "gguf.h"
#include "mlx_package.h"
#include "sampler.h"
#include "tokenizer.h"

using namespace Laplace;

namespace {

struct RunOptions {
    std::string package_path;
    std::string prompt;
    int token_count = 128;
    int maximum_context = 2048;
    bool chat = true;
    bool raw_channels = false;
    bool benchmark = false;
    bool inspect_only = true;
    SamplerParams sampler{.temperature = 0.7f, .top_k = 40, .top_p = 0.9f};
};

void print_usage(const char* program) {
    std::fprintf(stderr,
                 "Laplace - Apple Silicon canonical Metal inference\n"
                 "usage: %s <package> [-p prompt] [-n tokens] [--max-seq tokens]\n"
                 "       [--greedy] [-t temperature] [--top-k count] [--top-p probability]\n"
                 "       [--seed value] [--no-chat] [--raw-channels] [--no-spec] [--bench]\n",
                 program);
}

int print_report(const char* operation, const CompatibilityReport& report) {
    std::fprintf(stderr, "%s: code=%u phase=%u operator=%u tensor=%u detail=%s\n",
                 operation, static_cast<unsigned>(report.code),
                 static_cast<unsigned>(report.phase), report.operator_id,
                 report.tensor_id, report.detail.c_str());
    return 1;
}

int reject_non_gguf_package(const std::string& path) {
    auto physical = load_mlx_physical_package(path);
    if (const auto* report = std::get_if<CompatibilityReport>(&physical))
        return print_report("package detection", *report);
    return print_report("package semantics", std::get<MlxPhysicalPackage>(physical).semantic_refusal());
}

bool requires_single_token_metal_prefill(const SemanticModel& semantics) {
    const bool recurrent_operator = std::any_of(
        semantics.operators.begin(), semantics.operators.end(), [](const SemanticOperator& op) {
            return op.kind == OperatorKind::GatedDeltaNet;
        });
    const bool recurrent_state = std::any_of(
        semantics.states.begin(), semantics.states.end(), [](const SemanticState& state) {
            return state.kind == StateKind::RecurrentConvHistory ||
                   state.kind == StateKind::RecurrentDeltaMatrix;
        });
    return recurrent_operator || recurrent_state;
}

int run_canonical(const RunOptions& options) {
    struct stat status {};
    if (stat(options.package_path.c_str(), &status) == 0 && S_ISDIR(status.st_mode))
        return reject_non_gguf_package(options.package_path);

    auto artifacts = ArtifactSet::load_single_file(options.package_path);
    if (const auto* report = std::get_if<CompatibilityReport>(&artifacts))
        return print_report("package detection", *report);
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    if (const auto* report = std::get_if<CompatibilityReport>(&view))
        return print_report("package detection", *report);
    const PackageView package_view = std::get<PackageView>(view);
    auto loaded = load_validated_gguf(package_view);
    if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
        if (report->code == CompatibilityError::PACKAGE_BAD_MAGIC)
            return reject_non_gguf_package(options.package_path);
        return print_report("GGUF semantic import", *report);
    }
    const std::shared_ptr<const RuntimePackage> package =
        std::get<ValidatedPackage>(loaded).runtime_package();
    auto created = create_canonical_metal_program(
        package, static_cast<uint32_t>(options.maximum_context));
    if (const auto* report = std::get_if<CompatibilityReport>(&created))
        return print_report("canonical Metal plan", *report);
    CanonicalMetalProgram program = std::get<CanonicalMetalProgram>(std::move(created));
    std::fprintf(stderr, "canonical Metal plan: layers=%u vocab=%u state=FP32-global\n",
                 program.layer_count(), package->semantics().vocabulary_size);
    if (options.inspect_only) return 0;

    GGUFContext tokenizer_context;
    Tokenizer tokenizer;
    if (!tokenizer_context.parse(package_view.bytes()) || !tokenizer.init(tokenizer_context)) {
        std::fprintf(stderr, "TOKENIZER_RUNTIME_UNSUPPORTED: tokenizer initialization failed\n");
        return 1;
    }
    std::vector<int> encoded;
    if (options.chat && ((tokenizer.im_start_id() >= 0 && tokenizer.im_end_id() >= 0) ||
                         (tokenizer.turn_start_id() >= 0 && tokenizer.turn_end_id() >= 0))) {
        encoded = tokenizer.encode_chat(options.prompt);
    } else {
        if (tokenizer.bos_id() >= 0) encoded.push_back(tokenizer.bos_id());
        const std::vector<int> prompt_tokens = tokenizer.encode(options.prompt);
        encoded.insert(encoded.end(), prompt_tokens.begin(), prompt_tokens.end());
    }
    if (encoded.empty() || encoded.size() >= static_cast<size_t>(options.maximum_context)) {
        std::fprintf(stderr, "RUNTIME_INPUT_INVALID: prompt has %zu tokens, max is %d\n",
                     encoded.size(), options.maximum_context);
        return 1;
    }
    std::vector<uint32_t> prompt_tokens;
    prompt_tokens.reserve(encoded.size());
    for (int token : encoded) {
        if (token < 0 || static_cast<uint32_t>(token) >= package->semantics().vocabulary_size) {
            std::fprintf(stderr, "RUNTIME_INPUT_INVALID: tokenizer produced a token outside the package vocabulary\n");
            return 1;
        }
        prompt_tokens.push_back(static_cast<uint32_t>(token));
    }

    CanonicalMetalOutput logits;
    uint64_t prefill_command_buffers = 0;
    uint64_t decode_command_buffers = 0;
    const auto prefill_start = std::chrono::steady_clock::now();
    if (requires_single_token_metal_prefill(package->semantics())) {
        for (uint32_t token : prompt_tokens) {
            auto prefill = program.prefill(std::span<const uint32_t>(&token, 1));
            if (const auto* report = std::get_if<CompatibilityReport>(&prefill))
                return print_report("canonical Metal recurrent prefill", *report);
            logits = std::get<CanonicalMetalOutput>(std::move(prefill));
            prefill_command_buffers += logits.command_buffers;
        }
    } else {
        auto prefill = program.prefill(prompt_tokens);
        if (const auto* report = std::get_if<CompatibilityReport>(&prefill))
            return print_report("canonical Metal prefill", *report);
        logits = std::get<CanonicalMetalOutput>(std::move(prefill));
        prefill_command_buffers = logits.command_buffers;
    }
    const auto prefill_end = std::chrono::steady_clock::now();

    GenerationMetrics metrics;
    record_prefill(&metrics, prompt_tokens.size());
    Sampler sampler(options.sampler);
    auto emit = [&](int token) {
        if (!options.raw_channels &&
            (token == tokenizer.channel_start_id() || token == tokenizer.channel_end_id())) return;
        const std::string piece = tokenizer.decode(token);
        std::fputs(piece.c_str(), stdout);
    };
    auto stop = [&](int token) {
        return token == tokenizer.eos_id() || token == tokenizer.im_end_id() || token == tokenizer.turn_end_id();
    };
    int current = sampler.sample(logits.logits.data(), static_cast<int>(logits.logits.size()));
    int emitted = 0;
    const auto decode_start = std::chrono::steady_clock::now();
    while (!stop(current) && emitted < options.token_count &&
           program.position() < static_cast<uint32_t>(options.maximum_context)) {
        emit(current);
        std::fflush(stdout);
        ++emitted;
        record_emitted(&metrics, 1);
        if (emitted == options.token_count) break;
        auto result = program.decode(static_cast<uint32_t>(current));
        if (const auto* report = std::get_if<CompatibilityReport>(&result))
            return print_report("canonical Metal decode", *report);
        logits = std::get<CanonicalMetalOutput>(std::move(result));
        decode_command_buffers += logits.command_buffers;
        record_decode_forward(&metrics);
        current = sampler.sample(logits.logits.data(), static_cast<int>(logits.logits.size()));
        if (!stop(current)) record_decode_output(&metrics, 1);
    }
    const auto decode_end = std::chrono::steady_clock::now();
    std::fputc('\n', stdout);
    if (options.benchmark) {
        const double prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
        const double decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
        std::fprintf(stderr, "[bench] prefill: %zu tokens in %.1f ms (%.1f tok/s)\n",
                     prompt_tokens.size(), prefill_ms,
                     1000.0 * prompt_tokens.size() / std::max(1.0, prefill_ms));
        finalize_decode_metrics(&metrics, decode_ms);
        std::fputs(render_decode_benchmark(metrics, decode_ms).c_str(), stderr);
        std::fprintf(stderr, "[bench] Metal command buffers: prefill=%llu decode=%llu\n",
                     static_cast<unsigned long long>(prefill_command_buffers),
                     static_cast<unsigned long long>(decode_command_buffers));
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    RunOptions options;
    options.package_path = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto next = [&]() -> const char* {
            return index + 1 < argc ? argv[++index] : nullptr;
        };
        if (argument == "-p") {
            const char* value = next();
            if (!value) return print_usage(argv[0]), 1;
            options.prompt = value;
            options.inspect_only = false;
        } else if (argument == "-n") {
            const char* value = next();
            if (!value) return print_usage(argv[0]), 1;
            options.token_count = std::atoi(value);
        } else if (argument == "--max-seq") {
            const char* value = next();
            if (!value) return print_usage(argv[0]), 1;
            options.maximum_context = std::atoi(value);
        } else if (argument == "-t") {
            const char* value = next();
            if (!value) return print_usage(argv[0]), 1;
            options.sampler.temperature = std::atof(value);
        } else if (argument == "--top-k") {
            const char* value = next();
            if (!value) return print_usage(argv[0]), 1;
            options.sampler.top_k = std::atoi(value);
        } else if (argument == "--top-p") {
            const char* value = next();
            if (!value) return print_usage(argv[0]), 1;
            options.sampler.top_p = std::atof(value);
        } else if (argument == "--seed") {
            const char* value = next();
            if (!value) return print_usage(argv[0]), 1;
            options.sampler.seed = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
        } else if (argument == "--greedy") {
            options.sampler.temperature = 0.0f;
        } else if (argument == "--no-chat") {
            options.chat = false;
        } else if (argument == "--raw-channels") {
            options.raw_channels = true;
        } else if (argument == "--no-spec") {
            continue;
        } else if (argument == "--bench") {
            options.benchmark = true;
        } else if (argument == "--laplace-stream") {
            std::fprintf(stderr, "STREAMING_UNSUPPORTED: canonical Metal does not admit streamed KV\n");
            return 1;
        } else if (argument == "--laplace-kv" || argument == "--laplace-resident" ||
                   argument == "--laplace-kv-q4" || argument == "--kv-fp16" ||
                   argument == "--kv-fp32") {
            std::fprintf(stderr, "CACHE_MODE_UNQUALIFIED: canonical Metal owns FP32 global state only\n");
            return 1;
        } else {
            std::fprintf(stderr, "RUNTIME_INPUT_INVALID: unknown option %s\n", argument.c_str());
            return 1;
        }
    }
    if (options.token_count < 0 || options.maximum_context <= 0) {
        std::fprintf(stderr, "RUNTIME_INPUT_INVALID: token and context limits must be positive\n");
        return 1;
    }
    return run_canonical(options);
}
