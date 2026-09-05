// Laplace - an Apple Silicon LLM inference engine.
//
// A from-scratch C++20 inference engine with one capability-qualified
// canonical Metal execution route.
//
// Usage:
//   laplace <model-path> [-p prompt] [--prompt-file PATH] [-n N]
//          [-t T] [--top-k K] [--top-p P] [--greedy] [--seed N]
//          [--max-seq L] [--program-manifest PATH] [--raw-prompt] [--bench]
//
// With no prompt options, starts chat. --info prints model metadata.
#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <variant>
#include <vector>

#include "gguf.h"
#include "generation_loop.h"
#include "source_program_compiler.h"
#include "program_package.h"
#include "runtime_session.h"
#include "sampler.h"


using namespace Laplace;

static int run_dump(const std::string& path);
static int run_generate(const std::string& path,
                        const std::string& program_manifest,
                        const std::string& prompt,
                        int n_tokens,
                        const SamplerParams& sp,
                        std::optional<uint32_t> requested_max_seq,
                        bool use_template,
                        bool bench,
                        bool interactive);

static bool parse_int_value(const char* text, int minimum, int maximum, int& value) {
    if (!text || *text == '\0') return false;
    const std::string_view input(text);
    const auto parsed = std::from_chars(input.data(), input.data() + input.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == input.data() + input.size() &&
           value >= minimum && value <= maximum;
}

static bool parse_seed_value(const char* text, uint32_t& value) {
    if (!text || *text == '\0') return false;
    const std::string_view input(text);
    const auto parsed = std::from_chars(input.data(), input.data() + input.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == input.data() + input.size();
}

static bool parse_float_value(const char* text, double minimum, double maximum, double& value) {
    if (!text || *text == '\0') return false;
    if (std::isspace(static_cast<unsigned char>(*text))) return false;
    errno = 0;
    char* end = nullptr;
    value = std::strtod(text, &end);
    return end != text && *end == '\0' && errno != ERANGE && std::isfinite(value) &&
           value >= minimum && value <= maximum;
}

static void print_help(const char* program) {
    printf(
        "laplace - Apple Silicon native Metal inference engine\n"
        "\n"
        "usage: %s <model-path> [options]\n"
        "       %s -- <model-path> [options]   when the model path starts with '-'\n"
        "\n"
        "With no prompt options, starts chat. /help shows commands; /exit or Ctrl-D exits.\n"
        "\n"
        "prompt:\n"
        "  -p <text>              generate from the given prompt text\n"
        "  --prompt-file <path>   read prompt text from a file (max 16 MB)\n"
        "  --info                 print model metadata and exit\n"
        "  --raw-prompt           tokenize the prompt as-is, without the\n"
        "                         model's chat template\n"
        "  --program-manifest <path>\n"
        "                         load a verified program package instead of\n"
        "                         compiling the model package on the fly\n"
        "\n"
        "generation:\n"
        "  -n <count>             tokens per response (default: 200)\n"
        "  --max-seq <length>     maximum sequence length, prompt included\n"
        "                         (default: the package context, capped at 2048)\n"
        "  --bench                report prefill and decode timing separately\n"
        "\n"
        "sampling:\n"
        "  -t <value>             temperature; 0 selects greedy decoding\n"
        "                         (default: 0.7)\n"
        "  --top-k <count>        top-k filter, 0 = off (default: 40)\n"
        "  --top-p <value>        nucleus probability mass, 0 < p <= 1\n"
        "                         (default: 0.9)\n"
        "  --greedy               greedy decoding; same as -t 0\n"
        "  --seed <value>         sampler seed; 0 picks a fresh seed each run\n"
        "                         (default: 0)\n"
        "\n"
        "examples:\n"
        "  %s /path/to/model.gguf              start a conversation\n"
        "  %s /path/to/model.gguf --info       inspect a model\n"
        "  %s /path/to/model.gguf -p \"Hello, Laplace\" -n 32 \\\n"
        "      --greedy --seed 7 --max-seq 2048 --bench\n",
        program, program, program, program, program);
}

static constexpr size_t kMaxPromptFileBytes = 16 * 1024 * 1024;

static bool read_prompt_file(const std::string& path, std::string& prompt) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (!file) {
        fprintf(stderr, "cannot open prompt file %s\n", path.c_str());
        return false;
    }
    struct stat info{};
    if (fstat(fileno(file), &info) != 0 || !S_ISREG(info.st_mode)) {
        fprintf(stderr, "prompt file must be a regular file: %s\n", path.c_str());
        std::fclose(file);
        return false;
    }
    if (std::fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "cannot seek prompt file %s\n", path.c_str());
        std::fclose(file);
        return false;
    }
    const long size = std::ftell(file);
    if (size < 0 || static_cast<uint64_t>(size) > kMaxPromptFileBytes) {
        fprintf(stderr, "prompt file is too large or has an invalid size: %s\n", path.c_str());
        std::fclose(file);
        return false;
    }
    if (std::fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "cannot seek prompt file %s\n", path.c_str());
        std::fclose(file);
        return false;
    }
    prompt.assign(static_cast<size_t>(size), '\0');
    const size_t read = prompt.empty() ? 0 : std::fread(prompt.data(), 1, prompt.size(), file);
    if (read != prompt.size() || std::ferror(file)) {
        fprintf(stderr, "cannot read prompt file %s\n", path.c_str());
        std::fclose(file);
        return false;
    }
    if (std::fclose(file) != 0) {
        fprintf(stderr, "cannot close prompt file %s\n", path.c_str());
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_help(argv[0]);
        return 0;
    }
    if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_help(argv[0]);
        return 0;
    }

    int first_option = 2;
    std::string model_path = argv[1];
    if (model_path == "--") {
        if (argc < 3) {
            fprintf(stderr, "missing model path after --\n");
            return 1;
        }
        model_path = argv[2];
        first_option = 3;
    } else if (!model_path.empty() && model_path.front() == '-') {
        fprintf(stderr, "unknown option: %s\n", model_path.c_str());
        return 1;
    }

    std::string prompt;
    std::string prompt_file;
    bool prompt_from_arg = false;
    bool prompt_from_file = false;
    std::string program_manifest;
    int n_tokens = 200;
    std::optional<uint32_t> requested_max_seq;
    bool use_template = true;
    SamplerParams sp;
    sp.temperature = 0.7f;
    sp.top_k = 40;
    sp.top_p = 0.9f;
    bool bench = false;
    bool info_only = false;

    for (int i = first_option; i < argc; i++) {
        const std::string option = argv[i];
        auto next_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (option == "-p") {
            const char* value = next_value("-p");
            if (!value) return 1;
            prompt = value;
            prompt_from_arg = true;
        } else if (option == "--prompt-file") {
            const char* value = next_value("--prompt-file");
            if (!value) return 1;
            prompt_file = value;
            prompt_from_file = true;
        } else if (option == "-h" || option == "--help") {
            print_help(argv[0]);
            return 0;
        } else if (option == "-n") {
            const char* value = next_value("-n");
            if (!value) return 1;
            if (!parse_int_value(value, 0, std::numeric_limits<int>::max(), n_tokens)) {
                fprintf(stderr, "invalid value for -n: %s\n", value);
                return 1;
            }
        } else if (option == "-t") {
            const char* value = next_value("-t");
            if (!value) return 1;
            double parsed = 0.0;
            if (!parse_float_value(value, 0.0, std::numeric_limits<float>::max(), parsed) ||
                !std::isfinite(static_cast<float>(parsed))) {
                fprintf(stderr, "invalid value for -t: %s\n", value);
                return 1;
            }
            sp.temperature = static_cast<float>(parsed);
        } else if (option == "--top-k") {
            const char* value = next_value("--top-k");
            if (!value) return 1;
            if (!parse_int_value(value, 0, std::numeric_limits<int>::max(), sp.top_k)) {
                fprintf(stderr, "invalid value for --top-k: %s\n", value);
                return 1;
            }
        } else if (option == "--top-p") {
            const char* value = next_value("--top-p");
            if (!value) return 1;
            double parsed = 0.0;
            if (!parse_float_value(value, 0.0, 1.0, parsed) || parsed == 0.0) {
                fprintf(stderr, "invalid value for --top-p: %s\n", value);
                return 1;
            }
            sp.top_p = static_cast<float>(parsed);
        } else if (option == "--greedy") {
            sp.temperature = 0.0f;
        } else if (option == "--seed") {
            const char* value = next_value("--seed");
            if (!value) return 1;
            if (!parse_seed_value(value, sp.seed)) {
                fprintf(stderr, "invalid value for --seed: %s\n", value);
                return 1;
            }
        } else if (option == "--max-seq") {
            const char* value = next_value("--max-seq");
            if (!value) return 1;
            int parsed = 0;
            if (!parse_int_value(value, 1, std::numeric_limits<int>::max(), parsed)) {
                fprintf(stderr, "invalid value for --max-seq: %s\n", value);
                return 1;
            }
            requested_max_seq = static_cast<uint32_t>(parsed);
        } else if (option == "--program-manifest") {
            const char* value = next_value("--program-manifest");
            if (!value) return 1;
            program_manifest = value;
        } else if (option == "--raw-prompt") {
            use_template = false;
        } else if (option == "--info") {
            info_only = true;
        } else if (option == "--bench") {
            bench = true;
        } else {
            fprintf(stderr, "unknown option: %s\n", option.c_str());
            return 1;
        }
    }

    if (prompt_from_arg && prompt_from_file) {
        fprintf(stderr, "use either -p or --prompt-file, not both\n");
        return 1;
    }
    if (info_only && (prompt_from_arg || prompt_from_file)) {
        fprintf(stderr, "--info takes no prompt\n");
        return 1;
    }
    if (!prompt_file.empty() && !read_prompt_file(prompt_file, prompt)) return 1;
    struct stat model_status {};
    if (stat(model_path.c_str(), &model_status) != 0) {
        fprintf(stderr, "cannot open model %s\n", model_path.c_str());
        fprintf(stderr, "get a test model: python3 scripts/download_model.py\n");
        return 1;
    }
    if (info_only) return run_dump(model_path);
    return run_generate(model_path, program_manifest, prompt, n_tokens, sp,
                        requested_max_seq, use_template, bench,
                        !prompt_from_arg && !prompt_from_file);
}

static int run_dump(const std::string& path) {
    GGUFContext ctx;
    if (!ctx.open(path.c_str())) { fprintf(stderr, "cannot load %s\n", path.c_str()); return 1; }
    const auto& m = ctx.metadata();

    auto arch = meta_str(m, "general.architecture");
    printf("file:    %s\n", path.c_str());
    printf("size:    %.2f MB\n", ctx.file_size() / 1e6);
    if (arch) printf("arch:    %s\n", arch->c_str());
    if (auto p = meta_str(m, "general.name")) printf("name:    %s\n", p->c_str());
    if (auto p = meta_uint32(m, "general.file_type")) printf("file_type: %u\n", *p);
    if (auto p = meta_uint32(m, "general.quantization_version")) printf("quant_version: %u\n", *p);
    printf("\n");
    return 0;
}

static int run_generate(const std::string& path,
                        const std::string& program_manifest,
                        const std::string& prompt,
                        int n_tokens,
                        const SamplerParams& sp,
                        std::optional<uint32_t> requested_max_seq,
                        bool use_template,
                        bool bench,
                        bool interactive) {
    if (n_tokens < 0) {
        fprintf(stderr, "canonical: invalid generation bounds\n");
        return 1;
    }

    if (interactive) fprintf(stderr, "Loading model...\n");
    std::optional<VerifiedProgramPackage> program;
    const TokenProgram* tokenizer = nullptr;
    uint32_t vocabulary_size = 0;
    uint32_t eos_id = kTokenProgramNoTokenId;
    std::vector<uint32_t> stop_ids;
    uint32_t route_units = 0;
    const char* route_name = nullptr;
    bool device_greedy = false;
    uint32_t max_seq = 0;

    SessionRequest request;
    request.max_batch = 1;
    request.memory_limit = UINT64_MAX;
    request.enable_prefill = true;
    request.enable_decode = true;
    request.minimum_class = NumericalClass::ExactFp32;
    request.objective = RuntimeObjective::Latency;

    std::optional<RuntimeSession> session_owner;
    if (!program_manifest.empty()) {
        auto loaded = load_program_package(path, program_manifest);
        if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
            fprintf(stderr, "program package: code=%u artifact=%u detail=%s\n",
                    static_cast<unsigned>(report->code), report->artifact_id.value,
                    report->detail.c_str());
            return 1;
        }
        program.emplace(
            std::get<VerifiedProgramPackage>(std::move(loaded)));
        tokenizer = &program->token_program();
        vocabulary_size = static_cast<uint32_t>(
            tokenizer->definition().vocabulary.size());
        eos_id = tokenizer->definition().postprocessor.eos_token_id;
        stop_ids = tokenizer->definition().stop_ids;
        max_seq = requested_max_seq.value_or(2048);
        request.max_context = max_seq;
        auto created = create_runtime_session(*program, request);
        if (const auto* report = std::get_if<CompatibilityReport>(&created)) {
            fprintf(stderr, "runtime session: code=%u operator=%u detail=%s\n",
                    static_cast<unsigned>(report->code), report->operator_id,
                    report->detail.c_str());
            return 1;
        }
        session_owner.emplace(
            std::get<RuntimeSession>(std::move(created)));
        route_units = static_cast<uint32_t>(
            program_definition(program->semantic_program()).functions.size());
        route_name = "program";
    } else {
        auto loaded = load_source_program_package(path, requested_max_seq.value_or(0));
        if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
            fprintf(stderr, "source program: code=%u artifact=%u detail=%s\n",
                    static_cast<unsigned>(report->code), report->artifact_id.value,
                    report->detail.c_str());
            return 1;
        }
        auto source = std::get<LoadedSourceProgram>(std::move(loaded));
        max_seq = source.max_context;
        eos_id = source.eos_id;
        stop_ids = std::move(source.stop_ids);
        program.emplace(std::move(source.package));
        tokenizer = &program->token_program();
        vocabulary_size = static_cast<uint32_t>(
            tokenizer->definition().vocabulary.size());
        request.max_context = max_seq;
        auto created = create_runtime_session(*program, request);
        if (const auto* report = std::get_if<CompatibilityReport>(&created)) {
            fprintf(stderr, "runtime session: code=%u operator=%u detail=%s\n",
                    static_cast<unsigned>(report->code), report->operator_id,
                    report->detail.c_str());
            return 1;
        }
        session_owner.emplace(
            std::get<RuntimeSession>(std::move(created)));
        route_units = static_cast<uint32_t>(
            program_definition(program->semantic_program()).functions.size());
        route_name = "program";
    }
    if (!tokenizer || !session_owner || vocabulary_size == 0 || max_seq == 0)
        return 1;
    RuntimeSession session = std::move(*session_owner);
    if (!interactive || bench) {
        fprintf(stderr, "[%s] Metal session: %u program units, vocab=%u, transactional state\n",
                route_name, route_units, vocabulary_size);
        fprintf(stderr, "[%s] sampling: %s\n", route_name,
                device_greedy ? "Metal greedy (16-byte host result)" : "host sampling (full logits)");
    }

    bool first_turn = true;
    uint32_t previous_stop = kTokenProgramNoTokenId;
    size_t previous_reply_tokens = 0;
    const bool plain_framing = !use_template || tokenizer->definition().turn.empty();
    if (interactive) {
        fprintf(stderr, "Ready. Type a message; /help for commands, /exit or Ctrl-D to leave.\n");
        if (use_template && plain_framing)
            fprintf(stderr, "chat: no supported chat template; using plain text\n");
    }
    std::optional<Sampler> sampler;
    if (!device_greedy) sampler.emplace(sp);
    while (true) {
        std::string input = prompt;
        if (interactive) {
            if (session.token_history().size() >= max_seq) {
                fprintf(stderr, "context is full (%u tokens); start a new session\n", max_seq);
                break;
            }
            fprintf(stderr, "> ");
            fflush(stderr);
            if (!std::getline(std::cin, input)) break;
            if (!input.empty() && input.back() == '\r') input.pop_back();
            if (input.empty()) continue;
            if (input == "/exit") break;
            if (input == "/help") {
                fprintf(stderr, "Enter one message per line. Conversation history stays in this session.\n"
                                "/help  show this help\n"
                                "/exit  leave chat (or press Ctrl-D)\n"
                                "Context: %zu of %u tokens used.\n",
                        session.token_history().size(), max_seq);
                continue;
            }
        }
        std::string rendered_prompt = interactive && plain_framing && !first_turn
            ? "\n" + input : input;
        if (use_template && (first_turn || !plain_framing || !interactive)) {
            auto rendered = first_turn ? tokenizer->render_prompt(input)
                                       : tokenizer->render_turn(input);
            if (const auto* status = std::get_if<TokenProgramStatus>(&rendered)) {
                fprintf(stderr, "tokenizer prompt: code=%u detail=%s\n",
                        static_cast<unsigned>(status->error), status->detail.c_str());
                return 1;
            }
            rendered_prompt = std::get<std::string>(std::move(rendered));
        }
        auto encoded = first_turn ? tokenizer->encode(rendered_prompt)
                                  : tokenizer->encode_continuation(rendered_prompt);
        if (const auto* status = std::get_if<TokenProgramStatus>(&encoded)) {
            fprintf(stderr, "tokenizer encode: code=%u detail=%s\n",
                    static_cast<unsigned>(status->error), status->detail.c_str());
            return 1;
        }
        std::vector<uint32_t> prompt_tokens = std::get<std::vector<uint32_t>>(std::move(encoded));
        if (interactive && !first_turn && !plain_framing &&
            previous_stop != kTokenProgramNoTokenId &&
            !tokenizer->definition().turn.empty() &&
            tokenizer->definition().turn.front().opcode == PromptOpcode::EmitLiteralUtf8) {
            const auto close = tokenizer->encode_continuation(
                tokenizer->definition().turn.front().literal);
            if (const auto* tokens = std::get_if<std::vector<uint32_t>>(&close)) {
                const auto& history = session.token_history();
                const size_t overlap = emitted_close_prefix(
                    std::span<const uint32_t>(history).last(previous_reply_tokens), *tokens, previous_stop);
                if (overlap <= prompt_tokens.size() &&
                    std::equal(tokens->begin(), tokens->begin() + overlap, prompt_tokens.begin()))
                    prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.begin() + overlap);
            }
        }
        if (prompt_tokens.empty() || prompt_tokens.size() >=
            static_cast<size_t>(max_seq) - session.token_history().size()) {
            fprintf(stderr, "prompt empty or too long (%zu tokens, %zu remaining)\n",
                    prompt_tokens.size(), max_seq - session.token_history().size());
            if (interactive) continue;
            return 1;
        }
        for (uint32_t token : prompt_tokens) {
            if (token >= vocabulary_size) {
                fprintf(stderr, "RUNTIME_INPUT_INVALID: tokenizer produced token outside canonical vocabulary\n");
                return 1;
            }
        }
        if (!interactive || bench)
            fprintf(stderr, "prompt: %zu tokens\n", prompt_tokens.size());

        uint64_t prefill_commands = 0;
        uint64_t decode_commands = 0;
        double prefill_gpu_ms = 0.0;
        double prefill_wait_ms = 0.0;
        double decode_gpu_ms = 0.0;
        double decode_wait_ms = 0.0;
        RuntimeOutput logits;
        GenerationMetrics generation;
        const auto prefill_start = std::chrono::steady_clock::now();
        const std::span<const uint32_t> prompt_span(prompt_tokens.data(), prompt_tokens.size());
        auto executed = device_greedy ? session.prefill_sampled(prompt_span)
                                      : session.prefill(prompt_span);
        if (const auto* report = std::get_if<CompatibilityReport>(&executed)) {
            fprintf(stderr, "canonical prefill: code=%u operator=%u detail=%s\n",
                    static_cast<unsigned>(report->code), report->operator_id, report->detail.c_str());
            return 1;
        }
        logits = std::get<RuntimeOutput>(std::move(executed));
        prefill_commands += logits.command_buffers;
        prefill_gpu_ms += logits.gpu_time_ms;
        prefill_wait_ms += logits.cpu_wait_ms;
        const auto prefill_end = std::chrono::steady_clock::now();
        record_prefill(&generation, prompt_tokens.size());

        TokenProgram::StreamState decode_state;
        auto emit = [&](uint32_t token) {
            const std::array<uint32_t, 1> one = {token};
            auto decoded = tokenizer->decode_chunk(one, decode_state);
            if (const auto* status = std::get_if<TokenProgramStatus>(&decoded)) {
                fprintf(stderr, "tokenizer decode: code=%u detail=%s\n",
                        static_cast<unsigned>(status->error), status->detail.c_str());
                return false;
            }
            const std::string& piece = std::get<std::string>(decoded);
            std::fwrite(piece.data(), 1, piece.size(), stdout);
            return true;
        };
        const auto is_stop = [&](int token) {
            if (token < 0) return true;
            const uint32_t value = static_cast<uint32_t>(token);
            if (eos_id != kTokenProgramNoTokenId && value == eos_id) return true;
            return std::find(stop_ids.begin(), stop_ids.end(), value) != stop_ids.end();
        };

        const auto select_token = [&](const RuntimeOutput& output, int& token) {
            if (device_greedy) {
                if (!output.sampled || output.host_result_bytes != 16 ||
                    output.sampled_token_id >= vocabulary_size ||
                    output.sampled_token_id > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
                    fprintf(stderr, "canonical sampler returned an invalid Metal result\n");
                    return false;
                }
                token = static_cast<int>(output.sampled_token_id);
                return true;
            }
            if (output.sampled || !sampler || output.logits.empty() ||
                output.logits.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
                fprintf(stderr, "canonical sampler did not receive full host logits\n");
                return false;
            }
            token = sampler->sample(output.logits.data(), static_cast<int>(output.logits.size()));
            return token >= 0;
        };
        int current = -1;
        if (!select_token(logits, current)) return 1;
        int generated = 0;
        bool stopped = is_stop(current);
        if (!stopped && generated < n_tokens) {
            if (!emit(static_cast<uint32_t>(current))) return 1;
            fflush(stdout);
            ++generated;
            record_emitted(&generation, 1);
        }
        const auto decode_output_start = std::chrono::steady_clock::now();
        double raw_decode_forward_ms = 0.0;
        while (!stopped && generated < n_tokens && session.token_history().size() + 1 < max_seq) {
            const auto forward_start = std::chrono::steady_clock::now();
            auto executed = device_greedy ? session.decode_sampled(static_cast<uint32_t>(current))
                                          : session.decode(static_cast<uint32_t>(current));
            const auto forward_end = std::chrono::steady_clock::now();
            if (const auto* report = std::get_if<CompatibilityReport>(&executed)) {
                fprintf(stderr, "canonical decode: code=%u operator=%u detail=%s\n",
                        static_cast<unsigned>(report->code), report->operator_id, report->detail.c_str());
                return 1;
            }
            logits = std::get<RuntimeOutput>(std::move(executed));
            decode_commands += logits.command_buffers;
            decode_gpu_ms += logits.gpu_time_ms;
            decode_wait_ms += logits.cpu_wait_ms;
            record_decode_forward(&generation);
            raw_decode_forward_ms +=
                std::chrono::duration<double, std::milli>(forward_end - forward_start).count();
            if (!select_token(logits, current)) return 1;
            if (is_stop(current)) {
                stopped = true;
                break;
            }
            if (!emit(static_cast<uint32_t>(current))) return 1;
            fflush(stdout);
            ++generated;
            record_decode_output(&generation, 1);
            record_emitted(&generation, 1);
        }
        const auto decode_output_end = std::chrono::steady_clock::now();
        auto tail = tokenizer->decode_chunk({}, decode_state, true);
        if (const auto* text = std::get_if<std::string>(&tail))
            std::fwrite(text->data(), 1, text->size(), stdout);
        else {
            const auto& status = std::get<TokenProgramStatus>(tail);
            fprintf(stderr, "tokenizer stream: %s\n", status.detail.c_str());
            return 1;
        }
        printf("\n");
        if (stopped && (!interactive || bench)) fprintf(stderr, "[eos]\n");
        if (bench) {
            fprintf(stderr, "[bench] diagnostic (unqualified)\n");
            const double prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
            const double decode_output_ms =
                std::chrono::duration<double, std::milli>(decode_output_end - decode_output_start).count();
            fprintf(stderr, "[bench] prefill: %zu tokens in %.1f ms (%.1f tok/s)\n", prompt_tokens.size(), prefill_ms,
                    1000.0 * prompt_tokens.size() / std::max(1.0, prefill_ms));
            finalize_decode_metrics(&generation, raw_decode_forward_ms, decode_output_ms);
            fputs(render_decode_benchmark(generation, decode_output_ms).c_str(), stderr);
            fprintf(stderr,
                    "[bench] metal: prefill %llu command buffers, %.3f ms GPU, %.3f ms CPU wait; "
                    "decode %llu command buffers, %.3f ms GPU, %.3f ms CPU wait\n",
                    static_cast<unsigned long long>(prefill_commands), prefill_gpu_ms, prefill_wait_ms,
                    static_cast<unsigned long long>(decode_commands), decode_gpu_ms, decode_wait_ms);
        }
        if (!interactive) break;
        // The final emitted token has not entered the cache yet.
        if (!stopped && generated > 0 && session.token_history().size() < max_seq) {
            auto fed = session.decode(static_cast<uint32_t>(current));
            if (const auto* report = std::get_if<CompatibilityReport>(&fed)) {
                fprintf(stderr, "chat decode: %s\n", report->detail.c_str());
                return 1;
            }
        }
        previous_stop = stopped ? static_cast<uint32_t>(current) : kTokenProgramNoTokenId;
        previous_reply_tokens = std::min<size_t>(generated, session.token_history().size());
        first_turn = false;
    }
    return 0;
}
