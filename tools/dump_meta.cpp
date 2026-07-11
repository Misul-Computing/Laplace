// dump_meta.cpp - dump all GGUF metadata for a model
#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "gguf.h"
#include "tensor.h"

using namespace Laplace;

static void print_variant(const MetaValue& v) {
    std::visit([](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::string>) {
            printf("\"%s\"", arg.c_str());
        } else if constexpr (std::is_same_v<T, bool>) {
            printf("%s", arg ? "true" : "false");
        } else if constexpr (std::is_same_v<T, MetaArrayStr>) {
            printf("[");
            for (size_t i = 0; i < arg.size(); i++) {
                if (i) printf(", ");
                printf("\"%s\"", arg[i].c_str());
            }
            printf("]");
        } else if constexpr (std::is_same_v<T, MetaArrayU32> || std::is_same_v<T, MetaArrayI32>) {
            printf("[");
            for (size_t i = 0; i < arg.size(); i++) {
                if (i) printf(", ");
                printf("%d", arg[i]);
            }
            printf("]");
        } else if constexpr (std::is_same_v<T, MetaArrayU64> || std::is_same_v<T, MetaArrayI64>) {
            printf("[");
            for (size_t i = 0; i < arg.size(); i++) {
                if (i) printf(", ");
                printf("%lld", (long long)arg[i]);
            }
            printf("]");
        } else if constexpr (std::is_same_v<T, MetaArrayF32> || std::is_same_v<T, MetaArrayF64>) {
            printf("[");
            for (size_t i = 0; i < arg.size(); i++) {
                if (i) printf(", ");
                printf("%g", (double)arg[i]);
            }
            printf("]");
        } else {
            printf("%g", (double)arg);
        }
    }, v);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 1; }
    GGUFContext ctx;
    if (!ctx.open(argv[1])) { fprintf(stderr, "failed to open %s\n", argv[1]); return 1; }
    const auto& m = ctx.metadata();
    std::vector<std::string> keys;
    keys.reserve(m.size());
    for (const auto& [k, _] : m) keys.push_back(k);
    std::sort(keys.begin(), keys.end());
    for (const auto& k : keys) {
        // Skip huge vocab arrays for readability
        if (k.rfind("tokenizer.ggml.", 0) == 0) {
            if (k == "tokenizer.ggml.model" || k == "tokenizer.ggml.bos_token_id" ||
                k == "tokenizer.ggml.eos_token_id" || k == "tokenizer.ggml.unknown_token_id" ||
                k == "tokenizer.ggml.padding_token_id" || k == "tokenizer.ggml.add_bos_token" ||
                k == "tokenizer.ggml.add_eos_token" || k == "tokenizer.ggml.token_type_count" ||
                k == "tokenizer.ggml.prefix_token_id" || k == "tokenizer.ggml.suffix_token_id" ||
                k == "tokenizer.ggml.separator_token_id") {
                printf("%-55s = ", k.c_str());
                print_variant(m.at(k));
                printf("\n");
            }
            continue;
        }
        printf("%-55s = ", k.c_str());
        print_variant(m.at(k));
        printf("\n");
    }
    return 0;
}
