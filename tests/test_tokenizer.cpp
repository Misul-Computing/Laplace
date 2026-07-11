// test_tokenizer - byte-level BPE: pre-tokenization, merge order, decode
// inverse mapping, and chat template construction.
//
// The primary invariant is exact round-trip: concatenating decode() of every
// token of encode(text) must reproduce text byte-for-byte (the vocab contains
// all 256 byte-level symbols, so nothing is ever unencodable).

#include <cstdio>
#include <string>
#include <vector>

#include "gguf.h"
#include "tokenizer.h"

#include "gguf_writer.h"
#include "test_util.h"

using namespace Laplace;

namespace {

// GPT-2 byte -> unicode mapping, reimplemented from the spec (independent of
// the engine's table). Printable bytes map to themselves; the rest to
// U+0100 + n in order.
std::string byte_to_uni_spec(int b) {
    bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    int cp;
    if (printable) {
        cp = b;
    } else {
        int n = 0;
        for (int x = 0; x < b; x++) {
            bool pr = (x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174 && x <= 255);
            if (!pr) n++;
        }
        cp = 0x100 + n;
    }
    std::string s;
    if (cp < 0x80) {
        s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return s;
}

struct TestVocab {
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    int id(const std::string& s) const {
        for (size_t i = 0; i < tokens.size(); i++)
            if (tokens[i] == s) return static_cast<int>(i);
        return -1;
    }
};

TestVocab build_vocab() {
    TestVocab v;
    // All 256 byte-level symbols first (like every real GPT-2-style vocab).
    for (int b = 0; b < 256; b++) v.tokens.push_back(byte_to_uni_spec(b));

    const std::string G = byte_to_uni_spec(' ');   // "Ġ"

    // Merged tokens. Every merge result must exist as a token.
    for (const char* t : {"he", "ll", "llo", "hello",
                          "or", "ld"}) v.tokens.push_back(t);
    v.tokens.push_back(G + "w");
    v.tokens.push_back(G + "wor");
    v.tokens.push_back(G + "world");
    v.tokens.push_back("ab");   // exists in vocab but has NO merge rule
    v.tokens.push_back("bc");
    v.tokens.push_back("<|im_start|>");
    v.tokens.push_back("<|im_end|>");

    v.merges = {
        "h e", "l l", "ll o", "he llo",
        "o r", "l d",
        G + " w",          // Ġ + w
        G + "w or",        // Ġw + or
        G + "wor ld",      // Ġwor + ld
        "b c",             // NOTE: "a b" is intentionally absent
    };
    return v;
}

std::string write_vocab_gguf(const TestVocab& v, const char* path) {
    gguf_writer::Writer w;
    w.kv_str("general.architecture", "test");
    w.kv_arr_str("tokenizer.ggml.tokens", v.tokens);
    w.kv_arr_str("tokenizer.ggml.merges", v.merges);
    w.kv_u32("tokenizer.ggml.bos_token_id", static_cast<uint32_t>(v.id("<|im_start|>")));
    w.kv_u32("tokenizer.ggml.eos_token_id", static_cast<uint32_t>(v.id("<|im_end|>")));
    if (!w.write_file(path)) return "";
    return path;
}

std::string decode_all(const Tokenizer& tok, const std::vector<int>& ids) {
    std::string out;
    for (int id : ids) out += tok.decode(id);
    return out;
}

void test_round_trip(const Tokenizer& tok) {
    const char* cases[] = {
        "hello world",
        "hello  world",            // double space must not be duplicated/lost
        "a\nb",                    // newline must survive encode->decode
        "line one\n\nline two\n",
        "tab\there",
        "punct, and; more!",
        "  leading and trailing  ",
        "mixed 123 numbers42",
        "non-ascii: caf\xC3\xA9 \xE2\x82\xAC",  // UTF-8 é and €
    };
    for (const char* c : cases) {
        std::vector<int> ids = tok.encode(c);
        std::string back = decode_all(tok, ids);
        CHECK_MSG(back == c, "round trip '%s' -> '%s'", c, back.c_str());
    }
}

void test_bpe_merge_order(const Tokenizer& tok, const TestVocab& v) {
    // "abc": merges contain "b c" but not "a b". Correct BPE yields [a, bc].
    // (Greedy longest-match would wrongly yield [ab, c].)
    std::vector<int> ids = tok.encode("abc");
    CHECK_MSG(ids.size() == 2, "encode('abc') gave %zu tokens", ids.size());
    if (ids.size() == 2) {
        CHECK_MSG(ids[0] == v.id("a") && ids[1] == v.id("bc"),
                  "encode('abc') = [%d, %d], expected [%d, %d]",
                  ids[0], ids[1], v.id("a"), v.id("bc"));
    }

    // "hello world" must use the merge chain to reach the merged tokens.
    const std::string G = byte_to_uni_spec(' ');
    std::vector<int> hw = tok.encode("hello world");
    CHECK_MSG(hw.size() == 2, "encode('hello world') gave %zu tokens", hw.size());
    if (hw.size() == 2) {
        CHECK(hw[0] == v.id("hello"));
        CHECK(hw[1] == v.id(G + "world"));
    }
}

void test_special_ids(const Tokenizer& tok, const TestVocab& v) {
    CHECK(tok.im_start_id() == v.id("<|im_start|>"));
    CHECK(tok.im_end_id()   == v.id("<|im_end|>"));
    CHECK(tok.eos_id()      == v.id("<|im_end|>"));
}

void test_chat_template(const Tokenizer& tok) {
    // The chat sequence must contain the user prompt and decode to the exact
    // Qwen template string.
    std::vector<int> ids = tok.encode_chat("hi there");
    std::string flat = decode_all(tok, ids);
    const char* expected =
        "<|im_start|>user\nhi there<|im_end|>\n<|im_start|>assistant\n";
    CHECK_MSG(flat == expected, "chat template decodes to '%s'", flat.c_str());

    int n_start = 0, n_end = 0;
    for (int id : ids) {
        if (id == tok.im_start_id()) n_start++;
        if (id == tok.im_end_id())   n_end++;
    }
    CHECK(n_start == 2);
    CHECK(n_end == 1);
}

} // namespace

int main() {
    TestVocab v = build_vocab();
    std::string path = write_vocab_gguf(v, "test_tokenizer_vocab.gguf");
    if (path.empty()) {
        fprintf(stderr, "could not write synthetic vocab gguf\n");
        return 2;
    }

    GGUFContext ctx;
    if (!ctx.open(path.c_str())) {
        fprintf(stderr, "could not parse synthetic vocab gguf\n");
        remove(path.c_str());
        return 2;
    }

    Tokenizer tok;
    if (!tok.init(ctx)) {
        fprintf(stderr, "tokenizer init failed\n");
        remove(path.c_str());
        return 2;
    }

    test_round_trip(tok);
    test_bpe_merge_order(tok, v);
    test_special_ids(tok, v);
    test_chat_template(tok);

    remove(path.c_str());
    return test_summary("test_tokenizer");
}
