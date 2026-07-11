// tokenizer.cpp - GPT-2 byte-level BPE tokenizer
#include "tokenizer.h"

#include <cstdio>
#include <cstring>
#include <utility>

namespace Laplace {

namespace {

// GPT-2 / Qwen byte-to-unicode table.
// Printable bytes (33..126, 161..172, 174..255) map to themselves as single
// chars. All other bytes are assigned to codepoints starting at U+0100 in
// order. We precompute the UTF-8 string for each byte.
void build_byte_table(std::string out[256]) {
    auto emit_codepoint = [](int cp, std::string& s) {
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
    };
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool printable = (b >= 33 && b <= 126)
                      || (b >= 161 && b <= 172)
                      || (b >= 174 && b <= 255);
        int cp = printable ? b : (0x100 + n++);
        out[b].clear();
        emit_codepoint(cp, out[b]);
    }
}

// Split `s` (a canonical-unicode string) into its UTF-8 codepoint substrings.
void split_codepoints(const std::string& s, std::vector<std::string>& out) {
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if      (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        if (i + len > s.size()) len = 1;
        out.push_back(s.substr(i, len));
        i += len;
    }
}

// Pre-tokenize text into chunks, approximating the GPT-2/Qwen regex:
//   ' ?[letters]+ | ?[digits]+ | ?[other]+ | whitespace'
// A single space directly before a letter/digit/other run attaches to that
// run; the remainder of a whitespace run is its own chunk, preserved verbatim
// (newlines and tabs are NOT collapsed).
void pre_tokenize(const std::string& text, std::vector<std::string>& out) {
    auto is_alpha = [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    };
    auto is_digit = [](unsigned char c) {
        return c >= '0' && c <= '9';
    };
    auto is_space = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    };

    const int n = static_cast<int>(text.size());
    int i = 0;
    bool pending_space = false;  // a single ' ' that attaches to the next chunk
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        if (is_space(c)) {
            int j = i;
            while (j < n && is_space(static_cast<unsigned char>(text[j]))) j++;
            // If the run ends with ' ' and a word follows, that last space
            // attaches to the word (GPT-2 " ?X" behaviour).
            bool attach_last = (j < n) && (text[j - 1] == ' ');
            int run_end = attach_last ? j - 1 : j;
            if (run_end > i) out.push_back(text.substr(i, run_end - i));
            pending_space = attach_last;
            i = j;
            continue;
        }

        int j = i;
        if (is_alpha(c)) {
            while (j < n && is_alpha(static_cast<unsigned char>(text[j]))) j++;
        } else if (is_digit(c)) {
            while (j < n && is_digit(static_cast<unsigned char>(text[j]))) j++;
        } else {
            // Punctuation / non-ASCII: a run of anything else.
            while (j < n) {
                unsigned char cj = static_cast<unsigned char>(text[j]);
                if (is_space(cj) || is_alpha(cj) || is_digit(cj)) break;
                j++;
            }
        }
        std::string chunk;
        if (pending_space) chunk.push_back(' ');
        chunk.append(text, i, j - i);
        out.push_back(std::move(chunk));
        pending_space = false;
        i = j;
    }
}

} // namespace

bool Tokenizer::init(const GGUFContext& gguf) {
    const auto& m = gguf.metadata();

    build_byte_table(byte_to_uni_);
    uni_to_byte_.reserve(512);
    for (int b = 0; b < 256; b++) uni_to_byte_[byte_to_uni_[b]] = static_cast<unsigned char>(b);

    auto* tok_arr = meta_as<MetaArrayStr>(m, "tokenizer.ggml.tokens");
    if (!tok_arr) {
        fprintf(stderr, "tokenizer: no tokenizer.ggml.tokens array\n");
        return false;
    }
    id_to_token_ = *tok_arr;
    token_to_id_.reserve(id_to_token_.size() * 2);
    for (int i = 0; i < (int)id_to_token_.size(); i++) {
        token_to_id_[id_to_token_[i]] = i;
    }

    // Detect SentencePiece tokenizer (gemma, llama, etc.)
    if (auto* model = meta_str(m, "tokenizer.ggml.model")) {
        is_spm_ = (*model == "gemma4" || *model == "gemma" ||
                   *model == "llama" || *model == "llama3");
    }

    // BPE merge ranks: array of "left right" strings, rank = position.
    if (auto* merges = meta_as<MetaArrayStr>(m, "tokenizer.ggml.merges")) {
        merge_ranks_.reserve(merges->size() * 2);
        for (int r = 0; r < (int)merges->size(); r++) {
            merge_ranks_[(*merges)[r]] = r;
        }
    }

    bos_id_ = -1;
    eos_id_ = -1;
    if (auto p = meta_as<int32_t>(m, "tokenizer.ggml.bos_token_id")) bos_id_ = *p;
    if (auto p = meta_as<int32_t>(m, "tokenizer.ggml.eos_token_id")) eos_id_ = *p;
    if (auto p = meta_as<uint32_t>(m, "tokenizer.ggml.bos_token_id")) bos_id_ = (int)*p;
    if (auto p = meta_as<uint32_t>(m, "tokenizer.ggml.eos_token_id")) eos_id_ = (int)*p;

    // Find <|im_start|> / <|im_end|> by lookup (Qwen chat tokens).
    auto find = [&](const char* s) {
        auto it = token_to_id_.find(s);
        return it == token_to_id_.end() ? -1 : it->second;
    };
    im_start_id_ = find("<|im_start|>");
    im_end_id_   = find("<|im_end|>");
    turn_start_id_ = find("<|turn>");
    turn_end_id_   = find("<turn|>");

    return true;
}

void Tokenizer::bpe_piece(const std::string& canon, std::vector<int>& out) const {
    // Start from individual canonical codepoints and apply the lowest-rank
    // merge until no adjacent pair has a rule.
    std::vector<std::string> symbols;
    split_codepoints(canon, symbols);

    while (symbols.size() > 1) {
        int best_rank = -1;
        size_t best_i = 0;
        std::string key;
        for (size_t i = 0; i + 1 < symbols.size(); i++) {
            key = symbols[i];
            key.push_back(' ');
            key += symbols[i + 1];
            auto it = merge_ranks_.find(key);
            if (it != merge_ranks_.end() && (best_rank < 0 || it->second < best_rank)) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_rank < 0) break;
        symbols[best_i] += symbols[best_i + 1];
        symbols.erase(symbols.begin() + best_i + 1);
    }

    for (const auto& s : symbols) {
        auto it = token_to_id_.find(s);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
            continue;
        }
        // Merged symbol missing from the vocab (malformed model): fall back
        // to its individual codepoints.
        std::vector<std::string> cps;
        split_codepoints(s, cps);
        for (const auto& cp : cps) {
            auto cit = token_to_id_.find(cp);
            if (cit != token_to_id_.end()) out.push_back(cit->second);
        }
    }
}

void Tokenizer::greedy_piece(const std::string& canon, std::vector<int>& out) const {
    // No merge table: greedy longest-match (approximation).
    const int n = static_cast<int>(canon.size());
    int i = 0;
    while (i < n) {
        int best_len = 0;
        int best_id  = -1;
        for (int len = std::min(n - i, 64); len > 0; len--) {
            auto it = token_to_id_.find(canon.substr(i, len));
            if (it != token_to_id_.end()) {
                best_len = len;
                best_id  = it->second;
                break;
            }
        }
        if (best_id < 0) { i++; continue; }  // skip unencodable byte
        out.push_back(best_id);
        i += best_len;
    }
}

void Tokenizer::spm_encode(const std::string& text, std::vector<int>& out) const {
    // SentencePiece: replace spaces with U+2581, prepend U+2581 to first word.
    std::string buf;
    buf.reserve(text.size() + 3);
    // U+2581 in UTF-8: 0xE2 0x96 0x81
    const char sp[3] = {(char)0xE2, (char)0x96, (char)0x81};
    // SentencePiece always starts with a word boundary marker.
    buf.append(sp, 3);
    for (size_t i = 0; i < text.size(); i++) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == ' ') {
            buf.append(sp, 3);
        } else {
            buf.push_back((char)c);
        }
    }

    // Greedy longest-match on the U+2581-substituted text.
    const int n = static_cast<int>(buf.size());
    int i = 0;
    while (i < n) {
        int best_len = 0;
        int best_id  = -1;
        for (int len = std::min(n - i, 80); len > 0; len--) {
            auto it = token_to_id_.find(buf.substr(i, len));
            if (it != token_to_id_.end()) {
                best_len = len;
                best_id  = it->second;
                break;
            }
        }
        if (best_id < 0) {
            // Byte fallback: use <0xNN> token for this byte.
            char hex[8];
            snprintf(hex, sizeof(hex), "<0x%02X>",
                     static_cast<unsigned char>(buf[i]));
            auto it = token_to_id_.find(hex);
            if (it != token_to_id_.end()) {
                out.push_back(it->second);
            }
            i++;
            continue;
        }
        out.push_back(best_id);
        i += best_len;
    }
}

void Tokenizer::encode_piece(const std::string& canon, std::vector<int>& out) const {
    if (!merge_ranks_.empty()) bpe_piece(canon, out);
    else                       greedy_piece(canon, out);
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    if (is_spm_) {
        std::vector<int> out;
        out.reserve(text.size() / 3 + 4);
        spm_encode(text, out);
        return out;
    }
    std::vector<std::string> pieces;
    pre_tokenize(text, pieces);
    std::vector<int> out;
    out.reserve(text.size() / 3 + 4);
    std::string canon;
    for (const auto& p : pieces) {
        canon.clear();
        for (unsigned char b : p) canon += byte_to_uni_[b];
        encode_piece(canon, out);
    }
    return out;
}

std::vector<int> Tokenizer::encode_chat(const std::string& user_text) const {
    std::vector<int> out;

    // Gemma4 <|turn> format.
    if (turn_start_id_ >= 0 && turn_end_id_ >= 0) {
        if (bos_id_ >= 0) out.push_back(bos_id_);
        // System turn
        out.push_back(turn_start_id_);
        for (int t : encode("system\n")) out.push_back(t);
        out.push_back(turn_end_id_);
        for (int t : encode("\n")) out.push_back(t);
        // User turn
        out.push_back(turn_start_id_);
        for (int t : encode("user\n" + user_text)) out.push_back(t);
        out.push_back(turn_end_id_);
        for (int t : encode("\n")) out.push_back(t);
        // Model turn (generation starts here)
        out.push_back(turn_start_id_);
        for (int t : encode("model\n")) out.push_back(t);
        return out;
    }

    // Qwen <|im_start|> format.
    out.push_back(im_start_id_);
    for (int t : encode("user\n" + user_text)) out.push_back(t);
    out.push_back(im_end_id_);
    for (int t : encode("\n")) out.push_back(t);
    out.push_back(im_start_id_);
    for (int t : encode("assistant\n")) out.push_back(t);
    return out;
}

std::string Tokenizer::decode(int id) const {
    if (id < 0 || id >= (int)id_to_token_.size()) return "";
    const std::string& s = id_to_token_[id];
    if (is_spm_) {
        // SentencePiece: replace U+2581 with space, pass through everything else.
        std::string out;
        out.reserve(s.size());
        for (size_t i = 0; i < s.size(); i++) {
            if (i + 2 < s.size() &&
                (unsigned char)s[i] == 0xE2 &&
                (unsigned char)s[i+1] == 0x96 &&
                (unsigned char)s[i+2] == 0x81) {
                out.push_back(' ');
                i += 2;
            } else {
                out.push_back(s[i]);
            }
        }
        return out;
    }
    // BPE: invert the byte-level encoding codepoint by codepoint.
    std::vector<std::string> cps;
    split_codepoints(s, cps);
    std::string out;
    out.reserve(s.size());
    for (const auto& cp : cps) {
        auto it = uni_to_byte_.find(cp);
        if (it != uni_to_byte_.end()) out.push_back(static_cast<char>(it->second));
        else out += cp;
    }
    return out;
}

} // namespace Laplace
