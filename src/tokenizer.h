// tokenizer.h - GPT-2 byte-level BPE tokenizer
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "gguf.h"

namespace Laplace {

class Tokenizer {
public:
    bool init(const GGUFContext& gguf);

    // Encode text to a sequence of token IDs.
    std::vector<int> encode(const std::string& text) const;

    // Build the Qwen chat sequence for a single user turn:
    //   <|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n
    std::vector<int> encode_chat(const std::string& user_text) const;

    // Decode a single token ID to its raw byte string.
    std::string decode(int id) const;

    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }

    // Special token IDs (-1 if not present in the vocab).
    int im_start_id() const { return im_start_id_; }
    int im_end_id()   const { return im_end_id_; }
    int turn_start_id() const { return turn_start_id_; }
    int turn_end_id()   const { return turn_end_id_; }

private:
    // Tokenize one pre-token (already byte->unicode mapped) via BPE merges,
    // or greedy longest-match when no merges are available.
    void encode_piece(const std::string& canon, std::vector<int>& out) const;
    void bpe_piece(const std::string& canon, std::vector<int>& out) const;
    void greedy_piece(const std::string& canon, std::vector<int>& out) const;
    // SentencePiece-style encode: replace spaces with U+2581, greedy match.
    void spm_encode(const std::string& text, std::vector<int>& out) const;

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    // BPE merge ranks, keyed "left right" (canonical symbols never contain a
    // raw space, so ' ' is a safe separator).
    std::unordered_map<std::string, int> merge_ranks_;
    int bos_id_ = -1;
    int eos_id_ = -1;
    int im_start_id_ = -1;
    int im_end_id_   = -1;
    int turn_start_id_ = -1;  // <|turn> (Gemma4)
    int turn_end_id_   = -1;  // <turn|> (Gemma4)
    bool is_spm_ = false;  // SentencePiece tokenizer (gemma/llama)

    // Precomputed byte (0..255) -> canonical GPT-2/Qwen unicode string, and
    // its inverse for decoding.
    std::string byte_to_uni_[256];
    std::unordered_map<std::string, unsigned char> uni_to_byte_;
};

} // namespace Laplace
