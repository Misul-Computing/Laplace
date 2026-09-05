#include "token_program.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <new>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace Laplace {
namespace {

constexpr std::array<uint8_t, 10> kMagic = {
    'L', 'A', 'P', 'T', 'O', 'K', 'D', 'A', 'T', '1',
};
constexpr std::array<uint8_t, 10> kV2Magic = {
    'L', 'A', 'P', 'T', 'O', 'K', 'D', 'A', 'T', '2',
};
constexpr std::array<uint8_t, 10> kV3Magic = {
    'L', 'A', 'P', 'T', 'O', 'K', 'D', 'A', 'T', '3',
};
constexpr uint16_t kMajorVersion = 1;
constexpr uint16_t kMinorVersion = 0;
constexpr uint16_t kV2SectionRequired = static_cast<uint16_t>(TokenProgramV2SectionFlags::Required);
constexpr uint16_t kKnownV2SectionFlags = kV2SectionRequired;
constexpr uint16_t kKnownVocabFlags = static_cast<uint16_t>(VocabFlags::Special);
constexpr uint16_t kKnownAddedFlags = static_cast<uint16_t>(AddedTokenFlags::SingleWord) |
                                       static_cast<uint16_t>(AddedTokenFlags::LeftStrip) |
                                       static_cast<uint16_t>(AddedTokenFlags::RightStrip) |
                                       static_cast<uint16_t>(AddedTokenFlags::Normalized) |
                                       static_cast<uint16_t>(AddedTokenFlags::Special);
constexpr uint8_t kKnownPretokenizerFlags = static_cast<uint8_t>(PretokenizerFlags::AddPrefixSpace) |
                                            static_cast<uint8_t>(PretokenizerFlags::SplitAsciiWhitespace) |
                                            static_cast<uint8_t>(PretokenizerFlags::GroupNewlineRuns);
constexpr uint8_t kKnownPostprocessorFlags = static_cast<uint8_t>(PostprocessorFlags::AddBos) |
                                             static_cast<uint8_t>(PostprocessorFlags::AddEos);
constexpr uint16_t kKnownBpeFlags = static_cast<uint16_t>(BpeFlags::FuseUnknown);

TokenProgramStatus failure(TokenProgramError error, std::string detail,
                           size_t offset = SIZE_MAX, uint32_t index = UINT32_MAX) {
    return {error, offset, index, std::move(detail)};
}

bool valid_utf8(std::string_view value) {
    size_t offset = 0;
    while (offset < value.size()) {
        const uint8_t first = static_cast<uint8_t>(value[offset]);
        size_t length = 0;
        uint32_t codepoint = 0;
        if (first <= 0x7f) {
            length = 1;
            codepoint = first;
        } else if (first >= 0xc2 && first <= 0xdf) {
            length = 2;
            codepoint = first & 0x1f;
        } else if (first >= 0xe0 && first <= 0xef) {
            length = 3;
            codepoint = first & 0x0f;
        } else if (first >= 0xf0 && first <= 0xf4) {
            length = 4;
            codepoint = first & 0x07;
        } else {
            return false;
        }
        if (length > value.size() - offset) return false;
        for (size_t index = 1; index != length; ++index) {
            const uint8_t continuation = static_cast<uint8_t>(value[offset + index]);
            if ((continuation & 0xc0) != 0x80) return false;
            codepoint = (codepoint << 6) | (continuation & 0x3f);
        }
        if ((length == 2 && codepoint < 0x80) ||
            (length == 3 && codepoint < 0x800) ||
            (length == 4 && codepoint < 0x10000) ||
            (codepoint >= 0xd800 && codepoint <= 0xdfff) || codepoint > 0x10ffff) {
            return false;
        }
        offset += length;
    }
    return true;
}

bool is_word_byte(uint8_t value) {
    return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
           (value >= '0' && value <= '9') || value == '_' || value >= 0x80;
}

bool is_ascii_space(uint8_t value) {
    return value == ' ' || value == '\t' || value == '\n' || value == '\r' || value == '\f' || value == '\v';
}

uint64_t pair_key(uint32_t left, uint32_t right) {
    return (static_cast<uint64_t>(left) << 32) | right;
}

void append_u16(std::vector<uint8_t>& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8));
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8) {
        bytes.push_back(static_cast<uint8_t>(value >> shift));
    }
}

class CanonicalDigest {
public:
    CanonicalDigest() { CC_SHA256_Init(&context_); }

    void bytes(std::span<const uint8_t> value) {
        size_t offset = 0;
        while (offset != value.size()) {
            const size_t count = std::min<size_t>(value.size() - offset, UINT32_MAX);
            CC_SHA256_Update(&context_, value.data() + offset, static_cast<CC_LONG>(count));
            offset += count;
        }
    }

    void text(std::string_view value) {
        bytes(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(value.data()), value.size()));
    }

    void u8(uint8_t value) { bytes(std::span<const uint8_t>(&value, 1)); }

    void u16(uint16_t value) {
        const std::array<uint8_t, 2> encoded = {
            static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8),
        };
        bytes(encoded);
    }

    void u32(uint32_t value) {
        const std::array<uint8_t, 4> encoded = {
            static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8),
            static_cast<uint8_t>(value >> 16), static_cast<uint8_t>(value >> 24),
        };
        bytes(encoded);
    }

    Sha256Digest finish() {
        Sha256Digest digest;
        CC_SHA256_Final(digest.bytes.data(), &context_);
        return digest;
    }

private:
    CC_SHA256_CTX context_{};
};

class Reader {
public:
    explicit Reader(std::span<const uint8_t> bytes) : bytes_(bytes) {}

    size_t offset() const noexcept { return offset_; }
    size_t remaining() const noexcept { return bytes_.size() - offset_; }

    bool take(size_t count, std::span<const uint8_t>& result) {
        if (count > remaining()) return false;
        result = bytes_.subspan(offset_, count);
        offset_ += count;
        return true;
    }

    bool u8(uint8_t& value) {
        std::span<const uint8_t> bytes;
        if (!take(1, bytes)) return false;
        value = bytes[0];
        return true;
    }

    bool u16(uint16_t& value) {
        std::span<const uint8_t> bytes;
        if (!take(2, bytes)) return false;
        value = static_cast<uint16_t>(bytes[0]) | (static_cast<uint16_t>(bytes[1]) << 8);
        return true;
    }

    bool u32(uint32_t& value) {
        std::span<const uint8_t> bytes;
        if (!take(4, bytes)) return false;
        value = static_cast<uint32_t>(bytes[0]) |
                (static_cast<uint32_t>(bytes[1]) << 8) |
                (static_cast<uint32_t>(bytes[2]) << 16) |
                (static_cast<uint32_t>(bytes[3]) << 24);
        return true;
    }

private:
    std::span<const uint8_t> bytes_;
    size_t offset_ = 0;
};

bool has_size_for_count(size_t count, size_t bytes_per_record, size_t remaining) {
    return count <= remaining / bytes_per_record;
}

bool add_minimum_bytes(size_t count, size_t bytes_per_record, size_t& total) {
    if (count > (SIZE_MAX - total) / bytes_per_record) return false;
    total += count * bytes_per_record;
    return true;
}

TokenProgramStatus normalize_text(const NormalizerSpec& spec, std::string_view input,
                                  std::string& output);

TokenProgramStatus validate_definition(const TokenProgramDefinition& definition) {
    if (definition.vocabulary.empty() || definition.vocabulary.size() > token_program_limits::kMaxVocabulary) {
        return failure(TokenProgramError::VocabularyNotDense, "vocabulary must be non-empty and bounded");
    }
    if ((definition.bpe_flags & ~kKnownBpeFlags) != 0) {
        return failure(TokenProgramError::InvalidParameter, "BPE flags are unsupported");
    }

    std::array<bool, 256> seen_map{};
    for (size_t index = 0; index != definition.byte_map.size(); ++index) {
        const uint8_t mapped = definition.byte_map[index];
        if (seen_map[mapped]) {
            return failure(TokenProgramError::InvalidParameter, "byte map must be a permutation", index);
        }
        seen_map[mapped] = true;
    }

    std::unordered_set<std::string> vocabulary_pieces;
    vocabulary_pieces.reserve(definition.vocabulary.size());
    for (size_t index = 0; index != definition.vocabulary.size(); ++index) {
        const VocabEntry& entry = definition.vocabulary[index];
        if (entry.piece.empty() || entry.piece.size() > token_program_limits::kMaxPieceBytes) {
            return failure(TokenProgramError::VocabularyNotDense, "vocabulary piece is empty or too large", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if ((entry.flags & ~kKnownVocabFlags) != 0) {
            return failure(TokenProgramError::InvalidParameter, "vocabulary flags are unsupported", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if (!vocabulary_pieces.emplace(entry.piece).second) {
            return failure(TokenProgramError::DuplicateRecord, "duplicate vocabulary piece", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if ((entry.flags & static_cast<uint16_t>(VocabFlags::Special)) != 0 && !valid_utf8(entry.piece)) {
            return failure(TokenProgramError::InvalidUtf8, "special vocabulary piece is not UTF-8", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if (definition.decoder.kind == DecoderKind::Identity && !valid_utf8(entry.piece)) {
            return failure(TokenProgramError::InvalidUtf8, "identity decoder requires UTF-8 vocabulary pieces", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
    }

    if (definition.unknown_token_id != kTokenProgramNoTokenId &&
        definition.unknown_token_id >= definition.vocabulary.size()) {
        return failure(TokenProgramError::InvalidTokenId, "unknown token ID is outside the vocabulary");
    }

    switch (definition.normalizer.kind) {
    case NormalizerKind::None:
    case NormalizerKind::AsciiLowercase:
        if (definition.normalizer.flags != 0 || definition.normalizer.parameter != 0 ||
            definition.normalizer.argument != 0) {
            return failure(TokenProgramError::InvalidParameter, "normalizer parameters are not canonical");
        }
        break;
    default:
        return failure(TokenProgramError::UnsupportedEnum, "normalizer kind is unsupported");
    }

    if (definition.pretokenizer.kind != PretokenizerKind::ByteLevel ||
        (definition.pretokenizer.flags & ~kKnownPretokenizerFlags) != 0 ||
        definition.pretokenizer.parameter != 0 || definition.pretokenizer.argument != 0) {
        return definition.pretokenizer.kind != PretokenizerKind::ByteLevel
                   ? failure(TokenProgramError::UnsupportedEnum, "pretokenizer kind is unsupported")
                   : failure(TokenProgramError::InvalidParameter, "pretokenizer parameters are not canonical");
    }

    switch (definition.postprocessor.kind) {
    case PostprocessorKind::None:
        if (definition.postprocessor.flags != 0 || definition.postprocessor.parameter != 0 ||
            definition.postprocessor.bos_token_id != kTokenProgramNoTokenId ||
            definition.postprocessor.eos_token_id != kTokenProgramNoTokenId) {
            return failure(TokenProgramError::InvalidParameter, "empty postprocessor has non-empty parameters");
        }
        break;
    case PostprocessorKind::AddBosEos: {
        if ((definition.postprocessor.flags & ~kKnownPostprocessorFlags) != 0 ||
            definition.postprocessor.parameter != 0 || definition.postprocessor.flags == 0) {
            return failure(TokenProgramError::InvalidParameter, "postprocessor parameters are unsupported");
        }
        if ((definition.postprocessor.flags & static_cast<uint8_t>(PostprocessorFlags::AddBos)) != 0) {
            if (definition.postprocessor.bos_token_id >= definition.vocabulary.size()) {
                return failure(TokenProgramError::InvalidTokenId, "BOS token ID is outside the vocabulary");
            }
        } else if (definition.postprocessor.bos_token_id != kTokenProgramNoTokenId) {
            return failure(TokenProgramError::InvalidParameter, "unused BOS token ID is not canonical");
        }
        if ((definition.postprocessor.flags & static_cast<uint8_t>(PostprocessorFlags::AddEos)) != 0) {
            if (definition.postprocessor.eos_token_id >= definition.vocabulary.size()) {
                return failure(TokenProgramError::InvalidTokenId, "EOS token ID is outside the vocabulary");
            }
        } else if (definition.postprocessor.eos_token_id != kTokenProgramNoTokenId) {
            return failure(TokenProgramError::InvalidParameter, "unused EOS token ID is not canonical");
        }
        break;
    }
    default:
        return failure(TokenProgramError::UnsupportedEnum, "postprocessor kind is unsupported");
    }

    switch (definition.decoder.kind) {
    case DecoderKind::ByteLevel:
        if ((definition.decoder.flags & ~static_cast<uint8_t>(DecoderFlags::SkipSpecial)) != 0 ||
            definition.decoder.parameter != 0 || definition.decoder.argument != 0) {
            return failure(TokenProgramError::InvalidParameter, "byte-level decoder parameters are unsupported");
        }
        break;
    case DecoderKind::Identity:
        if ((definition.decoder.flags & ~static_cast<uint8_t>(DecoderFlags::SkipSpecial)) != 0 ||
            definition.decoder.parameter != 0 || definition.decoder.argument != 0) {
            return failure(TokenProgramError::InvalidParameter, "identity decoder parameters are unsupported");
        }
        break;
    default:
        return failure(TokenProgramError::UnsupportedEnum, "decoder kind is unsupported");
    }

    if (definition.merges.size() > token_program_limits::kMaxMerges) {
        return failure(TokenProgramError::InvalidMerge, "merge table is too large");
    }
    std::unordered_set<uint64_t> merge_pairs;
    merge_pairs.reserve(definition.merges.size());
    for (size_t index = 0; index != definition.merges.size(); ++index) {
        const MergeRecord& merge = definition.merges[index];
        if (merge.rank != index || merge.left_id >= definition.vocabulary.size() ||
            merge.right_id >= definition.vocabulary.size() || merge.result_id >= definition.vocabulary.size()) {
            return failure(TokenProgramError::InvalidMerge, "merge rank or token ID is invalid", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        const VocabEntry& left = definition.vocabulary[merge.left_id];
        const VocabEntry& right = definition.vocabulary[merge.right_id];
        const VocabEntry& result = definition.vocabulary[merge.result_id];
        const uint16_t special = static_cast<uint16_t>(VocabFlags::Special);
        if ((left.flags & special) != 0 || (right.flags & special) != 0 || (result.flags & special) != 0 ||
            left.piece.size() > token_program_limits::kMaxPieceBytes - right.piece.size() ||
            result.piece != left.piece + right.piece) {
            return failure(TokenProgramError::InvalidMerge, "merge pieces do not concatenate exactly", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if (!merge_pairs.emplace(pair_key(merge.left_id, merge.right_id)).second) {
            return failure(TokenProgramError::DuplicateRecord, "duplicate merge pair", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
    }

    if (definition.added_tokens.size() > token_program_limits::kMaxAddedTokens) {
        return failure(TokenProgramError::InvalidAddedToken, "added-token table is too large");
    }
    std::unordered_set<uint32_t> added_ids;
    std::unordered_set<std::string> added_matches;
    added_ids.reserve(definition.added_tokens.size());
    added_matches.reserve(definition.added_tokens.size());
    for (size_t index = 0; index != definition.added_tokens.size(); ++index) {
        const AddedTokenRecord& added = definition.added_tokens[index];
        if (added.token_id >= definition.vocabulary.size() || added.match.empty() ||
            added.match.size() > token_program_limits::kMaxPieceBytes || added.normalized.empty() ||
            added.normalized.size() > token_program_limits::kMaxPieceBytes) {
            return failure(TokenProgramError::InvalidAddedToken, "added-token record is malformed", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if ((added.flags & ~kKnownAddedFlags) != 0 || !valid_utf8(added.match) || !valid_utf8(added.normalized)) {
            return !valid_utf8(added.match) || !valid_utf8(added.normalized)
                       ? failure(TokenProgramError::InvalidUtf8, "added-token text is not UTF-8", SIZE_MAX,
                                 static_cast<uint32_t>(index))
                       : failure(TokenProgramError::InvalidAddedToken, "added-token flags are unsupported", SIZE_MAX,
                                 static_cast<uint32_t>(index));
        }
        if ((added.flags & static_cast<uint16_t>(AddedTokenFlags::Normalized)) == 0 &&
            added.match != added.normalized) {
            return failure(TokenProgramError::InvalidAddedToken,
                           "non-normalized added token must have equal match forms", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if ((added.flags & static_cast<uint16_t>(AddedTokenFlags::Normalized)) != 0) {
            std::string expected;
            const TokenProgramStatus status = normalize_text(definition.normalizer, added.match, expected);
            if (!status.ok() || expected != added.normalized) {
                return failure(TokenProgramError::InvalidAddedToken,
                               "normalized added token does not match the declared normalizer", SIZE_MAX,
                               static_cast<uint32_t>(index));
            }
        }
        const bool added_special =
            (added.flags & static_cast<uint16_t>(AddedTokenFlags::Special)) != 0;
        const bool vocabulary_special =
            (definition.vocabulary[added.token_id].flags & static_cast<uint16_t>(VocabFlags::Special)) != 0;
        if (added_special != vocabulary_special) {
            return failure(TokenProgramError::InvalidAddedToken,
                           "added-token and vocabulary special flags disagree", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if (!added_ids.emplace(added.token_id).second || !added_matches.emplace(added.match).second) {
            return failure(TokenProgramError::DuplicateRecord, "duplicate added-token record", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
    }

    if (definition.prompt.empty() || definition.prompt.size() > token_program_limits::kMaxPromptInstructions ||
        definition.prompt_max_bytes == 0 || definition.prompt_max_bytes > token_program_limits::kMaxPromptBytes) {
        return failure(TokenProgramError::InvalidPrompt, "prompt program is empty or outside its bounds");
    }
    size_t literal_bytes = 0;
    size_t user_count = 0;
    size_t generation_count = 0;
    for (size_t index = 0; index != definition.prompt.size(); ++index) {
        const PromptInstruction& instruction = definition.prompt[index];
        switch (instruction.opcode) {
        case PromptOpcode::EmitLiteralUtf8:
            if (instruction.literal.empty() || !valid_utf8(instruction.literal)) {
                return failure(TokenProgramError::InvalidUtf8, "prompt literal is empty or invalid UTF-8", SIZE_MAX,
                               static_cast<uint32_t>(index));
            }
            break;
        case PromptOpcode::EmitUserText:
            if (!instruction.literal.empty()) {
                return failure(TokenProgramError::InvalidPrompt, "user-text instruction has a literal", SIZE_MAX,
                               static_cast<uint32_t>(index));
            }
            ++user_count;
            break;
        case PromptOpcode::EmitGenerationPrompt:
            if (instruction.literal.empty() || !valid_utf8(instruction.literal)) {
                return failure(TokenProgramError::InvalidPrompt,
                               "generation instruction needs a valid UTF-8 marker", SIZE_MAX,
                               static_cast<uint32_t>(index));
            }
            ++generation_count;
            break;
        case PromptOpcode::End:
            if (index + 1 != definition.prompt.size() || !instruction.literal.empty()) {
                return failure(TokenProgramError::InvalidPrompt, "prompt end must be the final empty instruction", SIZE_MAX,
                               static_cast<uint32_t>(index));
            }
            break;
        default:
            return failure(TokenProgramError::UnsupportedEnum, "prompt opcode is unsupported", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if (instruction.literal.size() > token_program_limits::kMaxPromptLiteralBytes ||
            literal_bytes > token_program_limits::kMaxPromptLiteralBytes - instruction.literal.size()) {
            return failure(TokenProgramError::InvalidPrompt, "prompt literal pool is too large", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        literal_bytes += instruction.literal.size();
    }
    if (definition.prompt.back().opcode != PromptOpcode::End || user_count != 1 || generation_count != 1) {
        return failure(TokenProgramError::InvalidPrompt,
                       "prompt needs exactly one user-text and one generation instruction");
    }
    return {};
}

void append_spec(std::vector<uint8_t>& bytes, NormalizerSpec spec) {
    bytes.push_back(static_cast<uint8_t>(spec.kind));
    bytes.push_back(spec.flags);
    append_u16(bytes, spec.parameter);
    append_u32(bytes, spec.argument);
}

void append_spec(std::vector<uint8_t>& bytes, PretokenizerSpec spec) {
    bytes.push_back(static_cast<uint8_t>(spec.kind));
    bytes.push_back(spec.flags);
    append_u16(bytes, spec.parameter);
    append_u32(bytes, spec.argument);
}

void append_spec(std::vector<uint8_t>& bytes, PostprocessorSpec spec) {
    bytes.push_back(static_cast<uint8_t>(spec.kind));
    bytes.push_back(spec.flags);
    append_u16(bytes, spec.parameter);
    append_u32(bytes, spec.bos_token_id);
    append_u32(bytes, spec.eos_token_id);
}

void append_spec(std::vector<uint8_t>& bytes, DecoderSpec spec) {
    bytes.push_back(static_cast<uint8_t>(spec.kind));
    bytes.push_back(spec.flags);
    append_u16(bytes, spec.parameter);
    append_u32(bytes, spec.argument);
}

bool read_normalizer(Reader& reader, NormalizerSpec& spec) {
    uint8_t kind = 0;
    return reader.u8(kind) && (spec.kind = static_cast<NormalizerKind>(kind), reader.u8(spec.flags)) &&
           reader.u16(spec.parameter) && reader.u32(spec.argument);
}

bool read_pretokenizer(Reader& reader, PretokenizerSpec& spec) {
    uint8_t kind = 0;
    return reader.u8(kind) && (spec.kind = static_cast<PretokenizerKind>(kind), reader.u8(spec.flags)) &&
           reader.u16(spec.parameter) && reader.u32(spec.argument);
}

bool read_postprocessor(Reader& reader, PostprocessorSpec& spec) {
    uint8_t kind = 0;
    return reader.u8(kind) && (spec.kind = static_cast<PostprocessorKind>(kind), reader.u8(spec.flags)) &&
           reader.u16(spec.parameter) && reader.u32(spec.bos_token_id) && reader.u32(spec.eos_token_id);
}

bool read_decoder(Reader& reader, DecoderSpec& spec) {
    uint8_t kind = 0;
    return reader.u8(kind) && (spec.kind = static_cast<DecoderKind>(kind), reader.u8(spec.flags)) &&
           reader.u16(spec.parameter) && reader.u32(spec.argument);
}

TokenProgramStatus read_failure(Reader& reader, const char* detail) {
    return failure(TokenProgramError::PayloadMalformed, detail, reader.offset());
}

} // namespace

namespace {

constexpr uint16_t kV3HeaderFlags = 0;
constexpr uint16_t kV3Reserved = 0;
constexpr uint16_t kV3SectionRequired = static_cast<uint16_t>(TokenProgramV3SectionFlags::Required);
constexpr uint16_t kV3KnownSectionFlags = kV3SectionRequired;
constexpr uint16_t kV3KnownVocabFlags = static_cast<uint16_t>(VocabFlags::Special);
constexpr uint8_t kV3KnownNormalizerFlags =
    static_cast<uint8_t>(SentencePieceNormalizerFlags::AddDummyPrefix) |
    static_cast<uint8_t>(SentencePieceNormalizerFlags::RemoveExtraWhitespaces) |
    static_cast<uint8_t>(SentencePieceNormalizerFlags::EscapeWhitespaces) |
    static_cast<uint8_t>(SentencePieceNormalizerFlags::TreatWhitespaceAsSuffix);
constexpr uint32_t kV3MaxSections = 16;
constexpr uint64_t kV3MaxExecutionWork = 64u * 1000u * 1000u;

// These scalar helpers are defined with the V2 implementation below.  Keep
// their declarations here so the V3 executor can share the same bounded
// UTF-8 machinery without duplicating it.
struct ScalarSpan {
    size_t begin = 0;
    size_t end = 0;
    uint32_t value = 0;
};

bool decode_scalar_at(std::string_view text, size_t offset, uint32_t& value, size_t& length);
bool valid_scalar(uint32_t value);
bool collect_scalars(std::string_view text, std::vector<ScalarSpan>& result);
bool scalar_is_whitespace(uint32_t value);
std::string scalar_utf8(uint32_t value);

std::vector<uint8_t> make_v2_added(const std::vector<AddedTokenRecord>& added_tokens);
std::vector<uint8_t> make_v2_postprocessor(const PostprocessorSpec& spec);
std::vector<uint8_t> make_v2_decoder(const DecoderSpec& spec);
std::vector<uint8_t> make_v2_prompt(const std::vector<PromptInstruction>& prompt);

Sha256Digest v3_vocabulary_digest(const TokenProgramDefinition& definition);
Sha256Digest v3_prompt_digest(const TokenProgramDefinition& definition);

uint32_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bits_float(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void append_digest_v3(std::vector<uint8_t>& bytes, const Sha256Digest& digest) {
    bytes.insert(bytes.end(), digest.bytes.begin(), digest.bytes.end());
}

bool read_digest_v3(Reader& reader, Sha256Digest& digest) {
    std::span<const uint8_t> bytes;
    if (!reader.take(digest.bytes.size(), bytes)) return false;
    std::copy(bytes.begin(), bytes.end(), digest.bytes.begin());
    return true;
}

std::string v3_byte_piece(uint8_t value) {
    constexpr char hex[] = "0123456789abcdef";
    std::string piece = "<0x00>";
    piece[3] = hex[value >> 4];
    piece[4] = hex[value & 0x0f];
    return piece;
}

bool v3_valid_piece_type(uint8_t type) {
    switch (static_cast<TokenPieceType>(type)) {
    case TokenPieceType::Normal:
    case TokenPieceType::Unknown:
    case TokenPieceType::Control:
    case TokenPieceType::UserDefined:
    case TokenPieceType::Byte:
    case TokenPieceType::Unused:
        return true;
    }
    return false;
}

bool v3_is_regular(const VocabEntry& entry) {
    const auto type = static_cast<TokenPieceType>(entry.type);
    return type == TokenPieceType::Normal || type == TokenPieceType::UserDefined;
}

bool v3_is_special(const VocabEntry& entry) {
    const auto type = static_cast<TokenPieceType>(entry.type);
    return type == TokenPieceType::Control || type == TokenPieceType::UserDefined;
}

bool v3_is_byte_piece(std::string_view piece) {
    if (piece.size() != 6 || piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>') {
        return false;
    }
    const auto hex = [](char value) {
        return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f') ||
               (value >= 'A' && value <= 'F');
    };
    return hex(piece[3]) && hex(piece[4]);
}

TokenProgramStatus validate_v3_postprocessor(const TokenProgramDefinition& definition) {
    const PostprocessorSpec& spec = definition.postprocessor;
    if (spec.kind == PostprocessorKind::None) {
        if (spec.flags != 0 || spec.parameter != 0 || spec.bos_token_id != kTokenProgramNoTokenId ||
            spec.eos_token_id != kTokenProgramNoTokenId) {
            return failure(TokenProgramError::InvalidParameter, "V3 empty postprocessor has non-empty parameters");
        }
        return {};
    }
    const uint8_t known_flags = static_cast<uint8_t>(PostprocessorFlags::AddBos) |
                                static_cast<uint8_t>(PostprocessorFlags::AddEos);
    if (spec.kind != PostprocessorKind::AddBosEos || (spec.flags & ~known_flags) != 0 ||
        spec.flags == 0 || spec.parameter != 0) {
        return failure(TokenProgramError::UnsupportedEnum, "V3 postprocessor is unsupported");
    }
    if ((spec.flags & static_cast<uint8_t>(PostprocessorFlags::AddBos)) != 0) {
        if (spec.bos_token_id >= definition.vocabulary.size()) {
            return failure(TokenProgramError::InvalidTokenId, "V3 BOS token ID is outside the vocabulary");
        }
    } else if (spec.bos_token_id != kTokenProgramNoTokenId) {
        return failure(TokenProgramError::InvalidParameter, "V3 unused BOS token ID is not canonical");
    }
    if ((spec.flags & static_cast<uint8_t>(PostprocessorFlags::AddEos)) != 0) {
        if (spec.eos_token_id >= definition.vocabulary.size()) {
            return failure(TokenProgramError::InvalidTokenId, "V3 EOS token ID is outside the vocabulary");
        }
    } else if (spec.eos_token_id != kTokenProgramNoTokenId) {
        return failure(TokenProgramError::InvalidParameter, "V3 unused EOS token ID is not canonical");
    }
    return {};
}

TokenProgramStatus validate_v3_prompt(const TokenProgramDefinition& definition) {
    if (definition.prompt.empty() || definition.prompt.size() > token_program_limits::kMaxPromptInstructions ||
        definition.prompt_max_bytes == 0 || definition.prompt_max_bytes > token_program_limits::kMaxPromptBytes) {
        return failure(TokenProgramError::InvalidPrompt, "V3 prompt program is empty or outside its bounds");
    }
    size_t literal_bytes = 0;
    size_t user_count = 0;
    size_t generation_count = 0;
    for (size_t index = 0; index != definition.prompt.size(); ++index) {
        const PromptInstruction& instruction = definition.prompt[index];
        switch (instruction.opcode) {
        case PromptOpcode::EmitLiteralUtf8:
            if (instruction.literal.empty() || !valid_utf8(instruction.literal)) {
                return failure(TokenProgramError::InvalidUtf8, "V3 prompt literal is invalid UTF-8", SIZE_MAX,
                               static_cast<uint32_t>(index));
            }
            break;
        case PromptOpcode::EmitUserText:
            if (!instruction.literal.empty()) {
                return failure(TokenProgramError::InvalidPrompt, "V3 user-text instruction has a literal", SIZE_MAX,
                               static_cast<uint32_t>(index));
            }
            ++user_count;
            break;
        case PromptOpcode::EmitGenerationPrompt:
            if (instruction.literal.empty() || !valid_utf8(instruction.literal)) {
                return failure(TokenProgramError::InvalidPrompt, "V3 generation instruction is invalid", SIZE_MAX,
                               static_cast<uint32_t>(index));
            }
            ++generation_count;
            break;
        case PromptOpcode::End:
            if (index + 1 != definition.prompt.size() || !instruction.literal.empty()) {
                return failure(TokenProgramError::InvalidPrompt, "V3 prompt end is not canonical", SIZE_MAX,
                               static_cast<uint32_t>(index));
            }
            break;
        default:
            return failure(TokenProgramError::UnsupportedEnum, "V3 prompt opcode is unsupported", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if (instruction.literal.size() > token_program_limits::kMaxPromptLiteralBytes ||
            literal_bytes > token_program_limits::kMaxPromptLiteralBytes - instruction.literal.size()) {
            return failure(TokenProgramError::InvalidPrompt, "V3 prompt literal pool is too large", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        literal_bytes += instruction.literal.size();
    }
    if (definition.prompt.back().opcode != PromptOpcode::End || user_count != 1 || generation_count != 1) {
        return failure(TokenProgramError::InvalidPrompt, "V3 prompt needs one input and one generation instruction");
    }
    if (!definition.turn.empty()) {
        if (definition.turn.size() > token_program_limits::kMaxPromptInstructions) {
            return failure(TokenProgramError::InvalidPrompt,
                           "V3 turn program is outside its bounds");
        }
        size_t turn_literal_bytes = 0;
        size_t turn_user_count = 0;
        size_t turn_generation_count = 0;
        for (size_t index = 0; index != definition.turn.size(); ++index) {
            const PromptInstruction& instruction = definition.turn[index];
            switch (instruction.opcode) {
            case PromptOpcode::EmitLiteralUtf8:
                if (instruction.literal.empty() || !valid_utf8(instruction.literal)) {
                    return failure(TokenProgramError::InvalidUtf8,
                                   "V3 turn literal is invalid UTF-8", SIZE_MAX,
                                   static_cast<uint32_t>(index));
                }
                break;
            case PromptOpcode::EmitUserText:
                if (!instruction.literal.empty()) {
                    return failure(TokenProgramError::InvalidPrompt,
                                   "V3 turn user-text instruction has a literal", SIZE_MAX,
                                   static_cast<uint32_t>(index));
                }
                ++turn_user_count;
                break;
            case PromptOpcode::EmitGenerationPrompt:
                if (instruction.literal.empty() || !valid_utf8(instruction.literal)) {
                    return failure(TokenProgramError::InvalidPrompt,
                                   "V3 turn generation instruction is invalid", SIZE_MAX,
                                   static_cast<uint32_t>(index));
                }
                ++turn_generation_count;
                break;
            case PromptOpcode::End:
                if (index + 1 != definition.turn.size() || !instruction.literal.empty()) {
                    return failure(TokenProgramError::InvalidPrompt,
                                   "V3 turn end is not canonical", SIZE_MAX,
                                   static_cast<uint32_t>(index));
                }
                break;
            default:
                return failure(TokenProgramError::UnsupportedEnum,
                               "V3 turn opcode is unsupported", SIZE_MAX,
                               static_cast<uint32_t>(index));
            }
            if (instruction.literal.size() > token_program_limits::kMaxPromptLiteralBytes ||
                turn_literal_bytes >
                    token_program_limits::kMaxPromptLiteralBytes - instruction.literal.size()) {
                return failure(TokenProgramError::InvalidPrompt,
                               "V3 turn literal pool is too large", SIZE_MAX,
                               static_cast<uint32_t>(index));
            }
            turn_literal_bytes += instruction.literal.size();
        }
        if (definition.turn.back().opcode != PromptOpcode::End || turn_user_count != 1 ||
            turn_generation_count != 1) {
            return failure(TokenProgramError::InvalidPrompt,
                           "V3 turn program needs one input and one generation instruction");
        }
    }
    return {};
}

TokenProgramStatus validate_v3_definition(const TokenProgramDefinition& definition) {
    if (definition.vocabulary.empty() || definition.vocabulary.size() > token_program_limits::kMaxVocabulary) {
        return failure(TokenProgramError::VocabularyNotDense, "V3 vocabulary is empty or too large");
    }
    if (definition.stream_max_bytes == 0 || definition.stream_max_bytes > token_program_limits::kMaxInputBytes) {
        return failure(TokenProgramError::InvalidParameter, "V3 stream bound is outside its limit");
    }
    if ((definition.bpe_flags & ~kKnownBpeFlags) != 0) {
        return failure(TokenProgramError::InvalidParameter, "V3 BPE flags are unsupported");
    }
    switch (definition.model_kind) {
    case TokenProgramModelKind::ByteBpe:
        if (definition.normalizer.kind != NormalizerKind::None &&
            definition.normalizer.kind != NormalizerKind::AsciiLowercase) {
            return failure(TokenProgramError::UnsupportedEnum, "V3 byte-BPE normalizer is unsupported");
        }
        if (definition.normalizer.flags != 0 || definition.normalizer.parameter != 0 ||
            definition.normalizer.argument != 0 || !definition.normalizer_data.empty()) {
            return failure(TokenProgramError::InvalidParameter, "V3 byte-BPE normalizer parameters are not canonical");
        }
        if (definition.pretokenizer.kind != PretokenizerKind::ByteLevel &&
            definition.pretokenizer.kind != PretokenizerKind::Regex) {
            return failure(TokenProgramError::UnsupportedEnum, "V3 byte-BPE pretokenizer is unsupported");
        }
        break;
    case TokenProgramModelKind::SentencePiece:
        if (definition.normalizer.kind != NormalizerKind::SentencePiece) {
            return failure(TokenProgramError::UnsupportedEnum, "V3 SentencePiece normalizer is required");
        }
        if ((definition.normalizer.flags & ~kV3KnownNormalizerFlags) != 0 ||
            definition.normalizer.parameter != 0 || definition.normalizer.argument != 0 ||
            definition.normalizer_data.size() > token_program_limits::kMaxNormalizerDataBytes) {
            return failure(TokenProgramError::InvalidParameter, "V3 SentencePiece normalizer data is unsupported");
        }
        if (definition.pretokenizer.kind != PretokenizerKind::SentencePiece ||
            definition.pretokenizer.flags != 0 || definition.pretokenizer.parameter != 0 ||
            definition.pretokenizer.argument != 0 || !definition.pretokenizer_regex.empty()) {
            return failure(TokenProgramError::InvalidParameter, "V3 SentencePiece pretokenizer is not canonical");
        }
        break;
    default:
        return failure(TokenProgramError::UnsupportedEnum, "V3 tokenizer model kind is unsupported");
    }
    if (definition.pretokenizer_regex.size() > token_program_limits::kMaxPretokenizerBytes ||
        !valid_utf8(definition.pretokenizer_regex)) {
        return failure(TokenProgramError::InvalidUtf8, "V3 pretokenizer regex is not UTF-8");
    }

    if (definition.model_kind == TokenProgramModelKind::ByteBpe) {
        for (size_t index = 0; index != definition.byte_to_unicode.size(); ++index) {
            const uint32_t value = definition.byte_to_unicode[index];
            if (!valid_scalar(value)) {
                return failure(TokenProgramError::InvalidParameter, "V3 byte map contains a duplicate or invalid scalar",
                               SIZE_MAX, static_cast<uint32_t>(index));
            }
            for (size_t prior = 0; prior != index; ++prior) {
                if (definition.byte_to_unicode[prior] == value) {
                    return failure(TokenProgramError::InvalidParameter, "V3 byte map is not a permutation", SIZE_MAX,
                                   static_cast<uint32_t>(index));
                }
            }
        }
    }

    std::unordered_set<std::string> pieces;
    pieces.reserve(definition.vocabulary.size());
    size_t unknown_count = 0;
    for (size_t index = 0; index != definition.vocabulary.size(); ++index) {
        const VocabEntry& entry = definition.vocabulary[index];
        if (entry.piece.empty() || entry.piece.size() > token_program_limits::kMaxPieceBytes ||
            !valid_utf8(entry.piece) || !v3_valid_piece_type(entry.type) ||
            (entry.flags & ~kV3KnownVocabFlags) != 0 || !std::isfinite(entry.score)) {
            return !valid_utf8(entry.piece)
                       ? failure(TokenProgramError::InvalidUtf8, "V3 vocabulary piece is not UTF-8", SIZE_MAX,
                                 static_cast<uint32_t>(index))
                       : failure(TokenProgramError::VocabularyNotDense, "V3 vocabulary record is malformed", SIZE_MAX,
                                 static_cast<uint32_t>(index));
        }
        if (!pieces.emplace(entry.piece).second) {
            return failure(TokenProgramError::DuplicateRecord, "V3 vocabulary contains duplicate pieces", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if (static_cast<TokenPieceType>(entry.type) == TokenPieceType::Byte && !v3_is_byte_piece(entry.piece)) {
            return failure(TokenProgramError::UnknownPiece, "V3 byte vocabulary piece is malformed", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if (static_cast<TokenPieceType>(entry.type) == TokenPieceType::Unknown) ++unknown_count;
    }
    if (unknown_count > 1) return failure(TokenProgramError::DuplicateRecord, "V3 vocabulary has multiple unknown pieces");
    if (definition.unknown_token_id != kTokenProgramNoTokenId) {
        if (definition.unknown_token_id >= definition.vocabulary.size() ||
            static_cast<TokenPieceType>(definition.vocabulary[definition.unknown_token_id].type) != TokenPieceType::Unknown) {
            return failure(TokenProgramError::InvalidTokenId, "V3 unknown token ID is not an unknown piece");
        }
    }

    if (definition.merges.size() > token_program_limits::kMaxMerges) {
        return failure(TokenProgramError::InvalidMerge, "V3 merge table is too large");
    }
    std::unordered_set<uint64_t> merge_pairs;
    std::unordered_set<uint32_t> merge_ranks;
    merge_pairs.reserve(definition.merges.size());
    merge_ranks.reserve(definition.merges.size());
    for (size_t index = 0; index != definition.merges.size(); ++index) {
        const MergeRecord& merge = definition.merges[index];
        if (merge.left_id >= definition.vocabulary.size() ||
            merge.right_id >= definition.vocabulary.size() || merge.result_id >= definition.vocabulary.size() ||
            !merge_ranks.emplace(merge.rank).second ||
            !v3_is_regular(definition.vocabulary[merge.left_id]) ||
            !v3_is_regular(definition.vocabulary[merge.right_id]) ||
            !v3_is_regular(definition.vocabulary[merge.result_id])) {
            return failure(TokenProgramError::InvalidMerge, "V3 merge rank or token ID is invalid", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        const std::string expected = definition.vocabulary[merge.left_id].piece +
                                     definition.vocabulary[merge.right_id].piece;
        if (expected != definition.vocabulary[merge.result_id].piece ||
            !merge_pairs.emplace(pair_key(merge.left_id, merge.right_id)).second) {
            return failure(TokenProgramError::InvalidMerge, "V3 merge pieces do not concatenate exactly", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
    }

    if (definition.added_tokens.size() > token_program_limits::kMaxAddedTokens) {
        return failure(TokenProgramError::InvalidAddedToken, "V3 added-token table is too large");
    }
    std::unordered_set<uint32_t> added_ids;
    for (size_t index = 0; index != definition.added_tokens.size(); ++index) {
        const AddedTokenRecord& added = definition.added_tokens[index];
        if (added.token_id >= definition.vocabulary.size() || added.match.empty() ||
            added.match.size() > token_program_limits::kMaxPieceBytes || added.normalized.empty() ||
            added.normalized.size() > token_program_limits::kMaxPieceBytes || !valid_utf8(added.match) ||
            !valid_utf8(added.normalized) || (added.flags & ~kKnownAddedFlags) != 0 ||
            !added_ids.emplace(added.token_id).second) {
            return !valid_utf8(added.match) || !valid_utf8(added.normalized)
                       ? failure(TokenProgramError::InvalidUtf8, "V3 added-token text is not UTF-8", SIZE_MAX,
                                 static_cast<uint32_t>(index))
                       : failure(TokenProgramError::InvalidAddedToken, "V3 added-token record is malformed", SIZE_MAX,
                                 static_cast<uint32_t>(index));
        }
    }
    const TokenProgramStatus postprocessor = validate_v3_postprocessor(definition);
    if (!postprocessor.ok()) return postprocessor;
    if (definition.stop_ids.size() > token_program_limits::kMaxStopIds) {
        return failure(TokenProgramError::InvalidParameter, "V3 stop list is too large");
    }
    for (size_t index = 0; index != definition.stop_ids.size(); ++index) {
        if (definition.stop_ids[index] >= definition.vocabulary.size() ||
            (index != 0 && definition.stop_ids[index - 1] >= definition.stop_ids[index])) {
            return failure(TokenProgramError::InvalidParameter, "V3 stop IDs must be strictly increasing", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
    }
    const TokenProgramStatus prompt = validate_v3_prompt(definition);
    if (!prompt.ok()) return prompt;
    const Sha256Digest vocabulary_digest = v3_vocabulary_digest(definition);
    if (!std::all_of(definition.vocabulary_digest.bytes.begin(), definition.vocabulary_digest.bytes.end(),
                     [](uint8_t byte) { return byte == 0; }) &&
        definition.vocabulary_digest != vocabulary_digest) {
        return failure(TokenProgramError::DigestMismatch, "V3 vocabulary digest does not match its records");
    }
    const Sha256Digest prompt_digest = v3_prompt_digest(definition);
    if (!std::all_of(definition.prompt_program_digest.bytes.begin(), definition.prompt_program_digest.bytes.end(),
                     [](uint8_t byte) { return byte == 0; }) &&
        definition.prompt_program_digest != prompt_digest) {
        return failure(TokenProgramError::DigestMismatch, "V3 prompt digest does not match its program");
    }
    return {};
}

Sha256Digest v3_vocabulary_digest(const TokenProgramDefinition& definition) {
    CanonicalDigest digest;
    digest.text("LAPTOK-VOCAB-v3");
    digest.u8(0);
    digest.u32(static_cast<uint32_t>(definition.vocabulary.size()));
    for (size_t index = 0; index != definition.vocabulary.size(); ++index) {
        const VocabEntry& entry = definition.vocabulary[index];
        digest.u32(static_cast<uint32_t>(index));
        digest.u8(entry.type);
        digest.u16(entry.flags);
        digest.u16(entry.priority);
        digest.u32(float_bits(entry.score));
        digest.u32(static_cast<uint32_t>(entry.piece.size()));
        digest.text(entry.piece);
    }
    return digest.finish();
}

Sha256Digest v3_prompt_digest(const TokenProgramDefinition& definition) {
    CanonicalDigest digest;
    digest.text("LAPTOK-PROMPT-v3");
    digest.u8(0);
    digest.u32(definition.prompt_max_bytes);
    digest.u32(static_cast<uint32_t>(definition.prompt.size()));
    for (size_t index = 0; index != definition.prompt.size(); ++index) {
        const PromptInstruction& instruction = definition.prompt[index];
        digest.u32(static_cast<uint32_t>(index));
        digest.u8(static_cast<uint8_t>(instruction.opcode));
        digest.u32(static_cast<uint32_t>(instruction.literal.size()));
        digest.text(instruction.literal);
    }
    // Continuation-turn framing joins the same digest when present; packages
    // without it keep their previous digest bytes exactly.
    if (!definition.turn.empty()) {
        digest.u8(1);
        digest.u32(static_cast<uint32_t>(definition.turn.size()));
        for (size_t index = 0; index != definition.turn.size(); ++index) {
            const PromptInstruction& instruction = definition.turn[index];
            digest.u32(static_cast<uint32_t>(index));
            digest.u8(static_cast<uint8_t>(instruction.opcode));
            digest.u32(static_cast<uint32_t>(instruction.literal.size()));
            digest.text(instruction.literal);
        }
    }
    return digest.finish();
}

void append_v3_section(std::vector<uint8_t>& output, TokenProgramV3Section section,
                       const std::vector<uint8_t>& body) {
    append_u16(output, static_cast<uint16_t>(section));
    append_u16(output, kV3SectionRequired);
    append_u32(output, static_cast<uint32_t>(body.size()));
    output.insert(output.end(), body.begin(), body.end());
}

std::vector<uint8_t> v3_options(const TokenProgramDefinition& definition,
                                const Sha256Digest& vocabulary_digest,
                                const Sha256Digest& prompt_digest) {
    std::vector<uint8_t> body;
    body.push_back(static_cast<uint8_t>(definition.model_kind));
    body.push_back(definition.byte_fallback ? 1 : 0);
    append_u16(body, definition.bpe_flags);
    append_u32(body, static_cast<uint32_t>(definition.vocabulary.size()));
    append_u32(body, definition.unknown_token_id);
    append_u32(body, definition.postprocessor.bos_token_id);
    append_u32(body, definition.postprocessor.eos_token_id);
    append_u32(body, definition.prompt_max_bytes);
    append_u32(body, definition.stream_max_bytes);
    append_u32(body, static_cast<uint32_t>(definition.stop_ids.size()));
    for (const uint32_t id : definition.stop_ids) append_u32(body, id);
    append_digest_v3(body, vocabulary_digest);
    append_digest_v3(body, prompt_digest);
    return body;
}

std::vector<uint8_t> v3_normalizer(const TokenProgramDefinition& definition) {
    std::vector<uint8_t> body;
    append_spec(body, definition.normalizer);
    append_u32(body, static_cast<uint32_t>(definition.normalizer_data.size()));
    body.insert(body.end(), definition.normalizer_data.begin(), definition.normalizer_data.end());
    return body;
}

std::vector<uint8_t> v3_pretokenizer(const TokenProgramDefinition& definition) {
    std::vector<uint8_t> body;
    append_spec(body, definition.pretokenizer);
    append_u32(body, static_cast<uint32_t>(definition.pretokenizer_regex.size()));
    body.insert(body.end(), definition.pretokenizer_regex.begin(), definition.pretokenizer_regex.end());
    return body;
}

std::vector<uint8_t> v3_vocabulary(const TokenProgramDefinition& definition) {
    std::vector<uint8_t> body;
    append_u32(body, static_cast<uint32_t>(definition.vocabulary.size()));
    for (const VocabEntry& entry : definition.vocabulary) {
        append_u32(body, static_cast<uint32_t>(entry.piece.size()));
        body.push_back(entry.type);
        body.push_back(0);
        append_u16(body, entry.flags);
        append_u16(body, entry.priority);
        append_u16(body, 0);
        append_u32(body, float_bits(entry.score));
        body.insert(body.end(), entry.piece.begin(), entry.piece.end());
    }
    return body;
}

std::vector<uint8_t> v3_merges(const std::vector<MergeRecord>& merges) {
    std::vector<uint8_t> body;
    append_u32(body, static_cast<uint32_t>(merges.size()));
    for (const MergeRecord& merge : merges) {
        append_u32(body, merge.left_id);
        append_u32(body, merge.right_id);
        append_u32(body, merge.result_id);
        append_u32(body, merge.rank);
    }
    return body;
}

} // namespace

TokenProgram::SerializeResult serialize_token_program_v3(const TokenProgramDefinition& definition) {
    try {
        const TokenProgramStatus valid = validate_v3_definition(definition);
        if (!valid.ok()) return valid;
        const Sha256Digest vocabulary_digest = v3_vocabulary_digest(definition);
        const Sha256Digest prompt_digest = v3_prompt_digest(definition);
        std::vector<uint8_t> output;
        output.insert(output.end(), kV3Magic.begin(), kV3Magic.end());
        append_u16(output, kTokenProgramV3MajorVersion);
        append_u16(output, kTokenProgramV3MinorVersion);
        append_u16(output, kV3HeaderFlags);
        append_u16(output, kV3Reserved);
        append_u16(output, definition.turn.empty() ? 10 : 11);
        const std::vector<uint8_t> options = v3_options(definition, vocabulary_digest, prompt_digest);
        const std::vector<uint8_t> normalizer = v3_normalizer(definition);
        const std::vector<uint8_t> pretokenizer = v3_pretokenizer(definition);
        const std::vector<uint8_t> vocabulary = v3_vocabulary(definition);
        const std::vector<uint8_t> merges = v3_merges(definition.merges);
        const std::vector<uint8_t> added = make_v2_added(definition.added_tokens);
        const std::vector<uint8_t> postprocessor = make_v2_postprocessor(definition.postprocessor);
        const std::vector<uint8_t> decoder = make_v2_decoder(definition.decoder);
        const std::vector<uint8_t> prompt = make_v2_prompt(definition.prompt);
        std::vector<uint8_t> map;
        map.reserve(definition.byte_to_unicode.size() * sizeof(uint32_t));
        for (const uint32_t value : definition.byte_to_unicode) append_u32(map, value);
        append_v3_section(output, TokenProgramV3Section::Options, options);
        append_v3_section(output, TokenProgramV3Section::Normalizer, normalizer);
        append_v3_section(output, TokenProgramV3Section::Pretokenizer, pretokenizer);
        append_v3_section(output, TokenProgramV3Section::Vocabulary, vocabulary);
        append_v3_section(output, TokenProgramV3Section::Merges, merges);
        append_v3_section(output, TokenProgramV3Section::AddedTokens, added);
        append_v3_section(output, TokenProgramV3Section::Postprocessor, postprocessor);
        append_v3_section(output, TokenProgramV3Section::Decoder, decoder);
        append_v3_section(output, TokenProgramV3Section::Prompt, prompt);
        append_v3_section(output, TokenProgramV3Section::ByteToUnicode, map);
        // Packages without chat framing stay byte-identical to the previous
        // format. A turn-bearing package changes the prompt digest and
        // requires a reader that understands the new section.
        if (!definition.turn.empty()) {
            append_v3_section(output, TokenProgramV3Section::TurnPrompt,
                              make_v2_prompt(definition.turn));
        }
        if (output.size() > token_program_limits::kMaxPayloadBytes) {
            return failure(TokenProgramError::OutputTooLarge, "V3 tokenizer program is too large");
        }
        return output;
    } catch (const std::bad_alloc&) {
        return failure(TokenProgramError::OutputTooLarge, "V3 tokenizer program allocation failed");
    }
}

TokenProgram::CompileResult TokenProgram::compile_v3(std::span<const uint8_t> payload) {
    try {
        if (payload.size() > token_program_limits::kMaxPayloadBytes) {
            return failure(TokenProgramError::PayloadMalformed, "V3 tokenizer program payload is too large");
        }
        Reader reader(payload);
        std::span<const uint8_t> magic;
        if (!reader.take(kV3Magic.size(), magic) || !std::equal(magic.begin(), magic.end(), kV3Magic.begin())) {
            return failure(TokenProgramError::PayloadMalformed, "V3 tokenizer program magic is invalid", reader.offset());
        }
        uint16_t major = 0, minor = 0, header_flags = 0, reserved = 0, section_count = 0;
        if (!reader.u16(major) || !reader.u16(minor) || !reader.u16(header_flags) || !reader.u16(reserved) ||
            !reader.u16(section_count)) {
            return read_failure(reader, "V3 tokenizer program header is truncated");
        }
        if (major != kTokenProgramV3MajorVersion || minor != kTokenProgramV3MinorVersion) {
            return failure(TokenProgramError::UnsupportedVersion, "V3 tokenizer program version is unsupported", 10);
        }
        if (header_flags != kV3HeaderFlags || reserved != kV3Reserved || section_count == 0 ||
            section_count > kV3MaxSections) {
            return failure(TokenProgramError::InvalidParameter, "V3 tokenizer program header is not canonical",
                           reader.offset());
        }
        struct Section {
            uint16_t kind = 0;
            uint16_t flags = 0;
            std::span<const uint8_t> payload;
            size_t offset = 0;
        };
        std::vector<Section> sections;
        sections.reserve(section_count);
        std::array<bool, 12> seen{};
        uint16_t previous_kind = 0;
        for (uint16_t index = 0; index != section_count; ++index) {
            Section section;
            uint32_t length = 0;
            section.offset = reader.offset();
            if (!reader.u16(section.kind) || !reader.u16(section.flags) || !reader.u32(length) ||
                section.kind <= previous_kind || (section.flags & ~kV3KnownSectionFlags) != 0 ||
                length > reader.remaining() || !reader.take(length, section.payload)) {
                return failure(TokenProgramError::PayloadMalformed, "V3 section is malformed", section.offset);
            }
            previous_kind = section.kind;
            if (section.kind >= seen.size()) {
                if ((section.flags & kV3SectionRequired) != 0) {
                    return failure(TokenProgramError::UnknownRequiredSection, "unknown required V3 section", section.offset);
                }
            } else if (seen[section.kind]) {
                return failure(TokenProgramError::DuplicateRecord, "duplicate V3 section", section.offset);
            } else {
                seen[section.kind] = true;
            }
            sections.push_back(section);
        }
        if (reader.remaining() != 0) {
            return failure(TokenProgramError::TrailingBytes, "V3 tokenizer program has trailing bytes", reader.offset());
        }
        const auto find_section = [&](TokenProgramV3Section kind) -> const Section* {
            for (const Section& section : sections) {
                if (section.kind == static_cast<uint16_t>(kind)) return &section;
            }
            return nullptr;
        };
        const auto require_section = [&](TokenProgramV3Section kind) -> const Section* {
            const Section* section = find_section(kind);
            return section != nullptr && (section->flags & kV3SectionRequired) != 0 ? section : nullptr;
        };
        const Section* options_section = require_section(TokenProgramV3Section::Options);
        const Section* normalizer_section = require_section(TokenProgramV3Section::Normalizer);
        const Section* pretokenizer_section = require_section(TokenProgramV3Section::Pretokenizer);
        const Section* vocabulary_section = require_section(TokenProgramV3Section::Vocabulary);
        const Section* merges_section = require_section(TokenProgramV3Section::Merges);
        const Section* added_section = require_section(TokenProgramV3Section::AddedTokens);
        const Section* postprocessor_section = require_section(TokenProgramV3Section::Postprocessor);
        const Section* decoder_section = require_section(TokenProgramV3Section::Decoder);
        const Section* prompt_section = require_section(TokenProgramV3Section::Prompt);
        const Section* map_section = require_section(TokenProgramV3Section::ByteToUnicode);
        if (options_section == nullptr || normalizer_section == nullptr || pretokenizer_section == nullptr ||
            vocabulary_section == nullptr || merges_section == nullptr || added_section == nullptr ||
            postprocessor_section == nullptr || decoder_section == nullptr || prompt_section == nullptr ||
            map_section == nullptr) {
            return failure(TokenProgramError::UnknownRequiredSection, "required V3 section is missing");
        }
        TokenProgramDefinition definition;
        uint32_t expected_vocabulary_count = 0;
        uint32_t expected_bos = kTokenProgramNoTokenId;
        uint32_t expected_eos = kTokenProgramNoTokenId;
        Sha256Digest expected_vocabulary_digest;
        Sha256Digest expected_prompt_digest;
        {
            Reader section(options_section->payload);
            uint8_t model_kind = 0;
            uint8_t byte_fallback = 0;
            uint16_t bpe_flags = 0;
            uint32_t stop_count = 0;
            if (!section.u8(model_kind) || !section.u8(byte_fallback) || !section.u16(bpe_flags) ||
                !section.u32(expected_vocabulary_count) || !section.u32(definition.unknown_token_id) ||
                !section.u32(expected_bos) || !section.u32(expected_eos) || !section.u32(definition.prompt_max_bytes) ||
                !section.u32(definition.stream_max_bytes) || !section.u32(stop_count) ||
                stop_count > token_program_limits::kMaxStopIds || stop_count > section.remaining() / sizeof(uint32_t)) {
                return failure(TokenProgramError::PayloadMalformed, "V3 options section is malformed",
                               options_section->offset);
            }
            if (byte_fallback > 1) return failure(TokenProgramError::InvalidParameter, "V3 byte fallback flag is invalid",
                                                   options_section->offset);
            definition.model_kind = static_cast<TokenProgramModelKind>(model_kind);
            definition.byte_fallback = byte_fallback != 0;
            definition.bpe_flags = bpe_flags;
            definition.stop_ids.resize(stop_count);
            for (uint32_t& id : definition.stop_ids) if (!section.u32(id)) {
                return read_failure(section, "V3 stop list is truncated");
            }
            if (!read_digest_v3(section, expected_vocabulary_digest) ||
                !read_digest_v3(section, expected_prompt_digest) || section.remaining() != 0) {
                return failure(TokenProgramError::PayloadMalformed, "V3 options digest fields are malformed",
                               options_section->offset);
            }
        }
        {
            if (map_section->payload.size() != definition.byte_to_unicode.size() * sizeof(uint32_t)) {
                return failure(TokenProgramError::PayloadMalformed, "V3 byte map section has wrong length",
                               map_section->offset);
            }
            Reader section(map_section->payload);
            for (uint32_t& value : definition.byte_to_unicode) if (!section.u32(value)) {
                return read_failure(section, "V3 byte map is truncated");
            }
        }
        {
            Reader section(normalizer_section->payload);
            uint32_t length = 0;
            if (!read_normalizer(section, definition.normalizer) || !section.u32(length) ||
                length > token_program_limits::kMaxNormalizerDataBytes || length > section.remaining()) {
                return failure(TokenProgramError::PayloadMalformed, "V3 normalizer section is malformed",
                               normalizer_section->offset);
            }
            std::span<const uint8_t> data;
            section.take(length, data);
            definition.normalizer_data.assign(data.begin(), data.end());
            if (section.remaining() != 0) return failure(TokenProgramError::TrailingBytes,
                                                          "V3 normalizer section has trailing bytes",
                                                          normalizer_section->offset);
        }
        {
            Reader section(pretokenizer_section->payload);
            uint32_t length = 0;
            if (!read_pretokenizer(section, definition.pretokenizer) || !section.u32(length) ||
                length > token_program_limits::kMaxPretokenizerBytes || length > section.remaining()) {
                return failure(TokenProgramError::PayloadMalformed, "V3 pretokenizer section is malformed",
                               pretokenizer_section->offset);
            }
            std::span<const uint8_t> data;
            section.take(length, data);
            definition.pretokenizer_regex.assign(reinterpret_cast<const char*>(data.data()), data.size());
            if (section.remaining() != 0) return failure(TokenProgramError::TrailingBytes,
                                                          "V3 pretokenizer section has trailing bytes",
                                                          pretokenizer_section->offset);
        }
        {
            Reader section(vocabulary_section->payload);
            uint32_t count = 0;
            if (!section.u32(count) || count == 0 || count > token_program_limits::kMaxVocabulary ||
                count != expected_vocabulary_count) {
                return failure(TokenProgramError::PayloadMalformed, "V3 vocabulary count is outside its bound",
                               vocabulary_section->offset);
            }
            definition.vocabulary.reserve(count);
            for (uint32_t index = 0; index != count; ++index) {
                uint32_t length = 0, score_bits = 0;
                uint8_t type = 0, record_reserved = 0;
                uint16_t flags = 0, priority = 0, score_reserved = 0;
                if (!section.u32(length) || !section.u8(type) || !section.u8(record_reserved) ||
                    !section.u16(flags) || !section.u16(priority) || !section.u16(score_reserved) ||
                    !section.u32(score_bits) || record_reserved != 0 || score_reserved != 0 || length == 0 ||
                    length > token_program_limits::kMaxPieceBytes || length > section.remaining()) {
                    return failure(TokenProgramError::PayloadMalformed, "V3 vocabulary record is malformed",
                                   vocabulary_section->offset, index);
                }
                std::span<const uint8_t> piece;
                section.take(length, piece);
                definition.vocabulary.push_back({
                    std::string(reinterpret_cast<const char*>(piece.data()), piece.size()), flags, priority, type,
                    bits_float(score_bits)});
            }
            if (section.remaining() != 0) return failure(TokenProgramError::TrailingBytes,
                                                          "V3 vocabulary section has trailing bytes",
                                                          vocabulary_section->offset);
        }
        {
            Reader section(merges_section->payload);
            uint32_t count = 0;
            if (!section.u32(count) || count > token_program_limits::kMaxMerges || count > section.remaining() / 16) {
                return failure(TokenProgramError::PayloadMalformed, "V3 merge count is outside its bound",
                               merges_section->offset);
            }
            definition.merges.reserve(count);
            for (uint32_t index = 0; index != count; ++index) {
                MergeRecord merge;
                if (!section.u32(merge.left_id) || !section.u32(merge.right_id) || !section.u32(merge.result_id) ||
                    !section.u32(merge.rank)) return read_failure(section, "V3 merge record is truncated");
                definition.merges.push_back(merge);
            }
            if (section.remaining() != 0) return failure(TokenProgramError::TrailingBytes,
                                                          "V3 merge section has trailing bytes", merges_section->offset);
        }
        {
            Reader section(added_section->payload);
            uint32_t count = 0;
            if (!section.u32(count) || count > token_program_limits::kMaxAddedTokens) {
                return failure(TokenProgramError::PayloadMalformed, "V3 added-token count is outside its bound",
                               added_section->offset);
            }
            definition.added_tokens.reserve(count);
            for (uint32_t index = 0; index != count; ++index) {
                AddedTokenRecord added;
                uint32_t match_length = 0, normalized_length = 0;
                if (!section.u32(added.token_id) || !section.u16(added.flags) || !section.u16(added.priority) ||
                    !section.u32(match_length) || !section.u32(normalized_length) || match_length == 0 ||
                    normalized_length == 0 || match_length > token_program_limits::kMaxPieceBytes ||
                    normalized_length > token_program_limits::kMaxPieceBytes || match_length > section.remaining() ||
                    normalized_length > section.remaining() - match_length) {
                    return failure(TokenProgramError::PayloadMalformed, "V3 added-token record is malformed",
                                   added_section->offset, index);
                }
                std::span<const uint8_t> match, normalized;
                section.take(match_length, match);
                section.take(normalized_length, normalized);
                added.match.assign(reinterpret_cast<const char*>(match.data()), match.size());
                added.normalized.assign(reinterpret_cast<const char*>(normalized.data()), normalized.size());
                definition.added_tokens.push_back(std::move(added));
            }
            if (section.remaining() != 0) return failure(TokenProgramError::TrailingBytes,
                                                          "V3 added-token section has trailing bytes", added_section->offset);
        }
        {
            Reader section(postprocessor_section->payload);
            if (!read_postprocessor(section, definition.postprocessor) || section.remaining() != 0) {
                return failure(TokenProgramError::PayloadMalformed, "V3 postprocessor section is malformed",
                               postprocessor_section->offset);
            }
            if (definition.postprocessor.bos_token_id != expected_bos ||
                definition.postprocessor.eos_token_id != expected_eos) {
                return failure(TokenProgramError::InvalidParameter, "V3 special-token policy is inconsistent",
                               postprocessor_section->offset);
            }
        }
        {
            Reader section(decoder_section->payload);
            if (!read_decoder(section, definition.decoder) || section.remaining() != 0) {
                return failure(TokenProgramError::PayloadMalformed, "V3 decoder section is malformed",
                               decoder_section->offset);
            }
        }
        {
            Reader section(prompt_section->payload);
            uint32_t count = 0;
            if (!section.u32(count) || count == 0 || count > token_program_limits::kMaxPromptInstructions) {
                return failure(TokenProgramError::PayloadMalformed, "V3 prompt count is outside its bound",
                               prompt_section->offset);
            }
            definition.prompt.reserve(count);
            for (uint32_t index = 0; index != count; ++index) {
                uint8_t opcode = 0, instruction_flags = 0;
                uint16_t instruction_reserved = 0;
                uint32_t argument = 0, length = 0;
                if (!section.u8(opcode) || !section.u8(instruction_flags) || !section.u16(instruction_reserved) ||
                    !section.u32(argument) || !section.u32(length) || instruction_flags != 0 ||
                    instruction_reserved != 0 || argument != 0 || length > token_program_limits::kMaxPromptLiteralBytes ||
                    length > section.remaining()) {
                    return failure(TokenProgramError::PayloadMalformed, "V3 prompt instruction is malformed",
                                   prompt_section->offset, index);
                }
                std::span<const uint8_t> literal;
                section.take(length, literal);
                definition.prompt.push_back({static_cast<PromptOpcode>(opcode),
                                             std::string(reinterpret_cast<const char*>(literal.data()), literal.size())});
            }
            if (section.remaining() != 0) return failure(TokenProgramError::TrailingBytes,
                                                          "V3 prompt section has trailing bytes", prompt_section->offset);
        }
        {
            const Section* turn_section = nullptr;
            for (const Section& candidate : sections)
                if (candidate.kind == static_cast<uint16_t>(TokenProgramV3Section::TurnPrompt))
                    turn_section = &candidate;
            if (turn_section != nullptr) {
                Reader section(turn_section->payload);
                uint32_t count = 0;
                if (!section.u32(count) || count == 0 ||
                    count > token_program_limits::kMaxPromptInstructions) {
                    return failure(TokenProgramError::PayloadMalformed,
                                   "V3 turn-prompt count is outside its bound",
                                   turn_section->offset);
                }
                definition.turn.reserve(count);
                for (uint32_t index = 0; index != count; ++index) {
                    uint8_t opcode = 0, instruction_flags = 0;
                    uint16_t instruction_reserved = 0;
                    uint32_t argument = 0, length = 0;
                    if (!section.u8(opcode) || !section.u8(instruction_flags) ||
                        !section.u16(instruction_reserved) || !section.u32(argument) ||
                        !section.u32(length) || instruction_flags != 0 ||
                        instruction_reserved != 0 || argument != 0 ||
                        length > token_program_limits::kMaxPromptLiteralBytes ||
                        length > section.remaining()) {
                        return failure(TokenProgramError::PayloadMalformed,
                                       "V3 turn-prompt instruction is malformed",
                                       turn_section->offset, index);
                    }
                    std::span<const uint8_t> literal;
                    section.take(length, literal);
                    definition.turn.push_back(
                        {static_cast<PromptOpcode>(opcode),
                         std::string(reinterpret_cast<const char*>(literal.data()), literal.size())});
                }
                if (section.remaining() != 0)
                    return failure(TokenProgramError::TrailingBytes,
                                   "V3 turn-prompt section has trailing bytes",
                                   turn_section->offset);
            }
        }
        const TokenProgramStatus valid = validate_v3_definition(definition);
        if (!valid.ok()) return valid;
        if (v3_vocabulary_digest(definition) != expected_vocabulary_digest ||
            v3_prompt_digest(definition) != expected_prompt_digest) {
            return failure(TokenProgramError::DigestMismatch, "V3 tokenizer digest does not match its sections");
        }
        definition.vocabulary_digest = expected_vocabulary_digest;
        definition.prompt_program_digest = expected_prompt_digest;
        TokenProgram program(std::move(definition));
        program.wire_major_ = kTokenProgramV3MajorVersion;
        return program;
    } catch (const std::bad_alloc&) {
        return failure(TokenProgramError::PayloadMalformed, "V3 tokenizer program allocation failed");
    }
}

TokenProgram::CompileResult compile_token_program_v3(std::span<const uint8_t> payload) {
    return TokenProgram::compile_v3(payload);
}

namespace {

TokenProgramStatus normalize_sentencepiece_v3(std::string_view input, const TokenProgramDefinition& definition,
                                               std::string& output) {
    if (!valid_utf8(input)) return failure(TokenProgramError::InvalidUtf8, "V3 input text is not valid UTF-8");
    if (!definition.normalizer_data.empty()) {
        return failure(TokenProgramError::UnsupportedEnum,
                       "V3 precompiled SentencePiece normalizer data is not executable in this build");
    }
    const uint8_t flags = definition.normalizer.flags;
    std::string source;
    source.reserve(input.size() + 3);
    bool prior_space = false;
    bool emitted = false;
    std::vector<ScalarSpan> scalars;
    if (!collect_scalars(input, scalars)) return failure(TokenProgramError::InvalidUtf8, "V3 input UTF-8 is invalid");
    const bool remove_extra = (flags & static_cast<uint8_t>(SentencePieceNormalizerFlags::RemoveExtraWhitespaces)) != 0;
    for (const ScalarSpan scalar : scalars) {
        const bool space = scalar_is_whitespace(scalar.value);
        if (remove_extra && space) {
            if (!emitted || prior_space) continue;
            prior_space = true;
            source.push_back(' ');
            continue;
        }
        if (remove_extra && source.size() == 1 && source[0] == ' ' && space) continue;
        prior_space = space;
        emitted = true;
        source.append(input.substr(scalar.begin, scalar.end - scalar.begin));
    }
    if (remove_extra && !source.empty() && source.back() == ' ') source.pop_back();
    output.clear();
    const bool add_dummy = (flags & static_cast<uint8_t>(SentencePieceNormalizerFlags::AddDummyPrefix)) != 0;
    const bool escape = (flags & static_cast<uint8_t>(SentencePieceNormalizerFlags::EscapeWhitespaces)) != 0;
    if (add_dummy && (source.empty() || source.front() != ' ')) source.insert(source.begin(), ' ');
    std::vector<ScalarSpan> normalized_scalars;
    if (!collect_scalars(source, normalized_scalars)) return failure(TokenProgramError::InvalidUtf8, "V3 normalized UTF-8 is invalid");
    for (const ScalarSpan scalar : normalized_scalars) {
        // SentencePiece escapes the ASCII space only; newlines and tabs
        // stay literal and match their own pieces.
        if (escape && scalar.value == ' ') {
            output.append("\xe2\x96\x81");
        } else {
            output.append(source.substr(scalar.begin, scalar.end - scalar.begin));
        }
    }
    if ((flags & static_cast<uint8_t>(SentencePieceNormalizerFlags::TreatWhitespaceAsSuffix)) != 0 &&
        output.size() >= 3 && output.compare(output.size() - 3, 3, "\xe2\x96\x81") != 0) {
        output.append("\xe2\x96\x81");
    }
    return {};
}

std::vector<std::string_view> scan_unicode_scalars(std::string_view text, bool group_newline_runs);

TokenProgram::EncodeResult encode_v3(const TokenProgram& program, std::string_view text,
                                    bool add_postprocessor_start = true) {
    const TokenProgramDefinition& definition = program.definition();
    std::unordered_map<std::string, uint32_t> piece_ids;
    piece_ids.reserve(definition.vocabulary.size());
    for (size_t index = 0; index != definition.vocabulary.size(); ++index) {
        piece_ids.emplace(definition.vocabulary[index].piece, static_cast<uint32_t>(index));
    }
    std::unordered_map<uint64_t, uint32_t> merge_indices;
    merge_indices.reserve(definition.merges.size());
    for (size_t index = 0; index != definition.merges.size(); ++index) {
        const MergeRecord& merge = definition.merges[index];
        merge_indices.emplace(pair_key(merge.left_id, merge.right_id), static_cast<uint32_t>(index));
    }
    if (text.size() > token_program_limits::kMaxInputBytes) {
        return failure(TokenProgramError::InputTooLarge, "V3 input text is too large");
    }
    std::vector<uint32_t> output;
    output.reserve(std::min<size_t>(text.size() + 2, token_program_limits::kMaxInputBytes));
    const auto append_bos = [&]() {
        if (definition.postprocessor.kind == PostprocessorKind::AddBosEos &&
            (definition.postprocessor.flags & static_cast<uint8_t>(PostprocessorFlags::AddBos)) != 0) {
            output.push_back(definition.postprocessor.bos_token_id);
        }
    };
    const auto append_eos = [&]() {
        if (definition.postprocessor.kind == PostprocessorKind::AddBosEos &&
            (definition.postprocessor.flags & static_cast<uint8_t>(PostprocessorFlags::AddEos)) != 0) {
            output.push_back(definition.postprocessor.eos_token_id);
        }
    };
    if (add_postprocessor_start) append_bos();
    uint64_t work = 0;
    if (definition.model_kind == TokenProgramModelKind::SentencePiece) {
        std::unordered_map<std::string, uint32_t> byte_ids;
        for (size_t index = 0; index != definition.vocabulary.size(); ++index) {
            if (static_cast<TokenPieceType>(definition.vocabulary[index].type) == TokenPieceType::Byte) {
                byte_ids.emplace(definition.vocabulary[index].piece, static_cast<uint32_t>(index));
            }
        }
        // Declared control and user-defined symbols match raw text before
        // normalization; everything else is normalized per segment and
        // segmented greedily.
        struct RawItem {
            bool is_token = false;
            uint32_t token = kTokenProgramNoTokenId;
            std::string text;
        };
        std::vector<RawItem> items;
        {
            std::vector<std::pair<size_t, uint32_t>> user_defined;
            for (size_t index = 0; index != definition.vocabulary.size(); ++index) {
                if (!v3_is_special(definition.vocabulary[index]) ||
                    definition.vocabulary[index].piece.empty()) {
                    continue;
                }
                user_defined.emplace_back(definition.vocabulary[index].piece.size(),
                                          static_cast<uint32_t>(index));
            }
            std::sort(user_defined.begin(), user_defined.end(),
                      [](const auto& left, const auto& right) {
                          if (left.first != right.first) return left.first > right.first;
                          return left.second < right.second;
                      });
            std::string pending;
            size_t offset = 0;
            while (offset < text.size()) {
                bool matched = false;
                for (const auto& [size, id] : user_defined) {
                    if (++work > kV3MaxExecutionWork)
                        return failure(TokenProgramError::InputTooLarge, "token matching exceeds its work bound");
                    if (size > text.size() - offset) continue;
                    if (text.compare(offset, size,
                                     definition.vocabulary[id].piece) != 0) {
                        continue;
                    }
                    if (!pending.empty()) {
                        items.push_back({false, kTokenProgramNoTokenId, std::move(pending)});
                        pending.clear();
                    }
                    items.push_back({true, id, {}});
                    offset += size;
                    matched = true;
                    break;
                }
                if (!matched) {
                    pending.push_back(text[offset]);
                    ++offset;
                }
            }
            if (!pending.empty()) {
                items.push_back({false, kTokenProgramNoTokenId, std::move(pending)});
            }
        }
        bool first_segment = true;
        for (const RawItem& item : items) {
            if (item.is_token) {
                output.push_back(item.token);
                first_segment = false;
                continue;
            }
            std::string normalized;
            const TokenProgramStatus status =
                normalize_sentencepiece_v3(item.text, definition, normalized);
            if (!status.ok()) return status;
            if (first_segment) {
                first_segment = false;
            } else if ((definition.normalizer.flags &
                        static_cast<uint8_t>(SentencePieceNormalizerFlags::AddDummyPrefix)) != 0 &&
                       normalized.size() >= 3 &&
                       static_cast<uint8_t>(normalized[0]) == 0xe2 &&
                       static_cast<uint8_t>(normalized[1]) == 0x96 &&
                       static_cast<uint8_t>(normalized[2]) == 0x81) {
                // The dummy prefix belongs to the whole input, not to each
                // segment split by a user-defined symbol.
                normalized.erase(0, 3);
            }
            size_t offset = 0;
            while (offset < normalized.size()) {
                uint32_t best_id = kTokenProgramNoTokenId;
                size_t best_size = 0;
                for (size_t index = 0; index != definition.vocabulary.size(); ++index) {
                    const VocabEntry& entry = definition.vocabulary[index];
                    if (!v3_is_regular(entry) || entry.piece.size() <= best_size ||
                        entry.piece.size() > normalized.size() - offset) {
                        continue;
                    }
                    if (++work > kV3MaxExecutionWork) {
                        return failure(TokenProgramError::InputTooLarge,
                                       "V3 tokenizer work bound exceeded");
                    }
                    if (normalized.compare(offset, entry.piece.size(), entry.piece) != 0) {
                        continue;
                    }
                    best_id = static_cast<uint32_t>(index);
                    best_size = entry.piece.size();
                }
                if (best_id != kTokenProgramNoTokenId) {
                    output.push_back(best_id);
                    offset += best_size;
                    continue;
                }
                if (definition.byte_fallback) {
                    const uint8_t byte = static_cast<uint8_t>(normalized[offset]);
                    const auto found = byte_ids.find(v3_byte_piece(byte));
                    if (found != byte_ids.end()) {
                        output.push_back(found->second);
                        ++offset;
                        continue;
                    }
                }
                if (definition.unknown_token_id != kTokenProgramNoTokenId) {
                    uint32_t scalar = 0;
                    size_t scalar_length = 0;
                    if (!decode_scalar_at(normalized, offset, scalar, scalar_length)) {
                        if ((static_cast<uint8_t>(normalized[offset]) & 0xc0) == 0x80) {
                            ++offset;
                            continue;
                        }
                        return failure(TokenProgramError::InvalidUtf8,
                                       "V3 normalized SentencePiece text is invalid");
                    }
                    output.push_back(definition.unknown_token_id);
                    offset += scalar_length;
                    continue;
                }
                return failure(TokenProgramError::UnknownPiece,
                               "V3 SentencePiece position has no piece, byte, or unknown coverage",
                               SIZE_MAX, static_cast<uint32_t>(offset));
            }
        }
        append_eos();
        return output;
    }

    if (definition.pretokenizer.kind == PretokenizerKind::Regex) {
        return failure(TokenProgramError::UnsupportedEnum, "V3 regex pretokenizer execution is unsupported");
    }
    auto append_merged = [&](std::string_view segment) -> TokenProgramStatus {
        if (segment.empty()) return {};
        std::vector<uint32_t> symbols;
        symbols.reserve(segment.size());
        for (const uint8_t byte : std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(segment.data()), segment.size())) {
            const std::string mapped = scalar_utf8(definition.byte_to_unicode[byte]);
            const auto found = piece_ids.find(mapped);
            if (found != piece_ids.end() && v3_is_regular(definition.vocabulary[found->second])) {
                symbols.push_back(found->second);
                continue;
            }
            const auto fallback = piece_ids.find(v3_byte_piece(byte));
            if (definition.byte_fallback && fallback != piece_ids.end() &&
                static_cast<TokenPieceType>(definition.vocabulary[fallback->second].type) == TokenPieceType::Byte) {
                symbols.push_back(fallback->second);
            } else if (definition.unknown_token_id != kTokenProgramNoTokenId) {
                symbols.push_back(definition.unknown_token_id);
            } else {
                return failure(TokenProgramError::UnknownPiece, "V3 byte has no vocabulary entry");
            }
        }
        struct Node { uint32_t token; uint32_t previous; uint32_t next; uint32_t generation; bool live; };
        struct Candidate { uint32_t rank, left, right, left_generation, right_generation, result; };
        struct Later {
            bool operator()(const Candidate& left, const Candidate& right) const {
                if (left.rank != right.rank) return left.rank > right.rank;
                return left.left > right.left;
            }
        };
        std::vector<Node> nodes(symbols.size());
        for (size_t index = 0; index != symbols.size(); ++index) {
            nodes[index] = {symbols[index], index == 0 ? UINT32_MAX : static_cast<uint32_t>(index - 1),
                            index + 1 == symbols.size() ? UINT32_MAX : static_cast<uint32_t>(index + 1), 0, true};
        }
        std::priority_queue<Candidate, std::vector<Candidate>, Later> candidates;
        const auto add_candidate = [&](uint32_t left) {
            if (left == UINT32_MAX || !nodes[left].live || nodes[left].next == UINT32_MAX) return;
            const uint32_t right = nodes[left].next;
            const auto found = merge_indices.find(pair_key(nodes[left].token, nodes[right].token));
            if (found == merge_indices.end()) return;
            const MergeRecord& merge = definition.merges[found->second];
            candidates.push({merge.rank, left, right, nodes[left].generation, nodes[right].generation, merge.result_id});
        };
        for (uint32_t index = 0; index + 1 < nodes.size(); ++index) add_candidate(index);
        while (!candidates.empty()) {
            const Candidate candidate = candidates.top();
            candidates.pop();
            Node& left = nodes[candidate.left];
            Node& right = nodes[candidate.right];
            if (!left.live || !right.live || left.next != candidate.right ||
                left.generation != candidate.left_generation || right.generation != candidate.right_generation) continue;
            left.token = candidate.result;
            ++left.generation;
            right.live = false;
            ++right.generation;
            left.next = right.next;
            if (right.next != UINT32_MAX) nodes[right.next].previous = candidate.left;
            add_candidate(left.previous);
            add_candidate(candidate.left);
        }
        for (uint32_t index = nodes.empty() ? UINT32_MAX : 0; index != UINT32_MAX; index = nodes[index].next) {
            if (nodes[index].live) output.push_back(nodes[index].token);
        }
        return {};
    };
    const bool split = (definition.pretokenizer.flags & static_cast<uint8_t>(PretokenizerFlags::SplitAsciiWhitespace)) != 0;
    const bool prefix = (definition.pretokenizer.flags & static_cast<uint8_t>(PretokenizerFlags::AddPrefixSpace)) != 0;
    const auto append_regular = [&](std::string_view source, bool add_prefix) -> TokenProgramStatus {
        std::string canonical;
        const TokenProgramStatus status = normalize_text(definition.normalizer, source, canonical);
        if (!status.ok()) return status;
        if (add_prefix && prefix && !canonical.empty()) canonical.insert(canonical.begin(), ' ');
        if (split) {
            size_t begin = 0;
            for (size_t index = 0; index <= canonical.size(); ++index) {
                if (index != canonical.size() && !is_ascii_space(static_cast<uint8_t>(canonical[index]))) continue;
                if (index != begin) {
                    const TokenProgramStatus status = append_merged(std::string_view(canonical).substr(begin, index - begin));
                    if (!status.ok()) return status;
                }
                if (index != canonical.size()) {
                    const TokenProgramStatus status = append_merged(std::string_view(canonical).substr(index, 1));
                    if (!status.ok()) return status;
                }
                begin = index + 1;
            }
        } else if (definition.pretokenizer.kind == PretokenizerKind::ByteLevel) {
            // Byte-level BPE groups by the family-neutral scalar scanner
            // before merging, so merges never cross a word boundary.
            for (std::string_view segment : scan_unicode_scalars(
                     canonical,
                     (definition.pretokenizer.flags &
                      static_cast<uint8_t>(PretokenizerFlags::GroupNewlineRuns)) != 0)) {
                const TokenProgramStatus status = append_merged(segment);
                if (!status.ok()) return status;
            }
        } else {
            const TokenProgramStatus status = append_merged(canonical);
            if (!status.ok()) return status;
        }
        return {};
    };
    struct RawSegment {
        size_t begin = 0;
        size_t end = 0;
        uint32_t token = kTokenProgramNoTokenId;
    };
    std::vector<std::pair<size_t, uint32_t>> special_tokens;
    for (size_t index = 0; index != definition.vocabulary.size(); ++index) {
        if (!v3_is_special(definition.vocabulary[index]) || definition.vocabulary[index].piece.empty()) continue;
        special_tokens.emplace_back(definition.vocabulary[index].piece.size(), static_cast<uint32_t>(index));
    }
    std::sort(special_tokens.begin(), special_tokens.end(), [](const auto& left, const auto& right) {
        if (left.first != right.first) return left.first > right.first;
        return left.second < right.second;
    });
    std::vector<RawSegment> segments;
    size_t pending_begin = 0;
    size_t offset = 0;
    while (offset < text.size()) {
        size_t matched_size = 0;
        uint32_t matched_id = kTokenProgramNoTokenId;
        for (const auto& [size, id] : special_tokens) {
            if (++work > kV3MaxExecutionWork)
                return failure(TokenProgramError::InputTooLarge, "token matching exceeds its work bound");
            if (size <= text.size() - offset && text.compare(offset, size, definition.vocabulary[id].piece) == 0) {
                matched_size = size;
                matched_id = id;
                break;
            }
        }
        if (matched_id == kTokenProgramNoTokenId) {
            ++offset;
            continue;
        }
        if (pending_begin != offset) segments.push_back({pending_begin, offset});
        segments.push_back({offset, offset + matched_size, matched_id});
        offset += matched_size;
        pending_begin = offset;
    }
    if (pending_begin != text.size()) segments.push_back({pending_begin, text.size()});
    if (segments.empty()) {
        const TokenProgramStatus status = append_regular(text, true);
        if (!status.ok()) return status;
    } else {
        bool prefix_pending = true;
        for (const RawSegment& segment : segments) {
            if (segment.token != kTokenProgramNoTokenId) {
                output.push_back(segment.token);
                continue;
            }
            const TokenProgramStatus status = append_regular(
                text.substr(segment.begin, segment.end - segment.begin), prefix_pending);
            if (!status.ok()) return status;
            prefix_pending = false;
        }
    }
    append_eos();
    return output;
}

TokenProgram::DecodeResult decode_v3(const TokenProgram& program, std::span<const uint32_t> token_ids,
                                     bool strip_sentencepiece_prefix) {
    const TokenProgramDefinition& definition = program.definition();
    std::unordered_map<uint32_t, uint8_t> inverse_unicode;
    inverse_unicode.reserve(definition.byte_to_unicode.size());
    for (size_t index = 0; index != definition.byte_to_unicode.size(); ++index) {
        inverse_unicode.emplace(definition.byte_to_unicode[index], static_cast<uint8_t>(index));
    }
    if (token_ids.size() > token_program_limits::kMaxInputBytes) {
        return failure(TokenProgramError::InputTooLarge, "V3 token sequence is too large");
    }
    const bool skip_special = (definition.decoder.flags & static_cast<uint8_t>(DecoderFlags::SkipSpecial)) != 0;
    std::string output;
    for (size_t index = 0; index != token_ids.size(); ++index) {
        const uint32_t id = token_ids[index];
        if (id >= definition.vocabulary.size()) return failure(TokenProgramError::InvalidTokenId,
                                                                "V3 decode token ID is outside the vocabulary", SIZE_MAX,
                                                                static_cast<uint32_t>(index));
        const VocabEntry& entry = definition.vocabulary[id];
        const TokenPieceType type = static_cast<TokenPieceType>(entry.type);
        const bool special = (entry.flags & static_cast<uint16_t>(VocabFlags::Special)) != 0 ||
                             type == TokenPieceType::Control;
        if (special && skip_special) continue;
        if (type == TokenPieceType::Control) continue;
        if (type == TokenPieceType::Byte) {
            if (entry.piece.size() != 6 || entry.piece[0] != '<' || entry.piece[1] != '0' || entry.piece[2] != 'x' || entry.piece[5] != '>') {
                return failure(TokenProgramError::UnknownPiece, "V3 byte piece is malformed", SIZE_MAX,
                               static_cast<uint32_t>(index));
            }
            auto hex = [](char value) -> int {
                if (value >= '0' && value <= '9') return value - '0';
                if (value >= 'a' && value <= 'f') return value - 'a' + 10;
                if (value >= 'A' && value <= 'F') return value - 'A' + 10;
                return -1;
            };
            const int high = hex(entry.piece[3]);
            const int low = hex(entry.piece[4]);
            if (high < 0 || low < 0) return failure(TokenProgramError::UnknownPiece, "V3 byte piece is malformed", SIZE_MAX,
                                                     static_cast<uint32_t>(index));
            output.push_back(static_cast<char>((high << 4) | low));
        } else if (definition.model_kind == TokenProgramModelKind::SentencePiece) {
            // SentencePiece uses U+2581 as its serialized word-boundary
            // marker.  Decode it back to a space; the leading marker added by
            // the normalizer is synthetic and is removed after the token
            // sequence is assembled.
            size_t piece_offset = 0;
            while (piece_offset < entry.piece.size()) {
                if (piece_offset + 3 <= entry.piece.size() &&
                    entry.piece.compare(piece_offset, 3, "\xe2\x96\x81") == 0) {
                    output.push_back(' ');
                    piece_offset += 3;
                } else {
                    output.push_back(entry.piece[piece_offset++]);
                }
            }
        } else if (definition.decoder.kind == DecoderKind::Identity) {
            output.append(entry.piece);
        } else {
            std::vector<ScalarSpan> scalars;
            if (!collect_scalars(entry.piece, scalars)) return failure(TokenProgramError::InvalidUtf8,
                                                                        "V3 byte-BPE piece is not UTF-8", SIZE_MAX,
                                                                        static_cast<uint32_t>(index));
            for (const ScalarSpan scalar : scalars) {
                const auto found = inverse_unicode.find(scalar.value);
                if (found == inverse_unicode.end()) return failure(TokenProgramError::UnknownPiece,
                                                                                  "V3 decoder encountered a scalar outside the byte map",
                                                                                  SIZE_MAX, static_cast<uint32_t>(index));
                output.push_back(static_cast<char>(found->second));
            }
        }
        if (output.size() > definition.stream_max_bytes) return failure(TokenProgramError::OutputTooLarge,
                                                                          "V3 decoded text exceeds its stream bound", SIZE_MAX,
                                                                          static_cast<uint32_t>(index));
    }
    if (strip_sentencepiece_prefix && definition.model_kind == TokenProgramModelKind::SentencePiece &&
        !output.empty() && output.front() == ' ') {
        output.erase(output.begin());
    }
    return output;
}

} // namespace

TokenProgram::DecodeResult TokenProgram::decode_chunk(std::span<const uint32_t> token_ids, StreamState& state,
                                                      bool final_chunk) const {
    if (state.finished) return failure(TokenProgramError::InvalidParameter, "V3 decode stream is already finished");
    DecodeResult decoded = is_v3() ? decode_v3(*this, token_ids, false) : decode(token_ids);
    if (const auto* status = std::get_if<TokenProgramStatus>(&decoded)) return *status;
    std::string text = std::get<std::string>(std::move(decoded));
    if (is_v3() && definition_.model_kind == TokenProgramModelKind::SentencePiece &&
        !state.sentencepiece_prefix_stripped && !text.empty()) {
        state.sentencepiece_prefix_stripped = true;
        if (text.front() == ' ') text.erase(text.begin());
    }
    if (state.pending_utf8.size() > definition_.stream_max_bytes || text.size() > definition_.stream_max_bytes ||
        state.pending_utf8.size() > definition_.stream_max_bytes - text.size() ||
        state.decoded_bytes > definition_.stream_max_bytes - state.pending_utf8.size() - text.size()) {
        return failure(TokenProgramError::OutputTooLarge, "V3 decode stream exceeds its bound");
    }
    std::string combined = state.pending_utf8;
    combined.append(text);
    state.pending_utf8.clear();
    std::string emitted = combined;
    {
        size_t offset = 0;
        size_t incomplete_offset = SIZE_MAX;
        while (offset < combined.size()) {
            uint32_t scalar = 0;
            size_t length = 0;
            if (decode_scalar_at(combined, offset, scalar, length)) {
                offset += length;
                continue;
            }
            const uint8_t first = static_cast<uint8_t>(combined[offset]);
            const size_t expected = first <= 0x7f ? 1 :
                                    (first >= 0xc2 && first <= 0xdf ? 2 :
                                     (first >= 0xe0 && first <= 0xef ? 3 :
                                      (first >= 0xf0 && first <= 0xf4 ? 4 : 0)));
            if (expected != 0 && expected > combined.size() - offset) {
                incomplete_offset = offset;
            } else {
                return failure(TokenProgramError::InvalidUtf8,
                               "V3 decode stream contains invalid UTF-8");
            }
            break;
        }
        if (incomplete_offset != SIZE_MAX) {
            if (final_chunk) {
                return failure(TokenProgramError::InvalidUtf8,
                               "V3 decode stream ends with incomplete UTF-8");
            }
            emitted.assign(combined.data(), incomplete_offset);
            state.pending_utf8.assign(combined.data() + incomplete_offset, combined.size() - incomplete_offset);
        }
    }
    state.decoded_bytes += emitted.size();
    state.finished = final_chunk;
    return emitted;
}

namespace {
TokenProgram::EncodeResult encode_v2(const TokenProgram&, std::string_view);
TokenProgram::DecodeResult decode_v2(const TokenProgram&, std::span<const uint32_t>);
TokenProgram::EncodeResult encode_v3(const TokenProgram&, std::string_view, bool add_postprocessor_start);
TokenProgram::DecodeResult decode_v3(const TokenProgram&, std::span<const uint32_t>, bool strip_sentencepiece_prefix);
}

TokenProgram::TokenProgram(TokenProgramDefinition definition) : definition_(std::move(definition)) {
    for (size_t index = 0; index != inverse_byte_map_.size(); ++index) {
        inverse_byte_map_[definition_.byte_map[index]] = static_cast<uint8_t>(index);
    }
    inverse_unicode_map_.reserve(definition_.byte_to_unicode.size());
    for (size_t index = 0; index != definition_.byte_to_unicode.size(); ++index) {
        inverse_unicode_map_.emplace(definition_.byte_to_unicode[index], static_cast<uint8_t>(index));
    }
    piece_ids_.reserve(definition_.vocabulary.size());
    for (size_t index = 0; index != definition_.vocabulary.size(); ++index) {
        piece_ids_.emplace(definition_.vocabulary[index].piece, static_cast<uint32_t>(index));
    }
    merge_indices_.reserve(definition_.merges.size());
    for (size_t index = 0; index != definition_.merges.size(); ++index) {
        const MergeRecord& merge = definition_.merges[index];
        merge_indices_.emplace(pair_key(merge.left_id, merge.right_id), static_cast<uint32_t>(index));
    }
    for (size_t index = 0; index != definition_.added_tokens.size(); ++index) {
        const AddedTokenRecord& added = definition_.added_tokens[index];
        const bool normalized =
            (added.flags & static_cast<uint16_t>(AddedTokenFlags::Normalized)) != 0;
        const std::string& key = normalized ? added.normalized : added.match;
        auto& bucket = normalized_added_by_first_byte_[static_cast<uint8_t>(key[0])];
        if (!normalized) {
            original_added_by_first_byte_[static_cast<uint8_t>(key[0])].push_back(
                static_cast<uint32_t>(index));
        } else {
            bucket.push_back(static_cast<uint32_t>(index));
        }
    }
}

TokenProgram TokenProgram::token_ids_only(uint32_t vocabulary_size) {
    TokenProgram program(TokenProgramDefinition{});
    (void)vocabulary_size;
    program.wire_major_ = 0;
    program.token_ids_only_ = true;
    return program;
}

TokenProgram::SerializeResult serialize_token_program(const TokenProgramDefinition& definition) {
    const TokenProgramStatus valid = validate_definition(definition);
    if (!valid.ok()) return valid;

    std::vector<uint8_t> bytes;
    bytes.reserve(1024);
    bytes.insert(bytes.end(), kMagic.begin(), kMagic.end());
    append_u16(bytes, kMajorVersion);
    append_u16(bytes, kMinorVersion);
    append_u16(bytes, definition.bpe_flags);
    append_u32(bytes, static_cast<uint32_t>(definition.vocabulary.size()));
    append_u32(bytes, static_cast<uint32_t>(definition.merges.size()));
    append_u32(bytes, static_cast<uint32_t>(definition.added_tokens.size()));
    append_u32(bytes, static_cast<uint32_t>(definition.prompt.size()));
    append_u32(bytes, definition.prompt_max_bytes);
    append_u32(bytes, definition.unknown_token_id);
    append_u32(bytes, definition.postprocessor.kind == PostprocessorKind::AddBosEos &&
                            (definition.postprocessor.flags & static_cast<uint8_t>(PostprocessorFlags::AddBos))
                        ? definition.postprocessor.bos_token_id
                        : kTokenProgramNoTokenId);
    append_u32(bytes, definition.postprocessor.kind == PostprocessorKind::AddBosEos &&
                            (definition.postprocessor.flags & static_cast<uint8_t>(PostprocessorFlags::AddEos))
                        ? definition.postprocessor.eos_token_id
                        : kTokenProgramNoTokenId);
    bytes.insert(bytes.end(), definition.byte_map.begin(), definition.byte_map.end());
    append_spec(bytes, definition.normalizer);
    append_spec(bytes, definition.pretokenizer);
    append_spec(bytes, definition.postprocessor);
    append_spec(bytes, definition.decoder);
    for (const VocabEntry& entry : definition.vocabulary) {
        append_u32(bytes, static_cast<uint32_t>(entry.piece.size()));
        append_u16(bytes, entry.flags);
        append_u16(bytes, entry.priority);
        bytes.insert(bytes.end(), entry.piece.begin(), entry.piece.end());
    }
    for (const MergeRecord& merge : definition.merges) {
        append_u32(bytes, merge.left_id);
        append_u32(bytes, merge.right_id);
        append_u32(bytes, merge.result_id);
        append_u32(bytes, merge.rank);
    }
    for (const AddedTokenRecord& added : definition.added_tokens) {
        append_u32(bytes, added.token_id);
        append_u16(bytes, added.flags);
        append_u16(bytes, added.priority);
        append_u32(bytes, static_cast<uint32_t>(added.match.size()));
        append_u32(bytes, static_cast<uint32_t>(added.normalized.size()));
        bytes.insert(bytes.end(), added.match.begin(), added.match.end());
        bytes.insert(bytes.end(), added.normalized.begin(), added.normalized.end());
    }
    for (const PromptInstruction& instruction : definition.prompt) {
        bytes.push_back(static_cast<uint8_t>(instruction.opcode));
        bytes.push_back(0); // instruction flags
        append_u16(bytes, 0); // reserved
        append_u32(bytes, 0); // argument
        append_u32(bytes, static_cast<uint32_t>(instruction.literal.size()));
        bytes.insert(bytes.end(), instruction.literal.begin(), instruction.literal.end());
    }
    if (bytes.size() > token_program_limits::kMaxPayloadBytes) {
        return failure(TokenProgramError::OutputTooLarge, "serialized tokenizer program is too large");
    }
    return bytes;
}

TokenProgram::CompileResult TokenProgram::compile(std::span<const uint8_t> payload) {
    if (payload.size() >= kV3Magic.size() &&
        std::equal(kV3Magic.begin(), kV3Magic.end(), payload.begin())) {
        return compile_token_program_v3(payload);
    }
    if (payload.size() >= kV2Magic.size() &&
        std::equal(kV2Magic.begin(), kV2Magic.end(), payload.begin())) {
        return compile_token_program_v2(payload);
    }
    try {
    if (payload.size() > token_program_limits::kMaxPayloadBytes) {
        return failure(TokenProgramError::PayloadMalformed, "tokenizer program payload is too large");
    }
    Reader reader(payload);
    std::span<const uint8_t> magic;
    if (!reader.take(kMagic.size(), magic) || !std::equal(magic.begin(), magic.end(), kMagic.begin())) {
        return failure(TokenProgramError::PayloadMalformed, "tokenizer program magic is invalid", reader.offset());
    }
    uint16_t major = 0;
    uint16_t minor = 0;
    uint16_t flags = 0;
    uint32_t vocabulary_count = 0;
    uint32_t merge_count = 0;
    uint32_t added_count = 0;
    uint32_t prompt_count = 0;
    uint32_t prompt_max_bytes = 0;
    uint32_t unknown_token_id = kTokenProgramNoTokenId;
    uint32_t header_bos_id = kTokenProgramNoTokenId;
    uint32_t header_eos_id = kTokenProgramNoTokenId;
    if (!reader.u16(major) || !reader.u16(minor) || !reader.u16(flags) || !reader.u32(vocabulary_count) ||
        !reader.u32(merge_count) || !reader.u32(added_count) || !reader.u32(prompt_count) ||
        !reader.u32(prompt_max_bytes) || !reader.u32(unknown_token_id) || !reader.u32(header_bos_id) ||
        !reader.u32(header_eos_id)) {
        return read_failure(reader, "tokenizer program header is truncated");
    }
    if (major != kMajorVersion || minor != kMinorVersion) {
        return failure(TokenProgramError::UnsupportedVersion, "tokenizer program version is unsupported", 10);
    }
    if ((flags & ~kKnownBpeFlags) != 0) {
        return failure(TokenProgramError::InvalidParameter, "tokenizer program BPE flags are unsupported", 14);
    }
    if (vocabulary_count == 0 || vocabulary_count > token_program_limits::kMaxVocabulary ||
        merge_count > token_program_limits::kMaxMerges || added_count > token_program_limits::kMaxAddedTokens ||
        prompt_count == 0 || prompt_count > token_program_limits::kMaxPromptInstructions) {
        return failure(TokenProgramError::PayloadMalformed, "tokenizer program count is outside its bound", reader.offset());
    }
    if (prompt_max_bytes == 0 || prompt_max_bytes > token_program_limits::kMaxPromptBytes) {
        return failure(TokenProgramError::InvalidPrompt,
                       "tokenizer program prompt bound is invalid", reader.offset());
    }

    TokenProgramDefinition definition;
    definition.bpe_flags = flags;
    definition.prompt_max_bytes = prompt_max_bytes;
    definition.unknown_token_id = unknown_token_id;
    if (reader.remaining() < definition.byte_map.size()) return read_failure(reader, "byte map is truncated");
    std::span<const uint8_t> map;
    reader.take(definition.byte_map.size(), map);
    std::copy(map.begin(), map.end(), definition.byte_map.begin());
    if (!read_normalizer(reader, definition.normalizer) || !read_pretokenizer(reader, definition.pretokenizer) ||
        !read_postprocessor(reader, definition.postprocessor) || !read_decoder(reader, definition.decoder)) {
        return read_failure(reader, "component specification is truncated");
    }
    if (definition.postprocessor.kind == PostprocessorKind::AddBosEos) {
        if ((definition.postprocessor.flags & static_cast<uint8_t>(PostprocessorFlags::AddBos)) != 0) {
            if (definition.postprocessor.bos_token_id != header_bos_id) {
                return failure(TokenProgramError::InvalidParameter,
                               "BOS token ID is inconsistent between header and postprocessor", reader.offset());
            }
            definition.postprocessor.bos_token_id = header_bos_id;
        }
        if ((definition.postprocessor.flags & static_cast<uint8_t>(PostprocessorFlags::AddEos)) != 0) {
            if (definition.postprocessor.eos_token_id != header_eos_id) {
                return failure(TokenProgramError::InvalidParameter,
                               "EOS token ID is inconsistent between header and postprocessor", reader.offset());
            }
            definition.postprocessor.eos_token_id = header_eos_id;
        }
    } else if (header_bos_id != kTokenProgramNoTokenId || header_eos_id != kTokenProgramNoTokenId) {
        return failure(TokenProgramError::InvalidParameter, "unused header special-token IDs are not canonical",
                       reader.offset());
    }

    size_t minimum_record_bytes = 0;
    if (!add_minimum_bytes(vocabulary_count, 9, minimum_record_bytes) ||
        !add_minimum_bytes(merge_count, 16, minimum_record_bytes) ||
        !add_minimum_bytes(added_count, 18, minimum_record_bytes) ||
        !add_minimum_bytes(prompt_count, 12, minimum_record_bytes) ||
        minimum_record_bytes > reader.remaining()) {
        return failure(TokenProgramError::PayloadMalformed,
                       "tokenizer program counts exceed the remaining payload", reader.offset());
    }

    definition.vocabulary.reserve(vocabulary_count);
    for (uint32_t index = 0; index != vocabulary_count; ++index) {
        uint32_t length = 0;
        uint16_t entry_flags = 0;
        uint16_t priority = 0;
        if (!reader.u32(length) || !reader.u16(entry_flags) || !reader.u16(priority)) {
            return read_failure(reader, "vocabulary record is truncated");
        }
        if (length == 0 || length > token_program_limits::kMaxPieceBytes || length > reader.remaining()) {
            return failure(TokenProgramError::PayloadMalformed, "vocabulary piece length is invalid", reader.offset(), index);
        }
        std::span<const uint8_t> piece;
        reader.take(length, piece);
        definition.vocabulary.push_back({std::string(reinterpret_cast<const char*>(piece.data()), piece.size()),
                                         entry_flags, priority});
    }

    if (!has_size_for_count(merge_count, 16, reader.remaining())) {
        return failure(TokenProgramError::PayloadMalformed, "merge table is truncated", reader.offset());
    }
    definition.merges.reserve(merge_count);
    for (uint32_t index = 0; index != merge_count; ++index) {
        MergeRecord merge;
        if (!reader.u32(merge.left_id) || !reader.u32(merge.right_id) || !reader.u32(merge.result_id) ||
            !reader.u32(merge.rank)) {
            return read_failure(reader, "merge record is truncated");
        }
        definition.merges.push_back(merge);
    }

    definition.added_tokens.reserve(added_count);
    for (uint32_t index = 0; index != added_count; ++index) {
        AddedTokenRecord added;
        uint32_t match_length = 0;
        uint32_t normalized_length = 0;
        if (!reader.u32(added.token_id) || !reader.u16(added.flags) || !reader.u16(added.priority) ||
            !reader.u32(match_length) || !reader.u32(normalized_length)) {
            return read_failure(reader, "added-token record is truncated");
        }
        if (match_length == 0 || normalized_length == 0 ||
            match_length > token_program_limits::kMaxPieceBytes ||
            normalized_length > token_program_limits::kMaxPieceBytes ||
            match_length > reader.remaining() || normalized_length > reader.remaining() - match_length) {
            return failure(TokenProgramError::PayloadMalformed, "added-token text length is invalid", reader.offset(), index);
        }
        std::span<const uint8_t> match;
        std::span<const uint8_t> normalized;
        reader.take(match_length, match);
        reader.take(normalized_length, normalized);
        added.match.assign(reinterpret_cast<const char*>(match.data()), match.size());
        added.normalized.assign(reinterpret_cast<const char*>(normalized.data()), normalized.size());
        definition.added_tokens.push_back(std::move(added));
    }

    definition.prompt.reserve(prompt_count);
    for (uint32_t index = 0; index != prompt_count; ++index) {
        uint8_t opcode = 0;
        uint8_t instruction_flags = 0;
        uint16_t reserved = 0;
        uint32_t argument = 0;
        uint32_t literal_length = 0;
        if (!reader.u8(opcode) || !reader.u8(instruction_flags) || !reader.u16(reserved) ||
            !reader.u32(argument) || !reader.u32(literal_length)) {
            return read_failure(reader, "prompt instruction is truncated");
        }
        if (instruction_flags != 0 || reserved != 0 || argument != 0 || literal_length > reader.remaining()) {
            return failure(TokenProgramError::InvalidPrompt, "prompt instruction parameters are not canonical",
                           reader.offset(), index);
        }
        if (literal_length > token_program_limits::kMaxPromptLiteralBytes) {
            return failure(TokenProgramError::InvalidPrompt, "prompt literal is too large", reader.offset(), index);
        }
        std::span<const uint8_t> literal;
        reader.take(literal_length, literal);
        definition.prompt.push_back({static_cast<PromptOpcode>(opcode),
                                     std::string(reinterpret_cast<const char*>(literal.data()), literal.size())});
    }
    if (reader.remaining() != 0) {
        return failure(TokenProgramError::TrailingBytes, "tokenizer program has trailing bytes", reader.offset());
    }
    const TokenProgramStatus valid = validate_definition(definition);
    if (!valid.ok()) return valid;
    return TokenProgram(std::move(definition));
    } catch (const std::bad_alloc&) {
        return failure(TokenProgramError::PayloadMalformed,
                       "tokenizer program allocation failed");
    }
}

TokenProgram::SerializeResult TokenProgram::serialize() const {
    if (token_ids_only_) return failure(TokenProgramError::UnsupportedVersion,
                                       "token-ID-only package has no text tokenizer program");
    if (is_v3()) return serialize_token_program_v3(definition_);
    if (is_v2()) return serialize_token_program_v2(definition_);
    return serialize_token_program(definition_);
}

Sha256Digest TokenProgram::vocabulary_digest() const {
    if (token_ids_only_) return {};
    if (is_v3()) return v3_vocabulary_digest(definition_);
    CanonicalDigest digest;
    digest.text("LAPTOK-VOCAB-v1");
    digest.u8(0);
    digest.u32(static_cast<uint32_t>(definition_.vocabulary.size()));
    for (size_t index = 0; index != definition_.vocabulary.size(); ++index) {
        const VocabEntry& entry = definition_.vocabulary[index];
        digest.u32(static_cast<uint32_t>(index));
        digest.u16(entry.flags);
        digest.u16(entry.priority);
        digest.u32(static_cast<uint32_t>(entry.piece.size()));
        digest.text(entry.piece);
    }
    return digest.finish();
}

Sha256Digest TokenProgram::prompt_digest() const {
    if (token_ids_only_) return {};
    if (is_v3()) return v3_prompt_digest(definition_);
    CanonicalDigest digest;
    digest.text("LAPTOK-PROMPT-v1");
    digest.u8(0);
    digest.u32(definition_.prompt_max_bytes);
    digest.u32(static_cast<uint32_t>(definition_.prompt.size()));
    for (size_t index = 0; index != definition_.prompt.size(); ++index) {
        const PromptInstruction& instruction = definition_.prompt[index];
        digest.u32(static_cast<uint32_t>(index));
        digest.u8(static_cast<uint8_t>(instruction.opcode));
        digest.u32(static_cast<uint32_t>(instruction.literal.size()));
        digest.text(instruction.literal);
    }
    return digest.finish();
}

TokenProgram::CompileResult compile_token_program(std::span<const uint8_t> payload) {
    return TokenProgram::compile(payload);
}

namespace {

TokenProgramStatus normalize_text(const NormalizerSpec& spec, std::string_view input, std::string& output) {
    if (!valid_utf8(input)) return failure(TokenProgramError::InvalidUtf8, "input text is not valid UTF-8");
    output.assign(input.begin(), input.end());
    if (spec.kind == NormalizerKind::AsciiLowercase) {
        for (char& value : output) {
            const uint8_t byte = static_cast<uint8_t>(value);
            if (byte >= 'A' && byte <= 'Z') value = static_cast<char>(byte + ('a' - 'A'));
        }
    }
    return {};
}

struct AddedMatch {
    const AddedTokenRecord* record = nullptr;
    size_t start = 0;
    size_t end = 0;
};

bool better_added_match(const AddedMatch& candidate, const AddedMatch& current) {
    if (current.record == nullptr) return true;
    if (candidate.record->priority != current.record->priority) {
        return candidate.record->priority > current.record->priority;
    }
    const size_t candidate_length = candidate.record->match.size();
    const size_t current_length = current.record->match.size();
    if (candidate_length != current_length) return candidate_length > current_length;
    return candidate.record->token_id < current.record->token_id;
}

bool candidate_added_match(const AddedTokenRecord& record, std::string_view original,
                           std::string_view normalized, size_t cursor, AddedMatch& result) {
    const bool use_normalized = (record.flags & static_cast<uint16_t>(AddedTokenFlags::Normalized)) != 0;
    const std::string_view source = use_normalized ? normalized : original;
    const std::string_view needle = use_normalized ? std::string_view(record.normalized) : std::string_view(record.match);
    if (cursor > source.size() || needle.size() > source.size() - cursor ||
        source.compare(cursor, needle.size(), needle) != 0) {
        return false;
    }
    size_t start = cursor;
    size_t end = cursor + needle.size();
    if ((record.flags & static_cast<uint16_t>(AddedTokenFlags::LeftStrip)) != 0) {
        while (start != 0 && is_ascii_space(static_cast<uint8_t>(original[start - 1]))) --start;
    }
    if ((record.flags & static_cast<uint16_t>(AddedTokenFlags::RightStrip)) != 0) {
        while (end < original.size() && is_ascii_space(static_cast<uint8_t>(original[end]))) ++end;
    }
    if ((record.flags & static_cast<uint16_t>(AddedTokenFlags::SingleWord)) != 0) {
        if ((start != 0 && is_word_byte(static_cast<uint8_t>(original[start - 1]))) ||
            (end != original.size() && is_word_byte(static_cast<uint8_t>(original[end])))) {
            return false;
        }
    }
    result = {&record, start, end};
    return true;
}

} // namespace

TokenProgram::EncodeResult TokenProgram::encode(std::string_view text) const {
    if (token_ids_only_) return failure(TokenProgramError::UnsupportedVersion,
                                       "token-ID-only package does not encode text");
    if (is_v3()) return encode_v3(*this, text);
    if (is_v2()) return encode_v2(*this, text);
    if (text.size() > token_program_limits::kMaxInputBytes) {
        return failure(TokenProgramError::InputTooLarge, "input text is too large");
    }
    std::string normalized;
    const TokenProgramStatus normalized_status = normalize_text(definition_.normalizer, text, normalized);
    if (!normalized_status.ok()) return normalized_status;

    std::vector<uint32_t> output;
    output.reserve(text.size() + 2);
    const auto append_postprocessor_start = [&]() {
        if (definition_.postprocessor.kind == PostprocessorKind::AddBosEos &&
            (definition_.postprocessor.flags & static_cast<uint8_t>(PostprocessorFlags::AddBos)) != 0) {
            output.push_back(definition_.postprocessor.bos_token_id);
        }
    };
    const auto append_postprocessor_end = [&]() {
        if (definition_.postprocessor.kind == PostprocessorKind::AddBosEos &&
            (definition_.postprocessor.flags & static_cast<uint8_t>(PostprocessorFlags::AddEos)) != 0) {
            output.push_back(definition_.postprocessor.eos_token_id);
        }
    };
    append_postprocessor_start();

    const auto encode_regular = [&](std::string_view segment, bool prefix_space) -> TokenProgramStatus {
        if (segment.empty()) return {};
        std::string canonical;
        canonical.reserve(segment.size() + (prefix_space ? 1 : 0));
        if (prefix_space) canonical.push_back(' ');
        canonical.append(segment.begin(), segment.end());
        std::vector<std::string_view> pieces;
        if ((definition_.pretokenizer.flags & static_cast<uint8_t>(PretokenizerFlags::SplitAsciiWhitespace)) != 0) {
            size_t begin = 0;
            for (size_t index = 0; index != canonical.size(); ++index) {
                if (is_ascii_space(static_cast<uint8_t>(canonical[index]))) {
                    if (begin != index) pieces.emplace_back(canonical.data() + begin, index - begin);
                    pieces.emplace_back(canonical.data() + index, 1);
                    begin = index + 1;
                }
            }
            if (begin != canonical.size()) pieces.emplace_back(canonical.data() + begin, canonical.size() - begin);
        } else {
            pieces.emplace_back(canonical);
        }

        struct Node {
            uint32_t token = kTokenProgramNoTokenId;
            uint32_t previous = UINT32_MAX;
            uint32_t next = UINT32_MAX;
            uint32_t generation = 0;
            bool live = true;
        };
        struct Candidate {
            uint32_t rank = UINT32_MAX;
            uint32_t left = UINT32_MAX;
            uint32_t right = UINT32_MAX;
            uint32_t left_generation = 0;
            uint32_t right_generation = 0;
            uint32_t result = kTokenProgramNoTokenId;
        };
        struct LaterCandidate {
            bool operator()(const Candidate& left, const Candidate& right) const {
                if (left.rank != right.rank) return left.rank > right.rank;
                return left.left > right.left;
            }
        };

        const auto append_merged = [&](std::vector<uint32_t>& symbols) {
            if (symbols.empty()) return;
            std::vector<Node> nodes(symbols.size());
            for (size_t index = 0; index != symbols.size(); ++index) {
                Node& node = nodes[index];
                node.token = symbols[index];
                node.previous = index == 0 ? UINT32_MAX : static_cast<uint32_t>(index - 1);
                node.next = index + 1 == symbols.size() ? UINT32_MAX : static_cast<uint32_t>(index + 1);
            }
            std::priority_queue<Candidate, std::vector<Candidate>, LaterCandidate> candidates;
            const auto add_candidate = [&](uint32_t left) {
                if (left == UINT32_MAX || !nodes[left].live || nodes[left].next == UINT32_MAX) return;
                const uint32_t right = nodes[left].next;
                const auto found = merge_indices_.find(pair_key(nodes[left].token, nodes[right].token));
                if (found == merge_indices_.end()) return;
                const MergeRecord& merge = definition_.merges[found->second];
                candidates.push({merge.rank, left, right, nodes[left].generation,
                                 nodes[right].generation, merge.result_id});
            };
            for (uint32_t index = 0; index + 1 < nodes.size(); ++index) add_candidate(index);
            while (!candidates.empty()) {
                const Candidate candidate = candidates.top();
                candidates.pop();
                Node& left = nodes[candidate.left];
                Node& right = nodes[candidate.right];
                if (!left.live || !right.live || left.next != candidate.right ||
                    left.generation != candidate.left_generation ||
                    right.generation != candidate.right_generation) {
                    continue;
                }
                left.token = candidate.result;
                ++left.generation;
                right.live = false;
                ++right.generation;
                left.next = right.next;
                if (right.next != UINT32_MAX) nodes[right.next].previous = candidate.left;
                add_candidate(left.previous);
                add_candidate(candidate.left);
            }
            for (uint32_t index = 0; index != UINT32_MAX; index = nodes[index].next) {
                output.push_back(nodes[index].token);
            }
            symbols.clear();
        };

        for (const std::string_view piece : pieces) {
            std::vector<uint32_t> symbols;
            symbols.reserve(piece.size());
            bool previous_was_unknown = false;
            const bool fuse_unknown =
                (definition_.bpe_flags & static_cast<uint16_t>(BpeFlags::FuseUnknown)) != 0;
            for (const char value : piece) {
                const std::string mapped(1, static_cast<char>(definition_.byte_map[static_cast<uint8_t>(value)]));
                const auto found = piece_ids_.find(mapped);
                if (found != piece_ids_.end()) {
                    previous_was_unknown = false;
                    symbols.push_back(found->second);
                    continue;
                }
                if (definition_.unknown_token_id == kTokenProgramNoTokenId) {
                    return failure(TokenProgramError::UnknownPiece, "byte has no vocabulary entry");
                }
                if (!fuse_unknown || !previous_was_unknown) {
                    symbols.push_back(definition_.unknown_token_id);
                }
                previous_was_unknown = true;
            }
            append_merged(symbols);
        }
        return {};
    };

    const bool prefix_space = (definition_.pretokenizer.flags & static_cast<uint8_t>(PretokenizerFlags::AddPrefixSpace)) != 0;
    size_t segment_start = 0;
    size_t cursor = 0;
    bool first_regular = true;
    while (cursor < text.size()) {
        AddedMatch best;
        const auto consider = [&](const std::vector<uint32_t>& indices) {
            for (const uint32_t index : indices) {
                AddedMatch candidate;
                const AddedTokenRecord& added = definition_.added_tokens[index];
                if (candidate_added_match(added, text, normalized, cursor, candidate) &&
                    better_added_match(candidate, best)) {
                    best = candidate;
                }
            }
        };
        consider(original_added_by_first_byte_[static_cast<uint8_t>(text[cursor])]);
        consider(normalized_added_by_first_byte_[static_cast<uint8_t>(normalized[cursor])]);
        if (best.record == nullptr) {
            ++cursor;
            continue;
        }
        if (best.start < segment_start) {
            return failure(TokenProgramError::InvalidAddedToken, "added-token strip overlaps a prior match");
        }
        const std::string_view gap(normalized.data() + segment_start, best.start - segment_start);
        const TokenProgramStatus regular_status = encode_regular(gap, first_regular && prefix_space);
        if (!regular_status.ok()) return regular_status;
        first_regular = false;
        output.push_back(best.record->token_id);
        segment_start = best.end;
        cursor = best.end;
    }
    const std::string_view tail(normalized.data() + segment_start, normalized.size() - segment_start);
    const TokenProgramStatus tail_status = encode_regular(tail, first_regular && prefix_space);
    if (!tail_status.ok()) return tail_status;
    append_postprocessor_end();
    return output;
}

TokenProgram::PromptResult TokenProgram::render_turn(std::string_view user_text) const {
    if (token_ids_only_) return failure(TokenProgramError::UnsupportedVersion,
                                       "token-ID-only package has no text prompt");
    if (definition_.turn.empty()) return failure(TokenProgramError::InvalidPrompt,
                                                 "package has no continuation turn program");
    if (user_text.size() > token_program_limits::kMaxInputBytes) {
        return failure(TokenProgramError::InputTooLarge, "turn user text is too large");
    }
    if (!valid_utf8(user_text)) return failure(TokenProgramError::InvalidUtf8, "turn user text is not UTF-8");
    std::string rendered;
    rendered.reserve(std::min<size_t>(definition_.prompt_max_bytes, user_text.size() + 64));
    for (size_t index = 0; index != definition_.turn.size(); ++index) {
        const PromptInstruction& instruction = definition_.turn[index];
        const std::string_view emitted = instruction.opcode == PromptOpcode::EmitUserText
                                              ? user_text
                                              : std::string_view(instruction.literal);
        if (emitted.size() > definition_.prompt_max_bytes ||
            rendered.size() > definition_.prompt_max_bytes - emitted.size()) {
            return failure(TokenProgramError::PromptTooLarge, "rendered turn exceeds its bound",
                           SIZE_MAX, static_cast<uint32_t>(index));
        }
        rendered.append(emitted.begin(), emitted.end());
    }
    return rendered;
}

TokenProgram::EncodeResult TokenProgram::encode_continuation(std::string_view text) const {
    if (is_v3()) return encode_v3(*this, text, false);
    auto encoded = encode(text);
    if (const auto* status = std::get_if<TokenProgramStatus>(&encoded))
        return *status;
    std::vector<uint32_t> output =
        std::get<std::vector<uint32_t>>(std::move(encoded));
    const PostprocessorSpec& postprocessor = definition_.postprocessor;
    const uint8_t add_bos = static_cast<uint8_t>(PostprocessorFlags::AddBos);
    if (postprocessor.kind != PostprocessorKind::AddBosEos ||
        (postprocessor.flags & add_bos) == 0)
        return output;
    if (postprocessor.bos_token_id == kTokenProgramNoTokenId || output.empty() ||
        output.front() != postprocessor.bos_token_id) {
        return failure(TokenProgramError::InvalidParameter,
                       "continuation BOS contract is inconsistent");
    }
    output.erase(output.begin());
    return output;
}

TokenProgram::PromptResult TokenProgram::render_prompt(std::string_view user_text) const {
    if (token_ids_only_) return failure(TokenProgramError::UnsupportedVersion,
                                       "token-ID-only package has no text prompt");
    if (user_text.size() > token_program_limits::kMaxInputBytes) {
        return failure(TokenProgramError::InputTooLarge, "prompt user text is too large");
    }
    if (!valid_utf8(user_text)) return failure(TokenProgramError::InvalidUtf8, "prompt user text is not UTF-8");
    std::string rendered;
    rendered.reserve(std::min<size_t>(definition_.prompt_max_bytes, user_text.size() + 64));
    for (size_t index = 0; index != definition_.prompt.size(); ++index) {
        const PromptInstruction& instruction = definition_.prompt[index];
        const std::string_view emitted = instruction.opcode == PromptOpcode::EmitUserText
                                              ? user_text
                                              : std::string_view(instruction.literal);
        if (emitted.size() > definition_.prompt_max_bytes || rendered.size() > definition_.prompt_max_bytes - emitted.size()) {
            return failure(TokenProgramError::PromptTooLarge, "rendered prompt exceeds its bound", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        rendered.append(emitted.begin(), emitted.end());
    }
    return rendered;
}

TokenProgram::DecodeResult TokenProgram::decode(std::span<const uint32_t> token_ids) const {
    if (token_ids_only_) return failure(TokenProgramError::UnsupportedVersion,
                                       "token-ID-only package does not decode text");
    if (is_v3()) return decode_v3(*this, token_ids, true);
    if (is_v2()) return decode_v2(*this, token_ids);
    if (token_ids.size() > token_program_limits::kMaxInputBytes) {
        return failure(TokenProgramError::InputTooLarge, "token sequence is too large");
    }
    const bool skip_special = (definition_.decoder.flags & static_cast<uint8_t>(DecoderFlags::SkipSpecial)) != 0;
    std::string output;
    for (size_t index = 0; index != token_ids.size(); ++index) {
        const uint32_t token_id = token_ids[index];
        if (token_id >= definition_.vocabulary.size()) {
            return failure(TokenProgramError::InvalidTokenId, "decode token ID is outside the vocabulary", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        const VocabEntry& entry = definition_.vocabulary[token_id];
        const bool special = (entry.flags & static_cast<uint16_t>(VocabFlags::Special)) != 0;
        if (special && skip_special) continue;
        if (entry.piece.size() > token_program_limits::kMaxInputBytes ||
            output.size() > token_program_limits::kMaxInputBytes - entry.piece.size()) {
            return failure(TokenProgramError::OutputTooLarge, "decoded text exceeds its bound", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if (definition_.decoder.kind == DecoderKind::Identity || special) {
            output.append(entry.piece.begin(), entry.piece.end());
        } else {
            for (const char value : entry.piece) {
                output.push_back(static_cast<char>(inverse_byte_map_[static_cast<uint8_t>(value)]));
            }
        }
    }
    return output;
}

namespace {

constexpr uint16_t kV2HeaderFlags = 0;
constexpr uint16_t kV2Reserved = 0;
constexpr uint32_t kV2MaxSections = 64;

bool decode_scalar_at(std::string_view text, size_t offset, uint32_t& value, size_t& length) {
    if (offset >= text.size()) return false;
    const uint8_t first = static_cast<uint8_t>(text[offset]);
    if (first <= 0x7f) {
        value = first;
        length = 1;
    } else if (first >= 0xc2 && first <= 0xdf) {
        value = first & 0x1f;
        length = 2;
    } else if (first >= 0xe0 && first <= 0xef) {
        value = first & 0x0f;
        length = 3;
    } else if (first >= 0xf0 && first <= 0xf4) {
        value = first & 0x07;
        length = 4;
    } else {
        return false;
    }
    if (length > text.size() - offset) return false;
    for (size_t index = 1; index != length; ++index) {
        const uint8_t continuation = static_cast<uint8_t>(text[offset + index]);
        if ((continuation & 0xc0) != 0x80) return false;
        value = (value << 6) | (continuation & 0x3f);
    }
    if ((length == 2 && value < 0x80) || (length == 3 && value < 0x800) ||
        (length == 4 && value < 0x10000) || (value >= 0xd800 && value <= 0xdfff) ||
        value > 0x10ffff) {
        return false;
    }
    return true;
}

bool valid_scalar(uint32_t value) {
    return value <= 0x10ffff && !(value >= 0xd800 && value <= 0xdfff);
}

void append_scalar_utf8(std::vector<uint8_t>& out, uint32_t value) {
    if (value <= 0x7f) {
        out.push_back(static_cast<uint8_t>(value));
    } else if (value <= 0x7ff) {
        out.push_back(static_cast<uint8_t>(0xc0 | (value >> 6)));
        out.push_back(static_cast<uint8_t>(0x80 | (value & 0x3f)));
    } else if (value <= 0xffff) {
        out.push_back(static_cast<uint8_t>(0xe0 | (value >> 12)));
        out.push_back(static_cast<uint8_t>(0x80 | ((value >> 6) & 0x3f)));
        out.push_back(static_cast<uint8_t>(0x80 | (value & 0x3f)));
    } else {
        out.push_back(static_cast<uint8_t>(0xf0 | (value >> 18)));
        out.push_back(static_cast<uint8_t>(0x80 | ((value >> 12) & 0x3f)));
        out.push_back(static_cast<uint8_t>(0x80 | ((value >> 6) & 0x3f)));
        out.push_back(static_cast<uint8_t>(0x80 | (value & 0x3f)));
    }
}

std::string scalar_utf8(uint32_t value) {
    std::vector<uint8_t> bytes;
    bytes.reserve(4);
    append_scalar_utf8(bytes, value);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

bool collect_scalars(std::string_view text, std::vector<ScalarSpan>& result) {
    result.clear();
    size_t offset = 0;
    while (offset < text.size()) {
        uint32_t value = 0;
        size_t length = 0;
        if (!decode_scalar_at(text, offset, value, length)) return false;
        result.push_back({offset, offset + length, value});
        offset += length;
    }
    return true;
}

bool scalar_is_mark(uint32_t value) {
    return (value >= 0x0300 && value <= 0x036f) || (value >= 0x0483 && value <= 0x0489) ||
           (value >= 0x0591 && value <= 0x05bd) || value == 0x05bf ||
           (value >= 0x05c1 && value <= 0x05c2) || (value >= 0x05c4 && value <= 0x05c5) ||
           value == 0x05c7 || (value >= 0x0610 && value <= 0x061a) ||
           (value >= 0x064b && value <= 0x065f) || (value >= 0x0670 && value <= 0x0670) ||
           (value >= 0x06d6 && value <= 0x06ed) || (value >= 0x0900 && value <= 0x0903) ||
           (value >= 0x093a && value <= 0x094f) || (value >= 0x0951 && value <= 0x0957) ||
           (value >= 0x1ab0 && value <= 0x1aff) || (value >= 0x1dc0 && value <= 0x1dff) ||
           (value >= 0x20d0 && value <= 0x20ff) || (value >= 0xfe20 && value <= 0xfe2f);
}

bool scalar_is_number(uint32_t value) {
    return (value >= '0' && value <= '9') || (value >= 0x0660 && value <= 0x0669) ||
           (value >= 0x06f0 && value <= 0x06f9) || (value >= 0x0966 && value <= 0x096f) ||
           (value >= 0x09e6 && value <= 0x09ef) || (value >= 0x0a66 && value <= 0x0a6f) ||
           (value >= 0x0ae6 && value <= 0x0aef) || (value >= 0x0b66 && value <= 0x0b6f) ||
           (value >= 0x0be6 && value <= 0x0bef) || (value >= 0x0c66 && value <= 0x0c6f) ||
           (value >= 0x0ce6 && value <= 0x0cef) || (value >= 0x0d66 && value <= 0x0d6f) ||
           (value >= 0x0de6 && value <= 0x0def) || (value >= 0x0e50 && value <= 0x0e59) ||
           (value >= 0x0ed0 && value <= 0x0ed9) || (value >= 0x0f20 && value <= 0x0f29) ||
           (value >= 0x1040 && value <= 0x1049) || (value >= 0x17e0 && value <= 0x17e9) ||
           (value >= 0xff10 && value <= 0xff19);
}

bool scalar_is_whitespace(uint32_t value) {
    return value == ' ' || value == '\t' || value == '\n' || value == '\r' || value == '\f' ||
           value == '\v' || value == 0x85 || value == 0xa0 || value == 0x1680 ||
           (value >= 0x2000 && value <= 0x200a) || value == 0x2028 || value == 0x2029 ||
           value == 0x202f || value == 0x205f || value == 0x3000;
}

bool scalar_is_letter(uint32_t value) {
    if ((value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z')) return true;
    if (scalar_is_mark(value) || scalar_is_number(value) || scalar_is_whitespace(value)) return false;
    if ((value >= 0x2000 && value <= 0x206f) || (value >= 0x20a0 && value <= 0x2fff) ||
        (value >= 0x1f000 && value <= 0x1faff) || (value >= 0xff00 && value <= 0xff65)) {
        return false;
    }
    // The listed ranges cover punctuation and symbols that occur in the
    // scanner contract.  For other non-ASCII scalars, treating the scalar as
    // a letter keeps the scanner deterministic without a locale-dependent
    // Unicode database.
    return value >= 0x80;
}

bool scalar_is_word(uint32_t value) {
    return scalar_is_letter(value) || scalar_is_number(value) || scalar_is_mark(value) || value == '_';
}

bool scalar_is_newline(uint32_t value) { return value == '\n' || value == '\r'; }

std::vector<std::string_view> scan_unicode_scalars(std::string_view text, bool group_newline_runs);

std::vector<std::string_view> scan_unicode_scalars(std::string_view text, bool group_newline_runs) {
    std::vector<ScalarSpan> scalars;
    if (!collect_scalars(text, scalars)) return {};
    std::vector<std::string_view> result;
    size_t cursor = 0;
    while (cursor < scalars.size()) {
        const auto at = [&](size_t index) -> uint32_t { return scalars[index].value; };
        const size_t begin = cursor;

        // Keep the contraction alternatives explicit and deterministic:
        // 's 't 're 've 'm 'll 'd.
        if (at(cursor) == '\'' && cursor + 1 < scalars.size() && at(cursor + 1) < 0x80) {
            const char first = static_cast<char>(at(cursor + 1) | 0x20);
            size_t take = 0;
            if (first == 's' || first == 't' || first == 'm' || first == 'd') {
                take = 2;
            } else if (cursor + 2 < scalars.size() && at(cursor + 2) < 0x80) {
                const char second = static_cast<char>(at(cursor + 2) | 0x20);
                if ((first == 'r' && second == 'e') || (first == 'v' && second == 'e') ||
                    (first == 'l' && second == 'l')) {
                    take = 3;
                }
            }
            if (take != 0) {
                cursor += take;
                result.emplace_back(text.data() + scalars[begin].begin,
                                    scalars[cursor - 1].end - scalars[begin].begin);
                continue;
            }
        }

        if (scalar_is_letter(at(cursor))) {
            while (cursor < scalars.size() && scalar_is_letter(at(cursor))) ++cursor;
        } else if (!scalar_is_newline(at(cursor)) && !scalar_is_letter(at(cursor)) &&
                   !scalar_is_number(at(cursor)) && cursor + 1 < scalars.size() &&
                   scalar_is_letter(at(cursor + 1))) {
            // [^\\r\\n\\p{L}\\p{N}]?\\p{L}+
            ++cursor;
            while (cursor < scalars.size() && scalar_is_letter(at(cursor))) ++cursor;
        } else if (scalar_is_number(at(cursor))) {
            // Qwen/GPT-2's numeric branch consumes three Unicode scalars,
            // rather than three UTF-8 bytes.
            size_t count = 0;
            while (cursor < scalars.size() && count != 3 && scalar_is_number(at(cursor))) {
                ++cursor;
                ++count;
            }
        } else if (at(cursor) == ' ' && cursor + 1 < scalars.size() &&
                   !scalar_is_whitespace(at(cursor + 1)) && !scalar_is_letter(at(cursor + 1)) &&
                   !scalar_is_number(at(cursor + 1))) {
            // ?[^\\s\\p{L}\\p{N}]+ with its optional ASCII space. The
            // trailing [\\r\\n]* is Qwen's; the GPT-2 ByteLevel regex stops
            // at the newline and lets the whitespace rule take it.
            ++cursor;
            while (cursor < scalars.size() && !scalar_is_whitespace(at(cursor)) &&
                   !scalar_is_letter(at(cursor)) && !scalar_is_number(at(cursor))) ++cursor;
            if (group_newline_runs) {
                while (cursor < scalars.size() && scalar_is_newline(at(cursor))) ++cursor;
            }
        } else if (scalar_is_whitespace(at(cursor)) && group_newline_runs &&
                   scalar_is_newline(at(cursor))) {
            // Qwen-class \s*[\r\n]+ at a newline: fold the whole run.
            while (cursor < scalars.size() && scalar_is_newline(at(cursor))) ++cursor;
        } else if (scalar_is_whitespace(at(cursor))) {
            if (group_newline_runs && !scalar_is_newline(at(cursor))) {
                size_t probe = cursor;
                while (probe < scalars.size() && !scalar_is_newline(at(probe)) &&
                       scalar_is_whitespace(at(probe))) ++probe;
                if (probe < scalars.size() && scalar_is_newline(at(probe))) {
                    cursor = probe;
                    while (cursor < scalars.size() && scalar_is_newline(at(cursor))) ++cursor;
                } else {
                    while (cursor < scalars.size() && !scalar_is_newline(at(cursor)) &&
                           scalar_is_whitespace(at(cursor))) ++cursor;
                    if (cursor < scalars.size() && cursor - begin > 1) --cursor;
                }
            } else {
                // GPT-2 \s+(?!\S): a whitespace run of any kind keeps its
                // final scalar for whatever follows; a space joins the
                // next word, other whitespace becomes its own segment.
                while (cursor < scalars.size() && scalar_is_whitespace(at(cursor))) ++cursor;
                if (cursor < scalars.size() && cursor - begin > 1) --cursor;
            }
        } else if (scalar_is_newline(at(cursor))) {
            // A lone leftover newline: the regex's final \s+ alternative.
            ++cursor;
        } else {
            // ?[^\\s\\p{L}\\p{N}]+, including combining marks. Trailing
            // newlines belong to the Qwen contract only.
            if (scalar_is_whitespace(at(cursor))) ++cursor;
            while (cursor < scalars.size() && !scalar_is_whitespace(at(cursor)) &&
                   !scalar_is_letter(at(cursor)) && !scalar_is_number(at(cursor))) ++cursor;
            if (group_newline_runs) {
                while (cursor < scalars.size() && scalar_is_newline(at(cursor))) ++cursor;
            }
        }
        if (cursor == begin) ++cursor;
        result.emplace_back(text.data() + scalars[begin].begin,
                            scalars[cursor - 1].end - scalars[begin].begin);
    }
    return result;
}

std::vector<std::string_view> pretokenize_v2(std::string_view text, const PretokenizerSpec& spec) {
    if (spec.kind == PretokenizerKind::UnicodeScalarScanner)
        return scan_unicode_scalars(
            text, (spec.flags & static_cast<uint8_t>(PretokenizerFlags::GroupNewlineRuns)) != 0);
    if ((spec.flags & static_cast<uint8_t>(PretokenizerFlags::SplitAsciiWhitespace)) == 0) {
        return {text};
    }
    std::vector<std::string_view> result;
    size_t begin = 0;
    for (size_t index = 0; index != text.size(); ++index) {
        if (is_ascii_space(static_cast<uint8_t>(text[index]))) {
            if (begin != index) result.emplace_back(text.data() + begin, index - begin);
            result.emplace_back(text.data() + index, 1);
            begin = index + 1;
        }
    }
    if (begin != text.size()) result.emplace_back(text.data() + begin, text.size() - begin);
    return result;
}

TokenProgramStatus validate_v2_definition(const TokenProgramDefinition& definition) {
    if (definition.vocabulary.empty() || definition.vocabulary.size() > token_program_limits::kMaxVocabulary) {
        return failure(TokenProgramError::VocabularyNotDense, "V2 vocabulary must be non-empty and bounded");
    }
    if ((definition.bpe_flags & ~kKnownBpeFlags) != 0) {
        return failure(TokenProgramError::InvalidParameter, "V2 BPE flags are unsupported");
    }
    std::unordered_set<uint32_t> map_values;
    map_values.reserve(definition.byte_to_unicode.size());
    for (size_t index = 0; index != definition.byte_to_unicode.size(); ++index) {
        const uint32_t value = definition.byte_to_unicode[index];
        if (!valid_scalar(value) || !map_values.emplace(value).second) {
            return failure(TokenProgramError::InvalidParameter,
                           "V2 byte-to-Unicode map must be a scalar permutation", index);
        }
    }

    std::unordered_set<std::string> vocabulary_pieces;
    vocabulary_pieces.reserve(definition.vocabulary.size());
    for (size_t index = 0; index != definition.vocabulary.size(); ++index) {
        const VocabEntry& entry = definition.vocabulary[index];
        if (entry.piece.empty() || entry.piece.size() > token_program_limits::kMaxPieceBytes) {
            return failure(TokenProgramError::VocabularyNotDense, "V2 vocabulary piece is empty or too large",
                           SIZE_MAX, static_cast<uint32_t>(index));
        }
        if ((entry.flags & ~kKnownVocabFlags) != 0) {
            return failure(TokenProgramError::InvalidParameter, "V2 vocabulary flags are unsupported", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if (!valid_utf8(entry.piece) || !vocabulary_pieces.emplace(entry.piece).second) {
            return !valid_utf8(entry.piece)
                       ? failure(TokenProgramError::InvalidUtf8, "V2 vocabulary piece is not UTF-8", SIZE_MAX,
                                 static_cast<uint32_t>(index))
                       : failure(TokenProgramError::DuplicateRecord, "duplicate V2 vocabulary piece", SIZE_MAX,
                                 static_cast<uint32_t>(index));
        }
    }
    if (definition.unknown_token_id != kTokenProgramNoTokenId &&
        definition.unknown_token_id >= definition.vocabulary.size()) {
        return failure(TokenProgramError::InvalidTokenId, "V2 unknown token ID is outside the vocabulary");
    }

    switch (definition.normalizer.kind) {
    case NormalizerKind::None:
    case NormalizerKind::AsciiLowercase:
        if (definition.normalizer.flags != 0 || definition.normalizer.parameter != 0 ||
            definition.normalizer.argument != 0) {
            return failure(TokenProgramError::InvalidParameter, "V2 normalizer parameters are not canonical");
        }
        break;
    default:
        return failure(TokenProgramError::UnknownRequiredOperation, "unknown required V2 normalizer operation");
    }
    if (definition.pretokenizer.kind != PretokenizerKind::ByteLevel &&
        definition.pretokenizer.kind != PretokenizerKind::UnicodeScalarScanner) {
        return failure(TokenProgramError::UnknownRequiredOperation, "unknown required V2 pretokenizer operation");
    }
    if ((definition.pretokenizer.flags & ~kKnownPretokenizerFlags) != 0 ||
        definition.pretokenizer.parameter != 0 || definition.pretokenizer.argument != 0) {
        return failure(TokenProgramError::InvalidParameter, "V2 pretokenizer parameters are not canonical");
    }

    switch (definition.postprocessor.kind) {
    case PostprocessorKind::None:
        if (definition.postprocessor.flags != 0 || definition.postprocessor.parameter != 0 ||
            definition.postprocessor.bos_token_id != kTokenProgramNoTokenId ||
            definition.postprocessor.eos_token_id != kTokenProgramNoTokenId) {
            return failure(TokenProgramError::InvalidParameter, "V2 empty postprocessor has non-empty parameters");
        }
        break;
    case PostprocessorKind::AddBosEos:
        if ((definition.postprocessor.flags & ~kKnownPostprocessorFlags) != 0 ||
            definition.postprocessor.parameter != 0 || definition.postprocessor.flags == 0) {
            return failure(TokenProgramError::InvalidParameter, "V2 postprocessor parameters are unsupported");
        }
        if ((definition.postprocessor.flags & static_cast<uint8_t>(PostprocessorFlags::AddBos)) != 0) {
            if (definition.postprocessor.bos_token_id >= definition.vocabulary.size()) {
                return failure(TokenProgramError::InvalidTokenId, "V2 BOS token ID is outside the vocabulary");
            }
            if ((definition.vocabulary[definition.postprocessor.bos_token_id].flags &
                 static_cast<uint16_t>(VocabFlags::Special)) == 0) {
                return failure(TokenProgramError::InvalidTokenId, "V2 BOS token ID is not a declared special token");
            }
        } else if (definition.postprocessor.bos_token_id != kTokenProgramNoTokenId) {
            return failure(TokenProgramError::InvalidParameter, "unused V2 BOS token ID is not canonical");
        }
        if ((definition.postprocessor.flags & static_cast<uint8_t>(PostprocessorFlags::AddEos)) != 0) {
            if (definition.postprocessor.eos_token_id >= definition.vocabulary.size()) {
                return failure(TokenProgramError::InvalidTokenId, "V2 EOS token ID is outside the vocabulary");
            }
            if ((definition.vocabulary[definition.postprocessor.eos_token_id].flags &
                 static_cast<uint16_t>(VocabFlags::Special)) == 0) {
                return failure(TokenProgramError::InvalidTokenId, "V2 EOS token ID is not a declared special token");
            }
        } else if (definition.postprocessor.eos_token_id != kTokenProgramNoTokenId) {
            return failure(TokenProgramError::InvalidParameter, "unused V2 EOS token ID is not canonical");
        }
        break;
    default:
        return failure(TokenProgramError::UnknownRequiredOperation, "unknown required V2 postprocessor operation");
    }

    if (definition.decoder.kind != DecoderKind::ByteLevel && definition.decoder.kind != DecoderKind::Identity) {
        return failure(TokenProgramError::UnknownRequiredOperation, "unknown required V2 decoder operation");
    }
    if ((definition.decoder.flags & ~static_cast<uint8_t>(DecoderFlags::SkipSpecial)) != 0 ||
        definition.decoder.parameter != 0 || definition.decoder.argument != 0) {
        return failure(TokenProgramError::InvalidParameter, "V2 decoder parameters are not canonical");
    }

    if (definition.merges.size() > token_program_limits::kMaxMerges) {
        return failure(TokenProgramError::InvalidMerge, "V2 merge table is too large");
    }
    std::unordered_set<uint64_t> merge_pairs;
    merge_pairs.reserve(definition.merges.size());
    const uint16_t special = static_cast<uint16_t>(VocabFlags::Special);
    uint32_t previous_rank = 0;
    bool first_rank = true;
    for (size_t index = 0; index != definition.merges.size(); ++index) {
        const MergeRecord& merge = definition.merges[index];
        if ((!first_rank && merge.rank <= previous_rank) || merge.left_id >= definition.vocabulary.size() ||
            merge.right_id >= definition.vocabulary.size() || merge.result_id >= definition.vocabulary.size()) {
            return failure(TokenProgramError::InvalidMerge, "V2 merge rank or token ID is invalid", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        const VocabEntry& left = definition.vocabulary[merge.left_id];
        const VocabEntry& right = definition.vocabulary[merge.right_id];
        const VocabEntry& result = definition.vocabulary[merge.result_id];
        if ((left.flags & special) != 0 || (right.flags & special) != 0 || (result.flags & special) != 0 ||
            left.piece.size() > token_program_limits::kMaxPieceBytes - right.piece.size() ||
            result.piece != left.piece + right.piece) {
            return failure(TokenProgramError::InvalidMerge, "V2 merge pieces do not concatenate exactly", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if (!merge_pairs.emplace(pair_key(merge.left_id, merge.right_id)).second) {
            return failure(TokenProgramError::DuplicateRecord, "duplicate V2 merge pair", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        previous_rank = merge.rank;
        first_rank = false;
    }

    if (definition.added_tokens.size() > token_program_limits::kMaxAddedTokens) {
        return failure(TokenProgramError::InvalidAddedToken, "V2 added-token table is too large");
    }
    std::unordered_set<uint32_t> added_ids;
    std::unordered_set<std::string> added_matches;
    for (size_t index = 0; index != definition.added_tokens.size(); ++index) {
        const AddedTokenRecord& added = definition.added_tokens[index];
        if (added.token_id >= definition.vocabulary.size() || added.match.empty() ||
            added.match.size() > token_program_limits::kMaxPieceBytes || added.normalized.empty() ||
            added.normalized.size() > token_program_limits::kMaxPieceBytes) {
            return failure(TokenProgramError::InvalidAddedToken, "V2 added-token record is malformed", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if ((added.flags & ~kKnownAddedFlags) != 0 || !valid_utf8(added.match) || !valid_utf8(added.normalized)) {
            return !valid_utf8(added.match) || !valid_utf8(added.normalized)
                       ? failure(TokenProgramError::InvalidUtf8, "V2 added-token text is not UTF-8", SIZE_MAX,
                                 static_cast<uint32_t>(index))
                       : failure(TokenProgramError::InvalidAddedToken, "V2 added-token flags are unsupported", SIZE_MAX,
                                 static_cast<uint32_t>(index));
        }
        if ((added.flags & static_cast<uint16_t>(AddedTokenFlags::Normalized)) == 0 &&
            added.match != added.normalized) {
            return failure(TokenProgramError::InvalidAddedToken,
                           "non-normalized V2 added token must have equal match forms", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if ((added.flags & static_cast<uint16_t>(AddedTokenFlags::Normalized)) != 0) {
            std::string expected;
            const TokenProgramStatus status = normalize_text(definition.normalizer, added.match, expected);
            if (!status.ok() || expected != added.normalized) {
                return failure(TokenProgramError::InvalidAddedToken,
                               "normalized V2 added token does not match its normalizer", SIZE_MAX,
                               static_cast<uint32_t>(index));
            }
        }
        const bool added_special = (added.flags & static_cast<uint16_t>(AddedTokenFlags::Special)) != 0;
        const bool vocabulary_special = (definition.vocabulary[added.token_id].flags & special) != 0;
        if (added_special != vocabulary_special) {
            return failure(TokenProgramError::InvalidAddedToken,
                           "V2 added-token and vocabulary special flags disagree", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        if (!added_ids.emplace(added.token_id).second || !added_matches.emplace(added.match).second) {
            return failure(TokenProgramError::DuplicateRecord, "duplicate V2 added-token record", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
    }

    if (definition.prompt_max_bytes == 0 || definition.prompt_max_bytes > token_program_limits::kMaxPromptBytes) {
        return failure(TokenProgramError::InvalidPrompt, "V2 prompt bound is invalid");
    }
    if (definition.prompt.size() > token_program_limits::kMaxPromptInstructions) {
        return failure(TokenProgramError::InvalidPrompt, "V2 prompt program is too large");
    }
    if (!definition.prompt.empty()) {
        size_t literal_bytes = 0;
        size_t user_count = 0;
        size_t generation_count = 0;
        for (size_t index = 0; index != definition.prompt.size(); ++index) {
            const PromptInstruction& instruction = definition.prompt[index];
            switch (instruction.opcode) {
            case PromptOpcode::EmitLiteralUtf8:
                if (instruction.literal.empty() || !valid_utf8(instruction.literal)) {
                    return failure(TokenProgramError::InvalidUtf8, "V2 prompt literal is invalid UTF-8", SIZE_MAX,
                                   static_cast<uint32_t>(index));
                }
                break;
            case PromptOpcode::EmitUserText:
                if (!instruction.literal.empty()) {
                    return failure(TokenProgramError::InvalidPrompt, "V2 user-text instruction has a literal", SIZE_MAX,
                                   static_cast<uint32_t>(index));
                }
                ++user_count;
                break;
            case PromptOpcode::EmitGenerationPrompt:
                if (instruction.literal.empty() || !valid_utf8(instruction.literal)) {
                    return failure(TokenProgramError::InvalidPrompt,
                                   "V2 generation instruction needs valid UTF-8", SIZE_MAX,
                                   static_cast<uint32_t>(index));
                }
                ++generation_count;
                break;
            case PromptOpcode::End:
                if (index + 1 != definition.prompt.size() || !instruction.literal.empty()) {
                    return failure(TokenProgramError::InvalidPrompt, "V2 prompt end is not final", SIZE_MAX,
                                   static_cast<uint32_t>(index));
                }
                break;
            default:
                return failure(TokenProgramError::UnknownRequiredOperation, "unknown required V2 prompt operation",
                               SIZE_MAX, static_cast<uint32_t>(index));
            }
            if (instruction.literal.size() > token_program_limits::kMaxPromptLiteralBytes ||
                literal_bytes > token_program_limits::kMaxPromptLiteralBytes - instruction.literal.size()) {
                return failure(TokenProgramError::InvalidPrompt, "V2 prompt literal pool is too large", SIZE_MAX,
                               static_cast<uint32_t>(index));
            }
            literal_bytes += instruction.literal.size();
        }
        if (definition.prompt.back().opcode != PromptOpcode::End || user_count != 1 || generation_count != 1) {
            return failure(TokenProgramError::InvalidPrompt,
                           "V2 prompt needs one user-text and one generation instruction");
        }
    }
    return {};
}

void append_v2_section(std::vector<uint8_t>& output, TokenProgramV2Section section,
                       std::span<const uint8_t> payload, bool required = true) {
    append_u16(output, static_cast<uint16_t>(section));
    append_u16(output, required ? kV2SectionRequired : 0);
    append_u32(output, static_cast<uint32_t>(payload.size()));
    output.insert(output.end(), payload.begin(), payload.end());
}

std::vector<uint8_t> make_v2_options(const TokenProgramDefinition& definition) {
    std::vector<uint8_t> output;
    append_u32(output, definition.unknown_token_id);
    append_u32(output, definition.prompt_max_bytes);
    append_u16(output, definition.bpe_flags);
    append_u16(output, 0);
    return output;
}

std::vector<uint8_t> make_v2_map(const TokenProgramDefinition& definition) {
    std::vector<uint8_t> output;
    output.reserve(256 * 4);
    for (const uint32_t value : definition.byte_to_unicode) append_u32(output, value);
    return output;
}

std::vector<uint8_t> make_v2_normalizer(const NormalizerSpec& spec) {
    std::vector<uint8_t> output;
    append_spec(output, spec);
    return output;
}

std::vector<uint8_t> make_v2_pretokenizer(const PretokenizerSpec& spec) {
    std::vector<uint8_t> output;
    append_spec(output, spec);
    return output;
}

std::vector<uint8_t> make_v2_vocabulary(const std::vector<VocabEntry>& vocabulary) {
    std::vector<uint8_t> output;
    append_u32(output, static_cast<uint32_t>(vocabulary.size()));
    for (const VocabEntry& entry : vocabulary) {
        append_u32(output, static_cast<uint32_t>(entry.piece.size()));
        append_u16(output, entry.flags);
        append_u16(output, entry.priority);
        output.insert(output.end(), entry.piece.begin(), entry.piece.end());
    }
    return output;
}

std::vector<uint8_t> make_v2_bpe(const std::vector<MergeRecord>& merges) {
    std::vector<uint8_t> output;
    append_u32(output, static_cast<uint32_t>(merges.size()));
    for (const MergeRecord& merge : merges) {
        append_u32(output, merge.left_id);
        append_u32(output, merge.right_id);
        append_u32(output, merge.result_id);
        append_u32(output, merge.rank);
    }
    return output;
}

std::vector<uint8_t> make_v2_added(const std::vector<AddedTokenRecord>& added_tokens) {
    std::vector<uint8_t> output;
    append_u32(output, static_cast<uint32_t>(added_tokens.size()));
    for (const AddedTokenRecord& added : added_tokens) {
        append_u32(output, added.token_id);
        append_u16(output, added.flags);
        append_u16(output, added.priority);
        append_u32(output, static_cast<uint32_t>(added.match.size()));
        append_u32(output, static_cast<uint32_t>(added.normalized.size()));
        output.insert(output.end(), added.match.begin(), added.match.end());
        output.insert(output.end(), added.normalized.begin(), added.normalized.end());
    }
    return output;
}

std::vector<uint8_t> make_v2_postprocessor(const PostprocessorSpec& spec) {
    std::vector<uint8_t> output;
    append_spec(output, spec);
    return output;
}

std::vector<uint8_t> make_v2_decoder(const DecoderSpec& spec) {
    std::vector<uint8_t> output;
    append_spec(output, spec);
    return output;
}

std::vector<uint8_t> make_v2_prompt(const std::vector<PromptInstruction>& prompt) {
    std::vector<uint8_t> output;
    append_u32(output, static_cast<uint32_t>(prompt.size()));
    for (const PromptInstruction& instruction : prompt) {
        output.push_back(static_cast<uint8_t>(instruction.opcode));
        output.push_back(0);
        append_u16(output, 0);
        append_u32(output, 0);
        append_u32(output, static_cast<uint32_t>(instruction.literal.size()));
        output.insert(output.end(), instruction.literal.begin(), instruction.literal.end());
    }
    return output;
}

bool read_v2_spec(Reader& reader, NormalizerSpec& spec) { return read_normalizer(reader, spec); }
bool read_v2_spec(Reader& reader, PretokenizerSpec& spec) { return read_pretokenizer(reader, spec); }
bool read_v2_spec(Reader& reader, PostprocessorSpec& spec) { return read_postprocessor(reader, spec); }
bool read_v2_spec(Reader& reader, DecoderSpec& spec) { return read_decoder(reader, spec); }

bool section_is(TokenProgramV2Section actual, TokenProgramV2Section expected) {
    return static_cast<uint16_t>(actual) == static_cast<uint16_t>(expected);
}

} // namespace

TokenProgram::SerializeResult serialize_token_program_v2(const TokenProgramDefinition& definition) {
    try {
        const TokenProgramStatus valid = validate_v2_definition(definition);
        if (!valid.ok()) return valid;

        const std::vector<uint8_t> options = make_v2_options(definition);
        const std::vector<uint8_t> map = make_v2_map(definition);
        const std::vector<uint8_t> normalizer = make_v2_normalizer(definition.normalizer);
        const std::vector<uint8_t> vocabulary = make_v2_vocabulary(definition.vocabulary);
        const std::vector<uint8_t> pretokenizer = make_v2_pretokenizer(definition.pretokenizer);
        const std::vector<uint8_t> bpe = make_v2_bpe(definition.merges);
        const std::vector<uint8_t> added = make_v2_added(definition.added_tokens);
        const std::vector<uint8_t> postprocessor = make_v2_postprocessor(definition.postprocessor);
        const std::vector<uint8_t> decoder = make_v2_decoder(definition.decoder);
        const std::vector<uint8_t> prompt = make_v2_prompt(definition.prompt);
        std::vector<uint8_t> output;
        output.insert(output.end(), kV2Magic.begin(), kV2Magic.end());
        append_u16(output, kTokenProgramV2MajorVersion);
        append_u16(output, kTokenProgramV2MinorVersion);
        append_u16(output, kV2HeaderFlags);
        append_u16(output, kV2Reserved);
        const uint16_t section_count = static_cast<uint16_t>(definition.prompt.empty() ? 9 : 10);
        append_u16(output, section_count);
        append_v2_section(output, TokenProgramV2Section::Options, options);
        append_v2_section(output, TokenProgramV2Section::ByteToUnicode, map);
        append_v2_section(output, TokenProgramV2Section::Normalizer, normalizer);
        append_v2_section(output, TokenProgramV2Section::Vocabulary, vocabulary);
        append_v2_section(output, TokenProgramV2Section::Pretokenizer, pretokenizer);
        append_v2_section(output, TokenProgramV2Section::Bpe, bpe);
        append_v2_section(output, TokenProgramV2Section::AddedTokens, added);
        append_v2_section(output, TokenProgramV2Section::Postprocessor, postprocessor);
        append_v2_section(output, TokenProgramV2Section::Decoder, decoder);
        if (!definition.prompt.empty()) append_v2_section(output, TokenProgramV2Section::Prompt, prompt);
        if (output.size() > token_program_limits::kMaxPayloadBytes) {
            return failure(TokenProgramError::OutputTooLarge, "V2 tokenizer program is too large");
        }
        return output;
    } catch (const std::bad_alloc&) {
        return failure(TokenProgramError::OutputTooLarge, "V2 tokenizer program allocation failed");
    }
}

TokenProgram::CompileResult TokenProgram::compile_v2(std::span<const uint8_t> payload) {
    try {
        if (payload.size() > token_program_limits::kMaxPayloadBytes) {
            return failure(TokenProgramError::PayloadMalformed, "V2 tokenizer program payload is too large");
        }
        Reader reader(payload);
        std::span<const uint8_t> magic;
        if (!reader.take(kV2Magic.size(), magic) || !std::equal(magic.begin(), magic.end(), kV2Magic.begin())) {
            return failure(TokenProgramError::PayloadMalformed, "V2 tokenizer program magic is invalid", reader.offset());
        }
        uint16_t major = 0;
        uint16_t minor = 0;
        uint16_t header_flags = 0;
        uint16_t reserved = 0;
        uint16_t section_count = 0;
        if (!reader.u16(major) || !reader.u16(minor) || !reader.u16(header_flags) || !reader.u16(reserved) ||
            !reader.u16(section_count)) {
            return read_failure(reader, "V2 tokenizer program header is truncated");
        }
        if (major != kTokenProgramV2MajorVersion || minor != kTokenProgramV2MinorVersion) {
            return failure(TokenProgramError::UnsupportedVersion, "V2 tokenizer program version is unsupported", 10);
        }
        if (header_flags != kV2HeaderFlags || reserved != kV2Reserved || section_count == 0 ||
            section_count > kV2MaxSections) {
            return failure(TokenProgramError::InvalidParameter, "V2 tokenizer program header is not canonical",
                           reader.offset());
        }
        struct Section {
            uint16_t kind = 0;
            uint16_t flags = 0;
            std::span<const uint8_t> payload;
            size_t offset = 0;
        };
        std::vector<Section> sections;
        sections.reserve(section_count);
        uint16_t previous_kind = 0;
    std::array<bool, 11> seen{};
        for (uint16_t index = 0; index != section_count; ++index) {
            Section section;
            uint32_t length = 0;
            section.offset = reader.offset();
            if (!reader.u16(section.kind) || !reader.u16(section.flags) || !reader.u32(length)) {
                return read_failure(reader, "V2 section header is truncated");
            }
            if (section.kind <= previous_kind || section.flags & ~kKnownV2SectionFlags ||
                (length > reader.remaining()) || !reader.take(length, section.payload)) {
                return failure(TokenProgramError::PayloadMalformed, "V2 section is malformed", section.offset);
            }
            previous_kind = section.kind;
            if (section.kind < seen.size()) {
                if (seen[section.kind]) {
                    return failure(TokenProgramError::DuplicateRecord, "duplicate V2 section", section.offset);
                }
                seen[section.kind] = true;
            } else if ((section.flags & kV2SectionRequired) != 0) {
                return failure(TokenProgramError::UnknownRequiredSection, "unknown required V2 section", section.offset);
            }
            sections.push_back(section);
        }
        if (reader.remaining() != 0) {
            return failure(TokenProgramError::TrailingBytes, "V2 tokenizer program has trailing bytes", reader.offset());
        }

        const auto find_section = [&](TokenProgramV2Section kind) -> const Section* {
            for (const Section& section : sections) {
                if (section_is(static_cast<TokenProgramV2Section>(section.kind), kind)) return &section;
            }
            return nullptr;
        };
        const auto require_section = [&](TokenProgramV2Section kind) -> const Section* {
            const Section* section = find_section(kind);
            return section != nullptr && (section->flags & kV2SectionRequired) != 0 ? section : nullptr;
        };
        const Section* options_section = require_section(TokenProgramV2Section::Options);
        const Section* map_section = require_section(TokenProgramV2Section::ByteToUnicode);
        const Section* normalizer_section = require_section(TokenProgramV2Section::Normalizer);
        const Section* vocabulary_section = require_section(TokenProgramV2Section::Vocabulary);
        const Section* pretokenizer_section = require_section(TokenProgramV2Section::Pretokenizer);
        const Section* bpe_section = require_section(TokenProgramV2Section::Bpe);
        const Section* added_section = require_section(TokenProgramV2Section::AddedTokens);
        const Section* postprocessor_section = require_section(TokenProgramV2Section::Postprocessor);
        const Section* decoder_section = require_section(TokenProgramV2Section::Decoder);
        if (options_section == nullptr || map_section == nullptr || normalizer_section == nullptr ||
            vocabulary_section == nullptr ||
            pretokenizer_section == nullptr || bpe_section == nullptr || added_section == nullptr ||
            postprocessor_section == nullptr || decoder_section == nullptr) {
            return failure(TokenProgramError::UnknownRequiredSection, "required V2 section is missing");
        }

        TokenProgramDefinition definition;
        {
            Reader section(options_section->payload);
            uint16_t flags = 0;
            uint16_t section_reserved = 0;
            if (!section.u32(definition.unknown_token_id) || !section.u32(definition.prompt_max_bytes) ||
                !section.u16(flags) || !section.u16(section_reserved) || section.remaining() != 0) {
                return failure(TokenProgramError::PayloadMalformed, "V2 options section is malformed",
                               options_section->offset);
            }
            definition.bpe_flags = flags;
            if (section_reserved != 0) {
                return failure(TokenProgramError::InvalidParameter, "V2 options section is not canonical",
                               options_section->offset);
            }
        }
        {
            if (map_section->payload.size() != definition.byte_to_unicode.size() * sizeof(uint32_t)) {
                return failure(TokenProgramError::PayloadMalformed, "V2 scalar map section has wrong length",
                               map_section->offset);
            }
            Reader section(map_section->payload);
            for (uint32_t& value : definition.byte_to_unicode) {
                if (!section.u32(value)) return read_failure(section, "V2 scalar map is truncated");
            }
        }
        {
            Reader section(normalizer_section->payload);
            if (!read_v2_spec(section, definition.normalizer) || section.remaining() != 0) {
                return failure(TokenProgramError::PayloadMalformed, "V2 normalizer section is malformed",
                               normalizer_section->offset);
            }
        }
        {
            Reader section(vocabulary_section->payload);
            uint32_t count = 0;
            if (!section.u32(count) || count == 0 || count > token_program_limits::kMaxVocabulary) {
                return failure(TokenProgramError::PayloadMalformed, "V2 vocabulary count is outside its bound",
                               vocabulary_section->offset);
            }
            definition.vocabulary.reserve(count);
            for (uint32_t index = 0; index != count; ++index) {
                uint32_t length = 0;
                uint16_t flags = 0;
                uint16_t priority = 0;
                if (!section.u32(length) || !section.u16(flags) || !section.u16(priority) ||
                    length == 0 || length > token_program_limits::kMaxPieceBytes || length > section.remaining()) {
                    return failure(TokenProgramError::PayloadMalformed, "V2 vocabulary record is malformed",
                                   vocabulary_section->offset, index);
                }
                std::span<const uint8_t> piece;
                section.take(length, piece);
                definition.vocabulary.push_back({
                    std::string(reinterpret_cast<const char*>(piece.data()), piece.size()), flags, priority});
            }
            if (section.remaining() != 0) {
                return failure(TokenProgramError::TrailingBytes, "V2 vocabulary section has trailing bytes",
                               vocabulary_section->offset);
            }
        }
        {
            Reader section(pretokenizer_section->payload);
            if (!read_v2_spec(section, definition.pretokenizer) || section.remaining() != 0) {
                return failure(TokenProgramError::PayloadMalformed, "V2 pretokenizer section is malformed",
                               pretokenizer_section->offset);
            }
        }
        {
            Reader section(bpe_section->payload);
            uint32_t count = 0;
            if (!section.u32(count) || count > token_program_limits::kMaxMerges || count > section.remaining() / 16) {
                return failure(TokenProgramError::PayloadMalformed, "V2 BPE section count is outside its bound",
                               bpe_section->offset);
            }
            definition.merges.reserve(count);
            for (uint32_t index = 0; index != count; ++index) {
                MergeRecord merge;
                if (!section.u32(merge.left_id) || !section.u32(merge.right_id) || !section.u32(merge.result_id) ||
                    !section.u32(merge.rank)) {
                    return read_failure(section, "V2 BPE record is truncated");
                }
                definition.merges.push_back(merge);
            }
            if (section.remaining() != 0) {
                return failure(TokenProgramError::TrailingBytes, "V2 BPE section has trailing bytes", bpe_section->offset);
            }
        }
        {
            Reader section(added_section->payload);
            uint32_t count = 0;
            if (!section.u32(count) || count > token_program_limits::kMaxAddedTokens) {
                return failure(TokenProgramError::PayloadMalformed, "V2 added-token count is outside its bound",
                               added_section->offset);
            }
            definition.added_tokens.reserve(count);
            for (uint32_t index = 0; index != count; ++index) {
                AddedTokenRecord added;
                uint32_t match_length = 0;
                uint32_t normalized_length = 0;
                if (!section.u32(added.token_id) || !section.u16(added.flags) || !section.u16(added.priority) ||
                    !section.u32(match_length) || !section.u32(normalized_length) || match_length == 0 ||
                    normalized_length == 0 || match_length > token_program_limits::kMaxPieceBytes ||
                    normalized_length > token_program_limits::kMaxPieceBytes || match_length > section.remaining() ||
                    normalized_length > section.remaining() - match_length) {
                    return failure(TokenProgramError::PayloadMalformed, "V2 added-token record is malformed",
                                   added_section->offset, index);
                }
                std::span<const uint8_t> match;
                std::span<const uint8_t> normalized;
                section.take(match_length, match);
                section.take(normalized_length, normalized);
                added.match.assign(reinterpret_cast<const char*>(match.data()), match.size());
                added.normalized.assign(reinterpret_cast<const char*>(normalized.data()), normalized.size());
                definition.added_tokens.push_back(std::move(added));
            }
            if (section.remaining() != 0) {
                return failure(TokenProgramError::TrailingBytes, "V2 added-token section has trailing bytes",
                               added_section->offset);
            }
        }
        {
            Reader section(postprocessor_section->payload);
            if (!read_v2_spec(section, definition.postprocessor) || section.remaining() != 0) {
                return failure(TokenProgramError::PayloadMalformed, "V2 postprocessor section is malformed",
                               postprocessor_section->offset);
            }
        }
        {
            Reader section(decoder_section->payload);
            if (!read_v2_spec(section, definition.decoder) || section.remaining() != 0) {
                return failure(TokenProgramError::PayloadMalformed, "V2 decoder section is malformed",
                               decoder_section->offset);
            }
        }
        if (const Section* prompt_section = find_section(TokenProgramV2Section::Prompt); prompt_section != nullptr) {
            Reader section(prompt_section->payload);
            uint32_t count = 0;
            if ((prompt_section->flags & kV2SectionRequired) == 0 || !section.u32(count) || count == 0 ||
                count > token_program_limits::kMaxPromptInstructions) {
                return failure(TokenProgramError::PayloadMalformed, "V2 prompt section is malformed",
                               prompt_section->offset);
            }
            definition.prompt.reserve(count);
            for (uint32_t index = 0; index != count; ++index) {
                uint8_t opcode = 0;
                uint8_t instruction_flags = 0;
                uint16_t instruction_reserved = 0;
                uint32_t argument = 0;
                uint32_t length = 0;
                if (!section.u8(opcode) || !section.u8(instruction_flags) || !section.u16(instruction_reserved) ||
                    !section.u32(argument) || !section.u32(length) || instruction_flags != 0 ||
                    instruction_reserved != 0 || argument != 0 || length > section.remaining() ||
                    length > token_program_limits::kMaxPromptLiteralBytes) {
                    return failure(TokenProgramError::PayloadMalformed, "V2 prompt instruction is malformed",
                                   prompt_section->offset, index);
                }
                std::span<const uint8_t> literal;
                section.take(length, literal);
                definition.prompt.push_back({
                    static_cast<PromptOpcode>(opcode),
                    std::string(reinterpret_cast<const char*>(literal.data()), literal.size())});
            }
            if (section.remaining() != 0) {
                return failure(TokenProgramError::TrailingBytes, "V2 prompt section has trailing bytes",
                               prompt_section->offset);
            }
        }
        const TokenProgramStatus valid = validate_v2_definition(definition);
        if (!valid.ok()) return valid;
        TokenProgram program(std::move(definition));
        program.wire_major_ = kTokenProgramV2MajorVersion;
        return program;
    } catch (const std::bad_alloc&) {
        return failure(TokenProgramError::PayloadMalformed, "V2 tokenizer program allocation failed");
    }
}

TokenProgram::CompileResult compile_token_program_v2(std::span<const uint8_t> payload) {
    return TokenProgram::compile_v2(payload);
}

namespace {

TokenProgram::EncodeResult encode_v2(const TokenProgram& program, std::string_view text) {
    const TokenProgramDefinition& definition = program.definition();
    if (text.size() > token_program_limits::kMaxInputBytes) {
        return failure(TokenProgramError::InputTooLarge, "V2 input text is too large");
    }
    std::string normalized;
    const TokenProgramStatus normalized_status = normalize_text(definition.normalizer, text, normalized);
    if (!normalized_status.ok()) return normalized_status;
    std::vector<ScalarSpan> normalized_scalars;
    if (!collect_scalars(normalized, normalized_scalars)) {
        return failure(TokenProgramError::InvalidUtf8, "V2 input text is not UTF-8");
    }
    std::vector<uint32_t> output;
    output.reserve(text.size() + 2);
    if (definition.postprocessor.kind == PostprocessorKind::AddBosEos &&
        (definition.postprocessor.flags & static_cast<uint8_t>(PostprocessorFlags::AddBos)) != 0) {
        output.push_back(definition.postprocessor.bos_token_id);
    }

    std::unordered_map<std::string, uint32_t> piece_ids;
    piece_ids.reserve(definition.vocabulary.size());
    for (size_t index = 0; index != definition.vocabulary.size(); ++index) {
        piece_ids.emplace(definition.vocabulary[index].piece, static_cast<uint32_t>(index));
    }
    std::unordered_map<uint64_t, uint32_t> merge_indices;
    merge_indices.reserve(definition.merges.size());
    for (size_t index = 0; index != definition.merges.size(); ++index) {
        merge_indices.emplace(pair_key(definition.merges[index].left_id, definition.merges[index].right_id),
                              static_cast<uint32_t>(index));
    }

    const auto append_merged = [&](const std::vector<uint32_t>& input) {
        if (input.empty()) return;
        struct Node {
            uint32_t token = kTokenProgramNoTokenId;
            uint32_t previous = UINT32_MAX;
            uint32_t next = UINT32_MAX;
            uint32_t generation = 0;
            bool live = true;
        };
        struct Candidate {
            uint32_t rank = UINT32_MAX;
            uint32_t left = UINT32_MAX;
            uint32_t right = UINT32_MAX;
            uint32_t left_generation = 0;
            uint32_t right_generation = 0;
            uint32_t result = kTokenProgramNoTokenId;
        };
        struct LaterCandidate {
            bool operator()(const Candidate& left, const Candidate& right) const {
                if (left.rank != right.rank) return left.rank > right.rank;
                return left.left > right.left;
            }
        };
        std::vector<Node> nodes(input.size());
        for (size_t index = 0; index != input.size(); ++index) {
            nodes[index].token = input[index];
            nodes[index].previous = index == 0 ? UINT32_MAX : static_cast<uint32_t>(index - 1);
            nodes[index].next = index + 1 == input.size() ? UINT32_MAX : static_cast<uint32_t>(index + 1);
        }
        std::priority_queue<Candidate, std::vector<Candidate>, LaterCandidate> candidates;
        const auto add_candidate = [&](uint32_t left) {
            if (left == UINT32_MAX || !nodes[left].live || nodes[left].next == UINT32_MAX) return;
            const uint32_t right = nodes[left].next;
            const auto found = merge_indices.find(pair_key(nodes[left].token, nodes[right].token));
            if (found == merge_indices.end()) return;
            const MergeRecord& merge = definition.merges[found->second];
            candidates.push({merge.rank, left, right, nodes[left].generation, nodes[right].generation,
                             merge.result_id});
        };
        for (uint32_t index = 0; index + 1 < nodes.size(); ++index) add_candidate(index);
        while (!candidates.empty()) {
            const Candidate candidate = candidates.top();
            candidates.pop();
            Node& left = nodes[candidate.left];
            Node& right = nodes[candidate.right];
            if (!left.live || !right.live || left.next != candidate.right ||
                left.generation != candidate.left_generation || right.generation != candidate.right_generation) {
                continue;
            }
            left.token = candidate.result;
            ++left.generation;
            right.live = false;
            ++right.generation;
            left.next = right.next;
            if (right.next != UINT32_MAX) nodes[right.next].previous = candidate.left;
            add_candidate(left.previous);
            add_candidate(candidate.left);
        }
        for (uint32_t index = 0; index != UINT32_MAX; index = nodes[index].next) output.push_back(nodes[index].token);
    };

    const bool fuse_unknown = (definition.bpe_flags & static_cast<uint16_t>(BpeFlags::FuseUnknown)) != 0;
    const bool prefix_space = (definition.pretokenizer.flags & static_cast<uint8_t>(PretokenizerFlags::AddPrefixSpace)) != 0;
    size_t segment_start = 0;
    size_t cursor = 0;
    bool first_regular = true;
    while (cursor < normalized.size()) {
        // Added-token matching is intentionally a metadata-driven leftmost,
        // then longest selection.  It is performed only at scalar boundaries.
        bool found_match = false;
        AddedMatch best;
        size_t scan_cursor = cursor;
        while (scan_cursor < normalized.size()) {
            uint32_t scalar = 0;
            size_t scalar_length = 0;
            if (!decode_scalar_at(text, scan_cursor, scalar, scalar_length)) {
                return failure(TokenProgramError::InvalidUtf8, "V2 input text is not UTF-8");
            }
            (void)scalar;
            for (const AddedTokenRecord& record : definition.added_tokens) {
                const bool use_normalized = (record.flags & static_cast<uint16_t>(AddedTokenFlags::Normalized)) != 0;
                const std::string_view source = use_normalized ? std::string_view(normalized) : text;
                const std::string_view needle = use_normalized ? std::string_view(record.normalized)
                                                                : std::string_view(record.match);
                if (scan_cursor > source.size() || needle.size() > source.size() - scan_cursor ||
                    source.compare(scan_cursor, needle.size(), needle) != 0) {
                    continue;
                }
                size_t start = scan_cursor;
                size_t end = scan_cursor + needle.size();
                if ((record.flags & static_cast<uint16_t>(AddedTokenFlags::LeftStrip)) != 0) {
                    while (start != 0) {
                        uint32_t previous = 0;
                        size_t previous_length = 0;
                        size_t previous_offset = start - 1;
                        while (previous_offset > 0 &&
                               (static_cast<uint8_t>(text[previous_offset]) & 0xc0) == 0x80) --previous_offset;
                        if (!decode_scalar_at(text, previous_offset, previous, previous_length) ||
                            !scalar_is_whitespace(previous)) break;
                        start = previous_offset;
                    }
                }
                if ((record.flags & static_cast<uint16_t>(AddedTokenFlags::RightStrip)) != 0) {
                    while (end < text.size()) {
                        uint32_t next = 0;
                        size_t next_length = 0;
                        if (!decode_scalar_at(text, end, next, next_length) || !scalar_is_whitespace(next)) break;
                        end += next_length;
                    }
                }
                if ((record.flags & static_cast<uint16_t>(AddedTokenFlags::SingleWord)) != 0) {
                    if (start != 0) {
                        size_t previous_offset = start - 1;
                        while (previous_offset > 0 &&
                               (static_cast<uint8_t>(text[previous_offset]) & 0xc0) == 0x80) --previous_offset;
                        uint32_t previous = 0;
                        size_t previous_length = 0;
                        if (decode_scalar_at(text, previous_offset, previous, previous_length) &&
                            scalar_is_word(previous)) continue;
                    }
                    if (end < text.size()) {
                        uint32_t next = 0;
                        size_t next_length = 0;
                        if (decode_scalar_at(text, end, next, next_length) && scalar_is_word(next)) continue;
                    }
                }
                AddedMatch candidate{&record, start, end};
                if (!found_match || candidate.start < best.start ||
                    (candidate.start == best.start &&
                     (candidate.end - candidate.start > best.end - best.start ||
                      (candidate.end - candidate.start == best.end - best.start &&
                       candidate.record->priority > best.record->priority)))) {
                    best = candidate;
                    found_match = true;
                }
            }
            scan_cursor += scalar_length;
        }
        if (!found_match) break;
        if (best.start < segment_start) {
            return failure(TokenProgramError::InvalidAddedToken, "V2 added-token strip overlaps a prior match");
        }
        const std::string_view gap(normalized.data() + segment_start, best.start - segment_start);
        const std::string canonical = (first_regular && prefix_space && !gap.empty() ? " " : "") + std::string(gap);
        const std::vector<std::string_view> segments = pretokenize_v2(canonical, definition.pretokenizer);
        for (const std::string_view segment : segments) {
            std::vector<uint32_t> symbols;
            bool previous_was_unknown = false;
            for (const uint8_t byte : std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(segment.data()), segment.size())) {
                const std::string symbol = scalar_utf8(definition.byte_to_unicode[byte]);
                const auto found = piece_ids.find(symbol);
                if (found != piece_ids.end()) {
                    symbols.push_back(found->second);
                    previous_was_unknown = false;
                } else {
                    if (definition.unknown_token_id == kTokenProgramNoTokenId) {
                        return failure(TokenProgramError::UnknownPiece, "V2 byte has no vocabulary entry");
                    }
                    if (!fuse_unknown || !previous_was_unknown) symbols.push_back(definition.unknown_token_id);
                    previous_was_unknown = true;
                }
            }
            append_merged(symbols);
        }
        first_regular = false;
        output.push_back(best.record->token_id);
        segment_start = best.end;
        cursor = best.end;
    }
    const std::string_view tail(normalized.data() + segment_start, normalized.size() - segment_start);
    const std::string canonical = (first_regular && prefix_space && !tail.empty() ? " " : "") + std::string(tail);
    const std::vector<std::string_view> segments = pretokenize_v2(canonical, definition.pretokenizer);
    for (const std::string_view segment : segments) {
        std::vector<uint32_t> symbols;
        bool previous_was_unknown = false;
        for (const uint8_t byte : std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(segment.data()), segment.size())) {
            const std::string symbol = scalar_utf8(definition.byte_to_unicode[byte]);
            const auto found = piece_ids.find(symbol);
            if (found != piece_ids.end()) {
                symbols.push_back(found->second);
                previous_was_unknown = false;
            } else {
                if (definition.unknown_token_id == kTokenProgramNoTokenId) {
                    return failure(TokenProgramError::UnknownPiece, "V2 byte has no vocabulary entry");
                }
                if (!fuse_unknown || !previous_was_unknown) symbols.push_back(definition.unknown_token_id);
                previous_was_unknown = true;
            }
        }
        append_merged(symbols);
    }
    if (definition.postprocessor.kind == PostprocessorKind::AddBosEos &&
        (definition.postprocessor.flags & static_cast<uint8_t>(PostprocessorFlags::AddEos)) != 0) {
        output.push_back(definition.postprocessor.eos_token_id);
    }
    return output;
}

TokenProgram::DecodeResult decode_v2(const TokenProgram& program, std::span<const uint32_t> token_ids) {
    const TokenProgramDefinition& definition = program.definition();
    if (token_ids.size() > token_program_limits::kMaxInputBytes) {
        return failure(TokenProgramError::InputTooLarge, "V2 token sequence is too large");
    }
    std::unordered_map<uint32_t, uint8_t> inverse;
    inverse.reserve(definition.byte_to_unicode.size());
    for (size_t index = 0; index != definition.byte_to_unicode.size(); ++index) {
        inverse.emplace(definition.byte_to_unicode[index], static_cast<uint8_t>(index));
    }
    const bool skip_special = (definition.decoder.flags & static_cast<uint8_t>(DecoderFlags::SkipSpecial)) != 0;
    std::string output;
    for (size_t index = 0; index != token_ids.size(); ++index) {
        const uint32_t token_id = token_ids[index];
        if (token_id >= definition.vocabulary.size()) {
            return failure(TokenProgramError::InvalidTokenId, "V2 decode token ID is outside the vocabulary", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        const VocabEntry& entry = definition.vocabulary[token_id];
        const bool special = (entry.flags & static_cast<uint16_t>(VocabFlags::Special)) != 0;
        if (special && skip_special) continue;
        if (special || definition.decoder.kind == DecoderKind::Identity) {
            if (output.size() > token_program_limits::kMaxInputBytes - entry.piece.size()) {
                return failure(TokenProgramError::OutputTooLarge, "V2 decoded text exceeds its bound", SIZE_MAX,
                               static_cast<uint32_t>(index));
            }
            output.append(entry.piece);
            continue;
        }
        std::vector<ScalarSpan> scalars;
        if (!collect_scalars(entry.piece, scalars)) {
            return failure(TokenProgramError::InvalidUtf8, "V2 vocabulary piece is not UTF-8", SIZE_MAX,
                           static_cast<uint32_t>(index));
        }
        for (const ScalarSpan scalar : scalars) {
            const auto found = inverse.find(scalar.value);
            if (found == inverse.end()) {
                return failure(TokenProgramError::UnknownPiece,
                               "V2 decoder encountered a scalar outside the inverse map", SIZE_MAX,
                               static_cast<uint32_t>(index));
            }
            if (output.size() == token_program_limits::kMaxInputBytes) {
                return failure(TokenProgramError::OutputTooLarge, "V2 decoded text exceeds its bound", SIZE_MAX,
                               static_cast<uint32_t>(index));
            }
            output.push_back(static_cast<char>(found->second));
        }
    }
    return output;
}

} // namespace

} // namespace Laplace
