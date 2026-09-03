#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include <unordered_map>
#include <utility>

#include "artifact_set.h"

namespace Laplace {

constexpr uint32_t kTokenProgramNoTokenId = UINT32_MAX;
inline constexpr std::string_view kTokenProgramMagic = "LAPTOKDAT1";
inline constexpr std::string_view kTokenProgramV2Magic = "LAPTOKDAT2";
inline constexpr std::string_view kTokenProgramV3Magic = "LAPTOKDAT3";

// V2 deliberately has a different wire identity.  A V1 payload is never
// interpreted as a V2 payload (or vice versa), which keeps the old fixed
// layout and its byte-level behavior stable.
inline constexpr uint16_t kTokenProgramV2MajorVersion = 2;
inline constexpr uint16_t kTokenProgramV2MinorVersion = 0;
inline constexpr uint16_t kTokenProgramV3MajorVersion = 3;
inline constexpr uint16_t kTokenProgramV3MinorVersion = 0;

enum class TokenProgramModelKind : uint8_t {
    ByteBpe = 1,
    SentencePiece = 2,
};

enum class TokenPieceType : uint8_t {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Byte = 6,
    Unused = 5,
};

enum class SentencePieceNormalizerFlags : uint8_t {
    None = 0,
    AddDummyPrefix = 1u << 0,
    RemoveExtraWhitespaces = 1u << 1,
    EscapeWhitespaces = 1u << 2,
    TreatWhitespaceAsSuffix = 1u << 3,
};

enum class VocabFlags : uint16_t {
    None = 0,
    Special = 1u << 0,
};

enum class AddedTokenFlags : uint16_t {
    None = 0,
    SingleWord = 1u << 0,
    LeftStrip = 1u << 1,
    RightStrip = 1u << 2,
    Normalized = 1u << 3,
    Special = 1u << 4,
};

enum class NormalizerKind : uint8_t {
    None = 1,
    AsciiLowercase = 2,
    SentencePiece = 3,
};

enum class PretokenizerKind : uint8_t {
    ByteLevel = 1,
    // Unicode-scalar scanner with deterministic GPT-2/Qwen-compatible
    // grouping rules.  The wire name is intentionally family-neutral.
    UnicodeScalarScanner = 2,
    ScalarScanner = UnicodeScalarScanner,
    SentencePiece = 3,
    Regex = 4,
};

enum class PretokenizerFlags : uint8_t {
    None = 0,
    AddPrefixSpace = 1u << 0,
    SplitAsciiWhitespace = 1u << 1,
    // Qwen-class sources group consecutive newlines (and horizontal space
    // before them) into one segment; the default follows GPT-2's
    // backtracking whitespace rule instead. Selected from package facts.
    GroupNewlineRuns = 1u << 2,
};

enum class BpeFlags : uint16_t {
    None = 0,
    FuseUnknown = 1u << 0,
};

enum class PostprocessorKind : uint8_t {
    None = 1,
    AddBosEos = 2,
};

enum class PostprocessorFlags : uint8_t {
    None = 0,
    AddBos = 1u << 0,
    AddEos = 1u << 1,
};

enum class DecoderKind : uint8_t {
    ByteLevel = 1,
    Identity = 2,
};

enum class DecoderFlags : uint8_t {
    None = 0,
    SkipSpecial = 1u << 0,
};

enum class PromptOpcode : uint8_t {
    EmitLiteralUtf8 = 1,
    EmitUserText = 2,
    EmitGenerationPrompt = 3,
    End = 255,
};

enum class TokenProgramError : uint16_t {
    None = 0,
    PayloadMalformed = 1,
    UnsupportedVersion = 2,
    TrailingBytes = 3,
    UnsupportedEnum = 4,
    InvalidParameter = 5,
    InvalidUtf8 = 6,
    DuplicateRecord = 7,
    InvalidTokenId = 8,
    VocabularyNotDense = 9,
    InvalidMerge = 10,
    InvalidAddedToken = 11,
    InvalidPrompt = 12,
    InputTooLarge = 13,
    PromptTooLarge = 14,
    UnknownPiece = 15,
    OutputTooLarge = 16,
    UnknownRequiredSection = 17,
    UnknownRequiredOperation = 18,
    DigestMismatch = 19,
};

enum class TokenProgramV2Section : uint16_t {
    Options = 1,
    ByteToUnicode = 2,
    Normalizer = 3,
    Vocabulary = 4,
    Pretokenizer = 5,
    Bpe = 6,
    AddedTokens = 7,
    Postprocessor = 8,
    Decoder = 9,
    Prompt = 10,
};

enum class TokenProgramV2SectionFlags : uint16_t {
    None = 0,
    Required = 1u << 0,
};

enum class TokenProgramV3Section : uint16_t {
    Options = 1,
    Normalizer = 2,
    Pretokenizer = 3,
    Vocabulary = 4,
    Merges = 5,
    AddedTokens = 6,
    Postprocessor = 7,
    Decoder = 8,
    Prompt = 9,
    ByteToUnicode = 10,
};

enum class TokenProgramV3SectionFlags : uint16_t {
    None = 0,
    Required = 1u << 0,
};

struct TokenProgramStatus {
    TokenProgramError error = TokenProgramError::None;
    size_t offset = SIZE_MAX;
    uint32_t index = UINT32_MAX;
    std::string detail;

    bool ok() const noexcept { return error == TokenProgramError::None; }
    explicit operator bool() const noexcept { return ok(); }
};

struct VocabEntry {
    std::string piece;
    uint16_t flags = 0;
    uint16_t priority = 0;
    uint8_t type = static_cast<uint8_t>(TokenPieceType::Normal);
    float score = 0.0f;
    friend bool operator==(const VocabEntry&, const VocabEntry&) = default;
};

struct MergeRecord {
    uint32_t left_id = kTokenProgramNoTokenId;
    uint32_t right_id = kTokenProgramNoTokenId;
    uint32_t result_id = kTokenProgramNoTokenId;
    uint32_t rank = 0;
    friend bool operator==(const MergeRecord&, const MergeRecord&) = default;
};

struct AddedTokenRecord {
    uint32_t token_id = kTokenProgramNoTokenId;
    uint16_t flags = 0;
    uint16_t priority = 0;
    std::string match;
    std::string normalized;
    friend bool operator==(const AddedTokenRecord&, const AddedTokenRecord&) = default;
};

struct NormalizerSpec {
    NormalizerKind kind = NormalizerKind::None;
    uint8_t flags = 0;
    uint16_t parameter = 0;
    uint32_t argument = 0;
    friend bool operator==(const NormalizerSpec&, const NormalizerSpec&) = default;
};

struct PretokenizerSpec {
    PretokenizerKind kind = PretokenizerKind::ByteLevel;
    uint8_t flags = 0;
    uint16_t parameter = 0;
    uint32_t argument = 0;
    friend bool operator==(const PretokenizerSpec&, const PretokenizerSpec&) = default;
};

struct PostprocessorSpec {
    PostprocessorKind kind = PostprocessorKind::None;
    uint8_t flags = 0;
    uint16_t parameter = 0;
    uint32_t bos_token_id = kTokenProgramNoTokenId;
    uint32_t eos_token_id = kTokenProgramNoTokenId;
    friend bool operator==(const PostprocessorSpec&, const PostprocessorSpec&) = default;
};

struct DecoderSpec {
    DecoderKind kind = DecoderKind::ByteLevel;
    uint8_t flags = 0;
    uint16_t parameter = 0;
    uint32_t argument = 0;
    friend bool operator==(const DecoderSpec&, const DecoderSpec&) = default;
};

struct PromptInstruction {
    PromptOpcode opcode = PromptOpcode::End;
    std::string literal;
    friend bool operator==(const PromptInstruction&, const PromptInstruction&) = default;
};

struct TokenProgramDefinition {
    std::array<uint8_t, 256> byte_map{};
    // V2's byte-level map stores Unicode scalar values rather than UTF-8
    // bytes.  V1 uses byte_map above unchanged.
    std::array<uint32_t, 256> byte_to_unicode{};
    std::vector<VocabEntry> vocabulary;
    std::vector<MergeRecord> merges;
    std::vector<AddedTokenRecord> added_tokens;
    NormalizerSpec normalizer;
    PretokenizerSpec pretokenizer;
    PostprocessorSpec postprocessor;
    DecoderSpec decoder;
    std::vector<PromptInstruction> prompt;
    uint32_t prompt_max_bytes = 64u * 1024u;
    uint32_t unknown_token_id = kTokenProgramNoTokenId;
    uint16_t bpe_flags = 0;

    // V3 typed tokenizer facts.  They are ignored by V1/V2 serializers.
    TokenProgramModelKind model_kind = TokenProgramModelKind::ByteBpe;
    bool byte_fallback = false;
    std::vector<uint32_t> stop_ids;
    std::vector<uint8_t> normalizer_data;
    std::string pretokenizer_regex;
    uint32_t stream_max_bytes = 16u * 1024u * 1024u;
    Sha256Digest vocabulary_digest{};
    Sha256Digest prompt_program_digest{};

    friend bool operator==(const TokenProgramDefinition&, const TokenProgramDefinition&) = default;
};

class TokenProgram {
public:
    using CompileResult = std::variant<TokenProgram, TokenProgramStatus>;
    using SerializeResult = std::variant<std::vector<uint8_t>, TokenProgramStatus>;
    using EncodeResult = std::variant<std::vector<uint32_t>, TokenProgramStatus>;
    using PromptResult = std::variant<std::string, TokenProgramStatus>;
    using DecodeResult = std::variant<std::string, TokenProgramStatus>;

    struct StreamState {
        std::string pending_utf8;
        size_t decoded_bytes = 0;
        bool finished = false;
    };

    static CompileResult compile(std::span<const uint8_t> payload);
    static CompileResult compile_v2(std::span<const uint8_t> payload);
    static CompileResult compile_v3(std::span<const uint8_t> payload);
    static CompileResult compile(const std::vector<uint8_t>& payload) {
        return compile(std::span<const uint8_t>(payload.data(), payload.size()));
    }
    // Package/session admission for an authoritative token-ID-only contract.
    // Text interpretation is intentionally unavailable at this boundary.
    static TokenProgram token_ids_only(uint32_t vocabulary_size);

    SerializeResult serialize() const;
    EncodeResult encode(std::string_view text) const;
    PromptResult render_prompt(std::string_view user_text) const;
    DecodeResult decode(std::span<const uint32_t> token_ids) const;
    DecodeResult decode_chunk(std::span<const uint32_t> token_ids, StreamState& state,
                              bool final_chunk = false) const;
    Sha256Digest vocabulary_digest() const;
    Sha256Digest prompt_digest() const;

    template <size_t N>
    DecodeResult decode(const std::array<uint32_t, N>& token_ids) const {
        return decode(std::span<const uint32_t>(token_ids.data(), token_ids.size()));
    }

    const TokenProgramDefinition& definition() const noexcept { return definition_; }
    bool is_v2() const noexcept { return wire_major_ == kTokenProgramV2MajorVersion; }
    bool is_v3() const noexcept { return wire_major_ == kTokenProgramV3MajorVersion; }
    uint16_t wire_major_version() const noexcept { return wire_major_; }
    bool token_ids_only() const noexcept { return token_ids_only_; }

private:
    explicit TokenProgram(TokenProgramDefinition definition);
    TokenProgramDefinition definition_;
    uint16_t wire_major_ = 1;
    bool token_ids_only_ = false;
    std::array<uint8_t, 256> inverse_byte_map_{};
    std::unordered_map<uint32_t, uint8_t> inverse_unicode_map_;
    std::unordered_map<std::string, uint32_t> piece_ids_;
    // Pair key to the index in the rank-ordered merge table.
    std::unordered_map<uint64_t, uint32_t> merge_indices_;
    std::array<std::vector<uint32_t>, 256> original_added_by_first_byte_;
    std::array<std::vector<uint32_t>, 256> normalized_added_by_first_byte_;
};

TokenProgram::SerializeResult serialize_token_program(const TokenProgramDefinition& definition);
TokenProgram::SerializeResult serialize_token_program_v2(const TokenProgramDefinition& definition);
TokenProgram::SerializeResult serialize_token_program_v3(const TokenProgramDefinition& definition);
TokenProgram::CompileResult compile_token_program(std::span<const uint8_t> payload);
TokenProgram::CompileResult compile_token_program_v2(std::span<const uint8_t> payload);
TokenProgram::CompileResult compile_token_program_v3(std::span<const uint8_t> payload);

using TokenizerProgram = TokenProgram;

namespace token_program_limits {
constexpr size_t kMaxPayloadBytes = 128u * 1024u * 1024u;
constexpr size_t kMaxInputBytes = 16u * 1024u * 1024u;
constexpr size_t kMaxVocabulary = 1u * 1000u * 1000u;
constexpr size_t kMaxPieceBytes = 1u * 1024u * 1024u;
constexpr size_t kMaxMerges = 4u * 1000u * 1000u;
constexpr size_t kMaxAddedTokens = 65536u;
constexpr size_t kMaxPromptInstructions = 256u;
constexpr size_t kMaxPromptLiteralBytes = 64u * 1024u;
constexpr size_t kMaxPromptBytes = 64u * 1024u;
constexpr size_t kMaxNormalizerDataBytes = 4u * 1024u * 1024u;
constexpr size_t kMaxPretokenizerBytes = 1u * 1024u * 1024u;
constexpr size_t kMaxStopIds = 1u * 1000u * 1000u;
};

} // namespace Laplace
