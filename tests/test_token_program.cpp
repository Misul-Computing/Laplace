#include "token_program.h"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "test_util.h"

using namespace Laplace;

namespace {

void u16(std::vector<uint8_t>& out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value));
    out.push_back(static_cast<uint8_t>(value >> 8));
}

void u32(std::vector<uint8_t>& out, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8) {
        out.push_back(static_cast<uint8_t>(value >> shift));
    }
}

void set_u32(std::vector<uint8_t>& out, size_t offset, uint32_t value) {
    CHECK(offset <= out.size() && out.size() - offset >= 4);
    if (offset > out.size() || out.size() - offset < 4) return;
    for (unsigned shift = 0; shift != 32; shift += 8) {
        out[offset + shift / 8] = static_cast<uint8_t>(value >> shift);
    }
}

void set_u16(std::vector<uint8_t>& out, size_t offset, uint16_t value) {
    CHECK(offset <= out.size() && out.size() - offset >= 2);
    if (offset > out.size() || out.size() - offset < 2) return;
    out[offset] = static_cast<uint8_t>(value);
    out[offset + 1] = static_cast<uint8_t>(value >> 8);
}

void bytes(std::vector<uint8_t>& out, std::string_view value) {
    out.insert(out.end(), value.begin(), value.end());
}

struct RawPayload {
    std::vector<uint8_t> bytes;
    size_t prompt_opcode_offset = 0;
    size_t merge_result_offset = 0;
};

RawPayload canonical_payload() {
    RawPayload raw;
    auto& out = raw.bytes;

    bytes(out, "LAPTOKDAT1");
    u16(out, 1); // major
    u16(out, 0); // minor
    u16(out, 0); // flags
    u32(out, 10); // vocabulary entries
    u32(out, 1); // merge entries
    u32(out, 1); // added-token entries
    u32(out, 4); // prompt instructions
    u32(out, 128); // rendered prompt byte bound
    u32(out, 0); // unknown token ID
    u32(out, kTokenProgramNoTokenId); // BOS
    u32(out, kTokenProgramNoTokenId); // EOS
    for (unsigned value = 0; value != 256; ++value) out.push_back(static_cast<uint8_t>(value));

    // Normalizer: none, with all exact parameters zero.
    out.push_back(static_cast<uint8_t>(NormalizerKind::None));
    out.push_back(0);
    u16(out, 0);
    u32(out, 0);
    // Pretokenizer: byte-level, no prefix and no splitting.
    out.push_back(static_cast<uint8_t>(PretokenizerKind::ByteLevel));
    out.push_back(0);
    u16(out, 0);
    u32(out, 0);
    // Postprocessor: no implicit tokens.
    out.push_back(static_cast<uint8_t>(PostprocessorKind::None));
    out.push_back(0);
    u16(out, 0);
    u32(out, kTokenProgramNoTokenId);
    u32(out, kTokenProgramNoTokenId);
    // Decoder: byte-level, no special-token suppression.
    out.push_back(static_cast<uint8_t>(DecoderKind::ByteLevel));
    out.push_back(0);
    u16(out, 0);
    u32(out, 0);

    auto vocab = [&](uint16_t flags, std::string_view piece) {
        u32(out, static_cast<uint32_t>(piece.size()));
        u16(out, flags);
        u16(out, 0);
        bytes(out, piece);
    };
    vocab(static_cast<uint16_t>(VocabFlags::Special), "<unk>");
    vocab(0, "a");
    vocab(0, "b");
    vocab(0, "ab");
    vocab(0, " ");
    vocab(0, "A");
    vocab(static_cast<uint16_t>(VocabFlags::Special), "<bos>");
    vocab(static_cast<uint16_t>(VocabFlags::Special), "<eos>");
    vocab(0, "x");
    vocab(static_cast<uint16_t>(VocabFlags::Special), "<x>");

    // Merge rank 0: a + b -> ab.
    u32(out, 1);
    u32(out, 2);
    raw.merge_result_offset = out.size();
    u32(out, 3);
    u32(out, 0);

    // Added token <x>, with a higher priority than ordinary vocabulary pieces.
    u32(out, 9);
    u16(out, static_cast<uint16_t>(AddedTokenFlags::Special));
    u16(out, 7);
    u32(out, 3);
    u32(out, 3);
    bytes(out, "<x>");
    bytes(out, "<x>");

    auto prompt = [&](PromptOpcode opcode, std::string_view literal = {}) {
        if (opcode == PromptOpcode::EmitUserText) raw.prompt_opcode_offset = out.size();
        out.push_back(static_cast<uint8_t>(opcode));
        out.push_back(0);
        u16(out, 0);
        u32(out, 0);
        u32(out, static_cast<uint32_t>(literal.size()));
        bytes(out, literal);
    };
    prompt(PromptOpcode::EmitLiteralUtf8, "<user>");
    prompt(PromptOpcode::EmitUserText);
    prompt(PromptOpcode::EmitGenerationPrompt, "<assistant>\n");
    prompt(PromptOpcode::End);
    return raw;
}

TokenProgramDefinition definition_for_serializer() {
    TokenProgramDefinition definition;
    for (unsigned value = 0; value != 256; ++value) {
        definition.byte_map[value] = static_cast<uint8_t>(value);
    }
    definition.unknown_token_id = 0;
    definition.vocabulary = {
        {"<unk>", static_cast<uint16_t>(VocabFlags::Special), 0},
        {"a", 0, 0},
        {"b", 0, 0},
        {"ab", 0, 0},
        {" ", 0, 0},
        {"A", 0, 0},
        {"<bos>", static_cast<uint16_t>(VocabFlags::Special), 0},
        {"<eos>", static_cast<uint16_t>(VocabFlags::Special), 0},
        {"x", 0, 0},
        {"<x>", static_cast<uint16_t>(VocabFlags::Special), 0},
    };
    definition.merges = {{1, 2, 3, 0}};
    definition.added_tokens = {{9, static_cast<uint16_t>(AddedTokenFlags::Special), 7, "<x>", "<x>"}};
    definition.prompt = {
        {PromptOpcode::EmitLiteralUtf8, "<user>"},
        {PromptOpcode::EmitUserText, {}},
        {PromptOpcode::EmitGenerationPrompt, "<assistant>\n"},
        {PromptOpcode::End, {}},
    };
    definition.prompt_max_bytes = 128;
    definition.normalizer.kind = NormalizerKind::None;
    definition.pretokenizer.kind = PretokenizerKind::ByteLevel;
    definition.postprocessor.kind = PostprocessorKind::None;
    definition.decoder.kind = DecoderKind::ByteLevel;
    return definition;
}

void check_independent_golden() {
    const RawPayload raw = canonical_payload();
    const auto result = TokenProgram::compile(raw.bytes);
    CHECK(std::holds_alternative<TokenProgram>(result));
    if (!std::holds_alternative<TokenProgram>(result)) return;
    const TokenProgram& program = std::get<TokenProgram>(result);
    const auto canonical = program.serialize();
    CHECK(std::holds_alternative<std::vector<uint8_t>>(canonical));
    if (const auto* bytes = std::get_if<std::vector<uint8_t>>(&canonical)) CHECK(*bytes == raw.bytes);

    const auto encoded = program.encode("ab");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(encoded));
    if (const auto* ids = std::get_if<std::vector<uint32_t>>(&encoded)) {
        CHECK(*ids == std::vector<uint32_t>({3}));
    }
    const auto added = program.encode("a<x>b");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(added));
    if (const auto* ids = std::get_if<std::vector<uint32_t>>(&added)) {
        CHECK(*ids == std::vector<uint32_t>({1, 9, 2}));
    }
    const auto unknown = program.encode("az");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(unknown));
    if (const auto* ids = std::get_if<std::vector<uint32_t>>(&unknown)) {
        CHECK(*ids == std::vector<uint32_t>({1, 0}));
    }
    const auto rendered = program.render_prompt("ab");
    CHECK(std::holds_alternative<std::string>(rendered));
    if (const auto* text = std::get_if<std::string>(&rendered)) {
        CHECK(*text == "<user>ab<assistant>\n");
    }
    const auto decoded = program.decode(std::array<uint32_t, 3>{3, 4, 9});
    CHECK(std::holds_alternative<std::string>(decoded));
    if (const auto* text = std::get_if<std::string>(&decoded)) CHECK(*text == "ab <x>");
}

void check_canonical_serializer_matches_hand_built() {
    const auto hand_built = canonical_payload().bytes;
    const auto definition = definition_for_serializer();
    const auto serialized = serialize_token_program(definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (const auto* bytes = std::get_if<std::vector<uint8_t>>(&serialized)) {
        CHECK(*bytes == hand_built);
    }
}

void check_component_parameters_and_postprocessor() {
    TokenProgramDefinition definition = definition_for_serializer();
    definition.postprocessor.kind = PostprocessorKind::AddBosEos;
    definition.postprocessor.flags = static_cast<uint8_t>(PostprocessorFlags::AddBos) |
                                     static_cast<uint8_t>(PostprocessorFlags::AddEos);
    definition.postprocessor.bos_token_id = 6;
    definition.postprocessor.eos_token_id = 7;
    const auto serialized = serialize_token_program(definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized)) return;
    const auto compiled = TokenProgram::compile(std::get<std::vector<uint8_t>>(serialized));
    CHECK(std::holds_alternative<TokenProgram>(compiled));
    if (!std::holds_alternative<TokenProgram>(compiled)) return;
    const auto ids = std::get<TokenProgram>(compiled).encode("ab");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(ids));
    if (const auto* output = std::get_if<std::vector<uint32_t>>(&ids)) {
        CHECK(*output == std::vector<uint32_t>({6, 3, 7}));
    }

    definition.postprocessor = {};
    definition.normalizer.kind = NormalizerKind::AsciiLowercase;
    const auto lowercase_payload = serialize_token_program(definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(lowercase_payload));
    if (!std::holds_alternative<std::vector<uint8_t>>(lowercase_payload)) return;
    const auto lowercase_program = TokenProgram::compile(std::get<std::vector<uint8_t>>(lowercase_payload));
    CHECK(std::holds_alternative<TokenProgram>(lowercase_program));
    if (!std::holds_alternative<TokenProgram>(lowercase_program)) return;
    const auto lowercase = std::get<TokenProgram>(lowercase_program).encode("A");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(lowercase));
    if (const auto* output = std::get_if<std::vector<uint32_t>>(&lowercase)) {
        CHECK(*output == std::vector<uint32_t>({1}));
    }

    definition = definition_for_serializer();
    definition.byte_map[static_cast<uint8_t>('a')] = static_cast<uint8_t>('*');
    definition.byte_map[static_cast<uint8_t>('*')] = static_cast<uint8_t>('a');
    definition.vocabulary[1].piece = "*";
    definition.merges.clear();
    const auto mapped_payload = serialize_token_program(definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(mapped_payload));
    if (!std::holds_alternative<std::vector<uint8_t>>(mapped_payload)) return;
    const auto mapped_program = TokenProgram::compile(std::get<std::vector<uint8_t>>(mapped_payload));
    CHECK(std::holds_alternative<TokenProgram>(mapped_program));
    if (!std::holds_alternative<TokenProgram>(mapped_program)) return;
    const auto mapped_ids = std::get<TokenProgram>(mapped_program).encode("a");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(mapped_ids));
    if (const auto* output = std::get_if<std::vector<uint32_t>>(&mapped_ids)) {
        CHECK(*output == std::vector<uint32_t>({1}));
    }
    const auto mapped_text = std::get<TokenProgram>(mapped_program).decode(std::array<uint32_t, 1>{1});
    CHECK(std::holds_alternative<std::string>(mapped_text));
    if (const auto* text = std::get_if<std::string>(&mapped_text)) CHECK(*text == "a");
}

void check_added_token_precedence() {
    TokenProgramDefinition definition = definition_for_serializer();
    definition.vocabulary.push_back({"<x>y", static_cast<uint16_t>(VocabFlags::Special), 0});
    definition.added_tokens.push_back({
        10,
        static_cast<uint16_t>(AddedTokenFlags::SingleWord) |
            static_cast<uint16_t>(AddedTokenFlags::Special),
        1, "<x>y", "<x>y"});
    auto serialized = serialize_token_program(definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized)) return;
    auto compiled = TokenProgram::compile(std::get<std::vector<uint8_t>>(serialized));
    CHECK(std::holds_alternative<TokenProgram>(compiled));
    if (!std::holds_alternative<TokenProgram>(compiled)) return;
    const auto priority_wins = std::get<TokenProgram>(compiled).encode("<x>y");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(priority_wins));
    if (const auto* ids = std::get_if<std::vector<uint32_t>>(&priority_wins)) {
        CHECK(*ids == std::vector<uint32_t>({9, 0}));
    }

    definition.added_tokens[0].priority = 7;
    definition.added_tokens[1].priority = 7;
    serialized = serialize_token_program(definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized)) return;
    compiled = TokenProgram::compile(std::get<std::vector<uint8_t>>(serialized));
    CHECK(std::holds_alternative<TokenProgram>(compiled));
    if (!std::holds_alternative<TokenProgram>(compiled)) return;
    const auto longer_wins = std::get<TokenProgram>(compiled).encode("<x>y");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(longer_wins));
    if (const auto* ids = std::get_if<std::vector<uint32_t>>(&longer_wins)) {
        CHECK(*ids == std::vector<uint32_t>({10}));
    }

    definition = definition_for_serializer();
    definition.postprocessor.kind = PostprocessorKind::AddBosEos;
    definition.postprocessor.flags = static_cast<uint8_t>(PostprocessorFlags::AddBos);
    definition.postprocessor.bos_token_id = 6;
    serialized = serialize_token_program(definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized)) return;
    auto mismatch = std::get<std::vector<uint8_t>>(serialized);
    mismatch[324] = 7; // postprocessor BOS differs from the header BOS field
    const auto mismatch_result = TokenProgram::compile(mismatch);
    CHECK(std::holds_alternative<TokenProgramStatus>(mismatch_result));
    if (const auto* status = std::get_if<TokenProgramStatus>(&mismatch_result)) {
        CHECK(status->error == TokenProgramError::InvalidParameter);
    }

    definition = definition_for_serializer();
    definition.vocabulary.push_back({"<single-x>", 0, 0});
    definition.added_tokens.push_back({10, static_cast<uint16_t>(AddedTokenFlags::SingleWord),
                                       9, "x", "x"});
    serialized = serialize_token_program(definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized)) return;
    compiled = TokenProgram::compile(std::get<std::vector<uint8_t>>(serialized));
    CHECK(std::holds_alternative<TokenProgram>(compiled));
    if (!std::holds_alternative<TokenProgram>(compiled)) return;
    const auto unicode_word = std::get<TokenProgram>(compiled).encode("éx");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(unicode_word));
    if (const auto* ids = std::get_if<std::vector<uint32_t>>(&unicode_word)) {
        CHECK(*ids == std::vector<uint32_t>({0, 0, 8}));
    }

    definition = definition_for_serializer();
    definition.vocabulary[9].flags = 0;
    const auto inconsistent_special = serialize_token_program(definition);
    CHECK(std::holds_alternative<TokenProgramStatus>(inconsistent_special));
    if (const auto* status = std::get_if<TokenProgramStatus>(&inconsistent_special)) {
        CHECK(status->error == TokenProgramError::InvalidAddedToken);
    }
}

void check_component_digests_are_domain_separated() {
    auto serialized = serialize_token_program(definition_for_serializer());
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized)) return;
    auto compiled = TokenProgram::compile(std::get<std::vector<uint8_t>>(serialized));
    CHECK(std::holds_alternative<TokenProgram>(compiled));
    if (!std::holds_alternative<TokenProgram>(compiled)) return;
    const Sha256Digest original_vocabulary = std::get<TokenProgram>(compiled).vocabulary_digest();
    const Sha256Digest original_prompt = std::get<TokenProgram>(compiled).prompt_digest();
    CHECK(original_vocabulary != Sha256Digest{});
    CHECK(original_prompt != Sha256Digest{});
    CHECK(original_vocabulary != original_prompt);

    TokenProgramDefinition vocabulary_changed = definition_for_serializer();
    vocabulary_changed.merges.clear();
    vocabulary_changed.vocabulary[3].piece = "abc";
    serialized = serialize_token_program(vocabulary_changed);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized)) return;
    compiled = TokenProgram::compile(std::get<std::vector<uint8_t>>(serialized));
    CHECK(std::holds_alternative<TokenProgram>(compiled));
    if (!std::holds_alternative<TokenProgram>(compiled)) return;
    CHECK(std::get<TokenProgram>(compiled).vocabulary_digest() != original_vocabulary);
    CHECK(std::get<TokenProgram>(compiled).prompt_digest() == original_prompt);

    TokenProgramDefinition prompt_changed = definition_for_serializer();
    prompt_changed.prompt[0].literal = "<human>";
    serialized = serialize_token_program(prompt_changed);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized)) return;
    compiled = TokenProgram::compile(std::get<std::vector<uint8_t>>(serialized));
    CHECK(std::holds_alternative<TokenProgram>(compiled));
    if (!std::holds_alternative<TokenProgram>(compiled)) return;
    CHECK(std::get<TokenProgram>(compiled).vocabulary_digest() == original_vocabulary);
    CHECK(std::get<TokenProgram>(compiled).prompt_digest() != original_prompt);
}

void check_unknown_fusion_preserves_known_symbols() {
    auto definition = definition_for_serializer();
    auto serialized = serialize_token_program(definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized)) return;
    auto compiled = TokenProgram::compile(std::get<std::vector<uint8_t>>(serialized));
    CHECK(std::holds_alternative<TokenProgram>(compiled));
    if (!std::holds_alternative<TokenProgram>(compiled)) return;
    const auto separate = std::get<TokenProgram>(compiled).encode("azzb");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(separate));
    if (const auto* ids = std::get_if<std::vector<uint32_t>>(&separate)) {
        CHECK(*ids == std::vector<uint32_t>({1, 0, 0, 2}));
    }
    const auto trailing_merge = std::get<TokenProgram>(compiled).encode("azab");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(trailing_merge));
    if (const auto* ids = std::get_if<std::vector<uint32_t>>(&trailing_merge)) {
        CHECK(*ids == std::vector<uint32_t>({1, 0, 3}));
    }

    definition.bpe_flags = static_cast<uint16_t>(BpeFlags::FuseUnknown);
    serialized = serialize_token_program(definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized)) return;
    compiled = TokenProgram::compile(std::get<std::vector<uint8_t>>(serialized));
    CHECK(std::holds_alternative<TokenProgram>(compiled));
    if (!std::holds_alternative<TokenProgram>(compiled)) return;
    const auto fused = std::get<TokenProgram>(compiled).encode("azzb");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(fused));
    if (const auto* ids = std::get_if<std::vector<uint32_t>>(&fused)) {
        CHECK(*ids == std::vector<uint32_t>({1, 0, 2}));
    }
}

void check_corruption_fails_closed() {
    const RawPayload original = canonical_payload();
    auto check_error = [](std::span<const uint8_t> bytes, TokenProgramError error) {
        const auto result = TokenProgram::compile(bytes);
        CHECK(std::holds_alternative<TokenProgramStatus>(result));
        if (const auto* status = std::get_if<TokenProgramStatus>(&result)) {
            CHECK_MSG(status->error == error, "expected %u got %u", static_cast<unsigned>(error),
                      static_cast<unsigned>(status->error));
        }
    };

    auto bytes = original.bytes;
    bytes[0] ^= 0xff;
    check_error(bytes, TokenProgramError::PayloadMalformed);
    bytes = original.bytes;
    bytes[10] = 2;
    check_error(bytes, TokenProgramError::UnsupportedVersion);
    bytes = original.bytes;
    bytes.push_back(0);
    check_error(bytes, TokenProgramError::TrailingBytes);
    bytes = original.bytes;
    bytes[304] = 99; // first component kind after the fixed header and map
    check_error(bytes, TokenProgramError::UnsupportedEnum);
    bytes = original.bytes;
    bytes[original.merge_result_offset] = 0xff;
    bytes[original.merge_result_offset + 1] = 0xff;
    bytes[original.merge_result_offset + 2] = 0xff;
    bytes[original.merge_result_offset + 3] = 0xff;
    check_error(bytes, TokenProgramError::InvalidMerge);
    bytes = original.bytes;
    bytes[original.prompt_opcode_offset] = 99;
    check_error(bytes, TokenProgramError::UnsupportedEnum);
    bytes = original.bytes;
    bytes[306] = 1; // nonzero normalizer parameter
    check_error(bytes, TokenProgramError::InvalidParameter);
    bytes = original.bytes;
    bytes[348] = 0xc0; // malformed UTF-8 in the special <unk> vocabulary entry
    check_error(bytes, TokenProgramError::InvalidUtf8);
    bytes = original.bytes;
    set_u32(bytes, 16, static_cast<uint32_t>(token_program_limits::kMaxVocabulary));
    check_error(bytes, TokenProgramError::PayloadMalformed);
    bytes = original.bytes;
    set_u32(bytes, 32, 0);
    check_error(bytes, TokenProgramError::InvalidPrompt);

    TokenProgramDefinition definition = definition_for_serializer();
    definition.vocabulary[2].piece = definition.vocabulary[1].piece;
    const auto duplicate_vocab = serialize_token_program(definition);
    CHECK(std::holds_alternative<TokenProgramStatus>(duplicate_vocab));
    if (const auto* status = std::get_if<TokenProgramStatus>(&duplicate_vocab)) {
        CHECK(status->error == TokenProgramError::DuplicateRecord);
    }
}

void check_input_and_prompt_errors() {
    const auto compiled = TokenProgram::compile(canonical_payload().bytes);
    CHECK(std::holds_alternative<TokenProgram>(compiled));
    if (!std::holds_alternative<TokenProgram>(compiled)) return;
    const auto& program = std::get<TokenProgram>(compiled);
    const std::string invalid("\xc0\xaf", 2);
    const auto encoded = program.encode(invalid);
    CHECK(std::holds_alternative<TokenProgramStatus>(encoded));
    if (const auto* status = std::get_if<TokenProgramStatus>(&encoded)) {
        CHECK(status->error == TokenProgramError::InvalidUtf8);
    }
    const auto rendered = program.render_prompt(invalid);
    CHECK(std::holds_alternative<TokenProgramStatus>(rendered));
    if (const auto* status = std::get_if<TokenProgramStatus>(&rendered)) {
        CHECK(status->error == TokenProgramError::InvalidUtf8);
    }
    const auto bad_id = program.decode(std::array<uint32_t, 1>{100});
    CHECK(std::holds_alternative<TokenProgramStatus>(bad_id));
    if (const auto* status = std::get_if<TokenProgramStatus>(&bad_id)) {
        CHECK(status->error == TokenProgramError::InvalidTokenId);
    }
}

void append_scalar(std::string& output, uint32_t value) {
    if (value <= 0x7f) {
        output.push_back(static_cast<char>(value));
    } else if (value <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0 | (value >> 6)));
        output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    } else if (value <= 0xffff) {
        output.push_back(static_cast<char>(0xe0 | (value >> 12)));
        output.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xf0 | (value >> 18)));
        output.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    }
}

std::string mapped_byte(uint32_t value) {
    std::string output;
    append_scalar(output, value);
    return output;
}

std::string mapped_bytes(const TokenProgramDefinition& definition, std::string_view bytes_value) {
    std::string output;
    for (const unsigned char value : bytes_value) output += mapped_byte(definition.byte_to_unicode[value]);
    return output;
}

TokenProgramDefinition v2_definition() {
    TokenProgramDefinition definition;
    for (unsigned value = 0; value != definition.byte_to_unicode.size(); ++value) {
        definition.byte_to_unicode[value] = value;
    }
    // GPT-2's printable-space/newline symbols are non-ASCII Unicode scalars,
    // so their UTF-8 encoding is deliberately multi-byte in the vocabulary.
    definition.byte_to_unicode[static_cast<unsigned>(' ')] = 0x0120;
    definition.byte_to_unicode[static_cast<unsigned>('\n')] = 0x010a;
    definition.unknown_token_id = 0;
    definition.pretokenizer.kind = PretokenizerKind::UnicodeScalarScanner;
    definition.decoder.kind = DecoderKind::ByteLevel;

    definition.vocabulary = {
        {"<unk>", static_cast<uint16_t>(VocabFlags::Special), 0},
        {mapped_bytes(definition, " "), 0, 0},
        {mapped_bytes(definition, "a"), 0, 0},
        {mapped_bytes(definition, "\n"), 0, 0},
        {mapped_bytes(definition, "1"), 0, 0},
        {mapped_bytes(definition, "2"), 0, 0},
        {mapped_bytes(definition, "3"), 0, 0},
        {mapped_bytes(definition, "4"), 0, 0},
        {mapped_bytes(definition, "123"), 0, 0},
        {mapped_bytes(definition, "1234"), 0, 0},
        {mapped_bytes(definition, "e"), 0, 0},
        {mapped_byte(definition.byte_to_unicode[0x82]), 0, 0},
        {mapped_bytes(definition, "q"), 0, 0},
        {mapped_bytes(definition, " a"), 0, 0},
        {mapped_bytes(definition, "x"), 0, 0},
        {mapped_bytes(definition, "xy"), 0, 0},
        {"<bos>", static_cast<uint16_t>(VocabFlags::Special), 0},
        {"<eos>", static_cast<uint16_t>(VocabFlags::Special), 0},
        {"<x>", static_cast<uint16_t>(VocabFlags::Special), 0},
        {"<xy>", static_cast<uint16_t>(VocabFlags::Special), 0},
    };
    definition.merges = {
        {4, 5, 20, 0}, // overwritten below after the vocabulary is extended
    };
    // Keep the rank table explicit and exact.  IDs 20..23 are ordinary
    // pieces used only as intermediate/results for the independent fixture.
    definition.vocabulary.push_back({mapped_bytes(definition, "12"), 0, 0}); // 20
    definition.vocabulary.push_back({mapped_bytes(definition, "123"), 0, 0}); // 21 (duplicate avoided below)
    definition.vocabulary.push_back({mapped_bytes(definition, "1234"), 0, 0}); // 22
    definition.vocabulary[8].piece = mapped_bytes(definition, "12");
    definition.vocabulary[9].piece = mapped_bytes(definition, "123");
    definition.vocabulary[20].piece = mapped_bytes(definition, "1234");
    // Rebuild the compact fixture with unique result IDs and no accidental
    // duplicate pieces.
    definition.vocabulary.erase(definition.vocabulary.begin() + 21, definition.vocabulary.end());
    definition.vocabulary.push_back({mapped_byte(definition.byte_to_unicode[0xcc]), 0, 0}); // 21
    definition.vocabulary.push_back({mapped_byte(definition.byte_to_unicode[0x81]), 0, 0}); // 22
    definition.vocabulary.push_back({mapped_bytes(definition, "\xcc\x81"), 0, 0}); // 23
    definition.vocabulary.push_back({mapped_bytes(definition, "e\xcc\x81"), 0, 0}); // 24
    definition.merges = {{4, 5, 8, 0}, {8, 6, 9, 1}, {9, 7, 20, 2}, {21, 22, 23, 3},
                         {10, 23, 24, 4}, {1, 2, 13, 5}};
    return definition;
}

struct V2SectionLocation {
    uint16_t kind = 0;
    uint16_t flags = 0;
    size_t header = 0;
    size_t body = 0;
    uint32_t length = 0;
};

bool locate_v2_section(const std::vector<uint8_t>& payload, uint16_t wanted, V2SectionLocation& result) {
    if (payload.size() < 20 || std::string_view(reinterpret_cast<const char*>(payload.data()), 10) !=
                                  kTokenProgramV2Magic) {
        return false;
    }
    auto read16 = [&](size_t offset) {
        return static_cast<uint16_t>(payload[offset]) | (static_cast<uint16_t>(payload[offset + 1]) << 8);
    };
    auto read32 = [&](size_t offset) {
        return static_cast<uint32_t>(payload[offset]) | (static_cast<uint32_t>(payload[offset + 1]) << 8) |
               (static_cast<uint32_t>(payload[offset + 2]) << 16) |
               (static_cast<uint32_t>(payload[offset + 3]) << 24);
    };
    size_t offset = 20;
    const uint16_t count = read16(18);
    for (uint16_t index = 0; index != count; ++index) {
        if (offset > payload.size() || payload.size() - offset < 8) return false;
        const uint16_t kind = read16(offset);
        const uint16_t flags = read16(offset + 2);
        const uint32_t length = read32(offset + 4);
        if (length > payload.size() - offset - 8) return false;
        if (kind == wanted) {
            result = {kind, flags, offset, offset + 8, length};
            return true;
        }
        offset += 8 + length;
    }
    return false;
}

void check_v2_wire_and_unicode_edges() {
    TokenProgramDefinition definition = v2_definition();
    const auto serialized = serialize_token_program_v2(definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized)) return;
    const std::vector<uint8_t>& bytes_value = std::get<std::vector<uint8_t>>(serialized);
    CHECK(std::string_view(reinterpret_cast<const char*>(bytes_value.data()), 10) == kTokenProgramV2Magic);
    const auto compiled = TokenProgram::compile(bytes_value);
    CHECK(std::holds_alternative<TokenProgram>(compiled));
    if (!std::holds_alternative<TokenProgram>(compiled)) return;
    const TokenProgram& program = std::get<TokenProgram>(compiled);
    CHECK(program.is_v2());
    CHECK(program.wire_major_version() == 2);
    const auto roundtrip = program.serialize();
    CHECK(std::holds_alternative<std::vector<uint8_t>>(roundtrip));
    if (const auto* roundtrip_bytes = std::get_if<std::vector<uint8_t>>(&roundtrip)) {
        CHECK(*roundtrip_bytes == bytes_value);
    }

    const auto spaced = program.encode(" a\n");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(spaced));
    if (const auto* ids = std::get_if<std::vector<uint32_t>>(&spaced)) CHECK(*ids == std::vector<uint32_t>({13, 3}));
    const auto digits = program.encode("1234");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(digits));
    if (const auto* ids = std::get_if<std::vector<uint32_t>>(&digits)) CHECK(*ids == std::vector<uint32_t>({9, 7}));
    const auto combining = program.encode("e\xCC\x81");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(combining));
    if (const auto* ids = std::get_if<std::vector<uint32_t>>(&combining)) {
        CHECK_MSG(*ids == std::vector<uint32_t>({10, 23}), "combining ids size=%zu first=%u second=%u",
                  ids->size(), ids->empty() ? 0 : (*ids)[0], ids->size() < 2 ? 0 : (*ids)[1]);
    }
    const auto decoded = program.decode(std::array<uint32_t, 2>{13, 3});
    CHECK(std::holds_alternative<std::string>(decoded));
    if (const auto* text = std::get_if<std::string>(&decoded)) CHECK(*text == " a\n");

    definition.normalizer.kind = NormalizerKind::AsciiLowercase;
    const auto normalized_serialized = serialize_token_program_v2(definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(normalized_serialized));
    if (const auto* payload = std::get_if<std::vector<uint8_t>>(&normalized_serialized)) {
        const auto normalized_program = TokenProgram::compile(*payload);
        CHECK(std::holds_alternative<TokenProgram>(normalized_program));
        if (const auto* compiled_normalized = std::get_if<TokenProgram>(&normalized_program)) {
            CHECK(compiled_normalized->definition().normalizer.kind == NormalizerKind::AsciiLowercase);
            const auto ids = compiled_normalized->encode("A");
            CHECK(std::holds_alternative<std::vector<uint32_t>>(ids));
            if (const auto* output = std::get_if<std::vector<uint32_t>>(&ids)) CHECK(*output == std::vector<uint32_t>({2}));
        }
    }

    definition.added_tokens = {
        {18, static_cast<uint16_t>(AddedTokenFlags::Special), 0, "<x>", "<x>"},
        {19, static_cast<uint16_t>(AddedTokenFlags::Special), 0, "<xy>", "<xy>"},
    };
    const auto with_added = serialize_token_program_v2(definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(with_added));
    if (const auto* payload = std::get_if<std::vector<uint8_t>>(&with_added)) {
        const auto added_program = TokenProgram::compile(*payload);
        CHECK(std::holds_alternative<TokenProgram>(added_program));
        if (const auto* compiled_added = std::get_if<TokenProgram>(&added_program)) {
            const auto ids = compiled_added->encode("<xy>");
            CHECK(std::holds_alternative<std::vector<uint32_t>>(ids));
            if (const auto* output = std::get_if<std::vector<uint32_t>>(&ids)) CHECK(*output == std::vector<uint32_t>({19}));
        }
    }

    definition = v2_definition();
    definition.pretokenizer.flags = static_cast<uint8_t>(PretokenizerFlags::AddPrefixSpace);
    const auto prefixed = serialize_token_program_v2(definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(prefixed));
    if (const auto* payload = std::get_if<std::vector<uint8_t>>(&prefixed)) {
        const auto prefixed_program = TokenProgram::compile(*payload);
        CHECK(std::holds_alternative<TokenProgram>(prefixed_program));
        if (const auto* compiled_prefixed = std::get_if<TokenProgram>(&prefixed_program)) {
            const auto empty = compiled_prefixed->encode("");
            CHECK(std::holds_alternative<std::vector<uint32_t>>(empty));
            if (const auto* ids = std::get_if<std::vector<uint32_t>>(&empty)) CHECK(ids->empty());
        }
    }
}

void check_v2_fail_closed() {
    const auto serialized = serialize_token_program_v2(v2_definition());
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized)) return;
    const auto original = std::get<std::vector<uint8_t>>(serialized);
    const auto check_error = [](const std::vector<uint8_t>& payload, TokenProgramError expected) {
        const auto result = TokenProgram::compile(payload);
        CHECK(std::holds_alternative<TokenProgramStatus>(result));
        if (const auto* status = std::get_if<TokenProgramStatus>(&result)) CHECK(status->error == expected);
    };

    auto invalid_map = original;
    V2SectionLocation map;
    CHECK(locate_v2_section(invalid_map, static_cast<uint16_t>(TokenProgramV2Section::ByteToUnicode), map));
    if (locate_v2_section(invalid_map, static_cast<uint16_t>(TokenProgramV2Section::ByteToUnicode), map)) {
        set_u32(invalid_map, map.body + 4, 0);
        check_error(invalid_map, TokenProgramError::InvalidParameter);
    }
    auto invalid_utf8 = original;
    V2SectionLocation vocabulary;
    CHECK(locate_v2_section(invalid_utf8, static_cast<uint16_t>(TokenProgramV2Section::Vocabulary), vocabulary));
    if (locate_v2_section(invalid_utf8, static_cast<uint16_t>(TokenProgramV2Section::Vocabulary), vocabulary)) {
        // First vocabulary record is <unk>: its text starts after count +
        // length/flags/priority.  Replace the first byte with an invalid lead.
        invalid_utf8[vocabulary.body + 4 + 8] = 0xc0;
        check_error(invalid_utf8, TokenProgramError::InvalidUtf8);
    }
    auto missing_result = original;
    V2SectionLocation bpe;
    CHECK(locate_v2_section(missing_result, static_cast<uint16_t>(TokenProgramV2Section::Bpe), bpe));
    if (locate_v2_section(missing_result, static_cast<uint16_t>(TokenProgramV2Section::Bpe), bpe)) {
        set_u32(missing_result, bpe.body + 4 + 8, 999);
        check_error(missing_result, TokenProgramError::InvalidMerge);
    }
    auto trailing = original;
    trailing.push_back(0);
    check_error(trailing, TokenProgramError::TrailingBytes);

    auto unknown_section = original;
    set_u16(unknown_section, 18, 10);
    u16(unknown_section, 99);
    u16(unknown_section, static_cast<uint16_t>(TokenProgramV2SectionFlags::Required));
    u32(unknown_section, 0);
    check_error(unknown_section, TokenProgramError::UnknownRequiredSection);

    TokenProgramDefinition prompt_definition = v2_definition();
    prompt_definition.prompt = {
        {PromptOpcode::EmitLiteralUtf8, "<u>"},
        {PromptOpcode::EmitUserText, {}},
        {PromptOpcode::EmitGenerationPrompt, "<g>"},
        {PromptOpcode::End, {}},
    };
    const auto prompt_serialized = serialize_token_program_v2(prompt_definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(prompt_serialized));
    if (const auto* prompt_payload = std::get_if<std::vector<uint8_t>>(&prompt_serialized)) {
        auto unknown_op = *prompt_payload;
        V2SectionLocation prompt;
        CHECK(locate_v2_section(unknown_op, static_cast<uint16_t>(TokenProgramV2Section::Prompt), prompt));
        if (locate_v2_section(unknown_op, static_cast<uint16_t>(TokenProgramV2Section::Prompt), prompt)) {
            unknown_op[prompt.body + 4] = 99;
            check_error(unknown_op, TokenProgramError::UnknownRequiredOperation);
        }
    }

    TokenProgramDefinition invalid_definition = v2_definition();
    invalid_definition.vocabulary[1].piece = "\xc0";
    const auto invalid_serialized = serialize_token_program_v2(invalid_definition);
    CHECK(std::holds_alternative<TokenProgramStatus>(invalid_serialized));
    if (const auto* status = std::get_if<TokenProgramStatus>(&invalid_serialized)) CHECK(status->error == TokenProgramError::InvalidUtf8);

    invalid_definition = v2_definition();
    invalid_definition.postprocessor.kind = PostprocessorKind::AddBosEos;
    invalid_definition.postprocessor.flags = static_cast<uint8_t>(PostprocessorFlags::AddBos);
    invalid_definition.postprocessor.bos_token_id = 2;
    const auto regular_bos = serialize_token_program_v2(invalid_definition);
    CHECK(std::holds_alternative<TokenProgramStatus>(regular_bos));
    if (const auto* status = std::get_if<TokenProgramStatus>(&regular_bos)) {
        CHECK(status->error == TokenProgramError::InvalidTokenId);
    }
}

void check_v2_postprocessor_and_decoder_policy() {
    TokenProgramDefinition definition = v2_definition();
    definition.postprocessor.kind = PostprocessorKind::AddBosEos;
    definition.postprocessor.flags = static_cast<uint8_t>(PostprocessorFlags::AddBos) |
                                     static_cast<uint8_t>(PostprocessorFlags::AddEos);
    definition.postprocessor.bos_token_id = 16;
    definition.postprocessor.eos_token_id = 17;
    definition.decoder.flags = static_cast<uint8_t>(DecoderFlags::SkipSpecial);
    const auto serialized = serialize_token_program_v2(definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized)) return;
    const auto compiled = TokenProgram::compile(std::get<std::vector<uint8_t>>(serialized));
    CHECK(std::holds_alternative<TokenProgram>(compiled));
    if (!std::holds_alternative<TokenProgram>(compiled)) return;
    const auto ids = std::get<TokenProgram>(compiled).encode("a");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(ids));
    if (const auto* output = std::get_if<std::vector<uint32_t>>(&ids)) CHECK(*output == std::vector<uint32_t>({16, 2, 17}));
    const auto decoded = std::get<TokenProgram>(compiled).decode(std::array<uint32_t, 3>{16, 2, 17});
    CHECK(std::holds_alternative<std::string>(decoded));
    if (const auto* text = std::get_if<std::string>(&decoded)) CHECK(*text == "a");
}

TokenProgramDefinition v3_byte_bpe_definition() {
    TokenProgramDefinition definition;
    definition.model_kind = TokenProgramModelKind::ByteBpe;
    for (unsigned value = 0; value != 256; ++value) definition.byte_to_unicode[value] = value;
    definition.vocabulary = {
        {"<unk>", static_cast<uint16_t>(VocabFlags::Special), 0, static_cast<uint8_t>(TokenPieceType::Unknown), 0.0f},
        {"a", 0, 0, static_cast<uint8_t>(TokenPieceType::Normal), 0.0f},
        {"b", 0, 0, static_cast<uint8_t>(TokenPieceType::Normal), 0.0f},
        {"ab", 0, 0, static_cast<uint8_t>(TokenPieceType::Normal), 0.0f},
        {"<0x7a>", 0, 0, static_cast<uint8_t>(TokenPieceType::Byte), 0.0f},
        {"<bos>", static_cast<uint16_t>(VocabFlags::Special), 0, static_cast<uint8_t>(TokenPieceType::Control), 0.0f},
        {"<eos>", static_cast<uint16_t>(VocabFlags::Special), 0, static_cast<uint8_t>(TokenPieceType::Control), 0.0f},
    };
    definition.merges = {{1, 2, 3, 0}};
    definition.unknown_token_id = 0;
    definition.byte_fallback = true;
    definition.postprocessor.kind = PostprocessorKind::AddBosEos;
    definition.postprocessor.flags = static_cast<uint8_t>(PostprocessorFlags::AddBos) |
                                     static_cast<uint8_t>(PostprocessorFlags::AddEos);
    definition.postprocessor.bos_token_id = 5;
    definition.postprocessor.eos_token_id = 6;
    definition.prompt = {
        {PromptOpcode::EmitLiteralUtf8, "<u>"},
        {PromptOpcode::EmitUserText, {}},
        {PromptOpcode::EmitGenerationPrompt, "<g>"},
        {PromptOpcode::End, {}},
    };
    definition.prompt_max_bytes = 64;
    definition.stop_ids = {6};
    definition.pretokenizer.kind = PretokenizerKind::ByteLevel;
    definition.normalizer.kind = NormalizerKind::None;
    definition.decoder.kind = DecoderKind::ByteLevel;
    definition.decoder.flags = static_cast<uint8_t>(DecoderFlags::SkipSpecial);
    return definition;
}

TokenProgramDefinition v3_sentencepiece_definition() {
    TokenProgramDefinition definition;
    definition.model_kind = TokenProgramModelKind::SentencePiece;
    definition.normalizer.kind = NormalizerKind::SentencePiece;
    definition.normalizer.flags = static_cast<uint8_t>(SentencePieceNormalizerFlags::AddDummyPrefix) |
                                  static_cast<uint8_t>(SentencePieceNormalizerFlags::EscapeWhitespaces);
    definition.pretokenizer.kind = PretokenizerKind::SentencePiece;
    definition.vocabulary = {
        {"<unk>", static_cast<uint16_t>(VocabFlags::Special), 0, static_cast<uint8_t>(TokenPieceType::Unknown), -10.0f},
        {"<s>", static_cast<uint16_t>(VocabFlags::Special), 0, static_cast<uint8_t>(TokenPieceType::Control), 0.0f},
        {"</s>", static_cast<uint16_t>(VocabFlags::Special), 0, static_cast<uint8_t>(TokenPieceType::Control), 0.0f},
        {"▁hello", 0, 0, static_cast<uint8_t>(TokenPieceType::Normal), -0.1f},
        {"▁world", 0, 0, static_cast<uint8_t>(TokenPieceType::Normal), -0.2f},
        {"▁", 0, 0, static_cast<uint8_t>(TokenPieceType::Normal), -2.0f},
        {"world", 0, 0, static_cast<uint8_t>(TokenPieceType::Normal), -1.0f},
        {"<0xc3>", 0, 0, static_cast<uint8_t>(TokenPieceType::Byte), 0.0f},
        {"<0xa9>", 0, 0, static_cast<uint8_t>(TokenPieceType::Byte), 0.0f},
    };
    definition.unknown_token_id = 0;
    definition.byte_fallback = true;
    definition.postprocessor.kind = PostprocessorKind::None;
    definition.prompt = {
        {PromptOpcode::EmitLiteralUtf8, "<u>"},
        {PromptOpcode::EmitUserText, {}},
        {PromptOpcode::EmitGenerationPrompt, "<g>"},
        {PromptOpcode::End, {}},
    };
    definition.prompt_max_bytes = 64;
    definition.stop_ids = {2};
    definition.decoder.kind = DecoderKind::Identity;
    return definition;
}

void check_v3_byte_bpe_execution_and_streaming() {
    const auto serialized = serialize_token_program_v3(v3_byte_bpe_definition());
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized)) return;
    const auto& bytes_value = std::get<std::vector<uint8_t>>(serialized);
    CHECK(std::string_view(reinterpret_cast<const char*>(bytes_value.data()), 10) == kTokenProgramV3Magic);
    const auto compiled = TokenProgram::compile(bytes_value);
    CHECK(std::holds_alternative<TokenProgram>(compiled));
    if (!std::holds_alternative<TokenProgram>(compiled)) return;
    const auto& program = std::get<TokenProgram>(compiled);
    CHECK(program.is_v3());
    const auto encoded = program.encode("ab");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(encoded));
    if (const auto* ids = std::get_if<std::vector<uint32_t>>(&encoded)) {
        CHECK(*ids == std::vector<uint32_t>({5, 3, 6}));
    }
    const auto fallback = program.encode("z");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(fallback));
    if (const auto* ids = std::get_if<std::vector<uint32_t>>(&fallback)) {
        CHECK(*ids == std::vector<uint32_t>({5, 4, 6}));
    }
    const auto decoded = program.decode(std::array<uint32_t, 3>{5, 3, 6});
    CHECK(std::holds_alternative<std::string>(decoded));
    if (const auto* text = std::get_if<std::string>(&decoded)) CHECK(*text == "ab");

    TokenProgram::StreamState state;
    const auto first = program.decode_chunk(std::array<uint32_t, 1>{3}, state, false);
    const auto second = program.decode_chunk(std::array<uint32_t, 1>{4}, state, true);
    CHECK(std::holds_alternative<std::string>(first));
    CHECK(std::holds_alternative<std::string>(second));
    if (const auto* left = std::get_if<std::string>(&first)) {
        if (const auto* right = std::get_if<std::string>(&second)) CHECK(*left + *right == "abz");
    }
}

void check_v3_sentencepiece_execution_and_fail_closed() {
    const auto source_definition = v3_sentencepiece_definition();
    const auto serialized = serialize_token_program_v3(source_definition);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized)) return;
    const auto compiled = TokenProgram::compile(std::get<std::vector<uint8_t>>(serialized));
    CHECK(std::holds_alternative<TokenProgram>(compiled));
    if (!std::holds_alternative<TokenProgram>(compiled)) return;
    const auto& program = std::get<TokenProgram>(compiled);
    const auto encoded = program.encode("hello world");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(encoded));
    if (const auto* ids = std::get_if<std::vector<uint32_t>>(&encoded)) {
        CHECK(*ids == std::vector<uint32_t>({3, 4}));
    }
    const auto fallback = program.encode("é");
    CHECK(std::holds_alternative<std::vector<uint32_t>>(fallback));
    if (const auto* ids = std::get_if<std::vector<uint32_t>>(&fallback)) {
        CHECK(*ids == std::vector<uint32_t>({5, 7, 8}));
    }
    const auto decoded = program.decode(std::array<uint32_t, 2>{3, 4});
    CHECK(std::holds_alternative<std::string>(decoded));
    if (const auto* text = std::get_if<std::string>(&decoded)) CHECK(*text == "hello world");

    TokenProgram::StreamState utf8_state;
    const auto one_shot = program.decode(std::array<uint32_t, 2>{7, 8});
    const auto first_byte = program.decode_chunk(std::array<uint32_t, 1>{7}, utf8_state, false);
    const auto second_byte = program.decode_chunk(std::array<uint32_t, 1>{8}, utf8_state, true);
    CHECK(std::holds_alternative<std::string>(first_byte));
    CHECK(std::holds_alternative<std::string>(second_byte));
    if (const auto* left = std::get_if<std::string>(&first_byte)) {
        if (const auto* right = std::get_if<std::string>(&second_byte)) {
            if (const auto* expected = std::get_if<std::string>(&one_shot)) CHECK(*left + *right == *expected);
        }
    }

    auto malformed = std::get<std::vector<uint8_t>>(serialized);
    malformed.pop_back();
    const auto truncated = TokenProgram::compile(malformed);
    CHECK(std::holds_alternative<TokenProgramStatus>(truncated));

    malformed = std::get<std::vector<uint8_t>>(serialized);
    set_u32(malformed, 24, UINT32_MAX);
    const auto malformed_length = TokenProgram::compile(malformed);
    CHECK(std::holds_alternative<TokenProgramStatus>(malformed_length));

    auto unknown = v3_byte_bpe_definition();
    unknown.byte_fallback = false;
    unknown.unknown_token_id = kTokenProgramNoTokenId;
    const auto unknown_payload = serialize_token_program_v3(unknown);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(unknown_payload));
    if (const auto* unknown_bytes = std::get_if<std::vector<uint8_t>>(&unknown_payload)) {
        const auto unknown_program = TokenProgram::compile(*unknown_bytes);
        CHECK(std::holds_alternative<TokenProgram>(unknown_program));
        if (const auto* program_value = std::get_if<TokenProgram>(&unknown_program)) {
            const auto unknown_encoded = program_value->encode("z");
            CHECK(std::holds_alternative<TokenProgramStatus>(unknown_encoded));
            if (const auto* status = std::get_if<TokenProgramStatus>(&unknown_encoded)) {
                CHECK(status->error == TokenProgramError::UnknownPiece);
            }
        }
    }

    auto invalid = v3_sentencepiece_definition();
    invalid.vocabulary[3].piece = "\xc0";
    const auto invalid_payload = serialize_token_program_v3(invalid);
    CHECK(std::holds_alternative<TokenProgramStatus>(invalid_payload));
    if (const auto* status = std::get_if<TokenProgramStatus>(&invalid_payload)) {
        CHECK(status->error == TokenProgramError::InvalidUtf8);
    }
    invalid = v3_sentencepiece_definition();
    invalid.merges = {{3, 4, 6, 3}};
    const auto invalid_rank = serialize_token_program_v3(invalid);
    CHECK(std::holds_alternative<TokenProgramStatus>(invalid_rank));
    if (const auto* status = std::get_if<TokenProgramStatus>(&invalid_rank)) {
        CHECK(status->error == TokenProgramError::InvalidMerge);
    }

    auto duplicate_rank = v3_byte_bpe_definition();
    duplicate_rank.vocabulary.push_back({"abb", 0, 0, static_cast<uint8_t>(TokenPieceType::Normal), 0.0f});
    duplicate_rank.merges = {{1, 2, 3, 4}, {3, 2, 7, 4}};
    const auto duplicate_rank_payload = serialize_token_program_v3(duplicate_rank);
    CHECK(std::holds_alternative<TokenProgramStatus>(duplicate_rank_payload));
    if (const auto* status = std::get_if<TokenProgramStatus>(&duplicate_rank_payload)) {
        CHECK(status->error == TokenProgramError::InvalidMerge);
    }
}

} // namespace

int main() {
    check_independent_golden();
    check_canonical_serializer_matches_hand_built();
    check_component_parameters_and_postprocessor();
    check_added_token_precedence();
    check_component_digests_are_domain_separated();
    check_unknown_fusion_preserves_known_symbols();
    check_corruption_fails_closed();
    check_input_and_prompt_errors();
    check_v2_wire_and_unicode_edges();
    check_v2_fail_closed();
    check_v2_postprocessor_and_decoder_policy();
    check_v3_byte_bpe_execution_and_streaming();
    check_v3_sentencepiece_execution_and_fail_closed();
    return test_summary("test_token_program");
}
