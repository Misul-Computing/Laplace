#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "artifact_set.h"

namespace Laplace {

constexpr uint32_t kNoTokenId = UINT32_MAX;

enum class TokenizerAlgorithm : uint8_t {
    TokenIdsOnly = 1, // Qualification-only input; no text implementation is claimed.
    ByteBpe = 2,      // Bounded descriptor kind; execution remains a separate gate.
    SentencePiece = 3,// Bounded ModelProto-derived descriptor; execution is TokenProgram V3.
};

enum class TokenPromptMode : uint8_t {
    TokenIdsOnly = 1,
    SerializedTemplate = 2,
};

enum class PromptOperationKind : uint8_t {
    AppendLiteral = 1,
    AppendInputText = 2,
    AppendTokenId = 3,
};

struct TokenArtifactReference {
    ArtifactId artifact_id{};
    uint64_t offset = 0;
    uint64_t length = 0;
    Sha256Digest digest{};
    friend bool operator==(const TokenArtifactReference&, const TokenArtifactReference&) = default;
};

struct PromptOperation {
    PromptOperationKind kind = PromptOperationKind::AppendLiteral;
    std::vector<uint8_t> literal;
    uint32_t token_id = kNoTokenId;
    friend bool operator==(const PromptOperation&, const PromptOperation&) = default;
};

struct PromptTemplate {
    uint16_t version = 0;
    std::vector<PromptOperation> operations;
    friend bool operator==(const PromptTemplate&, const PromptTemplate&) = default;
};

struct TokenIdsOnlyPrompt {
    friend bool operator==(TokenIdsOnlyPrompt, TokenIdsOnlyPrompt) = default;
};

using TokenPrompt = std::variant<TokenIdsOnlyPrompt, PromptTemplate>;

enum class TokenContractError : uint16_t {
    None = 0,
    AlgorithmUnsupported = 1,
    AlgorithmVersionUnsupported = 2,
    TokenizerDataReferenceInvalid = 3,
    VocabularySizeInvalid = 4,
    VocabularyDigestMissing = 5,
    TokenizerDigestMissing = 6,
    TemplateDigestMissing = 7,
    BosIdOutOfRange = 8,
    EosIdOutOfRange = 9,
    StopIdsNotCanonical = 10,
    PromptVersionUnsupported = 11,
    PromptOperationInvalid = 12,
    InputTokenOutOfRange = 13,
    TextEncodingUnavailable = 14,
    SerializationMalformed = 15,
    SerializationVersionUnsupported = 16,
    TokenizerDigestMismatch = 17,
};

struct TokenContractStatus {
    TokenContractError error = TokenContractError::None;
    uint32_t index = UINT32_MAX;
    std::string detail;

    bool ok() const noexcept { return error == TokenContractError::None; }
    explicit operator bool() const noexcept { return ok(); }
    friend bool operator==(const TokenContractStatus&, const TokenContractStatus&) = default;
};

// This is the sole token contract. TokenIdContract remains an alias below for
// source compatibility while SemanticManifest migrates to this authority.
struct TokenContract {
    TokenizerAlgorithm tokenizer_algorithm = TokenizerAlgorithm::TokenIdsOnly;
    uint16_t tokenizer_version = 0;
    TokenArtifactReference tokenizer_data;

    uint32_t vocabulary_size = 0;
    Sha256Digest vocabulary_digest{};
    uint32_t bos_id = kNoTokenId;
    uint32_t eos_id = kNoTokenId;
    std::vector<uint32_t> stop_ids;

    // These retain the existing SemanticModel digest projection. They are
    // part of the canonical contract and identify the exact producer data.
    Sha256Digest authoritative_tokenizer_digest{};
    Sha256Digest authoritative_template_digest{};
    TokenPrompt prompt = TokenIdsOnlyPrompt{};

    TokenContractStatus validate() const;

    using SerializationResult = std::variant<std::vector<uint8_t>, TokenContractStatus>;
    using DecodeResult = std::variant<TokenContract, TokenContractStatus>;
    using EncodeResult = std::variant<std::vector<uint32_t>, TokenContractStatus>;

    SerializationResult serialize() const;
    static DecodeResult deserialize(std::span<const uint8_t> bytes);

    // Token-ID-only is intentionally the only executable seam in this slice.
    // A product descriptor is authoritative package data, but its text
    // interpreter is not claimed until a later runtime change.
    EncodeResult encode_token_ids(std::span<const uint32_t> token_ids) const;

    TokenPromptMode prompt_mode() const noexcept;
    bool has_text_descriptor() const noexcept {
        return tokenizer_algorithm == TokenizerAlgorithm::ByteBpe ||
               tokenizer_algorithm == TokenizerAlgorithm::SentencePiece;
    }
    const TokenPrompt& prompt_encoding() const noexcept { return prompt; }
    friend bool operator==(const TokenContract&, const TokenContract&) = default;
};

// The old name is a compatibility spelling, not a second representation.
using TokenIdContract = TokenContract;

namespace token_contract_detail {

constexpr std::array<uint8_t, 8> kMagic = {'L', 'A', 'P', 'T', 'O', 'K', '0', '2'};
constexpr uint16_t kFormatMajor = 1;
constexpr uint16_t kFormatMinor = 0;
constexpr size_t kMaxStops = 1u << 20;
constexpr size_t kMaxPromptOperations = 256;
constexpr size_t kMaxLiteralBytes = 4096;
constexpr size_t kMaxPromptBytes = 64u * 1024u;
constexpr size_t kMaxTokenizerDataBytes = 128u * 1024u * 1024u;
constexpr size_t kMaxSerializedBytes = kMaxStops * sizeof(uint32_t) + kMaxPromptBytes + 1024u;

inline bool all_zero(const Sha256Digest& digest) {
    return std::all_of(digest.bytes.begin(), digest.bytes.end(), [](uint8_t byte) { return byte == 0; });
}

inline TokenContractStatus failure(TokenContractError error, std::string detail,
                                   uint32_t index = UINT32_MAX) {
    return {error, index, std::move(detail)};
}

inline void append_u16(std::vector<uint8_t>& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8));
}

inline void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8) bytes.push_back(static_cast<uint8_t>(value >> shift));
}

inline void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8) bytes.push_back(static_cast<uint8_t>(value >> shift));
}

inline void append_digest(std::vector<uint8_t>& bytes, const Sha256Digest& digest) {
    bytes.insert(bytes.end(), digest.bytes.begin(), digest.bytes.end());
}

inline void append_reference(std::vector<uint8_t>& bytes, const TokenArtifactReference& reference) {
    append_u32(bytes, reference.artifact_id.value);
    append_u64(bytes, reference.offset);
    append_u64(bytes, reference.length);
    append_digest(bytes, reference.digest);
}

inline std::vector<uint8_t> serialized_bytes(const TokenContract& contract) {
    std::vector<uint8_t> bytes;
    bytes.reserve(256 + contract.stop_ids.size() * sizeof(uint32_t));
    bytes.insert(bytes.end(), kMagic.begin(), kMagic.end());
    append_u16(bytes, kFormatMajor);
    append_u16(bytes, kFormatMinor);
    bytes.push_back(static_cast<uint8_t>(contract.tokenizer_algorithm));
    bytes.push_back(static_cast<uint8_t>(contract.prompt_mode()));
    append_u16(bytes, contract.tokenizer_version);
    append_reference(bytes, contract.tokenizer_data);
    append_u32(bytes, contract.vocabulary_size);
    append_digest(bytes, contract.vocabulary_digest);
    append_u32(bytes, contract.bos_id);
    append_u32(bytes, contract.eos_id);
    append_u32(bytes, static_cast<uint32_t>(contract.stop_ids.size()));
    for (uint32_t id : contract.stop_ids) append_u32(bytes, id);
    append_digest(bytes, contract.authoritative_tokenizer_digest);
    append_digest(bytes, contract.authoritative_template_digest);
    if (const auto* template_data = std::get_if<PromptTemplate>(&contract.prompt)) {
        append_u16(bytes, template_data->version);
        append_u16(bytes, 0);
        append_u32(bytes, static_cast<uint32_t>(template_data->operations.size()));
        for (const PromptOperation& operation : template_data->operations) {
            bytes.push_back(static_cast<uint8_t>(operation.kind));
            bytes.push_back(0);
            append_u16(bytes, static_cast<uint16_t>(operation.literal.size()));
            append_u32(bytes, operation.token_id);
            bytes.insert(bytes.end(), operation.literal.begin(), operation.literal.end());
        }
    }
    return bytes;
}

class Reader {
public:
    explicit Reader(std::span<const uint8_t> bytes) : bytes_(bytes) {}
    size_t remaining() const noexcept { return bytes_.size() - offset_; }
    bool take(size_t count, std::span<const uint8_t>& result) {
        if (count > remaining()) return false;
        result = bytes_.subspan(offset_, count);
        offset_ += count;
        return true;
    }
    bool u16(uint16_t& result) {
        std::span<const uint8_t> value;
        if (!take(2, value)) return false;
        result = static_cast<uint16_t>(value[0]) | (static_cast<uint16_t>(value[1]) << 8);
        return true;
    }
    bool u32(uint32_t& result) {
        std::span<const uint8_t> value;
        if (!take(4, value)) return false;
        result = static_cast<uint32_t>(value[0]) |
                 (static_cast<uint32_t>(value[1]) << 8) |
                 (static_cast<uint32_t>(value[2]) << 16) |
                 (static_cast<uint32_t>(value[3]) << 24);
        return true;
    }
    bool u64(uint64_t& result) {
        std::span<const uint8_t> value;
        if (!take(8, value)) return false;
        result = 0;
        for (unsigned shift = 0; shift != 64; shift += 8) {
            result |= static_cast<uint64_t>(value[shift / 8]) << shift;
        }
        return true;
    }
    bool u8(uint8_t& result) {
        std::span<const uint8_t> value;
        if (!take(1, value)) return false;
        result = value[0];
        return true;
    }

private:
    std::span<const uint8_t> bytes_;
    size_t offset_ = 0;
};

inline bool read_digest(Reader& reader, Sha256Digest& digest) {
    std::span<const uint8_t> bytes;
    if (!reader.take(digest.bytes.size(), bytes)) return false;
    std::copy(bytes.begin(), bytes.end(), digest.bytes.begin());
    return true;
}

inline bool read_reference(Reader& reader, TokenArtifactReference& reference) {
    return reader.u32(reference.artifact_id.value) && reader.u64(reference.offset) &&
           reader.u64(reference.length) && read_digest(reader, reference.digest);
}

} // namespace token_contract_detail

inline std::vector<uint8_t> token_contract_canonical_bytes(const TokenContract& contract) {
    return token_contract_detail::serialized_bytes(contract);
}

inline TokenContractStatus TokenContract::validate() const {
    switch (tokenizer_algorithm) {
    case TokenizerAlgorithm::TokenIdsOnly:
        if (tokenizer_version != 0 || tokenizer_data != TokenArtifactReference{}) {
            return token_contract_detail::failure(TokenContractError::AlgorithmVersionUnsupported,
                                                  "token-ID-only qualification has no tokenizer payload");
        }
        break;
    case TokenizerAlgorithm::ByteBpe:
        if (tokenizer_version != 1 && tokenizer_version != 2 && tokenizer_version != 3) {
            return token_contract_detail::failure(TokenContractError::AlgorithmVersionUnsupported,
                                                  "byte-BPE descriptor version is unsupported");
        }
        [[fallthrough]];
    case TokenizerAlgorithm::SentencePiece:
        if (tokenizer_algorithm == TokenizerAlgorithm::SentencePiece && tokenizer_version != 3) {
            return token_contract_detail::failure(TokenContractError::AlgorithmVersionUnsupported,
                                                  "SentencePiece descriptor version is unsupported");
        }
        if (token_contract_detail::all_zero(vocabulary_digest)) {
            return token_contract_detail::failure(TokenContractError::VocabularyDigestMissing,
                                                  "text tokenizer descriptor needs an exact vocabulary digest");
        }
        if (tokenizer_data.artifact_id.value == UINT32_MAX || tokenizer_data.length == 0 ||
            tokenizer_data.length > token_contract_detail::kMaxTokenizerDataBytes ||
            tokenizer_data.offset > std::numeric_limits<uint64_t>::max() - tokenizer_data.length ||
            token_contract_detail::all_zero(tokenizer_data.digest)) {
            return token_contract_detail::failure(TokenContractError::TokenizerDataReferenceInvalid,
                                                  "text tokenizer descriptor needs a bounded package artifact reference");
        }
        if (authoritative_tokenizer_digest != tokenizer_data.digest) {
            return token_contract_detail::failure(TokenContractError::TokenizerDigestMismatch,
                                                  "authoritative tokenizer digest does not match tokenizer data");
        }
        break;
    default:
        return token_contract_detail::failure(TokenContractError::AlgorithmUnsupported,
                                              "tokenizer algorithm kind is unsupported");
    }
    if (vocabulary_size == 0) {
        return token_contract_detail::failure(TokenContractError::VocabularySizeInvalid,
                                              "vocabulary size must be non-zero");
    }
    if (token_contract_detail::all_zero(authoritative_tokenizer_digest)) {
        return token_contract_detail::failure(TokenContractError::TokenizerDigestMissing,
                                              "authoritative tokenizer digest must be present");
    }
    if (bos_id != kNoTokenId && bos_id >= vocabulary_size) {
        return token_contract_detail::failure(TokenContractError::BosIdOutOfRange,
                                              "BOS token ID is outside the vocabulary");
    }
    if (eos_id != kNoTokenId && eos_id >= vocabulary_size) {
        return token_contract_detail::failure(TokenContractError::EosIdOutOfRange,
                                              "EOS token ID is outside the vocabulary");
    }
    if (stop_ids.size() > token_contract_detail::kMaxStops) {
        return token_contract_detail::failure(TokenContractError::StopIdsNotCanonical,
                                              "stop ID list exceeds the package limit");
    }
    for (size_t index = 0; index != stop_ids.size(); ++index) {
        if (stop_ids[index] >= vocabulary_size ||
            (index != 0 && stop_ids[index - 1] >= stop_ids[index])) {
            return token_contract_detail::failure(TokenContractError::StopIdsNotCanonical,
                                                  "stop IDs must be in strictly increasing vocabulary order",
                                                  static_cast<uint32_t>(index));
        }
    }
    if ((tokenizer_algorithm == TokenizerAlgorithm::ByteBpe ||
         tokenizer_algorithm == TokenizerAlgorithm::SentencePiece) &&
        !std::holds_alternative<PromptTemplate>(prompt)) {
        return token_contract_detail::failure(TokenContractError::PromptOperationInvalid,
                                              "text tokenizer descriptor needs a versioned text prompt template");
    }
    if (const auto* template_data = std::get_if<PromptTemplate>(&prompt)) {
        if (tokenizer_algorithm == TokenizerAlgorithm::TokenIdsOnly) {
            return token_contract_detail::failure(TokenContractError::PromptOperationInvalid,
                                                  "token-ID-only qualification cannot claim a text prompt template");
        }
        if (template_data->version != 1) {
            return token_contract_detail::failure(TokenContractError::PromptVersionUnsupported,
                                                  "prompt operation grammar version is unsupported");
        }
        if (template_data->operations.empty() ||
            template_data->operations.size() > token_contract_detail::kMaxPromptOperations) {
            return token_contract_detail::failure(TokenContractError::PromptOperationInvalid,
                                                  "prompt template needs bounded operations");
        }
        if (token_contract_detail::all_zero(authoritative_template_digest)) {
            return token_contract_detail::failure(TokenContractError::TemplateDigestMissing,
                                                  "prompt template needs an authoritative digest");
        }
        size_t literal_bytes = 0;
        bool has_input_text = false;
        for (size_t index = 0; index != template_data->operations.size(); ++index) {
            const PromptOperation& operation = template_data->operations[index];
            switch (operation.kind) {
            case PromptOperationKind::AppendLiteral:
                if (operation.literal.empty() || operation.literal.size() > token_contract_detail::kMaxLiteralBytes ||
                    operation.token_id != kNoTokenId) {
                    return token_contract_detail::failure(TokenContractError::PromptOperationInvalid,
                                                          "literal prompt operation is malformed",
                                                          static_cast<uint32_t>(index));
                }
                break;
            case PromptOperationKind::AppendInputText:
                if (!operation.literal.empty() || operation.token_id != kNoTokenId) {
                    return token_contract_detail::failure(TokenContractError::PromptOperationInvalid,
                                                          "input-text prompt operation carries an unexpected payload",
                                                          static_cast<uint32_t>(index));
                }
                has_input_text = true;
                break;
            case PromptOperationKind::AppendTokenId:
                if (!operation.literal.empty() || operation.token_id >= vocabulary_size) {
                    return token_contract_detail::failure(TokenContractError::PromptOperationInvalid,
                                                          "token prompt operation is outside the vocabulary",
                                                          static_cast<uint32_t>(index));
                }
                break;
            default:
                return token_contract_detail::failure(TokenContractError::PromptOperationInvalid,
                                                      "prompt operation kind is unsupported",
                                                      static_cast<uint32_t>(index));
            }
            if (literal_bytes > token_contract_detail::kMaxPromptBytes - operation.literal.size()) {
                return token_contract_detail::failure(TokenContractError::PromptOperationInvalid,
                                                      "prompt literal bytes exceed the package limit",
                                                      static_cast<uint32_t>(index));
            }
            literal_bytes += operation.literal.size();
        }
        if (!has_input_text) {
            return token_contract_detail::failure(TokenContractError::PromptOperationInvalid,
                                                  "prompt template must expose an input-text operation");
        }
    } else if (!std::holds_alternative<TokenIdsOnlyPrompt>(prompt)) {
        return token_contract_detail::failure(TokenContractError::PromptOperationInvalid,
                                              "prompt encoding variant is unsupported");
    }
    return {};
}

inline TokenContract::SerializationResult TokenContract::serialize() const {
    const TokenContractStatus status = validate();
    if (!status.ok()) return status;
    return token_contract_detail::serialized_bytes(*this);
}

inline TokenContract::DecodeResult TokenContract::deserialize(std::span<const uint8_t> bytes) {
    using namespace token_contract_detail;
    if (bytes.size() > kMaxSerializedBytes) {
        return failure(TokenContractError::SerializationMalformed,
                       "token contract serialization exceeds the package limit");
    }
    Reader reader(bytes);
    std::span<const uint8_t> magic;
    uint16_t major = 0, minor = 0, tokenizer_version = 0;
    uint8_t algorithm = 0, prompt_mode = 0;
    if (!reader.take(kMagic.size(), magic) || !std::equal(kMagic.begin(), kMagic.end(), magic.begin()) ||
        !reader.u16(major) || !reader.u16(minor)) {
        return failure(TokenContractError::SerializationMalformed,
                       "token contract serialization header is malformed");
    }
    if (major != kFormatMajor || minor != kFormatMinor) {
        return failure(TokenContractError::SerializationVersionUnsupported,
                       "token contract serialization version is unsupported");
    }
    if (!reader.u8(algorithm) || !reader.u8(prompt_mode) || !reader.u16(tokenizer_version)) {
        return failure(TokenContractError::SerializationMalformed,
                       "token contract serialization header is truncated");
    }
    if (algorithm != static_cast<uint8_t>(TokenizerAlgorithm::TokenIdsOnly) &&
        algorithm != static_cast<uint8_t>(TokenizerAlgorithm::ByteBpe) &&
        algorithm != static_cast<uint8_t>(TokenizerAlgorithm::SentencePiece)) {
        return failure(TokenContractError::AlgorithmUnsupported,
                       "tokenizer algorithm kind is unsupported");
    }
    if (prompt_mode != static_cast<uint8_t>(TokenPromptMode::TokenIdsOnly) &&
        prompt_mode != static_cast<uint8_t>(TokenPromptMode::SerializedTemplate)) {
        return failure(TokenContractError::SerializationMalformed,
                       "token contract prompt mode is malformed");
    }
    TokenContract contract;
    contract.tokenizer_algorithm = static_cast<TokenizerAlgorithm>(algorithm);
    contract.tokenizer_version = tokenizer_version;
    if (!read_reference(reader, contract.tokenizer_data) || !reader.u32(contract.vocabulary_size) ||
        !read_digest(reader, contract.vocabulary_digest) || !reader.u32(contract.bos_id) ||
        !reader.u32(contract.eos_id)) {
        return failure(TokenContractError::SerializationMalformed,
                       "token contract common fields are truncated");
    }
    uint32_t stop_count = 0;
    if (!reader.u32(stop_count) || stop_count > kMaxStops ||
        stop_count > reader.remaining() / sizeof(uint32_t)) {
        return failure(TokenContractError::SerializationMalformed,
                       "token contract stop list is malformed");
    }
    contract.stop_ids.resize(stop_count);
    for (uint32_t& id : contract.stop_ids) if (!reader.u32(id)) {
        return failure(TokenContractError::SerializationMalformed,
                       "token contract stop list is truncated");
    }
    if (!read_digest(reader, contract.authoritative_tokenizer_digest) ||
        !read_digest(reader, contract.authoritative_template_digest)) {
        return failure(TokenContractError::SerializationMalformed,
                       "token contract producer digests are truncated");
    }
    if (prompt_mode == static_cast<uint8_t>(TokenPromptMode::SerializedTemplate)) {
        PromptTemplate template_data;
        uint16_t template_reserved = 0;
        uint32_t operation_count = 0;
        if (!reader.u16(template_data.version) || !reader.u16(template_reserved) ||
            template_reserved != 0 || !reader.u32(operation_count) ||
            operation_count > kMaxPromptOperations) {
            return failure(TokenContractError::SerializationMalformed,
                           "prompt operation grammar header is malformed");
        }
        template_data.operations.reserve(operation_count);
        for (uint32_t index = 0; index != operation_count; ++index) {
            uint8_t kind = 0, operation_reserved = 0;
            uint16_t literal_length = 0;
            PromptOperation operation;
            if (!reader.u8(kind) || !reader.u8(operation_reserved) || operation_reserved != 0 ||
                !reader.u16(literal_length) || !reader.u32(operation.token_id) ||
                literal_length > kMaxLiteralBytes || literal_length > reader.remaining()) {
                return failure(TokenContractError::SerializationMalformed,
                               "prompt operation is malformed", index);
            }
            operation.kind = static_cast<PromptOperationKind>(kind);
            std::span<const uint8_t> literal;
            if (!reader.take(literal_length, literal)) {
                return failure(TokenContractError::SerializationMalformed,
                               "prompt operation literal is truncated", index);
            }
            operation.literal.assign(literal.begin(), literal.end());
            template_data.operations.push_back(std::move(operation));
        }
        contract.prompt = std::move(template_data);
    } else {
        contract.prompt = TokenIdsOnlyPrompt{};
    }
    if (reader.remaining() != 0) {
        return failure(TokenContractError::SerializationMalformed,
                       "token contract serialization has trailing bytes");
    }
    const TokenContractStatus status = contract.validate();
    if (!status.ok()) return status;
    return contract;
}

inline TokenContract::EncodeResult TokenContract::encode_token_ids(std::span<const uint32_t> token_ids) const {
    const TokenContractStatus status = validate();
    if (!status.ok()) return status;
    if (tokenizer_algorithm != TokenizerAlgorithm::TokenIdsOnly) {
        return token_contract_detail::failure(TokenContractError::TextEncodingUnavailable,
                                              "text tokenizer executor is not part of this authority slice");
    }
    std::vector<uint32_t> encoded;
    encoded.reserve(token_ids.size());
    for (size_t index = 0; index != token_ids.size(); ++index) {
        if (token_ids[index] >= vocabulary_size) {
            return token_contract_detail::failure(TokenContractError::InputTokenOutOfRange,
                                                  "input token ID is outside the vocabulary",
                                                  static_cast<uint32_t>(index));
        }
        encoded.push_back(token_ids[index]);
    }
    return encoded;
}

inline TokenPromptMode TokenContract::prompt_mode() const noexcept {
    return std::holds_alternative<TokenIdsOnlyPrompt>(prompt)
               ? TokenPromptMode::TokenIdsOnly
               : TokenPromptMode::SerializedTemplate;
}

inline std::string_view token_contract_error_name(TokenContractError error) noexcept {
    switch (error) {
    case TokenContractError::None: return "none";
    case TokenContractError::AlgorithmUnsupported: return "algorithm_unsupported";
    case TokenContractError::AlgorithmVersionUnsupported: return "algorithm_version_unsupported";
    case TokenContractError::TokenizerDataReferenceInvalid: return "tokenizer_data_reference_invalid";
    case TokenContractError::VocabularySizeInvalid: return "vocabulary_size_invalid";
    case TokenContractError::VocabularyDigestMissing: return "vocabulary_digest_missing";
    case TokenContractError::TokenizerDigestMissing: return "tokenizer_digest_missing";
    case TokenContractError::TemplateDigestMissing: return "template_digest_missing";
    case TokenContractError::BosIdOutOfRange: return "bos_id_out_of_range";
    case TokenContractError::EosIdOutOfRange: return "eos_id_out_of_range";
    case TokenContractError::StopIdsNotCanonical: return "stop_ids_not_canonical";
    case TokenContractError::PromptVersionUnsupported: return "prompt_version_unsupported";
    case TokenContractError::PromptOperationInvalid: return "prompt_operation_invalid";
    case TokenContractError::InputTokenOutOfRange: return "input_token_out_of_range";
    case TokenContractError::TextEncodingUnavailable: return "text_encoding_unavailable";
    case TokenContractError::SerializationMalformed: return "serialization_malformed";
    case TokenContractError::SerializationVersionUnsupported: return "serialization_version_unsupported";
    case TokenContractError::TokenizerDigestMismatch: return "tokenizer_digest_mismatch";
    }
    return "unknown";
}

} // namespace Laplace
