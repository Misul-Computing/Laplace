#include "token_contract.h"

#include <array>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include "test_util.h"

using namespace Laplace;

namespace {

TokenContract qualification_contract() {
    TokenContract contract;
    contract.tokenizer_algorithm = TokenizerAlgorithm::TokenIdsOnly;
    contract.tokenizer_version = 0;
    contract.vocabulary_size = 32;
    contract.bos_id = 1;
    contract.eos_id = 2;
    contract.stop_ids = {2, 7};
    for (size_t i = 0; i != contract.authoritative_tokenizer_digest.bytes.size(); ++i) {
        contract.authoritative_tokenizer_digest.bytes[i] = static_cast<uint8_t>(i + 1);
    }
    contract.prompt = TokenIdsOnlyPrompt{};
    return contract;
}

TokenContract product_contract() {
    TokenContract contract = qualification_contract();
    contract.tokenizer_algorithm = TokenizerAlgorithm::ByteBpe;
    contract.tokenizer_version = 1;
    contract.tokenizer_data = {ArtifactId{0}, 64, 128, {}};
    contract.tokenizer_data.digest = contract.authoritative_tokenizer_digest;
    contract.vocabulary_digest.bytes[0] = 0x35;
    contract.prompt = PromptTemplate{
        1,
        {
            {PromptOperationKind::AppendLiteral, {'u', 's', 'e', 'r', ':'}, kNoTokenId},
            {PromptOperationKind::AppendInputText, {}, kNoTokenId},
        },
    };
    contract.authoritative_template_digest.bytes[0] = 0x11;
    return contract;
}

void check_qualification_token_ids() {
    const TokenContract contract = qualification_contract();
    CHECK(contract.validate().ok());
    CHECK(contract.prompt_mode() == TokenPromptMode::TokenIdsOnly);
    CHECK(contract.has_text_descriptor() == false);

    const auto encoded = contract.encode_token_ids(std::array<uint32_t, 3>{4, 8, 2});
    CHECK(std::holds_alternative<std::vector<uint32_t>>(encoded));
    if (const auto* ids = std::get_if<std::vector<uint32_t>>(&encoded)) {
        CHECK(*ids == std::vector<uint32_t>({4, 8, 2}));
    }

    const auto first = contract.serialize();
    const auto second = contract.serialize();
    CHECK(std::holds_alternative<std::vector<uint8_t>>(first));
    CHECK(first == second);
    if (const auto* bytes = std::get_if<std::vector<uint8_t>>(&first)) {
        const auto decoded = TokenContract::deserialize(*bytes);
        CHECK(std::holds_alternative<TokenContract>(decoded));
        if (const auto* round_trip = std::get_if<TokenContract>(&decoded)) CHECK(*round_trip == contract);
    }
}

void check_bounded_product_descriptor() {
    const TokenContract contract = product_contract();
    CHECK(contract.validate().ok());
    CHECK(contract.prompt_mode() == TokenPromptMode::SerializedTemplate);
    CHECK(contract.has_text_descriptor());

    const auto encoded = contract.encode_token_ids(std::array<uint32_t, 1>{1});
    CHECK(std::holds_alternative<TokenContractStatus>(encoded));
    if (const auto* status = std::get_if<TokenContractStatus>(&encoded)) {
        CHECK(status->error == TokenContractError::TextEncodingUnavailable);
    }

    const auto serialized = contract.serialize();
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (const auto* bytes = std::get_if<std::vector<uint8_t>>(&serialized)) {
        const auto decoded = TokenContract::deserialize(*bytes);
        CHECK(std::holds_alternative<TokenContract>(decoded));
        if (const auto* round_trip = std::get_if<TokenContract>(&decoded)) CHECK(*round_trip == contract);
    }

    TokenContract v2 = contract;
    v2.tokenizer_version = 2;
    CHECK(v2.validate().ok());

    TokenContract sentencepiece = contract;
    sentencepiece.tokenizer_algorithm = TokenizerAlgorithm::SentencePiece;
    sentencepiece.tokenizer_version = 3;
    CHECK(sentencepiece.validate().ok());
    const auto sentencepiece_serialized = sentencepiece.serialize();
    CHECK(std::holds_alternative<std::vector<uint8_t>>(sentencepiece_serialized));
    if (const auto* bytes = std::get_if<std::vector<uint8_t>>(&sentencepiece_serialized)) {
        const auto decoded = TokenContract::deserialize(*bytes);
        CHECK(std::holds_alternative<TokenContract>(decoded));
        if (const auto* round_trip = std::get_if<TokenContract>(&decoded)) CHECK(*round_trip == sentencepiece);
    }
}

void check_exact_validation_and_fail_closed() {
    TokenContract contract = qualification_contract();

    contract.tokenizer_algorithm = static_cast<TokenizerAlgorithm>(99);
    CHECK(contract.validate().error == TokenContractError::AlgorithmUnsupported);
    contract = qualification_contract();
    contract.tokenizer_version = 1;
    CHECK(contract.validate().error == TokenContractError::AlgorithmVersionUnsupported);
    contract = qualification_contract();
    contract.vocabulary_size = 0;
    CHECK(contract.validate().error == TokenContractError::VocabularySizeInvalid);
    contract = qualification_contract();
    contract.authoritative_tokenizer_digest = {};
    CHECK(contract.validate().error == TokenContractError::TokenizerDigestMissing);
    contract = qualification_contract();
    contract.bos_id = contract.vocabulary_size;
    CHECK(contract.validate().error == TokenContractError::BosIdOutOfRange);
    contract = qualification_contract();
    contract.stop_ids = {7, 7};
    CHECK(contract.validate().error == TokenContractError::StopIdsNotCanonical);
    contract = qualification_contract();
    contract.stop_ids = {9, 3};
    CHECK(contract.validate().error == TokenContractError::StopIdsNotCanonical);

    contract = qualification_contract();
    const auto out_of_range = contract.encode_token_ids(std::array<uint32_t, 1>{contract.vocabulary_size});
    CHECK(std::holds_alternative<TokenContractStatus>(out_of_range));
    if (const auto* status = std::get_if<TokenContractStatus>(&out_of_range)) {
        CHECK(status->error == TokenContractError::InputTokenOutOfRange);
    }

    contract = product_contract();
    contract.tokenizer_version = 4;
    CHECK(contract.validate().error == TokenContractError::AlgorithmVersionUnsupported);
    contract = product_contract();
    contract.tokenizer_algorithm = TokenizerAlgorithm::SentencePiece;
    contract.tokenizer_version = 2;
    CHECK(contract.validate().error == TokenContractError::AlgorithmVersionUnsupported);
    contract = product_contract();
    contract.prompt = TokenIdsOnlyPrompt{};
    CHECK(contract.validate().error == TokenContractError::PromptOperationInvalid);
    contract = product_contract();
    contract.vocabulary_digest = {};
    CHECK(contract.validate().error == TokenContractError::VocabularyDigestMissing);
    contract = product_contract();
    contract.tokenizer_data.length = 0;
    CHECK(contract.validate().error == TokenContractError::TokenizerDataReferenceInvalid);
    contract = product_contract();
    contract.authoritative_tokenizer_digest.bytes[0] ^= 0xff;
    CHECK(contract.validate().error == TokenContractError::TokenizerDigestMismatch);
    contract = product_contract();
    contract.prompt = PromptTemplate{99, {}};
    CHECK(contract.validate().error == TokenContractError::PromptVersionUnsupported);
    contract = product_contract();
    contract.prompt = PromptTemplate{1, {{static_cast<PromptOperationKind>(99), {}, kNoTokenId}}};
    CHECK(contract.validate().error == TokenContractError::PromptOperationInvalid);
    contract = product_contract();
    contract.authoritative_template_digest = {};
    CHECK(contract.validate().error == TokenContractError::TemplateDigestMissing);
}

void check_serialization_rejects_tampering() {
    const TokenContract contract = product_contract();
    const auto serialized = contract.serialize();
    CHECK(std::holds_alternative<std::vector<uint8_t>>(serialized));
    if (!std::holds_alternative<std::vector<uint8_t>>(serialized)) return;

    auto bytes = std::get<std::vector<uint8_t>>(serialized);
    CHECK(!bytes.empty());
    bytes[0] ^= 0xff;
    const auto bad_magic = TokenContract::deserialize(bytes);
    CHECK(std::holds_alternative<TokenContractStatus>(bad_magic));
    if (const auto* status = std::get_if<TokenContractStatus>(&bad_magic)) {
        CHECK(status->error == TokenContractError::SerializationMalformed);
    }

    bytes = std::get<std::vector<uint8_t>>(serialized);
    bytes[8] = 2;
    const auto bad_version = TokenContract::deserialize(bytes);
    CHECK(std::holds_alternative<TokenContractStatus>(bad_version));
    if (const auto* status = std::get_if<TokenContractStatus>(&bad_version)) {
        CHECK(status->error == TokenContractError::SerializationVersionUnsupported);
    }

    bytes = std::get<std::vector<uint8_t>>(serialized);
    bytes.push_back(0);
    const auto trailing = TokenContract::deserialize(bytes);
    CHECK(std::holds_alternative<TokenContractStatus>(trailing));
    if (const auto* status = std::get_if<TokenContractStatus>(&trailing)) {
        CHECK(status->error == TokenContractError::SerializationMalformed);
    }

    for (size_t length = 0; length + 1 < bytes.size(); ++length) {
        const auto truncated = TokenContract::deserialize(
            std::span<const uint8_t>(bytes.data(), length));
        CHECK(std::holds_alternative<TokenContractStatus>(truncated));
    }
}

} // namespace

int main() {
    check_qualification_token_ids();
    check_bounded_product_descriptor();
    check_exact_validation_and_fail_closed();
    check_serialization_rejects_tampering();
    return test_summary("test_token_contract");
}
