#pragma once

namespace Laplace::TestEvidence {

struct DigestRecord {
    const char* name;
    const char* sha256;
};

struct WheelRecord {
    const char* filename;
    const char* sha256;
};

inline constexpr char kSchema[] = "LAPEVID1";
inline constexpr char kModelRepository[] = "Qwen/Qwen2.5-0.5B";
inline constexpr char kModelRevision[] = "060db6499f32faf8b98477b0a26969ef7d8b9987";
inline constexpr char kModelLicense[] = "Apache-2.0";
inline constexpr char kLlamaCppRevision[] = "8144f3192e5a3131cd043f284525e6ceebf82d0f";
inline constexpr char kTransformersRevision[] = "052e652d6d53c2b26ffde87e039b723949a53493";
inline constexpr char kTransformersVersion[] = "4.46.3";
inline constexpr char kPythonVersion[] = "3.11.9";
inline constexpr char kPipVersion[] = "24.3.1";
inline constexpr char kTorchVersion[] = "2.5.1";
inline constexpr char kGgufVersion[] = "0.10.0";
inline constexpr char kAccelerateVersion[] = "1.1.1";
inline constexpr char kSentencePieceVersion[] = "0.1.98";
inline constexpr char kLockSha256[] = "c08338a06e80a4a596eec4ac8c7fab71759f56fdf113c1b18f961936fafbbb3c";
inline constexpr char kArtifactSha256[] = "73f932c0a3a3daf1122545473973b4189ef0811cd6f2f14d73afe9fa1edd8efd";
inline constexpr char kRuleFingerprint[] = "da0ecff3a0e97b3f61338fa301416da841e4401b853f5785dfe56c33baca4263";
inline constexpr char kGoldenSha256[] = "66a98374245b8f313a95560d567e6957551951fd492fed57cfa8653a7acc3ce2";
inline constexpr char kGoldenScriptSha256[] = "7456afb8c4a0ef319c40a4462be8c754b60bd2d591d90cb703ced4ae285d161f";
inline constexpr char kConverterSha256[] = "e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f";
inline constexpr char kSentencePieceWheel[] = "sentencepiece-0.1.98-cp311-cp311-macosx_11_0_arm64.whl";
inline constexpr char kSentencePieceWheelUrl[] = "https://files.pythonhosted.org/packages/51/12/00397a20c3934efb2ecc5778e3d0dd0912dd60229f8a1b974e3d38b692d4/sentencepiece-0.1.98-cp311-cp311-macosx_11_0_arm64.whl";
inline constexpr char kSentencePieceWheelSha256[] = "7f7dc2fc175623529fb60a2799748f8877cd48c4541b32cd97b8523465e88b69";
inline constexpr char kArtifactCommand[] = "python convert_hf_to_gguf.py qwen2.5-0.5b --outfile qwen2.5-0.5b-f16.gguf --outtype f16 --split-max-size 0 --split-max-tensors 0";
inline constexpr char kGoldenCommand[] = "PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python make_qwen2_5_0_5b_golden.py --input-ids 1,2,3,4,5,6,7,8";
inline constexpr unsigned kMaximumContext = 32768;
inline constexpr unsigned kVocabulary = 151936;
inline constexpr unsigned kPrefixCount = 8;
inline constexpr float kAtol = 1e-4f;
inline constexpr float kRtol = 1e-4f;
inline constexpr unsigned kF16Rank2PlaneCount = 169;
inline constexpr unsigned kF32Rank1PlaneCount = 121;

inline constexpr DigestRecord kModelFiles[] = {
    {"config.json", "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b"},
    {"model.safetensors", "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"},
    {"tokenizer.json", "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"},
    {"tokenizer_config.json", "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9"},
    {"vocab.json", "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"},
    {"merges.txt", "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"},
};

inline constexpr WheelRecord kWheels[] = {
    {"accelerate-1.1.1-py3-none-any.whl", "61edd81762131b8d4bede008643fa1e1f3bf59bec710ebda9771443e24feae02"},
    {"certifi-2024.8.30-py3-none-any.whl", "922820b53db7a7257ffbda3f597266d435245903d80737e34f8a45ff3e3230d8"},
    {"charset_normalizer-3.4.0-cp311-cp311-macosx_11_0_arm64.whl", "6dba5d19c4dfab08e58d5b36304b3f92f3bd5d42c1a3fa37b5ba5cdf6dfcbcee"},
    {"filelock-3.16.1-py3-none-any.whl", "2082e5703d51fbf98ea75855d9d5527e33d8ff23099bec374a134febee6946b0"},
    {"fsspec-2024.10.0-py3-none-any.whl", "03b9a6785766a4de40368b88906366755e2819e758b83705c88cd7cb5fe81871"},
    {"gguf-0.10.0-py3-none-any.whl", "706089fba756a06913227841b4a6c8398360fa991569fd974e663a92b224e33f"},
    {"huggingface_hub-0.26.5-py3-none-any.whl", "fb7386090bbe892072e64b85f7c4479fd2d65eea5f2543327c970d5169e83924"},
    {"idna-3.10-py3-none-any.whl", "946d195a0d259cbba61165e88e65941f16e9b36ea6ddb97f00452bae8b1287d3"},
    {"jinja2-3.1.4-py3-none-any.whl", "bc5dd2abb727a5319567b7a813e6a2e7318c39f4f487cfe6c89c6f9c7d25197d"},
    {"MarkupSafe-3.0.2-cp311-cp311-macosx_11_0_arm64.whl", "93335ca3812df2f366e80509ae119189886b0f3c2b81325d39efdb84a1e2ae93"},
    {"mpmath-1.3.0-py3-none-any.whl", "a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c"},
    {"networkx-3.4.2-py3-none-any.whl", "df5d4365b724cf81b8c6a7312509d0c22386097011ad1abe274afd5e9d3bbc5f"},
    {"numpy-1.26.4-cp311-cp311-macosx_11_0_arm64.whl", "edd8b5fe47dab091176d21bb6de568acdd906d1887a4584a15a9a96a1dca06ef"},
    {"packaging-24.2-py3-none-any.whl", "09abb1bccd265c01f4a3aa3f7a7db064b36514d2cba19a2f694fe6150451a759"},
    {"pip-24.3.1-py3-none-any.whl", "3790624780082365f47549d032f3770eeb2b1e8bd1f7b2e02dace1afa361b4ed"},
    {"psutil-6.1.0-cp36-abi3-macosx_11_0_arm64.whl", "0895b8414afafc526712c498bd9de2b063deaac4021a3b3c34566283464aff8e"},
    {"PyYAML-6.0.2-cp311-cp311-macosx_11_0_arm64.whl", "1e2120ef853f59c7419231f3bf4e7021f1b936f6ebd222406c3b60212205d2ee"},
    {"regex-2024.11.6-cp311-cp311-macosx_11_0_arm64.whl", "94d87b689cdd831934fa3ce16cc15cd65748e6d689f5d2b8f4f4df2065c9fa20"},
    {"requests-2.32.3-py3-none-any.whl", "70761cfe03c773ceb22aa2f671b4757976145175cdfca038c02654d061d6dcc6"},
    {"safetensors-0.4.5-cp311-cp311-macosx_11_0_arm64.whl", "bb07000b19d41e35eecef9a454f31a8b4718a185293f0d0b1c4b61d6e4487971"},
    {"sentencepiece-0.1.98-cp311-cp311-macosx_11_0_arm64.whl", "7f7dc2fc175623529fb60a2799748f8877cd48c4541b32cd97b8523465e88b69"},
    {"setuptools-75.6.0-py3-none-any.whl", "ce74b49e8f7110f9bf04883b730f4765b774ef3ef28f722cce7c273d253aaf7d"},
    {"sympy-1.13.1-py3-none-any.whl", "db36cdc64bf61b9b24578b6f7bab1ecdd2452cf008f34faa33776680c26d66f8"},
    {"tokenizers-0.20.3-cp311-cp311-macosx_11_0_arm64.whl", "61cbf11954f3b481d08723ebd048ba4b11e582986f9be74d2c3bdd9293a4538d"},
    {"torch-2.5.1-cp311-none-macosx_11_0_arm64.whl", "31f8c39660962f9ae4eeec995e3049b5492eb7360dd4f07377658ef4d728fa4c"},
    {"tqdm-4.67.1-py3-none-any.whl", "26445eca388f82e72884e0d580d5464cd801a3ea01e63e5601bdff9ba6a48de2"},
    {"transformers-4.46.3-py3-none-any.whl", "a12ef6f52841fd190a3e5602145b542d03507222f2c64ebb7ee92e8788093aef"},
    {"typing_extensions-4.12.2-py3-none-any.whl", "04e5ca0351e0f3f85c6853954072df659d0d13fac324d0072316b67d7794700d"},
    {"urllib3-2.2.3-py3-none-any.whl", "ca899ca043dcb1bafa3e262d73aa25c465bfb49e0bd9dd5d59f1d0acba2f8fac"},
};

} // namespace Laplace::TestEvidence
