#include <cmath>
#include <CommonCrypto/CommonDigest.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "artifact_set.h"
#include "compat_rule.h"
#include "reference_fp32.h"
#include "test_util.h"
#include "fixtures/qwen2.5-0.5b-f16-golden.inc"

using namespace Laplace;

namespace {

uint16_t read_u16(const std::vector<uint8_t>& bytes, size_t offset) {
    return static_cast<uint16_t>(bytes[offset]) | (static_cast<uint16_t>(bytes[offset + 1]) << 8);
}

uint32_t read_u32(const std::vector<uint8_t>& bytes, size_t offset) {
    return static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 16) | (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

float read_f32(const std::vector<uint8_t>& bytes, size_t offset) {
    float value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

std::vector<uint8_t> read_file(const char* path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

Sha256Digest digest_bytes(std::span<const uint8_t> bytes) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    constexpr size_t kChunk = 1024 * 1024;
    for (size_t offset = 0; offset != bytes.size();) {
        const size_t length = std::min(kChunk, bytes.size() - offset);
        CC_SHA256_Update(&context, bytes.data() + offset, static_cast<CC_LONG>(length));
        offset += length;
    }
    Sha256Digest result;
    CC_SHA256_Final(result.bytes.data(), &context);
    return result;
}

uint32_t argmax(const std::vector<float>& values) {
    uint32_t result = 0;
    for (uint32_t index = 1; index != values.size(); ++index) if (values[index] > values[result]) result = index;
    return result;
}

void test_external_gate_a(const char* path, uint32_t first_prefix, uint32_t last_prefix) {
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return;
    auto view = std::get<ArtifactSet>(artifacts).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return;
    const PackageView& package = std::get<PackageView>(view);
    auto loaded = load_expected_fixture_gguf(package, bundled_compatibility_rules());
    CHECK(std::holds_alternative<ValidatedPackage>(loaded));
    if (!std::holds_alternative<ValidatedPackage>(loaded)) return;
    auto runtime = std::get<ValidatedPackage>(loaded).runtime_package();
    const std::string golden_path = std::string(LAPLACE_SOURCE_DIR) + "/tests/fixtures/qwen2.5-0.5b-f16-golden.bin";
    const auto golden = read_file(golden_path.c_str());
    CHECK(golden.size() == 4862139);
    if (golden.size() != 4862139) return;
    CHECK(std::memcmp(golden.data(), "LAPGLD1", 7) == 0);
    CHECK(read_u16(golden, 7) == 1);
    const Sha256Digest trailing_digest = digest_bytes(std::span<const uint8_t>(golden.data(), golden.size() - 32));
    CHECK(std::memcmp(trailing_digest.bytes.data(), golden.data() + golden.size() - 32, trailing_digest.bytes.size()) == 0);
    auto corrupted_golden = golden;
    corrupted_golden.back() ^= 1;
    CHECK(std::memcmp(trailing_digest.bytes.data(), corrupted_golden.data() + corrupted_golden.size() - 32,
                      trailing_digest.bytes.size()) != 0);
    CHECK(digest_bytes(golden).hex() == TestEvidence::kGoldenSha256);
    CHECK(package.digest().hex() == TestEvidence::kArtifactSha256);
    CHECK(runtime->rule_fingerprint().hex() == TestEvidence::kRuleFingerprint);
    CHECK(runtime->rule_revision() == 1);
    CHECK(runtime->qualification_state() == RuleQualificationState::CanonicalRuntimePassed);
    CHECK(std::memcmp(golden.data() + 11, package.digest().bytes.data(), 32) == 0);
    CHECK(read_u32(golden, 43) == 8);
    CHECK(read_u32(golden, 47) == 151936);
    std::vector<uint32_t> all_ids;
    for (size_t index = 0; index != 8; ++index) all_ids.push_back(read_u32(golden, 59 + index * 4));
    constexpr size_t kHeaderLength = 91;
    constexpr size_t kRecordLength = 8 + 151936 * sizeof(float);
    for (uint32_t prefix = first_prefix; prefix <= last_prefix; ++prefix) {
        auto result = reference_fp32(*runtime, std::span<const uint32_t>(all_ids.data(), prefix));
        CHECK(std::holds_alternative<ReferenceOutput>(result));
        if (!std::holds_alternative<ReferenceOutput>(result)) continue;
        const auto& output = std::get<ReferenceOutput>(result);
        CHECK(output.logits.size() == 151936);
        CHECK(output.operator_outputs.size() == 339);
        CHECK(output.key_state.tokens == prefix);
        CHECK(std::isfinite(output.logits[0]));
        const size_t record = kHeaderLength + static_cast<size_t>(prefix - 1) * kRecordLength;
        CHECK(read_u32(golden, record) == prefix);
        const uint32_t expected_argmax = read_u32(golden, record + 4);
        float max_error = 0;
        uint32_t max_index = 0;
        uint32_t mismatches = 0;
        for (uint32_t index = 0; index != output.logits.size(); ++index) {
            const float expected = read_f32(golden, record + 8 + static_cast<size_t>(index) * 4);
            const float actual = output.logits[index];
            if (std::abs(actual - expected) > max_error) {
                max_error = std::abs(actual - expected);
                max_index = index;
            }
            CHECK(std::isfinite(actual));
            if (std::abs(actual - expected) > 1e-4f + 1e-4f * std::abs(expected)) {
                if (mismatches < 3) {
                    std::fprintf(stderr, "Gate A mismatch prefix=%u index=%u actual=%g expected=%g error=%g bound=%g\n",
                                 prefix, index, actual, expected, std::abs(actual - expected),
                                 1e-4f + 1e-4f * std::abs(expected));
                }
                ++mismatches;
            }
        }
        CHECK(mismatches == 0);
        CHECK(argmax(output.logits) == expected_argmax);
        std::fprintf(stderr, "Gate A prefix=%u mismatch_count=%u max_abs_error=%g index=%u actual=%g expected=%g\n",
                     prefix, mismatches, max_error, max_index, output.logits[max_index],
                     read_f32(golden, record + 8 + static_cast<size_t>(max_index) * 4));
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 1 && argc != 2) {
        std::fprintf(stderr, "usage: test_external_golden [prefix]\n");
        return 2;
    }
    uint32_t first_prefix = 1;
    uint32_t last_prefix = 8;
    if (argc == 2) {
        const long parsed = std::strtol(argv[1], nullptr, 10);
        if (parsed < 1 || parsed > 8) return 2;
        first_prefix = static_cast<uint32_t>(parsed);
        last_prefix = first_prefix;
    }
    test_external_gate_a(LAPLACE_QUALIFICATION_GGUF, first_prefix, last_prefix);
    return test_summary("test_external_golden");
}
