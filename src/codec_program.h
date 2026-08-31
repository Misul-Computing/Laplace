#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

namespace Laplace {

struct PhysicalCodecIdentity;

using CodecProgramDigest = std::array<uint8_t, 32>;

// This is the only identity a planner or package declaration may carry.  The
// decode grammar is intentionally not represented by a public format enum.
struct CodecProgramIdentity {
    uint16_t abi_version = 0;
    CodecProgramDigest contract_digest{};
    friend bool operator==(const CodecProgramIdentity&, const CodecProgramIdentity&) = default;
};

enum class CodecProgramError : uint8_t {
    None = 0,
    UnknownIdentity = 1,
    ContractDigestMismatch = 2,
    InvalidPlaneCount = 3,
    InvalidElementCount = 4,
    InvalidOffset = 5,
    InvalidLength = 6,
    InvalidStride = 7,
    AccessOutOfBounds = 8,
    ArithmeticOverflow = 9,
    InvalidContract = 10,
    RoundingModeUnavailable = 11,
};

// `storage` is the immutable backing artifact.  offset and length describe
// the declared plane span inside it; stride is the byte distance between
// encoded units.  A decoder must prove the final byte access against both the
// declared span and the backing storage before reading.
struct CodecPlaneBinding {
    std::span<const uint8_t> storage{};
    uint64_t offset = 0;
    uint64_t length = 0;
    uint64_t stride = 0;
};

struct CodecProgramDeclaration {
    CodecProgramIdentity identity;
    uint64_t element_count = 0;
    std::vector<CodecPlaneBinding> planes;
};

using CodecProgramDecodeResult = std::variant<std::vector<float>, CodecProgramError>;

class CodecProgramRegistry;

class CodecProgram {
public:
    CodecProgramIdentity identity() const noexcept { return identity_; }
    std::span<const uint8_t> canonical_bytes() const noexcept { return canonical_bytes_; }
    CodecProgramError validate(const CodecProgramDeclaration&) const;
    // Compare the complete typed physical contract bound by this program.
    // The arithmetic digest is this program's opaque contract digest.
    bool matches_physical_identity(const PhysicalCodecIdentity&) const;

private:
    CodecProgramIdentity identity_;
    std::vector<uint8_t> canonical_bytes_;

    friend class CodecProgramRegistry;
    friend CodecProgramRegistry make_application_codec_registry();
};

class CodecProgramRegistry {
public:
    const std::vector<CodecProgram>& programs() const noexcept { return programs_; }
    const CodecProgram* resolve(const CodecProgramIdentity&) const noexcept;
    // Reference decoding allocates output; the caller supplies its allocation policy.
    // CodecProgram::validate() remains nonallocating and has no such cap.
    CodecProgramDecodeResult decode(const CodecProgramDeclaration&,
                                    uint64_t max_decode_elements) const;

private:
    std::vector<CodecProgram> programs_;
    friend CodecProgramRegistry make_application_codec_registry();
};

// Application-owned, exact contracts for the two representations already
// decoded by the repository.  The returned registry is immutable by API:
// package input can only refer to one of these registered identities.
CodecProgramRegistry make_application_codec_registry();

CodecProgramDigest codec_program_digest(std::span<const uint8_t> canonical_bytes);

} // namespace Laplace
