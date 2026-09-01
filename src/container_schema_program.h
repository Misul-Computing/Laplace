#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include "compatibility_report.h"

namespace Laplace {

enum class ContainerSchemaOpcode : uint8_t {
    MatchBytes = 1,
    ReadU32Le = 2,
    ReadU64Le = 3,
    SetConstant = 4,
    CaptureCursor = 5,
    CaptureFileSize = 6,
    Add = 7,
    Multiply = 8,
    AlignCursor = 9,
    Advance = 10,
    SetCursor = 11,
    EmitRange = 12,
    LoopBegin = 13,
    LoopEnd = 14,
    RequireCursorEnd = 15,
};

struct ContainerSchemaInstruction {
    ContainerSchemaOpcode opcode = ContainerSchemaOpcode::MatchBytes;
    uint32_t destination = UINT32_MAX;
    uint32_t input_a = UINT32_MAX;
    uint32_t input_b = UINT32_MAX;
    uint32_t section_id = 0;
    uint64_t immediate = 0;
    std::vector<uint8_t> literal;
};

struct ContainerSchemaProgram {
    uint16_t major = 1;
    uint16_t minor = 0;
    uint32_t register_count = 0;
    uint32_t predicate_count = 0;
    uint64_t maximum_steps = 1'000'000;
    uint64_t maximum_loop_iterations = 1'000'000;
    uint32_t maximum_ranges = 65'536;
    std::vector<ContainerSchemaInstruction> instructions;
};

struct ContainerSchemaDigest {
    std::array<uint8_t, 32> bytes{};
    friend bool operator==(ContainerSchemaDigest,
                           ContainerSchemaDigest) = default;
};

struct ContainerRange {
    uint32_t section_id = 0;
    uint64_t offset = 0;
    uint64_t length = 0;
    friend bool operator==(const ContainerRange&, const ContainerRange&) = default;
};

struct ContainerExtraction {
    ContainerSchemaDigest schema_digest;
    std::vector<ContainerRange> ranges;
};

using ContainerSchemaDigestResult =
    std::variant<ContainerSchemaDigest, CompatibilityReport>;
using ContainerSchemaResult =
    std::variant<ContainerExtraction, CompatibilityReport>;
using ContainerSchemaWireResult =
    std::variant<std::vector<uint8_t>, CompatibilityReport>;
using ContainerSchemaSetResult =
    std::variant<std::vector<ContainerSchemaProgram>, CompatibilityReport>;

ContainerSchemaDigestResult
container_schema_digest(const ContainerSchemaProgram& program);

ContainerSchemaResult select_container_schema(
    std::span<const ContainerSchemaProgram> schemas,
    std::span<const uint8_t> bytes);

ContainerSchemaWireResult encode_container_schema_set(
    std::span<const ContainerSchemaProgram> schemas);

ContainerSchemaSetResult decode_container_schema_set(
    std::span<const uint8_t> wire);

} // namespace Laplace
