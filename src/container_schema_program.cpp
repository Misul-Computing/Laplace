#include "container_schema_program.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace Laplace {
namespace {

constexpr uint32_t kMaximumRegisters = 64;
constexpr uint32_t kMaximumInstructions = 4096;
constexpr size_t kMaximumSchemas = 1024;
constexpr uint64_t kMaximumSteps = 1'000'000;
constexpr uint64_t kMaximumLoopIterations = 1'000'000;
constexpr uint32_t kMaximumRanges = 65'536;
constexpr uint64_t kMaximumSelectionSteps = 4'000'000;
constexpr size_t kMaximumLiteralBytes = 65'536;

CompatibilityReport schema_error(CompatibilityError code, std::string detail) {
    return compatibility_report(code, std::move(detail));
}

bool register_valid(uint32_t index, const ContainerSchemaProgram& program) {
    return index < program.register_count;
}

enum InstructionField : uint8_t {
    FieldDestination = 1u << 0,
    FieldInputA = 1u << 1,
    FieldInputB = 1u << 2,
    FieldSection = 1u << 3,
    FieldImmediate = 1u << 4,
    FieldLiteral = 1u << 5,
};

bool fields_canonical(const ContainerSchemaInstruction& instruction,
                      uint8_t allowed) {
    return ((allowed & FieldDestination) != 0 ||
            instruction.destination == UINT32_MAX) &&
           ((allowed & FieldInputA) != 0 ||
            instruction.input_a == UINT32_MAX) &&
           ((allowed & FieldInputB) != 0 ||
            instruction.input_b == UINT32_MAX) &&
           ((allowed & FieldSection) != 0 || instruction.section_id == 0) &&
           ((allowed & FieldImmediate) != 0 || instruction.immediate == 0) &&
           ((allowed & FieldLiteral) != 0 || instruction.literal.empty());
}

std::optional<CompatibilityReport>
validate_schema(const ContainerSchemaProgram& program) {
    if (program.major != 1 || program.minor != 0) {
        return schema_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                            "container schema program version is unsupported");
    }
    if (program.register_count > kMaximumRegisters ||
        program.instructions.empty() ||
        program.instructions.size() > kMaximumInstructions ||
        program.predicate_count == 0 ||
        program.predicate_count > program.instructions.size() ||
        program.maximum_steps == 0 ||
        program.maximum_steps > kMaximumSteps ||
        program.maximum_loop_iterations == 0 ||
        program.maximum_loop_iterations > kMaximumLoopIterations ||
        program.maximum_ranges == 0 ||
        program.maximum_ranges > kMaximumRanges) {
        return schema_error(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                            "container schema program exceeds its bounded contract");
    }

    size_t literal_bytes = 0;
    std::vector<size_t> loop_stack;
    loop_stack.reserve(program.instructions.size());
    std::vector<bool> defined(program.register_count, false);
    std::vector<std::vector<bool>> definition_stack;
    definition_stack.reserve(program.instructions.size());
    const auto require_defined = [&](uint32_t index) {
        return register_valid(index, program) && defined[index];
    };
    for (size_t index = 0; index < program.instructions.size(); ++index) {
        const ContainerSchemaInstruction& instruction =
            program.instructions[index];
        if (index < program.predicate_count) {
            if (instruction.opcode != ContainerSchemaOpcode::MatchBytes ||
                instruction.literal.empty()) {
                return schema_error(
                    CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                    "container schema predicates must be nonempty byte matches");
            }
        } else if (instruction.opcode == ContainerSchemaOpcode::MatchBytes) {
            return schema_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                                "container schema byte matches must precede execution");
        }
        if (instruction.literal.size() >
            kMaximumLiteralBytes - literal_bytes) {
            return schema_error(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                                "container schema literals exceed their byte limit");
        }
        literal_bytes += instruction.literal.size();

        switch (instruction.opcode) {
        case ContainerSchemaOpcode::MatchBytes:
            if (!fields_canonical(instruction, FieldLiteral)) {
                return schema_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                                    "container schema instruction has ignored fields");
            }
            break;
        case ContainerSchemaOpcode::ReadU32Le:
        case ContainerSchemaOpcode::ReadU64Le:
        case ContainerSchemaOpcode::CaptureCursor:
        case ContainerSchemaOpcode::CaptureFileSize:
            if (!fields_canonical(instruction, FieldDestination) ||
                !register_valid(instruction.destination, program)) {
                return schema_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                                    "container schema destination register is invalid");
            }
            defined[instruction.destination] = true;
            break;
        case ContainerSchemaOpcode::SetConstant:
            if (!fields_canonical(instruction,
                                  FieldDestination | FieldImmediate) ||
                !register_valid(instruction.destination, program)) {
                return schema_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                                    "container schema constant register is invalid");
            }
            defined[instruction.destination] = true;
            break;
        case ContainerSchemaOpcode::Add:
        case ContainerSchemaOpcode::Multiply:
            if (!fields_canonical(instruction,
                                  FieldDestination | FieldInputA | FieldInputB) ||
                !register_valid(instruction.destination, program) ||
                !require_defined(instruction.input_a) ||
                !require_defined(instruction.input_b)) {
                return schema_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                                    "container schema arithmetic register is invalid");
            }
            defined[instruction.destination] = true;
            break;
        case ContainerSchemaOpcode::AlignCursor:
        case ContainerSchemaOpcode::Advance:
        case ContainerSchemaOpcode::SetCursor:
            if (!fields_canonical(instruction, FieldInputA) ||
                !require_defined(instruction.input_a)) {
                return schema_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                                    "container schema cursor register is invalid");
            }
            break;
        case ContainerSchemaOpcode::EmitRange:
            if (!fields_canonical(
                    instruction, FieldInputA | FieldInputB | FieldSection) ||
                !require_defined(instruction.input_a) ||
                !require_defined(instruction.input_b)) {
                return schema_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                                    "container schema range register is invalid");
            }
            break;
        case ContainerSchemaOpcode::LoopBegin:
            if (!fields_canonical(instruction,
                                  FieldInputA | FieldImmediate) ||
                !require_defined(instruction.input_a) ||
                instruction.immediate <= index ||
                instruction.immediate >= program.instructions.size() ||
                program.instructions[static_cast<size_t>(instruction.immediate)].opcode !=
                    ContainerSchemaOpcode::LoopEnd ||
                program.instructions[static_cast<size_t>(instruction.immediate)].immediate !=
                    index) {
                return schema_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                                    "container schema loop boundary is invalid");
            }
            loop_stack.push_back(index);
            definition_stack.push_back(defined);
            break;
        case ContainerSchemaOpcode::LoopEnd:
            if (!fields_canonical(instruction, FieldImmediate) ||
                loop_stack.empty() || loop_stack.back() != instruction.immediate) {
                return schema_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                                    "container schema loops are not properly nested");
            }
            loop_stack.pop_back();
            defined = std::move(definition_stack.back());
            definition_stack.pop_back();
            break;
        case ContainerSchemaOpcode::RequireCursorEnd:
            if (!fields_canonical(instruction, 0)) {
                return schema_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                                    "container schema instruction has ignored fields");
            }
            break;
        default:
            return schema_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                                "container schema opcode is unsupported");
        }
    }
    if (!loop_stack.empty()) {
        return schema_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                            "container schema loop is unterminated");
    }
    return std::nullopt;
}

struct DigestBuilder {
    CC_SHA256_CTX context;

    DigestBuilder() { CC_SHA256_Init(&context); }

    void bytes(std::span<const uint8_t> value) {
        size_t offset = 0;
        while (offset < value.size()) {
            const size_t count =
                std::min<size_t>(value.size() - offset, 1024 * 1024);
            CC_SHA256_Update(&context, value.data() + offset,
                             static_cast<CC_LONG>(count));
            offset += count;
        }
    }
    void u8(uint8_t value) { bytes(std::span<const uint8_t>(&value, 1)); }
    void u16(uint16_t value) {
        const std::array<uint8_t, 2> encoded = {
            static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8)};
        bytes(encoded);
    }
    void u32(uint32_t value) {
        std::array<uint8_t, 4> encoded{};
        for (unsigned shift = 0; shift != 32; shift += 8)
            encoded[shift / 8] = static_cast<uint8_t>(value >> shift);
        bytes(encoded);
    }
    void u64(uint64_t value) {
        std::array<uint8_t, 8> encoded{};
        for (unsigned shift = 0; shift != 64; shift += 8)
            encoded[shift / 8] = static_cast<uint8_t>(value >> shift);
        bytes(encoded);
    }
    ContainerSchemaDigest finish() {
        ContainerSchemaDigest digest;
        CC_SHA256_Final(digest.bytes.data(), &context);
        return digest;
    }
};

ContainerSchemaDigest digest_unchecked(const ContainerSchemaProgram& program) {
    DigestBuilder builder;
    constexpr std::array<uint8_t, 35> domain = {
        'l','a','p','l','a','c','e','-','c','o','n','t','a','i','n','e','r','-',
        's','c','h','e','m','a','-','p','r','o','g','r','a','m','-','v','1'};
    builder.bytes(domain);
    builder.u16(program.major);
    builder.u16(program.minor);
    builder.u32(program.register_count);
    builder.u32(program.predicate_count);
    builder.u64(program.maximum_steps);
    builder.u64(program.maximum_loop_iterations);
    builder.u32(program.maximum_ranges);
    builder.u32(static_cast<uint32_t>(program.instructions.size()));
    for (const ContainerSchemaInstruction& instruction : program.instructions) {
        builder.u8(static_cast<uint8_t>(instruction.opcode));
        builder.u32(instruction.destination);
        builder.u32(instruction.input_a);
        builder.u32(instruction.input_b);
        builder.u32(instruction.section_id);
        builder.u64(instruction.immediate);
        builder.u32(static_cast<uint32_t>(instruction.literal.size()));
        builder.bytes(instruction.literal);
    }
    return builder.finish();
}

bool predicates_match(const ContainerSchemaProgram& program,
                      std::span<const uint8_t> bytes) {
    size_t cursor = 0;
    for (size_t index = 0; index < program.predicate_count; ++index) {
        const auto& literal = program.instructions[index].literal;
        if (literal.size() > bytes.size() - std::min(cursor, bytes.size()))
            return false;
        if (!std::equal(literal.begin(), literal.end(), bytes.begin() + cursor))
            return false;
        cursor += literal.size();
    }
    return true;
}

struct LoopFrame {
    size_t begin = 0;
    size_t end = 0;
    uint64_t remaining = 0;
};

ContainerSchemaResult execute_schema(const ContainerSchemaProgram& program,
                                     std::span<const uint8_t> bytes) {
    size_t cursor = 0;
    for (size_t index = 0; index < program.predicate_count; ++index)
        cursor += program.instructions[index].literal.size();
    std::vector<uint64_t> registers(program.register_count, 0);
    std::vector<LoopFrame> loops;
    loops.reserve(program.instructions.size());
    std::vector<ContainerRange> ranges;
    ranges.reserve(std::min<uint32_t>(program.maximum_ranges, 64));
    uint64_t steps = 0;
    uint64_t loop_iterations = 0;

    for (size_t pc = program.predicate_count; pc < program.instructions.size();) {
        if (++steps > program.maximum_steps) {
            return schema_error(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                                "container schema step limit was exceeded");
        }
        const ContainerSchemaInstruction& instruction = program.instructions[pc];
        switch (instruction.opcode) {
        case ContainerSchemaOpcode::ReadU32Le:
        case ContainerSchemaOpcode::ReadU64Le: {
            const size_t width = instruction.opcode == ContainerSchemaOpcode::ReadU32Le
                                     ? 4 : 8;
            if (cursor > bytes.size() || width > bytes.size() - cursor) {
                return schema_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                    "container schema scalar read is outside the file");
            }
            uint64_t value = 0;
            for (size_t byte = 0; byte < width; ++byte)
                value |= static_cast<uint64_t>(bytes[cursor + byte]) << (byte * 8);
            registers[instruction.destination] = value;
            cursor += width;
            ++pc;
            break;
        }
        case ContainerSchemaOpcode::SetConstant:
            registers[instruction.destination] = instruction.immediate;
            ++pc;
            break;
        case ContainerSchemaOpcode::CaptureCursor:
            registers[instruction.destination] = cursor;
            ++pc;
            break;
        case ContainerSchemaOpcode::CaptureFileSize:
            registers[instruction.destination] = bytes.size();
            ++pc;
            break;
        case ContainerSchemaOpcode::Add: {
            const uint64_t left = registers[instruction.input_a];
            const uint64_t right = registers[instruction.input_b];
            if (right > std::numeric_limits<uint64_t>::max() - left) {
                return schema_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                    "container schema addition overflowed");
            }
            registers[instruction.destination] = left + right;
            ++pc;
            break;
        }
        case ContainerSchemaOpcode::Multiply: {
            const uint64_t left = registers[instruction.input_a];
            const uint64_t right = registers[instruction.input_b];
            if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
                return schema_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                    "container schema multiplication overflowed");
            }
            registers[instruction.destination] = left * right;
            ++pc;
            break;
        }
        case ContainerSchemaOpcode::AlignCursor: {
            const uint64_t alignment = registers[instruction.input_a];
            if (alignment == 0) {
                return schema_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                    "container schema alignment is zero");
            }
            const uint64_t remainder = cursor % alignment;
            const uint64_t addition = remainder == 0 ? 0 : alignment - remainder;
            if (addition > std::numeric_limits<uint64_t>::max() - cursor ||
                cursor + addition > bytes.size()) {
                return schema_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                    "container schema alignment is outside the file");
            }
            cursor += static_cast<size_t>(addition);
            ++pc;
            break;
        }
        case ContainerSchemaOpcode::Advance:
        case ContainerSchemaOpcode::SetCursor: {
            const uint64_t value = registers[instruction.input_a];
            if (instruction.opcode == ContainerSchemaOpcode::Advance) {
                if (value > bytes.size() - cursor) {
                    return schema_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                        "container schema advance is outside the file");
                }
                cursor += static_cast<size_t>(value);
            } else {
                if (value > bytes.size()) {
                    return schema_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                        "container schema cursor is outside the file");
                }
                cursor = static_cast<size_t>(value);
            }
            ++pc;
            break;
        }
        case ContainerSchemaOpcode::EmitRange: {
            const uint64_t offset = registers[instruction.input_a];
            const uint64_t length = registers[instruction.input_b];
            if (offset > bytes.size() || length > bytes.size() - offset) {
                return schema_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                    "container schema range is outside the file");
            }
            if (ranges.size() == program.maximum_ranges) {
                return schema_error(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                                    "container schema range limit was exceeded");
            }
            ranges.push_back({instruction.section_id, offset, length});
            ++pc;
            break;
        }
        case ContainerSchemaOpcode::LoopBegin: {
            const uint64_t count = registers[instruction.input_a];
            if (count == 0) {
                pc = static_cast<size_t>(instruction.immediate) + 1;
                break;
            }
            if (count > program.maximum_loop_iterations - loop_iterations) {
                return schema_error(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                                    "container schema loop limit was exceeded");
            }
            loop_iterations += count;
            loops.push_back({pc, static_cast<size_t>(instruction.immediate), count});
            ++pc;
            break;
        }
        case ContainerSchemaOpcode::LoopEnd:
            if (--loops.back().remaining != 0)
                pc = loops.back().begin + 1;
            else {
                loops.pop_back();
                ++pc;
            }
            break;
        case ContainerSchemaOpcode::RequireCursorEnd:
            if (cursor != bytes.size()) {
                return schema_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                    "container schema did not consume the declared file");
            }
            ++pc;
            break;
        case ContainerSchemaOpcode::MatchBytes:
            return schema_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                                "container schema predicate entered execution");
        default:
            return schema_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                                "container schema opcode is unsupported");
        }
    }
    if (!loops.empty()) {
        return schema_error(CompatibilityError::IMPORT_SCHEMA_INCOMPLETE,
                            "container schema execution ended inside a loop");
    }
    std::vector<std::pair<uint64_t, uint64_t>> intervals;
    intervals.reserve(ranges.size());
    for (const ContainerRange& range : ranges) {
        if (range.length != 0)
            intervals.emplace_back(range.offset, range.offset + range.length);
    }
    std::sort(intervals.begin(), intervals.end());
    for (size_t index = 1; index < intervals.size(); ++index) {
        if (intervals[index].first < intervals[index - 1].second) {
            return schema_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                "container schema ranges overlap");
        }
    }
    return ContainerExtraction{digest_unchecked(program), std::move(ranges)};
}

} // namespace

ContainerSchemaDigestResult
container_schema_digest(const ContainerSchemaProgram& program) {
    if (auto invalid = validate_schema(program)) return std::move(*invalid);
    return digest_unchecked(program);
}

ContainerSchemaResult select_container_schema(
    std::span<const ContainerSchemaProgram> schemas,
    std::span<const uint8_t> bytes) {
    if (schemas.size() > kMaximumSchemas) {
        return schema_error(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                            "container schema set exceeds its count limit");
    }
    std::vector<const ContainerSchemaProgram*> candidates;
    candidates.reserve(std::min(schemas.size(), kMaximumSchemas));
    uint64_t selection_steps = 0;
    for (const ContainerSchemaProgram& schema : schemas) {
        if (auto invalid = validate_schema(schema)) return std::move(*invalid);
        if (!predicates_match(schema, bytes)) continue;
        if (schema.maximum_steps > kMaximumSelectionSteps - selection_steps) {
            return schema_error(CompatibilityError::IMPORT_SCHEMA_LIMIT,
                                "matching container schemas exceed the selection work limit");
        }
        selection_steps += schema.maximum_steps;
        candidates.push_back(&schema);
    }
    if (candidates.empty()) {
        return schema_error(CompatibilityError::IMPORT_SCHEMA_NOT_FOUND,
                            "no container schema program matches the file");
    }
    std::optional<ContainerExtraction> selected;
    std::optional<CompatibilityReport> single_failure;
    for (const ContainerSchemaProgram* candidate : candidates) {
        ContainerSchemaResult result = execute_schema(*candidate, bytes);
        if (auto* extraction = std::get_if<ContainerExtraction>(&result)) {
            if (selected) {
                return schema_error(
                    CompatibilityError::IMPORT_SCHEMA_AMBIGUOUS,
                    "multiple container schema programs completely match the file");
            }
            selected = std::move(*extraction);
        } else if (candidates.size() == 1) {
            single_failure = std::get<CompatibilityReport>(std::move(result));
        }
    }
    if (selected) return std::move(*selected);
    if (single_failure) return std::move(*single_failure);
    return schema_error(CompatibilityError::IMPORT_SCHEMA_NOT_FOUND,
                        "matching container schemas did not complete interpretation");
}

} // namespace Laplace
