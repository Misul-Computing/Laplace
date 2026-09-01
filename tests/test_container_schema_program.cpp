#include "container_schema_program.h"
#include "test_util.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <span>
#include <variant>
#include <vector>

using namespace Laplace;

namespace {

std::atomic<bool> track_allocations = false;
std::atomic<size_t> largest_tracked_allocation = 0;

void record_allocation(size_t size) {
    if (!track_allocations.load(std::memory_order_relaxed)) return;
    size_t prior = largest_tracked_allocation.load(std::memory_order_relaxed);
    while (prior < size &&
           !largest_tracked_allocation.compare_exchange_weak(
               prior, size, std::memory_order_relaxed)) {}
}

struct AllocationScope {
    AllocationScope() {
        largest_tracked_allocation.store(0, std::memory_order_relaxed);
        track_allocations.store(true, std::memory_order_relaxed);
    }
    ~AllocationScope() {
        track_allocations.store(false, std::memory_order_relaxed);
    }
};

ContainerSchemaInstruction instruction(ContainerSchemaOpcode opcode) {
    ContainerSchemaInstruction result;
    result.opcode = opcode;
    return result;
}

ContainerSchemaProgram first_schema() {
    ContainerSchemaProgram program;
    program.register_count = 3;
    program.predicate_count = 1;
    auto match = instruction(ContainerSchemaOpcode::MatchBytes);
    match.literal = {'A', '1'};
    auto length = instruction(ContainerSchemaOpcode::ReadU32Le);
    length.destination = 0;
    auto cursor = instruction(ContainerSchemaOpcode::CaptureCursor);
    cursor.destination = 1;
    auto emit = instruction(ContainerSchemaOpcode::EmitRange);
    emit.input_a = 1;
    emit.input_b = 0;
    emit.section_id = 7;
    auto advance = instruction(ContainerSchemaOpcode::Advance);
    advance.input_a = 0;
    program.instructions = {
        std::move(match), length, cursor, emit, advance,
        instruction(ContainerSchemaOpcode::RequireCursorEnd),
    };
    return program;
}

ContainerSchemaProgram second_schema() {
    ContainerSchemaProgram program;
    program.register_count = 4;
    program.predicate_count = 1;
    auto match = instruction(ContainerSchemaOpcode::MatchBytes);
    match.literal = {'B', '2'};
    auto offset = instruction(ContainerSchemaOpcode::ReadU64Le);
    offset.destination = 0;
    auto length = instruction(ContainerSchemaOpcode::ReadU32Le);
    length.destination = 1;
    auto seek = instruction(ContainerSchemaOpcode::SetCursor);
    seek.input_a = 0;
    auto cursor = instruction(ContainerSchemaOpcode::CaptureCursor);
    cursor.destination = 2;
    auto emit = instruction(ContainerSchemaOpcode::EmitRange);
    emit.input_a = 2;
    emit.input_b = 1;
    emit.section_id = 7;
    auto advance = instruction(ContainerSchemaOpcode::Advance);
    advance.input_a = 1;
    program.instructions = {
        std::move(match), offset, length, seek, cursor, emit, advance,
        instruction(ContainerSchemaOpcode::RequireCursorEnd),
    };
    return program;
}

std::vector<uint8_t> first_container(uint32_t declared_length = 3) {
    return {
        'A', '1',
        static_cast<uint8_t>(declared_length),
        static_cast<uint8_t>(declared_length >> 8),
        static_cast<uint8_t>(declared_length >> 16),
        static_cast<uint8_t>(declared_length >> 24),
        'x', 'y', 'z',
    };
}

std::vector<uint8_t> second_container() {
    constexpr uint64_t offset = 14;
    std::vector<uint8_t> bytes = {'B', '2'};
    for (unsigned shift = 0; shift != 64; shift += 8)
        bytes.push_back(static_cast<uint8_t>(offset >> shift));
    bytes.insert(bytes.end(), {3, 0, 0, 0, 'x', 'y', 'z'});
    return bytes;
}

const CompatibilityReport* report(const ContainerSchemaResult& result) {
    return std::get_if<CompatibilityReport>(&result);
}

void test_two_presentations_extract_equivalent_sections() {
    const std::array<ContainerSchemaProgram, 2> schemas = {
        first_schema(), second_schema()};
    const auto first = select_container_schema(schemas, first_container());
    const auto second = select_container_schema(schemas, second_container());
    CHECK(std::holds_alternative<ContainerExtraction>(first));
    CHECK(std::holds_alternative<ContainerExtraction>(second));
    if (!std::holds_alternative<ContainerExtraction>(first) ||
        !std::holds_alternative<ContainerExtraction>(second))
        return;
    const auto& left = std::get<ContainerExtraction>(first);
    const auto& right = std::get<ContainerExtraction>(second);
    CHECK(left.ranges.size() == 1);
    CHECK(right.ranges.size() == 1);
    CHECK(left.ranges[0].section_id == right.ranges[0].section_id);
    const auto first_bytes = first_container();
    const auto second_bytes = second_container();
    const auto left_payload = std::span<const uint8_t>(first_bytes).subspan(
        left.ranges[0].offset, left.ranges[0].length);
    const auto right_payload = std::span<const uint8_t>(second_bytes).subspan(
        right.ranges[0].offset, right.ranges[0].length);
    CHECK(std::equal(left_payload.begin(), left_payload.end(),
                     right_payload.begin(), right_payload.end()));
    CHECK(left.schema_digest != right.schema_digest);
}

void test_schema_selection_is_exact() {
    const std::array<ContainerSchemaProgram, 2> schemas = {
        first_schema(), second_schema()};
    const std::array<uint8_t, 2> unknown = {'C', '3'};
    const auto missing = select_container_schema(schemas, unknown);
    CHECK(report(missing) != nullptr);
    if (report(missing))
        CHECK(report(missing)->code == CompatibilityError::IMPORT_SCHEMA_NOT_FOUND);

    const std::array<ContainerSchemaProgram, 2> ambiguous = {
        first_schema(), first_schema()};
    const auto multiple = select_container_schema(ambiguous, first_container());
    CHECK(report(multiple) != nullptr);
    if (report(multiple))
        CHECK(report(multiple)->code == CompatibilityError::IMPORT_SCHEMA_AMBIGUOUS);

    ContainerSchemaProgram incomplete = first_schema();
    auto wide_read = instruction(ContainerSchemaOpcode::ReadU64Le);
    wide_read.destination = 2;
    incomplete.instructions.insert(incomplete.instructions.begin() + 1,
                                   wide_read);
    const std::array<ContainerSchemaProgram, 2> one_complete = {
        first_schema(), std::move(incomplete)};
    const auto complete = select_container_schema(one_complete, first_container());
    CHECK(std::holds_alternative<ContainerExtraction>(complete));

    std::vector<ContainerSchemaProgram> oversized(1025, second_schema());
    const auto excessive = select_container_schema(oversized, unknown);
    CHECK(report(excessive) != nullptr);
    if (report(excessive))
        CHECK(report(excessive)->code == CompatibilityError::IMPORT_SCHEMA_LIMIT);

    ContainerSchemaProgram prefix_only = first_schema();
    prefix_only.instructions.pop_back();
    auto trailing_input = first_container();
    trailing_input.insert(trailing_input.end(), {'T', 'R', 'A', 'I', 'L'});
    const std::array<ContainerSchemaProgram, 1> prefix_set = {
        std::move(prefix_only)};
    const auto prefix_result = select_container_schema(prefix_set, trailing_input);
    CHECK(report(prefix_result) != nullptr);
    if (report(prefix_result))
        CHECK(report(prefix_result)->code ==
              CompatibilityError::PACKAGE_BOUNDS_INVALID);
}

void test_schema_program_is_canonical_and_explicit() {
    ContainerSchemaProgram unused = first_schema();
    unused.instructions.back().destination = 0;
    CHECK(std::holds_alternative<CompatibilityReport>(
        container_schema_digest(unused)));

    ContainerSchemaProgram implicit_zero;
    implicit_zero.register_count = 1;
    implicit_zero.predicate_count = 1;
    auto match = instruction(ContainerSchemaOpcode::MatchBytes);
    match.literal = {'D'};
    auto advance = instruction(ContainerSchemaOpcode::Advance);
    advance.input_a = 0;
    implicit_zero.instructions = {
        std::move(match), advance,
        instruction(ContainerSchemaOpcode::RequireCursorEnd),
    };
    const std::array<ContainerSchemaProgram, 1> schemas = {implicit_zero};
    const std::array<uint8_t, 1> bytes = {'D'};
    const auto result = select_container_schema(schemas, bytes);
    CHECK(report(result) != nullptr);
    if (report(result))
        CHECK(report(result)->code == CompatibilityError::IMPORT_SCHEMA_INCOMPLETE);
}

ContainerSchemaProgram repeated_schema(uint64_t maximum_iterations = 4) {
    ContainerSchemaProgram program;
    program.register_count = 3;
    program.predicate_count = 1;
    program.maximum_loop_iterations = maximum_iterations;
    auto match = instruction(ContainerSchemaOpcode::MatchBytes);
    match.literal = {'R', '1'};
    auto count = instruction(ContainerSchemaOpcode::ReadU32Le);
    count.destination = 0;
    auto begin = instruction(ContainerSchemaOpcode::LoopBegin);
    begin.input_a = 0;
    begin.immediate = 7;
    auto length = instruction(ContainerSchemaOpcode::ReadU32Le);
    length.destination = 1;
    auto cursor = instruction(ContainerSchemaOpcode::CaptureCursor);
    cursor.destination = 2;
    auto emit = instruction(ContainerSchemaOpcode::EmitRange);
    emit.input_a = 2;
    emit.input_b = 1;
    emit.section_id = 9;
    auto advance = instruction(ContainerSchemaOpcode::Advance);
    advance.input_a = 1;
    auto end = instruction(ContainerSchemaOpcode::LoopEnd);
    end.immediate = 2;
    program.instructions = {
        std::move(match), count, begin, length, cursor, emit, advance, end,
        instruction(ContainerSchemaOpcode::RequireCursorEnd),
    };
    return program;
}

std::vector<uint8_t> repeated_container(uint32_t count = 2) {
    std::vector<uint8_t> bytes = {
        'R', '1', static_cast<uint8_t>(count), 0, 0, 0};
    if (count >= 1) bytes.insert(bytes.end(), {2, 0, 0, 0, 'a', 'b'});
    if (count >= 2) bytes.insert(bytes.end(), {3, 0, 0, 0, 'c', 'd', 'e'});
    return bytes;
}

void test_repetition_and_limits() {
    const std::array<ContainerSchemaProgram, 1> schema = {repeated_schema()};
    const auto result = select_container_schema(schema, repeated_container());
    CHECK(std::holds_alternative<ContainerExtraction>(result));
    if (const auto* extraction = std::get_if<ContainerExtraction>(&result)) {
        CHECK(extraction->ranges.size() == 2);
        CHECK(extraction->ranges[0].length == 2);
        CHECK(extraction->ranges[1].length == 3);
    }

    const std::array<ContainerSchemaProgram, 1> limited = {repeated_schema(1)};
    const auto over_limit = select_container_schema(limited, repeated_container());
    CHECK(report(over_limit) != nullptr);
    if (report(over_limit))
        CHECK(report(over_limit)->code == CompatibilityError::IMPORT_SCHEMA_LIMIT);
}

void test_malformed_arithmetic_and_ranges_fail_closed() {
    const std::array<ContainerSchemaProgram, 1> schema = {first_schema()};
    const auto short_range = select_container_schema(schema, first_container(5));
    CHECK(report(short_range) != nullptr);
    if (report(short_range))
        CHECK(report(short_range)->code == CompatibilityError::PACKAGE_BOUNDS_INVALID);

    ContainerSchemaProgram overflow;
    overflow.register_count = 3;
    overflow.predicate_count = 1;
    auto match = instruction(ContainerSchemaOpcode::MatchBytes);
    match.literal = {'O', '1'};
    auto maximum = instruction(ContainerSchemaOpcode::SetConstant);
    maximum.destination = 0;
    maximum.immediate = std::numeric_limits<uint64_t>::max();
    auto one = instruction(ContainerSchemaOpcode::SetConstant);
    one.destination = 1;
    one.immediate = 1;
    auto add = instruction(ContainerSchemaOpcode::Add);
    add.destination = 2;
    add.input_a = 0;
    add.input_b = 1;
    overflow.instructions = {std::move(match), maximum, one, add};
    const std::array<ContainerSchemaProgram, 1> overflow_set = {overflow};
    const std::array<uint8_t, 2> overflow_input = {'O', '1'};
    const auto overflow_result = select_container_schema(overflow_set, overflow_input);
    CHECK(report(overflow_result) != nullptr);
    if (report(overflow_result))
        CHECK(report(overflow_result)->code == CompatibilityError::PACKAGE_BOUNDS_INVALID);

    ContainerSchemaProgram overlap = first_schema();
    overlap.instructions.insert(overlap.instructions.end() - 2,
                                overlap.instructions[3]);
    const std::array<ContainerSchemaProgram, 1> overlap_set = {overlap};
    const auto overlap_result = select_container_schema(overlap_set, first_container());
    CHECK(report(overlap_result) != nullptr);
    if (report(overlap_result))
        CHECK(report(overlap_result)->code == CompatibilityError::PACKAGE_BOUNDS_INVALID);

    ContainerSchemaProgram malformed = repeated_schema();
    malformed.instructions[7].immediate = 1;
    const std::array<ContainerSchemaProgram, 1> malformed_set = {malformed};
    const auto malformed_result = select_container_schema(malformed_set, repeated_container());
    CHECK(report(malformed_result) != nullptr);
    if (report(malformed_result))
        CHECK(report(malformed_result)->code == CompatibilityError::IMPORT_SCHEMA_INCOMPLETE);
}

void test_schema_digest_binds_program() {
    ContainerSchemaProgram first = first_schema();
    ContainerSchemaProgram changed = first;
    changed.instructions[3].section_id = 8;
    const auto first_digest = container_schema_digest(first);
    const auto changed_digest = container_schema_digest(changed);
    CHECK(std::holds_alternative<ContainerSchemaDigest>(first_digest));
    CHECK(std::holds_alternative<ContainerSchemaDigest>(changed_digest));
    if (std::holds_alternative<ContainerSchemaDigest>(first_digest) &&
        std::holds_alternative<ContainerSchemaDigest>(changed_digest)) {
        CHECK(std::get<ContainerSchemaDigest>(first_digest) !=
              std::get<ContainerSchemaDigest>(changed_digest));
    }
}

void test_schema_set_wire_is_canonical_and_bounded() {
    const std::array<ContainerSchemaProgram, 2> schemas = {
        second_schema(), first_schema()};
    const auto encoded = encode_container_schema_set(schemas);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(encoded));
    if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) return;
    const auto& bytes = std::get<std::vector<uint8_t>>(encoded);

    const std::array<ContainerSchemaProgram, 2> reversed = {
        first_schema(), second_schema()};
    const auto reordered = encode_container_schema_set(reversed);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(reordered));
    if (const auto* reordered_bytes =
            std::get_if<std::vector<uint8_t>>(&reordered))
        CHECK(*reordered_bytes == bytes);

    const auto decoded = decode_container_schema_set(bytes);
    CHECK(std::holds_alternative<std::vector<ContainerSchemaProgram>>(decoded));
    if (const auto* programs =
            std::get_if<std::vector<ContainerSchemaProgram>>(&decoded)) {
        CHECK(programs->size() == 2);
        const auto roundtrip = encode_container_schema_set(*programs);
        CHECK(std::holds_alternative<std::vector<uint8_t>>(roundtrip));
        if (const auto* roundtrip_bytes =
                std::get_if<std::vector<uint8_t>>(&roundtrip))
            CHECK(*roundtrip_bytes == bytes);
    }

    const auto rejected = [](std::vector<uint8_t> wire) {
        return std::holds_alternative<CompatibilityReport>(
            decode_container_schema_set(wire));
    };
    auto bad_magic = bytes;
    bad_magic[0] ^= 1;
    CHECK(rejected(std::move(bad_magic)));
    auto bad_version = bytes;
    bad_version[8] = 2;
    CHECK(rejected(std::move(bad_version)));
    auto bad_reserved = bytes;
    bad_reserved[10] = 1;
    CHECK(rejected(std::move(bad_reserved)));
    auto bad_total = bytes;
    bad_total[12] ^= 1;
    CHECK(rejected(std::move(bad_total)));
    auto too_many = bytes;
    too_many[16] = 1;
    too_many[17] = 4;
    too_many[18] = 0;
    too_many[19] = 0;
    CHECK(rejected(std::move(too_many)));
    auto bad_header_reserved = bytes;
    bad_header_reserved[20] = 1;
    CHECK(rejected(std::move(bad_header_reserved)));
    auto bad_record_length = bytes;
    std::fill(bad_record_length.begin() + 24,
              bad_record_length.begin() + 28, 0xff);
    CHECK(rejected(std::move(bad_record_length)));
    auto bad_instruction_reserved = bytes;
    bad_instruction_reserved[65] = 1;
    CHECK(rejected(std::move(bad_instruction_reserved)));
    auto bad_literal_length = bytes;
    std::fill(bad_literal_length.begin() + 92,
              bad_literal_length.begin() + 96, 0xff);
    CHECK(rejected(std::move(bad_literal_length)));

    auto impossible_instruction_count = bytes;
    impossible_instruction_count[60] = 0;
    impossible_instruction_count[61] = 16;
    impossible_instruction_count[62] = 0;
    impossible_instruction_count[63] = 0;
    {
        AllocationScope scope;
        CHECK(rejected(std::move(impossible_instruction_count)));
    }
    CHECK(largest_tracked_allocation.load(std::memory_order_relaxed) <
          64 * 1024);
    const auto read_u32 = [](const std::vector<uint8_t>& value, size_t offset) {
        uint32_t result = 0;
        for (unsigned shift = 0; shift != 32; shift += 8)
            result |= static_cast<uint32_t>(value[offset + shift / 8]) << shift;
        return result;
    };
    const uint32_t first_record_length = read_u32(bytes, 24);
    auto bad_program_digest = bytes;
    bad_program_digest[28 + first_record_length - 1] ^= 1;
    CHECK(rejected(std::move(bad_program_digest)));
    const size_t second_record = 28 + first_record_length;
    const uint32_t second_record_length = read_u32(bytes, second_record);
    const size_t records_end = second_record + 4 + second_record_length;
    std::vector<uint8_t> wrong_order(bytes.begin(), bytes.begin() + 24);
    wrong_order.insert(wrong_order.end(), bytes.begin() + second_record,
                       bytes.begin() + records_end);
    wrong_order.insert(wrong_order.end(), bytes.begin() + 24,
                       bytes.begin() + second_record);
    wrong_order.insert(wrong_order.end(), bytes.begin() + records_end,
                       bytes.end());
    CHECK(rejected(std::move(wrong_order)));
    auto bad_digest = bytes;
    bad_digest.back() ^= 1;
    CHECK(rejected(std::move(bad_digest)));
    auto trailing = bytes;
    trailing.push_back(0);
    CHECK(rejected(std::move(trailing)));
    auto truncated = bytes;
    truncated.pop_back();
    CHECK(rejected(std::move(truncated)));
    for (size_t length = 0; length < bytes.size(); ++length)
        CHECK(rejected(std::vector<uint8_t>(bytes.begin(),
                                            bytes.begin() + length)));

    const std::array<ContainerSchemaProgram, 2> duplicate = {
        first_schema(), first_schema()};
    CHECK(std::holds_alternative<CompatibilityReport>(
        encode_container_schema_set(duplicate)));
    CHECK(std::holds_alternative<CompatibilityReport>(
        encode_container_schema_set(
            std::span<const ContainerSchemaProgram>{})));

    ContainerSchemaProgram large = first_schema();
    large.instructions.front().literal.assign(65'536, 'A');
    std::vector<ContainerSchemaProgram> over_wire_limit;
    over_wire_limit.reserve(512);
    for (uint32_t index = 0; index < 512; ++index) {
        ContainerSchemaProgram item = large;
        item.instructions[3].section_id = index;
        over_wire_limit.push_back(std::move(item));
    }
    ContainerSchemaWireResult oversized_wire;
    {
        AllocationScope scope;
        oversized_wire = encode_container_schema_set(over_wire_limit);
    }
    CHECK(std::holds_alternative<CompatibilityReport>(oversized_wire));
    CHECK(largest_tracked_allocation.load(std::memory_order_relaxed) <
          64 * 1024);
}

void test_maximum_ranges_remain_bounded() {
    ContainerSchemaProgram program;
    program.register_count = 4;
    program.predicate_count = 1;
    auto match = instruction(ContainerSchemaOpcode::MatchBytes);
    match.literal = {'Z'};
    auto count = instruction(ContainerSchemaOpcode::SetConstant);
    count.destination = 0;
    count.immediate = 65'536;
    auto offset = instruction(ContainerSchemaOpcode::SetConstant);
    offset.destination = 1;
    offset.immediate = 1;
    auto length = instruction(ContainerSchemaOpcode::SetConstant);
    length.destination = 2;
    length.immediate = 1;
    auto one = instruction(ContainerSchemaOpcode::SetConstant);
    one.destination = 3;
    one.immediate = 1;
    auto begin = instruction(ContainerSchemaOpcode::LoopBegin);
    begin.input_a = 0;
    begin.immediate = 8;
    auto emit = instruction(ContainerSchemaOpcode::EmitRange);
    emit.input_a = 1;
    emit.input_b = 2;
    auto increment = instruction(ContainerSchemaOpcode::Add);
    increment.destination = 1;
    increment.input_a = 1;
    increment.input_b = 3;
    auto end = instruction(ContainerSchemaOpcode::LoopEnd);
    end.immediate = 5;
    auto seek = instruction(ContainerSchemaOpcode::SetCursor);
    seek.input_a = 1;
    program.instructions = {
        std::move(match), count, offset, length, one, begin, emit, increment,
        end, seek,
        instruction(ContainerSchemaOpcode::RequireCursorEnd),
    };
    const std::array<ContainerSchemaProgram, 1> schemas = {std::move(program)};
    std::vector<uint8_t> bytes(65'537, 0);
    bytes[0] = 'Z';
    const auto result = select_container_schema(schemas, bytes);
    CHECK(std::holds_alternative<ContainerExtraction>(result));
    if (const auto* extraction = std::get_if<ContainerExtraction>(&result))
        CHECK(extraction->ranges.size() == 65'536);
}

} // namespace

void* operator new(std::size_t size) {
    record_allocation(size);
    if (void* value = std::malloc(size == 0 ? 1 : size)) return value;
    throw std::bad_alloc();
}

void operator delete(void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }

int main() {
    test_two_presentations_extract_equivalent_sections();
    test_schema_selection_is_exact();
    test_schema_program_is_canonical_and_explicit();
    test_repetition_and_limits();
    test_malformed_arithmetic_and_ranges_fail_closed();
    test_schema_digest_binds_program();
    test_schema_set_wire_is_canonical_and_bounded();
    test_maximum_ranges_remain_bounded();
    return test_summary("test_container_schema_program");
}
