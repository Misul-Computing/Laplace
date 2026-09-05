#include "physical_program_package.h"
#include "test_util.h"

#include <bit>
#include <array>
#include <cstdint>
#include <span>
#include <type_traits>
#include <variant>
#include <vector>

using namespace Laplace;

namespace {

PhysicalProgram load_program() {
    PhysicalProgram p;
    p.planes.push_back({PhysicalPlaneStorage::External, 1, 0, 0});
    p.instructions.push_back({PhysicalOpcode::ConstIndex, PhysicalValueType::Index,
                              {kNoPhysicalValue, kNoPhysicalValue, kNoPhysicalValue},
                              kNoPhysicalPlane, kNoPhysicalPolicy, 0, 0,
                              PhysicalBitOrder::Lsb0Little});
    p.instructions.push_back({PhysicalOpcode::LoadBits, PhysicalValueType::U32,
                              {0, kNoPhysicalValue, kNoPhysicalValue}, 0,
                              kNoPhysicalPolicy, 0, 8,
                              PhysicalBitOrder::Lsb0Little});
    p.result = 1;
    return p;
}

Program semantic_program() {
    Region region;
    region.id = 0;
    region.arguments = {{0, {ElementType::U32, {}}}};
    region.yields = {0};
    Function function;
    function.id = 0;
    function.entry_region_id = 0;
    function.regions = {region};
    function.result_types = {{ElementType::U32, {}}};
    Program program;
    program.functions = {function};
    program.exports = {{0, 0, {ElementType::U32, {}}}};
    return program;
}

Program two_input_semantic_program() {
    const ValueType u32{ElementType::U32, {}};
    Region region;
    region.id = 0;
    region.arguments = {{0, u32}, {1, u32}};
    region.yields = {0};
    Function function;
    function.id = 0;
    function.entry_region_id = 0;
    function.regions = {region};
    function.result_types = {u32};
    Program program;
    program.functions = {function};
    program.exports = {{0, 0, u32}};
    return program;
}

ArtifactIndex make_index(ArtifactId id, std::span<const uint8_t> bytes) {
    auto view_result = ArtifactSet::make_owned_blob(id, ArtifactRole::Primary, bytes);
    CHECK(std::holds_alternative<PackageView>(view_result));
    if (!std::holds_alternative<PackageView>(view_result)) return {};
    ArtifactIndexInput input;
    input.artifacts.push_back(std::get<PackageView>(std::move(view_result)));
    auto index_result = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<ArtifactIndex>(index_result));
    if (!std::holds_alternative<ArtifactIndex>(index_result)) return {};
    return std::get<ArtifactIndex>(std::move(index_result));
}

ArtifactTensorRecord alias_tensor(uint32_t id, ArtifactId artifact) {
    ArtifactTensorRecord tensor;
    tensor.id = id;
    tensor.logical_type = ArtifactScalarType::F32;
    tensor.logical_dimensions = {1};
    tensor.axis.source_rank = 1;
    tensor.axis.source_axis_order[0] = 0;
    tensor.layout.rank = 1;
    tensor.layout.axis_order[0] = 0;
    tensor.layout.strides[0] = 1;
    tensor.planes.push_back({PlaneKind::Values, ArtifactScalarType::F32,
                             {artifact, 0, 4}, 1, 4, 1, 4});
    return tensor;
}

ArtifactIndex make_alias_index(ArtifactId artifact) {
    const std::array<uint8_t, 8> bytes = {0x5a, 0xa5, 0, 0, 0, 0, 0, 0};
    auto view_result = ArtifactSet::make_owned_blob(artifact, ArtifactRole::Primary, bytes);
    CHECK(std::holds_alternative<PackageView>(view_result));
    if (!std::holds_alternative<PackageView>(view_result)) return {};
    ArtifactIndexInput input;
    input.artifacts.push_back(std::get<PackageView>(std::move(view_result)));
    auto source = alias_tensor(10, artifact);
    auto target = alias_tensor(20, artifact);
    target.role_evidence.push_back({TensorRole::OutputWeight, {0}, ArtifactFactAuthority::Structural, {}});
    input.tensors = {std::move(source), std::move(target)};
    input.aliases.push_back({ArtifactAliasKind::TiedOutput,
                             ArtifactAliasDirection::SourceToTarget, 10, 20,
                             TensorRole::OutputWeight, {0}, ArtifactFactAuthority::Structural, {}});
    auto index_result = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<ArtifactIndex>(index_result));
    if (!std::holds_alternative<ArtifactIndex>(index_result)) return {};
    return std::get<ArtifactIndex>(std::move(index_result));
}

void package_accepts_canonical_record() {
    const std::vector<uint8_t> bytes = {0x5a, 0xa5};
    PhysicalProgram program = load_program();
    const auto wire_result = encode_physical_program(program);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(wire_result));
    if (!std::holds_alternative<std::vector<uint8_t>>(wire_result)) return;
    const auto digest_result = physical_program_digest(program);
    CHECK(std::holds_alternative<PhysicalProgramDigest>(digest_result));
    if (!std::holds_alternative<PhysicalProgramDigest>(digest_result)) return;
    auto semantic_result = verify_and_canonicalize_program(semantic_program());
    CHECK(std::holds_alternative<VerifiedProgram>(semantic_result));
    if (!std::holds_alternative<VerifiedProgram>(semantic_result)) return;
    const ArtifactId artifact{7};
    ArtifactIndex index = make_index(artifact, bytes);
    PhysicalProgramRecord record{std::get<PhysicalProgramDigest>(digest_result),
                                 std::get<std::vector<uint8_t>>(wire_result),
                                 {ElementType::U32, {}}};
    PhysicalResourceBinding binding;
    binding.resource_id = 3;
    binding.program_digest = record.digest;
    binding.semantic_function_id = 0;
    binding.semantic_value_id = 0;
    binding.planes.push_back({0, artifact, 0, bytes.size()});
    const std::array<PhysicalProgramRecord, 1> records = {record};
    const std::array<PhysicalResourceBinding, 1> bindings = {binding};
    const auto package_wire = encode_physical_program_package_records(records, bindings);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(package_wire));
    if (std::holds_alternative<std::vector<uint8_t>>(package_wire)) {
        const auto& encoded = std::get<std::vector<uint8_t>>(package_wire);
        const auto decoded = decode_physical_program_package_records(encoded);
        CHECK(std::holds_alternative<PhysicalProgramPackageRecords>(decoded));
        if (std::holds_alternative<PhysicalProgramPackageRecords>(decoded)) {
            CHECK(std::get<PhysicalProgramPackageRecords>(decoded).first ==
                  std::vector<PhysicalProgramRecord>(records.begin(), records.end()));
            CHECK(std::get<PhysicalProgramPackageRecords>(decoded).second ==
                  std::vector<PhysicalResourceBinding>(bindings.begin(), bindings.end()));
        }
        auto trailing = encoded;
        trailing.push_back(0);
        CHECK(std::holds_alternative<CompatibilityReport>(
            decode_physical_program_package_records(trailing)));
        auto reserved = encoded;
        reserved[10] = 1;
        CHECK(std::holds_alternative<CompatibilityReport>(
            decode_physical_program_package_records(reserved)));
    }
    const auto result = load_physical_program_package(
        std::move(index), std::get<VerifiedProgram>(std::move(semantic_result)),
        records, bindings);
    CHECK(std::holds_alternative<VerifiedPhysicalProgramPackage>(result));
    if (std::holds_alternative<VerifiedPhysicalProgramPackage>(result)) {
        const auto& package = std::get<VerifiedPhysicalProgramPackage>(result);
        CHECK(package.programs().size() == 1);
        CHECK(package.program_logical_types().size() == 1);
        CHECK(package.program_logical_types()[0].element_type == ElementType::U32);
        CHECK(package.resources().size() == 1);
        CHECK(package.resources()[0].semantic_function_id == 0);
        CHECK(package.resources()[0].semantic_value_id == 0);
        CHECK(package.resources()[0].planes[0].artifact_id == artifact);
        CHECK(package.digest().bytes != Sha256Digest{}.bytes);
        CHECK(package.find_program(record.digest) != nullptr);
        PhysicalProgramDigest missing = record.digest;
        missing[0] ^= 1;
        CHECK(package.find_program(missing) == nullptr);
        const auto resolved = package.resolve_resource(3);
        CHECK(std::holds_alternative<std::vector<PhysicalResourcePlaneView>>(resolved));
        if (std::holds_alternative<std::vector<PhysicalResourcePlaneView>>(resolved)) {
            const auto& views = std::get<std::vector<PhysicalResourcePlaneView>>(resolved);
            CHECK(views.size() == 1);
            CHECK(views[0].bytes.size() == bytes.size());
            CHECK(views[0].bytes[0] == bytes[0]);
        }

        auto semantic_again = verify_and_canonicalize_program(semantic_program());
        ArtifactIndex shifted_index = make_index(artifact, bytes);
        PhysicalResourceBinding shifted = binding;
        shifted.planes[0].offset = 1;
        shifted.planes[0].length = 1;
        const auto shifted_result = load_physical_program_package(
            std::move(shifted_index),
            std::get<VerifiedProgram>(std::move(semantic_again)), records,
            std::span<const PhysicalResourceBinding>(&shifted, 1));
        CHECK(std::holds_alternative<VerifiedPhysicalProgramPackage>(shifted_result));
        if (std::holds_alternative<VerifiedPhysicalProgramPackage>(shifted_result))
            CHECK(std::get<VerifiedPhysicalProgramPackage>(shifted_result).digest() !=
                  package.digest());

        auto unlinked_semantic = verify_and_canonicalize_program(semantic_program());
        ArtifactIndex unlinked_index = make_index(artifact, bytes);
        PhysicalResourceBinding unlinked = binding;
        unlinked.semantic_function_id = UINT32_MAX;
        unlinked.semantic_value_id = UINT32_MAX;
        const auto unlinked_result = load_physical_program_package(
            std::move(unlinked_index),
            std::get<VerifiedProgram>(std::move(unlinked_semantic)), records,
            std::span<const PhysicalResourceBinding>(&unlinked, 1));
        CHECK(std::holds_alternative<CompatibilityReport>(unlinked_result));
        if (const auto* report = std::get_if<CompatibilityReport>(&unlinked_result))
            CHECK(report->code == CompatibilityError::IR_REFERENCE_INVALID);

        auto semantic_link_again = verify_and_canonicalize_program(semantic_program());
        ArtifactIndex link_index = make_index(artifact, bytes);
        PhysicalResourceBinding unknown_link = binding;
        unknown_link.semantic_value_id = 99;
        const auto unknown_link_result = load_physical_program_package(
            std::move(link_index), std::get<VerifiedProgram>(std::move(semantic_link_again)),
            records, std::span<const PhysicalResourceBinding>(&unknown_link, 1));
        CHECK(std::holds_alternative<CompatibilityReport>(unknown_link_result));
        if (const auto* report = std::get_if<CompatibilityReport>(&unknown_link_result))
            CHECK(report->code == CompatibilityError::IR_REFERENCE_INVALID);

        auto wrong_function_semantic = verify_and_canonicalize_program(semantic_program());
        ArtifactIndex wrong_function_index = make_index(artifact, bytes);
        PhysicalResourceBinding wrong_function = binding;
        wrong_function.semantic_function_id = 99;
        const auto wrong_function_result = load_physical_program_package(
            std::move(wrong_function_index),
            std::get<VerifiedProgram>(std::move(wrong_function_semantic)), records,
            std::span<const PhysicalResourceBinding>(&wrong_function, 1));
        CHECK(std::holds_alternative<CompatibilityReport>(wrong_function_result));
        if (const auto* report = std::get_if<CompatibilityReport>(&wrong_function_result))
            CHECK(report->code == CompatibilityError::IR_REFERENCE_INVALID);

        auto wrong_type_semantic = verify_and_canonicalize_program(semantic_program());
        ArtifactIndex wrong_type_index = make_index(artifact, bytes);
        PhysicalProgramRecord wrong_type_record = record;
        wrong_type_record.logical_type.element_type = ElementType::F32;
        const std::array<PhysicalProgramRecord, 1> wrong_type_records = {
            wrong_type_record};
        PhysicalResourceBinding wrong_type = binding;
        wrong_type.program_digest = wrong_type_record.digest;
        const auto wrong_type_result = load_physical_program_package(
            std::move(wrong_type_index),
            std::get<VerifiedProgram>(std::move(wrong_type_semantic)),
            wrong_type_records,
            std::span<const PhysicalResourceBinding>(&wrong_type, 1));
        CHECK(std::holds_alternative<CompatibilityReport>(wrong_type_result));
        if (const auto* report = std::get_if<CompatibilityReport>(&wrong_type_result))
            CHECK(report->code == CompatibilityError::IR_SHAPE_MISMATCH);

        auto duplicate_semantic = verify_and_canonicalize_program(semantic_program());
        ArtifactIndex duplicate_index = make_index(artifact, bytes);
        PhysicalResourceBinding first_duplicate = binding;
        first_duplicate.planes.front().length = 1;
        PhysicalResourceBinding duplicate = first_duplicate;
        duplicate.resource_id = binding.resource_id + 1;
        duplicate.planes.front().offset = 1;
        const std::array<PhysicalResourceBinding, 2> duplicate_bindings = {
            first_duplicate, duplicate};
        const auto duplicate_result = load_physical_program_package(
            std::move(duplicate_index),
            std::get<VerifiedProgram>(std::move(duplicate_semantic)), records,
            duplicate_bindings);
        CHECK(std::holds_alternative<CompatibilityReport>(duplicate_result));
        if (const auto* report = std::get_if<CompatibilityReport>(&duplicate_result))
            CHECK_MSG(report->code == CompatibilityError::IR_REFERENCE_INVALID,
                      "duplicate link code=%u detail=%s",
                      static_cast<unsigned>(report->code), report->detail.c_str());
    }
}

void package_rejects_bad_source_and_digest() {
    const std::vector<uint8_t> bytes = {0x5a};
    PhysicalProgram program = load_program();
    const auto wire_result = encode_physical_program(program);
    const auto digest_result = physical_program_digest(program);
    auto semantic_result = verify_and_canonicalize_program(semantic_program());
    CHECK(std::holds_alternative<VerifiedProgram>(semantic_result));
    if (!std::holds_alternative<std::vector<uint8_t>>(wire_result) ||
        !std::holds_alternative<PhysicalProgramDigest>(digest_result) ||
        !std::holds_alternative<VerifiedProgram>(semantic_result)) return;
    ArtifactId artifact{8};
    ArtifactIndex index = make_index(artifact, bytes);
    PhysicalProgramRecord record{std::get<PhysicalProgramDigest>(digest_result),
                                 std::get<std::vector<uint8_t>>(wire_result),
                                 {ElementType::U32, {}}};
    PhysicalResourceBinding binding;
    binding.resource_id = 1;
    binding.program_digest = record.digest;
    binding.semantic_function_id = 0;
    binding.semantic_value_id = 0;
    binding.planes = {{0, artifact, 1, 1}};
    const auto bad_range = load_physical_program_package(
        std::move(index), std::get<VerifiedProgram>(std::move(semantic_result)),
        std::span<const PhysicalProgramRecord>(&record, 1),
        std::span<const PhysicalResourceBinding>(&binding, 1));
    CHECK(std::holds_alternative<CompatibilityReport>(bad_range));

    auto semantic_again = verify_and_canonicalize_program(semantic_program());
    ArtifactIndex index_again = make_index(artifact, bytes);
    record.digest[0] ^= 1;
    const auto bad_digest = load_physical_program_package(
        std::move(index_again), std::get<VerifiedProgram>(std::move(semantic_again)),
        std::span<const PhysicalProgramRecord>(&record, 1), {});
    CHECK(std::holds_alternative<CompatibilityReport>(bad_digest));
}

void package_rejects_overlapping_resource_ranges() {
    const std::array<uint8_t, 2> bytes = {0x5a, 0xa5};
    const PhysicalProgram program = load_program();
    const auto wire = encode_physical_program(program);
    const auto digest = physical_program_digest(program);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
    CHECK(std::holds_alternative<PhysicalProgramDigest>(digest));
    if (!std::holds_alternative<std::vector<uint8_t>>(wire) ||
        !std::holds_alternative<PhysicalProgramDigest>(digest))
        return;
    const ArtifactId artifact{9};
    PhysicalProgramRecord record{std::get<PhysicalProgramDigest>(digest),
                                 std::get<std::vector<uint8_t>>(wire),
                                 {ElementType::U32, {}}};
    PhysicalResourceBinding first;
    first.resource_id = 0;
    first.program_digest = record.digest;
    first.planes = {{0, artifact, 0, 1}};
    first.semantic_function_id = 0;
    first.semantic_value_id = 0;
    PhysicalResourceBinding second = first;
    second.resource_id = 1;
    second.semantic_value_id = 1;
    const std::array<PhysicalResourceBinding, 2> overlapping = {first, second};

    auto semantic = verify_and_canonicalize_program(two_input_semantic_program());
    CHECK(std::holds_alternative<VerifiedProgram>(semantic));
    if (!std::holds_alternative<VerifiedProgram>(semantic)) return;
    auto rejected = load_physical_program_package(
        make_index(artifact, bytes),
        std::get<VerifiedProgram>(std::move(semantic)),
        std::span<const PhysicalProgramRecord>(&record, 1), overlapping);
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    if (const auto* report = std::get_if<CompatibilityReport>(&rejected))
        CHECK(report->code == CompatibilityError::PACKAGE_BOUNDS_INVALID);

    second.planes.front().offset = 1;
    const std::array<PhysicalResourceBinding, 2> adjacent = {first, second};
    auto semantic_again = verify_and_canonicalize_program(two_input_semantic_program());
    CHECK(std::holds_alternative<VerifiedProgram>(semantic_again));
    if (!std::holds_alternative<VerifiedProgram>(semantic_again)) return;
    auto accepted = load_physical_program_package(
        make_index(artifact, bytes),
        std::get<VerifiedProgram>(std::move(semantic_again)),
        std::span<const PhysicalProgramRecord>(&record, 1), adjacent);
    CHECK(std::holds_alternative<VerifiedPhysicalProgramPackage>(accepted));
}

void package_accepts_only_authorized_alias_ranges() {
    const std::array<uint8_t, 8> bytes = {0x5a, 0xa5, 0, 0, 0, 0, 0, 0};
    const PhysicalProgram program = load_program();
    const auto wire = encode_physical_program(program);
    const auto digest = physical_program_digest(program);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
    CHECK(std::holds_alternative<PhysicalProgramDigest>(digest));
    if (!std::holds_alternative<std::vector<uint8_t>>(wire) ||
        !std::holds_alternative<PhysicalProgramDigest>(digest)) return;

    const ArtifactId artifact{10};
    const PhysicalProgramRecord record{
        std::get<PhysicalProgramDigest>(digest),
        std::get<std::vector<uint8_t>>(wire),
        {ElementType::U32, {}}};
    PhysicalResourceBinding first;
    first.resource_id = 0;
    first.program_digest = record.digest;
    first.semantic_function_id = 0;
    first.semantic_value_id = 0;
    first.planes = {{0, artifact, 0, 4}};
    PhysicalResourceBinding second = first;
    second.resource_id = 1;
    second.semantic_value_id = 1;
    const std::array<PhysicalResourceBinding, 2> bindings = {first, second};
    auto semantic = verify_and_canonicalize_program(two_input_semantic_program());
    CHECK(std::holds_alternative<VerifiedProgram>(semantic));
    if (!std::holds_alternative<VerifiedProgram>(semantic)) return;
    auto accepted = load_physical_program_package(
        make_alias_index(artifact), std::get<VerifiedProgram>(std::move(semantic)),
        std::span<const PhysicalProgramRecord>(&record, 1), bindings);
    CHECK(std::holds_alternative<VerifiedPhysicalProgramPackage>(accepted));

    PhysicalProgram different = load_program();
    different.instructions[1].bit_width = 16;
    const auto different_wire = encode_physical_program(different);
    const auto different_digest = physical_program_digest(different);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(different_wire));
    CHECK(std::holds_alternative<PhysicalProgramDigest>(different_digest));
    if (!std::holds_alternative<std::vector<uint8_t>>(different_wire) ||
        !std::holds_alternative<PhysicalProgramDigest>(different_digest)) return;
    PhysicalProgramRecord different_record{
        std::get<PhysicalProgramDigest>(different_digest),
        std::get<std::vector<uint8_t>>(different_wire),
        {ElementType::U32, {}}};
    std::array<PhysicalProgramRecord, 2> records = {record, different_record};
    std::sort(records.begin(), records.end(), [](const auto& left, const auto& right) {
        return left.digest < right.digest;
    });
    PhysicalResourceBinding different_resource = second;
    different_resource.program_digest = different_record.digest;
    const std::array<PhysicalResourceBinding, 2> different_bindings = {
        first, different_resource};
    auto different_semantic = verify_and_canonicalize_program(two_input_semantic_program());
    CHECK(std::holds_alternative<VerifiedProgram>(different_semantic));
    if (std::holds_alternative<VerifiedProgram>(different_semantic)) {
        const auto rejected = load_physical_program_package(
            make_alias_index(artifact),
            std::get<VerifiedProgram>(std::move(different_semantic)), records,
            different_bindings);
        CHECK(std::holds_alternative<CompatibilityReport>(rejected));
        if (const auto* report = std::get_if<CompatibilityReport>(&rejected))
            CHECK(report->code == CompatibilityError::PACKAGE_BOUNDS_INVALID);
    }

    PhysicalResourceBinding partial = second;
    partial.planes[0].offset = 1;
    partial.planes[0].length = 3;
    const std::array<PhysicalResourceBinding, 2> partial_bindings = {first, partial};
    auto partial_semantic = verify_and_canonicalize_program(two_input_semantic_program());
    CHECK(std::holds_alternative<VerifiedProgram>(partial_semantic));
    if (std::holds_alternative<VerifiedProgram>(partial_semantic)) {
        const std::array<PhysicalProgramRecord, 1> one_record = {record};
        const auto rejected = load_physical_program_package(
            make_alias_index(artifact),
            std::get<VerifiedProgram>(std::move(partial_semantic)), one_record,
            partial_bindings);
        CHECK(std::holds_alternative<CompatibilityReport>(rejected));
        if (const auto* report = std::get_if<CompatibilityReport>(&rejected))
            CHECK(report->code == CompatibilityError::PACKAGE_BOUNDS_INVALID);
    }
}

} // namespace

int main() {
    package_accepts_canonical_record();
    package_rejects_bad_source_and_digest();
    package_rejects_overlapping_resource_ranges();
    package_accepts_only_authorized_alias_ranges();
    return test_summary("test_physical_program_package");
}
