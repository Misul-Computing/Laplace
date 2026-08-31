#include "physical_program_package.h"
#include "compat_rule.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string_view>

namespace Laplace {
namespace {

constexpr size_t kPackageWireLimit = 64 * 1024;

void append_u16(std::vector<uint8_t>& out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value));
    out.push_back(static_cast<uint8_t>(value >> 8));
}
void append_u32(std::vector<uint8_t>& out, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8)
        out.push_back(static_cast<uint8_t>(value >> shift));
}
void append_u64(std::vector<uint8_t>& out, uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8)
        out.push_back(static_cast<uint8_t>(value >> shift));
}
bool read_u16(std::span<const uint8_t> wire, size_t& at, uint16_t& value) {
    if (at > wire.size() || wire.size() - at < 2) return false;
    value = static_cast<uint16_t>(wire[at]) |
            static_cast<uint16_t>(wire[at + 1]) << 8;
    at += 2;
    return true;
}
bool read_u32(std::span<const uint8_t> wire, size_t& at, uint32_t& value) {
    if (at > wire.size() || wire.size() - at < 4) return false;
    value = 0;
    for (unsigned shift = 0; shift != 32; shift += 8)
        value |= static_cast<uint32_t>(wire[at++]) << shift;
    return true;
}
bool read_u64(std::span<const uint8_t> wire, size_t& at, uint64_t& value) {
    if (at > wire.size() || wire.size() - at < 8) return false;
    value = 0;
    for (unsigned shift = 0; shift != 64; shift += 8)
        value |= static_cast<uint64_t>(wire[at++]) << shift;
    return true;
}
bool read_bytes(std::span<const uint8_t> wire, size_t& at, size_t length,
                std::vector<uint8_t>& out) {
    if (length > wire.size() - std::min(at, wire.size())) return false;
    out.assign(wire.begin() + static_cast<ptrdiff_t>(at),
               wire.begin() + static_cast<ptrdiff_t>(at + length));
    at += length;
    return true;
}

CompatibilityReport reject(CompatibilityError code, std::string detail,
                           uint32_t resource = UINT32_MAX) {
    CompatibilityReport report = compatibility_report(code, std::move(detail));
    report.artifact_index = resource;
    return report;
}

bool digest_less(const PhysicalProgramDigest& a, const PhysicalProgramDigest& b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

bool digest_zero(const PhysicalProgramDigest& digest) {
    for (uint8_t byte : digest) if (byte != 0) return false;
    return true;
}

const PackageView* find_artifact(const ArtifactIndex& index, ArtifactId id) {
    auto artifacts = index.artifacts();
    auto it = std::lower_bound(artifacts.begin(), artifacts.end(), id,
                               [](const PackageView& view, ArtifactId wanted) {
                                   return view.artifact_id().value < wanted.value;
                               });
    return it != artifacts.end() && it->artifact_id() == id ? &*it : nullptr;
}

const ValueType* semantic_entry_value_type(const VerifiedProgram& semantic,
                                           uint32_t function_id,
                                           uint32_t value_id) {
    const Program& program = program_definition(semantic);
    for (const Function& function : program.functions) {
        if (function.id != function_id) continue;
        for (const Region& region : function.regions) {
            if (region.id != function.entry_region_id) continue;
            for (const TypedValue& argument : region.arguments)
                if (argument.id == value_id) return &argument.type;
            return nullptr;
        }
        return nullptr;
    }
    return nullptr;
}

bool logical_type_matches(const LogicalTensorType& logical,
                          const ValueType& semantic) {
    if (logical.element_type != semantic.element_type ||
        logical.extents.size() != semantic.dimensions.size())
        return false;
    for (size_t index = 0; index < logical.extents.size(); ++index) {
        const DimensionExpr& dimension = semantic.dimensions[index];
        if (dimension.expression != DimensionExpression::Constant ||
            !dimension.operands.empty() || dimension.value != logical.extents[index])
            return false;
    }
    return true;
}

Sha256Digest package_digest(const ArtifactIndex& physical,
                            const Sha256Digest& semantic_digest,
                            std::span<const PhysicalProgramRecord> records,
                            std::span<const PhysicalResourceBinding> resources) {
    std::vector<uint8_t> bytes;
    constexpr std::string_view domain = "laplace-physical-package-v2\0";
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    bytes.insert(bytes.end(), semantic_digest.bytes.begin(), semantic_digest.bytes.end());
    bytes.insert(bytes.end(), physical.digest().bytes.begin(), physical.digest().bytes.end());
    auto append_u32 = [&](uint32_t value) {
        for (unsigned shift = 0; shift != 32; shift += 8)
            bytes.push_back(static_cast<uint8_t>(value >> shift));
    };
    auto append_u64 = [&](uint64_t value) {
        for (unsigned shift = 0; shift != 64; shift += 8)
            bytes.push_back(static_cast<uint8_t>(value >> shift));
    };
    append_u32(static_cast<uint32_t>(records.size()));
    for (const auto& record : records) {
        bytes.insert(bytes.end(), record.digest.begin(), record.digest.end());
        append_u32(static_cast<uint32_t>(record.logical_type.element_type));
        append_u32(static_cast<uint32_t>(record.logical_type.extents.size()));
        for (uint64_t extent : record.logical_type.extents) append_u64(extent);
        append_u64(record.wire.size());
        bytes.insert(bytes.end(), record.wire.begin(), record.wire.end());
    }
    append_u32(static_cast<uint32_t>(resources.size()));
    for (const auto& resource : resources) {
        append_u32(resource.resource_id);
        append_u32(resource.semantic_function_id);
        append_u32(resource.semantic_value_id);
        bytes.insert(bytes.end(), resource.program_digest.begin(), resource.program_digest.end());
        append_u32(static_cast<uint32_t>(resource.planes.size()));
        for (const auto& plane : resource.planes) {
            append_u32(plane.plane);
            append_u32(plane.artifact_id.value);
            append_u64(plane.offset);
            append_u64(plane.length);
        }
    }
    Sha256Digest digest;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.bytes.data());
    return digest;
}

struct PackageValidation {
    std::vector<PhysicalProgramRecord> records;
    std::vector<PhysicalProgram> programs;
    std::vector<PhysicalResourceBinding> resources;
};

std::vector<LogicalTensorType> logical_types(
    std::span<const PhysicalProgramRecord> records) {
    std::vector<LogicalTensorType> result;
    result.reserve(records.size());
    for (const auto& record : records) result.push_back(record.logical_type);
    return result;
}

std::variant<PackageValidation, CompatibilityReport> validate_package(
    const ArtifactIndex& physical, std::span<const PhysicalProgramRecord> records,
    std::span<const PhysicalResourceBinding> resources) {
    if (records.empty() || records.size() > UINT32_MAX || resources.size() > UINT32_MAX)
        return reject(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                      "physical program package exceeds bounded representation");
    PackageValidation result;
    result.records.assign(records.begin(), records.end());
    for (size_t i = 0; i < result.records.size(); ++i) {
        const auto& record = result.records[i];
        if (digest_zero(record.digest) || record.wire.empty() ||
            (i != 0 && !digest_less(result.records[i - 1].digest, record.digest)))
            return reject(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                          "physical program records are not sorted-unique and nonempty");
        const auto parsed = parse_physical_program(record.wire);
        if (!std::holds_alternative<PhysicalProgram>(parsed))
            return reject(CompatibilityError::IMPORT_MANIFEST_INVALID,
                          "physical program wire is not valid");
        const PhysicalProgram& parsed_program = std::get<PhysicalProgram>(parsed);
        const uint8_t logical_type = static_cast<uint8_t>(record.logical_type.element_type);
        if (logical_type < static_cast<uint8_t>(ElementType::I1) ||
            logical_type > static_cast<uint8_t>(ElementType::F32) ||
            record.logical_type.extents.size() != parsed_program.logical_rank ||
            record.logical_type.extents.size() > 8 ||
            std::any_of(record.logical_type.extents.begin(),
                        record.logical_type.extents.end(),
                        [](uint64_t extent) { return extent == 0; }))
            return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                          "physical program logical type does not match its wire");
        const auto canonical = encode_physical_program(std::get<PhysicalProgram>(parsed));
        if (!std::holds_alternative<std::vector<uint8_t>>(canonical) ||
            std::get<std::vector<uint8_t>>(canonical) != record.wire)
            return reject(CompatibilityError::IMPORT_MANIFEST_INVALID,
                          "physical program wire is not canonical");
        const auto digest = physical_program_digest(std::get<PhysicalProgram>(parsed));
        if (!std::holds_alternative<PhysicalProgramDigest>(digest) ||
            std::get<PhysicalProgramDigest>(digest) != record.digest)
            return reject(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH,
                          "physical program digest does not match canonical wire");
        result.programs.push_back(std::get<PhysicalProgram>(parsed));
    }
    result.resources.assign(resources.begin(), resources.end());
    struct SourceRange {
        uint32_t artifact = UINT32_MAX;
        uint64_t begin = 0;
        uint64_t end = 0;
        uint32_t resource = UINT32_MAX;
    };
    std::vector<SourceRange> source_ranges;
    for (size_t i = 0; i < result.resources.size(); ++i) {
        const auto& resource = result.resources[i];
        if (resource.resource_id == UINT32_MAX ||
            (i != 0 && result.resources[i - 1].resource_id >= resource.resource_id))
            return reject(CompatibilityError::IMPORT_TENSOR_DUPLICATE,
                          "physical resource bindings are not unique", resource.resource_id);
        auto record_it = std::lower_bound(
            result.records.begin(), result.records.end(), resource.program_digest,
            [](const auto& record, const auto& digest) { return digest_less(record.digest, digest); });
        if (record_it == result.records.end() || record_it->digest != resource.program_digest)
            return reject(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                          "resource binding references an unknown physical program",
                          resource.resource_id);
        const PhysicalProgram& program = result.programs[static_cast<size_t>(record_it - result.records.begin())];
        size_t external_count = 0;
        for (const auto& plane : program.planes)
            external_count += plane.storage == PhysicalPlaneStorage::External;
        if (resource.planes.size() != external_count)
            return reject(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                          "resource binding does not provide exactly one source per external plane",
                          resource.resource_id);
        std::vector<bool> seen(program.planes.size(), false);
        uint32_t previous_plane = UINT32_MAX;
        for (const auto& plane : resource.planes) {
            if (plane.plane >= program.planes.size() ||
                program.planes[plane.plane].storage != PhysicalPlaneStorage::External ||
                seen[plane.plane] || plane.artifact_id.value == UINT32_MAX || plane.length == 0 ||
                (previous_plane != UINT32_MAX && previous_plane >= plane.plane) ||
                plane.offset % program.planes[plane.plane].alignment != 0)
                return reject(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                              "physical resource plane binding is malformed", resource.resource_id);
            const PackageView* source = find_artifact(physical, plane.artifact_id);
            if (!source || plane.offset > source->bytes().size() ||
                plane.length > source->bytes().size() - plane.offset)
                return reject(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                              "physical resource plane is not owned by the package index",
                              resource.resource_id);
            if (reinterpret_cast<uintptr_t>(source->bytes().data() + plane.offset) %
                    program.planes[plane.plane].alignment != 0)
                return reject(CompatibilityError::IR_CONSTRAINT_FAILED,
                              "physical source address does not satisfy plane alignment",
                              resource.resource_id);
            seen[plane.plane] = true;
            previous_plane = plane.plane;
            source_ranges.push_back(
                {plane.artifact_id.value, plane.offset,
                 plane.offset + plane.length, resource.resource_id});
        }
    }
    std::sort(source_ranges.begin(), source_ranges.end(),
              [](const SourceRange& left, const SourceRange& right) {
                  if (left.artifact != right.artifact)
                      return left.artifact < right.artifact;
                  if (left.begin != right.begin) return left.begin < right.begin;
                  if (left.end != right.end) return left.end < right.end;
                  return left.resource < right.resource;
              });
    uint32_t active_artifact = UINT32_MAX;
    uint64_t active_end = 0;
    for (const SourceRange& current : source_ranges) {
        if (current.artifact != active_artifact) {
            active_artifact = current.artifact;
            active_end = current.end;
            continue;
        }
        if (current.begin < active_end)
            return reject(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                          "physical resource source ranges overlap without an alias contract",
                          current.resource);
        active_end = current.end;
    }
    return result;
}

} // namespace

std::variant<std::monostate, CompatibilityReport>
validate_physical_program_package_for_runtime(
    const RuntimePackage& runtime,
    const VerifiedPhysicalProgramPackage& physical) {
    const auto& manifest = physical.semantic_manifest();
    if (!manifest || !runtime.product_authoritative() ||
        manifest->package_fingerprint() != runtime.package_fingerprint())
        return compatibility_report(
            CompatibilityError::AUTHORITY_INVALID,
            "physical package semantic identity does not match the runtime package");
    if (physical.physical_index().digest() != runtime.physical_index().digest())
        return compatibility_report(
            CompatibilityError::PACKAGE_CHECKSUM_MISMATCH,
            "physical package source index does not match the runtime package");
    if (physical.digest() == Sha256Digest{})
        return compatibility_report(
            CompatibilityError::PACKAGE_CHECKSUM_MISMATCH,
            "physical package digest is empty");
    return std::monostate{};
}

PhysicalProgramPackageWireResult encode_physical_program_package_records(
    std::span<const PhysicalProgramRecord> programs,
    std::span<const PhysicalResourceBinding> resources) {
    if (programs.empty() || programs.size() > UINT32_MAX || resources.size() > UINT32_MAX)
        return reject(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                      "physical package wire has invalid record counts");
    std::vector<uint8_t> wire;
    wire.insert(wire.end(), {'L','A','P','P','K','G','0','2'});
    append_u16(wire, 2);
    append_u16(wire, 0);
    append_u32(wire, 0);
    append_u32(wire, static_cast<uint32_t>(programs.size()));
    append_u32(wire, static_cast<uint32_t>(resources.size()));
    for (const auto& record : programs) {
        wire.insert(wire.end(), record.digest.begin(), record.digest.end());
        append_u32(wire, static_cast<uint32_t>(record.logical_type.element_type));
        if (record.logical_type.extents.size() > UINT32_MAX)
            return reject(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                          "physical package logical rank exceeds wire bounds");
        append_u32(wire, static_cast<uint32_t>(record.logical_type.extents.size()));
        for (uint64_t extent : record.logical_type.extents) append_u64(wire, extent);
        if (record.wire.size() > UINT32_MAX) return reject(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                                            "physical program wire exceeds bounds");
        append_u32(wire, static_cast<uint32_t>(record.wire.size()));
        wire.insert(wire.end(), record.wire.begin(), record.wire.end());
    }
    for (const auto& resource : resources) {
        append_u32(wire, resource.resource_id);
        append_u32(wire, resource.semantic_function_id);
        append_u32(wire, resource.semantic_value_id);
        wire.insert(wire.end(), resource.program_digest.begin(), resource.program_digest.end());
        if (resource.planes.size() > UINT32_MAX) return reject(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                                                 "physical resource planes exceed wire bounds");
        append_u32(wire, static_cast<uint32_t>(resource.planes.size()));
        for (const auto& plane : resource.planes) {
            append_u32(wire, plane.plane);
            append_u32(wire, plane.artifact_id.value);
            append_u64(wire, plane.offset);
            append_u64(wire, plane.length);
        }
    }
    if (wire.size() > kPackageWireLimit) return reject(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                                        "physical package wire exceeds its size limit");
    const uint32_t total = static_cast<uint32_t>(wire.size());
    for (unsigned shift = 0; shift != 32; shift += 8) wire[12 + shift / 8] = static_cast<uint8_t>(total >> shift);
    return wire;
}

PhysicalProgramPackageDecodeResult decode_physical_program_package_records(
    std::span<const uint8_t> wire) {
    if (wire.size() < 24 || wire.size() > kPackageWireLimit ||
        !std::equal(wire.begin(), wire.begin() + 8,
                    std::array<uint8_t, 8>{'L','A','P','P','K','G','0','2'}.begin()))
        return reject(CompatibilityError::PACKAGE_BAD_MAGIC,
                      "physical package wire header is invalid");
    size_t at = 8;
    uint16_t version = 0, reserved = 0;
    uint32_t total = 0, record_count = 0, resource_count = 0;
    if (!read_u16(wire, at, version) || !read_u16(wire, at, reserved) ||
        !read_u32(wire, at, total) || !read_u32(wire, at, record_count) ||
        !read_u32(wire, at, resource_count) || version != 2 || reserved != 0 ||
        total != wire.size() || record_count == 0 || record_count > 1024 || resource_count > 65535)
        return reject(CompatibilityError::PACKAGE_VERSION_UNSUPPORTED,
                      "physical package wire header fields are invalid");
    PhysicalProgramPackageRecords result;
    result.first.reserve(record_count);
    for (uint32_t i = 0; i < record_count; ++i) {
        PhysicalProgramRecord record;
        if (wire.size() - at < record.digest.size())
            return reject(CompatibilityError::PACKAGE_BOUNDS_INVALID, "physical package record is truncated");
        std::copy_n(wire.begin() + static_cast<ptrdiff_t>(at), record.digest.size(), record.digest.begin());
        at += record.digest.size();
        uint32_t element = 0, rank = 0, length = 0;
        if (!read_u32(wire, at, element) || !read_u32(wire, at, rank) || rank > 8)
            return reject(CompatibilityError::PACKAGE_BOUNDS_INVALID, "physical package logical type is invalid");
        record.logical_type.element_type = static_cast<ElementType>(element);
        record.logical_type.extents.resize(rank);
        for (uint64_t& extent : record.logical_type.extents)
            if (!read_u64(wire, at, extent) || extent == 0)
                return reject(CompatibilityError::IR_SHAPE_MISMATCH, "physical package extent is invalid");
        if (!read_u32(wire, at, length) || length == 0 || length > wire.size() - at ||
            !read_bytes(wire, at, length, record.wire))
            return reject(CompatibilityError::PACKAGE_BOUNDS_INVALID, "physical package program wire is truncated");
        result.first.push_back(std::move(record));
    }
    result.second.reserve(resource_count);
    for (uint32_t i = 0; i < resource_count; ++i) {
        PhysicalResourceBinding resource;
        uint32_t count = 0;
        if (!read_u32(wire, at, resource.resource_id) ||
            !read_u32(wire, at, resource.semantic_function_id) ||
            !read_u32(wire, at, resource.semantic_value_id) ||
            wire.size() - at < resource.program_digest.size())
            return reject(CompatibilityError::PACKAGE_BOUNDS_INVALID, "physical package resource is truncated");
        std::copy_n(wire.begin() + static_cast<ptrdiff_t>(at), resource.program_digest.size(), resource.program_digest.begin());
        at += resource.program_digest.size();
        if (!read_u32(wire, at, count) || count > 32)
            return reject(CompatibilityError::PACKAGE_BOUNDS_INVALID, "physical package plane count is invalid");
        resource.planes.resize(count);
        for (auto& plane : resource.planes)
            if (!read_u32(wire, at, plane.plane) || !read_u32(wire, at, plane.artifact_id.value) ||
                !read_u64(wire, at, plane.offset) || !read_u64(wire, at, plane.length))
                return reject(CompatibilityError::PACKAGE_BOUNDS_INVALID, "physical package plane is truncated");
        result.second.push_back(std::move(resource));
    }
    if (at != wire.size())
        return reject(CompatibilityError::PACKAGE_BOUNDS_INVALID, "physical package wire has trailing bytes");
    return result;
}

VerifiedPhysicalProgramPackage::VerifiedPhysicalProgramPackage(
    ArtifactIndex physical, VerifiedProgram semantic,
    std::vector<PhysicalProgram> programs,
    std::vector<LogicalTensorType> logical_types,
    std::vector<PhysicalResourceBinding> resources, Sha256Digest digest)
    : physical_(std::move(physical)), semantic_(std::move(semantic)),
      programs_(std::move(programs)), logical_types_(std::move(logical_types)),
      resources_(std::move(resources)),
      digest_(digest) {}

VerifiedPhysicalProgramPackage::VerifiedPhysicalProgramPackage(
    ArtifactIndex physical, SemanticManifest manifest,
    std::vector<PhysicalProgram> programs,
    std::vector<LogicalTensorType> logical_types,
    std::vector<PhysicalResourceBinding> resources, Sha256Digest digest)
    : physical_(std::move(physical)), manifest_(std::move(manifest)),
      programs_(std::move(programs)), logical_types_(std::move(logical_types)),
      resources_(std::move(resources)), digest_(digest) {}

const PhysicalProgram* VerifiedPhysicalProgramPackage::find_program(
    const PhysicalProgramDigest& digest) const noexcept {
    for (size_t index = 0; index < programs_.size(); ++index) {
        const auto encoded = physical_program_digest(programs_[index]);
        if (std::holds_alternative<PhysicalProgramDigest>(encoded) &&
            std::get<PhysicalProgramDigest>(encoded) == digest)
            return &programs_[index];
    }
    return nullptr;
}

std::variant<std::vector<PhysicalResourcePlaneView>, CompatibilityReport>
VerifiedPhysicalProgramPackage::resolve_resource(uint32_t resource_id) const {
    const auto it = std::lower_bound(
        resources_.begin(), resources_.end(), resource_id,
        [](const PhysicalResourceBinding& binding, uint32_t id) {
            return binding.resource_id < id;
        });
    if (it == resources_.end() || it->resource_id != resource_id)
        return reject(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                      "physical resource occurrence is not present", resource_id);
    std::vector<PhysicalResourcePlaneView> views;
    views.reserve(it->planes.size());
    for (const auto& plane : it->planes) {
        const PackageView* artifact = find_artifact(physical_, plane.artifact_id);
        if (!artifact || plane.offset > artifact->bytes().size() ||
            plane.length > artifact->bytes().size() - plane.offset)
            return reject(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                          "physical resource occurrence is no longer resolvable",
                          resource_id);
        views.push_back({plane.plane, plane.artifact_id,
                         artifact->bytes().subspan(static_cast<size_t>(plane.offset),
                                                   static_cast<size_t>(plane.length))});
    }
    return views;
}

PhysicalProgramPackageResult load_physical_program_package(
    ArtifactIndex physical, VerifiedProgram semantic,
    std::span<const PhysicalProgramRecord> records,
    std::span<const PhysicalResourceBinding> resources) {
    auto validation = validate_package(physical, records, resources);
    if (const auto* report = std::get_if<CompatibilityReport>(&validation)) return *report;
    auto checked = std::get<PackageValidation>(std::move(validation));
    std::vector<std::pair<uint32_t, uint32_t>> linked_values;
    for (const auto& resource : checked.resources) {
        if (resource.semantic_function_id == UINT32_MAX ||
            resource.semantic_value_id == UINT32_MAX)
            return reject(CompatibilityError::IR_REFERENCE_INVALID,
                          "physical resource lacks a scoped semantic entry-value link",
                          resource.resource_id);
        const auto scoped =
            std::pair{resource.semantic_function_id, resource.semantic_value_id};
        const ValueType* semantic_type = semantic_entry_value_type(
            semantic, resource.semantic_function_id, resource.semantic_value_id);
        if (!semantic_type ||
            std::find(linked_values.begin(), linked_values.end(), scoped) !=
                linked_values.end())
            return reject(CompatibilityError::IR_REFERENCE_INVALID,
                          "physical resource semantic entry-value link is unknown or duplicated",
                          resource.resource_id);
        const auto record_it = std::lower_bound(
            checked.records.begin(), checked.records.end(), resource.program_digest,
            [](const auto& record, const auto& digest) {
                return digest_less(record.digest, digest);
            });
        if (record_it == checked.records.end() ||
            !logical_type_matches(record_it->logical_type, *semantic_type))
            return reject(CompatibilityError::IR_SHAPE_MISMATCH,
                          "physical resource logical type does not match its semantic entry value",
                          resource.resource_id);
        linked_values.push_back(scoped);
    }
    const ProgramDigest semantic_digest = program_digest(semantic);
    const Sha256Digest digest{semantic_digest.bytes};
    const Sha256Digest package = package_digest(physical, digest,
                                               checked.records, checked.resources);
    return VerifiedPhysicalProgramPackage(std::move(physical), std::move(semantic),
                                          std::move(checked.programs),
                                          logical_types(checked.records),
                                          std::move(checked.resources), package);
}

PhysicalProgramPackageResult load_physical_program_package(
    ArtifactIndex physical, SemanticManifest manifest,
    std::span<const PhysicalProgramRecord> records,
    std::span<const PhysicalResourceBinding> resources) {
    if (!manifest.graph_proof() || manifest.package_fingerprint() == Sha256Digest{})
        return reject(CompatibilityError::AUTHORITY_INVALID,
                      "physical package requires complete semantic authority");
    auto validation = validate_package(physical, records, resources);
    if (const auto* report = std::get_if<CompatibilityReport>(&validation)) return *report;
    auto checked = std::get<PackageValidation>(std::move(validation));
    for (const auto& resource : checked.resources) {
        if (resource.semantic_function_id != UINT32_MAX ||
            resource.semantic_value_id != UINT32_MAX)
            return reject(CompatibilityError::IR_REFERENCE_INVALID,
                          "manifest adapter cannot authenticate semantic SSA links",
                          resource.resource_id);
    }
    const Sha256Digest digest = package_digest(physical, manifest.package_fingerprint(),
                                               checked.records, checked.resources);
    return VerifiedPhysicalProgramPackage(std::move(physical), std::move(manifest),
                                          std::move(checked.programs),
                                          logical_types(checked.records),
                                          std::move(checked.resources), digest);
}

PhysicalProgramPackageResult load_physical_program_package_wire(
    ArtifactIndex physical, SemanticManifest manifest,
    std::span<const uint8_t> wire) {
    auto decoded = decode_physical_program_package_records(wire);
    if (const auto* report = std::get_if<CompatibilityReport>(&decoded)) return *report;
    auto records = std::get<PhysicalProgramPackageRecords>(std::move(decoded));
    return load_physical_program_package(std::move(physical), std::move(manifest),
                                         records.first, records.second);
}

} // namespace Laplace
