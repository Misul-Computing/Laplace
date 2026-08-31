#include "program_package.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <optional>
#include <tuple>

namespace Laplace {
namespace {

constexpr size_t kProgramPackageWireLimit = 32u * 1024u * 1024u;

CompatibilityReport package_error(CompatibilityError code,
                                  const char* detail) {
    CompatibilityReport report = compatibility_report(code, detail);
    report.stage = CompatibilityStage::Package;
    return report;
}

const PackageView* find_artifact(const ArtifactIndex& index, ArtifactId id) {
    for (const PackageView& artifact : index.artifacts())
        if (artifact.artifact_id() == id) return &artifact;
    return nullptr;
}

Sha256Digest digest_bytes(std::span<const uint8_t> bytes) {
    Sha256Digest digest;
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    for (size_t offset = 0; offset < bytes.size();) {
        const size_t count = std::min<size_t>(bytes.size() - offset, 1u << 20);
        CC_SHA256_Update(&context, bytes.data() + offset,
                         static_cast<CC_LONG>(count));
        offset += count;
    }
    CC_SHA256_Final(digest.bytes.data(), &context);
    return digest;
}

class DigestWriter {
public:
    void bytes(std::span<const uint8_t> value) {
        bytes_.insert(bytes_.end(), value.begin(), value.end());
    }
    void u8(uint8_t value) { bytes_.push_back(value); }
    void u32(uint32_t value) {
        for (unsigned shift = 0; shift != 32; shift += 8)
            u8(static_cast<uint8_t>(value >> shift));
    }
    void u64(uint64_t value) {
        for (unsigned shift = 0; shift != 64; shift += 8)
            u8(static_cast<uint8_t>(value >> shift));
    }
    Sha256Digest finish() const { return digest_bytes(bytes_); }

private:
    std::vector<uint8_t> bytes_;
};

std::optional<uint32_t> canonical_function_slot(
    const VerifiedProgram& program, uint32_t function_id) {
    const auto ids = canonical_function_ids(program);
    for (uint32_t slot = 0; slot < ids.size(); ++slot)
        if (ids[slot] == function_id) return slot;
    return std::nullopt;
}

struct EntryCoordinate {
    uint32_t function = UINT32_MAX;
    uint32_t argument = UINT32_MAX;
    friend bool operator==(const EntryCoordinate&, const EntryCoordinate&) = default;
    friend bool operator<(const EntryCoordinate& left,
                          const EntryCoordinate& right) {
        return std::tie(left.function, left.argument) <
               std::tie(right.function, right.argument);
    }
};

std::optional<EntryCoordinate> entry_coordinate(
    const VerifiedProgram& program, uint32_t function_id, uint32_t value_id,
    const ValueType** type = nullptr) {
    const auto function_slot = canonical_function_slot(program, function_id);
    if (!function_slot) return std::nullopt;
    for (const Function& function : program_definition(program).functions) {
        if (function.id != function_id) continue;
        for (const Region& region : function.regions) {
            if (region.id != function.entry_region_id) continue;
            for (uint32_t index = 0; index < region.arguments.size(); ++index) {
                if (region.arguments[index].id != value_id) continue;
                if (type) *type = &region.arguments[index].type;
                return EntryCoordinate{*function_slot, index};
            }
            return std::nullopt;
        }
    }
    return std::nullopt;
}

std::optional<std::pair<uint32_t, uint32_t>> output_coordinate(
    const VerifiedProgram& program, uint32_t function_id, uint32_t result_index,
    const ValueType** type = nullptr) {
    const auto function_slot = canonical_function_slot(program, function_id);
    if (!function_slot) return std::nullopt;
    const Program& definition = program_definition(program);
    const Function* function = nullptr;
    for (const Function& candidate : definition.functions)
        if (candidate.id == function_id) function = &candidate;
    if (!function || result_index >= function->result_types.size())
        return std::nullopt;
    const auto exported = std::find_if(
        definition.exports.begin(), definition.exports.end(),
        [&](const ProgramExport& item) {
            return item.function_id == function_id &&
                   item.result_index == result_index;
        });
    if (exported == definition.exports.end()) return std::nullopt;
    if (type) *type = &function->result_types[result_index];
    return std::pair{*function_slot, result_index};
}

bool token_input_type(const ValueType& type) {
    return type.element_type == ElementType::U32 && type.dimensions.empty();
}

bool token_output_type(const ValueType& type, size_t vocabulary_size) {
    return type.element_type == ElementType::F32 && type.dimensions.size() == 1 &&
           type.dimensions[0].expression == DimensionExpression::Constant &&
           type.dimensions[0].operands.empty() &&
           type.dimensions[0].value == vocabulary_size;
}

std::variant<std::vector<TokenEndpointBinding>, CompatibilityReport>
validate_token_bindings(const VerifiedProgram& program,
                        const TokenProgram& token,
                        std::span<const TokenEndpointBinding> bindings,
                        EntryCoordinate* input_coordinate,
                        std::pair<uint32_t, uint32_t>* score_coordinate) {
    if (bindings.size() != 2)
        return package_error(CompatibilityError::IR_REFERENCE_INVALID,
                             "program package requires exactly two token endpoints");
    std::vector<TokenEndpointBinding> canonical(bindings.begin(), bindings.end());
    std::sort(canonical.begin(), canonical.end(),
              [](const auto& left, const auto& right) {
                  return static_cast<uint8_t>(left.kind) <
                         static_cast<uint8_t>(right.kind);
              });
    if (canonical[0].kind != TokenEndpointKind::InputToken ||
        canonical[1].kind != TokenEndpointKind::OutputScores)
        return package_error(CompatibilityError::IR_REFERENCE_INVALID,
                             "token endpoints are missing, repeated, or unknown");
    const ValueType* input_type = nullptr;
    const auto input = entry_coordinate(
        program, canonical[0].semantic_function_id,
        canonical[0].semantic_value, &input_type);
    const ValueType* output_type = nullptr;
    const auto output = output_coordinate(
        program, canonical[1].semantic_function_id,
        canonical[1].semantic_value, &output_type);
    if (!input || !input_type || !token_input_type(*input_type) ||
        !output || !output_type ||
        !token_output_type(*output_type, token.definition().vocabulary.size()))
        return package_error(CompatibilityError::IR_SHAPE_MISMATCH,
                             "token endpoints do not match the semantic program");
    *input_coordinate = *input;
    *score_coordinate = *output;
    return canonical;
}

std::variant<std::vector<uint8_t>, CompatibilityReport>
token_payload(const ArtifactIndex& index, const TokenProgramSource& source) {
    const PackageView* artifact = find_artifact(index, source.artifact_id);
    if (!artifact || source.length == 0 ||
        source.offset > artifact->bytes().size() ||
        source.length > artifact->bytes().size() - source.offset)
        return package_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                             "token program source range is not package-owned");
    const auto payload = artifact->bytes().subspan(
        static_cast<size_t>(source.offset), static_cast<size_t>(source.length));
    if (digest_bytes(payload) != source.digest)
        return package_error(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH,
                             "token program source digest does not match");
    return std::vector<uint8_t>(payload.begin(), payload.end());
}

std::variant<std::vector<PhysicalProgramRecord>, CompatibilityReport>
physical_records(const VerifiedPhysicalProgramPackage& package) {
    const auto programs = package.programs();
    const auto logical = package.program_logical_types();
    if (programs.size() != logical.size())
        return package_error(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                             "physical program and logical-type counts differ");
    std::vector<PhysicalProgramRecord> records;
    records.reserve(programs.size());
    for (size_t index = 0; index < programs.size(); ++index) {
        auto wire = encode_physical_program(programs[index]);
        auto digest = physical_program_digest(programs[index]);
        if (!std::holds_alternative<std::vector<uint8_t>>(wire) ||
            !std::holds_alternative<PhysicalProgramDigest>(digest))
            return package_error(CompatibilityError::IMPORT_MANIFEST_INVALID,
                                 "verified physical program cannot be re-encoded");
        records.push_back({
            std::get<PhysicalProgramDigest>(digest),
            std::get<std::vector<uint8_t>>(std::move(wire)), logical[index]});
    }
    return records;
}

std::variant<Sha256Digest, CompatibilityReport> closure_digest(
    const VerifiedPhysicalProgramPackage& physical,
    const VerifiedStateSchema& state,
    const TokenProgramSource& token_source,
    std::span<const TokenEndpointBinding> token_bindings) {
    if (!physical.semantic_program())
        return package_error(CompatibilityError::AUTHORITY_INVALID,
                             "complete package lacks a verified semantic program");
    const VerifiedProgram& program = *physical.semantic_program();

    auto records_result = physical_records(physical);
    if (const auto* report = std::get_if<CompatibilityReport>(&records_result))
        return *report;
    const auto& records = std::get<std::vector<PhysicalProgramRecord>>(records_result);

    DigestWriter digest;
    static constexpr std::array<uint8_t, 27> domain = {
        'l','a','p','l','a','c','e','-','p','r','o','g','r','a','m','-','p','a','c','k','a','g','e','-','v','1','\0'};
    digest.bytes(domain);
    digest.bytes(program_digest(program).bytes);
    digest.bytes(state_schema_digest(state).bytes);
    digest.u64(token_source.offset);
    digest.u64(token_source.length);
    digest.bytes(token_source.digest.bytes);
    digest.u32(static_cast<uint32_t>(token_bindings.size()));
    for (const TokenEndpointBinding& binding : token_bindings) {
        digest.u8(static_cast<uint8_t>(binding.kind));
        if (binding.kind == TokenEndpointKind::InputToken) {
            const auto coordinate = entry_coordinate(
                program, binding.semantic_function_id, binding.semantic_value);
            if (!coordinate)
                return package_error(CompatibilityError::IR_REFERENCE_INVALID,
                                     "token input endpoint is no longer resolvable");
            digest.u32(coordinate->function);
            digest.u32(coordinate->argument);
        } else {
            const auto coordinate = output_coordinate(
                program, binding.semantic_function_id, binding.semantic_value);
            if (!coordinate)
                return package_error(CompatibilityError::IR_REFERENCE_INVALID,
                                     "token output endpoint is no longer resolvable");
            digest.u32(coordinate->first);
            digest.u32(coordinate->second);
        }
    }
    digest.u32(static_cast<uint32_t>(records.size()));
    for (const PhysicalProgramRecord& record : records) {
        digest.bytes(record.digest);
        digest.u8(static_cast<uint8_t>(record.logical_type.element_type));
        digest.u32(static_cast<uint32_t>(record.logical_type.extents.size()));
        for (uint64_t extent : record.logical_type.extents) digest.u64(extent);
    }

    struct CanonicalResource {
        EntryCoordinate coordinate;
        const PhysicalResourceBinding* binding = nullptr;
    };
    std::vector<CanonicalResource> resources;
    resources.reserve(physical.resources().size());
    for (const PhysicalResourceBinding& resource : physical.resources()) {
        const auto coordinate = entry_coordinate(
            program, resource.semantic_function_id, resource.semantic_value_id);
        if (!coordinate)
            return package_error(CompatibilityError::IR_REFERENCE_INVALID,
                                 "physical semantic binding is no longer resolvable");
        resources.push_back({*coordinate, &resource});
    }
    std::sort(resources.begin(), resources.end(),
              [](const auto& left, const auto& right) {
                  return left.coordinate < right.coordinate;
              });
    digest.u32(static_cast<uint32_t>(resources.size()));
    for (const CanonicalResource& resource : resources) {
        digest.u32(resource.coordinate.function);
        digest.u32(resource.coordinate.argument);
        digest.bytes(resource.binding->program_digest);
        digest.u32(static_cast<uint32_t>(resource.binding->planes.size()));
        for (const PhysicalPlaneSource& plane : resource.binding->planes) {
            const PackageView* artifact =
                find_artifact(physical.physical_index(), plane.artifact_id);
            if (!artifact || plane.offset > artifact->bytes().size() ||
                plane.length > artifact->bytes().size() - plane.offset)
                return package_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                     "physical plane is no longer package-owned");
            digest.u32(plane.plane);
            digest.u64(plane.offset);
            digest.u64(plane.length);
            const auto bytes = artifact->bytes().subspan(
                static_cast<size_t>(plane.offset),
                static_cast<size_t>(plane.length));
            digest.bytes(digest_bytes(bytes).bytes);
        }
    }
    return digest.finish();
}

class WireWriter {
public:
    void bytes(std::span<const uint8_t> value) {
        bytes_.insert(bytes_.end(), value.begin(), value.end());
    }
    void u8(uint8_t value) { bytes_.push_back(value); }
    void u16(uint16_t value) {
        for (unsigned shift = 0; shift != 16; shift += 8)
            u8(static_cast<uint8_t>(value >> shift));
    }
    void u32(uint32_t value) {
        for (unsigned shift = 0; shift != 32; shift += 8)
            u8(static_cast<uint8_t>(value >> shift));
    }
    void u64(uint64_t value) {
        for (unsigned shift = 0; shift != 64; shift += 8)
            u8(static_cast<uint8_t>(value >> shift));
    }
    std::vector<uint8_t> finish() {
        const uint32_t total = static_cast<uint32_t>(bytes_.size());
        for (unsigned shift = 0; shift != 32; shift += 8)
            bytes_[12 + shift / 8] = static_cast<uint8_t>(total >> shift);
        return std::move(bytes_);
    }

private:
    std::vector<uint8_t> bytes_;
};

class WireReader {
public:
    explicit WireReader(std::span<const uint8_t> bytes) : bytes_(bytes) {}
    bool u8(uint8_t* value) {
        if (!value || remaining() < 1) return false;
        *value = bytes_[at_++];
        return true;
    }
    bool u16(uint16_t* value) {
        if (!value || remaining() < 2) return false;
        *value = static_cast<uint16_t>(bytes_[at_]) |
                 static_cast<uint16_t>(bytes_[at_ + 1]) << 8;
        at_ += 2;
        return true;
    }
    bool u32(uint32_t* value) {
        if (!value || remaining() < 4) return false;
        *value = 0;
        for (unsigned shift = 0; shift != 32; shift += 8)
            *value |= static_cast<uint32_t>(bytes_[at_++]) << shift;
        return true;
    }
    bool u64(uint64_t* value) {
        if (!value || remaining() < 8) return false;
        *value = 0;
        for (unsigned shift = 0; shift != 64; shift += 8)
            *value |= static_cast<uint64_t>(bytes_[at_++]) << shift;
        return true;
    }
    bool take(size_t count, std::span<const uint8_t>* result) {
        if (!result || count > remaining()) return false;
        *result = bytes_.subspan(at_, count);
        at_ += count;
        return true;
    }
    size_t remaining() const noexcept { return bytes_.size() - at_; }
    bool finished() const noexcept { return at_ == bytes_.size(); }

private:
    std::span<const uint8_t> bytes_;
    size_t at_ = 0;
};

} // namespace

VerifiedProgramPackage::VerifiedProgramPackage(
    VerifiedPhysicalProgramPackage physical, VerifiedStateSchema state,
    TokenProgram token, TokenProgramSource token_source,
    std::vector<TokenEndpointBinding> token_bindings, Sha256Digest digest)
    : physical_(std::move(physical)), state_(std::move(state)),
      token_(std::move(token)), token_source_(token_source),
      token_bindings_(std::move(token_bindings)), digest_(digest) {}

const VerifiedProgram& VerifiedProgramPackage::semantic_program() const noexcept {
    return *physical_.semantic_program();
}

ProgramPackageResult build_program_package(
    ArtifactIndex physical, VerifiedProgram semantic,
    VerifiedStateSchema state, TokenProgramSource token_source,
    std::span<const TokenEndpointBinding> token_bindings,
    std::span<const PhysicalProgramRecord> programs,
    std::span<const PhysicalResourceBinding> resources) {
    if (state_program_digest(state) != program_digest(semantic))
        return package_error(CompatibilityError::AUTHORITY_INVALID,
                             "state schema belongs to a different semantic program");
    auto payload_result = token_payload(physical, token_source);
    if (const auto* report = std::get_if<CompatibilityReport>(&payload_result))
        return *report;
    auto compiled = TokenProgram::compile(
        std::get<std::vector<uint8_t>>(payload_result));
    if (!std::holds_alternative<TokenProgram>(compiled))
        return package_error(CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED,
                             "carried token program did not compile");
    TokenProgram token = std::get<TokenProgram>(std::move(compiled));
    EntryCoordinate token_input;
    std::pair<uint32_t, uint32_t> token_output;
    auto endpoints = validate_token_bindings(
        semantic, token, token_bindings, &token_input, &token_output);
    if (const auto* report = std::get_if<CompatibilityReport>(&endpoints))
        return *report;

    auto loaded = load_physical_program_package(
        std::move(physical), std::move(semantic), programs, resources);
    if (const auto* report = std::get_if<CompatibilityReport>(&loaded))
        return *report;
    VerifiedPhysicalProgramPackage verified_physical =
        std::get<VerifiedPhysicalProgramPackage>(std::move(loaded));
    const VerifiedProgram& verified_program = *verified_physical.semantic_program();

    std::vector<EntryCoordinate> required;
    for (const Function& function : program_definition(verified_program).functions) {
        const auto function_slot =
            canonical_function_slot(verified_program, function.id);
        if (!function_slot)
            return package_error(CompatibilityError::IR_REFERENCE_INVALID,
                                 "semantic function lacks canonical identity");
        for (const Region& region : function.regions) {
            if (region.id != function.entry_region_id) continue;
            for (uint32_t index = 0; index < region.arguments.size(); ++index)
                required.push_back({*function_slot, index});
        }
    }
    std::sort(required.begin(), required.end());
    std::vector<EntryCoordinate> supplied = {token_input};
    std::vector<PhysicalProgramDigest> used_programs;
    for (const PhysicalResourceBinding& resource : verified_physical.resources()) {
        const auto coordinate = entry_coordinate(
            verified_program, resource.semantic_function_id,
            resource.semantic_value_id);
        if (!coordinate)
            return package_error(CompatibilityError::IR_REFERENCE_INVALID,
                                 "physical resource lost its semantic binding");
        supplied.push_back(*coordinate);
        used_programs.push_back(resource.program_digest);
    }
    std::sort(supplied.begin(), supplied.end());
    if (supplied != required ||
        std::adjacent_find(supplied.begin(), supplied.end()) != supplied.end())
        return package_error(CompatibilityError::IMPORT_CLOSURE_INCOMPLETE,
                             "semantic entry values are not covered exactly once");
    std::sort(used_programs.begin(), used_programs.end());
    used_programs.erase(std::unique(used_programs.begin(), used_programs.end()),
                        used_programs.end());
    auto records_result = physical_records(verified_physical);
    if (const auto* report = std::get_if<CompatibilityReport>(&records_result))
        return *report;
    const auto& verified_records =
        std::get<std::vector<PhysicalProgramRecord>>(records_result);
    if (used_programs.size() != verified_records.size())
        return package_error(CompatibilityError::IMPORT_CLOSURE_INCOMPLETE,
                             "physical program package contains unused records");
    for (size_t index = 0; index < used_programs.size(); ++index)
        if (used_programs[index] != verified_records[index].digest)
            return package_error(CompatibilityError::IMPORT_CLOSURE_INCOMPLETE,
                                 "physical program use set is incomplete");

    const auto canonical_endpoints =
        std::get<std::vector<TokenEndpointBinding>>(std::move(endpoints));
    auto digest = closure_digest(verified_physical, state, token_source,
                                 canonical_endpoints);
    if (const auto* report = std::get_if<CompatibilityReport>(&digest))
        return *report;
    return VerifiedProgramPackage(
        std::move(verified_physical), std::move(state), std::move(token),
        token_source, canonical_endpoints, std::get<Sha256Digest>(digest));
}

ProgramPackageWireResult
encode_program_package(const VerifiedProgramPackage& package) {
    try {
        auto semantic = encode_program_wire(package.semantic_program());
        auto state = encode_state_schema_wire(package.state_schema());
        auto records = physical_records(package.physical_package());
        if (const auto* report = std::get_if<CompatibilityReport>(&semantic))
            return *report;
        if (const auto* report = std::get_if<CompatibilityReport>(&state))
            return *report;
        if (const auto* report = std::get_if<CompatibilityReport>(&records))
            return *report;
        auto physical = encode_physical_program_package_records(
            std::get<std::vector<PhysicalProgramRecord>>(records),
            package.physical_package().resources());
        if (const auto* report = std::get_if<CompatibilityReport>(&physical))
            return *report;
        const auto& semantic_bytes = std::get<std::vector<uint8_t>>(semantic);
        const auto& state_bytes = std::get<std::vector<uint8_t>>(state);
        const auto& physical_bytes = std::get<std::vector<uint8_t>>(physical);
        if (semantic_bytes.size() > UINT32_MAX || state_bytes.size() > UINT32_MAX ||
            physical_bytes.size() > UINT32_MAX)
            return package_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                 "program package section exceeds wire bounds");
        WireWriter writer;
        static constexpr std::array<uint8_t, 8> magic = {
            'L','A','P','P','R','G','0','1'};
        writer.bytes(magic);
        writer.u16(1);
        writer.u16(0);
        writer.u32(0);
        writer.u32(static_cast<uint32_t>(semantic_bytes.size()));
        writer.u32(static_cast<uint32_t>(state_bytes.size()));
        writer.u32(static_cast<uint32_t>(physical_bytes.size()));
        writer.u32(package.token_source().artifact_id.value);
        writer.u64(package.token_source().offset);
        writer.u64(package.token_source().length);
        writer.bytes(package.token_source().digest.bytes);
        writer.u32(static_cast<uint32_t>(package.token_bindings().size()));
        for (const TokenEndpointBinding& binding : package.token_bindings()) {
            writer.u32(static_cast<uint32_t>(binding.kind));
            writer.u32(binding.semantic_function_id);
            writer.u32(binding.semantic_value);
        }
        writer.bytes(semantic_bytes);
        writer.bytes(state_bytes);
        writer.bytes(physical_bytes);
        writer.bytes(package.digest().bytes);
        std::vector<uint8_t> wire = writer.finish();
        if (wire.size() > kProgramPackageWireLimit)
            return package_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                 "program package wire exceeds its bounded size");
        return wire;
    } catch (const std::bad_alloc&) {
        return package_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                             "program package wire allocation failed");
    }
}

ProgramPackageResult decode_program_package(
    ArtifactIndex physical, std::span<const uint8_t> wire) {
    static constexpr std::array<uint8_t, 8> magic = {
        'L','A','P','P','R','G','0','1'};
    if (wire.size() < 116 || wire.size() > kProgramPackageWireLimit ||
        !std::equal(magic.begin(), magic.end(), wire.begin()))
        return package_error(CompatibilityError::PACKAGE_BAD_MAGIC,
                             "program package wire header is invalid");
    try {
        WireReader reader(wire.subspan(8));
        uint16_t version = 0, reserved = 0;
        uint32_t total = 0, semantic_length = 0, state_length = 0,
                 physical_length = 0, endpoint_count = 0;
        TokenProgramSource source;
        if (!reader.u16(&version) || !reader.u16(&reserved) ||
            !reader.u32(&total) || !reader.u32(&semantic_length) ||
            !reader.u32(&state_length) || !reader.u32(&physical_length) ||
            !reader.u32(&source.artifact_id.value) ||
            !reader.u64(&source.offset) || !reader.u64(&source.length) ||
            version != 1 || reserved != 0 || total != wire.size())
            return package_error(CompatibilityError::PACKAGE_VERSION_UNSUPPORTED,
                                 "program package wire version or length is invalid");
        std::span<const uint8_t> digest;
        if (!reader.take(source.digest.bytes.size(), &digest))
            return package_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                 "program package token digest is truncated");
        std::copy(digest.begin(), digest.end(), source.digest.bytes.begin());
        if (!reader.u32(&endpoint_count) || endpoint_count != 2)
            return package_error(CompatibilityError::IR_REFERENCE_INVALID,
                                 "program package token endpoint count is invalid");
        std::vector<TokenEndpointBinding> endpoints(endpoint_count);
        for (TokenEndpointBinding& endpoint : endpoints) {
            uint32_t kind = 0;
            if (!reader.u32(&kind) ||
                !reader.u32(&endpoint.semantic_function_id) ||
                !reader.u32(&endpoint.semantic_value))
                return package_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                     "program package token endpoint is truncated");
            endpoint.kind = static_cast<TokenEndpointKind>(kind);
        }
        const uint64_t sections = static_cast<uint64_t>(semantic_length) +
                                  state_length + physical_length + 32;
        if (sections != reader.remaining())
            return package_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                 "program package section lengths are invalid");
        std::span<const uint8_t> semantic_wire, state_wire, physical_wire,
                                 expected_digest;
        if (!reader.take(semantic_length, &semantic_wire) ||
            !reader.take(state_length, &state_wire) ||
            !reader.take(physical_length, &physical_wire) ||
            !reader.take(32, &expected_digest) || !reader.finished())
            return package_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                 "program package sections are truncated");
        auto semantic = decode_program_wire(semantic_wire);
        if (const auto* report = std::get_if<CompatibilityReport>(&semantic))
            return *report;
        VerifiedProgram program =
            std::get<VerifiedProgram>(std::move(semantic));
        auto state = decode_state_schema_wire(state_wire, program);
        if (const auto* report = std::get_if<CompatibilityReport>(&state))
            return *report;
        auto physical_records =
            decode_physical_program_package_records(physical_wire);
        if (const auto* report =
                std::get_if<CompatibilityReport>(&physical_records))
            return *report;
        auto records = std::get<PhysicalProgramPackageRecords>(
            std::move(physical_records));
        auto package = build_program_package(
            std::move(physical), std::move(program),
            std::get<VerifiedStateSchema>(std::move(state)), source, endpoints,
            records.first, records.second);
        if (const auto* report = std::get_if<CompatibilityReport>(&package))
            return *report;
        VerifiedProgramPackage verified =
            std::get<VerifiedProgramPackage>(std::move(package));
        if (!std::equal(expected_digest.begin(), expected_digest.end(),
                        verified.digest().bytes.begin()))
            return package_error(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH,
                                 "program package closure digest does not match");
        return verified;
    } catch (const std::bad_alloc&) {
        return package_error(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                             "program package wire allocation failed");
    }
}

} // namespace Laplace
