#include "normalized_codec_program.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

namespace Laplace {
namespace {

constexpr size_t kMaximumNodes = 128;
constexpr size_t kMaximumMaps = 64;
constexpr size_t kMaximumAccesses = 4096;

CompatibilityReport error_report(CompatibilityError code,
                                  const char* detail) {
    return compatibility_report(code, detail);
}

bool less_access(const CodecCertificateAccessSummary& left,
                 const CodecCertificateAccessSummary& right) noexcept {
    if (left.byte_offset != right.byte_offset)
        return left.byte_offset < right.byte_offset;
    if (left.bit_offset != right.bit_offset)
        return left.bit_offset < right.bit_offset;
    if (left.width_bits != right.width_bits)
        return left.width_bits < right.width_bits;
    if (left.encoding != right.encoding)
        return static_cast<uint8_t>(left.encoding) <
               static_cast<uint8_t>(right.encoding);
    if (left.flags != right.flags) return left.flags < right.flags;
    return left.value_shift < right.value_shift;
}

bool same_access(const CodecCertificateAccessSummary& left,
                 const CodecCertificateAccessSummary& right) noexcept {
    return left.byte_offset == right.byte_offset &&
           left.bit_offset == right.bit_offset &&
           left.width_bits == right.width_bits &&
           left.encoding == right.encoding && left.flags == right.flags &&
           left.value_shift == right.value_shift;
}

struct MapGroup {
    uint8_t plane = 0;
    std::vector<CodecCertificateAccessSummary> records;
    size_t original_index = 0;
};

bool less_group(const MapGroup& left, const MapGroup& right) noexcept {
    if (left.plane != right.plane) return left.plane < right.plane;
    if (left.records.size() != right.records.size())
        return left.records.size() < right.records.size();
    for (size_t index = 0; index < left.records.size(); ++index) {
        if (same_access(left.records[index], right.records[index])) continue;
        return less_access(left.records[index], right.records[index]);
    }
    return left.original_index < right.original_index;
}

void append_u8(std::vector<uint8_t>& bytes, uint8_t value) {
    bytes.push_back(value);
}

void append_u16(std::vector<uint8_t>& bytes, uint16_t value) {
    append_u8(bytes, static_cast<uint8_t>(value));
    append_u8(bytes, static_cast<uint8_t>(value >> 8u));
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8)
        append_u8(bytes, static_cast<uint8_t>(value >> shift));
}

void append_summary(std::vector<uint8_t>& bytes,
                    const CodecCertificateSummary& summary) {
    append_u16(bytes, summary.abi_version);
    append_u16(bytes, summary.certificate_version);
    append_u8(bytes, static_cast<uint8_t>(summary.source_scalar));
    append_u8(bytes, static_cast<uint8_t>(summary.output_scalar));
    append_u8(bytes, static_cast<uint8_t>(summary.special_value_policy));
    append_u8(bytes, summary.rank_independent ? 1 : 0);
    append_u32(bytes, summary.unit_elements);
    append_u32(bytes, summary.unit_bytes);
    append_u32(bytes, summary.maximum_units);
    append_u32(bytes, summary.plane_count);
    append_u32(bytes, summary.node_count);
    append_u32(bytes, summary.constant_count);
    append_u32(bytes, summary.expression_depth);
    append_u8(bytes, summary.physical_layout_kind);
    append_u8(bytes, summary.physical_layout_packing);
    append_u8(bytes, summary.physical_layout_block_rank);
    append_u8(bytes, 0);
    append_u32(bytes, summary.physical_layout_block_elements);
    append_u32(bytes, summary.physical_layout_block_bytes);
    append_u8(bytes, summary.physical_quantization_kind);
    append_u8(bytes, 0);
    append_u16(bytes, 0);
    append_u32(bytes, summary.physical_quantization_block_elements);
    append_u32(bytes, summary.physical_quantization_block_bytes);
    append_u32(bytes, summary.physical_quantization_group_size);
    append_u32(bytes, summary.physical_quantization_required_plane_mask);
}

void append_plane(std::vector<uint8_t>& bytes,
                  const CodecCertificatePlaneSummary& plane) {
    append_u8(bytes, static_cast<uint8_t>(plane.role));
    append_u8(bytes, static_cast<uint8_t>(plane.scalar));
    append_u8(bytes, static_cast<uint8_t>(plane.storage_scalar));
    append_u8(bytes, plane.bit_order);
    append_u8(bytes, plane.byte_order);
    append_u16(bytes, plane.width_bits);
    append_u32(bytes, plane.elements_per_unit);
    append_u32(bytes, plane.bytes_per_unit);
    append_u32(bytes, plane.alignment);
    append_u32(bytes, plane.base);
    append_u32(bytes, plane.stride);
}

void append_node(std::vector<uint8_t>& bytes,
                 const CodecCertificateNodeSummary& node) {
    append_u8(bytes, static_cast<uint8_t>(node.operation));
    append_u8(bytes, static_cast<uint8_t>(node.value_type));
    append_u8(bytes, node.plane);
    append_u8(bytes, node.flags);
    append_u16(bytes, node.argument0);
    append_u16(bytes, node.argument1);
    append_u16(bytes, node.argument2);
    append_u32(bytes, node.immediate);
}

void append_map(std::vector<uint8_t>& bytes,
                const CodecCertificateAccessMapSummary& map) {
    append_u8(bytes, map.plane);
    append_u32(bytes, map.first);
    append_u32(bytes, map.count);
}

void append_access(std::vector<uint8_t>& bytes,
                   const CodecCertificateAccessSummary& access) {
    append_u32(bytes, access.byte_offset);
    append_u8(bytes, access.bit_offset);
    append_u8(bytes, access.width_bits);
    append_u8(bytes, static_cast<uint8_t>(access.encoding));
    append_u8(bytes, access.flags);
    append_u8(bytes, access.value_shift);
}

NormalizedCodecDigest semantic_digest(const NormalizedCodecProgram& program) {
    std::vector<uint8_t> bytes;
    bytes.reserve(128 + program.nodes.size() * 14 +
                  program.accesses.size() * 9);
    const std::array<uint8_t, 8> domain =
        {'L', 'A', 'P', 'N', 'O', 'R', 'M', 1};
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    append_summary(bytes, program.summary);
    append_u8(bytes, static_cast<uint8_t>(program.rounding));
    for (const auto& plane : program.planes) append_plane(bytes, plane);
    for (const auto& node : program.nodes) append_node(bytes, node);
    for (uint32_t value : program.constant_words) append_u32(bytes, value);
    for (const auto& map : program.access_maps) append_map(bytes, map);
    for (const auto& access : program.accesses) append_access(bytes, access);

    NormalizedCodecDigest digest{};
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
    return digest;
}

} // namespace

NormalizedCodecProgramResult normalize_codec_program(
    const CodecCertificate& certificate,
    NormalizedCodecRounding rounding) {
    try {
        const CodecCertificateSummary& source = certificate.summary();
        const auto& source_planes = certificate.plane_summaries();
        const auto& source_nodes = certificate.node_summaries();
        const auto& source_maps = certificate.access_map_summaries();
        const auto& source_accesses = certificate.access_summaries();
        if (source.abi_version == 0 || source.certificate_version == 0 ||
            source.unit_elements == 0 || source.unit_bytes == 0 ||
            source_planes.empty() || source_planes.size() > 8 ||
            source_nodes.empty() || source_nodes.size() > kMaximumNodes ||
            source_maps.empty() || source_maps.size() > kMaximumMaps ||
            source_accesses.empty() || source_accesses.size() > kMaximumAccesses)
            return error_report(CompatibilityError::IR_CONSTRAINT_FAILED,
                                "codec semantic summary is invalid");
        if (source.plane_count != source_planes.size() ||
            source.node_count != source_nodes.size() ||
            source.constant_count != certificate.constant_words().size())
            return error_report(CompatibilityError::IR_REFERENCE_INVALID,
                                "codec semantic counts do not match");
        if (rounding != NormalizedCodecRounding::ExactNearestEven)
            return error_report(CompatibilityError::IR_CONSTRAINT_FAILED,
                                "codec rounding policy is unsupported");

        std::vector<MapGroup> groups;
        groups.reserve(source_maps.size());
        for (size_t index = 0; index < source_maps.size(); ++index) {
            const auto& map = source_maps[index];
            const uint64_t end = static_cast<uint64_t>(map.first) + map.count;
            if (map.count == 0 || end > source_accesses.size())
                return error_report(CompatibilityError::IR_REFERENCE_INVALID,
                                    "codec access map is out of range");
            MapGroup group;
            group.plane = map.plane;
            group.original_index = index;
            group.records.insert(group.records.end(),
                                 source_accesses.begin() + map.first,
                                 source_accesses.begin() + end);
            groups.push_back(std::move(group));
        }
        std::sort(groups.begin(), groups.end(), less_group);
        std::vector<uint16_t> map_remap(source_maps.size(), 0xffff);
        for (size_t index = 0; index < groups.size(); ++index)
            map_remap[groups[index].original_index] =
                static_cast<uint16_t>(index);

        NormalizedCodecProgram normalized;
        normalized.abi_version = source.abi_version;
        normalized.certificate_version = source.certificate_version;
        normalized.summary = source;
        normalized.rounding = rounding;
        normalized.planes.assign(source_planes.begin(), source_planes.end());
        normalized.constant_words.assign(certificate.constant_words().begin(),
                                         certificate.constant_words().end());
        normalized.access_maps.reserve(groups.size());
        for (const MapGroup& group : groups) {
            const size_t first = normalized.accesses.size();
            normalized.accesses.insert(normalized.accesses.end(),
                                       group.records.begin(),
                                       group.records.end());
            if (first > std::numeric_limits<uint32_t>::max() ||
                group.records.size() > std::numeric_limits<uint32_t>::max())
                return error_report(CompatibilityError::IR_CONSTRAINT_FAILED,
                                    "codec access map is too large");
            normalized.access_maps.push_back(
                {group.plane, static_cast<uint32_t>(first),
                 static_cast<uint32_t>(group.records.size())});
        }

        std::vector<uint16_t> node_remap(source_nodes.size(), 0xffff);
        std::vector<uint8_t> depths;
        depths.reserve(source_nodes.size());
        for (size_t old_index = 0; old_index < source_nodes.size();
             ++old_index) {
            CodecCertificateNodeSummary node = source_nodes[old_index];
            const auto valid_argument = [old_index](uint16_t argument) {
                return argument == 0xffff || argument < old_index;
            };
            if (!valid_argument(node.argument0) ||
                !valid_argument(node.argument1) ||
                !valid_argument(node.argument2))
                return error_report(CompatibilityError::IR_REFERENCE_INVALID,
                                    "codec node reference is out of order");
            if (node.operation == CodecCertificateNodeOperation::LoadScalar ||
                node.operation == CodecCertificateNodeOperation::LoadBits) {
                if (node.immediate >= map_remap.size() ||
                    node.plane >= normalized.planes.size())
                    return error_report(CompatibilityError::IR_REFERENCE_INVALID,
                                        "codec load reference is invalid");
                node.immediate = map_remap[node.immediate];
            }
            for (uint16_t* argument : {&node.argument0, &node.argument1,
                                       &node.argument2}) {
                if (*argument == 0xffff) continue;
                *argument = node_remap[*argument];
                if (*argument == 0xffff)
                    return error_report(CompatibilityError::IR_REFERENCE_INVALID,
                                        "codec node remap is invalid");
            }
            const bool redundant_cast =
                node.operation == CodecCertificateNodeOperation::CastFloat &&
                node.argument0 != 0xffff &&
                source_nodes[old_index].argument0 < source_nodes.size() &&
                source_nodes[source_nodes[old_index].argument0].value_type ==
                    CodecCertificateNodeValueType::Float;
            if (redundant_cast) {
                node_remap[old_index] = node.argument0;
                continue;
            }
            node_remap[old_index] =
                static_cast<uint16_t>(normalized.nodes.size());
            normalized.nodes.push_back(node);
            uint8_t depth = 1;
            for (uint16_t argument :
                 {node.argument0, node.argument1, node.argument2}) {
                if (argument != 0xffff) {
                    if (argument >= depths.size())
                        return error_report(
                            CompatibilityError::IR_REFERENCE_INVALID,
                            "codec node depth is invalid");
                    depth = std::max<uint8_t>(
                        depth, static_cast<uint8_t>(depths[argument] + 1));
                }
            }
            depths.push_back(depth);
        }
        if (normalized.nodes.empty())
            return error_report(CompatibilityError::IR_REFERENCE_INVALID,
                                "codec graph has no output node");
        normalized.summary.node_count =
            static_cast<uint32_t>(normalized.nodes.size());
        normalized.summary.plane_count =
            static_cast<uint32_t>(normalized.planes.size());
        normalized.summary.constant_count =
            static_cast<uint32_t>(normalized.constant_words.size());
        normalized.summary.expression_depth =
            depths.empty() ? 0 : *std::max_element(depths.begin(), depths.end());
        normalized.semantic_signature = semantic_digest(normalized);
        return normalized;
    } catch (const std::bad_alloc&) {
        return error_report(
            CompatibilityError::IMPORT_SCHEMA_LIMIT,
            "codec semantic normalization exceeded memory limits");
    }
}

NormalizedCodecProvenance normalized_codec_provenance(
    const CodecCertificate& certificate) noexcept {
    return {certificate.identity().abi_version, certificate.identity().digest};
}

} // namespace Laplace
