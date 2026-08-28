#include "compat_rule.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cstring>
#include <limits>

namespace Laplace {

namespace {

constexpr uint32_t kRuleMaxBytes = 1024 * 1024;
constexpr uint32_t kRuleMaxString = 64 * 1024;
constexpr uint32_t kRuleMaxMetadata = 4096;
constexpr uint32_t kRuleMaxPatterns = 65536;
constexpr uint32_t kRuleMaxVector = 1048576;

void append_u16(std::vector<uint8_t>& out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value));
    out.push_back(static_cast<uint8_t>(value >> 8));
}

void append_u32(std::vector<uint8_t>& out, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8) out.push_back(static_cast<uint8_t>(value >> shift));
}

void append_u64(std::vector<uint8_t>& out, uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8) out.push_back(static_cast<uint8_t>(value >> shift));
}

bool read_u16(const std::vector<uint8_t>& bytes, size_t& offset, size_t end, uint16_t& value) {
    if (offset > end || end - offset < 2) return false;
    value = static_cast<uint16_t>(bytes[offset]) | (static_cast<uint16_t>(bytes[offset + 1]) << 8);
    offset += 2;
    return true;
}

bool read_u32(const std::vector<uint8_t>& bytes, size_t& offset, size_t end, uint32_t& value) {
    if (offset > end || end - offset < 4) return false;
    value = 0;
    for (unsigned shift = 0; shift != 32; shift += 8) value |= static_cast<uint32_t>(bytes[offset++]) << shift;
    return true;
}

bool read_u64(const std::vector<uint8_t>& bytes, size_t& offset, size_t end, uint64_t& value) {
    if (offset > end || end - offset < 8) return false;
    value = 0;
    for (unsigned shift = 0; shift != 64; shift += 8) value |= static_cast<uint64_t>(bytes[offset++]) << shift;
    return true;
}

void append_padding(std::vector<uint8_t>& out) {
    while (out.size() % 8 != 0) out.push_back(0);
}

bool consume_padding(const std::vector<uint8_t>& bytes, size_t& offset, size_t end) {
    while (offset % 8 != 0) {
        if (offset == end || bytes[offset++] != 0) return false;
    }
    return true;
}

std::array<uint8_t, 32> sha256(const uint8_t* data, size_t size) {
    std::array<uint8_t, 32> output{};
    CC_SHA256(data, static_cast<CC_LONG>(size), output.data());
    return output;
}

CompatibilityReport rule_error(CompatibilityError error) {
    CompatibilityReport report = package_report(error);
    report.stage = CompatibilityStage::Rule;
    return report;
}

CompatibilityReport import_error(CompatibilityError error, std::string detail = {}) {
    CompatibilityReport report = package_report(error, std::move(detail));
    report.stage = CompatibilityStage::Import;
    return report;
}

bool valid_utf8(const std::string& value) {
    size_t index = 0;
    while (index < value.size()) {
        uint8_t first = static_cast<uint8_t>(value[index++]);
        if (first < 0x80) continue;
        uint8_t need = first >= 0xf0 ? 3 : first >= 0xe0 ? 2 : first >= 0xc2 ? 1 : 255;
        if (need == 255 || value.size() - index < need) return false;
        for (uint8_t count = 0; count != need; ++count) {
            if ((static_cast<uint8_t>(value[index++]) & 0xc0) != 0x80) return false;
        }
    }
    return true;
}

bool valid_string(const std::string& value, uint32_t limit = kRuleMaxString) {
    return value.size() <= limit && valid_utf8(value);
}

void append_string(std::vector<uint8_t>& out, const std::string& value) {
    append_u32(out, static_cast<uint32_t>(value.size()));
    out.insert(out.end(), value.begin(), value.end());
}

bool read_string(const std::vector<uint8_t>& bytes, size_t& offset, size_t end, std::string& value) {
    uint32_t length = 0;
    if (!read_u32(bytes, offset, end, length) || length > kRuleMaxString || offset > end || end - offset < length) return false;
    value.assign(reinterpret_cast<const char*>(bytes.data() + offset), length);
    offset += length;
    return valid_string(value);
}

void append_dimension(std::vector<uint8_t>& out, const Dimension& dimension) {
    out.push_back(static_cast<uint8_t>(dimension.kind));
    out.insert(out.end(), 7, 0);
    append_u64(out, dimension.constant_or_symbol);
}

bool read_dimension(const std::vector<uint8_t>& bytes, size_t& offset, size_t end, Dimension& dimension) {
    if (offset == end) return false;
    uint8_t kind = bytes[offset++];
    if (end - offset < 7) return false;
    for (unsigned index = 0; index != 7; ++index) if (bytes[offset++] != 0) return false;
    if (!read_u64(bytes, offset, end, dimension.constant_or_symbol)) return false;
    dimension.kind = static_cast<DimensionKind>(kind);
    return (dimension.kind == DimensionKind::Constant || dimension.kind == DimensionKind::Symbol) &&
           dimension.constant_or_symbol != 0;
}

void append_frame(std::vector<uint8_t>& out, uint16_t tag, const std::vector<uint8_t>& payload) {
    append_u16(out, tag);
    append_u16(out, 0);
    append_u32(out, static_cast<uint32_t>(payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
    append_padding(out);
}

bool read_frame(const std::vector<uint8_t>& bytes, size_t& offset, size_t end,
                uint16_t& tag, size_t& payload_begin, size_t& payload_end) {
    uint16_t flags = 0;
    uint32_t length = 0;
    if (!read_u16(bytes, offset, end, tag) || !read_u16(bytes, offset, end, flags) ||
        !read_u32(bytes, offset, end, length) || flags != 0 || offset > end || end - offset < length) return false;
    payload_begin = offset;
    payload_end = offset + length;
    offset = payload_end;
    return consume_padding(bytes, offset, end);
}

bool valid_pattern(const std::string& pattern) {
    if (!valid_string(pattern)) return false;
    size_t marker = pattern.find("{d}");
    return marker != std::string::npos && marker == pattern.rfind("{d}") &&
           pattern.find('{') == marker && pattern.find('}') == marker + 2;
}

bool zero_digest(const Sha256Digest& digest) {
    return std::all_of(digest.bytes.begin(), digest.bytes.end(), [](uint8_t value) { return value == 0; });
}

bool decimal_capture_matches(const std::string& pattern, const std::string& name, uint32_t& capture) {
    size_t marker = pattern.find("{d}");
    const std::string prefix = pattern.substr(0, marker);
    const std::string suffix = pattern.substr(marker + 3);
    if (name.size() < prefix.size() + suffix.size() + 1 || name.compare(0, prefix.size(), prefix) != 0 ||
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) return false;
    size_t begin = prefix.size();
    size_t end = name.size() - suffix.size();
    if (name[begin] == '0' && end - begin != 1) return false;
    uint64_t value = 0;
    for (size_t index = begin; index != end; ++index) {
        char character = name[index];
        if (character < '0' || character > '9' || value > (std::numeric_limits<uint64_t>::max() - 9) / 10) return false;
        value = value * 10 + static_cast<uint64_t>(character - '0');
    }
    if (value > std::numeric_limits<uint32_t>::max()) return false;
    capture = static_cast<uint32_t>(value);
    return true;
}

bool dimensions_match(const std::vector<Dimension>& expected, const std::vector<uint64_t>& actual) {
    if (expected.size() != actual.size()) return false;
    for (size_t index = 0; index != expected.size(); ++index) {
        if (expected[index].kind != DimensionKind::Constant || expected[index].constant_or_symbol != actual[index]) return false;
    }
    return true;
}

bool logical_dimensions_match(const SemanticTensor& tensor, const PackageTensorEvidence& source,
                              TensorTransformKind transform) {
    if (transform == TensorTransformKind::Identity) return dimensions_match(tensor.dimensions, source.dimensions);
    if (transform != TensorTransformKind::LogicalTranspose || source.dimensions.size() != 2 || tensor.dimensions.size() != 2) return false;
    return tensor.dimensions[0].kind == DimensionKind::Constant && tensor.dimensions[1].kind == DimensionKind::Constant &&
           tensor.dimensions[0].constant_or_symbol == source.dimensions[1] &&
           tensor.dimensions[1].constant_or_symbol == source.dimensions[0];
}

bool predicate_matches(const MetadataPredicate& predicate, const PackageEvidence& package) {
    auto it = package.metadata.find(predicate.key);
    if (predicate.kind == MetadataPredicateKind::Exists) return it != package.metadata.end();
    if (it == package.metadata.end()) return false;
    switch (predicate.kind) {
    case MetadataPredicateKind::ExactU64: {
        auto* value = std::get_if<uint64_t>(&it->second);
        return value && *value == predicate.exact_u64;
    }
    case MetadataPredicateKind::ExactString: {
        auto* value = std::get_if<std::string>(&it->second);
        return value && *value == predicate.exact_string;
    }
    case MetadataPredicateKind::EnumU64: {
        auto* value = std::get_if<uint64_t>(&it->second);
        return value && std::binary_search(predicate.enum_u64.begin(), predicate.enum_u64.end(), *value);
    }
    case MetadataPredicateKind::ListLength: {
        auto* value = std::get_if<std::vector<uint64_t>>(&it->second);
        return value && value->size() == predicate.list_length;
    }
    case MetadataPredicateKind::Digest: {
        auto* value = std::get_if<Sha256Digest>(&it->second);
        return value && *value == predicate.digest;
    }
    case MetadataPredicateKind::Exists:
        return true;
    }
    return false;
}

enum class MatchResult { NoMatch, Unmapped, Match };

MatchResult match_rule(const CompatibilityRule& rule, const PackageEvidence& package, SemanticModel& model) {
    if (rule.schema_major != 1 || rule.schema_minor != 0 || rule.evaluator_major != 1 || rule.evaluator_minor != 0 ||
        rule.package_format != PackageFormat::Gguf || rule.metadata.size() > kRuleMaxMetadata ||
        rule.tensors.size() > kRuleMaxPatterns) return MatchResult::NoMatch;
    if (zero_digest(semantic_model_digest(rule.semantic_template)) ||
        (!zero_digest(rule.semantic_template_digest) && semantic_model_digest(rule.semantic_template) != rule.semantic_template_digest)) return MatchResult::NoMatch;
    for (const MetadataPredicate& predicate : rule.metadata) if (!predicate_matches(predicate, package)) return MatchResult::NoMatch;
    std::vector<bool> consumed(package.tensors.size());
    model = rule.semantic_template;
    for (const TensorPattern& pattern : rule.tensors) {
        if ((pattern.kind != TensorPatternKind::AnchoredDecimalCapture && pattern.kind != TensorPatternKind::ExactName) ||
            (pattern.kind == TensorPatternKind::AnchoredDecimalCapture && !valid_pattern(pattern.pattern)) ||
            (pattern.kind == TensorPatternKind::ExactName && !valid_string(pattern.pattern)) ||
            pattern.repetition_count == 0 || pattern.template_stride == 0 ||
            pattern.template_id > model.tensors.size() ||
            pattern.repetition_count > (model.tensors.size() - pattern.template_id + pattern.template_stride - 1) / pattern.template_stride) return MatchResult::NoMatch;
        std::vector<size_t> found(pattern.repetition_count, package.tensors.size());
        for (size_t index = 0; index != package.tensors.size(); ++index) {
            const PackageTensorEvidence& tensor = package.tensors[index];
            uint32_t capture = 0;
            bool name_matches = pattern.kind == TensorPatternKind::AnchoredDecimalCapture
                                    ? decimal_capture_matches(pattern.pattern, tensor.name, capture)
                                    : (capture = 0, tensor.name == pattern.pattern);
            if (consumed[index] || !name_matches || capture >= pattern.repetition_count ||
                !dimensions_match(pattern.dimensions, tensor.dimensions) || tensor.storage_type != pattern.storage_type ||
                tensor.layout != pattern.layout || tensor.quantization != pattern.quantization) continue;
            if (found[capture] != package.tensors.size()) return MatchResult::NoMatch;
            found[capture] = index;
        }
        for (uint32_t capture = 0; capture != pattern.repetition_count; ++capture) {
            if (found[capture] == package.tensors.size()) return MatchResult::NoMatch;
            consumed[found[capture]] = true;
            const PackageTensorEvidence& source = package.tensors[found[capture]];
            SemanticTensor& destination = model.tensors[pattern.template_id + capture * pattern.template_stride];
            if (destination.role != pattern.role || destination.logical_type != pattern.logical_type ||
                destination.layout.kind != pattern.layout || destination.quantization.kind != pattern.quantization ||
                !logical_dimensions_match(destination, source, pattern.transform)) return MatchResult::NoMatch;
            if (destination.layout.kind == PhysicalLayoutKind::GgufBlocked &&
                (destination.layout.block_elements != source.block_elements ||
                 destination.layout.block_bytes != source.block_bytes ||
                 destination.quantization.block_elements != source.block_elements ||
                 destination.quantization.block_bytes != source.block_bytes ||
                 destination.quantization.group_size != source.block_elements)) return MatchResult::NoMatch;
            if (destination.planes.size() != 1 || destination.planes[0].kind != PlaneKind::Values) return MatchResult::NoMatch;
            destination.planes[0] = {PlaneKind::Values, source.storage_type, source.artifact_id,
                                     source.offset, source.length, source.alignment, 0};
            if (capture != 0 && !pattern.alias_template_ids.empty()) return MatchResult::NoMatch;
            for (uint32_t alias_id : pattern.alias_template_ids) {
                if (alias_id >= model.tensors.size() || model.tensors[alias_id].planes.size() != 1 ||
                    !logical_dimensions_match(model.tensors[alias_id], source, pattern.transform)) return MatchResult::NoMatch;
                model.tensors[alias_id].planes[0] = {PlaneKind::Values, source.storage_type, source.artifact_id,
                                                     source.offset, source.length, source.alignment, 1};
            }
        }
    }
    for (bool was_consumed : consumed) if (!was_consumed) return MatchResult::Unmapped;
    auto encoded = encode_semantic_model(model);
    return std::holds_alternative<std::vector<uint8_t>>(encoded) ? MatchResult::Match : MatchResult::NoMatch;
}

} // namespace

RuleEncodeResult encode_compatibility_rule(const CompatibilityRule& rule) {
    if (rule.schema_major != 1 || rule.schema_minor != 0 || rule.evaluator_major != 1 || rule.evaluator_minor != 0 ||
        rule.rule_id.empty() || rule.rule_id.size() > 128 || !valid_string(rule.rule_id) ||
        rule.package_format != PackageFormat::Gguf || static_cast<uint16_t>(rule.qualification_state) < 1 ||
        static_cast<uint16_t>(rule.qualification_state) > 4 || rule.metadata.size() > kRuleMaxMetadata ||
        rule.tensors.size() > kRuleMaxPatterns || rule.constraints.size() > kRuleMaxVector ||
        rule.capabilities.size() > kRuleMaxVector || rule.fallbacks.size() > kRuleMaxVector) return rule_error(CompatibilityError::RULE_LIMIT_EXCEEDED);
    const Sha256Digest semantics = semantic_model_digest(rule.semantic_template);
    if (zero_digest(semantics)) {
        return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
    }
    const Sha256Digest semantic_template_hash = zero_digest(rule.semantic_template_digest)
                                                    ? semantics : rule.semantic_template_digest;

    std::vector<uint8_t> body;
    append_u16(body, static_cast<uint16_t>(rule.package_format));
    append_u16(body, static_cast<uint16_t>(rule.qualification_state));
    append_u32(body, static_cast<uint32_t>(rule.metadata.size()));
    append_u32(body, static_cast<uint32_t>(rule.tensors.size()));
    append_u32(body, static_cast<uint32_t>(rule.semantic_template.values.size())); // symbol/value bindings
    append_u32(body, static_cast<uint32_t>(rule.constraints.size()));
    append_u32(body, static_cast<uint32_t>(rule.semantic_template.layers.size())); // layer templates
    append_u32(body, rule.semantic_template.input_values_count); // input template values
    append_u32(body, rule.semantic_template.output_values_count); // output template values
    append_u32(body, static_cast<uint32_t>(rule.semantic_template.states.size())); // state templates
    append_u32(body, static_cast<uint32_t>(rule.capabilities.size()));
    append_u32(body, static_cast<uint32_t>(rule.fallbacks.size()));
    append_u32(body, static_cast<uint32_t>(rule.semantic_template.operators.size()));

    for (const MetadataPredicate& predicate : rule.metadata) {
        if (!valid_string(predicate.key)) return rule_error(CompatibilityError::RULE_LIMIT_EXCEEDED);
        std::vector<uint8_t> payload;
        append_string(payload, predicate.key);
        switch (predicate.kind) {
        case MetadataPredicateKind::Exists: break;
        case MetadataPredicateKind::ExactU64: append_u64(payload, predicate.exact_u64); break;
        case MetadataPredicateKind::ExactString:
            if (!valid_string(predicate.exact_string)) return rule_error(CompatibilityError::RULE_LIMIT_EXCEEDED);
            append_string(payload, predicate.exact_string); break;
        case MetadataPredicateKind::EnumU64:
            if (predicate.enum_u64.empty() || !std::is_sorted(predicate.enum_u64.begin(), predicate.enum_u64.end()) ||
                std::adjacent_find(predicate.enum_u64.begin(), predicate.enum_u64.end()) != predicate.enum_u64.end()) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
            append_u32(payload, static_cast<uint32_t>(predicate.enum_u64.size()));
            for (uint64_t value : predicate.enum_u64) append_u64(payload, value);
            break;
        case MetadataPredicateKind::ListLength: append_u32(payload, predicate.list_length); break;
        case MetadataPredicateKind::Digest: payload.insert(payload.end(), predicate.digest.bytes.begin(), predicate.digest.bytes.end()); break;
        }
        append_frame(body, static_cast<uint16_t>(predicate.kind), payload);
    }
    for (const TensorPattern& pattern : rule.tensors) {
        if ((pattern.kind != TensorPatternKind::AnchoredDecimalCapture && pattern.kind != TensorPatternKind::ExactName) ||
            (pattern.kind == TensorPatternKind::AnchoredDecimalCapture && !valid_pattern(pattern.pattern)) ||
            (pattern.kind == TensorPatternKind::ExactName && !valid_string(pattern.pattern)) ||
            pattern.dimensions.empty() || pattern.dimensions.size() > 8 || pattern.required_plane_mask == 0 ||
            static_cast<uint16_t>(pattern.role) < 1 || static_cast<uint16_t>(pattern.role) > 27 ||
            static_cast<uint16_t>(pattern.logical_type) < 1 || static_cast<uint16_t>(pattern.logical_type) > 5 ||
            static_cast<uint16_t>(pattern.layout) < 1 || static_cast<uint16_t>(pattern.layout) > 2 ||
            static_cast<uint16_t>(pattern.quantization) > 1 || static_cast<uint16_t>(pattern.storage_type) < 1 ||
            static_cast<uint16_t>(pattern.storage_type) > 5 || static_cast<uint16_t>(pattern.transform) < 1 ||
            static_cast<uint16_t>(pattern.transform) > 6 || pattern.alias_template_ids.size() > 64 ||
            pattern.repetition_count == 0 || pattern.template_stride == 0) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        std::vector<uint8_t> payload;
        append_u32(payload, pattern.template_id);
        append_u16(payload, static_cast<uint16_t>(pattern.role));
        append_u16(payload, static_cast<uint16_t>(pattern.logical_type));
        append_u16(payload, static_cast<uint16_t>(pattern.layout));
        append_u16(payload, static_cast<uint16_t>(pattern.quantization));
        payload.push_back(static_cast<uint8_t>(pattern.dimensions.size()));
        payload.push_back(pattern.kind == TensorPatternKind::AnchoredDecimalCapture ? 1 : 0);
        append_u16(payload, 0);
        append_u32(payload, pattern.capture_symbol);
        append_u32(payload, pattern.required_plane_mask);
        append_u16(payload, static_cast<uint16_t>(pattern.storage_type));
        append_u16(payload, 0);
        append_string(payload, pattern.pattern);
        for (const Dimension& dimension : pattern.dimensions) append_dimension(payload, dimension);
        append_u32(payload, 1);
        append_u16(payload, static_cast<uint16_t>(pattern.transform));
        append_u16(payload, 0);
        append_u32(payload, 0);
        while (payload.size() % 8 != 0) payload.push_back(0);
        append_u32(payload, static_cast<uint32_t>(pattern.alias_template_ids.size()));
        for (uint32_t alias_id : pattern.alias_template_ids) append_u32(payload, alias_id);
        append_u32(payload, pattern.repetition_count);
        append_u32(payload, pattern.template_stride);
        append_frame(body, static_cast<uint16_t>(pattern.kind), payload);
    }
    for (const SemanticConstraint& constraint : rule.constraints) {
        std::vector<uint8_t> payload;
        append_u16(payload, static_cast<uint16_t>(constraint.kind)); append_u16(payload, 0);
        payload.push_back(static_cast<uint8_t>(constraint.lhs_kind)); payload.push_back(static_cast<uint8_t>(constraint.rhs_kind));
        payload.insert(payload.end(), 6, 0); append_u32(payload, constraint.lhs_id); append_u32(payload, constraint.rhs_id);
        append_u32(payload, constraint.lhs_axis); append_u32(payload, constraint.rhs_axis);
        append_u64(payload, constraint.constant); append_u64(payload, constraint.divisor);
        append_frame(body, 1, payload);
    }
    for (const CapabilityRequirement& capability : rule.capabilities) {
        std::vector<uint8_t> payload; append_u16(payload, static_cast<uint16_t>(capability.capability));
        append_u16(payload, capability.minimum_version); append_u32(payload, capability.flags); append_frame(body, 1, payload);
    }
    for (const SemanticFallback& fallback : rule.fallbacks) {
        std::vector<uint8_t> payload; append_u16(payload, static_cast<uint16_t>(fallback.kind));
        append_u16(payload, static_cast<uint16_t>(fallback.phase)); append_u16(payload, static_cast<uint16_t>(fallback.numerical_class));
        append_u16(payload, fallback.flags); append_frame(body, 1, payload);
    }
    auto semantic_bytes = encode_semantic_model(rule.semantic_template);
    if (!std::holds_alternative<std::vector<uint8_t>>(semantic_bytes)) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
    append_frame(body, 7, std::get<std::vector<uint8_t>>(semantic_bytes));
    body.insert(body.end(), semantic_template_hash.bytes.begin(), semantic_template_hash.bytes.end());
    if (body.size() > kRuleMaxBytes) return rule_error(CompatibilityError::RULE_LIMIT_EXCEEDED);

    uint32_t header_length = static_cast<uint32_t>((80 + rule.rule_id.size() + 7) & ~size_t(7));
    std::vector<uint8_t> output;
    output.reserve(header_length + body.size());
    output.insert(output.end(), {'L', 'A', 'P', 'R', 'U', 'L', '1', '0'});
    append_u16(output, 1); append_u16(output, 0); append_u16(output, 1); append_u16(output, 0);
    append_u32(output, rule.rule_revision); append_u32(output, header_length);
    append_u64(output, body.size()); append_u64(output, header_length + body.size());
    const auto digest = sha256(body.data(), body.size());
    output.insert(output.end(), digest.begin(), digest.end());
    append_u32(output, static_cast<uint32_t>(rule.rule_id.size())); append_u32(output, 0);
    output.insert(output.end(), rule.rule_id.begin(), rule.rule_id.end());
    output.insert(output.end(), header_length - output.size(), 0);
    output.insert(output.end(), body.begin(), body.end());
    return output;
}

RuleDecodeResult decode_compatibility_rule(const std::vector<uint8_t>& bytes) {
    if (bytes.size() < 128 || std::memcmp(bytes.data(), "LAPRUL10", 8) != 0) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
    size_t offset = 8; uint16_t schema_major = 0, schema_minor = 0, evaluator_major = 0, evaluator_minor = 0;
    uint32_t header_length = 0, id_length = 0, flags = 0, revision = 0; uint64_t body_length = 0, total_length = 0;
    if (!read_u16(bytes, offset, bytes.size(), schema_major) || !read_u16(bytes, offset, bytes.size(), schema_minor) ||
        !read_u16(bytes, offset, bytes.size(), evaluator_major) || !read_u16(bytes, offset, bytes.size(), evaluator_minor) ||
        !read_u32(bytes, offset, bytes.size(), revision) || !read_u32(bytes, offset, bytes.size(), header_length) ||
        !read_u64(bytes, offset, bytes.size(), body_length) || !read_u64(bytes, offset, bytes.size(), total_length) ||
        schema_major != 1 || schema_minor != 0 || evaluator_major != 1 || evaluator_minor != 0 ||
        header_length < 80 || header_length % 8 != 0 || total_length != bytes.size() ||
        body_length != bytes.size() - header_length || body_length > kRuleMaxBytes || bytes.size() - offset < 32) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
    std::array<uint8_t, 32> expected{}; std::memcpy(expected.data(), bytes.data() + offset, expected.size()); offset += 32;
    if (!read_u32(bytes, offset, bytes.size(), id_length) || !read_u32(bytes, offset, bytes.size(), flags) || flags != 0 || id_length == 0 || id_length > 128 ||
        offset > header_length || header_length - offset < id_length) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
    CompatibilityRule rule;
    rule.schema_major = schema_major; rule.schema_minor = schema_minor; rule.evaluator_major = evaluator_major; rule.evaluator_minor = evaluator_minor; rule.rule_revision = revision;
    rule.rule_id.assign(reinterpret_cast<const char*>(bytes.data() + offset), id_length); offset += id_length;
    if (!valid_string(rule.rule_id)) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
    while (offset < header_length) if (bytes[offset++] != 0) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
    const size_t body_end = bytes.size();
    if (expected != sha256(bytes.data() + header_length, static_cast<size_t>(body_length))) return rule_error(CompatibilityError::PACKAGE_CHECKSUM_MISMATCH);
    offset = header_length;
    uint16_t format = 0, qualification = 0; std::array<uint32_t, 10> counts{}; uint32_t max_ops = 0;
    if (!read_u16(bytes, offset, body_end, format) || !read_u16(bytes, offset, body_end, qualification)) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
    for (uint32_t& count : counts) if (!read_u32(bytes, offset, body_end, count)) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
    if (!read_u32(bytes, offset, body_end, max_ops) || format != static_cast<uint16_t>(PackageFormat::Gguf) || qualification < 1 || qualification > 4 ||
        counts[0] > kRuleMaxMetadata || counts[1] > kRuleMaxPatterns || counts[2] > kRuleMaxVector || counts[3] > kRuleMaxVector ||
        counts[4] > 4096 || counts[5] > kRuleMaxVector || counts[6] > kRuleMaxVector || counts[7] > kRuleMaxVector ||
        counts[8] > kRuleMaxVector || counts[9] > kRuleMaxVector || max_ops > kRuleMaxVector) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
    rule.package_format = static_cast<PackageFormat>(format); rule.qualification_state = static_cast<RuleQualificationState>(qualification);
    for (uint32_t index = 0; index != counts[0]; ++index) {
        uint16_t tag = 0; size_t begin = 0, end = 0;
        if (!read_frame(bytes, offset, body_end, tag, begin, end) || tag < 1 || tag > 6) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        MetadataPredicate predicate; predicate.kind = static_cast<MetadataPredicateKind>(tag); size_t cursor = begin;
        if (!read_string(bytes, cursor, end, predicate.key)) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        switch (predicate.kind) {
        case MetadataPredicateKind::Exists: break;
        case MetadataPredicateKind::ExactU64: if (!read_u64(bytes, cursor, end, predicate.exact_u64)) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED); break;
        case MetadataPredicateKind::ExactString: if (!read_string(bytes, cursor, end, predicate.exact_string)) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED); break;
        case MetadataPredicateKind::EnumU64: { uint32_t count = 0; if (!read_u32(bytes, cursor, end, count) || count == 0 || count > kRuleMaxVector) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED); predicate.enum_u64.resize(count); for (uint64_t& value : predicate.enum_u64) if (!read_u64(bytes, cursor, end, value)) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED); if (!std::is_sorted(predicate.enum_u64.begin(), predicate.enum_u64.end()) || std::adjacent_find(predicate.enum_u64.begin(), predicate.enum_u64.end()) != predicate.enum_u64.end()) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED); break; }
        case MetadataPredicateKind::ListLength: if (!read_u32(bytes, cursor, end, predicate.list_length)) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED); break;
        case MetadataPredicateKind::Digest: if (end - cursor != 32) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED); std::memcpy(predicate.digest.bytes.data(), bytes.data() + cursor, 32); cursor += 32; break;
        }
        if (cursor != end) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        rule.metadata.push_back(std::move(predicate));
    }
    for (uint32_t index = 0; index != counts[1]; ++index) {
        uint16_t tag = 0; size_t begin = 0, end = 0;
        if (!read_frame(bytes, offset, body_end, tag, begin, end) || tag < static_cast<uint16_t>(TensorPatternKind::AnchoredDecimalCapture) || tag > static_cast<uint16_t>(TensorPatternKind::ExactName)) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        TensorPattern pattern; size_t cursor = begin; uint16_t role = 0, type = 0, layout = 0, quantization = 0, flags = 0; uint8_t rank = 0, captures = 0;
        if (!read_u32(bytes, cursor, end, pattern.template_id) || !read_u16(bytes, cursor, end, role) || !read_u16(bytes, cursor, end, type) ||
            !read_u16(bytes, cursor, end, layout) || !read_u16(bytes, cursor, end, quantization) || cursor == end) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        rank = bytes[cursor++]; if (cursor == end) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED); captures = bytes[cursor++];
        uint16_t storage = 0, storage_reserved = 0;
        if (!read_u16(bytes, cursor, end, flags) || !read_u32(bytes, cursor, end, pattern.capture_symbol) || !read_u32(bytes, cursor, end, pattern.required_plane_mask) ||
            !read_u16(bytes, cursor, end, storage) || !read_u16(bytes, cursor, end, storage_reserved) || storage_reserved != 0 ||
            !read_string(bytes, cursor, end, pattern.pattern) || rank == 0 || rank > 8 || flags != 0) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        pattern.kind = static_cast<TensorPatternKind>(tag);
        if ((pattern.kind == TensorPatternKind::AnchoredDecimalCapture && (captures != 1 || !valid_pattern(pattern.pattern))) ||
            (pattern.kind == TensorPatternKind::ExactName && (captures != 0 || !valid_string(pattern.pattern)))) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        pattern.role = static_cast<TensorRole>(role); pattern.logical_type = static_cast<ScalarType>(type); pattern.layout = static_cast<PhysicalLayoutKind>(layout); pattern.quantization = static_cast<QuantizationKind>(quantization); pattern.storage_type = static_cast<ScalarType>(storage);
        pattern.dimensions.resize(rank); for (Dimension& dimension : pattern.dimensions) if (!read_dimension(bytes, cursor, end, dimension)) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        uint32_t transform_count = 0; uint16_t transform = 0, transform_flags = 0; uint32_t transform_length = 0;
        if (!read_u32(bytes, cursor, end, transform_count) || transform_count != 1 || !read_u16(bytes, cursor, end, transform) ||
            !read_u16(bytes, cursor, end, transform_flags) || !read_u32(bytes, cursor, end, transform_length) ||
            transform_flags != 0 || transform_length != 0 || static_cast<uint16_t>(transform) < 1 || static_cast<uint16_t>(transform) > 6) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        while (cursor % 8 != 0) { if (cursor == end || bytes[cursor++] != 0) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED); }
        uint32_t alias_count = 0;
        if (!read_u32(bytes, cursor, end, alias_count) || alias_count > 64) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        pattern.alias_template_ids.resize(alias_count);
        for (uint32_t& alias_id : pattern.alias_template_ids) if (!read_u32(bytes, cursor, end, alias_id)) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        if (!read_u32(bytes, cursor, end, pattern.repetition_count) || !read_u32(bytes, cursor, end, pattern.template_stride) ||
            pattern.repetition_count == 0 || pattern.template_stride == 0) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        if (cursor != end) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        pattern.transform = static_cast<TensorTransformKind>(transform);
        rule.tensors.push_back(std::move(pattern));
    }
    for (uint32_t index = 0; index != counts[3]; ++index) {
        uint16_t tag = 0; size_t begin = 0, end = 0;
        if (!read_frame(bytes, offset, body_end, tag, begin, end) || tag != 1 || end - begin != 44) {
            return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        }
        SemanticConstraint constraint;
        size_t cursor = begin;
        uint16_t kind = 0, flags = 0;
        if (!read_u16(bytes, cursor, end, kind) || !read_u16(bytes, cursor, end, flags) || flags != 0 ||
            cursor == end) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        uint8_t lhs_kind = bytes[cursor++];
        if (cursor == end) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        uint8_t rhs_kind = bytes[cursor++];
        if (end - cursor < 6) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        for (uint32_t reserved = 0; reserved != 6; ++reserved) if (bytes[cursor++] != 0) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        if (!read_u32(bytes, cursor, end, constraint.lhs_id) || !read_u32(bytes, cursor, end, constraint.rhs_id) ||
            !read_u32(bytes, cursor, end, constraint.lhs_axis) || !read_u32(bytes, cursor, end, constraint.rhs_axis) ||
            !read_u64(bytes, cursor, end, constraint.constant) || !read_u64(bytes, cursor, end, constraint.divisor) || cursor != end ||
            kind < static_cast<uint16_t>(ConstraintKind::Equal) || kind > static_cast<uint16_t>(ConstraintKind::TensorBytesEqual) ||
            lhs_kind < static_cast<uint8_t>(ConstraintOperandKind::Dimension) || lhs_kind > static_cast<uint8_t>(ConstraintOperandKind::Constant) ||
            rhs_kind < static_cast<uint8_t>(ConstraintOperandKind::Dimension) || rhs_kind > static_cast<uint8_t>(ConstraintOperandKind::Constant)) {
            return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        }
        constraint.kind = static_cast<ConstraintKind>(kind);
        constraint.lhs_kind = static_cast<ConstraintOperandKind>(lhs_kind);
        constraint.rhs_kind = static_cast<ConstraintOperandKind>(rhs_kind);
        rule.constraints.push_back(constraint);
    }
    for (uint32_t index = 0; index != counts[8]; ++index) {
        uint16_t tag = 0; size_t begin = 0, end = 0;
        CapabilityRequirement capability;
        uint16_t value = 0;
        if (!read_frame(bytes, offset, body_end, tag, begin, end) || tag != 1 || end - begin != 8 ||
            !read_u16(bytes, begin, end, value) || !read_u16(bytes, begin, end, capability.minimum_version) ||
            !read_u32(bytes, begin, end, capability.flags) || begin != end ||
            value < static_cast<uint16_t>(Capability::ScalarFp32) || value > static_cast<uint16_t>(Capability::TransactionalState)) {
            return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        }
        capability.capability = static_cast<Capability>(value);
        rule.capabilities.push_back(capability);
    }
    for (uint32_t index = 0; index != counts[9]; ++index) {
        uint16_t tag = 0; size_t begin = 0, end = 0;
        SemanticFallback fallback;
        uint16_t kind = 0, phase = 0, numerical = 0;
        if (!read_frame(bytes, offset, body_end, tag, begin, end) || tag != 1 || end - begin != 8 ||
            !read_u16(bytes, begin, end, kind) || !read_u16(bytes, begin, end, phase) ||
            !read_u16(bytes, begin, end, numerical) || !read_u16(bytes, begin, end, fallback.flags) || begin != end ||
            kind > static_cast<uint16_t>(FallbackKind::ExactCpu) || phase < static_cast<uint16_t>(ExecutionPhase::Prefill) ||
            phase > static_cast<uint16_t>(ExecutionPhase::Output) || numerical != static_cast<uint16_t>(NumericalClass::ExactFp32)) {
            return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
        }
        fallback.kind = static_cast<FallbackKind>(kind);
        fallback.phase = static_cast<ExecutionPhase>(phase);
        fallback.numerical_class = static_cast<NumericalClass>(numerical);
        rule.fallbacks.push_back(fallback);
    }
    uint16_t semantic_tag = 0; size_t semantic_begin = 0, semantic_end = 0;
    if (!read_frame(bytes, offset, body_end, semantic_tag, semantic_begin, semantic_end) || semantic_tag != 7) {
        return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
    }
    std::vector<uint8_t> semantic_bytes(bytes.begin() + static_cast<ptrdiff_t>(semantic_begin),
                                        bytes.begin() + static_cast<ptrdiff_t>(semantic_end));
    auto decoded_semantics = decode_semantic_model(semantic_bytes);
    if (!std::holds_alternative<SemanticModel>(decoded_semantics)) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
    rule.semantic_template = std::get<SemanticModel>(std::move(decoded_semantics));
    if (rule.semantic_template.values.size() != counts[2] || rule.semantic_template.layers.size() != counts[4] ||
        rule.semantic_template.input_values_count != counts[5] || rule.semantic_template.output_values_count != counts[6] ||
        rule.semantic_template.states.size() != counts[7] || rule.semantic_template.operators.size() != max_ops) {
        return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
    }
    if (body_end - offset != 32) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
    std::memcpy(rule.semantic_template_digest.bytes.data(), bytes.data() + offset, 32);
    offset += 32;
    auto reencoded = encode_compatibility_rule(rule);
    if (!std::holds_alternative<std::vector<uint8_t>>(reencoded) || std::get<std::vector<uint8_t>>(reencoded) != bytes) return rule_error(CompatibilityError::RULE_VERSION_UNSUPPORTED);
    return rule;
}

Sha256Digest rule_fingerprint(const CompatibilityRule& rule) {
    Sha256Digest output;
    auto encoded = encode_compatibility_rule(rule);
    if (!std::holds_alternative<std::vector<uint8_t>>(encoded)) return output;
    const auto& bytes = std::get<std::vector<uint8_t>>(encoded);
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    constexpr char domain[] = "laplace-rule-fingerprint-v1";
    CC_SHA256_Update(&context, domain, sizeof(domain));
    CC_SHA256_Update(&context, bytes.data(), static_cast<CC_LONG>(bytes.size()));
    CC_SHA256_Final(output.bytes.data(), &context);
    return output;
}

RuleEvaluationResult evaluate_rules(const std::vector<CompatibilityRule>& rules, const PackageEvidence& package) {
    std::vector<SemanticModel> matches;
    bool unmapped = false;
    for (const CompatibilityRule& rule : rules) {
        SemanticModel model;
        MatchResult result = match_rule(rule, package, model);
        if (result == MatchResult::Unmapped) unmapped = true;
        if (result == MatchResult::Match) matches.push_back(std::move(model));
    }
    if (matches.size() == 1) return matches.front();
    if (matches.size() > 1) return import_error(CompatibilityError::IMPORT_RULE_CONFLICT);
    if (unmapped) return import_error(CompatibilityError::IMPORT_TENSOR_UNMAPPED);
    return import_error(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                        "no bundled LAPRUL10 rule matches package metadata and tensor contracts");
}

} // namespace Laplace
