#include "mlx_package.h"

#include <CommonCrypto/CommonDigest.h>

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact_set.h"
#include "safetensors.h"
#include "safetensors_adapter.h"

namespace Laplace {
namespace {

constexpr size_t k_max_json_bytes = 64u * 1024u * 1024u;
constexpr size_t k_max_json_depth = 64;
constexpr size_t k_max_json_members = 1u << 20;

enum class JsonKind : uint8_t { Null, Bool, Number, String, Array, Object };

struct JsonValue {
    JsonKind kind = JsonKind::Null;
    bool boolean = false;
    std::string string;
    std::vector<JsonValue> array;
    std::map<std::string, JsonValue> object;
};

class JsonParser {
public:
    explicit JsonParser(std::span<const uint8_t> bytes) : bytes_(bytes) {}

    std::optional<JsonValue> parse() {
        skip_space();
        auto value = parse_value(0);
        skip_space();
        if (!value || position_ != bytes_.size()) return std::nullopt;
        return value;
    }

private:
    bool at_end() const { return position_ == bytes_.size(); }

    void skip_space() {
        while (!at_end() && (bytes_[position_] == ' ' || bytes_[position_] == '\n' ||
                             bytes_[position_] == '\r' || bytes_[position_] == '\t')) {
            ++position_;
        }
    }

    bool consume(std::string_view literal) {
        if (literal.size() > bytes_.size() - position_ ||
            std::memcmp(bytes_.data() + position_, literal.data(), literal.size()) != 0) {
            return false;
        }
        position_ += literal.size();
        return true;
    }

    static bool hex_value(uint8_t byte, uint32_t& value) {
        if (byte >= '0' && byte <= '9') value = byte - '0';
        else if (byte >= 'a' && byte <= 'f') value = byte - 'a' + 10;
        else if (byte >= 'A' && byte <= 'F') value = byte - 'A' + 10;
        else return false;
        return true;
    }

    bool unicode_escape(uint32_t& codepoint) {
        if (bytes_.size() - position_ < 4) return false;
        codepoint = 0;
        for (size_t index = 0; index != 4; ++index) {
            uint32_t value = 0;
            if (!hex_value(bytes_[position_ + index], value)) return false;
            codepoint = (codepoint << 4) | value;
        }
        position_ += 4;
        return true;
    }

    static void append_utf8(std::string& text, uint32_t codepoint) {
        if (codepoint <= 0x7f) text.push_back(static_cast<char>(codepoint));
        else if (codepoint <= 0x7ff) {
            text.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            text.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else if (codepoint <= 0xffff) {
            text.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            text.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            text.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else {
            text.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
            text.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
            text.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            text.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
    }

    bool copy_utf8(std::string& text) {
        const uint8_t lead = bytes_[position_];
        size_t continuation_count = 0;
        uint32_t codepoint = 0;
        uint32_t lower_bound = 0;
        if (lead >= 0xc2 && lead <= 0xdf) {
            continuation_count = 1;
            codepoint = lead & 0x1f;
            lower_bound = 0x80;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            continuation_count = 2;
            codepoint = lead & 0x0f;
            lower_bound = 0x800;
        } else if (lead >= 0xf0 && lead <= 0xf4) {
            continuation_count = 3;
            codepoint = lead & 0x07;
            lower_bound = 0x10000;
        } else {
            return false;
        }
        if (bytes_.size() - position_ <= continuation_count) return false;
        for (size_t index = 1; index <= continuation_count; ++index) {
            const uint8_t byte = bytes_[position_ + index];
            if ((byte & 0xc0) != 0x80) return false;
            codepoint = (codepoint << 6) | (byte & 0x3f);
        }
        if (codepoint < lower_bound || codepoint > 0x10ffff ||
            (codepoint >= 0xd800 && codepoint <= 0xdfff)) return false;
        text.append(reinterpret_cast<const char*>(bytes_.data() + position_), continuation_count + 1);
        position_ += continuation_count + 1;
        return true;
    }

    std::optional<std::string> parse_string() {
        if (at_end() || bytes_[position_++] != '"') return std::nullopt;
        std::string text;
        while (!at_end()) {
            const uint8_t byte = bytes_[position_++];
            if (byte == '"') return text;
            if (byte < 0x20) return std::nullopt;
            if (byte == '\\') {
                if (at_end()) return std::nullopt;
                switch (bytes_[position_++]) {
                case '"': text.push_back('"'); break;
                case '\\': text.push_back('\\'); break;
                case '/': text.push_back('/'); break;
                case 'b': text.push_back('\b'); break;
                case 'f': text.push_back('\f'); break;
                case 'n': text.push_back('\n'); break;
                case 'r': text.push_back('\r'); break;
                case 't': text.push_back('\t'); break;
                case 'u': {
                    uint32_t codepoint = 0;
                    if (!unicode_escape(codepoint)) return std::nullopt;
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                        if (bytes_.size() - position_ < 2 || bytes_[position_] != '\\' ||
                            bytes_[position_ + 1] != 'u') return std::nullopt;
                        position_ += 2;
                        uint32_t low = 0;
                        if (!unicode_escape(low) || low < 0xdc00 || low > 0xdfff) return std::nullopt;
                        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                    } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                        return std::nullopt;
                    }
                    append_utf8(text, codepoint);
                    break;
                }
                default: return std::nullopt;
                }
            } else if (byte < 0x80) {
                text.push_back(static_cast<char>(byte));
            } else {
                --position_;
                if (!copy_utf8(text)) return std::nullopt;
            }
        }
        return std::nullopt;
    }

    bool parse_number() {
        const size_t begin = position_;
        if (!at_end() && bytes_[position_] == '-') ++position_;
        if (at_end()) return false;
        if (bytes_[position_] == '0') ++position_;
        else if (bytes_[position_] >= '1' && bytes_[position_] <= '9') {
            do { ++position_; } while (!at_end() && bytes_[position_] >= '0' && bytes_[position_] <= '9');
        } else {
            return false;
        }
        if (!at_end() && bytes_[position_] == '.') {
            ++position_;
            const size_t fraction = position_;
            while (!at_end() && bytes_[position_] >= '0' && bytes_[position_] <= '9') ++position_;
            if (position_ == fraction) return false;
        }
        if (!at_end() && (bytes_[position_] == 'e' || bytes_[position_] == 'E')) {
            ++position_;
            if (!at_end() && (bytes_[position_] == '+' || bytes_[position_] == '-')) ++position_;
            const size_t exponent = position_;
            while (!at_end() && bytes_[position_] >= '0' && bytes_[position_] <= '9') ++position_;
            if (position_ == exponent) return false;
        }
        return position_ != begin;
    }

    std::optional<JsonValue> parse_array(size_t depth) {
        JsonValue value;
        value.kind = JsonKind::Array;
        ++position_;
        skip_space();
        if (!at_end() && bytes_[position_] == ']') {
            ++position_;
            return value;
        }
        while (true) {
            if (value.array.size() >= k_max_json_members) return std::nullopt;
            auto element = parse_value(depth + 1);
            if (!element) return std::nullopt;
            value.array.push_back(std::move(*element));
            skip_space();
            if (at_end()) return std::nullopt;
            if (bytes_[position_] == ']') {
                ++position_;
                return value;
            }
            if (bytes_[position_++] != ',') return std::nullopt;
            skip_space();
        }
    }

    std::optional<JsonValue> parse_object(size_t depth) {
        JsonValue value;
        value.kind = JsonKind::Object;
        ++position_;
        skip_space();
        if (!at_end() && bytes_[position_] == '}') {
            ++position_;
            return value;
        }
        while (true) {
            if (value.object.size() >= k_max_json_members) return std::nullopt;
            auto key = parse_string();
            if (!key) return std::nullopt;
            skip_space();
            if (at_end() || bytes_[position_++] != ':') return std::nullopt;
            skip_space();
            auto member = parse_value(depth + 1);
            if (!member || !value.object.emplace(std::move(*key), std::move(*member)).second) {
                return std::nullopt;
            }
            skip_space();
            if (at_end()) return std::nullopt;
            if (bytes_[position_] == '}') {
                ++position_;
                return value;
            }
            if (bytes_[position_++] != ',') return std::nullopt;
            skip_space();
        }
    }

    std::optional<JsonValue> parse_value(size_t depth) {
        if (depth > k_max_json_depth || at_end()) return std::nullopt;
        if (bytes_[position_] == '{') return parse_object(depth);
        if (bytes_[position_] == '[') return parse_array(depth);
        if (bytes_[position_] == '"') {
            auto string = parse_string();
            if (!string) return std::nullopt;
            JsonValue value;
            value.kind = JsonKind::String;
            value.string = std::move(*string);
            return value;
        }
        if (consume("true")) return JsonValue{JsonKind::Bool, true};
        if (consume("false")) return JsonValue{JsonKind::Bool, false};
        if (consume("null")) return JsonValue{JsonKind::Null};
        if (parse_number()) return JsonValue{JsonKind::Number};
        return std::nullopt;
    }

    std::span<const uint8_t> bytes_;
    size_t position_ = 0;
};

CompatibilityReport mlx_failure(CompatibilityError code, std::string detail,
                                ArtifactId id = {}) {
    CompatibilityReport report = compatibility_report(code, std::move(detail));
    report.artifact_id = id;
    report.artifact_index = id.value;
    return report;
}

std::optional<JsonValue> parse_json_object(const PackageView& view,
                                           CompatibilityReport& failure) {
    if (view.bytes().size() > k_max_json_bytes) {
        failure = mlx_failure(CompatibilityError::RULE_LIMIT_EXCEEDED,
                              "MLX JSON sidecar exceeds the bounded package parser", view.artifact_id());
        return std::nullopt;
    }
    JsonParser parser(view.bytes());
    auto root = parser.parse();
    if (!root || root->kind != JsonKind::Object) {
        failure = mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                              "MLX package sidecar is not one strict JSON object", view.artifact_id());
        return std::nullopt;
    }
    return root;
}

bool safe_leaf(std::string_view leaf) {
    if (leaf.empty() || leaf == "." || leaf == ".." || leaf.find('/') != std::string_view::npos ||
        leaf.find('\\') != std::string_view::npos || leaf.find('\0') != std::string_view::npos) return false;
    return std::all_of(leaf.begin(), leaf.end(), [](unsigned char byte) { return byte >= 0x20; });
}

bool has_suffix(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

bool checked_add(uint64_t left, uint64_t right, uint64_t& result) {
    if (left > std::numeric_limits<uint64_t>::max() - right) return false;
    result = left + right;
    return true;
}

bool checked_multiply(uint64_t left, uint64_t right, uint64_t& result) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) return false;
    result = left * right;
    return true;
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8) {
        bytes.push_back(static_cast<uint8_t>(value >> shift));
    }
}

Sha256Digest storage_resources_digest(
    const ArtifactIndex& physical,
    std::span<const SafeTensorsStorageResource> resources) {
    std::vector<uint8_t> bytes;
    static constexpr std::array<uint8_t, 8> domain = {
        'L', 'A', 'P', 'S', 'T', 'R', '1', '0'};
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    bytes.insert(bytes.end(), physical.digest().bytes.begin(),
                 physical.digest().bytes.end());
    append_u32(bytes, static_cast<uint32_t>(resources.size()));
    for (const SafeTensorsStorageResource& resource : resources) {
        append_u32(bytes, resource.id);
        append_u32(bytes, static_cast<uint32_t>(resource.tensor_ids.size()));
        for (uint32_t tensor_id : resource.tensor_ids) {
            append_u32(bytes, tensor_id);
        }
    }
    Sha256Digest digest;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()),
              digest.bytes.data());
    return digest;
}

bool digits_to_uint(std::string_view digits, uint32_t& value) {
    if (digits.empty()) return false;
    uint64_t result = 0;
    for (unsigned char byte : digits) {
        if (byte < '0' || byte > '9') return false;
        result = result * 10 + (byte - '0');
        if (result > std::numeric_limits<uint32_t>::max()) return false;
    }
    value = static_cast<uint32_t>(result);
    return true;
}

bool canonical_shard_names(const std::set<std::string>& leaves) {
    if (leaves.size() == 1) return *leaves.begin() == "model.safetensors";
    const uint32_t expected_total = static_cast<uint32_t>(leaves.size());
    std::set<uint32_t> indices;
    for (const std::string& leaf : leaves) {
        constexpr std::string_view prefix = "model-";
        constexpr std::string_view middle = "-of-";
        constexpr std::string_view suffix = ".safetensors";
        if (leaf.size() != prefix.size() + 5 + middle.size() + 5 + suffix.size() ||
            !leaf.starts_with(prefix) || !has_suffix(leaf, suffix)) return false;
        const size_t first_digits = prefix.size();
        const size_t middle_offset = first_digits + 5;
        const size_t last_digits = middle_offset + middle.size();
        if (leaf.substr(middle_offset, middle.size()) != middle) return false;
        uint32_t index = 0;
        uint32_t total = 0;
        if (!digits_to_uint({leaf.data() + first_digits, 5}, index) ||
            !digits_to_uint({leaf.data() + last_digits, 5}, total) || index == 0 ||
            total != expected_total || index > total || !indices.insert(index).second) return false;
    }
    return indices.size() == expected_total && *indices.begin() == 1 && *indices.rbegin() == expected_total;
}

std::optional<std::set<std::string>> index_shards(const JsonValue& index,
                                                  CompatibilityReport& failure,
                                                  ArtifactId id) {
    const auto map = index.object.find("weight_map");
    if (map == index.object.end() || map->second.kind != JsonKind::Object || map->second.object.empty()) {
        failure = mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                              "MLX package index has no nonempty string weight_map", id);
        return std::nullopt;
    }
    std::set<std::string> leaves;
    for (const auto& [tensor_name, shard] : map->second.object) {
        if (tensor_name.empty() || shard.kind != JsonKind::String || !safe_leaf(shard.string) ||
            !has_suffix(shard.string, ".safetensors")) {
            failure = mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                                  "MLX weight_map has an unsafe or non-SafeTensors leaf", id);
            return std::nullopt;
        }
        leaves.insert(shard.string);
    }
    if (!canonical_shard_names(leaves)) {
        failure = mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                              "MLX weight_map does not name the canonical closed shard set", id);
        return std::nullopt;
    }
    return leaves;
}

bool config_requires_code(const JsonValue& config) {
    const auto model_file = config.object.find("model_file");
    if (model_file != config.object.end() && model_file->second.kind != JsonKind::Null) return true;
    const auto remote = config.object.find("trust_remote_code");
    return remote != config.object.end() && remote->second.kind == JsonKind::Bool && remote->second.boolean;
}

std::optional<std::set<std::string>> safetensors_leaves(const std::string& directory,
                                                        CompatibilityReport& failure) {
    DIR* raw = opendir(directory.c_str());
    if (!raw) {
        failure = mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                              "MLX package directory cannot be enumerated");
        return std::nullopt;
    }
    std::set<std::string> leaves;
    while (const dirent* entry = readdir(raw)) {
        const std::string_view name(entry->d_name);
        if (name == "." || name == ".." || !has_suffix(name, ".safetensors")) continue;
        if (!safe_leaf(name)) {
            closedir(raw);
            failure = mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                                  "MLX package directory contains an unsafe SafeTensors leaf");
            return std::nullopt;
        }
        leaves.emplace(name);
    }
    closedir(raw);
    return leaves;
}

std::variant<ArtifactIndex, CompatibilityReport>
combine_shard_indexes(const std::vector<PackageView>& artifacts,
                      const std::vector<PackageView>& shards) {
    ArtifactIndexInput input;
    input.artifacts = artifacts;
    uint32_t tensor_base = 0;
    std::set<std::string> seen_names;
    for (const PackageView& shard : shards) {
        SafeTensorsParseResult parsed = parse_safetensors(shard.bytes());
        if (const auto* error = std::get_if<SafeTensorsParseError>(&parsed)) {
            CompatibilityReport report = error->report;
            report.artifact_id = shard.artifact_id();
            report.artifact_index = shard.artifact_id().value;
            return report;
        }
        const SafeTensorsFile& file = std::get<SafeTensorsFile>(parsed);
        // The public single-file adapter requires a Primary artifact. Package
        // shards are intentionally not primary, so lower their verified wire
        // records here without changing that one-file contract.
        std::vector<const SafeTensorsTensor*> ordered;
        ordered.reserve(file.tensors().size());
        for (const SafeTensorsTensor& tensor : file.tensors()) ordered.push_back(&tensor);
        std::sort(ordered.begin(), ordered.end(), [](const auto* left, const auto* right) {
            return std::tie(left->data_offset, left->data_length, left->dtype, left->shape) <
                   std::tie(right->data_offset, right->data_length, right->dtype, right->shape);
        });
        if (ordered.size() > UINT32_MAX - tensor_base) {
            return mlx_failure(CompatibilityError::RULE_LIMIT_EXCEEDED,
                               "MLX package exceeds the tensor-ID space", shard.artifact_id());
        }
        uint64_t data_base = 0;
        if (!checked_add(8, file.header_length(), data_base)) {
            return mlx_failure(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                               "MLX SafeTensors data base overflows uint64", shard.artifact_id());
        }
        for (size_t local_id = 0; local_id != ordered.size(); ++local_id) {
            const SafeTensorsTensor& source = *ordered[local_id];
            const uint32_t tensor_id = tensor_base + static_cast<uint32_t>(local_id);
            if (local_id != 0) {
                const SafeTensorsTensor& previous = *ordered[local_id - 1];
                if (source.data_offset == previous.data_offset && source.data_length == previous.data_length &&
                    source.dtype == previous.dtype && source.shape == previous.shape) {
                    return mlx_failure(CompatibilityError::IMPORT_TENSOR_DUPLICATE,
                                       "MLX shard contains physically indistinguishable tensor records",
                                       shard.artifact_id());
                }
            }
            const auto scalar = safetensors_physical_scalar(source.dtype);
            if (!scalar) {
                return mlx_failure(CompatibilityError::IR_QUANTIZATION_UNSUPPORTED,
                                   "MLX SafeTensors dtype has no complete physical plane contract",
                                   shard.artifact_id());
            }
            if (source.shape.size() > 8) {
                return mlx_failure(CompatibilityError::IR_SHAPE_MISMATCH,
                                   "MLX SafeTensors rank exceeds the physical index ABI",
                                   shard.artifact_id());
            }
            ArtifactTensorRecord tensor;
            tensor.id = tensor_id;
            tensor.logical_type = scalar->first;
            if (scalar->first == ArtifactScalarType::F16 ||
                scalar->first == ArtifactScalarType::F32) {
                tensor.format.version = 1;
                tensor.format.encoding = scalar->first == ArtifactScalarType::F16
                    ? ArtifactPhysicalEncoding::F16
                    : ArtifactPhysicalEncoding::F32;
                tensor.format.value_type = scalar->first;
                tensor.format.block_elements = 1;
                tensor.format.block_bytes = scalar->second;
            }
            // Preserve the source tensor coordinate system. SafeTensors is C
            // row-major: the last declared axis is contiguous. Semantic axis
            // meaning and any transposition belong to the carried program,
            // never to this role-free physical inventory.
            tensor.logical_dimensions = source.shape;
            tensor.layout.kind = PhysicalLayoutKind::ContiguousRowMajor;
            tensor.layout.version = 1;
            tensor.layout.packing = PackingKind::None;
            tensor.layout.rank = static_cast<uint8_t>(source.shape.size());
            // MLX SafeTensors uses C row-major source order. The physical
            // format is explicit; semantic axis meaning is manifest data.
            tensor.axis.source_rank = static_cast<uint8_t>(source.shape.size());
            uint64_t elements = 1;
            for (size_t axis = 0; axis != tensor.logical_dimensions.size(); ++axis) {
                tensor.layout.axis_order[axis] = static_cast<uint8_t>(axis);
                tensor.axis.source_axis_order[axis] = static_cast<uint8_t>(axis);
            }
            for (size_t reverse = tensor.logical_dimensions.size(); reverse != 0; --reverse) {
                const size_t axis = reverse - 1;
                tensor.layout.strides[axis] = elements;
                if (!checked_multiply(elements, tensor.logical_dimensions[axis], elements)) {
                    return mlx_failure(CompatibilityError::IR_SHAPE_MISMATCH,
                                       "MLX SafeTensors row-major stride overflows uint64",
                                       shard.artifact_id());
                }
            }
            uint64_t row_stride_bytes = scalar->second;
            if (!tensor.logical_dimensions.empty() &&
                !checked_multiply(tensor.logical_dimensions.back(), scalar->second,
                                  row_stride_bytes)) {
                return mlx_failure(CompatibilityError::IR_SHAPE_MISMATCH,
                                   "MLX SafeTensors row stride overflows uint64",
                                   shard.artifact_id());
            }
            tensor.axis.row_stride_bytes = row_stride_bytes;
            tensor.quantization.kind = QuantizationKind::None;
            tensor.quantization.version = 1;
            // This physical slice has one Values plane and no quantization
            // auxiliary planes.
            tensor.quantization.required_plane_mask = 0;
            uint64_t source_offset = 0;
            if (!checked_add(data_base, source.data_offset, source_offset)) {
                return mlx_failure(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                                   "MLX SafeTensors tensor offset overflows uint64", shard.artifact_id());
            }
            tensor.planes.push_back({PlaneKind::Values, scalar->first,
                                     {shard.artifact_id(), source_offset, source.data_length},
                                     elements, scalar->second, 1, 1});
            input.tensors.push_back(std::move(tensor));
            input.diagnostics.push_back({shard.artifact_id(), {}, tensor_id, {}, source.name});
        }
        tensor_base += static_cast<uint32_t>(ordered.size());
    }
    return ArtifactIndex::build(std::move(input));
}

// The index map names a shard with strings while ArtifactId is intentionally
// opaque. Verify that closure before the canonical index is constructed.
std::optional<CompatibilityReport> verify_weight_map(
    const std::vector<std::pair<std::string, PackageView>>& named_shards,
    const std::map<std::string, std::string>& weight_map) {
    std::set<std::string> seen;
    for (const auto& [leaf, shard] : named_shards) {
        SafeTensorsParseResult parsed = parse_safetensors(shard.bytes());
        if (const auto* error = std::get_if<SafeTensorsParseError>(&parsed)) {
            CompatibilityReport report = error->report;
            report.artifact_id = shard.artifact_id();
            report.artifact_index = shard.artifact_id().value;
            return report;
        }
        const SafeTensorsFile& file = std::get<SafeTensorsFile>(parsed);
        for (const SafeTensorsTensor& tensor : file.tensors()) {
            const auto expected = weight_map.find(tensor.name);
            if (expected == weight_map.end() || expected->second != leaf || !seen.insert(tensor.name).second) {
                return mlx_failure(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                                   "MLX index and shard tensors are not a closed one-to-one mapping",
                                   shard.artifact_id());
            }
        }
    }
    if (seen.size() != weight_map.size()) {
        return mlx_failure(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                           "MLX index names a tensor absent from its declared shard");
    }
    return std::nullopt;
}

MlxPhysicalPackageResult load_single_safetensors(std::string_view path) {
    auto loaded = ArtifactSet::load_single_file(path);
    if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) return *report;
    auto view = std::get<ArtifactSet>(loaded).view(ArtifactId{0});
    if (const auto* report = std::get_if<CompatibilityReport>(&view)) return *report;
    auto index = build_safetensors_artifact_index(std::get<PackageView>(view));
    if (const auto* report = std::get_if<CompatibilityReport>(&index)) return *report;
    return MlxPhysicalPackage(std::get<ArtifactIndex>(std::move(index)));
}

MlxPhysicalPackageResult load_directory(std::string_view root) {
    const std::string directory(root);
    const std::string config_path = directory + "/config.json";
    const std::string index_path = directory + "/model.safetensors.index.json";
    const std::array<ArtifactSource, 2> temporary_sources = {{
        {config_path, ArtifactRole::Primary, ArtifactId{0}},
        {index_path, ArtifactRole::Sidecar, ArtifactId{1}},
    }};
    auto temporary = ArtifactSet::load_graph(temporary_sources);
    if (const auto* report = std::get_if<CompatibilityReport>(&temporary)) return *report;
    const ArtifactSet& temporary_set = std::get<ArtifactSet>(temporary);
    auto config_view = temporary_set.view(ArtifactId{0});
    auto index_view = temporary_set.view(ArtifactId{1});
    if (const auto* report = std::get_if<CompatibilityReport>(&config_view)) return *report;
    if (const auto* report = std::get_if<CompatibilityReport>(&index_view)) return *report;
    const PackageView config = std::get<PackageView>(config_view);
    const PackageView index = std::get<PackageView>(index_view);

    CompatibilityReport failure;
    auto config_json = parse_json_object(config, failure);
    if (!config_json) return failure;
    if (config_requires_code(*config_json)) {
        return mlx_failure(CompatibilityError::IMPORT_EXECUTABLE_CODE_REQUIRED,
                           "MLX config requests executable model code", config.artifact_id());
    }
    auto index_json = parse_json_object(index, failure);
    if (!index_json) return failure;
    auto leaves = index_shards(*index_json, failure, index.artifact_id());
    if (!leaves) return failure;
    auto observed_leaves = safetensors_leaves(directory, failure);
    if (!observed_leaves) return failure;
    if (*leaves != *observed_leaves) {
        return mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                           "MLX package SafeTensors leaves do not exactly match weight_map", index.artifact_id());
    }

    const auto map_member = index_json->object.find("weight_map");
    std::map<std::string, std::string> weight_map;
    for (const auto& [tensor_name, shard] : map_member->second.object) {
        weight_map.emplace(tensor_name, shard.string);
    }

    std::vector<std::string> paths;
    paths.reserve(leaves->size() + 2);
    paths.push_back(config_path);
    paths.push_back(index_path);
    for (const std::string& leaf : *leaves) paths.push_back(directory + "/" + leaf);
    std::vector<ArtifactSource> sources;
    sources.reserve(paths.size());
    sources.push_back({paths[0], ArtifactRole::Primary, ArtifactId{0}});
    sources.push_back({paths[1], ArtifactRole::Sidecar, ArtifactId{1}});
    for (size_t index_id = 2; index_id != paths.size(); ++index_id) {
        sources.push_back({paths[index_id], ArtifactRole::Shard,
                           ArtifactId{static_cast<uint32_t>(index_id)}});
    }
    auto final_set_result = ArtifactSet::load_graph(sources);
    if (const auto* report = std::get_if<CompatibilityReport>(&final_set_result)) return *report;
    const ArtifactSet& final_set = std::get<ArtifactSet>(final_set_result);
    auto final_config = final_set.view(ArtifactId{0});
    auto final_index = final_set.view(ArtifactId{1});
    if (const auto* report = std::get_if<CompatibilityReport>(&final_config)) return *report;
    if (const auto* report = std::get_if<CompatibilityReport>(&final_index)) return *report;
    if (std::get<PackageView>(final_config).digest() != config.digest() ||
        std::get<PackageView>(final_index).digest() != index.digest()) {
        return mlx_failure(CompatibilityError::PACKAGE_SOURCE_CHANGED,
                           "MLX config or index changed while closing the package graph");
    }

    std::vector<PackageView> artifacts;
    artifacts.reserve(paths.size());
    artifacts.push_back(std::get<PackageView>(final_config));
    artifacts.push_back(std::get<PackageView>(final_index));
    std::vector<PackageView> shards;
    std::vector<std::pair<std::string, PackageView>> named_shards;
    shards.reserve(leaves->size());
    named_shards.reserve(leaves->size());
    size_t source_index = 2;
    for (const std::string& leaf : *leaves) {
        auto shard = final_set.view(ArtifactId{static_cast<uint32_t>(source_index++)});
        if (const auto* report = std::get_if<CompatibilityReport>(&shard)) return *report;
        PackageView view = std::get<PackageView>(shard);
        artifacts.push_back(view);
        named_shards.emplace_back(leaf, view);
        shards.push_back(std::move(view));
    }
    if (auto map_error = verify_weight_map(named_shards, weight_map)) return *map_error;
    auto combined = combine_shard_indexes(artifacts, shards);
    if (const auto* report = std::get_if<CompatibilityReport>(&combined)) return *report;
    ArtifactIndex physical = std::get<ArtifactIndex>(std::move(combined));

    const std::string storage_path = directory + "/quantization_config.json";
    struct stat storage_status {};
    if (lstat(storage_path.c_str(), &storage_status) == 0) {
        if (!S_ISREG(storage_status.st_mode) || S_ISLNK(storage_status.st_mode) ||
            storage_status.st_size <= 0) {
            return mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                               "SafeTensors storage declaration is not a direct nonempty file");
        }
        auto declaration_set = ArtifactSet::load_single_file(storage_path);
        if (const auto* report =
                std::get_if<CompatibilityReport>(&declaration_set)) {
            return *report;
        }
        auto declaration = std::get<ArtifactSet>(declaration_set).view(
            ArtifactId{0});
        if (const auto* report =
                std::get_if<CompatibilityReport>(&declaration)) {
            return *report;
        }
        auto storage = compile_safetensors_storage_resources(
            physical, std::get<PackageView>(declaration));
        if (const auto* report = std::get_if<CompatibilityReport>(&storage)) {
            return *report;
        }
        return MlxPhysicalPackage(
            std::move(physical),
            std::get<VerifiedSafeTensorsStorageResources>(std::move(storage)));
    }
    if (errno != ENOENT) {
        return mlx_failure(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                           "SafeTensors storage declaration presence could not be determined");
    }
    return MlxPhysicalPackage(std::move(physical));
}

MlxProductPhysicalPackageResult load_product_directory(std::string_view root) {
    const std::string directory(root);
    const std::string manifest_path = directory + "/laplace.lapman";
    const std::string token_path = directory + "/laplace.laptok";
    const std::string physical_package_path = directory + "/laplace.lappkg";
    struct stat manifest_status {};
    if (lstat(manifest_path.c_str(), &manifest_status) != 0 ||
        !S_ISREG(manifest_status.st_mode) || manifest_status.st_size <= 0) {
        return mlx_failure(CompatibilityError::PACKAGE_AUTHORITY_REQUIRED,
                           "MLX product closure lacks a regular nonempty laplace.lapman",
                           kMlxProductManifestArtifactId);
    }
    struct stat token_status {};
    if (lstat(token_path.c_str(), &token_status) != 0 ||
        !S_ISREG(token_status.st_mode) || token_status.st_size <= 0) {
        return mlx_failure(CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED,
                           "MLX product closure lacks a regular nonempty laplace.laptok",
                           kMlxProductTokenArtifactId);
    }
    bool has_physical_package = false;
    struct stat physical_package_status {};
    if (lstat(physical_package_path.c_str(), &physical_package_status) == 0) {
        if (!S_ISREG(physical_package_status.st_mode) ||
            physical_package_status.st_size <= 0) {
            return mlx_failure(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                               "SafeTensors product physical-program carrier is not a direct nonempty file",
                               kSafeTensorsProductPhysicalPackageArtifactId);
        }
        has_physical_package = true;
    } else if (errno != ENOENT) {
        return mlx_failure(CompatibilityError::PACKAGE_BOUNDS_INVALID,
                           "SafeTensors product physical-program carrier presence could not be determined",
                           kSafeTensorsProductPhysicalPackageArtifactId);
    }
    auto physical_result = load_directory(root);
    if (const auto* report = std::get_if<CompatibilityReport>(&physical_result)) return *report;
    const ArtifactIndex& discovered = std::get<MlxPhysicalPackage>(physical_result).physical_index();

    const auto primary = std::find_if(discovered.artifacts().begin(), discovered.artifacts().end(),
                                      [](const PackageView& artifact) {
                                          return artifact.role() == ArtifactRole::Primary;
                                      });
    const auto index = std::find_if(discovered.artifacts().begin(), discovered.artifacts().end(),
                                    [](const PackageView& artifact) {
                                        return artifact.role() == ArtifactRole::Sidecar;
                                    });
    if (primary == discovered.artifacts().end() || index == discovered.artifacts().end()) {
        return mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                           "MLX product closure lacks its declared config and index");
    }

    std::set<std::string> leaves;
    CompatibilityReport failure;
    auto discovered_leaves = safetensors_leaves(directory, failure);
    if (!discovered_leaves) return failure;
    leaves = std::move(*discovered_leaves);
    const std::string config_path = directory + "/config.json";
    const std::string index_path = directory + "/model.safetensors.index.json";

    std::vector<ArtifactSource> sources;
    sources.reserve(leaves.size() + 5);
    sources.push_back({config_path, ArtifactRole::Primary, ArtifactId{0}});
    sources.push_back({index_path, ArtifactRole::Sidecar, ArtifactId{1}});
    std::vector<std::string> shard_paths;
    shard_paths.reserve(leaves.size());
    uint32_t shard_id = 2;
    for (const std::string& leaf : leaves) {
        shard_paths.push_back(directory + "/" + leaf);
        sources.push_back({shard_paths.back(), ArtifactRole::Shard,
                           ArtifactId{shard_id++}});
    }
    sources.push_back({token_path, ArtifactRole::Shard, kMlxProductTokenArtifactId});
    sources.push_back({manifest_path, ArtifactRole::Sidecar,
                       kMlxProductManifestArtifactId});
    if (has_physical_package) {
        sources.push_back({physical_package_path, ArtifactRole::Sidecar,
                           kSafeTensorsProductPhysicalPackageArtifactId});
    }
    auto closure_result = ArtifactSet::load_graph(sources);
    if (const auto* report = std::get_if<CompatibilityReport>(&closure_result)) {
        if (report->artifact_id == kMlxProductManifestArtifactId) {
            return mlx_failure(CompatibilityError::PACKAGE_AUTHORITY_REQUIRED,
                               "MLX product closure lacks laplace.lapman",
                               kMlxProductManifestArtifactId);
        }
        if (report->artifact_id == kMlxProductTokenArtifactId) {
            return mlx_failure(CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED,
                               "MLX product closure lacks laplace.laptok",
                               kMlxProductTokenArtifactId);
        }
        return *report;
    }
    ArtifactSet closure = std::get<ArtifactSet>(std::move(closure_result));
    auto closure_config = closure.view(ArtifactId{0});
    auto closure_index = closure.view(ArtifactId{1});
    auto closure_manifest = closure.view(kMlxProductManifestArtifactId);
    auto closure_token = closure.view(kMlxProductTokenArtifactId);
    if (const auto* report = std::get_if<CompatibilityReport>(&closure_config)) return *report;
    if (const auto* report = std::get_if<CompatibilityReport>(&closure_index)) return *report;
    if (const auto* report = std::get_if<CompatibilityReport>(&closure_manifest)) return *report;
    if (const auto* report = std::get_if<CompatibilityReport>(&closure_token)) return *report;
    std::optional<PackageView> closure_physical_package;
    if (has_physical_package) {
        auto physical_package = closure.view(
            kSafeTensorsProductPhysicalPackageArtifactId);
        if (const auto* report =
                std::get_if<CompatibilityReport>(&physical_package)) {
            return *report;
        }
        closure_physical_package.emplace(
            std::get<PackageView>(std::move(physical_package)));
    }
    if (std::get<PackageView>(closure_config).digest() != primary->digest() ||
        std::get<PackageView>(closure_index).digest() != index->digest()) {
        return mlx_failure(CompatibilityError::PACKAGE_SOURCE_CHANGED,
                           "MLX source changed while creating its product closure");
    }

    ArtifactIndexInput input;
    input.artifacts.push_back(std::get<PackageView>(closure_config));
    input.artifacts.push_back(std::get<PackageView>(closure_index));
    for (uint32_t id = 2; id != shard_id; ++id) {
        auto shard = closure.view(ArtifactId{id});
        if (const auto* report = std::get_if<CompatibilityReport>(&shard)) return *report;
        input.artifacts.push_back(std::get<PackageView>(std::move(shard)));
    }
    input.artifacts.push_back(std::get<PackageView>(closure_token));
    input.tensors.assign(discovered.tensors().begin(), discovered.tensors().end());
    input.diagnostics.assign(discovered.diagnostics().begin(), discovered.diagnostics().end());
    auto indexed = ArtifactIndex::build(std::move(input));
    if (const auto* report = std::get_if<CompatibilityReport>(&indexed)) return *report;
    return MlxProductPhysicalPackage{
        std::move(closure), std::get<ArtifactIndex>(std::move(indexed)),
        std::get<PackageView>(std::move(closure_manifest)),
        std::move(closure_physical_package)};
}

} // namespace

CompatibilityReport MlxPhysicalPackage::semantic_refusal() const {
    return compatibility_report(CompatibilityError::IMPORT_SEMANTICS_MISSING,
                                "SafeTensors physical package has no carried semantic program or unique graph proof");
}

SafeTensorsStorageResourcesResult compile_safetensors_storage_resources(
    const ArtifactIndex& physical, const PackageView& declaration) {
    CompatibilityReport failure;
    auto root = parse_json_object(declaration, failure);
    if (!root) return failure;
    const auto storage = root->object.find("tensor_storage");
    if (storage == root->object.end() ||
        storage->second.kind != JsonKind::Object ||
        storage->second.object.empty()) {
        return mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                           "storage declaration has no nonempty tensor_storage object",
                           declaration.artifact_id());
    }

    std::map<std::string, uint32_t> tensor_by_spelling;
    for (const ArtifactDiagnostic& diagnostic : physical.diagnostics()) {
        if (diagnostic.tensor_id == UINT32_MAX ||
            diagnostic.tensor_spelling.empty()) {
            continue;
        }
        if (!tensor_by_spelling.emplace(diagnostic.tensor_spelling,
                                        diagnostic.tensor_id).second) {
            return mlx_failure(CompatibilityError::IMPORT_TENSOR_DUPLICATE,
                               "physical index has duplicate tensor spellings",
                               declaration.artifact_id());
        }
    }

    VerifiedSafeTensorsStorageResources verified;
    std::set<uint32_t> consumed;
    try {
        verified.resources_.reserve(physical.tensors().size());
        for (const auto& [source_group, group] : storage->second.object) {
            (void)source_group;
            if (group.kind != JsonKind::Object) {
                return mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                                   "storage resource declaration is not an object",
                                   declaration.artifact_id());
            }
            const auto stored = group.object.find("stored_tensors");
            if (stored == group.object.end() ||
                stored->second.kind != JsonKind::Object ||
                stored->second.object.empty()) {
                return mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                                   "storage resource has no nonempty stored_tensors object",
                                   declaration.artifact_id());
            }
            SafeTensorsStorageResource resource;
            resource.id = static_cast<uint32_t>(verified.resources_.size());
            resource.tensor_ids.reserve(stored->second.object.size());
            for (const auto& [source_tensor, descriptor] :
                 stored->second.object) {
                if (descriptor.kind != JsonKind::Object) {
                    return mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                                       "stored tensor declaration is not an object",
                                       declaration.artifact_id());
                }
                const auto tensor = tensor_by_spelling.find(source_tensor);
                if (tensor == tensor_by_spelling.end()) {
                    return mlx_failure(CompatibilityError::IMPORT_TENSOR_UNMAPPED,
                                       "storage declaration names an absent tensor",
                                       declaration.artifact_id());
                }
                if (!consumed.insert(tensor->second).second) {
                    return mlx_failure(CompatibilityError::IMPORT_TENSOR_DUPLICATE,
                                       "storage declaration consumes one tensor more than once",
                                       declaration.artifact_id());
                }
                resource.tensor_ids.push_back(tensor->second);
            }
            std::sort(resource.tensor_ids.begin(), resource.tensor_ids.end());
            verified.resources_.push_back(std::move(resource));
        }
        verified.declared_resource_count_ =
            static_cast<uint32_t>(verified.resources_.size());

        std::vector<uint32_t> standalone;
        standalone.reserve(physical.tensors().size() - consumed.size());
        for (const ArtifactTensorRecord& tensor : physical.tensors()) {
            if (!consumed.contains(tensor.id)) standalone.push_back(tensor.id);
        }
        std::sort(standalone.begin(), standalone.end());
        for (uint32_t tensor_id : standalone) {
            SafeTensorsStorageResource resource;
            resource.id = static_cast<uint32_t>(verified.resources_.size());
            resource.tensor_ids.push_back(tensor_id);
            verified.resources_.push_back(std::move(resource));
        }
    } catch (const std::bad_alloc&) {
        return mlx_failure(CompatibilityError::RULE_LIMIT_EXCEEDED,
                           "storage resource compilation exhausted memory",
                           declaration.artifact_id());
    }

    if (consumed.size() > physical.tensors().size() ||
        verified.resources_.empty()) {
        return mlx_failure(CompatibilityError::IMPORT_CLOSURE_INCOMPLETE,
                           "storage resources do not cover the physical index",
                           declaration.artifact_id());
    }
    verified.canonical_digest_ =
        storage_resources_digest(physical, verified.resources_);
    verified.declaration_digest_ = declaration.digest();
    return verified;
}

MlxPhysicalPackageResult load_safetensors_physical_package(std::string_view path) {
    if (path.empty()) {
        return mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                           "MLX package path is empty");
    }
    const std::string native_path(path);
    struct stat status {};
    if (lstat(native_path.c_str(), &status) != 0 || S_ISLNK(status.st_mode)) {
        return mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                           "MLX package entrypoint is not a direct regular file or directory");
    }
    if (S_ISREG(status.st_mode)) return load_single_safetensors(path);
    if (S_ISDIR(status.st_mode)) return load_directory(path);
    return mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                       "MLX package entrypoint is not a regular file or directory");
}

MlxProductPhysicalPackageResult
load_safetensors_product_physical_package(std::string_view path) {
    if (path.empty()) {
        return mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                           "MLX product package path is empty");
    }
    const std::string native_path(path);
    struct stat status {};
    if (lstat(native_path.c_str(), &status) != 0 || !S_ISDIR(status.st_mode) ||
        S_ISLNK(status.st_mode)) {
        return mlx_failure(CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED,
                           "MLX product package must be a direct directory");
    }
    return load_product_directory(path);
}

MlxPhysicalPackageResult load_mlx_physical_package(std::string_view path) {
    return load_safetensors_physical_package(path);
}

MlxProductPhysicalPackageResult
load_mlx_product_physical_package(std::string_view path) {
    return load_safetensors_product_physical_package(path);
}

} // namespace Laplace
