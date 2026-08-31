#include "codec_program.h"
#include "physical_codec.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <limits>
#include <optional>
#include <utility>

namespace Laplace {

namespace {

constexpr uint16_t kAbiVersion = 1;
constexpr uint16_t kContractVersion = 1;
constexpr uint32_t kRoundToNearestEven = 1;
constexpr std::array<uint8_t, 8> kContractMagic =
    {0x43, 0x50, 0x52, 0x47, 0x00, 0x01, 0x00, 0x00};

struct Binary16Spec {
    uint32_t sign_bit = 0;
    uint32_t exponent_bits = 0;
    uint32_t significand_bits = 0;
    uint32_t exponent_bias = 0;
    uint32_t target_sign_bit = 0;
    uint32_t target_exponent_bits = 0;
    uint32_t target_significand_bits = 0;
    uint32_t target_exponent_bias = 0;
    uint32_t normal_policy = 0;
    uint32_t subnormal_policy = 0;
    uint32_t special_policy = 0;
    uint32_t rounding_mode = 0;
    uint32_t scaling_policy = 0;
};

struct PackedField {
    uint32_t source = 0;
    uint32_t extract_shift = 0;
    uint32_t mask = 0;
    uint32_t value_shift = 0;
};

struct Q4Group {
    PackedField scale_low;
    PackedField scale_high;
    PackedField minimum_low;
    PackedField minimum_high;
    uint32_t q_offset = 0;
    uint32_t output_offset = 0;
    uint32_t q_shift = 0;
    uint32_t q_mask = 0;
};

struct AddressSpec {
    uint32_t offset_bits = 0;
    uint32_t length_bits = 0;
    uint32_t stride_bits = 0;
    uint32_t minimum_stride = 0;
    uint32_t checked_span_rule = 0;
    uint32_t checked_access_rule = 0;
    uint32_t declared_length_rule = 0;
    uint32_t wrapping_rule = 0;
};

struct PlaneSpec {
    uint32_t count = 0;
    uint32_t storage_domain = 0;
    uint32_t encoded_width_bits = 0;
    uint32_t encoded_unit_elements = 0;
    uint32_t encoded_unit_bytes = 0;
    uint32_t signedness = 0;
    uint32_t byte_order = 0;
};

struct PhysicalPlaneSpec {
    uint16_t kind = 0;
    uint16_t storage_type = 0;
    uint32_t logical_elements_covered = 0;
    uint32_t bytes_per_block = 0;
    uint32_t flags = 0;
};

struct PhysicalSpec {
    uint16_t identity_version = 0;
    uint16_t arithmetic_version = 0;
    std::array<uint8_t, 32> codebook_digest{};
    uint16_t layout_kind = 0;
    uint16_t layout_version = 0;
    uint16_t layout_packing = 0;
    uint8_t layout_rank = 0;
    uint8_t layout_block_rank = 0;
    std::array<uint8_t, 8> layout_axis_order{};
    uint32_t layout_block_elements = 0;
    uint32_t layout_block_bytes = 0;
    uint32_t layout_flags = 0;
    uint16_t quantization_kind = 0;
    uint16_t quantization_version = 0;
    uint16_t quantization_accumulation_type = 0;
    uint16_t quantization_scale_type = 0;
    uint16_t quantization_zero_type = 0;
    uint16_t quantization_bias_type = 0;
    uint32_t quantization_block_elements = 0;
    uint32_t quantization_block_bytes = 0;
    uint32_t quantization_group_size = 0;
    uint32_t quantization_required_plane_mask = 0;
    uint32_t quantization_flags = 0;
    uint32_t plane_count = 0;
    std::array<PhysicalPlaneSpec, 6> planes{};
};

struct Q4Spec {
    Binary16Spec scalar;
    uint32_t d_offset = 0;
    uint32_t d_width = 0;
    uint32_t dmin_offset = 0;
    uint32_t dmin_width = 0;
    uint32_t scales_offset = 0;
    uint32_t scales_width = 0;
    uint32_t q_offset = 0;
    uint32_t q_width = 0;
    uint32_t group_count = 0;
    uint32_t values_per_half = 0;
    uint32_t q_bytes_per_half = 0;
    uint32_t d_encoding = 0;
    uint32_t scale_conversion = 0;
    uint32_t value_formula = 0;
    uint32_t integer_rounding = 0;
    uint32_t special_policy = 0;
    uint32_t fma_policy = 0;
    uint32_t nibble_signedness = 0;
    std::array<Q4Group, 8> groups{};
};

struct ParsedProgram {
    bool q4_mode = false;
    uint16_t abi_version = 0;
    uint32_t unit_elements = 0;
    uint32_t unit_bytes = 0;
    uint32_t minimum_stride = 0;
    uint32_t plane_count = 0;
    uint32_t output_domain = 0;
    uint32_t output_signedness = 0;
    PlaneSpec plane;
    PhysicalSpec physical;
    AddressSpec address;
    Binary16Spec binary16;
    Q4Spec q4;
};

class Reader {
public:
    explicit Reader(std::span<const uint8_t> bytes) : bytes_(bytes) {}

    bool read_u8(uint8_t* value) {
        if (remaining() < 1) return false;
        *value = bytes_[offset_++];
        return true;
    }

    bool read_u16(uint16_t* value) {
        if (remaining() < 2) return false;
        *value = static_cast<uint16_t>(bytes_[offset_]) |
                 (static_cast<uint16_t>(bytes_[offset_ + 1]) << 8u);
        offset_ += 2;
        return true;
    }

    bool read_u32(uint32_t* value) {
        if (remaining() < 4) return false;
        *value = static_cast<uint32_t>(bytes_[offset_]) |
                 (static_cast<uint32_t>(bytes_[offset_ + 1]) << 8u) |
                 (static_cast<uint32_t>(bytes_[offset_ + 2]) << 16u) |
                 (static_cast<uint32_t>(bytes_[offset_ + 3]) << 24u);
        offset_ += 4;
        return true;
    }

    bool read_magic(std::span<const uint8_t> expected) {
        if (remaining() < expected.size()) return false;
        if (!std::equal(expected.begin(), expected.end(), bytes_.begin() + offset_)) return false;
        offset_ += expected.size();
        return true;
    }

    bool read_digest(std::array<uint8_t, 32>* digest) {
        if (remaining() < digest->size()) return false;
        std::copy_n(bytes_.data() + offset_, digest->size(), digest->data());
        offset_ += digest->size();
        return true;
    }

    bool done() const { return offset_ == bytes_.size(); }

private:
    size_t remaining() const { return bytes_.size() - offset_; }

    std::span<const uint8_t> bytes_;
    size_t offset_ = 0;
};

void append_u16(std::vector<uint8_t>& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8u));
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void append_u8(std::vector<uint8_t>& bytes, uint8_t value) { bytes.push_back(value); }

void append_bytes(std::vector<uint8_t>& bytes, std::span<const uint8_t> input) {
    bytes.insert(bytes.end(), input.begin(), input.end());
}

void append_digest(std::vector<uint8_t>& bytes, const std::array<uint8_t, 32>& digest) {
    append_bytes(bytes, std::span<const uint8_t>(digest));
}

bool checked_add(uint64_t left, uint64_t right, uint64_t* result) {
    if (right > std::numeric_limits<uint64_t>::max() - left) return false;
    *result = left + right;
    return true;
}

bool checked_mul(uint64_t left, uint64_t right, uint64_t* result) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) return false;
    *result = left * right;
    return true;
}

std::vector<uint8_t> contract_header(uint16_t algorithm, uint32_t unit_elements,
                                     uint32_t unit_bytes, uint32_t plane_count) {
    std::vector<uint8_t> bytes;
    append_bytes(bytes, kContractMagic);
    append_u16(bytes, kContractVersion);
    append_u16(bytes, kAbiVersion);
    append_u16(bytes, algorithm);
    append_u16(bytes, 0); // flags
    append_u32(bytes, unit_elements);
    append_u32(bytes, unit_bytes);
    append_u32(bytes, unit_bytes); // minimum legal stride
    append_u32(bytes, plane_count);
    append_u32(bytes, 1); // decoded output domain: binary32
    append_u32(bytes, 1); // decoded output signedness: signed
    return bytes;
}

void append_plane_contract(std::vector<uint8_t>& bytes, uint32_t encoded_width_bits,
                           uint32_t encoded_unit_elements, uint32_t encoded_unit_bytes,
                           uint32_t storage_domain, uint32_t signedness) {
    append_u32(bytes, 1); // one values plane
    append_u32(bytes, storage_domain);
    append_u32(bytes, encoded_width_bits);
    append_u32(bytes, encoded_unit_elements);
    append_u32(bytes, encoded_unit_bytes);
    append_u32(bytes, signedness);
    append_u32(bytes, 1); // little-endian byte order
}

void append_physical_contract(std::vector<uint8_t>& bytes, bool q4) {
    append_u16(bytes, 1); // physical identity schema version
    append_u16(bytes, 1); // arithmetic identity version
    append_digest(bytes, {}); // no codebook

    append_u16(bytes, q4 ? static_cast<uint16_t>(PhysicalLayoutKind::GgufBlocked)
                         : static_cast<uint16_t>(PhysicalLayoutKind::ContiguousRowMajor));
    append_u16(bytes, 1); // layout schema version
    append_u16(bytes, q4 ? static_cast<uint16_t>(PackingKind::Gguf)
                         : static_cast<uint16_t>(PackingKind::None));
    append_u8(bytes, q4 ? 2 : 1); // physical rank
    append_u8(bytes, q4 ? 1 : 0); // block rank
    const std::array<uint8_t, 8> axis_order =
        q4 ? std::array<uint8_t, 8>{0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}
           : std::array<uint8_t, 8>{0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    for (uint8_t axis : axis_order) append_u8(bytes, axis);
    append_u32(bytes, q4 ? 256 : 0); // block elements
    append_u32(bytes, q4 ? 144 : 0); // block bytes
    append_u32(bytes, 0); // layout flags

    append_u16(bytes, q4 ? static_cast<uint16_t>(QuantizationKind::BlockedAffine)
                         : static_cast<uint16_t>(QuantizationKind::None));
    append_u16(bytes, 1); // quantization schema version
    append_u16(bytes, static_cast<uint16_t>(ScalarType::F32)); // accumulation
    append_u16(bytes, q4 ? static_cast<uint16_t>(ScalarType::F16) : 0); // scale
    append_u16(bytes, q4 ? static_cast<uint16_t>(ScalarType::F16) : 0); // zero/minimum
    append_u16(bytes, 0); // bias
    append_u32(bytes, q4 ? 256 : 0);
    append_u32(bytes, q4 ? 144 : 0);
    append_u32(bytes, q4 ? 256 : 0); // group size
    append_u32(bytes, q4 ? 1 : 0); // values plane
    append_u32(bytes, 0); // quantization flags

    append_u32(bytes, 1); // one packed values plane
    append_u16(bytes, static_cast<uint16_t>(PlaneKind::Values));
    append_u16(bytes, q4 ? static_cast<uint16_t>(ScalarType::U8)
                         : static_cast<uint16_t>(ScalarType::F16));
    append_u32(bytes, q4 ? 256 : 1);
    append_u32(bytes, q4 ? 144 : 2);
    append_u32(bytes, 0);
}

void append_binary16_contract(std::vector<uint8_t>& bytes) {
    append_u32(bytes, 15); // source sign bit position
    append_u32(bytes, 5); // source exponent bits
    append_u32(bytes, 10); // source significand bits
    append_u32(bytes, 15); // source exponent bias
    append_u32(bytes, 31); // binary32 sign bit position
    append_u32(bytes, 8); // binary32 exponent bits
    append_u32(bytes, 23); // binary32 significand bits
    append_u32(bytes, 127); // binary32 exponent bias
    append_u32(bytes, 1); // normal decode: exact binary16 -> binary32
    append_u32(bytes, 2); // subnormal policy: normalize exactly
    append_u32(bytes, 3); // special policy: preserve signed zero and NaN payload
    append_u32(bytes, kRoundToNearestEven); // conversion/arithmetic rounding: RN-even
    append_u32(bytes, 0); // arithmetic scaling: identity
}

void append_address_contract(std::vector<uint8_t>& bytes, uint32_t unit_bytes) {
    append_u32(bytes, 64); // offset width
    append_u32(bytes, 64); // length width
    append_u32(bytes, 64); // stride width
    append_u32(bytes, unit_bytes); // stride >= encoded unit bytes
    append_u32(bytes, 1); // checked offset + access <= backing bytes
    append_u32(bytes, 2); // checked (count - 1) * stride + unit bytes
    append_u32(bytes, 3); // maximum access <= declared plane length
    append_u32(bytes, 0); // no implicit padding or address wrapping
}

void append_field(std::vector<uint8_t>& bytes, const PackedField& field) {
    append_u32(bytes, field.source);
    append_u32(bytes, field.extract_shift);
    append_u32(bytes, field.mask);
    append_u32(bytes, field.value_shift);
}

std::vector<uint8_t> make_f16_contract() {
    auto bytes = contract_header(1, 1, 2, 1);
    append_physical_contract(bytes, false);
    append_plane_contract(bytes, 16, 1, 2, 2, 0);
    append_binary16_contract(bytes);
    append_address_contract(bytes, 2);
    return bytes;
}

std::vector<uint8_t> make_q4_contract() {
    auto bytes = contract_header(2, 256, 144, 1);
    append_physical_contract(bytes, true);
    append_plane_contract(bytes, 8, 256, 144, 3, 0);
    append_binary16_contract(bytes); // d and dmin conversion is part of this program
    append_u32(bytes, 0); // d offset
    append_u32(bytes, 2); // d width
    append_u32(bytes, 2); // dmin offset
    append_u32(bytes, 2); // dmin width
    append_u32(bytes, 4); // packed scales offset
    append_u32(bytes, 12); // packed scales width
    append_u32(bytes, 16); // packed nibbles offset
    append_u32(bytes, 128); // packed nibbles width
    append_u32(bytes, 8); // eight scale/min groups
    append_u32(bytes, 32); // values per nibble half
    append_u32(bytes, 32); // q bytes per nibble half
    append_u32(bytes, 1); // d/dmin are binary16 little-endian
    append_u32(bytes, 1); // scale/min integer conversion is binary32
    append_u32(bytes, 2); // value = (d * scale) * q - (dmin * min)
    append_u32(bytes, 0); // no integer rounding
    append_u32(bytes, 3); // preserve IEEE special values from d/dmin
    append_u32(bytes, 4); // no fused multiply-add; round each binary32 primitive
    append_u32(bytes, 0); // nibble signedness: unsigned

    const auto direct = [](uint32_t source, uint32_t mask) {
        return PackedField{source, 0, mask, 0};
    };
    const auto packed_low = [](uint32_t source) {
        return PackedField{source, 0, 15, 0};
    };
    const auto packed_high = [](uint32_t source) {
        return PackedField{source, 6, 3, 4};
    };
    for (uint32_t group = 0; group != 8; ++group) {
        const bool direct_fields = group < 4;
        const uint32_t scale_index = group;
        const uint32_t minimum_index = group + 4;
        const PackedField scale_low = direct_fields ? direct(scale_index, 63)
                                                    : packed_low(scale_index + 4);
        const PackedField scale_high = direct_fields ? PackedField{}
                                                     : packed_high(scale_index - 4);
        const PackedField minimum_low = direct_fields ? direct(minimum_index, 63)
                                                      : PackedField{scale_index + 4, 4, 15, 0};
        const PackedField minimum_high = direct_fields ? PackedField{}
                                                       : packed_high(scale_index);
        append_field(bytes, scale_low);
        append_field(bytes, scale_high);
        append_field(bytes, minimum_low);
        append_field(bytes, minimum_high);
        append_u32(bytes, 16 + (group / 2) * 32); // q source for this output half
        append_u32(bytes, (group / 2) * 64 + (group % 2) * 32); // output half
        append_u32(bytes, (group % 2) * 4); // nibble shift
        append_u32(bytes, 15); // nibble mask
    }
    append_address_contract(bytes, 144);
    return bytes;
}

bool read_binary16_contract(Reader& reader, Binary16Spec* spec) {
    return reader.read_u32(&spec->sign_bit) &&
           reader.read_u32(&spec->exponent_bits) &&
           reader.read_u32(&spec->significand_bits) &&
           reader.read_u32(&spec->exponent_bias) &&
           reader.read_u32(&spec->target_sign_bit) &&
           reader.read_u32(&spec->target_exponent_bits) &&
           reader.read_u32(&spec->target_significand_bits) &&
           reader.read_u32(&spec->target_exponent_bias) &&
           reader.read_u32(&spec->normal_policy) &&
           reader.read_u32(&spec->subnormal_policy) &&
           reader.read_u32(&spec->special_policy) &&
           reader.read_u32(&spec->rounding_mode) &&
           reader.read_u32(&spec->scaling_policy);
}

bool read_field(Reader& reader, PackedField* field) {
    return reader.read_u32(&field->source) && reader.read_u32(&field->extract_shift) &&
           reader.read_u32(&field->mask) && reader.read_u32(&field->value_shift);
}

bool read_physical_contract(Reader& reader, PhysicalSpec* spec) {
    if (!reader.read_u16(&spec->identity_version) ||
        !reader.read_u16(&spec->arithmetic_version) ||
        !reader.read_digest(&spec->codebook_digest) ||
        !reader.read_u16(&spec->layout_kind) || !reader.read_u16(&spec->layout_version) ||
        !reader.read_u16(&spec->layout_packing) || !reader.read_u8(&spec->layout_rank) ||
        !reader.read_u8(&spec->layout_block_rank))
        return false;
    for (uint8_t& axis : spec->layout_axis_order)
        if (!reader.read_u8(&axis)) return false;
    if (!reader.read_u32(&spec->layout_block_elements) ||
        !reader.read_u32(&spec->layout_block_bytes) ||
        !reader.read_u32(&spec->layout_flags) ||
        !reader.read_u16(&spec->quantization_kind) ||
        !reader.read_u16(&spec->quantization_version) ||
        !reader.read_u16(&spec->quantization_accumulation_type) ||
        !reader.read_u16(&spec->quantization_scale_type) ||
        !reader.read_u16(&spec->quantization_zero_type) ||
        !reader.read_u16(&spec->quantization_bias_type) ||
        !reader.read_u32(&spec->quantization_block_elements) ||
        !reader.read_u32(&spec->quantization_block_bytes) ||
        !reader.read_u32(&spec->quantization_group_size) ||
        !reader.read_u32(&spec->quantization_required_plane_mask) ||
        !reader.read_u32(&spec->quantization_flags) ||
        !reader.read_u32(&spec->plane_count) || spec->plane_count > spec->planes.size())
        return false;
    for (uint32_t index = 0; index != spec->plane_count; ++index) {
        PhysicalPlaneSpec& plane = spec->planes[index];
        if (!reader.read_u16(&plane.kind) || !reader.read_u16(&plane.storage_type) ||
            !reader.read_u32(&plane.logical_elements_covered) ||
            !reader.read_u32(&plane.bytes_per_block) || !reader.read_u32(&plane.flags))
            return false;
    }
    return true;
}

bool read_address(Reader& reader, AddressSpec* address) {
    return reader.read_u32(&address->offset_bits) && reader.read_u32(&address->length_bits) &&
           reader.read_u32(&address->stride_bits) && reader.read_u32(&address->minimum_stride) &&
           reader.read_u32(&address->checked_span_rule) &&
           reader.read_u32(&address->checked_access_rule) &&
           reader.read_u32(&address->declared_length_rule) &&
           reader.read_u32(&address->wrapping_rule);
}

bool valid_binary16(const Binary16Spec& spec) {
    return spec.sign_bit == 15 && spec.exponent_bits == 5 && spec.significand_bits == 10 &&
           spec.exponent_bias == 15 && spec.target_sign_bit == 31 &&
           spec.target_exponent_bits == 8 && spec.target_significand_bits == 23 &&
           spec.target_exponent_bias == 127 && spec.normal_policy == 1 &&
           spec.subnormal_policy == 2 && spec.special_policy == 3 &&
           spec.rounding_mode == kRoundToNearestEven && spec.scaling_policy == 0;
}

bool valid_field(const PackedField& field, uint32_t width) {
    if (field.source >= width || field.extract_shift >= 8 || field.value_shift >= 8)
        return false;
    if (field.mask > 0xffu || (field.mask << field.extract_shift) > 0xffu)
        return false;
    return (field.mask << field.value_shift) <= 0xffu;
}

bool valid_range(uint32_t offset, uint32_t length, uint32_t limit) {
    return static_cast<uint64_t>(offset) + length <= limit;
}

bool valid_address(const AddressSpec& address, uint32_t unit_bytes) {
    return address.offset_bits == 64 && address.length_bits == 64 &&
           address.stride_bits == 64 && address.minimum_stride == unit_bytes &&
           address.checked_span_rule == 1 && address.checked_access_rule == 2 &&
           address.declared_length_rule == 3 && address.wrapping_rule == 0;
}

bool valid_common(const ParsedProgram& program) {
    return program.abi_version == kAbiVersion && program.unit_elements != 0 &&
           program.unit_bytes != 0 && program.minimum_stride == program.unit_bytes &&
           program.plane_count == 1 && program.output_domain == 1 &&
           program.output_signedness == 1 && program.plane.count == 1 &&
           program.plane.encoded_unit_elements == program.unit_elements &&
           program.plane.encoded_unit_bytes == program.unit_bytes &&
           program.plane.byte_order == 1 && program.plane.signedness == 0 &&
           valid_address(program.address, program.unit_bytes);
}

bool zero_digest(const std::array<uint8_t, 32>& digest) {
    return std::all_of(digest.begin(), digest.end(), [](uint8_t value) { return value == 0; });
}

bool valid_physical_contract(const ParsedProgram& program) {
    const PhysicalSpec& physical = program.physical;
    const std::array<uint8_t, 8> f16_axes =
        {0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    const std::array<uint8_t, 8> q4_axes =
        {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    if (physical.identity_version != 1 || physical.arithmetic_version != kAbiVersion ||
        !zero_digest(physical.codebook_digest) || physical.layout_version != 1 ||
        physical.layout_flags != 0 || physical.quantization_version != 1 ||
        physical.quantization_accumulation_type != static_cast<uint16_t>(ScalarType::F32) ||
        physical.quantization_flags != 0 || physical.plane_count == 0)
        return false;
    const bool q4 = program.q4_mode;
    if (physical.layout_kind != static_cast<uint16_t>(q4 ? PhysicalLayoutKind::GgufBlocked
                                                         : PhysicalLayoutKind::ContiguousRowMajor) ||
        physical.layout_packing != static_cast<uint16_t>(q4 ? PackingKind::Gguf
                                                             : PackingKind::None) ||
        physical.layout_rank != (q4 ? 2 : 1) || physical.layout_block_rank != (q4 ? 1 : 0) ||
        physical.layout_axis_order != (q4 ? q4_axes : f16_axes) ||
        physical.layout_block_elements != (q4 ? 256 : 0) ||
        physical.layout_block_bytes != (q4 ? 144 : 0) ||
        physical.quantization_kind != static_cast<uint16_t>(q4 ? QuantizationKind::BlockedAffine
                                                               : QuantizationKind::None) ||
        physical.quantization_scale_type !=
            static_cast<uint16_t>(q4 ? ScalarType::F16 : static_cast<ScalarType>(0)) ||
        physical.quantization_zero_type !=
            static_cast<uint16_t>(q4 ? ScalarType::F16 : static_cast<ScalarType>(0)) ||
        physical.quantization_bias_type != 0 ||
        physical.quantization_block_elements != (q4 ? 256 : 0) ||
        physical.quantization_block_bytes != (q4 ? 144 : 0) ||
        physical.quantization_group_size != (q4 ? 256 : 0) ||
        physical.quantization_required_plane_mask != (q4 ? 1 : 0) ||
        physical.plane_count != 1)
        return false;
    const PhysicalPlaneSpec& plane = physical.planes[0];
    return plane.kind == static_cast<uint16_t>(PlaneKind::Values) &&
           plane.storage_type == static_cast<uint16_t>(q4 ? ScalarType::U8 : ScalarType::F16) &&
           plane.logical_elements_covered == (q4 ? 256 : 1) &&
           plane.bytes_per_block == (q4 ? 144 : 2) && plane.flags == 0;
}

bool valid_q4(const ParsedProgram& program) {
    const Q4Spec& q4 = program.q4;
    if (!valid_binary16(q4.scalar) || q4.d_encoding != 1 || q4.scale_conversion != 1 ||
        q4.value_formula != 2 || q4.integer_rounding != 0 || q4.special_policy != 3 ||
        q4.fma_policy != 4 || q4.nibble_signedness != 0 ||
        !valid_range(q4.d_offset, q4.d_width, 144) ||
        !valid_range(q4.dmin_offset, q4.dmin_width, 144) ||
        !valid_range(q4.scales_offset, q4.scales_width, 144) ||
        !valid_range(q4.q_offset, q4.q_width, 144) || q4.group_count != 8 ||
        q4.values_per_half != 32 || q4.q_bytes_per_half != 32)
        return false;
    std::array<bool, 256> outputs{};
    std::array<bool, 128> q_bytes{};
    for (const Q4Group& group : q4.groups) {
        if (!valid_field(group.scale_low, q4.scales_width) ||
            !valid_field(group.scale_high, q4.scales_width) ||
            !valid_field(group.minimum_low, q4.scales_width) ||
            !valid_field(group.minimum_high, q4.scales_width) ||
            group.q_shift >= 8 || group.q_mask > 15 ||
            group.q_offset < q4.q_offset ||
            !valid_range(group.q_offset, q4.q_bytes_per_half,
                         q4.q_offset + q4.q_width) ||
            !valid_range(group.output_offset, q4.values_per_half, program.unit_elements))
            return false;
        for (uint32_t i = 0; i != q4.q_bytes_per_half; ++i) {
            const uint32_t index = group.q_offset - q4.q_offset + i;
            if (index >= q_bytes.size()) return false;
            q_bytes[index] = true;
        }
        for (uint32_t i = 0; i != q4.values_per_half; ++i) {
            if (outputs[group.output_offset + i]) return false;
            outputs[group.output_offset + i] = true;
        }
    }
    return std::all_of(outputs.begin(), outputs.end(), [](bool value) { return value; }) &&
           std::all_of(q_bytes.begin(), q_bytes.end(), [](bool value) { return value; });
}

std::optional<ParsedProgram> parse_program(std::span<const uint8_t> bytes) {
    Reader reader(bytes);
    uint16_t contract_version = 0;
    uint16_t algorithm = 0;
    uint16_t flags = 0;
    ParsedProgram program;
    if (!reader.read_magic(kContractMagic) || !reader.read_u16(&contract_version) ||
        !reader.read_u16(&program.abi_version) || !reader.read_u16(&algorithm) ||
        !reader.read_u16(&flags) || !reader.read_u32(&program.unit_elements) ||
        !reader.read_u32(&program.unit_bytes) || !reader.read_u32(&program.minimum_stride) ||
        !reader.read_u32(&program.plane_count) || !reader.read_u32(&program.output_domain) ||
        !reader.read_u32(&program.output_signedness) ||
        contract_version != kContractVersion || flags != 0)
        return std::nullopt;
    if (algorithm == 1)
        program.q4_mode = false;
    else if (algorithm == 2)
        program.q4_mode = true;
    else
        return std::nullopt;
    if (!read_physical_contract(reader, &program.physical)) return std::nullopt;
    if (!reader.read_u32(&program.plane.count) || !reader.read_u32(&program.plane.storage_domain) ||
        !reader.read_u32(&program.plane.encoded_width_bits) ||
        !reader.read_u32(&program.plane.encoded_unit_elements) ||
        !reader.read_u32(&program.plane.encoded_unit_bytes) ||
        !reader.read_u32(&program.plane.signedness) || !reader.read_u32(&program.plane.byte_order))
        return std::nullopt;

    if (!program.q4_mode) {
        if (!read_binary16_contract(reader, &program.binary16) ||
            !read_address(reader, &program.address) || !reader.done())
            return std::nullopt;
        if (!valid_common(program) || program.unit_elements != 1 || program.unit_bytes != 2 ||
            program.plane.storage_domain != 2 || program.plane.encoded_width_bits != 16 ||
            !valid_binary16(program.binary16) || !valid_physical_contract(program))
            return std::nullopt;
        return program;
    }

    Q4Spec& q4 = program.q4;
    if (!read_binary16_contract(reader, &q4.scalar) ||
        !reader.read_u32(&q4.d_offset) || !reader.read_u32(&q4.d_width) ||
        !reader.read_u32(&q4.dmin_offset) || !reader.read_u32(&q4.dmin_width) ||
        !reader.read_u32(&q4.scales_offset) || !reader.read_u32(&q4.scales_width) ||
        !reader.read_u32(&q4.q_offset) || !reader.read_u32(&q4.q_width) ||
        !reader.read_u32(&q4.group_count) || !reader.read_u32(&q4.values_per_half) ||
        !reader.read_u32(&q4.q_bytes_per_half) || !reader.read_u32(&q4.d_encoding) ||
        !reader.read_u32(&q4.scale_conversion) || !reader.read_u32(&q4.value_formula) ||
        !reader.read_u32(&q4.integer_rounding) || !reader.read_u32(&q4.special_policy) ||
        !reader.read_u32(&q4.fma_policy) || !reader.read_u32(&q4.nibble_signedness))
        return std::nullopt;
    if (q4.group_count != 8) return std::nullopt;
    for (Q4Group& group : q4.groups) {
        if (!read_field(reader, &group.scale_low) || !read_field(reader, &group.scale_high) ||
            !read_field(reader, &group.minimum_low) || !read_field(reader, &group.minimum_high) ||
            !reader.read_u32(&group.q_offset) || !reader.read_u32(&group.output_offset) ||
            !reader.read_u32(&group.q_shift) || !reader.read_u32(&group.q_mask))
            return std::nullopt;
    }
    if (!read_address(reader, &program.address) || !reader.done()) return std::nullopt;
    if (!valid_common(program) || program.unit_elements != 256 || program.unit_bytes != 144 ||
        program.plane.storage_domain != 3 || program.plane.encoded_width_bits != 8 ||
        !valid_q4(program) || !valid_physical_contract(program))
        return std::nullopt;
    return program;
}

uint32_t mask_for(uint32_t bits) {
    return bits == 32 ? std::numeric_limits<uint32_t>::max() : ((1u << bits) - 1u);
}

float binary16_to_binary32(uint16_t value, const Binary16Spec& spec) {
    const uint32_t sign = (static_cast<uint32_t>(value) >> spec.sign_bit) & 1u;
    const uint32_t source_exponent =
        (static_cast<uint32_t>(value) >> spec.significand_bits) & mask_for(spec.exponent_bits);
    uint32_t mantissa = static_cast<uint32_t>(value) & mask_for(spec.significand_bits);
    uint32_t result = 0;
    if (source_exponent == 0) {
        if (mantissa == 0) {
            result = sign << spec.target_sign_bit;
        } else {
            int32_t exponent = -static_cast<int32_t>(spec.exponent_bias);
            while ((mantissa & (1u << spec.significand_bits)) == 0) {
                mantissa <<= 1u;
                --exponent;
            }
            ++exponent;
            const uint32_t target_exponent =
                static_cast<uint32_t>(exponent + static_cast<int32_t>(spec.target_exponent_bias));
            result = (sign << spec.target_sign_bit) |
                     ((target_exponent & mask_for(spec.target_exponent_bits)) <<
                      spec.target_significand_bits) |
                     ((mantissa & mask_for(spec.significand_bits)) <<
                      (spec.target_significand_bits - spec.significand_bits));
        }
    } else if (source_exponent == mask_for(spec.exponent_bits)) {
        result = (sign << spec.target_sign_bit) |
                 (mask_for(spec.target_exponent_bits) << spec.target_significand_bits) |
                 (mantissa << (spec.target_significand_bits - spec.significand_bits));
    } else {
        const uint32_t target_exponent =
            source_exponent - spec.exponent_bias + spec.target_exponent_bias;
        result = (sign << spec.target_sign_bit) |
                 (target_exponent << spec.target_significand_bits) |
                 (mantissa << (spec.target_significand_bits - spec.significand_bits));
    }
    return std::bit_cast<float>(result);
}

uint16_t read_u16(const uint8_t* bytes) {
    return static_cast<uint16_t>(bytes[0]) |
           (static_cast<uint16_t>(bytes[1]) << 8u);
}

uint32_t packed_value(const uint8_t* scales, const PackedField& field) {
    return ((static_cast<uint32_t>(scales[field.source]) >> field.extract_shift) & field.mask)
           << field.value_shift;
}

float rounded_mul(float left, float right, uint32_t rounding_mode) {
    if (rounding_mode != kRoundToNearestEven || std::fegetround() != FE_TONEAREST)
        return std::numeric_limits<float>::quiet_NaN();
    volatile float result = left * right;
    return result;
}

float rounded_sub(float left, float right, uint32_t rounding_mode, uint32_t fma_policy) {
    if (rounding_mode != kRoundToNearestEven || std::fegetround() != FE_TONEAREST ||
        fma_policy != 4)
        return std::numeric_limits<float>::quiet_NaN();
    volatile float result = left - right;
    return result;
}

void decode_binary16(const ParsedProgram& program, const CodecPlaneBinding& plane,
                     uint64_t units, std::vector<float>& output) {
    output.reserve(static_cast<size_t>(units));
    const uint8_t* base = plane.storage.data() + static_cast<size_t>(plane.offset);
    for (uint64_t unit = 0; unit < units; ++unit) {
        const uint8_t* encoded = base + static_cast<size_t>(unit * plane.stride);
        output.push_back(binary16_to_binary32(read_u16(encoded), program.binary16));
    }
}

void decode_q4(const ParsedProgram& program, const CodecPlaneBinding& plane,
               uint64_t units, std::vector<float>& output) {
    const Q4Spec& q4 = program.q4;
    output.resize(static_cast<size_t>(units * program.unit_elements));
    const uint8_t* base = plane.storage.data() + static_cast<size_t>(plane.offset);
    for (uint64_t unit = 0; unit < units; ++unit) {
        const uint8_t* block = base + static_cast<size_t>(unit * plane.stride);
        const float d = binary16_to_binary32(read_u16(block + q4.d_offset), q4.scalar);
        const float dmin = binary16_to_binary32(read_u16(block + q4.dmin_offset), q4.scalar);
        const uint8_t* scales = block + q4.scales_offset;
        float* values = output.data() + static_cast<size_t>(unit * program.unit_elements);
        for (const Q4Group& group : q4.groups) {
            const uint32_t scale = packed_value(scales, group.scale_low) |
                                   packed_value(scales, group.scale_high);
            const uint32_t minimum = packed_value(scales, group.minimum_low) |
                                     packed_value(scales, group.minimum_high);
            const float scaled = rounded_mul(d, static_cast<float>(scale),
                                             q4.scalar.rounding_mode);
            const float minimum_scaled = rounded_mul(dmin, static_cast<float>(minimum),
                                                    q4.scalar.rounding_mode);
            const uint8_t* q = block + group.q_offset;
            for (uint32_t i = 0; i != q4.values_per_half; ++i) {
                const uint32_t quantized = (static_cast<uint32_t>(q[i]) >> group.q_shift) &
                                           group.q_mask;
                values[group.output_offset + i] = rounded_sub(
                    rounded_mul(scaled, static_cast<float>(quantized),
                                q4.scalar.rounding_mode),
                    minimum_scaled, q4.scalar.rounding_mode, q4.fma_policy);
            }
        }
    }
}

CodecProgramError validate_binding(const CodecProgram& program, const ParsedProgram& parsed,
                                   const CodecProgramDeclaration& declaration,
                                   std::optional<uint64_t> max_decode_elements,
                                   uint64_t* units) {
    if (declaration.identity != program.identity()) return CodecProgramError::UnknownIdentity;
    if (declaration.planes.size() != parsed.plane_count)
        return CodecProgramError::InvalidPlaneCount;
    if (declaration.element_count == 0 ||
        (max_decode_elements && declaration.element_count > *max_decode_elements))
        return CodecProgramError::InvalidElementCount;
    if (declaration.element_count % parsed.unit_elements != 0)
        return CodecProgramError::InvalidElementCount;
    *units = declaration.element_count / parsed.unit_elements;
    const CodecPlaneBinding& plane = declaration.planes[0];
    if (plane.offset > plane.storage.size()) return CodecProgramError::InvalidOffset;
    const uint64_t storage_size = static_cast<uint64_t>(plane.storage.size());
    if (plane.length > storage_size - plane.offset)
        return CodecProgramError::InvalidLength;
    if (plane.stride < parsed.address.minimum_stride) return CodecProgramError::InvalidStride;

    uint64_t displacement = 0;
    if (!checked_mul(*units - 1, plane.stride, &displacement))
        return CodecProgramError::ArithmeticOverflow;
    uint64_t max_access = 0;
    if (!checked_add(displacement, parsed.unit_bytes, &max_access))
        return CodecProgramError::ArithmeticOverflow;
    if (max_access > plane.length) return CodecProgramError::AccessOutOfBounds;
    uint64_t absolute_end = 0;
    if (!checked_add(plane.offset, max_access, &absolute_end) || absolute_end > storage_size)
        return CodecProgramError::AccessOutOfBounds;
    return CodecProgramError::None;
}

bool decode_environment_available(const ParsedProgram& parsed) {
    const uint32_t rounding_mode = !parsed.q4_mode
                                       ? parsed.binary16.rounding_mode
                                       : parsed.q4.scalar.rounding_mode;
    return rounding_mode == kRoundToNearestEven && std::fegetround() == FE_TONEAREST;
}

bool same_identity(const CodecProgramIdentity& left, const CodecProgramIdentity& right) {
    return left.abi_version == right.abi_version && left.contract_digest == right.contract_digest;
}

} // namespace

CodecProgramDigest codec_program_digest(std::span<const uint8_t> canonical_bytes) {
    CodecProgramDigest digest{};
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    constexpr size_t chunk_size = 1024 * 1024;
    for (size_t offset = 0; offset < canonical_bytes.size(); offset += chunk_size) {
        const size_t count = std::min(chunk_size, canonical_bytes.size() - offset);
        CC_SHA256_Update(&context, canonical_bytes.data() + offset,
                         static_cast<CC_LONG>(count));
    }
    CC_SHA256_Final(digest.data(), &context);
    return digest;
}

CodecProgramRegistry make_application_codec_registry() {
    CodecProgramRegistry registry;
    const auto add_if_conforming = [&registry](std::vector<uint8_t> canonical_bytes) {
        if (!parse_program(canonical_bytes)) return;
        CodecProgram program;
        program.canonical_bytes_ = std::move(canonical_bytes);
        program.identity_.abi_version = kAbiVersion;
        program.identity_.contract_digest = codec_program_digest(program.canonical_bytes_);
        registry.programs_.push_back(std::move(program));
    };
    add_if_conforming(make_f16_contract());
    add_if_conforming(make_q4_contract());
    return registry;
}

const CodecProgram* CodecProgramRegistry::resolve(const CodecProgramIdentity& identity) const noexcept {
    for (const CodecProgram& program : programs_) {
        if (!same_identity(program.identity_, identity)) continue;
        if (!parse_program(program.canonical_bytes_) ||
            codec_program_digest(program.canonical_bytes_) != program.identity_.contract_digest)
            return nullptr;
        return &program;
    }
    return nullptr;
}

CodecProgramDecodeResult CodecProgramRegistry::decode(
    const CodecProgramDeclaration& declaration, uint64_t max_decode_elements) const {
    const CodecProgram* program = resolve(declaration.identity);
    if (!program) {
        for (const CodecProgram& candidate : programs_) {
            if (candidate.identity_.abi_version == declaration.identity.abi_version &&
                candidate.identity_.contract_digest == declaration.identity.contract_digest)
                return CodecProgramError::ContractDigestMismatch;
        }
        return CodecProgramError::UnknownIdentity;
    }
    const auto parsed = parse_program(program->canonical_bytes_);
    if (!parsed) return CodecProgramError::InvalidContract;
    if (!decode_environment_available(*parsed))
        return CodecProgramError::RoundingModeUnavailable;
    uint64_t units = 0;
    const CodecProgramError validation = validate_binding(
        *program, *parsed, declaration, max_decode_elements, &units);
    if (validation != CodecProgramError::None) return validation;
    std::vector<float> output;
    if (!parsed->q4_mode)
        decode_binary16(*parsed, declaration.planes[0], units, output);
    else
        decode_q4(*parsed, declaration.planes[0], units, output);
    return output;
}

CodecProgramError CodecProgram::validate(const CodecProgramDeclaration& declaration) const {
    if (declaration.identity != identity_) return CodecProgramError::UnknownIdentity;
    const auto parsed = parse_program(canonical_bytes_);
    if (!parsed) return CodecProgramError::InvalidContract;
    if (codec_program_digest(canonical_bytes_) != identity_.contract_digest)
        return CodecProgramError::ContractDigestMismatch;
    uint64_t units = 0;
    return validate_binding(*this, *parsed, declaration, std::nullopt, &units);
}

bool CodecProgram::matches_physical_identity(
    const PhysicalCodecIdentity& identity) const {
    if (identity.arithmetic_digest != identity_.contract_digest) return false;
    if (codec_program_digest(canonical_bytes_) != identity_.contract_digest) return false;
    const auto parsed = parse_program(canonical_bytes_);
    if (!parsed) return false;
    const PhysicalSpec& physical = parsed->physical;
    if (identity.identity_version != physical.identity_version ||
        identity.arithmetic_version != physical.arithmetic_version ||
        identity.codebook_digest != physical.codebook_digest ||
        static_cast<uint16_t>(identity.layout.kind) != physical.layout_kind ||
        identity.layout.version != physical.layout_version ||
        static_cast<uint16_t>(identity.layout.packing) != physical.layout_packing ||
        identity.layout.rank != physical.layout_rank ||
        identity.layout.block_rank != physical.layout_block_rank ||
        identity.layout.axis_order != physical.layout_axis_order ||
        identity.layout.block_elements != physical.layout_block_elements ||
        identity.layout.block_bytes != physical.layout_block_bytes ||
        identity.layout.flags != physical.layout_flags ||
        static_cast<uint16_t>(identity.quantization.kind) != physical.quantization_kind ||
        identity.quantization.version != physical.quantization_version ||
        static_cast<uint16_t>(identity.quantization.accumulation_type) !=
            physical.quantization_accumulation_type ||
        static_cast<uint16_t>(identity.quantization.scale_type) !=
            physical.quantization_scale_type ||
        static_cast<uint16_t>(identity.quantization.zero_type) != physical.quantization_zero_type ||
        static_cast<uint16_t>(identity.quantization.bias_type) != physical.quantization_bias_type ||
        identity.quantization.block_elements != physical.quantization_block_elements ||
        identity.quantization.block_bytes != physical.quantization_block_bytes ||
        identity.quantization.group_size != physical.quantization_group_size ||
        identity.quantization.required_plane_mask != physical.quantization_required_plane_mask ||
        identity.quantization.flags != physical.quantization_flags ||
        identity.planes.size() != physical.plane_count)
        return false;
    for (size_t index = 0; index != identity.planes.size(); ++index) {
        const PhysicalPlaneSchema& actual = identity.planes[index];
        const PhysicalPlaneSpec& expected = physical.planes[index];
        if (static_cast<uint16_t>(actual.kind) != expected.kind ||
            static_cast<uint16_t>(actual.storage_type) != expected.storage_type ||
            actual.logical_elements_covered != expected.logical_elements_covered ||
            actual.bytes_per_block != expected.bytes_per_block || actual.flags != expected.flags)
            return false;
    }
    return valid_physical_contract(*parsed);
}

} // namespace Laplace
