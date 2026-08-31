#include "codec_certificate.h"
#include "physical_codec.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <limits>
#include <new>
#include <optional>
#include <utility>

namespace Laplace {

namespace {

constexpr uint16_t kAbiVersion = 1;
constexpr uint16_t kCertificateVersion = 2;
constexpr uint16_t kFlags = 0;
constexpr uint32_t kMaximumBytes = 64u * 1024u;
constexpr uint32_t kMaximumUnits = 1u << 24;
constexpr uint16_t kMaximumPlanes = 8;
constexpr uint16_t kMaximumNodes = 128;
constexpr uint16_t kMaximumConstants = 64;
constexpr uint8_t kMaximumDepth = 32;
constexpr uint8_t kMaximumMaps = 64;
constexpr uint32_t kMaximumRecords = 4096;
constexpr uint8_t kLittleEndian = 1;
constexpr uint8_t kBitsLeastSignificantFirst = 1;
constexpr uint8_t kAccessZero = 1;
constexpr std::array<uint8_t, 8> kMagic =
    {'L', 'P', 'C', 'E', 'R', 'T', 1, 0};

enum class NodeOp : uint8_t {
    LoadScalar = 1,
    LoadBits = 2,
    CastFloat = 3,
    Add = 4,
    Sub = 5,
    Mul = 6,
    Fma = 7,
    Neg = 8,
    Constant = 9,
};
enum class NodeType : uint8_t { Unsigned = 1, Float = 2, Signed = 3 };
enum class AccessEncoding : uint8_t {
    Binary16 = 1,
    Binary32 = 2,
    Unsigned8 = 3,
    Unsigned16 = 4,
    Unsigned32 = 5,
    Signed8 = 6,
};

struct PlaneSpec {
    CodecCertificatePlaneRole role = CodecCertificatePlaneRole::Values;
    CodecCertificateScalar scalar = CodecCertificateScalar::Binary16;
    CodecCertificateStorageScalar storage_scalar =
        CodecCertificateStorageScalar::Binary16;
    uint8_t bit_order = 0;
    uint8_t byte_order = 0;
    uint16_t width_bits = 0;
    uint32_t elements_per_unit = 0;
    uint32_t bytes_per_unit = 0;
    uint32_t alignment = 0;
    uint32_t base = 0;
    uint32_t stride = 0;
};
struct NodeSpec {
    NodeOp op = NodeOp::LoadScalar;
    NodeType type = NodeType::Unsigned;
    uint8_t plane = 0xff;
    uint8_t flags = 0;
    uint16_t arg0 = 0xffff;
    uint16_t arg1 = 0xffff;
    uint16_t arg2 = 0xffff;
    uint32_t immediate = 0;
};
struct AccessRecord {
    uint32_t byte_offset = 0;
    uint8_t bit_offset = 0;
    uint8_t width_bits = 0;
    AccessEncoding encoding = AccessEncoding::Unsigned8;
    uint8_t flags = 0;
    uint8_t value_shift = 0;
};
struct AccessMap {
    uint8_t plane = 0;
    uint32_t first = 0;
    uint32_t count = 0;
};
struct Parsed {
    CodecCertificateSummary summary;
    std::vector<PlaneSpec> planes;
    std::vector<NodeSpec> nodes;
    std::vector<uint32_t> constants;
    std::vector<AccessMap> maps;
    std::vector<AccessRecord> records;
};

class Reader {
public:
    explicit Reader(std::span<const uint8_t> bytes) : bytes_(bytes) {}
    bool bytes(size_t count, std::span<const uint8_t>* result) {
        if (count > remaining()) return false;
        *result = bytes_.subspan(offset_, count);
        offset_ += count;
        return true;
    }
    bool u8(uint8_t* value) {
        std::span<const uint8_t> bytes;
        if (!this->bytes(1, &bytes)) return false;
        *value = bytes[0];
        return true;
    }
    bool u16(uint16_t* value) {
        std::span<const uint8_t> bytes;
        if (!this->bytes(2, &bytes)) return false;
        *value = static_cast<uint16_t>(bytes[0]) |
                 (static_cast<uint16_t>(bytes[1]) << 8u);
        return true;
    }
    bool u32(uint32_t* value) {
        std::span<const uint8_t> bytes;
        if (!this->bytes(4, &bytes)) return false;
        *value = static_cast<uint32_t>(bytes[0]) |
                 (static_cast<uint32_t>(bytes[1]) << 8u) |
                 (static_cast<uint32_t>(bytes[2]) << 16u) |
                 (static_cast<uint32_t>(bytes[3]) << 24u);
        return true;
    }
    bool magic() {
        std::span<const uint8_t> bytes;
        return this->bytes(kMagic.size(), &bytes) &&
               std::equal(bytes.begin(), bytes.end(), kMagic.begin());
    }
    size_t remaining() const noexcept { return bytes_.size() - offset_; }

private:
    std::span<const uint8_t> bytes_;
    size_t offset_ = 0;
};

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
void append_node(std::vector<uint8_t>& bytes, NodeOp op, NodeType type,
                 uint8_t plane = 0xff, uint16_t arg0 = 0xffff,
                 uint16_t arg1 = 0xffff, uint16_t arg2 = 0xffff,
                 uint32_t immediate = 0) {
    append_u8(bytes, static_cast<uint8_t>(op));
    append_u8(bytes, static_cast<uint8_t>(type));
    append_u8(bytes, plane);
    append_u8(bytes, 0);
    append_u16(bytes, arg0);
    append_u16(bytes, arg1);
    append_u16(bytes, arg2);
    append_u32(bytes, immediate);
    append_u16(bytes, 0);
}
void append_plane(std::vector<uint8_t>& bytes, CodecCertificatePlaneRole role,
                  CodecCertificateScalar scalar,
                  CodecCertificateStorageScalar storage_scalar,
                  uint8_t bit_order, uint16_t width_bits, uint32_t elements,
                  uint32_t unit_bytes, uint32_t alignment, uint32_t base,
                  uint32_t stride) {
    append_u8(bytes, static_cast<uint8_t>(role));
    append_u8(bytes, static_cast<uint8_t>(scalar));
    append_u8(bytes, static_cast<uint8_t>(storage_scalar));
    append_u8(bytes, bit_order);
    append_u8(bytes, kLittleEndian);
    append_u16(bytes, width_bits);
    append_u32(bytes, elements);
    append_u32(bytes, unit_bytes);
    append_u32(bytes, alignment);
    append_u32(bytes, base);
    append_u32(bytes, stride);
}

struct PhysicalFields {
    uint8_t layout_kind = 1;
    uint8_t packing = 0;
    uint8_t block_rank = 0;
    uint32_t block_elements = 0;
    uint32_t block_bytes = 0;
    uint8_t quantization_kind = 0;
    uint32_t quantization_block_elements = 0;
    uint32_t quantization_block_bytes = 0;
    uint32_t quantization_group_size = 0;
    uint32_t required_plane_mask = 0;
};
std::vector<uint8_t> header(uint32_t elements, uint32_t unit_bytes,
                            uint8_t plane_count, uint16_t node_count,
                            uint8_t constant_count, uint8_t depth,
                            PhysicalFields physical) {
    std::vector<uint8_t> bytes;
    bytes.reserve(kMaximumBytes);
    bytes.insert(bytes.end(), kMagic.begin(), kMagic.end());
    append_u16(bytes, kAbiVersion);
    append_u16(bytes, kCertificateVersion);
    append_u16(bytes, kFlags);
    append_u8(bytes, static_cast<uint8_t>(CodecCertificateScalar::Binary32));
    append_u8(bytes, static_cast<uint8_t>(
                            CodecCertificateSpecialValuePolicy::PreserveIeee));
    append_u16(bytes, 0);
    append_u32(bytes, elements);
    append_u32(bytes, unit_bytes);
    append_u32(bytes, kMaximumUnits);
    append_u8(bytes, plane_count);
    append_u8(bytes, depth);
    append_u16(bytes, node_count);
    append_u8(bytes, constant_count);
    append_u8(bytes, 0);
    append_u8(bytes, physical.layout_kind);
    append_u8(bytes, physical.packing);
    append_u8(bytes, physical.block_rank);
    append_u8(bytes, 0);
    append_u32(bytes, physical.block_elements);
    append_u32(bytes, physical.block_bytes);
    append_u8(bytes, physical.quantization_kind);
    append_u8(bytes, 0);
    append_u16(bytes, 0);
    append_u32(bytes, physical.quantization_block_elements);
    append_u32(bytes, physical.quantization_block_bytes);
    append_u32(bytes, physical.quantization_group_size);
    append_u32(bytes, physical.required_plane_mask);
    return bytes;
}

struct BuilderMap {
    uint8_t plane = 0;
    std::vector<AccessRecord> records;
};
class Builder {
public:
    Builder(uint32_t elements, uint32_t bytes, PhysicalFields physical)
        : elements_(elements), bytes_(bytes), physical_(physical) {}
    uint8_t add_map(uint8_t plane, std::vector<AccessRecord> records) {
        maps_.push_back({plane, std::move(records)});
        return static_cast<uint8_t>(maps_.size() - 1);
    }
    std::vector<uint8_t> finish(std::vector<PlaneSpec> planes,
                                std::vector<NodeSpec> nodes,
                                std::vector<uint32_t> constants,
                                uint8_t depth) const {
        auto bytes = header(elements_, bytes_,
                            static_cast<uint8_t>(planes.size()),
                            static_cast<uint16_t>(nodes.size()),
                            static_cast<uint8_t>(constants.size()), depth,
                            physical_);
        for (const PlaneSpec& plane : planes)
            append_plane(bytes, plane.role, plane.scalar,
                         plane.storage_scalar, plane.bit_order,
                         plane.width_bits, plane.elements_per_unit,
                         plane.bytes_per_unit, plane.alignment, plane.base,
                         plane.stride);
        for (const NodeSpec& node : nodes)
            append_node(bytes, node.op, node.type, node.plane, node.arg0,
                        node.arg1, node.arg2, node.immediate);
        for (uint32_t constant : constants) append_u32(bytes, constant);
        append_u32(bytes, 2);
        append_u16(bytes, static_cast<uint16_t>(maps_.size()));
        append_u16(bytes, 0);
        uint32_t record_count = 0;
        for (const BuilderMap& map : maps_)
            record_count += static_cast<uint32_t>(map.records.size());
        append_u32(bytes, record_count);
        uint32_t first = 0;
        for (const BuilderMap& map : maps_) {
            append_u8(bytes, map.plane);
            append_u8(bytes, 0);
            append_u8(bytes, 0);
            append_u8(bytes, 0);
            append_u32(bytes, first);
            append_u32(bytes, static_cast<uint32_t>(map.records.size()));
            first += static_cast<uint32_t>(map.records.size());
        }
        for (const BuilderMap& map : maps_) {
            for (const AccessRecord& record : map.records) {
                append_u32(bytes, record.byte_offset);
                append_u8(bytes, record.bit_offset);
                append_u8(bytes, record.width_bits);
                append_u8(bytes, static_cast<uint8_t>(record.encoding));
                append_u8(bytes, record.flags);
                append_u8(bytes, record.value_shift);
                append_u8(bytes, 0);
                append_u8(bytes, 0);
                append_u8(bytes, 0);
            }
        }
        return bytes;
    }

private:
    uint32_t elements_;
    uint32_t bytes_;
    PhysicalFields physical_;
    std::vector<BuilderMap> maps_;
};

AccessRecord access(uint32_t offset, uint8_t bit, uint8_t width,
                    AccessEncoding encoding, uint8_t shift = 0) {
    return {offset, bit, width, encoding, 0, shift};
}
AccessRecord zero_access() {
    return {0, 0, 0, AccessEncoding::Unsigned8, kAccessZero, 0};
}
std::vector<AccessRecord> repeated(uint32_t count, AccessRecord record) {
    return std::vector<AccessRecord>(count, record);
}
PhysicalFields raw_physical() { return {}; }
PhysicalFields gguf_physical(uint32_t bytes) {
    return {2, 1, 1, 256, bytes, 1, 256, bytes, 256, 1};
}
PhysicalFields gguf_legacy_physical(uint32_t elements, uint32_t bytes) {
    return {2, 1, 1, elements, bytes, 1, elements, bytes, elements, 1};
}
PhysicalFields grouped_physical() {
    return {3, 2, 1, 256, 64, 1, 256, 64, 256, 7};
}

std::vector<uint8_t> build_raw_f16() {
    Builder builder(1, 2, raw_physical());
    const uint8_t map = builder.add_map(
        0, {access(0, 0, 16, AccessEncoding::Binary16)});
    return builder.finish(
        {{CodecCertificatePlaneRole::Values,
          CodecCertificateScalar::Binary16,
          CodecCertificateStorageScalar::Binary16, 0, 0, 16, 1, 2, 2, 0,
          2}},
        {{NodeOp::LoadScalar, NodeType::Float, 0, 0, 0xffff, 0xffff,
          0xffff, map}},
        {}, 1);
}
std::vector<uint8_t> build_raw_f32() {
    Builder builder(1, 4, raw_physical());
    const uint8_t map = builder.add_map(
        0, {access(0, 0, 32, AccessEncoding::Binary32)});
    return builder.finish(
        {{CodecCertificatePlaneRole::Values,
          CodecCertificateScalar::Binary32,
          CodecCertificateStorageScalar::Binary32, 0, 0, 32, 1, 4, 4, 0,
          4}},
        {{NodeOp::LoadScalar, NodeType::Float, 0, 0, 0xffff, 0xffff,
          0xffff, map}},
        {}, 1);
}

std::vector<uint8_t> build_q4() {
    Builder builder(256, 144, gguf_physical(144));
    std::vector<AccessRecord> d = repeated(
        1, access(0, 0, 16, AccessEncoding::Binary16));
    std::vector<AccessRecord> dmin = repeated(
        1, access(2, 0, 16, AccessEncoding::Binary16));
    std::vector<AccessRecord> q, sl, sh, ml, mh;
    q.reserve(256); sl.reserve(256); sh.reserve(256);
    ml.reserve(256); mh.reserve(256);
    for (uint32_t element = 0; element < 256; ++element) {
        const uint32_t group = element / 32;
        const uint32_t within = element % 32;
        const uint32_t half = group % 2;
        q.push_back(access(16 + (group / 2) * 32 + within,
                           static_cast<uint8_t>(half * 4), 4,
                           AccessEncoding::Unsigned8));
        if (group < 4) {
            sl.push_back(access(4 + group, 0, 6, AccessEncoding::Unsigned8));
            sh.push_back(zero_access());
            ml.push_back(access(8 + group, 0, 6, AccessEncoding::Unsigned8));
            mh.push_back(zero_access());
        } else {
            sl.push_back(access(4 + group + 4, 0, 4,
                                AccessEncoding::Unsigned8));
            sh.push_back(access(4 + group - 4, 6, 2,
                                AccessEncoding::Unsigned8, 4));
            ml.push_back(access(4 + group + 4, 4, 4,
                                AccessEncoding::Unsigned8));
            mh.push_back(access(4 + group + 4, 6, 2,
                                AccessEncoding::Unsigned8, 4));
        }
    }
    const uint8_t dm = builder.add_map(0, std::move(d));
    const uint8_t dmm = builder.add_map(0, std::move(dmin));
    const uint8_t qm = builder.add_map(0, std::move(q));
    const uint8_t slm = builder.add_map(0, std::move(sl));
    const uint8_t shm = builder.add_map(0, std::move(sh));
    const uint8_t mlm = builder.add_map(0, std::move(ml));
    const uint8_t mhm = builder.add_map(0, std::move(mh));
    std::vector<NodeSpec> nodes = {
        {NodeOp::LoadScalar, NodeType::Float, 0, 0, 0xffff, 0xffff, 0xffff,
         dm},
        {NodeOp::LoadScalar, NodeType::Float, 0, 0, 0xffff, 0xffff, 0xffff,
         dmm},
        {NodeOp::LoadBits, NodeType::Unsigned, 0, 0, 0xffff, 0xffff, 0xffff,
         qm},
        {NodeOp::LoadBits, NodeType::Unsigned, 0, 0, 0xffff, 0xffff, 0xffff,
         slm},
        {NodeOp::LoadBits, NodeType::Unsigned, 0, 0, 0xffff, 0xffff, 0xffff,
         shm},
        {NodeOp::LoadBits, NodeType::Unsigned, 0, 0, 0xffff, 0xffff, 0xffff,
         mlm},
        {NodeOp::LoadBits, NodeType::Unsigned, 0, 0, 0xffff, 0xffff, 0xffff,
         mhm},
        {NodeOp::CastFloat, NodeType::Float, 0xff, 0, 2, 0xffff, 0xffff, 0},
        {NodeOp::CastFloat, NodeType::Float, 0xff, 0, 3, 0xffff, 0xffff, 0},
        {NodeOp::CastFloat, NodeType::Float, 0xff, 0, 4, 0xffff, 0xffff, 0},
        {NodeOp::CastFloat, NodeType::Float, 0xff, 0, 5, 0xffff, 0xffff, 0},
        {NodeOp::CastFloat, NodeType::Float, 0xff, 0, 6, 0xffff, 0xffff, 0},
        {NodeOp::Add, NodeType::Float, 0xff, 0, 8, 9, 0xffff, 0},
        {NodeOp::Add, NodeType::Float, 0xff, 0, 10, 11, 0xffff, 0},
        {NodeOp::Mul, NodeType::Float, 0xff, 0, 0, 12, 0xffff, 0},
        {NodeOp::Mul, NodeType::Float, 0xff, 0, 1, 13, 0xffff, 0},
        {NodeOp::CastFloat, NodeType::Float, 0xff, 0, 7, 0xffff, 0xffff, 0},
        {NodeOp::Mul, NodeType::Float, 0xff, 0, 14, 16, 0xffff, 0},
        {NodeOp::Neg, NodeType::Float, 0xff, 0, 15, 0xffff, 0xffff, 0},
        {NodeOp::Add, NodeType::Float, 0xff, 0, 17, 18, 0xffff, 0},
    };
    return builder.finish(
        {{CodecCertificatePlaneRole::Values,
          CodecCertificateScalar::PackedUnsigned,
          CodecCertificateStorageScalar::Unsigned8,
          kBitsLeastSignificantFirst, kLittleEndian, 8, 256, 144, 32, 0,
          144}},
        std::move(nodes), {}, 6);
}

std::vector<uint8_t> build_q5_0() {
    Builder builder(32, 22, gguf_legacy_physical(32, 22));
    std::vector<AccessRecord> ql, qh;
    ql.reserve(32);
    qh.reserve(32);
    for (uint32_t element = 0; element != 32; ++element) {
        ql.push_back(access(6 + element / 2,
                             static_cast<uint8_t>((element & 1u) * 4u), 4,
                             AccessEncoding::Unsigned8));
        qh.push_back(access(2 + element / 8,
                            static_cast<uint8_t>(element & 7u), 1,
                            AccessEncoding::Unsigned8, 4));
    }
    const uint8_t dm = builder.add_map(0, {access(0, 0, 16, AccessEncoding::Binary16)});
    const uint8_t qlm = builder.add_map(0, std::move(ql));
    const uint8_t qhm = builder.add_map(0, std::move(qh));
    std::vector<NodeSpec> nodes = {
        {NodeOp::LoadScalar, NodeType::Float, 0, 0, 0xffff, 0xffff, 0xffff, dm},
        {NodeOp::LoadBits, NodeType::Unsigned, 0, 0, 0xffff, 0xffff, 0xffff, qlm},
        {NodeOp::LoadBits, NodeType::Unsigned, 0, 0, 0xffff, 0xffff, 0xffff, qhm},
        {NodeOp::CastFloat, NodeType::Float, 0xff, 0, 1, 0xffff, 0xffff, 0},
        {NodeOp::CastFloat, NodeType::Float, 0xff, 0, 2, 0xffff, 0xffff, 0},
        {NodeOp::Add, NodeType::Float, 0xff, 0, 3, 4, 0xffff, 0},
        {NodeOp::Constant, NodeType::Float, 0xff, 0, 0xffff, 0xffff, 0xffff, 0},
        {NodeOp::Add, NodeType::Float, 0xff, 0, 5, 6, 0xffff, 0},
        {NodeOp::Mul, NodeType::Float, 0xff, 0, 0, 7, 0xffff, 0},
    };
    return builder.finish(
        {{CodecCertificatePlaneRole::Values,
          CodecCertificateScalar::PackedUnsigned,
          CodecCertificateStorageScalar::Unsigned8,
          kBitsLeastSignificantFirst, kLittleEndian, 8, 32, 22, 32, 0,
          22}},
        std::move(nodes), {std::bit_cast<uint32_t>(-16.0f)}, 5);
}

std::vector<uint8_t> build_q8_0() {
    Builder builder(32, 34, gguf_legacy_physical(32, 34));
    const uint8_t dm = builder.add_map(0, {access(0, 0, 16, AccessEncoding::Binary16)});
    const uint8_t qm = builder.add_map(0, [&] {
        std::vector<AccessRecord> records;
        records.reserve(32);
        for (uint32_t element = 0; element != 32; ++element)
            records.push_back(access(2 + element, 0, 8, AccessEncoding::Signed8));
        return records;
    }());
    std::vector<NodeSpec> nodes = {
        {NodeOp::LoadScalar, NodeType::Float, 0, 0, 0xffff, 0xffff, 0xffff, dm},
        {NodeOp::LoadBits, NodeType::Signed, 0, 0, 0xffff, 0xffff, 0xffff, qm},
        {NodeOp::CastFloat, NodeType::Float, 0xff, 0, 1, 0xffff, 0xffff, 0},
        {NodeOp::Mul, NodeType::Float, 0xff, 0, 0, 2, 0xffff, 0},
    };
    return builder.finish(
        {{CodecCertificatePlaneRole::Values,
          CodecCertificateScalar::PackedUnsigned,
          CodecCertificateStorageScalar::Unsigned8,
          kBitsLeastSignificantFirst, kLittleEndian, 8, 32, 34, 32, 0,
          34}},
        std::move(nodes), {}, 3);
}

std::vector<uint8_t> build_q6() {
    Builder builder(256, 210, gguf_physical(210));
    std::vector<AccessRecord> d = repeated(
        1, access(0, 0, 16, AccessEncoding::Binary16));
    std::vector<AccessRecord> ql, qh, scales;
    ql.reserve(256); qh.reserve(256); scales.reserve(256);
    for (uint32_t element = 0; element < 256; ++element) {
        const uint32_t segment = element / 128;
        const uint32_t within = element % 128;
        const uint32_t quarter = within / 32;
        const uint32_t lane = within % 32;
        ql.push_back(access(2 + segment * 64 + (quarter / 2) * 32 + lane,
                            static_cast<uint8_t>((quarter % 2) * 4), 4,
                            AccessEncoding::Unsigned8));
        qh.push_back(access(2 + 128 + segment * 32 + lane,
                            static_cast<uint8_t>(quarter * 2), 2,
                            AccessEncoding::Unsigned8, 4));
        scales.push_back(access(194 + element / 16, 0, 8,
                                AccessEncoding::Signed8));
    }
    const uint8_t dm = builder.add_map(0, std::move(d));
    const uint8_t qlm = builder.add_map(0, std::move(ql));
    const uint8_t qhm = builder.add_map(0, std::move(qh));
    const uint8_t sm = builder.add_map(0, std::move(scales));
    std::vector<uint32_t> constants = {std::bit_cast<uint32_t>(-32.0f)};
    std::vector<NodeSpec> nodes = {
        {NodeOp::LoadScalar, NodeType::Float, 0, 0, 0xffff, 0xffff, 0xffff,
         dm},
        {NodeOp::LoadBits, NodeType::Unsigned, 0, 0, 0xffff, 0xffff, 0xffff,
         qlm},
        {NodeOp::LoadBits, NodeType::Unsigned, 0, 0, 0xffff, 0xffff, 0xffff,
         qhm},
        {NodeOp::LoadBits, NodeType::Signed, 0, 0, 0xffff, 0xffff, 0xffff,
         sm},
        {NodeOp::CastFloat, NodeType::Float, 0xff, 0, 1, 0xffff, 0xffff, 0},
        {NodeOp::CastFloat, NodeType::Float, 0xff, 0, 2, 0xffff, 0xffff, 0},
        {NodeOp::CastFloat, NodeType::Float, 0xff, 0, 3, 0xffff, 0xffff, 0},
        {NodeOp::Add, NodeType::Float, 0xff, 0, 4, 5, 0xffff, 0},
        {NodeOp::Constant, NodeType::Float, 0xff, 0, 0xffff, 0xffff, 0xffff,
         0},
        {NodeOp::Add, NodeType::Float, 0xff, 0, 7, 8, 0xffff, 0},
        {NodeOp::Mul, NodeType::Float, 0xff, 0, 9, 6, 0xffff, 0},
        {NodeOp::Mul, NodeType::Float, 0xff, 0, 10, 0, 0xffff, 0},
    };
    return builder.finish(
        {{CodecCertificatePlaneRole::Values,
          CodecCertificateScalar::PackedUnsigned,
          CodecCertificateStorageScalar::Unsigned8,
          kBitsLeastSignificantFirst, kLittleEndian, 8, 256, 210, 32, 0,
          210}},
        std::move(nodes), std::move(constants), 6);
}

std::vector<uint8_t> build_grouped() {
    Builder builder(256, 64, grouped_physical());
    std::vector<AccessRecord> values, scales, biases;
    values.reserve(256); scales.reserve(256); biases.reserve(256);
    for (uint32_t element = 0; element < 256; ++element) {
        values.push_back(access((element / 16) * 4,
                                static_cast<uint8_t>((element % 16) * 2), 2,
                                AccessEncoding::Unsigned32));
        scales.push_back(access(0, 0, 16, AccessEncoding::Binary16));
        biases.push_back(access(0, 0, 16, AccessEncoding::Binary16));
    }
    const uint8_t vm = builder.add_map(0, std::move(values));
    const uint8_t sm = builder.add_map(1, std::move(scales));
    const uint8_t bm = builder.add_map(2, std::move(biases));
    std::vector<NodeSpec> nodes = {
        {NodeOp::LoadBits, NodeType::Unsigned, 0, 0, 0xffff, 0xffff, 0xffff,
         vm},
        {NodeOp::LoadScalar, NodeType::Float, 1, 0, 0xffff, 0xffff, 0xffff,
         sm},
        {NodeOp::LoadScalar, NodeType::Float, 2, 0, 0xffff, 0xffff, 0xffff,
         bm},
        {NodeOp::CastFloat, NodeType::Float, 0xff, 0, 0, 0xffff, 0xffff, 0},
        {NodeOp::Mul, NodeType::Float, 0xff, 0, 1, 3, 0xffff, 0},
        {NodeOp::Add, NodeType::Float, 0xff, 0, 4, 2, 0xffff, 0},
    };
    return builder.finish(
        {{CodecCertificatePlaneRole::Values,
          CodecCertificateScalar::PackedUnsigned,
          CodecCertificateStorageScalar::Unsigned32,
          kBitsLeastSignificantFirst, kLittleEndian, 2, 256, 64, 128, 0,
          64},
         {CodecCertificatePlaneRole::Scales,
          CodecCertificateScalar::Binary16,
          CodecCertificateStorageScalar::Binary16, 0, kLittleEndian, 16, 256,
          2, 2, 0, 2},
         {CodecCertificatePlaneRole::Biases,
          CodecCertificateScalar::Binary16,
          CodecCertificateStorageScalar::Binary16, 0, kLittleEndian, 16, 256,
          2, 2, 0, 2}},
        std::move(nodes), {}, 4);
}

bool checked_add(uint64_t a, uint64_t b, uint64_t* result) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    *result = a + b;
    return true;
}
bool checked_mul(uint64_t a, uint64_t b, uint64_t* result) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    *result = a * b;
    return true;
}
bool valid_encoding(AccessEncoding encoding) {
    return static_cast<uint8_t>(encoding) >= 1 &&
           static_cast<uint8_t>(encoding) <= 6;
}
bool valid_record(const AccessRecord& record) {
    if (record.flags & ~kAccessZero) return false;
    if (record.flags)
        return record.width_bits == 0 && record.bit_offset == 0 &&
               record.value_shift == 0 &&
               record.encoding == AccessEncoding::Unsigned8;
    if (record.width_bits == 0 || record.width_bits > 32 ||
        record.value_shift > 31 ||
        record.width_bits + record.value_shift > 32)
        return false;
    switch (record.encoding) {
    case AccessEncoding::Binary16:
        return record.bit_offset == 0 && record.width_bits == 16 &&
               record.value_shift == 0;
    case AccessEncoding::Binary32:
        return record.bit_offset == 0 && record.width_bits == 32 &&
               record.value_shift == 0;
    case AccessEncoding::Unsigned8:
        return record.bit_offset < 8;
    case AccessEncoding::Unsigned16:
        return record.bit_offset < 16 && record.width_bits <= 16;
    case AccessEncoding::Unsigned32:
        return record.bit_offset < 32;
    case AccessEncoding::Signed8:
        return record.bit_offset == 0 && record.width_bits <= 8;
    }
    return false;
}
uint32_t encoded_bytes(AccessEncoding encoding, uint8_t width,
                       uint8_t bit) {
    if (encoding == AccessEncoding::Binary16) return 2;
    if (encoding == AccessEncoding::Binary32) return 4;
    return (static_cast<uint32_t>(bit) + width + 7u) / 8u;
}
bool valid_storage(CodecCertificateStorageScalar scalar, uint16_t width) {
    switch (scalar) {
    case CodecCertificateStorageScalar::Binary16: return width == 16;
    case CodecCertificateStorageScalar::Binary32: return width == 32;
    case CodecCertificateStorageScalar::Unsigned8: return width <= 8;
    case CodecCertificateStorageScalar::Unsigned32: return width <= 32;
    // The current ScalarType ABI has no U16 or I8 storage representation.
    // Keep the wire enum values reserved, but reject them until the ABI can
    // carry their identity and binding semantics explicitly.
    case CodecCertificateStorageScalar::Unsigned16:
    case CodecCertificateStorageScalar::Signed8: return false;
    }
    return false;
}
bool access_storage_compatible(
    AccessEncoding encoding, CodecCertificateStorageScalar storage) {
    switch (encoding) {
    case AccessEncoding::Binary16:
        return storage == CodecCertificateStorageScalar::Binary16 ||
               storage == CodecCertificateStorageScalar::Unsigned8;
    case AccessEncoding::Binary32:
        return storage == CodecCertificateStorageScalar::Binary32 ||
               storage == CodecCertificateStorageScalar::Unsigned8;
    case AccessEncoding::Unsigned8:
        return storage == CodecCertificateStorageScalar::Unsigned8;
    case AccessEncoding::Unsigned16:
        return false;
    case AccessEncoding::Unsigned32:
        return storage == CodecCertificateStorageScalar::Unsigned32;
    case AccessEncoding::Signed8:
        // Signed byte values are encoded in an unsigned byte plane in the
        // current ABI (for example, Q6 scales).
        return storage == CodecCertificateStorageScalar::Unsigned8;
    }
    return false;
}
bool valid_logical(CodecCertificateScalar scalar, uint16_t width) {
    if (scalar == CodecCertificateScalar::Binary16) return width == 16;
    if (scalar == CodecCertificateScalar::Binary32) return width == 32;
    return width > 0 && width <= 32;
}
bool valid_op(NodeOp op) {
    switch (op) {
    case NodeOp::LoadScalar: case NodeOp::LoadBits: case NodeOp::CastFloat:
    case NodeOp::Add: case NodeOp::Sub: case NodeOp::Mul: case NodeOp::Fma:
    case NodeOp::Neg: case NodeOp::Constant: return true;
    }
    return false;
}

std::optional<Parsed> parse_internal(std::span<const uint8_t> bytes,
                                     CodecCertificateError* error) {
    if (bytes.size() > kMaximumBytes) {
        *error = CodecCertificateError::TooLarge;
        return std::nullopt;
    }
    Reader reader(bytes);
    Parsed parsed;
    uint16_t abi = 0, version = 0, flags = 0, reserved = 0;
    uint8_t output = 0, special = 0, plane_count = 0, depth = 0;
    uint16_t node_count = 0;
    uint8_t constant_count = 0, reserved_byte = 0;
    if (!reader.magic() || !reader.u16(&abi) || !reader.u16(&version) ||
        !reader.u16(&flags) || !reader.u8(&output) ||
        !reader.u8(&special) || !reader.u16(&reserved) ||
        !reader.u32(&parsed.summary.unit_elements) ||
        !reader.u32(&parsed.summary.unit_bytes) ||
        !reader.u32(&parsed.summary.maximum_units) ||
        !reader.u8(&plane_count) || !reader.u8(&depth) ||
        !reader.u16(&node_count) || !reader.u8(&constant_count) ||
        !reader.u8(&reserved_byte)) {
        *error = CodecCertificateError::NonCanonical;
        return std::nullopt;
    }
    if (abi != kAbiVersion || version != kCertificateVersion) {
        *error = CodecCertificateError::UnsupportedVersion;
        return std::nullopt;
    }
    if (output != static_cast<uint8_t>(CodecCertificateScalar::Binary32) ||
        special != static_cast<uint8_t>(
                        CodecCertificateSpecialValuePolicy::PreserveIeee) ||
        flags != kFlags || reserved != 0 || reserved_byte != 0) {
        *error = CodecCertificateError::NonCanonical;
        return std::nullopt;
    }
    if (!reader.u8(&parsed.summary.physical_layout_kind) ||
        !reader.u8(&parsed.summary.physical_layout_packing) ||
        !reader.u8(&parsed.summary.physical_layout_block_rank) ||
        !reader.u8(&reserved_byte) ||
        !reader.u32(&parsed.summary.physical_layout_block_elements) ||
        !reader.u32(&parsed.summary.physical_layout_block_bytes) ||
        !reader.u8(&parsed.summary.physical_quantization_kind) ||
        !reader.u8(&reserved_byte) || !reader.u16(&reserved) ||
        !reader.u32(&parsed.summary.physical_quantization_block_elements) ||
        !reader.u32(&parsed.summary.physical_quantization_block_bytes) ||
        !reader.u32(&parsed.summary.physical_quantization_group_size) ||
        !reader.u32(&parsed.summary.physical_quantization_required_plane_mask) ||
        reserved_byte != 0 || reserved != 0) {
        *error = CodecCertificateError::NonCanonical;
        return std::nullopt;
    }
    if (plane_count == 0 || plane_count > kMaximumPlanes || node_count == 0 ||
        node_count > kMaximumNodes || constant_count > kMaximumConstants ||
        depth == 0 || depth > kMaximumDepth ||
        parsed.summary.unit_elements == 0 ||
        parsed.summary.unit_elements > 65536 ||
        parsed.summary.unit_bytes == 0 ||
        parsed.summary.unit_bytes > 1024 * 1024 ||
        parsed.summary.maximum_units == 0 ||
        parsed.summary.maximum_units > kMaximumUnits) {
        *error = CodecCertificateError::ResourceLimit;
        return std::nullopt;
    }
    parsed.summary.abi_version = abi;
    parsed.summary.certificate_version = version;
    parsed.summary.output_scalar = CodecCertificateScalar::Binary32;
    parsed.summary.special_value_policy =
        CodecCertificateSpecialValuePolicy::PreserveIeee;
    parsed.summary.plane_count = plane_count;
    parsed.summary.node_count = node_count;
    parsed.summary.constant_count = constant_count;
    parsed.summary.expression_depth = depth;
    parsed.summary.rank_independent = true;
    try {
        parsed.planes.reserve(plane_count);
        for (uint8_t index = 0; index < plane_count; ++index) {
            PlaneSpec plane;
            uint8_t role = 0, scalar = 0, storage = 0;
            if (!reader.u8(&role) || !reader.u8(&scalar) ||
                !reader.u8(&storage) || !reader.u8(&plane.bit_order) ||
                !reader.u8(&plane.byte_order) ||
                !reader.u16(&plane.width_bits) ||
                !reader.u32(&plane.elements_per_unit) ||
                !reader.u32(&plane.bytes_per_unit) ||
                !reader.u32(&plane.alignment) ||
                !reader.u32(&plane.base) || !reader.u32(&plane.stride)) {
                *error = CodecCertificateError::NonCanonical;
                return std::nullopt;
            }
            plane.role = static_cast<CodecCertificatePlaneRole>(role);
            plane.scalar = static_cast<CodecCertificateScalar>(scalar);
            plane.storage_scalar =
                static_cast<CodecCertificateStorageScalar>(storage);
            if (role < 1 || role > 4 || scalar < 1 || scalar > 3 ||
                storage < 1 || storage > 6 ||
                !valid_logical(plane.scalar, plane.width_bits) ||
                !valid_storage(plane.storage_scalar, plane.width_bits) ||
                plane.elements_per_unit != parsed.summary.unit_elements ||
                plane.bytes_per_unit == 0 || plane.alignment == 0 ||
                (plane.alignment & (plane.alignment - 1)) != 0 ||
                plane.alignment > 4096 || plane.stride < plane.bytes_per_unit ||
                plane.base != 0 || plane.byte_order != kLittleEndian ||
                ((plane.scalar == CodecCertificateScalar::PackedUnsigned)
                     ? plane.bit_order != kBitsLeastSignificantFirst
                     : plane.bit_order != 0)) {
                *error = CodecCertificateError::InvalidPlane;
                return std::nullopt;
            }
            if (std::any_of(parsed.planes.begin(), parsed.planes.end(),
                            [&](const PlaneSpec& item) {
                                return item.role == plane.role;
                            })) {
                *error = CodecCertificateError::InvalidPlane;
                return std::nullopt;
            }
            parsed.planes.push_back(plane);
        }
        if (parsed.planes.front().role != CodecCertificatePlaneRole::Values) {
            *error = CodecCertificateError::InvalidPlane;
            return std::nullopt;
        }
        parsed.nodes.reserve(node_count);
        std::vector<uint8_t> node_depth;
        std::vector<uint16_t> use_count(node_count, 0);
        uint8_t observed_depth = 0;
        for (uint16_t index = 0; index < node_count; ++index) {
            NodeSpec node;
            uint8_t op = 0, type = 0;
            uint16_t node_reserved = 0;
            if (!reader.u8(&op) || !reader.u8(&type) ||
                !reader.u8(&node.plane) || !reader.u8(&node.flags) ||
                !reader.u16(&node.arg0) || !reader.u16(&node.arg1) ||
                !reader.u16(&node.arg2) || !reader.u32(&node.immediate) ||
                !reader.u16(&node_reserved)) {
                *error = CodecCertificateError::NonCanonical;
                return std::nullopt;
            }
            node.op = static_cast<NodeOp>(op);
            node.type = static_cast<NodeType>(type);
            if (!valid_op(node.op) || node.flags != 0 ||
                node_reserved != 0 ||
                (node.type != NodeType::Unsigned &&
                 node.type != NodeType::Float &&
                 node.type != NodeType::Signed)) {
                *error = CodecCertificateError::InvalidNodeType;
                return std::nullopt;
            }
            const bool type_valid =
                (node.op == NodeOp::LoadBits &&
                 (node.type == NodeType::Unsigned ||
                  node.type == NodeType::Signed)) ||
                ((node.op == NodeOp::LoadScalar ||
                  node.op == NodeOp::CastFloat ||
                  node.op == NodeOp::Add || node.op == NodeOp::Sub ||
                  node.op == NodeOp::Mul || node.op == NodeOp::Fma ||
                  node.op == NodeOp::Neg || node.op == NodeOp::Constant) &&
                 node.type == NodeType::Float);
            if (!type_valid) {
                *error = CodecCertificateError::InvalidNodeType;
                return std::nullopt;
            }
            const bool binary = node.op == NodeOp::Add ||
                                node.op == NodeOp::Sub ||
                                node.op == NodeOp::Mul;
            const bool ternary = node.op == NodeOp::Fma;
            const bool unary = node.op == NodeOp::CastFloat ||
                               node.op == NodeOp::Neg;
            const auto valid_arg = [index](uint16_t value, bool required) {
                return required ? value < index : value == 0xffff;
            };
            if (!valid_arg(node.arg0, binary || ternary || unary) ||
                !valid_arg(node.arg1, binary || ternary) ||
                !valid_arg(node.arg2, ternary) ||
                (!binary && !ternary && !unary &&
                 (node.arg0 != 0xffff || node.arg1 != 0xffff ||
                  node.arg2 != 0xffff))) {
                *error = CodecCertificateError::InvalidExpression;
                return std::nullopt;
            }
            const auto float_argument = [&](uint16_t argument) {
                return argument == 0xffff ||
                       parsed.nodes[argument].type == NodeType::Float;
            };
            if ((binary || ternary || node.op == NodeOp::Neg) &&
                (!float_argument(node.arg0) ||
                 !float_argument(node.arg1) ||
                 !float_argument(node.arg2))) {
                *error = CodecCertificateError::InvalidNodeType;
                return std::nullopt;
            }
            if (node.op == NodeOp::LoadScalar ||
                node.op == NodeOp::LoadBits) {
                if (node.plane >= plane_count || node.immediate >= kMaximumMaps) {
                    *error = CodecCertificateError::InvalidExpression;
                    return std::nullopt;
                }
            } else if (node.plane != 0xff ||
                       (node.op != NodeOp::Constant && node.immediate != 0)) {
                *error = CodecCertificateError::InvalidExpression;
                return std::nullopt;
            }
            uint8_t current_depth = 1;
            for (uint16_t argument : {node.arg0, node.arg1, node.arg2}) {
                if (argument == 0xffff) continue;
                ++use_count[argument];
                current_depth = std::max(
                    current_depth,
                    static_cast<uint8_t>(node_depth[argument] + 1));
            }
            if (current_depth > kMaximumDepth) {
                *error = CodecCertificateError::ResourceLimit;
                return std::nullopt;
            }
            observed_depth = std::max(observed_depth, current_depth);
            node_depth.push_back(current_depth);
            parsed.nodes.push_back(node);
        }
        if (observed_depth != depth || parsed.nodes.back().type != NodeType::Float ||
            use_count.back() != 0 ||
            std::any_of(use_count.begin(), use_count.end() - 1,
                        [](uint16_t value) { return value == 0; })) {
            *error = CodecCertificateError::InvalidTopology;
            return std::nullopt;
        }
        parsed.constants.reserve(constant_count);
        for (uint8_t index = 0; index < constant_count; ++index) {
            uint32_t value = 0;
            if (!reader.u32(&value)) {
                *error = CodecCertificateError::NonCanonical;
                return std::nullopt;
            }
            parsed.constants.push_back(value);
        }
        uint32_t extension = 0, record_count = 0;
        uint16_t map_count = 0, map_reserved = 0;
        if (!reader.u32(&extension) || extension != 2 ||
            !reader.u16(&map_count) || !reader.u16(&map_reserved) ||
            !reader.u32(&record_count) || map_reserved != 0 ||
            map_count == 0 || map_count > kMaximumMaps ||
            record_count == 0 || record_count > kMaximumRecords) {
            *error = CodecCertificateError::InvalidAccessMap;
            return std::nullopt;
        }
        parsed.maps.reserve(map_count);
        uint32_t previous = 0;
        for (uint16_t index = 0; index < map_count; ++index) {
            AccessMap map;
            uint8_t r0 = 0, r1 = 0, r2 = 0;
            if (!reader.u8(&map.plane) || !reader.u8(&r0) ||
                !reader.u8(&r1) || !reader.u8(&r2) ||
                !reader.u32(&map.first) || !reader.u32(&map.count) ||
                r0 != 0 || r1 != 0 || r2 != 0 ||
                map.plane >= plane_count || map.count == 0 ||
                (map.count != 1 && map.count != parsed.summary.unit_elements) ||
                map.first != previous || map.first > record_count ||
                map.count > record_count - map.first) {
                *error = CodecCertificateError::InvalidAccessMap;
                return std::nullopt;
            }
            previous = map.first + map.count;
            parsed.maps.push_back(map);
        }
        if (previous != record_count) {
            *error = CodecCertificateError::InvalidAccessMap;
            return std::nullopt;
        }
        parsed.records.reserve(record_count);
        for (uint32_t index = 0; index < record_count; ++index) {
            AccessRecord record;
            uint8_t encoding = 0, r0 = 0, r1 = 0, r2 = 0;
            if (!reader.u32(&record.byte_offset) ||
                !reader.u8(&record.bit_offset) ||
                !reader.u8(&record.width_bits) || !reader.u8(&encoding) ||
                !reader.u8(&record.flags) ||
                !reader.u8(&record.value_shift) || !reader.u8(&r0) ||
                !reader.u8(&r1) || !reader.u8(&r2)) {
                *error = CodecCertificateError::NonCanonical;
                return std::nullopt;
            }
            record.encoding = static_cast<AccessEncoding>(encoding);
            if (!valid_encoding(record.encoding) || !valid_record(record) ||
                r0 != 0 || r1 != 0 || r2 != 0 ||
                record.bit_offset >=
                    (record.encoding == AccessEncoding::Unsigned32 ? 32u
                     : record.encoding == AccessEncoding::Unsigned16 ? 16u
                     : 8u) ||
                (record.flags && record.width_bits != 0) ||
                (!record.flags &&
                 (record.width_bits == 0 || record.width_bits > 32)) ||
                record.value_shift > 31 ||
                (!record.flags &&
                 record.width_bits + record.value_shift > 32)) {
                *error = CodecCertificateError::InvalidAccessMap;
                return std::nullopt;
            }
            parsed.records.push_back(record);
        }
        if (reader.remaining() != 0) {
            *error = CodecCertificateError::NonCanonical;
            return std::nullopt;
        }
        std::vector<bool> map_used(map_count, false);
        for (const NodeSpec& node : parsed.nodes) {
            if (node.op == NodeOp::LoadScalar ||
                node.op == NodeOp::LoadBits) {
                if (node.immediate >= map_count ||
                    parsed.maps[node.immediate].plane != node.plane) {
                    *error = CodecCertificateError::InvalidAccessMap;
                    return std::nullopt;
                }
                const AccessMap& map = parsed.maps[node.immediate];
                for (uint32_t index = map.first;
                     index < map.first + map.count; ++index) {
                    const AccessRecord& record = parsed.records[index];
                    if (record.flags) continue;
                    const bool scalar_encoding =
                        record.encoding == AccessEncoding::Binary16 ||
                        record.encoding == AccessEncoding::Binary32;
                    const bool unsigned_encoding =
                        record.encoding == AccessEncoding::Unsigned8 ||
                        record.encoding == AccessEncoding::Unsigned16 ||
                        record.encoding == AccessEncoding::Unsigned32;
                    const bool encoding_matches =
                        (node.op == NodeOp::LoadScalar && scalar_encoding) ||
                        (node.op == NodeOp::LoadBits &&
                         node.type == NodeType::Unsigned && unsigned_encoding) ||
                        (node.op == NodeOp::LoadBits &&
                         node.type == NodeType::Signed &&
                         record.encoding == AccessEncoding::Signed8);
                    if (!encoding_matches) {
                        *error = CodecCertificateError::InvalidAccessMap;
                        return std::nullopt;
                    }
                }
                map_used[node.immediate] = true;
            }
            if (node.op == NodeOp::Constant &&
                node.immediate >= parsed.constants.size()) {
                *error = CodecCertificateError::InvalidExpression;
                return std::nullopt;
            }
        }
        if (std::any_of(map_used.begin(), map_used.end(),
                        [](bool used) { return !used; })) {
            *error = CodecCertificateError::InvalidAccessMap;
            return std::nullopt;
        }
        for (const AccessMap& map : parsed.maps) {
            const PlaneSpec& plane = parsed.planes[map.plane];
            for (uint32_t index = map.first; index < map.first + map.count;
                 ++index) {
                const AccessRecord& record = parsed.records[index];
                if (record.flags) continue;
                const uint64_t end = static_cast<uint64_t>(record.byte_offset) +
                                     encoded_bytes(record.encoding,
                                                   record.width_bits,
                                                   record.bit_offset);
                if (end > plane.bytes_per_unit ||
                    record.byte_offset >= plane.bytes_per_unit) {
                    *error = CodecCertificateError::AccessOutOfBounds;
                    return std::nullopt;
                }
                if (!access_storage_compatible(record.encoding,
                                                plane.storage_scalar)) {
                    *error = CodecCertificateError::InvalidAccessMap;
                    return std::nullopt;
                }
            }
        }
        parsed.summary.source_scalar = parsed.planes.front().scalar;
        return parsed;
    } catch (const std::bad_alloc&) {
        *error = CodecCertificateError::TooLarge;
        return std::nullopt;
    }
}

uint16_t load_u16(const uint8_t* bytes) {
    return static_cast<uint16_t>(bytes[0]) |
           (static_cast<uint16_t>(bytes[1]) << 8u);
}
uint32_t load_u32(const uint8_t* bytes) {
    return static_cast<uint32_t>(bytes[0]) |
           (static_cast<uint32_t>(bytes[1]) << 8u) |
           (static_cast<uint32_t>(bytes[2]) << 16u) |
           (static_cast<uint32_t>(bytes[3]) << 24u);
}
float f16_to_f32(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value >> 15u) & 1u;
    int32_t exponent = static_cast<int32_t>((value >> 10u) & 0x1fu);
    uint32_t mantissa = value & 0x3ffu;
    uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign << 31u;
        } else {
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1u;
                --exponent;
            }
            ++exponent;
            result = (sign << 31u) |
                     (static_cast<uint32_t>(exponent + 112) << 23u) |
                     ((mantissa & 0x3ffu) << 13u);
        }
    } else if (exponent == 31) {
        result = (sign << 31u) | (0xffu << 23u) | (mantissa << 13u);
    } else {
        result = (sign << 31u) |
                 (static_cast<uint32_t>(exponent + 112) << 23u) |
                 (mantissa << 13u);
    }
    return std::bit_cast<float>(result);
}
float rounded_add(float a, float b) {
    volatile float result = a + b;
    return result;
}
float rounded_sub(float a, float b) {
    volatile float result = a - b;
    return result;
}
float rounded_mul(float a, float b) {
    volatile float result = a * b;
    return result;
}

CodecCertificateError validate_binding(const Parsed& parsed,
                                       const CodecCertificateBinding& binding,
                                       uint64_t max_elements,
                                       uint64_t* count) {
    if (binding.planes.size() != parsed.planes.size())
        return CodecCertificateError::InvalidPlaneBinding;
    if (binding.unit_count == 0 ||
        binding.unit_count > parsed.summary.maximum_units)
        return CodecCertificateError::InvalidUnitCount;
    uint64_t total = 0;
    if (!checked_mul(binding.unit_count, parsed.summary.unit_elements, &total))
        return CodecCertificateError::ArithmeticOverflow;
    if (max_elements != 0 && total > max_elements)
        return CodecCertificateError::DecodeLimit;
    for (size_t index = 0; index < parsed.planes.size(); ++index) {
        const PlaneSpec& spec = parsed.planes[index];
        const CodecCertificatePlaneBinding& plane = binding.planes[index];
        if (plane.offset > plane.storage.size())
            return CodecCertificateError::InvalidOffset;
        if (plane.offset % spec.alignment != 0)
            return CodecCertificateError::InvalidOffset;
        if (plane.length > plane.storage.size() - plane.offset)
            return CodecCertificateError::InvalidLength;
        if (plane.stride < spec.stride ||
            plane.stride < spec.bytes_per_unit)
            return CodecCertificateError::InvalidStride;
        uint64_t displacement = 0;
        if (!checked_mul(binding.unit_count - 1, plane.stride, &displacement))
            return CodecCertificateError::ArithmeticOverflow;
        uint64_t access = 0;
        if (!checked_add(displacement, spec.bytes_per_unit, &access))
            return CodecCertificateError::ArithmeticOverflow;
        if (access > plane.length)
            return CodecCertificateError::AccessOutOfBounds;
        uint64_t end = 0;
        if (!checked_add(plane.offset, access, &end) ||
            end > plane.storage.size())
            return CodecCertificateError::AccessOutOfBounds;
        for (const AccessMap& map : parsed.maps) {
            if (map.plane != index) continue;
            uint32_t max_end = 0;
            for (uint32_t r = map.first; r < map.first + map.count; ++r) {
                const AccessRecord& record = parsed.records[r];
                if (record.flags) continue;
                max_end = std::max(
                    max_end, record.byte_offset +
                                 encoded_bytes(record.encoding,
                                               record.width_bits,
                                               record.bit_offset));
            }
            if (max_end > spec.bytes_per_unit)
                return CodecCertificateError::AccessOutOfBounds;
        }
    }
    *count = total;
    return CodecCertificateError::None;
}
uint32_t record_index(const AccessMap& map, uint32_t element) {
    return map.first + (map.count == 1 ? 0 : element);
}
float load_record(const AccessRecord& record,
                  const CodecCertificatePlaneBinding& binding, uint64_t unit,
                  uint64_t stride) {
    if (record.flags & kAccessZero) return 0.0f;
    const uint8_t* bytes = binding.storage.data() + binding.offset +
                           static_cast<size_t>(unit * stride +
                                               record.byte_offset);
    switch (record.encoding) {
    case AccessEncoding::Binary16:
        return f16_to_f32(load_u16(bytes));
    case AccessEncoding::Binary32:
        return std::bit_cast<float>(load_u32(bytes));
    case AccessEncoding::Unsigned8:
    case AccessEncoding::Unsigned16:
    case AccessEncoding::Unsigned32:
    case AccessEncoding::Signed8: {
        const uint32_t byte_count =
            encoded_bytes(record.encoding, record.width_bits,
                          record.bit_offset);
        uint64_t value = 0;
        for (uint32_t index = 0; index < byte_count; ++index)
            value |= static_cast<uint64_t>(bytes[index]) << (index * 8u);
        value >>= record.bit_offset;
        const uint64_t mask = record.width_bits == 32
            ? 0xffffffffull : ((1ull << record.width_bits) - 1ull);
        value &= mask;
        if (record.encoding == AccessEncoding::Signed8) {
            const int64_t sign_bit = 1ll << (record.width_bits - 1u);
            int64_t signed_value = static_cast<int64_t>(value);
            if ((signed_value & sign_bit) != 0)
                signed_value |= ~static_cast<int64_t>(mask);
            return static_cast<float>(signed_value *
                                      (int64_t{1} << record.value_shift));
        }
        return static_cast<float>(value << record.value_shift);
    }
    }
    return 0.0f;
}

CodecCertificateDecodeResult decode_parsed(
    const Parsed& parsed, const CodecCertificateBinding& binding,
    uint64_t max_elements) {
    uint64_t count = 0;
    const CodecCertificateError validation =
        validate_binding(parsed, binding, max_elements, &count);
    if (validation != CodecCertificateError::None) return validation;
    try {
        std::vector<float> output(static_cast<size_t>(count));
        std::vector<float> values(parsed.nodes.size());
        for (uint64_t unit = 0; unit < binding.unit_count; ++unit) {
            for (uint32_t element = 0; element < parsed.summary.unit_elements;
                 ++element) {
                for (size_t index = 0; index < parsed.nodes.size(); ++index) {
                    const NodeSpec& node = parsed.nodes[index];
                    switch (node.op) {
                    case NodeOp::LoadScalar:
                    case NodeOp::LoadBits: {
                        const AccessMap& map = parsed.maps[node.immediate];
                        const AccessRecord& record =
                            parsed.records[record_index(map, element)];
                        values[index] = load_record(
                            record, binding.planes[node.plane], unit,
                            binding.planes[node.plane].stride);
                        break;
                    }
                    case NodeOp::CastFloat:
                        values[index] = values[node.arg0];
                        break;
                    case NodeOp::Constant:
                        values[index] = std::bit_cast<float>(
                            parsed.constants[node.immediate]);
                        break;
                    case NodeOp::Add:
                        values[index] = rounded_add(values[node.arg0],
                                                    values[node.arg1]);
                        break;
                    case NodeOp::Sub:
                        values[index] = rounded_sub(values[node.arg0],
                                                    values[node.arg1]);
                        break;
                    case NodeOp::Mul:
                        values[index] = rounded_mul(values[node.arg0],
                                                    values[node.arg1]);
                        break;
                    case NodeOp::Fma:
                        values[index] = std::fma(values[node.arg0],
                                                 values[node.arg1],
                                                 values[node.arg2]);
                        break;
                    case NodeOp::Neg:
                        values[index] = -values[node.arg0];
                        break;
                    }
                }
                output[static_cast<size_t>(
                    unit * parsed.summary.unit_elements + element)] =
                    values.back();
            }
        }
        return output;
    } catch (const std::bad_alloc&) {
        return CodecCertificateError::TooLarge;
    }
}
ScalarType storage_type(CodecCertificateStorageScalar scalar) {
    switch (scalar) {
    case CodecCertificateStorageScalar::Binary16: return ScalarType::F16;
    case CodecCertificateStorageScalar::Binary32: return ScalarType::F32;
    case CodecCertificateStorageScalar::Unsigned8: return ScalarType::U8;
    case CodecCertificateStorageScalar::Unsigned32: return ScalarType::U32;
    default: return static_cast<ScalarType>(0);
    }
}
PlaneKind plane_kind(CodecCertificatePlaneRole role) {
    switch (role) {
    case CodecCertificatePlaneRole::Values: return PlaneKind::Values;
    case CodecCertificatePlaneRole::Scales: return PlaneKind::Scales;
    case CodecCertificatePlaneRole::Biases: return PlaneKind::Biases;
    case CodecCertificatePlaneRole::Indexes: return PlaneKind::Indexes;
    }
    return static_cast<PlaneKind>(0);
}

} // namespace

class CodecCertificateFactory {
public:
    static CodecCertificate make(CodecCertificateIdentity identity,
                                 CodecCertificateSummary summary,
                                 std::vector<uint8_t> bytes,
                                 std::vector<CodecCertificatePlaneSummary> planes,
                                 std::vector<CodecCertificateNodeSummary> nodes,
                                 std::vector<uint32_t> constants,
                                 std::vector<CodecCertificateAccessMapSummary>
                                     maps,
                                 std::vector<CodecCertificateAccessSummary>
                                     records) {
        return CodecCertificate(std::move(identity), std::move(summary),
                                std::move(bytes), std::move(planes),
                                std::move(nodes), std::move(constants),
                                std::move(maps),
                                std::move(records));
    }
};

CodecCertificateDigest codec_certificate_digest(std::span<const uint8_t> bytes) {
    CodecCertificateDigest digest{};
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    constexpr size_t chunk = 1024 * 1024;
    for (size_t offset = 0; offset < bytes.size(); offset += chunk) {
        const size_t count = std::min(chunk, bytes.size() - offset);
        CC_SHA256_Update(&context, bytes.data() + offset,
                         static_cast<CC_LONG>(count));
    }
    CC_SHA256_Final(digest.data(), &context);
    return digest;
}
CodecCertificateParseResult parse_codec_certificate(
    std::span<const uint8_t> bytes) {
    try {
        CodecCertificateError error = CodecCertificateError::None;
        const auto parsed = parse_internal(bytes, &error);
        if (!parsed) return error;
        CodecCertificateIdentity identity;
        identity.abi_version = parsed->summary.abi_version;
        identity.digest = codec_certificate_digest(bytes);
        std::vector<CodecCertificatePlaneSummary> planes;
        planes.reserve(parsed->planes.size());
        for (const PlaneSpec& plane : parsed->planes)
            planes.push_back({plane.role, plane.scalar, plane.storage_scalar,
                              plane.bit_order, plane.byte_order,
                              plane.width_bits, plane.elements_per_unit,
                              plane.bytes_per_unit, plane.alignment,
                              plane.base, plane.stride});
        std::vector<CodecCertificateNodeSummary> nodes;
        nodes.reserve(parsed->nodes.size());
        for (const NodeSpec& node : parsed->nodes)
            nodes.push_back({static_cast<CodecCertificateNodeOperation>(node.op),
                             static_cast<CodecCertificateNodeValueType>(node.type),
                             node.plane, node.flags, node.arg0, node.arg1,
                             node.arg2, node.immediate});
        std::vector<CodecCertificateAccessMapSummary> maps;
        maps.reserve(parsed->maps.size());
        for (const AccessMap& map : parsed->maps)
            maps.push_back({map.plane, map.first, map.count});
        std::vector<CodecCertificateAccessSummary> records;
        records.reserve(parsed->records.size());
        for (const AccessRecord& record : parsed->records)
            records.push_back({record.byte_offset, record.bit_offset,
                               record.width_bits,
                               static_cast<CodecCertificateAccessEncoding>(
                                   record.encoding),
                               record.flags, record.value_shift});
        return CodecCertificateFactory::make(
            identity, parsed->summary,
            std::vector<uint8_t>(bytes.begin(), bytes.end()),
            std::move(planes), std::move(nodes), std::move(parsed->constants),
            std::move(maps),
            std::move(records));
    } catch (const std::bad_alloc&) {
        return CodecCertificateError::TooLarge;
    }
}
CodecCertificateError CodecCertificate::validate(
    const CodecCertificateBinding& binding) const noexcept {
    try {
        CodecCertificateError error = CodecCertificateError::None;
        const auto parsed = parse_internal(canonical_bytes_, &error);
        if (!parsed) return error;
        if (codec_certificate_digest(canonical_bytes_) != identity_.digest)
            return CodecCertificateError::NonCanonical;
        uint64_t ignored = 0;
        return validate_binding(*parsed, binding, 0, &ignored);
    } catch (const std::bad_alloc&) {
        return CodecCertificateError::TooLarge;
    }
}
bool CodecCertificate::matches_physical_identity(
    const PhysicalCodecIdentity& identity) const noexcept {
    try {
        CodecCertificateError error = CodecCertificateError::None;
        if (!parse_internal(canonical_bytes_, &error) ||
            codec_certificate_digest(canonical_bytes_) != identity_.digest) {
            return false;
        }
    } catch (const std::bad_alloc&) {
        return false;
    }
    if (identity.identity_version != 1 ||
        identity.arithmetic_version != identity_.abi_version ||
        identity.arithmetic_digest != identity_.digest ||
        !std::all_of(identity.codebook_digest.begin(),
                     identity.codebook_digest.end(),
                     [](uint8_t value) { return value == 0; }) ||
        identity.layout.version != 1 || identity.quantization.version != 1 ||
        identity.planes.size() != plane_summaries_.size()) {
        return false;
    }
    const auto& summary = summary_;
    if (static_cast<uint8_t>(identity.layout.kind) !=
            summary.physical_layout_kind ||
        static_cast<uint8_t>(identity.layout.packing) !=
            summary.physical_layout_packing ||
        identity.layout.block_rank != summary.physical_layout_block_rank ||
        identity.layout.block_elements !=
            summary.physical_layout_block_elements ||
        identity.layout.block_bytes != summary.physical_layout_block_bytes ||
        static_cast<uint8_t>(identity.quantization.kind) !=
            summary.physical_quantization_kind ||
        identity.quantization.block_elements !=
            summary.physical_quantization_block_elements ||
        identity.quantization.block_bytes !=
            summary.physical_quantization_block_bytes ||
        identity.quantization.group_size !=
            summary.physical_quantization_group_size ||
        identity.quantization.required_plane_mask !=
            summary.physical_quantization_required_plane_mask) {
        return false;
    }
    if (identity.layout.kind == PhysicalLayoutKind::GgufBlocked) {
        // The blocked certificates in this registry use FP32 accumulation and
        // binary16 scales.  Authenticate these scalar fields too; block byte
        // size by itself is not a physical format identity.
        if (identity.quantization.accumulation_type != ScalarType::F32 ||
            identity.quantization.scale_type != ScalarType::F16 ||
            identity.quantization.bias_type != static_cast<ScalarType>(0))
            return false;
        const bool q4_k = identity.layout.block_elements == 256 &&
                          identity.layout.block_bytes == 144;
        if ((q4_k && identity.quantization.zero_type != ScalarType::F16) ||
            (!q4_k && identity.quantization.zero_type != static_cast<ScalarType>(0)))
            return false;
    }
    uint32_t required = 0;
    for (const CodecCertificatePlaneSummary& declared : plane_summaries_) {
        const PlaneKind kind = plane_kind(declared.role);
        if (static_cast<uint16_t>(kind) == 0) return false;
        required |= 1u << (static_cast<uint16_t>(kind) - 1u);
        const auto found = std::find_if(
            identity.planes.begin(), identity.planes.end(),
            [&](const PhysicalPlaneSchema& plane) {
                return plane.kind == kind;
            });
        if (found == identity.planes.end() ||
            found->storage_type != storage_type(declared.storage_scalar) ||
            found->logical_elements_covered != declared.elements_per_unit ||
            found->bytes_per_block != declared.bytes_per_unit ||
            found->flags != 0) {
            return false;
        }
    }
    const uint32_t expected_mask =
        summary.source_scalar == CodecCertificateScalar::PackedUnsigned
            ? required : 0u;
    return identity.quantization.required_plane_mask == expected_mask;
}
CodecCertificateDecodeResult CodecCertificate::decode(
    const CodecCertificateBinding& binding, uint64_t max_elements) const {
    try {
        CodecCertificateError error = CodecCertificateError::None;
        const auto parsed = parse_internal(canonical_bytes_, &error);
        if (!parsed) return error;
        if (codec_certificate_digest(canonical_bytes_) != identity_.digest)
            return CodecCertificateError::NonCanonical;
        if (std::fegetround() != FE_TONEAREST)
            return CodecCertificateError::UnsupportedEncoding;
        return decode_parsed(*parsed, binding, max_elements);
    } catch (const std::bad_alloc&) {
        return CodecCertificateError::TooLarge;
    }
}
std::vector<uint8_t> make_raw_f16_codec_certificate() {
    return build_raw_f16();
}
std::vector<uint8_t> make_raw_f32_codec_certificate() {
    return build_raw_f32();
}
std::vector<uint8_t> make_q4_k_codec_certificate() {
    return build_q4();
}
std::vector<uint8_t> make_q5_0_codec_certificate() {
    return build_q5_0();
}
std::vector<uint8_t> make_q6_k_codec_certificate() {
    return build_q6();
}
std::vector<uint8_t> make_q8_0_codec_certificate() {
    return build_q8_0();
}
std::vector<uint8_t> make_grouped_affine_u2_codec_certificate() {
    return build_grouped();
}

} // namespace Laplace
