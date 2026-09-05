#include "codec_certificate_physical_program.h"

#include <algorithm>
#include <functional>
#include <limits>

namespace Laplace {
namespace {
using Op = PhysicalOpcode;
using Type = PhysicalValueType;

struct Builder {
    PhysicalProgram program;
    std::vector<uint16_t> sources;

    uint32_t emit(Op op, Type type, uint32_t a = kNoPhysicalValue,
                  uint32_t b = kNoPhysicalValue, uint32_t c = kNoPhysicalValue,
                  uint64_t immediate = 0, uint16_t policy = kNoPhysicalPolicy) {
        PhysicalInstruction i;
        i.opcode = op;
        i.result_type = type;
        i.operands = {a, b, c};
        i.immediate = immediate;
        i.policy = policy;
        program.instructions.push_back(i);
        return static_cast<uint32_t>(program.instructions.size() - 1);
    }
    uint32_t constant(uint64_t n, Type type = Type::Index) {
        return emit(type == Type::Index ? Op::ConstIndex : Op::ConstU32,
                    type, kNoPhysicalValue, kNoPhysicalValue, kNoPhysicalValue, n);
    }
    uint32_t unary(Op op, Type type, uint32_t a, uint64_t immediate = 0,
                   uint16_t policy = kNoPhysicalPolicy) {
        return emit(op, type, a, kNoPhysicalValue, kNoPhysicalValue, immediate, policy);
    }
    uint32_t load(uint16_t plane, uint32_t address, uint8_t width) {
        const auto value = emit(Op::LoadBits, Type::U32, address);
        program.instructions[value].plane = plane;
        program.instructions[value].bit_width = width;
        return value;
    }
    uint32_t field(uint32_t control, uint8_t shift, uint32_t mask) {
        if (shift) control = unary(Op::U32ShiftRightConstant, Type::U32, control, shift);
        return emit(Op::U32And, Type::U32, control, constant(mask, Type::U32));
    }

    CodecCertificatePhysicalResult finish(uint64_t required_units) {
        // Discard unused helpers and record planes in the same operand-first
        // traversal used by the canonicalizer. Constant interning cannot change
        // the first occurrence of a plane.
        PhysicalProgram live;
        live.logical_rank = program.logical_rank;
        live.inline_bytes = program.inline_bytes;
        std::vector<uint32_t> values(program.instructions.size(), kNoPhysicalValue);
        std::vector<uint16_t> planes(program.planes.size(), kNoPhysicalPlane);
        std::vector<uint16_t> policies(program.policies.size(), kNoPhysicalPolicy);
        std::vector<uint16_t> live_sources;
        const std::function<uint32_t(uint32_t)> visit = [&](uint32_t old) {
            if (values[old] != kNoPhysicalValue) return values[old];
            auto i = program.instructions[old];
            for (auto& operand : i.operands)
                if (operand != kNoPhysicalValue) operand = visit(operand);
            if (i.plane != kNoPhysicalPlane) {
                auto& mapped = planes[i.plane];
                if (mapped == kNoPhysicalPlane) {
                    mapped = static_cast<uint16_t>(live.planes.size());
                    live.planes.push_back(program.planes[i.plane]);
                    live_sources.push_back(sources[i.plane]);
                }
                i.plane = mapped;
            }
            if (i.policy != kNoPhysicalPolicy) {
                auto& mapped = policies[i.policy];
                if (mapped == kNoPhysicalPolicy) {
                    mapped = static_cast<uint16_t>(live.policies.size());
                    live.policies.push_back(program.policies[i.policy]);
                }
                i.policy = mapped;
            }
            values[old] = static_cast<uint32_t>(live.instructions.size());
            live.instructions.push_back(i);
            return values[old];
        };
        live.result = visit(program.result);
        auto canonical = canonicalize_physical_program(std::move(live));
        if (auto* error = std::get_if<CompatibilityReport>(&canonical)) return *error;
        return TranslatedCodecCertificate{
            std::get<PhysicalProgram>(std::move(canonical)), required_units, std::move(live_sources)};
    }
};

CompatibilityReport error(const char* detail) {
    return compatibility_report(CompatibilityError::IR_CONSTRAINT_FAILED, detail);
}
} // namespace

CodecCertificatePhysicalResult translate_codec_certificate(
    const CodecCertificate& certificate, const LogicalTensorType& logical,
    std::span<const uint64_t> element_strides,
    std::span<const uint64_t> strides) {
    const auto& summary = certificate.summary();
    const auto& planes = certificate.plane_summaries();
    if (certificate.canonical_bytes().empty() || logical.element_type != ElementType::F32 ||
        logical.extents.size() > 8 || strides.size() != planes.size() ||
        element_strides.size() != logical.extents.size() || !summary.unit_elements)
        return error("certificate translation inputs are invalid");
    uint64_t elements = 1;
    for (auto extent : logical.extents) {
        if (!extent || elements > UINT64_MAX / extent)
            return error("certificate logical extent overflows");
        elements *= extent;
    }
    uint64_t maximum_index = 0;
    for (size_t axis = 0; axis < logical.extents.size(); ++axis) {
        const uint64_t extent = logical.extents[axis] - 1;
        if (extent && element_strides[axis] > (UINT64_MAX - maximum_index) / extent)
            return error("certificate logical element map overflows");
        maximum_index += extent * element_strides[axis];
    }
    if (maximum_index / summary.unit_elements == UINT64_MAX)
        return error("certificate mapped unit count overflows");
    const uint64_t required_units = maximum_index / summary.unit_elements + 1;
    for (size_t p = 0; p < planes.size(); ++p) {
        if (strides[p] < planes[p].stride || strides[p] < planes[p].bytes_per_unit ||
            strides[p] > UINT64_MAX / 8)
            return error("certificate plane stride is invalid");
    }

    Builder b;
    b.program.logical_rank = static_cast<uint8_t>(logical.extents.size());
    PhysicalNumericPolicy preserve;
    preserve.nan = PhysicalNanPolicy::PreserveIeee;
    auto fused = preserve;
    fused.contraction = PhysicalContractionPolicy::Fused;
    b.program.policies = {PhysicalNumericPolicy{}, preserve, fused};
    for (uint16_t p = 0; p < planes.size(); ++p) {
        b.program.planes.push_back({PhysicalPlaneStorage::External, 1});
        b.sources.push_back(p);
    }
    const uint16_t table_plane = static_cast<uint16_t>(planes.size());
    const auto& accesses = certificate.access_summaries();
    b.program.planes.push_back({PhysicalPlaneStorage::Inline, 1, 0, accesses.size() * 8});
    b.sources.push_back(kNoPhysicalPlane);
    for (const auto& map : certificate.access_map_summaries()) {
        const uint32_t bits = planes[map.plane].bytes_per_unit * 8;
        const uint32_t window = std::min(32u, bits);
        for (uint32_t j = 0; j < map.count; ++j) {
            const auto& access = accesses[map.first + j];
            const uint32_t start = access.flags ? 0 : access.byte_offset * 8 + access.bit_offset;
            const uint32_t address = std::min(start, bits - window);
            const uint32_t control = (start - address) | (uint32_t(access.width_bits) << 5) |
                (uint32_t(access.value_shift) << 11) |
                (uint32_t(access.encoding == CodecCertificateAccessEncoding::Binary32) << 16);
            for (auto word : {address, control})
                for (unsigned byte = 0; byte < 4; ++byte)
                    b.program.inline_bytes.push_back(static_cast<uint8_t>(word >> (8 * byte)));
        }
    }
    uint32_t flat = b.constant(0);
    for (uint32_t axis = 0; axis < logical.extents.size(); ++axis) {
        auto coordinate = b.unary(Op::Coordinate, Type::Index, kNoPhysicalValue, axis);
        // Commit the declared extent without changing valid coordinates.
        coordinate = b.unary(Op::IndexRemainderConstant, Type::Index, coordinate, logical.extents[axis]);
        auto term = b.emit(Op::IndexMultiply, Type::Index, coordinate, b.constant(element_strides[axis]));
        flat = b.emit(Op::IndexAdd, Type::Index, flat, term);
    }
    const auto unit = b.unary(Op::IndexDivideConstant, Type::Index, flat, summary.unit_elements);
    const auto element = b.unary(Op::IndexRemainderConstant, Type::Index, flat, summary.unit_elements);
    const auto zero = b.constant(0, Type::U32);
    const auto ones = b.constant(UINT32_MAX, Type::U32);
    const auto one = b.constant(1, Type::U32);
    const auto thirty_two = b.constant(32);
    std::vector<uint32_t> bases;
    for (auto stride : strides)
        bases.push_back(b.emit(Op::IndexMultiply, Type::Index, unit, b.constant(stride * 8)));
    std::vector<uint32_t> maps(certificate.access_map_summaries().size(), kNoPhysicalValue);
    std::vector<uint32_t> nodes;
    for (const auto& node : certificate.node_summaries()) {
        using NodeOp = CodecCertificateNodeOperation;
        uint32_t value = kNoPhysicalValue;
        if (node.operation == NodeOp::LoadScalar || node.operation == NodeOp::LoadBits) {
            value = maps[node.immediate];
            if (value == kNoPhysicalValue) {
                const auto& map = certificate.access_map_summaries()[node.immediate];
                const uint32_t bits = planes[map.plane].bytes_per_unit * 8;
                const auto window_width = static_cast<uint8_t>(std::min(32u, bits));
                auto record = b.constant(map.first);
                if (map.count != 1) record = b.emit(Op::IndexAdd, Type::Index, record, element);
                record = b.emit(Op::IndexMultiply, Type::Index, record, b.constant(64));
                auto address = b.unary(Op::IndexFromU32, Type::Index, b.load(table_plane, record, 32));
                // The table is compiler-owned. Remainder exposes its established
                // bound to the interval verifier without trusting loaded bytes.
                address = b.unary(Op::IndexRemainderConstant, Type::Index, address, bits - window_width + 1);
                auto control_address = b.emit(Op::IndexAdd, Type::Index, record, thirty_two);
                auto control = b.load(table_plane, control_address, 32);
                address = b.emit(Op::IndexAdd, Type::Index, bases[map.plane], address);
                // LoadBits accepts arbitrary bit addresses, including 32-bit
                // fields crossing byte boundaries. A=min(S,B-W), R=S-A;
                // S+D<=B and D<=W imply R+D<=W, including the final byte.
                auto window = b.load(map.plane, address, window_width);
                auto shift = b.unary(Op::IndexFromU32, Type::Index, b.field(control, 0, 31));
                auto raw = b.emit(Op::U32FunnelShiftRight, Type::U32, window, zero, shift);
                auto width = b.unary(Op::IndexFromU32, Type::Index, b.field(control, 5, 63));
                width = b.unary(Op::IndexRemainderConstant, Type::Index, width, 33);
                auto complement = b.emit(Op::IndexSubtract, Type::Index, thirty_two, width);
                auto mask = b.emit(Op::U32FunnelShiftRight, Type::U32, ones, zero, complement);
                raw = b.emit(Op::U32And, Type::U32, raw, mask);
                if (node.operation == NodeOp::LoadScalar) {
                    auto mode = b.unary(Op::IndexFromU32, Type::Index, b.field(control, 16, 1));
                    auto half_mode = b.emit(Op::IndexLess, Type::Predicate, mode, b.constant(1));
                    auto half = b.unary(Op::F16ToF32, Type::F32, raw, 0, 1);
                    auto full = b.unary(Op::BitsToF32, Type::F32, raw, 0, 1);
                    value = b.emit(Op::Select, Type::F32, half_mode, half, full);
                } else {
                    auto amount = b.unary(Op::IndexFromU32, Type::Index, b.field(control, 11, 31));
                    amount = b.emit(Op::IndexSubtract, Type::Index, thirty_two, amount);
                    auto factor = b.emit(Op::U32FunnelShiftRight, Type::U32, zero, one, amount);
                    raw = b.emit(Op::U32Multiply, Type::U32, raw, factor);
                    if (node.value_type == CodecCertificateNodeValueType::Signed) {
                        auto lower = b.unary(Op::U32ShiftRightConstant, Type::U32, mask, 1);
                        auto signbit = b.emit(Op::U32Xor, Type::U32, mask, lower);
                        signbit = b.emit(Op::U32Multiply, Type::U32, signbit, factor);
                        auto sign = b.emit(Op::U32And, Type::U32, raw, signbit);
                        auto twice = b.emit(Op::U32Add, Type::U32, sign, sign);
                        auto negative = b.emit(Op::U32Xor, Type::U32, twice, ones);
                        negative = b.emit(Op::U32Add, Type::U32, negative, one);
                        raw = b.emit(Op::U32Add, Type::U32, raw, negative);
                        raw = b.unary(Op::SignExtend, Type::I32, raw);
                        b.program.instructions[raw].bit_width = 32;
                        value = b.unary(Op::I32ToF32, Type::F32, raw, 0, 0);
                    } else {
                        value = b.unary(Op::U32ToF32, Type::F32, raw, 0, 0);
                    }
                }
                maps[node.immediate] = value;
            }
        } else if (node.operation == NodeOp::Constant) {
            value = b.emit(Op::ConstF32Bits, Type::F32, kNoPhysicalValue,
                           kNoPhysicalValue, kNoPhysicalValue, certificate.constant_words()[node.immediate]);
        } else if (node.operation == NodeOp::CastFloat) {
            value = nodes[node.argument0];
        } else {
            Op op = Op::F32Add;
            switch (node.operation) {
                case NodeOp::Add: op = Op::F32Add; break;
                case NodeOp::Sub: op = Op::F32Subtract; break;
                case NodeOp::Mul: op = Op::F32Multiply; break;
                case NodeOp::Fma: op = Op::F32Fma; break;
                case NodeOp::Neg: op = Op::F32Negate; break;
                default: return error("certificate node operation is unsupported");
            }
            value = b.emit(op, Type::F32, nodes[node.argument0],
                node.argument1 == 0xffff ? kNoPhysicalValue : nodes[node.argument1],
                node.argument2 == 0xffff ? kNoPhysicalValue : nodes[node.argument2],
                0, node.operation == NodeOp::Fma ? 2 : 1);
        }
        nodes.push_back(value);
    }
    b.program.result = nodes.back();
    return b.finish(required_units);
}
} // namespace Laplace
