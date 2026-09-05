#include <array>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include "artifact_index.h"
#include "artifact_set.h"
#include "canonical_metal.h"
#include "codec_binding.h"
#include "codec_certificate.h"
#include "codec_program.h"
#include "compat_rule.h"
#include "product_package.h"
#include "program_metal.h"
#include "program_package.h"
#include "runtime_session.h"
#include "semantic_manifest.h"
#include "test_util.h"

using namespace Laplace;

namespace Laplace {
bool metal_device_present();
}

namespace {

struct Storage {
    std::vector<uint8_t> bytes;

    uint64_t reserve(size_t length, size_t alignment = 64) {
        const size_t mask = alignment - 1;
        const size_t offset = (bytes.size() + mask) & ~mask;
        bytes.resize(offset + length, 0);
        return offset;
    }

    template<class T>
    void put(uint64_t offset, size_t index, T value) {
        std::memcpy(bytes.data() + offset + index * sizeof(T), &value, sizeof(T));
    }
};

struct ProductFixture {
    std::string source_path;
    ArtifactIndex index;
    SemanticModel model;
    std::shared_ptr<const RuntimePackage> diagnostic;
    // Test-only closed-package route; this is not a carried-manifest witness.
    std::shared_ptr<const RuntimePackage> closed_route;
};

SessionRequest request();

using ProductSessionFactory = SessionCreateResult (*)(
    const ProductPackage&, SessionRequest, SessionFaultPoint);
static_assert(!std::is_invocable_v<ProductSessionFactory,
                                   std::shared_ptr<const RuntimePackage>,
                                   SessionRequest, SessionFaultPoint>);
static_assert(!std::is_copy_constructible_v<ProductPackage>);
static_assert(!std::is_copy_assignable_v<ProductPackage>);
static_assert(std::is_nothrow_move_constructible_v<ProductPackage>);
static_assert(std::is_nothrow_move_assignable_v<ProductPackage>);

std::string temporary_path(const char* pattern) {
    char path[64] = {};
    std::strncpy(path, pattern, sizeof(path) - 1);
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd >= 0) close(fd);
    return path;
}

bool write_bytes(const std::string& path, std::span<const uint8_t> bytes) {
    const int fd = open(path.c_str(), O_WRONLY | O_TRUNC);
    if (fd < 0) return false;
    const bool result = write(fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size());
    close(fd);
    return result;
}

void cleanup(ProductFixture& fixture) {
    if (!fixture.source_path.empty()) unlink(fixture.source_path.c_str());
}

SemanticValue value(uint32_t id, uint32_t width) {
    return {id, ScalarType::F32,
            {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, width}}, 0};
}

SemanticTensor vector_tensor(uint32_t id, TensorRole role, uint32_t width,
                             uint64_t offset, uint64_t length) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, width}};
    tensor.layout.rank = 1;
    tensor.layout.axis_order = {0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides[0] = 1;
    tensor.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, offset, length, 64, 0}};
    return tensor;
}

SemanticTensor dense_tensor(uint32_t id, TensorRole role, uint32_t columns, uint32_t rows,
                            ScalarType storage, uint64_t offset, uint64_t length) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    // Semantic values accumulate in FP32 even when the physical weight plane
    // is F16; this is the canonical Metal matrix contract.
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, columns}, {DimensionKind::Constant, rows}};
    tensor.layout.rank = 2;
    tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides = {rows, 1, 0, 0, 0, 0, 0, 0};
    tensor.planes = {{PlaneKind::Values, storage, ArtifactId{0}, offset, length, 64, 0}};
    return tensor;
}

SemanticTensor blocked_tensor(uint32_t id, TensorRole role, uint32_t columns, uint32_t rows,
                              uint32_t block_bytes, uint64_t offset, uint64_t length) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, columns}, {DimensionKind::Constant, rows}};
    tensor.layout.kind = PhysicalLayoutKind::GgufBlocked;
    tensor.layout.packing = PackingKind::Gguf;
    tensor.layout.rank = 2;
    tensor.layout.block_rank = 1;
    tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides = {1, columns, 0, 0, 0, 0, 0, 0};
    tensor.layout.block_elements = 256;
    tensor.layout.block_bytes = block_bytes;
    tensor.quantization.kind = QuantizationKind::BlockedAffine;
    tensor.quantization.accumulation_type = ScalarType::F32;
    tensor.quantization.scale_type = ScalarType::F16;
    tensor.quantization.zero_type = block_bytes == 144 ? ScalarType::F16 : static_cast<ScalarType>(0);
    tensor.quantization.block_elements = 256;
    tensor.quantization.block_bytes = block_bytes;
    tensor.quantization.group_size = 256;
    tensor.quantization.required_plane_mask = artifact_plane_mask(PlaneKind::Values);
    tensor.planes = {{PlaneKind::Values, ScalarType::U8, ArtifactId{0}, offset, length, 64, 0}};
    return tensor;
}

SemanticTensor grouped_affine_u2_tensor(uint32_t id, TensorRole role, uint32_t output,
                                        uint32_t input, uint64_t values_offset,
                                        uint64_t values_length, uint64_t scales_offset,
                                        uint64_t scales_length, uint64_t biases_offset,
                                        uint64_t biases_length) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, output},
                         {DimensionKind::Constant, input}};
    tensor.layout.kind = PhysicalLayoutKind::GroupedAffine;
    tensor.layout.packing = PackingKind::LsbBitPacked;
    tensor.layout.rank = 2;
    tensor.layout.block_rank = 1;
    tensor.layout.axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides = {1, input, 0, 0, 0, 0, 0, 0};
    tensor.layout.block_elements = 256;
    tensor.layout.block_bytes = 64;
    tensor.quantization.kind = QuantizationKind::BlockedAffine;
    tensor.quantization.accumulation_type = ScalarType::F32;
    tensor.quantization.scale_type = ScalarType::F16;
    tensor.quantization.bias_type = ScalarType::F16;
    tensor.quantization.block_elements = 256;
    tensor.quantization.block_bytes = 64;
    tensor.quantization.group_size = 256;
    tensor.quantization.required_plane_mask = artifact_plane_mask(PlaneKind::Values) |
        artifact_plane_mask(PlaneKind::Scales) | artifact_plane_mask(PlaneKind::Biases);
    tensor.planes = {
        {PlaneKind::Values, ScalarType::U32, ArtifactId{0}, values_offset,
         values_length, 128, 0},
        {PlaneKind::Scales, ScalarType::F16, ArtifactId{0}, scales_offset,
         scales_length, 128, 0},
        {PlaneKind::Biases, ScalarType::F16, ArtifactId{0}, biases_offset,
         biases_length, 128, 0},
    };
    return tensor;
}

ArtifactTensorRecord physical_record(const SemanticTensor& tensor) {
    ArtifactTensorRecord record;
    record.id = tensor.id;
    record.coordinate.root = 0;
    record.logical_type = tensor.logical_type == ScalarType::F16
        ? ArtifactScalarType::F16
        : tensor.logical_type == ScalarType::U8 ? ArtifactScalarType::Packed : ArtifactScalarType::F32;
    for (const Dimension& dimension : tensor.dimensions) {
        record.logical_dimensions.push_back(dimension.constant_or_symbol);
    }
    record.layout = tensor.layout;
    record.quantization = tensor.quantization;
    if (tensor.layout.kind == PhysicalLayoutKind::GroupedAffine) {
        uint64_t logical_elements = 1;
        for (uint64_t dimension : record.logical_dimensions) logical_elements *= dimension;
        const uint64_t group_count = logical_elements / 256;
        for (const TensorPlane& source : tensor.planes) {
            if (source.kind == PlaneKind::Values) {
                record.planes.push_back({source.kind, ArtifactScalarType::U32,
                                         {source.artifact_id, source.offset, source.length},
                                         logical_elements, 4, 16, source.alignment});
            } else {
                record.planes.push_back({source.kind, ArtifactScalarType::F16,
                                         {source.artifact_id, source.offset, source.length},
                                         group_count, 2, 1, source.alignment});
            }
        }
        record.axis.source_rank = 2;
        record.axis.source_axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        record.axis.block_axis = 1;
        record.axis.block_elements = 256;
        record.axis.bytes_per_block = 64;
        record.axis.row_stride_bytes = record.logical_dimensions[1] / 4;
        record.format = {2, ArtifactPhysicalEncoding::GroupedAffineU2_256,
                         ArtifactScalarType::U32, ArtifactScalarType::F16,
                         ArtifactScalarType::None, ArtifactScalarType::None,
                         ArtifactScalarType::None, 256, 64, 2, 0, 0, 0,
                         ArtifactScalarType::F16, 2};
        return record;
    }
    bool transposed = false;
    for (size_t axis = 0; axis != record.logical_dimensions.size(); ++axis)
        transposed = transposed || tensor.layout.axis_order[axis] != axis;
    if (transposed) {
        std::vector<uint64_t> physical_dimensions(record.logical_dimensions.size());
        for (size_t physical_axis = 0; physical_axis != physical_dimensions.size(); ++physical_axis) {
            const uint8_t logical_axis = tensor.layout.axis_order[physical_axis];
            if (logical_axis < record.logical_dimensions.size())
                physical_dimensions[physical_axis] = record.logical_dimensions[logical_axis];
        }
        record.logical_dimensions = std::move(physical_dimensions);
        for (size_t axis = 0; axis != record.logical_dimensions.size(); ++axis) {
            record.layout.axis_order[axis] = static_cast<uint8_t>(axis);
            record.axis.source_axis_order[axis] = static_cast<uint8_t>(axis);
        }
        record.axis.source_rank = static_cast<uint8_t>(record.logical_dimensions.size());
    }
    const TensorPlane& source = tensor.planes.front();
    record.planes.push_back({PlaneKind::Values,
                             source.storage_type == ScalarType::U8 ? ArtifactScalarType::Packed
                                                                    : source.storage_type == ScalarType::F16
                                                                          ? ArtifactScalarType::F16
                                                                          : ArtifactScalarType::F32,
                             {ArtifactId{0}, source.offset, source.length},
                             [&] {
                                 uint64_t count = 1;
                                 for (uint64_t dimension : record.logical_dimensions) count *= dimension;
                                 return count;
                             }(),
                             source.storage_type == ScalarType::U8 ? tensor.layout.block_bytes
                                                                    : source.storage_type == ScalarType::F16 ? 2u : 4u,
                             source.storage_type == ScalarType::U8 ? tensor.layout.block_elements : 1u,
                             source.alignment});
    size_t unit_stride_axis = 0;
    for (size_t axis = 0; axis < record.logical_dimensions.size(); ++axis)
        if (record.layout.strides[axis] == 1) unit_stride_axis = axis;
    const uint64_t physical_row_width = record.logical_dimensions.empty()
        ? 0 : record.logical_dimensions[unit_stride_axis];
    record.axis.row_stride_bytes = source.storage_type == ScalarType::U8
        ? tensor.layout.block_elements == 0
              ? 0
              : static_cast<uint64_t>(tensor.layout.block_bytes) *
                    (physical_row_width / tensor.layout.block_elements)
        : static_cast<uint64_t>(source.storage_type == ScalarType::F16 ? 2 : 4) *
              physical_row_width;
    if (source.storage_type == ScalarType::U8) {
        record.axis.source_rank = 2;
        record.axis.source_axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        record.axis.block_axis = 0;
        record.axis.block_elements = tensor.layout.block_elements;
        record.axis.bytes_per_block = tensor.layout.block_bytes;
        record.format = tensor.layout.block_bytes == 144
            ? ArtifactPhysicalFormat{1, ArtifactPhysicalEncoding::Q4_K, ArtifactScalarType::Packed,
                                      ArtifactScalarType::F16, ArtifactScalarType::F16,
                                      ArtifactScalarType::Packed, ArtifactScalarType::None,
                                      256, 144, 2, 2, 12, 0}
            : ArtifactPhysicalFormat{1, ArtifactPhysicalEncoding::Q6_K, ArtifactScalarType::Packed,
                                     ArtifactScalarType::F16, ArtifactScalarType::None,
                                     ArtifactScalarType::I8, ArtifactScalarType::None,
                                     256, 210, 2, 0, 16, 0};
    } else {
        record.format = source.storage_type == ScalarType::F16
            ? ArtifactPhysicalFormat{1, ArtifactPhysicalEncoding::F16, ArtifactScalarType::F16,
                                     ArtifactScalarType::None, ArtifactScalarType::None,
                                     ArtifactScalarType::None, ArtifactScalarType::None,
                                     1, 2, 0, 0, 0, 0}
            : ArtifactPhysicalFormat{1, ArtifactPhysicalEncoding::F32, ArtifactScalarType::F32,
                                     ArtifactScalarType::None, ArtifactScalarType::None,
                                     ArtifactScalarType::None, ArtifactScalarType::None,
                                     1, 4, 0, 0, 0, 0};
    }
    return record;
}

PhysicalCodecRegistry explicit_codec_registry(const SemanticModel& model) {
    PhysicalCodecRegistry registry;
    for (const SemanticTensor& tensor : model.tensors) {
        std::vector<uint8_t> certificate_bytes;
        if (tensor.planes.size() == 1 &&
            tensor.planes[0].storage_type == ScalarType::F16) {
            certificate_bytes = make_raw_f16_codec_certificate();
        } else if (tensor.planes.size() == 1 &&
                   tensor.planes[0].storage_type == ScalarType::F32) {
            certificate_bytes = make_raw_f32_codec_certificate();
        } else if (tensor.planes.size() == 1 &&
                   tensor.planes[0].storage_type == ScalarType::U8 &&
                   tensor.layout.block_elements == 256 &&
                   tensor.layout.block_bytes == 144) {
            certificate_bytes = make_q4_k_codec_certificate();
        } else if (tensor.planes.size() == 1 &&
                   tensor.planes[0].storage_type == ScalarType::U8 &&
                   tensor.layout.block_elements == 256 &&
                   tensor.layout.block_bytes == 210) {
            certificate_bytes = make_q6_k_codec_certificate();
        } else if (tensor.layout.kind == PhysicalLayoutKind::GroupedAffine &&
                   tensor.layout.block_elements == 256 &&
                   tensor.layout.block_bytes == 64 && tensor.planes.size() == 3 &&
                   tensor.planes[0].storage_type == ScalarType::U32 &&
                   tensor.planes[1].storage_type == ScalarType::F16 &&
                   tensor.planes[2].storage_type == ScalarType::F16) {
            certificate_bytes = make_grouped_affine_u2_codec_certificate();
        } else {
            CHECK_MSG(false,
                      "product fixture tensor has no certificate: id=%u planes=%zu storage=%u block=%ux%u layout=%u",
                      tensor.id, tensor.planes.size(),
                      tensor.planes.empty() ? UINT32_MAX
                                            : static_cast<unsigned>(tensor.planes[0].storage_type),
                      tensor.layout.block_elements, tensor.layout.block_bytes,
                      static_cast<unsigned>(tensor.layout.kind));
            return {};
        }
        const CodecCertificateParseResult parsed =
            parse_codec_certificate(certificate_bytes);
        const auto* certificate = std::get_if<CodecCertificate>(&parsed);
        CHECK(certificate != nullptr);
        if (!certificate) return {};
        auto identity = physical_codec_identity(
            tensor, certificate->identity().abi_version,
            certificate->identity().digest);
        CHECK(identity.has_value());
        if (!identity) continue;
        if (std::none_of(registry.codecs.begin(), registry.codecs.end(),
                         [&](const PhysicalCodecSpec& candidate) {
                             return candidate.identity == *identity;
                         })) {
            registry.codecs.push_back({*identity, certificate_bytes});
        }
        registry.tensors.push_back({tensor.id, *identity});
    }
    std::sort(registry.codecs.begin(), registry.codecs.end(), [](const auto& left, const auto& right) {
        return physical_codec_identity_less(left.identity, right.identity);
    });
    std::sort(registry.tensors.begin(), registry.tensors.end(), [](const auto& left, const auto& right) {
        return left.tensor_id < right.tensor_id;
    });
    return registry;
}

ProductFixture make_closed_route_fixture(SemanticModel model, Storage storage) {
    ProductFixture fixture;
    fixture.model = model;
    fixture.source_path = temporary_path("/private/tmp/laplace-product-weight-XXXXXX");
    CHECK(write_bytes(fixture.source_path, storage.bytes));

    auto artifact_set = ArtifactSet::load_single_file(fixture.source_path);
    CHECK(std::holds_alternative<ArtifactSet>(artifact_set));
    if (!std::holds_alternative<ArtifactSet>(artifact_set)) return fixture;
    auto view = std::get<ArtifactSet>(std::move(artifact_set)).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view));
    if (!std::holds_alternative<PackageView>(view)) return fixture;
    const PackageView primary = std::get<PackageView>(view);

    ArtifactIndexInput input;
    input.artifacts = {primary};
    for (const SemanticTensor& tensor : model.tensors) input.tensors.push_back(physical_record(tensor));
    auto index = ArtifactIndex::build(std::move(input));
    if (const auto* report = std::get_if<CompatibilityReport>(&index)) {
        CHECK_MSG(false, "product fixture artifact index failed: code=%u tensor=%u detail=%s",
                  static_cast<unsigned>(report->code), report->tensor_id,
                  report->detail.c_str());
    } else {
        CHECK(std::holds_alternative<ArtifactIndex>(index));
    }
    if (!std::holds_alternative<ArtifactIndex>(index)) return fixture;
    fixture.index = std::get<ArtifactIndex>(std::move(index));

    TokenIdContract contract;
    contract.vocabulary_size = model.vocabulary_size;
    contract.bos_id = model.bos_id;
    contract.eos_id = model.eos_id;
    contract.stop_ids = model.stop_ids;
    contract.authoritative_tokenizer_digest = {model.tokenizer_digest};
    contract.authoritative_template_digest = {model.template_digest};
    auto diagnostic_manifest = SemanticManifest::build(fixture.index, model, contract);
    CHECK(std::holds_alternative<SemanticManifest>(diagnostic_manifest));
    if (std::holds_alternative<SemanticManifest>(diagnostic_manifest)) {
        auto rejected = RuntimePackage::make_closed_v1_test_only(
            std::get<SemanticManifest>(std::move(diagnostic_manifest)));
        CHECK(rejected != nullptr);
        if (rejected) {
            CHECK(rejected->authority_kind() == PackageAuthorityKind::DiagnosticRaw);
            CHECK(!rejected->product_authoritative());
        }
    }
    const PhysicalCodecRegistry codec_registry = explicit_codec_registry(model);
    CHECK_MSG(!codec_registry.codecs.empty() && !codec_registry.tensors.empty(),
              "product fixture codec registry construction failed: codecs=%zu tensors=%zu model_tensors=%zu",
              codec_registry.codecs.size(), codec_registry.tensors.size(), model.tensors.size());
    auto manifest = SemanticManifest::build(fixture.index, model, contract, codec_registry);
    if (!std::holds_alternative<SemanticManifest>(manifest)) {
        // LAPIR001 deliberately admits only the dense V1 state contract. The
        // recurrent rollback case belongs to a separate carried-manifest gate
        // tranche (LAPIR002), so keep this fixture's rejection explicit.
        CHECK(!model.states.empty());
        return fixture;
    }
    SemanticManifest semantic = std::get<SemanticManifest>(manifest);
    CHECK_MSG(semantic.has_physical_codec_authority(),
              "product fixture manifest lost codec authority: codecs=%zu tensors=%zu",
              semantic.physical_codec_registry().codecs.size(),
              semantic.physical_codec_registry().tensors.size());
    fixture.diagnostic = RuntimePackage::make_diagnostic(semantic, Sha256Digest{}, 0);
    CHECK_MSG(semantic.has_physical_codec_authority(),
              "copying the product fixture manifest into a diagnostic package changed codec authority");
    fixture.closed_route = RuntimePackage::make_closed_v1_test_only(std::move(semantic));
    CHECK(fixture.closed_route != nullptr);
    if (fixture.closed_route) {
        CHECK(fixture.closed_route->authority_kind() == PackageAuthorityKind::ClosedV1Compiled);
        CHECK(fixture.closed_route->product_authoritative());
    }
    return fixture;
}

void test_closed_route_rejects_invalid_compiler_identity(const ProductFixture& fixture) {
    CHECK(fixture.closed_route != nullptr);
    if (!fixture.closed_route) return;
    const SemanticManifest manifest = fixture.closed_route->manifest();
    const ClosedCompilerIdentity valid = fixture.closed_route->closed_compiler_identity();
    CHECK(valid.major == 1);
    CHECK(valid.minor == 0);
    CHECK(valid.revision != 0);
    CHECK(valid.digest != Sha256Digest{});
    const auto expect_diagnostic = [&](ClosedCompilerIdentity invalid) {
        const auto package = RuntimePackage::make_closed_v1_test_only(manifest, invalid);
        CHECK(package != nullptr);
        if (package) {
            CHECK(package->authority_kind() == PackageAuthorityKind::DiagnosticRaw);
            CHECK(!package->product_authoritative());
        }
    };

    ClosedCompilerIdentity invalid = valid;
    invalid.major = 2;
    expect_diagnostic(invalid);
    invalid = valid;
    invalid.minor = 1;
    expect_diagnostic(invalid);
    invalid = valid;
    invalid.revision = 0;
    expect_diagnostic(invalid);
    invalid = valid;
    invalid.digest = {};
    expect_diagnostic(invalid);
}

void test_runtime_carries_verified_physical_package(const ProductFixture& fixture) {
    CHECK(fixture.closed_route != nullptr);
    if (!fixture.closed_route) return;
    PhysicalProgram program;
    program.planes.push_back({PhysicalPlaneStorage::External, 1, 0, 0});
    program.instructions.push_back({PhysicalOpcode::ConstIndex, PhysicalValueType::Index,
                                    {kNoPhysicalValue, kNoPhysicalValue, kNoPhysicalValue},
                                    kNoPhysicalPlane, kNoPhysicalPolicy, 0, 0,
                                    PhysicalBitOrder::Lsb0Little});
    program.instructions.push_back({PhysicalOpcode::LoadBits, PhysicalValueType::U32,
                                    {0, kNoPhysicalValue, kNoPhysicalValue}, 0,
                                    kNoPhysicalPolicy, 0, 8,
                                    PhysicalBitOrder::Lsb0Little});
    program.result = 1;
    const auto wire = encode_physical_program(program);
    const auto digest = physical_program_digest(program);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
    CHECK(std::holds_alternative<PhysicalProgramDigest>(digest));
    if (!std::holds_alternative<std::vector<uint8_t>>(wire) ||
        !std::holds_alternative<PhysicalProgramDigest>(digest)) return;
    PhysicalProgramRecord record{std::get<PhysicalProgramDigest>(digest),
                                 std::get<std::vector<uint8_t>>(wire),
                                 {ElementType::U32, {}}};
    PhysicalResourceBinding binding;
    binding.resource_id = 0;
    binding.program_digest = record.digest;
    binding.planes.push_back({0, ArtifactId{0}, 0, 1});
    const auto loaded = load_physical_program_package(
        fixture.index, fixture.closed_route->manifest(),
        std::span<const PhysicalProgramRecord>(&record, 1),
        std::span<const PhysicalResourceBinding>(&binding, 1));
    CHECK(std::holds_alternative<VerifiedPhysicalProgramPackage>(loaded));
    if (!std::holds_alternative<VerifiedPhysicalProgramPackage>(loaded)) return;
    auto physical = std::make_shared<const VerifiedPhysicalProgramPackage>(
        std::get<VerifiedPhysicalProgramPackage>(loaded));
    auto runtime = RuntimePackage::make_closed_v1_test_only(
        fixture.closed_route->manifest(), fixture.closed_route->closed_compiler_identity(), physical);
    CHECK(runtime != nullptr);
    if (!runtime) return;
    CHECK(runtime->physical_program_package() == physical);
    const auto bound = validate_physical_program_package_for_runtime(*runtime, *physical);
    CHECK(std::holds_alternative<std::monostate>(bound));
    auto compiled_resource = compile_metal_physical_resource(*runtime, 0);
    CHECK(std::holds_alternative<MetalPhysicalProgramExecutable>(compiled_resource));
    if (std::holds_alternative<MetalPhysicalProgramExecutable>(compiled_resource)) {
        MetalPhysicalProgramExecutable executable =
            std::get<MetalPhysicalProgramExecutable>(std::move(compiled_resource));
        const auto executed = executable.execute();
        CHECK(std::holds_alternative<MetalPhysicalProgramResult>(executed));
        if (const auto* result =
                std::get_if<MetalPhysicalProgramResult>(&executed)) {
            CHECK(result->value.type == ElementType::U32);
            CHECK(result->value.extents.empty());
            CHECK(result->value.bits.size() == 1);
            CHECK(result->value.bits[0] == fixture.index.artifacts()[0].bytes()[0]);
            CHECK(result->audit.explicit_upload_bytes == 0);
            CHECK(result->audit.zero_copy_plane_bytes == 1);
            CHECK(result->audit.persistent_plane_bytes == 1);
            CHECK(result->audit.explicit_download_bytes == sizeof(uint32_t));
            CHECK(result->audit.command_buffers == 1);
            CHECK(result->audit.implicit_weight_copies == 0);
        }
    }
    auto session = create_product_runtime_session_for_testing(runtime, request());
    CHECK(std::holds_alternative<CompatibilityReport>(session));
    if (const auto* report = std::get_if<CompatibilityReport>(&session)) {
        CHECK(report->code == CompatibilityError::KERNEL_UNAVAILABLE);
        CHECK(report->detail ==
              "verified physical program package requires the category-free Metal executor");
    }
    const auto diagnostic = RuntimePackage::make_diagnostic(
        fixture.closed_route->manifest(), Sha256Digest{}, 0);
    const auto rejected = validate_physical_program_package_for_runtime(*diagnostic, *physical);
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    if (const auto* report = std::get_if<CompatibilityReport>(&rejected))
        CHECK(report->code == CompatibilityError::AUTHORITY_INVALID);
    const auto unauthorized = compile_metal_physical_resource(*diagnostic, 0);
    CHECK(std::holds_alternative<CompatibilityReport>(unauthorized));
    if (const auto* report = std::get_if<CompatibilityReport>(&unauthorized))
        CHECK(report->code == CompatibilityError::AUTHORITY_INVALID);
    const auto missing = compile_metal_physical_resource(*runtime, 1);
    CHECK(std::holds_alternative<CompatibilityReport>(missing));
    if (const auto* report = std::get_if<CompatibilityReport>(&missing))
        CHECK(report->code == CompatibilityError::IMPORT_TENSOR_UNMAPPED);
}

DimensionExpr program_dimension(uint64_t value) {
    return {DimensionExpression::Constant, value, {}};
}

ValueType program_vector(uint64_t extent) {
    return {ElementType::F32, {program_dimension(extent)}};
}

Program program_session_graph(bool shaped = false) {
    const ValueType token{ElementType::U32, {}};
    const ValueType scores = program_vector(10);
    const ValueType scalar{ElementType::F32, {}};
    Region body;
    body.id = 20;
    body.arguments = {{21, scalar}, {22, token}, {23, scalar}};
    body.instructions = {
        Instruction{24, {Primitive::Add, 1, 0}, {21, 23}, {{25, scalar}},
                    {}, {}, NoAttributes{}}};
    body.yields = {25};
    StructuredTensorAttributes attributes;
    attributes.source_count = 2;
    attributes.iteration_dimensions = {program_dimension(10)};
    attributes.iterator_kinds = {TensorIteratorKind::Parallel};
    attributes.indexing_maps = {
        TensorIndexMap{TensorBoundsMode::Reject,
                       {{TensorIndexExpression::Iterator, 0, {}}}},
        TensorIndexMap{TensorBoundsMode::Reject, {}},
        TensorIndexMap{TensorBoundsMode::Reject,
                       {{TensorIndexExpression::Iterator, 0, {}}}}};
    Region root;
    root.id = 1;
    root.arguments = {{100, token}, {200, scores}};
    root.instructions = {
        Instruction{201, {Primitive::Constant, 1, 0}, {}, {{202, scores}},
                    {}, {}, ConstantAttributes{0}},
        Instruction{203, {Primitive::StructuredTensor, 1, 0},
                    {200, 100, 202}, {{204, scores}}, {20}, {},
                    std::move(attributes)}};
    if (shaped) {
        const ValueType shaped_token{ElementType::U32,
            {program_dimension(1), program_dimension(1)}};
        const ValueType shaped_scores{ElementType::F32,
            {program_dimension(1), program_dimension(10)}};
        root.arguments[0].type = shaped_token;
        root.instructions[0].outputs[0].type = shaped_scores;
        root.instructions[1].outputs[0].type = shaped_scores;
        auto& attributes = std::get<StructuredTensorAttributes>(
            root.instructions[1].attributes);
        attributes.iteration_dimensions.insert(
            attributes.iteration_dimensions.begin(), program_dimension(1));
        attributes.iterator_kinds.insert(attributes.iterator_kinds.begin(),
            TensorIteratorKind::Parallel);
        auto& maps = attributes.indexing_maps;
        maps[0].results[0].value = 1;
        maps[2].results[0].value = 1;
        maps[1].results = {
            {TensorIndexExpression::Constant, 0, {}},
            {TensorIndexExpression::Constant, 0, {}}};
        maps[2].results.insert(maps[2].results.begin(),
            {TensorIndexExpression::Iterator, 0, {}});
    }
    root.yields = {204};
    Program program;
    program.minor = 1;
    program.functions = {{10, 1, {std::move(body), std::move(root)},
                          {scores}}};
    if (shaped) program.functions[0].result_types[0] =
        {ElementType::F32, {program_dimension(1), program_dimension(10)}};
    program.exports = {{10, 0, program.functions[0].result_types[0]}};
    return program;
}

Program stateful_program_session_graph() {
    const ValueType token{ElementType::U32, {}};
    const ValueType scores = program_vector(10);
    const ValueType state_type = program_vector(1);
    const ValueType scalar{ElementType::F32, {}};

    Region increment;
    increment.id = 20;
    increment.arguments = {{21, scalar}, {22, scalar}};
    increment.instructions = {
        Instruction{23, {Primitive::Constant, 1, 0}, {}, {{24, scalar}},
                    {}, {}, ConstantAttributes{std::bit_cast<uint32_t>(1.0f)}},
        Instruction{25, {Primitive::Add, 1, 0}, {21, 24}, {{26, scalar}},
                    {}, {}, NoAttributes{}},
        Instruction{27, {Primitive::Add, 1, 0}, {22, 26}, {{28, scalar}},
                    {}, {}, NoAttributes{}}};
    increment.yields = {28};

    Region scores_body;
    scores_body.id = 30;
    scores_body.arguments = {
        {31, scalar}, {32, scalar}, {33, scalar}};
    scores_body.instructions = {
        Instruction{34, {Primitive::Add, 1, 0}, {31, 32}, {{35, scalar}},
                    {}, {}, NoAttributes{}},
        Instruction{36, {Primitive::Add, 1, 0}, {33, 35}, {{37, scalar}},
                    {}, {}, NoAttributes{}}};
    scores_body.yields = {37};

    StructuredTensorAttributes increment_attributes;
    increment_attributes.source_count = 1;
    increment_attributes.iteration_dimensions = {program_dimension(1)};
    increment_attributes.iterator_kinds = {TensorIteratorKind::Parallel};
    increment_attributes.indexing_maps = {
        TensorIndexMap{TensorBoundsMode::Reject,
                       {{TensorIndexExpression::Iterator, 0, {}}}},
        TensorIndexMap{TensorBoundsMode::Reject,
                       {{TensorIndexExpression::Iterator, 0, {}}}}};
    StructuredTensorAttributes score_attributes;
    score_attributes.source_count = 2;
    score_attributes.iteration_dimensions = {program_dimension(10)};
    score_attributes.iterator_kinds = {TensorIteratorKind::Parallel};
    score_attributes.indexing_maps = {
        TensorIndexMap{TensorBoundsMode::Reject,
                       {{TensorIndexExpression::Iterator, 0, {}}}},
        TensorIndexMap{TensorBoundsMode::Reject,
                       {{TensorIndexExpression::Constant, 0, {}}}},
        TensorIndexMap{TensorBoundsMode::Reject,
                       {{TensorIndexExpression::Iterator, 0, {}}}}};

    Region root;
    root.id = 1;
    root.arguments = {{100, token}, {200, scores}};
    root.instructions = {
        Instruction{201, {Primitive::StateRead, 1, 0}, {},
                    {{202, state_type}}, {}, {}, StateAttributes{30}},
        Instruction{203, {Primitive::Constant, 1, 0}, {},
                    {{204, state_type}}, {}, {}, ConstantAttributes{0}},
        Instruction{205, {Primitive::StructuredTensor, 1, 0}, {202, 204},
                    {{206, state_type}}, {20}, {},
                    std::move(increment_attributes)},
        Instruction{207, {Primitive::StateWrite, 1, 0}, {206}, {}, {},
                    {201}, StateAttributes{30}},
        Instruction{208, {Primitive::Constant, 1, 0}, {}, {{209, scores}},
                    {}, {}, ConstantAttributes{0}},
        Instruction{210, {Primitive::StructuredTensor, 1, 0},
                    {200, 206, 209}, {{211, scores}}, {30}, {},
                    std::move(score_attributes)}};
    root.yields = {211};
    Program program;
    program.minor = 1;
    program.state_references = {{30, state_type, UINT32_MAX, true}};
    program.functions = {{10, 1,
                          {std::move(increment), std::move(scores_body),
                           std::move(root)},
                          {scores}}};
    program.exports = {{10, 0, scores}};
    return program;
}

Program guarded_stateful_program_session_graph() {
    Program program = stateful_program_session_graph();
    Function& function = program.functions.front();
    Region& scores_body = *std::find_if(
        function.regions.begin(), function.regions.end(),
        [](const Region& region) { return region.id == 30; });
    const ValueType token{ElementType::U32, {}};
    const ValueType scalar{ElementType::F32, {}};
    scores_body.arguments = {
        {31, scalar}, {38, scalar}, {39, token}, {32, scalar}, {33, scalar}};
    Region& root = *std::find_if(
        function.regions.begin(), function.regions.end(),
        [](const Region& region) { return region.id == 1; });
    Instruction& scores = *std::find_if(
        root.instructions.begin(), root.instructions.end(),
        [](const Instruction& instruction) { return instruction.id == 210; });
    scores.inputs = {200, 200, 100, 206, 209};
    auto& attributes = std::get<StructuredTensorAttributes>(scores.attributes);
    attributes.source_count = 4;
    TensorIndexExpr guarded_index;
    guarded_index.expression = TensorIndexExpression::Add;
    guarded_index.operands = {
        {TensorIndexExpression::SourceScalar, 2, {}},
        {TensorIndexExpression::Constant, 9, {}}};
    attributes.indexing_maps = {
        TensorIndexMap{TensorBoundsMode::Reject,
                       {{TensorIndexExpression::Iterator, 0, {}}}},
        TensorIndexMap{TensorBoundsMode::Reject,
                       {std::move(guarded_index)}},
        TensorIndexMap{TensorBoundsMode::Reject, {}},
        TensorIndexMap{TensorBoundsMode::Reject,
                       {{TensorIndexExpression::Constant, 0, {}}}},
        TensorIndexMap{TensorBoundsMode::Reject,
                       {{TensorIndexExpression::Iterator, 0, {}}}}};
    return program;
}

TokenProgramDefinition program_session_tokens() {
    TokenProgramDefinition definition;
    for (uint32_t value = 0; value != 256; ++value)
        definition.byte_map[value] = static_cast<uint8_t>(value);
    definition.unknown_token_id = 0;
    definition.vocabulary = {
        {"<unk>", static_cast<uint16_t>(VocabFlags::Special), 0},
        {"a", 0, 0}, {"b", 0, 0}, {"ab", 0, 0}, {" ", 0, 0},
        {"A", 0, 0},
        {"<bos>", static_cast<uint16_t>(VocabFlags::Special), 0},
        {"<eos>", static_cast<uint16_t>(VocabFlags::Special), 0},
        {"x", 0, 0},
        {"<x>", static_cast<uint16_t>(VocabFlags::Special), 0}};
    definition.merges = {{1, 2, 3, 0}};
    definition.prompt = {
        {PromptOpcode::EmitUserText, {}},
        {PromptOpcode::EmitGenerationPrompt, "!"},
        {PromptOpcode::End, {}}};
    definition.prompt_max_bytes = 128;
    definition.normalizer.kind = NormalizerKind::None;
    definition.pretokenizer.kind = PretokenizerKind::ByteLevel;
    definition.postprocessor.kind = PostprocessorKind::None;
    definition.decoder.kind = DecoderKind::ByteLevel;
    return definition;
}

PhysicalProgram program_session_decoder() {
    PhysicalProgram program;
    program.logical_rank = 1;
    program.planes.push_back({PhysicalPlaneStorage::External, 1, 0, 0});
    program.policies.push_back({});
    program.instructions = {
        {PhysicalOpcode::Coordinate, PhysicalValueType::Index,
         {kNoPhysicalValue, kNoPhysicalValue, kNoPhysicalValue},
         kNoPhysicalPlane, kNoPhysicalPolicy, 0, 0,
         PhysicalBitOrder::Lsb0Little},
        {PhysicalOpcode::ConstIndex, PhysicalValueType::Index,
         {kNoPhysicalValue, kNoPhysicalValue, kNoPhysicalValue},
         kNoPhysicalPlane, kNoPhysicalPolicy, 32, 0,
         PhysicalBitOrder::Lsb0Little},
        {PhysicalOpcode::IndexMultiply, PhysicalValueType::Index,
         {0, 1, kNoPhysicalValue}, kNoPhysicalPlane, kNoPhysicalPolicy, 0, 0,
         PhysicalBitOrder::Lsb0Little},
        {PhysicalOpcode::LoadBits, PhysicalValueType::U32,
         {2, kNoPhysicalValue, kNoPhysicalValue}, 0, kNoPhysicalPolicy, 0, 32,
         PhysicalBitOrder::Lsb0Little},
        {PhysicalOpcode::BitsToF32, PhysicalValueType::F32,
         {3, kNoPhysicalValue, kNoPhysicalValue}, kNoPhysicalPlane, 0, 0, 0,
         PhysicalBitOrder::Lsb0Little}};
    program.result = 4;
    return program;
}

std::optional<VerifiedProgramPackage> make_program_session_package(
    bool stateful = false, bool guarded = false, bool shaped = false) {
    auto token_wire = serialize_token_program(program_session_tokens());
    CHECK(std::holds_alternative<std::vector<uint8_t>>(token_wire));
    if (!std::holds_alternative<std::vector<uint8_t>>(token_wire))
        return std::nullopt;
    std::vector<uint8_t> weight_bytes;
    for (float value : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
        const uint32_t bits = std::bit_cast<uint32_t>(value);
        for (unsigned shift = 0; shift != 32; shift += 8)
            weight_bytes.push_back(static_cast<uint8_t>(bits >> shift));
    }
    char path[] = "/private/tmp/laplace-program-session-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd < 0) return std::nullopt;
    CHECK(write(fd, weight_bytes.data(), weight_bytes.size()) ==
          static_cast<ssize_t>(weight_bytes.size()));
    CHECK(close(fd) == 0);
    auto weights = ArtifactSet::load_single_file(path);
    unlink(path);
    CHECK(std::holds_alternative<ArtifactSet>(weights));
    if (!std::holds_alternative<ArtifactSet>(weights)) return std::nullopt;
    auto weight_view =
        std::get<ArtifactSet>(std::move(weights)).view(ArtifactId{0});
    auto token_view = ArtifactSet::make_owned_blob(
        ArtifactId{2}, ArtifactRole::Shard,
        std::get<std::vector<uint8_t>>(token_wire));
    CHECK(std::holds_alternative<PackageView>(weight_view));
    CHECK(std::holds_alternative<PackageView>(token_view));
    if (!std::holds_alternative<PackageView>(weight_view) ||
        !std::holds_alternative<PackageView>(token_view))
        return std::nullopt;
    const Sha256Digest token_digest =
        std::get<PackageView>(token_view).digest();
    ArtifactIndexInput index_input;
    index_input.artifacts.push_back(
        std::get<PackageView>(std::move(weight_view)));
    index_input.artifacts.push_back(
        std::get<PackageView>(std::move(token_view)));
    auto index = ArtifactIndex::build(std::move(index_input));
    CHECK(std::holds_alternative<ArtifactIndex>(index));
    if (!std::holds_alternative<ArtifactIndex>(index)) return std::nullopt;
    auto semantic = verify_and_canonicalize_program(
        guarded ? guarded_stateful_program_session_graph()
                : (stateful ? stateful_program_session_graph()
                            : program_session_graph(shaped)));
    const auto* semantic_error = std::get_if<CompatibilityReport>(&semantic);
    CHECK_MSG(!semantic_error, "program graph: %s", semantic_error ? semantic_error->detail.c_str() : "none");
    if (!std::holds_alternative<VerifiedProgram>(semantic))
        return std::nullopt;
    VerifiedProgram verified =
        std::get<VerifiedProgram>(std::move(semantic));
    StateSchema state_source;
    if (stateful)
        state_source.slots = {{30, ElementType::F32, {1}, 0}};
    auto state = verify_state_schema(std::move(state_source), verified);
    CHECK(std::holds_alternative<VerifiedStateSchema>(state));
    if (!std::holds_alternative<VerifiedStateSchema>(state))
        return std::nullopt;
    const PhysicalProgram decoder = program_session_decoder();
    auto wire = encode_physical_program(decoder);
    auto digest = physical_program_digest(decoder);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(wire));
    CHECK(std::holds_alternative<PhysicalProgramDigest>(digest));
    if (!std::holds_alternative<std::vector<uint8_t>>(wire) ||
        !std::holds_alternative<PhysicalProgramDigest>(digest))
        return std::nullopt;
    PhysicalProgramRecord record{
        std::get<PhysicalProgramDigest>(digest),
        std::get<std::vector<uint8_t>>(std::move(wire)),
        {ElementType::F32, {10}}};
    PhysicalResourceBinding resource;
    resource.resource_id = 1;
    resource.program_digest = record.digest;
    resource.semantic_function_id = 10;
    resource.semantic_value_id = 200;
    resource.planes = {{0, ArtifactId{0}, 0, weight_bytes.size()}};
    const std::array<TokenEndpointBinding, 2> endpoints = {
        TokenEndpointBinding{TokenEndpointKind::InputToken, 10, 100},
        TokenEndpointBinding{TokenEndpointKind::OutputScores, 10, 0}};
    const TokenProgramSource token_source{
        ArtifactId{2}, 0,
        std::get<std::vector<uint8_t>>(token_wire).size(), token_digest};
    auto package = build_program_package(
        std::get<ArtifactIndex>(std::move(index)), std::move(verified),
        std::get<VerifiedStateSchema>(std::move(state)), token_source,
        endpoints, std::span<const PhysicalProgramRecord>(&record, 1),
        std::span<const PhysicalResourceBinding>(&resource, 1));
    CHECK(std::holds_alternative<VerifiedProgramPackage>(package));
    if (!std::holds_alternative<VerifiedProgramPackage>(package))
        return std::nullopt;
    return std::get<VerifiedProgramPackage>(std::move(package));
}

ContainerSchemaProgram program_session_container_schema(uint64_t page_size) {
    ContainerSchemaProgram program;
    program.register_count = 7;
    program.predicate_count = 1;
    ContainerSchemaInstruction match;
    match.opcode = ContainerSchemaOpcode::MatchBytes;
    match.literal = {'L', 'P', 'S', 'X'};
    ContainerSchemaInstruction weight_length;
    weight_length.opcode = ContainerSchemaOpcode::ReadU64Le;
    weight_length.destination = 0;
    ContainerSchemaInstruction token_length = weight_length;
    token_length.destination = 1;
    ContainerSchemaInstruction package_length = weight_length;
    package_length.destination = 2;
    ContainerSchemaInstruction alignment;
    alignment.opcode = ContainerSchemaOpcode::SetConstant;
    alignment.destination = 6;
    alignment.immediate = page_size;
    ContainerSchemaInstruction align;
    align.opcode = ContainerSchemaOpcode::AlignCursor;
    align.input_a = 6;
    const auto emit = [](uint32_t cursor_register, uint32_t length_register,
                         uint32_t section_id) {
        std::array<ContainerSchemaInstruction, 3> result;
        result[0].opcode = ContainerSchemaOpcode::CaptureCursor;
        result[0].destination = cursor_register;
        result[1].opcode = ContainerSchemaOpcode::EmitRange;
        result[1].input_a = cursor_register;
        result[1].input_b = length_register;
        result[1].section_id = section_id;
        result[2].opcode = ContainerSchemaOpcode::Advance;
        result[2].input_a = length_register;
        return result;
    };
    const auto weights = emit(3, 0, 41);
    const auto tokens = emit(4, 1, 42);
    const auto package = emit(5, 2, 23);
    ContainerSchemaInstruction end;
    end.opcode = ContainerSchemaOpcode::RequireCursorEnd;
    program.instructions = {
        match, weight_length, token_length, package_length,
        alignment, align,
        weights[0], weights[1], weights[2],
        align,
        tokens[0], tokens[1], tokens[2],
        align,
        package[0], package[1], package[2], end};
    return program;
}

std::vector<uint8_t> program_session_container(
    const VerifiedProgramPackage& package,
    std::span<const uint8_t> package_wire, size_t page_size) {
    const ArtifactIndex& index = package.physical_package().physical_index();
    const PackageView* weights = nullptr;
    const PackageView* tokens = nullptr;
    for (const PackageView& artifact : index.artifacts()) {
        if (artifact.artifact_id() == ArtifactId{0}) weights = &artifact;
        if (artifact.artifact_id() == ArtifactId{2}) tokens = &artifact;
    }
    CHECK(weights != nullptr);
    CHECK(tokens != nullptr);
    if (!weights || !tokens) return {};
    std::vector<uint8_t> result = {'L', 'P', 'S', 'X'};
    const auto append_u64 = [&](uint64_t value) {
        for (unsigned shift = 0; shift != 64; shift += 8)
            result.push_back(static_cast<uint8_t>(value >> shift));
    };
    append_u64(weights->bytes().size());
    append_u64(tokens->bytes().size());
    append_u64(package_wire.size());
    const auto align = [&](std::vector<uint8_t>& bytes) {
        const size_t remainder = bytes.size() % page_size;
        if (remainder != 0) bytes.resize(bytes.size() + page_size - remainder, 0);
    };
    align(result);
    result.insert(result.end(), weights->bytes().begin(), weights->bytes().end());
    align(result);
    result.insert(result.end(), tokens->bytes().begin(), tokens->bytes().end());
    align(result);
    result.insert(result.end(), package_wire.begin(), package_wire.end());
    return result;
}

void test_program_ingress_creates_runtime_session() {
    auto source = make_program_session_package();
    CHECK(source.has_value());
    if (!source) return;
    auto package_wire = encode_program_package(*source);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(package_wire));
    if (!std::holds_alternative<std::vector<uint8_t>>(package_wire)) return;
    const long native_page_size = sysconf(_SC_PAGESIZE);
    CHECK(native_page_size > 0);
    if (native_page_size <= 0) return;
    const size_t page_size = static_cast<size_t>(native_page_size);
    const auto container = program_session_container(
        *source, std::get<std::vector<uint8_t>>(package_wire), page_size);

    ProgramIngressManifest manifest;
    manifest.package_section_id = 23;
    manifest.schemas = {
        program_session_container_schema(static_cast<uint64_t>(page_size))};
    manifest.artifact_sections = {
        {41, ArtifactId{0}, ArtifactRole::Primary},
        {42, ArtifactId{2}, ArtifactRole::Shard},
    };
    auto ingress_wire = encode_program_ingress_manifest(manifest);
    CHECK(std::holds_alternative<std::vector<uint8_t>>(ingress_wire));
    if (!std::holds_alternative<std::vector<uint8_t>>(ingress_wire)) return;

    const std::string container_path = temporary_path(
        "/private/tmp/laplace-program-ingress-container-XXXXXX");
    const std::string manifest_path = temporary_path(
        "/private/tmp/laplace-program-ingress-manifest-XXXXXX");
    CHECK(write_bytes(container_path, container));
    CHECK(write_bytes(manifest_path,
                      std::get<std::vector<uint8_t>>(ingress_wire)));
    auto loaded = load_program_package(container_path, manifest_path);
    unlink(container_path.c_str());
    unlink(manifest_path.c_str());
    CHECK(std::holds_alternative<VerifiedProgramPackage>(loaded));
    if (!std::holds_alternative<VerifiedProgramPackage>(loaded)) return;
    VerifiedProgramPackage verified =
        std::get<VerifiedProgramPackage>(std::move(loaded));
    CHECK(verified.digest() == source->digest());

    SessionRequest program_request;
    program_request.max_context = 4;
    program_request.max_batch = 1;
    program_request.enable_prefill = true;
    program_request.enable_decode = true;
    auto created = create_runtime_session(verified, program_request);
    const auto* create_error = std::get_if<CompatibilityReport>(&created);
    CHECK_MSG(std::holds_alternative<RuntimeSession>(created),
              "ingress program session create code=%u detail=%s",
              create_error ? static_cast<unsigned>(create_error->code) : 0,
              create_error ? create_error->detail.c_str() : "none");
    if (!std::holds_alternative<RuntimeSession>(created)) return;
    RuntimeSession session = std::get<RuntimeSession>(std::move(created));
    const uint32_t token = 3;
    auto output = session.prefill(std::span<const uint32_t>(&token, 1));
    CHECK(std::holds_alternative<RuntimeOutput>(output));
    if (const auto* result = std::get_if<RuntimeOutput>(&output)) {
        CHECK(result->logits ==
              std::vector<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
        CHECK(result->token_history == std::vector<uint32_t>({3}));
        CHECK(result->command_buffers == 1);
        CHECK(result->completed);
    }
}

bool emit_program_ingress_fixture(const std::string& container_path,
                                  const std::string& manifest_path) {
    auto source = make_program_session_package();
    if (!source) return false;
    auto package_wire = encode_program_package(*source);
    if (!std::holds_alternative<std::vector<uint8_t>>(package_wire))
        return false;
    const long native_page_size = sysconf(_SC_PAGESIZE);
    if (native_page_size <= 0) return false;
    const size_t page_size = static_cast<size_t>(native_page_size);
    const auto container = program_session_container(
        *source, std::get<std::vector<uint8_t>>(package_wire), page_size);
    ProgramIngressManifest manifest;
    manifest.package_section_id = 23;
    manifest.schemas = {
        program_session_container_schema(static_cast<uint64_t>(page_size))};
    manifest.artifact_sections = {
        {41, ArtifactId{0}, ArtifactRole::Primary},
        {42, ArtifactId{2}, ArtifactRole::Shard},
    };
    auto ingress_wire = encode_program_ingress_manifest(manifest);
    if (!std::holds_alternative<std::vector<uint8_t>>(ingress_wire))
        return false;
    return write_bytes(container_path, container) &&
           write_bytes(manifest_path,
                       std::get<std::vector<uint8_t>>(ingress_wire));
}

void test_verified_program_package_creates_runtime_session(bool shaped = false) {
    auto package = make_program_session_package(false, false, shaped);
    CHECK(package.has_value());
    if (!package) return;
    SessionRequest program_request;
    program_request.max_context = 4;
    program_request.max_batch = 1;
    program_request.enable_prefill = true;
    program_request.enable_decode = true;
    auto created = create_runtime_session(*package, program_request);
    const auto* create_error = std::get_if<CompatibilityReport>(&created);
    CHECK_MSG(std::holds_alternative<RuntimeSession>(created),
              "program session create code=%u detail=%s",
              create_error ? static_cast<unsigned>(create_error->code) : 0,
              create_error ? create_error->detail.c_str() : "none");
    if (!std::holds_alternative<RuntimeSession>(created)) return;
    RuntimeSession session = std::get<RuntimeSession>(std::move(created));
    const uint32_t first_token = 3;
    auto first = session.prefill(
        std::span<const uint32_t>(&first_token, 1));
    CHECK(std::holds_alternative<RuntimeOutput>(first));
    if (const auto* output = std::get_if<RuntimeOutput>(&first)) {
        CHECK(output->logits ==
              std::vector<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
        CHECK(output->token_history == std::vector<uint32_t>({3}));
        CHECK(output->command_buffers == 1);
        CHECK(output->host_result_bytes == 10 * sizeof(float));
        CHECK(output->completed);
    }
    auto second = session.decode(4);
    CHECK(std::holds_alternative<RuntimeOutput>(second));
    CHECK(session.token_history() == std::vector<uint32_t>({3, 4}));
    const std::array<uint32_t, 2> too_wide = {1, 2};
    CHECK(std::holds_alternative<CompatibilityReport>(
        session.prefill(too_wide)));
    CHECK(session.token_history() == std::vector<uint32_t>({3, 4}));
    CHECK(std::holds_alternative<CompatibilityReport>(session.decode(10)));
    CHECK(session.token_history() == std::vector<uint32_t>({3, 4}));
}

void test_stateful_program_package_publishes_gpu_state() {
    auto package = make_program_session_package(true);
    CHECK(package.has_value());
    if (!package) return;
    SessionRequest program_request;
    program_request.max_context = 4;
    program_request.max_batch = 1;
    program_request.enable_prefill = true;
    program_request.enable_decode = true;
    auto created = create_runtime_session(*package, program_request);
    const auto* create_error = std::get_if<CompatibilityReport>(&created);
    CHECK_MSG(std::holds_alternative<RuntimeSession>(created),
              "stateful program session create code=%u detail=%s",
              create_error ? static_cast<unsigned>(create_error->code) : 0,
              create_error ? create_error->detail.c_str() : "none");
    if (!std::holds_alternative<RuntimeSession>(created)) return;
    RuntimeSession session = std::get<RuntimeSession>(std::move(created));
    const uint32_t first_token = 1;
    auto first = session.prefill(
        std::span<const uint32_t>(&first_token, 1));
    CHECK(std::holds_alternative<RuntimeOutput>(first));
    if (const auto* output = std::get_if<RuntimeOutput>(&first)) {
        CHECK(output->logits ==
              std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
        CHECK(output->command_buffers == 1);
        CHECK(output->completed);
    }
    auto second = session.decode(2);
    CHECK(std::holds_alternative<RuntimeOutput>(second));
    if (const auto* output = std::get_if<RuntimeOutput>(&second))
        CHECK(output->logits ==
              std::vector<float>({2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    CHECK(session.token_history() == std::vector<uint32_t>({1, 2}));

    auto batch_package = make_program_session_package(true);
    CHECK(batch_package.has_value());
    if (!batch_package) return;
    auto batch_created = create_runtime_session(*batch_package, program_request);
    CHECK(std::holds_alternative<RuntimeSession>(batch_created));
    if (!std::holds_alternative<RuntimeSession>(batch_created)) return;
    RuntimeSession batch =
        std::get<RuntimeSession>(std::move(batch_created));
    const std::array<uint32_t, 2> prompt = {1, 2};
    auto batched = batch.prefill(prompt);
    CHECK(std::holds_alternative<RuntimeOutput>(batched));
    if (const auto* output = std::get_if<RuntimeOutput>(&batched)) {
        CHECK(output->logits ==
              std::vector<float>({2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
        CHECK(output->token_history == std::vector<uint32_t>({1, 2}));
        CHECK(output->command_buffers == 1);
        CHECK(output->completed);
    }
}

void test_program_sequence_failure_does_not_publish_state() {
    auto package = make_program_session_package(true, true);
    CHECK(package.has_value());
    if (!package) return;
    SessionRequest program_request;
    program_request.max_context = 4;
    program_request.max_batch = 2;
    program_request.enable_prefill = true;
    program_request.enable_decode = true;
    auto created = create_runtime_session(*package, program_request);
    const auto* create_error = std::get_if<CompatibilityReport>(&created);
    CHECK_MSG(std::holds_alternative<RuntimeSession>(created),
              "guarded program session create code=%u detail=%s",
              create_error ? static_cast<unsigned>(create_error->code) : 0,
              create_error ? create_error->detail.c_str() : "none");
    if (!std::holds_alternative<RuntimeSession>(created)) return;
    RuntimeSession session = std::get<RuntimeSession>(std::move(created));

    const std::array<uint32_t, 2> failing_prompt = {0, 1};
    auto failed = session.prefill(failing_prompt);
    CHECK(std::holds_alternative<CompatibilityReport>(failed));
    CHECK(session.token_history().empty());

    const uint32_t retry_token = 0;
    auto retry = session.prefill(
        std::span<const uint32_t>(&retry_token, 1));
    CHECK(std::holds_alternative<RuntimeOutput>(retry));
    if (const auto* output = std::get_if<RuntimeOutput>(&retry)) {
        CHECK(output->logits ==
              std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
        CHECK(output->token_history == std::vector<uint32_t>({0}));
        CHECK(output->command_buffers == 1);
        CHECK(output->completed);
    }
    CHECK(session.token_history() == std::vector<uint32_t>({0}));
}

SemanticModel make_dense_model(Storage& storage, uint32_t hidden = 32,
                               bool affine_gate = false) {
    constexpr uint32_t vocabulary = 3;
    const size_t affine_values = static_cast<size_t>(hidden) * hidden / 4;
    const size_t affine_groups = static_cast<size_t>(hidden) * hidden / 256;
    const std::array<size_t, 12> sizes = {
        hidden * vocabulary * 2u, hidden * 4u, hidden * hidden * 2u, hidden * hidden * 2u,
        hidden * hidden * 2u, hidden * hidden * 2u, hidden * 4u, hidden * hidden * 2u,
        hidden * hidden * 2u, hidden * hidden * 2u, hidden * 4u, hidden * vocabulary * 2u};
    std::array<uint64_t, 12> offsets{};
    for (size_t i = 0; i != sizes.size(); ++i) {
        const size_t length = i == 7 && affine_gate ? affine_values : sizes[i];
        offsets[i] = storage.reserve(length, i == 7 && affine_gate ? 128 : 64);
    }
    const uint64_t affine_scales_offset = affine_gate
        ? storage.reserve(affine_groups * sizeof(uint16_t), 128) : 0;
    const uint64_t affine_biases_offset = affine_gate
        ? storage.reserve(affine_groups * sizeof(uint16_t), 128) : 0;
    const auto put_f16 = [&](uint32_t tensor, size_t index, uint16_t bits) { storage.put(offsets[tensor], index, bits); };
    const auto put_f32 = [&](uint32_t tensor, size_t index, float value_bits) { storage.put(offsets[tensor], index, value_bits); };
    for (uint32_t token = 0; token != vocabulary; ++token)
        for (uint32_t channel = 0; channel != hidden; ++channel)
            put_f16(0, token * hidden + channel, token == 0 ? 0x3c00u : 0x3800u);
    for (uint32_t tensor : {1u, 6u, 10u})
        for (uint32_t channel = 0; channel != hidden; ++channel) put_f32(tensor, channel, 1.0f);
    const auto diagonal = [&](uint32_t tensor, uint16_t bits) {
        for (uint32_t channel = 0; channel != hidden; ++channel) put_f16(tensor, channel * hidden + channel, bits);
    };
    for (uint32_t tensor : {2u, 3u, 4u}) diagonal(tensor, 0x3c00u);
    diagonal(5, 0x3800u);
    if (!affine_gate) diagonal(7, 0x3800u);
    diagonal(8, 0x3800u); diagonal(9, 0x3400u);
    if (affine_gate) {
        for (size_t group = 0; group != affine_groups; ++group) {
            storage.put(affine_scales_offset, group, uint16_t{0x3c00u});
            storage.put(affine_biases_offset, group, uint16_t{0});
        }
    }
    put_f16(11, 0, 0x3c00u); put_f16(11, hidden + 1, 0x3800u); put_f16(11, 2 * hidden + 2, 0xb400u);

    SemanticModel model;
    model.schema_major = 1; model.opset_major = 1; model.maximum_context = 32768;
    model.entry_kind = EntryKind::TokenIds; model.vocabulary_size = vocabulary;
    model.bos_id = 0; model.eos_id = 2; model.stop_ids = {2};
    for (size_t i = 0; i != model.tokenizer_digest.size(); ++i) {
        model.tokenizer_digest[i] = static_cast<uint8_t>(i + 1);
        model.template_digest[i] = static_cast<uint8_t>(i + 33);
    }
    model.tensors = {
        dense_tensor(0, TensorRole::TokenEmbedding, hidden, vocabulary, ScalarType::F16, offsets[0], sizes[0]),
        vector_tensor(1, TensorRole::AttentionNormWeight, hidden, offsets[1], sizes[1]),
        dense_tensor(2, TensorRole::QueryWeight, hidden, hidden, ScalarType::F16, offsets[2], sizes[2]),
        dense_tensor(3, TensorRole::KeyWeight, hidden, hidden, ScalarType::F16, offsets[3], sizes[3]),
        dense_tensor(4, TensorRole::ValueWeight, hidden, hidden, ScalarType::F16, offsets[4], sizes[4]),
        dense_tensor(5, TensorRole::AttentionOutputWeight, hidden, hidden, ScalarType::F16, offsets[5], sizes[5]),
        vector_tensor(6, TensorRole::FfnNormWeight, hidden, offsets[6], sizes[6]),
        affine_gate
            ? grouped_affine_u2_tensor(7, TensorRole::FfnGateWeight, hidden, hidden,
                                       offsets[7], affine_values, affine_scales_offset,
                                       affine_groups * sizeof(uint16_t), affine_biases_offset,
                                       affine_groups * sizeof(uint16_t))
            : dense_tensor(7, TensorRole::FfnGateWeight, hidden, hidden,
                           ScalarType::F16, offsets[7], sizes[7]),
        dense_tensor(8, TensorRole::FfnUpWeight, hidden, hidden, ScalarType::F16, offsets[8], sizes[8]),
        dense_tensor(9, TensorRole::FfnDownWeight, hidden, hidden, ScalarType::F16, offsets[9], sizes[9]),
        vector_tensor(10, TensorRole::FinalNormWeight, hidden, offsets[10], sizes[10]),
        dense_tensor(11, TensorRole::OutputWeight, hidden, vocabulary, ScalarType::F16, offsets[11], sizes[11]),
    };
    for (uint32_t id = 0; id != 18; ++id) model.values.push_back(value(id, id == 17 ? vocabulary : hidden));
    model.values.push_back({18, ScalarType::U32, {{DimensionKind::Constant, 1}}, 0});
    model.input_values_first = 18;
    model.input_values_count = 1;
    model.output_values_first = 17;
    model.output_values_count = 1;
    const auto add = [&](OperatorKind kind, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                         std::vector<uint32_t> tensors, std::vector<uint32_t> states, OperatorPayload payload) {
        SemanticOperator op;
        op.id = static_cast<uint32_t>(model.operators.size()); op.semantic_version = 1; op.kind = kind;
        op.inputs = std::move(inputs); op.outputs = std::move(outputs); op.tensors = std::move(tensors);
        op.states = std::move(states); op.payload = std::move(payload); model.operators.push_back(std::move(op));
    };
    constexpr uint32_t one = 0x3f800000u, epsilon = 0x358637bdu;
    add(OperatorKind::EmbeddingLookup, {18}, {0}, {0}, {}, EmbeddingLookupPayload{one, vocabulary, hidden, 0});
    add(OperatorKind::RmsNorm, {0}, {1}, {1}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {1}, {2}, {2}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {3}, {3}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {4}, {4}, {}, LinearPayload{});
    add(OperatorKind::Rope, {2, 3}, {5, 6}, {}, {}, RopePayload{RopePairing::HalfSplit, true, hidden, 0x49742400u, one});
    add(OperatorKind::CausalAttention, {5, 6, 4}, {7}, {}, {0, 1},
        CausalAttentionPayload{1, 1, hidden, 0x3e800000u, AttentionMask::Causal, CachePolicy::Global});
    add(OperatorKind::Linear, {7}, {8}, {5}, {}, LinearPayload{});
    add(OperatorKind::Add, {0, 8}, {9}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {9}, {10}, {6}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {10}, {11}, {7}, {}, LinearPayload{});
    add(OperatorKind::Linear, {10}, {12}, {8}, {}, LinearPayload{});
    add(OperatorKind::SwiGlu, {11, 12}, {13}, {}, {}, SwiGluPayload{ActivationKind::Silu});
    add(OperatorKind::Linear, {13}, {14}, {9}, {}, LinearPayload{});
    add(OperatorKind::Add, {9, 14}, {15}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {15}, {16}, {10}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {16}, {17}, {11}, {}, LinearPayload{});
    model.layers = {{0, 1, 14, 0}};
    StateFormat key_format;
    key_format.logical_type = ScalarType::F32; key_format.encoded_type = ScalarType::F32;
    key_format.encoded_domain = TransformDomain::RopeApplied; key_format.codec = CodecKind::Fp32;
    key_format.cache_policy = CachePolicy::Global; key_format.layout_policy = LayoutPolicy::TokenMajorContiguous;
    key_format.alignment = 64;
    StateFormat value_format = key_format; value_format.encoded_domain = TransformDomain::Untransformed;
    model.states = {
        {0, StateKind::KeyCache, 1, StateUpdateKind::AppendKey, PositionPolicy::AppendOnly,
         {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1}, {DimensionKind::Constant, hidden}}, {key_format}, 0},
        {1, StateKind::ValueCache, 1, StateUpdateKind::AppendValue, PositionPolicy::AppendOnly,
         {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1}, {DimensionKind::Constant, hidden}}, {value_format}, 0},
    };
    return model;
}

SemanticModel make_recurrent_model(Storage& storage) {
    constexpr uint32_t hidden = 512, vocabulary = 3, qk_heads = 4, value_heads = 4;
    constexpr uint32_t head = 128, kernel = 2, channels = head * (2 * qk_heads + value_heads),
                       qk_width = qk_heads * head, inner = value_heads * head, intermediate = 256;
    const std::array<uint32_t, 17> columns = {vocabulary, hidden, hidden, hidden, hidden, hidden, channels,
                                              value_heads, value_heads, head, inner, hidden, hidden, hidden,
                                              intermediate, hidden, vocabulary};
    const std::array<uint32_t, 17> rows = {hidden, 1, channels, inner, value_heads, value_heads, kernel,
                                           1, 1, 1, hidden, 1, intermediate, intermediate, hidden, 1, hidden};
    const std::array<uint32_t, 17> block_bytes = {0, 0, 144, 144, 210, 144, 0, 0, 0, 0, 210, 0, 144, 210, 144, 0, 0};
    Storage* s = &storage;
    std::array<uint64_t, 17> offsets{};
    std::array<uint64_t, 17> lengths{};
    for (size_t i = 0; i != columns.size(); ++i) {
        const uint64_t elements = static_cast<uint64_t>(columns[i]) * rows[i];
        lengths[i] = block_bytes[i] == 0 ? elements * (i == 0 || i == 16 ? 2 : 4)
                                         : elements / 256 * block_bytes[i];
        offsets[i] = s->reserve(static_cast<size_t>(lengths[i]));
    }
    const auto f16 = [&](uint32_t t, size_t i, uint16_t bits) { s->put(offsets[t], i, bits); };
    const auto f32 = [&](uint32_t t, size_t i, float value_bits) { s->put(offsets[t], i, value_bits); };
    for (uint32_t channel = 0; channel != hidden; ++channel) {
        f16(0, channel, 0x3c00u);
        f32(1, channel, 1.0f); f32(11, channel, 1.0f);
    }
    for (uint32_t channel = 0; channel != head; ++channel) f32(9, channel, 1.0f);
    for (uint32_t channel = 0; channel != hidden; ++channel) f32(15, channel, 1.0f);
    for (uint32_t token = 0; token != vocabulary; ++token) f16(16, token * hidden + (token % hidden), 0x3c00u);

    SemanticModel model;
    model.schema_major = 3; model.opset_major = 3; model.maximum_context = 32768;
    model.entry_kind = EntryKind::TokenIds; model.vocabulary_size = vocabulary; model.bos_id = 0; model.eos_id = 2;
    model.stop_ids = {2};
    for (size_t i = 0; i != model.tokenizer_digest.size(); ++i) {
        model.tokenizer_digest[i] = static_cast<uint8_t>(i + 1);
        model.template_digest[i] = static_cast<uint8_t>(i + 33);
    }
    model.tensors.push_back(dense_tensor(0, TensorRole::TokenEmbedding, hidden, vocabulary, ScalarType::F16, offsets[0], lengths[0]));
    model.tensors.push_back(vector_tensor(1, TensorRole::AttentionNormWeight, hidden, offsets[1], lengths[1]));
    model.tensors.push_back(blocked_tensor(2, TensorRole::RecurrentQkvWeight, hidden, channels, 144, offsets[2], lengths[2]));
    model.tensors.push_back(blocked_tensor(3, TensorRole::RecurrentGateWeight, hidden, inner, 144, offsets[3], lengths[3]));
    model.tensors.push_back(blocked_tensor(4, TensorRole::RecurrentBetaWeight, hidden, value_heads, 210, offsets[4], lengths[4]));
    model.tensors.push_back(blocked_tensor(5, TensorRole::RecurrentAlphaWeight, hidden, value_heads, 144, offsets[5], lengths[5]));
    SemanticTensor conv = dense_tensor(6, TensorRole::RecurrentConvWeight,
                                       channels, kernel, ScalarType::F32,
                                       offsets[6], lengths[6]);
    conv.layout.axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    conv.layout.strides = {1, kernel, 0, 0, 0, 0, 0, 0};
    model.tensors.push_back(std::move(conv));
    model.tensors.push_back(vector_tensor(7, TensorRole::RecurrentDtBias, value_heads, offsets[7], lengths[7]));
    model.tensors.push_back(vector_tensor(8, TensorRole::RecurrentDecayWeight, value_heads, offsets[8], lengths[8]));
    model.tensors.push_back(vector_tensor(9, TensorRole::RecurrentNormWeight, head, offsets[9], lengths[9]));
    model.tensors.push_back(blocked_tensor(10, TensorRole::RecurrentOutputWeight, inner, hidden, 210, offsets[10], lengths[10]));
    model.tensors.push_back(vector_tensor(11, TensorRole::FfnNormWeight, hidden, offsets[11], lengths[11]));
    model.tensors.push_back(blocked_tensor(12, TensorRole::FfnGateWeight, hidden, intermediate, 144, offsets[12], lengths[12]));
    model.tensors.push_back(blocked_tensor(13, TensorRole::FfnUpWeight, hidden, intermediate, 210, offsets[13], lengths[13]));
    model.tensors.push_back(blocked_tensor(14, TensorRole::FfnDownWeight, intermediate, hidden, 144, offsets[14], lengths[14]));
    model.tensors.push_back(vector_tensor(15, TensorRole::FinalNormWeight, hidden, offsets[15], lengths[15]));
    model.tensors.push_back(dense_tensor(16, TensorRole::OutputWeight, hidden, vocabulary, ScalarType::F16, offsets[16], lengths[16]));
    const std::array<uint32_t, 23> value_widths = {
        hidden, hidden, channels, inner, value_heads, value_heads, qk_width, qk_width, inner, qk_width, qk_width,
        inner, inner, hidden, hidden, hidden, intermediate, intermediate, intermediate, hidden, hidden, hidden, vocabulary};
    for (uint32_t id = 0; id != value_widths.size(); ++id)
        model.values.push_back(value(id, value_widths[id]));
    model.values.push_back({23, ScalarType::U32, {{DimensionKind::Constant, 1}}, 0});
    model.input_values_first = 23;
    model.input_values_count = 1;
    model.output_values_first = 22;
    model.output_values_count = 1;
    const auto add = [&](OperatorKind kind, std::vector<uint32_t> inputs, std::vector<uint32_t> outputs,
                         std::vector<uint32_t> tensors, std::vector<uint32_t> states, OperatorPayload payload) {
        SemanticOperator op;
        op.id = static_cast<uint32_t>(model.operators.size()); op.semantic_version = 3; op.kind = kind;
        op.inputs = std::move(inputs); op.outputs = std::move(outputs); op.tensors = std::move(tensors);
        op.states = std::move(states); op.payload = std::move(payload); model.operators.push_back(std::move(op));
    };
    constexpr uint32_t epsilon = 0x358637bdu;
    add(OperatorKind::EmbeddingLookup, {23}, {0}, {0}, {}, EmbeddingLookupPayload{0x3f800000u, vocabulary, hidden, 0});
    add(OperatorKind::RmsNorm, {0}, {1}, {1}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {1}, {2}, {2}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {3}, {3}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {4}, {4}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {5}, {5}, {}, LinearPayload{});
    add(OperatorKind::DepthwiseConvSilu, {2}, {6, 7, 8}, {6}, {0}, DepthwiseConvSiluPayload{qk_heads, value_heads, head, kernel});
    add(OperatorKind::L2Normalize, {6}, {9}, {}, {}, L2NormalizePayload{epsilon});
    add(OperatorKind::L2Normalize, {7}, {10}, {}, {}, L2NormalizePayload{epsilon});
    add(OperatorKind::GatedDeltaNet, {9, 10, 8, 4, 5}, {11}, {7, 8}, {1},
        GatedDeltaNetPayload{qk_heads, value_heads, head, QkHeadMapping::ValueHeadModulo,
                             BetaTransform::Sigmoid, DecayTransform::NegativeSoftplus,
                             DeltaStateLayout::ValueHeadKeyRowOutputColumn, 0});
    add(OperatorKind::GatedRmsNorm, {11, 3}, {12}, {9}, {}, GatedRmsNormPayload{epsilon, ActivationKind::Silu, 1});
    add(OperatorKind::Linear, {12}, {13}, {10}, {}, LinearPayload{});
    add(OperatorKind::Add, {0, 13}, {14}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {14}, {15}, {11}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {15}, {16}, {12}, {}, LinearPayload{});
    add(OperatorKind::Linear, {15}, {17}, {13}, {}, LinearPayload{});
    add(OperatorKind::SwiGlu, {16, 17}, {18}, {}, {}, SwiGluPayload{ActivationKind::Silu});
    add(OperatorKind::Linear, {18}, {19}, {14}, {}, LinearPayload{});
    add(OperatorKind::Add, {14, 19}, {20}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {20}, {21}, {15}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {21}, {22}, {16}, {}, LinearPayload{});
    model.layers = {{0, 1, 18, 0}};
    StateFormat recurrent;
    recurrent.kind = StateFormatKind::RecurrentContiguous; recurrent.logical_type = ScalarType::F32;
    recurrent.encoded_type = ScalarType::F32; recurrent.codec = CodecKind::Fp32;
    recurrent.cache_policy = CachePolicy::Recurrent; recurrent.layout_policy = LayoutPolicy::ChannelMajorHistory;
    recurrent.alignment = 64;
    model.states.push_back({0, StateKind::RecurrentConvHistory, 3, StateUpdateKind::ShiftHistory,
                            PositionPolicy::ReplaceAtCursor,
                            {{DimensionKind::Constant, channels}, {DimensionKind::Constant, kernel - 1}}, {recurrent}, 0});
    recurrent.layout_policy = LayoutPolicy::ValueHeadKeyRowOutputColumn;
    model.states.push_back({1, StateKind::RecurrentDeltaMatrix, 3, StateUpdateKind::DeltaMatrix,
                            PositionPolicy::ReplaceAtCursor,
                            {{DimensionKind::Constant, value_heads}, {DimensionKind::Constant, head},
                             {DimensionKind::Constant, head}}, {recurrent}, 0});
    return model;
}

SessionRequest request() {
    SessionRequest result;
    result.max_context = 4;
    result.max_batch = 1;
    result.memory_limit = UINT64_MAX;
    result.enable_prefill = true;
    result.enable_decode = true;
    result.minimum_class = NumericalClass::ExactFp32;
    return result;
}

void test_product_codec_preflight_red(const ProductFixture& fixture) {
    CHECK(fixture.closed_route != nullptr);
    if (!fixture.closed_route) return;
    CodecBindingPreflightResult preflight =
        preflight_codec_bindings(*fixture.closed_route);
    if (const auto* report = std::get_if<CompatibilityReport>(&preflight)) {
        CHECK_MSG(false,
                  "authoritative product codec preflight failed: code=%u operator=%u tensor=%u detail=%s",
                  static_cast<unsigned>(report->code), report->operator_id,
                  report->tensor_id, report->detail.c_str());
    }
    CHECK(std::holds_alternative<ResolvedCodecBindings>(preflight));
    if (const auto* bindings = std::get_if<ResolvedCodecBindings>(&preflight)) {
        CHECK(bindings->matches_package(*fixture.closed_route));
        CHECK(bindings->operators().size() == fixture.model.operators.size());
        uint32_t expected_occurrence = 0;
        for (const ResolvedCodecOperator& operation : bindings->operators()) {
            CHECK(operation.operator_id() < fixture.model.operators.size());
            for (const ResolvedCodecTensor& tensor : operation.tensors()) {
                CHECK(tensor.occurrence_index() == expected_occurrence++);
                CHECK(tensor.program_identity().abi_version ==
                      tensor.physical_identity().arithmetic_version);
                CHECK(tensor.program_identity().contract_digest ==
                      tensor.physical_identity().arithmetic_digest);
                CHECK(!tensor.planes().empty());
            }
        }
        CHECK(expected_occurrence == 12);
    }
}

void test_closed_route_authority_plumbing(const ProductFixture& fixture) {
    if (!fixture.diagnostic) {
        // Recurrent semantic graphs are intentionally outside LAPIR001. Do
        // not silently turn that format boundary into a product session.
        CHECK(fixture.closed_route == nullptr);
        return;
    }
    auto direct = create_canonical_metal_program(fixture.diagnostic, request());
    CHECK(std::holds_alternative<CompatibilityReport>(direct));
    if (const auto* report = std::get_if<CompatibilityReport>(&direct))
        CHECK(report->code == CompatibilityError::PACKAGE_AUTHORITY_REQUIRED);
    CHECK(fixture.closed_route != nullptr);
}

void test_dense_product_session(const ProductFixture& fixture, bool require_grouped_affine = false) {
    if (!fixture.closed_route || !metal_device_present()) {
        if (std::getenv("LAPLACE_TEST_REQUIRE_NATIVE") != nullptr) {
            CHECK_MSG(false, "native product Metal execution is required");
            return;
        }
        std::printf("SKIP: product Metal device unavailable\n");
        return;
    }
    auto created = create_product_runtime_session_for_testing(fixture.closed_route, request());
    if (const auto* report = std::get_if<CompatibilityReport>(&created)) {
        CHECK_MSG(false, "product session construction failed: code=%u operator=%u detail=%s",
                  static_cast<unsigned>(report->code), report->operator_id,
                  report->detail.c_str());
    }
    if (!std::holds_alternative<RuntimeSession>(created)) return;
    RuntimeSession session = std::get<RuntimeSession>(std::move(created));
    CHECK(session.plan().codec_aware);
    CHECK(session.plan().package_fingerprint ==
          fixture.closed_route->package_fingerprint());
    CHECK(session.plan().dispatch_programs.size() == 4);
    CHECK(session.plan().entries.empty());
    auto first = session.prefill(std::vector<uint32_t>{0});
    CHECK(std::holds_alternative<RuntimeOutput>(first));
    if (const auto* output = std::get_if<RuntimeOutput>(&first)) {
        CHECK(output->command_buffers == 1);
        CHECK(output->operator_count != 0);
        CHECK(output->completed);
        CHECK(output->gpu_time_ms >= 0.0);
        CHECK(output->cpu_wait_ms >= 0.0);
        if (require_grouped_affine) {
            CHECK(output->requested_projection_source_bytes > 0);
            CHECK(output->grouped_affine_u2_projection_dispatches == 1);
        }
    }
    CHECK(session.token_history() == std::vector<uint32_t>{0});
    CHECK(session.implicit_weight_copy_count_for_testing() == 0);
    const StateCursor cursor = session.checkpoint();
    auto second = session.decode(1);
    CHECK(std::holds_alternative<RuntimeOutput>(second));
    std::vector<float> expected;
    if (auto* output = std::get_if<RuntimeOutput>(&second)) expected = output->logits;
    CHECK(std::holds_alternative<std::monostate>(session.rollback(cursor)));
    CHECK(session.token_history() == std::vector<uint32_t>{0});
    auto replay = session.decode(1);
    CHECK(std::holds_alternative<RuntimeOutput>(replay));
    if (auto* output = std::get_if<RuntimeOutput>(&replay)) CHECK(output->logits == expected);
    CHECK(session.implicit_weight_copy_count_for_testing() == 0);
    CHECK(std::holds_alternative<CompatibilityReport>(session.save_state()));
    CHECK(std::holds_alternative<CompatibilityReport>(session.restore_state({})));

    auto failed_created = create_product_runtime_session_for_testing(fixture.closed_route, request());
    CHECK(std::holds_alternative<RuntimeSession>(failed_created));
    if (std::holds_alternative<RuntimeSession>(failed_created)) {
        RuntimeSession failed = std::get<RuntimeSession>(std::move(failed_created));
        failed.fail_after_completed_submission_for_testing();
        CHECK(std::holds_alternative<CompatibilityReport>(failed.prefill(std::vector<uint32_t>{0})));
        CHECK(failed.token_history().empty());
        CHECK(std::holds_alternative<CompatibilityReport>(failed.decode(1)));
    }
}

void test_dense_product_two_row_prefill_matches_sequential(
    const ProductFixture& fixture) {
    if (!fixture.closed_route || !metal_device_present()) return;
    SessionRequest batch_request = request();
    batch_request.max_batch = 2;
    auto batch_created = create_product_runtime_session_for_testing(
        fixture.closed_route, batch_request);
    auto capacity_created = create_product_runtime_session_for_testing(
        fixture.closed_route, batch_request);
    auto sequential_created = create_product_runtime_session_for_testing(
        fixture.closed_route, request());
    if (const auto* report = std::get_if<CompatibilityReport>(&batch_created))
        CHECK_MSG(false,
                  "two-row product session construction failed: code=%u operator=%u detail=%s",
                  static_cast<unsigned>(report->code), report->operator_id,
                  report->detail.c_str());
    CHECK(std::holds_alternative<RuntimeSession>(batch_created));
    CHECK(std::holds_alternative<RuntimeSession>(capacity_created));
    CHECK(std::holds_alternative<RuntimeSession>(sequential_created));
    if (!std::holds_alternative<RuntimeSession>(batch_created) ||
        !std::holds_alternative<RuntimeSession>(capacity_created) ||
        !std::holds_alternative<RuntimeSession>(sequential_created))
        return;

    RuntimeSession batch = std::get<RuntimeSession>(std::move(batch_created));
    RuntimeSession capacity =
        std::get<RuntimeSession>(std::move(capacity_created));
    RuntimeSession sequential =
        std::get<RuntimeSession>(std::move(sequential_created));
    const std::vector<uint32_t> tokens = {0, 1};
    const StateCursor empty_capacity = capacity.checkpoint();
    const auto batched = batch.prefill(tokens);
    const auto first = sequential.prefill(std::span<const uint32_t>(tokens.data(), 1));
    const auto second = sequential.decode(tokens[1]);
    const auto one_row = capacity.prefill(
        std::span<const uint32_t>(tokens.data(), 1));
    if (const auto* report = std::get_if<CompatibilityReport>(&batched)) {
        std::fprintf(stderr,
                     "two-row prefill failed: code=%u layer=%u operator=%u detail=%s\n",
                     static_cast<unsigned>(report->code), report->layer,
                     report->operator_id, report->detail.c_str());
    }
    CHECK(std::holds_alternative<RuntimeOutput>(batched));
    CHECK(std::holds_alternative<RuntimeOutput>(first));
    CHECK(std::holds_alternative<RuntimeOutput>(second));
    CHECK(std::holds_alternative<RuntimeOutput>(one_row));
    if (const auto* actual = std::get_if<RuntimeOutput>(&batched)) {
        if (const auto* expected = std::get_if<RuntimeOutput>(&second)) {
            CHECK(actual->completed);
            CHECK(actual->command_buffers == 1);
            CHECK(actual->projection_dispatches != 0);
            CHECK(actual->token_history == tokens);
            CHECK(actual->logits.size() == expected->logits.size());
            for (size_t index = 0;
                 index < actual->logits.size() && index < expected->logits.size();
                 ++index) {
                CHECK(std::abs(actual->logits[index] - expected->logits[index]) <=
                      1e-4f + 1e-4f * std::abs(expected->logits[index]));
            }
        }
    }
    if (const auto* actual = std::get_if<RuntimeOutput>(&one_row)) {
        if (const auto* expected = std::get_if<RuntimeOutput>(&first)) {
            CHECK(actual->command_buffers == 1);
            CHECK(actual->logits == expected->logits);
        }
    }
    CHECK(std::holds_alternative<std::monostate>(
        capacity.rollback(empty_capacity)));
    const auto grown = capacity.prefill(tokens);
    if (const auto* report = std::get_if<CompatibilityReport>(&grown)) {
        std::fprintf(stderr,
                     "two-row prefill after rollback failed: code=%u layer=%u operator=%u detail=%s\n",
                     static_cast<unsigned>(report->code), report->layer,
                     report->operator_id, report->detail.c_str());
    }
    CHECK(std::holds_alternative<RuntimeOutput>(grown));
    if (const auto* actual = std::get_if<RuntimeOutput>(&grown)) {
        if (const auto* expected = std::get_if<RuntimeOutput>(&batched)) {
            CHECK(actual->completed);
            CHECK(actual->command_buffers == 1);
            CHECK(actual->logits == expected->logits);
        }
    }
    CHECK(batch.token_history() == tokens);
    CHECK(capacity.token_history() == tokens);
    CHECK(sequential.token_history() == tokens);
    CHECK(batch.implicit_weight_copy_count_for_testing() == 0);
    CHECK(capacity.implicit_weight_copy_count_for_testing() == 0);
    CHECK(sequential.implicit_weight_copy_count_for_testing() == 0);
}

void test_dense_product_multitoken_prefill_schedules_state_only_programs(
    const ProductFixture& fixture) {
    if (!fixture.closed_route || !metal_device_present()) return;
    const std::vector<uint32_t> prompt = {0, 1, 0};

    auto chained_created = create_product_runtime_session_for_testing(
        fixture.closed_route, request());
    auto sequential_created = create_product_runtime_session_for_testing(
        fixture.closed_route, request());
    CHECK(std::holds_alternative<RuntimeSession>(chained_created));
    CHECK(std::holds_alternative<RuntimeSession>(sequential_created));
    if (!std::holds_alternative<RuntimeSession>(chained_created) ||
        !std::holds_alternative<RuntimeSession>(sequential_created))
        return;
    RuntimeSession chained =
        std::get<RuntimeSession>(std::move(chained_created));
    RuntimeSession sequential =
        std::get<RuntimeSession>(std::move(sequential_created));
    const auto chained_output = chained.prefill(prompt);
    const auto first = sequential.prefill(
        std::span<const uint32_t>(prompt.data(), 1));
    const auto second = sequential.decode(prompt[1]);
    const auto third = sequential.decode(prompt[2]);
    CHECK(std::holds_alternative<RuntimeOutput>(chained_output));
    CHECK(std::holds_alternative<RuntimeOutput>(first));
    CHECK(std::holds_alternative<RuntimeOutput>(second));
    CHECK(std::holds_alternative<RuntimeOutput>(third));
    if (const auto* actual = std::get_if<RuntimeOutput>(&chained_output)) {
        if (const auto* expected = std::get_if<RuntimeOutput>(&third)) {
            CHECK(actual->completed);
            CHECK(actual->command_buffers == prompt.size());
            CHECK(actual->logits == expected->logits);
        }
    }
    CHECK(chained.token_history() == prompt);
    CHECK(sequential.token_history() == prompt);
    CHECK(chained.implicit_weight_copy_count_for_testing() == 0);

    SessionRequest batch_request = request();
    batch_request.max_batch = 2;
    auto batched_created = create_product_runtime_session_for_testing(
        fixture.closed_route, batch_request);
    CHECK(std::holds_alternative<RuntimeSession>(batched_created));
    if (std::holds_alternative<RuntimeSession>(batched_created)) {
        RuntimeSession batched =
            std::get<RuntimeSession>(std::move(batched_created));
        const auto output = batched.prefill(prompt);
        if (const auto* report = std::get_if<CompatibilityReport>(&output)) {
            std::fprintf(stderr,
                         "three-token batched prefill failed: code=%u layer=%u operator=%u detail=%s\n",
                         static_cast<unsigned>(report->code), report->layer,
                         report->operator_id, report->detail.c_str());
        }
        CHECK(std::holds_alternative<RuntimeOutput>(output));
        if (const auto* actual = std::get_if<RuntimeOutput>(&output)) {
            if (const auto* expected = std::get_if<RuntimeOutput>(&third)) {
                CHECK(actual->completed);
                CHECK(actual->command_buffers == 2);
                CHECK(actual->logits.size() == expected->logits.size());
                for (size_t index = 0;
                     index < actual->logits.size() &&
                     index < expected->logits.size(); ++index) {
                    CHECK(std::abs(actual->logits[index] -
                                   expected->logits[index]) <=
                          1e-4f + 1e-4f *
                              std::abs(expected->logits[index]));
                }
            }
        }
        CHECK(batched.token_history() == prompt);
        CHECK(batched.implicit_weight_copy_count_for_testing() == 0);
    }

    auto failed_created = create_product_runtime_session_for_testing(
        fixture.closed_route, request());
    CHECK(std::holds_alternative<RuntimeSession>(failed_created));
    if (std::holds_alternative<RuntimeSession>(failed_created)) {
        RuntimeSession failed =
            std::get<RuntimeSession>(std::move(failed_created));
        failed.fail_after_prefill_tokens_for_testing(1);
        CHECK(std::holds_alternative<CompatibilityReport>(failed.prefill(prompt)));
        CHECK(failed.token_history().empty());
        CHECK(std::holds_alternative<RuntimeOutput>(failed.prefill(prompt)));
        CHECK(failed.token_history() == prompt);
    }
}

void check_moved_from_product_session(RuntimeSession& session, StateCursor cursor) {
    CHECK(session.checkpoint() == StateCursor{});
    CHECK(session.plan().entries.empty());
    CHECK(session.token_history().empty());
    CHECK(std::holds_alternative<CompatibilityReport>(session.prefill(std::vector<uint32_t>{0})));
    CHECK(std::holds_alternative<CompatibilityReport>(session.decode(0)));
    CHECK(std::holds_alternative<CompatibilityReport>(session.prefill_sampled(std::vector<uint32_t>{0})));
    CHECK(std::holds_alternative<CompatibilityReport>(session.decode_sampled(0)));
    CHECK(std::holds_alternative<CompatibilityReport>(session.commit(cursor)));
    CHECK(std::holds_alternative<CompatibilityReport>(session.rollback(cursor)));
    CHECK(std::holds_alternative<CompatibilityReport>(session.save_state()));
    CHECK(std::holds_alternative<CompatibilityReport>(session.restore_state({})));
    CHECK(session.implicit_weight_copy_count_for_testing() == UINT64_MAX);
}

void test_product_session_move_construction(const ProductFixture& fixture) {
    if (!fixture.closed_route || !metal_device_present()) return;
    auto created = create_product_runtime_session_for_testing(fixture.closed_route, request());
    CHECK(std::holds_alternative<RuntimeSession>(created));
    if (!std::holds_alternative<RuntimeSession>(created)) return;
    RuntimeSession source = std::get<RuntimeSession>(std::move(created));
    const StateCursor cursor = source.checkpoint();
    CHECK(cursor.accepted_tokens == 0);
    RuntimeSession destination(std::move(source));
    check_moved_from_product_session(source, cursor);
    CHECK(std::holds_alternative<std::monostate>(destination.commit(cursor)));
    CHECK(std::holds_alternative<RuntimeOutput>(destination.prefill(std::vector<uint32_t>{0})));
}

void test_product_session_move_assignment(const ProductFixture& fixture) {
    if (!fixture.closed_route || !metal_device_present()) return;
    auto source_created = create_product_runtime_session_for_testing(fixture.closed_route, request());
    auto destination_created = create_product_runtime_session_for_testing(fixture.closed_route, request());
    CHECK(std::holds_alternative<RuntimeSession>(source_created));
    CHECK(std::holds_alternative<RuntimeSession>(destination_created));
    if (!std::holds_alternative<RuntimeSession>(source_created) ||
        !std::holds_alternative<RuntimeSession>(destination_created)) return;
    RuntimeSession source = std::get<RuntimeSession>(std::move(source_created));
    RuntimeSession destination = std::get<RuntimeSession>(std::move(destination_created));
    const StateCursor cursor = source.checkpoint();
    CHECK(cursor.accepted_tokens == 0);
    destination = std::move(source);
    check_moved_from_product_session(source, cursor);
    CHECK(std::holds_alternative<std::monostate>(destination.commit(cursor)));
    CHECK(std::holds_alternative<RuntimeOutput>(destination.prefill(std::vector<uint32_t>{0})));
}

uint32_t first_argmax(std::span<const float> values) {
    uint32_t best = 0;
    for (uint32_t index = 1; index < values.size(); ++index) {
        if (values[index] > values[best]) best = index;
    }
    return best;
}

void test_dense_product_device_sampler(const ProductFixture& fixture) {
    if (!fixture.closed_route || !metal_device_present()) {
        if (std::getenv("LAPLACE_TEST_REQUIRE_NATIVE") != nullptr) {
            CHECK_MSG(false, "native product Metal sampler execution is required");
            return;
        }
        std::printf("SKIP: product Metal sampler device unavailable\n");
        return;
    }
    auto control_created = create_product_runtime_session_for_testing(fixture.closed_route, request());
    auto sampled_created = create_product_runtime_session_for_testing(fixture.closed_route, request());
    CHECK(std::holds_alternative<RuntimeSession>(control_created));
    CHECK(std::holds_alternative<RuntimeSession>(sampled_created));
    if (!std::holds_alternative<RuntimeSession>(control_created) ||
        !std::holds_alternative<RuntimeSession>(sampled_created)) return;
    RuntimeSession control = std::get<RuntimeSession>(std::move(control_created));
    RuntimeSession sampled = std::get<RuntimeSession>(std::move(sampled_created));

    const auto compare = [&](const RuntimeRunResult& logits_result,
                             const RuntimeRunResult& sampled_result) {
        CHECK(std::holds_alternative<RuntimeOutput>(logits_result));
        CHECK(std::holds_alternative<RuntimeOutput>(sampled_result));
        const auto* logits = std::get_if<RuntimeOutput>(&logits_result);
        const auto* token = std::get_if<RuntimeOutput>(&sampled_result);
        if (!logits || !token) return;
        CHECK(!logits->logits.empty());
        CHECK(!logits->sampled);
        CHECK(token->logits.empty());
        CHECK(token->sampled);
        CHECK(token->sampled_token_id == first_argmax(logits->logits));
        CHECK(token->sampled_token_id < logits->logits.size());
        CHECK(token->sampled_logit == logits->logits[token->sampled_token_id]);
        CHECK(token->host_result_bytes == 16);
        CHECK(token->command_buffers == 1);
        CHECK(token->completed);
    };

    auto control_first = control.prefill(std::vector<uint32_t>{0});
    auto sampled_first = sampled.prefill_sampled(std::vector<uint32_t>{0});
    compare(control_first, sampled_first);
    CHECK(control.token_history() == sampled.token_history());
    CHECK(sampled.implicit_weight_copy_count_for_testing() == 0);

    const StateCursor cursor = sampled.checkpoint();
    auto control_second = control.decode(1);
    auto sampled_second = sampled.decode_sampled(1);
    compare(control_second, sampled_second);
    uint32_t expected_token = 0;
    float expected_logit = 0.0f;
    if (const auto* output = std::get_if<RuntimeOutput>(&sampled_second)) {
        expected_token = output->sampled_token_id;
        expected_logit = output->sampled_logit;
    }
    CHECK(std::holds_alternative<std::monostate>(sampled.rollback(cursor)));
    auto replay = sampled.decode_sampled(1);
    CHECK(std::holds_alternative<RuntimeOutput>(replay));
    if (const auto* output = std::get_if<RuntimeOutput>(&replay)) {
        CHECK(output->sampled_token_id == expected_token);
        CHECK(output->sampled_logit == expected_logit);
        CHECK(output->logits.empty());
        CHECK(output->host_result_bytes == 16);
    }
    CHECK(sampled.implicit_weight_copy_count_for_testing() == 0);
}

void test_product_request_contract_fails_before_device_use(const ProductFixture& fixture) {
    if (!fixture.closed_route) return;
    const auto expect_error = [&](SessionRequest value, CompatibilityError expected) {
        auto created = create_product_runtime_session_for_testing(fixture.closed_route, value);
        CHECK(std::holds_alternative<CompatibilityReport>(created));
        if (const auto* report = std::get_if<CompatibilityReport>(&created)) {
            CHECK(report->code == expected);
        }
    };

    SessionRequest value = request();
    value.max_batch = 0;
    expect_error(value, CompatibilityError::RUNTIME_INPUT_INVALID);
    value = request();
    value.memory_limit = 0;
    expect_error(value, CompatibilityError::PLAN_MEMORY_EXCEEDED);
    value = request();
    value.enable_prefill = false;
    value.enable_decode = false;
    expect_error(value, CompatibilityError::RUNTIME_INPUT_INVALID);
    value = request();
    value.enable_streaming = true;
    expect_error(value, CompatibilityError::STREAMING_UNSUPPORTED);
    value = request();
    value.enable_speculation = true;
    expect_error(value, CompatibilityError::FALLBACK_FORBIDDEN);
    value = request();
    value.minimum_class = static_cast<NumericalClass>(UINT16_MAX);
    expect_error(value, CompatibilityError::RUNTIME_INPUT_INVALID);
    value = request();
    value.objective = static_cast<RuntimeObjective>(UINT16_MAX);
    expect_error(value, CompatibilityError::RUNTIME_INPUT_INVALID);
}

void test_unqualified_derived_weights_are_not_product_authoritative(
    const ProductFixture& fixture) {
    if (!fixture.closed_route) return;
    const auto expect_rejected = [&](const CanonicalDerivedQ2KPolicy& q2,
                                     const CanonicalSparseFfnPolicy& sparse,
                                     const CanonicalDerivedIQ2XXSPolicy& iq2,
                                     const CanonicalCalibrationPolicy& calibration,
                                     const CanonicalDerivedColumnGroupedU2Policy& u2) {
        auto created = create_canonical_metal_program(
            fixture.closed_route, request(), q2, sparse, iq2, calibration, u2);
        CHECK(std::holds_alternative<CompatibilityReport>(created));
        if (const auto* report = std::get_if<CompatibilityReport>(&created)) {
            CHECK(report->code == CompatibilityError::PACKAGE_AUTHORITY_REQUIRED);
        }
    };

    CanonicalDerivedQ2KPolicy q2;
    q2.tensor_roles = {TensorRole::OutputWeight};
    expect_rejected(q2, {}, {}, {}, {});

    CanonicalSparseFfnPolicy sparse;
    sparse.runs = {{0, 1}};
    expect_rejected({}, sparse, {}, {}, {});

    CanonicalDerivedIQ2XXSPolicy iq2;
    iq2.tensor_roles = {TensorRole::OutputWeight};
    expect_rejected({}, {}, iq2, {}, {});

    CanonicalCalibrationPolicy calibration;
    calibration.targets.push_back({});
    expect_rejected({}, {}, {}, calibration, {});

    CanonicalDerivedColumnGroupedU2Policy u2;
    u2.tensor_roles = {TensorRole::OutputWeight};
    expect_rejected({}, {}, {}, {}, u2);
}

void test_recurrent_rollback_is_closed(const ProductFixture& fixture) {
    if (!fixture.closed_route || !metal_device_present()) return;
    auto created = create_product_runtime_session_for_testing(fixture.closed_route, request());
    if (const auto* report = std::get_if<CompatibilityReport>(&created))
        CHECK_MSG(false, "recurrent product session construction failed: code=%u operator=%u tensor=%u detail=%s",
                  static_cast<unsigned>(report->code), report->operator_id,
                  report->tensor_id, report->detail.c_str());
    CHECK(std::holds_alternative<RuntimeSession>(created));
    if (!std::holds_alternative<RuntimeSession>(created)) return;
    RuntimeSession session = std::get<RuntimeSession>(std::move(created));
    auto output = session.prefill(std::vector<uint32_t>{0});
    if (const auto* report = std::get_if<CompatibilityReport>(&output)) {
        CHECK_MSG(false, "recurrent product token failed: code=%u operator=%u detail=%s",
                  static_cast<unsigned>(report->code), report->operator_id,
                  report->detail.c_str());
    } else {
        CHECK(std::holds_alternative<RuntimeOutput>(output));
    }
    const StateCursor cursor = session.checkpoint();
    auto rollback = session.rollback(cursor);
    CHECK(std::holds_alternative<CompatibilityReport>(rollback));
    if (const auto* report = std::get_if<CompatibilityReport>(&rollback))
        CHECK(report->code == CompatibilityError::CACHE_MODE_UNQUALIFIED);
    CHECK(std::holds_alternative<CompatibilityReport>(session.save_state()));
}

void test_recurrent_prefill_chains_and_rolls_back(const ProductFixture& fixture) {
    if (!fixture.closed_route || !metal_device_present()) return;
    const std::vector<uint32_t> prompt = {0, 1, 0};

    auto chained_created = create_product_runtime_session_for_testing(fixture.closed_route, request());
    auto sequential_created = create_product_runtime_session_for_testing(fixture.closed_route, request());
    if (const auto* report = std::get_if<CompatibilityReport>(&chained_created))
        CHECK_MSG(false, "recurrent chained session construction failed: code=%u operator=%u tensor=%u detail=%s",
                  static_cast<unsigned>(report->code), report->operator_id,
                  report->tensor_id, report->detail.c_str());
    if (const auto* report = std::get_if<CompatibilityReport>(&sequential_created))
        CHECK_MSG(false, "recurrent sequential session construction failed: code=%u operator=%u tensor=%u detail=%s",
                  static_cast<unsigned>(report->code), report->operator_id,
                  report->tensor_id, report->detail.c_str());
    CHECK(std::holds_alternative<RuntimeSession>(chained_created));
    CHECK(std::holds_alternative<RuntimeSession>(sequential_created));
    if (!std::holds_alternative<RuntimeSession>(chained_created) ||
        !std::holds_alternative<RuntimeSession>(sequential_created)) return;
    RuntimeSession chained = std::get<RuntimeSession>(std::move(chained_created));
    RuntimeSession sequential = std::get<RuntimeSession>(std::move(sequential_created));

    auto chained_output = chained.prefill(prompt);
    auto first = sequential.prefill(std::vector<uint32_t>{prompt[0]});
    auto second = sequential.decode(prompt[1]);
    auto third = sequential.decode(prompt[2]);
    if (const auto* report = std::get_if<CompatibilityReport>(&first)) {
        CHECK_MSG(false, "recurrent sequential first token failed: code=%u operator=%u detail=%s",
                  static_cast<unsigned>(report->code), report->operator_id,
                  report->detail.c_str());
    }
    CHECK(std::holds_alternative<RuntimeOutput>(chained_output));
    CHECK(std::holds_alternative<RuntimeOutput>(first));
    CHECK(std::holds_alternative<RuntimeOutput>(second));
    CHECK(std::holds_alternative<RuntimeOutput>(third));
    if (const auto* actual = std::get_if<RuntimeOutput>(&chained_output)) {
        if (const auto* expected = std::get_if<RuntimeOutput>(&third)) {
            CHECK(actual->logits == expected->logits);
            CHECK(actual->token_history == prompt);
            CHECK(actual->command_buffers == 3);
            CHECK(actual->completed);
            CHECK(expected->completed);
        }
    }
    CHECK(chained.implicit_weight_copy_count_for_testing() == 0);
    CHECK(sequential.implicit_weight_copy_count_for_testing() == 0);

    auto chained_next = chained.decode(1);
    auto sequential_next = sequential.decode(1);
    CHECK(std::holds_alternative<RuntimeOutput>(chained_next));
    CHECK(std::holds_alternative<RuntimeOutput>(sequential_next));
    if (const auto* actual = std::get_if<RuntimeOutput>(&chained_next)) {
        if (const auto* expected = std::get_if<RuntimeOutput>(&sequential_next)) {
            CHECK(actual->logits == expected->logits);
            CHECK(actual->token_history == std::vector<uint32_t>({0, 1, 0, 1}));
        }
    }

    auto failed_created = create_product_runtime_session_for_testing(fixture.closed_route, request());
    CHECK(std::holds_alternative<RuntimeSession>(failed_created));
    if (!std::holds_alternative<RuntimeSession>(failed_created)) return;
    RuntimeSession failed = std::get<RuntimeSession>(std::move(failed_created));
    failed.fail_after_prefill_tokens_for_testing(1);
    auto failed_output = failed.prefill(prompt);
    CHECK(std::holds_alternative<CompatibilityReport>(failed_output));
    CHECK(failed.token_history().empty());
    auto retry = failed.prefill(prompt);
    CHECK(std::holds_alternative<RuntimeOutput>(retry));
    CHECK(failed.token_history() == prompt);
    CHECK(failed.implicit_weight_copy_count_for_testing() == 0);
}

void test_recurrent_product_device_sampler(const ProductFixture& fixture) {
    if (!fixture.closed_route || !metal_device_present()) {
        if (std::getenv("LAPLACE_TEST_REQUIRE_NATIVE") != nullptr) {
            CHECK_MSG(false, "native recurrent product Metal sampler execution is required");
            return;
        }
        std::printf("SKIP: recurrent product Metal sampler device unavailable\n");
        return;
    }
    auto control_created = create_product_runtime_session_for_testing(fixture.closed_route, request());
    auto sampled_created = create_product_runtime_session_for_testing(fixture.closed_route, request());
    if (const auto* report = std::get_if<CompatibilityReport>(&control_created))
        CHECK_MSG(false, "recurrent control sampler session construction failed: code=%u operator=%u tensor=%u detail=%s",
                  static_cast<unsigned>(report->code), report->operator_id,
                  report->tensor_id, report->detail.c_str());
    if (const auto* report = std::get_if<CompatibilityReport>(&sampled_created))
        CHECK_MSG(false, "recurrent sampled session construction failed: code=%u operator=%u tensor=%u detail=%s",
                  static_cast<unsigned>(report->code), report->operator_id,
                  report->tensor_id, report->detail.c_str());
    CHECK(std::holds_alternative<RuntimeSession>(control_created));
    CHECK(std::holds_alternative<RuntimeSession>(sampled_created));
    if (!std::holds_alternative<RuntimeSession>(control_created) ||
        !std::holds_alternative<RuntimeSession>(sampled_created)) return;
    RuntimeSession control = std::get<RuntimeSession>(std::move(control_created));
    RuntimeSession sampled = std::get<RuntimeSession>(std::move(sampled_created));
    const std::vector<uint32_t> prompt = {0, 1, 0};
    auto control_output = control.prefill(prompt);
    auto sampled_output = sampled.prefill_sampled(prompt);
    CHECK(std::holds_alternative<RuntimeOutput>(control_output));
    CHECK(std::holds_alternative<RuntimeOutput>(sampled_output));
    if (const auto* logits = std::get_if<RuntimeOutput>(&control_output)) {
        if (const auto* token = std::get_if<RuntimeOutput>(&sampled_output)) {
            CHECK(token->sampled);
            CHECK(token->logits.empty());
            CHECK(token->host_result_bytes == 16);
            CHECK(token->command_buffers == prompt.size());
            CHECK(token->sampled_token_id == first_argmax(logits->logits));
            CHECK(token->sampled_logit == logits->logits[token->sampled_token_id]);
        }
    }
    CHECK(sampled.token_history() == prompt);
    CHECK(sampled.implicit_weight_copy_count_for_testing() == 0);
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::strcmp(argv[1], "--emit-program-ingress") == 0)
        return emit_program_ingress_fixture(argv[2], argv[3]) ? 0 : 1;
    if (argc == 2 && std::strcmp(argv[1], "--program-session") == 0) {
        test_program_ingress_creates_runtime_session();
        test_verified_program_package_creates_runtime_session();
        test_verified_program_package_creates_runtime_session(true);
        test_stateful_program_package_publishes_gpu_state();
        test_program_sequence_failure_does_not_publish_state();
        return test_summary("test_runtime_session_product");
    }
    Storage dense_storage;
    ProductFixture dense = make_closed_route_fixture(make_dense_model(dense_storage), std::move(dense_storage));
    if (argc == 2 && std::strcmp(argv[1], "--product-planner") == 0) {
        test_product_codec_preflight_red(dense);
        cleanup(dense);
        return test_summary("test_runtime_session_product");
    }
    test_closed_route_authority_plumbing(dense);
    test_runtime_carries_verified_physical_package(dense);
    test_closed_route_rejects_invalid_compiler_identity(dense);
    test_product_request_contract_fails_before_device_use(dense);
    test_unqualified_derived_weights_are_not_product_authoritative(dense);
    test_dense_product_session(dense);
    test_dense_product_two_row_prefill_matches_sequential(dense);
    test_dense_product_multitoken_prefill_schedules_state_only_programs(dense);
    test_product_session_move_construction(dense);
    test_product_session_move_assignment(dense);
    test_dense_product_device_sampler(dense);
    cleanup(dense);

    Storage affine_storage;
    ProductFixture affine = make_closed_route_fixture(make_dense_model(affine_storage, 512, true),
                                                      std::move(affine_storage));
    test_closed_route_authority_plumbing(affine);
    test_dense_product_session(affine, true);
    cleanup(affine);

    Storage recurrent_storage;
    ProductFixture recurrent = make_closed_route_fixture(make_recurrent_model(recurrent_storage),
                                                         std::move(recurrent_storage));
    test_closed_route_authority_plumbing(recurrent);
    test_recurrent_rollback_is_closed(recurrent);
    test_recurrent_prefill_chains_and_rolls_back(recurrent);
    test_recurrent_product_device_sampler(recurrent);
    cleanup(recurrent);
    return test_summary("test_runtime_session_product");
}
