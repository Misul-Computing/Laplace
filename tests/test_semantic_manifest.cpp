#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <optional>
#include <string>
#include <unistd.h>
#include <vector>

#include "semantic_manifest.h"
#include "codec_certificate.h"
#include "execution_plan.h"
#include "test_util.h"

using namespace Laplace;

namespace {

void put16(std::vector<uint8_t>& bytes, size_t offset, uint16_t value) {
    bytes[offset] = static_cast<uint8_t>(value);
    bytes[offset + 1] = static_cast<uint8_t>(value >> 8);
}

void put64(std::vector<uint8_t>& bytes, size_t offset, uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8) bytes[offset + shift / 8] = static_cast<uint8_t>(value >> shift);
}

uint64_t get64(const std::vector<uint8_t>& bytes, size_t offset) {
    uint64_t result = 0;
    for (unsigned shift = 0; shift != 64; shift += 8) result |= static_cast<uint64_t>(bytes[offset + shift / 8]) << shift;
    return result;
}

uint32_t get32(const std::vector<uint8_t>& bytes, size_t offset) {
    return static_cast<uint32_t>(bytes[offset]) |
           (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

struct ManifestBodySpans {
    size_t body_end = 0;
    std::vector<size_t> artifact_records;
    size_t codec_digest = 0;
    size_t token_digest = 0;
    size_t semantic_length = 0;
    size_t semantic_payload = 0;
    size_t codec_registry_length = 0;
    size_t codec_registry_payload = 0;
};

std::optional<ManifestBodySpans> locate_manifest_body_spans(
    const std::vector<uint8_t>& bytes) {
    constexpr size_t envelope_header = 64;
    constexpr size_t envelope_digest = 32;
    constexpr size_t artifact_record = 48;
    if (bytes.size() < envelope_header + envelope_digest ||
        get64(bytes, 16) > bytes.size() - envelope_header - envelope_digest) {
        return std::nullopt;
    }
    ManifestBodySpans spans;
    const size_t body_start = envelope_header;
    spans.body_end = body_start + static_cast<size_t>(get64(bytes, 16));
    size_t cursor = body_start;
    const auto take = [&](size_t count) {
        if (count > spans.body_end - cursor) return false;
        cursor += count;
        return true;
    };
    if (!take(8)) return std::nullopt;
    const uint32_t artifact_count = get32(bytes, cursor - 4);
    if (artifact_count > (spans.body_end - cursor) / artifact_record) return std::nullopt;
    spans.artifact_records.reserve(artifact_count);
    for (uint32_t index = 0; index != artifact_count; ++index) {
        spans.artifact_records.push_back(cursor);
        if (!take(artifact_record)) return std::nullopt;
    }
    if (!take(4 * 32)) return std::nullopt;
    spans.codec_digest = cursor;
    if (!take(32)) return std::nullopt;
    if (!take(20)) return std::nullopt;
    const size_t stop_count_offset = cursor - 4;
    const uint32_t stop_count = get32(bytes, stop_count_offset);
    if (stop_count > (spans.body_end - cursor) / sizeof(uint32_t) ||
        !take(static_cast<size_t>(stop_count) * sizeof(uint32_t))) {
        return std::nullopt;
    }
    if (!take(64)) return std::nullopt;
    spans.token_digest = cursor;
    if (!take(32)) return std::nullopt;
    spans.semantic_length = cursor;
    if (!take(8)) return std::nullopt;
    spans.semantic_payload = cursor;
    const uint64_t semantic_length = get64(bytes, spans.semantic_length);
    if (semantic_length > spans.body_end - cursor || !take(static_cast<size_t>(semantic_length))) {
        return std::nullopt;
    }
    if (!take(8)) return std::nullopt;
    const uint64_t token_authority_length = get64(bytes, cursor - 8);
    if (token_authority_length > spans.body_end - cursor ||
        !take(static_cast<size_t>(token_authority_length))) return std::nullopt;
    spans.codec_registry_length = cursor;
    if (!take(8)) return std::nullopt;
    const uint64_t codec_registry_length = get64(bytes, spans.codec_registry_length);
    spans.codec_registry_payload = cursor;
    if (codec_registry_length > spans.body_end - cursor ||
        !take(static_cast<size_t>(codec_registry_length)) || cursor != spans.body_end) {
        return std::nullopt;
    }
    return spans;
}

Sha256Digest sha256(std::span<const uint8_t> bytes) {
    Sha256Digest result;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), result.bytes.data());
    return result;
}

void reseal(std::vector<uint8_t>& bytes) {
    const size_t body_length = static_cast<size_t>(get64(bytes, 16));
    const Sha256Digest body = sha256(std::span<const uint8_t>(bytes).subspan(64, body_length));
    std::copy(body.bytes.begin(), body.bytes.end(), bytes.begin() + 32);
    const Sha256Digest record = sha256(std::span<const uint8_t>(bytes).first(bytes.size() - 32));
    std::copy(record.bytes.begin(), record.bytes.end(), bytes.end() - 32);
}

std::string temporary_path() {
    char path[] = "/private/tmp/laplace-manifest-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd >= 0) close(fd);
    return path;
}

struct Fixture {
    std::string path;
    ArtifactIndex index;
    SemanticModel model;
    TokenIdContract contract;
};

PhysicalCodecRegistry raw_f32_codec_registry(const SemanticModel& model) {
    PhysicalCodecRegistry registry;
    if (model.tensors.size() != 1) return registry;
    const std::vector<uint8_t> bytes = make_raw_f32_codec_certificate();
    const CodecCertificateParseResult parsed = parse_codec_certificate(bytes);
    const auto* certificate = std::get_if<CodecCertificate>(&parsed);
    CHECK(certificate != nullptr);
    if (!certificate) return registry;
    const auto identity = physical_codec_identity(
        model.tensors[0], certificate->identity().abi_version,
        certificate->identity().digest);
    CHECK(identity.has_value());
    if (!identity) return registry;
    registry.codecs.push_back({*identity, bytes});
    registry.tensors.push_back({model.tensors[0].id, *identity});
    return registry;
}

struct MultiFixture {
    std::array<std::string, 2> paths;
    ArtifactIndex index;
    SemanticModel model;
    TokenIdContract contract;
};

struct TiedFixture {
    std::string path;
    ArtifactIndex index;
    SemanticModel model;
    TokenIdContract contract;
};

Fixture make_fixture(bool quantized, uint64_t offset = 64, bool multi_plane = false,
                     bool reverse_planes = false, bool transposed = false) {
    Fixture fixture;
    fixture.path = temporary_path();
    std::vector<uint8_t> source(1024, 0x5a);
    const int fd = open(fixture.path.c_str(), O_WRONLY | O_TRUNC);
    CHECK(fd >= 0);
    if (fd >= 0) {
        CHECK(write(fd, source.data(), source.size()) == static_cast<ssize_t>(source.size()));
        close(fd);
    }
    auto loaded = ArtifactSet::load_single_file(fixture.path);
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    std::vector<PackageView> artifacts;
    if (auto* set = std::get_if<ArtifactSet>(&loaded)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) artifacts.push_back(*package);
    }

    ArtifactTensorRecord physical;
    physical.id = 0;
    physical.logical_type = ArtifactScalarType::F32;
    physical.logical_dimensions = quantized ? std::vector<uint64_t>{256, 1} :
        (transposed ? std::vector<uint64_t>{3, 2} : std::vector<uint64_t>{2, 2});
    physical.coordinate.root = 0;
    physical.layout.rank = 2;
    physical.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    physical.format.version = 1;
    physical.quantization.required_plane_mask = quantized ? artifact_plane_mask(PlaneKind::Values) : 0;
    ArtifactTensorPlane plane;
    plane.kind = PlaneKind::Values;
    plane.source = {ArtifactId{0}, offset, quantized ? 144u : 16u};
    plane.logical_elements = quantized ? 256 : 4;
    plane.alignment = 32;
    if (quantized) {
        physical.layout.kind = PhysicalLayoutKind::GgufBlocked;
        physical.layout.packing = PackingKind::Gguf;
        physical.layout.block_rank = 1;
        physical.layout.strides = {1, 256, 0, 0, 0, 0, 0, 0};
        physical.layout.block_elements = 256;
        physical.layout.block_bytes = 144;
        physical.axis.source_rank = 2;
        physical.axis.source_axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        physical.axis.block_axis = 0;
        physical.axis.block_elements = 256;
        physical.axis.bytes_per_block = 144;
        physical.axis.row_stride_bytes = 144;
        physical.format = {1, ArtifactPhysicalEncoding::Q4_K, ArtifactScalarType::Packed,
                           ArtifactScalarType::F16, ArtifactScalarType::F16, ArtifactScalarType::Packed,
                           ArtifactScalarType::None, 256, 144, 2, 2, 12, 0};
        physical.quantization.kind = QuantizationKind::BlockedAffine;
        physical.quantization.scale_type = ScalarType::F16;
        physical.quantization.zero_type = ScalarType::F16;
        physical.quantization.block_elements = 256;
        physical.quantization.block_bytes = 144;
        physical.quantization.group_size = 256;
        physical.quantization.required_plane_mask = artifact_plane_mask(PlaneKind::Values) |
            (multi_plane ? artifact_plane_mask(PlaneKind::Scales) : 0);
        plane.storage_type = ArtifactScalarType::Packed;
        plane.bytes_per_block = 144;
        plane.elements_per_block = 256;
    } else {
        physical.layout.strides = transposed
            ? std::array<uint64_t, 8>{1, 3, 0, 0, 0, 0, 0, 0}
            : std::array<uint64_t, 8>{2, 1, 0, 0, 0, 0, 0, 0};
        physical.format = {1, ArtifactPhysicalEncoding::F32, ArtifactScalarType::F32,
                           ArtifactScalarType::None, ArtifactScalarType::None, ArtifactScalarType::None,
                           ArtifactScalarType::None, 1, 4, 0, 0, 0, 0};
        physical.axis.row_stride_bytes = transposed ? 12 : 8;
        if (transposed) {
            physical.axis.source_rank = 2;
            physical.axis.source_axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        }
        plane.storage_type = ArtifactScalarType::F32;
        plane.bytes_per_block = 4;
        plane.elements_per_block = 1;
    }
    if (transposed) {
        physical.planes.clear();
    }
    if (transposed && !quantized) {
        plane.source.length = 24;
        plane.logical_elements = 6;
    }
    physical.planes.push_back(plane);
    if (quantized && multi_plane) {
        ArtifactTensorPlane scales;
        scales.kind = PlaneKind::Scales;
        scales.storage_type = ArtifactScalarType::F16;
        scales.source = {ArtifactId{0}, 224, 16};
        scales.logical_elements = 256;
        scales.alignment = 32;
        scales.bytes_per_block = 2;
        scales.elements_per_block = 32;
        physical.planes.push_back(scales);
    }
    ArtifactIndexInput input;
    input.artifacts = artifacts;
    input.tensors.push_back(physical);
    auto built = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<ArtifactIndex>(built));
    if (auto* result = std::get_if<ArtifactIndex>(&built)) fixture.index = std::move(*result);
    else if (auto* report = std::get_if<CompatibilityReport>(&built)) {
        fprintf(stderr, "fixture index error: %u %s\n", static_cast<unsigned>(report->code), report->detail.c_str());
    }

    SemanticTensor semantic;
    semantic.id = 0;
    semantic.role = TensorRole::TokenEmbedding;
    semantic.logical_type = ScalarType::F32;
    semantic.dimensions = quantized
        ? std::vector<Dimension>{{DimensionKind::Constant, 256}, {DimensionKind::Constant, 1}}
        : (transposed ? std::vector<Dimension>{{DimensionKind::Constant, 2}, {DimensionKind::Constant, 3}}
                      : std::vector<Dimension>{{DimensionKind::Constant, 2}, {DimensionKind::Constant, 2}});
    semantic.layout = physical.layout;
    if (transposed) semantic.layout.axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    semantic.quantization = physical.quantization;
    TensorPlane semantic_plane;
    semantic_plane.kind = PlaneKind::Values;
    semantic_plane.storage_type = quantized ? ScalarType::U8 : ScalarType::F32;
    semantic_plane.artifact_id = ArtifactId{0};
    semantic_plane.offset = offset;
    semantic_plane.length = quantized ? 144 : (transposed ? 24 : 16);
    semantic_plane.alignment = 32;
    TensorPlane semantic_scales;
    semantic_scales.kind = PlaneKind::Scales;
    semantic_scales.storage_type = ScalarType::F16;
    semantic_scales.artifact_id = ArtifactId{0};
    semantic_scales.offset = 224;
    semantic_scales.length = 16;
    semantic_scales.alignment = 32;
    if (quantized && multi_plane && reverse_planes) semantic.planes.push_back(semantic_scales);
    semantic.planes.push_back(semantic_plane);
    if (quantized && multi_plane && !reverse_planes) semantic.planes.push_back(semantic_scales);
    fixture.model.maximum_context = 32768;
    fixture.model.vocabulary_size = 32;
    fixture.model.bos_id = 1;
    fixture.model.eos_id = 2;
    fixture.model.stop_ids = {2};
    for (size_t i = 0; i != fixture.model.tokenizer_digest.size(); ++i) fixture.model.tokenizer_digest[i] = static_cast<uint8_t>(i + 1);
    fixture.model.tensors.push_back(semantic);
    fixture.contract.vocabulary_size = fixture.model.vocabulary_size;
    fixture.contract.bos_id = fixture.model.bos_id;
    fixture.contract.eos_id = fixture.model.eos_id;
    fixture.contract.stop_ids = fixture.model.stop_ids;
    fixture.contract.authoritative_tokenizer_digest = {fixture.model.tokenizer_digest};
    fixture.contract.authoritative_template_digest = {fixture.model.template_digest};
    return fixture;
}

Fixture make_column_grouped_u2_fixture() {
    Fixture fixture;
    fixture.path = temporary_path();
    std::vector<uint8_t> source(2048, 0x5a);
    const int fd = open(fixture.path.c_str(), O_WRONLY | O_TRUNC);
    CHECK(fd >= 0);
    if (fd >= 0) {
        CHECK(write(fd, source.data(), source.size()) == static_cast<ssize_t>(source.size()));
        close(fd);
    }
    auto loaded = ArtifactSet::load_single_file(fixture.path);
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    std::vector<PackageView> artifacts;
    if (auto* set = std::get_if<ArtifactSet>(&loaded)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) artifacts.push_back(*package);
    }

    ArtifactTensorRecord physical;
    physical.id = 0;
    physical.logical_type = ArtifactScalarType::F32;
    physical.logical_dimensions = {256, 8};
    physical.layout.kind = PhysicalLayoutKind::ColumnGroupedAffineUInt2Skip;
    physical.layout.packing = PackingKind::LsbBitPacked;
    physical.layout.rank = 2;
    physical.layout.block_rank = 1;
    physical.layout.axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    physical.layout.strides = {1, 8, 0, 0, 0, 0, 0, 0};
    physical.layout.block_elements = 256;
    physical.layout.block_bytes = 64;
    physical.axis.source_rank = 2;
    physical.axis.source_axis_order = {1, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    physical.axis.block_axis = 0;
    physical.axis.block_elements = 256;
    physical.axis.bytes_per_block = 64;
    physical.axis.row_stride_bytes = 512;
    physical.format = {1, ArtifactPhysicalEncoding::ColumnGroupedAffineU2Skip256,
                       ArtifactScalarType::U8, ArtifactScalarType::F16,
                       ArtifactScalarType::None, ArtifactScalarType::None,
                       ArtifactScalarType::None, 256, 64, 2, 0, 0, 0,
                       ArtifactScalarType::F16, 2};
    physical.quantization.kind = QuantizationKind::BlockedAffine;
    physical.quantization.accumulation_type = ScalarType::F32;
    physical.quantization.scale_type = ScalarType::F16;
    physical.quantization.bias_type = ScalarType::F16;
    physical.quantization.block_elements = 256;
    physical.quantization.block_bytes = 64;
    physical.quantization.group_size = 256;
    physical.quantization.required_plane_mask = 7;
    physical.planes = {
        {PlaneKind::Values, ArtifactScalarType::U8, {ArtifactId{0}, 128, 512},
         2048, 64, 256, 128},
        {PlaneKind::Scales, ArtifactScalarType::F16, {ArtifactId{0}, 768, 16},
         8, 2, 1, 128},
        {PlaneKind::Biases, ArtifactScalarType::F16, {ArtifactId{0}, 896, 16},
         8, 2, 1, 128},
    };
    ArtifactIndexInput input;
    input.artifacts = artifacts;
    input.tensors.push_back(physical);
    auto built = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<ArtifactIndex>(built));
    if (auto* index = std::get_if<ArtifactIndex>(&built)) fixture.index = std::move(*index);

    SemanticTensor semantic;
    semantic.id = 0;
    semantic.role = TensorRole::TokenEmbedding;
    semantic.logical_type = ScalarType::F32;
    semantic.dimensions = {{DimensionKind::Constant, 256}, {DimensionKind::Constant, 8}};
    semantic.layout = physical.layout;
    semantic.quantization = physical.quantization;
    semantic.planes = {
        {PlaneKind::Values, ScalarType::U8, ArtifactId{0}, 128, 512, 128, 0},
        {PlaneKind::Scales, ScalarType::F16, ArtifactId{0}, 768, 16, 128, 0},
        {PlaneKind::Biases, ScalarType::F16, ArtifactId{0}, 896, 16, 128, 0},
    };
    fixture.model.maximum_context = 32768;
    fixture.model.vocabulary_size = 32;
    fixture.model.bos_id = 1;
    fixture.model.eos_id = 2;
    fixture.model.stop_ids = {2};
    for (size_t index = 0; index != fixture.model.tokenizer_digest.size(); ++index)
        fixture.model.tokenizer_digest[index] = static_cast<uint8_t>(index + 1);
    fixture.model.tensors.push_back(std::move(semantic));
    fixture.contract.vocabulary_size = fixture.model.vocabulary_size;
    fixture.contract.bos_id = fixture.model.bos_id;
    fixture.contract.eos_id = fixture.model.eos_id;
    fixture.contract.stop_ids = fixture.model.stop_ids;
    fixture.contract.authoritative_tokenizer_digest = {fixture.model.tokenizer_digest};
    fixture.contract.authoritative_template_digest = {fixture.model.template_digest};
    return fixture;
}

MultiFixture make_multi_fixture() {
    MultiFixture fixture;
    std::array<std::vector<uint8_t>, 2> sources = {
        std::vector<uint8_t>(1024, 0x31), std::vector<uint8_t>(1024, 0x72)};
    std::array<ArtifactSource, 2> source_descriptors{};
    for (size_t index = 0; index != fixture.paths.size(); ++index) {
        fixture.paths[index] = temporary_path();
        const int fd = open(fixture.paths[index].c_str(), O_WRONLY | O_TRUNC);
        CHECK(fd >= 0);
        if (fd >= 0) {
            CHECK(write(fd, sources[index].data(), sources[index].size()) ==
                  static_cast<ssize_t>(sources[index].size()));
            close(fd);
        }
        source_descriptors[index] = {fixture.paths[index],
                                     index == 0 ? ArtifactRole::Primary : ArtifactRole::Shard,
                                     ArtifactId{static_cast<uint32_t>(index)}};
    }
    auto loaded = ArtifactSet::load_graph(source_descriptors);
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    std::vector<PackageView> artifacts;
    if (auto* set = std::get_if<ArtifactSet>(&loaded)) {
        for (uint32_t id = 0; id != 2; ++id) {
            auto view = set->view(ArtifactId{id});
            CHECK(std::holds_alternative<PackageView>(view));
            if (auto* package = std::get_if<PackageView>(&view)) artifacts.push_back(*package);
        }
    }
    ArtifactIndexInput input;
    input.artifacts = artifacts;
    for (uint32_t id = 0; id != 2; ++id) {
        ArtifactTensorRecord tensor;
        tensor.id = id;
        tensor.logical_type = ArtifactScalarType::F32;
        tensor.logical_dimensions = {2, 2};
        tensor.coordinate.root = id;
        tensor.layout.rank = 2;
        tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        tensor.layout.strides = {2, 1, 0, 0, 0, 0, 0, 0};
        tensor.format = {1, ArtifactPhysicalEncoding::F32, ArtifactScalarType::F32,
                         ArtifactScalarType::None, ArtifactScalarType::None,
                         ArtifactScalarType::None, ArtifactScalarType::None, 1, 4, 0, 0, 0, 0};
        tensor.axis.row_stride_bytes = 8;
        ArtifactTensorPlane plane;
        plane.kind = PlaneKind::Values;
        plane.storage_type = ArtifactScalarType::F32;
        plane.source = {ArtifactId{id}, 64, 16};
        plane.logical_elements = 4;
        plane.bytes_per_block = 4;
        plane.elements_per_block = 1;
        plane.alignment = 32;
        tensor.planes.push_back(plane);
        input.tensors.push_back(tensor);

        SemanticTensor semantic;
        semantic.id = id;
        semantic.role = id == 0 ? TensorRole::TokenEmbedding : TensorRole::FinalNormWeight;
        semantic.logical_type = ScalarType::F32;
        semantic.dimensions = {{DimensionKind::Constant, 2}, {DimensionKind::Constant, 2}};
        semantic.layout = tensor.layout;
        semantic.quantization = tensor.quantization;
        TensorPlane semantic_plane;
        semantic_plane.kind = PlaneKind::Values;
        semantic_plane.storage_type = ScalarType::F32;
        semantic_plane.artifact_id = ArtifactId{id};
        semantic_plane.offset = 64;
        semantic_plane.length = 16;
        semantic_plane.alignment = 32;
        semantic.planes.push_back(semantic_plane);
        fixture.model.tensors.push_back(semantic);
    }
    auto built = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<ArtifactIndex>(built));
    if (auto* result = std::get_if<ArtifactIndex>(&built)) fixture.index = std::move(*result);

    fixture.model.maximum_context = 32768;
    fixture.model.vocabulary_size = 32;
    fixture.model.bos_id = 1;
    fixture.model.eos_id = 2;
    fixture.model.stop_ids = {2};
    for (size_t index = 0; index != fixture.model.tokenizer_digest.size(); ++index) {
        fixture.model.tokenizer_digest[index] = static_cast<uint8_t>(index + 1);
    }
    fixture.contract.vocabulary_size = fixture.model.vocabulary_size;
    fixture.contract.bos_id = fixture.model.bos_id;
    fixture.contract.eos_id = fixture.model.eos_id;
    fixture.contract.stop_ids = fixture.model.stop_ids;
    fixture.contract.authoritative_tokenizer_digest = {fixture.model.tokenizer_digest};
    fixture.contract.authoritative_template_digest = {fixture.model.template_digest};
    return fixture;
}

TiedFixture make_tied_fixture() {
    TiedFixture fixture;
    fixture.path = temporary_path();
    std::vector<uint8_t> source(1024, 0x44);
    const int fd = open(fixture.path.c_str(), O_WRONLY | O_TRUNC);
    CHECK(fd >= 0);
    if (fd >= 0) {
        CHECK(write(fd, source.data(), source.size()) == static_cast<ssize_t>(source.size()));
        close(fd);
    }
    auto loaded = ArtifactSet::load_single_file(fixture.path);
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    std::vector<PackageView> artifacts;
    if (auto* set = std::get_if<ArtifactSet>(&loaded)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) artifacts.push_back(*package);
    }

    ArtifactIndexInput input;
    input.artifacts = artifacts;
    for (uint32_t id = 0; id != 2; ++id) {
        ArtifactTensorRecord tensor;
        tensor.id = id;
        tensor.logical_type = ArtifactScalarType::F32;
        tensor.logical_dimensions = {2, 2};
        tensor.coordinate.root = id;
        tensor.layout.rank = 2;
        tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        tensor.layout.strides = {2, 1, 0, 0, 0, 0, 0, 0};
        tensor.format = {1, ArtifactPhysicalEncoding::F32, ArtifactScalarType::F32,
                         ArtifactScalarType::None, ArtifactScalarType::None,
                         ArtifactScalarType::None, ArtifactScalarType::None, 1, 4, 0, 0, 0, 0};
        tensor.axis.row_stride_bytes = 8;
        ArtifactTensorPlane plane;
        plane.kind = PlaneKind::Values;
        plane.storage_type = ArtifactScalarType::F32;
        plane.source = {ArtifactId{0}, 64, 16};
        plane.logical_elements = 4;
        plane.bytes_per_block = 4;
        plane.elements_per_block = 1;
        plane.alignment = 32;
        tensor.planes.push_back(plane);
        if (id == 1) tensor.role_evidence.push_back({TensorRole::OutputWeight, CanonicalFactKey{7},
                                                       ArtifactFactAuthority::Structural, {}});
        input.tensors.push_back(tensor);

        SemanticTensor semantic;
        semantic.id = id;
        semantic.role = id == 0 ? TensorRole::TokenEmbedding : TensorRole::OutputWeight;
        semantic.logical_type = ScalarType::F32;
        semantic.dimensions = {{DimensionKind::Constant, 2}, {DimensionKind::Constant, 2}};
        semantic.layout = tensor.layout;
        semantic.quantization = tensor.quantization;
        TensorPlane semantic_plane;
        semantic_plane.kind = PlaneKind::Values;
        semantic_plane.storage_type = ScalarType::F32;
        semantic_plane.artifact_id = ArtifactId{0};
        semantic_plane.offset = 64;
        semantic_plane.length = 16;
        semantic_plane.alignment = 32;
        semantic.planes.push_back(semantic_plane);
        fixture.model.tensors.push_back(semantic);
    }
    input.aliases.push_back({ArtifactAliasKind::TiedOutput, ArtifactAliasDirection::SourceToTarget,
                             0, 1, TensorRole::OutputWeight, CanonicalFactKey{7},
                             ArtifactFactAuthority::Structural, {}});
    auto built = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<ArtifactIndex>(built));
    if (auto* result = std::get_if<ArtifactIndex>(&built)) {
        fixture.index = std::move(*result);
        CHECK(fixture.index.aliases().size() == 1);
    }

    fixture.model.maximum_context = 32768;
    fixture.model.vocabulary_size = 32;
    fixture.model.bos_id = 1;
    fixture.model.eos_id = 2;
    fixture.model.stop_ids = {2};
    for (size_t index = 0; index != fixture.model.tokenizer_digest.size(); ++index) {
        fixture.model.tokenizer_digest[index] = static_cast<uint8_t>(index + 1);
    }
    fixture.contract.vocabulary_size = fixture.model.vocabulary_size;
    fixture.contract.bos_id = fixture.model.bos_id;
    fixture.contract.eos_id = fixture.model.eos_id;
    fixture.contract.stop_ids = fixture.model.stop_ids;
    fixture.contract.authoritative_tokenizer_digest = {fixture.model.tokenizer_digest};
    fixture.contract.authoritative_template_digest = {fixture.model.template_digest};
    return fixture;
}

void remove_fixture(Fixture& fixture) {
    unlink(fixture.path.c_str());
}

void remove_fixture(MultiFixture& fixture) {
    for (const std::string& path : fixture.paths) unlink(path.c_str());
}

void remove_fixture(TiedFixture& fixture) {
    unlink(fixture.path.c_str());
}

bool is_manifest(const SemanticManifestResult& result) {
    return std::holds_alternative<SemanticManifest>(result);
}

void test_product_token_authority_is_closure_bound() {
    Fixture fixture = make_fixture(false);
    TokenContract product = fixture.contract;
    product.tokenizer_algorithm = TokenizerAlgorithm::ByteBpe;
    product.tokenizer_version = 1;
    product.tokenizer_data.artifact_id = ArtifactId{0};
    product.tokenizer_data.offset = 8;
    product.tokenizer_data.length = 24;
    product.tokenizer_data.digest = sha256(fixture.index.artifacts()[0].bytes().subspan(8, 24));
    product.vocabulary_digest = product.tokenizer_data.digest;
    product.authoritative_tokenizer_digest = product.tokenizer_data.digest;
    fixture.model.tokenizer_digest = product.tokenizer_data.digest.bytes;
    product.prompt = PromptTemplate{
        1,
        {
            {PromptOperationKind::AppendLiteral, {'u', 's', 'e', 'r', ':'}, kNoTokenId},
            {PromptOperationKind::AppendInputText, {}, kNoTokenId},
        },
    };
    product.authoritative_template_digest.bytes[0] = 0x71;
    fixture.model.template_digest[0] = 0x71;

    auto built = SemanticManifest::build(fixture.index, fixture.model, product);
    CHECK(is_manifest(built));
    if (auto* manifest = std::get_if<SemanticManifest>(&built)) {
        CHECK(manifest->token_contract() == product);
        CHECK(manifest->token_contract().tokenizer_data.artifact_id == ArtifactId{0});
        CHECK(manifest->token_contract().prompt_mode() == TokenPromptMode::SerializedTemplate);
        auto decoded = SemanticManifest::decode(fixture.index, manifest->bytes());
        CHECK(is_manifest(decoded));
        if (auto* roundtrip = std::get_if<SemanticManifest>(&decoded)) {
            CHECK(roundtrip->token_contract() == product);
        }

        std::vector<uint8_t> tampered(manifest->bytes().begin(), manifest->bytes().end());
        tampered[tampered.size() - 33] ^= 1;
        reseal(tampered);
        auto rejected = SemanticManifest::decode(fixture.index, tampered);
        CHECK(std::holds_alternative<CompatibilityReport>(rejected));

        std::vector<uint8_t> digest_tampered(manifest->bytes().begin(), manifest->bytes().end());
        const auto codec_digest = manifest->physical_codec_registry_digest();
        const auto digest_start = std::search(digest_tampered.begin(), digest_tampered.end(),
                                              codec_digest.bytes.begin(), codec_digest.bytes.end());
        CHECK(digest_start != digest_tampered.end());
        if (digest_start != digest_tampered.end()) {
            *digest_start ^= 1;
            reseal(digest_tampered);
            auto digest_rejected = SemanticManifest::decode(fixture.index, digest_tampered);
            CHECK(std::holds_alternative<CompatibilityReport>(digest_rejected));
            if (const auto* report = std::get_if<CompatibilityReport>(&digest_rejected)) {
                CHECK(report->code == CompatibilityError::PACKAGE_CHECKSUM_MISMATCH);
            }
        }
    }

    TokenContract out_of_closure = product;
    out_of_closure.tokenizer_data.offset = fixture.index.artifacts()[0].bytes().size();
    auto rejected = SemanticManifest::build(fixture.index, fixture.model, out_of_closure);
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    remove_fixture(fixture);
}

SemanticModel make_expert_digest_model() {
    SemanticModel model;
    model.schema_major = 7;
    model.schema_minor = 0;
    model.opset_major = 7;
    model.opset_minor = 0;
    model.maximum_context = 1024;
    model.vocabulary_size = 32;
    model.bos_id = 1;
    model.eos_id = 2;

    SemanticTensor tensor;
    tensor.id = 0;
    tensor.role = TensorRole::FfnGateWeight;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, 2}, {DimensionKind::Constant, 2},
                         {DimensionKind::Constant, 2}, {DimensionKind::Constant, 2}};
    tensor.layout.rank = 4;
    tensor.layout.axis_order = {0, 1, 2, 3, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides = {8, 4, 2, 1, 0, 0, 0, 0};
    tensor.expert_axis = {ExpertAxisKind::ExpertBank, 0, 3, 1, 2, 2, 16, 0};
    tensor.planes.push_back({PlaneKind::Values, ScalarType::F32, ArtifactId{0}, 64, 32, 0});
    model.tensors.push_back(std::move(tensor));
    return model;
}

void test_golden_and_digests() {
    Fixture fixture = make_fixture(false);
    auto result = SemanticManifest::build(fixture.index, fixture.model, fixture.contract);
    CHECK(is_manifest(result));
    if (auto* report = std::get_if<CompatibilityReport>(&result)) fprintf(stderr, "manifest error: %u %s\n", static_cast<unsigned>(report->code), report->detail.c_str());
    if (auto* manifest = std::get_if<SemanticManifest>(&result)) {
        CHECK(manifest->bytes().size() > 96);
        CHECK(std::memcmp(manifest->bytes().data(), "LAPMAN01", 8) == 0);
        const size_t body_length = static_cast<size_t>(get64(std::vector<uint8_t>(manifest->bytes().begin(), manifest->bytes().end()), 16));
        CHECK(manifest->bytes().size() == 64 + body_length + 32);
        const Sha256Digest body = sha256(manifest->bytes().subspan(64, body_length));
        CHECK(body == manifest->body_digest());
        CHECK(sha256(manifest->bytes().first(manifest->bytes().size() - 32)) == manifest->record_digest());
        CHECK(manifest->artifact_id() == ArtifactId{0});
        CHECK(manifest->artifact_size() == 1024);
        CHECK(manifest->artifact_digest() == fixture.index.artifacts()[0].digest());
        CHECK(manifest->semantic_graph_digest() == manifest->semantic_digest());
        CHECK(manifest->interaction_contract_digest() != Sha256Digest{});
        CHECK(manifest->physical_binding_set_digest() != Sha256Digest{});
        CHECK(manifest->package_fingerprint() != Sha256Digest{});
        auto expected_semantic = encode_semantic_model(fixture.model);
        CHECK(std::holds_alternative<std::vector<uint8_t>>(expected_semantic));
        if (auto* expected = std::get_if<std::vector<uint8_t>>(&expected_semantic)) {
            CHECK(std::vector<uint8_t>(manifest->semantic_bytes().begin(), manifest->semantic_bytes().end()) == *expected);
        }
        CHECK(manifest->token_contract() == fixture.contract);
        CHECK(semantic_manifest_token_contract_digest(fixture.contract) ==
              semantic_manifest_token_contract_digest(manifest->token_contract()));

        std::vector<uint8_t> bytes(manifest->bytes().begin(), manifest->bytes().end());
        auto decoded = SemanticManifest::decode(fixture.index, bytes);
        CHECK(is_manifest(decoded));
        if (auto* roundtrip = std::get_if<SemanticManifest>(&decoded)) {
            CHECK(std::vector<uint8_t>(roundtrip->bytes().begin(), roundtrip->bytes().end()) == bytes);
            CHECK(std::vector<uint8_t>(roundtrip->semantic_bytes().begin(), roundtrip->semantic_bytes().end()) ==
                  std::vector<uint8_t>(manifest->semantic_bytes().begin(), manifest->semantic_bytes().end()));
            CHECK(roundtrip->physical_index().digest() == fixture.index.digest());
        }
    }
    remove_fixture(fixture);
}

void test_quantized_physical_binding() {
    Fixture fixture = make_fixture(true);
    auto result = SemanticManifest::build(fixture.index, fixture.model, fixture.contract);
    CHECK(is_manifest(result));
    if (auto* report = std::get_if<CompatibilityReport>(&result)) fprintf(stderr, "quant manifest error: %u %s\n", static_cast<unsigned>(report->code), report->detail.c_str());
    if (auto* manifest = std::get_if<SemanticManifest>(&result)) {
        CHECK(manifest->semantic_model().tensors[0].planes[0].storage_type == ScalarType::U8);
        CHECK(manifest->physical_index().tensors()[0].planes[0].storage_type == ArtifactScalarType::Packed);
        auto decoded = SemanticManifest::decode(fixture.index, manifest->bytes());
        CHECK(is_manifest(decoded));
    }
    remove_fixture(fixture);
}

void test_digest_and_boundary_rejections() {
    Fixture fixture = make_fixture(false);
    auto built = SemanticManifest::build(fixture.index, fixture.model, fixture.contract);
    CHECK(is_manifest(built));
    if (auto* report = std::get_if<CompatibilityReport>(&built)) fprintf(stderr, "boundary manifest error: %u %s\n", static_cast<unsigned>(report->code), report->detail.c_str());
    if (auto* manifest = std::get_if<SemanticManifest>(&built)) {
        std::vector<uint8_t> version(manifest->bytes().begin(), manifest->bytes().end());
        put16(version, 8, 2);
        auto bad_version = SemanticManifest::decode(fixture.index, version);
        CHECK(std::holds_alternative<CompatibilityReport>(bad_version));
        if (auto* report = std::get_if<CompatibilityReport>(&bad_version)) CHECK(report->code == CompatibilityError::IR_VERSION_UNSUPPORTED);

        std::vector<uint8_t> body(manifest->bytes().begin(), manifest->bytes().end());
        body[64] ^= 1;
        auto bad_body = SemanticManifest::decode(fixture.index, body);
        CHECK(std::holds_alternative<CompatibilityReport>(bad_body));
        if (auto* report = std::get_if<CompatibilityReport>(&bad_body)) CHECK(report->code == CompatibilityError::PACKAGE_CHECKSUM_MISMATCH);

        std::vector<uint8_t> record(manifest->bytes().begin(), manifest->bytes().end());
        record.back() ^= 1;
        auto bad_record = SemanticManifest::decode(fixture.index, record);
        CHECK(std::holds_alternative<CompatibilityReport>(bad_record));
        if (auto* report = std::get_if<CompatibilityReport>(&bad_record)) CHECK(report->code == CompatibilityError::PACKAGE_CHECKSUM_MISMATCH);

        std::vector<uint8_t> trailing(manifest->bytes().begin(), manifest->bytes().end());
        trailing.insert(trailing.end() - 32, 0);
        put64(trailing, 16, get64(trailing, 16) + 1);
        put64(trailing, 24, get64(trailing, 24) + 1);
        reseal(trailing);
        auto bad_trailing = SemanticManifest::decode(fixture.index, trailing);
        CHECK(std::holds_alternative<CompatibilityReport>(bad_trailing));

        // A one-byte mutation must never become an accepted record.  This is
        // intentionally independent of the parser's field offsets: every
        // mutation is checked through the public decoder and digest gates.
        const std::vector<uint8_t> original(manifest->bytes().begin(), manifest->bytes().end());
        for (size_t index = 0; index != original.size(); ++index) {
            std::vector<uint8_t> mutated = original;
            mutated[index] ^= 0x80;
            auto result = SemanticManifest::decode(fixture.index, mutated);
            CHECK(std::holds_alternative<CompatibilityReport>(result));
        }

        const auto body_spans = locate_manifest_body_spans(original);
        CHECK(body_spans.has_value());
        if (body_spans) {
            std::vector<uint8_t> bad_token_digest = original;
            bad_token_digest[body_spans->token_digest] ^= 1;
            reseal(bad_token_digest);
            auto bad_token_digest_result = SemanticManifest::decode(fixture.index, bad_token_digest);
            CHECK(std::holds_alternative<CompatibilityReport>(bad_token_digest_result));
            if (auto* report = std::get_if<CompatibilityReport>(&bad_token_digest_result)) {
                CHECK(report->code == CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED);
            }
        }

        auto under_token = fixture.contract;
        under_token.authoritative_tokenizer_digest = {};
        auto bad_token = SemanticManifest::build(fixture.index, fixture.model, under_token);
        CHECK(std::holds_alternative<CompatibilityReport>(bad_token));

        auto bad_model = fixture.model;
        bad_model.tensors[0].planes[0].offset += 32;
        auto bad_plane = SemanticManifest::build(fixture.index, bad_model, fixture.contract);
        CHECK(std::holds_alternative<CompatibilityReport>(bad_plane));
        if (auto* report = std::get_if<CompatibilityReport>(&bad_plane)) CHECK(report->code == CompatibilityError::IR_LAYOUT_MISMATCH);

        auto bad_id = fixture.model;
        bad_id.tensors[0].id = 1;
        auto bad_dangling = SemanticManifest::build(fixture.index, bad_id, fixture.contract);
        CHECK(std::holds_alternative<CompatibilityReport>(bad_dangling));

        auto bad_state = fixture.model;
        bad_state.values.push_back({0, ScalarType::F32, {{DimensionKind::Constant, 1}}, 0});
        bad_state.operators.push_back({0, OperatorKind::Add, 1, {0, 0}, {0}, {}, {9}, AddPayload{}});
        bad_state.layers.push_back({0, 0, 1, 0});
        auto bad_state_result = SemanticManifest::build(fixture.index, bad_state, fixture.contract);
        CHECK(std::holds_alternative<CompatibilityReport>(bad_state_result));
    }
    remove_fixture(fixture);
}

void test_identity_domains_are_separate() {
    Fixture original = make_fixture(false, 64);
    Fixture shifted = make_fixture(false, 96);
    Fixture interaction_changed = make_fixture(false, 64);
    interaction_changed.model.template_digest[0] = 0x7a;
    interaction_changed.contract.authoritative_template_digest = {interaction_changed.model.template_digest};
    auto first = SemanticManifest::build(original.index, original.model, original.contract);
    auto second = SemanticManifest::build(shifted.index, shifted.model, shifted.contract);
    auto third = SemanticManifest::build(interaction_changed.index, interaction_changed.model,
                                         interaction_changed.contract);
    CHECK(is_manifest(first));
    CHECK(is_manifest(second));
    CHECK(is_manifest(third));
    if (auto* left = std::get_if<SemanticManifest>(&first)) {
        if (auto* right = std::get_if<SemanticManifest>(&second)) {
            CHECK(left->semantic_graph_digest() == right->semantic_graph_digest());
            CHECK(left->interaction_contract_digest() == right->interaction_contract_digest());
            CHECK(left->physical_binding_set_digest() != right->physical_binding_set_digest());
            CHECK(left->package_fingerprint() != right->package_fingerprint());
        }
        if (auto* interaction = std::get_if<SemanticManifest>(&third)) {
            CHECK(left->semantic_graph_digest() == interaction->semantic_graph_digest());
            CHECK(left->physical_binding_set_digest() == interaction->physical_binding_set_digest());
            CHECK(left->interaction_contract_digest() != interaction->interaction_contract_digest());
            CHECK(left->package_fingerprint() != interaction->package_fingerprint());
        }
    }
    remove_fixture(original);
    remove_fixture(shifted);
    remove_fixture(interaction_changed);
}

void test_multi_artifact_closure() {
    MultiFixture fixture = make_multi_fixture();
    auto result = SemanticManifest::build(fixture.index, fixture.model, fixture.contract);
    CHECK(is_manifest(result));
    if (auto* report = std::get_if<CompatibilityReport>(&result)) {
        fprintf(stderr, "multi-artifact manifest error: %u %s\n",
                static_cast<unsigned>(report->code), report->detail.c_str());
    }
    if (auto* manifest = std::get_if<SemanticManifest>(&result)) {
        CHECK(manifest->physical_index().artifacts().size() == 2);
        CHECK(manifest->artifact_id() == ArtifactId{0});
        CHECK(manifest->semantic_model().tensors[1].planes[0].artifact_id == ArtifactId{1});
        auto decoded = SemanticManifest::decode(fixture.index, manifest->bytes());
        CHECK(is_manifest(decoded));
        if (auto* roundtrip = std::get_if<SemanticManifest>(&decoded)) {
            CHECK(roundtrip->semantic_model().tensors[1].planes[0].artifact_id == ArtifactId{1});
        }

        auto single = make_fixture(false);
        auto wrong_closure = SemanticManifest::decode(single.index, manifest->bytes());
        CHECK(std::holds_alternative<CompatibilityReport>(wrong_closure));
        if (auto* report = std::get_if<CompatibilityReport>(&wrong_closure)) {
            CHECK(report->code == CompatibilityError::PACKAGE_SOURCE_CHANGED);
        }
        remove_fixture(single);

        std::vector<uint8_t> reordered(manifest->bytes().begin(), manifest->bytes().end());
        const auto spans = locate_manifest_body_spans(reordered);
        CHECK(spans.has_value());
        if (spans && spans->artifact_records.size() == 2) {
            const size_t artifact_record_bytes =
                spans->artifact_records[1] - spans->artifact_records[0];
            for (size_t index = 0; index != artifact_record_bytes; ++index) {
                std::swap(reordered[spans->artifact_records[0] + index],
                          reordered[spans->artifact_records[1] + index]);
            }
        }
        reseal(reordered);
        auto bad_order = SemanticManifest::decode(fixture.index, reordered);
        CHECK(std::holds_alternative<CompatibilityReport>(bad_order));
        if (auto* report = std::get_if<CompatibilityReport>(&bad_order)) {
            CHECK(report->code == CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED);
        }
    }
    remove_fixture(fixture);
}

void test_tied_alias_roundtrip() {
    TiedFixture fixture = make_tied_fixture();
    auto result = SemanticManifest::build(fixture.index, fixture.model, fixture.contract);
    CHECK(is_manifest(result));
    if (auto* report = std::get_if<CompatibilityReport>(&result)) {
        fprintf(stderr, "tied alias manifest error: %u %s\n",
                static_cast<unsigned>(report->code), report->detail.c_str());
    }
    if (auto* manifest = std::get_if<SemanticManifest>(&result)) {
        CHECK(manifest->semantic_model().tensors.size() == 2);
        auto decoded = SemanticManifest::decode(fixture.index, manifest->bytes());
        CHECK(is_manifest(decoded));
    }

    auto mismatched_model = fixture.model;
    mismatched_model.tensors[1].role = TensorRole::FinalNormWeight;
    auto mismatched = SemanticManifest::build(fixture.index, mismatched_model, fixture.contract);
    CHECK(std::holds_alternative<CompatibilityReport>(mismatched));
    remove_fixture(fixture);
}

void test_resealed_field_mutation_reaches_parser() {
    Fixture fixture = make_fixture(false);
    auto built = SemanticManifest::build(fixture.index, fixture.model, fixture.contract);
    CHECK(is_manifest(built));
    if (auto* manifest = std::get_if<SemanticManifest>(&built)) {
        const std::vector<uint8_t> original(manifest->bytes().begin(), manifest->bytes().end());
        const auto spans = locate_manifest_body_spans(original);
        CHECK(spans.has_value());
        if (spans) {
            std::vector<uint8_t> semantic = original;
            semantic[spans->semantic_payload] ^= 1;
            reseal(semantic);
            auto semantic_result = SemanticManifest::decode(fixture.index, semantic);
            CHECK(std::holds_alternative<CompatibilityReport>(semantic_result));
            if (const auto* report = std::get_if<CompatibilityReport>(&semantic_result)) {
                CHECK(report->code == CompatibilityError::IR_VERSION_UNSUPPORTED);
            }

            std::vector<uint8_t> length = original;
            put64(length, spans->semantic_length,
                  static_cast<uint64_t>(spans->body_end - spans->semantic_payload + 1));
            reseal(length);
            auto length_result = SemanticManifest::decode(fixture.index, length);
            CHECK(std::holds_alternative<CompatibilityReport>(length_result));
            if (const auto* report = std::get_if<CompatibilityReport>(&length_result)) {
                CHECK(report->code == CompatibilityError::PACKAGE_BOUNDS_INVALID);
            }

            const PhysicalCodecRegistry registry =
                raw_f32_codec_registry(fixture.model);
            auto with_registry = SemanticManifest::build(
                fixture.index, fixture.model, fixture.contract, registry);
            CHECK(is_manifest(with_registry));
            if (auto* codec_manifest = std::get_if<SemanticManifest>(&with_registry)) {
                const std::vector<uint8_t> codec_original(
                    codec_manifest->bytes().begin(), codec_manifest->bytes().end());
                const auto codec_spans = locate_manifest_body_spans(codec_original);
                CHECK(codec_spans.has_value());
                if (codec_spans) {
                    CHECK(std::equal(
                        codec_original.begin() + static_cast<ptrdiff_t>(codec_spans->codec_digest),
                        codec_original.begin() + static_cast<ptrdiff_t>(codec_spans->codec_digest + 32),
                        codec_manifest->physical_codec_registry_digest().bytes.begin()));
                    std::vector<uint8_t> codec_digest = codec_original;
                    codec_digest[codec_spans->codec_digest] ^= 1;
                    reseal(codec_digest);
                    auto codec_digest_result = SemanticManifest::decode(
                        fixture.index, codec_digest);
                    CHECK(std::holds_alternative<CompatibilityReport>(codec_digest_result));
                    if (const auto* report = std::get_if<CompatibilityReport>(
                            &codec_digest_result)) {
                        CHECK(report->code == CompatibilityError::PACKAGE_CHECKSUM_MISMATCH);
                    }

                    std::vector<uint8_t> codec_payload = codec_original;
                    codec_payload[codec_spans->codec_registry_payload] ^= 1;
                    reseal(codec_payload);
                    auto codec_payload_result = SemanticManifest::decode(
                        fixture.index, codec_payload);
                    CHECK(std::holds_alternative<CompatibilityReport>(codec_payload_result));
                    if (const auto* report = std::get_if<CompatibilityReport>(
                            &codec_payload_result)) {
                        CHECK(report->code == CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED);
                    }
                }
            }
        }
    }
    remove_fixture(fixture);
}

void test_graph_digest_preserves_expert_identity_and_neutralizes_stride() {
    const SemanticModel base = make_expert_digest_model();
    auto base_result = semantic_manifest_graph_digest(base);
    CHECK(std::holds_alternative<Sha256Digest>(base_result));
    if (!std::holds_alternative<Sha256Digest>(base_result)) return;

    SemanticModel axis_changed = base;
    axis_changed.tensors[0].expert_axis.member_axis = 0xff;
    auto axis_result = semantic_manifest_graph_digest(axis_changed);
    CHECK(std::holds_alternative<Sha256Digest>(axis_result));
    if (std::holds_alternative<Sha256Digest>(axis_result)) {
        CHECK(std::get<Sha256Digest>(base_result) != std::get<Sha256Digest>(axis_result));
    }

    SemanticModel count_changed = base;
    count_changed.tensors[0].dimensions[0].constant_or_symbol = 3;
    count_changed.tensors[0].expert_axis.expert_count = 3;
    count_changed.tensors[0].planes[0].length = 48;
    auto count_result = semantic_manifest_graph_digest(count_changed);
    CHECK(std::holds_alternative<Sha256Digest>(count_result));
    if (std::holds_alternative<Sha256Digest>(count_result)) {
        CHECK(std::get<Sha256Digest>(base_result) != std::get<Sha256Digest>(count_result));
    }

    SemanticModel stride_changed = base;
    stride_changed.tensors[0].layout.strides[0] = 1234;
    stride_changed.tensors[0].expert_axis.per_expert_byte_stride = 9999;
    stride_changed.tensors[0].planes[0].length = 19998;
    auto stride_result = semantic_manifest_graph_digest(stride_changed);
    CHECK(std::holds_alternative<Sha256Digest>(stride_result));
    if (std::holds_alternative<Sha256Digest>(stride_result)) {
        CHECK(std::get<Sha256Digest>(base_result) == std::get<Sha256Digest>(stride_result));
    }

    SemanticModel symbolic = base;
    symbolic.tensors[0].dimensions[1] = {DimensionKind::Symbol, 17};
    auto symbolic_result = semantic_manifest_graph_digest(symbolic);
    CHECK(std::holds_alternative<Sha256Digest>(symbolic_result));

    SemanticModel invalid = base;
    invalid.tensors[0].dimensions[1].constant_or_symbol = 0;
    auto invalid_result = semantic_manifest_graph_digest(invalid);
    CHECK(std::holds_alternative<CompatibilityReport>(invalid_result));
    if (auto* report = std::get_if<CompatibilityReport>(&invalid_result)) {
        CHECK(report->code == CompatibilityError::IR_CONSTRAINT_FAILED);
    }
}

void test_plane_order_is_canonical() {
    Fixture canonical_fixture = make_fixture(true, 64, true, false);
    auto canonical = SemanticManifest::build(canonical_fixture.index, canonical_fixture.model,
                                             canonical_fixture.contract);
    CHECK(is_manifest(canonical));
    if (auto* manifest = std::get_if<SemanticManifest>(&canonical)) {
        // The two plane records in this fixture are fixed-width. Reorder the
        // encoded payload, refresh both digest layers, and require decode to
        // reject the alternate wire representation.
        std::vector<uint8_t> reordered(manifest->bytes().begin(), manifest->bytes().end());
        constexpr std::array<uint8_t, 8> lapir_magic = {'L', 'A', 'P', 'I', 'R', '0', '0', '1'};
        const auto semantic_magic = std::search(reordered.begin(), reordered.end(),
                                                 lapir_magic.begin(), lapir_magic.end());
        CHECK(semantic_magic != reordered.end());
        if (semantic_magic != reordered.end()) {
            const size_t semantic_start = static_cast<size_t>(semantic_magic - reordered.begin());
            constexpr size_t tensor_plane_start_delta = 376;
            constexpr size_t plane_record_bytes = 64;
            const size_t plane_start = semantic_start + tensor_plane_start_delta;
            for (size_t index = 0; index != plane_record_bytes; ++index) {
                std::swap(reordered[plane_start + index], reordered[plane_start + plane_record_bytes + index]);
            }
            const uint64_t body_length = get64(reordered, semantic_start + 16);
            const Sha256Digest semantic_body = sha256(
                std::span<const uint8_t>(reordered).subspan(semantic_start + 64,
                                                              static_cast<size_t>(body_length)));
            std::copy(semantic_body.bytes.begin(), semantic_body.bytes.end(),
                      reordered.begin() + semantic_start + 32);
            reseal(reordered);
            auto decoded = SemanticManifest::decode(canonical_fixture.index, reordered);
            CHECK(std::holds_alternative<CompatibilityReport>(decoded));
        }
    }
    remove_fixture(canonical_fixture);

    Fixture reversed_fixture = make_fixture(true, 64, true, true);
    auto reversed = SemanticManifest::build(reversed_fixture.index, reversed_fixture.model,
                                            reversed_fixture.contract);
    CHECK(std::holds_alternative<CompatibilityReport>(reversed));
    remove_fixture(reversed_fixture);
}

void test_manifest_envelope_accepts_supported_canonical_schema_versions() {
    Fixture fixture = make_fixture(false);
    fixture.model.schema_major = 3;
    fixture.model.schema_minor = 0;
    fixture.model.opset_major = 3;
    fixture.model.opset_minor = 0;
    fixture.model.maximum_context = 1024;

    auto built = SemanticManifest::build(fixture.index, fixture.model, fixture.contract);
    CHECK(is_manifest(built));
    if (auto* manifest = std::get_if<SemanticManifest>(&built)) {
        CHECK(manifest->semantic_model().schema_major == 3);
        CHECK(manifest->semantic_model().opset_major == 3);
        CHECK(manifest->semantic_model().maximum_context == 1024);
        auto decoded = SemanticManifest::decode(fixture.index, manifest->bytes());
        CHECK(is_manifest(decoded));
        if (auto* roundtrip = std::get_if<SemanticManifest>(&decoded)) {
            CHECK(roundtrip->semantic_model().schema_major == 3);
            CHECK(roundtrip->semantic_model().opset_major == 3);
            CHECK(roundtrip->semantic_model().maximum_context == 1024);
        }
    }
    remove_fixture(fixture);
}

void test_column_grouped_u2_manifest_binding_is_byte_typed() {
    Fixture fixture = make_column_grouped_u2_fixture();
    auto built = SemanticManifest::build(fixture.index, fixture.model, fixture.contract);
    CHECK(is_manifest(built));
    if (auto* manifest = std::get_if<SemanticManifest>(&built)) {
        auto decoded = SemanticManifest::decode(fixture.index, manifest->bytes());
        CHECK(is_manifest(decoded));
    }

    fixture.model.tensors[0].planes[0].storage_type = ScalarType::U32;
    auto u32_alias = SemanticManifest::build(fixture.index, fixture.model, fixture.contract);
    CHECK(std::holds_alternative<CompatibilityReport>(u32_alias));

    fixture.model.tensors[0].planes[0].storage_type = ScalarType::U8;
    fixture.model.tensors[0].layout.kind = PhysicalLayoutKind::GroupedAffine;
    auto legacy_layout = SemanticManifest::build(fixture.index, fixture.model, fixture.contract);
    CHECK(std::holds_alternative<CompatibilityReport>(legacy_layout));
    remove_fixture(fixture);
}

void test_transposed_rank_two_binding_uses_declared_axes() {
    Fixture fixture = make_fixture(false, 64, false, false, true);
    auto built = SemanticManifest::build(fixture.index, fixture.model, fixture.contract);
    CHECK(is_manifest(built));
    fixture.model.tensors[0].layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    auto rejected = SemanticManifest::build(fixture.index, fixture.model, fixture.contract);
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    remove_fixture(fixture);
}

void test_manifest_carries_authoritative_physical_codec_registry() {
    Fixture fixture = make_fixture(false);
    const PhysicalCodecRegistry registry = raw_f32_codec_registry(fixture.model);

    auto built = SemanticManifest::build(fixture.index, fixture.model, fixture.contract, registry);
    CHECK(is_manifest(built));
    if (auto* manifest = std::get_if<SemanticManifest>(&built)) {
        CHECK(manifest->has_physical_codec_authority());
        CHECK(manifest->physical_codec_registry().codecs == registry.codecs);
        CHECK(manifest->physical_codec_registry().tensors == registry.tensors);
        CHECK(manifest->physical_codec_registry_digest() != Sha256Digest{});
        auto decoded = SemanticManifest::decode(fixture.index, manifest->bytes());
        CHECK(is_manifest(decoded));
        if (auto* roundtrip = std::get_if<SemanticManifest>(&decoded)) {
            CHECK(roundtrip->physical_codec_registry().codecs == registry.codecs);
            CHECK(roundtrip->physical_codec_registry().tensors == registry.tensors);
            CHECK(roundtrip->physical_codec_registry_digest() == manifest->physical_codec_registry_digest());
        }
        std::vector<uint8_t> tampered(manifest->bytes().begin(), manifest->bytes().end());
        tampered[tampered.size() - 33] ^= 1; // final byte belongs to the exact codec registry body
        reseal(tampered);
        auto rejected = SemanticManifest::decode(fixture.index, tampered);
        CHECK(std::holds_alternative<CompatibilityReport>(rejected));

        PhysicalCodecRegistry arithmetic_changed = registry;
        arithmetic_changed.codecs[0].identity.arithmetic_digest[0] ^= 1;
        arithmetic_changed.tensors[0].identity.arithmetic_digest[0] ^= 1;
        auto changed = SemanticManifest::build(fixture.index, fixture.model,
                                               fixture.contract, arithmetic_changed);
        CHECK(std::holds_alternative<CompatibilityReport>(changed));

        PhysicalCodecRegistry certificate_changed = registry;
        certificate_changed.codecs[0].certificate_bytes.back() ^= 1;
        auto changed_certificate = SemanticManifest::build(
            fixture.index, fixture.model, fixture.contract, certificate_changed);
        CHECK(std::holds_alternative<CompatibilityReport>(changed_certificate));

        Fixture shifted = make_fixture(false, 96);
        auto shifted_result = SemanticManifest::build(shifted.index, shifted.model,
                                                      shifted.contract, registry);
        CHECK(is_manifest(shifted_result));
        if (auto* shifted_manifest = std::get_if<SemanticManifest>(&shifted_result)) {
            CHECK(shifted_manifest->physical_codec_registry_digest() !=
                  manifest->physical_codec_registry_digest());
            CHECK(shifted_manifest->package_fingerprint() != manifest->package_fingerprint());
        }
        remove_fixture(shifted);
    }
    PhysicalCodecRegistry incomplete = registry;
    incomplete.tensors.clear();
    auto incomplete_result = SemanticManifest::build(fixture.index, fixture.model,
                                                     fixture.contract, incomplete);
    CHECK(std::holds_alternative<CompatibilityReport>(incomplete_result));
    remove_fixture(fixture);
}

void test_manifest_rejects_digest_only_codec_authority() {
    Fixture fixture = make_fixture(false);
    PhysicalCodecIdentity identity;
    identity.arithmetic_version = 1;
    identity.arithmetic_digest[0] = 0x41;
    identity.layout.rank = 2;
    identity.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    identity.quantization.accumulation_type = ScalarType::F32;
    identity.planes = {{PlaneKind::Values, ScalarType::F32, 1, 4, 0}};
    PhysicalCodecRegistry digest_only;
    digest_only.codecs = {{identity}};
    digest_only.tensors = {{fixture.model.tensors[0].id, identity}};

    auto built = SemanticManifest::build(
        fixture.index, fixture.model, fixture.contract, digest_only);
    CHECK(std::holds_alternative<CompatibilityReport>(built));
    remove_fixture(fixture);
}

} // namespace

int main() {
    test_product_token_authority_is_closure_bound();
    test_golden_and_digests();
    test_quantized_physical_binding();
    test_digest_and_boundary_rejections();
    test_identity_domains_are_separate();
    test_multi_artifact_closure();
    test_tied_alias_roundtrip();
    test_resealed_field_mutation_reaches_parser();
    test_graph_digest_preserves_expert_identity_and_neutralizes_stride();
    test_plane_order_is_canonical();
    test_manifest_envelope_accepts_supported_canonical_schema_versions();
    test_column_grouped_u2_manifest_binding_is_byte_typed();
    test_transposed_rank_two_binding_uses_declared_axes();
    test_manifest_rejects_digest_only_codec_authority();
    test_manifest_carries_authoritative_physical_codec_registry();
    return test_summary("test_semantic_manifest");
}
