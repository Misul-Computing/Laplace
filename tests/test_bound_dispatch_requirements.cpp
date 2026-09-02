#define LAPLACE_RUNTIME_PACKAGE_TESTING 1

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "artifact_index.h"
#include "artifact_set.h"
#include "bound_dispatch_requirements.h"
#include "codec_binding.h"
#include "codec_certificate.h"
#include "compat_rule.h"
#include "execution_plan.h"
#include "semantic_dispatch_program.h"
#include "semantic_manifest.h"
#include "test_util.h"

using namespace Laplace;

namespace {

SemanticValue value(uint32_t id, uint32_t width, ScalarType type = ScalarType::F32) {
    return {id, type, {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, width}}, 0};
}

SemanticTensor vector_tensor(uint32_t id, TensorRole role, uint32_t width,
                             uint64_t offset) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, width}};
    tensor.layout.rank = 1;
    tensor.layout.axis_order = {0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides[0] = 1;
    tensor.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, offset,
                      static_cast<uint64_t>(width) * sizeof(float), 64, 0}};
    return tensor;
}

SemanticTensor matrix_tensor(uint32_t id, TensorRole role, uint32_t input,
                             uint32_t output, uint64_t offset) {
    SemanticTensor tensor;
    tensor.id = id;
    tensor.role = role;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, input},
                         {DimensionKind::Constant, output}};
    tensor.layout.rank = 2;
    tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides[0] = output;
    tensor.layout.strides[1] = 1;
    tensor.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, offset,
                      static_cast<uint64_t>(input) * output * sizeof(float), 64, 0}};
    return tensor;
}

// This is the smallest complete dense graph that exercises graph-level
// weights, a stateful attention root, repeated semantic requirements, and the
// output sampler. It intentionally contains no model-family or tensor-name
// routing information.
SemanticModel dense_model() {
    constexpr uint32_t width = 4;
    constexpr uint32_t vocabulary = 3;
    SemanticModel model;
    model.schema_major = 1;
    model.opset_major = 1;
    model.maximum_context = 32768;
    model.entry_kind = EntryKind::TokenIds;
    model.vocabulary_size = vocabulary;
    model.bos_id = 0;
    model.eos_id = 2;
    model.stop_ids = {2};
    for (size_t i = 0; i != model.tokenizer_digest.size(); ++i) {
        model.tokenizer_digest[i] = static_cast<uint8_t>(i + 1);
        model.template_digest[i] = static_cast<uint8_t>(i + 33);
    }
    model.tensors = {
        matrix_tensor(0, TensorRole::TokenEmbedding, vocabulary, width, 0),
        vector_tensor(1, TensorRole::AttentionNormWeight, width, 256),
        matrix_tensor(2, TensorRole::QueryWeight, width, width, 512),
        matrix_tensor(3, TensorRole::KeyWeight, width, width, 768),
        matrix_tensor(4, TensorRole::ValueWeight, width, width, 1024),
        matrix_tensor(5, TensorRole::AttentionOutputWeight, width, width, 1280),
        vector_tensor(6, TensorRole::FfnNormWeight, width, 1536),
        matrix_tensor(7, TensorRole::FfnGateWeight, width, width, 1792),
        matrix_tensor(8, TensorRole::FfnUpWeight, width, width, 2048),
        matrix_tensor(9, TensorRole::FfnDownWeight, width, width, 2304),
        vector_tensor(10, TensorRole::FinalNormWeight, width, 2560),
        matrix_tensor(11, TensorRole::OutputWeight, width, vocabulary, 2816),
    };
    for (uint32_t id = 0; id != 18; ++id)
        model.values.push_back(value(id, id == 17 ? vocabulary : width));
    model.values.push_back({18, ScalarType::U32, {{DimensionKind::Symbol, 1}}, 0});
    model.input_values_first = 18;
    model.input_values_count = 1;
    model.output_values_first = 17;
    model.output_values_count = 1;
    const auto add = [&](OperatorKind kind, std::vector<uint32_t> inputs,
                         std::vector<uint32_t> outputs, std::vector<uint32_t> tensors,
                         std::vector<uint32_t> states, OperatorPayload payload) {
        SemanticOperator op;
        op.id = static_cast<uint32_t>(model.operators.size());
        op.kind = kind;
        op.semantic_version = 1;
        op.inputs = std::move(inputs);
        op.outputs = std::move(outputs);
        op.tensors = std::move(tensors);
        op.states = std::move(states);
        op.payload = std::move(payload);
        model.operators.push_back(std::move(op));
    };
    constexpr uint32_t epsilon = 0x358637bdu;
    add(OperatorKind::EmbeddingLookup, {18}, {0}, {0}, {},
        EmbeddingLookupPayload{0x3f800000u, vocabulary, width, 0});
    add(OperatorKind::RmsNorm, {0}, {1}, {1}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {1}, {2}, {2}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {3}, {3}, {}, LinearPayload{});
    add(OperatorKind::Linear, {1}, {4}, {4}, {}, LinearPayload{});
    add(OperatorKind::Rope, {2, 3}, {5, 6}, {}, {},
        RopePayload{RopePairing::HalfSplit, true, width, 0x49742400u, 0x3f800000u});
    add(OperatorKind::CausalAttention, {5, 6, 4}, {7}, {}, {0, 1},
        CausalAttentionPayload{1, 1, width, 0x3e800000u, AttentionMask::Causal,
                                CachePolicy::Global});
    add(OperatorKind::Linear, {7}, {8}, {5}, {}, LinearPayload{});
    add(OperatorKind::Add, {0, 8}, {9}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {9}, {10}, {6}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {10}, {11}, {7}, {}, LinearPayload{});
    add(OperatorKind::Linear, {10}, {12}, {8}, {}, LinearPayload{});
    add(OperatorKind::SwiGlu, {11, 12}, {13}, {}, {}, SwiGluPayload{});
    add(OperatorKind::Linear, {13}, {14}, {9}, {}, LinearPayload{});
    add(OperatorKind::Add, {9, 14}, {15}, {}, {}, AddPayload{});
    add(OperatorKind::RmsNorm, {15}, {16}, {10}, {}, RmsNormPayload{epsilon, -1, 1});
    add(OperatorKind::Linear, {16}, {17}, {11}, {}, LinearPayload{});
    model.layers = {{0, 1, 14, 0}};

    StateFormat key_format;
    key_format.encoded_domain = TransformDomain::RopeApplied;
    key_format.alignment = 64;
    SemanticState key_state;
    key_state.id = 0;
    key_state.kind = StateKind::KeyCache;
    key_state.update_kind = StateUpdateKind::AppendKey;
    key_state.dimensions = {{DimensionKind::Symbol, 1}, {DimensionKind::Constant, 1},
                            {DimensionKind::Constant, width}};
    key_state.formats = {key_format};
    SemanticState value_state = key_state;
    value_state.id = 1;
    value_state.kind = StateKind::ValueCache;
    value_state.update_kind = StateUpdateKind::AppendValue;
    value_state.formats[0].encoded_domain = TransformDomain::Untransformed;
    model.states = {key_state, value_state};
    return model;
}

ArtifactTensorRecord physical_record(const SemanticTensor& tensor) {
    ArtifactTensorRecord record;
    record.id = tensor.id;
    record.coordinate.root = 0;
    record.logical_type = ArtifactScalarType::F32;
    for (const Dimension& dimension : tensor.dimensions)
        record.logical_dimensions.push_back(dimension.constant_or_symbol);
    record.layout = tensor.layout;
    record.quantization = tensor.quantization;
    record.format = {1, ArtifactPhysicalEncoding::F32, ArtifactScalarType::F32,
                     ArtifactScalarType::None, ArtifactScalarType::None,
                     ArtifactScalarType::None, ArtifactScalarType::None,
                     1, 4, 0, 0, 0, 0};
    const TensorPlane& source = tensor.planes.front();
    uint64_t elements = 1;
    for (uint64_t dimension : record.logical_dimensions) elements *= dimension;
    record.planes.push_back({PlaneKind::Values, ArtifactScalarType::F32,
                             {source.artifact_id, source.offset, source.length},
                             elements, 4, 1, source.alignment});
    size_t unit_stride_axis = 0;
    for (size_t axis = 0; axis < record.logical_dimensions.size(); ++axis)
        if (record.layout.strides[axis] == 1) unit_stride_axis = axis;
    record.axis.row_stride_bytes = record.logical_dimensions.empty()
        ? 0 : record.logical_dimensions[unit_stride_axis] * sizeof(float);
    return record;
}

std::vector<uint8_t> equivalent_f32_certificate() {
    // Add a no-op CastFloat node to the canonical raw-F32 certificate. Both
    // certificates decode the same bytes, but the certificate identities are
    // intentionally distinct for the binding-domain test.
    std::vector<uint8_t> bytes = make_raw_f32_codec_certificate();
    constexpr size_t node_count_offset = 32;
    constexpr size_t depth_offset = 31;
    constexpr size_t extension_offset = 111;
    bytes[node_count_offset] = 2;
    bytes[node_count_offset + 1] = 0;
    bytes[depth_offset] = 2;
    const std::array<uint8_t, 16> cast = {
        3, 2, 0xff, 0, 0, 0, 0xff, 0xff,
        0xff, 0xff, 0, 0, 0, 0, 0, 0};
    bytes.insert(bytes.begin() + extension_offset, cast.begin(), cast.end());
    return bytes;
}

struct Fixture {
    SemanticModel model;
    std::shared_ptr<const RuntimePackage> package;
};

Fixture make_fixture(uint8_t fill) {
    Fixture fixture;
    fixture.model = dense_model();
    std::vector<uint8_t> bytes(4096, fill);
    auto artifact = ArtifactSet::make_owned_blob(ArtifactId{0}, ArtifactRole::Primary, bytes);
    CHECK(std::holds_alternative<PackageView>(artifact));
    if (!std::holds_alternative<PackageView>(artifact)) return fixture;

    ArtifactIndexInput input;
    input.artifacts.push_back(std::get<PackageView>(std::move(artifact)));
    for (const SemanticTensor& tensor : fixture.model.tensors)
        input.tensors.push_back(physical_record(tensor));
    auto built_index = ArtifactIndex::build(std::move(input));
    CHECK(std::holds_alternative<ArtifactIndex>(built_index));
    if (!std::holds_alternative<ArtifactIndex>(built_index)) return fixture;
    ArtifactIndex index = std::get<ArtifactIndex>(std::move(built_index));

    TokenContract contract;
    contract.vocabulary_size = fixture.model.vocabulary_size;
    contract.bos_id = fixture.model.bos_id;
    contract.eos_id = fixture.model.eos_id;
    contract.stop_ids = fixture.model.stop_ids;
    contract.authoritative_tokenizer_digest = {fixture.model.tokenizer_digest};
    contract.authoritative_template_digest = {fixture.model.template_digest};

    PhysicalCodecRegistry registry;
    const std::vector<uint8_t> certificate_bytes = make_raw_f32_codec_certificate();
    const std::vector<uint8_t> alternate_certificate = equivalent_f32_certificate();
    const auto parsed = parse_codec_certificate(certificate_bytes);
    const auto alternate_parsed = parse_codec_certificate(alternate_certificate);
    const auto* certificate = std::get_if<CodecCertificate>(&parsed);
    const auto* alternate = std::get_if<CodecCertificate>(&alternate_parsed);
    CHECK(certificate != nullptr);
    CHECK_MSG(alternate != nullptr, "alternate certificate error=%u bytes=%zu",
              std::get_if<CodecCertificateError>(&alternate_parsed)
                  ? static_cast<unsigned>(*std::get_if<CodecCertificateError>(&alternate_parsed))
                  : 0u, alternate_certificate.size());
    if (!certificate || !alternate) return fixture;
    for (const SemanticTensor& tensor : fixture.model.tensors) {
        const CodecCertificate& selected = tensor.id == 3 ? *alternate : *certificate;
        const auto identity = physical_codec_identity(
            tensor, selected.identity().abi_version, selected.identity().digest);
        CHECK(identity.has_value());
        if (!identity) return fixture;
        if (std::none_of(registry.codecs.begin(), registry.codecs.end(),
                         [&](const PhysicalCodecSpec& spec) {
                             return spec.identity == *identity;
                         })) {
            registry.codecs.push_back({*identity, tensor.id == 3
                ? alternate_certificate : certificate_bytes});
        }
        registry.tensors.push_back({tensor.id, *identity});
    }
    std::sort(registry.codecs.begin(), registry.codecs.end(),
              [](const auto& left, const auto& right) {
                  return physical_codec_identity_less(left.identity, right.identity);
              });
    std::sort(registry.tensors.begin(), registry.tensors.end(),
              [](const auto& left, const auto& right) {
                  return left.tensor_id < right.tensor_id;
              });
    auto manifest = SemanticManifest::build(index, fixture.model, contract, registry);
    CHECK(std::holds_alternative<SemanticManifest>(manifest));
    if (!std::holds_alternative<SemanticManifest>(manifest)) return fixture;
    fixture.package = RuntimePackage::make_closed_v1_test_only(
        std::get<SemanticManifest>(std::move(manifest)));
    CHECK(fixture.package != nullptr);
    return fixture;
}

bool bound_failure(const RuntimePackage& package, const ResolvedCodecBindings& bindings,
                   const SessionRequest& request,
                   std::span<const SemanticDispatchProgram> programs) {
    return std::holds_alternative<CompatibilityReport>(
        bind_dispatch_requirements(package, bindings, request, programs));
}

void test_positive_records_and_domain() {
    const Fixture fixture = make_fixture(0x11);
    CHECK(fixture.package != nullptr);
    if (!fixture.package) return;
    const auto bindings_result = preflight_codec_bindings(*fixture.package);
    CHECK(std::holds_alternative<ResolvedCodecBindings>(bindings_result));
    if (!std::holds_alternative<ResolvedCodecBindings>(bindings_result)) return;
    const ResolvedCodecBindings bindings = std::get<ResolvedCodecBindings>(bindings_result);

    SessionRequest request;
    request.max_context = 4;
    request.max_batch = 1;
    request.enable_prefill = true;
    request.enable_decode = true;
    const auto prefill = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Prefill, 1, NumericalClass::ExactFp32, false, false});
    const auto prefill_sample = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Prefill, 1, NumericalClass::ExactFp32, false, true});
    const auto decode = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Decode, 1, NumericalClass::ExactFp32, false, false});
    const auto decode_sample = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Decode, 1, NumericalClass::ExactFp32, false, true});
    CHECK_MSG(std::holds_alternative<SemanticDispatchProgram>(prefill),
              "prefill code=%u detail=%s",
              std::holds_alternative<CompatibilityReport>(prefill)
                  ? static_cast<unsigned>(std::get<CompatibilityReport>(prefill).code) : 0u,
              std::holds_alternative<CompatibilityReport>(prefill)
                  ? std::get<CompatibilityReport>(prefill).detail.c_str() : "");
    CHECK_MSG(std::holds_alternative<SemanticDispatchProgram>(decode),
              "decode code=%u detail=%s",
              std::holds_alternative<CompatibilityReport>(decode)
                  ? static_cast<unsigned>(std::get<CompatibilityReport>(decode).code) : 0u,
              std::holds_alternative<CompatibilityReport>(decode)
                  ? std::get<CompatibilityReport>(decode).detail.c_str() : "");
    CHECK(std::holds_alternative<SemanticDispatchProgram>(prefill_sample));
    CHECK(std::holds_alternative<SemanticDispatchProgram>(decode_sample));
    if (!std::holds_alternative<SemanticDispatchProgram>(prefill) ||
        !std::holds_alternative<SemanticDispatchProgram>(prefill_sample) ||
        !std::holds_alternative<SemanticDispatchProgram>(decode) ||
        !std::holds_alternative<SemanticDispatchProgram>(decode_sample)) return;
    const std::array<SemanticDispatchProgram, 4> programs = {
        std::get<SemanticDispatchProgram>(prefill),
        std::get<SemanticDispatchProgram>(prefill_sample),
        std::get<SemanticDispatchProgram>(decode),
        std::get<SemanticDispatchProgram>(decode_sample)};
    const auto bound = bind_dispatch_requirements(*fixture.package, bindings, request, programs);
    CHECK(std::holds_alternative<BoundDispatchRequirements>(bound));
    if (!std::holds_alternative<BoundDispatchRequirements>(bound)) return;
    const BoundDispatchRequirements& result = std::get<BoundDispatchRequirements>(bound);
    CHECK(result.package_fingerprint() == fixture.package->package_fingerprint());
    CHECK(result.programs().size() == 4);
    for (const BoundDispatchProgram& program : result.programs()) {
        CHECK(!program.steps().empty());
        for (const BoundDispatchStep& step : program.steps()) {
            CHECK(step.bound_digest() != Sha256Digest{});
            CHECK(step.codec_occurrence_indices().size() ==
                  step.codec_program_identities().size());
            CHECK(step.codec_occurrence_indices().size() == step.physical_identities().size());
            if (step.requirement().step_kind == SemanticDispatchStepKind::GreedySampler)
                CHECK(step.codec_occurrence_indices().empty());
        }
    }
    // Query and key are structurally identical requirements. Tensor 3 uses a
    // certificate with the same physical decode but a different certificate
    // identity, so step-aligned binding must keep their digests distinct.
    const auto& decode_steps = result.programs()[2].steps();
    CHECK(decode_steps.size() > 3);
    if (decode_steps.size() > 3) {
        CHECK(decode_steps[2].requirement() == decode_steps[3].requirement());
        CHECK(decode_steps[2].codec_program_identities()[0] !=
              decode_steps[3].codec_program_identities()[0]);
        CHECK(decode_steps[2].bound_digest() != decode_steps[3].bound_digest());
    }
}

void test_v2_exact_binding_record() {
    const Fixture first = make_fixture(0x41);
    const Fixture second = make_fixture(0x42);
    CHECK(first.package != nullptr);
    CHECK(second.package != nullptr);
    if (!first.package || !second.package) return;
    const auto first_bindings_result = preflight_codec_bindings(*first.package);
    const auto second_bindings_result = preflight_codec_bindings(*second.package);
    CHECK(std::holds_alternative<ResolvedCodecBindings>(first_bindings_result));
    CHECK(std::holds_alternative<ResolvedCodecBindings>(second_bindings_result));
    if (!std::holds_alternative<ResolvedCodecBindings>(first_bindings_result) ||
        !std::holds_alternative<ResolvedCodecBindings>(second_bindings_result)) return;
    const auto unsampled_result = build_semantic_dispatch_program(
        first.model, {ExecutionPhase::Decode, 1, NumericalClass::ExactFp32, false, false});
    const auto sampled_result = build_semantic_dispatch_program(
        first.model, {ExecutionPhase::Decode, 1, NumericalClass::ExactFp32, false, true});
    CHECK(std::holds_alternative<SemanticDispatchProgram>(unsampled_result));
    CHECK(std::holds_alternative<SemanticDispatchProgram>(sampled_result));
    if (!std::holds_alternative<SemanticDispatchProgram>(unsampled_result) ||
        !std::holds_alternative<SemanticDispatchProgram>(sampled_result)) return;
    const auto program = std::get<SemanticDispatchProgram>(unsampled_result);
    const auto sampled_program = std::get<SemanticDispatchProgram>(sampled_result);
    const SessionRequest request{4, 1, 0, false, true, false, false,
                                 NumericalClass::ExactFp32, RuntimeObjective::Latency};
    const std::array<SemanticDispatchProgram, 2> programs = {program, sampled_program};
    const auto first_bound = bind_dispatch_requirements(
        *first.package, std::get<ResolvedCodecBindings>(first_bindings_result),
        request, programs);
    const auto second_bound = bind_dispatch_requirements(
        *second.package, std::get<ResolvedCodecBindings>(second_bindings_result),
        request, programs);
    CHECK_MSG(std::holds_alternative<BoundDispatchRequirements>(first_bound),
              "first bound code=%u detail=%s",
              std::holds_alternative<CompatibilityReport>(first_bound)
                  ? static_cast<unsigned>(std::get<CompatibilityReport>(first_bound).code) : 0u,
              std::holds_alternative<CompatibilityReport>(first_bound)
                  ? std::get<CompatibilityReport>(first_bound).detail.c_str() : "");
    CHECK_MSG(std::holds_alternative<BoundDispatchRequirements>(second_bound),
              "second bound code=%u detail=%s",
              std::holds_alternative<CompatibilityReport>(second_bound)
                  ? static_cast<unsigned>(std::get<CompatibilityReport>(second_bound).code) : 0u,
              std::holds_alternative<CompatibilityReport>(second_bound)
                  ? std::get<CompatibilityReport>(second_bound).detail.c_str() : "");
    if (!std::holds_alternative<BoundDispatchRequirements>(first_bound) ||
        !std::holds_alternative<BoundDispatchRequirements>(second_bound)) return;
    const BoundDispatchRequirements& first_records =
        std::get<BoundDispatchRequirements>(first_bound);
    CHECK(first_records.version() == kBoundDispatchRequirementsVersionV2);
    CHECK(first_records.programs().size() == 2);
    if (first_records.programs().empty()) return;
    const BoundDispatchProgram& record = first_records.programs()[1];
    CHECK(record.version() == kBoundDispatchRequirementsVersionV2);
    CHECK(record.output_binding().has_value());
    CHECK(record.bound_digest() != Sha256Digest{});
    CHECK(!validate_bound_dispatch_requirements(first_records));
    CHECK(!validate_bound_dispatch_program(record));
    CHECK(record.steps().size() == sampled_program.steps.size());
    if (record.steps().size() <= 6) return;
    const BoundDispatchStep& attention = record.steps()[6];
    CHECK(attention.version() == kBoundDispatchRequirementsVersionV2);
    CHECK(!validate_bound_dispatch_step(attention));
    CHECK(attention.tensors().size() == 0);
    CHECK(attention.input_values().size() == 3);
    CHECK(attention.input_values()[0].descriptor_digest != Sha256Digest{});
    CHECK(attention.states().size() == 2);
    CHECK(attention.states()[0].descriptor_digest != Sha256Digest{});
    CHECK(attention.session_effects().empty());
    const BoundDispatchStep& norm = record.steps()[1];
    CHECK(norm.tensors().size() == 1);
    if (norm.tensors().size() == 1) {
        CHECK(norm.tensors()[0].occurrence_index == 1);
        CHECK(norm.tensors()[0].tensor_id == 1);
        CHECK(norm.tensors()[0].tensor_slot == 0);
        CHECK(norm.tensors()[0].semantic_tensor_digest != Sha256Digest{});
        CHECK(norm.tensors()[0].source_span_digest != Sha256Digest{});
        CHECK(norm.tensors()[0].codec_program_identity.contract_digest !=
              CodecProgramDigest{});
        CHECK(norm.tensors()[0].physical_identity.arithmetic_digest !=
              PhysicalIdentityDigest{});
    }
    const BoundDispatchStep& sampler = record.steps().back();
    CHECK(sampler.sampler_binding().has_value());
    CHECK(sampler.session_effects().size() == 3);
    CHECK(sampler.workspace().descriptor_digest != Sha256Digest{});
    // Source bytes are provenance, not package-fingerprint behavior. They
    // must still prevent two different source spans from sharing a record.
    const BoundDispatchProgram& second_record =
        std::get<BoundDispatchRequirements>(second_bound).programs()[1];
    CHECK(record.steps()[1].bound_digest() != second_record.steps()[1].bound_digest());
}

void test_v2_mutations_fail_closed() {
    const Fixture fixture = make_fixture(0x51);
    CHECK(fixture.package != nullptr);
    if (!fixture.package) return;
    const auto bindings_result = preflight_codec_bindings(*fixture.package);
    CHECK(std::holds_alternative<ResolvedCodecBindings>(bindings_result));
    if (!std::holds_alternative<ResolvedCodecBindings>(bindings_result)) return;
    const auto program_result = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Decode, 1, NumericalClass::ExactFp32, false, false});
    const auto sampled_result = build_semantic_dispatch_program(
        fixture.model, {ExecutionPhase::Decode, 1, NumericalClass::ExactFp32, false, true});
    CHECK(std::holds_alternative<SemanticDispatchProgram>(program_result));
    CHECK(std::holds_alternative<SemanticDispatchProgram>(sampled_result));
    if (!std::holds_alternative<SemanticDispatchProgram>(program_result) ||
        !std::holds_alternative<SemanticDispatchProgram>(sampled_result)) return;
    const SessionRequest request{4, 1, 0, false, true, false, false,
                                 NumericalClass::ExactFp32, RuntimeObjective::Latency};
    const auto original = std::get<SemanticDispatchProgram>(program_result);
    const auto sampled = std::get<SemanticDispatchProgram>(sampled_result);
    const std::array<SemanticDispatchProgram, 2> original_programs = {original, sampled};
    const auto rejected = [&](auto mutate) {
        auto candidate = original_programs;
        mutate(candidate);
        CHECK(bound_failure(*fixture.package,
                            std::get<ResolvedCodecBindings>(bindings_result),
                            request, candidate));
    };
    rejected([](auto& candidate) {
        candidate[0].steps[6].input_values[0] ^= 1;
    });
    rejected([](auto& candidate) {
        candidate[0].steps[1].tensor_ids[0] ^= 1;
    });
    rejected([](auto& candidate) {
        candidate[0].steps[6].state_effects[0].state_id ^= 1;
    });
    rejected([](auto& candidate) {
        candidate[1].output_binding->selected_row = 1;
    });
    rejected([](auto& candidate) {
        candidate[1].output_binding->logits_value_id = UINT32_MAX;
    });
    rejected([](auto& candidate) {
        candidate[0].steps.back().workspace_id ^= 1;
    });
}

void test_prefill_capacity_and_canonical_program_order() {
    const Fixture fixture = make_fixture(0x31);
    CHECK(fixture.package != nullptr);
    if (!fixture.package) return;
    const auto bindings_result = preflight_codec_bindings(*fixture.package);
    CHECK(std::holds_alternative<ResolvedCodecBindings>(bindings_result));
    if (!std::holds_alternative<ResolvedCodecBindings>(bindings_result)) return;

    const auto make_program = [&](ExecutionPhase phase, uint32_t batch_rows,
                                  bool sampled) -> SemanticDispatchProgram {
        const auto result = build_semantic_dispatch_program(
            fixture.model, {phase, batch_rows, NumericalClass::ExactFp32, false, sampled});
        CHECK(std::holds_alternative<SemanticDispatchProgram>(result));
        return std::holds_alternative<SemanticDispatchProgram>(result)
            ? std::get<SemanticDispatchProgram>(result)
            : SemanticDispatchProgram{};
    };
    const std::array<SemanticDispatchProgram, 6> programs = {
        make_program(ExecutionPhase::Prefill, 1, false),
        make_program(ExecutionPhase::Prefill, 1, true),
        make_program(ExecutionPhase::Prefill, 2, false),
        make_program(ExecutionPhase::Prefill, 2, true),
        make_program(ExecutionPhase::Decode, 1, false),
        make_program(ExecutionPhase::Decode, 1, true),
    };
    const SessionRequest request{4, 2, 0, true, true, false, false,
                                 NumericalClass::ExactFp32, RuntimeObjective::Latency};
    const auto bound = bind_dispatch_requirements(
        *fixture.package, std::get<ResolvedCodecBindings>(bindings_result),
        request, programs);
    CHECK(std::holds_alternative<BoundDispatchRequirements>(bound));

    const auto expect_rejected = [&](auto mutate) {
        auto candidate = programs;
        mutate(candidate);
        CHECK(bound_failure(
            *fixture.package, std::get<ResolvedCodecBindings>(bindings_result),
            request, candidate));
    };
    expect_rejected([](auto& candidate) {
        std::swap(candidate[0], candidate[2]);
    });
    expect_rejected([](auto& candidate) {
        candidate[3] = candidate[2];
    });
    expect_rejected([](auto& candidate) {
        candidate[2] = candidate[3];
    });
    expect_rejected([](auto& candidate) {
        candidate[1] = candidate[2];
    });
    expect_rejected([](auto& candidate) {
        candidate[3] = candidate[4];
    });
}

void test_fail_closed_tamper_and_package_binding() {
    const Fixture first = make_fixture(0x21);
    const Fixture second = make_fixture(0x22);
    CHECK(first.package != nullptr);
    CHECK(second.package != nullptr);
    if (!first.package || !second.package) return;
    const auto first_bindings_result = preflight_codec_bindings(*first.package);
    const auto second_bindings_result = preflight_codec_bindings(*second.package);
    CHECK(std::holds_alternative<ResolvedCodecBindings>(first_bindings_result));
    CHECK(std::holds_alternative<ResolvedCodecBindings>(second_bindings_result));
    if (!std::holds_alternative<ResolvedCodecBindings>(first_bindings_result) ||
        !std::holds_alternative<ResolvedCodecBindings>(second_bindings_result)) return;
    const auto first_program_result = build_semantic_dispatch_program(
        first.model, {ExecutionPhase::Decode, 1, NumericalClass::ExactFp32, false, false});
    CHECK(std::holds_alternative<SemanticDispatchProgram>(first_program_result));
    if (!std::holds_alternative<SemanticDispatchProgram>(first_program_result)) return;
    const SemanticDispatchProgram original =
        std::get<SemanticDispatchProgram>(first_program_result);
    const SessionRequest request{4, 1, 0, false, true, false, false,
                                 NumericalClass::ExactFp32, RuntimeObjective::Latency};
    const auto first_bindings = std::get<ResolvedCodecBindings>(first_bindings_result);
    const auto second_bindings = std::get<ResolvedCodecBindings>(second_bindings_result);
    const std::array<SemanticDispatchProgram, 1> programs = {original};
    CHECK(bound_failure(*first.package, second_bindings, request, programs));

    SessionRequest prefill_request = request;
    prefill_request.max_batch = 2;
    prefill_request.enable_prefill = true;
    prefill_request.enable_decode = false;
    const auto small_prefill_result = build_semantic_dispatch_program(
        first.model, {ExecutionPhase::Prefill, 1, NumericalClass::ExactFp32, false, false});
    CHECK(std::holds_alternative<SemanticDispatchProgram>(small_prefill_result));
    if (std::holds_alternative<SemanticDispatchProgram>(small_prefill_result)) {
        const auto small_prefill = std::get<SemanticDispatchProgram>(small_prefill_result);
        CHECK(!bound_failure(*first.package, first_bindings, prefill_request,
                             std::span<const SemanticDispatchProgram>(&small_prefill, 1)));
    }

    SessionRequest wide_decode_request = request;
    wide_decode_request.max_batch = 2;
    const auto wide_decode_result = build_semantic_dispatch_program(
        first.model, {ExecutionPhase::Decode, 2, NumericalClass::ExactFp32, false, false});
    CHECK(std::holds_alternative<SemanticDispatchProgram>(wide_decode_result));
    if (std::holds_alternative<SemanticDispatchProgram>(wide_decode_result)) {
        const auto wide_decode = std::get<SemanticDispatchProgram>(wide_decode_result);
        CHECK(bound_failure(*first.package, first_bindings, wide_decode_request,
                            std::span<const SemanticDispatchProgram>(&wide_decode, 1)));
    }

    auto missing = original;
    missing.steps.pop_back();
    CHECK(bound_failure(*first.package, first_bindings, request,
                        std::span<const SemanticDispatchProgram>(&missing, 1)));
    auto reordered = original;
    if (reordered.steps.size() > 2) std::swap(reordered.steps[0], reordered.steps[1]);
    CHECK(bound_failure(*first.package, first_bindings, request,
                        std::span<const SemanticDispatchProgram>(&reordered, 1)));
    auto replayed = original;
    if (replayed.steps.size() > 1) replayed.steps[1] = replayed.steps[0];
    CHECK(bound_failure(*first.package, first_bindings, request,
                        std::span<const SemanticDispatchProgram>(&replayed, 1)));
}

} // namespace

int main() {
    test_positive_records_and_domain();
    test_v2_exact_binding_record();
    test_v2_mutations_fail_closed();
    test_prefill_capacity_and_canonical_program_order();
    test_fail_closed_tamper_and_package_binding();
    return test_summary("test_bound_dispatch_requirements");
}
