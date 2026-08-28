#include <algorithm>
#include <cstdio>
#include <cstring>
#include <variant>

#include "artifact_set.h"
#include "compat_rule.h"
#include "gguf_writer.h"
#include "tensor.h"
#include "test_util.h"

using namespace Laplace;

namespace {

CompatibilityRule import_rule() {
    CompatibilityRule rule;
    rule.rule_id = "dense-import-v1";
    rule.package_format = PackageFormat::Gguf;
    rule.metadata = {{MetadataPredicateKind::ExactU64, "layers", 1, {}},
                     {MetadataPredicateKind::ExactU64, "rope_base_bits", 0x49742400u, {}}};
    rule.tensors = {{0, TensorPatternKind::AnchoredDecimalCapture, "blk.{d}.weight", 0,
                     TensorRole::TokenEmbedding, ScalarType::F32, PhysicalLayoutKind::ContiguousRowMajor,
                     QuantizationKind::None, {{DimensionKind::Constant, 2}, {DimensionKind::Constant, 2}}, 1}};
    rule.semantic_template.maximum_context = 32768;
    rule.semantic_template.vocabulary_size = 3;
    rule.semantic_template.bos_id = 1;
    rule.semantic_template.eos_id = 2;
    SemanticTensor tensor;
    tensor.id = 0;
    tensor.role = TensorRole::TokenEmbedding;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, 2}, {DimensionKind::Constant, 2}};
    tensor.layout.rank = 2;
    tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides[0] = 2;
    tensor.layout.strides[1] = 1;
    tensor.planes = {{PlaneKind::Values, ScalarType::F32, ArtifactId{0}, 0, 16, 4, 0}};
    rule.semantic_template.tensors = {tensor};
    return rule;
}

CompatibilityRule q4k_import_rule(uint32_t block_bytes = 144) {
    CompatibilityRule rule;
    rule.rule_id = "blocked-import-v1";
    rule.package_format = PackageFormat::Gguf;
    rule.metadata = {{MetadataPredicateKind::ExactU64, "layers", 1, {}}};

    TensorPattern pattern;
    pattern.template_id = 0;
    pattern.kind = TensorPatternKind::AnchoredDecimalCapture;
    pattern.pattern = "blk.{d}.weight";
    pattern.role = TensorRole::TokenEmbedding;
    pattern.logical_type = ScalarType::F32;
    pattern.layout = PhysicalLayoutKind::GgufBlocked;
    pattern.quantization = QuantizationKind::BlockedAffine;
    pattern.dimensions = {{DimensionKind::Constant, 256}, {DimensionKind::Constant, 1}};
    pattern.storage_type = ScalarType::U8;
    rule.tensors = {std::move(pattern)};

    rule.semantic_template.maximum_context = 32768;
    rule.semantic_template.vocabulary_size = 3;
    rule.semantic_template.bos_id = 1;
    rule.semantic_template.eos_id = 2;
    SemanticTensor tensor;
    tensor.id = 0;
    tensor.role = TensorRole::TokenEmbedding;
    tensor.logical_type = ScalarType::F32;
    tensor.dimensions = {{DimensionKind::Constant, 256}, {DimensionKind::Constant, 1}};
    tensor.layout.kind = PhysicalLayoutKind::GgufBlocked;
    tensor.layout.packing = PackingKind::Gguf;
    tensor.layout.rank = 2;
    tensor.layout.block_rank = 1;
    tensor.layout.axis_order = {0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    tensor.layout.strides[0] = 1;
    tensor.layout.strides[1] = 256;
    tensor.layout.block_elements = 256;
    tensor.layout.block_bytes = block_bytes;
    tensor.quantization.kind = QuantizationKind::BlockedAffine;
    tensor.quantization.accumulation_type = ScalarType::F32;
    tensor.quantization.scale_type = ScalarType::F16;
    tensor.quantization.zero_type = ScalarType::F16;
    tensor.quantization.block_elements = 256;
    tensor.quantization.block_bytes = block_bytes;
    tensor.quantization.group_size = 256;
    tensor.quantization.required_plane_mask = 1;
    tensor.planes = {{PlaneKind::Values, ScalarType::U8, ArtifactId{0}, 0, block_bytes, 32, 0}};
    rule.semantic_template.tensors = {tensor};
    return rule;
}

void test_imports_checked_gguf_bytes() {
    gguf_writer::Writer writer;
    writer.kv_u32("layers", 1);
    writer.kv_f32("rope_base_bits", 1000000.0f);
    gguf_writer::TensorDecl tensor;
    tensor.name = "blk.0.weight";
    tensor.dims = {2, 2};
    tensor.type = static_cast<uint32_t>(GGMLType::F32);
    tensor.data.resize(4 * sizeof(float));
    writer.add_tensor(std::move(tensor));
    CHECK(writer.write_file("test_gguf_import.gguf"));

    auto artifacts = ArtifactSet::load_single_file("test_gguf_import.gguf");
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (auto* set = std::get_if<ArtifactSet>(&artifacts)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) {
            auto imported = import_expected_fixture_gguf(*package, {import_rule()});
            CHECK(std::holds_alternative<SemanticModel>(imported));
            if (auto* semantic = std::get_if<SemanticModel>(&imported)) {
                CHECK(semantic->tensors.size() == 1);
                CHECK(semantic->tensors[0].planes[0].storage_type == ScalarType::F32);
                CHECK(semantic->tensors[0].planes[0].length == 16);
            }
        }
    }
    remove("test_gguf_import.gguf");
}

void test_imports_exact_blocked_q4k_plane() {
    gguf_writer::Writer writer;
    writer.kv_u32("layers", 1);
    gguf_writer::TensorDecl tensor;
    tensor.name = "blk.0.weight";
    tensor.dims = {256, 1};
    tensor.type = static_cast<uint32_t>(GGMLType::Q4_K);
    tensor.data.resize(144);
    writer.add_tensor(std::move(tensor));
    CHECK(writer.write_file("test_gguf_import_q4k.gguf"));

    auto artifacts = ArtifactSet::load_single_file("test_gguf_import_q4k.gguf");
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (auto* set = std::get_if<ArtifactSet>(&artifacts)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) {
            auto imported = import_expected_fixture_gguf(*package, {q4k_import_rule()});
            CHECK(std::holds_alternative<SemanticModel>(imported));
            if (auto* semantic = std::get_if<SemanticModel>(&imported)) {
                const SemanticTensor& mapped = semantic->tensors[0];
                CHECK(mapped.planes[0].storage_type == ScalarType::U8);
                CHECK(mapped.planes[0].length == 144);
                CHECK(mapped.layout.kind == PhysicalLayoutKind::GgufBlocked);
                CHECK(mapped.layout.block_elements == 256);
                CHECK(mapped.layout.block_bytes == 144);
                CHECK(mapped.quantization.kind == QuantizationKind::BlockedAffine);
                CHECK(mapped.quantization.block_elements == 256);
                CHECK(mapped.quantization.block_bytes == 144);
            }
            auto incompatible = import_expected_fixture_gguf(*package, {q4k_import_rule(143)});
            CHECK(!std::holds_alternative<SemanticModel>(incompatible));
            if (const auto* report = std::get_if<CompatibilityReport>(&incompatible)) {
                CHECK(report->code == CompatibilityError::IMPORT_SEMANTICS_MISSING);
                CHECK(report->detail == "no bundled LAPRUL10 rule matches package metadata and tensor contracts");
            }
        }
    }
    remove("test_gguf_import_q4k.gguf");
}

PackageEvidence generic_hybrid_evidence(bool ambiguous_first_layer = false, bool fused_attention_gate = false,
                                        bool terminal_mtp = false) {
    PackageEvidence package;
    package.metadata = {
        {"model.block_count", terminal_mtp ? uint64_t{3} : uint64_t{2}},
        {"model.context_length", uint64_t{256}},
        {"model.embedding_length", uint64_t{256}},
        {"model.feed_forward_length", uint64_t{256}},
        {"model.attention.head_count", uint64_t{1}},
        {"model.attention.head_count_kv", uint64_t{1}},
        {"model.attention.key_length", uint64_t{256}},
        {"model.attention.value_length", uint64_t{256}},
        {"model.rope.dimension_count", uint64_t{256}},
        {"model.rope.freq_base", uint64_t{0x49742400u}},
        {"model.attention.layer_norm_rms_epsilon", uint64_t{0x358637bdu}},
        {"model.ssm.conv_kernel", uint64_t{2}},
        {"model.ssm.group_count", uint64_t{1}},
        {"model.ssm.inner_size", uint64_t{256}},
        {"model.ssm.state_size", uint64_t{256}},
        {"model.ssm.time_step_rank", uint64_t{1}},
        {"tokenizer.ggml.bos_token_id", uint64_t{1}},
        {"tokenizer.ggml.eos_token_id", uint64_t{2}},
    };
    if (terminal_mtp) package.metadata.emplace("model.nextn_predict_layers", uint64_t{1});
    uint64_t offset = 0;
    const auto add = [&](std::string name, std::vector<uint64_t> dimensions) {
        uint64_t elements = 1;
        for (uint64_t dimension : dimensions) elements *= dimension;
        PackageTensorEvidence tensor;
        tensor.name = std::move(name);
        tensor.dimensions = std::move(dimensions);
        tensor.storage_type = ScalarType::F32;
        tensor.layout = PhysicalLayoutKind::ContiguousRowMajor;
        tensor.quantization = QuantizationKind::None;
        tensor.artifact_id = ArtifactId{0};
        tensor.offset = offset;
        tensor.length = elements * sizeof(float);
        tensor.alignment = 32;
        offset = (offset + tensor.length + 31) & ~uint64_t{31};
        package.tensors.push_back(std::move(tensor));
    };
    add("token_embd.weight", {256, 5});
    add("output_norm.weight", {256});
    add("output.weight", {256, 5});
    add("blk.0.attn_norm.weight", {256});
    add("blk.0.attn_qkv.weight", {256, 768});
    add("blk.0.attn_gate.weight", {256, 256});
    add("blk.0.ssm_alpha.weight", {256, 1});
    add("blk.0.ssm_beta.weight", {256, 1});
    add("blk.0.ssm_a", {1});
    add("blk.0.ssm_conv1d.weight", {2, 768});
    add("blk.0.ssm_dt.bias", {1});
    add("blk.0.ssm_norm.weight", {256});
    add("blk.0.ssm_out.weight", {256, 256});
    add("blk.0.post_attention_norm.weight", {256});
    add("blk.0.ffn_gate.weight", {256, 256});
    add("blk.0.ffn_up.weight", {256, 256});
    add("blk.0.ffn_down.weight", {256, 256});
    add("blk.1.attn_norm.weight", {256});
    add("blk.1.attn_q.weight", {256, fused_attention_gate ? 512u : 256u});
    if (fused_attention_gate) {
        add("blk.1.attn_q_norm.weight", {256});
        add("blk.1.attn_k_norm.weight", {256});
    }
    add("blk.1.attn_k.weight", {256, 256});
    add("blk.1.attn_v.weight", {256, 256});
    add("blk.1.attn_output.weight", {256, 256});
    add("blk.1.post_attention_norm.weight", {256});
    add("blk.1.ffn_gate.weight", {256, 256});
    add("blk.1.ffn_up.weight", {256, 256});
    add("blk.1.ffn_down.weight", {256, 256});
    if (terminal_mtp) {
        add("blk.2.attn_norm.weight", {256});
        add("blk.2.attn_q.weight", {256, 512});
        add("blk.2.attn_q_norm.weight", {256});
        add("blk.2.attn_k.weight", {256, 256});
        add("blk.2.attn_k_norm.weight", {256});
        add("blk.2.attn_v.weight", {256, 256});
        add("blk.2.attn_output.weight", {256, 256});
        add("blk.2.post_attention_norm.weight", {256});
        add("blk.2.ffn_gate.weight", {256, 256});
        add("blk.2.ffn_up.weight", {256, 256});
        add("blk.2.ffn_down.weight", {256, 256});
        add("blk.2.nextn.eh_proj.weight", {512, 256});
        add("blk.2.nextn.enorm.weight", {256});
        add("blk.2.nextn.hnorm.weight", {256});
        add("blk.2.nextn.shared_head_norm.weight", {256});
    }
    if (ambiguous_first_layer) add("blk.0.attn_q.weight", {256, 256});
    return package;
}

void test_generic_resolver_builds_mixed_known_operator_schedule() {
    auto resolved = resolve_gguf_semantics(generic_hybrid_evidence());
    CHECK(std::holds_alternative<SemanticModel>(resolved));
    if (const auto* semantics = std::get_if<SemanticModel>(&resolved)) {
        CHECK(semantics->layers.size() == 2);
        CHECK(semantics->states.size() == 4);
        uint32_t recurrent = 0;
        uint32_t attention = 0;
        for (const SemanticOperator& op : semantics->operators) {
            recurrent += op.kind == OperatorKind::GatedDeltaNet;
            attention += op.kind == OperatorKind::CausalAttention;
        }
        CHECK(recurrent == 1);
        CHECK(attention == 1);
        CHECK(semantics->maximum_context == 256);
        const auto conv = std::find_if(semantics->tensors.begin(), semantics->tensors.end(),
                                       [](const SemanticTensor& tensor) {
                                           return tensor.role == TensorRole::RecurrentConvWeight;
                                       });
        CHECK(conv != semantics->tensors.end());
        if (conv != semantics->tensors.end()) {
            CHECK(conv->dimensions.size() == 2);
            CHECK(conv->dimensions[0].kind == DimensionKind::Constant);
            CHECK(conv->dimensions[0].constant_or_symbol == 768);
            CHECK(conv->dimensions[1].kind == DimensionKind::Constant);
            CHECK(conv->dimensions[1].constant_or_symbol == 2);
            CHECK(conv->layout.axis_order[0] == 1);
            CHECK(conv->layout.axis_order[1] == 0);
            CHECK(conv->layout.strides[0] == 1);
            CHECK(conv->layout.strides[1] == 2);
        }
    }
    auto multi_section = generic_hybrid_evidence();
    multi_section.metadata.emplace("model.rope.dimension_sections",
                                   std::vector<uint64_t>{64, 64, 0, 0});
    auto multi_resolved = resolve_gguf_semantics(multi_section);
    CHECK(std::holds_alternative<SemanticModel>(multi_resolved));
    if (const auto* semantics = std::get_if<SemanticModel>(&multi_resolved)) {
        CHECK(semantics->schema_major == 5);
        const auto rope = std::find_if(semantics->operators.begin(), semantics->operators.end(),
                                       [](const SemanticOperator& op) { return op.kind == OperatorKind::Rope; });
        CHECK(rope != semantics->operators.end());
        if (rope != semantics->operators.end()) {
            const auto* payload = std::get_if<RopePayload>(&rope->payload);
            CHECK(payload != nullptr);
            if (payload) {
                CHECK(payload->pairing == RopePairing::MultiSectionHalfSplit);
                CHECK((payload->position_sections == std::array<uint32_t, 4>{64, 64, 0, 0}));
            }
        }
    }
    auto fused = resolve_gguf_semantics(generic_hybrid_evidence(false, true));
    CHECK(std::holds_alternative<SemanticModel>(fused));
    if (const auto* semantics = std::get_if<SemanticModel>(&fused)) {
        const auto split = std::count_if(semantics->operators.begin(), semantics->operators.end(),
                                         [](const SemanticOperator& op) { return op.kind == OperatorKind::AxisSplit; });
        const auto gated = std::count_if(semantics->operators.begin(), semantics->operators.end(),
                                         [](const SemanticOperator& op) { return op.kind == OperatorKind::GatedAttention; });
        CHECK(split == 1);
        CHECK(gated == 1);
    }
    auto mtp = resolve_gguf_semantics(generic_hybrid_evidence(false, false, true));
    CHECK(std::holds_alternative<SemanticModel>(mtp));
    if (const auto* semantics = std::get_if<SemanticModel>(&mtp)) {
        CHECK(semantics->schema_major == 6);
        CHECK(semantics->layers.size() == 3);
        CHECK(semantics->layers[2].flags == 1);
        const auto concat = std::count_if(semantics->operators.begin(), semantics->operators.end(),
                                          [](const SemanticOperator& op) { return op.kind == OperatorKind::Concat; });
        CHECK(concat == 1);
        CHECK(semantics->states.size() == 6);
    }
    auto per_layer_kv = generic_hybrid_evidence();
    per_layer_kv.metadata.erase("model.attention.head_count_kv");
    per_layer_kv.metadata.emplace("model.attention.head_count_kv", std::vector<uint64_t>{1, 1});
    auto per_layer_result = resolve_gguf_semantics(per_layer_kv);
    CHECK(std::holds_alternative<CompatibilityReport>(per_layer_result));
    if (const auto* report = std::get_if<CompatibilityReport>(&per_layer_result)) {
        CHECK(report->code == CompatibilityError::IMPORT_SEMANTICS_MISSING);
        CHECK(report->detail == "generic resolver has no per-layer attention.head_count_kv geometry");
    }
    auto ambiguous = resolve_gguf_semantics(generic_hybrid_evidence(true));
    CHECK(std::holds_alternative<CompatibilityReport>(ambiguous));
    if (const auto* report = std::get_if<CompatibilityReport>(&ambiguous)) {
        CHECK(report->code == CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS);
        CHECK(report->detail == "layer 0 declares both recurrent and attention compositions");
    }
}

void add_generic_tensor(gguf_writer::Writer& writer, std::string name, std::vector<uint64_t> dimensions) {
    uint64_t elements = 1;
    for (uint64_t dimension : dimensions) elements *= dimension;
    gguf_writer::TensorDecl tensor;
    tensor.name = std::move(name);
    tensor.dims = std::move(dimensions);
    tensor.type = static_cast<uint32_t>(GGMLType::F32);
    tensor.data.resize(static_cast<size_t>(elements) * sizeof(float));
    writer.add_tensor(std::move(tensor));
}

void write_generic_hybrid_gguf(gguf_writer::Writer& writer) {
    writer.kv_u32("fixture.block_count", 2);
    writer.kv_u32("fixture.context_length", 16);
    writer.kv_u32("fixture.embedding_length", 2);
    writer.kv_u32("fixture.feed_forward_length", 2);
    writer.kv_u32("fixture.attention.head_count", 1);
    writer.kv_u32("fixture.attention.head_count_kv", 1);
    writer.kv_u32("fixture.attention.key_length", 2);
    writer.kv_u32("fixture.attention.value_length", 2);
    writer.kv_u32("fixture.rope.dimension_count", 2);
    writer.kv_f32("fixture.rope.freq_base", 10000.0f);
    writer.kv_f32("fixture.attention.layer_norm_rms_epsilon", 1.0e-5f);
    writer.kv_u32("fixture.ssm.conv_kernel", 2);
    writer.kv_u32("fixture.ssm.group_count", 1);
    writer.kv_u32("fixture.ssm.inner_size", 2);
    writer.kv_u32("fixture.ssm.state_size", 2);
    writer.kv_u32("fixture.ssm.time_step_rank", 1);
    writer.kv_u32("fixture.ggml.bos_token_id", 1);
    writer.kv_u32("fixture.ggml.eos_token_id", 2);
    add_generic_tensor(writer, "token_embd.weight", {2, 3});
    add_generic_tensor(writer, "output_norm.weight", {2});
    add_generic_tensor(writer, "output.weight", {2, 3});
    add_generic_tensor(writer, "blk.0.attn_norm.weight", {2});
    add_generic_tensor(writer, "blk.0.attn_qkv.weight", {2, 6});
    add_generic_tensor(writer, "blk.0.attn_gate.weight", {2, 2});
    add_generic_tensor(writer, "blk.0.ssm_alpha.weight", {2, 1});
    add_generic_tensor(writer, "blk.0.ssm_beta.weight", {2, 1});
    add_generic_tensor(writer, "blk.0.ssm_a", {1});
    add_generic_tensor(writer, "blk.0.ssm_conv1d.weight", {2, 6});
    add_generic_tensor(writer, "blk.0.ssm_dt.bias", {1});
    add_generic_tensor(writer, "blk.0.ssm_norm.weight", {2});
    add_generic_tensor(writer, "blk.0.ssm_out.weight", {2, 2});
    add_generic_tensor(writer, "blk.0.post_attention_norm.weight", {2});
    add_generic_tensor(writer, "blk.0.ffn_gate.weight", {2, 2});
    add_generic_tensor(writer, "blk.0.ffn_up.weight", {2, 2});
    add_generic_tensor(writer, "blk.0.ffn_down.weight", {2, 2});
    add_generic_tensor(writer, "blk.1.attn_norm.weight", {2});
    add_generic_tensor(writer, "blk.1.attn_q.weight", {2, 2});
    add_generic_tensor(writer, "blk.1.attn_k.weight", {2, 2});
    add_generic_tensor(writer, "blk.1.attn_v.weight", {2, 2});
    add_generic_tensor(writer, "blk.1.attn_output.weight", {2, 2});
    add_generic_tensor(writer, "blk.1.post_attention_norm.weight", {2});
    add_generic_tensor(writer, "blk.1.ffn_gate.weight", {2, 2});
    add_generic_tensor(writer, "blk.1.ffn_up.weight", {2, 2});
    add_generic_tensor(writer, "blk.1.ffn_down.weight", {2, 2});
}

void test_imports_generic_gguf_without_bundled_rule() {
    gguf_writer::Writer writer;
    write_generic_hybrid_gguf(writer);
    CHECK(writer.write_file("test_gguf_import_generic.gguf"));
    auto artifacts = ArtifactSet::load_single_file("test_gguf_import_generic.gguf");
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (auto* set = std::get_if<ArtifactSet>(&artifacts)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) {
            auto imported = import_gguf(*package);
            CHECK(std::holds_alternative<SemanticModel>(imported));
            if (const auto* semantic = std::get_if<SemanticModel>(&imported)) {
                CHECK(semantic->layers.size() == 2);
                CHECK(semantic->states.size() == 4);
            }
            auto loaded = load_validated_gguf(*package);
            CHECK(std::holds_alternative<ValidatedPackage>(loaded));
            if (const auto* validated = std::get_if<ValidatedPackage>(&loaded)) {
                CHECK(validated->runtime_package()->semantics().layers.size() == 2);
                CHECK(validated->diagnostics().rule_id == "gguf-semantic-resolver-v1");
            }
        }
    }
    remove("test_gguf_import_generic.gguf");
}

void test_rejects_malformed_signed_multi_rope_metadata() {
    gguf_writer::Writer writer;
    write_generic_hybrid_gguf(writer);
    writer.kv_arr_i32("fixture.rope.dimension_sections", {1, 1, 0, 0});
    CHECK(writer.write_file("test_gguf_import_mrope.gguf"));
    auto artifacts = ArtifactSet::load_single_file("test_gguf_import_mrope.gguf");
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (auto* set = std::get_if<ArtifactSet>(&artifacts)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) {
            auto imported = import_gguf(*package);
            CHECK(std::holds_alternative<CompatibilityReport>(imported));
            if (const auto* report = std::get_if<CompatibilityReport>(&imported)) {
                CHECK(report->code == CompatibilityError::IMPORT_SEMANTICS_MISSING);
                CHECK(report->detail == "multi-section RoPE metadata is malformed");
            }
        }
    }
    remove("test_gguf_import_mrope.gguf");
}

void test_generic_artifact_when_provided(const char* path) {
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (auto* set = std::get_if<ArtifactSet>(&artifacts)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) {
            auto imported = import_gguf(*package);
            if (const auto* report = std::get_if<CompatibilityReport>(&imported)) {
                fprintf(stderr, "generic import: code=%u detail=%s\n", static_cast<unsigned>(report->code), report->detail.c_str());
            }
            CHECK(std::holds_alternative<SemanticModel>(imported));
            if (const auto* semantic = std::get_if<SemanticModel>(&imported)) {
                const size_t recurrent = static_cast<size_t>(std::count_if(
                    semantic->operators.begin(), semantic->operators.end(),
                    [](const SemanticOperator& op) { return op.kind == OperatorKind::GatedDeltaNet; }));
                const size_t attention = static_cast<size_t>(std::count_if(
                    semantic->operators.begin(), semantic->operators.end(),
                    [](const SemanticOperator& op) { return op.kind == OperatorKind::CausalAttention; }));
                const size_t speculative = static_cast<size_t>(std::count_if(
                    semantic->layers.begin(), semantic->layers.end(),
                    [](const SemanticLayer& layer) { return layer.flags == kSemanticLayerFlagSpeculative; }));
                fprintf(stderr, "generic import: layers=%zu recurrent=%zu attention=%zu speculative=%zu\n",
                        semantic->layers.size(), recurrent, attention, speculative);
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    test_imports_checked_gguf_bytes();
    test_imports_exact_blocked_q4k_plane();
    test_generic_resolver_builds_mixed_known_operator_schedule();
    test_imports_generic_gguf_without_bundled_rule();
    test_rejects_malformed_signed_multi_rope_metadata();
    if (argc == 3 && std::strcmp(argv[1], "--generic") == 0) test_generic_artifact_when_provided(argv[2]);
    return test_summary("test_gguf_import");
}
