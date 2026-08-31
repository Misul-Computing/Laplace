#include <algorithm>
#include <cstdio>
#include <cstring>
#include <optional>
#include <type_traits>
#include <variant>

#include "artifact_set.h"
#include "compat_rule.h"
#include "gguf_fact_keys.h"
#include "gguf_index.h"
#include "gguf_writer.h"
#include "source_compiler_graph_proof.h"
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
    writer.kv_str("general.architecture", "fixture");
    writer.kv_u32("layers", 1);
    writer.kv_f32("rope_base_bits", 1000000.0f);
    gguf_writer::TensorDecl tensor;
    tensor.name = "blk.0.weight";
    tensor.dims = {2, 2};
    tensor.type = static_cast<uint32_t>(GGMLType::F32);
    tensor.data.resize(4 * sizeof(float));
    writer.add_tensor(std::move(tensor));
    CHECK(writer.write_file("/private/tmp/laplace-test-gguf-import.gguf"));

    auto artifacts = ArtifactSet::load_single_file("/private/tmp/laplace-test-gguf-import.gguf");
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
    remove("/private/tmp/laplace-test-gguf-import.gguf");
}

void test_imports_exact_blocked_q4k_plane() {
    gguf_writer::Writer writer;
    writer.kv_str("general.architecture", "fixture");
    writer.kv_u32("general.quantization_version", 2);
    writer.kv_u32("layers", 1);
    gguf_writer::TensorDecl tensor;
    tensor.name = "blk.0.weight";
    tensor.dims = {256, 1};
    tensor.type = static_cast<uint32_t>(GGMLType::Q4_K);
    tensor.data.resize(144);
    writer.add_tensor(std::move(tensor));
    CHECK(writer.write_file("/private/tmp/laplace-test-gguf-import-q4k.gguf"));

    auto artifacts = ArtifactSet::load_single_file("/private/tmp/laplace-test-gguf-import-q4k.gguf");
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
    remove("/private/tmp/laplace-test-gguf-import-q4k.gguf");
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

PackageEvidence heterogeneous_kv_evidence(bool renamed_namespace = false,
                                          std::vector<uint64_t> kv_heads = {1, 2}) {
    auto package = generic_hybrid_evidence();
    package.metadata.erase("model.attention.head_count");
    package.metadata.emplace("model.attention.head_count", uint64_t{2});
    package.metadata.erase("model.attention.head_count_kv");
    package.metadata.emplace("model.attention.head_count_kv", std::move(kv_heads));
    for (PackageTensorEvidence& tensor : package.tensors) {
        if (tensor.name == "blk.1.attn_q.weight") {
            tensor.dimensions = {256, 512};
        } else if (tensor.name == "blk.1.attn_k.weight" || tensor.name == "blk.1.attn_v.weight") {
            tensor.dimensions = {256, 512};
        } else if (tensor.name == "blk.1.attn_output.weight") {
            tensor.dimensions = {512, 256};
        }
    }
    if (renamed_namespace) {
        std::map<std::string, PackageMetadataValue> renamed;
        for (const auto& [key, value] : package.metadata) {
            std::string renamed_key = key;
            if (renamed_key.starts_with("model.")) renamed_key.replace(0, 6, "renamed.");
            renamed.emplace(std::move(renamed_key), value);
        }
        package.metadata = std::move(renamed);
    }
    return package;
}

PackageTensorEvidence routed_tensor(std::string name, std::vector<uint64_t> dimensions) {
    uint64_t elements = 1;
    for (const uint64_t dimension : dimensions) elements *= dimension;
    PackageTensorEvidence tensor;
    tensor.name = std::move(name);
    tensor.dimensions = std::move(dimensions);
    tensor.storage_type = ScalarType::F32;
    tensor.layout = PhysicalLayoutKind::ContiguousRowMajor;
    tensor.quantization = QuantizationKind::None;
    tensor.artifact_id = ArtifactId{0};
    tensor.length = elements * sizeof(float);
    tensor.alignment = 32;
    return tensor;
}

PackageEvidence raw_routed_evidence() {
    auto package = generic_hybrid_evidence();
    uint64_t offset = 0;
    for (const auto& tensor : package.tensors) offset = std::max(offset, tensor.offset + tensor.length);
    const auto append = [&](std::string name, std::vector<uint64_t> dimensions) {
        auto tensor = routed_tensor(std::move(name), std::move(dimensions));
        tensor.offset = offset;
        offset = (offset + tensor.length + 31) & ~uint64_t{31};
        package.tensors.push_back(std::move(tensor));
    };
    append("blk.0.ffn_gate_inp.weight", {256, 4});
    append("blk.0.ffn_gate_inp.scale", {4});
    append("blk.0.ffn_gate_up_exps.weight", {256, 128, 4});
    append("blk.0.ffn_down_exps.weight", {256, 128, 4});
    append("blk.0.ffn_down_exps.scale", {4});
    return package;
}

void check_raw_routed_rejected(const PackageEvidence& package) {
    constexpr std::string_view expected_detail =
        "routed GGUF evidence requires a complete semantic manifest or closed declarative source schema";
    const auto resolved = resolve_gguf_semantics(package);
    CHECK(std::holds_alternative<CompatibilityReport>(resolved));
    if (const auto* report = std::get_if<CompatibilityReport>(&resolved)) {
        CHECK(report->code == CompatibilityError::IMPORT_SEMANTICS_MISSING);
        CHECK(report->detail == expected_detail);
    }
}

void test_generic_resolver_rejects_routed_moe_without_authority() {
    const PackageEvidence complete = raw_routed_evidence();
    check_raw_routed_rejected(complete);

    auto reordered = complete;
    std::reverse(reordered.tensors.begin(), reordered.tensors.end());
    check_raw_routed_rejected(reordered);

    auto partial = complete;
    partial.tensors.erase(std::remove_if(partial.tensors.begin(), partial.tensors.end(), [](const auto& tensor) {
        return tensor.name != "blk.0.ffn_gate_inp.scale" &&
               tensor.name.find("ffn_gate_inp.scale") == std::string::npos &&
               tensor.name.find("ffn_gate_inp.weight") == std::string::npos;
    }), partial.tensors.end());
    check_raw_routed_rejected(partial);

    auto optional_only = generic_hybrid_evidence();
    uint64_t optional_offset = 0;
    for (const auto& tensor : optional_only.tensors) optional_offset = std::max(optional_offset, tensor.offset + tensor.length);
    auto optional = routed_tensor("blk.0.pre_ffw_norm_2.weight", {256});
    optional.offset = optional_offset;
    optional_only.tensors.push_back(std::move(optional));
    check_raw_routed_rejected(optional_only);

    auto extra = complete;
    uint64_t extra_offset = 0;
    for (const auto& tensor : extra.tensors) extra_offset = std::max(extra_offset, tensor.offset + tensor.length);
    auto unknown = routed_tensor("blk.0.unclaimed.weight", {256});
    unknown.offset = extra_offset;
    extra.tensors.push_back(std::move(unknown));
    check_raw_routed_rejected(extra);

    auto ambiguous = complete;
    auto alias = routed_tensor("blk.0.ffn_gate_exps.weight", {256, 128, 4});
    alias.offset = extra_offset;
    ambiguous.tensors.push_back(std::move(alias));
    check_raw_routed_rejected(ambiguous);

    auto wrong_format = complete;
    if (auto found = std::find_if(wrong_format.tensors.begin(), wrong_format.tensors.end(), [](const auto& tensor) {
            return tensor.name == "blk.0.ffn_gate_up_exps.weight";
        }); found != wrong_format.tensors.end()) {
        found->storage_type = ScalarType::U8;
        found->layout = PhysicalLayoutKind::GgufBlocked;
        found->quantization = QuantizationKind::BlockedAffine;
        found->block_elements = 32;
        found->block_bytes = 22;
    }
    check_raw_routed_rejected(wrong_format);
}

void add_generic_tensor(gguf_writer::Writer& writer, std::string name, std::vector<uint64_t> dimensions);

void test_raw_routed_moe_never_becomes_product_authority() {
    gguf_writer::Writer writer;
    writer.kv_str("general.architecture", "fixture");
    add_generic_tensor(writer, "blk.0.ffn_gate_inp.weight", {2, 2});
    const char* path = "/private/tmp/laplace-test-gguf-import-routed-authority.gguf";
    CHECK(writer.write_file(path));
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (auto* set = std::get_if<ArtifactSet>(&artifacts)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) {
            auto loaded = load_validated_gguf(*package);
            CHECK(std::holds_alternative<CompatibilityReport>(loaded));
            CHECK(!std::holds_alternative<ValidatedPackage>(loaded));
            if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
                CHECK(report->code == CompatibilityError::IMPORT_SEMANTICS_MISSING);
                CHECK(report->detail ==
                      "routed GGUF evidence requires a complete semantic manifest or closed declarative source schema");
            }
        }
    }
    remove(path);
}

void check_heterogeneous_kv_geometry(const PackageEvidence& package) {
    auto resolved = resolve_gguf_semantics(package);
    CHECK(std::holds_alternative<SemanticModel>(resolved));
    if (const auto* semantics = std::get_if<SemanticModel>(&resolved)) {
        const auto attention = std::find_if(semantics->operators.begin(), semantics->operators.end(),
                                            [](const SemanticOperator& op) {
                                                return op.kind == OperatorKind::CausalAttention;
                                            });
        CHECK(attention != semantics->operators.end());
        if (attention != semantics->operators.end()) {
            const auto* payload = std::get_if<CausalAttentionPayload>(&attention->payload);
            CHECK(payload != nullptr);
            if (payload) {
                CHECK(payload->query_heads == 2);
                CHECK(payload->kv_heads == 2);
                CHECK(payload->head_dimension == 256);
            }
        }
        const auto key_state = std::find_if(semantics->states.begin(), semantics->states.end(),
                                            [](const SemanticState& state) {
                                                return state.kind == StateKind::KeyCache;
                                            });
        CHECK(key_state != semantics->states.end());
        if (key_state != semantics->states.end()) {
            CHECK(key_state->dimensions.size() == 3);
            CHECK(key_state->dimensions[1].constant_or_symbol == 2);
        }
    }
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
        CHECK(semantics->schema_major == 8);
        CHECK(semantics->layers.size() == 2);
        CHECK(std::all_of(semantics->layers.begin(), semantics->layers.end(),
                          [](const SemanticLayer& layer) { return layer.flags == 0; }));
        const auto concat = std::count_if(semantics->operators.begin(), semantics->operators.end(),
                                          [](const SemanticOperator& op) { return op.kind == OperatorKind::Concat; });
        CHECK(concat == 0);
        CHECK(semantics->states.size() == 4);
        const size_t inactive = static_cast<size_t>(std::count_if(
            semantics->tensors.begin(), semantics->tensors.end(),
            [](const SemanticTensor& tensor) {
                return tensor.flags == kSemanticTensorFlagInactiveProgram;
            }));
        CHECK(inactive > 0);
        for (const SemanticOperator& op : semantics->operators) {
            for (uint32_t tensor_id : op.tensors) {
                CHECK(tensor_id < semantics->tensors.size());
                if (tensor_id < semantics->tensors.size())
                    CHECK(semantics->tensors[tensor_id].flags == 0);
            }
        }
        CHECK(std::holds_alternative<SourceCompilerGraphProof>(
            prove_source_candidate_graph(*semantics)));
    }
    check_heterogeneous_kv_geometry(heterogeneous_kv_evidence());
    check_heterogeneous_kv_geometry(heterogeneous_kv_evidence(true));
    auto malformed_kv = heterogeneous_kv_evidence(false, {1, 2, 3});
    auto malformed_result = resolve_gguf_semantics(malformed_kv);
    CHECK(std::holds_alternative<CompatibilityReport>(malformed_result));
    if (const auto* report = std::get_if<CompatibilityReport>(&malformed_result)) {
        CHECK(report->code == CompatibilityError::IMPORT_SEMANTICS_MISSING);
        CHECK(report->detail == "generic resolver per-layer attention.head_count_kv length must be 1 or block_count");
    }
    auto ambiguous = resolve_gguf_semantics(generic_hybrid_evidence(true));
    CHECK(std::holds_alternative<CompatibilityReport>(ambiguous));
    if (const auto* report = std::get_if<CompatibilityReport>(&ambiguous)) {
        CHECK(report->code == CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS);
        CHECK(report->detail == "layer 0 declares both recurrent and attention compositions");
    }
}

void test_generic_resolver_emits_grouped_qk_normalization_geometry() {
    auto package = generic_hybrid_evidence(false, true);
    package.metadata["model.attention.head_count"] = uint64_t{2};
    for (PackageTensorEvidence& tensor : package.tensors) {
        if (tensor.name == "blk.1.attn_q.weight") tensor.dimensions = {256, 1024};
        if (tensor.name == "blk.1.attn_output.weight") tensor.dimensions = {512, 256};
    }

    auto resolved = resolve_gguf_semantics(package);
    CHECK(std::holds_alternative<SemanticModel>(resolved));
    if (const auto* semantics = std::get_if<SemanticModel>(&resolved)) {
        CHECK(semantics->schema_major == 8);
        uint32_t grouped = 0;
        for (const SemanticOperator& op : semantics->operators) {
            if (op.kind != OperatorKind::RmsNorm || op.tensors.size() != 1) continue;
            const TensorRole role = semantics->tensors[op.tensors[0]].role;
            if (role != TensorRole::AttentionQueryNormWeight &&
                role != TensorRole::AttentionKeyNormWeight) continue;
            const auto* payload = std::get_if<RmsNormPayload>(&op.payload);
            CHECK(payload != nullptr);
            if (payload) {
                CHECK(payload->affine_geometry == RmsNormAffineGeometry::SharedAcrossGroups);
                CHECK(payload->reduction_extent == 256);
                ++grouped;
            }
        }
        CHECK(grouped == 2);
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
    writer.kv_str("general.architecture", "fixture");
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

void test_physical_index_does_not_require_architecture() {
    gguf_writer::Writer writer;
    gguf_writer::TensorDecl tensor;
    tensor.name = "tensor.weight";
    tensor.dims = {2, 2};
    tensor.type = static_cast<uint32_t>(GGMLType::F32);
    tensor.data.resize(4 * sizeof(float), 0);
    writer.add_tensor(std::move(tensor));
    const char* path = "/private/tmp/laplace-test-gguf-import-missing-architecture.gguf";
    CHECK(writer.write_file(path));
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (auto* set = std::get_if<ArtifactSet>(&artifacts)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) {
            auto physical = build_gguf_artifact_index(*package);
            CHECK(std::holds_alternative<ArtifactIndex>(physical));
            auto imported = import_gguf(*package);
            CHECK(std::holds_alternative<CompatibilityReport>(imported));
            if (const auto* report = std::get_if<CompatibilityReport>(&imported)) {
                CHECK(report->code == CompatibilityError::IMPORT_SEMANTICS_MISSING);
            }
            auto loaded = load_validated_gguf(*package);
            CHECK(std::holds_alternative<CompatibilityReport>(loaded));
            if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
                CHECK(report->code == CompatibilityError::IMPORT_SEMANTICS_MISSING);
            }
        }
    }
    remove(path);
}

std::optional<ArtifactIndex> build_physical_index(const char* path) {
    auto artifacts = ArtifactSet::load_single_file(path);
    if (!std::holds_alternative<ArtifactSet>(artifacts)) return std::nullopt;
    auto* set = std::get_if<ArtifactSet>(&artifacts);
    auto view = set->view(ArtifactId{0});
    if (!std::holds_alternative<PackageView>(view)) return std::nullopt;
    auto result = build_gguf_artifact_index(*std::get_if<PackageView>(&view));
    if (!std::holds_alternative<ArtifactIndex>(result)) return std::nullopt;
    return std::move(*std::get_if<ArtifactIndex>(&result));
}

void test_physical_index_emits_source_independent_facts() {
    auto write = [](const char* path, const char* architecture, uint32_t block_count) {
        gguf_writer::Writer writer;
        writer.kv_str("general.architecture", architecture);
        writer.kv_u32("fixture.block_count", block_count);
        writer.kv_u32("fixture.context_length", 128);
        writer.kv_u32("fixture.embedding_length", 256);
        writer.kv_u32("fixture.feed_forward_length", 1024);
        writer.kv_u32("fixture.attention.head_count", 8);
        writer.kv_arr_u32("fixture.attention.head_count_kv", {2, 4});
        writer.kv_u32("fixture.attention.key_length", 64);
        writer.kv_u32("fixture.attention.value_length", 64);
        writer.kv_u32("fixture.rope.dimension_count", 64);
        writer.kv_f32("fixture.rope.freq_base", 10000.0f);
        writer.kv_f32("fixture.attention.layer_norm_rms_epsilon", 1.0e-5f);
        writer.kv_u32("fixture.ggml.bos_token_id", 1);
        writer.kv_u32("fixture.ggml.eos_token_id", 2);
        add_generic_tensor(writer, "tensor.weight", {2, 2});
        return writer.write_file(path);
    };
    const char* first_path = "/private/tmp/laplace-test-gguf-facts-a.gguf";
    const char* second_path = "/private/tmp/laplace-test-gguf-facts-b.gguf";
    CHECK(write(first_path, "dense", 2));
    CHECK(write(second_path, "recurrent", 2));
    auto first = build_physical_index(first_path);
    auto second = build_physical_index(second_path);
    CHECK(first.has_value() && second.has_value());
    if (first && second) {
        CHECK(first->metadata_facts().size() == gguf_fact_keys::descriptors.size());
        CHECK(first->normalized_digest() == second->normalized_digest());
        CHECK(first->canonical_bytes() == second->canonical_bytes());
        const auto block = std::find_if(first->metadata_facts().begin(), first->metadata_facts().end(),
                                       [](const ArtifactFact& fact) {
                                           return fact.key == gguf_fact_keys::block_count;
                                       });
        CHECK(block != first->metadata_facts().end());
        if (block != first->metadata_facts().end()) {
            CHECK(block->state == ArtifactFactState::Present);
            CHECK(std::holds_alternative<uint64_t>(block->value));
            CHECK(std::get<uint64_t>(block->value) == 2);
            CHECK(block->source.length != 0);
            CHECK(block->source.offset < first->artifacts()[0].bytes().size());
        }
        const auto kv = std::find_if(first->metadata_facts().begin(), first->metadata_facts().end(),
                                    [](const ArtifactFact& fact) {
                                        return fact.key == gguf_fact_keys::attention_head_count_kv;
                                    });
        CHECK(kv != first->metadata_facts().end());
        if (kv != first->metadata_facts().end()) {
            CHECK(kv->scope.layer == UINT32_MAX);
            CHECK(std::holds_alternative<std::vector<uint64_t>>(kv->value));
            CHECK((std::get<std::vector<uint64_t>>(kv->value) == std::vector<uint64_t>{2, 4}));
        }
    }
    auto changed_path = "/private/tmp/laplace-test-gguf-facts-changed.gguf";
    CHECK(write(changed_path, "recurrent", 3));
    auto changed = build_physical_index(changed_path);
    CHECK(changed.has_value());
    if (first && changed) CHECK(first->normalized_digest() != changed->normalized_digest());
    remove(first_path);
    remove(second_path);
    remove(changed_path);
}

void test_physical_index_records_fact_states() {
    gguf_writer::Writer writer;
    writer.kv_u32("fixture.block_count", 2);
    writer.kv_str("fixture.context_length", "wrong");
    writer.kv_u32("one.embedding_length", 256);
    writer.kv_u32("two.embedding_length", 256);
    add_generic_tensor(writer, "tensor.weight", {2, 2});
    const char* path = "/private/tmp/laplace-test-gguf-fact-states.gguf";
    CHECK(writer.write_file(path));
    auto index = build_physical_index(path);
    CHECK(index.has_value());
    if (index) {
        const auto find = [&](CanonicalFactKey key) {
            return std::find_if(index->metadata_facts().begin(), index->metadata_facts().end(),
                                [&](const ArtifactFact& fact) { return fact.key == key; });
        };
        const auto block = find(gguf_fact_keys::block_count);
        const auto context = find(gguf_fact_keys::context_length);
        const auto embedding = find(gguf_fact_keys::embedding_length);
        CHECK(block != index->metadata_facts().end() && block->state == ArtifactFactState::Present);
        CHECK(context != index->metadata_facts().end() && context->state == ArtifactFactState::WrongType);
        CHECK(embedding != index->metadata_facts().end() && embedding->state == ArtifactFactState::Ambiguous);
        CHECK(index->normalized_digest() != Sha256Digest{});
    }
    remove(path);
}

void test_product_import_requires_quantization_version() {
    gguf_writer::Writer writer;
    writer.kv_str("general.architecture", "fixture");
    gguf_writer::TensorDecl tensor;
    tensor.name = "tensor.weight";
    tensor.dims = {256, 1};
    tensor.type = static_cast<uint32_t>(GGMLType::Q4_K);
    tensor.data.resize(144);
    writer.add_tensor(std::move(tensor));
    const char* path = "/private/tmp/laplace-test-gguf-import-missing-quant-version.gguf";
    CHECK(writer.write_file(path));
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (auto* set = std::get_if<ArtifactSet>(&artifacts)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) {
            auto imported = import_gguf(*package);
            CHECK(std::holds_alternative<CompatibilityReport>(imported));
            if (const auto* report = std::get_if<CompatibilityReport>(&imported)) {
                CHECK(report->code == CompatibilityError::PACKAGE_BOUNDS_INVALID);
                CHECK(report->detail == "quantized GGUF is missing general.quantization_version");
            }
        }
    }
    remove(path);
}

void test_imports_generic_gguf_without_bundled_rule() {
    gguf_writer::Writer writer;
    write_generic_hybrid_gguf(writer);
    CHECK(writer.write_file("/private/tmp/laplace-test-gguf-import-generic.gguf"));
    auto artifacts = ArtifactSet::load_single_file("/private/tmp/laplace-test-gguf-import-generic.gguf");
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
            CHECK(std::holds_alternative<CompatibilityReport>(loaded));
            if (const auto* report = std::get_if<CompatibilityReport>(&loaded)) {
                CHECK(report->code == CompatibilityError::IR_VERSION_UNSUPPORTED);
            }
        }
    }
    remove("/private/tmp/laplace-test-gguf-import-generic.gguf");
}

void test_validated_gguf_derives_tokenizer_digest_from_metadata() {
    gguf_writer::Writer writer;
    writer.kv_str("general.architecture", "fixture");
    writer.kv_u32("fixture.block_count", 1);
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
    writer.kv_u32("fixture.ggml.bos_token_id", 1);
    writer.kv_u32("fixture.ggml.eos_token_id", 2);
    writer.kv_arr_str("tokenizer.ggml.tokens", {"<unk>", "a", "b"});
    writer.kv_str("tokenizer.ggml.model", "gpt2");
    const auto add_tensor = [&](const char* name, std::vector<uint64_t> dimensions) {
        uint64_t elements = 1;
        for (uint64_t dimension : dimensions) elements *= dimension;
        gguf_writer::TensorDecl tensor;
        tensor.name = name;
        tensor.dims = std::move(dimensions);
        tensor.type = static_cast<uint32_t>(GGMLType::F32);
        tensor.data.resize(static_cast<size_t>(elements) * sizeof(float));
        writer.add_tensor(std::move(tensor));
    };
    add_tensor("token_embd.weight", {2, 3});
    add_tensor("output_norm.weight", {2});
    add_tensor("output.weight", {2, 3});
    add_tensor("blk.0.attn_norm.weight", {2});
    add_tensor("blk.0.attn_q.weight", {2, 2});
    add_tensor("blk.0.attn_k.weight", {2, 2});
    add_tensor("blk.0.attn_v.weight", {2, 2});
    add_tensor("blk.0.attn_output.weight", {2, 2});
    add_tensor("blk.0.post_attention_norm.weight", {2});
    add_tensor("blk.0.ffn_gate.weight", {2, 2});
    add_tensor("blk.0.ffn_up.weight", {2, 2});
    add_tensor("blk.0.ffn_down.weight", {2, 2});
    const char* path = "/private/tmp/laplace-test-gguf-import-token-contract.gguf";
    CHECK(writer.write_file(path));
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (auto* set = std::get_if<ArtifactSet>(&artifacts)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) {
            auto loaded = load_validated_gguf(*package);
            CHECK(std::holds_alternative<ValidatedPackage>(loaded));
            if (auto* validated = std::get_if<ValidatedPackage>(&loaded)) {
                const auto runtime = validated->runtime_package();
                CHECK(runtime != nullptr);
                if (runtime) {
                    const std::array<uint8_t, 32> zero_digest{};
                    CHECK(runtime->semantics().tokenizer_digest != zero_digest);
                    CHECK(runtime->semantics().template_digest == zero_digest);
                }
            }
        }
    }
    remove(path);
}

void test_rejects_malformed_signed_multi_rope_metadata() {
    gguf_writer::Writer writer;
    write_generic_hybrid_gguf(writer);
    writer.kv_arr_i32("fixture.rope.dimension_sections", {1, 1, 0, 0});
    CHECK(writer.write_file("/private/tmp/laplace-test-gguf-import-mrope.gguf"));
    auto artifacts = ArtifactSet::load_single_file("/private/tmp/laplace-test-gguf-import-mrope.gguf");
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
    remove("/private/tmp/laplace-test-gguf-import-mrope.gguf");
}

void test_bundled_artifact_when_provided(const char* path) {
    if (!path) return;
    auto artifacts = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(artifacts));
    if (auto* set = std::get_if<ArtifactSet>(&artifacts)) {
        auto view = set->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) {
            auto imported = import_expected_fixture_gguf(*package, bundled_compatibility_rules());
            CHECK(std::holds_alternative<SemanticModel>(imported));
            if (auto* semantic = std::get_if<SemanticModel>(&imported)) {
                CHECK(semantic->tensors.size() == 291);
                CHECK(semantic->tensors[0].planes[0].storage_type == ScalarType::F16);
                CHECK(semantic->tensors[290].planes[0].flags == 1);
            }
            auto validated = load_expected_fixture_gguf(*package, bundled_compatibility_rules());
            CHECK(std::holds_alternative<ValidatedPackage>(validated));
            if (auto* loaded = std::get_if<ValidatedPackage>(&validated)) {
                auto runtime = loaded->runtime_package();
                CHECK(runtime != nullptr);
                if (runtime) {
                    CHECK(runtime->semantics().operators.size() == 339);
                    CHECK(runtime->artifact_bytes(ArtifactId{0}).size() == package->bytes().size());
                    CHECK(runtime->fingerprint() != Sha256Digest{});
                }
            }
        }
    }
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
    test_generic_resolver_emits_grouped_qk_normalization_geometry();
    test_generic_resolver_rejects_routed_moe_without_authority();
    test_raw_routed_moe_never_becomes_product_authority();
    test_imports_generic_gguf_without_bundled_rule();
    test_validated_gguf_derives_tokenizer_digest_from_metadata();
    test_physical_index_does_not_require_architecture();
    test_physical_index_emits_source_independent_facts();
    test_physical_index_records_fact_states();
    test_product_import_requires_quantization_version();
    test_rejects_malformed_signed_multi_rope_metadata();
    if (argc == 2) test_bundled_artifact_when_provided(argv[1]);
    if (argc == 3 && std::strcmp(argv[1], "--generic") == 0) test_generic_artifact_when_provided(argv[2]);
    return test_summary("test_gguf_import");
}
