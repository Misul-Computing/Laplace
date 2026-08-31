#pragma once

#include <cstdint>
#include <array>
#include <optional>
#include <span>
#include <vector>

#include "artifact_index.h"
#include "source_compiler_graph_proof.h"
#include "source_schema.h"

namespace Laplace {

inline constexpr uint32_t kClosedV1NoId = UINT32_MAX;

// These values are adapter facts, not canonical IR fields. Equality is
// intentionally variant-strict: a signed integer and an unsigned integer
// carrying the same bits are different facts.
struct ClosedV1FactPredicate {
    CanonicalFactKey key{};
    ArtifactFactValue value = uint64_t{0};
    friend bool operator==(const ClosedV1FactPredicate&, const ClosedV1FactPredicate&) = default;
};

struct ClosedV1SourcePackage {
    SourcePackageEvidence source;
    std::vector<ClosedV1FactPredicate> facts;
};

struct ClosedV1Symbol {
    uint32_t id = kClosedV1NoId;
    // A no-id fact key makes literal the source-independent constant. Any
    // other key requires one exact uint64_t package fact.
    CanonicalFactKey fact_key{};
    uint64_t literal = 0;
    uint64_t minimum = 1;
    uint64_t maximum = UINT64_MAX;
};

struct ClosedV1Dimension {
    DimensionKind kind = DimensionKind::Constant;
    uint64_t value = 0;
};

struct ClosedV1Repeat {
    uint32_t count = 1;
    uint32_t maximum = 1;
};

struct ClosedV1TemplateControl {
    bool has_variant = false;
    ClosedV1FactPredicate variant;
    ClosedV1Repeat repeat;
};

// Tokenization and prompt formatting are separate schema authorities. Their
// predicates are checked against package facts, while only their authenticated
// digests and vocabulary reach the semantic model.
struct ClosedV1TokenizerTemplate {
    uint32_t vocabulary_size = 0;
    std::vector<ClosedV1FactPredicate> predicates;
    std::array<uint8_t, 32> digest{};
};

struct ClosedV1PromptTemplate {
    std::vector<ClosedV1FactPredicate> predicates;
    std::array<uint8_t, 32> digest{};
};

// Source names and source slots are consumed only by the binding adapter. No
// spelling is copied into ClosedV1SourceCompilation or SemanticModel.
struct ClosedV1TensorTemplate {
    uint32_t id = kClosedV1NoId;
    uint32_t source_slot_id = kClosedV1NoId;
    TensorRole role = TensorRole::TokenEmbedding;
    ScalarType logical_type = ScalarType::F32;
    std::vector<ClosedV1Dimension> dimensions;
    std::vector<ClosedV1Dimension> source_dimensions;
    std::vector<SourceNamePattern> source_names;
    PhysicalLayout layout;
    Quantization quantization;
    ScalarType plane_storage_type = ScalarType::F32;
    uint32_t physical_type_code = 0;
    uint32_t physical_block_axis = 0;
    uint32_t physical_block_elements = 1;
    uint32_t physical_block_bytes = 4;
    uint32_t plane_alignment = 64;
    uint16_t flags = 0;
    ClosedV1TemplateControl control;
};

struct ClosedV1ValueTemplate {
    uint32_t id = kClosedV1NoId;
    ScalarType logical_type = ScalarType::F32;
    std::vector<ClosedV1Dimension> dimensions;
    uint8_t flags = 0;
    ClosedV1TemplateControl control;
};

struct ClosedV1StateTemplate {
    uint32_t id = kClosedV1NoId;
    StateKind kind = StateKind::KeyCache;
    uint16_t semantic_version = 1;
    StateUpdateKind update_kind = StateUpdateKind::AppendKey;
    PositionPolicy position_policy = PositionPolicy::AppendOnly;
    std::vector<ClosedV1Dimension> dimensions;
    std::vector<StateFormat> formats;
    uint16_t flags = 0;
    ClosedV1TemplateControl control;
};

struct ClosedV1AliasTemplate {
    uint32_t source_slot_id = kClosedV1NoId;
    uint32_t target_slot_id = kClosedV1NoId;
    SourceAliasKind kind = SourceAliasKind::ExactSharedSpan;
    uint32_t proof_key = kClosedV1NoId;
};

struct ClosedV1OperatorTemplate {
    uint32_t id = kClosedV1NoId;
    OperatorKind kind = OperatorKind::Add;
    uint16_t semantic_version = 1;
    std::vector<uint32_t> inputs;
    std::vector<uint32_t> outputs;
    std::vector<uint32_t> tensors;
    std::vector<uint32_t> states;
    OperatorPayload payload = AddPayload{};
    ClosedV1TemplateControl control;
};

struct ClosedV1LayerTemplate {
    uint32_t layer_index = kClosedV1NoId;
    uint32_t first_operator = 0;
    uint32_t operator_count = 0;
    uint32_t flags = 0;
    ClosedV1TemplateControl control;
};

struct ClosedV1GraphTemplate {
    uint16_t schema_major = 1;
    uint16_t schema_minor = 0;
    uint16_t opset_major = 1;
    uint16_t opset_minor = 0;
    uint32_t maximum_context = 32768;
    uint32_t bos_id = UINT32_MAX;
    uint32_t eos_id = UINT32_MAX;
    std::vector<uint32_t> stop_ids;
    ClosedV1TokenizerTemplate tokenizer;
    ClosedV1PromptTemplate prompt;
    uint32_t input_values_first = 0;
    uint32_t input_values_count = 0;
    uint32_t output_values_first = 0;
    uint32_t output_values_count = 0;
    std::vector<ClosedV1Symbol> symbols;
    // This is a bounded declaration. V1 emits one materialized graph; a
    // count greater than one is refused until relative-id expansion exists.
    ClosedV1Repeat repeat;
    std::vector<ClosedV1TensorTemplate> tensors;
    std::vector<ClosedV1ValueTemplate> values;
    std::vector<ClosedV1OperatorTemplate> operators;
    std::vector<ClosedV1LayerTemplate> layers;
    std::vector<ClosedV1StateTemplate> states;
    std::vector<ClosedV1AliasTemplate> aliases;
};

struct ClosedV1Schema {
    uint32_t schema_id = kClosedV1NoId;
    uint16_t schema_major = 1;
    uint16_t schema_minor = 0;
    SourceFormat source_format = SourceFormat::Gguf;
    uint32_t maximum_expansion_steps = 1'000'000;
    std::vector<ClosedV1FactPredicate> predicates;
    ClosedV1GraphTemplate graph;
};

enum class ClosedV1SchemaError : uint16_t {
    None = 0,
    SchemaVersionUnsupported,
    SourceFormatMismatch,
    NoMatchingSchema,
    AmbiguousSchema,
    MissingFact,
    WrongFact,
    DuplicateId,
    InvalidTemplate,
    RepeatBoundExceeded,
    InvalidReference,
    GraphCycle,
    MissingTensorBinding,
    UnmatchedSourceTensor,
    PhysicalContractMismatch,
    GraphProofFailed,
};

struct ClosedV1SourceCompilation {
    // This is the only output authority. It is newly instantiated from the
    // closed graph and physical spans; it never carries embedded semantic
    // bytes or diagnostic source/model/family strings.
    SemanticModel model;
    SourceCompilerGraphProof graph_proof;
};

struct ClosedV1SourceSchemaResult {
    ClosedV1SchemaError error = ClosedV1SchemaError::None;
    uint32_t schema_id = kClosedV1NoId;
    CanonicalFactKey fact_key{};
    uint32_t id = kClosedV1NoId;
    uint32_t source_tensor_id = kClosedV1NoId;
    std::optional<ClosedV1SourceCompilation> compiled;

    bool success() const noexcept { return error == ClosedV1SchemaError::None; }
};

ClosedV1SourceSchemaResult compile_closed_v1_source_schema(
    const ClosedV1Schema& schema, const ClosedV1SourcePackage& package);
ClosedV1SourceSchemaResult compile_closed_v1_source_schemas(
    std::span<const ClosedV1Schema> schemas, const ClosedV1SourcePackage& package);

} // namespace Laplace
