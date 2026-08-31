#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace Laplace {

constexpr uint32_t kSourceSchemaNoId = UINT32_MAX;

struct SourceSchemaVersion {
    uint16_t major = 1;
    uint16_t minor = 0;
    friend bool operator==(const SourceSchemaVersion&, const SourceSchemaVersion&) = default;
};

enum class SourceFormat : uint16_t {
    Gguf = 1,
    Mlx = 2,
};

enum class SourceSchemaError : uint16_t {
    None = 0,
    SchemaVersionUnsupported,
    EvaluatorVersionUnsupported,
    SourceFormatMismatch,
    SchemaLimitExceeded,
    InvalidPhysicalSpan,
    DuplicateId,
    InvalidSelector,
    SelectorNotMatched,
    SelectorCardinalityMismatch,
    TensorConsumedTwice,
    UnmatchedTensor,
    MissingRequiredSlot,
    MissingOutput,
    AliasRequired,
    AliasInvalid,
    AliasMismatch,
    NoMatchingSchema,
    AmbiguousSchemas,
};

enum class SourceNamePatternKind : uint8_t {
    Exact = 1,
    AnchoredDecimal = 2,
};

// Source spellings are schema data. There is no wildcard or executable match
// callback. AnchoredDecimal accepts prefix + decimal capture + suffix only.
struct SourceNamePattern {
    SourceNamePatternKind kind = SourceNamePatternKind::Exact;
    std::string literal;
    std::string prefix;
    std::string suffix;
    uint32_t maximum_digits = 10;
};

struct PhysicalSpan {
    uint32_t artifact_id = kSourceSchemaNoId;
    uint64_t offset = 0;
    uint64_t length = 0;
    friend bool operator==(const PhysicalSpan&, const PhysicalSpan&) = default;
};

enum class SourceAliasKind : uint8_t {
    ExactSharedSpan = 1,
    TiedOutput = 2,
};

struct SourceArtifactEvidence {
    uint32_t artifact_id = kSourceSchemaNoId;
    uint64_t byte_length = 0;
};

struct SourceAliasProofEvidence {
    uint32_t proof_key = kSourceSchemaNoId;
    uint32_t source_slot_id = kSourceSchemaNoId;
    uint32_t target_slot_id = kSourceSchemaNoId;
    SourceAliasKind kind = SourceAliasKind::TiedOutput;
};

struct SourceTensorEvidence {
    uint32_t physical_tensor_id = kSourceSchemaNoId;
    std::string source_spelling;
    std::vector<uint64_t> dimensions;
    PhysicalSpan span;
    // Interpreted in the namespace of SourcePackageEvidence::format (for
    // example, a GGML type ID for GGUF or a dtype code for MLX).
    uint32_t physical_type_code = 0;
    // Physical storage is described as fixed-size blocks along one source
    // tensor axis. Scalar F32 is {1 element, 4 bytes}; GGUF Q4_K is
    // {256 elements, 144 bytes}. This also covers packed MLX planes without
    // pretending that a quantized value has an integer byte size.
    uint32_t physical_block_axis = 0;
    uint32_t physical_block_elements = 1;
    uint32_t physical_block_bytes = 4;
    // Evidence only. The evaluator never compares this field to infer an alias.
    std::array<uint8_t, 32> content_digest{};
};

struct SourcePackageEvidence {
    SourceFormat format = SourceFormat::Gguf;
    std::vector<SourceArtifactEvidence> artifacts;
    std::vector<SourceAliasProofEvidence> alias_proofs;
    std::vector<SourceTensorEvidence> tensors;
};

struct SchemaLogicalSlot {
    uint32_t id = kSourceSchemaNoId;
    uint32_t selector_id = kSourceSchemaNoId;
    std::vector<uint64_t> dimensions;
    bool required = true;
    bool output = false;
};

struct SchemaTensorSelector {
    uint32_t id = kSourceSchemaNoId;
    // A single binding_slot is the common case. binding_slots permits one
    // finite selector to bind a declared, ordered set of logical slots.
    uint32_t binding_slot = kSourceSchemaNoId;
    std::vector<uint32_t> binding_slots;
    std::vector<SourceNamePattern> names;
    std::vector<uint64_t> source_dimensions;
    uint32_t exact_cardinality = 1;
};

struct SchemaAliasTemplate {
    uint32_t source_slot_id = kSourceSchemaNoId;
    uint32_t target_slot_id = kSourceSchemaNoId;
    SourceAliasKind kind = SourceAliasKind::ExactSharedSpan;
    uint32_t proof_key = kSourceSchemaNoId;
};

struct SourceSchema {
    uint16_t schema_major = 1;
    uint16_t schema_minor = 0;
    uint16_t evaluator_major = 1;
    uint16_t evaluator_minor = 0;
    SourceFormat source_format = SourceFormat::Gguf;
    uint32_t maximum_expansion_steps = 1'000'000;
    std::vector<SchemaLogicalSlot> slots;
    std::vector<SchemaTensorSelector> selectors;
    std::vector<SchemaAliasTemplate> aliases;

    static constexpr SourceSchemaVersion version() noexcept { return {1, 0}; }
    static constexpr SourceSchemaVersion evaluator_version() noexcept { return {1, 0}; }
};

struct CompiledGraphSlot {
    uint32_t logical_slot_id = kSourceSchemaNoId;
    std::vector<uint64_t> dimensions;
    bool required = true;
    bool output = false;
};

struct CompiledBinding {
    uint32_t logical_slot_id = kSourceSchemaNoId;
    uint32_t physical_tensor_id = kSourceSchemaNoId;
    PhysicalSpan span;
    uint32_t physical_type_code = 0;
    uint32_t physical_block_axis = 0;
    uint32_t physical_block_elements = 1;
    uint32_t physical_block_bytes = 4;
};

struct CompiledAlias {
    uint32_t source_slot_id = kSourceSchemaNoId;
    uint32_t target_slot_id = kSourceSchemaNoId;
    SourceAliasKind kind = SourceAliasKind::ExactSharedSpan;
    uint32_t proof_key = kSourceSchemaNoId;
};

// This object intentionally has no source/model/family strings or runtime
// policy. Canonical bytes are derived only from logical slots and declarations
// plus physical bindings.
struct CompiledSourceSchema {
    std::vector<CompiledGraphSlot> graph_slots;
    std::vector<CompiledBinding> bindings;
    std::vector<CompiledAlias> aliases;
    std::vector<uint8_t> graph_bytes;
    std::vector<uint8_t> binding_bytes;
};

struct SourceSchemaResult {
    SourceSchemaError error = SourceSchemaError::None;
    uint32_t matching_schema_count = 0;
    uint32_t selector_id = kSourceSchemaNoId;
    uint32_t slot_id = kSourceSchemaNoId;
    uint32_t tensor_id = kSourceSchemaNoId;
    CompiledSourceSchema compiled;

    bool success() const noexcept { return error == SourceSchemaError::None; }
};

class SourceSchemaEvaluator {
public:
    static constexpr SourceSchemaVersion version() noexcept { return {1, 0}; }

    static SourceSchemaResult evaluate(const SourceSchema& schema,
                                        const SourcePackageEvidence& package);
    static SourceSchemaResult evaluate(std::span<const SourceSchema> schemas,
                                       const SourcePackageEvidence& package);
};

SourceSchemaResult compile_source_schema(const SourceSchema& schema,
                                         const SourcePackageEvidence& package);
SourceSchemaResult evaluate_source_schemas(std::span<const SourceSchema> schemas,
                                           const SourcePackageEvidence& package);

} // namespace Laplace
