#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace Laplace {

struct ArtifactId {
    uint32_t value = UINT32_MAX;
    friend bool operator==(ArtifactId, ArtifactId) = default;
};

struct CanonicalFactKey {
    uint32_t value = UINT32_MAX;
    friend bool operator==(CanonicalFactKey, CanonicalFactKey) = default;
};

enum class CompatibilityStage : uint16_t {
    Package = 1,
    Semantic = 2,
    PhysicalFormat = 3,
    Capability = 4,
    StateRollback = 5,
    Tokenizer = 6,
    Benchmark = 7,
    Rule = Semantic,
    Import = Semantic,
    Plan = Capability,
    Session = Capability,
    State = StateRollback,
};

enum class CompatibilityPhase : uint8_t {
    Package = 1,
    Import = 2,
    Semantic = 3,
    Plan = 4,
    Session = 5,
    StateRollback = 6,
    Tokenizer = 7,
    Benchmark = 8,
};

enum class CompatibilityError : uint16_t {
    PACKAGE_BAD_MAGIC = 1,
    PACKAGE_VERSION_UNSUPPORTED = 2,
    PACKAGE_BOUNDS_INVALID = 3,
    PACKAGE_GRAPH_UNSUPPORTED = 4,
    PACKAGE_CHECKSUM_MISMATCH = 5,
    PACKAGE_SOURCE_CHANGED = 6,
    PACKAGE_SNAPSHOT_UNAVAILABLE = 38,
    RULE_VERSION_UNSUPPORTED = 7,
    RULE_LIMIT_EXCEEDED = 8,
    IMPORT_EXECUTABLE_CODE_REQUIRED = 9,
    IMPORT_SEMANTICS_MISSING = 10,
    IMPORT_SEMANTICS_AMBIGUOUS = 11,
    IMPORT_RULE_CONFLICT = 12,
    IMPORT_TENSOR_UNMAPPED = 13,
    IMPORT_TENSOR_DUPLICATE = 14,
    IR_VERSION_UNSUPPORTED = 15,
    IR_REFERENCE_INVALID = 16,
    IR_CONSTRAINT_FAILED = 17,
    IR_SHAPE_MISMATCH = 18,
    IR_LAYOUT_MISMATCH = 19,
    IR_QUANTIZATION_UNSUPPORTED = 20,
    IR_STATE_INVALID = 21,
    TOKENIZER_RUNTIME_UNSUPPORTED = 22,
    MODALITY_UNSUPPORTED = 23,
    CAPABILITY_MISSING = 24,
    KERNEL_UNAVAILABLE = 25,
    KERNEL_AMBIGUOUS = 26,
    PLAN_MEMORY_EXCEEDED = 27,
    FALLBACK_FORBIDDEN = 28,
    SESSION_CONSTRUCTION_FAILED = 29,
    STATE_ABI_MISMATCH = 30,
    RUNTIME_INPUT_INVALID = 31,
    STREAMING_UNSUPPORTED = 32,
    PLAN_CONTEXT_EXCEEDED = 33,
    RUNTIME_NUMERICAL_FAILURE = 34,
    CACHE_MODE_UNQUALIFIED = 35,
    RULE_QUALIFICATION_REQUIRED = 36,
    PACKAGE_AUTHORITY_REQUIRED = 37,
    IMPORT_SCHEMA_NOT_FOUND = 39,
    IMPORT_SCHEMA_AMBIGUOUS = 40,
    IMPORT_SCHEMA_INCOMPLETE = 41,
    IMPORT_SCHEMA_LIMIT = 42,
    IMPORT_MANIFEST_INVALID = 43,
    IMPORT_MANIFEST_DOWNGRADE = 44,
    IMPORT_CLOSURE_INCOMPLETE = 45,
    IMPORT_ALIAS_AMBIGUOUS = 46,
    IMPORT_INTERACTION_UNSUPPORTED = 47,
    AUTHORITY_INVALID = 48,
};

struct CompatibilityReport {
    CompatibilityStage stage = CompatibilityStage::Package;
    CompatibilityError code = CompatibilityError::PACKAGE_BOUNDS_INVALID;
    CompatibilityPhase phase = CompatibilityPhase::Package;
    ArtifactId artifact_id{};
    uint32_t artifact_index = UINT32_MAX;
    uint32_t layer = UINT32_MAX;
    uint32_t operator_id = UINT32_MAX;
    uint32_t tensor_id = UINT32_MAX;
    uint32_t state_id = UINT32_MAX;
    CanonicalFactKey fact_key{};
    uint64_t source_offset = UINT64_MAX;
    uint64_t source_length = 0;
    std::string metadata_key;
    std::string rule_id;
    std::string expected;
    std::string observed;
    std::string message;
    std::string detail;
};

// Refusal classification depends only on the stable error code. Location and
// message fields are diagnostic output, not planner input.
CompatibilityStage compatibility_stage(CompatibilityError code) noexcept;
CompatibilityPhase compatibility_phase(CompatibilityError code) noexcept;
std::string_view compatibility_message(CompatibilityError code) noexcept;
CompatibilityReport compatibility_report(CompatibilityError code, std::string detail = {});
CompatibilityReport package_report(CompatibilityError code, std::string detail = {});

} // namespace Laplace
