#include "compatibility_report.h"

#include <utility>

namespace Laplace {

CompatibilityStage compatibility_stage(CompatibilityError code) noexcept {
    switch (code) {
    case CompatibilityError::PACKAGE_BAD_MAGIC:
    case CompatibilityError::PACKAGE_VERSION_UNSUPPORTED:
    case CompatibilityError::PACKAGE_BOUNDS_INVALID:
    case CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED:
    case CompatibilityError::PACKAGE_CHECKSUM_MISMATCH:
    case CompatibilityError::PACKAGE_SOURCE_CHANGED:
        return CompatibilityStage::Package;
    case CompatibilityError::RULE_VERSION_UNSUPPORTED:
    case CompatibilityError::RULE_LIMIT_EXCEEDED:
    case CompatibilityError::IMPORT_EXECUTABLE_CODE_REQUIRED:
    case CompatibilityError::IMPORT_SEMANTICS_MISSING:
    case CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS:
    case CompatibilityError::IMPORT_RULE_CONFLICT:
    case CompatibilityError::IMPORT_TENSOR_UNMAPPED:
    case CompatibilityError::IMPORT_TENSOR_DUPLICATE:
    case CompatibilityError::IR_VERSION_UNSUPPORTED:
    case CompatibilityError::IR_REFERENCE_INVALID:
    case CompatibilityError::IR_CONSTRAINT_FAILED:
    case CompatibilityError::IR_SHAPE_MISMATCH:
    case CompatibilityError::MODALITY_UNSUPPORTED:
    case CompatibilityError::RUNTIME_INPUT_INVALID:
        return CompatibilityStage::Semantic;
    case CompatibilityError::IR_LAYOUT_MISMATCH:
    case CompatibilityError::IR_QUANTIZATION_UNSUPPORTED:
        return CompatibilityStage::PhysicalFormat;
    case CompatibilityError::IR_STATE_INVALID:
    case CompatibilityError::STATE_ABI_MISMATCH:
    case CompatibilityError::RUNTIME_NUMERICAL_FAILURE:
        return CompatibilityStage::StateRollback;
    case CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED:
        return CompatibilityStage::Tokenizer;
    case CompatibilityError::CAPABILITY_MISSING:
    case CompatibilityError::KERNEL_UNAVAILABLE:
    case CompatibilityError::KERNEL_AMBIGUOUS:
    case CompatibilityError::PLAN_MEMORY_EXCEEDED:
    case CompatibilityError::FALLBACK_FORBIDDEN:
    case CompatibilityError::SESSION_CONSTRUCTION_FAILED:
    case CompatibilityError::STREAMING_UNSUPPORTED:
    case CompatibilityError::PLAN_CONTEXT_EXCEEDED:
        return CompatibilityStage::Capability;
    case CompatibilityError::CACHE_MODE_UNQUALIFIED:
    case CompatibilityError::RULE_QUALIFICATION_REQUIRED:
        return CompatibilityStage::Benchmark;
    }
    return CompatibilityStage::Semantic;
}

CompatibilityPhase compatibility_phase(CompatibilityError code) noexcept {
    switch (code) {
    case CompatibilityError::PACKAGE_BAD_MAGIC:
    case CompatibilityError::PACKAGE_VERSION_UNSUPPORTED:
    case CompatibilityError::PACKAGE_BOUNDS_INVALID:
    case CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED:
    case CompatibilityError::PACKAGE_CHECKSUM_MISMATCH:
    case CompatibilityError::PACKAGE_SOURCE_CHANGED:
        return CompatibilityPhase::Package;
    case CompatibilityError::RULE_VERSION_UNSUPPORTED:
    case CompatibilityError::RULE_LIMIT_EXCEEDED:
    case CompatibilityError::IMPORT_EXECUTABLE_CODE_REQUIRED:
    case CompatibilityError::IMPORT_SEMANTICS_MISSING:
    case CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS:
    case CompatibilityError::IMPORT_RULE_CONFLICT:
    case CompatibilityError::IMPORT_TENSOR_UNMAPPED:
    case CompatibilityError::IMPORT_TENSOR_DUPLICATE:
        return CompatibilityPhase::Import;
    case CompatibilityError::IR_VERSION_UNSUPPORTED:
    case CompatibilityError::IR_REFERENCE_INVALID:
    case CompatibilityError::IR_CONSTRAINT_FAILED:
    case CompatibilityError::IR_SHAPE_MISMATCH:
    case CompatibilityError::IR_LAYOUT_MISMATCH:
    case CompatibilityError::IR_QUANTIZATION_UNSUPPORTED:
    case CompatibilityError::MODALITY_UNSUPPORTED:
        return CompatibilityPhase::Semantic;
    case CompatibilityError::IR_STATE_INVALID:
    case CompatibilityError::STATE_ABI_MISMATCH:
    case CompatibilityError::RUNTIME_NUMERICAL_FAILURE:
        return CompatibilityPhase::StateRollback;
    case CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED:
        return CompatibilityPhase::Tokenizer;
    case CompatibilityError::CAPABILITY_MISSING:
    case CompatibilityError::KERNEL_UNAVAILABLE:
    case CompatibilityError::KERNEL_AMBIGUOUS:
    case CompatibilityError::PLAN_MEMORY_EXCEEDED:
    case CompatibilityError::FALLBACK_FORBIDDEN:
    case CompatibilityError::STREAMING_UNSUPPORTED:
    case CompatibilityError::PLAN_CONTEXT_EXCEEDED:
        return CompatibilityPhase::Plan;
    case CompatibilityError::SESSION_CONSTRUCTION_FAILED:
    case CompatibilityError::RUNTIME_INPUT_INVALID:
        return CompatibilityPhase::Session;
    case CompatibilityError::CACHE_MODE_UNQUALIFIED:
    case CompatibilityError::RULE_QUALIFICATION_REQUIRED:
        return CompatibilityPhase::Benchmark;
    }
    return CompatibilityPhase::Semantic;
}

std::string_view compatibility_message(CompatibilityError code) noexcept {
    switch (code) {
    case CompatibilityError::PACKAGE_BAD_MAGIC: return "package magic is invalid";
    case CompatibilityError::PACKAGE_VERSION_UNSUPPORTED: return "package version is unsupported";
    case CompatibilityError::PACKAGE_BOUNDS_INVALID: return "package bounds are invalid";
    case CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED: return "package graph is unsupported";
    case CompatibilityError::PACKAGE_CHECKSUM_MISMATCH: return "package checksum does not match";
    case CompatibilityError::PACKAGE_SOURCE_CHANGED: return "package source changed during validation";
    case CompatibilityError::RULE_VERSION_UNSUPPORTED: return "rule version is unsupported";
    case CompatibilityError::RULE_LIMIT_EXCEEDED: return "rule limit is exceeded";
    case CompatibilityError::IMPORT_EXECUTABLE_CODE_REQUIRED: return "import requires executable code";
    case CompatibilityError::IMPORT_SEMANTICS_MISSING: return "import semantics are missing";
    case CompatibilityError::IMPORT_SEMANTICS_AMBIGUOUS: return "import semantics are ambiguous";
    case CompatibilityError::IMPORT_RULE_CONFLICT: return "import rules conflict";
    case CompatibilityError::IMPORT_TENSOR_UNMAPPED: return "import tensor is unmapped";
    case CompatibilityError::IMPORT_TENSOR_DUPLICATE: return "import tensor is duplicated";
    case CompatibilityError::IR_VERSION_UNSUPPORTED: return "semantic version is unsupported";
    case CompatibilityError::IR_REFERENCE_INVALID: return "semantic reference is invalid";
    case CompatibilityError::IR_CONSTRAINT_FAILED: return "semantic constraint failed";
    case CompatibilityError::IR_SHAPE_MISMATCH: return "semantic shape does not match";
    case CompatibilityError::IR_LAYOUT_MISMATCH: return "physical layout does not match";
    case CompatibilityError::IR_QUANTIZATION_UNSUPPORTED: return "physical quantization is unsupported";
    case CompatibilityError::IR_STATE_INVALID: return "state contract is invalid";
    case CompatibilityError::TOKENIZER_RUNTIME_UNSUPPORTED: return "tokenizer runtime is unsupported";
    case CompatibilityError::MODALITY_UNSUPPORTED: return "semantic modality is unsupported";
    case CompatibilityError::CAPABILITY_MISSING: return "required capability is missing";
    case CompatibilityError::KERNEL_UNAVAILABLE: return "required kernel is unavailable";
    case CompatibilityError::KERNEL_AMBIGUOUS: return "kernel selection is ambiguous";
    case CompatibilityError::PLAN_MEMORY_EXCEEDED: return "plan memory limit is exceeded";
    case CompatibilityError::FALLBACK_FORBIDDEN: return "required fallback is forbidden";
    case CompatibilityError::SESSION_CONSTRUCTION_FAILED: return "session construction failed";
    case CompatibilityError::STATE_ABI_MISMATCH: return "state ABI does not match";
    case CompatibilityError::RUNTIME_INPUT_INVALID: return "session input is invalid";
    case CompatibilityError::STREAMING_UNSUPPORTED: return "streaming is unsupported";
    case CompatibilityError::PLAN_CONTEXT_EXCEEDED: return "plan context limit is exceeded";
    case CompatibilityError::RUNTIME_NUMERICAL_FAILURE: return "state transaction failed numerically";
    case CompatibilityError::CACHE_MODE_UNQUALIFIED: return "cache mode is unqualified";
    case CompatibilityError::RULE_QUALIFICATION_REQUIRED: return "rule qualification is required";
    }
    return "semantic compatibility failure";
}

CompatibilityReport compatibility_report(CompatibilityError code, std::string detail) {
    CompatibilityReport report;
    report.stage = compatibility_stage(code);
    report.code = code;
    report.phase = compatibility_phase(code);
    report.message = compatibility_message(code);
    report.detail = std::move(detail);
    return report;
}

CompatibilityReport package_report(CompatibilityError code, std::string detail) {
    return compatibility_report(code, std::move(detail));
}

} // namespace Laplace
