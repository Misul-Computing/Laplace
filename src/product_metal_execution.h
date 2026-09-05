#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <variant>
#include <vector>

#include "bound_dispatch_requirements.h"
#include "execution_plan.h"
#include "matmul.h"
#include "product_metal_codec_capabilities.h"
#include "structural_metal_compiler.h"

namespace Laplace {

class RuntimePackage;

enum class ProductMetalProgramKind : uint8_t {
    PrefillStateOnly = 1,
    PrefillLogits = 2,
    PrefillGreedySample = 3,
    DecodeLogits = 4,
    DecodeGreedySample = 5,
    PrefillBatch2Logits = 6,
    PrefillBatch2GreedySample = 7,
};

struct ProductMetalProgramBinding {
    ProductMetalProgramKind kind = ProductMetalProgramKind::DecodeLogits;
    uint32_t id = UINT32_MAX;
    uint32_t semantic_program_index = UINT32_MAX;

    friend bool operator==(const ProductMetalProgramBinding&,
                           const ProductMetalProgramBinding&) = default;
};

// Immutable host-side identity for one physical Metal invocation. The bound
// program digest transitively authenticates exact tensor, value, state, span,
// and codec facts; the remaining fields authenticate the structural expansion
// and row geometry. No pointer, source name, or model identity is stored here.
struct ProductMetalInvocationAuthority {
    uint16_t version = 1;
    uint32_t program_id = UINT32_MAX;
    uint32_t invocation_ordinal = UINT32_MAX;
    uint32_t group_ordinal = UINT32_MAX;
    StructuralMetalBundleGroupKind group_kind =
        StructuralMetalBundleGroupKind::Graph;
    StructuralMetalExecutionShape group_shape =
        StructuralMetalExecutionShape::GraphEmbedding;
    StructuralMetalPrimitive primitive = StructuralMetalPrimitive::VecAdd;
    uint32_t primitive_order = UINT32_MAX;
    uint32_t recipe_index = UINT32_MAX;
    uint32_t batch_rows = 1;
    uint32_t row_index = UINT32_MAX;
    uint32_t row_count = 1;
    Sha256Digest semantic_program_digest{};
    Sha256Digest bound_program_digest{};
    Sha256Digest invocation_digest{};

    friend bool operator==(const ProductMetalInvocationAuthority&,
                           const ProductMetalInvocationAuthority&) = default;
};

// Immutable host-side result of compiling one authoritative package and one
// session request. It contains no device object and performs no global lookup.
class ProductMetalExecutionContract {
public:
    std::span<const SemanticDispatchProgram> semantic_programs() const noexcept {
        return semantic_programs_;
    }
    std::span<const StructuralMetalProgramBundle> structural_programs() const noexcept {
        return structural_.programs();
    }
    std::span<const MetalPipelineRecipe> recipes() const noexcept {
        return structural_.recipes();
    }
    std::span<const uint32_t> flattened_invocation_recipe_indices() const noexcept {
        return flattened_invocations_;
    }
    std::span<const ProductMetalInvocationAuthority>
    invocation_authorities() const noexcept {
        return invocation_authorities_;
    }
    std::span<const MetalTokProgramRange> program_ranges() const noexcept {
        return program_ranges_;
    }
    std::span<const ProductMetalProgramBinding> program_bindings() const noexcept {
        return program_bindings_;
    }
    const ProductMetalProgramBinding* find(ProductMetalProgramKind kind) const noexcept;
    const MetalTokProgramRange* range(uint32_t id) const noexcept;

private:
    ProductMetalExecutionContract(
        std::vector<SemanticDispatchProgram> semantic_programs,
        ResolvedCodecBindings codec_bindings,
        BoundDispatchRequirements bound_requirements,
        ProductMetalCodecCapabilityCompilation codec_capabilities,
        StructuralMetalCompilation structural,
        std::vector<uint32_t> flattened_invocations,
        std::vector<ProductMetalInvocationAuthority> invocation_authorities,
        std::vector<MetalTokProgramRange> program_ranges,
        std::vector<ProductMetalProgramBinding> program_bindings)
        : semantic_programs_(std::move(semantic_programs)),
          codec_bindings_(std::move(codec_bindings)),
          bound_requirements_(std::move(bound_requirements)),
          codec_capabilities_(std::move(codec_capabilities)),
          structural_(std::move(structural)),
          flattened_invocations_(std::move(flattened_invocations)),
          invocation_authorities_(std::move(invocation_authorities)),
          program_ranges_(std::move(program_ranges)),
          program_bindings_(std::move(program_bindings)) {}

    std::vector<SemanticDispatchProgram> semantic_programs_;
    ResolvedCodecBindings codec_bindings_;
    BoundDispatchRequirements bound_requirements_;
    ProductMetalCodecCapabilityCompilation codec_capabilities_;
    StructuralMetalCompilation structural_;
    std::vector<uint32_t> flattened_invocations_;
    std::vector<ProductMetalInvocationAuthority> invocation_authorities_;
    std::vector<MetalTokProgramRange> program_ranges_;
    std::vector<ProductMetalProgramBinding> program_bindings_;

    friend std::variant<ProductMetalExecutionContract, CompatibilityReport>
    compile_product_metal_execution_contract(const RuntimePackage&,
                                             const SessionRequest&);
};

using ProductMetalExecutionContractResult =
    std::variant<ProductMetalExecutionContract, CompatibilityReport>;

ProductMetalExecutionContractResult compile_product_metal_execution_contract(
    const RuntimePackage& package, const SessionRequest& request);

class ProductMetalPreparedSession {
public:
    bool valid() const noexcept { return session_ != nullptr && contract_ != nullptr; }
    uint32_t program_id(ProductMetalProgramKind kind) const noexcept;
    const ProductMetalExecutionContract* contract() const noexcept {
        return contract_.get();
    }
    std::shared_ptr<MetalTokSession> take_session() noexcept {
        return std::move(session_);
    }
    std::shared_ptr<const ProductMetalExecutionContract> take_contract() noexcept {
        return std::move(contract_);
    }

private:
    ProductMetalPreparedSession(
        std::shared_ptr<MetalTokSession> session,
        std::shared_ptr<const ProductMetalExecutionContract> contract)
        : session_(std::move(session)), contract_(std::move(contract)) {}

    std::shared_ptr<MetalTokSession> session_;
    std::shared_ptr<const ProductMetalExecutionContract> contract_;

    friend std::variant<ProductMetalPreparedSession, CompatibilityReport>
    prepare_product_metal_session(const RuntimePackage&, const SessionRequest&);
};

using ProductMetalPreparedSessionResult =
    std::variant<ProductMetalPreparedSession, CompatibilityReport>;

// Device construction is atomic: all recipes compile on one device before the
// queue and lease-backed token session are published.
ProductMetalPreparedSessionResult prepare_product_metal_session(
    const RuntimePackage& package, const SessionRequest& request);

} // namespace Laplace
