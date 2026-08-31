#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>
#include <utility>
#include <vector>

#include "bound_dispatch_requirements.h"
#include "codec_binding.h"
#include "metal_codec_capability.h"
#include "normalized_codec_program.h"
#include "semantic_dispatch_program.h"

namespace Laplace {

// One occurrence in a semantic dispatch program.  The certificate identity is
// retained as provenance only; it is deliberately absent from the capability
// lookup key.
struct ProductMetalCodecCapabilityRecord {
    uint32_t program_index = 0;
    uint32_t step_ordinal = 0;
    uint32_t tensor_slot = 0;
    uint32_t codec_occurrence_index = 0;
    uint32_t operator_id = 0;
    uint32_t tensor_id = 0;
    MetalCodecRequirement requirement;
    MetalCodecLowering lowering;
    MetalCodecCapabilityDigest capability_digest{};
    NormalizedCodecProvenance provenance;
    CodecProgramIdentity program_identity;
    PhysicalCodecIdentity physical_identity;
};

class ProductMetalCodecCapabilityCompilation {
public:
    // Construction is intentionally value-based; callers cannot mutate the
    // registry or records after compilation.
    ProductMetalCodecCapabilityCompilation(
        MetalCodecCapabilityRegistry registry,
        std::vector<ProductMetalCodecCapabilityRecord> records)
        : registry_(std::move(registry)), records_(std::move(records)) {}

    const MetalCodecCapabilityRegistry& registry() const noexcept {
        return registry_;
    }
    std::span<const ProductMetalCodecCapabilityRecord> records() const noexcept {
        return records_;
    }

private:
    MetalCodecCapabilityRegistry registry_;
    std::vector<ProductMetalCodecCapabilityRecord> records_;
};

using ProductMetalCodecCapabilityResult =
    std::variant<ProductMetalCodecCapabilityCompilation, CompatibilityReport>;

// Compiles package-authoritative certificate bindings into application-owned
// Metal recipes.  Only normalized decoder semantics and the complete physical
// tuple affect the recipe.  This function performs no device query or device
// allocation.
ProductMetalCodecCapabilityResult compile_product_metal_codec_capabilities(
    const RuntimePackage& package, const ResolvedCodecBindings& bindings,
    std::span<const SemanticDispatchProgram> programs,
    const SemanticModel& semantic_model);

// Equivalent entry point when occurrence identity has already been bound to a
// dispatch program.  The package still supplies the authoritative certificate
// bytes and physical identities.
ProductMetalCodecCapabilityResult compile_product_metal_codec_capabilities(
    const RuntimePackage& package, const BoundDispatchRequirements& bound,
    std::span<const SemanticDispatchProgram> programs,
    const SemanticModel& semantic_model);

} // namespace Laplace
