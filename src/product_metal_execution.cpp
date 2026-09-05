#include "product_metal_execution.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string>

#include "compat_rule.h"
#include "metal_library_source_catalog.h"

namespace Laplace {

namespace {

constexpr uint32_t kPrefillStateOnlyProgram = 1;
constexpr uint32_t kPrefillLogitsProgram = 2;
constexpr uint32_t kPrefillSampleProgram = 3;
constexpr uint32_t kDecodeLogitsProgram = 4;
constexpr uint32_t kDecodeSampleProgram = 5;
constexpr uint32_t kPrefillBatch2LogitsProgram = 6;
constexpr uint32_t kPrefillBatch2SampleProgram = 7;

void append_u16(std::vector<uint8_t>& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8u));
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void append_digest(std::vector<uint8_t>& bytes, const Sha256Digest& digest) {
    bytes.insert(bytes.end(), digest.bytes.begin(), digest.bytes.end());
}

Sha256Digest invocation_authority_digest(
    const ProductMetalInvocationAuthority& authority) {
    std::vector<uint8_t> bytes;
    static constexpr std::array<uint8_t, 8> domain = {
        'L', 'P', 'M', 'I', 'N', 'V', 1, 0};
    bytes.insert(bytes.end(), domain.begin(), domain.end());
    append_u16(bytes, authority.version);
    append_u32(bytes, authority.program_id);
    append_u32(bytes, authority.invocation_ordinal);
    append_u32(bytes, authority.group_ordinal);
    append_u16(bytes, static_cast<uint16_t>(authority.group_kind));
    append_u16(bytes, static_cast<uint16_t>(authority.group_shape));
    append_u16(bytes, static_cast<uint16_t>(authority.primitive));
    append_u32(bytes, authority.primitive_order);
    append_u32(bytes, authority.recipe_index);
    append_u32(bytes, authority.batch_rows);
    append_u32(bytes, authority.row_index);
    append_u32(bytes, authority.row_count);
    append_digest(bytes, authority.semantic_program_digest);
    append_digest(bytes, authority.bound_program_digest);
    Sha256Digest digest;
    if (bytes.size() <= static_cast<size_t>(std::numeric_limits<CC_LONG>::max()))
        CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()),
                  digest.bytes.data());
    return digest;
}

CompatibilityReport execution_error(CompatibilityError code,
                                    std::string detail = {}) {
    CompatibilityReport report = package_report(code);
    report.stage = CompatibilityStage::Session;
    report.detail = std::move(detail);
    return report;
}

bool request_valid(const RuntimePackage& package, const SessionRequest& request,
                   CompatibilityReport& report) {
    if (!package.product_authoritative() ||
        !package.manifest().has_physical_codec_authority()) {
        report = execution_error(
            CompatibilityError::PACKAGE_AUTHORITY_REQUIRED,
            "product Metal execution requires manifest-bound codec authority");
        return false;
    };
    if (request.max_context == 0 ||
        request.max_context > package.semantics().maximum_context) {
        report = execution_error(CompatibilityError::PLAN_CONTEXT_EXCEEDED);
        return false;
    }
    if (request.max_batch == 0 ||
        (!request.enable_prefill && !request.enable_decode) ||
        request.minimum_class != NumericalClass::ExactFp32 ||
        (request.objective != RuntimeObjective::Latency &&
         request.objective != RuntimeObjective::Throughput)) {
        report = execution_error(CompatibilityError::RUNTIME_INPUT_INVALID);
        return false;
    }
    if (request.max_batch > 2) {
        report = execution_error(
            CompatibilityError::KERNEL_UNAVAILABLE,
            "product Metal executor currently admits prefill batches of one or two rows");
        return false;
    }
    if (request.memory_limit == 0) {
        report = execution_error(CompatibilityError::PLAN_MEMORY_EXCEEDED);
        return false;
    }
    if (request.enable_streaming) {
        report = execution_error(CompatibilityError::STREAMING_UNSUPPORTED);
        return false;
    }
    if (request.enable_speculation) {
        report = execution_error(CompatibilityError::FALLBACK_FORBIDDEN);
        return false;
    }
    return true;
}

uint32_t program_id(ExecutionPhase phase, uint32_t batch_rows,
                    bool sampled) noexcept {
    if (phase == ExecutionPhase::Prefill) {
        if (batch_rows == 2)
            return sampled ? kPrefillBatch2SampleProgram
                           : kPrefillBatch2LogitsProgram;
        return sampled ? kPrefillSampleProgram : kPrefillLogitsProgram;
    }
    return sampled ? kDecodeSampleProgram : kDecodeLogitsProgram;
}

ProductMetalProgramKind program_kind(ExecutionPhase phase,
                                     uint32_t batch_rows,
                                     bool sampled) noexcept {
    if (phase == ExecutionPhase::Prefill) {
        if (batch_rows == 2)
            return sampled
                ? ProductMetalProgramKind::PrefillBatch2GreedySample
                : ProductMetalProgramKind::PrefillBatch2Logits;
        return sampled ? ProductMetalProgramKind::PrefillGreedySample
                       : ProductMetalProgramKind::PrefillLogits;
    }
    return sampled ? ProductMetalProgramKind::DecodeGreedySample
                   : ProductMetalProgramKind::DecodeLogits;
}

bool append_invocation(
    const StructuralMetalPrimitiveInvocation& invocation,
    uint32_t recipe_count, std::vector<uint32_t>& flattened) {
    if (invocation.recipe_index >= recipe_count ||
        flattened.size() == UINT32_MAX)
        return false;
    flattened.push_back(invocation.recipe_index);
    return true;
}

bool append_authorized_invocation(
    const StructuralMetalPrimitiveInvocation& invocation,
    const StructuralMetalBundleGroup& group,
    const SemanticDispatchProgram& semantic,
    const BoundDispatchProgram& bound_program,
    uint32_t program_id, uint32_t recipe_count,
    uint32_t row_index, uint32_t row_count,
    std::vector<uint32_t>& flattened,
    std::vector<ProductMetalInvocationAuthority>& authorities) {
    if (authorities.size() != flattened.size() ||
        bound_program.version() != kBoundDispatchRequirementsVersionV2 ||
        bound_program.program_digest() != semantic.program_digest ||
        bound_program.request() != semantic.request || row_count == 0 ||
        (row_index != UINT32_MAX && row_index >= semantic.request.batch_rows) ||
        (row_count == 2 && semantic.request.batch_rows != 2) ||
        !append_invocation(invocation, recipe_count, flattened))
        return false;
    ProductMetalInvocationAuthority authority;
    authority.program_id = program_id;
    authority.invocation_ordinal =
        static_cast<uint32_t>(flattened.size() - 1);
    authority.group_ordinal = group.ordinal();
    authority.group_kind = group.kind();
    authority.group_shape = group.shape();
    authority.primitive = invocation.primitive;
    authority.primitive_order = invocation.order;
    authority.recipe_index = invocation.recipe_index;
    authority.batch_rows = semantic.request.batch_rows;
    authority.row_index = row_index;
    authority.row_count = row_count;
    authority.semantic_program_digest = semantic.program_digest;
    authority.bound_program_digest = bound_program.bound_digest();
    authority.invocation_digest = invocation_authority_digest(authority);
    if (authority.bound_program_digest == Sha256Digest{} ||
        authority.invocation_digest == Sha256Digest{}) {
        flattened.pop_back();
        return false;
    }
    try {
        authorities.push_back(std::move(authority));
    } catch (...) {
        flattened.pop_back();
        throw;
    }
    return true;
}

bool append_two_row_group(
    const StructuralMetalBundleGroup& group,
    const SemanticDispatchProgram& semantic,
    const BoundDispatchProgram& bound_program,
    uint32_t program_id, uint32_t recipe_count,
    std::vector<uint32_t>& flattened,
    std::vector<ProductMetalInvocationAuthority>& authorities) {
    const auto primitives = group.primitives();
    if (group.shape() == StructuralMetalExecutionShape::GraphEmbedding) {
        return primitives.size() == 1 &&
               append_authorized_invocation(
                   primitives[0], group, semantic, bound_program, program_id,
                   recipe_count, 0, 1, flattened, authorities) &&
               append_authorized_invocation(
                   primitives[0], group, semantic, bound_program, program_id,
                   recipe_count, 1, 1, flattened, authorities);
    }
    if (group.kind() != StructuralMetalBundleGroupKind::Layer) {
        for (const auto& invocation : primitives)
            if (!append_authorized_invocation(
                    invocation, group, semantic, bound_program, program_id,
                    recipe_count,
                    semantic.output_binding
                        ? semantic.output_binding->selected_row
                        : UINT32_MAX,
                    1, flattened, authorities))
                return false;
        return true;
    }
    if (group.shape() != StructuralMetalExecutionShape::DenseAttention)
        return false;

    static constexpr std::array<StructuralMetalPrimitive, 17> expected = {
        StructuralMetalPrimitive::RmsNormF32,
        StructuralMetalPrimitive::PrefillF16Rows,
        StructuralMetalPrimitive::PrefillF16Rows,
        StructuralMetalPrimitive::PrefillF16Rows,
        StructuralMetalPrimitive::RopeHalfSplit,
        StructuralMetalPrimitive::RopeHalfSplit,
        StructuralMetalPrimitive::KvWrite,
        StructuralMetalPrimitive::KvWrite,
        StructuralMetalPrimitive::Attention,
        StructuralMetalPrimitive::PrefillF16Rows,
        StructuralMetalPrimitive::VecAdd,
        StructuralMetalPrimitive::RmsNormF32,
        StructuralMetalPrimitive::PrefillF16Rows,
        StructuralMetalPrimitive::PrefillF16Rows,
        StructuralMetalPrimitive::SwiGlu,
        StructuralMetalPrimitive::PrefillF16Rows,
        StructuralMetalPrimitive::VecAdd,
    };
    if (primitives.size() != expected.size()) return false;
    for (size_t index = 0; index < expected.size(); ++index)
        if (primitives[index].primitive != expected[index]) return false;

    // The composite dense executor launches row-local primitives once per row
    // and matrix projections once for the full two-row batch.
    struct PhysicalInvocation {
        uint8_t primitive_index;
        uint32_t row_index;
        uint32_t row_count;
    };
    static constexpr std::array<PhysicalInvocation, 27> physical_order = {{
        {0, 0, 1}, {0, 1, 1}, {1, UINT32_MAX, 2},
        {2, UINT32_MAX, 2}, {3, UINT32_MAX, 2},
        {4, 0, 1}, {5, 0, 1}, {6, 0, 1}, {7, 0, 1},
        {4, 1, 1}, {5, 1, 1}, {6, 1, 1}, {7, 1, 1},
        {8, 0, 1}, {8, 1, 1}, {9, UINT32_MAX, 2},
        {10, 0, 1}, {11, 0, 1}, {10, 1, 1}, {11, 1, 1},
        {12, UINT32_MAX, 2}, {13, UINT32_MAX, 2},
        {14, 0, 1}, {14, 1, 1}, {15, UINT32_MAX, 2},
        {16, 0, 1}, {16, 1, 1},
    }};
    for (const PhysicalInvocation& physical : physical_order) {
        if (!append_authorized_invocation(
                primitives[physical.primitive_index], group, semantic,
                bound_program, program_id, recipe_count,
                physical.row_index, physical.row_count,
                flattened, authorities))
            return false;
    }
    return true;
}

bool append_program_invocations(
    const StructuralMetalProgramBundle& program,
    const SemanticDispatchProgram& semantic,
    const BoundDispatchProgram& bound_program, uint32_t id,
    uint32_t recipe_count, std::vector<uint32_t>& flattened,
    std::vector<ProductMetalInvocationAuthority>& authorities,
    std::vector<MetalTokProgramRange>& ranges,
    bool stop_before_final) {
    if (program.program_digest() != semantic.program_digest ||
        program.batch_rows() != semantic.request.batch_rows ||
        (!stop_before_final &&
         (!semantic.output_binding ||
          semantic.output_binding->selected_row >= semantic.request.batch_rows)))
        return false;
    if (flattened.size() > UINT32_MAX) return false;
    const uint32_t first = static_cast<uint32_t>(flattened.size());
    bool saw_final = false;
    uint32_t expected_group = 0;
    for (const StructuralMetalBundleGroup& group : program.groups()) {
        if (group.ordinal() != expected_group++) return false;
        if (group.shape() == StructuralMetalExecutionShape::FinalOutput) {
            saw_final = true;
            if (stop_before_final) break;
        }
        if (program.batch_rows() == 2) {
            if (!append_two_row_group(
                    group, semantic, bound_program, id, recipe_count,
                    flattened, authorities))
                return false;
        } else {
            for (const StructuralMetalPrimitiveInvocation& invocation :
                 group.primitives())
                if (!append_authorized_invocation(
                        invocation, group, semantic, bound_program, id,
                        recipe_count, 0, 1, flattened, authorities))
                    return false;
        }
    }
    if (!saw_final || flattened.size() <= first ||
        flattened.size() - first > UINT32_MAX)
        return false;
    ranges.push_back({
        id, first, static_cast<uint32_t>(flattened.size() - first),
        semantic.request.batch_rows,
        stop_before_final ? UINT32_MAX
                          : semantic.output_binding->selected_row,
        semantic.program_digest.bytes});
    return true;
}

CompatibilityReport transaction_error(
    const MetalPipelineTransactionFailure& failure) {
    const CompatibilityError code =
        failure.code == MetalPipelineTransactionError::NoDevice
            ? CompatibilityError::CAPABILITY_MISSING
            : CompatibilityError::SESSION_CONSTRUCTION_FAILED;
    return execution_error(code, failure.detail);
}

} // namespace

const ProductMetalProgramBinding* ProductMetalExecutionContract::find(
    ProductMetalProgramKind kind) const noexcept {
    for (const ProductMetalProgramBinding& binding : program_bindings_)
        if (binding.kind == kind) return &binding;
    return nullptr;
}

const MetalTokProgramRange* ProductMetalExecutionContract::range(
    uint32_t id) const noexcept {
    for (const MetalTokProgramRange& range : program_ranges_)
        if (range.id == id) return &range;
    return nullptr;
}

uint32_t ProductMetalPreparedSession::program_id(
    ProductMetalProgramKind kind) const noexcept {
    const ProductMetalProgramBinding* binding =
        contract_ ? contract_->find(kind) : nullptr;
    return binding ? binding->id : UINT32_MAX;
}

ProductMetalExecutionContractResult compile_product_metal_execution_contract(
    const RuntimePackage& package, const SessionRequest& request) {
    try {
        CompatibilityReport report;
        if (!request_valid(package, request, report)) return report;

        CodecBindingPreflightResult preflight = preflight_codec_bindings(package);
        if (auto* error = std::get_if<CompatibilityReport>(&preflight)) return *error;
        ResolvedCodecBindings bindings =
            std::get<ResolvedCodecBindings>(std::move(preflight));

        std::vector<SemanticDispatchProgram> programs;
        std::vector<ProductMetalProgramBinding> program_bindings;
        const size_t prefill_program_count =
            request.enable_prefill ? (request.max_batch == 2 ? 4u : 2u) : 0u;
        programs.reserve(prefill_program_count +
                         (request.enable_decode ? 2u : 0u));
        program_bindings.reserve(programs.capacity() +
                                 (request.enable_prefill ? 1u : 0u));
        const auto append_program = [&](ExecutionPhase phase, uint32_t batch_rows,
                                        bool sampled) -> bool {
            SemanticDispatchRequest dispatch;
            dispatch.phase = phase;
            dispatch.batch_rows = batch_rows;
            dispatch.numerical_class = request.minimum_class;
            dispatch.include_speculative = false;
            dispatch.include_greedy_sampler = sampled;
            SemanticDispatchProgramResult built =
                build_semantic_dispatch_program(package.semantics(), dispatch);
            if (auto* error = std::get_if<CompatibilityReport>(&built)) {
                report = *error;
                return false;
            }
            const uint32_t index = static_cast<uint32_t>(programs.size());
            programs.push_back(
                std::get<SemanticDispatchProgram>(std::move(built)));
            program_bindings.push_back(
                {program_kind(phase, batch_rows, sampled),
                 program_id(phase, batch_rows, sampled), index});
            return true;
        };
        if (request.enable_prefill &&
            (!append_program(ExecutionPhase::Prefill, 1, false) ||
             !append_program(ExecutionPhase::Prefill, 1, true)))
            return report;
        if (request.enable_prefill && request.max_batch == 2 &&
            (!append_program(ExecutionPhase::Prefill, 2, false) ||
             !append_program(ExecutionPhase::Prefill, 2, true)))
            return report;
        if (request.enable_decode &&
            (!append_program(ExecutionPhase::Decode, 1, false) ||
             !append_program(ExecutionPhase::Decode, 1, true)))
            return report;

        BoundDispatchRequirementsResult bound_result = bind_dispatch_requirements(
            package, bindings, request, programs);
        if (auto* error = std::get_if<CompatibilityReport>(&bound_result))
            return *error;
        BoundDispatchRequirements bound =
            std::get<BoundDispatchRequirements>(std::move(bound_result));
        if (bound.version() != kBoundDispatchRequirementsVersionV2 ||
            bound.programs().size() != programs.size()) {
            return execution_error(
                CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                "bound dispatch authority does not cover every product program");
        }

        ProductMetalCodecCapabilityResult capability_result =
            compile_product_metal_codec_capabilities(
                package, bound, programs, package.semantics());
        if (auto* error = std::get_if<CompatibilityReport>(&capability_result))
            return *error;
        ProductMetalCodecCapabilityCompilation capabilities =
            std::get<ProductMetalCodecCapabilityCompilation>(
                std::move(capability_result));

        const MetalLibrarySourceCatalog& catalog =
            product_metal_library_source_catalog();
        if (!catalog.valid()) {
            return execution_error(
                CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                "product Metal library source catalog is invalid");
        }
        StructuralMetalCompilerResult structural_result =
            compile_structural_metal(
                bound, programs, package.semantics(),
                package.physical_codec_registry().codecs,
                {std::vector<StructuralMetalLibraryIdentity>(
                    catalog.compiler_identities().begin(),
                    catalog.compiler_identities().end())},
                &capabilities.registry());
        if (auto* error = std::get_if<CompatibilityReport>(&structural_result))
            return *error;
        StructuralMetalCompilation structural =
            std::get<StructuralMetalCompilation>(std::move(structural_result));
        if (structural.programs().size() != programs.size() ||
            structural.recipes().empty()) {
            return execution_error(
                CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                "structural Metal compilation did not cover every product program");
        }

        std::vector<uint32_t> flattened;
        std::vector<ProductMetalInvocationAuthority> authorities;
        std::vector<MetalTokProgramRange> ranges;
        ranges.reserve(programs.size() + (request.enable_prefill ? 1u : 0u));
        for (size_t index = 0; index != structural.programs().size(); ++index) {
            const ProductMetalProgramBinding& binding = program_bindings[index];
            if (!append_program_invocations(
                    structural.programs()[index], programs[index],
                    bound.programs()[index], binding.id,
                    static_cast<uint32_t>(structural.recipes().size()),
                    flattened, authorities, ranges, false)) {
                return execution_error(
                    CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                    "structural Metal program has an invalid invocation stream");
            }
        }
        if (request.enable_prefill) {
            const auto prefill = std::find_if(
                program_bindings.begin(), program_bindings.end(),
                [](const ProductMetalProgramBinding& binding) {
                    return binding.kind == ProductMetalProgramKind::PrefillLogits;
                });
            if (prefill == program_bindings.end() ||
                prefill->semantic_program_index >= structural.programs().size() ||
                !append_program_invocations(
                    structural.programs()[prefill->semantic_program_index],
                    programs[prefill->semantic_program_index],
                    bound.programs()[prefill->semantic_program_index],
                    kPrefillStateOnlyProgram,
                    static_cast<uint32_t>(structural.recipes().size()),
                    flattened, authorities, ranges, true)) {
                return execution_error(
                    CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                    "product prefill state-only recipe range is invalid");
            }
            program_bindings.push_back(
                {ProductMetalProgramKind::PrefillStateOnly,
                 kPrefillStateOnlyProgram, prefill->semantic_program_index});
        }

        return ProductMetalExecutionContract(
            std::move(programs), std::move(bindings), std::move(bound),
            std::move(capabilities), std::move(structural),
            std::move(flattened), std::move(authorities), std::move(ranges),
            std::move(program_bindings));
    } catch (const std::bad_alloc&) {
        return execution_error(CompatibilityError::PLAN_MEMORY_EXCEEDED);
    }
}

ProductMetalPreparedSessionResult prepare_product_metal_session(
    const RuntimePackage& package, const SessionRequest& request) {
    ProductMetalExecutionContractResult compiled =
        compile_product_metal_execution_contract(package, request);
    if (auto* error = std::get_if<CompatibilityReport>(&compiled)) return *error;
    ProductMetalExecutionContract contract =
        std::get<ProductMetalExecutionContract>(std::move(compiled));
    const MetalLibrarySourceCatalog& catalog =
        product_metal_library_source_catalog();
    std::vector<MetalPipelineLibrarySource> used_sources;
    used_sources.reserve(catalog.transaction_sources().size());
    for (const MetalPipelineLibrarySource& source :
         catalog.transaction_sources()) {
        if (std::any_of(contract.recipes().begin(), contract.recipes().end(),
                        [&](const MetalPipelineRecipe& recipe) {
                            return recipe.library_source_digest ==
                                   source.source_digest;
                        })) {
            used_sources.push_back(source);
        }
    }
    if (used_sources.empty() ||
        std::any_of(contract.recipes().begin(), contract.recipes().end(),
                    [&](const MetalPipelineRecipe& recipe) {
                        return std::none_of(
                            used_sources.begin(), used_sources.end(),
                            [&](const MetalPipelineLibrarySource& source) {
                                return source.source_digest ==
                                       recipe.library_source_digest;
                            });
                    })) {
        return execution_error(
            CompatibilityError::SESSION_CONSTRUCTION_FAILED,
            "product Metal recipe references no immutable library source");
    }
    MetalPipelineTransactionResult transaction_result =
        build_metal_pipeline_transaction(contract.recipes(),
                                         used_sources);
    if (auto* error =
            std::get_if<MetalPipelineTransactionFailure>(&transaction_result))
        return transaction_error(*error);
    MetalPipelineLease lease =
        std::get<MetalPipelineTransaction>(std::move(transaction_result))
            .take_lease();
    std::vector<MetalPipelineRecipe> lease_ordered_recipes;
    std::vector<uint32_t> recipe_to_slot;
    std::vector<uint32_t> leased_invocations;
    std::vector<MetalTokInvocationAuthority> leased_authorities;
    try {
        lease_ordered_recipes.resize(lease.slot_count());
        recipe_to_slot.resize(contract.recipes().size(), UINT32_MAX);
        std::vector<bool> occupied(lease.slot_count(), false);
        for (uint32_t recipe_index = 0;
             recipe_index != contract.recipes().size(); ++recipe_index) {
            uint32_t slot = UINT32_MAX;
            const void* pipeline = nullptr;
            if (!lease.resolve_recipe(contract.recipes()[recipe_index], &slot,
                                      &pipeline) ||
                slot >= lease_ordered_recipes.size() || !pipeline ||
                occupied[slot]) {
                return execution_error(
                    CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                    "product Metal recipe cannot be mapped to one leased pipeline slot");
            }
            occupied[slot] = true;
            recipe_to_slot[recipe_index] = slot;
            lease_ordered_recipes[slot] = contract.recipes()[recipe_index];
        }
        if (std::find(occupied.begin(), occupied.end(), false) != occupied.end()) {
            return execution_error(
                CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                "product Metal lease contains an unbound pipeline slot");
        }
        leased_invocations.reserve(
            contract.flattened_invocation_recipe_indices().size());
        for (uint32_t recipe_index :
             contract.flattened_invocation_recipe_indices()) {
            if (recipe_index >= recipe_to_slot.size() ||
                recipe_to_slot[recipe_index] == UINT32_MAX) {
                return execution_error(
                    CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                    "product Metal invocation references an unleased recipe");
            }
            leased_invocations.push_back(recipe_to_slot[recipe_index]);
        }
        if (contract.invocation_authorities().size() !=
            contract.flattened_invocation_recipe_indices().size()) {
            return execution_error(
                CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                "product Metal invocation authority is incomplete");
        }
        leased_authorities.reserve(contract.invocation_authorities().size());
        for (size_t index = 0;
             index != contract.invocation_authorities().size(); ++index) {
            const ProductMetalInvocationAuthority& source =
                contract.invocation_authorities()[index];
            if (source.invocation_ordinal != index ||
                source.recipe_index !=
                    contract.flattened_invocation_recipe_indices()[index] ||
                source.recipe_index >= recipe_to_slot.size() ||
                recipe_to_slot[source.recipe_index] != leased_invocations[index]) {
                return execution_error(
                    CompatibilityError::SESSION_CONSTRUCTION_FAILED,
                    "product Metal invocation authority cannot be mapped to the lease");
            }
            MetalTokInvocationAuthority authority;
            authority.version = source.version;
            authority.group_kind =
                static_cast<uint16_t>(source.group_kind);
            authority.group_shape =
                static_cast<uint16_t>(source.group_shape);
            authority.primitive = static_cast<uint16_t>(source.primitive);
            authority.program_id = source.program_id;
            authority.invocation_ordinal = source.invocation_ordinal;
            authority.group_ordinal = source.group_ordinal;
            authority.primitive_order = source.primitive_order;
            authority.recipe_index = source.recipe_index;
            authority.pipeline_slot = leased_invocations[index];
            authority.batch_rows = source.batch_rows;
            authority.row_index = source.row_index;
            authority.row_count = source.row_count;
            authority.semantic_program_digest =
                source.semantic_program_digest.bytes;
            authority.bound_program_digest =
                source.bound_program_digest.bytes;
            authority.invocation_digest = source.invocation_digest.bytes;
            leased_authorities.push_back(std::move(authority));
        }
    } catch (const std::bad_alloc&) {
        return execution_error(CompatibilityError::PLAN_MEMORY_EXCEEDED);
    }
    std::shared_ptr<MetalTokSession> session =
        metal_tok_session_create_from_pipeline_lease(
            std::move(lease), lease_ordered_recipes, leased_invocations,
            contract.program_ranges(), leased_authorities);
    if (!session) {
        return execution_error(
            CompatibilityError::SESSION_CONSTRUCTION_FAILED,
            "product Metal lease-backed session construction failed");
    }
    metal_tok_session_require_registered_weights(*session, true);
    try {
        auto retained_contract =
            std::make_shared<const ProductMetalExecutionContract>(
                std::move(contract));
        return ProductMetalPreparedSession(std::move(session),
                                           std::move(retained_contract));
    } catch (const std::bad_alloc&) {
        return execution_error(CompatibilityError::PLAN_MEMORY_EXCEEDED);
    }
}

} // namespace Laplace
