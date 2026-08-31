#pragma once

#include <memory>
#include <string_view>
#include <utility>
#include <variant>

#include "compat_rule.h"
#include "physical_program_package.h"
#include "token_program.h"

namespace Laplace {

struct GgufProductCompilation;

class ProductPackage {
public:
    ProductPackage(ProductPackage&&) noexcept = default;
    ProductPackage& operator=(ProductPackage&&) noexcept = default;
    ProductPackage(const ProductPackage&) = delete;
    ProductPackage& operator=(const ProductPackage&) = delete;

    std::shared_ptr<const RuntimePackage> runtime_package() const noexcept {
        return runtime_;
    }
    const TokenProgram& token_program() const noexcept { return token_program_; }
    const std::shared_ptr<const VerifiedPhysicalProgramPackage>&
    physical_program_package() const noexcept { return physical_package_; }

private:
    ProductPackage(std::shared_ptr<const RuntimePackage> runtime,
                   TokenProgram token_program,
                   std::shared_ptr<const VerifiedPhysicalProgramPackage> physical_package = {})
        : runtime_(std::move(runtime)), token_program_(std::move(token_program)),
          physical_package_(std::move(physical_package)) {}

    static std::variant<ProductPackage, CompatibilityReport>
    finish_closed_v1(GgufProductCompilation compilation);
    static std::variant<ProductPackage, CompatibilityReport>
    finish_product_package(ArtifactIndex physical, PackageView carrier,
                            const PackageView* physical_package_carrier = nullptr);

    std::shared_ptr<const RuntimePackage> runtime_;
    TokenProgram token_program_;
    std::shared_ptr<const VerifiedPhysicalProgramPackage> physical_package_;

    friend std::variant<ProductPackage, CompatibilityReport>
    load_product_package(std::string_view package_path);
};

using ProductPackageLoadResult = std::variant<ProductPackage, CompatibilityReport>;

// Loads one immutable primary artifact, its carried semantic manifest, its
// compiled tokenizer program, and an optional adjacent `.lappkg` physical
// program carrier. Raw package evidence cannot grant execution authority.
ProductPackageLoadResult load_product_package(std::string_view package_path);

} // namespace Laplace
