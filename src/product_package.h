#pragma once

#include <memory>
#include <string_view>
#include <utility>
#include <variant>

#include "compat_rule.h"
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

private:
    ProductPackage(std::shared_ptr<const RuntimePackage> runtime,
                   TokenProgram token_program)
        : runtime_(std::move(runtime)), token_program_(std::move(token_program)) {}

    static std::variant<ProductPackage, CompatibilityReport>
    finish_closed_v1(GgufProductCompilation compilation);
    static std::variant<ProductPackage, CompatibilityReport>
    finish_product_package(ArtifactIndex physical, PackageView carrier);

    std::shared_ptr<const RuntimePackage> runtime_;
    TokenProgram token_program_;

    friend std::variant<ProductPackage, CompatibilityReport>
    load_product_package(std::string_view package_path);
};

using ProductPackageLoadResult = std::variant<ProductPackage, CompatibilityReport>;

// Loads one immutable GGUF weight artifact, its carried semantic manifest,
// and its compiled tokenizer program. Raw package evidence cannot grant
// execution authority.
ProductPackageLoadResult load_product_package(std::string_view package_path);

} // namespace Laplace
