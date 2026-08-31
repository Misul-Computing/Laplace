#pragma once

#include "gguf_product_compiler.h"

namespace Laplace {

// Source-schema-neutral GGUF admission.  The adapter supplies typed facts;
// this compiler derives the semantic graph and immutable package closure from
// those facts without a family, name, or artifact catalog branch.
GgufProductCompilationResult compile_normalized_gguf_product(
    const PackageView& package);

} // namespace Laplace
