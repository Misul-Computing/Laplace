// laplace_arch.cpp - model architecture registry
#include "arch.h"

#include <memory>
#include <string>

#include "arch_gemma4.h"
#include "arch_llama.h"
#include "arch_phi3.h"
#include "arch_qwen3next.h"

namespace Laplace {

std::unique_ptr<ModelArch> create_arch(const std::string& name) {
    if (name == "gemma4") return std::make_unique<Gemma4Arch>();
    if (name == "qwen3next" || name == "qwen35") {
        return std::make_unique<Qwen3NextArch>();
    }
    if (name == "llama" || name == "qwen2" || name == "qwen3") {
        return std::make_unique<LlamaArch>();
    }
    if (name == "phi3") return std::make_unique<Phi3Arch>();
    return nullptr;
}

} // namespace Laplace
