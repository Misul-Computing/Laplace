#include "token_graph_backend.h"

namespace Laplace {

namespace {
thread_local TokenGraphBackend* current_backend = nullptr;
}

TokenGraphBackend& active_token_graph_backend() {
    return current_backend ? *current_backend : metal_token_graph_backend();
}

ScopedTokenGraphBackend::ScopedTokenGraphBackend(TokenGraphBackend& backend)
    : previous_(current_backend) {
    current_backend = &backend;
}

ScopedTokenGraphBackend::~ScopedTokenGraphBackend() {
    current_backend = previous_;
}

} // namespace Laplace
