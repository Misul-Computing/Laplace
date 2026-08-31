// token_graph_backend.h - injectable boundary for the decode token graph.
#pragma once

#include "matmul.h"

namespace Laplace {

class TokenGraphBackend {
public:
    virtual ~TokenGraphBackend() = default;

    virtual bool available() const = 0;
    virtual bool begin(int H, int inter, int exp_inter, int n_used,
                       int n_experts, int Hq, int Hk, int Dh, int max_seq,
                       int n_layers, int pos) = 0;
    virtual bool active() const = 0;
    virtual void upload_x(const float* x, int H) = 0;
    virtual bool layer(const MetalTokLayer& layer) = 0;
    virtual bool end(float* x, int H) = 0;
    // Abandon an incomplete token transaction. Implementations must make its
    // provisional residual and KV writes unavailable to the next token.
    virtual void abort() = 0;
};

// The production backend remains the existing Metal token graph. Tests may
// replace it per Model instance without creating a Metal device.
TokenGraphBackend& metal_token_graph_backend();
TokenGraphBackend& active_token_graph_backend();

class ScopedTokenGraphBackend {
public:
    explicit ScopedTokenGraphBackend(TokenGraphBackend& backend);
    ~ScopedTokenGraphBackend();

private:
    TokenGraphBackend* previous_ = nullptr;
};

} // namespace Laplace
