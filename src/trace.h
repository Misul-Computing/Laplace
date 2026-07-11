// trace.h - activation tracing for layer-diff debugging against a reference
// runtime. Enabled with LAPLACE_TRACE=1; zero overhead otherwise.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace Laplace {

inline bool trace_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("LAPLACE_TRACE");
        return v && v[0] == '1';
    }();
    return on;
}

inline void trace(const char* name, int layer, const float* x, int n) {
    if (!trace_enabled()) return;
    double sum = 0.0, sumsq = 0.0;
    for (int i = 0; i < n; i++) { sum += x[i]; sumsq += static_cast<double>(x[i]) * x[i]; }
    fprintf(stderr, "TRACE %-24s il=%-2d n=%-6d [%9.5f %9.5f %9.5f %9.5f] sum=%11.5f l2=%11.5f\n",
            name, layer, n,
            n > 0 ? x[0] : 0.0f, n > 1 ? x[1] : 0.0f,
            n > 2 ? x[2] : 0.0f, n > 3 ? x[3] : 0.0f,
            sum, std::sqrt(sumsq));
}

} // namespace Laplace
