// trace.h - activation tracing for layer-diff debugging against a reference
// runtime. Enabled with LAPLACE_TRACE=1; zero overhead otherwise.
// LAPLACE_TRACE_DUMP=<dir> additionally writes raw activations as .f32 files.
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
    static const bool on = trace_enabled();
    static const char* dump_dir = std::getenv("LAPLACE_TRACE_DUMP");
    if (!on && !dump_dir) return;
    if (dump_dir) {
        static int seq = 0;
        char path[512];
        snprintf(path, sizeof(path), "%s/%04d_%s_il%d.f32",
                 dump_dir, seq++, name, layer);
        if (FILE* f = fopen(path, "wb")) {
            fwrite(x, sizeof(float), n, f);
            fclose(f);
        }
    }
    if (!on) return;
    double sum = 0.0, sumsq = 0.0;
    for (int i = 0; i < n; i++) { sum += x[i]; sumsq += static_cast<double>(x[i]) * x[i]; }
    fprintf(stderr, "TRACE %-24s il=%-2d n=%-6d [%9.5f %9.5f %9.5f %9.5f] sum=%11.5f l2=%11.5f\n",
            name, layer, n,
            n > 0 ? x[0] : 0.0f, n > 1 ? x[1] : 0.0f,
            n > 2 ? x[2] : 0.0f, n > 3 ? x[3] : 0.0f,
            sum, std::sqrt(sumsq));
}

inline void trace_dump_file(const char* name, const void* p, size_t bytes) {
    const char* dump_dir = std::getenv("LAPLACE_TRACE_DUMP");
    if (!dump_dir) return;
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dump_dir, name);
    if (FILE* f = fopen(path, "wb")) { fwrite(p, 1, bytes, f); fclose(f); }
}

} // namespace Laplace
