#include "generation_loop.h"

#include <algorithm>
#include <cstdio>

namespace Laplace {

void record_prefill(GenerationMetrics* metrics, uint64_t tokens) {
    metrics->prefill_tokens += tokens;
}

void record_decode_forward(GenerationMetrics* metrics) {
    ++metrics->decode_forward_calls;
}

void record_decode_output(GenerationMetrics* metrics, uint64_t tokens) {
    metrics->decode_output_tokens += tokens;
}

void record_emitted(GenerationMetrics* metrics, uint64_t tokens) {
    metrics->emitted_tokens += tokens;
}

void record_draft(GenerationMetrics* metrics, uint64_t tokens) {
    metrics->draft_tokens += tokens;
}

void record_accepted(GenerationMetrics* metrics, uint64_t tokens) {
    metrics->accepted_tokens += tokens;
}

void finalize_decode_metrics(GenerationMetrics* metrics, double elapsed_ms) {
    metrics->decode_tokens_per_second = metrics->decode_output_tokens == 0
        ? 0.0
        : 1000.0 * metrics->decode_output_tokens /
              std::max(1.0, elapsed_ms);
}

std::string render_decode_benchmark(const GenerationMetrics& metrics,
                                    double elapsed_ms) {
    if (metrics.decode_output_tokens == 0) {
        return "[bench] decode: unavailable (decode_tok_s=unavailable)\n";
    }
    char line[160];
    std::snprintf(line, sizeof(line),
                  "[bench] decode:  %llu tokens in %.1f ms (%.1f tok/s, "
                  "decode_tok_s=%.1f)\n",
                  static_cast<unsigned long long>(metrics.decode_output_tokens),
                  elapsed_ms, metrics.decode_tokens_per_second,
                  metrics.decode_tokens_per_second);
    return line;
}

} // namespace Laplace
