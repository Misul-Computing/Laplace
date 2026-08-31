#include "generation_loop.h"

#include <cmath>
#include <cstdio>
#include <limits>

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

void finalize_decode_metrics(GenerationMetrics* metrics,
                             double raw_forward_elapsed_ms,
                             double end_to_end_elapsed_ms) {
    const auto rate = [](uint64_t count, double elapsed_ms) {
        if (count == 0) return 0.0;
        if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0)
            return std::numeric_limits<double>::quiet_NaN();
        return 1000.0 * static_cast<double>(count) / elapsed_ms;
    };
    metrics->raw_decode_forwards_per_second =
        rate(metrics->decode_forward_calls, raw_forward_elapsed_ms);
    metrics->generated_output_tokens_per_second =
        rate(metrics->decode_output_tokens, end_to_end_elapsed_ms);
    metrics->accepted_tokens_per_second =
        rate(metrics->accepted_tokens, end_to_end_elapsed_ms);
}

std::string render_decode_benchmark(const GenerationMetrics& metrics,
                                    double elapsed_ms) {
    (void)elapsed_ms;
    const auto rate = [](uint64_t count, double value, char* out, size_t size) {
        if (count == 0 || !std::isfinite(value)) std::snprintf(out, size, "unavailable");
        else std::snprintf(out, size, "%.1f", value);
    };
    char raw[32], generated[32], accepted[32];
    rate(metrics.decode_forward_calls, metrics.raw_decode_forwards_per_second, raw, sizeof(raw));
    rate(metrics.decode_output_tokens, metrics.generated_output_tokens_per_second, generated, sizeof(generated));
    rate(metrics.accepted_tokens, metrics.accepted_tokens_per_second, accepted, sizeof(accepted));
    char line[256];
    std::snprintf(line, sizeof(line),
                  "[bench] decode: raw_decode_forward_s=%s "
                  "generated_output_tok_s=%s accepted_tok_s=%s\n",
                  raw, generated, accepted);
    return line;
}

} // namespace Laplace
