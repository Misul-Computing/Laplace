// generation_loop.h - causal generation accounting shared by CLI and tests.
#pragma once

#include <cstdint>
#include <string>

namespace Laplace {

struct GenerationMetrics {
    uint64_t prefill_tokens = 0;
    uint64_t decode_output_tokens = 0;
    uint64_t decode_forward_calls = 0;
    uint64_t emitted_tokens = 0;
    uint64_t draft_tokens = 0;
    uint64_t accepted_tokens = 0;
    double raw_decode_forwards_per_second = 0.0;
    double generated_output_tokens_per_second = 0.0;
    double accepted_tokens_per_second = 0.0;
};

void record_prefill(GenerationMetrics* metrics, uint64_t tokens);
void record_decode_forward(GenerationMetrics* metrics);
void record_decode_output(GenerationMetrics* metrics, uint64_t tokens);
void record_emitted(GenerationMetrics* metrics, uint64_t tokens);
void record_draft(GenerationMetrics* metrics, uint64_t tokens);
void record_accepted(GenerationMetrics* metrics, uint64_t tokens);
void finalize_decode_metrics(GenerationMetrics* metrics,
                             double raw_forward_elapsed_ms,
                             double end_to_end_elapsed_ms);
std::string render_decode_benchmark(const GenerationMetrics& metrics,
                                    double elapsed_ms);

} // namespace Laplace
