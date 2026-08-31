#include <limits>
#include <string>

#include "generation_loop.h"
#include "test_util.h"

using namespace Laplace;

namespace {

struct DeterministicLoop {
    GenerationMetrics run_plain() {
        GenerationMetrics metrics;
        record_prefill(&metrics, 3);
        record_emitted(&metrics, 1);  // first token came from prefill logits
        for (int step = 0; step < 2; ++step) {
            (void)step;
            record_decode_forward(&metrics);
            record_decode_output(&metrics, 1);
            record_emitted(&metrics, 1);
        }
        finalize_decode_metrics(&metrics, 500.0, 1000.0);
        return metrics;
    }

    GenerationMetrics run_speculative() {
        GenerationMetrics metrics;
        record_prefill(&metrics, 3);
        record_emitted(&metrics, 1);
        record_decode_forward(&metrics);  // one verify batch
        record_draft(&metrics, 2);
        record_accepted(&metrics, 2);
        record_decode_output(&metrics, 3);
        record_emitted(&metrics, 3);
        finalize_decode_metrics(&metrics, 1000.0, 2000.0);
        return metrics;
    }
};

void test_plain_metrics_and_rendering() {
    GenerationMetrics metrics = DeterministicLoop{}.run_plain();
    CHECK(metrics.prefill_tokens == 3);
    CHECK(metrics.decode_output_tokens == 2);
    CHECK(metrics.decode_forward_calls == 2);
    CHECK(metrics.emitted_tokens == 3);
    CHECK(metrics.raw_decode_forwards_per_second == 4.0);
    CHECK(metrics.generated_output_tokens_per_second == 2.0);
    CHECK(metrics.accepted_tokens_per_second == 0.0);
    const std::string bench = render_decode_benchmark(metrics, 500.0);
    CHECK(bench.find("raw_decode_forward_s=4.0") != std::string::npos);
    CHECK(bench.find("generated_output_tok_s=2.0") != std::string::npos);
    CHECK(bench.find("accepted_tok_s=unavailable") != std::string::npos);
}

void test_speculative_metrics() {
    GenerationMetrics metrics = DeterministicLoop{}.run_speculative();
    CHECK(metrics.prefill_tokens == 3);
    CHECK(metrics.decode_output_tokens == 3);
    CHECK(metrics.decode_forward_calls == 1);
    CHECK(metrics.emitted_tokens == 4);
    CHECK(metrics.draft_tokens == 2);
    CHECK(metrics.accepted_tokens == 2);
    CHECK(metrics.raw_decode_forwards_per_second == 1.0);
    CHECK(metrics.generated_output_tokens_per_second == 1.5);
    CHECK(metrics.accepted_tokens_per_second == 1.0);
    const std::string bench = render_decode_benchmark(metrics, 1000.0);
    CHECK(bench.find("raw_decode_forward_s=1.0") != std::string::npos);
    CHECK(bench.find("generated_output_tok_s=1.5") != std::string::npos);
    CHECK(bench.find("accepted_tok_s=1.0") != std::string::npos);
}

void test_zero_decode_is_explicitly_unavailable() {
    GenerationMetrics metrics;
    record_prefill(&metrics, 3);
    record_emitted(&metrics, 1);
    finalize_decode_metrics(&metrics, 10.0, 10.0);
    const std::string bench = render_decode_benchmark(metrics, 10.0);
    CHECK(bench.find("raw_decode_forward_s=unavailable") != std::string::npos);
    CHECK(bench.find("generated_output_tok_s=unavailable") != std::string::npos);
    CHECK(bench.find("accepted_tok_s=unavailable") != std::string::npos);
}

void test_timing_domains_and_invalid_inputs() {
    GenerationMetrics fast_output;
    record_decode_forward(&fast_output);
    record_decode_output(&fast_output, 1);
    finalize_decode_metrics(&fast_output, 0.5, 1.0);
    CHECK(fast_output.raw_decode_forwards_per_second == 2000.0);
    CHECK(fast_output.generated_output_tokens_per_second == 1000.0);

    GenerationMetrics slow_output;
    record_decode_forward(&slow_output);
    record_decode_output(&slow_output, 1);
    finalize_decode_metrics(&slow_output, 0.5, 1000.0);
    CHECK(slow_output.raw_decode_forwards_per_second == fast_output.raw_decode_forwards_per_second);
    CHECK(slow_output.generated_output_tokens_per_second == 1.0);

    for (double invalid : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()}) {
        GenerationMetrics metrics;
        record_decode_forward(&metrics);
        record_decode_output(&metrics, 1);
        finalize_decode_metrics(&metrics, invalid, invalid);
        const std::string bench = render_decode_benchmark(metrics, invalid);
        CHECK(bench.find("raw_decode_forward_s=unavailable") != std::string::npos);
        CHECK(bench.find("generated_output_tok_s=unavailable") != std::string::npos);
    }
}

} // namespace

int main() {
    test_plain_metrics_and_rendering();
    test_speculative_metrics();
    test_zero_decode_is_explicitly_unavailable();
    test_timing_domains_and_invalid_inputs();
    return test_summary("test_generation_loop");
}
