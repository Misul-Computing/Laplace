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
        finalize_decode_metrics(&metrics, 500.0);
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
        finalize_decode_metrics(&metrics, 1000.0);
        return metrics;
    }
};

void test_plain_metrics_and_rendering() {
    GenerationMetrics metrics = DeterministicLoop{}.run_plain();
    CHECK(metrics.prefill_tokens == 3);
    CHECK(metrics.decode_output_tokens == 2);
    CHECK(metrics.decode_forward_calls == 2);
    CHECK(metrics.emitted_tokens == 3);
    CHECK(metrics.decode_tokens_per_second == 4.0);
    const std::string bench = render_decode_benchmark(metrics, 500.0);
    CHECK(bench.find("decode:  2 tokens") != std::string::npos);
    CHECK(bench.find("4.0 tok/s") != std::string::npos);
}

void test_speculative_metrics() {
    GenerationMetrics metrics = DeterministicLoop{}.run_speculative();
    CHECK(metrics.prefill_tokens == 3);
    CHECK(metrics.decode_output_tokens == 3);
    CHECK(metrics.decode_forward_calls == 1);
    CHECK(metrics.emitted_tokens == 4);
    CHECK(metrics.draft_tokens == 2);
    CHECK(metrics.accepted_tokens == 2);
}

void test_zero_decode_is_explicitly_unavailable() {
    GenerationMetrics metrics;
    record_prefill(&metrics, 3);
    record_emitted(&metrics, 1);
    finalize_decode_metrics(&metrics, 10.0);
    CHECK(render_decode_benchmark(metrics, 10.0).find(
              "decode_tok_s=unavailable") != std::string::npos);
}

} // namespace

int main() {
    test_plain_metrics_and_rendering();
    test_speculative_metrics();
    test_zero_decode_is_explicitly_unavailable();
    return test_summary("test_generation_loop");
}
