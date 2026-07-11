// test_kvcache - layout contract: every (layer, head, pos) slot for K and V is
// a distinct head_dim-sized region inside a buffer of exactly
// n_layers * n_kv_heads * capacity * head_dim floats per side.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "fp16.h"
#include "kvcache.h"
#include "test_util.h"

using namespace Laplace;

namespace {

std::vector<float> random_unit_vector(int dim, std::mt19937& rng) {
    std::normal_distribution<float> N(0.0f, 1.0f);
    std::vector<float> x(dim);
    float n2 = 0.0f;
    for (int i = 0; i < dim; i++) {
        x[i] = N(rng);
        n2 += x[i] * x[i];
    }
    float inv = 1.0f / std::sqrt(n2);
    for (int i = 0; i < dim; i++) x[i] *= inv;
    return x;
}

} // namespace

int main() {
    const int L = 6, H = 2, D = 8, C = 16;

    KVCache kv;
    CHECK(kv.init(L, H, D, C, KVCacheMode::FP32));

    // Slot addresses must stay within [base, base + L*H*C*D) per buffer.
    const float* k_base = kv.slot_k(0, 0, 0);
    const float* v_base = kv.slot_v(0, 0, 0);
    const ptrdiff_t per_buf = static_cast<ptrdiff_t>(L) * H * C * D;
    int oob = 0;
    for (int l = 0; l < L; l++)
        for (int h = 0; h < H; h++)
            for (int p = 0; p < C; p++) {
                ptrdiff_t ko = kv.slot_k(l, h, p) - k_base;
                ptrdiff_t vo = kv.slot_v(l, h, p) - v_base;
                if (ko < 0 || ko + D > per_buf) oob++;
                if (vo < 0 || vo + D > per_buf) oob++;
            }
    CHECK_MSG(oob == 0, "%d slots out of bounds", oob);

    // Round-trip: write a unique value into every slot, then read all back.
    // Any aliasing between slots (K/K, V/V, or K/V) shows up as a mismatch.
    auto tag = [&](int l, int h, int p, int d, int side) {
        return static_cast<float>((((l * H + h) * C + p) * D + d) * 2 + side);
    };
    for (int l = 0; l < L; l++)
        for (int h = 0; h < H; h++)
            for (int p = 0; p < C; p++)
                for (int d = 0; d < D; d++) {
                    kv.slot_k(l, h, p)[d] = tag(l, h, p, d, 0);
                    kv.slot_v(l, h, p)[d] = tag(l, h, p, d, 1);
                }
    int bad = 0;
    for (int l = 0; l < L; l++)
        for (int h = 0; h < H; h++)
            for (int p = 0; p < C; p++)
                for (int d = 0; d < D; d++) {
                    if (kv.slot_k(l, h, p)[d] != tag(l, h, p, d, 0)) bad++;
                    if (kv.slot_v(l, h, p)[d] != tag(l, h, p, d, 1)) bad++;
                }
    CHECK_MSG(bad == 0, "%d slot values aliased/corrupted", bad);

    // Positions within one (layer, head) must be contiguous rows so the
    // attention loop can walk them with slot_k(l, h, t).
    CHECK(kv.slot_k(2, 1, 5) - kv.slot_k(2, 1, 4) == D);
    CHECK(kv.slot_v(2, 1, 5) - kv.slot_v(2, 1, 4) == D);

    {
        KVCache accounting;
        CHECK(accounting.init(2, 2, 64, 128, KVCacheMode::FP32));
        CHECK(accounting.storage_bytes() == 2 * 2 * 64 * 128 * 2 * 4);
        CHECK(accounting.init(1, 1, 64, 64, KVCacheMode::FP16));
        CHECK(accounting.storage_bytes() == 1 * 1 * 64 * 64 * 2 * 2);
    }

    // LaplaceKV keeps full tiles compressed and the newest partial tile
    // in FP32. Both per-token compatibility loads and whole-head dots must
    // cover the boundary without changing positions.
    {
        constexpr int dim = 96;
        constexpr int context = 130;
        KVCache laplace;
        CHECK(laplace.init(1, 1, dim, context, KVCacheMode::LAPLACE));
        CHECK(!laplace.laplace_rotated());
        size_t tile_bytes = LaplaceKVTile::storage_words(dim)
                          * sizeof(uint32_t);
        CHECK(laplace.encoded_bytes(0) == 0);
        CHECK(laplace.encoded_bytes(1) == 2 * dim * sizeof(float) + 1);
        CHECK(laplace.encoded_bytes(64) == tile_bytes + 1);
        CHECK(laplace.encoded_bytes(65)
              == tile_bytes + 2 * dim * sizeof(float) + 2);
        CHECK(laplace.encoded_bytes(context)
              == 2 * tile_bytes + 2 * 2 * dim * sizeof(float) + 3);
        std::mt19937 rng(77);
        std::vector<float> keys(static_cast<size_t>(context) * dim);
        std::vector<float> values(keys.size());
        for (int token = 0; token < context; token++) {
            std::vector<float> key = random_unit_vector(dim, rng);
            std::vector<float> value = random_unit_vector(dim, rng);
            std::copy(key.begin(), key.end(), keys.begin() + static_cast<size_t>(token) * dim);
            std::copy(value.begin(), value.end(), values.begin() + static_cast<size_t>(token) * dim);
            laplace.store_k(0, 0, token, key.data());
            laplace.store_v(0, 0, token, value.data());
        }

        for (int token : {0, 63, 64, 129}) {
            std::vector<float> loaded(dim);
            laplace.load_k(0, 0, token, loaded.data());
            double error = 0.0;
            for (int d = 0; d < dim; d++) {
                double delta = loaded[d] - keys[static_cast<size_t>(token) * dim + d];
                error += delta * delta;
            }
            CHECK_MSG(std::sqrt(error) < 0.15,
                      "LaplaceKV key roundtrip error %.3f at token %d",
                      std::sqrt(error), token);
        }

        std::vector<float> query = random_unit_vector(dim, rng);
        std::vector<float> query_wh = query;
        std::vector<float> scores(context);
        laplace.dot_k_all_wh(
            0, 0, context, query_wh.data(), 1.0f, scores.data());
        double score_error = 0.0;
        for (int token = 0; token < context; token++) {
            float exact = 0.0f;
            for (int d = 0; d < dim; d++) {
                exact += query[d] * keys[static_cast<size_t>(token) * dim + d];
            }
            double delta = scores[token] - exact;
            score_error += delta * delta;
        }
        CHECK_MSG(std::sqrt(score_error / context) < 0.025,
                  "LaplaceKV whole-head score RMSE %.4f",
                  std::sqrt(score_error / context));

    }

#if defined(LAPLACE_KV_CAPTURE)
    {
        KVCache kivi;
        CHECK(kivi.init(1, 1, 64, 160, KVCacheMode::FP32));
        CHECK(kivi.set_research_kivi_2());
        float key[64];
        float value[64];
        for (int token = 0; token <= 128; token++) {
            for (int dim = 0; dim < 64; dim++) {
                key[dim] = 0.1f * token + 0.001f * dim;
                value[dim] = 0.1f * dim + 0.001f * token;
            }
            kivi.store_k(0, 0, token, key);
            kivi.store_v(0, 0, token, value);
        }
        kivi.load_v(0, 0, 0, value);
        CHECK(std::fabs(value[10] - 1.0f) < 1e-6f);
        for (int dim = 0; dim < 64; dim++) {
            key[dim] = 12.9f + 0.001f * dim;
            value[dim] = 0.1f * dim + 0.129f;
        }
        kivi.store_k(0, 0, 129, key);
        kivi.store_v(0, 0, 129, value);
        kivi.load_k(0, 0, 10, key);
        kivi.load_v(0, 0, 0, value);
        CHECK(std::fabs(key[0] - 1.033203125f) < 1e-6f);
        CHECK(std::fabs(value[10] - 1.033203125f) < 1e-6f);
        kivi.load_k(0, 0, 128, key);
        kivi.load_v(0, 0, 1, value);
        CHECK(std::fabs(key[0] - 12.796875f) < 1e-6f);
        CHECK(std::fabs(value[10] - 1.0009765625f) < 1e-6f);

        KVCache unsupported;
        CHECK(unsupported.init(1, 1, 96, 160, KVCacheMode::FP16));
        CHECK(!unsupported.set_research_kivi_2());
    }

    {
        constexpr int dim = 64;
        KVCache mlx;
        CHECK(mlx.init(1, 1, dim, 1, KVCacheMode::FP32));
        CHECK(mlx.set_research_mlx_q2());
        float key[dim];
        float value[dim];
        for (int index = 0; index < dim; index++) {
            key[index] = -2.0f + 3.0f * index / (dim - 1);
            value[index] = 1.0f + 3.0f * index / (dim - 1);
        }
        mlx.store_k(0, 0, 0, key);
        mlx.store_v(0, 0, 0, value);
        CHECK(mlx.slot_k(0, 0, 0)[0] == -2.0f);
        CHECK(mlx.slot_k(0, 0, 0)[63] == 1.0f);
        CHECK(mlx.slot_v(0, 0, 0)[0] == 1.0f);
        CHECK(mlx.slot_v(0, 0, 0)[63] == 4.0f);

        KVCache unsupported;
        CHECK(unsupported.init(1, 1, 96, 1, KVCacheMode::FP32));
        CHECK(!unsupported.set_research_mlx_q2());
    }

    {
        constexpr int dim = 64;
        KVCache turboquant;
        CHECK(turboquant.init(1, 1, dim, 1, KVCacheMode::FP32));
        CHECK(turboquant.set_research_turboquant_2_5());
        float key[dim];
        float value[dim];
        for (int index = 0; index < dim; index++) {
            key[index] = std::sin((index + 1) * 0.17f)
                       + 0.2f * std::cos((index + 3) * 0.11f);
            value[index] = std::cos((index + 1) * 0.13f)
                         - 0.15f * std::sin((index + 2) * 0.07f);
        }
        turboquant.store_k(0, 0, 0, key);
        turboquant.store_v(0, 0, 0, value);
        CHECK(std::fabs(turboquant.slot_k(0, 0, 0)[0]
                        - 0.0890438706f) < 2e-6f);
        CHECK(std::fabs(turboquant.slot_k(0, 0, 0)[63]
                        + 0.473547965f) < 2e-6f);
        CHECK(std::fabs(turboquant.slot_v(0, 0, 0)[0]
                        - 1.08121371f) < 2e-6f);
        CHECK(std::fabs(turboquant.slot_v(0, 0, 0)[63]
                        + 0.343742847f) < 2e-6f);

        KVCache unsupported;
        CHECK(unsupported.init(1, 1, 96, 1, KVCacheMode::FP32));
        CHECK(!unsupported.set_research_turboquant_2_5());
    }

    // Research baseline seals only after store_v completes a full tile. The
    // fixed checksums come from prototype_kvarn_official.py using the same
    // deterministic input and the released eight-iteration FP16 metadata path.
    {
        constexpr int dim = 8;
        constexpr int group = 8;
        KVCache baseline;
        CHECK(baseline.init(1, 1, dim, 10, KVCacheMode::FP32));
        CHECK(baseline.set_research_baseline(4, 2, group, 0));
        std::vector<float> keys(group * dim);
        std::vector<float> values(group * dim);
        for (int token = 0; token < group; token++) {
            for (int d = 0; d < dim; d++) {
                keys[token * dim + d] =
                    std::sin((token + 1) * (d + 1) * 0.13f)
                    + 0.07f * std::cos((token + 2) * (d + 3) * 0.11f);
                values[token * dim + d] =
                    std::cos((token + 1) * (d + 2) * 0.17f)
                    - 0.05f * std::sin((token + 3) * (d + 1) * 0.07f);
            }
            baseline.store_k(0, 0, token, keys.data() + token * dim);
            if (token == group - 1) {
                CHECK(std::fabs(baseline.slot_k(0, 0, token)[0]
                                - keys[token * dim]) < 1e-7f);
            }
            baseline.store_v(0, 0, token, values.data() + token * dim);
        }

        double key_sum = 0.0;
        double key_square = 0.0;
        double value_sum = 0.0;
        double value_square = 0.0;
        for (int token = 0; token < group; token++) {
            for (int d = 0; d < dim; d++) {
                double key = baseline.slot_k(0, 0, token)[d];
                double value = baseline.slot_v(0, 0, token)[d];
                key_sum += key;
                key_square += key * key;
                value_sum += value;
                value_square += value * value;
            }
        }
        CHECK_MSG(std::fabs(key_sum - 19.80417633) < 0.002,
                  "research K sum %.9f", key_sum);
        CHECK_MSG(std::fabs(key_square - 29.33460541) < 0.004,
                  "research K square %.9f", key_square);
        CHECK_MSG(std::fabs(value_sum + 5.20322418) < 0.002,
                  "research V sum %.9f", value_sum);
        CHECK_MSG(std::fabs(value_square - 29.28296536) < 0.004,
                  "research V square %.9f", value_square);
        CHECK(std::fabs(baseline.slot_k(0, 0, 0)[0] - 0.165649414f) < 1e-5f);
        CHECK(std::fabs(baseline.slot_v(0, 0, 7)[7] - 1.067382812f) < 1e-5f);

        float tail[dim];
        for (int d = 0; d < dim; d++) tail[d] = 0.25f * d;
        baseline.store_k(0, 0, group, tail);
        baseline.store_v(0, 0, group, tail);
        for (int d = 0; d < dim; d++) {
            CHECK(baseline.slot_k(0, 0, group)[d]
                  == fp16_to_fp32(fp32_to_fp16(tail[d])));
            CHECK(baseline.slot_v(0, 0, group)[d]
                  == fp16_to_fp32(fp32_to_fp16(tail[d])));
        }

        baseline.free();
        CHECK(baseline.init(1, 1, dim, group, KVCacheMode::FP32));
        for (int token = 0; token < group; token++) {
            baseline.store_k(0, 0, token, keys.data() + token * dim);
            baseline.store_v(0, 0, token, values.data() + token * dim);
        }
        CHECK(std::equal(keys.begin(), keys.end(), baseline.head_k(0, 0)));

        KVCache fp16;
        CHECK(fp16.init(1, 1, dim, group, KVCacheMode::FP16));
        CHECK(!fp16.set_research_baseline(4, 2, group, 0));
        KVCache unsupported;
        CHECK(unsupported.init(1, 1, 7, group, KVCacheMode::FP32));
        CHECK(!unsupported.set_research_baseline(4, 2, group, 0));

        constexpr int even_dim = 6;
        KVCache even;
        CHECK(even.init(1, 1, even_dim, group, KVCacheMode::FP32));
        CHECK(even.set_research_baseline(4, 2, group, 0));
        std::vector<float> even_keys(group * even_dim);
        std::vector<float> even_values(group * even_dim);
        for (int token = 0; token < group; token++) {
            for (int d = 0; d < even_dim; d++) {
                even_keys[token * even_dim + d] =
                    std::sin((token + 1) * (d + 1) * 0.13f)
                    + 0.07f * std::cos((token + 2) * (d + 3) * 0.11f);
                even_values[token * even_dim + d] =
                    std::cos((token + 1) * (d + 2) * 0.17f)
                    - 0.05f * std::sin((token + 3) * (d + 1) * 0.07f);
            }
            even.store_k(0, 0, token,
                         even_keys.data() + token * even_dim);
            even.store_v(0, 0, token,
                         even_values.data() + token * even_dim);
        }
        double even_key_sum = 0.0;
        double even_value_sum = 0.0;
        for (int index = 0; index < group * even_dim; index++) {
            even_key_sum += even.head_k(0, 0)[index];
            even_value_sum += even.head_v(0, 0)[index];
        }
        CHECK_MSG(std::fabs(even_key_sum - 17.323296) < 0.002,
                  "even research K sum %.9f", even_key_sum);
        CHECK_MSG(std::fabs(even_value_sum + 5.704529) < 0.002,
                  "even research V sum %.9f", even_value_sum);

        KVCache tail_codec;
        CHECK(tail_codec.init(1, 1, dim, group, KVCacheMode::FP32));
        CHECK(tail_codec.set_research_baseline(
            4, 2, group, 0, 8, 3, 2));
        tail_codec.store_k(0, 0, 0, keys.data());
        tail_codec.store_v(0, 0, 0, values.data());
        CHECK(!std::equal(keys.begin(), keys.begin() + dim,
                          tail_codec.slot_k(0, 0, 0)));
        CHECK(!std::equal(values.begin(), values.begin() + dim,
                          tail_codec.slot_v(0, 0, 0)));

        constexpr int bfp_dim = 16;
        KVCache bfp;
        CHECK(bfp.init(1, 1, bfp_dim, 1, KVCacheMode::FP32));
        CHECK(bfp.set_research_bfp3());
        float bfp_key[bfp_dim];
        float bfp_value[bfp_dim];
        for (int d = 0; d < bfp_dim; d++) {
            bfp_key[d] = std::sin((d + 1) * 0.31f);
            bfp_value[d] = std::cos((d + 1) * 0.23f);
        }
        bfp.store_k(0, 0, 0, bfp_key);
        bfp.store_v(0, 0, 0, bfp_value);
        CHECK(!std::equal(bfp_key, bfp_key + bfp_dim,
                          bfp.slot_k(0, 0, 0)));
        CHECK(!std::equal(bfp_value, bfp_value + bfp_dim,
                          bfp.slot_v(0, 0, 0)));
    }
#endif

    return test_summary("test_kvcache");
}
