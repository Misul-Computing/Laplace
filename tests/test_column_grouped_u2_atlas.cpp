#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include <unistd.h>

#include "column_grouped_u2_atlas.h"
#include "test_util.h"

using namespace Laplace;

namespace {

ColumnGroupedU2AtlasSource source(uint32_t id, GGMLType format,
                                  std::vector<uint8_t>& bytes,
                                  std::vector<float>& importance,
                                  uint64_t k = 256, uint64_t n = 256) {
    return {id, format, bytes, k, n, importance};
}

void test_atlas_is_one_transactional_immutable_mapping() {
    std::vector<uint8_t> q4(256u * 144u, 0);
    std::vector<uint8_t> q6(256u * 210u, 0);
    std::vector<float> importance(256, 1.0f);
    std::array<ColumnGroupedU2AtlasSource, 2> sources = {
        source(7, GGMLType::Q4_K, q4, importance),
        source(8, GGMLType::Q6_K, q6, importance),
    };
    auto built = build_column_grouped_u2_atlas(sources);
    CHECK(std::holds_alternative<ColumnGroupedU2Atlas>(built));
    if (!std::holds_alternative<ColumnGroupedU2Atlas>(built)) return;
    const auto& atlas = std::get<ColumnGroupedU2Atlas>(built);
    CHECK(atlas.entries().size() == sources.size());
    CHECK(atlas.data() != nullptr);
    CHECK(atlas.logical_bytes() > 0);
    CHECK(atlas.mapped_bytes() >= atlas.logical_bytes());
    CHECK(atlas.source_bytes() == q4.size() + q6.size());
    const long page = ::sysconf(_SC_PAGESIZE);
    CHECK(page > 0);
    if (page > 0) CHECK(atlas.mapped_bytes() % static_cast<size_t>(page) == 0);
    std::fill(q4.begin(), q4.end(), 0xff);
    std::fill(q6.begin(), q6.end(), 0xff);
    for (const auto& entry : atlas.entries()) {
        CHECK(entry.values_offset % 128u == 0);
        CHECK(entry.scales_offset % 128u == 0);
        CHECK(entry.biases_offset % 128u == 0);
        CHECK(entry.storage.planes.values == atlas.data() + entry.values_offset);
        CHECK(reinterpret_cast<const uint8_t*>(entry.storage.planes.scales) ==
              atlas.data() + entry.scales_offset);
        CHECK(reinterpret_cast<const uint8_t*>(entry.storage.planes.biases) ==
              atlas.data() + entry.biases_offset);
        ColumnGroupedAffineUInt2SkipV1Error error{};
        CHECK(column_grouped_affine_uint2_skip_v1_validate(
            entry.storage, entry.storage.source_digest,
            entry.storage.provenance_digest, &error));
        std::vector<float> decoded(256u * 256u, 1.0f);
        CHECK(column_grouped_affine_uint2_skip_v1_decode(
            entry.storage, entry.storage.source_digest,
            entry.storage.provenance_digest, decoded, &error));
        CHECK(std::all_of(decoded.begin(), decoded.end(),
                          [](float value) { return value == 0.0f; }));
    }
}

void test_atlas_rejects_ambiguous_or_malformed_sources() {
    std::vector<uint8_t> q4(256u * 144u, 0);
    std::vector<float> importance(256, 1.0f);
    const ColumnGroupedU2AtlasSource valid =
        source(7, GGMLType::Q4_K, q4, importance);
    CHECK(std::holds_alternative<ColumnGroupedU2AtlasError>(
        build_column_grouped_u2_atlas(
            std::span<const ColumnGroupedU2AtlasSource>{})));
    std::array<ColumnGroupedU2AtlasSource, 2> duplicate = {valid, valid};
    CHECK(std::holds_alternative<ColumnGroupedU2AtlasError>(
        build_column_grouped_u2_atlas(duplicate)));
    auto malformed = valid;
    malformed.source = malformed.source.first(malformed.source.size() - 1u);
    CHECK(std::holds_alternative<ColumnGroupedU2AtlasError>(
        build_column_grouped_u2_atlas(std::span(&malformed, 1))));
    malformed = valid;
    malformed.source_format = GGMLType::F16;
    CHECK(std::holds_alternative<ColumnGroupedU2AtlasError>(
        build_column_grouped_u2_atlas(std::span(&malformed, 1))));
    malformed = valid;
    malformed.logical_n = 255;
    CHECK(std::holds_alternative<ColumnGroupedU2AtlasError>(
        build_column_grouped_u2_atlas(std::span(&malformed, 1))));
    malformed = valid;
    malformed.importance = malformed.importance.first(255);
    CHECK(std::holds_alternative<ColumnGroupedU2AtlasError>(
        build_column_grouped_u2_atlas(std::span(&malformed, 1))));
}

}  // namespace

int main() {
    test_atlas_is_one_transactional_immutable_mapping();
    test_atlas_rejects_ambiguous_or_malformed_sources();
    return test_summary("test_column_grouped_u2_atlas");
}
