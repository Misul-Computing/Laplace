#include <cstdint>
#include <vector>

#include "laplace_moe.h"
#include "test_util.h"

using namespace Laplace;

int main() {
    constexpr int experts = 4;
    constexpr int bytes_per_expert = 4096;
    std::vector<uint8_t> storage(experts * bytes_per_expert);
    for (int e = 0; e < experts; ++e)
        for (int b = 0; b < bytes_per_expert; ++b)
            storage[static_cast<size_t>(e) * bytes_per_expert + b] =
                static_cast<uint8_t>(e * 17 + b);

    Tensor tensor;
    tensor.type = GGMLType::U8;
    tensor.n_dims = 3;
    tensor.dims[0] = bytes_per_expert;
    tensor.dims[1] = 1;
    tensor.dims[2] = experts;
    tensor.data = storage.data();

    LaplaceMoE::set_cache_budget(storage.size());
    const int selected[] = {1, 3};

    Tensor packed;
    ExpertAcquireStats copied =
        LaplaceMoE::load_experts(tensor, selected, 2, &packed);
    CHECK(copied.requested == 2);
    CHECK(copied.misses == 2);
    CHECK(packed.dims[2] == 2);
    CHECK(packed.data != nullptr);
    CHECK(packed.data != storage.data());
    for (int i = 0; i < bytes_per_expert; ++i) {
        CHECK(packed.data[i] ==
              storage[1 * bytes_per_expert + i]);
        CHECK(packed.data[bytes_per_expert + i] ==
              storage[3 * bytes_per_expert + i]);
    }

    const uint8_t* bases[2] = {};
    ExpertAcquireStats aliased =
        LaplaceMoE::load_experts(tensor, selected, 2, nullptr, bases);
    CHECK(aliased.requested == 2);
    CHECK(bases[0] == storage.data() + 1 * bytes_per_expert);
    CHECK(bases[1] == storage.data() + 3 * bytes_per_expert);

    ExpertAcquireStats first =
        LaplaceMoE::acquire(&tensor, selected, 2);
    CHECK(first.requested == 2);
    CHECK(first.hits == 2);
    CHECK(first.misses == 0);

    ExpertAcquireStats second =
        LaplaceMoE::acquire(&tensor, selected, 2);
    CHECK(second.requested == 2);
    CHECK(second.hits == 2);
    CHECK(second.misses == 0);
    CHECK(second.bytes_read == 0);

    const int invalid[] = {experts};
    ExpertAcquireStats rejected =
        LaplaceMoE::acquire(&tensor, invalid, 1);
    CHECK(rejected.requested == 0);
    CHECK(rejected.invalid == 1);

    std::vector<uint8_t> uncached_storage(experts * bytes_per_expert, 9);
    tensor.data = uncached_storage.data();
    LaplaceMoE::set_cache_budget(0);
    ExpertAcquireStats uncached_first =
        LaplaceMoE::acquire(&tensor, selected, 2);
    const int workers_after_first = LaplaceMoE::io_worker_count();
    ExpertAcquireStats uncached_second =
        LaplaceMoE::acquire(&tensor, selected, 2);
    CHECK(uncached_first.misses == 2);
    CHECK(uncached_second.misses == 2);
    CHECK(workers_after_first > 0);
    CHECK(LaplaceMoE::io_worker_count() == workers_after_first);
    CHECK(workers_after_first <= 4);

    return test_summary("test_laplace_moe");
}
