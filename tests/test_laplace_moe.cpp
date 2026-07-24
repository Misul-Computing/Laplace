#include <cstdint>
#include <vector>

#include "laplace_moe.h"
#include "test_util.h"

using namespace Laplace;

int main() {
    constexpr int experts = 4;
    constexpr int bytes_per_expert = 4096;
    std::vector<uint8_t> storage(experts * bytes_per_expert, 7);

    Tensor tensor;
    tensor.type = GGMLType::U8;
    tensor.n_dims = 3;
    tensor.dims[0] = bytes_per_expert;
    tensor.dims[1] = 1;
    tensor.dims[2] = experts;
    tensor.data = storage.data();

    LaplaceMoE::set_cache_budget(storage.size());
    const int selected[] = {1, 3};

    ExpertAcquireStats first =
        LaplaceMoE::acquire(&tensor, selected, 2);
    CHECK(first.requested == 2);
    CHECK(first.hits == 0);
    CHECK(first.misses == 2);

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
