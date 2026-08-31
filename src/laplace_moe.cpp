// laplace_moe.cpp - GEMV from the file map; WILLNEED on first touch
#include "laplace_moe.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __APPLE__
#include <fcntl.h>
#include <sys/mman.h>
#endif

#include <unistd.h>

namespace Laplace {

bool LaplaceMoE::streaming_enabled_ = false;

struct ExpertAcquireState {
    ExpertAcquireStats stats;
    int pending = 0;
};

namespace {
int g_direct_fd = -1;
const uint8_t* g_mmap_base = nullptr;
int g_io_threads = 4;

struct Key {
    const Tensor* tensor;
    int id;
    bool operator==(const Key& o) const {
        return tensor == o.tensor && id == o.id;
    }
};

struct KeyHash {
    size_t operator()(const Key& k) const {
        return std::hash<const Tensor*>{}(k.tensor) ^
               (static_cast<size_t>(k.id) * 0x9e3779b97f4a7c15ull);
    }
};

std::unordered_set<Key, KeyHash> g_seen;
std::mutex g_mu;
size_t g_budget = 0;
int g_current_token = 0;
std::atomic<size_t> g_hits{0};
std::atomic<size_t> g_misses{0};
std::atomic<size_t> g_bytes{0};

struct PackedView {
    std::vector<uint8_t> data;
};
std::unordered_map<const Tensor*, PackedView> g_views;

struct StageReport {
    ~StageReport() {
        const size_t hits = g_hits.load();
        const size_t misses = g_misses.load();
        const size_t bytes = g_bytes.load();
        if (hits + misses == 0) return;
        fprintf(stderr, "moe-stage: %zu hits, %zu misses, %.2f GB advised\n",
                hits, misses, bytes / 1e9);
    }
};
StageReport g_stage_report;

void close_direct_fd() {
    if (g_direct_fd >= 0) {
        ::close(g_direct_fd);
        g_direct_fd = -1;
    }
}

void advise_slice(const Tensor* tensor, int expert) {
#ifdef __APPLE__
    if (!tensor || !tensor->data || expert < 0) return;
    if (tensor->n_dims < 3 ||
        static_cast<uint64_t>(expert) >= tensor->dims[2]) return;
    const uint8_t* p = LaplaceMoE::expert_data(tensor, expert);
    const size_t n = LaplaceMoE::per_expert_bytes(tensor);
    (void)madvise(const_cast<uint8_t*>(p), n, MADV_WILLNEED);
    if (g_direct_fd >= 0 && g_mmap_base && p >= g_mmap_base) {
        struct radvisory ra;
        ra.ra_offset = static_cast<off_t>(p - g_mmap_base);
        ra.ra_count = n > static_cast<size_t>(INT_MAX)
                          ? INT_MAX
                          : static_cast<int>(n);
        (void)fcntl(g_direct_fd, F_RDADVISE, &ra);
    }
#else
    (void)tensor;
    (void)expert;
#endif
}

void fill_dest(Tensor* dest, const Tensor& src, uint8_t* data, int n) {
    dest->name = src.name;
    dest->type = src.type;
    dest->n_dims = 3;
    dest->dims[0] = src.dims[0];
    dest->dims[1] = src.dims[1];
    dest->dims[2] = static_cast<uint64_t>(n);
    dest->dims[3] = 0;
    dest->data = data;
    dest->scales = src.scales;
    dest->biases = src.biases;
    dest->mlx_bits = src.mlx_bits;
    dest->mlx_group_size = src.mlx_group_size;
}
} // namespace

void LaplaceMoE::set_file_fd(int fd) {
    close_direct_fd();
#ifdef __APPLE__
    if (fd >= 0) g_direct_fd = ::dup(fd);
#endif
    if (const char* env = std::getenv("LAPLACE_IO_THREADS")) {
        int n = std::atoi(env);
        if (n > 0 && n <= 32) g_io_threads = n;
    }
}

void LaplaceMoE::set_mmap_base(const uint8_t* base) {
    g_mmap_base = base;
}

size_t LaplaceMoE::per_expert_bytes(const Tensor* tensor) {
    if (tensor->n_dims < 3) return tensor->nbytes();
    return tensor->nbytes() / tensor->dims[2];
}

const uint8_t* LaplaceMoE::expert_data(const Tensor* tensor, int expert_idx) {
    return tensor->data + static_cast<size_t>(expert_idx) * per_expert_bytes(tensor);
}

void LaplaceMoE::drop_mmap(const Tensor* tensor) {
#ifdef __APPLE__
    if (!tensor || !tensor->data) return;
    const size_t n = tensor->nbytes();
    if (n == 0) return;
    (void)madvise(const_cast<uint8_t*>(tensor->data), n, MADV_DONTNEED);
#else
    (void)tensor;
#endif
}

void LaplaceMoE::hint(const Tensor* tensor, int expert_idx) {
    advise_slice(tensor, expert_idx);
}

ExpertAcquireStats LaplaceMoE::load_experts(
        const Tensor& src, const int* ids, int n, Tensor* dest,
        const uint8_t** bases) {
    ExpertAcquireStats stats;
    if (dest) *dest = Tensor{};
    if (!src.data || !ids || n <= 0 || src.n_dims < 3 || src.dims[2] == 0)
        return stats;

    const size_t bytes = per_expert_bytes(&src);
    std::vector<const uint8_t*> got(static_cast<size_t>(n), nullptr);

    for (int i = 0; i < n; ++i) {
        const int id = ids[i];
        if (id < 0 || static_cast<uint64_t>(id) >= src.dims[2]) {
            stats.invalid++;
            continue;
        }
        stats.requested++;
        const uint8_t* p = expert_data(&src, id);
        got[static_cast<size_t>(i)] = p;

        bool hit = false;
        if (g_budget > 0) {
            std::lock_guard<std::mutex> lock(g_mu);
            Key key{&src, id};
            hit = !g_seen.insert(key).second;
        }
        if (hit) {
            stats.hits++;
        } else {
            stats.misses++;
            stats.bytes_read += bytes;
            advise_slice(&src, id);
        }
    }

    g_hits += static_cast<size_t>(stats.hits);
    g_misses += static_cast<size_t>(stats.misses);
    g_bytes += stats.bytes_read;

    if (bases) {
        for (int i = 0; i < n; ++i) bases[i] = got[static_cast<size_t>(i)];
    }
    if (dest) {
        PackedView& view = g_views[&src];
        view.data.resize(static_cast<size_t>(n) * bytes);
        for (int i = 0; i < n; ++i) {
            if (!got[static_cast<size_t>(i)]) continue;
            std::memcpy(view.data.data() + static_cast<size_t>(i) * bytes,
                        got[static_cast<size_t>(i)], bytes);
        }
        fill_dest(dest, src, view.data.data(), n);
    }
    return stats;
}

ExpertAcquireTicket LaplaceMoE::prefetch(
        const Tensor* tensor, const int* expert_idx, int n) {
    auto state = std::make_shared<ExpertAcquireState>();
    if (!tensor) {
        state->stats.invalid = std::max(n, 0);
        return ExpertAcquireTicket(std::move(state));
    }
    Tensor dest;
    state->stats = load_experts(*tensor, expert_idx, n, &dest, nullptr);
    state->pending = 0;
    return ExpertAcquireTicket(std::move(state));
}

ExpertAcquireStats LaplaceMoE::wait(const ExpertAcquireTicket& ticket) {
    if (!ticket.state_) return {};
    return ticket.state_->stats;
}

ExpertAcquireStats LaplaceMoE::acquire(
        const Tensor* tensor, const int* expert_idx, int n) {
    return wait(prefetch(tensor, expert_idx, n));
}

int LaplaceMoE::io_worker_count() {
    return g_io_threads;
}

void LaplaceMoE::set_cache_budget(size_t bytes) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_budget = bytes;
    if (bytes == 0) g_seen.clear();
}

void LaplaceMoE::set_current_token(int n) {
    g_current_token = n;
}

void LaplaceMoE::touch_expert(const Tensor* tensor, int expert_idx) {
    (void)tensor;
    (void)expert_idx;
}

void LaplaceMoE::evict_cold(int k_tokens) {
    (void)k_tokens;
}

} // namespace Laplace
