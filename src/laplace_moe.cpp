// laplace_moe.cpp - Mac M-series SSD expert streaming
#include "laplace_moe.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#include <sys/mman.h>
#endif

#include <unistd.h>

namespace Laplace {

bool LaplaceMoE::streaming_enabled_ = false;

struct ExpertAcquireState {
    std::mutex mutex;
    std::condition_variable complete;
    ExpertAcquireStats stats;
    int pending = 0;
};

namespace {
int g_file_fd = -1;
const uint8_t* g_mmap_base = nullptr;
int g_io_threads = 4;
const size_t IO_CHUNK = 1 << 20;

struct ExpertIoJob {
    const Tensor* tensor = nullptr;
    int expert = 0;
    size_t bytes = 0;
    std::shared_ptr<ExpertAcquireState> state;
};

void read_expert(const ExpertIoJob& job, std::vector<uint8_t>* scratch) {
    const uint8_t* source = LaplaceMoE::expert_data(
        job.tensor, job.expert);
    if (g_file_fd >= 0 && g_mmap_base &&
        source >= g_mmap_base) {
        const off_t file_offset =
            static_cast<off_t>(source - g_mmap_base);
        scratch->resize(std::min(IO_CHUNK, job.bytes));
        size_t offset = 0;
        while (offset < job.bytes) {
            const size_t count =
                std::min(scratch->size(), job.bytes - offset);
            const ssize_t result = pread(
                g_file_fd, scratch->data(), count,
                file_offset + static_cast<off_t>(offset));
            if (result > 0) {
                offset += static_cast<size_t>(result);
            } else if (result < 0 && errno == EINTR) {
                continue;
            } else {
                break;
            }
        }
        return;
    }

    const long page_result = sysconf(_SC_PAGESIZE);
    const size_t page =
        page_result > 0 ? static_cast<size_t>(page_result) : 4096;
    const volatile uint8_t* bytes = source;
    for (size_t offset = 0; offset < job.bytes; offset += page) {
        (void)bytes[offset];
    }
    if (job.bytes) (void)bytes[job.bytes - 1];
}

class ExpertIoPool {
public:
    ExpertIoPool() {
        const int count = std::clamp(g_io_threads, 1, 32);
        workers_.reserve(count);
        for (int i = 0; i < count; ++i) {
            workers_.emplace_back([this] { worker(); });
        }
    }

    ~ExpertIoPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (std::thread& worker : workers_) worker.join();
    }

    void submit(ExpertIoJob job) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            jobs_.push_back(std::move(job));
        }
        ready_.notify_one();
    }

    int worker_count() const {
        return static_cast<int>(workers_.size());
    }

private:
    void worker() {
        std::vector<uint8_t> scratch;
        while (true) {
            ExpertIoJob job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [&] {
                    return stopping_ || !jobs_.empty();
                });
                if (stopping_ && jobs_.empty()) return;
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            read_expert(job, &scratch);
            {
                std::lock_guard<std::mutex> lock(job.state->mutex);
                job.state->stats.bytes_read += job.bytes;
                job.state->pending--;
            }
            job.state->complete.notify_all();
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<ExpertIoJob> jobs_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};

ExpertIoPool& io_pool() {
    static ExpertIoPool pool;
    return pool;
}
} // namespace

void LaplaceMoE::set_file_fd(int fd) {
    g_file_fd = fd;
    if (const char* env = std::getenv("LAPLACE_IO_THREADS")) {
        int n = std::atoi(env);
        if (n > 0 && n <= 32) g_io_threads = n;
    }
}

void LaplaceMoE::set_mmap_base(const uint8_t* base) {
    g_mmap_base = base;
}

namespace {
struct CacheEntry {
    size_t bytes = 0;
    int last_accessed = 0;
    bool locked = false;
};
std::unordered_map<const uint8_t*, std::unordered_map<int, CacheEntry>> g_cache;
size_t g_budget = 0;
size_t g_usage = 0;
int g_current_token = 0;
std::mutex g_cache_mutex;
bool g_mlock_ok = true;
} // namespace

size_t LaplaceMoE::per_expert_bytes(const Tensor* tensor) {
    // 3D tensor: dims[2] is the expert count. Each expert is dims[0]*dims[1].
    if (tensor->n_dims < 3) return tensor->nbytes();
    return tensor->nbytes() / tensor->dims[2];
}

const uint8_t* LaplaceMoE::expert_data(const Tensor* tensor, int expert_idx) {
    return tensor->data + static_cast<size_t>(expert_idx) * per_expert_bytes(tensor);
}

void LaplaceMoE::pagein_expert_mt(const Tensor* tensor, int expert_idx) {
    (void)acquire(tensor, &expert_idx, 1);
}

void LaplaceMoE::pagein_all_mt(const Tensor* tensor, const int* expert_idx, int n) {
    (void)acquire(tensor, expert_idx, n);
}

ExpertAcquireTicket LaplaceMoE::prefetch(
        const Tensor* tensor, const int* expert_idx, int n) {
    auto state = std::make_shared<ExpertAcquireState>();
    if (!tensor || !tensor->data || !expert_idx || n <= 0 ||
        tensor->n_dims < 3 || tensor->dims[2] == 0) {
        state->stats.invalid = std::max(n, 0);
        return ExpertAcquireTicket(std::move(state));
    }

    const size_t bytes = per_expert_bytes(tensor);
    std::vector<ExpertIoJob> jobs;
    for (int i = 0; i < n; ++i) {
        const int id = expert_idx[i];
        if (id < 0 || static_cast<uint64_t>(id) >= tensor->dims[2]) {
            state->stats.invalid++;
            continue;
        }
        state->stats.requested++;

        bool resident = false;
        {
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            auto tensor_it = g_cache.find(tensor->data);
            if (tensor_it != g_cache.end()) {
                auto expert_it = tensor_it->second.find(id);
                resident = expert_it != tensor_it->second.end() &&
                           expert_it->second.locked;
                if (resident) {
                    expert_it->second.last_accessed = g_current_token;
                }
            }
        }
        if (resident) {
            state->stats.hits++;
            continue;
        }

        touch_expert(tensor, id);
        {
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            auto tensor_it = g_cache.find(tensor->data);
            if (tensor_it != g_cache.end()) {
                auto expert_it = tensor_it->second.find(id);
                resident = expert_it != tensor_it->second.end() &&
                           expert_it->second.locked;
            }
        }
        state->stats.misses++;
        if (!resident) {
            jobs.push_back({tensor, id, bytes, state});
        }
    }
    state->pending = static_cast<int>(jobs.size());
    for (ExpertIoJob& job : jobs) io_pool().submit(std::move(job));
    return ExpertAcquireTicket(std::move(state));
}

ExpertAcquireStats LaplaceMoE::wait(
        const ExpertAcquireTicket& ticket) {
    if (!ticket.state_) return {};
    std::unique_lock<std::mutex> lock(ticket.state_->mutex);
    ticket.state_->complete.wait(lock, [&] {
        return ticket.state_->pending == 0;
    });
    return ticket.state_->stats;
}

ExpertAcquireStats LaplaceMoE::acquire(
        const Tensor* tensor, const int* expert_idx, int n) {
    return wait(prefetch(tensor, expert_idx, n));
}

int LaplaceMoE::io_worker_count() {
    return io_pool().worker_count();
}

void LaplaceMoE::set_cache_budget(size_t bytes) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_budget = bytes;
}

void LaplaceMoE::set_current_token(int n) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_current_token = n;
}

void LaplaceMoE::touch_expert(const Tensor* tensor, int expert_idx) {
#ifdef __APPLE__
    if (!tensor || !tensor->data) return;
    if (g_budget == 0 || !g_mlock_ok) return;

    size_t sz = per_expert_bytes(tensor);
    const uint8_t* key = tensor->data;

    std::lock_guard<std::mutex> lock(g_cache_mutex);

    auto& experts = g_cache[key];
    auto it = experts.find(expert_idx);
    if (it == experts.end()) {
        it = experts.emplace(expert_idx, CacheEntry{sz, g_current_token, false}).first;
    }
    CacheEntry& entry = it->second;
    entry.last_accessed = g_current_token;

    if (entry.locked) return;

    while (g_usage + sz > g_budget) {
        const uint8_t* lru_key = nullptr;
        int lru_idx = -1;
        int lru_time = INT_MAX;
        for (auto& [tk, exps] : g_cache) {
            for (auto& [idx, e] : exps) {
                if (e.locked && e.last_accessed < lru_time) {
                    lru_time = e.last_accessed;
                    lru_key = tk;
                    lru_idx = idx;
                }
            }
        }
        if (lru_key == nullptr) break;

        CacheEntry& lru = g_cache[lru_key][lru_idx];
        const uint8_t* ptr = lru_key + static_cast<size_t>(lru_idx) * lru.bytes;
        if (munlock(const_cast<uint8_t*>(ptr), lru.bytes) != 0)
            perror("laplace-moe: munlock (evict)");
        lru.locked = false;
        g_usage -= lru.bytes;
    }

    if (g_usage + sz > g_budget) return;

    const uint8_t* ptr = key + static_cast<size_t>(expert_idx) * sz;
    if (mlock(const_cast<uint8_t*>(ptr), sz) != 0) {
        fprintf(stderr, "laplace-moe: mlock failed (%zu bytes), disabling expert cache\n", sz);
        g_mlock_ok = false;
        return;
    }
    entry.locked = true;
    g_usage += sz;
#else
    (void)tensor; (void)expert_idx;
#endif
}

void LaplaceMoE::evict_cold(int k_tokens) {
#ifdef __APPLE__
    if (g_budget == 0 || !g_mlock_ok) return;

    std::lock_guard<std::mutex> lock(g_cache_mutex);
    int threshold = g_current_token - k_tokens;

    for (auto& [key, experts] : g_cache) {
        for (auto& [idx, entry] : experts) {
            if (entry.locked && entry.last_accessed < threshold) {
                const uint8_t* ptr = key + static_cast<size_t>(idx) * entry.bytes;
                if (munlock(const_cast<uint8_t*>(ptr), entry.bytes) != 0)
                    perror("laplace-moe: munlock (cold)");
                entry.locked = false;
                g_usage -= entry.bytes;
            }
        }
    }
#else
    (void)k_tokens;
#endif
}

} // namespace Laplace
