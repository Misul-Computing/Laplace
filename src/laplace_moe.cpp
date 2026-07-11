// laplace_moe.cpp - Mac M-series SSD expert streaming
#include "laplace_moe.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
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

namespace {
int g_file_fd = -1;
const uint8_t* g_mmap_base = nullptr;
int g_io_threads = 4;
const size_t IO_CHUNK = 1 << 20;
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
    if (!tensor || !tensor->data) return;
    size_t per = per_expert_bytes(tensor);
    if (per == 0) return;
    const uint8_t* src = expert_data(tensor, expert_idx);
    int nthreads = g_io_threads;

    if (g_file_fd >= 0 && g_mmap_base) {
        off_t file_offset = static_cast<off_t>(src - g_mmap_base);
        size_t per_thread = (per + nthreads - 1) / nthreads;

        std::thread threads[32];
        for (int t = 0; t < nthreads; t++) {
            size_t start = static_cast<size_t>(t) * per_thread;
            size_t end = std::min(start + per_thread, per);
            if (start >= end) continue;

            threads[t] = std::thread([start, end, file_offset]() {
                std::vector<uint8_t> buf(IO_CHUNK);
                size_t off = start;
                while (off < end) {
                    size_t to_read = std::min(IO_CHUNK, end - off);
                    ssize_t n = pread(g_file_fd, buf.data(), to_read,
                                      file_offset + static_cast<off_t>(off));
                    if (n <= 0) break;
                    off += static_cast<size_t>(n);
                }
            });
        }
        for (int t = 0; t < nthreads; t++) {
            if (threads[t].joinable()) threads[t].join();
        }
    } else {
        long ps = sysconf(_SC_PAGESIZE);
        size_t page = ps > 0 ? static_cast<size_t>(ps) : 4096;
        size_t per_thread = (per + nthreads - 1) / nthreads;

        std::thread threads[32];
        for (int t = 0; t < nthreads; t++) {
            size_t start = (static_cast<size_t>(t) * per_thread) & ~(page - 1);
            size_t end = std::min(start + per_thread, per);
            if (start >= end) continue;

            threads[t] = std::thread([src, start, end, page]() {
                const volatile uint8_t* p = src + start;
                for (size_t off = 0; off + page <= end - start; off += page)
                    (void)p[off];
                if (start < end) (void)src[end - 1];
            });
        }
        for (int t = 0; t < nthreads; t++) {
            if (threads[t].joinable()) threads[t].join();
        }
    }
}

void LaplaceMoE::pagein_all_mt(const Tensor* tensor, const int* expert_idx, int n) {
    if (!tensor || !tensor->data || n <= 0) return;
    int nthreads = g_io_threads;
    if (n < nthreads) nthreads = n;

    std::thread threads[32];
    int per_thread = (n + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; t++) {
        int start = t * per_thread;
        int end = std::min(start + per_thread, n);
        if (start >= end) continue;
        threads[t] = std::thread([tensor, expert_idx, start, end]() {
            for (int i = start; i < end; i++)
                pagein_expert_mt(tensor, expert_idx[i]);
        });
    }
    for (int t = 0; t < nthreads; t++) {
        if (threads[t].joinable()) threads[t].join();
    }
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
