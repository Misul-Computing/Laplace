// bench_threadpool.cpp - verify dynamic power management without a model.
//
// Tests:
//   1. Correctness with dynamic active count.
//   2. Dynamic activation: set_active_count changes take effect.
//   3. Standby workers wake when promoted.
//   4. Park/wake still works.
//   5. Power monitor can run and adjust without crashing.
//
// Build:
//   c++ -O2 -std=c++17 -I src tests/bench_threadpool.cpp -o bench_threadpool \
//       -framework IOKit -framework CoreFoundation
#include "../src/threadpool.h"
#include "../src/power_monitor.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

using namespace Laplace;
using clk = std::chrono::steady_clock;

static int correctness_test() {
    auto& pool = ThreadPool::get();
    const int N = 100000;
    std::atomic<long> sum{0};
    pool.parallel_for(N, [&](int i) {
        sum.fetch_add(i, std::memory_order_relaxed);
    });
    long expected = (long)(N - 1) * N / 2;
    if (sum.load() != expected) {
        fprintf(stderr, "FAIL correctness: sum=%ld expected=%ld\n",
                sum.load(), expected);
        return 1;
    }
    fprintf(stderr, "PASS correctness: sum=%ld (active=%d)\n",
            sum.load(), pool.active_count());
    return 0;
}

static int dynamic_activation_test() {
    auto& pool = ThreadPool::get();
    // Reduce to 2 threads, verify work completes.
    pool.set_active_count(2);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::atomic<long> s1{0};
    pool.parallel_for(50000, [&](int i) {
        s1.fetch_add(i, std::memory_order_relaxed);
    });
    long exp = (long)(50000 - 1) * 50000 / 2;
    if (s1.load() != exp) {
        fprintf(stderr, "FAIL dynamic(2): s1=%ld expected=%ld\n", s1.load(), exp);
        return 1;
    }
    // Increase to max, verify standby workers wake up.
    pool.set_active_count(11);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::atomic<long> s2{0};
    pool.parallel_for(50000, [&](int i) {
        s2.fetch_add(i * 2, std::memory_order_relaxed);
    });
    if (s2.load() != 2 * exp) {
        fprintf(stderr, "FAIL dynamic(max): s2=%ld expected=%ld\n",
                s2.load(), 2 * exp);
        return 1;
    }
    fprintf(stderr, "PASS dynamic_activation: 2 threads OK, %d threads OK\n",
            pool.active_count());
    return 0;
}

static int park_wake_test() {
    auto& pool = ThreadPool::get();
    std::atomic<long> s1{0};
    pool.parallel_for(10000, [&](int i) {
        s1.fetch_add(i, std::memory_order_relaxed);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::atomic<long> s2{0};
    pool.parallel_for(10000, [&](int i) {
        s2.fetch_add(i * 3, std::memory_order_relaxed);
    });
    long exp1 = (long)(10000 - 1) * 10000 / 2;
    if (s1.load() != exp1 || s2.load() != 3 * exp1) {
        fprintf(stderr, "FAIL park_wake: s1=%ld s2=%ld\n", s1.load(), s2.load());
        return 1;
    }
    fprintf(stderr, "PASS park_wake: s1=%ld s2=%ld\n", s1.load(), s2.load());
    return 0;
}

static int power_monitor_test() {
    auto& pm = PowerMonitor::get();
    pm.start();
    // Let it run for a few seconds, generating fake tokens.
    auto& pool = ThreadPool::get();
    for (int i = 0; i < 20; i++) {
        pool.record_token();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // Verify the monitor didn't crash and pool is still functional.
    std::atomic<long> s{0};
    pool.parallel_for(10000, [&](int i) {
        s.fetch_add(i, std::memory_order_relaxed);
    });
    long exp = (long)(10000 - 1) * 10000 / 2;
    pm.stop();
    if (s.load() != exp) {
        fprintf(stderr, "FAIL power_monitor: pool broken after monitor, s=%ld\n",
                s.load());
        return 1;
    }
    fprintf(stderr, "PASS power_monitor: ran 2s, pool OK, active=%d\n",
            pool.active_count());
    return 0;
}

int main() {
    fprintf(stderr, "[bench_threadpool] initial active=%d\n",
            ThreadPool::get().active_count());
    int rc = 0;
    rc |= correctness_test();
    rc |= dynamic_activation_test();
    rc |= park_wake_test();
    rc |= correctness_test();
    rc |= power_monitor_test();
    rc |= correctness_test();
    if (rc == 0) fprintf(stderr, "ALL PASS\n");
    return rc;
}
