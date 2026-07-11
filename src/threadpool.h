// threadpool.h - spin-then-park thread pool with dynamic power management.
//
// Apple Silicon only. Workers spin briefly for back-to-back matmuls (zero
// latency), then park via os_sync_wait_on_address (Apple's futex) during
// gaps. Idle power drops to near zero between matmuls.
//
// Dynamic activation: all possible workers are created at startup. The
// power monitor adjusts how many are active at runtime via active_count_.
// Inactive workers park on standby_epoch_ and consume zero CPU. Active
// workers can be confined to E-cores (QOS_CLASS_BACKGROUND) or run on
// mixed P+E cores, switched at runtime via core_mode_.
//
// The barrier is a flat barrier: each worker has its own done-flag on a
// separate cache line (alignas(64)), so there is zero atomic RMW contention.
// The main thread (tid=0) participates in the work, then polls each active
// worker's flag.
//
// Usage:
//   Laplace::ThreadPool::get().parallel_for(N, [&](int j) { ... });
#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>
#include <type_traits>

#include <os/os_sync_wait_on_address.h>
#include <IOKit/ps/IOPowerSources.h>
#include <IOKit/ps/IOPSKeys.h>
#include <CoreFoundation/CoreFoundation.h>
#include <pthread/qos.h>
#include <sys/sysctl.h>

#define LAPLACE_PAUSE() asm volatile("yield")

namespace Laplace {

// Power source state, queried from IOKit.
struct PowerState {
    bool on_battery;
    int battery_pct; // 0-100, 100 if on AC
};

static inline PowerState laplace_power_state() {
    PowerState ps{false, 100};
    CFTypeRef snap = IOPSCopyPowerSourcesInfo();
    if (!snap) return ps;
    CFStringRef src = IOPSGetProvidingPowerSourceType(snap);
    ps.on_battery = src && CFStringCompare(src, CFSTR(kIOPMBatteryPowerKey), 0)
                          == kCFCompareEqualTo;
    if (ps.on_battery) {
        CFArrayRef list = IOPSCopyPowerSourcesList(snap);
        if (list && CFArrayGetCount(list) > 0) {
            CFDictionaryRef desc = (CFDictionaryRef)IOPSGetPowerSourceDescription(
                snap, CFArrayGetValueAtIndex(list, 0));
            if (desc) {
                int cur = 0, max = 100;
                CFNumberGetValue((CFNumberRef)CFDictionaryGetValue(desc,
                    CFSTR(kIOPSCurrentCapacityKey)), kCFNumberSInt32Type, &cur);
                CFNumberGetValue((CFNumberRef)CFDictionaryGetValue(desc,
                    CFSTR(kIOPSMaxCapacityKey)), kCFNumberSInt32Type, &max);
                ps.battery_pct = max > 0 ? (cur * 100) / max : 100;
            }
        }
        if (list) CFRelease(list);
    }
    CFRelease(snap);
    return ps;
}

// Thermal pressure level via Darwin notification (pure C, no Obj-C++).
// 0=Nominal, 1=Moderate, 2=Heavy, 3=Trapping, 4=Sleeping.
#include <notify.h>
static inline int laplace_thermal_level() {
    int token = 0;
    if (notify_register_check("com.apple.system.thermalpressurelevel", &token) != 0)
        return 0;
    uint64_t state = 0;
    notify_get_state(token, &state);
    notify_cancel(token);
    return (int)state;
}

// P/E core topology via sysctl. Apple Silicon: perflevel0=P, perflevel1=E.
struct CoreTopology { int p_cores; int e_cores; };
static inline CoreTopology laplace_core_topology() {
    int p = 0, e = 0;
    size_t sz = sizeof(int);
    sysctlbyname("hw.perflevel0.physicalcpu", &p, &sz, nullptr, 0);
    sysctlbyname("hw.perflevel1.physicalcpu", &e, &sz, nullptr, 0);
    if (p == 0) {
        unsigned hw = std::thread::hardware_concurrency();
        p = hw > 0 ? (int)hw : 4;
    }
    return {p, e};
}

// Core mode: which cores workers run on and at what QoS.
enum class CoreMode : int {
    Performance = 0, // mixed P+E, default QoS
    Ecore = 1,       // E-cores only, QOS_CLASS_BACKGROUND
    Hybrid = 2,      // P-cores (default QoS) + E-cores (BACKGROUND)
};

class ThreadPool {
public:
    static ThreadPool& get() {
        static ThreadPool pool;
        return pool;
    }

    // Initial thread count based on power state at startup. The power
    // monitor adjusts this at runtime via set_active_count().
    static int initial_active_count() {
        const char* env = std::getenv("LAPLACE_THREADS");
        if (env && env[0]) {
            int n = std::atoi(env);
            if (n > 0) return n;
        }
        auto topo = laplace_core_topology();
        auto ps = laplace_power_state();
        const char* ecore_env = std::getenv("LAPLACE_ECORES");
        CoreMode mode;
        if (ecore_env && ecore_env[0])
            mode = std::atoi(ecore_env) != 0 ? CoreMode::Ecore : CoreMode::Performance;
        else
            mode = (ps.on_battery && topo.e_cores > 0) ? CoreMode::Hybrid : CoreMode::Performance;
        return compute_target(topo, ps, 0, mode);
    }

    // Policy: compute target active thread count from system state.
    // Called by the power monitor every few seconds.
    static int compute_target(CoreTopology topo, PowerState ps,
                              int thermal, CoreMode mode) {
        if (topo.e_cores <= 0) return topo.p_cores;

        if (mode == CoreMode::Ecore) {
            // E-core mode: 1 P-core (main) + E-cores scaled by battery
            // and thermal. Workers on E-cores only.
            int e = topo.e_cores;
            if (ps.battery_pct < 20)      e = e / 4;
            else if (ps.battery_pct < 50) e = e / 2;
            else                          e = (e * 3) / 4;
            if (thermal >= 3)      e = 1;
            else if (thermal == 2) e = e / 2;
            else if (thermal == 1) e = (e * 3) / 4;
            if (e < 1) e = 1;
            return 1 + e;
        }

        if (mode == CoreMode::Hybrid) {
            // Hybrid: 1 P-core (main) + some P-core workers + E-cores.
            // Use 2-3 P-core workers for compute-bound matmuls, rest E.
            // Fewer P-core workers than performance mode to save power.
            int p_workers = std::min(2, topo.p_cores - 1);
            int e = topo.e_cores;
            if (ps.battery_pct < 20)      e = e / 4;
            else if (ps.battery_pct < 50) e = e / 2;
            else                          e = (e * 3) / 4;
            if (thermal >= 3)      e = 0;
            else if (thermal == 2) e = e / 2;
            else if (thermal == 1) e = (e * 3) / 4;
            if (e < 1) e = 1;
            return 1 + p_workers + e;
        }

        // Performance: P-cores only for memory-bound decode. Adding E-cores
        // increases bandwidth pressure without improving throughput, and
        // E-cores are slower so they become the bottleneck in flat barriers.
        // Research on M3 Max shows >P-core count hurts due to L2 contention.
        int e_used = 0;
        if (thermal >= 3)      e_used = 0;
        else if (thermal == 2) e_used = e_used / 2;
        else if (thermal == 1) e_used = (e_used * 3) / 4;
        return topo.p_cores + e_used;
    }

    // Called by the power monitor to adjust active thread count at runtime.
    // Never demotes workers during an in-progress dispatch to avoid
    // demoting workers that the main thread is waiting on.
    void set_active_count(int n) {
        n = std::min(n, max_threads_);
        n = std::max(n, 1);
        int cur = active_count_.load(std::memory_order_relaxed);
        // Don't demote during a dispatch. The main thread is waiting
        // for done flags from all active workers. Promotion is fine.
        if (n < cur && dispatch_in_progress_.load(std::memory_order_acquire))
            return;
        if (n == cur) return;
        active_count_.store(n, std::memory_order_release);
        // Wake standby workers so they can check if they're now active.
        standby_epoch_.fetch_add(1, std::memory_order_release);
        laplace_futex_wake(&standby_epoch_);
        if (debug_)
            fprintf(stderr, "[threadpool] active_count -> %d\n", n);
    }

    // Called by the power monitor to switch core mode at runtime.
    void set_core_mode(CoreMode mode) {
        core_mode_.store((int)mode, std::memory_order_release);
        if (debug_)
            fprintf(stderr, "[threadpool] core_mode -> %s\n",
                    mode == CoreMode::Hybrid ? "hybrid" :
                    mode == CoreMode::Ecore ? "ecore" : "performance");
    }

    int active_count() const {
        return active_count_.load(std::memory_order_relaxed);
    }

    CoreMode core_mode() const {
        return (CoreMode)core_mode_.load(std::memory_order_relaxed);
    }

    // Throughput feedback: the engine calls this per generated token.
    // The power monitor reads tokens_generated_ to measure actual tok/s
    // and decide whether to adjust thread count.
    void record_token() {
        tokens_generated_.fetch_add(1, std::memory_order_relaxed);
    }
    long tokens_generated() const {
        return tokens_generated_.load(std::memory_order_relaxed);
    }

    template<typename F>
    void parallel_for(int n, F&& body) {
        if (n <= 0) return;
        int active = active_count_.load(std::memory_order_acquire);
        if (active <= 1) {
            for (int i = 0; i < n; i++) body(i);
            return;
        }
        using Fd = std::decay_t<F>;
        fn_.store([](void* ctx, int i) { (*static_cast<Fd*>(ctx))(i); },
                  std::memory_order_relaxed);
        ctx_.store(&body, std::memory_order_relaxed);
        n_items_.store(n, std::memory_order_relaxed);
        active_for_dispatch_.store(active, std::memory_order_release);
        dispatch_in_progress_.store(true, std::memory_order_release);
        // Clear done flags before dispatch so stale flags from any
        // spurious worker runs don't cause us to skip waiting.
        for (int i = 1; i < active; i++)
            done_[i].flag.store(false, std::memory_order_relaxed);
        // Release ensures workers see fn_/ctx_/n_items_ before the epoch bump.
        epoch_.fetch_add(1, std::memory_order_release);
        laplace_futex_wake(&epoch_);
        // Main thread (tid=0) does its share, then waits for active workers.
        run(0);
        for (int i = 1; i < active; i++) {
            while (!done_[i].flag.load(std::memory_order_acquire))
                LAPLACE_PAUSE();
        }
        // Dispatch complete. Allow demotion again.
        dispatch_in_progress_.store(false, std::memory_order_release);
    }

    ~ThreadPool() {
        stop_.store(true, std::memory_order_release);
        epoch_.fetch_add(1, std::memory_order_release);
        laplace_futex_wake(&epoch_);
        standby_epoch_.fetch_add(1, std::memory_order_release);
        laplace_futex_wake(&standby_epoch_);
        for (auto& t : workers_) t.join();
    }

private:
    using fn_t = void(*)(void*, int);

    struct alignas(64) DoneFlag {
        std::atomic<bool> flag{false};
    };

    ThreadPool() {
        debug_ = std::getenv("LAPLACE_DEBUG_THREADS") != nullptr;
        auto topo = laplace_core_topology();
        auto ps = laplace_power_state();
        max_threads_ = topo.p_cores + topo.e_cores;
        p_cores_ = topo.p_cores;
        done_ = std::make_unique<DoneFlag[]>(max_threads_);
        active_count_.store(initial_active_count(), std::memory_order_relaxed);
        // E-core mode: auto from battery, or forced via LAPLACE_ECORES.
        const char* ecore_env = std::getenv("LAPLACE_ECORES");
        CoreMode init_mode;
        if (ecore_env && ecore_env[0]) {
            // Forced mode: 1=Ecore, 0=Performance.
            init_mode = std::atoi(ecore_env) != 0
                ? CoreMode::Ecore : CoreMode::Performance;
        } else {
            // Auto: Hybrid on battery, Performance on AC.
            init_mode = (ps.on_battery && topo.e_cores > 0)
                ? CoreMode::Hybrid : CoreMode::Performance;
        }
        core_mode_.store((int)init_mode, std::memory_order_relaxed);
        if (debug_)
            fprintf(stderr, "[threadpool] %dP+%dE, initial active=%d, mode=%s, "
                    "battery=%d(%d%%)\n", topo.p_cores, topo.e_cores,
                    active_count_.load(),
                    init_mode == CoreMode::Hybrid ? "hybrid" :
                    init_mode == CoreMode::Ecore ? "ecore" : "performance",
                    (int)ps.on_battery, ps.battery_pct);
        if (max_threads_ <= 1) return;
        workers_.reserve(max_threads_ - 1);
        for (int i = 1; i < max_threads_; i++)
            workers_.emplace_back([this, i] { worker(i); });
    }

    void run(int tid) {
        int active = active_for_dispatch_.load(std::memory_order_acquire);
        int n = n_items_.load(std::memory_order_acquire);
        int chunk = (n + active - 1) / active;
        int start = tid * chunk;
        int end = start + chunk;
        if (end > n) end = n;
        fn_t fn = fn_.load(std::memory_order_acquire);
        void* ctx = ctx_.load(std::memory_order_acquire);
        for (int i = start; i < end; i++) fn(ctx, i);
    }

    void worker(int tid) {
        uint32_t local_epoch = 0;
        uint32_t local_standby = 0;
        CoreMode last_mode = core_mode();
        apply_qos(last_mode, tid);

        while (true) {
            // Check if this worker is active or on standby.
            int active = active_count_.load(std::memory_order_acquire);
            if (tid >= active) {
                // Standby: park until promoted (standby_epoch_ changes).
                // Do NOT check epoch_ here. Workers on standby were not
                // invited to the current dispatch. The main thread only
                // waits for done flags from workers < active_for_dispatch_.
                // The race where a worker is demoted while a dispatch
                // starts is handled by the demotion check in the active
                // path: the worker is still in the active path when the
                // dispatch starts, sees epoch_ advance, and runs.
                while (standby_epoch_.load(std::memory_order_relaxed)
                       == local_standby) {
                    if (stop_.load(std::memory_order_relaxed)) return;
                    int spins = 0;
                    while (standby_epoch_.load(std::memory_order_relaxed)
                           == local_standby) {
                        if (stop_.load(std::memory_order_relaxed)) return;
                        if (++spins < kSpinBeforePark) {
                            LAPLACE_PAUSE();
                        } else {
                            uint32_t cur = standby_epoch_.load(
                                std::memory_order_acquire);
                            if (cur != local_standby) break;
                            if (stop_.load(std::memory_order_relaxed)) return;
                            laplace_futex_park(&standby_epoch_, local_standby);
                            spins = 0;
                        }
                    }
                    break;
                }
                local_standby = standby_epoch_.load(std::memory_order_acquire);
                // Do NOT sync local_epoch here. Keep the old value so
                // that when we enter the active path, epoch_ !=
                // local_epoch (dispatches happened while on standby).
                // The active path will check dispatch_in_progress_ and
                // active_for_dispatch_ to decide whether to run.
                // Re-check QoS in case mode changed while on standby.
                CoreMode mode = core_mode();
                if (mode != last_mode) { apply_qos(mode, tid); last_mode = mode; }
                continue;
            }

            // Active: spin/park for work on epoch_.
            CoreMode mode = core_mode();
            if (mode != last_mode) { apply_qos(mode, tid); last_mode = mode; }

            int spins = 0;
            bool demoted = false;
            while (epoch_.load(std::memory_order_relaxed) == local_epoch) {
                if (stop_.load(std::memory_order_relaxed)) return;
                if (active_count_.load(std::memory_order_relaxed) <= tid) {
                    demoted = true;
                    break;
                }
                if (++spins < kSpinBeforePark) {
                    LAPLACE_PAUSE();
                } else {
                    uint32_t cur = epoch_.load(std::memory_order_acquire);
                    if (cur != local_epoch) break;
                    if (stop_.load(std::memory_order_relaxed)) return;
                    if (active_count_.load(std::memory_order_relaxed) <= tid) {
                        demoted = true;
                        break;
                    }
                    laplace_futex_park(&epoch_, local_epoch);
                    spins = 0;
                }
            }
            if (demoted) {
                uint32_t cur = epoch_.load(std::memory_order_acquire);
                if (cur == local_epoch)
                    continue;
            }
            // Epoch advanced. Sync local_epoch to the current value.
            local_epoch = epoch_.load(std::memory_order_acquire);
            if (stop_.load(std::memory_order_relaxed)) return;
            // Only run if this dispatch is still in progress and
            // this worker is part of it. A dispatch that already
            // completed (main thread read all done flags and moved
            // on) is not safe to run - fn_/ctx_/n_items_ may have
            // been overwritten by the next dispatch.
            if (!dispatch_in_progress_.load(std::memory_order_acquire)) {
                continue; // dispatch already completed, skip
            }
            int dispatch_active = active_for_dispatch_.load(std::memory_order_acquire);
            if (tid >= dispatch_active) {
                continue; // not part of this dispatch
            }
            run(tid);
            done_[tid].flag.store(true, std::memory_order_release);
        }
    }

    void apply_qos(CoreMode mode, int tid) {
        if (mode == CoreMode::Ecore) {
            pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
        } else if (mode == CoreMode::Hybrid) {
            // In hybrid mode, workers with tid < p_cores_ run on P-cores
            // (default QoS), the rest on E-cores (BACKGROUND).
            if (tid < p_cores_)
                pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
            else
                pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
        } else {
            // Performance: all workers on default QoS (P+E mix).
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        }
    }

    static void laplace_futex_wake(std::atomic<uint32_t>* addr) {
        os_sync_wake_by_address_all(addr, sizeof(uint32_t),
                                    OS_SYNC_WAKE_BY_ADDRESS_NONE);
    }
    static void laplace_futex_park(std::atomic<uint32_t>* addr,
                                   uint32_t expected) {
        os_sync_wait_on_address(addr, expected, sizeof(uint32_t),
                                OS_SYNC_WAIT_ON_ADDRESS_NONE);
    }

    bool debug_ = false;
    int max_threads_ = 0;
    int p_cores_ = 0;  // P-core count, for hybrid mode split
    std::vector<std::thread> workers_;
    std::unique_ptr<DoneFlag[]> done_;

    // Work descriptor (set by caller before signaling epoch_).
    // Atomic to prevent compiler reordering across the epoch_ release.
    std::atomic<fn_t> fn_{nullptr};
    std::atomic<void*> ctx_{nullptr};
    std::atomic<int> n_items_{0};
    std::atomic<int> active_for_dispatch_{1}; // snapshot during dispatch

    std::atomic<uint32_t> epoch_{0};
    std::atomic<uint32_t> standby_epoch_{0};
    std::atomic<int> active_count_{1};
    std::atomic<int> core_mode_{0}; // CoreMode
    std::atomic<bool> stop_{false};
    std::atomic<bool> dispatch_in_progress_{false};
    std::atomic<long> tokens_generated_{0};

    static constexpr int kSpinBeforePark = 4096;
};

} // namespace Laplace
