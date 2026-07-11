// power_monitor.h - runtime power management for Apple Silicon.
//
// A background thread (QOS_CLASS_BACKGROUND, runs on an E-core) polls
// power state, thermal level, and actual throughput every 3 seconds.
// It adjusts the thread pool's active thread count and core mode at
// runtime to minimize power draw on battery and maximize throughput on AC.
//
// Online performance probing: every few poll cycles, the monitor tries
// adjusting the thread count by +/-1 and measures the effect on tok/s.
// If throughput improves (or holds with fewer threads on battery), the
// change sticks. If it degrades, the monitor reverts. This adapts to the
// actual workload and conditions without offline training.
//
// Hysteresis: changes are only applied if the target differs from the
// current count by more than 1, or if thermal pressure is heavy+. This
// prevents oscillation from small fluctuations.
//
// The monitor thread costs near zero: it sleeps for 3 seconds at a time
// at BACKGROUND QoS and does a handful of syscalls per poll. It never
// touches the hot path.
#pragma once

#include "threadpool.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace Laplace {

class PowerMonitor {
public:
    static PowerMonitor& get() {
        static PowerMonitor pm;
        return pm;
    }

    void start() {
        if (running_.load(std::memory_order_relaxed)) return;
        running_.store(true, std::memory_order_relaxed);
        monitor_ = std::thread([this] { run(); });
    }

    void stop() {
        running_.store(false, std::memory_order_relaxed);
        if (monitor_.joinable()) monitor_.join();
    }

    ~PowerMonitor() { stop(); }

private:
    PowerMonitor() {
        debug_ = std::getenv("LAPLACE_DEBUG_POWER") != nullptr;
        topo_ = laplace_core_topology();
        max_threads_ = topo_.p_cores + topo_.e_cores;
        manual_threads_ = std::getenv("LAPLACE_THREADS") != nullptr;
        // If LAPLACE_ECORES is set, don't auto-switch mode.
        const char* ecore_env = std::getenv("LAPLACE_ECORES");
        manual_mode_ = ecore_env && ecore_env[0];
    }

    // Measure tok/s over the interval since last_time.
    double measure_tps(ThreadPool& pool, long& prev_tokens,
                       std::chrono::steady_clock::time_point& last_time) {
        long cur_tokens = pool.tokens_generated();
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_time).count();
        double tps = elapsed > 0.0
            ? (double)(cur_tokens - prev_tokens) / elapsed : 0.0;
        prev_tokens = cur_tokens;
        last_time = now;
        return tps;
    }

    void run() {
        // Monitor runs at BACKGROUND QoS so it never competes with workers.
        pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);

        auto& pool = ThreadPool::get();
        long prev_tokens = 0;
        auto last_time = std::chrono::steady_clock::now();

        // Probing state.
        bool probing = false;
        int probe_prev_count = 0;
        double probe_baseline_tps = 0;
        int probe_samples = 0;

        while (running_.load(std::memory_order_relaxed)) {
            // Sleep for poll_interval_ in small increments so stop() is
            // responsive. 100ms granularity is fine for a 3s poll.
            for (int i = 0; i < poll_interval_ * 10; i++) {
                if (!running_.load(std::memory_order_relaxed)) return;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (!running_.load(std::memory_order_relaxed)) return;

            // Poll system state.
            auto ps = laplace_power_state();
            int thermal = laplace_thermal_level();
            double tps = measure_tps(pool, prev_tokens, last_time);

            int cur_active = pool.active_count();
            CoreMode cur_mode = pool.core_mode();

            // Mode switch: Hybrid on battery, Performance on AC.
            // Skip if LAPLACE_ECORES is set (manual mode control).
            if (manual_mode_) {
                cur_mode = pool.core_mode();
            } else {
                CoreMode target_mode = (ps.on_battery && topo_.e_cores > 0)
                    ? CoreMode::Hybrid : CoreMode::Performance;
                if (target_mode != cur_mode) {
                    pool.set_core_mode(target_mode);
                    cur_mode = target_mode;
                    probing = false;
                    probe_cooldown_ = kProbeCooldown;
                }
            }

            // If probing, collect samples. Discard the first sample after
            // the thread count change (it includes transition overhead).
            if (probing) {
                probe_samples++;
                if (probe_samples < kProbeSamples) {
                    continue; // still warming up under new count
                }
                // Evaluate: compare tps under probe count to baseline.
                bool keep = ps.on_battery
                    ? tps >= probe_baseline_tps * 0.9
                    : tps > probe_baseline_tps * 1.02;
                if (debug_)
                    fprintf(stderr, "[power] probe %d->%d: %.1f->%.1f tps, %s\n",
                            probe_prev_count, pool.active_count(),
                            probe_baseline_tps, tps, keep ? "keep" : "revert");
                if (!keep) pool.set_active_count(probe_prev_count);
                probing = false;
                probe_cooldown_ = kProbeCooldown;
                continue;
            }

            // Apply policy target with hysteresis. Skip if user set
            // LAPLACE_THREADS (manual control) or if we're in prefill
            // (tps == 0 means no tokens generated yet).
            if (!manual_threads_ && tps > 0.0) {
                int target = ThreadPool::compute_target(
                    topo_, ps, thermal, cur_mode);

                if (std::abs(target - cur_active) > 1) {
                    pool.set_active_count(target);
                } else if (target != cur_active && thermal >= 2) {
                    pool.set_active_count(target);
                }
            }

            // Online probing: try +/-1 thread to find the real optimum.
            // Only when thermally stable, cooldown expired, not in
            // manual mode, and actually generating tokens (tps > 0).
            if (!manual_threads_ && thermal == 0 && tps > 0.0 &&
                probe_cooldown_ <= 0) {
                bool can_probe_up = !ps.on_battery
                    && cur_active < max_threads_;
                bool can_probe_down = ps.on_battery && cur_active > 2;
                if (can_probe_up || can_probe_down) {
                    int delta = ps.on_battery ? -1 : +1;
                    probing = true;
                    probe_prev_count = cur_active;
                    probe_baseline_tps = tps;
                    probe_samples = 0;
                    pool.set_active_count(cur_active + delta);
                    if (debug_)
                        fprintf(stderr, "[power] probing %+d thread (%d->%d)\n",
                                delta, cur_active, cur_active + delta);
                    continue;
                }
            }
            if (probe_cooldown_ > 0) probe_cooldown_--;

            if (debug_ && (ps.on_battery || thermal > 0 || probing))
                fprintf(stderr, "[power] battery=%d%% thermal=%d active=%d "
                        "mode=%s tps=%.1f\n",
                        ps.battery_pct, thermal, pool.active_count(),
                        cur_mode == CoreMode::Hybrid ? "hybrid" :
                        cur_mode == CoreMode::Ecore ? "ecore" : "perf",
                        tps);
        }
    }

    bool debug_ = false;
    bool manual_threads_ = false;
    bool manual_mode_ = false;
    CoreTopology topo_;
    int max_threads_ = 0;
    std::thread monitor_;
    std::atomic<bool> running_{false};

    static constexpr int poll_interval_ = 3;   // seconds between polls
    int probe_cooldown_ = 3;                   // polls to wait before probing
    static constexpr int kProbeCooldown = 3;   // reset value after a probe
    static constexpr int kProbeSamples = 2;    // samples to discard + measure
};

} // namespace Laplace
