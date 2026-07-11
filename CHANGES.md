# Changes

## Dynamic power management (2026-06-30)

A runtime power management system that automatically adjusts thread count,
core allocation, and QoS class based on real-time system conditions and
measured throughput. No user configuration needed. Apple Silicon only.

### Architecture

Three components:

1. **ThreadPool** (`src/threadpool.h`): all possible workers (P+E core
   count) are created at startup. A dynamic `active_count_` controls how
   many are working at any moment. Inactive workers park on a separate
   futex (`standby_epoch_`) and consume zero CPU. Active workers can be
   confined to E-cores (`QOS_CLASS_BACKGROUND`) or run at default QoS
   (mixed P+E), switched at runtime via `core_mode_`.

2. **PowerMonitor** (`src/power_monitor.h`): a background thread at
   `QOS_CLASS_BACKGROUND` (runs on an E-core, near zero overhead) that
   polls system state every 3 seconds and adjusts the pool. Reads power
   source, battery level, thermal pressure, and actual throughput.

3. **Throughput feedback**: the engine calls `ThreadPool::record_token()`
   per generated token. The power monitor reads the counter to measure
   actual tok/s and decide whether adjustments help or hurt.

### Policy engine

The monitor computes a target thread count from system state:

**Battery (E-core mode):**
- Workers on E-cores, main thread on a P-core.
- E-core count scaled by battery level: full at >50%, half at 20-50%,
  quarter at <20%.
- Thermal backoff: shed E-cores as thermal rises (moderate -25%, heavy
  -50%, critical -1 total).
- Minimum 1 E-core + 1 P-core (main thread).

**AC (performance mode):**
- All P-cores + ~30% of E-cores (capped to avoid thermal throttling).
- Thermal backoff same as above.

### Online performance probing

Every few poll cycles (with cooldown), the monitor tries adjusting the
thread count by +/-1 and measures the effect on tok/s over 2 samples:

- **Battery**: tries removing a thread. If throughput stays above 90%
  of before, keeps the lower count (saves power). Otherwise reverts.
- **AC**: tries adding a thread. If throughput improves by >2%, keeps
  it. Otherwise reverts.

This adapts to the actual workload. Decode is memory-bound, so the
optimal thread count varies with model size, context length, and KV
cache mode. The probe finds it automatically.

### Hysteresis

Policy target changes are only applied if they differ from the current
count by more than 1, or if thermal pressure is heavy+. This prevents
oscillation from small fluctuations.

### Spin-then-park idle waiting

Workers spin briefly (4096 iterations, ~5us) for back-to-back matmuls
that need zero latency, then park via `os_sync_wait_on_address` (Apple's
futex, macOS 14.4+). Idle power drops to near zero during attention,
RMSNorm, and sampling gaps between matmuls.

### Auto-detected core topology

P-core and E-core counts via `sysctlbyname`:
- `hw.perflevel0.physicalcpu` = P-cores
- `hw.perflevel1.physicalcpu` = E-cores

No hardcoded core counts. Falls back to `hardware_concurrency()` if
perflevel sysctls are unavailable.

### Thermal monitoring

Thermal pressure via Darwin notification
`com.apple.system.thermalpressurelevel` (pure C, no Obj-C++):
0=Nominal, 1=Moderate, 2=Heavy, 3=Trapping, 4=Sleeping.

### Overrides

- `LAPLACE_THREADS=N`: force thread count (disables auto-adjust)
- `LAPLACE_ECORES=1/0`: force E-core mode on/off
- `LAPLACE_DEBUG_THREADS=1`: print topology, thread count, mode at startup
- `LAPLACE_DEBUG_POWER=1`: print power monitor decisions (probes, adjustments)

### What was dropped

x86 and non-M-series support dropped to simplify the power management
code. Will be picked up after 1.0.
