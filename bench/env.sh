#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Shared paths, and the system preparation every xTier run needs.
#
#     source bench/env.sh
#     sudo -E bash -c 'source bench/env.sh; xtier_sysprep'
#
# Nothing here is specific to our machines.

# Where this checkout lives.
export XTIER_ROOT="${XTIER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Where collected features and run output are written.
export XTIER_RESULTS="${XTIER_RESULTS:-/var/tmp/xtier-results}"

# Workload binaries and generated datasets. Wants tens of GB free.
export WORKLOAD_DIR="${WORKLOAD_DIR:-/var/tmp/xtier-workloads}"

# Applies the settings documented in docs/SETUP.md section 3, which explains
# why each one matters. The short version: without the two perf sysctls the
# kernel throttles PEBS delivery and xTier runs nearly blind.
xtier_sysprep() {
    [ "$(id -u)" -eq 0 ] || { echo "xtier_sysprep: run as root" >&2; return 1; }

    modprobe msr 2>/dev/null || true

    # Stop the kernel throttling PEBS delivery. The percentage has to be
    # non-zero to set the rate, and zero to stop the throttling.
    echo 25     > /proc/sys/kernel/perf_cpu_time_max_percent
    echo 100000 > /proc/sys/kernel/perf_event_max_sample_rate
    echo 0      > /proc/sys/kernel/perf_cpu_time_max_percent
    echo 0      > /proc/sys/kernel/nmi_watchdog
    echo 0      > /proc/sys/kernel/perf_event_paranoid

    # The page-state map is keyed by virtual address.
    echo 0      > /proc/sys/kernel/randomize_va_space

    # AutoNUMA would migrate pages underneath xTier.
    echo 0      > /proc/sys/kernel/numa_balancing
    echo 0      > /sys/kernel/mm/numa/demotion_enabled 2>/dev/null || true

    # Keep clocks from drifting between trials.
    echo 1      > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance > "$g" 2>/dev/null || true
    done

    # Disable all four hardware prefetchers, so PEBS samples reflect demand
    # accesses rather than speculative ones.
    wrmsr -a 0x1A4 0xF 2>/dev/null \
        || echo "xtier_sysprep: wrmsr failed -- prefetchers still enabled" >&2

    sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    echo "xtier_sysprep: done"
}
