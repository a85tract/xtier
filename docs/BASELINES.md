<!-- SPDX-License-Identifier: Apache-2.0 -->
# Baseline systems

How to set up the systems xTier is compared against. Each needs the kernel
built a particular way, or a sysctl set, or both.

All of these need the kernel built with the baseline patch:

```sh
sudo kernel/build.sh --stack=all
```

That single patch carries everything below. See `../kernel/README.md`.

## How to launch the workload

This is not the same for every system, and getting it wrong does not produce
an error.

| System | Launch |
|---|---|
| AutoNUMA, TPP, HybridTier, Memtis | `numactl --cpunodebind=0 <workload>` |
| xTier | `numactl --cpunodebind=0 <workload>`, and `--preferred=1` for a cold start on the slow tier |

**AutoNUMA, TPP and HybridTier must run with no explicit memory policy.** Both
`numactl --membind` and `numactl --preferred` install a policy whose flags lack
`MPOL_F_MOF`; `task_numa_work()` skips every such VMA, and the system then
migrates nothing at all while still reporting a plausible wall time. Only the
kernel's implicit per-node policy carries `MPOL_F_MOF`. The `--preferred=1`
line in the top-level README is the xTier configuration, and it silences these
baselines. xTier is unaffected. It migrates from kthread context and does not
consult mempolicy.

Check the counters before trusting any number: `numa_pages_migrated` and
`pgpromote_success` for AutoNUMA, `pgdemote_kswapd` for TPP, and
`htmm_nr_promoted` for Memtis. A system that moved no pages did not run.

## AutoNUMA

Mainline. Nothing to build.

```sh
echo 1 > /proc/sys/kernel/numa_balancing
echo 0 > /sys/kernel/mm/numa/demotion_enabled
```

The baseline patch also pins the scan period to 1000 ms, so it does not
oscillate during a run.

## TPP

Upstream since Linux 5.18, so no rebuild is needed for TPP itself.

```sh
echo 2 > /proc/sys/kernel/numa_balancing
echo 1 > /sys/kernel/mm/numa/demotion_enabled
```

Mode 2 is `NUMA_BALANCING_MEMORY_TIERING`, which is what enables promotion out
of a lower tier. The measurements in the paper were taken with mode 1. If you
are comparing against those numbers, use 1.

`vm.zone_reclaim_mode` should be 0, the default.

Under CXL emulation you do need the baseline patch, which makes
`node_is_toptier()` true only for node 0 and gives node 1 a slower abstract
distance. A machine emulating CXL with a second NUMA node has no HMAT from its
BIOS, so every node otherwise lands in the same tier and TPP never demotes.

To confirm it is running: with a workload larger than node 0,
`pgdemote_kswapd` in `/proc/vmstat` should be non-zero.

## HybridTier

The baseline patch adds a runtime toggle for kswapd shrink. Then:

```sh
echo 1 > /proc/sys/kernel/numa_balancing
echo 1 > /sys/kernel/mm/numa/demotion_enabled
echo 1 > /proc/sys/kernel/freqtier_disable_kswapd_shrink
```

The runtime itself is a userspace LD_PRELOAD library belonging to its authors,
and is not redistributed here. You can obtain it from
<https://github.com/kevins981/hybridtier-asplos25-artifact>.

## Memtis

`kernel/build.sh --stack=all` enables `CONFIG_HTMM`. Then disable transparent
huge pages and set the tunables:

```sh
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag

echo 0        > /sys/kernel/mm/htmm/htmm_thres_hot
echo disabled > /sys/kernel/mm/htmm/htmm_skip_cooling
echo 1        > /sys/kernel/mm/htmm/htmm_nowarm
echo 1000     > /sys/kernel/mm/htmm/htmm_cooling_period
```

Create a cgroup v2 group for the workload:

```sh
mkdir /sys/fs/cgroup/memtis_run
echo max                    > /sys/fs/cgroup/memtis_run/memory.max
echo enabled                > /sys/fs/cgroup/memtis_run/memory.htmm_enabled
echo "0 <node-0-pages>"     > /sys/fs/cgroup/memtis_run/memory.htmm_max_at_node
```

Launch through `htmm_launcher`, which starts access tracking for the workload
PID:

```sh
cc -O2 -o bench/htmm_launcher bench/htmm_launcher.c
cgexec -g memory:memtis_run numactl --cpunodebind=0 \
  bench/htmm_launcher 0 -- <workload>
```

`cgexec` comes from the `cgroup-tools` package.

The launcher is required. Writing `memory.htmm_enabled` alone is not enough --
tracking starts on the `htmm_start` syscall. Check that `htmm_nr_promoted` and
`htmm_nr_demoted` are non-zero before using a Memtis result.
