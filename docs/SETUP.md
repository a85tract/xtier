<!-- SPDX-License-Identifier: Apache-2.0 -->
# Setup

This document describes how to install and run xTier on a two-socket machine.
It covers the kernel, the build, the system configuration required before a
measurement, and a set of operational notes.

## 0. Dependencies

```sh
sudo ./setup.sh
```

Installs the distribution packages, builds `bpftool`, and installs the Python
packages. Run it before anything else.

`bpftool` is built from the kernel source rather than installed from a package
because the packaged one is tied to the distribution kernel. On Ubuntu 22.04
that is v5.15, and it fails the `vmlinux.h` step against a 6.11 kernel with
`failed to load BTF from /sys/kernel/btf/vmlinux: Invalid argument`.

## 1. Kernel

xTier requires a patched Linux 6.11. The script `kernel/build.sh` performs the
full sequence. It fetches the source, applies the patch, configures the tree,
builds it, installs it, and sets the new boot entry.

```sh
sudo kernel/build.sh                 # xTier only
sudo kernel/build.sh --stack=all     # also the baseline systems
sudo kernel/build.sh --slim          # only modules this machine has loaded
```

The configuration is seeded from the running kernel. On a distribution install
that means building several thousand modules, which requires roughly 30 GB of
disk and takes 30 to 60 minutes. The `--slim` option reduces this to about
12 GB and 15 minutes by building only the modules currently loaded, which is
appropriate on the machine you intend to boot. Re-running the script resumes an
interrupted build rather than starting again.

After rebooting, confirm the result.

```sh
uname -r                                  # 6.11.0-xtier
grep xtier_migrate_range /proc/kallsyms   # one line
ls /sys/kernel/btf/vmlinux                # the BPF program is built from this
```

### What the patch does

The file `src/xtier_executor.c` declares the following function.

```c
extern long xtier_migrate_range(struct mm_struct *mm, unsigned long start,
                                unsigned long end, int target_node);
```

The patch adds that function to `mm/mempolicy.c` and exports it with
`EXPORT_SYMBOL_GPL`. The function queues the pages in the given range and
passes them to `migrate_pages()` for the target node.

The module requires this because no suitable entry point is exported to
modules. The migration functions in `mm/` are internal to the kernel, and
`move_pages(2)` is a system call that acts on the calling task. The executor
runs in kthread context and moves pages belonging to another process. Without
the patch the module compiles, but `insmod` fails on an unresolved symbol.

The export is GPL-only, so any module that links against it must also be GPL.
This is why `src/` is licensed GPL-2.0.

Two configuration options are easy to overlook and both are required.
`CONFIG_DEBUG_INFO_BTF` produces the BTF from which `vmlinux.h` is generated,
and it depends on `pahole` from the `dwarves` package. Without it the kernel
boots normally but the BPF program cannot be compiled. `CONFIG_MEMORY_HOTREMOVE`
allows `bench/set_cxl_ratio.sh` to offline node 0 blocks. The build script sets
both.

The file `kernel/README.md` describes the baseline patches and the manual build
procedure.

## 2. Build

```sh
make                 # module, BPF object, loader
make help            # targets and variables
```

The loader compiles a model in, and collecting the features to train one needs
the loader. `make bootstrap-model` writes placeholder weights so the first
build succeeds. Use that build for `collect` mode only.

```sh
make bootstrap-model
make MODEL_HDR=$PWD/ml/models/mlp_q8_bootstrap.h
```

Two variables are commonly overridden.

```sh
make LIBBPF_SRC=                              # link the distribution libbpf
make MODEL_HDR=$PWD/ml/models/<model>.h       # select the model to compile in
```

The loader compiles the model weights into the binary and writes them to the
BPF maps at startup. Changing the model therefore requires rebuilding the
loader rather than restarting it. No model is distributed with this repository.
Train one first, as described in `../ml/README.md`.

## 3. System configuration

Apply the following settings before every measurement. Section 5 explains the
less obvious ones.

```sh
modprobe msr
echo 25     > /proc/sys/kernel/perf_cpu_time_max_percent
echo 100000 > /proc/sys/kernel/perf_event_max_sample_rate
echo 0      > /proc/sys/kernel/perf_cpu_time_max_percent
echo 0      > /proc/sys/kernel/nmi_watchdog
echo 0      > /proc/sys/kernel/perf_event_paranoid
echo 0      > /proc/sys/kernel/randomize_va_space
echo 0      > /proc/sys/kernel/numa_balancing
echo 1      > /sys/devices/system/cpu/intel_pstate/no_turbo
echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
wrmsr -a 0x1A4 0xF            # disable all four hardware prefetchers
echo 3 > /proc/sys/vm/drop_caches
```

The two `perf` settings prevent the kernel from throttling PEBS delivery, as
described in section 5, and must be applied in the order shown: the kernel
refuses to change `perf_event_max_sample_rate` while
`perf_cpu_time_max_percent` is 0. Setting `randomize_va_space` to 0 is required because
the page state map is keyed by virtual address. Setting `numa_balancing` to 0
prevents AutoNUMA from migrating pages underneath xTier, which would otherwise
make the measurement reflect both systems at once. Disabling turbo and fixing
the governor keep clock frequencies stable across trials.

The function `xtier_sysprep` in `bench/env.sh` applies all of these.

## 4. Setting the DRAM to CXL ratio

Node 0 is the fast tier and node 1 represents CXL-attached memory. xTier only
makes a placement decision when the working set does not fit in node 0. On a
machine with equally sized nodes it always fits, so node 0 must be reduced.

```sh
sudo bench/set_cxl_ratio.sh 1:10     # roughly 8 GB fast tier, 80 GB slow tier
sudo bench/set_cxl_ratio.sh reset
```

The script offlines node 0 memory blocks at runtime. Confirm the result with
`numactl -H`.

Two limits apply. A block held by unmovable kernel allocations cannot be
offlined, so a tight ratio may fall short of the requested value. The script
reports the ratio it achieved. Blocks are also offlined whole, at the
granularity given in `/sys/devices/system/memory/block_size_bytes`, which is
usually 128 MB.

If a ratio cannot be reached by offlining, the kernel `memmap=` parameter hides
a range from node 0 at boot instead. For example, `memmap=72G$8G` leaves a
72 GB node 0 on an 80 GB node. The `$` character **must be escaped as `\$` in
`/boot/grub/grub.cfg`**. If it is not, GRUB expands it as a variable and the
resulting parameter is `memmap=72GG`. This method also requires a reboot for
each ratio. Verify the result with `numactl -H` after booting.

## 5. Operational notes

### PEBS throttling

The kernel throttles a perf event that exceeds `perf_cpu_time_max_percent` or
`perf_event_max_sample_rate`, and the unthrottle path does not always recover
the event. This does not produce an error. Instead, sample delivery falls to
10 or 20 samples per tick rather than several thousand, so xTier appears to run
normally while making almost no decisions.

Setting both sysctls as shown in section 3 prevents this. The loader also
includes a watchdog that disables and re-enables the event to restore delivery
if the sample count collapses during a run. If the profiler reports a low
`pebs_seen` value, check those two settings first.

### PMU counter contention

The processor provides four general-purpose counters. The PEBS-eligible event
families, namely `mem_load_uops_retired.*`, `mem_uops_retired.*`, and
`mem_load_uops_l3_miss_retired.*`, compete for those counters with the sampling
events that xTier opens. When both are active the multiplexer favours whichever
was opened first, which is normally `perf stat`. In that case xTier can receive
no samples at all.

Do not run `perf stat` with PEBS-eligible events against a workload that xTier
is managing. Collect those counters from a separate run.

### Hardware prefetchers

A prefetcher fetches data ahead of the demand load. On a predictable access
pattern the cache line is already resident when the load issues, so the
processor does not stall. The latency that page placement is intended to avoid
has already been absorbed, and two different placements produce the same
result.

The command `wrmsr -a 0x1A4 0xF` disables all four prefetchers. Doing so also
ensures that PEBS samples reflect demand accesses rather than speculative ones,
which matches the data the model is trained on.

### PEBS attaches to a thread rather than a process

The call `perf_event_open(pid=X, cpu=-1)` attaches to a single task rather than
a thread group. Attaching it to the main thread of a multithreaded workload
samples only that thread, which typically performs very little memory access,
while the compute threads are not observed at all.

The function `open_perf_for_threads()` addresses this by enumerating
`/proc/PID/task/*` and opening one event for each thread. This matters if you
modify the attach path, because an incorrect version produces a run that looks
plausible but contains a small fraction of the expected samples.

### Migration counters

The function `xtier_migrate_range` runs in kthread context and does not
increment `pgmigrate_success` or `numa_pages_migrated`. Both counters read zero
regardless of how many pages xTier has moved. Migration counts should be read
from the per-epoch output that the profiler writes.

### Transparent huge pages

xTier promotes pages at 4 KB granularity. When `THP` is set to `always`,
promoting scattered pages out of a 2 MB huge page splits that huge page. The
kernel then runs compaction to rebuild it, which migrates pages within node 0
and records them in `pgmigrate_success` alongside genuine tier traffic. If you
intend to compare migration volume, either set `THP` to `never` or use a
placement metric that compaction cannot affect.

## 6. Running

```sh
sudo make restart                                         # load module, clear stale maps
sudo src/page_profiler_user <pid> <freq> <epoch_ms> run    # attach to a workload
```

The available modes are `monitor`, `collect`, `evaluate`, and `run`. The
`collect` mode writes the feature data used for training.

```sh
make show-stats      # counters
make teardown        # unload and clear pinned maps
```

The intended configuration allocates memory on the slow tier and allows xTier
to promote pages upward.

```sh
numactl --cpunodebind=0 --preferred=1 <workload>
```

The loader reads the following environment variables.

| Variable | Meaning |
|---|---|
| `XTIER_DRAM_BUDGET_MB` | Node 0 budget for the top-K admission gate. Approximately 80 percent of node 0 size |
| `XTIER_PEBS_SAMPLE_FREQ` | Sampling period. Depends on the workload, and too short a period is worse than too long |
| `XTIER_TOPK_FRACTION` | Fraction of scored pages admitted per epoch. Default 0.15 |
| `XTIER_COOLDOWN_EPOCHS` | Epochs a migrated page waits before reconsideration. Default 15 |
| `XTIER_DRAM_HITS_CSV` | Destination for per-epoch counter output |
| `XTIER_FEATURES_CSV` | Destination for training features in `collect` mode |
| `XTIER_EVAL_CSV` | Destination for output in `evaluate` mode |
