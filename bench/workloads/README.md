<!-- SPDX-License-Identifier: Apache-2.0 -->
# Workloads

xTier does not care what it is managing. It attaches to a PID and places
that process's pages. Any memory-bound workload will do. These are the ones we
used, with enough setup to get each running.

Each directory documents how to get the workload, and where we ship a script,
how to collect training features with it. You need features from a workload
before you can train a model for it (`../../ml/README.md`).

| Directory | Workload | What you get |
|---|---|---|
| `xgboost/` | Two-phase XGBoost training and inference | Workload script and `collect.sh` |
| `lightgbm/` | LightGBM training on a synthetic dataset | Workload script and `collect.sh` |
| `gapbs/` | GAP Benchmark Suite graph kernels | Setup instructions; the suite is upstream |
| `microbench/` | Synthetic access patterns | C sources and a Makefile |

## What makes a workload useful here

Two properties, and without them you will measure nothing:

**It has to be memory-bound.** If the CPU is not stalled waiting on loads, no
placement decision can change the wall time. A workload whose working set fits
in cache will show identical results under every tiering system.

**Its working set has to exceed node 0.** Set the DRAM:CXL ratio so it does --
see `../set_cxl_ratio.sh`. At a loose ratio a workload that fits in fast memory
never triggers a migration, and the comparison is vacuous.

Scale these workloads up rather than down. The defaults here are sized for a
machine with ~80 GB per node. On a smaller one, shrink node 0 further rather
than shrinking the workload.

## Collecting features

Each `collect.sh` follows the same shape, and any workload of your own can too:

1. Prepare the system (`xtier_sysprep` from `../env.sh`).
2. Unload the xtier module, so nothing migrates while you observe.
3. Start the workload with `numactl --cpunodebind=0 --preferred=1`.
4. Attach `page_profiler_user <pid> <period> <epoch_ms> collect`.
5. Let it run ~15 minutes. The CSV lands at `$XTIER_FEATURES_CSV`.

Collect at the ratio you intend to run at. The label is how much promoting a
page is worth, which depends on how much fast memory there is to compete for.
