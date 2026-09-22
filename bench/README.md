<!-- SPDX-License-Identifier: Apache-2.0 -->
# bench

What you need to put xTier under load: a way to create memory pressure, the
system preparation every run wants, and workloads to point it at.

```sh
source bench/env.sh
```

| | |
|---|---|
| `env.sh` | Paths, and `xtier_sysprep`, which applies `../docs/SETUP.md` section 3 |
| `set_cxl_ratio.sh` | Sets the DRAM:CXL ratio by offlining node-0 memory blocks |
| `workloads/` | One directory per workload, with setup instructions |

## Creating memory pressure

xTier only has a decision to make when the working set does not fit in node 0.
On a machine with equal-sized nodes it always fits, so you shrink node 0:

```sh
sudo ./set_cxl_ratio.sh 1:10     # ~8 GB fast tier against ~80 GB slow
sudo ./set_cxl_ratio.sh reset    # bring it all back online
```

This offlines memory blocks at runtime, so it takes effect immediately and
survives until you reset it. Check the result with `numactl -H`.

There are two limits. Offlining cannot reclaim a block holding unmovable
kernel allocations, so very tight ratios may come up short. The script reports
what it actually achieved. The granularity is the block size in
`/sys/devices/system/memory/block_size_bytes`, usually 128 MB.

If you need a ratio runtime offlining will not reach, the alternative is the
kernel's `memmap=` parameter, which hides a range from node 0 at boot. Add
`memmap=<size>$<offset>` to the kernel command line. For a 72 GB node 0 out of
80 GB, that is `memmap=72G$8G`. Two things go wrong with it. The `$` must be
escaped as `\$` in `/boot/grub/grub.cfg`, or GRUB expands it as a variable and
you silently get `memmap=72GG`. It also costs a reboot per ratio. Verify with
`numactl -H` after booting, not before.

## Before every run

```sh
sudo -E bash -c 'source bench/env.sh; xtier_sysprep'
```

Run this every time. Two of those sysctls decide whether xTier samples
normally or the kernel throttles PEBS delivery down to a trickle. A throttled
run looks like a working one, but makes almost no decisions.
`../docs/SETUP.md` section 5 covers that and the other quiet failure modes.

## Workloads

See `workloads/README.md`. xTier attaches to a PID and places that process's
pages, so anything memory-bound will do. What is here is what we used, not
what is required.
