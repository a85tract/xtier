<!-- SPDX-License-Identifier: Apache-2.0 -->
# Microbenchmarks

Synthetic access patterns with known ground truth, for exercising behaviour the
real workloads only produce incidentally. Useful when you want to know whether
the system does the right thing, rather than how fast it is.

```sh
make
```

The three chase workloads need OpenMP. Nothing else beyond libc is required,
and the NUMA calls go through `syscall(2)` directly.

| Binary | Pattern |
|---|---|
| `phase_chase` | Pointer chase whose hot region relocates partway through. Tests how fast a system follows a phase change |
| `phase_zipf` | Same relocation, but with Zipf-skewed hotness inside the active region, so pages are rankable rather than uniformly hot |
| `zipf_chase` | Zipf-skewed hotness with no phase change -- the steady-state case |
| `remote_stream` | Streams from the far node. Measures the bandwidth gap the tiers actually have |
| `placement_sampler` | Observes where another process's pages live, without disturbing it |
| `mof_bind` | Sets `MPOL_BIND` with migrate-on-fault, which `numactl --membind` does not set |

## placement_sampler

Answers "where does the hot set actually live, right now", which migration
counters cannot. It queries the NUMA node of a sample of pages from a target
process every second using `move_pages(2)` in query mode.

```sh
sudo ./placement_sampler <pid> > placement.csv
```

It is passive. It uses no PMU counters, so it does not compete with xTier's
sampling. It does not touch page data, so it does not perturb the access
pattern it is measuring. It reads where pages are, not how they got there, so
compaction traffic cannot pollute it. That makes it valid against any tiering
system, including ones whose migrations never reach the global counters.

Needs root. Querying another process's mapping requires `PTRACE_MODE_READ`.
