<!-- SPDX-License-Identifier: Apache-2.0 -->
# Cost and footprint

What the system costs to run, measured on dual-socket Haswell-EP at a 1:20
DRAM:CXL ratio. Absolute numbers will differ on other hardware. The shape
should not.

## Steady-state memory footprint

| Component | Resident | Notes |
|---|---|---|
| BPF maps | **218 MB** | Dominated by `page_state_map` (LRU hash, 1M entries x 128 B ~ 200 MB), plus the two 1 MB region-queue buffers and the MLP weight maps |
| Quantized MLP | **114 KB** | The INT8 header linked into the loader |
| `xtier_executor` module | **24 KB** | Resident per `lsmod`. The on-disk `.ko` is larger, almost all debug info |

Essentially all of it is the page-state map. The model and the module are
rounding error. `page_state_map` is sized in `src/pebs_mlp_kern.c` and is the
knob to turn if 218 MB is too much.

## Latency of one decision

| Stage | Latency | What happens |
|---|---|---|
| Telemetry | **348 ns** | Read a PEBS sample from the BPF region queue, pop the next pending page |
| MLP inference | **34-92 us** | Quantized 32 -> 192 -> 96 -> 1 forward pass. Varies with workload |
| Migration | **19-20 us** | One `migrate_pages()`: unmap, copy, remap. Dominated by the TLB flush |
| **End to end** | **~55-112 us** | Sample arriving in the ring to the page resident on the new node |

PEBS NMI handler cost is not attributed separately: the kernel charges it to
whichever CPU was interrupted.

## Training cost, per workload

| Stage | Action | Time | Output |
|---|---|---|---|
| Collect | Workload under `page_profiler_user collect` | 12-15 min | `features.csv`, ~3-5 GB, ~10M rows |
| Train | `ml/train_and_export.py` | 1-2 min on CPU | quantized weights |
| Export | INT8 quantize, emit C header | < 1 s | a 114 KB header |
| Rebuild | `make MODEL_HDR=...` | ~10 s | loader binary |
| **Total** | | **~15 min** | |

A one-time cost per workload, and the only artifact it produces is that
header. See `../ml/README.md`.
