# xTier

xTier performs machine-learned page placement for tiered memory. The decision
logic runs inside the kernel.

Intel PEBS provides memory access samples. An eBPF program maintains per-page
access state and evaluates a quantized INT8 multilayer perceptron over that
state to score how much each page would benefit from promotion. A kernel module
then migrates the selected pages between the fast tier, which is DRAM, and the
slow tier, which is CXL-attached memory. In this work the slow tier is emulated
using the far NUMA node.

The entire decision path remains in the kernel. A promotion decision takes
approximately 55 microseconds from sample to completed migration. The resident
footprint of the system is approximately 218 MB, of which almost all is the
page state map. The model occupies 114 KB.

## Repository layout

| Directory | Contents |
|---|---|
| `src/` | The system itself. Kernel module, eBPF program, userspace loader, and shared headers |
| `kernel/` | The Linux patches and a script that builds and installs them |
| `ml/` | Model training and INT8 export. No trained models are distributed |
| `bench/` | Ratio control, system preparation, and the workloads used in evaluation |
| `figures/` | Plot scripts and the data behind each published figure |
| `docs/` | Setup, baseline configuration, and measured cost |

## Installation

There are five steps. `docs/SETUP.md` describes each in detail.

### 0. Install the dependencies

```sh
sudo ./setup.sh
```

This installs the distribution packages, builds `bpftool` from the kernel tree,
and installs the Python packages. The distribution `bpftool` is tied to the
distribution kernel and cannot read the BTF of the kernel built in step 1, so
it is built here instead.

### 1. Build and install the kernel

xTier calls `xtier_migrate_range()`, which is not present in mainline Linux.
The module will not load on an unmodified kernel. One script performs the whole
sequence, which is to fetch the source, apply the patch, configure, build,
install, and set the boot entry.

```sh
sudo kernel/build.sh           # approximately 30 GB and 45 minutes
sudo kernel/build.sh --slim    # approximately 12 GB and 15 minutes
```

Reboot when it completes.

### 2. Train a model

xTier does not operate without a model, and no trained model is distributed
with this repository. A model is fitted to the access pattern of one workload
on your own hardware, so a model trained elsewhere would not describe your
system. Run the workload under the profiler in `collect` mode and then train.
The whole process takes about 15 minutes. See `ml/README.md`.

```sh
sudo src/page_profiler_user <pid> <freq> <epoch_ms> collect
python3 ml/train_and_export.py
```

### 3. Build xTier

`setup.sh` installed everything this needs.

```sh
make MODEL_HDR=$PWD/ml/models/mlp_q8.h
make help
```

Collecting the features in step 2 needs the loader, and the loader compiles a
model in, so the first build has nothing to compile. `make bootstrap-model`
writes placeholder weights to break that loop.

```sh
make bootstrap-model
make MODEL_HDR=$PWD/ml/models/mlp_q8_bootstrap.h   # collect mode only
```

Use `make LIBBPF_SRC=` to link the distribution copy of libbpf rather than the
one in the kernel tree. The model is compiled into the loader, so selecting a
different model requires a rebuild rather than a restart.

### 4. Configure the system and run

The system configuration in `docs/SETUP.md` section 3 is required rather than
optional. Two of those sysctls determine whether xTier receives samples at all.
Section 5 of the same document explains why.

```sh
sudo make restart
numactl --cpunodebind=0 --preferred=1 <workload> &
sudo src/page_profiler_user $! <freq> <epoch_ms> run
```

Allocating on the slow tier and allowing xTier to promote pages upward is the
configuration the system is designed for.

## Operational notes

Section 5 of `docs/SETUP.md` is worth reading before the first run. It covers
four conditions that produce misleading results without producing an error.
The kernel throttles PEBS delivery if two sysctls are left unset. Running
`perf stat` with PEBS-eligible events takes the PMU counters away from xTier.
Hardware prefetchers absorb the latency that page placement is intended to
avoid. Migrations performed by xTier do not appear in `pgmigrate_success` or
`numa_pages_migrated`, because they occur in kthread context.

## Figures

```sh
pip install -r figures/requirements.txt
python3 figures/plot_phase_response.py /tmp/figs
```

Every figure regenerates from the committed data without network access. See
`figures/README.md`.

## Licensing

Every file carries an `SPDX-License-Identifier`. See `LICENSE` (GPL-2.0) and
`LICENSE.apache` (Apache-2.0).
