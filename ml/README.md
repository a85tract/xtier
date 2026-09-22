<!-- SPDX-License-Identifier: Apache-2.0 -->
# Model training and export

The script `train_and_export.py` trains the page benefit model, quantizes it to
INT8, and writes a C header containing the weights. The loader compiles that
header into its binary and writes the weights to the BPF program's maps at
startup. Inference therefore runs entirely in the kernel, with no userspace
round trip for each decision.

**No trained model is distributed with this repository.** A model is fitted to
the access pattern of a single workload, on particular hardware, at a
particular DRAM to CXL ratio. A model trained under other conditions would not
describe your system. Training your own takes approximately 15 minutes.

## 1. Collect features

The profiler compiles a model in, and there is no trained model yet. Build it
against placeholder weights first.

```sh
cd .. && make bootstrap-model && make MODEL_HDR=$PWD/ml/models/mlp_q8_bootstrap.h
```

Run the workload with the profiler in `collect` mode. The profiler writes one
row for each page and epoch it observes.

```sh
sudo make restart
numactl --cpunodebind=0 --preferred=1 <workload> &
sudo src/page_profiler_user $! <freq> <epoch_ms> collect
```

This produces `features.csv`, which is typically 3 to 5 GB for a long run.
Collect under the same ratio and system configuration you intend to run under.
See `../docs/SETUP.md`.

The scripts at `../bench/workloads/*/collect.sh` are worked examples. Any other
workload can follow the same procedure, which is described in
`../bench/workloads/README.md`.

## 2. Train

```sh
pip install -r requirements.txt
python3 train_and_export.py               # reads features.csv, writes models/mlp_q8.h
```

| Variable | Default | Meaning |
|---|---|---|
| `XTIER_OUTPUT_HEADER` | `models/mlp_q8.h` | Destination for the generated header |
| `XTIER_EPOCHS` | `8` | Number of training epochs |
| `XTIER_LABEL_TF` | `rank` | Target transform, either `rank` or `log1p` |

Naming the output after the workload allows several models to be kept side by
side.

```sh
XTIER_OUTPUT_HEADER=models/mlp_q8_<workload>.h python3 train_and_export.py
```

## 3. Build with the model

```sh
cd .. && make MODEL_HDR=$PWD/ml/models/mlp_q8_<workload>.h
```

`MODEL_HDR` is a prerequisite of the loader target, so changing it triggers a
rebuild. The harness scripts look for `ml/models/mlp_q8_<workload>.h` by
default, and use the value of `XTIER_MODEL` when that variable is set.

## What the model predicts

The model does not predict whether a page is hot. It predicts how much
promoting that page would be worth.

```
reuse_intensity = hits on this page over the next 10 epochs
tenure          = how many of those 10 epochs saw a hit
reuse_speed     = epochs until the first re-access, capped at 10
benefit         = reuse_intensity * (tenure / 10) / (1 + reuse_speed)
```

A page that is re-accessed soon, frequently, and over a sustained period
receives a high score. A page touched once receives a score near zero.

The resulting distribution is a point mass at zero with a heavy tail, because
most pages are never re-accessed within the window. For this reason the default
target transform is percentile rank. The top-K admission gate consumes a
ranking rather than a magnitude, and a ranking is what survives quantization to
the range 0 to 255.

## Generated headers

Generated headers are written to `models/`, which is distributed empty. Each
one is marked as generated and carries an SPDX GPL-2.0 tag, because it is
compiled into the loader and the loader is licensed GPL-2.0.
The licence of a generator does not determine the licence of its output, so
this remains correct regardless of which licence is eventually chosen for
`ml/`.
