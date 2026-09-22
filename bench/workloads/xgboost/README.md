<!-- SPDX-License-Identifier: Apache-2.0 -->
# XGBoost

A two-phase workload. It trains a model, then runs inference with it. The
phase boundary is the interesting part. The hot working set moves to a
disjoint set of pages, so a tiering system has to notice and follow it.

```sh
pip install xgboost numpy
```

## The two variants

`phase_change.py` frees phase 1's training data at the boundary, so phase 2
allocates into memory that has just been returned to the OS.

`phase_change_held.py` keeps a live reference to it instead. Phase 2 therefore
allocates under pressure: node 0 is still full of phase 1's now-cold data, so
the new data lands on the slow tier by first touch. This is the harder case,
and the one that separates systems that demote cold pages from systems that
only promote hot ones.

```sh
numactl --cpunodebind=0 --preferred=1 \
  python3 phase_change.py --n-samples 7000000 --n-features 100 --n-rounds 50
```

Each phase targets a couple of minutes on a 20-core machine. `--n-samples`
scales the working set roughly linearly. The default is about 6.4 GB resident.

## Collecting features

```sh
sudo ./collect.sh 900        # 15 minutes
```

Writes to `$XTIER_RESULTS/features_xgboost.csv`. Override with `OUT=`,
`PEBS_PERIOD=`, `EPOCH_MS=`, or the `N_SAMPLES`/`N_FEATURES`/`N_ROUNDS` knobs.

Then train:

```sh
XTIER_OUTPUT_HEADER=ml/models/mlp_q8_xgboost.h \
  python3 ml/train_and_export.py --input $XTIER_RESULTS/features_xgboost.csv
```
