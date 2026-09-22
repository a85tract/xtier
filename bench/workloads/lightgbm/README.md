<!-- SPDX-License-Identifier: Apache-2.0 -->
# LightGBM

Gradient-boosted tree training on a synthetic dataset. The feature matrix is
large and scanned repeatedly across boosting rounds. That gives a big working
set with stable, rankable hotness, which is close to the case xTier is built
for.

```sh
pip install lightgbm numpy
```

## Running

```sh
N_ROWS=7000000 N_FEATURES=400 N_ROUNDS=200 \
  numactl --cpunodebind=0 --preferred=1 python3 lightgbm_bench.py
```

Defaults are 7M x 400 over 200 rounds: about 10.4 GB for the feature matrix and
~42 GB peak resident once LightGBM builds its histograms. `SEED` and
`OMP_NUM_THREADS` are also honoured.

Smaller configurations fit inside node 0 at loose ratios and stop exercising
tiering at all, so scale the ratio down rather than the workload.

## Collecting features

```sh
sudo ./collect.sh 900        # 15 minutes
```

Writes to `$XTIER_RESULTS/features_lightgbm.csv`.

Note the default sampling period here is 30000, against 2000 for the tree and
graph workloads. LightGBM touches memory more sparsely, and sampling it too
aggressively costs more in NMI handling than it returns in signal.

Then train:

```sh
XTIER_OUTPUT_HEADER=ml/models/mlp_q8_lightgbm.h \
  python3 ml/train_and_export.py --input $XTIER_RESULTS/features_lightgbm.csv
```
