#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
LightGBM training bench -- allocates a large dense dataset and trains a model.
Allocates ~10 GB (7M rows x 400 features x 4 bytes = 11.2 GB).
Env knobs: N_ROWS, N_FEATURES, N_ROUNDS.
"""
import os
import sys
import time

import numpy as np
import lightgbm as lgb


def main():
    n_rows = int(os.environ.get("N_ROWS", 7_000_000))
    n_feat = int(os.environ.get("N_FEATURES", 400))
    n_rounds = int(os.environ.get("N_ROUNDS", 300))
    seed = int(os.environ.get("SEED", 42))

    print(f"[lgbm] N_ROWS={n_rows} N_FEATURES={n_feat} N_ROUNDS={n_rounds}", flush=True)

    rng = np.random.default_rng(seed)
    alloc_gb = n_rows * n_feat * 4 / 1024**3
    print(f"[lgbm] allocating X: ~{alloc_gb:.1f} GB (float32)", flush=True)

    # Allocate in chunks to keep peak memory bounded.
    X = np.empty((n_rows, n_feat), dtype=np.float32)
    chunk = 500_000
    for start in range(0, n_rows, chunk):
        end = min(start + chunk, n_rows)
        X[start:end] = rng.standard_normal((end - start, n_feat), dtype=np.float32)
    print(f"[lgbm] X allocated, shape={X.shape}, nbytes={X.nbytes/1024**3:.2f} GB", flush=True)

    # Binary target derived from a random linear combination of a subset of features
    w = rng.standard_normal(n_feat, dtype=np.float32)
    # work in chunks to avoid 7M x 1 extra allocation blowup
    logits = np.empty(n_rows, dtype=np.float32)
    for start in range(0, n_rows, chunk):
        end = min(start + chunk, n_rows)
        logits[start:end] = X[start:end] @ w
    y = (logits > np.median(logits)).astype(np.int8)
    del logits
    print(f"[lgbm] y built, positive frac={y.mean():.3f}", flush=True)

    t_ds = time.time()
    ds = lgb.Dataset(X, label=y, free_raw_data=False)
    print(f"[lgbm] Dataset build: {time.time()-t_ds:.2f}s", flush=True)

    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=127,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_data_in_leaf=2000,
        verbose=-1,
        num_threads=int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 10)),
    )

    t_train = time.time()
    booster = lgb.train(params, ds, num_boost_round=n_rounds)
    dt = time.time() - t_train
    print(f"[lgbm] Training: {dt:.2f}s for {n_rounds} rounds ({dt/n_rounds*1000:.2f} ms/round)", flush=True)

    # Sanity prediction
    t_pred = time.time()
    preds = booster.predict(X[:100_000])
    print(f"[lgbm] Prediction (100K rows): {time.time()-t_pred:.2f}s preds[0:5]={preds[:5]}", flush=True)

    print("[lgbm] done", flush=True)


if __name__ == "__main__":
    main()
