#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Two-phase XGBoost workload -- phase change with HELD prior data.

Same as phase_change.py except phase 1's training data is NOT freed at the
phase boundary. Instead it's held by a reference so its pages stay
resident on whichever node they were placed on. Phase 2 allocates a
fresh inference dataset under DRAM pressure: N0 is already full of
phase 1's now-cold data, so phase 2's pages spill to CXL by first-touch
(or get demoted by kswapd shortly after).

This is the structure that makes "phase change" mean what it should
mean for tier-migration experiments: at t=phase_boundary, the new hot
set is on the slow tier and the system has to migrate.
"""
import argparse
import ctypes
import gc
import os
import time

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification


PHASE1_SAMPLES   = 20_000_000
PHASE1_FEATURES  = 30
PHASE1_ROUNDS    = 60
PHASE2_SAMPLES   = 20_000_000
PHASE2_FEATURES  = 30
PHASE2_ITERS     = 200


def build_dataset(n_samples, n_features, seed, label):
    print(f"[+] [{label}] generating {n_samples:,} samples x {n_features} features (seed={seed})")
    t0 = time.time()
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, int(n_features * 0.6)),
        n_redundant=int(n_features * 0.2),
        n_repeated=0,
        n_classes=2,
        flip_y=0.01,
        class_sep=1.0,
        random_state=seed,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    dmat = xgb.DMatrix(X, label=y)
    print(f"[+] [{label}] dataset built in {time.time() - t0:.1f}s "
          f"(matrix size: {X.nbytes / (1024**3):.2f} GiB)")
    return X, y, dmat


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1-rounds", type=int, default=PHASE1_ROUNDS)
    p.add_argument("--phase2-iters",  type=int, default=PHASE2_ITERS)
    return p.parse_args()


def main():
    args = parse_args()

    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "logloss",
        "tree_method":      "hist",
        "max_depth":        6,
        "learning_rate":    0.1,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
    }

    print(f"[+] PID: {os.getpid()}")
    print(f"[+] Phase 1 (train, KEPT alive across boundary):   "
          f"{PHASE1_SAMPLES:,} x {PHASE1_FEATURES}, {args.phase1_rounds} rounds")
    print(f"[+] Phase 2 (predict, fresh alloc on full N0):     "
          f"{PHASE2_SAMPLES:,} x {PHASE2_FEATURES}, {args.phase2_iters} iterations")
    print(f"[PHASE_MARKER] start_of_run epoch_unix={time.time():.3f}")

    # Phase 1: build + train. Hold X1, y1, dtrain in scope across the boundary.
    X1, y1, dtrain = build_dataset(PHASE1_SAMPLES, PHASE1_FEATURES, 42, "phase1")

    print(f"[PHASE_MARKER] phase1_train_start epoch_unix={time.time():.3f}")
    t1_start = time.time()
    bst = xgb.train(params, dtrain, num_boost_round=args.phase1_rounds, verbose_eval=50)
    t1_end = time.time()
    print(f"[+] [phase1] training finished in {(t1_end - t1_start) / 60:.2f} min")
    print(f"[PHASE_MARKER] phase1_train_end epoch_unix={t1_end:.3f}")

    # CRITICAL: do NOT free X1, y1, dtrain. They stay alive on N0 as cold
    # ballast. Phase 2 must allocate around them. No malloc_trim, no gc.
    time.sleep(2.0)
    print(f"[PHASE_MARKER] phase_boundary epoch_unix={time.time():.3f}")

    # Phase 2: build inference set under DRAM pressure.
    X2, y2, dtest = build_dataset(PHASE2_SAMPLES, PHASE2_FEATURES, 12345, "phase2")

    print(f"[PHASE_MARKER] phase2_predict_start epoch_unix={time.time():.3f}")
    t2_start = time.time()
    n_correct = 0
    for i in range(args.phase2_iters):
        preds = bst.predict(dtest)
        n_correct += int(((preds > 0.5).astype(np.int32) == y2).sum())
        if (i + 1) % 25 == 0:
            print(f"[+] [phase2] {i + 1}/{args.phase2_iters} passes")
    t2_end = time.time()
    print(f"[+] [phase2] prediction finished in {(t2_end - t2_start) / 60:.2f} min "
          f"(running accuracy sum: {n_correct})")
    print(f"[PHASE_MARKER] phase2_predict_end epoch_unix={t2_end:.3f}")

    # Defensive use so phase 1 data isn't optimized away
    _phase1_alive = (X1.shape, y1.shape, len(dtrain.feature_names or []))
    print(f"[+] phase1 still alive at end (shape: {_phase1_alive[0]})")


if __name__ == "__main__":
    main()
