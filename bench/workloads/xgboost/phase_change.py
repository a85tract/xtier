#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Two-phase XGBoost workload for memory-tiering phase-change experiments.

Phase 1: training on a synthetic dataset. Hot pages are the row-major
         feature matrix and the gradient histograms, accessed in a
         streaming pattern across boosting rounds.

Phase 2: prediction using the trained model on a fresh inference dataset.
         The training data is freed and returned to the OS at the phase
         boundary. Phase 2's hot working set is the trained tree structure
         and the inference feature matrix, both of which occupy disjoint
         virtual addresses from phase 1.

Each phase targets ~150-200 seconds on a Haswell-EP 20-core box.
"""
import argparse
import ctypes
import gc
import os
import time

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification


# Phase sizing. 30 M rows matches the long-form sweep config (workload3
# with --phase1-rounds 170 --phase2-iters 1700, free between phases).
PHASE1_SAMPLES   = 30_000_000
PHASE1_FEATURES  = 30
PHASE1_ROUNDS    = 170

PHASE2_SAMPLES   = 30_000_000
PHASE2_FEATURES  = 30
PHASE2_ITERS     = 1700


def trim_libc():
    """Force glibc to return freed memory to the OS so phase 2 allocates
    at fresh virtual addresses rather than reusing phase 1's arena."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass  # non-glibc, skip


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
    p = argparse.ArgumentParser(
        description="Two-phase XGBoost workload: train, then predict, with a sharp "
                    "working-set shift at the phase boundary."
    )
    p.add_argument("--phase1-rounds", type=int, default=PHASE1_ROUNDS,
                   help=f"Boosting rounds in phase 1 (default: {PHASE1_ROUNDS})")
    p.add_argument("--phase2-iters", type=int, default=PHASE2_ITERS,
                   help=f"Number of inference passes in phase 2 (default: {PHASE2_ITERS})")
    p.add_argument("--save-model", action="store_true",
                   help="Save the trained model at the end (default: don't)")
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
    print(f"[+] Phase 1 (train):   {PHASE1_SAMPLES:,} x {PHASE1_FEATURES}, "
          f"{args.phase1_rounds} rounds")
    print(f"[+] Phase 2 (predict): {PHASE2_SAMPLES:,} x {PHASE2_FEATURES}, "
          f"{args.phase2_iters} iterations")
    print(f"[PHASE_MARKER] start_of_run epoch_unix={time.time():.3f}")

    # ---------------- Phase 1: training ----------------
    X1, y1, dtrain = build_dataset(
        PHASE1_SAMPLES, PHASE1_FEATURES, seed=42, label="phase1"
    )

    print(f"[PHASE_MARKER] phase1_train_start epoch_unix={time.time():.3f}")
    t1_start = time.time()
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=args.phase1_rounds,
        verbose_eval=50,
    )
    t1_end = time.time()
    print(f"[+] [phase1] training finished in {(t1_end - t1_start) / 60:.2f} min")
    print(f"[PHASE_MARKER] phase1_train_end epoch_unix={t1_end:.3f}")

    # ---------------- Phase boundary ----------------
    # Free phase 1's training set and force the allocator to release it
    # back to the OS. Without this, phase 2 may reuse the same virtual
    # addresses, which would defeat the experiment.
    print("[+] [boundary] freeing phase 1 training set")
    del X1, y1, dtrain
    gc.collect()
    trim_libc()
    # Brief pause so the boundary is clearly visible in time-series traces.
    time.sleep(2.0)
    print(f"[PHASE_MARKER] phase_boundary epoch_unix={time.time():.3f}")

    # ---------------- Phase 2: prediction ----------------
    X2, y2, dtest = build_dataset(
        PHASE2_SAMPLES, PHASE2_FEATURES, seed=12345, label="phase2"
    )

    print(f"[PHASE_MARKER] phase2_predict_start epoch_unix={time.time():.3f}")
    t2_start = time.time()
    # Run multiple inference passes over the same DMatrix. The hot working
    # set during prediction is the trained tree structure (read-mostly,
    # cache-friendly per node, but pointer-chase across nodes) and the
    # inference feature matrix (streamed once per pass). This is a
    # structurally different access pattern from training's gradient
    # histogram updates.
    n_correct = 0
    for i in range(args.phase2_iters):
        preds = bst.predict(dtest)
        # Touch the predictions so the optimizer can't elide the call.
        n_correct += int(((preds > 0.5).astype(np.int32) == y2).sum())
        if (i + 1) % 10 == 0:
            print(f"[+] [phase2] completed {i + 1}/{args.phase2_iters} passes")
    t2_end = time.time()
    print(f"[+] [phase2] prediction finished in {(t2_end - t2_start) / 60:.2f} min "
          f"(running accuracy sum: {n_correct})")
    print(f"[PHASE_MARKER] phase2_predict_end epoch_unix={t2_end:.3f}")

    # ---------------- Summary ----------------
    print(f"[+] phase 1 wall: {(t1_end - t1_start) / 60:.2f} min")
    print(f"[+] phase 2 wall: {(t2_end - t2_start) / 60:.2f} min")
    print(f"[+] total wall:   {(t2_end - t1_start) / 60:.2f} min")

    if args.save_model:
        bst.save_model("xgb_two_phase.model")
        print("[+] model saved to xgb_two_phase.model")


if __name__ == "__main__":
    main()
