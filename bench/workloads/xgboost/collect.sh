#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Collect MLP training features from an XGBoost run.
#
#   sudo ./collect.sh [seconds]
#
# Runs the workload allocating on the slow tier with the xtier module NOT
# loaded, so nothing migrates while we watch. The profiler runs in collect
# mode, writing one row per sampled page per epoch to $OUT.
#
# Collect under the same DRAM:CXL ratio you intend to run under -- the model
# learns what is worth promoting given that much fast memory.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../env.sh"

SECONDS_TO_RUN="${1:-900}"
OUT="${OUT:-$XTIER_RESULTS/features_xgboost.csv}"
PEBS_PERIOD="${PEBS_PERIOD:-2000}"
EPOCH_MS="${EPOCH_MS:-100}"

N_SAMPLES="${N_SAMPLES:-7000000}"
N_FEATURES="${N_FEATURES:-100}"
N_ROUNDS="${N_ROUNDS:-50}"

mkdir -p "$(dirname "$OUT")"
rmmod xtier_executor 2>/dev/null || true
rm -f /sys/fs/bpf/mprof_* 2>/dev/null || true
xtier_sysprep || exit 1

echo "[*] xgboost: ${N_SAMPLES}x${N_FEATURES}, ${N_ROUNDS} rounds, ${SECONDS_TO_RUN}s"
numactl --cpunodebind=0 --preferred=1 \
    python3 "$HERE/phase_change.py" \
        --n-samples "$N_SAMPLES" --n-features "$N_FEATURES" --n-rounds "$N_ROUNDS" \
    > /tmp/xtier_collect_workload.log 2>&1 &
WL_PID=$!
sleep 6
kill -0 "$WL_PID" 2>/dev/null || { echo "workload died:"; tail -20 /tmp/xtier_collect_workload.log; exit 1; }

echo "[*] profiler collecting -> $OUT"
XTIER_FEATURES_CSV="$OUT" \
    "$XTIER_ROOT/src/page_profiler_user" "$WL_PID" "$PEBS_PERIOD" "$EPOCH_MS" collect \
    > /tmp/xtier_collect_profiler.log 2>&1 &
PROF_PID=$!

sleep "$SECONDS_TO_RUN"
kill -INT "$PROF_PID" 2>/dev/null; wait "$PROF_PID" 2>/dev/null
kill -9 "$WL_PID" 2>/dev/null; wait "$WL_PID" 2>/dev/null

if [ ! -s "$OUT" ]; then
    echo "no features written -- check /tmp/xtier_collect_profiler.log" >&2
    exit 1
fi
echo "[+] $(wc -l < "$OUT") rows -> $OUT"
echo "    train with: XTIER_OUTPUT_HEADER=ml/models/mlp_q8_xgboost.h \\"
echo "                python3 ml/train_and_export.py --input $OUT"
