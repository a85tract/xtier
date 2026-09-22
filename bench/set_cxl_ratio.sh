#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# CXL emulation: set a DRAM:CXL ratio by offlining node-0 memory blocks
# =============================================================================
#
# THIS EMULATES CXL. There is no CXL device involved. The setup assumed is an
# ordinary two-socket NUMA machine, where node 0 is the fast tier and node 1 --
# memory on the far socket -- stands in for CXL-attached memory. It is slower
# in the same way CXL is: higher latency, same programming model, addressable
# and cacheable. Not the same, though: the far socket sits at ~1.5-1.6x local
# latency where CXL is typically further out and behaves differently under
# bandwidth load, so expect the ordering of results to carry over to real
# hardware and the magnitudes not to. With a real CXL device, present it as a
# memory-only NUMA node and skip this script -- nothing in xTier cares how
# node 1 came to exist.
#
# Why shrink node 0: the two nodes are usually the same size, so a workload
# that fits in one fits in the fast tier and no tiering system ever has a
# decision to make. Offlining blocks on node 0 creates that decision. Blocks
# come off whole (usually 128 MB), and one pinned by unmovable kernel memory
# cannot be reclaimed at all, so a tight ratio may fall short -- the script
# reports what it achieved.
#
# Usage: sudo ./set_cxl_ratio.sh <ratio>
#
# Examples, on a machine with ~80GB per node:
#   sudo ./set_cxl_ratio.sh 1:10   -> ~8GB fast tier, ~80GB slow
#   sudo ./set_cxl_ratio.sh 1:4    -> ~16GB fast tier, ~80GB slow
#   sudo ./set_cxl_ratio.sh 1:2    -> ~40GB fast tier, ~80GB slow
#   sudo ./set_cxl_ratio.sh reset  -> bring all node-0 memory back online
#
# Verify the result with `numactl -H` before trusting a measurement.
# =============================================================================

RATIO="${1:-}"

if [ -z "$RATIO" ]; then
    echo "Usage: sudo $0 <ratio>"
    echo "  e.g.: sudo $0 1:10"
    echo "  e.g.: sudo $0 reset"
    exit 1
fi

set -e

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Run as root (sudo)."
    exit 1
fi

BLOCK_SIZE_BYTES=$(cat /sys/devices/system/memory/block_size_bytes)
BLOCK_SIZE_MB=$(( 16#${BLOCK_SIZE_BYTES#0x} / 1024 / 1024 ))

if [ "$BLOCK_SIZE_MB" -eq 0 ]; then
    echo "ERROR: Could not determine memory block size."
    exit 1
fi

echo "Memory block size: ${BLOCK_SIZE_MB}MB"

get_node0_full_capacity_mb() {
    local total=0
    for block in /sys/devices/system/node/node0/memory*; do
        [ -d "$block" ] || continue
        total=$((total + BLOCK_SIZE_MB))
    done
    echo "$total"
}

get_node0_online_mb() {
    local total=0
    for block in /sys/devices/system/node/node0/memory*; do
        [ -d "$block" ] || continue
        if [ "$(cat "$block/state" 2>/dev/null)" = "online" ]; then
            total=$((total + BLOCK_SIZE_MB))
        fi
    done
    echo "$total"
}

NODE1_MB=$(awk '/MemTotal/ {printf "%d", $4/1024}' /sys/devices/system/node/node1/meminfo)
NODE0_FULL_MB=$(get_node0_full_capacity_mb)
NODE0_ONLINE_MB=$(get_node0_online_mb)

echo "Node 0 full capacity:  ${NODE0_FULL_MB}MB"
echo "Node 0 currently online: ${NODE0_ONLINE_MB}MB"
echo "Node 1 (CXL tier):    ${NODE1_MB}MB"
echo ""

if [ "$RATIO" = "reset" ]; then
    TARGET_MB=$NODE0_FULL_MB
    echo "Mode: RESET -- bringing all node 0 memory back online"
else
    DRAM_PARTS=$(echo "$RATIO" | cut -d: -f1)
    CXL_PARTS=$(echo "$RATIO" | cut -d: -f2)

    if [ -z "$DRAM_PARTS" ] || [ -z "$CXL_PARTS" ] || [ "$CXL_PARTS" -eq 0 ]; then
        echo "ERROR: Invalid ratio format. Use '1:N' (e.g., '1:10') or 'reset'."
        exit 1
    fi

    TARGET_MB=$(( NODE1_MB * DRAM_PARTS / CXL_PARTS ))
    TARGET_MB=$(( (TARGET_MB / BLOCK_SIZE_MB) * BLOCK_SIZE_MB ))

    MIN_MB=$((BLOCK_SIZE_MB * 2))
    if [ "$TARGET_MB" -lt "$MIN_MB" ]; then
        TARGET_MB=$MIN_MB
        echo "WARNING: Target too small, clamping to ${TARGET_MB}MB (${MIN_MB}MB minimum)."
    fi

    if [ "$TARGET_MB" -gt "$NODE0_FULL_MB" ]; then
        TARGET_MB=$NODE0_FULL_MB
        echo "WARNING: Target exceeds node 0 capacity, clamping to ${TARGET_MB}MB."
    fi

    echo "Ratio ${RATIO} -> target node 0 online: ${TARGET_MB}MB (~$((TARGET_MB/1024))GB)"
fi

DIFF_MB=$((NODE0_ONLINE_MB - TARGET_MB))

if [ "$DIFF_MB" -eq 0 ]; then
    echo "Already at target. Nothing to do."
    numactl -H | grep -E "node [01] (size|free)"
    exit 0
fi

if [ "$DIFF_MB" -gt 0 ]; then
    BLOCKS_NEEDED=$((DIFF_MB / BLOCK_SIZE_MB))
    echo "Need to offline $BLOCKS_NEEDED blocks (${DIFF_MB}MB)..."
    echo ""

    BLOCKS=()
    for block in /sys/devices/system/node/node0/memory*; do
        [ -d "$block" ] || continue
        if [ "$(cat "$block/state" 2>/dev/null)" = "online" ]; then
            num=$(basename "$block" | sed 's/memory//')
            BLOCKS+=("$num:$block")
        fi
    done

    IFS=$'\n' SORTED=($(printf '%s\n' "${BLOCKS[@]}" | sort -t: -k1 -n -r)); unset IFS

    offlined=0
    failed=0
    for entry in "${SORTED[@]}"; do
        if [ "$offlined" -ge "$BLOCKS_NEEDED" ]; then
            break
        fi
        block="${entry#*:}"

        if echo "offline" > "$block/state" 2>/dev/null; then
            offlined=$((offlined + 1))
            printf "\r  Offlined %d / %d blocks..." "$offlined" "$BLOCKS_NEEDED"
        else
            failed=$((failed + 1))
        fi
    done
    echo ""

    if [ "$offlined" -lt "$BLOCKS_NEEDED" ]; then
        echo "WARNING: Could only offline $offlined of $BLOCKS_NEEDED blocks ($failed failed -- pinned kernel memory)."
        echo "         Achieved: ~$(( (NODE0_ONLINE_MB - offlined * BLOCK_SIZE_MB) ))MB on node 0."
    else
        echo "Done. Offlined $offlined blocks."
    fi
else
    GROW_MB=$(( -DIFF_MB ))
    BLOCKS_NEEDED=$((GROW_MB / BLOCK_SIZE_MB))
    echo "Need to online $BLOCKS_NEEDED blocks (${GROW_MB}MB)..."
    echo ""

    BLOCKS=()
    for block in /sys/devices/system/node/node0/memory*; do
        [ -d "$block" ] || continue
        if [ "$(cat "$block/state" 2>/dev/null)" = "offline" ]; then
            num=$(basename "$block" | sed 's/memory//')
            BLOCKS+=("$num:$block")
        fi
    done

    IFS=$'\n' SORTED=($(printf '%s\n' "${BLOCKS[@]}" | sort -t: -k1 -n)); unset IFS

    onlined=0
    for entry in "${SORTED[@]}"; do
        if [ "$onlined" -ge "$BLOCKS_NEEDED" ]; then
            break
        fi
        block="${entry#*:}"

        if echo "online" > "$block/state" 2>/dev/null; then
            onlined=$((onlined + 1))
            printf "\r  Onlined %d / %d blocks..." "$onlined" "$BLOCKS_NEEDED"
        fi
    done
    echo ""
    echo "Done. Onlined $onlined blocks."
fi

echo ""
echo "=== Current state ==="
numactl -H | grep -E "node [01] (size|free)"
FINAL_NODE0=$(get_node0_online_mb)
echo ""
echo "Effective ratio: 1:$(( NODE1_MB / FINAL_NODE0 )) (node0=${FINAL_NODE0}MB, node1=${NODE1_MB}MB)"
