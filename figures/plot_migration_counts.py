#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial'],
    'font.size': 22,
    'axes.labelsize': 28,
    'axes.titlesize': 28,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'text.color': '#000000',
    'axes.labelcolor': '#000000',
    'axes.edgecolor': '#000000',
    'xtick.color': '#000000',
    'ytick.color': '#000000',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

ratios = ['1:5', '1:10', '1:15', '1:20', '1:25']

# Column order per row: [autonuma, tpp, memtis, HybridTier, xTier]
mig_data = [
    ('XGBoost', {
        '1:5':  [39902,    74123,    212182,   59436,   1706],
        '1:10': [198632,   567183,   5280457,  427527,  1830],
        '1:15': [487819,   2153370,  19164787, 724635,  1810],
        '1:20': [1324614,  1807576,  33443129, 955210,  1867],
        '1:25': [2646949,  2646949,  30894000, 987834,  1938],
    }),
    ('BC', {
        '1:5':  [187,     3072,     1002,     3480,    675],
        '1:10': [38538,   17283,    1161153,  6539,    624],
        '1:15': [30853,   372933,   2469841,  1689,    551],
        '1:20': [84696,   936565,   2412895,  1683,    523],
        '1:25': [41049,   697385,   2457771,  7240,    577],
    }),
    ('TC', {
        '1:5':  [311055,  375252,   110339,   282114,  5111],
        '1:10': [296758,  366650,   1440619,  234126,  5039],
        '1:15': [245344,  1338911,  2708986,  231518,  5007],
        '1:20': [348643,  1555445,  3041206,  231980,  4823],
        '1:25': [276084,  282933,   3932693,  231290,  4940],
    }),
    ('BFS', {
        '1:5':  [0,       0,        2173,     0,       506],
        '1:10': [0,       0,        6097,     0,       675],
        '1:15': [9995,    2815,     152895,   759,     619],
        '1:20': [27446,   53354,    1764089,  57420,   602],
        '1:25': [35673,   65423,    1872789,  19452,   632],
    }),
    ('PR', {
        '1:5':  [5,       57934,    1596,     320,     120],
        '1:10': [13519,   48593,    1790510,  3717,    140],
        '1:15': [29048,   713096,   1542198,  30236,   600],
        '1:20': [11861,   1215124,  1492103,  97839,   754],
        '1:25': [48123,   862013,   1679313,  9612,    890],
    }),
    ('LightGBM', {
        '1:5':  [91882,   304138,   1491,     18912,   4781],
        '1:10': [64422,   96797,    441922,   30364,   4635],
        '1:15': [51367,   86610,    2140674,  32920,   6351],
        '1:20': [33271,   168549,   2755784,  45051,   6759],
        '1:25': [35978,   443282,   2350658,  45813,   6410],
    }),
]

systems = ['autonuma', 'tpp', 'memtis', 'HybridTier', 'xTier']
system_labels = ['AutoNUMA', 'TPP', 'Memtis', 'HybridTier', 'xTier']

C_PRIMARY = '#2F73A3'  # overview primary flow
C_BLACK   = '#1A1A1A'  # overview borders/text
C_GRAY    = '#777777'
C_FILL    = '#C5DCE7'  # overview memory/eBPF fill
C_GRID    = '#D9D9D9'
C_AXIS    = '#000000'  # axes, ticks, labels, annotations

# Baselines in grayscale, HybridTier in the light overview blue,
# xTier (ours) in the primary blue
C_AUTONUMA = '#FFFFFF'
C_TPP      = '#BDBDBD'
C_MEMTIS   = C_BLACK
C_HYBRIDTIER = C_FILL
C_XTIER    = C_PRIMARY
colors = [C_AUTONUMA, C_TPP, C_MEMTIS, C_HYBRIDTIER, C_XTIER]


def normalize_migrations(data, ratios):
    """Per panel: normalize to minimum non-zero migration count.
    Zeros tracked separately (they can't be log-plotted)."""
    all_vals = [v for r in ratios for v in data[r]
                if v is not None and v > 0]
    fastest = min(all_vals)
    result, zero_marks = {}, {}
    for r in ratios:
        result[r], zero_marks[r] = [], []
        for v in data[r]:
            if v is None:
                result[r].append(None); zero_marks[r].append(False)
            elif v == 0:
                result[r].append(None); zero_marks[r].append(True)
            else:
                result[r].append(v / fastest); zero_marks[r].append(False)
    return result, zero_marks


n_ratios = len(ratios)
n_sys = len(systems)
width = 0.16
x = np.arange(n_ratios)

legend_elements = [
    Patch(facecolor=c, edgecolor=C_BLACK, linewidth=0.8, label=l)
    for c, l in zip(colors, system_labels)
]

fig, axes = plt.subplots(2, 3, figsize=(22, 7.5))
axes = axes.flatten()

for i, (name, data) in enumerate(mig_data):
    ax = axes[i]
    norm, zero_marks = normalize_migrations(data, ratios)

    for j, color in enumerate(colors):
        positions = x + (j - (n_sys - 1) / 2) * width
        for k, r in enumerate(ratios):
            v = norm[r][j]
            is_zero = zero_marks[r][j]
            pos = positions[k]
            if is_zero:
                # Mark zeros with '0' since they can't be log-plotted
                ax.text(pos, 0.5, '0', ha='center', va='bottom',
                        fontsize=16, color=C_AXIS, fontweight='bold',
                        zorder=4)
            elif v is not None:
                ax.bar(pos, v, width, color=color, edgecolor=C_BLACK,
                       linewidth=0.8, zorder=3)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(ratios)
    ax.set_xlabel(name, fontsize=28, labelpad=10)

    all_norm = [v for r in ratios for v in norm[r] if v is not None]
    ax.set_ylim(0.4, max(all_norm) * 3.0)

    ax.grid(axis='y', which='major', color=C_GRID, linestyle=':',
            linewidth=0.9, alpha=0.85, zorder=0)
    ax.grid(False, axis='y', which='minor')
    ax.tick_params(axis='y', which='minor', left=False, labelleft=False)
    ax.tick_params(axis='both', labelsize=20, colors=C_AXIS)
    ax.spines['left'].set_color(C_AXIS)
    ax.spines['bottom'].set_color(C_AXIS)

fig.legend(
    handles=legend_elements,
    loc='upper center',
    ncol=5,
    bbox_to_anchor=(0.5, 1.00),
    frameon=True,
    fancybox=False,
    framealpha=1.0,
    facecolor='white',
    edgecolor='#999999',
    fontsize=26,
    handletextpad=0.6,
    columnspacing=1.6
)

plt.tight_layout(rect=[0, 0, 1, 0.92])

# Single shared y-axis label, placed flush against the left column's
# tick labels (measured after layout so the gap is uniform)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
left_edge = min(
    ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).x0
    for ax in axes
)
fig.supylabel('# of page migrations (log scale)', fontsize=28,
              x=left_edge - 0.004, ha='right')

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 'migrations_2x3.png'), dpi=180, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'migrations_2x3.pdf'), bbox_inches='tight')
print("Done.")
