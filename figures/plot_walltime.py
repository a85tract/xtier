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
wall_data = [
    ('XGBoost', {
        '1:5':  [517, 520, 704, 520, 588],
        '1:10': [614, 570, 762, 583, 588],
        '1:15': [637, 589, 764, 598, 591],
        '1:20': [619, 596, 783, 596, 589],
        '1:25': [621, 593, 773, 586, 586],
    }),
    ('BC', {
        '1:5':  [106, 97, 139, 95, 102],
        '1:10': [153, 132, 156, 132, 104],
        '1:15': [249, 135, 157, 134, 105],
        '1:20': [426, 151, 154, 136, 106],
        '1:25': [371, 158, 153, 136, 109],
    }),
    ('TC', {
        '1:5':  [549, 547, 667, 549, 615],
        '1:10': [638, 603, 715, 608, 613],
        '1:15': [627, 593, 699, 606, 631],
        '1:20': [657, 589, 717, 605, 632],
        '1:25': [665, 603, 724, 602, 612],
    }),
    ('BFS', {
        '1:5':  [92, 84, 113, 85, 88],
        '1:10': [135, 113, 127, 99, 99],
        '1:15': [190, 117, 128, 123, 100],
        '1:20': [225, 121, 130, 120, 92],
        '1:25': [273, 121, 129, 121, 91],
    }),
    ('PR', {
        '1:5':  [119, 107, 177, 104, 111],
        '1:10': [179, 135, 192, 138, 118],
        '1:15': [197, 158, 183, 156, 119],
        '1:20': [237, 176, 183, 176, 119],
        '1:25': [244, 178, 190, 178, 120],
    }),
    ('LightGBM', {
        '1:5':  [1265, 1260, 1603, 1255, 1412],
        '1:10': [1492, 1421, 1684, 1422, 1401],
        '1:15': [1468, 1470, 1730, 1438, 1421],
        '1:20': [1488, 1445, 1671, 1475, 1432],
        '1:25': [1481, 1607, 1731, 1464, 1411],
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


def normalize_walltime(data, ratios):
    """Per panel: normalize to slowest (worst) result -> faster = taller bar."""
    all_vals = [v for r in ratios for v in data[r] if v is not None]
    slowest = max(all_vals)
    return {r: [slowest / v if v is not None else None for v in data[r]]
            for r in ratios}


n_ratios = len(ratios)
n_sys = len(systems)
width = 0.16
x = np.arange(n_ratios)

legend_elements = [
    Patch(facecolor=c, edgecolor=C_BLACK, linewidth=0.8, label=l)
    for c, l in zip(colors, system_labels)
]

fig, axes = plt.subplots(2, 3, figsize=(22, 8))
axes = axes.flatten()

for i, (name, data) in enumerate(wall_data):
    ax = axes[i]
    norm = normalize_walltime(data, ratios)

    for j, color in enumerate(colors):
        positions = x + (j - (n_sys - 1) / 2) * width
        vals = [norm[r][j] if norm[r][j] is not None else 0 for r in ratios]
        ax.bar(positions, vals, width, color=color, edgecolor=C_BLACK,
               linewidth=0.8, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(ratios)
    ax.set_xlabel(name, fontsize=28, labelpad=10)

    all_norm = [v for r in ratios for v in norm[r] if v is not None]
    ax.set_ylim(0, max(all_norm) * 1.08)

    ax.grid(axis='y', color=C_GRID, linestyle=':', linewidth=0.9,
            alpha=0.85, zorder=0)
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
fig.supylabel('Normalized speedup', fontsize=28,
              x=left_edge - 0.004, ha='right')

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 'walltime_2x3_bar.png'), dpi=180, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'walltime_2x3_bar.pdf'), bbox_inches='tight')
print("Done.")
