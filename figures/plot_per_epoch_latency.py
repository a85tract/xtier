#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
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

categories = ['Userspace\n(Idle)', 'Userspace\n(Contended)',
              'Kernel\n(Idle)',    'Kernel\n(Contended)']

p50   = [0.11, 0.17, 0.05, 0.25]
p99   = [0.37, 9.63, 0.05, 0.29]
p99_9 = [0.45, 34.7, 0.05, 0.30]

DEADLINE = 10.0

C_PRIMARY = '#2F73A3'  # overview primary flow
C_BLACK   = '#1A1A1A'  # overview borders/text
C_GRAY    = '#777777'
C_FILL    = '#C5DCE7'  # overview memory/eBPF fill
C_GRID    = '#D9D9D9'
C_AXIS    = '#000000'  # axes, ticks, labels, annotations

C_P50  = C_FILL
C_P99  = C_PRIMARY
C_P999 = C_BLACK

C_DEADLINE = C_BLACK

fig, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(len(categories))
width = 0.26

ax.bar(x - width, p50,   width, color=C_P50,  edgecolor=C_BLACK,
       linewidth=0.8, label='p50 (median)', zorder=3)
ax.bar(x,         p99,   width, color=C_P99,  edgecolor=C_BLACK,
       linewidth=0.8, label='p99', zorder=3)
ax.bar(x + width, p99_9, width, color=C_P999, edgecolor=C_BLACK,
       linewidth=0.8, label='p99.9', zorder=3)

ax.axhline(DEADLINE, color=C_DEADLINE, linestyle=(0, (4, 3)),
           linewidth=1.5, zorder=2)

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel('Decision latency (ms)', fontsize=28, labelpad=8)

ax.set_ylim(0.03, 100)

ax.grid(axis='y', which='major', color=C_GRID, linestyle=':',
        linewidth=0.9, alpha=0.85, zorder=0)
ax.grid(False, axis='y', which='minor')
ax.tick_params(axis='y', which='minor', left=False, labelleft=False)
ax.tick_params(axis='both', labelsize=20, colors=C_AXIS)
ax.spines['left'].set_color(C_AXIS)
ax.spines['bottom'].set_color(C_AXIS)

legend_elements = [
    Patch(facecolor=C_P50,  edgecolor=C_BLACK, linewidth=0.8, label='p50 (median)'),
    Patch(facecolor=C_P99,  edgecolor=C_BLACK, linewidth=0.8, label='p99'),
    Patch(facecolor=C_P999, edgecolor=C_BLACK, linewidth=0.8, label='p99.9'),
    Line2D([0], [0], color=C_DEADLINE, linestyle=(0, (4, 3)), linewidth=1.5,
           label='Epoch deadline'),
]
ax.legend(
    handles=legend_elements,
    loc='upper right',
    frameon=True,
    fancybox=False,
    framealpha=1.0,
    facecolor='white',
    edgecolor='#999999',
    fontsize=26,
    ncol=1,
    handletextpad=0.6
)

plt.tight_layout()
out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 'per_epoch_latency.png'), dpi=180, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'per_epoch_latency.pdf'), bbox_inches='tight')
print("Done.")
