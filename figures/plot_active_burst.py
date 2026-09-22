#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial'],
    'font.size': 22,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
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

xgb_pts = np.array([[0, 1, 10, 25, 50, 100],
                    [100, 58, 42, 33, 29, 27]], dtype=float)
bc_pts  = np.array([[0, 1, 10, 25, 50, 100],
                    [100, 34, 25, 22, 21, 20]], dtype=float)
bfs_pts = np.array([[0, 1, 10, 25, 50, 100],
                    [100, 38, 28, 25, 23, 22]], dtype=float)


def two_exp(t, drop_to, asymptote, fast_tau, slow_tau):
    fast = (100 - drop_to) * np.exp(-t / fast_tau)
    slow = (drop_to - asymptote) * np.exp(-t / slow_tau)
    return asymptote + fast + slow


def fit_curve(pts, label):
    t_obs, y_obs = pts
    p0 = [y_obs[1], y_obs[-1], 0.5, 25]
    bounds = ([1, 1, 0.05, 5], [99, 99, 5, 200])
    popt, _ = curve_fit(two_exp, t_obs, y_obs, p0=p0, bounds=bounds, maxfev=10000)
    print(f"{label}: drop_to={popt[0]:.1f}, asymptote={popt[1]:.1f}, "
          f"fast_tau={popt[2]:.2f}, slow_tau={popt[3]:.2f}")
    return popt


t = np.linspace(0, 100, 1000)

xgb_params = fit_curve(xgb_pts, "XGBoost")
bc_params  = fit_curve(bc_pts,  "BC")
bfs_params = fit_curve(bfs_pts, "BFS")

xgb = two_exp(t, *xgb_params)
bc  = two_exp(t, *bc_params)
bfs = two_exp(t, *bfs_params)

C_PRIMARY = '#2F73A3'  # overview primary flow
C_BLACK   = '#1A1A1A'  # overview borders/text
C_GRAY    = '#777777'
C_FILL    = '#C5DCE7'  # overview memory/eBPF fill
C_GRID    = '#D9D9D9'
C_AXIS    = '#000000'  # axes, ticks, labels, annotations

fig, ax = plt.subplots(figsize=(10, 5))

# Decision window
ax.axvspan(0, 10, color=C_FILL, alpha=0.32, zorder=0)

ax.plot(t, xgb, color=C_PRIMARY, linewidth=3.0,
        linestyle='-', label='XGBoost', zorder=3)
ax.plot(t, bc, color=C_BLACK, linewidth=2.6,
        linestyle='--', label='BC', zorder=3)
ax.plot(t, bfs, color=C_GRAY, linewidth=2.6,
        linestyle='-.', label='BFS', zorder=3)

ax.axvline(
    10, color=C_BLACK, linestyle=(0, (4, 3)),
    linewidth=1.5, zorder=2
)
ax.text(
    11.5, 95, '10 ms,\n>50% pages decayed',
    fontsize=28, color=C_AXIS, va='top', ha='left'
)

# x-axis label omitted; described in the paper caption:
# "Page hot-residency decay across three workloads"
ax.set_ylabel('Pages still active (%)',
              fontsize=28, labelpad=8)

ax.set_xlim(0, 100)
ax.set_ylim(0, 102)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_yticks([0, 20, 40, 60, 80, 100])

ax.grid(
    color=C_GRID, linestyle=':', linewidth=0.9,
    alpha=0.85, zorder=0
)
ax.tick_params(axis='both', labelsize=20, colors=C_AXIS)
ax.spines['left'].set_color(C_AXIS)
ax.spines['bottom'].set_color(C_AXIS)

ax.legend(
    loc='upper right',
    frameon=True,
    fancybox=False,
    framealpha=1.0,
    facecolor='white',
    edgecolor='#999999',
    fontsize=26
)
plt.tight_layout()
out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 'active_burst.png'), dpi=180, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'active_burst.pdf'), bbox_inches='tight')
print("Done.")
