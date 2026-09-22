#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

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

C_PRIMARY = '#2F73A3'  # overview primary flow
C_BLACK   = '#1A1A1A'  # overview borders/text
C_GRAY    = '#777777'
C_FILL    = '#C5DCE7'  # overview memory/eBPF fill
C_GRID    = '#D9D9D9'
C_AXIS    = '#000000'  # axes, ticks, labels, annotations
C_DEEP    = '#1A4A6B'

# (csv name, label, colour, linestyle, linewidth) -- same line vocabulary
# as plot_phase_response.py so the two line figures read together
SYSTEMS = [
    ("autonuma",   "AutoNUMA",  C_GRAY,    (0, (1.5, 1.5)),   2.5),
    ("tpp",        "TPP",       C_BLACK,   (0, (5, 2)),       2.5),
    ("memtis",     "Memtis",    C_BLACK,   '-',               1.8),
    ("hybridtier", "HybridTier",  C_DEEP,    (0, (4, 2, 1, 2)), 2.5),
    ("xtier",      "xTier",     C_PRIMARY, '-',               3.2),
]

POWER = 0.35


def load(name):
    t, m = [], []
    with open(os.path.join(HERE, f"migrations_{name}.csv")) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            t.append(float(row[0]))
            m.append(int(row[1]))
    return t, m


def truncate(t, m, t_max):
    out_t, out_m = [], []
    for ti, mi in zip(t, m):
        if ti > t_max:
            break
        out_t.append(ti)
        out_m.append(mi)
    return out_t, out_m


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else HERE
    os.makedirs(out_dir, exist_ok=True)

    series = {name: load(name) for name, *_ in SYSTEMS}
    t_end = series["xtier"][0][-1]

    fig, ax = plt.subplots(figsize=(10, 5))

    for name, label, color, ls, lw in SYSTEMS:
        t_all, m_all = truncate(*series[name], t_end)
        ax.plot(t_all, m_all, label=label, color=color, linewidth=lw,
                linestyle=ls, zorder=5 if name == "xtier" else 3,
                solid_capstyle='round')

    ax.set_yscale("function", functions=(
        lambda x: np.power(np.maximum(x, 0), POWER),
        lambda x: np.power(np.maximum(x, 0), 1.0 / POWER),
    ))
    ax.set_xlabel("Time", fontsize=28, labelpad=8)
    ax.set_ylabel("Cumulative migrations", fontsize=28, labelpad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, t_end)

    ax.tick_params(axis='both', colors=C_AXIS)
    ax.spines['left'].set_color(C_AXIS)
    ax.spines['bottom'].set_color(C_AXIS)

    ax.legend(
        loc='center right',
        bbox_to_anchor=(1.0, 0.67),
        labelspacing=0.35,
        borderpad=0.4,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor='white',
        edgecolor='#999999',
        fontsize=20,
        handlelength=2.8,
        handletextpad=0.6
    )

    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 180})):
        out = os.path.join(out_dir, f"cumulative_migrations.{ext}")
        fig.savefig(out, bbox_inches="tight", **kw)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
