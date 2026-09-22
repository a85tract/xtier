#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import csv
import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'phase_response.csv')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 22,
    'axes.labelsize': 25,
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

C_PRIMARY = '#2F73A3'
C_BLACK   = '#1A1A1A'
C_GRAY    = '#777777'
C_FILL    = '#C5DCE7'
C_GRID    = '#D9D9D9'
C_AXIS    = '#000000'
C_DEEP    = '#1A4A6B'

STYLE = OrderedDict([
    ('xTier',    (C_PRIMARY, '-',                3.2)),
    ('TPP',      (C_BLACK,   (0, (5, 2)),        2.5)),
    ('AutoNUMA', (C_GRAY,    (0, (1, 1.6)),      3.0)),
    ('HybridTier', (C_DEEP,    (0, (4, 2, 1, 2)),  2.5)),
    ('Memtis',   (C_BLACK,   '-',                1.8)),
])

T0, T1 = -230, 260
WINDOW = 60


def load(path):
    series = OrderedDict((k, ([], [])) for k in STYLE)
    with open(path) as f:
        for row in csv.DictReader(f):
            s = row['system']
            if s not in series:
                continue
            series[s][0].append(float(row['t_seconds']))
            series[s][1].append(float(row['hot_set_in_dram_pct']))
    missing = [k for k, v in series.items() if not v[0]]
    if missing:
        raise SystemExit(f'no rows for: {", ".join(missing)}')
    return series


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else HERE
    os.makedirs(out_dir, exist_ok=True)
    series = load(CSV)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(0, WINDOW, color=C_FILL, alpha=0.32, zorder=0)
    ax.axvline(0, color=C_BLACK, linestyle=(0, (4, 3)), linewidth=1.5, zorder=2)
    ax.text(4, 101, 'change in phase', fontsize=22, color=C_AXIS,
            va='top', ha='left')

    handles = []
    for label, (colour, ls, lw) in STYLE.items():
        t, y = series[label]
        ax.plot(t, y, color=colour, linestyle=ls, linewidth=lw,
                zorder=5 if label == 'xTier' else 3, solid_capstyle='round')
        handles.append(Line2D([0], [0], color=colour, linestyle=ls,
                              linewidth=lw, label=label))

    ax.set_xlabel('Time relative to phase change (s)', fontsize=25, labelpad=8)
    ax.set_ylabel('Hot set in DRAM (%)', fontsize=25, labelpad=8)
    ax.set_xlim(T0, T1)
    ax.set_ylim(0, 104)
    ax.set_xticks([-200, -100, 0, 100, 200])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(color=C_GRID, linestyle=':', linewidth=0.9, alpha=0.85, zorder=0)
    ax.tick_params(axis='both', labelsize=20, colors=C_AXIS)
    ax.spines['left'].set_color(C_AXIS)
    ax.spines['bottom'].set_color(C_AXIS)

    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.01),
              ncol=3, frameon=True, fancybox=False, framealpha=1.0,
              facecolor='white', edgecolor='#999999', fontsize=20,
              handlelength=2.8, handletextpad=0.6, borderpad=0.5,
              columnspacing=1.4)

    plt.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 180})):
        path = os.path.join(out_dir, f'phase_response.{ext}')
        plt.savefig(path, bbox_inches='tight', **kw)
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
