<!-- SPDX-License-Identifier: Apache-2.0 -->
# Figures

Every figure regenerates from what is committed here. No network, no
absolute paths, nothing to fetch. Each script takes the output directory as its
one argument and writes both a PNG and a PDF.

    pip install -r requirements.txt
    python plot_phase_response.py /tmp/figs

With no argument a script writes next to itself.

| Script | Outputs | Shows | Input |
|---|---|---|---|
| `plot_active_burst.py` | `active_burst.{png,pdf}` | Access-recency decay, with a fitted curve | in-script |
| `plot_per_epoch_latency.py` | `per_epoch_latency.{png,pdf}` | Per-epoch decision latency, in-kernel vs userspace, idle vs contended | in-script |
| `plot_walltime.py` | `walltime_2x3_bar.{png,pdf}` | Wall-clock time across 6 workloads x 5 DRAM:CXL ratios | in-script |
| `plot_migration_counts.py` | `migrations_2x3.{png,pdf}` | Migration *counts*, 6 workloads x 5 ratios, log scale | in-script |
| `migrations/plot_cumulative_migrations.py` | `cumulative_migrations.{png,pdf}` | Cumulative migrations over time, 5 systems | `migrations/migrations_*.csv` |
| `plot_phase_response.py` | `phase_response.{png,pdf}` | Hot-set coverage through a phase change, 5 systems | `phase_response.csv` |

Scripts that read data resolve it relative to their own file, so the data files
have to stay beside the script that reads them: `phase_response.csv` here, and
`migrations_*.csv` inside `migrations/`.

The four scripts marked *in-script* carry their measured values as literals
rather than reading a CSV, so the numbers in the paper and the numbers plotted
are the same object. The run logs they came from are not distributed.

Output PNGs and PDFs are gitignored. The inputs are what is tracked.
