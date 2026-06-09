#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "numpy"]
# ///
"""Generate the experiment figures for PhishProof / GEA as PNGs (REAL numbers).

Run (from the paper root):
    uv run --with matplotlib --with numpy python figures/gen_experiment_figs.py

All numbers below are the measured results on the 998-page test split
(results/bundle_final.jsonl), kept consistent with tab:main and tab:detect:
- fig_riskcoverage: selective risk (= 100 - selective accuracy, %) vs coverage. The
  panel methods meet at coverage=1.0 at the shared full-coverage panel error 8.1%
  (GEA does not change the majority label, only the abstention order). The 80%-coverage
  points reproduce SelAcc@80 of tab:main: GEA 3.0 (97.0), B5 6.4 (93.6), B6 5.4 (94.6),
  B1 7.6 (92.4). The standalone detector D3 (PhishLLM) has no trust ranking and a 16.3%
  full-coverage error, so it is not plotted here and is discussed in prose instead.
- fig_pareto: cost-quality. x = per-page model calls (single-model=1, M=3 panel=3);
  y = AURC (lower=better) from tab:main (B1 8.6, B2 5.7, B4 8.5, B5 3.7, B6 3.0, GEA 2.3).
  At the 3-call panel budget GEA has the lowest AURC. Self-consistency (B3, 5 calls) was
  not run and is omitted. Every point is an analytical call count or a tab:main cell.
fig_diagnostic and fig_sensitivity are NOT generated: the per-decile reliability/flip
diagram and the M/threshold sweeps were not measured on the real benchmark.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import use_paper_style, figure, savefig, COLORS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

use_paper_style()
OUT = Path(__file__).resolve().parent

BLUE   = COLORS["primary"]    # PhishProof / GEA (proposed) -- hero series
RED    = COLORS["ablation"]   # single-model confidence baseline (B1)
GREEN  = COLORS["good"]       # label-agreement baseline (B5)
ORANGE = COLORS["secondary"]  # MultiPhishGuard baseline (B6, strongest)
SLATE  = COLORS["anno"]       # guides / reference

# ================================================================ RQ1: risk-coverage
cov = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
gea = np.array([2.6, 2.5, 2.7, 3.0, 4.2, 8.1])   # PhishProof (proposed)
b6  = np.array([2.2, 3.7, 5.4, 5.4, 6.1, 8.1])   # B6 MultiPhishGuard (strongest baseline)
b5  = np.array([3.4, 4.8, 6.3, 6.4, 7.6, 8.1])   # B5 label-agreement
b1  = np.array([9.0, 7.7, 7.3, 7.6, 7.9, 8.1])   # B1 single-model confidence
COV_MAIN = 0.80

fig = figure(width=3.45, ratio=0.62)
ax = fig.add_subplot(111)
ax.axvline(COV_MAIN, color=SLATE, lw=0.9, ls=(0, (4, 3)), alpha=0.7, zorder=1)
ax.plot(cov, b1, "o-", color=RED,    ms=4, label="B1 confidence",
        markeredgecolor="white", markeredgewidth=0.4, zorder=3)
ax.plot(cov, b5, "D-", color=GREEN,  ms=3.6, label="B5 label-agreement",
        markeredgecolor="white", markeredgewidth=0.4, zorder=3)
ax.plot(cov, b6, "^-", color=ORANGE, ms=4, label="B6 MultiPhishGuard",
        markeredgecolor="white", markeredgewidth=0.4, zorder=3)
ax.plot(cov, gea, "s-", color=BLUE,  ms=4.2, label="PhishProof (GEA)",
        markeredgecolor="white", markeredgewidth=0.4, zorder=4)
ax.plot([COV_MAIN], [3.0], marker="o", ms=6.5, mfc="none", mec=BLUE, mew=1.3, zorder=5)
ax.set_xlabel("coverage (fraction of pages acted on)")
ax.set_ylabel("selective risk (%)")
ax.set_xlim(0.47, 1.02)
ax.set_ylim(0, 9.6)
ax.annotate("shared\nfull-cov. error", xy=(1.0, 8.15),
            xytext=(0.88, 6.0), fontsize=6.2, color=SLATE, ha="center", va="top",
            arrowprops=dict(arrowstyle="->", color=SLATE, lw=0.7, shrinkA=1, shrinkB=3))
ax.annotate(r"@80% cov.: $3.0$ vs $6.4$", xy=(COV_MAIN, 3.0),
            xytext=(0.485, 0.55), fontsize=6.5, color=BLUE, ha="left", va="bottom")
ax.legend(loc="upper left", frameon=False, handlelength=1.4, fontsize=6.6,
          handletextpad=0.5, labelspacing=0.22, borderaxespad=0.25, ncol=2,
          columnspacing=1.0)
fig.tight_layout()
savefig(fig, OUT / "fig_riskcoverage.png")
plt.close(fig)

# ================================================================ cost-quality Pareto (AURC)
# x = per-page model calls (analytical); y = AURC (lower = better) from tab:main.
methods = [
    # name,             calls, aurc, color,  marker, ms,  dx,    dy,    ha,      va
    ("B1 confidence",     1, 8.6, RED,    "o", 4.2, 0.14,  0.0,  "left",  "center"),
    ("B2 verbalized",     1, 5.7, COLORS["neutral"], "o", 4.2, 0.14, 0.0, "left", "center"),
    ("B4 majority",       3, 8.5, COLORS["neutral"], "^", 4.4, 0.16, 0.0, "left", "center"),
    ("B5 label-agree.",   3, 3.7, GREEN,  "D", 4.0, 0.16,  0.30, "left",  "bottom"),
    ("B6 MultiPhishGuard",3, 3.0, ORANGE, "^", 4.4, 0.16, -0.34, "left",  "top"),
    ("PhishProof (GEA)",  3, 2.3, BLUE,   "*", 9.5, -0.14, 0.0,  "right", "center"),
]

fig = figure(width=3.45, ratio=0.74)
ax = fig.add_subplot(111)
# Pareto frontier (minimise calls AND AURC): B2 (1, 5.7) -> GEA (3, 2.3)
ax.plot([1, 3], [5.7, 2.3], ls=(0, (5, 3)), color=BLUE, lw=1.0, alpha=0.55, zorder=2)
# "same cost, lower AURC" gap at the 3-call panel budget (GEA vs B5)
ax.annotate("", xy=(3, 2.3), xytext=(3, 3.7),
            arrowprops=dict(arrowstyle="<->", color=SLATE, lw=0.8, shrinkA=2, shrinkB=2),
            zorder=2)
ax.text(3.12, 3.0, "same cost,\nlower AURC", fontsize=6.4, color=SLATE,
        ha="left", va="center")
for name, calls, a, color, mk, ms, dx, dy, ha, va in methods:
    hero = name.startswith("PhishProof")
    ax.plot([calls], [a], marker=mk, ms=ms, color=color,
            markeredgecolor="white", markeredgewidth=0.5, zorder=6 if hero else 4)
    ax.annotate(name, xy=(calls, a), xytext=(calls + dx, a + dy),
                fontsize=7.0 if hero else 6.4, color=color, ha=ha, va=va,
                fontweight="bold" if hero else "normal")
ax.set_xlabel("model calls per page (analytical)")
ax.set_ylabel(r"AURC ($\downarrow$, $\times100$)")
ax.set_xlim(0.5, 4.2)
ax.set_ylim(1.6, 9.3)
ax.set_xticks([1, 2, 3, 4])
ax.text(0.62, 2.1, "better", fontsize=6.6, color=SLATE, style="italic", ha="left")
ax.annotate("", xy=(0.62, 1.85), xytext=(1.05, 2.5),
            arrowprops=dict(arrowstyle="->", color=SLATE, lw=0.7))
fig.tight_layout()
savefig(fig, OUT / "fig_pareto.png")
plt.close(fig)
