"""Reusable matplotlib style for publication-quality experiment figures.

Adapted from the experiment-results-plotter skill's paper_style module for the
PhishProof paper. Targets the ACM acmart sigconf double-column layout. The
proposed method (PhishProof / GEA) always gets COLORS["primary"]; baselines stay
muted so the eye lands on the hero curve before reading the legend. ``savefig``
writes PNG at 300 dpi (matches the paper's existing figure pipeline).
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

COL_W = 3.45    # single \columnwidth (in) for acmart sigconf
TEXT_W = 7.00   # \textwidth for figure*
GOLDEN = (1 + 5 ** 0.5) / 2

COLORS = {
    "primary":   "#1f4e9d",  # PROPOSED METHOD (PhishProof / GEA) -- hero
    "primary2":  "#2e86c1",
    "secondary": "#d77b1f",  # majority-vote baseline (B4)
    "ablation":  "#c0392b",  # single-model confidence baseline (B1) / ablation
    "ablation2": "#7d3c98",
    "good":      "#1e8449",  # label-agreement baseline (B5) / quality axis
    "neutral":   "#566573",  # faded baselines / self-consistency
    "muted":     "#95a5a6",
    "cost":      "#c0392b",  # cost axis on dual-axis plots
    "bound":     "#d62728",
    "bound_fill":"#f5b7b1",
    "anno":      "#34495e",  # guides / annotation text
    "grid":      "#d5d8dc",
}


def use_paper_style() -> None:
    """Apply paper rcParams. Idempotent."""
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Liberation Serif",
                               "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset":   "stix",
        "font.size":          9,
        "axes.titlesize":     9,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    7.5,
        "legend.title_fontsize": 8,
        "axes.linewidth":     0.8,
        "axes.edgecolor":     "#2c3e50",
        "axes.labelcolor":    "#2c3e50",
        "axes.titlepad":      4.0,
        "axes.labelpad":      3.0,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.color":        "#2c3e50",
        "ytick.color":        "#2c3e50",
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.size":   3.0,
        "ytick.major.size":   3.0,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "axes.grid":          True,
        "grid.color":         COLORS["grid"],
        "grid.linewidth":     0.5,
        "grid.alpha":         0.8,
        "axes.axisbelow":     True,
        "lines.linewidth":    1.7,
        "lines.markersize":   4.5,
        "lines.markeredgewidth": 0.0,
        "legend.frameon":     False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.3,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "axes.prop_cycle":    cycler(color=[
            COLORS["primary"], COLORS["secondary"], COLORS["ablation"],
            COLORS["good"], COLORS["neutral"], COLORS["ablation2"],
        ]),
    })


HEIGHT_SCALE = 0.6


def figure(width: float = COL_W, ratio: float = HEIGHT_SCALE / GOLDEN, **kw):
    """Tight figure sized for paper inclusion. ``ratio`` is height/width."""
    return plt.figure(figsize=(width, width * ratio), **kw)


def savefig(fig, path) -> None:
    """Save as 300-dpi PNG (adds .png if missing)."""
    path = str(path)
    if not path.endswith(".png"):
        path = path + ".png"
    fig.savefig(path)
    print(f"[paper_style] wrote {path}")
