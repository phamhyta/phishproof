"""Format metric dicts into mean +/- 95% CI strings and LaTeX table rows."""

from __future__ import annotations


def fmt_ci(triple: tuple[float, float, float], scale: float = 1.0, decimals: int = 3) -> str:
    point, lo, hi = (v * scale for v in triple)
    half = (hi - lo) / 2
    return f"{point:.{decimals}f}$_{{\\pm{half:.{decimals}f}}}$"


def latex_row(name: str, metrics: dict[str, tuple[float, float, float]], keys, scales=None,
              decimals=3) -> str:
    scales = scales or {}
    cells = [fmt_ci(metrics[k], scales.get(k, 1.0), decimals) for k in keys]
    return f"{name} & " + " & ".join(cells) + r" \\"


def print_table(rows: dict[str, dict], keys, scales=None, decimals=3) -> str:
    """rows: {method_name: metrics_dict}. Returns a plain aligned text table + LaTeX."""
    scales = scales or {}
    lines = ["  ".join([f"{'method':14}"] + [f"{k:>16}" for k in keys])]
    for name, m in rows.items():
        cells = []
        for k in keys:
            p, lo, hi = (v * scales.get(k, 1.0) for v in m[k])
            cells.append(f"{p:>8.3f}[{lo:.3f},{hi:.3f}]"[:16].rjust(16))
        lines.append("  ".join([f"{name:14}"] + cells))
    return "\n".join(lines)
