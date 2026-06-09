"""Phase 5 runner: results bundle -> RQ1 selective table + detection table + LaTeX values.

Consumes results/<bundle>.jsonl (written by the panel+baseline run) and emits:
  - results/rq1_main.json         (machine-readable metrics + CIs)
  - results/rq1_main.txt          (aligned table for inspection)
  - results/tab_main_values.tex   (rows to paste/input into tables/tab_main.tex)

Usage:
    .venv/bin/python scripts/run_experiments.py --bundle results/bundle.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.eval.bundle import best_baseline, evaluate_bundle, load_bundle
from phishproof.eval.report import latex_row, print_table

KEYS = ("AURC", "SelAcc80", "FPR80", "Cov99", "ECE")
# AURC reported x100 for readability (matches paper's ~3-5 scale); others as-is.
SCALES = {"AURC": 100.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    bundle = load_bundle(args.bundle)
    res = evaluate_bundle(bundle, n_boot=args.n_boot)
    methods, order = res["methods"], res["order"]
    baselines = order[1:]
    best = best_baseline(methods, baselines, "AURC", lower_better=True)

    args.out.mkdir(parents=True, exist_ok=True)
    table = print_table({m: methods[m] for m in order}, KEYS, SCALES)
    print(table)
    print(f"\nproposed = {order[0]} (bold); best baseline on AURC = {best} (underline)")
    print(f"detection (full coverage): "
          + "  ".join(f"{k}={v[0]:.3f}" for k, v in res["detection"].items()))

    (args.out / "rq1_main.txt").write_text(table + "\n")
    (args.out / "rq1_main.json").write_text(json.dumps(
        {"methods": methods, "detection": res["detection"], "best_baseline": best}, indent=2))

    tex_lines = []
    for m in order:
        name = f"\\textbf{{{m}}}" if m == order[0] else (f"\\underline{{{m}}}" if m == best else m)
        tex_lines.append(latex_row(name, methods[m], KEYS, SCALES))
    (args.out / "tab_main_values.tex").write_text("\n".join(tex_lines) + "\n")
    print(f"\n[ok] wrote {args.out}/rq1_main.json, rq1_main.txt, tab_main_values.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
