"""R5.12 -- reason-fragility flip rates by GEA quintile on the larger panel (no API calls).

R5.12 asks us to "repeat flip rates by GEA quintile on the main panel or restrict RQ2". The
manuscript currently reports them on the smaller panel only (tab_rlwr), so RQ2 is the last
component claim still confined to that panel.

No new inference is needed. results/_or/bundle_adversarial_or.jsonl already stores, for each of
the 150 correctly-detected phishing pages and each attack, the clean GEA, the clean verdict and
the attacked verdict -- which is exactly what the quintile analysis consumes. The population
matches the smaller-panel run: both start from pages the clean system called phishing correctly.

Method follows scripts/run_rq2.py: sort pages by clean GEA, split into five equal groups, and
report the flip rate of the lowest-GEA quintile against the pooled remainder. A flip is a page
whose verdict changes under the perturbation while its label is held fixed, so it measures
reason fragility rather than a changed ground truth. Denominators here are 30 (Q1) and 120
(Q2-5), so every rate carries a Wilson interval; the smaller-panel table quotes bare rates on
~85 and ~340, and at n=30 a bare rate would be misleading.

Usage
    uv run scripts/revision_reason_fragility_or.py --out results/revision/t2_rlwr_or.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> list[float | None]:
    if n == 0:
        return [None, None]
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [round(100 * max(0.0, c - h), 1), round(100 * min(1.0, c + h), 1)]


def rate(k: int, n: int) -> dict:
    return {"flips": k, "n": n,
            "pct": round(100 * k / n, 1) if n else None,
            "ci95": wilson(k, n)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, action="append", default=None,
                    help="repeatable; the smaller panel keeps one file per attack")
    ap.add_argument("--clean-g", type=Path, default=Path("results/_or/clean_g.json"))
    ap.add_argument("--out", type=Path, default=Path("results/revision/t2_rlwr_or.json"))
    ap.add_argument("--tie-perms", type=int, default=500)
    args = ap.parse_args()

    bundles = args.bundle or [Path("results/_or/bundle_adversarial_or.jsonl")]
    rows = [json.loads(l) for b in bundles
            for l in b.read_text().splitlines() if l.strip()]
    clean_g = json.loads(args.clean_g.read_text()) if args.clean_g.exists() else {}

    by_attack: dict[str, list[dict]] = {}
    for r in rows:
        by_attack.setdefault(r["attack"], []).append(r)

    def g_of(r: dict, side: str) -> float:
        if side == "clean" and r["page_id"] in clean_g:
            return float(clean_g[r["page_id"]])
        v = r[side].get("G")
        return float(v) if v is not None else 1.0

    out: dict = {
        "source": [str(b) for b in bundles],
        "definition": ("flip = clean verdict != attacked verdict, label held fixed; "
                       "quintiles are over the CLEAN GEA, Q1 = lowest"),
        "attacks": {},
    }

    for attack, rs in sorted(by_attack.items()):
        gea = np.array([r["clean"]["gea"] for r in rs], dtype=float)
        flip = np.array([r["clean"]["verdict"] != r["attacked"]["verdict"] for r in rs],
                        dtype=float)
        # Quintiles over a TIED score are not well defined. On the larger panel the clean GEA
        # takes ten distinct values and the Q1 boundary falls inside a block of ~20 equal
        # scores, so which of them lands in Q1 is decided by input order, not by the score.
        # Break ties at random and repeat, so the reported split is an expectation over tie
        # orderings rather than an artefact of one of them.
        n_perm = args.tie_perms
        rngp = np.random.RandomState(0)
        q1_rates, rest_rates, pvals = [], [], []
        from scipy.stats import fisher_exact
        for _ in range(n_perm):
            order = np.lexsort((rngp.permutation(len(gea)), gea))
            gs = np.array_split(order, 5)
            i1, ir = gs[0], np.concatenate(gs[1:])
            a_, b_ = int(flip[i1].sum()), len(i1) - int(flip[i1].sum())
            c_, d_ = int(flip[ir].sum()), len(ir) - int(flip[ir].sum())
            q1_rates.append(flip[i1].mean())
            rest_rates.append(flip[ir].mean())
            pvals.append(fisher_exact([[a_, b_], [c_, d_]], alternative="greater")[1])
        # a single representative ordering, for the point estimate reported in the table
        order = np.lexsort((np.zeros(len(gea)), gea))
        groups = np.array_split(order, 5)
        q1 = groups[0]
        rest = np.concatenate(groups[1:])

        out["attacks"][attack] = {
            "n": len(rs),
            "overall": rate(int(flip.sum()), len(rs)),
            "q1": rate(int(flip[q1].sum()), len(q1)),
            "q2_5": rate(int(flip[rest].sum()), len(rest)),
            "quintiles": [
                {"gea_mean": round(float(gea[g].mean()), 3),
                 **rate(int(flip[g].sum()), len(g))}
                for g in groups
            ],
            "G_clean_mean": round(float(np.mean([g_of(r, "clean") for r in rs])), 3),
            "G_attacked_mean": round(float(np.mean([g_of(r, "attacked") for r in rs])), 3),
        }
        qa, ra, pa = np.array(q1_rates), np.array(rest_rates), np.array(pvals)
        out["attacks"][attack]["distinct_gea_values"] = int(len(set(np.round(gea, 9).tolist())))
        out["attacks"][attack]["q1_vs_rest_tie_robust"] = {
            "test": "Fisher exact, one-sided (Q1 more fragile), over random tie orderings",
            "tie_perms": n_perm,
            "q1_flip_pct": [round(100 * float(qa.mean()), 1),
                            [round(100 * float(np.percentile(qa, 2.5)), 1),
                             round(100 * float(np.percentile(qa, 97.5)), 1)]],
            "rest_flip_pct": [round(100 * float(ra.mean()), 1),
                              [round(100 * float(np.percentile(ra, 2.5)), 1),
                               round(100 * float(np.percentile(ra, 97.5)), 1)]],
            "p_median": round(float(np.median(pa)), 4),
            "p_95th": round(float(np.percentile(pa, 95)), 4),
            "frac_orderings_significant": round(float((pa < 0.05).mean()), 3),
        }

    text = json.dumps(out, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)

    def fmt(d: dict) -> str:
        return f"{d['pct']}% [{d['ci95'][0]}, {d['ci95'][1]}] n={d['n']}"

    print(f"{'attack':10} {'n':>4} {'overall':>8} {'Q1 flip [95% CI]':>26} "
          f"{'Q2-5 flip [95% CI]':>27} {'G shift':>14}")
    print("-" * 94)
    for a, s_ in out["attacks"].items():
        shift = f"{s_['G_clean_mean']:.2f}->{s_['G_attacked_mean']:.2f}"
        print(f"{a:10} {s_['n']:>4} {s_['overall']['pct']:>7.1f}% "
              f"{fmt(s_['q1']):>26} {fmt(s_['q2_5']):>27} {shift:>14}")
    print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
