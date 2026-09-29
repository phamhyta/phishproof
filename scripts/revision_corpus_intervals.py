"""R5.14 -- bootstrap intervals for the cross-corpus rows (no API calls).

R5.14 asks for "paired uncertainty on the actual 70B outputs for Phishpedia/APWG and intervals
for TR-OP". The paired part is delivered by scripts/revision_matched.py, whose paired_vs_s and
paired_vs_gea blocks are computed on the replayed 70B outputs for both Phishpedia and APWG. What
was missing is the TR-OP side: tab_brands_out reports single numbers with no uncertainty at all,
which is what the reviewer objected to.

TR-OP and Phishpedia are different page sets, so their difference is not paired and a paired
resample would be wrong here. This reports an independent page-level bootstrap interval for each
row instead, which is the right uncertainty for "is the transferred operating point still where
the table says it is".

Both rows rank by the deployed score under the OPERATING calibrator (results/calibrator_or.json)
-- the fixed map fitted on the separate 710-page calibration split -- because that is what the
transfer claim is about: applying an unchanged map to a new corpus.

Usage
    uv run scripts/revision_corpus_intervals.py --out results/revision/t1_corpus_intervals.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VIS = "agent_c_vision"
N_BOOT = 2000


def deployed_score(rows: list[dict], cal: dict) -> np.ndarray:
    out = []
    for r in rows:
        ag = r["agents"]
        vlm = next((x for x in ag if x["id"] == VIS), None)
        conf = vlm["confidence"] if (vlm and vlm.get("confidence") is not None) else 0.5
        ref = vlm["verdict"] if vlm else r["verdict"]
        text = [x for x in ag if x["id"] != VIS]
        a = sum(1 for x in text if x["verdict"] == ref) / len(text) if text else 0.0
        out.append(float(np.interp(conf + 1e-3 * a, cal["x"], cal["y"])))
    return np.array(out)


def aurc(score: np.ndarray, correct: np.ndarray) -> float:
    o = np.argsort(-score, kind="stable")
    c = correct[o]
    return 100 * float(np.mean(1 - np.cumsum(c) / np.arange(1, len(c) + 1)))


def sel_acc(score: np.ndarray, correct: np.ndarray, cov: float = 0.80) -> float:
    o = np.argsort(-score, kind="stable")
    k = max(1, int(round(cov * len(o))))
    return 100 * float(np.mean(correct[o[:k]]))


def ece(p: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    p = np.clip(p, 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    t = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p <= edges[i + 1]) if i == bins - 1 else (p < edges[i + 1]))
        if m.sum():
            t += m.sum() / len(p) * abs(correct[m].mean() - p[m].mean())
    return float(t)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path("results/revision/t1_corpus_intervals.json"))
    ap.add_argument("--calibrator", type=Path, default=Path("results/calibrator_or.json"))
    args = ap.parse_args()

    cal = json.loads(args.calibrator.read_text())
    corpora = {
        "Phishpedia (in-distribution)": "results/revision/bundle_or_full.jsonl",
        "TR-OP (transferred calibrator)": "results/revision/bundle_trop_or_full.jsonl",
        "APWG (transferred calibrator)": "results/revision/bundle_apwg_or_full.jsonl",
    }
    rng = np.random.RandomState(0)
    out: dict = {"n_boot": N_BOOT, "calibrator": str(args.calibrator),
                 "note": ("independent page-level bootstrap per corpus; the corpora are "
                          "different page sets, so a paired resample across them would be "
                          "invalid. Paired comparisons within a corpus live in "
                          "results/revision/t1_*.json"),
                 "corpora": {}}

    for name, path in corpora.items():
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        rows = [r for r in rows if not r.get("replay_failed")]
        correct = np.array([r["verdict"] == r["label"] for r in rows], dtype=float)
        score = deployed_score(rows, cal)
        n = len(rows)
        idx = [rng.randint(0, n, n) for _ in range(N_BOOT)]
        stats = {
            "AURC": (aurc(score, correct), [aurc(score[i], correct[i]) for i in idx]),
            "SelAcc80": (sel_acc(score, correct), [sel_acc(score[i], correct[i]) for i in idx]),
            "ECE": (ece(score, correct), [ece(score[i], correct[i]) for i in idx]),
        }
        entry = {"n": n, "accuracy": round(100 * float(correct.mean()), 2)}
        for k, (point, boots) in stats.items():
            b = np.array(boots)
            entry[k] = {
                "point": round(point, 3 if k == "ECE" else 2),
                "ci95": [round(float(np.percentile(b, 2.5)), 3 if k == "ECE" else 2),
                         round(float(np.percentile(b, 97.5)), 3 if k == "ECE" else 2)],
            }
        out["corpora"][name] = entry

    text = json.dumps(out, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)

    print(f"{'corpus':34} {'n':>5} {'AURC [95% CI]':>24} {'SelAcc80 [95% CI]':>24} "
          f"{'ECE [95% CI]':>24}")
    print("-" * 114)
    for name, e in out["corpora"].items():
        def f(k, nd):
            d = e[k]
            return f"{d['point']:.{nd}f} [{d['ci95'][0]:.{nd}f}, {d['ci95'][1]:.{nd}f}]"
        print(f"{name:34} {e['n']:>5} {f('AURC', 2):>24} {f('SelAcc80', 1):>24} "
              f"{f('ECE', 3):>24}")
    print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
