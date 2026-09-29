#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""Duplicate-aware sensitivity of the deployed score on the 70B Phishpedia panel, emitted
to a committed artifact so exp_setup's overlap paragraph is traceable. Reports the
cross-fitted calibrated AURC under (a) the reported default of duplicate-group-disjoint
folds, (b) random folds, and (c) one representative row per duplicate group, with that
subset's accuracy. Writes results/revision_v2/t10/dedup_70b.json.

Usage: uv run scripts/revision_dedup_70b.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

ELIG = Path("results/revision_v2/t8/eligibility_phishpedia.jsonl")
GROUPS = Path("results/revision/t5_grouped_split.json")
OUT = Path("results/revision_v2/t10/dedup_70b.json")
EPS = 1e-3


def main():
    rows = [json.loads(l) for l in ELIG.read_text().splitlines() if l.strip()]
    n = len(rows)
    pids = [r["page_id"] for r in rows]
    pr = np.argsort(np.argsort(np.array(pids))).astype(float)
    cor = np.array([r["verdict"] == r["label"] if r["verdict"] else False for r in rows], float)
    valid = np.array([r["complete"]["valid_inference"] for r in rows])
    conf = np.array([r["conf_vlm"] if r["conf_vlm"] is not None else np.nan for r in rows])
    a_text = np.array([r["a_text"] for r in rows])
    s = np.where(np.isnan(conf), np.nan, conf + EPS * a_text)
    gmap = json.loads(GROUPS.read_text())["test_group_map"]
    grp = np.array([gmap.get(p, f"solo::{p}") for p in pids])
    el = valid & ~np.isnan(s)

    def aurc(sc, mask, prank):
        idx = np.where(mask)[0]
        o = idx[np.lexsort((prank[idx], -sc[idx]))]
        return 100.0 * float(np.mean(np.cumsum(1 - cor[o]) / np.arange(1, len(o) + 1)))

    def crossfit(groups, seed=0):
        rng = np.random.RandomState(seed)
        if groups is None:
            folds = np.array_split(rng.permutation(n), 5)
        else:
            uniq = np.array(sorted(set(groups.tolist())))
            asg = {g: i % 5 for i, g in enumerate(rng.permutation(uniq))}
            which = np.array([asg[g] for g in groups])
            folds = [np.where(which == i)[0] for i in range(5)]
        out = np.full(n, np.nan)
        for i in range(5):
            te = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i])
            tr = tr[el[tr]]; te2 = te[el[te]]
            if len(tr) < 10 or len(te2) == 0:
                out[te2] = s[te2]; continue
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            ir.fit(s[tr], cor[tr]); out[te2] = ir.predict(s[te2])
        return out

    cal_grp = crossfit(grp)
    cal_rand = crossfit(None)
    grp_aurc = aurc(np.where(np.isnan(cal_grp), 0, cal_grp), el & ~np.isnan(cal_grp), pr)
    rand_aurc = aurc(np.where(np.isnan(cal_rand), 0, cal_rand), el & ~np.isnan(cal_rand), pr)

    seen, keep = set(), []
    for i, g in enumerate(grp):
        if g not in seen:
            seen.add(g); keep.append(i)
    keep = np.array(keep)
    kp = np.argsort(np.argsort(np.array([pids[i] for i in keep]))).astype(float)
    elk = el[keep]
    dedup_aurc = aurc(s[keep], elk, kp) if elk.any() else None
    dedup_acc = round(100 * float(cor[keep].mean()), 2)

    out = {"panel": "70B", "corpus": "phishpedia",
           "note": "cross-fitted calibrated AURC (attainable coverage, page-id ties). The "
                   "reported default uses duplicate-group-disjoint folds.",
           "group_disjoint_calibrated_aurc": round(grp_aurc, 2),
           "random_fold_calibrated_aurc": round(rand_aurc, 2),
           "dedup_n_groups": len(keep),
           "dedup_raw_aurc": round(dedup_aurc, 2) if dedup_aurc is not None else None,
           "dedup_accuracy_pct": dedup_acc}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[ok] wrote {OUT}")


if __name__ == "__main__":
    main()
