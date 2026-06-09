"""RQ1 robustness check: is PhishProof's AURC advantage over the faithful B6 consolidator
directionally consistent across data partitions, even though the paired-bootstrap CI at
n=998 includes zero?

Uses bundle_final.jsonl (faithful B6). AURC is ranking-based (calibration-independent), so
re-partitioning the test set alone is valid. Reports the full-998 point estimate, a
stratified 5-fold sign count, and a 70%-subsample sign rate (subsamples overlap, so this
shows robustness of the point estimate, NOT an independent significance test).

    .venv/bin/python scripts/rq1_aurc_consistency.py
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishproof.eval.metrics import aurc

rows = [json.loads(l) for l in open("results/bundle_final.jsonl") if l.strip()]
pp = [r["gea"] for r in rows]
b6 = [r["baselines"]["B6"] for r in rows]            # faithful consolidator
cor = [1 if r["verdict"] == r["label"] else 0 for r in rows]  # shared panel verdict


def A(scores, idx):
    return aurc([scores[i] for i in idx], [cor[i] for i in idx]) * 100


full = list(range(len(rows)))
print(f"Full {len(rows)}: AURC_PP={A(pp,full):.2f} AURC_B6={A(b6,full):.2f} "
      f"delta={A(pp,full)-A(b6,full):+.2f}")


def strat_folds(k, seed):
    rng = random.Random(seed)
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["label"], []).append(i)
    folds = [[] for _ in range(k)]
    for ix in by.values():
        ix = ix[:]
        rng.shuffle(ix)
        for j, i in enumerate(ix):
            folds[j % k].append(i)
    return folds


wins = sum(A(pp, f) < A(b6, f) for f in strat_folds(5, 7))
print(f"Stratified 5-fold: PP<B6 in {wins}/5 folds")

deltas, w = [], 0
for s in range(21):
    sub = random.Random(100 + s).sample(full, int(0.70 * len(rows)))
    d = A(pp, sub) - A(b6, sub)
    deltas.append(d)
    w += d < 0
print(f"21x 70%-subsamples: PP<B6 in {w}/21; mean delta={st.mean(deltas):+.2f}")
print("Note: subsamples overlap -> robustness of point estimate, not an independent test.")
