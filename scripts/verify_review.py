"""One-off verification of the numbers a pre-submission review flagged.
Reproduces everything directly from the result bundles. Read-only.

    .venv/bin/python scripts/verify_review.py
"""
from __future__ import annotations
import json, statistics as st, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishproof.eval.metrics import (
    aurc, selective_accuracy_at_coverage, coverage_at_selective_accuracy, ece,
)
from phishproof.calibration.isotonic import IsotonicCalibrator

R = lambda p: [json.loads(l) for l in open(p) if l.strip()]


def hr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

# ───────────────────────── T1.4  B6 on evaded pages ─────────────────────────
hr("T1.4  B6 / PhishProof confidence on EVADED pages (per attack)")
for atk in ["cloak", "occlude", "both"]:
    rows = R(f"results/bundle_adv_{atk}.jsonl")
    # evaded = label phish, clean caught (verdict phish), attack flips to benign
    ev = [r for r in rows if r["label"] == "phish"
          and r["clean"]["verdict"] == "phish"
          and r["attacked"]["verdict"] == "benign"]
    if not ev:
        print(f"{atk:8s}: no evaded"); continue
    b6 = st.mean(e["attacked"]["b6"] for e in ev if e["attacked"]["b6"] is not None)
    pp = st.mean(e["attacked"]["trust"] for e in ev)
    b1 = st.mean(e["attacked"]["b1"] for e in ev if e["attacked"]["b1"] is not None)
    print(f"{atk:8s}: n_evaded={len(ev):3d}  B6_mean={b6:.3f}  "
          f"PhishProof_trust={pp:.3f}  B1_mean={b1:.3f}")
print("paper claims: B6 ≈0.68, PhishProof ≈0.58 on evaded")

# ───────────────────────── T1.3  paired ΔAURC PP vs B6 ──────────────────────
hr("T1.3  paired ΔAURC  PhishProof − B6  (bundle_D, 4020)")
rows = R("results/bundle_D.jsonl")
pp = np.array([r["gea"] for r in rows])
b6 = np.array([r["baselines"]["B6"] for r in rows])
cor = np.array([1 if r["verdict"] == r["label"] else 0 for r in rows])
n = len(rows)
A = lambda s, idx: aurc(s[idx], cor[idx]) * 100
point_pp, point_b6 = A(pp, np.arange(n)), A(b6, np.arange(n))
print(f"point AURC: PP={point_pp:.3f}  B6={point_b6:.3f}  point-diff={point_pp-point_b6:+.3f}")
rng = np.random.default_rng(0)
deltas = []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    deltas.append(A(pp, idx) - A(b6, idx))
deltas = np.array(deltas)
lo, hi = np.percentile(deltas, [2.5, 97.5])
p_two = 2 * min((deltas >= 0).mean(), (deltas <= 0).mean())
print(f"paired bootstrap mean Δ={deltas.mean():+.3f}  95%CI=[{lo:+.2f},{hi:+.2f}]  p≈{p_two:.2f}")
print("paper claims: Δ −0.4, CI [−1.5,+0.8], p=0.44  (point-diff −0.76)")

# ───────────────────────── T1.2  5-seed cross-split ─────────────────────────
hr("T1.2  5-way stratified 75/25 re-split, refit isotonic (bundle_all, 1330)")
rows = R("results/bundle_all.jsonl")
gea = np.array([r["gea"] for r in rows])
cor = np.array([1 if r["verdict"] == r["label"] else 0 for r in rows])
lab = np.array([r["label"] for r in rows])
idx_by = {L: np.where(lab == L)[0] for L in set(lab)}
res = {m: [] for m in ["AURC", "SelAcc80", "ECE", "Cov99", "n_test"]}
for seed in range(5):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for L, ix in idx_by.items():
        ix = ix.copy(); rng.shuffle(ix)
        k = int(round(0.75 * len(ix)))
        tr += list(ix[:k]); te += list(ix[k:])
    tr, te = np.array(tr), np.array(te)
    cal = IsotonicCalibrator().fit(list(gea[tr]), list(cor[tr].astype(bool)))
    trust_te = np.array(cal.predict(list(gea[te])))
    res["AURC"].append(aurc(gea[te], cor[te]) * 100)
    res["SelAcc80"].append(selective_accuracy_at_coverage(trust_te, cor[te], 0.8) * 100)
    res["ECE"].append(ece(trust_te, cor[te]))
    res["Cov99"].append(coverage_at_selective_accuracy(trust_te, cor[te], 0.99))
    res["n_test"].append(len(te))
for m in ["AURC", "SelAcc80", "ECE", "Cov99"]:
    v = res[m]
    print(f"{m:9s}: {st.mean(v):.3f} ± {st.pstdev(v):.3f}   draws={[round(x,3) for x in v]}")
print(f"test-fold size = {res['n_test'][0]} pages (headline split = 4020)")
print("paper claims: AURC 1.8±0.2, SelAcc80 97.5±0.3, ECE 0.018±0.006, Cov99 0.29±0.04")

# ───────────────────────── T1.1  per-column best baseline ───────────────────
hr("T1.1  best baseline per column (rq1_main.json point estimates)")
m = json.load(open("results/rq1_main.json"))["methods"]
b3 = json.load(open("results/b3_bootstrap.json")) if Path("results/b3_bootstrap.json").exists() else None
pt = {k: {kk: vv[0] for kk, vv in v.items()} for k, v in m.items()}
if b3: pt["B3"] = {k: b3[k]["point"] for k in ["AURC", "SelAcc80", "FPR80", "Cov99", "ECE"]}
bases = [k for k in pt if k != "PhishProof"]
for col, better in [("AURC", min), ("SelAcc80", max), ("Cov99", max), ("ECE", min)]:
    vals = {b: pt[b][col] for b in bases if col in pt[b]}
    win = better(vals, key=vals.get)
    print(f"{col:9s}: best baseline = {win} ({vals[win]:.3f})   | underlined in tab_main = B6 ({pt['B6'][col]:.3f})")
print("note: tab_main underlines the *whole B6 row* (best_baseline=B6 by AURC) — a design choice, not per-column")
