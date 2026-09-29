#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""Component ablation of the evidence-agreement ranker on the 70B Phishpedia panel
(verdict-level parse), reading only results/revision_v2/t8/eligibility_phishpedia.jsonl.

Rows (all AURC over attainable coverage, deterministic page-id ties, ECE ten-bin):
  full GEA           mean per-type agreement (parameter-free, uniform)   [the method]
  - per-type         use the consensus FRACTION |consensus|/|shared cues| instead
  - diversity        per-type agreement over the TEXT agents only (drop the vision agent)
  learned weights    logistic on the four per-type agreements, cross-fitted (vs uniform)
  - calibration      GEA ECE without the isotonic map (raw) vs with it

Usage: uv run scripts/revision_ablation_70b.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ACTIVE = ("brand_claim", "form_action_domain", "credential_intent", "logo_brand")
ELIG = Path("results/revision_v2/t8/eligibility_phishpedia.jsonl")
VISION = "agent_c_vision"


def load():
    return [json.loads(l) for l in ELIG.read_text().splitlines() if l.strip()]


def aurc(score, correct, elig, pid_rank):
    idx = np.where(elig)[0]
    o = idx[np.lexsort((pid_rank[idx], -score[idx]))]
    err = np.cumsum(1.0 - correct[o])
    return 100.0 * float(np.mean(err / np.arange(1, len(o) + 1)))


def ece(prob, correct, elig, bins=10):
    p = np.clip(prob[elig], 0, 1); c = correct[elig]
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        hi = i == bins - 1
        m = (p >= edges[i]) & ((p <= edges[i + 1]) if hi else (p < edges[i + 1]))
        if m.sum():
            e += m.sum() / len(p) * abs(c[m].mean() - p[m].mean())
    return float(e)


def crossfit_iso(score, correct, elig, k=5, seed=0):
    rng = np.random.RandomState(seed)
    folds = np.array_split(rng.permutation(len(score)), k)
    out = np.full(len(score), np.nan)
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        tr = tr[elig[tr]]; te2 = te[elig[te]]
        if len(tr) < 10 or len(te2) == 0:
            out[te2] = score[te2]; continue
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        ir.fit(score[tr], correct[tr]); out[te2] = ir.predict(score[te2])
    return out


def per_type_maps(row, agent_filter=None):
    """max asserters (optionally restricted to a subset of agents) per active type."""
    by_type = defaultdict(lambda: defaultdict(int))
    for c in row["cues"]:
        if c["source"] != "model":
            continue
        agents = c["asserted_by"]
        if agent_filter is not None:
            agents = [a for a in agents if a in agent_filter]
        if agents:
            by_type[c["type"]][c["value"]] = max(by_type[c["type"]][c["value"]], len(agents))
    return by_type


def main():
    rows = load()
    n = len(rows)
    pids = np.array([r["page_id"] for r in rows])
    pr = np.argsort(np.argsort(pids)).astype(float)
    correct = np.array([r["verdict"] == r["label"] if r["verdict"] else False for r in rows], float)
    valid = np.array([r["complete"]["valid_inference"] for r in rows])
    M = 3
    text_agents = {a["id"] for a in rows[0]["agent_outputs"] if a["id"] != VISION}

    gea_full = np.zeros(n); gea_text = np.zeros(n); cons_frac = np.zeros(n)
    feats = np.zeros((n, 4))
    for i, r in enumerate(rows):
        bt = per_type_maps(r)
        pt = [max(bt[t].values(), default=0) / M for t in ACTIVE]
        gea_full[i] = float(np.mean(pt)); feats[i] = pt
        btt = per_type_maps(r, agent_filter=text_agents)
        gea_text[i] = float(np.mean([max(btt[t].values(), default=0) / max(len(text_agents), 1)
                                     for t in ACTIVE]))
        model_cues = [c for c in r["cues"] if c["source"] == "model"]
        cons = [c for c in model_cues if c["in_strict_consensus"]]
        cons_frac[i] = len(cons) / len(model_cues) if model_cues else 0.0

    # learned per-type weights: cross-fitted logistic on the four agreements
    learned = np.full(n, np.nan)
    rng = np.random.RandomState(0)
    folds = np.array_split(rng.permutation(n), 5)
    for i in range(5):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i])
        tr = tr[valid[tr]]; te2 = te[valid[te]]
        if len(tr) < 20 or len(te2) == 0:
            continue
        lr = LogisticRegression(max_iter=1000).fit(feats[tr], correct[tr])
        learned[te2] = lr.predict_proba(feats[te2])[:, 1]

    def report(name, score, note=""):
        A = aurc(score, correct, valid, pr)
        cal = crossfit_iso(score, correct, valid)
        E_raw = ece(np.clip(score, 0, 1), correct, valid)
        E_cal = ece(np.where(np.isnan(cal), 0, cal), correct, valid & ~np.isnan(cal))
        print(f"  {name:34s} AURC {A:5.2f}  ECE_raw {E_raw:.3f}  ECE_cal {E_cal:.3f}  {note}")
        return dict(AURC=round(A, 2), ECE_raw=round(E_raw, 3), ECE_cal=round(E_cal, 3))

    print("70B Phishpedia component ablation (n=%d, valid=%d):" % (n, int(valid.sum())))
    out = {}
    out["full_gea_pertype_uniform"] = report("full GEA (per-type, uniform)", gea_full)
    out["consensus_fraction"] = report("- per-type -> consensus fraction", cons_frac)
    out["diversity_text_only"] = report("- diversity (text agents only)", gea_text)
    lm = valid & ~np.isnan(learned)
    A = aurc(np.where(np.isnan(learned), 0, learned), correct, lm, pr)
    print(f"  {'learned per-type weights':34s} AURC {A:5.2f}  (cross-fitted logistic)")
    out["learned_pertype_weights"] = dict(AURC=round(A, 2))
    print("\n  calibration ablation: GEA ECE raw %.3f vs calibrated %.3f"
          % (out["full_gea_pertype_uniform"]["ECE_raw"],
             out["full_gea_pertype_uniform"]["ECE_cal"]))
    Path("results/revision_v2/t10/ablation_70b.json").write_text(json.dumps(out, indent=2))
    print("\n[ok] wrote results/revision_v2/t10/ablation_70b.json")


if __name__ == "__main__":
    main()
