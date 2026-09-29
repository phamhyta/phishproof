#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Attribution ladder on the 70B Phishpedia panel, emitted to a committed artifact so
every cell of tab_attribution is traceable (fixes the provenance gap the audit flagged).

Uses the SAME conventions as scripts/revision_t10_metrics.py: raw attainable-coverage AURC
with deterministic page-id tie ordering, and a paired page/group bootstrap over the
duplicate groups (t5_grouped_split.json) that holds the raw scores fixed. So conf_VLM's
delta here reproduces t10_phishpedia.json's paired_bootstrap value, and the evidence-only
rows (GEA, consensus fraction, groundedness, GEA*G) that t10 does not pair against s are
added here. Writes results/revision_v2/t10/attribution_70b.json.

Usage: uv run scripts/revision_attribution_70b.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EPS = 1e-3
N_BOOT = 2000
ELIG = Path("results/revision_v2/t8/eligibility_phishpedia.jsonl")
GROUPS = Path("results/revision/t5_grouped_split.json")
OUT = Path("results/revision_v2/t10/attribution_70b.json")
ACTIVE = ("brand_claim", "form_action_domain", "credential_intent", "logo_brand")


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main():
    rows = load(ELIG)
    n = len(rows)
    pids = [r["page_id"] for r in rows]
    pid_rank = np.argsort(np.argsort(np.array(pids))).astype(float)
    correct = np.array([r["verdict"] == r["label"] if r["verdict"] else False
                        for r in rows], float)
    valid = np.array([r["complete"]["valid_inference"] for r in rows])
    conf = np.array([r["conf_vlm"] if r["conf_vlm"] is not None else np.nan for r in rows])
    a_text = np.array([r["a_text"] for r in rows])
    gea = np.array([r["gea"] for r in rows])
    G = np.array([r["surrogate"]["groundedness_mean"] for r in rows])
    s = np.where(np.isnan(conf), np.nan, conf + EPS * a_text)
    cf = np.zeros(n)
    for i, r in enumerate(rows):
        mc = [c for c in r["cues"] if c["source"] == "model"]
        cons = [c for c in mc if c["in_strict_consensus"]]
        cf[i] = len(cons) / len(mc) if mc else 0.0

    gmap = json.loads(GROUPS.read_text())["test_group_map"]
    unit = np.array([gmap.get(p, f"solo::{p}") for p in pids])
    uniq = np.unique(unit)
    unit_idx = {u: np.where(unit == u)[0] for u in uniq}
    rng = np.random.RandomState(0)
    resamples = [np.concatenate([unit_idx[u] for u in rng.choice(uniq, len(uniq), replace=True)])
                 for _ in range(N_BOOT)]

    def aurc(sc, el):
        idx = np.where(el)[0]
        o = idx[np.lexsort((pid_rank[idx], -sc[idx]))]
        err = np.cumsum(1.0 - correct[o])
        return 100.0 * float(np.mean(err / np.arange(1, len(o) + 1)))

    def selacc(sc, el, cov=0.80):
        k_el = int(el.sum())
        if k_el / n < cov:
            return None
        idx = np.where(el)[0]
        o = idx[np.lexsort((pid_rank[idx], -sc[idx]))]
        k = int(round(cov * n))
        return round(100.0 * float(np.mean(correct[o[:k]])), 1)

    def aurc_idx(sc, el, idx):
        e = el[idx]
        if not e.any():
            return np.nan
        si = sc[idx][e]; ci = correct[idx][e]
        o = np.argsort(-si, kind="mergesort")
        err = np.cumsum(1.0 - ci[o])
        return 100.0 * float(np.mean(err / np.arange(1, len(o) + 1)))

    els = valid & ~np.isnan(s)

    def paired(sc, el):
        d = np.array([aurc_idx(sc, el, i) - aurc_idx(s, els, i) for i in resamples])
        d = d[~np.isnan(d)]
        return round(float(d.mean()), 2), [round(float(np.percentile(d, 2.5)), 2),
                                           round(float(np.percentile(d, 97.5)), 2)]

    scores = {
        "conf_VLM": (conf, valid & ~np.isnan(conf)),
        "concurrence": (a_text, valid),
        "GEA": (gea, valid),
        "consensus_fraction": (cf, valid),
        "GEAxG": (gea * G, valid),
        "groundedness": (G, valid),
        "s": (s, els),
    }
    out = {"panel": "70B", "corpus": "phishpedia", "n": n,
           "convention": "raw attainable-coverage AURC, deterministic page-id ties; paired "
                         "delta vs s over duplicate-group resamples, scores held fixed "
                         "(same as revision_t10_metrics.py)",
           "rows": {}}
    for name, (sc, el) in scores.items():
        row = {"AURC": round(aurc(sc, el), 2), "SelAcc80": selacc(sc, el),
               "n_eligible": int(el.sum())}
        if name != "s":
            dm, ci = paired(sc, el)
            row["delta_vs_s"] = dm
            row["delta_vs_s_ci95"] = ci
        out["rows"][name] = row
    OUT.write_text(json.dumps(out, indent=2))
    for name, r in out["rows"].items():
        d = f" delta_vs_s {r.get('delta_vs_s'):+.2f} {r.get('delta_vs_s_ci95')}" if "delta_vs_s" in r else " (ref)"
        print(f"  {name:20s} AURC {r['AURC']:5.2f} SelAcc {r['SelAcc80']}{d}")
    print(f"[ok] wrote {OUT}")


if __name__ == "__main__":
    main()
