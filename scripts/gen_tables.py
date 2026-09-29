"""Regenerate every numeric value the paper reports from the committed canonical bundles,
into results/_D/table_values.json. This is the single pinned source check_tables.py verifies
against, closing the "hand-typed table" gap (review F3) for the secondary tables/prose too.

All numbers here recompute from:
  results/bundle_D.jsonl        (Phishpedia method-D canonical)
  results/bundle_adv_*.jsonl    (RQ7 adversarial, grounding-driven -> method-agnostic)
  results/_D/rq1_main.json, results/{apwg,trop}/rq1_main.json (headline, already pinned)

Usage:  .venv/bin/python scripts/gen_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))


def load(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def aurc(scores, correct):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    err, areas = 0, []
    for rank, i in enumerate(order, 1):
        if not correct[i]:
            err += 1
        areas.append(err / rank)
    return 100 * sum(areas) / len(areas)


def factor_ablation(rows, correct):
    """RQ3 -A&-G decomposition (experiments.tex)."""
    A_pt = [r["gea"] for r in rows]
    A_fr = [r["agreement"] for r in rows]
    G = [r["groundedness"] for r in rows]
    B4 = [r["baselines"]["B4"] for r in rows]
    return {
        "A_per_type": round(aurc(A_pt, correct), 2),
        "A_consensus_fraction": round(aurc(A_fr, correct), 2),
        "G_alone": round(aurc(G, correct), 2),
        "A_times_G": round(aurc([a * g for a, g in zip(A_pt, G)], correct), 2),
        "label_only_B4": round(aurc(B4, correct), 2),
    }


def conformal(rows, correct, targets=(0.01, 0.02, 0.05), S=200):
    import numpy as np
    gea = np.array([r["gea"] for r in rows]); c = np.array(correct); n = len(rows)
    rng = np.random.RandomState(0); out = {}
    for a in targets:
        risks, covs = [], []
        for _ in range(S):
            idx = rng.permutation(n); cal, te = idx[: n // 2], idx[n // 2:]
            gc, cc = gea[cal], c[cal]; best = gc.max() + 1
            for tau in sorted(set(gc), reverse=True):
                m = gc >= tau
                if m.sum() and (1 - cc[m].mean()) <= a:
                    best = tau
                elif m.sum():
                    break
            mt = gea[te] >= best
            if mt.sum():
                risks.append(1 - c[te][mt].mean()); covs.append(mt.mean())
        out[f"{a}"] = {"risk": round(float(np.mean(risks)), 3), "cov": round(float(np.mean(covs)), 2)}
    return out


def adversarial():
    """RQ7 tab_adversarial: evade%, G clean->attacked, grounding-veto abstain on evaded (G<0.3)."""
    files = {"cloak": "bundle_adv_cloak.jsonl", "occlude": "bundle_adv_occlude.jsonl",
             "both": "bundle_adv_both.jsonl", "adaptive": "bundle_adversarial_adaptive.jsonl"}
    out = {}
    for k, f in files.items():
        rows = load(RES / f)
        evaded = [r for r in rows if r["attacked"]["verdict"] == "benign"]
        out[k] = {
            "evade_pct": round(100 * len(evaded) / len(rows), 1),
            "G_clean": round(sum(r["clean"]["G"] for r in rows) / len(rows), 2),
            "G_attacked": round(sum(r["attacked"]["G"] for r in rows) / len(rows), 2),
            "abstain_evaded_pct": round(100 * sum(1 for r in evaded if r["attacked"]["G"] < 0.3)
                                        / max(1, len(evaded))),
        }
    return out


def cov99_recovery(rows, correct):
    """RQ1 Cov99 weakness fix: GEA is discrete -> low Cov99; breaking ties by the panel's mean
    confidence (re-orders only within an equal-agreement bucket) recovers it. AURC/SelAcc unchanged."""
    import numpy as np

    from phishproof.eval.metrics import coverage_at_selective_accuracy
    trust = np.array([r["calibrated_trust"] if r.get("calibrated_trust") is not None else r["gea"]
                      for r in rows])

    def mc(r):
        cs = [a["confidence"] for a in r["agents"] if a.get("confidence") is not None]
        return sum(cs) / len(cs) if cs else 0.0
    conf = np.array([mc(r) for r in rows])
    conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-9)
    c = np.array(correct)
    return {
        "cov99_parameter_free": round(float(coverage_at_selective_accuracy(trust, c, 0.99)), 2),
        "cov99_conf_tiebreak": round(float(coverage_at_selective_accuracy(trust + 1e-6 * conf, c, 0.99)), 2),
    }


def ece_matched(rows, correct):
    """RQ4 fairness (review M-2): re-fit the SAME held-out isotonic calibrator on each method's
    score and recompute ECE -- shows the ECE 'win' is the shipped calibrator, not GEA itself."""
    import numpy as np

    from phishproof.calibration import IsotonicCalibrator
    from phishproof.eval.metrics import ece
    c = np.array(correct); n = len(rows)

    def matched(score):
        s = np.array(score); out = []
        for seed in range(10):
            rng = np.random.default_rng(seed); idx = rng.permutation(n); cal, te = idx[:n // 2], idx[n // 2:]
            m = IsotonicCalibrator().fit(s[cal].tolist(), c[cal].tolist())
            out.append(ece(np.array(m.predict(s[te].tolist())), c[te]))
        return round(float(np.mean(out)), 3)
    return {"PhishProof": matched([r["gea"] for r in rows]),
            "B6": matched([r["baselines"]["B6"] for r in rows])}


def verifier_soundness():
    """tab_verifier_soundness + Prop-2 epsilons (eps_t = 1 - specificity)."""
    rows = json.loads((RES / "verifier_soundness.json").read_text())
    out = {}
    for r in rows:
        out[r["cue"]] = {k: (round(r[k], 3) if r.get(k) is not None else None)
                         for k in ("precision", "recall", "specificity")}
    return out


def detection():
    """tab_detect: D1 (d1_import.json), D3 (detector_d3.jsonl), PhishProof (_D detection)."""
    d1 = json.loads((RES / "d1_import.json").read_text())["detection"]
    d3rows = load(RES / "detector_d3.jsonl")
    tp = sum(1 for x in d3rows if x["verdict"] == "phish" and x["label"] == "phish")
    fp = sum(1 for x in d3rows if x["verdict"] == "phish" and x["label"] == "benign")
    fn = sum(1 for x in d3rows if x["verdict"] == "benign" and x["label"] == "phish")
    tn = sum(1 for x in d3rows if x["verdict"] == "benign" and x["label"] == "benign")
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    pp = json.loads((RES / "_D" / "rq1_main.json").read_text())["detection"]
    r1 = lambda x: round(100 * x, 1)
    return {
        "D1": {"acc": r1(d1["accuracy"]["point"]), "prec": r1(d1["precision"]["point"]),
               "rec": r1(d1["recall"]["point"]), "f1": r1(d1["f1"]["point"])},
        "D3": {"acc": r1((tp + tn) / len(d3rows)), "prec": r1(prec), "rec": r1(rec),
               "f1": r1(2 * prec * rec / (prec + rec))},
        "PhishProof": {"acc": r1(pp["accuracy"][0]), "prec": r1(pp["precision"][0]),
                       "rec": r1(pp["recall"][0]), "f1": r1(pp["f1"][0])},
    }


def brands_out():
    """tab_brands_out: Phishpedia in-dist (=headline), unseen-brands LBO, TR-OP."""
    lbo = json.loads((RES / "leave_brands_out.json").read_text())["summary"]
    trop = json.loads((RES / "trop" / "rq1_main.json").read_text())["methods"]["PhishProof"]
    return {
        "LBO": {"AURC": round(lbo["AURC"]["mean"], 2), "SelAcc80": round(lbo["SelAcc80"]["mean"], 1),
                "Cov99": round(lbo["Cov99"]["mean"], 2), "ECE": round(lbo["ECE"]["mean"], 3)},
        "TROP": {"AURC": round(trop["AURC"][0] * 100, 2), "SelAcc80": round(trop["SelAcc80"][0] * 100, 1),
                 "Cov99": round(trop["Cov99"][0], 2), "ECE": round(trop["ECE"][0], 3)},
    }


def main() -> int:
    rows = load(RES / "bundle_D.jsonl")
    correct = [1 if r["verdict"] == r["label"] else 0 for r in rows]
    fa = factor_ablation(rows, correct)
    extra_p = RES / "_D" / "ablation_extra.json"   # -diversity / -calib ECE (cached re-score)
    extra = json.loads(extra_p.read_text()) if extra_p.exists() else {}
    sv_p = RES / "_D" / "seed_variance.json"       # 5-seed re-split (run_seed_variance.py)
    sv = json.loads(sv_p.read_text())["summary"] if sv_p.exists() else {}
    values = {
        "_note": "auto-generated by scripts/gen_tables.py from committed bundles; do not hand-edit",
        "headline_phishpedia": json.loads((RES / "_D" / "rq1_main.json").read_text())["methods"],
        "factor_ablation_rq3": fa,
        "conformal_rq4": conformal(rows, correct),
        "cov99_recovery_rq1": cov99_recovery(rows, correct),
        "ece_matched_rq4": ece_matched(rows, correct),
        "adversarial_rq7": adversarial(),
        "detection": detection(),
        "verifier_soundness": verifier_soundness(),
        "brands_out": brands_out(),
        "ablation": {"full": fa["A_per_type"], "minus_per_type": fa["A_consensus_fraction"],
                     **extra},
        "hfgea": {"uniform": fa["A_per_type"], "consensus_fraction": fa["A_consensus_fraction"]},
        "seed_variance": sv,
    }
    out = RES / "_D" / "table_values.json"
    out.write_text(json.dumps(values, indent=2))
    print(f"[ok] wrote {out}")
    print(json.dumps({k: v for k, v in values.items() if k not in ("_note", "headline_phishpedia")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
