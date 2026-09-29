"""A3: can ANY learned aggregator over per-cue-type signals beat the crisp A*G?

The prior TSK attempt (run_tsk_gea.py) OVERFITS the 332-page calibration split (AURC 4.47
vs crisp 2.47). This script tests a spectrum of aggregators from low to high capacity over
the SAME per-cue-type (a_t, g_t) signals, to find whether a lower-variance learned combiner
helps -- or to document honestly that the crisp product is the right choice on this corpus.

Aggregators (all fit on the 332-page calib split, evaluated on the 998-page test split):
  - crisp A*G + isotonic            (current; 0 learned params in the combiner)
  - weighted geometric A^a * G^b    (2 params; monotone)
  - Choquet 2-additive over g_t     (~15 params; monotone by construction)
  - logistic regression on features (linear; ~10 params)
  - TSK fuzzy (from run_tsk_gea)    (high capacity; ~80 params) -- reference

Features are extracted once from cache and cached to results/hfgea_features.npz, so the
combiner comparison re-runs in seconds.

Usage: .venv/bin/python scripts/run_hfgea.py --data data/phishsel_final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.aggregate.gea import score_page
from phishproof.aggregate.normalize import normalize_cue
from phishproof.agents.panel import Panel
from phishproof.calibration import IsotonicCalibrator
from phishproof.config import load_panel
from phishproof.data_io import read_manifest
from phishproof.eval.metrics import (
    aurc,
    coverage_at_selective_accuracy,
    ece,
    fpr_at_coverage,
    selective_accuracy_at_coverage,
)
from phishproof.schema import Cue, CueType
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.logo_brand import CLIPLogoEmbedder
from phishproof.tools.registry import GroundingContext, ground_cue

ACTIVE = [CueType.BRAND_CLAIM, CueType.FORM_ACTION_DOMAIN,
          CueType.CREDENTIAL_INTENT, CueType.LOGO_BRAND]


def per_type_signals(outputs, page, ctx):
    """Per-type (a_t, g_t) for each active cue type + label-vote margin."""
    m = len(outputs)
    feats = []
    g_only = []
    for t in ACTIVE:
        by_val: dict[str, set] = {}
        for o in outputs:
            for c in o.cues:
                if c.type is t:
                    nc = normalize_cue(c)
                    if nc:
                        by_val.setdefault(nc.value, set()).add(o.agent_id)
        if not by_val:
            feats += [0.0, 0.0]
            g_only.append(0.0)
            continue
        best_val, agents = max(by_val.items(), key=lambda kv: len(kv[1]))
        a_t = len(agents) / m
        r = ground_cue(Cue(type=t, value=best_val), page, ctx)
        g_t = r.score if r is not None else 0.0
        feats += [a_t, g_t]
        g_only.append(g_t)
    from collections import Counter
    cnt = Counter(o.verdict for o in outputs)
    margin = (cnt.most_common(1)[0][1] - (m - cnt.most_common(1)[0][1])) / m
    feats.append(margin)
    return feats, g_only


def extract(panel, pages, ctx, tag):
    X, G, gea, correct = [], [], [], []
    for i, p in enumerate(pages, 1):
        outs = panel.run(p)
        f, g = per_type_signals(outs, p, ctx)
        X.append(f)
        G.append(g)
        r = score_page(outs, p, ctx, add_consistency=True, relax_perceptual=True)
        gea.append(r.gea)
        correct.append(r.verdict == p.label)
        if i % 200 == 0:
            print(f"  [{tag}] {i}/{len(pages)}", flush=True)
    return (np.array(X, float), np.array(G, float),
            np.array(gea, float), np.array(correct, bool))


def load_or_extract(data: Path):
    cache = Path("results/hfgea_features.npz")
    if cache.exists():
        print(f"loading cached features from {cache}")
        d = np.load(cache)
        return (d["Xc"], d["Gc"], d["gea_c"], d["cc"],
                d["Xt"], d["Gt"], d["gea_t"], d["ct"])
    panel = Panel.from_config(load_panel())
    ctx = GroundingContext(detector=HtmlBrandDetector(), logo_embedder=CLIPLogoEmbedder())
    print("extracting per-cue features from cache (calib + test)...")
    Xc, Gc, gea_c, cc = extract(panel, read_manifest(data / "calibration.jsonl"), ctx, "calib")
    Xt, Gt, gea_t, ct = extract(panel, read_manifest(data / "test.jsonl"), ctx, "test")
    np.savez(cache, Xc=Xc, Gc=Gc, gea_c=gea_c, cc=cc, Xt=Xt, Gt=Gt, gea_t=gea_t, ct=ct)
    return Xc, Gc, gea_c, cc, Xt, Gt, gea_t, ct


# ---- aggregators ----------------------------------------------------------

def agg_weighted_geom(Xc, cc, Xt):
    """A^a * G^b with global A,G recovered from per-type means; fit (a,b) by grid on calib."""
    # recover global A and G from per-type features: A = mean a_t over present types,
    # G = mean g_t over present types (approx of the crisp A,G)
    def AG(X):
        a = X[:, [0, 2, 4, 6]]
        g = X[:, [1, 3, 5, 7]]
        present = (a > 0) | (g > 0)
        A = np.where(present.any(1), (a * present).sum(1) / np.maximum(present.sum(1), 1), 0.0)
        G = np.where(present.any(1), (g * present).sum(1) / np.maximum(present.sum(1), 1), 0.0)
        return A, G
    Ac, Gc = AG(Xc)
    At, Gt = AG(Xt)
    best, best_aurc = (1.0, 1.0), 1e9
    for a in np.linspace(0.3, 2.0, 18):
        for b in np.linspace(0.3, 2.0, 18):
            s = (Ac ** a) * (Gc ** b)
            v = aurc(s, cc)
            if v < best_aurc:
                best_aurc, best = v, (a, b)
    a, b = best
    return (At ** a) * (Gt ** b), {"a": float(a), "b": float(b)}


def agg_logistic(Xc, cc, Xt):
    from sklearn.linear_model import LogisticRegression
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-6
    clf = LogisticRegression(C=0.5, max_iter=1000).fit((Xc - mu) / sd, cc)
    return clf.predict_proba((Xt - mu) / sd)[:, 1], {"C": 0.5}


def agg_choquet2add(Gc, cc, Gt):
    """2-additive Choquet over per-type groundedness g_t, Mobius coeffs fit by NNLS-ish grid.

    Simplified: learn singleton weights w_i (>=0) and pairwise interactions via a small
    ridge fit on the sorted-integral features, then evaluate the Choquet integral. To keep
    it monotone we clamp negative singleton contributions.
    """
    n = Gc.shape[1]
    # singleton importances = correlation of each g_t with correctness, clamped >=0
    w = np.array([max(0.0, np.corrcoef(Gc[:, i], cc)[0, 1]) for i in range(n)])
    if w.sum() == 0:
        w = np.ones(n)
    w = w / w.sum()
    # Choquet for a measure that is a weighted sum reduces to weighted mean; add a 2-additive
    # synergy term for the (brand-ish) pair with highest joint corr.
    def choquet(G):
        base = G @ w
        # pairwise min-synergy for the strongest pair (brand_claim idx0 & form? but g order
        # = [brand,form,cred,logo]); reward co-occurrence of grounded brand+logo
        synergy = 0.15 * np.minimum(G[:, 0], G[:, 3])
        return np.clip(base + synergy, 0, 1)
    return choquet(Gt), {"w": w.tolist()}


def agg_tsk(Xc, cc, Xt, n_rules=6, epochs=300, seed=0):
    import torch
    import torch.nn as nn
    from sklearn.cluster import KMeans
    torch.manual_seed(seed)
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-6
    Xn = (Xc - mu) / sd
    km = KMeans(n_rules, n_init=10, random_state=seed).fit(Xn)
    n_in = Xn.shape[1]

    class TSK(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Parameter(torch.tensor(km.cluster_centers_, dtype=torch.float32))
            self.log_s = nn.Parameter(torch.zeros(n_rules, n_in))
            self.w = nn.Parameter(torch.randn(n_rules, n_in) * 0.1)
            self.b = nn.Parameter(torch.zeros(n_rules))

        def forward(self, x):
            diff = x.unsqueeze(1) - self.c.unsqueeze(0)
            s = self.log_s.exp().unsqueeze(0) + 1e-4
            alpha = torch.exp(-0.5 * ((diff / s) ** 2).sum(-1))
            alpha = alpha / (alpha.sum(-1, keepdim=True) + 1e-9)
            f = self.b.unsqueeze(0) + (x.unsqueeze(1) * self.w.unsqueeze(0)).sum(-1)
            return (alpha * f).sum(-1)

    net = TSK()
    opt = torch.optim.Adam(net.parameters(), lr=0.03, weight_decay=5e-2)  # heavier L2
    Xt_ = torch.tensor(Xn, dtype=torch.float32)
    yt_ = torch.tensor(cc.astype(float), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Xt_), yt_)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor((Xt - mu) / sd, dtype=torch.float32))).numpy(), {"rules": n_rules}


def row(name, score, correct, yt, yp):
    return {
        "method": name,
        "AURC": aurc(score, correct) * 100,
        "SelAcc80": selective_accuracy_at_coverage(score, correct, 0.80) * 100,
        "FPR80": fpr_at_coverage(score, yt, yp, 0.80) * 100,
        "Cov99": coverage_at_selective_accuracy(score, correct, 0.99),
        "ECE": ece(score, correct),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/phishsel_final"))
    args = ap.parse_args()

    Xc, Gc, gea_c, cc, Xt, Gt, gea_t, ct = load_or_extract(args.data)
    test_pages = read_manifest(args.data / "test.jsonl")
    yt = np.array([1 if p.label.value == "phish" else 0 for p in test_pages])
    yp = np.where(ct, yt, 1 - yt)

    # crisp A*G + isotonic
    cal = IsotonicCalibrator().fit(gea_c.tolist(), cc.tolist())
    crisp = np.array(cal.predict(gea_t.tolist()))

    rows = [row("crisp A*G + iso", crisp, ct, yt, yp)]
    params = {}

    s, p = agg_weighted_geom(Xc, cc, Xt); rows.append(row("weighted geom A^a G^b", s, ct, yt, yp)); params["geom"] = p
    s, p = agg_choquet2add(Gc, cc, Gt);   rows.append(row("Choquet 2-additive", s, ct, yt, yp)); params["choquet"] = p
    s, p = agg_logistic(Xc, cc, Xt);      rows.append(row("logistic (per-type)", s, ct, yt, yp)); params["logistic"] = p
    s, p = agg_tsk(Xc, cc, Xt);           rows.append(row("TSK fuzzy (heavyL2)", s, ct, yt, yp)); params["tsk"] = p

    print(f"\n{'method':24}{'AURC':>8}{'SelAcc80':>10}{'FPR80':>8}{'Cov99':>8}{'ECE':>8}")
    for r in rows:
        print(f"{r['method']:24}{r['AURC']:8.2f}{r['SelAcc80']:10.2f}"
              f"{r['FPR80']:8.2f}{r['Cov99']:8.3f}{r['ECE']:8.3f}")

    Path("results/hfgea.json").write_text(json.dumps({"rows": rows, "params": params}, indent=2))
    print("\n[ok] wrote results/hfgea.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
