"""TSK-GEA: a small trained first-order TSK fuzzy aggregator over the per-cue
(agreement, groundedness) signals, vs the crisp A*G + isotonic calibrator.

Diagnostic showed corr(A*G, correct)=0.37 < corr(A)=0.39 < corr(G)=0.43 — the hard-AND
product wastes signal that A and G each carry. This trains a tiny TSK over the SAME
per-cue evidence (the agents produce the inputs; the TSK does interpretable fusion) on the
SAME calibration split, so the delta isolates "trained fuzzy fusion vs fixed product".
Conservative variant: TSK predicts P(verdict correct) — the majority label is unchanged,
so this lifts the selective metrics + calibration, not full-coverage accuracy.

Usage: .venv/bin/python scripts/run_tsk_gea.py --data data/phishsel_final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.agents.panel import Panel
from phishproof.aggregate.gea import score_page
from phishproof.aggregate.normalize import normalize_cue
from phishproof.config import load_panel
from phishproof.data_io import read_manifest
from phishproof.eval.metrics import aurc, coverage_at_selective_accuracy, ece, fpr_at_coverage
from phishproof.schema import Cue, CueType
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.logo_brand import CLIPLogoEmbedder
from phishproof.tools.registry import GroundingContext, ground_cue

ACTIVE = [CueType.BRAND_CLAIM, CueType.FORM_ACTION_DOMAIN,
          CueType.CREDENTIAL_INTENT, CueType.LOGO_BRAND]


def per_type_signals(outputs, page, ctx):
    """Return the per-type (a_t, g_t) feature vector + label-vote margin."""
    m = len(outputs)
    feats = []
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
            continue
        best_val, agents = max(by_val.items(), key=lambda kv: len(kv[1]))
        a_t = len(agents) / m
        r = ground_cue(Cue(type=t, value=best_val), page, ctx)
        g_t = r.score if r is not None else 0.0
        feats += [a_t, g_t]
    from collections import Counter
    cnt = Counter(o.verdict for o in outputs)
    margin = (cnt.most_common(1)[0][1] - (m - cnt.most_common(1)[0][1])) / m
    feats.append(margin)
    return feats


def extract(panel, pages, ctx):
    X, gea, correct = [], [], []
    for i, p in enumerate(pages, 1):
        outs = panel.run(p)  # cache hit
        X.append(per_type_signals(outs, p, ctx))
        r = score_page(outs, p, ctx, add_consistency=True, relax_perceptual=True)
        gea.append(r.gea)
        correct.append(r.verdict == p.label)
        if i % 200 == 0:
            print(f"  extracted {i}/{len(pages)}", flush=True)
    return np.array(X, float), np.array(gea, float), np.array(correct, bool)


def train_tsk(Xc, yc, n_rules=8, epochs=400, seed=0):
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
    opt = torch.optim.Adam(net.parameters(), lr=0.03, weight_decay=1e-3)
    Xt = torch.tensor(Xn, dtype=torch.float32)
    yt = torch.tensor(yc.astype(float), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Xt), yt)
        loss.backward()
        opt.step()
    return net, mu, sd


def predict(net, X, mu, sd):
    import torch
    Xn = torch.tensor((X - mu) / sd, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(net(Xn)).numpy()


def evalrow(name, score, gea_correct, yt, yp):
    a = aurc(score, gea_correct) * 100
    cov = coverage_at_selective_accuracy(score, gea_correct, 0.99)
    fpr = fpr_at_coverage(score, yt, yp, 0.80) * 100
    e = ece(score, gea_correct)
    print(f"{name:18}{a:8.2f}{cov:8.3f}{fpr:8.2f}{e:8.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/phishsel_final"))
    args = ap.parse_args()

    panel = Panel.from_config(load_panel())
    ctx = GroundingContext(detector=HtmlBrandDetector(), logo_embedder=CLIPLogoEmbedder())
    print("extracting per-cue features from cache (calib + test)...")
    Xc, gea_c, cc = extract(panel, read_manifest(args.data / "calibration.jsonl"), ctx)
    Xt, gea_t, ct = extract(panel, read_manifest(args.data / "test.jsonl"), ctx)

    # crisp A*G + isotonic (the current method)
    from phishproof.calibration import IsotonicCalibrator
    cal = IsotonicCalibrator().fit(gea_c.tolist(), cc.tolist())
    gea_trust = np.array(cal.predict(gea_t.tolist()))

    # trained TSK over the same per-cue signals
    net, mu, sd = train_tsk(Xc, cc)
    tsk_trust = predict(net, Xt, mu, sd)

    # test labels (verdict fixed = panel majority; conservative variant)
    test_pages = read_manifest(args.data / "test.jsonl")
    yt = np.array([1 if p.label.value == "phish" else 0 for p in test_pages])
    # verdict: reconstruct from correct + label (correct => verdict==label)
    yp = np.where(ct, yt, 1 - yt)

    print(f"\nfeatures: {Xt.shape[1]} per page ({len(ACTIVE)}x(a,g)+margin)")
    print(f"corr(A*G, correct)={np.corrcoef(gea_t, ct)[0,1]:.3f}  "
          f"corr(TSK, correct)={np.corrcoef(tsk_trust, ct)[0,1]:.3f}")
    print(f"\n{'method':18}{'AURC':>8}{'Cov99':>8}{'FPR80':>8}{'ECE':>8}")
    evalrow("crisp A*G+iso", gea_trust, ct, yt, yp)
    evalrow("TSK-GEA (fuzzy)", tsk_trust, ct, yt, yp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
