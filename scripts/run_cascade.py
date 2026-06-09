"""RO-3 / Upgrade A: adaptive panel cascade — defer the paid vision call.

Cost reality for this project: the two text agents are local Ollama models ($0), the vision
agent is GPT-4o (the only paid call). So the deployment-cost story is "what fraction of pages
need the paid vision call?". The cascade runs the text tier first and escalates to the vision
tier only when the text-tier trust is in an uncertain band.

  TEXT tier  : run the 2 local text agents, score text-only GEA, calibrate.
               trust_text >= tau_hi  -> ACT     (no vision call)
               trust_text <= tau_lo  -> ABSTAIN (no vision call)
               otherwise             -> escalate
  VISION tier: add GPT-4o, use the full 3-agent GEA (the current v1 score).

We sweep the uncertain band to trace a cost-quality Pareto: x = fraction of pages sent to
the paid vision tier, y = AURC / SelAcc80 of the resulting cascade trust score. The cascade
trust = trust_text for text-decided pages, trust_full for escalated pages.

Features are extracted once from cache to results/cascade_features.npz.

Usage: .venv/bin/python scripts/run_cascade.py --data data/phishsel_final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.aggregate.gea import score_page
from phishproof.agents.panel import Panel
from phishproof.calibration import IsotonicCalibrator
from phishproof.config import load_panel
from phishproof.data_io import read_manifest
from phishproof.eval.bundle import load_bundle
from phishproof.eval.metrics import aurc, selective_accuracy_at_coverage
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.logo_brand import CLIPLogoEmbedder
from phishproof.tools.registry import GroundingContext


def text_agents(panel: Panel):
    """The non-vision agents (local text models)."""
    return [a for a in panel.agents if getattr(a.cfg, "modality", "text") != "vision"]


def extract_text_tier(data: Path):
    """Per page (calib+test): text-only GEA, text verdict-correctness, label, page_id."""
    cache = Path("results/cascade_features_v2.npz")
    if cache.exists():
        print(f"loading cached features from {cache}")
        d = np.load(cache, allow_pickle=True)
        return (d["gt_c"], d["cc"], d["gt_t"], d["ct"], d["lab_t"], d["pid_t"])

    panel = Panel.from_config(load_panel())
    tpanel = Panel(text_agents(panel))
    ctx = GroundingContext(detector=HtmlBrandDetector(), logo_embedder=CLIPLogoEmbedder())

    def run(pages, tag):
        gt, correct, lab, pids = [], [], [], []
        for i, p in enumerate(pages, 1):
            outs = tpanel.run(p)
            r = score_page(outs, p, ctx, add_consistency=True, relax_perceptual=True)
            gt.append(r.gea)
            correct.append(r.verdict == p.label)
            lab.append(1 if p.label.value == "phish" else 0)
            pids.append(p.page_id)
            if i % 200 == 0:
                print(f"  [{tag}] {i}/{len(pages)}", flush=True)
        return (np.array(gt, float), np.array(correct, bool),
                np.array(lab, int), np.array(pids))

    print("extracting TEXT-tier (2 local agents) GEA from cache...")
    gt_c, cc, _, _ = run(read_manifest(data / "calibration.jsonl"), "calib")
    gt_t, ct, lab_t, pid_t = run(read_manifest(data / "test.jsonl"), "test")
    np.savez(cache, gt_c=gt_c, cc=cc, gt_t=gt_t, ct=ct, lab_t=lab_t, pid_t=pid_t)
    return gt_c, cc, gt_t, ct, lab_t, pid_t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/phishsel_final"))
    ap.add_argument("--bundle", type=Path, default=Path("results/bundle_final.jsonl"))
    args = ap.parse_args()

    gt_c, cc, gt_t, ct, lab_t, pid_t = extract_text_tier(args.data)

    # full 3-agent score (v1) from bundle, aligned by page_id
    bundle = {r["page_id"]: r for r in load_bundle(args.bundle)}
    full_trust = np.array([bundle[p]["calibrated_trust"] if bundle[p].get("calibrated_trust")
                           is not None else bundle[p]["gea"] for p in pid_t])
    full_correct = np.array([bundle[p]["verdict"] == bundle[p]["label"] for p in pid_t])

    # text-tier calibrator (fit on calib split text GEA)
    kappa_text = IsotonicCalibrator().fit(gt_c.tolist(), cc.tolist())
    trust_text = np.array(kappa_text.predict(gt_t.tolist()))

    print(f"text-tier verdict accuracy={ct.mean():.3f}  full-panel accuracy={full_correct.mean():.3f}")

    # full-panel reference metrics (full trust vs FULL-panel correctness)
    aurc_full = aurc(full_trust, full_correct) * 100
    sel_full = selective_accuracy_at_coverage(full_trust, full_correct, 0.80) * 100

    # text-only reference (always 0 vision calls; text trust vs TEXT correctness)
    aurc_text = aurc(trust_text, ct) * 100
    sel_text = selective_accuracy_at_coverage(trust_text, ct, 0.80) * 100

    # sweep the uncertain band [tau_lo, tau_hi] symmetric around 0.5 in trust_text.
    # Escalated pages adopt the full-panel trust AND full-panel correctness; text-decided
    # pages keep the text-tier trust and text-tier correctness.
    print(f"\n{'band(half-width)':18}{'vision%':>9}{'AURC':>8}{'SelAcc80':>10}")
    pareto = []
    for hw in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        lo, hi = 0.5 - hw, 0.5 + hw
        escalate = (trust_text > lo) & (trust_text < hi)
        cascade_trust = np.where(escalate, full_trust, trust_text)
        cascade_correct = np.where(escalate, full_correct, ct)
        vis_frac = float(escalate.mean())
        a = aurc(cascade_trust, cascade_correct) * 100
        s = selective_accuracy_at_coverage(cascade_trust, cascade_correct, 0.80) * 100
        pareto.append({"band_halfwidth": hw, "vision_frac": vis_frac, "AURC": a, "SelAcc80": s})
        print(f"{hw:18.2f}{vis_frac*100:8.1f}%{a:8.2f}{s:10.2f}")

    print(f"\n  reference  text-only (0% vision):  AURC {aurc_text:.2f}  SelAcc80 {sel_text:.2f}")
    print(f"  reference  full v1 (100% vision):   AURC {aurc_full:.2f}  SelAcc80 {sel_full:.2f}")

    out = {
        "pareto": pareto,
        "text_only": {"AURC": aurc_text, "SelAcc80": sel_text, "vision_frac": 0.0},
        "full_v1": {"AURC": aurc_full, "SelAcc80": sel_full, "vision_frac": 1.0},
    }
    Path("results/cascade.json").write_text(json.dumps(out, indent=2))
    print("\n[ok] wrote results/cascade.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
