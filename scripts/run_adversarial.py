"""RQ7 -- adversarial robustness of grounded evidence-agreement (experiments.tex sec:exp_adv).

On clean Phishpedia, agreement A and groundedness G are near-collinear (Pearson ~0.71), so
the product GEA = A*G looks redundant. This script exercises the regime the clean corpus does
NOT contain: evidence corruption. We take correctly-detected phishing pages and apply an
evidence-targeted perturbation that PRESERVES the true label (still phishing):

  - form-action cloak  : rewrite the credential form to an on-brand-looking domain
  - logo morph         : blur the rendered brand logo region
  - both               : apply both

then re-run the panel and re-score. The hypothesis (the fail-safe story):
  * the attacked cue stops grounding -> G drops -> GEA collapses -> PhishProof ABSTAINS,
    rather than confidently emitting the now-wrong 'benign' verdict;
  * label-confidence baselines (B1 single-model confidence, B6 proxy) stay overconfident on
    the evaded verdict, because they read trust at the label, not at the grounded evidence.

Cost note: the cache keys on (model, prompt, image_hash), so cloak (DOM change) re-runs only
the text agents and REUSES the clean vision call; morph (screenshot change) re-runs only the
vision agent and REUSES the clean text calls.

Usage:
    .venv/bin/python scripts/run_adversarial.py --attack both --limit 150 --hard-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phishproof.aggregate.gea import score_page
from phishproof.agents.panel import Panel
from phishproof.baselines import compute_panel_baselines
from phishproof.calibration import IsotonicCalibrator
from phishproof.config import load_panel
from phishproof.data_io import read_manifest
from phishproof.eval.perturb import (
    cloak_form_action,
    morph_logo,
    occlude_logo,
    strip_brand_text,
)
from phishproof.schema import Label
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.logo_brand import CLIPLogoEmbedder
from phishproof.tools.registry import GroundingContext

ATTACKS = {
    "cloak": [cloak_form_action],          # form-action -> on-brand domain (DOM cue)
    "morph": [morph_logo],                 # blur the logo (weak perceptual perturbation)
    "occlude": [occlude_logo],             # remove the logo entirely (strong; logo-free regime)
    "both": [cloak_form_action, occlude_logo],  # corrupt both the DOM and the visual cue
    # white-box adaptive adversary: attack EVERY active verifier at once -- cloak the
    # form-action to an on-brand domain, erase the brand text from the DOM, and occlude the
    # logo -- so no cited cue grounds while the page still collects credentials.
    "adaptive": [cloak_form_action, strip_brand_text, occlude_logo],
}


def _pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _apply(page, fns, out_dir):
    """Apply each perturbation fn in turn; return (attacked_page, changed?)."""
    p = page
    for fn in fns:
        p = fn(p, out_dir)
    return p, (p.page_id != page.page_id)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/phishsel_final"))
    ap.add_argument("--bundle", type=Path, default=Path("results/bundle_final.jsonl"))
    ap.add_argument("--calibrator", type=Path, default=Path("results/calibrator.json"))
    ap.add_argument("--operating-point", type=Path, default=Path("results/operating_point.json"))
    ap.add_argument("--panel", type=Path, default=Path("configs/panel.yaml"))
    ap.add_argument("--attack", choices=list(ATTACKS), default="both")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--hard-only", action="store_true",
                    help="restrict to the lookalike (brand-in-domain) subset")
    ap.add_argument("--out", type=Path, default=Path("results/bundle_adversarial.jsonl"))
    ap.add_argument("--summary", type=Path, default=Path("results/rq7_adversarial.json"))
    args = ap.parse_args()

    clean = {json.loads(l)["page_id"]: json.loads(l)
             for l in args.bundle.read_text().splitlines() if l.strip()}
    pages = {p.page_id: p for p in read_manifest(args.data / "test.jsonl")}
    calibrator = IsotonicCalibrator.from_dict(json.loads(args.calibrator.read_text()))
    tau = json.loads(args.operating_point.read_text())["tau"]

    # correctly-detected phishing pages, spread across the clean GEA range
    sel = [pid for pid, r in clean.items()
           if r["label"] == "phish" and r["verdict"] == "phish" and pid in pages
           and (not args.hard_only or r.get("is_hard"))]
    sel.sort(key=lambda pid: clean[pid]["gea"])
    if len(sel) > args.limit:
        idx = np.linspace(0, len(sel) - 1, args.limit).astype(int)
        sel = [sel[i] for i in idx]
    print(f"RQ7 {args.attack} attack on {len(sel)} correctly-detected phish pages "
          f"(hard_only={args.hard_only})")

    out_dir = Path("results/perturbed")
    fns = ATTACKS[args.attack]
    attacked_pages, pid_map, skipped = [], {}, 0
    for pid in sel:
        ap_page, changed = _apply(pages[pid], fns, out_dir)
        if not changed:                      # e.g. morph with no logo box -> skip honestly
            skipped += 1
            continue
        attacked_pages.append(ap_page)
        pid_map[ap_page.page_id] = pid
    if skipped:
        print(f"  ({skipped} pages had no targetable cue for this attack -> skipped)")

    panel = Panel.from_config(load_panel(args.panel))
    ctx = GroundingContext(detector=HtmlBrandDetector(), logo_embedder=CLIPLogoEmbedder())

    seen = {"a": None}

    def progress(agent_id, i, n):
        if agent_id != seen["a"]:
            seen["a"] = agent_id
            print(f"  sweeping {agent_id} over {n} attacked pages...", flush=True)
        if i % 50 == 0 and i:
            print(f"      {agent_id}: {i}/{n}", flush=True)

    outs_by_page = panel.run_batched(attacked_pages, progress=progress)
    a_baselines = compute_panel_baselines(outs_by_page)

    rows = []
    for ap_page in attacked_pages:
        pid = pid_map[ap_page.page_id]
        r = score_page(outs_by_page[ap_page.page_id], ap_page, ctx,
                       add_consistency=True, relax_perceptual=True)
        c = clean[pid]
        rows.append({
            "page_id": pid,
            "label": "phish",
            "attack": args.attack,
            "clean": {"gea": c["gea"], "A": c["agreement"], "G": c["groundedness"],
                      "verdict": c["verdict"], "trust": c["calibrated_trust"],
                      "b1": c["baselines"].get("B1"), "b6": c["baselines"].get("B6")},
            "attacked": {"gea": r.gea, "A": r.agreement, "G": r.groundedness,
                         "verdict": r.verdict.value, "trust": calibrator(r.gea),
                         "b1": a_baselines.get("B1", {}).get(ap_page.page_id),
                         "b6": a_baselines.get("B6", {}).get(ap_page.page_id)},
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # ---- summary metrics ----
    def col(side, k):
        return [x[side][k] for x in rows]

    def fmean(side, k):
        v = [x[side][k] for x in rows if x[side][k] is not None]
        return float(np.mean(v)) if v else None

    evaded = [x for x in rows if x["attacked"]["verdict"] != Label.PHISH.value]
    n = len(rows)
    acted_clean = sum(1 for x in rows if x["clean"]["gea"] >= tau)
    acted_att = sum(1 for x in rows if x["attacked"]["gea"] >= tau)
    summary = {
        "attack": args.attack, "n": n, "tau": tau,
        "evasion_rate": len(evaded) / n if n else None,
        "gea_mean_clean": float(np.mean(col("clean", "gea"))),
        "gea_mean_attacked": float(np.mean(col("attacked", "gea"))),
        "G_mean_clean": float(np.mean(col("clean", "G"))),
        "G_mean_attacked": float(np.mean(col("attacked", "G"))),
        "abstain_rate_clean": 1 - acted_clean / n if n else None,
        "abstain_rate_attacked": 1 - acted_att / n if n else None,
        "corr_AG_clean": _pearson(col("clean", "A"), col("clean", "G")),
        "corr_AG_attacked": _pearson(col("attacked", "A"), col("attacked", "G")),
        # Trust contrast: PhishProof's calibrated trust should DROP under attack while
        # label-confidence baselines (B1/B6) stay flat -- they read trust at the label.
        "phishproof_trust_clean": fmean("clean", "trust"),
        "phishproof_trust_attacked": fmean("attacked", "trust"),
        "b1_clean": fmean("clean", "b1"),
        "b1_attacked": fmean("attacked", "b1"),
        "b6_clean": fmean("clean", "b6"),
        "b6_attacked": fmean("attacked", "b6"),
    }
    if evaded:
        # On evaded pages: does PhishProof abstain while B1 stays confident?
        pp_trust = [x["attacked"]["trust"] for x in evaded]
        pp_abstain = sum(1 for x in evaded if x["attacked"]["gea"] < tau) / len(evaded)
        b1 = [x["attacked"]["b1"] for x in evaded if x["attacked"]["b1"] is not None]
        summary["on_evaded"] = {
            "n": len(evaded),
            "phishproof_mean_trust": float(np.mean(pp_trust)),
            "phishproof_abstain_rate": pp_abstain,
            "b1_mean_confidence": float(np.mean(b1)) if b1 else None,
        }

    args.summary.write_text(json.dumps(summary, indent=2))
    print("\n=== RQ7 adversarial summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
