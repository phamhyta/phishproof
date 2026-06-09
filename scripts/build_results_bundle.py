"""Glue runner: panel -> calibrate on calib split -> score test split -> baselines -> bundle.

Ties Phases 3-5 into the results bundle that run_experiments.py consumes. Requires the
panel models reachable (Ollama) and a built PhishSel (calibration.jsonl + test.jsonl).

  1. run the panel on the calibration split; fit the isotonic calibrator + target-risk
     threshold on (GEA, correctness)
  2. run the panel on the test split; compute calibrated trust per page
  3. compute the panel baselines (B1/B2/B4/B5/B6) on the test outputs
  4. write results/bundle.jsonl  (one line per test page)

Usage:
    .venv/bin/python scripts/build_results_bundle.py --data data/phishsel \
        --out results/bundle.jsonl --no-logo --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.aggregate.gea import score_page
from phishproof.agents.panel import Panel
from phishproof.baselines import compute_panel_baselines
from phishproof.calibration import (
    IsotonicCalibrator,
    threshold_for_target_risk,
)
from phishproof.config import load_experiment, load_panel
from phishproof.data_io import read_manifest
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.logo_brand import CLIPLogoEmbedder
from phishproof.tools.registry import GroundingContext


def run_split(panel: Panel, pages, ctx, tag: str = "",
              consistency: bool = True, relax: bool = True):
    """Return (gea_results, outputs_by_page). Batched by agent so each model loads once."""
    seen = {"a": None, "i": 0}

    def progress(agent_id, i, n):
        if agent_id != seen["a"]:
            seen["a"] = agent_id
            print(f"  [{tag}] sweeping {agent_id} over {n} pages...", flush=True)
        if i % 50 == 0:
            print(f"      {agent_id}: {i}/{n}", flush=True)

    outputs_by_page = panel.run_batched(pages, progress=progress)
    results = [score_page(outputs_by_page[p.page_id], p, ctx,
                          add_consistency=consistency, relax_perceptual=relax)
               for p in pages]
    return results, outputs_by_page


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/phishsel"))
    ap.add_argument("--out", type=Path, default=Path("results/bundle.jsonl"))
    ap.add_argument("--panel", type=Path, default=Path("configs/panel.yaml"))
    ap.add_argument("--no-logo", action="store_true")
    ap.add_argument("--no-consistency", action="store_true")
    ap.add_argument("--no-relax", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    consistency, relax = not args.no_consistency, not args.no_relax

    exp = load_experiment()
    calib_pages = read_manifest(args.data / "calibration.jsonl")
    test_pages = read_manifest(args.data / "test.jsonl")
    if args.limit:
        calib_pages = calib_pages[: args.limit]
        test_pages = test_pages[: args.limit]

    panel = Panel.from_config(load_panel(args.panel))
    ctx = GroundingContext(
        detector=HtmlBrandDetector(),  # page-content brand grounding (non-circular)
        logo_embedder=None if args.no_logo else CLIPLogoEmbedder(),
    )

    # 1) calibration split -> fit calibrator + threshold
    print(f"running panel on {len(calib_pages)} calibration pages...")
    cal_res, _ = run_split(panel, calib_pages, ctx, tag="calib",
                           consistency=consistency, relax=relax)
    cal_labels = {p.page_id: p.label for p in calib_pages}
    cal_scores = [r.gea for r in cal_res]
    cal_correct = [r.verdict == cal_labels[r.page_id] for r in cal_res]
    calibrator = IsotonicCalibrator().fit(cal_scores, cal_correct)
    tau = threshold_for_target_risk(cal_scores, cal_correct, exp.target_selective_risk)
    print(f"  fitted calibrator; tau(target_risk={exp.target_selective_risk})={tau:.3f}")

    # 2) test split -> score + calibrated trust
    print(f"running panel on {len(test_pages)} test pages...")
    test_res, outputs_by_page = run_split(panel, test_pages, ctx, tag="test",
                                          consistency=consistency, relax=relax)

    # 3) baselines on test outputs
    baselines = compute_panel_baselines(outputs_by_page)

    # 4) write RICH bundle (A, G, consensus cues, per-agent verdicts, is_hard) so the RQ3
    #    ablations and other variants can be recomputed from this file without re-running.
    test_pg = {p.page_id: p for p in test_pages}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in test_res:
            page = test_pg[r.page_id]
            outs = outputs_by_page[r.page_id]
            line = {
                "page_id": r.page_id,
                "label": page.label.value,
                "verdict": r.verdict.value,
                "gea": r.gea,
                "agreement": r.agreement,
                "groundedness": r.groundedness,
                "calibrated_trust": calibrator(r.gea),
                "is_hard": (page.source or "").endswith("hard"),
                "consensus_cues": [{"type": c.type.value, "value": c.value} for c in r.consensus_cues],
                "agents": [{"id": o.agent_id, "verdict": o.verdict.value,
                            "confidence": o.confidence} for o in outs],
                "baselines": {b: baselines[b][r.page_id] for b in baselines},
            }
            f.write(json.dumps(line) + "\n")

    # persist calibrator + threshold for reproducibility
    (args.out.parent / "calibrator.json").write_text(json.dumps(calibrator.to_dict()))
    (args.out.parent / "operating_point.json").write_text(json.dumps({"tau": tau}))
    print(f"[ok] wrote {args.out} ({len(test_res)} pages) + calibrator.json")
    print("     next: .venv/bin/python scripts/run_experiments.py --bundle " + str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
