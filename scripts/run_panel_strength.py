"""W2.3: does a STRONGER panel change the story? (3B -> 7B/8B text agents).

Reviewer concern (audit C1): the panel is 3B-class; would larger models change the result?
We run the stronger panel (Llama-3.1-8B + Qwen2.5-7B + the same GPT-4o vision) on a stratified
subset of the test split and compare its ranking (AURC) and full-coverage accuracy to the 3B
panel on the SAME pages (from results/bundle_final.jsonl). AURC is a ranking metric, so no
recalibration is needed for the comparison.

Heavy on 16 GB: the batch-by-agent runner keeps one model resident at a time. Run on a subset
and NOT concurrently with other Ollama jobs.

Usage: .venv/bin/python scripts/run_panel_strength.py --data data/phishsel_final --limit 120
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
from phishproof.config import load_panel
from phishproof.data_io import read_manifest
from phishproof.eval.bundle import load_bundle
from phishproof.eval.metrics import aurc, detection_metrics
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.logo_brand import CLIPLogoEmbedder
from phishproof.tools.registry import GroundingContext


def stratified_subset(pages, n, seed=0):
    rng = np.random.default_rng(seed)
    phish = [p for p in pages if p.label.value == "phish"]
    benign = [p for p in pages if p.label.value == "benign"]
    rng.shuffle(phish)
    rng.shuffle(benign)
    half = n // 2
    return phish[:half] + benign[: n - half]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/phishsel_final"))
    ap.add_argument("--panel", type=Path, default=Path("configs/panel_strong.yaml"))
    ap.add_argument("--bundle", type=Path, default=Path("results/bundle_final.jsonl"))
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--out", type=Path, default=Path("results/panel_strength.json"))
    args = ap.parse_args()

    test = read_manifest(args.data / "test.jsonl")
    subset = stratified_subset(test, args.limit)
    pids = [p.page_id for p in subset]
    print(f"panel-strength on {len(subset)} pages (strong: {args.panel.name})")

    panel = Panel.from_config(load_panel(args.panel))
    ctx = GroundingContext(detector=HtmlBrandDetector(), logo_embedder=CLIPLogoEmbedder())

    seen = {"a": None}

    def progress(agent_id, i, n):
        if agent_id != seen["a"]:
            seen["a"] = agent_id
            print(f"  sweeping {agent_id} over {n} pages...", flush=True)
        if i % 40 == 0 and i:
            print(f"      {agent_id}: {i}/{n}", flush=True)

    outs = panel.run_batched(subset, progress=progress)
    pg = {p.page_id: p for p in subset}
    gea_s, correct_s, yt, yp_s = [], [], [], []
    for pid in pids:
        r = score_page(outs[pid], pg[pid], ctx, add_consistency=True, relax_perceptual=True)
        gea_s.append(r.gea)
        correct_s.append(r.verdict.value == pg[pid].label.value)
        yt.append(1 if pg[pid].label.value == "phish" else 0)
        yp_s.append(1 if r.verdict.value == "phish" else 0)
    gea_s = np.array(gea_s); correct_s = np.array(correct_s)

    # 3B panel on the same pages
    bundle = {r["page_id"]: r for r in load_bundle(args.bundle)}
    gea_3 = np.array([bundle[pid]["gea"] for pid in pids])
    correct_3 = np.array([bundle[pid]["verdict"] == bundle[pid]["label"] for pid in pids])
    yp_3 = [1 if bundle[pid]["verdict"] == "phish" else 0 for pid in pids]

    m_s = detection_metrics(yt, yp_s)
    m_3 = detection_metrics(yt, yp_3)
    res = {
        "n": len(pids),
        "strong": {"panel": args.panel.name, "AURC": aurc(gea_s, correct_s) * 100,
                   "acc": m_s["accuracy"] * 100, "f1": m_s["f1"] * 100},
        "base_3b": {"AURC": aurc(gea_3, correct_3) * 100,
                    "acc": m_3["accuracy"] * 100, "f1": m_3["f1"] * 100},
    }
    print(f"\n{'panel':14}{'AURC':>8}{'acc':>8}{'F1':>8}")
    print(f"{'3B (base)':14}{res['base_3b']['AURC']:8.2f}{res['base_3b']['acc']:8.1f}{res['base_3b']['f1']:8.1f}")
    print(f"{'7B/8B':14}{res['strong']['AURC']:8.2f}{res['strong']['acc']:8.1f}{res['strong']['f1']:8.1f}")
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\n[ok] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
