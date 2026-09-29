"""Compute clean consensus groundedness G for the 70B adversarial pages (cached, no new LLM).

bundle_or stores only per-agent {id, verdict, confidence} -- not cues -- so clean G cannot be
reconstructed from it. This re-runs the clean panel (all calls hit the cache, since these pages
were scored when bundle_or was built) and applies score_page's CPU grounding to get the clean
consensus G per page. Writes results/_or/clean_g.json = {page_id: G}, which
postprocess_adversarial_or.py reads to fill the "G clean -> attacked" column.

Run AFTER the adversarial job finishes (to avoid CLIP/CPU contention).

Usage:
    .venv/bin/python scripts/compute_clean_g_or.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.aggregate.gea import score_page
from phishproof.agents.client import ChatClient
from phishproof.agents.panel import Panel
from phishproof.config import load_panel
from phishproof.data_io import read_manifest
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.logo_brand import CLIPLogoEmbedder
from phishproof.tools.registry import GroundingContext

BUNDLE = Path("results/_or/bundle_adversarial_or.jsonl")
OUT = Path("results/_or/clean_g.json")
PANEL = Path("configs/panel_or.yaml")
DATA = Path("data/phishsel_final")


def main() -> int:
    if not BUNDLE.exists():
        print(f"[wait] {BUNDLE} not present yet")
        return 1
    page_ids = sorted({json.loads(l)["page_id"]
                       for l in BUNDLE.read_text().splitlines() if l.strip()})
    pages = {p.page_id: p for p in read_manifest(DATA / "test.jsonl")}
    todo = [pages[pid] for pid in page_ids if pid in pages]
    print(f"Computing clean G for {len(todo)} pages (cache-hit panel + CPU grounding)")

    panel = Panel.from_config(load_panel(PANEL), ChatClient())
    ctx = GroundingContext(detector=HtmlBrandDetector(), logo_embedder=CLIPLogoEmbedder())

    def progress(agent_id, i, n):
        if i % 50 == 0 and i:
            print(f"  {agent_id}: {i}/{n}", flush=True)

    outs_by_page = panel.run_batched(todo, progress=progress)
    clean_g = {}
    for p in todo:
        r = score_page(outs_by_page[p.page_id], p, ctx,
                       add_consistency=True, relax_perceptual=True)
        clean_g[p.page_id] = round(float(r.groundedness), 4)

    OUT.write_text(json.dumps(clean_g))
    import numpy as np
    vals = np.array(list(clean_g.values()))
    print(f"Clean G: mean={vals.mean():.3f} median={np.median(vals):.3f} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
