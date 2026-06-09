"""Replace the B6 proxy in a bundle with the faithful MultiPhishGuard consolidator score.

Re-runs the panel (cached -> instant) to recover each page's agent verdicts + cues, runs a
separate GPT-4o consolidator that deliberates a final verdict + calibrated confidence, and
overwrites the bundle's baselines.B6 with that confidence. Then re-run run_experiments.py.

Usage:
    .venv/bin/python scripts/run_multiphishguard.py --data data/phishsel_hard_big \
        --bundle results/bundle_hard_big.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.agents.client import ChatClient
from phishproof.agents.panel import Panel
from phishproof.baselines.multiphishguard import MultiPhishGuardConsolidator
from phishproof.config import AgentConfig, load_panel
from phishproof.data_io import read_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--model", default="gpt-4o", help="consolidator model")
    args = ap.parse_args()

    pages = {p.page_id: p for p in read_manifest(args.data / "test.jsonl")}
    bundle = [json.loads(l) for l in args.bundle.read_text().splitlines() if l.strip()]

    client = ChatClient()
    panel = Panel.from_config(load_panel(), client)            # cached agent calls
    cons_cfg = AgentConfig(id="mpg_consolidator", provider="openai",
                           model=args.model, modality="text")
    mpg = MultiPhishGuardConsolidator(client, cons_cfg)

    n = len(bundle)
    for i, row in enumerate(bundle, 1):
        page = pages.get(row["page_id"])
        if page is None:
            continue
        outs = panel.run(page)                                 # cache hit
        _verdict, conf = mpg.score(outs)
        row.setdefault("baselines", {})["B6"] = conf
        if i % 40 == 0:
            print(f"  {i}/{n} consolidated", flush=True)

    with args.bundle.open("w", encoding="utf-8") as f:
        for row in bundle:
            f.write(json.dumps(row) + "\n")
    print(f"[ok] patched B6 (faithful MultiPhishGuard) in {args.bundle}")
    print(f"     next: .venv/bin/python scripts/run_experiments.py --bundle {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
