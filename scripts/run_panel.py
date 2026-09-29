"""Run the real evidence-agent panel over a manifest and write per-page GEA results.

Requires the panel models to be reachable (local Ollama by default — see configs/panel.yaml):
    ollama pull llama3.1:8b qwen2.5:7b qwen2.5vl:7b

Usage:
    .venv/bin/python scripts/run_panel.py --manifest data/phishsel/test.jsonl \
        --out results/gea_test.jsonl --limit 50
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.agents.panel import Panel
from phishproof.aggregate.gea import score_page
from phishproof.config import load_panel
from phishproof.data_io import read_manifest
from phishproof.tools.detector import GoldBrandDetector
from phishproof.tools.logo_brand import CLIPLogoEmbedder
from phishproof.tools.registry import GroundingContext


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-logo", action="store_true", help="skip CLIP logo grounding")
    args = ap.parse_args()

    records = read_manifest(args.manifest)
    if args.limit:
        records = records[: args.limit]

    panel = Panel.from_config(load_panel())
    # NOTE: GoldBrandDetector is a stand-in; wire real D1/D2/D3 in Phase 5.
    ctx = GroundingContext(
        detector=GoldBrandDetector(),
        logo_embedder=None if args.no_logo else CLIPLogoEmbedder(),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_act = 0
    errors = 0
    with args.out.open("w", encoding="utf-8") as f:
        for i, page in enumerate(records, 1):
            try:
                outputs = panel.run(page)
                res = score_page(outputs, page, ctx)
                f.write(res.model_dump_json() + "\n")
                n_act += int(res.gea > 0)
                if i % 25 == 0:
                    print(f"  {i}/{len(records)}  last GEA={res.gea:.3f}")
            except Exception:  # noqa: BLE001
                errors += 1
                traceback.print_exc()
    print(f"[ok] scored {len(records)} pages ({errors} errors) -> {args.out}")
    print(f"     pages with GEA>0: {n_act}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
