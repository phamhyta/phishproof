"""Patch B8 = single strong-VLM (GPT-4o vision) verbalized confidence into a bundle.

This is the reviewer-requested "VLM-alone + same calibration" baseline. Because the panel's
verdict equals the vision agent's verdict on 100% of pages, B8 ranks exactly the same verdicts
the panel does, but by the VLM's own self-reported confidence instead of grounded agreement --
isolating what the panel/grounding machinery adds over the single discriminating model. AURC is
rank-based, so isotonic calibration (which run_experiments applies to every score) does not change
B8's AURC; B8 is the strongest clean ranker yet trusts adversarial evasions the grounding veto
rejects (see tab_adversarial / RQ7).

Run run_experiments.py afterwards to (re)compute the bootstrapped metrics + CIs.

Usage:
    .venv/bin/python scripts/add_vlm_baseline.py --bundle results/bundle_D.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--agent", default="agent_c_vision", help="id of the strong single VLM agent")
    ap.add_argument("--field", default="B8", help="bundle baselines.<field> to write")
    ap.add_argument("--default", type=float, default=0.5, help="score for pages with no agent confidence")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.bundle.read_text().splitlines() if l.strip()]
    for r in rows:
        conf = next((a["confidence"] for a in r.get("agents", [])
                     if a.get("id") == args.agent and a.get("confidence") is not None), args.default)
        r.setdefault("baselines", {})[args.field] = conf

    tmp = args.bundle.with_suffix(args.bundle.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")   # atomic write
    tmp.replace(args.bundle)
    print(f"[ok] patched baselines.{args.field} = {args.agent} confidence on {len(rows)} pages "
          f"in {args.bundle}\n     next: scripts/run_experiments.py --bundle {args.bundle} --out results/_D")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
