"""W2.1: the real B3 self-consistency baseline (Wang et al. 2023).

The draft used B1/B2 as stand-ins for B3 because B3 needs k extra samples per page. This
runs the actual baseline: one text model (Qwen2.5-3B), sampled k times at temperature>0; the
self-consistency score is the agreement fraction of the majority verdict, and the verdict is
that majority. A per-sample suffix makes each of the k draws a distinct, reproducibly-cached
call.

Outputs results/b3_selfconsistency.jsonl (per page) + prints the five selective metrics so
the row can be added to tab_main and the "B3 not run" caveat removed.

Usage: .venv/bin/python scripts/run_b3_selfconsistency.py --data data/phishsel_final --k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.agents.base import _parse
from phishproof.agents.client import ChatClient
from phishproof.agents.page_context import render_context
from phishproof.agents.prompts import SYSTEM_PROMPT, build_user_prompt
from phishproof.config import load_panel
from phishproof.data_io import read_manifest
from phishproof.eval.metrics import (
    aurc,
    coverage_at_selective_accuracy,
    ece,
    fpr_at_coverage,
    selective_accuracy_at_coverage,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/phishsel_final"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--agent", choices=["vision", "text"], default="vision",
                    help="which single model to resample; 'vision' = the strongest single "
                         "model (GPT-4o), matching B1/B2's choice")
    ap.add_argument("--out", type=Path, default=Path("results/b3_selfconsistency.jsonl"))
    args = ap.parse_args()

    # B3 = self-consistency of the SAME strongest single model B1/B2 use (the GPT-4o vision
    # agent), resampled at temperature>0. --agent text resamples a 3B text model instead.
    panel = load_panel()
    if args.agent == "vision":
        base = next(a for a in panel.panel if a.modality == "vision")
    else:
        base = next(a for a in panel.panel if a.modality == "text"
                    and "qwen" in a.model.lower())
    cfg = base.model_copy(update={"temperature": args.temperature})
    client = ChatClient()
    print(f"B3 self-consistency: model={cfg.model} ({args.agent}) k={args.k} "
          f"temp={args.temperature}")

    pages = read_manifest(args.data / "test.jsonl")
    if args.limit:
        pages = pages[: args.limit]

    rows = []
    for i, p in enumerate(pages, 1):
        ctx = render_context(p)
        user = build_user_prompt(ctx)
        image = p.screenshot_path if cfg.modality == "vision" else None
        verdicts = []
        for s in range(args.k):
            sampled_user = f"{user}\n\n[self-consistency sample {s}]"
            raw = client.complete_json(cfg, SYSTEM_PROMPT, sampled_user, image)
            out = _parse(cfg.id, raw)
            verdicts.append(out.verdict.value)
        cnt = Counter(verdicts)
        verdict, n_maj = cnt.most_common(1)[0]
        trust = n_maj / args.k
        rows.append({"page_id": p.page_id, "label": p.label.value,
                     "verdict": verdict, "trust": trust})
        if i % 100 == 0:
            print(f"  {i}/{len(pages)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    trust = np.array([r["trust"] for r in rows])
    correct = np.array([r["verdict"] == r["label"] for r in rows])
    yt = np.array([1 if r["label"] == "phish" else 0 for r in rows])
    yp = np.array([1 if r["verdict"] == "phish" else 0 for r in rows])
    print(f"\nB3 self-consistency (n={len(rows)}, acc={correct.mean():.3f}):")
    print(f"  AURC      = {aurc(trust, correct)*100:.2f}")
    print(f"  SelAcc80  = {selective_accuracy_at_coverage(trust, correct, 0.80)*100:.2f}")
    print(f"  FPR80     = {fpr_at_coverage(trust, yt, yp, 0.80)*100:.2f}")
    print(f"  Cov99     = {coverage_at_selective_accuracy(trust, correct, 0.99):.3f}")
    print(f"  ECE       = {ece(trust, correct):.3f}")
    print(f"\n[ok] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
