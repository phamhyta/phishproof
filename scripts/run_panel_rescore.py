"""Re-score a panel config over the test manifest into a bundle (page_id, label, verdict, gea,
agents[]). Headline GEA = per_type_agreement (method D; grounding is the separate veto). Parallel
+ checkpointed + resumable, for the panel-upgrade pilot (3B text -> 70B OpenRouter).

Usage:
  .venv/bin/python scripts/run_panel_rescore.py --panel configs/panel_or.yaml \
      --out results/bundle_or.jsonl --workers 8 --checkpoint 100
"""
from __future__ import annotations
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishproof.agents.client import ChatClient
from phishproof.agents.panel import Panel
from phishproof.aggregate.consensus import per_type_agreement
from phishproof.config import load_panel
from phishproof.data_io import read_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, type=Path)
    ap.add_argument("--data", default=Path("data/phishsel_final"), type=Path)
    ap.add_argument("--manifest", default=None, type=Path,
                    help="Override manifest path (default: --data/test.jsonl)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--checkpoint", type=int, default=100)
    args = ap.parse_args()

    manifest_path = args.manifest if args.manifest else (args.data / "test.jsonl")
    pages = read_manifest(manifest_path)
    if args.limit:
        pages = pages[: args.limit]
    panel = Panel.from_config(load_panel(args.panel), ChatClient())

    rows: dict[str, dict] = {}
    if args.out.exists():
        rows = {(r := json.loads(l))["page_id"]: r for l in args.out.read_text().splitlines() if l.strip()}
    todo = [p for p in pages if p.page_id not in rows or rows[p.page_id].get("verdict") is None]
    lock = threading.Lock()
    n = [0]; t0 = time.time()
    print(f"[start] {len(todo)} to score, {len(rows)} resumed, workers={args.workers}", flush=True)

    def vv(v):
        return v.value if hasattr(v, "value") else v

    def work(p):
        last = None
        for _ in range(3):
            try:
                outs = panel.run(p)
                return {"page_id": p.page_id, "label": vv(p.label),
                        "verdict": vv(Panel.majority_label(outs)),
                        "gea": per_type_agreement(outs)[0],
                        "agents": [{"id": o.agent_id, "verdict": vv(o.verdict),
                                    "confidence": o.confidence} for o in outs]}
            except Exception as e:  # noqa: BLE001
                last = e; time.sleep(2.5)
        return {"page_id": p.page_id, "label": vv(p.label), "verdict": None, "gea": 0.0,
                "agents": [], "error": str(last)}

    def checkpoint():
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        tmp.write_text("\n".join(json.dumps(rows[k]) for k in rows) + "\n")
        tmp.replace(args.out)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, p): p for p in todo}
        for f in as_completed(futs):
            r = f.result()
            with lock:
                rows[r["page_id"]] = r
                n[0] += 1
                if n[0] % 20 == 0:
                    rate = (time.time() - t0) / n[0]
                    print(f"  {n[0]}/{len(todo)}  {rate:.1f}s/pg  ~{rate*(len(todo)-n[0])/60:.0f} min "
                          f"(fail {sum(1 for x in rows.values() if x.get('verdict') is None)})", flush=True)
                if n[0] % args.checkpoint == 0:
                    checkpoint(); print(f"  [ckpt] {n[0]} -> {args.out}", flush=True)
    checkpoint()
    print(f"[ok] wrote {args.out} ({len(rows)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
