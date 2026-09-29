"""Rebuild the FULL larger-panel bundle by replaying cached model calls (zero API calls).

Why this exists
---------------
`results/bundle_or.jsonl` was written by `run_panel_rescore.py`, which stores only
(page_id, label, verdict, gea, agents[id, verdict, confidence]). It drops the per-model
CUE SETS, and it never computed `consensus_cues`, `agreement`, `groundedness`, or the
`baselines` block that the matched-panel bundle carries. Those missing fields are what
T1/T2/T4 of the revision plan need, so the plan's "free re-analysis" is not free from
that file alone.

It *is* free from the cache. Every model call is keyed sha256(model+prompt+image_hash)
under `data/cache/`, and every larger-panel call is still there. Re-running the panel
with a cache-only client therefore reconstructs the full `AgentOutput`s -- cues included
-- without a single API request. A cache miss is an error, never a network call.

Validation: the replayed `gea` and `verdict` must reproduce `results/bundle_or.jsonl`
exactly. The script refuses to write a bundle that does not.

Usage
    uv run scripts/revision_replay_or.py --panel configs/panel_or.yaml \
        --manifest data/phishsel_final/test.jsonl \
        --validate-against results/bundle_or.jsonl \
        --out results/revision/bundle_or_full.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.aggregate.gea import score_page
from phishproof.agents.client import ChatClient
from phishproof.agents.panel import Panel
from phishproof.baselines import compute_panel_baselines
from phishproof.cache import JsonCache
from phishproof.config import load_panel
from phishproof.data_io import read_manifest
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.logo_brand import CLIPLogoEmbedder
from phishproof.tools.registry import GroundingContext


class CacheOnlyClient(ChatClient):
    """ChatClient that serves from the cache and raises on a miss. Never hits the network."""

    def __init__(self) -> None:
        super().__init__()
        self.misses: list[tuple[str, str]] = []
        self._mlock = threading.Lock()

    def complete_json(self, cfg, system, user, image_path=None):  # type: ignore[override]
        img_hash = JsonCache.hash_image(image_path) if image_path else None
        cache_prompt = f"SYS:{system}\nUSR:{user}\nDETAIL:{cfg.detail}"
        cached = self.cache.get(cfg.model, cache_prompt, img_hash)
        if cached is None:
            with self._mlock:
                self.misses.append((cfg.id, cfg.model))
            raise LookupError(f"cache miss for {cfg.id}/{cfg.model}")
        return cached


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=Path("configs/panel_or.yaml"))
    ap.add_argument("--manifest", type=Path, default=Path("data/phishsel_final/test.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/revision/bundle_or_full.jsonl"))
    ap.add_argument("--validate-against", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-logo", action="store_true", help="skip CLIP (logo cues become N/A)")
    args = ap.parse_args()

    pages = read_manifest(args.manifest)
    if args.limit:
        pages = pages[: args.limit]
    client = CacheOnlyClient()
    panel = Panel.from_config(load_panel(args.panel), client)
    ctx = GroundingContext(
        detector=HtmlBrandDetector(),
        logo_embedder=None if args.no_logo else CLIPLogoEmbedder(),
    )
    if not args.no_logo:  # load CLIP once, single-threaded, before the pool starts
        ctx.logo_embedder._ensure()

    outs_by_page: dict[str, list] = {}
    rows: dict[str, dict] = {}
    failed: list[str] = []
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def work(p):
        try:
            outs = panel.run(p)
        except LookupError:
            return p.page_id, None, None
        r = score_page(outs, p, ctx, add_consistency=True, relax_perceptual=True)
        return p.page_id, outs, r

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, p) for p in pages]
        for f in as_completed(futs):
            pid, outs, r = f.result()
            with lock:
                done[0] += 1
                if outs is None:
                    failed.append(pid)
                else:
                    outs_by_page[pid] = outs
                    rows[pid] = r
                if done[0] % 250 == 0:
                    el = time.time() - t0
                    print(f"  {done[0]}/{len(pages)}  {el:.0f}s  "
                          f"({el/done[0]:.2f}s/pg, ~{el/done[0]*(len(pages)-done[0])/60:.1f} min left)",
                          flush=True)

    print(f"[replay] {len(rows)} scored, {len(failed)} cache-miss failures")
    if client.misses:
        from collections import Counter
        print("[replay] MISSES by model:", Counter(m for _, m in client.misses).most_common())

    baselines = compute_panel_baselines(outs_by_page)

    by_page = {p.page_id: p for p in pages}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for p in pages:
            pid = p.page_id
            if pid not in rows:
                fh.write(json.dumps({"page_id": pid, "label": p.label.value, "verdict": None,
                                     "gea": 0.0, "agreement": 0.0, "groundedness": 0.0,
                                     "consensus_cues": [], "agents": [], "baselines": {},
                                     "replay_failed": True}) + "\n")
                n_written += 1
                continue
            r = rows[pid]
            outs = outs_by_page[pid]
            line = {
                "page_id": pid,
                "label": by_page[pid].label.value,
                "verdict": r.verdict.value,
                "gea": r.gea,
                "agreement": r.agreement,
                "groundedness": r.groundedness,
                "is_hard": (by_page[pid].source or "").endswith("hard"),
                "consensus_cues": [{"type": c.type.value, "value": c.value}
                                   for c in r.consensus_cues],
                # per-agent cues restored -- this is what run_panel_rescore.py dropped (T4)
                "agents": [{"id": o.agent_id, "verdict": o.verdict.value,
                            "confidence": o.confidence,
                            "cues": [{"type": c.type.value, "value": c.value}
                                     for c in o.cues]} for o in outs],
                "baselines": {b: baselines[b][pid] for b in baselines},
            }
            fh.write(json.dumps(line) + "\n")
            n_written += 1
    print(f"[ok] wrote {args.out} ({n_written} rows)")

    # ---- validation against the historical bundle -------------------------------
    if args.validate_against and args.validate_against.exists():
        old = {json.loads(l)["page_id"]: json.loads(l)
               for l in args.validate_against.read_text().splitlines() if l.strip()}
        bad_gea = bad_verdict = compared = 0
        for pid, r in rows.items():
            if pid not in old:
                continue
            compared += 1
            if abs(float(old[pid]["gea"]) - float(r.gea)) > 1e-9:
                bad_gea += 1
            if old[pid]["verdict"] != r.verdict.value:
                bad_verdict += 1
        print(f"[validate] compared {compared} pages vs {args.validate_against}")
        print(f"[validate] gea mismatches: {bad_gea}   verdict mismatches: {bad_verdict}")
        if bad_gea or bad_verdict:
            print("[validate] FAIL - replay does not reproduce the historical bundle")
            return 2
        print("[validate] OK - replay reproduces the historical bundle exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
