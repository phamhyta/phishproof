"""Patch B7 = PhishDebate (role-based debate + moderator) into a results bundle.

Re-runs the panel (cached -> instant) to recover each page's round-0 positions, runs R rounds
of cross-examination among the same M agents (text calls), then a GPT-4o moderator resolves the
debate into a confidence used as the selective trust score. Overwrites the bundle's
baselines.B7 with that confidence. Then re-run run_experiments.py.

Resilient + resumable: each page is wrapped in a retry/try-except (a transient API timeout
falls back to the panel-confidence proxy instead of aborting the whole run), the bundle is
checkpointed every --checkpoint pages, and pages that already carry baselines.<field> are
skipped on resume (re-run the same command to continue; pass --force to recompute).

Usage:
    .venv/bin/python scripts/run_phishdebate.py --data data/phishsel_final \
        --bundle results/bundle_D.jsonl --rounds 2
    # pilot:  --bundle results/_b7_pilot.jsonl --limit 20
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
from phishproof.baselines.multiphishguard import b6_multiphishguard_proxy
from phishproof.baselines.phishdebate import PhishDebate
from phishproof.config import AgentConfig, load_panel
from phishproof.data_io import read_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--moderator", default="gpt-4o", help="moderator/judge model")
    ap.add_argument("--limit", type=int, default=0, help="0 = all pages; e.g. 20 for a pilot")
    ap.add_argument("--field", default="B7", help="bundle baselines.<field> to write")
    ap.add_argument("--checkpoint", type=int, default=250, help="flush the bundle every N pages")
    ap.add_argument("--force", action="store_true", help="recompute pages that already have the field")
    ap.add_argument("--api-only", action="store_true",
                    help="route ALL debaters through --moderator (no local Ollama; round-0 stays cached)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel page workers (use with --api-only; cache is file-per-key => thread-safe)")
    args = ap.parse_args()

    pages = {p.page_id: p for p in read_manifest(args.data / "test.jsonl")}
    bundle = [json.loads(l) for l in args.bundle.read_text().splitlines() if l.strip()]

    client = ChatClient()
    panel_cfg = load_panel()
    panel = Panel.from_config(panel_cfg, client)                  # cached round-0 agent calls
    # Debaters = the same panel agents, reasoning over TEXT in the debate rounds (each agent's
    # round-0 visual/textual evidence is already in its cited cues, recovered above).
    # Debaters reason over TEXT in the debate rounds; route the paid (OpenAI) agent through the
    # SAME model as the moderator, so a cheap moderator (e.g. gpt-4o-mini) yields a cheap run --
    # the local Ollama agents stay free either way.  With --api-only, route EVERY debater through
    # the moderator model: the debate rounds become pure API calls (no local Ollama load), while
    # round-0 evidence still comes from the real cached panel (cue text already recovered above).
    debater_cfgs = []
    for c in panel_cfg.panel:
        if args.api_only:
            upd = {"modality": "text", "provider": "openai", "model": args.moderator, "base_url": None}
        else:
            upd = {"modality": "text"}
            if c.provider == "openai":
                upd["model"] = args.moderator
        debater_cfgs.append(c.model_copy(update=upd))
    mod_cfg = AgentConfig(id="phishdebate_moderator", provider="openai",
                          model=args.moderator, modality="text")
    debate = PhishDebate(client, debater_cfgs, mod_cfg, rounds=args.rounds)

    def checkpoint() -> None:
        tmp = args.bundle.with_suffix(args.bundle.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in bundle:
                f.write(json.dumps(r) + "\n")
        tmp.replace(args.bundle)                                   # atomic: a crash can't truncate

    def score_page(page, attempts: int = 3):
        """Debate score with retries+backoff; on repeated failure fall back to the panel-conf proxy."""
        outs: list = []
        last = None
        for k in range(attempts):
            try:
                outs = panel.run(page)                            # cache hit (round 0)
                _verdict, conf = debate.score(outs, page)
                return conf, True
            except Exception as e:  # noqa: BLE001 - one transient error must not abort the sweep
                last = e
                time.sleep(1.5 * (k + 1))                          # back off (rate-limit friendly)
        fb = b6_multiphishguard_proxy(outs) if outs else 0.5
        print(f"  [warn] {page.page_id} failed after {attempts} ({last}); fallback {args.field}={fb:.3f}",
              flush=True)
        return fb, False

    rows = bundle if not args.limit else bundle[: args.limit]
    n, t0 = len(rows), time.time()
    todo = [row for row in rows
            if (args.force or row.get("baselines", {}).get(args.field) is None)
            and pages.get(row["page_id"]) is not None]
    skipped = n - len(todo)
    done = failed = 0
    lock = threading.Lock()
    print(f"[start] {len(todo)} pages to score, {skipped} skipped/resumed, workers={args.workers}", flush=True)

    def finish(row, conf: float, ok: bool) -> None:
        """Record one page's result + periodic progress/checkpoint (call under `lock`)."""
        nonlocal done, failed
        row.setdefault("baselines", {})[args.field] = conf
        done += 1
        failed += 0 if ok else 1
        if done % 20 == 0:
            rate = (time.time() - t0) / max(1, done)
            eta = rate * (len(todo) - done) / 60
            print(f"  {done}/{len(todo)}  (skip {skipped}, fail {failed}, "
                  f"{rate:.2f}s/page wall, ~{eta:.0f} min left)", flush=True)
        if done % args.checkpoint == 0:
            checkpoint()
            print(f"  [checkpoint] flushed {args.bundle} at done={done}/{len(todo)}", flush=True)

    if args.workers > 1:                                          # parallel: API-only, file-per-key cache
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(score_page, pages[row["page_id"]]): row for row in todo}
            for fut in as_completed(futs):
                conf, ok = fut.result()
                with lock:
                    finish(futs[fut], conf, ok)
    else:
        for row in todo:
            conf, ok = score_page(pages[row["page_id"]])
            finish(row, conf, ok)

    checkpoint()
    print(f"[ok] patched baselines.{args.field} (PhishDebate, R={args.rounds}) -- "
          f"done {done}, skipped {skipped}, fallback {failed}, in {args.bundle}")
    print(f"     next: .venv/bin/python scripts/run_experiments.py "
          f"--bundle {args.bundle} --out results/_D")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
