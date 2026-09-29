#!/usr/bin/env -S uv run --quiet
"""T9 — execute the requested baseline mechanisms.

Three subcommands, run in this order:

  recover-smaller   Recover the smaller-panel consolidator's PROVENANCE from the response
                    cache: for every bundle_D page, replay the 3B panel outputs (cache
                    only), rebuild the exact consolidator prompt, look up the stored raw
                    GPT-4o response, re-parse it, and compare with the bundle's stored
                    baselines.B6. Detects silent parse-fallbacks (score == proxy). ZERO
                    API calls; a cache miss is recorded as unrecovered, never fetched.

  run               Run the ACTUAL consolidator (a separately identified model call) on
                    the recorded larger-panel page outputs for one corpus. Uses the same
                    cache key convention as ChatClient so a re-run is free, and logs
                    per-request usage + latency (T12) during this single paid pass.
                    Pages whose panel outputs were never recovered (replay_failed) are
                    recorded as missing, not fabricated.

  dawid-skene       Disclose B5's (paper label: B4 weighted label agreement) fitting
                    population and EM settings, and produce BOTH variants the plan
                    allows: the published transductive fit (clearly labelled) and a
                    calibration-frozen fit (confusion matrices + priors fitted on the
                    calibration bundle, then applied to test pages without refitting).

Usage
    uv run scripts/revision_t9_baselines.py recover-smaller
    uv run scripts/revision_t9_baselines.py run --corpus phishpedia
    uv run scripts/revision_t9_baselines.py dawid-skene
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phishproof.baselines.multiphishguard import (
    CONSOLIDATOR_SYSTEM,
    _render_agents,
    b6_multiphishguard_proxy,
)
from phishproof.cache import JsonCache
from phishproof.schema import AgentOutput, Cue, CueType, Label

OUT_DIR = Path("results/revision_v2/t9")
CONSOLIDATOR_MODEL = "gpt-4o"

BUNDLES = {
    "phishpedia": "results/revision/bundle_or_full.jsonl",
    "apwg": "results/revision/bundle_apwg_or_full.jsonl",
    "trop": "results/revision/bundle_trop_or_full.jsonl",
}


def sha256_file(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def meta(extra: dict | None = None) -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    d = {
        "git_sha": sha,
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "consolidator_model": CONSOLIDATOR_MODEL,
        "consolidator_prompt_sha256": hashlib.sha256(
            CONSOLIDATOR_SYSTEM.encode()).hexdigest(),
        "schema": '{"verdict": "phish"|"benign", "confidence": 0..1} (strict JSON)',
    }
    if extra:
        d.update(extra)
    return d


def outs_from_bundle_row(row: dict) -> list[AgentOutput] | None:
    """Rebuild the exact AgentOutput list the consolidator prompt renders from."""
    if row.get("replay_failed") or not row.get("agents"):
        return None
    outs = []
    for a in row["agents"]:
        cues = [Cue(type=CueType(c["type"]), value=c["value"])
                for c in a.get("cues", [])]
        outs.append(AgentOutput(agent_id=a["id"], verdict=Label(a["verdict"]),
                                cues=cues, confidence=a.get("confidence")))
    return outs


def consolidator_prompt(outs: list[AgentOutput]) -> str:
    return ("Consolidate these analysts into a final verdict + confidence.\n\n"
            + _render_agents(outs)
            + '\n\nReturn JSON: {"verdict": "phish"|"benign", "confidence": 0..1}')


def cache_key_prompt(user: str) -> str:
    # ChatClient convention: text-modality AgentConfig has detail=None
    return f"SYS:{CONSOLIDATOR_SYSTEM}\nUSR:{user}\nDETAIL:None"


def parse_consolidator(raw: str) -> dict:
    """Strict re-parse of the consolidator's raw response (never a silent fallback)."""
    try:
        txt = raw[raw.find("{"): raw.rfind("}") + 1]
        d = json.loads(txt)
        verdict = str(d.get("verdict", "")).lower()
        conf = d.get("confidence", None)
        ok = verdict in ("phish", "benign") and isinstance(conf, (int, float)) \
            and 0.0 <= float(conf) <= 1.0
        return {"parse_ok": ok, "verdict": verdict if ok else None,
                "confidence": float(conf) if ok else None}
    except Exception:  # noqa: BLE001
        return {"parse_ok": False, "verdict": None, "confidence": None}


# ------------------------------------------------------------------ recover-smaller
def recover_smaller(args) -> int:
    """Provenance for the smaller-panel consolidator: cache lookups only."""
    from phishproof.agents.base import _parse
    from phishproof.agents.page_context import render_context
    from phishproof.agents.prompts import SYSTEM_PROMPT, build_user_prompt
    from phishproof.config import load_panel
    from phishproof.data_io import read_manifest

    cache = JsonCache()
    panel_cfg = load_panel(Path("configs/panel.yaml"))       # the 3B smaller panel
    pages = {p.page_id: p for p in read_manifest(Path("data/phishsel_final/test.jsonl"))}
    bundle = [json.loads(l) for l in Path(args.bundle).read_text().splitlines()
              if l.strip()]

    recs, stats = [], Counter()
    t0 = time.time()
    for i, row in enumerate(bundle, 1):
        pid = row["page_id"]
        page = pages.get(pid)
        if page is None:
            stats["no_manifest_page"] += 1
            continue
        outs = []
        panel_miss = False
        for cfg in panel_cfg.panel:
            user = build_user_prompt(render_context(page))
            image = page.screenshot_path if cfg.modality == "vision" else None
            img_hash = JsonCache.hash_image(image) if image else None
            raw = cache.get(cfg.model, f"SYS:{SYSTEM_PROMPT}\nUSR:{user}"
                                       f"\nDETAIL:{cfg.detail}", img_hash)
            if raw is None:
                panel_miss = True
                break
            outs.append(_parse(cfg.id, raw))
        if panel_miss:
            stats["panel_cache_miss"] += 1
            recs.append({"page_id": pid, "status": "panel_unrecovered"})
            continue

        user = consolidator_prompt(outs)
        raw = cache.get(CONSOLIDATOR_MODEL, cache_key_prompt(user))
        stored = row.get("baselines", {}).get("B6")
        proxy = b6_multiphishguard_proxy(outs)
        if raw is None:
            stats["consolidator_cache_miss"] += 1
            recs.append({"page_id": pid, "status": "consolidator_unrecovered",
                         "stored_B6": stored, "proxy": proxy,
                         "stored_equals_proxy": stored is not None
                         and abs(stored - proxy) < 1e-9})
            continue
        parsed = parse_consolidator(raw)
        fallback = not parsed["parse_ok"]
        match = (stored is not None and parsed["confidence"] is not None
                 and abs(stored - parsed["confidence"]) < 1e-9)
        match_proxy = stored is not None and abs(stored - proxy) < 1e-9
        stats["recovered"] += 1
        stats["parse_fallback"] += int(fallback)
        stats["stored_matches_parsed_conf"] += int(match)
        stats["stored_matches_proxy"] += int(match_proxy)
        recs.append({
            "page_id": pid, "status": "recovered",
            "prompt_sha256": hashlib.sha256(user.encode()).hexdigest(),
            "raw_response": raw,
            **parsed,
            "fallback_used": fallback,
            "stored_B6": stored, "proxy": proxy,
            "stored_matches_parsed_conf": match,
            "stored_matches_proxy": match_proxy,
        })
        if i % 400 == 0:
            print(f"  {i}/{len(bundle)}  {time.time()-t0:.0f}s", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "consolidator_provenance_smaller.jsonl"
    with out.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    summary = {
        "meta": meta({"panel_config": "configs/panel.yaml (3B smaller panel)",
                      "bundle": str(args.bundle),
                      "bundle_sha256": sha256_file(args.bundle)}),
        "n_bundle_rows": len(bundle),
        "counts": dict(stats),
        "note": "provenance is the stored raw response + strict re-parse; the stored "
                "B6 score field alone is never treated as provenance",
    }
    (OUT_DIR / "consolidator_provenance_smaller_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary["counts"], indent=2))
    print(f"[ok] wrote {out}")
    return 0


# ------------------------------------------------------------------ run (actual consolidator)
def run_consolidator(args) -> int:
    from phishproof.env import load_env
    load_env()
    import os

    from openai import OpenAI

    if args.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("[err] OPENAI_API_KEY missing")
            return 1
        client = OpenAI(timeout=90.0, max_retries=2)
        model = CONSOLIDATOR_MODEL
    else:  # openrouter (same underlying gpt-4o; recorded via provider + returned model)
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("[err] OPENROUTER_API_KEY missing")
            return 1
        client = OpenAI(base_url="https://openrouter.ai/api/v1",
                        api_key=os.environ["OPENROUTER_API_KEY"],
                        timeout=90.0, max_retries=2)
        model = "openai/gpt-4o"
    cache = JsonCache()

    bundle_path = Path(BUNDLES[args.corpus])
    rows = [json.loads(l) for l in bundle_path.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"consolidator_{args.corpus}.jsonl"
    usage_path = OUT_DIR / f"consolidator_usage_{args.corpus}.jsonl"
    done: set[str] = set()
    if out_path.exists():                       # resumable: skip already-consolidated pages
        done = {json.loads(l)["page_id"] for l in out_path.read_text().splitlines()
                if l.strip()}

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    stats = Counter()
    t0 = time.time()
    wlock = threading.Lock()

    def process(row: dict) -> None:
        pid = row["page_id"]
        outs = outs_from_bundle_row(row)
        if outs is None:
            with wlock:
                stats["panel_missing"] += 1
                fh.write(json.dumps({"page_id": pid, "status": "panel_missing",
                                     "reason": "replay_failed panel outputs; the "
                                     "consolidator has no inputs"}) + "\n")
            return
        user = consolidator_prompt(outs)
        key_prompt = cache_key_prompt(user)
        raw = cache.get(model, key_prompt)
        usage_rec = None
        if raw is None:
            t_req = time.time()
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[{"role": "system", "content": CONSOLIDATOR_SYSTEM},
                          {"role": "user", "content": user}],
            )
            latency = time.time() - t_req
            raw = resp.choices[0].message.content or ""
            cache.set(model, key_prompt, raw)
            u = resp.usage
            usage_rec = {
                "page_id": pid, "corpus": args.corpus,
                "provider": args.provider,
                "model_requested": model,
                "model": resp.model, "system_fingerprint":
                    getattr(resp, "system_fingerprint", None),
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
                "latency_s": round(latency, 3),
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "source": "provider-reported",
            }
        parsed = parse_consolidator(raw)
        panel_verdict = row.get("verdict")
        rec = {
            "page_id": pid, "status": "ok",
            "prompt_sha256": hashlib.sha256(user.encode()).hexdigest(),
            "raw_response": raw,
            **parsed,
            "panel_verdict": panel_verdict,
            "changes_prediction": (parsed["verdict"] is not None
                                   and parsed["verdict"] != panel_verdict),
            "label": row.get("label"),
            "proxy_b5p": b6_multiphishguard_proxy(outs),
        }
        with wlock:
            if usage_rec is not None:
                uh.write(json.dumps(usage_rec) + "\n")
                uh.flush()
                stats["api_calls"] += 1
            else:
                stats["cache_hits"] += 1
            stats["parse_fail"] += int(not parsed["parse_ok"])
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            stats["done"] += 1
            if stats["done"] % 100 == 0:
                el = time.time() - t0
                print(f"  {args.corpus} {stats['done']}/{len(rows)} "
                      f"api={stats['api_calls']} cache={stats['cache_hits']} "
                      f"{el:.0f}s", flush=True)

    todo = [r for r in rows if r["page_id"] not in done]
    stats["skipped_done"] = len(rows) - len(todo)
    with out_path.open("a") as fh, usage_path.open("a") as uh:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process, r) for r in todo]
            for f in as_completed(futs):
                f.result()   # surface the first worker exception

    summary = {
        "meta": meta({"bundle": str(bundle_path),
                      "bundle_sha256": sha256_file(bundle_path),
                      "settings": {"temperature": 0.0, "timeout_s": 90,
                                   "max_retries": 2, "provider": args.provider,
                                   "model_requested": model}}),
        "corpus": args.corpus,
        "counts": dict(stats),
    }
    (OUT_DIR / f"consolidator_{args.corpus}_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary["counts"], indent=2))
    print(f"[ok] wrote {out_path} (+usage log)")
    return 0


# ------------------------------------------------------------------ dawid-skene
def _ds_fit(obs: list[dict[int, int]], n_workers: int, n_iter: int = 50):
    """Dawid-Skene EM (mirrors phishproof.baselines.dawid_skene) + convergence trace."""
    n_items, n_classes = len(obs), 2
    T = np.full((n_items, n_classes), 0.5)
    for i, o in enumerate(obs):
        c = np.zeros(n_classes)
        for _w, l in o.items():
            c[l] += 1
        if c.sum():
            T[i] = c / c.sum()
    deltas = []
    prior = conf = None
    for _ in range(n_iter):
        prior = T.mean(0) + 1e-9
        prior /= prior.sum()
        conf = np.full((n_workers, n_classes, n_classes), 1e-9)
        for i, o in enumerate(obs):
            for w, l in o.items():
                conf[w, :, l] += T[i]
        conf /= conf.sum(axis=2, keepdims=True)
        newT = np.zeros_like(T)
        for i, o in enumerate(obs):
            logp = np.log(prior).copy()
            for w, l in o.items():
                logp += np.log(conf[w, :, l])
            p = np.exp(logp - logp.max())
            newT[i] = p / p.sum()
        deltas.append(float(np.abs(newT - T).max()))
        T = newT
    return T, prior, conf, deltas


def _ds_apply(obs: list[dict[int, int]], prior: np.ndarray, conf: np.ndarray):
    """Posterior inference with FROZEN prior + confusion matrices (no refitting)."""
    out = np.zeros((len(obs), 2))
    for i, o in enumerate(obs):
        logp = np.log(prior).copy()
        for w, l in o.items():
            logp += np.log(conf[w, :, l])
        p = np.exp(logp - logp.max())
        out[i] = p / p.sum()
    return out


def dawid_skene_disclosure(args) -> int:
    lab = {"benign": 0, "phish": 1}

    def load_obs(path: str):
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r.get("agents") and not r.get("replay_failed")]
        workers = sorted({a["id"] for r in rows for a in r["agents"]})
        widx = {w: i for i, w in enumerate(workers)}
        obs = [{widx[a["id"]]: lab[a["verdict"]] for a in r["agents"]} for r in rows]
        return rows, workers, obs

    out: dict = {"meta": meta(), "results": {}}
    calib_rows, calib_workers, calib_obs = load_obs("results/bundle_calib_or.jsonl")
    _, calib_prior, calib_conf, calib_deltas = _ds_fit(calib_obs, len(calib_workers))

    for corpus, path in BUNDLES.items():
        rows, workers, obs = load_obs(path)
        assert workers == calib_workers, f"worker mismatch on {corpus}"
        T_trans, prior_t, conf_t, deltas_t = _ds_fit(obs, len(workers))
        T_frozen = _ds_apply(obs, calib_prior, calib_conf)
        correct = np.array([r["verdict"] == r["label"] for r in rows], float)
        s_trans = T_trans.max(axis=1)
        s_frozen = T_frozen.max(axis=1)
        rho = float(np.corrcoef(s_trans, s_frozen)[0, 1])
        out["results"][corpus] = {
            "bundle": path,
            "fitting_population_transductive": f"the {len(rows)} scored test rows of "
                                               f"{corpus} (published variant)",
            "n_rows": len(rows),
            "em": {"n_iter": 50, "convergence_last_delta_transductive": deltas_t[-1],
                   "init": "vote-share posteriors", "smoothing": "1e-9 additive",
                   "note": "fixed 50 iterations, no early-stopping criterion; "
                           "this EM is FITTED, not closed-form or parameter-free"},
            "frozen_variant": {
                "fit_on": "results/bundle_calib_or.jsonl "
                          f"({len(calib_rows)} calibration rows, larger panel)",
                "convergence_last_delta": calib_deltas[-1],
                "applied": "posterior inference only on test rows (no refit)",
            },
            "score_correlation_trans_vs_frozen": round(rho, 4),
            "acc_of_argmax_transductive": round(float(
                (T_trans.argmax(1) == np.array([lab[r["label"]] for r in rows])).mean()), 4),
            "mean_correct": round(float(correct.mean()), 4),
        }
        per_page = {r["page_id"]: {"transductive": float(a), "frozen": float(b)}
                    for r, a, b in zip(rows, s_trans, s_frozen)}
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"dawid_skene_scores_{corpus}.json").write_text(
            json.dumps(per_page))

    (OUT_DIR / "dawid_skene_disclosure.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "bundle"}
                      for k, v in out["results"].items()}, indent=2))
    print(f"[ok] wrote {OUT_DIR}/dawid_skene_disclosure.json (+per-corpus scores)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recover-smaller")
    r.add_argument("--bundle", type=Path, default=Path("results/bundle_D.jsonl"))
    x = sub.add_parser("run")
    x.add_argument("--corpus", choices=list(BUNDLES), required=True)
    x.add_argument("--limit", type=int, default=0)
    x.add_argument("--workers", type=int, default=8)
    x.add_argument("--provider", choices=["openai", "openrouter"], default="openrouter",
                   help="openai account has no credits (2026-09-28); openrouter serves "
                        "the same gpt-4o and is the panel's existing provider")
    sub.add_parser("dawid-skene")
    args = ap.parse_args()
    if args.cmd == "recover-smaller":
        return recover_smaller(args)
    if args.cmd == "run":
        return run_consolidator(args)
    return dawid_skene_disclosure(args)


if __name__ == "__main__":
    raise SystemExit(main())
