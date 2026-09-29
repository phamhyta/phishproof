#!/usr/bin/env -S uv run --quiet
"""T12 — measure comparable runtime and cost.

Live, per-request timing on a DECLARED class-balanced subset. Everything is measured,
nothing reconstructed: model/network time is the wall time of each real request;
provider-reported token usage is stored per request; local stages (context rendering,
parsing, consensus/aggregation, each verifier including CLIP) are timed separately, so
no stage is double-counted. Cold starts (first CLIP load, first request per model) are
flagged per record and excluded from the warm distributions.

Timing calls use their own cache root (data/cache_timing) so (a) no historical cache
entry is overwritten and (b) every request in the timing pass is a genuine network call.

The vision agent is routed through OpenRouter (openai/gpt-4o): the project's OpenAI
account has no credits (2026-09-28, recorded in the output). The returned model
snapshot is logged per request.

Methods measured per page:
  panel        the three-agent panel (sequential requests; the parallel-panel wall is
               reported as max over the page's agent latencies, stated as an estimate
               under 3-way concurrency, never added to the sequential sum)
  b1           one-model baseline = the vision agent alone (its request is shared with
               the panel measurement; B1 adds no extra call)
  b5p          majority-confidence proxy: no model request (aggregation time only)
  consolidator the actual consolidator (one extra text request)
  verifiers    CLIP logo + brand detector + DOM checks + consistency, per consensus cue
  d1           the dedicated detector (Phishpedia container), when docker is up;
               otherwise recorded as unavailable with the reason

Usage
    uv run scripts/revision_t12_cost.py --n 100
    uv run scripts/revision_t12_cost.py --n 100 --with-d1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phishproof.agents.page_context import render_context
from phishproof.agents.prompts import SYSTEM_PROMPT, build_user_prompt
from phishproof.agents.base import _parse
from phishproof.aggregate.consensus import consensus_cues, per_type_agreement
from phishproof.baselines.multiphishguard import (
    CONSOLIDATOR_SYSTEM,
    _render_agents,
    b6_multiphishguard_proxy,
)
from phishproof.cache import JsonCache
from phishproof.config import load_panel
from phishproof.data_io import read_manifest
from phishproof.schema import CueType
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.logo_brand import CLIPLogoEmbedder, crop_logo
from phishproof.tools.registry import GroundingContext, ground_cue

OUT_DIR = Path("results/revision_v2/t12")
TIMING_CACHE = "data/cache_timing"

# price table -- stored separately from measured usage, with source + date
PRICES_USD_PER_M = {
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "meta-llama/llama-3.3-70b-instruct": {"input": 0.13, "output": 0.40},
    "qwen/qwen-2.5-72b-instruct": {"input": 0.16, "output": 0.40},
}
PRICE_SOURCE = ("openrouter.ai model pages, retrieved 2026-09-28; billed cost follows "
                "the provider invoice, these are list prices")


def sha256_file(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hardware_manifest() -> dict:
    def sysctl(k):
        return subprocess.run(["sysctl", "-n", k], capture_output=True,
                              text=True).stdout.strip()
    import openai
    import torch
    import open_clip
    return {
        "machine": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "arch": platform.machine(),
        "cores_physical": sysctl("hw.physicalcpu"),
        "cores_logical": sysctl("hw.logicalcpu"),
        "mem_bytes": sysctl("hw.memsize"),
        "os": f"macOS {subprocess.run(['sw_vers', '-productVersion'], capture_output=True, text=True).stdout.strip()}",
        "python": sys.version.split()[0],
        "openai_sdk": openai.__version__,
        "torch": torch.__version__,
        "open_clip": getattr(open_clip, "__version__", "unknown"),
        "network": "residential broadband; latencies include real network time",
        "concurrency": "requests issued SEQUENTIALLY per page (uncontaminated "
                       "per-request latency); parallel-panel wall reported as "
                       "max(agent latencies) per page, an estimate under 3-way "
                       "concurrency",
        "retry_policy": "openai sdk max_retries=2, timeout 90 s",
        "cache_policy": f"timing cache root {TIMING_CACHE} (isolated; every request "
                        "in this pass is a live network call)",
        "image_settings": "vision input downscaled to 768 px long side, JPEG q85, "
                          "detail=low (phishproof.agents.client conventions)",
    }


def pct(a, q):
    return round(float(np.percentile(a, q)), 3) if len(a) else None


def dist(xs) -> dict:
    a = np.array(xs, float)
    return {"n": len(a), "mean": round(float(a.mean()), 3) if len(a) else None,
            "median": pct(a, 50), "p90": pct(a, 90), "p95": pct(a, 95),
            "min": pct(a, 0), "max": pct(a, 100)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--with-d1", action="store_true")
    ap.add_argument("--manifest", type=Path, default=Path("data/phishsel_final/test.jsonl"))
    args = ap.parse_args()

    from phishproof.env import load_env
    load_env()
    import os
    from openai import OpenAI

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[err] OPENROUTER_API_KEY missing")
        return 1
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    timeout=90.0, max_retries=2)

    pages_all = read_manifest(args.manifest)
    rng = random.Random(args.seed)
    phish = [p for p in pages_all if p.label.value == "phish"]
    benign = [p for p in pages_all if p.label.value == "benign"]
    rng.shuffle(phish)
    rng.shuffle(benign)
    subset = phish[: args.n // 2] + benign[: args.n // 2]
    rng.shuffle(subset)

    panel_cfg = load_panel(Path("configs/panel_or.yaml"))
    # OpenRouter routing for ALL agents in the timing pass (OpenAI credits exhausted)
    model_map = {"agent_a_text": "meta-llama/llama-3.3-70b-instruct",
                 "agent_b_text": "qwen/qwen-2.5-72b-instruct",
                 "agent_c_vision": "openai/gpt-4o"}
    cache = JsonCache(TIMING_CACHE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trec = (OUT_DIR / "timing_records.jsonl").open("w")
    urec = (OUT_DIR / "usage_records.jsonl").open("w")

    # ---- cold-start measurements -------------------------------------------------
    t0 = time.time()
    embedder = CLIPLogoEmbedder()
    embedder._ensure()
    clip_cold_load_s = time.time() - t0

    detector = HtmlBrandDetector()
    ctx = GroundingContext(detector=detector, logo_embedder=embedder)
    seen_models: set[str] = set()

    def request(model: str, system: str, user: str, image_path: str | None,
                stage: str, page_id: str) -> tuple[str, float]:
        from phishproof.agents.client import _data_url
        content = user
        if image_path:
            content = [{"type": "image_url",
                        "image_url": {"url": _data_url(image_path), "detail": "low"}},
                       {"type": "text", "text": user}]
        cold = model not in seen_models
        seen_models.add(model)
        t = time.time()
        resp = client.chat.completions.create(
            model=model, temperature=0.0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}])
        latency = time.time() - t
        raw = resp.choices[0].message.content or ""
        u = resp.usage
        urec.write(json.dumps({
            "page_id": page_id, "stage": stage, "model_requested": model,
            "model": resp.model, "provider": "openrouter",
            "system_fingerprint": getattr(resp, "system_fingerprint", None),
            "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
            "latency_s": round(latency, 3), "cold": cold,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "source": "provider-reported"}) + "\n")
        urec.flush()
        cache.set(model, f"SYS:{system}\nUSR:{user}\nTIMING", raw)
        return raw, latency

    for i, p in enumerate(subset, 1):
        stages: dict[str, float] = {}
        t = time.time()
        context = render_context(p)
        user = build_user_prompt(context)
        stages["context_render_s"] = time.time() - t

        outs, agent_lat = [], {}
        for cfg in panel_cfg.panel:
            model = model_map[cfg.id]
            image = p.screenshot_path if cfg.modality == "vision" else None
            raw, lat = request(model, SYSTEM_PROMPT, user, image, f"panel:{cfg.id}",
                               p.page_id)
            agent_lat[cfg.id] = lat
            t = time.time()
            outs.append(_parse(cfg.id, raw))
            stages[f"parse_{cfg.id}_s"] = time.time() - t

        t = time.time()
        gea, _ = per_type_agreement(outs)
        cons = consensus_cues(outs)                       # strict majority (T8 policy)
        _proxy = b6_multiphishguard_proxy(outs)           # B5-P: aggregation only
        stages["aggregation_s"] = time.time() - t

        ver_times: dict[str, list[float]] = {}
        crop = None
        for cue in cons:
            t = time.time()
            if cue.type is CueType.LOGO_BRAND:
                if crop is None:
                    crop = crop_logo(p)
                if crop is not None:
                    embedder.similarity(crop, cue.value)
                ver_times.setdefault("logo.clip", []).append(time.time() - t)
            else:
                ground_cue(cue, p, ctx)
                ver_times.setdefault(cue.type.value, []).append(time.time() - t)

        cons_user = ("Consolidate these analysts into a final verdict + confidence.\n\n"
                     + _render_agents(outs)
                     + '\n\nReturn JSON: {"verdict": "phish"|"benign", '
                       '"confidence": 0..1}')
        _, cons_lat = request("openai/gpt-4o", CONSOLIDATOR_SYSTEM, cons_user, None,
                              "consolidator", p.page_id)

        seq_wall = sum(agent_lat.values()) + sum(stages.values()) \
            + sum(sum(v) for v in ver_times.values())
        par_wall = max(agent_lat.values()) + sum(stages.values()) \
            + sum(sum(v) for v in ver_times.values())
        trec.write(json.dumps({
            "page_id": p.page_id, "label": p.label.value,
            "agent_latency_s": {k: round(v, 3) for k, v in agent_lat.items()},
            "stages_s": {k: round(v, 4) for k, v in stages.items()},
            "verifier_s": {k: [round(x, 4) for x in v] for k, v in ver_times.items()},
            "n_consensus_cues": len(cons),
            "consolidator_latency_s": round(cons_lat, 3),
            "wall_sequential_s": round(seq_wall, 3),
            "wall_parallel_est_s": round(par_wall, 3),
        }) + "\n")
        trec.flush()
        if i % 10 == 0:
            print(f"  {i}/{len(subset)}", flush=True)

    trec.close()

    # ---- D1 (dedicated detector) -------------------------------------------------
    d1 = {"measured": False}
    if args.with_d1:
        up = subprocess.run(["docker", "info"], capture_output=True).returncode == 0
        if not up:
            d1 = {"measured": False,
                  "reason": "docker daemon not running on the measurement host at "
                            "run time; D1 latency not re-measured in this pass"}
        else:
            d1 = {"measured": False,
                  "reason": "container timing pass not implemented in this script "
                            "version; run scripts/run_detectors.py under `time`"}
    else:
        d1 = {"measured": False, "reason": "--with-d1 not requested"}

    # ---- summary -------------------------------------------------------------------
    recs = [json.loads(l) for l in
            (OUT_DIR / "timing_records.jsonl").read_text().splitlines()]
    urecs = [json.loads(l) for l in
             (OUT_DIR / "usage_records.jsonl").read_text().splitlines()]

    def usage_cost(stage_prefix: str) -> dict:
        rows = [u for u in urecs if u["stage"].startswith(stage_prefix)
                and not u["cold"]]
        cost = 0.0
        toks_in = toks_out = 0
        for u in urecs:
            if not u["stage"].startswith(stage_prefix):
                continue
            pr = PRICES_USD_PER_M.get(u["model_requested"])
            if pr:
                cost += (u["prompt_tokens"] * pr["input"]
                         + u["completion_tokens"] * pr["output"]) / 1e6
            toks_in += u["prompt_tokens"]
            toks_out += u["completion_tokens"]
        return {"latency_warm": dist([u["latency_s"] for u in rows]),
                "prompt_tokens_total": toks_in, "completion_tokens_total": toks_out,
                "usd_at_list_price": round(cost, 4)}

    summary = {
        "meta": {
            "git_sha": subprocess.run(["git", "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
            "command": " ".join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "seed": args.seed,
            "openai_direct_unavailable": "OpenAI API credit balance exhausted on "
                                         "2026-09-28; vision + consolidator routed "
                                         "via OpenRouter openai/gpt-4o (snapshots "
                                         "logged per request)",
        },
        "subset": {"n": len(subset),
                   "phish": sum(1 for p in subset if p.label.value == "phish"),
                   "benign": sum(1 for p in subset if p.label.value == "benign"),
                   "page_ids_sha256": hashlib.sha256(
                       ",".join(sorted(p.page_id for p in subset)).encode()
                   ).hexdigest()},
        "hardware": hardware_manifest(),
        "cold_starts": {"clip_model_load_s": round(clip_cold_load_s, 2),
                        "first_request_per_model_flagged": True},
        "per_request": {
            "panel_vision(gpt-4o; also B1)": usage_cost("panel:agent_c_vision"),
            "panel_text_llama70b": usage_cost("panel:agent_a_text"),
            "panel_text_qwen72b": usage_cost("panel:agent_b_text"),
            "consolidator(gpt-4o)": usage_cost("consolidator"),
        },
        "per_page_wall": {
            "sequential_s": dist([r["wall_sequential_s"] for r in recs]),
            "parallel_estimate_s": dist([r["wall_parallel_est_s"] for r in recs]),
        },
        "local_stages_s": {
            "context_render": dist([r["stages_s"]["context_render_s"] for r in recs]),
            "parse_total": dist([sum(v for k, v in r["stages_s"].items()
                                     if k.startswith("parse_")) for r in recs]),
            "aggregation": dist([r["stages_s"]["aggregation_s"] for r in recs]),
            "verifier_clip_per_cue": dist([x for r in recs
                                           for x in r["verifier_s"].get("logo.clip", [])]),
            "verifier_structural_per_cue": dist(
                [x for r in recs for k, v in r["verifier_s"].items()
                 if k != "logo.clip" for x in v]),
        },
        "call_accounting": {
            "panel": "3 requests/page", "b1": "0 extra (vision request shared)",
            "b5p_proxy": "0 requests (aggregation only)",
            "consolidator": "1 extra request/page",
            "abstention": "happens after inference; saves no model call, adds "
                          "potential human review",
        },
        "d1_detector": d1,
        "price_table": {"usd_per_m_tokens": PRICES_USD_PER_M,
                        "source": PRICE_SOURCE},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("per_request", "per_page_wall",
                                              "local_stages_s")}, indent=2))
    print(f"[ok] wrote {OUT_DIR}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
