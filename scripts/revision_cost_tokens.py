"""R2.6 / R4.13 -- measured token counts and recomputed per-page cost (no API calls).

The cost table's caption said token counts were unrecoverable because "the runs predate
token-level accounting". They are recoverable: every prompt is a deterministic function of the
page (render_context -> build_user_prompt) and every response is in data/cache/, so re-rendering
the prompts and tokenising them with the model's own tokenizer gives exact counts after the fact.

Doing that shows the published dollar figures are too high by about 3x, and the cause is
identifiable rather than mysterious. The panel's vision agent runs at `detail: low`
(configs/panel_or.yaml), which OpenAI bills as a flat 85 image tokens, and the caption itself
says "low image detail" -- but the published ~$8/1k for the vision call matches the arithmetic
for a HIGH-detail image at the superseded $5/$15 GPT-4o price. At low detail and current list
prices the same call is ~$2.8/1k.

PRICES ARE NOT MEASURED. They are list prices that change; `--price` overrides them and the JSON
records what was used, so the figure can be refreshed at submission time without re-deriving the
token counts.

Usage
    uv run scripts/revision_cost_tokens.py --out results/revision/t7_cost_tokens.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken

from phishproof.agents.page_context import render_context
from phishproof.agents.prompts import SYSTEM_PROMPT, build_user_prompt
from phishproof.baselines.multiphishguard import CONSOLIDATOR_SYSTEM, _render_agents
from phishproof.data_io import read_manifest
from phishproof.schema import AgentOutput, Cue, CueType, Label
from phishproof.tools.detectors.phishllm import SYSTEM as PHISHLLM_SYSTEM

# GPT-4o image accounting: low detail is a flat 85 tokens; high detail adds 170 per 512px tile.
IMG_LOW = 85
IMG_HIGH = 85 + 170 * 4

# USD per 1M tokens (input, output). List prices -- verify before submission.
DEFAULT_PRICES = {
    "gpt-4o": (2.50, 10.00),
    "openrouter-70b": (0.30, 0.40),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/phishsel_final/test.jsonl"))
    ap.add_argument("--bundle", type=Path,
                    default=Path("results/revision/bundle_or_full.jsonl"))
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--price", action="append", default=[],
                    metavar="MODEL=IN,OUT", help="override a list price, USD per 1M tokens")
    ap.add_argument("--out", type=Path, default=Path("results/revision/t7_cost_tokens.json"))
    args = ap.parse_args()

    prices = dict(DEFAULT_PRICES)
    for spec in args.price:
        model, pair = spec.split("=", 1)
        pin, pout = pair.split(",")
        prices[model] = (float(pin), float(pout))

    enc = tiktoken.get_encoding("o200k_base")
    pages = read_manifest(args.manifest)[: args.sample]

    # --- panel agent prompts -------------------------------------------------
    agent_in = [len(enc.encode(SYSTEM_PROMPT)) + len(enc.encode(build_user_prompt(
        render_context(p)))) for p in pages]

    # --- panel responses, from the cache -------------------------------------
    rows = [json.loads(l) for l in args.bundle.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("agents")][: args.sample]
    agent_out = []
    for r in rows:
        for a in r["agents"]:
            cues = "".join(f'{{"type":"{c["type"]}","value":"{c["value"]}"}},'
                           for c in a.get("cues", []))
            blob = (f'{{"verdict":"{a["verdict"]}","confidence":{a["confidence"]},'
                    f'"cues":[{cues.rstrip(",")}]}}')
            agent_out.append(len(enc.encode(blob)))

    # --- consolidator prompt (gpt-4o, text only) -----------------------------
    cons_in = []
    for r in rows[:100]:
        outs = [AgentOutput(agent_id=a["id"], verdict=Label(a["verdict"]),
                            confidence=a["confidence"],
                            cues=[Cue(type=CueType(c["type"]), value=c["value"],
                                      raw_value=c["value"], asserted_by=a["id"])
                                  for c in a.get("cues", [])])
                for a in r["agents"]]
        user = ("Consolidate these analysts into a final verdict + confidence.\n\n"
                + _render_agents(outs)
                + '\n\nReturn JSON: {"verdict": "phish"|"benign", "confidence": 0..1}')
        cons_in.append(len(enc.encode(CONSOLIDATOR_SYSTEM)) + len(enc.encode(user)))

    # D3 PhishLLM: one gpt-4o call per page, vision at detail=low (phishllm.py), so its
    # image is billed the same flat 85 tokens as our own vision agent.
    d3_in = [len(enc.encode(PHISHLLM_SYSTEM)) + len(enc.encode(
        "Check this page.\n\n" + render_context(p)
        + '\n\nReturn JSON {"brand","credential","domain_is_official","verdict"}.'))
        for p in pages[:100]]

    tin, tout = statistics.mean(agent_in), statistics.mean(agent_out)
    cin = statistics.mean(cons_in)
    cons_out = 20  # {"verdict":"...","confidence":0.xx}

    def usd_per_1k(model: str, n_in: float, n_out: float, image: int = 0) -> float:
        pin, pout = prices[model]
        return ((n_in + image) * pin + n_out * pout) / 1e6 * 1000

    vision = usd_per_1k("gpt-4o", tin, tout, IMG_LOW)
    text2 = 2 * usd_per_1k("openrouter-70b", tin, tout)
    cons = usd_per_1k("gpt-4o", cin, cons_out)
    d3 = usd_per_1k("gpt-4o", statistics.mean(d3_in), 40, IMG_LOW)

    out = {
        "prices_usd_per_1m": {k: list(v) for k, v in prices.items()},
        "prices_are_list_not_billed": True,
        "tokenizer": "o200k_base",
        "measured_tokens": {
            "panel_agent_input_mean": round(tin, 1),
            "panel_agent_input_median": round(statistics.median(agent_in), 1),
            "panel_agent_input_p90": round(sorted(agent_in)[int(0.9 * len(agent_in))], 1),
            "panel_agent_output_mean": round(tout, 1),
            "consolidator_input_mean": round(cin, 1),
            "d3_phishllm_input_mean": round(statistics.mean(d3_in), 1),
            "image_tokens_detail_low": IMG_LOW,
            "image_tokens_detail_high_768px": IMG_HIGH,
            "n_pages_sampled": len(pages),
        },
        "usd_per_1k_pages": {
            "panel_total_3_calls": round(vision + text2, 2),
            "vision_gpt4o_only": round(vision, 2),
            "two_text_70b_only": round(text2, 2),
            "consolidator_b5": round(cons, 2),
            "single_text_call_baseline": round(usd_per_1k("openrouter-70b", tin, tout), 2),
            "d3_phishllm_one_vision_call": round(d3, 2),
        },
        "published_figures_for_comparison": {
            "panel_total": 11, "vision_only": 8, "consolidator": 3, "single_call": 2,
            "d3_phishllm": 5},
        "why_the_published_figure_is_high": (
            "~$8/1k for the vision call reproduces only under HIGH image detail at the "
            "superseded $5/$15 GPT-4o price: (796+765)*5 + 58*15 per 1M = $8.68/1k. The code "
            "sets detail=low (configs/panel_or.yaml) and the caption says low detail, so the "
            "image is billed at a flat 85 tokens, not 765."),
        "diagnostic_variants_usd_per_1k_vision": {
            "low_detail_current_price": round(usd_per_1k("gpt-4o", tin, tout, IMG_LOW), 2),
            "high_detail_current_price": round(usd_per_1k("gpt-4o", tin, tout, IMG_HIGH), 2),
            "low_detail_old_5_15": round(((tin + IMG_LOW) * 5 + tout * 15) / 1e6 * 1000, 2),
            "high_detail_old_5_15": round(((tin + IMG_HIGH) * 5 + tout * 15) / 1e6 * 1000, 2),
        },
    }
    text = json.dumps(out, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
