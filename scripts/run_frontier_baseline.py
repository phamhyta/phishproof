"""Frontier-model single-call baselines for Q1-reviewer robustness check.

Run Claude (Sonnet 4.5) and Gemini (2.5 Pro/Flash) as single-vision-call detectors
on a stratified subset of PhishSel test. Reports detection F1 + ECE + AURC so the
paper can claim "even a strongest frontier single-model baseline does not beat
PhishProof on selective trust" — closing the "what about better LLM" critique.

Cost (60 pages): ~$3-5 total. Time: ~5 min.

Usage:
  export ANTHROPIC_API_KEY=...
  export GOOGLE_API_KEY=...
  python scripts/run_frontier_baseline.py --n 60

Outputs:
  results/frontier_claude.jsonl
  results/frontier_gemini.jsonl
  results/frontier_summary.json
"""
from __future__ import annotations
import argparse
import base64
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phishproof.agents.page_context import build_page_context  # noqa: E402
from phishproof.eval.evaluate import evaluate_method  # noqa: E402


PROMPT = (
    "You are a phishing detector. Given a webpage screenshot and a compact page context, "
    "decide whether the page is PHISHING or BENIGN. Output JSON only: "
    '{"verdict": "phish|benign", "confidence": 0.0..1.0, "brief": "..."}.\n\n'
    "Page context:\n{ctx}\n"
)


def b64_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


def call_claude(ctx: str, img_path: Path, model: str = "claude-sonnet-4-5") -> dict:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": b64_image(img_path)}},
                {"type": "text", "text": PROMPT.format(ctx=ctx[:8000])},
            ],
        }],
    )
    txt = msg.content[0].text
    return parse_json(txt)


def call_gemini(ctx: str, img_path: Path, model: str = "gemini-2.5-flash") -> dict:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    m = genai.GenerativeModel(model)
    resp = m.generate_content([
        PROMPT.format(ctx=ctx[:8000]),
        {"mime_type": "image/png", "data": img_path.read_bytes()},
    ])
    return parse_json(resp.text)


def parse_json(txt: str) -> dict:
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.strip("`").lstrip("json").strip()
    try:
        j = json.loads(txt)
    except Exception:
        # very lenient fallback
        import re
        m = re.search(r"\{[^{}]*\}", txt, re.S)
        j = json.loads(m.group()) if m else {"verdict": "benign", "confidence": 0.5}
    v = str(j.get("verdict", "benign")).lower()
    if v.startswith("p"):
        v = "phish"
    else:
        v = "benign"
    conf = float(j.get("confidence", 0.5))
    return {"verdict": v, "confidence": max(0.0, min(1.0, conf))}


def load_test(n: int, seed: int) -> list[dict]:
    test_file = ROOT / "data/phishsel_final/test.jsonl"
    rows = [json.loads(l) for l in test_file.read_text().splitlines() if l.strip()]
    random.seed(seed)
    # stratify
    phish = [r for r in rows if r["label"] == "phish"]
    benign = [r for r in rows if r["label"] == "benign"]
    random.shuffle(phish); random.shuffle(benign)
    half = n // 2
    return phish[:half] + benign[:half]


def run_provider(provider: str, model: str, items: list[dict], out_path: Path) -> list[dict]:
    print(f"\n=== {provider} {model} ===")
    results = []
    for i, r in enumerate(items):
        raw = Path(r["raw_dir"])
        shot = raw / "shot.png"
        if not shot.exists():
            print(f"  skip {r['page_id']} (no shot)")
            continue
        html = (raw / "html.txt").read_text(errors="replace") if (raw / "html.txt").exists() else ""
        ctx = build_page_context(html=html, url=r.get("url", ""))
        try:
            if provider == "claude":
                resp = call_claude(ctx, shot, model)
            else:
                resp = call_gemini(ctx, shot, model)
        except Exception as e:
            print(f"  err {r['page_id']}: {e}")
            continue
        results.append({
            "page_id": r["page_id"], "label": r["label"],
            "verdict": resp["verdict"], "confidence": resp["confidence"],
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(items)}")
    out_path.write_text("\n".join(json.dumps(r) for r in results))
    return results


def summarize(name: str, rows: list[dict]) -> dict:
    if not rows:
        return {}
    yt = np.array([r["label"] == "phish" for r in rows])
    yp = np.array([r["verdict"] == "phish" for r in rows])
    conf = np.array([r["confidence"] for r in rows])
    # Use confidence as trust (no calibration; reviewer wants raw baseline).
    res = evaluate_method(conf, yt, yp, trust=conf, n_boot=500, seed=0)
    acc = float((yt == yp).mean())
    out = {"n": len(rows), "accuracy": acc}
    for k in ("AURC", "SelAcc80", "FPR80", "Cov99", "ECE"):
        out[k] = {"point": res[k][0], "ci_lo": res[k][1], "ci_hi": res[k][2]}
    print(f"\n{name}: n={len(rows)} acc={acc:.3f}")
    for k in ("AURC", "SelAcc80", "ECE"):
        p = res[k][0] * (100 if k in ("AURC", "SelAcc80") else 1)
        print(f"  {k:>10} {p:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--provider", default="both", choices=["claude", "gemini", "both"])
    ap.add_argument("--claude-model", default="claude-sonnet-4-5")
    ap.add_argument("--gemini-model", default="gemini-2.5-flash")
    args = ap.parse_args()

    items = load_test(args.n, args.seed)
    print(f"loaded {len(items)} stratified test pages")

    summary = {}
    if args.provider in ("claude", "both"):
        out = ROOT / "results/frontier_claude.jsonl"
        rows = run_provider("claude", args.claude_model, items, out)
        summary[f"claude:{args.claude_model}"] = summarize("Claude", rows)
    if args.provider in ("gemini", "both"):
        out = ROOT / "results/frontier_gemini.jsonl"
        rows = run_provider("gemini", args.gemini_model, items, out)
        summary[f"gemini:{args.gemini_model}"] = summarize("Gemini", rows)

    (ROOT / "results/frontier_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ wrote results/frontier_summary.json")


if __name__ == "__main__":
    main()
