"""Build a cross-feed mini benchmark (Tier 1 #5) for distribution-shift validation.

Fetches fresh phishing URLs from PhishTank's open feed and pairs them with Tranco
login-page benigns, then renders each page (HTML + screenshot) and writes a
manifest compatible with build_results_bundle.py.

This script is the EXPENSIVE part the user runs:
  - PhishTank: needs http access (no key required for valid_online.json).
  - Rendering: needs Playwright (`pip install playwright && playwright install chromium`).
  - Panel re-run will cost ~$10 OpenAI (vision) + ~4h text-local CPU on 200 phish + 200 benign.

Outputs:
  data/crossfeed/raw/<page_id>/{html.txt, shot.png, info.txt}
  data/crossfeed/manifest.jsonl
  data/crossfeed/calibration.jsonl    (=  empty: reuse existing calibrator)
  data/crossfeed/test.jsonl           (200 phish + 200 benign)

Then run:
  python scripts/build_results_bundle.py --data data/crossfeed --out results/bundle_crossfeed.jsonl \\
         --calibrator results/calibrator.json --operating-point results/operating_point.json --no-logo
  python scripts/run_experiments.py    --bundle results/bundle_crossfeed.jsonl --out results/rq_crossfeed.json
"""
from __future__ import annotations
import argparse
import asyncio
import csv
import hashlib
import io
import json
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/crossfeed"

PHISHTANK_URL = "https://data.phishtank.com/data/online-valid.json"
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"

LOGIN_PATHS = ["/login", "/signin", "/account/login", "/auth/login", "/users/sign_in"]


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s)[:60].strip("-").lower()


def page_id(url: str) -> str:
    h = hashlib.sha1(url.encode()).hexdigest()[:8]
    host = re.findall(r"https?://([^/]+)", url + " ")
    host = host[0] if host else "unknown"
    return f"{slug(host)}-{h}"


def fetch_phishtank(n: int) -> list[dict]:
    print(f"[phishtank] fetching {PHISHTANK_URL}")
    with urllib.request.urlopen(PHISHTANK_URL, timeout=60) as r:
        data = json.load(r)
    print(f"[phishtank] {len(data)} valid_online entries")
    random.shuffle(data)
    out = []
    for e in data:
        url = e.get("url")
        if not url:
            continue
        out.append({"url": url, "target": e.get("target", "")})
        if len(out) >= n:
            break
    return out


def fetch_tranco_logins(n: int) -> list[dict]:
    print(f"[tranco] fetching {TRANCO_URL}")
    with urllib.request.urlopen(TRANCO_URL, timeout=120) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    with z.open(z.namelist()[0]) as f:
        reader = csv.reader(io.TextIOWrapper(f))
        domains = [row[1] for row in reader][:5000]
    random.shuffle(domains)
    out = []
    for d in domains:
        path = random.choice(LOGIN_PATHS)
        out.append({"url": f"https://{d}{path}", "target": ""})
        if len(out) >= n:
            break
    return out


async def render(url: str, out_dir: Path, timeout_ms: int = 20000) -> bool:
    """Render with Playwright. Returns True on success."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        sys.stderr.write("ERROR: playwright not installed. `pip install playwright && playwright install chromium`\n")
        raise
    out_dir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await ctx.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            await page.wait_for_timeout(1500)
            html = await page.content()
            (out_dir / "html.txt").write_text(html, errors="replace")
            await page.screenshot(path=str(out_dir / "shot.png"), full_page=False)
            (out_dir / "info.txt").write_text(f"url={url}\n")
            return True
        except Exception as e:
            sys.stderr.write(f"  render fail {url}: {e}\n")
            return False
        finally:
            await ctx.close()
            await browser.close()


async def render_all(entries: list[dict], label: str) -> list[dict]:
    out = []
    for i, e in enumerate(entries):
        pid = page_id(e["url"])
        raw = OUT_DIR / "raw" / pid
        if not (raw / "html.txt").exists():
            ok = await render(e["url"], raw)
            if not ok:
                continue
        out.append({
            "page_id": pid,
            "label": label,
            "url": e["url"],
            "brand": e.get("target", ""),
            "raw_dir": str(raw),
        })
        if (i + 1) % 20 == 0:
            print(f"  [{label}] rendered {i+1}/{len(entries)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-phish", type=int, default=200)
    ap.add_argument("--n-benign", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-render", action="store_true",
                    help="Only fetch URL lists; do not download/render pages (for inspection).")
    args = ap.parse_args()
    random.seed(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "raw").mkdir(exist_ok=True)

    phish = fetch_phishtank(args.n_phish * 2)  # over-fetch; many fail render
    benign = fetch_tranco_logins(args.n_benign * 2)
    (OUT_DIR / "urls_phish.json").write_text(json.dumps(phish, indent=2))
    (OUT_DIR / "urls_benign.json").write_text(json.dumps(benign, indent=2))
    print(f"wrote {len(phish)} phish + {len(benign)} benign URL candidates")

    if args.skip_render:
        return

    print("[render] phish ...")
    phish_rec = asyncio.run(render_all(phish[: args.n_phish * 2], "phish"))[: args.n_phish]
    print("[render] benign ...")
    benign_rec = asyncio.run(render_all(benign[: args.n_benign * 2], "benign"))[: args.n_benign]

    test = phish_rec + benign_rec
    random.shuffle(test)
    with open(OUT_DIR / "test.jsonl", "w") as f:
        for r in test:
            f.write(json.dumps(r) + "\n")
    with open(OUT_DIR / "calibration.jsonl", "w") as f:
        pass  # empty: reuse existing calibrator
    print(f"\nwrote {OUT_DIR/'test.jsonl'} ({len(test)} pages, {len(phish_rec)} phish + {len(benign_rec)} benign)")
    print("\nNext:")
    print(f"  python scripts/build_results_bundle.py --data {OUT_DIR} \\")
    print( "      --out results/bundle_crossfeed.jsonl \\")
    print( "      --calibrator results/calibrator.json \\")
    print( "      --operating-point results/operating_point.json --no-logo")
    print(f"  python scripts/run_experiments.py --bundle results/bundle_crossfeed.jsonl \\")
    print( "      --out results/rq_crossfeed.json")


if __name__ == "__main__":
    main()
