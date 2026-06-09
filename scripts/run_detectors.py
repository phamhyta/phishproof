"""Run a standalone specialized detector over the test split -> tab_detect numbers.

Saves per-page verdicts to results/detector_<name>.jsonl and prints full-coverage
detection metrics (accuracy / precision / recall / F1) with bootstrap CIs.

Usage:
    .venv/bin/python scripts/run_detectors.py --detector d3 --data data/phishsel_final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.data_io import read_manifest
from phishproof.eval.evaluate import evaluate_detection
from phishproof.tools.detectors.phishllm import PhishLLMDetector
from phishproof.tools.detectors.phishpedia import PhishpediaDetector


def get_detector(name: str):
    if name == "d3":
        return PhishLLMDetector()
    if name == "d1":
        cache = Path("results/phishpedia_d1.jsonl")
        if not cache.exists():
            raise SystemExit(
                "missing results/phishpedia_d1.jsonl — run scripts/import_phishpedia_d1.py"
            )
        return PhishpediaDetector(cache)
    raise SystemExit(f"unknown detector {name} (d1/d3 supported)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/phishsel_final"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    pages = read_manifest(args.data / "test.jsonl")
    if args.limit:
        pages = pages[: args.limit]
    det = get_detector(args.detector)

    out = Path(f"results/detector_{args.detector}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    yt, yp = [], []
    errors = 0
    with out.open("w", encoding="utf-8") as f:
        for i, p in enumerate(pages, 1):
            try:
                v = det.predict(p).value
            except Exception:  # noqa: BLE001 - a transient API error must not abort the run
                errors += 1
                v = "benign"
            yt.append(1 if p.label.value == "phish" else 0)
            yp.append(1 if v == "phish" else 0)
            f.write(json.dumps({"page_id": p.page_id, "label": p.label.value,
                                "verdict": v}) + "\n")
            f.flush()  # survive a crash + make progress visible on disk
            if i % 50 == 0:
                print(f"  {i}/{len(pages)} ({errors} errors)", flush=True)

    m = evaluate_detection(yt, yp, n_boot=1000)
    print(f"\n{det.name} detection (n={len(pages)}):")
    for k, (pt, lo, hi) in m.items():
        print(f"  {k:10} {pt*100:5.1f}  [{lo*100:.1f},{hi*100:.1f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
