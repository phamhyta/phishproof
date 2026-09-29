"""Import containerized Phishpedia outputs into results/phishpedia_d1.jsonl and
results/detector_d1.jsonl.

Reads external/pp_res*.txt (tab-separated Phishpedia run) plus external/pp_labels.json.
Skips Docker re-run when shard outputs already cover the test split.

Usage:
    .venv/bin/python scripts/import_phishpedia_d1.py
    .venv/bin/python scripts/import_phishpedia_d1.py --external external --out results
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.data_io import read_manifest
from phishproof.eval.evaluate import evaluate_detection


def parse_pp_line(line: str) -> dict | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        return None
    page_id = parts[0]
    try:
        pred_phish = int(parts[2])
    except ValueError:
        return None
    brand = parts[3] if len(parts) > 3 and parts[3] not in ("None", "") else None
    domains = None
    if len(parts) > 4 and parts[4] not in ("None", ""):
        try:
            domains = ast.literal_eval(parts[4])
        except (ValueError, SyntaxError):
            domains = parts[4]
    return {
        "page_id": page_id,
        "url": parts[1] if len(parts) > 1 else None,
        "verdict": "phish" if pred_phish == 1 else "benign",
        "brand": brand,
        "domains": domains,
    }


def load_shard_results(external: Path) -> dict[str, dict]:
    pred: dict[str, dict] = {}
    for path in sorted(external.glob("pp_res*.txt")):
        for line in path.read_text(encoding="ISO-8859-1").splitlines():
            row = parse_pp_line(line)
            if row:
                pred[row["page_id"]] = row
    return pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", type=Path, default=Path("external"))
    ap.add_argument("--data", type=Path, default=Path("data/phishsel_final"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    labels_path = args.external / "pp_labels.json"
    if not labels_path.exists():
        raise SystemExit(f"missing {labels_path}")

    labels = json.loads(labels_path.read_text())
    pred = load_shard_results(args.external)
    test_pages = read_manifest(args.data / "test.jsonl")

    missing = [p.page_id for p in test_pages if p.page_id not in pred]
    if missing:
        print(f"[WARN] {len(missing)} test pages missing Phishpedia output "
              f"(first: {missing[0]})")

    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / "phishpedia_d1.jsonl"
    det_path = args.out / "detector_d1.jsonl"

    rows = []
    yt, yp = [], []
    for page in test_pages:
        row = pred.get(page.page_id)
        if row is None:
            row = {
                "page_id": page.page_id,
                "url": page.url,
                "verdict": "benign",
                "brand": None,
                "domains": None,
            }
        row = {**row, "label": page.label.value}
        rows.append(row)
        yt.append(1 if page.label.value == "phish" else 0)
        yp.append(1 if row["verdict"] == "phish" else 0)

    with cache_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with det_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({
                "page_id": row["page_id"],
                "label": row["label"],
                "verdict": row["verdict"],
            }) + "\n")

    m = evaluate_detection(yt, yp, n_boot=1000)
    print(f"Imported {len(rows)} pages -> {cache_path.name}, {det_path.name}")
    print(f"D1 Phishpedia (n={len(rows)}):")
    for k, (pt, lo, hi) in m.items():
        print(f"  {k:10} {pt*100:5.1f}  [{lo*100:.1f},{hi*100:.1f}]")

    subsumes = {
        "n": len(rows),
        "detection": {k: {"point": pt, "lo": lo, "hi": hi} for k, (pt, lo, hi) in m.items()},
        "source": "external/pp_res*.txt",
    }
    (args.out / "d1_import.json").write_text(json.dumps(subsumes, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
