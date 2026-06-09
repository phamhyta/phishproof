"""Naturally-evasive phish subset analysis (RQ2 #9, free from cache).

For each phishing test page, mark it as 'evasive' if it satisfies any of:
  (a) no detectable logo bounding box (yolo_coords.txt empty / missing),
  (b) very small DOM (html.txt < 5 KB),
  (c) lookalike host (brand token embedded in registrable domain).

Compare flip rates and GEA distributions on evasive vs non-evasive correctly-labelled
phish from results/bundle_final.jsonl + results/bundle_adv_*.jsonl.

Writes results/rq2_evasive.json + prints a small table.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data"


def load_bundle(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["page_id"]] = r
    return out


def load_manifest() -> dict[str, str]:
    m = {}
    for split in ("test.jsonl", "calibration.jsonl"):
        f = ROOT / "data/phishsel_final" / split
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            m[r["page_id"]] = r
    return m


def classify_evasive(rec: dict) -> tuple[bool, list[str]]:
    raw = rec.get("raw_dir")
    if not raw:
        return False, []
    p = Path(raw)
    reasons = []

    yc = p / "yolo_coords.txt"
    if not yc.exists() or yc.stat().st_size == 0 or yc.read_text().strip() == "":
        reasons.append("no_logo_bbox")

    html = p / "html.txt"
    if html.exists() and html.stat().st_size < 5 * 1024:
        reasons.append("small_dom")

    info = p / "info.txt"
    brand = rec.get("brand", "")
    url = ""
    if info.exists():
        for line in info.read_text(errors="replace").splitlines():
            if line.lower().startswith("url"):
                url = line.split("=", 1)[-1].split(":", 1)[-1].strip()
                break
    if brand and url:
        b = re.sub(r"[^a-z0-9]", "", brand.lower())[:6]
        host = re.findall(r"https?://([^/]+)", url + " ")
        if host and b and len(b) >= 4 and b in re.sub(r"[^a-z0-9]", "", host[0].lower()):
            # only count lookalike if the host's registrable domain is NOT the canonical brand domain
            if b not in host[0].lower().split(".")[-2:]:
                reasons.append("lookalike_host")

    return bool(reasons), reasons


def main() -> int:
    bundle = load_bundle(ROOT / "results/bundle_final.jsonl")
    if not bundle:
        print("missing bundle_final.jsonl")
        return 1
    manifest = load_manifest()

    # Adversarial bundles (cloak/occlude) tell us flip-on-perturbation.
    adv = {
        "cloak": load_bundle(ROOT / "results/bundle_adv_cloak.jsonl"),
        "occlude": load_bundle(ROOT / "results/bundle_adv_occlude.jsonl"),
        "both": load_bundle(ROOT / "results/bundle_adv_both.jsonl"),
    }

    rows = []
    for pid, r in bundle.items():
        if r["label"] != "phish":
            continue
        if r["verdict"] != r["label"]:  # only correctly-labelled
            continue
        m = manifest.get(pid, {})
        is_ev, reasons = classify_evasive(m)
        flips = {}
        for k, b in adv.items():
            br = b.get(pid)
            if br is not None and "attacked" in br:
                flips[k] = int(br["attacked"]["verdict"] != r["label"])
        rows.append({
            "page_id": pid,
            "evasive": is_ev,
            "reasons": reasons,
            "gea": r["gea"],
            "agreement": r["agreement"],
            "groundedness": r["groundedness"],
            "flips": flips,
        })

    n_total = len(rows)
    n_ev = sum(1 for r in rows if r["evasive"])
    print(f"n_total_correct_phish = {n_total}")
    print(f"n_evasive             = {n_ev} ({n_ev/max(n_total,1):.1%})")
    by_reason = {}
    for r in rows:
        for x in r["reasons"]:
            by_reason[x] = by_reason.get(x, 0) + 1
    for k, v in sorted(by_reason.items()):
        print(f"  reason {k:18s} {v}")

    def stats(subset, key):
        vals = [r[key] for r in subset]
        return (float(np.mean(vals)) if vals else 0.0, len(vals))

    ev = [r for r in rows if r["evasive"]]
    ne = [r for r in rows if not r["evasive"]]

    summary = {"n_total": n_total, "n_evasive": n_ev, "by_reason": by_reason}
    print("\n=== GEA / A / G distribution ===")
    print(f"{'group':<14}{'n':>6}{'mean_A':>9}{'mean_G':>9}{'mean_GEA':>10}")
    for name, sub in (("evasive", ev), ("non_evasive", ne)):
        mA, _ = stats(sub, "agreement")
        mG, _ = stats(sub, "groundedness")
        mGEA, _ = stats(sub, "gea")
        print(f"{name:<14}{len(sub):>6}{mA:>9.3f}{mG:>9.3f}{mGEA:>10.3f}")
        summary[name] = {"n": len(sub), "A": mA, "G": mG, "GEA": mGEA}

    print("\n=== flip rate under perturbation (correctly-labelled phish only) ===")
    print(f"{'attack':<10}{'evasive_flip':>16}{'n_ev':>7}{'nonev_flip':>14}{'n_ne':>7}{'ratio':>8}")
    summary["flip"] = {}
    for attack in ("cloak", "occlude", "both"):
        ev_flips = [r["flips"][attack] for r in ev if attack in r["flips"]]
        ne_flips = [r["flips"][attack] for r in ne if attack in r["flips"]]
        if not ev_flips or not ne_flips:
            continue
        ev_rate = float(np.mean(ev_flips))
        ne_rate = float(np.mean(ne_flips))
        ratio = ev_rate / max(ne_rate, 1e-9)
        print(f"{attack:<10}{ev_rate:>16.3f}{len(ev_flips):>7}{ne_rate:>14.3f}{len(ne_flips):>7}{ratio:>8.2f}")
        summary["flip"][attack] = {
            "evasive": {"rate": ev_rate, "n": len(ev_flips)},
            "non_evasive": {"rate": ne_rate, "n": len(ne_flips)},
            "ratio": ratio,
        }

    out = ROOT / "results/rq2_evasive.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n→ wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
