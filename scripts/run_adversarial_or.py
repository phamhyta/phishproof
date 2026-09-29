"""RQ7 adversarial robustness -- 70B panel variant.

Adapts run_adversarial.py for the 70B panel:
  - Formula trust s = conf_VLM + eps*agree_text (not GEA) for act/abstain
  - Calibrator trained on formula scores (calibrator_or.json)
  - Attacks: cloak, occlude, both, adaptive (same perturbations as 3B run)

Usage:
    .venv/bin/python scripts/run_adversarial_or.py --attack cloak
    .venv/bin/python scripts/run_adversarial_or.py --attack occlude
    .venv/bin/python scripts/run_adversarial_or.py --attack both
    .venv/bin/python scripts/run_adversarial_or.py --attack adaptive
    # combine all into table_values.json:
    .venv/bin/python scripts/run_adversarial_or.py --attack all
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phishproof.aggregate.gea import score_page
from phishproof.agents.client import ChatClient
from phishproof.agents.panel import Panel
from phishproof.calibration import IsotonicCalibrator
from phishproof.config import load_panel
from phishproof.data_io import read_manifest
from phishproof.eval.perturb import (
    cloak_form_action,
    occlude_logo,
    strip_brand_text,
)
from phishproof.schema import Label
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.logo_brand import CLIPLogoEmbedder
from phishproof.tools.registry import GroundingContext

ATTACKS = {
    "cloak":    [cloak_form_action],
    "occlude":  [occlude_logo],
    "both":     [cloak_form_action, occlude_logo],
    "adaptive": [cloak_form_action, strip_brand_text, occlude_logo],
}
EPS = 1e-3


def formula_trust(outs) -> float:
    """s = conf_VLM + eps * agree_text (70B formula trust)."""
    vlm = next((o for o in outs if o.agent_id == "agent_c_vision"), None)
    conf = vlm.confidence if (vlm and vlm.confidence is not None) else 0.5
    vlm_v = vlm.verdict if vlm else Panel.majority_label(outs)
    text = [o for o in outs if o.agent_id != "agent_c_vision"]
    agree = sum(1 for t in text if t.verdict == vlm_v) / len(text) if text else 0.0
    return float(conf) + EPS * agree


def run_attack(attack: str, args) -> dict:
    clean = {json.loads(l)["page_id"]: json.loads(l)
             for l in args.bundle.read_text().splitlines() if l.strip()}
    pages = {p.page_id: p for p in read_manifest(args.data / "test.jsonl")}
    calibrator = IsotonicCalibrator.from_dict(json.loads(args.calibrator.read_text()))
    tau = json.loads(args.operating_point.read_text())["tau"]

    # Correctly-detected phishing pages, spread across formula-trust range
    sel = [pid for pid, r in clean.items()
           if r.get("label") == "phish" and r.get("verdict") == "phish" and pid in pages]
    if not sel:
        raise RuntimeError("No eligible pages in bundle -- check bundle path / verdicts")

    # Sort by gea (or formula trust if available) for even sampling
    sel.sort(key=lambda pid: clean[pid].get("gea", 0.0))
    if len(sel) > args.limit:
        idx = np.linspace(0, len(sel) - 1, args.limit).astype(int)
        sel = [sel[i] for i in idx]
    print(f"[{attack}] {len(sel)} correctly-detected phish pages selected")

    fns = ATTACKS[attack]
    out_dir = Path("results/perturbed_or")
    out_dir.mkdir(parents=True, exist_ok=True)

    attacked_pages, pid_map, skipped = [], {}, 0
    for pid in sel:
        p = pages[pid]
        for fn in fns:
            p = fn(p, out_dir)
        if p.page_id == pid:   # nothing changed
            skipped += 1
            continue
        attacked_pages.append(p)
        pid_map[p.page_id] = pid
    if skipped:
        print(f"  ({skipped} pages had no targetable cue for this attack -> skipped)")

    panel = Panel.from_config(load_panel(args.panel), ChatClient())
    ctx = GroundingContext(detector=HtmlBrandDetector(), logo_embedder=CLIPLogoEmbedder())

    seen = {"a": None}

    def progress(agent_id, i, n):
        if agent_id != seen["a"]:
            seen["a"] = agent_id
            print(f"  sweeping {agent_id} over {n} attacked pages...", flush=True)
        if i % 25 == 0 and i:
            print(f"      {agent_id}: {i}/{n}", flush=True)

    outs_by_page = panel.run_batched(attacked_pages, progress=progress)

    rows = []
    for ap_page in attacked_pages:
        pid = pid_map[ap_page.page_id]
        c = clean[pid]
        # Compute clean formula trust from bundle_or agents field
        a_agents = c.get("agents", [])
        vlm_c = next((a for a in a_agents if a.get("id") == "agent_c_vision"), None)
        clean_conf = vlm_c["confidence"] if (vlm_c and vlm_c.get("confidence") is not None) else 0.5
        vlm_v = vlm_c["verdict"] if vlm_c else c["verdict"]
        text_c = [a for a in a_agents if a.get("id") != "agent_c_vision"]
        clean_agree = sum(1 for a in text_c if a.get("verdict") == vlm_v) / len(text_c) if text_c else 0.0
        clean_formula = clean_conf + EPS * clean_agree
        clean_trust = calibrator(clean_formula)

        # Attacked outputs
        outs = outs_by_page.get(ap_page.page_id, [])
        att_formula = formula_trust(outs)
        att_trust = calibrator(att_formula)
        r = score_page(outs, ap_page, ctx, add_consistency=True, relax_perceptual=True)

        att_verdict = r.verdict.value if hasattr(r.verdict, "value") else str(r.verdict)
        rows.append({
            "page_id": pid,
            "label": "phish",
            "attack": attack,
            "clean": {"gea": c.get("gea", 0.0), "G": c.get("groundedness", None),
                      "formula": clean_formula, "trust": clean_trust,
                      "verdict": c["verdict"]},
            "attacked": {"gea": r.gea, "G": r.groundedness,
                         "formula": att_formula, "trust": att_trust,
                         "verdict": att_verdict},
        })

    n = len(rows)
    evaded = [x for x in rows if x["attacked"]["verdict"] != Label.PHISH.value]
    acted_att = sum(1 for x in rows if x["attacked"]["trust"] >= tau)

    G_clean_vals = [x["clean"]["G"] for x in rows if x["clean"]["G"] is not None]
    G_att_vals = [x["attacked"]["G"] for x in rows if x["attacked"]["G"] is not None]

    summary = {
        "attack": attack, "n": n, "tau": tau, "skipped": skipped,
        "evasion_rate": len(evaded) / n if n else None,
        "evade_pct": round(100 * len(evaded) / n, 1) if n else None,
        "G_mean_clean": round(float(np.mean(G_clean_vals)), 3) if G_clean_vals else None,
        "G_mean_attacked": round(float(np.mean(G_att_vals)), 3) if G_att_vals else None,
        "abstain_rate_attacked": 1 - acted_att / n if n else None,
    }
    if evaded:
        pp_abstain = sum(1 for x in evaded if x["attacked"]["trust"] < tau) / len(evaded)
        summary["on_evaded"] = {
            "n": len(evaded),
            "abstain_rate": pp_abstain,
            "abstain_pct": round(100 * pp_abstain, 1),
        }
    return summary, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/phishsel_final"))
    ap.add_argument("--bundle", type=Path, default=Path("results/bundle_or.jsonl"))
    ap.add_argument("--calibrator", type=Path, default=Path("results/calibrator_or.json"))
    ap.add_argument("--operating-point", type=Path, default=Path("results/operating_point_or.json"))
    ap.add_argument("--panel", type=Path, default=Path("configs/panel_or.yaml"))
    ap.add_argument("--attack", choices=[*list(ATTACKS), "all"], default="both")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--out-dir", type=Path, default=Path("results/_or"))
    args = ap.parse_args()

    attacks = list(ATTACKS) if args.attack == "all" else [args.attack]
    results = {}
    all_rows = []
    for atk in attacks:
        print(f"\n=== {atk.upper()} ===")
        summ, rows = run_attack(atk, args)
        results[atk] = summ
        all_rows.extend(rows)
        print(json.dumps(summ, indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "bundle_adversarial_or.jsonl").write_text(
        "\n".join(json.dumps(r) for r in all_rows) + "\n")

    # Build table_values-compatible summary
    tv_adv = {}
    for atk, s in results.items():
        tv_adv[atk] = {
            "evade_pct": s.get("evade_pct"),
            "G_clean": s.get("G_mean_clean"),
            "G_attacked": s.get("G_mean_attacked"),
            "abstain_evaded_pct": s.get("on_evaded", {}).get("abstain_pct"),
        }

    tv_path = Path("results/_D/table_values.json")
    if tv_path.exists():
        tv = json.loads(tv_path.read_text())
        tv["adversarial_rq7"] = tv_adv
        tv_path.write_text(json.dumps(tv, indent=2))
        print(f"\nUpdated {tv_path} with 70B adversarial results")

    # Also write standalone summary
    (args.out_dir / "rq7_adversarial_or.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out_dir}/rq7_adversarial_or.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
