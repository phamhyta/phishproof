"""Post-process the 70B adversarial bundle into faithful RQ7 numbers (no LLM calls).

The 70B method's selective signal is the formula trust s = conf_VLM + eps*agree (Option A).
Empirically s ALSO carries the adversarial fail-safe: on an evidence-targeted attack the vision
agent loses confidence (its DOM/perceptual context is corrupted), so the calibrator maps s into
its low bucket and s < tau -> the detector abstains. This holds across attacks (cloak 95%,
occlude 100%, both 95%), so we report abstain-on-evaded via the trust rule (s < tau_formula).

For transparency we also report two secondary signals: the retained per-type GEA rule
(gea < tau_gea, the 3B mechanism, kept for the consensus set + grounding veto per method.tex)
and the strict grounding veto (G < 1). Both corroborate the trust rule.

Reads the per-page bundle (clean/attacked: gea, G, trust, verdict) + results/_or/clean_g.json
(clean consensus G, from compute_clean_g_or.py) and reports, per attack:
  - evade_pct, evaded_n, G_clean -> G_attacked
  - abstain_evaded_pct via the deployed rule: s < tau_formula
  - secondary: abstain via retained GEA rule and strict grounding veto

Usage:
    .venv/bin/python scripts/postprocess_adversarial_or.py
    .venv/bin/python scripts/postprocess_adversarial_or.py --write
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def summarize(rows: list[dict], tau_gea: float, tau_formula: float, g_min: float,
              clean_g: dict | None = None) -> dict:
    clean_g = clean_g or {}
    by_attack: dict[str, list[dict]] = {}
    for r in rows:
        by_attack.setdefault(r["attack"], []).append(r)

    out = {}
    for attack, rs in sorted(by_attack.items()):
        n = len(rs)
        evaded = [r for r in rs if r["attacked"]["verdict"] != "phish"]
        ne = len(evaded)

        def gval(r, side):
            if side == "clean" and r["page_id"] in clean_g:
                return clean_g[r["page_id"]]
            v = r[side].get("G")
            return v if v is not None else 1.0

        def frac(pred):
            return round(100 * sum(1 for r in evaded if pred(r)) / ne, 1) if ne else None

        out[attack] = {
            "n": n,
            "evaded_n": ne,
            "evade_pct": round(100 * ne / n, 1) if n else None,
            "G_clean": round(float(np.mean([gval(r, "clean") for r in rs])), 2),
            "G_attacked": round(float(np.mean([gval(r, "attacked") for r in rs])), 2),
            # DEPLOYED rule (Option A): formula trust s = conf_VLM + eps*agree < tau.
            # Empirically s drops on evaded pages (vision loses confidence under cue
            # corruption -> calibrator maps to the low bucket), so it abstains 95-100%.
            "abstain_evaded_pct": frac(lambda r: r["attacked"]["trust"] < tau_formula),
            # secondary signals for prose / honesty
            "abstain_gea_rule_pct": frac(lambda r: r["attacked"]["gea"] < tau_gea),
            # BUGFIX (revision T3): this must be a STRICT partition bucket -- the veto only
            # "saves" a page the SCORE GATE ACCEPTED. Testing G < g_min alone counted
            # score-gate rejections as veto saves and reported 100% for both columns at once.
            # Authoritative version with Wilson intervals: scripts/revision_adversarial_or.py
            "abstain_veto_only_pct": frac(lambda r: r["attacked"]["trust"] >= tau_formula
                                          and gval(r, "attacked") < g_min),
            "wrong_act_pct": frac(lambda r: r["attacked"]["trust"] >= tau_formula
                                  and gval(r, "attacked") >= g_min),
            "gea_clean_mean": round(float(np.mean([r["clean"]["gea"] for r in rs])), 3),
            "gea_attacked_mean": round(float(np.mean([r["attacked"]["gea"] for r in rs])), 3),
            "gea_evaded_mean": round(float(np.mean([r["attacked"]["gea"] for r in evaded])), 3) if ne else None,
            "trust_evaded_mean": round(float(np.mean([r["attacked"]["trust"] for r in evaded])), 3) if ne else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=Path("results/_or/bundle_adversarial_or.jsonl"))
    ap.add_argument("--tau-gea", type=float, default=0.9167,
                    help="retained per-type GEA operating point (fit on 70B clean, risk<=0.01)")
    ap.add_argument("--tau-formula", type=float, default=0.9651,
                    help="formula-trust operating point, applied to the CALIBRATED trust "
                         "kappa(s) (verified: clean formula 0.9005 -> trust 0.9444 under "
                         "results/calibrator_or.json). The manuscript's 0.965 rounds this.")
    ap.add_argument("--g-min", type=float, default=1.0, help="strict grounding veto (Prop-2)")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    if not args.bundle.exists():
        print(f"[wait] {args.bundle} not present yet -- run still in progress")
        return 1

    rows = load_rows(args.bundle)
    clean_g_path = Path("results/_or/clean_g.json")
    clean_g = json.loads(clean_g_path.read_text()) if clean_g_path.exists() else {}
    print(f"Loaded {len(rows)} per-page rows; tau_gea={args.tau_gea}, "
          f"tau_formula={args.tau_formula}, g_min={args.g_min}; "
          f"clean_g={'yes' if clean_g else 'MISSING (run compute_clean_g_or.py)'}")
    summary = summarize(rows, args.tau_gea, args.tau_formula, args.g_min, clean_g)
    print(json.dumps(summary, indent=2))

    if args.write:
        tv_path = Path("results/_D/table_values.json")
        tv = json.loads(tv_path.read_text())
        tv["adversarial_rq7"] = {
            atk: {"evade_pct": s["evade_pct"], "G_clean": s["G_clean"],
                  "G_attacked": s["G_attacked"], "abstain_evaded_pct": s["abstain_evaded_pct"]}
            for atk, s in summary.items()
        }
        tv_path.write_text(json.dumps(tv, indent=2))
        (Path("results/_or") / "rq7_summary_full.json").write_text(json.dumps(summary, indent=2))
        print(f"\nWrote adversarial_rq7 -> {tv_path}")
        print("Wrote results/_or/rq7_summary_full.json (all signals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
