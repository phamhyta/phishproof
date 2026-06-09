"""Load a results bundle (one JSON line per page) and evaluate every method into tables.

Bundle line schema (written by the Phase 5 runner from real panel + baseline scores):
    {"page_id","label","verdict","gea","calibrated_trust",
     "baselines": {"B1":..,"B2":..,"B4":..,"B5":..,"B6":..,"D3":..}}

PhishProof's selective score is the calibrated trust (falls back to raw GEA); each
baseline supplies its own trust score. The same page draws (boot_idx) are shared across
methods so the CIs are paired.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .evaluate import evaluate_detection, evaluate_method
from .bootstrap import paired_bootstrap_indices


def load_bundle(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _label01(v: str) -> int:
    return 1 if v == "phish" else 0


def evaluate_bundle(
    bundle: list[dict],
    proposed_name: str = "PhishProof",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Return {'methods': {name: metrics}, 'detection': metrics, 'order': [...]}.

    methods includes PhishProof + every baseline key present in bundle[*]['baselines'].
    """
    y_true = np.array([_label01(r["label"]) for r in bundle])
    y_pred = np.array([_label01(r["verdict"]) for r in bundle])
    boot = paired_bootstrap_indices(len(bundle), n_boot, seed)

    methods: dict[str, dict] = {}

    # Proposed method: calibrated trust if present else raw GEA.
    gea = np.array([r["gea"] for r in bundle])
    trust = np.array([r.get("calibrated_trust") if r.get("calibrated_trust") is not None
                      else r["gea"] for r in bundle])
    methods[proposed_name] = evaluate_method(gea, y_true, y_pred, trust=trust, boot_idx=boot)

    # Baselines.
    baseline_keys: list[str] = []
    for r in bundle:
        for k in r.get("baselines", {}):
            if k not in baseline_keys:
                baseline_keys.append(k)
    for k in baseline_keys:
        s = np.array([r.get("baselines", {}).get(k, 0.0) for r in bundle])
        methods[k] = evaluate_method(s, y_true, y_pred, trust=s, boot_idx=boot)

    detection = evaluate_detection(y_true, y_pred, boot_idx=boot)
    order = [proposed_name] + baseline_keys
    return {"methods": methods, "detection": detection, "order": order}


def best_baseline(methods: dict, baseline_names: list[str], key: str = "AURC",
                  lower_better: bool = True) -> str | None:
    """Name of the best baseline on a metric (underlined in the paper's tables)."""
    cand = [(methods[b][key][0], b) for b in baseline_names if b in methods]
    if not cand:
        return None
    return (min if lower_better else max)(cand)[1]
