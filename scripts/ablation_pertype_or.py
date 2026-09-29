"""70B per-type-vs-global GEA ablation + cue-level agreement (cache-only, no new LLM calls).

Reviewers asked whether the per-type-agreement mechanism (defended only on the 3B panel, where
text agents share a cue on 1/4020 pages) still holds on the deployed 70B panel. This re-runs the
70B panel from cache, captures per-agent cues, and reports:
  - cue-level agreement: fraction of pages where the two text agents share >=1 cue, and the
    mean per-type GEA (so we can state the mechanism is active, not degenerate, at 70B);
  - the per-type-vs-global ranking ablation at 70B: AURC ranking by per-type GEA vs the global
    consensus-fraction, each calibrated on the 710-page calibration split (same protocol as 3B).

Usage:
    .venv/bin/python scripts/ablation_pertype_or.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phishproof.agents.client import ChatClient
from phishproof.agents.panel import Panel
from phishproof.aggregate.consensus import agreement, per_type_agreement
from phishproof.calibration import IsotonicCalibrator
from phishproof.config import load_panel
from phishproof.data_io import read_manifest

PANEL = Path("configs/panel_or.yaml")
DATA = Path("data/phishsel_final")


def vv(v):
    return v.value if hasattr(v, "value") else v


def score_split(pages, panel):
    """Return per-page (per_type, global_frac, correct, share_flag)."""
    outs_by = panel.run_batched(pages, progress=_progress)
    rows = []
    for p in pages:
        outs = outs_by[p.page_id]
        pta = per_type_agreement(outs)[0]
        glob = agreement(outs)[0]
        verdict = vv(Panel.majority_label(outs))
        correct = 1 if verdict == vv(p.label) else 0
        txt = [o for o in outs if o.agent_id != "agent_c_vision"]
        share = 0
        if len(txt) == 2:
            v1 = {(c.type, c.value) for c in txt[0].cues}
            v2 = {(c.type, c.value) for c in txt[1].cues}
            share = 1 if (v1 & v2) else 0
        rows.append((pta, glob, correct, share))
    return rows


_seen = {"a": None}
def _progress(agent_id, i, n):
    if agent_id != _seen["a"]:
        _seen["a"] = agent_id
        print(f"  {agent_id}: 0/{n}", flush=True)
    if i % 500 == 0 and i:
        print(f"  {agent_id}: {i}/{n}", flush=True)


def aurc(score, correct):
    score = np.asarray(score, float); correct = np.asarray(correct, float)
    o = np.argsort(-score, kind="stable"); c = correct[o]
    k = np.arange(1, len(c) + 1)
    return 100 * float(np.mean(1 - np.cumsum(c) / k))


def cov99(score, correct):
    score = np.asarray(score, float); correct = np.asarray(correct, float)
    o = np.argsort(-score, kind="stable"); c = correct[o]
    for k in range(1, len(c) + 1):
        if c[:k].mean() < 0.99:
            return 100 * (k - 1) / len(c)
    return 100.0


def main() -> int:
    panel = Panel.from_config(load_panel(PANEL), ChatClient())
    print("[calib] scoring 710-page calibration split (cache)...")
    calib = score_split(read_manifest(DATA / "calibration.jsonl"), panel)
    print("[test] scoring test split (cache)...")
    test = score_split(read_manifest(DATA / "test.jsonl"), panel)

    cpta, cglob, ccorr, _ = map(np.array, zip(*calib))
    tpta, tglob, tcorr, tshare = map(np.array, zip(*test))

    print(f"\n=== 70B cue-level agreement (test, n={len(test)}) ===")
    print(f"pages where 2 text agents share >=1 cue: {tshare.sum()}/{len(tshare)} "
          f"({100*tshare.mean():.1f}%)  [3B panel: 1/4020 = 0.02%]")
    print(f"mean per-type GEA: {tpta.mean():.3f}   mean global-fraction: {tglob.mean():.3f}")

    # Calibrate each score on calib split, evaluate AURC/Cov99 on test
    print(f"\n=== per-type vs global ranking ablation at 70B (calibrated) ===")
    for name, cs, ts in [("per-type GEA", cpta, tpta), ("global-fraction", cglob, tglob)]:
        cal = IsotonicCalibrator().fit(cs, ccorr)
        tcal = np.array([cal(s) for s in ts])
        print(f"  {name:<16} AURC={aurc(tcal, tcorr):.2f}  Cov99={cov99(tcal, tcorr):.1f}  "
              f"(raw AURC={aurc(ts, tcorr):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
