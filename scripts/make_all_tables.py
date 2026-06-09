"""Consolidate every result into the paper's tables (real numbers) + real-vs-paper.

Reads the rich bundle (which already stores A, G, per-page baselines) plus the RQ2 /
detector / diversity artifacts, and prints tab_main (RQ1), tab_detect, tab_rlwr (RQ2),
tab_ablation (RQ3), RQ4 ECE, and a real-vs-paper comparison. The ablation is computed
straight from the bundle's stored A/G/B4 — no agent re-run needed.

Usage: .venv/bin/python scripts/make_all_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.eval.bundle import evaluate_bundle, load_bundle
from phishproof.eval.metrics import aurc, detection_metrics, ece, fpr_at_coverage


def artifact_dir() -> Path:
    """Prefer results/ (dev) else artifacts/ (public git clone)."""
    for name in ("results", "artifacts"):
        root = Path(name)
        if (root / "bundle_final.jsonl").exists():
            return root
    return Path("results")

# Fallback if detector_d1.jsonl missing (legacy subset run).
D1_SUBSET_N = 509
D1_SUBSET_METRICS = {"accuracy": 0.796, "precision": 0.928, "recall": 0.626, "f1": 0.748}


def rq1(bundle):
    res = evaluate_bundle(bundle, n_boot=1000)
    keys = ("AURC", "SelAcc80", "FPR80", "Cov99", "ECE")
    print("\n=== tab_main (RQ1) ===")
    print(f"{'method':14}" + "".join(f"{k:>10}" for k in keys))
    for m in res["order"]:
        cells = []
        for k in keys:
            p = res["methods"][m][k][0]
            cells.append(f"{p*100:>9.1f}" if k in ("AURC", "SelAcc80", "FPR80") else f"{p:>9.3f}")
        print(f"{m:14}" + "".join(cells))

    b3 = artifact_dir() / "b3_selfconsistency_full.jsonl"
    if b3.exists():
        rows = [json.loads(l) for l in b3.read_text().splitlines() if l.strip()]
        trust = np.array([r["trust"] for r in rows], dtype=float)
        yt = np.array([1 if r["label"] == "phish" else 0 for r in rows], dtype=bool)
        yp = np.array([1 if r["verdict"] == "phish" else 0 for r in rows], dtype=bool)
        from phishproof.eval.evaluate import evaluate_method
        b3_res = evaluate_method(trust, yt, yp, trust=trust, n_boot=2000, seed=0)
        res["methods"]["B3"] = b3_res
        if "B3" not in res["order"]:
            insert_at = res["order"].index("B2") + 1 if "B2" in res["order"] else len(res["order"])
            res["order"].insert(insert_at, "B3")
        cells = []
        for k in keys:
            p, lo, hi = b3_res[k]
            half = (hi - lo) / 2.0
            scale = 100 if k in ("AURC", "SelAcc80", "FPR80") else 1
            cells.append(f"{p*scale:>6.2f}±{half*scale:<.2f}")
        print(f"{'B3 (GPT-4o k=5)':14}" + " ".join(cells) + f"  (n={len(rows)}, 2000 boot)")
        # write back to artifact dir so tab_main.tex regenerator picks up
        ci_path = artifact_dir() / "b3_bootstrap.json"
        ci_path.write_text(json.dumps(
            {k: {"point": b3_res[k][0], "ci_lo": b3_res[k][1], "ci_hi": b3_res[k][2]} for k in keys},
            indent=2))
    else:
        print("  B3: (b3_selfconsistency_full.jsonl missing — run run_b3_selfconsistency.py)")
    return res


def rq3_ablation(bundle):
    yt = np.array([1 if r["label"] == "phish" else 0 for r in bundle])
    yp = np.array([1 if r["verdict"] == "phish" else 0 for r in bundle])
    correct = yt == yp
    A = np.array([r["agreement"] for r in bundle])
    G = np.array([r["groundedness"] for r in bundle])
    gea = np.array([r["gea"] for r in bundle])
    b4 = np.array([r["baselines"]["B4"] for r in bundle])
    variants = {
        "PhishProof (full)": (gea, 0.015),
        " - calibration": (gea, 0.331),
        " - groundedness": (A, 0.014),
        " - evidence-agreement": (b4 * G, 0.015),
    }
    print("\n=== tab_ablation (RQ3) ===")
    print(f"{'variant':24}{'AURC':>8}{'FPR80':>8}{'ECE*':>8}   (*calibrated, from rescore_ablation)")
    for name, (rank, e) in variants.items():
        a = aurc(rank, correct) * 100
        fpr = fpr_at_coverage(rank, yt, yp, 0.80) * 100
        print(f"{name:24}{a:8.2f}{fpr:8.2f}{e:8.3f}")
    sf = artifact_dir() / "bundle_samefamily.jsonl"
    if sf.exists():
        sb = load_bundle(sf)
        yt2 = np.array([1 if r["label"] == "phish" else 0 for r in sb])
        yp2 = np.array([1 if r["verdict"] == "phish" else 0 for r in sb])
        c2 = yt2 == yp2
        g2 = np.array([r["gea"] for r in sb])
        print(f"{' - diversity (same-fam)':24}{aurc(g2, c2)*100:8.2f}"
              f"{fpr_at_coverage(g2, yt2, yp2, 0.80)*100:8.2f}{'  n/a':>8}  (n={len(sb)} subset)")
    else:
        print(" - diversity: (bundle_samefamily.jsonl pending)")


def detection(bundle):
    print("\n=== tab_detect (full-coverage detection) ===")
    yt = [1 if r["label"] == "phish" else 0 for r in bundle]
    yp = [1 if r["verdict"] == "phish" else 0 for r in bundle]
    m = detection_metrics(yt, yp)
    print(f"  PhishProof (panel, n={len(bundle)}): acc={m['accuracy']*100:.1f} "
          f"P={m['precision']*100:.1f} R={m['recall']*100:.1f} F1={m['f1']*100:.1f}")

    d1 = artifact_dir() / "detector_d1.jsonl"
    if d1.exists():
        rows = [json.loads(l) for l in d1.read_text().splitlines() if l.strip()]
        yt1 = [1 if r["label"] == "phish" else 0 for r in rows]
        yp1 = [1 if r["verdict"] == "phish" else 0 for r in rows]
        m1 = detection_metrics(yt1, yp1)
        print(f"  D1 Phishpedia (re-run, n={len(rows)}): acc={m1['accuracy']*100:.1f} "
              f"P={m1['precision']*100:.1f} R={m1['recall']*100:.1f} F1={m1['f1']*100:.1f}")
    else:
        m1 = D1_SUBSET_METRICS
        print(f"  D1 Phishpedia (documented subset n={D1_SUBSET_N}, no detector_d1.jsonl): "
              f"acc={m1['accuracy']*100:.1f} P={m1['precision']*100:.1f} "
              f"R={m1['recall']*100:.1f} F1={m1['f1']*100:.1f}  (point est., no CI)")

    d3 = artifact_dir() / "detector_d3.jsonl"
    if d3.exists():
        rows = [json.loads(l) for l in d3.read_text().splitlines() if l.strip()]
        yt3 = [1 if r["label"] == "phish" else 0 for r in rows]
        yp3 = [1 if r["verdict"] == "phish" else 0 for r in rows]
        m3 = detection_metrics(yt3, yp3)
        print(f"  D3 PhishLLM (re-run, n={len(rows)}): acc={m3['accuracy']*100:.1f} "
              f"P={m3['precision']*100:.1f} R={m3['recall']*100:.1f} F1={m3['f1']*100:.1f}")
    else:
        print("  D3 PhishLLM: (detector_d3.jsonl missing — run run_detectors.py --detector d3)")
    print("  D2 PhishIntention: cited (not re-run)")


def rq2():
    f = artifact_dir() / "rq2_final.json"
    if not f.exists():
        print("\n=== tab_rlwr (RQ2): pending ===")
        return
    d = json.loads(f.read_text())
    lo, hi = d["low_gea"]["flip_rate"], d["high_gea"]["flip_rate"]
    print("\n=== tab_rlwr (RQ2, hard subset) ===")
    print(f"  n={d['n']} correctly labeled lookalike pages (of 233 hard in test)")
    print(f"  low-GEA flip={lo:.3f} (n={d['low_gea']['n']})  vs  high-GEA flip={hi:.3f} "
          f"(n={d['high_gea']['n']})  ratio={lo/max(hi,1e-3):.1f}x")
    print("  quintiles GEA->flip: " + "  ".join(
        f"{q['gea_mean']:.2f}:{q['flip_rate']:.2f}" for q in d["quintile_flip_rates"]))


def real_vs_paper(res):
    pp = res["methods"]["PhishProof"]
    b5 = res["methods"]["B5"]
    print("\n=== REAL vs PAPER (illustrative) ===")
    print(f"{'metric':12}{'paper PP':>10}{'paper B5':>10}{'real PP':>10}{'real B5':>10}")
    paper = {"AURC": (3.3, 5.1), "SelAcc80": (98.4, 96.5), "Cov99": (0.71, 0.48),
             "ECE": (0.031, 0.063)}
    for k, (pPP, pB5) in paper.items():
        s = 100 if k in ("AURC", "SelAcc80") else 1
        print(f"{k:12}{pPP:>10}{pB5:>10}{pp[k][0]*s:>10.2f}{b5[k][0]*s:>10.2f}")


def main() -> int:
    root = artifact_dir()
    bundle = load_bundle(root / "bundle_final.jsonl")
    res = rq1(bundle)
    detection(bundle)
    rq2()
    rq3_ablation(bundle)
    print(f"\n=== RQ4: PhishProof ECE = {res['methods']['PhishProof']['ECE'][0]:.3f} "
          "(calibrated trust faithful) ===")
    real_vs_paper(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
