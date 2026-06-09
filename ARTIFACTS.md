# Artifact tiers

This git repository ships **Tier A + manifests** so every table number can be verified
without re-running models.

## Included in git (`artifacts/`)

| File | Used for |
|---|---|
| `bundle_final.jsonl` | RQ1 — PhishProof + baselines |
| `calibrator.json`, `operating_point.json` | Calibration / threshold |
| `detector_d1.jsonl`, `detector_d3.jsonl` | `tab_detect` |
| `phishpedia_d1.jsonl`, `subsumes_d1.json` | D1 tool subsumption |
| `b3_selfconsistency_full.jsonl` | B3 baseline |
| `bundle_samefamily.jsonl` | Diversity ablation |
| `rq1_main.json`, `rq2_final.json` | Aggregated metrics |
| `leave_brands_out.json` | Brand generalization |
| `rq7_*.json` | Adversarial robustness |
| `verifier_soundness.json`, `failure_modes.json` | Tool soundness / errors |
| `hfgea.json`, `cascade.json`, `esp.json`, `panel_strength.json` | Design ablations |

Regenerate tables:

```bash
# copy or symlink artifacts into results/ for scripts that read results/
mkdir -p results
cp artifacts/* results/
python scripts/make_all_tables.py
```

## Zenodo (planned) — Tier B/C

| Tier | Size | Contents |
|---|---|---|
| B | ~100 MB | `data/cache/` (panel agent cache) + D1 raw `pp_res*.txt` |
| C | ~800 MB | Tier B + 1,330 page snapshots (HTML + PNG) for full PhishSel |

## Checksums

`CHECKSUMS.sha256` covers `data/phishsel_final/calibration.jsonl` and `test.jsonl`.
Paper reports:

- `calibration.jsonl` → `9f2c9bb812af06a5a6757010e0899f6b1515204ec798526032e2384d0b69d40e`
- `test.jsonl` → `391fc594fa9aa26474e7f4ded57f94c1a8acaba8b0c41b2b158ca35b3bed1173`
