# PhishProof v1.0-paper

Code and artifacts accompanying the paper *"Trustworthy and Explainable Agents for
Phishing Detection with Grounded Cross-Model Evidence Agreement"*.

This release reproduces every number, table, and figure in the paper from cached
agent outputs — no model or API calls required.

## What's in this release

### Code (`./`)
- `phishproof/` — the framework (panel, aggregation, calibration, tools, evaluation).
- `scripts/` — runners (panel, baselines, adversarial, calibration), evaluation,
  and `make_all_tables.py` to regenerate every table.
- `phishing-detection-latex/` — the LaTeX source.
- `external/` — Docker recipes for D1 Phishpedia (and D2 PhishIntention, optional).

### Tier A artifacts (`artifacts/`, in git, ~10 MB)
Every aggregated result the paper reports, in jsonl/json. See
[`public-release/ARTIFACTS.md`](ARTIFACTS.md). Sufficient to regenerate every
table without re-running models:

```bash
python scripts/make_all_tables.py
```

### Tier B / C artifacts (Zenodo)
- Tier B (~100 MB): agent cache + D1 raw outputs — restores re-scoring without
  GPU/API. **DOI:** `TIER_B_DOI` (paste after upload).
- Tier C (~800 MB): page snapshots (HTML + PNG) for the 1,330 PhishSel pages —
  enables full from-scratch reproduction. **DOI:** `TIER_C_DOI`.

Build the bundles locally with `./scripts/build_zenodo_bundle.sh`.

## Reproducibility smoke test

```bash
./scripts/test_reproducibility.sh
```

Runs offline checks for schema/panel/tools, regenerates every table from
artifacts, and verifies dataset SHA-256 checksums. CI runs this on every push
(see `.github/workflows/reproducibility.yml`).

## Checksums

- `data/phishsel_final/calibration.jsonl` — `9f2c9bb812af06a5a6757010e0899f6b1515204ec798526032e2384d0b69d40e`
- `data/phishsel_final/test.jsonl` — `391fc594fa9aa26474e7f4ded57f94c1a8acaba8b0c41b2b158ca35b3bed1173`

## License

Code: MIT. Cached agent outputs: CC-BY-4.0. Page snapshots: redistributed under
fair use for research, original sources retain rights.

## Citation

```
@article{phishproof2026,
  title={Trustworthy and Explainable Agents for Phishing Detection with
         Grounded Cross-Model Evidence Agreement},
  author={Tran, Xuan Huong and Pham, Trinh and Nguyen, Thanh Tam},
  year={2026},
  journal={(under review)}
}
```
