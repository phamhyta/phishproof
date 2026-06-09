# PhishProof

**Repository:** [github.com/phamhyta/phishproof](https://github.com/phamhyta/phishproof)

Selective phishing detection via **Grounded Evidence-Agreement (GEA)**:
`GEA = A · G`, where **A** is cross-agent consensus over typed cues and **G** is mean
tool-verification on those cues.

Paper target: *Computers & Electrical Engineering*. LaTeX sources in `paper/`.

## PhishSel benchmark

| Split | Pages | Phishing | Benign |
|---|---:|---:|---:|
| Calibration | 332 | 166 | 166 |
| Test | 998 | 499 | 499 |

Manifests: `data/phishsel_final/` (SHA-256 in `CHECKSUMS.sha256`).

Page snapshots (HTML + screenshots, ~700 MB) are distributed separately on Zenodo —
see `ARTIFACTS.md`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[logo,dev]"

# verify manifests + tools (no API)
python scripts/preflight.py --data data/phishsel_final

# reproduce headline tables from artifacts/ (no API, no copy step)
python scripts/make_all_tables.py

# import D1 Phishpedia outputs (if you have external/pp_res*.txt)
python scripts/import_phishpedia_d1.py
python scripts/run_subsumes_d1.py
```

Full panel run requires Ollama (Llama-3.2-3B, Qwen2.5-3B) + OpenAI API (GPT-4o vision).
Agent calls are content-hash cached under `data/cache/` (not shipped — see `ARTIFACTS.md`).

## Repository layout

| Path | Contents |
|---|---|
| `phishproof/` | Core pipeline |
| `scripts/` | Experiments + table generation |
| `configs/` | Panel / experiment YAML |
| `data/phishsel_final/` | Calibration + test manifests |
| `artifacts/` | Headline result files (bundles, detectors, RQ JSON) |
| `paper/` | LaTeX paper (tables, figures, `tex/main_phishproof.tex`) |

## Citation

```bibtex
@article{phishproof2026,
  title   = {PhishProof: Trustworthy Phishing Detection with Grounded Cross-Model Evidence Agreement},
  author  = {TODO},
  journal = {Computers and Electrical Engineering},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT — see `LICENSE`. PhishSel page snapshots derive from the
[Phishpedia](https://github.com/lindsey98/Phishpedia) corpus; cite accordingly.
