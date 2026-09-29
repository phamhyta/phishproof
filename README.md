# PhishProof

Code and input data for the paper *Selective Phishing Detection with Grounded Evidence Agreement
across Language and Vision Models*. A panel of three models (two text LLMs and one vision LLM) cites typed
evidence cues for a web page. PhishProof measures how far the agents agree on each cue type,
checks the agreed cues against the page with deterministic tools, calibrates the resulting
trust score, and abstains when the score is below the operating threshold.

## Repository layout

| Path | Contents |
|------|----------|
| `phishproof/` | Implementation: agents and prompts, evidence schema, per-type agreement, grounding tools, calibration, selective rule, baselines, metrics |
| `scripts/` | Corpus ingestion, panel runs, baselines, experiments, and revision analyses |
| `configs/` | Panel and experiment configuration (`panel.yaml` = deployed panel) |
| `reproducibility/` | System prompt, user template, response schema, configured panel, source hashes |
| `data/` | Split manifests (calibration/test) for the three corpora, with SHA-256 checksums |
| `artifacts/` | Per-page score bundles and the fitted operating calibrator |
| `tests/` | Tests for the verification policy |

## Setup

```bash
pip install -e .            # add ".[logo]" for the CLIP logo check
cp .env.example .env        # then fill in OPENAI_API_KEY and OPENROUTER_API_KEY
```

The deployed panel (`configs/panel.yaml`) uses `meta-llama/llama-3.3-70b-instruct` and
`qwen/qwen-2.5-72b-instruct` through OpenRouter and `gpt-4o` (image detail `low`) through
OpenAI, all at temperature 0. Hosted models can change over time, so new runs may not match
the stored responses exactly.

## Data

Each manifest line is one page: `page_id`, `url`, `label`, `brand`, `split`, `source`, and
paths to `html.txt` and `shot.png`.

| Corpus | Manifests | Phishing pages from |
|--------|-----------|---------------------|
| Phishpedia | `data/phishsel_final/` (710 calib / 4,020 test) | Phishpedia `phish_sample_30k` ([lindsey98/Phishpedia](https://github.com/lindsey98/Phishpedia)) |
| APWG | `data/apwg_final/` (212 / 1,200) | APWG `phishing4190` ([Zenodo 14668190](https://zenodo.org/records/14668190)) |
| TR-OP | `data/trop_final/` (710 / 1,200) | KnowPhish TR-OP, `openphish_5000` |

The benign side of every corpus is a pool of credential-login pages we collected.
Page captures of the source corpora are not redistributed here: obtain them from the links
above under their own terms, then rebuild the manifests with `scripts/ingest_phishpedia.py`,
`scripts/ingest_apwg.py`, and `scripts/ingest_trop.py`.

## Large files

| File | Location |
|------|----------|
| Per-page score bundles (`bundle_or.jsonl`, `bundle_apwg_or.jsonl`, `bundle_trop_or.jsonl`, `bundle_calib_or.jsonl`) | [`artifacts/`](artifacts/) |
| Operating calibrator (`calibrator_or.json`) | [`artifacts/`](artifacts/) |
| Cached model responses (`phishproof_cache.tar.gz`, 18 MB, extracts to `data/cache/`) | [Google Drive](https://drive.google.com/drive/folders/1HeombUfo5jFxrOk1mTC9nZFJY4U2dI23?usp=sharing) |
| Benign login-page captures used in the splits (`phishproof_benign_captures.tar.gz`, 2.4 GB, 3,261 pages, extracts to `data/benign_raw/`) | [Google Drive](https://drive.google.com/drive/folders/1HeombUfo5jFxrOk1mTC9nZFJY4U2dI23?usp=sharing) |

The score bundles hold panel predictions, confidences, consensus cues, agreement and
grounding diagnostics, and baseline scores. With them, the score analyses can be rerun
without new model calls. Extract both archives in the repository root; the cached
responses let the panel replay from the cache instead of calling the models again.
SHA-256 checksums:

```
604379bc5c72a575499f4af148a9963645628c32b63485fdbe5176e78062e243  phishproof_cache.tar.gz
1e0711040f859e59b7c765285954cbbf085cf6f02f6edc8742d8b66d7f0d7cde  phishproof_benign_captures.tar.gz
```

## Running

```bash
# Panel run on a manifest (uses data/cache/ when available)
python scripts/run_panel.py --manifest data/phishsel_final/test.jsonl --out results/panel.jsonl

# Selective-prediction metrics with bootstrap intervals from a score bundle
python scripts/run_experiments.py --bundle artifacts/bundle_or.jsonl --out results/
```

## License

MIT (see `LICENSE`). Page captures remain under the terms of their source corpora.
