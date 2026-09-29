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
| `results/` | Per-page panel bundles, the operating calibrator, the frozen policy, and the revision results (`results/revision_v2/`) |
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

## Results and large files

| File | Location |
|------|----------|
| Per-page panel bundles with verdicts, confidences, cited cues, agreement and grounding diagnostics (`bundle_or_full.jsonl`, `bundle_apwg_or_full.jsonl`, `bundle_trop_or_full.jsonl`) | [`results/revision/`](results/revision/) |
| Operating calibrator (`calibrator_or.json`) | [`results/`](results/) |
| Frozen verification policy: operating threshold and calibrated logo threshold (`policy_frozen.json`) | [`results/revision_v2/t8/`](results/revision_v2/t8/) |
| Metric, baseline, verifier, attack and cost summaries (`t8`--`t12` JSON) | [`results/revision_v2/`](results/revision_v2/) |
| Per-page records: eligibility, consolidator outputs, verifier units, attack policy, timing (`per_page_records.tar.gz`, 2.8 MB) | [`results/revision_v2/`](results/revision_v2/) |
| Cached model responses (`phishproof_cache.tar.gz`, 18 MB, extracts to `data/cache/`) | [Google Drive](https://drive.google.com/drive/folders/1HeombUfo5jFxrOk1mTC9nZFJY4U2dI23?usp=sharing) |
| Benign login-page captures used in the splits (`phishproof_benign_captures.tar.gz`, 2.4 GB, 3,261 pages, extracts to `data/benign_raw/`) | [Google Drive](https://drive.google.com/drive/folders/1HeombUfo5jFxrOk1mTC9nZFJY4U2dI23?usp=sharing) |

Extract the per-page records in the repository root
(`tar -xzf results/revision_v2/per_page_records.tar.gz`). With the bundles, the frozen
policy and these records, the score analyses and the complete-policy decisions can be rerun
without new model calls, and `scripts/check_revision_v2.py` re-derives the reported policy
and validity counts. Extract the two Drive archives in the
repository root as well; the cached responses let the panel replay from the cache instead of
calling the models again. SHA-256 checksums of the Drive archives:

```
604379bc5c72a575499f4af148a9963645628c32b63485fdbe5176e78062e243  phishproof_cache.tar.gz
1e0711040f859e59b7c765285954cbbf085cf6f02f6edc8742d8b66d7f0d7cde  phishproof_benign_captures.tar.gz
```

## Running

```bash
# Panel run on a manifest (uses data/cache/ when available)
python scripts/run_panel.py --manifest data/phishsel_final/test.jsonl --out results/panel.jsonl

# Selective-prediction metrics with bootstrap intervals from a score bundle
python scripts/run_experiments.py --bundle results/revision/bundle_or_full.jsonl --out results/
```

## License

MIT (see `LICENSE`). Page captures remain under the terms of their source corpora.
