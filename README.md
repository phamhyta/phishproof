# PhishProof

PhishProof is the public code scaffold for our paper under review.

This repository intentionally does not include benchmark data, experiment outputs, model responses, or reproduction artifacts.

The full research code and artifacts will be released after the paper is
accepted.

## What is included

- A small installable Python package.
- A stable JSONL input/output shape for page-level phishing checks.
- A toy URL-signal detector used only for smoke tests.
- A minimal CLI showing how the final pipeline will be invoked.

The detector shipped here is not the paper implementation and should not be used
as a benchmark result.

## Install

```bash
uv sync
```

## Run the demo

```bash
uv run phishproof \
  --input examples/pages.jsonl \
  --config configs/demo.json \
  --output /tmp/phishproof-demo.jsonl
```

Print the output:

```bash
cat /tmp/phishproof-demo.jsonl
```

## Input format

Each line is one JSON object:

```json
{"id": "sample-1", "url": "https://login.example.test/account", "title": "Account sign-in"}
```

Supported fields:

- `id`: stable page identifier.
- `url`: page URL.
- `title`: optional page title.
- `html_path`: optional path to a saved HTML file.
- `screenshot_path`: optional path to a screenshot.
- `metadata`: optional object with caller-side notes.

The CLI writes one JSON object per input page with `id`, `label`,
`confidence`, and `reasons`.

## Development

```bash
uv run ruff check phishproof
uv run python -m phishproof --input examples/pages.jsonl
```

## Release status

This is a pre-acceptance public scaffold. The private research repository keeps
the manuscript, experiment scripts, result bundles, and data manifests until the
review process is complete.

## License

MIT. See `LICENSE`.
