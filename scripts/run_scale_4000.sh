#!/usr/bin/env bash
# Scale PhishSel to 4,000 pages and re-run the panel + baselines + experiments.
#
# COST: ~12h wall-clock (text-local CPU + GPT-4o vision) + ~$30 OpenAI.
# DO NOT RUN BLIND. The 998-page paper numbers are honest; scaling up only buys
# tighter CIs. Run only if you have decided the additional power is worth $30+12h.
#
# Cache reuse: agent outputs are content-hash cached. Any page already in
# data/cache/ from the 998-run is reused for free, so the marginal cost is only
# the ~3000 NEW pages.
#
# Usage:
#   bash scripts/run_scale_4000.sh
# Resumable: re-running picks up from data/cache and skips finished pages.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3

OUT_DATA=data/phishsel_4000
OUT_BUNDLE=results/bundle_scale_4000.jsonl
OUT_EXP=results/rq1_scale_4000.json

echo "=== [1/4] Rebuild PhishSel at n_pages=4000 (seed=0) ==="
"$PYTHON" scripts/build_phishsel.py \
    --n-pages 4000 \
    --calib-frac 0.15 \
    --seed 0 \
    --out "$OUT_DATA"

echo "=== [2/4] Run panel + baselines (cache reuse) ==="
# Make sure Ollama is running:
#   ollama serve &  (with the env in CLAUDE/README)
"$PYTHON" scripts/build_results_bundle.py \
    --data "$OUT_DATA" \
    --out "$OUT_BUNDLE" \
    --no-logo

echo "=== [3/4] Aggregate experiment numbers ==="
"$PYTHON" scripts/run_experiments.py \
    --bundle "$OUT_BUNDLE" \
    --out "$OUT_EXP"

echo "=== [4/4] Done. Compare to 998-page numbers ==="
echo "  998-page: results/rq1_main.json   →  AURC 2.3 (CI [1.4, 3.2])"
echo "  4000:     $OUT_EXP                 →  expect CI ~½ width"
echo
echo "If the 4000-page AURC vs B6 CI excludes 0, update tab_main + abstract."
echo "If not, KEEP the 998-page numbers (honest); 4000 only confirms direction."
