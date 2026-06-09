#!/usr/bin/env bash
# Smoke test for artifact reproducibility.
# Runs offline checks + regenerates every paper table from artifacts/ (or results/).
# Exits non-zero on any failure.
#
# Usage:
#   ./scripts/test_reproducibility.sh
#
# Environment:
#   PYTHON: python interpreter (default: .venv/bin/python if exists, else python3)
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

echo "=== [1/3] Schema + calibration + metrics smoke tests (no raw data needed) ==="
"$PYTHON" scripts/check_setup.py
"$PYTHON" scripts/check_calibration.py
"$PYTHON" scripts/check_metrics.py

echo "=== [2/3] Regenerate every paper table from artifacts ==="
"$PYTHON" scripts/make_all_tables.py > /tmp/phishproof_tables.log 2>&1 || {
  cat /tmp/phishproof_tables.log
  echo "make_all_tables.py FAILED"
  exit 1
}
cat /tmp/phishproof_tables.log
for expect in "tab_main (RQ1)" "tab_detect" "tab_rlwr" "tab_ablation"; do
  if ! grep -q "$expect" /tmp/phishproof_tables.log; then
    echo "MISSING expected section: $expect"
    exit 1
  fi
done

echo "=== [3/3] Verify checksums of released splits ==="
if [[ -f public-release/CHECKSUMS.sha256 ]]; then
  ( cd public-release && shasum -a 256 -c CHECKSUMS.sha256 )
elif [[ -f data/phishsel_final/CHECKSUMS.sha256 ]]; then
  ( cd data/phishsel_final && shasum -a 256 -c CHECKSUMS.sha256 )
else
  echo "WARN: no CHECKSUMS.sha256 found (skip)"
fi

echo
echo "ALL REPRODUCIBILITY CHECKS PASSED"
