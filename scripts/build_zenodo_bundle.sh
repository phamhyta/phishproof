#!/usr/bin/env bash
# Build Zenodo upload bundles (Tier B and Tier C).
#
# Tier B: agent cache + D1 raw outputs (~100MB)  — fully restores re-scoring without re-running models.
# Tier C: page snapshots (HTML + PNG) for the 1,330 PhishSel pages (~800MB) — full from-scratch reproduction.
#
# Outputs go to public-release/zenodo/ as .tar.zst (or .tar.gz fallback) with .sha256 sidecars.
# These files are NOT committed to git — upload them manually to Zenodo, then paste DOI into
# public-release/ARTIFACTS.md (TIER_B_DOI / TIER_C_DOI placeholders).

set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="public-release/zenodo"
mkdir -p "$OUT_DIR"

if command -v zstd >/dev/null; then
  COMPRESS=(tar --use-compress-program="zstd -19 -T0" -cf)
  EXT="tar.zst"
else
  COMPRESS=(tar -czf)
  EXT="tar.gz"
fi

# ---------- Tier B: agent cache + D1 raw ----------
B_OUT="$OUT_DIR/phishproof-tierB-cache.$EXT"
echo "[1/2] Building Tier B → $B_OUT"
B_LIST=(data/cache)
if compgen -G "external/pp_res*.txt" >/dev/null; then
  B_LIST+=(external/pp_res*.txt)
fi
[[ -f results/detector_d1.jsonl ]] && B_LIST+=(results/detector_d1.jsonl)
[[ -f results/phishpedia_d1.jsonl ]] && B_LIST+=(results/phishpedia_d1.jsonl)
"${COMPRESS[@]}" "$B_OUT" "${B_LIST[@]}"
shasum -a 256 "$B_OUT" | tee "$B_OUT.sha256"
ls -lh "$B_OUT"

# ---------- Tier C: page snapshots for the 1,330 PhishSel pages ----------
C_OUT="$OUT_DIR/phishproof-tierC-snapshots.$EXT"
echo "[2/2] Building Tier C → $C_OUT"
LIST_FILE=$(mktemp)
trap 'rm -f "$LIST_FILE"' EXIT

PYTHON="${PYTHON:-.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3

"$PYTHON" - <<'PY' >"$LIST_FILE"
import json, sys
from pathlib import Path
paths = []
for split in ("calibration.jsonl", "test.jsonl"):
    p = Path("data/phishsel_final") / split
    if not p.exists():
        sys.stderr.write(f"missing {p}\n"); sys.exit(1)
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        raw = r.get("raw_dir")
        if raw and Path(raw).is_dir():
            paths.append(raw)
print("\n".join(sorted(set(paths))))
PY

N=$(wc -l <"$LIST_FILE")
echo "  packing $N page folders"
"${COMPRESS[@]}" "$C_OUT" -T "$LIST_FILE" data/phishsel_final/calibration.jsonl data/phishsel_final/test.jsonl
shasum -a 256 "$C_OUT" | tee "$C_OUT.sha256"
ls -lh "$C_OUT"

echo
echo "DONE. Upload to Zenodo:"
echo "  $B_OUT  ($(du -h "$B_OUT" | cut -f1))"
echo "  $C_OUT  ($(du -h "$C_OUT" | cut -f1))"
echo "After upload, paste DOIs into public-release/ARTIFACTS.md."
