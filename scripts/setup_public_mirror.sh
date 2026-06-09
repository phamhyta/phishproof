#!/usr/bin/env bash
# Create / refresh the public git mirror beside this repo (symlink-based).
#
# Layout:
#   ../phishing-detection-public/   ← git remote target (symlinks → this repo)
#   ./public                        ← symlink → ../phishing-detection-public
#
# Edit code in phishing-detection/; the public folder reflects changes instantly.
# Before `git push` from the public folder, run:
#   ./scripts/materialize_public_for_git.sh
# After push, restore symlinks:
#   ./scripts/setup_public_mirror.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PUBLIC="${PUBLIC_DIR:-$ROOT/../phishing-detection-public}"
MAIN="$ROOT"

link() {
  local target="$1" linkpath="$2"
  mkdir -p "$(dirname "$linkpath")"
  if [[ -e "$linkpath" && ! -L "$linkpath" ]]; then
    echo "[skip] $linkpath exists and is not a symlink"
    return
  fi
  ln -sfn "$target" "$linkpath"
}

mkdir -p "$PUBLIC"

# --- code (always synced) ---
link "$MAIN/pyproject.toml"           "$PUBLIC/pyproject.toml"
link "$MAIN/phishproof"               "$PUBLIC/phishproof"
link "$MAIN/scripts"                  "$PUBLIC/scripts"
link "$MAIN/configs"                  "$PUBLIC/configs"
if [[ -d "$MAIN/docs" ]]; then
  link "$MAIN/docs"                   "$PUBLIC/docs"
fi

# --- PhishSel manifests (~560 KB) ---
mkdir -p "$PUBLIC/data"
link "$MAIN/data/phishsel_final"      "$PUBLIC/data/phishsel_final"

# --- paper source (compile on Overleaf / local TeX) ---
link "$MAIN/phishing-detection-latex" "$PUBLIC/paper"

# --- headline experiment artifacts (Tier A) ---
mkdir -p "$PUBLIC/artifacts"
ART=(
  bundle_final.jsonl
  calibrator.json
  operating_point.json
  detector_d1.jsonl
  detector_d3.jsonl
  phishpedia_d1.jsonl
  subsumes_d1.json
  d1_import.json
  b3_selfconsistency_full.jsonl
  bundle_samefamily.jsonl
  rq1_main.json
  rq2_final.json
  leave_brands_out.json
  rq7_adaptive.json
  rq7_occlude.json
  rq7_cloak.json
  rq7_both.json
  verifier_soundness.json
  failure_modes.json
  hfgea.json
  cascade.json
  esp.json
  panel_strength.json
  per_class_calibration.json
)
for f in "${ART[@]}"; do
  if [[ -f "$MAIN/results/$f" ]]; then
    link "$MAIN/results/$f" "$PUBLIC/artifacts/$f"
  fi
done

# --- public-only files (real files, not symlinks) ---
if [[ ! -f "$PUBLIC/LICENSE" ]]; then
  cp "$MAIN/public-release/LICENSE" "$PUBLIC/LICENSE" 2>/dev/null || true
fi
if [[ -f "$MAIN/public-release/README.md" && ! -L "$PUBLIC/README.md" ]]; then
  cp "$MAIN/public-release/README.md" "$PUBLIC/README.md"
fi
if [[ -f "$MAIN/public-release/.gitignore" ]]; then
  cp "$MAIN/public-release/.gitignore" "$PUBLIC/.gitignore"
fi
if [[ -f "$MAIN/public-release/ARTIFACTS.md" ]]; then
  cp "$MAIN/public-release/ARTIFACTS.md" "$PUBLIC/ARTIFACTS.md"
fi
if [[ -f "$PUBLIC/data/phishsel_final/calibration.jsonl" ]]; then
  (cd "$PUBLIC/data/phishsel_final" && shasum -a 256 calibration.jsonl test.jsonl) \
    > "$PUBLIC/CHECKSUMS.sha256"
fi

# --- pointer back into main repo ---
link "$PUBLIC" "$MAIN/public"

echo "[ok] public mirror: $PUBLIC"
echo "     enter via:  cd $MAIN/public"
echo "     git push:   ./scripts/materialize_public_for_git.sh  (then commit in public/)"
