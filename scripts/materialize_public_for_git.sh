#!/usr/bin/env bash
# Replace symlinks in ../phishing-detection-public with real file copies for git commit.
# Git stores symlinks as pointers — clones would break without this step.
#
# Usage:
#   ./scripts/materialize_public_for_git.sh
#   cd ../phishing-detection-public && git add -A && git commit ...
#   ./scripts/setup_public_mirror.sh    # restore symlinks after commit
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PUBLIC="${PUBLIC_DIR:-$ROOT/../phishing-detection-public}"

if [[ ! -d "$PUBLIC" ]]; then
  echo "[fail] $PUBLIC missing — run ./scripts/setup_public_mirror.sh first"
  exit 1
fi

echo "[materialize] copying into $PUBLIC (dereferencing symlinks)..."

copy_tree() {
  local src="$1" dst="$2"
  rm -rf "$dst"
  mkdir -p "$dst"
  rsync -a --copy-links "$src/" "$dst/"
}

copy_file() {
  local src="$1" dst="$2"
  rm -f "$dst"
  mkdir -p "$(dirname "$dst")"
  cp -f "$src" "$dst"
}

copy_tree "$ROOT/phishproof"               "$PUBLIC/phishproof"
copy_tree "$ROOT/scripts"                  "$PUBLIC/scripts"
copy_tree "$ROOT/configs"                  "$PUBLIC/configs"
copy_file  "$ROOT/pyproject.toml"          "$PUBLIC/pyproject.toml"
[[ -d "$ROOT/docs" ]] && copy_tree "$ROOT/docs" "$PUBLIC/docs"

copy_tree "$ROOT/data/phishsel_final"      "$PUBLIC/data/phishsel_final"

# paper: ship compile-ready subset (exclude internal idea/review drafts)
rm -rf "$PUBLIC/paper"
mkdir -p "$PUBLIC/paper"
rsync -a \
  --exclude 'ideas/' \
  --exclude 'reviews/' \
  --exclude 'advanced/' \
  --exclude 'notes*.md' \
  --exclude 'outline.md' \
  --exclude 'extended-abstract.md' \
  --exclude 'idea-notes.md' \
  --exclude 'CHANGELOG_review.md' \
  --exclude '.git' \
  "$ROOT/phishing-detection-latex/" "$PUBLIC/paper/"

mkdir -p "$PUBLIC/artifacts"
for f in "$PUBLIC/artifacts"/*; do
  [[ -L "$f" ]] && rm -f "$f"
done
ARTIFACTS=(
  bundle_final.jsonl calibrator.json operating_point.json
  detector_d1.jsonl detector_d3.jsonl phishpedia_d1.jsonl subsumes_d1.json
  d1_import.json b3_selfconsistency_full.jsonl bundle_samefamily.jsonl
  rq1_main.json rq2_final.json leave_brands_out.json
  rq7_adaptive.json rq7_occlude.json rq7_cloak.json rq7_both.json
  verifier_soundness.json failure_modes.json hfgea.json cascade.json esp.json
  panel_strength.json per_class_calibration.json
)
for base in "${ARTIFACTS[@]}"; do
  if [[ -f "$ROOT/results/$base" ]]; then
    cp -f "$ROOT/results/$base" "$PUBLIC/artifacts/$base"
  fi
done

# refresh public-only metadata
cp -f "$ROOT/public-release/README.md"     "$PUBLIC/README.md"
cp -f "$ROOT/public-release/.gitignore"    "$PUBLIC/.gitignore"
cp -f "$ROOT/public-release/LICENSE"       "$PUBLIC/LICENSE"
cp -f "$ROOT/public-release/ARTIFACTS.md"  "$PUBLIC/ARTIFACTS.md"
shasum -a 256 "$PUBLIC/data/phishsel_final/calibration.jsonl" \
            "$PUBLIC/data/phishsel_final/test.jsonl" \
  > "$PUBLIC/CHECKSUMS.sha256"

# keep navigation symlink in main repo (not materialized)
ln -sfn "$PUBLIC" "$ROOT/public"

echo "[ok] materialized — safe to git add/commit in $PUBLIC"
echo "     restore dev symlinks: ./scripts/setup_public_mirror.sh"
