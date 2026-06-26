#!/usr/bin/env bash
# Download bicycle compare renders from Modal (per-experiment, avoids dir download issues).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-random5_seed42}"
OUT="${2:-$ROOT/local_outputs/compare_renders/${TAG}_all}"
VOLUME="nerf-outputs"
REMOTE="compare_renders/${TAG}"

mkdir -p "$OUT"

mapfile -t EXPS < <(modal volume ls "$VOLUME" "$REMOTE" 2>/dev/null | sed "s|${REMOTE}/||" | grep "^bicycle_sparse" | sort)
echo "Downloading ${#EXPS[@]} experiments -> $OUT"

for exp in "${EXPS[@]}"; do
  dest="$OUT/$exp"
  mkdir -p "$dest"
  mapfile -t FILES < <(
    modal volume ls "$VOLUME" "$REMOTE/$exp" 2>/dev/null \
      | sed "s|${REMOTE}/${exp}/||" \
      | grep -E '\.(png|json)$|/[^/]+$' || true
  )
  # List view subdirs
  mapfile -t VIEWS < <(
    modal volume ls "$VOLUME" "$REMOTE/$exp" 2>/dev/null \
      | sed "s|${REMOTE}/${exp}/||" \
      | grep -v manifest || true
  )
  for view in "${VIEWS[@]}"; do
    for kind in render.png gt.png; do
      remote="$REMOTE/$exp/$view/$kind"
      local_file="$dest/$view/$kind"
      if modal volume ls "$VOLUME" "$remote" &>/dev/null; then
        mkdir -p "$(dirname "$local_file")"
        modal volume get "$VOLUME" "$remote" "$local_file" 2>/dev/null || true
      fi
    done
  done
  modal volume get "$VOLUME" "$REMOTE/$exp/manifest.json" "$dest/manifest.json" 2>/dev/null || true
  echo "  $exp"
done

echo "Done. Group with:"
echo "  python3 scripts/group_compare_renders.py $OUT --output-dir $ROOT/local_outputs/compare_renders/${TAG}/grouped"
