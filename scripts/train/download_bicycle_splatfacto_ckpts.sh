#!/usr/bin/env bash
# Download final splatfacto-da2 checkpoints (+ config, eval) for all bicycle experiments
# from the Modal nerf-outputs volume.
#
# Usage:
#   ./scripts/train/download_bicycle_splatfacto_ckpts.sh [local_dest]
#
# Default local_dest: ./local_outputs/bicycle_ckpts

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${1:-$ROOT/local_outputs/bicycle_ckpts}"
VOLUME="nerf-outputs"
PREFIX="bicycle_sparse_da2"

mkdir -p "$OUT_DIR"

echo "Listing experiments on Modal volume '$VOLUME'..."
EXPS=()
while IFS= read -r line; do
  EXPS+=("$line")
done < <(modal volume ls "$VOLUME" | grep "^${PREFIX}" | sort)

if [[ ${#EXPS[@]} -eq 0 ]]; then
  echo "No experiments matching ${PREFIX}* found on volume."
  exit 1
fi

echo "Found ${#EXPS[@]} bicycle experiments."
echo "Downloading to: $OUT_DIR"
echo

downloaded=0
skipped=0

for exp in "${EXPS[@]}"; do
  dest="$OUT_DIR/$exp"
  mkdir -p "$dest"

  timestamps=()
  while IFS= read -r line; do
    ts="${line#${exp}/splatfacto-da2/}"
    timestamps+=("$ts")
  done < <(
    modal volume ls "$VOLUME" "$exp/splatfacto-da2" 2>/dev/null \
      | grep "${exp}/splatfacto-da2/20" \
      | sort
  )

  if [[ ${#timestamps[@]} -eq 0 ]]; then
    echo "[skip] $exp — no run timestamps"
    skipped=$((skipped + 1))
    continue
  fi

  chosen_ts=""
  while IFS= read -r ts; do
    if modal volume ls "$VOLUME" "$exp/splatfacto-da2/$ts/nerfstudio_models" 2>/dev/null \
        | grep -q "step-000049999.ckpt"; then
      chosen_ts="$ts"
      break
    fi
  done < <(printf '%s\n' "${timestamps[@]}" | sort -r)

  if [[ -z "$chosen_ts" ]]; then
    chosen_ts="${timestamps[-1]}"
    ckpts=()
    while IFS= read -r ck; do
      ckpts+=("$ck")
    done < <(
      modal volume ls "$VOLUME" "$exp/splatfacto-da2/$chosen_ts/nerfstudio_models" 2>/dev/null \
        | grep '\.ckpt$' \
        | sed "s|.*/||" \
        | sort -V
    )
    if [[ ${#ckpts[@]} -eq 0 ]]; then
      echo "[skip] $exp — no checkpoints in $chosen_ts"
      skipped=$((skipped + 1))
      continue
    fi
    ckpt_name="${ckpts[-1]}"
  else
    ckpt_name="step-000049999.ckpt"
  fi

  remote_base="$exp/splatfacto-da2/$chosen_ts"
  echo "[get]  $exp  (run $chosen_ts, $ckpt_name)"

  modal volume get "$VOLUME" "$remote_base/config.yml" "$dest/config.yml" 2>/dev/null || true
  modal volume get "$VOLUME" "$remote_base/eval_output.json" "$dest/eval_output.json" 2>/dev/null || true
  modal volume get "$VOLUME" "$remote_base/nerfstudio_models/$ckpt_name" "$dest/$ckpt_name"

  echo "$chosen_ts" > "$dest/run_timestamp.txt"
  downloaded=$((downloaded + 1))
done

echo
echo "Done. Downloaded: $downloaded  Skipped: $skipped"
echo "Checkpoints at: $OUT_DIR"
