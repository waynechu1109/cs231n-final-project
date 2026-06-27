#!/usr/bin/env bash
# Generate fixed photometric masks for MipNeRF-360 (from an RGB-only checkpoint).
#
# Example (bicycle sparse, threshold 0.14, low mode):
#   SCENE=bicycle \
#   DATA_DIR=${REPO}/data/mip360_sparse/bicycle \
#   RGB_CHECKPOINT_DIR=${DATA_DIR}/logs/checkpoints_bicycle_sparse_rgbonly_50000 \
#   PHOTO_MASK_THRESHOLD=0.14 \
#   PHOTO_MASK_MODE=low \
#   SAMPLE_EVERY=1 \
#   bash scripts/masks/generate_mipnerf_photo_masks.sh
#
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MIPNERF_DIR="${MIPNERF_DIR:-${REPO}/outdoor-nerf-depth/nerf-methods/mipnerf360}"
MIP360_SPARSE_ROOT="${MIP360_SPARSE_ROOT:-${REPO}/data/mip360_sparse}"

SCENE="${SCENE:-bicycle}"
DATA_DIR="${DATA_DIR:-${MIP360_SPARSE_ROOT}/${SCENE}}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1}"
PHOTO_MASK_THRESHOLD="${PHOTO_MASK_THRESHOLD:-0.14}"
PHOTO_MASK_MODE="${PHOTO_MASK_MODE:-low}"
RESTORE_STEP="${RESTORE_STEP:--1}"
MAX_STEPS="${MAX_STEPS:-50000}"
CONFIG_FACTOR="${CONFIG_FACTOR:-4}"

DS_TAG="${DS_TAG:-${SCENE}_sparse}"
thresh_tag="${PHOTO_MASK_THRESHOLD//./}"
MASK_DIR="${DATA_DIR}/photo_masks_rgbonly_${PHOTO_MASK_MODE}${thresh_tag}_sampleevery${SAMPLE_EVERY}"
RGB_CHECKPOINT_DIR="${RGB_CHECKPOINT_DIR:-${DATA_DIR}/logs/checkpoints_${DS_TAG}_rgbonly_${MAX_STEPS}}"

if [[ -d "${DATA_DIR}/images_4" ]]; then
  CONFIG_FACTOR=4
elif [[ -d "${DATA_DIR}/images_2" ]]; then
  CONFIG_FACTOR=2
else
  CONFIG_FACTOR=0
fi

if [[ ! -d "${RGB_CHECKPOINT_DIR}" ]]; then
  echo "Missing RGB checkpoint dir: ${RGB_CHECKPOINT_DIR}" >&2
  echo "Train an RGB-only run first (lambda_depth=0, compute_disp_metrics=False)." >&2
  exit 1
fi

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate multinerf
fi

cd "${MIPNERF_DIR}"

python generate_fixed_photo_masks.py \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir='${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir='${RGB_CHECKPOINT_DIR}'" \
  --gin_bindings="Config.restore_step=${RESTORE_STEP}" \
  --gin_bindings="Config.max_steps=${MAX_STEPS}" \
  --gin_bindings="Config.factor=${CONFIG_FACTOR}" \
  --gin_bindings="Config.sample_every=${SAMPLE_EVERY}" \
  --gin_bindings="Config.compute_disp_metrics=False" \
  --gin_bindings="Config.lambda_depth=0.0" \
  --gin_bindings="Config.fixed_photo_mask_dir='${MASK_DIR}'" \
  --gin_bindings="Config.photo_mask_mode='${PHOTO_MASK_MODE}'" \
  --gin_bindings="Config.photo_mask_threshold=${PHOTO_MASK_THRESHOLD}" \
  --logtostderr

echo "Wrote masks to: ${MASK_DIR}"
