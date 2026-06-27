#!/usr/bin/env bash
# RGB-only MipNeRF base run (lambda_depth=0) for mask generation / nomask baseline.
#
#   SCENE=bicycle DATA_DIR=.../mip360_sparse/bicycle SAMPLE_EVERY=1 \
#   bash scripts/train/train_mipnerf_sparse_rgbonly.sh
#
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MIPNERF_DIR="${MIPNERF_DIR:-${REPO}/outdoor-nerf-depth/nerf-methods/mipnerf360}"
MIP360_SPARSE_ROOT="${MIP360_SPARSE_ROOT:-${REPO}/data/mip360_sparse}"

SCENE="${SCENE:-bicycle}"
DATA_DIR="${DATA_DIR:-${MIP360_SPARSE_ROOT}/${SCENE}}"
DS_TAG="${DS_TAG:-${SCENE}_sparse}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1}"
MAX_STEPS="${MAX_STEPS:-50000}"
CONFIG_FACTOR="${CONFIG_FACTOR:-4}"

if [[ -d "${DATA_DIR}/images_4" ]]; then
  CONFIG_FACTOR=4
elif [[ -d "${DATA_DIR}/images_2" ]]; then
  CONFIG_FACTOR=2
else
  CONFIG_FACTOR=0
fi

CKPT_DIR="${DATA_DIR}/logs/checkpoints_${DS_TAG}_rgbonly_${MAX_STEPS}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate multinerf
fi

cd "${MIPNERF_DIR}"

python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir='${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir='${CKPT_DIR}'" \
  --gin_bindings="Config.max_steps=${MAX_STEPS}" \
  --gin_bindings="Config.factor=${CONFIG_FACTOR}" \
  --gin_bindings="Config.compute_disp_metrics=False" \
  --gin_bindings="Config.lambda_depth=0.0" \
  --gin_bindings="Config.sample_every=${SAMPLE_EVERY}" \
  --gin_bindings="Config.auto_adjust_near_far=True" \
  --gin_bindings="Config.near=0.2" \
  --gin_bindings="Config.far=1000000.0" \
  --logtostderr

echo "Checkpoint: ${CKPT_DIR}"
