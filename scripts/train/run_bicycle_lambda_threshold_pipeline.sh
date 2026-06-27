#!/usr/bin/env bash
# End-to-end bicycle sparse: same lambda + threshold (0.14 low) workflow as KITTI seq02.
#
# Prereqs on GPU machine:
#   1. Place Kaggle sparse bicycle under data/mip360_sparse/bicycle
#      (images/, sparse/0/, depths_colmap/, depths_da2/ — no depths_gt/)
#   2. bash scripts/data_prep/align_da2_mip360_colmap.sh   # if depths_da2/ not built yet
#   3. bash scripts/data_prep/prepare_mip360_sparse_scene.sh
#   4. bash scripts/train/train_mipnerf_sparse_rgbonly.sh   # for MipNeRF masks
#   5. bash scripts/masks/generate_mipnerf_photo_masks.sh
#
# Local splatfacto (sequential):
#   RUN_SPLATFACTO=1 bash scripts/train/run_bicycle_lambda_threshold_pipeline.sh
#
# Modal splatfacto (parallel, after volume upload — see docs/mip360-bicycle-sparse.md):
#   RUN_MODAL=1 bash scripts/train/run_bicycle_lambda_threshold_pipeline.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPTS_ROOT}/.." && pwd)"

export SCENE=bicycle
export DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/mip360_sparse/bicycle}"
export DS_TAG=bicycle_sparse
export SAMPLE_EVERY=1
export PHOTO_MASK_THRESHOLD=0.14
export PHOTO_MASK_MODE=low
export SWEEP_LAMBDAS="${SWEEP_LAMBDAS:-0.0 0.05 0.1 0.15}"
export MAX_STEPS="${MAX_STEPS:-50000}"
export MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-50000}"

echo "=== Bicycle sparse pipeline ==="
echo "DATA_DIR=${DATA_DIR}"
echo "Lambdas: ${SWEEP_LAMBDAS}"
echo "Mask: ${PHOTO_MASK_MODE} @ ${PHOTO_MASK_THRESHOLD}"

if [[ "${RUN_PREPARE:-0}" == 1 ]]; then
  bash "${SCRIPTS_ROOT}/data_prep/prepare_mip360_sparse_scene.sh"
fi

if [[ "${RUN_MIPNERF_RGB:-0}" == 1 ]]; then
  bash "${SCRIPT_DIR}/train_mipnerf_sparse_rgbonly.sh"
fi

if [[ "${RUN_MIPNERF_MASKS:-0}" == 1 ]]; then
  bash "${SCRIPTS_ROOT}/masks/generate_mipnerf_photo_masks.sh"
fi

if [[ "${RUN_MIPNERF_SWEEP:-1}" == 1 ]]; then
  bash "${SCRIPT_DIR}/train_mipnerf_sparse_da2_sweep.sh"
fi

if [[ "${RUN_SPLATFACTO:-0}" == 1 ]]; then
  read -r -a LAMBDAS <<< "${SWEEP_LAMBDAS}"
  for LAM in "${LAMBDAS[@]}"; do
    DATASET_FAMILY=mip360 MIP360_SCENE=bicycle LAMBDA_DEPTH="${LAM}" \
      bash "${SCRIPT_DIR}/train_splatfacto_kitti_sparse_da2.sh"
  done
fi

if [[ "${RUN_MODAL:-0}" == 1 ]]; then
  modal run --detach "${PROJECT_ROOT}/modal_train_splatfacto.py::sweep" \
    --dataset-family mip360 \
    --mip360-scene bicycle \
    --lambdas "${SWEEP_LAMBDAS}" \
    --max-num-iterations "${MAX_NUM_ITERATIONS}"

  modal run --detach "${PROJECT_ROOT}/modal_train_splatfacto.py::sweep_lambda_threshold" \
    --dataset-family mip360 \
    --mip360-scene bicycle \
    --lambdas "${SWEEP_LAMBDAS}" \
    --thresholds "${PHOTO_MASK_THRESHOLD}" \
    --photo-mask-mode "${PHOTO_MASK_MODE}" \
    --max-num-iterations "${MAX_NUM_ITERATIONS}"
fi

if [[ "${RUN_MODAL_EVAL:-0}" == 1 ]]; then
  modal run "${PROJECT_ROOT}/modal_train_splatfacto.py::sweep_eval" \
    --dataset-family mip360 \
    --mip360-scene bicycle \
    --lambdas "${SWEEP_LAMBDAS}" \
    --photo-mask-mode "${PHOTO_MASK_MODE}" \
    --photo-mask-threshold "${PHOTO_MASK_THRESHOLD}" \
    --max-num-iterations "${MAX_NUM_ITERATIONS}" \
    --masked
fi

echo "Done."
