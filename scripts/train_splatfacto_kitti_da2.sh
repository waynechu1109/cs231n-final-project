#!/usr/bin/env bash
# Train splatfacto with DA2 depth supervision on the dense KITTI nerfstudio dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NERFSTUDIO_DIR="${PROJECT_ROOT}/nerfstudio"

DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/nerfstudio/kitti_seq02_0034_da2}"
KITTI_SEQ_DIR="${KITTI_SEQ_DIR:-${PROJECT_ROOT}/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt}"
NERFSTUDIO_SRC="${NERFSTUDIO_SRC:-${PROJECT_ROOT}/data/nerfstudio/kitti_seq02_0034}"

DEPTH_SUP_TYPE="${DEPTH_SUP_TYPE:-da2}"
LAMBDA_DEPTH="${LAMBDA_DEPTH:-0.05}"
DEPTH_LOSS_TYPE="${DEPTH_LOSS_TYPE:-mse}"
MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-50000}"

if [[ ! -f "${DATA_DIR}/transforms.json" ]]; then
  python "${SCRIPT_DIR}/make_nerfstudio_kitti_depth.py" \
    --src "${NERFSTUDIO_SRC}" \
    --dst "${DATA_DIR}" \
    --depth-dir "${KITTI_SEQ_DIR}/depths_${DEPTH_SUP_TYPE}" \
    --depth-sup-type "${DEPTH_SUP_TYPE}" \
    --overwrite
fi

cd "${NERFSTUDIO_DIR}"

ns-train splatfacto-da2 \
  --data "${DATA_DIR}" \
  --max-num-iterations "${MAX_NUM_ITERATIONS}" \
  --pipeline.model.lambda-depth "${LAMBDA_DEPTH}" \
  --pipeline.model.depth-loss-type "${DEPTH_LOSS_TYPE}" \
  --vis tensorboard
