#!/usr/bin/env bash

export MAX_JOBS=1
export TORCHDYNAMO_DISABLE=1

# Train splatfacto with DA2 depth supervision on a sparse every-2 KITTI nerfstudio dataset.
#
# Default: KITTI seq02 / drive 0034. For another sequence, set KITTI_SEQ_DIR (paths are derived
# automatically unless you override NERFSTUDIO_SRC / DATA_DIR):
#
#   KITTI_SEQ_DIR=/path/to/KITTISeq05_..._densegt bash scripts/train_splatfacto_kitti_sparse_da2.sh
#
# Prerequisites (seq05 is not prebuilt in this repo):
#   1. Dense nerfstudio: data/nerfstudio/kitti_seq05_0018/transforms.json
#   2. Sparse nerfstudio: data/nerfstudio/kitti_seq05_0018_sparse_every2 (see scripts/make_nerfstudio_kitti_sparse.py)
#   3. Depth maps: ${KITTI_SEQ_DIR}/depths_da2 (see docs/depth-anything-v2.md)
#
# Optional photometric masked depth:
#   1) Train once without masks (default below).
#   2) bash scripts/generate_splatfacto_photo_masks.py --load-config <config.yml> --output-dir <masks>
#   3) PHOTO_MASK_DIR=<masks> bash scripts/train_splatfacto_kitti_sparse_da2.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NERFSTUDIO_DIR="${PROJECT_ROOT}/nerfstudio"

DATASET_FAMILY="${DATASET_FAMILY:-kitti}"
SPARSE_KITTI_ROOT="${SPARSE_KITTI_ROOT:-${PROJECT_ROOT}/data/kitti/kitti_select_static_5seq_sparse_every2}"
MIP360_SPARSE_ROOT="${MIP360_SPARSE_ROOT:-${PROJECT_ROOT}/data/mip360_sparse}"
MIP360_SCENE="${MIP360_SCENE:-bicycle}"
KITTI_SEQ_DIR="${KITTI_SEQ_DIR:-${SPARSE_KITTI_ROOT}/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt}"

derive_nerfstudio_slug() {
  local base
  base="$(basename "${KITTI_SEQ_DIR}")"
  if [[ "${base}" =~ KITTISeq([0-9]+)_.*drive_([0-9]+)_sync ]]; then
    printf 'kitti_seq%02d_%s' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    return 0
  fi
  echo "Could not derive nerfstudio dataset name from: ${base}" >&2
  echo "Expected folder name like KITTISeq05_2011_09_30_drive_0018_sync_...." >&2
  echo "Set NERFSTUDIO_SRC and DATA_DIR explicitly." >&2
  return 1
}

if [[ "${DATASET_FAMILY}" == mip360 ]]; then
  SEQ_SLUG="${SEQ_SLUG:-${MIP360_SCENE}_sparse}"
  NERFSTUDIO_SRC="${NERFSTUDIO_SRC:-${PROJECT_ROOT}/data/nerfstudio/${SEQ_SLUG}}"
  DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/nerfstudio/${SEQ_SLUG}_da2}"
  KITTI_SEQ_DIR="${KITTI_SEQ_DIR:-${MIP360_SPARSE_ROOT}/${MIP360_SCENE}}"
  SPARSE_TAG=""
else
  SEQ_SLUG="${SEQ_SLUG:-$(derive_nerfstudio_slug)}"
  NERFSTUDIO_SRC="${NERFSTUDIO_SRC:-${PROJECT_ROOT}/data/nerfstudio/${SEQ_SLUG}_sparse_every2}"
  DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/nerfstudio/${SEQ_SLUG}_sparse_every2_da2}"
  SPARSE_TAG="_sparse_every2"
fi

DEPTH_SUP_TYPE="${DEPTH_SUP_TYPE:-da2}"
LAMBDA_DEPTH="${LAMBDA_DEPTH:-0.05}"
DEPTH_LOSS_TYPE="${DEPTH_LOSS_TYPE:-mse}"
MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-50000}"
PHOTO_MASK_DIR="${PHOTO_MASK_DIR:-}"
PHOTO_MASK_THRESHOLD="${PHOTO_MASK_THRESHOLD:-0.12}"
PHOTO_MASK_MODE="${PHOTO_MASK_MODE:-high}"

# Naming mirrors MipNeRF sweep convention:
#   {ds_tag}_sparse_every2_{sup_type}_lambda{lambda}_{mode}{thresh_tag}_{iters}
# e.g. kitti_seq02_0034_sparse_every2_da2_lambda0.05_low012_50000
#      kitti_seq02_0034_sparse_every2_da2_lambda0.05_nomask_50000
THRESH_TAG="${PHOTO_MASK_THRESHOLD//./}"   # 0.12 -> "012"
if [[ -n "${PHOTO_MASK_DIR}" ]]; then
  MASK_LABEL="${PHOTO_MASK_MODE}${THRESH_TAG}"
else
  MASK_LABEL="nomask"
fi
EXP_NAME="${EXP_NAME:-${SEQ_SLUG}${SPARSE_TAG}_${DEPTH_SUP_TYPE}_lambda${LAMBDA_DEPTH}_${MASK_LABEL}_${MAX_NUM_ITERATIONS}}"

DENSE_NERFSTUDIO="${DENSE_NERFSTUDIO:-${PROJECT_ROOT}/data/nerfstudio/${SEQ_SLUG%%_sparse*}}"
DEPTH_DIR="${DEPTH_DIR:-${KITTI_SEQ_DIR}/depths_${DEPTH_SUP_TYPE}}"

if [[ ! -d "${KITTI_SEQ_DIR}" ]]; then
  echo "Missing sequence directory: ${KITTI_SEQ_DIR}" >&2
  exit 1
fi

if [[ ! -f "${NERFSTUDIO_SRC}/transforms.json" ]]; then
  echo "Missing sparse nerfstudio dataset: ${NERFSTUDIO_SRC}/transforms.json" >&2
  if [[ ! -f "${DENSE_NERFSTUDIO}/transforms.json" ]]; then
    echo "Also missing dense nerfstudio dataset: ${DENSE_NERFSTUDIO}/transforms.json" >&2
    echo "Only seq02 dense nerfstudio is checked into this repo. Create dense transforms for ${SEQ_SLUG} first." >&2
  else
    echo "Create the sparse dataset with:" >&2
    echo "  cd ${PROJECT_ROOT}" >&2
    echo "  python scripts/make_nerfstudio_kitti_sparse.py \\" >&2
    echo "    --src ${DENSE_NERFSTUDIO} \\" >&2
    echo "    --dst ${NERFSTUDIO_SRC} \\" >&2
    echo "    --images ${KITTI_SEQ_DIR}/images \\" >&2
    echo "    --stride 2 --overwrite" >&2
  fi
  exit 1
fi

if [[ ! -d "${DEPTH_DIR}" ]]; then
  echo "Missing depth supervision folder: ${DEPTH_DIR}" >&2
  echo "Generate aligned DA2 depth (docs/depth-anything-v2.md) or set DEPTH_SUP_TYPE=gt if using depths_gt." >&2
  exit 1
fi

echo "KITTI sequence:     ${KITTI_SEQ_DIR}"
echo "Nerfstudio (sparse): ${NERFSTUDIO_SRC}"
echo "Training data:      ${DATA_DIR}"
echo "Depth supervision:  ${DEPTH_DIR}"
echo "Experiment name:    ${EXP_NAME}"

if [[ ! -f "${DATA_DIR}/transforms.json" ]]; then
  python "${SCRIPT_DIR}/make_nerfstudio_kitti_depth.py" \
    --src "${NERFSTUDIO_SRC}" \
    --dst "${DATA_DIR}" \
    --depth-dir "${DEPTH_DIR}" \
    --depth-sup-type "${DEPTH_SUP_TYPE}" \
    --overwrite
fi

if [[ -n "${PHOTO_MASK_DIR}" ]]; then
  if [[ ! -d "${PHOTO_MASK_DIR}" ]]; then
    echo "Missing PHOTO_MASK_DIR: ${PHOTO_MASK_DIR}" >&2
    exit 1
  fi
  python "${SCRIPT_DIR}/attach_nerfstudio_photo_masks.py" \
    --data-dir "${DATA_DIR}" \
    --mask-dir "${PHOTO_MASK_DIR}" \
    --overwrite
  echo "Photometric masks: ${PHOTO_MASK_DIR} (mode=${PHOTO_MASK_MODE})"
fi

cd "${NERFSTUDIO_DIR}"

NS_ARGS=(
  --data "${DATA_DIR}"
  --experiment-name "${EXP_NAME}"
  --max-num-iterations "${MAX_NUM_ITERATIONS}"
  --pipeline.model.lambda-depth "${LAMBDA_DEPTH}"
  --pipeline.model.depth-loss-type "${DEPTH_LOSS_TYPE}"
  --pipeline.model.photo-mask-mode "${PHOTO_MASK_MODE}"
  --vis tensorboard
)

ns-train splatfacto-da2 "${NS_ARGS[@]}"
