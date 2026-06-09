#!/usr/bin/env bash
# Lambda × photo-mask sweep for a sparse DA2 dataset (KITTI seq02 or Mip-NeRF 360 bicycle, etc.).
#
# Mip-NeRF 360 bicycle (pre-sparse on disk, sample_every=1):
#   SCENE=bicycle \
#   DATA_DIR=/path/to/data/mip360_sparse/bicycle \
#   DS_TAG=bicycle_sparse \
#   SAMPLE_EVERY=1 \
#   PHOTO_MASK_THRESHOLD=0.14 \
#   PHOTO_MASK_MODE=low \
#   SWEEP_LAMBDAS="0.0 0.05 0.1 0.15" \
#   bash scripts/train_mipnerf_sparse_da2_sweep.sh
#
# KITTI sparse every-2 (default):
#   bash scripts/train_mipnerf_sparse_da2_sweep.sh
#
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MIPNERF_DIR="${MIPNERF_DIR:-${REPO}/outdoor-nerf-depth/nerf-methods/mipnerf360}"

SCENE="${SCENE:-}"
SPARSE_KITTI_ROOT="${SPARSE_KITTI_ROOT:-${REPO}/data/kitti/kitti_select_static_5seq_sparse_every2}"
MIP360_SPARSE_ROOT="${MIP360_SPARSE_ROOT:-${REPO}/data/mip360_sparse}"

LAMBDAS=(0.05 0.1 0.15)
if [[ -n "${SWEEP_LAMBDAS:-}" ]]; then
  read -r -a LAMBDAS <<< "${SWEEP_LAMBDAS}"
fi

PHOTO_MASK_THRESHOLD="${PHOTO_MASK_THRESHOLD:-0.14}"
PHOTO_MASK_MODE="${PHOTO_MASK_MODE:-low}"
DEPTH_SUP_TYPE="${DEPTH_SUP_TYPE:-da2}"
DEPTH_LOSS_TYPE="${DEPTH_LOSS_TYPE:-mse}"
DEPTH_KEEP_RATIO="${DEPTH_KEEP_RATIO:-0.0}"
SAMPLE_EVERY="${SAMPLE_EVERY:-2}"
MAX_STEPS="${MAX_STEPS:-50000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
if [[ -n "${DATA_DIR:-}" ]]; then
  :
elif [[ -n "${SCENE}" ]]; then
  DATA_DIR="${MIP360_SPARSE_ROOT}/${SCENE}"
  SAMPLE_EVERY="${SAMPLE_EVERY:-1}"
else
  KITTI_DIR_NAME="${KITTI_DIR_NAME:-KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt}"
  DATA_DIR="${SPARSE_KITTI_ROOT}/${KITTI_DIR_NAME}"
fi

if [[ -n "${DS_TAG:-}" ]]; then
  ds_tag="${DS_TAG}"
elif [[ -n "${SCENE}" ]]; then
  ds_tag="${SCENE}_sparse"
else
  ds_tag="$(basename "${DATA_DIR}")"
  if [[ "${ds_tag}" =~ KITTISeq([0-9]+)_.*drive_([0-9]+)_sync ]]; then
    ds_tag="$(printf 'kitti_seq%02d_%s' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}")"
  fi
fi

thresh_tag="${PHOTO_MASK_THRESHOLD//./}"
MASK_DIR="${DATA_DIR}/photo_masks_rgbonly_${PHOTO_MASK_MODE}${thresh_tag}_sampleevery${SAMPLE_EVERY}"

if [[ ! -d "${MIPNERF_DIR}" ]]; then
  echo "Missing mipnerf360 dir: ${MIPNERF_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Missing dataset dir: ${DATA_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}/depths_${DEPTH_SUP_TYPE}" ]]; then
  echo "Missing depth folder: ${DATA_DIR}/depths_${DEPTH_SUP_TYPE}" >&2
  exit 1
fi
if [[ -z "${CONFIG_FACTOR:-}" ]]; then
  if [[ -d "${DATA_DIR}/images_4" ]]; then
    CONFIG_FACTOR=4
  elif [[ -d "${DATA_DIR}/images_2" ]]; then
    CONFIG_FACTOR=2
  else
    CONFIG_FACTOR=0
  fi
fi
if [[ ! -d "${MASK_DIR}" ]]; then
  echo "Missing photo-mask dir: ${MASK_DIR}" >&2
  echo "Generate masks first, e.g.:" >&2
  echo "  DATA_DIR=${DATA_DIR} PHOTO_MASK_THRESHOLD=${PHOTO_MASK_THRESHOLD} \\" >&2
  echo "  PHOTO_MASK_MODE=${PHOTO_MASK_MODE} SAMPLE_EVERY=${SAMPLE_EVERY} \\" >&2
  echo "  bash scripts/generate_mipnerf_photo_masks.sh" >&2
  exit 1
fi

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate multinerf
fi

LOG_DIR="${LOG_DIR:-${REPO}/sweep_logs}"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "Sweep dataset:        ${ds_tag}  (${DATA_DIR})"
echo "Sweep LAMBDA_DEPTH:   ${LAMBDAS[*]}"
echo "Photo mask:           ${PHOTO_MASK_MODE} thresh=${PHOTO_MASK_THRESHOLD}  (${MASK_DIR})"
echo "sample_every:         ${SAMPLE_EVERY}  factor: ${CONFIG_FACTOR}"
echo "Total runs:           ${#LAMBDAS[@]}"
echo "Logs:                 ${LOG_DIR}/sweep_mipnerf_${STAMP}_<exp>.log"

cd "${MIPNERF_DIR}"

declare -a RESULTS=()
for LAMBDA in "${LAMBDAS[@]}"; do
  EXP="${ds_tag}_lambda${LAMBDA}_${PHOTO_MASK_MODE}${thresh_tag}"
  CKPT_DIR="${DATA_DIR}/logs/checkpoints_${EXP}_${MAX_STEPS}"
  LOG_FILE="${LOG_DIR}/sweep_mipnerf_${STAMP}_${EXP}.log"

  echo ""
  echo "============================================================"
  echo ">>> Sweep run: ${EXP}"
  echo ">>> CKPT_DIR:  ${CKPT_DIR}"
  echo ">>> Log:       ${LOG_FILE}"
  echo "============================================================"

  if python -m train \
      --gin_configs=configs/360.gin \
      --gin_bindings="Config.data_dir='${DATA_DIR}'" \
      --gin_bindings="Config.checkpoint_dir='${CKPT_DIR}'" \
      --gin_bindings="Config.max_steps=${MAX_STEPS}" \
      --gin_bindings="Config.checkpoint_every=${CHECKPOINT_EVERY}" \
      --gin_bindings="Config.batch_size=${BATCH_SIZE}" \
      --gin_bindings="Config.factor=${CONFIG_FACTOR}" \
      --gin_bindings="Config.compute_disp_metrics=True" \
      --gin_bindings="Config.depth_sup_type='${DEPTH_SUP_TYPE}'" \
      --gin_bindings="Config.depth_keep_ratio=${DEPTH_KEEP_RATIO}" \
      --gin_bindings="Config.fixed_photo_mask_dir='${MASK_DIR}'" \
      --gin_bindings="Config.depth_loss_type='${DEPTH_LOSS_TYPE}'" \
      --gin_bindings="Config.lambda_depth=${LAMBDA}" \
      --gin_bindings="Config.sample_every=${SAMPLE_EVERY}" \
      --gin_bindings="Config.auto_adjust_near_far=True" \
      --gin_bindings="Config.near=0.2" \
      --gin_bindings="Config.far=1000000.0" \
      --gin_bindings="Model.opaque_background=True" \
      --gin_bindings="Model.raydist_fn=@jnp.reciprocal" \
      --gin_bindings="NerfMLP.disable_density_normals=True" \
      --gin_bindings="NerfMLP.net_depth=8" \
      --gin_bindings="NerfMLP.net_width=1024" \
      --gin_bindings="NerfMLP.warp_fn=@coord.contract" \
      --gin_bindings="PropMLP.disable_density_normals=True" \
      --gin_bindings="PropMLP.disable_rgb=True" \
      --gin_bindings="PropMLP.net_depth=4" \
      --gin_bindings="PropMLP.net_width=256" \
      --gin_bindings="PropMLP.warp_fn=@coord.contract" \
      --logtostderr 2>&1 | tee "${LOG_FILE}"; then
    RESULTS+=("OK    ${EXP}")
  else
    RESULTS+=("FAIL  ${EXP}  (see ${LOG_FILE})")
  fi
done

echo ""
echo "==================== sweep summary ===================="
for r in "${RESULTS[@]}"; do echo "  ${r}"; done
echo "======================================================="
