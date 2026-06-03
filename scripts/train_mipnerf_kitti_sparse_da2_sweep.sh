#!/usr/bin/env bash
# Sweep MipNeRF-360 (masked DA2) over Config.lambda_depth on the sparse-every-2 KITTI
# nerfstudio sequence. Each run gets its own CKPT_DIR whose name encodes:
#   <dataset slug>_lambda<value>_<mode><threshold compact>_<max steps>
# e.g. checkpoints_kitti_seq02_0034_lambda0.1_low014_50000/
#
# Prerequisites (the script aborts before training if any are missing):
#   - multinerf conda env installed
#   - DATA_DIR exists with depths_<DEPTH_SUP_TYPE>/
#   - The fixed photometric mask folder produced by Masks.txt exists, i.e. the
#     PHOTO_MASK_MODE / PHOTO_MASK_THRESHOLD combination below must already have masks.
#
# Quick start (defaults: lambda in {0.05, 0.1, 0.15} on seq02, low/0.14 masks):
#   bash scripts/train_mipnerf_kitti_sparse_da2_sweep.sh
#
# Override the swept values without editing the file:
#   SWEEP_LAMBDAS="0.0 0.05 0.1 0.2" bash scripts/train_mipnerf_kitti_sparse_da2_sweep.sh
#
# Other pass-through env vars: DEPTH_SUP_TYPE, DEPTH_LOSS_TYPE, DEPTH_KEEP_RATIO,
# SAMPLE_EVERY, MAX_STEPS, CHECKPOINT_EVERY, BATCH_SIZE, KITTI_DIR_NAME,
# PHOTO_MASK_THRESHOLD, PHOTO_MASK_MODE.
#
# pipefail is on, set -e is intentionally off: one failed run does not abort the
# whole sweep; failures are collected and reported in the final summary.

set -uo pipefail

REPO="${REPO:-/home/ubuntu/final_project}"
MIPNERF_DIR="${MIPNERF_DIR:-${REPO}/outdoor-nerf-depth/nerf-methods/mipnerf360}"
SPARSE_KITTI_ROOT="${SPARSE_KITTI_ROOT:-${REPO}/data/kitti/kitti_select_static_5seq}"

# ----------------------- sweep configuration -----------------------
LAMBDAS=(0.05 0.1 0.15)
if [[ -n "${SWEEP_LAMBDAS:-}" ]]; then
  read -r -a LAMBDAS <<< "${SWEEP_LAMBDAS}"
fi

# Photo-mask combination used for this sweep. Must match a directory that Masks.txt
# already produced under DATA_DIR. Encoded into the output folder name.
PHOTO_MASK_THRESHOLD="${PHOTO_MASK_THRESHOLD:-0.22}"
PHOTO_MASK_MODE="${PHOTO_MASK_MODE:-low}"   # low | high

# Training hyper-parameters (defaults match Train.txt)
DEPTH_SUP_TYPE="${DEPTH_SUP_TYPE:-da2}"
DEPTH_LOSS_TYPE="${DEPTH_LOSS_TYPE:-mse}"
DEPTH_KEEP_RATIO="${DEPTH_KEEP_RATIO:-0.0}"
SAMPLE_EVERY="${SAMPLE_EVERY:-2}"
MAX_STEPS="${MAX_STEPS:-50000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"

# Dataset selector (single dataset for now; bare folder name resolved under SPARSE_KITTI_ROOT)
KITTI_DIR_NAME="${KITTI_DIR_NAME:-KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt}"
DATA_DIR="${DATA_DIR:-${SPARSE_KITTI_ROOT}/${KITTI_DIR_NAME}}"
# -------------------------------------------------------------------

# Short dataset slug (e.g. kitti_seq02_0034) for nice output folder names.
ds_tag="$(basename "${DATA_DIR}")"
if [[ "${ds_tag}" =~ KITTISeq([0-9]+)_.*drive_([0-9]+)_sync ]]; then
  ds_tag="$(printf 'kitti_seq%02d_%s' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}")"
fi

# Compact threshold tag: 0.14 -> "014", 0.2 -> "02" (matches Masks.txt naming).
thresh_tag="${PHOTO_MASK_THRESHOLD//./}"

# Fixed-mask dir produced by Masks.txt for this (mode, threshold) combination.
MASK_DIR="${DATA_DIR}/photo_masks_rgbonly_${PHOTO_MASK_MODE}${thresh_tag}_sampleevery${SAMPLE_EVERY}"

# Sanity checks
if [[ ! -d "${MIPNERF_DIR}" ]]; then
  echo "Missing mipnerf360 dir: ${MIPNERF_DIR}" >&2; exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Missing dataset dir: ${DATA_DIR}" >&2; exit 1
fi
if [[ ! -d "${DATA_DIR}/depths_${DEPTH_SUP_TYPE}" ]]; then
  echo "Missing depth folder: ${DATA_DIR}/depths_${DEPTH_SUP_TYPE}" >&2; exit 1
fi
if [[ ! -d "${MASK_DIR}" ]]; then
  echo "Missing photo-mask dir: ${MASK_DIR}" >&2
  echo "Generate it first with Masks.txt using PHOTO_MASK_MODE=${PHOTO_MASK_MODE} and PHOTO_MASK_THRESHOLD=${PHOTO_MASK_THRESHOLD}." >&2
  exit 1
fi

# Activate conda env (matches Train.txt's first three lines).
# shellcheck disable=SC1091
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multinerf

LOG_DIR="${LOG_DIR:-${REPO}/sweep_logs}"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "Sweep dataset:        ${ds_tag}  (${DATA_DIR})"
echo "Sweep LAMBDA_DEPTH:   ${LAMBDAS[*]}"
echo "Photo mask:           ${PHOTO_MASK_MODE} thresh=${PHOTO_MASK_THRESHOLD}  (${MASK_DIR})"
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
