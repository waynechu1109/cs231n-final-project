#!/usr/bin/env bash

export MAX_JOBS=1
export TORCHDYNAMO_DISABLE=1

# Sweep wrapper around scripts/train_splatfacto_kitti_sparse_da2.sh.
#
# Runs the splatfacto-da2 training once per swept value. Each run gets a distinct
# experiment name (the base script encodes LAMBDA_DEPTH into EXP_NAME), so results
# land in separate outputs/ folders, and each run is also logged separately.
#
# Quick start (uses the LAMBDAS below = 0.1 0.15):
#   bash scripts/train_splatfacto_kitti_sparse_da2_sweep.sh
#
# Override the swept values without editing this file:
#   SWEEP_LAMBDAS="0.05 0.1 0.15 0.2" bash scripts/train_splatfacto_kitti_sparse_da2_sweep.sh
#
# Other knobs pass straight through to the base script via env vars, e.g.:
#   DEPTH_LOSS_TYPE=l1 MAX_NUM_ITERATIONS=30000 bash scripts/train_splatfacto_kitti_sparse_da2_sweep.sh
#
# Note: pipefail is on but set -e is intentionally off, so one failed run does not
# abort the whole sweep; failures are collected and reported in the final summary.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/train_splatfacto_kitti_sparse_da2.sh"

# ----------------------- sweep configuration -----------------------
# Root that bare dataset names below are resolved against.
SPARSE_KITTI_ROOT="${SPARSE_KITTI_ROOT:-${PROJECT_ROOT}/data/kitti/kitti_select_static_5seq_sparse_every2}"

# Training datasets to sweep. Each entry is passed to the base script as KITTI_SEQ_DIR,
# which then derives the nerfstudio DATA_DIR and experiment name automatically (so each
# dataset lands in its own outputs/ folder). An entry may be either a bare KITTI sequence
# folder name (resolved under SPARSE_KITTI_ROOT) or a full/relative path containing a '/'.
# NOTE: each dataset's sparse nerfstudio + depths must already exist (only seq02 is prebuilt).
KITTI_SEQ_DIRS=(
  "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt"
)
# Allow overriding from the environment: SWEEP_SEQ_DIRS="SeqA SeqB /abs/path/SeqC"
if [[ -n "${SWEEP_SEQ_DIRS:-}" ]]; then
  read -r -a KITTI_SEQ_DIRS <<< "${SWEEP_SEQ_DIRS}"
fi

# Values of --pipeline.model.lambda-depth to try, one training run each.
LAMBDAS=(0.1 0.15)
# Allow overriding from the environment: SWEEP_LAMBDAS="0.05 0.1 0.2"
if [[ -n "${SWEEP_LAMBDAS:-}" ]]; then
  read -r -a LAMBDAS <<< "${SWEEP_LAMBDAS}"
fi
# Total runs = (#datasets) x (#lambdas), executed sequentially.
# -------------------------------------------------------------------

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Base training script not found: ${BASE_SCRIPT}" >&2
  exit 1
fi

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/sweep_logs}"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "Sweep datasets:     ${KITTI_SEQ_DIRS[*]}"
echo "Sweep LAMBDA_DEPTH: ${LAMBDAS[*]}"
echo "Total runs:         $(( ${#KITTI_SEQ_DIRS[@]} * ${#LAMBDAS[@]} ))"
echo "Logs:               ${LOG_DIR}/sweep_${STAMP}_<dataset>_lambda<val>.log"

declare -a RESULTS=()
for DS in "${KITTI_SEQ_DIRS[@]}"; do
  # Resolve to a full KITTI_SEQ_DIR: bare name -> under SPARSE_KITTI_ROOT; path -> use as-is.
  if [[ "${DS}" == /* || "${DS}" == */* ]]; then
    seq_dir="${DS}"
  else
    seq_dir="${SPARSE_KITTI_ROOT}/${DS}"
  fi

  # Short dataset tag for log/summary (mirrors the base script's slug).
  ds_tag="$(basename "${seq_dir}")"
  if [[ "${ds_tag}" =~ KITTISeq([0-9]+)_.*drive_([0-9]+)_sync ]]; then
    ds_tag="$(printf 'kitti_seq%02d_%s' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}")"
  fi

  for LAMBDA in "${LAMBDAS[@]}"; do
    log_file="${LOG_DIR}/sweep_${STAMP}_${ds_tag}_lambda${LAMBDA}.log"
    echo ""
    echo "============================================================"
    echo ">>> Sweep run: dataset=${ds_tag}  LAMBDA_DEPTH=${LAMBDA}"
    echo ">>> KITTI_SEQ_DIR=${seq_dir}"
    echo ">>> Log: ${log_file}"
    echo "============================================================"

    if KITTI_SEQ_DIR="${seq_dir}" LAMBDA_DEPTH="${LAMBDA}" bash "${BASE_SCRIPT}" 2>&1 | tee "${log_file}"; then
      RESULTS+=("OK    ${ds_tag}  lambda=${LAMBDA}")
    else
      RESULTS+=("FAIL  ${ds_tag}  lambda=${LAMBDA}  (see ${log_file})")
    fi
  done
done

echo ""
echo "==================== sweep summary ===================="
for r in "${RESULTS[@]}"; do
  echo "  ${r}"
done
echo "======================================================="
