#!/usr/bin/env bash
# Scale-shift align DA2 .npy depth to COLMAP sparse depth (Mip-NeRF 360 / bicycle).
#
# Uses the COLMAP model already in the dataset (sparse/0), not KITTI LiDAR.
#
# Prereq: depths_da2_npy/ from run_da2_save_npy.py
#
# Usage:
#   SCENE_DIR=data/mip360_sparse/bicycle bash scripts/data_prep/align_da2_mip360_colmap.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPTS_ROOT}/.." && pwd)"

SCENE_DIR="${SCENE_DIR:-${PROJECT_ROOT}/data/mip360_sparse/bicycle}"
PYTHON="${PYTHON:-${HOME}/miniconda3/envs/cs231n/bin/python}"
MAX_DEPTH="${MAX_DEPTH:-80}"

DA2_NPY="${SCENE_DIR}/depths_da2_npy"
COLMAP_REF="${SCENE_DIR}/depths_colmap"
DA2_OUT="${SCENE_DIR}/depths_da2"
COLMAP_MODEL="${SCENE_DIR}/sparse/0"

if [[ ! -d "${DA2_NPY}" ]]; then
  echo "Missing ${DA2_NPY}. Run run_da2_save_npy.py on ${SCENE_DIR}/images first." >&2
  exit 1
fi
if [[ ! -d "${COLMAP_MODEL}" ]]; then
  echo "Missing COLMAP model: ${COLMAP_MODEL}" >&2
  exit 1
fi

echo "Exporting COLMAP sparse depth -> ${COLMAP_REF}"
"${PYTHON}" "${SCRIPT_DIR}/export_colmap_depths.py" --scene-dir "${SCENE_DIR}"

echo "Aligning DA2 to COLMAP reference -> ${DA2_OUT}"
"${PYTHON}" "${PROJECT_ROOT}/Depth-Anything-V2/align_da2_to_kitti.py" \
  --da2-npy-dir "${DA2_NPY}" \
  --ref-depth-dir "${COLMAP_REF}" \
  --out-dir "${DA2_OUT}" \
  --max-depth "${MAX_DEPTH}"

echo "Done. depths_da2 ready for prepare_mip360_sparse_scene.sh"
