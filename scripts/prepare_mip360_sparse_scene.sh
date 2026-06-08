#!/usr/bin/env bash
# Prepare an existing Mip-NeRF 360 *sparse* scene (e.g. bicycle) for the KITTI-style DA2 pipeline.
#
# Expected layout under SCENE_DIR (from Kaggle / 360_v2 sparse pack):
#   images/          RGB frames used by COLMAP
#   sparse/0/        COLMAP model (cameras.bin, images.bin, ...)
#   depths_da2_npy/  raw DA2 .npy (from run_da2_save_npy.py)
#   depths_colmap/   COLMAP sparse depth (align reference; scripts/export_colmap_depths.py)
#   depths_da2/      DA2 scale-shift aligned to depths_colmap
#
# No depths_gt/ — Mip-360 has no KITTI LiDAR. Align via depths_colmap only.
#
# Creates:
#   data/nerfstudio/${SCENE}_sparse/transforms.json  (+ KITTI-style splits)
#   data/nerfstudio/${SCENE}_sparse_da2/             (depth paths wired)
#
# Usage (from repo root):
#   SCENE=bicycle SCENE_DIR=/path/to/bicycle_sparse bash scripts/prepare_mip360_sparse_scene.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCENE="${SCENE:-bicycle}"
SCENE_DIR="${SCENE_DIR:-${PROJECT_ROOT}/data/mip360_sparse/${SCENE}}"
NERFSTUDIO_SRC="${NERFSTUDIO_SRC:-${PROJECT_ROOT}/data/nerfstudio/${SCENE}_sparse}"
NERFSTUDIO_DA2="${NERFSTUDIO_DA2:-${PROJECT_ROOT}/data/nerfstudio/${SCENE}_sparse_da2}"
DEPTH_SUP_TYPE="${DEPTH_SUP_TYPE:-da2}"
COLMAP_SUBDIR="${COLMAP_SUBDIR:-sparse/0}"
HOLD_EVERY="${HOLD_EVERY:-10}"

if [[ ! -d "${SCENE_DIR}" ]]; then
  echo "Scene directory does not exist: ${SCENE_DIR}" >&2
  echo "" >&2
  echo "Download the Kaggle sparse pack, then point SCENE_DIR at the bicycle folder:" >&2
  echo "  kaggle datasets download -d thnhdg/testing -p ~/Downloads" >&2
  echo "  unzip ~/Downloads/testing.zip -d ~/Downloads/testing" >&2
  echo "  # find bicycle inside the zip, then either:" >&2
  echo "  mkdir -p ${PROJECT_ROOT}/data/mip360_sparse" >&2
  echo "  cp -R /path/from/zip/bicycle ${PROJECT_ROOT}/data/mip360_sparse/bicycle" >&2
  echo "  # or run with the real path:" >&2
  echo "  SCENE=bicycle SCENE_DIR=/path/from/zip/bicycle bash scripts/prepare_mip360_sparse_scene.sh" >&2
  exit 1
fi

# Some 360 packs use images_4/ for training resolution; COLMAP still references those names.
if [[ ! -d "${SCENE_DIR}/images" ]]; then
  if [[ -d "${SCENE_DIR}/images_4" ]]; then
    echo "Linking images -> images_4 (360 downsampled training set)"
    ln -sfn "images_4" "${SCENE_DIR}/images"
  elif [[ -d "${SCENE_DIR}/images_2" ]]; then
    echo "Linking images -> images_2"
    ln -sfn "images_2" "${SCENE_DIR}/images"
  else
    echo "Missing images/ (or images_4/) under: ${SCENE_DIR}" >&2
    echo "Contents:" >&2
    ls -la "${SCENE_DIR}" >&2 || true
    exit 1
  fi
fi

if [[ ! -d "${SCENE_DIR}/${COLMAP_SUBDIR}" ]]; then
  if [[ -d "${SCENE_DIR}/colmap/sparse/0" ]]; then
    COLMAP_SUBDIR="colmap/sparse/0"
    mkdir -p "${SCENE_DIR}/sparse"
    ln -sfn "../colmap/sparse/0" "${SCENE_DIR}/sparse/0"
  else
    echo "Missing COLMAP at ${SCENE_DIR}/${COLMAP_SUBDIR} or colmap/sparse/0" >&2
    exit 1
  fi
fi
if [[ ! -d "${SCENE_DIR}/depths_${DEPTH_SUP_TYPE}" ]]; then
  echo "Missing depths_${DEPTH_SUP_TYPE}/ under ${SCENE_DIR}" >&2
  echo "Run DA2 + COLMAP align: SCENE_DIR=${SCENE_DIR} bash scripts/align_da2_mip360_colmap.sh" >&2
  exit 1
fi

# Older prep runs symlinked depths_gt -> depths_da2 for the MipNeRF loader; that is no longer needed.
if [[ -e "${SCENE_DIR}/depths_gt" ]]; then
  if [[ -L "${SCENE_DIR}/depths_gt" ]]; then
    echo "Removing stale depths_gt symlink (Mip-360 uses depths_colmap + depths_da2 only)"
    rm "${SCENE_DIR}/depths_gt"
  else
    echo "Note: ${SCENE_DIR}/depths_gt exists but is not used for Mip-360 DA2 training." >&2
    echo "  Safe to remove if it was pseudo-GT from an old export: rm -rf ${SCENE_DIR}/depths_gt" >&2
  fi
fi

# Export transforms.json from existing COLMAP (360 uses sparse/0, not colmap/sparse/0).
if [[ ! -f "${NERFSTUDIO_SRC}/transforms.json" ]]; then
  mkdir -p "${NERFSTUDIO_SRC}"
  # colmap-model-path is relative to output-dir; link scene COLMAP in place.
  mkdir -p "${NERFSTUDIO_SRC}/sparse"
  if [[ ! -e "${NERFSTUDIO_SRC}/sparse/0" ]]; then
    ln -sfn "$(cd "${SCENE_DIR}/sparse/0" && pwd)" "${NERFSTUDIO_SRC}/sparse/0"
  fi
  if [[ ! -e "${NERFSTUDIO_SRC}/images" ]]; then
    ln -sfn "$(cd "${SCENE_DIR}/images" && pwd)" "${NERFSTUDIO_SRC}/images"
  fi
  if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "${HOME}/miniconda3/envs/cs231n/bin/python" ]]; then
      PYTHON="${HOME}/miniconda3/envs/cs231n/bin/python"
    else
      PYTHON="python3"
    fi
  fi
  if command -v ns-process-data >/dev/null 2>&1; then
    echo "Building transforms.json via ns-process-data (skip COLMAP)..."
    ns-process-data images \
      --data "${SCENE_DIR}" \
      --output-dir "${NERFSTUDIO_SRC}" \
      --skip-colmap \
      --skip-image-processing \
      --colmap-model-path sparse/0 \
      --num-downscales 0
  else
    echo "ns-process-data not found; using colmap_to_nerfstudio_transforms.py (${PYTHON})..."
    "${PYTHON}" "${SCRIPT_DIR}/colmap_to_nerfstudio_transforms.py" \
      --output-dir "${NERFSTUDIO_SRC}"
  fi
fi

PYTHON="${PYTHON:-python3}"
"${PYTHON}" "${SCRIPT_DIR}/patch_nerfstudio_splits.py" \
  --data-dir "${NERFSTUDIO_SRC}" \
  --hold-every "${HOLD_EVERY}"

"${PYTHON}" "${SCRIPT_DIR}/make_nerfstudio_kitti_depth.py" \
  --src "${NERFSTUDIO_SRC}" \
  --dst "${NERFSTUDIO_DA2}" \
  --depth-dir "${SCENE_DIR}/depths_${DEPTH_SUP_TYPE}" \
  --depth-sup-type "${DEPTH_SUP_TYPE}" \
  --skip-missing-frames \
  --overwrite

echo ""
echo "Ready for training:"
echo "  MipNeRF DATA_DIR=${SCENE_DIR}"
echo "  Splatfacto DATA_DIR=${NERFSTUDIO_DA2}"
echo "  DS_TAG=${SCENE}_sparse"
