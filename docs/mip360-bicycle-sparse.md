# Mip-NeRF 360 bicycle (sparse) — lambda / threshold pipeline

Run the **same** DA2 + photometric-mask sweep as KITTI seq02 (`λ ∈ {0, 0.05, 0.1, 0.15}`, `low` @ `0.14`, 50k steps) on the **existing sparse** bicycle pack from [Kaggle](https://www.kaggle.com/datasets/thnhdg/testing).

## 1. Layout on disk

Put the sparse bicycle scene here (names must match COLMAP `images.bin`):

```text
data/mip360_sparse/bicycle/
  images/           # sparse RGB frames
  sparse/0/         # COLMAP from the Kaggle pack (used for depth reference)
  depths_da2_npy/   # raw DA2
  depths_colmap/    # sparse depth rasterized from sparse/0 (align reference)
  depths_da2/       # DA2 scale-shift aligned to depths_colmap
```

No KITTI LiDAR and no `depths_gt/` folder. Alignment reference is **COLMAP** in `depths_colmap/` (from `sparse/0`); MipNeRF DA2 training reads `depths_da2/` only.

### DA2 + COLMAP align

1. Raw DA2 — [depth-anything-v2.md](depth-anything-v2.md) (`run_da2_save_npy.py` on `images/`).

2. One-shot COLMAP align (from repo root):

```bash
export PROJECT=/path/to/cs231n-final-project
SCENE_DIR="$PROJECT/data/mip360_sparse/bicycle" \
  bash "$PROJECT/scripts/align_da2_mip360_colmap.sh"
```

Or step by step:

```bash
$CS231N="$HOME/miniconda3/envs/cs231n/bin/python"

$CS231N "$PROJECT/scripts/export_colmap_depths.py" --scene-dir "$PROJECT/data/mip360_sparse/bicycle"

$CS231N "$PROJECT/Depth-Anything-V2/align_da2_to_kitti.py" \
  --da2-npy-dir "$PROJECT/data/mip360_sparse/bicycle/depths_da2_npy" \
  --ref-depth-dir "$PROJECT/data/mip360_sparse/bicycle/depths_colmap" \
  --out-dir "$PROJECT/data/mip360_sparse/bicycle/depths_da2" \
  --max-depth 80
```

## 2. Prepare nerfstudio + splits

```bash
cd /path/to/cs231n-final-project
SCENE=bicycle SCENE_DIR=data/mip360_sparse/bicycle bash scripts/prepare_mip360_sparse_scene.sh
```

Creates:

- `data/nerfstudio/bicycle_sparse/transforms.json` (KITTI-style train/val/test holdout every 10)
- `data/nerfstudio/bicycle_sparse_da2/` (depth paths for splatfacto-da2)

Use **`sample_every=1`** in MipNeRF (data is already sparse on disk).

## 3. MipNeRF-360 sweep (local GPU)

```bash
# RGB-only base for mask generation
SCENE=bicycle SAMPLE_EVERY=1 MAX_STEPS=50000 bash scripts/train_mipnerf_sparse_rgbonly.sh

PHOTO_MASK_THRESHOLD=0.14 PHOTO_MASK_MODE=low SAMPLE_EVERY=1 \
  bash scripts/generate_mipnerf_photo_masks.sh

SWEEP_LAMBDAS="0.0 0.05 0.1 0.15" PHOTO_MASK_THRESHOLD=0.14 PHOTO_MASK_MODE=low \
  SCENE=bicycle SAMPLE_EVERY=1 bash scripts/train_mipnerf_sparse_da2_sweep.sh
```

Checkpoints: `data/mip360_sparse/bicycle/logs/checkpoints_bicycle_sparse_lambda{λ}_low014_50000/`

Eval (after training): use `outdoor-nerf-depth/utils/eval.py` on `test_preds_*` vs `images/`, or render from checkpoint.

## 4. Splatfacto-da2 on Modal

**Do not** `modal volume put` `bicycle_sparse_da2` from your Mac (symlinks break on Modal). Delete it on the volume if present, then let `::sweep` build `_da2` in the container.

Upload once:

```bash
modal volume put kitti-nerf-data data/mip360_sparse/bicycle mip360_sparse/bicycle
modal volume put kitti-nerf-data data/nerfstudio/bicycle_sparse nerfstudio/bicycle_sparse
```

Nomask λ sweep:

```bash
modal run --detach modal_train_splatfacto.py::sweep \
  --dataset-family mip360 \
  --mip360-scene bicycle \
  --lambdas "0.0 0.05 0.1 0.15" \
  --max-num-iterations 50000
```

Mask @ 0.14 + retrain (same as KITTI):

```bash
modal run --detach modal_train_splatfacto.py::sweep_lambda_threshold \
  --dataset-family mip360 \
  --mip360-scene bicycle \
  --lambdas "0.0 0.05 0.1 0.15" \
  --thresholds "0.14" \
  --photo-mask-mode low \
  --max-num-iterations 50000
```

Eval per λ:

```bash
# All λ @ low 0.14 in parallel (after sweep_lambda_threshold)
modal run modal_train_splatfacto.py::sweep_eval \
  --dataset-family mip360 --mip360-scene bicycle \
  --lambdas "0.0 0.05 0.1 0.15" \
  --photo-mask-mode low --photo-mask-threshold 0.14 --masked

# Single run
modal run modal_train_splatfacto.py::run_eval \
  --dataset-family mip360 --mip360-scene bicycle \
  --lambda-depth 0.05 --max-num-iterations 50000 \
  --photo-mask-mode low --photo-mask-threshold 0.14 --masked
```

Experiment names: `bicycle_sparse_da2_lambda{λ}_nomask_50000` or `..._low014_50000`

## 5. Local splatfacto (no Modal)

```bash
DATASET_FAMILY=mip360 MIP360_SCENE=bicycle LAMBDA_DEPTH=0.05 \
  bash scripts/train_splatfacto_kitti_sparse_da2.sh
```

## 6. One-shot helper

```bash
# MipNeRF only (after prep + masks):
bash scripts/run_bicycle_lambda_threshold_pipeline.sh

# With optional steps:
RUN_PREPARE=1 RUN_MIPNERF_RGB=1 RUN_MIPNERF_MASKS=1 bash scripts/run_bicycle_lambda_threshold_pipeline.sh
```

## Notes

- **No `depths_gt/`** — bicycle uses `depths_colmap/` (align only) and `depths_da2/` (training). MipNeRF loads `depths_da2` via `Config.depth_sup_type='da2'`; LiDAR GT is optional in the loader.
- `prepare_mip360_sparse_scene.sh` auto-removes a stale `depths_gt` **symlink**. To remove a real directory from an old pseudo-GT export:
  ```bash
  ls -la data/mip360_sparse/bicycle/depths_gt   # symlink -> depths_da2, or old PNG folder
  rm data/mip360_sparse/bicycle/depths_gt         # symlink only
  # rm -rf data/mip360_sparse/bicycle/depths_gt   # if it is a directory you no longer need
  ```
- Split logic matches KITTI sparse (test every 10th frame in sorted list), not the paper’s `llffhold=8`.
- If `images_4/` exists, MipNeRF uses `Config.factor=4`; otherwise full-res `images/`.
- Garden: set `SCENE=garden` / `--mip360-scene garden` and the same paths under `data/mip360_sparse/garden`.
