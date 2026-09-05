# End-to-End Workflow

The full pipeline for the headline result (`Splatfacto + DA-V2` on KITTISeq02 sparse-every-2, τ=0.18, λ=0.10) is **one Modal command** — `run_new_seq_experiments` (see "Reproduce the headline result" below). The numbered phases describe what that command does internally, and the manual entry points if you want to run a phase by itself.

## Phase 1 — Data preparation (`scripts/data_prep/`)

Convert raw KITTI or Mip-NeRF-360 scenes into a sparse, nerfstudio-shaped dataset.

| Step | Script | What it does |
|------|--------|--------------|
| 1.1 | `colmap_to_nerfstudio_transforms.py` | Build `transforms.json` from a COLMAP `sparse/0` model. |
| 1.2 | `make_nerfstudio_kitti_sparse.py` | Subsample dense → every-N sparse + write train/val/test split. |
| 1.3 | `patch_nerfstudio_splits.py` | Add holdout-every-10 KITTI-style splits. |
| 1.4 | `make_nerfstudio_kitti_depth.py` | Symlink `depths_da2/` and add `depth_file_path` to each frame. |
| 1.5 | `export_colmap_depths.py` / `align_da2_mip360_colmap.sh` | Mip-360-only — export COLMAP sparse depths, scale-shift align DA-V2. |
| 1.6 | `prepare_mip360_sparse_scene.sh` | One-shot wrapper for steps 1.1–1.5 on Mip-NeRF-360. |

For KITTI, DA-V2 inference is run beforehand inside `Depth-Anything-V2/` (`run_da2_save_npy.py`) and aligned to LiDAR via `align_da2_to_kitti.py`. See [depth-anything-v2.md](depth-anything-v2.md).

## Phase 2 — Stage-1 RGB-only baseline

Train a vanilla model for 50 000 iterations so its render error can be turned into a reliability mask. Triggered automatically by the Modal `sweep` / `run_new_seq_experiments` entry points; locally:

```bash
# Splatfacto (Nerfstudio)
bash scripts/train/train_splatfacto_kitti_sparse_da2.sh   # LAMBDA_DEPTH=0 by default
# Mip-NeRF-360 (JAX)
bash scripts/train/train_mipnerf_sparse_rgbonly.sh
```

## Phase 3 — Reliability mask generation (`scripts/masks/`)

Render each train view from the Stage-1 checkpoint, compute `e(u) = mean_c |Î_c(u) - I_c(u)|`, and threshold at τ to produce a fixed per-frame binary mask `M(u) = 1[e(u) < τ]`.

| Variant | Script |
|---------|--------|
| Splatfacto fixed photo mask | `generate_splatfacto_photo_masks.py` |
| Mip-NeRF fixed photo mask | `generate_mipnerf_photo_masks.sh` |
| Matched-ratio control masks (high-error / random, same pixel count as low018) | `generate_matched_ratio_masks.py` |
| Wire mask paths into `transforms.json` | `attach_nerfstudio_photo_masks.py` |

## Phase 4 — Stage-2 depth-supervised training

Retrain with the masked depth loss `L = L_rgb + λ * L_depth_masked`. Mask × depth-validity gating is applied per pixel.

```bash
# Splatfacto (single run)
PHOTO_MASK_DIR=<masks> LAMBDA_DEPTH=0.10 PHOTO_MASK_THRESHOLD=0.18 PHOTO_MASK_MODE=low \
  bash scripts/train/train_splatfacto_kitti_sparse_da2.sh

# Splatfacto λ-sweep (sequential, one log per run)
SWEEP_LAMBDAS="0.05 0.1 0.15" bash scripts/train/train_splatfacto_kitti_sparse_da2_sweep.sh

# Mip-NeRF τ × λ sweep
SWEEP_LAMBDAS="0.05 0.1 0.15" bash scripts/train/train_mipnerf_sparse_da2_sweep.sh
```

## Phase 5 — Evaluation (`scripts/eval/` + Modal `run_eval`)

`compute_mipnerf_metrics.py` produces PSNR / SSIM / LPIPS / RMSE from Mip-NeRF's `test_preds_50000/` directory. Splatfacto uses Modal `::run_eval` (and `::sweep_eval` for the whole grid in parallel) — see [modal-splatfacto.md](modal-splatfacto.md).

## Reproduce the headline result in one command

The Modal flow handles Phases 1–4 in a single call:

```bash
modal run --detach modal_train_splatfacto.py::run_new_seq_experiments \
  --kitti-seq-dir "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt"
```

This trains three experiments back-to-back: RGB-only baseline → global depth (λ=0.10) → masked depth (τ=0.18, λ=0.10). Evaluation:

```bash
modal run modal_train_splatfacto.py::run_eval \
  --kitti-seq-dir "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt" \
  --lambda-depth 0.10 --photo-mask-threshold 0.18 --masked
```
