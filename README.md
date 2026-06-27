# Reliability-Aware Monocular Depth Supervision for Sparse-View Neural Reconstruction

**CS231N Final Project — Stanford University**
Wayne Chu · Yashasvini Gopalan · Changju Yuan

---

## Overview

Reconstructing outdoor driving scenes from sparse views is difficult due to the narrow forward-facing trajectory of the camera motion and limited multi-view overlap. Although monocular depth estimators can provide dense geometric priors, their predictions are noisy and not consistently reliable over the image regions.

This project focuses on **photometric-masked monocular depth supervision** for sparse-view outdoor scene reconstruction. We leverage [Depth Anything V2 (DA-V2)](https://github.com/DepthAnything/Depth-Anything-V2) as a dense monocular depth prior, calibrate its predictions to metric depth with scale-shift fitting, and make use of depth supervision selectively with photometric masks trained on an RGB-only baseline model. We evaluate on two representative scene representations: **Mip-NeRF-360** and **Splatfacto (3DGS)** .

### Key Results

| Setting | Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | RMSE ↓ |
|---|---|---|---|---|---|
| KITTISeq02 every2 | Splatfacto (RGB-only) | 14.903 | 0.433 | 0.446 | 0.542 |
| KITTISeq02 every2 | **Splatfacto + DA-V2** (τ=0.18, λ=0.10) | **15.932** | **0.477** | **0.408** | **0.100** |
| KITTISeq02 every2 | Mip-NeRF-360 (RGB-only) | 20.378 | **0.601** | **0.409** | 2.703 |
| KITTISeq02 every2 | Mip-NeRF-360 + DA-V2 (τ=1.0, λ=0.15) | **20.607** | 0.595 | 0.412 | 3.580 |

> Splatfacto benefits strongly from masked depth supervision (+1.03 dB PSNR, RMSE drops from 0.542 → 0.100). Mip-NeRF-360 shows only marginal RGB gains without geometry improvement.

---

## Method

### Monocular Depth Prior Alignment

DA-V2 predicts relative depth. We align each prediction to metric depth via per-image least-squares scale-shift fitting:

$$s^{*}, t^{*} = \arg\min_{s,t} \sum_{u \in \Omega} \left(s\, d_m(u) + t - d_r(u)\right)^{2}$$

For KITTI, reference depth comes from projected LiDAR points. For Mip-NeRF-360 Bicycle, sparse COLMAP points are used. Aligned depth is clipped to 80 m for KITTI.

### Photometric-Masked Depth Supervision

Instead of applying the depth loss everywhere, we only supervise the pixels where the RGB-only baseline is already reliable. A pre-trained RGB-only model is used to compute a per-pixel photometric error map:

$$e(u) = \frac{1}{3} \sum_{c \in \lbrace R,G,B \rbrace} \left| \hat{I}_c(u) - I_c(u) \right|$$

Pixels with $e(u) < \tau$ form the binary reliability mask $M(u)$. This mask is fixed for the duration of depth-supervised training.

### Training Pipeline

1. **Stage 1** — Train RGB-only baseline for 50 000 iterations.
2. **Stage 2** — Render all training views; compute per-pixel photometric error; generate fixed masks at threshold τ.
3. **Stage 3** — Retrain with depth supervision: $\mathcal{L} = \mathcal{L}_\text{rgb} + \lambda_\text{depth}\,\mathcal{L}_\text{depth}$, where $\mathcal{L}_\text{depth}$ is MSE gated by $M_\text{eff}(u) = M(u) \land D(u)$ ($D$ = depth-validity mask).

We sweep τ ∈ {0.14, 0.16, 0.18, 0.20, 0.22, 1.0} and λ ∈ {0.05, 0.10, 0.15}.

### Matched-Ratio Ablation

To test whether the benefit of low-error masking comes from **selecting reliable pixels** vs simply **using fewer pixels**, we construct two control masks that match the exact pixel budget of `low018` (i.e. the same number of supervised pixels per frame):

- **High-error matched** — selects the *k* valid-depth pixels with the **highest** photometric error per frame.
- **Random matched** — selects *k* valid-depth pixels **uniformly at random** per frame (three seeds for stability).

where *k* = |{u : valid_depth(u) ∧ e(u) < 0.18}| per frame, computed from the RGB-only baseline.

All ablation runs use λ = 0.10, τ = 0.18 on KITTISeq02 every-2.

---

## Repository Layout

```text
.
├── README.md
├── modal_train_splatfacto.py             # Modal entry: splatfacto-da2 training, eval, sweeps
├── modal_train_mipnerf.py                # Modal entry: Mip-NeRF-360 (JAX) training, eval
│
├── scripts/                              # All local helper scripts, grouped by phase
│   ├── data_prep/                        # COLMAP → nerfstudio, KITTI sparse, DA2 align
│   │   ├── colmap_to_nerfstudio_transforms.py
│   │   ├── make_nerfstudio_kitti_depth.py
│   │   ├── make_nerfstudio_kitti_sparse.py
│   │   ├── patch_nerfstudio_splits.py
│   │   ├── prepare_mip360_sparse_scene.sh
│   │   ├── export_colmap_depths.py
│   │   └── align_da2_mip360_colmap.sh
│   ├── masks/                            # Reliability mask generation (photometric / matched-ratio)
│   │   ├── generate_splatfacto_photo_masks.py
│   │   ├── generate_matched_ratio_masks.py
│   │   ├── generate_mipnerf_photo_masks.sh
│   │   └── attach_nerfstudio_photo_masks.py
│   ├── train/                            # Splatfacto + Mip-NeRF training shells (single + sweep)
│   │   ├── train_splatfacto_kitti_da2.sh
│   │   ├── train_splatfacto_kitti_sparse_da2.sh
│   │   ├── train_splatfacto_kitti_sparse_da2_sweep.sh
│   │   ├── train_mipnerf_sparse_rgbonly.sh
│   │   ├── train_mipnerf_sparse_da2_sweep.sh
│   │   ├── train_mipnerf_kitti_sparse_da2_sweep.sh
│   │   ├── run_bicycle_lambda_threshold_pipeline.sh
│   │   └── download_bicycle_splatfacto_ckpts.sh
│   └── eval/
│       └── compute_mipnerf_metrics.py
│
├── outdoor-nerf-depth/                   # Mip-NeRF-360 backbone (vendored)
│   └── nerf-methods/mipnerf360/
│       ├── configs/360.gin
│       └── internal/                     # image.py is patched (photo-mask loader)
├── nerfstudio/                           # Splatfacto backbone (vendored)
├── Depth-Anything-V2/                    # DA-V2 inference + scale-shift align (vendored)
│
├── data/                                 # Datasets (gitignored)
│   ├── kitti/kitti_select_static_5seq/
│   ├── kitti/kitti_select_static_5seq_sparse_every2/
│   └── nerfstudio/
│
├── docs/                                 # Per-topic guides (see "Documentation" below)
├── paper_plots/                          # Figure-assembly scripts (gitignored)
│   ├── pull_figure/                      # 6-panel pipeline figure
│   └── splatfacto_plot/                  # threshold × lambda render grid
└── local_outputs/                        # Locally downloaded Modal outputs (gitignored)
```

---

## End-to-End Workflow

The full pipeline for the headline result (`Splatfacto + DA-V2` on KITTISeq02 sparse-every-2,
τ=0.18, λ=0.10) is **one Modal command** — `run_new_seq_experiments` (see Step 4 below).
The numbered phases describe what that command does internally, and the manual entry points if
you want to run a phase by itself.

### Phase 1 — Data preparation (`scripts/data_prep/`)

Convert raw KITTI or Mip-NeRF-360 scenes into a sparse, nerfstudio-shaped dataset.

| Step | Script | What it does |
|------|--------|--------------|
| 1.1 | `colmap_to_nerfstudio_transforms.py` | Build `transforms.json` from a COLMAP `sparse/0` model. |
| 1.2 | `make_nerfstudio_kitti_sparse.py` | Subsample dense → every-N sparse + write train/val/test split. |
| 1.3 | `patch_nerfstudio_splits.py` | Add holdout-every-10 KITTI-style splits. |
| 1.4 | `make_nerfstudio_kitti_depth.py` | Symlink `depths_da2/` and add `depth_file_path` to each frame. |
| 1.5 | `export_colmap_depths.py` / `align_da2_mip360_colmap.sh` | Mip-360-only — export COLMAP sparse depths, scale-shift align DA-V2. |
| 1.6 | `prepare_mip360_sparse_scene.sh` | One-shot wrapper for steps 1.1–1.5 on Mip-NeRF-360. |

For KITTI, DA-V2 inference is run beforehand inside `Depth-Anything-V2/` (`run_da2_save_npy.py`)
and aligned to LiDAR via `align_da2_to_kitti.py`. See [docs/depth-anything-v2.md](docs/depth-anything-v2.md).

### Phase 2 — Stage-1 RGB-only baseline

Train a vanilla model for 50 000 iterations so its render error can be turned into a
reliability mask. Triggered automatically by the Modal `sweep` / `run_new_seq_experiments`
entry points; locally:

```bash
# Splatfacto (Nerfstudio)
bash scripts/train/train_splatfacto_kitti_sparse_da2.sh   # LAMBDA_DEPTH=0 by default
# Mip-NeRF-360 (JAX)
bash scripts/train/train_mipnerf_sparse_rgbonly.sh
```

### Phase 3 — Reliability mask generation (`scripts/masks/`)

Render each train view from the Stage-1 checkpoint, compute `e(u) = mean_c |Î_c(u) - I_c(u)|`,
and threshold at τ to produce a fixed per-frame binary mask `M(u) = 1[e(u) < τ]`.

| Variant | Script |
|---------|--------|
| Splatfacto fixed photo mask | `generate_splatfacto_photo_masks.py` |
| Mip-NeRF fixed photo mask | `generate_mipnerf_photo_masks.sh` |
| Matched-ratio control masks (high-error / random, same pixel count as low018) | `generate_matched_ratio_masks.py` |
| Wire mask paths into `transforms.json` | `attach_nerfstudio_photo_masks.py` |

### Phase 4 — Stage-2 depth-supervised training

Retrain with the masked depth loss `L = L_rgb + λ * L_depth_masked`. Mask × depth-validity
gating is applied per pixel.

```bash
# Splatfacto (single run)
PHOTO_MASK_DIR=<masks> LAMBDA_DEPTH=0.10 PHOTO_MASK_THRESHOLD=0.18 PHOTO_MASK_MODE=low \
  bash scripts/train/train_splatfacto_kitti_sparse_da2.sh

# Splatfacto λ-sweep (sequential, one log per run)
SWEEP_LAMBDAS="0.05 0.1 0.15" bash scripts/train/train_splatfacto_kitti_sparse_da2_sweep.sh

# Mip-NeRF τ × λ sweep
SWEEP_LAMBDAS="0.05 0.1 0.15" bash scripts/train/train_mipnerf_sparse_da2_sweep.sh
```

### Phase 5 — Evaluation (`scripts/eval/` + Modal `run_eval`)

`compute_mipnerf_metrics.py` produces PSNR / SSIM / LPIPS / RMSE from Mip-NeRF's
`test_preds_50000/` directory. Splatfacto uses Modal `::run_eval` (and `::sweep_eval`
for the whole grid in parallel) — see "Modal Cloud Training" below.

### Reproduce the headline result in one command

The Modal flow handles Phases 1–4 in a single call:

```bash
modal run --detach modal_train_splatfacto.py::run_new_seq_experiments \
  --kitti-seq-dir "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt"
```

This trains three experiments back-to-back: RGB-only baseline → global depth (λ=0.10) →
masked depth (τ=0.18, λ=0.10). Evaluation:

```bash
modal run modal_train_splatfacto.py::run_eval \
  --kitti-seq-dir "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt" \
  --lambda-depth 0.10 --photo-mask-threshold 0.18 --masked
```

---

## Datasets

### KITTI Odometry

Five static sequences (00, 02, 05, 06) with 125–320 frames each. Calibrated odometry poses are used directly (no SfM). Every 10th frame is held for testing.

- **Dense** — all training frames
- **Sparse every-2** — 50% subsampled (simulates 2.5 Hz capture)

Default sequence used in most experiments:
```
KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt
```

### Mip-NeRF-360 Bicycle

194 sparse views of an outdoor object-centric scene. Every 10th frame held for validation. Images downscaled 4× for 3DGS training; depth priors computed at full resolution and resized at load time.

---

## Data Paths (remote training server)

| Data | Path |
|---|---|
| Dense KITTI (5 seq) | `/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq` |
| Sparse every-2 | `/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq_sparse_every2` |
| Nerfstudio KITTI seq02 | `/home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034` |
| Nerfstudio sparse every-2 | `/home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034_sparse_every2` |
| DA-V2 depth maps | `<seq_dir>/depths_da2` and `<seq_dir>/depths_da2_npy` |

---

## Environment Setup

Three separate conda environments are used:

| Environment | Purpose |
|---|---|
| `multinerf` | Mip-NeRF-360 training |
| `nerfstudio` | Splatfacto training |
| `da2` | Depth Anything V2 inference |

---

## Quick Start

### 1. Generate DA-V2 Depth Maps

```bash
cd /home/ubuntu/final_project/Depth-Anything-V2
conda activate da2
python run_da2_save_npy.py --help
```

### 2. Mip-NeRF-360 on KITTI (dense)

```bash
cd /home/ubuntu/final_project/outdoor-nerf-depth/nerf-methods/mipnerf360
conda activate multinerf
bash scripts/train_kitti.sh
```

### 3. Mip-NeRF-360 sparse every-2 with DA-V2 supervision

```bash
cd /home/ubuntu/final_project/outdoor-nerf-depth/nerf-methods/mipnerf360
conda activate multinerf

export DATA_DIR=/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq_sparse_every2/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt

python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.max_steps = 50000" \
  --gin_bindings="Config.sample_every = 1" \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.compute_disp_metrics = True" \
  --gin_bindings="Config.depth_loss_type = 'mse'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/logs/checkpoints-1-7.5w-mse-debug'" \
  --logtostderr
```

### 4. Splatfacto sparse every-2 (RGB-only)

```bash
cd /home/ubuntu/final_project/nerfstudio
conda activate nerfstudio
ns-train splatfacto \
  --data /home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034_sparse_every2 \
  --vis tensorboard
```

### 5. Splatfacto with DA-V2 depth supervision

```bash
cd /home/ubuntu/final_project
conda activate nerfstudio
bash scripts/train/train_splatfacto_kitti_sparse_da2.sh
```

### 6. Hyperparameter sweep (τ × λ grid)

```bash
cd /home/ubuntu/final_project
conda activate nerfstudio

# Default sweep: KITTISeq02, λ ∈ {0.1, 0.15}
bash scripts/train/train_splatfacto_kitti_sparse_da2_sweep.sh

# Custom sweep values
SWEEP_LAMBDAS="0.05 0.1 0.15 0.2" \
SWEEP_SEQ_DIRS="KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt" \
bash scripts/train/train_splatfacto_kitti_sparse_da2_sweep.sh
```

Each run writes to `outputs/<dataset>_lambda<value>/splatfacto-da2/<timestamp>/`. Sweep logs are saved under `sweep_logs/`. A failing run does not abort the sweep; a pass/fail summary is printed at the end.

---

## Modal Cloud Training

```bash
pip install modal
modal setup          # one-time auth
```

### Single run / lambda sweep

```bash
# Single run (A10G, default seq02, λ=0.05)
modal run --detach modal_train_splatfacto.py::main

# Lambda sweep — parallel A10G containers
modal run --detach modal_train_splatfacto.py::sweep --lambdas "0.0 0.05 0.1 0.15"
```

### Threshold sweep (two-stage: mask gen → retrain)

Requires a nomask base run to exist first (produced by `sweep` above).

```bash
modal run modal_train_splatfacto.py::sweep_threshold \
  --base-exp-name "kitti_seq02_0034_sparse_every2_da2_lambda0.1_nomask_50000" \
  --kitti-seq-dir "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt" \
  --lambda-depth 0.1 \
  --thresholds "0.16 0.18 0.20 0.22 1.0"
```

### Lambda × threshold joint sweep

```bash
modal run modal_train_splatfacto.py::sweep_lambda_threshold \
  --lambdas "0.05 0.1 0.15" \
  --thresholds "0.16 0.18 0.20 0.22 1.0"
```

### Matched-ratio ablation (high-error vs random masks)

```bash
# Generates both high_error_matched and random_matched (seed 0) masks, then trains both.
modal run modal_train_splatfacto.py::run_matched_ablation \
  --base-exp-name "kitti_seq02_0034_sparse_every2_da2_lambda0.1_nomask_50000" \
  --lambda-depth 0.1
```

### Random-seed stability sweep

```bash
# Add seed 1 and seed 2 (seed 0 is produced by run_matched_ablation).
modal run modal_train_splatfacto.py::run_random_seed_sweep --seeds "1,2"
```

### Cross-sequence generalization (new KITTI sequence)

Handles full data preparation (COLMAP → transforms, DA-V2 inference, GT alignment) and
trains all three key experiments in one command.

```bash
# KITTISeq05 — three experiments: RGB-only, Global depth, Low-error mask τ=0.18
modal run modal_train_splatfacto.py::run_new_seq_experiments \
  --kitti-seq-dir "KITTISeq05_2011_09_30_drive_0018_sync_llffdtu_s400_e725_densegt"
```

### Evaluation

```bash
# Single experiment
modal run modal_train_splatfacto.py::run_eval \
  --kitti-seq-dir "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt" \
  --lambda-depth 0.1

# With mask (add --masked for threshold-based experiments)
modal run modal_train_splatfacto.py::run_eval \
  --kitti-seq-dir "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt" \
  --lambda-depth 0.1 --photo-mask-threshold 0.18 --masked

# Ablation experiments (use --mask-label)
modal run modal_train_splatfacto.py::run_eval \
  --lambda-depth 0.1 --mask-label "high_error_matched_low018"
modal run modal_train_splatfacto.py::run_eval \
  --lambda-depth 0.1 --mask-label "random_matched_low018_seed0"

# Parallel eval of all lambda values
modal run modal_train_splatfacto.py::sweep_eval \
  --kitti-seq-dir "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt" \
  --lambdas "0.0 0.05 0.1 0.15" --photo-mask-threshold 0.18 --masked
```

### Download outputs

```bash
modal volume get nerf-outputs <exp_name> ./local_outputs
tensorboard --logdir ./local_outputs/<exp_name>/splatfacto-da2/<timestamp>
```

See [docs/modal-splatfacto.md](docs/modal-splatfacto.md) for full setup, dataset upload, sweep options, and cost estimates.

---

## Modal Cloud Training — Mip-NeRF-360

```bash
pip install modal
modal setup          # one-time auth
```

**`modal_train_mipnerf.py`** — JAX/Flax-based Mip-NeRF-360 training on Modal (A10G).

Prerequisite: `depths_da2` must already exist in `kitti-nerf-data` for the target
sequence. Run `modal_train_splatfacto.py::run_new_seq_experiments` first to populate it.

### Three-experiment generalization check (new KITTI sequence)

Runs all three key experiments in one command:
RGB-only → Global depth → Low-error mask (τ=0.18, λ=0.15).

```bash
modal run modal_train_mipnerf.py::run_kitti_seq_experiments \
  --kitti-seq-dir "KITTISeq05_2011_09_30_drive_0018_sync_llffdtu_s400_e725_densegt" \
  --lambda-depth 0.15 \
  --ref-threshold 0.18
```

Experiment names produced:
- `kitti_seq05_0018_sparse_every2_da2_lambda0.0_nomask_50000` (RGB-only)
- `kitti_seq05_0018_sparse_every2_da2_lambda0.15_nomask_50000` (Global depth)
- `kitti_seq05_0018_sparse_every2_da2_lambda0.15_low018_50000` (Low-error mask)

### Evaluation

```bash
# RGB-only
modal run modal_train_mipnerf.py::run_eval \
  --kitti-seq-dir "KITTISeq05_..." --lambda-depth 0.0

# Global depth
modal run modal_train_mipnerf.py::run_eval \
  --kitti-seq-dir "KITTISeq05_..." --lambda-depth 0.15

# Low-error mask (add --masked)
modal run modal_train_mipnerf.py::run_eval \
  --kitti-seq-dir "KITTISeq05_..." --lambda-depth 0.15 --masked
```

### Download outputs

```bash
modal volume get nerf-outputs <exp_name>/mipnerf360 ./local_mipnerf_ckpt
```

---

## Results Summary

### KITTISeq02 — Sparse Every-2

**Mip-NeRF-360:** Masked depth supervision provides marginal PSNR gains (+0.23 dB at best) but does not improve geometry metrics. SSIM and LPIPS slightly degrade. The implicit density-field representation is more sensitive to noisy monocular depth, which may interfere with the geometry learned through volume rendering.

**Splatfacto:** Clear improvements across all metrics at τ=0.18, λ=0.10: +1.03 dB PSNR and RMSE drops 5× (0.542 → 0.100). Increasing λ further improves RMSE (0.100 → 0.096) but reduces PSNR (15.932 → 15.588), showing a geometry–photometry tradeoff. The explicit Gaussian representation benefits more directly from depth supervision for Gaussian placement.

### Mip-NeRF-360 Bicycle — Circular Sparse Views

RGB-only Splatfacto achieves the best rendering quality (17.731 PSNR). Depth supervision consistently reduces RMSE (1.479 → 0.722) but degrades PSNR/SSIM/LPIPS. Multi-view coverage from a circular trajectory already strongly constrains geometry, so depth regularization over-constrains RGB optimization.

### Depth Prior Quality

Average scale-shift alignment error on KITTI LiDAR: **4.22 m**. DA-V2 is most reliable on well-textured mid-range surfaces and noisiest on reflective surfaces, thin structures, and distant regions.

---

## Documentation

| Topic | File |
|---|---|
| Mip-NeRF-360 KITTI setup and training | [docs/mipnerf360-kitti.md](docs/mipnerf360-kitti.md) |
| Mip-NeRF-360 Bicycle sparse (λ / mask sweep) | [docs/mip360-bicycle-sparse.md](docs/mip360-bicycle-sparse.md) |
| Nerfstudio Splatfacto dense/sparse training | [docs/nerfstudio-splatfacto.md](docs/nerfstudio-splatfacto.md) |
| Nerfstudio Splatfacto DA-V2 depth supervision | [docs/nerfstudio-splatfacto-da2.md](docs/nerfstudio-splatfacto-da2.md) |
| Modal cloud training (splatfacto-da2 sweep) | [docs/modal-splatfacto.md](docs/modal-splatfacto.md) |
| Depth Anything V2 preprocessing pipeline | [docs/depth-anything-v2.md](docs/depth-anything-v2.md) |
| Troubleshooting and citation | [docs/troubleshooting.md](docs/troubleshooting.md) |

---


This project builds on:
- [Digging into Depth Priors for Outdoor Neural Radiance Fields](https://github.com/barbararoessle/e2e_multi_view_stereo) (primary baseline)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Mip-NeRF 360](https://jonbarron.info/mipnerf360/)
- [Nerfstudio / Splatfacto](https://docs.nerf.studio/)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
