# Reliability-Aware Monocular Depth Supervision for Sparse-View Neural Reconstruction

**CS231N Final Project — Stanford University**
Wayne Chu · Yashasvini Gopalan · Changju Yuan

<div class="links">
  <a href="https://waynechu1109.github.io/cs231n-final-project/"><img src="https://img.shields.io/badge/Project_Page-blue" alt="Project Page">
  <a href="https://arxiv.org/abs/2607.02554v1"><img src="https://img.shields.io/badge/arXiv-2607.02554-b31b1b?logo=arxiv" alt="arXiv"></a>
  <a href="https://waynechu1109.github.io/slides/cs231n_poster.pdf"><img src="https://img.shields.io/badge/Poster-PDF-1f6feb" alt="Poster"></a>
</div>

---

## Overview

Reconstructing outdoor driving scenes from sparse views is difficult due to the narrow forward-facing trajectory of the camera motion and limited multi-view overlap. Monocular depth estimators can provide dense geometric priors, but their predictions are noisy and not consistently reliable across image regions.

This project focuses on **photometric-masked monocular depth supervision** for sparse-view outdoor scene reconstruction. We leverage [Depth Anything V2 (DA-V2)](https://github.com/DepthAnything/Depth-Anything-V2) as a dense monocular depth prior, calibrate its predictions to metric depth with per-image scale-shift fitting on sparse anchors (LiDAR on KITTI, COLMAP on Bicycle), and apply depth supervision selectively via photometric masks derived from an RGB-only baseline. Since the mask modifies an existing depth loss rather than introducing one, we compare against **global (τ=1.0, unmasked) supervision** as the primary baseline. We evaluate on two representative scene representations: **Mip-NeRF-360** and **Splatfacto (3DGS)**.

### Key Results

**Splatfacto — masked-vs-global on KITTI 00 / 02 / 05.** The mask adds +0.44 to +0.70 dB PSNR over global supervision at tied or better RMSE. The RMSE drop comes from using a depth prior at all; masking is what improves rendering fidelity.

| Seq     | Method                       | PSNR ↑ | SSIM ↑ | LPIPS ↓ | RMSE ↓ |
|---------|------------------------------|--------|--------|---------|--------|
| 02/034  | RGB-only                     | 14.90  | 0.433  | 0.446   | 0.542  |
| 02/034  | Global (τ=1.0)               | 15.49  | 0.448  | 0.434   | 0.101  |
| 02/034  | **Masked (τ=0.18)**          | **15.93** | **0.477** | **0.408** | **0.100** |
| 05/018  | RGB-only                     | 14.89  | 0.521  | 0.493   | 0.807  |
| 05/018  | Global (τ=1.0)               | 15.26  | 0.534  | 0.469   | 0.096  |
| 05/018  | **Masked (τ=0.18)**          | **15.90** | **0.548** | **0.446** | **0.096** |
| 00/027  | RGB-only                     | 14.67  | 0.487  | 0.401   | 0.514  |
| 00/027  | Global (τ=1.0)               | 16.69  | 0.554  | 0.302   | 0.125  |
| 00/027  | **Masked (τ=0.18)**          | **17.39** | **0.571** | **0.288** | **0.114** |

**Mip-NeRF-360 — KITTISeq02.** The optimal setting bypasses the mask entirely (global τ=1.0 gives the best PSNR); the implicit density field is more sensitive to noisy monocular depth.

| Setting                 | PSNR ↑ | SSIM ↑    | LPIPS ↓   | RMSE ↓    |
|-------------------------|--------|-----------|-----------|-----------|
| RGB-only                | 20.378 | **0.601** | **0.409** | **2.703** |
| Masked (τ=0.18, λ=0.10) | 20.384 | 0.594     | 0.416     | 3.532     |
| **Global (τ=1.0, λ=0.15)** | **20.607** | 0.595 | 0.412 | 3.580 |

**Comparison vs LiDAR-supervised baselines (KITTISeq02).** Without any GT depth, our mask outperforms three of four LiDAR-supervised methods.

| Method                              | Depth source | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------------------------------------|--------------|--------|--------|---------|
| RGB-only                            | none         | 14.903 | 0.433  | 0.446   |
| DA-V2 depth, no mask (global)       | DA-V2        | 15.494 | 0.448  | 0.434   |
| **Ours (τ=0.18, λ=0.10)**           | DA-V2        | **15.932** | **0.477** | **0.408** |
| DNGaussian                          | GT LiDAR     | 9.98   | 0.303  | 0.710   |
| DepthRegGS                          | GT LiDAR     | 8.71   | 0.229  | 0.737   |
| SparseGS                            | GT LiDAR     | 12.20  | 0.359  | 0.648   |
| DN-Splatter                         | GT LiDAR     | **16.22** | **0.489** | **0.289** |

---

## Method

### Monocular Depth Prior Alignment

DA-V2 predicts relative depth. We align each prediction to metric depth via per-image least-squares scale-shift fitting — solving for the scale *s* and shift *t* that best match the predicted depth *d_m* to the reference depth *d_r* over valid anchor pixels.

For KITTI, reference depth comes from projected LiDAR points. For Mip-NeRF-360 Bicycle, sparse COLMAP points are used. Aligned depth is clipped to 80 m for KITTI.

### Photometric-Masked Depth Supervision

Instead of applying the depth loss everywhere, we only supervise the pixels where the RGB-only baseline is already reliable. A pre-trained RGB-only model is used to compute a per-pixel photometric error map:

$$e(u) = \frac{1}{3} \sum_{c \in \lbrace R,G,B \rbrace} \left| \hat{I}_c(u) - I_c(u) \right|$$

Pixels with $e(u) < \tau$ form the binary reliability mask $M(u)$. This mask is fixed for the duration of depth-supervised training.

### Training Pipeline

1. **Stage 1** — Train RGB-only baseline for 50 000 iterations.
2. **Stage 2** — Render all training views; compute per-pixel photometric error; generate fixed masks at threshold τ.
3. **Stage 3** — Retrain from a fresh initialization with depth supervision: $\mathcal{L} = \mathcal{L}_\text{rgb} + \lambda_\text{depth}\,\mathcal{L}_\text{depth}$, where $\mathcal{L}_\text{depth}$ is MSE gated by $M_\text{eff}(u) = M(u) \land D(u)$ ($D$ = depth-validity mask).

We sweep τ ∈ {0.14, 0.16, 0.18, 0.20, 0.22, 1.0} and λ ∈ {0.05, 0.10, 0.15}. **Setting τ = 1.0 gives $M \equiv 1$ — global (unmasked) depth supervision** — which is our primary baseline for judging the mask's contribution, since it isolates the effect of *where* depth is supervised from *whether* depth is supervised at all.

### Matched-Ratio Ablation

To test whether the benefit of low-error masking comes from **selecting reliable pixels** vs simply **using fewer pixels**, we construct two control masks that match the exact pixel budget of `low018` (i.e. the same number of supervised pixels per frame):

- **High-error matched** — selects the *k* valid-depth pixels with the **highest** photometric error per frame.
- **Random matched** — selects *k* valid-depth pixels **uniformly at random** per frame (three seeds for stability).

where *k* = |{u : valid_depth(u) ∧ e(u) < 0.18}| per frame, computed from the RGB-only baseline. All ablation runs use λ = 0.10, τ = 0.18 on KITTISeq02 every-2. The low-error mask wins on every metric — the improvement is not from fewer supervised pixels.

| Mask (λ=0.10)                    | PSNR ↑ | SSIM ↑ | LPIPS ↓ | RMSE ↓ |
|----------------------------------|--------|--------|---------|--------|
| High-error, matched              | 14.932 | 0.437  | 0.455   | 0.111  |
| Random, matched (3 seeds)        | 15.036 | 0.442  | 0.456   | 0.109  |
| **Low-error, τ=0.18 (ours)**     | **15.932** | **0.477** | **0.408** | **0.100** |

### Mask Validity Against GT LiDAR

Because photometric and depth errors are measured against different references, low photometric error does not by construction imply low depth error. We validate the mask directly against GT LiDAR. Depth RMSE is **34–39% lower inside the mask** at every threshold (6.33 vs 10.01 m at τ=0.18), and inside-mask RMSE increases monotonically with τ (r=1.0). Pixel-level differences are highly significant (p < 10⁻⁵⁰); frame-level paired t-test gives p = 0.047 (n = 8).

### Comparison to a Depth-Inconsistency Mask

We also compare against a static, one-shot approximation of the Depth-Inconsistency Mask (DIM) at a matched reliable-pixel fraction. Ratio = MAE(outside) / MAE(inside): >1 means the mask isolates accurate DA-V2 depth. Our mask separates accurate from inaccurate depth (ratio **1.17**) while the DIM proxy does not (**0.87**): DIM keys on distortions that depth-supervised training itself induces, so a one-shot proxy has nothing to key on.

| Mask                | Threshold | Fraction | MAE in / out (m) | Ratio |
|---------------------|-----------|----------|------------------|-------|
| **Photometric (ours)** | τ = 0.18 | 95.9%   | 4.14 / 4.82     | **1.17** |
| DIM proxy           | ε = 17.6 m | 96.1%  | 4.19 / 3.66     | 0.87  |

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

### Splatfacto — KITTI 00 / 02 / 05 (Sparse Every-2)

Splatfacto benefits from monocular depth supervision. Using any DA-V2 depth prior — even without a mask — cuts RMSE by ~80% (KITTISeq02: 0.542 → 0.101). Masking the depth loss adds a further **+0.44 to +0.70 dB PSNR** over global supervision at tied or better RMSE across sequences 00 / 02 / 05. Over three seeds on KITTISeq02: masked 15.30 ± 0.57 vs global 15.07 ± 0.38 dB (mean +0.23 dB). Bottom line: the RMSE drop comes from using a depth prior at all, and masking is what improves rendering fidelity. Increasing λ from 0.10 to 0.15 at τ=0.18 improves RMSE (0.100 → 0.096) but reduces PSNR (15.932 → 15.588), showing a geometry–photometry tradeoff.

### Mip-NeRF-360 — KITTI 02 and 05 (Sparse Every-2)

The implicit density field is far more sensitive to noisy monocular depth. On KITTISeq02 the optimal setting is global (τ=1.0) — the mask brings no gain; all depth-supervised settings degrade SSIM, LPIPS, and geometry relative to RGB-only. On KITTISeq05 both global and masked supervision degrade rendering and geometry vs RGB-only; masking recovers 0.22 dB PSNR over global (16.612 vs 16.389), so it mitigates but does not reverse the harm.

| Seq | Setting                | PSNR ↑ | SSIM ↑    | LPIPS ↓   | AbsRel ↓ | RMSE ↓    |
|-----|------------------------|--------|-----------|-----------|----------|-----------|
| 05  | **RGB-only**           | **17.068** | **0.546** | **0.529** | **0.1166** | **2.978** |
| 05  | Global (τ=1.0, λ=0.15) | 16.389 | 0.527     | 0.569     | 0.1527   | 4.803     |
| 05  | Masked (τ=0.18, λ=0.15)| 16.612 | 0.530     | 0.563     | 0.1399   | 4.454     |

### Mip-NeRF-360 Bicycle — Circular Sparse Views

RGB-only Splatfacto achieves the best rendering quality (17.731 PSNR). Depth supervision consistently reduces RMSE (1.479 → 0.722) but degrades PSNR/SSIM/LPIPS. Global supervision gives the best PSNR among depth-supervised runs (17.593 vs 17.466 for the best mask), confirming the mask's rendering gain is specific to sparse forward-facing trajectories — depth regularization over-constrains an already well-posed reconstruction.

### Depth Prior Quality

Average scale-shift alignment error on KITTI LiDAR: **4.22 m**. Sweeping the number of anchors used for scale-shift fitting shows the 2-DOF optimization saturates well before typical LiDAR density: MAE is stable from ~95k anchors down to 500 per frame, only degrading under extreme sparsity (4.29 m at 100 anchors, 4.51 m at 20). The 4.22 m error floor reflects DA-V2's intrinsic local structural limitations, not calibration artifacts. DA-V2 is most reliable on well-textured mid-range surfaces and noisiest on reflective surfaces, thin structures, and distant regions.

### Takeaways

Monocular depth priors are most useful for **explicit Gaussian representations** in under-constrained, forward-facing sparse-view scenes, and less reliable for implicit NeRF-style density fields. The mask's contribution is rendering fidelity, not metric geometry: +0.44 to +0.70 dB PSNR over global supervision on KITTI 00/02/05 at tied or better RMSE, no gain for Mip-NeRF-360 or the object-centric Bicycle scene. Without any ground-truth depth, our mask also outperforms three LiDAR-supervised baselines. The explicit-vs-implicit architectural insights are prior-agnostic; sweep values (τ, λ) are dataset-specific.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{chu2026reliability,
    title         = {Reliability-Aware Monocular Depth Supervision for Sparse-View Neural Reconstruction},
    author        = {Wei-Teng Chu and Yashasvini Gopalan and Changju Yuan},
    year          = {2026},
    eprint        = {2607.02554},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV},
    url           = {https://arxiv.org/abs/2607.02554}
}
```

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
