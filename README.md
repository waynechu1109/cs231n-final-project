# Reliability-Aware Monocular Depth Supervision for Sparse-View Neural Reconstruction

**CS231N Final Project — Stanford University**
Wayne Chu · Yashasvini Gopalan · Changju Yuan

---

## Overview

Sparse-view neural reconstruction in outdoor driving scenes is challenging because cameras move along a narrow forward-facing trajectory with limited multi-view overlap. Although monocular depth estimators can provide dense geometric priors, their predictions are noisy and not uniformly reliable across image regions.

This project investigates **photometric-masked monocular depth supervision** for sparse-view outdoor scene reconstruction. We use [Depth Anything V2 (DA-V2)](https://github.com/DepthAnything/Depth-Anything-V2) as a dense monocular depth prior, align its predictions to metric depth via scale-shift fitting, and apply depth supervision selectively using photometric masks generated from an RGB-only baseline model. We evaluate on two representative scene representations: **Mip-NeRF-360** and **Splatfacto (3DGS)**.

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

Rather than applying depth loss uniformly, we restrict supervision to pixels where the RGB-only baseline is already reliable. A per-pixel photometric error map is computed from a pre-trained RGB-only model:

$$e(u) = \frac{1}{3} \sum_{c \in \lbrace R,G,B \rbrace} \left| \hat{I}_c(u) - I_c(u) \right|$$

Pixels with $e(u) < \tau$ form the binary reliability mask $M(u)$. This mask is fixed for the duration of depth-supervised training.

### Training Pipeline

1. **Stage 1** — Train RGB-only baseline for 50 000 iterations.
2. **Stage 2** — Render all training views; compute per-pixel photometric error; generate fixed masks at threshold τ.
3. **Stage 3** — Retrain with depth supervision: $\mathcal{L} = \mathcal{L}_\text{rgb} + \lambda_\text{depth}\,\mathcal{L}_\text{depth}$, where $\mathcal{L}_\text{depth}$ is MSE gated by $M_\text{eff}(u) = M(u) \land D(u)$ ($D$ = depth-validity mask).

We sweep τ ∈ {0.14, 0.16, 0.18, 0.20, 0.22, 1.0} and λ ∈ {0.05, 0.10, 0.15}.

---

## Repository Layout

```text
.
├── outdoor-nerf-depth/               # Mip-NeRF-360 backbone (Digging into Depth Priors baseline)
│   └── nerf-methods/mipnerf360/
│       ├── configs/360.gin
│       └── scripts/
├── nerfstudio/                       # Splatfacto backbone
│   └── outputs/
├── Depth-Anything-V2/                # DA-V2 depth estimation
├── scripts/                          # Data prep, training sweeps, evaluation
├── modal_train_splatfacto.py         # Modal cloud training entry point
├── data/
│   ├── kitti/kitti_select_static_5seq/
│   └── nerfstudio/
└── docs/                             # Extended per-topic guides
```

---

## Datasets

### KITTI Odometry

Five static sequences (00, 02, 05, 06) with 125–320 frames each. Calibrated odometry poses are used directly (no SfM). Every 10th frame is held for testing.

- **Dense** — all training frames
- **Sparse every-2** — 50% subsampled (simulates 2.5 Hz capture)
- **Sparse every-4** — 25% subsampled

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
bash scripts/train_splatfacto_kitti_sparse_da2.sh
```

### 6. Hyperparameter sweep (τ × λ grid)

```bash
cd /home/ubuntu/final_project
conda activate nerfstudio

# Default sweep: KITTISeq02, λ ∈ {0.1, 0.15}
bash scripts/train_splatfacto_kitti_sparse_da2_sweep.sh

# Custom sweep values
SWEEP_LAMBDAS="0.05 0.1 0.15 0.2" \
SWEEP_SEQ_DIRS="KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt" \
bash scripts/train_splatfacto_kitti_sparse_da2_sweep.sh
```

Each run writes to `outputs/<dataset>_lambda<value>/splatfacto-da2/<timestamp>/`. Sweep logs are saved under `sweep_logs/`. A failing run does not abort the sweep; a pass/fail summary is printed at the end.

---

## Modal Cloud Training

```bash
pip install modal
modal setup          # one-time auth

# Single run (A10G, detached)
modal run --detach modal_train_splatfacto.py::main

# Lambda sweep (parallel A10G containers)
modal run --detach modal_train_splatfacto.py::sweep --lambdas "0.0 0.05 0.1 0.2"

# Eval a completed run
modal run modal_train_splatfacto.py::run_eval --lambda-depth 0.05

# Download outputs
modal volume get nerf-outputs <exp_name> ./local_outputs
tensorboard --logdir ./local_outputs/<exp_name>/splatfacto-da2/<timestamp>
```

See [docs/modal-splatfacto.md](docs/modal-splatfacto.md) for full setup, dataset upload, sweep options, and cost estimates.

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

## Citation

```bibtex
@misc{chu2026reliability,
  title   = {Reliability-Aware Monocular Depth Supervision for Sparse-View Neural Reconstruction},
  author  = {Chu, Wayne and Gopalan, Yashasvini and Yuan, Changju},
  year    = {2026},
  note    = {CS231N Final Project, Stanford University}
}
```

This project builds on:
- [Digging into Depth Priors for Outdoor Neural Radiance Fields](https://github.com/barbararoessle/e2e_multi_view_stereo) (primary baseline)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Mip-NeRF 360](https://jonbarron.info/mipnerf360/)
- [Nerfstudio / Splatfacto](https://docs.nerf.studio/)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
