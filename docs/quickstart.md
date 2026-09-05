# Quick Start

Local training commands. For Modal cloud training see [modal-splatfacto.md](modal-splatfacto.md) and [modal-mipnerf.md](modal-mipnerf.md).

## Environment Setup

Three separate conda environments are used:

| Environment | Purpose |
|---|---|
| `multinerf` | Mip-NeRF-360 training |
| `nerfstudio` | Splatfacto training |
| `da2` | Depth Anything V2 inference |

## 1. Generate DA-V2 Depth Maps

```bash
cd /home/ubuntu/final_project/Depth-Anything-V2
conda activate da2
python run_da2_save_npy.py --help
```

See [depth-anything-v2.md](depth-anything-v2.md) for the full DA-V2 preprocessing pipeline.

## 2. Mip-NeRF-360 on KITTI (dense)

```bash
cd /home/ubuntu/final_project/outdoor-nerf-depth/nerf-methods/mipnerf360
conda activate multinerf
bash scripts/train_kitti.sh
```

## 3. Mip-NeRF-360 sparse every-2 with DA-V2 supervision

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

## 4. Splatfacto sparse every-2 (RGB-only)

```bash
cd /home/ubuntu/final_project/nerfstudio
conda activate nerfstudio
ns-train splatfacto \
  --data /home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034_sparse_every2 \
  --vis tensorboard
```

## 5. Splatfacto with DA-V2 depth supervision

```bash
cd /home/ubuntu/final_project
conda activate nerfstudio
bash scripts/train/train_splatfacto_kitti_sparse_da2.sh
```

## 6. Hyperparameter sweep (τ × λ grid)

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
