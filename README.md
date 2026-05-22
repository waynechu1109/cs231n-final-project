# cs231n-final-project

This repository contains the CS231N final project code and data setup for training depth-supervised outdoor NeRF models. The main codebase is based on **Digging into Depth Priors for Outdoor Neural Radiance Fields**, with local KITTI data already placed under `data/kitti/kitti_select_static_5seq`.

## Repository Layout

```text
.
├── data/
│   └── kitti/
│       └── kitti_select_static_5seq/
├── outdoor-nerf-depth/
│   ├── README.md
│   ├── nerf-methods/
│   │   ├── mipnerf360/
│   │   ├── nerfplusplus/
│   │   └── ngp-depth/
│   └── utils/
└── nerfstudio/
```

The most ready-to-run path in this workspace is:

```text
outdoor-nerf-depth/nerf-methods/mipnerf360
```

The default local training script uses this KITTI sequence:

```text
data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt
```


## Wayne Handoff Paths

Use this section when sharing the current workspace state with Wayne.

### Current Repo

```text
/home/ubuntu/final_project
```

Main code roots:

```text
/home/ubuntu/final_project/outdoor-nerf-depth/nerf-methods/mipnerf360
/home/ubuntu/final_project/nerfstudio
/home/ubuntu/final_project/Depth-Anything-V2
```

### Processed KITTI Data

Dense 5-sequence KITTI data:

```text
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq
```

Sparse KITTI data prepared from the dense data:

```text
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq_sparse_every2
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq_sparse_every4
```

Default sequence used in most runs:

```text
KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt
```

Nerfstudio-formatted KITTI data for `splatfacto`:

```text
/home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034
/home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034_sparse_every2
/home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034_sparse_every4
```

### Generated Depth Maps

For the default KITTI sequence, generated and prepared depth maps live under:

```text
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/depths_da2
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/depths_da2_npy
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/depths_mff_crop
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/depths_mono_crop
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/depths_ste_conf_-1_crop
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/depths_gt
```

Sparse every-2 and every-4 copies also include matching depth folders under their default sequence directories.

### Previous Outputs, Checkpoints, And Logs

MipNeRF-360 dense/default-sequence runs:

```text
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/logs/checkpoints-1-7.5w-mse-debug
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/logs/checkpoints-da2-vits-debug-1000
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/logs/checkpoints-da2-vits-lambda005-25k
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/logs/checkpoints-sparseview-da2-vits-lambda005-50k
```

MipNeRF-360 sparse runs:

```text
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq_sparse_every2/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/logs/checkpoints-1-7.5w-mse-debug
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq_sparse_every4/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/logs/checkpoints-sparse-every4-7.5w-mse
```

Nerfstudio `splatfacto` outputs:

```text
/home/ubuntu/final_project/nerfstudio/outputs/kitti_seq02_0034/splatfacto
/home/ubuntu/final_project/nerfstudio/outputs/kitti_seq02_0034_sparse_every2/splatfacto
/home/ubuntu/final_project/nerfstudio/outputs/kitti_seq02_0034_sparse_every4/splatfacto
```

Observed run directories include:

```text
/home/ubuntu/final_project/nerfstudio/outputs/kitti_seq02_0034/splatfacto/2026-05-18_041846
/home/ubuntu/final_project/nerfstudio/outputs/kitti_seq02_0034/splatfacto/2026-05-21_180416
/home/ubuntu/final_project/nerfstudio/outputs/kitti_seq02_0034_sparse_every2/splatfacto/2026-05-22_165305
/home/ubuntu/final_project/nerfstudio/outputs/kitti_seq02_0034_sparse_every2/splatfacto/2026-05-22_165432
/home/ubuntu/final_project/nerfstudio/outputs/kitti_seq02_0034_sparse_every4/splatfacto/2026-05-20_164544
/home/ubuntu/final_project/nerfstudio/outputs/kitti_seq02_0034_sparse_every4/splatfacto/2026-05-20_164659
/home/ubuntu/final_project/nerfstudio/outputs/kitti_seq02_0034_sparse_every4/splatfacto/2026-05-20_175757
/home/ubuntu/final_project/nerfstudio/outputs/kitti_seq02_0034_sparse_every4/splatfacto/2026-05-20_182311
```

### Exact Sparse Training Commands

MipNeRF-360 sparse every-2 command/config currently used:

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

MipNeRF-360 sparse every-4 command/config used by `scripts/train_kitti_sparse.sh`:

```bash
cd /home/ubuntu/final_project/outdoor-nerf-depth/nerf-methods/mipnerf360
conda activate multinerf
bash scripts/train_kitti_sparse.sh
```

That script sets `Config.max_steps = 75000`, `Config.sample_every = 1`, `Config.depth_loss_type = 'mse'`, `Config.compute_disp_metrics = True`, `Config.data_dir` to the sparse every-4 KITTI sequence, and `Config.checkpoint_dir` to `${DATA_DIR}/logs/checkpoints-sparse-every4-7.5w-mse`.

Nerfstudio `splatfacto` sparse every-2 command:

```bash
cd /home/ubuntu/final_project/nerfstudio
conda activate nerfstudio

ns-train splatfacto \
  --data /home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034_sparse_every2 \
  --vis tensorboard
```

## Documentation

The original long README has been split into focused guides:

| Topic | File |
| --- | --- |
| MipNeRF-360 KITTI setup and training | [docs/mipnerf360-kitti.md](docs/mipnerf360-kitti.md) |
| Nerfstudio Splatfacto dense/sparse training | [docs/nerfstudio-splatfacto.md](docs/nerfstudio-splatfacto.md) |
| Depth Anything V2 depth-supervision pipeline | [docs/depth-anything-v2.md](docs/depth-anything-v2.md) |
| NeRF++ and Instant-NGP Depth notes | [docs/other-methods.md](docs/other-methods.md) |
| Troubleshooting and citation | [docs/troubleshooting.md](docs/troubleshooting.md) |

## Quick Start

MipNeRF-360 on the default KITTI sequence:

```bash
cd /home/ubuntu/final_project/outdoor-nerf-depth/nerf-methods/mipnerf360
conda activate multinerf
bash scripts/train_kitti.sh
```

Sparse MipNeRF-360 on the prepared every-4th-frame KITTI sequence:

```bash
cd /home/ubuntu/final_project/outdoor-nerf-depth/nerf-methods/mipnerf360
conda activate multinerf
bash scripts/train_kitti_sparse.sh
```

Splatfacto on the prepared sparse nerfstudio dataset:

```bash
cd /home/ubuntu/final_project/nerfstudio
conda activate nerfstudio
ns-train splatfacto \
  --data /home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034_sparse_every4 \
  --vis tensorboard
```

Depth Anything V2 preprocessing:

```bash
cd /home/ubuntu/final_project/Depth-Anything-V2
conda activate da2
python run_da2_save_npy.py --help
```
