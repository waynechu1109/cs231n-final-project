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
