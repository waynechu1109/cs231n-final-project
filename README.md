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

## Fixed Photometric Mask Depth Supervision

A two-stage pipeline is provided for applying photometric-error-based masks to DA2 depth supervision.

1. Train an RGB-only baseline with `sample_every=2` and `lambda_depth=0.0`.
2. Render the training views from the RGB-only checkpoint and compute raw per-pixel photometric error.
3. Generate fixed binary masks using a selected photometric error threshold.
4. Train MipNeRF-360 with DA2 depth supervision while keeping supervision only on pixels selected by the fixed mask.

The mask is generated from the RGB-only baseline and remains fixed during depth-supervised training. This avoids coupling mask generation with the model currently being optimized.

### Generate Fixed Masks

The following example generates a low-error mask with threshold `0.18`.

```bash
cd /home/ubuntu/cs231n_project/code/cs231n-final-project/outdoor-nerf-depth/nerf-methods/mipnerf360
conda activate multinerf

export REPO=/home/ubuntu/cs231n_project/code/cs231n-final-project
export DATA_DIR=$REPO/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt
export RGB_CKPT=$DATA_DIR/logs/checkpoints-sparseview-rgbonly-sampleevery2-50k
export MASK_DIR=$DATA_DIR/photo_masks_rgbonly_low018_sampleevery2

python -m generate_fixed_photo_masks \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir='${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir='${RGB_CKPT}'" \
  --gin_bindings="Config.max_steps=50000" \
  --gin_bindings="Config.batch_size=4096" \
  --gin_bindings="Config.compute_disp_metrics=False" \
  --gin_bindings="Config.lambda_depth=0.0" \
  --gin_bindings="Config.sample_every=2" \
  --gin_bindings="Config.fixed_photo_mask_dir='${MASK_DIR}'" \
  --gin_bindings="Config.photo_mask_threshold=0.18" \
  --gin_bindings="Config.photo_mask_mode='low'" \
  --gin_bindings="Config.auto_adjust_near_far=True" \
  --gin_bindings="Config.near=0.2" \
  --gin_bindings="Config.far=1000000.0" \
  --gin_bindings="Model.opaque_background=True" \
  --gin_bindings="Model.raydist_fn=@jnp.reciprocal" \
  --gin_bindings="NerfMLP.disable_density_normals=True" \
  --gin_bindings="NerfMLP.net_depth=8" \
  --gin_bindings="NerfMLP.net_width=1024" \
  --gin_bindings="NerfMLP.warp_fn=@coord.contract" \
  --gin_bindings="PropMLP.disable_density_normals=True" \
  --gin_bindings="PropMLP.disable_rgb=True" \
  --gin_bindings="PropMLP.net_depth=4" \
  --gin_bindings="PropMLP.net_width=256" \
  --gin_bindings="PropMLP.warp_fn=@coord.contract" \
  --logtostderr
```

The generated mask directory should contain 79 PNG masks for the `sample_every=2` training split.

### Train with Fixed Masks

```bash
cd /home/ubuntu/cs231n_project/code/cs231n-final-project/outdoor-nerf-depth/nerf-methods/mipnerf360
conda activate multinerf

export REPO=/home/ubuntu/cs231n_project/code/cs231n-final-project
export DATA_DIR=$REPO/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt
export MASK_DIR=$DATA_DIR/photo_masks_rgbonly_low018_sampleevery2
export CKPT_DIR=$DATA_DIR/logs/checkpoints-sparseview-da2-vits-lambda005-sampleevery2-fixedmask-low018-50k

python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir='${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir='${CKPT_DIR}'" \
  --gin_bindings="Config.max_steps=50000" \
  --gin_bindings="Config.checkpoint_every=25000" \
  --gin_bindings="Config.batch_size=4096" \
  --gin_bindings="Config.compute_disp_metrics=True" \
  --gin_bindings="Config.depth_sup_type='da2'" \
  --gin_bindings="Config.depth_keep_ratio=0.0" \
  --gin_bindings="Config.fixed_photo_mask_dir='${MASK_DIR}'" \
  --gin_bindings="Config.depth_loss_type='mse'" \
  --gin_bindings="Config.lambda_depth=0.05" \
  --gin_bindings="Config.sample_every=2" \
  --gin_bindings="Config.auto_adjust_near_far=True" \
  --gin_bindings="Config.near=0.2" \
  --gin_bindings="Config.far=1000000.0" \
  --gin_bindings="Model.opaque_background=True" \
  --gin_bindings="Model.raydist_fn=@jnp.reciprocal" \
  --gin_bindings="NerfMLP.disable_density_normals=True" \
  --gin_bindings="NerfMLP.net_depth=8" \
  --gin_bindings="NerfMLP.net_width=1024" \
  --gin_bindings="NerfMLP.warp_fn=@coord.contract" \
  --gin_bindings="PropMLP.disable_density_normals=True" \
  --gin_bindings="PropMLP.disable_rgb=True" \
  --gin_bindings="PropMLP.net_depth=4" \
  --gin_bindings="PropMLP.net_width=256" \
  --gin_bindings="PropMLP.warp_fn=@coord.contract" \
  --logtostderr
```

### MipNeRF-360 Ablation Results

The following PSNR values are from KITTI Seq02 using `sample_every=2`.

| Setting | Mean PSNR |
| --- | ---: |
| RGB-only, no depth | 20.3882 |
| Global DA2 depth, `lambda_depth=0.05` | 19.8834 |
| Global DA2 depth, `lambda_depth=0.10` | 20.2170 |
| Global DA2 depth, `lambda_depth=0.15` | 20.6066 |
| Fixed low-error mask, threshold 0.16, `lambda_depth=0.15` | 20.5807 |

The results indicate that DA2 depth supervision is sensitive to the depth loss weight. Fixed photometric masks can improve some low-weight settings by filtering high-error pixels, while the best observed configuration in this set uses global DA2 supervision with `lambda_depth=0.15`.

