# Splatfacto DA2 Depth Supervision

This guide covers DA2 depth-supervised `splatfacto` training on KITTI, mirroring the MipNeRF-360 DA2 setup documented in [depth-anything-v2.md](depth-anything-v2.md).

## Overview

The MipNeRF-360 KITTI pipeline supervises training with scale-shift aligned Depth Anything V2 depth maps from `depths_da2/`. The splatfacto equivalent uses the same depth folder and the same core hyperparameters:

| MipNeRF-360 | Splatfacto |
|---|---|
| `Config.depth_sup_type='da2'` | `depths_da2/` wired into `transforms.json` |
| `Config.lambda_depth=0.05` | `--pipeline.model.lambda-depth 0.05` |
| `Config.depth_loss_type='mse'` | `--pipeline.model.depth-loss-type mse` |
| `Config.max_steps=25000` (example) | `--max-num-iterations 50000` (default) |

Training method name: `splatfacto-da2`.

## Prerequisites

1. Generate and align DA2 depth for the target KITTI sequence. See [depth-anything-v2.md](depth-anything-v2.md).
2. Activate the nerfstudio environment:

```bash
conda activate nerfstudio
cd /home/ubuntu/final_project/nerfstudio
pip install -e .
```

3. Use the AWS/CUDA compile setup from [nerfstudio-splatfacto.md](nerfstudio-splatfacto.md) if this is the first splatfacto run on the machine.

## Prepare the Nerfstudio Dataset

The script below creates a nerfstudio dataset variant with `depth_file_path` entries and a symlink to the KITTI depth folder. It mirrors MipNeRF's `depths_<depth_sup_type>` naming.

### Dense sequence

```bash
cd /home/ubuntu/final_project

python scripts/make_nerfstudio_kitti_depth.py \
  --src data/nerfstudio/kitti_seq02_0034 \
  --dst data/nerfstudio/kitti_seq02_0034_da2 \
  --depth-dir data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/depths_da2 \
  --depth-sup-type da2
```

Output:

```text
data/nerfstudio/kitti_seq02_0034_da2/
  images -> .../images
  depths_da2 -> .../depths_da2
  transforms.json
  DEPTH_DA2_MANIFEST.txt
```

### Sparse every-2 sequence

```bash
cd /home/ubuntu/final_project

python scripts/make_nerfstudio_kitti_depth.py \
  --src data/nerfstudio/kitti_seq02_0034_sparse_every2 \
  --dst data/nerfstudio/kitti_seq02_0034_sparse_every2_da2 \
  --depth-dir data/kitti/kitti_select_static_5seq_sparse_every2/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/depths_da2 \
  --depth-sup-type da2
```

Each frame in `transforms.json` gets a field like:

```json
"depth_file_path": "./depths_da2/00000000.png"
```

Depth PNGs use the same KITTI encoding as MipNeRF (`depth_m = uint16 / 256`, `0` = invalid). The dataparser uses `depth_unit_scale_factor = 1/256`.

## Train DA2-Supervised Splatfacto

### Dense KITTI

```bash
cd /home/ubuntu/final_project
bash scripts/train_splatfacto_kitti_da2.sh
```

Equivalent manual command:

```bash
cd /home/ubuntu/final_project/nerfstudio

ns-train splatfacto-da2 \
  --data /home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034_da2 \
  --max-num-iterations 50000 \
  --pipeline.model.lambda-depth 0.05 \
  --pipeline.model.depth-loss-type mse \
  --vis tensorboard
```

### Sparse every-2 KITTI

```bash
cd /home/ubuntu/final_project
bash scripts/train_splatfacto_kitti_sparse_da2.sh
```

Equivalent manual command:

```bash
cd /home/ubuntu/final_project/nerfstudio

ns-train splatfacto-da2 \
  --data /home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034_sparse_every2_da2 \
  --max-num-iterations 50000 \
  --pipeline.model.lambda-depth 0.05 \
  --pipeline.model.depth-loss-type mse \
  --vis tensorboard
```

The training scripts create the depth-enabled nerfstudio dataset automatically if it does not exist yet.

## Key Settings

These mirror the MipNeRF-360 KITTI DA2 run in [depth-anything-v2.md](depth-anything-v2.md):

| Setting | Default | Notes |
|---|---|---|
| `depth_sup_type` | `da2` | Selects `depths_da2/` when preparing the dataset |
| `lambda_depth` | `0.05` | Depth loss weight |
| `depth_loss_type` | `mse` | Valid values: `mse`, `l1` |
| `depth_crop_range` | `80.0` | Stored in `transforms.json`; ignore supervision depth above 80 m |
| `depth_unit_scale_factor` | `1/256` | Converts KITTI PNG encoding to meters |

Override at launch time:

```bash
LAMBDA_DEPTH=0.05 DEPTH_LOSS_TYPE=mse bash scripts/train_splatfacto_kitti_sparse_da2.sh
```

## Evaluate and Export

Use the same commands as RGB-only splatfacto. Point `--load-config` at the DA2 run config:

```bash
cd /home/ubuntu/final_project/nerfstudio

ns-eval \
  --load-config outputs/kitti_seq02_0034_sparse_every2_da2/splatfacto-da2/<timestamp>/config.yml \
  --output-path eval_splatfacto_kitti_sparse_da2.json
```

Export the trained Gaussian splat:

```bash
ns-export gaussian-splat \
  --load-config outputs/kitti_seq02_0034_sparse_every2_da2/splatfacto-da2/<timestamp>/config.yml \
  --output-dir exports/kitti_seq02_0034_sparse_every2_da2_splat
```

Tensorboard logs include `depth_loss`, `depth_mse`, and `depth_mae` on eval images when depth maps are present.

## Comparison With RGB-Only Splatfacto

| Run | Method | Data |
|---|---|---|
| RGB-only baseline | `splatfacto` | `data/nerfstudio/kitti_seq02_0034_sparse_every2` |
| DA2 depth supervision | `splatfacto-da2` | `data/nerfstudio/kitti_seq02_0034_sparse_every2_da2` |

Compare against the corresponding MipNeRF runs:

| MipNeRF | Splatfacto |
|---|---|
| `Config.depth_sup_type='rgbonly'` | `ns-train splatfacto` |
| `Config.depth_sup_type='da2'` | `ns-train splatfacto-da2` |

## Notes

- The DA2 depth maps must already exist under the KITTI sequence folder before preparing the nerfstudio dataset.
- GT depth supervision (`depth_sup_type='gt'`) can be prepared the same way by pointing `--depth-dir` at `depths_gt`.
- This method supervises rendered Gaussian depth against the input depth maps during training. It does not use depth only for initialization.
