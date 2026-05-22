# MipNeRF-360 KITTI Training

This guide covers the main KITTI MipNeRF-360 environment, dataset checks, training commands, and monitoring workflow.

## 1. Create the Training Environment

The MipNeRF-360 implementation uses Python 3.9 and JAX. From the repository root:

```bash
cd /home/ubuntu/final_project/outdoor-nerf-depth/nerf-methods/mipnerf360
conda env create -f environment.yml
conda activate multinerf
```

If the environment already exists:

```bash
conda activate multinerf
```

The environment file installs the main dependencies, including `jax[cuda12]==0.4.30`, `flax`, `gin-config`, `opencv-python`, `tensorboard`, `tensorflow`, `lpips`, and `numpy<2`.

## 2. Check the Dataset

From the repository root, confirm that the local KITTI data exists:

```bash
ls data/kitti/kitti_select_static_5seq
```

Each KITTI sequence should contain folders like:

```text
images/
depths_gt/
depths_mff_crop/
depths_mono_crop/
depths_ste_conf_-1_crop/
sparse/0/
cameras.npz
```

The training script currently defaults to:

```bash
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt
```

To train on a different sequence, pass `DATA_DIR` when launching the script.

## 3. Start Training with MipNeRF-360

This is the recommended first training command for this project.

```bash
cd /home/ubuntu/final_project/outdoor-nerf-depth/nerf-methods/mipnerf360
bash scripts/train_kitti.sh
```

The script runs:

```bash
python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.max_steps = 75000" \
  --gin_bindings="Config.sample_every = 1" \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.compute_disp_metrics = True" \
  --gin_bindings="Config.depth_loss_type = 'mse'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/logs/checkpoints-1-7.5w-mse-debug'" \
  --logtostderr
```

Training outputs and checkpoints will be written inside:

```text
${DATA_DIR}/logs/checkpoints-1-7.5w-mse-debug
```

## 4. Train a Different KITTI Sequence

Set `DATA_DIR` before running the script:

```bash
cd /home/ubuntu/final_project/outdoor-nerf-depth/nerf-methods/mipnerf360

DATA_DIR=/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s2700_e3000_densegt \
bash scripts/train_kitti.sh
```

Available local KITTI sequences include:

```text
KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s657_e787_densegt
KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s890_e1028_densegt
KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s2700_e3000_densegt
KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt
KITTISeq05_2011_09_30_drive_0018_sync_llffdtu_s400_e725_densegt
```

## 5. Useful Training Parameters

The depth-supervised NeRF scripts expose several important settings:

| Parameter | Meaning | Common values |
| --- | --- | --- |
| `sample_every` | Training image sparsity. If `n`, use 1 image every `n` frames. | `1`, `2`, `4` |
| `depth_loss_type` | Depth loss type. | `mse`, `l1`, `kl` |
| `depth_sup_type` | Depth supervision source. | `gt`, `stereo_crop`, `mono_crop`, `mff_crop`, `rgbonly` |
| `lambda_depth` | Weight of the depth loss. | method-dependent |
| `max_steps` | Number of training steps. | `75000` for MipNeRF-360 default |
| `checkpoint_dir` | Output directory for checkpoints and predictions. | any writable path |

For MipNeRF-360, these parameters are passed through `--gin_bindings`. Example:

```bash
python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.max_steps = 75000" \
  --gin_bindings="Config.sample_every = 2" \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.compute_disp_metrics = True" \
  --gin_bindings="Config.lambda_depth = 10" \
  --gin_bindings="Config.depth_loss_type = 'mse'" \
  --gin_bindings="Config.depth_sup_type = 'mff_crop'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/logs/checkpoints-2-mse-mff-lambda10'" \
  --logtostderr
```

## 6. Monitor Training

The training script writes logs and checkpoints to the configured `checkpoint_dir`. To inspect TensorBoard logs:

```bash
tensorboard --logdir /home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/logs
```

Then open the TensorBoard URL printed in the terminal.


## 7. Handoff Paths And Sparse Runs

Current repo root:

```text
/home/ubuntu/final_project
```

Processed KITTI roots:

```text
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq_sparse_every2
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq_sparse_every4
```

Default sequence:

```text
KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt
```

Depth map folders for the default dense sequence include `depths_gt`, `depths_mff_crop`, `depths_mono_crop`, `depths_ste_conf_-1_crop`, `depths_da2`, and `depths_da2_npy` under:

```text
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt
```

Sparse every-2 MipNeRF-360 checkpoints/logs:

```text
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq_sparse_every2/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/logs/checkpoints-1-7.5w-mse-debug
```

Sparse every-4 MipNeRF-360 checkpoints/logs:

```text
/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq_sparse_every4/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/logs/checkpoints-sparse-every4-7.5w-mse
```

Exact sparse every-2 command/config:

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

Exact sparse every-4 command/config:

```bash
cd /home/ubuntu/final_project/outdoor-nerf-depth/nerf-methods/mipnerf360
conda activate multinerf
bash scripts/train_kitti_sparse.sh
```
