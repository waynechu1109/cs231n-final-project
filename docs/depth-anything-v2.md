# Depth Anything V2 Pipeline

This guide documents the Depth Anything V2 monocular depth supervision pipeline used for KITTI MipNeRF-360 experiments.

## Depth Anything V2 + MipNeRF-360 Pipeline

This section documents the Depth Anything V2 monocular depth supervision pipeline implemented for the KITTI MipNeRF-360 experiments.

Please do not overwrite existing checkpoint folders. Always use a new `Config.checkpoint_dir` for each run.

### Environments

Use `da2` for Depth Anything V2 inference:

```bash
conda activate da2
cd ~/final_project/Depth-Anything-V2
```

Check PyTorch GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Use `multinerf` for MipNeRF-360 training:

```bash
conda activate multinerf
cd ~/final_project/outdoor-nerf-depth/nerf-methods/mipnerf360
```

Check JAX GPU:

```bash
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

### Dataset

Current KITTI sequence:

```bash
~/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt
```

Important subfolders:

```text
images/          RGB images
depths_gt/       KITTI GT / dense GT depth
depths_da2_npy/  raw Depth Anything V2 predictions, saved as .npy
depths_da2/      scale-shift aligned DA2 depth, saved as KITTI-style 16-bit PNG
logs/            training outputs
```

### Generate DA2 raw depth

```bash
conda activate da2
cd ~/final_project/Depth-Anything-V2

python run_da2_save_npy.py \
  --img-dir ~/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/images \
  --out-dir ~/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/depths_da2_npy \
  --encoder vits \
  --checkpoint checkpoints/depth_anything_v2_vits.pth
```

Expected file count:

```bash
ls ~/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/depths_da2_npy | wc -l
```

Expected output:

```text
175
```

### Scale-shift align DA2 depth

Depth Anything V2 gives relative depth, so we align each prediction to KITTI depth:

```text
D_aligned = a * D_DA2 + b
```

Run:

```bash
conda activate da2
cd ~/final_project/Depth-Anything-V2

python align_da2_to_kitti.py \
  --da2-npy-dir ~/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/depths_da2_npy \
  --gt-depth-dir ~/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/depths_gt \
  --out-dir ~/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/depths_da2 \
  --max-depth 80
```

Current alignment result:

```text
Mean alignment Abs Error on valid LiDAR pixels: 4.2195 m
```

### Train DA2-supervised MipNeRF-360

Use tmux for long training:

```bash
tmux new -s da2_25k
conda activate multinerf
cd ~/final_project/outdoor-nerf-depth/nerf-methods/mipnerf360
```

Run 25k training:

```bash
python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir='/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt'" \
  --gin_bindings="Config.checkpoint_dir='/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/logs/checkpoints-da2-vits-lambda005-25k'" \
  --gin_bindings="Config.max_steps=25000" \
  --gin_bindings="Config.checkpoint_every=25000" \
  --gin_bindings="Config.batch_size=4096" \
  --gin_bindings="Config.compute_disp_metrics=True" \
  --gin_bindings="Config.depth_sup_type='da2'" \
  --gin_bindings="Config.depth_loss_type='mse'" \
  --gin_bindings="Config.lambda_depth=0.05" \
  --gin_bindings="Config.sample_every=1" \
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

Detach from tmux:

```text
Ctrl+b, then d
```

Reattach:

```bash
tmux attach -t da2_25k
```

### Current 25k results

GT depth supervision:

```text
PSNR   = 20.85
AbsRel = 0.091
RMSE   = 2.46
```

Scale-aligned DA2 supervision, lambda=0.05:

```text
PSNR   = 20.45
AbsRel = 0.108
RMSE   = 3.29
```

Observation: scale-aligned DA2 recovers much of the benefit of GT depth supervision, but depth metrics are worse because monocular depth is noisy.

### Sparse experiments

For sparse depth experiments, compare methods under the same sparse supervision budget.

Recommended comparisons:

```text
1. RGB-only baseline
2. Sparse GT depth supervision
3. Sparse scale-aligned DA2 depth supervision
4. Sparse / masked DA2 depth supervision
5. Dense GT depth supervision as upper bound
```

Quick sparse experiment using built-in random sparse mask:

```bash
--gin_bindings="Config.depth_keep_ratio=0.05"
```

Sparse GT example:

```bash
--gin_bindings="Config.depth_sup_type='gt'" \
--gin_bindings="Config.depth_keep_ratio=0.05" \
--gin_bindings="Config.checkpoint_dir='.../logs/checkpoints-sparsegt-r005-25k'"
```

Sparse DA2 example:

```bash
--gin_bindings="Config.depth_sup_type='da2'" \
--gin_bindings="Config.depth_keep_ratio=0.05" \
--gin_bindings="Config.checkpoint_dir='.../logs/checkpoints-sparseda2-r005-25k'"
```

For a stricter fair comparison, sparse GT and sparse DA2 should use the exact same pixel mask. This can be done by creating offline folders such as:

```text
depths_sparse_gt_r005/
depths_sparse_da2_r005/
```

Then train with:

```bash
--gin_bindings="Config.depth_sup_type='sparse_gt_r005'"
--gin_bindings="Config.depth_sup_type='sparse_da2_r005'"
```

### Read metrics

After training, metrics are under:

```text
logs/<checkpoint_dir>/test_preds_<step>/
```

Read mean metrics from the last line:

```bash
tail -n 1 <test_preds_dir>/metric_psnr_<step>.txt
tail -n 1 <test_preds_dir>/metric_absrel_<step>.txt
tail -n 1 <test_preds_dir>/metric_rmse_<step>.txt
```
