#!/bin/bash
set -e

cd ~/cs231n_project/code/cs231n-final-project/outdoor-nerf-depth/nerf-methods/mipnerf360
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multinerf

export REPO=/home/ubuntu/cs231n_project/code/cs231n-final-project
export DATA_DIR=$REPO/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt
export CKPT_DIR=$DATA_DIR/logs/checkpoints-sparseview-rgbonly-sampleevery2-50k

python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir='${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir='${CKPT_DIR}'" \
  --gin_bindings="Config.max_steps=50000" \
  --gin_bindings="Config.checkpoint_every=5000" \
  --gin_bindings="Config.train_render_every=50000" \
  --gin_bindings="Config.batch_size=4096" \
  --gin_bindings="Config.compute_disp_metrics=False" \
  --gin_bindings="Config.lambda_depth=0.0" \
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
