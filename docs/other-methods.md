# Other Supported Methods

This guide summarizes the NeRF++ and Instant-NGP Depth entry points included in the workspace.

## 8. Other Supported Methods

### NeRF++

Environment:

```bash
cd /home/ubuntu/final_project/outdoor-nerf-depth/nerf-methods/nerfplusplus
conda env create -f environment.yml
conda activate nerfplusplus
```

Basic training command:

```bash
python ddp_train_nerf.py --config configs/kitti.txt
```

Depth-supervised training is controlled with flags such as:

```bash
python ddp_train_nerf.py --config configs/kitti.txt \
  --depth_loss_type mse \
  --depth_sup_type mff_crop \
  --lambda_depth 0.1 \
  --trainskip 4 \
  --scene KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt \
  --basedir /home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/logs/nerfplusplus \
  --expname mse0.1_mff_crop_4 \
  --N_iters 100001 \
  --use_depth
```

There is also a batch script:

```bash
bash scripts/train.sh
```

Before using it, update `DATA_ROOT` and `SAVE_ROOT` inside the script because the original paths point to the authors' machine.

### Instant-NGP Depth

Install the Python dependencies after preparing a compatible PyTorch/CUDA environment:

```bash
cd /home/ubuntu/final_project/outdoor-nerf-depth/nerf-methods/ngp-depth
pip install -r requirements.txt
```

Single-scene training follows this pattern:

```bash
python train.py \
  --root_dir /home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt \
  --scale 10 \
  --mod_ratio 1 \
  --eval_lpips \
  --depth_loss_w 1 \
  --depth_folder depths_mff_crop \
  --check_val_every_n_epoch 1 \
  --batch_size 15000 \
  --exp_name kitti_seq02_mff_scale10_depth1
```

The batch script is:

```bash
bash auto_batch_run_kittiseq.sh
```

Before using it, update `ROOT` inside the script because the original path is not this workspace path.
