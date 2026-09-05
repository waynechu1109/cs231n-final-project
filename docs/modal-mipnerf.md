# Modal Cloud Training — Mip-NeRF-360

JAX/Flax-based Mip-NeRF-360 training on Modal (A10G) via `modal_train_mipnerf.py`.

## Setup

```bash
pip install modal
modal setup          # one-time auth
```

Prerequisite: `depths_da2` must already exist in `kitti-nerf-data` for the target sequence. Run `modal_train_splatfacto.py::run_new_seq_experiments` first to populate it — see [modal-splatfacto.md](modal-splatfacto.md).

## Three-experiment generalization check (new KITTI sequence)

Runs all three key experiments in one command: RGB-only → Global depth → Low-error mask (τ=0.18, λ=0.15).

```bash
modal run modal_train_mipnerf.py::run_kitti_seq_experiments \
  --kitti-seq-dir "KITTISeq05_2011_09_30_drive_0018_sync_llffdtu_s400_e725_densegt" \
  --lambda-depth 0.15 \
  --ref-threshold 0.18
```

Experiment names produced:

- `kitti_seq05_0018_sparse_every2_da2_lambda0.0_nomask_50000` (RGB-only)
- `kitti_seq05_0018_sparse_every2_da2_lambda0.15_nomask_50000` (Global depth)
- `kitti_seq05_0018_sparse_every2_da2_lambda0.15_low018_50000` (Low-error mask)

## Evaluation

```bash
# RGB-only
modal run modal_train_mipnerf.py::run_eval \
  --kitti-seq-dir "KITTISeq05_..." --lambda-depth 0.0

# Global depth
modal run modal_train_mipnerf.py::run_eval \
  --kitti-seq-dir "KITTISeq05_..." --lambda-depth 0.15

# Low-error mask (add --masked)
modal run modal_train_mipnerf.py::run_eval \
  --kitti-seq-dir "KITTISeq05_..." --lambda-depth 0.15 --masked
```

## Download outputs

```bash
modal volume get nerf-outputs <exp_name>/mipnerf360 ./local_mipnerf_ckpt
```
