# Run from mipnerf360 root so `python -m train` can find train.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.." || exit 1
# Local KITTI data (download from project README if missing)
DATA_DIR="${DATA_DIR:-/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt}"

python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.max_steps = 75000" \
  --gin_bindings="Config.sample_every = 1" \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.compute_disp_metrics = True" \
  --gin_bindings="Config.depth_loss_type = 'mse'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/logs/checkpoints-1-7.5w-mse-debug'" \
  --logtostderr