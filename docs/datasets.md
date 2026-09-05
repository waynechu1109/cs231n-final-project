# Datasets

## KITTI Odometry

KITTI comprises large-scale real-world outdoor driving scenes with a forward-facing camera trajectory. We use selected odometry sequences 00, 02, and 05, following the evaluation protocol of prior outdoor NeRF work. For each sequence, every 10th frame is held for the test set and the remaining frames are used for training. To assess robustness under sparse viewpoints, we simulate low-frequency imaging at 2.5 Hz by subsampling 50% of the training frames (retaining every second frame).

- **Dense** — all training frames
- **Sparse every-2** — 50% subsampled (simulates 2.5 Hz capture)

Default sequence used in most experiments:

```
KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt
```

Fragment 034 spans 88 frames (every 2nd of a 175-frame raw sequence) with 8 test frames at the every-10th sparse index; fragments 018 (Seq05) and 027 (Seq00) follow the identical protocol.

## Mip-NeRF-360 Bicycle

194 sparse views of an outdoor object-centric scene. Every 10th frame held for validation. Images downscaled 4× for 3DGS training; depth priors computed at full resolution and resized at load time.

## Data Paths (remote training server)

| Data | Path |
|---|---|
| Dense KITTI (5 seq) | `/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq` |
| Sparse every-2 | `/home/ubuntu/final_project/data/kitti/kitti_select_static_5seq_sparse_every2` |
| Nerfstudio KITTI seq02 | `/home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034` |
| Nerfstudio sparse every-2 | `/home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034_sparse_every2` |
| DA-V2 depth maps | `<seq_dir>/depths_da2` and `<seq_dir>/depths_da2_npy` |
