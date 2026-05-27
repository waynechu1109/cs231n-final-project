# Nerfstudio Splatfacto Training

This guide covers dense and sparse nerfstudio KITTI datasets and `splatfacto` training/evaluation/export commands.

## 7. Train 3DGS with Nerfstudio Splatfacto

This workspace also includes `nerfstudio`, which can train 3D Gaussian Splatting through the `splatfacto` method. The important detail is that nerfstudio does not directly consume the original MipNeRF-360 KITTI folder in the same way. It expects a nerfstudio dataset with a `transforms.json` file.

### Prepared Nerfstudio Dataset

The KITTI sequence used by `scripts/train_kitti.sh` has already been converted for nerfstudio here:

```text
/home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034
```

This folder contains:

```text
images -> /home/ubuntu/final_project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/images
transforms.json
```

The `transforms.json` file tells nerfstudio:

- where each image is located,
- each image's camera pose,
- camera intrinsics such as focal length and principal point,
- the exact train/validation/test split.

The split was written explicitly so `splatfacto` uses the same held-out images as the MipNeRF-360 KITTI training run. The validation/test frames are:

```text
00000009.png, 00000019.png, 00000029.png, 00000039.png, 00000049.png,
00000059.png, 00000069.png, 00000079.png, 00000089.png, 00000099.png,
00000109.png, 00000119.png, 00000129.png, 00000139.png, 00000149.png,
00000159.png, 00000169.png
```

This gives:

```text
175 total frames
158 train frames
17 validation frames
17 test frames
```

When nerfstudio starts correctly, the log should include:

```text
Dataset is overriding train_indices ...
Dataset is overriding val_indices to [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149, 159, 169]
```

That message confirms that the custom split in `transforms.json` is being used.

### Prepared Sparse Nerfstudio Dataset

For sparse `splatfacto` training, use the nerfstudio-formatted sparse dataset here:

```text
/home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034_sparse_every4
```

This dataset is derived from the dense nerfstudio `transforms.json`, but only keeps every 4th KITTI frame. Its `images` entry is a symlink to the sparse KITTI image folder:

```text
images -> /home/ubuntu/final_project/data/kitti/kitti_select_static_5seq_sparse_every4/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/images
transforms.json
SPARSE_EVERY4_MANIFEST.txt
```

The sparse nerfstudio dataset was created with:

```bash
cd /home/ubuntu/final_project
python scripts/make_nerfstudio_kitti_sparse.py
```

The script does the following:

- reads `data/nerfstudio/kitti_seq02_0034/transforms.json`,
- keeps frames whose numeric filename is divisible by 4,
- writes a new sparse `transforms.json`,
- symlinks `images` to the already prepared sparse KITTI images,
- writes `SPARSE_EVERY4_MANIFEST.txt` with the split summary.

The resulting split is:

```text
44 total frames
40 train frames
4 validation frames
4 test frames
```

The validation/test frames are aligned with the sparse MipNeRF split:

```text
00000036.png, 00000076.png, 00000116.png, 00000156.png
```


### Prepared Sparse Every-2 Nerfstudio Dataset

The sparse every-2 dataset for `splatfacto` is here:

```text
/home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034_sparse_every2
```

It contains:

```text
images -> /home/ubuntu/final_project/data/kitti/kitti_select_static_5seq_sparse_every2/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/images
transforms.json
SPARSE_EVERY2_MANIFEST.txt
```

The split is:

```text
88 total frames
71 train frames
9 validation frames
8 test frames
```

The validation frames are interleaved halfway between the held-out test frames:

```text
00000008.png, 00000028.png, 00000048.png, 00000068.png, 00000088.png,
00000108.png, 00000128.png, 00000148.png, 00000168.png
```

The test frames keep the MipNeRF every-10 held-out rule:

```text
00000018.png, 00000038.png, 00000058.png, 00000078.png,
00000098.png, 00000118.png, 00000138.png, 00000158.png
```

Train with:

```bash
cd /home/ubuntu/final_project/nerfstudio
conda activate nerfstudio

ns-train splatfacto \
  --data /home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034_sparse_every2 \
  --vis tensorboard
```

Previous sparse `splatfacto` outputs are under:

```text
/home/ubuntu/final_project/nerfstudio/outputs/kitti_seq02_0034_sparse_every2/splatfacto
/home/ubuntu/final_project/nerfstudio/outputs/kitti_seq02_0034_sparse_every4/splatfacto
```

### Nerfstudio Environment

Activate the nerfstudio environment:

```bash
conda activate nerfstudio
cd /home/ubuntu/final_project/nerfstudio
```

If `ns-train` is not available, install the local nerfstudio package:

```bash
cd /home/ubuntu/final_project/nerfstudio
pip install -e .
```

### AWS/CUDA Compile Setup

The first `splatfacto` run compiles CUDA extensions from `gsplat`. On AWS machines this can fail or even make the instance appear to crash if too many compiler jobs run in parallel. Use this conservative setup before launching training:

```bash
conda activate nerfstudio
cd /home/ubuntu/final_project/nerfstudio

# Make Python/Triton/CUDA extension builds find conda headers such as crypt.h.
export CPATH=$CONDA_PREFIX/include:$CPATH
export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH

# Use the conda GCC/G++ 11 toolchain, which is compatible with CUDA 11.8.
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export TORCH_CUDA_ARCH_LIST="8.6"

# Reduce compile-time memory pressure and avoid PyTorch inductor/Triton crashes.
export MAX_JOBS=1
export TORCHDYNAMO_DISABLE=1

# Clear stale failed extension builds.
rm -rf ~/.cache/torch_extensions/py38_cu118/gsplat_cuda
rm -rf /tmp/torchinductor_ubuntu
```

If `$CONDA_PREFIX/include/crypt.h` is missing, install `libxcrypt`:

```bash
conda install -c conda-forge libxcrypt
ls $CONDA_PREFIX/include/crypt.h
```

If CUDA compilation complains that GCC is unsupported, make sure the conda compiler packages are installed:

```bash
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11
```

### Start 3DGS Training

Recommended stable launch command:

```bash
cd /home/ubuntu/final_project/nerfstudio

ns-train splatfacto \
  --data /home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034_sparse_every2 \
  --vis tensorboard
```

For sparse training, point `--data` to the sparse nerfstudio dataset:

```bash
cd /home/ubuntu/final_project/nerfstudio

ns-train splatfacto \
  --data /home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034_sparse_every4 \
  --vis tensorboard
```

Using `--vis tensorboard` avoids opening the viewer-only mode and enables eval logging. If you want the interactive viewer, use:

```bash
ns-train splatfacto \
  --data /home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034 \
  --vis viewer+tensorboard
```

Outputs are written under:

```text
/home/ubuntu/final_project/nerfstudio/outputs/kitti_seq02_0034/splatfacto/<timestamp>/
```

Checkpoints are saved under:

```text
/home/ubuntu/final_project/nerfstudio/outputs/kitti_seq02_0034/splatfacto/<timestamp>/nerfstudio_models
```

### Optional: Put Outputs on the Extra NVMe Disk

The root disk has enough space in the current AWS setup, but the instance also has an ephemeral disk mounted at `/opt/dlami/nvme`. To store training outputs there:

```bash
ns-train splatfacto \
  --data /home/ubuntu/final_project/data/nerfstudio/kitti_seq02_0034 \
  --output-dir /opt/dlami/nvme/nerfstudio_outputs \
  --vis tensorboard
```

### Evaluate a Finished Splatfacto Run

After training, evaluate with the saved config:

```bash
cd /home/ubuntu/final_project/nerfstudio

ns-eval \
  --load-config outputs/kitti_seq02_0034/splatfacto/<timestamp>/config.yml \
  --output-path eval_splatfacto_kitti.json
```

If outputs were written to `/opt/dlami/nvme`, point `--load-config` to that path instead.

### Export 3DGS and Point Cloud

If the model was trained with `splatfacto`, the trained checkpoint is already a 3D Gaussian Splatting model. To export it as a portable 3DGS `.ply`, use `ns-export gaussian-splat`:

```bash
cd /home/ubuntu/final_project/nerfstudio

ns-export gaussian-splat \
  --load-config outputs/kitti_seq02_0034/splatfacto/<timestamp>/config.yml \
  --output-dir exports/kitti_seq02_0034_splat
```

The exported file is usually:

```text
exports/kitti_seq02_0034_splat/splat.ply
```

This `splat.ply` contains the Gaussian parameters, including positions, scales, rotations, opacities, and color coefficients. Use this file with viewers or tools that support 3DGS `.ply` files.

To export a regular point cloud instead:

```bash
cd /home/ubuntu/final_project/nerfstudio

ns-export pointcloud \
  --load-config outputs/kitti_seq02_0034/splatfacto/<timestamp>/config.yml \
  --output-dir exports/kitti_seq02_0034_pointcloud \
  --num-points 1000000 \
  --normal-method open3d \
  --num-rays-per-batch 4096
```

The point cloud output is usually:

```text
exports/kitti_seq02_0034_pointcloud/point_cloud.ply
```

If export runs out of GPU memory, lower `--num-rays-per-batch`, for example to `2048` or `1024`.

Do not use the point cloud as a replacement for the Gaussian splat export. A point cloud stores sampled geometry and colors, while the Gaussian splat export stores the trained 3DGS representation itself.

## View Training Results Through Tensorboard on Local Machine
```bash
# run on local!
ssh -L 6006:localhost:6006 cs231n-final-project \
  "cd /home/ubuntu/final_project/nerfstudio && source ~/miniconda3/etc/profile.d/conda.sh && conda activate nerfstudio && tensorboard --logdir outputs/kitti_seq02_0034_sparse_every2/splatfacto --port 6006"
```

### Notes

- `tiny-cuda-nn` is not required for `splatfacto`; it is mainly relevant for NeRF/hash-grid methods.
- If the goal is a 3DGS asset, export with `ns-export gaussian-splat`; do not convert through `ns-export pointcloud`.
- This converted KITTI sequence has no usable `points3D.txt` points, so nerfstudio prints `load_3D_points set to true but no point cloud found`. That is expected. `splatfacto` falls back to random initialization.
- If the machine freezes during the first run, it is usually CUDA extension compilation memory pressure. Keep `MAX_JOBS=1`, `TORCHDYNAMO_DISABLE=1`, and close extra processes. The first successful compile is cached.
