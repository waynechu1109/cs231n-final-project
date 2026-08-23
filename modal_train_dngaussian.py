"""
Modal training script for DNGaussian on KITTI sparse-every-2 data.
Camera-ready comparison against splatfacto-da2 + photometric masking (ours).

DNGaussian: Li et al. 2024 — depth-normalized 3DGS for few-shot view synthesis.
GitHub: https://github.com/Zhihao-Li/DNGaussian
Uses COLMAP sparse/ format for camera poses; depth maps at images/depth_maps/.

Prerequisites:
  The nerfstudio dataset must already be on the kitti-nerf-data volume at:
    kitti/kitti_select_static_5seq_sparse_every2/<kitti_seq_dir>/depths_gt/
    nerfstudio/kitti_seq02_0034_sparse_every2/  (transforms.json + images/)

Train (default: GT LiDAR depth, 30k iters):
  modal run modal_train_dngaussian.py::main

Evaluate a saved checkpoint:
  modal run modal_train_dngaussian.py::run_eval

Download results:
  modal volume get nerf-outputs dngaussian_kitti_seq02_0034_sparse_every2_gt_30000 ./local_outputs/dngaussian
"""

from __future__ import annotations

import re
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Shared volumes (same as other training scripts)
# ---------------------------------------------------------------------------
data_vol = modal.Volume.from_name("kitti-nerf-data", create_if_missing=True)
out_vol = modal.Volume.from_name("nerf-outputs", create_if_missing=True)

DATA_MOUNT = "/vol/data"
OUT_MOUNT = "/vol/outputs"

SCRIPTS_LOCAL = Path(__file__).parent / "scripts"

# ---------------------------------------------------------------------------
# Image — CUDA 11.8 devel + PyTorch 2.1.2 + DNGaussian submodules.
# All three CUDA extensions (diff-gaussian-rasterization, simple-knn,
# gridencoder) need nvcc from the devel image.
# Layer 1: system deps + torch (matches our splatfacto/dn-splatter base →
#          Modal reuses the layer cache).
# ---------------------------------------------------------------------------
_base = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git", "wget", "build-essential", "ninja-build", "cmake",
        "libgl1", "ffmpeg", "libglib2.0-0",
        "python3.10-dev",
    )
    .pip_install(
        "torch==2.1.2+cu118",
        "torchvision==0.16.2+cu118",
        "numpy<2.0.0",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu118",
    )
)

# Layer 2: clone DNGaussian (with submodules) and compile CUDA extensions.
# simple-knn is on GitHub (graphdeco-inria mirror), not the INRIA gitlab,
# so it clones without auth.  gridencoder is the ashawkey torch-ngp encoder.
_with_dngaussian = _base.run_commands(
    # Clone repo + all submodules in one shot
    "git clone --recursive https://github.com/Zhihao-Li/DNGaussian /opt/DNGaussian",
    # diff-gaussian-rasterization (ashawkey fork with depth rendering)
    "cd /opt/DNGaussian && TORCH_CUDA_ARCH_LIST='8.0 8.6' MAX_JOBS=4 "
    "pip install --no-cache-dir submodules/diff-gaussian-rasterization",
    # simple-knn
    "cd /opt/DNGaussian && pip install --no-cache-dir submodules/simple-knn",
    # gridencoder (hash-grid CUDA encoder)
    "cd /opt/DNGaussian && TORCH_CUDA_ARCH_LIST='8.0 8.6' MAX_JOBS=4 "
    "pip install --no-cache-dir submodules/gridencoder",
    # Python-only deps
    "pip install --no-cache-dir plyfile tqdm lpips scikit-image imageio opencv-python",
)

# Layer 3: mount our data-prep scripts
image = _with_dngaussian.add_local_dir(SCRIPTS_LOCAL, "/opt/project_scripts")

app = modal.App("dngaussian-kitti")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SEQ_DEFAULT = "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt"


def _slug(kitti_seq_dir: str) -> str:
    m = re.search(r"KITTISeq(\d+)_.*drive_(\d+)_sync", Path(kitti_seq_dir).name)
    if not m:
        raise ValueError(f"Cannot derive slug from: {kitti_seq_dir}")
    return f"kitti_seq{int(m.group(1)):02d}_{m.group(2)}"


def _exp_name(slug: str, depth_type: str, iters: int) -> str:
    return f"dngaussian_{slug}_sparse_every2_{depth_type}_{iters}"


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 6,
    volumes={DATA_MOUNT: data_vol, OUT_MOUNT: out_vol},
    env={"TORCHDYNAMO_DISABLE": "1", "MAX_JOBS": "1"},
)
def train(
    kitti_seq_dir: str = _SEQ_DEFAULT,
    depth_type: str = "gt",           # "gt" for LiDAR depths_gt; "da2" for DA-V2
    max_iterations: int = 30000,
    llffhold: int = 8,                # DNGaussian default; every llffhold-th frame is test
):
    """Convert nerfstudio KITTI dataset → COLMAP format, then run DNGaussian."""
    import os
    import subprocess

    slug = _slug(kitti_seq_dir)
    sparse_root = f"{DATA_MOUNT}/kitti/kitti_select_static_5seq_sparse_every2"
    seq_dir = f"{sparse_root}/{kitti_seq_dir}"
    nerfstudio_src = f"{DATA_MOUNT}/nerfstudio/{slug}_sparse_every2"
    exp_name = _exp_name(slug, depth_type, max_iterations)

    # Depth maps directory on the Modal volume
    if depth_type == "gt":
        depth_dir = f"{seq_dir}/depths_gt"
    else:
        depth_dir = f"{DATA_MOUNT}/da2_depths/{slug}"

    # COLMAP-format dataset (idempotent build)
    colmap_data = f"{DATA_MOUNT}/dngaussian/{slug}_sparse_every2_{depth_type}"

    for path, label in [
        (seq_dir, "KITTI sequence dir"),
        (nerfstudio_src, "nerfstudio source dir (transforms.json + images/)"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {label}:\n  {path}")

    if depth_type != "gt":
        # GT LiDAR is always available; warn if DA-V2 depths are missing
        if not os.path.exists(depth_dir):
            print(f"WARNING: DA-V2 depth dir not found: {depth_dir}")
            print("Training without depth maps.")
            depth_dir = None

    # Build COLMAP dataset if not already done
    if not os.path.exists(f"{colmap_data}/sparse/0/cameras.bin"):
        print(f"Building COLMAP dataset → {colmap_data}")
        cmd = [
            "python", "/opt/project_scripts/data_prep/make_colmap_kitti_dngaussian.py",
            "--src", nerfstudio_src,
            "--dst", colmap_data,
            "--overwrite",
        ]
        if depth_dir and os.path.exists(depth_dir):
            cmd += ["--depth-dir", depth_dir]
        subprocess.run(cmd, check=True)
        data_vol.commit()
    else:
        print(f"COLMAP dataset already exists: {colmap_data}")

    out_dir = f"{OUT_MOUNT}/{exp_name}"
    has_depths = os.path.isdir(f"{colmap_data}/images/depth_maps")

    # DNGaussian train.py arguments
    cmd = [
        "python", "/opt/DNGaussian/train.py",
        "-s", colmap_data,
        "-m", out_dir,
        "--eval",
        "--iterations", str(max_iterations),
        "--llffhold", str(llffhold),
    ]
    # DNGaussian automatically uses depth ranking loss when depth maps are present;
    # no explicit --use_depth_loss flag needed in standard builds.

    print("=" * 60)
    print(f"Exp name   : {exp_name}")
    print(f"Data dir   : {colmap_data}")
    print(f"Out dir    : {out_dir}")
    print(f"Depths     : {'yes (' + depth_dir + ')' if has_depths else 'no'}")
    print(f"llffhold   : {llffhold}  (every {llffhold}-th frame is test)")
    print(f"Iters      : {max_iterations}")
    print(f"Command:\n  " + " \\\n  ".join(cmd))
    print("=" * 60)

    subprocess.run(cmd, cwd="/opt/DNGaussian", check=True)
    out_vol.commit()
    print(f"\nOutputs saved to Modal volume 'nerf-outputs': {exp_name}/")


# ---------------------------------------------------------------------------
# eval  (render + metrics on a saved checkpoint)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 1,
    volumes={DATA_MOUNT: data_vol, OUT_MOUNT: out_vol},
    env={"TORCHDYNAMO_DISABLE": "1"},
)
def eval_run(exp_name: str) -> dict:
    import glob
    import json
    import os
    import subprocess

    out_dir = f"{OUT_MOUNT}/{exp_name}"
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Experiment dir not found: {out_dir}")

    # DNGaussian stores results in <out_dir>/results.json after --eval training,
    # OR we re-render with render.py + metrics.py (3DGS style).
    results_json = f"{out_dir}/results.json"

    if not os.path.exists(results_json):
        # Re-render test views then compute metrics
        print("Rendering test views ...")
        subprocess.run(
            ["python", "/opt/DNGaussian/render.py", "-m", out_dir, "--eval"],
            cwd="/opt/DNGaussian",
            check=True,
        )
        print("Computing metrics ...")
        subprocess.run(
            ["python", "/opt/DNGaussian/metrics.py", "-m", out_dir],
            cwd="/opt/DNGaussian",
            check=True,
        )
        out_vol.commit()

    if not os.path.exists(results_json):
        # Fallback: glob for any results JSON
        candidates = glob.glob(f"{out_dir}/**/results.json", recursive=True)
        if not candidates:
            raise FileNotFoundError(
                f"No results.json found under {out_dir}. "
                "Check training logs for metric output."
            )
        results_json = sorted(candidates)[-1]

    with open(results_json) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    kitti_seq_dir: str = _SEQ_DEFAULT,
    depth_type: str = "gt",
    max_iterations: int = 30000,
    llffhold: int = 8,
):
    train.remote(
        kitti_seq_dir=kitti_seq_dir,
        depth_type=depth_type,
        max_iterations=max_iterations,
        llffhold=llffhold,
    )


@app.local_entrypoint()
def run_eval(
    kitti_seq_dir: str = _SEQ_DEFAULT,
    depth_type: str = "gt",
    max_iterations: int = 30000,
):
    import json

    slug = _slug(kitti_seq_dir)
    exp_name = _exp_name(slug, depth_type, max_iterations)
    print(f"Evaluating: {exp_name}")
    metrics = eval_run.remote(exp_name=exp_name)
    print("\n========== DNGaussian Eval Results ==========")
    print(json.dumps(metrics, indent=2))
    print("=============================================")
