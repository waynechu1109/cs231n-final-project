"""
Modal training script for Mip-NeRF-360 on KITTI sequences.

Three-experiment pipeline: RGB-only → Global depth → Low-error masked depth.

Prerequisite: depths_da2 must already exist in the kitti-nerf-data volume for the
target sequence. Run the Splatfacto pipeline first to populate it:
  modal run modal_train_splatfacto.py::run_new_seq_experiments \\
    --kitti-seq-dir "KITTISeq05_2011_09_30_drive_0018_sync_llffdtu_s400_e725_densegt"

Quick start (all three experiments on Seq05):
  modal run modal_train_mipnerf.py::run_kitti_seq_experiments \\
    --kitti-seq-dir "KITTISeq05_2011_09_30_drive_0018_sync_llffdtu_s400_e725_densegt"

Eval:
  modal run modal_train_mipnerf.py::run_eval \\
    --kitti-seq-dir "KITTISeq05_..." --lambda-depth 0.0

  modal run modal_train_mipnerf.py::run_eval \\
    --kitti-seq-dir "KITTISeq05_..." --lambda-depth 0.15

  modal run modal_train_mipnerf.py::run_eval \\
    --kitti-seq-dir "KITTISeq05_..." --lambda-depth 0.15 --masked

Download checkpoints:
  modal volume get nerf-outputs <exp_name>/mipnerf360 ./local_mipnerf_ckpt
"""

from __future__ import annotations

import re
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent
MIPNERF_LOCAL = PROJECT_ROOT / "outdoor-nerf-depth" / "nerf-methods" / "mipnerf360"

data_vol = modal.Volume.from_name("kitti-nerf-data", create_if_missing=True)
out_vol = modal.Volume.from_name("nerf-outputs", create_if_missing=True)

DATA_MOUNT = "/vol/data"
OUT_MOUNT = "/vol/outputs"

# ---------------------------------------------------------------------------
# Image — JAX + CUDA 11.8 + Mip-NeRF-360 dependencies
# ---------------------------------------------------------------------------
# Constraints:
#   flax < 0.7.0      →  flax.training.checkpoints still exists (removed in 0.7)
#   jax == 0.4.14     →  has define_bool_state; last jax with cuda11_pip support
#   tensorflow-cpu    →  needed by flax.metrics.tensorboard backend
#
# Install strategy: install CUDA jaxlib first, then lock it via a pip constraint
# file so subsequent packages (tensorflow-cpu, orbax, etc.) cannot upgrade it to
# the incompatible PyPI-latest jaxlib (0.6.x).
JAX_CUDA_URL = "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .run_commands(
        # Step 1: install JAX 0.4.14 + matching CUDA jaxlib from the releases URL.
        f"pip install --no-cache-dir 'jax==0.4.14' 'jaxlib==0.4.14+cuda11.cudnn86'"
        f" -f {JAX_CUDA_URL}",
        # Step 2: install remaining deps.  Re-pin jaxlib here so pip cannot
        # upgrade it to satisfy a transitive dep (the +cuda11 local-version tag
        # is invisible to PyPI resolvers, so we must point at the find-links URL
        # again to keep the pin resolvable).
        f"pip install --no-cache-dir"
        "  flax==0.6.11 optax==0.1.7 chex==0.1.7 dm-pix==0.4.0 gin-config==0.5.0"
        "  absl-py 'numpy<2.0' Pillow scikit-image lpips==0.1.4 rawpy mediapy"
        "  orbax-checkpoint==0.3.5 'tensorflow-cpu==2.13.0' 'tensorboard==2.13.0'"
        "  opencv-python-headless matplotlib plyfile imageio 'scipy<1.13'"
        f"  'jaxlib==0.4.14+cuda11.cudnn86' -f {JAX_CUDA_URL}",
    )
    .add_local_dir(MIPNERF_LOCAL, "/opt/mipnerf", copy=True)
    # pycolmap is a local pure-Python package bundled with mipnerf360 —
    # it must be installed after the source tree is mounted.
    .run_commands("pip install --no-cache-dir /opt/mipnerf/internal/pycolmap")
)

app = modal.App("mipnerf-kitti")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_slug(kitti_seq_dir: str) -> str:
    m = re.search(r"KITTISeq(\d+)_.*drive_(\d+)_sync", Path(kitti_seq_dir).name)
    if not m:
        raise ValueError(f"Cannot derive slug from: {kitti_seq_dir}")
    return f"kitti_seq{int(m.group(1)):02d}_{m.group(2)}"


def _exp_name(
    slug: str,
    lambda_depth: float,
    max_steps: int,
    *,
    mask_label: str = "nomask",
    depth_sup_type: str = "da2",
) -> str:
    return f"{slug}_sparse_every2_{depth_sup_type}_lambda{lambda_depth}_{mask_label}_{max_steps}"


def _gin_flags(
    data_dir: str,
    ckpt_dir: str,
    lambda_depth: float,
    max_steps: int,
    *,
    depth_sup_type: str = "da2",
    depth_loss_type: str = "mse",
    sample_every: int = 2,
    hold_every: int = 10,
    compute_disp: bool = True,
    fixed_photo_mask_dir: str = "",
    photo_mask_threshold: float = 0.18,
    photo_mask_mode: str = "low",
) -> list[str]:
    """Gin binding flags shared across train / mask-gen / eval."""
    flags = [
        "--gin_configs=configs/360.gin",
        f"--gin_bindings=Config.data_dir='{data_dir}'",
        f"--gin_bindings=Config.checkpoint_dir='{ckpt_dir}'",
        f"--gin_bindings=Config.max_steps={max_steps}",
        f"--gin_bindings=Config.batch_size=4096",
        f"--gin_bindings=Config.compute_disp_metrics={'True' if compute_disp else 'False'}",
        f"--gin_bindings=Config.depth_sup_type='{depth_sup_type}'",
        f"--gin_bindings=Config.depth_loss_type='{depth_loss_type}'",
        f"--gin_bindings=Config.depth_keep_ratio=0.0",
        f"--gin_bindings=Config.lambda_depth={lambda_depth}",
        f"--gin_bindings=Config.sample_every={sample_every}",
        f"--gin_bindings=Config.llffhold={hold_every}",
        f"--gin_bindings=Config.auto_adjust_near_far=True",
        f"--gin_bindings=Config.near=0.2",
        f"--gin_bindings=Config.far=1000000.0",
        f"--gin_bindings=Model.opaque_background=True",
        f"--gin_bindings=Model.raydist_fn=@jnp.reciprocal",
        f"--gin_bindings=NerfMLP.disable_density_normals=True",
        f"--gin_bindings=NerfMLP.net_depth=8",
        f"--gin_bindings=NerfMLP.net_width=1024",
        f"--gin_bindings=NerfMLP.warp_fn=@coord.contract",
        f"--gin_bindings=PropMLP.disable_density_normals=True",
        f"--gin_bindings=PropMLP.disable_rgb=True",
        f"--gin_bindings=PropMLP.net_depth=4",
        f"--gin_bindings=PropMLP.net_width=256",
        f"--gin_bindings=PropMLP.warp_fn=@coord.contract",
    ]
    if fixed_photo_mask_dir:
        flags += [
            f"--gin_bindings=Config.fixed_photo_mask_dir='{fixed_photo_mask_dir}'",
            f"--gin_bindings=Config.photo_mask_threshold={photo_mask_threshold}",
            f"--gin_bindings=Config.photo_mask_mode='{photo_mask_mode}'",
        ]
    return flags


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 12,
    volumes={DATA_MOUNT: data_vol, OUT_MOUNT: out_vol},
)
def train_mipnerf(
    kitti_seq_dir: str,
    lambda_depth: float = 0.0,
    max_steps: int = 50000,
    fixed_photo_mask_dir: str = "",
    photo_mask_threshold: float = 0.18,
    photo_mask_mode: str = "low",
    mask_label: str = "nomask",
    depth_loss_type: str = "mse",
    depth_sup_type: str = "da2",
    hold_every: int = 10,
):
    import os
    import subprocess

    slug = _derive_slug(kitti_seq_dir)
    sparse_root = f"{DATA_MOUNT}/kitti/kitti_select_static_5seq_sparse_every2"
    data_dir = f"{sparse_root}/{kitti_seq_dir}"
    ckpt_dir = (
        f"{OUT_MOUNT}/"
        f"{_exp_name(slug, lambda_depth, max_steps, mask_label=mask_label, depth_sup_type=depth_sup_type)}"
        f"/mipnerf360"
    )

    for p, label in [
        (data_dir, "KITTI sequence dir"),
        (f"{data_dir}/depths_{depth_sup_type}", "depth supervision folder"),
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {label}: {p}")

    if fixed_photo_mask_dir and not os.path.isdir(fixed_photo_mask_dir):
        raise FileNotFoundError(f"photo_mask_dir does not exist: {fixed_photo_mask_dir}")

    os.makedirs(ckpt_dir, exist_ok=True)

    cmd = [
        "python", "-m", "train",
        *_gin_flags(
            data_dir, ckpt_dir, lambda_depth, max_steps,
            depth_sup_type=depth_sup_type,
            depth_loss_type=depth_loss_type,
            sample_every=2,
            hold_every=hold_every,
            fixed_photo_mask_dir=fixed_photo_mask_dir,
            photo_mask_threshold=photo_mask_threshold,
            photo_mask_mode=photo_mask_mode,
        ),
        "--gin_bindings=Config.checkpoint_every=25000",
        "--logtostderr",
    ]

    print("=" * 60)
    print(f"Data dir:     {data_dir}")
    print(f"Ckpt dir:     {ckpt_dir}")
    print(f"lambda_depth: {lambda_depth}")
    print(f"mask_label:   {mask_label}")
    print("=" * 60)

    subprocess.run(cmd, check=True, cwd="/opt/mipnerf")
    out_vol.commit()
    print(f"\nOutputs saved to 'nerf-outputs' under: {ckpt_dir}/")


# ---------------------------------------------------------------------------
# Mask generation (from RGB-only checkpoint)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 2,
    volumes={DATA_MOUNT: data_vol, OUT_MOUNT: out_vol},
)
def generate_mipnerf_masks(
    kitti_seq_dir: str,
    base_exp_name: str,
    threshold: float = 0.18,
    photo_mask_mode: str = "low",
) -> str:
    """Render the RGB-only checkpoint and threshold photometric error to produce masks.

    Returns the mask directory path on the data volume.
    """
    import os
    import subprocess

    sparse_root = f"{DATA_MOUNT}/kitti/kitti_select_static_5seq_sparse_every2"
    data_dir = f"{sparse_root}/{kitti_seq_dir}"
    ckpt_dir = f"{OUT_MOUNT}/{base_exp_name}/mipnerf360"

    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"RGB-only checkpoint not found: {ckpt_dir}")

    thresh_tag = f"{threshold:.2f}".replace(".", "")
    mask_dir = f"{DATA_MOUNT}/masks/{base_exp_name}_{photo_mask_mode}{thresh_tag}"
    os.makedirs(mask_dir, exist_ok=True)

    cmd = [
        "python", "generate_fixed_photo_masks.py",
        *_gin_flags(
            data_dir, ckpt_dir, 0.0, 50000,
            compute_disp=False,
            sample_every=2,
            hold_every=10,
            fixed_photo_mask_dir=mask_dir,
            photo_mask_threshold=threshold,
            photo_mask_mode=photo_mask_mode,
        ),
        "--logtostderr",
    ]

    print(f"Generating {photo_mask_mode}{thresh_tag} masks → {mask_dir}")
    subprocess.run(cmd, check=True, cwd="/opt/mipnerf")
    data_vol.commit()
    print(f"Masks written to: {mask_dir}")
    return mask_dir


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 3,
    volumes={DATA_MOUNT: data_vol, OUT_MOUNT: out_vol},
)
def eval_mipnerf(
    kitti_seq_dir: str,
    exp_name: str,
    lambda_depth: float = 0.0,
    max_steps: int = 50000,
    depth_sup_type: str = "da2",
) -> dict:
    """Run Mip-NeRF eval once and collect metrics from the written txt files."""
    import glob
    import os
    import subprocess

    sparse_root = f"{DATA_MOUNT}/kitti/kitti_select_static_5seq_sparse_every2"
    data_dir = f"{sparse_root}/{kitti_seq_dir}"
    ckpt_dir = f"{OUT_MOUNT}/{exp_name}/mipnerf360"

    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    # eval_only_once and eval_save_output both default to True in configs.py.
    cmd = [
        "python", "-m", "eval",
        *_gin_flags(
            data_dir, ckpt_dir, lambda_depth, max_steps,
            depth_sup_type=depth_sup_type,
            sample_every=2,
            hold_every=10,
        ),
        "--logtostderr",
    ]

    print(f"Evaluating: {exp_name}")
    subprocess.run(cmd, check=True, cwd="/opt/mipnerf")
    out_vol.commit()

    # Collect per-metric txt files written by eval.py into test_eval_preds_/
    eval_dirs = sorted(glob.glob(f"{ckpt_dir}/test_eval_preds*"))
    if not eval_dirs:
        print("Warning: no eval output directory found")
        return {}

    eval_out = eval_dirs[-1]
    metrics: dict = {}
    for txt in sorted(glob.glob(f"{eval_out}/metric_*.txt")):
        stem = Path(txt).stem          # e.g. "metric_psnr_50000"
        name = stem.replace("metric_", "").rsplit("_", 1)[0]   # → "psnr"
        with open(txt) as f:
            lines = [l for l in f.read().strip().split("\n") if l]
        if lines:
            try:
                metrics[name] = float(lines[-1])   # last line is the mean
            except ValueError:
                pass

    return metrics


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run_kitti_seq_experiments(
    kitti_seq_dir: str = "KITTISeq05_2011_09_30_drive_0018_sync_llffdtu_s400_e725_densegt",
    lambda_depth: float = 0.15,
    ref_threshold: float = 0.18,
    depth_loss_type: str = "mse",
    depth_sup_type: str = "da2",
    max_num_iterations: int = 50000,
):
    """Three-experiment Mip-NeRF ablation on a KITTI sequence.

    Stage 1: RGB-only (lambda=0) and Global-depth (lambda) in parallel.
    Stage 2: Generate low018 masks from the RGB-only checkpoint.
    Stage 3: Train Low-error masked run (lambda, τ=ref_threshold).

    Experiment names produced:
      <slug>_sparse_every2_da2_lambda0.0_nomask_<iters>   (RGB-only)
      <slug>_sparse_every2_da2_lambda<l>_nomask_<iters>   (Global depth)
      <slug>_sparse_every2_da2_lambda<l>_low018_<iters>   (Low-error mask)
    """
    slug = _derive_slug(kitti_seq_dir)
    rgbonly_exp = _exp_name(slug, 0.0, max_num_iterations, mask_label="nomask", depth_sup_type=depth_sup_type)
    global_exp = _exp_name(slug, lambda_depth, max_num_iterations, mask_label="nomask", depth_sup_type=depth_sup_type)
    thresh_tag = f"{ref_threshold:.2f}".replace(".", "")
    mask_label = f"low{thresh_tag}"
    masked_exp = _exp_name(slug, lambda_depth, max_num_iterations, mask_label=mask_label, depth_sup_type=depth_sup_type)

    print(f"\n=== Mip-NeRF KITTI experiments: {kitti_seq_dir} ===")
    print(f"  RGB-only:     {rgbonly_exp}")
    print(f"  Global depth: {global_exp}")
    print(f"  Masked:       {masked_exp}")

    # Stage 1: RGB-only and Global depth in parallel
    print("\n[Stage 1] Training RGB-only and Global-depth in parallel...")
    for _ in train_mipnerf.starmap([
        (kitti_seq_dir, 0.0,         max_num_iterations, "", ref_threshold, "low", "nomask", depth_loss_type, depth_sup_type, 10),
        (kitti_seq_dir, lambda_depth, max_num_iterations, "", ref_threshold, "low", "nomask", depth_loss_type, depth_sup_type, 10),
    ]):
        pass

    # Stage 2: generate masks from RGB-only checkpoint
    print(f"\n[Stage 2] Generating low{thresh_tag} masks from RGB-only checkpoint...")
    mask_dir = generate_mipnerf_masks.remote(kitti_seq_dir, rgbonly_exp, ref_threshold, "low")
    print(f"  Masks → {mask_dir}")

    # Stage 3: retrain with masks
    print(f"\n[Stage 3] Training low{thresh_tag} masked run...")
    train_mipnerf.remote(
        kitti_seq_dir, lambda_depth, max_num_iterations,
        mask_dir, ref_threshold, "low", mask_label, depth_loss_type, depth_sup_type, 10,
    )

    print("\n========== Mip-NeRF new-sequence experiments complete ==========")
    print("Evaluate with:")
    print(f"\n  # RGB-only")
    print(f"  modal run modal_train_mipnerf.py::run_eval \\")
    print(f"    --kitti-seq-dir '{kitti_seq_dir}' --lambda-depth 0.0")
    print(f"\n  # Global depth (nomask)")
    print(f"  modal run modal_train_mipnerf.py::run_eval \\")
    print(f"    --kitti-seq-dir '{kitti_seq_dir}' --lambda-depth {lambda_depth}")
    print(f"\n  # Low-error mask τ={ref_threshold}")
    print(f"  modal run modal_train_mipnerf.py::run_eval \\")
    print(f"    --kitti-seq-dir '{kitti_seq_dir}' --lambda-depth {lambda_depth} --masked")


@app.local_entrypoint()
def run_eval(
    kitti_seq_dir: str = "KITTISeq05_2011_09_30_drive_0018_sync_llffdtu_s400_e725_densegt",
    lambda_depth: float = 0.0,
    photo_mask_threshold: float = 0.18,
    masked: bool = False,
    depth_sup_type: str = "da2",
    max_num_iterations: int = 50000,
):
    import json

    slug = _derive_slug(kitti_seq_dir)
    if masked:
        thresh_tag = f"{photo_mask_threshold:.2f}".replace(".", "")
        mask_label = f"low{thresh_tag}"
    else:
        mask_label = "nomask"

    exp_name = _exp_name(
        slug, lambda_depth, max_num_iterations,
        mask_label=mask_label, depth_sup_type=depth_sup_type,
    )
    print(f"Evaluating: {exp_name}")
    metrics = eval_mipnerf.remote(kitti_seq_dir, exp_name, lambda_depth, max_num_iterations, depth_sup_type)
    print("\n========== Eval Results ==========")
    print(json.dumps(metrics, indent=2))
    print("==")
