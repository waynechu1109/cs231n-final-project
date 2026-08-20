"""
Modal training script for DN-Splatter on KITTI sparse-every-2 data.
Rebuttal comparison against splatfacto-da2 + photometric masking (ours).

DN-Splatter: Turkulainen et al. 2024 — nerfstudio==1.1.3, gsplat==1.0.0.
Runs in a completely separate image from our custom nerfstudio fork.

Prerequisites — upload depths_gt once:
  modal volume put kitti-nerf-data \\
    data/kitti/kitti_select_static_5seq_sparse_every2/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/depths_gt \\
    kitti/kitti_select_static_5seq_sparse_every2/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/depths_gt

Train (default: LogL1 loss, depth_lambda=0.2, 50k iters):
  modal run modal_train_dn_splatter.py::main

Evaluate:
  modal run modal_train_dn_splatter.py::run_eval

Download results:
  modal volume get nerf-outputs dn_splatter_kitti_seq02_0034 ./local_outputs/dn_splatter
"""

from __future__ import annotations

import re
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Shared volumes (same as splatfacto runs)
# ---------------------------------------------------------------------------
data_vol = modal.Volume.from_name("kitti-nerf-data", create_if_missing=True)
out_vol = modal.Volume.from_name("nerf-outputs", create_if_missing=True)

DATA_MOUNT = "/vol/data"
OUT_MOUNT = "/vol/outputs"

SCRIPTS_LOCAL = Path(__file__).parent / "scripts"

# ---------------------------------------------------------------------------
# Image — fresh install, completely separate from our nerfstudio fork.
# pip install dn-splatter pulls nerfstudio==1.1.3 and gsplat==1.0.0.
# gsplat 1.0.0 may need source compilation (CUDA devel headers present).
# Layer 1 (torch) matches our existing image → Modal reuses the cache.
# ---------------------------------------------------------------------------
_base = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git", "wget", "build-essential", "ninja-build", "cmake",
        "clang",               # CUDA base image sets CXX=clang++; PyMCubes/fpsample/pyliblzfse need it
        "python3-setuptools",  # provides pkg_resources for the cp workaround below
        "python3.10-dev", "libgl1", "ffmpeg", "libglib2.0-0",
    )
    .run_commands(
        "cp -r /usr/lib/python3/dist-packages/pkg_resources "
        "/usr/local/lib/python3.10/site-packages/",
    )
    .pip_install(
        "torch==2.1.2+cu118",
        "torchvision==0.16.2+cu118",
        "numpy<2.0.0",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu118",
    )
)

# Layer 2: DN-Splatter + all its deps (nerfstudio==1.1.3, gsplat==1.0.0).
# gsplat 1.0.0 compilation targets A10G (sm_86) and A100 (sm_80).
# dn-splatter was never published to PyPI → install from GitHub.
# Strategy: install heavy runtime deps first (so pip can resolve them), then
# install dn-splatter --no-deps so we don't re-resolve the whole tree.
#
# open3d is needed: normal_nerfstudio.py imports it at module level.
# vdbfusion is listed as a dep but only used in optional mesh-extraction paths;
# its wheel is on PyPI, so let pip grab it normally.
# omnidata-tools / geffnet / pymeshlab are optional (normal estimation / meshing
# only) and not needed for our depth-only run — skip them to keep image small.
_with_dn_splatter = _base.run_commands(
    # Step 1 — build gsplat 1.0.0 from source (needs CUDA devel headers).
    "TORCH_CUDA_ARCH_LIST='8.0 8.6' MAX_JOBS=4 "
    "pip install --no-cache-dir 'gsplat==1.0.0'",
    # Step 2 — nerfstudio 1.1.3 + all packages imported at module-load time by
    # dn_splatter/__init__.py and its transitive import chain.
    # - geffnet:         dn_splatter/scripts/dsine/submodules.py (DSINE predictor)
    # - omnidata-tools:  dn_splatter/scripts/normals_from_pretrain.py (DPTDepthModel)
    # - PyMCubes==0.1.2 INTENTIONALLY OMITTED: its C API (PyArray_DOUBLE) fails
    #   to compile on clang++ with NumPy ≥ 1.20; we don't need mesh extraction.
    "pip install --no-cache-dir "
    "'nerfstudio==1.1.3' "
    "'open3d' "
    "'natsort' "
    "'vdbfusion' "
    "'rerun-sdk' "
    "'pytorch-lightning' "
    "'geffnet' "
    "'omnidata-tools' "
    "'numpy<2.0.0'",
    # Step 3 — dn-splatter code from GitHub, skip dep resolution (already satisfied
    # above). PyMCubes is skipped intentionally; training does not need it.
    "pip install --no-cache-dir --no-deps "
    "'git+https://github.com/maturk/dn-splatter.git'",
)

# Layer 3: our pure-Python data-prep scripts (no nerfstudio dependency).
image = _with_dn_splatter.add_local_dir(SCRIPTS_LOCAL, "/opt/project_scripts")

app = modal.App("dn-splatter-kitti")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SEQ_DEFAULT = "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt"


def _slug(kitti_seq_dir: str) -> str:
    m = re.search(r"KITTISeq(\d+)_.*drive_(\d+)_sync", Path(kitti_seq_dir).name)
    if not m:
        raise ValueError(f"Cannot derive slug from: {kitti_seq_dir}")
    return f"kitti_seq{int(m.group(1)):02d}_{m.group(2)}"


def _exp_name(slug: str, lambda_depth: float, loss_type: str, iters: int) -> str:
    return (
        f"dn_splatter_{slug}_sparse_every2_gt"
        f"_lambda{lambda_depth}_{loss_type.lower()}_{iters}"
    )


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 10,
    volumes={DATA_MOUNT: data_vol, OUT_MOUNT: out_vol},
    env={"TORCHDYNAMO_DISABLE": "1", "MAX_JOBS": "1"},
)
def train(
    kitti_seq_dir: str = _SEQ_DEFAULT,
    lambda_depth: float = 0.2,
    depth_loss_type: str = "LogL1",
    max_num_iterations: int = 50000,
):
    """Build GT-depth dataset (if needed) then run ns-train dn-splatter."""
    import os
    import subprocess

    slug = _slug(kitti_seq_dir)
    sparse_root = f"{DATA_MOUNT}/kitti/kitti_select_static_5seq_sparse_every2"
    seq_dir = f"{sparse_root}/{kitti_seq_dir}"
    depth_dir = f"{seq_dir}/depths_gt"
    nerfstudio_src = f"{DATA_MOUNT}/nerfstudio/{slug}_sparse_every2"
    data_dir = f"{DATA_MOUNT}/nerfstudio/{slug}_sparse_every2_gt"
    exp_name = _exp_name(slug, lambda_depth, depth_loss_type, max_num_iterations)

    for path, label in [
        (seq_dir, "KITTI sequence dir"),
        (depth_dir, "depths_gt dir — upload it first (see docstring)"),
        (f"{nerfstudio_src}/transforms.json", "sparse nerfstudio transforms.json"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {label}:\n  {path}")

    # Build the GT nerfstudio dataset (symlinks depths_gt, writes transforms.json
    # with per-frame depth_file_path entries) — idempotent.
    if not os.path.exists(f"{data_dir}/transforms.json"):
        print(f"Building GT depth dataset → {data_dir}")
        subprocess.run(
            [
                "python", "/opt/project_scripts/data_prep/make_nerfstudio_kitti_depth.py",
                "--src", nerfstudio_src,
                "--dst", data_dir,
                "--depth-dir", depth_dir,
                "--depth-sup-type", "gt",
                "--overwrite",
            ],
            check=True,
        )
        data_vol.commit()
    else:
        print(f"GT depth dataset already exists: {data_dir}")

    # KITTI depth maps are uint16 PNG, depth_meters = pixel_value / 256.
    # nerfstudio-data dataparser default depth_unit_scale_factor is 1e-3 (mm→m);
    # we override to 1/256 = 0.00390625 via --dataparser flag.
    #
    # Normal-related flags all disabled: we have no GT normals and the
    # NormalNerfstudio dataparser crashes if load_pcd_normals=True without a
    # point cloud, or if load_normals=True with an empty normal_filenames list.
    cmd = [
        "ns-train", "dn-splatter",
        "--experiment-name", exp_name,
        "--output-dir", OUT_MOUNT,
        "--max-num-iterations", str(max_num_iterations),
        "--vis", "tensorboard",
        # Depth loss
        "--pipeline.model.use-depth-loss", "True",
        "--pipeline.model.depth-lambda", str(lambda_depth),
        "--pipeline.model.depth-loss-type", depth_loss_type,
        # Disable all normal-related losses (no GT normals for KITTI)
        "--pipeline.model.use-normal-loss", "False",
        "--pipeline.model.use-normal-tv-loss", "False",
        "--pipeline.model.use-normal-cosine-loss", "False",
        "--pipeline.model.predict-normals", "False",
        # Dataparser: normal-nerfstudio (DN-Splatter's extended nerfstudio parser)
        "normal-nerfstudio",
        "--data", data_dir,
        # KITTI depth scale: uint16 / 256 = meters
        "--depth-unit-scale-factor", "0.00390625",
        # No normals_from_pretrain/ dir → disable to avoid crash on empty list
        "--load-normals", "False",
        # No COLMAP PLY → disable pcd-normals to avoid KeyError on points3D_xyz
        "--load-pcd-normals", "False",
        # Skip PLY/COLMAP loading (dataset has no sparse/ dir); uses random init
        # Field is load_3D_points (capital D) → tyro 0.6.x preserves case → --load-3D-points
        "--load-3D-points", "False",
    ]

    print("=" * 60)
    print(f"Seq dir:   {seq_dir}")
    print(f"Data dir:  {data_dir}")
    print(f"Exp name:  {exp_name}")
    print(f"Command:\n  " + " \\\n  ".join(cmd))
    print("=" * 60)

    subprocess.run(cmd, check=True)
    out_vol.commit()
    print(f"\nOutputs saved to Modal volume 'nerf-outputs' under: {exp_name}/")


# ---------------------------------------------------------------------------
# eval
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
    import subprocess

    pattern = f"{OUT_MOUNT}/{exp_name}/dn-splatter/*/config.yml"
    configs = sorted(glob.glob(pattern))
    if not configs:
        raise FileNotFoundError(
            f"No config.yml found matching:\n  {pattern}\n"
            "Run training first: modal run modal_train_dn_splatter.py"
        )
    config_path = configs[-1]
    output_path = config_path.replace("config.yml", "eval_output.json")
    print(f"Config:  {config_path}")
    print(f"Output:  {output_path}")

    subprocess.run(
        ["ns-eval", "--load-config", config_path, "--output-path", output_path],
        check=True,
    )
    out_vol.commit()

    with open(output_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    kitti_seq_dir: str = _SEQ_DEFAULT,
    lambda_depth: float = 0.2,
    depth_loss_type: str = "LogL1",
    max_num_iterations: int = 50000,
):
    train.remote(
        kitti_seq_dir=kitti_seq_dir,
        lambda_depth=lambda_depth,
        depth_loss_type=depth_loss_type,
        max_num_iterations=max_num_iterations,
    )


@app.local_entrypoint()
def run_eval(
    kitti_seq_dir: str = _SEQ_DEFAULT,
    lambda_depth: float = 0.2,
    depth_loss_type: str = "LogL1",
    max_num_iterations: int = 50000,
):
    import json

    slug = _slug(kitti_seq_dir)
    exp_name = _exp_name(slug, lambda_depth, depth_loss_type, max_num_iterations)
    print(f"Evaluating: {exp_name}")
    metrics = eval_run.remote(exp_name=exp_name)
    print("\n========== DN-Splatter Eval Results ==========")
    print(json.dumps(metrics, indent=2))
    print("==============================================")
